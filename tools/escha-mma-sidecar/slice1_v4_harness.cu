#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

// Reuse the frozen shared-B control, MMA primitives, parser, comparison, and
// resource helpers. Earlier harnesses remain byte-for-byte untouched.
#define main slice1_v1_embedded_main
#include "slice1_harness.cu"
#undef main

namespace cg = cooperative_groups;

constexpr int V4_MAX_DEPTH = 4;
constexpr int V4_NSUPER4 = NSTAGE / 4;
constexpr size_t V4_B_BYTES = 2 * V4_MAX_DEPTH * BN * 16 * sizeof(half);
constexpr size_t V4_A_BYTES = 8 * 32 * 16 * sizeof(half);
constexpr size_t V4_PAYLOAD_BYTES = 2 * V4_MAX_DEPTH * 2 * 4 * 16 * sizeof(uint32_t);
constexpr size_t V4_SMEM_BYTES = V4_B_BYTES + V4_A_BYTES + V4_PAYLOAD_BYTES;
static_assert(V4_B_BYTES == 32768, "V4 double-banked BK64 B ring must be 32 KiB");
static_assert(V4_A_BYTES == 8192, "V4 warp-private A stages must be 8 KiB");
static_assert(V4_PAYLOAD_BYTES == 4096, "V4 double-banked payload ring must be 4 KiB");
static_assert(V4_SMEM_BYTES == 45056, "V4 must use the official 45,056-byte class");

static __device__ __forceinline__ const uint32_t *v4_source_stage(
        const uint32_t *__restrict__ overlay, int cta, int stage, int band) {
    // V3 fixed payload order: [cta][BK64 super][band][K16][tile][word].
    const int super4 = stage >> 2;
    const int stage4 = stage & 3;
    return overlay + (((cta * V4_NSUPER4 + super4) * 2 + band) * 4 + stage4) * 64;
}

template <int D>
static __device__ __forceinline__ void v4_load_payload_super(
        const uint32_t *__restrict__ overlay,
        uint32_t *__restrict__ s_payload,
        int cta, int base_stage, int bank, int band, int band_tid) {
    static_assert(D == 2 || D == 4, "V4 supports only 2x and 4x rings");
    if (band_tid < 64) {
#pragma unroll 1
        for (int s = 0; s < D; ++s) {
            const uint32_t *src = v4_source_stage(overlay, cta, base_stage + s, band);
            const int dst_stage = bank * V4_MAX_DEPTH + s;
            s_payload[((dst_stage * 2 + band) * 64) + band_tid] = src[band_tid];
        }
    }
}

template <int D>
static __device__ __forceinline__ void v4_decode_super(
        const uint32_t *__restrict__ s_payload,
        half *__restrict__ s_b,
        int bank, int band, int band_tid,
        uint32_t map0, uint32_t map1) {
    const int r = band_tid & 15;
    const int ccl0 = band_tid >> 4;
    const int word0 = map0 & 15;
    const int previous0 = (map0 >> 4) & 15;
    const int shift0 = (map0 >> 8) & 31;
    const int word1 = map1 & 15;
    const int previous1 = (map1 >> 4) & 15;
    const int shift1 = (map1 >> 8) & 31;
#pragma unroll 1
    for (int s = 0; s < D; ++s) {
        const int ring_stage = bank * V4_MAX_DEPTH + s;
        const uint32_t *payload = s_payload + ((ring_stage * 2 + band) * 64);
        half *dst = s_b + ring_stage * BN * 16 + band * 64 * 16;
#pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            const uint32_t *pay = payload + tile * 16;
            const uint32_t ix0 = __funnelshift_r(pay[word0], pay[previous0], shift0) & 0xffffu;
            const uint32_t ix1 = __funnelshift_r(pay[word1], pay[previous1], shift1) & 0xffffu;
            dst[(ccl0 + tile * 16) * 16 + r] = codebook_h(ix0);
            dst[(ccl0 + 8 + tile * 16) * 16 + r] = codebook_h(ix1);
        }
    }
}

template <bool FP16_ACC>
static __device__ __forceinline__ void v4_consume_stage(
        const half *__restrict__ u,
        half *__restrict__ s_a,
        const half *__restrict__ s_b,
        int row0, int ti, int ring_stage, int warp, int band, int row_group, int lane,
        Acc16 (&ah)[2][8], Acc32 (&af)[2][8]) {
    half *warp_a = s_a + warp * 32 * 16;

    // The only writer and reader of this A stage is this warp. The prior async
    // copy is complete before LDSM; after both fragments are registers, the
    // same bytes can receive K16+1 while current HMMA issues.
    __pipeline_wait_prior(0);
    __syncwarp();
    AFrag a0, a1;
    const half2 *a_base = reinterpret_cast<const half2 *>(warp_a);
    load_a(a0, a_base + (lane & 15) * 8 + (lane >> 4) * 4);
    load_a(a1, a_base + 16 * 8 + (lane & 15) * 8 + (lane >> 4) * 4);

    if (ti + 1 < NSTAGE) {
        const half *src = u + (row0 + row_group * 32 + lane) * IC + (ti + 1) * 16;
        half *dst = warp_a + lane * 16;
        __pipeline_memcpy_async(dst, src, 16);
        __pipeline_memcpy_async(dst + 8, src + 8, 16);
        __pipeline_commit();
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
        BFrag b;
        const half2 *base = reinterpret_cast<const half2 *>(s_b) +
                            (ring_stage * BN + band * 64 + j * 8) * 8;
        const half2 *lane_base = base + (lane & 7) * 8 + ((lane >> 3) * 4) % 8;
        load_b(b, lane_base);
        if constexpr (FP16_ACC) {
            mma(ah[0][j], a0, b);
            mma(ah[1][j], a1, b);
        } else {
            mma(af[0][j], a0, b);
            mma(af[1][j], a1, b);
        }
    }
}

template <bool FP16_ACC, int D>
static __device__ __forceinline__ void v4_kernel_body(
        const uint32_t *__restrict__ overlay,
        const half *__restrict__ u,
        float *__restrict__ out) {
    extern __shared__ __align__(16) unsigned char v4_raw[];
    half *s_b = reinterpret_cast<half *>(v4_raw);
    half *s_a = reinterpret_cast<half *>(v4_raw + V4_B_BYTES);
    uint32_t *s_payload = reinterpret_cast<uint32_t *>(v4_raw + V4_B_BYTES + V4_A_BYTES);

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int band = warp >> 2;
    const int row_group = warp & 3;
    const int band_tid = row_group * 32 + lane;
    const int row0 = blockIdx.x * BM;
    const int cta = blockIdx.y;
    const int oc0 = cta * BN;

    // Two descriptor-free mappings cover ccl 0..7 and 8..15. They are loop
    // invariant and produce exactly eight values/thread/K16.
    const int dr = band_tid & 15;
    const int ccl0 = band_tid >> 4;
    int dsp0 = (30 - 2 * (dep_pi(dr) + 32 * ccl0 + 4 * (ccl0 >> 3))) % 512;
    int dsp1 = (30 - 2 * (dep_pi(dr) + 32 * (ccl0 + 8) + 4 * ((ccl0 + 8) >> 3))) % 512;
    if (dsp0 < 0) dsp0 += 512;
    if (dsp1 < 0) dsp1 += 512;
    const int group0 = dsp0 >> 5;
    const int group1 = dsp1 >> 5;
    const int word0 = group0 ? 16 - group0 : 0;
    const int word1 = group1 ? 16 - group1 : 0;
    const uint32_t map0 = word0 | ((word0 ? word0 - 1 : 15) << 4) | ((dsp0 & 31) << 8);
    const uint32_t map1 = word1 | ((word1 ? word1 - 1 : 15) << 4) | ((dsp1 & 31) << 8);

    // Prime the warp-private A stage.
    {
        half *dst = s_a + warp * 32 * 16 + lane * 16;
        const half *src = u + (row0 + row_group * 32 + lane) * IC;
        __pipeline_memcpy_async(dst, src, 16);
        __pipeline_memcpy_async(dst + 8, src + 8, 16);
        __pipeline_commit();
    }

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<128> band_group = cg::tiled_partition<128>(block);

    // Prime bank 0 with the first complete superstage. One rendezvous publishes
    // payload, one publishes decoded B. This prologue is outside steady state.
    v4_load_payload_super<D>(overlay, s_payload, cta, 0, 0, band, band_tid);
    band_group.sync();
    v4_decode_super<D>(s_payload, s_b, 0, band, band_tid, map0, map1);
    band_group.sync();

    Acc16 ah[2][8] = {};
    Acc32 af[2][8] = {};

#pragma unroll 1
    for (int super_base = 0; super_base < NSTAGE; super_base += D) {
        const int super_index = super_base / D;
        const int current_bank = super_index & 1;
        const int next_bank = current_bank ^ 1;
        const int future_base = super_base + D;

        // Opposite phase order is the overlap mechanism. The branch is uniform
        // per four-warp band, so the scheduler can issue band-0 integer decode
        // while band 1 feeds LDSM/HMMA, then issue the reciprocal phase.
        if (band == 0) {
            if (future_base < NSTAGE) {
                v4_load_payload_super<D>(overlay, s_payload, cta, future_base,
                                         next_bank, band, band_tid);
                band_group.sync();
                v4_decode_super<D>(s_payload, s_b, next_bank, band, band_tid, map0, map1);
            }
#pragma unroll
            for (int s = 0; s < D; ++s) {
                v4_consume_stage<FP16_ACC>(u, s_a, s_b, row0, super_base + s,
                    current_bank * V4_MAX_DEPTH + s, warp, band, row_group, lane, ah, af);
            }
        } else {
#pragma unroll
            for (int s = 0; s < D; ++s) {
                v4_consume_stage<FP16_ACC>(u, s_a, s_b, row0, super_base + s,
                    current_bank * V4_MAX_DEPTH + s, warp, band, row_group, lane, ah, af);
            }
            if (future_base < NSTAGE) {
                v4_load_payload_super<D>(overlay, s_payload, cta, future_base,
                                         next_bank, band, band_tid);
                band_group.sync();
                v4_decode_super<D>(s_payload, s_b, next_bank, band, band_tid, map0, map1);
            }
        }
        // Publishes future B and prevents either bank from being reused before
        // all four warps in its owning band finish current LDSM.
        band_group.sync();
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if constexpr (FP16_ACC) {
#pragma unroll
                for (int l = 0; l < 2; ++l) {
                    half2 v;
                    memcpy(&v, &ah[i][j].x[l], sizeof(v));
                    const int m0 = row_group * 32 + i * 16 + c_i(2 * l, lane);
                    const int n0 = band * 64 + j * 8 + c_j(2 * l, lane);
                    const int m1 = row_group * 32 + i * 16 + c_i(2 * l + 1, lane);
                    const int n1 = band * 64 + j * 8 + c_j(2 * l + 1, lane);
                    out[(row0 + m0) * OC + oc0 + n0] = __half2float(__low2half(v));
                    out[(row0 + m1) * OC + oc0 + n1] = __half2float(__high2half(v));
                }
            } else {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int m = row_group * 32 + i * 16 + c_i(l, lane);
                    const int n = band * 64 + j * 8 + c_j(l, lane);
                    out[(row0 + m) * OC + oc0 + n] = af[i][j].x[l];
                }
            }
        }
    }
}

extern "C" __global__ __maxnreg__(104)
void slice1_v4_d2_fp16(const uint32_t *overlay, const half *u, float *out) {
    v4_kernel_body<true, 2>(overlay, u, out);
}
extern "C" __global__ __launch_bounds__(256, 2)
void slice1_v4_d2_fp32(const uint32_t *overlay, const half *u, float *out) {
    v4_kernel_body<false, 2>(overlay, u, out);
}
extern "C" __global__ __maxnreg__(104)
void slice1_v4_d4_fp16(const uint32_t *overlay, const half *u, float *out) {
    v4_kernel_body<true, 4>(overlay, u, out);
}
extern "C" __global__ __launch_bounds__(256, 2)
void slice1_v4_d4_fp32(const uint32_t *overlay, const half *u, float *out) {
    v4_kernel_body<false, 4>(overlay, u, out);
}

template <typename Kernel>
static float time_v4(Kernel kernel, const uint32_t *overlay,
                     const half *u, float *out, int reps) {
    dim3 block_dim(32, 8), grid_dim(M / BM, OC / BN);
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "v4 event create start");
    cuda_check(cudaEventCreate(&stop), "v4 event create stop");
    for (int i = 0; i < 2; ++i) kernel<<<grid_dim, block_dim, V4_SMEM_BYTES>>>(overlay, u, out);
    cuda_check(cudaDeviceSynchronize(), "v4 warmup");
    cuda_check(cudaEventRecord(start), "v4 event start");
    for (int i = 0; i < reps; ++i) kernel<<<grid_dim, block_dim, V4_SMEM_BYTES>>>(overlay, u, out);
    cuda_check(cudaEventRecord(stop), "v4 event stop");
    cuda_check(cudaEventSynchronize(stop), "v4 event sync");
    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "v4 elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / reps;
}

template <typename Candidate>
static void collect_pairs(Candidate candidate,
                          const uint32_t *code, const uint32_t *overlay,
                          const half *u, float *control_out, float *candidate_out,
                          size_t control_smem, int reps,
                          std::vector<float> &control, std::vector<float> &result) {
    for (int pair = 0; pair < 5; ++pair) {
        if (pair & 1) {
            result.push_back(time_v4(candidate, overlay, u, candidate_out, reps));
            control.push_back(time_kernel(slice1_control_fp16, code, overlay, u,
                                          control_out, control_smem, reps));
        } else {
            control.push_back(time_kernel(slice1_control_fp16, code, overlay, u,
                                          control_out, control_smem, reps));
            result.push_back(time_v4(candidate, overlay, u, candidate_out, reps));
        }
    }
}

static void print_samples(const char *name, const std::vector<float> &samples) {
    std::printf("\"%s\":[", name);
    for (size_t i = 0; i < samples.size(); ++i) {
        std::printf("%s%.6f", i ? "," : "", samples[i]);
    }
    std::printf("]");
}

int main(int argc, char **argv) {
    std::string source, overlay_path;
    uint64_t code_offset = 0;
    int reps = 5;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--source") && ++i < argc) source = argv[i];
        else if (!std::strcmp(argv[i], "--overlay") && ++i < argc) overlay_path = argv[i];
        else if (!std::strcmp(argv[i], "--code-offset") && ++i < argc) code_offset = std::strtoull(argv[i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--reps") && ++i < argc) reps = std::atoi(argv[i]);
        else { std::fprintf(stderr, "bad argument\n"); return 2; }
    }
    if (source.empty() || overlay_path.empty() || !code_offset) return 2;

    auto prefix_bytes = read_range(overlay_path, 0, sizeof(HeaderPrefix));
    HeaderPrefix h{};
    std::memcpy(&h, prefix_bytes.data(), sizeof(h));
    if (std::memcmp(h.magic, "ESCHA-MMA-V3", 12) || h.value[0] != 3 ||
        h.value[2] != 2 || h.value[3] != IC || h.value[4] != OC ||
        h.value[6] != 4 || h.value[14] != 2048) {
        throw std::runtime_error("bad V3 fixed-mapping overlay header");
    }
    auto payload = read_range(overlay_path, h.payload_offset, static_cast<size_t>(h.payload_bytes));
    auto canonical = read_range(source, code_offset,
                                static_cast<size_t>(NSTAGE) * NCT * 16 * sizeof(uint32_t));

    std::vector<half> host_u(static_cast<size_t>(M) * IC);
    for (size_t i = 0; i < host_u.size(); ++i) {
        host_u[i] = __float2half(std::sin(float(i % 8191) * 0.001f));
    }

    uint32_t *d_code = nullptr, *d_overlay = nullptr;
    half *d_u = nullptr;
    float *d_control = nullptr, *d_d2 = nullptr, *d_d4 = nullptr;
    const size_t out_elems = static_cast<size_t>(M) * OC;
    cuda_check(cudaMalloc(&d_code, canonical.size()), "malloc code");
    cuda_check(cudaMalloc(&d_overlay, payload.size()), "malloc overlay");
    cuda_check(cudaMalloc(&d_u, host_u.size() * sizeof(half)), "malloc u");
    cuda_check(cudaMalloc(&d_control, out_elems * sizeof(float)), "malloc control out");
    cuda_check(cudaMalloc(&d_d2, out_elems * sizeof(float)), "malloc d2 out");
    cuda_check(cudaMalloc(&d_d4, out_elems * sizeof(float)), "malloc d4 out");
    cuda_check(cudaMemcpy(d_code, canonical.data(), canonical.size(), cudaMemcpyHostToDevice), "copy code");
    cuda_check(cudaMemcpy(d_overlay, payload.data(), payload.size(), cudaMemcpyHostToDevice), "copy overlay");
    cuda_check(cudaMemcpy(d_u, host_u.data(), host_u.size() * sizeof(half),
                          cudaMemcpyHostToDevice), "copy u");

    const size_t control_smem = 8 * 24 * sizeof(uint2) + 2 * BM * 16 * sizeof(half) + BN * 16 * sizeof(half);
    dim3 block_dim(32, 8), grid_dim(M / BM, OC / BN);
    slice1_control_fp16<<<grid_dim, block_dim, control_smem>>>(d_code, d_overlay, d_u, d_control);
    slice1_v4_d2_fp16<<<grid_dim, block_dim, V4_SMEM_BYTES>>>(d_overlay, d_u, d_d2);
    slice1_v4_d4_fp16<<<grid_dim, block_dim, V4_SMEM_BYTES>>>(d_overlay, d_u, d_d4);
    cuda_check(cudaDeviceSynchronize(), "correctness kernels");

    unsigned long long *d_bad2 = nullptr, *d_bad4 = nullptr, bad2 = 0, bad4 = 0;
    cuda_check(cudaMalloc(&d_bad2, sizeof(bad2)), "malloc d2 mismatch");
    cuda_check(cudaMalloc(&d_bad4, sizeof(bad4)), "malloc d4 mismatch");
    cuda_check(cudaMemset(d_bad2, 0, sizeof(bad2)), "clear d2 mismatch");
    cuda_check(cudaMemset(d_bad4, 0, sizeof(bad4)), "clear d4 mismatch");
    compare_bits<<<static_cast<unsigned>((out_elems + 255) / 256), 256>>>(
        reinterpret_cast<uint32_t *>(d_control), reinterpret_cast<uint32_t *>(d_d2), out_elems, d_bad2);
    compare_bits<<<static_cast<unsigned>((out_elems + 255) / 256), 256>>>(
        reinterpret_cast<uint32_t *>(d_control), reinterpret_cast<uint32_t *>(d_d4), out_elems, d_bad4);
    cuda_check(cudaMemcpy(&bad2, d_bad2, sizeof(bad2), cudaMemcpyDeviceToHost), "copy d2 mismatch");
    cuda_check(cudaMemcpy(&bad4, d_bad4, sizeof(bad4), cudaMemcpyDeviceToHost), "copy d4 mismatch");

    std::vector<float> control2, samples2, control4, samples4;
    collect_pairs(slice1_v4_d2_fp16, d_code, d_overlay, d_u, d_control, d_d2,
                  control_smem, reps, control2, samples2);
    collect_pairs(slice1_v4_d4_fp16, d_code, d_overlay, d_u, d_control, d_d4,
                  control_smem, reps, control4, samples4);
    auto median = [](std::vector<float> x) { std::sort(x.begin(), x.end()); return x[x.size() / 2]; };
    const float control2_ms = median(control2), d2_ms = median(samples2);
    const float control4_ms = median(control4), d4_ms = median(samples4);
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, 0), "device props");

    std::printf("{\"device\":\"%s\",\"compute_capability\":\"%d.%d\",\"m\":%d,\"reps_per_pair\":%d,",
                prop.name, prop.major, prop.minor, M, reps);
    std::printf("\"d2\":{\"control_ms\":%.6f,\"candidate_ms\":%.6f,\"bit_mismatches\":%llu,",
                control2_ms, d2_ms, bad2);
    print_samples("control_samples_ms", control2); std::printf(",");
    print_samples("candidate_samples_ms", samples2); std::printf("},");
    std::printf("\"d4\":{\"control_ms\":%.6f,\"candidate_ms\":%.6f,\"bit_mismatches\":%llu,",
                control4_ms, d4_ms, bad4);
    print_samples("control_samples_ms", control4); std::printf(",");
    print_samples("candidate_samples_ms", samples4); std::printf("},\"resources\":{");
    resource_json("control_fp16", slice1_control_fp16, control_smem); std::printf(",");
    resource_json("control_fp32", slice1_control_fp32, control_smem); std::printf(",");
    resource_json("d2_fp16", slice1_v4_d2_fp16, V4_SMEM_BYTES); std::printf(",");
    resource_json("d2_fp32", slice1_v4_d2_fp32, V4_SMEM_BYTES); std::printf(",");
    resource_json("d4_fp16", slice1_v4_d4_fp16, V4_SMEM_BYTES); std::printf(",");
    resource_json("d4_fp32", slice1_v4_d4_fp32, V4_SMEM_BYTES);
    std::printf("}}\n");

    cudaFree(d_bad4); cudaFree(d_bad2); cudaFree(d_d4); cudaFree(d_d2);
    cudaFree(d_control); cudaFree(d_u); cudaFree(d_overlay); cudaFree(d_code);
    return (bad2 == 0 && bad4 == 0) ? 0 : 3;
}
