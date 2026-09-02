#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

// Reuse the frozen V2 control symbol, MMA primitives, parser, comparison, and
// resource helpers without changing either earlier harness.
#define main slice1_v1_embedded_main
#include "slice1_harness.cu"
#undef main

constexpr int V3_STAGES_PER_SUPER = 4;
constexpr int V3_NSUPER = NSTAGE / V3_STAGES_PER_SUPER;
constexpr int V3_PAYLOAD_WORDS = 2 * V3_STAGES_PER_SUPER * 4 * 16;
constexpr size_t V3_PAYLOAD_SUPER_BYTES = V3_PAYLOAD_WORDS * sizeof(uint32_t);
constexpr int V3_PAYLOAD_SLOTS = 6;
constexpr int V3_A_SLOTS = 2 * V3_STAGES_PER_SUPER;
constexpr size_t V3_A_SLOT_BYTES = BM * 16 * sizeof(half);
constexpr size_t V3_A_BYTES = V3_A_SLOTS * V3_A_SLOT_BYTES;
constexpr size_t V3_PAYLOAD_BYTES = V3_PAYLOAD_SLOTS * V3_PAYLOAD_SUPER_BYTES;
constexpr size_t V3_SMEM_BYTES = V3_A_BYTES + V3_PAYLOAD_BYTES;
static_assert(V3_PAYLOAD_SUPER_BYTES == 2048, "one BK64 payload is 2 KiB");
static_assert(V3_SMEM_BYTES == 45056, "V3 must genuinely address the 45,056-byte class");

static __device__ __forceinline__ uint32_t v3_decode_window(
        const uint32_t current, const uint32_t previous, const int shift) {
    return __funnelshift_r(current, previous, shift) & 0xffffu;
}

static __device__ __forceinline__ uint32_t v3_decode_half2(
        const uint32_t current, const uint32_t previous,
        const int shift0, const int shift1) {
    const half2 value = __halves2half2(
        codebook_h(v3_decode_window(current, previous, shift0)),
        codebook_h(v3_decode_window(current, previous, shift1)));
    uint32_t packed;
    memcpy(&packed, &value, sizeof(packed));
    return packed;
}

template <bool FP16_ACC, bool DESCRIPTOR_ABLATION, int STAGE4>
static __device__ __forceinline__ void v3_consume_stage(
        const int super_stage,
        const int row0,
        const int band,
        const int row_group,
        const int lane,
        const int payload_slot,
        half *__restrict__ s_u,
        const uint32_t *__restrict__ s_payload,
        const half *__restrict__ u,
        Acc16 (&ah)[2][8],
        Acc32 (&af)[2][8]) {
    constexpr int A_SLOT = STAGE4;
    half *a_slot = s_u + (band * V3_STAGES_PER_SUPER + A_SLOT) * BM * 16;

    // The four warps of a band own disjoint 32-row A regions.  Each warp loads
    // both A fragments before overwriting only its own rows with next-BK64 A.
    // This puts future-A LDGSTS before the packed decode/HMMA body.
    AFrag a0, a1;
    const half2 *a_base = reinterpret_cast<const half2 *>(a_slot) + row_group * 32 * 8;
    load_a(a0, a_base + (lane & 15) * 8 + (lane >> 4) * 4);
    load_a(a1, a_base + 16 * 8 + (lane & 15) * 8 + (lane >> 4) * 4);

    const int future_stage = (super_stage + 1) * V3_STAGES_PER_SUPER + STAGE4;
    if (future_stage < NSTAGE) {
        const int owned_row = row_group * 32 + lane;
        half *dst = a_slot + owned_row * 16;
        const half *src = u + (row0 + owned_row) * IC + future_stage * 16;
        __pipeline_memcpy_async(dst, src, 16);
        __pipeline_memcpy_async(dst + 8, src + 8, 16);
        __pipeline_commit();
    }

    const uint32_t *stage_payload = s_payload +
        payload_slot * V3_PAYLOAD_WORDS +
        (band * V3_STAGES_PER_SUPER + STAGE4) * 4 * 16;
    const int fixed_word = lane >> 1;
    const int fixed_previous_word = (fixed_word + 15) & 15;
    const int fixed_lane_shift = 30 - ((lane & 1) << 4);

    // One register B fragment at a time.  K2 makes every lane's four windows
    // share one circular adjacent pair; fragment parity only subtracts eight
    // from the fixed lane shift.  There is no descriptor lookup or B smem.
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t *tile = stage_payload + (j >> 1) * 16;
        BFrag b;
        if constexpr (DESCRIPTOR_ABLATION) {
#pragma unroll
            for (int slot = 0; slot < 2; ++slot) {
                const uint32_t descriptor = g_access[j * 64 + slot * 32 + lane];
                const int word = descriptor & 15;
                const int previous_word = (descriptor >> 4) & 15;
                const int shift0 = (descriptor >> 8) & 31;
                const int shift1 = (descriptor >> 13) & 31;
                b.x[slot] = v3_decode_half2(
                    tile[word], tile[previous_word], shift0, shift1);
            }
        } else {
            const uint32_t current = tile[fixed_word];
            const uint32_t previous = tile[fixed_previous_word];
            const int shift = fixed_lane_shift - ((j & 1) << 3);
            b.x[0] = v3_decode_half2(current, previous, shift, shift - 2);
            b.x[1] = v3_decode_half2(current, previous, shift - 4, shift - 6);
        }
        if constexpr (FP16_ACC) {
            mma(ah[0][j], a0, b);
            mma(ah[1][j], a1, b);
        } else {
            mma(af[0][j], a0, b);
            mma(af[1][j], a1, b);
        }
    }
}

template <bool FP16_ACC, bool DESCRIPTOR_ABLATION = false>
static __device__ __forceinline__ void v3_kernel_body(
        const uint32_t *__restrict__ overlay,
        const half *__restrict__ u,
        float *__restrict__ out) {
    extern __shared__ __align__(16) unsigned char v3_raw[];
    half *s_u = reinterpret_cast<half *>(v3_raw);                         // [band][K16][128][16]
    uint32_t *s_payload = reinterpret_cast<uint32_t *>(v3_raw + V3_A_BYTES); // six BK64 slots

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int band = warp >> 2;
    const int row_group = warp & 3;
    const int row0 = blockIdx.x * BM;
    const int cta = blockIdx.y;
    const int oc0 = cta * BN;

    // Fill every byte of the eight-slot A arena: four K16 tiles duplicated for
    // the two independent 64-column bands.  The duplication permits each warp
    // to replace only rows it has already loaded, without producer warps.
#pragma unroll
    for (int stage4 = 0; stage4 < V3_STAGES_PER_SUPER; ++stage4) {
        const int owned_row = row_group * 32 + lane;
        half *dst = s_u + (band * V3_STAGES_PER_SUPER + stage4) * BM * 16 + owned_row * 16;
        const half *src = u + (row0 + owned_row) * IC + stage4 * 16;
        __pipeline_memcpy_async(dst, src, 16);
        __pipeline_memcpy_async(dst + 8, src + 8, 16);
    }

    // Five payload superstages are resident and the sixth slot is deliberately
    // empty.  During each body that free slot receives super+5, avoiding a
    // reuse race while all six 2-KiB slots are exercised over the loop.
    if (tid < V3_PAYLOAD_SUPER_BYTES / 16) {
#pragma unroll
        for (int preload = 0; preload < V3_PAYLOAD_SLOTS - 1; ++preload) {
            __pipeline_memcpy_async(
                reinterpret_cast<unsigned char *>(s_payload) +
                    preload * V3_PAYLOAD_SUPER_BYTES + tid * 16,
                reinterpret_cast<const unsigned char *>(overlay) +
                    (cta * V3_NSUPER + preload) * V3_PAYLOAD_SUPER_BYTES + tid * 16,
                16);
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    Acc16 ah[2][8] = {};
    Acc32 af[2][8] = {};

#pragma unroll 1
    for (int super_stage = 0; super_stage < V3_NSUPER; ++super_stage) {
        const int payload_slot = super_stage % V3_PAYLOAD_SLOTS;
        const int future_payload = super_stage + V3_PAYLOAD_SLOTS - 1;

        // The empty payload slot is filled before and concurrently with the
        // four-stage compute body.  No descriptor memory exists in V3.
        if (future_payload < V3_NSUPER && tid < V3_PAYLOAD_SUPER_BYTES / 16) {
            __pipeline_memcpy_async(
                reinterpret_cast<unsigned char *>(s_payload) +
                    (future_payload % V3_PAYLOAD_SLOTS) * V3_PAYLOAD_SUPER_BYTES + tid * 16,
                reinterpret_cast<const unsigned char *>(overlay) +
                    (cta * V3_NSUPER + future_payload) * V3_PAYLOAD_SUPER_BYTES + tid * 16,
                16);
            __pipeline_commit();
        }

        // Frozen BK64 superstage: four explicit K16 stages, 16 HMMA each.
        // Explicit calls plus the fully-unrolled j loop keep all accumulator
        // references compile-time fixed and prohibit EXP-10-style homing.
        v3_consume_stage<FP16_ACC, DESCRIPTOR_ABLATION, 0>(
            super_stage, row0, band, row_group, lane, payload_slot, s_u, s_payload, u, ah, af);
        v3_consume_stage<FP16_ACC, DESCRIPTOR_ABLATION, 1>(
            super_stage, row0, band, row_group, lane, payload_slot, s_u, s_payload, u, ah, af);
        v3_consume_stage<FP16_ACC, DESCRIPTOR_ABLATION, 2>(
            super_stage, row0, band, row_group, lane, payload_slot, s_u, s_payload, u, ah, af);
        v3_consume_stage<FP16_ACC, DESCRIPTOR_ABLATION, 3>(
            super_stage, row0, band, row_group, lane, payload_slot, s_u, s_payload, u, ah, af);

        if (super_stage + 1 < V3_NSUPER) {
            __pipeline_wait_prior(0);
            __syncthreads(); // the sole steady-state CTA rendezvous per BK64
        }
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

extern "C" __global__ __launch_bounds__(256, 2)
void slice1_v3_fp16(const uint32_t *overlay, const half *u, float *out) {
    v3_kernel_body<true>(overlay, u, out);
}

extern "C" __global__ __launch_bounds__(256, 2)
void slice1_v3_fp32(const uint32_t *overlay, const half *u, float *out) {
    v3_kernel_body<false>(overlay, u, out);
}

extern "C" __global__ __launch_bounds__(256, 2)
void slice1_v3_descriptor_fp16(const uint32_t *overlay, const half *u, float *out) {
    v3_kernel_body<true, true>(overlay, u, out);
}

extern "C" __global__ __launch_bounds__(256, 2)
void slice1_v3_descriptor_fp32(const uint32_t *overlay, const half *u, float *out) {
    v3_kernel_body<false, true>(overlay, u, out);
}

template <typename Kernel>
static float time_v3(Kernel kernel, const uint32_t *overlay,
                     const half *u, float *out, int reps) {
    dim3 block_dim(32, 8), grid_dim(M / BM, OC / BN);
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "v3 event create start");
    cuda_check(cudaEventCreate(&stop), "v3 event create stop");
    for (int i = 0; i < 2; ++i) {
        kernel<<<grid_dim, block_dim, V3_SMEM_BYTES>>>(overlay, u, out);
    }
    cuda_check(cudaDeviceSynchronize(), "v3 warmup");
    cuda_check(cudaEventRecord(start), "v3 event start");
    for (int i = 0; i < reps; ++i) {
        kernel<<<grid_dim, block_dim, V3_SMEM_BYTES>>>(overlay, u, out);
    }
    cuda_check(cudaEventRecord(stop), "v3 event stop");
    cuda_check(cudaEventSynchronize(stop), "v3 event sync");
    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "v3 elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / reps;
}

int main(int argc, char **argv) {
    std::string source, overlay_path;
    uint64_t code_offset = 0;
    int reps = 5;
    bool run_ablation = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--source") && ++i < argc) source = argv[i];
        else if (!std::strcmp(argv[i], "--overlay") && ++i < argc) overlay_path = argv[i];
        else if (!std::strcmp(argv[i], "--code-offset") && ++i < argc) code_offset = std::strtoull(argv[i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--reps") && ++i < argc) reps = std::atoi(argv[i]);
        else if (!std::strcmp(argv[i], "--run-ablation")) run_ablation = true;
        else { std::fprintf(stderr, "bad argument\n"); return 2; }
    }
    if (source.empty() || overlay_path.empty() || !code_offset) return 2;

    auto prefix_bytes = read_range(overlay_path, 0, sizeof(HeaderPrefix));
    HeaderPrefix h{};
    std::memcpy(&h, prefix_bytes.data(), sizeof(h));
    if (std::memcmp(h.magic, "ESCHA-MMA-V3", 12) || h.value[0] != 3 ||
        h.value[2] != 2 || h.value[3] != IC || h.value[4] != OC ||
        h.value[6] != V3_STAGES_PER_SUPER || h.value[14] != V3_PAYLOAD_SUPER_BYTES) {
        throw std::runtime_error("bad V3 overlay header");
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
    float *d_control = nullptr, *d_v3 = nullptr, *d_descriptor = nullptr;
    const size_t out_elems = static_cast<size_t>(M) * OC;
    cuda_check(cudaMalloc(&d_code, canonical.size()), "malloc code");
    cuda_check(cudaMalloc(&d_overlay, payload.size()), "malloc V3 overlay");
    cuda_check(cudaMalloc(&d_u, host_u.size() * sizeof(half)), "malloc u");
    cuda_check(cudaMalloc(&d_control, out_elems * sizeof(float)), "malloc control out");
    cuda_check(cudaMalloc(&d_v3, out_elems * sizeof(float)), "malloc v3 out");
    cuda_check(cudaMalloc(&d_descriptor, out_elems * sizeof(float)), "malloc descriptor ablation out");
    cuda_check(cudaMemcpy(d_code, canonical.data(), canonical.size(), cudaMemcpyHostToDevice), "copy code");
    cuda_check(cudaMemcpy(d_overlay, payload.data(), payload.size(), cudaMemcpyHostToDevice), "copy overlay");
    cuda_check(cudaMemcpy(d_u, host_u.data(), host_u.size() * sizeof(half), cudaMemcpyHostToDevice), "copy u");

    // The sole attribution ablation restores two runtime mapping reads for
    // each B fragment while retaining the exact V3 payload and BK64 schedule.
    std::vector<uint32_t> descriptor_access(DESC_COUNT);
    for (int j = 0; j < 8; ++j) {
        for (int slot = 0; slot < 2; ++slot) {
            for (int lane = 0; lane < 32; ++lane) {
                const int word = lane >> 1;
                const int previous_word = (word + 15) & 15;
                const int base_shift = 30 - ((lane & 1) << 4) - ((j & 1) << 3) - slot * 4;
                descriptor_access[j * 64 + slot * 32 + lane] =
                    word | (previous_word << 4) | (base_shift << 8) | ((base_shift - 2) << 13);
            }
        }
    }
    cuda_check(cudaMemcpyToSymbol(g_access, descriptor_access.data(),
                                  descriptor_access.size() * sizeof(uint32_t)),
               "copy attribution descriptors");

    const size_t control_smem = 8 * 24 * sizeof(uint2) + 2 * BM * 16 * sizeof(half) + BN * 16 * sizeof(half);
    dim3 block_dim(32, 8), grid_dim(M / BM, OC / BN);
    slice1_control_fp16<<<grid_dim, block_dim, control_smem>>>(d_code, d_overlay, d_u, d_control);
    slice1_v3_fp16<<<grid_dim, block_dim, V3_SMEM_BYTES>>>(d_overlay, d_u, d_v3);
    if (run_ablation) {
        slice1_v3_descriptor_fp16<<<grid_dim, block_dim, V3_SMEM_BYTES>>>(d_overlay, d_u, d_descriptor);
    }
    cuda_check(cudaDeviceSynchronize(), "correctness kernels");

    unsigned long long *d_bad = nullptr, bad = 0;
    cuda_check(cudaMalloc(&d_bad, sizeof(bad)), "malloc mismatch");
    cuda_check(cudaMemset(d_bad, 0, sizeof(bad)), "clear mismatch");
    compare_bits<<<static_cast<unsigned>((out_elems + 255) / 256), 256>>>(
        reinterpret_cast<uint32_t *>(d_control), reinterpret_cast<uint32_t *>(d_v3), out_elems, d_bad);
    cuda_check(cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost), "copy mismatch");

    unsigned long long *d_bad_descriptor = nullptr, bad_descriptor = 0;
    if (run_ablation) {
        cuda_check(cudaMalloc(&d_bad_descriptor, sizeof(bad_descriptor)), "malloc descriptor mismatch");
        cuda_check(cudaMemset(d_bad_descriptor, 0, sizeof(bad_descriptor)), "clear descriptor mismatch");
        compare_bits<<<static_cast<unsigned>((out_elems + 255) / 256), 256>>>(
            reinterpret_cast<uint32_t *>(d_control), reinterpret_cast<uint32_t *>(d_descriptor),
            out_elems, d_bad_descriptor);
        cuda_check(cudaMemcpy(&bad_descriptor, d_bad_descriptor, sizeof(bad_descriptor),
                              cudaMemcpyDeviceToHost), "copy descriptor mismatch");
    }

    std::vector<float> control_samples, v3_samples;
    for (int pair = 0; pair < 5; ++pair) {
        if (pair & 1) {
            v3_samples.push_back(time_v3(slice1_v3_fp16, d_overlay, d_u, d_v3, reps));
            control_samples.push_back(time_kernel(slice1_control_fp16, d_code, d_overlay, d_u,
                                                  d_control, control_smem, reps));
        } else {
            control_samples.push_back(time_kernel(slice1_control_fp16, d_code, d_overlay, d_u,
                                                  d_control, control_smem, reps));
            v3_samples.push_back(time_v3(slice1_v3_fp16, d_overlay, d_u, d_v3, reps));
        }
    }
    auto median = [](std::vector<float> x) { std::sort(x.begin(), x.end()); return x[x.size() / 2]; };
    const float control_ms = median(control_samples), v3_ms = median(v3_samples);

    std::vector<float> attribution_v3_samples, descriptor_samples;
    float attribution_v3_ms = 0.0f, descriptor_ms = 0.0f;
    if (run_ablation) {
        for (int pair = 0; pair < 5; ++pair) {
            if (pair & 1) {
                descriptor_samples.push_back(time_v3(slice1_v3_descriptor_fp16, d_overlay, d_u,
                                                     d_descriptor, reps));
                attribution_v3_samples.push_back(time_v3(slice1_v3_fp16, d_overlay, d_u, d_v3, reps));
            } else {
                attribution_v3_samples.push_back(time_v3(slice1_v3_fp16, d_overlay, d_u, d_v3, reps));
                descriptor_samples.push_back(time_v3(slice1_v3_descriptor_fp16, d_overlay, d_u,
                                                     d_descriptor, reps));
            }
        }
        attribution_v3_ms = median(attribution_v3_samples);
        descriptor_ms = median(descriptor_samples);
    }
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, 0), "device props");

    std::printf("{\"device\":\"%s\",\"compute_capability\":\"%d.%d\",\"m\":%d,\"reps_per_pair\":%d,",
                prop.name, prop.major, prop.minor, M, reps);
    std::printf("\"control_ms\":%.6f,\"v3_ms\":%.6f,\"speedup_vs_fixed_control_pct\":%.6f,",
                control_ms, v3_ms, (1.591085f / v3_ms - 1.0f) * 100.0f);
    std::printf("\"bit_mismatches\":%llu,\"control_samples_ms\":[", bad);
    for (size_t i = 0; i < control_samples.size(); ++i) std::printf("%s%.6f", i ? "," : "", control_samples[i]);
    std::printf("],\"v3_samples_ms\":[");
    for (size_t i = 0; i < v3_samples.size(); ++i) std::printf("%s%.6f", i ? "," : "", v3_samples[i]);
    if (run_ablation) {
        std::printf("],\"attribution_ablation\":{\"v3_ms\":%.6f,\"descriptor_ms\":%.6f,"
                    "\"descriptor_minus_v3_ms\":%.6f,\"descriptor_delta_pct\":%.6f,"
                    "\"bit_mismatches\":%llu,\"v3_samples_ms\":[",
                    attribution_v3_ms, descriptor_ms, descriptor_ms - attribution_v3_ms,
                    (descriptor_ms / attribution_v3_ms - 1.0f) * 100.0f, bad_descriptor);
        for (size_t i = 0; i < attribution_v3_samples.size(); ++i) {
            std::printf("%s%.6f", i ? "," : "", attribution_v3_samples[i]);
        }
        std::printf("],\"descriptor_samples_ms\":[");
        for (size_t i = 0; i < descriptor_samples.size(); ++i) {
            std::printf("%s%.6f", i ? "," : "", descriptor_samples[i]);
        }
        std::printf("]}");
    } else {
        std::printf("],\"attribution_ablation\":null");
    }
    std::printf(",\"resources\":{");
    resource_json("control_fp16", slice1_control_fp16, control_smem); std::printf(",");
    resource_json("control_fp32", slice1_control_fp32, control_smem); std::printf(",");
    resource_json("v3_fp16", slice1_v3_fp16, V3_SMEM_BYTES); std::printf(",");
    resource_json("v3_fp32", slice1_v3_fp32, V3_SMEM_BYTES); std::printf(",");
    resource_json("descriptor_fp16", slice1_v3_descriptor_fp16, V3_SMEM_BYTES); std::printf(",");
    resource_json("descriptor_fp32", slice1_v3_descriptor_fp32, V3_SMEM_BYTES);
    std::printf("}}\n");

    cudaFree(d_bad);
    if (d_bad_descriptor) cudaFree(d_bad_descriptor);
    cudaFree(d_descriptor);
    cudaFree(d_v3);
    cudaFree(d_control);
    cudaFree(d_u);
    cudaFree(d_overlay);
    cudaFree(d_code);
    return bad == 0 ? 0 : 3;
}
