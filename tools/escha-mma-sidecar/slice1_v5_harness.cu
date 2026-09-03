#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Reuse the frozen shared-B control, exact codebook, MMA primitives, parser,
// comparison kernel, and host helpers. V1-V4 remain untouched.
#define main slice1_v1_embedded_main
#include "slice1_harness.cu"
#undef main

constexpr int V5_PRODUCER_WARPS = 4;
constexpr int V5_CONSUMER_WARPS = 8;
constexpr int V5_WARPS = V5_PRODUCER_WARPS + V5_CONSUMER_WARPS;
constexpr int V5_THREADS = V5_WARPS * 32;
constexpr int V5_RING = 2;
constexpr int V5_NSUPER4 = NSTAGE / 4;
constexpr size_t V5_B_BYTES = V5_RING * BN * 16 * sizeof(half);
constexpr size_t V5_A_BYTES = V5_RING * BM * 16 * sizeof(half);
constexpr size_t V5_PAYLOAD_BYTES = V5_RING * 2 * 64 * sizeof(uint32_t);
constexpr size_t V5_SMEM_BYTES = V5_B_BYTES + V5_A_BYTES + V5_PAYLOAD_BYTES;
static_assert(V5_THREADS == 384, "V5 must launch 12 warps");
static_assert(V5_B_BYTES == 8192, "V5 B ring must be 8 KiB");
static_assert(V5_A_BYTES == 8192, "V5 shared A ring must be 8 KiB");
static_assert(V5_PAYLOAD_BYTES == 1024, "V5 payload ring must be 1 KiB");
static_assert(V5_SMEM_BYTES == 17408, "V5 shared footprint must be 17,408 B");

// Named barriers. Ready/free use all 384 threads split across role-specific
// arrive and wait endpoints. Payload and A barriers are role-local.
constexpr int V5_READY0 = 0;
constexpr int V5_READY1 = 1;
constexpr int V5_FREE0 = 2;
constexpr int V5_FREE1 = 3;
constexpr int V5_PAYLOAD_READY = 4;
constexpr int V5_A_READY = 5;

template <int ID, int COUNT>
static __device__ __forceinline__ void v5_bar_arrive() {
    asm volatile("bar.arrive %0, %1;" :: "n"(ID), "n"(COUNT) : "memory");
}

template <int ID, int COUNT>
static __device__ __forceinline__ void v5_bar_wait() {
    // bar.sync is the wait endpoint: it contributes this role's arrival and
    // blocks until the complementary nonblocking bar.arrive completes COUNT.
    asm volatile("bar.sync %0, %1;" :: "n"(ID), "n"(COUNT) : "memory");
}

static __device__ __forceinline__ void v5_ready_arrive(int slot) {
    if (slot == 0) v5_bar_arrive<V5_READY0, V5_THREADS>();
    else           v5_bar_arrive<V5_READY1, V5_THREADS>();
}

static __device__ __forceinline__ void v5_ready_wait(int slot) {
    if (slot == 0) v5_bar_wait<V5_READY0, V5_THREADS>();
    else           v5_bar_wait<V5_READY1, V5_THREADS>();
}

static __device__ __forceinline__ void v5_free_arrive(int slot) {
    if (slot == 0) v5_bar_arrive<V5_FREE0, V5_THREADS>();
    else           v5_bar_arrive<V5_FREE1, V5_THREADS>();
}

static __device__ __forceinline__ void v5_free_wait(int slot) {
    if (slot == 0) v5_bar_wait<V5_FREE0, V5_THREADS>();
    else           v5_bar_wait<V5_FREE1, V5_THREADS>();
}

static __device__ __forceinline__ const uint32_t *v5_source_stage(
        const uint32_t *__restrict__ overlay, int cta, int stage, int band) {
    // Frozen V3 order: [cta][BK64 super][band][K16][tile][word].
    const int super4 = stage >> 2;
    const int stage4 = stage & 3;
    return overlay + (((cta * V5_NSUPER4 + super4) * 2 + band) * 4 + stage4) * 64;
}

static __device__ __forceinline__ uint32_t v5_map(int r, int ccl) {
    int dsp = (30 - 2 * (dep_pi(r) + 32 * ccl + 4 * (ccl >> 3))) % 512;
    if (dsp < 0) dsp += 512;
    const int group = dsp >> 5;
    const int word = group ? 16 - group : 0;
    return word | ((word ? word - 1 : 15) << 4) | ((dsp & 31) << 8);
}

static __device__ __forceinline__ half v5_decode(
        const uint32_t *__restrict__ pay, uint32_t map) {
    const int word = map & 15;
    const int previous = (map >> 4) & 15;
    const int shift = (map >> 8) & 31;
    return codebook_h(__funnelshift_r(pay[word], pay[previous], shift) & 0xffffu);
}

static __device__ __forceinline__ void v5_producer(
        const uint32_t *__restrict__ overlay,
        half *__restrict__ s_b,
        uint32_t *__restrict__ s_payload,
        int lane, int warp, int cta) {
    const int producer_tid = warp * 32 + lane;
    const int r = producer_tid & 15;
    const int ccl = producer_tid >> 4;
    const uint32_t map0 = v5_map(r, ccl);
    const uint32_t map1 = v5_map(r, ccl + 8);

#pragma unroll 1
    for (int ti = 0; ti < NSTAGE; ++ti) {
        const int slot = ti & 1;
        if (ti >= V5_RING) {
            // Consumers arrived nonblocking after their final HMMA on this
            // slot. Producers wait only when the slot is about to be reused.
            v5_free_wait(slot);
        }

        const int band_word = producer_tid;
        const int band = band_word >> 6;
        const int word = band_word & 63;
        s_payload[slot * 128 + band * 64 + word] =
            v5_source_stage(overlay, cta, ti, band)[word];
        v5_bar_wait<V5_PAYLOAD_READY, V5_PRODUCER_WARPS * 32>();

#pragma unroll
        for (int b = 0; b < 2; ++b) {
            half *dst = s_b + (slot * BN + b * 64) * 16;
            const uint32_t *payload = s_payload + slot * 128 + b * 64;
#pragma unroll
            for (int tile = 0; tile < 4; ++tile) {
                const uint32_t *pay = payload + tile * 16;
                dst[(ccl + tile * 16) * 16 + r] = v5_decode(pay, map0);
                dst[(ccl + 8 + tile * 16) * 16 + r] = v5_decode(pay, map1);
            }
        }
        // Publish B without waiting. The producer group can decode into the
        // other slot while consumers wake, LDSM, and issue tensor work.
        v5_ready_arrive(slot);
    }
}

template <bool FP16_ACC>
static __device__ __forceinline__ void v5_consumer(
        const half *__restrict__ u,
        half *__restrict__ s_a,
        const half *__restrict__ s_b,
        float *__restrict__ out,
        int lane, int warp, int row0, int oc0) {
    const int consumer_warp = warp - V5_PRODUCER_WARPS;
    const int consumer_tid = consumer_warp * 32 + lane;
    const int row_group = consumer_warp >> 1;
    const int band = consumer_warp & 1;

    Acc16 ah[2][8] = {};
    Acc32 af[2][8] = {};

#pragma unroll 1
    for (int ti = 0; ti < NSTAGE; ++ti) {
        const int slot = ti & 1;
        half *a_slot = s_a + slot * BM * 16;

        // Exactly 256 vector copies cover one [128][16] A tile. This shared
        // copy is consumed by both column bands.
        const int cp_m = consumer_tid >> 1;
        const int cp_h = (consumer_tid & 1) * 8;
        const uint4 av = *reinterpret_cast<const uint4 *>(
            u + (row0 + cp_m) * IC + ti * 16 + cp_h);
        *reinterpret_cast<uint4 *>(a_slot + cp_m * 16 + cp_h) = av;
        v5_bar_wait<V5_A_READY, V5_CONSUMER_WARPS * 32>();

        // Producers have already arrived after decoded-B STS. Consumer sync
        // is the wait endpoint and completes the ready phase.
        v5_ready_wait(slot);

#pragma unroll
        for (int i = 0; i < 2; ++i) {
            AFrag a;
            const half2 *a_base = reinterpret_cast<const half2 *>(a_slot) +
                                  (row_group * 32 + i * 16) * 8;
            load_a(a, a_base + (lane & 15) * 8 + (lane >> 4) * 4);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                BFrag b;
                const half2 *b_base = reinterpret_cast<const half2 *>(s_b) +
                                      (slot * BN + band * 64 + j * 8) * 8;
                load_b(b, b_base + (lane & 7) * 8 + ((lane >> 3) * 4) % 8);
                if constexpr (FP16_ACC) mma(ah[i][j], a, b);
                else                    mma(af[i][j], a, b);
            }
        }

        // Release the B slot without waiting. Producers wait only on reuse.
        v5_free_arrive(slot);
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

template <bool FP16_ACC>
static __device__ __forceinline__ void v5_kernel_body(
        const uint32_t *__restrict__ overlay,
        const half *__restrict__ u,
        float *__restrict__ out) {
    extern __shared__ __align__(16) unsigned char v5_raw[];
    half *s_b = reinterpret_cast<half *>(v5_raw);
    half *s_a = reinterpret_cast<half *>(v5_raw + V5_B_BYTES);
    uint32_t *s_payload = reinterpret_cast<uint32_t *>(v5_raw + V5_B_BYTES + V5_A_BYTES);

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row0 = blockIdx.x * BM;
    const int cta = blockIdx.y;
    const int oc0 = cta * BN;

    if (warp < V5_PRODUCER_WARPS) {
        v5_producer(overlay, s_b, s_payload, lane, warp, cta);
    } else {
        v5_consumer<FP16_ACC>(u, s_a, s_b, out, lane, warp, row0, oc0);
    }
}

extern "C" __global__ __launch_bounds__(V5_THREADS, 2)
void slice1_v5_fp16(const uint32_t *overlay, const half *u, float *out) {
    v5_kernel_body<true>(overlay, u, out);
}

extern "C" __global__ __launch_bounds__(V5_THREADS, 2)
void slice1_v5_fp32(const uint32_t *overlay, const half *u, float *out) {
    v5_kernel_body<false>(overlay, u, out);
}

template <typename Kernel>
static float time_v5(Kernel kernel, const uint32_t *overlay,
                     const half *u, float *out, int reps) {
    dim3 block_dim(32, V5_WARPS), grid_dim(M / BM, OC / BN);
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "v5 event create start");
    cuda_check(cudaEventCreate(&stop), "v5 event create stop");
    for (int i = 0; i < 2; ++i) kernel<<<grid_dim, block_dim, V5_SMEM_BYTES>>>(overlay, u, out);
    cuda_check(cudaDeviceSynchronize(), "v5 warmup");
    cuda_check(cudaEventRecord(start), "v5 event start");
    for (int i = 0; i < reps; ++i) kernel<<<grid_dim, block_dim, V5_SMEM_BYTES>>>(overlay, u, out);
    cuda_check(cudaEventRecord(stop), "v5 event stop");
    cuda_check(cudaEventSynchronize(stop), "v5 event sync");
    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "v5 elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / reps;
}

template <typename Candidate>
static void collect_v5_pairs(Candidate candidate,
                             const uint32_t *code, const uint32_t *overlay,
                             const half *u, float *control_out, float *candidate_out,
                             size_t control_smem, int reps,
                             std::vector<float> &control, std::vector<float> &result) {
    for (int pair = 0; pair < 5; ++pair) {
        if (pair & 1) {
            result.push_back(time_v5(candidate, overlay, u, candidate_out, reps));
            control.push_back(time_kernel(slice1_control_fp16, code, overlay, u,
                                          control_out, control_smem, reps));
        } else {
            control.push_back(time_kernel(slice1_control_fp16, code, overlay, u,
                                          control_out, control_smem, reps));
            result.push_back(time_v5(candidate, overlay, u, candidate_out, reps));
        }
    }
}

template <typename Kernel>
static void v5_resource_json(const char *name, Kernel kernel, size_t dynamic_smem, int threads) {
    cudaFuncAttributes a{};
    cuda_check(cudaFuncGetAttributes(&a, kernel), "v5 func attrs");
    int blocks = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, threads, dynamic_smem), "v5 occupancy");
    std::printf("\"%s\":{\"registers\":%d,\"local_bytes\":%zu,\"static_shared_bytes\":%zu,"
                "\"dynamic_shared_bytes\":%zu,\"max_threads\":%d,\"active_ctas_per_sm\":%d}",
                name, a.numRegs, a.localSizeBytes, a.sharedSizeBytes,
                dynamic_smem, a.maxThreadsPerBlock, blocks);
}

static void v5_print_samples(const char *name, const std::vector<float> &samples) {
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
    float *d_control = nullptr, *d_v5 = nullptr;
    const size_t out_elems = static_cast<size_t>(M) * OC;
    cuda_check(cudaMalloc(&d_code, canonical.size()), "malloc code");
    cuda_check(cudaMalloc(&d_overlay, payload.size()), "malloc overlay");
    cuda_check(cudaMalloc(&d_u, host_u.size() * sizeof(half)), "malloc u");
    cuda_check(cudaMalloc(&d_control, out_elems * sizeof(float)), "malloc control out");
    cuda_check(cudaMalloc(&d_v5, out_elems * sizeof(float)), "malloc v5 out");
    cuda_check(cudaMemcpy(d_code, canonical.data(), canonical.size(), cudaMemcpyHostToDevice), "copy code");
    cuda_check(cudaMemcpy(d_overlay, payload.data(), payload.size(), cudaMemcpyHostToDevice), "copy overlay");
    cuda_check(cudaMemcpy(d_u, host_u.data(), host_u.size() * sizeof(half), cudaMemcpyHostToDevice), "copy u");

    const size_t control_smem = 8 * 24 * sizeof(uint2) +
                                2 * BM * 16 * sizeof(half) + BN * 16 * sizeof(half);
    dim3 control_block(32, 8), v5_block(32, V5_WARPS);
    dim3 grid(M / BM, OC / BN);
    slice1_control_fp16<<<grid, control_block, control_smem>>>(d_code, d_overlay, d_u, d_control);
    slice1_v5_fp16<<<grid, v5_block, V5_SMEM_BYTES>>>(d_overlay, d_u, d_v5);
    cuda_check(cudaDeviceSynchronize(), "v5 correctness kernels");

    unsigned long long *d_bad = nullptr, bad = 0;
    cuda_check(cudaMalloc(&d_bad, sizeof(bad)), "malloc v5 mismatch");
    cuda_check(cudaMemset(d_bad, 0, sizeof(bad)), "clear v5 mismatch");
    compare_bits<<<static_cast<unsigned>((out_elems + 255) / 256), 256>>>(
        reinterpret_cast<uint32_t *>(d_control), reinterpret_cast<uint32_t *>(d_v5),
        out_elems, d_bad);
    cuda_check(cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost), "copy v5 mismatch");

    std::vector<float> control_samples, candidate_samples;
    collect_v5_pairs(slice1_v5_fp16, d_code, d_overlay, d_u, d_control, d_v5,
                     control_smem, reps, control_samples, candidate_samples);
    auto median = [](std::vector<float> x) {
        std::sort(x.begin(), x.end());
        return x[x.size() / 2];
    };
    const float control_ms = median(control_samples);
    const float candidate_ms = median(candidate_samples);
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, 0), "v5 device props");

    std::printf("{\"device\":\"%s\",\"compute_capability\":\"%d.%d\",\"m\":%d,"
                "\"reps_per_pair\":%d,\"control_ms\":%.6f,\"candidate_ms\":%.6f,"
                "\"bit_mismatches\":%llu,", prop.name, prop.major, prop.minor,
                M, reps, control_ms, candidate_ms, bad);
    v5_print_samples("control_samples_ms", control_samples); std::printf(",");
    v5_print_samples("candidate_samples_ms", candidate_samples); std::printf(",\"resources\":{");
    v5_resource_json("control_fp16", slice1_control_fp16, control_smem, 256); std::printf(",");
    v5_resource_json("control_fp32", slice1_control_fp32, control_smem, 256); std::printf(",");
    v5_resource_json("v5_fp16", slice1_v5_fp16, V5_SMEM_BYTES, V5_THREADS); std::printf(",");
    v5_resource_json("v5_fp32", slice1_v5_fp32, V5_SMEM_BYTES, V5_THREADS);
    std::printf("}}\n");

    cudaFree(d_bad); cudaFree(d_v5); cudaFree(d_control); cudaFree(d_u);
    cudaFree(d_overlay); cudaFree(d_code);
    return bad == 0 ? 0 : 3;
}
