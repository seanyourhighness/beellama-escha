#include <cooperative_groups.h>

// Reuse the frozen Slice-1 control, MMA primitives, file parser, and helpers
// without changing the V1 source.  This translation unit supplies its own main.
#define main slice1_v1_embedded_main
#include "slice1_harness.cu"
#undef main

namespace cg = cooperative_groups;

constexpr size_t V2_DESC_BYTES = DESC_COUNT * sizeof(uint4);
constexpr int V2_A_SLOTS = 8;
constexpr size_t V2_A_BYTES = V2_A_SLOTS * BM * 16 * sizeof(half);
constexpr size_t V2_B_BYTES = BN * 16 * sizeof(half);
constexpr size_t V2_SMEM_BYTES = V2_DESC_BYTES + V2_A_BYTES + V2_B_BYTES;
static_assert(V2_SMEM_BYTES == 45056, "V2 must retain the official 45,056-byte shared class");

static __device__ __forceinline__ void v2_decode_descriptor(
        const uint32_t *__restrict__ record,
        const uint32_t runtime,
        half *__restrict__ sw,
        const int col,
        const int row) {
    const int off0 = runtime & 0x7f;
    const int sh0  = (runtime >> 7) & 0x1f;
    const int off1 = (runtime >> 12) & 0x7f;
    const int sh1  = (runtime >> 19) & 0x1f;
    const uint32_t ix0 = __funnelshift_r(record[off0], record[off0 + 1], sh0) & 0xffffu;
    const uint32_t ix1 = __funnelshift_r(record[off1], record[off1 + 1], sh1) & 0xffffu;
    sw[col * 16 + row]     = codebook_h(ix0);
    sw[col * 16 + row + 1] = codebook_h(ix1);
}

template <bool FP16_ACC>
static __device__ __forceinline__ void v2_kernel_body(
        const uint32_t *__restrict__ overlay,
        const uint4 *__restrict__ descriptors,
        const half *__restrict__ u,
        float *__restrict__ out) {
    extern __shared__ __align__(16) unsigned char v2_raw[];
    uint32_t *s_desc = reinterpret_cast<uint32_t *>(v2_raw);                   // [512] packed hot runtime words
    half *s_u = reinterpret_cast<half *>(v2_raw + V2_DESC_BYTES);              // [8][128][16]
    half *s_w = s_u + V2_A_SLOTS * BM * 16;                                   // [128][16]
    uint32_t *s_record = reinterpret_cast<uint32_t *>(s_u + 2 * BM * 16);      // [2][68], in inactive A arena

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int band = warp >> 2;
    const int row_group = warp & 3;
    const int band_tid = row_group * 32 + lane;
    const int row0 = blockIdx.x * BM;
    const int cta = blockIdx.y;
    const int oc0 = cta * BN;

    // Compact the hot word from each full A4 descriptor once.  Each thread
    // streams four descriptor words in the K loop; descriptor ordering fixes
    // the publication coordinates without rebuilding the Escha index chain.
    s_desc[tid] = descriptors[tid].w;
    s_desc[tid + 256] = descriptors[tid + 256].w;

    const int cp_m = tid >> 1;
    const int cp_h = (tid & 1) * 8;
    __pipeline_memcpy_async(s_u + cp_m * 16 + cp_h,
                            u + (row0 + cp_m) * IC + cp_h, 16);
    __pipeline_commit();
    __syncthreads();

    const int fragment_lane = band_tid & 31;
    const int base_col = 8 * (band_tid >> 6) + (fragment_lane >> 2);
    const int publish_row = 8 * ((band_tid >> 5) & 1) + 2 * (fragment_lane & 3);

    Acc16 ah[2][8] = {};
    Acc32 af[2][8] = {};
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<128> band_group = cg::tiled_partition<128>(block);

#pragma unroll 1
    for (int ti = 0; ti < NSTAGE; ++ti) {
        half *su_cur = s_u + (ti & 1) * BM * 16;
        half *su_nxt = s_u + ((ti & 1) ^ 1) * BM * 16;

        // Exactly 68 threads in each four-warp band publish its record once.
        if (band_tid < 68) {
            s_record[band * 68 + band_tid] =
                overlay[((cta * NSTAGE + ti) * 2 + band) * 68 + band_tid];
        }
        __pipeline_wait_prior(0);
        __syncthreads();

        if (ti + 1 < NSTAGE) {
            __pipeline_memcpy_async(su_nxt + cp_m * 16 + cp_h,
                                    u + (row0 + cp_m) * IC + (ti + 1) * 16 + cp_h, 16);
            __pipeline_commit();
        }

        const uint32_t *record = s_record + band * 68;
        half *band_w = s_w + band * 64 * 16;
#pragma unroll 4
        for (int g = 0; g < 4; ++g) {
            const uint32_t runtime = s_desc[band_tid + 128 * g];
            v2_decode_descriptor(record, runtime, band_w, base_col + 16 * g, publish_row);
        }

        // One four-warp publication.  Every row warp below consumes these same
        // 64x16 bytes; none re-evaluates a descriptor or codebook entry.
        band_group.sync();

        AFrag a[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const half2 *base = reinterpret_cast<const half2 *>(su_cur) +
                                (row_group * 32 + i * 16) * 8;
            const half2 *lane_base = base + (lane & 15) * 8 + (lane >> 4) * 4;
            load_a(a[i], lane_base);
        }

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            BFrag b;
            const half2 *base = reinterpret_cast<const half2 *>(s_w) +
                                (band * 64 + j * 8) * 8;
            const half2 *lane_base = base + (lane & 7) * 8 + ((lane >> 3) * 4) % 8;
            load_b(b, lane_base);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                if constexpr (FP16_ACC) mma(ah[i][j], a[i], b);
                else mma(af[i][j], a[i], b);
            }
        }
        __syncthreads();
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

extern "C" __global__ __launch_bounds__(256, 1)
void slice1_v2_fp16(const uint32_t *, const uint32_t *overlay, const uint4 *descriptors,
                    const half *u, float *out) {
    v2_kernel_body<true>(overlay, descriptors, u, out);
}

extern "C" __global__ __launch_bounds__(256, 1)
void slice1_v2_fp32(const uint32_t *, const uint32_t *overlay, const uint4 *descriptors,
                    const half *u, float *out) {
    v2_kernel_body<false>(overlay, descriptors, u, out);
}

template <typename Kernel>
static float time_v2(Kernel kernel, const uint32_t *code, const uint32_t *overlay,
                     const uint4 *descriptors, const half *u, float *out, int reps) {
    dim3 block_dim(32, 8), grid_dim(M / BM, OC / BN);
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "v2 event create start");
    cuda_check(cudaEventCreate(&stop), "v2 event create stop");
    for (int i = 0; i < 2; ++i) {
        kernel<<<grid_dim, block_dim, V2_SMEM_BYTES>>>(code, overlay, descriptors, u, out);
    }
    cuda_check(cudaDeviceSynchronize(), "v2 warmup");
    cuda_check(cudaEventRecord(start), "v2 event start");
    for (int i = 0; i < reps; ++i) {
        kernel<<<grid_dim, block_dim, V2_SMEM_BYTES>>>(code, overlay, descriptors, u, out);
    }
    cuda_check(cudaEventRecord(stop), "v2 event stop");
    cuda_check(cudaEventSynchronize(stop), "v2 event sync");
    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "v2 elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / reps;
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
    if (std::memcmp(h.magic, "ESCHA-MMA-V1", 12) || h.value[0] != 1 ||
        h.value[2] != 2 || h.value[3] != IC || h.value[4] != OC) {
        throw std::runtime_error("bad overlay header");
    }
    auto desc_bytes = read_range(overlay_path, h.descriptor_offset, DESC_COUNT * sizeof(Descriptor));
    auto payload = read_range(overlay_path, h.payload_offset, static_cast<size_t>(h.payload_bytes));
    auto canonical = read_range(source, code_offset,
                                static_cast<size_t>(NSTAGE) * NCT * 16 * sizeof(uint32_t));

    std::vector<half> host_u(static_cast<size_t>(M) * IC);
    for (size_t i = 0; i < host_u.size(); ++i) {
        host_u[i] = __float2half(std::sin(float(i % 8191) * 0.001f));
    }

    uint32_t *d_code = nullptr, *d_overlay = nullptr;
    uint4 *d_desc = nullptr;
    half *d_u = nullptr;
    float *d_control = nullptr, *d_v2 = nullptr;
    const size_t out_elems = static_cast<size_t>(M) * OC;
    cuda_check(cudaMalloc(&d_code, canonical.size()), "malloc code");
    cuda_check(cudaMalloc(&d_overlay, payload.size()), "malloc overlay");
    cuda_check(cudaMalloc(&d_desc, desc_bytes.size()), "malloc descriptors");
    cuda_check(cudaMalloc(&d_u, host_u.size() * sizeof(half)), "malloc u");
    cuda_check(cudaMalloc(&d_control, out_elems * sizeof(float)), "malloc control out");
    cuda_check(cudaMalloc(&d_v2, out_elems * sizeof(float)), "malloc v2 out");
    cuda_check(cudaMemcpy(d_code, canonical.data(), canonical.size(), cudaMemcpyHostToDevice), "copy code");
    cuda_check(cudaMemcpy(d_overlay, payload.data(), payload.size(), cudaMemcpyHostToDevice), "copy overlay");
    cuda_check(cudaMemcpy(d_desc, desc_bytes.data(), desc_bytes.size(), cudaMemcpyHostToDevice), "copy descriptors");
    cuda_check(cudaMemcpy(d_u, host_u.data(), host_u.size() * sizeof(half), cudaMemcpyHostToDevice), "copy u");

    const size_t control_smem = 8 * 24 * sizeof(uint2) + 2 * BM * 16 * sizeof(half) + BN * 16 * sizeof(half);
    dim3 block_dim(32, 8), grid_dim(M / BM, OC / BN);
    slice1_control_fp16<<<grid_dim, block_dim, control_smem>>>(d_code, d_overlay, d_u, d_control);
    slice1_v2_fp16<<<grid_dim, block_dim, V2_SMEM_BYTES>>>(d_code, d_overlay, d_desc, d_u, d_v2);
    cuda_check(cudaDeviceSynchronize(), "correctness kernels");

    unsigned long long *d_bad = nullptr, bad = 0;
    cuda_check(cudaMalloc(&d_bad, sizeof(bad)), "malloc mismatch");
    cuda_check(cudaMemset(d_bad, 0, sizeof(bad)), "clear mismatch");
    compare_bits<<<static_cast<unsigned>((out_elems + 255) / 256), 256>>>(
        reinterpret_cast<uint32_t *>(d_control), reinterpret_cast<uint32_t *>(d_v2), out_elems, d_bad);
    cuda_check(cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost), "copy mismatch");

    std::vector<float> ctl, cand;
    for (int pair = 0; pair < 5; ++pair) {
        if (pair & 1) {
            cand.push_back(time_v2(slice1_v2_fp16, d_code, d_overlay, d_desc, d_u, d_v2, reps));
            ctl.push_back(time_kernel(slice1_control_fp16, d_code, d_overlay, d_u, d_control, control_smem, reps));
        } else {
            ctl.push_back(time_kernel(slice1_control_fp16, d_code, d_overlay, d_u, d_control, control_smem, reps));
            cand.push_back(time_v2(slice1_v2_fp16, d_code, d_overlay, d_desc, d_u, d_v2, reps));
        }
    }
    auto median = [](std::vector<float> x) { std::sort(x.begin(), x.end()); return x[x.size() / 2]; };
    const float ctl_ms = median(ctl), v2_ms = median(cand);
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, 0), "device props");

    std::printf("{\"device\":\"%s\",\"compute_capability\":\"%d.%d\",\"m\":%d,\"reps_per_pair\":%d,",
                prop.name, prop.major, prop.minor, M, reps);
    std::printf("\"control_ms\":%.6f,\"v2_ms\":%.6f,\"speedup_pct\":%.6f,\"bit_mismatches\":%llu,\"resources\":{",
                ctl_ms, v2_ms, (ctl_ms / v2_ms - 1.0f) * 100.0f, bad);
    resource_json("control_fp16", slice1_control_fp16, control_smem); std::printf(",");
    resource_json("control_fp32", slice1_control_fp32, control_smem); std::printf(",");
    resource_json("v2_fp16", slice1_v2_fp16, V2_SMEM_BYTES); std::printf(",");
    resource_json("v2_fp32", slice1_v2_fp32, V2_SMEM_BYTES);
    std::printf("}}\n");

    cudaFree(d_bad);
    cudaFree(d_v2);
    cudaFree(d_control);
    cudaFree(d_u);
    cudaFree(d_desc);
    cudaFree(d_overlay);
    cudaFree(d_code);
    return bad == 0 ? 0 : 3;
}
