#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int IC = 5120;
constexpr int OC = 17408;
constexpr int M = 2048;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int NSTAGE = IC / 16;
constexpr int NCT = OC / 16;
constexpr int DESC_COUNT = 512;

__constant__ uint32_t g_access[DESC_COUNT];

static inline void cuda_check(cudaError_t e, const char *where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
        std::exit(2);
    }
}

struct AFrag { uint32_t x[4]; };
struct BFrag { uint32_t x[2]; };
struct Acc16 { uint32_t x[2]; };
struct Acc32 { float x[4]; };

static __device__ __forceinline__ int dep_pi(int r) {
    return (r & 1) | (((r >> 3) & 1) << 1) | (((r >> 1) & 3) << 3);
}

static __device__ __forceinline__ half codebook_h(uint32_t idx) {
    uint32_t x = idx * 0xcbac1fedu;
    asm("lop3.b32 %0, %1, %2, %3, 0x6a;"
        : "=r"(x) : "r"(x), "n"(0x8fff8fffu), "n"(0x3b603b60u));
    half2 h;
    memcpy(&h, &x, sizeof(h));
    return __hadd(__low2half(h), __high2half(h));
}

static __device__ __forceinline__ void load_a(AFrag &a, const half2 *base) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(base));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a.x[0]), "=r"(a.x[1]), "=r"(a.x[2]), "=r"(a.x[3]) : "r"(smem));
}

static __device__ __forceinline__ void load_b(BFrag &b, const half2 *base) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(base));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
        : "=r"(b.x[0]), "=r"(b.x[1]) : "r"(smem));
}

static __device__ __forceinline__ void mma(Acc16 &d, const AFrag &a, const BFrag &b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(d.x[0]), "+r"(d.x[1])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]),
          "r"(b.x[0]), "r"(b.x[1]));
}

static __device__ __forceinline__ void mma(Acc32 &d, const AFrag &a, const BFrag &b) {
    uint32_t *di = reinterpret_cast<uint32_t *>(d.x);
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(di[0]), "+r"(di[1]), "+r"(di[2]), "+r"(di[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]),
          "r"(b.x[0]), "r"(b.x[1]));
}

static __device__ __forceinline__ int c_i(int l, int lane) {
    return (l / 2) * 8 + lane / 4;
}

static __device__ __forceinline__ int c_j(int l, int lane) {
    return (lane & 3) * 2 + (l & 1);
}

template <bool FP16_ACC, bool SIDECAR>
static __device__ __forceinline__ void kernel_body(
        const uint32_t *__restrict__ code,
        const uint32_t *__restrict__ overlay,
        const half *__restrict__ u,
        float *__restrict__ out) {
    constexpr int NWD = 16;
    constexpr int MAX_W = 24;
    extern __shared__ char raw[];
    uint2 *s_pay = reinterpret_cast<uint2 *>(raw);
    half *s_u = reinterpret_cast<half *>(SIDECAR ? raw : raw + 8 * MAX_W * sizeof(uint2));
    half *s_w = s_u + 2 * BM * 16;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int row0 = blockIdx.x * BM;
    const int cta = blockIdx.y;
    const int oc0 = cta * BN;
    const int wm = warp >> 1;
    const int band = warp & 1;

    Acc16 ah[2][8] = {};
    Acc32 af[2][8] = {};

    const int cp_m = tid >> 1;
    const int cp_h = (tid & 1) * 8;
    {
        __pipeline_memcpy_async(s_u + cp_m * 16 + cp_h,
                                u + (row0 + cp_m) * IC + cp_h, 16);
        __pipeline_commit();
    }

    const int dr = tid & 15;
    const int dccl = tid >> 4;
    int dsp = 0, dw0 = 0, dw1 = 0, dsh = 0;
    if constexpr (!SIDECAR) {
        dsp = ((32 - 2) - 2 * (dep_pi(dr) + 32 * dccl + 4 * (dccl >> 3))) % 512;
        if (dsp < 0) dsp += 512;
        const int dg0 = dsp >> 5;
        dw0 = dg0 ? NWD - dg0 : 0;
        dw1 = dw0 ? dw0 - 1 : NWD - 1;
        dsh = dsp & 31;
    }

    const bool has_pay = tid < 8 * NWD;
    const int pt = tid / NWD;
    const int pw = tid & 15;
    uint32_t ppre = 0;
    if constexpr (!SIDECAR) {
        if (has_pay) ppre = code[(cta * 8 + pt) * NWD + pw];
    }

#pragma unroll 1
    for (int ti = 0; ti < NSTAGE; ++ti) {
        half *su_cur = s_u + (ti & 1) * BM * 16;
        half *su_nxt = s_u + ((ti & 1) ^ 1) * BM * 16;

        if constexpr (!SIDECAR) {
            if (has_pay) {
                s_pay[pt * MAX_W + pw].y = ppre;
                s_pay[pt * MAX_W + ((pw + 1) & 15)].x = ppre;
                if (ti + 1 < NSTAGE) ppre = code[((ti + 1) * NCT + cta * 8 + pt) * NWD + pw];
            }
        }
        __pipeline_wait_prior(0);
        __syncthreads();

        if (ti + 1 < NSTAGE) {
            __pipeline_memcpy_async(su_nxt + cp_m * 16 + cp_h,
                                    u + (row0 + cp_m) * IC + (ti + 1) * 16 + cp_h, 16);
            __pipeline_commit();
        }

        if constexpr (!SIDECAR) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const uint2 *pay = s_pay + k * MAX_W;
                s_w[(dccl + 16 * k) * 16 + dr] =
                    codebook_h(__funnelshift_r(pay[dw0].y, pay[dw0].x, dsh) & 0xffffu);
            }
            __syncthreads();
        }

        AFrag a[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const half2 *base = reinterpret_cast<const half2 *>(su_cur) + (wm * 32 + i * 16) * 8;
            const half2 *lane_base = base + (lane & 15) * 8 + (lane >> 4) * 4;
            load_a(a[i], lane_base);
        }

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            BFrag b;
            if constexpr (SIDECAR) {
                const uint32_t *record = overlay + (((cta * NSTAGE + ti) * 2 + band) * 68);
#pragma unroll
                for (int slot = 0; slot < 2; ++slot) {
                    const uint32_t d = g_access[j * 64 + slot * 32 + lane];
                    const int off0 = d & 0x7f;
                    const int sh0 = (d >> 7) & 0x1f;
                    const int off1 = (d >> 12) & 0x7f;
                    const int sh1 = (d >> 19) & 0x1f;
                    const uint32_t ix0 = __funnelshift_r(record[off0], record[off0 + 1], sh0) & 0xffffu;
                    const uint32_t ix1 = __funnelshift_r(record[off1], record[off1 + 1], sh1) & 0xffffu;
                    const half2 hv = __halves2half2(codebook_h(ix0), codebook_h(ix1));
                    memcpy(&b.x[slot], &hv, sizeof(hv));
                }
            } else {
                const half2 *base = reinterpret_cast<const half2 *>(s_w) + (band * 64 + j * 8) * 8;
                const half2 *lane_base = base + (lane & 7) * 8 + ((lane >> 3) * 4) % 8;
                load_b(b, lane_base);
            }
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
                    const int m0 = wm * 32 + i * 16 + c_i(2 * l, lane);
                    const int n0 = band * 64 + j * 8 + c_j(2 * l, lane);
                    const int m1 = wm * 32 + i * 16 + c_i(2 * l + 1, lane);
                    const int n1 = band * 64 + j * 8 + c_j(2 * l + 1, lane);
                    out[(row0 + m0) * OC + oc0 + n0] = __half2float(__low2half(v));
                    out[(row0 + m1) * OC + oc0 + n1] = __half2float(__high2half(v));
                }
            } else {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int m = wm * 32 + i * 16 + c_i(l, lane);
                    const int n = band * 64 + j * 8 + c_j(l, lane);
                    out[(row0 + m) * OC + oc0 + n] = af[i][j].x[l];
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void slice1_control_fp16(const uint32_t *code, const uint32_t *overlay, const half *u, float *out) {
    kernel_body<true, false>(code, overlay, u, out);
}
extern "C" __global__ __launch_bounds__(256, 1)
void slice1_control_fp32(const uint32_t *code, const uint32_t *overlay, const half *u, float *out) {
    kernel_body<false, false>(code, overlay, u, out);
}
extern "C" __global__ __launch_bounds__(256, 1)
void slice1_sidecar_fp16(const uint32_t *code, const uint32_t *overlay, const half *u, float *out) {
    kernel_body<true, true>(code, overlay, u, out);
}
extern "C" __global__ __launch_bounds__(256, 1)
void slice1_sidecar_fp32(const uint32_t *code, const uint32_t *overlay, const half *u, float *out) {
    kernel_body<false, true>(code, overlay, u, out);
}

__global__ void compare_bits(const uint32_t *a, const uint32_t *b, size_t n, unsigned long long *bad) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && a[i] != b[i]) atomicAdd(bad, 1ULL);
}

#pragma pack(push, 1)
struct HeaderPrefix {
    char magic[16];
    uint32_t value[15];
    uint64_t descriptor_offset, payload_offset, payload_bytes, total_bytes;
};
struct Descriptor {
    uint8_t semantic[12];
    uint32_t runtime;
};
#pragma pack(pop)
static_assert(sizeof(HeaderPrefix) == 108);
static_assert(sizeof(Descriptor) == 16);

static std::vector<uint8_t> read_range(const std::string &path, uint64_t offset, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(static_cast<std::streamoff>(offset));
    std::vector<uint8_t> out(bytes);
    f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(bytes));
    if (static_cast<size_t>(f.gcount()) != bytes) throw std::runtime_error("short read " + path);
    return out;
}

template <typename Kernel>
static float time_kernel(Kernel kernel, const uint32_t *code, const uint32_t *overlay,
                         const half *u, float *out, size_t smem, int reps) {
    dim3 block(32, 8), grid(M / BM, OC / BN);
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "event create start");
    cuda_check(cudaEventCreate(&stop), "event create stop");
    for (int i = 0; i < 2; ++i) kernel<<<grid, block, smem>>>(code, overlay, u, out);
    cuda_check(cudaDeviceSynchronize(), "warmup");
    cuda_check(cudaEventRecord(start), "event start");
    for (int i = 0; i < reps; ++i) kernel<<<grid, block, smem>>>(code, overlay, u, out);
    cuda_check(cudaEventRecord(stop), "event stop");
    cuda_check(cudaEventSynchronize(stop), "event sync");
    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / reps;
}

template <typename Kernel>
static void resource_json(const char *name, Kernel kernel, size_t dynamic_smem) {
    cudaFuncAttributes a{};
    cuda_check(cudaFuncGetAttributes(&a, kernel), "func attrs");
    int blocks = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 256, dynamic_smem), "occupancy");
    std::printf("\"%s\":{\"registers\":%d,\"local_bytes\":%zu,\"static_shared_bytes\":%zu,"
                "\"dynamic_shared_bytes\":%zu,\"max_threads\":%d,\"active_ctas_per_sm\":%d}",
                name, a.numRegs, a.localSizeBytes, a.sharedSizeBytes, dynamic_smem, a.maxThreadsPerBlock, blocks);
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
    if (std::memcmp(h.magic, "ESCHA-MMA-V1", 12) || h.value[0] != 1 || h.value[2] != 2 || h.value[3] != IC || h.value[4] != OC)
        throw std::runtime_error("bad overlay header");
    auto desc_bytes = read_range(overlay_path, h.descriptor_offset, DESC_COUNT * sizeof(Descriptor));
    auto payload = read_range(overlay_path, h.payload_offset, static_cast<size_t>(h.payload_bytes));
    auto canonical = read_range(source, code_offset, static_cast<size_t>(NSTAGE) * NCT * 16 * sizeof(uint32_t));
    std::vector<uint32_t> access(DESC_COUNT);
    for (int i = 0; i < DESC_COUNT; ++i) {
        Descriptor d{};
        std::memcpy(&d, desc_bytes.data() + i * sizeof(d), sizeof(d));
        access[i] = d.runtime;
    }

    std::vector<half> host_u(static_cast<size_t>(M) * IC);
    for (size_t i = 0; i < host_u.size(); ++i) host_u[i] = __float2half(std::sin(float(i % 8191) * 0.001f));

    uint32_t *d_code = nullptr, *d_overlay = nullptr;
    half *d_u = nullptr;
    float *d_control = nullptr, *d_sidecar = nullptr;
    const size_t out_elems = static_cast<size_t>(M) * OC;
    cuda_check(cudaMalloc(&d_code, canonical.size()), "malloc code");
    cuda_check(cudaMalloc(&d_overlay, payload.size()), "malloc overlay");
    cuda_check(cudaMalloc(&d_u, host_u.size() * sizeof(half)), "malloc u");
    cuda_check(cudaMalloc(&d_control, out_elems * sizeof(float)), "malloc control out");
    cuda_check(cudaMalloc(&d_sidecar, out_elems * sizeof(float)), "malloc sidecar out");
    cuda_check(cudaMemcpy(d_code, canonical.data(), canonical.size(), cudaMemcpyHostToDevice), "copy code");
    cuda_check(cudaMemcpy(d_overlay, payload.data(), payload.size(), cudaMemcpyHostToDevice), "copy overlay");
    cuda_check(cudaMemcpy(d_u, host_u.data(), host_u.size() * sizeof(half), cudaMemcpyHostToDevice), "copy u");
    cuda_check(cudaMemcpyToSymbol(g_access, access.data(), access.size() * sizeof(uint32_t)), "copy desc");

    const size_t control_smem = 8 * 24 * sizeof(uint2) + 2 * BM * 16 * sizeof(half) + BN * 16 * sizeof(half);
    const size_t sidecar_smem = 2 * BM * 16 * sizeof(half);
    dim3 block(32, 8), grid(M / BM, OC / BN);
    slice1_control_fp16<<<grid, block, control_smem>>>(d_code, d_overlay, d_u, d_control);
    slice1_sidecar_fp16<<<grid, block, sidecar_smem>>>(d_code, d_overlay, d_u, d_sidecar);
    cuda_check(cudaDeviceSynchronize(), "correctness kernels");
    unsigned long long *d_bad = nullptr, bad = 0;
    cuda_check(cudaMalloc(&d_bad, sizeof(bad)), "malloc mismatch");
    cuda_check(cudaMemset(d_bad, 0, sizeof(bad)), "clear mismatch");
    compare_bits<<<static_cast<unsigned>((out_elems + 255) / 256), 256>>>(
        reinterpret_cast<uint32_t *>(d_control), reinterpret_cast<uint32_t *>(d_sidecar), out_elems, d_bad);
    cuda_check(cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost), "copy mismatch");

    std::vector<float> ctl, side;
    ctl.reserve(reps);
    side.reserve(reps);
    for (int pair = 0; pair < 5; ++pair) {
        if (pair & 1) {
            side.push_back(time_kernel(slice1_sidecar_fp16, d_code, d_overlay, d_u, d_sidecar, sidecar_smem, reps));
            ctl.push_back(time_kernel(slice1_control_fp16, d_code, d_overlay, d_u, d_control, control_smem, reps));
        } else {
            ctl.push_back(time_kernel(slice1_control_fp16, d_code, d_overlay, d_u, d_control, control_smem, reps));
            side.push_back(time_kernel(slice1_sidecar_fp16, d_code, d_overlay, d_u, d_sidecar, sidecar_smem, reps));
        }
    }
    auto median = [](std::vector<float> x) { std::sort(x.begin(), x.end()); return x[x.size() / 2]; };
    const float ctl_ms = median(ctl), side_ms = median(side);
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, 0), "device props");

    std::printf("{\"device\":\"%s\",\"compute_capability\":\"%d.%d\",\"m\":%d,\"reps_per_pair\":%d,",
                prop.name, prop.major, prop.minor, M, reps);
    std::printf("\"control_ms\":%.6f,\"sidecar_ms\":%.6f,\"speedup_pct\":%.6f,\"bit_mismatches\":%llu,\"resources\":{",
                ctl_ms, side_ms, (ctl_ms / side_ms - 1.0f) * 100.0f, bad);
    resource_json("control_fp16", slice1_control_fp16, control_smem); std::printf(",");
    resource_json("control_fp32", slice1_control_fp32, control_smem); std::printf(",");
    resource_json("sidecar_fp16", slice1_sidecar_fp16, sidecar_smem); std::printf(",");
    resource_json("sidecar_fp32", slice1_sidecar_fp32, sidecar_smem);
    std::printf("}}\n");

    cudaFree(d_bad); cudaFree(d_sidecar); cudaFree(d_control); cudaFree(d_u); cudaFree(d_overlay); cudaFree(d_code);
    return bad == 0 ? 0 : 3;
}
