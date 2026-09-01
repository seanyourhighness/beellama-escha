#include "common.cuh"
#include "escha-moe.cuh"
#include "mmid.cuh"
#include "mma.cuh"
#include <cuda_pipeline.h>
#if !defined(GGML_USE_HIP)
#include <mma.h>
#endif

#include <cmath>
#include <cstdio>
#include <vector>

// Fused decode + routed matmul for Escha ESCHAM experts.
//
//   y = T128(T128(x * rin) @ decode(code)) * rout
//
// Weights decode independently -- weight[p] = lut[sum_j bit(payload, dep[p][j]) << j],
// no trellis state -- so the decode is a plain gather and the whole thing is one big
// embarrassingly parallel reduction.
//
// Split into three kernels because at batch 1 there is very little natural
// parallelism: 8 slots x OC/128 column groups is 32 blocks for gate/up, which leaves
// most of the GPU idle. Slicing the IC reduction across blocks and summing the
// partials afterwards is what fills it. The partials are summed in a fixed order
// rather than with atomics so results stay bit-reproducible run to run.
//
// There are two matmul kernels, picked by batch size:
//
//   escha_matmul_partial  one row per block, reduction sliced. Right when rows are
//                         scarce, but every row decodes the weights again.
//   escha_matmul_tiled    ESCHA_ROWS rows of ONE expert per block, so a decoded
//                         weight is reused across all of them. Needs the rows
//                         grouped by expert first, which is what mm_ids_helper
//                         already does for mul_mat_id.
//
// Both write the same partial layout and share escha_finalize.

#define ESCHA_TILE     16   // decode tile is 16x16
#define ESCHA_NT      128   // threads per block, one per output column
#define ESCHA_GROUPS  (ESCHA_NT/ESCHA_TILE)
#define ESCHA_MAX_W    24   // uint32 words per payload, 48 int16 at K=3
#define ESCHA_TARGET  512   // block count we try to reach by slicing the reduction
#define ESCHA_ROWS     16   // rows of one expert per block on the batched path
// dense reuses every decoded weight across this many rows. the decode, not bandwidth, is
// the limit (see the routed path's ablation), so this is the main prefill lever: 16 -> 64
// took a perplexity pass from 47.2 s to 18.9 s. kept separate from ESCHA_ROWS so the routed
// path stays exactly as measured.
//
// But acc[R] is per thread whatever the real row count is, so a big R costs occupancy at
// batch 1 and cripples generation (R=64 measured 1.84 t/s). Generation gets its own small
// instantiation instead -- the decode work per token is the same either way, so all that
// matters there is filling the device.
#define ESCHA_ROWS_DENSE      64
#define ESCHA_ROWS_DENSE_GEN   1
#define ESCHA_GEN_MAX_ROWS    16   // at or below this, use the generation instantiation
#define ESCHA_GEN_TARGET_MUL   5   // RTX 5090: split-K sweep 1/2/3/4/5/6/8 peaked at 5-8; 5 is stable
// prefill tile for the register-tiled kernel. BM*BN = TM*TN*NT, so these four fix the
// thread count too: NT = (BM/TM)*(BN/TN). BN stays 128 -- activation traffic is
// rows*IC*(OC/BN), so a narrower BN buys reuse with global bandwidth, which is a losing trade.
#define ESCHA_BM 128
#define ESCHA_BN 128
#define ESCHA_MMA_BM 128   // tensor-core prefill tile. Accumulators per thread are
#define ESCHA_MMA_BN 128   //   BM*BN/256, so BM drives register pressure and occupancy.
#define ESCHA_TM   8
#define ESCHA_TN   8
                                   //     prefill: at batch 1 there are only n_ocb blocks
                                   //     before slicing (136 for the FFN), which leaves an
                                   //     82-SM device idle. Swept 1/2/4/8/16/32 -- flat from
                                   //     10.3 to 11.3 tok/s, 4 marginally best.

// Escha codebook A, the one this checkpoint uses (its config leaves "codebook" unset,
// which eschamoe.py defaults to cbA / codebook_id 1). It is computed, not stored -- the
// same QTIP-family trick as 3INST but with its own multiplier and no addend. Recovered
// from escham_reconstruct_kernel<1, K> and checked against all 65536 entries.
// The fp16 add must stay in fp16 to match the table bit for bit.
static __device__ __forceinline__ float escha_codebook(uint32_t idx) {
    // (x & 0x8fff8fff) ^ 0x3b603b60 is one 3-input logic op; spelling it as lop3 stops
    // ptxas from splitting it across the two 16-bit lanes. immLut 0x6a = (a & b) ^ c.
    uint32_t x = idx*0xcbac1fedu;
    asm("lop3.b32 %0, %1, %2, %3, 0x6a;"
        : "=r"(x) : "r"(x), "n"(0x8fff8fffu), "n"(0x3b603b60u));
    __half2 h;
    memcpy(&h, &x, sizeof(h));
    return __half2float(__hadd(__low2half(h), __high2half(h)));
}

// as escha_codebook, but stopping at the half. The codebook's last operation is already
// an fp16 add, so the fp16 weight is exact -- only the ACTIVATIONS lose precision below.
static __device__ __forceinline__ half escha_codebook_h(uint32_t idx) {
    uint32_t x = idx*0xcbac1fedu;
    asm("lop3.b32 %0, %1, %2, %3, 0x6a;"
        : "=r"(x) : "r"(x), "n"(0x8fff8fffu), "n"(0x3b603b60u));
    __half2 h;
    memcpy(&h, &x, sizeof(h));
    return __hadd(__low2half(h), __high2half(h));
}

// in-place normalized Sylvester-Hadamard over each block of 128
static __device__ __forceinline__ void escha_hadamard_128(float * v, int n, int tid, int nt) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int idx = tid; idx < (n/128)*64; idx += nt) {
            const int blk = idx / 64;
            const int j   = idx % 64;
            const int i   = (j / len)*(2*len) + (j % len);

            float * b = v + blk*128 + i;
            const float a0 = b[0];
            const float a1 = b[len];

            b[0]   = a0 + a1;
            b[len] = a0 - a1;
        }
        __syncthreads();
    }

    const float scale = rsqrtf(128.0f);
    for (int i = tid; i < n; i += nt) {
        v[i] *= scale;
    }
    __syncthreads();
}

// u[row] = T128(x[row] * rin[expert]) -- hoisted out of the matmul so the column
// blocks do not each redo it
static __global__ void escha_rotate_in(
        const half    * __restrict__ rin,
        const float   * __restrict__ x,
        const int32_t * __restrict__ ids,
        float         * __restrict__ u,
        const int IC, const int n_x, const int n_ids,
        const int64_t nb_x1, const int64_t nb_x2,
        const int64_t nb_i0, const int64_t nb_i1) {
    extern __shared__ float s_u[];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int it  = row / n_ids;
    const int is  = row % n_ids;

    const int32_t e = *(const int32_t *)((const char *) ids + is*nb_i0 + it*nb_i1);

    const half  * rin_e = rin + (int64_t) e*IC;
    const float * x_row = (const float *)((const char *) x + (int64_t)(is % n_x)*nb_x1 + it*nb_x2);

    for (int i = tid; i < IC; i += blockDim.x) {
        s_u[i] = x_row[i]*__half2float(rin_e[i]);
    }
    __syncthreads();

    escha_hadamard_128(s_u, IC, tid, blockDim.x);

    float * dst = u + (int64_t) row*IC;
    for (int i = tid; i < IC; i += blockDim.x) {
        dst[i] = s_u[i];
    }
}

// Forward declaration: the opt-in WMMA experiment appears before the shared dense
// decoder helper so it can stay isolated from the established kernels below.
static __device__ __forceinline__ int escha_dep_pi(int r);

// Materialize one exact fp16 projection for cuBLAS prefill.  The packed tile at
// [ti, band] decodes to W[k, out]; writing k*OC + out makes the buffer a
// column-major [OC, IC] matrix without a separate transpose.  Unlike the fused
// row-tiled kernels, this pays the Escha decode once per projection rather than
// once for every group of prompt rows.
template <int K>
static __global__ void escha_dequant_dense_f16(
        const int16_t * __restrict__ code,
        half          * __restrict__ w,
        const int IC, const int OC) {
    constexpr int TILE = ESCHA_TILE;
    constexpr int NWD  = 8*K;
    constexpr int NB   = 32*NWD;

    const int ti   = blockIdx.y;
    const int band = blockIdx.x;
    const int j    = threadIdx.x;
    const int c    = j/TILE;
    const int r    = j%TILE;
    const int nct  = OC/TILE;

    const uint32_t * pay = (const uint32_t *)
        (code + (int64_t) (ti*nct + band)*(TILE*K));
    int sp = ((32 - K) - K*(escha_dep_pi(r) + 32*c + 4*(c >> 3))) % NB;
    if (sp < 0) {
        sp += NB;
    }
    const int g0 = sp >> 5;
    const int w0 = g0 ? (NWD - g0) : 0;
    const int w1 = w0 ? (w0 - 1) : (NWD - 1);
    w[(int64_t) (ti*TILE + r)*OC + band*TILE + c] =
        escha_codebook_h(__funnelshift_r(pay[w0], pay[w1], sp & 31) & 0xffffu);
}

template <int K>
static __global__ void escha_capture_tile_fragments(
        const int16_t * __restrict__ code,
        uint32_t      * __restrict__ fragments) {
#ifdef TURING_MMA_AVAILABLE
    constexpr int TILE = ESCHA_TILE;
    constexpr int NWD  = 8*K;
    constexpr int NB   = 32*NWD;

    __shared__ half s_w[TILE*TILE]; // [out][k], the MMA B staging order
    const int lane = threadIdx.x;
    const uint32_t * payload = (const uint32_t *) code;
    for (int j = lane; j < TILE*TILE; j += WARP_SIZE) {
        const int c = j/TILE;
        const int r = j%TILE;
        int sp = ((32 - K) - K*(escha_dep_pi(r) + 32*c + 4*(c >> 3))) % NB;
        if (sp < 0) {
            sp += NB;
        }
        const int g0 = sp >> 5;
        const int w0 = g0 ? (NWD - g0) : 0;
        const int w1 = w0 ? (w0 - 1) : (NWD - 1);
        s_w[c*TILE + r] = escha_codebook_h(
            __funnelshift_r(payload[w0], payload[w1], sp & 31) & 0xffffu);
    }
    __syncwarp();

    typedef ggml_cuda_mma::tile<8, 8, half2> tile_b;
    const half2 * sw2 = (const half2 *) s_w;
    tile_b b0;
    tile_b b1;
    ggml_cuda_mma::load_ldmatrix(b0, sw2 + 0*8*8, 8);
    ggml_cuda_mma::load_ldmatrix(b1, sw2 + 1*8*8, 8);
    for (int reg = 0; reg < tile_b::ne; ++reg) {
        memcpy(fragments + (0*WARP_SIZE + lane)*tile_b::ne + reg, &b0.x[reg], sizeof(uint32_t));
        memcpy(fragments + (1*WARP_SIZE + lane)*tile_b::ne + reg, &b1.x[reg], sizeof(uint32_t));
    }
#else
    GGML_UNUSED_VARS(code, fragments);
    NO_DEVICE_CODE;
#endif
}

// Blackwell WMMA prefill experiment. Unlike the ldmatrix kernel above, this uses the
// CUDA WMMA API and keeps one warp's input tile in shared memory while it decodes four
// consecutive 16-column bands. Every reduction slice writes its own partial buffer, so
// final accumulation remains deterministic and does not require atomic adds.
//
// This is deliberately opt-in. The normal dispatch remains the existing kernel until
// the candidate has passed prefix parity and long-prompt smoke tests on Blackwell.
template <int K, int NJ, int NWARPS, int BB>
static __global__ void __launch_bounds__(NWARPS*WARP_SIZE, 1) escha_matmul_dense_wmma_bw(
        const int16_t * __restrict__ code,
        const half    * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    using namespace nvcuda;

    constexpr int TILE = ESCHA_TILE;
    constexpr int NWD  = 8*K;
    constexpr int NB   = 32*NWD;
    constexpr int NG   = NJ/TILE;

    static_assert(NJ % TILE == 0, "escha: WMMA rows must be a multiple of 16");

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int band0 = blockIdx.x*BB;
    const int row0  = blockIdx.y*NJ;
    const int sl    = blockIdx.z;
    const int nit   = IC/TILE;
    const int nct   = OC/TILE;
    const int lo    = (int) (((int64_t) nit*sl)/n_slices);
    const int hi    = (int) (((int64_t) nit*(sl + 1))/n_slices);

    __shared__ half  s_w[NWARPS*TILE*TILE];
    __shared__ half  s_u[NWARPS*NJ*TILE];
    __shared__ float s_o[NWARPS*TILE*TILE];

    half  * w_w = s_w + warp*TILE*TILE;
    half  * u_w = s_u + warp*NJ*TILE;
    float * o_w = s_o + warp*TILE*TILE;

    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc[BB][NG];
#pragma unroll
    for (int bb = 0; bb < BB; ++bb) {
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            wmma::fill_fragment(acc[bb][g], 0.0f);
        }
    }

    for (int ti = lo; ti < hi; ++ti) {
        // B is input transposed, in column-major form: B[k, row] = u[row, k].
        for (int j = lane; j < NJ*TILE; j += WARP_SIZE) {
            const int row = row0 + j/TILE;
            const int r   = j % TILE;
            u_w[j] = row < n_rows ? u[(int64_t) row*IC + ti*TILE + r] : __float2half(0.0f);
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::col_major> b[NG];
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            wmma::load_matrix_sync(b[g], u_w + g*TILE*TILE, TILE);
        }

#pragma unroll
        for (int bb = 0; bb < BB; ++bb) {
            const int band = band0 + bb;
            const uint32_t * pay = (const uint32_t *) (code + (int64_t) (ti*nct + band)*(TILE*K));

            // A is W transposed, in column-major form: A[out, k] = W[k, out].
            for (int j = lane; j < TILE*TILE; j += WARP_SIZE) {
                const int c = j/TILE;
                const int r = j % TILE;
                int sp = ((32 - K) - K*(escha_dep_pi(r) + 32*c + 4*(c >> 3))) % NB;
                if (sp < 0) {
                    sp += NB;
                }
                const int g0 = sp >> 5;
                const int w0 = g0 ? (NWD - g0) : 0;
                const int w1 = w0 ? (w0 - 1) : (NWD - 1);
                // wmma::matrix_a is column-major here: element [out=c, k=r]
                // lives at r*TILE+c.  The previous c*TILE+r write transposed
                // the decoded 16x16 tile and made the candidate numerically wrong.
                w_w[r*TILE + c] = escha_codebook_h(__funnelshift_r(pay[w0], pay[w1], sp & 31) & 0xffffu);
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::col_major> a;
            wmma::load_matrix_sync(a, w_w, TILE);
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                wmma::mma_sync(acc[bb][g], a, b[g], acc[bb][g]);
            }
            __syncwarp();
        }
    }

#pragma unroll
    for (int bb = 0; bb < BB; ++bb) {
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            wmma::store_matrix_sync(o_w, acc[bb][g], TILE, wmma::mem_col_major);
            __syncwarp();
            for (int j = lane; j < TILE*TILE; j += WARP_SIZE) {
                const int row = row0 + g*TILE + j/TILE;
                const int out = j % TILE;
                if (row < n_rows) {
                    partial[((int64_t) sl*n_rows + row)*OC + (band0 + bb)*TILE + out] = o_w[j];
                }
            }
            __syncwarp();
        }
    }
#else
    GGML_UNUSED_VARS(code, u, partial, IC, OC, n_rows, n_slices);
    NO_DEVICE_CODE;
#endif
}

// partial[slice][row][c] = sum over this slice's input tiles of u . decode(code)
template <int K>
static __global__ void escha_matmul_partial(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        const int32_t * __restrict__ ids,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_ids, const int n_rows, const int n_slices,
        const int64_t nb_i0, const int64_t nb_i1) {
    extern __shared__ char s_raw[];

    // dep is stored transposed and packed two entries per word: [b/2][r][cc]. that makes
    // the 16 threads of a group read 16 consecutive words, which is conflict-free -- the
    // natural [p][b] layout puts them 32 bytes apart and costs an 8-way bank conflict
    uint32_t * s_dep = (uint32_t *) s_raw;                 // [8][16][16]
    uint32_t * s_pay = s_dep + 8*256;                      // [ESCHA_GROUPS][ESCHA_MAX_W]

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid   = threadIdx.x;
    const int row   = blockIdx.x;
    const int ocb   = blockIdx.y;
    const int slice = blockIdx.z;

    const int it = row / n_ids;
    const int is = row % n_ids;

    const int32_t e = *(const int32_t *)((const char *) ids + is*nb_i0 + it*nb_i1);
    const int16_t * code_e = code + (int64_t) e*nit*nct*(16*K);

    for (int j = tid; j < 8*256; j += ESCHA_NT) {
        const int b2 = j / 256;
        const int p  = j % 256;
        s_dep[j] = (uint32_t) (uint16_t) dep[p*16 + 2*b2]
                 | ((uint32_t) (uint16_t) dep[p*16 + 2*b2 + 1] << 16);
    }
    __syncthreads();

    const int grp = tid / ESCHA_TILE;
    const int cc  = tid % ESCHA_TILE;
    const int tj  = ocb*ESCHA_GROUPS + grp;

    const int per   = nit/n_slices;
    const int ti0   = slice*per;
    const float * u_row = u + (int64_t) row*IC;

    uint32_t * pay = s_pay + grp*ESCHA_MAX_W;
    float sum = 0.0f;

    for (int ti = ti0; ti < ti0 + per; ++ti) {
        const uint32_t * src = (const uint32_t *)(code_e + (int64_t)(ti*nct + tj)*(16*K));
        for (int w = cc; w < n_wd; w += ESCHA_TILE) {
            pay[w] = src[w];
        }
        __syncwarp();

        const float * uu = u_row + ti*ESCHA_TILE;

        #pragma unroll 4
        for (int r = 0; r < ESCHA_TILE; ++r) {
            const uint32_t * d = s_dep + r*ESCHA_TILE + cc;

            uint32_t idx = 0;
            #pragma unroll
            for (int b2 = 0; b2 < 8; ++b2) {
                const uint32_t dd = d[b2*256];
                const int d0 = dd & 0xffff;
                const int d1 = dd >> 16;

                idx |= ((pay[d0 >> 5] >> (d0 & 31)) & 1u) << (2*b2);
                idx |= ((pay[d1 >> 5] >> (d1 & 31)) & 1u) << (2*b2 + 1);
            }

            sum += uu[r]*escha_codebook(idx);
        }
        __syncwarp();
    }

    partial[((int64_t) slice*n_rows + row)*OC + ocb*ESCHA_NT + tid] = sum;
}

// one work item per block of the tiled kernel: up to ESCHA_ROWS compact rows of expert e.
// order does not matter -- items write disjoint output rows -- so an atomic counter is enough
static __global__ void escha_build_work(
        const int32_t * __restrict__ bounds,
        int4          * __restrict__ work,
        int32_t       * __restrict__ n_work,
        const int n_expert) {
    for (int e = blockIdx.x*blockDim.x + threadIdx.x; e < n_expert; e += gridDim.x*blockDim.x) {
        const int lo = bounds[e];
        const int hi = bounds[e + 1];
        for (int s = lo; s < hi; s += ESCHA_ROWS) {
            work[atomicAdd(n_work, 1)] = make_int4(e, s, min(ESCHA_ROWS, hi - s), 0);
        }
    }
}

// partial[row][c] = u[row] . decode(code), for R rows sharing one expert
template <int K, int R>
static __global__ void escha_matmul_tiled(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        const int32_t * __restrict__ ids_dst,
        const int4    * __restrict__ work,
        const int32_t * __restrict__ n_work,
        float         * __restrict__ partial,
        const int IC, const int OC) {
    extern __shared__ char s_raw[];

    uint32_t * s_dep = (uint32_t *) s_raw;                             // [8][16][16]
    uint32_t * s_pay = s_dep + 8*256;                                  // [ESCHA_GROUPS][ESCHA_MAX_W]
    float    * s_u   = (float *)(s_pay + ESCHA_GROUPS*ESCHA_MAX_W);    // [R][16]

    // the grid is sized to an upper bound, so the tail blocks have nothing to do.
    // uniform across the block, so the syncs below are still safe
    if (blockIdx.x >= (unsigned) *n_work) {
        return;
    }

    const int4 w = work[blockIdx.x];
    const int e     = w.x;
    const int start = w.y;
    const int nrow  = w.z;

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid = threadIdx.x;

    for (int j = tid; j < 8*256; j += ESCHA_NT) {
        const int b2 = j / 256;
        const int p  = j % 256;
        s_dep[j] = (uint32_t) (uint16_t) dep[p*16 + 2*b2]
                 | ((uint32_t) (uint16_t) dep[p*16 + 2*b2 + 1] << 16);
    }

    const int grp = tid / ESCHA_TILE;
    const int cc  = tid % ESCHA_TILE;
    const int tj  = blockIdx.y*ESCHA_GROUPS + grp;

    const int16_t * code_e = code + (int64_t) e*nit*nct*(16*K);
    uint32_t * pay = s_pay + grp*ESCHA_MAX_W;

    float acc[R];
#pragma unroll
    for (int m = 0; m < R; ++m) {
        acc[m] = 0.0f;
    }
    __syncthreads();

    for (int ti = 0; ti < nit; ++ti) {
        // the whole block cooperates on one 16-wide slice of u per row, then every
        // thread reads all of it -- 16 threads of a group hit the same address, so
        // the reads broadcast instead of conflicting
        for (int j = tid; j < R*ESCHA_TILE; j += ESCHA_NT) {
            const int m = j / ESCHA_TILE;
            const int r = j % ESCHA_TILE;
            s_u[j] = m < nrow ? u[(int64_t) ids_dst[start + m]*IC + ti*ESCHA_TILE + r] : 0.0f;
        }

        const uint32_t * src = (const uint32_t *)(code_e + (int64_t)(ti*nct + tj)*(16*K));
        for (int wd = cc; wd < n_wd; wd += ESCHA_TILE) {
            pay[wd] = src[wd];
        }
        __syncthreads();

#pragma unroll 4
        for (int r = 0; r < ESCHA_TILE; ++r) {
            const uint32_t * d = s_dep + r*ESCHA_TILE + cc;

            uint32_t idx = 0;
#pragma unroll
            for (int b2 = 0; b2 < 8; ++b2) {
                const uint32_t dd = d[b2*256];
                const int d0 = dd & 0xffff;
                const int d1 = dd >> 16;

                idx |= ((pay[d0 >> 5] >> (d0 & 31)) & 1u) << (2*b2);
                idx |= ((pay[d1 >> 5] >> (d1 & 31)) & 1u) << (2*b2 + 1);
            }

            const float wv = escha_codebook(idx);
#pragma unroll
            for (int m = 0; m < R; ++m) {
                acc[m] += s_u[m*ESCHA_TILE + r]*wv;
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < nrow; ++m) {
        partial[(int64_t) ids_dst[start + m]*OC + blockIdx.y*ESCHA_NT + tid] = acc[m];
    }
}

// sum the slices, rotate the 128-column group, scale by rout
static __global__ void escha_finalize(
        const half    * __restrict__ rout,
        const int32_t * __restrict__ ids,
        const float   * __restrict__ partial,
        float         * __restrict__ dst,
        const int OC, const int n_ids, const int n_rows, const int n_slices,
        const int64_t nb_i0, const int64_t nb_i1,
        const int64_t nb_d1, const int64_t nb_d2) {
    __shared__ float s_acc[ESCHA_NT];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int ocb = blockIdx.y;

    const int it = row / n_ids;
    const int is = row % n_ids;

    const int32_t e = *(const int32_t *)((const char *) ids + is*nb_i0 + it*nb_i1);
    const int c = ocb*ESCHA_NT + tid;

    float sum = 0.0f;
    for (int s = 0; s < n_slices; ++s) {
        sum += partial[((int64_t) s*n_rows + row)*OC + c];
    }
    s_acc[tid] = sum;
    __syncthreads();

    escha_hadamard_128(s_acc, ESCHA_NT, tid, ESCHA_NT);

    float * dst_row = (float *)((char *) dst + is*nb_d1 + it*nb_d2);
    dst_row[c] = s_acc[tid]*__half2float(rout[(int64_t) e*OC + c]);
}

void ggml_cuda_op_escha_moe(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code = dst->src[0];
    const ggml_tensor * rin  = dst->src[1];
    const ggml_tensor * rout = dst->src[2];
    const ggml_tensor * lut  = dst->src[3];
    const ggml_tensor * dep  = dst->src[4];
    const ggml_tensor * x    = dst->src[5];
    const ggml_tensor * ids  = dst->src[6];

    GGML_ASSERT(code->type == GGML_TYPE_I16 && dep->type == GGML_TYPE_I16);
    GGML_ASSERT(rin->type == GGML_TYPE_F16 && rout->type == GGML_TYPE_F16 && lut->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);

    const int K   = code->ne[0]/16;
    const int OC  = code->ne[1]*16;
    const int IC  = code->ne[2]*16;
    const int nit = IC/ESCHA_TILE;

    const int n_expert = code->ne[3];
    const int n_ids    = ids->ne[0];
    const int n_tokens = ids->ne[1];
    const int n_rows   = n_ids*n_tokens;
    const int n_ocb    = OC/ESCHA_NT;

    // reuse only pays once a block can expect a decent share of ESCHA_ROWS rows for
    // its expert. below that the sliced kernel wins, and at batch 1 it is the only
    // one that fills the device at all
    const bool tiled = n_rows >= 8*n_expert;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> u_buf(ctx.pool(), (size_t) n_rows*IC);

    escha_rotate_in<<<n_rows, 256, IC*sizeof(float), stream>>>(
        (const half *) rin->data, (const float *) x->data, (const int32_t *) ids->data,
        u_buf.get(), IC, (int) x->ne[1], n_ids,
        x->nb[1], x->nb[2], ids->nb[0], ids->nb[1]);

    const size_t smem_dep = 8*256*sizeof(uint32_t) + ESCHA_GROUPS*ESCHA_MAX_W*sizeof(uint32_t);

    if (tiled) {
        // mm_ids_helper assumes a token uses an expert at most once, which top-k routing
        // guarantees. duplicates would desync its offsets and write out of bounds
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));

        ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
        ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
        ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), n_expert + 1);

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            n_expert, n_tokens, n_ids, (int) x->ne[1], (int) (ids->nb[1]/ggml_element_size(ids)), 1,
            /*write_inverse =*/ false, stream);
        CUDA_CHECK(cudaGetLastError());

        // an expert can end with a part-full chunk, so the item count is bounded but not known
        const int n_work_max = (n_rows + ESCHA_ROWS - 1)/ESCHA_ROWS + n_expert;

        ggml_cuda_pool_alloc<int4>    work(ctx.pool(), n_work_max);
        ggml_cuda_pool_alloc<int32_t> n_work(ctx.pool(), 1);

        CUDA_CHECK(cudaMemsetAsync(n_work.get(), 0, sizeof(int32_t), stream));
        escha_build_work<<<1, 256, 0, stream>>>(bounds.get(), work.get(), n_work.get(), n_expert);

        ggml_cuda_pool_alloc<float> p_buf(ctx.pool(), (size_t) n_rows*OC);

        const size_t smem = smem_dep + ESCHA_ROWS*ESCHA_TILE*sizeof(float);

        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_work_max, n_ocb), ESCHA_NT, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                u_buf.get(), ids_dst.get(), work.get(), n_work.get(), p_buf.get(), IC, OC);
        };

        switch (K) {
            case 2: launch(escha_matmul_tiled<2, ESCHA_ROWS>); break;
            case 3: launch(escha_matmul_tiled<3, ESCHA_ROWS>); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }

        escha_finalize<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
            (const half *) rout->data, (const int32_t *) ids->data, p_buf.get(), (float *) dst->data,
            OC, n_ids, n_rows, 1, ids->nb[0], ids->nb[1], dst->nb[1], dst->nb[2]);
        return;
    }

    // slice the reduction until the launch is wide enough to fill the device, but only
    // by factors that divide nit evenly
    int n_slices = 1;
    while (n_rows*n_ocb*n_slices*2 <= ESCHA_TARGET && nit % (n_slices*2) == 0) {
        n_slices *= 2;
    }

    ggml_cuda_pool_alloc<float> p_buf(ctx.pool(), (size_t) n_slices*n_rows*OC);

    const dim3 grid(n_rows, n_ocb, n_slices);

    auto launch = [&](auto kernel) {
        kernel<<<grid, ESCHA_NT, smem_dep, stream>>>(
            (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
            u_buf.get(), (const int32_t *) ids->data, p_buf.get(),
            IC, OC, n_ids, n_rows, n_slices, ids->nb[0], ids->nb[1]);
    };

    switch (K) {
        case 2: launch(escha_matmul_partial<2>); break;
        case 3: launch(escha_matmul_partial<3>); break;
        default: GGML_ABORT("escha: unsupported K=%d", K);
    }

    escha_finalize<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
        (const half *) rout->data, (const int32_t *) ids->data, p_buf.get(), (float *) dst->data,
        OC, n_ids, n_rows, n_slices, ids->nb[0], ids->nb[1], dst->nb[1], dst->nb[2]);
}

// ===========================================================================
// dense escha (ggml_escha_mul_mat)
//
// Same codec and same rotations as the routed path above, minus the routing: one weight
// matrix, every row goes through it. That removes the ids indirection, mm_ids_helper and
// the work list -- a block owns R consecutive rows outright. The IC reduction is still
// sliced across blocks, because at batch 1 a single row would otherwise leave the device
// mostly idle.
// ===========================================================================

// u[row] = T128(x[row] * rin)
//
// Staged through shared memory in fixed chunks rather than all of IC at once: the dense
// projections go up to IC = 17408 (mlp.down), and 17408 floats is 68 KB, well past the
// 48 KB a block gets. The rotation is independent per 128-block, so any chunk that is a
// multiple of 128 splits it exactly.
#define ESCHA_ROT_CHUNK 2048   // 8 KB of shared memory

// U is float for the scalar paths and half for the tensor-core path. Emitting half here
// rather than converting during staging is bit-identical -- the same __float2half, moved
// earlier -- and it is what lets cp.async copy activations straight into shared, since
// cp.async moves bytes verbatim and cannot convert.
template <typename U>
static __global__ void escha_rotate_in_dense(
        const half  * __restrict__ rin,
        const float * __restrict__ x,
        U           * __restrict__ u,
        const int IC, const int ne1,
        const int64_t nb_x1, const int64_t nb_x2) {
    __shared__ float s_u[ESCHA_ROT_CHUNK];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    const float * x_row = (const float *)((const char *) x + (int64_t)(row % ne1)*nb_x1
                                                           + (int64_t)(row / ne1)*nb_x2);
    U * dst = u + (int64_t) row*IC;

    for (int off = 0; off < IC; off += ESCHA_ROT_CHUNK) {
        const int n = min(ESCHA_ROT_CHUNK, IC - off);

        for (int i = tid; i < n; i += blockDim.x) {
            s_u[i] = x_row[off + i]*__half2float(rin[off + i]);
        }
        __syncthreads();

        escha_hadamard_128(s_u, n, tid, blockDim.x);

        for (int i = tid; i < n; i += blockDim.x) {
            if constexpr (sizeof(U) == sizeof(half)) {
                dst[off + i] = __float2half(s_u[i]);
            } else {
                dst[off + i] = s_u[i];
            }
        }
        __syncthreads();
    }
}

// escha's dep table is computable, so the dense kernel never reads it.
//
// The 16 bits that form a weight's codebook index are always 16 cyclically-consecutive
// positions of a bit-stream over the tile payload, where the stream visits 32-bit word 0
// first and then walks the words downwards (0, NW-1, NW-2, ...). Only the start position
// varies, and it is affine in the tile row and column:
//
//   pi(r) = (r&1) | ((r>>3)&1)<<1 | ((r>>1)&3)<<3      // bit 2 is left free for c
//   t     = pi(r) + 32*c + 4*(c>>3)
//   s     = ((32-K) - K*t) mod 256K
//
// Verified exact against both shipped tables, all 4096 entries (dep3.py). This turns
// 8 shared dep reads + 16 payload reads + ~48 bit ops per weight into two reads and a
// funnel shift, and drops the 8 KB per-block dep table that made batch 1 setup-bound.
__device__ __forceinline__ int escha_dep_pi(int r) {
    return (r & 1) | (((r >> 3) & 1) << 1) | (((r >> 1) & 3) << 3);
}

// Register-tiled variant of the dense kernel, for prefill.
//
// The column-per-thread kernel below couples decode reuse to one thread's registers: it
// accumulates R rows in acc[R], so reusing a decode more means more registers in the SAME
// thread. At R=64 that is 255 registers with spill, 2 blocks/SM, ~17% occupancy -- and it
// is why prefill decodes at ~60 G/s while the batch-1 path manages ~372 G/s.
//
// Here the decoded weights go to SHARED memory instead, so every row group in the block
// reuses them. Reuse becomes BM (rows per BLOCK) while registers stay TM*TN (the thread's
// own output tile), which decouples the two:
//
//     BM * BN = (TM * TN) * NT
//
// BN is held at 128 on purpose: total activation traffic is rows * IC * (OC/BN), so
// narrowing BN to buy reuse would multiply global reads instead (that mistake was caught
// on paper, not in silicon -- BN=16 would have cost 8x the activation bandwidth).
//
// TN must be > 1. With one column per thread every shared read feeds exactly one MAC and
// shared bandwidth becomes the new ceiling; a TMxTN tile does TM*TN MACs per TM+TN reads.
template <int K, int BM, int BN, int TM, int TN>
static __global__ void escha_matmul_dense_tiled(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
    constexpr int NT  = (BM/TM)*(BN/TN);   // threads per block
    constexpr int NTJ = BN/ESCHA_TILE;     // output tiles covered
    constexpr int NCX = BN/TN;             // threads across the column axis

    extern __shared__ char s_raw[];
    uint32_t * s_pay = (uint32_t *) s_raw;                          // [NTJ][ESCHA_MAX_W]
    float    * s_w   = (float *)(s_pay + NTJ*ESCHA_MAX_W);          // [16][BN]
    float    * s_u   = s_w + ESCHA_TILE*BN;                         // [2][16][BM] transposed

    GGML_UNUSED(lut);
    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid  = threadIdx.x;
    const int cx   = tid % NCX;            // this thread's column strip
    const int ry   = tid / NCX;            // this thread's row strip
    const int row0 = blockIdx.x*BM;
    const int oc0  = blockIdx.y*BN;

    const int sl = blockIdx.z;
    const int lo = (int) (((int64_t) nit*sl)/n_slices);
    const int hi = (int) (((int64_t) nit*(sl + 1))/n_slices);

    float acc[TM*TN];
#pragma unroll
    for (int i = 0; i < TM*TN; ++i) {
        acc[i] = 0.0f;
    }

    // this thread's payload slot, fixed for the whole loop. One word per thread only:
    static_assert(NTJ*((16*K)/2) <= NT, "escha: payload needs more than one word per thread");
    const bool has_pay = tid < NTJ*n_wd;
    const int  pt = tid/n_wd, pw = tid % n_wd;
    uint32_t   ppre = 0;
    if (has_pay && lo < hi) {
        ppre = ((const uint32_t *)(code + (int64_t)(lo*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
    }

    // stage tile lo's activations into buffer 0 before the loop, so that inside the loop
    // the fetch for ti+1 can be issued into the OTHER buffer and overlap this tile's work
    if (lo < hi) {
        for (int j = tid; j < BM*ESCHA_TILE; j += NT) {
            const int m = j / ESCHA_TILE, r = j % ESCHA_TILE;
            const int row = row0 + m;
            s_u[r*BM + m] = row < n_rows ? u[(int64_t) row*IC + lo*ESCHA_TILE + r] : 0.0f;
        }
    }

    for (int ti = lo; ti < hi; ++ti) {
        float * su_cur = s_u + (((ti - lo) & 1)      )*(ESCHA_TILE*BM);
        float * su_nxt = s_u + (((ti - lo) & 1) ^ 1  )*(ESCHA_TILE*BM);
        // publish the payload fetched last round, then issue the next fetch before the
        // barrier, so the global latency overlaps the decode instead of stalling every warp
        if (has_pay) {
            s_pay[pt*ESCHA_MAX_W + pw] = ppre;
        }
        if (has_pay && ti + 1 < hi) {
            ppre = ((const uint32_t *)(code + (int64_t)((ti + 1)*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
        }
        __syncthreads();

        // next tile's activations go to the other buffer: no barrier separates them from
        // the reads of su_cur below, and the barrier at the top of the next iteration is
        // what makes them visible
        if (ti + 1 < hi) {
            for (int j = tid; j < BM*ESCHA_TILE; j += NT) {
                const int m = j / ESCHA_TILE, r = j % ESCHA_TILE;
                const int row = row0 + m;
                su_nxt[r*BM + m] = row < n_rows ? u[(int64_t) row*IC + (ti + 1)*ESCHA_TILE + r] : 0.0f;
            }
        }

        // Decode this input tile's 16 x BN weights once, for the whole block.
        //
        // The dependency table is the explicit form of this cyclic bit mapping.
        // Do not gather its sixteen entries here: the codebook window starts at
        // `sp`, then walks the payload words 0, NW-1, NW-2, ... (tail-biting).
        // Constructing the adjacent pair directly is exactly the same mapping and
        // removes sixteen shared/global dependency loads and bit tests per weight.
        for (int j = tid; j < ESCHA_TILE*BN; j += NT) {
            const int r = j / BN, c = j % BN;
            const int tile_j = oc0/ESCHA_TILE + c/ESCHA_TILE;
            const uint32_t * payload = (const uint32_t *)
                (code + (int64_t) (ti*nct + tile_j)*(16*K));
            constexpr int NW = 8*K;
            constexpr int NB = 32*NW;
            int sp = ((32 - K) - K*(escha_dep_pi(r) + 32*(c % ESCHA_TILE)
                                     + 4*((c % ESCHA_TILE) >> 3))) % NB;
            if (sp < 0) {
                sp += NB;
            }
            const int g0 = sp >> 5;
            const int w0 = g0 ? (NW - g0) : 0;
            const uint2 pair = make_uint2(payload[w0 ? w0 - 1 : NW - 1], payload[w0]);
            const uint32_t idx = __funnelshift_r(pair.y, pair.x, sp & 31) & 0xffffu;
            s_w[r*BN + c] = escha_codebook(idx);
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < ESCHA_TILE; ++r) {
            float a[TM], b[TN];
#pragma unroll
            for (int m = 0; m < TM; ++m) {
                a[m] = su_cur[r*BM + ry*TM + m];
            }
#pragma unroll
            for (int n = 0; n < TN; ++n) {
                b[n] = s_w[r*BN + cx*TN + n];
            }
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m*TN + n] += a[m]*b[n];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int row = row0 + ry*TM + m;
        if (row < n_rows) {
#pragma unroll
            for (int n = 0; n < TN; ++n) {
                partial[((int64_t) sl*n_rows + row)*OC + oc0 + cx*TN + n] = acc[m*TN + n];
            }
        }
    }
}


// Prefill on tensor cores. Same decode as escha_matmul_dense_tiled, but the GEMM runs on
// m16n8k16 HMMA with fp32 accumulate, which on GA102 is 71 TFLOPS against 35.6 for FP32 FMA.
//
// Two layout changes fall out of the fragment shapes, and both are cheap:
//   s_u becomes [m][k] (was [k][m]) -- which makes staging a straight contiguous copy
//   s_w becomes [n][k] (was [k][n]) -- B is the ".col" operand of mma.row.col
//
// The weights stay exact (escha_codebook_h). The ACTIVATIONS are rounded to fp16, which is
// what escha's own runtime does, and costs rel_rms ~2.1e-4 against the fp32 reference.
template <int K, int BM, int BN, bool FP16_ACC = false>
static __global__ void __launch_bounds__(256, 1) escha_matmul_dense_tiled_mma(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const half    * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
#ifdef TURING_MMA_AVAILABLE
    constexpr int NT   = 256;
    constexpr int NW   = NT/32;          // warps
    constexpr int WN   = 2;              // warps across the column axis
    constexpr int WM   = NW/WN;          // warps down the row axis
    constexpr int MT   = BM/16/WM;       // 16-row accumulator tiles per warp
    constexpr int NTT  = BN/8/WN;        // 8-col accumulator tiles per warp
    constexpr int NTJ  = BN/ESCHA_TILE;  // output tiles whose payload this block holds

    extern __shared__ char s_raw[];
    uint2    * s_pay = (uint2 *) s_raw;                               // [NTJ][ESCHA_MAX_W] pairs
    half     * s_u   = (half *)(s_pay + NTJ*ESCHA_MAX_W);             // [2][BM][16]
    half     * s_w   = s_u + 2*BM*ESCHA_TILE;                         // [BN][16]

    GGML_UNUSED(lut);
    GGML_UNUSED(dep);

    const int NWD = 8*K;
    const int NB  = 32*NWD;

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int lane = threadIdx.x;          // must stay the lane: mma.cuh indexes on it
    const int warp = threadIdx.y;
    const int tid  = warp*32 + lane;        // flat id, for the layout-agnostic staging loops
    const int row0 = blockIdx.x*BM;
    const int oc0  = blockIdx.y*BN;

    const int sl = blockIdx.z;
    const int lo = (int) (((int64_t) nit*sl)/n_slices);
    const int hi = (int) (((int64_t) nit*(sl + 1))/n_slices);

    const int wm   = warp / WN;
    const int wn   = warp % WN;

    constexpr int DPT = (ESCHA_TILE*BN)/NT;   // weights this thread decodes per tile
    static_assert(NT % ESCHA_TILE == 0,        "escha: r would not be thread-invariant");
    static_assert((ESCHA_TILE*BN) % NT == 0,   "escha: ragged decode assignment");
    static_assert(NT/ESCHA_TILE <= ESCHA_TILE, "escha: ccl would not be thread-invariant");

    // cp.async moves 16 bytes = 8 halves per thread; BM*ESCHA_TILE halves is exactly
    // NT*8 at BM=128/NT=256, so every thread issues one copy and none loops.
    constexpr int CPB = 16;                              // bytes per thread per tile
    static_assert(BM*ESCHA_TILE*sizeof(half) == NT*CPB,  "escha: activation copy is ragged");
    const int cp_m  = tid / (ESCHA_TILE*sizeof(half)/CPB);   // row this thread copies into
    const int cp_h  = (tid % (ESCHA_TILE*sizeof(half)/CPB))*(CPB/sizeof(half));

    const int dr   = tid % ESCHA_TILE;         // this thread's r, every k, every tile
    const int dccl = tid / ESCHA_TILE;         // and its column within the 16-wide tile
    int dsp = ((32 - K) - K*(escha_dep_pi(dr) + 32*dccl + 4*(dccl >> 3))) % NB;
    if (dsp < 0) {
        dsp += NB;
    }
    const int dg0 = dsp >> 5;
    const int dw0 = dg0 ? (NWD - dg0) : 0;
    const int dw1 = dw0 ? (dw0 - 1)   : (NWD - 1);
    const int dsh = dsp & 31;

    typedef ggml_cuda_mma::tile<16, 8, float> tile_c;
    typedef ggml_cuda_mma::tile<16, 8, half2> tile_a;
    typedef ggml_cuda_mma::tile<8,  8, half2> tile_b;
    typedef ggml_cuda_mma::tile<16, 4, half2> tile_ah;

    const int lane16 = lane % 16;

    tile_c acc[MT][NTT];
    tile_ah acc16[MT][NTT];

    // one payload word per thread, held a tile ahead
    static_assert(NTJ*((16*K)/2) <= NT, "escha: payload needs more than one word per thread");
    const bool has_pay = tid < NTJ*n_wd;
    const int  pt = tid/n_wd, pw = tid % n_wd;
    uint32_t   ppre = 0;
    if (has_pay && lo < hi) {
        ppre = ((const uint32_t *)(code + (int64_t)(lo*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
    }

    // Activations for tile lo into buffer 0.  SM120/Blackwell uses the
    // double-buffered cp.async/LDGSTS A-stage overlap by default (proven safe
    // and ~58% faster at matched 2k; see EXP-01).  A synchronous vector-copy
    // fallback remains available for WSL driver edge cases where cp.async can
    // leave a legacy HMMA kernel pending: define ESCHA_MMA_SM120_SYNC_FALLBACK
    // to select it.  Older architectures always use the overlap.
    if (lo < hi) {
        {
            const int row = row0 + cp_m;
#if defined(BLACKWELL_MMA_AVAILABLE) && defined(ESCHA_MMA_SM120_SYNC_FALLBACK)
            const uint4 v = row < n_rows
                ? *(const uint4 *) (u + (int64_t) row*IC + lo*ESCHA_TILE + cp_h)
                : make_uint4(0, 0, 0, 0);
            *(uint4 *) (s_u + cp_m*ESCHA_TILE + cp_h) = v;
#else
            const int src_row = row < n_rows ? row : 0;
            __pipeline_memcpy_async(s_u + cp_m*ESCHA_TILE + cp_h,
                                    u + (int64_t) src_row*IC + lo*ESCHA_TILE + cp_h,
                                    CPB, row < n_rows ? 0 : CPB);
#endif
        }
#if !defined(BLACKWELL_MMA_AVAILABLE) || !defined(ESCHA_MMA_SM120_SYNC_FALLBACK)
        __pipeline_commit();
#endif
    }

    for (int ti = lo; ti < hi; ++ti) {
        half * su_cur = s_u + (((ti - lo) & 1)     )*(BM*ESCHA_TILE);
        half * su_nxt = s_u + (((ti - lo) & 1) ^ 1 )*(BM*ESCHA_TILE);

        if (has_pay) {
            // word pw is the high half of pair pw and the low half of pair pw+1
            s_pay[pt*ESCHA_MAX_W + pw].y = ppre;
            s_pay[pt*ESCHA_MAX_W + (pw + 1 == NWD ? 0 : pw + 1)].x = ppre;
        }
        if (has_pay && ti + 1 < hi) {
            ppre = ((const uint32_t *)(code + (int64_t)((ti + 1)*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
        }
        // The copy for THIS tile was committed last round; drain it before the barrier
        // that publishes s_pay, so su_cur is visible to every warp below.
#if !defined(BLACKWELL_MMA_AVAILABLE) || !defined(ESCHA_MMA_SM120_SYNC_FALLBACK)
        __pipeline_wait_prior(0);
#endif
        __syncthreads();

#if !defined(BLACKWELL_MMA_AVAILABLE) || !defined(ESCHA_MMA_SM120_SYNC_FALLBACK)
        if (ti + 1 < hi) {
            const int row = row0 + cp_m;
            const int src_row = row < n_rows ? row : 0;
            __pipeline_memcpy_async(su_nxt + cp_m*ESCHA_TILE + cp_h,
                                    u + (int64_t) src_row*IC + (ti + 1)*ESCHA_TILE + cp_h,
                                    CPB, row < n_rows ? 0 : CPB);
            __pipeline_commit();
        }
#endif

        // decode into [n][k]. All the addressing is precomputed above; iteration k reads
        // payload tile k and writes column dccl + 16k, so this is just load, shift, codebook.
#pragma unroll
        for (int k = 0; k < DPT; ++k) {
            const uint2 * pay = s_pay + k*ESCHA_MAX_W;
            const int c = dccl + ESCHA_TILE*k;
            s_w[c*ESCHA_TILE + dr] =
                escha_codebook_h(__funnelshift_r(pay[dw0].y, pay[dw0].x, dsh) & 0xffffu);
        }
        __syncthreads();

        {
            const half2 * su2 = (const half2 *) su_cur;   // [BM][8] half2
            const half2 * sw2 = (const half2 *) s_w;      // [BN][8] half2

            tile_a A[MT];
            tile_b B[NTT];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                ggml_cuda_mma::load_ldmatrix(A[i], su2 + (size_t)(wm*(16*MT) + i*16)*8, 8);
            }
#pragma unroll
            for (int j = 0; j < NTT; ++j) {
                ggml_cuda_mma::load_ldmatrix(B[j], sw2 + (size_t)(wn*(8*NTT) + j*8)*8, 8);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
#pragma unroll
                for (int j = 0; j < NTT; ++j) {
                    if constexpr (FP16_ACC) {
                        ggml_cuda_mma::mma(acc16[i][j], A[i], B[j]);
                    } else {
                        ggml_cuda_mma::mma(acc[i][j], A[i], B[j]);
                    }
                }
            }
        }
        __syncthreads();

#if defined(BLACKWELL_MMA_AVAILABLE) && defined(ESCHA_MMA_SM120_SYNC_FALLBACK)
        // Publish the following input tile after all warps have finished reading
        // su_cur.  The top-of-loop barrier publishes this synchronous copy.
        if (ti + 1 < hi) {
            const int row = row0 + cp_m;
            const uint4 v = row < n_rows
                ? *(const uint4 *) (u + (int64_t) row*IC + (ti + 1)*ESCHA_TILE + cp_h)
                : make_uint4(0, 0, 0, 0);
            *(uint4 *) (su_nxt + cp_m*ESCHA_TILE + cp_h) = v;
        }
#endif
    }

#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NTT; ++j) {
            if constexpr (FP16_ACC) {
                // tile_ah is tile<16,4,half2>: 2 half2 registers per thread (ne=2),
                // NOT tile_c::ne=4 floats.  The m16n8k16.f16 D fragment packs two
                // f16 lanes per 32-bit register: x[l].x is the even column and
                // x[l].y the odd column of the same row; row offset +8 applies to
                // x[1].  Map to the fp32 fragment coordinates: x[0].x <-> l=0,
                // x[0].y <-> l=1, x[1].x <-> l=2, x[1].y <-> l=3.
#pragma unroll
                for (int l = 0; l < 2; ++l) {
                    const half2 v = acc16[i][j].x[l];
                    const int m0 = wm*(16*MT) + i*16 + tile_c::get_i(2*l + 0);
                    const int n0 = wn*(8*NTT)  + j*8  + tile_c::get_j(2*l + 0);
                    const int m1 = wm*(16*MT) + i*16 + tile_c::get_i(2*l + 1);
                    const int n1 = wn*(8*NTT)  + j*8  + tile_c::get_j(2*l + 1);
                    const int row_a = row0 + m0;
                    const int row_b = row0 + m1;
                    if (row_a < n_rows) {
                        partial[((int64_t) sl*n_rows + row_a)*OC + oc0 + n0] = __half2float(v.x);
                    }
                    if (row_b < n_rows) {
                        partial[((int64_t) sl*n_rows + row_b)*OC + oc0 + n1] = __half2float(v.y);
                    }
                }
            } else {
#pragma unroll
                for (int l = 0; l < tile_c::ne; ++l) {
                    const int m   = wm*(16*MT) + i*16 + tile_c::get_i(l);
                    const int n   = wn*(8*NTT) + j*8  + tile_c::get_j(l);
                    const int row = row0 + m;
                    if (row < n_rows) {
                        partial[((int64_t) sl*n_rows + row)*OC + oc0 + n] = acc[i][j].x[l];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(code, lut, dep, u, partial, IC, OC, n_rows, n_slices);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

#ifdef ESCHA_MMA_FUSED_FINALIZE_EXPERIMENT
// P-ARCH-14: fused-finalize variant of the tiled MMA kernel.  Compiled only
// with ESCHA_MMA_FUSED_FINALIZE_EXPERIMENT=1 and launched only for n_slices==1
// (no split-K reduction), where the separate finalize kernel's Hadamard-128 +
// rout epilogue can consume the CTA tile directly, removing the fp32 partial
// write+read round trip.  Decode, A-stage overlap, shared-B materialization,
// ldmatrix, HMMA, rotation, and geometry are byte-for-byte unchanged; the
// output transform runs the same stage order as escha_finalize_dense, so the
// numerics are bit-identical to the partial + finalize path.
template <int K, int BM, int BN>
static __global__ void __launch_bounds__(256, 1) escha_matmul_dense_tiled_mma_ff(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const half    * __restrict__ u,
        float         * __restrict__ partial,   // unused; kept for a faithful copy
        const half    * __restrict__ rout,
        float         * __restrict__ dst,
        const int IC, const int OC, const int n_rows, const int n_slices,
        const int ne1, const int64_t nb_d1, const int64_t nb_d2) {
#ifdef TURING_MMA_AVAILABLE
    static_assert(BN == 128, "escha: fused finalize assumes the 128-column Hadamard block");

    constexpr int NT   = 256;
    constexpr int NW   = NT/32;          // warps
    constexpr int WN   = 2;              // warps across the column axis
    constexpr int WM   = NW/WN;          // warps down the row axis
    constexpr int MT   = BM/16/WM;       // 16-row accumulator tiles per warp
    constexpr int NTT  = BN/8/WN;        // 8-col accumulator tiles per warp
    constexpr int NTJ  = BN/ESCHA_TILE;  // output tiles whose payload this block holds

    extern __shared__ char s_raw[];
    uint2    * s_pay = (uint2 *) s_raw;                               // [NTJ][ESCHA_MAX_W] pairs
    half     * s_u   = (half *)(s_pay + NTJ*ESCHA_MAX_W);             // [2][BM][16]
    half     * s_w   = s_u + 2*BM*ESCHA_TILE;                         // [BN][16]
    float (*s_fuse)[BN] = (float (*)[BN])(s_w + BN*ESCHA_TILE);       // [16][BN] fused epilogue

    GGML_UNUSED(lut);
    GGML_UNUSED(dep);
    GGML_UNUSED(partial);

    const int NWD = 8*K;
    const int NB  = 32*NWD;

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int lane = threadIdx.x;          // must stay the lane: mma.cuh indexes on it
    const int warp = threadIdx.y;
    const int tid  = warp*32 + lane;        // flat id, for the layout-agnostic staging loops
    const int row0 = blockIdx.x*BM;
    const int oc0  = blockIdx.y*BN;

    const int sl = blockIdx.z;
    const int lo = (int) (((int64_t) nit*sl)/n_slices);
    const int hi = (int) (((int64_t) nit*(sl + 1))/n_slices);

    const int wm   = warp / WN;
    const int wn   = warp % WN;

    constexpr int DPT = (ESCHA_TILE*BN)/NT;   // weights this thread decodes per tile
    static_assert(NT % ESCHA_TILE == 0,        "escha: r would not be thread-invariant");
    static_assert((ESCHA_TILE*BN) % NT == 0,   "escha: ragged decode assignment");
    static_assert(NT/ESCHA_TILE <= ESCHA_TILE, "escha: ccl would not be thread-invariant");

    // cp.async moves 16 bytes = 8 halves per thread; BM*ESCHA_TILE halves is exactly
    // NT*8 at BM=128/NT=256, so every thread issues one copy and none loops.
    constexpr int CPB = 16;                              // bytes per thread per tile
    static_assert(BM*ESCHA_TILE*sizeof(half) == NT*CPB,  "escha: activation copy is ragged");
    const int cp_m  = tid / (ESCHA_TILE*sizeof(half)/CPB);   // row this thread copies into
    const int cp_h  = (tid % (ESCHA_TILE*sizeof(half)/CPB))*(CPB/sizeof(half));

    const int dr   = tid % ESCHA_TILE;         // this thread's r, every k, every tile
    const int dccl = tid / ESCHA_TILE;         // and its column within the 16-wide tile
    int dsp = ((32 - K) - K*(escha_dep_pi(dr) + 32*dccl + 4*(dccl >> 3))) % NB;
    if (dsp < 0) {
        dsp += NB;
    }
    const int dg0 = dsp >> 5;
    const int dw0 = dg0 ? (NWD - dg0) : 0;
    const int dw1 = dw0 ? (dw0 - 1)   : (NWD - 1);
    const int dsh = dsp & 31;

    typedef ggml_cuda_mma::tile<16, 8, float> tile_c;
    typedef ggml_cuda_mma::tile<16, 8, half2> tile_a;
    typedef ggml_cuda_mma::tile<8,  8, half2> tile_b;

    tile_c acc[MT][NTT];

    // one payload word per thread, held a tile ahead
    static_assert(NTJ*((16*K)/2) <= NT, "escha: payload needs more than one word per thread");
    const bool has_pay = tid < NTJ*n_wd;
    const int  pt = tid/n_wd, pw = tid % n_wd;
    uint32_t   ppre = 0;
    if (has_pay && lo < hi) {
        ppre = ((const uint32_t *)(code + (int64_t)(lo*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
    }

    // Activations for tile lo into buffer 0 (same overlap policy as the
    // non-fused kernel: SM120 defaults to the cp.async path; the synchronous
    // fallback is selected only by ESCHA_MMA_SM120_SYNC_FALLBACK).
    if (lo < hi) {
        {
            const int row = row0 + cp_m;
#if defined(BLACKWELL_MMA_AVAILABLE) && defined(ESCHA_MMA_SM120_SYNC_FALLBACK)
            const uint4 v = row < n_rows
                ? *(const uint4 *) (u + (int64_t) row*IC + lo*ESCHA_TILE + cp_h)
                : make_uint4(0, 0, 0, 0);
            *(uint4 *) (s_u + cp_m*ESCHA_TILE + cp_h) = v;
#else
            const int src_row = row < n_rows ? row : 0;
            __pipeline_memcpy_async(s_u + cp_m*ESCHA_TILE + cp_h,
                                    u + (int64_t) src_row*IC + lo*ESCHA_TILE + cp_h,
                                    CPB, row < n_rows ? 0 : CPB);
#endif
        }
#if !defined(BLACKWELL_MMA_AVAILABLE) || !defined(ESCHA_MMA_SM120_SYNC_FALLBACK)
        __pipeline_commit();
#endif
    }

    for (int ti = lo; ti < hi; ++ti) {
        half * su_cur = s_u + (((ti - lo) & 1)     )*(BM*ESCHA_TILE);
        half * su_nxt = s_u + (((ti - lo) & 1) ^ 1 )*(BM*ESCHA_TILE);

        if (has_pay) {
            // word pw is the high half of pair pw and the low half of pair pw+1
            s_pay[pt*ESCHA_MAX_W + pw].y = ppre;
            s_pay[pt*ESCHA_MAX_W + (pw + 1 == NWD ? 0 : pw + 1)].x = ppre;
        }
        if (has_pay && ti + 1 < hi) {
            ppre = ((const uint32_t *)(code + (int64_t)((ti + 1)*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
        }
        // The copy for THIS tile was committed last round; drain it before the barrier
        // that publishes s_pay, so su_cur is visible to every warp below.
#if !defined(BLACKWELL_MMA_AVAILABLE) || !defined(ESCHA_MMA_SM120_SYNC_FALLBACK)
        __pipeline_wait_prior(0);
#endif
        __syncthreads();

#if !defined(BLACKWELL_MMA_AVAILABLE) || !defined(ESCHA_MMA_SM120_SYNC_FALLBACK)
        if (ti + 1 < hi) {
            const int row = row0 + cp_m;
            const int src_row = row < n_rows ? row : 0;
            __pipeline_memcpy_async(su_nxt + cp_m*ESCHA_TILE + cp_h,
                                    u + (int64_t) src_row*IC + (ti + 1)*ESCHA_TILE + cp_h,
                                    CPB, row < n_rows ? 0 : CPB);
            __pipeline_commit();
        }
#endif

        // decode into [n][k]. All the addressing is precomputed above; iteration k reads
        // payload tile k and writes column dccl + 16k, so this is just load, shift, codebook.
#pragma unroll
        for (int k = 0; k < DPT; ++k) {
            const uint2 * pay = s_pay + k*ESCHA_MAX_W;
            const int c = dccl + ESCHA_TILE*k;
            s_w[c*ESCHA_TILE + dr] =
                escha_codebook_h(__funnelshift_r(pay[dw0].y, pay[dw0].x, dsh) & 0xffffu);
        }
        __syncthreads();

        {
            const half2 * su2 = (const half2 *) su_cur;   // [BM][8] half2
            const half2 * sw2 = (const half2 *) s_w;      // [BN][8] half2

            tile_a A[MT];
            tile_b B[NTT];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                ggml_cuda_mma::load_ldmatrix(A[i], su2 + (size_t)(wm*(16*MT) + i*16)*8, 8);
            }
#pragma unroll
            for (int j = 0; j < NTT; ++j) {
                ggml_cuda_mma::load_ldmatrix(B[j], sw2 + (size_t)(wn*(8*NTT) + j*8)*8, 8);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
#pragma unroll
                for (int j = 0; j < NTT; ++j) {
                    ggml_cuda_mma::mma(acc[i][j], A[i], B[j]);
                }
            }
        }
        __syncthreads();

#if defined(BLACKWELL_MMA_AVAILABLE) && defined(ESCHA_MMA_SM120_SYNC_FALLBACK)
        // Publish the following input tile after all warps have finished reading
        // su_cur.  The top-of-loop barrier publishes this synchronous copy.
        if (ti + 1 < hi) {
            const int row = row0 + cp_m;
            const uint4 v = row < n_rows
                ? *(const uint4 *) (u + (int64_t) row*IC + (ti + 1)*ESCHA_TILE + cp_h)
                : make_uint4(0, 0, 0, 0);
            *(uint4 *) (su_nxt + cp_m*ESCHA_TILE + cp_h) = v;
        }
#endif
    }

    // Fused single-slice epilogue: for each 16-row chunk owned by one warp
    // pair, stage the row's BN columns, run the normalized Sylvester-Hadamard
    // with the same stage order as escha_hadamard_128 (len = 1..64), scale by
    // 1/sqrt(128) and rout, then store to dst with the finalize addressing.
    const float HAD_SCALE = rsqrtf(128.0f);
    for (int c = 0; c < BM/16; ++c) {
        const int r0 = c*16;
        if (wm == c/2) {
#pragma unroll
            for (int i = 0; i < MT; ++i) {
#pragma unroll
                for (int j = 0; j < NTT; ++j) {
#pragma unroll
                    for (int l = 0; l < tile_c::ne; ++l) {
                        const int m = wm*(16*MT) + i*16 + tile_c::get_i(l);
                        const int n = wn*(8*NTT) + j*8  + tile_c::get_j(l);
                        if (m >= r0 && m < r0 + 16) {
                            s_fuse[m - r0][n] = acc[i][j].x[l];
                        }
                    }
                }
            }
        }
        __syncthreads();

        for (int len = 1; len < BN; len <<= 1) {
            for (int idx = tid; idx < 16*(BN/2); idx += NT) {
                const int r = idx / (BN/2);
                const int j = idx % (BN/2);
                const int i = (j / len)*(2*len) + (j % len);
                float * b = s_fuse[r] + i;
                const float a0 = b[0];
                const float a1 = b[len];
                b[0] = a0 + a1;
                b[len] = a0 - a1;
            }
            __syncthreads();
        }

        for (int idx = tid; idx < 16*BN; idx += NT) {
            const int r = idx / BN;
            const int n = idx % BN;
            const int row = row0 + r0 + r;
            if (row < n_rows) {
                float * dst_row = (float *)((char *) dst + (int64_t)(row % ne1)*nb_d1
                                                           + (int64_t)(row / ne1)*nb_d2);
                dst_row[oc0 + n] = s_fuse[r][n]*HAD_SCALE*__half2float(rout[oc0 + n]);
            }
        }
        __syncthreads();
    }
#else
    GGML_UNUSED_VARS(code, lut, dep, u, partial, rout, dst, IC, OC, n_rows, n_slices,
                     ne1, nb_d1, nb_d2);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}
#endif // ESCHA_MMA_FUSED_FINALIZE_EXPERIMENT

// partial[slice][row][c] = sum over this slice's input tiles of u . decode(code)
template <int K, int R>
static __global__ void escha_matmul_dense(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
    extern __shared__ char s_raw[];

    // every decode wants the adjacent pair (pay[w0-1], pay[w0]), so hold the payload AS
    // overlapping pairs: one aligned LDS.64 then replaces the two LDS the funnel shift needed
    uint2 * s_pay = (uint2 *) s_raw;                                   // [ESCHA_GROUPS][ESCHA_MAX_W]
    float * s_u   = (float *)(s_pay + ESCHA_GROUPS*ESCHA_MAX_W);       // [R][16]

    GGML_UNUSED(lut);
    GGML_UNUSED(dep);
    const int NW = 8*K;          // 32-bit words in a tile payload
    const int NB = 32*NW;        // bits
    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid   = threadIdx.x;
    const int start = blockIdx.x*R;
    const int nrow  = min(R, n_rows - start);

    // this block's share of the input tiles
    const int sl  = blockIdx.z;
    const int lo  = (int) (((int64_t) nit*sl)/n_slices);
    const int hi  = (int) (((int64_t) nit*(sl + 1))/n_slices);

    const int grp = tid / ESCHA_TILE;
    const int cc  = tid % ESCHA_TILE;
    const int tj  = blockIdx.y*ESCHA_GROUPS + grp;
    uint2 * pay = s_pay + grp*ESCHA_MAX_W;

    // Start position for this thread's column before the row-dependent term.
    // The canonical K=2/3 Escha dep tables are an affine cyclic bit mapping;
    // use its closed form rather than a 16-entry gather for every weight.
    int s0 = ((32 - K) - K*(32*cc + 4*(cc >> 3))) % NB;
    if (s0 < 0) {
        s0 += NB;
    }

    float acc[R];
#pragma unroll
    for (int m = 0; m < R; ++m) {
        acc[m] = 0.0f;
    }
    // At R == 1 the block reads one row, so its entire slice of u fits in shared and is
    // staged once here -- the per-tile staging below then disappears, and with it the
    // block-wide barrier that ordered it. This is what their 10,240-byte allocation is.
    if constexpr (R == 1) {
        const float * u_row = u + (int64_t) start*IC + (int64_t) lo*ESCHA_TILE;
        const int n_stage = (hi - lo)*ESCHA_TILE;
        for (int j = tid; j < n_stage; j += ESCHA_NT) {
            s_u[j] = u_row[j];
        }
    }
    // payload words this thread owns, and the tile fetched one iteration ahead
    constexpr int NPW = (8*K + ESCHA_TILE - 1)/ESCHA_TILE;
    uint32_t pre[NPW];
    if (lo < hi) {
        const uint32_t * s0p = (const uint32_t *)(code + (int64_t)(lo*nct + tj)*(16*K));
#pragma unroll
        for (int i = 0; i < NPW; ++i) {
            const int wd = cc + i*ESCHA_TILE;
            if (wd < n_wd) {
                pre[i] = s0p[wd];
            }
        }
    }
    __syncthreads();

    for (int ti = lo; ti < hi; ++ti) {
        // as in the routed kernel: the block cooperates on one 16-wide slice of u per row,
        // and the 16 threads of a group then read the same entry, so it broadcasts
        if constexpr (R != 1) {
            for (int j = tid; j < R*ESCHA_TILE; j += ESCHA_NT) {
                const int m = j / ESCHA_TILE;
                const int r = j % ESCHA_TILE;
                s_u[j] = m < nrow ? u[(int64_t)(start + m)*IC + ti*ESCHA_TILE + r] : 0.0f;
            }
        }

        // publish the tile fetched last round, then issue the next fetch immediately: the
        // LDG then overlaps this tile's decode instead of stalling in front of it, which is
        // what the LDG -> dependent STS pair at the top of the loop was costing
#pragma unroll
        for (int i = 0; i < NPW; ++i) {
            const int wd = cc + i*ESCHA_TILE;
            if (wd < n_wd) {
                // word wd is the high half of pair wd and the low half of pair wd+1
                pay[wd].y = pre[i];
                pay[wd + 1 == NW ? 0 : wd + 1].x = pre[i];
            }
        }
        if (ti + 1 < hi) {
            const uint32_t * nxt = (const uint32_t *)(code + (int64_t)((ti + 1)*nct + tj)*(16*K));
#pragma unroll
            for (int i = 0; i < NPW; ++i) {
                const int wd = cc + i*ESCHA_TILE;
                if (wd < n_wd) {
                    pre[i] = nxt[wd];
                }
            }
        }
        if constexpr (R == 1) { __syncwarp(); } else { __syncthreads(); }

        const float * uu = s_u + (ti - lo)*ESCHA_TILE;

        // generation unrolls fully so pi(r) folds to a compile-time constant; prefill keeps
        // a partial unroll, where acc[R] already claims the registers
#pragma unroll (R <= 8 ? 16 : 4)
        for (int r = 0; r < ESCHA_TILE; ++r) {
            int sp = s0 - K*escha_dep_pi(r);
            if (sp < 0) {
                sp += NB;
            }

            const int g0 = sp >> 5;
            const int w0 = g0 ? (NW - g0) : 0;
            const uint2 p = pay[w0];
            const uint32_t idx = __funnelshift_r(p.y, p.x, sp & 31) & 0xffffu;

            const float wv = escha_codebook(idx);
            if constexpr (R == 1) {
                acc[0] += uu[r]*wv;
            } else {
#pragma unroll
                for (int m = 0; m < R; ++m) {
                    acc[m] += s_u[m*ESCHA_TILE + r]*wv;
                }
            }
        }
        if constexpr (R == 1) { __syncwarp(); } else { __syncthreads(); }
    }

    for (int m = 0; m < nrow; ++m) {
        partial[((int64_t) sl*n_rows + start + m)*OC + blockIdx.y*ESCHA_NT + tid] = acc[m];
    }
}

// Generation is a different problem from prefill: a row has no decode reuse, and
// the 128-thread kernel above spends two block barriers per 16-wide input tile to
// share both payload and activations.  The QTIP/Escha bit permutation admits a
// warp-local form instead.  A lane decodes eight weights which contribute to two
// output columns; four lanes reduce each column at the end.  That removes the
// shared-memory staging and every per-tile barrier while retaining the existing
// split-K partial layout (and therefore its deterministic final reduction).
//
// This is deliberately R=1 only.  Prefill needs activation reuse and remains on
// the tiled kernels above.  Four independent 16-column tiles share one CTA so
// its grid geometry is identical to the established 128-column decode path.
template <int K>
static __global__ void escha_matmul_dense_warp(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
    GGML_UNUSED(lut);
    GGML_UNUSED(dep);

    constexpr int NW = 8*K;
    const int nit = IC/ESCHA_TILE;
    const int nct = OC/ESCHA_TILE;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int tile = blockIdx.y*4 + warp;
    const int row = blockIdx.x;
    const int sl = blockIdx.z;
    const int lo = (int) (((int64_t) nit*sl)/n_slices);
    const int hi = (int) (((int64_t) nit*(sl + 1))/n_slices);

    // Keep all lanes participating in the shuffles for a ragged final CTA.
    const bool live = row < n_rows && tile < nct;

    // Each lane owns the same two columns for every 16x16 input tile.  This is
    // the closed-form inverse of escha_dep_pi()/the cyclic payload layout.
    const int r0 = 2*(lane & 3);
    const int c0 = 2*((lane >> 3) & 1) + 4*((lane >> 4) & 1);
    const int d  = (lane >> 2) & 1;
    const int cA = c0 + d;
    const int cB = cA + 8;

    int cur, prv, sh;
    if constexpr (K == 3) {
        const int b = lane*24;
        const int r86 = b + 791;
        const int r87 = r86 & 2016;
        cur = (((r86 >> 3) & 252) - 96)/4;
        prv = lane == 0 ? 23 : (((b + 755) >> 5) - 24);
        sh = r87 - b - 760;
    } else {
        cur = (lane >> 1) & 15;
        prv = (cur - 1) & 15;
        sh = (lane & 1) == 0 ? 16 : 0;
    }

    float acc_a = 0.0f;
    float acc_b = 0.0f;
    const float * const u_row = u + (int64_t) row*IC;
    const uint32_t * const code32 = (const uint32_t *) code;

    // Two loads are issued before either is consumed, providing useful ILP for
    // the code load -> shuffle -> codec -> FMA dependency chain.
    for (int ti = lo; ti < hi; ti += 2) {
        uint32_t words[2] = { 0, 0 };
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int t = ti + q;
            if (live && t < hi && lane < NW) {
                words[q] = code32[((int64_t) t*nct + tile)*NW + lane];
            }
        }

#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int t = ti + q;
            if (t >= hi) {
                continue;
            }
            const uint64_t pair = ((uint64_t) __shfl_sync(0xffffffffu, words[q], prv) << 32)
                                |  (uint64_t) __shfl_sync(0xffffffffu, words[q], cur);
            const uint32_t v0 = (uint32_t) (pair >> sh);
            const uint32_t v1 = K == 3 ? (uint32_t) (pair >> (sh + 18)) : 0u;
            const float2 x01 = live ? *(const float2 *) (u_row + t*ESCHA_TILE + r0)
                                     : make_float2(0.0f, 0.0f);
            const float2 x89 = live ? *(const float2 *) (u_row + t*ESCHA_TILE + r0 + 8)
                                     : make_float2(0.0f, 0.0f);
            const float xr[4] = { x89.y, x89.x, x01.y, x01.x };
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int off0 = K*s;
                const int off1 = K*(s + 4);
                const uint32_t src0 = K == 3 && s >= 6 ? v1 : v0;
                const int adj0 = K == 3 && s >= 6 ? 18 : 0;
                const uint32_t src1 = K == 3 && s + 4 >= 6 ? v1 : v0;
                const int adj1 = K == 3 && s + 4 >= 6 ? 18 : 0;
                acc_b += escha_codebook((src0 >> (off0 - adj0)) & 0xffffu)*xr[s];
                acc_a += escha_codebook((src1 >> (off1 - adj1)) & 0xffffu)*xr[s];
            }
        }
    }

    acc_a += __shfl_xor_sync(0xffffffffu, acc_a, 2, 4);
    acc_b += __shfl_xor_sync(0xffffffffu, acc_b, 2, 4);
    acc_a += __shfl_xor_sync(0xffffffffu, acc_a, 1, 4);
    acc_b += __shfl_xor_sync(0xffffffffu, acc_b, 1, 4);
    if (live && (lane & 3) == 0) {
        float * const out = partial + ((int64_t) sl*n_rows + row)*OC + tile*ESCHA_TILE;
        out[cA] = acc_a;
        out[cB] = acc_b;
    }
}

// sum the slices in a fixed order (so the result is reproducible), rotate the
// 128-column group, scale by rout
static __global__ void escha_finalize_dense(
        const half  * __restrict__ rout,
        const float * __restrict__ partial,
        float       * __restrict__ dst,
        const int OC, const int ne1, const int n_rows, const int n_slices,
        const int64_t nb_d1, const int64_t nb_d2) {
    __shared__ float s_acc[ESCHA_NT];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int c   = blockIdx.y*ESCHA_NT + tid;

    float sum = 0.0f;
    for (int s = 0; s < n_slices; ++s) {
        sum += partial[((int64_t) s*n_rows + row)*OC + c];
    }
    s_acc[tid] = sum;
    __syncthreads();

    escha_hadamard_128(s_acc, ESCHA_NT, tid, ESCHA_NT);

    float * dst_row = (float *)((char *) dst + (int64_t)(row % ne1)*nb_d1
                                             + (int64_t)(row / ne1)*nb_d2);
    dst_row[c] = s_acc[tid]*__half2float(rout[c]);
}

static __global__ void escha_debug_find_nonfinite(const float * data, int64_t n, int * result) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n && !isfinite(data[i])) {
        atomicCAS(result, -1, (int) i);
    }
}

static void escha_capture_parch01_blob(const char * dir, const char * name, const void * data, size_t size) {
    char path[1024];
    const int n = snprintf(path, sizeof(path), "%s/%s", dir, name);
    GGML_ASSERT(n > 0 && (size_t) n < sizeof(path));

    FILE * file = fopen(path, "wb");
    GGML_ASSERT(file != nullptr);
    GGML_ASSERT(fwrite(data, 1, size, file) == size);
    GGML_ASSERT(fclose(file) == 0);
}

void ggml_cuda_op_escha_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code = dst->src[0];
    const ggml_tensor * rin  = dst->src[1];
    const ggml_tensor * rout = dst->src[2];
    const ggml_tensor * lut  = dst->src[3];
    const ggml_tensor * dep  = dst->src[4];
    const ggml_tensor * x    = dst->src[5];

    GGML_ASSERT(code->type == GGML_TYPE_I16 && dep->type == GGML_TYPE_I16);
    GGML_ASSERT(rin->type == GGML_TYPE_F16 && rout->type == GGML_TYPE_F16 && lut->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int K   = code->ne[0]/16;
    const int OC  = code->ne[1]*16;
    const int IC  = code->ne[2]*16;
    const int nit = IC/ESCHA_TILE;

    const int n_rows = x->ne[1]*x->ne[2];
    const int n_ocb  = OC/ESCHA_NT;

    cudaStream_t stream = ctx.stream();
    if (const char * capture_dir = getenv("ESCHA_CAPTURE_PARCH01_DIR")) {
        static bool captured_k2 = false;
        static bool captured_k3 = false;
        bool * captured = K == 2 ? &captured_k2 : K == 3 ? &captured_k3 : nullptr;
        if (captured != nullptr && !*captured && IC == 5120 && OC == 17408 && n_rows == 512) {
            const size_t payload_size = (size_t) 16*K*sizeof(int16_t);
            const size_t dep_size     = ggml_nbytes(dep);
            const size_t rin_size     = ggml_nbytes(rin);
            const size_t rout_size    = ggml_nbytes(rout);
            const size_t x_s1         = x->nb[1]/sizeof(float);
            const size_t x_s2         = x->nb[2]/sizeof(float);
            const size_t x_span       = (size_t) (x->ne[1] - 1)*x_s1
                                      + (size_t) (x->ne[2] - 1)*x_s2 + IC;

            std::vector<int16_t> payload(payload_size/sizeof(int16_t));
            std::vector<char> host_dep(dep_size);
            std::vector<char> host_rin(rin_size);
            std::vector<char> host_rout(rout_size);
            std::vector<float> x_storage(x_span);
            std::vector<float> activation((size_t) n_rows*IC);
            std::vector<uint32_t> fragments(2*WARP_SIZE*2);
            ggml_cuda_pool_alloc<uint32_t> fragment_buf(ctx.pool(), fragments.size());

            CUDA_CHECK(cudaMemcpyAsync(payload.data(), code->data, payload_size, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(host_dep.data(), dep->data, dep_size, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(host_rin.data(), rin->data, rin_size, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(host_rout.data(), rout->data, rout_size, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(x_storage.data(), x->data, x_span*sizeof(float), cudaMemcpyDeviceToHost, stream));
            switch (K) {
                case 2:
                    escha_capture_tile_fragments<2><<<1, WARP_SIZE, 0, stream>>>(
                        (const int16_t *) code->data, fragment_buf.get());
                    break;
                case 3:
                    escha_capture_tile_fragments<3><<<1, WARP_SIZE, 0, stream>>>(
                        (const int16_t *) code->data, fragment_buf.get());
                    break;
                default:
                    GGML_ABORT("escha: unsupported P-ARCH-01 capture K=%d", K);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(fragments.data(), fragment_buf.get(),
                                       fragments.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            for (int row = 0; row < n_rows; ++row) {
                const int i1 = row % x->ne[1];
                const int i2 = row / x->ne[1];
                memcpy(activation.data() + (size_t) row*IC,
                       x_storage.data() + (size_t) i1*x_s1 + (size_t) i2*x_s2,
                       (size_t) IC*sizeof(float));
            }

            const char * prefix = K == 2 ? "k2" : "k3";
            char name[128];
#define ESCHA_CAPTURE_PARCH01(SUFFIX, DATA, SIZE) \
            snprintf(name, sizeof(name), "%s_%s", prefix, SUFFIX); \
            escha_capture_parch01_blob(capture_dir, name, DATA, SIZE)
            ESCHA_CAPTURE_PARCH01("tile_0_0_payload.i16.bin", payload.data(), payload_size);
            ESCHA_CAPTURE_PARCH01("dep.i16.bin", host_dep.data(), dep_size);
            ESCHA_CAPTURE_PARCH01("rin.f16.bin", host_rin.data(), rin_size);
            ESCHA_CAPTURE_PARCH01("rout.f16.bin", host_rout.data(), rout_size);
            ESCHA_CAPTURE_PARCH01("activation_512x5120.f32.bin", activation.data(), activation.size()*sizeof(float));
            ESCHA_CAPTURE_PARCH01("bee_b_fragments.u32.bin", fragments.data(), fragments.size()*sizeof(uint32_t));
#undef ESCHA_CAPTURE_PARCH01
            *captured = true;
            fprintf(stderr,
                    "ESCHA_CAPTURE_PARCH01 K=%d IC=%d OC=%d rows=%d tile=(0,0) dir=%s\n",
                    K, IC, OC, n_rows, capture_dir);
        }
    }
    const bool profile = getenv("ESCHA_PROFILE") != nullptr;
    // Profiling synchronizes profile_stop before the next invocation on this
    // backend thread, so the same events can be reused safely.  Keeping four
    // events per thread/device avoids exercising the CUDA/WSL event-object
    // create/destroy path thousands of times during a full prompt profile.
    static thread_local cudaEvent_t profile_events[GGML_CUDA_MAX_DEVICES][4] = {};
    cudaEvent_t & profile_start       = profile_events[ctx.device][0];
    cudaEvent_t & profile_rotate_stop = profile_events[ctx.device][1];
    cudaEvent_t & profile_matmul_stop = profile_events[ctx.device][2];
    cudaEvent_t & profile_stop        = profile_events[ctx.device][3];
    const char * profile_route = "unresolved";
    if (profile) {
        if (profile_start == nullptr) {
            CUDA_CHECK(cudaEventCreate(&profile_start));
            CUDA_CHECK(cudaEventCreate(&profile_rotate_stop));
            CUDA_CHECK(cudaEventCreate(&profile_matmul_stop));
            CUDA_CHECK(cudaEventCreate(&profile_stop));
        }
        CUDA_CHECK(cudaEventRecord(profile_start, stream));
    }
    const auto finish_profile = [&]() {
        if (!profile) {
            return;
        }
        CUDA_CHECK(cudaEventRecord(profile_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(profile_stop));
        float elapsed_ms = 0.0f;
        float rotate_ms = 0.0f;
        float matmul_ms = 0.0f;
        float epilogue_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, profile_start, profile_stop));
        CUDA_CHECK(cudaEventElapsedTime(&rotate_ms, profile_start, profile_rotate_stop));
        CUDA_CHECK(cudaEventElapsedTime(&matmul_ms, profile_rotate_stop, profile_matmul_stop));
        CUDA_CHECK(cudaEventElapsedTime(&epilogue_ms, profile_matmul_stop, profile_stop));
        fprintf(stderr, "ESCHA_PROFILE k=%d ic=%d oc=%d rows=%d gen=%d route=%s total_ms=%.4f rotate_ms=%.4f matmul_ms=%.4f epilogue_ms=%.4f\n",
                K, IC, OC, n_rows, (int) (n_rows <= ESCHA_GEN_MAX_ROWS), profile_route, elapsed_ms,
                rotate_ms, matmul_ms, epilogue_ms);
    };

    // The tensor-core path wants its activations already in fp16 so cp.async can move them
    // verbatim, so the rotation has to know its consumer before it runs.
    const bool gen = n_rows <= ESCHA_GEN_MAX_ROWS;
    // The tiled MMA kernel is production-qualified on NVIDIA architectures from
    // Turing through SM120.  Keep later, unqualified architectures opt-in until
    // they have equivalent completion, correctness, and stability evidence.
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool mma_arch_ok = cc >= GGML_CUDA_CC_TURING
                          && (cc <= GGML_CUDA_CC_BLACKWELL || getenv("ESCHA_FORCE_MMA") != nullptr);
    const bool use_cublas = !gen
                         && OC % ESCHA_TILE == 0
                         && getenv("ESCHA_CUBLAS_PREFILL") != nullptr;
    const bool use_wmma_bw = !use_cublas
                          && !gen
                          && blackwell_mma_available(cc)
                          && OC % (4*ESCHA_TILE) == 0
                          && getenv("ESCHA_WMMA_PREFILL") != nullptr;
    const bool use_mma = !use_wmma_bw
                      && !gen
                      && mma_arch_ok
                      && OC % ESCHA_MMA_BN == 0
                      && getenv("ESCHA_NO_MMA") == nullptr;

    if (gen) {
        profile_route = getenv("ESCHA_WARP_GEMV") != nullptr ? "warp-gemv-fp32" : "gen-splitk-fp32";
    } else if (use_cublas) {
        profile_route = "cublas-fp16";
    } else if (OC % ESCHA_BN != 0) {
        profile_route = "ragged-fp32";
    } else if (use_wmma_bw) {
        profile_route = "wmma-bw-fp16";
    } else if (use_mma) {
        // PROMOTED default (EXP-04 Stage 2): mixed accumulator policy (native
        // Escha mixed policy: fp16 MMA acc for IC <= 6144, fp32 above), applied
        // per projection across every K2/K3 prefill family.  The acc sidecar in
        // the tag makes the route proof per-family unambiguous.
        profile_route = IC <= 6144 ? "mma-fp16-mixedacc" : "mma-fp32-mixedacc";
    } else {
        profile_route = "tiled-fma-fp32";
    }

    ggml_cuda_pool_alloc<char> u_buf(ctx.pool(),
        (size_t) n_rows*IC*((use_cublas || use_mma || use_wmma_bw) ? sizeof(half) : sizeof(float)));

    if (use_cublas || use_mma || use_wmma_bw) {
        escha_rotate_in_dense<half><<<n_rows, 256, 0, stream>>>(
            (const half *) rin->data, (const float *) x->data, (half *) u_buf.get(),
            IC, (int) x->ne[1], x->nb[1], x->nb[2]);
    } else {
        escha_rotate_in_dense<float><<<n_rows, 256, 0, stream>>>(
            (const half *) rin->data, (const float *) x->data, (float *) u_buf.get(),
            IC, (int) x->ne[1], x->nb[1], x->nb[2]);
    }
    CUDA_CHECK(cudaGetLastError());
    if (profile) {
        CUDA_CHECK(cudaEventRecord(profile_rotate_stop, stream));
    }

    // slice the IC reduction only as far as it takes to fill the device: at batch 1 the
    // natural grid is just n_ocb blocks, but a long prompt already has plenty of rows
    const int  R   = gen ? ESCHA_ROWS_DENSE_GEN : ESCHA_ROWS_DENSE;

    const int n_rb = (n_rows + R - 1)/R;
    // the tiled prefill kernel blocks over BM rows x BN columns instead
    const int n_tb = (n_rows + ESCHA_BM - 1)/ESCHA_BM;
    const int n_cb = OC/ESCHA_BN;
    // batch 1 has only n_ocb blocks before slicing (136 for the FFN), which leaves an 82-SM
    // device mostly idle, so generation slices the reduction much harder than prefill
    int target = gen ? ESCHA_GEN_TARGET_MUL*ESCHA_TARGET : ESCHA_TARGET;
    // The generation path trades independent reduction slices for more CTA
    // occupancy.  Keep the validated default, but permit controlled sweeps on
    // new GPU architectures without rebuilding or changing the math.
    if (gen) {
        if (const char * value = getenv("ESCHA_GEN_TARGET_SCALE")) {
            const int scale = atoi(value);
            if (scale > 0) {
                target = scale*ESCHA_TARGET;
            }
        }
    }
    int n_slices = target/MAX(1, gen ? n_rb*n_ocb : n_tb*n_cb);
    n_slices = MIN(MAX(n_slices, 1), nit);
    if (use_cublas) {
        n_slices = 1;
    }

    // Blackwell needs substantially more independent decode CTAs than the legacy
    // 82-SM tuning target.  These two FFN shapes dominate generation here and both
    // exact splits divide their reduction dimension, so the floating-point tile
    // order within each partial and the fixed final slice order are unchanged.
    // Keep it opt-in until the full parity and context-shape matrix is qualified.
    if (gen && cc >= GGML_CUDA_CC_BLACKWELL && getenv("ESCHA_BW_SPLITK") != nullptr) {
        if (IC == 5120 && OC == 17408) {
            n_slices = 16;
        } else if (IC == 17408 && OC == 5120) {
            n_slices = 64;
        }
    }

    ggml_cuda_pool_alloc<float> p_buf(ctx.pool(), (size_t) n_slices*n_rows*OC);

    if (gen && getenv("ESCHA_WARP_GEMV") != nullptr) {
        // Experimental Blackwell decode path.  It writes the identical split-K
        // partial tensor as escha_matmul_dense, so finalize and its fixed slice
        // order remain unchanged.  Keep it opt-in until full model parity is
        // established on the target runtime.
        const int nct = OC/ESCHA_TILE;
        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_rows, (nct + 3)/4, n_slices), 128, 0, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const float *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
        switch (K) {
            case 2: launch(escha_matmul_dense_warp<2>); break;
            case 3: launch(escha_matmul_dense_warp<3>); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
    } else if (gen) {
        // widest slice any block gets, since lo/hi split nit unevenly by at most one tile
        const int tiles_max = (nit + n_slices - 1)/n_slices;
        const size_t smem = ESCHA_GROUPS*ESCHA_MAX_W*sizeof(uint2)
                          + (size_t) tiles_max*ESCHA_TILE*sizeof(float);
        GGML_ASSERT(smem <= 48*1024 && "escha: staged u exceeds the default shared budget");
        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_rb, n_ocb, n_slices), ESCHA_NT, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const float *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
        switch (K) {
            case 2: launch(escha_matmul_dense<2, ESCHA_ROWS_DENSE_GEN>); break;
            case 3: launch(escha_matmul_dense<3, ESCHA_ROWS_DENSE_GEN>); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
    } else if (use_cublas) {
        // Decode the packed projection once, then use the same tensor-core GEMM
        // machinery as a vanilla fp16 matrix.  p_buf is [row, out] in memory,
        // equivalently column-major [OC, rows], so the existing deterministic
        // rotation/output epilogue can consume it with n_slices == 1.
        ggml_cuda_pool_alloc<half> w_buf(ctx.pool(), (size_t) IC*OC);
        const dim3 dequant_grid(OC/ESCHA_TILE, IC/ESCHA_TILE);
        switch (K) {
            case 2:
                escha_dequant_dense_f16<2><<<dequant_grid, ESCHA_TILE*ESCHA_TILE, 0, stream>>>(
                    (const int16_t *) code->data, w_buf.get(), IC, OC);
                break;
            case 3:
                escha_dequant_dense_f16<3><<<dequant_grid, ESCHA_TILE*ESCHA_TILE, 0, stream>>>(
                    (const int16_t *) code->data, w_buf.get(), IC, OC);
                break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
        CUDA_CHECK(cudaGetLastError());

        const float alpha = 1.0f;
        const float beta  = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(
            ctx.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
            OC, n_rows, IC,
            &alpha, w_buf.get(), CUDA_R_16F, OC,
                    (const half *) u_buf.get(), CUDA_R_16F, IC,
            &beta, p_buf.get(), CUDA_R_32F, OC,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (OC % ESCHA_BN != 0) {
        // the tiled kernel blocks the output axis in exact BN steps; a ragged OC would
        // silently leave the tail columns unwritten. Every projection in this checkpoint is
        // 128-aligned, but that is a property of the model, not of the format.
        const size_t smem = ESCHA_GROUPS*ESCHA_MAX_W*sizeof(uint2)
                          + (size_t) ESCHA_ROWS_DENSE*ESCHA_TILE*sizeof(float);
        auto launch = [&](auto kernel) {
            kernel<<<dim3((n_rows + ESCHA_ROWS_DENSE - 1)/ESCHA_ROWS_DENSE, n_ocb, n_slices),
                     ESCHA_NT, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const float *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
        switch (K) {
            case 2: launch(escha_matmul_dense<2, ESCHA_ROWS_DENSE>); break;
            case 3: launch(escha_matmul_dense<3, ESCHA_ROWS_DENSE>); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
    } else if (use_wmma_bw) {
        // Consumer Blackwell fallback for the legacy ldmatrix/HMMA path.  This
        // uses the CUDA WMMA API, one warp per 16x16 weight band and four bands
        // per CTA; every split-K slice writes a disjoint partial region so the
        // existing deterministic finalize kernel remains unchanged.
        constexpr int NJ = 32;
        // The WMMA kernel assigns all work in this CTA to its sole warp.
        // Extra warps would only repeat the same tile and race the same partial
        // output, so launch a one-warp CTA until the work is explicitly sharded.
        constexpr int NWARPS = 1;
        // Two bands cuts the accumulator fragment footprint in half relative
        // to BB=4.  The larger grid is intentional: this kernel is register-
        // limited on Blackwell, and the measured 96-register shape wins.
        constexpr int BB = 2;
        const dim3 grid(OC/(BB*ESCHA_TILE), (n_rows + NJ - 1)/NJ, n_slices);
        const dim3 block(32, NWARPS);
        auto launch = [&](auto kernel) {
            kernel<<<grid, block, 0, stream>>>(
                (const int16_t *) code->data, (const half *) u_buf.get(), p_buf.get(),
                IC, OC, n_rows, n_slices);
        };
        switch (K) {
            case 2: launch((escha_matmul_dense_wmma_bw<2, NJ, NWARPS, BB>)); break;
            case 3: launch((escha_matmul_dense_wmma_bw<3, NJ, NWARPS, BB>)); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
    } else if (use_mma) {
        // tensor-core prefill. Weights are exact; activations are rounded to fp16, which is
        // what escha's runtime does. ESCHA_NO_MMA=1 falls back to the fp32 FMA kernel.
        constexpr int NTJ = ESCHA_MMA_BN/ESCHA_TILE;
        const size_t smem = NTJ*ESCHA_MAX_W*sizeof(uint2)
                          + (size_t) 2*ESCHA_MMA_BM*ESCHA_TILE*sizeof(half)
                          + (size_t) ESCHA_MMA_BN*ESCHA_TILE*sizeof(half);
        const int n_tb_mma = (n_rows + ESCHA_MMA_BM - 1)/ESCHA_MMA_BM;
        const int n_cb_mma = OC/ESCHA_MMA_BN;
        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_tb_mma, n_cb_mma, n_slices), dim3(32, 256/32), smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const half *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
        switch (K) {
            case 2: {
#ifdef ESCHA_MMA_SM120_K2_BN64_EXPERIMENT
                // P-ARCH-13 geometry-only experiment.  Only K2 changes output-tile
                // width: its grid, payload/shared-B extent, and accumulator footprint
                // follow BN=64.  Decode representation, A overlap, partial layout, and
                // the separate rotate/finalize kernels are deliberately unchanged.
                constexpr int K2_BN = 64;
                constexpr int K2_NTJ = K2_BN/ESCHA_TILE;
                const size_t k2_smem = K2_NTJ*ESCHA_MAX_W*sizeof(uint2)
                                     + (size_t) 2*ESCHA_MMA_BM*ESCHA_TILE*sizeof(half)
                                     + (size_t) K2_BN*ESCHA_TILE*sizeof(half);
                GGML_ASSERT(OC % K2_BN == 0);
                escha_matmul_dense_tiled_mma<2, ESCHA_MMA_BM, K2_BN>
                    <<<dim3(n_tb_mma, OC/K2_BN, n_slices), dim3(32, 256/32), k2_smem, stream>>>(
                        (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                        (const half *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
                break;
#else
                // PROMOTED default (EXP-04 Stage 2): mixed accumulator policy
                // (native Escha mixed policy: fp16 MMA acc for IC <= 6144, fp32
                // above), applied per projection across every K2 prefill family
                // by IC alone — NOT the rejected P-ARCH-20 single-shape toggle.
                // Geometry, staging, decode, A-stage overlap, partial layout,
                // and finalize are byte-for-byte unchanged; only the MMA
                // accumulator type is selected by the threshold.
                if (IC <= 6144) {
                    launch((escha_matmul_dense_tiled_mma<2, ESCHA_MMA_BM, ESCHA_MMA_BN, true>));
                } else {
                    launch((escha_matmul_dense_tiled_mma<2, ESCHA_MMA_BM, ESCHA_MMA_BN, false>));
                }
                break;
#endif // ESCHA_MMA_SM120_K2_BN64_EXPERIMENT
            }
            case 3: {
                // PROMOTED default (EXP-04 Stage 2): mixed accumulator policy
                // (native Escha mixed policy: fp16 MMA acc for IC <= 6144, fp32
                // above), applied per projection across every K3 prefill family
                // by IC alone — NOT the rejected P-ARCH-20 single-shape toggle.
                // Geometry, staging, decode, A-stage overlap, partial layout,
                // and finalize are byte-for-byte unchanged; only the MMA
                // accumulator type is selected by the threshold.
                if (IC <= 6144) {
                    launch((escha_matmul_dense_tiled_mma<3, ESCHA_MMA_BM, ESCHA_MMA_BN, true>));
                } else {
                    launch((escha_matmul_dense_tiled_mma<3, ESCHA_MMA_BM, ESCHA_MMA_BN, false>));
                }
                break;
            }
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
    } else {
        constexpr int NT  = (ESCHA_BM/ESCHA_TM)*(ESCHA_BN/ESCHA_TN);
        constexpr int NTJ = ESCHA_BN/ESCHA_TILE;
        const size_t smem = NTJ*ESCHA_MAX_W*sizeof(uint32_t)
                          + (size_t) ESCHA_TILE*ESCHA_BN*sizeof(float)
                          + (size_t) 2*ESCHA_TILE*ESCHA_BM*sizeof(float);
        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_tb, n_cb, n_slices), NT, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const float *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
        switch (K) {
            case 2: launch((escha_matmul_dense_tiled<2, ESCHA_BM, ESCHA_BN, ESCHA_TM, ESCHA_TN>)); break;
            case 3: launch((escha_matmul_dense_tiled<3, ESCHA_BM, ESCHA_BN, ESCHA_TM, ESCHA_TN>)); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    if (profile) {
        CUDA_CHECK(cudaEventRecord(profile_matmul_stop, stream));
    }

    {
        escha_finalize_dense<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
            (const half *) rout->data, p_buf.get(), (float *) dst->data,
            OC, (int) x->ne[1], n_rows, n_slices, dst->nb[1], dst->nb[2]);
        CUDA_CHECK(cudaGetLastError());
    }

    if (getenv("ESCHA_DEBUG_NAN") != nullptr) {
        static int call = 0;
        const int debug_call = call++;
        const int64_t n = ggml_nelements(dst);
        int * dev_bad = nullptr;
        int host_bad = -1;
        const size_t x_s1 = x->nb[1]/sizeof(float);
        const size_t x_s2 = x->nb[2]/sizeof(float);
        const size_t x_span = (size_t) (x->ne[1] - 1)*x_s1 + (size_t) (x->ne[2] - 1)*x_s2 + IC;
        std::vector<float> host_x(x_span);
        CUDA_CHECK(cudaMalloc(&dev_bad, sizeof(*dev_bad)));
        CUDA_CHECK(cudaMemcpyAsync(dev_bad, &host_bad, sizeof(host_bad), cudaMemcpyHostToDevice, stream));
        escha_debug_find_nonfinite<<<(unsigned) ((n + 255)/256), 256, 0, stream>>>(
                (const float *) dst->data, n, dev_bad);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(host_x.data(), x->data, host_x.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&host_bad, dev_bad, sizeof(host_bad), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(dev_bad));
        int input_bad = -1;
        float input_max = 0.0f;
        for (int ir = 0; ir < n_rows; ++ir) {
            const int i1 = ir % x->ne[1];
            const int i2 = ir / x->ne[1];
            const float * row = host_x.data() + (size_t) i1*x_s1 + (size_t) i2*x_s2;
            for (int k = 0; k < IC; ++k) {
                if (!std::isfinite(row[k]) && input_bad < 0) {
                    input_bad = ir*IC + k;
                }
                input_max = fmaxf(input_max, fabsf(row[k]));
            }
        }
        fprintf(stderr, "ESCHA_DEBUG_NAN call=%d IC=%d OC=%d rows=%d x_nb1=%zu x_nb2=%zu input_bad=%d input_max=%g output_bad=%d\n",
                debug_call, IC, OC, n_rows, x->nb[1], x->nb[2], input_bad, input_max, host_bad);
    }
    if (const char * capture_dir = getenv("ESCHA_CAPTURE_DST_DIR")) {
        // EXP-04 Stage 3 debug capture: dump the final fp32 output of the
        // bounded-K target family for numerical comparison (rel-RMS/max-abs/
        // cosine/NaN-Inf) across arms.  Env-gated, never in timed runs, and
        // the capture itself adds no code to the hot kernel path.
        if (!gen && use_mma && K == 3 && IC == 17408 && OC == 5120 && n_rows == 2048) {
            static bool captured = false;
            if (!captured) {
                captured = true;
                const int64_t n = ggml_nelements(dst);
                std::vector<float> host_dst(n);
                CUDA_CHECK(cudaMemcpyAsync(host_dst.data(), dst->data, n*sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                escha_capture_parch01_blob(capture_dir, "dst.f32.bin", host_dst.data(),
                                           host_dst.size()*sizeof(float));
                fprintf(stderr, "ESCHA_CAPTURE_DST K=%d IC=%d OC=%d rows=%d n=%lld written=%s/dst.f32.bin\n",
                        K, IC, OC, n_rows, (long long) n, capture_dir);
            }
        }
    }
    finish_profile();
}
