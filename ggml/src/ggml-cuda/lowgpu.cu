#include "lowgpu.cuh"

#include <cstdio>
#include <vector>

// LowGPU v1 3-bit vocab kernels (see source/lowgpu/format.py for the packing):
//   * 3 bits/code, 8 levels, group size 128 along the hidden dim;
//   * x_hat = (q - zp) * scale -> fp16;
//   * 8 codes packed per 3 bytes, little-endian bit order.

#define LOWGPU_GROUP 128

// decode one vocab row into fp16 [n_embd]
static __global__ void lowgpu_dequant_row_kernel(
        const uint8_t * __restrict__ codes,
        const half    * __restrict__ scales,
        const uint8_t * __restrict__ zps,
        half          * __restrict__ out,
        const int KB, const int G, const int ntri) {
    const int tid = threadIdx.x;
    const int r   = blockIdx.x;

    const uint8_t * cr = codes + (size_t) r*KB;
    const half    * sr = scales + (size_t) r*G;
    const uint8_t * zr = zps    + (size_t) r*G;
    half          * orow = out + (size_t) r*(KB*8/3);

    for (int t = tid; t < ntri; t += blockDim.x) {
        const uint8_t b0 = cr[3*t + 0];
        const uint8_t b1 = cr[3*t + 1];
        const uint8_t b2 = cr[3*t + 2];

        const uint8_t c0 = b0 & 7;
        const uint8_t c1 = (b0 >> 3) & 7;
        const uint8_t c2 = ((b0 >> 6) | (b1 << 2)) & 7;
        const uint8_t c3 = (b1 >> 1) & 7;
        const uint8_t c4 = (b1 >> 4) & 7;
        const uint8_t c5 = ((b1 >> 7) | (b2 << 1)) & 7;
        const uint8_t c6 = (b2 >> 2) & 7;
        const uint8_t c7 = (b2 >> 5) & 7;

        const int g = t*8/LOWGPU_GROUP;
        const float s = __half2float(sr[g]);
        const float z = (float) zr[g];

        orow[8*t + 0] = __float2half(((float) c0 - z) * s);
        orow[8*t + 1] = __float2half(((float) c1 - z) * s);
        orow[8*t + 2] = __float2half(((float) c2 - z) * s);
        orow[8*t + 3] = __float2half(((float) c3 - z) * s);
        orow[8*t + 4] = __float2half(((float) c4 - z) * s);
        orow[8*t + 5] = __float2half(((float) c5 - z) * s);
        orow[8*t + 6] = __float2half(((float) c6 - z) * s);
        orow[8*t + 7] = __float2half(((float) c7 - z) * s);
    }
}

// decode the row selected by ids[token] directly into dst[token]
static __global__ void lowgpu_get_rows_kernel(
        const uint8_t * __restrict__ codes,
        const half    * __restrict__ scales,
        const uint8_t * __restrict__ zps,
        const int32_t * __restrict__ ids,
        float         * __restrict__ out,
        const int KB, const int G, const int ntri, const int n_embd, const int V) {
    const int tid = threadIdx.x;
    const int tok = blockIdx.x;

    const int r = ids[tok];
    if (r < 0 || r >= V) {
        return;
    }

    const uint8_t * cr = codes + (size_t) r*KB;
    const half    * sr = scales + (size_t) r*G;
    const uint8_t * zr = zps    + (size_t) r*G;
    float         * orow = out + (size_t) tok*n_embd;

    for (int t = tid; t < ntri; t += blockDim.x) {
        const uint8_t b0 = cr[3*t + 0];
        const uint8_t b1 = cr[3*t + 1];
        const uint8_t b2 = cr[3*t + 2];

        const uint8_t c0 = b0 & 7;
        const uint8_t c1 = (b0 >> 3) & 7;
        const uint8_t c2 = ((b0 >> 6) | (b1 << 2)) & 7;
        const uint8_t c3 = (b1 >> 1) & 7;
        const uint8_t c4 = (b1 >> 4) & 7;
        const uint8_t c5 = ((b1 >> 7) | (b2 << 1)) & 7;
        const uint8_t c6 = (b2 >> 2) & 7;
        const uint8_t c7 = (b2 >> 5) & 7;

        const int g = t*8/LOWGPU_GROUP;
        const float s = __half2float(sr[g]);
        const float z = (float) zr[g];

        orow[8*t + 0] = __half2float(__float2half(((float) c0 - z) * s));
        orow[8*t + 1] = __half2float(__float2half(((float) c1 - z) * s));
        orow[8*t + 2] = __half2float(__float2half(((float) c2 - z) * s));
        orow[8*t + 3] = __half2float(__float2half(((float) c3 - z) * s));
        orow[8*t + 4] = __half2float(__float2half(((float) c4 - z) * s));
        orow[8*t + 5] = __half2float(__float2half(((float) c5 - z) * s));
        orow[8*t + 6] = __half2float(__float2half(((float) c6 - z) * s));
        orow[8*t + 7] = __half2float(__float2half(((float) c7 - z) * s));
    }
}

void ggml_cuda_op_lowgpu_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code  = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * zp    = dst->src[2];
    const ggml_tensor * ids   = dst->src[3];

    const int KB = code->ne[0];
    const int V  = code->ne[1];
    const int G  = scale->ne[0];
    const int n_ids = ids->ne[0];
    const int ntri  = KB/3;
    const int n_embd = KB*8/3;

    const bool profile = getenv("LOWGPU_PROFILE") != nullptr;
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop  = nullptr;
    if (profile) {
        CUDA_CHECK(cudaEventCreate(&profile_start));
        CUDA_CHECK(cudaEventCreate(&profile_stop));
        CUDA_CHECK(cudaEventRecord(profile_start, ctx.stream()));
    }

    dim3 grid(n_ids, 1, 1);
    lowgpu_get_rows_kernel<<<grid, 256, 0, ctx.stream()>>>(
            (const uint8_t *) code->data,
            (const half *)    scale->data,
            (const uint8_t *) zp->data,
            (const int32_t *) ids->data,
            (float *) dst->data,
            KB, G, ntri, n_embd, V);

    if (profile) {
        CUDA_CHECK(cudaEventRecord(profile_stop, ctx.stream()));
        CUDA_CHECK(cudaEventSynchronize(profile_stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, profile_start, profile_stop));
        fprintf(stderr, "LOWGPU_PROFILE get_rows vocab=%d embd=%d tokens=%d ms=%.4f\n",
                V, n_embd, n_ids, elapsed_ms);
        CUDA_CHECK(cudaEventDestroy(profile_start));
        CUDA_CHECK(cudaEventDestroy(profile_stop));
    }
}

static __global__ void lowgpu_f32_to_f16_kernel(
        const float * __restrict__ in, half * __restrict__ out, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(in[i]);
    }
}

// Decode one packed vocabulary row and dot it with one hidden vector in the
// same block.  Materializing all V*hidden weights for a single decode token
// costs 2.5 GiB of transient writes before GEMM can start; this path reads the
// compact representation once and writes only one FP32 logit per row.
static __global__ void lowgpu_packed_gemv_kernel(
        const uint8_t * __restrict__ codes,
        const half    * __restrict__ scales,
        const uint8_t * __restrict__ zps,
        const float   * __restrict__ x,
        float         * __restrict__ dst,
        const int KB, const int G, const int ntri, const int n_embd, const int V) {
    const int row = blockIdx.x;
    const int tok = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= V) {
        return;
    }

    const uint8_t * cr = codes  + (size_t) row*KB;
    const half    * sr = scales + (size_t) row*G;
    const uint8_t * zr = zps    + (size_t) row*G;
    const float   * xv = x + (size_t) tok*n_embd;

    float sum = 0.0f;
    for (int t = tid; t < ntri; t += blockDim.x) {
        const uint8_t b0 = cr[3*t + 0];
        const uint8_t b1 = cr[3*t + 1];
        const uint8_t b2 = cr[3*t + 2];
        const uint8_t q[8] = {
            (uint8_t) (b0 & 7), (uint8_t) ((b0 >> 3) & 7),
            (uint8_t) (((b0 >> 6) | (b1 << 2)) & 7), (uint8_t) ((b1 >> 1) & 7),
            (uint8_t) ((b1 >> 4) & 7), (uint8_t) (((b1 >> 7) | (b2 << 1)) & 7),
            (uint8_t) ((b2 >> 2) & 7), (uint8_t) ((b2 >> 5) & 7),
        };
        const float scale_v = __half2float(sr[t*8/LOWGPU_GROUP]);
        const float zp_v = zr[t*8/LOWGPU_GROUP];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // Match the existing GEMM path's FP32 -> FP16 activation boundary.
            const float a = __half2float(__float2half(xv[8*t + i]));
            sum += ((float) q[i] - zp_v)*scale_v*a;
        }
    }

    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dst[(size_t) tok*V + row] = partial[0];
    }
}

void ggml_cuda_op_lowgpu_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code  = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * zp    = dst->src[2];
    const ggml_tensor * x     = dst->src[3];

    const int KB = code->ne[0];
    const int V  = code->ne[1];
    const int G  = scale->ne[0];
    const int n_embd = KB*8/3;
    const int n_tokens = x->ne[1]*x->ne[2];
    const int ntri = KB/3;
    const bool profile = getenv("LOWGPU_PROFILE") != nullptr;
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop  = nullptr;
    if (profile) {
        CUDA_CHECK(cudaEventCreate(&profile_start));
        CUDA_CHECK(cudaEventCreate(&profile_stop));
        CUDA_CHECK(cudaEventRecord(profile_start, ctx.stream()));
    }
    const auto finish_profile = [&]() {
        if (!profile) {
            return;
        }
        CUDA_CHECK(cudaEventRecord(profile_stop, ctx.stream()));
        CUDA_CHECK(cudaEventSynchronize(profile_stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, profile_start, profile_stop));
        fprintf(stderr, "LOWGPU_PROFILE vocab=%d embd=%d tokens=%d ms=%.4f\n",
                V, n_embd, n_tokens, elapsed_ms);
        CUDA_CHECK(cudaEventDestroy(profile_start));
        CUDA_CHECK(cudaEventDestroy(profile_stop));
    };

    // Autoregressive decode dominates serving.  Fusing unpack + dot avoids a
    // full dequantized vocabulary allocation and its bandwidth cost.  Prefill
    // uses GEMM, but it must not expand the entire vocabulary: that alone is
    // 2.37 GiB for this 248k x 5120 model and defeats its 12 GiB deployment.
    if (n_tokens <= 4) {
        lowgpu_packed_gemv_kernel<<<dim3(V, n_tokens), 256, 0, ctx.stream()>>>(
                (const uint8_t *) code->data,
                (const half *)    scale->data,
                (const uint8_t *) zp->data,
                (const float *)   x->data,
                (float *)         dst->data,
                KB, G, ntri, n_embd, V);
        CUDA_CHECK(cudaGetLastError());
        finish_profile();
        return;
    }

    // activations to fp16 (same-type GEMM; mixed F16/F32 GemmEx is unsupported on sm_120)
    ggml_cuda_pool_alloc<half> xh_alloc(ctx.pool(), (size_t) n_tokens*n_embd);
    half * xh = xh_alloc.ptr;
    const int64_t nx = (int64_t) n_tokens*n_embd;
    const int64_t nblocks = (nx + 255)/256;
    lowgpu_f32_to_f16_kernel<<<(int) nblocks, 256, 0, ctx.stream()>>>(
            (const float *) x->data, xh, nx);

    // dst[V, n_tokens] = W^T @ x, W row-major [n_embd, V]
    // CUBLAS_COMPUTE_32F makes alpha/beta FP32 host scalars.  Passing half
    // pointers here lets cuBLAS read four bytes from two unrelated half
    // values, which scales every logit toward zero on Blackwell.
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    // Dequantize a small row stripe immediately before its GEMM.  This keeps
    // source weights packed in VRAM and caps temporary weight storage at 40
    // MiB, while preserving the exact fp16 weight/activation boundary of the
    // former full-vocabulary path.
    constexpr int LOWGPU_PREFILL_ROWS = 4096;
    const uint8_t * code_d = (const uint8_t *) code->data;
    const half    * scale_d = (const half *) scale->data;
    const uint8_t * zp_d = (const uint8_t *) zp->data;
    float * dst_d = (float *) dst->data;
    ggml_cuda_pool_alloc<half> w_alloc(ctx.pool(), (size_t) LOWGPU_PREFILL_ROWS*n_embd);
    for (int row0 = 0; row0 < V; row0 += LOWGPU_PREFILL_ROWS) {
        const int rows = MIN(LOWGPU_PREFILL_ROWS, V - row0);
        half * w = w_alloc.ptr;
        lowgpu_dequant_row_kernel<<<rows, 256, 0, ctx.stream()>>>(
                code_d  + (size_t) row0*KB,
                scale_d + (size_t) row0*G,
                zp_d    + (size_t) row0*G,
                w, KB, G, ntri);
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasGemmEx(
                ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                rows, n_tokens, n_embd,
                &alpha, w, CUDA_R_16F, n_embd,
                        xh, CUDA_R_16F, n_embd,
                &beta, dst_d + row0, CUDA_R_32F, V,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    finish_profile();

    // A very small, opt-in end-to-end check for the packed-vocabulary path.
    // It is deliberately host-side and only samples four rows, so it never
    // materializes a dense model or changes normal inference behavior.
    if (getenv("LOWGPU_DEBUG_VERIFY") != nullptr && n_tokens > 0) {
        constexpr int samples = 4;
        std::vector<float> hx(n_embd);
        std::vector<uint8_t> hcode((size_t) samples*KB);
        std::vector<half> hscale((size_t) samples*G);
        std::vector<uint8_t> hzp((size_t) samples*G);
        std::vector<float> hdst((size_t) samples*n_tokens);

        CUDA_CHECK(cudaMemcpyAsync(hx.data(), x->data, (size_t) n_embd*sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaMemcpyAsync(hcode.data(), code->data, hcode.size()*sizeof(uint8_t), cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaMemcpyAsync(hscale.data(), scale->data, hscale.size()*sizeof(half), cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaMemcpyAsync(hzp.data(), zp->data, hzp.size()*sizeof(uint8_t), cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaMemcpyAsync(hdst.data(), dst->data, hdst.size()*sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

        for (int r = 0; r < samples; ++r) {
            float expected = 0.0f;
            for (int t = 0; t < ntri; ++t) {
                const uint8_t b0 = hcode[(size_t) r*KB + 3*t + 0];
                const uint8_t b1 = hcode[(size_t) r*KB + 3*t + 1];
                const uint8_t b2 = hcode[(size_t) r*KB + 3*t + 2];
                const uint8_t q[8] = {
                    (uint8_t) (b0 & 7), (uint8_t) ((b0 >> 3) & 7),
                    (uint8_t) (((b0 >> 6) | (b1 << 2)) & 7), (uint8_t) ((b1 >> 1) & 7),
                    (uint8_t) ((b1 >> 4) & 7), (uint8_t) (((b1 >> 7) | (b2 << 1)) & 7),
                    (uint8_t) ((b2 >> 2) & 7), (uint8_t) ((b2 >> 5) & 7),
                };
                const float s = __half2float(hscale[(size_t) r*G + t*8/LOWGPU_GROUP]);
                const float z = hzp[(size_t) r*G + t*8/LOWGPU_GROUP];
                for (int i = 0; i < 8; ++i) {
                    const float wv = __half2float(__float2half(((float) q[i] - z)*s));
                    const float xv = __half2float(__float2half(hx[8*t + i]));
                    expected += wv*xv;
                }
            }
            fprintf(stderr, "LOWGPU_DEBUG_VERIFY row=%d gpu=%+.7g host=%+.7g delta=%+.7g\\n",
                    r, hdst[(size_t) r*n_tokens], expected, hdst[(size_t) r*n_tokens] - expected);
        }
    }
}
