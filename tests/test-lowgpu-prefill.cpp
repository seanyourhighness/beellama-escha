#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

// Exercise the n_tokens > 4 CUDA path.  Values are integral so the reference
// also exactly represents the F32 -> F16 activation boundary used by GEMM.
int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        fprintf(stderr, "no GPU backend available, skipping\\n");
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) return 1;

    ggml_init_params params = { 1u << 20, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 1;

    constexpr int H = 128;
    constexpr int V = 17;
    constexpr int T = 6;
    constexpr int KB = H*3/8;
    ggml_tensor * code  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8,  KB, V);
    ggml_tensor * scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1,  V);
    ggml_tensor * zp    = ggml_new_tensor_2d(ctx, GGML_TYPE_I8,  1,  V);
    ggml_tensor * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H,  T);
    ggml_tensor * y     = ggml_lowgpu_mul_mat(ctx, code, scale, zp, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 1;

    std::vector<uint8_t> hc((size_t) KB*V), hz(V, 3);
    std::vector<ggml_fp16_t> hs(V, 0x3c00); // 1.0
    std::vector<float> hx((size_t) H*T), expected((size_t) V*T);
    for (int r = 0; r < V; ++r) {
        for (int i = 0; i < H; i += 8) {
            const uint8_t q0 = (r + i + 0) & 7, q1 = (r + i + 1) & 7;
            const uint8_t q2 = (r + i + 2) & 7, q3 = (r + i + 3) & 7;
            const uint8_t q4 = (r + i + 4) & 7, q5 = (r + i + 5) & 7;
            const uint8_t q6 = (r + i + 6) & 7, q7 = (r + i + 7) & 7;
            const int o = r*KB + i*3/8;
            hc[o + 0] = q0 | (q1 << 3) | ((q2 & 3) << 6);
            hc[o + 1] = (q2 >> 2) | (q3 << 1) | (q4 << 4) | ((q5 & 1) << 7);
            hc[o + 2] = (q5 >> 1) | (q6 << 2) | (q7 << 5);
        }
    }
    for (int t = 0; t < T; ++t) for (int i = 0; i < H; ++i) hx[t*H + i] = ((t + i) & 1) ? -1.0f : 1.0f;
    for (int r = 0; r < V; ++r) for (int t = 0; t < T; ++t) for (int i = 0; i < H; ++i) {
        const int q = (r + i) & 7;
        expected[t*V + r] += (q - 3)*hx[t*H + i];
    }
    ggml_backend_tensor_set(code, hc.data(), 0, hc.size());
    ggml_backend_tensor_set(scale, hs.data(), 0, hs.size()*sizeof(hs[0]));
    ggml_backend_tensor_set(zp, hz.data(), 0, hz.size());
    ggml_backend_tensor_set(x, hx.data(), 0, hx.size()*sizeof(hx[0]));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 1;
    std::vector<float> got(expected.size());
    ggml_backend_tensor_get(y, got.data(), 0, got.size()*sizeof(got[0]));
    for (size_t i = 0; i < got.size(); ++i) if (std::fabs(got[i] - expected[i]) > 1e-3f) {
        fprintf(stderr, "lowgpu prefill mismatch at %zu: got %g expected %g\\n", i, got[i], expected[i]);
        return 1;
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
