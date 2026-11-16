#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "config.h"

__device__ __forceinline__ float block_sum(float v, float *red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int s = (int)(blockDim.x >> 1); s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float r = red[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_max(float v, float *red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int s = (int)(blockDim.x >> 1); s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    float r = red[0];
    __syncthreads();
    return r;
}

__global__ void k_embed(float *x, const float *wte, const int *tok, int n, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * D) return;
    int r = i / D;
    int d = i - r * D;
    x[i] = wte[(size_t)tok[r] * D + d];
}

__global__ void k_embed_bwd(float *dwte, const float *dx, const int *tok, int n, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * D) return;
    int r = i / D;
    int d = i - r * D;
    atomicAdd(&dwte[(size_t)tok[r] * D + d], dx[i]);
}

__global__ void k_rms_fwd(float *out, float *inv, const float *in, const float *g, int D) {
    __shared__ float red[LN_THREADS];
    int row = blockIdx.x;
    const float *x = in + (size_t)row * D;
    float s = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += x[i] * x[i];
    float ss = block_sum(s, red) / D;
    float r = 1.0f / sqrtf(ss + 1e-5f);
    float *o = out + (size_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) o[i] = x[i] * r * g[i];
    if (threadIdx.x == 0) inv[row] = r;
}

__global__ void k_rms_bwd_dx(float *dx, const float *dout, const float *in, const float *inv,
                             const float *g, int D) {
    __shared__ float red[LN_THREADS];
    int row = blockIdx.x;
    const float *x = in + (size_t)row * D;
    const float *dy = dout + (size_t)row * D;
    float *dxr = dx + (size_t)row * D;
    float r = inv[row];
    float a = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) a += dy[i] * g[i] * x[i] * r;
    float c = block_sum(a, red) / D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float xhat = x[i] * r;
        dxr[i] += r * (dy[i] * g[i] - xhat * c);
    }
}

__global__ void k_rms_param_grad(float *dg, const float *dout, const float *in, const float *inv,
                                 int rows, int D, int rpb) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    int r0 = blockIdx.y * rpb;
    int r1 = r0 + rpb < rows ? r0 + rpb : rows;
    float sg = 0.0f;
    for (int r = r0; r < r1; r++) sg += dout[(size_t)r * D + c] * in[(size_t)r * D + c] * inv[r];
    atomicAdd(&dg[c], sg);
}

__global__ void k_rope(float *qkv, const float *cs, const float *sn, int n, int tl, float sign) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int per_row = 2 * N_HEAD * HALF_HEAD;
    if (i >= n * per_row) return;
    int r = i / per_row;
    int rem = i - r * per_row;
    int hh = rem / HALF_HEAD;
    int j = rem - hh * HALF_HEAD;
    int pos = r % tl;
    float c = cs[pos * HALF_HEAD + j];
    float s = sign * sn[pos * HALF_HEAD + j];
    float *p = qkv + (size_t)r * 3 * N_EMBD + hh * HEAD_SIZE + j;
    float x1 = p[0], x2 = p[HALF_HEAD];
    p[0] = x1 * c - x2 * s;
    p[HALF_HEAD] = x1 * s + x2 * c;
}

__global__ void k_swiglu_fwd(float *hact, const float *h13, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t r = i / FF_DIM;
    size_t c = i - r * FF_DIM;
    float g = h13[r * 2 * FF_DIM + c];
    float u = h13[r * 2 * FF_DIM + FF_DIM + c];
    float s = 1.0f / (1.0f + expf(-g));
    hact[i] = g * s * u;
}

__global__ void k_swiglu_bwd(float *dh13, const float *dhact, const float *h13, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t r = i / FF_DIM;
    size_t c = i - r * FF_DIM;
    float g = h13[r * 2 * FF_DIM + c];
    float u = h13[r * 2 * FF_DIM + FF_DIM + c];
    float s = 1.0f / (1.0f + expf(-g));
    float d = dhact[i];
    dh13[r * 2 * FF_DIM + c] = d * u * s * (1.0f + g * (1.0f - s));
    dh13[r * 2 * FF_DIM + FF_DIM + c] = d * g * s;
}

// Inverted-dropout mask as a pure function of (seed, index), so backward recomputes it instead of storing it.
__device__ __forceinline__ float drop_scale(unsigned seed, size_t idx, float p) {
    unsigned x = (unsigned)idx * 2654435761u + seed * 2246822519u + 374761393u;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    float u = (float)(x >> 8) * (1.0f / 16777216.0f);
    return u < p ? 0.0f : 1.0f / (1.0f - p);
}

__global__ void k_dropout_apply(float *dst, const float *src, size_t n, float p, unsigned seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i] * drop_scale(seed, i, p);
}

#define GEMM_BK 8

struct GemmArgs {
    const float *A;
    const float *B;
    const float *res;
    float *C;
    int M, N, K;
    int lda, ldb, ldc;
    long long sa_o, sa_i, sb_o, sb_i, sc_o, sc_i;
    int inner;
    int acc;
    float dp;
    unsigned dseed;
    int vec_a, vec_b;
};

__device__ __forceinline__ float4 load4(const float *p, int avail, int vec) {
    float4 v;
    if (avail >= 4 && vec) {
        v = *reinterpret_cast<const float4 *>(p);
    } else {
        v.x = avail > 0 ? p[0] : 0.0f;
        v.y = avail > 1 ? p[1] : 0.0f;
        v.z = avail > 2 ? p[2] : 0.0f;
        v.w = avail > 3 ? p[3] : 0.0f;
    }
    return v;
}

template <int BM, bool TA, int NV>
__device__ __forceinline__ void gemm_fetch_a(float4 *r, const float *A, const GemmArgs &g, int tid, int m0, int k0) {
#pragma unroll
    for (int i = 0; i < NV; i++) {
        const int idx = tid + i * 256;
        float4 v;
        v.x = v.y = v.z = v.w = 0.0f;
        if ((BM * GEMM_BK / 4) % 256 == 0 || idx < BM * GEMM_BK / 4) {
            if (TA) {
                const int kk = idx / (BM / 4), gm = m0 + (idx % (BM / 4)) * 4, gk = k0 + kk;
                v = load4(A + (long long)gk * g.lda + gm, gk < g.K ? g.M - gm : 0, g.vec_a);
            } else {
                const int gm = m0 + (idx >> 1), gk = k0 + (idx & 1) * 4;
                v = load4(A + (long long)gm * g.lda + gk, gm < g.M ? g.K - gk : 0, g.vec_a);
            }
        }
        r[i] = v;
    }
}

template <int BN, bool TB, int NV>
__device__ __forceinline__ void gemm_fetch_b(float4 *r, const float *B, const GemmArgs &g, int tid, int n0, int k0) {
#pragma unroll
    for (int i = 0; i < NV; i++) {
        const int idx = tid + i * 256;
        float4 v;
        v.x = v.y = v.z = v.w = 0.0f;
        if ((BN * GEMM_BK / 4) % 256 == 0 || idx < BN * GEMM_BK / 4) {
            if (TB) {
                const int gn = n0 + (idx >> 1), gk = k0 + (idx & 1) * 4;
                v = load4(B + (long long)gn * g.ldb + gk, gn < g.N ? g.K - gk : 0, g.vec_b);
            } else {
                const int kk = idx / (BN / 4), gn = n0 + (idx % (BN / 4)) * 4, gk = k0 + kk;
                v = load4(B + (long long)gk * g.ldb + gn, gk < g.K ? g.N - gn : 0, g.vec_b);
            }
        }
        r[i] = v;
    }
}

template <int BM, bool TA, int NV>
__device__ __forceinline__ void gemm_commit_a(float (*S)[BM + 4], const float4 *r, int tid) {
#pragma unroll
    for (int i = 0; i < NV; i++) {
        const int idx = tid + i * 256;
        if ((BM * GEMM_BK / 4) % 256 != 0 && idx >= BM * GEMM_BK / 4) continue;
        if (TA) {
            *reinterpret_cast<float4 *>(&S[idx / (BM / 4)][(idx % (BM / 4)) * 4]) = r[i];
        } else {
            const int m = idx >> 1, k4 = (idx & 1) * 4;
            S[k4][m] = r[i].x;
            S[k4 + 1][m] = r[i].y;
            S[k4 + 2][m] = r[i].z;
            S[k4 + 3][m] = r[i].w;
        }
    }
}

template <int BN, bool TB, int NV>
__device__ __forceinline__ void gemm_commit_b(float (*S)[BN + 4], const float4 *r, int tid) {
#pragma unroll
    for (int i = 0; i < NV; i++) {
        const int idx = tid + i * 256;
        if ((BN * GEMM_BK / 4) % 256 != 0 && idx >= BN * GEMM_BK / 4) continue;
        if (TB) {
            const int n = idx >> 1, k4 = (idx & 1) * 4;
            S[k4][n] = r[i].x;
            S[k4 + 1][n] = r[i].y;
            S[k4 + 2][n] = r[i].z;
            S[k4 + 3][n] = r[i].w;
        } else {
            *reinterpret_cast<float4 *>(&S[idx / (BN / 4)][(idx % (BN / 4)) * 4]) = r[i];
        }
    }
}

// C = A * B on 16*TM x 16*TN output tiles; 256 threads, each computes TM x TN outputs. TA/TB select transposed operands.
// CAUSAL: 0 dense, 1 skip tiles above the diagonal (scores), 2 A is lower-triangular (k range ends at the row tile),
// 3 A^T is lower-triangular (k range starts at the row tile). Modes 2 and 3 only skip work that is exactly zero.
// Global loads for the next k step are staged in registers and stored to the other shared buffer: one barrier per step.
template <int TM, int TN, bool TA, bool TB, int CAUSAL>
__global__ void __launch_bounds__(256) k_gemm(GemmArgs g) {
    constexpr int BM = 16 * TM, BN = 16 * TN, BK = GEMM_BK;
    constexpr int NA = (BM * BK / 4 + 255) / 256, NB = (BN * BK / 4 + 255) / 256;
    constexpr int SEG_M = BM / (TM / 4), SEG_N = BN / (TN / 4);
    __shared__ float As[2][BK][BM + 4];
    __shared__ float Bs[2][BK][BN + 4];

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    if (CAUSAL == 1 && n0 > m0 + BM - 1) return;

    const int zo = (int)blockIdx.z / g.inner, zi = (int)blockIdx.z - zo * g.inner;
    const float *A = g.A + (long long)zo * g.sa_o + (long long)zi * g.sa_i;
    const float *B = g.B + (long long)zo * g.sb_o + (long long)zi * g.sb_i;
    const long long c_off = (long long)zo * g.sc_o + (long long)zi * g.sc_i;
    float *C = g.C + c_off;
    const float *R = g.res ? g.res + c_off : (const float *)0;

    int kbeg = 0, kend = g.K;
    if (CAUSAL == 2 && m0 + BM < kend) kend = m0 + BM;
    if (CAUSAL == 3) kbeg = m0;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    float4 ra[NA], rb[NB];
    if (kbeg < kend) {
        gemm_fetch_a<BM, TA, NA>(ra, A, g, tid, m0, kbeg);
        gemm_fetch_b<BN, TB, NB>(rb, B, g, tid, n0, kbeg);
        gemm_commit_a<BM, TA, NA>(As[0], ra, tid);
        gemm_commit_b<BN, TB, NB>(Bs[0], rb, tid);
        __syncthreads();
    }
    for (int k0 = kbeg, cur = 0; k0 < kend; k0 += BK, cur ^= 1) {
        const bool more = k0 + BK < kend;
        if (more) {
            gemm_fetch_a<BM, TA, NA>(ra, A, g, tid, m0, k0 + BK);
            gemm_fetch_b<BN, TB, NB>(rb, B, g, tid, n0, k0 + BK);
        }
#pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a[TM], b[TN];
#pragma unroll
            for (int s = 0; s < TM / 4; s++) {
                const float4 v = *reinterpret_cast<const float4 *>(&As[cur][kk][s * SEG_M + ty * 4]);
                a[4 * s] = v.x;
                a[4 * s + 1] = v.y;
                a[4 * s + 2] = v.z;
                a[4 * s + 3] = v.w;
            }
#pragma unroll
            for (int s = 0; s < TN / 4; s++) {
                const float4 v = *reinterpret_cast<const float4 *>(&Bs[cur][kk][s * SEG_N + tx * 4]);
                b[4 * s] = v.x;
                b[4 * s + 1] = v.y;
                b[4 * s + 2] = v.z;
                b[4 * s + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i] * b[j];
        }
        if (more) {
            gemm_commit_a<BM, TA, NA>(As[cur ^ 1], ra, tid);
            gemm_commit_b<BN, TB, NB>(Bs[cur ^ 1], rb, tid);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        const int row = m0 + (i >> 2) * SEG_M + ty * 4 + (i & 3);
        if (row >= g.M) continue;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            const int col = n0 + (j >> 2) * SEG_N + tx * 4 + (j & 3);
            if (col >= g.N) continue;
            const long long idx = (long long)row * g.ldc + col;
            float cv = acc[i][j];
            if (g.dp > 0.0f) cv *= drop_scale(g.dseed, (size_t)row * g.N + col, g.dp);
            const float v = cv + (R ? R[idx] : 0.0f);
            C[idx] = g.acc ? C[idx] + v : v;
        }
    }
}

__global__ void k_softmax_causal(float *p, int tl, float scale) {
    __shared__ float red[ATT_THREADS];
    const int t = blockIdx.x;
    float *row = p + (((size_t)blockIdx.z * N_HEAD + blockIdx.y) * BLOCK_SIZE + t) * BLOCK_SIZE;
    float mx = -1e30f;
    for (int s = threadIdx.x; s <= t; s += blockDim.x) mx = fmaxf(mx, row[s] * scale);
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int s = threadIdx.x; s <= t; s += blockDim.x) {
        const float e = expf(row[s] * scale - mx);
        row[s] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    const float inv = 1.0f / sum;
    for (int s = threadIdx.x; s < tl; s += blockDim.x) row[s] = s <= t ? row[s] * inv : 0.0f;
}

__global__ void k_softmax_bwd(float *ds, const float *p, int tl, float scale) {
    __shared__ float red[ATT_THREADS];
    const int t = blockIdx.x;
    const size_t off = (((size_t)blockIdx.z * N_HEAD + blockIdx.y) * BLOCK_SIZE + t) * BLOCK_SIZE;
    float *dr = ds + off;
    const float *pr = p + off;
    float w = 0.0f;
    for (int s = threadIdx.x; s <= t; s += blockDim.x) w += pr[s] * dr[s];
    w = block_sum(w, red);
    for (int s = threadIdx.x; s < tl; s += blockDim.x) dr[s] = s <= t ? pr[s] * (dr[s] - w) * scale : 0.0f;
}

__global__ void k_softmax_xent(float *logits, const int *targets, float *loss_acc, int V, float scale) {
    __shared__ float red[XENT_THREADS];
    int row = blockIdx.x;
    float *l = logits + (size_t)row * V;
    float mx = -1e30f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) mx = fmaxf(mx, l[i]);
    mx = block_max(mx, red);
    float s = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) s += expf(l[i] - mx);
    s = block_sum(s, red);
    int tgt = targets[row];
    if (threadIdx.x == 0) atomicAdd(loss_acc, logf(s) + mx - l[tgt]);
    __syncthreads();
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float p = expf(l[i] - mx) / s;
        l[i] = (p - (i == tgt ? 1.0f : 0.0f)) * scale;
    }
}

__global__ void k_sumsq(const float *g, size_t n, float *out) {
    __shared__ float red[EW_THREADS];
    float s = 0.0f;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) s += g[i] * g[i];
    s = block_sum(s, red);
    if (threadIdx.x == 0) atomicAdd(out, s);
}

__global__ void k_adamw(float *w, const float *g, float *m, float *v, float *ema, size_t n,
                        float lr, float c1, float c2, float wd, float ema_decay,
                        const float *sumsq, float clip) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f;
    float norm = sqrtf(*sumsq);
    float k = norm > clip ? clip / norm : 1.0f;
    float gi = g[i] * k;
    float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float wi = w[i];
    wi -= lr * ((mi / c1) / (sqrtf(vi / c2) + eps) + wd * wi);
    w[i] = wi;
    ema[i] = ema_decay * ema[i] + (1.0f - ema_decay) * wi;
}

__global__ void k_add(float *dst, const float *src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += src[i];
}


static int ew_blocks(size_t n) { return (int)((n + EW_THREADS - 1) / EW_THREADS); }

static int g_gemm_tile = 0;

static int aligned16(const void *p) { return ((uintptr_t)p & 15) == 0; }

template <int TM, int TN, bool TA, bool TB, int CAUSAL>
static void launch_gemm(dim3 grid, const GemmArgs &g) {
#ifdef CUDA_EMULATION
    emu_launch(grid, dim3(256), [&]() { k_gemm<TM, TN, TA, TB, CAUSAL>(g); });
#else
    k_gemm<TM, TN, TA, TB, CAUSAL><<<grid, 256>>>(g);
#endif
}

template <int TM, int TN>
static int gemm_select(dim3 grid, const GemmArgs &g, int ta, int tb, int causal) {
#define GEMM_CASE(A_, B_, C_)                                  \
    if (ta == A_ && tb == B_ && causal == C_) {                \
        launch_gemm<TM, TN, A_ == 1, B_ == 1, C_>(grid, g);    \
        return 1;                                              \
    }
    GEMM_CASE(0, 0, 0)
    GEMM_CASE(0, 1, 0)
    GEMM_CASE(1, 0, 0)
    GEMM_CASE(0, 1, 1)
    GEMM_CASE(0, 0, 2)
    GEMM_CASE(1, 0, 3)
#undef GEMM_CASE
    return 0;
}

// Tile choice: 128x128 when it fills the GPU, otherwise 64x64 (small matrices, per-head attention GEMMs).
static void gemm_run(GemmArgs g, int ta, int tb, int causal, int batch) {
    g.vec_a = g.lda % 4 == 0 && g.sa_o % 4 == 0 && g.sa_i % 4 == 0 && aligned16(g.A);
    g.vec_b = g.ldb % 4 == 0 && g.sb_o % 4 == 0 && g.sb_i % 4 == 0 && aligned16(g.B);
    const int big = g_gemm_tile ? g_gemm_tile == 128
                                : g.M >= 128 && g.N >= 128 &&
                                      ((g.M + 127) / 128) * ((g.N + 127) / 128) * batch >= GEMM_BIG_MIN_BLOCKS;
    const int t = big ? 128 : 64;
    dim3 grid((g.N + t - 1) / t, (g.M + t - 1) / t, batch);
    const int ok = big ? gemm_select<8, 8>(grid, g, ta, tb, causal) : gemm_select<4, 4>(grid, g, ta, tb, causal);
    if (!ok) {
        fprintf(stderr, "Unsupported GEMM configuration (ta=%d tb=%d causal=%d)\n", ta, tb, causal);
        exit(1);
    }
}

static void gemm(const float *A, const float *B, float *C, const float *res, int M, int N, int K, int ta, int tb,
                 int acc, float dp = 0.0f, unsigned dseed = 0, int causal = 0) {
    GemmArgs g;
    memset(&g, 0, sizeof(g));
    g.A = A;
    g.B = B;
    g.C = C;
    g.res = res;
    g.M = M;
    g.N = N;
    g.K = K;
    g.lda = ta ? M : K;
    g.ldb = tb ? K : N;
    g.ldc = N;
    g.inner = 1;
    g.acc = acc;
    g.dp = dp;
    g.dseed = dseed;
    gemm_run(g, ta, tb, causal, 1);
}

static void rms_fwd(float *out, float *inv, const float *in, const float *g, int rows) {
    LAUNCH(k_rms_fwd, rows, LN_THREADS, out, inv, in, g, N_EMBD);
}

static void rms_bwd(float *dx, float *dg, const float *dout, const float *in, const float *inv, const float *g,
                    int rows) {
    const int rpb = 128;
    LAUNCH(k_rms_bwd_dx, rows, LN_THREADS, dx, dout, in, inv, g, N_EMBD);
    dim3 grid((N_EMBD + 63) / 64, (rows + rpb - 1) / rpb);
    LAUNCH(k_rms_param_grad, grid, 64, dg, dout, in, inv, rows, N_EMBD, rpb);
}

static void rope(float *qkv, const float *cs, const float *sn, int n, int tl, float sign) {
    const size_t total = (size_t)n * 2 * N_HEAD * HALF_HEAD;
    LAUNCH(k_rope, ew_blocks(total), EW_THREADS, qkv, cs, sn, n, tl, sign);
}

// Attention as batched GEMMs over (batch, head). Q, K, V are column slices of the packed qkv rows;
// att and dscore are [batch][head][BLOCK_SIZE][BLOCK_SIZE].
static GemmArgs batched(const float *A, long long a_o, long long a_i, int lda, const float *B, long long b_o,
                        long long b_i, int ldb, float *C, long long c_o, long long c_i, int ldc, int M, int N, int K) {
    GemmArgs g;
    memset(&g, 0, sizeof(g));
    g.A = A;
    g.B = B;
    g.C = C;
    g.M = M;
    g.N = N;
    g.K = K;
    g.lda = lda;
    g.ldb = ldb;
    g.ldc = ldc;
    g.sa_o = a_o;
    g.sa_i = a_i;
    g.sb_o = b_o;
    g.sb_i = b_i;
    g.sc_o = c_o;
    g.sc_i = c_i;
    g.inner = N_HEAD;
    return g;
}

static void attn_fwd(float *out, float *att, const float *qkv, int B, int tl) {
    const long long D = N_EMBD, R = (long long)tl * 3 * D, P = (long long)BLOCK_SIZE * BLOCK_SIZE;
    const float *q = qkv, *k = qkv + D, *v = qkv + 2 * D;
    const int bh = B * N_HEAD;
    gemm_run(batched(q, R, HEAD_SIZE, 3 * D, k, R, HEAD_SIZE, 3 * D, att, N_HEAD * P, P, BLOCK_SIZE, tl, tl, HEAD_SIZE),
             0, 1, 1, bh);
    LAUNCH(k_softmax_causal, dim3(tl, N_HEAD, B), ATT_THREADS, att, tl, 1.0f / sqrtf((float)HEAD_SIZE));
    gemm_run(batched(att, N_HEAD * P, P, BLOCK_SIZE, v, R, HEAD_SIZE, 3 * D, out, (long long)tl * D, HEAD_SIZE, D, tl,
                     HEAD_SIZE, tl),
             0, 0, 2, bh);
}

static void attn_bwd(float *dqkv, float *dscore, const float *dout, const float *att, const float *qkv, int B, int tl) {
    const long long D = N_EMBD, R = (long long)tl * 3 * D, RD = (long long)tl * D, P = (long long)BLOCK_SIZE * BLOCK_SIZE;
    const float *q = qkv, *k = qkv + D, *v = qkv + 2 * D;
    const int bh = B * N_HEAD;
    gemm_run(batched(dout, RD, HEAD_SIZE, D, v, R, HEAD_SIZE, 3 * D, dscore, N_HEAD * P, P, BLOCK_SIZE, tl, tl,
                     HEAD_SIZE),
             0, 1, 1, bh);
    LAUNCH(k_softmax_bwd, dim3(tl, N_HEAD, B), ATT_THREADS, dscore, att, tl, 1.0f / sqrtf((float)HEAD_SIZE));
    gemm_run(batched(dscore, N_HEAD * P, P, BLOCK_SIZE, k, R, HEAD_SIZE, 3 * D, dqkv, R, HEAD_SIZE, 3 * D, tl,
                     HEAD_SIZE, tl),
             0, 0, 2, bh);
    gemm_run(batched(dscore, N_HEAD * P, P, BLOCK_SIZE, q, R, HEAD_SIZE, 3 * D, dqkv + D, R, HEAD_SIZE, 3 * D, tl,
                     HEAD_SIZE, tl),
             1, 0, 3, bh);
    gemm_run(batched(att, N_HEAD * P, P, BLOCK_SIZE, dout, RD, HEAD_SIZE, D, dqkv + 2 * D, R, HEAD_SIZE, 3 * D, tl,
                     HEAD_SIZE, tl),
             1, 0, 3, bh);
}


// Parameter-free RMS normalization of every q and k head (applied before RoPE). inv keeps 1/rms for backward.
__global__ void k_qknorm_fwd(float *qkv, float *inv, int n) {
    __shared__ float red[LN_THREADS];
    const int row = blockIdx.y, h = blockIdx.x;
    float *x = qkv + (size_t)row * 3 * N_EMBD + h * HEAD_SIZE;
    float s = 0.0f;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) s += x[i] * x[i];
    const float r = 1.0f / sqrtf(block_sum(s, red) / HEAD_SIZE + 1e-6f);
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) x[i] *= r;
    if (threadIdx.x == 0) inv[(size_t)row * 2 * N_HEAD + h] = r;
    (void)n;
}

// xhat is the normalized (pre-RoPE) vector, dy the gradient w.r.t. it; computed in place on dqkv.
__global__ void k_qknorm_bwd(float *dqkv, const float *qkv, const float *inv, int n) {
    __shared__ float red[LN_THREADS];
    const int row = blockIdx.y, h = blockIdx.x;
    const size_t off = (size_t)row * 3 * N_EMBD + h * HEAD_SIZE;
    float *dy = dqkv + off;
    const float *xh = qkv + off;
    const float r = inv[(size_t)row * 2 * N_HEAD + h];
    float a = 0.0f;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) a += dy[i] * xh[i];
    const float c = block_sum(a, red) / HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) dy[i] = r * (dy[i] - xh[i] * c);
    (void)n;
}

__global__ void k_cache_store(float *ck, float *cv, const float *qkv, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * N_EMBD) return;
    const int r = i / N_EMBD, d = i - r * N_EMBD;
    ck[i] = qkv[(size_t)r * 3 * N_EMBD + N_EMBD + d];
    cv[i] = qkv[(size_t)r * 3 * N_EMBD + 2 * N_EMBD + d];
}

// Single-query attention over the first len cached positions; one block per head.
__global__ void k_attn_decode(float *out, const float *qrow, const float *ck, const float *cv, int len) {
    __shared__ float sq[HEAD_SIZE];
    __shared__ float sp[BLOCK_SIZE];
    __shared__ float red[ATT_THREADS];
    const int h = blockIdx.x;
    for (int j = threadIdx.x; j < HEAD_SIZE; j += blockDim.x) sq[j] = qrow[h * HEAD_SIZE + j];
    __syncthreads();
    const float scale = 1.0f / sqrtf((float)HEAD_SIZE);
    float mx = -1e30f;
    for (int s = threadIdx.x; s < len; s += blockDim.x) {
        const float *k = ck + (size_t)s * N_EMBD + h * HEAD_SIZE;
        float dot = 0.0f;
        for (int j = 0; j < HEAD_SIZE; j++) dot += sq[j] * k[j];
        dot *= scale;
        sp[s] = dot;
        mx = fmaxf(mx, dot);
    }
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int s = threadIdx.x; s < len; s += blockDim.x) {
        const float e = expf(sp[s] - mx);
        sp[s] = e;
        sum += e;
    }
    const float inv = 1.0f / block_sum(sum, red);
    for (int s = threadIdx.x; s < len; s += blockDim.x) sp[s] *= inv;
    __syncthreads();
    for (int j = threadIdx.x; j < HEAD_SIZE; j += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < len; s++) acc += sp[s] * cv[(size_t)s * N_EMBD + h * HEAD_SIZE + j];
        out[h * HEAD_SIZE + j] = acc;
    }
}

// Muon: Nesterov momentum on the clipped gradient, written to u.
__global__ void k_muon_mom(float *buf, const float *g, float *u, size_t n, float mom, const float *sumsq, float clip) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float norm = sqrtf(*sumsq);
    const float gi = g[i] * (norm > clip ? clip / norm : 1.0f);
    const float b = mom * buf[i] + gi;
    buf[i] = b;
    u[i] = gi + mom * b;
}

__global__ void k_scale_by_norm(float *x, size_t n, const float *sumsq) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (sqrtf(*sumsq) + 1e-7f);
}

// mx = a*I + b*A + c*A^2: one Newton-Schulz step is X <- mx * X (or X * mx for tall matrices).
__global__ void k_ns_poly(float *mx, const float *a1, const float *a2, int d, float a, float b, float c) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d * d) return;
    mx[i] = b * a1[i] + c * a2[i] + (i / d == i % d ? a : 0.0f);
}

__global__ void k_muon_apply(float *w, const float *o, float *ema, size_t n, float lr, float scale, float wd,
                             float ema_decay) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float wi = w[i];
    wi -= lr * (scale * o[i] + wd * wi);
    w[i] = wi;
    ema[i] = ema_decay * ema[i] + (1.0f - ema_decay) * wi;
}

static void qknorm(float *qkv, float *inv, int rows) {
    LAUNCH(k_qknorm_fwd, dim3(2 * N_HEAD, rows), LN_THREADS, qkv, inv, rows);
}

static void qknorm_bwd(float *dqkv, const float *qkv, const float *inv, int rows) {
    LAUNCH(k_qknorm_bwd, dim3(2 * N_HEAD, rows), LN_THREADS, dqkv, qkv, inv, rows);
}

#endif
