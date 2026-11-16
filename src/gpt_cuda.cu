#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <chrono>

#include "config.h"
#include "rng.h"
#include "kernels.cuh"
#include "bpe.h"
#include "sampler.h"

static double now_seconds(void) {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}
typedef struct {
    float *rms1, *wqkv, *wo, *rms2, *w13, *w2;
} LayerParams;

typedef struct {
    float *wte, *rmsf;
    LayerParams layer[N_LAYER];
} Params;

typedef struct {
    size_t off, n;
    int decay;
    int rows, cols;
} TensorInfo;

static size_t bind_params(Params *p, float *base, int V, TensorInfo *ti, int *nt) {
    size_t off = 0;
    int count = 0;
#define TAKE(field, cnt_, dec_, rows_, cols_)                    \
    do {                                                         \
        p->field = base ? base + off : NULL;                     \
        if (ti) {                                                \
            ti[count].off = off;                                 \
            ti[count].n = (size_t)(cnt_);                        \
            ti[count].decay = (dec_);                            \
            ti[count].rows = OPTIMIZER_MUON ? (rows_) : 0;       \
            ti[count].cols = (cols_);                            \
        }                                                        \
        count++;                                                 \
        off += (size_t)(cnt_);                                   \
    } while (0)
    TAKE(wte, (size_t)V * N_EMBD, 0, 0, 0);
    for (int l = 0; l < N_LAYER; l++) {
        TAKE(layer[l].rms1, N_EMBD, 0, 0, 0);
        TAKE(layer[l].wqkv, (size_t)N_EMBD * 3 * N_EMBD, 1, N_EMBD, 3 * N_EMBD);
        TAKE(layer[l].wo, (size_t)N_EMBD * N_EMBD, 1, N_EMBD, N_EMBD);
        TAKE(layer[l].rms2, N_EMBD, 0, 0, 0);
        TAKE(layer[l].w13, (size_t)N_EMBD * 2 * FF_DIM, 1, N_EMBD, 2 * FF_DIM);
        TAKE(layer[l].w2, (size_t)FF_DIM * N_EMBD, 1, FF_DIM, N_EMBD);
    }
    TAKE(rmsf, N_EMBD, 0, 0, 0);
#undef TAKE
    if (nt) *nt = count;
    return off;
}

typedef struct {
    float *x[N_LAYER + 1];
    float *ln1[N_LAYER], *inv1[N_LAYER];
    float *qkv[N_LAYER], *att[N_LAYER], *atty[N_LAYER], *qkr[N_LAYER];
    float *mid[N_LAYER];
    float *ln2[N_LAYER], *inv2[N_LAYER];
    float *h13[N_LAYER], *hact[N_LAYER];
    float *lnf, *invf, *logits;
    float *dx, *dln, *dqkv, *dhact, *dh13, *datty, *dscore;
    float *rope_cos, *rope_sin;
    float *loss_train, *loss_eval, *sumsq;
    int *tok, *tgt;
    float *block;
} Acts;

static size_t bind_acts(Acts *a, float *base, int V) {
    size_t off = 0;
    const size_t att_size = (size_t)BATCH * N_HEAD * BLOCK_SIZE * BLOCK_SIZE;
#define TAKE(field, n) do { a->field = base ? base + off : NULL; off += (size_t)(n); } while (0)
    for (int l = 0; l <= N_LAYER; l++) TAKE(x[l], (size_t)MAX_ROWS * N_EMBD);
    for (int l = 0; l < N_LAYER; l++) {
        TAKE(ln1[l], (size_t)MAX_ROWS * N_EMBD);
        TAKE(inv1[l], MAX_ROWS);
        TAKE(qkv[l], (size_t)MAX_ROWS * 3 * N_EMBD);
        if (QK_NORM) TAKE(qkr[l], (size_t)MAX_ROWS * 2 * N_HEAD);
        TAKE(att[l], att_size);
        TAKE(atty[l], (size_t)MAX_ROWS * N_EMBD);
        TAKE(mid[l], (size_t)MAX_ROWS * N_EMBD);
        TAKE(ln2[l], (size_t)MAX_ROWS * N_EMBD);
        TAKE(inv2[l], MAX_ROWS);
        TAKE(h13[l], (size_t)MAX_ROWS * 2 * FF_DIM);
        TAKE(hact[l], (size_t)MAX_ROWS * FF_DIM);
    }
    TAKE(lnf, (size_t)MAX_ROWS * N_EMBD);
    TAKE(invf, MAX_ROWS);
    TAKE(logits, (size_t)MAX_ROWS * V);
    TAKE(dx, (size_t)MAX_ROWS * N_EMBD);
    TAKE(dln, (size_t)MAX_ROWS * N_EMBD);
    TAKE(dqkv, (size_t)MAX_ROWS * 3 * N_EMBD);
    TAKE(dhact, (size_t)MAX_ROWS * FF_DIM);
    TAKE(dh13, (size_t)MAX_ROWS * 2 * FF_DIM);
    TAKE(datty, (size_t)MAX_ROWS * N_EMBD);
    TAKE(dscore, att_size);
    TAKE(rope_cos, (size_t)BLOCK_SIZE * HALF_HEAD);
    TAKE(rope_sin, (size_t)BLOCK_SIZE * HALF_HEAD);
    TAKE(loss_train, 1);
    TAKE(loss_eval, 1);
    TAKE(sumsq, 1);
#undef TAKE
    return off;
}

static void acts_alloc(Acts *a, int V) {
    size_t total = bind_acts(a, NULL, V);
    CUDA_CHECK(cudaMalloc((void **)&a->block, total * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->block, 0, total * sizeof(float)));
    bind_acts(a, a->block, V);
    CUDA_CHECK(cudaMalloc((void **)&a->tok, (size_t)MAX_ROWS * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&a->tgt, (size_t)MAX_ROWS * sizeof(int)));
    float *hc = XALLOC(float, (size_t)BLOCK_SIZE * HALF_HEAD);
    float *hs = XALLOC(float, (size_t)BLOCK_SIZE * HALF_HEAD);
    for (int t = 0; t < BLOCK_SIZE; t++) {
        for (int j = 0; j < HALF_HEAD; j++) {
            double inv_freq = pow((double)ROPE_THETA, -(double)j / (double)HALF_HEAD);
            double ang = (double)t * inv_freq;
            hc[t * HALF_HEAD + j] = (float)cos(ang);
            hs[t * HALF_HEAD + j] = (float)sin(ang);
        }
    }
    CUDA_CHECK(cudaMemcpy(a->rope_cos, hc, (size_t)BLOCK_SIZE * HALF_HEAD * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(a->rope_sin, hs, (size_t)BLOCK_SIZE * HALF_HEAD * sizeof(float), cudaMemcpyHostToDevice));
    free(hc);
    free(hs);
}

static void acts_free(Acts *a) {
    cudaFree(a->block);
    cudaFree(a->tok);
    cudaFree(a->tgt);
}

typedef struct {
    int V;
    size_t P;
    float *dw, *dg, *dm, *dv, *de;
    Params w, g, e;
    TensorInfo ti[MAX_TENSORS];
    int nt;
    long step;
    float best_val;
#if OPTIMIZER_MUON
    float *mu_x, *mu_xn, *mu_a, *mu_a2, *mu_mx, *mu_ss;
#endif
} Model;

#if OPTIMIZER_MUON
static void muon_scratch_alloc(Model *M) {
    size_t max_rc = 1, max_d2 = 1;
    for (int t = 0; t < M->nt; t++) {
        if (!M->ti[t].rows) continue;
        const size_t d = (size_t)(M->ti[t].rows < M->ti[t].cols ? M->ti[t].rows : M->ti[t].cols);
        if (M->ti[t].n > max_rc) max_rc = M->ti[t].n;
        if (d * d > max_d2) max_d2 = d * d;
    }
    float **bufs[5] = {&M->mu_x, &M->mu_xn, &M->mu_a, &M->mu_a2, &M->mu_mx};
    const size_t sizes[5] = {max_rc, max_rc, max_d2, max_d2, max_d2};
    for (int i = 0; i < 5; i++) CUDA_CHECK(cudaMalloc((void **)bufs[i], sizes[i] * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&M->mu_ss, sizeof(float)));
}

static void muon_scratch_free(Model *M) {
    cudaFree(M->mu_x);
    cudaFree(M->mu_xn);
    cudaFree(M->mu_a);
    cudaFree(M->mu_a2);
    cudaFree(M->mu_mx);
    cudaFree(M->mu_ss);
}

// Quintic Newton-Schulz iteration that pushes the singular values of x (R x C) towards 1. The result may be in
// either x or xn; the returned pointer says which. Tall matrices use X^T X so the Gram matrix stays small.
static float *muon_orthogonalize(float *x, float *xn, float *a1, float *a2, float *mx, float *ss, int R, int C) {
    const size_t n = (size_t)R * C;
    int blocks = ew_blocks(n);
    if (blocks > 1024) blocks = 1024;
    CUDA_CHECK(cudaMemset(ss, 0, sizeof(float)));
    LAUNCH(k_sumsq, blocks, EW_THREADS, x, n, ss);
    LAUNCH(k_scale_by_norm, ew_blocks(n), EW_THREADS, x, n, ss);
    const int wide = R <= C, d = wide ? R : C;
    for (int it = 0; it < MUON_NS_STEPS; it++) {
        if (wide) gemm(x, x, a1, NULL, R, R, C, 0, 1, 0);
        else gemm(x, x, a1, NULL, C, C, R, 1, 0, 0);
        gemm(a1, a1, a2, NULL, d, d, d, 0, 0, 0);
        LAUNCH(k_ns_poly, ew_blocks((size_t)d * d), EW_THREADS, mx, a1, a2, d, 3.4445f, -4.7750f, 2.0315f);
        if (wide) gemm(mx, x, xn, NULL, R, C, R, 0, 0, 0);
        else gemm(x, mx, xn, NULL, R, C, C, 0, 0, 0);
        float *t = x;
        x = xn;
        xn = t;
    }
    return x;
}
#endif

static void model_alloc(Model *M, int V) {
    M->V = V;
    M->P = bind_params(&M->w, NULL, V, M->ti, &M->nt);
    size_t bytes = M->P * sizeof(float);
    float **bufs[5] = {&M->dw, &M->dg, &M->dm, &M->dv, &M->de};
    for (int i = 0; i < 5; i++) {
        CUDA_CHECK(cudaMalloc((void **)bufs[i], bytes));
        CUDA_CHECK(cudaMemset(*bufs[i], 0, bytes));
    }
    bind_params(&M->w, M->dw, V, NULL, NULL);
    bind_params(&M->g, M->dg, V, NULL, NULL);
    bind_params(&M->e, M->de, V, NULL, NULL);
#if OPTIMIZER_MUON
    muon_scratch_alloc(M);
#endif
    M->step = 0;
    M->best_val = INFINITY;
}

static void model_alloc_worker(Model *M, int V) {
    M->V = V;
    M->P = bind_params(&M->w, NULL, V, M->ti, &M->nt);
    size_t bytes = M->P * sizeof(float);
    M->dm = M->dv = M->de = NULL;
    float **bufs[2] = {&M->dw, &M->dg};
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaMalloc((void **)bufs[i], bytes));
        CUDA_CHECK(cudaMemset(*bufs[i], 0, bytes));
    }
    bind_params(&M->w, M->dw, V, NULL, NULL);
    bind_params(&M->g, M->dg, V, NULL, NULL);
#if OPTIMIZER_MUON
    M->mu_x = M->mu_xn = M->mu_a = M->mu_a2 = M->mu_mx = M->mu_ss = NULL;
#endif
    M->step = 0;
    M->best_val = INFINITY;
}

static void model_free(Model *M) {
    cudaFree(M->dw);
    cudaFree(M->dg);
    cudaFree(M->dm);
    cudaFree(M->dv);
    cudaFree(M->de);
#if OPTIMIZER_MUON
    muon_scratch_free(M);
#endif
}

static void fill_normal(float *p, size_t n, float stddev) {
    for (size_t i = 0; i < n; i++) p[i] = stddev * rand_normal();
}

static void fill_const(float *p, size_t n, float value) {
    for (size_t i = 0; i < n; i++) p[i] = value;
}

static void model_random_init(Model *M, float noise) {
    int V = M->V;
    float *h = XALLOC(float, M->P);
    Params hp;
    bind_params(&hp, h, V, NULL, NULL);
    float proj_std = 0.02f / sqrtf(2.0f * N_LAYER);
    fill_normal(hp.wte, (size_t)V * N_EMBD, 0.02f);
    for (int l = 0; l < N_LAYER; l++) {
        LayerParams *lp = &hp.layer[l];
        fill_const(lp->rms1, N_EMBD, 1.0f);
        fill_const(lp->rms2, N_EMBD, 1.0f);
        fill_normal(lp->wqkv, (size_t)N_EMBD * 3 * N_EMBD, 0.02f);
        fill_normal(lp->wo, (size_t)N_EMBD * N_EMBD, proj_std);
        fill_normal(lp->w13, (size_t)N_EMBD * 2 * FF_DIM, 0.02f);
        fill_normal(lp->w2, (size_t)FF_DIM * N_EMBD, proj_std);
    }
    fill_const(hp.rmsf, N_EMBD, 1.0f);
    if (noise > 0.0f) {
        for (size_t i = 0; i < M->P; i++) h[i] += noise * rand_normal();
    }
    CUDA_CHECK(cudaMemcpy(M->dw, h, M->P * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(M->de, h, M->P * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}
static unsigned branch_seed(unsigned dseed, int layer, int which) {
    return dseed ? (dseed + (unsigned)(2 * layer + which + 1) * 0x9E3779B1u) | 1u : 0u;
}

static void forward(const Params *P, int V, Acts *A, const int *tok, int B, int tl,
                    unsigned dseed = 0, int last_only = 0) {
    const float dp = (dseed && DROPOUT > 0.0f) ? (float)DROPOUT : 0.0f;
    const int D = N_EMBD;
    const int n = B * tl;
    LAUNCH(k_embed, ew_blocks((size_t)n * D), EW_THREADS, A->x[0], P->wte, tok, n, D);
    for (int l = 0; l < N_LAYER; l++) {
        const LayerParams *lp = &P->layer[l];
        rms_fwd(A->ln1[l], A->inv1[l], A->x[l], lp->rms1, n);
        gemm(A->ln1[l], lp->wqkv, A->qkv[l], NULL, n, 3 * D, D, 0, 0, 0);
        if (QK_NORM) qknorm(A->qkv[l], A->qkr[l], n);
        rope(A->qkv[l], A->rope_cos, A->rope_sin, n, tl, 1.0f);
        attn_fwd(A->atty[l], A->att[l], A->qkv[l], B, tl);
        gemm(A->atty[l], lp->wo, A->mid[l], A->x[l], n, D, D, 0, 0, 0, dp, branch_seed(dseed, l, 0));
        rms_fwd(A->ln2[l], A->inv2[l], A->mid[l], lp->rms2, n);
        gemm(A->ln2[l], lp->w13, A->h13[l], NULL, n, 2 * FF_DIM, D, 0, 0, 0);
        LAUNCH(k_swiglu_fwd, ew_blocks((size_t)n * FF_DIM), EW_THREADS, A->hact[l], A->h13[l], (size_t)n * FF_DIM);
        gemm(A->hact[l], lp->w2, A->x[l + 1], A->mid[l], n, D, FF_DIM, 0, 0, 0, dp, branch_seed(dseed, l, 1));
    }
    rms_fwd(A->lnf, A->invf, A->x[N_LAYER], P->rmsf, n);
    if (last_only) gemm(A->lnf + (size_t)(n - 1) * D, P->wte, A->logits, NULL, 1, V, D, 0, 1, 0);
    else gemm(A->lnf, P->wte, A->logits, NULL, n, V, D, 0, 1, 0);
}

static void loss_and_grad_logits(Acts *A, const int *tgt, float *loss_acc, int V, int n, float scale) {
    LAUNCH(k_softmax_xent, n, XENT_THREADS, A->logits, tgt, loss_acc, V, scale);
}

static void backward(const Model *M, Acts *A, int B, int tl, unsigned dseed = 0) {
    const float dp = (dseed && DROPOUT > 0.0f) ? (float)DROPOUT : 0.0f;
    const Params *P = &M->w;
    const Params *G = &M->g;
    const int D = N_EMBD, V = M->V;
    const int n = B * tl;

    gemm(A->logits, P->wte, A->dln, NULL, n, D, V, 0, 0, 0);
    gemm(A->logits, A->lnf, G->wte, NULL, V, D, n, 1, 0, 1);

    CUDA_CHECK(cudaMemset(A->dx, 0, (size_t)n * D * sizeof(float)));
    rms_bwd(A->dx, G->rmsf, A->dln, A->x[N_LAYER], A->invf, P->rmsf, n);

    for (int l = N_LAYER - 1; l >= 0; l--) {
        const LayerParams *lp = &P->layer[l];
        const LayerParams *lg = &G->layer[l];

        const float *dbr = A->dx;
        if (dp > 0.0f) {
            LAUNCH(k_dropout_apply, ew_blocks((size_t)n * D), EW_THREADS, A->dln, A->dx, (size_t)n * D, dp,
                   branch_seed(dseed, l, 1));
            dbr = A->dln;
        }
        gemm(dbr, lp->w2, A->dhact, NULL, n, FF_DIM, D, 0, 1, 0);
        gemm(A->hact[l], dbr, lg->w2, NULL, FF_DIM, D, n, 1, 0, 1);
        LAUNCH(k_swiglu_bwd, ew_blocks((size_t)n * FF_DIM), EW_THREADS, A->dh13, A->dhact, A->h13[l], (size_t)n * FF_DIM);

        gemm(A->dh13, lp->w13, A->dln, NULL, n, D, 2 * FF_DIM, 0, 1, 0);
        gemm(A->ln2[l], A->dh13, lg->w13, NULL, D, 2 * FF_DIM, n, 1, 0, 1);
        rms_bwd(A->dx, lg->rms2, A->dln, A->mid[l], A->inv2[l], lp->rms2, n);

        dbr = A->dx;
        if (dp > 0.0f) {
            LAUNCH(k_dropout_apply, ew_blocks((size_t)n * D), EW_THREADS, A->dln, A->dx, (size_t)n * D, dp,
                   branch_seed(dseed, l, 0));
            dbr = A->dln;
        }
        gemm(dbr, lp->wo, A->datty, NULL, n, D, D, 0, 1, 0);
        gemm(A->atty[l], dbr, lg->wo, NULL, D, D, n, 1, 0, 1);

        attn_bwd(A->dqkv, A->dscore, A->datty, A->att[l], A->qkv[l], B, tl);
        rope(A->dqkv, A->rope_cos, A->rope_sin, n, tl, -1.0f);
        if (QK_NORM) {
            rope(A->qkv[l], A->rope_cos, A->rope_sin, n, tl, -1.0f);
            qknorm_bwd(A->dqkv, A->qkv[l], A->qkr[l], n);
        }

        gemm(A->dqkv, lp->wqkv, A->dln, NULL, n, D, 3 * D, 0, 1, 0);
        gemm(A->ln1[l], A->dqkv, lg->wqkv, NULL, D, 3 * D, n, 1, 0, 1);
        rms_bwd(A->dx, lg->rms1, A->dln, A->x[l], A->inv1[l], lp->rms1, n);
    }

    LAUNCH(k_embed_bwd, ew_blocks((size_t)n * D), EW_THREADS, G->wte, A->dx, A->tok, n, D);
}

static void zero_grads(Model *M) {
    CUDA_CHECK(cudaMemset(M->dg, 0, M->P * sizeof(float)));
}

static void adam_step(Model *M, Acts *A, float lr, float ema_decay) {
    M->step++;
    float c1 = 1.0f - powf(0.9f, (float)M->step);
    float c2 = 1.0f - powf(0.95f, (float)M->step);
    CUDA_CHECK(cudaMemset(A->sumsq, 0, sizeof(float)));
    int ss_blocks = ew_blocks(M->P);
    if (ss_blocks > 1024) ss_blocks = 1024;
    LAUNCH(k_sumsq, ss_blocks, EW_THREADS, M->dg, M->P, A->sumsq);
    for (int t = 0; t < M->nt; t++) {
        size_t off = M->ti[t].off, n = M->ti[t].n;
        float wd = M->ti[t].decay ? WEIGHT_DECAY : 0.0f;
#if OPTIMIZER_MUON
        if (M->ti[t].rows) {
            const int R = M->ti[t].rows, C = M->ti[t].cols;
            LAUNCH(k_muon_mom, ew_blocks(n), EW_THREADS, M->dm + off, M->dg + off, M->mu_x, n, MUON_MOMENTUM,
                   A->sumsq, GRAD_CLIP);
            const float *o = muon_orthogonalize(M->mu_x, M->mu_xn, M->mu_a, M->mu_a2, M->mu_mx, M->mu_ss, R, C);
            const float scale = MUON_UPDATE_SCALE * sqrtf((float)(R > C ? R : C));
            LAUNCH(k_muon_apply, ew_blocks(n), EW_THREADS, M->dw + off, o, M->de + off, n, lr, scale, wd, ema_decay);
            continue;
        }
#endif
        LAUNCH(k_adamw, ew_blocks(n), EW_THREADS, M->dw + off, M->dg + off, M->dm + off, M->dv + off,
               M->de + off, n, lr, c1, c2, wd, ema_decay, A->sumsq, GRAD_CLIP);
    }
}

static float read_scalar(const float *dptr) {
    float v = 0.0f;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&v, dptr, sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}

static void upload_batch(Acts *A, const int *htok, const int *htgt, int n) {
    CUDA_CHECK(cudaMemcpy(A->tok, htok, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A->tgt, htgt, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
}

static void fill_sequence(const uint16_t *data, size_t pos, int tl, int *htok, int *htgt) {
    for (int t = 0; t < tl; t++) {
        htok[t] = data[pos + t];
        htgt[t] = data[pos + t + 1];
    }
}

static void make_batch(const uint16_t *data, size_t from, size_t span, int B, int tl, int *htok, int *htgt) {
    for (int b = 0; b < B; b++) {
        size_t pos = from + rng_next() % span;
        fill_sequence(data, pos, tl, htok + b * tl, htgt + b * tl);
    }
}

static void dev_write(FILE *f, const float *d, size_t n, int *ok) {
    const size_t chunk = 1 << 20;
    float *h = XALLOC(float, chunk);
    for (size_t off = 0; off < n && *ok; off += chunk) {
        size_t c = n - off < chunk ? n - off : chunk;
        CUDA_CHECK(cudaMemcpy(h, d + off, c * sizeof(float), cudaMemcpyDeviceToHost));
        if (fwrite(h, sizeof(float), c, f) != c) *ok = 0;
    }
    free(h);
}

static int dev_read(FILE *f, float *d, size_t n) {
    const size_t chunk = 1 << 20;
    float *h = XALLOC(float, chunk);
    int ok = 1;
    for (size_t off = 0; off < n && ok; off += chunk) {
        size_t c = n - off < chunk ? n - off : chunk;
        if (fread(h, sizeof(float), c, f) != c) {
            ok = 0;
            break;
        }
        CUDA_CHECK(cudaMemcpy(d + off, h, c * sizeof(float), cudaMemcpyHostToDevice));
    }
    free(h);
    return ok;
}

static uint32_t header_flags(uint32_t kind) {
    return (QK_NORM ? 0x100u : 0u) | (kind == 1 && OPTIMIZER_MUON ? 0x200u : 0u);
}

static void make_header(uint32_t *h, int V, uint32_t kind) {
    h[0] = MAGIC;
    h[1] = (uint32_t)V;
    h[2] = BLOCK_SIZE;
    h[3] = N_EMBD;
    h[4] = N_HEAD;
    h[5] = N_LAYER;
    h[6] = FF_DIM;
    h[7] = kind | header_flags(kind);
}

static int check_header(const uint32_t *h, uint32_t kind) {
    return h[0] == MAGIC && h[1] > 0 && h[1] <= BPE_MAX_VOCAB && h[2] == BLOCK_SIZE && h[3] == N_EMBD &&
           h[4] == N_HEAD && h[5] == N_LAYER && h[6] == FF_DIM && h[7] == (kind | header_flags(kind));
}

static int save_infer(const char *path, const Model *M, const Tokenizer *T) {
    char tmp[1200];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    uint32_t h[8];
    make_header(h, M->V, 0);
    int ok = fwrite(h, sizeof(uint32_t), 8, f) == 8;
    ok = ok && tok_write(f, T) == 0;
    dev_write(f, M->de, M->P, &ok);
    ok = (fclose(f) == 0) && ok;
    if (!ok) return -1;
    return rename(tmp, path);
}

static int load_infer(const char *path, Model *M, Tokenizer *T) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t h[8];
    if (fread(h, sizeof(uint32_t), 8, f) != 8 || !check_header(h, 0)) {
        fclose(f);
        return -2;
    }
    if (tok_read(f, T) != 0 || T->V != (int)h[1]) {
        fclose(f);
        return -2;
    }
    model_alloc(M, (int)h[1]);
    int ok = dev_read(f, M->dw, M->P);
    fclose(f);
    if (!ok) {
        model_free(M);
        return -2;
    }
    return 0;
}

static int save_ckpt(const char *path, const Model *M, const Tokenizer *T) {
    char tmp[1200];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    uint32_t h[8];
    make_header(h, M->V, 1);
    int ok = fwrite(h, sizeof(uint32_t), 8, f) == 8;
    ok = ok && tok_write(f, T) == 0;
    uint64_t step = (uint64_t)M->step;
    ok = ok && fwrite(&step, sizeof(step), 1, f) == 1;
    dev_write(f, M->dw, M->P, &ok);
    dev_write(f, M->de, M->P, &ok);
    dev_write(f, M->dm, M->P, &ok);
    dev_write(f, M->dv, M->P, &ok);
    ok = ok && fwrite(&M->best_val, sizeof(float), 1, f) == 1;
    ok = (fclose(f) == 0) && ok;
    if (!ok) return -1;
    return rename(tmp, path);
}

static int load_ckpt(const char *path, Model *M, Tokenizer *T) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t h[8];
    if (fread(h, sizeof(uint32_t), 8, f) != 8 || !check_header(h, 1)) {
        fclose(f);
        return -2;
    }
    if (tok_read(f, T) != 0 || T->V != (int)h[1]) {
        fclose(f);
        return -2;
    }
    uint64_t step;
    if (fread(&step, sizeof(step), 1, f) != 1) {
        fclose(f);
        return -2;
    }
    model_alloc(M, (int)h[1]);
    int ok = dev_read(f, M->dw, M->P) && dev_read(f, M->de, M->P) && dev_read(f, M->dm, M->P) &&
             dev_read(f, M->dv, M->P);
    float best = INFINITY;
    if (!ok || fread(&best, sizeof(float), 1, f) != 1 || !(best > 0.0f)) best = INFINITY;
    fclose(f);
    if (!ok) {
        model_free(M);
        return -2;
    }
    M->step = (long)step;
    M->best_val = best;
    return 0;
}

static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    unsigned char *buf = (unsigned char *)malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    *len = fread(buf, 1, (size_t)n, f);
    buf[*len] = 0;
    fclose(f);
    return buf;
}

typedef struct {
    float *k[N_LAYER], *v[N_LAYER];
    int len;
} KVCache;

static void kv_alloc(KVCache *c) {
    for (int l = 0; l < N_LAYER; l++) {
        CUDA_CHECK(cudaMalloc((void **)&c->k[l], (size_t)BLOCK_SIZE * N_EMBD * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&c->v[l], (size_t)BLOCK_SIZE * N_EMBD * sizeof(float)));
    }
    c->len = 0;
}

static void kv_free(KVCache *c) {
    for (int l = 0; l < N_LAYER; l++) {
        cudaFree(c->k[l]);
        cudaFree(c->v[l]);
    }
}

static void kv_fill_from_prefill(KVCache *c, const Acts *A, int n) {
    for (int l = 0; l < N_LAYER; l++)
        LAUNCH(k_cache_store, ew_blocks((size_t)n * N_EMBD), EW_THREADS, c->k[l], c->v[l], A->qkv[l], n);
    c->len = n;
}

// One new token at position c->len using the cached keys and values; logits land in A->logits row 0.
static void decode_step(const Params *P, int V, Acts *A, KVCache *c, int token) {
    const int D = N_EMBD, pos = c->len;
    CUDA_CHECK(cudaMemcpy(A->tok, &token, sizeof(int), cudaMemcpyHostToDevice));
    LAUNCH(k_embed, ew_blocks(D), EW_THREADS, A->x[0], P->wte, A->tok, 1, D);
    for (int l = 0; l < N_LAYER; l++) {
        const LayerParams *lp = &P->layer[l];
        rms_fwd(A->ln1[l], A->inv1[l], A->x[l], lp->rms1, 1);
        gemm(A->ln1[l], lp->wqkv, A->qkv[l], NULL, 1, 3 * D, D, 0, 0, 0);
        if (QK_NORM) qknorm(A->qkv[l], A->qkr[l], 1);
        rope(A->qkv[l], A->rope_cos + (size_t)pos * HALF_HEAD, A->rope_sin + (size_t)pos * HALF_HEAD, 1, 1, 1.0f);
        LAUNCH(k_cache_store, ew_blocks(D), EW_THREADS, c->k[l] + (size_t)pos * D, c->v[l] + (size_t)pos * D,
               A->qkv[l], 1);
        LAUNCH(k_attn_decode, N_HEAD, ATT_THREADS, A->atty[l], A->qkv[l], c->k[l], c->v[l], pos + 1);
        gemm(A->atty[l], lp->wo, A->mid[l], A->x[l], 1, D, D, 0, 0, 0);
        rms_fwd(A->ln2[l], A->inv2[l], A->mid[l], lp->rms2, 1);
        gemm(A->ln2[l], lp->w13, A->h13[l], NULL, 1, 2 * FF_DIM, D, 0, 0, 0);
        LAUNCH(k_swiglu_fwd, ew_blocks(FF_DIM), EW_THREADS, A->hact[l], A->h13[l], (size_t)FF_DIM);
        gemm(A->hact[l], lp->w2, A->x[l + 1], A->mid[l], 1, D, FF_DIM, 0, 0, 0);
    }
    rms_fwd(A->lnf, A->invf, A->x[N_LAYER], P->rmsf, 1);
    gemm(A->lnf, P->wte, A->logits, NULL, 1, V, D, 0, 1, 0);
    c->len = pos + 1;
}

static void generate(const Params *P, int V, Acts *A, const Tokenizer *T, const char *prompt,
                     int n_tokens, float temp, int top_k, float top_p, float rep, float min_p, int ngram) {
    size_t plen = strlen(prompt);
    size_t n_prompt = 0;
    uint16_t *pids = NULL;
    if (plen > 0) pids = tok_encode(T, (const uint8_t *)prompt, plen, &n_prompt);
    size_t cap = n_prompt + (size_t)n_tokens + 2;
    int *seq = XALLOC(int, cap);
    float *row = XALLOC(float, V);
    unsigned char *banned = XALLOC(unsigned char, V);
    size_t len = 0;
    for (size_t i = 0; i < n_prompt; i++) seq[len++] = pids[i];
    free(pids);
    if (plen > 0) fwrite(prompt, 1, plen, stdout);
    if (len == 0) {
        uint8_t nl = '\n';
        size_t k = 0;
        uint16_t *ids = tok_encode(T, &nl, 1, &k);
        seq[len++] = k > 0 ? ids[0] : 0;
        free(ids);
    }
    uint8_t buf[512];
    KVCache cache;
    kv_alloc(&cache);
    for (int i = 0; i < n_tokens; i++) {
        if (i == 0 || cache.len >= BLOCK_SIZE) {
            // Prefill: the whole prompt first, afterwards only the newest half window when the cache is full.
            const int n = i == 0 ? (len < BLOCK_SIZE ? (int)len : BLOCK_SIZE) : BLOCK_SIZE / 2;
            CUDA_CHECK(cudaMemcpy(A->tok, seq + len - n, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
            forward(P, V, A, A->tok, 1, n, 0, 1);
            kv_fill_from_prefill(&cache, A, n);
        } else {
            decode_step(P, V, A, &cache, seq[len - 1]);
        }
        CUDA_CHECK(cudaMemcpy(row, A->logits, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        int nr = len < REP_WINDOW ? (int)len : REP_WINDOW;
        ban_ngrams(seq, len, ngram, banned, V);
        int next = sample_token(row, V, seq + len - nr, nr, banned, temp, top_k, top_p, min_p, rep);
        seq[len++] = next;
        size_t bl = tok_decode_token(T, next, buf);
        fwrite(buf, 1, bl, stdout);
        fflush(stdout);
    }
    putchar('\n');
    kv_free(&cache);
    free(banned);
    free(row);
    free(seq);
}

static float eval_loss(const Params *P, int V, Acts *A, const uint16_t *data, size_t from, size_t to,
                       int batches, int *htok, int *htgt) {
    const int tl = BLOCK_SIZE;
    const int n = BATCH * tl;
    size_t span = to - from - tl - 1;
    size_t total = (size_t)batches * BATCH;
    CUDA_CHECK(cudaMemset(A->loss_eval, 0, sizeof(float)));
    for (int i = 0; i < batches; i++) {
        for (int j = 0; j < BATCH; j++) {
            size_t pos = from + ((size_t)i * BATCH + j) * span / total;
            fill_sequence(data, pos, tl, htok + j * tl, htgt + j * tl);
        }
        upload_batch(A, htok, htgt, n);
        forward(P, V, A, A->tok, BATCH, tl);
        loss_and_grad_logits(A, A->tgt, A->loss_eval, V, n, 1.0f / (float)n);
    }
    return read_scalar(A->loss_eval) / ((float)batches * n);
}

typedef struct {
    const char *gpus;
    const char *devices;
    int has_seed;
    uint64_t seed;
    int patience;
} Options;

typedef struct {
    int n;
    int id[MAX_GPUS];
} GpuPlan;

typedef struct {
    int dev;
    Model M;
    Acts A;
} Replica;

static size_t train_mem_estimate(int V) {
    Acts a;
    Params p;
    size_t acts = bind_acts(&a, NULL, V);
    size_t par = bind_params(&p, NULL, V, NULL, NULL);

    return (acts + 6 * par) * sizeof(float) + ((size_t)256 << 20);
}

static int parse_device_list(const char *list, int count, GpuPlan *plan) {
    const char *c = list;
    plan->n = 0;
    while (*c) {
        char *end;
        long d = strtol(c, &end, 10);
        if (end == c || d < 0 || d >= count) {
            fprintf(stderr, "--devices: invalid device in \"%s\" (%d GPU(s) visible, ids 0..%d)\n", list, count,
                    count - 1);
            return -1;
        }
        for (int i = 0; i < plan->n; i++) {
            if (plan->id[i] == (int)d) {
                fprintf(stderr, "--devices: device %ld listed twice\n", d);
                return -1;
            }
        }
        if (plan->n >= MAX_GPUS) {
            fprintf(stderr, "--devices: at most %d GPUs are supported\n", MAX_GPUS);
            return -1;
        }
        plan->id[plan->n++] = (int)d;
        c = end;
        if (*c == ',') c++;
        else if (*c) {
            fprintf(stderr, "--devices: cannot parse \"%s\"\n", list);
            return -1;
        }
    }
    if (plan->n == 0) {
        fprintf(stderr, "--devices: empty list\n");
        return -1;
    }
    return 0;
}

static int plan_gpus(const Options *o, int V, GpuPlan *plan) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 1) {
        fprintf(stderr, "No CUDA device found\n");
        return -1;
    }
    if (o->gpus && o->devices) {
        fprintf(stderr, "Use either --gpus or --devices, not both\n");
        return -1;
    }
    const char *spec = o->gpus ? o->gpus : DEFAULT_GPUS;
    int is_auto = 0;
    plan->n = 0;
    if (o->devices) {
        if (parse_device_list(o->devices, count, plan) != 0) return -1;
    } else if (strcmp(spec, "auto") == 0) {
        is_auto = 1;
        size_t need = train_mem_estimate(V);
        for (int d = 0; d < count && plan->n < MAX_GPUS; d++) {
            size_t fr = 0, total = 0;
            CUDA_CHECK(cudaSetDevice(d));
            CUDA_CHECK(cudaMemGetInfo(&fr, &total));
            if (fr < need) {
                printf("GPU %d skipped: %.1f GB free, about %.1f GB needed\n", d, fr / 1073741824.0,
                       need / 1073741824.0);
                continue;
            }
            plan->id[plan->n++] = d;
        }
        if (plan->n == 0) {
            fprintf(stderr, "No GPU has enough free memory (about %.1f GB needed)\n", need / 1073741824.0);
            return -1;
        }
    } else {
        char *end;
        long k = strtol(spec, &end, 10);
        if (*end || k < 1) {
            fprintf(stderr, "--gpus: expected \"auto\" or a number >= 1, got \"%s\"\n", spec);
            return -1;
        }
        if (k > count) {
            fprintf(stderr, "--gpus %ld requested but only %d GPU(s) are visible\n", k, count);
            return -1;
        }
        if (k > MAX_GPUS) {
            fprintf(stderr, "--gpus: at most %d GPUs are supported\n", MAX_GPUS);
            return -1;
        }
        for (int i = 0; i < (int)k; i++) plan->id[plan->n++] = i;
    }
    if (plan->n > ACCUM) {
        printf("Note: ACCUM=%d, so only %d of %d selected GPU(s) are used (rebuild with -DACCUM=%d to use all)%s\n",
               ACCUM, ACCUM, plan->n, plan->n, is_auto ? "" : ".");
        plan->n = ACCUM;
    }
    return 0;
}

static void print_gpu_plan(const GpuPlan *plan) {
    cudaDeviceProp prop;
    if (plan->n == 1) {
        CUDA_CHECK(cudaGetDeviceProperties(&prop, plan->id[0]));
        printf("GPU: %s\n", prop.name);
        return;
    }
    printf("GPUs: %d (data parallel, %d microbatch(es) per step split round-robin)\n", plan->n, ACCUM);
    char first[256] = "";
    for (int i = 0; i < plan->n; i++) {
        CUDA_CHECK(cudaGetDeviceProperties(&prop, plan->id[i]));
        printf("  [%d] %s%s\n", plan->id[i], prop.name, i == 0 ? " (master)" : "");
        if (i == 0) snprintf(first, sizeof(first), "%s", prop.name);
        else if (strcmp(first, prop.name) != 0)
            printf("  warning: mixed GPU models, the slowest one sets the pace\n");
    }
}

static int enable_peer(int from, int to) {
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, from, to) != cudaSuccess || !can) return 0;
    CUDA_CHECK(cudaSetDevice(from));
    cudaError_t e = cudaDeviceEnablePeerAccess(to, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return 0;
    }
    return 1;
}

static void reduce_grads(Model *master, int mdev, float *stage, Replica *W, int nw) {
    CUDA_CHECK(cudaSetDevice(mdev));
    for (int i = 0; i < nw; i++) {
        CUDA_CHECK(cudaMemcpyPeer(stage, mdev, W[i].M.dg, W[i].dev, master->P * sizeof(float)));
        LAUNCH(k_add, ew_blocks(master->P), EW_THREADS, master->dg, stage, master->P);
    }
}

static void broadcast_weights(const Model *master, int mdev, Replica *W, int nw) {
    CUDA_CHECK(cudaSetDevice(mdev));
    for (int i = 0; i < nw; i++)
        CUDA_CHECK(cudaMemcpyPeer(W[i].M.dw, W[i].dev, master->dw, mdev, master->P * sizeof(float)));
}

static int run_generate(const char *model_path, const char *prompt, int n_tokens, float temp,
                        int top_k, float top_p, float rep, float min_p, int ngram);

static int run_train(const char *corpus_path, int steps, const char *model_path, int vocab,
                     const Options *opt) {
    size_t len;
    unsigned char *text = read_file(corpus_path, &len);
    if (!text) {
        fprintf(stderr, "Failed to open %s\n", corpus_path);
        return 1;
    }

    GpuPlan plan;
    int est_vocab = vocab > BPE_MAX_VOCAB ? BPE_MAX_VOCAB : (vocab < 1 ? 1 : vocab);
    if (plan_gpus(opt, est_vocab, &plan) != 0) {
        free(text);
        return 1;
    }
    const int G = plan.n;
    const int nw = G - 1;
    const int mdev = plan.id[0];
    CUDA_CHECK(cudaSetDevice(mdev));
    print_gpu_plan(&plan);

    char ckpt_path[1100];
    snprintf(ckpt_path, sizeof(ckpt_path), "%s.ckpt", model_path);

    Model M;
    Tokenizer T;
    int status = load_ckpt(ckpt_path, &M, &T);
    int resumed = status == 0;
    if (resumed) {
        rng_state ^= (uint64_t)(M.step + 1) * 0xD6E8FEB86659FD93ULL;
        for (int i = 0; i < 8; i++) rng_next();
    }
    if (status == -2) {
        fprintf(stderr, "%s is corrupt or was built with different hyperparameters (architecture, QK_NORM, optimizer)\n", ckpt_path);
        free(text);
        return 1;
    }
    if (!resumed) {
        char tok_path[1100];
        snprintf(tok_path, sizeof(tok_path), "%s.v%d.tok", corpus_path, vocab);
        FILE *tf = fopen(tok_path, "rb");
        int loaded = tf && tok_read(tf, &T) == 0;
        if (tf) fclose(tf);
        if (!loaded) {
            double t0 = now_seconds();
            tok_train(&T, text, len, vocab, 1);
            printf("Tokenizer trained in %.1f s\n", now_seconds() - t0);
            tf = fopen(tok_path, "wb");
            if (tf) {
                tok_write(tf, &T);
                fclose(tf);
            }
        } else {
            printf("Tokenizer loaded from %s\n", tok_path);
        }
    }
    const int V = T.V;

    size_t n_data = 0;
    uint16_t *data = tok_encode(&T, text, len, &n_data);
    free(text);

    if (n_data < (size_t)(4 * BLOCK_SIZE)) {
        fprintf(stderr, "Corpus too small: need at least %d tokens\n", 4 * BLOCK_SIZE);
        free(data);
        return 1;
    }

    size_t split = n_data * 98 / 100;
    if (n_data - split < (size_t)(BLOCK_SIZE + 2)) split = n_data - (size_t)(BLOCK_SIZE + 2);
    if (split < (size_t)(BLOCK_SIZE + 2)) split = n_data;
    int has_val = split < n_data;
    double bytes_per_token = (double)len / (double)n_data;

    if (!resumed) {
        model_alloc(&M, V);
        model_random_init(&M, 0.0f);
    }
    Acts A;
    acts_alloc(&A, V);

    Replica W[MAX_GPUS];
    float *gstage = NULL;
    if (nw > 0) {
        CUDA_CHECK(cudaMalloc((void **)&gstage, M.P * sizeof(float)));
        int p2p = 1;
        for (int i = 0; i < nw; i++) {
            W[i].dev = plan.id[i + 1];
            p2p &= enable_peer(mdev, W[i].dev);
            p2p &= enable_peer(W[i].dev, mdev);
            CUDA_CHECK(cudaSetDevice(W[i].dev));
            model_alloc_worker(&W[i].M, V);
            acts_alloc(&W[i].A, V);
        }
        CUDA_CHECK(cudaSetDevice(mdev));
        broadcast_weights(&M, mdev, W, nw);
        printf("Peer-to-peer copies: %s\n\n", p2p ? "enabled" : "unavailable, gradients and weights are staged through the host");
    }

    const int tl = BLOCK_SIZE;
    const int n = BATCH * tl;
    const double tokens_per_step = (double)n * ACCUM;
    printf("%s | tokens: %zu | bytes/token: %.2f | vocab: %d | parameters: %zu\n",
           resumed ? "Resumed from checkpoint" : "New model", n_data, bytes_per_token, V, M.P);
    printf("block %d | embd %d | heads %d | layers %d | ffn %d | batch %d x accum %d | %.0f tokens/step | dropout %.2f\n",
           BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, FF_DIM, BATCH, ACCUM, tokens_per_step, (double)DROPOUT);
    printf("total tokens seen at end: %.1fM (%.1f epochs)\n\n", tokens_per_step * steps / 1e6,
           tokens_per_step * steps / (double)split);

    int *htok = XALLOC(int, n);
    int *htgt = XALLOC(int, n);

    const long start_step = M.step;
    int warmup = steps / 20 < 200 ? (steps / 20 > 1 ? steps / 20 : 1) : 200;
    double t_start = now_seconds(), t_last = t_start;
    int since_log = 0;
    int bad_evals = 0, stop_early = 0;
    const int patience = opt->patience > 0 ? opt->patience : EVAL_PATIENCE;
    if (resumed && isfinite(M.best_val)) printf("Best val_ema so far: %.4f\n", M.best_val);

    CUDA_CHECK(cudaMemset(A.loss_train, 0, sizeof(float)));
    for (int i = 0; i < nw; i++) {
        CUDA_CHECK(cudaSetDevice(W[i].dev));
        CUDA_CHECK(cudaMemset(W[i].A.loss_train, 0, sizeof(float)));
    }
    CUDA_CHECK(cudaSetDevice(mdev));

    for (long step = start_step + 1; step <= steps; step++) {
        zero_grads(&M);
        for (int i = 0; i < nw; i++) {
            CUDA_CHECK(cudaSetDevice(W[i].dev));
            zero_grads(&W[i].M);
        }

        for (int a = 0; a < ACCUM; a++) {
            const int r = a % G;
            Model *m = r == 0 ? &M : &W[r - 1].M;
            Acts *ac = r == 0 ? &A : &W[r - 1].A;
            CUDA_CHECK(cudaSetDevice(plan.id[r]));
            make_batch(data, 0, split - tl - 1, BATCH, tl, htok, htgt);
            upload_batch(ac, htok, htgt, n);

            const unsigned dseed = DROPOUT > 0.0f ? ((unsigned)step * 0x9E3779B1u + (unsigned)a * 0x85EBCA6Bu + 12345u) | 1u : 0u;
            forward(&m->w, V, ac, ac->tok, BATCH, tl, dseed);
            loss_and_grad_logits(ac, ac->tgt, ac->loss_train, V, n, 1.0f / ((float)n * ACCUM));
            backward(m, ac, BATCH, tl, dseed);
        }
        if (nw > 0) reduce_grads(&M, mdev, gstage, W, nw);
        CUDA_CHECK(cudaSetDevice(mdev));

        float progress = (float)step / (float)steps;
        float lr = LEARNING_RATE * (MIN_LR_RATIO + (1.0f - MIN_LR_RATIO) * 0.5f * (1.0f + cosf(3.14159265f * progress)));
        if (step < warmup) lr *= (float)step / (float)warmup;
        float ema_decay = (1.0f + (float)step) / (10.0f + (float)step);
        if (ema_decay > EMA_DECAY) ema_decay = EMA_DECAY;
        adam_step(&M, &A, lr, ema_decay);
        if (nw > 0) broadcast_weights(&M, mdev, W, nw);
        since_log++;

        int do_eval = has_val && (step % EVAL_EVERY == 0 || step == steps);
        if (step % LOG_EVERY == 0 || step == steps || do_eval) {
            float loss_sum = read_scalar(A.loss_train);
            CUDA_CHECK(cudaMemset(A.loss_train, 0, sizeof(float)));
            for (int i = 0; i < nw; i++) {
                CUDA_CHECK(cudaSetDevice(W[i].dev));
                loss_sum += read_scalar(W[i].A.loss_train);
                CUDA_CHECK(cudaMemset(W[i].A.loss_train, 0, sizeof(float)));
            }
            CUDA_CHECK(cudaSetDevice(mdev));
            float train_loss = loss_sum / ((float)since_log * ACCUM * n);
            if (!isfinite(train_loss)) {
                fprintf(stderr, "Loss diverged at step %ld\n", step);
                return 1;
            }
            double now = now_seconds();
            double ms = (now - t_last) * 1000.0 / since_log;
            since_log = 0;
            printf("step %6ld | train %.4f | lr %.5f | %.0f ms/step | %.0f tok/s", step, train_loss, lr, ms,
                   tokens_per_step / (ms / 1000.0));
            if (do_eval) {
                float v_raw = eval_loss(&M.w, V, &A, data, split, n_data, EVAL_BATCHES, htok, htgt);
                float v_ema = eval_loss(&M.e, V, &A, data, split, n_data, EVAL_BATCHES, htok, htgt);
                printf(" | val %.4f | val_ema %.4f | bpc_ema %.3f", v_raw, v_ema,
                       v_ema / logf(2.0f) / bytes_per_token);
                if (v_ema < M.best_val) {

                    M.best_val = v_ema;
                    bad_evals = 0;
                    if (save_infer(model_path, &M, &T) == 0) printf(" | saved best");
                    else fprintf(stderr, "Warning: failed to save %s\n", model_path);
                } else {
                    bad_evals++;
                    if (patience > 0) printf(" | no gain %d/%d", bad_evals, patience);
                    else printf(" | no gain %d", bad_evals);
                    if (patience > 0 && bad_evals >= patience) stop_early = 1;
                }
            }
            printf("\n");
            fflush(stdout);
            t_last = now_seconds();
        }
        if (step % SAVE_EVERY == 0 || step == steps || stop_early) {
            if (!has_val && save_infer(model_path, &M, &T) != 0)
                fprintf(stderr, "Warning: failed to save %s\n", model_path);
            if (save_ckpt(ckpt_path, &M, &T) != 0) fprintf(stderr, "Warning: failed to save %s\n", ckpt_path);
        }
        if (stop_early) {
            printf("\nEarly stop at step %ld: val_ema did not improve for %d evaluations.\n", step, patience);
            break;
        }
    }

    printf("\nTotal time: %.1f s | %s = best EMA weights%s | %s = full checkpoint (resume)\n",
           now_seconds() - t_start, model_path, has_val ? "" : " (no validation split, last weights)", ckpt_path);
    if (has_val && isfinite(M.best_val)) printf("Best val_ema: %.4f (perplexity %.1f)\n", M.best_val, expf(M.best_val));

    free(htok);
    free(htgt);
    free(data);
    for (int i = 0; i < nw; i++) {
        CUDA_CHECK(cudaSetDevice(W[i].dev));
        acts_free(&W[i].A);
        model_free(&W[i].M);
    }
    CUDA_CHECK(cudaSetDevice(mdev));
    if (gstage) cudaFree(gstage);
    acts_free(&A);
    model_free(&M);
    tok_free(&T);
    printf("\n--- Sample (from %s) ---\n", model_path);
    return run_generate(model_path, "", 300, DEF_TEMP, DEF_TOP_K, DEF_TOP_P, DEF_REP, DEF_MIN_P, DEF_NGRAM);
}

static int run_generate(const char *model_path, const char *prompt, int n_tokens, float temp,
                        int top_k, float top_p, float rep, float min_p, int ngram) {
    Model M;
    Tokenizer T;
    int status = load_infer(model_path, &M, &T);
    if (status != 0) {
        fprintf(stderr, "Cannot load %s (%s)\n", model_path,
                status == -1 ? "file not found" : "corrupt or built with different hyperparameters (architecture, QK_NORM, optimizer)");
        return 1;
    }
    Acts A;
    acts_alloc(&A, M.V);
    generate(&M.w, M.V, &A, &T, prompt, n_tokens, temp, top_k, top_p, rep, min_p, ngram);
    acts_free(&A);
    model_free(&M);
    tok_free(&T);
    return 0;
}

static const unsigned CHECK_DROP_SEED = 1234567u;

static float batch_loss(Model *M, Acts *A, int B, int tl) {
    const int n = B * tl;
    CUDA_CHECK(cudaMemset(A->loss_eval, 0, sizeof(float)));
    forward(&M->w, M->V, A, A->tok, B, tl, CHECK_DROP_SEED);
    loss_and_grad_logits(A, A->tgt, A->loss_eval, M->V, n, 1.0f / (float)n);
    return read_scalar(A->loss_eval) / (float)n;
}

static int run_gemm_test(void);
#if OPTIMIZER_MUON
static int run_muon_test(void);
#endif

static int run_check(int trials) {
    const int V = 12;
    const int B = BATCH < 2 ? BATCH : 2;
    const int tl = BLOCK_SIZE < 80 ? BLOCK_SIZE : 80;
    const int n = B * tl;
    const float eps = 5e-3f;

    Model M;
    model_alloc(&M, V);
    model_random_init(&M, 0.05f);
    Acts A;
    acts_alloc(&A, V);

    int *htok = XALLOC(int, n);
    int *htgt = XALLOC(int, n);
    for (int i = 0; i < n; i++) {
        htok[i] = (int)(rng_next() % V);
        htgt[i] = (int)(rng_next() % V);
    }
    upload_batch(&A, htok, htgt, n);

    zero_grads(&M);
    CUDA_CHECK(cudaMemset(A.loss_eval, 0, sizeof(float)));
    forward(&M.w, V, &A, A.tok, B, tl, CHECK_DROP_SEED);
    loss_and_grad_logits(&A, A.tgt, A.loss_eval, V, n, 1.0f / (float)n);
    float base = read_scalar(A.loss_eval) / (float)n;
    backward(&M, &A, B, tl, CHECK_DROP_SEED);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *hg = XALLOC(float, M.P);
    float *hw = XALLOC(float, M.P);
    CUDA_CHECK(cudaMemcpy(hg, M.dg, M.P * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hw, M.dw, M.P * sizeof(float), cudaMemcpyDeviceToHost));

    printf("loss %.5f (random guess would be %.5f), %zu parameters, dropout %.2f\n", base, logf((float)V), M.P,
           (double)DROPOUT);

    int bad = run_gemm_test(), significant = 0;
#if OPTIMIZER_MUON
    bad += run_muon_test();
#endif
    float worst = 0.0f;
    for (int k = 0; k < trials; k++) {
        size_t i;
        if (k < M.nt) {
            i = M.ti[k].off + (size_t)(rng_next() % M.ti[k].n);
        } else {
            i = (size_t)(rng_next() % M.P);
        }
        float old = hw[i];
        float plus = old + eps, minus = old - eps;
        CUDA_CHECK(cudaMemcpy(M.dw + i, &plus, sizeof(float), cudaMemcpyHostToDevice));
        float lp = batch_loss(&M, &A, B, tl);
        CUDA_CHECK(cudaMemcpy(M.dw + i, &minus, sizeof(float), cudaMemcpyHostToDevice));
        float lm = batch_loss(&M, &A, B, tl);
        CUDA_CHECK(cudaMemcpy(M.dw + i, &old, sizeof(float), cudaMemcpyHostToDevice));
        float num = (lp - lm) / (2.0f * eps);
        float ana = hg[i];
        float diff = fabsf(num - ana);
        float mag = fabsf(num) + fabsf(ana);
        if (mag > 2e-3f) {
            significant++;
            float rel = diff / mag;
            if (rel > worst) worst = rel;
            if (rel > 0.05f && diff > 2e-4f) {
                bad++;
                printf("  mismatch at %zu: numeric %.6f analytic %.6f\n", i, num, ana);
            }
        } else if (diff > 2e-4f) {
            bad++;
            printf("  mismatch at %zu: numeric %.6f analytic %.6f\n", i, num, ana);
        }
    }

    {
        float *full = XALLOC(float, V), *last = XALLOC(float, V);
        forward(&M.w, V, &A, A.tok, 1, tl);
        CUDA_CHECK(cudaMemcpy(full, A.logits + (size_t)(tl - 1) * V, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        forward(&M.w, V, &A, A.tok, 1, tl, 0, 1);
        CUDA_CHECK(cudaMemcpy(last, A.logits, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        float md = 0.0f;
        for (int v = 0; v < V; v++) md = fmaxf(md, fabsf(full[v] - last[v]));
        printf("last-row logits vs full forward: max abs diff %.2e\n", md);
        if (md > 1e-4f) {
            bad++;
            printf("  mismatch in last_only forward\n");
        }
        const int m = tl / 2;
        KVCache cache;
        kv_alloc(&cache);
        forward(&M.w, V, &A, A.tok, 1, m, 0, 1);
        kv_fill_from_prefill(&cache, &A, m);
        for (int t = m; t < tl; t++) decode_step(&M.w, V, &A, &cache, htok[t]);
        CUDA_CHECK(cudaMemcpy(last, A.logits, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
        float dd = 0.0f;
        for (int v = 0; v < V; v++) dd = fmaxf(dd, fabsf(full[v] - last[v]));
        printf("KV-cache decode vs full forward: max abs diff %.2e\n", dd);
        if (dd > 5e-5f) {
            bad++;
            printf("  mismatch in cached decode\n");
        }
        kv_free(&cache);
        free(full);
        free(last);
    }
    printf("checked %d parameters (%d with significant gradient), mismatches %d, worst relative error %.4f\n",
           trials, significant, bad, worst);
    printf(bad == 0 ? "RESULT: PASS\n" : "RESULT: FAIL\n");

    free(hg);
    free(hw);
    free(htok);
    free(htgt);
    acts_free(&A);
    model_free(&M);
    return bad == 0 ? 0 : 2;
}

static int run_gemm_test(void) {
    static const int dims[][3] = {{37, 53, 29}, {130, 200, 70}, {1, 100, 40}, {64, 64, 64}, {129, 131, 17}, {200, 72, 200}};
    static const int modes[][3] = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 1, 1}, {0, 0, 2}, {1, 0, 3}};
    int bad = 0, total = 0;
    float worst = 0.0f;
    for (int tile = 64; tile <= 128; tile *= 2) {
        g_gemm_tile = tile;
        for (int d = 0; d < 6; d++) {
            for (int mi = 0; mi < 6; mi++) {
                const int ta = modes[mi][0], tb = modes[mi][1], cz = modes[mi][2];
                int M = dims[d][0], N = dims[d][1], K = dims[d][2];
                if (cz) M = N = K = dims[d][1];
                const int lda = ta ? M : K, ldb = tb ? K : N;
                float *ha = XALLOC(float, (size_t)M * K), *hb = XALLOC(float, (size_t)K * N);
                float *hc = XALLOC(float, (size_t)M * N), *hr = XALLOC(float, (size_t)M * N);
                float *hres = XALLOC(float, (size_t)M * N);
                for (size_t i = 0; i < (size_t)M * K; i++) ha[i] = rand_normal();
                for (size_t i = 0; i < (size_t)K * N; i++) hb[i] = rand_normal();
                for (size_t i = 0; i < (size_t)M * N; i++) {
                    hc[i] = rand_normal();
                    hres[i] = rand_normal();
                }
                for (int m = 0; m < M; m++)
                    for (int k = 0; k < K; k++) {
                        const int zero = (cz == 2 && k > m) || (cz == 3 && k < m);
                        if (zero) ha[ta ? (size_t)k * lda + m : (size_t)m * lda + k] = 0.0f;
                    }
                const int use_acc = (mi == 0);
                for (int m = 0; m < M; m++)
                    for (int n = 0; n < N; n++) {
                        double s = 0.0;
                        for (int k = 0; k < K; k++) {
                            const double av = ta ? ha[(size_t)k * lda + m] : ha[(size_t)m * lda + k];
                            const double bv = tb ? hb[(size_t)n * ldb + k] : hb[(size_t)k * ldb + n];
                            s += av * bv;
                        }
                        if (use_acc) s += (double)hc[(size_t)m * N + n] + (double)hres[(size_t)m * N + n];
                        hr[(size_t)m * N + n] = (float)s;
                    }
                float *da, *db, *dc, *dr;
                CUDA_CHECK(cudaMalloc((void **)&da, (size_t)M * K * sizeof(float)));
                CUDA_CHECK(cudaMalloc((void **)&db, (size_t)K * N * sizeof(float)));
                CUDA_CHECK(cudaMalloc((void **)&dc, (size_t)M * N * sizeof(float)));
                CUDA_CHECK(cudaMalloc((void **)&dr, (size_t)M * N * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(da, ha, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(db, hb, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(dc, hc, (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(dr, hres, (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice));
                gemm(da, db, dc, use_acc ? dr : NULL, M, N, K, ta, tb, use_acc, 0.0f, 0, cz);
                CUDA_CHECK(cudaMemcpy(hc, dc, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
                float err = 0.0f, ref = 1e-6f;
                for (int m = 0; m < M; m++)
                    for (int n = 0; n < N; n++) {
                        if (cz == 1 && n > m) continue;
                        const float r = hr[(size_t)m * N + n];
                        err = fmaxf(err, fabsf(hc[(size_t)m * N + n] - r));
                        ref = fmaxf(ref, fabsf(r));
                    }
                total++;
                if (err / ref > worst) worst = err / ref;
                if (err / ref > 1e-4f) {
                    bad++;
                    printf("  GEMM mismatch: tile %d M %d N %d K %d ta %d tb %d causal %d rel err %.2e\n", tile, M, N, K,
                           ta, tb, cz, err / ref);
                }
                cudaFree(da);
                cudaFree(db);
                cudaFree(dc);
                cudaFree(dr);
                free(ha);
                free(hb);
                free(hc);
                free(hr);
                free(hres);
            }
        }
    }
    g_gemm_tile = 0;
    printf("GEMM kernels: %d configurations, mismatches %d, worst relative error %.2e\n", total, bad, worst);
    return bad;
}

#if OPTIMIZER_MUON
static int run_muon_test(void) {
    static const int shapes[][2] = {{24, 40}, {40, 24}, {32, 32}};
    int bad = 0;
    for (int si = 0; si < 3; si++) {
        const int R = shapes[si][0], C = shapes[si][1];
        const int d = R <= C ? R : C;
        const size_t n = (size_t)R * C;
        float *hx = XALLOC(float, n);
        for (size_t i = 0; i < n; i++) hx[i] = rand_normal();
        float *x, *xn, *a1, *a2, *mx, *ss;
        CUDA_CHECK(cudaMalloc((void **)&x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&xn, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&a1, (size_t)d * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&a2, (size_t)d * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&mx, (size_t)d * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ss, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(x, hx, n * sizeof(float), cudaMemcpyHostToDevice));
        const float *o = muon_orthogonalize(x, xn, a1, a2, mx, ss, R, C);
        CUDA_CHECK(cudaMemcpy(hx, o, n * sizeof(float), cudaMemcpyDeviceToHost));
        double dev = 0.0, diag = 0.0;
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) {
                double g = 0.0;
                for (int k = 0; k < (R <= C ? C : R); k++)
                    g += R <= C ? (double)hx[(size_t)i * C + k] * hx[(size_t)j * C + k]
                                : (double)hx[(size_t)k * C + i] * hx[(size_t)k * C + j];
                if (i == j) diag += g;
                dev += (g - (i == j ? 1.0 : 0.0)) * (g - (i == j ? 1.0 : 0.0));
            }
        dev = sqrt(dev / d);
        diag /= d;
        printf("Muon orthogonalization %dx%d: mean Gram diagonal %.3f, deviation from identity %.3f\n", R, C, diag, dev);
        if (diag < 0.5 || diag > 1.4 || dev > 0.5) bad++;
        cudaFree(x);
        cudaFree(xn);
        cudaFree(a1);
        cudaFree(a2);
        cudaFree(mx);
        cudaFree(ss);
        free(hx);
    }
    return bad;
}
#endif

static void fill_device_random(float *d, size_t n) {
    const size_t chunk = (size_t)1 << 20;
    float *h = XALLOC(float, chunk);
    for (size_t i = 0; i < chunk; i++) h[i] = 0.05f * rand_normal();
    for (size_t off = 0; off < n; off += chunk) {
        const size_t c = n - off < chunk ? n - off : chunk;
        CUDA_CHECK(cudaMemcpy(d + off, h, c * sizeof(float), cudaMemcpyHostToDevice));
    }
    free(h);
}

static int run_bench(int V) {
    struct Shape {
        const char *name;
        int M, N, K, ta, tb;
    };
    const int rows = MAX_ROWS;
    const Shape shapes[] = {
        {"qkv forward", rows, 3 * N_EMBD, N_EMBD, 0, 0},
        {"attention out forward", rows, N_EMBD, N_EMBD, 0, 0},
        {"ffn up forward", rows, 2 * FF_DIM, N_EMBD, 0, 0},
        {"ffn down forward", rows, N_EMBD, FF_DIM, 0, 0},
        {"lm head forward", rows, V, N_EMBD, 0, 1},
        {"ffn up dX", rows, N_EMBD, 2 * FF_DIM, 0, 1},
        {"ffn up dW", N_EMBD, 2 * FF_DIM, rows, 1, 0},
        {"lm head dW", V, N_EMBD, rows, 1, 0},
    };
    const int n_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));
    size_t maxn = 1;
    for (int i = 0; i < n_shapes; i++) {
        const Shape &s = shapes[i];
        size_t m = (size_t)s.M * s.K;
        if ((size_t)s.K * s.N > m) m = (size_t)s.K * s.N;
        if ((size_t)s.M * s.N > m) m = (size_t)s.M * s.N;
        if (m > maxn) maxn = m;
    }
    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc((void **)&a, maxn * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&b, maxn * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&c, maxn * sizeof(float)));
    fill_device_random(a, maxn);
    fill_device_random(b, maxn);
    printf("%-24s %6s %6s %6s %10s %9s\n", "GEMM", "M", "N", "K", "ms", "GFLOP/s");
    for (int i = 0; i < n_shapes; i++) {
        const Shape &s = shapes[i];
        const int reps = 5;
        gemm(a, b, c, NULL, s.M, s.N, s.K, s.ta, s.tb, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        const double t0 = now_seconds();
        for (int r = 0; r < reps; r++) gemm(a, b, c, NULL, s.M, s.N, s.K, s.ta, s.tb, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        const double ms = (now_seconds() - t0) * 1000.0 / reps;
        printf("%-24s %6d %6d %6d %10.3f %9.0f\n", s.name, s.M, s.N, s.K, ms, 2.0 * s.M * s.N * s.K / (ms * 1e6));
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    Model M;
    model_alloc(&M, V);
    model_random_init(&M, 0.0f);
    Acts A;
    acts_alloc(&A, V);
    const int n = BATCH * BLOCK_SIZE;
    int *htok = XALLOC(int, n), *htgt = XALLOC(int, n);
    for (int i = 0; i < n; i++) {
        htok[i] = (int)(rng_next() % V);
        htgt[i] = (int)(rng_next() % V);
    }
    upload_batch(&A, htok, htgt, n);
    double t_step = 0.0, t_attn = 0.0;
    for (int rep = 0; rep < 4; rep++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = now_seconds();
        forward(&M.w, V, &A, A.tok, BATCH, BLOCK_SIZE, 1234567u);
        loss_and_grad_logits(&A, A.tgt, A.loss_eval, V, n, 1.0f / (float)n);
        backward(&M, &A, BATCH, BLOCK_SIZE, 1234567u);
        CUDA_CHECK(cudaDeviceSynchronize());
        const double dt = now_seconds() - t0;
        t0 = now_seconds();
        for (int l = 0; l < N_LAYER; l++) {
            attn_fwd(A.atty[l], A.att[l], A.qkv[l], BATCH, BLOCK_SIZE);
            attn_bwd(A.dqkv, A.dscore, A.datty, A.att[l], A.qkv[l], BATCH, BLOCK_SIZE);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        if (rep > 0) {
            t_step += dt;
            t_attn += now_seconds() - t0;
        }
    }
    t_step /= 3.0;
    t_attn /= 3.0;
    printf("\nmicrobatch forward+backward: %.1f ms (%.0f tok/s on this GPU), attention part: %.1f ms (%.0f%%)\n",
           t_step * 1000.0, n / t_step, t_attn * 1000.0, 100.0 * t_attn / t_step);
    free(htok);
    free(htgt);
    acts_free(&A);
    model_free(&M);
    return 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage:\n"
            "  %s check [trials=80]        (gradient check on the device)\n"
            "  %s selftest                 (sampler / anti-loop self-test, no GPU needed)\n"
            "  %s bench [vocab=8192]       (GEMM throughput and one forward+backward microbatch)\n"
            "  %s train <corpus.txt> [steps=6000] [model.bin=model.bin] [vocab=4096] [--gpus auto|N] [--devices a,b]\n"
            "           [--seed S] [--patience N]\n"
            "  %s generate <model.bin> [prompt=\"\"] [tokens=300] [temperature=0.7] [top_k=40] [top_p=0.9]\n"
            "           [repeat_penalty=1.1] [min_p=0.05] [no_repeat_ngram=4]\n"
            "\n"
            "Training:\n"
            "  model.bin always holds the best val_ema EMA weights; model.bin.ckpt is the full resume checkpoint.\n"
            "  --patience N     stop when val_ema has not improved for N evaluations (default: never)\n"
            "\n"
            "Generation guards: no_repeat_ngram N forbids repeating any N-token sequence (0 = off), min_p drops\n"
            "tokens far less likely than the best one, repeat_penalty discourages recently used tokens.\n"
            "\n"
            "Multi-GPU (train only, data parallel over the ACCUM microbatches of a step):\n"
            "  --gpus auto      use every visible GPU with enough free memory (default when no flag is given)\n"
            "  --gpus N         use the first N GPUs; --gpus 1 forces single-GPU training\n"
            "  --devices a,b    use exactly these CUDA device ids, the first one is the master\n"
            "  --seed S         fixed random seed (reproducible batches)\n"
            "  Options may appear anywhere after the command. The GPU count is capped at ACCUM.\n",
            prog, prog, prog, prog, prog);
}

static int parse_options(int *argc, char **argv, Options *o) {
    static const char *names[] = {"--gpus", "--devices", "--seed", "--patience"};
    int w = 1;
    for (int i = 1; i < *argc; i++) {
        const char *a = argv[i];
        const char *val = NULL;
        int which = -1;
        for (int k = 0; k < 4 && which < 0; k++) {
            size_t L = strlen(names[k]);
            if (strcmp(a, names[k]) == 0) {
                if (i + 1 >= *argc) {
                    fprintf(stderr, "%s needs a value\n", names[k]);
                    return -1;
                }
                val = argv[++i];
                which = k;
            } else if (strncmp(a, names[k], L) == 0 && a[L] == '=') {
                val = a + L + 1;
                which = k;
            }
        }
        if (which < 0) {
            argv[w++] = argv[i];
            continue;
        }
        if (which == 0) o->gpus = val;
        else if (which == 1) o->devices = val;
        else if (which == 3) {
            char *end;
            long pv = strtol(val, &end, 10);
            if (!*val || *end || pv < 0) {
                fprintf(stderr, "--patience: expected a non-negative integer, got \"%s\"\n", val);
                return -1;
            }
            o->patience = (int)pv;
        } else {
            char *end;
            unsigned long long sd = strtoull(val, &end, 10);
            if (!*val || *end) {
                fprintf(stderr, "--seed: expected a non-negative integer, got \"%s\"\n", val);
                return -1;
            }
            o->has_seed = 1;
            o->seed = (uint64_t)sd;
        }
    }
    argv[w] = NULL;
    *argc = w;
    return 0;
}

int main(int argc, char **argv) {
    Options opt;
    memset(&opt, 0, sizeof(opt));
    if (argc >= 2 && (strcmp(argv[1], "train") == 0 || strcmp(argv[1], "check") == 0)) {
        if (parse_options(&argc, argv, &opt) != 0) return 1;
    }
    if (opt.has_seed) rng_state ^= (opt.seed + 1) * 0x9E3779B97F4A7C15ULL;
    else rng_state ^= (uint64_t)time(NULL) * 0x9E3779B97F4A7C15ULL;
    if (rng_state == 0) rng_state = 88172645463325252ULL;
    for (int i = 0; i < 8; i++) rng_next();

    if (argc >= 2 && strcmp(argv[1], "check") == 0) return run_check(argc > 2 ? atoi(argv[2]) : 80);
    if (argc >= 2 && strcmp(argv[1], "selftest") == 0) return run_selftest();
    if (argc >= 2 && strcmp(argv[1], "bench") == 0) return run_bench(argc > 2 ? atoi(argv[2]) : 8192);
    if (argc >= 3 && strcmp(argv[1], "train") == 0) {
        int steps = argc > 3 ? atoi(argv[3]) : 6000;
        const char *model_path = argc > 4 ? argv[4] : "model.bin";
        int vocab = argc > 5 ? atoi(argv[5]) : 4096;
        if (steps < 1) steps = 1;
        return run_train(argv[2], steps, model_path, vocab, &opt);
    }
    if (argc >= 3 && strcmp(argv[1], "generate") == 0) {
        const char *prompt = argc > 3 ? argv[3] : "";
        int length = argc > 4 ? atoi(argv[4]) : 300;
        float temp = argc > 5 ? (float)atof(argv[5]) : DEF_TEMP;
        int top_k = argc > 6 ? atoi(argv[6]) : DEF_TOP_K;
        float top_p = argc > 7 ? (float)atof(argv[7]) : DEF_TOP_P;
        float rep = argc > 8 ? (float)atof(argv[8]) : DEF_REP;
        float min_p = argc > 9 ? (float)atof(argv[9]) : DEF_MIN_P;
        int ngram = argc > 10 ? atoi(argv[10]) : DEF_NGRAM;
        return run_generate(argv[2], prompt, length, temp, top_k, top_p, rep, min_p, ngram);
    }
    print_usage(argv[0]);
    return 1;
}
