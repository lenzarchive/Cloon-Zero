#ifndef CONFIG_H
#define CONFIG_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef CUDA_EMULATION
#include "cuda_emu.h"
#else
#include <cuda_runtime.h>
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef N_EMBD
#define N_EMBD 384
#endif
#ifndef N_HEAD
#define N_HEAD 6
#endif
#ifndef N_LAYER
#define N_LAYER 6
#endif
#ifndef BATCH
#define BATCH 16
#endif
#ifndef ACCUM
#define ACCUM 2
#endif
#ifndef LEARNING_RATE
#define LEARNING_RATE 8e-4f
#endif
#ifndef MIN_LR_RATIO
#define MIN_LR_RATIO 0.1f
#endif
#ifndef WEIGHT_DECAY
#define WEIGHT_DECAY 0.1f
#endif
#ifndef GRAD_CLIP
#define GRAD_CLIP 1.0f
#endif
#ifndef EMA_DECAY
#define EMA_DECAY 0.999f
#endif
#ifndef ROPE_THETA
#define ROPE_THETA 10000.0f
#endif
#ifndef DROPOUT
#define DROPOUT 0.1f
#endif
#ifndef LOG_EVERY
#define LOG_EVERY 50
#endif
#ifndef EVAL_EVERY
#define EVAL_EVERY 250
#endif
#ifndef EVAL_BATCHES
#define EVAL_BATCHES 16
#endif
#ifndef SAVE_EVERY
#define SAVE_EVERY 1000
#endif
#ifndef EVAL_PATIENCE
#define EVAL_PATIENCE 0
#endif
#ifndef MAX_GPUS
#define MAX_GPUS 16
#endif
#ifndef DEFAULT_GPUS
#define DEFAULT_GPUS "auto"
#endif
#ifndef REP_WINDOW
#define REP_WINDOW 128
#endif
#ifndef DEF_TEMP
#define DEF_TEMP 0.7f
#endif
#ifndef DEF_TOP_K
#define DEF_TOP_K 40
#endif
#ifndef DEF_TOP_P
#define DEF_TOP_P 0.9f
#endif
#ifndef DEF_REP
#define DEF_REP 1.1f
#endif
#ifndef DEF_MIN_P
#define DEF_MIN_P 0.05f
#endif
#ifndef DEF_NGRAM
#define DEF_NGRAM 4
#endif
#ifndef EW_THREADS
#define EW_THREADS 256
#endif
#ifndef LN_THREADS
#define LN_THREADS 128
#endif
#ifndef ATT_THREADS
#define ATT_THREADS 128
#endif
#ifndef XENT_THREADS
#define XENT_THREADS 128
#endif
#ifndef QK_NORM
#define QK_NORM 0
#endif
#ifndef OPTIMIZER_MUON
#define OPTIMIZER_MUON 0
#endif
#ifndef MUON_MOMENTUM
#define MUON_MOMENTUM 0.95f
#endif
#ifndef MUON_NS_STEPS
#define MUON_NS_STEPS 5
#endif
#ifndef MUON_UPDATE_SCALE
#define MUON_UPDATE_SCALE 0.2f
#endif
#ifndef GEMM_BIG_MIN_BLOCKS
#define GEMM_BIG_MIN_BLOCKS 80
#endif

#define HEAD_SIZE (N_EMBD / N_HEAD)
#define HALF_HEAD (HEAD_SIZE / 2)
#define FF_DIM (((8 * N_EMBD / 3) + 63) / 64 * 64)
#define MAX_ROWS (BATCH * BLOCK_SIZE)
#define MAGIC 0x47505433u
#define MAX_TENSORS (2 + 6 * N_LAYER)

#define STATIC_ASSERT(name, cond) typedef char static_assert_##name[(cond) ? 1 : -1]
#define IS_POW2(x) (((x) & ((x) - 1)) == 0)
STATIC_ASSERT(heads_divide_embd, N_EMBD % N_HEAD == 0);
STATIC_ASSERT(head_even, HEAD_SIZE % 2 == 0);
STATIC_ASSERT(ew_pow2, IS_POW2(EW_THREADS));
STATIC_ASSERT(ln_pow2, IS_POW2(LN_THREADS));
STATIC_ASSERT(att_pow2, IS_POW2(ATT_THREADS));
STATIC_ASSERT(xent_pow2, IS_POW2(XENT_THREADS));

#ifndef LAUNCH
#ifdef CUDA_EMULATION
#define LAUNCH(kernel, grid, block, ...) emu_launch(dim3(grid), dim3(block), [&]() { kernel(__VA_ARGS__); })
#else
#define LAUNCH(kernel, grid, block, ...) kernel<<<grid, block>>>(__VA_ARGS__)
#endif
#endif

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                             \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    return p;
}

#define XALLOC(type, n) ((type *)xcalloc((size_t)(n), sizeof(type)))

#endif
