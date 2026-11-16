#ifndef CUDA_EMU_H
#define CUDA_EMU_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>

#include <functional>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __shared__ static
#define __forceinline__ inline
#define __launch_bounds__(...)

struct float4 {
    float x, y, z, w;
};

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};

static dim3 threadIdx, blockIdx, blockDim, gridDim;

typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorInvalidDevice = 101, cudaErrorPeerAccessAlreadyEnabled = 704 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };

struct cudaDeviceProp {
    char name[256];
};

static inline const char *cudaGetErrorString(cudaError_t) { return "emulation error"; }

static inline cudaError_t cudaMalloc(void **p, size_t n) {
    *p = malloc(n ? n : 1);
    return *p ? cudaSuccess : 1;
}

static inline cudaError_t cudaFree(void *p) {
    free(p);
    return cudaSuccess;
}

static inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, cudaMemcpyKind) {
    memcpy(dst, src, n);
    return cudaSuccess;
}

static inline cudaError_t cudaMemset(void *p, int v, size_t n) {
    memset(p, v, n);
    return cudaSuccess;
}

static inline cudaError_t cudaDeviceSynchronize(void) { return cudaSuccess; }

// Every emulated device is the host; only the device bookkeeping (count, current device, peer copies) is real.
// EMU_DEVICES sets the device count, EMU_FREE_MB the free memory each device reports.
static int emu_current_device = 0;

static inline int emu_device_count(void) {
    const char *e = getenv("EMU_DEVICES");
    int n = e ? atoi(e) : 1;
    return n < 1 ? 1 : n;
}

static inline cudaError_t cudaGetDeviceCount(int *n) {
    *n = emu_device_count();
    return cudaSuccess;
}

static inline cudaError_t cudaSetDevice(int d) {
    if (d < 0 || d >= emu_device_count()) return cudaErrorInvalidDevice;
    emu_current_device = d;
    return cudaSuccess;
}

static inline cudaError_t cudaGetDevice(int *d) {
    *d = emu_current_device;
    return cudaSuccess;
}

static inline cudaError_t cudaGetLastError(void) { return cudaSuccess; }

static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int d) {
    snprintf(prop->name, sizeof(prop->name), "CPU emulation %d", d);
    return cudaSuccess;
}

static inline cudaError_t cudaMemGetInfo(size_t *free_b, size_t *total_b) {
    const char *e = getenv("EMU_FREE_MB");
    size_t mb = e ? (size_t)atoll(e) : (size_t)(1 << 20);
    *free_b = mb << 20;
    *total_b = (size_t)(1 << 20) << 20;
    return cudaSuccess;
}

static inline cudaError_t cudaMemcpyPeer(void *dst, int, const void *src, int, size_t n) {
    memcpy(dst, src, n);
    return cudaSuccess;
}

static inline cudaError_t cudaDeviceCanAccessPeer(int *can, int, int) {
    *can = 1;
    return cudaSuccess;
}

static inline cudaError_t cudaDeviceEnablePeerAccess(int, unsigned) { return cudaSuccess; }

static inline float atomicAdd(float *addr, float v) {
    float old = *addr;
    *addr = old + v;
    return old;
}

struct EmuFiber {
    ucontext_t ctx;
    char *stack;
    bool done;
    dim3 tid;
};

static ucontext_t emu_sched;
static EmuFiber *emu_cur = NULL;
static const std::function<void()> *emu_fn = NULL;
static std::vector<EmuFiber *> emu_fibers;
static const size_t EMU_STACK = 1 << 16;

static void emu_entry(void) {
    (*emu_fn)();
    emu_cur->done = true;
    swapcontext(&emu_cur->ctx, &emu_sched);
}

static inline void __syncthreads(void) { swapcontext(&emu_cur->ctx, &emu_sched); }

static void emu_launch(dim3 grid, dim3 block, const std::function<void()> &fn) {
    const int nt = (int)(block.x * block.y * block.z);
    while ((int)emu_fibers.size() < nt) {
        EmuFiber *f = new EmuFiber();
        f->stack = (char *)malloc(EMU_STACK);
        emu_fibers.push_back(f);
    }
    gridDim = grid;
    blockDim = block;
    emu_fn = &fn;
    for (unsigned bz = 0; bz < grid.z; bz++) {
        for (unsigned by = 0; by < grid.y; by++) {
            for (unsigned bx = 0; bx < grid.x; bx++) {
                blockIdx = dim3(bx, by, bz);
                for (int t = 0; t < nt; t++) {
                    EmuFiber *f = emu_fibers[(size_t)t];
                    f->done = false;
                    f->tid = dim3((unsigned)t % block.x, ((unsigned)t / block.x) % block.y,
                                  (unsigned)t / (block.x * block.y));
                    getcontext(&f->ctx);
                    f->ctx.uc_stack.ss_sp = f->stack;
                    f->ctx.uc_stack.ss_size = EMU_STACK;
                    f->ctx.uc_link = NULL;
                    makecontext(&f->ctx, emu_entry, 0);
                }
                int remaining = nt;
                while (remaining > 0) {
                    for (int t = 0; t < nt; t++) {
                        EmuFiber *f = emu_fibers[(size_t)t];
                        if (f->done) continue;
                        emu_cur = f;
                        threadIdx = f->tid;
                        swapcontext(&emu_sched, &f->ctx);
                        if (f->done) remaining--;
                    }
                }
            }
        }
    }
}

#endif
