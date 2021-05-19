#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

using std::fprintf;
using std::exit;




// LAUNCH CONFIG
// -------------

#define GRID_LIMIT  65535
#define BLOCK_LIMIT 1024




// TRY
// ---
// Log error if CUDA function call fails.

#ifndef TRY_CUDA
void tryCuda(cudaError err, const char* exp, const char* func, int line, const char* file) {
  if (err == cudaSuccess) return;
  fprintf(stderr,
    "%s: %s\n"
    "  in expression %s\n"
    "  at %s:%d in %s\n",
    cudaGetErrorName(err), cudaGetErrorString(err), exp, func, line, file);
  exit(err);
}

#define TRY_CUDA(exp) tryCuda(exp, #exp, __func__, __LINE__, __FILE__)
#endif

#ifndef TRY
#define TRY(exp) TRY_CUDA(exp)
#endif




// DEFINE
// ------
// Define thread, block variables.

#ifndef DEFINE_CUDA
#define DEFINE_CUDA(t, b, B, G) \
  int t = threadIdx.x; \
  int b = blockIdx.x; \
  int B = blockDim.x; \
  int G = gridDim.x;
#define DEFINE_CUDA2D(tx, ty, bx, by, BX, BY, GX, GY) \
  int tx = threadIdx.x; \
  int ty = threadIdx.y; \
  int bx = blockIdx.x; \
  int by = blockIdx.y; \
  int BX = blockDim.x; \
  int BY = blockDim.y; \
  int GX = gridDim.x;  \
  int GY = gridDim.y;
#endif

#ifndef DEFINE
#define DEFINE(t, b, B, G) \
  DEFINE_CUDA(t, b, B, G)
#define DEFINE2D(tx, ty, bx, by, BX, BY, GX, GY) \
  DEFINE_CUDA2D(tx, ty, bx, by, BX, BY, GX, GY)
#endif




// UNUSED
// ------
// Mark CUDA kernel variables as unused.

template <class T>
__device__ void unusedCuda(T&&) {}

#ifndef UNUSED_CUDA
#define UNUSED_CUDA(...) ARG_CALL(unusedCuda, ##__VA_ARGS__)
#endif

#ifndef UNUSED
#define UNUSED UNUSED_CUDA
#endif




// REMOVE IDE SQUIGGLES
// --------------------

#ifndef __SYNCTHREADS
void __syncthreads();
#define __SYNCTHREADS() __syncthreads()
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __shared__
#define __shared__
#endif
