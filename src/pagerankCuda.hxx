#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "pagerank.hxx"

using std::vector;
using std::swap;




template <class T>
__global__ void pagerankFactorKernel(T *a, int *vdata, T p, int N) {
  DEFINE(t, b, B, G);
  for (int v=B*b+t, DV=G*B; v<N; v+=DV) {
    int d = vdata[v];
    a[v] = d>0? p/d : 0;
  }
}


template <class T>
__global__ void pagerankThreadKernel(T *a, T *r, T *c, int *vfrom, int *efrom, T c0, int N) {
  DEFINE(t, b, B, G);

  for (int v=B*b+t; v<N; v+=G*B) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    a[v] = c0 + sumAtKernelLoop(c, efrom+ebgn, ideg, 0, 1);
  }
}


template <class T>
int pagerankCudaLoop(int G, int B, T *e, T *r0, T *eD, T *r0D, T *&aD, T *&rD, T *cD, T *fD, int *vfromD, int *efromD, int *vdataD, int N, T p, T E, int L) {
  int G1 = G * sizeof(T), l = 1;
  for (; l<L; l++) {
    sumIfNotKernel<<<G, B>>>(r0D, rD, vdataD, N);
    multiplyKernel<<<G, B>>>(cD,  rD, fD,     N);
    TRY( cudaMemcpy(r0, r0D, G1, cudaMemcpyDeviceToHost) );
    T c0 = (1-p)/N + p*sum(r0, G)/N;
    pagerankThreadKernel<<<G, B>>>(aD, rD, cD, vfromD, efromD, c0, N);
    absErrorKernel<<<G, B>>>(eD, rD, aD, N);
    TRY( cudaMemcpy(e, eD, G1, cudaMemcpyDeviceToHost) );
    T e1 = sum(e, G);
    if (e1 < E) break;
    swap(aD, rD);
  }
  return l;
}


template <class T>
int pagerankCudaCore(int G, int B, T *e, T *r0, T *eD, T *r0D, T *&aD, T *&rD, T *cD, T *fD, int *vfromD, int *efromD, int *vdataD, int N, T p, T E, int L) {
  fillKernel<<<G, B>>>(rD, N, T(1)/N);
  pagerankFactorKernel<<<G, B>>>(fD, vdataD, p, N);
  return pagerankCudaLoop(G, B, e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, vdataD, N, p, E, L);
}


template <class H, class T=float>
PagerankResult<T> pagerankCuda(H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o=PagerankOptions<T>()) {
  int G = o.gridLimit;
  int b = o.blockSize;
  T   p = o.damping;
  T   E = o.tolerance;
  int L = o.maxIterations, l;
  int N      = xt.order();
  int g      = min(ceilDiv(N, b), G);
  auto vfrom = sourceOffsets(xt);
  auto efrom = destinationIndices(xt);
  auto vdata = vertexData(xt);
  int VFROM1 = vfrom.size() * sizeof(int);
  int EFROM1 = efrom.size() * sizeof(int);
  int VDATA1 = vdata.size() * sizeof(int);
  int G1     = g * sizeof(T);
  int N1     = N * sizeof(T);
  vector<T> a(N);

  T *e,  *r0;
  T *eD, *r0D, *fD, *rD, *cD, *aD;
  int *vfromD, *efromD, *vdataD;
  // TRY( cudaProfilerStart() );
  TRY( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY( cudaHostAlloc(&e,  G1, cudaHostAllocDefault) );
  TRY( cudaHostAlloc(&r0, G1, cudaHostAllocDefault) );
  TRY( cudaMalloc(&eD,  G1) );
  TRY( cudaMalloc(&r0D, G1) );
  TRY( cudaMalloc(&aD, N1) );
  TRY( cudaMalloc(&rD, N1) );
  TRY( cudaMalloc(&cD, N1) );
  TRY( cudaMalloc(&fD, N1) );
  TRY( cudaMalloc(&vfromD, VFROM1) );
  TRY( cudaMalloc(&efromD, EFROM1) );
  TRY( cudaMalloc(&vdataD, VDATA1) );
  TRY( cudaMemcpy(vfromD, vfrom.data(), VFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(efromD, efrom.data(), EFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(vdataD, vdata.data(), VDATA1, cudaMemcpyHostToDevice) );

  float t = measureDuration([&]() { l = pagerankCudaCore(g, b, e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, vdataD, N, p, E, L); }, o.repeat);
  TRY( cudaMemcpy(a.data(), aD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFreeHost(e) );
  TRY( cudaFreeHost(r0) );
  TRY( cudaFree(eD) );
  TRY( cudaFree(r0D) );
  TRY( cudaFree(aD) );
  TRY( cudaFree(rD) );
  TRY( cudaFree(cD) );
  TRY( cudaFree(fD) );
  TRY( cudaFree(vfromD) );
  TRY( cudaFree(efromD) );
  TRY( cudaFree(vdataD) );
  // TRY( cudaProfilerStop() );
  return {decompressContainer(xt, a), l, t};
}
