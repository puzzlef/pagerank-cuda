#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "pagerank.hxx"
#include "_main.hxx"

using std::vector;
using std::swap;
using std::min;




template <class T>
__global__ void pagerankFactorKernel(T *a, const int *vdata, int N, T p) {
  DEFINE(t, b, B, G);
  for (int v=B*b+t, DV=G*B; v<N; v+=DV) {
    int d = vdata[v];
    a[v] = d>0? p/d : 0;
  }
}

template <class T>
void pagerankFactorCu(T *a, const int *vdata, int N, T p) {
  int B = BLOCK_DIM_M;
  int G = min(ceilDiv(N, B), GRID_DIM_M);
  pagerankFactorKernel<<<G, B>>>(a, vdata, N, p);
}


template <class T>
__global__ void pagerankBlockKernel(T *a, const T *r, const T *c, const int *vfrom, const int *efrom, int N, T c0) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_LIMIT];
  for (int v=b; v<N; v+=G) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    cache[t] = sumAtKernelLoop(c, efrom+ebgn, ideg, t, B);
    sumKernelReduce(cache, B, t);
    if (t == 0) a[v] = c0 + cache[0];
  }
}

template <class T>
void pagerankBlockCu(T *a, const T *r, const T *c, const int *vfrom, const int *efrom, int N, T c0) {
  int B = BLOCK_DIM_PRB;
  int G = min(N, GRID_DIM_PRB);
  pagerankBlockKernel<<<G, B>>>(a, r, c, vfrom, efrom, N, c0);
}


template <class T>
int pagerankCudaLoop(T *e, T *r0, T *eD, T *r0D, T *&aD, T *&rD, T *cD, const T *fD, const int *vfromD, const int *efromD, const int *vdataD, int N, T p, T E, int L) {
  int R = reduceSizeCu(N);
  size_t R1 = R * sizeof(T);
  int l = 1;
  for (; l<L; l++) {
    sumIfNotCu(r0D, rD, vdataD, N);
    multiplyCu(cD, rD, fD, N);
    TRY( cudaMemcpy(r0, r0D, R1, cudaMemcpyDeviceToHost) );
    T c0 = (1-p)/N + p*sum(r0, R)/N;
    pagerankBlockCu(aD, rD, cD, vfromD, efromD, N, c0);
    l1NormCu(eD, rD, aD, N);
    TRY( cudaMemcpy(e, eD, R1, cudaMemcpyDeviceToHost) );
    T e1 = sum(e, R);
    if (e1 < E) break;
    swap(aD, rD);
  }
  return l;
}


template <class T>
int pagerankCudaCore(T *e, T *r0, T *eD, T *r0D, T *&aD, T *&rD, T *cD, T *fD, const int *vfromD, const int *efromD, const int *vdataD, int N, T p, T E, int L) {
  fillCu(rD, N, T(1)/N);
  pagerankFactorCu(fD, vdataD, N, p);
  return pagerankCudaLoop(e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, vdataD, N, p, E, L);
}


template <class H, class T=float>
PagerankResult<T> pagerankCuda(H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o=PagerankOptions<T>()) {
  typedef PagerankSort Sort;
  T    p   = o.damping;
  T    E   = o.tolerance;
  int  L   = o.maxIterations, l;
  Sort SV  = o.sortVertices;
  Sort SE  = o.sortEdges;
  int  N   = xt.order();
  int  R   = reduceSizeCu(N);
  auto fm  = [](int u) { return u; };
  auto fp  = [&](Sort S, auto ib, auto ie) {
    if (S==Sort::ASC)  sort(ib, ie, [&](int u, int v) { return xt.degree(u) < xt.degree(v); });
    if (S==Sort::DESC) sort(ib, ie, [&](int u, int v) { return xt.degree(u) > xt.degree(v); });
  };
  auto ks    = vertices(xt, fm, [&](auto ib, auto ie) { fp(SV, ib, ie); });
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks, [&](auto ib, auto ie) { fp(SE, ib, ie); });
  auto vdata = vertexData(xt, ks);
  int VFROM1 = vfrom.size() * sizeof(int);
  int EFROM1 = efrom.size() * sizeof(int);
  int VDATA1 = vdata.size() * sizeof(int);
  int N1 = N * sizeof(T);
  int R1 = R * sizeof(T);
  vector<T> a(N);

  T *e,  *r0;
  T *eD, *r0D, *fD, *rD, *cD, *aD;
  int *vfromD, *efromD, *vdataD;
  // TRY( cudaProfilerStart() );
  TRY( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY( cudaHostAlloc(&e,  R1, cudaHostAllocDefault) );
  TRY( cudaHostAlloc(&r0, R1, cudaHostAllocDefault) );
  TRY( cudaMalloc(&eD,  R1) );
  TRY( cudaMalloc(&r0D, R1) );
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

  float t = measureDuration([&]() { l = pagerankCudaCore(e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, vdataD, N, p, E, L); }, o.repeat);
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
  return {decompressContainer(xt, a, ks), l, t};
}
