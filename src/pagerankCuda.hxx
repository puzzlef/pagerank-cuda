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
using std::min;




// PAGERANK-FACTOR
// ---------------

template <class T>
__global__ void pagerankFactorKernel(T *a, const int *vdata, int i, int n, T p) {
  DEFINE(t, b, B, G);
  for (int v=i+B*b+t; v<i+n; v+=G*B) {
    int d = vdata[v];
    a[v] = d>0? p/d : 0;
  }
}

template <class T>
void pagerankFactorCu(T *a, const int *vdata, int i, int n, T p) {
  int B = BLOCK_DIM_M<T>();
  int G = min(ceilDiv(n, B), GRID_DIM_M<T>());
  pagerankFactorKernel<<<G, B>>>(a, vdata, i, n, p);
}




// PAGERANK-THREAD
// ---------------

template <class T>
__global__ void pagerankThreadKernel(T *a, const T *c, const int *vfrom, const int *efrom, int i, int n, T c0) {
  DEFINE(t, b, B, G);
  for (int v=i+B*b+t; v<i+n; v+=G*B) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    a[v] = c0 + sumAtKernelLoop(c, efrom+ebgn, ideg, 0, 1);
  }
}

template <class T>
void pagerankThreadCu(T *a, const T *c, const int *vfrom, const int *efrom, int i, int n, T c0) {
  int B = BLOCK_DIM_PRT<T>();
  int G = min(ceilDiv(n, B), GRID_DIM_PRT<T>());
  pagerankThreadKernel<<<G, B>>>(a, c, vfrom, efrom, i, n, c0);
}

template <class G, class J>
auto pagerankWave(const G& xt, J&& ks) {
  vector<int> a {int(ks.size())};
  return a;
}




// PAGERANK (CUDA)
// ---------------

template <class T, class J>
int pagerankCudaLoop(T *e, T *r0, T *eD, T *r0D, T *&aD, T *&rD, T *cD, const T *fD, const int *vfromD, const int *efromD, const int *vdataD, int i, J&& ns, int N, T p, T E, int L) {
  int n = sumAbs(ns);
  int R = reduceSizeCu<T>(N);
  size_t R1 = R * sizeof(T);
  int l = 1;
  for (; l<L; l++) {
    sumIfNotCu(r0D, rD, vdataD, N);
    multiplyCu(cD+i, rD+i, fD+i, n);
    TRY( cudaMemcpy(r0, r0D, R1, cudaMemcpyDeviceToHost) );
    T c0 = (1-p)/N + p*sum(r0, R)/N;
    pagerankThreadCu(aD, cD, vfromD, efromD, i, n, c0);
    l1NormCu(eD, rD+i, aD+i, n);
    TRY( cudaMemcpy(e, eD, R1, cudaMemcpyDeviceToHost) );
    T el = sum(e, R);
    if (el < E) break;
    swap(aD, rD);
  }
  return l;
}


template <class H, class T=float>
PagerankResult<T> pagerankCuda(const H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o=PagerankOptions<T>()) {
  typedef PagerankSort Sort;
  T    p   = o.damping;
  T    E   = o.tolerance;
  int  L   = o.maxIterations, l;
  Sort SV  = o.sortVertices;
  Sort SE  = o.sortEdges;
  int  N   = xt.order();
  int  R   = reduceSizeCu<T>(N);
  auto fm  = [](int u) { return u; };
  auto fp  = [&](Sort S, auto ib, auto ie) {
    if (S==Sort::ASC)  sort(ib, ie, [&](int u, int v) { return xt.degree(u) < xt.degree(v); });
    if (S==Sort::DESC) sort(ib, ie, [&](int u, int v) { return xt.degree(u) > xt.degree(v); });
  };
  auto ks    = vertices(xt, fm, [&](auto ib, auto ie) { fp(SV, ib, ie); });
  auto ns    = pagerankWave(xt, ks);
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks, [&](auto ib, auto ie) { fp(SE, ib, ie); });
  auto vdata = vertexData(xt, ks);
  int VFROM1 = vfrom.size() * sizeof(int);
  int EFROM1 = efrom.size() * sizeof(int);
  int VDATA1 = vdata.size() * sizeof(int);
  int N1 = N * sizeof(T);
  int R1 = R * sizeof(T);
  vector<T> a(N), r(N);

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

  float t = measureDurationMarked([&](auto mark) {
    if (q) r = compressContainer(xt, *q, ks);
    else fill(r, T(1)/N);
    TRY( cudaMemcpy(aD, r.data(), N1, cudaMemcpyHostToDevice) );
    TRY( cudaMemcpy(rD, r.data(), N1, cudaMemcpyHostToDevice) );
    mark([&] { pagerankFactorCu(fD, vdataD, 0, N, p); });
    mark([&] { l = pagerankCudaLoop(e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, vdataD, 0, ns, N, p, E, L); });
  }, o.repeat);
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
