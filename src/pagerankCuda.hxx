#pragma once
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "pagerank.hxx"

using std::array;
using std::vector;
using std::sqrt;
using std::partition;
using std::swap;
using std::min;
using std::max;




// PAGERANK-PARTITON
// -----------------

template <class H>
void pagerankPartition(const H& xt, vector<int>& ks, int i, int n) {
  auto ib = ks.begin()+i, ie = ib+n;
  partition(ib, ie, [&](int u) { return xt.degree(u) < SWITCH_DEGREE_PRC(); });
}

template <class H>
void pagerankPartition(const H& xt, vector<int>& ks) {
  pagerankPartition(xt, ks, 0, ks.size());
}




// PAGERANK-FACTOR
// ---------------
// For contribution factors of vertices (unchanging).

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




// PAGERANK-BLOCK
// --------------

template <class T, int S=BLOCK_LIMIT>
__global__ void pagerankBlockKernel(T *a, const T *c, const int *vfrom, const int *efrom, int i, int n, T c0) {
  DEFINE(t, b, B, G);
  __shared__ T cache[S];
  for (int v=i+b; v<i+n; v+=G) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    cache[t] = sumAtKernelLoop(c, efrom+ebgn, ideg, t, B);
    sumKernelReduce(cache, B, t);
    if (t==0) a[v] = c0 + cache[0];
  }
}

template <class T>
void pagerankBlockCu(T *a, const T *c, const int *vfrom, const int *efrom, int i, int n, T c0) {
  int B = BLOCK_DIM_PRCB<T>();
  int G = min(n, GRID_DIM_PRCB<T>());
  pagerankBlockKernel<<<G, B>>>(a, c, vfrom, efrom, i, n, c0);
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
  int B = BLOCK_DIM_PRCT<T>();
  int G = min(ceilDiv(n, B), GRID_DIM_PRCT<T>());
  pagerankThreadKernel<<<G, B>>>(a, c, vfrom, efrom, i, n, c0);
}




// PAGERANK-SWITCHED
// -----------------

template <class T, class J>
void pagerankSwitchedCu(T *a, const T *c, const int *vfrom, const int *efrom, int i, const J& ns, T c0) {
  for (int n : ns) {
    if (n>0) pagerankBlockCu (a, c, vfrom, efrom, i,  n, c0);
    else     pagerankThreadCu(a, c, vfrom, efrom, i, -n, c0);
    i += abs(n);
  }
}




// PAGERANK-SWITCH-POINT
// ---------------------

template <class H, class J>
int pagerankSwitchPoint(const H& xt, const J& ks) {
  int a = countIf(ks, [&](int u) { return xt.degree(u) < SWITCH_DEGREE_PRC(); });
  int L = SWITCH_LIMIT_PRC(), N = ks.size();
  return a<L? 0 : (N-a<L? N : a);
}




// PAGERANK-WAVE
// -------------

template <class H>
void pagerankPairWave(vector<array<int,2>>& a, const H& xt, const vector2d<int>& cs) {
  for (const auto& ks : cs) {
    int N = ks.size();
    int s = pagerankSwitchPoint(xt, ks);
    a.push_back({s, N-s});
  }
}

template <class H>
auto pagerankPairWave(const H& xt, const vector2d<int>& cs) {
  vector<array<int,2>> a; pagerankPairWave(a, xt, cs);
  return a;
}


void pagerankAddStep(vector<int>& a, int n) {
  if (n==0) return;
  if (a.empty() || sgn(a.back())!=sgn(n)) a.push_back(n);
  else a.back() += n;
}

template <class H, class J>
void pagerankWave(vector<int>& a, const H& xt, const J& ks) {
  int N = ks.size();
  int s = pagerankSwitchPoint(xt, ks);
  pagerankAddStep(a,  -s);
  pagerankAddStep(a, N-s);
}

template <class H, class J>
auto pagerankWave(const H& xt, const J& ks) {
  vector<int> a; pagerankWave(a, xt, ks);
  return a;
}

template <class H, class J>
void pagerankComponentWave(vector<int>& a, const H& xt, const J& cs) {
  for (const auto& ks : cs)
    pagerankWave(a, xt, ks);
}
template <class H, class J>
auto pagerankComponentWave(const H& xt, const J& cs) {
  vector<int> a; pagerankComponentWave(a, xt, cs);
  return a;
}




// PAGERANK-ERROR
// --------------
// For convergence check.

template <class T>
void pagerankErrorCu(T *a, const T *x, const T *y, int N, int EF) {
  switch (EF) {
    case 1:  l1NormCu(a, x, y, N); break;
    case 2:  l2NormCu(a, x, y, N); break;
    default: liNormCu(a, x, y, N); break;
  }
}

template <class T>
T pagerankErrorReduce(const T *x, int N, int EF) {
  switch (EF) {
    case 1:  return sum(x, N);
    case 2:  return sqrt(sum(x, N));
    default: return max(x, N);
  }
}




// PAGERANK
// --------
// For Monolithic / Componentwise PageRank.

template <class H, class J, class M, class FL, class T=float>
PagerankResult<T> pagerankCuda(const H& xt, const J& ks, int i, const M& ns, FL fl, const vector<T> *q, const PagerankOptions<T>& o) {
  int  N  = xt.order();
  T    p  = o.damping;
  T    E  = o.tolerance;
  int  L  = o.maxIterations, l = 0;
  int  EF = o.toleranceNorm;
  int  R  = reduceSizeCu<T>(N);
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks);
  auto vdata = vertexData(xt, ks);
  int VFROM1 = vfrom.size() * sizeof(int);
  int EFROM1 = efrom.size() * sizeof(int);
  int VDATA1 = vdata.size() * sizeof(int);
  int N1 = N * sizeof(T);
  int R1 = R * sizeof(T);
  vector<T> a(N), r(N), qc;
  if (q) qc = compressContainer(xt, *q, ks);

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
    if (q) copy(r, qc);    // copy old ranks (qc), if given
    else fill(r, T(1)/N);
    TRY( cudaMemcpy(aD, r.data(), N1, cudaMemcpyHostToDevice) );
    TRY( cudaMemcpy(rD, r.data(), N1, cudaMemcpyHostToDevice) );
    mark([&] { pagerankFactorCu(fD, vdataD, 0, N, p); multiplyCu(cD, aD, fD, N); });                       // calculate factors (fD) and contributions (cD)
    mark([&] { l = fl(e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, vdataD, i, ns, N, p, E, L, EF); });  // calculate ranks of vertices
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
