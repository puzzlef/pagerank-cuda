#pragma once
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "pagerank.hxx"

using std::swap;




template <class T>
T pagerankTeleportOmp(const vector<T>& r, const vector<int>& vfrom, const vector<int>& efrom, const vector<int>& vdata, int N, T p) {
  T a = (1-p)/N;
  #pragma omp parallel for schedule(static,4096) reduction(+:a)
  for (int u=0; u<N; u++)
    if (vdata[u] == 0) a += p*r[u]/N;
  return a;
}

template <class T>
void pagerankFactorOmp(vector<T>& a, const vector<int>& vfrom, const vector<int>& efrom, const vector<int>& vdata, int N, T p) {
  #pragma omp parallel for schedule(static,4096)
  for (int u=0; u<N; u++) {
    int d = vdata[u];
    a[u] = d>0? p/d : 0;
  }
}

template <class T>
void pagerankOmpOnce(vector<T>& a, const vector<T>& c, const vector<int>& vfrom, const vector<int>& efrom, const vector<int>& vdata, int N, T c0) {
  #pragma omp parallel for schedule(static,4096)
  for (int v=0; v<N; v++)
    a[v] = c0 + sumAt(c, slice(efrom, vfrom[v], vfrom[v+1]));
}

template <class T>
int pagerankOmpLoop(vector<T>& a, vector<T>& r, const vector<T>& f, vector<T>& c, const vector<int>& vfrom, const vector<int>& efrom, const vector<int>& vdata, int N, T p, T E, int L) {
  int l = 0;
  T e0 = T();
  for (; l<L; l++) {
    T c0 = pagerankTeleportOmp(r, vfrom, efrom, vdata, N, p);
    multiplyOmp(c, r, f);
    pagerankOmpOnce(a, c, vfrom, efrom, vdata, N, c0);
    T e1 = absErrorOmp(a, r);
    if (e1 < E || e1 == e0) break;
    swap(a, r);
    e0 = e1;
  }
  return l;
}

template <class T>
int pagerankOmpCore(vector<T>& a, vector<T>& r, vector<T>& f, vector<T>& c, const vector<int>& vfrom, const vector<int>& efrom, const vector<int>& vdata, int N, const vector<T> *q, T p, T E, int L) {
  if (q) copyOmp(r, *q);
  else fillOmp(r, T(1)/N);
  pagerankFactorOmp(f, vfrom, efrom, vdata, N, p);
  return pagerankOmpLoop(a, r, f, c, vfrom, efrom, vdata, N, p, E, L);
}


// Find pagerank accelerated using OpenMP.
// @param xt transpose graph, with vertex-data=out-degree
// @param q initial ranks (optional)
// @param o options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class T=float>
PagerankResult<T> pagerankOmp(const G& xt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  T    p = o.damping;
  T    E = o.tolerance;
  int  L = o.maxIterations, l;
  auto vfrom = sourceOffsets(xt);
  auto efrom = destinationIndices(xt);
  auto vdata = vertexData(xt);
  int  N     = xt.order();
  vector<T> a(N), r(N), f(N), c(N);
  float t = measureDuration([&]() { l = pagerankOmpCore(a, r, f, c, vfrom, efrom, vdata, N, q, p, E, L); }, o.repeat);
  return {vertexContainer(xt, a), l, t};
}
