#pragma once
#include <vector>
#include "_main.hxx"

using std::vector;




// VERTICES
// --------

template <class G>
auto vertices(const G& x) {
  vector<int> a;
  a.reserve(x.order());
  for (int u : x.vertices())
    a.push_back(u);
  return a;
}


template <class G, class F>
auto verticesBy(const G& x, F fm) {
  auto a = vertices(x);
  sort(a.begin(), a.end(), [&](int u, int v) {
    return fm(u) < fm(v);
  });
  return a;
}




// VERTEX-DATA
// -----------

template <class G, class J>
auto vertexData(const G& x, J&& ks, int N) {
  using V = typename G::TVertex;
  vector<V> a;
  if (N>0) a.reserve(N);
  for (int u : ks)
    a.push_back(x.vertexData(u));
  return a;
}

template <class G, class J>
auto vertexData(const G& x, J&& ks) {
  return vertexData(x, ks, csize(ks));
}

template <class G>
auto vertexData(G& x) {
  return vertexData(x, x.vertices(), x.order());
}




// VERTEX-CONTAINER
// ----------------

template <class G, class T, class J>
auto vertexContainer(const G& x, const vector<T>& vs, J&& ks) {
  auto a = x.vertexContainer(T()); int i = 0;
  for (auto u : ks)
    a[u] = vs[i++];
  return a;
}

template <class G, class T>
auto vertexContainer(const G& x, const vector<T>& vs) {
  return vertexContainer(x, vs, x.vertices());
}




// SOURCE-OFFSETS
// --------------

template <class G, class J>
auto sourceOffsets(const G& x, J&& ks, int N) {
  int i = 0;
  vector<int> a;
  if (N>0) a.reserve(N+1);
  for (auto u : ks) {
    a.push_back(i);
    i += x.degree(u);
  }
  a.push_back(i);
  return a;
}

template <class G, class J>
auto sourceOffsets(const G& x, J&& ks) {
  return sourceOffsets(x, ks, csize(ks));
}

template <class G>
auto sourceOffsets(const G& x) {
  return sourceOffsets(x, x.vertices(), x.order());
}
