#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>

using std::vector;
using std::unordered_map;
using std::sort;




// EDGES
// -----

template <class G>
auto edges(const G& x, int u) {
  vector<int> a;
  for (int v : x.edges(u))
    a.push_back(v);
  return a;
}

template <class G>
auto inEdges(const G& x, int v) {
  vector<int> a;
  for (int u : x.inEdges(v))
    a.push_back(u);
  return a;
}




// EDGES-DATA
// ----------

template <class G, class J>
auto edgeData(const G& x, J&& ks) {
  using E = typename G::TEdge;
  vector<E> a;
  for (int u : ks) {
    for (int v : x.edges(u))
      a.push_back(x.edgeData(u, v));
  }
  return a;
}

template <class G>
auto edgeData(const G& x) {
  return edgeData(x, x.vertices());
}




// DESTINATION-INDICES
// -------------------

template <class G, class J>
auto destinationIndices(const G& x, J&& ks) {
  typedef int key;
  vector<int> a;
  unordered_map<key, int> idx;
  int i = 0;
  for (int u : ks)
    idx[u] = i++;
  for (int u : ks) {
    for (int v : x.edges(u))
      a.push_back(idx[v]);
    // sort(a.end()-x.degree(u), a.end());
  }
  return a;
}

template <class G>
auto destinationIndices(const G& x) {
  return destinationIndices(x, x.vertices());
}
