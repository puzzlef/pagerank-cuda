#pragma once
#include <utility>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "components.hxx"

using std::vector;
using std::unordered_set;
using std::max;
using std::make_pair;




// ADJUST-RANKS
// ------------
// For calculating inital ranks for incremental/dynamic pagerank.

template <class T, class J>
void adjustRanks(vector<T>& a, const vector<T>& r, const J& Kx, const J& Ky, T radd, T rmul, T rset) {
  unordered_set<int> Sx(Kx.begin(), Kx.end());
  for (int u : Ky)
    a[u] = Sx.count(u)==0? rset : (r[u]+radd)*rmul;  // vertex new/old
}

template <class T, class J>
auto adjustRanks(int N, const vector<T>& r, const J& ks0, const J& ks1, T radd, T rmul, T rset) {
  vector<T> a(N); adjustRanks(a, r, ks0, ks1, radd, rmul, rset);
  return a;
}




// CHANGED-VERTICES
// ----------------
// Find vertices with edges added/removed.

template <class G>
bool isChangedVertex(const G& x, const G& y, int u) {
  return !x.hasVertex(u) || !verticesEqual(x, u, y, u);
}
template <class G, class H>
bool isChangedVertex(const G& x, const H& xt, const G& y, const H& yt, int u) {
  return !x.hasVertex(u) || !verticesEqual(x, xt, u, y, yt, u);  // both ways
}

template <class G, class F>
void changedVerticesDo(const G& x, const G& y, F fn) {
  for (int u : y.vertices())
    if (isChangedVertex(x, y, u)) fn(u);
}
template <class G, class H, class F>
void changedVerticesDo(const G& x, const H& xt, const G& y, const H& yt, F fn) {
  for (int u : y.vertices())
    if (isChangedVertex(x, xt, y, yt, u)) fn(u);
}

template <class G>
auto changedVertices(const G& x, const G& y) {
  vector<int> a; changedVerticesDo(x, y, [&](int u) { a.push_back(u); });
  return a;
}
template <class G, class H>
auto changedVertices(const G& x, const H& xt, const G& y, const H& yt) {
  vector<int> a; changedVerticesDo(x, xt, y, yt, [&](int u) { a.push_back(u); });
  return a;
}




// AFFECTED-VERTICES
// -----------------
// Find vertices reachable from changed vertices.

template <class G>
bool hasAffectedDeadEnd(const G& x, const G& y, const vector<bool>& vis) {
  for (int u : x.vertices())
    if (isDeadEnd(x, u) && !y.hasVertex(u)) return true;
  for (int u : y.vertices())
    if (isDeadEnd(y, u) && vis[u]) return true;
  return false;
}


template <class G, class H>
bool affectedVerticesMark(vector<bool>& vis, const G& x, const H& xt, const G& y, const H& yt) {
  changedVerticesDo(x, xt, y, yt, [&](int u) { dfsMarkLoop(vis, y, u); });
  return hasAffectedDeadEnd(x, y, vis);
}

template <class G, class H>
bool affectedInVerticesMark(vector<bool>& vis, const G& x, const H& xt, const G& y, const H& yt) {
  changedVerticesDo(xt, yt, [&](int u) { dfsMarkLoop(vis, y, u); });
  return hasAffectedDeadEnd(x, y, vis);
}

template <class G>
bool affectedOutVerticesMark(vector<bool>& vis, const G& x, const G& y) {
  changedVerticesDo(x, y, [&](int u) { dfsMarkLoop(vis, y, u); });
  return hasAffectedDeadEnd(x, y, vis);
}


template <class G, class F>
void affectedVerticesDoInt(const G& x, const G& y, const vector<bool>& vis, F fn) {
  if (hasAffectedDeadEnd(x, y, vis)) forEach(y.vertices(), fn);
  else forEach(y.vertices(), [&](int u) { if (vis[u]) fn(u); });
}

template <class G, class H, class F>
void affectedVerticesDo(const G& x, const H& xt, const G& y, const H& yt, F fn) {
  auto vis = createContainer(y, bool());
  changedVerticesDo(x, xt, y, yt, [&](int u) { dfsMarkLoop(vis, y, u); });
  affectedVerticesDoInt(x, y, vis, fn);
}

template <class G, class H, class F>
void affectedInVerticesDo(const G& x, const H& xt, const G& y, const H& yt, F fn) {
  auto vis = createContainer(y, bool());
  changedVerticesDo(xt, yt, [&](int u) { dfsMarkLoop(vis, y, u); });
  affectedVerticesDoInt(x, y, vis, fn);
}

template <class G, class H, class F>
void affectedOutVerticesDo(const G& x, const G& y, F fn) {
  auto vis = createContainer(y, bool());
  changedVerticesDo(x, y, [&](int u) { dfsMarkLoop(vis, y, u); });
  affectedVerticesDoInt(x, y, vis, fn);
}


template <class G, class H>
auto affectedVertices(const G& x, const H& xt, const G& y, const H& yt) {
  vector<int> a; affectedVerticesDo(x, xt, y, yt, [&](int u) { a.push_back(u); });
  return a;
}
template <class G, class H>
auto affectedInVertices(const G& x, const H& xt, const G& y, const H& yt) {
  vector<int> a; affectedInVerticesDo(x, xt, y, yt, [&](int u) { a.push_back(u); });
  return a;
}
template <class G>
auto affectedOutVertices(const G& x, const G& y) {
  vector<int> a; affectedOutVerticesDo(x, y, [&](int u) { a.push_back(u); });
  return a;
}




// DYNAMIC-VERTICES
// ----------------
// Find affected, unaffected vertices (vertices, no. affected).


template <class G, class FA>
auto dynamicVerticesByMark(const G& y, FA fa) {
  auto vis = createContainer(y, bool());
  if(fa(vis)) return make_pair(vertices(y), y.order());
  vector<int> a; int n = 0;
  for (int u : y.vertices())
    if (vis[u]) { a.push_back(u); ++n; }
  return make_pair(a, n);
}

template <class G, class FA>
auto dynamicVerticesBy(const G& y, FA fa) {
  vector<int> a; unordered_set<int> aff;
  fa([&](int u) { a.push_back(u); aff.insert(u); });
  for (int u : y.vertices())
    if (aff.count(u)==0) a.push_back(u);
  return make_pair(a, aff.size());
}


template <class G, class H>
auto dynamicVertices(const G& x, const H& xt, const G& y, const H& yt) {
  return dynamicVerticesByMark(y, [&](auto& vis) { return affectedVerticesMark(vis, x, xt, y, yt); });
}
template <class G, class H>
auto dynamicInVertices(const G& x, const H& xt, const G& y, const H& yt) {
  return dynamicVerticesByMark(y, [&](auto& vis) { return affectedInVerticesMark(vis, x, xt, y, yt); });
}
template <class G>
auto dynamicOutVertices(const G& x, const G& y) {
  return dynamicVerticesByMark(y, [&](auto& vis) { return affectedOutVerticesMark(vis, x, y); });
}




// CHANGED-COMPONENTS (TOFIX!)
// ---------------------------
// Find components with edges added/removed.

template <class G, class F>
void changedComponentIndicesForEach(const G& x, const G& y, const vector2d<int>& cs, F fn) {
  for (int i=0, I=cs.size(); i<I; ++i)
    if (!componentsEqual(x, cs[i], y, cs[i])) fn(i);
}

template <class G, class H, class F>
void changedComponentIndicesForEach(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs, F fn) {
  for (int i=0, I=cs.size(); i<I; ++i)
    if (!componentsEqual(x, xt, cs[i], y, yt, cs[i])) fn(i);  // both ways
}

template <class G, class H, class F>
void changedInComponentIndicesForEach(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs, F fn) {
  return changedComponentIndicesForEach(xt, yt, cs, fn);
}
template <class G, class H, class F>
void changedOutComponentIndicesForEach(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs, F fn) {
  return changedComponentIndicesForEach(x, y, cs, fn);
}


template <class G>
auto changedComponentIndices(const G& x, const G& y, const vector2d<int>& cs) {
  vector<int> a; changedComponentIndicesForEach(x, y, cs, [&](int u) { a.push_back(u); });
  return a;
}
template <class G, class H>
auto changedComponentIndices(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs) {
  vector<int> a; changedVerticesForEach(x, xt, y, yt, cs, [&](int u) { a.push_back(u); });
  return a;
}
template <class G, class H>
auto changedInComponentIndices(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs) {
  return changedComponentIndices(xt, yt, cs);
}
template <class G, class H>
auto changedOutComponentIndices(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs) {
  return changedComponentIndices(x, y, cs);
}




// AFFECTED-COMPONENTS
// -------------------
// Find components reachable from changed components.

template <class G, class B, class F>
void affectedComponentIndicesForEach(const G& x, const G& y, const vector2d<int>& cs, const B& b, F fn) {
  auto vis = createContainer(b, bool());
  changedComponentIndicesForEach(x, y, cs, [&](int u) { dfsDoLoop(vis, b, u, fn); });
}

template <class G, class H, class B, class F>
void affectedComponentIndicesForEach(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs, const B& b, F fn) {
  auto vis = createContainer(b, bool());
  changedComponentIndicesForEach(x, xt, y, yt, cs, [&](int u) { dfsDoLoop(vis, b, u, fn); });
}

template <class G, class H, class B, class F>
void affectedInComponentIndicesForEach(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs, const B& b, F fn) {
  auto vis = createContainer(b, bool());
  changedInComponentIndicesForEach(x, xt, y, yt, cs, [&](int u) { dfsDoLoop(vis, b, u, fn); });
}

template <class G, class H, class B, class F>
void affectedOutComponentIndicesForEach(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs, const B& b, F fn) {
  auto vis = createContainer(b, bool());
  changedOutComponentIndicesForEach(x, xt, y, yt, cs, [&](int u) { dfsDoLoop(vis, b, u, fn); });
}


template <class G, class B>
auto affectedComponentIndices(const G& x, const G& y, const vector2d<int>& cs, const B& b) {
  vector<int> a; affectedComponentIndicesForEach(x, y, cs, b, [&](int u) { a.push_back(u); });
  return a;
}
template <class G, class H, class B>
auto affectedComponentIndices(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs, const B& b) {
  vector<int> a; affectedComponentIndicesForEach(x, xt, y, yt, cs, b, [&](int u) { a.push_back(u); });
  return a;
}
template <class G, class H, class B>
auto affectedInComponentIndices(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs, const B& b) {
  vector<int> a; affectedInComponentIndicesForEach(x, xt, y, yt, cs, b, [&](int u) { a.push_back(u); });
  return a;
}
template <class G, class H, class B>
auto affectedOutComponentIndices(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs, const B& b) {
  vector<int> a; affectedOutComponentIndicesForEach(x, xt, y, yt, cs, b, [&](int u) { a.push_back(u); });
  return a;
}




// DYNAMIC-COMPONENTS
// ------------------
// Find affected, unaffected components (components, no. affected).

template <class G, class FA>
auto dynamicComponentIndicesBy(const G& y, const vector2d<int>& cs, FA fa) {
  vector<int> a; unordered_set<int> aff;
  fa([&](int i) { a.push_back(i); aff.insert(i); });
  for (int i=0, I=cs.size(); i<I; ++i)
    if (aff.count(i)==0) a.push_back(i);
  return make_pair(a, aff.size());
}

template <class G, class B>
auto dynamicComponentIndices(const G& x, const G& y, const vector2d<int>& cs, const B& b) {
  return dynamicComponentIndicesBy(y, cs, [&](auto fn) {
    affectedComponentIndicesForEach(x, y, cs, b, fn);
  });
}

template <class G, class H, class B>
auto dynamicComponentIndices(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs, const B& b) {
  return dynamicComponentIndicesBy(y, cs, [&](auto fn) {
    affectedComponentIndicesForEach(x, xt, y, yt, cs, b, fn);
  });
}

template <class G, class H, class B>
auto dynamicInComponentIndices(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs, const B& b) {
  return dynamicComponentIndicesBy(y, cs, [&](auto fn) {
    affectedInComponentIndicesForEach(x, xt, y, yt, cs, b, fn);
  });
}

template <class G, class H, class B>
auto dynamicOutComponentIndices(const G& x, const H& xt, const G& y, const H& yt, const vector2d<int>& cs, const B& b) {
  return dynamicComponentIndicesBy(y, cs, [&](auto fn) {
    affectedOutComponentIndicesForEach(x, xt, y, yt, cs, b, fn);
  });
}
