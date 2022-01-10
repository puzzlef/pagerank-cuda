#pragma once
#include <utility>
#include <vector>
#include "vertices.hxx"

using std::pair;
using std::vector;




// DFS
// ---
// Traverses nodes in depth-first manner, listing on entry.

template <class G, class F>
void dfsDoLoop(vector<bool>& vis, const G& x, int u, F fn) {
  if (vis[u]) return;  // dont visit if done already!
  vis[u] = true; fn(u);
  for (int v : x.edges(u))
    if (!vis[v]) dfsDoLoop(vis, x, v, fn);
}
template <class G, class F>
void dfsDo(const G& x, int u, F fn) {
  auto vis = createContainer(x, bool());
  dfsDoLoop(vis, x, u, fn);
}

template <class G>
void dfsMarkLoop(vector<bool>& vis, const G& x, int u) {
  auto fn = [](int u) {};
  dfsDoLoop(vis, x, u, fn);
}
template <class G>
void dfsMark(const G& x, int u) {
  auto vis = createContainer(x, bool());
  dfsMarkLoop(vis, x, u);
  return vis;
}

template <class G>
void dfsLoop(vector<int>& a, vector<bool>& vis, const G& x, int u) {
  dfsDoLoop(vis, x, u, [&](int u) { a.push_back(u); });
}
template <class G>
auto dfs(const G& x, int u) {
  vector<int> a;
  auto vis = createContainer(x, bool());
  dfsLoop(a, vis, x, u);
  return a;
}




// DFS-END
// -------
// Traverses nodes in depth-first manner, listing on exit.

template <class G, class F>
void dfsEndDoLoop(vector<bool>& vis, const G& x, int u, F fn) {
  if (vis[u]) return;  // dont visit if done already!
  vis[u] = true;
  for (int v : x.edges(u))
    if (!vis[v]) dfsEndDoLoop(vis, x, v, fn);
  fn(u);
}
template <class G, class F>
void dfsEndDo(const G& x, int u, F fn) {
  auto vis = createContainer(x, bool());
  dfsEndDoLoop(vis, x, u, fn);
}

template <class G>
void dfsEndMarkLoop(vector<bool>& vis, const G& x, int u) {
  auto fn = [](int u) {};
  dfsEndDoLoop(vis, x, u, fn);
}
template <class G>
void dfsEndMark(const G& x, int u) {
  auto vis = createContainer(x, bool());
  dfsEndMarkLoop(vis, x, u);
  return vis;
}

template <class G>
void dfsEndLoop(vector<int>& a, vector<bool>& vis, const G& x, int u) {
  dfsEndDoLoop(vis, x, u, [&](int v) { a.push_back(v); });
}
template <class G>
auto dfsEnd(const G& x, int u) {
  vector<int> a;
  auto vis = createContainer(x, bool());
  dfsEndLoop(a, vis, x, u);
  return a;
}




// DFS DEPTH
// ---------
// Traverses nodes in depth-first manner, listing on entry.

template <class G, class F>
void dfsDepthDoLoop(vector<bool>& vis, const G& x, int u, int d, F fn) {
  if (vis[u]) return;  // dont visit if done already!
  vis[u] = true; fn(u, d++);
  for (int v : x.edges(u))
    if (!vis[v]) dfsDepthDoLoop(vis, x, v, d, fn);
}
template <class G, class F>
void dfsDepthDo(const G& x, int u, int d, F fn) {
  auto vis = createContainer(x, bool());
  dfsDepthDoLoop(vis, x, u, d, fn);
}

template <class G>
void dfsDepthMarkLoop(vector<bool>& vis, const G& x, int u, int d) {
  auto fn = [](int u) {};
  dfsDepthDoLoop(vis, x, u, d, fn);
}
template <class G>
void dfsDepthMark(const G& x, int u, int d) {
  auto vis = createContainer(x, bool());
  dfsDepthMarkLoop(vis, x, u, d);
  return vis;
}

template <class G>
void dfsDepthLoop(vector<pair<int,int>>& a, vector<bool>& vis, const G& x, int u, int d) {
  dfsDepthDoLoop(vis, x, u, d, [&](int v, int d) { a.push_back({v, d}); });
}
template <class G>
auto dfsDepth(const G& x, int u, int d) {
  vector<pair<int,int>> a;
  auto vis = createContainer(x, bool());
  dfsDepthLoop(a, vis, x, u, d);
  return a;
}
