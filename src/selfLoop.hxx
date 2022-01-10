#pragma once
#include "copy.hxx"




// HAS SELF-LOOP
// -------------

template <class G>
bool hasSelfLoop(const G& x, int u) {
  return x.hasEdge(u, u);
}




// SELF-LOOPS
// ----------

template <class G, class F>
void selfLoopForEach(const G& x, F fn) {
  for (int u : x.vertices())
    if (x.hasEdge(u, u)) fn(u);
}


template <class G>
auto selfLoops(const G& x) {
  vector<int> a; selfLoopForEach(x, [&](int u) { a.push_back(u); });
  return a;
}

template <class G>
int selfLoopCount(const G& x) {
  int a = 0; selfLoopForEach(x, [&](int u) { ++a; });
  return a;
}




// SELF-LOOPS
// ----------

template <class G, class F>
void selfLoopTo(G& a, F fn) {
  for (int u : a.vertices())
    if (fn(u)) a.addEdge(u, u);
}

template <class G, class F>
auto selfLoop(const G& x, F fn) {
  auto a = copy(x); selfLoopTo(a, fn);
  return a;
}
