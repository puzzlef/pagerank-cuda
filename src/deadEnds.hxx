#pragma once
#include <vector>
#include <unordered_set>

using std::vector;
using std::unordered_set;




// IS-DEAD-END
// -----------

template <class G>
bool isDeadEnd(const G& x, int u) {
  return x.degree(u) == 0;
}




// DEAD-ENDS
// ---------

template <class G, class F>
void deadEndsForEach(const G& x, F fn) {
  for (int u : x.vertices())
    if (isDeadEnd(x, u)) fn(u);
}


template <class G>
auto deadEnds(const G& x) {
  vector<int> a; deadEndsForEach(x, [&](int u) { a.push_back(u); });
  return a;
}

template <class G>
int deadEndCount(const G& x) {
  int a = 0; deadEndsForEach(x, [&](int u) { ++a; });
  return a;
}




// RECURSIVE DEAD-ENDS
// -------------------
// Vertices that can become dead ends if existing ones are removed.

template <class G, class F>
auto recursiveDeadEndsForEach(const G& x, F fn) {
  unordered_set<int> a; size_t N = 0;
  deadEndsForEach(x, [&](int u) { a.insert(u); fn(u); });
  auto fdead = [&](int u) { return a.count(u)>0; };
  while (a.size() > N) {
    N = a.size();
    for (int u : x.vertices())
      if (allOf(x.edges(u), fdead) && a.count(u)==0) { a.insert(u); fn(u); }
  }
  return a;
}


template <class G>
auto recursiveDeadEnds(const G& x) {
  vector<int> a; recursiveDeadEndsForEach(x, [&](int u) { a.push_back(u); });
  return a;
}

template <class G>
int recursiveDeadEndCount(const G& x) {
  int a = 0; recursiveDeadEndsForEach(x, [&](int u) { ++a; });
  return a;
}
