#pragma once
#include <vector>
#include <unordered_map>
#include <iterator>
#include <algorithm>
#include <functional>

using std::vector;
using std::unordered_map;
using std::iterator_traits;
using std::hash;
using std::for_each;
using std::any_of;
using std::all_of;
using std::find;
using std::find_if;
using std::lower_bound;
using std::count;
using std::count_if;
using std::transform;
using std::set_difference;
using std::back_inserter;




// FOR-EACH
// --------
// Perform a sale.

template <class I, class F>
auto forEach(I ib, I ie, F fn) {
  return for_each(ib, ie, fn);
}

template <class J, class F>
auto forEach(const J& x, F fn) {
  return for_each(x.begin(), x.end(), fn);
}

template <class J, class F>
auto forEach(J& x, F fn) {
  return for_each(x.begin(), x.end(), fn);
}




// ANY-OF
// ------
// Is anything useful there?

template <class I, class F>
auto anyOf(I ib, I ie, F fn) {
  return any_of(ib, ie, fn);
}

template <class J, class F>
auto anyOf(const J& x, F fn) {
  return any_of(x.begin(), x.end(), fn);
}




// ALL-OF
// ------
// Is everything there?

template <class I, class F>
auto allOf(I ib, I ie, F fn) {
  return all_of(ib, ie, fn);
}

template <class J, class F>
auto allOf(const J& x, F fn) {
  return all_of(x.begin(), x.end(), fn);
}




// FIND
// ----
// Find a business or its address.

template <class J, class T>
auto find(const J& x, const T& v) {
  return find(x.begin(), x.end(), v);
}

template <class J, class T>
int findIndex(const J& x, const T& v) {
  return find(x.begin(), x.end(), v) - x.begin();
}

template <class J, class T>
int findEqIndex(const J& x, const T& v) {
  auto it = find(x.begin(), x.end(), v);
  return it==x.end()? -1 : it-x.begin();
}


template <class I, class F>
auto findIf(I ib, I ie, F fn) {
  return find_if(ib, ie, fn);
}

template <class J, class F>
auto findIf(const J& x, F fn) {
  return find_if(x.begin(), x.end(), fn);
}

template <class J, class F>
int findIfIndex(const J& x, F fn) {
  return find_if(x.begin(), x.end(), fn) - x.begin();
}

template <class J, class F>
int findIfEqIndex(const J& x, F fn) {
  auto it = find_if(x.begin(), x.end(), fn);
  return it==x.end()? -1 : it-x.begin();
}




// LOWER-BOUND
// -----------
// Find closest business, or its address.

template <class J, class T>
auto lowerBound(const J& x, const T& v) {
  return lower_bound(x.begin(), x.end(), v);
}

template <class J, class T, class F>
auto lowerBound(const J& x, const T& v, F fl) {
  return lower_bound(x.begin(), x.end(), v, fl);
}

template <class J, class T>
int lowerBoundIndex(const J& x, const T& v) {
  return lower_bound(x.begin(), x.end(), v) - x.begin();
}

template <class J, class T, class F>
int lowerBoundIndex(const J& x, const T& v, F fl) {
  return lower_bound(x.begin(), x.end(), v, fl) - x.begin();
}

template <class J, class T>
int lowerBoundEqIndex(const J& x, const T& v) {
  auto it = lower_bound(x.begin(), x.end(), v);
  return it==x.end() || *it!=v? -1 : it-x.begin();
}

template <class J, class T, class F>
int lowerBoundEqIndex(const J& x, const T& v, F fl) {
  auto it = lower_bound(x.begin(), x.end(), v, fl);
  return it==x.end() || *it!=v? -1 : it-x.begin();
}

template <class J, class T, class F, class G>
int lowerBoundEqIndex(const J& x, const T& v, F fl, G fe) {
  auto it = lower_bound(x.begin(), x.end(), v, fl);
  return it==x.end() || !fe(*it, v)? -1 : it-x.begin();
}




// COUNT
// -----
// Count businesses in a sector.

template <class J, class T>
int count(const J& x, const T& v) {
  return count(x.begin(), x.end(), v);
}


template <class I, class F>
int countIf(I ib, I ie, F fn) {
  return count_if(ib, ie, fn);
}

template <class J, class F>
int countIf(const J& x, F fn) {
  return count_if(x.begin(), x.end(), fn);
}




// COUNT-ALL
// ---------
// Count businesses in each sector.

template <class I>
auto countAll(I ib, I ie) {
  using K = typename iterator_traits<I>::value_type;
  unordered_map<K, int> a;
  for_each(ib, ie, [&](const auto& v) { a[v]++; });
  return a;
}

template <class J>
auto countAll(const J& x) {
  return countAll(x.begin(), x.end());
}




// INDICES
// -------
// Keep the address of each business (yellow pages).

template <class I>
auto indices(I ib, I ie) {
  using K = typename iterator_traits<I>::value_type;
  unordered_map<K, int> a; int i = -1;
  for (I it=ib; it!=ie; ++it)
    a[*it] = ++i;
  return a;
}

template <class J>
auto indices(const J& x) {
  return indices(x.begin(), x.end());
}




// TRANSFORM
// ---------
// Switch around your portfolio.

template <class J, class K, class F>
void transform(const J& x, K& a, F fn) {
  transform(x.begin(), x.end(), a.begin(), fn);
}

template <class J, class F>
void transform(J& a, F fn) {
  transform(a, a, fn);
}




// SORT
// ----
// Arrange your portfolio by ROCE.

template <class J>
void sort(J& x, int i, int n) {
  sort(x.begin()+i, x.end()+i+n);
}
template <class J>
void sort(J& x, int i) {
  sort(x.begin()+i, x.end());
}
template <class J>
void sort(J& x) {
  sort(x.begin(), x.end());
}




// SET-DIFFERENCE
// --------------

template <class L, class J, class K>
void setDifference(L& a, const J& x, const K& y) {
  set_difference(x.begin(), x.end(), y.begin(), y.end(), a.begin());
}

template <class T, class J, class K>
void setDifference(vector<T>& a, const J& x, const K& y) {
  set_difference(x.begin(), x.end(), y.begin(), y.end(), back_inserter(a));
}

template <class J, class K>
auto setDifference(const J& x, const K& y) {
  using I = decltype(x.begin());
  using T = typename iterator_traits<I>::value_type;
  vector<T> a; setDifference(a, x, y);
  return a;
}




// WRITE
// -----

template <class T, class I>
void write(vector<T>& a, I ib, I ie) {
  a.clear();
  a.insert(a.begin(), ib, ie);
}

template <class T, class J>
void write(vector<T>& a, const J& vs) {
  write(a, vs.begin(), vs.end());
}




// TO-*
// ----

template <class T, class I>
auto toVector(I ib, I ie) {
  vector<T> a; write(a, ib, ie);
  return a;
}

template <class T, class J>
void toVector(const J& x) {
  return toVector<T>(x.begin(), x.end());
}




// HASH-VALUE
// ----------

template <class T, class I>
size_t hashValue(vector<T>& vs, I ib, I ie) {
  size_t a = 0;
  write(vs, ib, ie); sort(vs);
  for (const T& v : vs)
    a ^= hash<T>{}(v) + 0x9e3779b9 + (a<<6) + (a>>2); // from boost::hash_combine
  return a;
}

template <class I>
size_t hashValue(I ib, I ie) {
  using T = typename I::value_type;
  vector<T> vs;
  return hashValue(vs, ib, ie);
}

template <class T, class J>
size_t hashValue(vector<T>& vs, const J& x) {
  return hashValue(vs, x.begin(), x.end());
}

template <class J>
size_t hashValue(const J& x) {
  return hashValue(x.begin(), x.end());
}
