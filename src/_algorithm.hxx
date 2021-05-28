#pragma once
#include <cmath>
#include <vector>
#include <unordered_map>
#include <iterator>
#include <algorithm>

using std::vector;
using std::unordered_map;
using std::iterator_traits;
using std::back_inserter;
using std::find;
using std::count;
using std::count_if;
using std::set_difference;
using std::copy;
using std::abs;
using std::swap;




// FIND
// ----

template <class J, class T>
auto find(const J& x, const T& v) {
  return find(x.begin(), x.end(), v);
}

template <class J, class T>
int findIndex(const J& x, const T& v) {
  auto i = find(x.begin(), x.end(), v);
  return i==x.end()? -1 : i-x.begin();
}




// COUNT-*
// -------

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




// INDICES
// -------

template <class I>
auto indices(I ib, I ie) {
  using K = typename iterator_traits<I>::value_type;
  unordered_map<K, int> a; int i = 0;
  for (I it=ib; it!=ie; ++it)
    a[*it] = i++;
  return a;
}

template <class J>
auto indices(J&& x) {
  return indices(x.begin(), x.end());
}




// SET-DIFFERENCE
// --------------

template <class L, class J, class K>
void setDifference(L&& a, J&& x, K&& y) {
  set_difference(x.begin(), x.end(), y.begin(), y.end(), a.begin());
}

template <class T, class J, class K>
void setDifference(vector<T>& a, J&& x, K&& y) {
  set_difference(x.begin(), x.end(), y.begin(), y.end(), back_inserter(a));
}

template <class J, class K>
auto setDifference(J&& x, K&& y) {
  using I = decltype(x.begin());
  using T = typename iterator_traits<I>::value_type;
  vector<T> a; setDifference(a, x, y);
  return a;
}




// REORDER
// -------
// Ref: https://stackoverflow.com/a/22183350/1413259

template <class T>
void reorder(vector<T>& x, vector<int> is) {
  for(int i=0, N=x.size(); i<N; i++) {
    while(is[i] != is[is[i]]) {
      swap(x[is[i]], x[is[is[i]]]);
      swap(  is[i],    is[is[i]]);
    }
  }
}




// ERASE
// -----

template <class T>
void eraseIndex(vector<T>& x, int i) {
  x.erase(x.begin()+i);
}

template <class T>
void eraseIndex(vector<T>& x, int i, int I) {
  x.erase(x.begin()+i, x.begin()+I);
}




// JOIN
// ----

template <class T, class F>
auto joinIf(const vector<vector<T>>& xs, F fn) {
  vector<vector<T>> a;
  for (const auto& x : xs) {
    auto& b = a.back();
    if (a.empty() || !fn(b, x)) a.push_back(x);
    else b.insert(b.end(), x.begin(), x.end());
  }
  return a;
}

template <class T>
auto joinUntilSize(const vector<vector<T>>& xs, int N) {
  return joinIf(xs, [&](const auto& b, const auto& x) { return b.size()<N; });
}

template <class T>
auto join(const vector<vector<T>>& xs) {
  vector<T> a;
  for (const auto& x : xs)
    a.insert(a.end(), x.begin(), x.end());
  return a;
}




// COPY
// ----

template <class T, class U>
void copy(T *a, U *x, int N) {
  for (int i=0; i<N; i++)
    a[i] = x[i];
}

template <class T, class U>
void copy(vector<T>& a, const vector<U>& x) {
  copy(a.data(), x.data(), int(x.size()));
}

template <class T, class U>
void copy(vector<T>& a, const vector<T>& x, int i, int N) {
  copy(a.data()+i, x.data()+i, N);
}


template <class T, class U>
void copyOmp(T *a, U *x, int N) {
  #pragma omp parallel for schedule(static,4096)
  for (int i=0; i<N; i++)
    a[i] = x[i];
}

template <class T, class U>
void copyOmp(vector<T>& a, const vector<U>& x) {
  copyOmp(a.data(), x.data(), int(x.size()));
}

template <class T, class U>
void copyOmp(vector<T>& a, const vector<U>& x, int i, int N) {
  copyOmp(a.data()+i, x.data()+i, N);
}




// GATHER
// ------

template <class T, class U, class J>
void gather(T *a, const U *x, J&& is) {
  int j = 0;
  for (int i : is)
    a[j++] = x[i];
}

template <class T, class U, class J>
void gather(vector<T>& a, const vector<U>& x, J&& is) {
  gather(a.data(), x.data(), is);
}




// SCATTER
// -------

template <class T, class U, class J>
void scatter(T *a, const U *x, J&& is) {
  int j = 0;
  for (int i : is)
    a[i] = x[j++];
}

template <class T, class U, class J>
void scatter(vector<T>& a, const vector<U>& x, J&& is) {
  scatter(a.data(), x.data(), is);
}




// FILL
// ----

template <class T, class U>
void fill(T *a, int N, const U& v) {
  for (int i=0; i<N; i++)
    a[i] = v;
}

template <class T, class U>
void fill(vector<T>& a, const U& v) {
  fill(a.begin(), a.end(), v);
}

template <class T, class U>
void fill(vector<T>& a, int i, int N, const U& v) {
  fill(a.data()+i, N, v);
}


template <class T, class U>
void fillOmp(T *a, int N, const U& v) {
  #pragma omp parallel for schedule(static,4096)
  for (int i=0; i<N; i++)
    a[i] = v;
}

template <class T, class U>
void fillOmp(vector<T>& a, const U& v) {
  fillOmp(a.data(), int(a.size()), v);
}

template <class T, class U>
void fillOmp(vector<T>& a, int i, int N, const U& v) {
  fillOmp(a.data()+i, N, v);
}




// FILL-AT
// -------

template <class T, class U, class J>
void fillAt(T *a, const U& v, J&& is) {
  for (int i : is)
    a[i] = v;
}

template <class T, class U, class J>
void fillAt(vector<T>& a, const U& v, J&& is) {
  fillAt(a.data(), v, is);
}

template <class T, class U, class J>
void fillAt(vector<T>& a, int i, const U& v, J&& is) {
  fillAt(a.data()+i, v, is);
}




// SUM
// ---

template <class T, class U=T>
U sum(const T *x, int N, U a=U()) {
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}

template <class T, class U=T>
U sum(const vector<T>& x, U a=U()) {
  return sum(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U sum(const vector<T>& x, int i, int N, U a=U()) {
  return sum(x.data()+i, N, a);
}


template <class T, class U=T>
U sumOmp(const T *x, int N, U a=U()) {
  #pragma omp parallel for schedule(static,4096) reduction(+:a)
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}

template <class T, class U=T>
U sumOmp(const vector<T>& x, U a=U()) {
  return sumOmp(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U sumOmp(const vector<T>& x, int i, int N, U a=U()) {
  return sumOmp(x.data()+i, N, a);
}




// SUM-AT
// ------

template <class T, class J, class U=T>
U sumAt(const T *x, J&& is, U a=U()) {
  for (int i : is)
    a += x[i];
  return a;
}

template <class T, class J, class U=T>
U sumAt(const vector<T>& x, J&& is, U a=U()) {
  return sumAt(x.data(), is, a);
}

template <class T, class J, class U=T>
U sumAt(const vector<T>& x, int i, J&& is, U a=U()) {
  return sumAt(x.data()+i, is, a);
}




// ADD-VALUE
// ---------

template <class T, class U>
void addValue(T *a, int N, const U& v) {
  for (int i=0; i<N; i++)
    a[i] += v;
}

template <class T, class U>
void addValue(vector<T>& a, const U& v) {
  addValue(a.data(), int(a.size()), v);
}

template <class T, class U>
void addValue(vector<T>& a, int i, int N, const U& v) {
  addValue(a.data()+i, N, v);
}


template <class T, class U>
void addValueOmp(T *a, int N, const U& v) {
  #pragma omp parallel for schedule(static,4096)
  for (int i=0; i<N; i++)
    a[i] += v;
}

template <class T, class U>
void addValueOmp(vector<T>& a, const U& v) {
  addValueOmp(a.data(), int(a.size()), v);
}

template <class T, class U>
void addValueOmp(vector<T>& a, int i, int N, const U& v) {
  addValueOmp(a.data()+i, N, v);
}




// ADD-VALUE-AT
// ------------

template <class T, class U, class J>
void addValueAt(T *a, const U& v, J&& is) {
  for (int i : is)
    a[i] += v;
}

template <class T, class U, class J>
void addValueAt(vector<T>& a, const U& v, J&& is) {
  addValueAt(a.data(), v, is);
}

template <class T, class U, class J>
void addValueAt(vector<T>& a, int i, const U& v, J&& is) {
  addValueAt(a.data()+i, v, is);
}




// ABS-ERROR
// ---------

template <class T, class U, class V=T>
V absError(const T *x, const U *y, int N, V a=V()) {
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}

template <class T, class U, class V=T>
V absError(const vector<T>& x, const vector<U>& y, V a=V()) {
  return absError(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V absError(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return absError(x.data()+i, y.data()+i, N, a);
}


template <class T, class U, class V=T>
V absErrorOmp(const T *x, const U *y, int N, V a=V()) {
  #pragma omp parallel for schedule(static,4096) reduction(+:a)
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}

template <class T, class U, class V=T>
V absErrorOmp(const vector<T>& x, const vector<U>& y, V a=V()) {
  return absErrorOmp(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V absErrorOmp(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return absErrorOmp(x.data()+i, y.data()+i, N, a);
}




// MULTIPLY
// --------

template <class T, class U, class V>
void multiply(T *a, const U *x, const V *y, int N) {
  for (int i=0; i<N; i++)
    a[i] = x[i] * y[i];
}

template <class T, class U, class V>
void multiply(vector<T>& a, const vector<U>& x, const vector<V>& y) {
  multiply(a.data(), x.data(), y.data(), int(x.size()));
}

template <class T, class U, class V>
void multiply(vector<T>& a, const vector<U>& x, const vector<V>& y, int i, int N) {
  multiply(a.data()+i, x.data()+i, y.data()+i, N);
}


template <class T, class U, class V>
void multiplyOmp(T *a, const U *x, const V *y, int N) {
  #pragma omp parallel for schedule(static,4096)
  for (int i=0; i<N; i++)
    a[i] = x[i] * y[i];
}

template <class T, class U, class V>
void multiplyOmp(vector<T>& a, const vector<U>& x, const vector<V>& y) {
  multiplyOmp(a.data(), x.data(), y.data(), int(x.size()));
}

template <class T, class U, class V>
void multiplyOmp(vector<T>& a, const vector<U>& x, const vector<V>& y, int i, int N) {
  multiplyOmp(a.data()+i, x.data()+i, y.data()+i, N);
}
