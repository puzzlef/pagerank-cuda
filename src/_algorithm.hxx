#pragma once
#include <cmath>
#include <vector>
#include <algorithm>

using std::vector;
using std::find;
using std::count;
using std::count_if;
using std::copy;
using std::abs;




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




// COPY
// ----

template <class T>
void copy(vector<T>& a, const vector<T>& x) {
  copy(x.begin(), x.end(), a.begin());
}


template <class T>
void copyOmp(vector<T>& a, const vector<T>& x) {
  int N = x.size();
  #pragma omp parallel for schedule(static,4096)
  for (int i=0; i<N; i++)
    a[i] = x[i];
}




// FILL
// ----

template <class T>
void fill(T *a, int N, const T& v) {
  for (int i=0; i<N; i++)
    a[i] = v;
}

template <class T>
void fill(vector<T>& a, const T& v) {
  fill(a.begin(), a.end(), v);
}


template <class T>
void fillOmp(T *a, int N, const T& v) {
  #pragma omp parallel for schedule(static,4096)
  for (int i=0; i<N; i++)
    a[i] = v;
}

template <class T>
void fillOmp(vector<T>& a, const T& v) {
  fillOmp(a.data(), (int) a.size(), v);
}




// FILL-AT
// -------

template <class T, class I>
void fillAt(T *a, const T& v, I&& is) {
  for (int i : is)
    a[i] = v;
}

template <class T, class I>
void fillAt(vector<T>& a, const T& v, I&& is) {
  fillAt(a.data(), v, is);
}




// SUM
// ---

template <class T>
auto sum(const T *x, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}

template <class T>
auto sum(const vector<T>& x) {
  return sum(x.data(), x.size());
}




// SUM-AT
// ------

template <class T, class I>
auto sumAt(const T *x, I&& is) {
  T a = T();
  for (int i : is)
    a += x[i];
  return a;
}

template <class T, class I>
auto sumAt(const vector<T>& x, I&& is) {
  return sumAt(x.data(), is);
}




// ADD-VALUE
// ---------

template <class T>
void addValue(T *a, int N, const T& v) {
  for (int i=0; i<N; i++)
    a[i] += v;
}

template <class T>
void addValue(vector<T>& a, const T& v) {
  addValue(a.data(), a.size(), v);
}




// ADD-VALUE-AT
// ------------

template <class T, class I>
void addValueAt(T *a, const T& v, I&& is) {
  for (int i : is)
    a[i] += v;
}

template <class T, class I>
void addValueAt(vector<T>& a, const T& v, I&& is) {
  addValueAt(a.data(), v, is);
}




// ABS-ERROR
// ---------

template <class T>
auto absError(const T *x, const T *y, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}

template <class T>
auto absError(const vector<T>& x, const vector<T>& y) {
  return absError(x.data(), y.data(), x.size());
}


template <class T>
auto absErrorOmp(const T *x, const T *y, int N) {
  T a = T();
  #pragma omp parallel for schedule(static,4096) reduction(+:a)
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}

template <class T>
auto absErrorOmp(const vector<T>& x, const vector<T>& y) {
  return absErrorOmp(x.data(), y.data(), x.size());
}




// MULTIPLY
// --------

template <class T>
void multiply(T *a, const T *x, const T *y, int N) {
  for (int i=0; i<N; i++)
    a[i] = x[i] * y[i];
}

template <class T>
void multiply(vector<T>& a, const vector<T>& x, const vector<T>& y) {
  multiply(a.data(), x.data(), y.data(), x.size());
}


template <class T>
void multiplyOmp(T *a, const T *x, const T *y, int N) {
  #pragma omp parallel for schedule(static,4096)
  for (int i=0; i<N; i++)
    a[i] = x[i] * y[i];
}

template <class T>
void multiplyOmp(vector<T>& a, const vector<T>& x, const vector<T>& y) {
  multiplyOmp(a.data(), x.data(), y.data(), x.size());
}
