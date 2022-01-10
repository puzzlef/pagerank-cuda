#pragma once
#include <cmath>
#include <array>
#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include "_openmp.hxx"

using std::array;
using std::vector;
using std::map;
using std::copy;
using std::swap;
using std::move;
using std::abs;
using std::max;
using std::sqrt;




// 2D/3D
// -----

template <class T>
using vector2d = vector<vector<T>>;

template <class T>
using vector3d = vector<vector<vector<T>>>;




// SIZE
// ----

template <class T>
size_t size(const vector<T>& x) {
  return x.size();
}

template <class T>
size_t size2d(const vector2d<T>& x) {
  size_t a = 0;
  for (const auto& v : x)
    a += size(v);
  return a;
}

template <class T>
size_t size3d(const vector3d<T>& x) {
  size_t a = 0;
  for (const auto& v : x)
    a += size2d(v);
  return a;
}




// REORDER
// -------
// Ref: https://stackoverflow.com/a/22183350/1413259

template <class T>
void reorderDirty(vector<T>& x, vector<int>& is) {
  for(int i=0, N=x.size(); i<N; i++) {
    while(is[i] != is[is[i]]) {
      swap(x[is[i]], x[is[is[i]]]);
      swap(  is[i],    is[is[i]]);
    }
  }
}

template <class T>
void reorder(vector<T>& x, vector<int> is) {
  reorderDirty(x, is);
}




// ERASE
// -----

template <class T>
void eraseIndex(vector<T>& a, int i) {
  a.erase(a.begin()+i);
}

template <class T>
void eraseIndex(vector<T>& a, int i, int I) {
  a.erase(a.begin()+i, a.begin()+I);
}




// INSERT
// ------

template <class T>
void insertIndex(vector<T>& a, int i, const T& v) {
  a.insert(a.begin()+i, v);
}

template <class T>
void insertIndex(vector<T>& a, int i, int n, const T& v) {
  a.insert(a.begin()+i, n, v);
}




// APPEND
// ------

template <class T, class I>
void append(vector<T>& a, I ib, I ie) {
  for (auto i=ib; i!=ie; ++i)
    a.push_back(*i);
}

template <class T, class J>
void append(vector<T>& a, const J& vs) {
  append(a, vs.begin(), vs.end());
}




// GROUP-IF
// --------

// UNUSED!
template <class T, class J, class F>
void groupIf(vector2d<T>& a, const J& x, F fn) {
  for (const auto& v : x) {
    auto& b = a.back();
    if (a.empty() || !fn(b, v)) a.push_back({v});
    else b.push_back(v);
  }
}
template <class T, class J, class F>
auto groupIf(const J& x, F fn) {
  vector2d<T> a; groupIf(a, x, fn);
  return a;
}




// GROUP-BY
// --------

template <class T, class J, class F>
void groupBy(vector2d<T>& a, const J& x, F fn) {
  using K = decltype(fn(T()));
  map<K, vector<int>> gs;
  for (const auto& v : x)
    gs[fn(v)].push_back(v);
  for (const auto& [k, v] : gs)
    a.push_back(move(v));
}

template <class T, class J, class F>
auto groupBy(const J& x, F fn) {
  vector2d<T> a; groupBy(a, x, fn);
  return a;
}




// JOIN
// ----

template <class T, class J, class F>
void joinIf(vector2d<T>& a, const J& xs, F fn) {
  for (const auto& x : xs) {
    auto& b = a.back();
    if (a.empty() || !fn(b, x)) a.push_back(x);
    else b.insert(b.end(), x.begin(), x.end());
  }
}

template <class T, class J, class F>
auto joinIf(const J& xs, F fn) {
  vector2d<T> a; joinIf(a, xs, fn);
  return a;
}


template <class T, class J>
void joinUntilSize(vector2d<T>& a, const J& xs, int S) {
  auto fn = [&](const auto& b, const auto& x) { return b.size()<S; };
  joinIf(a, xs, fn);
}

template <class T, class J>
auto joinUntilSize(const J& xs, int S) {
  vector2d<T> a; joinUntilSize(a, xs, S);
  return a;
}


template <class T, class J>
void join(vector<T>& a, const J& xs) {
  for (const auto& x : xs)
    a.insert(a.end(), x.begin(), x.end());
}

template <class T, class J>
auto join(const J& xs) {
  vector<T> a; join(a, xs);
  return a;
}




// JOIN-AT
// -------

template <class T, class J, class F>
void joinAtIf(vector2d<T>& a, const vector2d<T>& xs, const J& is, F fn) {
  for (int i : is) {
    auto& b = a.back();
    if (a.empty() || !fn(b, xs[i])) a.push_back(xs[i]);
    else b.insert(b.end(), xs[i].begin(), xs[i].end());
  }
}

template <class T, class J, class F>
auto joinAtIf(const vector2d<T>& xs, const J& is, F fn) {
  vector2d<T> a; joinAtIf(a, xs, is, fn);
  return a;
}



template <class T, class J>
void joinAtUntilSize(vector2d<T>& a, const vector2d<T>& xs, const J& is, int N) {
  joinAtIf(a, xs, is, [&](const auto& b, const auto& x) { return b.size()<N; });
}

template <class T, class J>
auto joinAtUntilSize(const vector2d<T>& xs, const J& is, int N) {
  vector2d<T> a; joinAtUntilSize(a, xs, is, N);
  return a;
}


template <class T, class J>
void joinAt(vector<T>& a, const vector2d<T>& xs, const J& is) {
  for (int i : is)
    a.insert(a.end(), xs[i].begin(), xs[i].end());
}

template <class T, class J>
auto joinAt(const vector2d<T>& xs, const J& is) {
  vector<T> a; joinAt(a, xs, is);
  return a;
}


template <class T, class J>
void joinAt2d(vector2d<T>& a, const vector2d<T>& xs, const J& ig) {
  for (const auto& is : ig)
    a.push_back(joinAt(xs, is));
}

template <class T, class J>
auto joinAt2d(const vector2d<T>& xs, const J& ig) {
  vector2d<T> a; joinAt2d(a, xs, ig);
  return a;
}




// COPY-AT
// -------

template <class T, class J>
void copyAt(vector<T>& a, const vector<T>& xs, const J& is) {
  for (const auto& i : is)
    a.push_back(xs[i]);
}

template <class T, class J>
auto copyAt(const vector<T>& xs, const J& is) {
  vector<T> a; copyAt(a, xs, is);
  return a;
}




// GATHER
// ------

template <class T, class U, class J>
void gather(T *a, const U *x, const J& is) {
  int j = 0;
  for (int i : is)
    a[j++] = x[i];
}

template <class T, class U, class J>
void gather(vector<T>& a, const vector<U>& x, const J& is) {
  gather(a.data(), x.data(), is);
}




// SCATTER
// -------

template <class T, class U, class J>
void scatter(T *a, const U *x, const J& is) {
  int j = 0;
  for (int i : is)
    a[i] = x[j++];
}

template <class T, class U, class J>
void scatter(vector<T>& a, const vector<U>& x, const J& is) {
  scatter(a.data(), x.data(), is);
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
  if (N<SIZE_MIN_OMPM) { copy(a, x, N); return; }
  #pragma omp parallel for num_threads(32) schedule(auto)
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
  if (N<SIZE_MIN_OMPM) { fill(a, N, v); return; }
  #pragma omp parallel for num_threads(32) schedule(auto)
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
void fillAt(T *a, const U& v, const J& is) {
  for (int i : is)
    a[i] = v;
}

template <class T, class U, class J>
void fillAt(vector<T>& a, const U& v, const J& is) {
  fillAt(a.data(), v, is);
}

template <class T, class U, class J>
void fillAt(vector<T>& a, int i, const U& v, const J& is) {
  fillAt(a.data()+i, v, is);
}




// SUM
// ---

template <class T>
T sum(const T& v) {
  return v;
}


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
  if (N<SIZE_MIN_OMPR) return sum(x, N, a);
  #pragma omp parallel for num_threads(32) schedule(auto) reduction(+:a)
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




// SUM-ABS
// -------

template <class T, class U=T>
U sumAbs(const T *x, int N, U a=U()) {
  for (int i=0; i<N; i++)
    a += abs(x[i]);
  return a;
}

template <class T, class U=T>
U sumAbs(const vector<T>& x, U a=U()) {
  return sumAbs(x.data(), int(x.size()), a);
}

// NOTE: ADDITIONAL HELPER
template <class T, size_t N, class U=T>
U sumAbs(const array<T, N>& x, U a=U()) {
  return sumAbs(x.data(), int(N), a);
}

template <class T, class U=T>
U sumAbs(const vector<T>& x, int i, int N, U a=U()) {
  return sumAbs(x.data()+i, N, a);
}


template <class T, class U=T>
U sumAbsOmp(const T *x, int N, U a=U()) {
  if (N<SIZE_MIN_OMPR) return sumAbs(x, N, a);
  #pragma omp parallel for num_threads(32) schedule(auto) reduction(+:a)
  for (int i=0; i<N; i++)
    a += abs(x[i]);
  return a;
}

template <class T, class U=T>
U sumAbsOmp(const vector<T>& x, U a=U()) {
  return sumAbsOmp(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U sumAbsOmp(const vector<T>& x, int i, int N, U a=U()) {
  return sumAbsOmp(x.data()+i, N, a);
}




// SUM-SQR
// -------

template <class T, class U=T>
U sumSqr(const T *x, int N, U a=U()) {
  for (int i=0; i<N; i++)
    a += x[i]*x[i];
  return a;
}

template <class T, class U=T>
U sumSqr(const vector<T>& x, U a=U()) {
  return sumSqr(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U sumSqr(const vector<T>& x, int i, int N, U a=U()) {
  return sumSqr(x.data()+i, N, a);
}


template <class T, class U=T>
U sumSqrOmp(const T *x, int N, U a=U()) {
  if (N<SIZE_MIN_OMPR) return sumSqr(x, N, a);
  #pragma omp parallel for num_threads(32) schedule(auto) reduction(+:a)
  for (int i=0; i<N; i++)
    a += x[i]*x[i];
  return a;
}

template <class T, class U=T>
U sumSqrOmp(const vector<T>& x, U a=U()) {
  return sumSqrOmp(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U sumSqrOmp(const vector<T>& x, int i, int N, U a=U()) {
  return sumSqrOmp(x.data()+i, N, a);
}




// SUM-AT
// ------

template <class T, class J, class U=T>
U sumAt(const T *x, const J& is, U a=U()) {
  for (int i : is)
    a += x[i];
  return a;
}

template <class T, class J, class U=T>
U sumAt(const vector<T>& x, const J& is, U a=U()) {
  return sumAt(x.data(), is, a);
}

template <class T, class J, class U=T>
U sumAt(const vector<T>& x, int i, const J& is, U a=U()) {
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
  if (N<SIZE_MIN_OMPM) { addValue(a, N, v); return; }
  #pragma omp parallel for num_threads(32) schedule(auto)
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
void addValueAt(T *a, const U& v, const J& is) {
  for (int i : is)
    a[i] += v;
}

template <class T, class U, class J>
void addValueAt(vector<T>& a, const U& v, const J& is) {
  addValueAt(a.data(), v, is);
}

template <class T, class U, class J>
void addValueAt(vector<T>& a, int i, const U& v, const J& is) {
  addValueAt(a.data()+i, v, is);
}




// MAX
// ---

template <class T, class U=T>
U max(const T *x, int N, U a=U()) {
  for (int i=0; i<N; i++)
    a = max(a, x[i]);
  return a;
}

template <class T, class U=T>
U max(const vector<T>& x, U a=U()) {
  return max(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U max(const vector<T>& x, int i, int N, U a=U()) {
  return max(x.data()+i, N, a);
}


template <class T, class U=T>
U maxOmp(const T *x, int N, U a=U()) {
  if (N<SIZE_MIN_OMPR) return max(x, N, a);
  #pragma omp parallel for num_threads(32) schedule(auto) reduction(+:a)
  for (int i=0; i<N; i++)
    a = max(a, x[i]);
  return a;
}

template <class T, class U=T>
U maxOmp(const vector<T>& x, U a=U()) {
  return maxOmp(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U maxOmp(const vector<T>& x, int i, int N, U a=U()) {
  return maxOmp(x.data()+i, N, a);
}




// MAX-ABS
// -------

template <class T, class U=T>
U maxAbs(const T *x, int N, U a=U()) {
  for (int i=0; i<N; i++)
    a = max(a, abs(x[i]));
  return a;
}

template <class T, class U=T>
U maxAbs(const vector<T>& x, U a=U()) {
  return maxAbs(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U maxAbs(const vector<T>& x, int i, int N, U a=U()) {
  return maxAbs(x.data()+i, N, a);
}


template <class T, class U=T>
U maxAbsOmp(const T *x, int N, U a=U()) {
  if (N<SIZE_MIN_OMPR) return maxAbs(x, N, a);
  #pragma omp parallel for num_threads(32) schedule(auto) reduction(+:a)
  for (int i=0; i<N; i++)
    a = max(a, abs(x[i]));
  return a;
}

template <class T, class U=T>
U maxAbsOmp(const vector<T>& x, U a=U()) {
  return maxAbsOmp(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U maxAbsOmp(const vector<T>& x, int i, int N, U a=U()) {
  return maxAbsOmp(x.data()+i, N, a);
}




// MAX-AT
// ------

template <class T, class J, class U=T>
U maxAt(const T *x, const J& is, U a=U()) {
  for (int i : is)
    a = max(a, x[i]);
  return a;
}

template <class T, class J, class U=T>
U maxAt(const vector<T>& x, const J& is, U a=U()) {
  return maxAt(x.data(), is, a);
}

template <class T, class J, class U=T>
U maxAt(const vector<T>& x, int i, const J& is, U a=U()) {
  return maxAt(x.data()+i, is, a);
}




// MAX-VALUE
// ---------

template <class T, class U>
void maxValue(T *a, int N, const U& v) {
  for (int i=0; i<N; i++)
    a[i] = max(a[i], v);
}

template <class T, class U>
void maxValue(vector<T>& a, const U& v) {
  maxValue(a.data(), int(a.size()), v);
}

template <class T, class U>
void maxValue(vector<T>& a, int i, int N, const U& v) {
  maxValue(a.data()+i, N, v);
}


template <class T, class U>
void maxValueOmp(T *a, int N, const U& v) {
  if (N<SIZE_MIN_OMPM) { maxValue(a, N, v); return; }
  #pragma omp parallel for num_threads(32) schedule(auto)
  for (int i=0; i<N; i++)
    a[i] = max(a[i], v);
}

template <class T, class U>
void maxValueOmp(vector<T>& a, const U& v) {
  maxValueOmp(a.data(), int(a.size()), v);
}

template <class T, class U>
void maxValueOmp(vector<T>& a, int i, int N, const U& v) {
  maxValueOmp(a.data()+i, N, v);
}




// MAX-VALUE-AT
// ------------

template <class T, class U, class J>
void maxValueAt(T *a, const U& v, const J& is) {
  for (int i : is)
    a[i] = max(a[i], v);
}

template <class T, class U, class J>
void maxValueAt(vector<T>& a, const U& v, const J& is) {
  maxValueAt(a.data(), v, is);
}

template <class T, class U, class J>
void maxValueAt(vector<T>& a, int i, const U& v, const J& is) {
  maxValueAt(a.data()+i, v, is);
}




// MIN
// ---

template <class T, class U=T>
U min(const T *x, int N, U a=U()) {
  for (int i=0; i<N; i++)
    a = min(a, x[i]);
  return a;
}

template <class T, class U=T>
U min(const vector<T>& x, U a=U()) {
  return min(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U min(const vector<T>& x, int i, int N, U a=U()) {
  return min(x.data()+i, N, a);
}


template <class T, class U=T>
U minOmp(const T *x, int N, U a=U()) {
  if (N<SIZE_MIN_OMPR) return min(x, N, a);
  #pragma omp parallel for num_threads(32) schedule(auto) reduction(+:a)
  for (int i=0; i<N; i++)
    a = min(a, x[i]);
  return a;
}

template <class T, class U=T>
U minOmp(const vector<T>& x, U a=U()) {
  return minOmp(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U minOmp(const vector<T>& x, int i, int N, U a=U()) {
  return minOmp(x.data()+i, N, a);
}




// MIN-ABS
// -------

template <class T, class U=T>
U minAbs(const T *x, int N, U a=U()) {
  for (int i=0; i<N; i++)
    a = min(a, abs(x[i]));
  return a;
}

template <class T, class U=T>
U minAbs(const vector<T>& x, U a=U()) {
  return minAbs(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U minAbs(const vector<T>& x, int i, int N, U a=U()) {
  return minAbs(x.data()+i, N, a);
}


template <class T, class U=T>
U minAbsOmp(const T *x, int N, U a=U()) {
  if (N<SIZE_MIN_OMPR) return minAbs(x, N, a);
  #pragma omp parallel for num_threads(32) schedule(auto) reduction(+:a)
  for (int i=0; i<N; i++)
    a = min(a, abs(x[i]));
  return a;
}

template <class T, class U=T>
U minAbsOmp(const vector<T>& x, U a=U()) {
  return minAbsOmp(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U minAbsOmp(const vector<T>& x, int i, int N, U a=U()) {
  return minAbsOmp(x.data()+i, N, a);
}




// MIN-AT
// ------

template <class T, class J, class U=T>
U minAt(const T *x, const J& is, U a=U()) {
  for (int i : is)
    a = min(a, x[i]);
  return a;
}

template <class T, class J, class U=T>
U minAt(const vector<T>& x, const J& is, U a=U()) {
  return minAt(x.data(), is, a);
}

template <class T, class J, class U=T>
U minAt(const vector<T>& x, int i, const J& is, U a=U()) {
  return minAt(x.data()+i, is, a);
}




// MIN-VALUE
// ---------

template <class T, class U>
void minValue(T *a, int N, const U& v) {
  for (int i=0; i<N; i++)
    a[i] = min(a[i], v);
}

template <class T, class U>
void minValue(vector<T>& a, const U& v) {
  minValue(a.data(), int(a.size()), v);
}

template <class T, class U>
void minValue(vector<T>& a, int i, int N, const U& v) {
  minValue(a.data()+i, N, v);
}


template <class T, class U>
void minValueOmp(T *a, int N, const U& v) {
  if (N<SIZE_MIN_OMPM) { minValue(a, N, v); return; }
  #pragma omp parallel for num_threads(32) schedule(auto)
  for (int i=0; i<N; i++)
    a[i] = min(a[i], v);
}

template <class T, class U>
void minValueOmp(vector<T>& a, const U& v) {
  minValueOmp(a.data(), int(a.size()), v);
}

template <class T, class U>
void minValueOmp(vector<T>& a, int i, int N, const U& v) {
  minValueOmp(a.data()+i, N, v);
}




// MIN-VALUE-AT
// ------------

template <class T, class U, class J>
void minValueAt(T *a, const U& v, const J& is) {
  for (int i : is)
    a[i] = min(a[i], v);
}

template <class T, class U, class J>
void minValueAt(vector<T>& a, const U& v, const J& is) {
  minValueAt(a.data(), v, is);
}

template <class T, class U, class J>
void minValueAt(vector<T>& a, int i, const U& v, const J& is) {
  minValueAt(a.data()+i, v, is);
}




// L1-NORM
// -------

template <class T, class U, class V=T>
V l1Norm(const T *x, const U *y, int N, V a=V()) {
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}

template <class T, class U, class V=T>
V l1Norm(const vector<T>& x, const vector<U>& y, V a=V()) {
  return l1Norm(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V l1Norm(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return l1Norm(x.data()+i, y.data()+i, N, a);
}


template <class T, class U, class V=T>
V l1NormOmp(const T *x, const U *y, int N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return l1Norm(x, y, N, a);
  #pragma omp parallel for num_threads(32) schedule(auto) reduction(+:a)
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}

template <class T, class U, class V=T>
V l1NormOmp(const vector<T>& x, const vector<U>& y, V a=V()) {
  return l1NormOmp(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V l1NormOmp(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return l1NormOmp(x.data()+i, y.data()+i, N, a);
}




// L2-NORM
// -------

template <class T, class U, class V=T>
V l2Norm(const T *x, const U *y, int N, V a=V()) {
  for (int i=0; i<N; i++)
    a += (x[i] - y[i]) * (x[i] - y[i]);
  return sqrt(a);
}

template <class T, class U, class V=T>
V l2Norm(const vector<T>& x, const vector<U>& y, V a=V()) {
  return l2Norm(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V l2Norm(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return l2Norm(x.data()+i, y.data()+i, N, a);
}


template <class T, class U, class V=T>
V l2NormOmp(const T *x, const U *y, int N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return l2Norm(x, y, N, a);
  #pragma omp parallel for num_threads(32) schedule(auto) reduction(+:a)
  for (int i=0; i<N; i++)
    a += (x[i] - y[i]) * (x[i] - y[i]);
  return sqrt(a);
}

template <class T, class U, class V=T>
V l2NormOmp(const vector<T>& x, const vector<U>& y, V a=V()) {
  return l2NormOmp(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V l2NormOmp(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return l2NormOmp(x.data()+i, y.data()+i, N, a);
}




// LI-NORM (INFINITY)
// ------------------

template <class T, class U, class V=T>
V liNorm(const T *x, const U *y, int N, V a=V()) {
  for (int i=0; i<N; i++)
    a = max(a, abs(x[i] - y[i]));
  return a;
}

template <class T, class U, class V=T>
V liNorm(const vector<T>& x, const vector<U>& y, V a=V()) {
  return liNorm(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V liNorm(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return liNorm(x.data()+i, y.data()+i, N, a);
}


template <class T, class U, class V=T>
V liNormOmp(const T *x, const U *y, int N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return liNorm(x, y, N, a);
  #pragma omp parallel for num_threads(32) schedule(auto) reduction(+:a)
  for (int i=0; i<N; i++)
    a = max(a, abs(x[i] - y[i]));
  return a;
}

template <class T, class U, class V=T>
V liNormOmp(const vector<T>& x, const vector<U>& y, V a=V()) {
  return liNormOmp(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V liNormOmp(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return liNormOmp(x.data()+i, y.data()+i, N, a);
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
  if (N<SIZE_MIN_OMPM) { multiply(a, x, y, N); return; }
  #pragma omp parallel for num_threads(32) schedule(auto)
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




// MULTIPLY-VALUE
// --------------

template <class T, class U, class V>
void multiplyValue(T *a, const U *x, const V& v, int N) {
  for (int i=0; i<N; i++)
    a[i] = T(x[i] * v);
}

template <class T, class U, class V>
void multiplyValue(vector<T>& a, const vector<U>& x, const V& v) {
  multiplyValue(a.data(), x.data(), v, int(x.size()));
}

template <class T, class U, class V>
void multiplyValue(vector<T>& a, const vector<U>& x, const V& v, int i, int N) {
  multiplyValue(a.data()+i, x.data()+i, v, N);
}


template <class T, class U, class V>
void multiplyValueOmp(T *a, const U *x, const V& v, int N) {
  if (N<SIZE_MIN_OMPM) { multiplyValue(a, x, v, N); return; }
  #pragma omp parallel for num_threads(32) schedule(auto)
  for (int i=0; i<N; i++)
    a[i] = T(x[i] * v);
}

template <class T, class U, class V>
void multiplyValueOmp(vector<T>& a, const vector<U>& x, const V& v) {
  multiplyValueOmp(a.data(), x.data(), v, int(x.size()));
}

template <class T, class U, class V>
void multiplyValueOmp(vector<T>& a, const vector<U>& x, const V& v, int i, int N) {
  multiplyValueOmp(a.data()+i, x.data()+i, v, N);
}
