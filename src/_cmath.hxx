#pragma once
#include <cmath>
#include <type_traits>

using std::is_floating_point;
using std::ceil;




// CEIL-DIV
// --------

template <class T>
__host__ __device__ T ceilDiv(T x, T y) { return (x + y-1) / y; }
template <>
__host__ __device__ float ceilDiv<float>(float x, float y) { return ceil(x/y); }
template <>
__host__ __device__ double ceilDiv<double>(double x, double y) { return ceil(x/y); }




// POW-2
// -----

template <class T>
__host__ __device__ constexpr bool isPow2(T x) noexcept {
  return !(x & (x-1));
}


template <class T>
__host__ __device__ constexpr T prevPow2(T x) noexcept {
  return 1 << T(log2(x));
}


template <class T>
__host__ __device__ constexpr T nextPow2(T x) noexcept {
  return 1 << T(ceil(log2(x)));
}
