#pragma once
#include <vector>
#include <string>
#include <utility>
#include "_main.hxx"

using std::vector;
using std::string;
using std::move;




// LAUNCH CONFIG
// -------------

// For pagerank cuda block-per-vertex
template <class T=float>
constexpr int BLOCK_DIM_PRB() noexcept { return 64; }
template <class T=float>
constexpr int GRID_DIM_PRB() noexcept  { return GRID_LIMIT; }

// For pagerank cuda thread-per-vertex (default)
template <class T=float>
constexpr int BLOCK_DIM_PRT() noexcept { return 128; }
template <class T=float>
constexpr int GRID_DIM_PRT() noexcept  { return 8192; }
template <class T=float>
constexpr int BLOCK_DIM_PRT_LOWDENSITY() noexcept { return 512; }
template <class T=float>
constexpr int GRID_DIM_PRT_LOWDENSITY() noexcept  { return 8192; }
template <class T=float>
constexpr int BLOCK_DIM_PRT_HIGHDEGREE() noexcept { return 32; }
template <class T=float>
constexpr int GRID_DIM_PRT_HIGHDEGREE() noexcept  { return 8192; }




// PAGERANK-SORT
// -------------

enum class PagerankSort {
  NO,
  ASC,
  DESC
};

string stringify(PagerankSort x) {
  typedef PagerankSort Sort;
  switch (x) {
    default:
    case Sort::NO:   return "NO";
    case Sort::ASC:  return "ASC";
    case Sort::DESC: return "DESC";
  }
}




// PAGERANK-OPTIONS
// ----------------

template <class T>
struct PagerankOptions {
  typedef PagerankSort Sort;
  int  repeat;
  Sort sortVertices;
  Sort sortEdges;
  T    damping;
  T    tolerance;
  int  maxIterations;

  PagerankOptions(int repeat=1, Sort sortVertices=Sort::NO, Sort sortEdges=Sort::NO, T damping=0.85, T tolerance=1e-6, int maxIterations=500) :
  repeat(repeat), sortVertices(sortVertices), sortEdges(sortEdges), damping(damping), tolerance(tolerance), maxIterations(maxIterations) {}
};




// PAGERANK-RESULT
// ---------------

template <class T>
struct PagerankResult {
  vector<T> ranks;
  int   iterations;
  float time;

  PagerankResult(vector<T>&& ranks, int iterations=0, float time=0) :
  ranks(ranks), iterations(iterations), time(time) {}

  PagerankResult(vector<T>& ranks, int iterations=0, float time=0) :
  ranks(move(ranks)), iterations(iterations), time(time) {}
};
