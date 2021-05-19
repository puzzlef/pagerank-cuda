#pragma once
#include <vector>
#include <utility>

using std::vector;
using std::move;




template <class T>
struct PagerankOptions {
  int repeat;
  int gridSize;
  int blockSize;
  T   damping;
  T   tolerance;
  int maxIterations;

  PagerankOptions(int repeat=1, int gridSize=GRID_LIMIT, int blockSize=BLOCK_LIMIT, T damping=0.85, T tolerance=1e-6, int maxIterations=500) :
  repeat(repeat), gridSize(GRID_LIMIT), blockSize(BLOCK_LIMIT), damping(damping), tolerance(tolerance), maxIterations(maxIterations) {}
};


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


/*
template <class T>
struct PagerankCudaResult : PagerankResult<T> {
  int blocks;
  int threads;

  PagerankCudaResult(vector<T>&& ranks, int iterations=0, float time=0, int blocks=0, int threads=0) :
  PagerankResult(ranks, iterations, time), blocks(blocks), threads(threads) {}

  PagerankCudaResult(vector<T>& ranks, int iterations=0, float time=0, int blocks=0, int threads=0) :
  PagerankResult(ranks, iterations, time), blocks(blocks), threads(threads) {}
};
*/
