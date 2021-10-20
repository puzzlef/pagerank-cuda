#include <cmath>
#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include "src/main.hxx"

using namespace std;




#define TYPE float


template <class G, class H>
void runPagerank(const G& x, const H& xt, int repeat) {
  enum ToleranceNorm { L1=1, L2=2, Li=3 };
  vector<TYPE> *init = nullptr;
  const TYPE damping = 0.85;

  // Find pagerank using default tolerance 10^-6, L1-norm.
  auto a0 = pagerankNvgraph(xt, init, {repeat, damping});
  auto e0 = l1Norm(a0.ranks, a0.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph\n", a0.time, a0.iterations, e0);

  // Find pagerank using custom tolerance.
  for (int i=0; i<=20; i++) {
    float tolerance = pow(10.0f, -i/2) / (i&1? 2:1);

    // Find nvGraph pagerank.
    auto a1 = pagerankNvgraph(xt, init, {repeat, damping, tolerance});
    auto e1 = l1Norm(a1.ranks, a0.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph [tolerance=%.0e]\n", a1.time, a1.iterations, e1, tolerance);

    // Find pagerank using L1 norm for convergence check.
    auto a2 = pagerankCuda(xt, init, {repeat, damping, tolerance, L1});
    auto e2 = l1Norm(a2.ranks, a0.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankCudaL1Norm [tolerance=%.0e]\n", a2.time, a2.iterations, e2, tolerance);
    auto a3 = pagerankSeq(xt, init, {repeat, damping, tolerance, L1});
    auto e3 = l1Norm(a3.ranks, a0.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankSeqL1Norm [tolerance=%.0e]\n", a3.time, a3.iterations, e3, tolerance);

    // Find pagerank using L2 norm for convergence check.
    auto a4 = pagerankCuda(xt, init, {repeat, damping, tolerance, L2});
    auto e4 = l1Norm(a4.ranks, a0.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankCudaL2Norm [tolerance=%.0e]\n", a4.time, a4.iterations, e4, tolerance);
    auto a5 = pagerankSeq(xt, init, {repeat, damping, tolerance, L2});
    auto e5 = l1Norm(a5.ranks, a0.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankSeqL2Norm [tolerance=%.0e]\n", a5.time, a5.iterations, e5, tolerance);

    // Find pagerank using L∞ norm for convergence check.
    auto a6 = pagerankCuda(xt, init, {repeat, damping, tolerance, Li});
    auto e6 = l1Norm(a6.ranks, a0.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankCudaLiNorm [tolerance=%.0e]\n", a6.time, a6.iterations, e6, tolerance);
    auto a7 = pagerankCuda(xt, init, {repeat, damping, tolerance, Li});
    auto e7 = l1Norm(a7.ranks, a0.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankSeqLiNorm [tolerance=%.0e]\n", a7.time, a7.iterations, e7, tolerance);
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  int repeat = argc>2? stoi(argv[2]) : 5;
  printf("Loading graph %s ...\n", file);
  auto x  = readMtx(file); println(x);
  auto xt = transposeWithDegree(x); print(xt); printf(" (transposeWithDegree)\n");
  runPagerank(x, xt, repeat);
  printf("\n");
  return 0;
}
