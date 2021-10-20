#include <cmath>
#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include "src/main.hxx"

using namespace std;




template <class G, class H>
void runPagerank(const G& x, const H& xt, int repeat) {
  float damping = 0.85f;
  int L1 = 1, L2 = 2, Li = 3;
  vector<float> *init = nullptr;

  // Find pagerank using default tolerance 10^-6, L1-norm.
  auto a1 = pagerankMonolithic(xt, init, {repeat, damping});
  auto e1 = l1Norm(a1.ranks, a1.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerank\n", a1.time, a1.iterations, e1);

  // Find pagerank using custom tolerance.
  for (int i=0; i<=20; i++) {
    float tolerance = pow(10.0f, -i/2) / (i&1? 2:1);

    // Find pagerank using L1 norm for convergence check.
    auto a2 = pagerankMonolithic(xt, init, {repeat, damping, tolerance, L1});
    auto e2 = l1Norm(a2.ranks, a1.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankL1Norm [tolerance=%.0e]\n", a2.time, a2.iterations, e2, tolerance);

    // Find pagerank using L2 norm for convergence check.
    auto a3 = pagerankMonolithic(xt, init, {repeat, damping, tolerance, L2});
    auto e3 = l1Norm(a3.ranks, a1.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankL2Norm [tolerance=%.0e]\n", a3.time, a3.iterations, e3, tolerance);

    // Find pagerank using L∞ norm for convergence check.
    auto a4 = pagerankMonolithic(xt, init, {repeat, damping, tolerance, Li});
    auto e4 = l1Norm(a4.ranks, a1.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankLiNorm [tolerance=%.0e]\n", a4.time, a4.iterations, e4, tolerance);
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
