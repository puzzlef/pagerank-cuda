#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include "src/main.hxx"

using namespace std;




template <class G, class H>
void runPagerank(const G& x, const H& xt, int repeat) {
  int L1 = 1, L2 = 2, Li = 3;
  vector<float> *init = nullptr;

  // Find pagerank using L1 norm for convergence check.
  auto a1 = pagerankMonolithic(xt, init, {repeat, L1});
  auto e1 = l1Norm(a1.ranks, a1.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankL1Norm\n", a1.time, a1.iterations, e1);

  // Find pagerank using L2 norm for convergence check.
  auto a2 = pagerankMonolithic(xt, init, {repeat, L2});
  auto e2 = l1Norm(a2.ranks, a1.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankL2Norm\n", a2.time, a2.iterations, e2);

  // Find pagerank using L∞ norm for convergence check.
  auto a3 = pagerankMonolithic(xt, init, {repeat, Li});
  auto e3 = l1Norm(a3.ranks, a1.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankLiNorm\n", a3.time, a3.iterations, e3);
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
