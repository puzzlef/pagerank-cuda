#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include "src/main.hxx"

using namespace std;




#define TYPE float


template <class G, class H>
void runPagerank(const G& x, const H& xt, int repeat) {
  int L1 = 1, L2 = 2, Li = 3;
  vector<TYPE> *init = nullptr;

  // Find pagerank using nvGraph.
  auto a0 = pagerankNvgraph(xt, init, {repeat});
  auto e0 = l1Norm(a0.ranks, a0.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph\n", a0.time, a0.iterations, e0);

  // TODO: ACCEPT NORM!
  // Find pagerank using L1 norm for convergence check.
  auto a1 = pagerankCuda(xt, init, {repeat, L1});
  auto e1 = l1Norm(a1.ranks, a0.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankCudaL1Norm\n", a1.time, a1.iterations, e1);
  auto a2 = pagerankSeq(xt, init, {repeat, L1});
  auto e2 = l1Norm(a2.ranks, a0.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankSeqL1Norm\n", a2.time, a2.iterations, e2);

  // Find pagerank using L2 norm for convergence check.
  auto a3 = pagerankCuda(xt, init, {repeat, L2});
  auto e3 = l1Norm(a3.ranks, a0.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankCudaL2Norm\n", a3.time, a3.iterations, e3);
  auto a4 = pagerankSeq(xt, init, {repeat, L2});
  auto e4 = l1Norm(a4.ranks, a0.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankSeqL2Norm\n", a4.time, a4.iterations, e4);

  // Find pagerank using L∞ norm for convergence check.
  auto a5 = pagerankCuda(xt, init, {repeat, Li});
  auto e5 = l1Norm(a5.ranks, a0.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankCudaLiNorm\n", a5.time, a5.iterations, e5);
  auto a6 = pagerankCuda(xt, init, {repeat, Li});
  auto e6 = l1Norm(a6.ranks, a0.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankSeqLiNorm\n", a6.time, a6.iterations, e6);
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
