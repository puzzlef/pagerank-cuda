#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include "src/main.hxx"

using namespace std;




template <class G, class H>
void runPagerank(const G& x, const H& xt, int repeat) {
  enum NormFunction { L0=0, L1=1, L2=2, Li=3 };
  vector<float> *init = nullptr;

  // Find nvGraph pagerank.
  auto a0 = pagerankNvgraph(xt, init, {repeat});
  auto e0 = l1Norm(a0.ranks, a0.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph\n", a0.time, a0.iterations, e0);

  // Find pagerank using L1 norm.
  for (int IS=L0; IS<=L2; IS++) {
    auto a1 = pagerankCuda(xt, init, {repeat, L1, IS});
    auto e1 = l1Norm(a1.ranks, a0.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankCudaL1Norm [iteration-scaling=L%d]\n", a1.time, a1.iterations, e1, IS);
    auto a2 = pagerankSeq(xt, init, {repeat, L1, IS});
    auto e2 = l1Norm(a2.ranks, a0.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankSeqL1Norm [iteration-scaling=L%d]\n", a2.time, a2.iterations, e2, IS);
  }

  // Find pagerank using L2 norm.
  for (int IS=L0; IS<=L2; IS++) {
    auto a3 = pagerankCuda(xt, init, {repeat, L2, IS});
    auto e3 = l1Norm(a3.ranks, a0.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankCudaL2Norm [iteration-scaling=L%d]\n", a3.time, a3.iterations, e3, IS);
    auto a4 = pagerankSeq(xt, init, {repeat, L2, IS});
    auto e4 = l1Norm(a4.ranks, a0.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankSeqL2Norm [iteration-scaling=L%d]\n", a4.time, a4.iterations, e4, IS);
  }

  // Find pagerank using L∞ norm.
  for (int IS=L0; IS<=L2; IS++) {
    auto a5 = pagerankCuda(xt, init, {repeat, Li, IS});
    auto e5 = l1Norm(a5.ranks, a0.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankCudaLiNorm [iteration-scaling=L%d]\n", a5.time, a5.iterations, e5, IS);
    auto a6 = pagerankSeq(xt, init, {repeat, Li, IS});
    auto e6 = l1Norm(a6.ranks, a0.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankSeqLiNorm [iteration-scaling=L%d]\n", a6.time, a6.iterations, e6, IS);
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
