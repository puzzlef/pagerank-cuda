#include <cstdio>
#include <iostream>
#include "src/main.hxx"

using namespace std;




#define REPEAT 5

template <class G, class H>
void runPagerank(const G& x, const H& xt, bool show) {
  vector<float> *init = nullptr;

  // Find pagerank using nvGraph.
  auto a1 = pagerankNvgraph(xt, init, {REPEAT});
  auto e1 = l1Norm(a1.ranks, a1.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph\n", a1.time, a1.iterations, e1);
  if (show) println(a1.ranks);

  // Find pagerank accelerated using CUDA.
  auto a2 = pagerankCuda(xt, init, {REPEAT});
  auto e2 = l1Norm(a2.ranks, a1.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankCuda\n", a2.time, a2.iterations, e2);

  // Find pagerank using a single thread.
  auto a3 = pagerankSeq(xt, init, {REPEAT});
  auto e3 = l1Norm(a3.ranks, a1.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankSeq\n", a3.time, a3.iterations, e3);
}


int main(int argc, char **argv) {
  char *file = argv[1];
  bool  show = argc > 2;
  printf("Loading graph %s ...\n", file);
  auto x  = readMtx(file); println(x);
  auto xt = transposeWithDegree(x); print(xt); printf(" (transposeWithDegree)\n");
  runPagerank(x, xt, show);
  printf("\n");
  return 0;
}
