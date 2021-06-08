#include <cstdio>
#include <iostream>
#include <algorithm>
#include "src/main.hxx"

using namespace std;




#define REPEAT 1

template <class G, class H>
void runPagerank(const G& x, const H& xt, bool show) {
  vector<float> *init = nullptr;

  // Find pagerank using a single thread.
  auto a1 = pagerankNvgraph(xt, init, {REPEAT});
  auto e1 = l1Norm(a1.ranks, a1.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph\n", a1.time, a1.iterations, e1);

  // Find pagerank using CUDA thread-per-vertex.
  for (int degree=2; degree<=1024; degree*=2) {
    for (int limit=32; limit<=32; limit*=2) {
      auto a2 = pagerankCuda(xt, init, {REPEAT, degree, limit});
      auto e2 = l1Norm(a2.ranks, a1.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankCuda [degree=%d; limit=%d]\n", a2.time, a2.iterations, e2, degree, limit);
    }
  }
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
