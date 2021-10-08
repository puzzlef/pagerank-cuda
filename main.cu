#include <string>
#include <vector>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include "src/main.hxx"

using namespace std;




#define TYPE float


template <class G, class H>
void runPagerank(const G& x, const H& xt, int repeat) {
  vector<TYPE> *init = nullptr;

  // Find pagerank using a single thread.
  auto a1 = pagerankNvgraph(xt, init, {repeat});
  auto e1 = l1Norm(a1.ranks, a1.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph\n", a1.time, a1.iterations, e1);

  // Find pagerank using CUDA switched-per-vertex.
  for (int degree=2; degree<=BLOCK_LIMIT; degree*=2) {
    for (int limit=1; limit<=BLOCK_LIMIT; limit*=2) {
      auto a2 = pagerankCuda(xt, init, {repeat, degree, limit});
      auto e2 = l1Norm(a2.ranks, a1.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankCuda [degree=%d; limit=%d]\n", a2.time, a2.iterations, e2, degree, limit);
    }
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
