#include <string>
#include <vector>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include "src/main.hxx"

using namespace std;




template <class G, class H>
void runPagerank(const G& x, const H& xt, int repeat) {
  vector<float> *init = nullptr;

  // Find pagerank using a single thread.
  auto a1 = pagerankNvgraph(xt, init, {repeat});
  auto e1 = l1Norm(a1.ranks, a1.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph\n", a1.time, a1.iterations, e1);

  // Find pagerank using CUDA thread-per-vertex.
  for (int g=1024; g<=GRID_LIMIT; g*=2) {
    for (int b=32; b<=BLOCK_LIMIT; b*=2) {
      auto a2 = pagerankCuda(xt, init, {repeat, g, b});
      auto e2 = l1Norm(a2.ranks, a1.ranks);
      printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankCuda<<<%d, %d>>>\n", a2.time, a2.iterations, e2, g, b);
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
