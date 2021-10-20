#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include "src/main.hxx"

using namespace std;




template <class G, class H>
void runPagerank(const G& x, const H& xt, int repeat) {
  vector<float> *init = nullptr;

  // Find pagerank using default damping factor 0.85.
  auto a1 = pagerankMonolithic(xt, init, {repeat});
  auto e1 = absError(a1.ranks, a1.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerank\n", a1.time, a1.iterations, e1);

  // Find pagerank using custom damping factors.
  for (float damping=1.0f; damping>0.45f; damping-=0.05f) {
    auto a2 = pagerankMonolithic(xt, init, {repeat, damping});
    auto e2 = absError(a2.ranks, a1.ranks);
    printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerank [damping=%.2f]\n", a2.time, a2.iterations, e2, damping);
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
