Experimenting the effect of sorting vertices by in-degree for CUDA
**block-per-vertex** based PageRank ([pull], [CSR], [block-launch]).

This experiment was for comparing the performance between:
1. Find pagerank using [nvGraph].
2. Find pagerank using *CUDA block-per-vertex*.
3. Find pagerank using *CUDA block-per-vertex*, with **vertices sorted by in-degree**.

Each approach is run on multiple graphs, running each 5 times per graph for
good time measure. For CUDA pagerank `4096x64` launch config is used (see
[block-launch]). As expected, it appears sorting vertices by in-degree has
**no consisten**t performance advantage. This is most likely because blocks
run independently. In order to measure error, [nvGraph] pagerank is taken as
a reference.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets]. The input
data used for this experiment is available at ["graphs"] (for small ones), and
the [SuiteSparse Matrix Collection].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -lnvgraph -O3 main.cu
$ ./a.out ~/data/min-1DeadEnd.mtx
$ ./a.out ~/data/min-2SCC.mtx
$ ...

# ...
#
# Loading graph /home/subhajit/data/web-Stanford.mtx ...
# order: 281903 size: 2312497 {}
# order: 281903 size: 2312497 {} (transposeWithDegree)
# [00011.409 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00032.328 ms; 063 iters.] [7.1476e-07 err.] pagerankCuda [sortv=NO; sorte=NO]
# [00031.637 ms; 063 iters.] [6.7824e-07 err.] pagerankCuda [sortv=NO; sorte=ASC]
# [00031.500 ms; 063 iters.] [7.5812e-07 err.] pagerankCuda [sortv=NO; sorte=DESC]
# [00029.522 ms; 063 iters.] [7.1485e-07 err.] pagerankCuda [sortv=ASC; sorte=NO]
# [00029.258 ms; 063 iters.] [6.7824e-07 err.] pagerankCuda [sortv=ASC; sorte=ASC]
# [00029.390 ms; 063 iters.] [7.5812e-07 err.] pagerankCuda [sortv=ASC; sorte=DESC]
# [00025.924 ms; 063 iters.] [7.1485e-07 err.] pagerankCuda [sortv=DESC; sorte=NO]
# [00025.723 ms; 063 iters.] [6.7824e-07 err.] pagerankCuda [sortv=DESC; sorte=ASC]
# [00025.777 ms; 063 iters.] [7.5812e-07 err.] pagerankCuda [sortv=DESC; sorte=DESC]
#
# ...
```

[![](https://i.imgur.com/zLPn59G.gif)][sheets]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/aEQNi9z.jpg)](https://www.youtube.com/watch?v=Q5hnBsUWmAI)

[nvGraph]: https://github.com/rapidsai/nvgraph
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[csr]: https://github.com/puzzlef/pagerank-class-vs-csr
[block-launch]: https://github.com/puzzlef/pagerank-cuda-block-adjust-launch
[charts]: https://photos.app.goo.gl/kCxgy62fFNjWqc8u7
[sheets]: https://docs.google.com/spreadsheets/d/1vcdBUAa_XQh3G3JVCQMSWhSZmeCffEqXz7EYN30FRZ0/edit?usp=sharing
["graphs"]: https://github.com/puzzlef/graphs
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
