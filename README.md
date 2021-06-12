Experimenting the effect of sorting vertices and/or edges by in-degree for CUDA
**block-per-vertex** based PageRank ([pull], [CSR], [block-launch]).

This experiment was for comparing the performance between:
1. Find pagerank using [nvGraph].
3. Find pagerank using *CUDA block-per-vertex*, with **vertices sorted by in-degree**.

For this experiment, sorting of vertices and/or edges was either `NO`, `ASC`,
or `DESC`. This gives a total of `3 * 3 = 9` cases. Each case is run on
multiple graphs, running each 5 times per graph for good time measure. Results
show that sorting in *most cases* is **not faster**. In fact, in a number of
cases, sorting actually slows dows performance. Maybe (just maybe) this is
because sorted arrangement tend to overflood certain memory chunks with too
many requests. In order to measure error, [nvGraph] pagerank is taken as
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

[![](https://i.imgur.com/Ha2JGkg.gif)][sheets]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/TG0K9e0.jpg)](https://www.youtube.com/watch?v=eQqsP388S3Q)

[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
[nvGraph]: https://github.com/rapidsai/nvgraph
["graphs"]: https://github.com/puzzlef/graphs
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[csr]: https://github.com/puzzlef/pagerank-class-vs-csr
[block-launch]: https://github.com/puzzlef/pagerank-cuda-block-adjust-launch
[charts]: https://photos.app.goo.gl/EfwWVoXnAAhNJWdH7
[sheets]: https://docs.google.com/spreadsheets/d/16L-b5ofUZbA6xBYbFZESXbv5hs1FTtw_K0FHJLISlNM/edit?usp=sharing
