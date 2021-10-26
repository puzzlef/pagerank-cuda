Experimenting the effect of sorting vertices and/or edges by in-degree for
**CUDA** **thread-per-vertex** based PageRank ([pull], [CSR], [thread-launch]).

For this experiment, sorting of vertices and/or edges was either `NO`, `ASC`,
or `DESC`. This gives a total of `3 * 3 = 9` cases. Each case is run on
multiple graphs, running each 5 times per graph for good time measure. Results
show that sorting in *most cases* is **slower**. Maybe this is because sorted
arrangement tends to overflood certain memory chunks with too many requests.
In order to measure error, [nvGraph] pagerank is taken as a reference.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets]. The input
data used for this experiment is available at ["graphs"] (for small ones), and
the [SuiteSparse Matrix Collection]. This experiment was done with guidance
from [Prof. Dip Sankar Banerjee] and [Prof. Kishore Kothapalli].

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
# [00011.506 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00102.743 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda [sortv=NO; sorte=NO]
# [00101.473 ms; 063 iters.] [2.2609e-06 err.] pagerankCuda [sortv=NO; sorte=ASC]
# [00101.517 ms; 063 iters.] [1.0610e-05 err.] pagerankCuda [sortv=NO; sorte=DESC]
# [00115.918 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda [sortv=ASC; sorte=NO]
# [00113.659 ms; 063 iters.] [2.2609e-06 err.] pagerankCuda [sortv=ASC; sorte=ASC]
# [00115.622 ms; 063 iters.] [1.0610e-05 err.] pagerankCuda [sortv=ASC; sorte=DESC]
# [00131.644 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda [sortv=DESC; sorte=NO]
# [00126.931 ms; 063 iters.] [2.2609e-06 err.] pagerankCuda [sortv=DESC; sorte=ASC]
# [00131.776 ms; 063 iters.] [1.0610e-05 err.] pagerankCuda [sortv=DESC; sorte=DESC]
#
# ...
```

[![](https://i.imgur.com/XF9ByCY.gif)][sheetp]

[![](https://i.imgur.com/vURBt4y.png)][sheetp]
[![](https://i.imgur.com/YDq9PNQ.png)][sheetp]
[![](https://i.imgur.com/qx5lZxL.png)][sheetp]
[![](https://i.imgur.com/FBdY4h7.png)][sheetp]
[![](https://i.imgur.com/pyhVV6H.png)][sheetp]
[![](https://i.imgur.com/qS4xTa7.png)][sheetp]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/RTLTH4Q.jpg)](https://www.youtube.com/watch?v=1b8F1qa5-eM)
[![DOI](https://zenodo.org/badge/373940339.svg)](https://zenodo.org/badge/latestdoi/373940339)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
[nvGraph]: https://github.com/rapidsai/nvgraph
["graphs"]: https://github.com/puzzlef/graphs
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[csr]: https://github.com/puzzlef/pagerank-class-vs-csr
[thread-launch]: https://github.com/puzzlef/pagerank-cuda-thread-adjust-launch
[charts]: https://photos.app.goo.gl/whS1JrbAb165j53g8
[sheets]: https://docs.google.com/spreadsheets/d/1eaSvvIZw246yX59_SvW7apBTukn5tfTqlEZFJ7mkqKM/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vROIk883tXa4-Nwrpat6x7NRnPlnrM6kjplfNcrXmF4S0lFW_i656iBgnkp5RtPhAMIFlPpe79WohUD/pubhtml
