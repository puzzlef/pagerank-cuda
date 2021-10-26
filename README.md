Experimenting the effect of sorting vertices and/or edges by in-degree for CUDA **switched-per-vertex** based PageRank ([pull], [CSR], [block-launch], [thread-launch]).

For this experiment, sorting of vertices and/or edges was either `NO`, `ASC`,
or `DESC`. This gives a total of `3 * 3 = 9` cases. `NO` here means that
vertices are partitioned by in-degree (edges remain unchanged). Each case is
run on multiple graphs, running each 5 times per graph for good time measure.
Results show that **sorting** in most cases is **not faster**. Its better to
simply **partition** *vertices* by *degree*. In order to measure error,
[nvGraph] pagerank is taken as a reference.

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
# [00011.351 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00026.463 ms; 063 iters.] [7.2183e-07 err.] pagerankCuda [sortv=NO; sorte=NO]
# [00025.257 ms; 063 iters.] [6.7620e-07 err.] pagerankCuda [sortv=NO; sorte=ASC]
# [00025.195 ms; 063 iters.] [7.7441e-07 err.] pagerankCuda [sortv=NO; sorte=DESC]
# [00024.757 ms; 063 iters.] [7.2183e-07 err.] pagerankCuda [sortv=ASC; sorte=NO]
# [00024.401 ms; 063 iters.] [6.7620e-07 err.] pagerankCuda [sortv=ASC; sorte=ASC]
# [00024.518 ms; 063 iters.] [7.7441e-07 err.] pagerankCuda [sortv=ASC; sorte=DESC]
# [00024.559 ms; 063 iters.] [7.2210e-07 err.] pagerankCuda [sortv=DESC; sorte=NO]
# [00024.256 ms; 063 iters.] [6.7653e-07 err.] pagerankCuda [sortv=DESC; sorte=ASC]
# [00024.368 ms; 063 iters.] [7.7490e-07 err.] pagerankCuda [sortv=DESC; sorte=DESC]
#
# ...
```

[![](https://i.imgur.com/cJwEj0S.gif)][sheetp]

[![](https://i.imgur.com/bfBcTqH.png)][sheetp]
[![](https://i.imgur.com/Usj08Nw.png)][sheetp]
[![](https://i.imgur.com/I6BW30j.png)][sheetp]
[![](https://i.imgur.com/sLKFqVu.png)][sheetp]
[![](https://i.imgur.com/wOUaKH0.png)][sheetp]
[![](https://i.imgur.com/ImjIDzu.png)][sheetp]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/PQdIWEL.jpg)](https://www.youtube.com/watch?v=GAfOf26DuGk)
[![DOI](https://zenodo.org/badge/376314132.svg)](https://zenodo.org/badge/latestdoi/376314132)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
[nvGraph]: https://github.com/rapidsai/nvgraph
["graphs"]: https://github.com/puzzlef/graphs
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[csr]: https://github.com/puzzlef/pagerank-class-vs-csr
[block-launch]: https://github.com/puzzlef/pagerank-cuda-block-adjust-launch
[thread-launch]: https://github.com/puzzlef/pagerank-cuda-thread-adjust-launch
[charts]: https://photos.app.goo.gl/aoimeCU2px6SAP5z6
[sheets]: https://docs.google.com/spreadsheets/d/1EoP9whQLThF2UBKNGUeD_d_fCI6tVhtVeqiOt8fmmK8/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vToajY38qMw0rB7ipBXBjqPYXjiY6fb3h9Odx5BnHgRCP_xfJ81bRshVUNrN7RGsM0IktMM_j_jWUuZ/pubhtml
