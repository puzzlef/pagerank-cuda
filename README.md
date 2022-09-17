Performance of [sequential] execution based vs **CUDA** based PageRank ([pull], [CSR]).

This experiment was for comparing the performance between:
1. Find pagerank using [nvGraph].
2. Find pagerank using **CUDA**.
3. Find pagerank using a single thread ([sequential]).

Each technique was attempted on different types of graphs, running each
technique 5 times per graph to get a good time measure. **CUDA** is the
[switched-per-vertex] approach running on GPU. **CUDA** based pagerank is
indeed much faster than **sequential** (CPU). In order to measure error,
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
# [00011.345 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00011.150 ms; 063 iters.] [7.0094e-07 err.] pagerankCuda
# [00439.086 ms; 063 iters.] [5.0388e-06 err.] pagerankSeq
#
# ...
#
# Loading graph /home/subhajit/data/soc-LiveJournal1.mtx ...
# order: 4847571 size: 68993773 {}
# order: 4847571 size: 68993773 {} (transposeWithDegree)
# [00168.225 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00158.135 ms; 051 iters.] [3.2208e-06 err.] pagerankCuda
# [13072.056 ms; 051 iters.] [2.0581e-03 err.] pagerankSeq
#
# ...
```

[![](https://i.imgur.com/vDeiY1n.gif)][sheetp]

[![](https://i.imgur.com/N1EUPCS.png)][sheetp]
[![](https://i.imgur.com/5LaxhV4.png)][sheetp]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/fjeKRUf.jpg)](https://www.youtube.com/watch?v=TtTHBmL7N5U)
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/368720311.svg)](https://zenodo.org/badge/latestdoi/368720311)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
[nvGraph]: https://github.com/rapidsai/nvgraph
["graphs"]: https://github.com/puzzlef/graphs
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[csr]: https://github.com/puzzlef/pagerank-class-vs-csr
[sequential]: https://github.com/puzzlef/pagerank-sequential-vs-openmp
[switched-per-vertex]: https://github.com/puzzlef/pagerank-cuda-switched-adjust-switch-point
[charts]: https://photos.app.goo.gl/MLcbhUPmLEC7iaEm9
[sheets]: https://docs.google.com/spreadsheets/d/12u5yq49MLS2QRhWHkZF7SWs1JSS4u1sb7wKl8ExrJgg/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vTijFuWx76ZnNfJs5U0IEY1jMEWffi6Pc8uw4FbnXB1R3Puduyn-mPvq4kdMFyyhq0V7GJZQ0722nDS/pubhtml
