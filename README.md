Comparing various launch configs for CUDA thread-per-vertex based PageRank ([pull], [CSR]).

This experiment was for finding a suitable **launch config** for
**CUDA thread-per-vertex**. For the launch config, the **block-size** (threads)
was adjusted from `32`-`1024`, and the **grid-limit** (max grid-size) was
adjusted from `1024`-`32768`. Each config was run 5 times per graph to get a
good time measure. On average, the launch config doesn't seem to have a good
enough impact on performance. However `8192x128` appears to be a good config.
Here `8192` is the *grid-limit*, and `128` is the *block-size*. Comparing with
[graph properties], seems it would be better to use `8192x512` for graphs with
**high** *avg. density*, and `8192x32` for graphs with **high** *avg. degree*.
Maybe, sorting the vertices by degree can have a good effect (due to less warp
divergence). Note that this applies to **Tesla V100 PCIe 16GB**, and would be
different for other GPUs. In order to measure error, [nvGraph] pagerank is
taken as a reference.

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
# [00011.263 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00103.942 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<1024, 32>>>
# [00100.036 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<1024, 64>>>
# [00097.901 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<1024, 128>>>
# [00101.922 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<1024, 256>>>
# [00099.972 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<1024, 512>>>
# [00101.744 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<1024, 1024>>>
# [00100.130 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<2048, 32>>>
# [00100.370 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<2048, 64>>>
# [00100.683 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<2048, 128>>>
# [00101.702 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<2048, 256>>>
# [00099.575 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<2048, 512>>>
# [00101.145 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<2048, 1024>>>
# [00101.275 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<4096, 32>>>
# [00101.120 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<4096, 64>>>
# [00101.144 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<4096, 128>>>
# [00101.667 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<4096, 256>>>
# [00099.474 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<4096, 512>>>
# [00101.597 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<4096, 1024>>>
# [00101.624 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<8192, 32>>>
# [00101.279 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<8192, 64>>>
# [00100.481 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<8192, 128>>>
# [00101.429 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<8192, 256>>>
# [00099.456 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<8192, 512>>>
# [00101.233 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<8192, 1024>>>
# [00101.630 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<16384, 32>>>
# [00101.113 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<16384, 64>>>
# [00100.989 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<16384, 128>>>
# [00101.824 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<16384, 256>>>
# [00099.641 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<16384, 512>>>
# [00101.362 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<16384, 1024>>>
# [00101.535 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<32768, 32>>>
# [00101.010 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<32768, 64>>>
# [00100.467 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<32768, 128>>>
# [00101.605 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<32768, 256>>>
# [00099.558 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<32768, 512>>>
# [00101.050 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<32768, 1024>>>
#
# ...
```

[![](https://i.imgur.com/MgbYwZW.gif)][sheetp]
[![](https://i.imgur.com/noDgSTU.gif)][sheetp]
[![](https://i.imgur.com/iip3nyk.gif)][sheetp]
[![](https://i.imgur.com/jhxGnSj.gif)][sheetp]
[![](https://i.imgur.com/yewTKTf.gif)][sheetp]
[![](https://i.imgur.com/2WjE3xU.gif)][sheetp]
[![](https://i.imgur.com/sQKOoCi.gif)][sheetp]
[![](https://i.imgur.com/EfACavn.gif)][sheetp]
[![](https://i.imgur.com/xd9AUaf.gif)][sheetp]
[![](https://i.imgur.com/EEQ5May.gif)][sheetp]
[![](https://i.imgur.com/iiPBesX.gif)][sheetp]
[![](https://i.imgur.com/KiVeTer.gif)][sheetp]
[![](https://i.imgur.com/7cfd36t.gif)][sheetp]
[![](https://i.imgur.com/ZIPdnuR.gif)][sheetp]
[![](https://i.imgur.com/g5dcDf5.gif)][sheetp]
[![](https://i.imgur.com/FiTwKuL.gif)][sheetp]
[![](https://i.imgur.com/4B3LROo.gif)][sheetp]

[![](https://i.imgur.com/2Dnz5Lw.png)][sheetp]
[![](https://i.imgur.com/dBOYYxh.png)][sheetp]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/4Slx4Ma.jpg)](https://www.youtube.com/watch?v=4EG2up-jcKM)
[![DOI](https://zenodo.org/badge/368730692.svg)](https://zenodo.org/badge/latestdoi/368730692)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
[nvGraph]: https://github.com/rapidsai/nvgraph
["graphs"]: https://github.com/puzzlef/graphs
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[csr]: https://github.com/puzzlef/pagerank-class-vs-csr
[graph properties]: https://docs.google.com/spreadsheets/d/16viria4blm3e4AsF0iaPk03i_OXCFN8optcrOPwbCJ8/edit?usp=sharing
[charts]: https://photos.app.goo.gl/k4vQDiMwF3awyhJZA
[sheets]: https://docs.google.com/spreadsheets/d/1S818mfYL_zUbgWB-jxk1BIFDnB6oYH8IhrlGp4wHMsw/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vS83xuRKxv2XnXro49Cs8EK6VPyGaAR615sNoqsHtbDa2lmSyBdNJ62tzNAoIPUr-MHb5_W-5lXjMYr/pubhtml
