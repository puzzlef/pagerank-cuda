Comparing various launch configs for CUDA thread-per-vertex based PageRank ([pull], [CSR]).

This experiment was for finding a suitable **launch config** for
**CUDA thread-per-vertex**. For the launch config, the **block-size** (threads)
was adjusted from `32`-`512`, and the **grid-limit** (max grid-size) was
adjusted from `1024`-`16384`. Each config was run 5 times per graph to get a
good time measure. On average, the launch config doesn't seem to have a good
enough impact on performance. However `4096x128` appears to be a good config.
Here `4096` is the *grid-limit*, and `128` is the *block-size*. Maybe, sorting
the vertices by degree can have a good effect (due to less warp divergence).
Note that this applies to **Tesla V100 PCIe 16GB**, and would be different
for other GPUs. In order to measure error, [nvGraph] pagerank is taken as a
reference.

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
# [00011.251 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00103.863 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<1024, 32>>>
# [00099.834 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<1024, 64>>>
# [00097.572 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<1024, 128>>>
# [00101.079 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<1024, 256>>>
# [00099.433 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<1024, 512>>>
# [00099.875 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<2048, 32>>>
# [00100.523 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<2048, 64>>>
# [00101.368 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<2048, 128>>>
# [00101.532 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<2048, 256>>>
# [00099.603 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<2048, 512>>>
# [00103.162 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<4096, 32>>>
# [00102.150 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<4096, 64>>>
# [00100.963 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<4096, 128>>>
# [00101.582 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<4096, 256>>>
# [00099.681 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<4096, 512>>>
# [00104.623 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<8192, 32>>>
# [00102.420 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<8192, 64>>>
# [00100.976 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<8192, 128>>>
# [00101.441 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<8192, 256>>>
# [00099.665 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<8192, 512>>>
# [00104.600 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<16384, 32>>>
# [00102.470 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<16384, 64>>>
# [00100.898 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<16384, 128>>>
# [00101.555 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<16384, 256>>>
# [00099.522 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda<<<16384, 512>>>
#
# ...
```

[![](https://i.imgur.com/lOHnic2.gif)][sheets]
[![](https://i.imgur.com/RjGOcNQ.gif)][sheets]
[![](https://i.imgur.com/N4tJvq4.gif)][sheets]
[![](https://i.imgur.com/voJ7Yfd.gif)][sheets]
[![](https://i.imgur.com/QF9N3eG.gif)][sheets]
[![](https://i.imgur.com/0wCENO6.gif)][sheets]
[![](https://i.imgur.com/q6hu6fb.gif)][sheets]
[![](https://i.imgur.com/Dv4QLap.gif)][sheets]
[![](https://i.imgur.com/iWYMnqg.gif)][sheets]
[![](https://i.imgur.com/9luV2Jj.gif)][sheets]
[![](https://i.imgur.com/E6O8Y2f.gif)][sheets]
[![](https://i.imgur.com/gt1Wzzw.gif)][sheets]
[![](https://i.imgur.com/1YO2pix.gif)][sheets]
[![](https://i.imgur.com/wDALNSS.gif)][sheets]
[![](https://i.imgur.com/BcyvXG9.gif)][sheets]
[![](https://i.imgur.com/dRSKZMc.gif)][sheets]
[![](https://i.imgur.com/TyeTcDq.gif)][sheets]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/XbhF5s7.jpg)](https://www.youtube.com/watch?v=4EG2up-jcKM)

[nvGraph]: https://github.com/rapidsai/nvgraph
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[csr]: https://github.com/puzzlef/pagerank-class-vs-csr
[charts]: https://photos.app.goo.gl/k4vQDiMwF3awyhJZA
[sheets]: https://docs.google.com/spreadsheets/d/1NutV_Pe4WGBrYhkqU5Yu-bqCAcWbfP-qahI3ZnxVASo/edit?usp=sharing
["graphs"]: https://github.com/puzzlef/graphs
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
