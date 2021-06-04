Comparing various launch configs for CUDA block-per-vertex based PageRank ([pull], [CSR]).

This experiment was for finding a suitable **launch config** for
**CUDA block-per-vertex**. For the launch config, the **block-size** (threads)
was adjusted from `32`-`512`, and the **grid-limit** (max grid-size) was
adjusted from `1024`-`16384`. Each config was run 5 times per graph to get a
good time measure. `4096x64` appears to be a good config for most graphs. Here
`4096` is the *grid-limit*, and `64` is the *block-size*. Note that this
applies to **Tesla V100 PCIe 16GB**, and would be different for other GPUs. In
order to measure error, [nvGraph] pagerank is taken as a reference.

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
# [00011.269 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00060.501 ms; 063 iters.] [6.9758e-07 err.] pagerankCuda<<<1024, 32>>>
# [00042.322 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda<<<1024, 64>>>
# [00038.928 ms; 063 iters.] [6.9899e-07 err.] pagerankCuda<<<1024, 128>>>
# [00065.365 ms; 063 iters.] [6.8456e-07 err.] pagerankCuda<<<1024, 256>>>
# [00143.340 ms; 063 iters.] [6.8647e-07 err.] pagerankCuda<<<1024, 512>>>
# [00051.605 ms; 063 iters.] [6.9758e-07 err.] pagerankCuda<<<2048, 32>>>
# [00035.179 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda<<<2048, 64>>>
# [00036.248 ms; 063 iters.] [6.9899e-07 err.] pagerankCuda<<<2048, 128>>>
# [00062.885 ms; 063 iters.] [6.8456e-07 err.] pagerankCuda<<<2048, 256>>>
# [00143.210 ms; 063 iters.] [6.8647e-07 err.] pagerankCuda<<<2048, 512>>>
# [00047.988 ms; 063 iters.] [6.9758e-07 err.] pagerankCuda<<<4096, 32>>>
# [00031.465 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda<<<4096, 64>>>
# [00038.244 ms; 063 iters.] [6.9899e-07 err.] pagerankCuda<<<4096, 128>>>
# [00062.871 ms; 063 iters.] [6.8456e-07 err.] pagerankCuda<<<4096, 256>>>
# [00143.173 ms; 063 iters.] [6.8647e-07 err.] pagerankCuda<<<4096, 512>>>
# [00051.371 ms; 063 iters.] [6.9758e-07 err.] pagerankCuda<<<8192, 32>>>
# [00031.552 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda<<<8192, 64>>>
# [00038.243 ms; 063 iters.] [6.9899e-07 err.] pagerankCuda<<<8192, 128>>>
# [00062.835 ms; 063 iters.] [6.8456e-07 err.] pagerankCuda<<<8192, 256>>>
# [00143.185 ms; 063 iters.] [6.8647e-07 err.] pagerankCuda<<<8192, 512>>>
# [00051.492 ms; 063 iters.] [6.9758e-07 err.] pagerankCuda<<<16384, 32>>>
# [00031.478 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda<<<16384, 64>>>
# [00038.268 ms; 063 iters.] [6.9899e-07 err.] pagerankCuda<<<16384, 128>>>
# [00062.825 ms; 063 iters.] [6.8456e-07 err.] pagerankCuda<<<16384, 256>>>
# [00143.263 ms; 063 iters.] [6.8647e-07 err.] pagerankCuda<<<16384, 512>>>
#
# ...
```

[![](https://i.imgur.com/H3cJmIc.gif)][sheets]
[![](https://i.imgur.com/dlIWium.gif)][sheets]
[![](https://i.imgur.com/mhXIjGv.gif)][sheets]
[![](https://i.imgur.com/uvZvq1M.gif)][sheets]
[![](https://i.imgur.com/ed0AeO0.gif)][sheets]
[![](https://i.imgur.com/iopCtfm.gif)][sheets]
[![](https://i.imgur.com/OPoy5Nv.gif)][sheets]
[![](https://i.imgur.com/XxIXsm1.gif)][sheets]
[![](https://i.imgur.com/iKLC9XN.gif)][sheets]
[![](https://i.imgur.com/vkUDtHx.gif)][sheets]
[![](https://i.imgur.com/3F44wku.gif)][sheets]
[![](https://i.imgur.com/7fDcELX.gif)][sheets]
[![](https://i.imgur.com/tvoqR83.gif)][sheets]
[![](https://i.imgur.com/OwlDvLt.gif)][sheets]
[![](https://i.imgur.com/fxypucV.gif)][sheets]
[![](https://i.imgur.com/xLPVBVs.gif)][sheets]
[![](https://i.imgur.com/K5QNR1H.gif)][sheets]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/QIUy2ds.jpg)](https://www.youtube.com/watch?v=4EG2up-jcKM&t=12897s)

[nvGraph]: https://github.com/rapidsai/nvgraph
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[csr]: https://github.com/puzzlef/pagerank-class-vs-csr
[charts]: https://photos.app.goo.gl/Sj5u3P9dYzMk2tM8A
[sheets]: https://docs.google.com/spreadsheets/d/1Vqa9Kt1jU7Te9cB29HDZF8O_VfiwJOkNb1eu6mcUDrY/edit?usp=sharing
["graphs"]: https://github.com/puzzlef/graphs
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
