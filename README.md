Comparing various launch configs for **CUDA block-per-vertex** based PageRank
([pull], [CSR]).

This experiment was for finding a suitable **launch config** for
**CUDA block-per-vertex**. For the launch config, the **block-size** (threads)
was adjusted from `32`-`1024`, and the **grid-limit** (max grid-size) was
adjusted from `1024`-`32768`. Each config was run 5 times per graph to get a
good time measure. `MAXx64` appears to be a good config for most graphs. Here
`MAX` is the **grid-limit**, and `64` is the **block-size**. This launch
config is for the entire graph, and could be slightly different for subset of
graphs. Also note that this applies to *Tesla V100 PCIe 16GB*, and could be
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
# [00011.441 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00061.493 ms; 063 iters.] [6.9684e-07 err.] pagerankCuda<<<1024, 32>>>
# [00042.466 ms; 063 iters.] [7.1421e-07 err.] pagerankCuda<<<1024, 64>>>
# [00039.030 ms; 063 iters.] [6.9823e-07 err.] pagerankCuda<<<1024, 128>>>
# [00065.472 ms; 063 iters.] [6.8380e-07 err.] pagerankCuda<<<1024, 256>>>
# [00145.954 ms; 063 iters.] [6.8548e-07 err.] pagerankCuda<<<1024, 512>>>
# [00329.866 ms; 063 iters.] [6.8467e-07 err.] pagerankCuda<<<1024, 1024>>>
# [00051.042 ms; 063 iters.] [6.9684e-07 err.] pagerankCuda<<<2048, 32>>>
# [00034.991 ms; 063 iters.] [7.1421e-07 err.] pagerankCuda<<<2048, 64>>>
# [00036.222 ms; 063 iters.] [6.9823e-07 err.] pagerankCuda<<<2048, 128>>>
# [00065.839 ms; 063 iters.] [6.8380e-07 err.] pagerankCuda<<<2048, 256>>>
# [00133.297 ms; 063 iters.] [6.8548e-07 err.] pagerankCuda<<<2048, 512>>>
# [00307.478 ms; 063 iters.] [6.8467e-07 err.] pagerankCuda<<<2048, 1024>>>
# [00046.739 ms; 063 iters.] [6.9684e-07 err.] pagerankCuda<<<4096, 32>>>
# [00030.503 ms; 063 iters.] [7.1421e-07 err.] pagerankCuda<<<4096, 64>>>
# [00035.438 ms; 063 iters.] [6.9823e-07 err.] pagerankCuda<<<4096, 128>>>
# [00060.317 ms; 063 iters.] [6.8380e-07 err.] pagerankCuda<<<4096, 256>>>
# [00126.383 ms; 063 iters.] [6.8548e-07 err.] pagerankCuda<<<4096, 512>>>
# [00306.623 ms; 063 iters.] [6.8467e-07 err.] pagerankCuda<<<4096, 1024>>>
# [00048.424 ms; 063 iters.] [6.9684e-07 err.] pagerankCuda<<<8192, 32>>>
# [00032.419 ms; 063 iters.] [7.1421e-07 err.] pagerankCuda<<<8192, 64>>>
# [00033.686 ms; 063 iters.] [6.9823e-07 err.] pagerankCuda<<<8192, 128>>>
# [00059.334 ms; 063 iters.] [6.8380e-07 err.] pagerankCuda<<<8192, 256>>>
# [00125.928 ms; 063 iters.] [6.8548e-07 err.] pagerankCuda<<<8192, 512>>>
# [00304.776 ms; 063 iters.] [6.8467e-07 err.] pagerankCuda<<<8192, 1024>>>
# [00050.515 ms; 063 iters.] [6.9684e-07 err.] pagerankCuda<<<16384, 32>>>
# [00034.612 ms; 063 iters.] [7.1421e-07 err.] pagerankCuda<<<16384, 64>>>
# [00034.573 ms; 063 iters.] [6.9823e-07 err.] pagerankCuda<<<16384, 128>>>
# [00059.193 ms; 063 iters.] [6.8380e-07 err.] pagerankCuda<<<16384, 256>>>
# [00126.409 ms; 063 iters.] [6.8548e-07 err.] pagerankCuda<<<16384, 512>>>
# [00304.214 ms; 063 iters.] [6.8467e-07 err.] pagerankCuda<<<16384, 1024>>>
# [00051.362 ms; 063 iters.] [6.9684e-07 err.] pagerankCuda<<<32768, 32>>>
# [00035.592 ms; 063 iters.] [7.1421e-07 err.] pagerankCuda<<<32768, 64>>>
# [00036.476 ms; 063 iters.] [6.9823e-07 err.] pagerankCuda<<<32768, 128>>>
# [00058.218 ms; 063 iters.] [6.8380e-07 err.] pagerankCuda<<<32768, 256>>>
# [00126.185 ms; 063 iters.] [6.8548e-07 err.] pagerankCuda<<<32768, 512>>>
# [00306.288 ms; 063 iters.] [6.8467e-07 err.] pagerankCuda<<<32768, 1024>>>
#
# ...
```

[![](https://i.imgur.com/bwedZN8.gif)][sheets]
[![](https://i.imgur.com/SYY0VTV.gif)][sheets]
[![](https://i.imgur.com/0ThK2pd.gif)][sheets]
[![](https://i.imgur.com/a7AKdLx.gif)][sheets]
[![](https://i.imgur.com/sxbRgJF.gif)][sheets]
[![](https://i.imgur.com/crTZjmn.gif)][sheets]
[![](https://i.imgur.com/cxLbgqj.gif)][sheets]
[![](https://i.imgur.com/m9KGsyj.gif)][sheets]
[![](https://i.imgur.com/V5Xp74C.gif)][sheets]
[![](https://i.imgur.com/LW2qAcp.gif)][sheets]
[![](https://i.imgur.com/Kt1Uzyk.gif)][sheets]
[![](https://i.imgur.com/UGah41u.gif)][sheets]
[![](https://i.imgur.com/o9maK87.gif)][sheets]
[![](https://i.imgur.com/GQJRono.gif)][sheets]
[![](https://i.imgur.com/rou4VBX.gif)][sheets]
[![](https://i.imgur.com/D73ZUaf.gif)][sheets]
[![](https://i.imgur.com/sX2dCEb.gif)][sheets]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](https://www.youtube.com/watch?v=ke9g8hB0MEo)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://www.slideshare.net/SubhajitSahu/cuda-by-example-notes)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [nvGraph pagerank example, EN605.617, JHU-EP-Intro2GPU](https://github.com/JHU-EP-Intro2GPU/EN605.617/blob/master/module9/nvgraph_examples/nvgraph_Pagerank.cpp)
- [nvGraph pagerank example, CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/10.0/nvgraph/index.html#nvgraph-pagerank-example)
- [nvGraph Library User's Guide](https://docs.nvidia.com/cuda/archive/10.1/pdf/nvGRAPH_Library.pdf)
- [RAPIDS nvGraph NVIDIA graph library][nvGraph]
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/QIUy2ds.jpg)](https://www.youtube.com/watch?v=4EG2up-jcKM&t=12897s)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
["graphs"]: https://github.com/puzzlef/graphs
[nvGraph]: https://github.com/rapidsai/nvgraph
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[csr]: https://github.com/puzzlef/pagerank-class-vs-csr
[charts]: https://photos.app.goo.gl/8uvRf81gpiBFNjFS6
[sheets]: https://docs.google.com/spreadsheets/d/1Vqa9Kt1jU7Te9cB29HDZF8O_VfiwJOkNb1eu6mcUDrY/edit?usp=sharing
