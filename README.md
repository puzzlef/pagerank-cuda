Comparing various launch configs for CUDA **switched-per-vertex** based
PageRank, focusing on **block approach** ([pull], [CSR], [switch-point]).

This experiment was for finding a suitable **launch config** for
**CUDA switched-per-vertex** for block approach. For the launch config,
the **block-size** (threads) was adjusted from `32`-`1024`, and the
**grid-limit** (max grid-size) was adjusted from `1024`-`32768`. Each config
was run 5 times per graph to get a good time measure. `MAXx256` appears to be
a good config for most graphs. Here `MAX` is the *grid-limit*, and `256` is
the *block-size*. Note that this applies to **Tesla V100 PCIe 16GB**, and
would be different for other GPUs. In order to measure error, [nvGraph]
pagerank is taken as a reference.

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
# [00011.327 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00012.189 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<1024, 32>>>
# [00011.330 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<1024, 64>>>
# [00011.295 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<1024, 128>>>
# [00011.136 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<1024, 256>>>
# [00011.135 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<1024, 512>>>
# [00011.308 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<1024, 1024>>>
# [00011.370 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<2048, 32>>>
# [00011.251 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<2048, 64>>>
# [00011.132 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<2048, 128>>>
# [00011.112 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<2048, 256>>>
# [00011.155 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<2048, 512>>>
# [00011.300 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<2048, 1024>>>
# [00011.277 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<4096, 32>>>
# [00011.093 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<4096, 64>>>
# [00011.105 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<4096, 128>>>
# [00011.131 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<4096, 256>>>
# [00011.167 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<4096, 512>>>
# [00011.302 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<4096, 1024>>>
# [00011.111 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<8192, 32>>>
# [00011.410 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<8192, 64>>>
# [00011.092 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<8192, 128>>>
# [00011.113 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<8192, 256>>>
# [00011.157 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<8192, 512>>>
# [00011.314 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<8192, 1024>>>
# [00011.084 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<16384, 32>>>
# [00011.090 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<16384, 64>>>
# [00011.098 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<16384, 128>>>
# [00011.093 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<16384, 256>>>
# [00011.149 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<16384, 512>>>
# [00011.299 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<16384, 1024>>>
# [00011.101 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<32768, 32>>>
# [00011.081 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<32768, 64>>>
# [00011.099 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<32768, 128>>>
# [00011.108 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<32768, 256>>>
# [00011.152 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<32768, 512>>>
# [00011.304 ms; 063 iters.] [6.9886e-07 err.] pagerankCuda<<<32768, 1024>>>
#
# ...
```

[![](https://i.imgur.com/XVcsXgB.gif)][sheets]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/uOYmbJZ.jpg)](https://www.youtube.com/watch?v=EQy5YjewJeU)

[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
[nvGraph]: https://github.com/rapidsai/nvgraph
["graphs"]: https://github.com/puzzlef/graphs
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[csr]: https://github.com/puzzlef/pagerank-class-vs-csr
[switch-point]: https://github.com/puzzlef/pagerank-cuda-switched-adjust-switch-point
[charts]: https://photos.app.goo.gl/fQzccCkR8bCjX6ne8
[sheets]: https://docs.google.com/spreadsheets/d/1JCb295fcFPTqImCj9uKvzY5m4_exW7ssO66C9tYEdfU/edit?usp=sharing
