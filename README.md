Comparing various launch configs for CUDA thread-per-vertex based PageRank ([pull], [CSR]).

This experiment was for comparing the performance between:
1. Find pagerank using [nvGraph].
2. Find pagerank using **CUDA** *thread-per-vertex* with various launch configs.

For the launch config, the **block-size** (threads) was adjusted from
`32`-`512`, and the **grid-limit** (max grid-size) was adjusted from
`1024`-`16384`. Each config was run 5 times per graph to get a good time
measure. `4096x256` [appears] to be a good config for most graphs, except
for `soc-LiveJournal1`, `coPapersCiteseer`, and `coPapersDBLP` for which
`4096x32` appears better. For very large graphs like `indochina-2004`,
`8192x256` can also be used. Here `4096` is the *grid-limit*, and `256`
is the *block-size*.

All outputs are saved in [out/](out/), and output for `web-Stanford` and
`indochina-2004` is shown below. The input data used for this experiment is
available at ["graphs"] (for small ones), and the [SuiteSparse Matrix Collection].

```bash
$ nvcc -std=c++17 -Xcompiler -lnvgraph -O3 main.cu
$ ./a.out ~/data/web-Stanford.mtx

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
```

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -lnvgraph -O3 main.cu
$ ./a.out ~/data/indochina-2004.mtx

# Loading graph /home/subhajit/data/indochina-2004.mtx ...
# order: 7414866 size: 194109311 {}
# order: 7414866 size: 194109311 {} (transposeWithDegree)
# [00193.619 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00804.260 ms; 061 iters.] [9.7505e-06 err.] pagerankCuda<<<1024, 32>>>
# [00777.558 ms; 061 iters.] [9.7240e-06 err.] pagerankCuda<<<1024, 64>>>
# [00777.896 ms; 061 iters.] [9.5435e-06 err.] pagerankCuda<<<1024, 128>>>
# [00793.261 ms; 061 iters.] [9.5255e-06 err.] pagerankCuda<<<1024, 256>>>
# [00790.327 ms; 061 iters.] [9.4585e-06 err.] pagerankCuda<<<1024, 512>>>
# [00747.702 ms; 061 iters.] [9.5648e-06 err.] pagerankCuda<<<2048, 32>>>
# [00764.496 ms; 061 iters.] [9.8851e-06 err.] pagerankCuda<<<2048, 64>>>
# [00787.835 ms; 061 iters.] [9.6276e-06 err.] pagerankCuda<<<2048, 128>>>
# [00772.668 ms; 061 iters.] [9.6612e-06 err.] pagerankCuda<<<2048, 256>>>
# [00741.835 ms; 061 iters.] [9.7760e-06 err.] pagerankCuda<<<2048, 512>>>
# [00774.053 ms; 061 iters.] [9.6243e-06 err.] pagerankCuda<<<4096, 32>>>
# [00787.744 ms; 061 iters.] [9.4354e-06 err.] pagerankCuda<<<4096, 64>>>
# [00778.210 ms; 061 iters.] [9.8062e-06 err.] pagerankCuda<<<4096, 128>>>
# [00739.506 ms; 061 iters.] [9.7345e-06 err.] pagerankCuda<<<4096, 256>>>
# [00733.578 ms; 061 iters.] [9.7798e-06 err.] pagerankCuda<<<4096, 512>>>
# [00782.130 ms; 061 iters.] [9.5066e-06 err.] pagerankCuda<<<8192, 32>>>
# [00771.563 ms; 061 iters.] [9.5426e-06 err.] pagerankCuda<<<8192, 64>>>
# [00736.079 ms; 061 iters.] [1.0157e-05 err.] pagerankCuda<<<8192, 128>>>
# [00729.060 ms; 061 iters.] [9.8827e-06 err.] pagerankCuda<<<8192, 256>>>
# [00737.440 ms; 061 iters.] [9.5345e-06 err.] pagerankCuda<<<8192, 512>>>
# [00758.698 ms; 061 iters.] [9.2956e-06 err.] pagerankCuda<<<16384, 32>>>
# [00736.683 ms; 061 iters.] [9.8977e-06 err.] pagerankCuda<<<16384, 64>>>
# [00726.209 ms; 061 iters.] [9.2723e-06 err.] pagerankCuda<<<16384, 128>>>
# [00732.802 ms; 061 iters.] [9.2802e-06 err.] pagerankCuda<<<16384, 256>>>
# [00742.255 ms; 061 iters.] [9.3601e-06 err.] pagerankCuda<<<16384, 512>>>
```

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/8ehYEOB.jpg)](https://www.youtube.com/watch?v=LbndPOzyPXE)

[appears]: https://docs.google.com/spreadsheets/d/16viria4blm3e4AsF0iaPk03i_OXCFN8optcrOPwbCJ8/edit?usp=sharing
[nvGraph]: https://github.com/rapidsai/nvgraph
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[csr]: https://github.com/puzzlef/pagerank-class-vs-csr
["graphs"]: https://github.com/puzzlef/graphs
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
