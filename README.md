Comparing various switch points for CUDA **switched-per-vertex** based
PageRank ([pull], [CSR], [switched-partition]).

For this experiment, `switch_degree` was varied from `2` - `1024`, and
`switch_limit` was varied from `1` - `1024`. `switch_degree` defines the
*in-degree* at which *pagerank kernel* switches from **thread-per-vertex**
approach to **block-per-vertex**. `switch_limit` defines the minimum block
size for **thread-per-vertex** / **block-per-vertex** approach (if a block
size is too small, it is merged with the other approach block). Each case is
run on multiple graphs, running each 5 times per graph for good time measure.
It seems `switch_degree` of **64** and `switch_limit` of **32** would be a
good choice.

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
# [00011.336 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00042.836 ms; 063 iters.] [7.1387e-07 err.] pagerankCuda<<<1024, 32>>>
# [00023.820 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda<<<1024, 64>>>
# [00014.952 ms; 063 iters.] [7.1379e-07 err.] pagerankCuda<<<1024, 128>>>
# [00010.686 ms; 063 iters.] [7.0093e-07 err.] pagerankCuda<<<1024, 256>>>
# [00009.006 ms; 063 iters.] [7.0111e-07 err.] pagerankCuda<<<1024, 512>>>
# [00008.973 ms; 063 iters.] [6.9965e-07 err.] pagerankCuda<<<1024, 1024>>>
# [00040.667 ms; 063 iters.] [7.1387e-07 err.] pagerankCuda<<<2048, 32>>>
# [00023.203 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda<<<2048, 64>>>
# [00014.757 ms; 063 iters.] [7.1379e-07 err.] pagerankCuda<<<2048, 128>>>
# [00010.499 ms; 063 iters.] [7.0093e-07 err.] pagerankCuda<<<2048, 256>>>
# [00008.715 ms; 063 iters.] [7.0111e-07 err.] pagerankCuda<<<2048, 512>>>
# [00008.959 ms; 063 iters.] [6.9965e-07 err.] pagerankCuda<<<2048, 1024>>>
# [00040.502 ms; 063 iters.] [7.1387e-07 err.] pagerankCuda<<<4096, 32>>>
# [00023.416 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda<<<4096, 64>>>
# [00014.715 ms; 063 iters.] [7.1379e-07 err.] pagerankCuda<<<4096, 128>>>
# [00010.455 ms; 063 iters.] [7.0093e-07 err.] pagerankCuda<<<4096, 256>>>
# [00008.419 ms; 063 iters.] [7.0111e-07 err.] pagerankCuda<<<4096, 512>>>
# [00008.939 ms; 063 iters.] [6.9965e-07 err.] pagerankCuda<<<4096, 1024>>>
# [00040.559 ms; 063 iters.] [7.1387e-07 err.] pagerankCuda<<<8192, 32>>>
# [00023.026 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda<<<8192, 64>>>
# [00014.708 ms; 063 iters.] [7.1379e-07 err.] pagerankCuda<<<8192, 128>>>
# [00010.401 ms; 063 iters.] [7.0093e-07 err.] pagerankCuda<<<8192, 256>>>
# [00008.425 ms; 063 iters.] [7.0111e-07 err.] pagerankCuda<<<8192, 512>>>
# [00008.940 ms; 063 iters.] [6.9965e-07 err.] pagerankCuda<<<8192, 1024>>>
# [00040.528 ms; 063 iters.] [7.1387e-07 err.] pagerankCuda<<<16384, 32>>>
# [00023.055 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda<<<16384, 64>>>
# [00014.712 ms; 063 iters.] [7.1379e-07 err.] pagerankCuda<<<16384, 128>>>
# [00010.391 ms; 063 iters.] [7.0093e-07 err.] pagerankCuda<<<16384, 256>>>
# [00008.430 ms; 063 iters.] [7.0111e-07 err.] pagerankCuda<<<16384, 512>>>
# [00008.948 ms; 063 iters.] [6.9965e-07 err.] pagerankCuda<<<16384, 1024>>>
# [00040.539 ms; 063 iters.] [7.1387e-07 err.] pagerankCuda<<<32768, 32>>>
# [00023.072 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda<<<32768, 64>>>
# [00014.707 ms; 063 iters.] [7.1379e-07 err.] pagerankCuda<<<32768, 128>>>
# [00010.410 ms; 063 iters.] [7.0093e-07 err.] pagerankCuda<<<32768, 256>>>
# [00008.648 ms; 063 iters.] [7.0111e-07 err.] pagerankCuda<<<32768, 512>>>
# [00008.925 ms; 063 iters.] [6.9965e-07 err.] pagerankCuda<<<32768, 1024>>>
#
# ...
```

[![](https://i.imgur.com/CzE33L3.gif)][sheets]

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
[block-launch]: https://github.com/puzzlef/pagerank-cuda-block-adjust-launch
[thread-launch]: https://github.com/puzzlef/pagerank-cuda-thread-adjust-launch
[switched-partition]: https://github.com/puzzlef/pagerank-cuda-switched-sort-by-indegree
[charts]: https://photos.app.goo.gl/67DDHrtivnEGvXzQ7
[sheets]: https://docs.google.com/spreadsheets/d/186GuFf02uKEp2C1gQtpjenWyTTAh6IXOpLJOPxdOlPA/edit?usp=sharing
