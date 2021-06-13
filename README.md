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
# [00011.364 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00027.595 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda [degree=2; limit=1]
# [00026.095 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda [degree=2; limit=2]
# [00025.927 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda [degree=2; limit=4]
# [00025.961 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda [degree=2; limit=8]
# [00025.879 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda [degree=2; limit=16]
# [00025.896 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda [degree=2; limit=32]
# [00025.945 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda [degree=2; limit=64]
# [00025.916 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda [degree=2; limit=128]
# [00025.934 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda [degree=2; limit=256]
# [00025.954 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda [degree=2; limit=512]
# [00025.947 ms; 063 iters.] [7.1483e-07 err.] pagerankCuda [degree=2; limit=1024]
# [00026.654 ms; 063 iters.] [7.1593e-07 err.] pagerankCuda [degree=4; limit=1]
# [00026.688 ms; 063 iters.] [7.1593e-07 err.] pagerankCuda [degree=4; limit=2]
# [00026.649 ms; 063 iters.] [7.1593e-07 err.] pagerankCuda [degree=4; limit=4]
# [00026.639 ms; 063 iters.] [7.1593e-07 err.] pagerankCuda [degree=4; limit=8]
# [00026.663 ms; 063 iters.] [7.1593e-07 err.] pagerankCuda [degree=4; limit=16]
# [00026.594 ms; 063 iters.] [7.1593e-07 err.] pagerankCuda [degree=4; limit=32]
# [00026.619 ms; 063 iters.] [7.1593e-07 err.] pagerankCuda [degree=4; limit=64]
# [00026.627 ms; 063 iters.] [7.1593e-07 err.] pagerankCuda [degree=4; limit=128]
# [00026.593 ms; 063 iters.] [7.1593e-07 err.] pagerankCuda [degree=4; limit=256]
# [00026.655 ms; 063 iters.] [7.1593e-07 err.] pagerankCuda [degree=4; limit=512]
# [00026.677 ms; 063 iters.] [7.1593e-07 err.] pagerankCuda [degree=4; limit=1024]
# [00023.966 ms; 063 iters.] [7.1677e-07 err.] pagerankCuda [degree=8; limit=1]
# [00023.966 ms; 063 iters.] [7.1677e-07 err.] pagerankCuda [degree=8; limit=2]
# [00023.940 ms; 063 iters.] [7.1677e-07 err.] pagerankCuda [degree=8; limit=4]
# [00023.995 ms; 063 iters.] [7.1677e-07 err.] pagerankCuda [degree=8; limit=8]
# [00023.986 ms; 063 iters.] [7.1677e-07 err.] pagerankCuda [degree=8; limit=16]
# [00023.944 ms; 063 iters.] [7.1677e-07 err.] pagerankCuda [degree=8; limit=32]
# [00023.954 ms; 063 iters.] [7.1677e-07 err.] pagerankCuda [degree=8; limit=64]
# [00023.965 ms; 063 iters.] [7.1677e-07 err.] pagerankCuda [degree=8; limit=128]
# [00023.903 ms; 063 iters.] [7.1677e-07 err.] pagerankCuda [degree=8; limit=256]
# [00023.975 ms; 063 iters.] [7.1677e-07 err.] pagerankCuda [degree=8; limit=512]
# [00024.208 ms; 063 iters.] [7.1677e-07 err.] pagerankCuda [degree=8; limit=1024]
# [00023.770 ms; 063 iters.] [7.1813e-07 err.] pagerankCuda [degree=16; limit=1]
# [00023.777 ms; 063 iters.] [7.1813e-07 err.] pagerankCuda [degree=16; limit=2]
# [00023.780 ms; 063 iters.] [7.1813e-07 err.] pagerankCuda [degree=16; limit=4]
# [00023.778 ms; 063 iters.] [7.1813e-07 err.] pagerankCuda [degree=16; limit=8]
# [00023.791 ms; 063 iters.] [7.1813e-07 err.] pagerankCuda [degree=16; limit=16]
# [00023.739 ms; 063 iters.] [7.1813e-07 err.] pagerankCuda [degree=16; limit=32]
# [00023.725 ms; 063 iters.] [7.1813e-07 err.] pagerankCuda [degree=16; limit=64]
# [00023.747 ms; 063 iters.] [7.1813e-07 err.] pagerankCuda [degree=16; limit=128]
# [00023.730 ms; 063 iters.] [7.1813e-07 err.] pagerankCuda [degree=16; limit=256]
# [00023.790 ms; 063 iters.] [7.1813e-07 err.] pagerankCuda [degree=16; limit=512]
# [00023.766 ms; 063 iters.] [7.1813e-07 err.] pagerankCuda [degree=16; limit=1024]
# [00024.701 ms; 063 iters.] [7.2183e-07 err.] pagerankCuda [degree=32; limit=1]
# [00024.673 ms; 063 iters.] [7.2183e-07 err.] pagerankCuda [degree=32; limit=2]
# [00024.689 ms; 063 iters.] [7.2183e-07 err.] pagerankCuda [degree=32; limit=4]
# [00024.691 ms; 063 iters.] [7.2183e-07 err.] pagerankCuda [degree=32; limit=8]
# [00024.664 ms; 063 iters.] [7.2183e-07 err.] pagerankCuda [degree=32; limit=16]
# [00024.676 ms; 063 iters.] [7.2183e-07 err.] pagerankCuda [degree=32; limit=32]
# [00024.680 ms; 063 iters.] [7.2183e-07 err.] pagerankCuda [degree=32; limit=64]
# [00024.684 ms; 063 iters.] [7.2183e-07 err.] pagerankCuda [degree=32; limit=128]
# [00024.681 ms; 063 iters.] [7.2183e-07 err.] pagerankCuda [degree=32; limit=256]
# [00024.739 ms; 063 iters.] [7.2183e-07 err.] pagerankCuda [degree=32; limit=512]
# [00024.659 ms; 063 iters.] [7.2183e-07 err.] pagerankCuda [degree=32; limit=1024]
# [00023.101 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda [degree=64; limit=1]
# [00023.078 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda [degree=64; limit=2]
# [00023.076 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda [degree=64; limit=4]
# [00023.070 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda [degree=64; limit=8]
# [00023.047 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda [degree=64; limit=16]
# [00023.055 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda [degree=64; limit=32]
# [00023.046 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda [degree=64; limit=64]
# [00023.071 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda [degree=64; limit=128]
# [00023.075 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda [degree=64; limit=256]
# [00023.071 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda [degree=64; limit=512]
# [00023.047 ms; 063 iters.] [7.3103e-07 err.] pagerankCuda [degree=64; limit=1024]
# [00023.858 ms; 063 iters.] [7.6088e-07 err.] pagerankCuda [degree=128; limit=1]
# [00024.118 ms; 063 iters.] [7.6088e-07 err.] pagerankCuda [degree=128; limit=2]
# [00023.876 ms; 063 iters.] [7.6088e-07 err.] pagerankCuda [degree=128; limit=4]
# [00023.853 ms; 063 iters.] [7.6088e-07 err.] pagerankCuda [degree=128; limit=8]
# [00023.897 ms; 063 iters.] [7.6088e-07 err.] pagerankCuda [degree=128; limit=16]
# [00023.855 ms; 063 iters.] [7.6088e-07 err.] pagerankCuda [degree=128; limit=32]
# [00023.903 ms; 063 iters.] [7.6088e-07 err.] pagerankCuda [degree=128; limit=64]
# [00023.876 ms; 063 iters.] [7.6088e-07 err.] pagerankCuda [degree=128; limit=128]
# [00023.887 ms; 063 iters.] [7.6088e-07 err.] pagerankCuda [degree=128; limit=256]
# [00023.871 ms; 063 iters.] [7.6088e-07 err.] pagerankCuda [degree=128; limit=512]
# [00023.894 ms; 063 iters.] [7.6088e-07 err.] pagerankCuda [degree=128; limit=1024]
# [00023.572 ms; 063 iters.] [7.8018e-07 err.] pagerankCuda [degree=256; limit=1]
# [00023.588 ms; 063 iters.] [7.8018e-07 err.] pagerankCuda [degree=256; limit=2]
# [00023.575 ms; 063 iters.] [7.8018e-07 err.] pagerankCuda [degree=256; limit=4]
# [00023.590 ms; 063 iters.] [7.8018e-07 err.] pagerankCuda [degree=256; limit=8]
# [00023.593 ms; 063 iters.] [7.8018e-07 err.] pagerankCuda [degree=256; limit=16]
# [00023.558 ms; 063 iters.] [7.8018e-07 err.] pagerankCuda [degree=256; limit=32]
# [00023.565 ms; 063 iters.] [7.8018e-07 err.] pagerankCuda [degree=256; limit=64]
# [00023.565 ms; 063 iters.] [7.8018e-07 err.] pagerankCuda [degree=256; limit=128]
# [00023.592 ms; 063 iters.] [7.8018e-07 err.] pagerankCuda [degree=256; limit=256]
# [00023.565 ms; 063 iters.] [7.8018e-07 err.] pagerankCuda [degree=256; limit=512]
# [00117.923 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda [degree=256; limit=1024]
# [00025.107 ms; 063 iters.] [9.2778e-07 err.] pagerankCuda [degree=512; limit=1]
# [00025.128 ms; 063 iters.] [9.2778e-07 err.] pagerankCuda [degree=512; limit=2]
# [00025.110 ms; 063 iters.] [9.2778e-07 err.] pagerankCuda [degree=512; limit=4]
# [00025.110 ms; 063 iters.] [9.2778e-07 err.] pagerankCuda [degree=512; limit=8]
# [00025.128 ms; 063 iters.] [9.2778e-07 err.] pagerankCuda [degree=512; limit=16]
# [00025.114 ms; 063 iters.] [9.2778e-07 err.] pagerankCuda [degree=512; limit=32]
# [00025.133 ms; 063 iters.] [9.2778e-07 err.] pagerankCuda [degree=512; limit=64]
# [00025.112 ms; 063 iters.] [9.2778e-07 err.] pagerankCuda [degree=512; limit=128]
# [00025.116 ms; 063 iters.] [9.2778e-07 err.] pagerankCuda [degree=512; limit=256]
# [00121.039 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda [degree=512; limit=512]
# [00121.513 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda [degree=512; limit=1024]
# [00026.246 ms; 063 iters.] [1.0596e-06 err.] pagerankCuda [degree=1024; limit=1]
# [00026.232 ms; 063 iters.] [1.0596e-06 err.] pagerankCuda [degree=1024; limit=2]
# [00026.225 ms; 063 iters.] [1.0596e-06 err.] pagerankCuda [degree=1024; limit=4]
# [00026.233 ms; 063 iters.] [1.0596e-06 err.] pagerankCuda [degree=1024; limit=8]
# [00026.234 ms; 063 iters.] [1.0596e-06 err.] pagerankCuda [degree=1024; limit=16]
# [00026.235 ms; 063 iters.] [1.0596e-06 err.] pagerankCuda [degree=1024; limit=32]
# [00026.231 ms; 063 iters.] [1.0596e-06 err.] pagerankCuda [degree=1024; limit=64]
# [00026.222 ms; 063 iters.] [1.0596e-06 err.] pagerankCuda [degree=1024; limit=128]
# [00126.255 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda [degree=1024; limit=256]
# [00126.344 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda [degree=1024; limit=512]
# [00126.223 ms; 063 iters.] [4.9303e-06 err.] pagerankCuda [degree=1024; limit=1024]
#
# ...
```

[![](https://i.imgur.com/CzE33L3.gif)][sheets]
[![](https://i.imgur.com/LfwTsKA.gif)][sheets]
[![](https://i.imgur.com/hnzcjjP.gif)][sheets]
[![](https://i.imgur.com/aJIeelH.gif)][sheets]
[![](https://i.imgur.com/TiKRMFU.gif)][sheets]
[![](https://i.imgur.com/sJ7nRLX.gif)][sheets]
[![](https://i.imgur.com/Z58cLk1.gif)][sheets]
[![](https://i.imgur.com/WbB8X99.gif)][sheets]
[![](https://i.imgur.com/Qz4MaQu.gif)][sheets]
[![](https://i.imgur.com/WGhdeCy.gif)][sheets]
[![](https://i.imgur.com/Z8fwD1m.gif)][sheets]
[![](https://i.imgur.com/51OGaWq.gif)][sheets]
[![](https://i.imgur.com/Xd9byhu.gif)][sheets]
[![](https://i.imgur.com/MOOBk46.gif)][sheets]
[![](https://i.imgur.com/edjSyiU.gif)][sheets]
[![](https://i.imgur.com/WWS1N4M.gif)][sheets]
[![](https://i.imgur.com/zdXhaKj.gif)][sheets]

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
