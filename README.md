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
