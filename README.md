Experimenting the effect of sorting vertices by in-degree for CUDA
**block-per-vertex** based PageRank ([pull], [CSR], [block-launch]).

This experiment was for comparing the performance between:
1. Find pagerank using [nvGraph].
2. Find pagerank using *CUDA block-per-vertex*.
3. Find pagerank using *CUDA block-per-vertex*, with **vertices sorted by in-degree**.

Each approach is run on multiple graphs, running each 5 times per graph for
good time measure. For CUDA pagerank `4096x64` launch config is used (see
[block-launch]). As expected, it appears sorting vertices by in-degree has
**no consisten**t performance advantage. This is most likely because blocks
run independently. In order to measure error, [nvGraph] pagerank is taken as
a reference.

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

[![](https://i.imgur.com/zLPn59G.gif)][sheets]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/aEQNi9z.jpg)](https://www.youtube.com/watch?v=Q5hnBsUWmAI)

[nvGraph]: https://github.com/rapidsai/nvgraph
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[csr]: https://github.com/puzzlef/pagerank-class-vs-csr
[block-launch]: https://github.com/puzzlef/pagerank-cuda-block-adjust-launch
[charts]: https://photos.app.goo.gl/kCxgy62fFNjWqc8u7
[sheets]: https://docs.google.com/spreadsheets/d/1vcdBUAa_XQh3G3JVCQMSWhSZmeCffEqXz7EYN30FRZ0/edit?usp=sharing
["graphs"]: https://github.com/puzzlef/graphs
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
