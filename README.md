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
