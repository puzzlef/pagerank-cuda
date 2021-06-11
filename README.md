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
```

[![](https://i.imgur.com/lOHnic2.gif)][sheets]

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
