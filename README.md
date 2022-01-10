Switched thread/block-per-vertex CUDA-based PageRank (PR) algorithm ([pull], [CSR]).

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets]. The input
data used for this experiment is available at the [SuiteSparse Matrix
Collection].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out ~/data/web-Stanford.mtx
$ ./a.out ~/data/web-BerkStan.mtx
$ ...

# Loading graph /home/subhajit/data/web-Stanford.mtx ...
# order: 281903 size: 2312497 {}
# order: 281903 size: 2312497 {} (transposeWithDegree)
# [00011.300 ms; 063 iters.] [0.0000e+00 err.] pagerankCuda
#
# ...
#
# Loading graph /home/subhajit/data/soc-LiveJournal1.mtx ...
# order: 4847571 size: 68993773 {}
# order: 4847571 size: 68993773 {} (transposeWithDegree)
# [00157.767 ms; 051 iters.] [0.0000e+00 err.] pagerankCuda
#
# ...
```

[![](https://i.imgur.com/42hbeHL.png)][sheetp]
[![](https://i.imgur.com/5gHlECC.png)][sheetp]
[![](https://i.imgur.com/cBcC75q.png)][sheetp]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](https://www.youtube.com/watch?v=ke9g8hB0MEo)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://gist.github.com/wolfram77/72c51e494eaaea1c21a9c4021ad0f320)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[CSR]: https://github.com/puzzlef/pagerank-class-vs-csr
[charts]: https://photos.app.goo.gl/6gvHFBbuN9jwEPSw7
[sheets]: https://docs.google.com/spreadsheets/d/1hxCQrWodd_GGAR8KCe4HjKmdj9NobzKoH8qJjqKQqcY/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vSuhEybARymE9SDn2V_AcUHWYQbKvnWsCttbS-H9pFIvp7_sEoKgg95fDs4zALvcKkU36alE55uug4J/pubhtml
