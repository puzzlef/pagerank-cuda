Performance of sequential execution based vs OpenMP based PageRank ([pull], [CSR]).

This experiment was for comparing the performance between:
1. Find pagerank using a single thread (**sequential**).
2. Find pagerank accelerated using **OpenMP**.

Both techniques were attempted on different types of graphs, running each
technique 5 times per graph to get a good time measure. **OpenMP** does seem
to provide a clear benefit for most graphs (except the smallest ones). This
speedup is definitely not directly proportional to the number of threads, as
one would normally expect (Amdahl's law). Note that there is still room for
improvement with **OpenMP** by using sequential versions of certain routines
instead of OpenMP versions because not all calculations benefit from multiple
threads (ex. ["multiply-sequential-vs-openmp"]).

Number of threads for this experiment (using `OMP_NUM_THREADS`) was varied
from `2` to `48`. All outputs are saved in [out/](out/) and outputs for `4`,
`48` threads are listed here. The input data used for this experiment is
available at ["graphs"] (for small ones), and the [SuiteSparse Matrix Collection].

<br>

```bash
$ g++ -O3 main.cxx
$ export OMP_NUM_THREADS=4
$ ./a.out ~/data/min-1DeadEnd.mtx
$ ./a.out ~/data/min-2SCC.mtx
$ ...

# ...
```

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/5vdxPZ3.jpg)](https://www.youtube.com/watch?v=rKv_l1RnSqs)

[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[CSR]: https://github.com/puzzlef/pagerank-class-vs-csr
["multiply-sequential-vs-openmp"]: https://github.com/puzzlef/multiply-sequential-vs-openmp
["graphs"]: https://github.com/puzzlef/graphs
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
