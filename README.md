Comparing the effect of using different **per-iteration rank scaling**,
with **CUDA-based PageRank** ([pull], [CSR]).

[nvGraph PageRank] appears to use [L2-norm per-iteration scaling]. This
is (probably) required for finding a solution to **eigenvalue problem**.
However, as the *eigenvalue* for PageRank is `1`, this is not necessary.
This experiement was for observing if this was indeed true, and that
any such *per-iteration scaling* doesn't affect the number of *iterations*
needed to converge.

PageRank was computed with **L1**, **L2**, or **L∞-norm** and the
effect of **L1** or **L2-norm** *scaling of ranks* was compared with
**baseline (L0)**. All *seventeen* graphs used in this experiment are
stored in the *MatrixMarket (.mtx)* file format, and obtained from the
*SuiteSparse* *Matrix Collection*. These include: *web-Stanford, web-BerkStan,*
*web-Google, web-NotreDame, soc-Slashdot0811, soc-Slashdot0902,*
*soc-Epinions1, coAuthorsDBLP, coAuthorsCiteseer, soc-LiveJournal1,*
*coPapersCiteseer, coPapersDBLP, indochina-2004, italy_osm,*
*great-britain_osm, germany_osm, asia_osm*. The experiment is implemented
in *C++*, and compiled using *GCC 9* with *optimization level 3 (-O3)*.
The *iterations* taken with each test case is measured. `500` is the
*maximum iterations* allowed. Statistics of each test case is
printed to *standard output (stdout)*, and redirected to a *log file*,
which is then processed with a *script* to generate a *CSV file*, with
each *row* representing the details of a *single test case*.

Results match the above assumptions, and indeed no performance benefit
is observed (except a reduction in a single iteration for *soc-Slashdot0811*,
*soc-Slashdot-0902*, *soc-LiveJournal1*, and *italy_osm* graphs).

All outputs are saved in [out](out/) and a small part of the output is listed
here. The input data used for this experiment is available at ["graphs"] (for
small ones), and the [SuiteSparse Matrix Collection].

<br>

```bash
$ g++ -O3 main.cxx
$ ./a.out ~/data/min-1DeadEnd.mtx
$ ./a.out ~/data/min-2SCC.mtx
$ ...

# ...
#
# Loading graph /home/subhajit/data/soc-Slashdot0811.mtx ...
# order: 77360 size: 905468 {}
# order: 77360 size: 905468 {} (transposeWithDegree)
# [00006.911 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00003.584 ms; 055 iters.] [9.0883e-07 err.] pagerankCudaL1Norm [iteration-scaling=L0]
# [00101.015 ms; 055 iters.] [9.8525e-07 err.] pagerankSeqL1Norm [iteration-scaling=L0]
# [00004.657 ms; 054 iters.] [1.8232e-06 err.] pagerankCudaL1Norm [iteration-scaling=L1]
# [00109.816 ms; 055 iters.] [9.4585e-07 err.] pagerankSeqL1Norm [iteration-scaling=L1]
# [00042.708 ms; 500 iters.] [7.4785e+01 err.] pagerankCudaL1Norm [iteration-scaling=L2]
# [00989.602 ms; 500 iters.] [7.4784e+01 err.] pagerankSeqL1Norm [iteration-scaling=L2]
# [00001.844 ms; 029 iters.] [4.2798e-04 err.] pagerankCudaL2Norm [iteration-scaling=L0]
# [00053.427 ms; 029 iters.] [4.2798e-04 err.] pagerankSeqL2Norm [iteration-scaling=L0]
# [00002.456 ms; 029 iters.] [4.2805e-04 err.] pagerankCudaL2Norm [iteration-scaling=L1]
# [00057.631 ms; 029 iters.] [4.2797e-04 err.] pagerankSeqL2Norm [iteration-scaling=L1]
# [00041.511 ms; 500 iters.] [7.4785e+01 err.] pagerankCudaL2Norm [iteration-scaling=L2]
# [00993.422 ms; 500 iters.] [7.4784e+01 err.] pagerankSeqL2Norm [iteration-scaling=L2]
# [00001.289 ms; 020 iters.] [1.9676e-03 err.] pagerankCudaLiNorm [iteration-scaling=L0]
# [00039.009 ms; 020 iters.] [1.9677e-03 err.] pagerankSeqLiNorm [iteration-scaling=L0]
# [00001.707 ms; 020 iters.] [1.9675e-03 err.] pagerankCudaLiNorm [iteration-scaling=L1]
# [00042.109 ms; 020 iters.] [1.9676e-03 err.] pagerankSeqLiNorm [iteration-scaling=L1]
# [00041.756 ms; 500 iters.] [7.4785e+01 err.] pagerankCudaLiNorm [iteration-scaling=L2]
# [01046.375 ms; 500 iters.] [7.4784e+01 err.] pagerankSeqLiNorm [iteration-scaling=L2]
#
# Loading graph /home/subhajit/data/soc-Slashdot0902.mtx ...
# order: 82168 size: 948464 {}
# order: 82168 size: 948464 {} (transposeWithDegree)
# [00006.898 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00003.498 ms; 056 iters.] [8.5459e-07 err.] pagerankCudaL1Norm [iteration-scaling=L0]
# [00111.433 ms; 056 iters.] [4.4357e-06 err.] pagerankSeqL1Norm [iteration-scaling=L0]
# [00004.581 ms; 055 iters.] [1.9838e-06 err.] pagerankCudaL1Norm [iteration-scaling=L1]
# [00122.162 ms; 056 iters.] [1.2052e-06 err.] pagerankSeqL1Norm [iteration-scaling=L1]
# [00041.337 ms; 500 iters.] [7.8248e+01 err.] pagerankCudaL1Norm [iteration-scaling=L2]
# [01083.655 ms; 500 iters.] [7.8247e+01 err.] pagerankSeqL1Norm [iteration-scaling=L2]
# [00001.832 ms; 029 iters.] [5.0485e-04 err.] pagerankCudaL2Norm [iteration-scaling=L0]
# [00057.329 ms; 029 iters.] [5.0161e-04 err.] pagerankSeqL2Norm [iteration-scaling=L0]
# [00002.439 ms; 029 iters.] [5.0514e-04 err.] pagerankCudaL2Norm [iteration-scaling=L1]
# [00063.226 ms; 029 iters.] [5.0461e-04 err.] pagerankSeqL2Norm [iteration-scaling=L1]
# [00041.209 ms; 500 iters.] [7.8248e+01 err.] pagerankCudaL2Norm [iteration-scaling=L2]
# [01084.506 ms; 500 iters.] [7.8247e+01 err.] pagerankSeqL2Norm [iteration-scaling=L2]
# [00001.352 ms; 021 iters.] [1.9714e-03 err.] pagerankCudaLiNorm [iteration-scaling=L0]
# [00045.283 ms; 021 iters.] [1.9696e-03 err.] pagerankSeqLiNorm [iteration-scaling=L0]
# [00001.779 ms; 021 iters.] [1.9716e-03 err.] pagerankCudaLiNorm [iteration-scaling=L1]
# [00048.552 ms; 021 iters.] [1.9713e-03 err.] pagerankSeqLiNorm [iteration-scaling=L1]
# [00041.432 ms; 500 iters.] [7.8248e+01 err.] pagerankCudaLiNorm [iteration-scaling=L2]
# [01149.528 ms; 500 iters.] [7.8247e+01 err.] pagerankSeqLiNorm [iteration-scaling=L2]
#
# ...
```

<br>
<br>


## References

- [RAPIDS nvGraph](https://github.com/rapidsai/nvgraph)
- [Deeper Inside PageRank :: Amy N. Langville, Carl D. Meyer](https://www.slideshare.net/SubhajitSahu/deeper-inside-pagerank-notes)
- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](https://www.youtube.com/watch?v=ke9g8hB0MEo)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/BnCiig7.jpg)](https://www.youtube.com/watch?v=IJTvialxf_8)

[nvGraph PageRank]: https://github.com/rapidsai/nvgraph/blob/main/cpp/src/pagerank.cu
[L2-norm per-iteration scaling]: https://github.com/rapidsai/nvgraph/blob/main/cpp/src/pagerank.cu#L145
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
["graphs"]: https://github.com/puzzlef/graphs
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[CSR]: https://github.com/puzzlef/pagerank-class-vs-csr
