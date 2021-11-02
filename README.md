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
# Loading graph /kaggle/input/graphs/soc-Slashdot0811.mtx ...
# order: 77360 size: 905468 {}
# order: 77360 size: 905468 {} (transposeWithDegree)
# [00007.217 ms; 054 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00006.105 ms; 055 iters.] [9.1448e-07 err.] pagerankCudaL1Norm [iteration-scaling=L0]
# [00089.349 ms; 055 iters.] [8.8146e-07 err.] pagerankSeqL1Norm [iteration-scaling=L0]
# [00007.184 ms; 054 iters.] [0.0000e+00 err.] pagerankCudaL1Norm [iteration-scaling=L1]
# [00097.146 ms; 055 iters.] [9.1883e-07 err.] pagerankSeqL1Norm [iteration-scaling=L1]
# [00064.125 ms; 500 iters.] [7.4785e+01 err.] pagerankCudaL1Norm [iteration-scaling=L2]
# [00928.749 ms; 500 iters.] [7.4784e+01 err.] pagerankSeqL1Norm [iteration-scaling=L2]
# [00004.207 ms; 029 iters.] [4.2616e-04 err.] pagerankCudaL2Norm [iteration-scaling=L0]
# [00052.387 ms; 029 iters.] [4.2617e-04 err.] pagerankSeqL2Norm [iteration-scaling=L0]
# [00003.714 ms; 029 iters.] [4.2623e-04 err.] pagerankCudaL2Norm [iteration-scaling=L1]
# [00050.589 ms; 029 iters.] [4.2616e-04 err.] pagerankSeqL2Norm [iteration-scaling=L1]
# [00062.272 ms; 500 iters.] [7.4785e+01 err.] pagerankCudaL2Norm [iteration-scaling=L2]
# [00897.315 ms; 500 iters.] [7.4784e+01 err.] pagerankSeqL2Norm [iteration-scaling=L2]
# [00002.147 ms; 020 iters.] [1.9658e-03 err.] pagerankCudaLiNorm [iteration-scaling=L0]
# [00044.195 ms; 020 iters.] [1.9659e-03 err.] pagerankSeqLiNorm [iteration-scaling=L0]
# [00002.558 ms; 020 iters.] [1.9657e-03 err.] pagerankCudaLiNorm [iteration-scaling=L1]
# [00047.898 ms; 020 iters.] [1.9658e-03 err.] pagerankSeqLiNorm [iteration-scaling=L1]
# [00063.723 ms; 500 iters.] [7.4785e+01 err.] pagerankCudaLiNorm [iteration-scaling=L2]
# [01218.959 ms; 500 iters.] [7.4784e+01 err.] pagerankSeqLiNorm [iteration-scaling=L2]
#
# Loading graph /kaggle/input/graphs/soc-Slashdot0902.mtx ...
# order: 82168 size: 948464 {}
# order: 82168 size: 948464 {} (transposeWithDegree)
# [00007.058 ms; 055 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00005.985 ms; 056 iters.] [1.1293e-06 err.] pagerankCudaL1Norm [iteration-scaling=L0]
# [00098.479 ms; 056 iters.] [4.8344e-06 err.] pagerankSeqL1Norm [iteration-scaling=L0]
# [00006.933 ms; 055 iters.] [0.0000e+00 err.] pagerankCudaL1Norm [iteration-scaling=L1]
# [00108.133 ms; 056 iters.] [1.4502e-06 err.] pagerankSeqL1Norm [iteration-scaling=L1]
# [00062.889 ms; 500 iters.] [7.8248e+01 err.] pagerankCudaL1Norm [iteration-scaling=L2]
# [01006.808 ms; 500 iters.] [7.8247e+01 err.] pagerankSeqL1Norm [iteration-scaling=L2]
# [00003.133 ms; 029 iters.] [5.0287e-04 err.] pagerankCudaL2Norm [iteration-scaling=L0]
# [00058.954 ms; 029 iters.] [4.9963e-04 err.] pagerankSeqL2Norm [iteration-scaling=L0]
# [00004.060 ms; 029 iters.] [5.0315e-04 err.] pagerankCudaL2Norm [iteration-scaling=L1]
# [00074.190 ms; 029 iters.] [5.0262e-04 err.] pagerankSeqL2Norm [iteration-scaling=L1]
# [00070.101 ms; 500 iters.] [7.8248e+01 err.] pagerankCudaL2Norm [iteration-scaling=L2]
# [00949.488 ms; 500 iters.] [7.8247e+01 err.] pagerankSeqL2Norm [iteration-scaling=L2]
# [00002.301 ms; 021 iters.] [1.9694e-03 err.] pagerankCudaLiNorm [iteration-scaling=L0]
# [00050.244 ms; 021 iters.] [1.9676e-03 err.] pagerankSeqLiNorm [iteration-scaling=L0]
# [00002.745 ms; 021 iters.] [1.9696e-03 err.] pagerankCudaLiNorm [iteration-scaling=L1]
# [00055.429 ms; 021 iters.] [1.9693e-03 err.] pagerankSeqLiNorm [iteration-scaling=L1]
# [00064.156 ms; 500 iters.] [7.8248e+01 err.] pagerankCudaLiNorm [iteration-scaling=L2]
# [01312.879 ms; 500 iters.] [7.8247e+01 err.] pagerankSeqLiNorm [iteration-scaling=L2]
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
