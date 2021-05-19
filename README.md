Performance of sequential execution based vs OpenMP based PageRank (pull, CSR).

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
threads (for example, see ["multiply-sequential-vs-openmp"]).

Number of threads for this experiment (using `OMP_NUM_THREADS`) was varied
from `2` to `48`. All outputs are saved in [out/](out/) and outputs for `4`,
`48` threads are listed here. See ["pagerank-push-vs-pull"] for a discussion
on *push* vs *pull* method, and ["pagerank-class-vs-csr"] for a comparision
between using a C++ DiGraph class directly vs using its CSR representation.
The input data used for this experiment is available at ["graphs"] (for small
ones), and the [SuiteSparse Matrix Collection].

```bash
$ g++ -O3 main.cxx
$ export OMP_NUM_THREADS=4
$ ./a.out ~/data/min-1DeadEnd.mtx
$ ./a.out ~/data/min-2SCC.mtx
$ ...

# Loading graph /home/subhajit/data/min-1DeadEnd.mtx ...
# order: 5 size: 6 {}
# order: 5 size: 6 {} (transposeWithDegree)
# [00000.002 ms; 016 iters.] [0.0000e+00 err.] pagerankSeq
# [00000.563 ms; 016 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/min-2SCC.mtx ...
# order: 8 size: 12 {}
# order: 8 size: 12 {} (transposeWithDegree)
# [00000.004 ms; 039 iters.] [0.0000e+00 err.] pagerankSeq
# [00001.068 ms; 039 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/min-4SCC.mtx ...
# order: 21 size: 35 {}
# order: 21 size: 35 {} (transposeWithDegree)
# [00000.015 ms; 044 iters.] [0.0000e+00 err.] pagerankSeq
# [00001.387 ms; 044 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/min-NvgraphEx.mtx ...
# order: 6 size: 10 {}
# order: 6 size: 10 {} (transposeWithDegree)
# [00000.003 ms; 023 iters.] [0.0000e+00 err.] pagerankSeq
# [00000.729 ms; 023 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/web-Stanford.mtx ...
# order: 281903 size: 2312497 {}
# order: 281903 size: 2312497 {} (transposeWithDegree)
# [00404.625 ms; 062 iters.] [0.0000e+00 err.] pagerankSeq
# [00128.902 ms; 062 iters.] [2.8684e-07 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/web-BerkStan.mtx ...
# order: 685230 size: 7600595 {}
# order: 685230 size: 7600595 {} (transposeWithDegree)
# [00883.201 ms; 063 iters.] [0.0000e+00 err.] pagerankSeq
# [00278.621 ms; 063 iters.] [3.6427e-06 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/web-Google.mtx ...
# order: 916428 size: 5105039 {}
# order: 916428 size: 5105039 {} (transposeWithDegree)
# [01485.368 ms; 060 iters.] [0.0000e+00 err.] pagerankSeq
# [00448.469 ms; 061 iters.] [4.8997e-04 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/web-NotreDame.mtx ...
# order: 325729 size: 1497134 {}
# order: 325729 size: 1497134 {} (transposeWithDegree)
# [00215.877 ms; 057 iters.] [0.0000e+00 err.] pagerankSeq
# [00066.830 ms; 056 iters.] [7.9977e-04 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/soc-Slashdot0811.mtx ...
# order: 77360 size: 905468 {}
# order: 77360 size: 905468 {} (transposeWithDegree)
# [00089.451 ms; 054 iters.] [0.0000e+00 err.] pagerankSeq
# [00032.762 ms; 054 iters.] [9.1191e-08 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/soc-Slashdot0902.mtx ...
# order: 82168 size: 948464 {}
# order: 82168 size: 948464 {} (transposeWithDegree)
# [00098.628 ms; 055 iters.] [0.0000e+00 err.] pagerankSeq
# [00035.114 ms; 055 iters.] [4.4557e-06 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/soc-Epinions1.mtx ...
# order: 75888 size: 508837 {}
# order: 75888 size: 508837 {} (transposeWithDegree)
# [00080.170 ms; 053 iters.] [0.0000e+00 err.] pagerankSeq
# [00036.363 ms; 053 iters.] [2.4632e-05 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/coAuthorsDBLP.mtx ...
# order: 299067 size: 1955352 {}
# order: 299067 size: 1955352 {} (transposeWithDegree)
# [00235.572 ms; 044 iters.] [0.0000e+00 err.] pagerankSeq
# [00079.317 ms; 044 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/coAuthorsCiteseer.mtx ...
# order: 227320 size: 1628268 {}
# order: 227320 size: 1628268 {} (transposeWithDegree)
# [00190.043 ms; 047 iters.] [0.0000e+00 err.] pagerankSeq
# [00066.276 ms; 047 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/soc-LiveJournal1.mtx ...
# order: 4847571 size: 68993773 {}
# order: 4847571 size: 68993773 {} (transposeWithDegree)
# [14021.927 ms; 050 iters.] [0.0000e+00 err.] pagerankSeq
# [04175.903 ms; 050 iters.] [2.0568e-03 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/coPapersCiteseer.mtx ...
# order: 434102 size: 32073440 {}
# order: 434102 size: 32073440 {} (transposeWithDegree)
# [02152.283 ms; 050 iters.] [0.0000e+00 err.] pagerankSeq
# [00693.816 ms; 050 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/coPapersDBLP.mtx ...
# order: 540486 size: 30491458 {}
# order: 540486 size: 30491458 {} (transposeWithDegree)
# [02072.711 ms; 048 iters.] [0.0000e+00 err.] pagerankSeq
# [00670.138 ms; 048 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/indochina-2004.mtx ...
# order: 7414866 size: 194109311 {}
# order: 7414866 size: 194109311 {} (transposeWithDegree)
# [18706.791 ms; 060 iters.] [0.0000e+00 err.] pagerankSeq
# [06541.819 ms; 059 iters.] [9.0528e-04 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/italy_osm.mtx ...
# order: 6686493 size: 14027956 {}
# order: 6686493 size: 14027956 {} (transposeWithDegree)
# [04051.310 ms; 062 iters.] [0.0000e+00 err.] pagerankSeq
# [01451.414 ms; 062 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/great-britain_osm.mtx ...
# order: 7733822 size: 16313034 {}
# order: 7733822 size: 16313034 {} (transposeWithDegree)
# [06688.965 ms; 066 iters.] [0.0000e+00 err.] pagerankSeq
# [02073.560 ms; 066 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/germany_osm.mtx ...
# order: 11548845 size: 24738362 {}
# order: 11548845 size: 24738362 {} (transposeWithDegree)
# [10161.606 ms; 064 iters.] [0.0000e+00 err.] pagerankSeq
# [03396.862 ms; 064 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/asia_osm.mtx ...
# order: 11950757 size: 25423206 {}
# order: 11950757 size: 25423206 {} (transposeWithDegree)
# [09129.749 ms; 062 iters.] [0.0000e+00 err.] pagerankSeq
# [02563.671 ms; 062 iters.] [0.0000e+00 err.] pagerankOmp
```

```bash
$ g++ -O3 main.cxx
$ export OMP_NUM_THREADS=48
$ ./a.out ~/data/min-1DeadEnd.mtx
$ ./a.out ~/data/min-2SCC.mtx
$ ...

# Loading graph /home/subhajit/data/min-1DeadEnd.mtx ...
# order: 5 size: 6 {}
# order: 5 size: 6 {} (transposeWithDegree)
# [00000.002 ms; 016 iters.] [0.0000e+00 err.] pagerankSeq
# [00010.634 ms; 016 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/min-2SCC.mtx ...
# order: 8 size: 12 {}
# order: 8 size: 12 {} (transposeWithDegree)
# [00000.002 ms; 039 iters.] [0.0000e+00 err.] pagerankSeq
# [00021.616 ms; 039 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/min-4SCC.mtx ...
# order: 21 size: 35 {}
# order: 21 size: 35 {} (transposeWithDegree)
# [00000.005 ms; 044 iters.] [0.0000e+00 err.] pagerankSeq
# [00023.127 ms; 044 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/min-NvgraphEx.mtx ...
# order: 6 size: 10 {}
# order: 6 size: 10 {} (transposeWithDegree)
# [00000.001 ms; 023 iters.] [0.0000e+00 err.] pagerankSeq
# [00006.833 ms; 023 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/web-Stanford.mtx ...
# order: 281903 size: 2312497 {}
# order: 281903 size: 2312497 {} (transposeWithDegree)
# [00404.945 ms; 062 iters.] [0.0000e+00 err.] pagerankSeq
# [00050.575 ms; 062 iters.] [2.1697e-08 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/web-BerkStan.mtx ...
# order: 685230 size: 7600595 {}
# order: 685230 size: 7600595 {} (transposeWithDegree)
# [00886.528 ms; 063 iters.] [0.0000e+00 err.] pagerankSeq
# [00077.391 ms; 063 iters.] [3.5557e-06 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/web-Google.mtx ...
# order: 916428 size: 5105039 {}
# order: 916428 size: 5105039 {} (transposeWithDegree)
# [01510.642 ms; 060 iters.] [0.0000e+00 err.] pagerankSeq
# [00109.203 ms; 061 iters.] [5.1625e-04 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/web-NotreDame.mtx ...
# order: 325729 size: 1497134 {}
# order: 325729 size: 1497134 {} (transposeWithDegree)
# [00216.601 ms; 057 iters.] [0.0000e+00 err.] pagerankSeq
# [00042.557 ms; 057 iters.] [7.7114e-04 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/soc-Slashdot0811.mtx ...
# order: 77360 size: 905468 {}
# order: 77360 size: 905468 {} (transposeWithDegree)
# [00089.504 ms; 054 iters.] [0.0000e+00 err.] pagerankSeq
# [00029.239 ms; 054 iters.] [1.8204e-08 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/soc-Slashdot0902.mtx ...
# order: 82168 size: 948464 {}
# order: 82168 size: 948464 {} (transposeWithDegree)
# [00100.103 ms; 055 iters.] [0.0000e+00 err.] pagerankSeq
# [00040.314 ms; 055 iters.] [4.5224e-06 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/soc-Epinions1.mtx ...
# order: 75888 size: 508837 {}
# order: 75888 size: 508837 {} (transposeWithDegree)
# [00081.068 ms; 053 iters.] [0.0000e+00 err.] pagerankSeq
# [00048.388 ms; 053 iters.] [2.6388e-05 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/coAuthorsDBLP.mtx ...
# order: 299067 size: 1955352 {}
# order: 299067 size: 1955352 {} (transposeWithDegree)
# [00238.843 ms; 044 iters.] [0.0000e+00 err.] pagerankSeq
# [00041.123 ms; 044 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/coAuthorsCiteseer.mtx ...
# order: 227320 size: 1628268 {}
# order: 227320 size: 1628268 {} (transposeWithDegree)
# [00192.881 ms; 047 iters.] [0.0000e+00 err.] pagerankSeq
# [00035.379 ms; 047 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/soc-LiveJournal1.mtx ...
# order: 4847571 size: 68993773 {}
# order: 4847571 size: 68993773 {} (transposeWithDegree)
# [11737.315 ms; 050 iters.] [0.0000e+00 err.] pagerankSeq
# [00847.098 ms; 050 iters.] [2.0586e-03 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/coPapersCiteseer.mtx ...
# order: 434102 size: 32073440 {}
# order: 434102 size: 32073440 {} (transposeWithDegree)
# [02160.375 ms; 050 iters.] [0.0000e+00 err.] pagerankSeq
# [00220.652 ms; 050 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/coPapersDBLP.mtx ...
# order: 540486 size: 30491458 {}
# order: 540486 size: 30491458 {} (transposeWithDegree)
# [02050.854 ms; 048 iters.] [0.0000e+00 err.] pagerankSeq
# [00355.864 ms; 048 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/indochina-2004.mtx ...
# order: 7414866 size: 194109311 {}
# order: 7414866 size: 194109311 {} (transposeWithDegree)
# [18718.287 ms; 060 iters.] [0.0000e+00 err.] pagerankSeq
# [03765.799 ms; 060 iters.] [7.8820e-04 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/italy_osm.mtx ...
# order: 6686493 size: 14027956 {}
# order: 6686493 size: 14027956 {} (transposeWithDegree)
# [04047.122 ms; 062 iters.] [0.0000e+00 err.] pagerankSeq
# [00581.116 ms; 062 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/great-britain_osm.mtx ...
# order: 7733822 size: 16313034 {}
# order: 7733822 size: 16313034 {} (transposeWithDegree)
# [06801.301 ms; 066 iters.] [0.0000e+00 err.] pagerankSeq
# [00666.433 ms; 066 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/germany_osm.mtx ...
# order: 11548845 size: 24738362 {}
# order: 11548845 size: 24738362 {} (transposeWithDegree)
# [09918.595 ms; 064 iters.] [0.0000e+00 err.] pagerankSeq
# [00888.095 ms; 064 iters.] [0.0000e+00 err.] pagerankOmp
#
# Loading graph /home/subhajit/data/asia_osm.mtx ...
# order: 11950757 size: 25423206 {}
# order: 11950757 size: 25423206 {} (transposeWithDegree)
# [07575.640 ms; 062 iters.] [0.0000e+00 err.] pagerankSeq
# [00852.245 ms; 062 iters.] [0.0000e+00 err.] pagerankOmp
```

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/5vdxPZ3.jpg)](https://www.youtube.com/watch?v=rKv_l1RnSqs)

["multiply-sequential-vs-openmp"]: https://github.com/puzzlef/multiply-sequential-vs-openmp
["pagerank-push-vs-pull"]: https://github.com/puzzlef/pagerank-push-vs-pull
["pagerank-class-vs-csr"]: https://github.com/puzzlef/pagerank-class-vs-csr
["graphs"]: https://github.com/puzzlef/graphs
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
