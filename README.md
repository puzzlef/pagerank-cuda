Design of **CUDA-based** *PageRank algorithm* for link analysis.

All *seventeen* graphs used in below experiments are
stored in the *MatrixMarket (.mtx)* file format, and obtained from the
*SuiteSparse* *Matrix Collection*. These include: *web-Stanford, web-BerkStan,*
*web-Google, web-NotreDame, soc-Slashdot0811, soc-Slashdot0902,*
*soc-Epinions1, coAuthorsDBLP, coAuthorsCiteseer, soc-LiveJournal1,*
*coPapersCiteseer, coPapersDBLP, indochina-2004, italy_osm,*
*great-britain_osm, germany_osm, asia_osm*. The experiments are implemented
in *C++*, and compiled using *GCC 9* with *optimization level 3 (-O3)*.
The *iterations* taken with each test case is measured. `500` is the
*maximum iterations* allowed. Statistics of each test case is
printed to *standard output (stdout)*, and redirected to a *log file*,
which is then processed with a *script* to generate a *CSV file*, with
each *row* representing the details of a *single test case*.

<br>


### Finding Launch config for Block-per-vertex

This experiment ([block-adjust-launch]) was for finding a suitable **launch**
**config** for **CUDA block-per-vertex**. For the launch config, the
**block-size** (threads) was adjusted from `32`-`1024`, and the **grid-limit**
(max grid-size) was adjusted from `1024`-`32768`. Each config was run 5 times
per graph to get a good time measure.

`MAXx64` appears to be a good config for most graphs. Here `MAX` is the
**grid-limit**, and `64` is the **block-size**. This launch config is for the
entire graph, and could be slightly different for subset of graphs. Also note
that this applies to *Tesla V100 PCIe 16GB*, and could be different for other
GPUs. In order to measure error, [nvGraph] pagerank is taken as a reference.

[block-adjust-launch]: https://github.com/puzzlef/pagerank-cuda/tree/block-adjust-launch

<br>


### Finding Launch config for Thread-per-vertex

This experiment ([thread-adjust-launch]) was for finding a suitable **launch**
**config** for **CUDA thread-per-vertex**. For the launch config, the
**block-size** (threads) was adjusted from `32`-`1024`, and the **grid-limit**
(max grid-size) was adjusted from `1024`-`32768`. Each config was run 5 times
per graph to get a good time measure.

On average, the launch config doesn't seem to have a good enough impact on
performance. However `8192x128` appears to be a good config. Here `8192` is the
*grid-limit*, and `128` is the *block-size*. Comparing with [graph properties],
seems it would be better to use `8192x512` for graphs with **high** *avg.
density*, and `8192x32` for graphs with **high** *avg. degree*. Maybe, sorting
the vertices by degree can have a good effect (due to less warp divergence).
Note that this applies to **Tesla V100 PCIe 16GB**, and would be different for
other GPUs. In order to measure error, [nvGraph] pagerank is taken as a
reference.

[thread-adjust-launch]: https://github.com/puzzlef/pagerank-cuda/tree/thread-adjust-launch

<br>


### Sorting vertices by in-degree for Block-per-vertex?

This experiment ([block-sort-by-indegree]) was for finding the effect of sorting
vertices and/or edges by in-degree for CUDA **block-per-vertex** based PageRank.
For this experiment, sorting of vertices and/or edges was either `NO`, `ASC`, or
`DESC`. This gives a total of `3 * 3 = 9` cases. Each case is run on multiple
graphs, running each 5 times per graph for good time measure.

Results show that sorting in *most cases* is **not faster**. In fact, in a
number of cases, sorting actually slows dows performance. Maybe (just maybe)
this is because sorted arrangement tend to overflood certain memory chunks with
too many requests. In order to measure error, [nvGraph] pagerank is taken as a
reference.

[block-sort-by-indegree]: https://github.com/puzzlef/pagerank-cuda/tree/block-sort-by-indegree

<br>


### Sorting vertices by in-degree for Thread-per-vertex?

This experiment ([thread-sort-by-indegree]) was for finding the effect of
sorting vertices and/or edges by in-degree for CUDA **thread-per-vertex** based
PageRank. For this experiment, sorting of vertices and/or edges was either `NO`,
`ASC`, or `DESC`. This gives a total of `3 * 3 = 9` cases. Each case is run on
multiple graphs, running each 5 times per graph for good time measure.

Results show that sorting in *most cases* is **slower**. Maybe this is because
sorted arrangement tends to overflood certain memory chunks with too many
requests. In order to measure error, [nvGraph] pagerank is taken as a reference.

[thread-sort-by-indegree]: https://github.com/puzzlef/pagerank-cuda/tree/thread-sort-by-indegree

<br>


### Sorting vertices by in-degree for Switched-per-vertex?

This experiment ([switched-sort-by-indegree]) was for finding the effect of
sorting vertices and/or edges by in-degree for CUDA **switched-per-vertex**
based PageRank. For this experiment, sorting of vertices and/or edges was either
`NO`, `ASC`, or `DESC`. This gives a total of `3 * 3 = 9` cases. `NO` here means
that vertices are partitioned by in-degree (edges remain unchanged). Each case
is run on multiple graphs, running each 5 times per graph for good time measure.

Results show that **sorting** in most cases is **not faster**. Its better to
simply **partition** *vertices* by *degree*. In order to measure error,
[nvGraph] pagerank is taken as a reference.

[switched-sort-by-indegree]: https://github.com/puzzlef/pagerank-cuda/tree/switched-sort-by-indegree

<br>


### Finding Block Launch config for Switched-per-vertex

This experiment ([switched-adjust-block-launch]) was for finding a suitable
**launch config** for **CUDA switched-per-vertex** for block approach. For the
launch config, the **block-size** (threads) was adjusted from `32`-`1024`, and
the **grid-limit** (max grid-size) was adjusted from `1024`-`32768`. Each config
was run 5 times per graph to get a good time measure.

`MAXx256` appears to be a good config for most graphs. Here `MAX` is the
*grid-limit*, and `256` is the *block-size*. Note that this applies to **Tesla**
**V100 PCIe 16GB**, and would be different for other GPUs. In order to measure
error, [nvGraph] pagerank is taken as a reference.

[switched-adjust-block-launch]: https://github.com/puzzlef/pagerank-cuda/tree/switched-adjust-block-launch

<br>


### Finding Block Launch config for Switched-per-vertex

This experiment ([switched-adjust-thread-launch]) was for finding a suitable
**launch config** for **CUDA switched-per-vertex** for thread approach. For the
launch config, the **block-size** (threads) was adjusted from `32`-`1024`, and
the **grid-limit** (max grid-size) was adjusted from `1024`-`32768`. Each config
was run 5 times per graph to get a good time measure.

`MAXx512` appears to be a good config for most graphs. Here `MAX` is the
*grid-limit*, and `512` is the *block-size*. Note that this applies to **Tesla**
**V100 PCIe 16GB**, and would be different for other GPUs. In order to measure
error, [nvGraph] pagerank is taken as a reference.

[switched-adjust-thread-launch]: https://github.com/puzzlef/pagerank-cuda/tree/switched-adjust-thread-launch

<br>


### Finding Switch point for Switched-per-vertex

For this experiment ([switched-adjust-switch-point]), `switch_degree` was varied
from `2` - `1024`, and `switch_limit` was varied from `1` - `1024`.
`switch_degree` defines the *in-degree* at which *pagerank kernel* switches from
**thread-per-vertex** approach to **block-per-vertex**. `switch_limit` defines
the minimum block size for **thread-per-vertex** / **block-per-vertex** approach
(if a block size is too small, it is merged with the other approach block). Each
case is run on multiple graphs, running each 5 times per graph for good time
measure. It seems `switch_degree` of **64** and `switch_limit` of **32** would
be a good choice.

[switched-adjust-switch-point]: https://github.com/puzzlef/pagerank-cuda/tree/switched-adjust-switch-point

<br>


### Adjusting Per-iteration Rank scaling

[nvGraph PageRank] appears to use [L2-norm per-iteration scaling]. This is
(probably) required for finding a solution to **eigenvalue problem**. However,
as the *eigenvalue* for PageRank is `1`, this is not necessary. This experiement
was for observing if this was indeed true, and that any such *per-iteration
scaling* doesn't affect the number of *iterations* needed to converge.

In this experiment ([adjust-iteration-scaling]), PageRank was computed with
**L1**, **L2**, or **L∞-norm** and the effect of **L1** or **L2-norm** *scaling*
*of ranks* was compared with **baseline (L0)**. Results match the above
assumptions, and indeed no performance benefit is observed (except a reduction
in a single iteration for *soc-Slashdot0811*, *soc-Slashdot-0902*,
*soc-LiveJournal1*, and *italy_osm* graphs).

[adjust-iteration-scaling]: https://github.com/puzzlef/pagerank-cuda/tree/adjust-iteration-scaling
[nvGraph PageRank]: https://github.com/rapidsai/nvgraph/blob/main/cpp/src/pagerank.cu
[L2-norm per-iteration scaling]: https://github.com/rapidsai/nvgraph/blob/main/cpp/src/pagerank.cu#L145

<br>


### Comparing with nvGraph PageRank

This experiment ([compare-nvgraph]) was for comparing the performance between
finding pagerank using [nvGraph], finding pagerank using **CUDA**, and finding
pagerank using a single thread ([sequential]). Each technique was attempted on
different types of graphs, running each technique 5 times per graph to get a
good time measure. **CUDA** is the [switched-per-vertex] approach running on
GPU. **CUDA** based pagerank is indeed much faster than **sequential** (CPU). In
order to measure error, [nvGraph] pagerank is taken as a reference.

[![](https://i.imgur.com/vDeiY1n.gif)][sheetp]

[![](https://i.imgur.com/N1EUPCS.png)][sheetp]
[![](https://i.imgur.com/5LaxhV4.png)][sheetp]

[compare-nvgraph]: https://github.com/puzzlef/pagerank-cuda/tree/compare-nvgraph

<br>


### Other experiments

- [adjust-damping-factor](https://github.com/puzzlef/pagerank-cuda/tree/adjust-damping-factor)
- [adjust-tolerance](https://github.com/puzzlef/pagerank-cuda/tree/adjust-tolerance)
- [adjust-tolerance-function](https://github.com/puzzlef/pagerank-cuda/tree/adjust-tolerance-function)

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

<br>
<br>


[![](https://i.imgur.com/fjeKRUf.jpg)](https://www.youtube.com/watch?v=TtTHBmL7N5U)
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/374990003.svg)](https://zenodo.org/badge/latestdoi/374990003)
![](https://ga-beacon.deno.dev/G-KD28SG54JQ:hbAybl6nQFOtmVxW4if3xw/github.com/puzzlef/pagerank-cuda)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
[nvGraph]: https://github.com/rapidsai/nvgraph
[sequential]: https://github.com/puzzlef/pagerank-sequential-vs-openmp
[switched-per-vertex]: https://github.com/puzzlef/pagerank-cuda-switched-adjust-switch-point
[charts]: https://photos.app.goo.gl/MLcbhUPmLEC7iaEm9
[sheets]: https://docs.google.com/spreadsheets/d/12u5yq49MLS2QRhWHkZF7SWs1JSS4u1sb7wKl8ExrJgg/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vTijFuWx76ZnNfJs5U0IEY1jMEWffi6Pc8uw4FbnXB1R3Puduyn-mPvq4kdMFyyhq0V7GJZQ0722nDS/pubhtml
