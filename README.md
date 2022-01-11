Comparing the effect of using different values of **damping factor**,
with PageRank ([pull], [CSR]).

`TODO!`

Adjustment of the *damping factor α* is a delicate balancing act. For
smaller values of *α*, the convergence is fast, but the *link structure*
*of the graph* used to determine ranks is less true. Slightly different
values for *α* can produce *very different* rank vectors. Moreover, as
α → 1, convergence *slows down drastically*, and *sensitivity issues*
begin to surface.

For this experiment, the **damping factor** `α` (which is usually `0.85`)
is **varied** from `0.50` to `1.00` in steps of `0.05`. This is in order
to compare the performance variation with each *damping factor*. The
calculated error is the *L1 norm* with respect to default PageRank
(`α = 0.85`). The PageRank algorithm used here is the *standard*
*power-iteration (pull)* based PageRank. The rank of a vertex in an
iteration is calculated as `c₀ + αΣrₙ/dₙ`, where `c₀` is the *common*
*teleport contribution*, `α` is the *damping factor*, `rₙ` is the
*previous rank of vertex* with an incoming edge, `dₙ` is the *out-degree*
of the incoming-edge vertex, and `N` is the *total number of vertices*
in the graph. The *common teleport contribution* `c₀`, calculated as
`(1-α)/N + αΣrₙ/N` , includes the *contribution due to a teleport from*
*any vertex* in the graph due to the damping factor `(1-α)/N`, and
*teleport from dangling vertices* (with *no outgoing edges*) in the
graph `αΣrₙ/N`. This is because a random surfer jumps to a random page
upon visiting a page with *no links*, in order to avoid the *rank-sink*
effect.

All *seventeen* graphs used in this experiment are stored in the
*MatrixMarket (.mtx)* file format, and obtained from the *SuiteSparse*
*Matrix Collection*. These include: *web-Stanford, web-BerkStan,*
*web-Google, web-NotreDame, soc-Slashdot0811, soc-Slashdot0902,*
*soc-Epinions1, coAuthorsDBLP, coAuthorsCiteseer, soc-LiveJournal1,*
*coPapersCiteseer, coPapersDBLP, indochina-2004, italy_osm,*
*great-britain_osm, germany_osm, asia_osm*. The experiment is implemented
in *C++*, and compiled using *GCC 9* with *optimization level 3 (-O3)*.
The system used is a *Dell PowerEdge R740 Rack server* with two *Intel*
*Xeon Silver 4116 CPUs @ 2.10GHz*, *128GB DIMM DDR4 Synchronous Registered*
*(Buffered) 2666 MHz (8x16GB) DRAM*, and running *CentOS Linux release*
*7.9.2009 (Core)*. The *iterations* taken with each test case is measured.
`500` is the *maximum iterations* allowed. Statistics of each test case is
printed to *standard output (stdout)*, and redirected to a *log file*,
which is then processed with a *script* to generate a *CSV file*, with
each *row* representing the details of a *single test case*. This
*CSV file* is imported into *Google Sheets*, and necessary tables are set
up with the help of the *FILTER* function to create the *charts*.

Results indicate that **increasing the damping factor α beyond** `0.85`
**significantly increases convergence time** , and lowering it below
`0.85` decreases convergence time. As the *damping factor* `α` increases
*linearly*, the iterations needed for PageRank computation *increases*
*almost exponentially*. On average, using a *damping factor* `α = 0.95`
increases *iterations* needed by `190%` (`~2.9x`), and using a *damping*
*factor* `α = 0.75` *decreases* it by `41%` (`~0.6x`), compared to
*damping factor* `α = 0.85`. Note that a higher *damping factor* implies
that a random surfer follows links with *higher probability* (and jumps
to a random page with lower probability).

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets]. The input
data used for this experiment is available at ["graphs"] (for small ones), and
the [SuiteSparse Matrix Collection]. This experiment was done with guidance
from [Prof. Dip Sankar Banerjee] and [Prof. Kishore Kothapalli].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -lnvgraph -O3 main.cu
$ ./a.out ~/data/web-Stanford.mtx
$ ./a.out ~/data/web-BerkStan.mtx
$ ...

# Loading graph /home/subhajit/data/web-Stanford.mtx ...
# order: 281903 size: 2312497 {}
# order: 281903 size: 2312497 {} (transposeWithDegree)
# [00011.365 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
# [00084.847 ms; 500 iters.] [1.1739e+00 err.] pagerankMonolithicCuda_L1Norm [damping=1.00]
# [00083.004 ms; 500 iters.] [1.1739e+00 err.] pagerankMonolithicCuda_L2Norm [damping=1.00]
# [00083.787 ms; 500 iters.] [1.1739e+00 err.] pagerankMonolithicCuda_LiNorm [damping=1.00]
# [00031.957 ms; 192 iters.] [3.3279e-01 err.] pagerankMonolithicCuda_L1Norm [damping=0.95]
# [00022.684 ms; 136 iters.] [3.3279e-01 err.] pagerankMonolithicCuda_L2Norm [damping=0.95]
# [00021.703 ms; 129 iters.] [3.3279e-01 err.] pagerankMonolithicCuda_LiNorm [damping=0.95]
# [00016.064 ms; 096 iters.] [1.3471e-01 err.] pagerankMonolithicCuda_L1Norm [damping=0.90]
# ...
```

[![](https://i.imgur.com/EVy5ztW.png)][sheetp]
[![](https://i.imgur.com/XmdXy4x.png)][sheetp]
[![](https://i.imgur.com/w0x9X9C.png)][sheetp]
[![](https://i.imgur.com/LYGi6EK.png)][sheetp]

[![](https://i.imgur.com/RvJyGG4.png)][sheetp]
[![](https://i.imgur.com/Sag6jKq.png)][sheetp]
[![](https://i.imgur.com/nxIXAQm.png)][sheetp]
[![](https://i.imgur.com/AfXVtRr.png)][sheetp]

[![](https://i.imgur.com/bKRDqCc.gif)][sheetp]
[![](https://i.imgur.com/fnIcr6v.gif)][sheetp]
[![](https://i.imgur.com/mzrGTTL.gif)][sheetp]
[![](https://i.imgur.com/raFDkOK.gif)][sheetp]

<br>
<br>


## References

- [Deeper Inside PageRank :: Amy N. Langville, Carl D. Meyer](https://www.slideshare.net/SubhajitSahu/deeper-inside-pagerank-notes)
- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](https://www.youtube.com/watch?v=ke9g8hB0MEo)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/CxwDsTm.jpg)](https://www.youtube.com/watch?v=jcqkqJnTydU)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
["graphs"]: https://github.com/puzzlef/graphs
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[CSR]: https://github.com/puzzlef/pagerank-class-vs-csr
[charts]: https://photos.app.goo.gl/Vrt8o5DZvznDUsL8A
[sheets]: https://docs.google.com/spreadsheets/d/1fUIe0v-aUk7INX_8XcbWf65A06wsldi76imjltKKDcY/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vQKl0W270xWukKgjCIcQd3ygCzKNlLqicXF6T3FPpRF1W9LqaEjZPerylUz70RJz_LGvKQ65hj1xr0M/pubhtml
