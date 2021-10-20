Comparing the effect of using different **functions** for
**convergence check**, with PageRank ([pull], [CSR]).

This experiment was for comparing the performance between:
1. Find pagerank using **L1 norm** for convergence check.
2. Find pagerank using **L2 norm** for convergence check.
3. Find pagerank using **L∞ norm** for convergence check.

It is observed that a number of *error functions* are in use for checking
convergence of PageRank computation. Although [L1 norm] is commonly used
for convergence check, it appears [nvGraph] uses [L2 norm] instead. Another
person in stackoverflow seems to suggest the use of *per-vertex tolerance*
*comparison*, which is essentially the [L∞ norm]. The **L1 norm** `||E||₁`
between two *(rank) vectors* `r` and `s` is calculated as `Σ|rₙ - sₙ|`, or
as the *sum* of *absolute errors*. The **L2 norm** `||E||₂` is calculated
as `√Σ|rₙ - sₙ|2`, or as the *square-root* of the *sum* of *squared errors*
(*euclidean distance* between the two vectors). The **L∞ norm** `||E||ᵢ`
is calculated as `max(|rₙ - sₙ|)`, or as the *maximum* of *absolute errors*.

This experiment was for comparing the performance between PageRank computation
with *L1, L2* and *L∞ norms* as convergence check, for *damping factor
`α = 0.85`, and *tolerance* `τ = 10⁻⁶`. The PageRank algorithm used here is
the *standard power-iteration (pull)* based PageRank. The rank of a vertex in
an iteration is calculated as `c₀ + αΣrₙ/dₙ`, where `c₀` is the *common*
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
*7.9.2009 (Core)*. the execution time of each test case is measured using
*std::chrono::high_performance_timer*. This is done *5 times* for each
test case, and timings are *averaged (AM)*. The *iterations* taken with
each test case is also measured. `500` is the *maximum iterations* allowed.
Statistics of each test case is printed to *standard output (stdout)*, and
redirected to a *log file*, which is then processed with a *script* to
generate a *CSV file*, with each *row* representing the details of a
*single test case*. This *CSV file* is imported into *Google Sheets*,
and necessary tables are set up with the help of the *FILTER* function
to create the *charts*.

From the results it is clear that PageRank computation with **L∞ norm**
**as convergence check is the fastest** , quickly followed by *L2 norm*,
and finally *L1 norm*. Thus, when comparing two or more approaches for an
iterative algorithm, it is important to ensure that all of them use the same
error function as convergence check (and the same parameter values). This
would help ensure a level ground for a good relative performance comparison.

Also note in below charts that PageRank computation with **L∞ norm** as
convergence check **completes in a single iteration for all the road**
**networks** *(ending with _osm)*. This is likely because it is calculated
as `||E||ᵢ = max(|rₙ - sₙ|)`, and depending upon the *order (number of*
*vertices)* `N` of the graph (those graphs are quite large), the maximum
rank change for any single vertex does not exceed the *tolerance* `τ`
value of `10⁻⁶`.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets]. The input
data used for this experiment is available at ["graphs"] (for small ones), and
the [SuiteSparse Matrix Collection]. This experiment was done with guidance
from [Prof. Dip Sankar Banerjee] and [Prof. Kishore Kothapalli].

<br>

```bash
$ g++ -O3 main.cxx
$ ./a.out ~/data/min-1DeadEnd.mtx
$ ./a.out ~/data/min-2SCC.mtx
$ ...

# ...
#
# Loading graph /home/subhajit/data/web-Stanford.mtx ...
# order: 281903 size: 2312497 {}
# order: 281903 size: 2312497 {} (transposeWithDegree)
# [00457.972 ms; 063 iters.] [0.0000e+00 err.] pagerankL1Norm
# [00319.613 ms; 044 iters.] [3.6018e-05 err.] pagerankL2Norm
# [00298.079 ms; 041 iters.] [6.1306e-05 err.] pagerankLiNorm
#
# ...
#
# Loading graph /home/subhajit/data/soc-LiveJournal1.mtx ...
# order: 4847571 size: 68993773 {}
# order: 4847571 size: 68993773 {} (transposeWithDegree)
# [14178.771 ms; 051 iters.] [0.0000e+00 err.] pagerankL1Norm
# [06703.683 ms; 024 iters.] [4.4204e-04 err.] pagerankL2Norm
# [04735.651 ms; 017 iters.] [1.7652e-03 err.] pagerankLiNorm
#
# ...
```

[![](https://i.imgur.com/b8Ov3fB.png)][sheetp]
[![](https://i.imgur.com/7QWPhho.png)][sheetp]
[![](https://i.imgur.com/Gr5C43h.png)][sheetp]
[![](https://i.imgur.com/tEkTXCj.png)][sheetp]

<br>
<br>


## References

- [RAPIDS nvGraph NVIDIA graph library][nvGraph]
- [How to check for Page Rank convergence?][L∞ norm]
- [L0 Norm, L1 Norm, L2 Norm & L-Infinity Norm](https://montjoile.medium.com/l0-norm-l1-norm-l2-norm-l-infinity-norm-7a7d18a4f40c)
- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](https://www.youtube.com/watch?v=ke9g8hB0MEo)
- [Weighted Geometric Mean Selected for SPECviewperf® Composite Numbers](https://www.spec.org/gwpg/gpc.static/geometric.html)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/BnCiig7.jpg)](https://www.youtube.com/watch?v=04Uv44DRJAU)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
["graphs"]: https://github.com/puzzlef/graphs
[nvGraph]: https://github.com/rapidsai/nvgraph
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[CSR]: https://github.com/puzzlef/pagerank-class-vs-csr
[L1 norm]: https://github.com/rapidsai/nvgraph/blob/main/cpp/src/pagerank.cu#L154
[L2 norm]: https://github.com/rapidsai/nvgraph/blob/main/cpp/src/pagerank.cu#L149
[L∞ norm]: https://stackoverflow.com/a/29321153/1413259
[charts]: https://photos.app.goo.gl/WpPKW5ZRj8qHJkPN8
[sheets]: https://docs.google.com/spreadsheets/d/1TpoKE-WkbKvnym5zvm4-0CL-n5nRkxQkSM7f9qFKeLo/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vSN6xnlxOz8u4PMYxUbxP01qFq8lrYa6IC8DH2pYFGkMmWD4-BB4jdk4e3Cp9Yh_GUG5SzF5OG7ZSex/pubhtml
