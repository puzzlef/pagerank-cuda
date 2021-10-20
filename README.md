Comparing the effect of using **different values of tolerance**,
with PageRank ([pull], [CSR]).

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

Similar to the *damping factor* `α` and the *error function* used for
convergence check, **adjusting the value of tolerance** `τ` can have a
significant effect. This experiment was for comparing the performance
between PageRank computation with *L1, L2* and *L∞ norms* as convergence
check, for various *tolerance* `τ` values ranging from `10⁻⁰` to `10⁻¹⁰`
(`10⁻⁰`, `5×10⁻⁰`, `10⁻¹`, `5×10⁻¹`, ...). The PageRank algorithm used
here is the *standard power-iteration (pull)* based PageRank. The rank
of a vertex in an iteration is calculated as `c₀ + αΣrₙ/dₙ`, where `c₀`
is the *common teleport contribution*, `α` is the *damping factor*, `rₙ`
is the *previous rank of vertex* with an incoming edge, `dₙ` is the
*out-degree* of the incoming-edge vertex, and `N` is the *total number*
*of vertices* in the graph. The *common teleport contribution* `c₀`,
calculated as `(1-α)/N + αΣrₙ/N` , includes the *contribution due to a*
*teleport from any vertex* in the graph due to the damping factor `(1-α)/N`,
and *teleport from dangling vertices* (with *no outgoing edges*) in the
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

For various graphs it is observed that PageRank computation with *L1*, *L2*,
or *L∞ norm* as *convergence check* suffers from **sensitivity issues**
beyond certain (*smaller*) tolerance `τ` values, causing the computation to
halt at maximum iteration limit (`500`) without convergence. As *tolerance*
`τ` is decreased from `10⁻⁰` to `10⁻¹⁰`, *L1 norm* is the *first* to suffer
from this issue, followed by *L2 and L∞ norms (except road networks)*. This
*sensitivity issue* was recognized by the fact that a given approach *abruptly*
takes `500` *iterations* for the next lower *tolerance* `τ` value.

It is also observed that PageRank computation with *L∞ norm* as convergence
check **completes in just one iteration** (even for *tolerance* `τ ≥ 10⁻⁶`)
for large graphs *(road networks)*. This again, as mentioned above, is likely
because the maximum rank change for any single vertex for *L∞ norm*, and
the sum of squares of total rank change for all vertices, is quite low for
such large graphs. Thus, it does not exceed the given *tolerance* `τ` value,
causing a single iteration convergence.

On average, PageRank computation with **L∞ norm** as the error function is the
**fastest**, quickly **followed by** **L2 norm**, and **then** **L1 norm**. This
is the case with both geometric mean (GM) and arithmetic mean (AM) comparisons
of iterations needed for convergence with each of the three error functions. In
fact, this trend is observed with each of the individual graphs separately.

Based on **GM-RATIO** comparison, the *relative iterations* between
PageRank computation with *L1*, *L2*, and *L∞ norm* as convergence check
is `1.00 : 0.30 : 0.20`. Hence *L2 norm* is on *average* `70%` *faster*
than *L1 norm*, and *L∞ norm* is `33%` *faster* than *L2 norm*. This
ratio is calculated by first finding the *GM* of *iterations* based on
each *error function* for each *tolerance* `τ` value separately. These
*tolerance* `τ` specific means are then combined with *GM* to obtain a
*single mean value* for each *error function (norm)*. The *GM-RATIO* is
then the ratio of each *norm* with respect to the *L∞ norm*. The variation
of *tolerance* `τ` specific means with *L∞ norm* as baseline for various
*tolerance* `τ` values is shown below.

On the other hand, based on **AM-RATIO** comparison, the *relative*
*iterations* between PageRank computation with *L1*, *L2*, and *L∞ norm*
as convergence check is `1.00 : 0.39 : 0.31`. Hence, *L2 norm* is on
*average* `61%` *faster* than *L1 norm*, and *L∞ norm* is `26%` *faster*
than *L2 norm*. This ratio is calculated in a manner similar to that of
*GM-RATIO*, except that it uses *AM* instead of *GM*. The variation of
*tolerance* `τ` specific means with *L∞ norm* as baseline for various
*tolerance* `τ` values is shown below as well.

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
# [00422.797 ms; 063 iters.] [0.0000e+00 err.] pagerank
# [00007.232 ms; 001 iters.] [6.0894e-01 err.] pagerankL1Norm [tolerance=1e+00]
# [00007.219 ms; 001 iters.] [6.0894e-01 err.] pagerankL2Norm [tolerance=1e+00]
# [00007.290 ms; 001 iters.] [6.0894e-01 err.] pagerankLiNorm [tolerance=1e+00]
# [00013.929 ms; 002 iters.] [3.3244e-01 err.] pagerankL1Norm [tolerance=5e-01]
# [00007.227 ms; 001 iters.] [6.0894e-01 err.] pagerankL2Norm [tolerance=5e-01]
# [00007.238 ms; 001 iters.] [6.0894e-01 err.] pagerankLiNorm [tolerance=5e-01]
# [00034.029 ms; 005 iters.] [9.4240e-02 err.] pagerankL1Norm [tolerance=1e-01]
# [00007.210 ms; 001 iters.] [6.0894e-01 err.] pagerankL2Norm [tolerance=1e-01]
# [00007.349 ms; 001 iters.] [6.0894e-01 err.] pagerankLiNorm [tolerance=1e-01]
# [00040.713 ms; 006 iters.] [6.9689e-02 err.] pagerankL1Norm [tolerance=5e-02]
# [00007.246 ms; 001 iters.] [6.0894e-01 err.] pagerankL2Norm [tolerance=5e-02]
# [00007.233 ms; 001 iters.] [6.0894e-01 err.] pagerankLiNorm [tolerance=5e-02]
# [00081.089 ms; 012 iters.] [1.6086e-02 err.] pagerankL1Norm [tolerance=1e-02]
# [00020.607 ms; 003 iters.] [1.9915e-01 err.] pagerankL2Norm [tolerance=1e-02]
# [00020.676 ms; 003 iters.] [1.9915e-01 err.] pagerankLiNorm [tolerance=1e-02]
# [00101.054 ms; 015 iters.] [8.5030e-03 err.] pagerankL1Norm [tolerance=5e-03]
# [00027.407 ms; 004 iters.] [1.3228e-01 err.] pagerankL2Norm [tolerance=5e-03]
# [00020.672 ms; 003 iters.] [1.9915e-01 err.] pagerankLiNorm [tolerance=5e-03]
# [00154.549 ms; 023 iters.] [1.7352e-03 err.] pagerankL1Norm [tolerance=1e-03]
# [00047.428 ms; 007 iters.] [5.2610e-02 err.] pagerankL2Norm [tolerance=1e-03]
# [00040.745 ms; 006 iters.] [6.9689e-02 err.] pagerankLiNorm [tolerance=1e-03]
# [00181.434 ms; 027 iters.] [8.1235e-04 err.] pagerankL1Norm [tolerance=5e-04]
# [00060.830 ms; 009 iters.] [3.1849e-02 err.] pagerankL2Norm [tolerance=5e-04]
# [00040.944 ms; 006 iters.] [6.9689e-02 err.] pagerankLiNorm [tolerance=5e-04]
# [00241.445 ms; 036 iters.] [1.5340e-04 err.] pagerankL1Norm [tolerance=1e-04]
# [00114.457 ms; 017 iters.] [5.6468e-03 err.] pagerankL2Norm [tolerance=1e-04]
# [00087.461 ms; 013 iters.] [1.2954e-02 err.] pagerankLiNorm [tolerance=1e-04]
# [00268.354 ms; 040 iters.] [7.4190e-05 err.] pagerankL1Norm [tolerance=5e-05]
# [00134.240 ms; 020 iters.] [3.1039e-03 err.] pagerankL2Norm [tolerance=5e-05]
# [00114.528 ms; 017 iters.] [5.6468e-03 err.] pagerankLiNorm [tolerance=5e-05]
# [00328.363 ms; 049 iters.] [1.3774e-05 err.] pagerankL1Norm [tolerance=1e-05]
# [00201.158 ms; 030 iters.] [4.6385e-04 err.] pagerankL2Norm [tolerance=1e-05]
# [00181.756 ms; 027 iters.] [8.1235e-04 err.] pagerankLiNorm [tolerance=1e-05]
# [00355.525 ms; 053 iters.] [6.1334e-06 err.] pagerankL1Norm [tolerance=5e-06]
# [00227.883 ms; 034 iters.] [2.2126e-04 err.] pagerankL2Norm [tolerance=5e-06]
# [00208.412 ms; 031 iters.] [3.8458e-04 err.] pagerankLiNorm [tolerance=5e-06]
# [00422.065 ms; 063 iters.] [0.0000e+00 err.] pagerankL1Norm [tolerance=1e-06]
# [00294.168 ms; 044 iters.] [3.6018e-05 err.] pagerankL2Norm [tolerance=1e-06]
# [00275.266 ms; 041 iters.] [6.1306e-05 err.] pagerankLiNorm [tolerance=1e-06]
# [00448.916 ms; 067 iters.] [6.3236e-07 err.] pagerankL1Norm [tolerance=5e-07]
# [00321.751 ms; 048 iters.] [1.7339e-05 err.] pagerankL2Norm [tolerance=5e-07]
# [00302.102 ms; 045 iters.] [2.9406e-05 err.] pagerankLiNorm [tolerance=5e-07]
# [00516.339 ms; 077 iters.] [1.1385e-06 err.] pagerankL1Norm [tolerance=1e-07]
# [00388.001 ms; 058 iters.] [2.4505e-06 err.] pagerankL2Norm [tolerance=1e-07]
# [00368.500 ms; 055 iters.] [3.9247e-06 err.] pagerankLiNorm [tolerance=1e-07]
# [00562.078 ms; 084 iters.] [1.2488e-06 err.] pagerankL1Norm [tolerance=5e-08]
# [00414.634 ms; 062 iters.] [9.2167e-07 err.] pagerankL2Norm [tolerance=5e-08]
# [00395.889 ms; 059 iters.] [1.2932e-06 err.] pagerankLiNorm [tolerance=5e-08]
# [03337.923 ms; 500 iters.] [1.2680e-06 err.] pagerankL1Norm [tolerance=1e-08]
# [00484.968 ms; 072 iters.] [1.1465e-06 err.] pagerankL2Norm [tolerance=1e-08]
# [00469.444 ms; 070 iters.] [1.0973e-06 err.] pagerankLiNorm [tolerance=1e-08]
# [03331.529 ms; 500 iters.] [1.2680e-06 err.] pagerankL1Norm [tolerance=5e-09]
# [00555.127 ms; 083 iters.] [1.2008e-06 err.] pagerankL2Norm [tolerance=5e-09]
# [00488.407 ms; 073 iters.] [1.0328e-06 err.] pagerankLiNorm [tolerance=5e-09]
# [03334.091 ms; 500 iters.] [1.2680e-06 err.] pagerankL1Norm [tolerance=1e-09]
# [03332.581 ms; 500 iters.] [1.2680e-06 err.] pagerankL2Norm [tolerance=1e-09]
# [03337.859 ms; 500 iters.] [1.2680e-06 err.] pagerankLiNorm [tolerance=1e-09]
# [03333.502 ms; 500 iters.] [1.2680e-06 err.] pagerankL1Norm [tolerance=5e-10]
# [03326.985 ms; 500 iters.] [1.2680e-06 err.] pagerankL2Norm [tolerance=5e-10]
# [03335.266 ms; 500 iters.] [1.2680e-06 err.] pagerankLiNorm [tolerance=5e-10]
# [03335.210 ms; 500 iters.] [1.2680e-06 err.] pagerankL1Norm [tolerance=1e-10]
# [03329.489 ms; 500 iters.] [1.2680e-06 err.] pagerankL2Norm [tolerance=1e-10]
# [03337.096 ms; 500 iters.] [1.2680e-06 err.] pagerankLiNorm [tolerance=1e-10]
#
# ...
```

[![](https://i.imgur.com/4NbjAzk.png)][sheetp]
[![](https://i.imgur.com/taoe5p9.png)][sheetp]
[![](https://i.imgur.com/nZxC4H2.png)][sheetp]
[![](https://i.imgur.com/wr5ziJQ.png)][sheetp]

[![](https://i.imgur.com/oZtDiXX.png)][sheetp]
[![](https://i.imgur.com/8ugVRL6.png)][sheetp]
[![](https://i.imgur.com/xmQPqNU.png)][sheetp]
[![](https://i.imgur.com/XR7hHis.png)][sheetp]

[![](https://i.imgur.com/iiVc0mT.gif)][sheetp]
[![](https://i.imgur.com/TfdJbHX.gif)][sheetp]

<br>
<br>


## References

- [RAPIDS nvGraph NVIDIA graph library][nvGraph]
- [How to check for Page Rank convergence?][L∞ norm]
- [L0 Norm, L1 Norm, L2 Norm & L-Infinity Norm](https://montjoile.medium.com/l0-norm-l1-norm-l2-norm-l-infinity-norm-7a7d18a4f40c)
- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](https://www.youtube.com/watch?v=ke9g8hB0MEo)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/p8R1WIk.jpg)](https://www.youtube.com/watch?v=04Uv44DRJAU)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
["graphs"]: https://github.com/puzzlef/graphs
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[CSR]: https://github.com/puzzlef/pagerank-class-vs-csr
[tolerance function]: https://github.com/puzzlef/pagerank-adjust-tolerance-function
[nvGraph]: https://github.com/rapidsai/nvgraph
[L1 norm]: https://github.com/rapidsai/nvgraph/blob/main/cpp/src/pagerank.cu#L154
[L2 norm]: https://github.com/rapidsai/nvgraph/blob/main/cpp/src/pagerank.cu#L149
[L∞ norm]: https://stackoverflow.com/a/29321153/1413259
[charts]: https://photos.app.goo.gl/stdoXDUhRcDvZqZb6
[sheets]: https://docs.google.com/spreadsheets/d/1V-tanVXCIBemrC0jRtech5nA4sYUviwvUFC4G16oFKM/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vR2A2aGvONm_i4p_pun7jlKb8H2fIcYpMuXgV7BhbNAUbEeiHlTxKFWMgkE6_2LCznleVEWsjdsEqfy/pubhtml
