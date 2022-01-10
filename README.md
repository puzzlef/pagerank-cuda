PageRank (PR) example using NVIDIA's Graph Library [nvGraph].

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets]. The input
data used for this experiment is available at the [SuiteSparse Matrix
Collection].

```bash
$ nvcc -std=c++17 -Xcompiler -lnvgraph -O3 main.cu
$ ./a.out ~/data/web-Stanford.mtx
$ ./a.out ~/data/web-BerkStan.mtx
$ ...

# Loading graph /home/subhajit/data/web-Stanford.mtx ...
# order: 281903 size: 2312497 {}
# order: 281903 size: 2312497 {} (transposeWithDegree)
# [00011.348 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
#
# ...
#
# Loading graph /home/subhajit/data/soc-LiveJournal1.mtx ...
# order: 4847571 size: 68993773 {}
# order: 4847571 size: 68993773 {} (transposeWithDegree)
# [00168.232 ms; 000 iters.] [0.0000e+00 err.] pagerankNvgraph
#
# ...
```

[![](https://i.imgur.com/jUgH24r.png)][sheetp]
[![](https://i.imgur.com/bg2zNx0.png)][sheetp]

<br>
<br>


## References

- [nvGraph pagerank example, EN605.617, JHU-EP-Intro2GPU](https://github.com/JHU-EP-Intro2GPU/EN605.617/blob/master/module9/nvgraph_examples/nvgraph_Pagerank.cpp)
- [nvGraph pagerank example, CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/10.0/nvgraph/index.html#nvgraph-pagerank-example)
- [nvGraph, CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/10.0/nvgraph/index.html#introduction)
- [RAPIDS nvGraph NVIDIA graph library][nvGraph]

[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[nvGraph]: https://github.com/rapidsai/nvgraph
[charts]: https://photos.app.goo.gl/owJ8YMMpoQQUqLGx5
[sheets]: https://docs.google.com/spreadsheets/d/1gWzOw_715qpzYxjTvV5ti-lyMcsiqt43Vy9IFyibUGc/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vQjvsrRquyQ0wE1e33Raz3mOYnTRU-hhTbk1nSRPraM1GkgBBxitL5KQjYwGTgpK7lDhD_ZMqs0jYze/pubhtml
