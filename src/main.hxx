#pragma once
#include "_main.hxx"
#include "Graph.hxx"
#include "mtx.hxx"
#include "snap.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "duplicate.hxx"
#include "transpose.hxx"
#include "selfLoop.hxx"
#include "deadEnds.hxx"
#include "dfs.hxx"
#include "components.hxx"
#include "sort.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"
#include "pagerankSeq.hxx"
#include "pagerankMonolithicSeq.hxx"
#include "pagerankCuda.hxx"
#include "pagerankMonolithicCuda.hxx"

#ifndef NVGRAPH_DISABLE
#include "pagerankNvgraph.hxx"
#else
#define pagerankNvgraph pagerankCuda
#endif
