#pragma once
#include "_main.hxx"
#include "DiGraph.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "mtx.hxx"
#include "copy.hxx"
#include "transpose.hxx"
#include "pagerank.hxx"
#include "pagerankCuda.hxx"
#include "pagerankSeq.hxx"

#ifndef NVGRAPH_DISABLE
#include "pagerankNvgraph.hxx"
#else
#define pagerankNvgraph pagerankCuda
#endif
