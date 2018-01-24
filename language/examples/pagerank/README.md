# PageRank demo

This directory contains the source code implementing PageRank in the Regent language.

* pagerank.rg is a simple version of PageRank in Regent with both CPU and GPU implementations. It also includes a bishop mapper that will automatically map all tasks to GPUs if available. The following legion prof shows running PageRank on the twitter graph with 4 CPUs and 4 GPUs, respectively. The GPU version achieves around 10X speedup compares against the CPU version.

http://sapling.stanford.edu/~zhihao/pagerank_4cpu_tw/
http://sapling.stanford.edu/~zhihao/pagerank_4gpu_tw/

* pagerank_opt.rg is an optimized version that uses external hand-optimized CUDA kernel, which achieves around 2X speedup compared against the auto-generated CUDA kernels.

http://sapling.stanford.edu/~zhihao/pagerank_opt_4gpu_tw.4

* legion_interop.cu contains the implementation of an hand-optimized CUDA kernel for PageRank. It is written by using the Legion C++ interface and linked to Regent.

