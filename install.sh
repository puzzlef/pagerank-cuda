#!/usr/bin/env bash

# Download nvGraph
cd ~
git clone --recurse-submodules https://github.com/rapidsai/nvgraph

# Install nvGraph
export CUDA_ROOT="/usr/local/cuda"
cd ~/nvgraph && ./build.sh


# Build and Install the C/C++ CUDA components
# mkdir -p ~/nvgraph/cpp/build
# cd ~/nvgraph/cpp/build
# cmake .. -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT=1 -DTHRUST_IGNORE_CUB_VERSION_CHECK=1 -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
# make -j && make install


# C++ stand alone tests
cd ~/nvgraph/cpp/build
gtests/NVGRAPH_TEST
