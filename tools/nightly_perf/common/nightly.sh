#!/bin/bash

set -e

# Test suite configuration
export CC_FLAGS="-std=c++98"
export DEBUG=0

# Run perf test for each branch
for branch in master dma deppart; do
    if [[ -d _legion_$branch ]]; then
        git -C _legion_$branch pull --ff-only
        git -C _legion_$branch reset --hard HEAD
        git -C _legion_$branch clean -fdx
    else
        git clone -b $branch https://github.com/StanfordLegion/legion.git _legion_$branch
    fi
    pushd _legion_$branch
    ./test.py --test=perf
    popd
done
