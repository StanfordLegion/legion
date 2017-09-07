#!/bin/bash

set -e

source env.sh

# Configure test
export CC_FLAGS="-std=c++98"
export DEBUG=0
export LAUNCHER="mpirun -H n0002 -n 1 -npernode 1 -bind-to none -x TERRA_PATH -x INCLUDE_PATH -x LD_LIBRARY_PATH -x LG_RT_DIR -x USE_RDIR"
export CI_RUNNER_DESCRIPTION="n0002.sapling.stanford.edu"

# Run perf test for each branch
for branch in master; do
    if [[ -d _legion_$branch ]]; then
        git -C _legion_$branch pull --ff-only
        git -C _legion_$branch clean -fdx
    else
        git clone -b $branch git@github.com:StanfordLegion/legion.git _legion_$branch
    fi
    pushd _legion_$branch
    ./test.py --test=perf
    popd
done
