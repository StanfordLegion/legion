#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "$root_dir"

source env.sh

# Configure perf test
export CC_FLAGS="-std=c++98"
export DEBUG=0
export CI_RUNNER_DESCRIPTION="n0002.sapling.stanford.edu"

# Run perf test for each branch
for branch in master; do
    if [[ -d _legion_$branch ]]; then
        git -C _legion_$branch pull --ff-only
        git -C _legion_$branch reset --hard HEAD
        git -C _legion_$branch clean -fdx
    else
        git clone -b $branch git@github.com:StanfordLegion/legion.git _legion_$branch
    fi
    pushd _legion_$branch
    ./test.py --test=perf
    popd
done
