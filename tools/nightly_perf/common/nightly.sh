#!/bin/bash

set -e

# Test suite configuration
export CC_FLAGS="-std=c++98"
export DEBUG=0

# Run perf test for each branch
for branch in master deppart; do
    if [[ -d _legion_$branch ]]; then
        pushd _legion_$branch
        git reset --hard HEAD
        git clean -fdx
        git pull --ff-only
        git submodule update --init
        popd
    else
        git clone -b $branch https://github.com/StanfordLegion/legion.git _legion_$branch
    fi
    pushd _legion_$branch
    if [[ -d $TERRA_DIR ]]; then
        ln -s "$TERRA_DIR" language/terra
    fi
    ./test.py --test=perf
    popd
done
