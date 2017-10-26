#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "$root_dir"

source env.sh # defines PERF_ACCESS_TOKEN

export CI_RUNNER_DESCRIPTION="n0002.sapling.stanford.edu"

export PERF_MIN_NODES=1
export PERF_MAX_NODES=1

../common/nightly.sh
