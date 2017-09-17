#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "$root_dir"

export TERRA_DIR="$root_dir"/../../../language/terra

source env.sh # defines PERF_ACCESS_TOKEN

export CI_RUNNER_DESCRIPTION="sherlock.stanford.edu"

../common/nightly.sh
