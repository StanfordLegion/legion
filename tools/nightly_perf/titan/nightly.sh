#!/bin/bash

set -e

root_dir="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"

source "$root_dir"/build_vars.sh
export GASNET="$root_dir"/gasnet/release
export TERRA_DIR="$root_dir"/terra.build

source "$root_dir"/env.sh # defines PERF_ACCESS_TOKEN

export CI_RUNNER_DESCRIPTION="titan.ccs.ornl.gov"

export PERF_CORES_PER_NODE=12
export PERF_PRECOMPILE_REGENT=1
export LAUNCHER="aprun -n{NODES} -N1 -cc none -e LD_LIBRARY_PATH=."

scratch_dir="$MEMBERWORK"/csc103/nightly/"$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$scratch_dir"
cd "$scratch_dir"
"$root_dir"/../common/nightly.sh
