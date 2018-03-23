#!/bin/bash

module load intel/18.0.1

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

target_dir="$1"

set -x

icpc  -O3 -xHOST -Wall -Werror -DDTYPE=double -DRESTRICT=__restrict__ -DRADIUS=2 -shared -fPIC "$root_dir"/../../examples/stencil.cc -o "$target_dir"/libstencil.so
