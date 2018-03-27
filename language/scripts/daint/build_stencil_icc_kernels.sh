#!/bin/bash

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

target_dir="$1"

module unload PrgEnv-gnu
module load PrgEnv-intel/6.0.4

icpc --version

icpc  -O3 -xHOST -Wall -Werror -DDTYPE=double -DRESTRICT=__restrict__ -DRADIUS=2 -shared -fPIC "$root_dir"/../../examples/stencil.cc -o "$target_dir"/libstencil.so
