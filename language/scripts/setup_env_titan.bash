#!/bin/bash
root_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "${root_dir}"
echo root dir is ${root_dir}
pwd

unset LG_RT_DIR
CONDUIT=gemini CC=gcc CXX=CC HOST_CC=gcc HOST_CXX=g++ ./setup_env.py
