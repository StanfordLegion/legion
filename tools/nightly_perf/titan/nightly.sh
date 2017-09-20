#!/bin/bash

#set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "${root_dir}"

if [[ "${LG_RT_DIR}" == "" ]]
then
  echo LG_RT_DIR is not defined
  exit -1
fi

export TERRA_DIR="${LG_RT_DIR}/../language/terra"

source titan_setup.bash
source env.bash  # defines PERF_ACCESS_TOKEN, DO NOT CHECK env.bash INTO THE REPO

export CI_RUNNER_DESCRIPTION="titan.ccs.ornl.gov"

#export TEST_ARGUMENTS='--perf_max_nodes=1024'
export TEST_ARGUMENTS='--perf_max_nodes=4'

../common/nightly.sh

