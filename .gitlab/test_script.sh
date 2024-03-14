#!/bin/bash

set -e
set -x

# job directory
JOB_WORKDIR="${EXTERNAL_WORKDIR}_${CI_JOB_ID:-legion${TEST_LEGION_CXX:-1}_regent${TEST_REGENT:-1}}"
rm -rf $JOB_WORKDIR
cp -r $EXTERNAL_WORKDIR $JOB_WORKDIR
cd $JOB_WORKDIR
echo "Running tests in $JOB_WORKDIR"

set +x # don't spew all the module goop to the console
cat env.sh # but show us what's inside
source env.sh
set -x

# run test script
./tools/add_github_host_key.sh
grep 'model name' /proc/cpuinfo | uniq -c || true
which $CXX
$CXX --version
free
if [[ -z "$TEST_PYTHON_EXE" ]]; then
    export TEST_PYTHON_EXE=`which python3 python | head -1`
fi
$TEST_PYTHON_EXE ./test.py -j${THREADS:-16}
