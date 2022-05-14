export CI_PIPELINE_ID=manual
export CI_JOB_ID=test
export CI_PROJECT_DIR=$PWD

export USE_HIP="1"
export ROCM_VERSION="4.5.0"
export HIP_ARCH="gfx90a" # for runtime.mk
export GPU_ARCH="gfx90a" # for Regent

export REALM_NETWORKS="gasnetex"
export GASNET_CONDUIT="ofi-slingshot11"
export CONDUIT="ofi"

export CXXFLAGS="-std=c++11"
export CXX_STANDARD="11"

export SCHEDULER_PARAMETERS="-A CSC335_crusher -t 1:30:00 -N 1 -p batch"
export EXTERNAL_WORKDIR=/gpfs/alpine/csc335/proj-shared/ci/${CI_PIPELINE_ID}
export GIT_SUBMODULE_STRATEGY=recursive

export THREADS=16 # for parallel build

export TEST_REGENT="0"
export LEGION_WARNINGS_FATAL="1"
