export USE_HIP="1"
export ROCM_VERSION="5.4.3"
export HIP_ARCH="gfx90a" # for runtime.mk
export GPU_ARCH="gfx90a" # for Regent

export CI_PIPELINE_ID=manual_${USER}_rocm${ROCM_VERSION}
export CI_JOB_ID=test
export CI_PROJECT_DIR=$PWD

export REALM_NETWORKS="gasnetex"
export GASNET_CONDUIT="ofi-slingshot11"
export CONDUIT="ofi"

export CXXFLAGS="-std=c++17"
export HIPCC_FLAGS="-std=c++17"
export CXX_STANDARD="17"

export SCHEDULER_PARAMETERS="-A UMS036 -t 1:30:00 -N 1 -p batch"
export EXTERNAL_WORKDIR=/lustre/orion/proj-shared/ums036/ci/${CI_PIPELINE_ID}
export GIT_SUBMODULE_STRATEGY=recursive

export THREADS=16 # for parallel build

export TEST_REGENT="0"
export LEGION_WARNINGS_FATAL="1"
