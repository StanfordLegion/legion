export USE_HIP="1"
export ROCM_VERSION="${ROCM_VERSION:-7.2.0}"
export CMAKE_HIP_ARCHITECTURES="gfx90a" # for CMake
export HIP_ARCH="gfx90a" # for runtime.mk
export GPU_ARCH="gfx90a" # for Regent

export REALM_NETWORKS="gasnetex"
export GASNET_CONDUIT="ofi-slingshot11"
export CONDUIT="ofi"

export CXX_STANDARD="17"

# Legion/Realm don't build cleanly with ROCm, so don't error on warnings
export WARN_AS_ERROR="0"

export CACHE_DIR="$MEMBERWORK/ums036/ci_cache"

export THREADS=32 # for parallel build

export LEGION_WARNINGS_FATAL="1"

module load PrgEnv-gnu
module load cray-python
module load rocm/$ROCM_VERSION
export CC=cc
export CXX=CC

# Proxy settings for Frontier: https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html#proxy-settings
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
