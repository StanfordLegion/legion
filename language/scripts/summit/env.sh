module load cmake/3.9.2
module load gcc/6.4.0
module load cuda/9.1.85
export CC=gcc
export CXX=g++
export CONDUIT=ibv
export CUDA=$OLCF_CUDA_ROOT
export CUDA_HOME=$OLCF_CUDA_ROOT # for Terra
export GPU_ARCH=volta
