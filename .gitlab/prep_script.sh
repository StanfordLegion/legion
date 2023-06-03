#!/bin/bash

set -e
set -x

# be sure we're starting in a (really) clean repository
git clean -fxd
rm -rfv language/terra.build gasnet Thrust
git status

# setup workdir
mkdir -p $(dirname $EXTERNAL_WORKDIR)
rm -rf $EXTERNAL_WORKDIR
cp -r $CI_PROJECT_DIR $EXTERNAL_WORKDIR
cd $EXTERNAL_WORKDIR

# setup environment
if [[ "$LMOD_SYSTEM_NAME" = crusher ]]; then
    cat >>env.sh <<EOF
module load PrgEnv-gnu
module load rocm/$ROCM_VERSION
export HIP_PATH="\$ROCM_PATH/hip"
export CC=cc
export CXX=CC
if [[ "\$REALM_NETWORKS" != "" ]]; then
    RANKS_PER_NODE=4
    export LAUNCHER="srun -n\$(( RANKS_PER_NODE * SLURM_JOB_NUM_NODES )) --cpus-per-task \$(( 56 / RANKS_PER_NODE )) --gpus-per-task 1 --cpu-bind cores"
fi
EOF
else
    echo "Don't know how to build on this system"
    exit 1
fi

set +x # don't spew all the module goop to the console
source env.sh
set -x

# download and build Terra
(
    pushd language
    wget -nv https://github.com/terralang/llvm-build/releases/download/llvm-16.0.3/clang+llvm-16.0.3-x86_64-linux-gnu.tar.xz
    tar xf clang+llvm-16.0.3-x86_64-linux-gnu.tar.xz
    export CMAKE_PREFIX_PATH="$PWD/clang+llvm-16.0.3-x86_64-linux-gnu:$CMAKE_PREFIX_PATH"
    git clone https://github.com/terralang/terra.git terra.build
    ln -s terra.build terra
    pushd terra.build/build
    export CC=gcc
    export CXX=g++
    cmake -DCMAKE_INSTALL_PREFIX=$PWD/../release ..
    make install -j${THREADS:-16}
    popd
    rm -rf clang+llvm-16.0.3-x86_64-linux-gnu*
    popd
)

# download Thrust
git clone https://github.com/ROCmSoftwarePlatform/Thrust.git
cat >>env.sh <<EOF
# Important: has to be \$EXTERNAL_WORKDIR or else CMake sees it in-source
export THRUST_PATH=\$EXTERNAL_WORKDIR/Thrust
EOF

# download and build GASNet
if [[ "$REALM_NETWORKS" == gasnet* ]]; then
    git clone https://github.com/StanfordLegion/gasnet.git
    set +x # makes the build very noisy
    CONDUIT=$GASNET_CONDUIT make -C gasnet -j${THREADS:-16}
    set -x
    if [[ "$GASNET_DEBUG" -eq 1 ]]; then
        cat >>env.sh <<EOF
# Important: has to be \$EXTERNAL_WORKDIR or else CMake sees it in-source
export GASNET_ROOT="\$EXTERNAL_WORKDIR/gasnet/debug"
EOF
    else
        cat >>env.sh <<EOF
# Important: has to be \$EXTERNAL_WORKDIR or else CMake sees it in-source
export GASNET_ROOT="\$EXTERNAL_WORKDIR/gasnet/release"
EOF
    fi
fi
