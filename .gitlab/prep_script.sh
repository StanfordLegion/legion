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
mkdir $EXTERNAL_WORKDIR
cd $EXTERNAL_WORKDIR

# download Terra
wget -nv https://github.com/terralang/terra/releases/download/release-1.2.0/terra-Linux-x86_64-cc543db.tar.xz
tar xf terra-Linux-x86_64-cc543db.tar.xz
ln -s terra-Linux-x86_64-cc543db terra
wget -nv https://github.com/terralang/llvm-build/releases/download/llvm-18.1.7/clang+llvm-18.1.7-x86_64-linux-gnu.tar.xz
tar xf clang+llvm-18.1.7-x86_64-linux-gnu.tar.xz
ln -s clang+llvm-18.1.7-x86_64-linux-gnu llvm

# download Thrust
git clone https://github.com/ROCmSoftwarePlatform/Thrust.git

# download GASNet
if [[ "$REALM_NETWORKS" == gasnet* ]]; then
    git clone https://github.com/StanfordLegion/gasnet.git
fi
