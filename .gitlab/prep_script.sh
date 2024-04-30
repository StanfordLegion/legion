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

# download Terra
(
    pushd language
    wget -nv https://github.com/terralang/terra/releases/download/release-1.1.0/terra-Linux-x86_64-be89521.tar.xz
    tar xf terra-Linux-x86_64-be89521.tar.xz
    ln -s terra-Linux-x86_64-be89521 terra
    popd
)

# download Thrust
git clone https://github.com/ROCmSoftwarePlatform/Thrust.git

# download GASNet
if [[ "$REALM_NETWORKS" == gasnet* ]]; then
    git clone https://github.com/StanfordLegion/gasnet.git
fi
