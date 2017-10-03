#!/bin/bash

mkdir glibc
pushd glibc
git clone -b dlmopen-2.19 git@github.com:elliottslaughter/glibc.git
mkdir install
mkdir build
pushd build
../glibc/configure --prefix=$PWD/../install
make -j8
make install
popd
popd

# FIXME: Not sure this is working, because we need to put libpython2.7.so on LD_LIBRARY_PATH later
mv glibc/install/etc glibc/install/etc.backup
ln -s /etc glibc/install/etc

../../tools/gen_dlmproxy.py /lib/x86_64-linux-gnu/libpthread.so.0 | gcc -x c -fPIC -shared - -Wl,-soname=libpthread.so.0 -o dlmproxy_libpthread.so.0

LG_RT_DIR=$HOME/legion/runtime USE_DLMOPEN=1 make -j8

# FIXME: Need libpython2.7.so to be on LD_LIBRARY_PATH here, so ld.so.cache probably isn't configured properly
LD_LIBRARY_PATH=.:glibc/install/lib:/usr/lib/x86_64-linux-gnu ./glibc/install/lib/ld-linux-x86-64.so.2 ./python_interop -ll:py 1 -level python=2
