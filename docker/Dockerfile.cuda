# Regent + Legion with CUDA but no GASNET

FROM ubuntu:16.04

MAINTAINER Michael Bauer <mbauer@nvidia.com>

# Install dependencies.
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq && \
    apt-get install -qq software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    add-apt-repository ppa:pypy/ppa -y && \
    apt-get update -qq && \
    apt-get install -qq \
      build-essential git python-pip pypy time wget \
      g++-4.8 g++-4.9 g++-5 g++-6 \
      gcc-4.9-multilib g++-4.9-multilib \
      libncurses5-dev \
      zlib1g-dev \
      mpich libmpich-dev \
      mesa-common-dev \
      libblas-dev liblapack-dev libhdf5-dev \
      module-init-tools \
      gdb vim && \
    apt-get clean

# Build LLVM and Clang
RUN wget http://releases.llvm.org/3.8.1/llvm-3.8.1.src.tar.xz && \
    tar -xf llvm-3.8.1.src.tar.xz && \
    wget http://releases.llvm.org/3.8.1/cfe-3.8.1.src.tar.xz && \
    tar -xf cfe-3.8.1.src.tar.xz && \
    mv cfe-3.8.1.src llvm-3.8.1.src/tools/clang && \
    mkdir llvm-build && cd llvm-build && \
    ../llvm-3.8.1.src/configure --enable-optimized --disable-assertions --disable-terminfo --disable-libedit --disable-zlib && \
    make -j 20 && make install && cd ..

# Install CUDA
RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb && apt-get update -qq && \
    apt-get -y install cuda-command-line-tools-8-0 cuda-core-8-0 cuda-cublas-dev-8-0 && \
    ln -s /usr/local/cuda-8.0 /usr/local/cuda && \
    wget http://us.download.nvidia.com/XFree86/Linux-x86_64/375.66/NVIDIA-Linux-x86_64-375.66.run && \
    sh ./NVIDIA-Linux-x86_64-375.66.run -s -N --no-kernel-module && \
    rm cuda-repo-ubuntu1604_8.0.61-1_amd64.deb && rm NVIDIA-Linux-x86_64-375.66.run

# Configure the environment for CUDA
ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA /usr/local/cuda

# Install Regent.
RUN git clone -b master https://github.com/StanfordLegion/legion.git /usr/local/legion
RUN LLVM_CONFIG=llvm-config /usr/local/legion/language/install.py --rdir=auto --cuda && \
    ln -s /usr/local/legion/language/regent.py /usr/local/bin/regent

# Configure container startup.
CMD ["/bin/bash"]
