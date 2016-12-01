# Regent
# Note: This version does NOT include CUDA or GASNet.

FROM ubuntu:14.04

MAINTAINER Elliott Slaughter <slaughter@cs.stanford.edu>

# Install dependencies.
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y build-essential clang-3.5 git libclang-3.5-dev libncurses5-dev llvm-3.5-dev wget zlib1g-dev && \
    apt-get clean

# Install Regent.
COPY . /usr/local/legion
RUN LLVM_CONFIG=llvm-config-3.5 /usr/local/legion/language/install.py --rdir=auto && \
    ln -s /usr/local/legion/language/regent.py /usr/local/bin/regent

# Configure container startup.
CMD ["/usr/local/bin/regent"]
