# Build Image for Gitlab CI

FROM ubuntu:14.04

MAINTAINER Elliott Slaughter <slaughter@cs.stanford.edu>

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq && \
    apt-get install -qq software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    add-apt-repository ppa:pypy/ppa -y && \
    apt-get update -qq && \
    apt-get install -qq build-essential git time wget g++-4.9 clang-3.5 libclang-3.5-dev llvm-3.5-dev libncurses5-dev zlib1g-dev openmpi-bin libopenmpi-dev libblas-dev liblapack-dev pypy && \
    apt-get clean

ENV TERRA_DIR /usr/local/terra
RUN git clone https://github.com/zdevito/terra.git && cd terra && make && cd .. && mv terra /usr/local/terra
