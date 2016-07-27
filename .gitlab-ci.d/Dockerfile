# Build Image for Gitlab CI

FROM ubuntu:14.04

MAINTAINER Elliott Slaughter <slaughter@cs.stanford.edu>

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq && \
    apt-get install -qq software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    add-apt-repository ppa:pypy/ppa -y && \
    apt-get update -qq && \
    apt-get install -qq build-essential git wget g++-4.9 clang-3.5 libclang-3.5-dev llvm-3.5-dev openmpi-bin libopenmpi-dev libblas-dev liblapack-dev pypy && \
    apt-get clean
