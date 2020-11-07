# Regent Compiler

This directory contains the compiler for the Regent language.

## Quickstart for Ubuntu

```
sudo apt-get install llvm-3.5-dev libclang-3.5-dev clang-3.5
./install.py --debug
./regent.py examples/circuit.rg
```

## Prerequisites

Regent requires:

  * Python 3.5 or newer (for the self-installer and test suite)
  * LLVM and Clang **with headers**:
      * LLVM 3.8 is recommended for use with CUDA
      * LLVM 3.5 is recommended for debugging (other versions will be missing debug symbols)
      * LLVM 3.5-3.9 and 6.0 are also supported

Regent also has a number of transitive dependencies via Legion:

  * Linux, macOS, or another Unix
  * A C++ 11 or newer compiler (GCC, Clang, Intel, or PGI) and GNU Make
  * *Optional*: CUDA 7.0 or newer (for NVIDIA GPUs)
  * *Optional*: [GASNet](https://gasnet.lbl.gov/) (for networking)
  * *Optional*: LLVM 3.5-3.9 (for dynamic code generation)
  * *Optional*: HDF5 (for file I/O)

## Installing

Run the following command from the `language` directory:

```
./install.py [-h] [--with-terra TERRA] [--debug] [--gasnet]
             [--cuda] [-j [THREAD_COUNT]]
```

This command:

  * Downloads and builds [Terra](http://terralang.org/). (Terra will
    recursively download and build [LuaJIT](http://luajit.org/)).
  * Builds a dynamic library for [Legion](http://legion.stanford.edu/).
  * (OS X) Patches said dynamic library to avoid hard-coded absolute paths.
  * Sets everything up to run from `regent.py`.

Notes:

  * Use `--debug` to enable Legion's debug mode. Debug mode includes
    significantly more safety checks and better error messages, but
    also runs signficantly slower than release mode.
  * For CUDA support, use `--cuda`.
  * For GASNet (networking) support, use `--gasnet`.
  * If you want to build with your own copy of Terra, pass the path to
    the Terra directory via the `--with-terra` flag. You will be
    responsible for building Terra yourself if you do this.

## Running

To use Regent, run:

```
./regent.py [SCRIPT] [OPTIONS]
```

Regent source files use the extension `.rg`. A number of examples are
contained in the examples directory. For example:

```
./regent.py examples/circuit.rg
```

## Tests

To run the test suite, run:

    ./test.py [-h] [-j [THREAD_COUNT]] [-v]
