# Regent Compiler

This directory contains the compiler for the Regent language.

## Quickstart for Ubuntu

```bash
# install dependencies
sudo apt-get install build-essential cmake git wget
wget https://github.com/terralang/llvm-build/releases/download/llvm-13.0.0/clang+llvm-13.0.0-x86_64-linux-gnu.tar.xz
tar xf clang+llvm-13.0.0-x86_64-linux-gnu.tar.xz
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$PWD/clang+llvm-13.0.0-x86_64-linux-gnu"

# download and build Regent
git clone -b master https://github.com/StanfordLegion/legion.git
cd legion/language
./install.py --debug --rdir=auto

# run Regent example
./regent.py examples/circuit_sparse.rg
```

## Prerequisites

Regent requires:

  * Python 3.5 or newer (for the self-installer and test suite)
  * LLVM and Clang **with headers**:
      * LLVM 13.0 is recommended
      * See the [version support table](https://github.com/terralang/terra#supported-llvm-versions) for more details
      * Pre-built binaries are available [here](https://github.com/terralang/llvm-build/releases)
  * *Optional (but strongly recommended)*: CMake 3.5 or newer

Regent also has a number of transitive dependencies via Legion:

  * Linux, macOS, or another Unix
  * A C++ 11 or newer compiler (GCC, Clang, Intel, or PGI) and GNU Make
  * *Optional*: CUDA 7.0 or newer (for NVIDIA GPUs)
  * *Optional*: [GASNet](https://gasnet.lbl.gov/) (for networking)
  * *Optional*: HDF5 (for file I/O)

## Installing

Run the following command from the `language` directory:

```bash
./install.py [-h] [--debug] [--gasnet] [--cuda] [--openmp] [--python] [--hdf5]
             [--rdir {prompt,auto,manual,skip,never}] [--with-terra DIR]
             [-j [THREAD_COUNT]]
```

Run with `-h` to get a full list of available flags.

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
  * RDIR is an optional (but recommended) plugin for Regent; this is
    required for Regent's static control replication
    optimization. Selecting `auto` here will instruct the installer to
    download and manage RDIR automatically.
  * If you want to build with your own copy of Terra, pass the path to
    the Terra directory via the `--with-terra` flag. You will be
    responsible for building Terra yourself if you do this.

## Running

To use Regent, run:

```bash
./regent.py [SCRIPT] [COMPILER_AND_SCRIPT_OPTIONS]
```

Regent source files use the extension `.rg`. A number of examples are
contained in the examples directory. For example:

```bash
./regent.py examples/circuit_sparse.rg
```

Arguments to the Regent compiler and the script itself can be
intermixed following the script path. Here are some common flags for
the Regent compiler.

  * `-fbounds-checks 1`: Enable bounds checks. These checks cover both
    region and array accesses. This should be a first step when
    debugging misbehaving Regent programs.
  * `-fpretty 1`: Enable pretty mode. While compiling, Regent will
    print the annotated source code of each task. This can be used to
    verify the optimizations that Regent is applying to the program.
  * `-fdebug 1`: Enable debug mode. Among other things, this causes
    Regent to attach a unique number to every symbol and region in the
    program. This can help to debug certain types of compile errors
    where otherwise the names can be difficult to disambiguate.
  * `-fcuda 1`: Enable CUDA support. By default, Regent will attempt
    to auto-detect CUDA, but if this fails, it will silently ignore
    the failure. With this flag, Regent will produce an error if CUDA
    support cannot be enabled.
  * `-fcuda-offline 1`: When CUDA auto-detection fails (e.g., if the
    compute node the compiler is running on does not have a working
    GPU), but the CUDA toolkit is still installed, use this flag to
    force Regent to use CUDA in "offline" mode. You will not be able
    to run with CUDA support directly, but this mode can be used to
    dump a CUDA binary that can be run on a machine with a working
    GPU.
  * `-fcuda-arch`: In offline mode, it is necessary to manually
    specify the architecture of the GPU that is being targeted (e.g.,
    `ampere`).

## Tests

To run the test suite, run:

```bash
./test.py [-h] [-j [THREAD_COUNT]] [-v]
```

For additional flags, run with `-h`.
