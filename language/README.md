# Regent Compiler

This directory contains the compiler for the [Regent
language](http://regent-lang.org/).

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

## Quickstart for macOS

```bash
# install XCode command-line tools
sudo xcode-select --install

# download CMake
curl -L -O https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2-macos-universal.tar.gz
tar xfz cmake-3.22.2-macos-universal.tar.gz
export PATH="$PATH:$PWD/cmake-3.22.2-macos-universal/CMake.app/Contents/bin"

# download LLVM
curl -L -O https://github.com/terralang/llvm-build/releases/download/llvm-13.0.0/clang+llvm-13.0.0-x86_64-apple-darwin.tar.xz
tar xfJ clang+llvm-13.0.0-x86_64-apple-darwin.tar.xz
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$PWD/clang+llvm-13.0.0-x86_64-apple-darwin"

# environment variables needed to build/run Regent
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
export CXXFLAGS="-std=c++11"

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
  * *Optional (but strongly recommended)*: CMake 3.16 or newer

Regent also has a number of transitive dependencies via Legion:

  * Linux, macOS, or another Unix
  * A C++ 11 or newer compiler (GCC, Clang, Intel, or PGI) and GNU Make
  * *Optional*: CUDA 10.0 or newer (for NVIDIA GPUs)
  * *Optional*: [GASNet](https://gasnet.lbl.gov/) (for networking)
  * *Optional*: HDF5 (for file I/O)

## Installing Regent on a Personal Machine with `install.py`

There are two ways to install Regent. The first, `install.py`, assumes
that (most) dependencies are pre-installed (e.g., through a package
manager). The exception is Terra, which is downloaded automatically by
the script.

On supercomputers (which often do not come with these packages), the
`setup_env.py` script is recommended (see below).

From the `legion/language` directory:

```bash
./install.py [-h] [--debug] [--gasnet] [--cuda] [--openmp] [--python] [--hdf5]
             [--rdir {prompt,auto,manual,skip,never}] [--with-terra DIR]
             [-j [THREAD_COUNT]]
```

Run with `-h` to get a full list of available flags.

Notes:

  * Use `--debug` to enable Legion's debug mode. Debug mode includes
    significantly more safety checks and better error messages, but
    also runs significantly slower than release mode.
  * For CUDA support, use `--cuda`. This requires that CUDA be
    pre-installed and the location of CUDA be specified via
    `CUDA_HOME` or `CUDA` environment variables.
  * For GASNet (networking) support, use `--gasnet`. This requires
    that GASNet be pre-installed and the location of GASNet be
    specified via the `GASNET_ROOT` or `GASNET` environment variables.
  * RDIR is an optional (but recommended) plugin for Regent; this is
    required for Regent's static control replication
    optimization. Specifying `--rdir=auto` will instruct the installer to
    download and manage RDIR automatically.
  * If you prefer to build your own copy of Terra, pass the path to
    the Terra directory via the `--with-terra` flag. You will be
    responsible for building Terra yourself if you do this.

## Installing Regent on a Supercomputer via `setup_env.py`

The `setup_env.py` script is our main way of deploying Regent on
supercomputers. This script has built-in support for identifying
commonly used machines that we run on, and also builds all
dependencies. In contrast to `install.py`, GASNet is also enabled by
default. CUDA must still be enabled manually. Note that when using
`setup_env.py`, some of options differ from `install.py`.

From the `legion/language` directory:

```bash
./scripts/setup_env.py
```

There are a number of variables that you may need to pass in addition
to this, depending on your configuration:

  * On Cray machines, the C and C++ compilers are offered via wrappers
    `cc` and `CC`. These wrappers cannot be used to build all Regent
    dependencies. Therefore, on Cray systems, we require that the
    non-wrapped host compilers be passed via `HOST_CC` and `HOST_CXX`
    environment variables. We generally recommend using GCC as the
    host compiler, which would mean loading the `PrgEnv-gnu` module
    and then passing `gcc` and `g++` as `HOST_CC` and `HOST_CXX`.
  * By default the script will build LLVM 13.0. If for some reason
    another version is required, this can be specified via the
    `--llvm-version` flag.
  * For CUDA support, set `USE_CUDA` to `1`. Make sure that CUDA can
    be located through the `CUDA_HOME` environment variable.
  * Similarly for HIP set `USE_HIP` to `1` and make sure HIP is
    available via `HIP_PATH`. Please note that `HIP_ARCH` must be set
    to the AMD GPU architecture (e.g., `gfx90a`).
  * GASNet is enabled by default, and the network conduit will be
    automatically identified on common machines that we use with
    Regent. If your machine is not one of these, you may need to
    specify the `CONDUIT` environment variable (e.g, `ibv` for
    Infiniband, `aries` for Cray Aries, and `ofi` for HPE
    Slingshot). If you prefer to disable GASNet (e.g., for single-node
    runs), set `USE_GASNET` to `0`.
  * The script will enable RDIR in `auto` mode by default.
  * Unlike `install.py` the `setup_env.py` script will auto-detect the
    number of available CPU cores on the machine to parallelize the
    build.

Additional flags and options can be queried via the `-h` flag.

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
  * `-fgpu cuda`: Enable CUDA support. By default, Regent will attempt
    to auto-detect CUDA, but if this fails, it will silently ignore
    the failure. With this flag, Regent will produce an error if CUDA
    support cannot be enabled.
  * `-fgpu hip`: Enable HIP support. Currently Regent does not attempt
    to auto-detect HIP.
  * `-fgpu-arch`: Specify the GPU architecture. Note that this is
    auto-detected by default on NVIDIA GPUs, but must be manually
    specified for AMD GPUs. Example values include `ampere` (for
    NVIDIA) and `gfx90a` (for AMD). The GPU architecture can also be
    specified via the `GPU_ARCH` environment variable.
  * `-fgpu-offline 1`: When a GPU is not available on the current
    node, it is still possible to build GPU programs via this flag as
    long as the appropriate compiler toolchain is installed. Note that
    GPU architecture is mandatory in this mode (even with CUDA).

## Tests

To run the test suite, run:

```bash
./test.py [-h] [-j [THREAD_COUNT]] [-v]
```

For additional flags, run with `-h`.
