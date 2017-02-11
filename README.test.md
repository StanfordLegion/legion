# Legion Test Infrastructure

This document describes Legion's regression test infrastructure.

There are two key parts of Legion's test infrastructure:

  * The `test.py` script contains all the logic for running tests.
  * An automated CI framework is used to run this script on every commit.

This separation has the advantage the test suite itself should run
anywhere and ideally produce the same results (though environmental
factors may vary).

## Running Tests Manually

Everything runs through the main test script:

```
./test.py
```

By default, the script will choose an appropriate set of tests and
build options to run. The script will print its configuration at the
start so you can see what tests are enabled. There are two sets of
options that control tests and build flags respectively, which can
either be set on the command line or via environment variables.

Tests:

  * `--test=regent` or `TEST_REGENT`: Regent test suite
  * `--test=legion_cxx` or `TEST_LEGION_CXX`: Legion C++ examples and tests
  * `--test=fuzzer` or `TEST_FUZZER`: Legion fuzzer randomized tests
  * `--test=realm` or `TEST_REALM`: Realm tests
  * `--test=external` or `TEST_EXTERNAL`: Various external applications
  * `--test=private` or `TEST_PRIVATE`: Private external applications
  * `--test=perf` or `TEST_PERF`: Performance tests

Build flags:

  * `--debug` or `DEBUG`: Enable debug mode (disable with `DEBUG=0`)
  * `--use=gasnet` or `USE_GASNET`: Enable GASNet networking
  * `--use=cuda` or `USE_CUDA`: Enable CUDA for NVIDIA GPUs
  * `--use=llvm` or `USE_LLVM`: Enable LLVM support
  * `--use=hdf` or `USE_HDF`: Enable HDF5 I/O support
  * `--use=spy` or `USE_SPY`: Enable Legion Spy
  * `--use=cmake` or `USE_CMAKE`: Enable the CMake build system (vs Makefiles)
  * `--use=rdir` or `USE_RDIR`: Enable RDIR, a plugin for Regent

A note about command-line flags vs environment variables: flags are
exclusive, while variables are additive. That is, the following two
commands will do different things:

```
./test --test=regent
TEST_REGENT=1 ./test.py
```

The first will run **only** Regent tests. The second will run Regent
tests **in addition to** any other tests already enabled (in this
case, whatever defaults are chosen by the script). The same applies to
`--use` and `USE_` options as well.

## Automated Test Infrastructure
