# Regent Compiler

This directory contains the compiler for the Regent language. See
below for instructions for installing and running the compiler.

## Prerequisites

  * Linux, OS X, or another Unix.
  * Python >= 2.7 (for the self-installer and test suite).
  * LLVM and Clang **with headers** (as of December 2015 LLVM 3.5 is
    recommended; 3.6 works but is missing debug symbols).

## Installing

Run the following command from the `language` directory:

    ./install.py [-h] [--with-terra TERRA] [--debug] [--general] [--gasnet]
                 [--cuda] [-j [THREAD_COUNT]]

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

    ./regent.py [SCRIPT]

This starts the Regent frontend with the Legion dynamic library on
`LD_LIBRARY_PATH` (or `DYLD_LIBRARY_PATH` on OS X).

## Using

From the Regent shell, run the following command to import the
language:

    import "regent"

This imports the Regent compiler into scope and adds hooks to the
Regent compiler to the Terra parser so that you can start writing
Regent code.

## Tests

To run the test suite, run:

    ./test.py [-h] [-j [THREAD_COUNT]] [-v]
