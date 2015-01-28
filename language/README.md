# Legion Language Compiler

This directory contains the compiler for the Legion language. See
below for instructions for installing and running the compiler.

## Prerequisites

  * Terra-compatible LLVM installation on PATH (tested successfully
    with LLVM 3.4 and 3.5).
      * Or you can install Terra yourself; see below for instructions.

## Installing

Run the following command from the `language` directory:

    ./install.py [-h] [--debug] [--with-terra TERRA]

This command:

  * Downloads and builds Terra.
  * Builds a dynamic library with Legion and Lua bindings.
  * (OS X) Patches said dynamic library to avoid hard-coded absolute paths.
  * Sets everything up to run from `legion.py`.

Notes:

  * If you want a debug build Legion, pass `--debug` to `install.py`.
  * If you want to build with your own copy of Terra, pass the path to
    the Terra directory via the `--with-terra` flag. You will be
    responsible for building Terra yourself if you do this.

## Running

To use Legion, run:

    ./legion.py [SCRIPT]

This starts a Terra shell with the Legion dynamic library on
LD_LIBRARY_PATH (or DYLD_LIBRARY_PATH on Max OS X). From this shell,
you can either:

  * Use the Lua bindings via Lua/Terra.
  * Use the C bindings via Terra or LuaJIT FFI.
  * Use the language.

## Using

From the Legion shell, run the following command to import the
language:

    import "legion"

This imports the Legion compiler into scope and adds hooks to the
Legion compiler to the Terra parser so that you can start writing
Legion code.

## Tests

To run the test suite, run:

    ./test.py
