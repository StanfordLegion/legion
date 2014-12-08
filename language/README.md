# Legion Language Compiler

This directory contains the compiler for the Legion language. See
below for instructions for installing and running the compiler.

## Prerequisites

  * Terra-compatible LLVM installation on PATH (tested successfully
    with LLVM 3.4 and 3.5)
  * Boost including tools (Ubuntu packages libboost-dev and
    libboost-tools-dev, or equivalent)

## Installing

Run the following command from the `language` directory:

    ./install.py

This command:

  * Downloads and builds Terra
  * Downloads, patches and builds Luabind
  * Builds a dynamic library with Legion and Lua bindings
  * Sets everything up to run from `legion.py`

## Running

To use Legion, run:

    ./legion.py [script]

This starts a Terra shell with the Legion dynamic library on
LD_LIBRARY_PATH (or DYLD_LIBRARY_PATH on Max OS X). From this shell,
you can either:

  * Use the Luabind bindings via Lua/Terra
  * Use the C bindings via Terra or LuaJIT FFI
  * Use the language

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
