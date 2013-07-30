
# Legion Language Compiler

This directory contains the compiler for the Legion
language. Instructions for installing and running the compiler follow
below. For a tutorial on the language itself, see `doc/language.md`.

## Quickstart Instructions for Ubuntu

Open a terminal and run the following commands to install a C++
compiler, Git, and PLY:

```bash
sudo apt-get install build-essential git python-pip
sudo pip install ply
```

Checkout a copy of Legion:

```bash
git clone https://github.com/StanfordLegion/legion.git
cd legion/compiler
```

Create a file named `hello.lg` with the following contents:

``` {.legion #section1}
task main()
{
  let r = region<int>(1);
  let p = new<int@r>();
  *p = 0;
  *p = *p + 1;
  assert *p == 1;
}
```

Now run the compiler:

```bash
export LG_RT_DIR="$PWD/../runtime"
./lcomp hello.lg -o hello
./hello
```

With these instructions, the compiler will be unable to use C/C++ code
from Legion. See instructions below for installing the Clang bindings
for Python.

## Quickstart Instructions for Mac OS X

Install Xcode from the Mac OS X App Store. Once the installation is
complete, open Xcode and go to the menu Xcode > Preferences >
Downloads to install the Command Line Tools.

Open a terminal and run the following command to install PLY:

```bash
sudo easy_install pip
sudo pip install ply
```

Checkout a copy of Legion:

```bash
git clone https://github.com/StanfordLegion/legion.git
cd legion/compiler
```

Create a file named `hello.lg` with the following contents:

``` {.legion #section2}
task main()
{
  let r = region<int>(1);
  let p = new<int@r>();
  *p = 0;
  *p = *p + 1;
  assert *p == 1;
}
```

Now run the compiler:

```bash
export LG_RT_DIR="$PWD/../runtime"
./lcomp hello.lg -o hello
./hello
```

## Dependencies

The compiler requires at a minimum:

  * A working C++ compiler (e.g. G++ or Clang)

  * Python 2.7

      * Python 2.6 can be used as well, but requires a manual
        installation of the ordereddict library from
        <https://pypi.python.org/pypi/ordereddict>.

  * PLY <http://www.dabeaz.com/ply/>

  * The `LG_RT_DIR` environment variable must be set to the location
    of the `legion/runtime` directory.

The following is required for using C/C++ code from Legion:

  * Clang bindings for Python. Clang versions 3.2 and 3.3 have been
    tested successfully.

      * Binaries are available from <http://llvm.org/releases/>.

      * The Clang binaries in the Ubuntu package repository do not
        include `libclang.so`, which makes them unusable for the
        purposes of the compiler.

      * The Clang binaries in recent versions of Xcode for Mac OS X
        are sufficient for the purposes of the compiler. However, the
        Python bindings must still be installed separately (see
        below).

      * The Python bindings are included in the Clang source code,
        under `bindings/python`.

The following is required for building the documentation:

  * Pandoc <http://johnmacfarlane.net/pandoc/>

  * LaTeX (e.g. TeX Live <http://www.tug.org/texlive/>)

## Running the Compiler

The compiler frontend is called `lcomp`. The compiler supports the
following options:

`-h`: Prints a help message.

`-c`: Compiles to an object file.

`-S`: Compiles to a C++ source file.

Note: Without either `-c` or `-S` the compiler compiles to an
executable binary, built with debugging enabled and linked against the
shared low-level runtime.

`-o`: Specifies the name of the output file.

`-j`: Specifies the number of threads to use.

`--clean`: Rebuild the Legion runtime.

## Running the Testsuite

The test suite frontend is called `src/test.py`. The test suite
supports the following options:

`-h`: Prints a help message.

`-v`: Enables verbose output. This is most useful with `-j1`, because
otherwise the output gets mixed up.

`-j`: Specifies the number of threads to use.

`--clean`: Rebuild the Legion runtime.
