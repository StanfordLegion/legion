<!--
Copyright 2025 Stanford University, NVIDIA Corporation
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

[![ci](https://github.com/stanfordlegion/realm/actions/workflows/ci.yml/badge.svg)](https://github.com/StanfordLegion/realm/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/release/stanfordlegion/realm.svg?include_preleases)](https://github.com/StanfordLegion/realm/releases/latest)
[![Documentation](https://img.shields.io/badge/docs-grey.svg?logo=doxygen&logoColor=white&labelColor=blue)](https://legion.stanford.edu/realm/doc)
[![License](https://img.shields.io/github/license/stanfordlegion/realm.svg)](https://github.com/StanfordLegion/realm/blob/main/LICENSE.txt)
[![REUSE Compliance](https://img.shields.io/reuse/compliance/github.com%2FStanfordLegion%2Frealm)](https://api.reuse.software/info/github.com/StanfordLegion/realm)
[![codecov](https://codecov.io/github/stanfordlegion/realm/graph/badge.svg?token=XMFNJ4B756)](https://codecov.io/github/stanfordlegion/realm)
[![Issues](https://img.shields.io/github/issues/stanfordlegion/realm.svg)](https://github.com/StanfordLegion/realm/issues)
[![Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://legion.zulipchat.com)

[![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=fff)](#)
[![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](#)
[![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=F0F0F0)](#)
[![Windows](https://custom-icon-badges.demolab.com/badge/Windows-0078D6?logo=windows11&logoColor=white)](#)
[![ARM](https://img.shields.io/badge/ARM-white?logo=arm&logoColor=white&color=blue)](#)

# Realm

Realm is a distributed, **event–based tasking runtime** for building high-performance applications that span clusters of CPUs, GPUs, and other accelerators.  
It began life as the low-level substrate underneath the [Legion](https://github.com/StanfordLegion/legion) programming system but is now maintained as a standalone project for developers who want direct, fine-grained control of parallel and heterogeneous machines.

---

## Why Realm?

* **Asynchronous tasks & events** – Compose applications out of many light-weight tasks connected by events instead of blocking synchronization.
* **Heterogeneous execution** – Target CPUs, NVIDIA CUDA/HIP GPUs, OpenMP threads, and specialized fabrics with a single API.
* **Scalable networking** – Integrate GASNet-EX, UCX, MPI or shared memory transports for efficient inter-node communication.
* **Extensible modules** – Enable/disable features (CUDA, HIP, LLVM JIT, NVTX, PAPI …) at build time with simple CMake flags.
* **Portable performance** – Realm applications routinely scale from laptops to the world's largest supercomputers.

The runtime follows a *data-flow* execution model: tasks are launched asynchronously and start when their pre-condition events trigger. This design hides network and device latency, maximizes overlap, and gives programmers explicit control over when work becomes runnable.

For a deeper dive see the Realm white-paper published at PACT 2014:  
https://cs.stanford.edu/~sjt/pubs/pact14.pdf

---

## Quick start

### 1. Clone
```bash
git clone https://github.com/StanfordLegion/realm.git
cd realm
```

### 2. Build with CMake (recommended)
```bash
# Create an out-of-tree build directory
mkdir build && cd build
# Configure – pick the options that match your system
cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DREALM_ENABLE_OPENMP=ON \   # OpenMP support
      -DREALM_ENABLE_CUDA=OFF       # flip ON to target NVIDIA GPUs

# Compile everything
make -j$(nproc)

# (optional) run the unit tests
ctest --output-on-failure
```

The full list of CMake toggles is documented inside [`CMakeLists.txt`](CMakeLists.txt).  Common switches include:

| Option | Default | Purpose |
| ------ | ------- | ------- |
| `REALM_ENABLE_CUDA` | `ON`  | Build CUDA backend |
| `REALM_ENABLE_HIP`  | `ON`  | Build HIP/ROCm backend |
| `REALM_ENABLE_GASNETEX` | `ON on Linux` | GASNet-EX network |
| `REALM_ENABLE_UCX` | `ON on Linux` | UCX network |
| `REALM_ENABLE_MPI` | `OFF` | MPI network |
| `REALM_LOG_LEVEL`  | `WARNING` | Compile-time log level |

> **TIP:** combine `cmake -LAH` or `ccmake` to explore every option.

### 3. Install (optional)
```bash
make install   # honour DESTDIR / CMAKE_INSTALL_PREFIX as usual
```
Libraries, headers and CMake packages will be placed under `include/realm`, `lib/`, and `share/realm/` so that external projects can consume Realm via
```cmake
find_package(Realm REQUIRED)
```

### 4. Try the tutorials
The easiest way to get started is to build the tutorials as part of your normal CMake build tree:

```bash
# Configure (enable tutorials) if you did not already
cmake -B build -DREALM_BUILD_TUTORIALS=ON
cmake --build build --target realm_hello_world

# Run it (path will be inside the build directory)
./build/tutorials/hello_world/realm_hello_world -ll:cpu 4
```

If you have **installed** Realm (e.g. via `make install`) you can also build an individual tutorial stand-alone:

```bash
cd tutorials/hello_world         # Inside this repo or the copy installed under share/realm/tutorials
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/realm/install
cmake --build build
./build/realm_hello_world -ll:cpu 4
```

> Note • The tutorial directories only provide CMake build files. Traditional Makefiles are no longer shipped.

**Tutorials currently available**

* Hello World – minimal Realm program
* Events & Barriers – synchronization primitive
* Reservations - locks
* Reductions
* CUDA/HIP interoperability – calling GPU kernels from Realm tasks
* Profiling & Tracing – using `-lg:prof` and Legion Prof
* Machine Model Exploration – querying the machine model
* Index Space Operations – set algebra helpers for index spaces
* Region Instances – creating and using regional instances
* Copy ⁄ Fill – DMA-style data movement between instances
* Subgraph Launches – launching groups of tasks together
* Deferred Allocation – lazy allocation of physical memory
* Completion Queues – querying event completion programmatically

## Runtime command-line flags
Realm and its modules share a common set of `-ll:<flag>` options to tune processor/memory counts at runtime:

```
-ll:cpu <N>      # number of CPU cores per rank
-ll:gpu <N>      # number of GPUs per rank
-ll:util <N>     # number of util processors (communication helpers)
-ll:csize <MB>   # DRAM memory per rank
-ll:fsize <MB>   # framebuffer memory per GPU
-ll:zsize <MB>   # zero-copy (pinned) memory per GPU
-logfile <path>  # redirect logging (supports % for rank)
-level <cat>=<n> # change logging level per category
```
Run any Realm executable with `-hl:help` (high-level) or `-ll:help` (low-level) to see everything that is available.

---

## Documentation
* Current public documentation can be found [here](https://legion.stanford.edu/realm/doc/main)
* **API reference (Doxygen):** generate with `make docs` or `cmake --build . --target docs`.
* **Tutorials:** see the [`tutorials/`](tutorials) directory listed above.
* **Examples & Benchmarks:** under [`examples/`](examples) and [`benchmarks/`](benchmarks).

Please file an issue or pull request if something is missing or outdated.

---

## Contributing
We welcome contributions of all kinds – bug reports, documentation fixes, new features, and performance improvements.

1. Fork the repository and create a feature branch.
2. Follow the existing code style (clang-format is enforced in CI).
3. Make sure `ctest` passes on your machine *and* with `REALM_ENABLE_SANITIZER` if possible.
4. Open a pull request against `master` (or the feature branch you were asked to use).

See [`CONTRIBUTING.md`](.github/CONTRIBUTING.md) for the full guidelines.

---

## License

Realm is licensed under the **Apache License 2.0** – see [`LICENSE.txt`](LICENSE.txt) for details.

Commercial and academic use is free; attribution in papers and derivative works is appreciated.

---

## Acknowledgements
Realm is developed and maintained by the [Stanford Legion](https://legion.stanford.edu) team with significant contributions from NVIDIA, Los Alamos, Livermore, Sandia, and many members of the broader HPC community.
