# Copyright 2025 Stanford University, NVIDIA Corporation, Los Alamos National Laboratory
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

#region Versioning

set(Realm_SHORT_VERSION "")
set(Realm_VERSION_META "")

find_package(Git)
if(Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" describe --tags --abbrev=0
    OUTPUT_VARIABLE REALM_GIT_VERSION
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short=8 HEAD
    OUTPUT_VARIABLE REALM_VERSION_SHA
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" diff-files --quiet
    RESULT_VARIABLE REALM_DIRTY
    ERROR_QUIET
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  )
  if(REALM_GIT_VERSION MATCHES "legion-([0-9\.]+)")
    string(REGEX REPLACE "legion-([0-9\.]+)" "\\1" Realm_SHORT_VERSION "${REALM_GIT_VERSION}")
  elseif(REALM_GIT_VERSION MATCHES "v([0-9\.]+)(-.+)?")
    string(REGEX MATCH "v([0-9\.]+)(-.+)?" _ "${REALM_GIT_VERSION}")
    set(Realm_SHORT_VERSION ${CMAKE_MATCH_1})
    set(Realm_VERSION_META ${CMAKE_MATCH_2})
  endif()
  set(Realm_VERSION_META "${Realm_VERSION_META}+${REALM_VERSION_SHA}")
  if(REALM_DIRTY)
    string(APPEND Realm_VERSION_META "-dirty")
  endif()
endif()

if(Realm_SHORT_VERSION STREQUAL "" AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION _REALM_FULL_VERSION)
  string(STRIP "${_REALM_FULL_VERSION}" _REALM_FULL_VERSION)
  string(REGEX MATCH "v([0-9\.]+)(-.+)?" _ "${_REALM_FULL_VERSION}")
  if(CMAKE_MATCH_1)
    set(Realm_SHORT_VERSION ${CMAKE_MATCH_1})
    set(Realm_VERSION_META ${CMAKE_MATCH_2})
  endif()
endif()
if(Realm_SHORT_VERSION STREQUAL "")
  set(Realm_SHORT_VERSION 0.0.1)
  if(Realm_VERSION_META STREQUAL "")
    set(Realm_VERSION_META "-unknown")
  endif()
endif()
message("-- Realm version: ${Realm_SHORT_VERSION}${Realm_VERSION_META}")
#endregion

project(
  Realm
  VERSION ${Realm_SHORT_VERSION}
  LANGUAGES CXX C
  DESCRIPTION "A distributed, event-based tasking library"
  HOMEPAGE_URL https://legion.stanford.edu/realm
)
set(REALM_VERSION "${Realm_SHORT_VERSION}${Realm_VERSION_META}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#region External built-ins
include(GenerateExportHeader)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckIPOSupported)
include(CheckSymbolExists)
include(CheckCXXCompilerFlag)
include(CMakePushCheckState)
include(CMakeDependentOption)
include(ExternalProject)
include(FetchContent)
include(FeatureSummary)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake")

check_ipo_supported(RESULT REALM_SUPPORTS_LTO LANGUAGES CXX C)
#endregion

#region Depedency auto-detection
find_package(CUDAToolkit 11.7 QUIET)
find_package(Doxygen QUIET)
find_package(hip QUIET CONFIG)
find_package(HDF5 QUIET)
find_package(OpenMP QUIET COMPONENTS C CXX)
find_package(Papi QUIET)
find_package(Python3 QUIET COMPONENTS Development)

# Only enable CUDA automatically if driver is also available.
if(CUDAToolkit_FOUND AND TARGET CUDA::cuda_driver)
  set(REALM_CUDA_AUTO_DETECT ON)
else()
  set(REALM_CUDA_AUTO_DETECT OFF)
endif()
#endregion

#region Build options

# These should all be prefixed with 'REALM_' to prevent subproject name clashes
set(REALM_MAX_DIM
    3
    CACHE STRING "Maximum dimension compiled for"
)
if(BUILD_SHARED_LIBS)
  set(REALM_LIMIT_SYMBOL_VISIBILITY 1)
endif()
set(REALM_CXX_STANDARD
    17
    CACHE STRING "C++ standard used"
)
set(REALM_LOG_LEVEL
    "WARNING"
    CACHE STRING "Compile-time logging level"
)
set_property(
  CACHE REALM_LOG_LEVEL PROPERTY STRINGS "SPEW;DEBUG;INFO;PRINT;WARNING;ERROR;FATAL;NONE"
)

option(REALM_ENABLE_INSTALL "Enable installation support" ${PROJECT_IS_TOP_LEVEL})
option(REALM_BUILD_TESTS "Build tests" ${PROJECT_IS_TOPLEVEL})
option(REALM_BUILD_EXAMPLES "Build examples" ${PROJECT_IS_TOPLEVEL})
option(REALM_BUILD_BENCHMARKS "Build benchmarks" ${PROJECT_IS_TOPLEVEL})
option(REALM_BUILD_TUTORIALS "Build tutorials" ${PROJECT_IS_TOPLEVEL})
if(Doxygen_FOUND)
  option(REALM_BUILD_DOCS "Build documentation" ${PROJECT_IS_TOPLEVEL})
else()
  option(REALM_BUILD_DOCS "Build documentation" OFF)
endif()

cmake_dependent_option(REALM_ENABLE_RDTSC "Compile with RDTSC support" ON "NOT WIN32" OFF)
option(REALM_ENABLE_CUDA "Compile with CUDA support" ${REALM_CUDA_AUTO_DETECT})
option(REALM_ENABLE_HIP "Compile with HIP support" ${hip_FOUND})
option(REALM_ENABLE_LLVM "Compile with LLVMJIT support" OFF)
option(REALM_ENABLE_HDF5 "Compile with HDF5 support" ${HDF5_FOUND})
option(REALM_ENABLE_MPI "Compile with MPI support" OFF)
option(REALM_ENABLE_OPENMP "Compile with OpenMP support" ${OpenMP_FOUND})
option(REALM_ENABLE_KOKKOS "Compile with kokkos support" OFF)
cmake_dependent_option(REALM_ENABLE_UCX "Compile with UCX support" OFF "NOT WIN32" OFF)
cmake_dependent_option(
  REALM_INSTALL_UCX_BOOTSTRAPS "Install UCX bootstraps" ON "REALM_ENABLE_INSTALL; REALM_ENABLE_UCX"
  OFF
)
option(REALM_ENABLE_HWLOC "Compile with hwloc hardware discovery support" OFF)
option(REALM_ENABLE_PAPI "Compile with PAPI thread profiling support" ${Papi_FOUND})
option(REALM_ENABLE_CPPTRACE "Compile with cpptrace backtrace support" OFF)
option(REALM_ENABLE_PREALM "Compile with realm profiler" OFF)
cmake_dependent_option(
  REALM_ENABLE_LIBDL "Compile with dynamic runtime linking support" ON "NOT WIN32" OFF
)
cmake_dependent_option(REALM_ENABLE_AIO "Compile with AIO transfer support" ON "NOT WIN32" OFF)
cmake_dependent_option(REALM_ENABLE_GASNETEX "Compile with GASNetEx support" OFF "NOT WIN32" OFF)
cmake_dependent_option(
  REALM_ENABLE_GASNETEX_WRAPPER "Compile with dynamic GASNetEx ABI stability wrapper" OFF
  "NOT WIN32; REALM_ENABLE_LIBDL" OFF
)
cmake_dependent_option(
  REALM_INSTALL_GASNETEX_WRAPPER "Compile and install the dynamic GASNetEx ABI stability wrapper" OFF
  "REALM_ENABLE_GASNETEX_WRAPPER" OFF
)
cmake_dependent_option(
  REALM_ENABLE_PYTHON "Compile with Python support" ${Python3_FOUND} "REALM_ENABLE_LIBDL" OFF
)

cmake_dependent_option(
  REALM_ENABLE_COVERAGE "Enable coverage reports" OFF
  "REALM_BUILD_TESTS;CMAKE_BUILD_TYPE STREQUAL Debug" OFF
)
cmake_dependent_option(
  REALM_ENABLE_LTO "Enable link-time optimization" ON
  "REALM_SUPPORTS_LTO;CMAKE_BUILD_TYPE STREQUAL Release" OFF
)
cmake_dependent_option(
  REALM_OPENMP_SYSTEM_RUNTIME "Use system openmp, rather than Realm's wrapper" OFF
  "REALM_ENABLE_OPENMP" OFF
)
cmake_dependent_option(
  REALM_OPENMP_GOMP_SUPPORT "Enable GOMP support for Realm's wrapper OpenMP" ON
  "REALM_ENABLE_OPENMP; NOT REALM_OPENMP_SYSTEM_RUNTIME" OFF
)
cmake_dependent_option(
  REALM_OPENMP_KMP_SUPPORT "Enable KMP support for Realm's wrapper OpenMP" ON
  "REALM_ENABLE_OPENMP; NOT REALM_OPENMP_SYSTEM_RUNTIME" OFF
)
cmake_dependent_option(REALM_ENABLE_NVTX "Compile with NVTX support" ON "REALM_ENABLE_CUDA" OFF)
cmake_dependent_option(
  REALM_CUDA_DYNAMIC_LOAD "Dynamically load CUDA rather than link against it" ON
  "REALM_ENABLE_CUDA; REALM_ENABLE_LIBDL" OFF
)
cmake_dependent_option(
  REALM_ENABLE_SHM "Enable shared memory support for inter-process communication" ON
  "UNIX; NOT APPLE" OFF
)
cmake_dependent_option(
  REALM_ENABLE_UCC "Enable UCC for UCX networking module collectives" ON "REALM_ENABLE_UCX" OFF
)
set(REALM_SANITIZER
    "NONE"
    CACHE STRING "Enabled sanitizer"
)
set_property(CACHE REALM_SANITIZER PROPERTY STRINGS "NONE;ASAN;UBSAN;TSAN")
#endregion

set(REALM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/realm")
cmake_path(ABSOLUTE_PATH REALM_SOURCE_DIR NORMALIZE)

#region Dependencies
set(REALM_STATIC_DEPENDS)
set_package_properties(CUDAToolkit PROPERTIES TYPE RECOMMENDED)
set_package_properties(hip PROPERTIES TYPE OPTIONAL)
set_package_properties(LLVM PROPERTIES TYPE OPTIONAL)
set_package_properties(HDF5 PROPERTIES TYPE OPTIONAL)
set_package_properties(MPI PROPERTIES TYPE OPTIONAL)
set_package_properties(OpenMP PROPERTIES TYPE OPTIONAL)
set_package_properties(Kokkos PROPERTIES TYPE OPTIONAL)
set_package_properties(ucx PROPERTIES TYPE OPTIONAL)
set_package_properties(ucc PROPERTIES TYPE RECOMMENDED)
set_package_properties(GASNet PROPERTIES TYPE OPTIONAL)
set_package_properties(HWLOC PROPERTIES TYPE OPTIONAL)
set_package_properties(PAPI PROPERTIES TYPE OPTIONAL)
set_package_properties(
  cpptrace PROPERTIES
  TYPE RECOMMENDED
  PURPOSE "To display backtraces"
)

if (REALM_USE_CUDART_HIJACK)
  message(FATAL_ERROR "CUDART HIJACK is no longer supported")
endif()

if (REALM_USE_HIP_HIJACK)
  message(FATAL_ERROR "HIP HIJACK is no longer supported")
endif()

# TODO(cperry): rename with REALM_ prefix
set(COMPILE_TIME_MIN_LEVEL "LEVEL_${REALM_LOG_LEVEL}")
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(DEBUG_REALM 1)
endif()

if (NOT MAKE_EXECUTABLE MATCHES ".*make.*")
  find_program(MAKE_EXECUTABLE NAMES make gnumake smake gmake)
else()
  set(MAKE_EXECUTABLE ${CMAKE_MAKE_PROGRAM})
endif()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
list(APPEND REALM_LIBRARIES Threads::Threads)
list(APPEND REALM_STATIC_DEPENDS Threads)

if(UNIX AND NOT APPLE)
  find_library(RT_LIBS rt)
  if(RT_LIBS)
    list(APPEND REALM_LIBRARIES ${RT_LIBS})
  endif()
endif()

if(REALM_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
endif()

if(REALM_ENABLE_AIO)
  # TODO(cperry): REALM_USE_KERNEL_AIO bitrotted,
  # the default has been LIBAIO for a while
  if(NOT REALM_USE_KERNEL_AIO)
    cmake_push_check_state()
    if(RT_LIBS)
      list(APPEND CMAKE_REQUIRED_LIBRARIES ${RT_LIBS})
    endif()
    check_symbol_exists(aio_read aio.h REALM_USE_LIBAIO)
    cmake_pop_check_state()
  endif()
endif()

if(REALM_ENABLE_SHM)
  cmake_push_check_state()
  if(RT_LIBS)
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${RT_LIBS})
  endif()
  check_symbol_exists(shm_open sys/mman.h REALM_USE_SHM)
  cmake_pop_check_state()
endif()

if(UNIX)
  check_symbol_exists(posix_fallocate64 "fcntl.h;unistd.h" REALM_HAS_POSIX_FALLOCATE64)
endif()

if(REALM_ENABLE_LIBDL)
  list(APPEND REALM_LIBRARIES ${CMAKE_DL_LIBS})
endif()

if(REALM_ENABLE_CPPTRACE)
  cpmfindpackage(
    NAME
    cpptrace
    GIT_REPOSITORY
    https://github.com/jeremy-rifkin/cpptrace.git
    GIT_TAG
    v0.7.3
    EXCLUDE_FROM_ALL
    YES
    SYSTEM
    YES
    OPTIONS
    "CMAKE_SKIP_INSTALL_RULES OFF"
    "BUILD_SHARED_LIBS OFF"
  )
  set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND cpptrace)
  get_property(_NOT_FOUND GLOBAL PROPERTY PACKAGES_NOT_FOUND)
  list(REMOVE_ITEM _NOT_FOUND cpptrace)
  set_property(GLOBAL PROPERTY PACKAGES_NOT_FOUND "${_NOT_FOUND}")
  list(APPEND REALM_STATIC_DEPENDS cpptrace)
  list(APPEND REALM_LIBRARIES cpptrace::cpptrace)
endif()

if(REALM_ENABLE_HWLOC)
  find_package(HWLOC REQUIRED)
  list(APPEND REALM_STATIC_DEPENDS HWLOC)
  list(APPEND REALM_LIBRARIES HWLOC::HWLOC)
endif()

if(REALM_ENABLE_PAPI)
  find_package(Papi REQUIRED)
  list(APPEND REALM_STATIC_DEPENDS Papi)
  list(APPEND REALM_LIBRARIES Papi::Papi)
endif()

if(REALM_ENABLE_COVERAGE)
  if(MSVC)
    find_program(
      OPENCPPCOV_PATH OpenCppCoverage.exe REQUIRED HINTS $ENV{ProgramFiles\(x86\)}/OpenCppCoverage
                                                    $ENV{ProgramFiles}/OpenCppCoverage
    )
  else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CodeCoverage.cmake)
  endif()
endif()

if(REALM_ENABLE_CUDA OR (REALM_ENABLE_HIP AND CMAKE_HIP_PLATFORM STREQUAL "nvidia"))
  find_package(CUDAToolkit 11.7 REQUIRED)
  if(CUDAToolkit_FOUND)
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
      set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
    endif()
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      execute_process(
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} --list-gpu-arch OUTPUT_VARIABLE compute_archs
      )
      string(REPLACE "compute_" "" compute_archs "${compute_archs}")
      string(REPLACE "\n" ";" compute_archs "${compute_archs}")
      list(FILTER compute_archs EXCLUDE REGEX "^$")
      if (REALM_MIN_CUDA_ARCH)
        set(filtered_compute_archs)
        foreach(arch IN LISTS compute_archs)
          if (arch GREATER_EQUAL REALM_MIN_CUDA_ARCH)
            list(APPEND filtered_compute_archs ${arch})
          endif()
        endforeach()
        set(compute_archs ${filtered_compute_archs})
      endif()
      list(SORT compute_archs COMPARE NATURAL)
      list(GET compute_archs -1 newest_compute_arch)
      list(TRANSFORM compute_archs APPEND "-real")
      list(APPEND compute_archs ${newest_compute_arch}-virtual)
      set(CMAKE_CUDA_ARCHITECTURES ${compute_archs})
    endif()
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

    if(NOT CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
    endif()

    # cmake's CUDA compiler component statically defines known standards rather than querying for them dynamically
    # so, older cmake versions will error if newer unknown cuda standards are specified, even though
    # nvcc supports them.  So trick cmake's CUDA compiler component into thinking it does have the specified
    # cuda standard and pass in the necessacary nvcc flag ourselves.
    # See https://gitlab.kitware.com/cmake/cmake/-/issues/23079
    if(CMAKE_CUDA_STANDARD AND NOT DEFINED CMAKE_CUDA${CMAKE_CUDA_STANDARD}_STANDARD_COMPILE_OPTION)
      set(CMAKE_CUDA${CMAKE_CUDA_STANDARD}_STANDARD_COMPILE_OPTION "")
      set(CMAKE_CUDA${CMAKE_CUDA_STANDARD}_EXTENSION_COMPILE_OPTION "")
      list(APPEND CMAKE_CUDA_FLAGS "-std=c++${CMAKE_CUDA_STANDARD}")
    endif()

    if(REALM_ENABLE_CUDA)
      enable_language(CUDA)
    endif()
    get_filename_component(CUDAToolkit_LIBRARY_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY CACHE)
    # Unfortunately, FindCUDAToolkit doesn't always find CUPTI due to directory changes in the
    # toolkit, so correct it here
    find_path(
      CUDAToolkit_CUPTI_INCLUDE_DIR cupti.h
      PATHS ${CUDAToolkit_LIBRARY_ROOT}
      PATH_SUFFIXES include ../include extras/CUPTI/include ../extras/CUPTI/include
    )

    # Deal with the library name change in older versions of cmake.
    #
    # nvtx3 was added in cmake 3.25, and while later versions (at least 3.30+) claim that
    # the CUDA::nvtx3 and CUDA::nvToolsExt targets are mutually exclusive, they don't
    # appear to be so in earlier versions. So if we have nvtx3, prefer that, otherwise
    # fall back to nvToolsExt.
    set(REALM_USE_NVTX ${REALM_ENABLE_NVTX})
    if(TARGET CUDA::nvtx3)
      list(APPEND REALM_LIBRARIES CUDA::nvtx3)
    elseif(TARGET CUDA::nvToolsExt)
      list(APPEND REALM_LIBRARIES CUDA::nvToolsExt)
    else()
      # Silently disable NVTX if we cannot find it
      set(REALM_USE_NVTX OFF)
    endif()

    if(NOT REALM_CUDA_DYNAMIC_LOAD)
      list(APPEND REALM_LIBRARIES CUDA::cuda_driver)
    endif()
    list(APPEND REALM_STATIC_DEPENDS CUDAToolkit)
  endif()
endif()

if(REALM_ENABLE_HIP)
  if(NOT DEFINED ROCM_PATH)
    if(NOT DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH
          "/opt/rocm"
          CACHE PATH "Path to which HIP has been installed"
      )
    else()
      set(ROCM_PATH
          $ENV{ROCM_PATH}
          CACHE PATH "Path to which HIP has been installed"
      )
    endif()
  endif()
  list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
  set(HIP_PLATFORM ${CMAKE_HIP_PLATFORM})
  find_package(hip REQUIRED CONFIG)
  if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
    # nvidia HIP_PLATFORM requies 3.28+ for enable_language(HIP) to work properly
    cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
    set_target_properties(
      hip::host PROPERTIES INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_NVIDIA__"
                           INTERFACE_LINK_LIBRARIES CUDA::cudart
    )
    set(CMAKE_HIP_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
  endif()
  if(NOT HIP_ROOT_DIR)
    set(HIP_ROOT_DIR "${ROCM_PATH}")
  endif()
  set(CMAKE_HIP_COMPILER_ROCM_ROOT "${HIP_ROOT_DIR}")
  if(hip_FOUND)
    list(APPEND REALM_STATIC_DEPENDS hip)
    list(APPEND REALM_LIBRARIES hip::host)
    # Deal with an issue on some hip configs where the include directory is not
    # set in the interface
    get_target_property(HIPHOST_INC_DIR hip::host INTERFACE_INCLUDE_DIRECTORIES)
    if(HIPHOST_INC_DIR STREQUAL "HIPHOST_INC_DIR-NOTFOUND")
      set_target_properties(hip::host PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${HIP_INCLUDE_DIR})
      #message(FATAL_ERROR ${HIP_INCLUDE_DIR})
    endif()
    enable_language(HIP)
  endif()
endif()

if(REALM_ENABLE_MPI)
  find_package(MPI REQUIRED)
  cmake_push_check_state()
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${MPI_mpi_LIBRARY})
  list(APPEND CMAKE_REQUIRED_INCLUDES ${MPI_C_INCLUDE_DIRS})
  check_symbol_exists(MPI_Comm_split_type "mpi.h" REALM_MPI_HAS_COMM_SPLIT_TYPE)
  cmake_pop_check_state()
  list(APPEND REALM_STATIC_DEPENDS MPI)
  list(APPEND REALM_LIBRARIES MPI::MPI_CXX)
endif()

if(REALM_ENABLE_OPENMP)
  if(REALM_OPENMP_SYSTEM_RUNTIME)
    find_package(OpenMP REQUIRED COMPONENTS C CXX)
    list(APPEND REALM_STATIC_DEPENDS OpenMP)
    list(APPEND REALM_LIBRARIES OpenMP::OpenMP_CXX)
  endif()
endif()

if(REALM_ENABLE_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Development)
  list(APPEND REALM_STATIC_DEPENDS Python3)
endif()

if(REALM_ENABLE_KOKKOS)
  # Unfortunately, Kokkos really doesn't like to be used with CPM or anything
  # when cuda is enabled.  Only certain configuration options are available
  # when you use it that way and the build environment to get kokkos working
  # with cuda, hip, etc needs to be very finely tuned.  So we give up and ask
  # the user to install it themselves.
  find_package(Kokkos REQUIRED COMPONENTS separable_compilation)
  enable_language(${Kokkos_COMPILE_LANGUAGE})

  if(NOT BUILD_SHARED_LIBS AND "${Kokkos_COMPILE_LANGUAGE}" STREQUAL CXX)
    include(CheckSourceCompiles)
    include(CMakePushCheckState)

    cmake_push_check_state(RESET)
      set(CMAKE_REQUIRED_LIBRARIES Kokkos::kokkoscore)
      unset(REALM_${Kokkos_COMPILE_LANGUAGE}_COMPILER_CAN_COMPILE_KOKKOS CACHE)
      check_source_compiles(
          ${Kokkos_COMPILE_LANGUAGE} "
#include <Kokkos_Core.hpp>
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*> v(\"v\", 1);
    (void)v;
  }
  Kokkos::finalize();
  return 0;
}
" REALM_${Kokkos_COMPILE_LANGUAGE}_COMPILER_CAN_COMPILE_KOKKOS)
    cmake_pop_check_state()

    if(NOT ${REALM_${Kokkos_COMPILE_LANGUAGE}_COMPILER_CAN_COMPILE_KOKKOS})
      message(FATAL_ERROR "Current Realm ${Kokkos_COMPILE_LANGUAGE} compiler will not work with Kokkos during STATIC build.")
    endif()
  endif()

  list(APPEND REALM_PUBLIC_LIBRARIES realm_kokkos)
endif()

if(REALM_ENABLE_LLVM)
  find_package(LLVM REQUIRED)
  list(APPEND REALM_STATIC_DEPENDS LLVM)
  list(APPEND REALM_LIBRARIES LLVM::LLVM)
endif()

if(REALM_ENABLE_HDF5)
  find_package(HDF5 REQUIRED)
  list(APPEND REALM_STATIC_DEPENDS HDF5)
  list(APPEND REALM_LIBRARIES hdf5::hdf5)
endif()

if(REALM_ENABLE_PREALM)
  find_package(ZLIB REQUIRED)
  list(APPEND REALM_STATIC_DEPENDS ZLIB)
  list(APPEND REALM_LIBRARIES ZLIB::ZLIB)
endif()

if(REALM_ENABLE_UCX)
  find_package(ucx QUIET CONFIG)
  # TODO(cperry): UCX compilation on OSX currently runs into the following issue:
  # https://github.com/openucx/ucx/issues/9
  if(NOT ucx_FOUND AND NOT APPLE)
    list(
      APPEND
      UCX_CONFIG_ARGS
      "--without-java"
      "--without-go"
      "--without-iodemo-cuda"
      "--with-pic"
      "--disable-doxygen-doc"
      "--enable-static"
    )
    if(hip_FOUND)
      list(APPEND UCX_CONFIG_ARGS "--with-rocm=${HIP_ROOT_DIR}")
    endif()
    if(CUDAToolkit_FOUND)
      list(APPEND UCX_CONFIG_ARGS "--with-cuda=${CUDAToolkit_LIBRARY_ROOT}")
    endif()
    set(ucx_ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucx-install")
    set(ucx_VERSION 1.20.1)
    # cmake-format: off
    ExternalProject_Add(
      ucx_external
      URL "https://github.com/openucx/ucx/releases/download/v1.20.1/ucx-1.20.1.tar.gz"
      URL_HASH SHA256=545c419a7b5e04643cb8bff5a19b3b5071a8f8f0605f1e8efb36f8f3d7bfb9d3
      SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucx"
      INSTALL_DIR "${ucx_ROOT_DIR}"
      CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
      contrib/configure-release-mt "--prefix=<INSTALL_DIR>" ${UCX_CONFIG_ARGS}
      BUILD_IN_SOURCE TRUE
      BUILD_JOB_SERVER_AWARE TRUE
      BUILD_BYPRODUCTS <INSTALL_DIR>/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ucp${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
    # cmake-format: on
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_deps/ucx-install/include)
    add_library(ucp SHARED IMPORTED)
    set_target_properties(
      ucp
      PROPERTIES
        IMPORTED_LOCATION
        "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucx-install/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ucp${CMAKE_SHARED_LIBRARY_SUFFIX}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucx-install/include"
    )
    add_dependencies(ucp ucx_external)
    add_library(ucx::ucp ALIAS ucp)
    set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND ucx)
    get_property(_NOT_FOUND GLOBAL PROPERTY PACKAGES_NOT_FOUND)
    list(REMOVE_ITEM _NOT_FOUND ucx)
    set_property(GLOBAL PROPERTY PACKAGES_NOT_FOUND "${_NOT_FOUND}")
  endif()
  list(APPEND REALM_STATIC_DEPENDS ucx)
  list(APPEND REALM_LIBRARIES ucx::ucp)

  if(REALM_ENABLE_UCC AND TARGET ucx::ucp)
    find_package(ucc QUIET CONFIG)
    if(NOT ucc_FOUND)
      list(APPEND UCC_CONFIG_ARGS "--with-ucx=${ucx_ROOT_DIR}" "--disable-doxygen-doc" "--with-pic"
           "--enable-static"
      )
      if(hip_FOUND)
        list(APPEND UCC_CONFIG_ARGS "--with-rocm=${HIP_ROOT_DIR}")
      endif()
      if(CUDAToolkit_FOUND)
        list(APPEND UCC_CONFIG_ARGS "--with-cuda=${CUDAToolkit_LIBRARY_ROOT}")
      endif()
      set(ucc_ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc-install")
      set(ucc_VERSION 1.7.0)
      # cmake-format: off
      ExternalProject_Add(
        ucc_external
        URL "https://github.com/openucx/ucc/archive/refs/tags/v1.7.0.tar.gz"
        URL_HASH SHA256=b40df0db75b8505844547574a3a7dad16c9033d7e1ca099ea8508bc57a62b454
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc"
        INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc-install"
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} ./autogen.sh
        COMMAND ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} ./configure "--prefix=<INSTALL_DIR>" ${UCC_CONFIG_ARGS}
        BUILD_IN_SOURCE TRUE
        BUILD_JOB_SERVER_AWARE TRUE
        BUILD_BYPRODUCTS <INSTALL_DIR>/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ucc${CMAKE_SHARED_LIBRARY_SUFFIX}
      )
      # cmake-format: on
      add_dependencies(ucc_external ucx::ucp)
      file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc-install/include)
      add_library(ucc SHARED IMPORTED)
      set_target_properties(
        ucc
        PROPERTIES
          IMPORTED_LOCATION
          "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc-install/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ucc${CMAKE_SHARED_LIBRARY_SUFFIX}"
          INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc-install/include"
      )
      add_dependencies(ucc ucc_external)
      add_library(ucc::ucc ALIAS ucc)
      set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND ucc)
      get_property(_NOT_FOUND GLOBAL PROPERTY PACKAGES_NOT_FOUND)
      list(REMOVE_ITEM _NOT_FOUND ucc)
      set_property(GLOBAL PROPERTY PACKAGES_NOT_FOUND "${_NOT_FOUND}")
    endif()
    list(APPEND REALM_STATIC_DEPENDS ucc)
    list(APPEND REALM_LIBRARIES ucc::ucc)
  endif()

  # TODO(cperry): when we support more bootstraps, we should be able to select and pick one for testing
  if(REALM_BUILD_TESTS OR REALM_BUILD_BENCHMARKS)
    set(UCX_BOOTSTRAP_ENABLE_MPI ON)
  endif()
  if(REALM_INSTALL_UCX_BOOTSTRAPS)
    add_subdirectory(src/realm/ucx/bootstrap)
  else()
    add_subdirectory(src/realm/ucx/bootstrap EXCLUDE_FROM_ALL)
  endif()
endif()

if(REALM_ENABLE_GASNETEX)
  set(GASNET_CONDUIT
      ""
      CACHE STRING "GASNet required conduit"
  )
  option(GASNET_USE_MULTITHREADED "Use multithreaded gasnet" ON)
  option(GASNET_USE_SYNC "Use multithreaded sync gasnet" OFF)
  # Force cmake to always refresh its package database
  set(GASNet_FOUND
      ""
      CACHE INTERNAL "" FORCE
  )
  if(NOT GASNet_CONFIG_ARGS_INTERNAL)
    find_package(GASNet QUIET)
  endif()

  if(NOT GASNet_FOUND)
    # Unfortunately, GASNet needs to be built at configure time in order to retrieve all the
    # necessacary build flags via pkg-config. One could create imported library targets for each
    # conduit, including its external dependencies in order to enable these builds at build-time,
    # but this is not maintainable long term.
    set(GASNet_ROOT ${CMAKE_CURRENT_BINARY_DIR}/_deps/gasnet-install)
    set(GASNet_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/gasnet)

    if(NOT GASNet_CONFIG_ARGS)
      if(CUDAToolkit_FOUND)
        get_target_property(_cuda_driver_path CUDA::cuda_driver LOCATION)
        cmake_path(REMOVE_FILENAME _cuda_driver_path)
        list(APPEND GASNet_CONFIG_ARGS "--with-cuda-home=${CUDAToolkit_LIBRARY_ROOT}"
             "--with-cuda-ldflags=-L${_cuda_driver_path}" --enable-kind-cuda-uva
        )
      endif()
      if(hip_FOUND)
        list(APPEND GASNet_CONFIG_ARGS "--with-hip-home=${HIP_ROOT_DIR}" --enable-kind-hip)
      endif()
      if(MPI_FOUND)
        list(APPEND GASNet_CONFIG_ARGS --enable-mpi --with-mpi-cc=${MPI_C_COMPILER}
             --with-mpi-cflags=${MPI_C_COMPILER_FLAGS} --with-mpi-libs=${MPI_C_LIBRARIES}
        )
      endif()
      if(HWLOC_FOUND)
        list(APPEND GASNet_CONFIG_ARGS --enable-hwloc --with-hwloc-home=${HWLOC_PREFIX})
      endif()
      list(APPEND GASNet_CONFIG_ARGS --disable-backtrace)
    endif()

    if(GASNET_CONDUIT)
      list(APPEND GASNet_CONFIG_ARGS --enable-${GASNET_CONDUIT} --disable-auto-conduit-detect)
    endif()

    if(BUILD_SHARED_LIBS AND UNIX)
      set(_GASNet_FPIC -fPIC)
    endif()

    list(
      APPEND
      GASNet_CONFIG_ARGS
      "--prefix=${GASNet_ROOT}"
      "--with-cflags=${CMAKE_C_FLAGS} ${_GASNet_FPIC}"
      "--with-cxxflags=${CMAKE_CXX_FLAGS} ${_GASNet_FPIC}"
      "--with-mpi-cflags=${CMAKE_C_FLAGS} ${_GASNet_FPIC}"
      "--with-ldflags=${CMAKE_SHARED_LINKER_FLAGS} ${_GASNet_FPIC}"
      "--enable-pthreads"
    )

    FetchContent_Declare(
      GASNet
      URL https://gasnet.lbl.gov/EX/GASNet-2025.8.0.tar.gz
      URL_HASH SHA256=bd5919099477d1d2f59c247d006e9d1ac017c9190c974f5e069667418e5bf48d
      SOURCE_DIR "${GASNet_SOURCE_DIR}" INSTALL_DIR "${GASNet_ROOT}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(GASNet)

    if(NOT GASNet_CONFIG_ARGS STREQUAL GASNet_CONFIG_ARGS_INTERNAL)
      file(REMOVE ${GASNet_SOURCE_DIR}/config.cache)
      execute_process(
        COMMAND
          "${CMAKE_COMMAND}" -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
          "${GASNet_SOURCE_DIR}/configure" ${GASNet_CONFIG_ARGS} COMMAND_ECHO STDOUT
          COMMAND_ERROR_IS_FATAL ANY
        WORKING_DIRECTORY ${GASNet_SOURCE_DIR}
      )
      execute_process(
        COMMAND "${MAKE_EXECUTABLE}" -j install COMMAND_ECHO STDOUT COMMAND_ERROR_IS_FATAL ANY
        WORKING_DIRECTORY ${GASNet_SOURCE_DIR}
      )
      set(GASNet_CONFIG_ARGS_INTERNAL
          ${GASNet_CONFIG_ARGS}
          CACHE INTERNAL "for internal use only; do not modify" FORCE
      )
    endif()

    find_package(GASNet)
  endif()
  if(NOT GASNet_FOUND)
    message(FATAL_ERROR "Failed to find GASNet")
  else()
    set(GEX_WRAPPER_BUILD_SHARED ${REALM_ENABLE_GASNETEX_WRAPPER})
    if (REALM_INSTALL_GASNETEX_WRAPPER)
      set(GEX_NO_INSTALL FALSE)
      add_subdirectory(
        ${REALM_SOURCE_DIR}/gasnetex/gasnetex_wrapper ${CMAKE_CURRENT_BINARY_DIR}/gasnetex_wrapper
      )
    else()
      set(GEX_NO_INSTALL TRUE)
      add_subdirectory(
        ${REALM_SOURCE_DIR}/gasnetex/gasnetex_wrapper ${CMAKE_CURRENT_BINARY_DIR}/gasnetex_wrapper
        EXCLUDE_FROM_ALL
      )
    endif()
    if(NOT REALM_ENABLE_GASNETEX_WRAPPER)
      list(APPEND REALM_LIBRARIES GASNet::GASNet)
      list(APPEND REALM_STATIC_DEPENDS GASNet)
    endif()
  endif()
endif()

if(REALM_SANITIZER AND NOT REALM_SANITIZER STREQUAL "NONE")
  if(REALM_SANITIZER STREQUAL "ASAN")
    set(SANITIZER "address")
  elseif(REALM_SANITIZER STREQUAL "UBSAN")
    set(SANITIZER "undefined")
  else(REALM_SANITIZER STREQUAL "TSAN")
    set(SANITIZER "thread")
  endif()
  add_compile_options($<$<CONFIG:Debug>:-fsanitize=${SANITIZER}>)
  add_link_options($<$<CONFIG:Debug>:-fsanitize=${SANITIZER}>)
  add_compile_definitions(${REALM_SANITIZER}_ENABLED)
endif()

#endregion

#region Feature Defines
if(REALM_ENABLE_CUDA)
  set(REALM_USE_CUDA TRUE)
endif()

add_feature_info(nvtx REALM_USE_NVTX "Enables NVTX annotations for profiling visibility")

if(REALM_ENABLE_HIP)
  set(REALM_USE_HIP TRUE)
endif()
if(REALM_ENABLE_LLVM)
  set(REALM_USE_LLVM TRUE)
endif()
if(REALM_ENABLE_HDF5)
  set(REALM_USE_HDF5 TRUE)
endif()
if(REALM_ENABLE_PYTHON)
  set(REALM_USE_PYTHON TRUE)
endif()
if(REALM_ENABLE_MPI)
  set(REALM_USE_MPI TRUE)
endif()
if(REALM_ENABLE_OPENMP)
  set(REALM_USE_OPENMP TRUE)
endif()
add_feature_info(openmp REALM_USE_OPENMP "Enables OpenMP support")

if(REALM_ENABLE_PREALM)
  set(REALM_USE_PREALM TRUE)
endif()
add_feature_info(prealm REALM_USE_PREALM "Enables Realm Profiling support")

if(REALM_ENABLE_KOKKOS)
  set(REALM_USE_KOKKOS TRUE)
endif()
if(REALM_ENABLE_GASNETEX)
  set(REALM_USE_GASNETEX TRUE)
  set(REALM_USE_GASNETEX_WRAPPER ${REALM_ENABLE_GASNETEX_WRAPPER})
endif()
if(REALM_ENABLE_UCX)
  set(REALM_USE_UCX TRUE)
endif()
if(REALM_ENABLE_HWLOC)
  set(REALM_USE_HWLOC TRUE)
endif()
if(REALM_ENABLE_PAPI)
  set(REALM_USE_PAPI TRUE)
endif()
if(REALM_ENABLE_CPPTRACE)
  set(REALM_USE_CPPTRACE TRUE)
endif()

set(REALM_USE_LIBDL ${REALM_ENABLE_LIBDL})
add_feature_info(libdl REALM_USE_LIBDL "Enables use of dlopen")
add_feature_info(aio REALM_USE_LIBAIO "Enables use of AIO for dma transfers")
add_feature_info(shm REALM_USE_SHM "Enables use of shared memory for CPU IPC")
set(REALM_TIMERS_USE_RDTSC ${REALM_ENABLE_RDTSC})
add_feature_info(rdtsc REALM_TIMERS_USE_RDTSC "Enables use of rdtsc for Realm::Clock")
add_feature_info(
  mpi_comm_split REALM_MPI_HAS_COMM_SPLIT_TYPE
  "Enables use of MPI_Comm_split_type in realm to identify shared peers"
)
#endregion

#region Source lists
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

if(REALM_USE_CUDA)
  add_library(realm_cuda_fatbin OBJECT ${REALM_CUDA_SOURCES})
  target_compile_options(
    realm_cuda_fatbin
    PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
            -Xcudafe=--diag_suppress=boolean_controlling_expr_is_constant --fatbin>
  )
  target_compile_definitions(realm_cuda_fatbin PRIVATE "CUDA_FATBIN_COMPILATION")

  target_include_directories(
    realm_cuda_fatbin PRIVATE "${REALM_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}/include"
  )

  set_target_properties(realm_cuda_fatbin PROPERTIES CUDA_STANDARD ${REALM_CXX_STANDARD})

  set(realm_fatbin_cc "${CMAKE_CURRENT_BINARY_DIR}/realm_fatbin.cc")
  add_custom_command(
    OUTPUT "${realm_fatbin_cc}"
    COMMAND
      ${CMAKE_COMMAND} "-DVAR_NAME=realm_fatbin" "-DDEFINES_HEADER=realm/realm_config.h"
      "-DIN_FILE=$<TARGET_OBJECTS:realm_cuda_fatbin>" "-DOUT_FILE=${realm_fatbin_cc}" -P
      ${PROJECT_SOURCE_DIR}/cmake/bin2c.cmake
    VERBATIM
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS realm_cuda_fatbin
    COMMENT "Embedding binary objects realm_cuda_fatbin -> ${realm_fatbin_cc}"
  )
  add_custom_target(realm_fatbin_tgt DEPENDS "${realm_fatbin_cc}")
  list(APPEND REALM_SOURCES "${realm_fatbin_cc}")
endif()
#endregion

#region Library compilation

# Specifically for adding extra object libraries' target objects to prevent
# needing to use target_link_libraries, which would add the object libraries
# to the interface link libraries of the target, which is not what we want.
set(REALM_EXTRA_OBJS)

# Generate compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(TARGET realm_gex_wrapper_objs AND NOT REALM_ENABLE_GASNETEX_WRAPPER)
  # side rant(cperry) : it would be nice if object libraries were a little more
  # seamless and weren't added part of the interface link libraries for
  # concrete targets like static and shared libraries.
  list(APPEND REALM_EXTRA_OBJS $<TARGET_OBJECTS:realm_gex_wrapper_objs>)
endif()

# Remove NDEBUG from definitions for cxx, c, and cuda flags for those that might have it
# TODO(cperry): remove this once we clean up all the asserts that have side effects
foreach(_build RELEASE RELWITHDEBINFO MINSIZEREL)
  foreach(_lang C CXX CUDA)
    string(REPLACE "/DNDEBUG" "" CMAKE_${_lang}_FLAGS_${_build} "${CMAKE_${_lang}_FLAGS_${_build}}")
    string(REPLACE "-DNDEBUG" "" CMAKE_${_lang}_FLAGS_${_build} "${CMAKE_${_lang}_FLAGS_${_build}}")
  endforeach()
endforeach()

add_library(realm_obj OBJECT ${REALM_SOURCES})
set_target_properties(
  realm_obj
  PROPERTIES C_VISIBILITY_PRESET hidden
             CXX_VISIBILITY_PRESET hidden
             CUDA_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON
)

# Tell threads.cc to emit the anchor reference only when gdb_script.cc is
# actually compiled (Debug / RelWithDebInfo on Linux/FreeBSD).  Otherwise
# the anchor is undefined and static-archive links would fail.
if(REALM_EMBED_GDB_SCRIPT)
  target_compile_definitions(realm_obj PRIVATE REALM_EMBED_GDB_SCRIPT=1)
endif()
add_library(Realm ${REALM_RC_SOURCES} $<TARGET_OBJECTS:realm_obj> ${REALM_EXTRA_OBJS})
add_library(Realm::Realm ALIAS Realm)
set_target_properties(
  Realm
  PROPERTIES VERSION ${Realm_VERSION}
             SOVERSION ${Realm_VERSION_MAJOR}
             INTERPROCEDURAL_OPTIMIZATION ${REALM_ENABLE_LTO}
             C_VISIBILITY_PRESET hidden
             CXX_VISIBILITY_PRESET hidden
             CUDA_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "realm"
)

generate_export_header(
  Realm EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/realm/realm_exports.h"
)

if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(realm_obj PUBLIC REALM_STATIC_DEFINE=1)
  target_compile_definitions(Realm PUBLIC REALM_STATIC_DEFINE=1)
endif()

target_link_libraries(realm_obj PRIVATE ${REALM_LIBRARIES})
target_link_libraries(Realm PRIVATE ${REALM_LIBRARIES})
target_link_libraries(realm_obj PUBLIC ${REALM_PUBLIC_LIBRARIES})
target_link_libraries(Realm PUBLIC ${REALM_PUBLIC_LIBRARIES})

target_include_directories(
  realm_obj
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> $<BUILD_INTERFACE:${REALM_SOURCE_DIR}/..>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  PRIVATE $<$<TARGET_EXISTS:Python3::Python>:${Python_INCLUDE_DIRS}>
          $<$<TARGET_EXISTS:hip::host>:${HIP_INCLUDE_DIR}>
)
target_include_directories(
  Realm
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> $<BUILD_INTERFACE:${REALM_SOURCE_DIR}/..>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  PRIVATE $<$<TARGET_EXISTS:Python3::Python>:${Python_INCLUDE_DIRS}>
          $<$<TARGET_EXISTS:hip::host>:${HIP_INCLUDE_DIR}>
)
if(REALM_USE_CUDA)
  target_include_directories(
    realm_obj PRIVATE ${CUDAToolkit_CUPTI_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIR}
  )
  target_include_directories(
    Realm PRIVATE ${CUDAToolkit_CUPTI_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIR}
  )
endif()
if(REALM_USE_GASNETEX)
  # Force the creation of the export header for the wrapper we need, but nothing else
  add_dependencies(Realm realm_gex_wrapper_objs)
  add_dependencies(realm_obj realm_gex_wrapper_objs)
  target_include_directories(
    realm_obj PRIVATE $<TARGET_PROPERTY:realm_gex_wrapper_objs,INTERFACE_INCLUDE_DIRECTORIES>
  )
  target_include_directories(
    Realm PRIVATE $<TARGET_PROPERTY:realm_gex_wrapper_objs,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif()

if(TARGET realm_fatbin_tgt)
  add_dependencies(Realm realm_fatbin_tgt)
  add_dependencies(realm_obj realm_fatbin_tgt)
endif()

target_compile_features(Realm PUBLIC cxx_std_${REALM_CXX_STANDARD})
target_compile_features(realm_obj PUBLIC cxx_std_${REALM_CXX_STANDARD})
target_compile_options(
  Realm
  PRIVATE $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
          -Wall
          -Wextra
          -Wno-unused-parameter>
          $<$<CXX_COMPILER_ID:MSVC>:
          /W4
          /Zc:alignedNew
          /utf-8
          /Zi
          /permissive-
          /Zc:tlsGuards
          /wd4100>
  PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus>
)
target_compile_options(
  realm_obj
  PRIVATE $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
          -Wall
          -Wextra
          -Wno-unused-parameter>
          $<$<CXX_COMPILER_ID:MSVC>:
          /W4
          /Zc:alignedNew
          /utf-8
          /Zi
          /permissive-
          /Zc:tlsGuards
          /wd4100>
  PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus>
)
if(REALM_ENABLE_COVERAGE AND NOT OPENCPPCOV_PATH)
  append_coverage_compiler_flags_to_target(realm_obj)
  append_coverage_compiler_flags_to_target(Realm)
endif()
#endregion

#region Testing
if(REALM_BUILD_TESTS OR REALM_BUILD_BENCHMARKS)
  if(OPENCPPCOV_PATH)
    list(APPEND CMAKE_TEST_LAUNCHER ${OPENCPPCOV} --sources ${REALM_SOURCE_DIR} --)
  endif()
  if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
    enable_testing()
  endif()
endif()

if(REALM_BUILD_TESTS)
  add_subdirectory(tests)
endif()
add_feature_info(tests REALM_BUILD_TESTS "Build unit and integration tests")

if(REALM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
add_feature_info(examples REALM_BUILD_EXAMPLES "Build examples")

if(REALM_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
add_feature_info(benchmarks REALM_BUILD_BENCHMARKS "Build benchmarks")

if(REALM_BUILD_TUTORIALS)
  add_subdirectory(tutorials)
endif()
add_feature_info(tutorials REALM_BUILD_TUTORIALS "Build tutorials")

#endregion

#region Documentation
# TODO(cperry): fix me
if(REALM_BUILD_DOCS)
  set(DOXYGEN_PROJECT_LOGO ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/logo.png)
  set(DOXYGEN_PROJECT_NUMBER ${REALM_VERSION})
  set(DOXYGEN_GENERATE_HTML ${REALM_GENERATE_HTML})
  set(DOXYGEN_GENERATE_MAN ${REALM_GENERATE_MAN})
  set(DOXYGEN_GENERATE_TREEVIEW YES)
  set(DOXYGEN_DISABLE_INDEX NO)
  set(DOXYGEN_FULL_SIDEBAR NO)
  set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs")
  set(DOXYGEN_EXTRACT_ALL YES)
  set(DOXYGEN_FULL_PATH_NAMES YES)
  set(DOXYGEN_HTML_DYNAMIC_MENUS NO)
  set(DOXYGEN_HTML_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/header.html)
  set(DOXYGEN_HTML_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/footer.html)
  set(DOXYGEN_HTML_EXTRA_STYLESHEET
      ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/customdoxygen.css
      ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/doxygen-awesome.css
      ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/doxygen-awesome-sidebar-only.css
  )
  set(DOXYGEN_HTML_COLORSTYLE LIGHT)
  set(DOXYGEN_HTML_COLORSTYLE_HUE 209)
  set(DOXYGEN_HTML_COLORSTYLE_SAT 255)
  set(DOXYGEN_HTML_COLORSTYLE_GAMMA 113)
  set(DOXYGEN_DOT_IMAGE_FORMAT svg)
  set(DOXYGEN_DOT_TRANSPARENT YES)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
  if(NOT REALM_BUILD_INTERNAL_DOCS)
    set(DOXYGEN_STRIP_FROM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include
                                ${CMAKE_CURRENT_BINARY_DIR}/include
    )
  endif()
  # This should include all public header files and nothing else.
  file(GLOB_RECURSE REALM_DOC_FILES "src/*.h")
  list(APPEND REALM_DOC_FILES ${CMAKE_CURRENT_BINARY_DIR}/include
                              ${CMAKE_CURRENT_SOURCE_DIR}/README.md
  )
  doxygen_add_docs(docs ${REALM_DOC_FILES} COMMENT "Generate docs" ALL)
endif()
#endregion

#region Packaging
if(REALM_ENABLE_INSTALL)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RealmPackaging.cmake)
endif()
#endregion

if(PROJECT_IS_TOP_LEVEL)
  feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES)
endif()
