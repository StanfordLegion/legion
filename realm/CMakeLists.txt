# Copyright 2025 Stanford University, NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

#region Versioning
find_package(Git)
if(Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" describe --tags
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
  string(REGEX REPLACE "legion-([0-9\.]+)-.*" "\\1" Realm_SHORT_VERSION "${REALM_GIT_VERSION}")
  set(Realm_VERSION_META "+${REALM_VERSION_SHA}")
  if(REALM_DIRTY)
    string(APPEND Realm_VERSION_META "-dirty")
  endif()
endif()

if(Realm_SHORT_VERSION STREQUAL "" AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION _REALM_FULL_VERSION)
  string(REGEX REPLACE "[\\-\\+].*$" "" Realm_SHORT_VERSION "${_REALM_FULL_VERSION}")
  string(REPLACE "${Realm_SHORT_VERSION}" "" Realm_VERSION_META "${_REALM_FULL_VERSION}")
else()
  if(Realm_SHORT_VERSION STREQUAL "")
    set(Realm_SHORT_VERSION 0.0.1)
  endif()
endif()
#endregion

project(
  Realm
  VERSION ${Realm_SHORT_VERSION}
  LANGUAGES CXX C
  DESCRIPTION "A distributed, event-based tasking library"
  HOMEPAGE_URL https://gitlab.com/StanfordLegion/legion/
)
set(REALM_VERSION "${Realm_SHORT_VERSION}${Realm_VERSION_META}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#region External built-ins
include(GenerateExportHeader)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckIPOSupported)
include(CheckFunctionExists)
include(CheckCXXCompilerFlag)
include(CMakePushCheckState)
include(CMakeDependentOption)
include(ExternalProject)
include(FetchContent)
include(FeatureSummary)

set(_CPM_REQUESTED_VERSION 0.40.2)
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM-v${CPM_REQUESTED_VERSION}.cmake)
  file(
    DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v${_CPM_REQUESTED_VERSION}/CPM.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM-v${_CPM_REQUESTED_VERSION}.cmake"
    EXPECTED_HASH SHA256=c8cdc32c03816538ce22781ed72964dc864b2a34a310d3b7104812a5ca2d835d
  )
  include("${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM-v${_CPM_REQUESTED_VERSION}.cmake")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

check_ipo_supported(RESULT REALM_SUPPORTS_LTO LANGUAGES CXX C)
#endregion

#region Build options These should all be prefixed with 'REALM_' to prevent subproject name clashes
set(REALM_MAX_DIM
  3
  CACHE STRING "Maximum dimension compiled for"
)
set(REALM_LIMIT_SYMBOL_VISIBILITY 1)
set(REALM_CXX_STANDARD
  17
  CACHE STRING "C++ standard used"
)
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(DEBUG_REALM 1)
  set(COMPILE_TIME_MIN_LEVEL LEVEL_DEBUG)
else()
  set(COMPILE_TIME_MIN_LEVEL LEVEL_WARNING)
endif()

option(REALM_ENABLE_INSTALL "Enable installation support" ${PROJECT_IS_TOP_LEVEL})
option(REALM_BUILD_TESTS "Build tests" ${PROJECT_IS_TOPLEVEL})
option(REALM_BUILD_EXAMPLES "Build examples" ${PROJECT_IS_TOPLEVEL})
option(REALM_BUILD_BENCHMARKS "Build benchmarks" ${PROJECT_IS_TOPLEVEL})
option(REALM_BUILD_TUTORIALS "Build tutorials" ${PROJECT_IS_TOPLEVEL})

option(REALM_ENABLE_CUDA "Compile with CUDA support" ON)
option(REALM_ENABLE_HIP "Compile with HIP support" ON)
option(REALM_ENABLE_LLVM "Compile with LLVMJIT support" OFF)
option(REALM_ENABLE_HDF5 "Compile with HDF5 support" ON)
option(REALM_ENABLE_MPI "Compile with MPI support" OFF)
option(REALM_ENABLE_OPENMP "Compile with OpenMP support" ON)
option(REALM_ENABLE_KOKKOS "Compile with kokkos support" OFF)
option(REALM_ENABLE_UCX "Compile with UCX support" ON)
option(REALM_ENABLE_GASNETEX "Compile with GASNetEx support" ON)
option(REALM_ENABLE_GASNETEX_WRAPPER "Compile with dynamic GASNetEx ABI stability wrapper" OFF)
option(REALM_ENABLE_HWLOC "Compile with hwloc hardware discovery support" OFF)
option(REALM_ENABLE_PAPI "Compile with PAPI thread profiling support" ON)
option(REALM_ENABLE_CPPTRACE "Compile with cpptrace backtrace support" ON)
option(REALM_ENABLE_LIBDL "Compile with dynamic runtime linking support" ON)
cmake_dependent_option(
  REALM_ENABLE_PYTHON "Compile with Python support" ON "REALM_ENABLE_LIBDL" OFF
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
  REALM_OPENMP_GOMP_SUPPORT "Enable GOMP support for Realm's wrapper OpenMP" OFF
  "REALM_ENABLE_OPENMP; NOT REALM_OPENMP_SYSTEM_RUNTIME" OFF
)
cmake_dependent_option(
  REALM_OPENMP_KMP_SUPPORT "Enable KMP support for Realm's wrapper OpenMP" OFF
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
  REALM_ENABLE_UCC "Enable UCC for UCX networking module collectives" ON
  "REALM_ENABLE_UCX" OFF
)
#endregion

# TODO(cperry): move these under the current directory
set(REALM_SOURCE_DIR "../runtime/realm")
cmake_path(ABSOLUTE_PATH REALM_SOURCE_DIR NORMALIZE)

#region Dependencies
set(REALM_STATIC_DEPENDS)
set_package_properties(CUDAToolkit PROPERTIES TYPE RECOMMENDED)
set_package_properties(HIP PROPERTIES TYPE OPTIONAL)
set_package_properties(LLVM PROPERTIES TYPE OPTIONAL)
set_package_properties(hdf5 PROPERTIES TYPE OPTIONAL)
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

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

if(REALM_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
endif()

if(UNIX)
  check_function_exists(posix_fallocate64 REALM_HAS_POSIX_FALLOCATE64)
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
    "CMAKE_SKIP_INSTALL_RULES ON"
    "BUILD_SHARED_LIBS OFF"
  )
  set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND cpptrace)
  get_property(_NOT_FOUND GLOBAL PROPERTY PACKAGES_NOT_FOUND)
  list(REMOVE_ITEM _NOT_FOUND cpptrace)
  set_property(GLOBAL PROPERTY PACKAGES_NOT_FOUND "${_NOT_FOUND}")
  list(APPEND REALM_STATIC_DEPENDS cpptrace)
endif()

if(REALM_ENABLE_HWLOC)
  find_package(HWLOC)
  list(APPEND REALM_STATIC_DEPENDS HWLOC)
endif()

if(REALM_ENABLE_PAPI)
  find_package(Papi)
  list(APPEND REALM_STATIC_DEPENDS Papi)
endif()

if(REALM_ENABLE_COVERAGE)
  if(MSVC)
    find_program(
      OPENCPPCOV OpenCppCoverage.exe REQUIRED HINTS $ENV{ProgramFiles\(x86\)}/OpenCppCoverage
      $ENV{ProgramFiles}/OpenCppCoverage
    )
  else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CodeCoverage.cmake)
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
  find_package(hip CONFIG)
  if (NOT HIP_ROOT_DIR)
    set(HIP_ROOT_DIR "${ROCM_PATH}")
  endif()
  list(APPEND REALM_STATIC_DEPENDS hip)
endif()

if(REALM_ENABLE_CUDA)
  find_package(CUDAToolkit 11.7)
  if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      execute_process(
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} --list-gpu-arch OUTPUT_VARIABLE compute_archs
      )
      string(REPLACE "compute_" "" compute_archs "${compute_archs}")
      string(REPLACE "\n" ";" compute_archs "${compute_archs}")
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

    enable_language(CUDA)
    get_filename_component(CUDAToolkit_LIBRARY_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY CACHE)
    # Unfortunately, FindCUDAToolkit doesn't always find CUPTI due to directory changes in the
    # toolkit, so correct it here
    find_path(
      CUDAToolkit_CUPTI_INCLUDE_DIR cupti.h
      PATHS ${CUDAToolkit_LIBRARY_ROOT}
      PATH_SUFFIXES include ../include extras/CUPTI/include ../extras/CUPTI/include
    )
    # Deal with the library name change in older versions of cmake
    if(NOT TARGET CUDA::nvToolsExt)
      add_library(CUDA::nvToolsExt ALIAS CUDA::nvtx3)
    endif()
    if(NOT REALM_CUDA_DYNAMIC_LOAD)
      list(APPEND REALM_STATIC_DEPENDS CUDAToolkit)
    endif()
  endif()
endif()

if(REALM_ENABLE_MPI)
  find_package(MPI)
  if(MPI_FOUND)
    cmake_push_check_state()
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${MPI_mpi_LIBRARY})
    check_function_exists(MPI_Comm_split_type REALM_MPI_HAS_COMM_SPLIT_TYPE)
    cmake_pop_check_state()
  endif()
  list(APPEND REALM_STATIC_DEPENDS MPI)
endif()

if(REALM_ENABLE_OPENMP)
  if(REALM_OPENMP_SYSTEM_RUNTIME)
    find_package(OpenMP COMPONENTS C CXX)
  endif()
  list(APPEND REALM_STATIC_DEPENDS OpenMP)
endif()

if(REALM_ENABLE_PYTHON)
  find_package(Python3 COMPONENTS Development)
  list(APPEND REALM_STATIC_DEPENDS Python3)
endif()

if(REALM_ENABLE_KOKKOS)
  if(TARGET CUDA::cuda_driver)
    list(APPEND KOKKOS_CONFIG_ARGS "KOKKOS_USE_CUDA ON")
  endif()
  if(TARGET hip::hip_host)
    list(APPEND KOKKOS_CONFIG_ARGS "KOKKOS_USE_HIP ON")
  endif()
  if(TARGET OpenMP::OpenMP_CXX)
    list(APPEND KOKKOS_CONFIG_ARGS "KOKKOS_USE_OPENMP ON")
  endif()
  list(APPEND KOKKOS_CONFIG_ARGS "Kokkos_INSTALL_TESTING OFF" "CMAKE_SKIP_INSTALL_RULES ON"
    "BUILD_SHARED_LIBS OFF" "CMAKE_POSITION_INDEPENDENT_CODE ON"
  )
  cpmfindpackage(
    NAME
    Kokkos
    GIT_REPOSITORY
    https://github.com/kokkos/kokkos.git
    GIT_TAG
    4.5.00
    EXCLUDE_FROM_ALL
    YES
    SYSTEM
    YES
    OPTIONS
    ${KOKKOS_CONFIG_ARGS}
  )
  if(Kokkos_ADDED)
    set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND Kokkos)
    get_property(_NOT_FOUND GLOBAL PROPERTY PACKAGES_NOT_FOUND)
    list(REMOVE_ITEM _NOT_FOUND Kokkos)
    set_property(GLOBAL PROPERTY PACKAGES_NOT_FOUND "${_NOT_FOUND}")
  else()
    list(APPEND REALM_STATIC_DEPENDS Kokkos)
  endif()
endif()

if(REALM_ENABLE_LLVM)
  find_package(LLVM)
  list(APPEND REALM_STATIC_DEPENDS LLVM)
endif()

if(REALM_ENABLE_HDF5)
  find_package(HDF5)
  list(APPEND REALM_STATIC_DEPENDS HDF5)
endif()

if(REALM_ENABLE_UCX)
  find_package(ucx QUIET CONFIG)
  if(NOT ucx_FOUND)
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
    set(ucx_VERSION 1.17.0)
    # cmake-format: off
    ExternalProject_Add(
      ucx_external
      URL https://github.com/openucx/ucx/releases/download/v1.17.0/ucx-1.17.0.tar.gz
      URL_HASH SHA256=34658e282f99f89ce7a991c542e9727552734ac6ad408c52f22b4c2653b04276
      SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucx"
      INSTALL_DIR "${ucx_ROOT_DIR}"
      CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
      contrib/configure-release-mt "--prefix=<INSTALL_DIR>" ${UCX_CONFIG_ARGS}
      BUILD_COMMAND $(MAKE)
      INSTALL_COMMAND $(MAKE) install
      BUILD_IN_SOURCE TRUE
      BUILD_JOB_SERVER_AWARE TRUE
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
    list(APPEND REALM_STATIC_DEPENDS ucp)
  endif()

  if(REALM_ENABLE_UCC)
    find_package(ucc QUIET CONFIG)
    if(NOT ucc_FOUND)
      list(
        APPEND
        UCC_CONFIG_ARGS
        "--with-ucx=${ucx_ROOT_DIR}"
        "--disable-doxygen-doc"
        "--with-pic"
        "--enable-static"
      )
      if(hip_FOUND)
        list(APPEND UCC_CONFIG_ARGS "--with-rocm=${HIP_ROOT_DIR}")
      endif()
      if(CUDAToolkit_FOUND)
        list(APPEND UCC_CONFIG_ARGS "--with-cuda=${CUDAToolkit_LIBRARY_ROOT}")
      endif()
      set(ucc_VERSION 1.3.0)
      # cmake-format: off
      ExternalProject_Add(
        ucc_external
        URL "https://github.com/openucx/ucc/archive/refs/tags/v1.3.0.tar.gz"
        URL_HASH SHA256=b56379abe5f1c125bfa83be305d78d81a64aa271b7b5fff0ac17b86725ff3acf
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc"
        INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc-install"
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} ./autogen.sh
        COMMAND ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} ./configure "--prefix=<INSTALL_DIR>" ${UCC_CONFIG_ARGS}
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND $(MAKE) install
        BUILD_IN_SOURCE TRUE
        BUILD_JOB_SERVER_AWARE TRUE
      )
      # cmake-format: on
      add_dependencies(ucc_external ucx_external)
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
      list(APPEND REALM_STATIC_DEPENDS ucc)
    endif()
  endif()

  # TODO(cperry): when we support more bootstraps, we should be able to select and pick one for testing
  if(REALM_BUILD_TESTS OR REALM_BUILD_BENCHMARKS)
    set(BOOTSTRAP_ENABLE_MPI ON)
  endif()
  add_subdirectory(src/ucx/bootstrap EXCLUDE_FROM_ALL)
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
    set(GASNet_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/gasnet-install)
    set(GASNet_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/gasnet)

    if(NOT GASNet_CONFIG_ARGS)
      if(CUDAToolkit_FOUND)
        get_target_property(_cuda_driver_path CUDA::cuda_driver LOCATION)
        cmake_path(REMOVE_FILENAME _cuda_driver_path)
        list(APPEND GASNet_CONFIG_ARGS "--with-cuda-home=${CUDAToolkit_LIBRARY_ROOT}"
          "--with-cuda-ldflags=-L${_cuda_driver_path}"
          --enable-kind-cuda-uva
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
      "--prefix=${GASNet_ROOT_DIR}"
      "--with-cflags=${CMAKE_C_FLAGS} ${_GASNet_FPIC}"
      "--with-cxxflags=${CMAKE_CXX_FLAGS} ${_GASNet_FPIC}"
      "--with-mpi-cflags=${CMAKE_C_FLAGS} ${_GASNet_FPIC}"
      "--with-ldflags=${CMAKE_SHARED_LINKER_FLAGS} ${_GASNet_FPIC}"
      "--enable-pthreads"
    )

    FetchContent_Declare(
      GASNet
      URL https://gasnet.lbl.gov/EX/GASNet-2024.5.0.tar.gz
      URL_HASH SHA256=f945e80f71d340664766b66290496d230e021df5e5cd88f404d101258446daa9
      SOURCE_DIR "${GASNet_SOURCE_DIR}" INSTALL_DIR "${GASNet_ROOT_DIR}"
    )
    if(NOT GASNet_populated)
      FetchContent_Populate(GASNet)
    endif()

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
        COMMAND "${CMAKE_MAKE_PROGRAM}" -j install COMMAND_ECHO STDOUT COMMAND_ERROR_IS_FATAL ANY
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
  endif()
  list(APPEND REALM_STATIC_DEPENDS GASNet)
endif()

if(GASNet_FOUND)
  set(GEX_WRAPPER_BUILD_SHARED ${REALM_ENABLE_GASNETEX_WRAPPER})
  set(GEX_NO_INSTALL NOT ${REALM_ENABLE_GASNETEX_WRAPPER})
  add_subdirectory(
    ${REALM_SOURCE_DIR}/gasnetex/gasnetex_wrapper ${CMAKE_CURRENT_BINARY_DIR}/gasnetex_wrapper
    EXCLUDE_FROM_ALL
  )
endif()
#endregion

#region Feature Defines
if(TARGET CUDA::cuda_driver)
  set(REALM_USE_CUDA TRUE)
endif()

if(TARGET CUDA::nvToolsExt)
  set(REALM_USE_NVTX TRUE)
endif()
add_feature_info(nvtx REALM_USE_NVTX "Enables NVTX annotations for profiling visibility")

if(TARGET hip::hip_host)
  set(REALM_USE_HIP TRUE)
endif()
if(TARGET LLVM::LLVM)
  set(REALM_USE_LLVM TRUE)
endif()
if(TARGET hdf5::hdf5)
  set(REALM_USE_HDF5 TRUE)
endif()
if(TARGET MPI::MPI_CXX)
  set(REALM_USE_MPI TRUE)
endif()
if(TARGET OpenMP::OpenMP_CXX)
  set(REALM_USE_OPENMP TRUE)
endif()
if(TARGET Kokkos::kokkoscore)
  set(REALM_USE_KOKKOS TRUE)
endif()
if(TARGET GASNet::GASNet)
  set(REALM_USE_GASNETEX TRUE)
endif()
if(TARGET ucx::ucp)
  set(REALM_USE_UCX TRUE)
endif()
if(TARGET HWLOC::HWLOC)
  set(REALM_USE_HWLOC TRUE)
endif()
if(TARGET Papi::Papi)
  set(REALM_USE_PAPI TRUE)
endif()
if(TARGET cpptrace::cpptrace)
  set(REALM_USE_CPPTRACE cpptrace::cpptace)
endif()

set(REALM_USE_LIBDL ${REALM_ENABLE_LIBDL})
set(REALM_USE_SHM ${REALM_ENABLE_SHM})
add_feature_info(shm REALM_USE_SHM "Enables use of shared memory for CPU IPC")
#endregion

#region Source lists
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

if(TARGET CUDA::cuda_driver)
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
  list(APPEND REALM_SOURCES ${realm_fatbin_cc})
  add_custom_command(
    OUTPUT "${realm_fatbin_cc}"
    COMMAND
    ${CMAKE_COMMAND} "-DVAR_NAME=realm_fatbin" "-DDEFINES_HEADER=realm/realm_config.h"
    "-DIN_FILE=$<TARGET_OBJECTS:realm_cuda_fatbin>" "-DOUT_FILE=${realm_fatbin_cc}" -P
    ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/bin2c.cmake
    VERBATIM
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS realm_cuda_fatbin
    COMMENT "Embedding binary objects realm_cuda_fatbin -> ${realm_fatbin_cc}"
  )
endif()
#endregion

#region Library compilation

add_library(realm_obj OBJECT ${REALM_SOURCES})
set_target_properties(
  realm_obj
  PROPERTIES C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  POSITION_INDEPENDENT_CODE ON
)
add_library(realm ${REALM_RC_SOURCES})
target_link_libraries(realm PUBLIC realm_obj)
set_target_properties(
  realm
  PROPERTIES VERSION ${Realm_VERSION}
  SOVERSION ${Realm_VERSION_MAJOR}
  INTERPROCEDURAL_OPTIMIZATION ${REALM_ENABLE_LTO}
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  POSITION_INDEPENDENT_CODE ON
)
add_library(Realm::Realm ALIAS realm)

generate_export_header(realm EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/realm_exports.h")
if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(realm_obj PUBLIC REALM_STATIC_DEFINE=1)
endif()

target_link_libraries(
  realm_obj
  PRIVATE
  $<TARGET_NAME_IF_EXISTS:hip::hip_host>
  $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
  $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
  $<TARGET_NAME_IF_EXISTS:Kokkos::kokkoscore>
  $<TARGET_NAME_IF_EXISTS:LLVM::LLVM>
  $<TARGET_NAME_IF_EXISTS:hdf5::hdf5>
  $<$<NOT:$<BOOL:${REALM_ENABLE_GASNETEX_WRAPPER}>>:$<TARGET_NAME_IF_EXISTS:GASNet::GASNet>>
  $<TARGET_NAME_IF_EXISTS:ucx::ucp>
  $<TARGET_NAME_IF_EXISTS:ucc::ucc>
  $<TARGET_NAME_IF_EXISTS:HWLOC::HWLOC>
  $<TARGET_NAME_IF_EXISTS:Papi::Papi>
  $<TARGET_NAME_IF_EXISTS:cpptrace::cpptrace>
  $<$<NOT:$<BOOL:${REALM_CUDA_DYNAMIC_LOAD}>>:$<TARGET_NAME_IF_EXISTS:CUDA::cuda_driver>>
  $<$<BOOL:${REALM_ENABLE_NVTX}>:$<TARGET_NAME_IF_EXISTS:CUDA::nvToolsExt>>
  $<$<NOT:$<BOOL:${REALM_ENABLE_GASNETEX_WRAPPER}>>:$<TARGET_NAME_IF_EXISTS:realm_gex_wrapper>>
  $<$<BOOL:${REALM_ENABLE_LIBDL}>:${CMAKE_DL_LIBS}>
)

target_include_directories(
  realm_obj
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> $<BUILD_INTERFACE:${REALM_SOURCE_DIR}/..>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  PRIVATE $<$<TARGET_EXISTS:CUDA::cuda_driver>:${CUDAToolkit_CUPTI_INCLUDE_DIR}>
  $<$<TARGET_EXISTS:CUDA::cuda_driver>:${CUDAToolkit_INCLUDE_DIR}>
  $<$<TARGET_EXISTS:Python3::Python>:${Python_INCLUDE_DIRS}>
)
if(REALM_ENABLE_GASNETEX_WRAPPER)
  # Force the creation of the export header for the wrapper we need, but nothing else
  add_dependencies(realm_obj ${CMAKE_CURRENT_BINARY_DIR}/gasnetex_wrapper/exports/gex_export.h)
endif()

target_compile_features(realm_obj PUBLIC cxx_std_${REALM_CXX_STANDARD})
target_compile_options(
  realm_obj
  PRIVATE $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
  -Wall
  -Wextra
  -Wno-unused-parameter>
  $<$<CXX_COMPILER_ID:MSVC>:
  /W4
  /WX
  /Zc:alignedNew
  /utf-8
  /Zi
  /permissive-
  /Zc:tlsGuards>
)
if(REALM_ENABLE_COVERAGE AND NOT OPENCPPCOV_PATH)
  append_coverage_compiler_flags_to_target(realm_obj)
  append_coverage_compiler_flags_to_target(realm)
endif()
#endregion

#region Testing
if(REALM_BUILD_TESTS OR REALM_BUILD_BENCHMARKS)
  if(OPENCPPCOV_PATH)
    list(APPEND CMAKE_TEST_LAUNCHER ${OPENCPPCOV} --sources ${REALM_SOURCE_DIR} --)
  endif()
  enable_testing()
endif()

if(REALM_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(REALM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(REALM_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(REALM_BUILD_TUTORIALS)
  add_subdirectory(tutorials)
endif()

#endregion

#region Documentation
# TODO(cperry): fix me
if(REALM_BUILD_DOCS AND DOXYGEN_FOUND)
  set(DOXYGEN_GENERATE_HTML ${REALM_GENERATE_HTML})
  set(DOXYGEN_GENERATE_MAN ${REALM_GENERATE_MAN})
  set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs")
  set(DOXYGEN_EXTRACT_ALL YES)
  set(DOXYGEN_FULL_PATH_NAMES YES)
  set(DOXYGEN_HTML_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/header.html)
  set(DOXYGEN_HTML_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/footer.html)
  set(DOXYGEN_HTML_EXTRA_STYLESHEET ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/customdoxygen.css)
  if(NOT REALM_BUILD_INTERNAL_DOCS)
    set(DOXYGEN_STRIP_FROM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_BINARY_DIR}/include
    )
  endif()
  # This should include all public header files and nothing else.
  set(REALM_DOC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)
  if(REALM_BUILD_INTERNAL_DOCS)
    list(APPEND REALM_DOC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src)
  endif()
  doxygen_add_docs(docs ${REALM_DOC_FILES} COMMENT "Generate docs" ALL)
endif()
#endregion

#region Packaging
if(REALM_ENABLE_INSTALL)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RealmPackaging.cmake)
endif()
#endregion

feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES)
