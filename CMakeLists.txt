#------------------------------------------------------------------------------#
# Copyright 2023 Kitware, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

#------------------------------------------------------------------------------#
# Version information
#------------------------------------------------------------------------------#

# the Legion/Realm version string is set by the first of these that works:
# 1) a defined value for the cmake Legion_VERSION variable
# 2) the output of `git describe`, if successful
# 3) the contents of 'VERSION' (at the root of the source tree), if available
# 4) "unknown", if all else fails

set(Legion_VERSION "" CACHE STRING "string to use for Legion/Realm version")
if(Legion_VERSION STREQUAL "")
  find_package(Git)
  if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --dirty --match legion*
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE _git_result
      OUTPUT_VARIABLE Legion_VERSION
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  else()
    set(_git_result -1)
  endif()
  if(_git_result EQUAL 0)
    message(STATUS "Version string from git: ${Legion_VERSION}")
  else()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION)
      file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION _file_contents)
      string(STRIP "${_file_contents}" Legion_VERSION)
      unset(_file_contents)
      message(STATUS "Version string from VERSION file: ${Legion_VERSION}")
    else()
      set(Legion_VERSION "unknown")
      message(WARNING "Could not determine version string - using 'unknown'")
    endif()
  endif()
  unset(_git_result)
endif()
# these are the actual variables that are written into the headers
set(LEGION_VERSION "${Legion_VERSION}")
set(REALM_VERSION "${Legion_VERSION}")

# Parse the major.minor.patch from (potentially non-semver) Legion_VERSION
if(Legion_VERSION MATCHES "([0-9]+)\\.([0-9]+)\\.([0-9]+)") # get the CMake version
  set(Legion_VERSION_MAJOR "${CMAKE_MATCH_1}")
  set(Legion_VERSION_MINOR "${CMAKE_MATCH_2}")
  set(Legion_VERSION_PATCH "${CMAKE_MATCH_3}")
  if(NOT Legion_VERSION_MAJOR VERSION_EQUAL 0)
    string(REGEX REPLACE "^0+" "" Legion_VERSION_MAJOR "${Legion_VERSION_MAJOR}")
  endif()
  if(NOT Legion_VERSION_MINOR VERSION_EQUAL 0)
    string(REGEX REPLACE "^0+" "" Legion_VERSION_MINOR "${Legion_VERSION_MINOR}")
  endif()
  if(NOT Legion_VERSION_PATCH VERSION_EQUAL 0)
    string(REGEX REPLACE "^0+" "" Legion_VERSION_PATCH "${Legion_VERSION_PATCH}")
  endif()
  set(Legion_VERSION "${Legion_VERSION_MAJOR}.${Legion_VERSION_MINOR}.${Legion_VERSION_PATCH}")
else()
  message(WARNING "Could not parse semver-compatible version from Legion version string '${Legion_VERSION}' - setting cmake package version to '0.0.0'")
  set(Legion_VERSION "0.0.0")
endif()

project(Legion VERSION ${Legion_VERSION})

#------------------------------------------------------------------------------#
# Some boilerplate to setup nice output directories
#------------------------------------------------------------------------------#

#for multilib distros
include(GNUInstallDirs)

list(INSERT CMAKE_MODULE_PATH 0 "${Legion_SOURCE_DIR}/cmake")
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${Legion_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${Legion_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Legion_BINARY_DIR}/bin)
endif()

#library api version, bump from time to time
set(SOVERSION 1)

set(INSTALL_SUFFIX "" CACHE STRING "Suffix to add to installed libraries and binaries")

#------------------------------------------------------------------------------#
# Build type and shared vs static library
#------------------------------------------------------------------------------#
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  set(CMAKE_BUILD_TYPE  Debug CACHE STRING "Choose the type of build" FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

if (CMAKE_BUILD_TYPE AND ( CMAKE_BUILD_TYPE STREQUAL "Debug"))
  # define variable for realm_defines.h
  set(DEBUG_REALM ON)
  # define variable for legion_defines.h
  set(DEBUG_LEGION ON)
endif()

set(BUILD_SHARED_LIBS OFF   CACHE BOOL   "Whether or not to build shared libraries instead of static")

#------------------------------------------------------------------------------#
# Architecture
#------------------------------------------------------------------------------#
if(BUILD_MARCH AND BUILD_MCPU)
  message(FATAL_ERROR "BUILD_MARCH and BUILD_MCPU are incompatible")
endif()

# Try -march first. On platforms that don't support it, GCC will issue
# a hard error, so we'll know not to use it.
# Default is "native", but explicitly setting BUILD_MARCH="" disables use of
#  the flag
if(BUILD_MARCH)
  set(INTERNAL_BUILD_MARCH ${BUILD_MARCH})
elseif(NOT DEFINED BUILD_MARCH)
  set(INTERNAL_BUILD_MARCH "native")
endif()

include(CheckCXXCompilerFlag)
if(INTERNAL_BUILD_MARCH)
  check_cxx_compiler_flag("-march=${INTERNAL_BUILD_MARCH}" COMPILER_SUPPORTS_MARCH)
  if(COMPILER_SUPPORTS_MARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${INTERNAL_BUILD_MARCH}")
  elseif(BUILD_MARCH)
    message(FATAL_ERROR "The flag -march=${INTERNAL_BUILD_MARCH} is not supported by the compiler")
  else()
    unset(INTERNAL_BUILD_MARCH)
  endif()
endif()

# Try -mcpu. We do this second because it is deprecated on x86, but
# GCC won't issue a hard error, so we can't tell if it worked or not.
if (NOT INTERNAL_BUILD_MARCH AND NOT DEFINED BUILD_MARCH)
  if(BUILD_MCPU)
    set(INTERNAL_BUILD_MCPU ${BUILD_MCPU})
  else()
    set(INTERNAL_BUILD_MCPU "native")
  endif()

  check_cxx_compiler_flag("-mcpu=${INTERNAL_BUILD_MCPU}" COMPILER_SUPPORTS_MCPU)
  if(COMPILER_SUPPORTS_MCPU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=${INTERNAL_BUILD_MCPU}")
  elseif(BUILD_MCPU)
    message(FATAL_ERROR "The flag -mcpu=${INTERNAL_BUILD_MCPU} is not supported by the compiler")
  else()
    unset(INTERNAL_BUILD_MCPU)
  endif()
endif()

# Add flags for Power architectures
check_cxx_compiler_flag("-maltivec -Werror" COMPILER_SUPPORTS_MALTIVEC)
if(COMPILER_SUPPORTS_MALTIVEC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maltivec")
endif()
check_cxx_compiler_flag("-mabi=altivec -Werror" COMPILER_SUPPORTS_MABI_ALTIVEC)
if(COMPILER_SUPPORTS_MABI_ALTIVEC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mabi=altivec")
endif()
check_cxx_compiler_flag("-mvsx -Werror" COMPILER_SUPPORTS_MVSX)
if(COMPILER_SUPPORTS_MVSX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mvsx")
endif()

# detect use of the cxx_defcheck wrapper and enable build checks
check_cxx_compiler_flag("--defcheck __test__" COMPILER_SUPPORTS_DEFCHECK)

# task registration across nodes often relies on being able to map function
#  pointers back to symbol names, so ask cmake to export symbols in binaries
set(CMAKE_ENABLE_EXPORTS ON)

#------------------------------------------------------------------------------#
# Optimizations
#------------------------------------------------------------------------------#

# For now we want the optimization flags to match on both normal make and cmake
# builds so we override the cmake defaults here for release, this changes
# -O3 to -O2 and removes -DNDEBUG
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g")

# add build types for thread/ub sanitizer goodness
set(CMAKE_CXX_FLAGS_TSAN
  "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=thread -DTSAN_ENABLED"
  CACHE STRING "additional C++ compiler flags for thread sanitizer" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_TSAN
  "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=thread"
  CACHE STRING "additional linker flags for thread sanitizer" FORCE)
set(CMAKE_CXX_FLAGS_UBSAN
  "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined"
  CACHE STRING "additional C++ compiler flags for UB sanitizer" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_UBSAN
  "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=undefined"
  CACHE STRING "additional linker flags for UB sanitizer" FORCE)
set(CMAKE_CXX_FLAGS_ASAN
  "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -DASAN_ENABLED"
  CACHE STRING "additional C++ compiler flags for address sanitizer" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_ASAN
  "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address"
  CACHE STRING "additional linker flags for address sanitizer" FORCE)

# if requested (and we know how to) make any warnings during the build fatal
option(Legion_BUILD_WARN_AS_ERROR "Consider warnings during build of Legion to be fatal errors" OFF)
if(Legion_BUILD_WARN_AS_ERROR)
  check_cxx_compiler_flag("-Wall -Werror" COMPILER_SUPPORTS_WALL_WERROR)
  if(COMPILER_SUPPORTS_WALL_WERROR)
    set(CXX_BUILD_WARNING_FLAGS -Wall -Werror)
  else()
    message(FATAL_ERROR "cannot determine how to convert warnings to errors in C++ compiler")
  endif()
else()
  set(CXX_BUILD_WARNING_FLAGS "")
endif()

#------------------------------------------------------------------------------#
# Minimum log level
#------------------------------------------------------------------------------#
set(Legion_OUTPUT_LEVEL "DEBUG" CACHE STRING "Compile time logging level")
set_property(CACHE Legion_OUTPUT_LEVEL PROPERTY STRINGS SPEW DEBUG INFO PRINT WARNING ERROR FATAL NONE)

# define variable for realm_defines.h
set(COMPILE_TIME_MIN_LEVEL LEVEL_${Legion_OUTPUT_LEVEL})

#------------------------------------------------------------------------------#
# OpenMP
#------------------------------------------------------------------------------#
option(Legion_USE_OpenMP "Use OpenMP" OFF)
if(Legion_USE_OpenMP)
  option(Legion_OpenMP_SYSTEM_RUNTIME "Use system's OpenMP runtime instead of internal Realm version" OFF)
  if(Legion_OpenMP_SYSTEM_RUNTIME)
    find_package(OpenMP REQUIRED)
  endif()

  option(Legion_OpenMP_GOMP_SUPPORT "Enable GOMP support for OpenMP" ON)
  option(Legion_OpenMP_KMP_SUPPORT "Enable KMP support for OpenMP" ON)

  # define variable for realm_defines.h
  set(REALM_USE_OPENMP ON)
  if(Legion_OpenMP_SYSTEM_RUNTIME)
    set(REALM_OPENMP_SYSTEM_RUNTIME ON)
  endif()
  if(Legion_OpenMP_GOMP_SUPPORT)
    set(REALM_OPENMP_GOMP_SUPPORT ON)
  endif()
  if(Legion_OpenMP_KMP_SUPPORT)
    set(REALM_OPENMP_KMP_SUPPORT ON)
  endif()
endif()

#------------------------------------------------------------------------------#
# Python
#------------------------------------------------------------------------------#
option(Legion_USE_Python "Use Python" OFF)
if(Legion_USE_Python)
  set(Legion_Python_Version "" CACHE STRING "Minimum Python version")

  # Note: Legion now supports several versions, so no need to be so forceful
  # set(Python_ADDITIONAL_VERSIONS 2.7)
  find_package(PythonInterp ${Legion_Python_Version} REQUIRED) # Needed to invoke setup.py for install
  find_package(PythonLibs ${Legion_Python_Version} REQUIRED)

  # define variables for realm_defines.h
  set(REALM_USE_PYTHON ON)
  # Set REALM_PYTHON_LIB to Python library's filename without the path
  get_filename_component(REALM_PYTHON_LIB "${PYTHON_LIBRARIES}" NAME)
  set(REALM_PYTHON_VERSION_MAJOR ${PYTHON_VERSION_MAJOR})

  string(REPLACE "." ";" PYTHON_VERSION_SPLIT ${PYTHONLIBS_VERSION_STRING})
  list(GET PYTHON_VERSION_SPLIT 0 PYTHON_VERSION_MAJOR)
endif()

set(BINDINGS_DEFAULT_MODULE "" CACHE STRING "module to load by default in Python bindings, if any")

#------------------------------------------------------------------------------#
# Kokkos configuration
#------------------------------------------------------------------------------#
option(Legion_USE_Kokkos "Enable support for the Kokkos runtime" OFF)
if(Legion_USE_Kokkos)
  # cmake uses -DKokkos_DIR=.../lib/cmake to look in non-standard locations
  #
  # starting with Kokkos 3.3.0, including a cuda-enabled Kokkos will mess with
  #  the global compilation settings unless you ask it not to by requesting
  #  the 'separable_compilation' component
  find_package(Kokkos REQUIRED OPTIONAL_COMPONENTS separable_compilation)

  message(STATUS "Kokkos version: ${Kokkos_VERSION}")

  # in order to build using Kokkos' exported compile options, we need to use
  #  the same compiler - newer versions of Kokkos will tell us, but for older
  #  versions, we need it from the configuration or the environment
  if(Kokkos_CXX_COMPILER)
    set(KOKKOS_CXX_COMPILER ${Kokkos_CXX_COMPILER} CACHE STRING
                            "C++ compiler used by Kokkos")
  else()
    set(KOKKOS_CXX_COMPILER $ENV{KOKKOS_CXX_COMPILER} CACHE STRING
                            "C++ compiler used by Kokkos")
  endif()
  if(NOT KOKKOS_CXX_COMPILER)
    message(FATAL_ERROR "to build correctly with Kokkos, the exact compiler used in the Kokkos build (typically set via CXX=... or -DCMAKE_CXX_COMPILER=...) must be provided in KOKKOS_CXX_COMPILER (either on the command line or from the environment)")
  endif()

  set(REALM_USE_KOKKOS ON)
endif()

#------------------------------------------------------------------------------#
# network interface configuration
#------------------------------------------------------------------------------#
set(Legion_NETWORKS "" CACHE STRING "Network backend(s) to use")
separate_arguments(Legion_NETWORKS)

# compatibility mode - Legion_USE_GASNet==ON -> Legion_NETWORKS=gasnet1
option(Legion_USE_GASNet "Enable the distributed GASNet backend" OFF)
if(Legion_USE_GASNet)
  list(APPEND Legion_NETWORKS gasnet1)
endif()

#------------------------------------------------------------------------------#
# GASNet configuration
#------------------------------------------------------------------------------#
if("${Legion_NETWORKS}" MATCHES ".*gasnet(1|ex).*")
  option(Legion_EMBED_GASNet "Embed a custom GASNet build in Realm library" OFF)
  if(Legion_EMBED_GASNet)
    if(NOT GASNet_CONDUIT)
      message(FATAL_ERROR "GASNet_CONDUIT must be set when using an embedded GASNet build")
    endif()
    # an embedded GASNet uses the stanfordlegion/gasnet helper repository for
    #  a build wrapper and curated-for-Legion GASNet configurations - it can
    #  be fetched from github directly (the default) or use a local (and
    #  perhaps modified) clone of the repository
    set(Legion_EMBED_GASNet_GITREPO
      "https://github.com/StanfordLegion/gasnet.git"
      CACHE STRING "URL for cloning StanfordLegion/gasnet repository")
    set(Legion_EMBED_GASNet_GITREF
      "master"
      CACHE STRING "Branch/tag/commit to use from StanfordLegion/gasnet repository")
    set(Legion_EMBED_GASNet_LOCALSRC "" CACHE STRING "Path to local clone of stanfordlegion/gasnet repository")
    set(Legion_EMBED_GASNet_VERSION "" CACHE STRING "Override GASNet version to build")
    set(Legion_EMBED_GASNet_CONFIGURE_ARGS "" CACHE STRING "Extra configuration arguments for GASNet")
  else()
    set(GASNet_THREADING par)
    set(GASNet_PREFERRED_CONDUITS aries gemini ibv)
    find_package(GASNet REQUIRED)
    if(NOT GASNet_THREADING STREQUAL "par")
      message(FATAL_ERROR "GASNet threading mode \"${GASNet_THREADING}\" is not currently supported by Legion")
    endif()
  endif()

  # define variable for realm_defines.h
  if("${Legion_NETWORKS}" MATCHES .*gasnetex.*)
    set(REALM_USE_GASNETEX ON)
  else()
    set(REALM_USE_GASNET1 ON)
  endif()
  string(TOUPPER ${GASNet_CONDUIT} CONDUIT)
  set(GASNET_CONDUIT_${CONDUIT} ON)
  # conduits other than udp support MPI interop
  if(NOT ${GASNet_CONDUIT} STREQUAL "udp")
    set(Legion_MPI_INTEROP ON)
  endif()
endif()

#------------------------------------------------------------------------------#
# MPI configuration
#------------------------------------------------------------------------------#
if("${Legion_NETWORKS}" MATCHES .*mpi.*)
  find_package(MPI REQUIRED)
  # define variable for realm_defines.h
  set(REALM_USE_MPI ON)
  set(Legion_MPI_INTEROP ON)
endif()

#------------------------------------------------------------------------------#
# UCX configuration
#------------------------------------------------------------------------------#
if("${Legion_NETWORKS}" MATCHES .*ucx.*)
  find_package(ucx 1.14.0 REQUIRED CONFIG)
  message(STATUS "Found UCX: ${UCX_INCLUDE_DIRS} ${UCX_LIBRARIES})")
  find_package(MPI REQUIRED) # for MPI bootstrap plugin
  # define variable for realm_defines.h
  set(REALM_USE_UCX ON)
  set(Legion_MPI_INTEROP ON)
endif()

#------------------------------------------------------------------------------#
# GPU configuration
#------------------------------------------------------------------------------#
option(Legion_USE_CUDA "Enable support for the CUDA runtime" OFF)
option(Legion_USE_HIP "Enable support for the HIP runtime on AMD/NVIDIA GPU" OFF)

set(Legion_HIP_TARGETS "ROCM" "CUDA")
mark_as_advanced(Legion_HIP_TARGETS)
set(Legion_HIP_TARGET "ROCM" CACHE STRING "Select HIP target ${Legion_HIP_TARGETS}")
set_property(CACHE Legion_HIP_TARGET PROPERTY STRINGS ${Legion_HIP_TARGETS})

if(Legion_USE_CUDA OR Legion_USE_HIP)

  option(Legion_GPU_REDUCTIONS "Use tasks to perform reductions on GPU" OFF)
  # define variable for legion_defines.h
  set(LEGION_GPU_REDUCTIONS ${Legion_GPU_REDUCTIONS})

  if(Legion_USE_CUDA OR (Legion_HIP_TARGET STREQUAL "CUDA"))

    # Find the CUDA toolkit
    find_package(CUDAToolkit REQUIRED)

    # Install the vendored FindCUDAToolkit.cmake to support CMake < v3.19
    install(FILES ${Legion_SOURCE_DIR}/cmake/FindCUDAToolkit.cmake
      DESTINATION ${CMAKE_INSTALL_DATADIR}/Legion/cmake
    )
    install(FILES ${Legion_SOURCE_DIR}/cmake/newcmake/FindCUDAToolkit.cmake
      DESTINATION ${CMAKE_INSTALL_DATADIR}/Legion/cmake/newcmake
    )

    # CMAKE_CUDA_RUNTIME_LIBRARY determines whether CUDA objs link to the static
    # or shared CUDA runtime library. Must be set before `enable_language(CUDA)`
    # https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_RUNTIME_LIBRARY.html
    # if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    #   if(NOT BUILD_SHARED_LIBS)
    #     set(CMAKE_CUDA_RUNTIME_LIBRARY STATIC)
    #   else()
    #     set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
    #   endif()
    # endif()

    set(Legion_CUDA_ARCH "" CACHE STRING
      "Comma (or semicolon) separated list of CUDA architectures to build for (e.g. 60,70,80,90)")

    # Ensure CMAKE_CUDA_ARCHITECTURES is set before `enable_language(CUDA)`
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      include(cmake/cuda_arch.cmake)
      if(Legion_CUDA_ARCH)
        # Support comma-delimited lists in addition to semicolons
        string(REPLACE "," ";" CMAKE_CUDA_ARCHITECTURES "${Legion_CUDA_ARCH}")
      endif()
      populate_cuda_archs_list(CMAKE_CUDA_ARCHITECTURES)
    endif()

    # Default CMAKE_CUDA_STANDARD to CMAKE_CXX_STANDARD if unset
    if(CMAKE_CXX_STANDARD AND NOT CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
    endif()

    # CMake < 3.18 doesn't recognize >=17 as a CUDA standard, so unset it before enabling the CUDA language
    if(CMAKE_VERSION VERSION_LESS_EQUAL "3.17" AND CMAKE_CUDA_STANDARD GREATER_EQUAL 17)
      unset(CMAKE_CUDA_STANDARD)
      unset(CMAKE_CUDA_STANDARD CACHE)
      if(NOT "${CMAKE_CUDA_FLAGS}" MATCHES ".*-std=.*")
        # CMake < 3.18 doesn't recognize >=17 as a CUDA standard, so set it manually
        string(APPEND CMAKE_CUDA_FLAGS " -std=c++${CMAKE_CXX_STANDARD}")
      endif()
    endif()

    # Enable the CUDA language
    enable_language(CUDA)

    # Work around a clangd bug by generating compile commands
    # with `-isystem <path>` instead of `-isystem=<path>`.
    # Must come after `enable_language(CUDA)`
    set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-isystem ")

    #------------------------------------------------------------------------------#
    # NVTX configuration
    #------------------------------------------------------------------------------#
    option(Legion_USE_NVTX "Enable profiling using NVTX" OFF)
    # define variable for realm_defines.h
    set(REALM_USE_NVTX ${Legion_USE_NVTX})
    mark_as_advanced(REALM_USE_NVTX)
  endif()
endif()

#------------------------------------------------------------------------------#
# CUDA configuration
#------------------------------------------------------------------------------#
include(CMakeDependentOption)

if(Legion_USE_CUDA)

  option(Legion_CUDA_DYNAMIC_LOAD "Load CUDA libraries at runtime" OFF)

  cmake_dependent_option(Legion_HIJACK_CUDART "Hijack and rewrite application calls into the CUDA runtime"
    # The default is to hijack unless Kokkos interop is enabled, or network is UCX.
    # `cmake_dependent_option` uses python ternary notation, i.e `X if <condition> else Y`.
    OFF "Legion_USE_Kokkos OR \"ucx\" IN_LIST Legion_NETWORKS" ON
  )

  if(Legion_CUDA_DYNAMIC_LOAD AND Legion_HIJACK_CUDART)
    # not compatible with runtime hijack
    message(FATAL_ERROR "Dynamic CUDA library loading (Legion_CUDA_DYNAMIC_LOAD) is not compatible with CUDA runtime hijack (Legion_HIJACK_CUDART)")
  endif()

  # define variable for legion_defines.h
  set(LEGION_USE_CUDA ON)
  # define variable for realm_defines.h
  set(REALM_USE_CUDA ON)
  set(REALM_CUDA_DYNAMIC_LOAD ${Legion_CUDA_DYNAMIC_LOAD})
  set(REALM_USE_CUDART_HIJACK ${Legion_HIJACK_CUDART})
endif()

#------------------------------------------------------------------------------#
# HIP configuration
#------------------------------------------------------------------------------#
if(Legion_USE_HIP)

  set(HIPCC_FLAGS ${HIPCC_FLAGS})

  if(Legion_HIP_TARGET STREQUAL "CUDA")
    list(APPEND HIPCC_FLAGS -D__HIP_PLATFORM_NVCC__)
  elseif(Legion_HIP_TARGET STREQUAL "ROCM")
    set(Legion_HIP_ARCH "" CACHE STRING "Comma-separated list of HIP architectures to build for (e.g. gfx906,gfx908)")

    if(BUILD_SHARED_LIBS)
      list(APPEND HIPCC_FLAGS -fPIC)
    endif()

    install(FILES ${Legion_SOURCE_DIR}/cmake/FindHIP.cmake
      DESTINATION ${CMAKE_INSTALL_DATADIR}/Legion/cmake
    )

    list(APPEND HIPCC_FLAGS -fno-strict-aliasing)

    if("${Legion_HIP_ARCH}" STREQUAL "")
      set(HIP_GENCODE "")
    else()
      set(HIP_GENCODE "--offload-target=${Legion_HIP_ARCH}")
    endif()
  endif()

  option(Legion_HIJACK_HIP "Enable HIJACK HIP" ON)

  # find the hip library
  find_package(HIP REQUIRED)
  # define variable for legion_defines.h
  set(LEGION_USE_HIP ON)
  # define variable for realm_defines.h
  set(REALM_USE_HIP ON)
  set(REALM_USE_HIP_HIJACK ${Legion_HIJACK_HIP})
endif()

#------------------------------------------------------------------------------#
# LLVM configuration
#------------------------------------------------------------------------------#
option(Legion_USE_LLVM "Use LLVM JIT operations" OFF)
option(Legion_ALLOW_MISSING_LLVM_LIBS "Allow LLVM libraries to be missing at runtime" OFF)
option(Legion_LINK_LLVM_LIBS "Link LLVM libraries into Realm" ON)
if(Legion_USE_LLVM)
  set(Legion_LLVM_COMPONENTS irreader jit mcjit x86)
  find_package(LLVM REQUIRED COMPONENTS ${Legion_LLVM_COMPONENTS})
  install(FILES ${Legion_SOURCE_DIR}/cmake/FindLLVM.cmake
    DESTINATION ${CMAKE_INSTALL_DATADIR}/Legion/cmake
  )

  # define variables for realm_defines.h
  set(REALM_USE_LLVM ON)
  if(Legion_ALLOW_MISSING_LLVM_LIBS)
    set(REALM_ALLOW_MISSING_LLVM_LIBS ON)
  endif()
endif()

#------------------------------------------------------------------------------#
# HDF5 configuration
#------------------------------------------------------------------------------#
option(Legion_USE_HDF5 "Enable support for HDF5" OFF)
if(Legion_USE_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS C)

  # define variable for both legion_defines.h and realm_defines.h
  set(LEGION_USE_HDF5 ON)
  set(REALM_USE_HDF5 ON)
endif()

#------------------------------------------------------------------------------#
# libdl configuration
#------------------------------------------------------------------------------#
# NOTE: although the name of the variable is "libdl", this is dynamic linking
#  support on any system, even if it doesn't have a libdl
option(Legion_USE_LIBDL "Enable run-time support for dynamic shared objects" ON)
if(Legion_USE_LIBDL)
  # define variable for legion_defines.h
  set(LEGION_USE_LIBDL ON)
  # define variable for realm_defines.h
  set(REALM_USE_LIBDL ON)
endif()

#------------------------------------------------------------------------------#
# HWLOC configuration
#------------------------------------------------------------------------------#
option(Legion_USE_HWLOC "Use hwloc for topology awareness" OFF)
if(Legion_USE_HWLOC)
  find_package(HWLOC REQUIRED)
  install(FILES ${Legion_SOURCE_DIR}/cmake/FindHWLOC.cmake
    DESTINATION ${CMAKE_INSTALL_DATADIR}/Legion/cmake
  )
  set(REALM_USE_HWLOC ON)
endif()

#------------------------------------------------------------------------------#
# PAPI configuration
#------------------------------------------------------------------------------#
option(Legion_USE_PAPI "Use PAPI for performance measurements" OFF)
if(Legion_USE_PAPI)
  # no FindPAPI in cmake by default, so just find the lib and include path
  #  ourselves
  find_library(PAPI_LIBRARIES NAMES libpapi.so libpapi.a papi
    HINTS ${PAPI_ROOT}/lib $ENV{PAPI_ROOT}/lib ENV PAPI_LIB_DIR)
  find_path(PAPI_INCLUDE_DIRS NAMES papi.h
    HINTS ${PAPI_ROOT}/include $ENV{PAPI_ROOT}/include ENV PAPI_INC_DIR)
  if(PAPI_LIBRARIES AND PAPI_INCLUDE_DIRS)
    set(REALM_USE_PAPI ON)
  else()
    message(FATAL_ERROR
      " Could not find PAPI - set PAPI_{ROOT,INC_DIR,LIB_DIR} if needed -- got:\n"
      "   PAPI_LIBRARIES: ${PAPI_LIBRARIES}\n"
      "   PAPI_INCLUDE_DIRS: ${PAPI_INCLUDE_DIRS}")
  endif()
endif()

#------------------------------------------------------------------------------#
# zlib configuration
#------------------------------------------------------------------------------#
if(NOT DEFINED Legion_USE_ZLIB)
  # neither ON nor OFF at user's request, so use it if we can find it
  find_package(ZLIB)
  if(ZLIB_FOUND)
    set(Legion_USE_ZLIB ON)
  else()
    set(Legion_USE_ZLIB OFF)
  endif()
else()
  set(Legion_USE_ZLIB "$CACHE{Legion_USE_ZLIB}")
endif()
set(Legion_USE_ZLIB "${Legion_USE_ZLIB}" CACHE BOOL "Enable support for zlib")
if(Legion_USE_ZLIB)
  find_package(ZLIB REQUIRED)
  # define variable for legion_defines.h
  set(LEGION_USE_ZLIB ON)
endif()

#------------------------------------------------------------------------------#
# Fortran configuration
#------------------------------------------------------------------------------#
option(Legion_USE_Fortran "Enable support for Fortran" OFF)
if(Legion_USE_Fortran)
  enable_language(Fortran)
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")
endif()

#------------------------------------------------------------------------------#
# Miscelaneous other options
#------------------------------------------------------------------------------#
set(Legion_MAX_DIM 3 CACHE STRING "Maximum number of dimensions")
set_property(CACHE Legion_MAX_DIM PROPERTY STRINGS 1 2 3 4 5 6 7 8 9)
mark_as_advanced(Legion_MAX_DIM)

# define variable for legion_defines.h
set(LEGION_MAX_DIM ${Legion_MAX_DIM})
set(REALM_MAX_DIM ${Legion_MAX_DIM})

# legion uses statically-sized arrays for performance in some cases - make
#  sure they're big enough for your use case
set(Legion_MAX_FIELDS 512 CACHE STRING
    "Maximum number of fields allocated to a single field space -- MUST be a power of two.")
set(Legion_MAX_NUM_NODES 1024 CACHE STRING
    "Maximum number of nodes supported by the runtime -- MUST be a power of two.")
set(Legion_MAX_NUM_PROCS 64 CACHE STRING
    "Maximum number of processors (per node) supported by the runtime -- MUST be a power of two.")

# check that they're all powers of two

function(is_power_of_two x ret)
  set(${ret} FALSE PARENT_SCOPE)
  if (${x} LESS_EQUAL 0)
    return()
  endif()
  math(EXPR y "${x} & (${x} - 1)")
  if (${y} EQUAL 0)
    set(${ret} TRUE PARENT_SCOPE)
  endif()
endfunction()

is_power_of_two(${Legion_MAX_FIELDS} is_pow_two)
if (NOT is_pow_two)
  message(FATAL_ERROR "Legion_MAX_FIELDS must be a power of two.")
endif()

is_power_of_two(${Legion_MAX_NUM_NODES} is_pow_two)
if (NOT is_pow_two)
  message(FATAL_ERROR "Legion_MAX_NUM_NODES must be a power of two.")
endif()

is_power_of_two(${Legion_MAX_NUM_PROCS} is_pow_two)
if (NOT is_pow_two)
  message(FATAL_ERROR "Legion_MAX_NUM_PROCS must be a power of two.")
endif()

# define variables for legion_defines.h
set(LEGION_MAX_FIELDS ${Legion_MAX_FIELDS})
set(LEGION_MAX_NUM_NODES ${Legion_MAX_NUM_NODES})
set(LEGION_MAX_NUM_PROCS ${Legion_MAX_NUM_PROCS})

option(Legion_WARNINGS_FATAL "Make all runtime warnings fatal" OFF)
set(LEGION_WARNINGS_FATAL ${Legion_WARNINGS_FATAL})

option(Legion_SPY "Enable detailed logging for Legion Spy" OFF)
set(LEGION_SPY ${Legion_SPY})

option(Legion_BOUNDS_CHECKS "Enable bounds checking in Legion accessors" OFF)
set(LEGION_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS})

option(Legion_PRIVILEGE_CHECKS "Enable privilege checking in Legion accessors" OFF)
set(LEGION_PRIVILEGE_CHECKS ${Legion_PRIVILEGE_CHECKS})

# hiding internal symbols is much more complicated in Windows
if(NOT WIN32)
  option(REALM_LIMIT_SYMBOL_VISIBILITY "hide private symbols in realm" ON)
endif()

#------------------------------------------------------------------------------#
# Hacks for building only Realm
#------------------------------------------------------------------------------#
option(Legion_BUILD_REALM_ONLY "HACK: only build realm" OFF)
option(Legion_BUILD_REALM_TESTS "HACK: build realm-only tests" OFF)

#------------------------------------------------------------------------------#
# Runtime library targets
#------------------------------------------------------------------------------#
add_subdirectory(runtime)

#------------------------------------------------------------------------------#
# Tools
#------------------------------------------------------------------------------#
add_subdirectory(tools)

#------------------------------------------------------------------------------#
# Documentation
#------------------------------------------------------------------------------#
add_subdirectory(doc)

#------------------------------------------------------------------------------#
# configure header
#------------------------------------------------------------------------------#
# Checking for all defines in the CXX Flags

string(REPLACE " " ";" FLAGS_LIST "${CMAKE_CXX_FLAGS}")
FOREACH(FLAG ${FLAGS_LIST})
  string (FIND ${FLAG} "-D" START_STR)
  if (${START_STR} EQUAL "0")
  string(REPLACE "-D" "" NEW_DEFINE ${FLAG})
  set (${NEW_DEFINE} ON)
  endif()
endforeach()


#------------------------------------------------------------------------------#
# Build-tree package generation
#------------------------------------------------------------------------------#
include(CMakePackageConfigHelpers)

export(EXPORT LegionTargets
  NAMESPACE Legion::
  FILE ${Legion_BINARY_DIR}/LegionTargets.cmake
)
install(EXPORT LegionTargets
  NAMESPACE Legion::
  FILE LegionTargets.cmake
  DESTINATION ${CMAKE_INSTALL_DATADIR}/Legion/cmake
)

configure_file(
  cmake/LegionConfigCommon.cmake.in
  ${Legion_BINARY_DIR}/LegionConfigCommon.cmake
  @ONLY
)
install(FILES ${Legion_BINARY_DIR}/LegionConfigCommon.cmake
  DESTINATION ${CMAKE_INSTALL_DATADIR}/Legion/cmake
)

configure_file(
  cmake/LegionConfig-build.cmake.in
  ${Legion_BINARY_DIR}/LegionConfig.cmake
  @ONLY
)

write_basic_package_version_file(
  ${Legion_BINARY_DIR}/LegionConfig-version.cmake
  VERSION ${Legion_VERSION}
  COMPATIBILITY ExactVersion)

install(FILES ${Legion_BINARY_DIR}/LegionConfig-version.cmake
  DESTINATION ${CMAKE_INSTALL_DATADIR}/Legion/cmake
)

install(FILES cmake/LegionConfig-install.cmake
  DESTINATION ${CMAKE_INSTALL_DATADIR}/Legion/cmake
  RENAME LegionConfig.cmake
)

#------------------------------------------------------------------------------#
# Install into the CMake pacakge registry if explicitly selected to do so
#------------------------------------------------------------------------------#
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON CACHE INTERNAL "Disable the export(PACKAGE) command.")
option(Legion_USE_PACKAGE_REGISTRY "Register the build tree with the CMake package registry" OFF)
mark_as_advanced(Legion_USE_PACKAGE_REGISTRY)
if(Legion_USE_PACKAGE_REGISTRY)
  set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY OFF)
endif()
export(PACKAGE Legion)

#------------------------------------------------------------------------------#
# Examples and Applications
#------------------------------------------------------------------------------#
option(Legion_BUILD_ALL "Build all Legion bindings, examples, tutorials and tests" OFF)
option(Legion_BUILD_APPS "Build Legion sample applications" OFF)
option(Legion_BUILD_BINDINGS "Build Legion language bindings" OFF)
option(Legion_BUILD_EXAMPLES "Build Legion examples" OFF)
option(Legion_BUILD_TUTORIAL "Build Legion tutorial" OFF)
option(Legion_BUILD_TESTS "Build Legion tests" OFF)
option(Legion_BUILD_JUPYTER "Build Legion jupyter notebook support" OFF)
option(Legion_ENABLE_TESTING "Build and copy testing stuff" OFF)
option(Legion_BUILD_RUST_PROFILER "Build Rust version of Legion profiler" OFF)
if(Legion_ENABLE_TESTING)
  enable_testing()
  # if Legion_TEST_LAUNCHER is set, split the string into a list
  separate_arguments(Legion_TEST_LAUNCHER)
  separate_arguments(Legion_TEST_ARGS)
  # if we are using kokkos and cuda, we must hardcode the use of 1 gpu right now
  if(Legion_USE_Kokkos AND Legion_USE_CUDA)
    list(APPEND Legion_TEST_ARGS -ll:gpu 1)
  endif()
  # similar restrictions for openmp
  if(Legion_USE_Kokkos AND Legion_USE_OpenMP)
    list(APPEND Legion_TEST_ARGS -ll:ocpu 1 -ll:onuma 0)
  endif()
endif()
if(Legion_BUILD_ALL OR Legion_BUILD_APPS OR Legion_BUILD_BINDINGS OR Legion_BUILD_EXAMPLES OR Legion_BUILD_TUTORIAL OR Legion_BUILD_TESTS)
  # Make a namespaced alias so in-build examples can use it the same way as if
  # it were imported
  add_library(Legion::Legion ALIAS Legion)
  add_library(Legion::Realm ALIAS Realm)

  if(Legion_HIP_TARGET STREQUAL "ROCM")
    set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} ${HIPCC_FLAGS} ${HIP_GENCODE}")
  endif()

  if(Legion_BUILD_ALL OR Legion_BUILD_APPS)
    add_subdirectory(apps)
  endif()
  if(Legion_BUILD_ALL OR Legion_BUILD_BINDINGS)
    add_subdirectory(bindings)
  endif()
  if(Legion_BUILD_ALL OR Legion_BUILD_EXAMPLES)
    add_subdirectory(examples)
  endif()
  if(Legion_BUILD_ALL OR Legion_BUILD_TUTORIAL)
    add_subdirectory(tutorial)
  endif()
  if(Legion_BUILD_ALL OR Legion_BUILD_TESTS)
    add_subdirectory(test)
  endif()
  if(Legion_BUILD_ALL OR Legion_BUILD_JUPYTER)
    add_subdirectory(jupyter_notebook)
  endif()
elseif(Legion_BUILD_REALM_TESTS)
  # temporary hackery to just enable the realm tests
  add_library(Legion::Realm ALIAS Realm)
  add_subdirectory(test/realm)
endif()

#------------------------------------------------------------------------------#
# Complex support
#------------------------------------------------------------------------------#

if(Legion_BUILD_ALL OR Legion_BUILD_BINDINGS)
  option(Legion_REDOP_COMPLEX "Use reduction operators for complex types" ON)
else()
  option(Legion_REDOP_COMPLEX "Use reduction operators for complex types" OFF)
endif()
if((Legion_BUILD_ALL OR Legion_BUILD_BINDINGS) AND NOT Legion_REDOP_COMPLEX)
  message(FATAL_ERROR "Bindings require Legion_REDOP_COMPLEX to be set")
endif()
if(Legion_REDOP_COMPLEX)
  # define variable for legion_defines.h
  set(LEGION_REDOP_COMPLEX ON)
endif()
mark_as_advanced(Legion_REDOP_COMPLEX)
if(Legion_REDOP_HALF)
  # define variable for legion_defines.h
  set(LEGION_REDOP_HALF ON)
endif()
mark_as_advanced(Legion_REDOP_HALF)

if(Legion_REDOP_COMPLEX AND Legion_USE_HIP)
  set(HIP_THRUST_ROOT_DIR "" CACHE STRING "Location of HIP THRUST library")
  if(NOT HIP_THRUST_ROOT_DIR)
    message(FATAL_ERROR "When using HIP, complex support requires that the HIP version of Thrust be installed")
  endif()
endif()

#------------------------------------------------------------------------------#
# Build and install legion_prof_rs
#------------------------------------------------------------------------------#

if(Legion_BUILD_ALL OR Legion_BUILD_RUST_PROFILER)
  find_package(Corrosion 0.3.2 QUIET)

  if(NOT Corrosion_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      Corrosion
      GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
      GIT_TAG 020d11f092314a58942891f14b4cd20c0edff80a # v0.3.2
    )
    # Set any global configuration variables such as `Rust_TOOLCHAIN` before this line!
    FetchContent_MakeAvailable(Corrosion)
  endif()

  corrosion_import_crate(
    MANIFEST_PATH tools/legion_prof_rs/Cargo.toml
    ALL_FEATURES
    PROFILE release
  )

  install(
    PROGRAMS
      ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/legion_prof
    TYPE
      BIN
    PERMISSIONS
      OWNER_EXECUTE OWNER_WRITE OWNER_READ
      GROUP_EXECUTE GROUP_WRITE GROUP_READ
      WORLD_EXECUTE WORLD_READ)
endif()

#------------------------------------------------------------------------------#
# Configure definition headers
#------------------------------------------------------------------------------#

configure_file(${PROJECT_SOURCE_DIR}/cmake/legion_defines.h.in
  ${PROJECT_BINARY_DIR}/runtime/legion_defines.h @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/runtime/legion_defines.h
  DESTINATION  ${CMAKE_INSTALL_INCLUDEDIR})

configure_file(${PROJECT_SOURCE_DIR}/cmake/realm_defines.h.in
  ${PROJECT_BINARY_DIR}/runtime/realm_defines.h @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/runtime/realm_defines.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  FILES
    ${PROJECT_SOURCE_DIR}/tools/legion_prof.py
    ${PROJECT_SOURCE_DIR}/tools/legion_util.py
    ${PROJECT_SOURCE_DIR}/tools/legion_spy.py
  DESTINATION
    ${CMAKE_INSTALL_BINDIR}
  PERMISSIONS
    OWNER_EXECUTE OWNER_WRITE OWNER_READ
    GROUP_EXECUTE GROUP_WRITE GROUP_READ
    WORLD_EXECUTE WORLD_READ)

install(
  FILES
    ${PROJECT_SOURCE_DIR}/tools/legion_serializer.py
    ${PROJECT_SOURCE_DIR}/tools/legion_prof_copy.html.template
  DESTINATION
    ${CMAKE_INSTALL_BINDIR}
  PERMISSIONS
    OWNER_WRITE OWNER_READ
    GROUP_WRITE GROUP_READ
    WORLD_READ)

install(
  DIRECTORY
    ${PROJECT_SOURCE_DIR}/tools/serializer_examples
    ${PROJECT_SOURCE_DIR}/tools/legion_prof_files
  DESTINATION
    ${CMAKE_INSTALL_BINDIR}
  USE_SOURCE_PERMISSIONS)

#------------------------------------------------------------------------------#
# vim: set tabstop=2 shiftwidth=2 expandtab :
#------------------------------------------------------------------------------#
