#------------------------------------------------------------------------------#
# Copyright 2016 Kitware, Inc.
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

cmake_minimum_required(VERSION 3.1)
project(Legion)

#------------------------------------------------------------------------------#
# Some boilerplate to setup nice output directories
#------------------------------------------------------------------------------#
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

#------------------------------------------------------------------------------#
# More boilerplate to deal with build type and lib type
#------------------------------------------------------------------------------#
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Whether or not to build shared libraries instead of static")

#------------------------------------------------------------------------------#
# Minimum log level
#------------------------------------------------------------------------------#
set(Legion_OUTPUT_LEVEL "DEBUG" CACHE STRING "Compile time logging level")
set_property(CACHE Legion_OUTPUT_LEVEL PROPERTY STRINGS SPEW DEBUG INFO PRINT WARNING ERROR FATAL NONE)

#------------------------------------------------------------------------------#
# HWLOC configuration
#------------------------------------------------------------------------------#
option(Legion_USE_HWLOC "Use hwloc for topology awareness" OFF)
if(Legion_USE_HWLOC)
  find_package(HWLOC REQUIRED)
endif()

#------------------------------------------------------------------------------#
# GASNet configuration
#------------------------------------------------------------------------------#
option(Legion_USE_GASNet "Enable the distributed GASNet backend" OFF)
if(Legion_USE_GASNet)
  set(GASNet_THREADING "par" CACHE STRING "GASNet Threading model to use")
  set_property(CACHE GASNet_THREADING PROPERTY STRINGS seq par parsync)
  mark_as_advanced(GASNet_THREADING)
  if(NOT GASNet_THREADING STREQUAL "par")
    message(FATAL_ERROR "GASNet threading mode \"${GASNet_THREADING}\" is not currently supportedi by Legion")
  endif()
  set(GASNet_CONDUIT "" CACHE STRING "GASNet communication conduit to use")
  mark_as_advanced(GASNet_CONDUIT)
  find_package(GASNet REQUIRED)
endif()

#------------------------------------------------------------------------------#
# LLVM configuration
#------------------------------------------------------------------------------#
option(Legion_USE_LLVM "Use LLVM JIT oerations" OFF)
if(Legion_USE_LLVM)
  set(Legion_LLVM_COMPONENTS irreader jit mcjit x86)
  find_package(LLVM REQUIRED COMPONENTS ${Legion_LLVM_COMPONENTS})
endif()

#------------------------------------------------------------------------------#
# CUDA configuration
#------------------------------------------------------------------------------#
option(Legion_USE_CUDA "Enable support for the CUDA runtime" OFF)
if(Legion_USE_CUDA)
  if(NOT BUILD_SHARED_LIBS)
    set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
  else()
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
  endif()
  find_package(CUDA REQUIRED)
endif()

#------------------------------------------------------------------------------#
# MIscelaneous other options
#------------------------------------------------------------------------------#
option(Legion_USE_SHARED_LOWLEVEL     "Use shared-memory runtime" OFF)
option(Legion_USE_ALTERNATIVE_MAPPERS "Use alternative mappers" OFF)

#------------------------------------------------------------------------------#
# Runtime library targets
#------------------------------------------------------------------------------#
add_subdirectory(runtime)

export(EXPORT LegionTargets
  FILE ${Legion_BINARY_DIR}/LegionTargets.cmake
)
configure_file(
  cmake/LegionConfig.cmake.in
  ${Legion_BINARY_DIR}/LegionConfig.cmake
)
install(EXPORT LegionTargets
  FILE LegionTargets.cmake
  DESTINATION lib/cmake/Legion
)
install(FILES ${Legion_BINARY_DIR}/LegionConfig.cmake
  DESTINATION lib/cmake/Legion
)

#------------------------------------------------------------------------------#
# Examples
#------------------------------------------------------------------------------#

option(Legion_BUILD_EXAMPLES "Build Legion examples" OFF)
if(Legion_BUILD_EXAMPLES)
  set(Legion_DIR ${Legion_BINARY_DIR})
  add_subdirectory(examples)
endif()

#------------------------------------------------------------------------------#
# vim: set tabstop=2 shiftwidth=2 expandtab :
#------------------------------------------------------------------------------#
