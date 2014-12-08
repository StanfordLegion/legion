#=============================================================================
# Copyright 2014 Kitware, Inc.
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
#=============================================================================

cmake_minimum_required(VERSION 2.8.12)
project(Legion)

# Some boilerplate to setup nice output directories
set(CMAKE_MODULE_PATH "${Legion_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${Legion_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${Legion_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Legion_BINARY_DIR}/bin)
endif()

# Make sure a buld type is always set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build" FORCE)
endif()

option(BUILD_SHARED_LIBS "Whether to build shared or static libraries" OFF)

set(Legion_INSTALL_CMAKE_DIR "lib/cmake/Legion"
  CACHE STRING "The install location for the CMake package config files"
)
mark_as_advanced(LEGION_INSTALL_CMAKE_DIR)

# The CUDA backend can only be enabled if the shared backend is disabled
include(CMakeDependentOption)
option(Legion_USE_GASNet "Whether or not to enable GASNet support" OFF)
CMAKE_DEPENDENT_OPTION(
  Legion_USE_CUDA "Whether or not to enable CUDA support" OFF
  "Legion_USE_GASNet" ON
)
option(Legion_USE_ALTERNATE_MAPPERS
  "Whether or not to enable alternate mappers" ON)
set(Legion_RT_NAME "legion-rt" CACHE STRING
  "Output name of the runtime library")

add_subdirectory(runtime)
configure_file(
  ${Legion_SOURCE_DIR}/cmake/LegionConfig.cmake.in
  ${Legion_BINARY_DIR}/LegionConfig.cmake
  @ONLY
)
install(FILES ${Legion_BINARY_DIR}/LegionConfig.cmake
  DESTINATION ${Legion_INSTALL_CMAKE_DIR}
)

# Set the Legion_DIR variable so find_package can be used by all the other
# sub-projects
set(Legion_DIR ${Legion_BINARY_DIR})

option(BUILD_EXAMPLES "Whether or not to build tutorial examples" ON)
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

option(BUILD_APPS "Whether or not to build application examples" ON)
if(BUILD_APPS)
  add_subdirectory(apps)
endif()
