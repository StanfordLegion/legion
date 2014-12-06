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
project(LegionApps)

# Some boilerplate to setup nice output directories
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LegionApps_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LegionApps_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LegionApps_BINARY_DIR}/bin)
endif()

# Make sure a buld type is always set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build" FORCE)
endif()

find_package(Legion REQUIRED)

#add_subdirectory(circuit)

# Delaunay app uses explicit SSE intrinsics and thus can onlhy be built on
# x86_64 architectures
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  # Currently fails due to missing legion_vector.h
  #add_subdirectory(delaunay)
endif()

add_subdirectory(saxpy)
