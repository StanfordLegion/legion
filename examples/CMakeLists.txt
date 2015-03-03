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
project(LegionExamples)

# Some boilerplate to setup nice output directories
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LegionExamples_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LegionExamples_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LegionExamples_BINARY_DIR}/bin)
endif()

# Make sure a buld type is always set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build" FORCE)
endif()

find_package(Legion REQUIRED)

add_subdirectory(00_hello_world)
add_subdirectory(01_tasks_and_futures)
add_subdirectory(02_index_tasks)
add_subdirectory(03_global_vars)
add_subdirectory(04_logical_regions)
add_subdirectory(05_physical_regions_and_accessors)
add_subdirectory(06_privileges)
add_subdirectory(07_partitioning)
add_subdirectory(08_multiple_partitions)
add_subdirectory(09_custom_mappers)
#add_subdirectory(full_circuit)
#add_subdirectory(full_ghost)
