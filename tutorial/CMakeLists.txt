#------------------------------------------------------------------------------#
# Copyright 2023 Los Alamos National Security, LLC
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

if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

if(CMAKE_CXX_STANDARD AND NOT CMAKE_CUDA_STANDARD)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  elseif(NOT "${CMAKE_CUDA_FLAGS}" MATCHES ".*-std=.*")
    # CMake < 3.18 doesn't recognize >=17 as a CUDA standard, so set it manually
    string(APPEND CMAKE_CUDA_FLAGS " -std=c++${CMAKE_CXX_STANDARD}")
  endif()
endif()

add_subdirectory(00_hello_world)
add_subdirectory(01_tasks_and_futures)
add_subdirectory(02_index_tasks)
add_subdirectory(03_global_vars)
add_subdirectory(04_logical_regions)
add_subdirectory(05_physical_regions)
add_subdirectory(06_privileges)
add_subdirectory(07_partitioning)
add_subdirectory(08_multiple_partitions)
add_subdirectory(09_custom_mapper)
add_subdirectory(realm)
if(Legion_USE_Fortran)
  add_subdirectory(fortran)
endif()
