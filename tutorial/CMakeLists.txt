#------------------------------------------------------------------------------#
# Copyright 2022 Los Alamos National Security, LLC
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

if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

# enable build warnings for all tutorials/examples/tests/etc.
add_compile_options(${CXX_BUILD_WARNING_FLAGS})

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
if(Legion_USE_Fortran)
  add_subdirectory(fortran)
endif()
