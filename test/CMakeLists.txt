#------------------------------------------------------------------------------#
# Copyright 2026 Stanford University, NVIDIA
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

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

add_subdirectory(attach_file_mini)
add_subdirectory(output_requirements)
add_subdirectory(reduce_future)
add_subdirectory(ctrl_repl_safety)
add_subdirectory(rendering)
#add_subdirectory(realm)
#add_subdirectory(realm/c)
add_subdirectory(gather_perf)
add_subdirectory(legion_redop_test)
add_subdirectory(disjoint_complete)
add_subdirectory(trie)
add_subdirectory(nested_replication)
add_subdirectory(mapper)
add_subdirectory(auto_tracing)
if(NOT Legion_NETWORKS)
  add_subdirectory(empty_futures)
endif()

if(Legion_USE_PREALM)
  add_subdirectory(prealm)
endif()

if(Legion_USE_HDF5)
  add_subdirectory(hdf_attach_subregion_parallel)
endif()

if(Legion_USE_Python)
  add_subdirectory(python_bindings)
endif()

if(Legion_NETWORKS)
  add_subdirectory(bug954)
endif()
