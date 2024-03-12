#------------------------------------------------------------------------------#
# Copyright 2024 NVIDIA Corp.
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

add_subdirectory(hello_world)
add_subdirectory(machine_model)
add_subdirectory(events)
add_subdirectory(region_instances)
add_subdirectory(deferred_allocation)
add_subdirectory(index_space_ops)
add_subdirectory(index_space_copy_fill)
add_subdirectory(reductions)
add_subdirectory(barrier)
add_subdirectory(subgraph)
add_subdirectory(reservation)
add_subdirectory(completion_queue)
add_subdirectory(profiling)
# FIXME: https://github.com/StanfordLegion/legion/issues/1475
if (REALM_USE_CUDA)# OR REALM_USE_HIP)
  add_subdirectory(cuda_interop)
endif ()
