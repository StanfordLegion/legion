#------------------------------------------------------------------------------#
# Copyright 2022 Kitware, Inc.
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

add_subdirectory(attach_file)
add_subdirectory(circuit)
add_subdirectory(dynamic_registration)
add_subdirectory(ghost)
add_subdirectory(ghost_pull)
add_subdirectory(realm_saxpy)
add_subdirectory(realm_stencil)
add_subdirectory(spmd_cgsolver)
add_subdirectory(virtual_map)
add_subdirectory(attach_2darray_c_fortran_layout)
add_subdirectory(attach_array_daxpy)
add_subdirectory(attach_index_space)
add_subdirectory(predication)
add_subdirectory(inline_tasks)
add_subdirectory(local_function_tasks)
add_subdirectory(layout_constraints)
add_subdirectory(provenance)
add_subdirectory(implicit_top_task)
add_subdirectory(future_map_transforms)

if(Legion_MPI_INTEROP)
  add_subdirectory(mpi_interop)
  add_subdirectory(mpi_with_ctrl_repl)
endif()

if(Legion_USE_CUDA)
  add_subdirectory(thrust_interop)
  add_subdirectory(future_instance)
endif()

if(Legion_USE_Python)
  add_subdirectory(python_interop)
endif()

if(Legion_USE_OpenMP)
  add_subdirectory(omp_saxpy)
endif()

if(Legion_USE_Kokkos)
  add_subdirectory(kokkos_saxpy)
endif()
