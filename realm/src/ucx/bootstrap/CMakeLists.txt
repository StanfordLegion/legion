#------------------------------------------------------------------------------#
# Copyright 2025 Stanford, NVIDIA Corp..
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

if(NOT Realm_VERSION)
  message(FATAL_ERROR "Realm version not defined!")
endif()

project(
  RealmUcxBoostrap
  LANGUAGES C
  VERSION ${Realm_VERSION}
  DESCRIPTION "UCX Bootloader for Realm"
)

option(BOOTSTRAP_ENABLE_MPI "Compile with MPI support" OFF)
option(REALM_INSTALL "Enable installation support" OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#region UCX backends
set(UCX_BOOTSTRAP_ROOT_DIR ../../../../runtime/realm/ucx/bootstrap)
cmake_path(ABSOLUTE_PATH UCX_BOOTSTRAP_ROOT_DIR NORMALIZE)

set(UCX_BACKENDS)
macro(add_backend_plugin backend)
  # Arguments are:
  # 1) Backend name
  # 2) List of libraries to link to
  add_library(
    realm_ucp_bootstrap_${backend} MODULE "${UCX_BOOTSTRAP_ROOT_DIR}/bootstrap_${backend}.c"
  )
  target_link_libraries(realm_ucp_bootstrap_${backend} ${ARGN})
  set_target_properties(
    realm_ucp_bootstrap_${backend}
    PROPERTIES PREFIX ""
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}
  )
  list(APPEND UCX_BACKENDS ${backend})
endmacro()

if(BOOTSTRAP_ENABLE_MPI)
  find_package(MPI)
  add_backend_plugin(mpi MPI::MPI_C)
endif()
#endregion

if(REALM_INSTALL)
  install(TARGETS ${UCX_BACKENDS})
endif()
