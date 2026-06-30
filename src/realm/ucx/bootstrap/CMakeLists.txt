# Copyright 2025 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(
  RealmUcxBoostrap
  LANGUAGES C CXX
  DESCRIPTION "UCX Bootloaders for Realm"
)

if(NOT Realm_VERSION)
  find_package(Realm REQUIRED)
endif()
if(NOT Realm_VERSION_MAJOR)
  string(REGEX MATCH "([0-9]+)" _ ${Realm_VERSION})
  set(Realm_VERSION_MAJOR ${CMAKE_MATCH_1})
endif()

find_package(MPI QUIET)
option(UCX_BOOTSTRAP_ENABLE_MPI "Compile with MPI support" ${MPI_FOUND})
option(UCX_BOOTSTRAP_INSTALL "Enable installation support" OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#region UCX backends
set(UCX_BOOTSTRAP_ROOT_DIR ${PROJECT_SOURCE_DIR})
cmake_path(ABSOLUTE_PATH UCX_BOOTSTRAP_ROOT_DIR NORMALIZE)

set(UCX_BACKENDS)

function(add_backend_plugin backend)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "NAME" "SOURCES;LIBRARIES")
  add_library(${arg_NAME} MODULE ${arg_SOURCES})
  target_link_libraries(${arg_NAME} ${arg_LIBRARIES})
  set_target_properties(
    ${arg_NAME}
    PROPERTIES PREFIX ""
               VERSION ${Realm_VERSION}
               MACHO_CURRENT_VERSION 0
               MACHO_COMPATIBILITY_VERSION 0
               SOVERSION ${Realm_VERSION_MAJOR}
  )
  list(APPEND UCX_BACKENDS ${arg_NAME})
  set(UCX_BACKENDS
      ${UCX_BACKENDS}
      PARENT_SCOPE
  )
endfunction()

if(UCX_BOOTSTRAP_ENABLE_MPI)
  find_package(MPI REQUIRED)
  add_backend_plugin(
    NAME realm_ucp_bootstrap_mpi SOURCES ${UCX_BOOTSTRAP_ROOT_DIR}/bootstrap_mpi.c LIBRARIES
    MPI::MPI_C
  )
endif()
add_backend_plugin(
  NAME realm_ucp_bootstrap_p2p
  SOURCES
    ${UCX_BOOTSTRAP_ROOT_DIR}/bootstrap_p2p.cc
    ${UCX_BOOTSTRAP_ROOT_DIR}/p2p_comm.cc
    ${UCX_BOOTSTRAP_ROOT_DIR}/worker.cc
    ${UCX_BOOTSTRAP_ROOT_DIR}/client.cc
    ${UCX_BOOTSTRAP_ROOT_DIR}/server.cc
    ${UCX_BOOTSTRAP_ROOT_DIR}/logger.cc
)
#endregion

if(UCX_BOOTSTRAP_INSTALL)
  install(TARGETS ${UCX_BACKENDS})
endif()

set(UCX_BACKENDS
    ${UCX_BACKENDS}
    PARENT_SCOPE
)
