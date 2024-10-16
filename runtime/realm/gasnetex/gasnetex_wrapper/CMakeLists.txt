#------------------------------------------------------------------------------#
# Copyright 2024 Stanford, NVIDIA Corp..
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

project(GasNetExWrapper
        LANGUAGES C CXX
        VERSION 0.0.1
        DESCRIPTION "GasNetEx loader for Realm")

# This is needed to retrieve the FindGASNet and FetchAndBuildGASNet cmake
# scripts
set(LEGION_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." CACHE STRING "Location of legion source directory")
option(GEX_BUILD_SHARED "Build shared lib" ON)
option(GEX_DEBUG "Enable extra debugging output" OFF)
option(GEX_NO_INSTALL "Disable installing of gex_wrapper library" OFF)

include(GenerateExportHeader)

list(APPEND CMAKE_MODULE_PATH "${LEGION_SOURCE_DIR}/cmake")

# Get GASNet dependency
find_package(GASNet)
if (NOT GASNet_FOUND)
  set(GASNet_BUILD_SHARED TRUE)
  include(${LEGION_SOURCE_DIR}/cmake/FetchAndBuildGASNet.cmake)
  find_package(GASNet REQUIRED)
endif ()

set(REALM_GASNETEX_WRAPPER_SRCS
  gasnetex_handlers.cc
  gasnetex_wrapper_internal.h
  gasnetex_wrapper.h
  gasnetex_wrapper.cc
)

add_library(realm_gex_wrapper_objs OBJECT ${REALM_GASNETEX_WRAPPER_SRCS})

if (${GEX_WRAPPER_BUILD_SHARED})
  add_library(realm_gex_wrapper SHARED)
else()
  add_library(realm_gex_wrapper STATIC)
  target_compile_definitions(realm_gex_wrapper_objs PUBLIC GEX_STATIC_DEFINE)
endif()
target_link_libraries(realm_gex_wrapper PRIVATE realm_gex_wrapper_objs)
target_compile_definitions(realm_gex_wrapper PRIVATE <IF:$<BOOL:GEX_DEBUG>,DEBUG_REALM_GEX,>)

generate_export_header(realm_gex_wrapper_objs BASE_NAME GEX EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/exports/gex_export.h)
target_include_directories(realm_gex_wrapper_objs
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
      PUBLIC
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/exports>)
target_link_libraries(realm_gex_wrapper_objs PUBLIC GASNet::GASNet)
set_target_properties(realm_gex_wrapper_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(realm_gex_wrapper_objs PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(realm_gex_wrapper_objs PROPERTIES VISIBILITY_INLINES_HIDDEN YES)

set_target_properties(realm_gex_wrapper PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_property(TARGET realm_gex_wrapper APPEND PROPERTY BUILD_RPATH "\$ORIGIN")
  set_property(TARGET realm_gex_wrapper APPEND PROPERTY INSTALL_RPATH "\$ORIGIN")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set_property(TARGET realm_gex_wrapper APPEND PROPERTY BUILD_RPATH "@loader_path")
  set_property(TARGET realm_gex_wrapper APPEND PROPERTY INSTALL_RPATH "@loader_path")
endif()

if (NOT GEX_NO_INSTALL)
  install(TARGETS realm_gex_wrapper)
endif()
