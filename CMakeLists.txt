#------------------------------------------------------------------------------#
# Copyright 2015 Stanford University
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
#------------------------------------------------------------------------------#

project(legion)

set (legion_VERSION 2.0)

cmake_minimum_required(VERSION 3.0)

include (GenerateExportHeader)

#------------------------------------------------------------------------------#
# Add our cmake directory to search path
#------------------------------------------------------------------------------#

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

#------------------------------------------------------------------------------#
# Initialize project variables
#------------------------------------------------------------------------------#

set(EXTRA_LIBS)
set(EXTRA_LDFLAGS)

#------------------------------------------------------------------------------#
# GASNet configuration
#------------------------------------------------------------------------------#

option(Legion_USE_GASNet "Enable the distributed GASNet backend" OFF)
if(Legion_USE_GASNet)
  set(GASNet_THREADING "par" CACHE STRING "GASNet Threading model to use")
  set_property(CACHE GASNet_THREADING PROPERTY STRINGS seq par parsync)
  mark_as_advanced(GASNet_THREADING)
  if(NOT GASNet_THREADING STREQUAL "par")
    message(FATAL_ERROR "GASNet threading mode \"${GASNet_THREADING}\" is not currently supported by Legion")
  endif()
  set(GASNet_CONDUIT "" CACHE STRING "GASNet communication conduit to use")
  mark_as_advanced(GASNet_CONDUIT)
  find_package(GASNet REQUIRED)
endif()

#------------------------------------------------------------------------------#
# LIBRARIES configuration
#------------------------------------------------------------------------------#
option(USE_LIBDL "Use libdl" ON)
if (USE_LIBDL)
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl")
 list(APPEND EXTRA_LIBS dl)
endif (USE_LIBDL)

if (NOT APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt")
endif(NOT APPLE)
set (CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -ldl")
#------------------------------------------------------------------------------#
# THREADS configuration
#------------------------------------------------------------------------------#

option(ENABLE_THREADS "Enable support for the CUDA runtime" ON)

if(ENABLE_THREADS)
  find_package (Threads)

  if (NOT Threads_FOUND)
     message(FATAL_ERROR "Threads is required for this build configuration")
  endif (NOT Threads_FOUND)

  include_directories(${Threads_INCLUDE_DIRS})
  list(APPEND EXTRA_LIBS ${Threads_CUDA_LIBRARY})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

endif (ENABLE_THREADS)

#------------------------------------------------------------------------------#
# CUDA configuration
#------------------------------------------------------------------------------#

option(ENABLE_CUDA "Enable support for the CUDA runtime" OFF)

if(ENABLE_CUDA)
  find_package(CUDA)

  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA is required for this build configuration")
  endif(NOT CUDA_FOUND)

  include_directories(${CUDA_INCLUDE_DIRS})

  CUDA_INCLUDE_DIRECTORIES( "${CMAKE_SOURCE_DIR}/runtime"
                         "${CMAKE_SOURCE_DIR}/runtime/legion"
                         "${CMAKE_SOURCE_DIR}/runtime/mappers" 
                         "${CMAKE_SOURCE_DIR}/runtime/realm")

  list(APPEND EXTRA_LIBS ${CUDA_CUDA_LIBRARY})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDA")

  if(CUDA_ARCH STREQUAL "fermi")
    set(CMAKE_NVCC_FLAGS "${CMAKE_NVCC_FLAGS} -arch=compute_30 -code=sm_30")
  endif(CUDA_ARCH STREQUAL "fermi")

endif(ENABLE_CUDA)

#------------------------------------------------------------------------------#
# Legion options
#------------------------------------------------------------------------------#

option(ENABLE_SHARED_LOWLEVEL "Enable the shared memory low-level driver" OFF)
option(ENABLE_SHARED_MAPPER "Enable the shared memory mapper" OFF)
#------------------------------------------------------------------------------#
# Output directories
#------------------------------------------------------------------------------#

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

#------------------------------------------------------------------------------#
# install directories
#------------------------------------------------------------------------------#

set(LEGION_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} CACHE PATH "Installation directory for Legion")
set(INSTALL_LIB_DIR "${LEGION_INSTALL_DIR}/lib")
set(INSTALL_BIN_DIR "${LEGION_INSTALL_DIR}/bin")
set(INSTALL_INCLUDE_DIR "${LEGION_INSTALL_DIR}/include") 
set(INSTALL_CMAKE_DIR ${LEGION_INSTALL_DIR})

include_directories(
  "${PROJECT_SOURCE_DIR}"   
  "${PROJECT_BINARY_DIR}")

set (CMAKE_INSTALL_PREFIX ${LEGION_INSTALL_DIR})
#------------------------------------------------------------------------------#
# Runtime library target
#------------------------------------------------------------------------------#

add_subdirectory(runtime)

#------------------------------------------------------------------------------#
# Examples
#------------------------------------------------------------------------------#

add_subdirectory(examples)
option(BUILD_EXAMPLES "Build Legion examples" OFF)

#------------------------------------------------------------------------------#
# Export settings
#------------------------------------------------------------------------------#

# Add all targets to the build-tree export set

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/legionConfigVersion.cmake"
  VERSION ${legion_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(EXPORT realmTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/realmTargets.cmake"
)

export(EXPORT legionTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/legionTargets.cmake"
)

export(EXPORT legion_mappersTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/legion_mappersTargets.cmake"
)

configure_file(cmake/legionConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/legionConfig.cmake"
  @ONLY
)

install(EXPORT realmTargets
  DESTINATION
    "${INSTALL_CMAKE_DIR}/cmake"
)

install(EXPORT legionTargets
  DESTINATION
    "${INSTALL_CMAKE_DIR}/cmake"
)

install(EXPORT legion_mappersTargets
  DESTINATION
    "${INSTALL_CMAKE_DIR}/cmake"
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/legionConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/legionConfigVersion.cmake"
  DESTINATION
    ${INSTALL_CMAKE_DIR}
)

export (PACKAGE legion)

mark_as_advanced(FORCE CMAKE_INSTALL_PREFIX)

#------------------------------------------------------------------------------#
# vim: set tabstop=2 shiftwidth=2 expandtab :
#------------------------------------------------------------------------------#
