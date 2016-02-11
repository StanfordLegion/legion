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

#SET(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}../install")

project(legion)

cmake_minimum_required(VERSION 2.8.12)

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

# set the default GASNet model to par, others are not supported
set(GASNET_MODEL par CACHE STRING "Default GASNet model" FORCE)

include(FindGASNet)

option(ENABLE_GASNET "Enable GASNet support" OFF)

if(ENABLE_GASNET)

  if(GASNET_MODEL)
    if(NOT ${GASNET_MODEL} STREQUAL "par")
      message(FATAL_ERROR
        "${GASNET_MODEL} model is not supported, use 'par'(default)")
    endif(NOT ${GASNET_MODEL} STREQUAL "par")
  endif(GASNET_MODEL)

  find_package(GASNet)

  if(NOT GASNET_FOUND)
    message(FATAL_ERROR "GASNet is required for this build configuration")
  endif(NOT GASNET_FOUND)

  include_directories(${GASNET_INCLUDE_DIR})
  list(APPEND EXTRA_LIBS ${GASNET_LIBRARY_FOUND})

endif(ENABLE_GASNET)


#------------------------------------------------------------------------------#
# LIBRARIES configuration
#------------------------------------------------------------------------------#
option(USE_LIBDL "Use libdl" ON)
if (USE_LIBDL)
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl")
endif (USE_LIBDL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt") 
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
                         "${CMAKE_SOURCE_DIR}/runtime/mappers" )

  list(APPEND EXTRA_LIBS ${CUDA_CUDA_LIBRARY})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDA")

  if(CUDA_ARCH STREQUAL "fermi")
    set(CMAKE_NVCC_FLAGS "${CMAKE_NVCC_FLAGS} -arch=compute_30 -code=sm_30")
  endif(CUDA_ARCH STREQUAL "fermi")

endif(ENABLE_CUDA)


# FIXME: Debug
message(STATUS "Extra Libraries: ${EXTRA_LIBS}")

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
# Runtime library target
#------------------------------------------------------------------------------#

add_subdirectory(runtime)

#------------------------------------------------------------------------------#
# Examples
#------------------------------------------------------------------------------#

add_subdirectory(examples)
option(BUILD_EXAMPLES "Build Legion examples" OFF)

#------------------------------------------------------------------------------#
# vim: set tabstop=2 shiftwidth=2 expandtab :
#------------------------------------------------------------------------------#



