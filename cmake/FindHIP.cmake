#------------------------------------------------------------------------------#
# Copyright 2024 Kitware, Inc.
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

# find hip package
if(NOT DEFINED HIP_PATH)
  if(NOT DEFINED ENV{HIP_PATH})
      set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to where HIP has been installed")
  else()
      set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to where HIP has been installed")
  endif()
endif()
include(${HIP_PATH}/cmake/FindHIP.cmake)

if(NOT HIP_INCLUDE_DIRS)
  list(APPEND HIP_INCLUDE_DIRS
    ${HIP_THRUST_ROOT_DIR} ${HIP_ROOT_DIR}/include ${HIP_ROOT_DIR}/../include
  )
  set(HIP_INCLUDE_DIRS "${HIP_INCLUDE_DIRS}" CACHE STRING "List of HIP include paths")
endif()

###############################################################################
# (Internal) helper for manually added hip source files with specific targets
###############################################################################
macro(hip_compile_base hip_target format generated_files)
  set(_hip_target "${hip_target}")
  # Separate the sources from the options
  HIP_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _hipcc_options _clang_options _nvcc_options ${ARGN})
  # message( STATUS "_sources : ${_sources}" )
  HIP_PREPARE_TARGET_COMMANDS(${_hip_target} ${format} _generated_files _source_files ${_sources} HIPCC_OPTIONS ${_hipcc_options} CLANG_OPTIONS ${_clang_options} NVCC_OPTIONS ${_nvcc_options})
  set( ${generated_files} ${_generated_files})  
  # message( STATUS "generated_files : ${_generated_files}" )
  # message( STATUS "hip_target : ${_hip_target}" )
  add_library(${_hip_target} ${_cmake_options} ${_generated_files} "")
  set_target_properties(${_hip_target} PROPERTIES LINKER_LANGUAGE ${HIP_C_OR_CXX})
endmacro()

###############################################################################
# HIP_COMPILE
###############################################################################
macro(HIP_COMPILE generated_files)
  hip_compile_base(hip_compile OBJ ${generated_files} ${ARGN})
endmacro()
