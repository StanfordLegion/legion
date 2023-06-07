#------------------------------------------------------------------------------#
# Copyright 2023 Kitware, Inc.
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
project(LegionExample_agency_interop)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

set(CPU_SOURCES agency_interop.cc)
#set(GPU_SOURCES thrust_shim.h thrust_shim.cu)
add_executable(agency_interop ${CPU_SOURCES} ${GPU_SOURCES})
target_link_libraries(agency_interop Legion::Legion)

set_source_files_properties(${CPU_SOURCES} PROPERTIES LANGUAGE CUDA)

set_target_cuda_architectures(TARGET agency_interop ARCHS ${Legion_CUDA_ARCH})

# starting with nvcc 10.2, we can error out on all warnings
if(Legion_BUILD_WARN_AS_ERROR AND (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 10.2))
  target_compile_options(agency_interop PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Werror=all-warnings>)
endif()

target_compile_options(agency_interop PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Wno-deprecated-gpu-targets>)
