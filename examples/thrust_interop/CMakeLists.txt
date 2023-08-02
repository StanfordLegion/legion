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
project(LegionExample_thrust_interop)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()
if(NOT Legion_USE_CUDA)
  message(FATAL_ERROR
    "The thrust_interop example requires Legion to be built with CUDA")
endif()

set(CPU_SOURCES thrust_interop.cc)
set(GPU_SOURCES thrust_shim.h thrust_shim.cu)
add_executable(thrust_interop ${CPU_SOURCES} ${GPU_SOURCES})
set_target_cuda_architectures(TARGET thrust_interop ARCHS ${Legion_CUDA_ARCH})
# starting with nvcc 10.2, we can error out on all warnings
if(Legion_BUILD_WARN_AS_ERROR AND (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 10.2))
  target_compile_options(thrust_interop PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Werror=all-warnings>)
endif()
target_compile_options(thrust_interop PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Wno-deprecated-gpu-targets>)

target_link_libraries(thrust_interop Legion::Legion)
target_compile_options(thrust_interop PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
