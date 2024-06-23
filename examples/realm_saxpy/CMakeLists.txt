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

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(LegionExample_realm_saxpy)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

set(CPU_SOURCES realm_saxpy.h realm_saxpy_cpu.cc)
if(Legion_USE_CUDA)
  set(GPU_SOURCES realm_saxpy_gpu.cu)
  add_executable(realm_saxpy ${CPU_SOURCES} ${GPU_SOURCES})
  set_target_cuda_standard(realm_saxpy STANDARD ${Legion_CUDA_STANDARD})
  set_target_cuda_architectures(realm_saxpy ARCHITECTURES ${Legion_CUDA_ARCH})
  set_target_cuda_warnings_and_errors(realm_saxpy WARN_AS_ERROR ${Legion_BUILD_WARN_AS_ERROR})
  # Remove this once the Realm::Point class is updated
  target_compile_options(realm_saxpy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                           -Xcudafe=--diag_suppress=1444>)
elseif(Legion_USE_HIP)
  set(GPU_SOURCES realm_saxpy_gpu.cu)
  if(Legion_HIP_TARGET STREQUAL "CUDA")
    add_executable(realm_saxpy ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(realm_saxpy PRIVATE __HIP_PLATFORM_NVIDIA__)
  elseif (Legion_HIP_TARGET STREQUAL "ROCM")
    hip_add_executable(realm_saxpy ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(realm_saxpy PRIVATE __HIP_PLATFORM_AMD__)
  endif()
else()
  add_executable(realm_saxpy ${CPU_SOURCES})
endif()
target_link_libraries(realm_saxpy Legion::Legion)
target_compile_options(realm_saxpy PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
if(Legion_ENABLE_TESTING)
  add_test(NAME realm_saxpy COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:realm_saxpy> ${Legion_TEST_ARGS})
endif()
