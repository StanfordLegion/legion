#------------------------------------------------------------------------------#
# Copyright 2024 NVIDIA Corp., Stanford University
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
project(RealmExample_cuda_interop)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

if (NOT REALM_USE_CUDA AND NOT REALM_USE_HIP)
  message(FATAL_ERROR "CUDA or HIP is not enabled in REALM")
endif()

if (REALM_USE_CUDA)
  add_executable(realm_cuda_interop cuda_interop.cc gpu_kernel.cu)
  set_target_cuda_standard(realm_cuda_interop STANDARD ${Legion_CUDA_STANDARD})
  set_target_cuda_architectures(realm_cuda_interop ARCHITECTURES ${Legion_CUDA_ARCH})
  set_target_cuda_warnings_and_errors(realm_cuda_interop WARN_AS_ERROR ${Legion_BUILD_WARN_AS_ERROR})
elseif(Legion_USE_HIP)
  if(Legion_HIP_TARGET STREQUAL "CUDA")
    add_executable(realm_cuda_interop cuda_interop.cc gpu_kernel.cu)
    target_compile_definitions(realm_cuda_interop PRIVATE __HIP_PLATFORM_NVIDIA__)
  elseif(Legion_HIP_TARGET STREQUAL "ROCM")
    hip_add_executable(realm_cuda_interop cuda_interop.cc gpu_kernel.cu)
    target_compile_definitions(realm_cuda_interop PRIVATE __HIP_PLATFORM_AMD__)
  endif()
endif()

target_link_libraries(realm_cuda_interop Legion::Realm)
target_compile_options(realm_cuda_interop PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)

if(Legion_ENABLE_TESTING)
  add_test(NAME realm_cuda_interop COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:realm_cuda_interop> ${Legion_TEST_ARGS} -ll:gpu 1)
endif()
