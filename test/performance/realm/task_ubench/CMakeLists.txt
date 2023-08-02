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
project(RealmTest_perf_realm_task_ubench)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

set(CPU_SOURCES task_ubench.cc)
if(Legion_USE_CUDA)
  set(GPU_SOURCES task_ubench_gpu.cu)
  add_executable(task_ubench ${CPU_SOURCES} ${GPU_SOURCES})
  # starting with nvcc 10.2, we can error out on all warnings
  if(Legion_BUILD_WARN_AS_ERROR AND (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 10.2))
    target_compile_options(task_ubench PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Werror=all-warnings>)
  endif()
  target_compile_options(task_ubench PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Wno-deprecated-gpu-targets>)
  set_target_cuda_standard(task_ubench STANDARD ${Legion_CUDA_STANDARD})
  set_target_cuda_architectures(task_ubench ARCHITECTURES ${Legion_CUDA_ARCH})
elseif(Legion_USE_HIP)
  set(GPU_SOURCES task_ubench_gpu.cu)
  if(Legion_HIP_TARGET STREQUAL "CUDA")
    add_executable(task_ubench ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(task_ubench PRIVATE __HIP_PLATFORM_NVIDIA__)
  elseif (Legion_HIP_TARGET STREQUAL "ROCM")
    hip_add_executable(task_ubench ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(task_ubench PRIVATE __HIP_PLATFORM_AMD__)
  endif()
  target_include_directories(task_ubench PRIVATE ${HIP_INCLUDE_DIRS})
else()
  add_executable(task_ubench ${CPU_SOURCES})
endif()

target_link_libraries(task_ubench Legion::Realm)
target_compile_options(task_ubench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
if(Legion_ENABLE_TESTING)
  add_test(NAME task_ubench COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:task_ubench> ${Legion_TEST_ARGS})
endif()
