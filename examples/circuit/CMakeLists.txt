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

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(LegionExample_circuit)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

set(CPU_SOURCES
  circuit.h         circuit.cc
                    circuit_cpu.cc
                    circuit_init.cc
  circuit_mapper.h  circuit_mapper.cc
)
if(Legion_USE_CUDA)
  set(GPU_SOURCES circuit_gpu.cu)
  add_executable(circuit ${CPU_SOURCES} ${GPU_SOURCES})
  set_target_cuda_standard(circuit STANDARD ${Legion_CUDA_STANDARD})
  set_target_cuda_architectures(circuit ARCHITECTURES ${Legion_CUDA_ARCH})
  set_target_cuda_warnings_and_errors(circuit WARN_AS_ERROR ${Legion_BUILD_WARN_AS_ERROR})
elseif(Legion_USE_HIP)
  set(GPU_SOURCES circuit_gpu.cu)
  if(Legion_HIP_TARGET STREQUAL "CUDA")
    add_executable(circuit ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(circuit PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories(circuit PRIVATE ${HIP_ROOT_DIR}/include)
  elseif (Legion_HIP_TARGET STREQUAL "ROCM")
    hip_add_executable(circuit ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(circuit PRIVATE __HIP_PLATFORM_AMD__)
  endif()
else()
  add_executable(circuit ${CPU_SOURCES})
endif()

target_compile_options(circuit PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)

target_link_libraries(circuit Legion::Legion)
if(Legion_ENABLE_TESTING)
  add_test(NAME circuit COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:circuit> ${Legion_TEST_ARGS})
endif()
