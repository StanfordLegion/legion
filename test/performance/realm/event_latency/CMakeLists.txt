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
project(LegionTest_perf_realm_event_latency)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

set(CPU_SOURCES event_latency.cc)
add_executable(event_latency ${CPU_SOURCES})

if(Legion_USE_HIP)
  target_include_directories(event_latency PRIVATE ${HIP_INCLUDE_DIRS})
  if(Legion_HIP_TARGET STREQUAL "CUDA")
    target_compile_definitions(event_latency PRIVATE __HIP_PLATFORM_NVIDIA__)
  elseif (Legion_HIP_TARGET STREQUAL "ROCM")
    target_compile_definitions(event_latency PRIVATE __HIP_PLATFORM_AMD__)
  endif()
endif()

target_link_libraries(event_latency Legion::Realm)
target_compile_options(event_latency PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
if(Legion_ENABLE_TESTING)
  add_test(NAME event_latency COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:event_latency> ${Legion_TEST_ARGS})
endif()
