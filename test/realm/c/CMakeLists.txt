#------------------------------------------------------------------------------#
# Copyright 2025 Stanford University
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
project(LegionTest_realm_c)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

list(APPEND REALM_C_TESTS
  test_runtime
  test_processor_query
  test_memory_query
  test_event
  )

foreach(test IN LISTS REALM_C_TESTS)
  if(CUDASRC_${test})
    add_executable(${test} ${test}.cc ${CUDASRC_${test}})
    set_target_cuda_standard(${test} STANDARD ${Legion_CUDA_STANDARD})
    set_target_cuda_architectures(${test} ARCHITECTURES ${Legion_CUDA_ARCH})
    set_target_cuda_warnings_and_errors(${test} WARN_AS_ERROR ${Legion_BUILD_WARN_AS_ERROR})
  elseif(HIPSRC_${test})
    hip_add_executable(${test} ${test}.cc ${HIPSRC_${test}})
  else()
    add_executable(${test} ${test}.cc)
  endif()
  target_link_libraries(${test} Legion::Realm $<TARGET_NAME_IF_EXISTS:CUDA::cuda_driver>)
  target_compile_options(${test} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
  if(Legion_USE_HIP)
    target_include_directories(${test} PRIVATE ${HIP_INCLUDE_DIRS})
    if(Legion_HIP_TARGET STREQUAL "CUDA")
      target_compile_definitions(${test} PRIVATE __HIP_PLATFORM_NVIDIA__)
      target_include_directories(${test} PRIVATE ${HIP_ROOT_DIR}/include)
    elseif (Legion_HIP_TARGET STREQUAL "ROCM")
      target_compile_definitions(${test} PRIVATE __HIP_PLATFORM_AMD__)
    endif()
  endif()
endforeach()

if(Legion_USE_LIBDL)
  target_link_libraries(taskreg ${CMAKE_DL_LIBS})
endif()

if(REALM_LIMIT_SYMBOL_VISIBILITY)
  set_target_properties(machine_config PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(machine_config PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
endif()

if(Legion_ENABLE_TESTING)
  foreach(test IN LISTS REALM_C_TESTS)
    add_test(NAME ${test} COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:${test}> ${Legion_TEST_ARGS} ${TESTARGS_${test}})
  endforeach()

  if(Legion_NETWORKS)
    # For verifying the -ll:networks arguments, try each network we've compiled with
    string(REPLACE "," ";" NETWORK_LIST "${Legion_NETWORKS}")
    foreach(ITEM IN LISTS NETWORK_LIST)
      set(NETWORK_ARGS "-ll:networks ${ITEM}")
      message(${NETWORK_ARGS})
      add_test(NAME version_check_network_${ITEM} COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:version_check> ${NETWORK_ARGS} ${Legion_TEST_ARGS} ${TESTARGS_version_check})
    endforeach()
  endif()
endif()
