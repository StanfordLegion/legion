#------------------------------------------------------------------------------#
# Copyright 2023 Stanford University
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
project(LegionTest_realm)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

list(APPEND REALM_TESTS
  version_check
  serializing
  test_profiling
  ctxswitch
  barrier_reduce
  taskreg
  idcheck
  inst_reuse
  transpose
  proc_group
  deppart
  scatter
  compqueue
  event_subscribe
  deferred_allocs
  test_nodeset
  subgraphs
  large_tls
  memspeed
  memmodel
  coverings
  alltoall
  simple_reduce
  realm_reinit
  sparse_construct
  extres_alias
  reservations
  multiaffine
  machine_config
  )

if(Legion_USE_CUDA)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)

  # some tests have CUDA source files too
  set(CUDASRC_memspeed memspeed_gpu.cu)
  set(CUDASRC_simple_reduce simple_reduce_gpu.cu)
  set(CUDASRC_multiaffine multiaffine_gpu.cu)

  list(APPEND REALM_TESTS cuda_arrays)
  set(CUDASRC_cuda_arrays cuda_arrays_gpu.cu)

  list(APPEND REALM_TESTS task_stream)
  set(CUDASRC_task_stream task_stream_gpu.cu)
endif()

if(Legion_USE_HIP)
  if(Legion_HIP_TARGET STREQUAL "CUDA")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)
  endif()

  # some tests have HIP source files too
  set(HIPSRC_memspeed memspeed_gpu.cu)
  set(HIPSRC_simple_reduce simple_reduce_gpu.cu)
  # FIXME: https://github.com/StanfordLegion/legion/issues/1476
  # set(HIPSRC_multiaffine multiaffine_gpu.cu)

  # FIXME: https://github.com/StanfordLegion/legion/issues/1308
  # list(APPEND REALM_TESTS cuda_arrays)
  # set(HIPSRC_cuda_arrays cuda_arrays_gpu.cu)

  list(APPEND REALM_TESTS task_stream)
  set(HIPSRC_task_stream task_stream_gpu.cu)
endif()

foreach(test IN LISTS REALM_TESTS)
  if(CUDASRC_${test})
    cuda_add_executable(${test} ${test}.cc ${CUDASRC_${test}})
  elseif(HIPSRC_${test})
    hip_add_executable(${test} ${test}.cc ${HIPSRC_${test}})
  else()
    add_executable(${test} ${test}.cc)
  endif()
  target_link_libraries(${test} Legion::Realm)

  if(Legion_USE_HIP)
    target_include_directories(${test} PRIVATE ${HIP_INCLUDE_DIRS})
    if(Legion_HIP_TARGET STREQUAL "CUDA")
      target_compile_definitions(${test} PRIVATE __HIP_PLATFORM_NVIDIA__)
    elseif (Legion_HIP_TARGET STREQUAL "ROCM")
      target_compile_definitions(${test} PRIVATE __HIP_PLATFORM_AMD__)
    endif()
  endif()
endforeach()

# scatter uses C++11 lambdas
#target_compile_features(scatter PUBLIC cxx_std_11)
set_target_properties(scatter PROPERTIES CXX_STANDARD 11
                                         CXX_STANDARD_REQUIRED YES
                                         CXX_EXTENSIONS NO)

# some tests need test-specific arguments
set(TESTARGS_ctxswitch         -ll:io 1 -t 30 -i 10000)
set(TESTARGS_proc_group        -ll:cpu 4)
set(TESTARGS_compqueue         -ll:cpu 4)
set(TESTARGS_event_subscribe   -ll:cpu 4)
set(TESTARGS_deferred_allocs   -ll:gsize 0 -all)
set(TESTARGS_scatter           -p1 2 -p2 2)
set(TESTARGS_alltoall          -ll:csize 1024)
set(TESTARGS_simple_reduce     -all)
set(TESTARGS_sparse_construct  -verbose)
set(TESTARGS_cuda_arrays       -ll:gpu 1)

if(Legion_ENABLE_TESTING)
  foreach(test IN LISTS REALM_TESTS)
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
