#------------------------------------------------------------------------------#
# Copyright 2021 Stanford University
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

cmake_minimum_required(VERSION 3.1)
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
  coverings
  alltoall
  simple_reduce
  realm_reinit
  )

if(Legion_USE_CUDA)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)

  # some tests have CUDA source files too
  set(CUDASRC_memspeed memspeed_gpu.cu)
  set(CUDASRC_simple_reduce simple_reduce_gpu.cu)
endif()

foreach(test IN LISTS REALM_TESTS)
  if(CUDASRC_${test})
    cuda_add_executable(${test} ${test} ${CUDASRC_${test}})
  else()
    add_executable(${test} ${test}.cc)
  endif()
  target_link_libraries(${test} Legion::Realm)
endforeach()

# scatter uses C++11 lambdas
#target_compile_features(scatter PUBLIC cxx_std_11)
set_target_properties(scatter PROPERTIES CXX_STANDARD 11
                                         CXX_STANDARD_REQUIRED YES
                                         CXX_EXTENSIONS NO)

# some tests need test-specific arguments
set(TESTARGS_ctxswitch         -ll:io 1 -t 20 -i 10000)
set(TESTARGS_proc_group        -ll:cpu 4)
set(TESTARGS_compqueue         -ll:cpu 4)
set(TESTARGS_event_subscribe   -ll:cpu 4)
set(TESTARGS_deferred_allocs   -ll:gsize 0 -all)
set(TESTARGS_scatter           -p1 2 -p2 2)
set(TESTARGS_simple_reduce     -all)

if(Legion_ENABLE_TESTING)
  foreach(test IN LISTS REALM_TESTS)
    add_test(NAME ${test} COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:${test}> ${Legion_TEST_ARGS} ${TESTARGS_${test}})
  endforeach()
endif()
