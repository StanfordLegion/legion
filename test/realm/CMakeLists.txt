#------------------------------------------------------------------------------#
# Copyright 2024 Stanford University
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
  #deferred_allocs
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
  rsrv_acquire_poisoned
  inst_chain_redistrict
  )

if(Legion_USE_CUDA)
  
  # some tests have CUDA source files too
  set(CUDASRC_memspeed memspeed_gpu.cu)
  set(CUDASRC_simple_reduce simple_reduce_gpu.cu)
  set(CUDASRC_multiaffine multiaffine_gpu.cu)

  # FIXME: https://github.com/StanfordLegion/legion/issues/1635
  # list(APPEND REALM_TESTS cuda_arrays)
  # set(CUDASRC_cuda_arrays cuda_arrays_gpu.cu)
  list(APPEND REALM_TESTS cuda_memcpy_test)
  list(APPEND REALM_TESTS cuda_scatter_test)
  list(APPEND REALM_TESTS cuda_transpose_test)
  list(APPEND REALM_TESTS task_stream)
  set(CUDASRC_task_stream task_stream_gpu.cu)

  list(APPEND REALM_TESTS test_cuhook)
  set(CUDASRC_test_cuhook test_cuhook_gpu.cu)
endif()

if(Legion_USE_HIP)
  # some tests have HIP source files too
  set(HIPSRC_memspeed memspeed_gpu.cu)
  set(HIPSRC_simple_reduce simple_reduce_gpu.cu)
  set(HIPSRC_multiaffine multiaffine_gpu.cu)

  # FIXME: https://github.com/StanfordLegion/legion/issues/1308
  # list(APPEND REALM_TESTS cuda_arrays)
  # set(HIPSRC_cuda_arrays cuda_arrays_gpu.cu)

  list(APPEND REALM_TESTS task_stream)
  set(HIPSRC_task_stream task_stream_gpu.cu)
endif()

foreach(test IN LISTS REALM_TESTS)
  if(CUDASRC_${test})
    add_executable(${test} ${test}.cc ${CUDASRC_${test}})
    set_target_cuda_standard(${test} STANDARD ${Legion_CUDA_STANDARD})
    set_target_cuda_architectures(${test} ARCHITECTURES ${Legion_CUDA_ARCH})
    set_target_cuda_warnings_and_errors(${test} WARN_AS_ERROR ${Legion_BUILD_WARN_AS_ERROR})
    # Remove this once the Realm::Point class is updated
    target_compile_options(${CUDASRC_${test}} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                             -Xcudafe=--diag_suppress=1444>)
  elseif(HIPSRC_${test})
    hip_add_executable(${test} ${test}.cc ${HIPSRC_${test}})
  else()
    add_executable(${test} ${test}.cc)
  endif()
  target_link_libraries(${test} Legion::Realm)
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

# some tests need test-specific arguments
set(TESTARGS_ctxswitch         -ll:io 1 -t 30 -i 10000)
set(TESTARGS_proc_group        -ll:cpu 4)
set(TESTARGS_compqueue         -ll:cpu 4 -timeout 120)
set(TESTARGS_event_subscribe   -ll:cpu 4)
#set(TESTARGS_deferred_allocs   -ll:gsize 0 -all)
set(TESTARGS_scatter           -p1 2 -p2 2)
set(TESTARGS_alltoall          -ll:csize 1024)
set(TESTARGS_simple_reduce     -all)
set(TESTARGS_sparse_construct  -verbose)
# FIXME: https://github.com/StanfordLegion/legion/issues/1635
# set(TESTARGS_cuda_arrays       -ll:gpu 1)
set(TESTARGS_scatter_test       -ll:gpu 1)
set(TESTARGS_transpose_test       -ll:gpu 1 -unit_test 1)
set(TESTARGS_cuda_memcpy_test      -ll:gpu 1 -sparse 64 -gap 0 -chunks 32 -test-sparse -rects 8 -verify)
set(TESTARGS_machine_config_args    -test_args 1 -ll:cpu 4 -ll:util 2 -ll:io 1 -ll:csize 16 -ll:stacksize 4 -ll:pin_util 1 -ll:ext_sysmem 0 -ll:rsize 2 -ll:nsize 2 -ll:ncsize 1 -ll:ncpu 1 -numa:pin)
set(TESTARGS_cuda_transpose_test      -ll:gpu 1)

if(Legion_USE_CUDA)
  list(APPEND TESTARGS_machine_config_args -ll:gpu 1 -ll:fsize 1024 -ll:zsize 8 -ll:ib_fsize 16 -ll:ib_zsize 32 -ll:msize 64 -cuda:dynfb 1 -cuda:dynfb_max 128 -ll:streams 2 -ll:d2d_streams 2)
endif()
if(Legion_USE_HIP)
  list(APPEND TESTARGS_machine_config_args -ll:gpu 1 -ll:fsize 1024 -ll:zsize 8 -ll:ib_fsize 16 -ll:ib_zsize 32 -hip:dynfb 1 -hip:dynfb_max 128 -ll:streams 2 -ll:d2d_streams 2)
endif()
if(Legion_USE_OpenMP)
  list(APPEND TESTARGS_machine_config_args -ll:ocpu 1 -ll:othr 2 -ll:onuma 0 -ll:ostack 4)
endif()
if(Legion_USE_Python)
  list(APPEND TESTARGS_machine_config_args -ll:py 1 -ll:pystack 4)
endif()


if(Legion_ENABLE_TESTING)
  foreach(test IN LISTS REALM_TESTS)
    if (NOT test STREQUAL "machine_config")
      add_test(NAME ${test} COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:${test}> ${Legion_TEST_ARGS} ${TESTARGS_${test}})
    endif()
  endforeach()
  # test machine_config without any Legion_TEST_ARGS
  add_test(NAME machine_config COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:machine_config> ${TESTARGS_machine_config})
  # test machine_config with -test_args 1
  add_test(NAME machine_config_args COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:machine_config> ${TESTARGS_machine_config_args})

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
