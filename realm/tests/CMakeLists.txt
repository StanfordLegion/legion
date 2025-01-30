# Copyright 2025 Stanford University, NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

#region Global Test Properties
set(TEST_RESOURCE_LOCK)
set(TEST_DEFAULT_ARGS)
set(TEST_ARGS
    ""
    CACHE STRING "Default arguments for tests"
)

if(REALM_USE_MPI
   OR REALM_USE_UCX
   OR REALM_USE_GASNETEX
)
  list(PREPEND TEST_LAUNCHER "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS})
  set(TEST_PROCESSORS 2)
elseif(NOT TEST_PROCESSORS)
  set(TEST_PROCESSORS 1)
endif()

if(REALM_USE_CUDA OR REALM_USE_HIP)
  set(TEST_RESOURCE_LOCK gpu)
  list(APPEND TEST_DEFAULT_ARGS -ll:gpu 1)
  set(TEST_USE_GPU TRUE)
endif()

if(NOT TEST_ARGS)
  set(TEST_ARGS ${TEST_DEFAULT_ARGS})
endif()

if(REALM_USE_PYTHON)
  list(APPEND TEST_ENV "REALM_PYTHON_LIB=$<TARGET_FILE:Python3::Python>")
endif()

if(REALM_USE_UCX)
  list(APPEND TEST_LIB_PATH "$<TARGET_FILE_DIR:realm_ucp_bootstrap_mpi>")
endif()
cmake_path(CONVERT "${TEST_LIB_PATH}" TO_NATIVE_PATH_LIST TEST_LIB_PATH NORMALIZE)
#endregion

# TODO(cperry): Move these tests to this directory
set(REALM_TEST_DIR "${CMAKE}../../test/realm")
cmake_path(ABSOLUTE_PATH REALM_TEST_DIR BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE)

#region Unit Tests

cpmfindpackage(
  NAME
  googletest
  GIT_REPOSITORY
  https://github.com/google/googletest.git
  GIT_TAG
  v1.15.2
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
  OPTIONS
  "INSTALL_GTEST OFF" "BUILD_SHARED_LIBS OFF"
)
if (googletest_ADDED)
  set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND googletest)
  get_property(_NOT_FOUND GLOBAL PROPERTY PACKAGES_NOT_FOUND)
  list(REMOVE_ITEM _NOT_FOUND googletest)
  set_property(GLOBAL PROPERTY PACKAGES_NOT_FOUND "${_NOT_FOUND}")
endif()

list(
  APPEND
  REALM_UNIT_TESTS
  memcpy_channel_test.cc
  comp_queue_test.cc
  event_test.cc
  path_cache_test.cc
  transfer_utils_test.cc
  address_list_test.cc
  cmdline_parser_test.cc
  sequence_assembler_test.cc
  dynamic_table_test.cc
  range_allocator_test.cc
  pri_queue_test.cc
  intrusive_list_test.cc
  repl_heap_test.cc
  nodeset_test.cc
  transfer_iterator_test.cc
  lowlevel_dma_test.cc
  circ_queue_test.cc
  gather_scatter_test.cc
)
list(TRANSFORM REALM_UNIT_TESTS PREPEND "${REALM_TEST_DIR}/unit_tests/")

add_executable(realm_unit_tests ${REALM_UNIT_TESTS})
target_link_libraries(realm_unit_tests PRIVATE realm_obj GTest::gmock_main)

include(GoogleTest)
gtest_discover_tests(realm_unit_tests NO_PRETTY_TYPES NO_PRETTY_VALUES PROPERTIES LABELS "unit")

if(REALM_ENABLE_COVERAGE)
  if(OPENCPPCOV_PATH)
    cmake_path(CONVERT ${PROJECT_SOURCE_DIR} TO_NATIVE_PATH_LIST NATIVE_SRC_DIR)
    cmake_path(CONVERT ${PROJECT_BINARY_DIR} TO_NATIVE_PATH_LIST NATIVE_BIN_DIR)
    add_custom_target(
      coverage
      COMMAND
        "${OPENCPPCOV_PATH}" "--export_type=cobertura:${NATIVE_BIN_DIR}\\coverage.xml"
        --cover_children --sources "${NATIVE_SRC_DIR}" --modules "${NATIVE_BIN_DIR}"
        --excluded_sources "${NATIVE_BIN_DIR}\\_deps" -- "${CMAKE_CTEST_COMMAND}" -T Test -T
        Coverage -L unit
      BYPRODUCTS coverage.xml
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      DEPENDS realm_unit_tests VERBATIUM
      COMMENT "Running opencppcoverage to produce Cobertura coverage report"
    )
    # Show info where to find the report
    add_custom_command(
      TARGET coverage
      POST_BUILD
      COMMAND ;
      COMMENT "Cobertura coverage report saved in ${PROJECT_BINARY_DIR}/coverage.xml"
    )
  else()
    set(CODE_COVERAGE_VERBOSE TRUE)
    set(GCOVR_ADDITIONAL_ARGS "-j")
    append_coverage_compiler_flags_to_target(realm_unit_tests)
    # TODO(cperry): fix base directory when sources are moved over
    setup_target_for_coverage_gcovr_xml(
      NAME
      coverage
      EXECUTABLE
      "${CMAKE_CTEST_COMMAND}"
      -T
      Test
      -T
      Coverage
      -L
      unit
      DEPENDENCIES
      realm_unit_tests
      BASE_DIRECTORY
      "${REALM_TEST_DIR}/../.."
      EXCLUDE
      "${PROJECT_BINARY_DIR}/*"
      "${REALM_TEST_DIR}/*"
    )
  endif()
endif()

#endregion

#region Integration Tests
set(_integ_list)

# Given target and list of sources, compiles the executable and adds it as a test, with the default benchmark arguments
# TODO(cperry): make a cmake script that will allow to override test arguments at runtime
macro(add_integration_test target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} Realm::Realm)
  add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}> ${TEST_ARGS} ${${target}_ARGS})
  set_property(
    TEST ${target} PROPERTY RESOURCE_LOCK $<REMOVE_DUPLICATES:${TEST_RESOURCE_LOCK}
                            ${${target}_RESOURCE_LOCK}>
  )
  if(REALM_USE_UCX)
    add_dependencies(${target} realm_ucp_bootstrap_mpi)
  endif()
  list(APPEND _integ_list ${target})
endmacro()

add_integration_test(version_check "${REALM_TEST_DIR}/version_check.cc")
add_integration_test(serializing "${REALM_TEST_DIR}/serializing.cc")
set(ctxswitch_ARGS -ll:io 1 -t 30 -i 10000)
add_integration_test(ctxswitch "${REALM_TEST_DIR}/ctxswitch.cc")
add_integration_test(barrier_reduce "${REALM_TEST_DIR}/barrier_reduce.cc")
add_integration_test(taskreg "${REALM_TEST_DIR}/taskreg.cc")
add_integration_test(idcheck "${REALM_TEST_DIR}/idcheck.cc")
add_integration_test(inst_reuse "${REALM_TEST_DIR}/inst_reuse.cc")
add_integration_test(transpose "${REALM_TEST_DIR}/transpose.cc")
set(proc_group_ARGS -ll:cpu 4)
add_integration_test(proc_group "${REALM_TEST_DIR}/proc_group.cc")
add_integration_test(deppart "${REALM_TEST_DIR}/deppart.cc")
set(scatter_ARGS -p1 2 -p2 2)
add_integration_test(scatter "${REALM_TEST_DIR}/scatter.cc")
set(proc_group_ARGS -ll:cpu 4)
add_integration_test(event_subscribe "${REALM_TEST_DIR}/event_subscribe.cc")
set(compqueue_ARGS -ll:cpu 4 -timeout 120)
add_integration_test(compqueue "${REALM_TEST_DIR}/compqueue.cc")
add_integration_test(test_nodeset "${REALM_TEST_DIR}/test_nodeset.cc")
add_integration_test(subgraphs "${REALM_TEST_DIR}/subgraphs.cc")
add_integration_test(large_tls "${REALM_TEST_DIR}/large_tls.cc")
add_integration_test(memmodel "${REALM_TEST_DIR}/memmodel.cc")
add_integration_test(coverings "${REALM_TEST_DIR}/coverings.cc")
set(alltoall_ARGS -ll:csize 1024)
add_integration_test(alltoall "${REALM_TEST_DIR}/alltoall.cc")
add_integration_test(realm_reinit "${REALM_TEST_DIR}/realm_reinit.cc")
set(sparse_construct_ARGS -verbose)
add_integration_test(sparse_construct "${REALM_TEST_DIR}/sparse_construct.cc")
add_integration_test(extres_alias "${REALM_TEST_DIR}/extres_alias.cc")
add_integration_test(reservations "${REALM_TEST_DIR}/reservations.cc")
set(machine_config_ARGS
    -test_args
    1
    -ll:cpu
    4
    -ll:util
    2
    -ll:io
    1
    -ll:csize
    16
    -ll:stacksize
    4
    -ll:pin_util
    1
    -ll:ext_sysmem
    0
    -ll:rsize
    2
    -ll:nsize
    2
    -ll:ncsize
    1
    -ll:ncpu
    1
    -numa:pin
)
if(TEST_USE_GPU)
  list(
    APPEND
    machine_config_ARGS
    -ll:gpu
    1
    -ll:fsize
    1024
    -ll:zsize
    8
    -ll:ib_fsize
    16
    -ll:ib_zsize
    32
    -ll:msize
    64
    -ll:streams
    2
    -ll:d2d_streams
    2
  )
  if(REALM_USE_CUDA)
    list(APPEND machine_config_ARGS -cuda:dynfb 1 -cuda:dynfb_max 128)
  endif()
  if(REALM_USE_HIP)
    list(APPEND machine_config_ARGS -hip:dynfb 1 -hip:dynfb_max 128)
  endif()
endif()
if(REALM_USE_OPENMP)
  list(
    APPEND
    machine_config_ARGS
    -ll:ocpu
    1
    -ll:othr
    2
    -ll:onuma
    0
    -ll:ostack
    4
  )
endif()
if(REALM_USE_PYTHON)
  list(APPEND machine_config_ARGS -ll:py 1 -ll:pystack 4)
endif()
add_integration_test(machine_config "${REALM_TEST_DIR}/machine_config.cc")
add_integration_test(rsrv_acquire_poisoned "${REALM_TEST_DIR}/rsrv_acquire_poisoned.cc")
add_integration_test(refcount_image_test "${REALM_TEST_DIR}/refcount_image_test.cc")
set(inst_chain_redistrict_ARGS -i 2)
add_integration_test(inst_chain_redistrict "${REALM_TEST_DIR}/inst_chain_redistrict.cc")
add_integration_test(refcount_preimage_test "${REALM_TEST_DIR}/refcount_preimage_test.cc")
add_integration_test(
  test_profiling "${REALM_TEST_DIR}/test_profiling.cc"
  "$<$<BOOL:TEST_USE_GPU>:${REALM_TEST_DIR}/test_profiling_gpu.cu>"
)
add_integration_test(
  memspeed "${REALM_TEST_DIR}/memspeed.cc"
  "$<$<BOOL:TEST_USE_GPU>:${REALM_TEST_DIR}/memspeed_gpu.cu>"
)
set(simple_reduce_ARGS -all)
add_integration_test(
  simple_reduce "${REALM_TEST_DIR}/simple_reduce.cc"
  "$<$<BOOL:TEST_USE_GPU>:${REALM_TEST_DIR}/simple_reduce_gpu.cu>"
)
add_integration_test(
  multiaffine "${REALM_TEST_DIR}/multiaffine.cc"
  "$<$<BOOL:TEST_USE_GPU>:${REALM_TEST_DIR}/multiaffine_gpu.cu>"
)
if(TEST_USE_GPU)
  if(REALM_USE_CUDA)
    set(cuda_memcpy_test_ARGS
        -ll:gpu
        1
        -sparse
        64
        -gap
        0
        -chunks
        32
        -test-sparse
        -rects
        8
        -verify
    )
    set(cuda_memcpy_test_RESOURCE_LOCK gpu)
    add_integration_test(cuda_memcpy_test "${REALM_TEST_DIR}/cuda_memcpy_test.cc")
    set(cuda_scatter_test_ARGS -ll:gpu 1)
    set(cuda_scatter_test_RESOURCE_LOCK gpu)
    add_integration_test(cuda_scatter_test "${REALM_TEST_DIR}/cuda_scatter_test.cc")
    set(test_cuhook_ARGS -ll:gpu 1)
    set(test_cuhook_RESOURCE_LOCK gpu)
    add_integration_test(
      test_cuhook "${REALM_TEST_DIR}/test_cuhook.cc" "${REALM_TEST_DIR}/test_cuhook_gpu.cu"
    )
    target_link_libraries(cuda_memcpy_test CUDA::cudart)
    target_link_libraries(cuda_scatter_test CUDA::cudart)
    target_link_libraries(test_cuhook CUDA::cudart)
  endif()
  set(transpose_test_gpu_ARGS -ll:gpu 1)
  set(transpose_test_gpu_RESOURCE_LOCK gpu)
  add_integration_test(transpose_test_gpu "${REALM_TEST_DIR}/transpose_test_gpu.cc")
  set(task_stream_ARGS -ll:gpu 1)
  set(task_stream_RESOURCE_LOCK gpu)
  add_integration_test(
    task_stream "${REALM_TEST_DIR}/task_stream.cc" "${REALM_TEST_DIR}/task_stream_gpu.cu"
  )
  target_link_libraries(
    task_stream $<TARGET_NAME_IF_EXISTS:CUDA::cuda_driver> $<TARGET_NAME_IF_EXISTS:Hip::Hip>
  )
endif()

target_link_libraries(
  machine_config $<TARGET_NAME_IF_EXISTS:CUDA::cuda_driver> $<TARGET_NAME_IF_EXISTS:Hip::Hip>
)
target_link_libraries(
  test_profiling $<TARGET_NAME_IF_EXISTS:CUDA::cuda_driver> $<TARGET_NAME_IF_EXISTS:Hip::Hip>
)

# These executables use global registration and require -rdynamic to be set in order to retrieve the symbol name and offset
set_target_properties(taskreg PROPERTIES ENABLE_EXPORTS TRUE)
set_target_properties(scatter PROPERTIES ENABLE_EXPORTS TRUE)
set_target_properties(cuda_scatter_test PROPERTIES ENABLE_EXPORTS TRUE)

set_tests_properties(
  ${_integ_list}
  PROPERTIES TEST_LAUNCHER
             ${TEST_LAUNCHER}
             PROCESSORS
             ${TEST_PROCESSORS}
             ENVIRONMENT
             "${TEST_ENV}"
             LABELS
             "integration"
)

if(MSVC)
  set_tests_properties(
    ${_integ_list} PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_append:${TEST_LIB_PATH}"
  )
else()
  set_tests_properties(
    ${_integ_list} PROPERTIES ENVIRONMENT_MODIFICATION
                              "LD_LIBRARY_PATH=path_list_append:${TEST_LIB_PATH}"
  )
endif()

#endregion

add_custom_target(
  check
  DEPENDS ${_integ_list} realm_unit_tests
  COMMAND ${CMAKE_CTEST_COMMAND} -j --output-on-failure
)
