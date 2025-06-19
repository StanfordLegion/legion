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

#region Benchmark Properties
set(BENCHMARK_DEFAULT_ARGS)
set(BENCHMARK_ARGS
    ""
    CACHE STRING "Default arguments for benchmarks"
)

if(NOT BENCHMARK_PROCESSORS)
  if(MPIEXEC_EXECUTABLE)
    set(BENCHMARK_PROCESSORS 2)
  else()
    set(BENCHMARK_PROCESSORS 1)
  endif()
endif()

if(MPIEXEC_EXECUTABLE)
  list(PREPEND BENCHMARK_LAUNCHER "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG}
       ${BENCHMARK_PROCESSORS} ${MPIEXEC_PREFLAGS}
  )
endif()

if(REALM_USE_CUDA OR REALM_USE_HIP)
  set(BENCHMARK_DEFAULT_ARGS -ll:gpu 1)
  set(BENCHMARK_USE_GPU TRUE)
  set(TEST_GPU_LIBS $<TARGET_NAME_IF_EXISTS:CUDA::cudart> $<TARGET_NAME_IF_EXISTS:hip::host>)
endif()

if(REALM_USE_KOKKOS AND REALM_USE_OPENMP)
  list(APPEND BENCHMARK_DEFAULT_ARGS -ll:ocpu 1 -ll:onuma 0)
endif()

if(NOT BENCHMARK_ARGS)
  set(BENCHMARK_ARGS ${BENCHMARK_DEFAULT_ARGS})
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
set(REALM_BENCH_DIR "../../test/performance/realm")
cmake_path(ABSOLUTE_PATH REALM_BENCH_DIR BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE)

#region Benchmarks
set(_bench_list)

# Given target and list of sources, compiles the executable and adds it as a test, with the default benchmark arguments
# TODO(cperry): make a cmake script that will allow to override test arguments at runtime
macro(add_benchmark target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} Realm::Realm)
  add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}> ${BENCHMARK_ARGS} ${${target}_ARGS})
  # Benchmarks need to be run in isolation to get accurate results
  set_property(TEST ${target} PROPERTY RUN_SERIAL TRUE)
  if(REALM_USE_UCX)
    add_dependencies(${target} realm_ucp_bootstrap_mpi)
  endif()
  list(APPEND _bench_list ${target})
endmacro()

add_benchmark(memcpy "${REALM_BENCH_DIR}/memcpy/memcpy.cc")
add_benchmark(event_latency "${REALM_BENCH_DIR}/event_latency/event_latency.cc")
add_benchmark(event_throughput "${REALM_BENCH_DIR}/event_throughput/event_throughput.cc")
add_benchmark(event_ubench "${REALM_BENCH_DIR}/event_ubench/event_ubench.cc")
add_benchmark(lock_chains "${REALM_BENCH_DIR}/lock_chains/lock_chains.cc")
add_benchmark(lock_contention "${REALM_BENCH_DIR}/lock_contention/lock_contention.cc")
set(reducetest_ARGS -batches 1) # Run short version
add_benchmark(reducetest "${REALM_BENCH_DIR}/reducetest/reducetest.cc")

add_benchmark(
  task_throughput "${REALM_BENCH_DIR}/task_throughput/task_throughput.cc"
  "$<$<BOOL:BENCH_USE_GPU>:${REALM_BENCH_DIR}/task_throughput/task_throughput_gpu.cu>"
)

add_benchmark(
  task_ubench "${REALM_BENCH_DIR}/task_throughput/task_throughput.cc"
  "$<$<BOOL:BENCH_USE_GPU>:${REALM_BENCH_DIR}/task_throughput/task_throughput_gpu.cu>"
)

set_tests_properties(
  ${_bench_list}
  PROPERTIES TEST_LAUNCHER
             "${BENCHMARK_LAUNCHER}"
             PROCESSORS
             ${BENCHMARK_PROCESSORS}
             ENVIRONMENT
             "${TEST_ENV}"
             LABELS
             "benchmark"
)

if(MSVC)
  set_tests_properties(
    ${_bench_list} PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_append:${TEST_LIB_PATH}"
  )
else()
  set_tests_properties(
    ${_bench_list} PROPERTIES ENVIRONMENT_MODIFICATION
                              "LD_LIBRARY_PATH=path_list_append:${TEST_LIB_PATH}"
  )
endif()

#endregion

add_custom_target(
  bench
  DEPENDS ${_bench_list}
  COMMAND ${CMAKE_CTEST_COMMAND} -j --output-on-failure
)
