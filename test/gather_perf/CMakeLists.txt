#------------------------------------------------------------------------------#
# Copyright 2023 NVIDIA Corporation
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
project(LegionTest_gather_perf)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

add_executable(gather_perf gather_perf.cc)
target_link_libraries(gather_perf Legion::Legion)

if(Legion_ENABLE_TESTING)
  add_test(NAME gather_perf_0 COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:gather_perf> ${Legion_TEST_ARGS} -m 0)
  add_test(NAME gather_perf_1 COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:gather_perf> ${Legion_TEST_ARGS} -m 1)
  add_test(NAME gather_perf_2 COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:gather_perf> ${Legion_TEST_ARGS} -m 2)
  add_test(NAME gather_perf_3 COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:gather_perf> ${Legion_TEST_ARGS} -m 3)
  add_test(NAME gather_perf_4 COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:gather_perf> ${Legion_TEST_ARGS} -m 4)
  add_test(NAME gather_perf_5 COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:gather_perf> ${Legion_TEST_ARGS} -m 5)
  add_test(NAME gather_perf_6 COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:gather_perf> ${Legion_TEST_ARGS} -m 6)
  add_test(NAME gather_perf_7 COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:gather_perf> ${Legion_TEST_ARGS} -m 7)
endif()
