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

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(LegionExample_allreduce)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

add_executable(allreduce allreduce.cc)
target_link_libraries(allreduce Legion::Legion)
target_compile_options(allreduce PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
if(Legion_ENABLE_TESTING)
  add_test(NAME allreduce COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:allreduce> ${Legion_TEST_ARGS})
endif()

add_executable(intra_task_ordering intra_task_ordering.cc)
target_link_libraries(intra_task_ordering Legion::Legion)
target_compile_options(intra_task_ordering PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)

add_executable(disjoint_index_space_launch disjoint_index_space_launch.cc)
target_link_libraries(disjoint_index_space_launch Legion::Legion)
target_compile_options(disjoint_index_space_launch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)

add_executable(uninit_reproducer uninit_reproducer.cc)
target_link_libraries(uninit_reproducer Legion::Legion)
target_compile_options(uninit_reproducer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
