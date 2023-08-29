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
project(LegionTest_deferred_buffer_test)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

add_executable(deferred_buffer_test deferred_buffer_test.cc)
target_link_libraries(deferred_buffer_test Legion::Legion)
target_compile_options(deferred_buffer_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
if(Legion_ENABLE_TESTING)
  add_test(NAME deferred_buffer_test COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:deferred_buffer_test> ${Legion_TEST_ARGS})
endif()
