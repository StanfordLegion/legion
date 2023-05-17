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
project(RealmTest_perf_realm_task_ubench)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

set(CPU_SOURCES task_ubench.cc)
add_executable(task_ubench ${CPU_SOURCES})

target_link_libraries(task_ubench Legion::Realm)
if(Legion_ENABLE_TESTING)
  add_test(NAME task_ubench COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:task_ubench> ${Legion_TEST_ARGS})
endif()
