#------------------------------------------------------------------------------#
# Copyright 2018 Stanford University
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
project(LegionTest_rendering)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

add_executable(rendering rendering.cc)
target_link_libraries(rendering Legion::Legion)
if(Legion_ENABLE_TESTING)
  add_test(NAME rendering COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:rendering> -i 2 -n 64 -ll:cpu 4) 
endif()
