#------------------------------------------------------------------------------#
# Copyright 2024 NVIDIA Corporation
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
project(RealmExample_reservation)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

add_executable(realm_reservation reservation.cc)
target_link_libraries(realm_reservation Legion::Realm)
target_compile_options(realm_reservation PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)

if(Legion_ENABLE_TESTING)
  add_test(NAME realm_reservation COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:realm_reservation> ${Legion_TEST_ARGS})
endif()
