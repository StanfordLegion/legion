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
project(LegionTest_realm)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

list(APPEND REALM_TESTS
  address_list_test
  )

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

foreach(test IN LISTS REALM_TESTS)
   add_executable(${test} ${test}.cc)
  target_link_libraries(${test} Legion::Realm GTest::gtest_main)
  target_compile_options(${test} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)


  include(GoogleTest)
  gtest_discover_tests(${test})
endforeach()

if(REALM_LIMIT_SYMBOL_VISIBILITY)
  set_target_properties(machine_config PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(machine_config PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
endif()

#add_executable(
#ddress_list_test
#address_list_test.cc
#)
#target_link_libraries(
#  address_list_test
#  GTest::gtest_main
#)

#include(GoogleTest)
#gtest_discover_tests(address_list_test)

if(Legion_ENABLE_TESTING)
  foreach(test IN LISTS REALM_TESTS)
    add_test(NAME ${test} COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:${test}> ${Legion_TEST_ARGS} ${TESTARGS_${test}})
  endforeach()
  # test machine_config with -test_args 1
  add_test(NAME machine_config_args COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:machine_config> ${Legion_TEST_ARGS} ${TESTARGS_machine_config_args})

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
