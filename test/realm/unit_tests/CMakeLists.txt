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
project(LegionTest_realm_unittest)

if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

list(APPEND REALM_TESTS
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
  lowlevel_dma_test.cc
  circ_queue_test.cc
  gather_scatter_test.cc
 )

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(realm_tests ${REALM_TESTS})

list(APPEND REALM_UNIT_TESTS_LIST
  realm_tests
)

foreach(ITEM IN LISTS REALM_UNIT_TESTS_LIST)
  if(REALM_LIMIT_SYMBOL_VISIBILITY)
    set_target_properties(${ITEM} PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(${ITEM} PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
  endif()
  target_link_libraries(${ITEM} Legion::RealmTests GTest::gmock_main ${CMAKE_DL_LIBS})
  target_compile_options(${ITEM} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
  include(GoogleTest)
  gtest_discover_tests(${ITEM})
  if(Legion_ENABLE_TESTING)
    add_test(NAME ${ITEM} COMMAND $<TARGET_FILE:${ITEM}> ${Legion_TEST_ARGS} ${TESTARGS_${ITEM}})
  endif()
endforeach()
