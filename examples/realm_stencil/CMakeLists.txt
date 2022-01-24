#------------------------------------------------------------------------------#
# Copyright 2022 Stanford University
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
project(LegionExample_realm_stencil)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

add_executable(realm_stencil realm_stencil.cc cpu_kernels.cc)
target_compile_definitions(realm_stencil PRIVATE DTYPE=double)
target_compile_definitions(realm_stencil PRIVATE RESTRICT=__restrict__)
target_compile_definitions(realm_stencil PRIVATE RADIUS=2)
target_link_libraries(realm_stencil Legion::Legion)
if(Legion_ENABLE_TESTING)
  add_test(NAME realm_stencil COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:realm_stencil> ${Legion_TEST_ARGS} -ll:cpu 4)
endif()
