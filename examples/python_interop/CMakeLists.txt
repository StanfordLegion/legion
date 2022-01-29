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
project(LegionExample_python_interop)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

add_executable(python_interop python_interop.cc)
target_compile_definitions(python_interop PRIVATE PYTHON_MODULES_PATH="${CMAKE_INSTALL_FULL_DATADIR}/Legion/python:${CMAKE_INSTALL_FULL_DATADIR}/Legion/python/tests")
target_link_libraries(python_interop Legion::Legion)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python_interop.py
  DESTINATION ${CMAKE_INSTALL_DATADIR}/Legion/python/tests
)

if(Legion_ENABLE_TESTING)
  add_test(NAME python_interop COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:python_interop> ${Legion_TEST_ARGS} -ll:py 1)
endif()
