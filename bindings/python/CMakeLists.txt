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
project(LegionBindings_python)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

add_executable(legion_python
  main.cc
)
set_target_properties(legion_python PROPERTIES OUTPUT_NAME "legion_python${INSTALL_SUFFIX}")
target_link_libraries(legion_python Legion::Legion)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_property(TARGET legion_python APPEND PROPERTY BUILD_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
  set_property(TARGET legion_python APPEND PROPERTY INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set_property(TARGET legion_python APPEND PROPERTY BUILD_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
  set_property(TARGET legion_python APPEND PROPERTY INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
endif ()
if(BINDINGS_DEFAULT_MODULE)
  target_compile_definitions(legion_python PRIVATE "BINDINGS_DEFAULT_MODULE=${BINDINGS_DEFAULT_MODULE}")
endif()
target_compile_options(legion_python PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)

add_library(legion_canonical_python SHARED canonical_python.cc)
target_link_libraries(legion_canonical_python Legion::Legion)
set_target_properties(legion_canonical_python PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(legion_canonical_python PROPERTIES OUTPUT_NAME "legion_canonical_python${INSTALL_SUFFIX}")
set_target_properties(legion_canonical_python PROPERTIES SOVERSION ${SOVERSION})
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_property(TARGET legion_canonical_python APPEND PROPERTY BUILD_RPATH "\$ORIGIN")
  set_property(TARGET legion_canonical_python APPEND PROPERTY INSTALL_RPATH "\$ORIGIN")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set_property(TARGET legion_canonical_python APPEND PROPERTY BUILD_RPATH "@loader_path")
  set_property(TARGET legion_canonical_python APPEND PROPERTY INSTALL_RPATH "@loader_path")
endif ()

target_compile_options(legion_canonical_python PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)

install(TARGETS legion_python
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(TARGETS legion_canonical_python
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Allow passing extra args to `setup.py` on install like
# `--single-version-externally-managed --root=/`
# to work around setuptools changes.
if (NOT DEFINED Legion_PYTHON_EXTRA_INSTALL_ARGS)
  set(Legion_PYTHON_EXTRA_INSTALL_ARGS "")
endif()
# Use setup.py script to install the Python bindings library
install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install --global-option=--cmake-build-dir \"--global-option=${Legion_BINARY_DIR}/runtime\" --global-option=--prefix \"--global-option=\${CMAKE_INSTALL_PREFIX}\" --ignore-installed ${Legion_PYTHON_EXTRA_INSTALL_ARGS} . WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_ECHO STDOUT)")

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/examples/hello.py
  DESTINATION ${CMAKE_INSTALL_DATADIR}/Legion/python/examples
)

if(Legion_ENABLE_TESTING)
  add_test(NAME legion_python COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:legion_python> hello ${Legion_TEST_ARGS} -ll:py 1 -ll:cpu 0)
  set_tests_properties(
    legion_python
    PROPERTIES
      ENVIRONMENT PYTHONPATH="${CMAKE_INSTALL_FULL_DATADIR}/Legion/python:${CMAKE_INSTALL_FULL_DATADIR}/Legion/python/examples")
endif()
