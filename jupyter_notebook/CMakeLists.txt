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
project(LegionJupyter)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

# Allow passing extra args to `setup.py` on install like
# `--single-version-externally-managed --root=/`
# to work around setuptools changes.
if (NOT DEFINED Legion_PYTHON_EXTRA_INSTALL_ARGS)
  set(Legion_PYTHON_EXTRA_INSTALL_ARGS "--root / --prefix \"\${CMAKE_INSTALL_PREFIX}\"")
endif()
# Use setup.py script to install the Python bindings library
install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install --ignore-installed ${Legion_PYTHON_EXTRA_INSTALL_ARGS} . WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_ECHO STDOUT)")
