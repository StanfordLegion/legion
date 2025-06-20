#------------------------------------------------------------------------------#
# Copyright 2024 Kitware, Inc.
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

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(LegionExample_attach_file)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

add_executable(attach_file attach_file.cc)
target_link_libraries(attach_file Legion::Legion)
target_compile_options(attach_file PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
target_include_directories(attach_file PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(attach_file ${HDF5_LIBRARIES})
if(Legion_ENABLE_TESTING)
  add_test(NAME attach_file COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:attach_file> ${Legion_TEST_ARGS})
  if(Legion_USE_HDF5)
    add_test(NAME attach_file_hdf5 COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:attach_file> ${Legion_TEST_ARGS} -h checkpoint.h5)
  endif()
endif()
