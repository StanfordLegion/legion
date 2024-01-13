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
project(LegionTest_hdf_attach_subregion_parallel)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

add_executable(hdf_attach_subregion_parallel tester_io.cc legion_io.cc)
target_link_libraries(hdf_attach_subregion_parallel Legion::Legion)
target_compile_options(hdf_attach_subregion_parallel PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
target_include_directories(hdf_attach_subregion_parallel PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(hdf_attach_subregion_parallel ${HDF5_LIBRARIES})
if(Legion_ENABLE_TESTING)
  add_test(NAME hdf_attach_subregion_parallel COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:hdf_attach_subregion_parallel> ${Legion_TEST_ARGS} -ll:cpu 4)
endif()
