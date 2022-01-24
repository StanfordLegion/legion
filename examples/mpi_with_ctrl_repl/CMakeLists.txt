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
project(LegionExample_mpi_with_ctrl_repl)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

find_package(MPI REQUIRED)

add_executable(mpi_with_ctrl_repl mpi_with_ctrl_repl.cc)
target_link_libraries(mpi_with_ctrl_repl Legion::Legion ${MPI_CXX_LIBRARIES})
target_include_directories(mpi_with_ctrl_repl PRIVATE ${MPI_C_INCLUDE_PATH})
if(Legion_ENABLE_TESTING)
  add_test(NAME mpi_with_ctrl_repl COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:mpi_with_ctrl_repl> ${Legion_TEST_ARGS})
endif()
