#------------------------------------------------------------------------------#
# Copyright 2022 Kitware, Inc.
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
project(LegionExample_omp_saxpy)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

find_package(OpenMP REQUIRED)

add_executable(omp_saxpy omp_saxpy.cc
                         simple_blas.h simple_blas.inl simple_blas.cc)

target_link_libraries(omp_saxpy Legion::Legion)

# NOTE: we need the '-fopenmp' (or however it's spelled) in the compile options,
#  but do NOT want it in the link_libraries - Realm is providing the OMP runtime
target_compile_options(omp_saxpy PRIVATE "${OpenMP_CXX_FLAGS}")

if(Legion_ENABLE_TESTING)
  add_test(NAME omp_saxpy COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:omp_saxpy> -ll:ocpu 1 ${Legion_TEST_ARGS})
endif()
