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
project(LegionExample_kokkos_saxpy)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

# request Kokkos "separable_compilation" to avoid messing with the compilation
#  of other projects
find_package(Kokkos REQUIRED OPTIONAL_COMPONENTS separable_compilation)

# see if we have kokkos-kernels available
find_package(KokkosKernels QUIET)

add_executable(kokkos_saxpy kokkos_saxpy.cc)

target_link_libraries(kokkos_saxpy Kokkos::kokkoscore)
target_link_libraries(kokkos_saxpy Legion::Legion)

if(KokkosKernels_FOUND)
  target_link_libraries(kokkos_saxpy Kokkos::kokkoskernels)
  target_compile_definitions(kokkos_saxpy PRIVATE USE_KOKKOS_KERNELS)
endif()

if(Kokkos_ENABLE_CUDA)
  # have to use Kokkos' choice of compiler for cuda-related options to work
  set(CMAKE_CXX_COMPILER ${KOKKOS_CXX_COMPILER})
  if(KOKKOS_CXX_COMPILER MATCHES ".*clang.*")
    target_compile_options(kokkos_saxpy PRIVATE --cuda-path=${CUDA_TOOLKIT_ROOT_DIR})
  endif()
endif()

if(Legion_ENABLE_TESTING)
  add_test(NAME kokkos_saxpy COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:kokkos_saxpy> ${Legion_TEST_ARGS})
endif()
