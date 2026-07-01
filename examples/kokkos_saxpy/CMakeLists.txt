#------------------------------------------------------------------------------#
# Copyright 2026 Kitware, Inc.
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
project(LegionExample_kokkos_saxpy)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

# request Kokkos "separable_compilation" to avoid messing with the compilation
#  of other projects
find_package(Kokkos REQUIRED OPTIONAL_COMPONENTS separable_compilation)
enable_language(${Kokkos_COMPILE_LANGUAGE})

# see if we have kokkos-kernels available
find_package(KokkosKernels QUIET)

set_source_files_properties(kokkos_saxpy.cc PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
add_executable(kokkos_saxpy kokkos_saxpy.cc)

target_link_libraries(kokkos_saxpy Kokkos::kokkoscore Legion::Legion)
target_compile_options(kokkos_saxpy PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)

if(KokkosKernels_FOUND)
  target_link_libraries(kokkos_saxpy Kokkos::kokkoskernels)
  target_compile_definitions(kokkos_saxpy PRIVATE USE_KOKKOS_KERNELS)
endif()

if(Kokkos_COMPILE_LANGUAGE STREQUAL "CUDA")
  target_compile_options(kokkos_saxpy PRIVATE --expt-extended-lambda)
endif()

if(Kokkos_${Kokkos_COMPILE_LANGUAGE}_ARCHITECTURES)
  set_property(TARGET kokkos_saxpy PROPERTY ${Kokkos_COMPILE_LANGUAGE}_ARCHITECTURES ${Kokkos_${Kokkos_COMPILE_LANGUAGE}_ARCHITECTURES})
endif()

if(Legion_ENABLE_TESTING)
  add_test(NAME kokkos_saxpy COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:kokkos_saxpy> ${Legion_TEST_ARGS})
endif()
