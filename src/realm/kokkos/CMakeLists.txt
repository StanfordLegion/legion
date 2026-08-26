# Copyright 2025 Stanford University, NVIDIA Corporation, Los Alamos National Laboratory
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# since we use separable_compilation, we need a knob to override
# KOKKOS_CXX_COMPILER with the appropriate wrapper.
# In the STATIC case, CMAKE_CXX_COMPILER and KOKKOS_CXX_COMPILER must both be the correct wrapper.
set(KOKKOS_CXX_COMPILER ${Kokkos_CXX_COMPILER} CACHE STRING "C++ compiler used by Kokkos")

# Kokkos interop library - compiled separately with Kokkos-specific compiler settings
set(CMAKE_CXX_COMPILER "${KOKKOS_CXX_COMPILER}")

set(REALM_KOKKOS_SOURCES
    kokkos_interop.cc
)

# if this library is STATIC, the Kokkos INTERFACE options will propagate up
# and force us to use the same compiler (wrapper) as Kokkos globally.
add_library(realm_kokkos ${REALM_KOKKOS_SOURCES})
if(NOT "${Kokkos_COMPILE_LANGUAGE}" STREQUAL "CXX")
  # explicitly setting it to CXX adds a `-x c++`, which breaks hipcc compilation
  set_source_files_properties(${REALM_KOKKOS_SOURCES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
endif()
kokkos_compilation(SOURCE ${REALM_KOKKOS_SOURCES})

target_compile_features(realm_kokkos PUBLIC cxx_std_${REALM_CXX_STANDARD})

target_include_directories(
    realm_kokkos
    PUBLIC
        $<BUILD_INTERFACE:${REALM_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PRIVATE
        $<$<TARGET_EXISTS:CUDA::cuda_driver>:${CUDAToolkit_CUPTI_INCLUDE_DIR}>
        $<$<TARGET_EXISTS:CUDA::cuda_driver>:${CUDAToolkit_INCLUDE_DIR}>
        $<$<TARGET_EXISTS:hip::host>:${HIP_INCLUDE_DIR}>
)

# Link privately to Kokkos - this prevents Kokkos from propagating to the rest of the build
target_link_libraries(realm_kokkos PRIVATE Kokkos::kokkoscore)

# Apply standard compile options
target_compile_options(
    realm_kokkos
    PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
            -Wall
            -Wextra
            -Wno-unused-parameter>
        $<$<CXX_COMPILER_ID:MSVC>:
            /W4
            /Zc:alignedNew
            /utf-8
            /Zi
            /permissive-
            /Zc:tlsGuards
            /wd4100>
    PUBLIC
        $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus>
)

# Set visibility properties to match main library
set_target_properties(
    realm_kokkos
    PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON
)

# Apply coverage flags if enabled
if(REALM_ENABLE_COVERAGE AND NOT OPENCPPCOV_PATH)
    append_coverage_compiler_flags_to_target(realm_kokkos)
endif()
