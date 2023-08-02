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
include_guard(GLOBAL)

# Enable the CUDA language and find the CUDA toolkit.
# Works around bugs in CMake < 3.18 CUDA language support.
macro(enable_cuda_language_and_find_cuda_toolkit)

  # CMAKE_CUDA_RUNTIME_LIBRARY determines whether CUDA objs link to the static
  # or shared CUDA runtime library. Must be set before `enable_language(CUDA)`
  # https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_RUNTIME_LIBRARY.html
  if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    if(BUILD_SHARED_LIBS)
      set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
    else()
      set(CMAKE_CUDA_RUNTIME_LIBRARY STATIC)
    endif()
  endif()

  # CMake < 3.18 doesn't recognize >=17 as a CUDA standard,
  # so unset CMAKE_CUDA_STANDARD before enabling the CUDA language.
  if(CMAKE_VERSION VERSION_LESS_EQUAL "3.17" AND (CMAKE_CUDA_STANDARD GREATER_EQUAL 17))
    unset(CMAKE_CUDA_STANDARD)
    unset(CMAKE_CUDA_STANDARD CACHE)
  endif()

  # Enable the CUDA language
  enable_language(CUDA)

  # Polyfill a CMake < 3.18 missing CUDA compiler detection variable
  if(CMAKE_VERSION VERSION_LESS_EQUAL "3.17" AND (NOT CUDAToolkit_NVCC_EXECUTABLE))
    set(CUDAToolkit_NVCC_EXECUTABLE "${CMAKE_CUDA_COMPILER}")
  endif()

  # Find the CUDA toolkit
  find_package(CUDAToolkit REQUIRED)

  # Work around a clangd bug by generating compile commands
  # with `-isystem <path>` instead of `-isystem=<path>`.
  # Must come after `enable_language(CUDA)`
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-isystem ")

endmacro(enable_cuda_language_and_find_cuda_toolkit)

function(infer_cuda_standard STANDARD)
  if(CMAKE_CUDA_STANDARD)
    # Use CMAKE_CUDA_STANDARD if set
    set(${STANDARD} ${CMAKE_CUDA_STANDARD} PARENT_SCOPE)
  elseif(CMAKE_CXX_STANDARD)
    # Otherwise use CMAKE_CXX_STANDARD
    set(${STANDARD} ${CMAKE_CXX_STANDARD} PARENT_SCOPE)
  endif()
endfunction(infer_cuda_standard)

# If the variable supplied in ARCHS is empty, populate it with a list of the major CUDA architectures.
function(populate_cuda_archs_list ARCHS)

  set(archs )

  if(CMAKE_CUDA_ARCHITECTURES)
    set(archs ${CMAKE_CUDA_ARCHITECTURES})
  elseif(NOT ("${${ARCHS}}" STREQUAL ""))
    # Support comma-delimited lists in addition to semicolons
    string(REPLACE "," ";" archs "${${ARCHS}}")
  else()
    # Default to all major GPU archs

    # CMake 3.23 will figure out the list from `CUDA_ARCHITECTURES=all-major`,
    # but in older CMake we have to enumerate the architecture list manually.
    # https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.23")
      set(archs all-major)
    else()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "3.2" AND CUDAToolkit_VERSION VERSION_LESS "10.0")
        list(APPEND archs 20)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "5.0" AND CUDAToolkit_VERSION VERSION_LESS "11.0")
        list(APPEND archs 30)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "5.0" AND CUDAToolkit_VERSION VERSION_LESS "12.0")
        list(APPEND archs 35)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "6.0" AND CUDAToolkit_VERSION VERSION_LESS "12.0")
        list(APPEND archs 50)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "8.0")
        list(APPEND archs 60)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "9.0")
        list(APPEND archs 70)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
        list(APPEND archs 80)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8.0")
        list(APPEND archs 90)
      endif()
      # Compile all supported major real architectures (SASS),
      # and the highest major virtual architecture (PTX+SASS).
      list(SORT archs)
      list(POP_BACK archs highest_arch)
      list(TRANSFORM archs APPEND "-real")
      list(APPEND archs ${highest_arch})
    endif()
  endif()

  list(REMOVE_DUPLICATES archs)

  set(${ARCHS} ${archs} PARENT_SCOPE)
endfunction(populate_cuda_archs_list)

# Set a target's CUDA_STANDARD property.
# Since CMake < 3.18 doesn't recognize >=17 as a CUDA standard,
# this function falls back to setting the `-std=` compile option.
function(set_target_cuda_standard cuda_TARGET)
  set(options )
  set(oneValueArgs STANDARD)
  set(multiValueArgs)
  cmake_parse_arguments(cuda "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT ("${cuda_STANDARD}" STREQUAL ""))
    if((CMAKE_VERSION VERSION_GREATER_EQUAL "3.18") OR (cuda_STANDARD LESS 17))
      set_target_properties(${cuda_TARGET} PROPERTIES CUDA_STANDARD ${cuda_STANDARD} CUDA_STANDARD_REQUIRED ON)
    else()
      # CMake < 3.18 doesn't recognize >=17 as a CUDA standard, so set the `-std=` compile option instead.
      get_target_property(target_opts ${cuda_TARGET} COMPILE_OPTIONS)
      get_target_property(interface_opts ${cuda_TARGET} INTERFACE_COMPILE_OPTIONS)
      set(target_and_interface_opts ${target_opts} ${interface_opts})
      # Only set `-std=` if it's not already in the target's list of compile options
      if(NOT ("${target_and_interface_opts}" MATCHES ".*-std=.*"))
        target_compile_options(${cuda_TARGET} PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-std=c++${CMAKE_CXX_STANDARD}>)
      endif()
    endif()
  endif()
endfunction(set_target_cuda_standard)

# Set a target's CUDA_ARCHITECTURES property, or translate the archs list
# to --generate-code flags for CMake < 3.18
function(set_target_cuda_architectures cuda_TARGET)
  set(options )
  set(oneValueArgs)
  set(multiValueArgs ARCHITECTURES)
  cmake_parse_arguments(cuda "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Translate CUDA_ARCHITECTURES to -gencode flags for CMake < 3.18
  if(CMAKE_VERSION VERSION_LESS_EQUAL "3.17")
    set(flags )
    if(cuda_ARCHITECTURES)
      set(archs ${cuda_ARCHITECTURES})
    else()
      get_target_property(archs ${cuda_TARGET} CUDA_ARCHITECTURES)
    endif()

    # ARCH=75-real    : --generate-code=arch=compute_75,code=[sm_75]
    # ARCH=75-virtual : --generate-code=arch=compute_75,code=[compute_75]
    # ARCH=75         : --generate-code=arch=compute_75,code=[compute_75,sm_75]
    foreach(arch IN LISTS archs)
      set(codes "compute_XX" "sm_XX")
      if(arch MATCHES "-real")
        # remove "compute_XX"
        list(POP_FRONT codes)
        string(REPLACE "-real" "" arch "${arch}")
      elseif(arch MATCHES "-virtual")
        # remove "sm_XX"
        list(POP_BACK codes)
        string(REPLACE "-virtual" "" arch "${arch}")
      endif()
      list(TRANSFORM codes REPLACE "_XX" "_${arch}")
      list(JOIN codes "," codes)
      list(APPEND flags "--generate-code=arch=compute_${arch},code=[${codes}]")
    endforeach()

    target_compile_options(${cuda_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${flags}>)
  elseif(cuda_ARCHITECTURES)
    set_property(TARGET ${cuda_TARGET} PROPERTY CUDA_ARCHITECTURES ${cuda_ARCHITECTURES})
  endif()
endfunction(set_target_cuda_architectures)
  endif()
endfunction()
