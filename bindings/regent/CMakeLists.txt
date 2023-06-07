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
project(LegionBindings_regent)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

# Regent bindings
list(APPEND REGENT_SRC
  regent.h                regent.cc
  regent_partitions.h     regent_partitions.cc
  regent_partitions_cxx.h
  regent_util.h           regent_redop.h
)

if(Legion_USE_CUDA)
  list(APPEND REGENT_SRC
    regent_cudart_hijack.h regent_cudart_hijack.cc
    regent_cuda.cu
  )
elseif(Legion_USE_HIP)
  list(APPEND REGENT_SRC
    regent_cuda.cu
  )
endif()

# Bishop bindings
list(APPEND REGENT_SRC
  bishop_mapper.h bishop_mapper.cc
  bishop_c.h      bishop_c.cc
)

# Murmur3 hash
list(APPEND REGENT_SRC
  murmur_hash3.h  murmur_hash3.cc
)

if(Legion_USE_HIP)
  hip_add_library(Regent SHARED ${REGENT_SRC})
else()
  add_library(Regent SHARED ${REGENT_SRC})
endif()
add_library(Legion::Regent ALIAS Regent)
target_link_libraries(Regent PUBLIC Legion::Legion)

if(Legion_USE_CUDA)
  target_link_libraries(Regent PRIVATE CUDA::toolkit)
  set_target_cuda_architectures(TARGET Regent ARCHS ${Legion_CUDA_ARCH})
  # starting with nvcc 10.2, we can error out on all warnings
  if(Legion_BUILD_WARN_AS_ERROR AND (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 10.2))
    target_compile_options(Regent PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Werror=all-warnings>)
  endif()
  target_compile_options(Regent PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Wno-deprecated-gpu-targets>)
elseif(Legion_USE_HIP)
  target_include_directories(Regent PRIVATE ${HIP_INCLUDE_DIRS})
  if (Legion_HIP_TARGET STREQUAL "CUDA")
    target_compile_definitions(Regent PRIVATE __HIP_PLATFORM_NVIDIA__)
  else()
    target_compile_definitions(Regent PRIVATE __HIP_PLATFORM_AMD__)
  endif()
endif()

if(Legion_USE_HIP)
  target_include_directories(Regent PRIVATE ${HIP_INCLUDE_DIRS})
endif()

set_target_properties(Regent PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(Regent PROPERTIES OUTPUT_NAME "regent${INSTALL_SUFFIX}")
set_target_properties(Regent PROPERTIES SOVERSION ${SOVERSION})
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(Regent PROPERTIES BUILD_RPATH "\$ORIGIN")
  set_target_properties(Regent PROPERTIES INSTALL_RPATH "\$ORIGIN")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set_target_properties(Regent PROPERTIES BUILD_RPATH "@loader_path")
  set_target_properties(Regent PROPERTIES INSTALL_RPATH "@loader_path")
endif ()

install(TARGETS Regent EXPORT LegionTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
