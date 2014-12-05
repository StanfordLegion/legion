#=============================================================================
# Copyright 2014 Kitware, Inc.
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
#=============================================================================

find_package(Threads REQUIRED)

set(LEGION_COMPILE_DEFS_PUBLIC)
set(LEGION_COMPILE_DEFS_PRIVATE)

if(LEGION_USE_BACKEND_SHARED)
  set(LEGION_RT_LOW_SOURCES shared_lowlevel.cc)
  list(APPEND LEGION_COMPILE_DEFS_PRIVATE SHARED_LOWLEVEL)
else()
  message(WARNING "The CMake build for Legion + GASNet doesn't quite work yet")

  find_package(GASNet REQUIRED)
  set(LEGION_RT_LOW_SOURCES lowlevel.cc activemsg.cc lowlevel_dma.cc)
  if(LEGION_USE_BACKEND_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRECTORIES})
    list(APPEND LEGION_RT_LOW_SOURCES lowlevel_gpu.cc)
    list(APPEND LEGION_COMPILE_DEFS_PRIVATE USE_CUDA)
  endif()
endif()

set(LEGION_RT_MAPPER_SOURCES
  default_mapper.cc
  shim_mapper.cc
  mapping_utilities.cc
)
if(LEGION_USE_ALTERNATE_MAPPERS)
  list(APPEND LEGION_RT_MAPPER_SOURCES alt_mappers.cc)
endif()

set(LEGION_RT_HIGH_SOURCES
  legion.cc
  legion_c.cc
  legion_ops.cc
  legion_tasks.cc
  legion_trace.cc
  legion_spy.cc
  region_tree.cc
  runtime.cc
  garbage_collection.cc
)

# Traverse the various different output levels
foreach(L RANGE 0 6)

  # Build the library
  add_library(legion-rt-l${L}
    ${LEGION_RT_LOW_SOURCES}
    ${LEGION_RT_MAPPER_SOURCES}
    ${LEGION_RT_HIGH_SOURCES}
  )
  set_target_properties(legion-rt-l${L} PROPERTIES
    OUTPUT_NAME ${LEGION_RT_NAME}-l${L}
  )
  target_link_libraries(legion-rt-l${L} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
  target_compile_definitions(legion-rt-l${L}
    PUBLIC
      COMPILE_TIME_MIN_LEVEL=${L}
      $<$<CONFIG:Debug>:DEBUG_LOW_LEVEL>
      $<$<CONFIG:Debug>:DEBUG_HIGH_LEVEL>
      ${LEGION_COMPILE_DEFS_PUBLIC}
    PRIVATE ${LEGION_COMPILE_DEFS_PRIVATE}
  )
  target_include_directories(legion-rt-l${L} INTERFACE
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/legion>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

  # Target install
  install(TARGETS legion-rt-l${L} EXPORT legion-rt-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
  )
endforeach()

# Create the build-time targets
export(EXPORT legion-rt-targets
  FILE "${Legion_BINARY_DIR}/LegionRuntimeTargets.cmake"
)

# Create the install-time targets
install(EXPORT legion-rt-targets
  DESTINATION ${LEGION_INSTALL_CMAKE_DIR}
  FILE LegionRuntimeTargets.cmake
  EXPORT_LINK_INTERFACE_LIBRARIES
)

# Header install
install(DIRECTORY ./
  DESTINATION include/legion
  FILES_MATCHING PATTERN "*.h"
)

