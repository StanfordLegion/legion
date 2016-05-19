#------------------------------------------------------------------------------#
# Copyright 2016 Kitware, Inc.
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

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)

# Lowlevel runtime
if(Legion_USE_SHARED_LOWLEVEL)
  list(APPEND LOW_RUNTIME_SRC shared_lowlevel.cc)
else()
  list(APPEND LOW_RUNTIME_SRC
    realm/runtime_impl.cc
    lowlevel_dma.cc
    realm/module.cc
    realm/threads.cc
    realm/faults.cc
    realm/operation.cc
    realm/tasks.cc
    realm/metadata.cc
    realm/event_impl.cc
    realm/rsrv_impl.cc
    realm/proc_impl.cc
    realm/mem_impl.cc
    realm/inst_impl.cc
    realm/idx_impl.cc
    realm/machine_impl.cc
    realm/sampling_impl.cc
    lowlevel.cc
    lowlevel_disk.cc
    realm/numa/numa_module.cc
    realm/numa/numasysif.cc
  )
  if(Legion_USE_CUDA)
    list(APPEND LOW_RUNTIME_SRC
      realm/cuda/cuda_module.cc
      realm/cuda/cudart_hijack.cc
    )
  endif()
  if(Legion_USE_LLVM)
    list(APPEND LOW_RUNTIME_SRC
      realm/llvmjit/llvmjit_module.cc
      realm/llvmjit/llvmjit_internal.cc
    )
  endif()
  if(Legion_USE_GASNet)
    list(APPEND LOW_RUNTIME_SRC activemsg.cc)
  endif()
endif()
list(APPEND LOW_RUNTIME_SRC
  realm/logging.cc
  realm/cmdline.cc
  realm/profiling.cc
  realm/codedesc.cc
  realm/timers.cc
)
find_package(Threads REQUIRED)
add_library(legion_low_runtime ${LOW_RUNTIME_SRC})
target_link_libraries(legion_low_runtime
  PRIVATE ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties(legion_low_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_BUILD_TYPE MATCHES "DEBUG")
  target_compile_definitions(legion_low_runtime PUBLIC DEBUG_LOW_LEVEL)
endif()
target_compile_definitions(legion_low_runtime
  PUBLIC -DCOMPILE_TIME_MIN_LEVEL=LEVEL_${Legion_OUTPUT_LEVEL}
)

if(Legion_USE_HWLOC)
  target_compile_definitions(legion_low_runtime PRIVATE REALM_USE_HWLOC)
  target_link_libraries(legion_low_runtime PRIVATE HWLOC)
endif()

if(Legion_USE_GASNet)
  target_compile_definitions(legion_low_runtime PUBLIC USE_GASNET)
  target_link_libraries(legion_low_runtime PUBLIC GASNet)
endif()

if(Legion_USE_LLVM)
  target_compile_definitions(legion_low_runtime PRIVATE REALM_USE_LLVM)
  target_link_libraries(legion_low_runtime PRIVATE LLVM)
endif()

if(Legion_USE_CUDA)
  target_compile_definitions(legion_low_runtime PUBLIC USE_CUDA)
  target_include_directories(legion_low_runtime PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(legion_low_runtime PRIVATE ${CUDA_CUDA_LIBRARY})
endif()

if(Legion_USE_SHARED_LOWLEVEL)
  set_target_properties(legion_low_runtime PROPERTIES OUTPUT_NAME sharedllr)
else()
  set_target_properties(legion_low_runtime PROPERTIES OUTPUT_NAME realm)
endif()

target_include_directories(legion_low_runtime
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} legion realm
)

install(TARGETS legion_low_runtime EXPORT LegionTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# Mapper objects
list(APPEND MAPPER_SRC
  mappers/default_mapper.cc
  mappers/mapping_utilities.cc
  mappers/shim_mapper.cc
  mappers/test_mapper.cc
  mappers/replay_mapper.cc
  mappers/debug_mapper.cc
)
if(Legion_USE_ALTERNATIVE_MAPPERS)
  list(APPEND MAPPER_SRC mappers/alt_mappers.cc)
endif()

# High level runtime
list(APPEND HIGH_RUNTIME_SRC
  legion/legion.cc
  legion/legion_c.cc
  legion/legion_ops.cc
  legion/legion_tasks.cc
  legion/legion_trace.cc
  legion/legion_spy.cc
  legion/legion_profiling.cc
  legion/legion_instances.cc
  legion/legion_views.cc
  legion/legion_analysis.cc
  legion/legion_constraint.cc
  legion/legion_mapping.cc
  legion/region_tree.cc
  legion/runtime.cc
  legion/garbage_collection.cc
  legion/mapper_manager.cc
)

add_library(legion ${MAPPER_SRC} ${HIGH_RUNTIME_SRC})
target_link_libraries(legion legion_low_runtime)
set_target_properties(legion PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_BUILD_TYPE MATCHES "DEBUG")
  target_compile_definitions(legion PUBLIC -DDEBUG_HIGH_LEVEL)
endif()
target_compile_definitions(legion
  PUBLIC COMPILE_TIME_MIN_LEVEL=LEVEL_${Legion_OUTPUT_LEVEL}
)
target_include_directories(legion
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/legion>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/realm>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mappers>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} legion realm mappers
)

install(TARGETS legion EXPORT LegionTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include/legion
)
