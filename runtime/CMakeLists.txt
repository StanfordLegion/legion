#------------------------------------------------------------------------------#
# Copyright 2017 Kitware, Inc.
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
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION include
  FILES_MATCHING PATTERN "*.inl"
)

# Lowlevel runtime
if(Legion_USE_SHARED_LOWLEVEL)
  list(APPEND LOW_RUNTIME_SRC shared_lowlevel.cc)
else()
  list(APPEND LOW_RUNTIME_SRC
    lowlevel_config.h
    lowlevel_disk.cc
    channel.h                 channel.cc
    channel_disk.h            channel_disk.cc
    lowlevel_dma.h            lowlevel_dma.cc
    lowlevel_dma.inl
    lowlevel.h                lowlevel.cc
    lowlevel_impl.h
    realm/event_impl.h        realm/event_impl.cc
    realm/event_impl.inl
    realm/faults.h            realm/faults.cc
    realm/faults.inl
    realm/idx_impl.h          realm/idx_impl.cc
    realm/inst_impl.h         realm/inst_impl.cc
    realm/machine_impl.h      realm/machine_impl.cc
    realm/mem_impl.h          realm/mem_impl.cc
    realm/metadata.h          realm/metadata.cc
    realm/module.h            realm/module.cc
    realm/nodeset.h
    realm/numa/numa_module.h  realm/numa/numa_module.cc
    realm/numa/numasysif.h    realm/numa/numasysif.cc
    realm/operation.h         realm/operation.cc
    realm/operation.inl
    realm/proc_impl.h         realm/proc_impl.cc
    realm/procset/procset_module.h realm/procset/procset_module.cc
    realm/rsrv_impl.h         realm/rsrv_impl.cc
    realm/runtime_impl.h      realm/runtime_impl.cc
    realm/sampling_impl.h     realm/sampling_impl.cc
    realm/tasks.h             realm/tasks.cc
    realm/threads.h           realm/threads.cc
    realm/threads.inl
  )
  if(Legion_USE_CUDA)
    list(APPEND LOW_RUNTIME_SRC
      realm/cuda/cuda_module.h    realm/cuda/cuda_module.cc
      realm/cuda/cudart_hijack.h  realm/cuda/cudart_hijack.cc
    )
  endif()
  if(Legion_USE_LLVM)
    list(APPEND LOW_RUNTIME_SRC
      realm/llvmjit/llvmjit.h
      realm/llvmjit/llvmjit.inl
      realm/llvmjit/llvmjit_internal.h  realm/llvmjit/llvmjit_internal.cc
      realm/llvmjit/llvmjit_module.h    realm/llvmjit/llvmjit_module.cc
  )
  endif()
  if(Legion_USE_GASNet)
    list(APPEND LOW_RUNTIME_SRC activemsg.h activemsg.cc)
  endif()
endif()
list(APPEND LOW_RUNTIME_SRC
  accessor.h
  arrays.h
  atomics.h
  common.h
  realm/bytearray.h
  realm/bytearray.inl
  realm/circ_queue.h
  realm/circ_queue.inl
  realm/cmdline.h          realm/cmdline.cc
  realm/cmdline.inl
  realm/codedesc.h         realm/codedesc.cc
  realm/codedesc.inl
  realm/custom_serdez.h
  realm/custom_serdez.inl
  realm/dynamic_table.h
  realm/dynamic_table.inl
  realm/event.h
  realm/id.h
  realm/id.inl
  realm/indexspace.h
  realm/instance.h
  realm/logging.h          realm/logging.cc
  realm/logging.inl
  realm/machine.h
  realm/machine.inl
  realm/memory.h
  realm/pri_queue.h
  realm/pri_queue.inl
  realm/processor.h
  realm/processor.inl
  realm/profiling.h        realm/profiling.cc
  realm/profiling.inl
  realm/realm_config.h
  realm/realm.h
  realm/redop.h
  realm/reservation.h
  realm/runtime.h
  realm/sampling.h
  realm/sampling.inl
  realm/serialize.h
  realm/serialize.inl
  realm/timers.h           realm/timers.cc
  realm/timers.inl
  realm/utils.h
  utilities.h
)
find_package(Threads REQUIRED)
add_library(LowLevelRuntime ${LOW_RUNTIME_SRC})
target_compile_definitions(LowLevelRuntime PUBLIC __STDC_FORMAT_MACROS)
target_link_libraries(LowLevelRuntime
  PRIVATE ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT}
)
if(UNIX AND NOT APPLE)
  target_link_libraries(LowLevelRuntime PRIVATE rt)
endif()
set_target_properties(LowLevelRuntime PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(LowLevelRuntime PROPERTIES SOVERSION ${SOVERSION})
target_compile_definitions(LowLevelRuntime
  PRIVATE
    $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:DEBUG_LOW_LEVEL>
  PUBLIC
    COMPILE_TIME_MIN_LEVEL=LEVEL_${Legion_OUTPUT_LEVEL}
)

if(Legion_USE_HWLOC)
  target_compile_definitions(LowLevelRuntime PRIVATE REALM_USE_HWLOC)
  target_link_libraries(LowLevelRuntime PRIVATE HWLOC::HWLOC)
endif()

if(Legion_USE_GASNet)
  target_compile_definitions(LowLevelRuntime PUBLIC USE_GASNET)
  target_link_libraries(LowLevelRuntime PUBLIC GASNet::GASNet)
endif()

if(Legion_USE_LLVM)
  target_compile_definitions(LowLevelRuntime PRIVATE REALM_USE_LLVM)
  target_link_libraries(LowLevelRuntime PRIVATE LLVM::LLVM)
endif()

if(Legion_USE_CUDA)
  target_compile_definitions(LowLevelRuntime PUBLIC USE_CUDA)
  target_include_directories(LowLevelRuntime PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(LowLevelRuntime PRIVATE ${CUDA_CUDA_LIBRARY})
endif()

if(Legion_USE_SHARED_LOWLEVEL)
  set_target_properties(LowLevelRuntime PROPERTIES OUTPUT_NAME sharedllr)
else()
  set_target_properties(LowLevelRuntime PROPERTIES OUTPUT_NAME realm)
endif()

target_include_directories(LowLevelRuntime
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/legion>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/realm>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/legion>
    $<INSTALL_INTERFACE:include/realm>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} legion realm
)

install(TARGETS LowLevelRuntime EXPORT LegionTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Mapper objects
list(APPEND MAPPER_SRC
  mappers/debug_mapper.h       mappers/debug_mapper.cc
  mappers/default_mapper.h     mappers/default_mapper.cc
  mappers/mapping_utilities.h  mappers/mapping_utilities.cc
  mappers/replay_mapper.h      mappers/replay_mapper.cc
  mappers/shim_mapper.h        mappers/shim_mapper.cc
  mappers/test_mapper.h        mappers/test_mapper.cc
)

# High level runtime
list(APPEND HIGH_RUNTIME_SRC
  legion/field_tree.h
  legion/garbage_collection.h  legion/garbage_collection.cc
  legion/interval_tree.h
  legion/legion_allocation.h
  legion/legion_analysis.h     legion/legion_analysis.cc
  legion/legion_c.h            legion/legion_c.cc
  legion/legion_config.h
  legion/legion_constraint.h   legion/legion_constraint.cc
  legion/legion_context.h      legion/legion_context.cc
  legion/legion_c_util.h
  legion/legion.h              legion/legion.cc
  legion/legion.inl
  legion/legion_instances.h    legion/legion_instances.cc
  legion/legion_mapping.h      legion/legion_mapping.cc
  legion/legion_ops.h          legion/legion_ops.cc
  legion/legion_profiling.h    legion/legion_profiling.cc
  legion/legion_realm.h
  legion/legion_spy.h          legion/legion_spy.cc
  legion/legion_tasks.h        legion/legion_tasks.cc
  legion/legion_trace.h        legion/legion_trace.cc
  legion/legion_types.h
  legion/legion_utilities.h
  legion/legion_views.h        legion/legion_views.cc
  legion/mapper_manager.h      legion/mapper_manager.cc
  legion/rectangle_set.h
  legion/region_tree.h         legion/region_tree.cc
  legion/runtime.h             legion/runtime.cc
)

add_library(HighLevelRuntime ${MAPPER_SRC} ${HIGH_RUNTIME_SRC})
target_link_libraries(HighLevelRuntime LowLevelRuntime)
set_target_properties(HighLevelRuntime PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(HighLevelRuntime PROPERTIES OUTPUT_NAME legion)
set_target_properties(HighLevelRuntime PROPERTIES SOVERSION ${SOVERSION})
target_compile_definitions(HighLevelRuntime
  PRIVATE
    ASSUME_UNALLOCABLE
    $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:DEBUG_HIGH_LEVEL>
  PUBLIC
    MAX_FIELDS=${Legion_MAX_FIELDS}
)
target_include_directories(HighLevelRuntime
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/legion>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/realm>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mappers>
    $<INSTALL_INTERFACE:include/legion>
    $<INSTALL_INTERFACE:include/realm>
    $<INSTALL_INTERFACE:include/mappers>
  PRIVATE legion realm mappers
)

install(TARGETS HighLevelRuntime EXPORT LegionTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

add_library(Legion INTERFACE)
set_target_properties(Legion PROPERTIES
  INTERFACE_LINK_LIBRARIES HighLevelRuntime
)
install(TARGETS Legion EXPORT LegionTargets)
