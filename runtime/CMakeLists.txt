#------------------------------------------------------------------------------#
# Copyright 2018 Kitware, Inc.
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
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.inl"
)

# Realm runtime
list(APPEND REALM_SRC
  realm/realm_c.h
  realm/transfer/lowlevel_disk.cc
  realm/transfer/channel.h                 realm/transfer/channel.cc
  realm/transfer/channel_disk.h            realm/transfer/channel_disk.cc
  realm/transfer/transfer.h                realm/transfer/transfer.cc
  realm/transfer/lowlevel_dma.h            realm/transfer/lowlevel_dma.cc
  realm/deppart/byfield.h                  realm/deppart/byfield.cc
  realm/deppart/deppart_config.h
  realm/deppart/image.h                    realm/deppart/image.cc
  realm/deppart/inst_helper.h
  realm/deppart/partitions.h               realm/deppart/partitions.cc
  realm/deppart/preimage.h                 realm/deppart/preimage.cc
  realm/deppart/rectlist.h                 
  realm/deppart/rectlist.inl
  realm/deppart/setops.h                   realm/deppart/setops.cc
  realm/deppart/sparsity_impl.h            realm/deppart/sparsity_impl.cc
  realm/deppart/sparsity_impl.inl
  realm/event_impl.h        realm/event_impl.cc
  realm/event_impl.inl
  realm/faults.h            realm/faults.cc
  realm/faults.inl
  realm/inst_impl.h         realm/inst_impl.cc
  realm/inst_layout.h       realm/inst_layout.inl
  realm/inst_layout.cc
  realm/interval_tree.h     realm/interval_tree.inl
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
  list(APPEND REALM_SRC
    realm/cuda/cuda_module.h    realm/cuda/cuda_module.cc
    realm/cuda/cudart_hijack.h  realm/cuda/cudart_hijack.cc
  )
endif()

if(Legion_USE_LLVM)
  list(APPEND REALM_SRC
    realm/llvmjit/llvmjit.h
    realm/llvmjit/llvmjit.inl
    realm/llvmjit/llvmjit_internal.h  realm/llvmjit/llvmjit_internal.cc
    realm/llvmjit/llvmjit_module.h    realm/llvmjit/llvmjit_module.cc
  )
endif()

if(Legion_USE_HDF5)
  list(APPEND REALM_SRC
    realm/hdf5/hdf5_module.h realm/hdf5/hdf5_module.cc
    realm/hdf5/hdf5_internal.h realm/hdf5/hdf5_internal.cc
    realm/hdf5/hdf5_access.h   realm/hdf5/hdf5_access.inl
    realm/hdf5/hdf5_access.cc
  )
endif()

if (REALM_USE_OPENMP)
  list(APPEND REALM_SRC
    realm/openmp/openmp_module.h realm/openmp/openmp_module.cc
    realm/openmp/openmp_threadpool.h realm/openmp/openmp_threadpool.cc
    realm/openmp/openmp_threadpool.inl
    realm/openmp/openmp_api.cc
  )
endif()

if (Legion_USE_Python)
  list(APPEND REALM_SRC
    realm/python/python_module.h realm/python/python_module.cc
    realm/python/python_source.h realm/python/python_source.cc
    realm/python/python_source.inl
    realm/python/python_internal.h
  )
endif()

list(APPEND REALM_SRC
  realm.h
  realm/activemsg.h realm/activemsg.cc
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
  realm/redop.h
  realm/reservation.h
  realm/reservation.inl
  realm/runtime.h
  realm/sampling.h
  realm/sampling.inl
  realm/serialize.h
  realm/serialize.inl
  realm/timers.h           realm/timers.cc
  realm/timers.inl
  realm/utils.h
)
find_package(Threads REQUIRED)
add_library(RealmRuntime ${REALM_SRC})
target_compile_definitions(RealmRuntime PUBLIC REALM_USE_CMAKE)
if(Legion_USE_LIBDL)
  target_link_libraries(RealmRuntime PRIVATE ${CMAKE_DL_LIBS})
endif()
target_link_libraries(RealmRuntime PRIVATE ${CMAKE_THREAD_LIBS_INIT})
if(UNIX AND NOT APPLE)
  target_link_libraries(RealmRuntime PRIVATE rt)
endif()
set_target_properties(RealmRuntime PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(RealmRuntime PROPERTIES SOVERSION ${SOVERSION})

if(Legion_USE_HWLOC)
  target_compile_definitions(RealmRuntime PRIVATE REALM_USE_HWLOC)
  target_link_libraries(RealmRuntime PRIVATE HWLOC::HWLOC)
endif()

if(Legion_USE_GASNet)
  target_link_libraries(RealmRuntime PUBLIC GASNet::GASNet)
endif()

if(Legion_USE_LLVM)
  target_link_libraries(RealmRuntime PRIVATE LLVM::LLVM)
endif()

if(Legion_USE_CUDA)
  target_include_directories(RealmRuntime PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(RealmRuntime PRIVATE ${CUDA_CUDA_LIBRARY})
endif()

if(Legion_USE_HDF5)
  target_include_directories(RealmRuntime PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(RealmRuntime PRIVATE ${HDF5_LIBRARIES})
endif()

if(Legion_USE_Python)
  target_compile_definitions(RealmRuntime PRIVATE REALM_PYTHON_LIB="${PYTHON_LIBRARIES}")
  target_compile_definitions(RealmRuntime PRIVATE REALM_PYTHON_VERSION_MAJOR=${PYTHON_VERSION_MAJOR})
endif()

set_target_properties(RealmRuntime PROPERTIES OUTPUT_NAME realm)

target_include_directories(RealmRuntime
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}

  # Include paths for generated header files.
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/runtime>
  PRIVATE
    ${PROJECT_BINARY_DIR}/runtime
)

install(TARGETS RealmRuntime EXPORT LegionTargets
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

# Legion runtime
list(APPEND LEGION_SRC
  legion.h
  legion/accessor.h
  legion/arrays.h
  legion/atomics.h
  legion/field_tree.h
  legion/garbage_collection.h             legion/garbage_collection.cc
  legion/interval_tree.h                 
  legion/legion_allocation.h             
  legion/legion_analysis.h                legion/legion_analysis.cc
  legion/legion_c.h                       legion/legion_c.cc
  legion/legion_config.h                 
  legion/legion_constraint.h              legion/legion_constraint.cc
  legion/legion_context.h                 legion/legion_context.cc
  legion/legion_c_util.h                 
  legion/legion.cc
  legion/legion.inl                      
  legion/legion_domain.h
  legion/legion_domain.inl
  legion/legion_instances.h               legion/legion_instances.cc
  legion/legion_mapping.h                 legion/legion_mapping.cc
  legion/legion_ops.h                     legion/legion_ops.cc
  legion/legion_profiling.h               legion/legion_profiling.cc
  legion/legion_profiling_serializer.h    legion/legion_profiling_serializer.cc
  legion/legion_realm.h
  legion/legion_spy.h                     legion/legion_spy.cc
  legion/legion_tasks.h                   legion/legion_tasks.cc
  legion/legion_trace.h                   legion/legion_trace.cc
  legion/legion_types.h                 
  legion/legion_utilities.h             
  legion/legion_views.h                   legion/legion_views.cc
  legion/mapper_manager.h                 legion/mapper_manager.cc
  legion/rectangle_set.h                
  legion/region_tree.h                    legion/region_tree.cc
  legion/runtime.h                        legion/runtime.cc
)

add_library(LegionRuntime ${MAPPER_SRC} ${LEGION_SRC})
target_link_libraries(LegionRuntime RealmRuntime)
set_target_properties(LegionRuntime PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(LegionRuntime PROPERTIES OUTPUT_NAME legion)
set_target_properties(LegionRuntime PROPERTIES SOVERSION ${SOVERSION})
target_compile_definitions(LegionRuntime PUBLIC LEGION_USE_CMAKE)

target_include_directories(LegionRuntime
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mappers>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/mappers>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}

  # Include paths for generated header files.
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/runtime>
  PRIVATE
    ${PROJECT_BINARY_DIR}/runtime
)

install(TARGETS LegionRuntime EXPORT LegionTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

add_library(Legion INTERFACE)
set_target_properties(Legion PROPERTIES
  INTERFACE_LINK_LIBRARIES LegionRuntime
)
install(TARGETS Legion EXPORT LegionTargets)
