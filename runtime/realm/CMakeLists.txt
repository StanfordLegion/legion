#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

set(realm_HEADERS
  bytearray.h
  bytearray.inl
  circ_queue.h
  circ_queue.inl
  cmdline.h
  cmdline.inl
  codedesc.h
  codedesc.inl
  dynamic_table.h
  dynamic_table.inl
  event.h
  event_impl.h
  faults.h
  faults.inl
  id.h
  id.inl
  idx_impl.h
  indexspace.h
  inst_impl.h
  instance.h
  logging.h
  logging.inl
  machine.h
  machine.inl
  machine_impl.h
  mem_impl.h
  memory.h
  metadata.h
  module.h
  nodeset.h
  operation.h
  operation.inl
  pri_queue.h
  pri_queue.inl
  proc_impl.h
  processor.h
  processor.inl
  profiling.h
  profiling.inl
  realm.h
  realm_config.h
  redop.h
  reservation.h
  rsrv_impl.h
  rsrv_impl.inl
  runtime.h
  runtime_impl.h
  sampling.h
  sampling.inl
  sampling_impl.h
  serialize.h
  serialize.inl
  tasks.h
  threads.h
  threads.inl
  timers.h
  timers.inl
  custom_serdez.h
  custom_serdez.inl
  utils.h
  numa/numa_module.h
  numa/numasysif.h
)

set(realm_SOURCES
  cmdline.cc
  codedesc.cc
  event_impl.cc
  faults.cc
  idx_impl.cc
  inst_impl.cc
  logging.cc
  machine_impl.cc
  mem_impl.cc
  metadata.cc
  ../lowlevel.cc
  ../lowlevel_disk.cc
  ../lowlevel_dma.cc
  module.cc
  operation.cc
  proc_impl.cc
  profiling.cc
  rsrv_impl.cc
  runtime_impl.cc
  sampling_impl.cc
  tasks.cc
  threads.cc
  timers.cc
  numa/numa_module.cc
  numa/numasysif.cc
)
 
if (USE_LLVM)
set(realm_llvm_HEADERS
  llvmjit/llvmjit.h
  llvmjit/llvmjit.inl
  llvmjit/llvmjit_internal.h
  llvmjit/llvmjit_module.h)
set(realm_llvm_SOURCES
  llvmjit/llvmjit_internal.cc
  llvmjit/llvmjit_module.cc
 )
endif (USE_LLVM)

if (ENABLE_CUDA)
 set(realm_cuda_HEADERS
   cuda/cuda_module.h
   cuda/cudart_hijack.h
 )

 set(realm_cuda_SOURCES
   cuda/cuda_module.cc
   cuda/cudart_hijack.cc
 )
endif (ENABLE_CUDA)

add_library(realm ${realm_cuda_SOURCES} ${realm_SOURCES})
target_include_directories(realm
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_SOURCE_DIR}/runtime/legion
    ${CMAKE_SOURCE_DIR}/runtime/realm)
if(Legion_USE_GASNet)
   target_compile_definitions(realm PUBLIC USE_GASNET)
   target_link_libraries(realm PUBLIC GASNet::GASNet)
endif(Legion_USE_GASNet)

set ( PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
set ( PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
set ( PACKAGE_INCLUDE_DESTINATION "${CMAKE_INSTALL_PREFIX}/include/realm")

install(
  TARGETS realm EXPORT realmTargets
  ARCHIVE DESTINATION ${PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${PACKAGE_BIN_DESTINATION}
)


foreach (file ${realm_HEADERS})
   install (FILES ${file} DESTINATION ${PACKAGE_INCLUDE_DESTINATION})
endforeach()

if (ENABLE_CUDA)
foreach (file ${realm_cuda_HEADERS})
   install (FILES ${file} DESTINATION ${PACKAGE_INCLUDE_DESTINATION})
endforeach()
endif (ENABLE_CUDA)


#------------------------------------------------------------------------------#
# vim: set tabstop=2 shiftwidth=2 expandtab :
#------------------------------------------------------------------------------#
