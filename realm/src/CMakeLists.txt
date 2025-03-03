# Copyright 2025 Stanford University, NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

set(REALM_SOURCES
    activemsg.cc
    network.cc
    cmdline.cc
    codedesc.cc
    logging.cc
    mutex.cc
    profiling.cc
    timers.cc
    utils.cc
    bgwork.cc
    comp_queue_impl.cc
    event_impl.cc
    barrier_impl.cc
    faults.cc
    idx_impl.cc
    inst_impl.cc
    inst_layout.cc
    machine_impl.cc
    mem_impl.cc
    metadata.cc
    module.cc
    module_config.cc
    nodeset.cc
    operation.cc
    proc_impl.cc
    repl_heap.cc
    rsrv_impl.cc
    runtime_impl.cc
    sampling_impl.cc
    subgraph_impl.cc
    tasks.cc
    threads.cc
    shm.cc
    hardware_topology.cc
    transfer/lowlevel_disk.cc
    transfer/address_list.cc
    transfer/channel_common.cc
    transfer/memcpy_channel.cc
    transfer/channel.cc
    transfer/addrsplit_channel.cc
    transfer/channel_disk.cc
    transfer/transfer.cc
    transfer/lowlevel_dma.cc
    transfer/ib_memory.cc
    deppart/byfield.cc
    deppart/partitions.cc
    deppart/setops.cc
    deppart/sparsity_impl.cc
    numa/numa_module.cc
    numa/numasysif.cc
    procset/procset_module.cc
)

if(TARGET CUDA::cuda_driver)
  list(APPEND REALM_SOURCES cuda/cuda_module.cc cuda/cuda_internal.cc cuda/cuda_access.cc
       cuda/cuda_memcpy.cu
  )
  if(REALM_USE_NVTX)
    list(APPEND REALM_SOURCES nvtx.cc)
  endif()
  list(APPEND REALM_CUDA_SOURCES cuda/cuda_memcpy.cu)
endif()

if(TARGET hip::hip_host)
  list(APPEND REALM_SOURCES hip/hip_module.cc hip/hip_internal.cc hip/hip_access.cc)
endif()

if(TARGET LLVM::LLVM)
  list(APPEND REALM_SOURCES llvmjit/llvmjit_internal.cc llvmjit/llvmjit_module.cc)
endif()

if(TARGET hdf5::hdf5)
  list(APPEND REALM_SOURCES hdf5/hdf5_module.cc hdf5/hdf5_internal.cc hdf5/hdf5_access.cc)
endif()

if(REALM_USE_OPENMP)
  list(APPEND REALM_SOURCES openmp/openmp_module.cc)
  if(NOT TARGET OpenMP::OpenMP_CXX)
    list(APPEND REALM_SOURCES openmp/openmp_threadpool.cc openmp/openmp_api.cc)
  endif()
endif()

if(TARGET Python3::Python)
  list(APPEND REALM_SOURCES python/python_module.cc python/python_source.cc)
endif()

if(TARGET ucx::ucp)
  list(
    APPEND
    REALM_SOURCES
    ucx/ucp_module.cc
    ucx/ucp_internal.cc
    ucx/ucp_context.cc
    ucx/mpool.cc
    ucx/ucc_comm.cc
    ucx/oob_group_comm.cc
    ucx/bootstrap/bootstrap.cc
    ucx/bootstrap/bootstrap_loader.cc
  )
endif()

if(TARGET GASNet::GASNet)
  list(APPEND REALM_SOURCES gasnet1/gasnet1_module.cc gasnet1/gasnetmsg.cc)
  list(APPEND REALM_SOURCES gasnetex/gasnetex_module.cc gasnetex/gasnetex_internal.cc)
endif()

if(TARGET MPI::MPI_CXX)
  list(APPEND REALM_SOURCES mpi/mpi_module.cc mpi/am_mpi.cc)
endif()

if(TARGET Kokkos::kokkoscore)
  list(APPEND REALM_SOURCES kokkos_interop.cc)
endif()

# Replace me later
list(TRANSFORM REALM_SOURCES PREPEND "${REALM_SOURCE_DIR}/")
list(TRANSFORM REALM_CUDA_SOURCES PREPEND "${REALM_SOURCE_DIR}/")

# Generated sources
# TODO(cperry): fix path
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/realm_defines.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/../include/realm_defines.h" @ONLY
)

# generate per-dimension object files for deppart stuff
foreach(INST_N1 RANGE 1 ${REALM_MAX_DIM})
  foreach(INST_N2 RANGE 1 ${REALM_MAX_DIM})
    foreach(SRCFILE realm/deppart/image realm/deppart/preimage realm/deppart/byfield)
      set(_result_file "${CMAKE_CURRENT_BINARY_DIR}/${SRCFILE}_${INST_N1}_${INST_N2}.cc")
      # use cmake's configure_file for a portable way of creating wrapper source files
      configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/deppart_tmpl.cc.in" "${_result_file}")
      list(APPEND REALM_SOURCES "${_result_file}")
    endforeach()
  endforeach()
endforeach()

set(REALM_SOURCES
    ${REALM_SOURCES}
    PARENT_SCOPE
)
set(REALM_RC_SOURCES
    ${REALM_RC_SOURCES}
    PARENT_SCOPE
)
set(REALM_CUDA_SOURCES
    ${REALM_CUDA_SOURCES}
    PARENT_SCOPE
)
