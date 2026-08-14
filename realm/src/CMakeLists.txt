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

set(REALM_SOURCES
    activemsg.cc
    network.cc
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
    realm_assert.cc
    repl_heap.cc
    rsrv_impl.cc
    runtime_impl.cc
    sampling_impl.cc
    subgraph_impl.cc
    tasks.cc
    threads.cc
    shm.cc
    hardware_topology.cc
    realm_c.cc
    fragmented_message.cc
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

if(REALM_USE_CUDA)
  list(APPEND REALM_SOURCES cuda/cuda_module.cc cuda/cuda_internal.cc cuda/cuda_access.cc)
  if(REALM_USE_NVTX)
    list(APPEND REALM_SOURCES nvtx.cc)
  endif()
  list(APPEND REALM_CUDA_SOURCES cuda/cuda_memcpy.cu)
endif()

if(REALM_USE_HIP)
  list(APPEND REALM_SOURCES hip/hip_module.cc hip/hip_internal.cc hip/hip_access.cc)
  list(APPEND REALM_HIP_SOURCES hip/hip_memcpy.hip)
endif()

if(REALM_USE_LLVM)
  list(APPEND REALM_SOURCES llvmjit/llvmjit_internal.cc llvmjit/llvmjit_module.cc)
endif()

if(REALM_USE_HDF5)
  list(APPEND REALM_SOURCES hdf5/hdf5_module.cc hdf5/hdf5_internal.cc hdf5/hdf5_access.cc)
endif()

if(REALM_USE_OPENMP)
  list(APPEND REALM_SOURCES openmp/openmp_module.cc)
  if(REALM_OPENMP_KMP_SUPPORT OR REALM_OPENMP_GOMP_SUPPORT)
    list(APPEND REALM_SOURCES openmp/openmp_threadpool.cc openmp/openmp_api.cc)
  endif()
endif()

if(REALM_USE_PREALM)
  list(APPEND REALM_SOURCES prealm/prealm.cc)
endif()

if(REALM_USE_PYTHON)
  list(APPEND REALM_SOURCES python/python_module.cc python/python_source.cc)
endif()

if(REALM_USE_UCX)
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

if(REALM_USE_GASNETEX)
  if (NOT REALM_ENABLE_GASNETEX_WRAPPER)
    list(APPEND REALM_SOURCES gasnet1/gasnet1_module.cc gasnet1/gasnetmsg.cc)
  endif()
  list(APPEND REALM_SOURCES gasnetex/gasnetex_module.cc gasnetex/gasnetex_internal.cc)
endif()

if(REALM_USE_MPI)
  list(APPEND REALM_SOURCES mpi/mpi_module.cc mpi/am_mpi.cc)
endif()

if(REALM_USE_KOKKOS)
  add_subdirectory(${REALM_SOURCE_DIR}/kokkos)
endif()

# Replace me later
list(TRANSFORM REALM_SOURCES PREPEND "${REALM_SOURCE_DIR}/")
list(TRANSFORM REALM_CUDA_SOURCES PREPEND "${REALM_SOURCE_DIR}/")
list(TRANSFORM REALM_HIP_SOURCES PREPEND "${REALM_SOURCE_DIR}/")

# Generated sources
# TODO(cperry): fix path so realm_defines.h lives in realm directory
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/realm_defines.h.in" "${PROJECT_BINARY_DIR}/include/realm_defines.h"
  @ONLY
)

# Embed src/realm/realm-gdb.py into librealm's .debug_gdb_scripts ELF
# section so that gdb auto-loads it whenever Realm code is debugged.
# Only useful when the library carries DWARF, so restrict to Debug and
# RelWithDebInfo builds.  ELF only -- macOS/Windows skip this entirely.
set(_realm_embed_gdb_script FALSE)
if((CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
   AND (CMAKE_BUILD_TYPE STREQUAL "Debug"
        OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
  set(_realm_embed_gdb_script TRUE)
endif()

if(_realm_embed_gdb_script)
  set(REALM_GDB_PY_PATH "${REALM_SOURCE_DIR}/realm-gdb.py")

  # Re-run CMake configuration whenever realm-gdb.py changes so the
  # embedded bytes pick up edits.
  set_property(DIRECTORY APPEND PROPERTY
    CMAKE_CONFIGURE_DEPENDS "${REALM_GDB_PY_PATH}"
  )

  # Assemble the .debug_gdb_scripts payload at configure time, framed as
  # gdb expects (see gdb_script.cc.in for the format).  Doing it in CMake
  # keeps the generated source plain C++ -- no inline asm / .incbin -- so
  # it builds with any gcc/clang on any ELF target (x86_64, aarch64, etc.).
  file(READ "${REALM_GDB_PY_PATH}" _gdb_py_hex HEX)
  string(HEX "realm-gdb.py" _gdb_name_hex)
  # 04 = SECTION_SCRIPT_ID_PYTHON_TEXT
  # 0a = '\n' terminating the script name
  # 00 = '\0' terminating the entry
  set(_gdb_full_hex "04${_gdb_name_hex}0a${_gdb_py_hex}00")
  string(REGEX REPLACE
    "([0-9a-f][0-9a-f])"
    "0x\\1,"
    REALM_GDB_PAYLOAD_C_LITERAL
    "${_gdb_full_hex}"
  )

  set(_gdb_script_cc "${CMAKE_CURRENT_BINARY_DIR}/gdb_script.cc")
  configure_file(
    "${REALM_SOURCE_DIR}/gdb_script.cc.in"
    "${_gdb_script_cc}"
    @ONLY
  )
  list(APPEND REALM_SOURCES "${_gdb_script_cc}")
endif()

# Exported so the parent scope can add the matching compile define for
# threads.cc's anchor reference.
set(REALM_EMBED_GDB_SCRIPT
    ${_realm_embed_gdb_script}
    PARENT_SCOPE
)

# generate per-dimension object files for deppart stuff
foreach(INST_N1 RANGE 1 ${REALM_MAX_DIM})
  foreach(INST_N2 RANGE 1 ${REALM_MAX_DIM})
    foreach(SRCFILE realm/deppart/image realm/deppart/preimage realm/deppart/byfield)
      set(_result_file "${CMAKE_CURRENT_BINARY_DIR}/${SRCFILE}_${INST_N1}_${INST_N2}.cc")
      # use cmake's configure_file for a portable way of creating wrapper source files
      configure_file("${PROJECT_SOURCE_DIR}/cmake/deppart_tmpl.cc.in" "${_result_file}")
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
set(REALM_HIP_SOURCES
    ${REALM_HIP_SOURCES}
    PARENT_SCOPE
)
