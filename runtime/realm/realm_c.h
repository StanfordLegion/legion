/* Copyright 2022 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// C-only header for Realm - mostly includes typedefs right now,
//  but may be expanded to provide C bindings for the Realm API

#ifndef REALM_C_H
#define REALM_C_H

#include "realm/realm_config.h"

#ifndef LEGION_USE_PYTHON_CFFI
// for size_t
#include <stddef.h>
#endif // LEGION_USE_PYTHON_CFFI

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long long realm_id_t;
#define IDFMT "%llx"

typedef unsigned int realm_address_space_t;
typedef unsigned realm_task_func_id_t;
typedef int realm_reduction_op_id_t;
typedef int realm_custom_serdez_id_t;
typedef unsigned realm_event_gen_t;
#define REALM_EVENT_GENERATION_BITS  20
typedef unsigned long long realm_barrier_timestamp_t;

// Different Processor types
#define REALM_PROCESSOR_KINDS(__op__) \
  __op__(NO_KIND, "") \
  __op__(TOC_PROC, "Throughput core") \
  __op__(LOC_PROC, "Latency core") \
  __op__(UTIL_PROC, "Utility core") \
  __op__(IO_PROC, "I/O core") \
  __op__(PROC_GROUP, "Processor group") \
  __op__(PROC_SET, "Set of Processors for OpenMP/Kokkos etc.") \
  __op__(OMP_PROC, "OpenMP (or similar) thread pool") \
  __op__(PY_PROC, "Python interpreter")

typedef enum realm_processor_kind_t {
#define C_ENUMS(name, desc) name,
  REALM_PROCESSOR_KINDS(C_ENUMS)
#undef C_ENUMS
} realm_processor_kind_t;

// Different Memory types
#define REALM_MEMORY_KINDS(__op__) \
  __op__(NO_MEMKIND, "") \
  __op__(GLOBAL_MEM, "Guaranteed visible to all processors on all nodes (e.g. GASNet memory, universally slow)") \
  __op__(SYSTEM_MEM, "Visible to all processors on a node") \
  __op__(REGDMA_MEM, "Registered memory visible to all processors on a node, can be a target of RDMA") \
  __op__(SOCKET_MEM, "Memory visible to all processors within a node, better performance to processors on same socket") \
  __op__(Z_COPY_MEM, "Zero-Copy memory visible to all CPUs within a node and one or more GPUs") \
  __op__(GPU_FB_MEM, "Framebuffer memory for one GPU and all its SMs") \
  __op__(DISK_MEM, "Disk memory visible to all processors on a node") \
  __op__(HDF_MEM, "HDF memory visible to all processors on a node") \
  __op__(FILE_MEM, "file memory visible to all processors on a node") \
  __op__(LEVEL3_CACHE, "CPU L3 Visible to all processors on the node, better performance to processors on same socket") \
  __op__(LEVEL2_CACHE, "CPU L2 Visible to all processors on the node, better performance to one processor") \
  __op__(LEVEL1_CACHE, "CPU L1 Visible to all processors on the node, better performance to one processor") \
  __op__(GPU_MANAGED_MEM, "Managed memory that can be cached by either host or GPU") \
  __op__(GPU_DYNAMIC_MEM, "Dynamically-allocated framebuffer memory for one GPU and all its SMs")

typedef enum realm_memory_kind_t {
#define C_ENUMS(name, desc) name,
  REALM_MEMORY_KINDS(C_ENUMS)
#undef C_ENUMS
} realm_memory_kind_t;

// file modes - to be removed soon
typedef enum realm_file_mode_t {
  LEGION_FILE_READ_ONLY,
  LEGION_FILE_READ_WRITE,
  LEGION_FILE_CREATE
} realm_file_mode_t;

// Prototype for a Realm task
typedef
  void (*realm_task_pointer_t)(
    const void * /*data*/,
    size_t /*datalen*/,
    const void * /*userdata*/,
    size_t /*userlen*/,
    realm_id_t /*proc_id*/);

#ifdef __cplusplus
}
#endif

#endif // ifndef REALM_C_H
