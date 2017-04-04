/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#ifndef RUNTIME_LOWLEVEL_CONFIG_H
#define RUNTIME_LOWLEVEL_CONFIG_H

// for size_t
#include <stddef.h>

// ******************** IMPORTANT **************************
//
// This file is PURE C, **NOT** C++. Keep any C++-isms in
// legion_types.h, or elsewhere.
//
// ******************** IMPORTANT **************************

// The following types are all re-exported by
// LegionRuntime::LowLevel. These versions are here to facilitate the
// C API. If you are writing C++ code, use the namespaced versions.

typedef unsigned long long legion_lowlevel_id_t;
#define IDFMT "%llx"

typedef long long legion_lowlevel_coord_t;

typedef unsigned int legion_lowlevel_address_space_t;
typedef unsigned legion_lowlevel_task_func_id_t;
typedef int legion_lowlevel_reduction_op_id_t;
typedef int legion_lowlevel_custom_serdez_id_t;
typedef unsigned legion_lowlevel_event_gen_t;
typedef unsigned long long legion_lowlevel_barrier_timestamp_t;

// Different Processor types
// Keep this in sync with Processor::Kind in lowlevel.h
typedef enum legion_lowlevel_processor_kind_t {
  NO_KIND,
  TOC_PROC, // Throughput core
  LOC_PROC, // Latency core
  UTIL_PROC, // Utility core
  IO_PROC, // I/O core
  PROC_GROUP, // Processor group
  PROC_SET, // Set of Processors for OpenMP/Kokkos etc.
  OMP_PROC, // OpenMP (or similar) thread pool
  PY_PROC, // Python processor
} legion_lowlevel_processor_kind_t;

// Different Memory types
// Keep this in sync with Memory::Kind in lowlevel.h
typedef enum legion_lowlevel_memory_kind_t {
  GLOBAL_MEM, // Guaranteed visible to all processors on all nodes (e.g. GASNet memory, universally slow)
  SYSTEM_MEM, // Visible to all processors on a node
  REGDMA_MEM, // Registered memory visible to all processors on a node, can be a target of RDMA
  SOCKET_MEM, // Memory visible to all processors within a node, better performance to processors on same socket 
  Z_COPY_MEM, // Zero-Copy memory visible to all CPUs within a node and one or more GPUs 
  GPU_FB_MEM,   // Framebuffer memory for one GPU and all its SMs
  DISK_MEM,   // Disk memory visible to all processors on a node
  HDF_MEM,    // HDF memory visible to all processors on a node
  FILE_MEM,   // file memory visible to all processors on a node
  LEVEL3_CACHE, // CPU L3 Visible to all processors on the node, better performance to processors on same socket 
  LEVEL2_CACHE, // CPU L2 Visible to all processors on the node, better performance to one processor
  LEVEL1_CACHE, // CPU L1 Visible to all processors on the node, better performance to one processor
} legion_lowlevel_memory_kind_t;

typedef enum legion_lowlevel_file_mode_t {
  LEGION_FILE_READ_ONLY,
  LEGION_FILE_READ_WRITE,
  LEGION_FILE_CREATE
} legion_lowlevel_file_mode_t;

// Keep this in sync with Domain::MAX_RECT_DIM in lowlevel.h
#define REALM_MAX_POINT_DIM 3
#define REALM_MAX_RECT_DIM 3
typedef enum legion_lowlevel_domain_max_rect_dim_t {
  MAX_POINT_DIM = REALM_MAX_POINT_DIM,
  MAX_RECT_DIM = REALM_MAX_RECT_DIM,
} legion_lowlevel_domain_max_rect_dim_t;

// Prototype for a Realm task
typedef
  void (*legion_lowlevel_task_pointer_t)(
    const void * /*data*/,
    size_t /*datalen*/,
    const void * /*userdata*/,
    size_t /*userlen*/,
    legion_lowlevel_id_t /*proc_id*/);

#endif
