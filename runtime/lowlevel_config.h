/* Copyright 2014 Stanford University
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

// ******************** IMPORTANT **************************
//
// This file is PURE C, **NOT** C++. Keep any C++-isms in
// legion_types.h, or elsewhere.
//
// ******************** IMPORTANT **************************

// The following types are all re-exported by
// LegionRuntime::LowLevel. These versions are here to facilitate the
// C API. If you are writing C++ code, use the namespaced versions.

#ifdef LEGION_IDS_ARE_64BIT
typedef unsigned long long legion_lowlevel_id_t;
#define IDFMT "%llx"
#else
typedef unsigned legion_lowlevel_id_t;
#define IDFMT "%x"
#endif

typedef unsigned int legion_lowlevel_address_space_t;
typedef unsigned legion_lowlevel_task_func_id_t;
typedef int legion_lowlevel_reduction_op_id_t;

// Different Processor types
// Keep this in sync with Processor::Kind in lowlevel.h
typedef enum legion_lowlevel_processor_kind_t {
  TOC_PROC, // Throughput core
  LOC_PROC, // Latency core
  UTIL_PROC, // Utility core
  PROC_GROUP, // Processor group
} legion_lowlevel_processor_kind_t;

// Keep this in sync with Domain::MAX_RECT_DIM in lowlevel.h
typedef enum legion_lowlevel_domain_max_rect_dim_t {
  MAX_RECT_DIM = 3,
} legion_lowlevel_domain_max_rect_dim_t;

#endif
