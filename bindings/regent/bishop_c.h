/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#ifndef __BISHOP_C_H__
#define __BISHOP_C_H__

#define LEGION_ENABLE_C_BINDINGS
#include "legion.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LIST_TYPE(NAME, TYPE, BASE) typedef struct TYPE {  \
  BASE* list;                                              \
  unsigned size;                                           \
  unsigned persistent;                                     \
} TYPE;                                                    \
                                                           \
void bishop_create_##NAME##_list(TYPE);                    \
void bishop_delete_##NAME##_list(TYPE);                    \

LIST_TYPE(processor, bishop_processor_list_t, legion_processor_t)
LIST_TYPE(memory, bishop_memory_list_t, legion_memory_t)
LIST_TYPE(field, bishop_field_list_t, legion_field_id_t)

typedef void* bishop_mapper_state_t;

typedef
  void (*bishop_select_task_options_fn_t)(
      bishop_mapper_state_t,
      legion_mapper_runtime_t,
      legion_mapper_context_t,
      legion_task_t,
      legion_task_options_t*);

typedef
  void (*bishop_slice_task_fn_t)(
      bishop_mapper_state_t,
      legion_mapper_runtime_t,
      legion_mapper_context_t,
      legion_task_t,
      legion_slice_task_input_t,
      legion_slice_task_output_t);

typedef
  void (*bishop_map_task_fn_t)(
      bishop_mapper_state_t,
      legion_mapper_runtime_t,
      legion_mapper_context_t,
      legion_task_t,
      legion_map_task_input_t,
      legion_map_task_output_t);

typedef
  void (*bishop_mapper_state_init_fn_t)(
      bishop_mapper_state_t*);

typedef unsigned int bishop_matching_state_t;

typedef
  bishop_matching_state_t (*bishop_transition_fn_t)(
      legion_task_t);

typedef struct bishop_mapper_impl_t {
  bishop_select_task_options_fn_t select_task_options;
  bishop_slice_task_fn_t slice_task;
  bishop_map_task_fn_t map_task;
} bishop_mapper_impl_t;

typedef legion_isa_kind_t bishop_isa_t;

void
register_bishop_mappers(bishop_mapper_impl_t*,
                        unsigned,
                        bishop_transition_fn_t*,
                        unsigned,
                        bishop_mapper_state_init_fn_t);

bishop_processor_list_t
bishop_all_processors();

extern legion_processor_t NO_PROC;

legion_processor_t
bishop_get_no_processor();

legion_memory_t
bishop_get_no_memory();

bishop_processor_list_t
bishop_filter_processors_by_isa(bishop_processor_list_t,
                                bishop_isa_t);

bishop_processor_list_t
bishop_filter_processors_by_kind(bishop_processor_list_t,
                                 legion_processor_kind_t);

bishop_memory_list_t
bishop_filter_memories_by_visibility(legion_processor_t);

bishop_memory_list_t
bishop_filter_memories_by_kind(bishop_memory_list_t,
                               legion_memory_kind_t);

bishop_memory_list_t
bishop_all_memories();

bishop_isa_t
bishop_processor_get_isa(legion_processor_t);

legion_memory_t
bishop_physical_region_get_memory(legion_physical_region_t);

bishop_memory_list_t
bishop_physical_region_get_memories(legion_physical_region_t);

bishop_field_list_t
bishop_physical_region_get_fields(legion_physical_region_t);

typedef struct bishop_instance_cache_t {
  void *impl;
} bishop_instance_cache_t;

bishop_instance_cache_t
bishop_instance_cache_create();

legion_physical_instance_t*
bishop_instance_cache_get_cached_instances(bishop_instance_cache_t,
                                           size_t,
                                           legion_logical_region_t,
                                           legion_memory_t);

bool
bishop_instance_cache_register_instances(bishop_instance_cache_t,
                                         size_t,
                                         legion_logical_region_t,
                                         legion_memory_t,
                                         legion_physical_instance_t*);

typedef struct bishop_slice_cache_t {
  void *impl;
} bishop_slice_cache_t;

bishop_slice_cache_t
bishop_slice_cache_create();

bool
bishop_slice_cache_has_cached_slices(bishop_slice_cache_t,
                                     legion_domain_t);

void
bishop_slice_cache_copy_cached_slices(bishop_slice_cache_t,
                                      legion_domain_t,
                                      legion_slice_task_output_t);

void
bishop_slice_cache_add_entry(bishop_slice_cache_t,
                             legion_domain_t,
                             legion_slice_task_output_t);

coord_t
bishop_domain_point_linearize(legion_domain_point_t point,
                              legion_domain_t domain,
                              legion_task_t task);

void
bishop_logger_info(const char* msg, ...)
  __attribute__((format (printf, 1, 2)));

void
bishop_logger_warning(const char* msg, ...)
  __attribute__((format (printf, 1, 2)));

void
bishop_logger_debug(const char* msg, ...)
  __attribute__((format (printf, 1, 2)));

#ifdef __cplusplus
}
#endif

#endif // __BISHOP_C_H__
