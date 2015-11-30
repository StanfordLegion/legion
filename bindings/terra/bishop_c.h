/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#include "legion_c.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LIST_TYPE(NAME, TYPE, BASE) typedef struct TYPE {  \
  BASE* list;                                              \
  unsigned size;                                           \
} TYPE;                                                    \
                                                           \
void bishop_delete_##NAME##_list(TYPE);                    \

LIST_TYPE(processor, bishop_processor_list_t, legion_processor_t)
LIST_TYPE(memory, bishop_memory_list_t, legion_memory_t)

typedef bool (*bishop_task_predicate_t)(legion_task_t);
typedef bool (*bishop_region_predicate_t)(legion_task_t,
                                          legion_region_requirement_t);

typedef void (*bishop_task_callback_fn_t)(legion_task_t);
typedef void (*bishop_region_callback_fn_t)(legion_task_t,
                                            legion_region_requirement_t);

typedef struct bishop_task_rule_t {
  bishop_task_callback_fn_t select_task_options;
  bishop_task_callback_fn_t select_task_variant;
} bishop_task_rule_t;

typedef struct bishop_region_rule_t {
  bishop_region_callback_fn_t pre_map_task;
  bishop_region_callback_fn_t map_task;
} bishop_region_rule_t;

typedef enum bishop_isa_t {
  X86_ISA = 1,
  CUDA_ISA
} bishop_isa_t;

void register_bishop_mappers(bishop_task_rule_t*, unsigned,
                             bishop_region_rule_t*, unsigned);

bishop_processor_list_t bishop_all_processors();

bishop_processor_list_t bishop_filter_processors_by_isa(bishop_processor_list_t,
                                                        bishop_isa_t);

bishop_memory_list_t bishop_filter_memories_by_visibility(legion_processor_t);

bishop_memory_list_t bishop_filter_memories_by_kind(bishop_memory_list_t,
                                                    legion_memory_kind_t);

bool bishop_task_set_target_processor(legion_task_t, legion_processor_t);
bool bishop_task_set_target_processor_list(legion_task_t,
                                           bishop_processor_list_t);

bool bishop_region_set_target_memory(legion_region_requirement_t,
                                     legion_memory_t);
bool bishop_region_set_target_memory_list(legion_region_requirement_t,
                                          bishop_memory_list_t);

bishop_isa_t bishop_processor_get_isa(legion_processor_t);

void bishop_logger_info(const char* msg, ...)
  __attribute__((format (printf, 1, 2)));
void bishop_logger_warning(const char* msg, ...)
  __attribute__((format (printf, 1, 2)));

#ifdef __cplusplus
}
#endif

#endif // __BISHOP_C_H__
