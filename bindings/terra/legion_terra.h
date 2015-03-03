/* Copyright 2015 Stanford University
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

#ifndef __LEGION_TERRA_H__
#define __LEGION_TERRA_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "legion_c.h"

void register_reduction_plus_float(legion_reduction_op_id_t redop);
void register_reduction_plus_double(legion_reduction_op_id_t redop);
void register_reduction_plus_int32(legion_reduction_op_id_t redop);

void register_reduction_minus_float(legion_reduction_op_id_t redop);
void register_reduction_minus_double(legion_reduction_op_id_t redop);
void register_reduction_minus_int32(legion_reduction_op_id_t redop);

void register_reduction_times_float(legion_reduction_op_id_t redop);
void register_reduction_times_double(legion_reduction_op_id_t redop);
void register_reduction_times_int32(legion_reduction_op_id_t redop);

void register_reduction_divide_float(legion_reduction_op_id_t redop);
void register_reduction_divide_double(legion_reduction_op_id_t redop);
void register_reduction_divide_int32(legion_reduction_op_id_t redop);

void register_reduction_max_float(legion_reduction_op_id_t redop);
void register_reduction_max_double(legion_reduction_op_id_t redop);
void register_reduction_max_int32(legion_reduction_op_id_t redop);

void register_reduction_min_float(legion_reduction_op_id_t redop);
void register_reduction_min_double(legion_reduction_op_id_t redop);
void register_reduction_min_int32(legion_reduction_op_id_t redop);

void reduce_plus_float(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, float value);
void reduce_plus_double(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, double value);
void reduce_plus_int32(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, int value);

void reduce_minus_float(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, float value);
void reduce_minus_double(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, double value);
void reduce_minus_int32(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, int value);

void reduce_times_float(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, float value);
void reduce_times_double(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, double value);
void reduce_times_int32(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, int value);

void reduce_divide_float(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, float value);
void reduce_divide_double(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, double value);
void reduce_divide_int32(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, int value);

void reduce_max_float(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, float value);
void reduce_max_double(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, double value);
void reduce_max_int32(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, int value);

void reduce_min_float(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, float value);
void reduce_min_double(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, double value);
void reduce_min_int32(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, int value);

void safe_reduce_plus_float(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, float value);
void safe_reduce_plus_double(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, double value);
void safe_reduce_plus_int32(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, int value);

void safe_reduce_minus_float(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, float value);
void safe_reduce_minus_double(legion_accessor_generic_t accessor,
                              legion_ptr_t ptr, double value);
void safe_reduce_minus_int32(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, int value);

void safe_reduce_times_float(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, float value);
void safe_reduce_times_double(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, double value);
void safe_reduce_times_int32(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, int value);

void safe_reduce_divide_float(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, float value);
void safe_reduce_divide_double(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, double value);
void safe_reduce_divide_int32(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, int value);

void safe_reduce_max_float(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, float value);
void safe_reduce_max_double(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, double value);
void safe_reduce_max_int32(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, int value);

void safe_reduce_min_float(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, float value);
void safe_reduce_min_double(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, double value);
void safe_reduce_min_int32(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, int value);

void set_lua_registration_callback_name(char*);

legion_mapper_t create_mapper(const char*,
                              legion_machine_t,
                              legion_runtime_t,
                              legion_processor_t);

void lua_registration_callback_wrapper(legion_machine_t,
                                       legion_runtime_t,
                                       const legion_processor_t*,
                                       unsigned);

void lua_task_wrapper_void(legion_task_t,
                           const legion_physical_region_t*,
                           unsigned,
                           legion_context_t,
                           legion_runtime_t);

legion_task_result_t lua_task_wrapper(legion_task_t,
                                      const legion_physical_region_t*,
                                      unsigned,
                                      legion_context_t,
                                      legion_runtime_t);


typedef
  struct vector_legion_domain_split_t { void *impl; }
vector_legion_domain_split_t;

void
vector_legion_domain_split_push_back(vector_legion_domain_split_t,
                                     legion_domain_split_t);

unsigned
vector_legion_domain_split_size(vector_legion_domain_split_t);

legion_domain_split_t
vector_legion_domain_split_get(vector_legion_domain_split_t,
                               unsigned);

void decompose_index_space(legion_domain_t,
                           legion_processor_t*,
                           unsigned, unsigned,
                           vector_legion_domain_split_t);

#ifdef __cplusplus
}
#endif

#endif // __LEGION_TERRA_H__
