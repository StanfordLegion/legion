/* Copyright 2017 Stanford University
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

#include "legion_c.h"

#ifdef __cplusplus
extern "C" {
#endif

void register_reduction_plus_float(legion_reduction_op_id_t redop);
void register_reduction_plus_double(legion_reduction_op_id_t redop);
void register_reduction_plus_int32(legion_reduction_op_id_t redop);
void register_reduction_plus_int64(legion_reduction_op_id_t redop);

void register_reduction_minus_float(legion_reduction_op_id_t redop);
void register_reduction_minus_double(legion_reduction_op_id_t redop);
void register_reduction_minus_int32(legion_reduction_op_id_t redop);
void register_reduction_minus_int64(legion_reduction_op_id_t redop);

void register_reduction_times_float(legion_reduction_op_id_t redop);
void register_reduction_times_double(legion_reduction_op_id_t redop);
void register_reduction_times_int32(legion_reduction_op_id_t redop);
void register_reduction_times_int64(legion_reduction_op_id_t redop);

void register_reduction_divide_float(legion_reduction_op_id_t redop);
void register_reduction_divide_double(legion_reduction_op_id_t redop);
void register_reduction_divide_int32(legion_reduction_op_id_t redop);
void register_reduction_divide_int64(legion_reduction_op_id_t redop);

void register_reduction_max_float(legion_reduction_op_id_t redop);
void register_reduction_max_double(legion_reduction_op_id_t redop);
void register_reduction_max_int32(legion_reduction_op_id_t redop);
void register_reduction_max_int64(legion_reduction_op_id_t redop);

void register_reduction_min_float(legion_reduction_op_id_t redop);
void register_reduction_min_double(legion_reduction_op_id_t redop);
void register_reduction_min_int32(legion_reduction_op_id_t redop);
void register_reduction_min_int64(legion_reduction_op_id_t redop);

void reduce_plus_float(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, float value);
void reduce_plus_float_domain_point(legion_accessor_generic_t accessor,
                                    legion_domain_point_t, float value);
void reduce_plus_double(legion_accessor_generic_t accessor,
                        legion_ptr_t ptr, double value);
void reduce_plus_double_domain_point(legion_accessor_generic_t accessor,
                                     legion_domain_point_t, double value);
void reduce_plus_int32(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, int value);
void reduce_plus_int32_domain_point(legion_accessor_generic_t accessor,
                                    legion_domain_point_t, int value);
void reduce_plus_int64(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, long long int value);
void reduce_plus_int64_domain_point(legion_accessor_generic_t accessor,
                                    legion_domain_point_t, long long int value);

void reduce_minus_float(legion_accessor_generic_t accessor,
                        legion_ptr_t ptr, float value);
void reduce_minus_float_domain_point(legion_accessor_generic_t accessor,
                                     legion_domain_point_t, float value);
void reduce_minus_double(legion_accessor_generic_t accessor,
                         legion_ptr_t ptr, double value);
void reduce_minus_double_domain_point(legion_accessor_generic_t accessor,
                                      legion_domain_point_t, double value);
void reduce_minus_int32(legion_accessor_generic_t accessor,
                        legion_ptr_t ptr, int value);
void reduce_minus_int32_domain_point(legion_accessor_generic_t accessor,
                                     legion_domain_point_t, int value);
void reduce_minus_int64(legion_accessor_generic_t accessor,
                        legion_ptr_t ptr, long long int value);
void reduce_minus_int64_domain_point(legion_accessor_generic_t accessor,
                                     legion_domain_point_t, long long int value);

void reduce_times_float(legion_accessor_generic_t accessor,
                        legion_ptr_t ptr, float value);
void reduce_times_float_domain_point(legion_accessor_generic_t accessor,
                                     legion_domain_point_t, float value);
void reduce_times_double(legion_accessor_generic_t accessor,
                         legion_ptr_t ptr, double value);
void reduce_times_double_domain_point(legion_accessor_generic_t accessor,
                                      legion_domain_point_t, double value);
void reduce_times_int32(legion_accessor_generic_t accessor,
                        legion_ptr_t ptr, int value);
void reduce_times_int32_domain_point(legion_accessor_generic_t accessor,
                                     legion_domain_point_t, int value);
void reduce_times_int64(legion_accessor_generic_t accessor,
                        legion_ptr_t ptr, long long int value);
void reduce_times_int64_domain_point(legion_accessor_generic_t accessor,
                                     legion_domain_point_t, long long int value);

void reduce_divide_float(legion_accessor_generic_t accessor,
                         legion_ptr_t ptr, float value);
void reduce_divide_float_domain_point(legion_accessor_generic_t accessor,
                                      legion_domain_point_t, float value);
void reduce_divide_double(legion_accessor_generic_t accessor,
                          legion_ptr_t ptr, double value);
void reduce_divide_double_domain_point(legion_accessor_generic_t accessor,
                                       legion_domain_point_t, double value);
void reduce_divide_int32(legion_accessor_generic_t accessor,
                         legion_ptr_t ptr, int value);
void reduce_divide_int32_domain_point(legion_accessor_generic_t accessor,
                                      legion_domain_point_t, int value);
void reduce_divide_int64(legion_accessor_generic_t accessor,
                         legion_ptr_t ptr, long long int value);
void reduce_divide_int64_domain_point(legion_accessor_generic_t accessor,
                                      legion_domain_point_t, long long int value);

void reduce_max_float(legion_accessor_generic_t accessor,
                      legion_ptr_t ptr, float value);
void reduce_max_float_domain_point(legion_accessor_generic_t accessor,
                                   legion_domain_point_t, float value);
void reduce_max_double(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, double value);
void reduce_max_double_domain_point(legion_accessor_generic_t accessor,
                                    legion_domain_point_t, double value);
void reduce_max_int32(legion_accessor_generic_t accessor,
                      legion_ptr_t ptr, int value);
void reduce_max_int32_domain_point(legion_accessor_generic_t accessor,
                                   legion_domain_point_t, int value);
void reduce_max_int64(legion_accessor_generic_t accessor,
                      legion_ptr_t ptr, long long int value);
void reduce_max_int64_domain_point(legion_accessor_generic_t accessor,
                                   legion_domain_point_t, long long int value);

void reduce_min_float(legion_accessor_generic_t accessor,
                      legion_ptr_t ptr, float value);
void reduce_min_float_domain_point(legion_accessor_generic_t accessor,
                                   legion_domain_point_t, float value);
void reduce_min_double(legion_accessor_generic_t accessor,
                       legion_ptr_t ptr, double value);
void reduce_min_double_domain_point(legion_accessor_generic_t accessor,
                                    legion_domain_point_t, double value);
void reduce_min_int32(legion_accessor_generic_t accessor,
                      legion_ptr_t ptr, int value);
void reduce_min_int32_domain_point(legion_accessor_generic_t accessor,
                                   legion_domain_point_t, int value);
void reduce_min_int64(legion_accessor_generic_t accessor,
                      legion_ptr_t ptr, long long int value);
void reduce_min_int64_domain_point(legion_accessor_generic_t accessor,
                                   legion_domain_point_t, long long int value);

void safe_reduce_plus_float(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, float value);
void safe_reduce_plus_float_domain_point(legion_accessor_generic_t accessor,
                                         legion_domain_point_t, float value);
void safe_reduce_plus_double(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, double value);
void safe_reduce_plus_double_domain_point(legion_accessor_generic_t accessor,
                                          legion_domain_point_t, double value);
void safe_reduce_plus_int32(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, int value);
void safe_reduce_plus_int32_domain_point(legion_accessor_generic_t accessor,
                                         legion_domain_point_t, int value);
void safe_reduce_plus_int64(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, long long int value);
void safe_reduce_plus_int64_domain_point(legion_accessor_generic_t accessor,
                                         legion_domain_point_t, long long int value);

void safe_reduce_minus_float(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, float value);
void safe_reduce_minus_float_domain_point(legion_accessor_generic_t accessor,
                                          legion_domain_point_t, float value);
void safe_reduce_minus_double(legion_accessor_generic_t accessor,
                              legion_ptr_t ptr, double value);
void safe_reduce_minus_double_domain_point(legion_accessor_generic_t accessor,
                                           legion_domain_point_t, double value);
void safe_reduce_minus_int32(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, int value);
void safe_reduce_minus_int32_domain_point(legion_accessor_generic_t accessor,
                                          legion_domain_point_t, int value);
void safe_reduce_minus_int64(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, long long int value);
void safe_reduce_minus_int64_domain_point(legion_accessor_generic_t accessor,
                                          legion_domain_point_t, long long int value);

void safe_reduce_times_float(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, float value);
void safe_reduce_times_float_domain_point(legion_accessor_generic_t accessor,
                                          legion_domain_point_t, float value);
void safe_reduce_times_double(legion_accessor_generic_t accessor,
                              legion_ptr_t ptr, double value);
void safe_reduce_times_double_domain_point(legion_accessor_generic_t accessor,
                                           legion_domain_point_t, double value);
void safe_reduce_times_int32(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, int value);
void safe_reduce_times_int32_domain_point(legion_accessor_generic_t accessor,
                                          legion_domain_point_t, int value);
void safe_reduce_times_int64(legion_accessor_generic_t accessor,
                             legion_ptr_t ptr, long long int value);
void safe_reduce_times_int64_domain_point(legion_accessor_generic_t accessor,
                                          legion_domain_point_t, long long int value);

void safe_reduce_divide_float(legion_accessor_generic_t accessor,
                              legion_ptr_t ptr, float value);
void safe_reduce_divide_float_domain_point(legion_accessor_generic_t accessor,
                                           legion_domain_point_t, float value);
void safe_reduce_divide_double(legion_accessor_generic_t accessor,
                               legion_ptr_t ptr, double value);
void safe_reduce_divide_double_domain_point(legion_accessor_generic_t accessor,
                                            legion_domain_point_t, double value);
void safe_reduce_divide_int32(legion_accessor_generic_t accessor,
                              legion_ptr_t ptr, int value);
void safe_reduce_divide_int32_domain_point(legion_accessor_generic_t accessor,
                                           legion_domain_point_t, int value);
void safe_reduce_divide_int64(legion_accessor_generic_t accessor,
                              legion_ptr_t ptr, long long int value);
void safe_reduce_divide_int64_domain_point(legion_accessor_generic_t accessor,
                                           legion_domain_point_t, long long int value);

void safe_reduce_max_float(legion_accessor_generic_t accessor,
                           legion_ptr_t ptr, float value);
void safe_reduce_max_float_domain_point(legion_accessor_generic_t accessor,
                                        legion_domain_point_t, float value);
void safe_reduce_max_double(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, double value);
void safe_reduce_max_double_domain_point(legion_accessor_generic_t accessor,
                                         legion_domain_point_t, double value);
void safe_reduce_max_int32(legion_accessor_generic_t accessor,
                           legion_ptr_t ptr, int value);
void safe_reduce_max_int32_domain_point(legion_accessor_generic_t accessor,
                                        legion_domain_point_t, int value);
void safe_reduce_max_int64(legion_accessor_generic_t accessor,
                           legion_ptr_t ptr, long long int value);
void safe_reduce_max_int64_domain_point(legion_accessor_generic_t accessor,
                                        legion_domain_point_t, long long int value);

void safe_reduce_min_float(legion_accessor_generic_t accessor,
                           legion_ptr_t ptr, float value);
void safe_reduce_min_float_domain_point(legion_accessor_generic_t accessor,
                                        legion_domain_point_t, float value);
void safe_reduce_min_double(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, double value);
void safe_reduce_min_double_domain_point(legion_accessor_generic_t accessor,
                                         legion_domain_point_t, double value);
void safe_reduce_min_int32(legion_accessor_generic_t accessor,
                           legion_ptr_t ptr, int value);
void safe_reduce_min_int32_domain_point(legion_accessor_generic_t accessor,
                                        legion_domain_point_t, int value);
void safe_reduce_min_int64(legion_accessor_generic_t accessor,
                           legion_ptr_t ptr, long long int value);
void safe_reduce_min_int64_domain_point(legion_accessor_generic_t accessor,
                                        legion_domain_point_t, long long int value);

#ifdef __cplusplus
}
#endif

#endif // __LEGION_TERRA_H__
