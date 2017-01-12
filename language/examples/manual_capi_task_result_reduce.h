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

#ifndef __MANUAL_CAPI_TASK_RESULT_REDUCE_H__
#define __MANUAL_CAPI_TASK_RESULT_REDUCE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "legion_c.h"

// struct for types
typedef struct { int    value[1 ]; }    int_1 ;

// register plus on scalars
void register_reduction_global_plus_int32(legion_reduction_op_id_t redop);

#ifdef __cplusplus
}
#endif

#endif // __MANUAL_CAPI_TASK_RESULT_REDUCE_H__
