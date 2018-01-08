/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#warning This file is deprecated - include "realm/realm_c.h" instead (and rename legion_lowlevel_* to realm_*).

#include "realm/realm_c.h"

typedef realm_id_t legion_lowlevel_id_t;
typedef realm_address_space_t legion_lowlevel_address_space_t;
typedef realm_task_func_id_t legion_lowlevel_task_func_id_t;
typedef realm_reduction_op_id_t legion_lowlevel_reduction_op_id_t;
typedef realm_custom_serdez_id_t legion_lowlevel_custom_serdez_id_t;
typedef realm_event_gen_t legion_lowlevel_event_gen_t;
typedef realm_barrier_timestamp_t legion_lowlevel_barrier_timestamp_t;
typedef realm_processor_kind_t legion_lowlevel_processor_kind_t;
typedef realm_memory_kind_t legion_lowlevel_memory_kind_t;
typedef realm_file_mode_t legion_lowlevel_file_mode_t;
typedef realm_task_pointer_t legion_lowlevel_task_pointer_t;

typedef long long legion_lowlevel_coord_t;

#endif
