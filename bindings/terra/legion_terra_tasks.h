/* Copyright 2016 Stanford University
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

#ifndef __LEGION_TERRA_TASKS_H__
#define __LEGION_TERRA_TASKS_H__

#include "legion_c.h"

#ifdef __cplusplus
extern "C" {
#endif

void set_lua_registration_callback_name(char*);

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


#ifdef __cplusplus
}
#endif

#endif // __LEGION_TERRA_TASKS_H__
