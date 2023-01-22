/* Copyright 2023 Stanford University
 * Copyright 2023 Los Alamos National Laboratory 
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

#ifndef TASK_THROUGHPUT_H
#define TASK_THROUGHPUT_H

#include <realm.h>

using namespace Realm;

namespace TestConfig {
  extern int tasks_per_processor;
  extern int launching_processors;
  extern int task_argument_size;
  extern bool remote_tasks;
  extern bool with_profiling;
  extern bool chain_tasks;
  extern bool user_posttrigger_barrier;
};

void dummy_task_body(const void *args, size_t arglen, 
		     const void *userdata, size_t userlen, Processor p);

#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
void dummy_gpu_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p);
#endif

#endif
