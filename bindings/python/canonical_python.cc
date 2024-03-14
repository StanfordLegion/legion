/* Copyright 2024 Stanford University
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

#include "canonical_python.h"
#include "legion/legion_c_util.h"

#include <libgen.h>

using namespace Legion;

static Legion::Context canonical_top_level_ctx;

void legion_canonical_python_begin_top_level_task(int argc, char **argv)
{
  bool control_replicate = true;
  const char * const unique_name = "legion_canonical_python";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-ll:py")) {
      std::cerr << "-ll:py is not supported when using canonical python"
                << std::endl;
      abort();
    }
    if (!strcmp(argv[i], "--nocr")) {
      control_replicate = false;
      printf("Disable Control Replication\n");
    }
  }

  Runtime::start(argc, argv, true /*background*/);
  assert (Runtime::has_context() == false);
  Runtime *runtime = Runtime::get_runtime();
  const TaskID top_level_task_id = runtime->generate_library_task_ids(unique_name, 3); 
  canonical_top_level_ctx = runtime->begin_implicit_task(top_level_task_id,
                                     0 /*mapper id*/,
                                     Processor::LOC_PROC,
                                     "legion_python_top_level_task",
                                     control_replicate /*control replicable*/);
  assert(Runtime::has_context());
}

void legion_canonical_python_end_top_level_task(void) 
{
  Runtime *runtime = Runtime::get_runtime();
  assert(Runtime::has_context());
  runtime->finish_implicit_task(canonical_top_level_ctx);
  // The previous call is asynchronous so we still need to
  // wait for the shutdown of the runtime to complete
  Runtime::wait_for_shutdown();
}

