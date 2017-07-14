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

#include <stdint.h>
#include <stdio.h>
#include <assert.h> 
#include <unistd.h>

#include "legion_c.h"


// We use an enum to declare the IDs for user-level tasks
enum TaskID {
  TOP_LEVEL_TASK_ID,  
  HELLO_WORLD_TASK_ID,
};

void hello_world_task(const void *data, size_t datalen,
		    const void *userdata, size_t userlen, legion_lowlevel_id_t p)
{
  //  assert(task->arglen == sizeof(int));
//    int rank = *(const int*)task->args; 
    legion_task_t task;
    const legion_physical_region_t *regions;
    unsigned num_regions;
    legion_context_t ctx;
    legion_runtime_t runtime;
    legion_task_preamble(data, datalen, p,
  		       &task,
  		       &regions,
  		       &num_regions,
  		       &ctx,
  		       &runtime);
               
    int *args = (int *)legion_task_get_args(task);
    size_t arglen = legion_task_get_arglen(task);
    printf("Hello from hello_world_task %d, arglen %ld\n", *args, arglen);
    legion_task_postamble(runtime, ctx, NULL, 0);
}


void top_level_task(const void *data, size_t datalen,
		    const void *userdata, size_t userlen, legion_lowlevel_id_t p)
{
    legion_task_t task;
    const legion_physical_region_t *regions;
    unsigned num_regions;
    legion_context_t ctx;
    legion_runtime_t runtime;
    task.impl = NULL;
    ctx.impl = NULL;
    runtime.impl = NULL;
    legion_task_preamble(data, datalen, p,
    	       &task,
    	       &regions,
    	       &num_regions,
    	       &ctx,
    	       &runtime);
    printf("Hello from top_level_task (msg='%.*s')\n",
     (int)userlen, (const char *)userdata);
    //legion_task_postamble(runtime, ctx, 0, 0);
 
     /*
    legion_execution_constraint_set_t execution_constraints = legion_execution_constraint_set_create();
    legion_execution_constraint_set_add_processor_constraint(execution_constraints, LOC_PROC);
    legion_task_layout_constraint_set_t layout_constraints = legion_task_layout_constraint_set_create();
    legion_task_config_options_t config_options = {.leaf = false, .inner = false, .idempotent = false};
    legion_runtime_register_task_variant_fnptr(runtime, HELLO_WORLD_TASK_ID,
                                                "hello_world_task", true, 
                                                execution_constraints,
                                                layout_constraints,
                                                config_options,
                                                hello_world_task,
                                                NULL,
                                                0);*/

    for (int i = 0; i < 10; i++) {
        legion_task_argument_t task_args;
        task_args.args = &i;
        task_args.arglen = sizeof(i);
        legion_task_launcher_t launcher = legion_task_launcher_create(HELLO_WORLD_TASK_ID, task_args, legion_predicate_true(), 0, 0);
        legion_task_launcher_execute(runtime, ctx, launcher);
    }
    legion_task_postamble(runtime, ctx, 0, 0);
}

/*
void top_level_task(legion_task_t task, legion_physical_region_t *regions, unsigned num_regions, 
                    legion_context_t ctx, legion_runtime_t runtime)
{
    printf("Hello from top_level_task\n");
    legion_task_postamble(runtime, ctx, 0, 0);
}*/

// We have a main function just like a standard C++ program.
// Once we start the runtime, it will begin running the top-level task.
int main(int argc, char **argv)
{
    legion_runtime_set_top_level_task_id(TOP_LEVEL_TASK_ID);
    
    legion_execution_constraint_set_t execution_constraints = legion_execution_constraint_set_create();
    legion_execution_constraint_set_add_processor_constraint(execution_constraints, LOC_PROC);
    legion_task_layout_constraint_set_t layout_constraints = legion_task_layout_constraint_set_create();
    legion_task_config_options_t config_options = {.leaf = false, .inner = false, .idempotent = false};
    legion_runtime_preregister_task_variant_fnptr(TOP_LEVEL_TASK_ID,
                                                  "top_leve_task",
                                                  execution_constraints,
                                                  layout_constraints,
                                                  config_options,
                                                  top_level_task,
                                                  NULL,
                                                  0);
                                                  
    legion_runtime_preregister_task_variant_fnptr(HELLO_WORLD_TASK_ID,
                                                  "hello_world_task",
                                                  execution_constraints,
                                                  layout_constraints,
                                                  config_options,
                                                  hello_world_task,
                                                  NULL,
                                                  0);
                                                  

    legion_runtime_start(argc, argv, false);
}
