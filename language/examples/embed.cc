/* Copyright 2019 Stanford University
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

#include "embed.h"
#include "embed_tasks.h"

#include "legion.h"
#include "legion/legion_c_util.h"

using namespace Legion;

enum {
  TID_TOP_LEVEL_TASK,
};

enum {
  FID_X,
  FID_Y,
  FID_Z,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  printf("Hello from C++\n");

  // Create a dummy region and fill it

  IndexSpace ispace = runtime->create_index_space(ctx, Domain(Rect<1>(0, 9)));
  FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator falloc = runtime->create_field_allocator(ctx, fspace);
    falloc.allocate_field(sizeof(int), FID_X);
    falloc.allocate_field(sizeof(int), FID_Y);
    falloc.allocate_field(sizeof(int), FID_Z);
  }
  LogicalRegion region = runtime->create_logical_region(ctx, ispace, fspace);

  runtime->fill_field<int>(ctx, region, region, FID_X, 0);
  runtime->fill_field<int>(ctx, region, region, FID_Y, 0);
  runtime->fill_field<int>(ctx, region, region, FID_Z, 0);

  legion_field_id_t c_fields[3] = {FID_X, FID_Y, FID_Z};
  std::vector<FieldID> fields(c_fields, c_fields+3);

  // Call Regent task via its C++ API

  {
    my_regent_task_launcher launcher;
    launcher.add_argument_r(region, region, fields);
    launcher.add_argument_x(12345);
    launcher.execute(runtime, ctx);
  }

  // Same via C API

  {
    legion_runtime_t c_runtime = CObjectWrapper::wrap(runtime);
    CContext c_ctx(ctx);
    legion_context_t c_context = CObjectWrapper::wrap(&c_ctx);
    legion_logical_region_t c_region = CObjectWrapper::wrap(region);

    my_regent_task_launcher_t launcher = my_regent_task_launcher_create(legion_predicate_true(), 0, 0);
    my_regent_task_launcher_add_argument_r(launcher, c_region, c_region, c_fields, 3, false);
    my_regent_task_launcher_add_argument_x(launcher, 67890, false);
    legion_future_t f = my_regent_task_launcher_execute(c_runtime, c_context, launcher);
    my_regent_task_launcher_destroy(launcher);
    legion_future_destroy(f);
  }
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TID_TOP_LEVEL_TASK);

  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL_TASK, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  embed_tasks_h_register();

  return Runtime::start(argc, argv);
}
