/* Copyright 2022 Stanford University
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
#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

///
/// Mapper
///

class EmbedMapper : public DefaultMapper
{
public:
  EmbedMapper(MapperRuntime *rt, Machine machine, Processor local,
              const char *mapper_name);
  virtual void select_task_options(const MapperContext    ctx,
                                   const Task&            task,
                                         TaskOptions&     output);
};

EmbedMapper::EmbedMapper(MapperRuntime *rt, Machine machine, Processor local,
                         const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
}

void EmbedMapper::select_task_options(const MapperContext    ctx,
                                      const Task&            task,
                                            TaskOptions&     output)
{
  DefaultMapper::select_task_options(ctx, task, output);
  if (strcmp(task.get_task_name(), "inline_regent_task") == 0)
    output.inline_task = true;
}

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
  LogicalRegion other_region = runtime->create_logical_region(ctx, ispace, fspace);

  runtime->fill_field<int>(ctx, region, region, FID_X, 0);
  runtime->fill_field<int>(ctx, region, region, FID_Y, 0);
  runtime->fill_field<int>(ctx, region, region, FID_Z, 0);

  runtime->fill_field<int>(ctx, other_region, other_region, FID_X, 0);
  runtime->fill_field<int>(ctx, other_region, other_region, FID_Y, 0);
  runtime->fill_field<int>(ctx, other_region, other_region, FID_Z, 0);

  legion_field_id_t c_fields[3] = {FID_X, FID_Y, FID_Z};
  std::vector<FieldID> fields(c_fields, c_fields+3);

  // Call Regent task via its C++ API

  {
    my_regent_task_launcher launcher;
    launcher.add_argument_r(region, region, fields);
    launcher.add_argument_x(12345);
    launcher.add_argument_y(3.14);
    launcher.add_argument_z(true);
    float w[4] = {0.1, 0.2, 0.3, 0.4};
    launcher.add_argument_w(w);
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
    my_regent_task_launcher_add_argument_y(launcher, 4.56, false);
    my_regent_task_launcher_add_argument_z(launcher, false, false);
    float w[4] = {1.1, 1.2, 1.3, 1.4};
    my_regent_task_launcher_add_argument_w(launcher, w, false);
    legion_future_t f = my_regent_task_launcher_execute(c_runtime, c_context, launcher);
    my_regent_task_launcher_destroy(launcher);
    legion_future_destroy(f);
  }


  // Arguments can be passed in any order

  {
    other_regent_task_launcher launcher;
    launcher.add_argument_r(region, region, fields);
    launcher.add_argument_s(other_region, other_region, fields);
    launcher.execute(runtime, ctx);
  }

  {
    inline_regent_task_launcher launcher;
    launcher.set_enable_inlining(true);
    launcher.add_argument_r(region, region, fields);
    launcher.execute(runtime, ctx);
  }

  {
    other_regent_task_launcher launcher;
    launcher.add_argument_s(other_region, other_region, fields);
    launcher.add_argument_r(region, region, fields);
    launcher.execute(runtime, ctx);
  }
}

static void create_mappers(Machine machine,
                           Runtime *runtime,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    EmbedMapper* mapper = new EmbedMapper(runtime->get_mapper_runtime(),
                                          machine, *it, "circuit_mapper");
    runtime->replace_default_mapper(mapper, *it);
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

  embed_tasks_register();
  Runtime::add_registration_callback(create_mappers);

  return Runtime::start(argc, argv);
}
