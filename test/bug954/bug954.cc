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

#include "legion.h"

using namespace Legion;

enum TaskIDs {
  TID_TOP_LEVEL,
  TID_TEST,
};

enum FieldIDs {
  FID_MASK = 10,
  FID_DATA = 11,
};

enum LayoutConstraintIDs {
  LCID_ON_REGMEM = 22,
};

void task_test(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
}

void task_top_level(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  Rect<1> bounds(0, 99);
  IndexSpace is = runtime->create_index_space(ctx, bounds);
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator fsa = runtime->create_field_allocator(ctx, fs);
  fsa.allocate_field(sizeof(bool), FID_MASK);
  fsa.allocate_field(sizeof(int), FID_DATA);
  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
  runtime->fill_field(ctx, lr, lr, FID_MASK, true);
  runtime->fill_field(ctx, lr, lr, FID_DATA, 0);
  { TaskLauncher launcher(TID_TEST, TaskArgument());
    launcher.add_region_requirement(
      RegionRequirement(lr, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr)
      .add_field(FID_MASK));
    launcher.add_region_requirement(
      RegionRequirement(lr, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr)
      .add_field(FID_DATA));
    runtime->execute_task(ctx, launcher);
  }
  runtime->destroy_logical_region(ctx, lr);
}

int main(int argc, char **argv)
{
  { LayoutConstraintRegistrar registrar(FieldSpace::NO_SPACE);
    registrar.add_constraint(MemoryConstraint(Memory::REGDMA_MEM));
    Runtime::preregister_layout(registrar, LCID_ON_REGMEM);
  }
  { TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<task_top_level>(registrar, "top_level");
    Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  }
  { TaskVariantRegistrar registrar(TID_TEST, "test");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(1, LCID_ON_REGMEM);
    Runtime::preregister_task_variant<task_test>(registrar, "test");
  }
  return Runtime::start(argc, argv);
}
