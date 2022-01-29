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

#include "legion_interop.h"

#include "legion.h"

using namespace Legion;

void task_f(const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx, Runtime *runtime)
{
  printf("Hello Legion!\n");
}

void register_tasks()
{
  {
    TaskVariantRegistrar registrar(TID_F, "f");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<task_f>(registrar, "f");
  }
}
