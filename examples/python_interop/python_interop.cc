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

#include "legion.h"
#include "realm/python/python_module.h"
#include "realm/python/python_source.h"

using namespace Legion;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  MAIN_TASK_ID = 2,
  INIT_TASK_ID = 3,
};

enum FieldIDs {
  X_FIELD_ID = 1,
};

VariantID preregister_python_task_variant(
  const TaskVariantRegistrar &registrar,
  const char *module_name,
  const char *function_name,
  const void *userdata = NULL,
  size_t userlen = 0)
{
  CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
  code_desc.add_implementation(new Realm::PythonSourceImplementation(module_name, function_name));

  return Runtime::preregister_task_variant(
    registrar, code_desc, userdata, userlen,
    registrar.task_variant_name);
}

void init_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
  RegionAccessor<AccessorType::Generic, double> acc = 
    regions[0].get_field_accessor(X_FIELD_ID).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  // Fill memory with some recognizable pattern.
  for (GenericPointInRectIterator<2> pir(rect); pir; pir++) {
    double value = (double)(pir.p[0]*(rect.hi.x[1] - rect.lo.x[1] + 1) + pir.p[1]);
    acc.write(DomainPoint::from_point<2>(pir.p), value);
  }
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  TaskLauncher launcher(MAIN_TASK_ID, TaskArgument());
  runtime->execute_task(ctx, launcher);
}

int main(int argc, char **argv)
{
  // do this before any threads are spawned
#ifndef PYTHON_MODULES_PATH
#error PYTHON_MODULES_PATH not available at compile time
#endif
  setenv("PYTHONPATH", PYTHON_MODULES_PATH, true /*overwrite*/);

  Realm::Python::PythonModule::import_python_module("python_interop");

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
  }

  {
    TaskVariantRegistrar registrar(INIT_TASK_ID, "init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<init_task>(registrar, "init_task");
  }

  {
    TaskVariantRegistrar registrar(MAIN_TASK_ID, "main_task");
    registrar.add_constraint(ProcessorConstraint(Processor::PY_PROC));
    preregister_python_task_variant(registrar, "python_interop", "main_task");
  }

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  return Runtime::start(argc, argv);
}
