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
#include "realm/python/python_source.h"

using namespace Legion;

TaskID register_python_task_variant(
  TaskID id /* = AUTO_GENERATE_ID */,
  const char *task_name /* = NULL*/,
  const ExecutionConstraintSet &execution_constraints,
  const TaskLayoutConstraintSet &layout_constraints,
  const TaskConfigOptions &options,
  const char *module_name,
  const char *function_name,
  const void *userdata = NULL,
  size_t userlen = 0)
{
  if (id == AUTO_GENERATE_ID)
    id = Runtime::generate_static_task_id();

  TaskVariantRegistrar registrar(id, task_name);
  registrar.set_leaf(options.leaf);
  registrar.set_inner(options.inner);
  registrar.set_idempotent(options.idempotent);
  registrar.layout_constraints = layout_constraints;
  registrar.execution_constraints = execution_constraints;

  CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
  code_desc.add_implementation(new Realm::PythonSourceImplementation(module_name, function_name));

  /*VariantID vid =*/ Runtime::preregister_task_variant(
    registrar, code_desc, userdata, userlen, task_name);
  return id;
}

int main(int argc, char **argv)
{
#ifdef REALM_USE_PYTHON
  // do this before any threads are spawned
  setenv("PYTHONPATH", ".", true /*overwrite*/);
#endif

  ExecutionConstraintSet execution_constraints;
  execution_constraints.add_constraint(ProcessorConstraint(Processor::PY_PROC));
  TaskLayoutConstraintSet layout_constraints;
  TaskConfigOptions options(false /*leaf*/, false /*inner*/, false /*idempotent*/);
  register_python_task_variant(AUTO_GENERATE_ID, "main_task",
                               execution_constraints, layout_constraints,
                               options,
                               "python_interop", "main_task");
}
