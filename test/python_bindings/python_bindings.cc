/* Copyright 2023 Stanford University
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

enum TaskIDs {
  MAIN_TASK_ID = 1,
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

int main(int argc, char **argv)
{
  // do this before any threads are spawned
#ifndef PYTHON_MODULES_PATH
#error PYTHON_MODULES_PATH not available at compile time
#endif
  setenv("PYTHONPATH", PYTHON_MODULES_PATH, true /*overwrite*/);

  Realm::Python::PythonModule::import_python_module("python_bindings");

  {
    TaskVariantRegistrar registrar(MAIN_TASK_ID, "main_task");
    registrar.add_constraint(ProcessorConstraint(Processor::PY_PROC));
    preregister_python_task_variant(registrar, "python_bindings", "main_task");
  }

  Runtime::set_top_level_task_id(MAIN_TASK_ID);

  return Runtime::start(argc, argv);
}
