/* Copyright 2026 Stanford University
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


#include <cstdio>

#include "legion.h"

using namespace Legion;

enum TaskID {
  TOP_LEVEL_TASK_ID,
};

enum HandlerID {
  SUPPRESSOR_ID = 1,
  REWRITER_ID = 2,
};

// All exception handlers must inherit from the exception handler class
class WarningSuppressor : public ExceptionHandler {
public:
  // This exception handler can only handle warning
  virtual bool can_handle(ExceptionType type) const override
  { return (type == LEGION_WARNING_EXCEPTION); } 
  // To suppress all warnings, return true indicating that we handled
  // the warning and Legion no longer needs to report it
  virtual bool handle_exception(Exception& exception,
      const std::string_view& provenance, const Realm::Backtrace& bt) override
  { 
    printf("Suppressing warning\n"); 
    return true; 
  }
};

class WarningRewriter : public ExceptionHandler {
public:
  // This exception handler can only handle warning
  virtual bool can_handle(ExceptionType type) const override
  { return (type == LEGION_WARNING_EXCEPTION); }
  virtual bool handle_exception(Exception& exception,
      const std::string_view& provenance, const Realm::Backtrace& bt) override
  { 
    printf("Rewriting warning\n");
    exception.clear();  
    std::ostream stream(&exception);
    stream << "Here is a custom warning message from the warning rewriter";
    // We just rewrote the warning, we didn't actually handle it
    return false; 
  }
};

void top_level_task(const Task *task, 
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  // Making a deferred buffer in a leaf task without a memory pool will raise a warning
  DeferredValue<unsigned> one(0, alignof(unsigned), Memory::SYSTEM_MEM);
  // Push the exception handler on to the stack for this task's context
  runtime->push_exception_handler(ctx, SUPPRESSOR_ID);
  // Make a deferred buffer inline, normally this should trigger a warning
  // but in this case it will be caught and handled by the exception handler
  DeferredValue<unsigned> two(0, alignof(unsigned), Memory::SYSTEM_MEM);
  // Push another exception handler onto the stack
  runtime->push_exception_handler(ctx, REWRITER_ID);
  DeferredValue<unsigned> three(0, alignof(unsigned), Memory::SYSTEM_MEM);
  // You can pop exception handlers too, although it is optional
  runtime->pop_exception_handler(ctx);
  runtime->pop_exception_handler(ctx);
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
  }
  Runtime::preregister_exception_handler(SUPPRESSOR_ID, new WarningSuppressor);
  Runtime::preregister_exception_handler(REWRITER_ID, new WarningRewriter);
  return Runtime::start(argc, argv);
}
