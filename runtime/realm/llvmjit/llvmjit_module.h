/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#ifndef LLVMJIT_MODULE_H
#define LLVMJIT_MODULE_H

// NOTE: let's try to keep LLVM include files/etc. out of here, because they bring
//  a bunch of C++11 baggage

#include "realm/module.h"
#include "realm/threads.h"

#include <string>

namespace Realm {
  namespace LLVMJit {

    class LLVMJitInternal;

    // our interface to the rest of the runtime
    class LLVMJitModule : public Module {
    protected:
      LLVMJitModule(void);
      
    public:
      virtual ~LLVMJitModule(void);

      static Module *create_module(RuntimeImpl *runtime, std::vector<std::string>& cmdline);

      // do any general initialization - this is called after all configuration is
      //  complete
      virtual void initialize(RuntimeImpl *runtime);

      // create any memories provided by this module (default == do nothing)
      // virtual void create_memories(RuntimeImpl *runtime);

      // create any processors provided by the module (default == do nothing)
      // virtual void create_processors(RuntimeImpl *runtime);

      // create any DMA channels provided by the module (default == do nothing)
      // virtual void create_dma_channels(RuntimeImpl *runtime);

      // create any code translators provided by the module (default == do nothing)
      virtual void create_code_translators(RuntimeImpl *runtime);

      // clean up any common resources created by the module - this will be called
      //  after all memories/processors/etc. have been shut down and destroyed
      virtual void cleanup(void);

    public:

      LLVMJitInternal *internal;
    };

  }; // namespace LLVMJit

}; // namespace Realm

#endif
