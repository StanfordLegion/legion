/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#ifndef LLVMJIT_INTERNAL_H
#define LLVMJIT_INTERNAL_H

#include "realm/bytearray.h"

// instead of including LLVM headers here, we just forward-declare the things that need to
//  appear inside an LLVMJitInternal
namespace llvm {
  class LLVMContext;
  class TargetMachine;
  class ExecutionEngine;
};

namespace Realm {
  namespace LLVMJit {

    class LLVMJitInternal {
    public:
      LLVMJitInternal(void);
      ~LLVMJitInternal(void);

      void *llvmir_to_fnptr(const ByteArray& ir, const std::string& entry_symbol);

    protected:
      llvm::LLVMContext *context;
      llvm::ExecutionEngine *host_exec_engine;
      llvm::TargetMachine *nvptx_machine;
    };

  }; // namespace LLVMJit

}; // namespace Realm

#endif
