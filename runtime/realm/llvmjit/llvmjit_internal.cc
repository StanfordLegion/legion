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

#include "llvmjit_internal.h"

#include <iostream>
#include <cstdint>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/ExecutionEngine/JIT.h>
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/PassManager.h"

namespace Realm {

  namespace LLVMJit {

    ////////////////////////////////////////////////////////////////////////
    //
    // class LLVMJitInternal

    LLVMJitInternal::LLVMJitInternal(void)
    {
      context = new llvm::LLVMContext;

      // generative native target
      {
	llvm::InitializeNativeTarget();

	llvm::TargetOptions options;
	options.NoFramePointerElim = true;

	llvm::CodeGenOpt::Level OptLevel = llvm::CodeGenOpt::Aggressive;
	std::string Triple = llvm::sys::getDefaultTargetTriple();
	std::cout << Triple;
	printf("triple = '%s'\n", Triple.c_str());
	std::string err;
	const llvm::Target *TheTarget = llvm::TargetRegistry::lookupTarget(Triple, err);
	if(TheTarget) {
	  host_cpu_machine = TheTarget->createTargetMachine(Triple, "", 
							    0/*HostHasAVX()*/ ? "+avx" : "", 
							    options,
							    llvm::Reloc::Default,
							    llvm::CodeModel::Large,//Default,
							    OptLevel);
	  assert(host_cpu_machine != 0);
	} else {
	  std::cerr << "couldn't find target '" << Triple << "': " << err;
	  host_cpu_machine = 0;
	}
      }

      nvptx_machine = 0;
    }

    LLVMJitInternal::~LLVMJitInternal(void)
    {
      delete host_cpu_machine;
      delete context;
    }
    
    void *LLVMJitInternal::llvmir_to_fnptr(const ByteArray& ir,
					   const std::string& entry_symbol)
    {
      // do we even know how to jit?
      if(!host_cpu_machine)
	return 0;

      llvm::SMDiagnostic sm;
      llvm::MemoryBuffer *mb = llvm::MemoryBuffer::getMemBuffer((const char *)ir.base());
      llvm::Module *m = llvm::ParseIR(mb, sm, *context);
      if(!m) {
	llvm::raw_os_ostream oo(std::cout);
	std::cout << "LLVM IR PARSE ERROR:\n";
	sm.print("foo", oo);
	assert(1 != 2);
      }
      m->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
      assert(host_cpu_machine != 0);

      llvm::EngineBuilder eb(m);

      std::string err;
  
      eb.setErrorStr(&err)
	.setEngineKind(llvm::EngineKind::JIT)
	.setAllocateGVsWithCode(false)
	.setUseMCJIT(true);
      // relocation and code models are controlled by arguments to createTargetMachine() above
 
      llvm::ExecutionEngine *ee = eb.create(host_cpu_machine);

      llvm::Function* func = ee->FindFunctionNamed(entry_symbol.c_str());
      //printf("func = %p\n", func);
      
      void *fnptr = ee->getPointerToFunction(func);
      //printf("fnptr = %p\n", fnptr);
      //((void(*)())fnptr)();
      //printf("here\n");

      // TODO: should (or even can) we delete the execution engine?
      //  or maybe can we have it not tied to a particular module?
      return fnptr;
    }

  }; // namespace LLVMJit

}; // namespace Realm
