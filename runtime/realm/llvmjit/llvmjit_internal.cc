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

#include "realm/logging.h"

#include <iostream>

// xcode's clang isn't defining these?
#define __STDC_LIMIT_MACROS
#define __STDC_CONSTANT_MACROS

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

  extern Logger log_llvmjit;  // defined in llvmjit_module.cc

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

	std::string triple = llvm::sys::getDefaultTargetTriple();
	log_llvmjit.debug() << "default target triple = " << triple;

	std::string err;
	const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, err);
	if(!target) {
	  log_llvmjit.fatal() << "target not found: triple='" << triple << "'";
	  assert(0);
	}

	// TODO - allow configuration options to steer these
	llvm::Reloc::Model reloc_model = llvm::Reloc::Default;
	llvm::CodeModel::Model code_model = llvm::CodeModel::Large;
	llvm::CodeGenOpt::Level opt_level = llvm::CodeGenOpt::Aggressive;

	llvm::TargetOptions options;
	options.NoFramePointerElim = true;

	llvm::TargetMachine *host_cpu_machine = target->createTargetMachine(triple, "", 
									    0/*HostHasAVX()*/ ? "+avx" : "", 
									    options,
									    reloc_model,
									    code_model,
									    opt_level);
	assert(host_cpu_machine != 0);

	// you have to have a module to build an execution engine, so create
	//  a dummy one
	{
	  llvm::Module *m = new llvm::Module("eebuilder", *context);
	  m->setTargetTriple(triple);
	  m->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
	  llvm::EngineBuilder eb(m);

	  std::string err;
  
	  eb
	    .setErrorStr(&err)
	    .setEngineKind(llvm::EngineKind::JIT)
	    .setAllocateGVsWithCode(false)
	    .setUseMCJIT(true);
	  
	  host_exec_engine = eb.create(host_cpu_machine);

	  if(!host_exec_engine) {
	    log_llvmjit.fatal() << "failed to create execution engine: " << err;
	    assert(0);
	  }
	}
      }

      nvptx_machine = 0;
    }

    LLVMJitInternal::~LLVMJitInternal(void)
    {
      delete host_exec_engine;
      delete context;
    }
    
    void *LLVMJitInternal::llvmir_to_fnptr(const ByteArray& ir,
					   const std::string& entry_symbol)
    {
      // do we even know how to jit?
      if(!host_exec_engine)
	return 0;

      llvm::SMDiagnostic sm;
      llvm::MemoryBuffer *mb = llvm::MemoryBuffer::getMemBuffer((const char *)ir.base());
      llvm::Module *m = llvm::ParseIR(mb, sm, *context);
      if(!m) {
	std::string errstr;
	llvm::raw_string_ostream s(errstr);
	sm.print(entry_symbol.c_str(), s);
	log_llvmjit.fatal() << "LLVM IR PARSE ERROR:\n" << s.str();
	assert(0);
      }

      // get the entry function from the module before we add it to the exec engine - that way
      //  we're sure to get the one we want
      llvm::Function *func = m->getFunction(entry_symbol.c_str());
      if(!func) {
	log_llvmjit.fatal() << "entry symbol not found: " << entry_symbol;
	assert(0);
      }

      host_exec_engine->addModule(m);

      void *fnptr = host_exec_engine->getPointerToFunction(func);
      assert(fnptr != 0);

      return fnptr;
    }

  }; // namespace LLVMJit

}; // namespace Realm
