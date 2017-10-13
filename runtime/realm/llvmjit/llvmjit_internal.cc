/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#define LLVM_VERSION (10 * LLVM_VERSION_MAJOR) + LLVM_VERSION_MINOR
#if REALM_LLVM_VERSION != LLVM_VERSION
  #error mismatch between REALM_LLVM_VERSION and LLVM header files!
#endif
// JIT for 3.5, MCJIT for 3.8
#if LLVM_VERSION == 35
  #define USE_JIT
  #define USE_NO_FRAME_POINTER_ELIM
  #include <llvm/ExecutionEngine/JIT.h>
  #include "llvm/PassManager.h"
#elif LLVM_VERSION == 38
  #define USE_UNIQUE_PTRS
  #define USE_MCJIT
  #include <memory>
  #include <llvm/ExecutionEngine/MCJIT.h>
#else
  #error unsupported (or at least untested) LLVM version!
#endif

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/FormattedStream.h"

#ifdef DEBUG_MEMORY_MANAGEMENT
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include <iostream>
#endif

namespace Realm {

  extern Logger log_llvmjit;  // defined in llvmjit_module.cc

  namespace LLVMJit {

#ifdef DEBUG_MEMORY_MANAGEMENT
    class MemoryManagerWrap : public llvm::SectionMemoryManager {
    public:
      virtual uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment, unsigned SectionID, llvm::StringRef SectionName) override
      {
	std::cout << "allocateCodeSection(" << Size << ", " << Alignment << ", " << SectionID << ", " << SectionName.str() << ")";
	std::cout.flush();
	uint8_t *ret = llvm::SectionMemoryManager::allocateCodeSection(Size, Alignment, SectionID, SectionName);
	std::cout << " -> " << ((void *)ret) << "\n";
	return ret;
      }

      virtual uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment, unsigned SectionID, llvm::StringRef SectionName, bool isReadOnly) override
      {
	std::cout << "allocateDataSection(" << Size << ", " << Alignment << ", " << SectionID << ", " << SectionName.str() << ", " << isReadOnly << ")";
	std::cout.flush();
	uint8_t *ret = llvm::SectionMemoryManager::allocateDataSection(Size, Alignment, SectionID, SectionName, isReadOnly);
	std::cout << " -> " << ((void *)ret) << "\n";
	return ret;
      }

      bool finalizeMemory (std::string *ErrMsg=nullptr) override
      {
	std::cout << "finalizeMemory()";
	std::cout.flush();
	bool ret = llvm::SectionMemoryManager::finalizeMemory(ErrMsg);
	std::cout << " -> " << ret << "\n";
	return ret;
      }

      void invalidateInstructionCache() override
      {
	std::cout << "invalidateInstructionCache()\n";
	llvm::SectionMemoryManager::invalidateInstructionCache();
      }
    };
#endif

    ////////////////////////////////////////////////////////////////////////
    //
    // class LLVMJitInternal

    LLVMJitInternal::LLVMJitInternal(void)
    {
      context = new llvm::LLVMContext;

      // generative native target
      {
	llvm::InitializeNativeTarget();
#if LLVM_VERSION >= 38
	llvm::InitializeNativeTargetAsmParser();
	llvm::InitializeNativeTargetAsmPrinter();
#endif

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
#ifdef NO_FRAME_POINTER_ELIM
	options.NoFramePointerElim = true;
#endif

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
#ifdef USE_UNIQUE_PTRS
	  // the extra parens matter here for some reason...
	  llvm::EngineBuilder eb((std::unique_ptr<llvm::Module>(m)));
#else
	  llvm::EngineBuilder eb(m);
#endif

	  std::string err;
  
	  eb
	    .setErrorStr(&err)
	    .setEngineKind(llvm::EngineKind::JIT)
#ifdef USE_JIT
	    .setAllocateGVsWithCode(false)
	    .setUseMCJIT(true)
#endif
	    ;
#ifdef DEBUG_MEMORY_MANAGEMENT
	  eb.setMCJITMemoryManager(new MemoryManagerWrap);
#endif
	  
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
      // LLVM requires that the data be null-terminated (even for bitcode?)
      //  so make a copy and add one extra byte
      char *nullterm = new char[ir.size() + 1];
      memcpy(nullterm, ir.base(), ir.size());
      nullterm[ir.size()] = 0;
#ifdef USE_UNIQUE_PTRS
      llvm::MemoryBuffer *mb = llvm::MemoryBuffer::getMemBuffer(nullterm).release();
#else
      llvm::MemoryBuffer *mb = llvm::MemoryBuffer::getMemBuffer(nullterm);
#endif
      // TODO: when did this spelling actually change?
#if LLVM_VERSION >= 38
      llvm::Module *m = llvm::parseIR(llvm::MemoryBufferRef(*mb), sm, *context).release();
#else
      llvm::Module *m = llvm::ParseIR(mb, sm, *context);
#endif
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

#ifdef USE_UNIQUE_PTRS
      host_exec_engine->addModule(std::unique_ptr<llvm::Module>(m));
#else
      host_exec_engine->addModule(m);
#endif

      // this actually triggers the JIT, allocating space for code and data
      void *fnptr = host_exec_engine->getPointerToFunction(func);
      assert(fnptr != 0);

      // and this call actually marks that memory executable
      host_exec_engine->finalizeObject();

      // hopefully it's ok to delete the IR source buffer now...
      delete[] nullterm;

      return fnptr;
    }

  }; // namespace LLVMJit

}; // namespace Realm
