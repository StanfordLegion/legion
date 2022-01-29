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

#include "realm/llvmjit/llvmjit_internal.h"

#include "realm/logging.h"

// xcode's clang isn't defining these?
#define __STDC_LIMIT_MACROS
#define __STDC_CONSTANT_MACROS

#include <llvm-c/Core.h>
#include <llvm-c/Initialization.h>
#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/IRReader.h>

#ifdef REALM_ALLOW_MISSING_LLVM_LIBS
// declare all of the LLVM C API calls we use as weak symbols
#pragma weak LLVMAddModule
#pragma weak LLVMContextCreate
#pragma weak LLVMContextDispose
#pragma weak LLVMCreateMCJITCompilerForModule
#pragma weak LLVMCreateMemoryBufferWithMemoryRange
#pragma weak LLVMCreateMemoryBufferWithMemoryRangeCopy
#pragma weak LLVMCreateTargetMachine
#pragma weak LLVMDisposeExecutionEngine
#pragma weak LLVMDisposeMessage
#pragma weak LLVMGetDefaultTargetTriple
#pragma weak LLVMGetNamedFunction
#pragma weak LLVMGetPointerToGlobal
#pragma weak LLVMGetTargetFromTriple
#pragma weak LLVMInitializeMCJITCompilerOptions
#pragma weak LLVMInitializeX86AsmParser
#pragma weak LLVMInitializeX86AsmPrinter
#pragma weak LLVMInitializeX86Target
#pragma weak LLVMInitializeX86TargetInfo
#pragma weak LLVMInitializeX86TargetMC
#pragma weak LLVMLinkInMCJIT
#pragma weak LLVMModuleCreateWithNameInContext
#pragma weak LLVMParseIRInContext
#pragma weak LLVMSetDataLayout
#pragma weak LLVMSetTarget
#endif

namespace Realm {

  extern Logger log_llvmjit;  // defined in llvmjit_module.cc

  namespace LLVMJit {

#ifdef DEBUG_MEMORY_MANAGEMENT
    // TODO: the C API doesn't expose the default memory manager to "inherit"
    //  from, so using this would require a complete implementation
    //  (e.g. mmap, mprotect, ...)
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

#ifdef REALM_ALLOW_MISSING_LLVM_LIBS
    /*static*/ bool LLVMJitInternal::detect_llvm_libraries(void)
    {
      // we have weak symbols for our LLVM references, so see if an
      //  important one is non-null
      void *fnptr = reinterpret_cast<void *>(&LLVMContextCreate);
      return (fnptr != 0);
    }
#endif

    LLVMJitInternal::LLVMJitInternal(void)
    {
      context = LLVMContextCreate();

      // generative native target
      {
	LLVMInitializeNativeTarget();
#ifndef USE_OLD_JIT
	LLVMInitializeNativeAsmParser();
	LLVMInitializeNativeAsmPrinter();
#endif

	char *triple = LLVMGetDefaultTargetTriple();
	log_llvmjit.debug() << "default target triple = " << triple;

	LLVMTargetRef target;
	char *errmsg = 0;
	if(LLVMGetTargetFromTriple(triple, &target, &errmsg)) {
	  log_llvmjit.fatal() << "target not found: triple='" << triple << "': " << errmsg;
	  LLVMDisposeMessage(errmsg);
	  assert(0);
	}

	// TODO - allow configuration options to steer these
	LLVMRelocMode reloc_model = LLVMRelocStatic;
	LLVMCodeModel code_model = LLVMCodeModelLarge;
	LLVMCodeGenOptLevel opt_level = LLVMCodeGenLevelAggressive;

	LLVMTargetMachineRef host_cpu_machine = LLVMCreateTargetMachine(target,
									triple,
									"",
									0/*HostHasAVX()*/ ? "+avx" : "", 
									opt_level,
									reloc_model,
									code_model);
	assert(host_cpu_machine != 0);

	// you have to have a module to build an execution engine, so create
	//  a dummy one
	{
	  LLVMModuleRef m = LLVMModuleCreateWithNameInContext("eebuilder",
							      context);
	  LLVMSetTarget(m, triple);
	  // an empty data layout string causes it to be obtained from the target
	  //  machine, which is nice because they have to match anyway
	  LLVMSetDataLayout(m, "");

#ifdef USE_OLD_JIT
	  char *errmsg = 0;
	  LLVMLinkInJIT();
	  if(LLVMCreateJITCompilerForModule(&host_exec_engine, m,
					    opt_level,
					    &errmsg)) {
            log_llvmjit.fatal() << "failed to create execution engine: " << errmsg;
	    LLVMDisposeMessage(errmsg);
	    assert(0);
	  }
#else
	  struct LLVMMCJITCompilerOptions options;
	  LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
	  options.OptLevel = opt_level;
	  options.CodeModel = code_model;
	  options.NoFramePointerElim = true;
#ifdef DEBUG_MEMORY_MANAGEMENT
	  options.MCJMM = ...;
	  //eb.setMCJITMemoryManager(new MemoryManagerWrap);
#endif
	  
	  char *errmsg = 0;
	  LLVMLinkInMCJIT();
	  if(LLVMCreateMCJITCompilerForModule(&host_exec_engine, m,
					      &options, sizeof(options),
					      &errmsg)) {
	    log_llvmjit.fatal() << "failed to create execution engine: " << errmsg;
	    LLVMDisposeMessage(errmsg);
	    assert(0);
	  }
#endif
	}

	// should be safe to dispose of triple now?
	LLVMDisposeMessage(triple);
      }

      nvptx_machine = 0;
    }

    LLVMJitInternal::~LLVMJitInternal(void)
    {
      LLVMDisposeExecutionEngine(host_exec_engine);
      LLVMContextDispose(context);
    }

    void *LLVMJitInternal::llvmir_to_fnptr(const ByteArray& ir,
					   const std::string& entry_symbol)
    {
      // do we even know how to jit?
      if(!host_exec_engine)
	return 0;

      // may need to manually add null-termination here
      LLVMMemoryBufferRef mb;
      if((ir.size() == 0) || (((const char *)(ir.base()))[ir.size() - 1] != 0)) {
	char *nullterm = new char[ir.size() + 1];
	assert(nullterm != 0);
	memcpy(nullterm, ir.base(), ir.size());
	nullterm[ir.size()] = 0;
	mb = LLVMCreateMemoryBufferWithMemoryRangeCopy(nullterm,
	                                               ir.size()+1,
	                                               "membuf");
	delete[] nullterm;
      } else {
	mb = LLVMCreateMemoryBufferWithMemoryRange((const char *)(ir.base()),
						   ir.size() - 1, // do not count null byte at end
						   "membuf",
						   true /*RequiresTerminator*/);
      }

      char *errmsg = 0;
      LLVMModuleRef m;
      if(LLVMParseIRInContext(context, mb, &m, &errmsg)) {
	// TODO: return this via profiling interface
	log_llvmjit.fatal() << "LLVM IR PARSE ERROR:\n" << errmsg;
	log_llvmjit.fatal() << "IR source=\n" << std::string((const char *)(ir.base()),
	    ir.size());
	LLVMDisposeMessage(errmsg);
	assert(0);
      }

      // get the entry function from the module before we add it to the exec engine - that way
      //  we're sure to get the one we want
      LLVMValueRef func = LLVMGetNamedFunction(m, entry_symbol.c_str());
      if(!func) {
	log_llvmjit.fatal() << "entry symbol not found: " << entry_symbol;
	assert(0);
      }

      LLVMAddModule(host_exec_engine, m);

      // this actually triggers the JIT, allocating space for code and data
      void *fnptr = LLVMGetPointerToGlobal(host_exec_engine, func);
      assert(fnptr != 0);

#ifndef USE_OLD_JIT
      // so the C API doesn't expose finalizeObject, and the finalization in
      //   GetPointerToGlobal is done BEFORE the JIT, so to finalize the JIT'd
      //   thing we appear to have to call it twice...
      LLVMGetPointerToGlobal(host_exec_engine, func);
#endif

      // do NOT dispose the memory buffer - it belongs to the module now
      //LLVMDisposeMemoryBuffer(mb);

      return fnptr;
    }

  }; // namespace LLVMJit

}; // namespace Realm
