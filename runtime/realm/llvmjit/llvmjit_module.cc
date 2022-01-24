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

#include "realm/llvmjit/llvmjit.h"
#include "realm/llvmjit/llvmjit_module.h"
#include "realm/llvmjit/llvmjit_internal.h"

#include "realm/runtime_impl.h"
#include "realm/logging.h"

namespace Realm {

  Logger log_llvmjit("llvmjit");

  namespace LLVMJit {

#ifdef REALM_ALLOW_MISSING_LLVM_LIBS
    /*extern*/ bool llvmjit_available = false;
#endif

    ////////////////////////////////////////////////////////////////////////
    //
    // class LLVMCodeTranslator

    class LLVMCodeTranslator : public CodeTranslator {
    public:
      LLVMCodeTranslator(LLVMJitModule *_module);

      virtual ~LLVMCodeTranslator(void);

      virtual bool can_translate(const std::type_info& source_impl_type,
				 const std::type_info& target_impl_type);

      virtual CodeImplementation *translate(const CodeImplementation *source,
					    const std::type_info& target_impl_type);

      // C++ considers the above a "partial override" and wants these defined too
      virtual bool can_translate(const CodeDescriptor& source_codedesc,
				 const std::type_info& target_impl_type);

      virtual CodeImplementation *translate(const CodeDescriptor& source_codedesc,
					    const std::type_info& target_impl_type);

    protected:
      LLVMJitModule *module;
    };

    LLVMCodeTranslator::LLVMCodeTranslator(LLVMJitModule *_module)
      : CodeTranslator("llvmjit")
      , module(_module)
    {}

    LLVMCodeTranslator::~LLVMCodeTranslator(void)
    {}

    bool LLVMCodeTranslator::can_translate(const std::type_info& source_impl_type,
					   const std::type_info& target_impl_type)
    {
      // LLVM IR -> function pointer
      if((source_impl_type == typeid(LLVMIRImplementation)) &&
	 (target_impl_type == typeid(FunctionPointerImplementation)))
	return true;

      return false;
    }

    CodeImplementation *LLVMCodeTranslator::translate(const CodeImplementation *source,
						      const std::type_info& target_impl_type)
    {
      if(target_impl_type == typeid(FunctionPointerImplementation)) {
	const LLVMIRImplementation *llvmir = dynamic_cast<const LLVMIRImplementation *>(source);
	assert(llvmir != 0);

	void *fnptr = module->internal->llvmir_to_fnptr(llvmir->ir,
							llvmir->entry_symbol);
	return new FunctionPointerImplementation((void(*)())fnptr);
      }

      assert(0);
    }

    // these pass through to CodeTranslator's definitions
    bool LLVMCodeTranslator::can_translate(const CodeDescriptor& source_codedesc,
					   const std::type_info& target_impl_type)
    {
      return CodeTranslator::can_translate(source_codedesc, target_impl_type);
    }

    CodeImplementation *LLVMCodeTranslator::translate(const CodeDescriptor& source_codedesc,
						      const std::type_info& target_impl_type)
    {
      return CodeTranslator::translate(source_codedesc, target_impl_type);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class LLVMJitModule

    LLVMJitModule::LLVMJitModule(void)
      : Module("llvmjit")
      , internal(0)
    {}
      
    LLVMJitModule::~LLVMJitModule(void)
    {}

    /*static*/ Module *LLVMJitModule::create_module(RuntimeImpl *runtime,
						    std::vector<std::string>& cmdline)
    {
#ifdef REALM_ALLOW_MISSING_LLVM_LIBS
      if(LLVMJitInternal::detect_llvm_libraries()) {
	llvmjit_available = true;
      } else {
	log_llvmjit.info() << "LLVM libs not found - disabling JIT functionality";
	return 0;
      }
#endif
      LLVMJitModule *m = new LLVMJitModule;
      return m;
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void LLVMJitModule::initialize(RuntimeImpl *runtime)
    {
      Module::initialize(runtime);

      internal = new LLVMJitInternal;
    }

    // create any code translators provided by the module (default == do nothing)
    void LLVMJitModule::create_code_translators(RuntimeImpl *runtime)
    {
      Module::create_code_translators(runtime);

      runtime->add_code_translator(new LLVMCodeTranslator(this));
    }

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void LLVMJitModule::cleanup(void)
    {
      delete internal;

      Module::cleanup();
    }

  }; // namespace LLVMJit


  ////////////////////////////////////////////////////////////////////////
  //
  // class LLVMIRImplementation

  /*static*/ Serialization::PolymorphicSerdezSubclass<CodeImplementation,
						      LLVMIRImplementation> LLVMIRImplementation::serdez_subclass;

  LLVMIRImplementation::LLVMIRImplementation(void)
  {}

  LLVMIRImplementation::LLVMIRImplementation(const void *irdata, size_t irlen, const std::string& _entry_symbol)
    : ir(irdata, irlen)
    , entry_symbol(_entry_symbol)
  {}
  
  LLVMIRImplementation::~LLVMIRImplementation(void)
  {}

  CodeImplementation *LLVMIRImplementation::clone(void) const
  {
    LLVMIRImplementation *i = new LLVMIRImplementation;
    i->ir = ir;
    i->entry_symbol = entry_symbol;
    return i;
  }

  bool LLVMIRImplementation::is_portable(void) const
  {
    return true;
  }

  void LLVMIRImplementation::print(std::ostream& os) const
  {
    os << "LLVMIR(" << entry_symbol << "," << ir.size() << " bytes)";
  }

#if 0
    template <typename S>
    bool LLVMIRImplementation::serialize(S& serializer) const;

    template <typename S>
    static CodeImplementation *LLVMIRImplementation::deserialize_new(S& deserializer);
#endif

}; // namespace Realm
