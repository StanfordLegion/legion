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

// constructs for describing code blobs to Realm

#include "realm/codedesc.h"

#ifdef REALM_USE_DLFCN
#include <dlfcn.h>
#endif

#include "realm/logging.h"
#include "realm/utils.h"

namespace Realm {

  Logger log_codetrans("codetrans");


  ////////////////////////////////////////////////////////////////////////
  //
  // class Type

  std::ostream& operator<<(std::ostream& os, const Type& t)
  {
    switch(t.f_common.kind) {
    case Type::InvalidKind: os << "INVALIDTYPE"; break;
    case Type::OpaqueKind:
      {
	if(t.size_bits() == 0)
	  os << "void";
	else
	  os << "opaque(" << t.size_bits() << ")";
	break;
      }
    case Type::IntegerKind:
      {
	os << (t.f_integer.is_signed ? 's' : 'u') << "int(" << t.size_bits() << ")";
	break;
      }
    case Type::FloatingPointKind: os << "float(" << t.size_bits() << ")"; break;
    case Type::PointerKind:
      {
	os << *t.f_pointer.base_type;
	if(t.f_pointer.is_const) os << " const";
	os << " *";
	break;
      }
    case Type::FunctionPointerKind:
      {
	os << *t.f_funcptr.return_type << "(*)(";
	const std::vector<Type>& p = *t.f_funcptr.param_types;
	if(p.size()) {
	  for(size_t i = 0; i < p.size(); i++) {
	    if(i) os << ", ";
	    os << p[i];
	  }
	} else
	  os << "void";
	os << ")";
	break;
      }
    }
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CodeDescriptor

  CodeDescriptor::CodeDescriptor(void)
  {}

  CodeDescriptor::CodeDescriptor(const Type& _t)
    : m_type(_t)
  {}

  CodeDescriptor::CodeDescriptor(const CodeDescriptor& rhs)
  {
    copy_from(rhs);
  }

  CodeDescriptor& CodeDescriptor::operator=(const CodeDescriptor& rhs)
  {
    if(this != &rhs) {
      clear();
      copy_from(rhs);
    }
    return *this;
  }

  CodeDescriptor::~CodeDescriptor(void)
  {
    clear();
  }

  void CodeDescriptor::clear(void)
  {
    m_type = Type();
    delete_container_contents(m_impls);
    delete_container_contents(m_props);
  }

  void CodeDescriptor::copy_from(const CodeDescriptor& rhs)
  {
    m_type = rhs.m_type;
    {
      size_t s = rhs.m_impls.size();
      m_impls.resize(s);
      for(size_t i = 0; i < s; i++)
	m_impls[i] = rhs.m_impls[i]->clone();
    }
    {
      size_t s = rhs.m_props.size();
      m_props.resize(s);
      for(size_t i = 0; i < s; i++)
	m_props[i] = rhs.m_props[i]->clone();
    }
  }

  // are any of the code implementations marked as "portable" (i.e.
  //  usable in another process/address space)?
  bool CodeDescriptor::has_portable_implementations(void) const
  {
    for(std::vector<CodeImplementation *>::const_iterator it = m_impls.begin();
	it != m_impls.end();
	it++)
      if((*it)->is_portable())
	return true;
    return false;
  }

  // attempt to make a portable implementation from what we have
  bool CodeDescriptor::create_portable_implementation(void)
  {
    // TODO: actually have translators registered where we can find them
#if defined(REALM_USE_DLFCN) && defined(REALM_USE_DLADDR)
    const FunctionPointerImplementation *fpi = find_impl<FunctionPointerImplementation>();
    if(fpi) {
      DSOReferenceImplementation *dsoref = DSOReferenceImplementation::cvt_fnptr_to_dsoref(fpi, true /*quiet*/);
      if(dsoref) {
	m_impls.push_back(dsoref);
	return true;
      }
    }
#endif

    return false;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class FunctionPointerImplementation

  /*static*/ Serialization::PolymorphicSerdezSubclass<CodeImplementation,
						      FunctionPointerImplementation> FunctionPointerImplementation::serdez_subclass;

  FunctionPointerImplementation::FunctionPointerImplementation(void)
    : fnptr(0)
  {}

  FunctionPointerImplementation::FunctionPointerImplementation(void (*_fnptr)())
    : fnptr(_fnptr)
  {}

  FunctionPointerImplementation::~FunctionPointerImplementation(void)
  {}

  CodeImplementation *FunctionPointerImplementation::clone(void) const
  {
    return new FunctionPointerImplementation(fnptr);
  }

  bool FunctionPointerImplementation::is_portable(void) const
  {
    return false;
  }


#ifdef REALM_USE_DLFCN
  ////////////////////////////////////////////////////////////////////////
  //
  // class DSOReferenceImplementation

  /*static*/ Serialization::PolymorphicSerdezSubclass<CodeImplementation,
						      DSOReferenceImplementation> DSOReferenceImplementation::serdez_subclass;

  DSOReferenceImplementation::DSOReferenceImplementation(void)
  {}

  DSOReferenceImplementation::DSOReferenceImplementation(const std::string& _dso_name,
							 const std::string& _symbol_name)
    : dso_name(_dso_name), symbol_name(_symbol_name)
  {}

  DSOReferenceImplementation::~DSOReferenceImplementation(void)
  {}

  CodeImplementation *DSOReferenceImplementation::clone(void) const
  {
    return new DSOReferenceImplementation(dso_name, symbol_name);
  }

  bool DSOReferenceImplementation::is_portable(void) const
  {
    return true;
  }

#ifdef REALM_USE_DLADDR
  namespace {
    // neither pgcc nor icpc lets us declare a weak 'main'
#if !defined(__PGI) && !defined(__ICC)
    extern "C" { int main(int argc, const char *argv[]) __attribute__((weak)); };
#endif

    DSOReferenceImplementation *dladdr_helper(void *ptr, bool quiet)
    {
      // if dladdr() gives us something with the same base pointer, assume that's portable
      // note: return code is not-POSIX-y (i.e. 0 == failure)
      Dl_info inf;
      int ret = dladdr(ptr, &inf);
      if(ret == 0) {
	if(!quiet)
	  log_codetrans.warning() << "couldn't map fnptr " << ptr << " to a dynamic symbol";
	return 0;
      }

      if(inf.dli_saddr != ptr) {
	if(!quiet)
	  log_codetrans.warning() << "pointer " << ptr << " in middle of symbol '" << inf.dli_sname << " (" << inf.dli_saddr << ")?";
	return 0;
      }

      // try to detect symbols that are in the base executable and change the filename to ""
      // only do this if the weak 'main' reference found an actual main
      const char *fname = inf.dli_fname;
#if !defined(__PGI) && !defined(__ICC)
      if(((void *)main) != 0) {
	static std::string local_fname;
	if(local_fname.empty()) {
	  Dl_info inf2;
	  ret = dladdr((void *)main, &inf2);
	  assert(ret != 0);
	  local_fname = inf2.dli_fname;
	}
	if(local_fname.compare(fname) == 0)
	  fname = "";
      }
#endif
      return new DSOReferenceImplementation(fname, inf.dli_sname);
    }
  };

  /*static*/ DSOReferenceImplementation *DSOReferenceImplementation::cvt_fnptr_to_dsoref(const FunctionPointerImplementation *fpi,
											 bool quiet /*= false*/)
  {
    return dladdr_helper((void *)(fpi->fnptr), quiet);
  } 
#endif
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class CodeTranslator

  CodeTranslator::CodeTranslator(const std::string& _name)
    : name(_name)
  {}

  CodeTranslator::~CodeTranslator(void)
  {}

  // default version just iterates over all the implementations in the source
  bool CodeTranslator::can_translate(const CodeDescriptor& source_codedesc,
				     const std::type_info& target_impl_type)
  {
    const std::vector<CodeImplementation *>& impls = source_codedesc.implementations();
    for(std::vector<CodeImplementation *>::const_iterator it = impls.begin();
	it != impls.end();
	it++) {
      CodeImplementation &impl = **it;
      if(can_translate(typeid(impl), target_impl_type))
	return true;
    }

    return false;
  }

  // default version just iterates over all the implementations in the source
  CodeImplementation *CodeTranslator::translate(const CodeDescriptor& source_codedesc,
						const std::type_info& target_impl_type)
  {
    const std::vector<CodeImplementation *>& impls = source_codedesc.implementations();
    for(std::vector<CodeImplementation *>::const_iterator it = impls.begin();
	it != impls.end();
	it++) {
      CodeImplementation &impl = **it;
      if(can_translate(typeid(impl), target_impl_type))
	return translate(*it, target_impl_type);
    }

    return 0;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DSOCodeTranslator

#ifdef REALM_USE_DLFCN
  DSOCodeTranslator::DSOCodeTranslator(void)
    : CodeTranslator("dso")
  {}

  DSOCodeTranslator::~DSOCodeTranslator(void)
  {
    // unload any modules we have loaded
    for(std::map<std::string, void *>::iterator it = modules_loaded.begin();
	it != modules_loaded.end();
	it++) {
      int ret = dlclose(it->second);
      if(ret != 0)
	log_codetrans.warning() << "error on dlclose of '" << it->first << "': " << dlerror();
    }
  }

  bool DSOCodeTranslator::can_translate(const std::type_info& source_impl_type,
					   const std::type_info& target_impl_type)
  {
    // DSO ref -> function pointer
    if((source_impl_type == typeid(DSOReferenceImplementation)) &&
       (target_impl_type == typeid(FunctionPointerImplementation)))
      return true;

#ifdef REALM_USE_DLADDR
    if((source_impl_type == typeid(FunctionPointerImplementation)) &&
       (target_impl_type == typeid(DSOReferenceImplementation)))
      return true;
#endif

      return false;
    }

  CodeImplementation *DSOCodeTranslator::translate(const CodeImplementation *source,
						   const std::type_info& target_impl_type)
  {
    if(target_impl_type == typeid(FunctionPointerImplementation)) {
      const DSOReferenceImplementation *dsoref = dynamic_cast<const DSOReferenceImplementation *>(source);
      assert(dsoref != 0);

      void *handle = 0;
      // check to see if we've already loaded the module?
      std::map<std::string, void *>::iterator it = modules_loaded.find(dsoref->dso_name);
      if(it != modules_loaded.end()) {
	handle = it->second;
      } else {
	// try to load it - empty string for dso_name means the main executable
	const char *dso_name = dsoref->dso_name.c_str();
	handle = dlopen(*dso_name ? dso_name : 0, RTLD_NOW | RTLD_LOCAL);
	if(!handle) {
	  log_codetrans.warning() << "could not open DSO '" << dsoref->dso_name << "': " << dlerror();
	  return 0;
	}
	modules_loaded[dsoref->dso_name] = handle;
      }

      void *ptr = dlsym(handle, dsoref->symbol_name.c_str());
      if(!ptr) {
	log_codetrans.warning() << "could not find symbol '" << dsoref->symbol_name << "' in  DSO '" << dsoref->dso_name << "': " << dlerror();
	return 0;
      }

      return new FunctionPointerImplementation((void(*)())ptr);
    }

#ifdef REALM_USE_DLADDR
    if(target_impl_type == typeid(DSOReferenceImplementation)) {
      const FunctionPointerImplementation *fpi = dynamic_cast<const FunctionPointerImplementation *>(source);
      assert(fpi != 0);

      return dladdr_helper((void *)(fpi->fnptr), false /*!quiet*/);
    }
#endif

    return 0;
  }

  // these pass through to CodeTranslator's definitions
  bool DSOCodeTranslator::can_translate(const CodeDescriptor& source_codedesc,
					const std::type_info& target_impl_type)
  {
    return CodeTranslator::can_translate(source_codedesc, target_impl_type);
  }

  CodeImplementation *DSOCodeTranslator::translate(const CodeDescriptor& source_codedesc,
						   const std::type_info& target_impl_type)
  {
    return CodeTranslator::translate(source_codedesc, target_impl_type);
  }
#endif


};
