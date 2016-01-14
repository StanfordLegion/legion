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

// constructs for describing code blobs to Realm

#include "codedesc.h"

#include <dlfcn.h>

#include "logging.h"
#include "utils.h"

namespace Realm {

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


  ////////////////////////////////////////////////////////////////////////
  //
  // class FunctionPointerImplementation

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


  ////////////////////////////////////////////////////////////////////////
  //
  // class DSOReferenceImplementation

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


  ////////////////////////////////////////////////////////////////////////
  //
  // code translators

  Logger log_codetrans("codetrans");

  FunctionPointerImplementation *cvt_dsoref_to_fnptr(const DSOReferenceImplementation *dso)
  {
    // TODO: once this moves to a "code translator" object with state, actually keep
    //  track of all the handles we open, reuse when possible, and clean up at end

    void *handle = dlopen(dso->dso_name.c_str(), RTLD_NOW | RTLD_LOCAL);
    if(!handle) {
      log_codetrans.warning() << "could not open DSO '" << dso->dso_name << "': " << dlerror();
      return 0;
    }

    void *ptr = dlsym(handle, dso->symbol_name.c_str());
    if(!ptr) {
      log_codetrans.warning() << "could not find symbol '" << dso->symbol_name << "' in  DSO '" << dso->dso_name << "': " << dlerror();
      return 0;
    }

    return new FunctionPointerImplementation((void(*)())ptr);
  }

  DSOReferenceImplementation *cvt_fnptr_to_dsoref(const FunctionPointerImplementation *fpi)
  {
    // if dladdr() gives us something with the same base pointer, assume that's portable
    Dl_info inf;
    int ret;

    // note: return code is not-POSIX-y (i.e. 0 == failure)
    void *ptr = (void *)(fpi->fnptr);
    ret = dladdr(ptr, &inf);
    if(ret == 0) {
      log_codetrans.warning() << "couldn't map fnptr " << ptr << " to a dynamic symbol";
      return 0;
    }

    if(inf.dli_saddr != ptr) {
      log_codetrans.warning() << "pointer " << ptr << " in middle of symbol '" << inf.dli_sname << " (" << inf.dli_saddr << ")?";
      return 0;
    }

    return new DSOReferenceImplementation(inf.dli_fname, inf.dli_sname);
  }

};
