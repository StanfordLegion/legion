/* Copyright 2015 Stanford University, NVIDIA Corporation
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

// nop, but helps IDEs
#include "codedesc.h"

#include "serialize.h"
TYPE_IS_SERIALIZABLE(Realm::Type::Kind);

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class Type

  inline Type::Type(void)
  {
    f_common.kind = InvalidKind;
  }

  inline Type::Type(Kind _kind, size_t _size_bits, size_t _alignment_bits)
  {
    f_common.kind = _kind;
    f_common.size_bits = _size_bits;
    f_common.alignment_bits = _alignment_bits ? _alignment_bits : _size_bits;
  }

  inline Type::~Type(void)
  {
    destroy();
  }

  inline Type::Type(const Type& rhs)
  {
    copy_from(rhs);
  }

  inline Type& Type::operator=(const Type& rhs)
  {
    // skip self-copy
    if(this != &rhs) {
      destroy();
      copy_from(rhs);
    }
    return *this;
  }

  inline bool Type::is_valid(void) const
  {
    return (f_common.kind != InvalidKind);
  }

  template <typename T>
  inline bool Type::is(void) const
  {
    return (f_common.kind == T::KIND);
  }

  inline size_t& Type::size_bits(void)
  {
    return f_common.size_bits;
  }

  inline const size_t& Type::size_bits(void) const
  {
    return f_common.size_bits;
  }

  inline size_t& Type::alignment_bits(void)
  {
    return f_common.alignment_bits;
  }

  inline const size_t& Type::alignment_bits(void) const
  {
    return f_common.alignment_bits;
  }

  inline void Type::destroy(void)
  {
    switch(f_common.kind) {
    case InvalidKind: f_common.destroy(); break;
#define DESTROY_CASE(k, f, n) case k: n.destroy(); break;
      REALM_TYPE_KINDS(DESTROY_CASE);
#undef DESTROY_CASE
    }
  }

  inline void Type::copy_from(const Type& rhs)
  {
    // no call to destroy here - caller must do that if current state is valid
    switch(rhs.f_common.kind) {
    case InvalidKind: f_common.copy_from(rhs.f_common); break;
#define COPY_CASE(k, f, n) case k: n.copy_from(rhs.n); break;
      REALM_TYPE_KINDS(COPY_CASE);
#undef COPY_CASE
    }
  }

  inline std::ostream& operator<<(std::ostream& os, const Type& t)
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

  inline void Type::CommonFields::destroy(void) {}

  inline void Type::CommonFields::copy_from(const CommonFields& rhs) { *this = rhs; }


  ////////////////////////////////////////////////////////////////////////
  //
  // class OpaqueType

  inline void Type::OpaqueFields::destroy(void) {}

  inline void Type::OpaqueFields::copy_from(const OpaqueFields& rhs) { *this = rhs; }

  inline OpaqueType::OpaqueType(size_t _size_bits, size_t _alignment_bits /*= 0*/)
    : Type(KIND, _size_bits, _alignment_bits)
  {
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IntegerType

  inline void Type::IntegerFields::destroy(void) {}

  inline void Type::IntegerFields::copy_from(const IntegerFields& rhs) { *this = rhs; }

  inline IntegerType::IntegerType(size_t _size_bits, bool _signed, size_t _alignment_bits /*= 0*/)
    : Type(KIND, _size_bits, _alignment_bits)
  {
    f_integer.is_signed = _signed;
  }

  inline bool& IntegerType::is_signed(void)
  {
    return f_integer.is_signed;
  }

  inline const bool& IntegerType::is_signed(void) const
  {
    return f_integer.is_signed;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class FloatingPointType

  inline void Type::FloatingPointFields::destroy(void) {}

  inline void Type::FloatingPointFields::copy_from(const FloatingPointFields& rhs) { *this = rhs; }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PointerType

  inline void Type::PointerFields::destroy(void) { delete base_type; }

  inline void Type::PointerFields::copy_from(const PointerFields& rhs)
  { 
    *this = rhs;
    base_type = new Type(*rhs.base_type);
  }

  inline PointerType::PointerType(const Type& _base_type, bool _const /*= false*/,
				  size_t _size_bits /*= 0*/, size_t _alignment_bits /*= 0*/)
    : Type(KIND, _size_bits ? _size_bits : 8 * sizeof(void *), _alignment_bits)
  {
    f_pointer.base_type = new Type(_base_type);
    f_pointer.is_const = _const;
  }

  inline Type& PointerType::base_type(void)
  {
    return *f_pointer.base_type;
  }

  inline bool& PointerType::is_const(void)
  {
    return f_pointer.is_const;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class FunctionPointerType

  inline void Type::FunctionPointerFields::destroy(void)
  {
    delete return_type;
    delete param_types;
  }

  inline void Type::FunctionPointerFields::copy_from(const FunctionPointerFields& rhs)
  {
    *this = rhs;
    return_type = new Type(*rhs.return_type);
    param_types = new std::vector<Type>(*rhs.param_types);
  }

  inline FunctionPointerType::FunctionPointerType(const Type &_return_type,
						  size_t _size_bits /*= 0*/,
						  size_t _alignment_bits /*= 0*/)
    : Type(KIND, _size_bits ? _size_bits : 8 * sizeof(void *), _alignment_bits)
  {
    f_funcptr.return_type = new Type(_return_type);
    f_funcptr.param_types = new std::vector<Type>;
  }

  inline FunctionPointerType::FunctionPointerType(const Type &_return_type,
						  const Type &_param1_type,
						  size_t _size_bits /*= 0*/,
						  size_t _alignment_bits /*= 0*/)
    : Type(KIND, _size_bits ? _size_bits : 8 * sizeof(void *), _alignment_bits)
  {
    f_funcptr.return_type = new Type(_return_type);
    std::vector<Type> *v = new std::vector<Type>(1);
    (*v)[0] = _param1_type;
    f_funcptr.param_types = v;
  }

  inline FunctionPointerType::FunctionPointerType(const Type &_return_type,
						  const Type &_param1_type,
						  const Type &_param2_type,
						  size_t _size_bits /*= 0*/,
						  size_t _alignment_bits /*= 0*/)
    : Type(KIND, _size_bits ? _size_bits : 8 * sizeof(void *), _alignment_bits)
  {
    f_funcptr.return_type = new Type(_return_type);
    std::vector<Type> *v = new std::vector<Type>(2);
    (*v)[0] = _param1_type;
    (*v)[1] = _param2_type;
    f_funcptr.param_types = v;
  }

  inline FunctionPointerType::FunctionPointerType(const Type &_return_type,
						  const Type &_param1_type,
						  const Type &_param2_type,
						  const Type &_param3_type,
						  size_t _size_bits /*= 0*/,
						  size_t _alignment_bits /*= 0*/)
    : Type(KIND, _size_bits ? _size_bits : 8 * sizeof(void *), _alignment_bits)
  {
    f_funcptr.return_type = new Type(_return_type);
    std::vector<Type> *v = new std::vector<Type>(3);
    (*v)[0] = _param1_type;
    (*v)[1] = _param2_type;
    (*v)[2] = _param3_type;
    f_funcptr.param_types = v;
  }

  inline FunctionPointerType::FunctionPointerType(const Type &_return_type,
						  const Type &_param1_type,
						  const Type &_param2_type,
						  const Type &_param3_type,
						  const Type &_param4_type,
						  size_t _size_bits /*= 0*/,
						  size_t _alignment_bits /*= 0*/)
    : Type(KIND, _size_bits ? _size_bits : 8 * sizeof(void *), _alignment_bits)
  {
    f_funcptr.return_type = new Type(_return_type);
    std::vector<Type> *v = new std::vector<Type>(4);
    (*v)[0] = _param1_type;
    (*v)[1] = _param2_type;
    (*v)[2] = _param3_type;
    (*v)[3] = _param4_type;
    f_funcptr.param_types = v;
  }

  inline FunctionPointerType::FunctionPointerType(const Type &_return_type,
						  const Type &_param1_type,
						  const Type &_param2_type,
						  const Type &_param3_type,
						  const Type &_param4_type,
						  const Type &_param5_type,
						  size_t _size_bits /*= 0*/,
						  size_t _alignment_bits /*= 0*/)
    : Type(KIND, _size_bits ? _size_bits : 8 * sizeof(void *), _alignment_bits)
  {
    f_funcptr.return_type = new Type(_return_type);
    std::vector<Type> *v = new std::vector<Type>(5);
    (*v)[0] = _param1_type;
    (*v)[1] = _param2_type;
    (*v)[2] = _param3_type;
    (*v)[3] = _param4_type;
    (*v)[4] = _param5_type;
    f_funcptr.param_types = v;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // namespace TypeConv

  namespace TypeConv {
    // level of indirection here allows specialization of from_cpp_{type,value}
    //  in a common way
    template <typename T> struct CppTypeCapture {};
  
    inline Type from_cpp(CppTypeCapture<void>)
    {
      return OpaqueType(0, 0);
    }

    // TODO: restrict this default to just things that are Realm-serializable?
    template <typename T>
    inline Type from_cpp(CppTypeCapture<T>)
    {
      return OpaqueType(8 * sizeof(T));
    }

    inline Type from_cpp(CppTypeCapture<int>)
    {
      return IntegerType(8 * sizeof(int), true /*signed*/);
    }

    inline Type from_cpp(CppTypeCapture<char>)
    {
      return IntegerType(8 * sizeof(char), true /*signed*/);
    }

    inline Type from_cpp(CppTypeCapture<unsigned long>)
    {
      return IntegerType(8 * sizeof(unsigned long), false /*unsigned*/);
    }

    template <typename T>
    inline Type from_cpp(CppTypeCapture<T *>)
    {
      return PointerType(from_cpp_type<T>(), false /*non-const*/);
    }

    template <typename T>
    inline Type from_cpp(CppTypeCapture<const T *>)
    {
      return PointerType(from_cpp_type<T>(), true /*const*/);
    }

    template <typename RT>
    inline Type from_cpp(CppTypeCapture<RT (*)(void)>)
    {
      return FunctionPointerType(from_cpp_type<RT>());
    }
    
    template <typename RT, typename T1>
    inline Type from_cpp(CppTypeCapture<RT (*)(T1)>)
    {
      return FunctionPointerType(from_cpp_type<RT>(),
				 from_cpp_type<T1>());
    }

    template <typename RT, typename T1, typename T2>
    inline Type from_cpp(CppTypeCapture<RT (*)(T1, T2)>)
    {
      return FunctionPointerType(from_cpp_type<RT>(),
				 from_cpp_type<T1>(),
				 from_cpp_type<T2>());
    }

    template <typename RT, typename T1, typename T2, typename T3>
    inline Type from_cpp(CppTypeCapture<RT (*)(T1, T2, T3)>)
    {
      return FunctionPointerType(from_cpp_type<RT>(),
				 from_cpp_type<T1>(),
				 from_cpp_type<T2>(),
				 from_cpp_type<T3>());
    }

    template <typename RT, typename T1, typename T2, typename T3, typename T4>
    inline Type from_cpp(CppTypeCapture<RT (*)(T1, T2, T3, T4)>)
    {
      return FunctionPointerType(from_cpp_type<RT>(),
				 from_cpp_type<T1>(),
				 from_cpp_type<T2>(),
				 from_cpp_type<T3>(),
				 from_cpp_type<T4>());
    }

    template <typename RT, typename T1, typename T2, typename T3, typename T4, typename T5>
    inline Type from_cpp(CppTypeCapture<RT (*)(T1, T2, T3, T4, T5)>)
    {
      return FunctionPointerType(from_cpp_type<RT>(),
				 from_cpp_type<T1>(),
				 from_cpp_type<T2>(),
				 from_cpp_type<T3>(),
				 from_cpp_type<T4>(),
				 from_cpp_type<T5>());
    }

    template <typename T>
    inline Type from_cpp_type(void)
    {
      return from_cpp(CppTypeCapture<T>());
    }

    template <typename T>
    inline Type from_cpp_value(const T& value)
    {
      return from_cpp(CppTypeCapture<T>());
    }
  };


  ////////////////////////////////////////////////////////////////////////
  //
  // class CodeDescriptor

  inline CodeDescriptor::CodeDescriptor(void)
  {}

  inline CodeDescriptor::CodeDescriptor(const Type& _t)
    : m_type(_t)
  {}

  // a common pattern is to make a code descriptor from a function pointer - we
  //  can use template magic to do this all at once
  // TODO: use some SFINAE trick to make this only work if T is a function pointer?
  template <typename T>
  inline CodeDescriptor::CodeDescriptor(T fnptr)
    : m_type(TypeConv::from_cpp_type<T>())
  {
    assert(m_type.is<FunctionPointerType>());
    m_impls.push_back(new FunctionPointerImplementation((void(*)())(fnptr)));
  }

  inline CodeDescriptor::~CodeDescriptor(void)
  {
    // TODO: delete impls, props
  }

  inline const Type& CodeDescriptor::type(void) const
  {
    return m_type;
  }

  inline const std::vector<CodeImplementation *>& CodeDescriptor::implementations(void) const
  {
    return m_impls;
  }

  inline const std::vector<CodeProperty *>& CodeDescriptor::properties(void) const
  {
    return m_props;
  }

  template <typename T>
  const T *CodeDescriptor::find_impl(void) const
  {
    for(std::vector<CodeImplementation *>::const_iterator it = m_impls.begin();
	it != m_impls.end();
	it++) {
      const T *i = dynamic_cast<const T *>(*it);
      if(i)
	return i;
    }
    return 0;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CodeImplementation

  inline CodeImplementation::CodeImplementation(void)
  {}

  inline CodeImplementation::~CodeImplementation(void)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // class FunctionPointerImplementation

  inline FunctionPointerImplementation::FunctionPointerImplementation(void (*_fnptr)())
    : fnptr(_fnptr)
  {}

  inline FunctionPointerImplementation::~FunctionPointerImplementation(void)
  {}

  inline bool FunctionPointerImplementation::is_portable(void) const
  {
    return false;
  }

}; // namespace Realm
