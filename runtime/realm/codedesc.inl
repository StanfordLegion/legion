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

// nop, but helps IDEs
#include "realm/codedesc.h"

#include "realm/serialize.h"
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

  inline bool Type::operator==(const Type& rhs) const
  {
    if(f_common.kind != rhs.f_common.kind)
      return false;
    switch(f_common.kind) {
    case InvalidKind: return true;
#define COMPARE_CASE(k, f, n) case k: return n.is_equal(rhs.n);
      REALM_TYPE_KINDS(COMPARE_CASE);
#undef COMPARE_CASE
    }
    // unreachable
    return false;
  }

  inline bool Type::operator!=(const Type& rhs) const
  {
    return !(*this == rhs);
  }

  template <typename T>
  /*static*/ Type Type::from_cpp_type(void)
  {
    return TypeConv::from_cpp_type<T>();
  }

  template <typename T>
  /*static*/ Type Type::from_cpp_value(const T& value)
  {
    return TypeConv::from_cpp_type<T>();
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

  template <typename S>
  bool serialize(S& s, const Type& t)
  {
    if(!(s << t.f_common.kind)) return false;  
    switch(t.f_common.kind) {
    case Type::InvalidKind: return true;
#define SERIALIZE_CASE(k, f, n) case Type::k: return t.n.serialize(s);
      REALM_TYPE_KINDS(SERIALIZE_CASE);
#undef SERIALIZE_CASE
    }
    return false;
  }

  template <typename S>
  bool deserialize(S& s, Type& t)
  {
    t.destroy();
    Type::Kind kind;
    if(!(s >> kind)) return false;  
    t.f_common.kind = kind;
    switch(kind) {
    case Type::InvalidKind: return true;
#define DESERIALIZE_CASE(k, f, n) case Type::k: return t.n.deserialize(s);
      REALM_TYPE_KINDS(DESERIALIZE_CASE);
#undef DESERIALIZE_CASE
    }
    return false;
  }

  inline void Type::CommonFields::destroy(void) {}

  inline void Type::CommonFields::copy_from(const CommonFields& rhs) { *this = rhs; }

  inline bool Type::CommonFields::is_equal(const CommonFields& rhs) const
  {
    return ((size_bits == rhs.size_bits) &&
	    (alignment_bits == rhs.alignment_bits));
  }

  template <typename S>
  bool Type::CommonFields::serialize(S& s) const
  {
    return (s << size_bits) && (s << alignment_bits);
  }

  template <typename S>
  bool Type::CommonFields::deserialize(S& s)
  {
    return (s >> size_bits) && (s >> alignment_bits);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class OpaqueType

  inline void Type::OpaqueFields::destroy(void) {}

  inline void Type::OpaqueFields::copy_from(const OpaqueFields& rhs) { *this = rhs; }

  inline bool Type::OpaqueFields::is_equal(const OpaqueFields& rhs) const
  {
    return CommonFields::is_equal(rhs);
  }

  inline OpaqueType::OpaqueType(size_t _size_bits, size_t _alignment_bits /*= 0*/)
    : Type(KIND, _size_bits, _alignment_bits)
  {
  }

  template <typename S>
  bool Type::OpaqueFields::serialize(S& s) const
  {
    return CommonFields::serialize(s);
  }

  template <typename S>
  bool Type::OpaqueFields::deserialize(S& s)
  {
    return CommonFields::deserialize(s);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IntegerType

  inline void Type::IntegerFields::destroy(void) {}

  inline void Type::IntegerFields::copy_from(const IntegerFields& rhs) { *this = rhs; }

  inline bool Type::IntegerFields::is_equal(const IntegerFields& rhs) const
  {
    return (CommonFields::is_equal(rhs) &&
	    (is_signed == rhs.is_signed));
  }

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

  template <typename S>
  bool Type::IntegerFields::serialize(S& s) const
  {
    return (CommonFields::serialize(s) &&
	    (s << is_signed));
  }

  template <typename S>
  bool Type::IntegerFields::deserialize(S& s)
  {
    return (CommonFields::deserialize(s) &&
	    (s >> is_signed));
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class FloatingPointType

  inline void Type::FloatingPointFields::destroy(void) {}

  inline void Type::FloatingPointFields::copy_from(const FloatingPointFields& rhs) { *this = rhs; }

  inline bool Type::FloatingPointFields::is_equal(const FloatingPointFields& rhs) const
  {
    return CommonFields::is_equal(rhs);
  }

  template <typename S>
  bool Type::FloatingPointFields::serialize(S& s) const
  {
    return CommonFields::serialize(s);
  }

  template <typename S>
  bool Type::FloatingPointFields::deserialize(S& s)
  {
    return CommonFields::deserialize(s);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PointerType

  inline void Type::PointerFields::destroy(void) { delete base_type; }

  inline void Type::PointerFields::copy_from(const PointerFields& rhs)
  { 
    *this = rhs;
    base_type = new Type(*rhs.base_type);
  }

  inline bool Type::PointerFields::is_equal(const PointerFields& rhs) const
  {
    return (CommonFields::is_equal(rhs) &&
	    is_const == rhs.is_const &&
	    *base_type == *rhs.base_type);
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

  template <typename S>
  bool Type::PointerFields::serialize(S& s) const
  {
    return (CommonFields::serialize(s) &&
	    (s << *base_type) &&
	    (s << is_const));
  }

  template <typename S>
  bool Type::PointerFields::deserialize(S& s)
  {
    base_type = new Type;
    return (CommonFields::deserialize(s) &&
	    (s >> *base_type) &&
	    (s >> is_const));
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

  inline bool Type::FunctionPointerFields::is_equal(const FunctionPointerFields& rhs) const
  {
    if(!CommonFields::is_equal(rhs))
      return false;
    if(*return_type != *rhs.return_type)
      return false;
    size_t s = (*param_types).size();
    if(s != (*rhs.param_types).size())
      return false;
    for(size_t i = 0; i < s; i++)
      if((*param_types)[i] != (*rhs.param_types)[i])
	return false;
    return true;
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

  template <typename S>
  bool Type::FunctionPointerFields::serialize(S& s) const
  {
    return (CommonFields::serialize(s) &&
	    (s << *return_type) &&
	    (s << *param_types));
  }

  template <typename S>
  bool Type::FunctionPointerFields::deserialize(S& s)
  {
    return_type = new Type;
    param_types = new std::vector<Type>;
    return (CommonFields::deserialize(s) &&
	    (s >> *return_type) &&
	    (s >> *param_types));
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

  // a common pattern is to make a code descriptor from a function pointer - we
  //  can use template magic to do this all at once
  // TODO: use some SFINAE trick to make this only work if T is a function pointer?
  template <typename T>
  inline CodeDescriptor::CodeDescriptor(T fnptr)
    : m_type(TypeConv::from_cpp_type<T>())
  {
    assert(m_type.is<FunctionPointerType>());
    FunctionPointerImplementation *fpi = new FunctionPointerImplementation(reinterpret_cast<void(*)()>(fnptr));
    m_impls.push_back(fpi);
  }

  inline CodeDescriptor& CodeDescriptor::set_type(const Type& _t)
  {
    m_type = _t;
    return *this;
  }

  // add an implementation - becomes owned by the descriptor
  inline CodeDescriptor& CodeDescriptor::add_implementation(CodeImplementation *impl)
  {
    m_impls.push_back(impl);
    return *this;
  }

  // add a property - becomes owned by the descriptor
  inline CodeDescriptor& CodeDescriptor::add_property(CodeProperty *prop)
  {
    m_props.push_back(prop);
    return *this;
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

  template <typename S>
  bool CodeDescriptor::serialize(S& s, bool portable) const
  {
    if(!(s << m_type)) return false;
    if(portable) {
      // only count and serialize portable implementations
      size_t n = 0;
      for(size_t i = 0; i < m_impls.size(); i++)
	if(m_impls[i]->is_portable())
	  n++;
      if(!(s << n)) return false;
      for(size_t i = 0; i < m_impls.size(); i++)
	if(m_impls[i]->is_portable())
	  if(!(s << *m_impls[i])) return false;
    } else {
      // just do all the implementations
      if(!(s << m_impls.size())) return false;
      for(size_t i = 0; i < m_impls.size(); i++)
	if(!(s << *m_impls[i])) return false;
    }

    return true;
  }

  template <typename S>
  bool CodeDescriptor::deserialize(S& s)
  {
    if(!(s >> m_type)) return false;
    size_t n;
    if(!(s >> n)) return false;
    m_impls.clear();
    m_impls.resize(n);
    for(size_t i = 0; i < n; i++)
      m_impls[i] = CodeImplementation::deserialize_new(s);

    return true;
  }

  template <typename S>
  bool serialize(S& s, const CodeDescriptor& cd)
  {
    return cd.serialize(s, true /*portable*/);
  }

  template <typename S>
  bool deserialize(S& s, CodeDescriptor& cd)
  {
    return cd.deserialize(s);
  }

  inline std::ostream& operator<<(std::ostream& os, const CodeDescriptor& cd)
  {
    os << "CD{ type=" << cd.m_type << ", impls = [";
    if(!cd.m_impls.empty()) {
      os << " " << *cd.m_impls[0];
      for(size_t i = 1; i < cd.m_impls.size(); i++)
	os << ", " << *cd.m_impls[i];
      os << " ";
    }
    os << "] }";
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CodeImplementation

  inline CodeImplementation::CodeImplementation(void)
  {}

  inline CodeImplementation::~CodeImplementation(void)
  {}

  inline std::ostream& operator<<(std::ostream& os, const CodeImplementation& ci)
  {
    ci.print(os);
    return os;
  }

  template <typename S>
  //inline bool CodeImplementation::serialize(S& serializer) const
  inline bool serialize(S& serializer, const CodeImplementation& ci)
  {
    return Serialization::PolymorphicSerdezHelper<CodeImplementation>::serialize(serializer, ci);
  }

  template <typename S>
  /*static*/ inline CodeImplementation *CodeImplementation::deserialize_new(S& deserializer)
  {
    return Serialization::PolymorphicSerdezHelper<CodeImplementation>::deserialize_new(deserializer);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class FunctionPointerImplementation

  inline void FunctionPointerImplementation::print(std::ostream& os) const
  {
    os << "fnptr(0x" << std::hex << reinterpret_cast<intptr_t>(fnptr) << std::dec << ")";
  }

  template <typename S>
  inline bool FunctionPointerImplementation::serialize(S& serializer) const
  {
    intptr_t as_int = reinterpret_cast<intptr_t>(fnptr);
    return serializer << as_int;
  }

  template <typename S>
  inline /*static*/ CodeImplementation *FunctionPointerImplementation::deserialize_new(S& deserializer)
  {
    intptr_t as_int;
    if(!(deserializer >> as_int)) return 0;
    return new FunctionPointerImplementation(reinterpret_cast<void(*)()>(as_int));
  }


#ifdef REALM_USE_DLFCN
  ////////////////////////////////////////////////////////////////////////
  //
  // class DSOReferenceImplementation

  inline void DSOReferenceImplementation::print(std::ostream& os) const
  {
    os << "dsoref(" << dso_name << "," << symbol_name << ")";
  }

  template <typename S>
  inline bool DSOReferenceImplementation::serialize(S& serializer) const
  {
    return (serializer << dso_name) && (serializer << symbol_name);
  }

  template <typename S>
  inline /*static*/ CodeImplementation *DSOReferenceImplementation::deserialize_new(S& deserializer)
  {
    DSOReferenceImplementation *dsoref = new DSOReferenceImplementation;
    if((deserializer >> dsoref->dso_name) && (deserializer >> dsoref->symbol_name)) {
      return dsoref;
    } else {
      delete dsoref;
      return 0;
    }
  }
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class CodeProperty

  inline CodeProperty::CodeProperty(void)
  {}

  inline CodeProperty::~CodeProperty(void)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // class CodeTranslator

  template <typename TARGET_TYPE>
  bool CodeTranslator::can_translate(const std::type_info& source_impl_type)
  {
    return can_translate(source_impl_type, typeid(TARGET_TYPE));
  }

  template <typename TARGET_TYPE>
  bool CodeTranslator::can_translate(const CodeDescriptor& source_codedesc)
  {
    return can_translate(source_codedesc, typeid(TARGET_TYPE));
  }

  template <typename TARGET_TYPE>
  TARGET_TYPE *CodeTranslator::translate(const CodeImplementation *source)
  {
    return static_cast<TARGET_TYPE *>(translate(source, typeid(TARGET_TYPE)));
  }

  template <typename TARGET_TYPE>
  TARGET_TYPE *CodeTranslator::translate(const CodeDescriptor& source_codedesc)
  {
    return static_cast<TARGET_TYPE *>(translate(source_codedesc, typeid(TARGET_TYPE)));
  }


}; // namespace Realm
