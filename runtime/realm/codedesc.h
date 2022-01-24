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

#ifndef REALM_CODEDESC_H
#define REALM_CODEDESC_H

#include "realm/realm_config.h"

#include "realm/serialize.h"

#include <stddef.h>
#include <vector>
#include <iostream>

namespace Realm {

  // we need a way to describe types of functions and arguments - every JIT framework
  //  (e.g. llvm) has their own way of doing this, so this is intended to be a generic
  //  version that knows how to convert to/from implementation-specific versions

  // Types have value semantics and are implemented as a tagged union
  class Type {
  public:
    // default constructor creates an invalid type
    Type(void);

    // copy and assignment do what you want
    Type(const Type& rhs);
    Type& operator=(const Type& rhs);

    ~Type(void);

    // construction of useful types is usually done via template-fu
    template <typename T>
    static Type from_cpp_type(void);

    template <typename T>
    static Type from_cpp_value(const T& value);

    // exact equality
    bool operator==(const Type& rhs) const;
    bool operator!=(const Type& rhs) const;

    // testing for the kind
    bool is_valid(void) const;

    template <typename T>
    bool is(void) const;

    template <typename T>
    T& as(void);

    template <typename T>
    const T& as(void) const;

    // accessors/mutators for common fields
    size_t& size_bits(void);
    const size_t& size_bits(void) const;

    size_t& alignment_bits(void);
    const size_t& alignment_bits(void) const;

    // pretty-printing
    friend std::ostream& operator<<(std::ostream& os, const Type& t);

    // serializer/deserializer functions
    template <typename S>
    friend bool serialize(S& os, const Type& t);

    template <typename S>
    friend bool deserialize(S& os, Type& t);

#define REALM_TYPE_KINDS(__func__) \
    __func__(OpaqueKind, OpaqueFields, f_opaque) \
    __func__(IntegerKind, IntegerFields, f_integer) \
    __func__(FloatingPointKind, FloatingPointFields, f_float) \
    __func__(PointerKind, PointerFields, f_pointer) \
    __func__(FunctionPointerKind, FunctionPointerFields, f_funcptr)

    enum Kind {
      InvalidKind,
#define KINDS_ENUM(k, f, n) k,
      REALM_TYPE_KINDS(KINDS_ENUM)
#undef KINDS_ENUM
    };

  protected:
#define FIELDOBJ_METHODS(classname) \
      void destroy(void); \
      void copy_from(const classname &rhs); \
      bool is_equal(const classname& rhs) const; \
      template <typename S> bool serialize(S& s) const; \
      template <typename S> bool deserialize(S& s)
    struct CommonFields {
      Kind kind;
      size_t size_bits;
      size_t alignment_bits;

      FIELDOBJ_METHODS(CommonFields);
    };
    struct OpaqueFields : public CommonFields {
      // nothing

      FIELDOBJ_METHODS(OpaqueFields);
    };
    struct IntegerFields : public CommonFields {
      bool is_signed;

      FIELDOBJ_METHODS(IntegerFields);
    };
    struct FloatingPointFields : public CommonFields {
      // nothing

      FIELDOBJ_METHODS(FloatingPointFields);
    };
    struct PointerFields : public CommonFields {
      Type *base_type;
      bool is_const;

      FIELDOBJ_METHODS(PointerFields);
    };
    struct FunctionPointerFields : public CommonFields {
      Type *return_type;
      std::vector<Type> *param_types;

      FIELDOBJ_METHODS(FunctionPointerFields);
    };
#undef FIELDOBJ_METHODS

    union {
      CommonFields f_common;
#define FIELDS_ENTRY(k, f, n) f n;
      REALM_TYPE_KINDS(FIELDS_ENTRY)
#undef FIELDS_ENTRY
    };

    void destroy(void);
    void copy_from(const Type& rhs);

    // only used by subclass constructors
    Type(Kind _kind, size_t _size_bits, size_t _alignment_bits);


  };

  class OpaqueType : public Type {
  public:
    static const Type::Kind KIND = OpaqueKind;

    OpaqueType(size_t _size_bits, size_t _alignment_bits = 0);
  };

  class IntegerType : public Type {
  public:
    static const Type::Kind KIND = IntegerKind;

    IntegerType(size_t _size_bits, bool _signed, size_t _alignment_bits = 0);

    bool& is_signed(void);
    const bool& is_signed(void) const;
  };

  class PointerType : public Type {
  public:
    static const Type::Kind KIND = PointerKind;

    PointerType(const Type& _base_type, bool _const = false,
		size_t _size_bits = 0, size_t _alignment_bits = 0);

    Type& base_type(void);
    const Type& base_type(void) const;

    bool& is_const(void);
    const bool& is_const(void) const;
  };

  class FunctionPointerType : public Type {
  public:
    static const Type::Kind KIND = FunctionPointerKind;

    // different constructors for each number of parameters...
    FunctionPointerType(const Type &_return_type,
			size_t _size_bits = 0, size_t _alignment_bits = 0);
    FunctionPointerType(const Type &_return_type,
			const Type &_param1_type,
			size_t _size_bits = 0, size_t _alignment_bits = 0);
    FunctionPointerType(const Type &_return_type,
			const Type &_param1_type,
			const Type &_param2_type,
			size_t _size_bits = 0, size_t _alignment_bits = 0);
    FunctionPointerType(const Type &_return_type,
			const Type &_param1_type,
			const Type &_param2_type,
			const Type &_param3_type,
			size_t _size_bits = 0, size_t _alignment_bits = 0);
    FunctionPointerType(const Type &_return_type,
			const Type &_param1_type,
			const Type &_param2_type,
			const Type &_param3_type,
			const Type &_param4_type,
			size_t _size_bits = 0, size_t _alignment_bits = 0);
    FunctionPointerType(const Type &_return_type,
			const Type &_param1_type,
			const Type &_param2_type,
			const Type &_param3_type,
			const Type &_param4_type,
			const Type &_param5_type,
			size_t _size_bits = 0, size_t _alignment_bits = 0);

    Type& return_type(void);
    const Type& return_type(void) const;

    std::vector<Type>& param_types(void);
    const std::vector<Type>& param_types(void) const;
  };

  namespace TypeConv {
    // these generate a Type object from a C++ type or value
    template <typename T> Type from_cpp_type(void);
    template <typename T> Type from_cpp_value(const T& value);
  };


  // a CodeDescriptor is an object that describes a blob of code as a callable function
  // it includes:
  // a) its type, as a Type object
  // b) zero or more "implementations" (e.g. function pointer, or LLVM IR, or DSO reference)
  // c) zero or more "properties" (e.g. "this function is pure")

  class CodeImplementation;
  class CodeProperty;

  class REALM_PUBLIC_API CodeDescriptor {
  public:
    CodeDescriptor(void);
    explicit CodeDescriptor(const Type& _t);

    // a common pattern is to make a code descriptor from a function pointer - we
    //  can use template magic to do this all at once
    // TODO: use some SFINAE trick to make this only work if T is a function pointer?
    template <typename T>
    explicit CodeDescriptor(T fnptr);

    // copy and assignment
    CodeDescriptor(const CodeDescriptor& rhs);
    CodeDescriptor& operator=(const CodeDescriptor& rhs);

    ~CodeDescriptor(void);

    // erase contents
    void clear(void);

    CodeDescriptor& set_type(const Type& _t);

    // add an implementation - becomes owned by the descriptor
    CodeDescriptor& add_implementation(CodeImplementation *impl);

    // add a property - becomes owned by the descriptor
    CodeDescriptor& add_property(CodeProperty *prop);

    const Type& type(void) const;
    const std::vector<CodeImplementation *>& implementations(void) const;
    const std::vector<CodeProperty *>& properties(void) const;

    // are any of the code implementations marked as "portable" (i.e.
    //  usable in another process/address space)?
    bool has_portable_implementations(void) const;

    // attempt to make a portable implementation from what we have
    bool create_portable_implementation(void);

    template <typename T>
    const T *find_impl(void) const;

    // pretty-printing
    friend std::ostream& operator<<(std::ostream& os, const CodeDescriptor& cd);

    // serialization/deserialization - note that the standard << serializer will
    //  not serialize non-portable implementations
    template <typename S>
    bool serialize(S& serializer, bool portable) const;

    template <typename S>
    bool deserialize(S& deserializer);

  protected:
    void copy_from(const CodeDescriptor& rhs);

    Type m_type;
    std::vector<CodeImplementation *> m_impls;
    std::vector<CodeProperty *> m_props;
  };

  template <typename S>
  bool serialize(S& serializer, const CodeDescriptor& cd);

  template <typename S>
  bool deserialize(S& deserializer, CodeDescriptor& cd);

  // this is the interface that actual CodeImplementations must follow
  class REALM_PUBLIC_API CodeImplementation {
  protected:
    // not directly constructed
    CodeImplementation(void);

  public:
    virtual ~CodeImplementation(void);

    virtual CodeImplementation *clone(void) const = 0;

    // is this implementation meaningful in another address space?
    virtual bool is_portable(void) const = 0;

    //template <typename S>
    //bool serialize(S& serializer) const;

    template <typename S>
    static CodeImplementation *deserialize_new(S& deserializer);

    // pretty-printing
    friend std::ostream& operator<<(std::ostream& os, const CodeImplementation& ci);

  protected:
    virtual void print(std::ostream& os) const = 0;
  };

  template <typename S>
  bool serialize(S& serializer, const CodeImplementation& ci);

  // this is the interface that actual CodePropertys must follow
  class REALM_PUBLIC_API CodeProperty {
  protected:
    // not directly constructed
    CodeProperty(void);

  public:
    virtual ~CodeProperty(void);

    virtual CodeProperty *clone(void) const = 0;

    // is this implementation meaningful in another address space?
    virtual bool is_portable(void) const = 0;

    // TODO: serialization/deserialization stuff
  };

  // abstract class that describes a code translator that can convert a code implementation
  //  of one type into one of another type - in order to be extensible by add-on modules, it
  //  uses c++ dynamic type info support
  class CodeTranslator {
  protected:
    CodeTranslator(const std::string& _name);

  public:
    virtual ~CodeTranslator(void);

    virtual bool can_translate(const std::type_info& source_impl_type,
			       const std::type_info& target_impl_type) = 0;

    // default version just iterates over all the implementations in the source
    virtual bool can_translate(const CodeDescriptor& source_codedesc,
			       const std::type_info& target_impl_type);

    virtual CodeImplementation *translate(const CodeImplementation *source,
					  const std::type_info& target_impl_type) = 0;

    // default version just iterates over all the implementations in the source
    virtual CodeImplementation *translate(const CodeDescriptor& source_codedesc,
					  const std::type_info& target_impl_type);

    // template versions when target type is statically known
    template <typename TARGET_TYPE>
    bool can_translate(const std::type_info& source_impl_type);

    template <typename TARGET_TYPE>
    bool can_translate(const CodeDescriptor& source_codedesc);

    template <typename TARGET_TYPE>
    TARGET_TYPE *translate(const CodeImplementation *source);

    template <typename TARGET_TYPE>
    TARGET_TYPE *translate(const CodeDescriptor& source_codedesc);

    std::string name;
  };

  // two simple implementations:
  // 1) raw function pointers - non-portable
  // 2) DSO references (i.e. name of shared object, name of symbol) - portable

  class REALM_PUBLIC_API FunctionPointerImplementation : public CodeImplementation {
  public:
    // note that this implementation forgets the actual function prototype - it's
    //  up to the surrounding CodeDescriptor object to remember that
    FunctionPointerImplementation(void (*_fnptr)());

    virtual ~FunctionPointerImplementation(void);

    virtual CodeImplementation *clone(void) const;

    virtual bool is_portable(void) const;

    template<typename T>
    inline T get_impl(void) const { return reinterpret_cast<T>(fnptr); }

    template <typename S>
    bool serialize(S& serializer) const;

    template <typename S>
    static CodeImplementation *deserialize_new(S& deserializer);

  protected:
    REALM_INTERNAL_API_EXTERNAL_LINKAGE
    FunctionPointerImplementation(void);

    static Serialization::PolymorphicSerdezSubclass<CodeImplementation, FunctionPointerImplementation> serdez_subclass;

    virtual void print(std::ostream& os) const;

  public:
    void (*fnptr)();
  };

#ifdef REALM_USE_DLFCN
  class REALM_PUBLIC_API DSOReferenceImplementation : public CodeImplementation {
  public:
    DSOReferenceImplementation(const std::string& _dso_name,
			       const std::string& _symbol_name);

    virtual ~DSOReferenceImplementation(void);

    virtual CodeImplementation *clone(void) const;

    virtual bool is_portable(void) const;

    template <typename S>
    bool serialize(S& serializer) const;

    template <typename S>
    static CodeImplementation *deserialize_new(S& deserializer);

  protected:
    DSOReferenceImplementation(void);

    static Serialization::PolymorphicSerdezSubclass<CodeImplementation, DSOReferenceImplementation> serdez_subclass;

    virtual void print(std::ostream& os) const;

#ifdef REALM_USE_DLADDR
    friend class CodeDescriptor;
    static DSOReferenceImplementation *cvt_fnptr_to_dsoref(const FunctionPointerImplementation *fpi,
                                                           bool quiet = false);
#endif
  public:
    std::string dso_name, symbol_name;
  };

  // converts DSOReferenceImplementation -> FunctionPointerImplementation and
  //  (if dladdr is available) the reverse
  // NOTE: this is exported for now due to issue #818
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE DSOCodeTranslator : public CodeTranslator {
  public:
    DSOCodeTranslator(void);

    virtual ~DSOCodeTranslator(void);

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
    std::map<std::string, void *> modules_loaded;
  };
#endif

}; // namespace Realm

#include "realm/codedesc.inl"

#undef REALM_TYPE_KINDS

#endif
