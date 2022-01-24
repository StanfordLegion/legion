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

// serialization logic for packing/unpacking data in byte streams

#ifndef REALM_SERIALIZE_H
#define REALM_SERIALIZE_H

#include "realm/realm_config.h"
#include "realm/bytearray.h"
#include "realm/utils.h"

#include <stddef.h>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <string>
#include <typeinfo>

// To add serializability for a new data type, do one of the following:
//
// 1) If your type can be copied bit for bit (i.e. C++11's is_trivially_copyable), just do:
//  TYPE_IS_SERIALIZABLE(my_type);
//
// 2) If your type is just a struct with a bunch of members, do:
//  template<typename S>
//  bool serdez(S& s, const my_type& t)
//  {
//    return (s & t.field1) && (s & t.field2) ... ;
//  }
//
// 3) If your type needs to serialize and deserialize in a complicated way, define them separately:
//
//  template<typename S>
//  bool serialize(S& s, const my_type& t)
//  {
//    // serialization code here...
//  }
//
//  template<typename S>
//  bool deserialize(S& s, my_type& t)
//  {
//    // deserialization code here...
//  }


// helper template tells us which types can be directly copied into serialized streams
// (this is similar to C++11's is_trivially_copyable, but does not include pointers since
// their values are not meaningful to other nodes)
//
// The implementation is a bit weird, because we need to be able to add new serializable types
// from other namespaces.  This prevents us from doing a simple specialization of a base template
// because you can only specialize in the same namespace in which the template was originally
// declared.  Instead, we use the ability to overload operators with different return types and
// test the sizeof() the return type to see if a given type T has been specifically listed as
// serializable.  To make it extremely unlikely that it'll break any actual code, we use the comma
// operator and an unconstructable type on the left hand side.
//
// Addendum: It turns out namespaces are still problematic and this particular part needs to be in the
//   global namespace (unless you want to force everybody to import Realm::Serialization), even though
//   calls to TYPE_IS_SERIALIZABLE() _can_ be done from other namespaces.
struct is_copy_serializable {
public:
  class inner { public: inner(void) {} };
public:
  template <typename T>
  struct test { static const bool value = sizeof(inner(),*reinterpret_cast<T*>(0)) != sizeof(char); };
};
template <typename T>
char operator,(const is_copy_serializable::inner&, const T&);

#define TYPE_IS_SERIALIZABLE(T) \
  void *operator,(const is_copy_serializable::inner&, const T&);

#define TEMPLATE_TYPE_IS_SERIALIZABLE(P,T)				\
  template <P> void *operator,(const is_copy_serializable::inner&, const T&);
#define TEMPLATE_TYPE_IS_SERIALIZABLE2(P1,P2,T1,T2)			\
  template <P1,P2> void *operator,(const is_copy_serializable::inner&, const T1,T2&);

namespace Realm {
  namespace Serialization {
    // there are three kinds of serializer we use and a single deserializer:
    //  a) FixedBufferSerializer - accepts a fixed-size buffer and fills into while preventing overflow
    //  b) DynamicBufferSerializer - serializes data into an automatically-regrowing buffer
    //  c) ByteCountSerializer - doesn't actually store data, just counts how big it would be
    //  d) FixedBufferDeserializer - deserializes from a fixed-size buffer

    class FixedBufferSerializer {
    public:
      FixedBufferSerializer(void);
      FixedBufferSerializer(void *buffer, size_t size);
      FixedBufferSerializer(ByteArray &array);
      ~FixedBufferSerializer(void);

      void reset(void *buffer, size_t size);
      void reset(ByteArray &array);

      ptrdiff_t bytes_left(void) const;

      bool enforce_alignment(size_t granularity);
      bool append_bytes(const void *data, size_t datalen);
      template <typename T> bool append_serializable(const T& data);

      template <typename T> bool operator<<(const T& val);
      template <typename T> bool operator&(const T& val);

    protected:
      char *pos;
      char *limit;
    };

    class DynamicBufferSerializer {
    public:
      DynamicBufferSerializer(size_t initial_size);
      ~DynamicBufferSerializer(void);

      void reset(void);

      size_t bytes_used(void) const;
      const void *get_buffer(void) const;
      void *detach_buffer(ptrdiff_t max_wasted_bytes = 0);
      ByteArray detach_bytearray(ptrdiff_t max_wasted_bytes = 0);

      bool enforce_alignment(size_t granularity);
      bool append_bytes(const void *data, size_t datalen);
      template <typename T> bool append_serializable(const T& data);

      template <typename T> bool operator<<(const T& val);
      template <typename T> bool operator&(const T& val);

    protected:
      char *base;
      char *pos;
      char *limit;
    };

    class ByteCountSerializer {
    public:
      ByteCountSerializer(void);
      ~ByteCountSerializer(void);

      size_t bytes_used(void) const;

      bool enforce_alignment(size_t granularity);
      bool append_bytes(const void *data, size_t datalen);
      template <typename T> bool append_serializable(const T& data);

      template <typename T> bool operator<<(const T& val);
      template <typename T> bool operator&(const T& val);

    protected:
      size_t count;
    };

    class FixedBufferDeserializer {
    public:
      FixedBufferDeserializer(const void *buffer, size_t size);
      FixedBufferDeserializer(const ByteArrayRef& array);
      ~FixedBufferDeserializer(void);

      ptrdiff_t bytes_left(void) const;

      bool enforce_alignment(size_t granularity);
      bool extract_bytes(void *data, size_t datalen);
      const void *peek_bytes(size_t datalen);
      template <typename T> bool extract_serializable(T& data);

      template <typename T> bool operator>>(T& val);
      template <typename T> bool operator&(const T& val);

    protected:
      const char *pos;
      const char *limit;
    };

    // defaults if custom serializers/deserializers are not defined
    template <typename S, typename T>
      bool serdez(S&, const T&); // not implemented
    
    template <typename S, typename T>
      inline bool serialize(S& s, const T& t) { return serdez(s, t); }
    
    template <typename S, typename T>
      inline bool deserialize(S& s, T& t) { return serdez(s, t); }

    template <typename T, bool IS_COPY_SERIALIZABLE> struct SerializationHelper;

    template <typename T>
    struct SerializationHelper<T, true> {
      // this is the special case where we can copy bits directly, even for vectors
      template <typename S>
      static bool serialize_scalar(S& s, const T& data);
      template <typename S>
      static bool deserialize_scalar(S& s, T& data);
      template <typename S>
      static bool serialize_vector(S& s, const std::vector<T>& v);
      template <typename S>
      static bool deserialize_vector(S& s, std::vector<T>& v);
      template <typename S, size_t Extent>
      static bool serialize_span(S& s, span<T, Extent> sp);
      template <typename S, size_t Extent>
      static bool deserialize_span(S& s, span<T, Extent>& sp);
    };

    template <typename T>
    struct SerializationHelper<T, false> {
      // in this case, we have to fall through to custom-defined serializers
      template <typename S>
      static bool serialize_scalar(S& s, const T& data);
      template <typename S>
      static bool deserialize_scalar(S& s, T& data);
      template <typename S>
      static bool serialize_vector(S& s, const std::vector<T>& v);
      template <typename S>
      static bool deserialize_vector(S& s, std::vector<T>& v);
      template <typename S, size_t Extent>
      static bool serialize_span(S& s, span<T, Extent> sp);
      // no deserialization of spans for non-copy-serializable types
    };

    // support for static arrays
    template <typename S, typename T, size_t N>
      bool serialize(S& s, T (&a)[N]);

    template <typename S, typename T, size_t N>
      bool deserialize(S& s, T (&a)[N]);

    // support for some STL containers
    template <typename S, typename T1, typename T2>
      bool serialize(S& s, const std::pair<T1, T2>& p);

    template <typename S, typename T1, typename T2>
      bool deserialize(S& s, std::pair<T1, T2>& p);

    template <typename S, typename T>
      bool serialize(S& s, const std::vector<T>& v);

    template <typename S, typename T>
      bool deserialize(S& s, std::vector<T>& v);

    template <typename S, typename T>
      bool serialize(S& s, const std::list<T>& l);

    template <typename S, typename T>
      bool deserialize(S& s, std::list<T>& l);

    template <typename S, typename T>
      bool serialize(S& s, const std::set<T>& ss);

    template <typename S, typename T>
      bool deserialize(S& s, std::set<T>& ss);

    template <typename S, typename T1, typename T2>
      bool serialize(S& s, const std::map<T1, T2>& m);

    template <typename S, typename T1, typename T2>
      bool deserialize(S& s, std::map<T1, T2>& m);

    template <typename S>
      bool serialize(S& s, const std::string& str);

    template <typename S>
      bool deserialize(S& s, std::string& str);

    template <typename S, typename T, size_t Extent>
      bool serialize(S& s, span<T, Extent> sp);

    template <typename S, typename T, size_t Extent>
      bool deserialize(S& s, span<T, Extent>& sp);

    template <typename T>
    class PolymorphicSerdezIntfc;

    template <typename T>
    class REALM_INTERNAL_API_EXTERNAL_LINKAGE PolymorphicSerdezHelper {
    public:
      // not templated because we have to get through a virtual method
      static bool serialize(FixedBufferSerializer& serializer, const T& obj);
      static bool serialize(DynamicBufferSerializer& serializer, const T& obj);
      static bool serialize(ByteCountSerializer& serializer, const T& obj);

      static T *deserialize_new(FixedBufferDeserializer& deserializer);

    protected:
      typedef unsigned TypeTag;
      struct SubclassMap {
	std::map<const char *, const PolymorphicSerdezIntfc<T> *> by_typename;
	std::map<TypeTag, const PolymorphicSerdezIntfc<T> *> by_tag;
      };

      friend class PolymorphicSerdezIntfc<T>;

      template <typename T1, typename T2>
      friend class PolymorphicSerdezSubclass;

      static SubclassMap& get_subclasses(void);
    };

    template <typename T>
    class PolymorphicSerdezIntfc {
    public:
      PolymorphicSerdezIntfc(const char *type_name);
      virtual ~PolymorphicSerdezIntfc(void);

      virtual bool serialize(FixedBufferSerializer& serializer, const T& obj) const = 0;
      virtual bool serialize(DynamicBufferSerializer& serializer, const T& obj) const = 0;
      virtual bool serialize(ByteCountSerializer& serializer, const T& obj) const = 0;
      
      virtual T *deserialize_new(FixedBufferDeserializer& deserializer) const = 0;

    protected:
      friend class PolymorphicSerdezHelper<T>;

      typename PolymorphicSerdezHelper<T>::TypeTag tag;
    };

    template <typename T1, typename T2>
    class PolymorphicSerdezSubclass : public PolymorphicSerdezIntfc<T1> {
    public:
      PolymorphicSerdezSubclass(void);
      
      virtual bool serialize(FixedBufferSerializer& serializer, const T1& obj) const;
      virtual bool serialize(DynamicBufferSerializer& serializer, const T1& obj) const;
      virtual bool serialize(ByteCountSerializer& serializer, const T1& obj) const;
      
      virtual T1 *deserialize_new(FixedBufferDeserializer& deserializer) const;
    };

  }; // namespace Serialization

}; // namespace Realm

// inlined method definitions
#include "realm/serialize.inl"

#endif // ifndef REALM_SERIALIZE_H
