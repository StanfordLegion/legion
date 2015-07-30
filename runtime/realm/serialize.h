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

// serialization logic for packing/unpacking data in byte streams

#ifndef REALM_SERIALIZE_H
#define REALM_SERIALIZE_H

#include <cstddef>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <string>

// To add serializability for a new data type, do one of the following:
//
// 1) If your type can be copied bit for bit (i.e. C++11's is_trivially_copyable), just do:
//  TYPE_IS_SERIALIZABLE(my_type)
//
// 2) If your type is just a struct with a bunch of members, do:
//  template<typename S>
//  bool operator&(S& s, const my_type& t)
//  {
//    return (s & t.field1) && (s & t.field2) ... ;
//  }
//
// 3) If your type needs to serialize and deserialize in a complicated way, define them separately:
//
//  template<typename S>
//  bool operator<<(S& s, const my_type& t)
//  {
//    // serialization code here...
//  }
//
//  template<typename S>
//  bool operator>>(S& s, my_type& t)
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
struct is_directly_serializable {
public:
  class inner { public: inner(void) {} };
public:
  template <typename T> 
  struct test { static const bool value = sizeof(inner(),*(T*)0) != sizeof(char); };
};
template <typename T>
char operator,(const is_directly_serializable::inner&, const T&);

#define TYPE_IS_SERIALIZABLE(T) \
  void *operator,(const is_directly_serializable::inner&, const T&);

namespace Realm {
  namespace Serialization {
    // there are three kinds of serializer we use and a single deserializer:
    //  a) FixedBufferSerializer - accepts a fixed-size buffer and fills into while preventing overflow
    //  b) DynamicBufferSerializer - serializes data into an automatically-regrowing buffer
    //  c) ByteCountSerializer - doesn't actually store data, just counts how big it would be
    //  d) FixedBufferDeserializer - deserializes from a fixed-size buffer

    class FixedBufferSerializer {
    public:
      inline FixedBufferSerializer(void *buffer, size_t size);
      inline ~FixedBufferSerializer(void);

      inline ptrdiff_t bytes_left(void) const;

      inline bool enforce_alignment(size_t granularity);
      inline bool append_bytes(const void *data, size_t datalen);
      template <typename T> bool append_serializable(const T& data);

    protected:
      char *pos;
      char *limit;
    };

    class DynamicBufferSerializer {
    public:
      inline DynamicBufferSerializer(size_t initial_size);
      inline ~DynamicBufferSerializer(void);

      inline size_t bytes_used(void) const;
      inline const void *get_buffer(void) const;
      inline void *detach_buffer(size_t max_wasted_bytes = 0);

      inline bool enforce_alignment(size_t granularity);
      inline bool append_bytes(const void *data, size_t datalen);
      template <typename T> bool append_serializable(const T& data);

    protected:
      char *base;
      char *pos;
      char *limit;
    };

    class ByteCountSerializer {
    public:
      inline ByteCountSerializer(void);
      inline ~ByteCountSerializer(void);

      inline size_t bytes_used(void) const;

      inline bool enforce_alignment(size_t granularity);
      inline bool append_bytes(const void *data, size_t datalen);
      template <typename T> bool append_serializable(const T& data);

    protected:
      size_t count;
    };

    class FixedBufferDeserializer {
    public:
      inline FixedBufferDeserializer(const void *buffer, size_t size);
      inline ~FixedBufferDeserializer(void);

      inline ptrdiff_t bytes_left(void) const;

      inline bool enforce_alignment(size_t granularity);
      inline bool extract_bytes(void *data, size_t datalen);
      inline const void *peek_bytes(size_t datalen);
      template <typename T> bool extract_serializable(T& data);

    protected:
      const char *pos;
      const char *limit;
    };

    template <typename YOURTYPE>
      struct Your_Type_Needs_A_Custom_Serializer_Implementation {};

    template <typename T, bool CUSTOM_SERDEZ, bool IS_SERIALIZABLE>
    struct SerializationHelper {
      // the base case is when we can't copy bits and don't have an operator &
      // since we only got here because we didn't have << or >> either, we're toast

      template <typename S>
      static bool serialize(S& s, const T& data) 
      {
	return Your_Type_Needs_A_Custom_Serializer_Implementation<T>::FATAL_ERROR;
      }

      template <typename S>
      static bool deserialize(S& s, T& data) 
      {
	return Your_Type_Needs_A_Custom_Serializer_Implementation<T>::FATAL_ERROR;
      }

      // vectors might actually be ok - see if the inner type can be serialized somehow
      template <typename S>
      static bool serialize_vector(S& s, const std::vector<T>& v)
      {
	size_t c = v.size();
	if(!(s << c)) return false;
	for(size_t i = 0; i < c; i++)
	  if(!(s << v[i])) return false;
	return true;
      }

      template <typename S>
      static bool deserialize_vector(S& s, std::vector<T>& v)
      {
	size_t c;
	if(!(s >> c)) return false;
	// TODO: sanity-check size?
	v.resize(c);
	for(size_t i = 0; i < c; i++)
	  if(!(s >> v[i])) return false;
	return true;
      }
    };

    template <typename T, bool CUSTOM_SERDEZ>
    struct SerializationHelper<T, CUSTOM_SERDEZ, true> {
      // this is the special case where we can copy bits directly, even for vectors
      template <typename S>
      static bool serialize(S& s, const T& data) 
      {
	return (
		s.enforce_alignment(__alignof__(T)) &&
		s.template append_serializable<T>(data));
      }
      template <typename S>
      static bool deserialize(S& s, T& data)
      {
	return (
		s.enforce_alignment(__alignof__(T)) &&
		s.template extract_serializable<T>(data));
      }

      template <typename S>
      static bool serialize_vector(S& s, const std::vector<T>& v)
      {
	size_t c = v.size();
	return ((s << c) &&
		s.enforce_alignment(__alignof__(T)) &&
		s.append_bytes(&v[0], sizeof(T) * c));
      }

      template <typename S>
      static bool deserialize_vector(S& s, std::vector<T>& v)
      {
	size_t c;
	if(!(s >> c)) return false;
	// TODO: sanity-check size?
	v.resize(c);
	return (
		s.enforce_alignment(__alignof__(T)) &&
		s.extract_bytes(&v[0], sizeof(T) * c));
      }
    };

    template <typename T>
    struct SerializationHelper<T, true, false> {
      // this is the case where T has a custom operator &, so we can map both serialize
      //  and deserialize onto that
      template <typename S>
      static bool serialize(S& s, const T& data) 
      {
	return s & data;
      }
      template <typename S>
      static bool deserialize(S& s, T& data)
      {
	return s & data;
      }

      template <typename S>
      static bool serialize_vector(S& s, const std::vector<T>& v)
      {
	size_t c = v.size();
	if(!(s << c)) return false;
	for(size_t i = 0; i < c; i++)
	  if(!(s << v[i])) return false;
	return true;
      }

      template <typename S>
      static bool deserialize_vector(S& s, std::vector<T>& v)
      {
	size_t c;
	if(!(s >> c)) return false;
	// TODO: sanity-check size?
	v.resize(c);
	for(size_t i = 0; i < c; i++)
	  if(!(s >> v[i])) return false;
	return true;
      }
    };


    template <typename T>
    struct SerDezSplitter {
      static bool serdez(FixedBufferSerializer& s, const T& data) { return s << data; }
      static bool serdez(DynamicBufferSerializer& s, const T& data) { return s << data; }
      static bool serdez(ByteCountSerializer& s, const T& data) { return s << data; }
      static bool serdez(FixedBufferDeserializer& s, const T& data) { return s >> const_cast<T&>(data); }
    };

    // we use the & operator to allow a type to "distribute" both serialization and deserialization
    //  over its fields/etc. with a single method, but it has to serve two purposes:
    // a) a type-based split to << or >> for anything for which << and >> are defined, and
    // b) a fallback from << and >> when a custom & is defined
    //
    // to detect whether a custom & is defined, the default returns a funny bool2 type (so we can
    //  use template-fu that uses sizeof(returntype) that coerces back into a normal bool
    class bool2 {
    public:
      explicit bool2(bool b) : x(b), y(false) {}
      operator bool(void) const { return x; };
    protected:
      bool x, y;
    };

    template <typename S, typename T>
      bool2 operator&(S& s, const T& data) { return bool2(SerDezSplitter<T>::serdez(s,data)); }
    
    template <typename S, typename T>
      struct has_custom_serdez {
	static const bool value = (sizeof((*(S*)0) & (*(T*)0)) == sizeof(bool));
      };

    template <typename S, typename T>
      bool operator<<(S& s, const T& data)
    {
      return SerializationHelper<T,
	                         has_custom_serdez<S,T>::value,
	                         is_directly_serializable::test<T>::value>::serialize(s, data);
    }
    
    template <typename S, typename T>
      bool operator>>(S& s, T& data)
    {
      return SerializationHelper<T,
	                         has_custom_serdez<S,T>::value,
	                         is_directly_serializable::test<T>::value>::deserialize(s, data);
    }

    // support for some STL containers (vector is special because it can still be trivially_copyable)
    template <typename S, typename T>
      bool operator<<(S& s, const std::vector<T>& v) { return SerializationHelper<T, has_custom_serdez<S,T>::value,is_directly_serializable::test<T>::value>::serialize_vector(s, v); }

    template <typename S, typename T>
      bool operator>>(S& s, std::vector<T>& v) { return SerializationHelper<T, has_custom_serdez<S,T>::value,is_directly_serializable::test<T>::value>::deserialize_vector(s, v); }

    template <typename S, typename T>
      bool operator<<(S& s, const std::list<T>& l);

    template <typename S, typename T>
      bool operator>>(S& s, std::list<T>& l);

    template <typename S, typename T>
      bool operator<<(S& s, const std::set<T>& ss);

    template <typename S, typename T>
      bool operator>>(S& s, std::set<T>& ss);

    template <typename S, typename T1, typename T2>
      bool operator<<(S& s, const std::map<T1, T2>& m);

    template <typename S, typename T1, typename T2>
      bool operator>>(S& s, std::map<T1, T2>& m);

    template <typename S>
      bool operator<<(S& s, const std::string& str);

    template <typename S>
      bool operator>>(S& s, std::string& str);
  }; // namespace Serialization

}; // namespace Realm

// inlined method definitions
#include "serialize.inl"

#endif // ifndef REALM_SERIALIZE_H
