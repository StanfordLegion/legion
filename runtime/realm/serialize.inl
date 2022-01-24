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

// INCLDUED FROM serialize.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/serialize.h"

#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

#include <iostream>

// base integer/float types are all serializable
TYPE_IS_SERIALIZABLE(bool);
TYPE_IS_SERIALIZABLE(char);
TYPE_IS_SERIALIZABLE(unsigned char);
TYPE_IS_SERIALIZABLE(short);
TYPE_IS_SERIALIZABLE(unsigned short);
TYPE_IS_SERIALIZABLE(int);
TYPE_IS_SERIALIZABLE(unsigned int);
TYPE_IS_SERIALIZABLE(long);
TYPE_IS_SERIALIZABLE(unsigned long);
TYPE_IS_SERIALIZABLE(long long);
TYPE_IS_SERIALIZABLE(unsigned long long);
TYPE_IS_SERIALIZABLE(float);
TYPE_IS_SERIALIZABLE(double);

namespace Realm {
  namespace Serialization {

    template <typename T>
    static inline T align_offset(T offset, size_t granularity)
    {
      // guarantee remainder is calculated on unsigned value
      size_t extra = static_cast<size_t>(offset) % granularity;
      if(extra)
	offset += granularity - extra;
      return offset;
    }

    template <typename T>
    static inline T *align_pointer(T *ptr, size_t granularity)
    {
      uintptr_t i = reinterpret_cast<uintptr_t>(ptr);
      i = align_offset(i, granularity);
      return reinterpret_cast<T *>(i);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class FixedBufferSerializer
    //

    inline FixedBufferSerializer::FixedBufferSerializer(void)
      : pos(0)
      , limit(0)
    {}

    inline FixedBufferSerializer::FixedBufferSerializer(void *buffer, size_t size)
      : pos(static_cast<char *>(buffer)),
	limit(static_cast<char *>(buffer) + size)
    {}

    inline FixedBufferSerializer::FixedBufferSerializer(ByteArray& array)
      : pos(static_cast<char *>(array.base())),
	limit(static_cast<char *>(array.base()) + array.size())
    {}

    inline FixedBufferSerializer::~FixedBufferSerializer(void)
    {}

    inline void FixedBufferSerializer::reset(void *buffer, size_t size)
    {
      pos = static_cast<char *>(buffer);
      limit = static_cast<char *>(buffer) + size;
    }

    inline void FixedBufferSerializer::reset(ByteArray& array)
    {
      pos = static_cast<char *>(array.base());
      limit = static_cast<char *>(array.base()) + array.size();
    }

    inline ptrdiff_t FixedBufferSerializer::bytes_left(void) const
    {
      return limit - pos;
    }

    inline bool FixedBufferSerializer::enforce_alignment(size_t granularity)
    {
      // always move the pointer, but return false if we've overshot
      pos = align_pointer(pos, granularity);
      return (pos <= limit);
    }

    inline bool FixedBufferSerializer::append_bytes(const void *data, size_t datalen)
    {
      char *pos2 = pos + datalen;
      // only copy if it fits in the buffer
      bool ok_to_write = (pos2 <= limit);
      if(ok_to_write)
	memcpy(pos, data, datalen);
      pos = pos2;
      return ok_to_write;
    }

    template <typename T>
    bool FixedBufferSerializer::append_serializable(const T& data)
    {
      char *pos2 = pos + sizeof(T);
      // only copy if it fits in the buffer
      bool ok_to_write = (pos2 <= limit);
      if(ok_to_write)
	memcpy(pos, &data, sizeof(T));
      pos = pos2;
      return ok_to_write;
    }

    template <typename T>
    bool FixedBufferSerializer::operator<<(const T& data)
    {
      return SerializationHelper<T, is_copy_serializable::test<T>::value>::serialize_scalar(*this, data);
    }

    template <typename T>
    bool FixedBufferSerializer::operator&(const T& data)
    {
      return SerializationHelper<T, is_copy_serializable::test<T>::value>::serialize_scalar(*this, data);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class DynamicBufferSerializer
    //

    inline DynamicBufferSerializer::DynamicBufferSerializer(size_t initial_size)
    {
      // always allocate at least a little space
      if(initial_size < 16)
	initial_size = 16;
      base = static_cast<char *>(malloc(initial_size));
      assert(base != 0);
      pos = base;
      limit = base + initial_size;
    }

    inline DynamicBufferSerializer::~DynamicBufferSerializer(void)
    {
      // if we still own our pointer, release it
      if(base != 0) {
	free(base);
	base = 0;
      }
    }

    inline void DynamicBufferSerializer::reset(void)
    {
      // rewind position back to start of buffer
      pos = base;
    }

    inline size_t DynamicBufferSerializer::bytes_used(void) const
    {
      return pos - base;
    }

    inline const void *DynamicBufferSerializer::get_buffer(void) const
    {
      return base;
    }

    inline void *DynamicBufferSerializer::detach_buffer(ptrdiff_t max_wasted_bytes /*= 0*/)
    {
      assert(base != 0);
      assert(pos <= limit);

      // do we need to realloc to save space?
      if((max_wasted_bytes >= 0) &&
	 (size_t(limit - pos) > size_t(max_wasted_bytes))) {
	void *shrunk = realloc(base, pos - base);
	assert(shrunk != 0);
	base = 0;
	return shrunk;
      }

      // just return our raw array, clearing our pointer to it
      void *retval = base;
      base = 0;
      return retval;
    }

    inline ByteArray DynamicBufferSerializer::detach_bytearray(ptrdiff_t max_wasted_bytes /*= 0*/)
    {
      size_t size = pos - base;
      void *buffer = detach_buffer(max_wasted_bytes);
      return ByteArray().attach(buffer, size);
    }

    inline bool DynamicBufferSerializer::enforce_alignment(size_t granularity)
    {
      char *pos2 = align_pointer(pos, granularity);
      if(pos2 > limit) {
	size_t used = pos - base;
	size_t needed = pos2 - base;
	size_t size = limit - base;
	do { size <<= 1; } while(needed > size);
	char *newbase = static_cast<char *>(realloc(base, size));
	assert(newbase != 0);
	base = newbase;
	pos = newbase + used;
	limit = newbase + size;
	pos2 = pos + needed;
      }
      pos = pos2;
      return true;
    }

    inline bool DynamicBufferSerializer::append_bytes(const void *data, size_t datalen)
    {
      char *pos2 = pos + datalen;
      // resize as needed
      if(pos2 > limit) {
	size_t used = pos - base;
	size_t size = limit - base;
	do { size <<= 1; } while((used + datalen) > size);
	char *newbase = static_cast<char *>(realloc(base, size));
	assert(newbase != 0);
	base = newbase;
	pos = newbase + used;
	limit = newbase + size;
	pos2 = pos + datalen;
      }
      // copy always works now
      memcpy(pos, data, datalen);
      pos = pos2;
      return true;
    }

    template <typename T>
    bool DynamicBufferSerializer::append_serializable(const T& data)
    {
      char *pos2 = pos + sizeof(T);
      // resize as needed
      if(pos2 > limit) {
	size_t used = pos - base;
	size_t size = limit - base;
	do { size <<= 1; } while((used + sizeof(T)) > size);
	char *newbase = static_cast<char *>(realloc(base, size));
	assert(newbase != 0);
	base = newbase;
	pos = newbase + used;
	limit = newbase + size;
	pos2 = pos + sizeof(T);
      }
      // copy always works now
      memcpy(pos, &data, sizeof(T));
      pos = pos2;
      return true;
    }

    template <typename T>
    bool DynamicBufferSerializer::operator<<(const T& data)
    {
      return SerializationHelper<T, is_copy_serializable::test<T>::value>::serialize_scalar(*this, data);
    }

    template <typename T>
    bool DynamicBufferSerializer::operator&(const T& data)
    {
      return SerializationHelper<T, is_copy_serializable::test<T>::value>::serialize_scalar(*this, data);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class ByteCountSerializer
    //

    inline ByteCountSerializer::ByteCountSerializer(void)
      : count(0)
    {}

    inline ByteCountSerializer::~ByteCountSerializer(void)
    {}

    inline size_t ByteCountSerializer::bytes_used(void) const
    {
      return count;
    }

    inline bool ByteCountSerializer::enforce_alignment(size_t granularity)
    {
      count = align_offset(count, granularity);
      return true;
    }

    inline bool ByteCountSerializer::append_bytes(const void *data, size_t datalen)
    {
      count += datalen;
      return true;
    }

    template <typename T>
    bool ByteCountSerializer::append_serializable(const T& data)
    {
      count += sizeof(T);
      return true;
    }

    template <typename T>
    bool ByteCountSerializer::operator<<(const T& data)
    {
      return SerializationHelper<T, is_copy_serializable::test<T>::value>::serialize_scalar(*this, data);
    }

    template <typename T>
    bool ByteCountSerializer::operator&(const T& data)
    {
      return SerializationHelper<T, is_copy_serializable::test<T>::value>::serialize_scalar(*this, data);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class FixedBufferDeserializer
    //

    inline FixedBufferDeserializer::FixedBufferDeserializer(const void *buffer, size_t size)
      : pos(static_cast<const char *>(buffer)),
	limit(static_cast<const char *>(buffer) + size)
    {}

    inline FixedBufferDeserializer::FixedBufferDeserializer(const ByteArrayRef& array)
      : pos(static_cast<const char *>(array.base())),
	limit(static_cast<const char *>(array.base()) + array.size())
    {}

    inline FixedBufferDeserializer::~FixedBufferDeserializer(void)
    {}

    inline ptrdiff_t FixedBufferDeserializer::bytes_left(void) const
    {
      return limit - pos;
    }

    inline bool FixedBufferDeserializer::enforce_alignment(size_t granularity)
    {
      // always move the pointer, but return false if we've overshot
      pos = align_pointer(pos, granularity);
      return (pos <= limit);
    }

    inline bool FixedBufferDeserializer::extract_bytes(void *data, size_t datalen)
    {
      const char *pos2 = pos + datalen;
      // only copy if we have enough data
      bool ok_to_read = (pos2 <= limit);
      if(ok_to_read && data)
	memcpy(data, pos, datalen);
      pos = pos2;
      return ok_to_read;
    }

    inline const void *FixedBufferDeserializer::peek_bytes(size_t datalen)
    {
      const char *pos2 = pos + datalen;
      // only copy if we have enough data
      bool ok_to_read = (pos2 <= limit);
      if(ok_to_read)
	return pos;
      else
	return 0;
    }

    template <typename T>
    bool FixedBufferDeserializer::extract_serializable(T& data)
    {
      const char *pos2 = pos + sizeof(T);
      // only copy if we have enough data
      bool ok_to_read = (pos2 <= limit);
      if(ok_to_read)
	memcpy(&data, pos, sizeof(T));
      pos = pos2;
      return ok_to_read;
    }

    template <typename T>
    bool FixedBufferDeserializer::operator>>(T& data)
    {
      return SerializationHelper<T, is_copy_serializable::test<T>::value>::deserialize_scalar(*this, data);
    }

    template <typename T>
    bool FixedBufferDeserializer::operator&(const T& data)
    {
      return SerializationHelper<T, is_copy_serializable::test<T>::value>::deserialize_scalar(*this, const_cast<T&>(data));
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // SerializationHelper<T,true>
    //

    // this is the special case where we can copy bits directly, even for vectors

    template <typename T>
    template <typename S>
    /*static*/ bool SerializationHelper<T,true>::serialize_scalar(S& s, const T& data)
    {
      return (s.enforce_alignment(REALM_ALIGNOF(T)) &&
	      s.append_serializable(data));
    }

    template <typename T>
    template <typename S>
    /*static*/ bool SerializationHelper<T,true>::deserialize_scalar(S& s, T& data)
    {
      return (s.enforce_alignment(REALM_ALIGNOF(T)) &&
	      s.extract_serializable(data));
    }

    template <typename T>
    template <typename S>
    /*static*/ bool SerializationHelper<T,true>::serialize_vector(S& s, const std::vector<T>& v)
    {
      size_t c = v.size();
      return ((s << c) &&
	      ((c == 0) || (s.enforce_alignment(REALM_ALIGNOF(T)) &&
	                    s.append_bytes(&v[0], sizeof(T) * c))));
    }

    template <typename T>
    template <typename S>
    /*static*/ bool SerializationHelper<T,true>::deserialize_vector(S& s, std::vector<T>& v)
    {
      size_t c;
      if(!(s >> c)) return false;
      // TODO: sanity-check size?
      v.resize(c);
      return ((c == 0) || (s.enforce_alignment(REALM_ALIGNOF(T)) &&
	                   s.extract_bytes(&v[0], sizeof(T) * c)));
    }

    template <typename T>
    template <typename S, size_t Extent>
    /*static*/ bool SerializationHelper<T,true>::serialize_span(S& s, span<T, Extent> sp)
    {
      size_t c = sp.size();
      return ((s << c) &&
	      s.enforce_alignment(REALM_ALIGNOF(T)) &&
	      s.append_bytes(sp.data(), sizeof(T) * c));
    }

    template <typename T>
    template <typename S, size_t Extent>
    /*static*/ bool SerializationHelper<T,true>::deserialize_span(S& s, span<T, Extent>& sp)
    {
      size_t c;
      if(!(s >> c)) return false;
      // TODO: sanity-check size?
      if(!s.enforce_alignment(REALM_ALIGNOF(T))) return false;
      T *data = static_cast<T *>(s.peek_bytes(sizeof(T) * c));
      if(!data || !s.extract_bytes(0, sizeof(T) * c)) return false;
      sp = span<T, Extent>(data, c);
      return true;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // SerializationHelper<T,false>
    //

    // in this case, we have to fall through to custom-defined serializers

    template <typename T>
    template <typename S>
    /*static*/ bool SerializationHelper<T,false>::serialize_scalar(S& s, const T& data)
    {
      return serialize(s, data);
    }

    template <typename T>
    template <typename S>
    /*static*/ bool SerializationHelper<T,false>::deserialize_scalar(S& s, T& data)
    {
      return deserialize(s, data);
    }

    template <typename T>
    template <typename S>
    /*static*/ bool SerializationHelper<T,false>::serialize_vector(S& s, const std::vector<T>& v)
    {
      size_t c = v.size();
      if(!(s << c)) return false;
      for(size_t i = 0; i < c; i++)
	if(!serialize(s, v[i])) return false;
      return true;
    }

    template <typename T>
    template <typename S>
    /*static*/ bool SerializationHelper<T,false>::deserialize_vector(S& s, std::vector<T>& v)
    {
      size_t c;
      if(!(s >> c)) return false;
      // TODO: sanity-check size?
      v.resize(c);
      for(size_t i = 0; i < c; i++)
	if(!deserialize(s, v[i])) return false;
      return true;
    }

    template <typename T>
    template <typename S, size_t Extent>
    /*static*/ bool SerializationHelper<T,false>::serialize_span(S& s, span<T, Extent> sp)
    {
      size_t c = sp.size();
      if(!(s << c)) return false;
      for(size_t i = 0; i < c; i++)
	if(!serialize(s, sp[i])) return false;
      return true;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // STL container helpers
    //

    template <typename S, typename T, size_t N>
    bool serialize(S& s, T (&a)[N])
    {
      for(size_t i = 0; i < N; i++)
	if(!(s << a[i])) return false;
      return true;
    }

    template <typename S, typename T, size_t N>
    bool deserialize(S& s, T (&a)[N])
    {
      for(size_t i = 0; i < N; i++)
	if(!(s >> a[i])) return false;
      return true;
    }

    template <typename S, typename T1, typename T2>
    bool serialize(S& s, const std::pair<T1, T2>& p)
    {
      return (s << p.first) && (s << p.second);
    }

    template <typename S, typename T1, typename T2>
    bool deserialize(S& s, std::pair<T1, T2>& p)
    {
      return (s >> p.first) && (s >> p.second);
    }

    // vector is special because it can still be trivially_copyable
    template <typename S, typename T>
    bool serialize(S& s, const std::vector<T>& v)
    {
      return SerializationHelper<T, is_copy_serializable::test<T>::value>::serialize_vector(s, v);
    }

    template <typename S, typename T>
    bool deserialize(S& s, std::vector<T>& v)
    { 
      return SerializationHelper<T, is_copy_serializable::test<T>::value>::deserialize_vector(s, v);
    }

    template <typename S, typename T>
    inline bool serialize(S& s, const std::list<T>& l)
    {
      size_t len = l.size();
      if(!(s << len)) return false;
      for(typename std::list<T>::const_iterator it = l.begin();
	  it != l.end();
	  it++)
	if(!(s << *it)) return false;
      return true;
    }

    template <typename S, typename T>
    inline bool deserialize(S& s, std::list<T>& l)
    {
      size_t len;
      if(!(s >> len)) return false;
      l.clear(); // start from an empty list
      for(size_t i = 0; i < len; i++) {
	l.push_back(T()); // won't work if no default constructor for T
	if(!(s >> l.back())) return false;
      }
      return true;
    }

    template <typename S, typename T>
    inline bool serialize(S& s, const std::set<T>& ss)
    {
      size_t len = ss.size();
      if(!(s << len)) return false;
      for(typename std::set<T>::const_iterator it = ss.begin();
	  it != ss.end();
	  it++)
	if(!(s << *it)) return false;
      return true;
    }

    template <typename S, typename T>
    inline bool deserialize(S& s, std::set<T>& ss)
    {
      size_t len;
      if(!(s >> len)) return false;
      ss.clear(); // start from an empty set
      for(size_t i = 0; i < len; i++) {
	T v;  // won't work if no default constructor for T
	if(!(s >> v)) return false;
	ss.insert(v);
      }
      return true;
    }

    template <typename S, typename T1, typename T2>
    inline bool serialize(S& s, const std::map<T1,T2>& m)
    {
      size_t len = m.size();
      if(!(s << len)) return false;
      for(typename std::map<T1,T2>::const_iterator it = m.begin();
	  it != m.end();
	  it++) {
	if(!(s << it->first)) return false;
	if(!(s << it->second)) return false;
      }
      return true;
    }

    template <typename S, typename T1, typename T2>
    inline bool deserialize(S& s, std::map<T1,T2>& m)
    {
      size_t len;
      if(!(s >> len)) return false;
      m.clear(); // start from an empty map
      for(size_t i = 0; i < len; i++) {
	T1 k;
	T2 v;
	if(!(s >> k)) return false;
	if(!(s >> v)) return false;
	m[k] = v;
      }
      return true;
    }

    template <typename S>
    inline bool serialize(S& s, const std::string& str)
    {
      // strings are common, so use a shorter length - 32 bits is plenty
      unsigned len = str.length();
      if(!(s << len)) return false;
      // no alignment needed for character data
      return s.append_bytes(str.data(), len);
    }

    template <typename S>
    inline bool deserialize(S& s, std::string& str)
    {
      // strings are common, so use a shorter length - 32 bits is plenty
      unsigned len;
      if(!(s >> len)) return false;
      const void *p = s.peek_bytes(len);
      if(!p) return false;
      str.assign(static_cast<const char *>(p), len);
      return s.extract_bytes(0, len);
    }      

    // span works like vector...
    template <typename S, typename T, size_t Extent>
    bool serialize(S& s, span<T, Extent> sp)
    {
      return SerializationHelper<T, is_copy_serializable::test<T>::value>::serialize_span(s, sp);
    }

    // except deserialize is only going to work for copy-serializable things
    template <typename S, typename T, size_t Extent>
    bool deserialize(S& s, span<T, Extent>& sp)
    {
      return SerializationHelper<T, is_copy_serializable::test<T>::value>::deserialize_span(s, sp);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // PolymorphicSerdezHelper<T>
    //

    template <typename T>
    inline /*static*/ typename PolymorphicSerdezHelper<T>::SubclassMap& PolymorphicSerdezHelper<T>::get_subclasses(void)
    {
      static SubclassMap map;
      //std::cout << "returning " << &map << std::endl;
      return map;
    }

    template <typename T>
    inline /*static*/ bool PolymorphicSerdezHelper<T>::serialize(FixedBufferSerializer& serializer, const T& obj)
    {
      const char *type_name = typeid(obj).name();
      if(get_subclasses().by_typename.count(type_name) == 0) {
	std::cerr << "FATAL: class " << type_name << " not registered with serdez helper for " << typeid(T).name() << std::endl;
	assert(0);
      }
      const PolymorphicSerdezIntfc<T> *sc = get_subclasses().by_typename[type_name];
      return (serializer << sc->tag) && sc->serialize(serializer, obj);
    }

    template <typename T>
    inline /*static*/ bool PolymorphicSerdezHelper<T>::serialize(DynamicBufferSerializer& serializer, const T& obj)
    {
      const char *type_name = typeid(obj).name();
      if(get_subclasses().by_typename.count(type_name) == 0) {
	std::cerr << "FATAL: class " << type_name << " not registered with serdez helper for " << typeid(T).name() << std::endl;
	assert(0);
      }
      const PolymorphicSerdezIntfc<T> *sc = get_subclasses().by_typename[type_name];
      return (serializer << sc->tag) && sc->serialize(serializer, obj);
    }

    template <typename T>
    inline /*static*/ bool PolymorphicSerdezHelper<T>::serialize(ByteCountSerializer& serializer, const T& obj)
    {
      const char *type_name = typeid(obj).name();
      if(get_subclasses().by_typename.count(type_name) == 0) {
	std::cerr << "FATAL: class " << type_name << " not registered with serdez helper for " << typeid(T).name() << std::endl;
	assert(0);
      }
      const PolymorphicSerdezIntfc<T> *sc = get_subclasses().by_typename[type_name];
      return (serializer << sc->tag) && sc->serialize(serializer, obj);
    }

    template <typename T>
    inline /*static*/ T *PolymorphicSerdezHelper<T>::deserialize_new(FixedBufferDeserializer& deserializer)
    {
      TypeTag tag;
      if(!(deserializer >> tag)) return 0;
      if(get_subclasses().by_tag.count(tag) == 0) {
	std::cerr << "FATAL: unknown tag " << tag << " in serdez helper for " << typeid(T).name() << std::endl;
	assert(0);
      }
      return get_subclasses().by_tag[tag]->deserialize_new(deserializer);
    }
  

    ////////////////////////////////////////////////////////////////////////
    //
    // PolymorphicSerdezIntfc<T>
    //

    template <typename T>
    inline PolymorphicSerdezIntfc<T>::PolymorphicSerdezIntfc(const char *type_name)
    {
      tag = 0;
      const char *s = type_name;
      while(*s)
	tag = tag * 73 + *s++;
      //std::cout << "type: " << type_name << " -> tag = " << tag << std::endl;
      typename PolymorphicSerdezHelper<T>::SubclassMap& scmap = PolymorphicSerdezHelper<T>::get_subclasses();
      scmap.by_typename[type_name] = this;
      scmap.by_tag[tag] = this;
    }

    template <typename T>
    inline PolymorphicSerdezIntfc<T>::~PolymorphicSerdezIntfc(void) {}


    ////////////////////////////////////////////////////////////////////////
    //
    // PolymorphicSerdezSubclass<T1,T2>
    //

    template <typename T1, typename T2>
    inline PolymorphicSerdezSubclass<T1,T2>::PolymorphicSerdezSubclass(void)
      : PolymorphicSerdezIntfc<T1>(typeid(T2).name())
    {
      // TODO: some sort of template-based way to see if T2 defines deserialize_new?
    }

    template <typename T1, typename T2>
    inline bool PolymorphicSerdezSubclass<T1,T2>::serialize(FixedBufferSerializer& serializer, const T1& obj) const
    {
      return static_cast<const T2&>(obj).serialize(serializer);
    }

    template <typename T1, typename T2>
    inline bool PolymorphicSerdezSubclass<T1,T2>::serialize(DynamicBufferSerializer& serializer, const T1& obj) const
    {
      return static_cast<const T2&>(obj).serialize(serializer);
    }

    template <typename T1, typename T2>
    inline bool PolymorphicSerdezSubclass<T1,T2>::serialize(ByteCountSerializer& serializer, const T1& obj) const
    {
      return static_cast<const T2&>(obj).serialize(serializer);
    }
      
    template <typename T1, typename T2>
    inline T1 *PolymorphicSerdezSubclass<T1,T2>::deserialize_new(FixedBufferDeserializer& deserializer) const
    {
      return T2::deserialize_new(deserializer);
    }

    
  }; // namespace Serialization
}; // namespace Realm
