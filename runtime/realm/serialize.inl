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

// INCLDUED FROM serialize.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "serialize.h"

#include <string.h>
#include <stdlib.h>
#include <assert.h>

// base integer/float types are all serializable
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
    static inline T do_align(T ptr_or_int, size_t granularity)
    {
      size_t v = reinterpret_cast<size_t>(ptr_or_int);
      size_t offset = v % granularity;
      if(offset) {
	v += granularity - offset;
	return reinterpret_cast<T>(v);
      } else
	return ptr_or_int;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class FixedBufferSerializer
    //

    FixedBufferSerializer::FixedBufferSerializer(void *buffer, size_t size)
      : pos((char *)buffer), limit(((char *)buffer) + size)
    {}

    FixedBufferSerializer::FixedBufferSerializer(ByteArray& array)
      : pos((char *)(array.base())), limit(((char *)(array.base())) + array.size())
    {}

    FixedBufferSerializer::~FixedBufferSerializer(void)
    {}

    ptrdiff_t FixedBufferSerializer::bytes_left(void) const
    {
      return limit - pos;
    }

    bool FixedBufferSerializer::enforce_alignment(size_t granularity)
    {
      // always move the pointer, but return false if we've overshot
      pos = do_align(pos, granularity);
      return (pos <= limit);
    }

    bool FixedBufferSerializer::append_bytes(const void *data, size_t datalen)
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
	*(T *)pos = data;
      pos = pos2;
      return ok_to_write;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DynamicBufferSerializer
    //

    DynamicBufferSerializer::DynamicBufferSerializer(size_t initial_size)
    {
      // always allocate at least a little space
      if(initial_size < 16)
	initial_size = 16;
      base = (char *)malloc(initial_size);
      assert(base != 0);
      pos = base;
      limit = base + initial_size;
    }

    DynamicBufferSerializer::~DynamicBufferSerializer(void)
    {
      // if we still own our pointer, release it
      if(base != 0) {
	free(base);
	base = 0;
      }
    }

    size_t DynamicBufferSerializer::bytes_used(void) const
    {
      return pos - base;
    }

    const void *DynamicBufferSerializer::get_buffer(void) const
    {
      return base;
    }

    void *DynamicBufferSerializer::detach_buffer(ptrdiff_t max_wasted_bytes /*= 0*/)
    {
      assert(base != 0);
      assert(pos <= limit);

      // do we need to realloc to save space?
      if((max_wasted_bytes >= 0) &&
	 ((size_t)(limit - pos) > (size_t)max_wasted_bytes)) {
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

    ByteArray DynamicBufferSerializer::detach_bytearray(ptrdiff_t max_wasted_bytes /*= 0*/)
    {
      size_t size = pos - base;
      void *buffer = detach_buffer(max_wasted_bytes);
      return ByteArray().attach(buffer, size);
    }

    bool DynamicBufferSerializer::enforce_alignment(size_t granularity)
    {
      char *pos2 = do_align(pos, granularity);
      if(pos2 > limit) {
	size_t used = pos - base;
	size_t needed = pos2 - base;
	size_t size = limit - base;
	do { size <<= 1; } while(needed > size);
	char *newbase = (char *)realloc(base, size);
	assert(newbase != 0);
	base = newbase;
	pos = newbase + used;
	limit = newbase + size;
	pos2 = pos + needed;
      }
      pos = pos2;
      return true;
    }

    bool DynamicBufferSerializer::append_bytes(const void *data, size_t datalen)
    {
      char *pos2 = pos + datalen;
      // resize as needed
      if(pos2 > limit) {
	size_t used = pos - base;
	size_t size = limit - base;
	do { size <<= 1; } while((used + datalen) > size);
	char *newbase = (char *)realloc(base, size);
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
	char *newbase = (char *)realloc(base, size);
	assert(newbase != 0);
	base = newbase;
	pos = newbase + used;
	limit = newbase + size;
	pos2 = pos + sizeof(T);
      }
      // copy always works now
      *(T *)pos = data;
      pos = pos2;
      return true;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class ByteCountSerializer
    //

    ByteCountSerializer::ByteCountSerializer(void)
      : count(0)
    {}

    ByteCountSerializer::~ByteCountSerializer(void)
    {}

    size_t ByteCountSerializer::bytes_used(void) const
    {
      return count;
    }

    bool ByteCountSerializer::enforce_alignment(size_t granularity)
    {
      count = do_align(count, granularity);
      return true;
    }

    bool ByteCountSerializer::append_bytes(const void *data, size_t datalen)
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

    ////////////////////////////////////////////////////////////////////////
    //
    // class FixedBufferDeserializer
    //

    FixedBufferDeserializer::FixedBufferDeserializer(const void *buffer, size_t size)
      : pos((const char *)buffer), limit(((const char *)buffer) + size)
    {}

    FixedBufferDeserializer::FixedBufferDeserializer(const ByteArray& array)
      : pos((const char *)(array.base())), limit(((const char *)(array.base())) + array.size())
    {}

    FixedBufferDeserializer::~FixedBufferDeserializer(void)
    {}

    ptrdiff_t FixedBufferDeserializer::bytes_left(void) const
    {
      return limit - pos;
    }

    bool FixedBufferDeserializer::enforce_alignment(size_t granularity)
    {
      // always move the pointer, but return false if we've overshot
      pos = do_align(pos, granularity);
      return (pos <= limit);
    }

    bool FixedBufferDeserializer::extract_bytes(void *data, size_t datalen)
    {
      const char *pos2 = pos + datalen;
      // only copy if we have enough data
      bool ok_to_read = (pos2 <= limit);
      if(ok_to_read && data)
	memcpy(data, pos, datalen);
      pos = pos2;
      return ok_to_read;
    }

    const void *FixedBufferDeserializer::peek_bytes(size_t datalen)
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
	data = *(const T *)pos;
      pos = pos2;
      return ok_to_read;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // STL container helpers (except std::vector)
    //

    template <typename S, typename T>
    inline bool operator<<(S& s, const std::list<T>& l)
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
    inline bool operator>>(S& s, std::list<T>& l)
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
    inline bool operator<<(S& s, const std::set<T>& ss)
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
    inline bool operator>>(S& s, std::set<T>& ss)
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
    inline bool operator<<(S& s, const std::map<T1,T2>& m)
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
    inline bool operator>>(S& s, std::map<T1,T2>& m)
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
    inline bool operator<<(S& s, const std::string& str)
    {
      // strings are common, so use a shorter length - 32 bits is plenty
      unsigned len = str.length();
      if(!(s << len)) return false;
      // no alignment needed for character data
      return s.append_bytes(str.data(), len);
    }

    template <typename S>
    inline bool operator>>(S& s, std::string& str)
    {
      // strings are common, so use a shorter length - 32 bits is plenty
      unsigned len;
      if(!(s >> len)) return false;
      const void *p = s.peek_bytes(len);
      if(!p) return false;
      str.assign((const char *)p, len);
      return s.extract_bytes(0, len);
    }      
    
  }; // namespace Serialization
}; // namespace Realm
