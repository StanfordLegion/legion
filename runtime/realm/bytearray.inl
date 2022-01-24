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

// INCLDUED FROM bytearray.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/bytearray.h"

#include <string.h>
#include <stdlib.h>
#include <assert.h>

// for std::swap
#include <algorithm>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class ByteArrayRef
  //

  inline ByteArrayRef::ByteArrayRef(void)
    : array_base(0), array_size(0)
  {}

  inline ByteArrayRef::ByteArrayRef(const void *ref_base, size_t ref_size)
    : array_base(const_cast<void *>(ref_base)), array_size(ref_size)
  {}

  inline ByteArrayRef::ByteArrayRef(const ByteArrayRef& copy_from)
    : array_base(copy_from.array_base), array_size(copy_from.array_size)
  {}

  // change what this ByteArrayRef refers to
  inline ByteArrayRef& ByteArrayRef::changeref(const void *ref_base, size_t ref_size)
  {
    array_base = const_cast<void *>(ref_base);
    array_size = ref_size;
    return *this;
  }

  // access to base pointer and size
  inline const void *ByteArrayRef::base(void) const
  {
    return array_base;
  }

  inline size_t ByteArrayRef::size(void) const
  {
    return array_size;
  }

  // helper to access bytes as typed references
  template <typename T>
  const T& ByteArrayRef::at(size_t offset) const
  {
    // always range check?
    assert((offset + sizeof(T)) <= array_size);
    return *static_cast<const T *>(static_cast<char *>(array_base) + offset);
  }



  ////////////////////////////////////////////////////////////////////////
  //
  // class ByteArray
  //

  inline ByteArray::ByteArray(void)
  {}

  inline ByteArray::ByteArray(const void *copy_from, size_t copy_size)
  {
    make_copy(copy_from, copy_size);
  }

  inline ByteArray::ByteArray(const ByteArray& copy_from)
  {
    make_copy(copy_from.array_base, copy_from.array_size);
  }

  inline ByteArray::ByteArray(const ByteArrayRef& copy_from)
  {
    make_copy(copy_from.base(), copy_from.size());
  }

  inline ByteArray::~ByteArray(void)
  {
    clear();
  }

  // copies the contents of the rhs ByteArray
  inline ByteArray& ByteArray::operator=(const ByteArrayRef& copy_from)
  {
    if(this != &copy_from) {
      clear();  // throw away any data we had before
      make_copy(copy_from.base(), copy_from.size());
    }
    return *this;
  }

  inline ByteArray& ByteArray::operator=(const ByteArray& copy_from)
  {
    if(this != &copy_from) {
      clear();  // throw away any data we had before
      make_copy(copy_from.base(), copy_from.size());
    }
    return *this;
  }

  // swaps the contents of two ByteArrays - returns a reference to the first one
  // this allows you to transfer ownership of a byte array to a called function via:
  //   ByteArray().swap(old_array)
  inline ByteArray& ByteArray::swap(ByteArray& swap_with)
  {
    std::swap(array_base, swap_with.array_base);
    std::swap(array_size, swap_with.array_size);
    return *this;
  }

  // copy raw data in
  inline ByteArray& ByteArray::set(const void *copy_from, size_t copy_size)
  {
    clear();  // throw away any data we had before
    make_copy(copy_from, copy_size);
    return *this;
  }

  // access to base pointer and size
  inline void *ByteArray::base(void)
  {
    return array_base;
  }

  inline const void *ByteArray::base(void) const
  {
    return array_base;
  }

  // helper to access bytes as typed references
  template <typename T>
  T& ByteArray::at(size_t offset)
  {
    // always range check?
    assert((offset + sizeof(T)) <= array_size);
    return *static_cast<T *>(static_cast<char *>(array_base) + offset);
  }

  template <typename T>
  const T& ByteArray::at(size_t offset) const
  {
    // always range check?
    assert((offset + sizeof(T)) <= array_size);
    return *static_cast<const T *>(static_cast<char *>(array_base) + offset);
  }

  // give ownership of a buffer to a ByteArray
  inline ByteArray& ByteArray::attach(void *new_base, size_t new_size)
  {
    clear();  // throw away any data we had before
    if(new_size) {
      array_base = new_base;
      assert(array_base != 0);
      array_size = new_size;
    } else {
      // if we were given ownership of a 0-length allocation, free it rather than leaking it
      if(new_base)
	free(new_base);
    }
    return *this;
  }

  inline void ByteArray::make_copy(const void *copy_base, size_t copy_size)
  {
    if(copy_size) {
      array_base = malloc(copy_size);
      assert(array_base != 0);
      memcpy(array_base, copy_base, copy_size);
      array_size = copy_size;
    } else {
      array_base = 0;
      array_size = 0;
    }
  }

  // explicitly deallocate any held storage
  inline void ByteArray::clear(void)
  {
    if(array_size) {
      free(array_base);
      array_base = 0;
      array_size = 0;
    }
  }

  // extract the pointer from the ByteArray (caller assumes ownership)
  inline void *ByteArray::detach(void)
  {
    if(array_size) {
      void *retval = array_base;
      array_base = 0;
      array_size = 0;
      return retval;
    } else
      return 0;
  }

  // support for realm-style serialization
  template <typename S>
  bool serialize(S& serdez, const ByteArrayRef& a)
  {
    return((serdez << a.size()) &&
	   ((a.size() == 0) ||
	    serdez.append_bytes(a.base(), a.size())));
  }

  template <typename S>
  bool serialize(S& serdez, const ByteArray& a)
  {
    return serialize(serdez, static_cast<const ByteArrayRef&>(a));
  }

  template <typename S>
  bool deserialize(S& serdez, ByteArray& a)
  {
    size_t new_size;
    if(!(serdez >> new_size)) return false;
    void *new_base = 0;
    if(new_size) {
      new_base = malloc(new_size);
      assert(new_base != 0);
      if(!serdez.extract_bytes(new_base, new_size)) {
	free(new_base);  // no leaks plz
	return false;
      }
    }
    a.attach(new_base, new_size);
    return true;
  }

}; // namespace Realm
