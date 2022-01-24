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

// a little helper class for storing a dynamically allocated byte array
//  an accessing it in various ways

#ifndef REALM_BYTEARRAY_H
#define REALM_BYTEARRAY_H

#include <stddef.h>

namespace Realm {

  // a ByteArrayRef is a const reference to somebody else's storage (e.g. a
  //   ByteArray)
  class ByteArrayRef {
  public:
    ByteArrayRef(void);
    ByteArrayRef(const void *ref_base, size_t ref_size);
    ByteArrayRef(const ByteArrayRef& ref);

    // change what this ByteArrayRef refers to
    ByteArrayRef& changeref(const void *ref_base, size_t ref_size);

    // access to base pointer and size
    const void *base(void) const;

    size_t size(void) const;

    // helper to access bytes as typed references
    template <typename T>
    const T& at(size_t offset) const;

  protected:
    void *array_base;
    size_t array_size;
  };

  class ByteArray : public ByteArrayRef {
  public:
    ByteArray(void);
    ByteArray(const void *copy_from, size_t copy_size);
    ByteArray(const ByteArray& copy_from);

    // not actually a copy constructor!  blech...
    ByteArray(const ByteArrayRef& copy_from);

    ~ByteArray(void);

    // copies the contents of the rhs ByteArray (again two versions)
    ByteArray& operator=(const ByteArrayRef& copy_from);
    ByteArray& operator=(const ByteArray& copy_from);

    // swaps the contents of two ByteArrays - returns a reference to the first one
    // this allows you to transfer ownership of a byte array to a called function via:
    //   ByteArray().swap(old_array)
    ByteArray& swap(ByteArray& swap_with);

    // copy raw data in
    ByteArray& set(const void *copy_from, size_t copy_size);

    // access to base pointer and size
    // (const versions are redeclared due to some C++ weirdness)
    void *base(void);
    const void *base(void) const;

    // helper to access bytes as typed references
    template <typename T>
    T& at(size_t offset);

    template <typename T>
    const T& at(size_t offset) const;

    // give ownership of a buffer to a ByteArray
    ByteArray& attach(void *new_base, size_t new_size);

    // explicitly deallocate any held storage
    void clear(void);

    // extract the pointer from the ByteArray (caller assumes ownership)
    void *detach(void);

  protected:
    void make_copy(const void *copy_base, size_t copy_size);
  };

  // support for realm-style serialization
  template <typename S>
    bool serialize(S& serdez, const ByteArrayRef& a);

  template <typename S>
    bool serialize(S& serdez, const ByteArray& a);

  template <typename S>
    bool deserialize(S& serdez, ByteArray& a);

}; // namespace Realm

#include "realm/bytearray.inl"

#endif
