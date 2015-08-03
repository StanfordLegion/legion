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

// a little helper class for storing a dynamically allocated byte array
//  an accessing it in various ways

#ifndef REALM_BYTEARRAY_H
#define REALM_BYTEARRAY_H

#include <stddef.h>

namespace Realm {

  class ByteArray {
  public:
    inline ByteArray(void);
    inline ByteArray(const void *copy_from, size_t copy_size);
    inline ByteArray(const ByteArray& copy_from);

    inline ~ByteArray(void);

    // copies the contents of the rhs ByteArray
    inline ByteArray& operator=(const ByteArray& copy_from);

    // swaps the contents of two ByteArrays - returns a reference to the first one
    // this allows you to transfer ownership of a byte array to a called function via:
    //   ByteArray().swap(old_array)
    inline ByteArray& swap(ByteArray& swap_with);

    // copy raw data in
    inline ByteArray& set(const void *copy_from, size_t copy_size);

    // give ownership of a buffer to a ByteArray
    inline ByteArray& attach(void *new_base, size_t new_size);

    // explicitly deallocate any held storage
    inline void clear(void);

    // extract the pointer from the ByteArray (caller assumes ownership)
    inline void *detach(void);

    // access to base pointer and size
    inline void *base(void);
    inline const void *base(void) const;

    inline size_t size(void) const;

    // helper to access bytes as typed references
    template <typename T>
    T& at(size_t offset);

    template <typename T>
    const T& at(size_t offset) const;

  protected:
    void *array_base;
    size_t array_size;
  };

  // support for realm-style serialization
  template <typename S>
    bool operator<<(S& serdez, const ByteArray& a);

  template <typename S>
    bool operator>>(S& serdez, ByteArray& a);

}; // namespace Realm

#include "bytearray.inl"

#endif
