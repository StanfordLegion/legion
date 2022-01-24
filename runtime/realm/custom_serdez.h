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

// custom field serialization/deserialization for instance fields in Realm

#ifndef REALM_CUSTOM_SERDEZ_H
#define REALM_CUSTOM_SERDEZ_H

#include "realm/serialize.h"

#include <sys/types.h>

namespace Realm {

  // When Realm copies data between two instances, it is normally done by bit-wise copies of the data
  //  for each element.  There are two cases where this is either inefficient or incorrect:
  // 1) a complex field type might be compressible (e.g. have a large amount of "don't care" subfields),
  //   reducing the amount of network traffic required to move the instance (although with some increase
  //   in the CPU cost of moving it)
  // 2) a field may actually just hold a pointer to some more dynamic data structure - in this case,
  //   an instance copy needs to be a deep copy of the logical contents of the field

  // custom serdez objects are registered with the runtime at startup (for now) and associated with an ID
  //  that can be provided:
  // a) in CopySrcDstField objects in a copy operation (if one is provided for a field on the source side,
  //   the corresponding dst field also must have a custom serdez - it doesn't technically need to be
  //   the same one, but there's very few cases where having them different makes any sense)
  // b) in the deletion of an instance, allowing any data that was dynamically allocated during
  //   deserialization to be reclaimed

  // for now, a custom serdez is defined in the form of a class that looks like this:
  // (in the not-too-distant future, this will be replaced by a set of CodeDescriptor's)
#ifdef NOT_REALLY_CODE
  class CustomSerdezExample {
  public:
    typedef ... FIELD_TYPE;   // the "deserialized type" stored in the instance (e.g. Object * for a dynamic structure)

    // computes the number of bytes needed for the serialization of 'val'
    static size_t serialized_size(const FIELD_TYPE& val);

    // serializes 'val' into the provided buffer - no size is provided (serialized_size will be called first),
    //  but the function should return the bytes used
    static size_t serialize(const FIELD_TYPE& val, void *buffer);

    // deserializes the provided buffer into 'val' - the existing contents of 'val' are overwritten, but
    //  will have already been "destroyed" if this copy is overwriting previously valid data - this call
    //  should return the number of bytes consumed - note that the size of the buffer is not supplied (this
    //  means the serialization format must have some internal way of knowing how much to read)
    static size_t deserialize(FIELD_TYPE& val, const void *buffer);

    // destroys the contents of a field
    static void destroy(FIELD_TYPE& val);
  };
#endif
  template<typename T>
  class SerdezObject {
  public:
    typedef T* FIELD_TYPE;
    static const size_t MAX_SERIALIZED_SIZE = sizeof(T);

    static size_t serialized_size(const FIELD_TYPE& val) {
      return sizeof(T);
    }

    static size_t serialize(const FIELD_TYPE& val, void *buffer) {
      memcpy(buffer, val, sizeof(T));
      return sizeof(T);
    }

    static size_t deserialize(FIELD_TYPE& val, const void *buffer) {
      val = new T;
      memcpy(val, buffer, sizeof(T));
      return sizeof(T);
    }

    static void destroy(FIELD_TYPE& val) {
      delete val;
    }
  };


  // as a useful example, here's a templated serdez that works for anything that supports Realm's serialization
  //  framework
  template <typename T, size_t MAX_SER_SIZE = 4096>
  class SimpleSerdez {
  public:
    typedef T *FIELD_TYPE;  // field holds a pointer to the object
    static const size_t MAX_SERIALIZED_SIZE = MAX_SER_SIZE;

    static size_t serialized_size(const FIELD_TYPE& val)
    {
      Serialization::ByteCountSerializer bcs;
      bool ok = (bcs << (*val));
      assert(ok);
      return bcs.bytes_used();
    }

    static size_t serialize(const FIELD_TYPE& val, void *buffer)
    {
      // fake a max size, but it can't wrap the pointer
      size_t max_size = size_t(-1) - reinterpret_cast<size_t>(buffer);
      Serialization::FixedBufferSerializer fbs(buffer, max_size);
      bool ok = (fbs << *val);
      assert(ok);
      return max_size - fbs.bytes_left();  // because we didn't really tell it how many bytes we had
    }

    static size_t deserialize(FIELD_TYPE& val, const void *buffer)
    {
      val = new T;  // assumes existence of default constructor
      // fake a max size, but it can't wrap the pointer
      size_t max_size = size_t(-1) - reinterpret_cast<size_t>(buffer);
      Serialization::FixedBufferDeserializer fbd(buffer, max_size);
      bool ok = (fbd >> (*val));
      assert(ok);
      return max_size - fbd.bytes_left();
    }

    static void destroy(FIELD_TYPE& val)
    {
      delete val;
    }
  };
    
  // some template-based type-erasure stuff for registration for now
  typedef int CustomSerdezID;

  class CustomSerdezUntyped {
  public:
    size_t sizeof_field_type;
    size_t max_serialized_size;
  
    template <class SERDEZ>
    static CustomSerdezUntyped *create_custom_serdez(void);

    virtual ~CustomSerdezUntyped(void);

    virtual CustomSerdezUntyped *clone(void) const = 0;

    // each operator exists in two forms: single-element and strided-array-of-elements

    // computes the number of bytes needed for the serialization of 'val'
    virtual size_t serialized_size(const void *field_ptr) const = 0;
    virtual size_t serialized_size(const void *field_ptr, ptrdiff_t stride, size_t count) const = 0;

    // serializes 'val' into the provided buffer - no size is provided (serialized_size will be called first),
    //  but the function should return the bytes used
    virtual size_t serialize(const void *field_ptr, void *buffer) const = 0;
    virtual size_t serialize(const void *field_ptr, ptrdiff_t stride, size_t count,
			     void *buffer) const = 0;

    // deserializes the provided buffer into 'val' - the existing contents of 'val' are overwritten, but
    //  will have already been "destroyed" if this copy is overwriting previously valid data - this call
    //  should return the number of bytes consumed - note that the size of the buffer is not supplied (this
    //  means the serialization format must have some internal way of knowing how much to read)
    virtual size_t deserialize(void *field_ptr, const void *buffer) const = 0;
    virtual size_t deserialize(void *field_ptr, ptrdiff_t stride, size_t count,
			       const void *buffer) const = 0;

    // destroys the contents of a field
    virtual void destroy(void *field_ptr) const = 0;
    virtual void destroy(void *field_ptr, ptrdiff_t stride, size_t count) const = 0;
  };

  template <typename T> class CustomSerdezWrapper;


}; // namespace Realm

#include "realm/custom_serdez.inl"

#endif // ifndef REALM_REDOP_H


