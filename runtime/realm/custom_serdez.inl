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

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/custom_serdez.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class CustomSerdezUntyped
  //

  template <class SERDEZ>
  inline /*static*/ CustomSerdezUntyped *CustomSerdezUntyped::create_custom_serdez(void)
  {
    return new CustomSerdezWrapper<SERDEZ>;
  }

  inline CustomSerdezUntyped::~CustomSerdezUntyped(void)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // class CustomSerdezWrapper<T>
  //

  template <typename T> 
  class CustomSerdezWrapper : public CustomSerdezUntyped {
  public:
    CustomSerdezWrapper(void);

    virtual CustomSerdezUntyped *clone(void) const
    {
      return new CustomSerdezWrapper<T>;
    }

    // each operator exists in two forms: single-element and strided-array-of-elements

    // computes the number of bytes needed for the serialization of 'val'
    virtual size_t serialized_size(const void *field_ptr) const;
    virtual size_t serialized_size(const void *field_ptr, ptrdiff_t stride, size_t count) const;

    // serializes 'val' into the provided buffer - no size is provided (serialized_size will be called first),
    //  but the function should return the bytes used
    virtual size_t serialize(const void *field_ptr, void *buffer) const;
    virtual size_t serialize(const void *field_ptr, ptrdiff_t stride, size_t count,
			     void *buffer) const;

    // deserializes the provided buffer into 'val' - the existing contents of 'val' are overwritten, but
    //  will have already been "destroyed" if this copy is overwriting previously valid data - this call
    //  should return the number of bytes consumed - note that the size of the buffer is not supplied (this
    //  means the serialization format must have some internal way of knowing how much to read)
    virtual size_t deserialize(void *field_ptr, const void *buffer) const;
    virtual size_t deserialize(void *field_ptr, ptrdiff_t stride, size_t count,
			       const void *buffer) const;

    // destroys the contents of a field
    virtual void destroy(void *field_ptr) const;
    virtual void destroy(void *field_ptr, ptrdiff_t stride, size_t count) const;
  };

  template <typename T> 
  inline CustomSerdezWrapper<T>::CustomSerdezWrapper(void)
  {
    sizeof_field_type = sizeof(typename T::FIELD_TYPE);
    max_serialized_size = T::MAX_SERIALIZED_SIZE;
  }

  template <typename T> 
  inline size_t CustomSerdezWrapper<T>::serialized_size(const void *field_ptr) const
  {
    return T::serialized_size(*static_cast<const typename T::FIELD_TYPE *>(field_ptr));
  }

  template <typename T> 
  inline size_t CustomSerdezWrapper<T>::serialized_size(const void *field_ptr, ptrdiff_t stride, size_t count) const
  {
    size_t total = 0;
    for(size_t i = 0; i < count; i++) {
      total += T::serialized_size(*static_cast<const typename T::FIELD_TYPE *>(field_ptr));
      field_ptr = static_cast<const char *>(field_ptr) + stride;
    }
    return total;
  }

  // serializes 'val' into the provided buffer - no size is provided (serialized_size will be called first),
  //  but the function should return the bytes used
  template <typename T> 
  inline size_t CustomSerdezWrapper<T>::serialize(const void *field_ptr, void *buffer) const
  {
    return T::serialize(*static_cast<const typename T::FIELD_TYPE *>(field_ptr), buffer);
  }

  template <typename T> 
  inline size_t CustomSerdezWrapper<T>::serialize(const void *field_ptr, ptrdiff_t stride, size_t count,
						  void *buffer) const
  {
    size_t total = 0;
    for(size_t i = 0; i < count; i++) {
      size_t n = T::serialize(*static_cast<const typename T::FIELD_TYPE *>(field_ptr), buffer);
      field_ptr = static_cast<const char *>(field_ptr) + stride;
      buffer = static_cast<char *>(buffer) + n;
      total += n;
    }
    return total;
  }

  // deserializes the provided buffer into 'val' - the existing contents of 'val' are overwritten, but
  //  will have already been "destroyed" if this copy is overwriting previously valid data - this call
  //  should return the number of bytes consumed - note that the size of the buffer is not supplied (this
  //  means the serialization format must have some internal way of knowing how much to read)
  template <typename T> 
  inline size_t CustomSerdezWrapper<T>::deserialize(void *field_ptr, const void *buffer) const
  {
    return T::deserialize(*static_cast<typename T::FIELD_TYPE *>(field_ptr), buffer);
  }

  template <typename T> 
  inline size_t CustomSerdezWrapper<T>::deserialize(void *field_ptr, ptrdiff_t stride, size_t count,
						    const void *buffer) const
  {
    size_t total = 0;
    for(size_t i = 0; i < count; i++) {
      size_t n = T::deserialize(*static_cast<typename T::FIELD_TYPE *>(field_ptr), buffer);
      field_ptr = static_cast<char *>(field_ptr) + stride;
      buffer = static_cast<const char *>(buffer) + n;
      total += n;
    }
    return total;
  }

  // destroys the contents of a field
  template <typename T> 
  inline void CustomSerdezWrapper<T>::destroy(void *field_ptr) const
  {
    T::destroy(*static_cast<typename T::FIELD_TYPE *>(field_ptr));
  }

  template <typename T> 
  inline void CustomSerdezWrapper<T>::destroy(void *field_ptr, ptrdiff_t stride, size_t count) const
  {
    for(size_t i = 0; i < count; i++) {
      T::destroy(*static_cast<typename T::FIELD_TYPE *>(field_ptr));
      field_ptr = static_cast<char *>(field_ptr) + stride;
    }
  }


}; // namespace Realm


