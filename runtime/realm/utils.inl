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

// little helper utilities for Realm code
// none of this is Realm-specific, but it's put in the Realm namespace to
//  reduce the chance of conflicts

// nop, but helps IDEs
#include "realm/utils.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

namespace Realm {
    
  ////////////////////////////////////////////////////////////////////////
  //
  // class shortstringbuf<I,E>

  template <size_t I, size_t E>
  inline shortstringbuf<I,E>::shortstringbuf()
    : external_buffer(0)
    , external_buffer_size(0)
  {
    setp(internal_buffer, internal_buffer + INTERNAL_BUFFER_SIZE);
  }

  template <size_t I, size_t E>
  inline shortstringbuf<I,E>::~shortstringbuf()
  {
    if(external_buffer)
      free(external_buffer);
  }

  template <size_t I, size_t E>
  inline const char *shortstringbuf<I,E>::data() const
  {
    return (external_buffer ? external_buffer : internal_buffer);
  }

  template <size_t I, size_t E>
  inline size_t shortstringbuf<I,E>::size() const
  {
    return pptr() - data();
  }

  template <size_t I, size_t E>
  inline typename shortstringbuf<I,E>::int_type shortstringbuf<I,E>::overflow(typename shortstringbuf<I,E>::int_type c)
  {
    size_t curlen;
    if(external_buffer) {
      // grow existing external buffer
      curlen = pptr() - external_buffer;
      external_buffer_size = curlen * 2;
      char *new_buffer = (char *)malloc(external_buffer_size);
      assert(new_buffer != 0);
      memcpy(new_buffer, external_buffer, curlen);
      free(external_buffer);
      external_buffer = new_buffer;
    } else {
      // switch from internal to external
      curlen = pptr() - internal_buffer;
      external_buffer_size = INITIAL_EXTERNAL_BUFFER_SIZE;
      external_buffer = (char *)malloc(external_buffer_size);
      assert(external_buffer != 0);
      memcpy(external_buffer, internal_buffer, curlen);
    }
    if(c >= 0)
      external_buffer[curlen++] = c;
    setp(external_buffer + curlen, external_buffer + external_buffer_size);
    return 0;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DeferredConstructor<T>

  template <typename T>
  DeferredConstructor<T>::DeferredConstructor()
    : ptr(0)
  {}

  template <typename T>
  DeferredConstructor<T>::~DeferredConstructor()
  {
    if(ptr) ptr->~T();
  }

  template <typename T>
  T *DeferredConstructor<T>::construct()
  {
#ifdef DEBUG_REALM
    assert(!ptr);
#endif
    ptr = new(raw_storage) T();
    return ptr;
  }

  template <typename T>
  template <typename T1>
  T *DeferredConstructor<T>::construct(T1 arg1)
  {
#ifdef DEBUG_REALM
    assert(!ptr);
#endif
    ptr = new(raw_storage) T(arg1);
    return ptr;
  }

  template <typename T>
  T& DeferredConstructor<T>::operator*()
  {
#ifdef DEBUG_REALM
    assert(ptr != 0);
#endif
    return *ptr;
  }

  template <typename T>
  T *DeferredConstructor<T>::operator->()
  {
#ifdef DEBUG_REALM
    assert(ptr != 0);
#endif
    return ptr;
  }

  template <typename T>
  const T& DeferredConstructor<T>::operator*() const
  {
#ifdef DEBUG_REALM
    assert(ptr != 0);
#endif
    return *ptr;
  }

  template <typename T>
  const T *DeferredConstructor<T>::operator->() const
  {
#ifdef DEBUG_REALM
    assert(ptr != 0);
#endif
    return ptr;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // struct bitfield<_BITS, _SHIFT>

  template <unsigned _BITS, unsigned _SHIFT>
  template <typename T>
  inline /*static*/ T bitfield<_BITS, _SHIFT>::extract(T source)
  {
    T val = source;
    if(_SHIFT > 0)
      val >>= _SHIFT;
    if(_BITS < (8 * sizeof(T)))
      val &= ((T(1) << BITS) - 1);
    return val;
  }

  template <unsigned _BITS, unsigned _SHIFT>
  template <typename T>
  inline /*static*/ T bitfield<_BITS, _SHIFT>::insert(T target, T field)
  {
    if(_SHIFT > 0)
      field <<= _SHIFT;

    if(_BITS < (8 * sizeof(T))) {
      T mask = ((T(1) << _BITS) - 1);
      if(_SHIFT > 0)
	mask <<= _SHIFT;
      field &= mask;
      target &= ~mask;
      target |= field;
    } else
      target = field;

    return target;
  }

  template <unsigned _BITS, unsigned _SHIFT>
  template <typename T>
  inline /*static*/ T bitfield<_BITS, _SHIFT>::bit_or(T target, T field)
  {
    if(_BITS < (8 * sizeof(T))) {
      T mask = ((T(1) << _BITS) - 1);
      field &= mask;
    }

    if(_SHIFT > 0)
      field <<= _SHIFT;

    return (target | field);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class bitpack<T>

  template <typename T>
  inline bitpack<T>::bitpack()
  {
    // no initialization
  }

  template <typename T>
  inline bitpack<T>::bitpack(const bitpack<T>& copy_from)
    : value(copy_from.value)
  {}

  template <typename T>
  inline bitpack<T>::bitpack(T init_val)
    : value(init_val)
  {}

  template <typename T>
  inline bitpack<T>& bitpack<T>::operator=(const bitpack<T>& copy_from)
  {
    value = copy_from.value;
    return *this;
  }

  template <typename T>
  inline bitpack<T>& bitpack<T>::operator=(T new_val)
  {
    value = new_val;
    return *this;
  }

  template <typename T>
  inline bitpack<T>::operator T() const
  {
    return value;
  }

  template <typename T>
  template <typename BITFIELD>
  inline typename bitpack<T>::template bitsliceref<BITFIELD> bitpack<T>::slice()
  {
    return bitsliceref<BITFIELD>(value);
  }

  template <typename T>
  template <typename BITFIELD>
  inline typename bitpack<T>::template constbitsliceref<BITFIELD> bitpack<T>::slice() const
  {
    return constbitsliceref<BITFIELD>(value);
  }

  template <typename T>
  template <typename BITFIELD>
  inline typename bitpack<T>::template bitsliceref<BITFIELD> bitpack<T>::operator[](const BITFIELD& bitfield)
  {
    return bitsliceref<BITFIELD>(value);
  }

  template <typename T>
  template <typename BITFIELD>
  inline typename bitpack<T>::template constbitsliceref<BITFIELD> bitpack<T>::operator[](const BITFIELD& bitfield) const
  {
    return constbitsliceref<BITFIELD>(value);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class bitpack<T>::bitsliceref<BITSLICE>

  template <typename T>
  template <typename BITFIELD>
  inline bitpack<T>::bitsliceref<BITFIELD>::bitsliceref(T& _target)
    : target(_target)
  {}

  template <typename T>
  template <typename BITFIELD>
  inline bitpack<T>::bitsliceref<BITFIELD>::operator T() const
  {
    return BITFIELD::extract(target);
  }

  template <typename T>
  template <typename BITFIELD>
  inline typename bitpack<T>::template bitsliceref<BITFIELD>& bitpack<T>::bitsliceref<BITFIELD>::operator=(T field)
  {
    target = BITFIELD::insert(target, field);
    return *this;
  }

  template <typename T>
  template <typename BITFIELD>
  inline typename bitpack<T>::template bitsliceref<BITFIELD>& bitpack<T>::bitsliceref<BITFIELD>::operator|=(T field)
  {
    target = BITFIELD::bit_or(target, field);
    return *this;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class bitpack<T>::constbitsliceref<BITSLICE>

  template <typename T>
  template <typename BITFIELD>
  inline bitpack<T>::constbitsliceref<BITFIELD>::constbitsliceref(const T& _target)
    : target(_target)
  {}

  template <typename T>
  template <typename BITFIELD>
  inline bitpack<T>::constbitsliceref<BITFIELD>::operator T() const
  {
    return BITFIELD::extract(target);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PrettyVector<T>

  template <typename T>
  inline PrettyVector<T>::PrettyVector(const T *_data, size_t _size,
				       const char *_delim /*= ", "*/,
				       const char *_pfx /*= "["*/,
				       const char *_sfx /*= "]"*/)
    : data(_data), size(_size), delim(_delim), pfx(_pfx), sfx(_sfx)
  {}

  template <typename T>
  inline PrettyVector<T>::PrettyVector(const std::vector<T>& _v,
				       const char *_delim /*= ", "*/,
				       const char *_pfx /*= "["*/,
				       const char *_sfx /*= "]"*/)
    : data(_v.data()), size(_v.size()), delim(_delim), pfx(_pfx), sfx(_sfx)
  {}

  template <typename T>
  inline void PrettyVector<T>::print(std::ostream& os) const
  {
    os << pfx;
    if(size > 0) {
      os << data[0];
      for(size_t i = 1; i < size; i++)
	os << delim << data[i];
    }
    os << sfx;
  }

  template <typename T>
  inline std::ostream& operator<<(std::ostream& os, const PrettyVector<T>& pv)
  {
    pv.print(os);
    return os;
  }
  

}; // namespace Realm
