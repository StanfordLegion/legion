/* Copyright 2026 Stanford University, NVIDIA Corporation
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

// Included from values.h - do not include this directly

// Useful for IDEs
#include "legion/api/values.h"

namespace Legion {

  //--------------------------------------------------------------------------
  inline DeferredValueRequest::DeferredValueRequest(
      Memory mem, size_t size, size_t align, const void* initial, bool escape)
    : field_size(size), alignment(align), initial_value(initial),
      is_exact(true), escaping(escape)
  //--------------------------------------------------------------------------
  {
    memory.exact = mem;
  }

  //--------------------------------------------------------------------------
  inline DeferredValueRequest::DeferredValueRequest(
      Memory::Kind kind, size_t size, size_t align, const void* initial,
      bool escape)
    : field_size(size), alignment(align), initial_value(initial),
      is_exact(false), escaping(escape)
  //--------------------------------------------------------------------------
  {
    memory.kind = kind;
  }

  //--------------------------------------------------------------------------
  template<typename T>
  inline DeferredValue<T>::DeferredValue(void) : UntypedDeferredValue()
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<typename T>
  inline DeferredValue<T>::DeferredValue(
      T initial_value, size_t alignment, Memory::Kind memory_kind,
      bool escaping)
    : UntypedDeferredValue(
          sizeof(T), memory_kind, &initial_value, alignment, escaping)
  //--------------------------------------------------------------------------
  {
    if (!Realm::AffineAccessor<T, 1, coord_t>::is_compatible(instance, 0))
      UntypedDeferredValue::report_incompatible_accessor("AffineAccessor");
    // We can make the accessor
    accessor = Realm::AffineAccessor<T, 1, coord_t>(instance, 0 /*field id*/);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  inline DeferredValue<T>::DeferredValue(
      T initial_value, Memory memory, size_t alignment, bool escaping)
    : UntypedDeferredValue(
          sizeof(T), memory, &initial_value, alignment, escaping)
  //--------------------------------------------------------------------------
  {
    if (!Realm::AffineAccessor<T, 1, coord_t>::is_compatible(instance, 0))
      UntypedDeferredValue::report_incompatible_accessor("AffineAccessor");
    // We can make the accessor
    accessor = Realm::AffineAccessor<T, 1, coord_t>(instance, 0 /*field id*/);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  __LEGION_CUDA_HD__ inline T DeferredValue<T>::read(void) const
  //--------------------------------------------------------------------------
  {
    return accessor.read(Point<1, coord_t>(0));
  }

  //--------------------------------------------------------------------------
  template<typename T>
  __LEGION_CUDA_HD__ inline void DeferredValue<T>::write(T value) const
  //--------------------------------------------------------------------------
  {
    accessor.write(Point<1, coord_t>(0), value);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  __LEGION_CUDA_HD__ inline T* DeferredValue<T>::ptr(void) const
  //--------------------------------------------------------------------------
  {
    return accessor.ptr(Point<1, coord_t>(0));
  }

  //--------------------------------------------------------------------------
  template<typename T>
  __LEGION_CUDA_HD__ inline T& DeferredValue<T>::ref(void) const
  //--------------------------------------------------------------------------
  {
    return accessor[Point<1, coord_t>(0)];
  }

  //--------------------------------------------------------------------------
  template<typename T>
  __LEGION_CUDA_HD__ inline DeferredValue<T>::operator T(void) const
  //--------------------------------------------------------------------------
  {
    return accessor[Point<1, coord_t>(0)];
  }

  //--------------------------------------------------------------------------
  template<typename T>
  __LEGION_CUDA_HD__ inline DeferredValue<T>& DeferredValue<T>::operator=(
      T value)
  //--------------------------------------------------------------------------
  {
    accessor[Point<1, coord_t>(0)] = value;
    return *this;
  }

  //--------------------------------------------------------------------------
  template<typename REDOP, bool EXCLUSIVE>
  inline DeferredReduction<REDOP, EXCLUSIVE>::DeferredReduction(
      size_t align, bool escaping)
    : DeferredValue<typename REDOP::RHS>(REDOP::identity, align, escaping)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<typename REDOP, bool EXCLUSIVE>
  __LEGION_CUDA_HD__ inline void DeferredReduction<REDOP, EXCLUSIVE>::reduce(
      typename REDOP::RHS value) const
  //--------------------------------------------------------------------------
  {
    REDOP::template fold<EXCLUSIVE>(
        this->accessor[Point<1, coord_t>(0)], value);
  }

  //--------------------------------------------------------------------------
  template<typename REDOP, bool EXCLUSIVE>
  __LEGION_CUDA_HD__ inline void
      DeferredReduction<REDOP, EXCLUSIVE>::operator<<=(
          typename REDOP::RHS value) const
  //--------------------------------------------------------------------------
  {
    REDOP::template fold<EXCLUSIVE>(
        this->accessor[Point<1, coord_t>(0)], value);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  inline UntypedDeferredValue::operator DeferredValue<T>(void) const
  //--------------------------------------------------------------------------
  {
    assert(sizeof(T) == field_size());
    DeferredValue<T> result;
    result.instance = instance;
    if (!Realm::AffineAccessor<T, 1, coord_t>::is_compatible(instance, 0))
      UntypedDeferredValue::report_incompatible_accessor("AffineAccessor");
    // We can make the accessor
    result.accessor =
        Realm::AffineAccessor<T, 1, coord_t>(instance, 0 /*field id*/);
    return result;
  }

  //--------------------------------------------------------------------------
  template<typename REDOP, bool EXCLUSIVE>
  inline UntypedDeferredValue::operator DeferredReduction<REDOP, EXCLUSIVE>(
      void) const
  //--------------------------------------------------------------------------
  {
    assert(sizeof(REDOP::RHS) == field_size());
    DeferredReduction<typename REDOP::RHS, EXCLUSIVE> result;
    result.instance = instance;
    if (!Realm::AffineAccessor<typename REDOP::RHS, 1, coord_t>::is_compatible(
            instance, 0))
      UntypedDeferredValue::report_incompatible_accessor("AffineAccessor");
    // We can make the accessor
    result.accessor = Realm::AffineAccessor<typename REDOP::RHS, 1, coord_t>(
        instance, 0 /*field id*/);
    return result;
  }

  //--------------------------------------------------------------------------
  inline size_t UntypedDeferredValue::field_size(void) const
  //--------------------------------------------------------------------------
  {
    if (instance.exists())
      return instance.get_layout()->bytes_used;
    else
      return 0;
  }

}  // namespace Legion
