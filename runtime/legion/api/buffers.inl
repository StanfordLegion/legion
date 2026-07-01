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

// Included from buffers.h - do not include this directly

// Useful for IDEs
#include "legion/api/buffers.h"

namespace Legion {

  // Some helper methods for accessors and deferred buffers
  namespace Internal {
    template<int N, typename T>
    __LEGION_CUDA_HD__ static inline bool is_dense_layout(
        const Rect<N, T>& bounds, const size_t strides[N], size_t field_size)
    {
      ptrdiff_t exp_offset = field_size;
      int used_mask = 0;  // keep track of the dimensions we've already matched
      static_assert((N <= (8 * sizeof(used_mask))), "Mask dim exceeded");
      for (int i = 0; i < N; i++)
      {
        bool found = false;
        for (int j = 0; j < N; j++)
        {
          if ((used_mask >> j) & 1)
            continue;
          if (strides[j] != exp_offset)
          {
            // Mask off any dimensions with stride 0
            if (strides[j] == 0)
            {
              if (bounds.lo[j] != bounds.hi[j])
                return false;
              used_mask |= (1 << j);
              if (++i == N)
              {
                found = true;
                break;
              }
            }
            continue;
          }
          found = true;
          // It's possible other dimensions can have the same strides if
          // there are multiple dimensions with extents of size 1. At most
          // one dimension can have an extent >1 though
          int nontrivial = (bounds.lo[j] < bounds.hi[j]) ? j : -1;
          for (int k = j + 1; k < N; k++)
          {
            if ((used_mask >> k) & 1)
              continue;
            if (strides[k] == exp_offset)
            {
              if (bounds.lo[k] < bounds.hi[k])
              {
                // if we already saw a non-trivial dimension this is bad
                if (nontrivial >= 0)
                  return false;
                else
                  nontrivial = k;
              }
              used_mask |= (1 << k);
              i++;
            }
          }
          used_mask |= (1 << j);
          if (nontrivial >= 0)
            exp_offset *= (bounds.hi[nontrivial] - bounds.lo[nontrivial] + 1);
          break;
        }
        if (!found)
          return false;
      }
      return true;
    }

    // Same method as above but for realm points from affine accessors
    template<int N, typename T>
    __LEGION_CUDA_HD__ static inline bool is_dense_layout(
        const Rect<N, T>& bounds, const Realm::Point<N, size_t>& strides,
        size_t field_size)
    {
      size_t exp_offset = field_size;
      int used_mask = 0;  // keep track of the dimensions we've already matched
      static_assert((N <= (8 * sizeof(used_mask))), "Mask dim exceeded");
      for (int i = 0; i < N; i++)
      {
        bool found = false;
        for (int j = 0; j < N; j++)
        {
          if ((used_mask >> j) & 1)
            continue;
          if (strides[j] != exp_offset)
          {
            // Mask off any dimensions with stride 0
            if (strides[j] == 0)
            {
              if (bounds.lo[j] != bounds.hi[j])
                return false;
              used_mask |= (1 << j);
              if (++i == N)
              {
                found = true;
                break;
              }
            }
            continue;
          }
          found = true;
          // It's possible other dimensions can have the same strides if
          // there are multiple dimensions with extents of size 1. At most
          // one dimension can have an extent >1 though
          int nontrivial = (bounds.lo[j] < bounds.hi[j]) ? j : -1;
          for (int k = j + 1; k < N; k++)
          {
            if ((used_mask >> k) & 1)
              continue;
            if (strides[k] == exp_offset)
            {
              if (bounds.lo[k] < bounds.hi[k])
              {
                // if we already saw a non-trivial dimension this is bad
                if (nontrivial >= 0)
                  return false;
                else
                  nontrivial = k;
              }
              used_mask |= (1 << k);
              i++;
            }
          }
          used_mask |= (1 << j);
          if (nontrivial >= 0)
            exp_offset *= (bounds.hi[nontrivial] - bounds.lo[nontrivial] + 1);
          break;
        }
        if (!found)
          return false;
      }
      return true;
    }
  }  // namespace Internal

  //--------------------------------------------------------------------------
  inline DeferredBufferRequest::DeferredBufferRequest(
      Memory mem, const Domain& domain, size_t size, size_t align,
      bool fortran_order_dims, const void* initial, bool escape)
    : field_size(size), alignment(align), initial_value(initial),
      is_exact(true), is_value(true), escaping(escape)
  //--------------------------------------------------------------------------
  {
    memory.exact = mem;
    bounds.value = domain;
    dim_order.resize(domain.get_dim());
    if (fortran_order_dims)
    {
      for (unsigned idx = 0; idx < dim_order.size(); idx++)
        dim_order[idx] = static_cast<DimensionKind>(
            static_cast<unsigned>(LEGION_DIM_X) + idx);
    }
    else
    {
      for (unsigned idx = 0; idx < dim_order.size(); idx++)
        dim_order[idx] = static_cast<DimensionKind>(
            static_cast<unsigned>(LEGION_DIM_X) + dim_order.size() - (idx + 1));
    }
  }

  //--------------------------------------------------------------------------
  inline DeferredBufferRequest::DeferredBufferRequest(
      Memory::Kind kind, const Domain& domain, size_t size, size_t align,
      bool fortran_order_dims, const void* initial, bool escape)
    : field_size(size), alignment(align), initial_value(initial),
      is_exact(false), is_value(true), escaping(escape)
  //--------------------------------------------------------------------------
  {
    memory.kind = kind;
    bounds.value = domain;
    dim_order.resize(domain.get_dim());
    if (fortran_order_dims)
    {
      for (unsigned idx = 0; idx < dim_order.size(); idx++)
        dim_order[idx] = static_cast<DimensionKind>(
            static_cast<unsigned>(LEGION_DIM_X) + idx);
    }
    else
    {
      for (unsigned idx = 0; idx < dim_order.size(); idx++)
        dim_order[idx] = static_cast<DimensionKind>(
            static_cast<unsigned>(LEGION_DIM_X) + dim_order.size() - (idx + 1));
    }
  }

  //--------------------------------------------------------------------------
  inline DeferredBufferRequest::DeferredBufferRequest(
      Memory mem, const IndexSpace& space, size_t size, size_t align,
      bool fortran_order_dims, const void* initial, bool escape)
    : field_size(size), alignment(align), initial_value(initial),
      is_exact(true), is_value(false), escaping(escape)
  //--------------------------------------------------------------------------
  {
    memory.exact = mem;
    bounds.name = space;
    dim_order.resize(space.get_dim());
    if (fortran_order_dims)
    {
      for (unsigned idx = 0; idx < dim_order.size(); idx++)
        dim_order[idx] = static_cast<DimensionKind>(
            static_cast<unsigned>(LEGION_DIM_X) + idx);
    }
    else
    {
      for (unsigned idx = 0; idx < dim_order.size(); idx++)
        dim_order[idx] = static_cast<DimensionKind>(
            static_cast<unsigned>(LEGION_DIM_X) + dim_order.size() - (idx + 1));
    }
  }

  //--------------------------------------------------------------------------
  inline DeferredBufferRequest::DeferredBufferRequest(
      Memory::Kind kind, const IndexSpace& space, size_t size, size_t align,
      bool fortran_order_dims, const void* initial, bool escape)
    : field_size(size), alignment(align), initial_value(initial),
      is_exact(false), is_value(false), escaping(escape)
  //--------------------------------------------------------------------------
  {
    memory.kind = kind;
    bounds.name = space;
    dim_order.resize(space.get_dim());
    if (fortran_order_dims)
    {
      for (unsigned idx = 0; idx < dim_order.size(); idx++)
        dim_order[idx] = static_cast<DimensionKind>(
            static_cast<unsigned>(LEGION_DIM_X) + idx);
    }
    else
    {
      for (unsigned idx = 0; idx < dim_order.size(); idx++)
        dim_order[idx] = static_cast<DimensionKind>(
            static_cast<unsigned>(LEGION_DIM_X) + dim_order.size() - (idx + 1));
    }
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  inline DeferredBuffer<FT, N, T, CB>::DeferredBuffer(void)
    : instance(Realm::RegionInstance::NO_INST)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  inline DeferredBuffer<FT, N, T, CB>::DeferredBuffer(
      Memory::Kind kind, const Domain& space,
      const FT* initial_value /* = nullptr*/,
      size_t alignment /* = alignof(FT)*/, bool fortran_order_dims /* = false*/,
      bool escaping)
  //--------------------------------------------------------------------------
  {
    DeferredBufferRequest request(
        kind, space, sizeof(FT), alignment, fortran_order_dims, initial_value,
        escaping);
    *this = UntypedDeferredBuffer<T>::allocate_buffer(request);
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  inline DeferredBuffer<FT, N, T, CB>::DeferredBuffer(
      const Rect<N, T>& rect, Memory::Kind kind,
      const FT* initial_value /*= nullptr*/,
      size_t alignment /* = alignof(FT)*/, bool fortran_order_dims /*= false*/,
      bool escaping)
  //--------------------------------------------------------------------------
  {
    DeferredBufferRequest request(
        kind, Domain(rect), sizeof(FT), alignment, fortran_order_dims,
        initial_value, escaping);
    *this = UntypedDeferredBuffer<T>::allocate_buffer(request);
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  inline DeferredBuffer<FT, N, T, CB>::DeferredBuffer(
      Memory memory, const Domain& space,
      const FT* initial_value /* = nullptr*/,
      size_t alignment /* = alignof(FT)*/, bool fortran_order_dims /* = false*/,
      bool escaping)
  //--------------------------------------------------------------------------
  {
    DeferredBufferRequest request(
        memory, space, sizeof(FT), alignment, fortran_order_dims, initial_value,
        escaping);
    *this = UntypedDeferredBuffer<T>::allocate_buffer(request);
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  inline DeferredBuffer<FT, N, T, CB>::DeferredBuffer(
      const Rect<N, T>& rect, Memory memory,
      const FT* initial_value /*= nullptr*/,
      size_t alignment /* = alignof(FT)*/, bool fortran_order_dims /*= false*/,
      bool escaping)
  //--------------------------------------------------------------------------
  {
    DeferredBufferRequest request(
        memory, Domain(rect), sizeof(FT), alignment, fortran_order_dims,
        initial_value, escaping);
    *this = UntypedDeferredBuffer<T>::allocate_buffer(request);
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  inline DeferredBuffer<FT, N, T, CB>::DeferredBuffer(
      Memory::Kind kind, const Domain& space,
      std::array<DimensionKind, N> _ordering,
      const FT* initial_value /* = nullptr*/,
      size_t _alignment /* = alignof(FT)*/, bool escaping)
    : ordering(_ordering), alignment(_alignment)
  //--------------------------------------------------------------------------
  {
    DeferredBufferRequest request(kind, space, sizeof(FT), alignment);
    request.escaping = escaping;
    if (!ordering.empty())
      request.dim_order.insert(
          request.dim_order.end(), ordering.begin(), ordering.end());
    request.initial_value = initial_value;
    *this = UntypedDeferredBuffer<T>::allocate_buffer(request);
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  inline DeferredBuffer<FT, N, T, CB>::DeferredBuffer(
      const Rect<N, T>& rect, Memory::Kind kind,
      std::array<DimensionKind, N> _ordering,
      const FT* initial_value /*= nullptr*/,
      size_t _alignment /* = alignof(FT)*/, bool escaping)
    : ordering(_ordering), alignment(_alignment)
  //--------------------------------------------------------------------------
  {
    DeferredBufferRequest request(kind, Domain(rect), sizeof(FT), alignment);
    request.escaping = escaping;
    if (!ordering.empty())
    {
      request.dim_order.clear();
      request.dim_order.insert(
          request.dim_order.end(), ordering.begin(), ordering.end());
    }
    request.initial_value = initial_value;
    *this = UntypedDeferredBuffer<T>::allocate_buffer(request);
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  inline DeferredBuffer<FT, N, T, CB>::DeferredBuffer(
      Memory memory, const Domain& space,
      std::array<DimensionKind, N> _ordering,
      const FT* initial_value /* = nullptr*/,
      size_t _alignment /* = alignof(FT)*/, bool escaping)
    : ordering(_ordering), alignment(_alignment)
  //--------------------------------------------------------------------------
  {
    DeferredBufferRequest request(memory, space, sizeof(FT), alignment);
    request.escaping = escaping;
    if (!ordering.empty())
    {
      request.dim_order.clear();
      request.dim_order.insert(
          request.dim_order.end(), ordering.begin(), ordering.end());
    }
    request.initial_value = initial_value;
    *this = UntypedDeferredBuffer<T>::allocate_buffer(request);
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  inline DeferredBuffer<FT, N, T, CB>::DeferredBuffer(
      const Rect<N, T>& rect, Memory memory,
      std::array<DimensionKind, N> _ordering,
      const FT* initial_value /*= nullptr*/,
      size_t _alignment /* = alignof(FT)*/, bool escaping)
    : ordering(_ordering), alignment(_alignment)
  //--------------------------------------------------------------------------
  {
    DeferredBufferRequest request(memory, Domain(rect), sizeof(FT), alignment);
    request.escaping = escaping;
    if (!ordering.empty())
    {
      request.dim_order.clear();
      request.dim_order.insert(
          request.dim_order.end(), ordering.begin(), ordering.end());
    }
    request.initial_value = initial_value;
    *this = UntypedDeferredBuffer<T>::allocate_buffer(request);
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  __LEGION_CUDA_HD__ inline FT DeferredBuffer<FT, N, T, CB>::read(
      const Point<N, T>& p) const
  //--------------------------------------------------------------------------
  {
    assert(!CB || bounds.contains(p));
    return accessor.read(p);
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  __LEGION_CUDA_HD__ inline void DeferredBuffer<FT, N, T, CB>::write(
      const Point<N, T>& p, FT value) const
  //--------------------------------------------------------------------------
  {
    assert(!CB || bounds.contains(p));
    accessor.write(p, value);
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  __LEGION_CUDA_HD__ inline FT* DeferredBuffer<FT, N, T, CB>::ptr(
      const Point<N, T>& p) const
  //--------------------------------------------------------------------------
  {
    assert(!CB || bounds.contains(p));
    return accessor.ptr(p);
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  __LEGION_CUDA_HD__ inline FT* DeferredBuffer<FT, N, T, CB>::ptr(
      const Rect<N, T>& r) const
  //--------------------------------------------------------------------------
  {
    assert(!CB || bounds.contains(r));
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    assert(Internal::is_dense_layout(r, accessor.strides, sizeof(FT)));
#else
    if (!Internal::is_dense_layout(r, accessor.strides, sizeof(FT)))
      UntypedDeferredBuffer<T>::report_nondense_rect();
#endif
    return accessor.ptr(r.lo);
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  __LEGION_CUDA_HD__ inline FT* DeferredBuffer<FT, N, T, CB>::ptr(
      const Rect<N, T>& r, size_t strides[N]) const
  //--------------------------------------------------------------------------
  {
    assert(!CB || bounds.contains(r));
    for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / sizeof(FT);
    return accessor.ptr(r.lo);
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  __LEGION_CUDA_HD__ inline FT& DeferredBuffer<FT, N, T, CB>::operator[](
      const Point<N, T>& p) const
  //--------------------------------------------------------------------------
  {
    assert(!CB || bounds.contains(p));
    return accessor[p];
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  inline void DeferredBuffer<FT, N, T, CB>::destroy(Realm::Event precondition)
  //--------------------------------------------------------------------------
  {
    UntypedDeferredBuffer<T> untyped = *this;
    untyped.destroy(precondition);
    instance = Realm::RegionInstance::NO_INST;
    accessor.reset();
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  __LEGION_CUDA_HD__ inline Realm::RegionInstance
      DeferredBuffer<FT, N, T, CB>::get_instance(void) const
  //--------------------------------------------------------------------------
  {
    return instance;
  }

  //--------------------------------------------------------------------------
  template<typename FT, int N, typename T, bool CB>
  __LEGION_CUDA_HD__ inline Rect<N, T> DeferredBuffer<FT, N, T, CB>::get_bounds(
      void) const
  //--------------------------------------------------------------------------
  {
    return bounds;
  }

  //--------------------------------------------------------------------------
  template<typename T>
  UntypedDeferredBuffer<T>::UntypedDeferredBuffer(void)
    : instance(Realm::RegionInstance::NO_INST), field_size(0), dims(0)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<typename T>
  UntypedDeferredBuffer<T>::UntypedDeferredBuffer(
      size_t fs, int d, Memory::Kind memkind, IndexSpace space,
      const void* initial_value, size_t alignment, bool fortran_order_dims,
      bool escaping)
    : field_size(fs), dims(d)
  //--------------------------------------------------------------------------
  {
    DeferredBufferRequest request(
        memkind, space, fs, alignment, fortran_order_dims, initial_value,
        escaping);
    *this = UntypedDeferredBuffer<T>::allocate_buffer(request);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  UntypedDeferredBuffer<T>::UntypedDeferredBuffer(
      size_t fs, int d, Memory::Kind memkind, const Domain& space,
      const void* initial_value, size_t alignment, bool fortran_order_dims,
      bool escaping)
    : field_size(fs), dims(d)
  //--------------------------------------------------------------------------
  {
    DeferredBufferRequest request(
        memkind, space, fs, alignment, fortran_order_dims, initial_value,
        escaping);
    *this = UntypedDeferredBuffer<T>::allocate_buffer(request);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  UntypedDeferredBuffer<T>::UntypedDeferredBuffer(
      size_t fs, int d, Memory memory, IndexSpace space,
      const void* initial_value, size_t alignment, bool fortran_order_dims,
      bool escaping)
    : field_size(fs), dims(d)
  //--------------------------------------------------------------------------
  {
    DeferredBufferRequest request(
        memory, space, fs, alignment, fortran_order_dims, initial_value,
        escaping);
    *this = UntypedDeferredBuffer<T>::allocate_buffer(request);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  UntypedDeferredBuffer<T>::UntypedDeferredBuffer(
      size_t fs, int d, Memory memory, const Domain& space,
      const void* initial_value, size_t alignment, bool fortran_order_dims,
      bool escaping)
    : field_size(fs), dims(d)
  //--------------------------------------------------------------------------
  {
    DeferredBufferRequest request(
        memory, space, fs, alignment, fortran_order_dims, initial_value,
        escaping);
    *this = UntypedDeferredBuffer<T>::allocate_buffer(request);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  template<typename FT, int DIM>
  UntypedDeferredBuffer<T>::UntypedDeferredBuffer(
      const DeferredBuffer<FT, DIM, T>& rhs)
    : instance(rhs.instance), field_size(sizeof(FT)), dims(DIM)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<typename T>
  template<typename FT, int DIM, bool BC>
  inline UntypedDeferredBuffer<T>::operator DeferredBuffer<FT, DIM, T, BC>(
      void) const
  //--------------------------------------------------------------------------
  {
    static_assert(0 < DIM, "Only positive dimensions allowed");
    static_assert(DIM <= LEGION_MAX_DIM, "Exceeded LEGION_MAX_DIM");
    assert(field_size == sizeof(FT));
    assert(dims == DIM);
    DeferredBuffer<FT, DIM, T> result;
    result.instance = instance;
    result.bounds = instance.template get_indexspace<DIM, T>().bounds;
    legion_assert((Realm::AffineAccessor<FT, DIM, T>::is_compatible(
        instance, 0 /*field id*/, result.bounds)));
    // We can make the accessor
    result.accessor = Realm::AffineAccessor<FT, DIM, T>(
        instance, 0 /*field id*/, result.bounds);
    // Can't just use accessor.base because it might not point to the low point
    if (!result.bounds.empty())
    {
      uintptr_t ptr =
          reinterpret_cast<uintptr_t>(result.accessor.ptr(result.bounds.lo));
      result.alignment = ptr & -ptr;
      // Figure out the order of the dimensions based on strides
      bool used[DIM];
      for (int d = 0; d < DIM; d++) used[d] = false;
      for (int d1 = 0; d1 < DIM; d1++)
      {
        int smallest_dim = -1;
        size_t smallest_stride = std::numeric_limits<size_t>::max();
        for (int d2 = 0; d2 < DIM; d2++)
        {
          if (used[d2])
            continue;
          size_t stride = result.accessor.strides[d2];
          if (stride < smallest_stride)
          {
            smallest_dim = d2;
            smallest_stride = stride;
          }
        }
        legion_assert(smallest_dim >= 0);
        result.ordering[d1] = static_cast<DimensionKind>(
            static_cast<int>(LEGION_DIM_X) + smallest_dim);
        used[smallest_dim] = true;
      }
    }
    else
    {
      // Can have maximum alignment since it is empty
      result.alignment = std::numeric_limits<size_t>::max();
      // Dimension ordering is undefined for empty spaces
      // so just pick an arbitrary one
      for (int d = 0; d < DIM; d++)
        result.ordering[d] =
            static_cast<DimensionKind>(static_cast<int>(LEGION_DIM_X) + d);
    }
    return result;
  }

}  // namespace Legion
