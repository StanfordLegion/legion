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

// Included from physical_region.h - do not include this directly

// Useful for IDEs
#include "legion/api/physical_region.h"

namespace Legion {

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  DomainT<DIM, T> PhysicalRegion::get_bounds(void) const
  //--------------------------------------------------------------------------
  {
    DomainT<DIM, T> result;
    get_bounds(&result, Internal::NT_TemplateHelper::encode_tag<DIM, T>());
    return result;
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  PhysicalRegion::operator DomainT<DIM, T>(void) const
  //--------------------------------------------------------------------------
  {
    DomainT<DIM, T> result;
    get_bounds(&result, Internal::NT_TemplateHelper::encode_tag<DIM, T>());
    return result;
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  PhysicalRegion::operator Rect<DIM, T>(void) const
  //--------------------------------------------------------------------------
  {
    DomainT<DIM, T> result;
    get_bounds(&result, Internal::NT_TemplateHelper::encode_tag<DIM, T>());
    legion_assert(result.dense());
    return result.bounds;
  }

  // Specialization for Span with READ_ONLY privileges
  template<typename FT>
  class Span<FT, LEGION_READ_ONLY> {
  public:
    class iterator {
    public:
      // explicitly set iterator traits
      typedef std::random_access_iterator_tag iterator_category;
      typedef FT value_type;
      typedef std::ptrdiff_t difference_type;
      typedef FT* pointer;
      typedef FT& reference;

      iterator(void) : ptr(nullptr), stride(0) { }
    private:
      iterator(const uint8_t* p, size_t s) : ptr(p), stride(s) { }
    public:
      inline iterator& operator=(const iterator& rhs)
      {
        ptr = rhs.ptr;
        stride = rhs.stride;
        return *this;
      }
      inline iterator& operator+=(int rhs)
      {
        ptr += stride;
        return *this;
      }
      inline iterator& operator-=(int rhs)
      {
        ptr -= stride;
        return *this;
      }
      inline const FT& operator*(void) const
      {
        FT* result = nullptr;
        static_assert(sizeof(result) == sizeof(ptr));
        memcpy(&result, &ptr, sizeof(result));
        return *result;
      }
      inline const FT* operator->(void) const
      {
        FT* result = nullptr;
        static_assert(sizeof(result) == sizeof(ptr));
        memcpy(&result, &ptr, sizeof(result));
        return result;
      }
      inline const FT& operator[](int rhs) const
      {
        FT* result = nullptr;
        const uint8_t* ptr2 = ptr + rhs * stride;
        static_assert(sizeof(result) == sizeof(ptr2));
        memcpy(&result, &ptr2, sizeof(result));
        return *result;
      }
    public:
      inline iterator& operator++(void)
      {
        ptr += stride;
        return *this;
      }
      inline iterator& operator--(void)
      {
        ptr -= stride;
        return *this;
      }
      inline iterator operator++(int)
      {
        iterator it(ptr, stride);
        ptr += stride;
        return it;
      }
      inline iterator operator--(int)
      {
        iterator it(ptr, stride);
        ptr -= stride;
        return it;
      }
      inline iterator operator+(int rhs) const
      {
        return iterator(ptr + stride * rhs, stride);
      }
      inline iterator operator-(int rhs) const
      {
        return iterator(ptr - stride * rhs, stride);
      }
    public:
      inline bool operator==(const iterator& rhs) const
      {
        return (ptr == rhs.ptr);
      }
      inline bool operator!=(const iterator& rhs) const
      {
        return (ptr != rhs.ptr);
      }
      inline bool operator<(const iterator& rhs) const
      {
        return (ptr < rhs.ptr);
      }
      inline bool operator>(const iterator& rhs) const
      {
        return (ptr > rhs.ptr);
      }
      inline bool operator<=(const iterator& rhs) const
      {
        return (ptr <= rhs.ptr);
      }
      inline bool operator>=(const iterator& rhs) const
      {
        return (ptr >= rhs.ptr);
      }
    private:
      const uint8_t* ptr;
      size_t stride;
    };
    class reverse_iterator {
    public:
      // explicitly set iterator traits
      typedef std::random_access_iterator_tag iterator_category;
      typedef FT value_type;
      typedef std::ptrdiff_t difference_type;
      typedef FT* pointer;
      typedef FT& reference;

      reverse_iterator(void) : ptr(nullptr), stride(0) { }
    private:
      reverse_iterator(const uint8_t* p, size_t s) : ptr(p), stride(s) { }
    public:
      inline reverse_iterator& operator=(const reverse_iterator& rhs)
      {
        ptr = rhs.ptr;
        stride = rhs.stride;
        return *this;
      }
      inline reverse_iterator& operator+=(int rhs)
      {
        ptr -= stride;
        return *this;
      }
      inline reverse_iterator& operator-=(int rhs)
      {
        ptr += stride;
        return *this;
      }
      inline const FT& operator*(void) const
      {
        FT* result = nullptr;
        static_assert(sizeof(result) == sizeof(ptr));
        memcpy(&result, &ptr, sizeof(result));
        return *result;
      }
      inline const FT* operator->(void) const
      {
        FT* result = nullptr;
        static_assert(sizeof(result) == sizeof(ptr));
        memcpy(&result, &ptr, sizeof(result));
        return result;
      }
      inline const FT& operator[](int rhs) const
      {
        FT* result = nullptr;
        const uint8_t* ptr2 = ptr - rhs * stride;
        static_assert(sizeof(result) == sizeof(ptr2));
        memcpy(&result, &ptr2, sizeof(result));
        return *result;
      }
    public:
      inline reverse_iterator& operator++(void)
      {
        ptr -= stride;
        return *this;
      }
      inline reverse_iterator& operator--(void)
      {
        ptr += stride;
        return *this;
      }
      inline reverse_iterator operator++(int)
      {
        reverse_iterator it(ptr, stride);
        ptr -= stride;
        return it;
      }
      inline reverse_iterator operator--(int)
      {
        reverse_iterator it(ptr, stride);
        ptr += stride;
        return it;
      }
      inline reverse_iterator operator+(int rhs) const
      {
        return reverse_iterator(ptr - stride * rhs, stride);
      }
      inline reverse_iterator operator-(int rhs) const
      {
        return reverse_iterator(ptr + stride * rhs, stride);
      }
    public:
      inline bool operator==(const reverse_iterator& rhs) const
      {
        return (ptr == rhs.ptr);
      }
      inline bool operator!=(const reverse_iterator& rhs) const
      {
        return (ptr != rhs.ptr);
      }
      inline bool operator<(const reverse_iterator& rhs) const
      {
        return (ptr > rhs.ptr);
      }
      inline bool operator>(const reverse_iterator& rhs) const
      {
        return (ptr < rhs.ptr);
      }
      inline bool operator<=(const reverse_iterator& rhs) const
      {
        return (ptr >= rhs.ptr);
      }
      inline bool operator>=(const reverse_iterator& rhs) const
      {
        return (ptr <= rhs.ptr);
      }
    private:
      const uint8_t* ptr;
      size_t stride;
    };
  public:
    Span(void) : base(nullptr), extent(0), stride(0) { }
    Span(const FT* b, size_t e, size_t s = sizeof(FT))
      : base(nullptr), extent(e), stride(s)
    {
      static_assert(sizeof(base) == sizeof(b));
      memcpy(&base, &b, sizeof(base));
    }
  public:
    inline iterator begin(void) const { return iterator(base, stride); }
    inline iterator end(void) const
    {
      return iterator(base + extent * stride, stride);
    }
    inline reverse_iterator rbegin(void) const
    {
      return reverse_iterator(base + (extent - 1) * stride, stride);
    }
    inline reverse_iterator rend(void) const
    {
      return reverse_iterator(base - stride, stride);
    }
  public:
    inline const FT& front(void) const
    {
      FT* result = nullptr;
      static_assert(sizeof(result) == sizeof(base));
      memcpy(&result, &base, sizeof(result));
      return *result;
    }
    inline const FT& back(void) const
    {
      FT* result = nullptr;
      const uint8_t* ptr = base + (extent - 1) * stride;
      static_assert(sizeof(result) == sizeof(ptr));
      memcpy(&result, &ptr, sizeof(result));
      return *result;
    }
    inline const FT& operator[](int index) const
    {
      FT* result = nullptr;
      const uint8_t* ptr = base + index * stride;
      static_assert(sizeof(result) == sizeof(ptr));
      memcpy(&result, &ptr, sizeof(result));
      return *result;
    }
    inline const FT* data(void) const
    {
      FT* result = nullptr;
      static_assert(sizeof(result) == sizeof(base));
      memcpy(&result, &base, sizeof(result));
      return result;
    }
    inline uintptr_t get_base(void) const { return uintptr_t(base); }
  public:
    inline size_t size(void) const { return extent; }
    inline size_t step(void) const { return stride; }
    inline bool empty(void) const { return (extent == 0); }
  private:
    const uint8_t* base;
    size_t extent;  // number of elements
    size_t stride;  // byte stride
  };

  /**
   * \class UnsafeSpanIterator
   * This is a hidden class analogous to the UnsafeFieldAccessor that
   * allows for traversals over spans of elements in compact instances
   */
  template<typename FT, int DIM, typename T = coord_t>
  class UnsafeSpanIterator {
  private:
    static_assert(DIM > 0, "DIM must be positive");
    static_assert(DIM <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    UnsafeSpanIterator(void) { }
    UnsafeSpanIterator(
        const PhysicalRegion& region, FieldID fid, bool privileges_only = true,
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : piece_iterator(PieceIteratorT<DIM, T>(region, fid, privileges_only)),
        partial_piece(false)
    {
      DomainT<DIM, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, false /*check size*/);
      if (!Realm::MultiAffineAccessor<FT, DIM, T>::is_compatible(
              instance, fid, is.bounds))
        region.report_incompatible_accessor(
            "UnsafeSpanIterator", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, DIM, T>(
          instance, fid, is.bounds, offset);
      // initialize the first span
      step();
    }
  public:
    inline bool valid(void) const { return !current.empty(); }
    inline bool step(void)
    {
      // Handle the remains of a partial piece if that is what we're doing
      if (partial_piece)
      {
        bool carry = false;
        for (int idx = 0; idx < DIM; idx++)
        {
          const int dim = dim_order[idx];
          if (carry || (dim == partial_step_dim))
          {
            if (partial_step_point[dim] < piece_iterator->hi[dim])
            {
              partial_step_point[dim] += 1;
              carry = false;
              break;
            }
            // carry case so reset and roll-over
            partial_step_point[dim] = piece_iterator->lo[dim];
            carry = true;
          }
          // Skip any dimensions before the partial step dim
        }
        // Make the next span
        current = Span<FT, LEGION_READ_WRITE>(
            accessor.ptr(partial_step_point), current.size(), current.step());
        // See if we are done with this partial piece
        if (carry)
          partial_piece = false;
        return true;
      }
      // clear this for the next iteration
      current = Span<FT, LEGION_READ_WRITE>();
      // Otherwise try to group as many rectangles together as we can
      while (piece_iterator.valid())
      {
        size_t strides[DIM];
        FT* ptr = accessor.ptr(*piece_iterator, strides);
        // If we ever hit this it is a runtime error because the
        // runtime should already be guaranteeing these rectangles
        // are inside of pieces for the instance
        legion_assert(ptr != nullptr);
        // Find the minimum stride and see if this piece is dense
        size_t min_stride = SIZE_MAX;
        for (int dim = 0; dim < DIM; dim++)
          if (strides[dim] < min_stride)
            min_stride = strides[dim];
        if (Internal::is_dense_layout(*piece_iterator, strides, min_stride))
        {
          const size_t volume = piece_iterator->volume();
          if (!current.empty())
          {
            uintptr_t base = current.get_base();
            // See if we can append to the current span
            if ((current.step() == min_stride) &&
                ((base + (current.size() * min_stride)) == uintptr_t(ptr)))
              current = Span<FT, LEGION_READ_WRITE>(
                  current.data(), current.size() + volume, min_stride);
            else  // Save this rectangle for the next iteration
              break;
          }
          else  // Start a new span
            current = Span<FT, LEGION_READ_WRITE>(ptr, volume, min_stride);
        }
        else
        {
          // Not a uniform stride, so go to the partial piece case
          if (current.empty())
          {
            partial_piece = true;
            // Compute the dimension order from smallest to largest
            size_t stride_floor = 0;
            for (int idx = 0; idx < DIM; idx++)
            {
              int index = -1;
              size_t local_min = SIZE_MAX;
              for (int dim = 0; dim < DIM; dim++)
              {
                if (strides[dim] <= stride_floor)
                  continue;
                if (strides[dim] < local_min)
                {
                  local_min = strides[dim];
                  index = dim;
                }
              }
              legion_assert(index >= 0);
              dim_order[idx] = index;
              stride_floor = local_min;
            }
            // See which dimensions we can handle at once and which ones
            // we are going to need to walk over
            size_t extent = 1;
            size_t exp_offset = min_stride;
            partial_step_dim = -1;
            for (int idx = 0; idx < DIM; idx++)
            {
              const int dim = dim_order[idx];
              if (strides[dim] == exp_offset)
              {
                size_t pitch =
                    ((piece_iterator->hi[dim] - piece_iterator->lo[dim]) + 1);
                exp_offset *= pitch;
                extent *= pitch;
              }
              // First dimension that is not contiguous
              partial_step_dim = dim;
              break;
            }
            legion_assert(partial_step_dim >= 0);
            partial_step_point = piece_iterator->lo;
            current = Span<FT, LEGION_READ_WRITE>(
                accessor.ptr(partial_step_point), extent, min_stride);
          }
          // No matter what we are breaking out here
          break;
        }
        // Step the piece iterator for the next iteration
        piece_iterator.step();
      }
      return valid();
    }
  public:
    inline operator bool(void) const { return valid(); }
    inline bool operator()(void) const { return valid(); }
    inline Span<FT, LEGION_READ_WRITE> operator*(void) const { return current; }
    inline const Span<FT, LEGION_READ_WRITE>* operator->(void) const
    {
      return &current;
    }
    inline UnsafeSpanIterator<FT, DIM, T>& operator++(void)
    {
      step();
      return *this;
    }
    inline UnsafeSpanIterator<FT, DIM, T> operator++(int)
    {
      UnsafeSpanIterator<FT, DIM, T> result = *this;
      step();
      return result;
    }
  private:
    PieceIteratorT<DIM, T> piece_iterator;
    mutable Realm::MultiAffineAccessor<FT, DIM, T> accessor;
    Span<FT, LEGION_READ_WRITE> current;
    Point<DIM, T> partial_step_point;
    int dim_order[DIM];
    int partial_step_dim;
    bool partial_piece;
  };

  //--------------------------------------------------------------------------
  inline bool PieceIterator::valid(void) const
  //--------------------------------------------------------------------------
  {
    return (impl != nullptr) && (index >= 0);
  }

  //--------------------------------------------------------------------------
  inline PieceIterator::operator bool(void) const
  //--------------------------------------------------------------------------
  {
    return valid();
  }

  //--------------------------------------------------------------------------
  inline bool PieceIterator::operator()(void) const
  //--------------------------------------------------------------------------
  {
    return valid();
  }

  //--------------------------------------------------------------------------
  inline const Domain& PieceIterator::operator*(void) const
  //--------------------------------------------------------------------------
  {
    return current_piece;
  }

  //--------------------------------------------------------------------------
  inline const Domain* PieceIterator::operator->(void) const
  //--------------------------------------------------------------------------
  {
    return &current_piece;
  }

  //--------------------------------------------------------------------------
  inline PieceIterator& PieceIterator::operator++(void)
  //--------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //--------------------------------------------------------------------------
  inline PieceIterator PieceIterator::operator++(int)
  //--------------------------------------------------------------------------
  {
    PieceIterator result = *this;
    step();
    return result;
  }

  //--------------------------------------------------------------------------
  inline bool PieceIterator::operator<(const PieceIterator& rhs) const
  //--------------------------------------------------------------------------
  {
    if (impl < rhs.impl)
      return true;
    if (impl > rhs.impl)
      return false;
    if (index < rhs.index)
      return true;
    return false;
  }

  //--------------------------------------------------------------------------
  inline bool PieceIterator::operator==(const PieceIterator& rhs) const
  //--------------------------------------------------------------------------
  {
    if (impl != rhs.impl)
      return false;
    return index == rhs.index;
  }

  //--------------------------------------------------------------------------
  inline bool PieceIterator::operator!=(const PieceIterator& rhs) const
  //--------------------------------------------------------------------------
  {
    if (impl != rhs.impl)
      return true;
    return index != rhs.index;
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  inline PieceIteratorT<DIM, T>::PieceIteratorT(void) : PieceIterator()
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  inline PieceIteratorT<DIM, T>::PieceIteratorT(const PieceIteratorT& rhs)
    : PieceIterator(rhs), current_rect(rhs.current_rect)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  inline PieceIteratorT<DIM, T>::PieceIteratorT(PieceIteratorT&& rhs) noexcept
    : PieceIterator(rhs), current_rect(rhs.current_rect)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  inline PieceIteratorT<DIM, T>::PieceIteratorT(
      const PhysicalRegion& region, FieldID fid, bool privilege_only,
      bool silence_warn, const char* warn)
    : PieceIterator(region, fid, privilege_only, silence_warn, warn)
  //--------------------------------------------------------------------------
  {
    if (valid())
      current_rect = current_piece;
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  inline PieceIteratorT<DIM, T>& PieceIteratorT<DIM, T>::operator=(
      const PieceIteratorT& rhs)
  //--------------------------------------------------------------------------
  {
    PieceIterator::operator=(rhs);
    current_rect = rhs.current_rect;
    return *this;
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  inline PieceIteratorT<DIM, T>& PieceIteratorT<DIM, T>::operator=(
      PieceIteratorT&& rhs) noexcept
  //--------------------------------------------------------------------------
  {
    PieceIterator::operator=(rhs);
    current_rect = rhs.current_rect;
    return *this;
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  inline bool PieceIteratorT<DIM, T>::step(void)
  //--------------------------------------------------------------------------
  {
    const bool result = PieceIterator::step();
    current_rect = current_piece;
    return result;
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  inline const Rect<DIM, T>& PieceIteratorT<DIM, T>::operator*(void) const
  //--------------------------------------------------------------------------
  {
    return current_rect;
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  inline const Rect<DIM, T>* PieceIteratorT<DIM, T>::operator->(void) const
  //--------------------------------------------------------------------------
  {
    return &current_rect;
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  inline PieceIteratorT<DIM, T>& PieceIteratorT<DIM, T>::operator++(void)
  //--------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  inline PieceIteratorT<DIM, T> PieceIteratorT<DIM, T>::operator++(int)
  //--------------------------------------------------------------------------
  {
    PieceIteratorT<DIM, T> result = *this;
    step();
    return result;
  }

  //--------------------------------------------------------------------------
  template<PrivilegeMode PM, typename FT, int DIM, typename T>
  inline SpanIterator<PM, FT, DIM, T>::SpanIterator(
      const PhysicalRegion& region, FieldID fid, size_t actual_field_size,
      bool check_field_size, bool priv, bool silence_warnings,
      const char* warning_string)
    : piece_iterator(PieceIteratorT<DIM, T>(region, fid, priv)),
      partial_piece(false)
  //--------------------------------------------------------------------------
  {
    DomainT<DIM, T> is;
    const Realm::RegionInstance instance = region.get_instance_info(
        PM, fid, actual_field_size, &is,
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,
        silence_warnings, false /*generic accessor*/, check_field_size);
    if (!Realm::MultiAffineAccessor<FT, DIM, T>::is_compatible(
            instance, fid, is.bounds))
      region.report_incompatible_accessor("SpanIterator", instance, fid);
    accessor = Realm::MultiAffineAccessor<FT, DIM, T>(instance, fid, is.bounds);
    // initialize the first span
    step();
  }

  //--------------------------------------------------------------------------
  template<PrivilegeMode PM, typename FT, int DIM, typename T>
  inline bool SpanIterator<PM, FT, DIM, T>::valid(void) const
  //--------------------------------------------------------------------------
  {
    return !current.empty();
  }

  //--------------------------------------------------------------------------
  template<PrivilegeMode PM, typename FT, int DIM, typename T>
  inline bool SpanIterator<PM, FT, DIM, T>::step(void)
  //--------------------------------------------------------------------------
  {
    // Handle the remains of a partial piece if that is what we're doing
    if (partial_piece)
    {
      bool carry = false;
      for (int idx = 0; idx < DIM; idx++)
      {
        const int dim = dim_order[idx];
        if (carry || (dim == partial_step_dim))
        {
          if (partial_step_point[dim] < piece_iterator->hi[dim])
          {
            partial_step_point[dim] += 1;
            carry = false;
            break;
          }
          // carry case so reset and roll-over
          partial_step_point[dim] = piece_iterator->lo[dim];
          carry = true;
        }
        // Skip any dimensions before the partial step dim
      }
      // Make the next span
      current = Span<FT, PM>(
          accessor.ptr(partial_step_point), current.size(), current.step());
      // See if we are done with this partial piece
      if (carry)
        partial_piece = false;
      return true;
    }
    current = Span<FT, PM>();  // clear this for the next iteration
    // Otherwise try to group as many rectangles together as we can
    while (piece_iterator.valid())
    {
      size_t strides[DIM];
      FT* ptr = accessor.ptr(*piece_iterator, strides);
      // If we ever hit this it is a runtime error because the
      // runtime should already be guaranteeing these rectangles
      // are inside of pieces for the instance
      legion_assert(ptr != nullptr);
      // Find the minimum stride and see if this piece is dense
      size_t min_stride = SIZE_MAX;
      for (int dim = 0; dim < DIM; dim++)
        if (strides[dim] < min_stride)
          min_stride = strides[dim];
      if (Internal::is_dense_layout(*piece_iterator, strides, min_stride))
      {
        const size_t volume = piece_iterator->volume();
        if (!current.empty())
        {
          uintptr_t base = current.get_base();
          // See if we can append to the current span
          if ((current.step() == min_stride) &&
              ((base + (current.size() * min_stride)) == uintptr_t(ptr)))
            current = Span<FT, PM>(
                current.data(), current.size() + volume, min_stride);
          else  // Save this rectangle for the next iteration
            break;
        }
        else  // Start a new span
          current = Span<FT, PM>(ptr, volume, min_stride);
      }
      else
      {
        // Not a uniform stride, so go to the partial piece case
        if (current.empty())
        {
          partial_piece = true;
          // Compute the dimension order from smallest to largest
          size_t stride_floor = 0;
          for (int idx = 0; idx < DIM; idx++)
          {
            int index = -1;
            size_t local_min = SIZE_MAX;
            for (int dim = 0; dim < DIM; dim++)
            {
              if (strides[dim] <= stride_floor)
                continue;
              if (strides[dim] < local_min)
              {
                local_min = strides[dim];
                index = dim;
              }
            }
            legion_assert(index >= 0);
            dim_order[idx] = index;
            stride_floor = local_min;
          }
          // See which dimensions we can handle at once and which ones
          // we are going to need to walk over
          size_t extent = 1;
          size_t exp_offset = min_stride;
          partial_step_dim = -1;
          for (int idx = 0; idx < DIM; idx++)
          {
            const int dim = dim_order[idx];
            if (strides[dim] == exp_offset)
            {
              size_t pitch =
                  ((piece_iterator->hi[dim] - piece_iterator->lo[dim]) + 1);
              exp_offset *= pitch;
              extent *= pitch;
            }
            // First dimension that is not contiguous
            partial_step_dim = dim;
            break;
          }
          legion_assert(partial_step_dim >= 0);
          partial_step_point = piece_iterator->lo;
          current = Span<FT, PM>(
              accessor.ptr(partial_step_point), extent, min_stride);
        }
        // No matter what we are breaking out here
        break;
      }
      // Step the piece iterator for the next iteration
      piece_iterator.step();
    }
    return valid();
  }

  //--------------------------------------------------------------------------
  template<PrivilegeMode PM, typename FT, int DIM, typename T>
  inline SpanIterator<PM, FT, DIM, T>::operator bool(void) const
  //--------------------------------------------------------------------------
  {
    return valid();
  }

  //--------------------------------------------------------------------------
  template<PrivilegeMode PM, typename FT, int DIM, typename T>
  inline bool SpanIterator<PM, FT, DIM, T>::operator()(void) const
  //--------------------------------------------------------------------------
  {
    return valid();
  }

  //--------------------------------------------------------------------------
  template<PrivilegeMode PM, typename FT, int DIM, typename T>
  inline const Span<FT, PM>& SpanIterator<PM, FT, DIM, T>::operator*(void) const
  //--------------------------------------------------------------------------
  {
    return current;
  }

  //--------------------------------------------------------------------------
  template<PrivilegeMode PM, typename FT, int DIM, typename T>
  inline const Span<FT, PM>* SpanIterator<PM, FT, DIM, T>::operator->(
      void) const
  //--------------------------------------------------------------------------
  {
    return &current;
  }

  //--------------------------------------------------------------------------
  template<PrivilegeMode PM, typename FT, int DIM, typename T>
  inline SpanIterator<PM, FT, DIM, T>& SpanIterator<PM, FT, DIM, T>::operator++(
      void)
  //--------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //--------------------------------------------------------------------------
  template<PrivilegeMode PM, typename FT, int DIM, typename T>
  inline SpanIterator<PM, FT, DIM, T> SpanIterator<PM, FT, DIM, T>::operator++(
      int)
  //--------------------------------------------------------------------------
  {
    SpanIterator<PM, FT, DIM, T> result = *this;
    step();
    return result;
  }

}  // namespace Legion

namespace std {

#define LEGION_DEFINE_HASHABLE(__TYPE_NAME__)                     \
  template<>                                                      \
  struct hash<__TYPE_NAME__> {                                    \
    inline std::size_t operator()(const __TYPE_NAME__& obj) const \
    {                                                             \
      return obj.hash();                                          \
    }                                                             \
  };

  LEGION_DEFINE_HASHABLE(Legion::PhysicalRegion);

#undef LEGION_DEFINE_HASHABLE

}  // namespace std
