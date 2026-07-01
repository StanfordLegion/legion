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

#ifndef __LEGION_ACCESSORS_H__
#define __LEGION_ACCESSORS_H__

#include "legion/api/buffers.h"
#include "legion/api/future.h"
#include "legion/api/physical_region.h"
#include "legion/api/transforms.h"

namespace Legion {

  /**
   * \class FieldAccessor
   * A field accessor is a class used to get access to the data
   * inside of a PhysicalRegion object for a specific field. The
   * default version of this class is empty, but the following
   * specializations of this class with different privilege modes
   * will provide different methods specific to that privilege type
   * The ReduceAccessor class should be used for explicit reductions
   *
   * READ_ONLY
   *  - FT read(const Point<N,T>&) const
   *  ------ Methods below here for [Multi-]Affine Accessors only ------
   *  - const FT* ptr(const Point<N,T>&) const
   *  - const FT* ptr(const Rect<N,T>&, size_t = sizeof(FT)) const (dense)
   *  - const FT* ptr(const Rect<N,T>&, size_t strides[N],
   *                  size_t=sizeof(FT)) const
   *  - const FT& operator[](const Point<N,T>&) const
   *
   * READ_WRITE
   *  - FT read(const Point<N,T>&) const
   *  - void write(const Point<N,T>&, FT val) const
   *  ------ Methods below here for [Multi-]Affine Accessors only ------
   *  - FT* ptr(const Point<N,T>&) const
   *  - FT* ptr(const Rect<N,T>&, size_t = sizeof(FT)) const (must be dense)
   *  - FT* ptr(const Rect<N,T>&, size_t strides[N], size_t=sizeof(FT)) const
   *  - FT& operator[](const Point<N,T>&) const
   *  - template<typename REDOP, bool EXCLUSIVE>
   *      void reduce(const Point<N,T>&, REDOP::RHS) const
   *
   *  WRITE_DISCARD
   *  - void write(const Point<N,T>&, FT val) const
   *  ------ Methods below here for [Multi-]Affine Accessors only ------
   *  - FT* ptr(const Point<N,T>&) const
   *  - FT* ptr(const Rect<N,T>&, size_t = sizeof(FT)) const (must be dense)
   *  - FT* ptr(const Rect<N,T>&, size_t strides[N], size_t=sizeof(FT)) const
   *  - FT& operator[](const Point<N,T>&) const
   */
  template<
      PrivilegeMode MODE, typename FT, int N, typename COORD_T = coord_t,
      typename A = Realm::GenericAccessor<FT, N, COORD_T>,
#ifdef LEGION_BOUNDS_CHECKS
      bool CHECK_BOUNDS = true>
#else
      bool CHECK_BOUNDS = false>
#endif
  class FieldAccessor {
  private:
    static_assert(N > 0, "N must be positive");
    static_assert(N <= LEGION_MAX_DIM, "N must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
    FieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // For Realm::AffineAccessor specializations there are additional
    // methods for creating accessors with limited bounding boxes and
    // affine transformations for using alternative coordinates spaces
    // Specify a specific bounds rectangle to use for the accessor
    FieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        const Rect<N, COORD_T> bounds,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Specify a specific Affine transform to use for interpreting points
    // Not avalable for Realm::MultiAffineAccessor specializations
    template<int M>
    FieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        const AffineTransform<M, N, COORD_T> transform,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Specify both a transform and a bounds to use
    // Not avalable for Realm::MultiAffineAccessor specializations
    template<int M>
    FieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        const AffineTransform<M, N, COORD_T> transform,
        const Rect<N, COORD_T> bounds,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Create a field accessor for a Future
    // (only with READ-ONLY privileges and AffineAccessors)
    FieldAccessor(
        const Future& future, Memory::Kind kind = Memory::NO_MEMKIND,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Create a field accessor for a Future
    // (only with READ-ONLY privileges and AffineAccessors)
    FieldAccessor(
        const Future& future, const Rect<N, COORD_T> bounds,
        Memory::Kind kind = Memory::NO_MEMKIND,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
  public:
    // Variations of the above four methods but with multiple physical
    // regions specified using input iterators for colocation regions
    // Colocation regions from [start, stop)
    template<typename InputIterator>
    FieldAccessor(
        InputIterator start_region, InputIterator stop_region, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // For Realm::AffineAccessor specializations there are additional
    // methods for creating accessors with limited bounding boxes and
    // affine transformations for using alternative coordinates spaces
    // Specify a specific bounds rectangle to use for the accessor
    // Colocation regions from [start, stop)
    template<typename InputIterator>
    FieldAccessor(
        InputIterator start_region, InputIterator stop_region, FieldID fid,
        const Rect<N, COORD_T> bounds,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Specify a specific Affine transform to use for interpreting points
    // Not avalable for Realm::MultiAffineAccessor specializations
    // Colocation regions from [start, stop)
    template<typename InputIterator, int M>
    FieldAccessor(
        InputIterator start_region, InputIterator stop_region, FieldID fid,
        const AffineTransform<M, N, COORD_T> transform,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Specify both a transform and a bounds to use
    // Not avalable for Realm::MultiAffineAccessor specializations
    // Colocation regions from [start, stop)
    template<typename InputIterator, int M>
    FieldAccessor(
        InputIterator start_region, InputIterator stop_region, FieldID fid,
        const AffineTransform<M, N, COORD_T> transform,
        const Rect<N, COORD_T> bounds,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
  public:
    // Create a FieldAccessor for an UntypedDeferredValue
    // (only with AffineAccessors)
    FieldAccessor(
        const UntypedDeferredValue& value,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Create a FieldAccessor for an UntypedDeferredValue
    // Specify a specific bounds rectangle to use for the accessor
    // (only with AffineAccessors)
    FieldAccessor(
        const UntypedDeferredValue& value, const Rect<N, COORD_T>& bounds,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
  public:
    // Create a FieldAccessor for UntypedDeferredBuffer
    // (only with AffineAccessors)
    FieldAccessor(
        const UntypedDeferredBuffer<COORD_T>& buffer,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Create a FieldAccessor for UntypedDeferredBuffer
    // Specify a specific bounds rectangle to use for the accessor
    // (only with AffineAccessors)
    FieldAccessor(
        const UntypedDeferredBuffer<COORD_T>& buffer,
        const Rect<N, COORD_T>& bounds,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Create a FieldAccessor for UntypedDeferredBuffer
    // Specify a specific Affine transform to use for interpreting points
    // (only with AffineAccessors)
    template<int M>
    FieldAccessor(
        const UntypedDeferredBuffer<COORD_T>& buffer,
        const AffineTransform<M, N, COORD_T>& transform,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Create a FieldAccessor for UntypedDeferredBuffer
    // Specify both a transform and a bounds to use
    // (only with AffineAccessors)
    template<int M>
    FieldAccessor(
        const UntypedDeferredBuffer<COORD_T>& buffer,
        const AffineTransform<M, N, COORD_T>& transform,
        const Rect<N, COORD_T>& bounds,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  /**
   * \class ReductionAccessor
   * A field accessor is a class used to perform reductions to a given
   * field inside a PhysicalRegion object for a specific field. Reductions
   * can be performed directly or array indexing can be used along with
   * the <<= operator to perform the reduction. We also provide the same
   * variants of the 'ptr' method as normal accessors to obtain a pointer
   * to the underlying allocation. This seems to be useful when we need
   * to do reductions directly to a buffer as is often necessary when
   * invoking external libraries like BLAS.
   * This method currently only works with the Realm::AffineAccessor layout
   */
  template<
      typename REDOP, bool EXCLUSIVE, int N, typename COORD_T = coord_t,
      typename A = Realm::GenericAccessor<typename REDOP::RHS, N, COORD_T>,
#ifdef LEGION_BOUNDS_CHECKS
      bool CHECK_BOUNDS = true>
#else
      bool CHECK_BOUNDS = false>
#endif
  class ReductionAccessor {
  private:
    static_assert(N > 0, "N must be positive");
    static_assert(N <= LEGION_MAX_DIM, "N must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  public:
    ReductionAccessor(void) { }
    ReductionAccessor(
        const PhysicalRegion& region, FieldID fid, ReductionOpID redop,
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
    // For Realm::AffineAccessor specializations there are additional
    // methods for creating accessors with limited bounding boxes and
    // affine transformations for using alternative coordinates spaces
    // Specify a specific bounds rectangle to use for the accessor
    ReductionAccessor(
        const PhysicalRegion& region, FieldID fid, ReductionOpID redop,
        const Rect<N, COORD_T> bounds, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
    // Specify a specific Affine transform to use for interpreting points
    // Not available for Realm::MultiAffineAccessor specializations
    template<int M>
    ReductionAccessor(
        const PhysicalRegion& region, FieldID fid, ReductionOpID redop,
        const AffineTransform<M, N, COORD_T> transform,
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
    // Specify both a transform and a bounds to use
    // Not available for Realm::MultiAffineAccessor specializations
    template<int M>
    ReductionAccessor(
        const PhysicalRegion& region, FieldID fid, ReductionOpID redop,
        const AffineTransform<M, N, COORD_T> transform,
        const Rect<N, COORD_T> bounds, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
  public:
    // Variations of the same four methods above but with multiple
    // physical regions specified using input iterators for colocation regions
    // Colocation regions from [start, stop)
    template<typename InputIterator>
    ReductionAccessor(
        InputIterator start_region, InputIterator stop_region, FieldID fid,
        ReductionOpID redop, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
    // For Realm::AffineAccessor specializations there are additional
    // methods for creating accessors with limited bounding boxes and
    // affine transformations for using alternative coordinates spaces
    // Specify a specific bounds rectangle to use for the accessor
    // Colocation regions from [start, stop)
    template<typename InputIterator>
    ReductionAccessor(
        InputIterator start_region, InputIterator stop_region, FieldID fid,
        ReductionOpID redop, const Rect<N, COORD_T> bounds,
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
    // Specify a specific Affine transform to use for interpreting points
    // Not available for Realm::MultiAffineAccessor specializations
    // Colocation regions from [start, stop)
    template<typename InputIterator, int M>
    ReductionAccessor(
        InputIterator start_region, InputIterator stop_region, FieldID fid,
        ReductionOpID redop, const AffineTransform<M, N, COORD_T> transform,
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
    // Specify both a transform and a bounds to use
    // Not available for Realm::MultiAffineAccessor specializations
    // Colocation regions from [start, stop)
    template<typename InputIterator, int M>
    ReductionAccessor(
        InputIterator start_region, InputIterator stop_region, FieldID fid,
        ReductionOpID redop, const AffineTransform<M, N, COORD_T> transform,
        const Rect<N, COORD_T> bounds, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
  public:
    // Create a ReductionAccessor for an UntypedDeferredValue
    // (only with AffineAccessors)
    ReductionAccessor(
        const UntypedDeferredValue& value, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
    // Create a ReductionAccessor for an UntypedDeferredValue
    // Specify a specific bounds rectangle to use for the accessor
    // (only with AffineAccessors)
    ReductionAccessor(
        const UntypedDeferredValue& value, const Rect<N, COORD_T>& bounds,
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
  public:
    // Create a ReductionAccessor for an UntypedDeferredBuffer
    // (only with AffineAccessors)
    ReductionAccessor(
        const UntypedDeferredBuffer<COORD_T>& buffer,
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
    // Create a ReductionAccessor for an UntypedDeferredBuffer
    // Specify a specific bounds rectangle to use for the accessor
    // (only with AffineAccessors)
    ReductionAccessor(
        const UntypedDeferredBuffer<COORD_T>& buffer,
        const Rect<N, COORD_T>& bounds, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
    // Create a ReductionAccessor for an UntypedDeferredBuffer
    // Specify a specific Affine transform to use for interpreting points
    // (only with AffineAccessors)
    template<int M>
    ReductionAccessor(
        const UntypedDeferredBuffer<COORD_T>& buffer,
        const AffineTransform<M, N, COORD_T>& transform,
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
    // Create a ReductionAccessor for an UntypedDeferredBuffer
    // Specify both a transform and a bounds to use
    // (only with AffineAccessors)
    template<int M>
    ReductionAccessor(
        const UntypedDeferredBuffer<COORD_T>& buffer,
        const AffineTransform<M, N, COORD_T>& transform,
        const Rect<N, COORD_T>& bounds, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t subfield_offset = 0,
        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef LEGION_DEBUG
        bool check_field_size = true
#else
        bool check_field_size = false
#endif
    )
    { }
  public:
    typedef typename REDOP::RHS value_type;
    typedef typename REDOP::RHS& reference;
    typedef const typename REDOP::RHS& const_reference;
    static const int dim = N;
  };

  /**
   * \class PaddingAccessor
   * A padding accessor is used to obtain access to the padding space
   * available on a PhysicalRegion object. Note that all padding access
   * is always read-write (even if the privileges on the physical region
   * or less than read-write), because tasks are always guaranteed to not
   * interfere with other tasks using the padding area. Note that this
   * accessor only provides access to the padding space and you cannot
   * access other parts of the physical region (due to potential illegal
   * privilege escalation). Use a normal field accessor if you want access
   * to both the scratch space and the logical region part of the physical
   * region from the same accessor.
   *  - FT read(const Point<N,T>&) const
   *  - void write(const Point<N,T>&, FT val) const
   *  ------ Methods below here for Affine Accessors only ------
   *  - FT* ptr(const Point<N,T>&) const
   *  - FT* ptr(const Rect<N,T>&, size_t = sizeof(FT)) const (must be dense)
   *  - FT* ptr(const Rect<N,T>&, size_t strides[N], size_t=sizeof(FT)) const
   *  - FT& operator[](const Point<N,T>&) const
   *  - template<typename REDOP, bool EXCLUSIVE>
   *      void reduce(const Point<N,T>&, REDOP::RHS) const
   */
  template<
      typename FT, int N, typename COORD_T = coord_t,
      typename A = Realm::GenericAccessor<FT, N, COORD_T>,
#ifdef LEGION_BOUNDS_CHECKS
      bool CHECK_BOUNDS = true>
#else
      bool CHECK_BOUNDS = false>
#endif
  class PaddingAccessor {
  public:
    PaddingAccessor(void) { }
    PaddingAccessor(
        const PhysicalRegion& region, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
  };

#ifdef LEGION_MULTI_REGION_ACCESSOR
  // Multi-Region Accessors are a provisional feature now and are likely
  // to be deprecated and removed in the near future. Instead of multi-region
  // accessors you should be able to use the new colocation constructors
  // on the traditional Field Accessors.
  /**
   * \class MultiRegionAccessor
   * A multi-region accessor is a generalization of the field accessor class
   * to allow programs to access the same field for a common instance of
   * multiple logical region requirements. This is useful for performance
   * in cases where a task variant uses co-location constraints to
   * guarantee that data for certains region requirements are in the
   * same physical instance. This can avoid branching in application
   * code which rely on dynamic indexing into multiple logical regions.
   * There can be a productivity cost to using this accessor: it does
   * not capture privileges as part of its template parameters, so it
   * is easier to violate privileges without the C++ type checker
   * noticing. Users can enable privilege checks on an accessor by
   * setting the CHECK_PRIVILEGES template parameter to true or by
   * enabling PRIVILEGE_CHECKS throughout the entire application build.
   * Note that even in the affine case, in general you cannot directly
   * get a reference or a pointer to any elements so that we can enable
   * dynamic privilege checks to work. If you want to get a pointer or a
   * direct reference we encourage you to use the FieldAccessor for each
   * individual region at which point you can do any unsafe things you want.
   * If you do not use privilege checks then you can assume that auto == FT&
   * for operator[] methods. The following methods are supported:
   *
   *  - FT read(const Point<N,T>&) const
   *  - void write(const Point<N,T>&, FT val) const
   *  - auto operator[](const Point<N,T>&) const
   *  - template<typename REDOP, bool EXCLUSIVE>
   *      void reduce(const Point<N,T>&, REDOP::RHS) (Affine Accessor only)
   */
  template<
      typename FT, int N, typename COORD_T = coord_t,
      typename A = Realm::GenericAccessor<FT, N, COORD_T>,
#ifdef LEGION_BOUNDS_CHECKS
      bool CHECK_BOUNDS = true,
#else
      bool CHECK_BOUNDS = false,
#endif
#ifdef LEGION_PRIVILEGE_CHECKS
      bool CHECK_PRIVILEGES = true,
#else
      bool CHECK_PRIVILEGES = false,
#endif
      // Only used if bounds/privilege checks enabled
      // Can safely over-approximate, but may cost space
      // Especially GPU parameter space
      int MAX_REGIONS = 4>
  class MultiRegionAccessor {
  private:
    static_assert(N > 0, "N must be positive");
    static_assert(N <= LEGION_MAX_DIM, "N must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:  // iterator based construction of the multi-region accessors
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        // The actual field size in case it is different from
        // the one being used in FT and we still want to check
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Specify a specific bounds rectangle to use for the accessor
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<N, COORD_T> bounds,
        FieldID fid,
        // The actual field size in case it is different from
        // the one being used in FT and we still want to check
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Specify a specific Affine transform to use for interpreting points
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, N, COORD_T> transform,
        // The actual field size in case it is different from
        // the one being used in FT and we still want to check
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Specify both a transform and a bounds to use
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, N, COORD_T> transform,
        const Rect<N, COORD_T> bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
  public:  // explicit data structure versions of the implicit iterators above
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        // The actual field size in case it is different from
        // the one being used in FT and we still want to check
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Specify a specific bounds rectangle to use for the accessor
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<N, COORD_T> bounds, FieldID fid,
        // The actual field size in case it is different from
        // the one being used in FT and we still want to check
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Specify a specific Affine transform to use for interpreting points
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, N, COORD_T> transform,
        // The actual field size in case it is different from
        // the one being used in FT and we still want to check
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
    // Specify both a transform and a bounds to use
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, N, COORD_T> transform,
        const Rect<N, COORD_T> bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t subfield_offset = 0)
    { }
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };
#endif  // LEGION_MULTI_REGION_ACCESSOR

}  // namespace Legion

#include "accessors.inl"

#endif  // __LEGION_ACCESSORS_H__
