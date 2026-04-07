/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_PHYSICAL_REGION_H__
#define __LEGION_PHYSICAL_REGION_H__

#include "legion/api/types.h"
#include "legion/api/geometry.h"
#include "legion/api/buffers.h"

namespace Legion {

  /**
   * \class PhysicalRegion
   * Physical region objects are used to manage access to the
   * physical instances that hold data.  They are lightweight
   * handles that can be stored in data structures and passed
   * by value.  They should never escape the context in which
   * they are created.
   */
  class PhysicalRegion : public Unserializable {
  public:
    PhysicalRegion(void);
    PhysicalRegion(const PhysicalRegion& rhs);
    PhysicalRegion(PhysicalRegion&& rhs) noexcept;
    ~PhysicalRegion(void);
  private:
    Internal::PhysicalRegionImpl* impl;
  protected:
    FRIEND_ALL_RUNTIME_CLASSES
    explicit PhysicalRegion(Internal::PhysicalRegionImpl* impl);
  public:
    PhysicalRegion& operator=(const PhysicalRegion& rhs);
    PhysicalRegion& operator=(PhysicalRegion&& rhs) noexcept;
    inline bool exists(void) const { return (impl != nullptr); }
    inline bool operator==(const PhysicalRegion& reg) const
    {
      return (impl == reg.impl);
    }
    inline bool operator!=(const PhysicalRegion& reg) const
    {
      return (impl != reg.impl);
    }
    inline bool operator<(const PhysicalRegion& reg) const
    {
      return (impl < reg.impl);
    }
    std::size_t hash(void) const;
  public:
    /**
     * Check to see if this represents a mapped physical region.
     */
    bool is_mapped(void) const;
    /**
     * For physical regions returned as the result of an
     * inline mapping, this call will block until the physical
     * instance has a valid copy of the data. You can silence
     * warnings about this blocking call with the
     * 'silence_warnings' parameter.
     */
    void wait_until_valid(
        bool silence_warnings = false, const char* warning_string = nullptr);
    /**
     * For physical regions returned from inline mappings,
     * this call will query if the instance contains valid
     * data yet.
     * @return whether the region has valid data
     */
    bool is_valid(void) const;
    /**
     * @return the logical region for this physical region
     */
    LogicalRegion get_logical_region(void) const;
    /**
     * @return the privilege mode for this physical region
     */
    PrivilegeMode get_privilege(void) const;
    /**
     * Return the memories where the underlying physical instances locate.
     */
    void get_memories(
        std::set<Memory>& memories, bool silence_warnings = false,
        const char* warning_string = nullptr) const;
    /**
     * Return a list of fields that the physical region contains.
     */
    void get_fields(std::vector<FieldID>& fields) const;
  public:
    template<int DIM, typename COORD_T>
    DomainT<DIM, COORD_T> get_bounds(void) const;
    // We'll also allow this to implicitly cast to a realm index space
    // so that users can easily iterate over the points
    template<int DIM, typename COORD_T>
    operator DomainT<DIM, COORD_T>(void) const;
    // They can implicitly cast to a rectangle if there is no
    // sparsity map, runtime will check for this
    template<int DIM, typename COORD_T>
    operator Rect<DIM, COORD_T>(void) const;
  protected:
    // These methods can only be accessed by accessor classes
    template<PrivilegeMode, typename, int, typename, typename, bool>
    friend class FieldAccessor;
    template<typename, bool, int, typename, typename, bool>
    friend class ReductionAccessor;
    template<typename, int, typename, typename, bool, bool, int>
    friend class MultiRegionAccessor;
    template<typename, int, typename, typename, bool>
    friend class PaddingAccessor;
    template<typename, int, typename, typename>
    friend class UnsafeFieldAccessor;
    template<typename, PrivilegeMode>
    friend class ArraySyntax::AccessorRefHelper;
    template<typename>
    friend class ArraySyntax::AffineRefHelper;
    friend class PieceIterator;
    template<PrivilegeMode, typename, int, typename>
    friend class SpanIterator;
    template<typename, int, typename>
    friend class UnsafeSpanIterator;
    Realm::RegionInstance get_instance_info(
        PrivilegeMode mode, FieldID fid, size_t field_size, void* realm_is,
        TypeTag type_tag, const char* warning_string, bool silence_warnings,
        bool generic_accessor, bool check_field_size,
        ReductionOpID redop = 0) const;
    Realm::RegionInstance get_instance_info(
        PrivilegeMode mode, const std::vector<PhysicalRegion>& other_regions,
        FieldID fid, size_t field_size, void* realm_is, TypeTag type_tag,
        const char* warning_string, bool silence_warnings,
        bool generic_accessor, bool check_field_size, bool need_bounds,
        ReductionOpID redop = 0) const;
    Realm::RegionInstance get_padding_info(
        FieldID fid, size_t field_size, Domain* inner, Domain& outer,
        const char* warning_string, bool silence_warnings,
        bool generic_accessor, bool check_field_size) const;
    void report_incompatible_accessor(
        const char* accessor_kind, Realm::RegionInstance instance,
        FieldID fid) const;
    void report_incompatible_multi_accessor(
        unsigned index, FieldID fid, Realm::RegionInstance inst1,
        Realm::RegionInstance inst2) const;
    void report_colocation_violation(
        const char* accessor_kind, FieldID fid, Realm::RegionInstance inst1,
        Realm::RegionInstance inst2, const PhysicalRegion& other,
        bool reduction = false) const;
    static void empty_colocation_regions(
        const char* accessor_kind, FieldID fid, bool reduction = false);
    static void fail_bounds_check(
        DomainPoint p, FieldID fid, PrivilegeMode mode, bool multi = false);
    static void fail_bounds_check(
        Domain d, FieldID fid, PrivilegeMode mode, bool multi = false);
    static void fail_privilege_check(
        DomainPoint p, FieldID fid, PrivilegeMode mode);
    static void fail_privilege_check(Domain d, FieldID fid, PrivilegeMode mode);
    static void fail_padding_check(DomainPoint p, FieldID fid);
    static void fail_nondense_rect(void);
    static void fail_rect_piece(void);
  protected:
    void get_bounds(void* realm_is, TypeTag type_tag) const;
  };

  /**
   * \class ExternalResources
   * An external resources object stores a collection of physical
   * regions that were attached together using the same index space
   * attach operation. It acts as a vector-like container of the
   * physical regions and ensures that they are detached together.
   */
  class ExternalResources : public Unserializable {
  public:
    ExternalResources(void);
    ExternalResources(const ExternalResources& rhs);
    ExternalResources(ExternalResources&& rhs) noexcept;
    ~ExternalResources(void);
  private:
    Internal::ExternalResourcesImpl* impl;
  protected:
    FRIEND_ALL_RUNTIME_CLASSES
    explicit ExternalResources(Internal::ExternalResourcesImpl* impl);
  public:
    ExternalResources& operator=(const ExternalResources& rhs);
    ExternalResources& operator=(ExternalResources&& rhs) noexcept;
    inline bool exists(void) const { return (impl != nullptr); }
    inline bool operator==(const ExternalResources& reg) const
    {
      return (impl == reg.impl);
    }
    inline bool operator<(const ExternalResources& reg) const
    {
      return (impl < reg.impl);
    }
  public:
    size_t size(void) const;
    PhysicalRegion operator[](unsigned index) const;
  };

  /**
   * \class PieceIterator
   * When mappers create a physical instance of a logical region, they have
   * the option of choosing a layout that is affine or compact. Affine
   * layouts have space for the convex hull of a logical region and support
   * O(1) memory accesses. Compact layouts have affine "pieces" of memory
   * for subsets of the points in the logical region. A PieceIterator object
   * supports iteration over all such affine pieces in a compact instance so
   * that an accessor can be made for each one. Note that you can also make
   * a PieceIterator for a instance with an affine layout: it is just a
   * special case that contains a single piece. Note that the pieces are
   * rectangles which maybe different than the the rectangles in the original
   * index space for the logical region of this physical region. Furthermore,
   * the pieces are iterated in the order that they are laid out in memory
   * which is unrelated to the order rectangles are iterated for the index
   * space of the logical region for the physical region. The application can
   * control whether only rectangles with privileges are presented with the
   * privilege_only flag. If the privilege_only flag is set to true then each
   * rectangles will be for a dense set of points for which the task has
   * privileges. If it is set to false, the the iterator will just return
   * the rectangles for the pieces of the instance regardless of whether
   * the application has privileges on them or not.
   */
  class PieceIterator {
  public:
    PieceIterator(void);
    PieceIterator(const PieceIterator& rhs);
    PieceIterator(PieceIterator&& rhs) noexcept;
    PieceIterator(
        const PhysicalRegion& region, FieldID fid, bool privilege_only = true,
        bool silence_warnings = false, const char* warning_string = nullptr);
    ~PieceIterator(void);
  public:
    PieceIterator& operator=(const PieceIterator& rhs);
    PieceIterator& operator=(PieceIterator&& rhs) noexcept;
  public:
    inline bool valid(void) const;
    bool step(void);
  public:
    inline operator bool(void) const;
    inline bool operator()(void) const;
    inline const Domain& operator*(void) const;
    inline const Domain* operator->(void) const;
    inline PieceIterator& operator++(void);
    inline PieceIterator operator++(int /*postfix*/);
  public:
    bool operator<(const PieceIterator& rhs) const;
    bool operator==(const PieceIterator& rhs) const;
    bool operator!=(const PieceIterator& rhs) const;
  private:
    Internal::PieceIteratorImpl* impl;
    int index;
  protected:
    Domain current_piece;
  };

  /**
   * \class PieceIteratorT
   * This is the typed version of a PieceIterator for users that want
   * to get explicit rectangles instead of domains.
   */
  template<int DIM, typename COORD_T = coord_t>
  class PieceIteratorT : public PieceIterator {
  private:
    static_assert(DIM > 0, "DIM must be positive");
    static_assert(DIM <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  public:
    PieceIteratorT(void);
    PieceIteratorT(const PieceIteratorT& rhs);
    PieceIteratorT(PieceIteratorT&& rhs) noexcept;
    PieceIteratorT(
        const PhysicalRegion& region, FieldID fid, bool privilege_only,
        bool silence_warnings = false, const char* warning_string = nullptr);
  public:
    PieceIteratorT<DIM, COORD_T>& operator=(const PieceIteratorT& rhs);
    PieceIteratorT<DIM, COORD_T>& operator=(PieceIteratorT&& rhs) noexcept;
  public:
    inline bool step(void);
    inline const Rect<DIM, COORD_T>& operator*(void) const;
    inline const Rect<DIM, COORD_T>* operator->(void) const;
    inline PieceIteratorT<DIM, COORD_T>& operator++(void);
    inline PieceIteratorT<DIM, COORD_T> operator++(int /*postfix*/);
  protected:
    Rect<DIM, COORD_T> current_rect;
  };

  /**
   * \class Span
   * A span class is used for handing back allocations of elements with
   * a uniform stride that users can safely access simply by indexing
   * the pointer as an array of elements. Note that the Legion definition
   * of a span does not guarantee that elements are contiguous the same
   * as the c++20 definition of a span.
   */
  template<typename FT, PrivilegeMode PM = LEGION_READ_WRITE>
  class Span {
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
      iterator(uint8_t* p, size_t s) : ptr(p), stride(s) { }
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
      inline FT& operator*(void) const
      {
        FT* result = nullptr;
        static_assert(sizeof(result) == sizeof(ptr));
        memcpy(&result, &ptr, sizeof(result));
        return *result;
      }
      inline FT* operator->(void) const
      {
        FT* result = nullptr;
        static_assert(sizeof(result) == sizeof(ptr));
        memcpy(&result, &ptr, sizeof(result));
        return result;
      }
      inline FT& operator[](int rhs) const
      {
        FT* result = nullptr;
        uint8_t* ptr2 = ptr + rhs * stride;
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
      uint8_t* ptr;
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
      reverse_iterator(uint8_t* p, size_t s) : ptr(p), stride(s) { }
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
      inline FT& operator*(void) const
      {
        FT* result = nullptr;
        static_assert(sizeof(result) == sizeof(ptr));
        memcpy(&result, &ptr, sizeof(result));
        return *result;
      }
      inline FT* operator->(void) const
      {
        FT* result = nullptr;
        static_assert(sizeof(result) == sizeof(ptr));
        memcpy(&result, &ptr, sizeof(result));
        return result;
      }
      inline FT& operator[](int rhs) const
      {
        FT* result = nullptr;
        uint8_t* ptr2 = ptr - rhs * stride;
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
      uint8_t* ptr;
      size_t stride;
    };
  public:
    Span(void) : base(nullptr), extent(0), stride(0) { }
    Span(FT* b, size_t e, size_t s = sizeof(FT))
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
    inline FT& front(void) const
    {
      FT* result = nullptr;
      static_assert(sizeof(result) == sizeof(base));
      memcpy(&result, &base, sizeof(result));
      return *result;
    }
    inline FT& back(void) const
    {
      FT* result = nullptr;
      uint8_t* ptr = base + (extent - 1) * stride;
      static_assert(sizeof(result) == sizeof(ptr));
      memcpy(&result, &ptr, sizeof(result));
      return *result;
    }
    inline FT& operator[](int index) const
    {
      FT* result = nullptr;
      uint8_t* ptr = base + index * stride;
      static_assert(sizeof(result) == sizeof(ptr));
      memcpy(&result, &ptr, sizeof(result));
      return *result;
    }
    inline FT* data(void) const
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
    uint8_t* base;
    size_t extent;  // number of elements
    size_t stride;  // byte stride
  };

  /**
   * \class SpanIterator
   * While the common model for compact instances is to use a piece iterator
   * to walk over pieces and create a field accessor to index the elements in
   * each piece, some applications want to transpose these loops and walk
   * linearly over all spans of a field with a common stride without needing
   * to know which piece they belong to. The SpanIterator class allows this
   * piece-agnostic traversal of a field.
   */
  template<PrivilegeMode PM, typename FT, int DIM, typename COORD_T = coord_t>
  class SpanIterator {
  private:
    static_assert(DIM > 0, "DIM must be positive");
    static_assert(DIM <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  public:
    SpanIterator(void) { }
    SpanIterator(
        const PhysicalRegion& region, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        // Iterate only the spans that we have privileges on
        bool privileges_only = true, bool silence_warnings = false,
        const char* warning_string = nullptr);
  public:
    inline bool valid(void) const;
    inline bool step(void);
  public:
    inline operator bool(void) const;
    inline bool operator()(void) const;
    inline const Span<FT, PM>& operator*(void) const;
    inline const Span<FT, PM>* operator->(void) const;
    inline SpanIterator<PM, FT, DIM, COORD_T>& operator++(void);
    inline SpanIterator<PM, FT, DIM, COORD_T> operator++(int);
  private:
    PieceIteratorT<DIM, COORD_T> piece_iterator;
    Realm::MultiAffineAccessor<FT, DIM, COORD_T> accessor;
    Span<FT, PM> current;
    Point<DIM, COORD_T> partial_step_point;
    int dim_order[DIM];
    int partial_step_dim;
    bool partial_piece;
  };

}  // namespace Legion

#include "legion/api/physical_region.inl"

#endif  // __LEGION_PHYSICAL_REGION_H__
