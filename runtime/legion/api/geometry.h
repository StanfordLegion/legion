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

#ifndef __LEGION_GEOMETRY_H__
#define __LEGION_GEOMETRY_H__

#include <cstring>
#include "legion/api/types.h"

namespace Legion {

  /**
   * \class DomainPoint
   * This is a type erased point where the number of
   * dimensions is a runtime value
   */
  class DomainPoint {
  public:
    static constexpr int MAX_POINT_DIM = LEGION_MAX_DIM;

    __LEGION_CUDA_HD__
    DomainPoint(void);
    __LEGION_CUDA_HD__
    DomainPoint(coord_t index);
    __LEGION_CUDA_HD__
    DomainPoint(const DomainPoint& rhs);
    template<int DIM, typename T>
    __LEGION_CUDA_HD__ DomainPoint(const Point<DIM, T>& rhs);

    template<int DIM, typename T>
    __LEGION_CUDA_HD__ operator Point<DIM, T>(void) const;

    __LEGION_CUDA_HD__
    DomainPoint& operator=(const DomainPoint& rhs);
    template<int DIM, typename T>
    __LEGION_CUDA_HD__ DomainPoint& operator=(const Point<DIM, T>& rhs);
    __LEGION_CUDA_HD__
    bool operator==(const DomainPoint& rhs) const;
    __LEGION_CUDA_HD__
    bool operator!=(const DomainPoint& rhs) const;
    __LEGION_CUDA_HD__
    bool operator<(const DomainPoint& rhs) const;

    __LEGION_CUDA_HD__
    DomainPoint operator+(coord_t scalar) const;
    __LEGION_CUDA_HD__
    DomainPoint operator+(const DomainPoint& rhs) const;
    __LEGION_CUDA_HD__
    DomainPoint& operator+=(coord_t scalar);
    __LEGION_CUDA_HD__
    DomainPoint& operator+=(const DomainPoint& rhs);

    __LEGION_CUDA_HD__
    DomainPoint operator-(coord_t scalar) const;
    __LEGION_CUDA_HD__
    DomainPoint operator-(const DomainPoint& rhs) const;
    __LEGION_CUDA_HD__
    DomainPoint& operator-=(coord_t scalar);
    __LEGION_CUDA_HD__
    DomainPoint& operator-=(const DomainPoint& rhs);

    __LEGION_CUDA_HD__
    DomainPoint operator*(coord_t scalar) const;
    __LEGION_CUDA_HD__
    DomainPoint operator*(const DomainPoint& rhs) const;
    __LEGION_CUDA_HD__
    DomainPoint& operator*=(coord_t scalar);
    __LEGION_CUDA_HD__
    DomainPoint& operator*=(const DomainPoint& rhs);

    __LEGION_CUDA_HD__
    DomainPoint operator/(coord_t scalar) const;
    __LEGION_CUDA_HD__
    DomainPoint operator/(const DomainPoint& rhs) const;
    __LEGION_CUDA_HD__
    DomainPoint& operator/=(coord_t scalar);
    __LEGION_CUDA_HD__
    DomainPoint& operator/=(const DomainPoint& rhs);

    __LEGION_CUDA_HD__
    DomainPoint operator%(coord_t scalar) const;
    __LEGION_CUDA_HD__
    DomainPoint operator%(const DomainPoint& rhs) const;
    __LEGION_CUDA_HD__
    DomainPoint& operator%=(coord_t scalar);
    __LEGION_CUDA_HD__
    DomainPoint& operator%=(const DomainPoint& rhs);

    __LEGION_CUDA_HD__
    coord_t& operator[](unsigned index);
    __LEGION_CUDA_HD__
    const coord_t& operator[](unsigned index) const;

    struct STLComparator {
      __LEGION_CUDA_HD__
      bool operator()(const DomainPoint& a, const DomainPoint& b) const
      {
        if (a.dim < b.dim)
          return true;
        if (a.dim > b.dim)
          return false;
        for (int i = 0; (i == 0) || (i < a.dim); i++)
        {
          if (a.point_data[i] < b.point_data[i])
            return true;
          if (a.point_data[i] > b.point_data[i])
            return false;
        }
        return false;
      }
    };

    __LEGION_CUDA_HD__
    Color get_color(void) const;
    __LEGION_CUDA_HD__
    coord_t get_index(void) const;
    __LEGION_CUDA_HD__
    int get_dim(void) const;
    __LEGION_CUDA_HD__
    inline bool exists(void) const { return (get_dim() > 0); }

    __LEGION_CUDA_HD__
    bool is_null(void) const;

    __LEGION_CUDA_HD__
    static DomainPoint nil(void);
  protected:
    template<typename T>
    __LEGION_CUDA_HD__ static inline coord_t check_for_overflow(const T& value);
  public:
    int dim;
    coord_t point_data[MAX_POINT_DIM];

    friend std::ostream& operator<<(std::ostream& os, const DomainPoint& dp);
  };

  /**
   * \class Domain
   * This is a type erased rectangle where the number of
   * dimensions is stored as a runtime value
   */
  class Domain {
  public:
    // Keep this in sync with legion_domain_max_rect_dim_t
    // in legion_config.h
    static constexpr int MAX_RECT_DIM = LEGION_MAX_DIM;
    __LEGION_CUDA_HD__
    Domain(void);
    __LEGION_CUDA_HD__
    Domain(const Domain& other);
    __LEGION_CUDA_HD__
    Domain(Domain&& other) noexcept;
    __LEGION_CUDA_HD__
    Domain(const DomainPoint& lo, const DomainPoint& hi);

    template<int DIM, typename T>
    __LEGION_CUDA_HD__ Domain(const Rect<DIM, T>& other);

    template<int DIM, typename T>
    __LEGION_CUDA_HD__ Domain(const DomainT<DIM, T>& other);

    __LEGION_CUDA_HD__
    Domain& operator=(const Domain& other);
    __LEGION_CUDA_HD__
    Domain& operator=(Domain&& other) noexcept;
    template<int DIM, typename T>
    __LEGION_CUDA_HD__ Domain& operator=(const Rect<DIM, T>& other);
    template<int DIM, typename T>
    __LEGION_CUDA_HD__ Domain& operator=(const DomainT<DIM, T>& other);

    __LEGION_CUDA_HD__
    bool operator==(const Domain& rhs) const;
    __LEGION_CUDA_HD__
    bool operator!=(const Domain& rhs) const;
    __LEGION_CUDA_HD__
    bool operator<(const Domain& rhs) const;

    __LEGION_CUDA_HD__
    Domain operator+(const DomainPoint& point) const;
    __LEGION_CUDA_HD__
    Domain& operator+=(const DomainPoint& point);

    __LEGION_CUDA_HD__
    Domain operator-(const DomainPoint& point) const;
    __LEGION_CUDA_HD__
    Domain& operator-=(const DomainPoint& point);

    static const Domain NO_DOMAIN;

    __LEGION_CUDA_HD__
    bool exists(void) const;
    __LEGION_CUDA_HD__
    bool dense(void) const;

    template<int DIM, typename T>
    __LEGION_CUDA_HD__ Rect<DIM, T> bounds(void) const;

    template<int DIM, typename T>
    __LEGION_CUDA_HD__ operator Rect<DIM, T>(void) const;

    template<int DIM, typename T>
    operator DomainT<DIM, T>(void) const;

    // Only works for structured DomainPoint.
    static Domain from_domain_point(const DomainPoint& p);

    // No longer supported
    // Realm::IndexSpace get_index_space(void) const;

    __LEGION_CUDA_HD__
    bool is_valid(void) const;

    bool contains(const DomainPoint& point) const;

    // This will only check the bounds and not the sparsity map
    __LEGION_CUDA_HD__
    bool contains_bounds_only(const DomainPoint& point) const;

    __LEGION_CUDA_HD__
    int get_dim(void) const;

    bool empty(void) const;

    // Will destroy the underlying Realm index space
    void destroy(Realm::Event wait_on = Realm::Event::NO_EVENT);

    size_t get_volume(void) const;

    __LEGION_CUDA_HD__
    DomainPoint lo(void) const;

    __LEGION_CUDA_HD__
    DomainPoint hi(void) const;

    // Intersects this Domain with another Domain and returns the result.
    Domain intersection(const Domain& other) const;

    // Returns the bounding box for this Domain and a point.
    // WARNING: only works with structured Domain.
    Domain convex_hull(const DomainPoint& p) const;
  private:
    struct IteratorInitFunctor;
    struct IteratorStepFunctor;
  public:
    class DomainPointIterator {
    public:
      DomainPointIterator(const Domain& d, bool fortran_order = true);
      DomainPointIterator(const DomainPointIterator& rhs);

      bool step(void);

      operator bool(void) const;
      DomainPoint& operator*(void);
      DomainPointIterator& operator=(const DomainPointIterator& rhs);
      DomainPointIterator& operator++(void);
      DomainPointIterator operator++(int /*i am postfix*/);
    public:
      DomainPoint p;
    private:
      friend struct IteratorInitFunctor;
      friend struct IteratorStepFunctor;
      // Realm's iterators are copyable by value so we can just always
      // copy them in and out of some buffers
      static_assert(std::is_trivially_copyable<
                    Realm::IndexSpaceIterator<MAX_RECT_DIM, coord_t> >::value);
      uint8_t
          is_iterator[sizeof(Realm::IndexSpaceIterator<MAX_RECT_DIM, coord_t>)];
      DomainPoint rect_lo, rect_hi;
      TypeTag is_type = 0;
      bool iter_valid = false;
      bool rect_valid = false;
      bool fortran_order = true;
    };
  protected:
  public:
    IDType is_id;
    // For Realm index spaces we need to have a type tag to know
    // what the type of the original sparsity map was
    // Without it you can get undefined behavior trying to interpret
    // the sparsity map incorrectly. This doesn't matter for the bounds
    // data because we've done the conversion for ourselves.
    // Technically this is redundant with the dimension since it also
    // encodes the dimension, but we'll keep them separate for now for
    // backwards compatibility
    TypeTag is_type;
    int dim;
    coord_t rect_data[2 * MAX_RECT_DIM];
  private:
    // Helper functor classes for demux-ing templates when we have
    // non-trivial sparsity maps with unusual types
    // User's should never need to look at these hence they are private
    template<typename T>
    __LEGION_CUDA_HD__ static inline coord_t check_for_overflow(const T& value);
    struct ContainsFunctor {
    public:
      ContainsFunctor(const Domain& d, const DomainPoint& p, bool& res)
        : domain(d), point(p), result(res)
      { }
      template<typename N, typename T>
      static inline void demux(ContainsFunctor* functor)
      {
        DomainT<N::N, T> is = functor->domain;
        Point<N::N, T> p = functor->point;
        functor->result = is.contains(p);
      }
    public:
      const Domain& domain;
      const DomainPoint& point;
      bool& result;
    };
    struct VolumeFunctor {
    public:
      VolumeFunctor(const Domain& d, size_t& r) : domain(d), result(r) { }
      template<typename N, typename T>
      static inline void demux(VolumeFunctor* functor)
      {
        DomainT<N::N, T> is = functor->domain;
        functor->result = is.volume();
      }
    public:
      const Domain& domain;
      size_t& result;
    };
    struct DestroyFunctor {
    public:
      DestroyFunctor(const Domain& d, Realm::Event e) : domain(d), event(e) { }
      template<typename N, typename T>
      static inline void demux(DestroyFunctor* functor)
      {
        DomainT<N::N, T> is = functor->domain;
        is.destroy(functor->event);
      }
    public:
      const Domain& domain;
      const Realm::Event event;
    };
    struct IntersectionFunctor {
    public:
      IntersectionFunctor(const Domain& l, const Domain& r, Domain& res)
        : lhs(l), rhs(r), result(res)
      { }
    public:
      template<typename N, typename T>
      static inline void demux(IntersectionFunctor* functor)
      {
        DomainT<N::N, T> is1 = functor->lhs;
        DomainT<N::N, T> is2 = functor->rhs;
        assert(is1.dense() || is2.dense());
        // Intersect the index spaces
        DomainT<N::N, T> result;
        result.bounds = is1.bounds.intersection(is2.bounds);
        if (!result.bounds.empty())
        {
          if (!is1.dense())
            result.sparsity = is1.sparsity;
          else if (!is2.dense())
            result.sparsity = is2.sparsity;
          else
            result.sparsity.id = 0;
        }
        else
          result.sparsity.id = 0;
        functor->result = Domain(result);
      }
    public:
      const Domain& lhs;
      const Domain& rhs;
      Domain& result;
    };
    struct IteratorInitFunctor {
    public:
      IteratorInitFunctor(const Domain& d, DomainPointIterator& i)
        : domain(d), iterator(i)
      { }
    public:
      template<typename N, typename T>
      static inline void demux(IteratorInitFunctor* functor)
      {
        DomainT<N::N, T> is = functor->domain;
        Realm::IndexSpaceIterator<N::N, T> is_itr(is);
        static_assert(N::N <= LEGION_MAX_DIM);
        static_assert(sizeof(T) <= sizeof(coord_t));
        static_assert(sizeof(is_itr) <= sizeof(functor->iterator.is_iterator));
        functor->iterator.rect_valid = is_itr.valid;
        if (is_itr.valid)
        {
          functor->iterator.rect_lo = is_itr.rect.lo;
          functor->iterator.rect_hi = is_itr.rect.hi;
          functor->iterator.p = is_itr.rect.lo;
          if (is_itr.step())
          {
            functor->iterator.iter_valid = true;
            std::memcpy(functor->iterator.is_iterator, &is_itr, sizeof(is_itr));
          }
          else
            functor->iterator.iter_valid = false;
        }
        else
          functor->iterator.iter_valid = false;
      }
    public:
      const Domain& domain;
      DomainPointIterator& iterator;
    };
    struct IteratorStepFunctor {
    public:
      IteratorStepFunctor(DomainPointIterator& i) : iterator(i) { }
    public:
      template<typename N, typename T>
      static inline void demux(IteratorStepFunctor* functor)
      {
        Realm::IndexSpaceIterator<N::N, T> is_itr;
        std::memcpy(&is_itr, functor->iterator.is_iterator, sizeof(is_itr));
        legion_assert(is_itr.valid);
        functor->iterator.p = is_itr.rect.lo;
        functor->iterator.rect_lo = is_itr.rect.lo;
        functor->iterator.rect_hi = is_itr.rect.hi;
        is_itr.step();
        functor->iterator.iter_valid = is_itr.valid;
        if (is_itr.valid)
          std::memcpy(functor->iterator.is_iterator, &is_itr, sizeof(is_itr));
      }
    public:
      DomainPointIterator& iterator;
    };
  };

  template<int DIM, typename COORD_T = coord_t>
  class PointInRectIterator {
  private:
    static_assert(DIM > 0, "DIM must be positive");
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    PointInRectIterator(void);
    __LEGION_CUDA_HD__
    PointInRectIterator(
        const Rect<DIM, COORD_T>& r, bool column_major_order = true);
  public:
    __LEGION_CUDA_HD__
    inline bool valid(void) const;
    __LEGION_CUDA_HD__
    inline bool step(void);
  public:
    __LEGION_CUDA_HD__
    inline bool operator()(void) const;
    __LEGION_CUDA_HD__
    inline Point<DIM, COORD_T> operator*(void) const;
    __LEGION_CUDA_HD__
    inline COORD_T operator[](unsigned index) const;
    __LEGION_CUDA_HD__
    inline const Point<DIM, COORD_T>* operator->(void) const;
    __LEGION_CUDA_HD__
    inline PointInRectIterator<DIM, COORD_T>& operator++(void);
    __LEGION_CUDA_HD__
    inline PointInRectIterator<DIM, COORD_T> operator++(int /*postfix*/);
  protected:
    Realm::PointInRectIterator<DIM, COORD_T> itr;
  };

  template<int DIM, typename COORD_T = coord_t>
  class RectInDomainIterator {
  private:
    static_assert(DIM > 0, "DIM must be positive");
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  public:
    RectInDomainIterator(void);
    RectInDomainIterator(const DomainT<DIM, COORD_T>& d);
  public:
    inline bool valid(void) const;
    inline bool step(void);
  public:
    inline bool operator()(void) const;
    inline Rect<DIM, COORD_T> operator*(void) const;
    inline const Rect<DIM, COORD_T>* operator->(void) const;
    inline RectInDomainIterator<DIM, COORD_T>& operator++(void);
    inline RectInDomainIterator<DIM, COORD_T> operator++(int /*postfix*/);
  protected:
    Realm::IndexSpaceIterator<DIM, COORD_T> itr;
  };

  template<int DIM, typename COORD_T = coord_t>
  class PointInDomainIterator {
  private:
    static_assert(DIM > 0, "DIM must be positive");
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  public:
    PointInDomainIterator(void);
    PointInDomainIterator(
        const DomainT<DIM, COORD_T>& d, bool column_major_order = true);
  public:
    inline bool valid(void) const;
    inline bool step(void);
  public:
    inline bool operator()(void) const;
    inline Point<DIM, COORD_T> operator*(void) const;
    inline COORD_T operator[](unsigned index) const;
    inline const Point<DIM, COORD_T>* operator->(void) const;
    inline PointInDomainIterator& operator++(void);
    inline PointInDomainIterator operator++(int /*postfix*/);
  protected:
    RectInDomainIterator<DIM, COORD_T> rect_itr;
    PointInRectIterator<DIM, COORD_T> point_itr;
    bool column_major;
  };

}  // namespace Legion

#include "legion/api/geometry.inl"

#endif  // __LEGION_GEOMETRY_H__
