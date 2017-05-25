/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_DOMAIN_H__
#define __LEGION_DOMAIN_H__

#include "realm.h"
#include "legion_types.h"

/**
 * \file legion_domain.h
 * This file provides some untyped representations of points
 * and domains as well as backwards compatibility types 
 * necessary for maintaining older versions of the runtime
 */

namespace Legion {

    /**
     * \class DomainPoint
     * This is a type erased point where the number of 
     * dimensions is a runtime value
     */
    class DomainPoint {
    public:
      enum { MAX_POINT_DIM = ::MAX_POINT_DIM };

      DomainPoint(void) : dim(0)
      {
        for (int i = 0; i < MAX_POINT_DIM; i++)
          point_data[i] = 0;
      }
      DomainPoint(coord_t index) : dim(0)
      {
        point_data[0] = index;
        for (int i = 1; i < MAX_POINT_DIM; i++)
          point_data[i] = 0;
      }

      DomainPoint(const DomainPoint &rhs) : dim(rhs.dim)
      {
        for (int i = 0; i < MAX_POINT_DIM; i++)
          point_data[i] = rhs.point_data[i];
      }

      DomainPoint(const Realm::DomainPoint &rhs) : dim(rhs.dim)
      {
        for (int i = 0; i < MAX_POINT_DIM; i++)
          point_data[i] = rhs.point_data[i];
      }

      operator Realm::DomainPoint(void) const
      {
        Realm::DomainPoint result;
        result.dim = dim;
        for (int i = 0; i < MAX_POINT_DIM; i++)
          result.point_data[i] = point_data[i];
        return result;
      }

      template<int DIM, typename T>
      DomainPoint(const Realm::ZPoint<DIM,T> &rhs) : dim(DIM)
      {
        for (int i = 0; i < DIM; i++)
          point_data[i] = rhs[i];
      }

      template<int DIM, typename T>
      operator Realm::ZPoint<DIM,T>(void) const
      {
        assert(DIM == dim);
        Realm::ZPoint<DIM,T> result;
        for (int i = 0; i < DIM; i++)
          result[i] = point_data[i];
        return result;
      }

      DomainPoint& operator=(const DomainPoint &rhs)
      {
        dim = rhs.dim;
        for (int i = 0; i < MAX_POINT_DIM; i++)
          point_data[i] = rhs.point_data[i];
        return *this;
      }

      bool operator==(const DomainPoint &rhs) const
      {
	if(dim != rhs.dim) return false;
	for(int i = 0; (i == 0) || (i < dim); i++)
	  if(point_data[i] != rhs.point_data[i]) return false;
	return true;
      }

      bool operator!=(const DomainPoint &rhs) const
      {
        return !((*this) == rhs);
      }

      bool operator<(const DomainPoint &rhs) const
      {
        if (dim < rhs.dim) return true;
        if (dim > rhs.dim) return false;
        for (int i = 0; (i == 0) || (i < dim); i++) {
          if (point_data[i] < rhs.point_data[i]) return true;
          if (point_data[i] > rhs.point_data[i]) return false;
        }
        return false;
      }

      coord_t& operator[](unsigned index)
      {
        assert(index < MAX_POINT_DIM);
        return point_data[index];
      }

      const coord_t& operator[](unsigned index) const
      {
        assert(index < MAX_POINT_DIM);
        return point_data[index];
      }

      struct STLComparator {
	bool operator()(const DomainPoint& a, const DomainPoint& b) const
	{
	  if(a.dim < b.dim) return true;
	  if(a.dim > b.dim) return false;
	  for(int i = 0; (i == 0) || (i < a.dim); i++) {
	    if(a.point_data[i] < b.point_data[i]) return true;
	    if(a.point_data[i] > b.point_data[i]) return false;
	  }
	  return false;
	}
      };

      template<int DIM>
      static DomainPoint from_point(
          typename LegionRuntime::Arrays::Point<DIM> p)
      {
	DomainPoint dp;
	assert(DIM <= MAX_POINT_DIM);
	dp.dim = DIM;
	p.to_array(dp.point_data);
	return dp;
      }

      Color get_color(void) const
      {
        assert(dim == 1);
        return point_data[0];
      }

      coord_t get_index(void) const
      {
	assert(dim == 0);
	return point_data[0];
      }

      int get_dim(void) const { return dim; }

      template <int DIM>
      LegionRuntime::Arrays::Point<DIM> get_point(void) const 
      { 
        assert(dim == DIM); 
        return LegionRuntime::Arrays::Point<DIM>(point_data); 
      }

      bool is_null(void) const { return (dim == -1); }

      static DomainPoint nil(void) { DomainPoint p; p.dim = -1; return p; }

    protected:
    public:
      int dim;
      coord_t point_data[MAX_POINT_DIM];

      friend std::ostream& operator<<(std::ostream& os, const DomainPoint& dp);
    };

    inline /*friend */std::ostream& operator<<(std::ostream& os,
					       const DomainPoint& dp)
    {
      switch(dp.dim) {
      case 0: { os << '[' << dp.point_data[0] << ']'; break; }
      case 1: { os << '(' << dp.point_data[0] << ')'; break; }
      case 2: { os << '(' << dp.point_data[0]
		   << ',' << dp.point_data[1] << ')'; break; }
      case 3: { os << '(' << dp.point_data[0]
		   << ',' << dp.point_data[1]
		   << ',' << dp.point_data[2] << ')'; break; }
      default: assert(0);
      }
      return os;
    }

    /**
     * \class Domain
     * This is a type erased rectangle where the number of 
     * dimensions is stored as a runtime value
     */
    class Domain {
    public:
      typedef ::legion_lowlevel_id_t IDType;
      // Keep this in sync with legion_lowlevel_domain_max_rect_dim_t
      // in lowlevel_config.h
      enum { MAX_RECT_DIM = ::MAX_RECT_DIM };
      Domain(void) : is_id(0), dim(0) {}
      Domain(const Domain& other) : is_id(other.is_id), dim(other.dim)
      {
	for(int i = 0; i < MAX_RECT_DIM*2; i++)
	  rect_data[i] = other.rect_data[i];
      }

      template<int DIM, typename T>
      Domain(const Realm::ZRect<DIM,T> &other) : is_id(0), dim(DIM)
      {
        for (int i = 0; i < DIM; i++)
          rect_data[i] = other.lo[i];
        for (int i = 0; i < DIM; i++)
          rect_data[DIM+i] = other.hi[i];
      }

      template<int DIM, typename T>
      Domain(const Realm::ZIndexSpace<DIM,T> &other)
        : is_id(other.sparsity.id), dim(DIM)
      {
        for (int i = 0; i < DIM; i++)
          rect_data[i] = other.bounds.lo[i];
        for (int i = 0; i < DIM; i++)
          rect_data[DIM+i] = other.bounds.hi[i];
      }

      Domain(const Realm::Domain &other) : is_id(other.is_id), dim(other.dim)
      {
        for(int i = 0; i < MAX_RECT_DIM*2; i++)
	  rect_data[i] = other.rect_data[i];
      }

      operator Realm::Domain(void) const
      {
        Realm::Domain result;
        result.is_id = is_id;
        result.dim = dim;
        for(int i = 0; i < MAX_RECT_DIM*2; i++)
	  result.rect_data[i] = rect_data[i];
        return result;
      }

      Domain& operator=(const Domain& other)
      {
	is_id = other.is_id;
	dim = other.dim;
	for(int i = 0; i < MAX_RECT_DIM*2; i++)
	  rect_data[i] = other.rect_data[i];
	return *this;
      }

      bool operator==(const Domain &rhs) const
      {
	if(is_id != rhs.is_id) return false;
	if(dim != rhs.dim) return false;
	for(int i = 0; i < dim; i++) {
	  if(rect_data[i*2] != rhs.rect_data[i*2]) return false;
	  if(rect_data[i*2+1] != rhs.rect_data[i*2+1]) return false;
	}
	return true;
      }

      bool operator<(const Domain &rhs) const
      {
        if(is_id < rhs.is_id) return true;
        if(is_id > rhs.is_id) return false;
        if(dim < rhs.dim) return true;
        if(dim > rhs.dim) return false;
        for(int i = 0; i < 2*dim; i++) {
          if(rect_data[i] < rhs.rect_data[i]) return true;
          if(rect_data[i] > rhs.rect_data[i]) return false;
        }
        return false; // otherwise they are equal
      }

      bool operator!=(const Domain &rhs) const { return !(*this == rhs); }

      static const Domain NO_DOMAIN;

      bool exists(void) const { return (dim > 0); }

      bool dense(void) const { return (is_id == 0); }

      template<int DIM>
      static Domain from_rect(typename LegionRuntime::Arrays::Rect<DIM> r)
      {
	Domain d;
	assert(DIM <= MAX_RECT_DIM);
	d.dim = DIM;
	r.to_array(d.rect_data);
	return d;
      }

      template<int DIM>
      static Domain from_point(typename LegionRuntime::Arrays::Point<DIM> p)
      {
        Domain d;
        assert(DIM <= MAX_RECT_DIM);
        d.dim = DIM;
        p.to_array(d.rect_data);
        p.to_array(d.rect_data+DIM);
        return d;
      }

      template<int DIM, typename T>
      operator Realm::ZRect<DIM,T>(void) const
      {
        assert(DIM == dim);
        assert(is_id == 0); // better not be one of these
        Realm::ZRect<DIM,T> result;
        for (int i = 0; i < DIM; i++)
          result.lo[i] = rect_data[i];
        for (int i = 0; i < DIM; i++)
          result.hi[i] = rect_data[DIM+i];
        return result;
      }

      template<int DIM, typename T>
      operator Realm::ZIndexSpace<DIM,T>(void) const
      {
        assert(DIM == dim);
        Realm::ZIndexSpace<DIM,T> result;
        result.sparsity.id = is_id;
        for (int i = 0; i < DIM; i++)
          result.bounds.lo[i] = rect_data[i];
        for (int i = 0; i < DIM; i++)
          result.bounds.hi[i] = rect_data[DIM+i];
        return result;
      }

      // Only works for structured DomainPoint.
      static Domain from_domain_point(const DomainPoint &p) {
        switch (p.dim) {
          case 0:
            assert(false);
          case 1:
            return Domain::from_point<1>(p.get_point<1>());
          case 2:
            return Domain::from_point<2>(p.get_point<2>());
          case 3:
            return Domain::from_point<3>(p.get_point<3>());
          default:
            assert(false);
        }
        return Domain::NO_DOMAIN;
      }

      // No longer supported
      //Realm::IndexSpace get_index_space(void) const;

      bool is_valid(void) const { return exists(); }

      bool contains(DomainPoint point) const
      {
        assert(point.get_dim() == dim);
        bool result = false;
        switch (dim)
        {
          case 1:
            {
              Realm::ZPoint<1,coord_t> p1 = point;
              Realm::ZIndexSpace<1,coord_t> is1 = *this;
              result = is1.contains(p1);
              break;
            }
          case 2:
            {
              Realm::ZPoint<2,coord_t> p2 = point;
              Realm::ZIndexSpace<2,coord_t> is2 = *this;
              result = is2.contains(p2);
              break;
            }
          case 3:
            {
              Realm::ZPoint<3,coord_t> p3 = point;
              Realm::ZIndexSpace<3,coord_t> is3 = *this;
              result = is3.contains(p3);
              break;
            }
          default:
            assert(false);
        }
        return result;
      }

      int get_dim(void) const { return dim; }

      size_t get_volume(void) const
      {
        switch (dim)
        {
          case 1:
            {
              Realm::ZIndexSpace<1,coord_t> is = *this;
              return is.volume();
            }
          case 2:
            {
              Realm::ZIndexSpace<2,coord_t> is = *this;
              return is.volume();
            }
          case 3:
            {
              Realm::ZIndexSpace<3,coord_t> is = *this;
              return is.volume();
            }
          default:
            assert(false);
        }
        return 0;
      }

      // Intersects this Domain with another Domain and returns the result.
      // WARNING: currently only works with structured Domains.
      Domain intersection(const Domain &other) const
      {
        assert(dim == other.dim);
        Realm::ProfilingRequestSet dummy_requests;
        switch (dim)
        {
          case 1:
            {
              Realm::ZIndexSpace<1,coord_t> is1 = *this;
              Realm::ZIndexSpace<1,coord_t> is2 = other;
              Realm::ZIndexSpace<1,coord_t> result;
              LgEvent wait_on( 
                Realm::ZIndexSpace<1,coord_t>::compute_intersection(is1,is2,
                                                      result,dummy_requests));
              if (wait_on.exists())
                wait_on.lg_wait();
              return Domain(result);
            }
          case 2:
            {
              Realm::ZIndexSpace<2,coord_t> is1 = *this;
              Realm::ZIndexSpace<2,coord_t> is2 = other;
              Realm::ZIndexSpace<2,coord_t> result;
              LgEvent wait_on(
                Realm::ZIndexSpace<2,coord_t>::compute_intersection(is1,is2,
                                                      result,dummy_requests));
              if (wait_on.exists())
                wait_on.lg_wait();
              return Domain(result);
            }
          case 3:
            {
              Realm::ZIndexSpace<3,coord_t> is1 = *this;
              Realm::ZIndexSpace<3,coord_t> is2 = other;
              Realm::ZIndexSpace<3,coord_t> result;
              LgEvent wait_on(
                Realm::ZIndexSpace<3,coord_t>::compute_intersection(is1,is2,
                                                      result,dummy_requests));
              if (wait_on.exists())
                wait_on.lg_wait();
              return Domain(result);
            }
          default:
            assert(false);
        }
        return Domain::NO_DOMAIN;
      }

      // Returns the bounding box for this Domain and a point.
      // WARNING: only works with structured Domain.
      Domain convex_hull(const DomainPoint &p) const
      {
        assert(dim == p.dim);
        Realm::ProfilingRequestSet dummy_requests;
        switch (dim)
        {
          case 1:
            {
              Realm::ZIndexSpace<1,coord_t> is1 = *this;
              Realm::ZIndexSpace<1,coord_t> is2(Realm::ZRect<1,coord_t>(p, p));
              Realm::ZIndexSpace<1,coord_t> result;
              LgEvent wait_on( 
                Realm::ZIndexSpace<1,coord_t>::compute_union(is1,is2,
                                              result,dummy_requests));
              if (wait_on.exists())
                wait_on.lg_wait();
              return Domain(result);
            }
          case 2:
            {
              Realm::ZIndexSpace<2,coord_t> is1 = *this;
              Realm::ZIndexSpace<2,coord_t> is2(Realm::ZRect<2,coord_t>(p, p));
              Realm::ZIndexSpace<2,coord_t> result;
              LgEvent wait_on(
                Realm::ZIndexSpace<2,coord_t>::compute_union(is1,is2,
                                              result,dummy_requests));
              if (wait_on.exists())
                wait_on.lg_wait();
              return Domain(result);
            }
          case 3:
            {
              Realm::ZIndexSpace<3,coord_t> is1 = *this;
              Realm::ZIndexSpace<3,coord_t> is2(Realm::ZRect<3,coord_t>(p, p));
              Realm::ZIndexSpace<3,coord_t> result;
              LgEvent wait_on(
                Realm::ZIndexSpace<3,coord_t>::compute_union(is1,is2,
                                              result,dummy_requests));
              if (wait_on.exists())
                wait_on.lg_wait();
              return Domain(result);
            }
          default:
            assert(false);
        }
        return Domain::NO_DOMAIN;
      }

      template <int DIM>
      LegionRuntime::Arrays::Rect<DIM> get_rect(void) const 
      {
        assert(DIM > 0);
        assert(DIM == dim);
	if(is_id == 0) {
	  return LegionRuntime::Arrays::Rect<DIM>(rect_data); 
	} else {
	  // TODO: would be nice to do this only once rather than each call
	  Realm::ZIndexSpace<DIM,coord_t> tight = (operator Realm::ZIndexSpace<DIM,coord_t>()).tighten();
	  assert(tight.dense());
	  LegionRuntime::Arrays::Rect<DIM> r;
	  for(int i = 0; i < DIM; i++) {
	    r.lo.x[i] = tight.bounds.lo[i];
	    r.hi.x[i] = tight.bounds.hi[i];
	  }
	  return r;
	}
      }

      class DomainPointIterator {
      public:
        DomainPointIterator(const Domain& d)
	{
	  p.dim = d.get_dim();
	  switch(p.get_dim()) {
	  case 1:
	    {
              Realm::ZIndexSpaceIterator<1,coord_t> *is_itr = 
                new Realm::ZIndexSpaceIterator<1,coord_t>(d);
	      is_iterator = (void *)is_itr;
              is_valid = is_itr->valid;
              if (is_valid) {
                Realm::ZPointInRectIterator<1,coord_t> *rect_itr = 
                  new Realm::ZPointInRectIterator<1,coord_t>(is_itr->rect);
                rect_iterator = (void *)rect_itr;
                rect_valid = rect_itr->valid;
                p = rect_itr->p; 
              } else {
                rect_iterator = NULL;
                rect_valid = false;
              }
	      break;
	    }
	  case 2:
	    {
              Realm::ZIndexSpaceIterator<2,coord_t> *is_itr = 
                new Realm::ZIndexSpaceIterator<2,coord_t>(d);
	      is_iterator = (void *)is_itr;
              is_valid = is_itr->valid;
              if (is_valid) {
                Realm::ZPointInRectIterator<2,coord_t> *rect_itr = 
                  new Realm::ZPointInRectIterator<2,coord_t>(is_itr->rect);
                rect_iterator = (void *)rect_itr;
                rect_valid = rect_itr->valid;
                p = rect_itr->p; 
              } else {
                rect_iterator = NULL;
                rect_valid = false;
              }
	      break;
	    }
	  case 3:
	    {
              Realm::ZIndexSpaceIterator<3,coord_t> *is_itr = 
                new Realm::ZIndexSpaceIterator<3,coord_t>(d);
	      is_iterator = (void *)is_itr;
              is_valid = is_itr->valid;
              if (is_valid) {
                Realm::ZPointInRectIterator<3,coord_t> *rect_itr = 
                  new Realm::ZPointInRectIterator<3,coord_t>(is_itr->rect);
                rect_iterator = (void *)rect_itr;
                rect_valid = rect_itr->valid;
                p = rect_itr->p; 
              } else {
                rect_iterator = NULL;
                rect_valid = false;
              }
	      break;
	    }
	  default:
	    assert(0);
	  };
	}

	~DomainPointIterator(void)
	{
	  switch(p.get_dim()) {
	  case 1:
	    {
              Realm::ZIndexSpaceIterator<1,coord_t> *is_itr = 
                (Realm::ZIndexSpaceIterator<1,coord_t>*)is_iterator;
              delete is_itr;
              if (rect_iterator) {
                Realm::ZPointInRectIterator<1,coord_t> *rect_itr = 
                  (Realm::ZPointInRectIterator<1,coord_t>*)rect_iterator;
                delete rect_itr;
              }
              break;
	    }
	  case 2:
	    {
              Realm::ZIndexSpaceIterator<2,coord_t> *is_itr = 
                (Realm::ZIndexSpaceIterator<2,coord_t>*)is_iterator;
              delete is_itr;
              if (rect_iterator) {
                Realm::ZPointInRectIterator<2,coord_t> *rect_itr = 
                  (Realm::ZPointInRectIterator<2,coord_t>*)rect_iterator;
                delete rect_itr;
              }
              break;
	    }
	  case 3:
	    {
              Realm::ZIndexSpaceIterator<3,coord_t> *is_itr = 
                (Realm::ZIndexSpaceIterator<3,coord_t>*)is_iterator;
              delete is_itr;
              if (rect_iterator) {
                Realm::ZPointInRectIterator<3,coord_t> *rect_itr = 
                  (Realm::ZPointInRectIterator<3,coord_t>*)rect_iterator;
                delete rect_itr;
              }
              break;
	    }
	  default:
	    assert(0);
	  }
	}

        bool step(void)
	{
          assert(is_valid && rect_valid);
	  switch(p.get_dim()) {
	  case 1:
	    {
              // Step the rect iterator first
              Realm::ZPointInRectIterator<1,coord_t> *rect_itr = 
                (Realm::ZPointInRectIterator<1,coord_t>*)rect_iterator;
              rect_itr->step();
              rect_valid = rect_itr->valid;
              if (!rect_valid) {
                // If the rectangle iterator is not valid anymore
                // then try to start the next rectangle
                delete rect_itr;
                Realm::ZIndexSpaceIterator<1,coord_t> *is_itr = 
                  (Realm::ZIndexSpaceIterator<1,coord_t>*)is_iterator;
                is_itr->step();
                is_valid = is_itr->valid;
                if (is_valid) {
                  rect_itr = 
                    new Realm::ZPointInRectIterator<1,coord_t>(is_itr->rect);
                  p = rect_itr->p;
                  rect_valid = rect_itr->valid;
                } else {
                  rect_itr = NULL;
                  rect_valid = false;
                }
                rect_iterator = (void *)rect_itr;
              } else {
                p = rect_itr->p; 
              }
              break;
	    }
	  case 2:
	    {
              // Step the rect iterator first
              Realm::ZPointInRectIterator<2,coord_t> *rect_itr = 
                (Realm::ZPointInRectIterator<2,coord_t>*)rect_iterator;
              rect_itr->step();
              rect_valid = rect_itr->valid;
              if (!rect_valid) {
                // If the rectangle iterator is not valid anymore
                // then try to start the next rectangle
                delete rect_itr;
                Realm::ZIndexSpaceIterator<2,coord_t> *is_itr = 
                  (Realm::ZIndexSpaceIterator<2,coord_t>*)is_iterator;
                is_itr->step();
                is_valid = is_itr->valid;
                if (is_valid) {
                  rect_itr = 
                    new Realm::ZPointInRectIterator<2,coord_t>(is_itr->rect);
                  p = rect_itr->p;
                  rect_valid = rect_itr->valid;
                } else {
                  rect_itr = NULL;
                  rect_valid = false;
                }
                rect_iterator = (void *)rect_itr;
              } else {
                p = rect_itr->p; 
              }
              break;
	    }
	  case 3:
	    {
              // Step the rect iterator first
              Realm::ZPointInRectIterator<3,coord_t> *rect_itr = 
                (Realm::ZPointInRectIterator<3,coord_t>*)rect_iterator;
              rect_itr->step();
              rect_valid = rect_itr->valid;
              if (!rect_valid) {
                // If the rectangle iterator is not valid anymore
                // then try to start the next rectangle
                delete rect_itr;
                Realm::ZIndexSpaceIterator<3,coord_t> *is_itr = 
                  (Realm::ZIndexSpaceIterator<3,coord_t>*)is_iterator;
                is_itr->step();
                is_valid = is_itr->valid;
                if (is_valid) {
                  rect_itr = 
                    new Realm::ZPointInRectIterator<3,coord_t>(is_itr->rect);
                  p = rect_itr->p;
                  rect_valid = rect_itr->valid;
                } else {
                  rect_itr = NULL;
                  rect_valid = false;
                }
                rect_iterator = (void *)rect_itr;
              } else {
                p = rect_itr->p; 
              }
              break;
	    }
	  default:
	    assert(0);
	  }
          return is_valid && rect_valid;
	}

	operator bool(void) const { return is_valid && rect_valid; }
	DomainPointIterator& operator++(int /*i am postfix*/) 
          { step(); return *this; }
      public:
        DomainPoint p;
        void *is_iterator, *rect_iterator;
        bool is_valid, rect_valid;
      };
    protected:
    public:
      IDType is_id;
      int dim;
      coord_t rect_data[2 * MAX_RECT_DIM];
    };

    inline std::ostream& operator<<(std::ostream& os, Domain d) 
    {
      switch(d.get_dim()) {
      case 1: return os << d.get_rect<1>();
      case 2: return os << d.get_rect<2>();
      case 3: return os << d.get_rect<3>();
      default: assert(0);
      }
      return os;
    }

    template<int DIM, typename COORD_T>
    class DomainIterator {
    public:
#if __cplusplus < 201103L
      DomainIterator(const Realm::ZIndexSpace<DIM,COORD_T> &d,
                     bool column_major_order = true)
#else
      DomainIterator(const DomainT<DIM,COORD_T> &d,
                     bool column_major_order = true)
#endif
        : is_itr(d), column_major(column_major_order)
      {
        is_valid = is_itr.is_valid;
        if (is_valid) {
          rect_itr = 
            Realm::ZPointInRectIterator<DIM,COORD_T>(is_itr.rect, column_major);
          rect_valid = rect_itr.valid;
          p = rect_itr.p;
        } else
          rect_valid = false;
      }
    public:
      inline bool step(void) 
      {
        assert(is_valid && rect_valid);
        // Step the rect iterator first
        rect_itr.step();
        rect_valid = rect_itr.valid;
        if (!rect_valid) {
          // Rect iterator is no longer valid, start the
          // next rectangle if there is one
          is_itr.step();
          is_valid = is_itr.valid;
          if (is_valid) {
            rect_itr = Realm::ZPointInRectIterator<DIM,COORD_T>(is_itr.rect,
                                                                column_major);
            p = rect_itr.p;
            rect_valid = rect_itr.valid;
          } else {
            rect_valid = false;
          }
        } else {
          p = rect_itr.p;
        }
      }
    public:
      inline operator bool(void) const { return is_valid && rect_valid; }
      inline DomainIterator& operator++(int /*i am postfix*/) 
        { step(); return *this; }
    public:
#if __cplusplus < 201103L
      Realm::ZPoint<DIM,COORD_T> p;
#else
      Point<DIM,COORD_T> p;
#endif
    protected:
      Realm::ZIndexSpaceIterator<DIM,COORD_T> is_itr;
      Realm::ZPointInRectIterator<DIM,COORD_T> rect_itr;
      bool is_valid, rect_valid;
      bool column_major;
    };

    // This class exists for some very minimal backwards compatibility
    class IndexSpaceAllocator {
    public:
      IndexSpaceAllocator(const Domain &d)
        : iterator(Domain::DomainPointIterator(d)) { assert(d.get_dim() == 1); }
    public:
      LEGION_DEPRECATED("Dynamic allocation is no longer supported.")
      coord_t alloc(size_t count = 1) const 
        { assert(iterator); coord_t result = iterator.p[0]; 
          for (size_t i = 0; i < count; i++) iterator.step();
          return result; }
      LEGION_DEPRECATED("Dynamic allocation is no longer supported.")
      void reserve(coord_t ptr, size_t count = 1) const
        { printf("No backwards compatibility for 'reserve' "
                 "on index space allocators.\n"); assert(false); }
      LEGION_DEPRECATED("Dynamic allocation is no longer supported.")
      void free(coord_t ptr, size_t count = 1) const
        { printf("No backwards compatibility for 'free' "
                 "on index space allocators.\n"); assert(false); }
    protected:
      mutable Domain::DomainPointIterator iterator;
    };

}; // namespace Legion

#endif // __LEGION_DOMAIN_H__

