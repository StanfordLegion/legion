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

#ifndef __LEGION_BACKWARDS_H__
#define __LEGION_BACKWARDS_H__

#include "realm.h"

/**
 * \file legion_backwards.h
 * This file provides some backwards compatibility types necessary
 * for maintaining older versions of the runtime
 */

namespace Legion {

    /**
     * \class DomainPoint
     * This is a type erased point where the number of 
     * dimensions is a runtime value
     */
    class DomainPoint {
    public:
      enum { MAX_POINT_DIM = 3 };

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
      // For backwards compatibility
      typedef Realm::IndexSpace IndexSpace;
      typedef Realm::ElementMask ElementMask;
      typedef Realm::RegionInstance RegionInstance;
      typedef Realm::Event Event;
      typedef Realm::ProfilingRequestSet ProfilingRequestSet;
      typedef Realm::Domain::CopySrcDstField CopySrcDstField;

      // Keep this in sync with legion_lowlevel_domain_max_rect_dim_t
      // in lowlevel_config.h
      enum { MAX_RECT_DIM = ::MAX_RECT_DIM };
      Domain(void) : is_id(0), dim(0) {}
      Domain(IndexSpace is) : is_id(is.id), dim(0) {}
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

      bool exists(void) const { return (is_id != 0) || (dim > 0); }

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

      size_t compute_size(void) const
      {
        size_t result;
        if (dim == 0)
          result = (2 * sizeof(IDType));
        else
          result = ((1 + 2 * dim) * sizeof(IDType));
        return result;
      }

      IDType *serialize(IDType *data) const
      {
	*data++ = dim;
	if(dim == 0) {
	  *data++ = is_id;
	} else {
	  for(int i = 0; i < dim*2; i++)
	    *data++ = rect_data[i];
	}
	return data;
      }

      const IDType *deserialize(const IDType *data)
      {
	dim = *data++;
	if(dim == 0) {
	  is_id = *data++;
	} else {
	  for(int i = 0; i < dim*2; i++)
	    rect_data[i] = *data++;
	}
	return data;
      }

      IndexSpace get_index_space(void) const
      {
        if (is_id)
        {
          IndexSpace is;
          is.id = static_cast<IDType>(is_id);
          return is;
        }
        return IndexSpace::NO_SPACE;
      }

      bool is_valid(void) const
      {
        switch (dim)
        {
          case -1:
            return false;
          case 0:
            {
              if (is_id)
              {
                IndexSpace is;
                is.id = static_cast<IDType>(is_id);
                return is.exists();
              }
              return false;
            }
          case 3:
            {
              if (rect_data[4] > rect_data[5])
                return false;
            }
          case 2:
            {
              if (rect_data[2] > rect_data[3])
                return false;
            }
          case 1:
            {
              if (rect_data[0] > rect_data[1])
                return false;
              break;
            }
          default:
            assert(false);
        }
        return true;
      }

      bool contains(DomainPoint point) const
      {
        bool result = false;
        switch (dim)
        {
          case -1:
            break;
          case 0:
            {
              const ElementMask &mask = get_index_space().get_valid_mask();
              result = mask.is_set(point.point_data[0]);
              break;
            }
          case 1:
            {
              LegionRuntime::Arrays::Point<1> p1 = point.get_point<1>();
              LegionRuntime::Arrays::Rect<1> r1 = get_rect<1>();
              result = r1.contains(p1);
              break;
            }
          case 2:
            {
              LegionRuntime::Arrays::Point<2> p2 = point.get_point<2>();
              LegionRuntime::Arrays::Rect<2> r2 = get_rect<2>();
              result = r2.contains(p2);
              break;
            }
          case 3:
            {
              LegionRuntime::Arrays::Point<3> p3 = point.get_point<3>();
              LegionRuntime::Arrays::Rect<3> r3 = get_rect<3>();
              result = r3.contains(p3);
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
          case 0:
            return get_index_space().get_valid_mask().pop_count();
          case 1:
            {
              LegionRuntime::Arrays::Rect<1> r1 = get_rect<1>();
              return r1.volume();
            }
          case 2:
            {
              LegionRuntime::Arrays::Rect<2> r2 = get_rect<2>();
              return r2.volume();
            }
          case 3:
            {
              LegionRuntime::Arrays::Rect<3> r3 = get_rect<3>();
              return r3.volume();
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

        switch (dim)
        {
          case 0:
            assert(false);
          case 1:
            return Domain::from_rect<1>(get_rect<1>().intersection(
                                              other.get_rect<1>()));
          case 2:
            return Domain::from_rect<2>(get_rect<2>().intersection(
                                              other.get_rect<2>()));
          case 3:
            return Domain::from_rect<3>(get_rect<3>().intersection(
                                              other.get_rect<3>()));
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

        switch (dim)
        {
          case 0:
            assert(false);
          case 1:
            {
              LegionRuntime::Arrays::Point<1> pt = p.get_point<1>();
              return Domain::from_rect<1>(get_rect<1>().convex_hull(
                    LegionRuntime::Arrays::Rect<1>(pt, pt)));
             }
          case 2:
            {
              LegionRuntime::Arrays::Point<2> pt = p.get_point<2>();
              return Domain::from_rect<2>(get_rect<2>().convex_hull(
                    LegionRuntime::Arrays::Rect<2>(pt, pt)));
            }
          case 3:
            {
              LegionRuntime::Arrays::Point<3> pt = p.get_point<3>();
              return Domain::from_rect<3>(get_rect<3>().convex_hull(
                    LegionRuntime::Arrays::Rect<3>(pt, pt)));
            }
          default:
            assert(false);
        }
        return Domain::NO_DOMAIN;
      }

      template <int DIM>
      LegionRuntime::Arrays::Rect<DIM> get_rect(void) const 
      { 
        assert(dim == DIM); 
        return LegionRuntime::Arrays::Rect<DIM>(rect_data); 
      }

      class DomainPointIterator {
      public:
        DomainPointIterator(const Domain& d)
	{
	  p.dim = d.get_dim();
	  switch(p.get_dim()) {
	  case 0: // index space
	    {
	      const ElementMask *mask = &(d.get_index_space().get_valid_mask());
	      iterator = (void *)mask;
              int index = mask->find_enabled();
	      p.point_data[0] = index;
	      any_left = (index >= 0);
	    }
	    break;

	  case 1:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<1> *pir = 
                new LegionRuntime::Arrays::GenericPointInRectIterator<1>(
                                                          d.get_rect<1>());
	      iterator = (void *)pir;
	      pir->p.to_array(p.point_data);
	      any_left = pir->any_left;
	    }
	    break;

	  case 2:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<2> *pir = 
                new LegionRuntime::Arrays::GenericPointInRectIterator<2>(
                                                          d.get_rect<2>());
	      iterator = (void *)pir;
	      pir->p.to_array(p.point_data);
	      any_left = pir->any_left;
	    }
	    break;

	  case 3:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<3> *pir = 
                new LegionRuntime::Arrays::GenericPointInRectIterator<3>(
                                                          d.get_rect<3>());
	      iterator = (void *)pir;
	      pir->p.to_array(p.point_data);
	      any_left = pir->any_left;
	    }
	    break;

	  default:
	    assert(0);
	  };
	}

	~DomainPointIterator(void)
	{
	  switch(p.get_dim()) {
	  case 0:
	    // nothing to do
	    break;

	  case 1:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<1> *pir = 
               (LegionRuntime::Arrays::GenericPointInRectIterator<1> *)iterator;
	      delete pir;
	    }
	    break;

	  case 2:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<2> *pir = 
               (LegionRuntime::Arrays::GenericPointInRectIterator<2> *)iterator;
	      delete pir;
	    }
	    break;

	  case 3:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<3> *pir = 
               (LegionRuntime::Arrays::GenericPointInRectIterator<3> *)iterator;
	      delete pir;
	    }
	    break;

	  default:
	    assert(0);
	  }
	}

	DomainPoint p;
	bool any_left;
	void *iterator;

	bool step(void)
	{
	  switch(p.get_dim()) {
	  case 0:
	    {
	      const ElementMask *mask = (const ElementMask *)iterator;
	      int index = mask->find_enabled(1, p.point_data[0] + 1);
	      p.point_data[0] = index;
	      any_left = (index >= 0);
	    }
	    break;

	  case 1:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<1> *pir = 
               (LegionRuntime::Arrays::GenericPointInRectIterator<1> *)iterator;
	      any_left = pir->step();
	      pir->p.to_array(p.point_data);
	    }
	    break;

	  case 2:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<2> *pir = 
               (LegionRuntime::Arrays::GenericPointInRectIterator<2> *)iterator;
	      any_left = pir->step();
	      pir->p.to_array(p.point_data);
	    }
	    break;

	  case 3:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<3> *pir = 
               (LegionRuntime::Arrays::GenericPointInRectIterator<3> *)iterator;
	      any_left = pir->step();
	      pir->p.to_array(p.point_data);
	    }
	    break;

	  default:
	    assert(0);
	  }

	  return any_left;
	}

	operator bool(void) const { return any_left; }
	DomainPointIterator& operator++(int /*i am postfix*/) 
          { step(); return *this; }
      };
    public:
      // simple instance creation for the lazy
      RegionInstance create_instance(Memory memory, size_t elem_size,
			             ReductionOpID redop_id = 0) const;

      RegionInstance create_instance(Memory memory,
				     const std::vector<size_t> &field_sizes,
				     size_t block_size,
				     ReductionOpID redop_id = 0) const;

      RegionInstance create_instance(Memory memory, size_t elem_size,
                                     const ProfilingRequestSet &reqs,
                                     ReductionOpID redop_id = 0) const;

      RegionInstance create_instance(Memory memory,
				     const std::vector<size_t> &field_sizes,
				     size_t block_size,
                                     const ProfilingRequestSet &reqs,
				     ReductionOpID redop_id = 0) const;

#ifdef REALM_USE_LEGION_LAYOUT_CONSTRAINTS
      // Note that the constraints are not const so that Realm can add
      // to the set with additional constraints describing the exact 
      // instance that was created.
      Event create_instance(RegionInstance &result,
              const std::vector<std::pair<unsigned/*FieldID*/,size_t> > &fields,
              const Legion::LayoutConstraintSet &constraints, 
              const ProfilingRequestSet &reqs) const;
#endif

      RegionInstance create_hdf5_instance(const char *file_name,
                                          const std::vector<size_t> &field_sizes,
                                          const std::vector<const char*> &field_files,
                                          bool read_only) const;
      RegionInstance create_file_instance(const char *file_name,
                                          const std::vector<size_t> &field_sizes,
                                          legion_lowlevel_file_mode_t file_mode) const;

      Event fill(const std::vector<CopySrcDstField> &dsts,
                 const void *fill_value, size_t fill_value_size,
                 Event wait_on = Event::NO_EVENT) const;

      Event copy(RegionInstance src_inst, RegionInstance dst_inst,
		 size_t elem_size, Event wait_on = Event::NO_EVENT,
		 ReductionOpID redop_id = 0, bool red_fold = false) const;

      Event copy(const std::vector<CopySrcDstField>& srcs,
		 const std::vector<CopySrcDstField>& dsts,
		 Event wait_on = Event::NO_EVENT,
		 ReductionOpID redop_id = 0, bool red_fold = false) const;

      Event copy(const std::vector<CopySrcDstField>& srcs,
		 const std::vector<CopySrcDstField>& dsts,
		 const ElementMask& mask,
		 Event wait_on = Event::NO_EVENT,
		 ReductionOpID redop_id = 0, bool red_fold = false) const;

      Event copy_indirect(const CopySrcDstField& idx,
			  const std::vector<CopySrcDstField>& srcs,
			  const std::vector<CopySrcDstField>& dsts,
			  Event wait_on = Event::NO_EVENT,
			  ReductionOpID redop_id = 0, bool red_fold = false) const;

      Event copy_indirect(const CopySrcDstField& idx,
			  const std::vector<CopySrcDstField>& srcs,
			  const std::vector<CopySrcDstField>& dsts,
			  const ElementMask& mask,
			  Event wait_on = Event::NO_EVENT,
			  ReductionOpID redop_id = 0, bool red_fold = false) const;

      // Variants of the above for profiling
      Event fill(const std::vector<CopySrcDstField> &dsts,
                 const ProfilingRequestSet &requests,
                 const void *fill_value, size_t fill_value_size,
                 Event wait_on = Event::NO_EVENT) const;

      Event copy(const std::vector<CopySrcDstField>& srcs,
		 const std::vector<CopySrcDstField>& dsts,
                 const ProfilingRequestSet &reqeusts,
		 Event wait_on = Event::NO_EVENT,
		 ReductionOpID redop_id = 0, bool red_fold = false) const;
    protected:
    public:
      IDType is_id;
      int dim;
      coord_t rect_data[2 * MAX_RECT_DIM];
    };

    inline std::ostream& operator<<(std::ostream& os, Domain d) 
    {
      switch(d.get_dim()) {
      case 0: return os << d.get_index_space();
      case 1: return os << d.get_rect<1>();
      case 2: return os << d.get_rect<2>();
      case 3: return os << d.get_rect<3>();
      default: assert(0);
      }
      return os;
    }

}; // namespace Legion

#endif // __LEGION_BACKWARDS_H__

