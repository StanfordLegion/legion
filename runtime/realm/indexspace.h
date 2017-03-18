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

// index spaces for Realm

#ifndef REALM_INDEXSPACE_H
#define REALM_INDEXSPACE_H

#include "event.h"
#include "memory.h"
#include "instance.h"

#include "lowlevel_config.h"
#include "arrays.h"
//#include "layouts.h"
#include "custom_serdez.h"

#ifdef REALM_USE_LEGION_LAYOUT_CONSTRAINTS
#include "legion_realm.h" // forward declarations for legion types
#endif

namespace Realm {
  typedef ::legion_lowlevel_coord_t coord_t;

  class ProfilingRequestSet;

    class ElementMask {
    public:
      ElementMask(void);
      explicit ElementMask(size_t num_elements, coord_t first_element = 0);
      explicit ElementMask(const ElementMask &copy_from, size_t num_elements, coord_t first_element = -1LL);
      ElementMask(const ElementMask &copy_from, bool trim = false);
      ~ElementMask(void);

      void init(coord_t _first_element, size_t _num_elements, Memory _memory, coord_t _offset);

      coord_t get_first_element(void) const { return first_element; }
      size_t get_num_elmts(void) const { return num_elements; }

      void enable(coord_t start, size_t count = 1);
      void disable(coord_t start, size_t count = 1);

      coord_t find_enabled(size_t count = 1, coord_t start = 0) const;
      coord_t find_disabled(size_t count = 1, coord_t start = 0) const;

      bool is_set(coord_t ptr) const;
      size_t pop_count(bool enabled = true) const;
      bool operator!(void) const;
      bool operator==(const ElementMask &other) const;
      bool operator!=(const ElementMask &other) const;
      // union/intersect/subtract?
      ElementMask operator|(const ElementMask &other) const;
      ElementMask operator&(const ElementMask &other) const;
      ElementMask operator-(const ElementMask &other) const;

      ElementMask& operator|=(const ElementMask &other);
      ElementMask& operator&=(const ElementMask &other);
      ElementMask& operator-=(const ElementMask &other);

      coord_t first_enabled(void) const { return first_enabled_elmt; }
      coord_t last_enabled(void) const { return last_enabled_elmt; }

      ElementMask& operator=(const ElementMask &rhs);

      enum OverlapResult { OVERLAP_NO, OVERLAP_MAYBE, OVERLAP_YES };

      OverlapResult overlaps_with(const ElementMask& other,
				  coord_t max_effort = -1LL) const;

      ElementMask intersect_with(const ElementMask &other);

      class Enumerator {
      public:
	Enumerator(const ElementMask& _mask, coord_t _start, int _polarity);
	~Enumerator(void);

	bool get_next(coord_t &position, size_t &length);
	bool peek_next(coord_t &position, size_t &length);

      protected:
	const ElementMask& mask;
	coord_t pos;
	int polarity;
      };

      Enumerator *enumerate_enabled(coord_t start = 0) const;
      Enumerator *enumerate_disabled(coord_t start = 0) const;

      size_t raw_size(void) const;
      const void *get_raw(void) const;
      void set_raw(const void *data);

      // Implementations below
      template <class T>
      static size_t forall_ranges(T &executor,
                                  const ElementMask &mask,
			          coord_t start = 0, coord_t count = -1LL,
			          bool do_enabled = true);

      template <class T>
      static size_t forall_ranges(T &executor,
                                  const ElementMask &mask1,
			          const ElementMask &mask2,
			          coord_t start = 0, coord_t count = -1LL,
			          bool do_enabled1 = true,
			          bool do_enabled2 = true);

    public:
      void recalc_first_last_enabled(void);

      friend class Enumerator;
      coord_t first_element;
      size_t num_elements;
      Memory memory;
      coord_t offset;
      char *raw_data;
      coord_t first_enabled_elmt, last_enabled_elmt;
    };

    class IndexSpaceAllocator;
    class DomainPoint;
    class Domain;

    class IndexSpace {
    public:
      typedef ::legion_lowlevel_id_t id_t;
      id_t id;
      bool operator<(const IndexSpace &rhs) const { return id < rhs.id; }
      bool operator==(const IndexSpace &rhs) const { return id == rhs.id; }
      bool operator!=(const IndexSpace &rhs) const { return id != rhs.id; }

      static const IndexSpace NO_SPACE;

      bool exists(void) const { return id != 0; }

      static IndexSpace create_index_space(size_t num_elmts);
      static IndexSpace create_index_space(const ElementMask &mask);
      static IndexSpace create_index_space(IndexSpace parent,
					   const ElementMask &mask,
                                           bool allocable = true);

      static IndexSpace expand_index_space(IndexSpace child,
					   size_t num_elmts,
					   coord_t child_offset = 0);

      void destroy(Event wait_on = Event::NO_EVENT) const;

      IndexSpaceAllocator create_allocator(void) const;

      const ElementMask &get_valid_mask(void) const;

      // new interface for dependent indexspace computation

      // There are three categories of operation:
      //  1) Index-based partitions create subspaces based only on properties of the
      //       index space itself.  Since we're working with unstructured index spaces here,
      //       the only thing you can really do is chop it into N pieces - we support both
      //       equal and weighted distributions.  A 'granularity' larger than 1 forces the
      //       split points to that granularity, in an attempt to avoid ragged boundaries.
      //
      //  2) Logical operations on index spaces to compute other index spaces.  The supported
      //       logical operations are UNION, INTERSECT, and SUBTRACT (no COMPLEMENT, as that
      //       requires having some notion of what the universe is), and can be performed on
      //       a bunch of pairs (as a convenience for the Legion runtime which might want to
      //       perform a logical operation on two partitions) or as a reduction operation that
      //       generates a single IndexSpace from many inputs.
      //
      //  3) Field-based partitions that use the contents of a field to perform a partitioning.
      //       One version is a 'group by' on the values of the field, creating a sub-IndexSpace
      //       for each value requested.  The other is to use the field as a 'foreign key' and
      //       calculate sub-IndexSpaces by either mapping forward (an 'image') or backwards
      //       (a 'preimage') a set of IndexSpaces through the field.
      //
      // All variations have a few things in common:
      //  a) All return immediately, with 'names' of the new IndexSpaces filled in.  Result
      //       values/fields/vectors need not be initialized before calling the method.  For the
      //       data-dependent operations, the maps need to be populated, but the value of each
      //       entry in the map is overwritten by the operation.
      //  b) All return an Event, which says when the contents of the new IndexSpaces will
      //       actually be valid.
      //  c) All accept a 'wait_on' parameter to defer the operation.  The contents of input
      //       IndexSpaces need not be valid until that time.
      //  d) All accept a 'mutable_results' parameter that must be set to _true_ if you want
      //       to be able to perform alloc/free's on the resulting IndexSpaces.  Setting this
      //       to _false_ will likely result in the use of more efficient (in both time and
      //       space) data structures to describe the IndexSpaces.

      // first, operations that chop up an IndexSpace into N pieces, with no attention paid
      //  to the the data
      Event create_equal_subspaces(size_t count, size_t granularity,
				   std::vector<IndexSpace>& subspaces,
				   bool mutable_results,
				   Event wait_on = Event::NO_EVENT) const;
      Event create_weighted_subspaces(size_t count, size_t granularity,
				      const std::vector<int>& weights,
				      std::vector<IndexSpace>& subspaces,
				      bool mutable_results,
				      Event wait_on = Event::NO_EVENT) const;

      // logical operations on IndexSpaces can be either maps (performing operations in
      //   parallel on many pairs of IndexSpaces) or reductions (many IndexSpaces -> one)
      enum IndexSpaceOperation {
        ISO_UNION,
        ISO_INTERSECT,
        ISO_SUBTRACT,
      };
      struct BinaryOpDescriptor;
      static Event compute_index_spaces(std::vector<BinaryOpDescriptor>& pairs,
					bool mutable_results,
					Event wait_on = Event::NO_EVENT);
      static Event reduce_index_spaces(IndexSpaceOperation op,
				       const std::vector<IndexSpace>& spaces,
				       IndexSpace& result,
				       bool mutable_results,
                                       IndexSpace parent = IndexSpace::NO_SPACE,
				       Event wait_on = Event::NO_EVENT);

      // operations that use field data need to be able to describe where that data is
      // there might be multiple instaces with data for different subsets of the index space,
      //  and each might have a different layout
      struct FieldDataDescriptor;
      Event create_subspaces_by_field(const std::vector<FieldDataDescriptor>& field_data,
				      std::map<DomainPoint, IndexSpace>& subspaces,
				      bool mutable_results,
				      Event wait_on = Event::NO_EVENT) const;
      Event create_subspaces_by_image(const std::vector<FieldDataDescriptor>& field_data,
				      std::map<IndexSpace, IndexSpace>& subspaces,
				      bool mutable_results,
				      Event wait_on = Event::NO_EVENT) const;
      Event create_subspaces_by_preimage(const std::vector<FieldDataDescriptor>& field_data,
					 std::map<IndexSpace, IndexSpace>& subspaces,
					 bool mutable_results,
					 Event wait_on = Event::NO_EVENT) const;
      // Variants of the above but with profiling information
      Event create_equal_subspaces(size_t count, size_t granularity,
				   std::vector<IndexSpace>& subspaces,
                                   const ProfilingRequestSet &reqs,
				   bool mutable_results,
				   Event wait_on = Event::NO_EVENT) const;
      Event create_weighted_subspaces(size_t count, size_t granularity,
				      const std::vector<int>& weights,
				      std::vector<IndexSpace>& subspaces,
                                      const ProfilingRequestSet &reqs,
				      bool mutable_results,
				      Event wait_on = Event::NO_EVENT) const;
      static Event compute_index_spaces(std::vector<BinaryOpDescriptor>& pairs,
                                        const ProfilingRequestSet &reqs,
					bool mutable_results,
					Event wait_on = Event::NO_EVENT);
      static Event reduce_index_spaces(IndexSpaceOperation op,
				       const std::vector<IndexSpace>& spaces,
                                       const ProfilingRequestSet &reqs,
				       IndexSpace& result,
				       bool mutable_results,
                                       IndexSpace parent = IndexSpace::NO_SPACE,
				       Event wait_on = Event::NO_EVENT);
      Event create_subspaces_by_field(const std::vector<FieldDataDescriptor>& field_data,
				      std::map<DomainPoint, IndexSpace>& subspaces,
                                      const ProfilingRequestSet &reqs,
				      bool mutable_results,
				      Event wait_on = Event::NO_EVENT) const;
      Event create_subspaces_by_image(const std::vector<FieldDataDescriptor>& field_data,
				      std::map<IndexSpace, IndexSpace>& subspaces,
                                      const ProfilingRequestSet &reqs,
				      bool mutable_results,
				      Event wait_on = Event::NO_EVENT) const;
      Event create_subspaces_by_preimage(const std::vector<FieldDataDescriptor>& field_data,
					 std::map<IndexSpace, IndexSpace>& subspaces,
                                         const ProfilingRequestSet &reqs,
					 bool mutable_results,
					 Event wait_on = Event::NO_EVENT) const;
    };
    struct IndexSpace::BinaryOpDescriptor {
      IndexSpaceOperation op;
      IndexSpace parent;                       // filled in by caller
      IndexSpace left_operand, right_operand;  // filled in by caller
      IndexSpace result;                       // filled in by operation
    };
    struct IndexSpace::FieldDataDescriptor {
      IndexSpace index_space;
      RegionInstance inst;
      size_t field_offset, field_size;
    };

    inline std::ostream& operator<<(std::ostream& os, IndexSpace i) { return os << std::hex << i.id << std::dec; }

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
      static DomainPoint from_point(typename LegionRuntime::Arrays::Point<DIM> p)
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
      LegionRuntime::Arrays::Point<DIM> get_point(void) const { assert(dim == DIM); return LegionRuntime::Arrays::Point<DIM>(point_data); }

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

    class DomainLinearization {
    public:
      DomainLinearization(void) : dim(-1), lptr(0) {}
      DomainLinearization(const DomainLinearization& other)
        : dim(other.dim), lptr(other.lptr)
      {
        add_local_reference();
      }
      ~DomainLinearization(void)
      {
        remove_local_reference();
      }

      void add_local_reference(void)
      {
        if (lptr != NULL)
        {
          switch(dim) {
          case 1: ((LegionRuntime::Arrays::Mapping<1, 1> *)lptr)->add_reference(); break;
          case 2: ((LegionRuntime::Arrays::Mapping<2, 1> *)lptr)->add_reference(); break;
          case 3: ((LegionRuntime::Arrays::Mapping<3, 1> *)lptr)->add_reference(); break;
          default: assert(0);
          }
        }
      }

      void remove_local_reference(void)
      {
        if (lptr != NULL)
        {
          switch(dim)
          {
            case 1:
              {
                LegionRuntime::Arrays::Mapping<1, 1> *mapping = (LegionRuntime::Arrays::Mapping<1, 1>*)lptr;
                if (mapping->remove_reference())
                  delete mapping;
                break;
              }
            case 2:
              {
                LegionRuntime::Arrays::Mapping<2, 1> *mapping = (LegionRuntime::Arrays::Mapping<2, 1>*)lptr;
                if (mapping->remove_reference())
                  delete mapping;
                break;
              }
            case 3:
              {
                LegionRuntime::Arrays::Mapping<3, 1> *mapping = (LegionRuntime::Arrays::Mapping<3, 1>*)lptr;
                if (mapping->remove_reference())
                  delete mapping;
                break;
              }
            default:
              assert(0);
          }
        }
      }

      bool valid(void) const { return(dim >= 0); }

      DomainLinearization& operator=(const DomainLinearization& other)
      {
        remove_local_reference();
	dim = other.dim;
	lptr = other.lptr;
        add_local_reference();
	return *this;
      }

      static DomainLinearization from_index_space(int first_elmt)
      {
        DomainLinearization l;
        l.dim = 0;
        return l;
      }

      template <int DIM>
      static DomainLinearization from_mapping(typename LegionRuntime::Arrays::Mapping<DIM, 1> *mapping)
      {
	DomainLinearization l;
	l.dim = DIM;
	l.lptr = (void *)mapping;
        l.add_local_reference();
	return l;
      }

      void serialize(int *data) const
      {
	data[0] = dim;
	switch(dim) {
	case 0: break; // nothing to serialize
	case 1: ((LegionRuntime::Arrays::Mapping<1, 1> *)lptr)->serialize_mapping(data + 1); break;
	case 2: ((LegionRuntime::Arrays::Mapping<2, 1> *)lptr)->serialize_mapping(data + 1); break;
	case 3: ((LegionRuntime::Arrays::Mapping<3, 1> *)lptr)->serialize_mapping(data + 1); break;
	default: assert(0);
	}
      }

      void deserialize(const int *data)
      {
        remove_local_reference();
	dim = data[0];
	switch(dim) {
	case 0: break; // nothing to serialize
	case 1: lptr = (void *)(LegionRuntime::Arrays::Mapping<1, 1>::deserialize_mapping(data + 1)); break;
	case 2: lptr = (void *)(LegionRuntime::Arrays::Mapping<2, 1>::deserialize_mapping(data + 1)); break;
	case 3: lptr = (void *)(LegionRuntime::Arrays::Mapping<3, 1>::deserialize_mapping(data + 1)); break;
	default: assert(0);
	}
        add_local_reference();
      }

      int get_dim(void) const { return dim; }

      template <int DIM>
      LegionRuntime::Arrays::Mapping<DIM, 1> *get_mapping(void) const
      {
	assert(DIM == dim);
	return (LegionRuntime::Arrays::Mapping<DIM, 1> *)lptr;
      }

      int get_image(const DomainPoint& p) const
      {
	assert(dim == p.dim);
	switch(dim) {
	case 0: // index space
	  {
	    // assume no offset for now - probably wrong
	    return p.get_index();
	  }

	case 1:
	  {
	    //printf("%d -> %d\n", p.get_point<1>()[0], get_mapping<1>()->image(p.get_point<1>())[0]);
	    return get_mapping<1>()->image(p.get_point<1>());
	  }

	case 2:
	  {
	    //printf("%d -> %d\n", p.get_point<2>()[0], get_mapping<2>()->image(p.get_point<2>())[0]);
	    return get_mapping<2>()->image(p.get_point<2>());
	  }

	case 3:
	  {
	    //printf("%d -> %d\n", p.get_point<3>()[0], get_mapping<3>()->image(p.get_point<3>())[0]);
	    return get_mapping<3>()->image(p.get_point<3>());
	  }

	default:
	  assert(0);
	}
        return 0;
      }

      protected:
	int dim;
	void *lptr;
    };

    class Domain {
    public:
      typedef ::legion_lowlevel_id_t IDType;

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
            return Domain::from_rect<1>(get_rect<1>().intersection(other.get_rect<1>()));
          case 2:
            return Domain::from_rect<2>(get_rect<2>().intersection(other.get_rect<2>()));
          case 3:
            return Domain::from_rect<3>(get_rect<3>().intersection(other.get_rect<3>()));
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
      LegionRuntime::Arrays::Rect<DIM> get_rect(void) const { assert(dim == DIM); return LegionRuntime::Arrays::Rect<DIM>(rect_data); }

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
	      LegionRuntime::Arrays::GenericPointInRectIterator<1> *pir = new LegionRuntime::Arrays::GenericPointInRectIterator<1>(d.get_rect<1>());
	      iterator = (void *)pir;
	      pir->p.to_array(p.point_data);
	      any_left = pir->any_left;
	    }
	    break;

	  case 2:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<2> *pir = new LegionRuntime::Arrays::GenericPointInRectIterator<2>(d.get_rect<2>());
	      iterator = (void *)pir;
	      pir->p.to_array(p.point_data);
	      any_left = pir->any_left;
	    }
	    break;

	  case 3:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<3> *pir = new LegionRuntime::Arrays::GenericPointInRectIterator<3>(d.get_rect<3>());
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
	      LegionRuntime::Arrays::GenericPointInRectIterator<1> *pir = (LegionRuntime::Arrays::GenericPointInRectIterator<1> *)iterator;
	      delete pir;
	    }
	    break;

	  case 2:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<2> *pir = (LegionRuntime::Arrays::GenericPointInRectIterator<2> *)iterator;
	      delete pir;
	    }
	    break;

	  case 3:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<3> *pir = (LegionRuntime::Arrays::GenericPointInRectIterator<3> *)iterator;
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
	      LegionRuntime::Arrays::GenericPointInRectIterator<1> *pir = (LegionRuntime::Arrays::GenericPointInRectIterator<1> *)iterator;
	      any_left = pir->step();
	      pir->p.to_array(p.point_data);
	    }
	    break;

	  case 2:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<2> *pir = (LegionRuntime::Arrays::GenericPointInRectIterator<2> *)iterator;
	      any_left = pir->step();
	      pir->p.to_array(p.point_data);
	    }
	    break;

	  case 3:
	    {
	      LegionRuntime::Arrays::GenericPointInRectIterator<3> *pir = (LegionRuntime::Arrays::GenericPointInRectIterator<3> *)iterator;
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
	DomainPointIterator& operator++(int /*i am postfix*/) { step(); return *this; }
      };

    protected:
    public:
      IDType is_id;
      int dim;
      coord_t rect_data[2 * MAX_RECT_DIM];

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
      struct CopySrcDstField {
      public:
        CopySrcDstField(void) 
          : inst(RegionInstance::NO_INST), offset(0), size(0), 
            field_id(0), serdez_id(0) { }
        CopySrcDstField(RegionInstance i, coord_t o, size_t s)
          : inst(i), offset(o), size(s), field_id(0), serdez_id(0) { }
        CopySrcDstField(RegionInstance i, coord_t o, size_t s, unsigned f)
          : inst(i), offset(o), size(s), field_id(f), serdez_id(0) { }
        CopySrcDstField(RegionInstance i, coord_t o, size_t s, 
                        unsigned f, CustomSerdezID sid)
          : inst(i), offset(o), size(s), field_id(f), serdez_id(sid) { }
      public:
	RegionInstance inst;
	coord_t offset;
        size_t size;
        unsigned field_id;
	CustomSerdezID serdez_id;
      };

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

    class IndexSpaceAllocator {
    protected:
      friend class IndexSpace;

      explicit IndexSpaceAllocator(void *_impl) : impl(_impl) {}

      void *impl;

    public:
      IndexSpaceAllocator(const IndexSpaceAllocator& to_copy)
	: impl(to_copy.impl) {}

      coord_t alloc(size_t count = 1) const;
      void reserve(coord_t ptr, size_t count = 1) const;
      void free(coord_t ptr, size_t count = 1) const;

      template <typename LIN>
      void reserve(const LIN& linearizer, LegionRuntime::Arrays::Point<LIN::IDIM> point) const;

      void destroy(void);
    };

    // Implementations for template functions

    template <class T>
    /*static*/ size_t ElementMask::forall_ranges(T &executor,
                                                 const ElementMask &mask,
					         coord_t start /*= 0*/,
					         coord_t count /*= -1*/,
					         bool do_enabled /*= true*/)
    {
      if(count == 0) return 0;

      ElementMask::Enumerator enum1(mask, start, do_enabled ? 1 : 0);

      size_t total = 0;

      coord_t pos;
      size_t len;
      while(enum1.get_next(pos, len)) {
        coord_t len_ = len;
	if(pos < start) {
	  len_ -= (start - pos);
	  pos = start;
	}

	if((count > 0) && ((pos + len_) > (start + count))) {
	  len_ = start + count - pos;
	}

	if(len_ > 0) {
	  //printf("S:%d(%d)\n", pos, len);
	  executor.do_span(pos, len_);
	  total += len_;
	}
      }

      return total;
    }

    template <class T>
    /*static*/ size_t ElementMask::forall_ranges(T &executor,
                                                 const ElementMask &mask1,
					         const ElementMask &mask2,
					         coord_t start /*= 0*/,
					         coord_t count /*= -1*/,
					         bool do_enabled1 /*= true*/,
					         bool do_enabled2 /*= true*/)
    {
      ElementMask::Enumerator enum1(mask1, start, do_enabled1 ? 1 : 0);
      ElementMask::Enumerator enum2(mask2, start, do_enabled2 ? 1 : 0);

      coord_t pos1, pos2;
      size_t len1, len2;

      if(!enum1.get_next(pos1, len1)) return 0;
      if(!enum2.get_next(pos2, len2)) return 0;
      if(count == 0) return 0;

      size_t total = 0;

      coord_t len1_ = len1, len2_ = len1;
      while(true) {
	//printf("S:%d(%d) T:%d(%d)\n", pos1, len1, pos2, len2);

	if(len1_ <= 0) {
	  if(!enum1.get_next(pos1, len1)) break;
          len1_ = len1;
	  if((count > 0) && ((pos1 + len1_) > (start + count))) {
	    len1_ = (start + count) - pos1;
	    if(len1_ < 0) break;
	  }
	  continue;
	}

	if(len2_ <= 0) {
	  if(!enum2.get_next(pos2, len2)) break;
          len2_ = len2;
	  if((count > 0) && ((pos2 + len2_) > (start + count))) {
	    len2_ = (start + count) - pos2;
	    if(len2_ < 0) break;
	  }
	  continue;
	}

	if(pos1 < pos2) {
	  len1_ -= (pos2 - pos1);
	  pos1 = pos2;
	  continue;
	}

	if(pos2 < pos1) {
	  len2_ -= (pos1 - pos2);
	  pos2 = pos1;
	  continue;
	}

	assert((pos1 == pos2) && (len1_ > 0) && (len2_ > 0));

	size_t span_len = (len1_ < len2_) ? len1_ : len2_;

	executor.do_span(pos1, span_len);

	pos1 += span_len;
	len1_ -= span_len;
	pos2 += span_len;
	len2_ -= span_len;

	total += span_len;
      }

      return total;
    }

}; // namespace Realm

//include "indexspace.inl"

#endif // ifndef REALM_INDEXSPACE_H
