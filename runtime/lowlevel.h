/* Copyright 2014 Stanford University
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

#ifndef RUNTIME_LOWLEVEL_H
#define RUNTIME_LOWLEVEL_H

#include <string>
#include <vector>
#include <set>
#include <map>
#include <cstdarg>
#include <stdint.h>

#include "common.h"
#include "utilities.h"
#include "accessor.h"
#include "arrays.h"

#ifndef __GNUC__
#include "atomics.h" // for __sync_fetch_and_add
#endif

namespace LegionRuntime {
  namespace LowLevel {
    // forward class declarations because these things all refer to each other
    class Event;
    class UserEvent;
    class Reservation;
    class Memory;
    class Processor;

    class IndexSpace;
    class IndexSpaceAllocator;
    class RegionInstance;

    class Machine;

#ifdef LEGION_IDS_ARE_64BIT
    typedef unsigned long long IDType;
#define IDFMT "%llx"
#else
    typedef unsigned IDType;
#define IDFMT "%x"
#endif

    typedef unsigned int AddressSpace;

    class Event {
    public:
      typedef IDType id_t;
      typedef unsigned gen_t;

      id_t id;
      gen_t gen;
      bool operator<(const Event& rhs) const 
      { 
        if (id < rhs.id)
          return true;
        else if (id > rhs.id)
          return false;
        else
          return (gen < rhs.gen);
      }
      bool operator==(const Event& rhs) const { return (id == rhs.id) && (gen == rhs.gen); }
      bool operator!=(const Event& rhs) const { return (id != rhs.id) || (gen != rhs.gen); }

      class Impl;
      Impl *impl(void) const;

      static const Event NO_EVENT;

      bool exists(void) const { return id != 0; }

      // test whether an event has triggered without waiting
      bool has_triggered(void) const;

      // causes calling thread to block until event has occurred
      void wait(bool block = false) const;

      // used by non-legion threads to wait on an event - always blocking
      void external_wait(void) const;

      // creates an event that won't trigger until all input events have
      static Event merge_events(const std::set<Event>& wait_for);
      static Event merge_events(Event ev1, Event ev2,
				Event ev3 = NO_EVENT, Event ev4 = NO_EVENT,
				Event ev5 = NO_EVENT, Event ev6 = NO_EVENT);
    };

    // A user level event has all the properties of event, except
    // it can be triggered by the user.  This prevents users from
    // triggering arbitrary events without doing something like
    // an unsafe cast.
    class UserEvent : public Event {
    public:
      static UserEvent create_user_event(void);
      void trigger(Event wait_on = Event::NO_EVENT) const;

      static const UserEvent NO_USER_EVENT;
    };

    // a Barrier is similar to a UserEvent, except that it has a count of how
    //  many threads (or whatever) need to "trigger" before the actual trigger
    //  occurs
    class Barrier : public Event {
    public:
      typedef unsigned long long timestamp_t; // used to avoid race conditions with arrival adjustments

      timestamp_t timestamp;

      static Barrier create_barrier(unsigned expected_arrivals);
      void destroy_barrier(void);

      Barrier advance_barrier(void) const;
      Barrier alter_arrival_count(int delta) const;
      Event get_previous_phase(void) const;

      void arrive(unsigned count = 1, Event wait_on = Event::NO_EVENT) const;
    };

    class Reservation {
    public:
      typedef IDType id_t;
      id_t id;
      bool operator<(const Reservation& rhs) const { return id < rhs.id; }
      bool operator==(const Reservation& rhs) const { return id == rhs.id; }
      bool operator!=(const Reservation& rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const Reservation NO_RESERVATION;

      bool exists(void) const { return id != 0; }

      // requests ownership (either exclusive or shared) of the reservation with a 
      //   specified mode - returns an event that will trigger when the reservation 
      //   is granted
      Event acquire(unsigned mode = 0, bool exclusive = true, Event wait_on = Event::NO_EVENT) const;
      // releases a held reservation - release can be deferred until an event triggers
      void release(Event wait_on = Event::NO_EVENT) const;

      // Create a new reservation, destroy an existing reservation 
      static Reservation create_reservation(size_t _data_size = 0);
      void destroy_reservation();

      size_t data_size(void) const;
      void *data_ptr(void) const;
    };

    class Processor {
    public:
      typedef IDType id_t;
      id_t id;
      bool operator<(const Processor& rhs) const { return id < rhs.id; }
      bool operator==(const Processor& rhs) const { return id == rhs.id; }
      bool operator!=(const Processor& rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const Processor NO_PROC;

      bool exists(void) const { return id != 0; }

      Processor get_utility_processor(void) const;

      typedef unsigned TaskFuncID;
      typedef void (*TaskFuncPtr)(const void *args, size_t arglen, Processor proc);
      typedef std::map<TaskFuncID, TaskFuncPtr> TaskIDTable;

      // Different Processor types
      enum Kind {
	TOC_PROC, // Throughput core
	LOC_PROC, // Latency core
	UTIL_PROC, // Utility core
      };

      void enable_idle_task(void);
      void disable_idle_task(void);

      // Return the address space for this processor
      AddressSpace address_space(void) const;
      // Return the local ID within the address space
      IDType local_id(void) const;

      // special task IDs
      enum {
        // Save ID 0 for the force shutdown function
	TASK_ID_REQUEST_SHUTDOWN   = 0,
	TASK_ID_PROCESSOR_INIT     = 1, // only called by utility processors
	TASK_ID_PROCESSOR_SHUTDOWN = 2, // only called by utility processors
	TASK_ID_PROCESSOR_IDLE     = 3, // typically used for high-level scheduler, only called by utility processors
	TASK_ID_FIRST_AVAILABLE    = 4,
      };

      Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
		  Event wait_on = Event::NO_EVENT, int priority = 0) const;
    };

    class Memory {
    public:
      typedef IDType id_t;
      id_t id;
      bool operator<(const Memory &rhs) const { return id < rhs.id; }
      bool operator==(const Memory &rhs) const { return id == rhs.id; }
      bool operator!=(const Memory &rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const Memory NO_MEMORY;

      bool exists(void) const { return id != 0; }

      // Return the address space for this memory
      AddressSpace address_space(void) const;
      // Return the local ID within the address space
      IDType local_id(void) const;

      // Different Memory types
      enum Kind {
        GLOBAL_MEM, // Guaranteed visible to all processors on all nodes (e.g. GASNet memory, universally slow)
        SYSTEM_MEM, // Visible to all processors on a node
        REGDMA_MEM, // Registered memory visible to all processors on a node, can be a target of RDMA
        SOCKET_MEM, // Memory visible to all processors within a node, better performance to processors on same socket 
        Z_COPY_MEM, // Zero-Copy memory visible to all CPUs within a node and one or more GPUs 
        GPU_FB_MEM,   // Framebuffer memory for one GPU and all its SMs
        LEVEL3_CACHE, // CPU L3 Visible to all processors on the node, better performance to processors on same socket 
        LEVEL2_CACHE, // CPU L2 Visible to all processors on the node, better performance to one processor
        LEVEL1_CACHE, // CPU L1 Visible to all processors on the node, better performance to one processor
      };
    };

    class ElementMask {
    public:
      ElementMask(void);
      explicit ElementMask(int num_elements, int first_element = 0);
      ElementMask(const ElementMask &copy_from, int num_elements = -1, int first_element = 0);
      ~ElementMask(void);

      void init(int _first_element, int _num_elements, Memory _memory, off_t _offset);

      int get_num_elmts(void) const { return num_elements; }

      void enable(int start, int count = 1);
      void disable(int start, int count = 1);

      int find_enabled(int count = 1, int start = 0) const;
      int find_disabled(int count = 1, int start = 0) const;
      
      bool is_set(int ptr) const;
      size_t pop_count(bool enabled = true) const;
      bool operator!(void) const;
      // union/intersect/subtract?
      ElementMask operator|(const ElementMask &other) const;
      ElementMask operator&(const ElementMask &other) const;
      ElementMask operator-(const ElementMask &other) const;

      ElementMask& operator|=(const ElementMask &other);
      ElementMask& operator&=(const ElementMask &other);
      ElementMask& operator-=(const ElementMask &other);

      int first_enabled(void) const { return first_enabled_elmt; }
      int last_enabled(void) const { return last_enabled_elmt; }

      ElementMask& operator=(const ElementMask &rhs);

      enum OverlapResult { OVERLAP_NO, OVERLAP_MAYBE, OVERLAP_YES };

      OverlapResult overlaps_with(const ElementMask& other,
				  off_t max_effort = -1) const;

      ElementMask intersect_with(const ElementMask &other);

      class Enumerator {
      public:
	Enumerator(const ElementMask& _mask, int _start, int _polarity);
	~Enumerator(void);

	bool get_next(int &position, int &length);
	bool peek_next(int &position, int &length);

      protected:
	const ElementMask& mask;
	int pos;
	int polarity;
      };

      Enumerator *enumerate_enabled(int start = 0) const;
      Enumerator *enumerate_disabled(int start = 0) const;

      size_t raw_size(void) const;
      const void *get_raw(void) const;
      void set_raw(const void *data);

      // Implementations below
      template <class T>
      static int forall_ranges(T &executor,
			       const ElementMask &mask,
			       int start = 0, int count = -1,
			       bool do_enabled = true);

      template <class T>
      static int forall_ranges(T &executor,
			       const ElementMask &mask1, 
			       const ElementMask &mask2,
			       int start = 0, int count = -1,
			       bool do_enabled1 = true,
			       bool do_enabled2 = true);

    public:
      friend class Enumerator;
      int first_element;
      int num_elements;
      Memory memory;
      off_t offset;
      void *raw_data;
      int first_enabled_elmt, last_enabled_elmt;
    };

    // a reduction op needs to look like this
#ifdef NOT_REALLY_CODE
    class MyReductionOp {
    public:
      typedef int LHS;
      typedef int RHS;

      static void apply(LHS& lhs, RHS rhs);

      // both of these are optional
      static const RHS identity;
      static void fold(RHS& rhs1, RHS rhs2);
    };
#endif

    typedef int ReductionOpID;
    class ReductionOpUntyped {
    public:
      size_t sizeof_lhs;
      size_t sizeof_rhs;
      size_t sizeof_list_entry;
      bool has_identity;
      bool is_foldable;

      template <class REDOP>
	static ReductionOpUntyped *create_reduction_op(void);

      virtual void apply(void *lhs_ptr, const void *rhs_ptr, size_t count,
			 bool exclusive = false) const = 0;
      virtual void apply_strided(void *lhs_ptr, const void *rhs_ptr,
				 off_t lhs_stride, off_t rhs_stride, size_t count,
				 bool exclusive = false) const = 0;
      virtual void fold(void *rhs1_ptr, const void *rhs2_ptr, size_t count,
			bool exclusive = false) const = 0;
      virtual void fold_strided(void *lhs_ptr, const void *rhs_ptr,
				off_t lhs_stride, off_t rhs_stride, size_t count,
				bool exclusive = false) const = 0;
      virtual void init(void *rhs_ptr, size_t count) const = 0;

      virtual void apply_list_entry(void *lhs_ptr, const void *entry_ptr, size_t count,
				    off_t ptr_offset, bool exclusive = false) const = 0;
      virtual void fold_list_entry(void *rhs_ptr, const void *entry_ptr, size_t count,
                                    off_t ptr_offset, bool exclusive = false) const = 0;
      virtual void get_list_pointers(unsigned *ptrs, const void *entry_ptr, size_t count) const = 0;

    protected:
      ReductionOpUntyped(size_t _sizeof_lhs, size_t _sizeof_rhs,
			 size_t _sizeof_list_entry,
			 bool _has_identity, bool _is_foldable)
	: sizeof_lhs(_sizeof_lhs), sizeof_rhs(_sizeof_rhs),
	  sizeof_list_entry(_sizeof_list_entry),
  	  has_identity(_has_identity), is_foldable(_is_foldable) {}
    };
    typedef std::map<ReductionOpID, const ReductionOpUntyped *> ReductionOpTable;

    template <class LHS, class RHS>
    struct ReductionListEntry {
      ptr_t ptr;
      RHS rhs;
    };

    template <class REDOP>
    class ReductionOp : public ReductionOpUntyped {
    public:
      // TODO: don't assume identity and fold are available - use scary
      //  template-fu to figure it out
      ReductionOp(void)
	: ReductionOpUntyped(sizeof(typename REDOP::LHS), sizeof(typename REDOP::RHS),
			     sizeof(ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS>),
			     true, true) {}

      virtual void apply(void *lhs_ptr, const void *rhs_ptr, size_t count,
			 bool exclusive = false) const
      {
	typename REDOP::LHS *lhs = (typename REDOP::LHS *)lhs_ptr;
	const typename REDOP::RHS *rhs = (const typename REDOP::RHS *)rhs_ptr;
	if(exclusive) {
	  for(size_t i = 0; i < count; i++)
	    REDOP::template apply<true>(lhs[i], rhs[i]);
	} else {
	  for(size_t i = 0; i < count; i++)
	    REDOP::template apply<false>(lhs[i], rhs[i]);
	}
      }

      virtual void apply_strided(void *lhs_ptr, const void *rhs_ptr,
				 off_t lhs_stride, off_t rhs_stride, size_t count,
				 bool exclusive = false) const
      {
	char *lhs = (char *)lhs_ptr;
	const char *rhs = (const char *)rhs_ptr;
	if(exclusive) {
	  for(size_t i = 0; i < count; i++) {
	    REDOP::template apply<true>(*(typename REDOP::LHS *)lhs,
					*(const typename REDOP::RHS *)rhs);
	    lhs += lhs_stride;
	    rhs += rhs_stride;
	  }
	} else {
	  for(size_t i = 0; i < count; i++) {
	    REDOP::template apply<false>(*(typename REDOP::LHS *)lhs,
					 *(const typename REDOP::RHS *)rhs);
	    lhs += lhs_stride;
	    rhs += rhs_stride;
	  }
	}
      }

      virtual void fold(void *rhs1_ptr, const void *rhs2_ptr, size_t count,
			bool exclusive = false) const
      {
	typename REDOP::RHS *rhs1 = (typename REDOP::RHS *)rhs1_ptr;
	const typename REDOP::RHS *rhs2 = (const typename REDOP::RHS *)rhs2_ptr;
	if(exclusive) {
	  for(size_t i = 0; i < count; i++)
	    REDOP::template fold<true>(rhs1[i], rhs2[i]);
	} else {
	  for(size_t i = 0; i < count; i++)
	    REDOP::template fold<false>(rhs1[i], rhs2[i]);
	}
      }

      virtual void fold_strided(void *lhs_ptr, const void *rhs_ptr,
				off_t lhs_stride, off_t rhs_stride, size_t count,
				bool exclusive = false) const
      {
	char *lhs = (char *)lhs_ptr;
	const char *rhs = (const char *)rhs_ptr;
	if(exclusive) {
	  for(size_t i = 0; i < count; i++) {
	    REDOP::template fold<true>(*(typename REDOP::RHS *)lhs,
				       *(const typename REDOP::RHS *)rhs);
	    lhs += lhs_stride;
	    rhs += rhs_stride;
	  }
	} else {
	  for(size_t i = 0; i < count; i++) {
	    REDOP::template fold<false>(*(typename REDOP::RHS *)lhs,
					*(const typename REDOP::RHS *)rhs);
	    lhs += lhs_stride;
	    rhs += rhs_stride;
	  }
	}
      }

      virtual void init(void *ptr, size_t count) const
      {
        typename REDOP::RHS *rhs_ptr = (typename REDOP::RHS *)ptr;
        for (size_t i = 0; i < count; i++)
          memcpy(rhs_ptr++, &(REDOP::identity), sizeof_rhs);
      }

      virtual void apply_list_entry(void *lhs_ptr, const void *entry_ptr, size_t count,
				    off_t ptr_offset, bool exclusive = false) const
      {
	typename REDOP::LHS *lhs = (typename REDOP::LHS *)lhs_ptr;
	const ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS> *entry = (const ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS> *)entry_ptr;
	if(exclusive) {
	  for(size_t i = 0; i < count; i++)
	    REDOP::template apply<true>(lhs[entry[i].ptr.value - ptr_offset], entry[i].rhs);
	} else {
	  for(size_t i = 0; i < count; i++)
	    REDOP::template apply<false>(lhs[entry[i].ptr.value - ptr_offset], entry[i].rhs);
	}
      }

      virtual void fold_list_entry(void *rhs_ptr, const void *entry_ptr, size_t count,
                                    off_t ptr_offset, bool exclusive = false) const
      {
        typename REDOP::RHS *rhs = (typename REDOP::RHS*)rhs_ptr;
        const ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS> *entry = (const ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS> *)entry_ptr;
        if (exclusive)
        {
          for (size_t i = 0; i < count; i++)
            REDOP::template fold<true>(rhs[entry[i].ptr.value - ptr_offset], entry[i].rhs);
        }
        else
        {
          for (size_t i = 0; i < count; i++)
            REDOP::template fold<false>(rhs[entry[i].ptr.value - ptr_offset], entry[i].rhs);
        }
      }

      virtual void get_list_pointers(unsigned *ptrs, const void *entry_ptr, size_t count) const
      {
	const ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS> *entry = (const ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS> *)entry_ptr;
	for(size_t i = 0; i < count; i++) {
	  ptrs[i] = entry[i].ptr.value;
	  //printf("%d=%d\n", i, ptrs[i]);
	}
      }
    };

    template <class REDOP>
    ReductionOpUntyped *ReductionOpUntyped::create_reduction_op(void)
    {
      ReductionOp<REDOP> *redop = new ReductionOp<REDOP>();
      return redop;
    }

    class RegionInstance {
    public:
      typedef IDType id_t;
      id_t id;
      bool operator<(const RegionInstance &rhs) const { return id < rhs.id; }
      bool operator==(const RegionInstance &rhs) const { return id == rhs.id; }
      bool operator!=(const RegionInstance &rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const RegionInstance NO_INST;

      bool exists(void) const { return id != 0; }

      void destroy(Event wait_on = Event::NO_EVENT) const;

      AddressSpace address_space(void) const;
      IDType local_id(void) const;

      LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::Generic> get_accessor(void) const;
    };

    class IndexSpace {
    public:
      typedef IDType id_t;
      id_t id;
      bool operator<(const IndexSpace &rhs) const { return id < rhs.id; }
      bool operator==(const IndexSpace &rhs) const { return id == rhs.id; }
      bool operator!=(const IndexSpace &rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const IndexSpace NO_SPACE;

      bool exists(void) const { return id != 0; } 

      static IndexSpace create_index_space(size_t num_elmts);
      static IndexSpace create_index_space(const ElementMask &mask);
      static IndexSpace create_index_space(IndexSpace parent,
					   const ElementMask &mask);

      static IndexSpace expand_index_space(IndexSpace child,
					   size_t num_elmts,
					   off_t child_offset = 0);

      void destroy(void) const;

      IndexSpaceAllocator create_allocator(void) const;

      const ElementMask &get_valid_mask(void) const;
    };

    class DomainPoint {
    public:
      enum { MAX_POINT_DIM = 3 };

      DomainPoint(void) : dim(0) { point_data[0] = 0; }
      DomainPoint(int index) : dim(0) { point_data[0] = index; }

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
      static DomainPoint from_point(typename Arrays::Point<DIM> p)
      {
	DomainPoint dp;
	assert(DIM <= MAX_POINT_DIM); 
	dp.dim = DIM;
	p.to_array(dp.point_data);
	return dp;
      }

      int get_index(void) const
      {
	assert(dim == 0);
	return point_data[0];
      }

      int get_dim(void) const { return dim; }

      template <int DIM>
      Arrays::Point<DIM> get_point(void) const { assert(dim == DIM); return Arrays::Point<DIM>(point_data); }

      bool is_null(void) const { return (dim > -1); }

      static DomainPoint nil(void) { DomainPoint p; p.dim = -1; return p; }

    protected:
    public:
      int dim;
      int point_data[MAX_POINT_DIM];
    };

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
          case 1: ((Arrays::Mapping<1, 1> *)lptr)->add_reference(); break;
          case 2: ((Arrays::Mapping<2, 1> *)lptr)->add_reference(); break;
          case 3: ((Arrays::Mapping<3, 1> *)lptr)->add_reference(); break;
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
                Arrays::Mapping<1, 1> *mapping = (Arrays::Mapping<1, 1>*)lptr;
                if (mapping->remove_reference())
                  delete mapping;
                break;
              }
            case 2:
              {
                Arrays::Mapping<2, 1> *mapping = (Arrays::Mapping<2, 1>*)lptr;
                if (mapping->remove_reference())
                  delete mapping;
                break;
              }
            case 3:
              {
                Arrays::Mapping<3, 1> *mapping = (Arrays::Mapping<3, 1>*)lptr;
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
      static DomainLinearization from_mapping(typename Arrays::Mapping<DIM, 1> *mapping)
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
	case 1: ((Arrays::Mapping<1, 1> *)lptr)->serialize_mapping(data + 1); break;
	case 2: ((Arrays::Mapping<2, 1> *)lptr)->serialize_mapping(data + 1); break;
	case 3: ((Arrays::Mapping<3, 1> *)lptr)->serialize_mapping(data + 1); break;
	default: assert(0);
	}
      }

      void deserialize(const int *data)
      {
        remove_local_reference();
	dim = data[0];
	switch(dim) {
	case 0: break; // nothing to serialize
	case 1: lptr = (void *)(Arrays::Mapping<1, 1>::deserialize_mapping(data + 1)); break;
	case 2: lptr = (void *)(Arrays::Mapping<2, 1>::deserialize_mapping(data + 1)); break;
	case 3: lptr = (void *)(Arrays::Mapping<3, 1>::deserialize_mapping(data + 1)); break;
	default: assert(0);
	}
        add_local_reference();
      }

      int get_dim(void) const { return dim; }

      template <int DIM>
      Arrays::Mapping<DIM, 1> *get_mapping(void) const
      {
	assert(DIM == dim);
	return (Arrays::Mapping<DIM, 1> *)lptr;
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
      enum { MAX_RECT_DIM = 3 };
      Domain(void) : is_id(0), dim(0) {}
      Domain(LegionRuntime::LowLevel::IndexSpace is) : is_id(is.id), dim(0) {}
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
      static Domain from_rect(typename Arrays::Rect<DIM> r)
      {
	Domain d;
	assert(DIM <= MAX_RECT_DIM); 
	d.dim = DIM;
	r.to_array(d.rect_data);
	return d;
      }

      template<int DIM>
      static Domain from_point(typename Arrays::Point<DIM> p)
      {
        Domain d;
        assert(DIM <= MAX_RECT_DIM);
        d.dim = DIM;
        p.to_array(d.rect_data);
        p.to_array(d.rect_data+DIM);
        return d;
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

      LegionRuntime::LowLevel::IndexSpace get_index_space(void) const
      {
	assert(is_id);
	IndexSpace is = { static_cast<IDType>(is_id) };
	return is;
      }

      LegionRuntime::LowLevel::IndexSpace get_index_space(bool create_if_needed = false)
      {
	IndexSpace is;
	if(!is_id) {
	  assert(create_if_needed);
	  is_id = IndexSpace::create_index_space(1).id;
	}
	is.id = is_id;
	return is;
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
              Arrays::Point<1> p1 = point.get_point<1>();
              Arrays::Rect<1> r1 = get_rect<1>();
              result = r1.contains(p1);
              break;
            }
          case 2:
            {
              Arrays::Point<2> p2 = point.get_point<2>();
              Arrays::Rect<2> r2 = get_rect<2>();
              result = r2.contains(p2);
              break;
            }
          case 3:
            {
              Arrays::Point<3> p3 = point.get_point<3>();
              Arrays::Rect<3> r3 = get_rect<3>();
              result = r3.contains(p3);
              break;
            }
          default:
            assert(false);
        }
        return result;
      }

      int get_dim(void) const { return dim; }

      int get_volume(void) const
      {
        switch (dim)
        {
          case 0:
            return get_index_space().get_valid_mask().get_num_elmts();
          case 1:
            {
              Arrays::Rect<1> r1 = get_rect<1>();
              return r1.volume();
            }
          case 2:
            {
              Arrays::Rect<2> r2 = get_rect<2>();
              return r2.volume();
            }
          case 3:
            {
              Arrays::Rect<3> r3 = get_rect<3>();
              return r3.volume();
            }
          default:
            assert(false);
        }
        return 0;
      }

      template <int DIM>
      Arrays::Rect<DIM> get_rect(void) const { assert(dim == DIM); return Arrays::Rect<DIM>(rect_data); }

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
	      int index = mask->first_enabled();
	      p.point_data[0] = index;
	      any_left = (index >= 0);
	    }
	    break;

	  case 1:
	    {
	      Arrays::GenericPointInRectIterator<1> *pir = new Arrays::GenericPointInRectIterator<1>(d.get_rect<1>());
	      iterator = (void *)pir;
	      pir->p.to_array(p.point_data);
	      any_left = pir->any_left;
	    }
	    break;

	  case 2:
	    {
	      Arrays::GenericPointInRectIterator<2> *pir = new Arrays::GenericPointInRectIterator<2>(d.get_rect<2>());
	      iterator = (void *)pir;
	      pir->p.to_array(p.point_data);
	      any_left = pir->any_left;
	    }
	    break;

	  case 3:
	    {
	      Arrays::GenericPointInRectIterator<3> *pir = new Arrays::GenericPointInRectIterator<3>(d.get_rect<3>());
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
	      Arrays::GenericPointInRectIterator<1> *pir = (Arrays::GenericPointInRectIterator<1> *)iterator;
	      delete pir;
	    }
	    break;

	  case 2:
	    {
	      Arrays::GenericPointInRectIterator<2> *pir = (Arrays::GenericPointInRectIterator<2> *)iterator;
	      delete pir;
	    }
	    break;

	  case 3:
	    {
	      Arrays::GenericPointInRectIterator<3> *pir = (Arrays::GenericPointInRectIterator<3> *)iterator;
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
	      Arrays::GenericPointInRectIterator<1> *pir = (Arrays::GenericPointInRectIterator<1> *)iterator;
	      any_left = pir->step();
	      pir->p.to_array(p.point_data);
	    }
	    break;

	  case 2:
	    {
	      Arrays::GenericPointInRectIterator<2> *pir = (Arrays::GenericPointInRectIterator<2> *)iterator;
	      any_left = pir->step();
	      pir->p.to_array(p.point_data);
	    }
	    break;

	  case 3:
	    {
	      Arrays::GenericPointInRectIterator<3> *pir = (Arrays::GenericPointInRectIterator<3> *)iterator;
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
      int rect_data[2 * MAX_RECT_DIM];

    public:
      // simple instance creation for the lazy
      RegionInstance create_instance(Memory memory, size_t elem_size,
				     ReductionOpID redop_id = 0) const;

      RegionInstance create_instance(Memory memory,
				     const std::vector<size_t> &field_sizes,
				     size_t block_size,
				     ReductionOpID redop_id = 0) const;

      struct CopySrcDstField {
      public:
        CopySrcDstField(void) 
          : inst(RegionInstance::NO_INST), offset(0), size(0) { }
        CopySrcDstField(RegionInstance i, unsigned o, unsigned s)
          : inst(i), offset(o), size(s) { }
      public:
	RegionInstance inst;
	unsigned offset, size;
      };

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
    };

    class IndexSpaceAllocator {
    protected:
      friend class IndexSpace;
      friend class IndexSpace::Impl;
      class Impl;

      IndexSpaceAllocator(Impl *_impl) : impl(_impl) {}

      Impl *impl;
 
    public:
      IndexSpaceAllocator(const IndexSpaceAllocator& to_copy)
	: impl(to_copy.impl) {}

      unsigned alloc(unsigned count = 1) const;
      void reserve(unsigned ptr, unsigned count = 1) const;
      void free(unsigned ptr, unsigned count = 1) const;

      template <typename LIN>
      void reserve(const LIN& linearizer, Arrays::Point<LIN::IDIM> point) const;

      void destroy(void);
    };

#ifdef OLD_ACCESSOR_STUFF
    template <> class RegionInstanceAccessorUntyped<AccessorArray> {
    public:
      explicit RegionInstanceAccessorUntyped(void)
        : array_base(NULL) {}
      explicit RegionInstanceAccessorUntyped(void *_array_base)
	: array_base(_array_base) {}

      // Need copy constructors so we can move things around
      RegionInstanceAccessorUntyped(const RegionInstanceAccessorUntyped<AccessorArray> &old)
      { 
        array_base = old.array_base; 
#ifdef POINTER_CHECKS
        impl_ptr = old.impl_ptr;
#endif
      }

      bool operator<(const RegionInstanceAccessorUntyped<AccessorArray> &rhs) const
      { return array_base < rhs.array_base; }
      bool operator==(const RegionInstanceAccessorUntyped<AccessorArray> &rhs) const
      { return array_base == rhs.array_base; }
      bool operator!=(const RegionInstanceAccessorUntyped<AccessorArray> &rhs) const
      { return array_base != rhs.array_base; }

      void *array_base;
#ifdef POINTER_CHECKS
      void *impl_ptr;

      void verify_access(unsigned ptr) const;

      void set_impl(void *impl)
      {
        impl_ptr = impl;
      }
#endif

      template <class T>
      T read(ptr_t ptr) const 
      { 
#ifdef POINTER_CHECKS
        verify_access(ptr.value);
#endif
        return ((T*)array_base)[ptr.value]; 
      }

      template <class T>
      void write(ptr_t ptr, const T& newval) const 
      { 
#ifdef POINTER_CHECKS
        verify_access(ptr.value);
#endif
        ((T*)array_base)[ptr.value] = newval; 
      }

      template <class REDOP, class T, class RHS>
      void reduce(ptr_t ptr, RHS newval) const 
      { 
#ifdef POINTER_CHECKS
        verify_access(ptr.value);
#endif
        REDOP::template apply<false>(((T*)array_base)[ptr.value], newval); 
      }

      template <class T>
      T &ref(ptr_t ptr) const 
      { 
#ifdef POINTER_CHECKS
        verify_access(ptr.value);
#endif
        return ((T*)array_base)[ptr.value]; 
      }
    };

    template <> class RegionInstanceAccessorUntyped<AccessorArrayReductionFold> {
    public:
      explicit RegionInstanceAccessorUntyped(void *_array_base)
	: array_base(_array_base) {}

      // Need copy constructors so we can move things around
      RegionInstanceAccessorUntyped(const RegionInstanceAccessorUntyped<AccessorArrayReductionFold> &old)
      { array_base = old.array_base; }

      bool operator<(const RegionInstanceAccessorUntyped<AccessorArray> &rhs) const
      { return array_base < rhs.array_base; }
      bool operator==(const RegionInstanceAccessorUntyped<AccessorArray> &rhs) const
      { return array_base == rhs.array_base; }
      bool operator!=(const RegionInstanceAccessorUntyped<AccessorArray> &rhs) const
      { return array_base != rhs.array_base; }

      void *array_base;

      // can't read or write a fold-only accessor
      template <class REDOP, class T, class RHS>
      void reduce(ptr_t ptr, RHS newval) const { REDOP::template fold<true>(((RHS*)array_base)[ptr.value], newval); }
    };

    template <> class RegionInstanceAccessorUntyped<AccessorReductionList> {
    public:
      explicit RegionInstanceAccessorUntyped(void *_internal_data,
					     size_t _num_entries,
					     size_t _elem_size);

      // Need copy constructors so we can move things around
      RegionInstanceAccessorUntyped(const RegionInstanceAccessorUntyped<AccessorReductionList> &old)
	: internal_data(old.internal_data), cur_size(old.cur_size),
	  max_size(old.max_size), entry_list(old.entry_list) {}
      /*
      bool operator<(const RegionInstanceAccessorUntyped<AccessorArray> &rhs) const
      { return array_base < rhs.array_base; }
      bool operator==(const RegionInstanceAccessorUntyped<AccessorArray> &rhs) const
      { return array_base == rhs.array_base; }
      bool operator!=(const RegionInstanceAccessorUntyped<AccessorArray> &rhs) const
      { return array_base != rhs.array_base; }
      */
      void *internal_data;
      size_t *cur_size;
      size_t max_size;
      void *entry_list;

      // can't read or write a fold-only accessor
      template <class REDOP, class T, class RHS>
      void reduce(ptr_t ptr, RHS newval) const { 
        size_t my_pos = __sync_fetch_and_add(cur_size, 1);
	if(my_pos < max_size) {
	  ReductionListEntry<T,RHS> *entry = ((ReductionListEntry<T,RHS> *)entry_list)+my_pos;
	  entry->ptr = ptr;
	  entry->rhs = newval;
	} else {
	  ReductionListEntry<T,RHS> entry;
	  entry.ptr = ptr;
	  entry.rhs = newval;
	  reduce_slow_case(my_pos, ptr.value, &entry, sizeof(entry));
	}
      }

      void flush(void) const;
      
    protected:
      void reduce_slow_case(size_t my_pos, unsigned ptrvalue,
			    const void *entry, size_t sizeof_entry) const;
    };

    // only nvcc understands this
    template <> class RegionInstanceAccessorUntyped<AccessorGPU> {
    public:
      explicit RegionInstanceAccessorUntyped(void *_array_base)
	: array_base(_array_base) {}
      
      void *array_base;
#ifdef POINTER_CHECKS 
      size_t first_elmt;
      size_t last_elmt;
      unsigned *valid_mask_base;
#endif
#ifdef __CUDACC__
      // Need copy constructors so we can move things around
      __host__ __device__
#endif
      RegionInstanceAccessorUntyped(const RegionInstanceAccessorUntyped<AccessorGPU> &old)
      { 
        array_base = old.array_base; 
#ifdef POINTER_CHECKS 
        first_elmt = old.first_elmt;
        last_elmt = old.last_elmt;
	valid_mask_base = old.valid_mask_base;
#endif
      }

      template <class T>
      T *gpu_ptr(ptr_t ptr) const {
#ifdef POINTER_CHECKS 
        bounds_check(ptr);
#endif
	return &((T*)array_base)[ptr.value];
      }

#ifdef __CUDACC__
      template <class T>
      __device__ __forceinline__
      T read(ptr_t ptr) const { 
#ifdef POINTER_CHECKS 
        bounds_check(ptr);
#endif
        return ((T*)array_base)[ptr.value]; 
      }

      template <class T>
      __device__ __forceinline__
      void write(ptr_t ptr, const T& newval) const { 
#ifdef POINTER_CHECKS 
        bounds_check(ptr);
#endif
        ((T*)array_base)[ptr.value] = newval; 
      }

      template <class REDOP, class T, class RHS>
      __device__ __forceinline__
      void reduce(ptr_t ptr, RHS newval) const { 
#ifdef POINTER_CHECKS 
        bounds_check(ptr);
#endif
        REDOP::template apply<false>(((T*)array_base)[ptr.value], newval); 
      }

      template <class T>
      __device__ __forceinline__
      T &ref(ptr_t ptr) const { 
#ifdef POINTER_CHECKS 
        bounds_check(ptr);
#endif
        return ((T*)array_base)[ptr.value]; 
      }
#endif

#ifdef POINTER_CHECKS 
      template <class T>
#ifdef __CUDACC__
      __device__ __forceinline__
#endif
      void bounds_check(ptr_t ptr) const
      {
        assert((first_elmt <= ptr.value) && (ptr.value <= last_elmt));
	off_t rel_ptr = ptr.value ;//- first_elmt;
	unsigned bits = valid_mask_base[rel_ptr >> 5];
	assert(bits & (1U << (rel_ptr & 0x1f)));
      }
#endif
    };

    template <> class RegionInstanceAccessorUntyped<AccessorGPUReductionFold> {
    public:
      explicit RegionInstanceAccessorUntyped(void *_array_base)
	: array_base(_array_base) {}

      void *array_base;
#ifdef POINTER_CHECKS 
      size_t first_elmt;
      size_t last_elmt;
      unsigned *valid_mask_base;
#endif
#ifdef __CUDACC__
      // Need copy constructors so we can move things around
      __host__ __device__
      RegionInstanceAccessorUntyped(const RegionInstanceAccessorUntyped<AccessorGPUReductionFold> &old)
      { 
        array_base = old.array_base; 
#ifdef POINTER_CHECKS 
        first_elmt = old.first_elmt;
        last_elmt  = old.last_elmt;
	valid_mask_base = old.valid_mask_base;
#endif
      }
      // no read or write on a reduction-fold-only accessor
      template <class REDOP, class T, class RHS>
      __device__ __forceinline__
      void reduce(ptr_t ptr, RHS newval) const { 
#ifdef POINTER_CHECKS 
        assert((first_elmt <= ptr.value) && (ptr.value <= last_elmt));
	off_t rel_ptr = ptr.value ;//- first_elmt;
	unsigned bits = valid_mask_base[rel_ptr >> 5];
	assert(bits & (1U << (rel_ptr & 0x1f)));
#endif
        REDOP::fold<false>(((RHS*)array_base)[ptr.value], newval); 
      }
#else // __CUDACC__
      RegionInstanceAccessorUntyped(const RegionInstanceAccessorUntyped<AccessorGPUReductionFold> &old)
      {
        array_base = old.array_base;
#ifdef POINTER_CHECKS 
        first_elmt = old.first_elmt;
        last_elmt  = old.last_elmt;
	valid_mask_base = old.valid_mask_base;
#endif
      }
#endif
    };

    template <class ET, AccessorType AT = AccessorGeneric>
    class RegionInstanceAccessor {
    public:
      RegionInstanceAccessor(const RegionInstanceAccessorUntyped<AT> &_ria) : ria(_ria) {}

      RegionInstanceAccessorUntyped<AT> ria;

      ET read(ptr_t ptr) const { return ria.read(ptr); }
      void write(ptr_t ptr, const ET& newval) const { ria.write(ptr, newval); }
      ET& ref(ptr_t ptr) const { return ria.ref(ptr); }

      void read_partial(ptr_t ptr, off_t offset, void *dst, size_t size) const
      { ria.read_partial(ptr, offset, dst, size); }

      void write_partial(ptr_t ptr, off_t offset, const void *src, size_t size) const
      { ria.write_partial(ptr, offset, src, size); }

      template <class REDOP, class RHS>
      void reduce(ptr_t ptr, RHS newval) const { ria.template reduce<REDOP>(ptr, newval); }

      template <AccessorType AT2>
      bool can_convert(void) const { return ria.can_convert<AT2>(); }

      template <AccessorType AT2>
      RegionInstanceAccessor<ET,AT2> convert(void) const
      { return RegionInstanceAccessor<ET,AT2>(ria.convert<AT2>()); }

      template <AccessorType AT2, class T2>
      RegionInstanceAccessor<ET,AT2> convert2(T2 arg) const
      { return RegionInstanceAccessor<ET,AT2>(ria.convert2<AT2,T2>(arg)); }
    };

#ifdef __CUDACC__
    template <class ET>
    class RegionInstanceAccessor<ET,AccessorGPU> {
    public:
      __device__ RegionInstanceAccessor(const RegionInstanceAccessorUntyped<AccessorGPU> &_ria) : ria(_ria) {}

      RegionInstanceAccessorUntyped<AccessorGPU> ria;

      __device__ ET read(ptr_t ptr) const { return ria.read(ptr); }
      __device__ void write(ptr_t ptr, const ET& newval) const { ria.write(ptr, newval); }

      //template <class REDOP, class RHS>
      //void reduce(ptr_t<ET> ptr, RHS newval) const { ria.template reduce<REDOP>(ptr, newval); }
    };
#endif
#endif

    class Machine {
    public:
      Machine(int *argc, char ***argv,
	      const Processor::TaskIDTable &task_table,
	      const ReductionOpTable &redop_table,
	      bool cps_style = false, Processor::TaskFuncID init_id = 0);
      ~Machine(void);

      // there are three potentially interesting ways to start the initial
      // tasks:
      enum RunStyle {
	ONE_TASK_ONLY,  // a single task on a single node of the machine
	ONE_TASK_PER_NODE, // one task running on one proc of each node
	ONE_TASK_PER_PROC, // a task for every processor in the machine
      };


      void run(Processor::TaskFuncID task_id = 0, RunStyle style = ONE_TASK_ONLY,
	       const void *args = 0, size_t arglen = 0, bool background = false);

      // requests a shutdown of all the processors
      void shutdown(bool local_request = true);

      void wait_for_shutdown(void);

    public:
      const std::set<Memory>&    get_all_memories(void) const { return memories; }
      const std::set<Processor>& get_all_processors(void) const { return procs; }
      // Return the set of memories visible from a processor
      const std::set<Memory>&    get_visible_memories(Processor p) const
      { return visible_memories_from_procs.find(p)->second; }

      // Return the set of memories visible from a memory
      const std::set<Memory>&    get_visible_memories(Memory m) const
      { return visible_memories_from_memory.find(m)->second; }

      // Return the set of processors which can all see a given memory
      const std::set<Processor>& get_shared_processors(Memory m) const
      { return visible_procs_from_memory.find(m)->second; }

      // Return the set of processors "local" to a given other one
      const std::set<Processor>& get_local_processors(Processor p) const;
      // Return whether or not the machine is running with explicit utility processors
      bool has_explicit_utility_processors(void) const { return explicit_utility_procs; }

      Processor::Kind get_processor_kind(Processor p) const;
      Memory::Kind get_memory_kind(Memory m) const;
      size_t get_memory_size(const Memory m) const;

      //void add_processor(Processor p) { procs.insert(p); }
      static Machine* get_machine(void);

      static Processor get_executing_processor(void);
    public:
      struct ProcessorMemoryAffinity {
	Processor p;
	Memory m;
	unsigned bandwidth; // TODO: consider splitting read vs. write?
	unsigned latency;
      };

      struct MemoryMemoryAffinity {
	Memory m1, m2;
	unsigned bandwidth;
	unsigned latency;
      };

      int get_proc_mem_affinity(std::vector<ProcessorMemoryAffinity>& result,
				Processor restrict_proc = Processor::NO_PROC,
				Memory restrict_memory = Memory::NO_MEMORY);

      int get_mem_mem_affinity(std::vector<MemoryMemoryAffinity>& result,
			       Memory restrict_mem1 = Memory::NO_MEMORY,
			       Memory restrict_mem2 = Memory::NO_MEMORY);

    protected:
      std::set<Processor> procs;
      std::set<Memory> memories;
      std::vector<ProcessorMemoryAffinity> proc_mem_affinities;
      std::vector<MemoryMemoryAffinity> mem_mem_affinities;
      std::map<Processor,std::set<Memory> > visible_memories_from_procs;
      std::map<Memory,std::set<Memory> > visible_memories_from_memory;
      std::map<Memory,std::set<Processor> > visible_procs_from_memory;
      bool explicit_utility_procs;
      void *background_pthread; // pointer to pthread_t in the background
    public:
      struct NodeAnnounceData;

      void parse_node_announce_data(const void *args, size_t arglen,
				    const NodeAnnounceData& annc_data,
				    bool remote);
    };

    // Implementations for template functions

    template <class T>
    /*static*/ int ElementMask::forall_ranges(T &executor,
					      const ElementMask &mask,
					      int start /*= 0*/,
					      int count /*= -1*/,
					      bool do_enabled /*= true*/)
    {
      if(count == 0) return 0;

      ElementMask::Enumerator enum1(mask, start, do_enabled ? 1 : 0);

      int total = 0;

      int pos, len;
      while(enum1.get_next(pos, len)) {
	if(pos < start) {
	  len -= (start - pos);
	  pos = start;
	}

	if((count > 0) && ((pos + len) > (start + count))) {
	  len = start + count - pos;
	}

	if(len > 0) {
	  //printf("S:%d(%d)\n", pos, len);
	  executor.do_span(pos, len);
	  total += len;
	}
      }

      return total;
    }

    template <class T>
    /*static*/ int ElementMask::forall_ranges(T &executor,
					      const ElementMask &mask1, 
					      const ElementMask &mask2,
					      int start /*= 0*/,
					      int count /*= -1*/,
					      bool do_enabled1 /*= true*/,
					      bool do_enabled2 /*= true*/)
    {
      ElementMask::Enumerator enum1(mask1, start, do_enabled1 ? 1 : 0);
      ElementMask::Enumerator enum2(mask2, start, do_enabled2 ? 1 : 0);

      int pos1, len1, pos2, len2;

      if(!enum1.get_next(pos1, len1)) return 0;
      if(!enum2.get_next(pos2, len2)) return 0;
      if(count == 0) return 0;

      int total = 0;

      while(true) {
	//printf("S:%d(%d) T:%d(%d)\n", pos1, len1, pos2, len2);

	if(len1 <= 0) {
	  if(!enum1.get_next(pos1, len1)) break;
	  if((count > 0) && ((pos1 + len1) > (start + count))) {
	    len1 = (start + count) - pos1;
	    if(len1 < 0) break;
	  }
	  continue;
	}

	if(len2 <= 0) {
	  if(!enum2.get_next(pos2, len2)) break;
	  if((count > 0) && ((pos2 + len2) > (start + count))) {
	    len2 = (start + count) - pos2;
	    if(len2 < 0) break;
	  }
	  continue;
	}

	if(pos1 < pos2) {
	  len1 -= (pos2 - pos1);
	  pos1 = pos2;
	  continue;
	}

	if(pos2 < pos1) {
	  len2 -= (pos1 - pos2);
	  pos2 = pos1;
	  continue;
	}

	assert((pos1 == pos2) && (len1 > 0) && (len2 > 0));

	int span_len = (len1 < len2) ? len1 : len2;

	executor.do_span(pos1, span_len);

	pos1 += span_len;
	len1 -= span_len;
	pos2 += span_len;
	len2 -= span_len;

	total += span_len;
      }

      return total;
    }

  }; // namespace LowLevel
}; // namespace LegionRuntime

#endif
