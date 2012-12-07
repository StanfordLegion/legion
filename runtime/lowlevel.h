/* Copyright 2012 Stanford University
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

#include "common.h"
#include "utilities.h"

namespace LegionRuntime {
  namespace LowLevel {
    // forward class declarations because these things all refer to each other
    class Event;
    class UserEvent;
    class Lock;
    class Memory;
    class Processor;

    class IndexSpace;
    class IndexSpaceAllocator;
    class RegionInstance;

    class Machine;

    class Event {
    public:
      typedef unsigned id_t;
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
      void trigger(void) const;
    };

    // a Barrier is similar to a UserEvent, except that it has a count of how
    //  many threads (or whatever) need to "trigger" before the actual trigger
    //  occurs
    class Barrier : public Event {
    public:
      static Barrier create_barrier(unsigned expected_arrivals);

      void alter_arrival_count(int delta) const;

      void arrive(unsigned count = 1) const;
    };

    class Lock {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const Lock& rhs) const { return id < rhs.id; }
      bool operator==(const Lock& rhs) const { return id == rhs.id; }
      bool operator!=(const Lock& rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const Lock NO_LOCK;

      bool exists(void) const { return id != 0; }

      // requests ownership (either exclusive or shared) of the lock with a 
      //   specified mode - returns an event that will trigger when the lock
      //   is granted
      Event lock(unsigned mode = 0, bool exclusive = true, Event wait_on = Event::NO_EVENT) const;
      // releases a held lock - release can be deferred until an event triggers
      void unlock(Event wait_on = Event::NO_EVENT) const;

      // Create a new lock, destroy an existing lock
      static Lock create_lock(size_t _data_size = 0);
      void destroy_lock();

      size_t data_size(void) const;
      void *data_ptr(void) const;
    };

    class Processor {
    public:
      typedef unsigned id_t;
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
		  Event wait_on = Event::NO_EVENT) const;
    };

    class Memory {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const Memory &rhs) const { return id < rhs.id; }
      bool operator==(const Memory &rhs) const { return id == rhs.id; }
      bool operator!=(const Memory &rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const Memory NO_MEMORY;

      bool exists(void) const { return id != 0; }
    };

    class ElementMask {
    public:
      ElementMask(void);
      explicit ElementMask(int num_elements, int first_element = 0);
      ElementMask(const ElementMask &copy_from, int num_elements = -1, int first_element = 0);

      void init(int _first_element, int _num_elements, Memory _memory, off_t _offset);

      int get_num_elmts(void) const { return num_elements; }

      void enable(int start, int count = 1);
      void disable(int start, int count = 1);

      int find_enabled(int count = 1);
      int find_disabled(int count = 1);
      
      bool is_set(int ptr) const;
      // union/intersect/subtract?

      int first_enabled(void) const { return first_enabled_elmt; }
      int last_enabled(void) const { return last_enabled_elmt; }

      ElementMask& operator=(const ElementMask &rhs);

      enum OverlapResult { OVERLAP_NO, OVERLAP_MAYBE, OVERLAP_YES };

      OverlapResult overlaps_with(const ElementMask& other,
				  off_t max_effort = -1) const;

      class Enumerator {
      public:
	Enumerator(const ElementMask& _mask, int _start, int _polarity);
	~Enumerator(void);

	bool get_next(int &position, int &length);

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

    protected:
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

    typedef unsigned ReductionOpID;
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
      virtual void fold(void *rhs1_ptr, const void *rhs2_ptr, size_t count,
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

    enum AccessorType { AccessorGeneric, AccessorArray, 
                        AccessorArrayReductionFold, AccessorReductionList,
			AccessorGPU, AccessorGPUReductionFold, };

    template <AccessorType AT> class RegionAccessor;

    template <> class RegionAccessor<AccessorGeneric> {
    public:
      explicit RegionAccessor(void)
        : internal_data(NULL), field_offset(0) {}
      explicit RegionAccessor(void *_internal_data, off_t _field_offset = 0)
	: internal_data(_internal_data), field_offset(_field_offset) {}

      // Need copy constructors so we can move things around
      RegionAccessor(const RegionAccessor<AccessorGeneric> &old)
	: internal_data(old.internal_data), field_offset(old.field_offset) {}

      inline bool operator<(const RegionAccessor<AccessorGeneric> &rhs) const
      { return internal_data < rhs.internal_data; }
      inline bool operator==(const RegionAccessor<AccessorGeneric> &rhs) const
      { return internal_data == rhs.internal_data; }
      inline bool operator!=(const RegionAccessor<AccessorGeneric> &rhs) const
      { return internal_data != rhs.internal_data; }

      void *internal_data;
      off_t field_offset;

#ifdef POINTER_CHECKS
      void verify_access(unsigned ptr) const;
#endif
      void get_untyped(int index, off_t byte_offset, void *dst, size_t size) const;
      void put_untyped(int index, off_t byte_offset, const void *src, size_t size) const;

      template <typename T>
      inline T read(ptr_t ptr) const
	{ 
	  assert(!is_reduction_only());
#ifdef POINTER_CHECKS
          verify_access(ptr.value);
#endif
	  T val; get_untyped(ptr.value, 0, &val, sizeof(T)); return val;
	}

      template <typename T>
      inline void read_partial(ptr_t ptr, off_t offset, void *dst, size_t size) const
	{
#ifdef POINTER_CHECKS
          verify_access(ptr.value);
#endif
	  get_untyped(ptr.value, offset, dst, size);
	}

      template <typename T>
      inline void write(ptr_t ptr, const T& newval) const
	{
#ifdef POINTER_CHECKS
          verify_access(ptr.value);
#endif
	  assert(!is_reduction_only());
	  put_untyped(ptr.value, 0, &newval, sizeof(T));
	}

      template <typename T>
      inline void write_partial(ptr_t ptr, off_t offset, const void *src, size_t size) const
	{
#ifdef POINTER_CHECKS
          verify_access(ptr.value);
#endif
	  put_untyped(ptr.value, offset, src, size);
	}

      template <typename REDOP, typename T, typename RHS>
      inline void reduce(ptr_t ptr, RHS newval) const 
	{ 
#ifdef POINTER_CHECKS
          verify_access(ptr.value);
#endif
  	  if(is_reduction_only()) {
	    RHS val; 
	    get_untyped(ptr.value, 0, &val, sizeof(RHS));
	    REDOP::template fold<true>(val, newval); // made our own copy, so 'exclusive'
	    put_untyped(ptr.value, 0, &val, sizeof(RHS));
	  } else {
	    T val; 
	    get_untyped(ptr.value, 0, &val, sizeof(T));
	    REDOP::template apply<true>(val, newval); // made our own copy, so 'exclusive'
	    put_untyped(ptr.value, 0, &val, sizeof(T));
	  }
	}

      template <AccessorType AT2>
      bool can_convert(void) const;

      template <AccessorType AT2>
      RegionAccessor<AT2> convert(void) const;

      RegionAccessor<AccessorGeneric> get_field_accessor(off_t offset, size_t size) const;

    protected:
      bool is_reduction_only(void) const;
    };

    template <> class RegionAccessor<AccessorArray> {
    public:
      explicit RegionAccessor(void)
        : array_base(NULL) { }
      explicit RegionAccessor(void *_array_base, off_t _field_offset = 0)
        : array_base(_array_base), field_offset(_field_offset) { }

      // Copy constructors so we can move things around
      RegionAccessor(const RegionAccessor<AccessorArray> &rhs)
        {
          array_base = rhs.array_base;
          field_offset = rhs.field_offset;
#ifdef POINTER_CHECKS
          impl_ptr = rhs.impl_ptr;
#endif
        }

      inline bool operator<(const RegionAccessor<AccessorArray> &rhs) const
      { return array_base < rhs.array_base; }
      inline bool operator==(const RegionAccessor<AccessorArray> &rhs) const
      { return array_base == rhs.array_base; }
      inline bool operator!=(const RegionAccessor<AccessorArray> &rhs) const
      { return array_base != rhs.array_base; }

      void *array_base;
      off_t field_offset;
#ifdef POINTER_CHECKS
      void *impl_ptr;

      void verify_access(unsigned ptr) const;

      inline void set_impl(void *impl)
      {
        impl_ptr = impl;
      }
#endif

      template<typename T>
      inline T read(ptr_t ptr) const
        {
#ifdef POINTER_CHECKS
          verify_access(ptr.value);
#endif
          return ((T*)array_base)[ptr.value];
        }

      template<typename T>
      inline void write(ptr_t ptr, const T &newval) const
        {
#ifdef POINTER_CHECKS
          verify_access(ptr.value);
#endif
          ((T*)array_base)[ptr.value] = newval;
        }

      template<typename REDOP, typename T, typename RHS>
      inline void reduce(ptr_t ptr, RHS newval) const
        {
#ifdef POINTER_CHECKS
          verify_access(ptr.value);
#endif
          REDOP::template appy<false>(((T*)array_base)[ptr.value], newval);
        }

      template<typename T>
      inline T& ref(ptr_t ptr) const
        {
#ifdef POINTER_CHECKS
          verify_access(ptr.value);
#endif
          return ((T*)array_base)[ptr.value];
        }
    };

    template <> class RegionAccessor<AccessorArrayReductionFold> {
    public:
      explicit RegionAccessor(void)
        : array_base(NULL) { }
      explicit RegionAccessor(void *_array_base)
        : array_base(_array_base) { }

      // Need copy constructors so we can move things around
      RegionAccessor(const RegionAccessor<AccessorArrayReductionFold> &rhs)
      {
        array_base = rhs.array_base;
      }

      inline bool operator<(const RegionAccessor<AccessorArrayReductionFold> &rhs) const
      { return array_base < rhs.array_base; }
      inline bool operator==(const RegionAccessor<AccessorArrayReductionFold> &rhs) const
      { return array_base == rhs.array_base; }
      inline bool operator!=(const RegionAccessor<AccessorArrayReductionFold> &rhs) const
      { return array_base != rhs.array_base; }

      void *array_base;

      template<typename REDOP, typename T, typename RHS>
      inline void reduce(ptr_t ptr, RHS newval) const
        {
          REDOP::template fold<true>(((RHS*)array_base)[ptr.value], newval); 
        }
    };

    template <> class RegionAccessor<AccessorReductionList> {
    public:
      explicit RegionAccessor(void) 
        : internal_data(NULL), cur_size(NULL), max_size(0), entry_list(NULL) { }
      explicit RegionAccessor(void *_internal_data, size_t _num_entries, size_t _elem_size);

      RegionAccessor(const RegionAccessor<AccessorReductionList> &rhs) 
        : internal_data(rhs.internal_data), cur_size(rhs.cur_size),
          max_size(rhs.max_size), entry_list(rhs.entry_list) { }

      bool operator<(const RegionAccessor<AccessorReductionList> &rhs) const
      { return internal_data < rhs.internal_data; }
      bool operator==(const RegionAccessor<AccessorReductionList> &rhs) const
      { return internal_data == rhs.internal_data; }
      bool operator!=(const RegionAccessor<AccessorReductionList> &rhs) const
      { return internal_data != rhs.internal_data; }

      void *internal_data;
      size_t *cur_size;
      size_t max_size;
      void *entry_list;

      template<typename REDOP, typename T, typename RHS>
      inline void reduce(ptr_t ptr, RHS newval) const
        {
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
    template <> class RegionAccessor<AccessorGPU> {
    public:
      explicit RegionAccessor(void)
        : array_base(NULL) { }
      explicit RegionAccessor(void *_array_base)
        : array_base(_array_base) { }

#ifdef __CUDACC__
      __host__ __device__ __forceinline__
#endif
      bool operator<(const RegionAccessor<AccessorGPU> &rhs) const
      { return array_base < rhs.array_base; }
#ifdef __CUDACC__
      __host__ __device__ __forceinline__
#endif
      bool operator==(const RegionAccessor<AccessorGPU> &rhs) const
      { return array_base == rhs.array_base; }
#ifdef __CUDACC__
      __host__ __device__ __forceinline__
#endif
      bool operator!=(const RegionAccessor<AccessorGPU> &rhs) const
      { return array_base != rhs.array_base; }

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
      RegionAccessor(const RegionAccessor<AccessorGPU> &rhs)
      {
        array_base = rhs.array_base;
#ifdef POINTER_CHECKS
        first_elmt = rhs.first_elmt;
        last_elmt = rhs.last_elmt;
        valid_mask_base = rhs.valid_mask_base;
#endif
      }

#ifdef __CUDACC__
      template<typename T>
      __device__ __forceinline__
      T read(ptr_t ptr) const 
        {
#ifdef POINTER_CHECKS
          bounds_check(ptr);
#endif
          return ((T*)array_base)[ptr.value];
        }

      template <typename T>
      __device__ __forceinline__
      void write(ptr_t ptr, const T& newval) const 
        { 
#ifdef POINTER_CHECKS 
          bounds_check(ptr);
#endif
          ((T*)array_base)[ptr.value] = newval; 
        }

      template <typename REDOP, typename T, typename RHS>
      __device__ __forceinline__
      void reduce(ptr_t ptr, RHS newval) const 
        { 
#ifdef POINTER_CHECKS 
          bounds_check(ptr);
#endif
          REDOP::template apply<false>(((T*)array_base)[ptr.value], newval); 
        }

      template <typename T>
      __device__ __forceinline__
      T &ref(ptr_t ptr) const 
        { 
#ifdef POINTER_CHECKS 
          bounds_check(ptr);
#endif
          return ((T*)array_base)[ptr.value]; 
        }

#ifdef POINTER_CHECKS
      template<typename T>
      __device__ __forceinline__
      void bounds_checks(ptr_t ptr) const
        {
          // Assertions only work for CUDA 20 
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
          assert((first_elmt <= ptr.value) && (ptr.value <= last_elmt));
          off_t rel_ptr = ptr.value ;//- first_elmt;
          unsigned bits = valid_mask_base[rel_ptr >> 5];
          assert(bits & (1U << (rel_ptr & 0x1f)));
#else
          // If you got a compiler error here, it means you're trying to 
          // do pointer checks on a GPU that doesn't support assertions.
          // Rather than failing silently you get this nice compile error. :)
          LEGION_STATIC_ASSERT(false);
#endif
        }
#endif
#endif
    };

    template <> class RegionAccessor<AccessorGPUReductionFold> {
    public:
      explicit RegionAccessor(void)
        : array_base(NULL) { }
      explicit RegionAccessor(void *_array_base)
        : array_base(_array_base) { }

#ifdef __CUDACC__
      __host__ __device__ __forceinline__
#endif
      bool operator<(const RegionAccessor<AccessorGPU> &rhs) const
      { return array_base < rhs.array_base; }
#ifdef __CUDACC__
      __host__ __device__ __forceinline__
#endif
      bool operator==(const RegionAccessor<AccessorGPU> &rhs) const
      { return array_base == rhs.array_base; }
#ifdef __CUDACC__
      __host__ __device__ __forceinline__
#endif
      bool operator!=(const RegionAccessor<AccessorGPU> &rhs) const
      { return array_base != rhs.array_base; }

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
      RegionAccessor(const RegionAccessor<AccessorGPU> &rhs)
      {
        array_base = rhs.array_base;
#ifdef POINTER_CHECKS
        first_elmt = rhs.first_elmt;
        last_elmt = rhs.last_elmt;
        valid_mask_base = rhs.valid_mask_base;
#endif
      }
#ifdef __CUDACC__
      template<typename REDOP, typename T, typename RHS>
      __device__ __forceinline__
      void reduce(ptr_t ptr, RHS newval) const
        {
#ifdef POINTER_CHECKS
          // Assertions only work for CUDA 20 
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
          assert((first_elmt <= ptr.value) && (ptr.value <= last_elmt));
          off_t rel_ptr = ptr.value ;//- first_elmt;
          unsigned bits = valid_mask_base[rel_ptr >> 5];
          assert(bits & (1U << (rel_ptr & 0x1f)));
#else
          // If you got a compiler error here, it means you're trying to 
          // do pointer checks on a GPU that doesn't support assertions.
          // Rather than failing silently you get this nice compile error. :)
          LEGION_STATIC_ASSERT(false);
#endif
#endif
          REDOP::fold<false>(((RHS*)array_base)[ptr.value], newval);
        }
#endif
    };

    class RegionInstance {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const RegionInstance &rhs) const { return id < rhs.id; }
      bool operator==(const RegionInstance &rhs) const { return id == rhs.id; }
      bool operator!=(const RegionInstance &rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const RegionInstance NO_INST;

      bool exists(void) const { return id != 0; }

      void destroy(void) const;

      RegionAccessor<AccessorGeneric> get_accessor(void) const;
    };

    class IndexSpace {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const IndexSpace &rhs) const { return id < rhs.id; }
      bool operator==(const IndexSpace &rhs) const { return id == rhs.id; }
      bool operator!=(const IndexSpace &rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const IndexSpace NO_SPACE;

      bool exists(void) const { return id != 0; }

      static IndexSpace create_index_space(size_t num_elmts);
      static IndexSpace create_index_space(IndexSpace parent,
					   const ElementMask &mask);

      static IndexSpace expand_index_space(IndexSpace child,
					   size_t num_elmts,
					   off_t child_offset = 0);

      // simple instance creation for the lazy
      RegionInstance create_instance(Memory memory, size_t elem_size,
				     ReductionOpID redop_id = 0) const;

      RegionInstance create_instance(Memory memory,
				     const std::vector<size_t> &field_sizes,
				     size_t block_size,
				     ReductionOpID redop_id = 0) const;

      void destroy(void) const;

      IndexSpaceAllocator create_allocator(void) const;

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

      const ElementMask &get_valid_mask(void) const;
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
	       const void *args = 0, size_t arglen = 0);

      // requests a shutdown of all the processors
      void shutdown(void);

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

      Processor::Kind get_processor_kind(Processor p) const;
      size_t get_memory_size(const Memory m) const;

      //void add_processor(Processor p) { procs.insert(p); }
      static Machine* get_machine(void);
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
