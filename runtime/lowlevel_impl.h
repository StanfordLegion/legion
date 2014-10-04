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

#ifndef LOWLEVEL_IMPL_H
#define LOWLEVEL_IMPL_H

// For doing bit masks for maximum number of nodes
#include "legion_types.h"
#include "legion_utilities.h"

#define NODE_MASK_TYPE uint64_t
#define NODE_MASK_SHIFT 6
#define NODE_MASK_MASK 0x3F

#ifndef MAX_NUM_THREADS
#define MAX_NUM_THREADS 32
#endif

#include "lowlevel.h"

#include <assert.h>

#ifndef NO_INCLUDE_GASNET
#include "activemsg.h"
#endif

#ifdef CHECK_REENTRANT_MESSAGES
GASNETT_THREADKEY_DEFINE(in_handler);
#endif
GASNETT_THREADKEY_DECLARE(cur_thread);

#include <pthread.h>
#include <string.h>

#include <vector>
#include <deque>
#include <queue>
#include <set>
#include <list>
#include <map>

// Comment this out to disable shared task queue

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

#define CHECK_GASNET(cmd) do { \
  int ret = (cmd); \
  if(ret != GASNET_OK) { \
    fprintf(stderr, "GASNET: %s = %d (%s, %s)\n", #cmd, ret, gasnet_ErrorName(ret), gasnet_ErrorDesc(ret)); \
    exit(1); \
  } \
} while(0)

// GASnet helper stuff

namespace LegionRuntime {
  namespace LowLevel {
    extern Logger::Category log_mutex;

#ifdef EVENT_TRACING
    // For event tracing
    struct EventTraceItem {
    public:
      enum Action {
        ACT_CREATE = 0,
        ACT_QUERY = 1,
        ACT_TRIGGER = 2,
        ACT_WAIT = 3,
      };
    public:
      unsigned time_units, event_id, event_gen, action;
    };
#endif

#ifdef LOCK_TRACING
    // For lock tracing
    struct LockTraceItem {
    public:
      enum Action {
        ACT_LOCAL_REQUEST = 0, // request for a lock where the owner is local
        ACT_REMOTE_REQUEST = 1, // request for a lock where the owner is not local
        ACT_FORWARD_REQUEST = 2, // for forwarding of requests
        ACT_LOCAL_GRANT = 3, // local grant of the lock
        ACT_REMOTE_GRANT = 4, // remote grant of the lock (change owners)
        ACT_REMOTE_RELEASE = 5, // remote release of a shared lock
      };
    public:
      unsigned time_units, lock_id, owner, action;
    };
#endif

    class AutoHSLLock {
    public:
      AutoHSLLock(gasnet_hsl_t &mutex) : mutexp(&mutex), held(true)
      { 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK IN %p", mutexp);
	//printf("[%d] MUTEX LOCK IN %p\n", gasnet_mynode(), mutexp);
	gasnet_hsl_lock(mutexp); 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK HELD %p", mutexp);
	//printf("[%d] MUTEX LOCK HELD %p\n", gasnet_mynode(), mutexp);
      }
      AutoHSLLock(gasnet_hsl_t *_mutexp) : mutexp(_mutexp), held(true)
      { 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK IN %p", mutexp);
	//printf("[%d] MUTEX LOCK IN %p\n", gasnet_mynode(), mutexp);
	gasnet_hsl_lock(mutexp); 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK HELD %p", mutexp);
	//printf("[%d] MUTEX LOCK HELD %p\n", gasnet_mynode(), mutexp);
      }
      ~AutoHSLLock(void) 
      {
	if(held)
	  gasnet_hsl_unlock(mutexp);
	log_mutex(LEVEL_SPEW, "MUTEX LOCK OUT %p", mutexp);
	//printf("[%d] MUTEX LOCK OUT %p\n", gasnet_mynode(), mutexp);
      }
      void release(void)
      {
	assert(held);
	gasnet_hsl_unlock(mutexp);
	held = false;
      }
      void reacquire(void)
      {
	assert(!held);
	gasnet_hsl_lock(mutexp);
	held = true;
      }
    protected:
      gasnet_hsl_t *mutexp;
      bool held;
    };

    typedef LegionRuntime::HighLevel::BitMask<NODE_MASK_TYPE,MAX_NUM_NODES,
                                              NODE_MASK_SHIFT,NODE_MASK_MASK> NodeMask;

    // gasnet_hsl_t in object form for templating goodness
    class GASNetHSL {
    public:
      GASNetHSL(void) { gasnet_hsl_init(&mutex); }
      ~GASNetHSL(void) { gasnet_hsl_destroy(&mutex); }

      void lock(void) { gasnet_hsl_lock(&mutex); }
      void unlock(void) { gasnet_hsl_unlock(&mutex); }

    protected:
      gasnet_hsl_t mutex;
    };

    // we have a base type that's element-type agnostic
    template <typename LT, typename IT>
    struct DynamicTableNodeBase {
    public:
      DynamicTableNodeBase(int _level, IT _first_index, IT _last_index)
        : level(_level), first_index(_first_index), last_index(_last_index) {}

      int level;
      IT first_index, last_index;
      LT lock;
    };

    template <typename ET, size_t _SIZE, typename LT, typename IT>
      struct DynamicTableNode : public DynamicTableNodeBase<LT, IT> {
    public:
      static const size_t SIZE = _SIZE;

      DynamicTableNode(int _level, IT _first_index, IT _last_index)
        : DynamicTableNodeBase<LT, IT>(_level, _first_index, _last_index) {}

      ET elems[SIZE];
    };

    template <typename ALLOCATOR> class DynamicTableFreeList;

    template <typename ALLOCATOR>
    class DynamicTable {
    public:
      typedef typename ALLOCATOR::IT IT;
      typedef typename ALLOCATOR::ET ET;
      typedef typename ALLOCATOR::LT LT;
      typedef DynamicTableNodeBase<LT, IT> NodeBase;

      DynamicTable(void);
      ~DynamicTable(void);

      size_t max_entries(void) const;
      bool has_entry(IT index) const;
      ET *lookup_entry(IT index, int owner, typename ALLOCATOR::FreeList *free_list = 0);

    protected:
      NodeBase *new_tree_node(int level, IT first_index, IT last_index,
			      int owner, typename ALLOCATOR::FreeList *free_list);

      // lock protects _changes_ to 'root', but not access to it
      GASNetHSL lock;
      NodeBase * volatile root;
    };

    template <typename ALLOCATOR>
    class DynamicTableFreeList {
    public:
      typedef typename ALLOCATOR::IT IT;
      typedef typename ALLOCATOR::ET ET;
      typedef typename ALLOCATOR::LT LT;

      DynamicTableFreeList(DynamicTable<ALLOCATOR>& _table, int _owner);

      ET *alloc_entry(void);
      void free_entry(ET *entry);

      DynamicTable<ALLOCATOR>& table;
      int owner;
      GASNetHSL lock;
      ET * volatile first_free;
      IT volatile next_alloc;
    };

    template <typename ALLOCATOR>
    DynamicTable<ALLOCATOR>::DynamicTable(void)
      : root(0)
    {
    }

    template <typename ALLOCATOR>
    DynamicTable<ALLOCATOR>::~DynamicTable(void)
    {
    }

    template <typename ALLOCATOR>
    typename DynamicTable<ALLOCATOR>::NodeBase *DynamicTable<ALLOCATOR>::new_tree_node(int level, IT first_index, IT last_index, int owner, typename ALLOCATOR::FreeList *free_list /*= 0*/)
    {
      if(level > 0) {
	// an inner node - we can create that ourselves
	typename ALLOCATOR::INNER_TYPE *inner = new typename ALLOCATOR::INNER_TYPE(level, first_index, last_index);
	for(IT i = 0; i < ALLOCATOR::INNER_TYPE::SIZE; i++)
	  inner->elems[i] = 0;
	return inner;
      } else {
	return ALLOCATOR::new_leaf_node(first_index, last_index, owner, free_list);
      }
    }

    template<typename ALLOCATOR>
    size_t DynamicTable<ALLOCATOR>::max_entries(void) const
    {
      if (!root)
        return 0;
      size_t elems_addressable = 1 << ALLOCATOR::LEAF_BITS;
      for (int i = 0; i < root->level; i++)
        elems_addressable <<= ALLOCATOR::INNER_BITS;
      return elems_addressable;
    }

    template<typename ALLOCATOR>
    bool DynamicTable<ALLOCATOR>::has_entry(IT index) const
    {
      // first, figure out how many levels the tree must have to find our index
      int level_needed = 0;
      int elems_addressable = 1 << ALLOCATOR::LEAF_BITS;
      while(index >= elems_addressable) {
	level_needed++;
	elems_addressable <<= ALLOCATOR::INNER_BITS;
      }

      NodeBase *n = root;
      if (!n || (n->level < level_needed))
        return false;

      // when we get here, root is high enough
      assert((level_needed <= n->level) &&
	     (index >= n->first_index) &&
	     (index <= n->last_index));

      // now walk tree, populating the path we need
      while(n->level > 0) {
	// intermediate nodes
	typename ALLOCATOR::INNER_TYPE *inner = static_cast<typename ALLOCATOR::INNER_TYPE *>(n);

	IT i = ((index >> (ALLOCATOR::LEAF_BITS + (n->level - 1) * ALLOCATOR::INNER_BITS)) &
		((((IT)1) << ALLOCATOR::INNER_BITS) - 1));
	assert((i >= 0) && (i < ALLOCATOR::INNER_TYPE::SIZE));

	NodeBase *child = inner->elems[i];
	if(child == 0) {
          return false;	
        }
        assert((child != 0) &&
	       (child->level == (n->level - 1)) &&
	       (index >= child->first_index) &&
	       (index <= child->last_index));
	n = child;
      }
      return true;
    }

    template <typename ALLOCATOR>
    typename DynamicTable<ALLOCATOR>::ET *DynamicTable<ALLOCATOR>::lookup_entry(IT index, int owner, typename ALLOCATOR::FreeList *free_list /*= 0*/)
    {
      // first, figure out how many levels the tree must have to find our index
      int level_needed = 0;
      int elems_addressable = 1 << ALLOCATOR::LEAF_BITS;
      while(index >= elems_addressable) {
	level_needed++;
	elems_addressable <<= ALLOCATOR::INNER_BITS;
      }

      // in the common case, we won't need to add levels to the tree - grab the root (no lock)
      // and see if it covers the range that includes our index
      NodeBase *n = root;
      if(!n || (n->level < level_needed)) {
	// root doesn't appear to be high enough - take lock and fix it if it's really
	//  not high enough
	lock.lock();

	if(!root) {
	  // simple case - just create a root node at the level we want
	  root = new_tree_node(level_needed, 0, elems_addressable - 1, owner, free_list);
	} else {
	  // some of the tree already exists - add new layers on top
	  while(root->level < level_needed) {
	    int parent_level = root->level + 1;
	    IT parent_first = 0;
	    IT parent_last = (((root->last_index + 1) << ALLOCATOR::INNER_BITS) - 1);
	    NodeBase *parent = new_tree_node(parent_level, parent_first, parent_last, owner, free_list);
	    typename ALLOCATOR::INNER_TYPE *inner = static_cast<typename ALLOCATOR::INNER_TYPE *>(parent);
	    inner->elems[0] = root;
	    root = parent;
	  }
	}
	n = root;

	lock.unlock();
      }
      // when we get here, root is high enough
      assert((level_needed <= n->level) &&
	     (index >= n->first_index) &&
	     (index <= n->last_index));

      // now walk tree, populating the path we need
      while(n->level > 0) {
	// intermediate nodes
	typename ALLOCATOR::INNER_TYPE *inner = static_cast<typename ALLOCATOR::INNER_TYPE *>(n);

	IT i = ((index >> (ALLOCATOR::LEAF_BITS + (n->level - 1) * ALLOCATOR::INNER_BITS)) &
		((((IT)1) << ALLOCATOR::INNER_BITS) - 1));
	assert((i >= 0) && (i < ALLOCATOR::INNER_TYPE::SIZE));

	NodeBase *child = inner->elems[i];
	if(child == 0) {
	  // need to populate subtree

	  // take lock on inner node
	  inner->lock.lock();

	  // now that lock is held, see if we really need to make new node
	  if(inner->elems[i] == 0) {
	    int child_level = inner->level - 1;
	    int child_shift = (ALLOCATOR::LEAF_BITS + child_level * ALLOCATOR::INNER_BITS);
	    IT child_first = inner->first_index + (i << child_shift);
	    IT child_last = inner->first_index + ((i + 1) << child_shift) - 1;

	    inner->elems[i] = new_tree_node(child_level, child_first, child_last, owner, free_list);
	  }
	  child = inner->elems[i];

	  inner->lock.unlock();
	}
	assert((child != 0) &&
	       (child->level == (n->level - 1)) &&
	       (index >= child->first_index) &&
	       (index <= child->last_index));
	n = child;
      }

      // leaf node - just return pointer to the target element
      typename ALLOCATOR::LEAF_TYPE *leaf = static_cast<typename ALLOCATOR::LEAF_TYPE *>(n);
      int ofs = (index & ((((IT)1) << ALLOCATOR::LEAF_BITS) - 1));
      return &(leaf->elems[ofs]);
    }

    template <typename ALLOCATOR>
    DynamicTableFreeList<ALLOCATOR>::DynamicTableFreeList(DynamicTable<ALLOCATOR>& _table, int _owner)
      : table(_table), owner(_owner), first_free(0), next_alloc(0)
    {
    }

    template <typename ALLOCATOR>
    typename DynamicTableFreeList<ALLOCATOR>::ET *DynamicTableFreeList<ALLOCATOR>::alloc_entry(void)
    {
      // take the lock first, since we're messing with the free list
      lock.lock();

      // if the free list is empty, we can fill it up by referencing the next entry to be allocated -
      // this uses the existing dynamic-filling code to avoid race conditions
      while(!first_free) {
	IT to_lookup = next_alloc;
	next_alloc += ((IT)1) << ALLOCATOR::LEAF_BITS; // do this before letting go of lock
	lock.unlock();
	typename DynamicTable<ALLOCATOR>::ET *dummy = table.lookup_entry(to_lookup, owner, this);

	// can't actually use dummy because we let go of lock - retake lock and hopefully find non-empty
	//  list next time
	lock.lock();
      }

      typename DynamicTable<ALLOCATOR>::ET *entry = first_free;
      first_free = entry->next_free;
      lock.unlock();

      return entry;
    }

    template <typename ALLOCATOR>
    void DynamicTableFreeList<ALLOCATOR>::free_entry(ET *entry)
    {
      // just stick ourselves on front of free list
      lock.lock();
      entry->next_free = first_free;
      first_free = entry;
      lock.unlock();
    }

    template <class T>
    class Atomic {
    public:
      Atomic(T _value) : value(_value)
      {
	gasnet_hsl_init(&mutex);
	//printf("%d: atomic %p = %d\n", gasnet_mynode(), this, value);
      }

      T get(void) const { return (*((volatile T*)(&value))); }

      void decrement(void)
      {
	AutoHSLLock a(mutex);
	//T old_value(value);
	value--;
	//printf("%d: atomic %p %d -> %d\n", gasnet_mynode(), this, old_value, value);
      }

    protected:
      T value;
      gasnet_hsl_t mutex;
    };

    // prioritized list that maintains FIFO order within a priority level
    template <typename T>
    class pri_list : public std::list<T> {
    public:
      void pri_insert(T to_add) {
        // Common case: if the guy on the back has our priority or higher then just
        // put us on the back too.
        if (this->empty() || (this->back()->priority >= to_add->priority))
          this->push_back(to_add);
        else
        {
          // Uncommon case: go through the list until we find someone
          // who has a priority lower than ours.  We know they
          // exist since we saw them on the back.
          bool inserted = false;
          for (typename std::list<T>::iterator it = this->begin();
                it != this->end(); it++)
          {
            if ((*it)->priority < to_add->priority)
            {
              this->insert(it, to_add);
              inserted = true;
              break;
            }
          }
          // Technically we shouldn't need this, but just to be safe
          assert(inserted);
        }
      }
    };
     
    class ID {
    public:
#ifdef LEGION_IDS_ARE_64BIT
      enum {
	TYPE_BITS = 4,
	INDEX_H_BITS = 12,
	INDEX_L_BITS = 32,
	INDEX_BITS = INDEX_H_BITS + INDEX_L_BITS,
	NODE_BITS = 64 - TYPE_BITS - INDEX_BITS /* 16 = 64k nodes */
      };
#else
      // two forms of bit pack for IDs:
      //
      //  3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1
      //  1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
      // +-----+---------------------------------------------------------+
      // |  TYP  |   NODE  |         INDEX                               |
      // |  TYP  |   NODE  |  INDEX_H    |           INDEX_L             |
      // +-----+---------------------------------------------------------+

      enum {
	TYPE_BITS = 4,
	INDEX_H_BITS = 7,
	INDEX_L_BITS = 16,
	INDEX_BITS = INDEX_H_BITS + INDEX_L_BITS,
	NODE_BITS = 32 - TYPE_BITS - INDEX_BITS /* 5 = 32 nodes */
      };
#endif

      enum ID_Types {
	ID_SPECIAL,
	ID_UNUSED_1,
	ID_EVENT,
	ID_UNUSED_3,
	ID_LOCK,
	ID_UNUSED_5,
	ID_MEMORY,
	ID_UNUSED_7,
	ID_PROCESSOR,
	ID_PROCGROUP,
	ID_INDEXSPACE,
	ID_UNUSED_11,
	ID_ALLOCATOR,
	ID_UNUSED_13,
	ID_INSTANCE,
	ID_UNUSED_15,
      };

      enum ID_Specials {
	ID_INVALID = 0,
	ID_GLOBAL_MEM = (1U << INDEX_H_BITS) - 1,
      };

      ID(IDType _value) : value(_value) {}

      template <class T>
      ID(T thing_to_get_id_from) : value(thing_to_get_id_from.id) {}

      ID(ID_Types _type, unsigned _node, IDType _index)
	: value((((IDType)_type) << (NODE_BITS + INDEX_BITS)) |
		(((IDType)_node) << INDEX_BITS) |
		_index) {}

      ID(ID_Types _type, unsigned _node, IDType _index_h, IDType _index_l)
	: value((((IDType)_type) << (NODE_BITS + INDEX_BITS)) |
		(((IDType)_node) << INDEX_BITS) |
		(_index_h << INDEX_L_BITS) |
		_index_l) {}

      IDType id(void) const { return value; }
      ID_Types type(void) const { return (ID_Types)(value >> (NODE_BITS + INDEX_BITS)); }
      unsigned node(void) const { return ((value >> INDEX_BITS) & ((1U << NODE_BITS)-1)); }
      IDType index(void) const { return (value & ((((IDType)1) << INDEX_BITS) - 1)); }
      IDType index_h(void) const { return ((value >> INDEX_L_BITS) & ((((IDType)1) << INDEX_H_BITS)-1)); }
      IDType index_l(void) const { return (value & ((((IDType)1) << INDEX_L_BITS) - 1)); }

      template <class T>
      T convert(void) const { T thing_to_return = { value }; return thing_to_return; }
      
    protected:
      IDType value;
    };
    
    class Event::Impl {
    public:
      Impl(void);

      static const ID::ID_Types ID_TYPE = ID::ID_EVENT;

      void init(Event _me, unsigned _init_owner);

      static Event create_event(void);

      // test whether an event has triggered without waiting
      // make this a volatile method so that if we poll it
      // then it will do the right thing
      bool has_triggered(Event::gen_t needed_gen) volatile;

      // causes calling thread to block until event has occurred
      //void wait(Event::gen_t needed_gen);

      void external_wait(Event::gen_t needed_gen);

      // creates an event that won't trigger until all input events have
      static Event merge_events(const std::set<Event>& wait_for);
      static Event merge_events(Event ev1, Event ev2,
				Event ev3 = NO_EVENT, Event ev4 = NO_EVENT,
				Event ev5 = NO_EVENT, Event ev6 = NO_EVENT);

      // record that the event has triggered and notify anybody who cares
      void trigger(Event::gen_t gen_triggered, int trigger_node, Event wait_on = NO_EVENT);

      // used to adjust a barrier's arrival count either up or down
      // if delta > 0, timestamp is current time (on requesting node)
      // if delta < 0, timestamp says which positive adjustment this arrival must wait for
      void adjust_arrival(Event::gen_t barrier_gen, int delta, 
			  Barrier::timestamp_t timestamp, Event wait_on = NO_EVENT);

      class EventWaiter {
      public:
        EventWaiter(void) { }
        virtual ~EventWaiter(void) { }
      public:
	virtual bool event_triggered(void) = 0;
	virtual void print_info(FILE *f) = 0;
      };

      void add_waiter(Event event, EventWaiter *waiter, bool pre_subscribed = false);

    public: //protected:
      Event me;
      unsigned owner;
      Event::gen_t generation, gen_subscribed, free_generation;
      Event::Impl *next_free;
      //static Event::Impl *first_free;
      //static gasnet_hsl_t freelist_mutex;

      gasnet_hsl_t *mutex; // controls which local thread has access to internal data (not runtime-visible event)

      std::map<Event::gen_t,NodeMask> remote_waiters;
      std::map<Event::gen_t, std::vector<EventWaiter *> > local_waiters; // set of local threads that are waiting on event (keyed by generation)

      // for barriers
      unsigned base_arrival_count, current_arrival_count;

      class PendingUpdates;
      std::map<Event::gen_t, PendingUpdates *> pending_updates;
    };

    struct ElementMaskImpl {
      //int count, offset;
      typedef unsigned long long uint64;
      uint64_t dummy;
      uint64_t bits[0];

      static size_t bytes_needed(off_t offset, off_t count)
      {
	size_t need = sizeof(ElementMaskImpl) + (((count + 63) >> 6) << 3);
	return need;
      }
	
    };

    class Reservation::Impl {
    public:
      Impl(void);

      static const ID::ID_Types ID_TYPE = ID::ID_LOCK;

      void init(Reservation _me, unsigned _init_owner, size_t _data_size = 0);

      template <class T>
      void set_local_data(T *data)
      {
	local_data = data;
	local_data_size = sizeof(T);
        own_local = false;
      }

      //protected:
      Reservation me;
      unsigned owner; // which node owns the lock
      unsigned count; // number of locks held by local threads
      unsigned mode;  // lock mode
      bool in_use;

      enum { MODE_EXCL = 0, ZERO_COUNT = 0x11223344 };

      gasnet_hsl_t *mutex; // controls which local thread has access to internal data (not runtime-visible lock)

      // bitmasks of which remote nodes are waiting on a lock (or sharing it)
      NodeMask remote_waiter_mask, remote_sharer_mask;
      //std::list<LockWaiter *> local_waiters; // set of local threads that are waiting on lock
      std::map<unsigned, std::deque<Event> > local_waiters;
      bool requested; // do we have a request for the lock in flight?

      // local data protected by lock
      void *local_data;
      size_t local_data_size;
      bool own_local;

      static gasnet_hsl_t freelist_mutex;
      static Reservation::Impl *first_free;
      Reservation::Impl *next_free;

      Event acquire(unsigned new_mode, bool exclusive,
		 Event after_lock = Event::NO_EVENT);

      bool select_local_waiters(std::deque<Event>& to_wake);

      void release(void);

      bool is_locked(unsigned check_mode, bool excl_ok);

      void release_reservation(void);
    };

    template <class T>
    class StaticAccess {
    public:
      typedef typename T::StaticData StaticData;

      // if already_valid, just check that data is already valid
      StaticAccess(T* thing_with_data, bool already_valid = false)
	: data(&thing_with_data->locked_data)
      {
	if(already_valid) {
	  assert(data->valid);
	} else {
	  if(!data->valid) {
	    // get a valid copy of the static data by taking and then releasing
	    //  a shared lock
	    thing_with_data->lock.acquire(1, false).wait(true);// TODO: must this be blocking?
	    thing_with_data->lock.release();
	    assert(data->valid);
	  }
	}
      }

      ~StaticAccess(void) {}

      const StaticData *operator->(void) { return data; }

    protected:
      StaticData *data;
    };

    template <class T>
    class SharedAccess {
    public:
      typedef typename T::CoherentData CoherentData;

      // if already_held, just check that it's held (if in debug mode)
      SharedAccess(T* thing_with_data, bool already_held = false)
	: data(&thing_with_data->locked_data), lock(&thing_with_data->lock)
      {
	if(already_held) {
	  assert(lock->is_locked(1, true));
	} else {
	  lock->acquire(1, false).wait();
	}
      }

      ~SharedAccess(void)
      {
	lock->release();
      }

      const CoherentData *operator->(void) { return data; }

    protected:
      CoherentData *data;
      Reservation::Impl *lock;
    };

    template <class T>
    class ExclusiveAccess {
    public:
      typedef typename T::CoherentData CoherentData;

      // if already_held, just check that it's held (if in debug mode)
      ExclusiveAccess(T* thing_with_data, bool already_held = false)
	: data(&thing_with_data->locked_data), lock(&thing_with_data->lock)
      {
	if(already_held) {
	  assert(lock->is_locked(0, true));
	} else {
	  lock->acquire(0, true).wait();
	}
      }

      ~ExclusiveAccess(void)
      {
	lock->release();
      }

      CoherentData *operator->(void) { return data; }

    protected:
      CoherentData *data;
      Reservation::Impl *lock;
    };

    class ProcessorAssignment {
    public:
      ProcessorAssignment(int _num_local_procs);

      // binds a thread to the right set of cores based (-1 = not a local proc)
      void bind_thread(int core_id, pthread_attr_t *attr, const char *debug_name = 0);

    protected:
      // physical configuration of processors
      typedef std::map<int, std::vector<int> > NodeProcMap;
      typedef std::map<int, NodeProcMap> SystemProcMap;

      int num_local_procs;
      bool valid;
      std::vector<int> local_proc_assignments;
      cpu_set_t leftover_procs;
    };
    extern ProcessorAssignment *proc_assignment;

    extern Processor::TaskIDTable task_id_table;

    class UtilityProcessor;

    class ProcessorGroup;

    // information for a task launch
    class Task {
    public:
      Task(Processor _proc,
	   Processor::TaskFuncID _func_id,
	   const void *_args, size_t _arglen,
	   Event _finish_event, int _priority,
           int expected_count);

      virtual ~Task(void);

      Processor proc;
      Processor::TaskFuncID func_id;
      void *args;
      size_t arglen;
      Event finish_event;
      int priority;
      int run_count, finish_count;
    };

    class Processor::Impl {
    public:
      Impl(Processor _me, Processor::Kind _kind, Processor _util = Processor::NO_PROC);

      virtual ~Impl(void);

      void run(Atomic<int> *_run_counter)
      {
	run_counter = _run_counter;
      }

      virtual void tasks_available(int priority) = 0;

      virtual void enqueue_task(Task *task) = 0;

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority) = 0;

      void finished(void)
      {
	if(run_counter)
	  run_counter->decrement();
      }

      void set_utility_processor(UtilityProcessor *_util_proc);

      virtual void enable_idle_task(void) { assert(0); }
      virtual void disable_idle_task(void) { assert(0); }
      virtual bool is_idle_task_enabled(void) { return(false); }

    public:
      Processor me;
      Processor::Kind kind;
      Processor util;
      UtilityProcessor *util_proc;
      Atomic<int> *run_counter;
    }; 

    // generic way of keeping a prioritized queue of stuff to do
    // Needs to be protected by owner lock
    template <typename JOBTYPE>
    class JobQueue {
    public:
      JobQueue(void);

      bool empty(void) const;

      void insert(JOBTYPE *job, int priority);

      JOBTYPE *pop(void);

      std::map<int, std::deque<JOBTYPE*> > ready;
    };

    template <typename JOBTYPE>
    JobQueue<JOBTYPE>::JobQueue(void)
    {
    }

    template<typename JOBTYPES>
    bool JobQueue<JOBTYPES>::empty(void) const
    {
      return ready.empty();
    }

    template <typename JOBTYPE>
    void JobQueue<JOBTYPE>::insert(JOBTYPE *job, int priority)
    {
      std::deque<JOBTYPE *>& dq = ready[-priority];
      dq.push_back(job);
    }

    template <typename JOBTYPE>
    JOBTYPE *JobQueue<JOBTYPE>::pop(void)
    {
      if(ready.empty()) return 0;

      // get the sublist with the highest priority (remember, we negate before lookup)
      typename std::map<int, std::deque<JOBTYPE *> >::iterator it = ready.begin();

      // any deque that's present better be non-empty
      assert(!(it->second.empty()));
      JOBTYPE *job = it->second.front();
      it->second.pop_front();

      // if the list is now empty, remove it and update the new max priority
      if(it->second.empty()) {
	ready.erase(it);
      }

      return job;
    }

    class ProcessorGroup : public Processor::Impl {
    public:
      ProcessorGroup(void);

      virtual ~ProcessorGroup(void);

      static const ID::ID_Types ID_TYPE = ID::ID_PROCGROUP;

      void init(Processor _me, int _owner);

      void set_group_members(const std::vector<Processor>& member_list);

      void get_group_members(std::vector<Processor>& member_list);

      virtual void tasks_available(int priority);

      virtual void enqueue_task(Task *task);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority);

    public: //protected:
      bool members_valid;
      bool members_requested;
      std::vector<Processor::Impl *> members;
      Reservation::Impl lock;
      ProcessorGroup *next_free;

      void request_group_members(void);
    };
    
    class PreemptableThread {
    public:
      PreemptableThread(void) {}
      virtual ~PreemptableThread(void) {}

      void start_thread(size_t stack_size, int core_id, const char *debug_name);

      static bool preemptable_sleep(Event wait_for, bool block = false);

      virtual Processor get_processor(void) const = 0;

    protected:
      static void *thread_entry(void *data);

      virtual void thread_main(void) = 0;

      virtual void sleep_on_event(Event wait_for, bool block = false) = 0;

      pthread_t thread;
    };

    class UtilityProcessor : public Processor::Impl {
    public:
      UtilityProcessor(Processor _me, 
                       int core_id = -1, 
                       int _num_worker_threads = 1);

      virtual ~UtilityProcessor(void);

      void start_worker_threads(size_t stack_size);

      virtual void tasks_available(int priority);
      
      virtual void enqueue_task(Task *task);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority);

      void request_shutdown(void);

      void enable_idle_task(Processor::Impl *proc);
      void disable_idle_task(Processor::Impl *proc);

      void wait_for_shutdown(void);

      class UtilityThread;

    protected:
      //friend class UtilityThread;
      //friend class UtilityTask;

      //void enqueue_runnable_task(Task *task);
      int core_id;
      int num_worker_threads;
      bool shutdown_requested;
      //Event shutdown_event;

      gasnet_hsl_t mutex;
      gasnett_cond_t condvar;

      Task *idle_task;

      std::set<UtilityThread *> threads;

      JobQueue<Task> task_queue;

      std::set<Processor::Impl *> idle_procs;
      std::set<Processor::Impl *> procs_in_idle_task;
    };

    class Memory::Impl {
    public:
      enum MemoryKind {
	MKIND_SYSMEM,  // directly accessible from CPU
	MKIND_GLOBAL,  // accessible via GASnet (spread over all nodes)
	MKIND_RDMA,    // remote, but accessible via RDMA
	MKIND_REMOTE,  // not accessible
#ifdef USE_CUDA
	MKIND_GPUFB,   // GPU framebuffer memory (accessible via cudaMemcpy)
#endif
	MKIND_ZEROCOPY, // CPU memory, pinned for GPU access
      };

    Impl(Memory _me, size_t _size, MemoryKind _kind, size_t _alignment, Kind _lowlevel_kind)
      : me(_me), size(_size), kind(_kind), alignment(_alignment), lowlevel_kind(_lowlevel_kind)
      {
	gasnet_hsl_init(&mutex);
      }

      unsigned add_instance(RegionInstance::Impl *i);

      RegionInstance::Impl *get_instance(RegionInstance i);

      RegionInstance create_instance_local(IndexSpace is,
					   const int *linearization_bits,
					   size_t bytes_needed,
					   size_t block_size,
					   size_t element_size,
					   const std::vector<size_t>& field_sizes,
					   ReductionOpID redopid,
					   off_t list_size,
					   RegionInstance parent_inst);

      RegionInstance create_instance_remote(IndexSpace is,
					    const int *linearization_bits,
					    size_t bytes_needed,
					    size_t block_size,
					    size_t element_size,
					    const std::vector<size_t>& field_sizes,
					    ReductionOpID redopid,
					    off_t list_size,
					    RegionInstance parent_inst);

      virtual RegionInstance create_instance(IndexSpace is,
					     const int *linearization_bits,
					     size_t bytes_needed,
					     size_t block_size,
					     size_t element_size,
					     const std::vector<size_t>& field_sizes,
					     ReductionOpID redopid,
					     off_t list_size,
					     RegionInstance parent_inst) = 0;

      void destroy_instance_local(RegionInstance i, bool local_destroy);
      void destroy_instance_remote(RegionInstance i, bool local_destroy);

      virtual void destroy_instance(RegionInstance i, 
				    bool local_destroy) = 0;

      off_t alloc_bytes_local(size_t size);
      void free_bytes_local(off_t offset, size_t size);

      off_t alloc_bytes_remote(size_t size);
      void free_bytes_remote(off_t offset, size_t size);

      virtual off_t alloc_bytes(size_t size) = 0;
      virtual void free_bytes(off_t offset, size_t size) = 0;

      virtual void get_bytes(off_t offset, void *dst, size_t size) = 0;
      virtual void put_bytes(off_t offset, const void *src, size_t size) = 0;

      virtual void apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
					size_t count, const void *entry_buffer)
      {
	assert(0);
      }

      virtual void *get_direct_ptr(off_t offset, size_t size) = 0;
      virtual int get_home_node(off_t offset, size_t size) = 0;

      Memory::Kind get_kind(void) const;

    public:
      Memory me;
      size_t size;
      MemoryKind kind;
      size_t alignment;
      Kind lowlevel_kind;
      gasnet_hsl_t mutex; // protection for resizing vectors
      std::vector<RegionInstance::Impl *> instances;
      std::map<off_t, off_t> free_blocks;
    };

    class GASNetMemory : public Memory::Impl {
    public:
      static const size_t MEMORY_STRIDE = 1024;

      GASNetMemory(Memory _me, size_t size_per_node);

      virtual ~GASNetMemory(void);

      virtual RegionInstance create_instance(IndexSpace is,
					     const int *linearization_bits,
					     size_t bytes_needed,
					     size_t block_size,
					     size_t element_size,
					     const std::vector<size_t>& field_sizes,
					     ReductionOpID redopid,
					     off_t list_size,
					     RegionInstance parent_inst);

      virtual void destroy_instance(RegionInstance i, 
				    bool local_destroy);

      virtual off_t alloc_bytes(size_t size);

      virtual void free_bytes(off_t offset, size_t size);

      virtual void get_bytes(off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
					size_t count, const void *entry_buffer);

      virtual void *get_direct_ptr(off_t offset, size_t size);
      virtual int get_home_node(off_t offset, size_t size);

      void get_batch(size_t batch_size,
		     const off_t *offsets, void * const *dsts, 
		     const size_t *sizes);

      void put_batch(size_t batch_size,
		     const off_t *offsets, const void * const *srcs, 
		     const size_t *sizes);

    protected:
      int num_nodes;
      off_t memory_stride;
      gasnet_seginfo_t *seginfos;
      //std::map<off_t, off_t> free_blocks;
    };

    class RegionInstance::Impl {
    public:
      Impl(RegionInstance _me, IndexSpace _is, Memory _memory, off_t _offset, size_t _size, ReductionOpID _redopid,
	   const DomainLinearization& _linear, size_t _block_size, size_t _elmt_size, const std::vector<size_t>& _field_sizes,
	   off_t _count_offset = -1, off_t _red_list_size = -1, RegionInstance _parent_inst = NO_INST);

      // when we auto-create a remote instance, we don't know region/offset/linearization
      Impl(RegionInstance _me, Memory _memory);

      ~Impl(void);

#ifdef POINTER_CHECKS
      void verify_access(unsigned ptr);
      const ElementMask& get_element_mask(void);
#endif
      void get_bytes(int index, off_t byte_offset, void *dst, size_t size);
      void put_bytes(int index, off_t byte_offset, const void *src, size_t size);

#if 0
      static Event copy(RegionInstance src, 
			RegionInstance target,
			IndexSpace isegion,
			size_t elmt_size,
			size_t bytes_to_copy,
			Event after_copy = Event::NO_EVENT);
#endif

      bool get_strided_parameters(void *&base, size_t &stride,
				  off_t field_offset);

    public: //protected:
      friend class RegionInstance;

      RegionInstance me;
      Memory memory;
      DomainLinearization linearization;

      static const unsigned MAX_FIELDS_PER_INST = 2048;
      static const unsigned MAX_LINEARIZATION_LEN = 16;

      struct StaticData {
	IndexSpace is;
	off_t alloc_offset; //, access_offset;
	size_t size;
	//size_t first_elmt, last_elmt;
	//bool is_reduction;
	ReductionOpID redopid;
	off_t count_offset;
	off_t red_list_size;
	size_t block_size, elmt_size;
	int field_sizes[MAX_FIELDS_PER_INST];
	RegionInstance parent_inst;
	int linearization_bits[MAX_LINEARIZATION_LEN];
        // This had better damn well be the last field
        // in the struct in order to avoid race conditions!
	bool valid;
      } locked_data;

      Reservation::Impl lock;
    };

    class IndexSpace::Impl {
    public:
      Impl(void);
      ~Impl(void);

      void init(IndexSpace _me, unsigned _init_owner);

      void init(IndexSpace _me, IndexSpace _parent,
		size_t _num_elmts,
		const ElementMask *_initial_valid_mask = 0, bool _frozen = false);

      static const ID::ID_Types ID_TYPE = ID::ID_INDEXSPACE;

      bool is_parent_of(IndexSpace other);

      size_t instance_size(const ReductionOpUntyped *redop = 0,
			   off_t list_size = -1);

      off_t instance_adjust(const ReductionOpUntyped *redop = 0);

      Event request_valid_mask(void);

      IndexSpace me;
      Reservation::Impl lock;
      IndexSpace::Impl *next_free;

      struct StaticData {
	IndexSpace parent;
	bool frozen;
	size_t num_elmts;
        size_t first_elmt, last_elmt;
        // This had better damn well be the last field
        // in the struct in order to avoid race conditions!
	bool valid;
      };
      struct CoherentData : public StaticData {
	unsigned valid_mask_owners;
	int avail_mask_owner;
      };

      CoherentData locked_data;
      gasnet_hsl_t valid_mask_mutex;
      ElementMask *valid_mask;
      int valid_mask_count;
      bool valid_mask_complete;
      Event valid_mask_event;
      int valid_mask_first, valid_mask_last;
      bool valid_mask_contig;
      ElementMask *avail_mask;
    };

    template <typename _ET, size_t _INNER_BITS, size_t _LEAF_BITS>
    class DynamicTableAllocator {
    public:
      typedef _ET ET;
      static const size_t INNER_BITS = _INNER_BITS;
      static const size_t LEAF_BITS = _LEAF_BITS;

      typedef GASNetHSL LT;
      typedef int IT;
      typedef DynamicTableNode<DynamicTableNodeBase<LT, IT> *, 1 << INNER_BITS, LT, IT> INNER_TYPE;
      typedef DynamicTableNode<ET, 1 << LEAF_BITS, LT, IT> LEAF_TYPE;
      typedef DynamicTableFreeList<DynamicTableAllocator<ET, _INNER_BITS, _LEAF_BITS> > FreeList;
      
      static LEAF_TYPE *new_leaf_node(IT first_index, IT last_index, 
				      int owner, FreeList *free_list)
      {
	LEAF_TYPE *leaf = new LEAF_TYPE(0, first_index, last_index);
	IT last_ofs = (((IT)1) << LEAF_BITS) - 1;
	for(IT i = 0; i <= last_ofs; i++)
	  leaf->elems[i].init(ID(ET::ID_TYPE, owner, first_index + i).convert<typeof(leaf->elems[0].me)>(), owner);

	if(free_list) {
	  // stitch all the new elements into the free list
	  free_list->lock.lock();

	  for(IT i = 0; i <= last_ofs; i++)
	    leaf->elems[i].next_free = ((i < last_ofs) ? 
					  &(leaf->elems[i+1]) :
					  free_list->first_free);

	  free_list->first_free = &(leaf->elems[first_index ? 0 : 1]);

	  free_list->lock.unlock();
	}

	return leaf;
      }
    };

    typedef DynamicTableAllocator<Event::Impl, 10, 8> EventTableAllocator;
    typedef DynamicTableAllocator<Reservation::Impl, 10, 8> ReservationTableAllocator;
    typedef DynamicTableAllocator<IndexSpace::Impl, 10, 4> IndexSpaceTableAllocator;
    typedef DynamicTableAllocator<ProcessorGroup, 10, 4> ProcessorGroupTableAllocator;

    // for each of the ID-based runtime objects, we're going to have an
    //  implementation class and a table to look them up in
    struct Node {
      Node(void);

      // not currently resizable
      std::vector<Memory::Impl *> memories;
      std::vector<Processor::Impl *> processors;

      DynamicTable<EventTableAllocator> events;
      DynamicTable<ReservationTableAllocator> reservations;
      DynamicTable<IndexSpaceTableAllocator> index_spaces;
      DynamicTable<ProcessorGroupTableAllocator> proc_groups;
    };

    class Runtime {
    public:
      static Runtime *get_runtime(void) { return runtime; }

      Event::Impl *get_event_impl(ID id);
      Reservation::Impl *get_lock_impl(ID id);
      Memory::Impl *get_memory_impl(ID id);
      Processor::Impl *get_processor_impl(ID id);
      ProcessorGroup *get_procgroup_impl(ID id);
      IndexSpace::Impl *get_index_space_impl(ID id);
      RegionInstance::Impl *get_instance_impl(ID id);
#ifdef DEADLOCK_TRACE
      void add_thread(const pthread_t *thread);
#endif

    protected:
    public:
      static Runtime *runtime;

      Node *nodes;
      Memory::Impl *global_memory;
      EventTableAllocator::FreeList *local_event_free_list;
      ReservationTableAllocator::FreeList *local_reservation_free_list;
      IndexSpaceTableAllocator::FreeList *local_index_space_free_list;
      ProcessorGroupTableAllocator::FreeList *local_proc_group_free_list;

#ifdef DEADLOCK_TRACE
      unsigned next_thread;
      pthread_t all_threads[MAX_NUM_THREADS];
      unsigned thread_counts[MAX_NUM_THREADS];
#endif
    };

  }; // namespace LowLevel
}; // namespace LegionRuntime

#endif
