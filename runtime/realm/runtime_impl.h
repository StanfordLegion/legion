/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// Runtime implementation for Realm

#ifndef REALM_RUNTIME_IMPL_H
#define REALM_RUNTIME_IMPL_H

#include "realm/runtime.h"
#include "realm/id.h"

#include "realm/network.h"
#include "realm/operation.h"
#include "realm/profiling.h"

#include "realm/dynamic_table.h"
#include "realm/proc_impl.h"
#include "realm/deppart/partitions.h"

// event and reservation impls are included directly in the node's dynamic tables,
//  so we need their definitions here (not just declarations)
#include "realm/event_impl.h"
#include "realm/rsrv_impl.h"
#include "realm/subgraph_impl.h"

#include "realm/machine_impl.h"

#include "realm/threads.h"
#include "realm/sampling.h"

#include "realm/module.h"
#include "realm/network.h"

#include "realm/bgwork.h"
#include "realm/activemsg.h"

namespace Realm {

  class ProcessorGroupImpl;
  class MemoryImpl;
  class IBMemory;
  class ProcessorImpl;
  class RegionInstanceImpl;
  class NetworkSegment;

  class Channel; // from transfer/channel.h

    template <typename _ET, size_t _INNER_BITS, size_t _LEAF_BITS>
    class DynamicTableAllocator {
    public:
      typedef _ET ET;
      static const size_t INNER_BITS = _INNER_BITS;
      static const size_t LEAF_BITS = _LEAF_BITS;

      typedef Mutex LT;
      typedef ID::IDType IT;
      typedef DynamicTableNode<atomic<DynamicTableNodeBase<LT, IT> *>, 1 << INNER_BITS, LT, IT> INNER_TYPE;
      typedef DynamicTableNode<ET, 1 << LEAF_BITS, LT, IT> LEAF_TYPE;
      typedef DynamicTableFreeList<DynamicTableAllocator<ET, _INNER_BITS, _LEAF_BITS> > FreeList;

      // hack for now - these should be factored out
      static ID make_id(const GenEventImpl& dummy, int owner, IT index) { return ID::make_event(owner, index, 0); }
      static ID make_id(const BarrierImpl& dummy, int owner, IT index) { return ID::make_barrier(owner, index, 0); }
      static Reservation make_id(const ReservationImpl& dummy, int owner, IT index) { return ID::make_reservation(owner, index).convert<Reservation>(); }
      static Processor make_id(const ProcessorGroupImpl& dummy, int owner, IT index) { return ID::make_procgroup(owner, 0, index).convert<Processor>(); }
      static ID make_id(const SparsityMapImplWrapper& dummy, int owner, IT index) { return ID::make_sparsity(owner, 0, index); }
      static CompletionQueue make_id(const CompQueueImpl& dummy, int owner, IT index) { return ID::make_compqueue(owner, index).convert<CompletionQueue>(); }
      static ID make_id(const SubgraphImpl& dummy, int owner, IT index) { return ID::make_subgraph(owner, 0, index); }
      
      static LEAF_TYPE *new_leaf_node(IT first_index, IT last_index, 
				      int owner, FreeList *free_list)
      {
	LEAF_TYPE *leaf = new LEAF_TYPE(0, first_index, last_index);
	IT last_ofs = (((IT)1) << LEAF_BITS) - 1;
	for(IT i = 0; i <= last_ofs; i++)
	  leaf->elems[i].init(make_id(leaf->elems[0], owner, first_index + i), owner);
	  //leaf->elems[i].init(ID(ET::ID_TYPE, owner, first_index + i).convert<typeof(leaf->elems[0].me)>(), owner);

	if(free_list) {
	  // stitch all the new elements into the free list - we can do this
          //  with a single cmpxchg if we link up all of our new entries
          //  first

          // special case - if the first_index == 0, don't actually enqueue
          //  our first element, which would be global index 0
          IT first_ofs = ((first_index > 0) ? 0 : 1);
          ET *new_head = &leaf->elems[first_ofs];
          ET **tailp = &(leaf->elems[last_ofs].next_free);

          for(IT i = first_ofs; i < last_ofs; i++)
            leaf->elems[i].next_free = &leaf->elems[i+1];

          ET *old_head = free_list->first_free.load_acquire();
          while(true) {
            *tailp = old_head;
            if(free_list->first_free.compare_exchange(old_head, new_head))
              break;
          }
        }

	return leaf;
      }
    };

    // use a wide tree for local events - max depth will be 2
    typedef DynamicTableAllocator<GenEventImpl, 11, 16> LocalEventTableAllocator;
    // use a narrow tree for remote events - depth is 3, leaves have 128 events
    typedef DynamicTableAllocator<GenEventImpl, 10, 7> RemoteEventTableAllocator;
    typedef DynamicTableAllocator<BarrierImpl, 10, 4> BarrierTableAllocator;
    typedef DynamicTableAllocator<ReservationImpl, 10, 8> ReservationTableAllocator;
    typedef DynamicTableAllocator<ProcessorGroupImpl, 10, 4> ProcessorGroupTableAllocator;
    typedef DynamicTableAllocator<SparsityMapImplWrapper, 10, 4> SparsityMapTableAllocator;
    typedef DynamicTableAllocator<CompQueueImpl, 10, 4> CompQueueTableAllocator;
    typedef DynamicTableAllocator<SubgraphImpl, 10, 4> SubgraphTableAllocator;

    // for each of the ID-based runtime objects, we're going to have an
    //  implementation class and a table to look them up in
    struct Node {
      Node(void);

      // not currently resizable
      std::vector<MemoryImpl *> memories;
      std::vector<IBMemory *> ib_memories;
      std::vector<ProcessorImpl *> processors;
      std::vector<Channel *> dma_channels;

      DynamicTable<RemoteEventTableAllocator> remote_events;
      DynamicTable<BarrierTableAllocator> barriers;
      DynamicTable<ReservationTableAllocator> reservations;
      DynamicTable<CompQueueTableAllocator> compqueues;

      // sparsity maps can be created by other nodes, so keep a
      //  map per-creator_node
      std::vector<atomic<DynamicTable<SparsityMapTableAllocator> *> > sparsity_maps;
      std::vector<atomic<DynamicTable<SubgraphTableAllocator> *> > subgraphs;
      std::vector<atomic<DynamicTable<ProcessorGroupTableAllocator> *> > proc_groups;
    };

    // the "core" module provides the basic memories and processors used by Realm
    class CoreModule : public Module {
    public:
      CoreModule(void);
      virtual ~CoreModule(void);

      static Module *create_module(RuntimeImpl *runtime, std::vector<std::string>& cmdline);

      // create any memories provided by this module (default == do nothing)
      //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
      virtual void create_memories(RuntimeImpl *runtime);

      // create any processors provided by the module (default == do nothing)
      //  (each new ProcessorImpl should use a Processor from
      //   RuntimeImpl::next_local_processor_id)
      virtual void create_processors(RuntimeImpl *runtime);

      // create any DMA channels provided by the module (default == do nothing)
      virtual void create_dma_channels(RuntimeImpl *runtime);

      // create any code translators provided by the module (default == do nothing)
      virtual void create_code_translators(RuntimeImpl *runtime);

      // clean up any common resources created by the module - this will be called
      //  after all memories/processors/etc. have been shut down and destroyed
      virtual void cleanup(void);

    protected:
      int num_cpu_procs, num_util_procs, num_io_procs;
      int concurrent_io_threads;
      size_t sysmem_size, stack_size;
      bool pin_util_procs;
      long long cpu_bgwork_timeslice, util_bgwork_timeslice;
      bool use_ext_sysmem;

    public:
      MemoryImpl *ext_sysmem;
    };

    template <typename K, typename V, typename LT = Mutex>
    class LockedMap {
    public:
      bool exists(const K& key) const
      {
	AutoLock<LT> al(mutex);
	typename std::map<K, V>::const_iterator it = map.find(key);
	return (it != map.end());
      }

      bool put(const K& key, const V& value, bool replace = false)
      {
	AutoLock<LT> al(mutex);
	typename std::map<K, V>::iterator it = map.find(key);
	if(it != map.end()) {
	  if(replace) it->second = value;
	  return true;
	} else {
	  map.insert(std::make_pair(key, value));
	  return false;
	}
      }

      V get(const K& key, const V& defval) const
      {
	AutoLock<LT> al(mutex);
	typename std::map<K, V>::const_iterator it = map.find(key);
	if(it != map.end())
	  return it->second;
	else
	  return defval;
      }

    //protected:
      mutable LT mutex;
      std::map<K, V> map;
    };

    class RuntimeImpl {
    public:
      RuntimeImpl(void);
      ~RuntimeImpl(void);

      bool network_init(int *argc, char ***argv);

      bool configure_from_command_line(std::vector<std::string> &cmdline);

      void start(void);

      bool register_task(Processor::TaskFuncID taskid, Processor::TaskFuncPtr taskptr);
      bool register_reduction(ReductionOpID redop_id, const ReductionOpUntyped *redop);
      bool register_custom_serdez(CustomSerdezID serdez_id, const CustomSerdezUntyped *serdez);

      Event collective_spawn(Processor target_proc, Processor::TaskFuncID task_id, 
			     const void *args, size_t arglen,
			     Event wait_on = Event::NO_EVENT, int priority = 0);

      Event collective_spawn_by_kind(Processor::Kind target_kind, Processor::TaskFuncID task_id, 
				     const void *args, size_t arglen,
				     bool one_per_node = false,
				     Event wait_on = Event::NO_EVENT, int priority = 0);

      void run(Processor::TaskFuncID task_id = 0, 
	       Runtime::RunStyle style = Runtime::ONE_TASK_ONLY,
	       const void *args = 0, size_t arglen = 0, bool background = false);

      // requests a shutdown of the runtime - returns true if request is a duplicate
      bool request_shutdown(Event wait_on, int result_code);

      // indicates shutdown has been initiated, wakes up a waiter if already present
      void initiate_shutdown(void);

      // returns value of result_code passed to shutdown()
      int wait_for_shutdown(void);

      // three event-related impl calls - get_event_impl() will give you either
      //  a normal event or a barrier, but you won't be able to do specific things
      //  (e.g. trigger a GenEventImpl or adjust a BarrierImpl)
      EventImpl *get_event_impl(Event e);
      GenEventImpl *get_genevent_impl(Event e);
      BarrierImpl *get_barrier_impl(Event e);

      ReservationImpl *get_lock_impl(ID id);
      MemoryImpl *get_memory_impl(ID id);
      IBMemory *get_ib_memory_impl(ID id);
      ProcessorImpl *get_processor_impl(ID id);
      ProcessorGroupImpl *get_procgroup_impl(ID id);
      RegionInstanceImpl *get_instance_impl(ID id);
      SparsityMapImplWrapper *get_sparsity_impl(ID id);
      SparsityMapImplWrapper *get_available_sparsity_impl(NodeID target_node);
      CompQueueImpl *get_compqueue_impl(ID id);
      SubgraphImpl *get_subgraph_impl(ID id);

#ifdef DEADLOCK_TRACE
      void add_thread(const pthread_t *thread);
#endif
      static void realm_backtrace(int signal);

    public:
      MachineImpl *machine;

      LockedMap<ReductionOpID, ReductionOpUntyped *> reduce_op_table;
      LockedMap<CustomSerdezID, CustomSerdezUntyped *> custom_serdez_table;

#ifdef NODE_LOGGING
      std::string prefix;
#endif

      Node *nodes;
      DynamicTable<LocalEventTableAllocator> local_events;
      LocalEventTableAllocator::FreeList *local_event_free_list;
      BarrierTableAllocator::FreeList *local_barrier_free_list;
      ReservationTableAllocator::FreeList *local_reservation_free_list;
      CompQueueTableAllocator::FreeList *local_compqueue_free_list;

      // keep a free list for each node we allocate maps on (i.e. indexed
      //   by owner_node)
      std::vector<SparsityMapTableAllocator::FreeList *> local_sparsity_map_free_lists;
      std::vector<SubgraphTableAllocator::FreeList *> local_subgraph_free_lists;
      std::vector<ProcessorGroupTableAllocator::FreeList *> local_proc_group_free_lists;

      // legacy behavior if Runtime::run() is used
      bool run_method_called;
#ifdef DEADLOCK_TRACE
      unsigned next_thread;
      unsigned signaled_threads;
      pthread_t all_threads[MAX_NUM_THREADS];
      unsigned thread_counts[MAX_NUM_THREADS];
#endif
      Mutex shutdown_mutex;
      Mutex::CondVar shutdown_condvar;
      bool shutdown_request_received;  // has a request for shutdown arrived
      Event shutdown_precondition;
      int shutdown_result_code;
      bool shutdown_initiated;  // is it time to start shutting down
      atomic<bool> shutdown_in_progress; // are we actively shutting down?

      CoreMap *core_map;
      CoreReservationSet *core_reservations;
      BackgroundWorkManager bgwork;
      IncomingMessageManager *message_manager;
      EventTriggerNotifier event_triggerer;

      OperationTable optable;

      SamplingProfiler sampling_profiler;

      class DeferredShutdown : public EventWaiter {
      public:
	void defer(RuntimeImpl *_runtime, Event wait_on);

	virtual void event_triggered(bool poisoned, TimeLimit work_until);
	virtual void print(std::ostream& os) const;
	virtual Event get_finish_event(void) const;

      protected:
	RuntimeImpl *runtime;
      };
      DeferredShutdown deferred_shutdown;
      
    public:
      // used by modules to add processors, memories, etc.
      void add_memory(MemoryImpl *m);
      void add_ib_memory(IBMemory *m);
      void add_processor(ProcessorImpl *p);
      void add_dma_channel(Channel *c);
      void add_code_translator(CodeTranslator *t);

      void add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity& pma);
      void add_mem_mem_affinity(const Machine::MemoryMemoryAffinity& mma);

      Memory next_local_memory_id(void);
      Memory next_local_ib_memory_id(void);
      Processor next_local_processor_id(void);
      CoreReservationSet& core_reservation_set(void);

      const std::vector<CodeTranslator *>& get_code_translators(void) const;

      template <typename T>
      T *get_module(const char *name) const
      {
        Module *mod = get_module_untyped(name);
        if(mod)
          return checked_cast<T *>(mod);
        else
          return 0;
      }

    protected:
      Module *get_module_untyped(const char *name) const;

      ID::IDType num_local_memories, num_local_ib_memories, num_local_processors;
      NetworkSegment reg_ib_mem_segment;
      NetworkSegment reg_mem_segment;

      ModuleRegistrar module_registrar;
      bool modules_created;
      std::vector<Module *> modules;
      std::vector<CodeTranslator *> code_translators;

      std::vector<NetworkModule *> network_modules;
      std::vector<NetworkSegment *> network_segments;
    };

    extern RuntimeImpl *runtime_singleton;
    inline RuntimeImpl *get_runtime(void) { return runtime_singleton; }

    // due to circular dependencies in include files, we need versions of these that
    //  hide the RuntimeImpl intermediate
    inline EventImpl *get_event_impl(Event e) { return get_runtime()->get_event_impl(e); }
    inline GenEventImpl *get_genevent_impl(Event e) { return get_runtime()->get_genevent_impl(e); }
    inline BarrierImpl *get_barrier_impl(Event e) { return get_runtime()->get_barrier_impl(e); }

    // active messages

    struct RuntimeShutdownRequest {
      Event wait_on;
      int result_code;

      static void handle_message(NodeID sender,const RuntimeShutdownRequest &msg,
				 const void *data, size_t datalen);
    };
      
    struct RuntimeShutdownMessage {
      int result_code;

      static void handle_message(NodeID sender,const RuntimeShutdownMessage &msg,
				 const void *data, size_t datalen);
    };
      
}; // namespace Realm

#endif // ifndef REALM_RUNTIME_IMPL_H
