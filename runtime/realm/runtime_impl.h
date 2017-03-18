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

// Runtime implementation for Realm

#ifndef REALM_RUNTIME_IMPL_H
#define REALM_RUNTIME_IMPL_H

#include "runtime.h"
#include "id.h"

#include "activemsg.h"
#include "operation.h"
#include "profiling.h"

#include "dynamic_table.h"
#include "proc_impl.h"

// event and reservation impls are included directly in the node's dynamic tables,
//  so we need their definitions here (not just declarations)
#include "event_impl.h"
#include "rsrv_impl.h"

#include "machine_impl.h"

#include "threads.h"
#include "sampling.h"

#include "module.h"

#if __cplusplus >= 201103L
#define typeof decltype
#endif

// dma channels are still in old namespace
namespace LegionRuntime {
  namespace LowLevel {
    class MemPairCopierFactory;
  };
};

namespace Realm {

  class IndexSpaceImpl;
  class ProcessorGroup;
  class MemoryImpl;
  class ProcessorImpl;
  class RegionInstanceImpl;
  class Module;

  typedef LegionRuntime::LowLevel::MemPairCopierFactory DMAChannel;

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

      // hack for now - these should be factored out
      static ID make_id(const GenEventImpl& dummy, int owner, int index) { return ID::make_event(owner, index, 0); }
      static ID make_id(const BarrierImpl& dummy, int owner, int index) { return ID::make_barrier(owner, index, 0); }
      static Reservation make_id(const ReservationImpl& dummy, int owner, int index) { return ID::make_reservation(owner, index).convert<Reservation>(); }
      static Processor make_id(const ProcessorGroup& dummy, int owner, int index) { return ID::make_procgroup(owner, 0, index).convert<Processor>(); }
      static IndexSpace make_id(const IndexSpaceImpl& dummy, int owner, int index) { return ID::make_idxspace(owner, 0, index).convert<IndexSpace>(); }
      
      static LEAF_TYPE *new_leaf_node(IT first_index, IT last_index, 
				      int owner, FreeList *free_list)
      {
	LEAF_TYPE *leaf = new LEAF_TYPE(0, first_index, last_index);
	IT last_ofs = (((IT)1) << LEAF_BITS) - 1;
	for(IT i = 0; i <= last_ofs; i++)
	  leaf->elems[i].init(make_id(leaf->elems[0], owner, first_index + i), owner);
	  //leaf->elems[i].init(ID(ET::ID_TYPE, owner, first_index + i).convert<typeof(leaf->elems[0].me)>(), owner);

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

    typedef DynamicTableAllocator<GenEventImpl, 10, 8> EventTableAllocator;
    typedef DynamicTableAllocator<BarrierImpl, 10, 4> BarrierTableAllocator;
    typedef DynamicTableAllocator<ReservationImpl, 10, 8> ReservationTableAllocator;
    typedef DynamicTableAllocator<IndexSpaceImpl, 10, 4> IndexSpaceTableAllocator;
    typedef DynamicTableAllocator<ProcessorGroup, 10, 4> ProcessorGroupTableAllocator;

    // for each of the ID-based runtime objects, we're going to have an
    //  implementation class and a table to look them up in
    struct Node {
      Node(void);

      // not currently resizable
      std::vector<MemoryImpl *> memories;
      std::vector<MemoryImpl *> ib_memories;
      std::vector<ProcessorImpl *> processors;

      DynamicTable<EventTableAllocator> events;
      DynamicTable<BarrierTableAllocator> barriers;
      DynamicTable<ReservationTableAllocator> reservations;
      DynamicTable<IndexSpaceTableAllocator> index_spaces;
      DynamicTable<ProcessorGroupTableAllocator> proc_groups;
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
      size_t sysmem_size_in_mb, stack_size_in_mb;
    };

    REGISTER_REALM_MODULE(CoreModule);

    class RuntimeImpl {
    public:
      RuntimeImpl(void);
      ~RuntimeImpl(void);

      bool init(int *argc, char ***argv);

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

      // requests a shutdown of the runtime
      void shutdown(bool local_request);

      void wait_for_shutdown(void);

      // three event-related impl calls - get_event_impl() will give you either
      //  a normal event or a barrier, but you won't be able to do specific things
      //  (e.g. trigger a GenEventImpl or adjust a BarrierImpl)
      EventImpl *get_event_impl(Event e);
      GenEventImpl *get_genevent_impl(Event e);
      BarrierImpl *get_barrier_impl(Event e);

      ReservationImpl *get_lock_impl(ID id);
      MemoryImpl *get_memory_impl(ID id);
      ProcessorImpl *get_processor_impl(ID id);
      ProcessorGroup *get_procgroup_impl(ID id);
      IndexSpaceImpl *get_index_space_impl(ID id);
      RegionInstanceImpl *get_instance_impl(ID id);
#ifdef DEADLOCK_TRACE
      void add_thread(const pthread_t *thread);
#endif
      static void realm_backtrace(int signal);

    public:
      MachineImpl *machine;

      std::map<ReductionOpID, const ReductionOpUntyped *> reduce_op_table;
      std::map<CustomSerdezID, const CustomSerdezUntyped *> custom_serdez_table;

#ifdef NODE_LOGGING
      std::string prefix;
#endif

      Node *nodes;
      MemoryImpl *global_memory;
      EventTableAllocator::FreeList *local_event_free_list;
      BarrierTableAllocator::FreeList *local_barrier_free_list;
      ReservationTableAllocator::FreeList *local_reservation_free_list;
      IndexSpaceTableAllocator::FreeList *local_index_space_free_list;
      ProcessorGroupTableAllocator::FreeList *local_proc_group_free_list;

      // legacy behavior if Runtime::run() is used
      bool run_method_called;
#ifdef DEADLOCK_TRACE
      unsigned next_thread;
      unsigned signaled_threads;
      pthread_t all_threads[MAX_NUM_THREADS];
      unsigned thread_counts[MAX_NUM_THREADS];
#endif
      volatile bool shutdown_requested;
      GASNetHSL shutdown_mutex;
      GASNetCondVar shutdown_condvar;

      CoreMap *core_map;
      CoreReservationSet *core_reservations;

      OperationTable optable;

      SamplingProfiler sampling_profiler;

    public:
      // used by modules to add processors, memories, etc.
      void add_memory(MemoryImpl *m);
      void add_ib_memory(MemoryImpl *m);
      void add_processor(ProcessorImpl *p);
      void add_dma_channel(DMAChannel *c);
      void add_code_translator(CodeTranslator *t);

      void add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity& pma);
      void add_mem_mem_affinity(const Machine::MemoryMemoryAffinity& mma);

      Memory next_local_memory_id(void);
      Memory next_local_ib_memory_id(void);
      Processor next_local_processor_id(void);
      CoreReservationSet& core_reservation_set(void);

      const std::vector<DMAChannel *>& get_dma_channels(void) const;

      const std::vector<CodeTranslator *>& get_code_translators(void) const;

    protected:
      ID::IDType num_local_memories, num_local_ib_memories, num_local_processors;

      ModuleRegistrar module_registrar;
      std::vector<Module *> modules;
      std::vector<DMAChannel *> dma_channels;
      std::vector<CodeTranslator *> code_translators;
    };

    extern RuntimeImpl *runtime_singleton;
    inline RuntimeImpl *get_runtime(void) { return runtime_singleton; }

    // due to circular dependencies in include files, we need versions of these that
    //  hide the RuntimeImpl intermediate
    inline EventImpl *get_event_impl(Event e) { return get_runtime()->get_event_impl(e); }
    inline GenEventImpl *get_genevent_impl(Event e) { return get_runtime()->get_genevent_impl(e); }
    inline BarrierImpl *get_barrier_impl(Event e) { return get_runtime()->get_barrier_impl(e); }

    // active messages

    struct RuntimeShutdownMessage {
      struct RequestArgs {
	int initiating_node;
	int dummy; // needed to get sizeof() >= 8
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<MACHINE_SHUTDOWN_MSGID,
				        RequestArgs,
				        handle_request> Message;

      static void send_request(gasnet_node_t target);
    };
      
}; // namespace Realm

#endif // ifndef REALM_RUNTIME_IMPL_H
