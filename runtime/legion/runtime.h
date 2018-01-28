/* Copyright 2018 Stanford University, NVIDIA Corporation
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


#ifndef __RUNTIME_H__
#define __RUNTIME_H__

#include "legion.h"
#include "legion/legion_spy.h"
#include "legion/region_tree.h"
#include "legion/mapper_manager.h"
#include "legion/legion_utilities.h"
#include "legion/legion_profiling.h"
#include "legion/legion_allocation.h"
#include "legion/garbage_collection.h"

#define REPORT_LEGION_FATAL(code, fmt, ...)               \
{                                                         \
char message[4096];                                       \
snprintf(message, 4096, fmt, ##__VA_ARGS__);              \
Legion::Internal::Runtime::report_fatal_message(          \
code, __FILE__, __LINE__, message);                       \
}

#define REPORT_LEGION_ERROR(code, fmt, ...)               \
{                                                         \
char message[4096];                                       \
snprintf(message, 4096, fmt, ##__VA_ARGS__);              \
Legion::Internal::Runtime::report_error_message(          \
code, __FILE__, __LINE__, message);                       \
}

#define REPORT_LEGION_WARNING(code, fmt, ...)             \
{                                                         \
char message[4096];                                       \
snprintf(message, 4096, fmt, ##__VA_ARGS__);              \
Legion::Internal::Runtime::report_warning_message(        \
code, __FILE__, __LINE__, message);                       \
}

namespace Legion {
#ifndef DISABLE_PARTITION_SHIM
#define PARTITION_SHIM_MAPPER_ID                      (1729)
  // An internal namespace with some classes for providing help
  // with backwards compatibility for partitioning operations
  namespace PartitionShim {

    template<int COLOR_DIM>
    class ColorPoints : public TaskLauncher {
    public:
      ColorPoints(const Coloring &coloring, LogicalRegion region,
                  FieldID color_field, FieldID pointer_field);
      ColorPoints(const PointColoring &coloring, LogicalRegion region,
                  FieldID color_field, FieldID pointer_field);
    protected:
      Serializer rez;
    public:
      static TaskID TASK_ID;
      static void register_task(void);
      static void cpu_variant(const Task *task,
          const std::vector<PhysicalRegion> &regions, 
          Context ctx, Runtime *runtime);
    };

    template<int COLOR_DIM, int RANGE_DIM>
    class ColorRects : public TaskLauncher {
    public:
      ColorRects(const DomainColoring &coloring, LogicalRegion region,
                 FieldID color_field, FieldID range_field);
      ColorRects(const MultiDomainColoring &coloring, LogicalRegion region, 
                 FieldID color_field, FieldID range_field);
    public:
      ColorRects(const DomainPointColoring &coloring,
          LogicalRegion region, FieldID color_field, FieldID range_field);
      ColorRects(const MultiDomainPointColoring &coloring,
          LogicalRegion region, FieldID color_field, FieldID range_field);
    public:
      ColorRects(const Coloring &coloring, LogicalRegion region,
                 FieldID color_field, FieldID range_field);
      ColorRects(const PointColoring &coloring, LogicalRegion region,
                 FieldID color_field, FieldID range_field);
    protected:
      Serializer rez;
    public:
      static TaskID TASK_ID;
      static void register_task(void);
      static void cpu_variant(const Task *task,
          const std::vector<PhysicalRegion> &regions, 
          Context ctx, Runtime *runtime);
    };
  };
#endif

  namespace Internal { 

    // Special helper for when we need a dummy context
#define DUMMY_CONTEXT       0

    /**
     * A class for deduplicating memory used with task arguments
     * and knowing when to collect the data associated with it
     */
    class AllocManager : public Collectable,
                         public LegionHeapify<AllocManager> {
    public:
      static const AllocationType alloc_type = ALLOC_MANAGER_ALLOC;
    public:
      AllocManager(size_t arglen)
        : Collectable(), 
          allocation(legion_malloc(ALLOC_INTERNAL_ALLOC, arglen)), 
          allocation_size(arglen) { }
      AllocManager(const AllocManager &rhs)
        : Collectable(), allocation(NULL), allocation_size(0)
      { assert(false); /*should never be called*/ }
      ~AllocManager(void)
      { legion_free(ALLOC_INTERNAL_ALLOC, allocation, allocation_size); }
    public:
      AllocManager& operator=(const AllocManager &rhs)
      { assert(false); /*should never be called*/ return *this; }
    public:
      inline void* get_allocation(void) const { return allocation; }
      inline size_t get_allocation_size(void) const
      { return allocation_size; }
    private:
      void *const allocation;
      size_t allocation_size;
    };

    /**
     * \class ArgumentMapImpl
     * An argument map implementation that provides
     * the backing store for an argument map handle.
     * Argument maps maintain pairs of domain points
     * and task arguments.  To make re-use of argument
     * maps efficient with small deltas, argument map
     * implementations provide a nice versionining system
     * with all argument map implementations sharing
     * a single backing store to de-duplicate domain
     * points and values.
     */
    class ArgumentMapImpl : public Collectable,
                            public LegionHeapify<ArgumentMapImpl> {
    public:
      static const AllocationType alloc_type = ARGUMENT_MAP_ALLOC;
    public:
      ArgumentMapImpl(void);
      ArgumentMapImpl(const FutureMap &rhs);
      ArgumentMapImpl(const ArgumentMapImpl &impl);
      ~ArgumentMapImpl(void);
    public:
      ArgumentMapImpl& operator=(const ArgumentMapImpl &rhs);
    public:
      bool has_point(const DomainPoint &point);
      void set_point(const DomainPoint &point, const TaskArgument &arg,
                     bool replace);
      bool remove_point(const DomainPoint &point);
      TaskArgument get_point(const DomainPoint &point);
    public:
      FutureMapImpl* freeze(TaskContext *ctx);
      void unfreeze(void);
    public:
      Runtime *const runtime;
    private:
      FutureMapImpl *future_map;
      std::map<DomainPoint,Future> arguments;
      bool equivalent; // argument and future_map the same
    };

    /**
     * \class FutureImpl
     * The base implementation of a future object.  The runtime
     * manages future implementation objects and knows how to
     * copy them from one node to another.  Future implementations
     * are always made first on the owner node and then moved
     * remotely.  We use the distributed collectable scheme
     * to manage garbage collection of distributed futures
     */
    class FutureImpl : public DistributedCollectable,
                       public LegionHeapify<FutureImpl> {
    public:
      static const AllocationType alloc_type = FUTURE_ALLOC;
    public:
      struct ContributeCollectiveArgs : 
        public LgTaskArgs<ContributeCollectiveArgs> {
      public:
        static const LgTaskID TASK_ID = LG_CONTRIBUTE_COLLECTIVE_ID;
      public:
        FutureImpl *impl;
        DynamicCollective dc;
        unsigned count;
      };
    public:
      FutureImpl(Runtime *rt, bool register_future, DistributedID did, 
                 AddressSpaceID owner_space, Operation *op = NULL);
      FutureImpl(const FutureImpl &rhs);
      virtual ~FutureImpl(void);
    public:
      FutureImpl& operator=(const FutureImpl &rhs);
    public:
      virtual VirtualChannelKind get_virtual_channel(void) const 
        { return DEFAULT_VIRTUAL_CHANNEL; }
    public:
      void get_void_result(bool silence_warnings = true);
      void* get_untyped_result(bool silence_warnings = true);
      bool is_empty(bool block, bool silence_warnings = true);
      size_t get_untyped_size(void);
      ApEvent get_ready_event(void) const { return ready_event; }
    public:
      // This will simply save the value of the future
      void set_result(const void *args, size_t arglen, bool own);
      // This will save the value of the future locally
      void unpack_future(Deserializer &derez);
      // Cause the future value to complete
      void complete_future(void);
      // Reset the future in case we need to restart the
      // computation for resiliency reasons
      bool reset_future(void);
      // A special function for predicates to peek
      // at the boolean value of a future if it is set
      bool get_boolean_value(bool &valid);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
    public:
      void register_dependence(Operation *consumer_op);
    protected:
      void mark_sampled(void);
      void broadcast_result(void);
      void register_waiter(AddressSpaceID sid);
    public:
      void record_future_registered(ReferenceMutator *creator);
      static void handle_future_result(Deserializer &derez, Runtime *rt);
      static void handle_future_subscription(Deserializer &derez, Runtime *rt);
    public:
      void contribute_to_collective(const DynamicCollective &dc,unsigned count);
      static void handle_contribute_to_collective(const void *args);
    public:
      // These three fields are only valid on the owner node
      Operation *const producer_op;
      const GenerationID op_gen;
#ifdef LEGION_SPY
      const UniqueID producer_uid;
#endif
    private:
      FRIEND_ALL_RUNTIME_CLASSES
      ApUserEvent ready_event;
      void *result; 
      size_t result_size;
      volatile bool empty;
      volatile bool sampled;
      // On the owner node, keep track of the registered waiters
      std::set<AddressSpaceID> registered_waiters;
    };

    /**
     * \class FutureMapImpl
     * The base implementation of a future map object. Note
     * that this is now a distributed collectable object too
     * that can be used to find the name of a future for a
     * given point anywhere in the machine.
     */
    class FutureMapImpl : public DistributedCollectable,
                          public LegionHeapify<FutureMapImpl> {
    public:
      static const AllocationType alloc_type = FUTURE_MAP_ALLOC;
    public:
      FutureMapImpl(TaskContext *ctx, Operation *op, 
                    Runtime *rt, DistributedID did, AddressSpaceID owner_space);
      FutureMapImpl(TaskContext *ctx, Runtime *rt, 
                    DistributedID did, AddressSpaceID owner_space,
                    bool register_now = true); // empty map
      FutureMapImpl(const FutureMapImpl &rhs);
      virtual ~FutureMapImpl(void);
    public:
      FutureMapImpl& operator=(const FutureMapImpl &rhs);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
    public:
      Future get_future(const DomainPoint &point, bool allow_empty = false);
      void set_future(const DomainPoint &point, FutureImpl *impl,
                      ReferenceMutator *mutator);
      void get_void_result(const DomainPoint &point, 
                            bool silence_warnings = true);
      void wait_all_results(bool silence_warnings = true);
      void complete_all_futures(void);
      bool reset_all_futures(void);
    public:
      void get_all_futures(std::map<DomainPoint,Future> &futures) const;
      void set_all_futures(const std::map<DomainPoint,Future> &futures);
#ifdef DEBUG_LEGION
    public:
      void add_valid_domain(const Domain &d);
      void add_valid_point(const DomainPoint &dp);
#endif
    public:
      void record_future_map_registered(ReferenceMutator *creator);
      static void handle_future_map_future_request(Deserializer &derez,
                              Runtime *runtime, AddressSpaceID source);
      static void handle_future_map_future_response(Deserializer &derez,
                                                    Runtime *runtime);
    public:
      TaskContext *const context;
      // Either an index space task or a must epoch op
      Operation *const op;
      const GenerationID op_gen;
    private:
      ApEvent ready_event;
      std::map<DomainPoint,Future> futures;
      bool valid;
#ifdef DEBUG_LEGION
    private:
      std::vector<Domain> valid_domains;
      std::set<DomainPoint> valid_points;
#endif
    };

    /**
     * \class PhysicalRegionImpl
     * The base implementation of a physical region object.
     * Physical region objects are not allowed to move from the
     * node in which they are created.  Like other objects
     * available to both the user and runtime they are reference
     * counted to know when they can be deleted.
     *
     * Note that we don't need to protect physical region impls
     * with any kind of synchronization mechanism since they
     * will only be manipulated by a single task which is 
     * guaranteed to only be running on one processor.
     */
    class PhysicalRegionImpl : public Collectable,
                               public LegionHeapify<PhysicalRegionImpl> {
    public:
      static const AllocationType alloc_type = PHYSICAL_REGION_ALLOC;
    public:
      PhysicalRegionImpl(const RegionRequirement &req, ApEvent ready_event,
                         bool mapped, TaskContext *ctx, MapperID mid,
                         MappingTagID tag, bool leaf, bool virt, Runtime *rt);
      PhysicalRegionImpl(const PhysicalRegionImpl &rhs);
      ~PhysicalRegionImpl(void);
    public:
      PhysicalRegionImpl& operator=(const PhysicalRegionImpl &rhs);
    public:
      inline bool created_accessor(void) const { return made_accessor; }
    public:
      void wait_until_valid(bool silence_warnings, 
                            bool warn = false, const char *src = NULL);
      bool is_valid(void) const;
      bool is_mapped(void) const;
      bool is_external_region(void) const;
      LogicalRegion get_logical_region(void) const;
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(bool silence_warnings = true);
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic> 
          get_field_accessor(FieldID field, bool silence_warnings = true);
    public:
      void unmap_region(void);
      void remap_region(ApEvent new_ready_event);
      const RegionRequirement& get_requirement(void) const;
      void set_reference(const InstanceRef &references);
      void reset_references(const InstanceSet &instances,ApUserEvent term_event,
                            ApEvent wait_for = ApEvent::NO_AP_EVENT);
      ApEvent get_ready_event(void) const;
      bool has_references(void) const;
      void get_references(InstanceSet &instances) const;
      void get_memories(std::set<Memory>& memories) const;
      void get_fields(std::vector<FieldID>& fields) const;
#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
    public:
      const char* get_task_name(void) const;
#endif
#ifdef BOUNDS_CHECKS
    public:
      bool contains_ptr(ptr_t ptr);
      bool contains_point(const DomainPoint &dp);
#endif
    public:
      void get_bounds(void *realm_is, TypeTag type_tag);
      PhysicalInstance get_instance_info(PrivilegeMode mode, 
                                         FieldID fid, size_t field_size, 
                                         void *realm_is, TypeTag type_tag,
                                         bool silence_warnings, 
                                         bool generic_accessor,
                                         bool check_field_size,
                                         ReductionOpID redop);
      void fail_bounds_check(DomainPoint p, FieldID fid, PrivilegeMode mode);
      void fail_bounds_check(Domain d, FieldID fid, PrivilegeMode mode);
    public:
      Runtime *const runtime;
      TaskContext *const context;
      const MapperID map_id;
      const MappingTagID tag;
      const bool leaf_region;
      const bool virtual_mapped;
    private:
      // Event for when the instance ref is ready
      ApEvent ready_event;
      // Instance ref
      InstanceSet references;
      RegionRequirement req;
      bool mapped; // whether it is currently mapped
      bool valid; // whether it is currently valid
      // whether to trigger the termination event
      // upon unmap
      bool trigger_on_unmap;
      bool made_accessor;
      ApUserEvent termination_event;
      ApEvent wait_for_unmap;
#ifdef BOUNDS_CHECKS
    private:
      Domain bounds;
#endif
    };

    /**
     * \class GrantImpl
     * This is the base implementation of a grant object.
     * The grant implementation remembers the locks that
     * must be acquired and gives out an precondition event
     * for acquiring the locks whenever a user attempts
     * to register as using the grant.  Registering requires
     * providing a completion event for the operation which
     * the grant object then knows to use when releasing the
     * locks.  Grants continues accepting registrations
     * until the runtime marks that it is no longer active.
     */
    class GrantImpl : public Collectable, public LegionHeapify<GrantImpl> {
    public:
      static const AllocationType alloc_type = GRANT_ALLOC;
    public:
      struct ReservationRequest {
      public:
        ReservationRequest(void)
          : reservation(Reservation::NO_RESERVATION),
            mode(0), exclusive(true) { }
        ReservationRequest(Reservation r, unsigned m, bool e)
          : reservation(r), mode(m), exclusive(e) { }
      public:
        Reservation reservation;
        unsigned mode;
        bool exclusive;
      };
    public:
      GrantImpl(void);
      GrantImpl(const std::vector<ReservationRequest> &requests);
      GrantImpl(const GrantImpl &rhs);
      ~GrantImpl(void);
    public:
      GrantImpl& operator=(const GrantImpl &rhs);
    public:
      void register_operation(ApEvent completion_event);
      ApEvent acquire_grant(void);
      void release_grant(void);
    public:
      void pack_grant(Serializer &rez);
      void unpack_grant(Deserializer &derez);
    private:
      std::vector<ReservationRequest> requests;
      bool acquired;
      ApEvent grant_event;
      std::set<ApEvent> completion_events;
      mutable LocalLock grant_lock;
    };

    class MPILegionHandshakeImpl : public Collectable,
                       public LegionHeapify<MPILegionHandshakeImpl> {
    public:
      static const AllocationType alloc_type = MPI_HANDSHAKE_ALLOC;
    public:
      MPILegionHandshakeImpl(bool init_in_MPI, int mpi_participants, 
                             int legion_participants);
      MPILegionHandshakeImpl(const MPILegionHandshakeImpl &rhs);
      ~MPILegionHandshakeImpl(void);
    public:
      MPILegionHandshakeImpl& operator=(const MPILegionHandshakeImpl &rhs);
    public:
      void initialize(void);
    public:
      void mpi_handoff_to_legion(void);
      void mpi_wait_on_legion(void);
    public:
      void legion_handoff_to_mpi(void);
      void legion_wait_on_mpi(void);
    public:
      PhaseBarrier get_legion_wait_phase_barrier(void);
      PhaseBarrier get_legion_arrive_phase_barrier(void);
      void advance_legion_handshake(void);
    private:
      const bool init_in_MPI;
      const int mpi_participants;
      const int legion_participants;
    private:
      PhaseBarrier mpi_wait_barrier;
      PhaseBarrier mpi_arrive_barrier;
      PhaseBarrier legion_wait_barrier; // copy of mpi_arrive_barrier
      PhaseBarrier legion_arrive_barrier; // copy of mpi_wait_barrier
    };

    class MPIRankTable {
    public:
      MPIRankTable(Runtime *runtime);
      MPIRankTable(const MPIRankTable &rhs);
      ~MPIRankTable(void);
    public:
      MPIRankTable& operator=(const MPIRankTable &rhs);
    public:
      void perform_rank_exchange(void);
      void handle_mpi_rank_exchange(Deserializer &derez);
    protected:
      bool send_explicit_stage(int stage);
      bool send_ready_stages(void);
      void unpack_exchange(int stage, Deserializer &derez);
      void complete_exchange(void);
    public:
      Runtime *const runtime;
      const bool participating;
    public:
      std::map<int,AddressSpace> forward_mapping;
      std::map<AddressSpace,int> reverse_mapping;
    protected:
      mutable LocalLock reservation;
      RtUserEvent done_event;
      std::vector<int> stage_notifications;
      std::vector<bool> sent_stages;
    }; 

    /**
     * \class ProcessorManager
     * This class manages all the state for a single processor
     * within a given instance of the Internal runtime.  It keeps
     * queues for each of the different stages that operations
     * undergo and also tracks when the scheduling task needs
     * to be run for a processor.
     */
    class ProcessorManager {
    public:
      struct TriggerOpArgs : public LgTaskArgs<TriggerOpArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_OP_ID;
      public:
        Operation *op;
      };
      struct SchedulerArgs : public LgTaskArgs<SchedulerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_SCHEDULER_ID;
      public:
        Processor proc;
      };
      struct TriggerTaskArgs : public LgTaskArgs<TriggerTaskArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_TASK_ID;
      public:
        TaskOp *op;
      };
      struct DeferMapperSchedulerArgs : 
        public LgTaskArgs<DeferMapperSchedulerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_MAPPER_SCHEDULER_TASK_ID;
      public:
        ProcessorManager *proxy_this;
        MapperID map_id;
        RtEvent deferral_event;
      };
      struct MapperMessage {
      public:
        MapperMessage(void)
          : target(Processor::NO_PROC), message(NULL), length(0), radix(0) { }
        MapperMessage(Processor t, void *mes, size_t l)
          : target(t), message(mes), length(l), radix(-1) { }
        MapperMessage(void *mes, size_t l, int r)
          : target(Processor::NO_PROC), message(mes), length(l), radix(r) { }
      public:
        Processor target;
        void *message;
        size_t length;
        int radix;
      };
    public:
      ProcessorManager(Processor proc, Processor::Kind proc_kind,
                       Runtime *rt, unsigned default_mappers,  
                       bool no_steal, bool replay);
      ProcessorManager(const ProcessorManager &rhs);
      ~ProcessorManager(void);
    public:
      ProcessorManager& operator=(const ProcessorManager &rhs);
    public:
      void prepare_for_shutdown(void);
    public:
      void startup_mappers(void);
      void add_mapper(MapperID mid, MapperManager *m, 
                      bool check, bool own, bool skip_replay = false);
      void replace_default_mapper(MapperManager *m, bool own);
      MapperManager* find_mapper(MapperID mid) const;
    public:
      void perform_scheduling(void);
      void launch_task_scheduler(void);
      void notify_deferred_mapper(MapperID map_id, RtEvent deferred_event);
      static void handle_defer_mapper(const void *args);
    public:
      void activate_context(InnerContext *context);
      void deactivate_context(InnerContext *context);
      void update_max_context_count(unsigned max_contexts);
    public:
      void process_steal_request(Processor thief, 
                                 const std::vector<MapperID> &thieves);
      void process_advertisement(Processor advertiser, MapperID mid);
    public:
      void add_to_ready_queue(TaskOp *op);
      void add_to_local_ready_queue(Operation *op, LgPriority priority,
                                    RtEvent wait_on);
    public:
      inline void find_visible_memories(std::set<Memory> &visible) const
        { visible = visible_memories; }
    protected:
      void perform_mapping_operations(void);
      void issue_advertisements(MapperID mid);
    protected:
      void increment_active_contexts(void);
      void decrement_active_contexts(void);
    protected:
      void increment_active_mappers(void);
      void decrement_active_mappers(void);
    public:
      // Immutable state
      Runtime *const runtime;
      const Processor local_proc;
      const Processor::Kind proc_kind;
      // Is stealing disabled 
      const bool stealing_disabled;
      // are we doing replay execution
      const bool replay_execution;
    protected:
      // Local queue state
      mutable LocalLock local_queue_lock;
      unsigned next_local_index;
    protected:
      // Scheduling state
      mutable LocalLock queue_lock;
      bool task_scheduler_enabled;
      unsigned total_active_contexts;
      unsigned total_active_mappers;
      struct ContextState {
      public:
        ContextState(void)
          : owned_tasks(0), active(false) { }
      public:
        unsigned owned_tasks;
        bool active;
      };
      std::vector<ContextState> context_states;
    protected:
      // Mapper objects
      std::map<MapperID,std::pair<MapperManager*,bool/*own*/> > mappers;
      // For each mapper something to track its state
      struct MapperState {
      public:
        std::list<TaskOp*> ready_queue;
        RtEvent deferral_event;
        bool added_tasks;
      };
      // State for each mapper for scheduling purposes
      std::map<MapperID,MapperState> mapper_states;
      // Lock for accessing mappers
      mutable LocalLock mapper_lock;
      // The set of visible memories from this processor
      std::set<Memory> visible_memories;
    };

    /**
     * \class MemoryManager
     * The goal of the memory manager is to keep track of all of
     * the physical instances that the runtime knows about in various
     * memories throughout the system.  This will then allow for
     * feedback when mapping to know when memories are nearing
     * their capacity.
     */
    class MemoryManager {
    public:
      enum RequestKind {
        CREATE_INSTANCE_CONSTRAINTS,
        CREATE_INSTANCE_LAYOUT,
        FIND_OR_CREATE_CONSTRAINTS,
        FIND_OR_CREATE_LAYOUT,
        FIND_ONLY_CONSTRAINTS,
        FIND_ONLY_LAYOUT,
      };
      enum InstanceState {
        COLLECTABLE_STATE = 0,
        ACTIVE_STATE = 1,
        ACTIVE_COLLECTED_STATE = 2,
        VALID_STATE = 3,
      };
    public:
      struct InstanceInfo {
      public:
        InstanceInfo(void)
          : current_state(COLLECTABLE_STATE), 
            deferred_collect(RtUserEvent::NO_RT_USER_EVENT),
            instance_size(0), min_priority(0) { }
      public:
        InstanceState current_state;
        RtUserEvent deferred_collect;
        size_t instance_size;
        GCPriority min_priority;
        std::map<std::pair<MapperID,Processor>,GCPriority> mapper_priorities;
      };
      template<bool SMALLER>
      struct CollectableInfo {
      public:
        CollectableInfo(void)
          : manager(NULL), instance_size(0), priority(0) { }
        CollectableInfo(PhysicalManager *m, size_t size, GCPriority p);
        CollectableInfo(const CollectableInfo &rhs);
        ~CollectableInfo(void);
      public:
        CollectableInfo& operator=(const CollectableInfo &rhs);
      public:
        bool operator<(const CollectableInfo &rhs) const;
        bool operator==(const CollectableInfo &rhs) const;
      public:
        PhysicalManager *manager;
        size_t instance_size;
        GCPriority priority;
      };
    public:
      MemoryManager(Memory mem, Runtime *rt);
      MemoryManager(const MemoryManager &rhs);
      ~MemoryManager(void);
    public:
      MemoryManager& operator=(const MemoryManager &rhs);
    public:
      void prepare_for_shutdown(void);
      void finalize(void);
    public:
      void register_remote_instance(PhysicalManager *manager);
      void unregister_remote_instance(PhysicalManager *manager);
    public:
      void activate_instance(PhysicalManager *manager);
      void deactivate_instance(PhysicalManager *manager);
      void validate_instance(PhysicalManager *manager);
      void invalidate_instance(PhysicalManager *manager);
    public:
      bool create_physical_instance(const LayoutConstraintSet &contraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, MapperID mapper_id,
                                    Processor processor, bool acquire, 
                                    GCPriority priority, UniqueID creator_id,
                                    bool remote = false);
      bool create_physical_instance(LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, MapperID mapper_id,
                                    Processor processor, bool acquire, 
                                    GCPriority priority, UniqueID creator_id,
                                    bool remote = false);
      bool find_or_create_physical_instance(
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    MapperID mapper_id, Processor processor,
                                    bool acquire, GCPriority priority, 
                                    bool tight_region_bounds,
                                    UniqueID creator_id, bool remote = false);
      bool find_or_create_physical_instance(
                                    LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    MapperID mapper_id, Processor processor,
                                    bool acquire, GCPriority priority, 
                                    bool tight_region_bounds,
                                    UniqueID creator_id, bool remote = false);
      bool find_physical_instance(  const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_bounds, bool remote = false);
      bool find_physical_instance(  LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_bounds, bool remote = false);
      void release_tree_instances(RegionTreeID tree_id);
      void set_garbage_collection_priority(PhysicalManager *manager,
                                    MapperID mapper_id, Processor proc,
                                    GCPriority priority);
      RtEvent acquire_instances(const std::set<PhysicalManager*> &managers,
                                    std::vector<bool> &results);
      void record_created_instance( PhysicalManager *manager, bool acquire,
                                    MapperID mapper_id, Processor proc,
                                    GCPriority priority, bool remote);
    public:
      void process_instance_request(Deserializer &derez, AddressSpaceID source);
      void process_instance_response(Deserializer &derez,AddressSpaceID source);
      void process_gc_priority_update(Deserializer &derez, AddressSpaceID src);
      void process_never_gc_response(Deserializer &derez);
      void process_acquire_request(Deserializer &derez, AddressSpaceID source);
      void process_acquire_response(Deserializer &derez);
    protected:
      bool find_satisfying_instance(const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire, 
                                    bool tight_region_bounds, bool remote);
      bool find_satisfying_instance(LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire, 
                                    bool tight_region_bounds, bool remote);
      bool find_satisfying_instance(const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, 
                                    std::set<PhysicalManager*> &candidates,
                                    bool acquire,bool tight_bounds,bool remote);
      bool find_satisfying_instance(LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, 
                                    std::set<PhysicalManager*> &candidates,
                                    bool acquire,bool tight_bounds,bool remote);
      bool find_valid_instance(     const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire, 
                                    bool tight_region_bounds, bool remote);
      bool find_valid_instance(     LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire, 
                                    bool tight_region_bounds, bool remote);
      void release_candidate_references(const std::set<PhysicalManager*> 
                                                        &candidates) const;
      void release_candidate_references(const std::deque<PhysicalManager*>
                                                        &candidates) const;
    protected:
      PhysicalManager* allocate_physical_instance(
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    UniqueID creator_id);
      PhysicalManager* find_and_record(PhysicalManager *manager, 
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    std::set<PhysicalManager*> &candidates,
                                    bool acquire, MapperID mapper_id, 
                                    Processor proc, GCPriority priority,
                                    bool tight_region_bounds, bool remote);
      PhysicalManager* find_and_record(PhysicalManager *manager, 
                                    LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    std::set<PhysicalManager*> &candidates,
                                    bool acquire, MapperID mapper_id, 
                                    Processor proc, GCPriority priority,
                                    bool tight_region_bounds, bool remote);
      void record_deleted_instance(PhysicalManager *manager); 
      void find_instances_by_state(size_t needed_size, InstanceState state, 
                     std::set<CollectableInfo<true> > &smaller_instances,
                     std::set<CollectableInfo<false> > &larger_instances) const;
      template<bool SMALLER>
      PhysicalManager* delete_and_allocate(InstanceBuilder &builder, 
                            size_t needed_size, size_t &total_bytes_deleted,
                      const std::set<CollectableInfo<SMALLER> > &instances);
    public:
      // The memory that we are managing
      const Memory memory;
      // The owner address space
      const AddressSpaceID owner_space;
      // Is this the owner memory or not
      const bool is_owner;
      // The capacity in bytes of this memory
      const size_t capacity;
      // The remaining capacity in this memory
      size_t remaining_capacity;
      // The runtime we are associate with
      Runtime *const runtime;
    protected:
      // Lock for controlling access to the data
      // structures in this memory manager
      mutable LocalLock manager_lock;
      // We maintain several sets of instances here
      // This is a generic list that tracks all the allocated instances
      // It is only valid on the owner node
      LegionMap<PhysicalManager*,InstanceInfo,
                MEMORY_INSTANCES_ALLOC>::tracked current_instances;
    };

    /**
     * \class VirtualChannel
     * This class provides the basic support for sending and receiving
     * messages for a single virtual channel.
     */
    class VirtualChannel {
    public:
      // Implement a three-state state-machine for sending
      // messages.  Either fully self-contained messages
      // or chains of partial messages followed by a final
      // message.
      enum MessageHeader {
        FULL_MESSAGE,
        PARTIAL_MESSAGE,
        FINAL_MESSAGE,
      };
    public:
      VirtualChannel(VirtualChannelKind kind,AddressSpaceID local_address_space,
                     size_t max_message_size, LegionProfiler *profiler);
      VirtualChannel(const VirtualChannel &rhs);
      ~VirtualChannel(void);
    public:
      VirtualChannel& operator=(const VirtualChannel &rhs);
    public:
      void package_message(Serializer &rez, MessageKind k, bool flush,
                           Runtime *runtime, Processor target, 
                           bool response, bool shutdown);
      void process_message(const void *args, size_t arglen, 
                        Runtime *runtime, AddressSpaceID remote_address_space);
      void confirm_shutdown(ShutdownManager *shutdown_manager, bool phase_one);
    private:
      void send_message(bool complete, Runtime *runtime, 
                        Processor target, bool response, bool shutdown);
      void handle_messages(unsigned num_messages, Runtime *runtime, 
                           AddressSpaceID remote_address_space,
                           const char *args, size_t arglen);
      void buffer_messages(unsigned num_messages,
                           const void *args, size_t arglen);
    private:
      mutable LocalLock send_lock;
      char *const sending_buffer;
      unsigned sending_index;
      const size_t sending_buffer_size;
      RtEvent last_message_event;
      MessageHeader header;
      unsigned packaged_messages;
      bool partial;
      // State for receiving messages
      // No lock for receiving messages since we know
      // that they are ordered
      char *receiving_buffer;
      unsigned receiving_index;
      size_t receiving_buffer_size;
      unsigned received_messages;
      bool observed_recent;
    private:
      LegionProfiler *const profiler;
    }; 

    /**
     * \class MessageManager
     * This class manages sending and receiving of message between
     * instances of the Internal runtime residing on different nodes.
     * The manager also abstracts some of the details of sending these
     * messages.  Messages can be accumulated together in bulk messages
     * for performance reason.  The runtime can also place an upper
     * bound on the size of the data communicated between runtimes in
     * an active message, which the message manager then uses to
     * break down larger messages into smaller active messages.
     *
     * On the receiving side, the message manager unpacks the messages
     * that have been sent and then call the appropriate runtime
     * methods for handling the messages.  In cases where larger
     * messages were broken down into smaller messages, then message
     * manager waits until it has received all the active messages
     * before handling the message.
     */
    class MessageManager { 
    public:
      MessageManager(AddressSpaceID remote, 
                     Runtime *rt, size_t max,
                     const std::set<Processor> &procs);
      MessageManager(const MessageManager &rhs);
      ~MessageManager(void);
    public:
      MessageManager& operator=(const MessageManager &rhs);
    public:
      void send_message(Serializer &rez, MessageKind kind, 
                        VirtualChannelKind channel, bool flush, 
                        bool response = false, bool shutdown = false);
      void receive_message(const void *args, size_t arglen);
      void confirm_shutdown(ShutdownManager *shutdown_manager,
                            bool phase_one);
    public:
      const AddressSpaceID remote_address_space;
    private:
      Runtime *const runtime;
      // State for sending messages
      Processor target;
      VirtualChannel *const channels; 
    };

    /**
     * \class ShutdownManager
     * A class for helping to manage the shutdown of the 
     * runtime after the application has finished
     */
    class ShutdownManager {
    public:
      enum ShutdownPhase {
        CHECK_TERMINATION = 1,
        CONFIRM_TERMINATION = 2,
        CHECK_SHUTDOWN = 3,
        CONFIRM_SHUTDOWN = 4,
      };
    public:
      struct RetryShutdownArgs : public LgTaskArgs<RetryShutdownArgs> {
      public:
        static const LgTaskID TASK_ID = LG_RETRY_SHUTDOWN_TASK_ID;
      public:
        ShutdownPhase phase;
      };
    public:
      ShutdownManager(ShutdownPhase phase, Runtime *rt, AddressSpaceID source,
                      unsigned radix, ShutdownManager *owner = NULL);
      ShutdownManager(const ShutdownManager &rhs);
      ~ShutdownManager(void);
    public:
      ShutdownManager& operator=(const ShutdownManager &rhs);
    public:
      bool attempt_shutdown(void);
      bool handle_response(bool success, const std::set<RtEvent> &to_add);
    protected:
      void finalize(void);
    public:
      static void handle_shutdown_notification(Deserializer &derez, 
                          Runtime *runtime, AddressSpaceID source);
      static void handle_shutdown_response(Deserializer &derez);
    public:
      void record_outstanding_tasks(void);
      void record_recent_message(void);
      void record_pending_message(RtEvent pending_event);
    public:
      const ShutdownPhase phase;
      Runtime *const runtime;
      const AddressSpaceID source; 
      const unsigned radix;
      ShutdownManager *const owner;
    protected:
      mutable LocalLock shutdown_lock;
      unsigned needed_responses;
      std::set<RtEvent> wait_for;
      bool result;
    };

    /**
     * \class GarbageCollectionEpoch
     * A class for managing the a set of garbage collections
     */
    class GarbageCollectionEpoch {
    public:
      struct GarbageCollectionArgs : public LgTaskArgs<GarbageCollectionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_COLLECT_ID;
      public:
        GarbageCollectionEpoch *epoch;
        LogicalView *view;
      };
    public:
      GarbageCollectionEpoch(Runtime *runtime);
      GarbageCollectionEpoch(const GarbageCollectionEpoch &rhs);
      ~GarbageCollectionEpoch(void);
    public:
      GarbageCollectionEpoch& operator=(const GarbageCollectionEpoch &rhs);
    public:
      void add_collection(LogicalView *view, ApEvent term_event,
                          ReferenceMutator *mutator);
      RtEvent launch(void);
      bool handle_collection(const GarbageCollectionArgs *args);
    private:
      Runtime *const runtime;
      int remaining;
      std::map<LogicalView*,std::set<ApEvent> > collections;
    };

    /**
     * \struct RegionTreeContext
     * A struct for storing the necessary data for managering a context
     * in the region tree.
     */
    class RegionTreeContext {
    public:
      RegionTreeContext(void)
        : ctx(-1) { }
      RegionTreeContext(ContextID c)
        : ctx(c) { }
    public:
      inline bool exists(void) const { return (ctx >= 0); }
      inline ContextID get_id(void) const 
      {
#ifdef DEBUG_LEGION
        assert(exists());
#endif
        return ContextID(ctx);
      }
      inline bool operator==(const RegionTreeContext &rhs) const
      {
        return (ctx == rhs.ctx);
      }
      inline bool operator!=(const RegionTreeContext &rhs) const
      {
        return (ctx != rhs.ctx);
      }
    private:
      int ctx;
    };

    /**
     * \class PendingVariantRegistration
     * A small helper class for deferring the restration of task
     * variants until the runtime is started.
     */
    class PendingVariantRegistration {
    public:
      PendingVariantRegistration(VariantID vid, bool has_return,
                                 const TaskVariantRegistrar &registrar,
                                 const void *user_data, size_t user_data_size,
                                 CodeDescriptor *realm_desc, 
                                 const char *task_name);
      PendingVariantRegistration(const PendingVariantRegistration &rhs);
      ~PendingVariantRegistration(void);
    public:
      PendingVariantRegistration& operator=(
                                      const PendingVariantRegistration &rhs);
    public:
      void perform_registration(Runtime *runtime);
    private:
      VariantID vid;
      bool has_return;
      TaskVariantRegistrar registrar;
      void *user_data;
      size_t user_data_size;
      CodeDescriptor *realm_desc; 
      char *logical_task_name; // optional semantic info to attach to the task
    };

    /**
     * \class TaskImpl
     * This class is used for storing all the meta-data associated 
     * with a logical task
     */
    class TaskImpl : public LegionHeapify<TaskImpl> {
    public:
      static const AllocationType alloc_type = TASK_IMPL_ALLOC;
    public:
      struct SemanticRequestArgs : public LgTaskArgs<SemanticRequestArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TASK_IMPL_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        TaskImpl *proxy_this;
        SemanticTag tag;
        AddressSpaceID source;
      };
    public:
      TaskImpl(TaskID tid, Runtime *rt, const char *name = NULL);
      TaskImpl(const TaskImpl &rhs);
      ~TaskImpl(void);
    public:
      TaskImpl& operator=(const TaskImpl &rhs);
    public:
      inline bool returns_value(void) const { return has_return_type; }
    public:
      VariantID get_unique_variant_id(void);
      void add_variant(VariantImpl *impl);
      VariantImpl* find_variant_impl(VariantID variant_id, bool can_fail);
      void find_valid_variants(std::vector<VariantID> &valid_variants, 
                               Processor::Kind kind) const;
    public:
      const char* get_name(bool needs_lock = true);
      void attach_semantic_information(SemanticTag tag, AddressSpaceID source,
         const void *buffer, size_t size, bool is_mutable, bool send_to_owner);
      bool retrieve_semantic_information(SemanticTag tag,
                                         const void *&buffer, size_t &size,
                                         bool can_fail, bool wait_until);
      void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                        const void *value, size_t size, bool is_mutable,
                        RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT);
      void send_semantic_request(AddressSpaceID target, SemanticTag tag, 
                             bool can_fail, bool wait_until, RtUserEvent ready);
      void process_semantic_request(SemanticTag tag, AddressSpaceID target, 
                             bool can_fail, bool wait_until, RtUserEvent ready);
    public:
      inline AddressSpaceID get_owner_space(void) const
        { return get_owner_space(task_id, runtime); }
      static AddressSpaceID get_owner_space(TaskID task_id, Runtime *runtime);
    public:
      static void handle_semantic_request(Runtime *runtime, 
                          Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(Runtime *runtime,
                          Deserializer &derez, AddressSpaceID source);
      static void handle_variant_request(Runtime *runtime,
                          Deserializer &derez, AddressSpaceID source);
    public:
      const TaskID task_id;
      Runtime *const runtime;
      char *const initial_name;
    private:
      mutable LocalLock task_lock;
      std::map<VariantID,VariantImpl*> variants;
      std::map<VariantID,RtEvent> outstanding_requests;
      // VariantIDs that we've handed out but haven't registered yet
      std::set<VariantID> pending_variants;
      std::map<SemanticTag,SemanticInfo> semantic_infos;
      // Track whether all these variants have a return type or not
      bool has_return_type;
      // Track whether all these variants are idempotent or not
      bool all_idempotent;
    };

    /**
     * \class VariantImpl
     * This class is used for storing all the meta-data associated
     * with a particular variant implementation of a task
     */
    class VariantImpl : public LegionHeapify<VariantImpl> { 
    public:
      static const AllocationType alloc_type = VARIANT_IMPL_ALLOC;
    public:
      VariantImpl(Runtime *runtime, VariantID vid, TaskImpl *owner, 
                  const TaskVariantRegistrar &registrar, bool ret_val, 
                  CodeDescriptor *realm_desc,
                  const void *user_data = NULL, size_t user_data_size = 0);
      VariantImpl(const VariantImpl &rhs);
      ~VariantImpl(void);
    public:
      VariantImpl& operator=(const VariantImpl &rhs);
    public:
      inline bool is_leaf(void) const { return leaf_variant; }
      inline bool is_inner(void) const { return inner_variant; }
      inline bool is_idempotent(void) const { return idempotent_variant; }
      inline bool returns_value(void) const { return has_return_value; }
      inline const char* get_name(void) const { return variant_name; }
      inline const ExecutionConstraintSet&
        get_execution_constraints(void) const { return execution_constraints; }
      inline const TaskLayoutConstraintSet& 
        get_layout_constraints(void) const { return layout_constraints; } 
    public:
      bool is_no_access_region(unsigned idx) const;
    public:
      ApEvent dispatch_task(Processor target, SingleTask *task, 
          TaskContext *ctx, ApEvent precondition, PredEvent pred,
          int priority, Realm::ProfilingRequestSet &requests);
      void dispatch_inline(Processor current, InlineContext *ctx);
    public:
      Processor::Kind get_processor_kind(bool warn) const;
    public:
      void send_variant_response(AddressSpaceID source, RtUserEvent done_event);
      void broadcast_variant(RtUserEvent done, AddressSpaceID origin,
                             AddressSpaceID local);
    public:
      static void handle_variant_broadcast(Runtime *runtime, 
                                           Deserializer &derez);
      static AddressSpaceID get_owner_space(VariantID vid, Runtime *runtime);
      static void handle_variant_response(Runtime *runtime, 
                                          Deserializer &derez);
    public:
      const VariantID vid;
      TaskImpl *const owner;
      Runtime *const runtime;
      const bool global; // globally valid variant
      const bool has_return_value; // has a return value
    public:
      const CodeDescriptorID descriptor_id;
      CodeDescriptor *const realm_descriptor;
    private:
      ExecutionConstraintSet execution_constraints;
      TaskLayoutConstraintSet   layout_constraints;
    private:
      void *user_data;
      size_t user_data_size;
      ApEvent ready_event;
    private: // properties
      bool leaf_variant;
      bool inner_variant;
      bool idempotent_variant;
    private:
      char *variant_name; 
    };

    /**
     * \class LayoutConstraints
     * A class for tracking a long-lived set of constraints
     * These can be moved around the system and referred to in 
     * variout places so we make it a distributed collectable
     */
    class LayoutConstraints : 
      public LayoutConstraintSet, public Collectable,
      public LegionHeapify<LayoutConstraints> {
    public:
      static const AllocationType alloc_type = LAYOUT_CONSTRAINTS_ALLOC; 
    protected:
      struct RemoveFunctor {
      public:
        RemoveFunctor(Serializer &r, Runtime *rt)
          : rez(r), runtime(rt) { }
      public:
        void apply(AddressSpaceID target);
      private:
        Serializer &rez;
        Runtime *runtime;
      };
    public:
      LayoutConstraints(LayoutConstraintID layout_id, FieldSpace handle, 
                        Runtime *runtime, 
                        AddressSpace owner_space, AddressSpaceID local_space);
      LayoutConstraints(LayoutConstraintID layout_id, Runtime *runtime, 
                        const LayoutConstraintRegistrar &registrar);
      LayoutConstraints(LayoutConstraintID layout_id,
                        Runtime *runtime, const LayoutConstraintSet &cons,
                        FieldSpace handle);
      LayoutConstraints(const LayoutConstraints &rhs);
      virtual ~LayoutConstraints(void);
    public:
      LayoutConstraints& operator=(const LayoutConstraints &rhs);
    public:
      inline FieldSpace get_field_space(void) const { return handle; }
      inline const char* get_name(void) const { return constraints_name; }
      inline bool is_owner(void) const { return (owner_space == local_space); }
    public:
      void send_constraint_response(AddressSpaceID source,
                                    RtUserEvent done_event);
      void update_constraints(Deserializer &derez);
      void release_remote_instances(void);
    public:
      bool entails(LayoutConstraints *other_constraints, unsigned total_dims);
      bool entails(const LayoutConstraintSet &other, unsigned total_dims) const;
      bool conflicts(LayoutConstraints *other_constraints, unsigned total_dims);
      bool conflicts(const LayoutConstraintSet &other, 
                     unsigned total_dims) const;
      bool entails_without_pointer(LayoutConstraints *other,
                                   unsigned total_dims);
      bool entails_without_pointer(const LayoutConstraintSet &other,
                                   unsigned total_dims) const;
    public:
      static AddressSpaceID get_owner_space(LayoutConstraintID layout_id,
                                            Runtime *runtime);
    public:
      static void process_request(Runtime *runtime, Deserializer &derez,
                                  AddressSpaceID source);
      static LayoutConstraintID process_response(Runtime *runtime, 
                          Deserializer &derez, AddressSpaceID source);
    public:
      const LayoutConstraintID layout_id;
      const FieldSpace handle;
      const AddressSpace owner_space;
      const AddressSpace local_space;
      Runtime *const runtime;
    protected:
      char *constraints_name;
      mutable LocalLock layout_lock;
    protected:
      std::map<LayoutConstraintID,bool> conflict_cache;
      std::map<LayoutConstraintID,bool> entailment_cache;
      std::map<LayoutConstraintID,bool> no_pointer_entailment_cache;
    protected:
      NodeSet remote_instances;
    };

    /**
     * Identity Projection Functor
     * A class that implements the identity projection function
     */
    class IdentityProjectionFunctor : public ProjectionFunctor {
    public:
      IdentityProjectionFunctor(Legion::Runtime *rt);
      virtual ~IdentityProjectionFunctor(void);
    public:
      virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                    LogicalRegion upper_bound,
                                    const DomainPoint &point);
      virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                    LogicalPartition upper_bound,
                                    const DomainPoint &point);
      virtual bool is_exclusive(void) const;
      virtual unsigned get_depth(void) const;
    };

    /**
     * \class ProjectionPoint
     * An abstract class for passing to projection functions
     * for recording the results of a projection
     */
    class ProjectionPoint {
    public:
      virtual const DomainPoint& get_domain_point(void) const = 0;
      virtual void set_projection_result(unsigned idx,LogicalRegion result) = 0;
    };

    /**
     * \class ProjectionFunction
     * A class for wrapping projection functors
     */
    class ProjectionFunction {
    public:
      ProjectionFunction(ProjectionID pid, ProjectionFunctor *functor);
      ProjectionFunction(const ProjectionFunction &rhs);
      ~ProjectionFunction(void);
    public:
      ProjectionFunction& operator=(const ProjectionFunction &rhs);
    public:
      // The old path explicitly for tasks
      LogicalRegion project_point(Task *task, unsigned idx, Runtime *runtime,
                                  const DomainPoint &point);
      void project_points(const RegionRequirement &req, unsigned idx,
          Runtime *runtime, const std::vector<PointTask*> &point_tasks);
      // Generalized and annonymized
      void project_points(Operation *op, unsigned idx, 
                          const RegionRequirement &req, Runtime *runtime,
                          const std::vector<ProjectionPoint*> &points);
    protected:
      // Old checking code explicitly for tasks
      void check_projection_region_result(const RegionRequirement &req,
                                          const Task *task, unsigned idx,
                                          LogicalRegion result, Runtime *rt);
      void check_projection_partition_result(const RegionRequirement &req,
                                             const Task *task, unsigned idx,
                                             LogicalRegion result, Runtime *rt);
      // Annonymized checking code
      void check_projection_region_result(const RegionRequirement &req,
                                          Operation *op, unsigned idx,
                                          LogicalRegion result, Runtime *rt);
      void check_projection_partition_result(const RegionRequirement &req,
                                          Operation *op, unsigned idx,
                                          LogicalRegion result, Runtime *rt);
    public:
      const int depth; 
      const bool is_exclusive;
      const ProjectionID projection_id;
      ProjectionFunctor *const functor;
    private:
      mutable LocalLock projection_reservation;
    }; 

    /**
     * \class Runtime 
     * This is the actual implementation of the Legion runtime functionality
     * that implements the underlying interface for the Runtime 
     * objects.  Most of the calls in the Runtime class translate
     * directly to calls to this interface.  Unfortunately this adds
     * an extra function call overhead to every runtime call because C++
     * is terrible and doesn't have mix-in classes.
     */
    class Runtime {
    public:
      struct DeferredRecycleArgs : public LgTaskArgs<DeferredRecycleArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_RECYCLE_ID;
      public:
        DistributedID did;
      };
      struct DeferredFutureSetArgs : public LgTaskArgs<DeferredFutureSetArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_FUTURE_SET_ID;
      public:
        FutureImpl *target;
        FutureImpl *result;
        TaskOp *task_op;
      };
      struct DeferredFutureMapSetArgs : 
        public LgTaskArgs<DeferredFutureMapSetArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_FUTURE_MAP_SET_ID;
      public:
        FutureMapImpl *future_map;
        FutureImpl *result;
        Domain domain;
        TaskOp *task_op;
      };
      struct DeferredEnqueueArgs : public LgTaskArgs<DeferredEnqueueArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_ENQUEUE_TASK_ID;
      public:
        ProcessorManager *manager;
        TaskOp *task;
      };
      struct TopFinishArgs : public LgTaskArgs<TopFinishArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TOP_FINISH_TASK_ID;
      public:
        TopLevelContext *ctx;
      };
      struct MapperTaskArgs : public LgTaskArgs<MapperTaskArgs> {
      public:
        static const LgTaskID TASK_ID = LG_MAPPER_TASK_ID;
      public:
        FutureImpl *future;
        MapperID map_id;
        Processor proc;
        ApEvent event;
        TopLevelContext *ctx;
      }; 
      struct SelectTunableArgs : public LgTaskArgs<SelectTunableArgs> {
      public:
        static const LgTaskID TASK_ID = LG_SELECT_TUNABLE_TASK_ID;
      public:
        MapperID mapper_id;
        MappingTagID tag;
        TunableID tunable_id;
        unsigned tunable_index; // only valid for LegionSpy
        TaskContext *ctx;
        FutureImpl *result;
      }; 
    public:
      struct ProcessorGroupInfo {
      public:
        ProcessorGroupInfo(void)
          : processor_group(Processor::NO_PROC) { }
        ProcessorGroupInfo(Processor p, const ProcessorMask &m)
          : processor_group(p), processor_mask(m) { }
      public:
        Processor           processor_group;
        ProcessorMask       processor_mask;
      };
    public:
      Runtime(Machine m, AddressSpaceID space_id,
              const std::set<Processor> &local_procs,
              const std::set<Processor> &local_util_procs,
              const std::set<AddressSpaceID> &address_spaces,
              const std::map<Processor,AddressSpaceID> &proc_spaces);
      Runtime(const Runtime &rhs);
      ~Runtime(void);
    public:
      Runtime& operator=(const Runtime &rhs);
    public:
      void register_static_variants(void);
      void register_static_constraints(void);
      void register_static_projections(void);
      void initialize_legion_prof(void);
      void initialize_mappers(void);
      void startup_mappers(void);
      void finalize_runtime(void);
      void launch_top_level_task(void);
      ApEvent launch_mapper_task(Mapper *mapper, Processor proc, 
                                 Processor::TaskFuncID tid,
                                 const TaskArgument &arg, MapperID map_id);
      void process_mapper_task_result(const MapperTaskArgs *args); 
    public:
      IndexSpace create_index_space(Context ctx, const void *realm_is,
                                    TypeTag type_tag);
      IndexSpace union_index_spaces(Context ctx, 
                                    const std::vector<IndexSpace> &spaces);
      IndexSpace intersect_index_spaces(Context ctx,
                                    const std::vector<IndexSpace> &spaces);
      IndexSpace subtract_index_spaces(Context ctx,
                                    IndexSpace left, IndexSpace right);
      void destroy_index_space(Context ctx, IndexSpace handle);
      // Called from deletion op
      void finalize_index_space_destroy(IndexSpace handle);
    public:
      void destroy_index_partition(Context ctx, IndexPartition handle);
      // Called from deletion op
      void finalize_index_partition_destroy(IndexPartition handle);
    public:
      IndexPartition create_equal_partition(Context ctx, IndexSpace parent,
                                            IndexSpace color_space, 
                                            size_t granuarlity, Color color);
      IndexPartition create_partition_by_union(Context ctx, IndexSpace parent,
                                               IndexPartition handle1,
                                               IndexPartition handle2,
                                               IndexSpace color_space,
                                               PartitionKind kind, Color color);
      IndexPartition create_partition_by_intersection(Context ctx, 
                                               IndexSpace parent,
                                               IndexPartition handle1,
                                               IndexPartition handle2,
                                               IndexSpace color_space,
                                               PartitionKind kind, Color color);
      IndexPartition create_partition_by_difference(Context ctx, 
                                               IndexSpace parent,
                                               IndexPartition handle1,
                                               IndexPartition handle2,
                                               IndexSpace color_space,
                                               PartitionKind kind, Color color);
      Color create_cross_product_partitions(Context ctx, 
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                std::map<IndexSpace,IndexPartition> &handles,
                                            PartitionKind kind, Color color);
      void create_association(Context ctx,
                              LogicalRegion domain,
                              LogicalRegion domain_parent,
                              FieldID domain_fid,
                              IndexSpace range,
                              MapperID id, MappingTagID tag);
      IndexPartition create_restricted_partition(Context ctx,
                                                 IndexSpace parent,
                                                 IndexSpace color_space,
                                                 const void *transform,
                                                 size_t transform_size,
                                                 const void *extent,
                                                 size_t extent_size,
                                                 PartitionKind part_kind,
                                                 Color color);
      IndexPartition create_partition_by_field(Context ctx, 
                                               LogicalRegion handle,
                                               LogicalRegion parent,
                                               FieldID fid,
                                               IndexSpace color_space,
                                               Color color,
                                               MapperID id, MappingTagID tag);
      IndexPartition create_partition_by_image(Context ctx,
                                               IndexSpace handle,
                                               LogicalPartition projection,
                                               LogicalRegion parent,
                                               FieldID fid, 
                                               IndexSpace color_space,
                                               PartitionKind part_kind,
                                               Color color,
                                               MapperID id, MappingTagID tag);
      IndexPartition create_partition_by_image_range(Context ctx,
                                               IndexSpace handle,
                                               LogicalPartition projection,
                                               LogicalRegion parent,
                                               FieldID fid, 
                                               IndexSpace color_space,
                                               PartitionKind part_kind,
                                               Color color,
                                               MapperID id, MappingTagID tag);
      IndexPartition create_partition_by_preimage(Context ctx,
                                               IndexPartition projection,
                                               LogicalRegion handle,
                                               LogicalRegion parent,
                                               FieldID fid,
                                               IndexSpace color_space,
                                               PartitionKind part_kind,
                                               Color color,
                                               MapperID id, MappingTagID tag);
      IndexPartition create_partition_by_preimage_range(Context ctx,
                                               IndexPartition projection,
                                               LogicalRegion handle,
                                               LogicalRegion parent,
                                               FieldID fid,
                                               IndexSpace color_space,
                                               PartitionKind part_kind,
                                               Color color,
                                               MapperID id, MappingTagID tag);
      IndexPartition create_pending_partition(Context ctx, IndexSpace parent,
                                              IndexSpace color_space,
                                              PartitionKind part_kind,
                                              Color color);
      IndexSpace create_index_space_union(Context ctx, IndexPartition parent,
                                          const void *realm_color, 
                                          TypeTag type_tag,
                                        const std::vector<IndexSpace> &handles);
      IndexSpace create_index_space_union(Context ctx, IndexPartition parent,
                                          const void *realm_color,
                                          TypeTag type_tag,
                                          IndexPartition handle);
      IndexSpace create_index_space_intersection(Context ctx, 
                                                 IndexPartition parent,
                                                 const void *realm_color,
                                                 TypeTag type_tag,
                                       const std::vector<IndexSpace> &handles);
      IndexSpace create_index_space_intersection(Context ctx,
                                                 IndexPartition parent,
                                                 const void *realm_color,
                                                 TypeTag type_tag,
                                                 IndexPartition handle); 
      IndexSpace create_index_space_difference(Context ctx, 
                                               IndexPartition parent,
                                               const void *realm_color,
                                               TypeTag type_tag,
                                               IndexSpace initial,
                                       const std::vector<IndexSpace> &handles);
    public:
      IndexPartition get_index_partition(Context ctx, IndexSpace parent, 
                                         Color color);
      IndexPartition get_index_partition(IndexSpace parent, Color color);
      bool has_index_partition(Context ctx, IndexSpace parent, Color color);
      bool has_index_partition(IndexSpace parent, Color color); 
      IndexSpace get_index_subspace(Context ctx, IndexPartition p,
                                    const void *realm_color, TypeTag type_tag);
      IndexSpace get_index_subspace(IndexPartition p, 
                                    const void *realm_color, TypeTag type_tag);
      bool has_index_subspace(Context ctx, IndexPartition p,
                              const void *realm_color, TypeTag type_tag);
      bool has_index_subspace(IndexPartition p, 
                              const void *realm_color, TypeTag type_tag);
      void get_index_space_domain(Context ctx, IndexSpace handle,
                                  void *realm_is, TypeTag type_tag);
      void get_index_space_domain(IndexSpace handle, 
                                  void *realm_is, TypeTag type_tag);
      Domain get_index_partition_color_space(Context ctx, IndexPartition p);
      Domain get_index_partition_color_space(IndexPartition p);
      void get_index_partition_color_space(IndexPartition p, 
                                           void *realm_is, TypeTag type_tag);
      IndexSpace get_index_partition_color_space_name(Context ctx,
                                                      IndexPartition p);
      IndexSpace get_index_partition_color_space_name(IndexPartition p);
      void get_index_space_partition_colors(Context ctx, IndexSpace handle,
                                            std::set<Color> &colors);
      void get_index_space_partition_colors(IndexSpace handle,
                                            std::set<Color> &colors);
      bool is_index_partition_disjoint(Context ctx, IndexPartition p);
      bool is_index_partition_disjoint(IndexPartition p);
      bool is_index_partition_complete(Context ctx, IndexPartition p);
      bool is_index_partition_complete(IndexPartition p);
      void get_index_space_color_point(Context ctx, IndexSpace handle,
                                       void *realm_color, TypeTag type_tag);
      void get_index_space_color_point(IndexSpace handle,
                                       void *realm_color, TypeTag type_tag);
      DomainPoint get_index_space_color_point(Context ctx, IndexSpace handle);
      DomainPoint get_index_space_color_point(IndexSpace handle);
      Color get_index_partition_color(Context ctx, IndexPartition handle);
      Color get_index_partition_color(IndexPartition handle);
      IndexSpace get_parent_index_space(Context ctx, IndexPartition handle);
      IndexSpace get_parent_index_space(IndexPartition handle);
      bool has_parent_index_partition(Context ctx, IndexSpace handle);
      bool has_parent_index_partition(IndexSpace handle);
      IndexPartition get_parent_index_partition(Context ctx, IndexSpace handle);
      IndexPartition get_parent_index_partition(IndexSpace handle);
      unsigned get_index_space_depth(Context ctx, IndexSpace handle);
      unsigned get_index_space_depth(IndexSpace handle);
      unsigned get_index_partition_depth(Context ctx, IndexPartition handle);
      unsigned get_index_partition_depth(IndexPartition handle);
    public:
      bool safe_cast(Context ctx, LogicalRegion region,
                     const void *realm_point, TypeTag type_tag);
    public:
      FieldSpace create_field_space(Context ctx);
      void destroy_field_space(Context ctx, FieldSpace handle);
      size_t get_field_size(Context ctx, FieldSpace handle, FieldID fid);
      size_t get_field_size(FieldSpace handle, FieldID fid);
      void get_field_space_fields(Context ctx, FieldSpace handle,
                                  std::vector<FieldID> &fields);
      void get_field_space_fields(FieldSpace handle, 
                                  std::vector<FieldID> &fields);
      // Called from deletion op
      void finalize_field_space_destroy(FieldSpace handle);
      void finalize_field_destroy(FieldSpace handle, FieldID fid);
      void finalize_field_destroy(FieldSpace handle, 
                                  const std::set<FieldID> &to_free);
    public:
      LogicalRegion create_logical_region(Context ctx, IndexSpace index,
                                          FieldSpace fields);
      void destroy_logical_region(Context ctx, LogicalRegion handle);
      void destroy_logical_partition(Context ctx, LogicalPartition handle);
      // Called from deletion ops
      void finalize_logical_region_destroy(LogicalRegion handle);
      void finalize_logical_partition_destroy(LogicalPartition handle);
    public:
      LogicalPartition get_logical_partition(Context ctx, LogicalRegion parent, 
                                             IndexPartition handle);
      LogicalPartition get_logical_partition(LogicalRegion parent,
                                             IndexPartition handle);
      LogicalPartition get_logical_partition_by_color(Context ctx, 
                                                      LogicalRegion parent, 
                                                      Color c);
      LogicalPartition get_logical_partition_by_color(LogicalRegion parent,
                                                      Color c);
      bool has_logical_partition_by_color(Context ctx, LogicalRegion parent,
                                          Color c);
      bool has_logical_partition_by_color(LogicalRegion parent, Color c);
      LogicalPartition get_logical_partition_by_tree(Context ctx, 
                                                     IndexPartition handle, 
                                                     FieldSpace fspace, 
                                                     RegionTreeID tid); 
      LogicalPartition get_logical_partition_by_tree(IndexPartition handle,
                                                     FieldSpace fspace,
                                                     RegionTreeID tid);
      LogicalRegion get_logical_subregion(Context ctx, LogicalPartition parent, 
                                          IndexSpace handle);
      LogicalRegion get_logical_subregion(LogicalPartition parent,
                                          IndexSpace handle);
      LogicalRegion get_logical_subregion_by_color(Context ctx,
                                                   LogicalPartition parent,
                                                   const void *realm_color,
                                                   TypeTag type_tag);
      LogicalRegion get_logical_subregion_by_color(LogicalPartition parent,
                                                   const void *realm_color,
                                                   TypeTag type_tag);
      bool has_logical_subregion_by_color(Context ctx, LogicalPartition parent,
                                          const void *realm_color, 
                                          TypeTag type_tag);
      bool has_logical_subregion_by_color(LogicalPartition parent,
                                          const void *realm_color,
                                          TypeTag type_tag);
      LogicalRegion get_logical_subregion_by_tree(Context ctx, 
                                                  IndexSpace handle, 
                                                  FieldSpace fspace, 
                                                  RegionTreeID tid);
      LogicalRegion get_logical_subregion_by_tree(IndexSpace handle,
                                                  FieldSpace fspace,
                                                  RegionTreeID tid);
      void get_logical_region_color(Context ctx, LogicalRegion handle,
                                    void *realm_color, TypeTag type_tag);
      void get_logical_region_color(LogicalRegion handle, 
                                    void *realm_color, TypeTag type_tag);
      DomainPoint get_logical_region_color_point(Context ctx, 
                                                 LogicalRegion handle);
      DomainPoint get_logical_region_color_point(LogicalRegion handle);
      Color get_logical_partition_color(Context ctx, LogicalPartition handle);
      Color get_logical_partition_color(LogicalPartition handle);
      LogicalRegion get_parent_logical_region(Context ctx, 
                                              LogicalPartition handle);
      LogicalRegion get_parent_logical_region(LogicalPartition handle);
      bool has_parent_logical_partition(Context ctx, LogicalRegion handle);
      bool has_parent_logical_partition(LogicalRegion handle);
      LogicalPartition get_parent_logical_partition(Context ctx, 
                                                    LogicalRegion handle);
      LogicalPartition get_parent_logical_partition(LogicalRegion handle);
    public:
      FieldAllocator create_field_allocator(Context ctx, FieldSpace handle);
      ArgumentMap create_argument_map(void);
    public:
      Future execute_task(Context ctx, const TaskLauncher &launcher);
      FutureMap execute_index_space(Context ctx, 
                                    const IndexTaskLauncher &launcher);
      Future execute_index_space(Context ctx, 
                    const IndexTaskLauncher &launcher, ReductionOpID redop);
    public:
      PhysicalRegion map_region(Context ctx, 
                                const InlineLauncher &launcher);
      PhysicalRegion map_region(Context ctx, unsigned idx, 
                                MapperID id = 0, MappingTagID tag = 0);
      void remap_region(Context ctx, PhysicalRegion region);
      void unmap_region(Context ctx, PhysicalRegion region);
      void unmap_all_regions(Context ctx);
    public:
      void fill_fields(Context ctx, const FillLauncher &launcher);
      void fill_fields(Context ctx, const IndexFillLauncher &launcher);
      PhysicalRegion attach_external_resource(Context ctx,
                                              const AttachLauncher &launcher);
      void detach_external_resource(Context ctx, PhysicalRegion region);
      void issue_copy_operation(Context ctx, const CopyLauncher &launcher);
      void issue_copy_operation(Context ctx, const IndexCopyLauncher &launcher);
    public:
      Predicate create_predicate(Context ctx, const Future &f);
      Predicate predicate_not(Context ctx, const Predicate &p);
      Predicate create_predicate(Context ctx,const PredicateLauncher &launcher);
      Future get_predicate_future(Context ctx, const Predicate &p);
    public:
      Lock create_lock(Context ctx);
      void destroy_lock(Context ctx, Lock l);
      Grant acquire_grant(Context ctx, 
                          const std::vector<LockRequest> &requests);
      void release_grant(Context ctx, Grant grant);
    public:
      PhaseBarrier create_phase_barrier(Context ctx, unsigned arrivals);
      void destroy_phase_barrier(Context ctx, PhaseBarrier pb);
      PhaseBarrier advance_phase_barrier(Context ctx, PhaseBarrier pb);
    public:
      DynamicCollective create_dynamic_collective(Context ctx,
                                                  unsigned arrivals,
                                                  ReductionOpID redop,
                                                  const void *init_value,
                                                  size_t init_size);
      void destroy_dynamic_collective(Context ctx, DynamicCollective dc);
      void arrive_dynamic_collective(Context ctx, DynamicCollective dc,
                                     const void *buffer, size_t size,
                                     unsigned count);
      void defer_dynamic_collective_arrival(Context ctx, 
                                            DynamicCollective dc,
                                            const Future &f, unsigned count);
      Future get_dynamic_collective_result(Context ctx, DynamicCollective dc);
      DynamicCollective advance_dynamic_collective(Context ctx,
                                                   DynamicCollective dc);
    public:
      void issue_acquire(Context ctx, const AcquireLauncher &launcher);
      void issue_release(Context ctx, const ReleaseLauncher &launcher);
      void issue_mapping_fence(Context ctx);
      void issue_execution_fence(Context ctx);
      void begin_trace(Context ctx, TraceID tid);
      void end_trace(Context ctx, TraceID tid);
      void begin_static_trace(Context ctx, 
                              const std::set<RegionTreeID> *managed);
      void end_static_trace(Context ctx);
      void complete_frame(Context ctx);
      FutureMap execute_must_epoch(Context ctx, 
                                   const MustEpochLauncher &launcher);
      Future issue_timing_measurement(Context ctx,
                                      const TimingLauncher &launcher);
    public:
      Future select_tunable_value(Context ctx, TunableID tid,
                                  MapperID mid, MappingTagID tag);
      int get_tunable_value(Context ctx, TunableID tid, 
                            MapperID mid, MappingTagID tag);
      void perform_tunable_selection(const SelectTunableArgs *args);
    public:
      void* get_local_task_variable(Context ctx, LocalVariableID id);
      void set_local_task_variable(Context ctx, LocalVariableID id,
                      const void *value, void (*destructor)(void*));
    public:
      Mapper* get_mapper(Context ctx, MapperID id, Processor target);
      Processor get_executing_processor(Context ctx);
      void raise_region_exception(Context ctx, PhysicalRegion region, 
                                  bool nuclear);
    public:
      const std::map<int,AddressSpace>& find_forward_MPI_mapping(void);
      const std::map<AddressSpace,int>& find_reverse_MPI_mapping(void);
      int find_local_MPI_rank(void);
    public:
      Mapping::MapperRuntime* get_mapper_runtime(void);
      MapperID generate_dynamic_mapper_id(void);
      static MapperID& get_current_static_mapper_id(void);
      static MapperID generate_static_mapper_id(void);
      void add_mapper(MapperID map_id, Mapper *mapper, Processor proc);
      void replace_default_mapper(Mapper *mapper, Processor proc);
      MapperManager* find_mapper(Processor target, MapperID map_id);
      static MapperManager* wrap_mapper(Runtime *runtime, Mapper *mapper,
                                        MapperID map_id, Processor proc);
    public:
      ProjectionID generate_dynamic_projection_id(void);
      static ProjectionID& get_current_static_projection_id(void);
      static ProjectionID generate_static_projection_id(void);
      void register_projection_functor(ProjectionID pid, 
                                       ProjectionFunctor *func,
                                       bool need_zero_check = true,
                                       bool was_preregistered = false);
      static void preregister_projection_functor(ProjectionID pid,
                                       ProjectionFunctor *func);
      ProjectionFunction* find_projection_function(ProjectionID pid);
    public:
      void attach_semantic_information(TaskID task_id, SemanticTag,
                                   const void *buffer, size_t size, 
                                   bool is_mutable, bool send_to_owner = true);
      void attach_semantic_information(IndexSpace handle, SemanticTag tag,
                       const void *buffer, size_t size, bool is_mutable);
      void attach_semantic_information(IndexPartition handle, SemanticTag tag,
                       const void *buffer, size_t size, bool is_mutable);
      void attach_semantic_information(FieldSpace handle, SemanticTag tag,
                       const void *buffer, size_t size, bool is_mutable);
      void attach_semantic_information(FieldSpace handle, FieldID fid,
                                       SemanticTag tag, const void *buffer, 
                                       size_t size, bool is_mutable);
      void attach_semantic_information(LogicalRegion handle, SemanticTag tag,
                       const void *buffer, size_t size, bool is_mutable);
      void attach_semantic_information(LogicalPartition handle, SemanticTag tag,
                       const void *buffer, size_t size, bool is_mutable);
    public:
      bool retrieve_semantic_information(TaskID task_id, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
      bool retrieve_semantic_information(IndexSpace handle, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
      bool retrieve_semantic_information(IndexPartition handle, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
      bool retrieve_semantic_information(FieldSpace handle, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
      bool retrieve_semantic_information(FieldSpace handle, FieldID fid,
                                         SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
      bool retrieve_semantic_information(LogicalRegion handle, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
      bool retrieve_semantic_information(LogicalPartition part, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
    public:
      FieldID allocate_field(Context ctx, FieldSpace space, 
                             size_t field_size, FieldID fid, 
                             bool local, CustomSerdezID serdez);
      void free_field(Context ctx, FieldSpace space, FieldID fid);
      void allocate_fields(Context ctx, FieldSpace space, 
                           const std::vector<size_t> &sizes,
                           std::vector<FieldID> &resulting_fields, 
                           bool local, CustomSerdezID serdez);
      void free_fields(Context ctx, FieldSpace space, 
                       const std::set<FieldID> &to_free);
    public:
      TaskID generate_dynamic_task_id(void);
      VariantID register_variant(const TaskVariantRegistrar &registrar,
                                 const void *user_data, size_t user_data_size,
                                 CodeDescriptor *realm,
                                 bool ret, VariantID vid = AUTO_GENERATE_ID,
                                 bool check_task_id = true);
      TaskImpl* find_or_create_task_impl(TaskID task_id);
      TaskImpl* find_task_impl(TaskID task_id);
      VariantImpl* find_variant_impl(TaskID task_id, VariantID variant_id,
                                     bool can_fail = false);
    public:
      // Memory manager functions
      MemoryManager* find_memory_manager(Memory mem);
      AddressSpaceID find_address_space(Memory handle) const;
    public:
      // Messaging functions
      MessageManager* find_messenger(AddressSpaceID sid);
      MessageManager* find_messenger(Processor target);
      AddressSpaceID find_address_space(Processor target) const;
    public:
      void process_mapper_message(Processor target, MapperID map_id,
                                  Processor source, const void *message, 
                                  size_t message_size, unsigned message_kind);
      void process_mapper_broadcast(MapperID map_id, Processor source,
                                    const void *message, size_t message_size, 
                                    unsigned message_kind, int radix,int index);
    public:
      void send_task(TaskOp *task);
      void send_tasks(Processor target, const std::set<TaskOp*> &tasks);
      void send_steal_request(const std::multimap<Processor,MapperID> &targets,
                              Processor thief);
      void send_advertisements(const std::set<Processor> &targets,
                              MapperID map_id, Processor source);
      void send_index_space_node(AddressSpaceID target, Serializer &rez);
      void send_index_space_request(AddressSpaceID target, Serializer &rez);
      void send_index_space_return(AddressSpaceID target, Serializer &rez);
      void send_index_space_set(AddressSpaceID target, Serializer &rez);
      void send_index_space_child_request(AddressSpaceID target, 
                                          Serializer &rez);
      void send_index_space_child_response(AddressSpaceID target,
                                           Serializer &rez);
      void send_index_space_colors_request(AddressSpaceID target,
                                           Serializer &rez);
      void send_index_space_colors_response(AddressSpaceID target,
                                            Serializer &rez);
      void send_index_partition_notification(AddressSpaceID target, 
                                             Serializer &rez);
      void send_index_partition_node(AddressSpaceID target, Serializer &rez);
      void send_index_partition_request(AddressSpaceID target, Serializer &rez);
      void send_index_partition_return(AddressSpaceID target, Serializer &rez);
      void send_index_partition_child_request(AddressSpaceID target,
                                              Serializer &rez);
      void send_index_partition_child_response(AddressSpaceID target,
                                               Serializer &rez);
      void send_field_space_node(AddressSpaceID target, Serializer &rez);
      void send_field_space_request(AddressSpaceID target, Serializer &rez);
      void send_field_space_return(AddressSpaceID target, Serializer &rez);
      void send_field_alloc_request(AddressSpaceID target, Serializer &rez);
      void send_field_alloc_notification(AddressSpaceID target,Serializer &rez);
      void send_field_space_top_alloc(AddressSpaceID target, Serializer &rez);
      void send_field_free(AddressSpaceID target, Serializer &rez);
      void send_local_field_alloc_request(AddressSpaceID target, 
                                          Serializer &rez);
      void send_local_field_alloc_response(AddressSpaceID target,
                                           Serializer &rez);
      void send_local_field_free(AddressSpaceID target, Serializer &rez);
      void send_local_field_update(AddressSpaceID target, Serializer &rez);
      void send_top_level_region_request(AddressSpaceID target,Serializer &rez);
      void send_top_level_region_return(AddressSpaceID target, Serializer &rez);
      void send_logical_region_node(AddressSpaceID target, Serializer &rez);
      void send_index_space_destruction(IndexSpace handle, 
                                        AddressSpaceID target);
      void send_index_partition_destruction(IndexPartition handle, 
                                            AddressSpaceID target);
      void send_field_space_destruction(FieldSpace handle, 
                                        AddressSpaceID target);
      void send_logical_region_destruction(LogicalRegion handle, 
                                           AddressSpaceID target);
      void send_logical_partition_destruction(LogicalPartition handle,
                                              AddressSpaceID target);
      void send_individual_remote_mapped(Processor target, 
                                         Serializer &rez, bool flush = true);
      void send_individual_remote_complete(Processor target, Serializer &rez);
      void send_individual_remote_commit(Processor target, Serializer &rez);
      void send_slice_remote_mapped(Processor target, Serializer &rez);
      void send_slice_remote_complete(Processor target, Serializer &rez);
      void send_slice_remote_commit(Processor target, Serializer &rez);
      void send_did_remote_registration(AddressSpaceID target, Serializer &rez);
      void send_did_remote_valid_update(AddressSpaceID target, Serializer &rez);
      void send_did_remote_gc_update(AddressSpaceID target, Serializer &rez);
      void send_did_remote_resource_update(AddressSpaceID target,
                                           Serializer &rez);
      void send_did_remote_invalidate(AddressSpaceID target, Serializer &rez);
      void send_did_remote_deactivate(AddressSpaceID target, Serializer &rez);
      void send_did_add_create_reference(AddressSpaceID target,Serializer &rez);
      void send_did_remove_create_reference(AddressSpaceID target,
                                            Serializer &rez, bool flush = true);
      void send_did_remote_unregister(AddressSpaceID target, Serializer &rez,
                                      VirtualChannelKind vc);
      void send_back_logical_state(AddressSpaceID target, Serializer &rez);
      void send_back_atomic(AddressSpaceID target, Serializer &rez);
      void send_atomic_reservation_request(AddressSpaceID target, 
                                           Serializer &rez);
      void send_atomic_reservation_response(AddressSpaceID target, 
                                            Serializer &rez);
      void send_materialized_view(AddressSpaceID target, Serializer &rez);
      void send_composite_view(AddressSpaceID target, Serializer &rez);
      void send_fill_view(AddressSpaceID target, Serializer &rez);
      void send_phi_view(AddressSpaceID target, Serializer &rez);
      void send_reduction_view(AddressSpaceID target, Serializer &rez);
      void send_instance_manager(AddressSpaceID target, Serializer &rez);
      void send_reduction_manager(AddressSpaceID target, Serializer &rez);
      void send_create_top_view_request(AddressSpaceID target, Serializer &rez);
      void send_create_top_view_response(AddressSpaceID target,Serializer &rez);
      void send_subview_did_request(AddressSpaceID target, Serializer &rez);
      void send_subview_did_response(AddressSpaceID target, Serializer &rez);
      void send_view_update_request(AddressSpaceID target, Serializer &rez);
      void send_view_update_response(AddressSpaceID target, Serializer &rez);
      void send_view_remote_update(AddressSpaceID target, Serializer &rez);
      void send_view_remote_invalidate(AddressSpaceID target, Serializer &rez);
      void send_future_result(AddressSpaceID target, Serializer &rez);
      void send_future_subscription(AddressSpaceID target, Serializer &rez);
      void send_future_map_request_future(AddressSpaceID target, 
                                          Serializer &rez);
      void send_future_map_response_future(AddressSpaceID target,
                                           Serializer &rez);
      void send_mapper_message(AddressSpaceID target, Serializer &rez);
      void send_mapper_broadcast(AddressSpaceID target, Serializer &rez);
      void send_task_impl_semantic_request(AddressSpaceID target, 
                                           Serializer &rez);
      void send_index_space_semantic_request(AddressSpaceID target, 
                                             Serializer &rez);
      void send_index_partition_semantic_request(AddressSpaceID target,
                                                 Serializer &rez);
      void send_field_space_semantic_request(AddressSpaceID target,
                                             Serializer &rez);
      void send_field_semantic_request(AddressSpaceID target, Serializer &rez);
      void send_logical_region_semantic_request(AddressSpaceID target,
                                                Serializer &rez);
      void send_logical_partition_semantic_request(AddressSpaceID target,
                                                   Serializer &rez);
      void send_task_impl_semantic_info(AddressSpaceID target,
                                        Serializer &rez);
      void send_index_space_semantic_info(AddressSpaceID target, 
                                          Serializer &rez);
      void send_index_partition_semantic_info(AddressSpaceID target,
                                              Serializer &rez);
      void send_field_space_semantic_info(AddressSpaceID target,
                                          Serializer &rez);
      void send_field_semantic_info(AddressSpaceID target, Serializer &rez);
      void send_logical_region_semantic_info(AddressSpaceID target,
                                             Serializer &rez);
      void send_logical_partition_semantic_info(AddressSpaceID target,
                                                Serializer &rez);
      void send_remote_context_request(AddressSpaceID target, Serializer &rez);
      void send_remote_context_response(AddressSpaceID target, Serializer &rez);
      void send_remote_context_release(AddressSpaceID target, Serializer &rez);
      void send_remote_context_free(AddressSpaceID target, Serializer &rez);
      void send_remote_context_physical_request(AddressSpaceID target, 
                                                Serializer &rez);
      void send_remote_context_physical_response(AddressSpaceID target,
                                                 Serializer &rez);
      void send_version_owner_request(AddressSpaceID target, Serializer &rez);
      void send_version_owner_response(AddressSpaceID target, Serializer &rez);
      void send_version_state_response(AddressSpaceID target, Serializer &rez);
      void send_version_state_update_request(AddressSpaceID target, 
                                             Serializer &rez);
      void send_version_state_update_response(AddressSpaceID target, 
                                              Serializer &rez);
      void send_version_state_valid_notification(AddressSpaceID target,
                                                 Serializer &rez);
      void send_version_manager_advance(AddressSpaceID target, Serializer &rez);
      void send_version_manager_invalidate(AddressSpaceID target,
                                           Serializer &rez);
      void send_version_manager_request(AddressSpaceID target, Serializer &rez);
      void send_version_manager_response(AddressSpaceID target,Serializer &rez);
      void send_version_manager_unversioned_request(AddressSpaceID target,
                                                    Serializer &rez);
      void send_version_manager_unversioned_response(AddressSpaceID target,
                                                     Serializer &rez);
      void send_instance_request(AddressSpaceID target, Serializer &rez);
      void send_instance_response(AddressSpaceID target, Serializer &rez);
      void send_gc_priority_update(AddressSpaceID target, Serializer &rez);
      void send_never_gc_response(AddressSpaceID target, Serializer &rez);
      void send_acquire_request(AddressSpaceID target, Serializer &rez);
      void send_acquire_response(AddressSpaceID target, Serializer &rez);
      void send_variant_request(AddressSpaceID target, Serializer &rez);
      void send_variant_response(AddressSpaceID target, Serializer &rez);
      void send_variant_broadcast(AddressSpaceID target, Serializer &rez);
      void send_constraint_request(AddressSpaceID target, Serializer &rez);
      void send_constraint_response(AddressSpaceID target, Serializer &rez);
      void send_constraint_release(AddressSpaceID target, Serializer &rez);
      void send_constraint_removal(AddressSpaceID target, Serializer &rez);
      void send_mpi_rank_exchange(AddressSpaceID target, Serializer &rez);
      void send_shutdown_notification(AddressSpaceID target, Serializer &rez);
      void send_shutdown_response(AddressSpaceID target, Serializer &rez);
    public:
      // Complementary tasks for handling messages
      void handle_task(Deserializer &derez);
      void handle_steal(Deserializer &derez);
      void handle_advertisement(Deserializer &derez);
      void handle_index_space_node(Deserializer &derez, AddressSpaceID source);
      void handle_index_space_request(Deserializer &derez, 
                                      AddressSpaceID source);
      void handle_index_space_return(Deserializer &derez); 
      void handle_index_space_set(Deserializer &derez, AddressSpaceID source);
      void handle_index_space_child_request(Deserializer &derez, 
                                            AddressSpaceID source); 
      void handle_index_space_child_response(Deserializer &derez);
      void handle_index_space_colors_request(Deserializer &derez,
                                             AddressSpaceID source);
      void handle_index_space_colors_response(Deserializer &derez);
      void handle_index_partition_notification(Deserializer &derez);
      void handle_index_partition_node(Deserializer &derez,
                                       AddressSpaceID source);
      void handle_index_partition_request(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_index_partition_return(Deserializer &derez);
      void handle_index_partition_child_request(Deserializer &derez,
                                                AddressSpaceID source);
      void handle_index_partition_child_response(Deserializer &derez);
      void handle_field_space_node(Deserializer &derez, AddressSpaceID source);
      void handle_field_space_request(Deserializer &derez,
                                      AddressSpaceID source);
      void handle_field_space_return(Deserializer &derez);
      void handle_field_alloc_request(Deserializer &derez);
      void handle_field_alloc_notification(Deserializer &derez);
      void handle_field_space_top_alloc(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_field_free(Deserializer &derez, AddressSpaceID source);
      void handle_local_field_alloc_request(Deserializer &derez,
                                            AddressSpaceID source);
      void handle_local_field_alloc_response(Deserializer &derez);
      void handle_local_field_free(Deserializer &derez);
      void handle_local_field_update(Deserializer &derez);
      void handle_top_level_region_request(Deserializer &derez,
                                           AddressSpaceID source);
      void handle_top_level_region_return(Deserializer &derez);
      void handle_logical_region_node(Deserializer &derez, 
                                      AddressSpaceID source);
      void handle_index_space_destruction(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_index_partition_destruction(Deserializer &derez,
                                              AddressSpaceID source);
      void handle_field_space_destruction(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_logical_region_destruction(Deserializer &derez,
                                             AddressSpaceID source);
      void handle_logical_partition_destruction(Deserializer &derez,
                                                AddressSpaceID source);
      void handle_individual_remote_mapped(Deserializer &derez); 
      void handle_individual_remote_complete(Deserializer &derez);
      void handle_individual_remote_commit(Deserializer &derez);
      void handle_slice_remote_mapped(Deserializer &derez, 
                                      AddressSpaceID source);
      void handle_slice_remote_complete(Deserializer &derez);
      void handle_slice_remote_commit(Deserializer &derez);
      void handle_did_remote_registration(Deserializer &derez, 
                                          AddressSpaceID source);
      void handle_did_remote_valid_update(Deserializer &derez);
      void handle_did_remote_gc_update(Deserializer &derez);
      void handle_did_remote_resource_update(Deserializer &derez);
      void handle_did_remote_invalidate(Deserializer &derez);
      void handle_did_remote_deactivate(Deserializer &derez);
      void handle_did_create_add(Deserializer &derez);
      void handle_did_create_remove(Deserializer &derez);
      void handle_did_remote_unregister(Deserializer &derez);
      void handle_send_back_logical_state(Deserializer &derez);
      void handle_send_atomic_reservation_request(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_send_atomic_reservation_response(Deserializer &derez);
      void handle_send_materialized_view(Deserializer &derez, 
                                         AddressSpaceID source);
      void handle_send_composite_view(Deserializer &derez,
                                      AddressSpaceID source);
      void handle_send_fill_view(Deserializer &derez, AddressSpaceID source);
      void handle_send_phi_view(Deserializer &derez, AddressSpaceID source);
      void handle_send_reduction_view(Deserializer &derez,
                                      AddressSpaceID source);
      void handle_send_instance_manager(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_send_reduction_manager(Deserializer &derez,
                                         AddressSpaceID source);
      void handle_create_top_view_request(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_create_top_view_response(Deserializer &derez);
      void handle_subview_did_request(Deserializer &derez, 
                                      AddressSpaceID source);
      void handle_subview_did_response(Deserializer &derez);
      void handle_view_request(Deserializer &derez, AddressSpaceID source);
      void handle_view_update_request(Deserializer &derez, 
                                      AddressSpaceID source);
      void handle_view_update_response(Deserializer &derez);
      void handle_view_remote_update(Deserializer &derez,
                                     AddressSpaceID source);
      void handle_view_remote_invalidate(Deserializer &derez);
      void handle_manager_request(Deserializer &derez, AddressSpaceID source);
      void handle_future_result(Deserializer &derez);
      void handle_future_subscription(Deserializer &derez);
      void handle_future_map_future_request(Deserializer &derez,
                                            AddressSpaceID source);
      void handle_future_map_future_response(Deserializer &derez);
      void handle_mapper_message(Deserializer &derez);
      void handle_mapper_broadcast(Deserializer &derez);
      void handle_task_impl_semantic_request(Deserializer &derez,
                                             AddressSpaceID source);
      void handle_index_space_semantic_request(Deserializer &derez,
                                               AddressSpaceID source);
      void handle_index_partition_semantic_request(Deserializer &derez,
                                                   AddressSpaceID source);
      void handle_field_space_semantic_request(Deserializer &derez,
                                               AddressSpaceID source);
      void handle_field_semantic_request(Deserializer &derez,
                                         AddressSpaceID source);
      void handle_logical_region_semantic_request(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_logical_partition_semantic_request(Deserializer &derez,
                                                     AddressSpaceID source);
      void handle_task_impl_semantic_info(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_index_space_semantic_info(Deserializer &derez,
                                            AddressSpaceID source);
      void handle_index_partition_semantic_info(Deserializer &derez,
                                                AddressSpaceID source);
      void handle_field_space_semantic_info(Deserializer &derez,
                                            AddressSpaceID source);
      void handle_field_semantic_info(Deserializer &derez,
                                      AddressSpaceID source);
      void handle_logical_region_semantic_info(Deserializer &derez,
                                               AddressSpaceID source);
      void handle_logical_partition_semantic_info(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_remote_context_request(Deserializer &derez,
                                         AddressSpaceID source);
      void handle_remote_context_response(Deserializer &derez);
      void handle_remote_context_release(Deserializer &derez);
      void handle_remote_context_free(Deserializer &derez);
      void handle_remote_context_physical_request(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_remote_context_physical_response(Deserializer &derez);
      void handle_version_owner_request(Deserializer &derez, 
                                        AddressSpaceID source);
      void handle_version_owner_response(Deserializer &derez);
      void handle_version_state_request(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_version_state_response(Deserializer &derez,
                                         AddressSpaceID source);
      void handle_version_state_update_request(Deserializer &derez);
      void handle_version_state_update_response(Deserializer &derez);
      void handle_version_state_valid_notification(Deserializer &derez,
                                                   AddressSpaceID source);
      void handle_version_manager_advance(Deserializer &derez);
      void handle_version_manager_invalidate(Deserializer &derez);
      void handle_version_manager_request(Deserializer &derez, 
                                          AddressSpaceID source);
      void handle_version_manager_response(Deserializer &derez);
      void handle_version_manager_unversioned_request(Deserializer &derez,
                                                      AddressSpaceID source);
      void handle_version_manager_unversioned_response(Deserializer &derez);
      void handle_instance_request(Deserializer &derez, AddressSpaceID source);
      void handle_instance_response(Deserializer &derez,AddressSpaceID source);
      void handle_gc_priority_update(Deserializer &derez,AddressSpaceID source);
      void handle_never_gc_response(Deserializer &derez);
      void handle_acquire_request(Deserializer &derez, AddressSpaceID source);
      void handle_acquire_response(Deserializer &derez);
      void handle_variant_request(Deserializer &derez, AddressSpaceID source);
      void handle_variant_response(Deserializer &derez);
      void handle_variant_broadcast(Deserializer &derez);
      void handle_constraint_request(Deserializer &derez,AddressSpaceID source);
      void handle_constraint_response(Deserializer &derez,AddressSpaceID src);
      void handle_constraint_release(Deserializer &derez);
      void handle_constraint_removal(Deserializer &derez);
      void handle_top_level_task_request(Deserializer &derez);
      void handle_top_level_task_complete(Deserializer &derez);
      void handle_mpi_rank_exchange(Deserializer &derez);
      void handle_shutdown_notification(Deserializer &derez, 
                                        AddressSpaceID source);
      void handle_shutdown_response(Deserializer &derez);
    public: // Calls to handle mapper requests
      bool create_physical_instance(Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, MapperID mapper_id,
                                    Processor processor, bool acquire, 
                                    GCPriority priority, UniqueID creator_id);
      bool create_physical_instance(Memory target_memory, 
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, MapperID mapper_id,
                                    Processor processor, bool acquire, 
                                    GCPriority priority, UniqueID creator_id);
      bool find_or_create_physical_instance(Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    MapperID mapper_id, Processor processor,
                                    bool acquire, GCPriority priority,
                                    bool tight_bounds, UniqueID creator_id);
      bool find_or_create_physical_instance(Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    MapperID mapper_id, Processor processor,
                                    bool acquire, GCPriority priority,
                                    bool tight_bounds, UniqueID creator_id);
      bool find_physical_instance(Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_region_bounds);
      bool find_physical_instance(Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_region_bounds);
      void release_tree_instances(RegionTreeID tree_id);
    public:
      // Helper methods for the RegionTreeForest
      inline unsigned get_context_count(void) { return total_contexts; }
      inline unsigned get_start_color(void) const { return address_space; }
      inline unsigned get_color_modulus(void) const { return runtime_stride; }
    public:
      // Manage the execution of tasks within a context
      void activate_context(InnerContext *context);
      void deactivate_context(InnerContext *context);
    public:
      void add_to_dependence_queue(TaskContext *ctx, Processor p,Operation *op);
      void add_to_ready_queue(Processor p, TaskOp *task_op, 
                              RtEvent wait_on = RtEvent::NO_RT_EVENT);
      void add_to_local_queue(Processor p, Operation *op, LgPriority priority,
                              RtEvent wait_on = RtEvent::NO_RT_EVENT);
    public:
      inline Processor find_utility_group(void) { return utility_group; }
      Processor find_processor_group(const std::vector<Processor> &procs);
      ProcessorMask find_processor_mask(const std::vector<Processor> &procs);
      template<typename T>
      inline RtEvent issue_runtime_meta_task(const LgTaskArgs<T> &args,
                                   LgPriority lg_priority, Operation *op = NULL,
                                   RtEvent precondition = RtEvent::NO_RT_EVENT,
                                   Processor proc = Processor::NO_PROC);
    public:
      DistributedID get_available_distributed_id(void); 
      void free_distributed_id(DistributedID did);
      RtEvent recycle_distributed_id(DistributedID did, RtEvent recycle_event);
      AddressSpaceID determine_owner(DistributedID did) const;
    public:
      void register_distributed_collectable(DistributedID did,
                                            DistributedCollectable *dc);
      void unregister_distributed_collectable(DistributedID did);
      bool has_distributed_collectable(DistributedID did);
      DistributedCollectable* find_distributed_collectable(DistributedID did);
      DistributedCollectable* find_distributed_collectable(DistributedID did,
                                                           RtEvent &ready);
      DistributedCollectable* weak_find_distributed_collectable(
                                                           DistributedID did);
      bool find_pending_collectable_location(DistributedID did,void *&location);
    public:
      LogicalView* find_or_request_logical_view(DistributedID did,
                                                RtEvent &ready);
      PhysicalManager* find_or_request_physical_manager(DistributedID did, 
                                                        RtEvent &ready);
      VersionState* find_or_request_version_state(DistributedID did,
                                                  RtEvent &ready);
    protected:
      template<typename T, MessageKind MK, VirtualChannelKind VC>
      DistributedCollectable* find_or_request_distributed_collectable(
                                            DistributedID did, RtEvent &ready);
    public:
      FutureImpl* find_or_create_future(DistributedID did,
                                        ReferenceMutator *mutator);
      FutureMapImpl* find_or_create_future_map(DistributedID did, 
                      TaskContext *ctx, ReferenceMutator *mutator);
      IndexSpace find_or_create_index_launch_space(const Domain &launch_domain);
      IndexSpace find_or_create_index_launch_space(const Domain &launch_domain,
                                                   const void *realm_is,
                                                   TypeTag type_tag);
    public:
      void defer_collect_user(LogicalView *view, ApEvent term_event, 
                              ReferenceMutator *mutator);
      void complete_gc_epoch(GarbageCollectionEpoch *epoch);
    public:
      void increment_outstanding_top_level_tasks(void);
      void decrement_outstanding_top_level_tasks(void);
    public:
      void issue_runtime_shutdown_attempt(void);
      void initiate_runtime_shutdown(AddressSpaceID source, 
                                     ShutdownManager::ShutdownPhase phase,
                                     ShutdownManager *owner = NULL);
      void confirm_runtime_shutdown(ShutdownManager *shutdown_manager, 
                                    bool phase_one);
      void prepare_runtime_shutdown(void);
      void finalize_runtime_shutdown(void);
    public:
      bool has_outstanding_tasks(void);
#ifdef DEBUG_LEGION
      void increment_total_outstanding_tasks(unsigned tid, bool meta);
      void decrement_total_outstanding_tasks(unsigned tid, bool meta);
#else
      inline void increment_total_outstanding_tasks(void)
        { __sync_fetch_and_add(&total_outstanding_tasks,1); }
      inline void decrement_total_outstanding_tasks(void)
        { __sync_fetch_and_sub(&total_outstanding_tasks,1); }
#endif
    public:
      template<typename T>
      inline T* get_available(LocalLock &local_lock, std::deque<T*> &queue);

      template<bool CAN_BE_DELETED, typename T>
      inline void release_operation(std::deque<T*> &queue, T* operation);
    public:
      IndividualTask*       get_available_individual_task(void);
      PointTask*            get_available_point_task(void);
      IndexTask*            get_available_index_task(void);
      SliceTask*            get_available_slice_task(void);
      MapOp*                get_available_map_op(void);
      CopyOp*               get_available_copy_op(void);
      IndexCopyOp*          get_available_index_copy_op(void);
      PointCopyOp*          get_available_point_copy_op(void);
      FenceOp*              get_available_fence_op(void);
      FrameOp*              get_available_frame_op(void);
      DeletionOp*           get_available_deletion_op(void);
      OpenOp*               get_available_open_op(void);
      AdvanceOp*            get_available_advance_op(void);
      InterCloseOp*         get_available_inter_close_op(void);
      ReadCloseOp*          get_available_read_close_op(void);
      PostCloseOp*          get_available_post_close_op(void);
      VirtualCloseOp*       get_available_virtual_close_op(void);
      DynamicCollectiveOp*  get_available_dynamic_collective_op(void);
      FuturePredOp*         get_available_future_pred_op(void);
      NotPredOp*            get_available_not_pred_op(void);
      AndPredOp*            get_available_and_pred_op(void);
      OrPredOp*             get_available_or_pred_op(void);
      AcquireOp*            get_available_acquire_op(void);
      ReleaseOp*            get_available_release_op(void);
      TraceCaptureOp*       get_available_capture_op(void);
      TraceCompleteOp*      get_available_trace_op(void);
      MustEpochOp*          get_available_epoch_op(void);
      PendingPartitionOp*   get_available_pending_partition_op(void);
      DependentPartitionOp* get_available_dependent_partition_op(void);
      PointDepPartOp*       get_available_point_dep_part_op(void);
      FillOp*               get_available_fill_op(void);
      IndexFillOp*          get_available_index_fill_op(void);
      PointFillOp*          get_available_point_fill_op(void);
      AttachOp*             get_available_attach_op(void);
      DetachOp*             get_available_detach_op(void);
      TimingOp*             get_available_timing_op(void);
    public:
      void free_individual_task(IndividualTask *task);
      void free_point_task(PointTask *task);
      void free_index_task(IndexTask *task);
      void free_slice_task(SliceTask *task);
      void free_map_op(MapOp *op);
      void free_copy_op(CopyOp *op);
      void free_index_copy_op(IndexCopyOp *op);
      void free_point_copy_op(PointCopyOp *op);
      void free_fence_op(FenceOp *op);
      void free_frame_op(FrameOp *op);
      void free_deletion_op(DeletionOp *op);
      void free_open_op(OpenOp *op);
      void free_advance_op(AdvanceOp *op);
      void free_inter_close_op(InterCloseOp *op); 
      void free_read_close_op(ReadCloseOp *op);
      void free_post_close_op(PostCloseOp *op);
      void free_virtual_close_op(VirtualCloseOp *op);
      void free_dynamic_collective_op(DynamicCollectiveOp *op);
      void free_future_predicate_op(FuturePredOp *op);
      void free_not_predicate_op(NotPredOp *op);
      void free_and_predicate_op(AndPredOp *op);
      void free_or_predicate_op(OrPredOp *op);
      void free_acquire_op(AcquireOp *op);
      void free_release_op(ReleaseOp *op);
      void free_capture_op(TraceCaptureOp *op);
      void free_trace_op(TraceCompleteOp *op);
      void free_epoch_op(MustEpochOp *op);
      void free_pending_partition_op(PendingPartitionOp *op);
      void free_dependent_partition_op(DependentPartitionOp* op);
      void free_point_dep_part_op(PointDepPartOp *op);
      void free_fill_op(FillOp *op);
      void free_index_fill_op(IndexFillOp *op);
      void free_point_fill_op(PointFillOp *op);
      void free_attach_op(AttachOp *op);
      void free_detach_op(DetachOp *op);
      void free_timing_op(TimingOp *op);
    public:
      RegionTreeContext allocate_region_tree_context(void);
      void free_region_tree_context(RegionTreeContext tree_ctx); 
      void register_local_context(UniqueID context_uid, InnerContext *ctx);
      void unregister_local_context(UniqueID context_uid);
      void register_remote_context(UniqueID context_uid, RemoteContext *ctx,
                                   std::set<RtEvent> &preconditions);
      void unregister_remote_context(UniqueID context_uid);
      InnerContext* find_context(UniqueID context_uid, 
                                 bool return_null_if_not_found = false);
      inline AddressSpaceID get_runtime_owner(UniqueID uid) const
        { return (uid % runtime_stride); }
    public:
      bool is_local(Processor proc) const;
      void find_visible_memories(Processor proc, std::set<Memory> &visible);
    public:
      IndexSpaceID       get_unique_index_space_id(void);
      IndexPartitionID   get_unique_index_partition_id(void);
      FieldSpaceID       get_unique_field_space_id(void);
      IndexTreeID        get_unique_index_tree_id(void);
      RegionTreeID       get_unique_region_tree_id(void);
      UniqueID           get_unique_operation_id(void);
      FieldID            get_unique_field_id(void);
      CodeDescriptorID   get_unique_code_descriptor_id(void);
      LayoutConstraintID get_unique_constraint_id(void);
    public:
      // Verify that a region requirement is valid
      LegionErrorType verify_requirement(const RegionRequirement &req,
                                         FieldID &bad_field);
    public:
      // Methods for helping with dumb nested class scoping problems
      Future help_create_future(Operation *op = NULL);
      void help_complete_future(const Future &f);
      bool help_reset_future(const Future &f);
    public:
      unsigned generate_random_integer(void);
#ifdef TRACE_ALLOCATION
    public:
      void trace_allocation(AllocationType type, size_t size, int elems);
      void trace_free(AllocationType type, size_t size, int elems);
      void dump_allocation_info(void);
      static const char* get_allocation_name(AllocationType type);
#endif
    public:
      // These are the static methods that become the meta-tasks
      // for performing all the needed runtime operations
      static void initialize_runtime(
                          const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
      static void shutdown_runtime(
                          const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
      static void legion_runtime_task(
                          const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
      static void profiling_runtime_task(
                          const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
      static void launch_top_level(
                          const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
      static void init_mpi_interop(const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
      static void startup_sync(const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
      static void dummy_barrier(const void *args, size_t arglen,
                          const void *userdata, size_t userlen,
                          Processor p);
    protected:
      static void configure_collective_settings(int total_spaces);
    protected:
      // Internal runtime methods invoked by the above static methods
      // after the find the right runtime instance to call
      void process_schedule_request(Processor p);
      void process_message_task(const void *args, size_t arglen);
    public:
      // The Runtime wrapper for this class
      Legion::Runtime *const external;
      // The Mapper Runtime for this class
      Legion::Mapping::MapperRuntime *const mapper_runtime;
      // The machine object for this runtime
      const Machine machine;
      const AddressSpaceID address_space; 
      const unsigned total_address_spaces;
      const unsigned runtime_stride; // stride for uniqueness
      LegionProfiler *profiler;
      RegionTreeForest *const forest;
      Processor utility_group;
      const bool has_explicit_utility_procs;
    protected:
      bool prepared_for_shutdown;
    protected:
#ifdef DEBUG_LEGION
      mutable LocalLock outstanding_task_lock;
      std::map<std::pair<unsigned,bool>,unsigned> outstanding_task_counts;
#endif
      unsigned total_outstanding_tasks;
      unsigned outstanding_top_level_tasks;
#ifdef DEBUG_SHUTDOWN_HANG
    public:
      std::vector<int> outstanding_counts;
#endif
    protected:
      // Internal runtime state 
      // The local processor managed by this runtime
      const std::set<Processor> local_procs;
      // The local utility processors owned by this runtime
      const std::set<Processor> local_utils;
      // Processor managers for each of the local processors
      std::map<Processor,ProcessorManager*> proc_managers;
      // Lock for looking up memory managers
      mutable LocalLock memory_manager_lock;
      // Lock for initializing message managers
      mutable LocalLock message_manager_lock;
      // Memory managers for all the memories we know about
      std::map<Memory,MemoryManager*> memory_managers;
      // Message managers for each of the other runtimes
      MessageManager *message_managers[MAX_NUM_NODES];
      // For every processor map it to its address space
      const std::map<Processor,AddressSpaceID> proc_spaces;
    protected:
      // The task table 
      mutable LocalLock task_variant_lock;
      std::map<TaskID,TaskImpl*> task_table;
      std::deque<VariantImpl*> variant_table;
    protected:
      // Constraint sets
      mutable LocalLock layout_constraints_lock;
      std::map<LayoutConstraintID,LayoutConstraints*> layout_constraints_table;
      std::map<LayoutConstraintID,RtEvent> pending_constraint_requests;
    protected:
      struct MapperInfo {
        MapperInfo(void)
          : proc(Processor::NO_PROC), map_id(0) { }
        MapperInfo(Processor p, MapperID mid)
          : proc(p), map_id(mid) { }
      public:
        Processor proc;
        MapperID map_id;
      };
      mutable LocalLock mapper_info_lock;
      // For every mapper remember its mapper ID and processor
      std::map<Mapper*,MapperInfo> mapper_infos;
#ifdef DEBUG_LEGION
    protected:
      friend class TreeStateLogger;
      TreeStateLogger *get_tree_state_logger(void) { return tree_state_logger; }
#endif
    protected:
      unsigned unique_index_space_id;
      unsigned unique_index_partition_id;
      unsigned unique_field_space_id;
      unsigned unique_index_tree_id;
      unsigned unique_region_tree_id;
      unsigned unique_operation_id;
      unsigned unique_field_id; 
      unsigned unique_code_descriptor_id;
      unsigned unique_constraint_id;
      unsigned unique_task_id;
      unsigned unique_mapper_id;
      unsigned unique_projection_id;
    protected:
      mutable LocalLock projection_lock;
      std::map<ProjectionID,ProjectionFunction*> projection_functions;
    protected:
      mutable LocalLock group_lock;
      LegionMap<uint64_t,LegionDeque<ProcessorGroupInfo>::aligned,
                PROCESSOR_GROUP_ALLOC>::tracked processor_groups;
    protected:
      mutable LocalLock processor_mapping_lock;
      std::map<Processor,unsigned> processor_mapping;
    protected:
      mutable LocalLock distributed_id_lock;
      DistributedID unique_distributed_id;
      LegionDeque<DistributedID,
          RUNTIME_DISTRIBUTED_ALLOC>::tracked available_distributed_ids;
    protected:
      mutable LocalLock distributed_collectable_lock;
      LegionMap<DistributedID,DistributedCollectable*,
                RUNTIME_DIST_COLLECT_ALLOC>::tracked dist_collectables;
      std::map<DistributedID,
        std::pair<DistributedCollectable*,RtUserEvent> > pending_collectables;
    protected:
      mutable LocalLock is_launch_lock;
      std::map<std::pair<Domain,TypeTag>,IndexSpace> index_launch_spaces;
    protected:
      mutable LocalLock gc_epoch_lock;
      GarbageCollectionEpoch *current_gc_epoch;
      LegionSet<GarbageCollectionEpoch*,
                RUNTIME_GC_EPOCH_ALLOC>::tracked  pending_gc_epochs;
      unsigned gc_epoch_counter;
    protected:
      // The runtime keeps track of remote contexts so they
      // can be re-used by multiple tasks that get sent remotely
      mutable LocalLock context_lock;
      std::map<UniqueID,InnerContext*> local_contexts;
      LegionMap<UniqueID,RemoteContext*,
                RUNTIME_REMOTE_ALLOC>::tracked remote_contexts;
      std::map<UniqueID,RtUserEvent> pending_remote_contexts;
      unsigned total_contexts;
      std::deque<RegionTreeContext> available_contexts;
    protected:
      // For generating random numbers
      mutable LocalLock random_lock;
      unsigned short random_state[3];
#ifdef TRACE_ALLOCATION
    protected:
      struct AllocationTracker {
      public:
        AllocationTracker(void)
          : total_allocations(0), total_bytes(0),
            diff_allocations(0), diff_bytes(0) { }
      public:
        unsigned total_allocations;
        size_t         total_bytes;
        int       diff_allocations;
        off_t           diff_bytes;
      };
      mutable LocalLock allocation_lock; // leak this lock intentionally
      // Make these static so they live through the end of the runtime
      static std::map<AllocationType,AllocationTracker> allocation_manager;
      static unsigned long long allocation_tracing_count;
#endif
    protected:
      mutable LocalLock individual_task_lock;
      mutable LocalLock point_task_lock;
      mutable LocalLock index_task_lock;
      mutable LocalLock slice_task_lock;
      mutable LocalLock map_op_lock;
      mutable LocalLock copy_op_lock;
      mutable LocalLock fence_op_lock;
      mutable LocalLock frame_op_lock;
      mutable LocalLock deletion_op_lock;
      mutable LocalLock open_op_lock;
      mutable LocalLock advance_op_lock;
      mutable LocalLock inter_close_op_lock;
      mutable LocalLock read_close_op_lock;
      mutable LocalLock post_close_op_lock;
      mutable LocalLock virtual_close_op_lock;
      mutable LocalLock dynamic_collective_op_lock;
      mutable LocalLock future_pred_op_lock;
      mutable LocalLock not_pred_op_lock;
      mutable LocalLock and_pred_op_lock;
      mutable LocalLock or_pred_op_lock;
      mutable LocalLock acquire_op_lock;
      mutable LocalLock release_op_lock;
      mutable LocalLock capture_op_lock;
      mutable LocalLock trace_op_lock;
      mutable LocalLock epoch_op_lock;
      mutable LocalLock pending_partition_op_lock;
      mutable LocalLock dependent_partition_op_lock;
      mutable LocalLock fill_op_lock;
      mutable LocalLock attach_op_lock;
      mutable LocalLock detach_op_lock;
      mutable LocalLock timing_op_lock;
    protected:
      std::deque<IndividualTask*>       available_individual_tasks;
      std::deque<PointTask*>            available_point_tasks;
      std::deque<IndexTask*>            available_index_tasks;
      std::deque<SliceTask*>            available_slice_tasks;
      std::deque<MapOp*>                available_map_ops;
      std::deque<CopyOp*>               available_copy_ops;
      std::deque<IndexCopyOp*>          available_index_copy_ops;
      std::deque<PointCopyOp*>          available_point_copy_ops;
      std::deque<FenceOp*>              available_fence_ops;
      std::deque<FrameOp*>              available_frame_ops;
      std::deque<DeletionOp*>           available_deletion_ops;
      std::deque<OpenOp*>               available_open_ops;
      std::deque<AdvanceOp*>            available_advance_ops;
      std::deque<InterCloseOp*>         available_inter_close_ops;
      std::deque<ReadCloseOp*>          available_read_close_ops;
      std::deque<PostCloseOp*>          available_post_close_ops;
      std::deque<VirtualCloseOp*>       available_virtual_close_ops;
      std::deque<DynamicCollectiveOp*>  available_dynamic_collective_ops;
      std::deque<FuturePredOp*>         available_future_pred_ops;
      std::deque<NotPredOp*>            available_not_pred_ops;
      std::deque<AndPredOp*>            available_and_pred_ops;
      std::deque<OrPredOp*>             available_or_pred_ops;
      std::deque<AcquireOp*>            available_acquire_ops;
      std::deque<ReleaseOp*>            available_release_ops;
      std::deque<TraceCaptureOp*>       available_capture_ops;
      std::deque<TraceCompleteOp*>      available_trace_ops;
      std::deque<MustEpochOp*>          available_epoch_ops;
      std::deque<PendingPartitionOp*>   available_pending_partition_ops;
      std::deque<DependentPartitionOp*> available_dependent_partition_ops;
      std::deque<PointDepPartOp*>       available_point_dep_part_ops;
      std::deque<FillOp*>               available_fill_ops;
      std::deque<IndexFillOp*>          available_index_fill_ops;
      std::deque<PointFillOp*>          available_point_fill_ops;
      std::deque<AttachOp*>             available_attach_ops;
      std::deque<DetachOp*>             available_detach_ops;
      std::deque<TimingOp*>             available_timing_ops;
#ifdef DEBUG_LEGION
      TreeStateLogger *tree_state_logger;
      // For debugging purposes keep track of
      // some of the outstanding tasks
      std::set<IndividualTask*> out_individual_tasks;
      std::set<PointTask*>      out_point_tasks;
      std::set<IndexTask*>      out_index_tasks;
      std::set<SliceTask*>      out_slice_tasks;
      std::set<MustEpochOp*>    out_must_epoch;
    public:
      // These are debugging method for the above data
      // structures.  They are not called anywhere in
      // actual code.
      void print_out_individual_tasks(FILE *f = stdout, int cnt = -1);
      void print_out_index_tasks(FILE *f = stdout, int cnt = -1);
      void print_out_slice_tasks(FILE *f = stdout, int cnt = -1);
      void print_out_point_tasks(FILE *f = stdout, int cnt = -1);
      void print_outstanding_tasks(FILE *f = stdout, int cnt = -1);
#endif
    public:
      LayoutConstraintID register_layout(
          const LayoutConstraintRegistrar &registrar, 
          LayoutConstraintID id);
      LayoutConstraints* register_layout(FieldSpace handle,
                                         const LayoutConstraintSet &cons);
      bool register_layout(LayoutConstraints *new_constraints);
      void release_layout(LayoutConstraintID layout_id);
      void unregister_layout(LayoutConstraintID layout_id);
      static LayoutConstraintID preregister_layout(
                                     const LayoutConstraintRegistrar &registrar,
                                     LayoutConstraintID layout_id);
      FieldSpace get_layout_constraint_field_space(LayoutConstraintID id);
      void get_layout_constraints(LayoutConstraintID layout_id,
                                  LayoutConstraintSet &layout_constraints);
      const char* get_layout_constraints_name(LayoutConstraintID layout_id);
      LayoutConstraints* find_layout_constraints(LayoutConstraintID layout_id,
                                                 bool can_fail = false);
    public:
      // Static methods for start-up and callback phases
      static int start(int argc, char **argv, bool background);
      static void wait_for_shutdown(void);
      static void set_top_level_task_id(Processor::TaskFuncID top_id);
      static void configure_MPI_interoperability(int rank);
      static void register_handshake(MPILegionHandshake &handshake);
      static const ReductionOp* get_reduction_op(ReductionOpID redop_id);
      static const SerdezOp* get_serdez_op(CustomSerdezID serdez_id);
      static const SerdezRedopFns* get_serdez_redop_fns(ReductionOpID redop_id);
      static void add_registration_callback(RegistrationCallbackFnptr callback);
      static InputArgs& get_input_args(void);
      static Runtime* get_runtime(Processor p);
      static ReductionOpTable& get_reduction_table(void);
      static SerdezOpTable& get_serdez_table(void);
      static SerdezRedopTable& get_serdez_redop_table(void);
      static std::deque<PendingVariantRegistration*>&
                                get_pending_variant_table(void);
      static std::map<LayoutConstraintID,LayoutConstraintRegistrar>&
                                get_pending_constraint_table(void);
      static std::map<ProjectionID,ProjectionFunctor*>&
                                get_pending_projection_table(void);
      static TaskID& get_current_static_task_id(void);
      static TaskID generate_static_task_id(void);
      static VariantID preregister_variant(
                      const TaskVariantRegistrar &registrar,
                      const void *user_data, size_t user_data_size,
                      CodeDescriptor *realm_desc, bool has_ret, 
                      const char *task_name,VariantID vid,bool check_id = true);
    public:
      static void report_fatal_message(int code,
                                       const char *file_name,
                                       const int line_number,
                                       const char *message);
      static void report_error_message(int code,
                                       const char *file_name,
                                       const int line_number,
                                       const char *message);
      static void report_warning_message(int code,
                                         const char *file_name, 
                                         const int line_number,
                                         const char *message);
#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
    public:
      static const char* find_privilege_task_name(void *impl);
#endif
#ifdef BOUNDS_CHECKS
    public:
      static void check_bounds(void *impl, ptr_t ptr);
      static void check_bounds(void *impl, const DomainPoint &dp);
#endif
    private:
      static RtEvent register_runtime_tasks(RealmRuntime &realm);
      static Processor::TaskFuncID get_next_available_id(void);
      static void log_machine(Machine machine);
    public:
      // Static member variables
      static Runtime *the_runtime;
      // the runtime map is only valid when running with -lg:separate
      static std::map<Processor,Runtime*> *runtime_map;
      static std::vector<RegistrationCallbackFnptr> registration_callbacks;
      static Processor::TaskFuncID legion_main_id;
      static int initial_task_window_size;
      static unsigned initial_task_window_hysteresis;
      static unsigned initial_tasks_to_schedule;
      static unsigned max_message_size;
      static unsigned gc_epoch_size;
      static unsigned max_local_fields;
      static bool runtime_started;
      static bool runtime_backgrounded;
      static bool runtime_warnings;
      static bool separate_runtime_instances;
      static bool record_registration;
      static bool stealing_disabled;
      static bool resilient_mode;
      static bool unsafe_launch;
      static bool unsafe_mapper;
      static bool dynamic_independence_tests;
      static bool legion_spy_enabled;
      static bool enable_test_mapper;
      static bool legion_ldb_enabled;
      static const char* replay_file;
      // Collective settings
      static int legion_collective_radix;
      static int legion_collective_log_radix;
      static int legion_collective_stages;
      static int legion_collective_participating_spaces;
      static int legion_collective_last_radix;
      static int legion_collective_last_log_radix;
      // MPI Interoperability
      static int mpi_rank;
      static MPIRankTable *mpi_rank_table;
      static std::vector<MPILegionHandshake> *pending_handshakes;
#ifdef DEBUG_LEGION
      static bool logging_region_tree_state;
      static bool verbose_logging;
      static bool logical_logging_only;
      static bool physical_logging_only;
      static bool check_privileges;
      static bool bit_mask_logging;
#endif
      static bool program_order_execution;
      static bool verify_disjointness;
    public:
      static unsigned num_profiling_nodes;
      static const char* serializer_type;
      static const char* prof_logfile;
      static size_t prof_footprint_threshold;
      static size_t prof_target_latency;
      static bool slow_debug_ok;
    public:
      static inline ApEvent merge_events(ApEvent e1, ApEvent e2);
      static inline ApEvent merge_events(ApEvent e1, ApEvent e2, ApEvent e3);
      static inline ApEvent merge_events(const std::set<ApEvent> &events);
    public:
      static inline RtEvent merge_events(RtEvent e1, RtEvent e2);
      static inline RtEvent merge_events(RtEvent e1, RtEvent e2, RtEvent e3);
      static inline RtEvent merge_events(const std::set<RtEvent> &events);
    public:
      static inline ApUserEvent create_ap_user_event(void);
      static inline void trigger_event(ApUserEvent to_trigger,
                                   ApEvent precondition = ApEvent::NO_AP_EVENT);
      static inline void poison_event(ApUserEvent to_poison);
    public:
      static inline RtUserEvent create_rt_user_event(void);
      static inline void trigger_event(RtUserEvent to_trigger,
                                   RtEvent precondition = RtEvent::NO_RT_EVENT);
      static inline void poison_event(RtUserEvent to_poison);
    public:
      static inline PredEvent create_pred_event(void);
      static inline void trigger_event(PredEvent to_trigger);
      static inline void poison_event(PredEvent to_poison);
    public:
      static inline ApEvent ignorefaults(Realm::Event e);
      static inline RtEvent protect_event(ApEvent to_protect);
      static inline RtEvent protect_merge_events(
                                          const std::set<ApEvent> &events);
    public:
      static inline void phase_barrier_arrive(const PhaseBarrier &bar, 
                unsigned cnt, ApEvent precondition = ApEvent::NO_AP_EVENT,
                const void *reduce_value = NULL, size_t reduce_value_size = 0);
      static inline ApBarrier get_previous_phase(const PhaseBarrier &bar);
      static inline void alter_arrival_count(PhaseBarrier &bar, int delta);
      static inline void advance_barrier(PhaseBarrier &bar);
      static inline bool get_barrier_result(ApBarrier bar, void *result,
                                            size_t result_size);
    public:
      static inline ApEvent acquire_ap_reservation(Reservation r,bool exclusive,
                                   ApEvent precondition = ApEvent::NO_AP_EVENT);
      static inline RtEvent acquire_rt_reservation(Reservation r,bool exclusive,
                                   RtEvent precondition = RtEvent::NO_RT_EVENT);
      static inline void release_reservation(Reservation r,
                                   LgEvent precondition = LgEvent::NO_LG_EVENT);
    };

    //--------------------------------------------------------------------------
    template<typename T>
    inline T* Runtime::get_available(LocalLock &local_lock, 
                                     std::deque<T*> &queue)
    //--------------------------------------------------------------------------
    {
      T *result = NULL;
      {
        AutoLock l_lock(local_lock);
        if (!queue.empty())
        {
          result = queue.front();
          queue.pop_front();
        }
      }
      // Couldn't find one so make one
      if (result == NULL)
        result = new T(this);
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    template<bool CAN_BE_DELETED, typename T>
    inline void Runtime::release_operation(std::deque<T*> &queue, T* operation)
    //--------------------------------------------------------------------------
    {
      if (CAN_BE_DELETED && (queue.size() == LEGION_MAX_RECYCLABLE_OBJECTS))
        delete (operation);
      else
        queue.push_front(operation);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline RtEvent Runtime::issue_runtime_meta_task(const LgTaskArgs<T> &args,
                                      LgPriority priority, Operation *op, 
                                      RtEvent precondition, Processor target)
    //--------------------------------------------------------------------------
    {
      // If this is not a task directly related to shutdown or is a message, 
      // to a remote node then increment the number of outstanding tasks
#ifdef DEBUG_LEGION
      if (T::TASK_ID < LG_MESSAGE_ID)
        increment_total_outstanding_tasks(args.lg_task_id, true/*meta*/);
#else
      if (T::TASK_ID < LG_MESSAGE_ID)
        increment_total_outstanding_tasks();
#endif
#ifdef DEBUG_SHUTDOWN_HANG
      __sync_fetch_and_add(&outstanding_counts[T::TASK_ID],1);
#endif
      if (!target.exists())
      {
        // If we don't have a processor to explicitly target, figure
        // out which of our utility processors to use
        target = utility_group;
      }
#ifdef DEBUG_LEGION
      assert(target.exists());
#endif
      DETAILED_PROFILER(this, REALM_SPAWN_META_CALL);
      if ((T::TASK_ID < LG_MESSAGE_ID) && (profiler != NULL))
      {
        Realm::ProfilingRequestSet requests;
        profiler->add_meta_request(requests, T::TASK_ID, op);
        return RtEvent(target.spawn(LG_TASK_ID, &args, sizeof(T),
                                    requests, precondition, priority));
      }
      else
        return RtEvent(target.spawn(LG_TASK_ID, &args, sizeof(T), 
                                    precondition, priority));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApEvent Runtime::merge_events(ApEvent e1, ApEvent e2)
    //--------------------------------------------------------------------------
    {
      ApEvent result(Realm::Event::merge_events(e1, e2)); 
#ifdef LEGION_SPY
      if (!result.exists() || (result == e1) || (result == e2))
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        if (result == e1)
          rename.trigger(e1);
        else if (result == e2)
          rename.trigger(e2);
        else
          rename.trigger();
        result = ApEvent(rename);
      }
      LegionSpy::log_event_dependence(e1, result);
      LegionSpy::log_event_dependence(e2, result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApEvent Runtime::merge_events(ApEvent e1, 
                                                    ApEvent e2, ApEvent e3) 
    //--------------------------------------------------------------------------
    {
      ApEvent result(Realm::Event::merge_events(e1, e2, e3)); 
#ifdef LEGION_SPY
      if (!result.exists() || (result == e1) || (result == e2) ||(result == e3))
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        if (result == e1)
          rename.trigger(e1);
        else if (result == e2)
          rename.trigger(e2);
        else if (result == e3)
          rename.trigger(e3);
        else
          rename.trigger();
        result = ApEvent(rename);
      }
      LegionSpy::log_event_dependence(e1, result);
      LegionSpy::log_event_dependence(e2, result);
      LegionSpy::log_event_dependence(e3, result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApEvent Runtime::merge_events(
                                                const std::set<ApEvent> &events)
    //--------------------------------------------------------------------------
    {
#ifndef LEGION_SPY
      if (events.empty())
        return ApEvent::NO_AP_EVENT;
      if (events.size() == 1)
        return *(events.begin());
#endif
      const std::set<Realm::Event> *realm_events = 
        reinterpret_cast<const std::set<Realm::Event>*>(&events);
      ApEvent result(Realm::Event::merge_events(*realm_events));
#ifdef LEGION_SPY
      if (!result.exists() || (events.find(result) != events.end()))
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        if (events.find(result) != events.end())
          rename.trigger(result);
        else
          rename.trigger();
        result = ApEvent(rename);
      }
      for (std::set<ApEvent>::const_iterator it = events.begin();
            it != events.end(); it++)
        LegionSpy::log_event_dependence(*it, result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtEvent Runtime::merge_events(RtEvent e1, RtEvent e2)
    //--------------------------------------------------------------------------
    {
      // No logging for runtime operations currently
      return RtEvent(Realm::Event::merge_events(e1, e2)); 
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtEvent Runtime::merge_events(RtEvent e1, 
                                                    RtEvent e2, RtEvent e3) 
    //--------------------------------------------------------------------------
    {
      // No logging for runtime operations currently
      return RtEvent(Realm::Event::merge_events(e1, e2, e3)); 
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtEvent Runtime::merge_events(
                                                const std::set<RtEvent> &events)
    //--------------------------------------------------------------------------
    {
#ifndef LEGION_SPY
      if (events.empty())
        return RtEvent::NO_RT_EVENT;
      if (events.size() == 1)
        return *(events.begin());
#endif
      // No logging for runtime operations currently
      const std::set<Realm::Event> *realm_events = 
        reinterpret_cast<const std::set<Realm::Event>*>(&events);
      return RtEvent(Realm::Event::merge_events(*realm_events));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApUserEvent Runtime::create_ap_user_event(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      ApUserEvent result(Realm::UserEvent::create_user_event());
      LegionSpy::log_ap_user_event(result);
      return result;
#else
      return ApUserEvent(Realm::UserEvent::create_user_event());
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::trigger_event(ApUserEvent to_trigger,
                                                  ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      Realm::UserEvent copy = to_trigger;
      copy.trigger(precondition);
#ifdef LEGION_SPY
      LegionSpy::log_ap_user_event_trigger(to_trigger);
      if (precondition.exists())
        LegionSpy::log_event_dependence(precondition, to_trigger);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::poison_event(ApUserEvent to_poison)
    //--------------------------------------------------------------------------
    {
      Realm::UserEvent copy = to_poison;
      copy.cancel();
#ifdef LEGION_SPY
      // This counts as triggering
      LegionSpy::log_ap_user_event_trigger(to_poison);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtUserEvent Runtime::create_rt_user_event(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      RtUserEvent result(Realm::UserEvent::create_user_event());
      LegionSpy::log_rt_user_event(result);
      return result;
#else
      return RtUserEvent(Realm::UserEvent::create_user_event());
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::trigger_event(RtUserEvent to_trigger,
                                                  RtEvent precondition) 
    //--------------------------------------------------------------------------
    {
      Realm::UserEvent copy = to_trigger;
      copy.trigger(precondition);
#ifdef LEGION_SPY
      LegionSpy::log_rt_user_event_trigger(to_trigger);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::poison_event(RtUserEvent to_poison)
    //--------------------------------------------------------------------------
    {
      Realm::UserEvent copy = to_poison;
      copy.cancel();
#ifdef LEGION_SPY
      // This counts as triggering
      LegionSpy::log_rt_user_event_trigger(to_poison);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline PredEvent Runtime::create_pred_event(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      PredEvent result(Realm::UserEvent::create_user_event());
      LegionSpy::log_pred_event(result);
      return result;
#else
      return PredEvent(Realm::UserEvent::create_user_event());
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::trigger_event(PredEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      Realm::UserEvent copy = to_trigger;
      copy.trigger();
#ifdef LEGION_SPY
      LegionSpy::log_pred_event_trigger(to_trigger);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::poison_event(PredEvent to_poison)
    //--------------------------------------------------------------------------
    {
      Realm::UserEvent copy = to_poison;
      copy.cancel();
#ifdef LEGION_SPY
      // This counts as triggering
      LegionSpy::log_pred_event_trigger(to_poison);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApEvent Runtime::ignorefaults(Realm::Event e)
    //--------------------------------------------------------------------------
    {
      ApEvent result(Realm::Event::ignorefaults(e));
#ifdef LEGION_SPY
      if (!result.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        result = ApEvent(rename);
      }
      LegionSpy::log_event_dependence(ApEvent(e), result);
#endif
      return ApEvent(result);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtEvent Runtime::protect_event(ApEvent to_protect)
    //--------------------------------------------------------------------------
    {
      if (to_protect.exists())
        return RtEvent(Realm::Event::ignorefaults(to_protect));
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtEvent Runtime::protect_merge_events(
                                                const std::set<ApEvent> &events)
    //--------------------------------------------------------------------------
    {
      const std::set<Realm::Event> *realm_events = 
        reinterpret_cast<const std::set<Realm::Event>*>(&events);
      return RtEvent(Realm::Event::merge_events_ignorefaults(*realm_events));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::phase_barrier_arrive(
                  const PhaseBarrier &bar, unsigned count, ApEvent precondition,
                  const void *reduce_value, size_t reduce_value_size)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar.phase_barrier;
      copy.arrive(count, precondition, reduce_value, reduce_value_size);
#ifdef LEGION_SPY
      if (precondition.exists())
        LegionSpy::log_event_dependence(precondition, bar.phase_barrier);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApBarrier Runtime::get_previous_phase(
                                                        const PhaseBarrier &bar)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar.phase_barrier;
      return ApBarrier(copy.get_previous_phase());
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::alter_arrival_count(PhaseBarrier &bar,
                                                        int delta)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar.phase_barrier;
      bar.phase_barrier = ApBarrier(copy.alter_arrival_count(delta));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::advance_barrier(PhaseBarrier &bar)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar.phase_barrier;
      bar.phase_barrier = ApBarrier(copy.advance_barrier());
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool Runtime::get_barrier_result(ApBarrier bar,
                                               void *result, size_t result_size)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      return copy.get_result(result, result_size);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApEvent Runtime::acquire_ap_reservation(Reservation r,
                                           bool exclusive, ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      ApEvent result(r.acquire(exclusive ? 0 : 1, exclusive, precondition));
#ifdef LEGION_SPY
      if (precondition.exists() && !result.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        result = ApEvent(rename);
      }
      if (precondition.exists())
        LegionSpy::log_event_dependence(precondition, result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtEvent Runtime::acquire_rt_reservation(Reservation r,
                                           bool exclusive, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      return RtEvent(r.acquire(exclusive ? 0 : 1, exclusive, precondition)); 
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::release_reservation(Reservation r,
                                                           LgEvent precondition)
    //--------------------------------------------------------------------------
    {
      r.release(precondition);
    }

  }; // namespace Internal 
}; // namespace Legion 

#endif // __RUNTIME_H__

// EOF

