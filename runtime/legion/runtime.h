/* Copyright 2020 Stanford University, NVIDIA Corporation
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
#include "legion/legion_analysis.h"
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
      void set_point(const DomainPoint &point, const Future &f, bool replace);
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
     * \class FieldAllocatorImpl
     * The base implementation of a field allocator object. This
     * tracks how many outstanding copies of a field allocator
     * object there are for a task and once they've all been
     * destroyed it informs the context that there are no more
     * outstanding allocations.
     */
    class FieldAllocatorImpl : public Collectable {
    public:
      FieldAllocatorImpl(FieldSpace space, TaskContext *context);
      FieldAllocatorImpl(const FieldAllocatorImpl &rhs);
      ~FieldAllocatorImpl(void);
    public:
      FieldAllocatorImpl& operator=(const FieldAllocatorImpl &rhs);
    public:
      inline FieldSpace get_field_space(void) const { return field_space; }
    public:
      FieldID allocate_field(size_t field_size, 
                             FieldID desired_fieldid,
                             CustomSerdezID serdez_id, bool local);
      void free_field(FieldID fid, const bool unordered);
    public:
      void allocate_fields(const std::vector<size_t> &field_sizes,
                           std::vector<FieldID> &resulting_fields,
                           CustomSerdezID serdez_id, bool local);
      void free_fields(const std::set<FieldID> &to_free, const bool unordered);
    public:
      const FieldSpace field_space;
      TaskContext *const context;
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
        ContributeCollectiveArgs(FutureImpl *i, DynamicCollective d, unsigned c)
          : LgTaskArgs<ContributeCollectiveArgs>(implicit_provenance),
            impl(i), dc(d), count(c) { }
      public:
        FutureImpl *const impl;
        const DynamicCollective dc;
        const unsigned count;
      };
    public:
      FutureImpl(Runtime *rt, bool register_future, DistributedID did, 
                 AddressSpaceID owner_space, ApEvent complete_event,
                 Operation *op = NULL);
      FutureImpl(const FutureImpl &rhs);
      virtual ~FutureImpl(void);
    public:
      FutureImpl& operator=(const FutureImpl &rhs);
    public:
      // Wait without subscribing to the payload
      void wait(bool silence_warnings, const char *warning_string);
      void* get_untyped_result(bool silence_warnings = true,
                               const char *warning_string = NULL,
                               bool internal = false,
                               bool check_size = false,
                               size_t future_size = 0);
      bool is_empty(bool block, bool silence_warnings = true,
                    const char *warning_string = NULL,
                    bool internal = false);
      size_t get_untyped_size(bool internal = false);
      ApEvent get_ready_event(void) const { return future_complete; }
    public:
      // This will simply save the value of the future
      void set_result(const void *args, size_t arglen, bool own);
      // This will save the value of the future locally
      void unpack_future(Deserializer &derez);
      // Reset the future in case we need to restart the
      // computation for resiliency reasons
      bool reset_future(void);
      // A special function for predicates to peek
      // at the boolean value of a future if it is set
      bool get_boolean_value(bool &valid);
      // Request that the value be made ready on this node
      ApEvent subscribe(void);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
    public:
      void register_dependence(Operation *consumer_op);
      void register_remote(AddressSpaceID sid, ReferenceMutator *mutator);
    protected:
      void mark_sampled(void);
      void broadcast_result(std::set<AddressSpaceID> &targets,
                            ApEvent complete, const bool need_lock);
      void record_subscription(AddressSpaceID subscriber, bool need_lock);
      void notify_remote_set(AddressSpaceID remote_space);
    public:
      void record_future_registered(ReferenceMutator *mutator);
      static void handle_future_result(Deserializer &derez, Runtime *rt);
      static void handle_future_subscription(Deserializer &derez, Runtime *rt,
                                             AddressSpaceID source);
      static void handle_future_notification(Deserializer &derez, Runtime *rt,
                                             AddressSpaceID source);
      static void handle_future_broadcast(Deserializer &derez, Runtime *rt);
    public:
      void contribute_to_collective(const DynamicCollective &dc,unsigned count);
      static void handle_contribute_to_collective(const void *args);
    public:
      // These three fields are only valid on the owner node
      Operation *const producer_op;
      const GenerationID op_gen;
      // The depth of the context in which this was made
      const int producer_depth;
#ifdef LEGION_SPY
      const UniqueID producer_uid;
#endif
    private:
      FRIEND_ALL_RUNTIME_CLASSES
      mutable LocalLock future_lock;
      ApEvent future_complete;
      ApUserEvent subscription_event;
      // On the owner node, keep track of the registered waiters
      std::set<AddressSpaceID> subscribers;
      void *result; 
      size_t result_size;
      AddressSpaceID result_set_space; // space on which the result was set
      volatile bool empty;
      volatile bool sampled;
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
      FutureMapImpl(TaskContext *ctx, Operation *op, RtEvent ready_event, 
                    Runtime *rt, DistributedID did, AddressSpaceID owner_space);
      FutureMapImpl(TaskContext *ctx, Runtime *rt, 
                    DistributedID did, AddressSpaceID owner_space,
                    RtEvent ready_event, bool register_now = true); // remote
      FutureMapImpl(const FutureMapImpl &rhs);
      virtual ~FutureMapImpl(void);
    public:
      FutureMapImpl& operator=(const FutureMapImpl &rhs);
    public:
      inline RtEvent get_ready_event(void) const { return ready_event; }
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
    public:
      Future get_future(const DomainPoint &point, RtEvent *wait_on = NULL);
      // Will return NULL if it does not exist
      FutureImpl* find_future(const DomainPoint &point);
      void set_all_futures(const std::map<DomainPoint,Future> &others);
      void set_future(const DomainPoint &point, FutureImpl *impl,
                      ReferenceMutator *mutator);
      void get_void_result(const DomainPoint &point, 
                            bool silence_warnings = true,
                            const char *warning_string = NULL);
      void wait_all_results(bool silence_warnings = true,
                            const char *warning_string = NULL);
      // This marks that all the futures are ready somewhere
      bool reset_all_futures(RtEvent new_ready_event);
    public:
      void get_all_futures(std::map<DomainPoint,Future> &futures) const;
#ifdef DEBUG_LEGION
    public:
      void add_valid_domain(const Domain &d);
      void add_valid_point(const DomainPoint &dp);
#endif
    public:
      void register_dependence(Operation *consumer_op);
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
      const int op_depth;
#ifdef LEGION_SPY
      const UniqueID op_uid;
#endif
    private:
      mutable LocalLock future_map_lock;
      RtEvent ready_event;
      std::map<DomainPoint,Future> futures;
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
      PhysicalRegionImpl(const RegionRequirement &req, ApEvent mapped_event,
                         bool mapped, TaskContext *ctx, MapperID mid,
                         MappingTagID tag, bool leaf, bool virt, Runtime *rt);
      PhysicalRegionImpl(const PhysicalRegionImpl &rhs);
      ~PhysicalRegionImpl(void);
    public:
      PhysicalRegionImpl& operator=(const PhysicalRegionImpl &rhs);
    public:
      inline bool created_accessor(void) const { return made_accessor; }
    public:
      void wait_until_valid(bool silence_warnings, const char *warning_string, 
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
      void remap_region(ApEvent new_mapped_event);
      const RegionRequirement& get_requirement(void) const;
      void set_reference(const InstanceRef &references);
      void reset_references(const InstanceSet &instances,ApUserEvent term_event,
                            ApEvent wait_for = ApEvent::NO_AP_EVENT);
      ApEvent get_mapped_event(void) const;
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
                                         const char *warning_string,
                                         bool silence_warnings, 
                                         bool generic_accessor,
                                         bool check_field_size,
                                         ReductionOpID redop);
      void fail_bounds_check(DomainPoint p, FieldID fid, PrivilegeMode mode);
      void fail_bounds_check(Domain d, FieldID fid, PrivilegeMode mode);
      void report_incompatible_accessor(const char *accessor_kind,
                             PhysicalInstance instance, FieldID fid);
    public:
      Runtime *const runtime;
      TaskContext *const context;
      const MapperID map_id;
      const MappingTagID tag;
      const bool leaf_region;
      const bool virtual_mapped;
      const bool replaying;
    private:
      // Event for when the instance ref is ready
      ApEvent mapped_event;
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

    class LegionHandshakeImpl : public Collectable,
                       public LegionHeapify<LegionHandshakeImpl> {
    public:
      static const AllocationType alloc_type = MPI_HANDSHAKE_ALLOC;
    public:
      LegionHandshakeImpl(bool init_in_ext, int ext_participants, 
                          int legion_participants);
      LegionHandshakeImpl(const LegionHandshakeImpl &rhs);
      ~LegionHandshakeImpl(void);
    public:
      LegionHandshakeImpl& operator=(const LegionHandshakeImpl &rhs);
    public:
      void initialize(void);
    public:
      void ext_handoff_to_legion(void);
      void ext_wait_on_legion(void);
    public:
      void legion_handoff_to_ext(void);
      void legion_wait_on_ext(void);
    public:
      PhaseBarrier get_legion_wait_phase_barrier(void);
      PhaseBarrier get_legion_arrive_phase_barrier(void);
      void advance_legion_handshake(void);
    private:
      const bool init_in_ext;
      const int ext_participants;
      const int legion_participants;
    private:
      PhaseBarrier ext_wait_barrier;
      PhaseBarrier ext_arrive_barrier;
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
      bool initiate_exchange(void);
      void send_remainder_stage(void);
      bool send_ready_stages(const int start_stage=1);
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
      bool done_triggered;
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
      struct SchedulerArgs : public LgTaskArgs<SchedulerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_SCHEDULER_ID;
      public:
        SchedulerArgs(Processor p)
          : LgTaskArgs<SchedulerArgs>(0), proc(p) { }
      public:
        const Processor proc;
      }; 
      struct DeferMapperSchedulerArgs : 
        public LgTaskArgs<DeferMapperSchedulerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_MAPPER_SCHEDULER_TASK_ID;
      public:
        DeferMapperSchedulerArgs(ProcessorManager *proxy,
                                 MapperID mid, RtEvent defer)
          : LgTaskArgs<DeferMapperSchedulerArgs>(implicit_provenance),
            proxy_this(proxy), map_id(mid), deferral_event(defer) { }
      public:
        ProcessorManager *const proxy_this;
        const MapperID map_id;
        const RtEvent deferral_event;
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
      bool outstanding_task_scheduler;
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
        MapperState(void)
          : queue_guard(false) { }
      public:
        std::list<TaskOp*> ready_queue;
        RtEvent deferral_event;
        RtUserEvent queue_waiter;
        bool queue_guard;
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
        PENDING_COLLECTED_STATE = 2, // sticky
        VALID_STATE = 3,
        PENDING_ACQUIRE_STATE = 4,
      };
    public:
      struct InstanceInfo {
      public:
        InstanceInfo(void)
          : current_state(COLLECTABLE_STATE), 
            deferred_collect(RtUserEvent::NO_RT_USER_EVENT),
            instance_size(0), pending_acquires(0), min_priority(0) { }
      public:
        InstanceState current_state;
        RtUserEvent deferred_collect;
        size_t instance_size;
        unsigned pending_acquires;
        GCPriority min_priority;
        std::map<std::pair<MapperID,Processor>,GCPriority> mapper_priorities;
        // For tracking external instances and whether they can be used
      };
    public:
      MemoryManager(Memory mem, Runtime *rt);
      MemoryManager(const MemoryManager &rhs);
      ~MemoryManager(void);
    public:
      MemoryManager& operator=(const MemoryManager &rhs);
    public:
      void find_shutdown_preconditions(std::set<ApEvent> &preconditions);
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
      bool attempt_acquire(PhysicalManager *manager);
      void complete_acquire(PhysicalManager *manager);
    public:
      bool create_physical_instance(const LayoutConstraintSet &contraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, MapperID mapper_id,
                                    Processor processor, bool acquire, 
                                    GCPriority priority, bool tight_bounds,
                                    size_t *footprint, UniqueID creator_id,
                                    bool remote = false);
      bool create_physical_instance(LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, MapperID mapper_id,
                                    Processor processor, bool acquire, 
                                    GCPriority priority, bool tight_bounds,
                                    size_t *footprint, UniqueID creator_id,
                                    bool remote = false);
      bool find_or_create_physical_instance(
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    MapperID mapper_id, Processor processor,
                                    bool acquire, GCPriority priority, 
                                    bool tight_region_bounds, size_t *footprint,
                                    UniqueID creator_id, bool remote = false);
      bool find_or_create_physical_instance(
                                    LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    MapperID mapper_id, Processor processor,
                                    bool acquire, GCPriority priority, 
                                    bool tight_region_bounds, size_t *footprint,
                                    UniqueID creator_id, bool remote = false);
      bool find_physical_instance(  const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_bounds, bool remote = false);
      bool find_physical_instance(  LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_bounds, bool remote = false);
      void release_tree_instances(RegionTreeID tid);
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
      void process_acquire_response(Deserializer &derez, AddressSpaceID src);
    protected:
      bool find_satisfying_instance(const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire, 
                                    bool tight_region_bounds, bool remote);
      bool find_satisfying_instance(LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire, 
                                    bool tight_region_bounds, bool remote);
      bool find_valid_instance(     const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire, 
                                    bool tight_region_bounds, bool remote);
      bool find_valid_instance(     LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire, 
                                    bool tight_region_bounds, bool remote);
      void release_candidate_references(const std::deque<PhysicalManager*>
                                                        &candidates) const;
    protected:
      // We serialize all allocation attempts in a memory in order to 
      // ensure find_and_create calls will remain atomic
      RtEvent acquire_allocation_privilege(void);
      void release_allocation_privilege(void);
      PhysicalManager* allocate_physical_instance(InstanceBuilder &builder,
                                                  size_t *footprint);
    public:
      bool delete_by_size_and_state(const size_t needed_size, 
                                    InstanceState state, bool larger_only); 
      RtEvent attach_external_instance(PhysicalManager *manager);
      RtEvent detach_external_instance(PhysicalManager *manager);
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
      typedef LegionMap<PhysicalManager*,InstanceInfo,
                        MEMORY_INSTANCES_ALLOC>::tracked TreeInstances;
      std::map<RegionTreeID,TreeInstances> current_instances;
      // Keep track of outstanding requuests for allocations which 
      // will be tried in the order that they arrive
      std::deque<RtUserEvent> pending_allocation_attempts;
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
        FULL_MESSAGE = 0x1,
        PARTIAL_MESSAGE = 0x2,
        FINAL_MESSAGE = 0x3,
      };
      struct PartialMessage {
      public:
        PartialMessage(void)
          : buffer(NULL), size(0), index(0), messages(0), total(0) { }
      public:
        char *buffer;
        size_t size;
        unsigned index;
        unsigned messages;
        unsigned total;
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
      bool handle_messages(unsigned num_messages, Runtime *runtime, 
                           AddressSpaceID remote_address_space,
                           const char *args, size_t arglen) const;
      static void buffer_messages(unsigned num_messages,
                                  const void *args, size_t arglen,
                                  char *&receiving_buffer,
                                  size_t &receiving_buffer_size,
                                  unsigned &receiving_index,
                                  unsigned &received_messages,
                                  unsigned &partial_messages);
      void filter_unordered_events(void);
    private:
      mutable LocalLock channel_lock;
      char *const sending_buffer;
      unsigned sending_index;
      const size_t sending_buffer_size;
      RtEvent last_message_event;
      MessageHeader header;
      unsigned packaged_messages;
      // For unordered channels so we can group partial
      // messages from remote nodes
      unsigned partial_message_id;
      bool partial;
    private:
      const bool ordered_channel;
      const LgPriority request_priority;
      const LgPriority response_priority;
      static const unsigned MAX_UNORDERED_EVENTS = 32;
      std::set<RtEvent> unordered_events;
    private:
      // State for receiving messages
      // No lock for receiving messages since we know
      // that they are ordered for ordered virtual
      // channels, for un-ordered virtual channels then
      // we know that we do need the lock
      char *receiving_buffer;
      size_t receiving_buffer_size;
      unsigned receiving_index;
      unsigned received_messages;
      unsigned partial_messages;
      std::map<unsigned/*message id*/,PartialMessage> *partial_assembly;
      mutable bool observed_recent;
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
                     const Processor remote_util_group);
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
    public:
      Runtime *const runtime;
      // State for sending messages
      const Processor target;
    private:
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
        RetryShutdownArgs(ShutdownPhase p)
          : LgTaskArgs<RetryShutdownArgs>(0), phase(p) { }
      public:
        const ShutdownPhase phase;
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
        SemanticRequestArgs(TaskImpl *proxy, SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticRequestArgs>(implicit_provenance),
            proxy_this(proxy), tag(t), source(src) { }
      public:
        TaskImpl *const proxy_this;
        const SemanticTag tag;
        const AddressSpaceID source;
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
      bool can_use(Processor::Kind kind, bool warn) const;
    public:
      void broadcast_variant(RtUserEvent done, AddressSpaceID origin,
                             AddressSpaceID local);
    public:
      static void handle_variant_broadcast(Runtime *runtime, 
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
      public LayoutConstraintSet, public DistributedCollectable,
      public LegionHeapify<LayoutConstraints> {
    public:
      static const AllocationType alloc_type = LAYOUT_CONSTRAINTS_ALLOC; 
    public:
      LayoutConstraints(LayoutConstraintID layout_id, FieldSpace handle, 
                        Runtime *runtime, bool inter, DistributedID did = 0);
      LayoutConstraints(LayoutConstraintID layout_id, Runtime *runtime, 
                        const LayoutConstraintRegistrar &registrar, 
                        bool inter, DistributedID did = 0);
      LayoutConstraints(LayoutConstraintID layout_id,
                        Runtime *runtime, const LayoutConstraintSet &cons,
                        FieldSpace handle, bool inter);
      LayoutConstraints(const LayoutConstraints &rhs);
      virtual ~LayoutConstraints(void);
    public:
      LayoutConstraints& operator=(const LayoutConstraints &rhs);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      inline FieldSpace get_field_space(void) const { return handle; }
      inline const char* get_name(void) const { return constraints_name; }
    public:
      void send_constraint_response(AddressSpaceID source,
                                    RtUserEvent done_event);
      void update_constraints(Deserializer &derez);
    public:
      bool entails(LayoutConstraints *other_constraints, unsigned total_dims,
                   const LayoutConstraint **failed_constraint);
      bool entails(const LayoutConstraintSet &other, unsigned total_dims,
                   const LayoutConstraint **failed_constraint) const;
      bool conflicts(LayoutConstraints *other_constraints, unsigned total_dims,
                     const LayoutConstraint **conflict_constraint);
      bool conflicts(const LayoutConstraintSet &other, unsigned total_dims,
                     const LayoutConstraint **conflict_constraint) const;
      bool entails_without_pointer(LayoutConstraints *other,unsigned total_dims,
                                   const LayoutConstraint **failed_constraint);
      bool entails_without_pointer(const LayoutConstraintSet &other,
         unsigned total_dims, const LayoutConstraint **failed_constraint) const;
    public:
      static AddressSpaceID get_owner_space(LayoutConstraintID layout_id,
                                            Runtime *runtime);
    public:
      static void process_request(Runtime *runtime, Deserializer &derez,
                                  AddressSpaceID source);
      static void process_response(Runtime *runtime, Deserializer &derez, 
                                   AddressSpaceID source);
    public:
      const LayoutConstraintID layout_id;
      const FieldSpace handle;
      // True if this layout constraint object was made by the runtime
      // False if it was made by the application or the mapper
      const bool internal;
    protected:
      char *constraints_name;
      mutable LocalLock layout_lock;
    protected:
      std::map<std::pair<LayoutConstraintID,unsigned/*total dims*/>,
                const LayoutConstraint*> conflict_cache;
      std::map<std::pair<LayoutConstraintID,unsigned/*total dims*/>,
                const LayoutConstraint*> entailment_cache;
      std::map<std::pair<LayoutConstraintID,unsigned/*total dims*/>,
                const LayoutConstraint*> no_pointer_entailment_cache;
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
          Runtime *runtime, const std::vector<PointTask*> &point_tasks,
          IndexSpaceNode *launch_space_node);
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
      // Checking for inversion
      void check_inversion(const Task *task, unsigned idx,
                           const std::vector<DomainPoint> &ordered_points);
      void check_containment(const Task *task, unsigned idx,
                             const std::vector<DomainPoint> &ordered_points);
    public:
      const int depth; 
      const bool is_exclusive;
      const bool is_invertible;
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
      struct LegionConfiguration {
      public:
        LegionConfiguration(void)
          : delay_start(0),
            legion_collective_radix(LEGION_COLLECTIVE_RADIX),
            initial_task_window_size(LEGION_DEFAULT_MAX_TASK_WINDOW),
            initial_task_window_hysteresis(
                LEGION_DEFAULT_TASK_WINDOW_HYSTERESIS),
            initial_tasks_to_schedule(LEGION_DEFAULT_MIN_TASKS_TO_SCHEDULE),
            initial_meta_task_vector_width(
                LEGION_DEFAULT_META_TASK_VECTOR_WIDTH),
            max_message_size(LEGION_DEFAULT_MAX_MESSAGE_SIZE),
            gc_epoch_size(LEGION_DEFAULT_GC_EPOCH_SIZE),
            max_local_fields(LEGION_DEFAULT_LOCAL_FIELDS),
            max_replay_parallelism(LEGION_DEFAULT_MAX_REPLAY_PARALLELISM),
            program_order_execution(false),
            dump_physical_traces(false),
            no_tracing(false),
            no_physical_tracing(false),
            no_trace_optimization(false),
            no_fence_elision(false),
            replay_on_cpus(false),
            verify_partitions(false),
            runtime_warnings(false),
            warnings_backtrace(false),
            report_leaks(false),
            separate_runtime_instances(false),
            record_registration(false),
            stealing_disabled(false),
            resilient_mode(false),
            unsafe_launch(false),
            unsafe_mapper(false),
            safe_mapper(false),
            disable_independence_tests(false),
            legion_spy_enabled(false),
            enable_test_mapper(false),
            slow_config_ok(false),
#ifdef DEBUG_LEGION
            logging_region_tree_state(false),
            verbose_logging(false),
            logical_logging_only(false),
            physical_logging_only(false),
            check_privileges(true),
#else
            check_privileges(false),
#endif
            num_profiling_nodes(0),
            serializer_type("binary"),
            prof_footprint_threshold(128 << 20),
            prof_target_latency(100) { }
      public:
        int delay_start;
        mutable int legion_collective_radix;
        mutable int legion_collective_log_radix;
        mutable int legion_collective_stages;
        mutable int legion_collective_last_radix;
        mutable int legion_collective_participating_spaces;
        int initial_task_window_size;
        unsigned initial_task_window_hysteresis;
        unsigned initial_tasks_to_schedule;
        unsigned initial_meta_task_vector_width;
        unsigned max_message_size;
        unsigned gc_epoch_size;
        unsigned max_local_fields;
        unsigned max_replay_parallelism;
      public:
        bool program_order_execution;
        bool dump_physical_traces;
        bool no_tracing;
        bool no_physical_tracing;
        bool no_trace_optimization;
        bool no_fence_elision;
        bool replay_on_cpus;
        bool verify_partitions;
        bool runtime_warnings;
        bool warnings_backtrace;
        bool report_leaks;
        bool separate_runtime_instances;
        bool record_registration;
        bool stealing_disabled;
        bool resilient_mode;
        bool unsafe_launch;
        bool unsafe_mapper;
        bool safe_mapper;
        bool disable_independence_tests;
        bool legion_spy_enabled;
        bool enable_test_mapper;
        std::string replay_file;
        std::string ldb_file;
        bool slow_config_ok;
#ifdef DEBUG_LEGION
        bool logging_region_tree_state;
        bool verbose_logging;
        bool logical_logging_only;
        bool physical_logging_only;
#endif
        bool check_privileges;
      public:
        unsigned num_profiling_nodes;
        std::string serializer_type;
        std::string prof_logfile;
        size_t prof_footprint_threshold;
        size_t prof_target_latency;
      public:
        void configure_collective_settings(int total_spaces) const;
      };
    public:
      struct DeferredRecycleArgs : public LgTaskArgs<DeferredRecycleArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_RECYCLE_ID;
      public:
        DeferredRecycleArgs(DistributedID id)
          : LgTaskArgs<DeferredRecycleArgs>(implicit_provenance), did(id) { }
      public:
        const DistributedID did;
      }; 
      struct TopFinishArgs : public LgTaskArgs<TopFinishArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TOP_FINISH_TASK_ID;
      public:
        TopFinishArgs(TopLevelContext *c)
          : LgTaskArgs<TopFinishArgs>(0), ctx(c) { }
      public:
        TopLevelContext *const ctx;
      };
      struct MapperTaskArgs : public LgTaskArgs<MapperTaskArgs> {
      public:
        static const LgTaskID TASK_ID = LG_MAPPER_TASK_ID;
      public:
        MapperTaskArgs(FutureImpl *f, MapperID mid, Processor p,
                       ApEvent ae, TopLevelContext *c)
          : LgTaskArgs<MapperTaskArgs>(implicit_provenance),
            future(f), map_id(mid), proc(p), event(ae), ctx(c) { }
      public:
        FutureImpl *const future;
        const MapperID map_id;
        const Processor proc;
        const ApEvent event;
        TopLevelContext *const ctx;
      }; 
      struct SelectTunableArgs : public LgTaskArgs<SelectTunableArgs> {
      public:
        static const LgTaskID TASK_ID = LG_SELECT_TUNABLE_TASK_ID;
      public:
        SelectTunableArgs(UniqueID uid, MapperID mid, MappingTagID t,
                          TunableID tune, const void *arg, size_t size,
                          TaskContext *c, FutureImpl *f, ApUserEvent trig)
          : LgTaskArgs<SelectTunableArgs>(uid), mapper_id(mid), tag(t),
            tunable_id(tune), args((size > 0) ? malloc(size) : NULL),
            argsize(size), ctx(c), result(f), to_trigger(trig)
            { if (argsize > 0) memcpy(args, arg, argsize); }
      public:
        const MapperID mapper_id;
        const MappingTagID tag;
        const TunableID tunable_id;
        void *const args;
        const size_t argsize;
        unsigned tunable_index; // only valid for LegionSpy
        TaskContext *const ctx;
        FutureImpl *const result;
        const ApUserEvent to_trigger;
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
      Runtime(Machine m, const LegionConfiguration &config,
              InputArgs input_args, AddressSpaceID space_id,
              const std::set<Processor> &local_procs,
              const std::set<Processor> &local_util_procs,
              const std::set<AddressSpaceID> &address_spaces,
              const std::map<Processor,AddressSpaceID> &proc_spaces);
      Runtime(const Runtime &rhs);
      ~Runtime(void);
    public:
      Runtime& operator=(const Runtime &rhs);
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
      VirtualManager *virtual_manager;
      Processor utility_group;
      const size_t num_utility_procs;
    public:
      const InputArgs input_args;
      const int initial_task_window_size;
      const unsigned initial_task_window_hysteresis;
      const unsigned initial_tasks_to_schedule;
      const unsigned initial_meta_task_vector_width;
      const unsigned max_message_size;
      const unsigned gc_epoch_size;
      const unsigned max_local_fields;
      const unsigned max_replay_parallelism;
    public:
      const bool program_order_execution;
      const bool dump_physical_traces;
      const bool no_tracing;
      const bool no_physical_tracing;
      const bool no_trace_optimization;
      const bool no_fence_elision;
      const bool replay_on_cpus;
      const bool verify_partitions;
      const bool runtime_warnings;
      const bool warnings_backtrace;
      const bool report_leaks;
      const bool separate_runtime_instances;
      const bool record_registration;
      const bool stealing_disabled;
      const bool resilient_mode;
      const bool unsafe_launch;
      const bool unsafe_mapper;
      const bool disable_independence_tests;
      const bool legion_spy_enabled;
      const bool enable_test_mapper;
      const bool legion_ldb_enabled;
      const std::string replay_file;
#ifdef DEBUG_LEGION
      const bool logging_region_tree_state;
      const bool verbose_logging;
      const bool logical_logging_only;
      const bool physical_logging_only;
#endif
      const bool check_privileges;
    public:
      const unsigned num_profiling_nodes;
    public:
      const int legion_collective_radix;
      const int legion_collective_log_radix;
      const int legion_collective_stages;
      const int legion_collective_last_radix;
      const int legion_collective_participating_spaces;
      MPIRankTable *const mpi_rank_table;
    public:
      void register_static_variants(void);
      void register_static_constraints(void);
      void register_static_projections(void);
      void initialize_legion_prof(const LegionConfiguration &config);
      void log_machine(Machine machine) const;
      void initialize_mappers(void);
      void initialize_virtual_manager(void);
      void initialize_runtime(void);
      void perform_registration_callback(RegistrationCallbackFnptr callback);
      void startup_runtime(void);
      void finalize_runtime(void);
      ApEvent launch_mapper_task(Mapper *mapper, Processor proc, 
                                 TaskID tid,
                                 const TaskArgument &arg, MapperID map_id);
      void process_mapper_task_result(const MapperTaskArgs *args); 
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
      void destroy_field_space(Context ctx, FieldSpace handle,
                               const bool unordered);
      size_t get_field_size(Context ctx, FieldSpace handle, FieldID fid);
      size_t get_field_size(FieldSpace handle, FieldID fid);
      void get_field_space_fields(Context ctx, FieldSpace handle,
                                  std::vector<FieldID> &fields);
      void get_field_space_fields(FieldSpace handle, 
                                  std::vector<FieldID> &fields);
    public:
      LogicalRegion create_logical_region(Context ctx, IndexSpace index,
                                          FieldSpace fields, bool task_local);
      void destroy_logical_region(Context ctx, LogicalRegion handle,
                                  const bool unordered);
      void destroy_logical_partition(Context ctx, LogicalPartition handle,
                                     const bool unordered);
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
      Future execute_index_space(Context ctx, const IndexTaskLauncher &launcher,
                                 ReductionOpID redop, bool deterministic);
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
      Future detach_external_resource(Context ctx, PhysicalRegion region, 
                                      const bool flush, const bool unordered);
      void progress_unordered_operations(Context ctx);
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
      Future issue_mapping_fence(Context ctx);
      Future issue_execution_fence(Context ctx);
      void begin_trace(Context ctx, TraceID tid, bool logical_only);
      void end_trace(Context ctx, TraceID tid);
      void begin_static_trace(Context ctx, 
                              const std::set<RegionTreeID> *managed);
      void end_static_trace(Context ctx);
      TraceID generate_dynamic_trace_id(void);
      TraceID generate_library_trace_ids(const char *name, size_t count);
      static TraceID& get_current_static_trace_id(void);
      static TraceID generate_static_trace_id(void);
      void complete_frame(Context ctx);
      FutureMap execute_must_epoch(Context ctx, 
                                   const MustEpochLauncher &launcher);
      Future issue_timing_measurement(Context ctx,
                                      const TimingLauncher &launcher);
    public:
      Future select_tunable_value(Context ctx, TunableID tid,
                                  MapperID mid, MappingTagID tag,
                                  const void *args, size_t argsize);
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
      void yield(Context ctx);
    public:
      const std::map<int,AddressSpace>& find_forward_MPI_mapping(void);
      const std::map<AddressSpace,int>& find_reverse_MPI_mapping(void);
      int find_local_MPI_rank(void);
      bool is_MPI_interop_configured(void);
    public:
      Mapping::MapperRuntime* get_mapper_runtime(void);
      MapperID generate_dynamic_mapper_id(void);
      MapperID generate_library_mapper_ids(const char *name, size_t count);
      static MapperID& get_current_static_mapper_id(void);
      static MapperID generate_static_mapper_id(void);
      void add_mapper(MapperID map_id, Mapper *mapper, Processor proc);
      void replace_default_mapper(Mapper *mapper, Processor proc);
      MapperManager* find_mapper(MapperID map_id);
      MapperManager* find_mapper(Processor target, MapperID map_id);
      static MapperManager* wrap_mapper(Runtime *runtime, Mapper *mapper,
                                        MapperID map_id, Processor proc);
    public:
      ProjectionID generate_dynamic_projection_id(void);
      ProjectionID generate_library_projection_ids(const char *name,size_t cnt);
      static ProjectionID& get_current_static_projection_id(void);
      static ProjectionID generate_static_projection_id(void);
      void register_projection_functor(ProjectionID pid, 
                                       ProjectionFunctor *func,
                                       bool need_zero_check = true,
                                       bool silence_warnings = false,
                                       const char *warning_string = NULL);
      static void preregister_projection_functor(ProjectionID pid,
                                       ProjectionFunctor *func);
      ProjectionFunction* find_projection_function(ProjectionID pid);
    public:
      void register_reduction(ReductionOpID redop_id,
                              ReductionOp *redop,
                              SerdezInitFnptr init_fnptr,
                              SerdezFoldFnptr fold_fnptr,
                              bool permit_duplicates);
      void register_serdez(CustomSerdezID serdez_id,
                           SerdezOp *serdez_op,
                           bool permit_duplicates);
      const ReductionOp* get_reduction(ReductionOpID redop_id);
      const SerdezOp* get_serdez(CustomSerdezID serdez_id);
      const SerdezRedopFns* get_serdez_redop(ReductionOpID redop_id);
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
      TaskID generate_dynamic_task_id(void);
      TaskID generate_library_task_ids(const char *name, size_t count);
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
      ReductionOpID generate_dynamic_reduction_id(void);
      ReductionOpID generate_library_reduction_ids(const char *name, 
                                                   size_t count);
    public:
      CustomSerdezID generate_dynamic_serdez_id(void);
      CustomSerdezID generate_library_serdez_ids(const char *name,size_t count);
    public:
      // Memory manager functions
      MemoryManager* find_memory_manager(Memory mem);
      AddressSpaceID find_address_space(Memory handle) const;
    public:
      // Messaging functions
      MessageManager* find_messenger(AddressSpaceID sid);
      MessageManager* find_messenger(Processor target);
      AddressSpaceID find_address_space(Processor target) const;
      void handle_endpoint_creation(Deserializer &derez);
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
      void send_remote_task_replay(AddressSpaceID target, Serializer &rez);
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
      void send_index_space_remote_expression_request(AddressSpaceID target,
                                                      Serializer &rez);
      void send_index_space_remote_expression_response(AddressSpaceID target,
                                                       Serializer &rez);
      void send_index_space_remote_expression_invalidation(
                                    AddressSpaceID target, Serializer &rez);
      void send_index_partition_notification(AddressSpaceID target, 
                                             Serializer &rez);
      void send_index_partition_node(AddressSpaceID target, Serializer &rez);
      void send_index_partition_request(AddressSpaceID target, Serializer &rez);
      void send_index_partition_return(AddressSpaceID target, Serializer &rez);
      void send_index_partition_child_request(AddressSpaceID target,
                                              Serializer &rez);
      void send_index_partition_child_response(AddressSpaceID target,
                                               Serializer &rez);
      void send_index_partition_disjoint_update(AddressSpaceID target,
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
                                        AddressSpaceID target,
                                        std::set<RtEvent> &applied);
      void send_index_partition_destruction(IndexPartition handle, 
                                            AddressSpaceID target,
                                            std::set<RtEvent> &applied);
      void send_field_space_destruction(FieldSpace handle, 
                                        AddressSpaceID target,
                                        std::set<RtEvent> &applied);
      void send_logical_region_destruction(LogicalRegion handle, 
                                           AddressSpaceID target,
                                           std::set<RtEvent> *applied);
      void send_logical_partition_destruction(LogicalPartition handle,
                                              AddressSpaceID target,
                                              std::set<RtEvent> *applied);
      void send_individual_remote_complete(Processor target, Serializer &rez);
      void send_individual_remote_commit(Processor target, Serializer &rez);
      void send_slice_remote_mapped(Processor target, Serializer &rez);
      void send_slice_remote_complete(Processor target, Serializer &rez);
      void send_slice_remote_commit(Processor target, Serializer &rez);
      void send_slice_find_intra_space_dependence(Processor target, 
                                                  Serializer &rez);
      void send_slice_record_intra_space_dependence(Processor target,
                                                    Serializer &rez);
      void send_did_remote_registration(AddressSpaceID target, Serializer &rez);
      void send_did_remote_valid_update(AddressSpaceID target, Serializer &rez);
      void send_did_remote_gc_update(AddressSpaceID target, Serializer &rez);
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
      void send_fill_view(AddressSpaceID target, Serializer &rez);
      void send_phi_view(AddressSpaceID target, Serializer &rez);
      void send_reduction_view(AddressSpaceID target, Serializer &rez);
      void send_instance_manager(AddressSpaceID target, Serializer &rez);
      void send_reduction_manager(AddressSpaceID target, Serializer &rez);
      void send_create_top_view_request(AddressSpaceID target, Serializer &rez);
      void send_create_top_view_response(AddressSpaceID target,Serializer &rez);
      void send_view_register_user(AddressSpaceID target, Serializer &rez);
      void send_view_find_copy_preconditions_request(AddressSpaceID target,
                                                     Serializer &rez);
      void send_view_find_copy_preconditions_response(AddressSpaceID target,
                                                      Serializer &rez);
      void send_view_add_copy_user(AddressSpaceID target, Serializer &rez);
#ifdef ENABLE_VIEW_REPLICATION
      void send_view_replication_request(AddressSpaceID target,Serializer &rez);
      void send_view_replication_response(AddressSpaceID target,
                                          Serializer &rez);
      void send_view_replication_removal(AddressSpaceID target,Serializer &rez);
#endif
      void send_future_result(AddressSpaceID target, Serializer &rez);
      void send_future_subscription(AddressSpaceID target, Serializer &rez);
      void send_future_notification(AddressSpaceID target, Serializer &rez);
      void send_future_broadcast(AddressSpaceID target, Serializer &rez);
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
      void send_compute_equivalence_sets_request(AddressSpaceID target, 
                                                 Serializer &rez);
      void send_equivalence_set_response(AddressSpaceID target,Serializer &rez);
      void send_equivalence_set_subset_request(AddressSpaceID target, 
                                               Serializer &rez);
      void send_equivalence_set_subset_response(AddressSpaceID target, 
                                                Serializer &rez);
      void send_equivalence_set_subset_update(AddressSpaceID target,
                                              Serializer &rez);
      void send_equivalence_set_ray_trace_request(AddressSpaceID target,
                                                  Serializer &rez);
      void send_equivalence_set_ray_trace_response(AddressSpaceID target,
                                                   Serializer &rez);
      void send_equivalence_set_migration(AddressSpaceID target, 
                                          Serializer &rez);
      void send_equivalence_set_owner_update(AddressSpaceID target,
                                             Serializer &rez);
      void send_equivalence_set_remote_refinement(AddressSpaceID target,
                                                  Serializer &rez);
      void send_equivalence_set_remote_request_instances(AddressSpaceID target,
                                                         Serializer &rez);
      void send_equivalence_set_remote_request_invalid(AddressSpaceID target,
                                                       Serializer &rez);
      void send_equivalence_set_remote_updates(AddressSpaceID target,
                                               Serializer &rez);
      void send_equivalence_set_remote_acquires(AddressSpaceID target,
                                                Serializer &rez);
      void send_equivalence_set_remote_releases(AddressSpaceID target,
                                                Serializer &rez);
      void send_equivalence_set_remote_copies_across(AddressSpaceID target,
                                                     Serializer &rez);
      void send_equivalence_set_remote_overwrites(AddressSpaceID target,
                                                  Serializer &rez);
      void send_equivalence_set_remote_filters(AddressSpaceID target,
                                               Serializer &rez);
      void send_equivalence_set_remote_instances(AddressSpaceID target,
                                                 Serializer &rez);
      void send_equivalence_set_stale_update(AddressSpaceID target, 
                                             Serializer &rez);
      void send_instance_request(AddressSpaceID target, Serializer &rez);
      void send_instance_response(AddressSpaceID target, Serializer &rez);
      void send_external_create_request(AddressSpaceID target, Serializer &rez);
      void send_external_create_response(AddressSpaceID target,Serializer &rez);
      void send_external_attach(AddressSpaceID target, Serializer &rez);
      void send_external_detach(AddressSpaceID target, Serializer &rez);
      void send_gc_priority_update(AddressSpaceID target, Serializer &rez);
      void send_never_gc_response(AddressSpaceID target, Serializer &rez);
      void send_acquire_request(AddressSpaceID target, Serializer &rez);
      void send_acquire_response(AddressSpaceID target, Serializer &rez);
      void send_variant_broadcast(AddressSpaceID target, Serializer &rez);
      void send_constraint_request(AddressSpaceID target, Serializer &rez);
      void send_constraint_response(AddressSpaceID target, Serializer &rez);
      void send_constraint_release(AddressSpaceID target, Serializer &rez);
      void send_mpi_rank_exchange(AddressSpaceID target, Serializer &rez);
      void send_library_mapper_request(AddressSpaceID target, Serializer &rez);
      void send_library_mapper_response(AddressSpaceID target, Serializer &rez);
      void send_library_trace_request(AddressSpaceID target, Serializer &rez);
      void send_library_trace_response(AddressSpaceID target, Serializer &rez);
      void send_library_projection_request(AddressSpaceID target, 
                                           Serializer &rez);
      void send_library_projection_response(AddressSpaceID target,
                                            Serializer &rez);
      void send_library_task_request(AddressSpaceID target, Serializer &rez);
      void send_library_task_response(AddressSpaceID target, Serializer &rez);
      void send_library_redop_request(AddressSpaceID target, Serializer &rez);
      void send_library_redop_response(AddressSpaceID target, Serializer &rez);
      void send_library_serdez_request(AddressSpaceID target, Serializer &rez);
      void send_library_serdez_response(AddressSpaceID target, Serializer &rez);
      void send_remote_op_report_uninitialized(AddressSpaceID target,
                                               Serializer &rez);
      void send_remote_op_profiling_count_update(AddressSpaceID target,
                                                 Serializer &rez);
      void send_remote_trace_update(AddressSpaceID target, Serializer &rez);
      void send_remote_trace_response(AddressSpaceID target, Serializer &rez);
      void send_remote_trace_equivalence_sets_request(AddressSpaceID target,
                                                      Serializer &rez);
      void send_remote_trace_equivalence_sets_response(AddressSpaceID target,
                                                       Serializer &rez);
      void send_shutdown_notification(AddressSpaceID target, Serializer &rez);
      void send_shutdown_response(AddressSpaceID target, Serializer &rez);
    public:
      // Complementary tasks for handling messages
      void handle_task(Deserializer &derez);
      void handle_steal(Deserializer &derez);
      void handle_advertisement(Deserializer &derez);
      void handle_remote_task_replay(Deserializer &derez);
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
      void handle_index_space_remote_expression_request(Deserializer &derez,
                                                        AddressSpaceID source);
      void handle_index_space_remote_expression_response(Deserializer &derez,
                                                         AddressSpaceID source);
      void handle_index_space_remote_expression_invalidation(
                                                         Deserializer &derez);
      void handle_index_partition_notification(Deserializer &derez);
      void handle_index_partition_node(Deserializer &derez,
                                       AddressSpaceID source);
      void handle_index_partition_request(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_index_partition_return(Deserializer &derez);
      void handle_index_partition_child_request(Deserializer &derez,
                                                AddressSpaceID source);
      void handle_index_partition_child_response(Deserializer &derez);
      void handle_index_partition_disjoint_update(Deserializer &derez);
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
      void handle_individual_remote_complete(Deserializer &derez);
      void handle_individual_remote_commit(Deserializer &derez);
      void handle_slice_remote_mapped(Deserializer &derez, 
                                      AddressSpaceID source);
      void handle_slice_remote_complete(Deserializer &derez);
      void handle_slice_remote_commit(Deserializer &derez);
      void handle_slice_find_intra_dependence(Deserializer &derez);
      void handle_slice_record_intra_dependence(Deserializer &derez);
      void handle_did_remote_registration(Deserializer &derez, 
                                          AddressSpaceID source);
      void handle_did_remote_valid_update(Deserializer &derez);
      void handle_did_remote_gc_update(Deserializer &derez);
      void handle_did_create_add(Deserializer &derez);
      void handle_did_create_remove(Deserializer &derez);
      void handle_did_remote_unregister(Deserializer &derez);
      void handle_send_back_logical_state(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_send_atomic_reservation_request(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_send_atomic_reservation_response(Deserializer &derez);
      void handle_send_materialized_view(Deserializer &derez, 
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
      void handle_view_request(Deserializer &derez, AddressSpaceID source);
      void handle_view_register_user(Deserializer &derez,AddressSpaceID source);
      void handle_view_copy_pre_request(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_view_copy_pre_response(Deserializer &derez,
                                         AddressSpaceID source);
      void handle_view_add_copy_user(Deserializer &derez,AddressSpaceID source);
#ifdef ENABLE_VIEW_REPLICATION
      void handle_view_replication_request(Deserializer &derez,
                                           AddressSpaceID source);
      void handle_view_replication_response(Deserializer &derez);
      void handle_view_replication_removal(Deserializer &derez, 
                                           AddressSpaceID source);
#endif
      void handle_manager_request(Deserializer &derez, AddressSpaceID source);
      void handle_future_result(Deserializer &derez);
      void handle_future_subscription(Deserializer &derez, 
                                      AddressSpaceID source);
      void handle_future_notification(Deserializer &derez,
                                      AddressSpaceID source);
      void handle_future_broadcast(Deserializer &derez);
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
      void handle_compute_equivalence_sets_request(Deserializer &derez, 
                                                   AddressSpaceID source);
      void handle_equivalence_set_request(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_equivalence_set_response(Deserializer &derez,
                                           AddressSpaceID source);
      void handle_equivalence_set_subset_request(Deserializer &derez);
      void handle_equivalence_set_subset_response(Deserializer &derez);
      void handle_equivalence_set_subset_update(Deserializer &derez);
      void handle_equivalence_set_ray_trace_request(Deserializer &derez,
                                                    AddressSpaceID source);
      void handle_equivalence_set_ray_trace_response(Deserializer &derez);
      void handle_equivalence_set_migration(Deserializer &derez,
                                            AddressSpaceID source);
      void handle_equivalence_set_owner_update(Deserializer &derez);
      void handle_equivalence_set_remote_refinement(Deserializer &derez);
      void handle_equivalence_set_remote_request_instances(Deserializer &derez, 
                                                         AddressSpaceID srouce);
      void handle_equivalence_set_remote_request_invalid(Deserializer &derez, 
                                                         AddressSpaceID srouce);
      void handle_equivalence_set_remote_updates(Deserializer &derez,
                                                 AddressSpaceID source);
      void handle_equivalence_set_remote_acquires(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_equivalence_set_remote_releases(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_equivalence_set_remote_copies_across(Deserializer &derez,
                                                       AddressSpaceID source);
      void handle_equivalence_set_remote_overwrites(Deserializer &derez,
                                                    AddressSpaceID source);
      void handle_equivalence_set_remote_filters(Deserializer &derez,
                                                 AddressSpaceID source);
      void handle_equivalence_set_remote_instances(Deserializer &derez);
      void handle_equivalence_set_stale_update(Deserializer &derez); 
      void handle_instance_request(Deserializer &derez, AddressSpaceID source);
      void handle_instance_response(Deserializer &derez,AddressSpaceID source);
      void handle_external_create_request(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_external_create_response(Deserializer &derez);
      void handle_external_attach(Deserializer &derez);
      void handle_external_detach(Deserializer &derez);
      void handle_gc_priority_update(Deserializer &derez,AddressSpaceID source);
      void handle_never_gc_response(Deserializer &derez);
      void handle_acquire_request(Deserializer &derez, AddressSpaceID source);
      void handle_acquire_response(Deserializer &derez, AddressSpaceID source);
      void handle_variant_request(Deserializer &derez, AddressSpaceID source);
      void handle_variant_response(Deserializer &derez);
      void handle_variant_broadcast(Deserializer &derez);
      void handle_constraint_request(Deserializer &derez,AddressSpaceID source);
      void handle_constraint_response(Deserializer &derez,AddressSpaceID src);
      void handle_constraint_release(Deserializer &derez);
      void handle_top_level_task_request(Deserializer &derez);
      void handle_top_level_task_complete(Deserializer &derez);
      void handle_mpi_rank_exchange(Deserializer &derez);
      void handle_library_mapper_request(Deserializer &derez,
                                         AddressSpaceID source);
      void handle_library_mapper_response(Deserializer &derez);
      void handle_library_trace_request(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_library_trace_response(Deserializer &derez);
      void handle_library_projection_request(Deserializer &derez,
                                             AddressSpaceID source);
      void handle_library_projection_response(Deserializer &derez);
      void handle_library_task_request(Deserializer &derez,
                                       AddressSpaceID source);
      void handle_library_task_response(Deserializer &derez);
      void handle_library_redop_request(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_library_redop_response(Deserializer &derez);
      void handle_library_serdez_request(Deserializer &derez,
                                         AddressSpaceID source);
      void handle_library_serdez_response(Deserializer &derez);
      void handle_remote_op_report_uninitialized(Deserializer &derez);
      void handle_remote_op_profiling_count_update(Deserializer &derez);
      void handle_remote_tracing_update(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_remote_tracing_response(Deserializer &derez);
      void handle_remote_tracing_eq_request(Deserializer &derez,
                                            AddressSpaceID source);
      void handle_remote_tracing_eq_response(Deserializer &derez);
      void handle_shutdown_notification(Deserializer &derez, 
                                        AddressSpaceID source);
      void handle_shutdown_response(Deserializer &derez);
    public: // Calls to handle mapper requests
      bool create_physical_instance(Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, MapperID mapper_id,
                                    Processor processor, bool acquire, 
                                    GCPriority priority, bool tight_bounds,
                                    size_t *footprint, UniqueID creator_id);
      bool create_physical_instance(Memory target_memory, 
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, MapperID mapper_id,
                                    Processor processor, bool acquire, 
                                    GCPriority priority, bool tight_bounds,
                                    size_t *footprint, UniqueID creator_id);
      bool find_or_create_physical_instance(Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    MapperID mapper_id, Processor processor,
                                    bool acquire, GCPriority priority,
                                    bool tight_bounds, size_t *footprint,
                                    UniqueID creator_id);
      bool find_or_create_physical_instance(Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    MapperID mapper_id, Processor processor,
                                    bool acquire, GCPriority priority,
                                    bool tight_bounds, size_t *footprint,
                                    UniqueID creator_id);
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
      void release_tree_instances(RegionTreeID tid);
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
      void add_to_dependence_queue(TaskContext *ctx, Processor p,
                                   Operation *op, const bool unordered = false);
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
                                             LgPriority lg_priority,
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
      EquivalenceSet* find_or_request_equivalence_set(DistributedID did,
                                                      RtEvent &ready);
    protected:
      template<typename T, MessageKind MK, VirtualChannelKind VC>
      DistributedCollectable* find_or_request_distributed_collectable(
                                            DistributedID did, RtEvent &ready);
    public:
      FutureImpl* find_or_create_future(DistributedID did,
                                        ReferenceMutator *mutator);
      FutureMapImpl* find_or_create_future_map(DistributedID did, 
                TaskContext *ctx, RtEvent complete, ReferenceMutator *mutator);
      IndexSpace find_or_create_index_slice_space(const Domain &launch_domain,
                                                  TypeTag type_tag);
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
      CreationOp*           get_available_creation_op(void);
      DeletionOp*           get_available_deletion_op(void);
      MergeCloseOp*         get_available_merge_close_op(void);
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
      TraceReplayOp*        get_available_replay_op(void);
      TraceBeginOp*         get_available_begin_op(void);
      TraceSummaryOp*       get_available_summary_op(void);
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
      AllReduceOp*          get_available_all_reduce_op(void);
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
      void free_creation_op(CreationOp *op);
      void free_deletion_op(DeletionOp *op);
      void free_merge_close_op(MergeCloseOp *op); 
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
      void free_replay_op(TraceReplayOp *op);
      void free_begin_op(TraceBeginOp *op);
      void free_summary_op(TraceSummaryOp *op);
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
      void free_all_reduce_op(AllReduceOp *op);
    public:
      RegionTreeContext allocate_region_tree_context(void);
      void free_region_tree_context(RegionTreeContext tree_ctx); 
      void register_local_context(UniqueID context_uid, InnerContext *ctx);
      void unregister_local_context(UniqueID context_uid);
      void register_remote_context(UniqueID context_uid, RemoteContext *ctx,
                                   std::set<RtEvent> &preconditions);
      void unregister_remote_context(UniqueID context_uid);
      InnerContext* find_context(UniqueID context_uid, 
                                 bool return_null_if_not_found = false,
                                 RtEvent *wait_for = NULL);
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
      IndexSpaceExprID   get_unique_index_space_expr_id(void);
    public:
      // Verify that a region requirement is valid
      LegionErrorType verify_requirement(const RegionRequirement &req,
                                         FieldID &bad_field);
    public:
      // Methods for helping with dumb nested class scoping problems
      Future help_create_future(ApEvent complete, Operation *op = NULL);
      bool help_reset_future(const Future &f);
      IndexSpace help_create_index_space_handle(TypeTag type_tag);
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
      static void initialize_runtime_task(
                          const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
      static void shutdown_runtime_task(
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
      static void startup_runtime_task(
                          const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
      static void endpoint_runtime_task(
                          const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
    protected:
      static void configure_collective_settings(int total_spaces);
    protected:
      // Internal runtime methods invoked by the above static methods
      // after the find the right runtime instance to call
      void process_schedule_request(Processor p);
      void process_message_task(const void *args, size_t arglen); 
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
      MessageManager *message_managers[LEGION_MAX_NUM_NODES];
      // Pending message manager requests
      std::map<AddressSpaceID,RtUserEvent> pending_endpoint_requests;
      // For every processor map it to its address space
      const std::map<Processor,AddressSpaceID> proc_spaces;
      // For every endpoint processor map to its address space
      std::map<Processor,AddressSpaceID> endpoint_spaces;
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
      unsigned unique_is_expr_id;
      unsigned unique_task_id;
      unsigned unique_mapper_id;
      unsigned unique_trace_id;
      unsigned unique_projection_id;
      unsigned unique_redop_id;
      unsigned unique_serdez_id;
    protected:
      mutable LocalLock library_lock;
      struct LibraryMapperIDs {
      public:
        MapperID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string,LibraryMapperIDs> library_mapper_ids;
      // This is only valid on node 0
      unsigned unique_library_mapper_id;
    protected:
      struct LibraryTraceIDs {
        TraceID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string,LibraryTraceIDs> library_trace_ids;
      // This is only valid on node 0
      unsigned unique_library_trace_id;
    protected:
      struct LibraryProjectionIDs {
      public:
        ProjectionID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string,LibraryProjectionIDs> library_projection_ids;
      // This is only valid on node 0
      unsigned unique_library_projection_id;
    protected:
      struct LibraryTaskIDs {
      public:
        TaskID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string,LibraryTaskIDs> library_task_ids;
      // This is only valid on node 0
      unsigned unique_library_task_id;
    protected:
      struct LibraryRedopIDs {
      public:
        ReductionOpID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string,LibraryRedopIDs> library_redop_ids;
      // This is only valid on node 0
      unsigned unique_library_redop_id;
    protected:
      struct LibrarySerdezIDs {
      public:
        CustomSerdezID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string,LibrarySerdezIDs> library_serdez_ids;
      // This is only valid on node 0
      unsigned unique_library_serdez_id;
    protected:
      mutable LocalLock redop_lock;
      mutable LocalLock serdez_lock;
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
      mutable LocalLock is_slice_lock;
      std::map<std::pair<Domain,TypeTag>,IndexSpace> index_slice_spaces;
    protected:
      // The runtime keeps track of remote contexts so they
      // can be re-used by multiple tasks that get sent remotely
      mutable LocalLock context_lock;
      std::map<UniqueID,InnerContext*> local_contexts;
      LegionMap<UniqueID,RemoteContext*,
                RUNTIME_REMOTE_ALLOC>::tracked remote_contexts;
      std::map<UniqueID,
        std::pair<RtUserEvent,RemoteContext*> > pending_remote_contexts;
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
      std::map<AllocationType,AllocationTracker> allocation_manager;
      unsigned long long allocation_tracing_count;
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
      mutable LocalLock creation_op_lock;
      mutable LocalLock deletion_op_lock;
      mutable LocalLock merge_close_op_lock;
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
      mutable LocalLock replay_op_lock;
      mutable LocalLock begin_op_lock;
      mutable LocalLock summary_op_lock;
      mutable LocalLock epoch_op_lock;
      mutable LocalLock pending_partition_op_lock;
      mutable LocalLock dependent_partition_op_lock;
      mutable LocalLock fill_op_lock;
      mutable LocalLock attach_op_lock;
      mutable LocalLock detach_op_lock;
      mutable LocalLock timing_op_lock;
      mutable LocalLock all_reduce_op_lock;
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
      std::deque<CreationOp*>           available_creation_ops;
      std::deque<DeletionOp*>           available_deletion_ops;
      std::deque<MergeCloseOp*>         available_merge_close_ops;
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
      std::deque<TraceReplayOp*>        available_replay_ops;
      std::deque<TraceBeginOp*>         available_begin_ops;
      std::deque<TraceSummaryOp*>       available_summary_ops;
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
      std::deque<AllReduceOp*>          available_all_reduce_ops;
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
          LayoutConstraintID id, DistributedID did = 0);
      LayoutConstraints* register_layout(FieldSpace handle,
               const LayoutConstraintSet &cons, bool internal);
      bool register_layout(LayoutConstraints *new_constraints,
                           ReferenceMutator *mutator);
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
                                                 bool can_fail = false,
                                                 RtEvent *wait_for = NULL);
    public:
      // Static methods for start-up and callback phases
      static int start(int argc, char **argv, bool background);
      static void register_builtin_reduction_operators(void);
      static const LegionConfiguration& initialize(int *argc, char ***argv, 
                                                   bool filter);
      static LegionConfiguration parse_arguments(int argc, char **argv);
      static void perform_slow_config_checks(const LegionConfiguration &config);
      static void configure_interoperability(bool separate_runtimes);
      static RtEvent configure_runtime(int argc, char **argv,
          const LegionConfiguration &config, RealmRuntime &realm,
          Processor::Kind &startup_kind);
      static int wait_for_shutdown(void);
      Future launch_top_level_task(const TaskLauncher &launcher);
      Context begin_implicit_task(TaskID top_task_id,
                                  MapperID top_mapper_id,
                                  Processor::Kind proc_kind,
                                  const char *task_name,
                                  bool control_replicable,
                                  unsigned shard_per_address_space,
                                  int shard_id);
      void finish_implicit_task(Context ctx);
      static void set_top_level_task_id(TaskID top_id);
      static void set_top_level_task_mapper_id(MapperID mapper_id);
      static void configure_MPI_interoperability(int rank);
      static void register_handshake(LegionHandshake &handshake);
      static const ReductionOp* get_reduction_op(ReductionOpID redop_id,
                                                 bool has_lock = false);
      static const SerdezOp* get_serdez_op(CustomSerdezID serdez_id,
                                           bool has_lock = false);
      static const SerdezRedopFns* get_serdez_redop_fns(ReductionOpID redop_id,
                                                        bool has_lock = false);
      static void add_registration_callback(RegistrationCallbackFnptr callback);
      static ReductionOpTable& get_reduction_table(bool safe);
      static SerdezOpTable& get_serdez_table(bool safe);
      static SerdezRedopTable& get_serdez_redop_table(bool safe);
      static void register_reduction_op(ReductionOpID redop_id,
                                        ReductionOp *redop,
                                        SerdezInitFnptr init_fnptr,
                                        SerdezFoldFnptr fold_fnptr,
                                        bool permit_duplicates,
                                        bool has_lock = false);
      static void register_serdez_op(CustomSerdezID serdez_id,
                                     SerdezOp *serdez_op,
                                     bool permit_duplicates,
                                     bool has_lock = false);
      static std::deque<PendingVariantRegistration*>&
                                get_pending_variant_table(void);
      static std::map<LayoutConstraintID,LayoutConstraintRegistrar>&
                                get_pending_constraint_table(void);
      static std::map<ProjectionID,ProjectionFunctor*>&
                                get_pending_projection_table(void);
      static std::vector<LegionHandshake>&
                                get_pending_handshake_table(void);
      static std::vector<RegistrationCallbackFnptr>&
                                get_pending_registration_callbacks(void);
      static TaskID& get_current_static_task_id(void);
      static TaskID generate_static_task_id(void);
      static VariantID preregister_variant(
                      const TaskVariantRegistrar &registrar,
                      const void *user_data, size_t user_data_size,
                      CodeDescriptor *realm_desc, bool has_ret, 
                      const char *task_name,VariantID vid,bool check_id = true);
    public:
      static ReductionOpID& get_current_static_reduction_id(void);
      static ReductionOpID generate_static_reduction_id(void);
      static CustomSerdezID& get_current_static_serdez_id(void);
      static CustomSerdezID generate_static_serdez_id(void);
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
    public:
      // Static member variables
      static TaskID legion_main_id;
      static MapperID legion_main_mapper_id;
      static std::vector<RegistrationCallbackFnptr> registration_callbacks;
      static bool legion_main_set;
      static bool runtime_initialized;
      static bool runtime_started;
      static bool runtime_backgrounded;
      static Runtime *the_runtime;
      static RtUserEvent runtime_started_event;
      // Static member variables for MPI interop
      static int mpi_rank;
    public:
      static inline ApEvent merge_events(const TraceInfo *info,
                                         ApEvent e1, ApEvent e2);
      static inline ApEvent merge_events(const TraceInfo *info,
                                         ApEvent e1, ApEvent e2, ApEvent e3);
      static inline ApEvent merge_events(const TraceInfo *info,
                                         const std::set<ApEvent> &events);
    public:
      static inline RtEvent merge_events(RtEvent e1, RtEvent e2);
      static inline RtEvent merge_events(RtEvent e1, RtEvent e2, RtEvent e3);
      static inline RtEvent merge_events(const std::set<RtEvent> &events);
    public:
      static inline ApUserEvent create_ap_user_event(
                                                  const TraceInfo *info = NULL);
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
                    LgPriority priority, RtEvent precondition, Processor target)
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
        profiler->add_meta_request(requests, T::TASK_ID, args.provenance);
#ifdef LEGION_SEPARATE_META_TASKS
        return RtEvent(target.spawn(LG_TASK_ID + T::TASK_ID, &args, sizeof(T),
                                    requests, precondition, priority));
#else
        return RtEvent(target.spawn(LG_TASK_ID, &args, sizeof(T),
                                    requests, precondition, priority));
#endif
      }
      else
#ifdef LEGION_SEPARATE_META_TASKS
        return RtEvent(target.spawn(LG_TASK_ID + T::TASK_ID, &args, sizeof(T),
                                    precondition, priority));
#else
        return RtEvent(target.spawn(LG_TASK_ID, &args, sizeof(T), 
                                    precondition, priority));
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApEvent Runtime::merge_events(
                                  const TraceInfo *info, ApEvent e1, ApEvent e2)
    //--------------------------------------------------------------------------
    {
      ApEvent result(Realm::Event::merge_events(e1, e2)); 
#ifdef LEGION_DISABLE_EVENT_PRUNING
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
#endif
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(e1, result);
      LegionSpy::log_event_dependence(e2, result);
#endif
      if ((info != NULL) && info->recording)
        info->record_merge_events(result, e1, e2);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApEvent Runtime::merge_events(
                      const TraceInfo *info, ApEvent e1, ApEvent e2, ApEvent e3) 
    //--------------------------------------------------------------------------
    {
      ApEvent result(Realm::Event::merge_events(e1, e2, e3)); 
#ifdef LEGION_DISABLE_EVENT_PRUNING
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
#endif
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(e1, result);
      LegionSpy::log_event_dependence(e2, result);
      LegionSpy::log_event_dependence(e3, result);
#endif
      if ((info != NULL) && info->recording)
        info->record_merge_events(result, e1, e2, e3);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApEvent Runtime::merge_events(
                         const TraceInfo *info, const std::set<ApEvent> &events)
    //--------------------------------------------------------------------------
    {
#ifndef LEGION_DISABLE_EVENT_PRUNING
      if (events.empty())
      {
        // Still need to do this for tracing because of merge filter code
        if ((info != NULL) && info->recording)
        {
          ApEvent result;
          info->record_merge_events(result, events);
          return result;
        }
        else
          return ApEvent::NO_AP_EVENT;
      }
      if (events.size() == 1)
      {
        // Still need to do this for tracing because of merge filter code
        if ((info != NULL) && info->recording)
        {
          ApEvent result = *(events.begin());
          info->record_merge_events(result, events);
          return result;
        }
        else
          return *(events.begin());
      }
#endif
      const std::set<Realm::Event> *realm_events = 
        reinterpret_cast<const std::set<Realm::Event>*>(&events);
      ApEvent result(Realm::Event::merge_events(*realm_events));
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (events.find(result) != events.end()))
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        if (events.find(result) != events.end())
          rename.trigger(result);
        else
          rename.trigger();
        result = ApEvent(rename);
      }
#endif
#ifdef LEGION_SPY
      for (std::set<ApEvent>::const_iterator it = events.begin();
            it != events.end(); it++)
        LegionSpy::log_event_dependence(*it, result);
#endif
      if ((info != NULL) && info->recording)
        info->record_merge_events(result, events);
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
#ifndef LEGION_DISABLE_EVENT_PRUNING
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
    /*static*/ inline ApUserEvent Runtime::create_ap_user_event(
                                                          const TraceInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      ApUserEvent result(Realm::UserEvent::create_user_event());
      LegionSpy::log_ap_user_event(result);
#else
      ApUserEvent result(Realm::UserEvent::create_user_event());
#endif
      if ((info != NULL) && info->recording)
        info->record_create_ap_user_event(result);
      return result;
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
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        result = ApEvent(rename);
      }
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(ApEvent(e), result);
#endif
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
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (precondition.exists() && !result.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        result = ApEvent(rename);
      }
#endif
#ifdef LEGION_SPY
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

