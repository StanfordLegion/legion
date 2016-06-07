/* Copyright 2016 Stanford University, NVIDIA Corporation
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
#include "region_tree.h"
#include "legion_utilities.h"
#include "legion_allocation.h"
#include "garbage_collection.h"

namespace LegionRuntime {
  namespace HighLevel { 

    // Special helper for when we need a dummy context
#define DUMMY_CONTEXT       0

    /**
     * A class for deduplicating memory used with task arguments
     * and knowing when to collect the data associated with it
     */
    class AllocManager : public Collectable {
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
     * \class ArgumentMap::Impl
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
    class ArgumentMap::Impl : public Collectable {
    public:
      static const AllocationType alloc_type = ARGUMENT_MAP_ALLOC;
    public:
      FRIEND_ALL_RUNTIME_CLASSES
      Impl(void);
      Impl(ArgumentMapStore *st);
      Impl(ArgumentMapStore *st, const std::map<DomainPoint,TaskArgument> &);
      Impl(const Impl &impl);
      ~Impl(void);
    public:
      Impl& operator=(const Impl &rhs);
    public:
      bool has_point(const DomainPoint &point);
      void set_point(const DomainPoint &point, const TaskArgument &arg,
                     bool replace);
      bool remove_point(const DomainPoint &point);
      TaskArgument get_point(const DomainPoint &point) const;
    public:
      void pack_arguments(Serializer &rez, const Domain &domain);
      void unpack_arguments(Deserializer &derez);
    protected:
      Impl* freeze(void);
      Impl* clone(void);
    private:
      std::map<DomainPoint,TaskArgument> arguments;
      Impl *next;
      ArgumentMapStore *const store;
      bool frozen;
    };

    /**
     * \class ArgumentMapStore
     * Argument map stores are the backing stores for a chain of
     * argument maps so that the actual values of the arguments do
     * not need to be duplicated across every version of the
     * argument map.
     */
    class ArgumentMapStore {
    public:
      static const AllocationType alloc_type = ARGUMENT_MAP_STORE_ALLOC;
    public:
      FRIEND_ALL_RUNTIME_CLASSES
      ArgumentMapStore(void);
      ArgumentMapStore(const ArgumentMapStore &rhs);
      ~ArgumentMapStore(void);
    public:
      ArgumentMapStore& operator=(const ArgumentMapStore &rhs);
    public:
      TaskArgument add_arg(const TaskArgument &arg);
    private:
      std::set<TaskArgument> values;
    };

    /**
     * \class Future::Impl
     * The base implementation of a future object.  The runtime
     * manages future implementation objects and knows how to
     * copy them from one node to another.  Future implementations
     * are always made first on the owner node and then moved
     * remotely.  We use the distributed collectable scheme
     * to manage garbage collection of distributed futures
     */
    class Future::Impl : public DistributedCollectable {
    public:
      static const AllocationType alloc_type = FUTURE_ALLOC;
    public:
      struct ContributeCollectiveArgs {
      public:
        HLRTaskID hlr_id;
        Future::Impl *impl;
        Barrier barrier;
        unsigned count;
      };
    public:
      Impl(Internal *rt, bool register_future, DistributedID did, 
           AddressSpaceID owner_space, AddressSpaceID local_space,
           Operation *op = NULL);
      Impl(const Future::Impl &rhs);
      virtual ~Impl(void);
    public:
      Impl& operator=(const Future::Impl &rhs);
    public:
      void get_void_result(void);
      void* get_untyped_result(void);
      bool is_empty(bool block);
      size_t get_untyped_size(void);
      Event get_ready_event(void) const { return ready_event; }
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
      virtual void notify_active(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
      virtual void notify_inactive(void);
    public:
      void register_dependence(Operation *consumer_op);
    protected:
      void mark_sampled(void);
      void broadcast_result(void);
      void send_future(AddressSpaceID sid);
      void register_waiter(AddressSpaceID sid);
    public:
      static void handle_future_send(Deserializer &derez, Internal *rt, 
                                     AddressSpaceID source);
      static void handle_future_result(Deserializer &derez, Internal *rt);
      static void handle_future_subscription(Deserializer &derez, Internal *rt);
    public:
      void contribute_to_collective(Barrier barrier, unsigned count);
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
      UserEvent ready_event;
      void *result; 
      size_t result_size;
      volatile bool empty;
      volatile bool sampled;
      // On the owner node, keep track of the registered waiters
      std::set<AddressSpaceID> registered_waiters;
    };

    /**
     * \class FutureMap::Impl
     * The base implementation of a future map object.  Note
     * that while future objects can move from one node to
     * another, future maps will never leave the node on
     * which they are created.  The futures contained within
     * a future map are permitted to migrate.
     */
    class FutureMap::Impl : public Collectable {
    public:
      static const AllocationType alloc_type = FUTURE_MAP_ALLOC;
    public:
      Impl(SingleTask *ctx, TaskOp *task, Internal *rt);
      Impl(SingleTask *ctx, Event completion_event, Internal *rt);
      Impl(SingleTask *ctx, Internal *rt); // empty map
      Impl(const FutureMap::Impl &rhs);
      ~Impl(void);
    public:
      Impl& operator=(const FutureMap::Impl &rhs);
    public:
      Future get_future(const DomainPoint &point);
      void get_void_result(const DomainPoint &point);
      void wait_all_results(void);
      void complete_all_futures(void);
      bool reset_all_futures(void);
#ifdef DEBUG_HIGH_LEVEL
    public:
      void add_valid_domain(const Domain &d);
      void add_valid_point(const DomainPoint &dp);
#endif
    public:
      SingleTask *const context;
      TaskOp *const task;
      const GenerationID task_gen;
      const bool valid;
      Internal *const runtime;
    private:
      Event ready_event;
      std::map<DomainPoint,Future> futures;
      // Unlike futures, the future map is never used remotely
      // so it can create and destroy its own lock.
      Reservation lock;
#ifdef DEBUG_HIGH_LEVEL
    private:
      std::vector<Domain> valid_domains;
      std::set<DomainPoint> valid_points;
#endif
    };

    /**
     * \class PhysicalRegion::Impl
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
    class PhysicalRegion::Impl : public Collectable {
    public:
      static const AllocationType alloc_type = PHYSICAL_REGION_ALLOC;
    public:
      Impl(const RegionRequirement &req, Event ready_event,
           bool mapped, SingleTask *ctx, MapperID mid,
           MappingTagID tag, bool leaf, Internal *rt);
      Impl(const PhysicalRegion::Impl &rhs);
      ~Impl(void);
    public:
      Impl& operator=(const PhysicalRegion::Impl &rhs);
    public:
      void wait_until_valid(void);
      bool is_valid(void) const;
      bool is_mapped(void) const;
      LogicalRegion get_logical_region(void) const;
      Accessor::RegionAccessor<Accessor::AccessorType::Generic> 
        get_accessor(void);
      Accessor::RegionAccessor<Accessor::AccessorType::Generic> 
        get_field_accessor(FieldID field);
    public:
      void unmap_region(void);
      void remap_region(Event new_ready_event);
      const RegionRequirement& get_requirement(void) const;
      void set_reference(const InstanceRef &ref);
      void reset_reference(const InstanceRef &ref, 
                           UserEvent term_event);
      Event get_ready_event(void) const;
      const InstanceRef& get_reference(void) const;
      void get_memories(std::set<Memory>& memories) const;
      void get_fields(std::vector<FieldID>& fields) const;
#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
    public:
      const char* get_task_name(void) const;
#endif
#ifdef BOUNDS_CHECKS
    public:
      bool contains_ptr(ptr_t ptr) const;
      bool contains_point(const DomainPoint &dp) const;
#endif
    public:
      Internal *const runtime;
      SingleTask *const context;
      const MapperID map_id;
      const MappingTagID tag;
      const bool leaf_region;
    private:
      // Event for when the instance ref is ready
      Event ready_event;
      // Instance ref
      InstanceRef reference;
      RegionRequirement req;
      bool mapped; // whether it is currently mapped
      bool valid; // whether it is currently valid
      // whether to trigger the termination event
      // upon unmap
      bool trigger_on_unmap;
      UserEvent termination_event;
#ifdef BOUNDS_CHECKS
    private:
      Domain bounds;
#endif
    };

    /**
     * \class Grant::Impl
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
    class Grant::Impl : public Collectable {
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
      Impl(void);
      Impl(const std::vector<ReservationRequest> &requests);
      Impl(const Grant::Impl &rhs);
      ~Impl(void);
    public:
      Impl& operator=(const Grant::Impl &rhs);
    public:
      void register_operation(Event completion_event);
      Event acquire_grant(void);
      void release_grant(void);
    public:
      void pack_grant(Serializer &rez);
      void unpack_grant(Deserializer &derez);
    private:
      std::vector<ReservationRequest> requests;
      bool acquired;
      Event grant_event;
      std::set<Event> completion_events;
      Reservation grant_lock;
    };

    class MPILegionHandshake::Impl : public Collectable {
    public:
      static const AllocationType alloc_type = MPI_HANDSHAKE_ALLOC;
    public:
      enum ControlState {
        IN_MPI,
        IN_LEGION,
      };
    public:
      Impl(bool in_mpi, int mpi_participants, int legion_participants);
      Impl(const Impl &rhs);
      ~Impl(void);
    public:
      Impl& operator=(const Impl &rhs);
    public:
      void mpi_handoff_to_legion(void);
      void mpi_wait_on_legion(void);
    public:
      void legion_handoff_to_mpi(void);
      void legion_wait_on_mpi(void);
    private:
      const int mpi_participants;
      const int legion_participants;
      ControlState state;
    private:
      int mpi_count, legion_count;
    private:
      UserEvent mpi_ready, legion_ready;
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
      struct TriggerOpArgs {
      public:
        HLRTaskID hlr_id;
        Operation *op;
        ProcessorManager *manager;
      };
      struct SchedulerArgs {
      public:
        HLRTaskID hlr_id;
        Processor proc;
      };
      struct TriggerTaskArgs {
      public:
        HLRTaskID hlr_id;
        TaskOp *op;
        ProcessorManager *manager;
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
                       Internal *rt,
                       unsigned width, unsigned default_mappers,  
                       bool no_steal, unsigned max_steals);
      ProcessorManager(const ProcessorManager &rhs);
      ~ProcessorManager(void);
    public:
      ProcessorManager& operator=(const ProcessorManager &rhs);
    public:
      void add_mapper(MapperID mid, Mapper *m, bool check);
      void replace_default_mapper(Mapper *m);
      Mapper* find_mapper(MapperID mid) const; 
    public:
      template<typename MAPPER_CONTINUATION>
      void invoke_mapper(MapperID map_id, MAPPER_CONTINUATION &continuation,
                         bool block, bool has_lock);
      void check_mapper_messages(MapperID map_id, bool need_lock);
    public:
      // Functions that perform mapping calls
      // We really need variadic templates here
      void invoke_mapper_set_task_options(TaskOp *task);
      bool invoke_mapper_pre_map_task(TaskOp *task);
      void invoke_mapper_select_variant(TaskOp *task);
      bool invoke_mapper_map_task(TaskOp *task);
      void invoke_mapper_post_map_task(TaskOp *task);
      void invoke_mapper_failed_mapping(Mappable *mappable);
      void invoke_mapper_notify_result(Mappable *mappable);
      void invoke_mapper_slice_domain(TaskOp *task,
                                      std::vector<Mapper::DomainSplit> &splits);
      bool invoke_mapper_map_inline(Inline *op);
      bool invoke_mapper_map_copy(Copy *op);
      bool invoke_mapper_speculate(Mappable *op, bool &value); 
      void invoke_mapper_configure_context(TaskOp *task);
      bool invoke_mapper_rank_copy_targets(Mappable *mappable,
                                           LogicalRegion handle, 
                                           const std::set<Memory> &memories,
                                           bool complete,
                                           size_t max_blocking_factor,
                                           std::set<Memory> &to_reuse,
                                           std::vector<Memory> &to_create,
                                           bool &create_one,
                                           size_t &blocking_factor);
      void invoke_mapper_rank_copy_sources(Mappable *mappable,
                                           const std::set<Memory> &memories,
                                           Memory destination,
                                           std::vector<Memory> &order);
      void invoke_mapper_notify_profiling(TaskOp *task);
      bool invoke_mapper_map_must_epoch(const std::vector<Task*> &tasks,
          const std::vector<Mapper::MappingConstraint> &constraints,
          MapperID map_id, MappingTagID tag);
      int invoke_mapper_get_tunable_value(TaskOp *task, TunableID tid,
                                          MapperID mid, MappingTagID tag);
      void invoke_mapper_handle_message(MapperID map_id, Processor source,
                                        const void *message, size_t length);
      void invoke_mapper_task_result(MapperID map_id, Event event,
                                     const void *result, size_t result_size);
      void invoke_mapper_permit_task_steal(MapperID map_id, Processor thief, 
                                     const std::vector<const Task*> &stealable,
                                     std::set<const Task*> &to_steal);
      // This method cannot support waiting
      void invoke_mapper_target_task_steal(MapperID map_id,
                                           const std::set<Processor> &blacklist,
                                           std::set<Processor> &steal_targets);
      // This method cannot support waiting
      void invoke_mapper_select_tasks_to_schedule(MapperID map_id,
                                           const std::list<Task*> &ready_tasks);
    public:
      // Handle mapper messages
      void defer_mapper_message(Processor target, MapperID map_id,
                                const void *message, size_t length);
      void defer_mapper_broadcast(MapperID map_id, const void *message,
                                  size_t length, int radix);
      void defer_mapper_call(MapperID map_id, Event wait_on);
      void send_mapper_messages(MapperID map_id, 
                                std::vector<MapperMessage> &messages);
    public:
      void perform_scheduling(void);
      void launch_task_scheduler(void);
    public:
      void activate_context(SingleTask *context);
      void deactivate_context(SingleTask *context);
      void update_max_context_count(unsigned max_contexts);
    public:
      void process_steal_request(Processor thief, 
                                 const std::vector<MapperID> &thieves);
      void process_advertisement(Processor advertiser, MapperID mid);
    public:
      void add_to_ready_queue(TaskOp *op, bool previous_failure);
      void add_to_local_ready_queue(Operation *op, bool previous_failure);
    public:
      // Mapper introspection methods
      unsigned sample_unmapped_tasks(MapperID map_id);
#ifdef HANG_TRACE
    public:
      void dump_state(FILE *target);
#endif
    protected:
      void perform_mapping_operations(void);
      void issue_advertisements(MapperID mid);
    protected:
      void increment_active_contexts(void);
      void decrement_active_contexts(void);
    public:
      // Immutable state
      Internal *const runtime;
      const Processor local_proc;
      const Processor::Kind proc_kind;
      // Effective super-scalar width of the runtime
      const unsigned superscalar_width;
      // Is stealing disabled 
      const bool stealing_disabled;
      // Maximum number of outstanding steals permitted by any mapper
      const unsigned max_outstanding_steals;
    protected:
      // Local queue state
      Reservation local_queue_lock;
      unsigned next_local_index;
      std::vector<Event> local_scheduler_preconditions;
    protected:
      // Scheduling state
      Reservation queue_lock;
      bool task_scheduler_enabled;
      unsigned total_active_contexts;
      struct ContextState {
      public:
        ContextState(void)
          : active(false), owned_tasks(0) { }
      public:
        bool active;
        unsigned owned_tasks;
      };
      std::vector<ContextState> context_states;
    protected:
      // Reservation for protecting the list of messages that
      // need to be sent
      Reservation message_lock;
      // For each mapper, a list of tasks that are ready to map
      std::vector<std::list<TaskOp*> > ready_queues;
      // Mapper objects
      std::vector<Mapper*> mapper_objects;
      // Mapper locks
      std::vector<Reservation> mapper_locks;
      // Pending mapper messages
      std::vector<std::vector<MapperMessage> > mapper_messages;
      // Keep track of whether we are inside of a mapper call
      std::vector<bool> inside_mapper_call;
      // Events to wait on before retrying the mapper call
      std::vector<Event> defer_mapper_event;
      // For each mapper, the set of processors to which it
      // has outstanding steal requests
      std::map<MapperID,std::set<Processor> > outstanding_steal_requests;
      // Failed thiefs to notify when tasks become available
      std::multimap<MapperID,Processor> failed_thiefs;
      // Reservations for stealing and thieving
      Reservation stealing_lock;
      Reservation thieving_lock;
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
      MemoryManager(Memory mem, Internal *rt);
      MemoryManager(const MemoryManager &rhs);
      ~MemoryManager(void);
    public:
      MemoryManager& operator=(const MemoryManager &rhs);
    public:
      // Update the manager with information about physical instances
      void allocate_physical_instance(PhysicalManager *manager);
      void free_physical_instance(PhysicalManager *manager);
    public:
      // Method for mapper introspection
      size_t sample_allocated_space(void);
      size_t sample_free_space(void);
      unsigned sample_allocated_instances(void);
    protected:
      // The memory that we are managing
      const Memory memory;
      // The capacity in bytes of this memory
      const size_t capacity;
      // The remaining capacity in this memory
      size_t remaining_capacity;
      // The runtime we are associate with
      Internal *const runtime;
      // Reservation for controlling access to the data
      // structures in this memory manager
      Reservation manager_lock;
      // Current set of physical instances and their sizes
      LegionMap<InstanceManager*, size_t, 
                MEMORY_INSTANCES_ALLOC>::tracked physical_instances;
      // Current set of reduction instances and their sizes
      LegionMap<ReductionManager*, size_t,
                MEMORY_REDUCTION_ALLOC>::tracked reduction_instances;
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
                     size_t max_message_size, bool profile_messages);
      VirtualChannel(const VirtualChannel &rhs);
      ~VirtualChannel(void);
    public:
      VirtualChannel& operator=(const VirtualChannel &rhs);
    public:
      void package_message(Serializer &rez, MessageKind k, bool flush,
                           Internal *runtime, Processor target);
      void process_message(const void *args, size_t arglen, 
                        Internal *runtime, AddressSpaceID remote_address_space);
    private:
      void send_message(bool complete, Internal *runtime, Processor target);
      void handle_messages(unsigned num_messages, Internal *runtime, 
                           AddressSpaceID remote_address_space,
                           const char *args, size_t arglen);
      void buffer_messages(unsigned num_messages,
                           const void *args, size_t arglen);
    public:
      Event notify_pending_shutdown(void);
      bool has_recent_messages(void) const;
      void clear_recent_messages(void);
    private:
      Reservation send_lock;
      char *const sending_buffer;
      unsigned sending_index;
      const size_t sending_buffer_size;
      Event last_message_event;
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
      const bool profile_messages;
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
                     Internal *rt, size_t max,
                     const std::set<Processor> &procs);
      MessageManager(const MessageManager &rhs);
      ~MessageManager(void);
    public:
      MessageManager& operator=(const MessageManager &rhs);
    public:
      Event notify_pending_shutdown(void);
      bool has_recent_messages(void) const;
      void clear_recent_messages(void);
    public:
      void send_message(Serializer &rez, MessageKind kind, 
                        VirtualChannelKind channel, bool flush);

      void receive_message(const void *args, size_t arglen);
    public:
      const AddressSpaceID remote_address_space;
    private:
      Internal *const runtime;
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
      struct NotificationArgs {
      public:
        HLRTaskID hlr_id;
        MessageManager *manager;
      };
      struct ResponseArgs {
      public:
        HLRTaskID hlr_id;
        MessageManager *target;
        bool result;
      };
    public:
      ShutdownManager(Internal *rt, AddressSpaceID source, MessageManager *man);
      ShutdownManager(const ShutdownManager &rhs);
      ~ShutdownManager(void);
    public:
      ShutdownManager& operator=(const ShutdownManager &rhs);
    public:
      bool has_managers(void) const;
      void add_manager(AddressSpaceID target, MessageManager *manager);
    public:
      void send_notifications(void);
      void send_response(void);
      bool handle_response(AddressSpaceID sender, bool result);
      void record_outstanding_tasks(void);
      void record_outstanding_profiling_requests(void);
      void finalize(void);
    public:
      Internal *const runtime;
      const AddressSpaceID source; 
      MessageManager *const source_manager;
    protected:
      Reservation shutdown_lock;
      std::map<AddressSpaceID,MessageManager*> managers;
      unsigned observed_responses;
      bool result;
    };

    /**
     * \class GarbageCollectionEpoch
     * A class for managing the a set of garbage collections
     */
    class GarbageCollectionEpoch {
    public:
      struct GarbageCollectionArgs {
      public:
        HLRTaskID hlr_id;
        GarbageCollectionEpoch *epoch;
        LogicalView *view;
      };
    public:
      GarbageCollectionEpoch(Internal *runtime);
      GarbageCollectionEpoch(const GarbageCollectionEpoch &rhs);
      ~GarbageCollectionEpoch(void);
    public:
      GarbageCollectionEpoch& operator=(const GarbageCollectionEpoch &rhs);
    public:
      void add_collection(LogicalView *view, Event term_event);
      Event launch(int priority);
      bool handle_collection(const GarbageCollectionArgs *args);
    private:
      Internal *const runtime;
      int remaining;
      std::map<LogicalView*,std::set<Event> > collections;
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
#ifdef DEBUG_HIGH_LEVEL
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
     * \class LegionContinuation
     * A generic interface class for issuing a continuation
     */
    class LegionContinuation {
    public:
      struct ContinuationArgs {
        HLRTaskID hlr_id;
        LegionContinuation *continuation;
      };
    public:
      Event defer(Internal *runtime, Event precondition = Event::NO_EVENT);
    public:
      virtual void execute(void) = 0;
    public:
      static void handle_continuation(const void *args);
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
      void perform_registration(Internal *runtime);
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
    class TaskImpl {
    public:
      static const AllocationType alloc_type = TASK_IMPL_ALLOC;
    public:
      struct SemanticRequestArgs {
        HLRTaskID hlr_id;
        TaskImpl *proxy_this;
        SemanticTag tag;
        AddressSpaceID source;
      };
    public:
      TaskImpl(TaskID tid, Internal *rt, const char *name = NULL);
      TaskImpl(const TaskImpl &rhs);
      ~TaskImpl(void);
    public:
      TaskImpl& operator=(const TaskImpl &rhs);
    public:
      inline bool returns_value(void) const { return has_return_type; }
    public:
      void add_variant(VariantImpl *impl);
      VariantImpl* find_variant_impl(VariantID variant_id);
    public:
      // For backwards compatibility only
      TaskVariantCollection* get_collection(void); 
    public:
      const char* get_name(bool needs_lock = true) const;
      void attach_semantic_information(SemanticTag tag, AddressSpaceID source,
         const void *buffer, size_t size, bool is_mutable, bool send_to_owner);
      bool retrieve_semantic_information(SemanticTag tag,
                                         const void *&buffer, size_t &size,
                                         bool can_fail, bool wait_until);
      void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                              const void *value, size_t size, bool is_mutable);
      void send_semantic_request(AddressSpaceID target, SemanticTag tag, 
                               bool can_fail, bool wait_until, UserEvent ready);
      void process_semantic_request(SemanticTag tag, AddressSpaceID target, 
                               bool can_fail, bool wait_until, UserEvent ready);
    public:
      inline AddressSpaceID get_owner_space(void) const
        { return get_owner_space(task_id, runtime); }
      static AddressSpaceID get_owner_space(TaskID task_id, Internal *runtime);
    public:
      static void handle_semantic_request(Internal *runtime, 
                          Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(Internal *runtime,
                          Deserializer &derez, AddressSpaceID source);
      static void handle_variant_request(Internal *runtime,
                          Deserializer &derez, AddressSpaceID source);
    public:
      const TaskID task_id;
      Internal *const runtime;
    private:
      Reservation task_lock;
      std::map<VariantID,VariantImpl*> variants;
      std::map<SemanticTag,SemanticInfo> semantic_infos;
      // Track whether all these variants have a return type or not
      bool has_return_type;
      // Track whether all these variants are idempotent or not
      bool all_idempotent;
    private:
      // This is for backwards compatibility only
      TaskVariantCollection *collection;
    };

    /**
     * \class VariantImpl
     * This class is used for storing all the meta-data associated
     * with a particular variant implementation of a task
     */
    class VariantImpl { 
    public:
      static const AllocationType alloc_type = VARIANT_IMPL_ALLOC;
    public:
      VariantImpl(Internal *runtime, VariantID vid, TaskImpl *owner, 
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
    public:
      Event dispatch_task(Processor target, SingleTask *task, 
                          Event precondition, int priority,
                          Realm::ProfilingRequestSet &requests);
      void dispatch_inline(Processor current, TaskOp *task);
    public:
      Processor::Kind get_processor_kind(bool warn) const;
    public:
      void send_variant_response(AddressSpaceID source, Event done_event);
    public:
      static AddressSpaceID get_owner_space(VariantID vid, Internal *runtime);
      static void handle_variant_response(Internal *runtime, 
                                          Deserializer &derez);
    public:
      const VariantID vid;
      TaskImpl *const owner;
      Internal *const runtime;
      const bool global; // globally valid variant
      const bool has_return_value; // has a return value
    public:
      CodeDescriptor *const realm_descriptor;
    private:
      ExecutionConstraintSet execution_constraints;
      TaskLayoutConstraintSet   layout_constraints;
    private:
      void *user_data;
      size_t user_data_size;
      Event ready_event;
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
     */
    class LayoutConstraints : public LayoutConstraintSet {
    public:
      static const AllocationType alloc_type = LAYOUT_CONSTRAINTS_ALLOC; 
    public:
      struct ReleaseFunctor {
      public:
        ReleaseFunctor(Serializer &r, AddressSpaceID skip, Internal *rt) 
          : rez(r), to_skip(skip), runtime(rt) { }
      public:
        void apply(AddressSpaceID target);
      private:
        Serializer &rez;
        AddressSpaceID to_skip;
        Internal *runtime;
      };
    public:
      LayoutConstraints(LayoutConstraintID layout_id, Internal *runtime, 
                        const LayoutConstraintRegistrar &registrar);
      LayoutConstraints(LayoutConstraintID layout_id, Internal *runtime);
      LayoutConstraints(const LayoutConstraints &rhs);
      ~LayoutConstraints(void);
    public:
      LayoutConstraints& operator=(const LayoutConstraints &rhs);
    public:
      inline FieldSpace get_field_space(void) const { return handle; }
      inline Event get_ready_event(void) const { return ready_event; }
      inline const char* get_name(void) const { return constraints_name; }
    public:
      void release_constraints(AddressSpaceID source);
      void send_constraints(AddressSpaceID source, UserEvent done_event);
      void update_constraints(Deserializer &derez);
    public:
      AddressSpaceID get_owner(void) const;
      static AddressSpaceID get_owner(LayoutConstraintID layout_id,
                                      Internal *runtime);
    public:
      static void process_request(Internal *runtime, Deserializer &derez,
                                  AddressSpaceID source);
      static void process_response(Internal *runtime, Deserializer &derez);
      static void process_release(Internal *runtime, Deserializer &derez,
                                  AddressSpaceID source);
    public:
      const LayoutConstraintID layout_id;
      Internal *const runtime;
    protected:
      FieldSpace handle;
      Event ready_event;
      char *constraints_name;
    protected:
      Reservation layout_lock;
      NodeSet remote_instances; // only valid on the owner node
    };

    /**
     * \class Internal 
     * This is the actual implementation of the Legion runtime functionality
     * that implements the underlying interface for the Runtime 
     * objects.  Most of the calls in the Runtime class translate
     * directly to calls to this interface.  Unfortunately this adds
     * an extra function call overhead to every runtime call because C++
     * is terrible and doesn't have mix-in classes.
     */
    class Internal {
    public:
      struct DeferredRecycleArgs {
      public:
        HLRTaskID hlr_id;
        DistributedID did;
      };
      struct DeferredFutureSetArgs {
        HLRTaskID hlr_id;
        Future::Impl *target;
        Future::Impl *result;
        TaskOp *task_op;
      };
      struct DeferredFutureMapSetArgs {
        HLRTaskID hlr_id;
        FutureMap::Impl *future_map;
        Future::Impl *result;
        Domain domain;
        TaskOp *task_op;
      };
      struct MPIRankArgs {
        HLRTaskID hlr_id;
        int mpi_rank;
        AddressSpace source_space;
      };
      struct CollectiveFutureArgs {
        HLRTaskID hlr_id;
        ReductionOpID redop;
        Future::Impl *future;
        Barrier barrier;
      };
      struct MapperTaskArgs {
        HLRTaskID hlr_id;
        Future::Impl *future;
        MapperID map_id;
        Processor proc;
        Event event;
        RemoteTask *context;
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
      Internal(Machine m, AddressSpaceID space_id,
               const std::set<Processor> &local_procs,
               const std::set<Processor> &local_util_procs,
               const std::set<AddressSpaceID> &address_spaces,
               const std::map<Processor,AddressSpaceID> &proc_spaces);
      Internal(const Internal &rhs);
      ~Internal(void);
    public:
      Internal& operator=(const Internal&rhs);
    public:
      void construct_mpi_rank_tables(Processor proc, int rank);
      void launch_top_level_task(Processor target);
      Event launch_mapper_task(Mapper *mapper, Processor proc, 
                               Processor::TaskFuncID tid,
                               const TaskArgument &arg, MapperID map_id);
    public:
      IndexSpace create_index_space(Context ctx, size_t max_num_elmts);
      IndexSpace create_index_space(Context ctx, Domain domain);
      IndexSpace create_index_space(Context ctx, 
                                    const std::set<Domain> &domains);
      void destroy_index_space(Context ctx, IndexSpace handle);
      // Called from deletion op
      bool finalize_index_space_destroy(IndexSpace handle);
    public:
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                            const Domain &color_space,
                                            const PointColoring &coloring,
                                            PartitionKind part_kind,
                                            int color, bool allocable);
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                            const Coloring &coloring,
                                            bool disjoint,
                                            int part_color);
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainPointColoring &coloring,
                                            PartitionKind part_kind, int color);
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                            Domain color_space,
                                            const DomainColoring &coloring,
                                            bool disjoint,
                                            int part_color);
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                            const Domain &color_space,
                                       const MultiDomainPointColoring &coloring,
                                            PartitionKind part_kind, int color);
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                            Domain color_space,
                                            const MultiDomainColoring &coloring,
                                            bool disjoint, 
                                            int part_color);
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
      Accessor::RegionAccessor<Accessor::AccessorType::Generic> field_accessor,
                                            int part_color);
      void destroy_index_partition(Context ctx, IndexPartition handle);
      // Called from deletion op
      void finalize_index_partition_destroy(IndexPartition handle);
    public:
      // Helper methods for partition construction
      void validate_unstructured_disjointness(IndexPartition pid,
                                  const std::map<DomainPoint,Domain> &domains);
      void validate_structured_disjointness(IndexPartition pid,
                                  const std::map<DomainPoint,Domain> &domains);
      void validate_multi_structured_disjointness(IndexPartition pid,
                       const std::map<DomainPoint,std::set<Domain> > &domains);
      Domain construct_convex_hull(const std::set<Domain> &domains);
    public:
      IndexPartition create_equal_partition(Context ctx, IndexSpace parent,
                                            const Domain &color_space, 
                                            size_t granuarlity,
                                            int color, bool allocable);
      IndexPartition create_weighted_partition(Context ctx, IndexSpace parent,
                                            const Domain &color_space,
                                      const std::map<DomainPoint,int> &weights,
                                            size_t granularity, int color,
                                            bool allocable);
      IndexPartition create_partition_by_union(Context ctx, IndexSpace parent,
                                               IndexPartition handle1,
                                               IndexPartition handle2,
                                               PartitionKind kind,
                                               int color, bool allocable);
      IndexPartition create_partition_by_intersection(Context ctx, 
                                               IndexSpace parent,
                                               IndexPartition handle1,
                                               IndexPartition handle2,
                                               PartitionKind kind,
                                               int color, bool allocable);
      IndexPartition create_partition_by_difference(Context ctx, 
                                               IndexSpace parent,
                                               IndexPartition handle1,
                                               IndexPartition handle2,
                                               PartitionKind kind,
                                               int color, bool allocable);
      void create_cross_product_partition(Context ctx, 
                                          IndexPartition handle1,
                                          IndexPartition handle2,
                              std::map<DomainPoint,IndexPartition> &handles,
                                          PartitionKind kind,
                                          int color, bool allocable);
      IndexPartition create_partition_by_field(Context ctx, 
                                               LogicalRegion handle,
                                               LogicalRegion parent,
                                               FieldID fid,
                                               const Domain &color_space,
                                               int color, bool allocable);
      IndexPartition create_partition_by_image(Context ctx,
                                               IndexSpace handle,
                                               LogicalPartition projection,
                                               LogicalRegion parent,
                                               FieldID fid, 
                                               const Domain &color_space,
                                               PartitionKind part_kind,
                                               int color, bool allocable);
      IndexPartition create_partition_by_preimage(Context ctx,
                                               IndexPartition projection,
                                               LogicalRegion handle,
                                               LogicalRegion parent,
                                               FieldID fid,
                                               const Domain &color_space,
                                               PartitionKind part_kind,
                                               int color, bool allocable);
      IndexPartition create_pending_partition(Context ctx, IndexSpace parent,
                                              const Domain &color_space,
                                              PartitionKind part_kind,
                                              int color, bool allocable);
      IndexSpace create_index_space_union(Context ctx, IndexPartition parent,
                                          const DomainPoint &color, 
                                        const std::vector<IndexSpace> &handles);
      IndexSpace create_index_space_union(Context ctx, IndexPartition parent,
                                          const DomainPoint &color,
                                          IndexPartition handle);
      IndexSpace create_index_space_intersection(Context ctx, 
                                                 IndexPartition parent,
                                                 const DomainPoint &color,
                                       const std::vector<IndexSpace> &handles);
      IndexSpace create_index_space_intersection(Context ctx,
                                                 IndexPartition parent,
                                                 const DomainPoint &color,
                                                 IndexPartition handle); 
      IndexSpace create_index_space_difference(Context ctx, 
                                               IndexPartition parent,
                                               const DomainPoint &color,
                                               IndexSpace initial,
                                       const std::vector<IndexSpace> &handles);
    public:
      IndexPartition get_index_partition(Context ctx, IndexSpace parent, 
                                         Color color);
      IndexPartition get_index_partition(IndexSpace parent, Color color);
      IndexPartition get_index_partition(Context ctx, IndexSpace parent,
                                         const DomainPoint &color);
      IndexPartition get_index_partition(IndexSpace parent,
                                         const DomainPoint &color);
      bool has_index_partition(Context ctx, IndexSpace parent,
                               const DomainPoint &color);
      bool has_index_partition(IndexSpace parent, 
                               const DomainPoint &color);
      IndexSpace get_index_subspace(Context ctx, IndexPartition p, 
                                    Color color); 
      IndexSpace get_index_subspace(IndexPartition p, Color c);
      IndexSpace get_index_subspace(Context ctx, IndexPartition p,
                                    const DomainPoint &color);
      IndexSpace get_index_subspace(IndexPartition p, const DomainPoint &c);
      bool has_index_subspace(Context ctx, IndexPartition p,
                              const DomainPoint &color);
      bool has_index_subspace(IndexPartition p, 
                              const DomainPoint &color);
      bool has_multiple_domains(Context ctx, IndexSpace handle);
      bool has_multiple_domains(IndexSpace handle);
      Domain get_index_space_domain(Context ctx, IndexSpace handle);
      Domain get_index_space_domain(IndexSpace handle);
      void get_index_space_domains(Context ctx, IndexSpace handle,
                                   std::vector<Domain> &domains);
      void get_index_space_domains(IndexSpace handle,
                                   std::vector<Domain> &domains);
      Domain get_index_partition_color_space(Context ctx, IndexPartition p);
      Domain get_index_partition_color_space(IndexPartition p);
      void get_index_space_partition_colors(Context ctx, IndexSpace handle,
                                            std::set<Color> &colors);
      void get_index_space_partition_colors(IndexSpace handle,
                                            std::set<Color> &colors);
      void get_index_space_partition_colors(Context ctx, IndexSpace handle,
                                            std::set<DomainPoint> &colors);
      void get_index_space_partition_colors(IndexSpace handle,
                                            std::set<DomainPoint> &colors);
      bool is_index_partition_disjoint(Context ctx, IndexPartition p);
      bool is_index_partition_disjoint(IndexPartition p);
      bool is_index_partition_complete(Context ctx, IndexPartition p);
      bool is_index_partition_complete(IndexPartition p);
      Color get_index_space_color(Context ctx, IndexSpace handle);
      Color get_index_space_color(IndexSpace handle);
      DomainPoint get_index_space_color_point(Context ctx, IndexSpace handle);
      DomainPoint get_index_space_color_point(IndexSpace handle);
      Color get_index_partition_color(Context ctx, IndexPartition handle);
      Color get_index_partition_color(IndexPartition handle);
      DomainPoint get_index_partition_color_point(Context ctx, 
                                                  IndexPartition handle);
      DomainPoint get_index_partition_color_point(IndexPartition handle);
      IndexSpace get_parent_index_space(Context ctx, IndexPartition handle);
      IndexSpace get_parent_index_space(IndexPartition handle);
      bool has_parent_index_partition(Context ctx, IndexSpace handle);
      bool has_parent_index_partition(IndexSpace handle);
      IndexPartition get_parent_index_partition(Context ctx, IndexSpace handle);
      IndexPartition get_parent_index_partition(IndexSpace handle);
    public:
      ptr_t safe_cast(Context ctx, ptr_t pointer, LogicalRegion region);
      DomainPoint safe_cast(Context ctx, DomainPoint point, 
                            LogicalRegion region);
    public:
      FieldSpace create_field_space(Context ctx);
      void destroy_field_space(Context ctx, FieldSpace handle);
      size_t get_field_size(Context ctx, FieldSpace handle, FieldID fid);
      size_t get_field_size(FieldSpace handle, FieldID fid);
      void get_field_space_fields(Context ctx, FieldSpace handle,
                                  std::set<FieldID> &fields);
      void get_field_space_fields(FieldSpace handle, std::set<FieldID> &fields);
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
      bool finalize_logical_region_destroy(LogicalRegion handle);
      void finalize_logical_partition_destroy(LogicalPartition handle);
    public:
      LogicalPartition get_logical_partition(Context ctx, LogicalRegion parent, 
                                             IndexPartition handle);
      LogicalPartition get_logical_partition(LogicalRegion parent,
                                             IndexPartition handle);
      LogicalPartition get_logical_partition_by_color(Context ctx, 
                                                      LogicalRegion parent, 
                                                      Color c);
      LogicalPartition get_logical_partition_by_color(Context ctx,
                                                      LogicalRegion parent,
                                                      const DomainPoint &c);
      LogicalPartition get_logical_partition_by_color(LogicalRegion parent,
                                                      Color c);
      LogicalPartition get_logical_partition_by_color(LogicalRegion parent,
                                                      const DomainPoint &c);
      bool has_logical_partition_by_color(Context ctx, LogicalRegion parent,
                                          const DomainPoint &color);
      bool has_logical_partition_by_color(LogicalRegion parent,
                                          const DomainPoint &color);
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
                                                   Color c);
      LogicalRegion get_logical_subregion_by_color(Context ctx,
                                                   LogicalPartition parent,
                                                   const DomainPoint &c);
      LogicalRegion get_logical_subregion_by_color(LogicalPartition parent,
                                                   Color c);
      LogicalRegion get_logical_subregion_by_color(LogicalPartition parent,
                                                   const DomainPoint &c);
      bool has_logical_subregion_by_color(Context ctx, LogicalPartition parent,
                                          const DomainPoint &color);
      bool has_logical_subregion_by_color(LogicalPartition parent,
                                          const DomainPoint &color);
      LogicalRegion get_logical_subregion_by_tree(Context ctx, 
                                                  IndexSpace handle, 
                                                  FieldSpace fspace, 
                                                  RegionTreeID tid);
      LogicalRegion get_logical_subregion_by_tree(IndexSpace handle,
                                                  FieldSpace fspace,
                                                  RegionTreeID tid);
      Color get_logical_region_color(Context ctx, LogicalRegion handle);
      Color get_logical_region_color(LogicalRegion handle);
      DomainPoint get_logical_region_color_point(Context ctx, 
                                                 LogicalRegion handle);
      DomainPoint get_logical_region_color_point(LogicalRegion handle);
      Color get_logical_partition_color(Context ctx, LogicalPartition handle);
      Color get_logical_partition_color(LogicalPartition handle);
      DomainPoint get_logical_partition_color_point(Context ctx, 
                                                    LogicalPartition handle);
      DomainPoint get_logical_partition_color_point(LogicalPartition handle);
      LogicalRegion get_parent_logical_region(Context ctx, 
                                              LogicalPartition handle);
      LogicalRegion get_parent_logical_region(LogicalPartition handle);
      bool has_parent_logical_partition(Context ctx, LogicalRegion handle);
      bool has_parent_logical_partition(LogicalRegion handle);
      LogicalPartition get_parent_logical_partition(Context ctx, 
                                                    LogicalRegion handle);
      LogicalPartition get_parent_logical_partition(LogicalRegion handle);
    public:
      IndexAllocator create_index_allocator(Context ctx, IndexSpace handle);
      FieldAllocator create_field_allocator(Context ctx, FieldSpace handle);
      ArgumentMap create_argument_map(Context ctx);
    public:
      Future execute_task(Context ctx, const TaskLauncher &launcher);
      FutureMap execute_index_space(Context ctx, const IndexLauncher &launcher);
      Future execute_index_space(Context ctx, const IndexLauncher &launcher,
                                 ReductionOpID redop);
      Future execute_task(Context ctx, 
                          Processor::TaskFuncID task_id,
                          const std::vector<IndexSpaceRequirement> &indexes,
                          const std::vector<FieldSpaceRequirement> &fields,
                          const std::vector<RegionRequirement> &regions,
                          const TaskArgument &arg, 
                          const Predicate &predicate = Predicate::TRUE_PRED,
                          MapperID id = 0, 
                          MappingTagID tag = 0);
      FutureMap execute_index_space(Context ctx, 
                          Processor::TaskFuncID task_id,
                          const Domain domain,
                          const std::vector<IndexSpaceRequirement> &indexes,
                          const std::vector<FieldSpaceRequirement> &fields,
                          const std::vector<RegionRequirement> &regions,
                          const TaskArgument &global_arg, 
                          const ArgumentMap &arg_map,
                          const Predicate &predicate = Predicate::TRUE_PRED,
                          bool must_paralleism = false, 
                          MapperID id = 0, 
                          MappingTagID tag = 0);
      Future execute_index_space(Context ctx, 
                          Processor::TaskFuncID task_id,
                          const Domain domain,
                          const std::vector<IndexSpaceRequirement> &indexes,
                          const std::vector<FieldSpaceRequirement> &fields,
                          const std::vector<RegionRequirement> &regions,
                          const TaskArgument &global_arg, 
                          const ArgumentMap &arg_map,
                          ReductionOpID reduction, 
                          const TaskArgument &initial_value,
                          const Predicate &predicate = Predicate::TRUE_PRED,
                          bool must_parallelism = false, 
                          MapperID id = 0, 
                          MappingTagID tag = 0);
    public:
      PhysicalRegion map_region(Context ctx, const InlineLauncher &launcher);
      PhysicalRegion map_region(Context ctx, const RegionRequirement &req, 
                                MapperID id = 0, MappingTagID tag = 0);
      PhysicalRegion map_region(Context ctx, unsigned idx, 
                                MapperID id = 0, MappingTagID tag = 0);
      void remap_region(Context ctx, PhysicalRegion region);
      void unmap_region(Context ctx, PhysicalRegion region);
      void unmap_all_regions(Context ctx);
    public:
      void fill_field(Context ctx, LogicalRegion handle,
                      LogicalRegion parent, FieldID fid,
                      const void *value, size_t value_size,
                      const Predicate &pred);
      void fill_field(Context ctx, LogicalRegion handle,
                      LogicalRegion parent, FieldID fid,
                      Future f, const Predicate &pred);
      void fill_fields(Context ctx, LogicalRegion handle,
                       LogicalRegion parent,
                       const std::set<FieldID> &fields,
                       const void *value, size_t value_size,
                       const Predicate &pred);
      void fill_fields(Context ctx, LogicalRegion handle,
                       LogicalRegion parent,
                       const std::set<FieldID> &fields,
                       Future f, const Predicate &pred);
      void fill_fields(Context ctx, const FillLauncher &launcher);
    public:
      PhysicalRegion attach_hdf5(Context ctx, const char *file_name,
                                 LogicalRegion handle, LogicalRegion parent,
                                 const std::map<FieldID,const char*> field_map,
                                 LegionFileMode);
      void detach_hdf5(Context ctx, PhysicalRegion region);
      PhysicalRegion attach_file(Context ctx, const char *file_name,
                                 LogicalRegion handle, LogicalRegion parent,
                                 const std::vector<FieldID> field_vec,
                                 LegionFileMode);
      void detach_file(Context ctx, PhysicalRegion region);
    public:
      void issue_copy_operation(Context ctx, const CopyLauncher &launcher);
    public:
      Predicate create_predicate(Context ctx, const Future &f);
      Predicate predicate_not(Context ctx, const Predicate &p); 
      Predicate predicate_and(Context ctx, const Predicate &p1, 
                                           const Predicate &p2); 
      Predicate predicate_or(Context ctx, const Predicate &p1, 
                                          const Predicate &p2);  
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
                                            Future f, unsigned count);
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
      void complete_frame(Context ctx);
      FutureMap execute_must_epoch(Context ctx, 
                                   const MustEpochLauncher &launcher);
      int get_tunable_value(Context ctx, TunableID tid, 
                            MapperID mid, MappingTagID tag);
    public:
      Future get_current_time(Context ctx, const Future &precondition);
      Future get_current_time_in_microseconds(Context ctx, const Future &pre);
      Future get_current_time_in_nanoseconds(Context ctx, const Future &pre);
    public:
      Mapper* get_mapper(Context ctx, MapperID id, Processor target);
      Processor get_executing_processor(Context ctx);
      void raise_region_exception(Context ctx, PhysicalRegion region, 
                                  bool nuclear);
    public:
      const std::map<int,AddressSpace>& find_forward_MPI_mapping(void);
      const std::map<AddressSpace,int>& find_reverse_MPI_mapping(void);
    public:
      MapperID generate_dynamic_mapper_id(void);
      static MapperID& get_current_static_mapper_id(void);
      static MapperID generate_static_mapper_id(void);
      void add_mapper(MapperID map_id, Mapper *mapper, Processor proc);
      void replace_default_mapper(Mapper *mapper, Processor proc);
    public:
      void register_projection_functor(ProjectionID pid,
                                       ProjectionFunctor *func);
      ProjectionFunctor* find_projection_functor(ProjectionID pid);
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
      const std::vector<PhysicalRegion>& begin_task(TaskOp *task);
      void end_task(TaskOp *task, const void *result, size_t result_size,
                    bool owned);
      TaskID generate_dynamic_task_id(void);
      VariantID register_variant(const TaskVariantRegistrar &registrar,
                                 const void *user_data, size_t user_data_size,
                                 CodeDescriptor *realm,
                                 bool ret, VariantID vid = AUTO_GENERATE_ID,
                                 bool check_task_id = true);
      TaskImpl* find_or_create_task_impl(TaskID task_id);
      TaskImpl* find_task_impl(TaskID task_id);
      VariantImpl* find_variant_impl(TaskID task_id, VariantID variant_id);
      // For backwards compatiblity only
      TaskVariantCollection* get_collection(TaskID task_id);
    public:
      // Memory manager functions
      MemoryManager* find_memory(Memory mem);
      void allocate_physical_instance(PhysicalManager *instance);
      void free_physical_instance(PhysicalManager *instance);
      AddressSpaceID find_address_space(Memory handle) const;
    public:
      // Mapper introspection methods
      size_t sample_allocated_space(Memory mem);
      size_t sample_free_space(Memory mem);
      unsigned sample_allocated_instances(Memory mem);
      unsigned sample_unmapped_tasks(Processor proc, Mapper *mapper);
    public:
      // Messaging functions
      MessageManager* find_messenger(AddressSpaceID sid);
      MessageManager* find_messenger(Processor target);
      AddressSpaceID find_address_space(Processor target) const;
      void send_task(Processor target, TaskOp *task);
      void send_tasks(Processor target, const std::set<TaskOp*> &tasks);
      void send_steal_request(const std::multimap<Processor,MapperID> &targets,
                              Processor thief);
      void send_advertisements(const std::set<Processor> &targets,
                              MapperID map_id, Processor source);
      void send_index_space_node(AddressSpaceID target, Serializer &rez);
      void send_index_space_request(AddressSpaceID target, Serializer &rez);
      void send_index_space_return(AddressSpaceID target, Serializer &rez);
      void send_index_space_child_request(AddressSpaceID target, 
                                          Serializer &rez);
      void send_index_partition_node(AddressSpaceID target, Serializer &rez);
      void send_index_partition_request(AddressSpaceID target, Serializer &rez);
      void send_index_partition_return(AddressSpaceID target, Serializer &rez);
      void send_index_partition_child_request(AddressSpaceID target,
                                              Serializer &rez);
      void send_field_space_node(AddressSpaceID target, Serializer &rez);
      void send_field_space_request(AddressSpaceID target, Serializer &rez);
      void send_field_space_return(AddressSpaceID target, Serializer &rez);
      void send_field_alloc_request(AddressSpaceID target, Serializer &rez);
      void send_field_alloc_notification(AddressSpaceID target,Serializer &rez);
      void send_field_space_top_alloc(AddressSpaceID target, Serializer &rez);
      void send_field_free(AddressSpaceID target, Serializer &rez);
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
      void send_did_add_create_reference(AddressSpaceID target,Serializer &rez);
      void send_did_remove_create_reference(AddressSpaceID target,
                                            Serializer &rez, bool flush = true);
      void send_view_remote_registration(AddressSpaceID target,Serializer &rez);
      void send_back_atomic(AddressSpaceID target, Serializer &rez);
      void send_materialized_view(AddressSpaceID target, Serializer &rez);
      void send_materialized_update(AddressSpaceID target, Serializer &rez);
      void send_composite_view(AddressSpaceID target, Serializer &rez);
      void send_fill_view(AddressSpaceID target, Serializer &rez);
      void send_deferred_update(AddressSpaceID target, Serializer &rez);
      void send_reduction_view(AddressSpaceID target, Serializer &rez);
      void send_reduction_update(AddressSpaceID target, Serializer &rez);
      void send_instance_manager(AddressSpaceID target, Serializer &rez);
      void send_reduction_manager(AddressSpaceID target, Serializer &rez);
      void send_future(AddressSpaceID target, Serializer &rez);
      void send_future_result(AddressSpaceID target, Serializer &rez);
      void send_future_subscription(AddressSpaceID target, Serializer &rez);
      void send_make_persistent(AddressSpaceID target, Serializer &rez);
      void send_unmake_persistent(AddressSpaceID target, Serializer &rez);
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
      void send_subscribe_remote_context(AddressSpaceID target,Serializer &rez);
      void send_free_remote_context(AddressSpaceID target, Serializer &rez);
      void send_version_state_path_only(AddressSpaceID target, Serializer &rez);
      void send_version_state_initialization(AddressSpaceID target, 
                                             Serializer &rez);
      void send_version_state_request(AddressSpaceID target, Serializer &rez);
      void send_version_state_response(AddressSpaceID target, Serializer &rez);
      void send_remote_instance_creation_request(AddressSpaceID target,
                                                 Serializer &rez);
      void send_remote_reduction_creation_request(AddressSpaceID target,
                                                  Serializer &rez);
      void send_remote_creation_response(AddressSpaceID target,Serializer &rez);
      void send_back_logical_state(AddressSpaceID target, Serializer &rez);
      void send_variant_request(AddressSpaceID target, Serializer &rez);
      void send_variant_response(AddressSpaceID target, Serializer &rez);
      void send_constraint_request(AddressSpaceID target, Serializer &rez);
      void send_constraint_response(AddressSpaceID target, Serializer &rez);
      void send_constraint_release(AddressSpaceID target, Serializer &rez);
    public:
      // Complementary tasks for handling messages
      void handle_task(Deserializer &derez);
      void handle_steal(Deserializer &derez);
      void handle_advertisement(Deserializer &derez);
      void handle_index_space_node(Deserializer &derez, AddressSpaceID source);
      void handle_index_space_request(Deserializer &derez, 
                                      AddressSpaceID source);
      void handle_index_space_return(Deserializer &derez); 
      void handle_index_space_child_request(Deserializer &derez); 
      void handle_index_partition_node(Deserializer &derez,
                                       AddressSpaceID source);
      void handle_index_partition_request(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_index_partition_return(Deserializer &derez);
      void handle_index_partition_child_request(Deserializer &derez);
      void handle_field_space_node(Deserializer &derez, AddressSpaceID source);
      void handle_field_space_request(Deserializer &derez,
                                      AddressSpaceID source);
      void handle_field_space_return(Deserializer &derez);
      void handle_field_alloc_request(Deserializer &derez);
      void handle_field_alloc_notification(Deserializer &derez);
      void handle_field_space_top_alloc(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_field_free(Deserializer &derez, AddressSpaceID source);
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
      void handle_did_create_add(Deserializer &derez);
      void handle_did_create_remove(Deserializer &derez);
      void handle_view_remote_registration(Deserializer &derez, 
                                           AddressSpaceID source);
      void handle_send_back_atomic(Deserializer &derez, AddressSpaceID source);
      void handle_send_materialized_view(Deserializer &derez, 
                                         AddressSpaceID source);
      void handle_send_materialized_update(Deserializer &derez,
                                           AddressSpaceID source);
      void handle_send_composite_view(Deserializer &derez,
                                      AddressSpaceID source);
      void handle_send_fill_view(Deserializer &derez, AddressSpaceID source);
      void handle_send_deferred_update(Deserializer &derez, 
                                       AddressSpaceID source);
      void handle_send_reduction_view(Deserializer &derez,
                                      AddressSpaceID source);
      void handle_send_reduction_update(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_send_instance_manager(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_send_reduction_manager(Deserializer &derez,
                                         AddressSpaceID source);
      void handle_future_send(Deserializer &derez, AddressSpaceID source);
      void handle_future_result(Deserializer &derez);
      void handle_future_subscription(Deserializer &derez);
      void handle_make_persistent(Deserializer &derez, AddressSpaceID source);
      void handle_unmake_persistent(Deserializer &derez, AddressSpaceID source);
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
      void handle_subscribe_remote_context(Deserializer &derez,
                                           AddressSpaceID source);
      void handle_free_remote_context(Deserializer &derez);
      void handle_version_state_path_only(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_version_state_initialization(Deserializer &derez,
                                               AddressSpaceID source);
      void handle_version_state_request(Deserializer &derez);
      void handle_version_state_response(Deserializer &derez,
                                         AddressSpaceID source);
      void handle_remote_instance_creation(Deserializer &derez, 
                                           AddressSpaceID source);
      void handle_remote_reduction_creation(Deserializer &derez,
                                            AddressSpaceID source);
      void handle_remote_creation_response(Deserializer &derez);
      void handle_logical_state_return(Deserializer &derez,
                                       AddressSpaceID source);
      void handle_variant_request(Deserializer &derez, AddressSpaceID source);
      void handle_variant_response(Deserializer &derez);
      void handle_constraint_request(Deserializer &derez,AddressSpaceID source);
      void handle_constraint_response(Deserializer &derez);
      void handle_constraint_release(Deserializer &derez,AddressSpaceID source);
      void handle_top_level_task_request(Deserializer &derez);
      void handle_top_level_task_complete(Deserializer &derez);
      void handle_shutdown_notification(AddressSpaceID source);
      void handle_shutdown_response(Deserializer &derez, AddressSpaceID source);
    public:
      // Helper methods for the RegionTreeForest
      inline unsigned get_context_count(void) { return total_contexts; }
      inline unsigned get_start_color(void) const { return address_space; }
      inline unsigned get_color_modulus(void) const { return runtime_stride; }
#ifdef HANG_TRACE
    public:
      void dump_processor_states(FILE *target);
#endif
    public:
      // Manage the execution of tasks within a context
      void activate_context(SingleTask *context);
      void deactivate_context(SingleTask *context);
    public:
      void execute_task_launch(Context ctx, TaskOp *task_op);
      void add_to_dependence_queue(Processor p, Operation *op);
      void add_to_ready_queue(Processor p, TaskOp *task_op, bool prev_fail);
      void add_to_local_queue(Processor p, Operation *op, bool prev_fail);
    public:
      // These methods must be called before and after
      // pre-empting a task for any reason to update
      // the necessary processor manager to indicate
      // that there is one fewer task running on the processor
      void pre_wait(Processor proc);
      void post_wait(Processor proc);
    public:
      // Invoke the mapper for a given processor
      bool invoke_mapper_pre_map_task(Processor target, TaskOp *task);
      void invoke_mapper_select_variant(Processor target, TaskOp *task);
      bool invoke_mapper_map_task(Processor target, SingleTask *task);
      void invoke_mapper_post_map_task(Processor target, TaskOp *task);
      void invoke_mapper_failed_mapping(Processor target, Mappable *mappable);
      void invoke_mapper_notify_result(Processor target, Mappable *mappable);
      void invoke_mapper_slice_domain(Processor target, MultiTask *task,
                                      std::vector<Mapper::DomainSplit> &splits);
      bool invoke_mapper_map_inline(Processor target, Inline *op);
      bool invoke_mapper_map_copy(Processor target, Copy *op);
      bool invoke_mapper_speculate(Processor target, 
                                   Mappable *mappable, bool &value);
      void invoke_mapper_configure_context(Processor target, TaskOp *task);
      bool invoke_mapper_rank_copy_targets(Processor target, Mappable *mappable,
                                           LogicalRegion handle, 
                                           const std::set<Memory> &memories,
                                           bool complete,
                                           size_t max_blocking_factor,
                                           std::set<Memory> &to_reuse,
                                           std::vector<Memory> &to_create,
                                           bool &create_one,
                                           size_t &blocking_factor);
      void invoke_mapper_rank_copy_sources(Processor target, Mappable *mappable,
                                           const std::set<Memory> &memories,
                                           Memory destination,
                                           std::vector<Memory> &chosen_order);
      void invoke_mapper_notify_profiling(Processor target, TaskOp *task);
      bool invoke_mapper_map_must_epoch(Processor target, 
                                        const std::vector<Task*> &tasks,
                      const std::vector<Mapper::MappingConstraint> &constraints,
                                        MapperID map_id, MappingTagID tag);
      void invoke_mapper_handle_message(Processor target, MapperID map_id,
                          Processor source, const void *message, size_t length);
      void invoke_mapper_broadcast(MapperID map_id, Processor source,
                                   const void *message, size_t length,
                                   int radix, int index);
      void invoke_mapper_task_result(MapperID map_id, Processor source,
                                     Event event, const void *result,
                                     size_t result_size);
    public:
      // Handle directions and query requests from the mapper
      Processor locate_mapper_info(Mapper *mapper, MapperID &map_id);
      void handle_mapper_send_message(Mapper *mapper, Processor target, 
                                      const void *message, size_t length);
      void handle_mapper_broadcast(Mapper *mapper, const void *message,
                                   size_t length, int radix);
      Event launch_mapper_task(Mapper *mapper, Processor::TaskFuncID tid,
                               const TaskArgument &arg);
      void defer_mapper_call(Mapper *mapper, Event wait_on);
    public:
      inline Processor find_utility_group(void) { return utility_group; }
      Processor find_processor_group(const std::set<Processor> &procs);
      Event issue_runtime_meta_task(const void *args, size_t arglen,
                                    HLRTaskID tid, Operation *op = NULL,
                                    Event precondition = Event::NO_EVENT, 
                                    int priority = 0, 
                                    bool holds_reservation = false,
                                    Processor proc = Processor::NO_PROC);
    public:
      void allocate_context(SingleTask *task);
      void free_context(SingleTask *task);
    public:
      DistributedID get_available_distributed_id(bool need_cont, 
                                                 bool has_lock = false);
      void free_distributed_id(DistributedID did);
      void recycle_distributed_id(DistributedID did, Event recycle_event);
    public:
      void register_distributed_collectable(DistributedID did,
                                            DistributedCollectable *dc,
                                            bool needs_lock = true);
      void unregister_distributed_collectable(DistributedID did);
      bool has_distributed_collectable(DistributedID did);
      DistributedCollectable* find_distributed_collectable(DistributedID did);
      DistributedCollectable* weak_find_distributed_collectable(
                                                           DistributedID did);
    public:
      void register_future(DistributedID did, Future::Impl *impl);
      void unregister_future(DistributedID did);
      bool has_future(DistributedID did);
      Future::Impl* find_future(DistributedID did);
      Future::Impl* find_or_create_future(DistributedID did,
                                          AddressSpaceID owner_space);
    public:
      void defer_collect_user(LogicalView *view, Event term_event);
      void complete_gc_epoch(GarbageCollectionEpoch *epoch);
    public:
      void increment_outstanding_top_level_tasks(void);
      void decrement_outstanding_top_level_tasks(void);
    public:
      void issue_runtime_shutdown_attempt(void);
      void attempt_runtime_shutdown(void);
      void initiate_runtime_shutdown(AddressSpaceID source);
      void finalize_runtime_shutdown(void);
    public:
      bool has_outstanding_tasks(void);
      inline void increment_total_outstanding_tasks(void)
        { __sync_fetch_and_add(&total_outstanding_tasks,1); }
      inline void decrement_total_outstanding_tasks(void)
        { __sync_fetch_and_sub(&total_outstanding_tasks,1); }
    public:
      template<typename T>
      inline T* get_available(Reservation reservation,
                              std::deque<T*> &queue, bool has_lock);
    public:
      IndividualTask*       get_available_individual_task(bool need_cont,
                                                  bool has_lock = false);
      PointTask*            get_available_point_task(bool need_cont,
                                                  bool has_lock = false);
      IndexTask*            get_available_index_task(bool need_cont,
                                                  bool has_lock = false);
      SliceTask*            get_available_slice_task(bool need_cont,
                                                  bool has_lock = false);
      RemoteTask*           get_available_remote_task(bool need_cont,
                                                  bool has_lock = false);
      InlineTask*           get_available_inline_task(bool need_cont,
                                                  bool has_lock = false);
      MapOp*                get_available_map_op(bool need_cont,
                                                  bool has_lock = false);
      CopyOp*               get_available_copy_op(bool need_cont,
                                                  bool has_lock = false);
      FenceOp*              get_available_fence_op(bool need_cont,
                                                  bool has_lock = false);
      FrameOp*              get_available_frame_op(bool need_cont,
                                                  bool has_lock = false);
      DeletionOp*           get_available_deletion_op(bool need_cont,
                                                  bool has_lock = false);
      InterCloseOp*         get_available_inter_close_op(bool need_cont,
                                                  bool has_lock = false);
      ReadCloseOp*          get_available_read_close_op(bool need_cont,
                                                  bool has_lock = false);
      PostCloseOp*          get_available_post_close_op(bool need_cont,
                                                  bool has_lock = false);
      VirtualCloseOp*       get_available_virtual_close_op(bool need_cont,
                                                  bool has_lock = false);
      DynamicCollectiveOp*  get_available_dynamic_collective_op(bool need_cont,
                                                  bool has_lock = false);
      FuturePredOp*         get_available_future_pred_op(bool need_cont,
                                                  bool has_lock = false);
      NotPredOp*            get_available_not_pred_op(bool need_cont,
                                                  bool has_lock = false);
      AndPredOp*            get_available_and_pred_op(bool need_cont,
                                                  bool has_lock = false);
      OrPredOp*             get_available_or_pred_op(bool need_cont,
                                                  bool has_lock = false);
      AcquireOp*            get_available_acquire_op(bool need_cont,
                                                  bool has_lock = false);
      ReleaseOp*            get_available_release_op(bool need_cont,
                                                  bool has_lock = false);
      TraceCaptureOp*       get_available_capture_op(bool need_cont,
                                                  bool has_lock = false);
      TraceCompleteOp*      get_available_trace_op(bool need_cont,
                                                  bool has_lock = false);
      MustEpochOp*          get_available_epoch_op(bool need_cont,
                                                  bool has_lock = false);
      PendingPartitionOp*   get_available_pending_partition_op(bool need_cont,
                                                  bool has_lock = false);
      DependentPartitionOp* get_available_dependent_partition_op(bool need_cont,
                                                  bool has_lock = false);
      FillOp*               get_available_fill_op(bool need_cont,
                                                  bool has_lock = false);
      AttachOp*             get_available_attach_op(bool need_cont,
                                                  bool has_lock = false);
      DetachOp*             get_available_detach_op(bool need_cont,
                                                  bool has_lock = false);
      TimingOp*             get_available_timing_op(bool need_cont,
                                                  bool has_lock = false);
    public:
      void free_individual_task(IndividualTask *task);
      void free_point_task(PointTask *task);
      void free_index_task(IndexTask *task);
      void free_slice_task(SliceTask *task);
      void free_remote_task(RemoteTask *task);
      void free_inline_task(InlineTask *task);
      void free_map_op(MapOp *op);
      void free_copy_op(CopyOp *op);
      void free_fence_op(FenceOp *op);
      void free_frame_op(FrameOp *op);
      void free_deletion_op(DeletionOp *op);
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
      void free_fill_op(FillOp *op);
      void free_attach_op(AttachOp *op);
      void free_detach_op(DetachOp *op);
      void free_timing_op(TimingOp *op);
    public:
      RemoteTask* find_or_init_remote_context(UniqueID uid, Processor orig,
                                              SingleTask *remote_parent_ctx); 
      SingleTask* find_remote_context(UniqueID uid, SingleTask *remote_ctx);
      void release_remote_context(UniqueID remote_owner_uid);
    public:
      bool is_local(Processor proc) const;
      Processor find_utility_processor(Processor proc);
    public:
      IndexSpaceID       get_unique_index_space_id(void);
      IndexPartitionID   get_unique_index_partition_id(void);
      FieldSpaceID       get_unique_field_space_id(void);
      IndexTreeID        get_unique_index_tree_id(void);
      RegionTreeID       get_unique_region_tree_id(void);
      UniqueID           get_unique_operation_id(void);
      FieldID            get_unique_field_id(void);
      VariantID          get_unique_variant_id(void);
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
      static void high_level_runtime_task(
                          const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
      static void profiling_runtime_task(
                          const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
      static void profiling_mapper_task(
                          const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
      static void launch_top_level(
                          const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
    protected:
      // Internal runtime methods invoked by the above static methods
      // after the find the right runtime instance to call
      void process_schedule_request(Processor p);
      void process_profiling_task(Processor p, const void *args, size_t arglen);
      void process_message_task(const void *args, size_t arglen);
    public:
      // The Runtime wrapper for this class
      Runtime *const high_level;
      // The machine object for this runtime
      const Machine machine;
      const AddressSpaceID address_space; 
      const unsigned runtime_stride; // stride for uniqueness
      LegionProfiler *profiler;
      RegionTreeForest *const forest;
      Processor utility_group;
      const bool has_explicit_utility_procs;
    protected:
      unsigned total_outstanding_tasks;
      unsigned outstanding_top_level_tasks;
      ShutdownManager *shutdown_manager;
      Reservation shutdown_lock;
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
      // Reservation for looking up memory managers
      Reservation memory_manager_lock;
      // Reservation for initializing message managers
      Reservation message_manager_lock;
      // Memory managers for all the memories we know about
      std::map<Memory,MemoryManager*> memory_managers;
      // Message managers for each of the other runtimes
      MessageManager *message_managers[MAX_NUM_NODES];
      // For every processor map it to its address space
      const std::map<Processor,AddressSpaceID> proc_spaces;
    protected:
      // The task table 
      Reservation task_variant_lock;
      std::map<TaskID,TaskImpl*> task_table;
      std::deque<VariantImpl*> variant_table;
    protected:
      // Constraint sets
      Reservation layout_constraints_lock;
      std::map<LayoutConstraintID,LayoutConstraints*> layout_constraints_table;
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
      Reservation mapper_info_lock;
      // For every mapper remember its mapper ID and processor
      std::map<Mapper*,MapperInfo> mapper_infos;
#ifdef DEBUG_HIGH_LEVEL
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
      unsigned unique_variant_id;
      unsigned unique_constraint_id;
      unsigned unique_task_id;
      unsigned unique_mapper_id;
    protected:
      std::map<ProjectionID,ProjectionFunctor*> projection_functors;
    protected:
      // For MPI Inter-operability
      std::map<int,AddressSpace> forward_mpi_mapping;
      std::map<AddressSpace,int> reverse_mpi_mapping;
    protected:
      Reservation available_lock;
      unsigned total_contexts;
      std::deque<RegionTreeContext> available_contexts;
    protected:
      Reservation group_lock;
      LegionMap<uint64_t,LegionDeque<ProcessorGroupInfo>::aligned,
                PROCESSOR_GROUP_ALLOC>::tracked processor_groups;
    protected:
      Reservation distributed_id_lock;
      DistributedID unique_distributed_id;
      LegionDeque<DistributedID,
          RUNTIME_DISTRIBUTED_ALLOC>::tracked available_distributed_ids;
    protected:
      Reservation distributed_collectable_lock;
      LegionMap<DistributedID,DistributedCollectable*,
                RUNTIME_DIST_COLLECT_ALLOC>::tracked dist_collectables;
    protected:
      Reservation gc_epoch_lock;
      GarbageCollectionEpoch *current_gc_epoch;
      LegionSet<GarbageCollectionEpoch*,
                RUNTIME_GC_EPOCH_ALLOC>::tracked  pending_gc_epochs;
      unsigned gc_epoch_counter;
    protected:
      // Keep track of futures
      Reservation future_lock;
      LegionMap<DistributedID,Future::Impl*,
                RUNTIME_FUTURE_ALLOC>::tracked local_futures;
    protected:
      // The runtime keeps track of remote contexts so they
      // can be re-used by multiple tasks that get sent remotely
      Reservation remote_lock;
      LegionMap<UniqueID,RemoteTask*,
                RUNTIME_REMOTE_ALLOC>::tracked remote_contexts;
    protected:
      // For generating random numbers
      Reservation random_lock;
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
      Reservation allocation_lock;
      std::map<AllocationType,AllocationTracker> allocation_manager;
      unsigned long long allocation_tracing_count;
#endif
    protected:
      Reservation individual_task_lock;
      Reservation point_task_lock;
      Reservation index_task_lock;
      Reservation slice_task_lock;
      Reservation remote_task_lock;
      Reservation inline_task_lock;
      Reservation map_op_lock;
      Reservation copy_op_lock;
      Reservation fence_op_lock;
      Reservation frame_op_lock;
      Reservation deletion_op_lock;
      Reservation inter_close_op_lock;
      Reservation read_close_op_lock;
      Reservation post_close_op_lock;
      Reservation virtual_close_op_lock;
      Reservation dynamic_collective_op_lock;
      Reservation future_pred_op_lock;
      Reservation not_pred_op_lock;
      Reservation and_pred_op_lock;
      Reservation or_pred_op_lock;
      Reservation acquire_op_lock;
      Reservation release_op_lock;
      Reservation capture_op_lock;
      Reservation trace_op_lock;
      Reservation epoch_op_lock;
      Reservation pending_partition_op_lock;
      Reservation dependent_partition_op_lock;
      Reservation fill_op_lock;
      Reservation attach_op_lock;
      Reservation detach_op_lock;
      Reservation timing_op_lock;
    protected:
      std::deque<IndividualTask*>       available_individual_tasks;
      std::deque<PointTask*>            available_point_tasks;
      std::deque<IndexTask*>            available_index_tasks;
      std::deque<SliceTask*>            available_slice_tasks;
      std::deque<RemoteTask*>           available_remote_tasks;
      std::deque<InlineTask*>           available_inline_tasks;
      std::deque<MapOp*>                available_map_ops;
      std::deque<CopyOp*>               available_copy_ops;
      std::deque<FenceOp*>              available_fence_ops;
      std::deque<FrameOp*>              available_frame_ops;
      std::deque<DeletionOp*>           available_deletion_ops;
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
      std::deque<FillOp*>               available_fill_ops;
      std::deque<AttachOp*>             available_attach_ops;
      std::deque<DetachOp*>             available_detach_ops;
      std::deque<TimingOp*>             available_timing_ops;
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
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
      void release_layout(LayoutConstraintID layout_id, AddressSpaceID source);
      static LayoutConstraintID preregister_layout(
                                     const LayoutConstraintRegistrar &registrar,
                                     LayoutConstraintID layout_id);
      FieldSpace get_layout_constraint_field_space(LayoutConstraintID id);
      void get_layout_constraints(LayoutConstraintID layout_id,
                                  LayoutConstraintSet &layout_constraints);
      const char* get_layout_constraints_name(LayoutConstraintID layout_id);
      LayoutConstraints* find_layout_constraints(LayoutConstraintID layout_id);
    public:
      // Static methods for start-up and callback phases
      static int start(int argc, char **argv, bool background);
      static void wait_for_shutdown(void);
      static void set_top_level_task_id(Processor::TaskFuncID top_id);
      static void configure_MPI_interoperability(int rank);
      static const ReductionOp* get_reduction_op(ReductionOpID redop_id);
      static const SerdezOp* get_serdez_op(CustomSerdezID serdez_id);
      static const SerdezRedopFns* get_serdez_redop_fns(ReductionOpID redop_id);
      static void set_registration_callback(RegistrationCallbackFnptr callback);
      static InputArgs& get_input_args(void);
      static Internal* get_runtime(Processor p);
      static ReductionOpTable& get_reduction_table(void);
      static SerdezOpTable& get_serdez_table(void);
      static SerdezRedopTable& get_serdez_redop_table(void);
      static ProjectionID register_region_projection_function(
                                    ProjectionID handle, void *func_ptr);
      static ProjectionID register_partition_projection_function(
                                    ProjectionID handle, void *func_ptr);
      static std::deque<PendingVariantRegistration*>&
                                get_pending_variant_table(void);
      static std::map<LayoutConstraintID,LayoutConstraintRegistrar>&
                                get_pending_constraint_table(void);
      static TaskID& get_current_static_task_id(void);
      static TaskID generate_static_task_id(void);
      static VariantID preregister_variant(
                      const TaskVariantRegistrar &registrar,
                      const void *user_data, size_t user_data_size,
                      CodeDescriptor *realm_desc,
                      bool has_ret, const char *task_name,bool check_id = true);
      static PartitionProjectionFnptr 
                    find_partition_projection_function(ProjectionID pid);
      static RegionProjectionFnptr
                    find_region_projection_function(ProjectionID pid);
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
      static RegionProjectionTable& get_region_projection_table(void);
      static PartitionProjectionTable& get_partition_projection_table(void);
      static Event register_runtime_tasks(RealmRuntime &realm);
      static Processor::TaskFuncID get_next_available_id(void);
      static void log_machine(Machine machine);
    public:
      // Static member variables
      static Internal *runtime_map[(MAX_NUM_PROCS+1/*+1 for NO_PROC*/)];
      static volatile RegistrationCallbackFnptr registration_callback;
      static Processor::TaskFuncID legion_main_id;
      static int initial_task_window_size;
      static unsigned initial_task_window_hysteresis;
      static unsigned initial_tasks_to_schedule;
      static unsigned superscalar_width;
      static unsigned max_message_size;
      static unsigned gc_epoch_size;
      static bool runtime_started;
      static bool runtime_backgrounded;
      static bool separate_runtime_instances;
      static bool record_registration;
      static bool stealing_disabled;
      static bool resilient_mode;
      static bool unsafe_launch;
      static bool dynamic_independence_tests;
      static bool legion_spy_enabled;
      static int mpi_rank;
      static unsigned mpi_rank_table[MAX_NUM_NODES];
      static unsigned remaining_mpi_notifications;
      static UserEvent mpi_rank_event;
#ifdef DEBUG_HIGH_LEVEL
      static bool logging_region_tree_state;
      static bool verbose_logging;
      static bool logical_logging_only;
      static bool physical_logging_only;
      static bool check_privileges;
      static bool verify_disjointness;
      static bool bit_mask_logging;
#endif
#ifdef INORDER_EXECUTION
      static bool program_order_execution;
#endif
#ifdef DEBUG_PERF
    public:
      static unsigned long long perf_trace_tolerance;
#endif
    public:
      static unsigned num_profiling_nodes;
    };

    /**
     * \class GetAvailableContinuation
     * Continuation class for obtaining resources from the runtime
     */
    template<typename T, T (Internal::*FUNC_PTR)(bool,bool)>
    class GetAvailableContinuation : public LegionContinuation {
    public:
      GetAvailableContinuation(Internal *rt, Reservation r)
        : runtime(rt), reservation(r) { }
    public:
      inline T get_result(void)
      {
        // Try to take the reservation, see if we get it
        Event acquire_event = reservation.acquire();
        if (acquire_event.has_triggered())
        {
          // We got it! Do it now!
          result = (runtime->*FUNC_PTR)(false/*do continuation*/,
                                        true/*has lock*/);
          reservation.release();
          return result;
        }
        // Otherwise we didn't get so issue the deferred task
        // to avoid waiting for a reservation in an application task
        Event done_event = defer(runtime, acquire_event);
        done_event.wait();
        return result;
      }
      virtual void execute(void)
      {
        // If we got here we know we have the reservation
        result = (runtime->*FUNC_PTR)(false/*do continuation*/,
                                      true/*has lock*/); 
        // Now release the reservation 
        reservation.release();
      }
    protected:
      Internal *const runtime;
      Reservation reservation;
      T result;
    };

    /**
     * \class RegisterDistributedContinuation
     * Continuation class for registration of distributed collectables
     */
    class RegisterDistributedContinuation : public LegionContinuation {
    public:
      RegisterDistributedContinuation(DistributedID id,
                                      DistributedCollectable *d,
                                      Internal *rt)
        : did(id), dc(d), runtime(rt) { }
    public:
      virtual void execute(void)
      {
        runtime->register_distributed_collectable(did, dc, false/*need lock*/);
      }
    protected:
      const DistributedID did;
      DistributedCollectable *const dc;
      Internal *const runtime;
    };

    /**
     * \class CheckMessagesContinuation
     * A helper class for doing continuation for checking mapper messages
     */
    class CheckMessagesContinuation : public LegionContinuation {
    public:
      CheckMessagesContinuation(ProcessorManager *m, MapperID mid)
        : manager(m), map_id(mid) { }
    public:
      virtual void execute(void)
      {
        manager->check_mapper_messages(map_id, false/*need lock*/);
      }
    protected:
      ProcessorManager *const manager;
      MapperID map_id;
    };

    /**
     * \class MapperContinuation
     * A series of templated classes for mapper calls with varying
     * numbers of arguments. It would be really nice to use 
     * variadic templates here.
     */
    class MapperContinuation : public LegionContinuation {
    public:
      MapperContinuation(ProcessorManager *m, MapperID mid)
        : manager(m), map_id(mid) 
      { done_event = UserEvent::create_user_event(); }
    public:
      inline void complete_mapping(void)
      {
        done_event.trigger();
      }
      virtual void execute(void) = 0;
    protected:
      ProcessorManager *const manager;
      MapperID map_id;
      UserEvent done_event;
    };

    template<typename T1, void (Mapper::*CALL)(T1), bool BLOCK>
    class MapperContinuation1NoRet : public MapperContinuation {
    public:
      MapperContinuation1NoRet(ProcessorManager *m, MapperID mid,
                               const T1 &a1)
        : MapperContinuation(m, mid), arg1(a1) { }
    public:
      inline void perform_mapping(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, false/*has lock*/);
        if (!done_event.has_triggered())
          done_event.wait();
      }
      inline void invoke_mapper(Mapper *mapper)
      {
        (mapper->*CALL)(arg1);
      }
      virtual void execute(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, true/*has lock*/);
      }
    protected:
      const T1 &arg1;
    };

    template<typename T, typename T1, T (Mapper::*CALL)(T1), bool BLOCK>
    class MapperContinuation1Ret : public MapperContinuation {
    public:
      MapperContinuation1Ret(ProcessorManager *m, MapperID mid,
                             T init, const T1 &a1)
        : MapperContinuation(m, mid), arg1(a1), result(init) { }
    public:
      inline T perform_mapping(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, false/*has lock*/);
        if (!done_event.has_triggered())
          done_event.wait();
        return result;
      }
      inline void invoke_mapper(Mapper *mapper)
      {
        result = (mapper->*CALL)(arg1);
      }
      virtual void execute(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, true/*has lock*/);
      }
    protected:
      const T1 &arg1;
      T result;
    };

    template<typename T1, typename T2, void (Mapper::*CALL)(T1,T2), bool BLOCK>
    class MapperContinuation2NoRet : public MapperContinuation {
    public:
      MapperContinuation2NoRet(ProcessorManager *m, MapperID mid,
                               const T1 &a1, const T2 &a2)
        : MapperContinuation(m, mid), arg1(a1), arg2(a2) { }
    public:
      inline void perform_mapping(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, false/*has lock*/);
        if (!done_event.has_triggered())
          done_event.wait();
      }
      inline void invoke_mapper(Mapper *mapper)
      {
        (mapper->*CALL)(arg1, arg2);
      }
      virtual void execute(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, true/*has lock*/);
      }
    protected:
      const T1 &arg1;
      const T2 &arg2;
    };

    template<typename T, typename T1, typename T2, 
             T (Mapper::*CALL)(T1,T2), bool BLOCK>
    class MapperContinuation2Ret : public MapperContinuation {
    public:
      MapperContinuation2Ret(ProcessorManager *m, MapperID mid,
                             T init, const T1 &a1, const T2 &a2)
        : MapperContinuation(m, mid), arg1(a1), arg2(a2), result(init) { }
    public:
      inline T perform_mapping(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, false/*has lock*/);
        if (!done_event.has_triggered())
          done_event.wait();
        return result;
      }
      inline void invoke_mapper(Mapper *mapper)
      {
        result = (mapper->*CALL)(arg1, arg2);
      }
      virtual void execute(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, true/*has lock*/);
      }
    protected:
      const T1 &arg1;
      const T2 &arg2;
      T result;
    };

    template<typename T1, typename T2, typename T3,
             void (Mapper::*CALL)(T1,T2,T3), bool BLOCK>
    class MapperContinuation3NoRet : public MapperContinuation {
    public:
      MapperContinuation3NoRet(ProcessorManager *m, MapperID mid,
                               const T1 &a1, const T2 &a2, const T3 &a3)
        : MapperContinuation(m, mid), arg1(a1), arg2(a2), arg3(a3) { }
    public:
      inline void perform_mapping(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, false/*has lock*/);
        if (!done_event.has_triggered())
          done_event.wait();
      }
      inline void invoke_mapper(Mapper *mapper)
      {
        (mapper->*CALL)(arg1, arg2, arg3);
      }
      virtual void execute(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, true/*has lock*/);
      }
    protected:
      const T1 &arg1;
      const T2 &arg2;
      const T3 &arg3;
    };

    template<typename T, typename T1, typename T2, typename T3, 
             T (Mapper::*CALL)(T1,T2,T3), bool BLOCK>
    class MapperContinuation3Ret : public MapperContinuation {
    public:
      MapperContinuation3Ret(ProcessorManager *m, MapperID mid, T init,
                             const T1 &a1, const T2 &a2, const T3 &a3)
        : MapperContinuation(m, mid), 
          arg1(a1), arg2(a2), arg3(a3), result(init) { }
    public:
      inline T perform_mapping(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, false/*has lock*/);
        if (!done_event.has_triggered())
          done_event.wait();
        return result;
      }
      inline void invoke_mapper(Mapper *mapper)
      {
        result = (mapper->*CALL)(arg1, arg2, arg3);
      }
      virtual void execute(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, true/*has lock*/);
      }
    protected:
      const T1 &arg1;
      const T2 &arg2;
      const T3 &arg3;
      T result;
    };

    template<typename T1, typename T2, typename T3, typename T4,
             void (Mapper::*CALL)(T1,T2,T3,T4), bool BLOCK>
    class MapperContinuation4NoRet : public MapperContinuation {
    public:
      MapperContinuation4NoRet(ProcessorManager *m, MapperID mid,
                               const T1 &a1, const T2 &a2, 
                               const T3 &a3, const T4 &a4)
        : MapperContinuation(m, mid), 
          arg1(a1), arg2(a2), arg3(a3), arg4(a4) { }
    public:
      inline void perform_mapping(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, false/*has lock*/);
        if (!done_event.has_triggered())
          done_event.wait();
      }
      inline void invoke_mapper(Mapper *mapper)
      {
        (mapper->*CALL)(arg1, arg2, arg3, arg4);
      }
      virtual void execute(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, true/*has lock*/);
      }
    protected:
      const T1 &arg1;
      const T2 &arg2;
      const T3 &arg3;
      const T4 &arg4;
    };

    template<typename T, typename T1, typename T2, 
             typename T3, typename T4,
             T (Mapper::*CALL)(T1,T2,T3,T4), bool BLOCK>
    class MapperContinuation4Ret : public MapperContinuation {
    public:
      MapperContinuation4Ret(ProcessorManager *m, MapperID mid,
                             T init, const T1 &a1, const T2 &a2, 
                             const T3 &a3, const T4 &a4)
        : MapperContinuation(m, mid), 
          arg1(a1), arg2(a2), arg3(a3), arg4(a4), result(init) { }
    public:
      inline T perform_mapping(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, false/*has lock*/);
        if (!done_event.has_triggered())
          done_event.wait();
        return result;
      }
      inline void invoke_mapper(Mapper *mapper)
      {
        result = (mapper->*CALL)(arg1, arg2, arg3, arg4);
      }
      virtual void execute(void)
      {
        manager->invoke_mapper(map_id, *this, BLOCK, true/*has lock*/);
      }
    protected:
      const T1 &arg1;
      const T2 &arg2;
      const T3 &arg3;
      const T4 &arg4;
      T result;
    };

    //--------------------------------------------------------------------------
    template<typename T>
    inline T* Internal::get_available(Reservation reservation, 
                                      std::deque<T*> &queue, bool has_lock)
    //--------------------------------------------------------------------------
    {
      T *result = NULL;
      if (!has_lock)
      {
        AutoLock r_lock(reservation);
        if (!queue.empty())
        {
          result = queue.front();
          queue.pop_front();
        }
      }
      else
      {
        if (!queue.empty())
        {
          result = queue.front();
          queue.pop_front();
        }
      }
      // Couldn't find one so make one
      if (result == NULL)
        result = legion_new<T>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

  }; // namespace HighLevel
}; // namespace LegionRuntime

#endif // __RUNTIME_H__

// EOF

