/* Copyright 2013 Stanford University
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
#include "legion_utilities.h"
#include "region_tree.h"
#include "garbage_collection.h"

namespace LegionRuntime {
  namespace HighLevel {
    
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
      FRIEND_ALL_RUNTIME_CLASSES
      Impl(void);
      Impl(ArgumentMapStore *st);
      Impl(ArgumentMapStore *st,
        const std::map<DomainPoint,TaskArgument,DomainPoint::STLComparator> &);
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
      std::map<DomainPoint,TaskArgument,DomainPoint::STLComparator> arguments;
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
      Impl(Runtime *rt, DistributedID did,
           AddressSpaceID owner_space, AddressSpaceID local_space,
           TaskOp *task = NULL);
      Impl(const Future::Impl &rhs);
      virtual ~Impl(void);
    public:
      Impl& operator=(const Future::Impl &rhs);
    public:
      void get_void_result(void);
      void* get_untyped_result(void);
      bool is_empty(bool block);
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
      virtual void notify_activate(void);
      virtual void garbage_collect(void);
      virtual void notify_new_remote(AddressSpaceID);
    protected:
      void mark_sampled(void);
      void broadcast_result(void);
      bool send_future(AddressSpaceID sid);
    public:
      static void handle_future_send(Deserializer &derez, Runtime *rt, 
                                     AddressSpaceID source);
      static void handle_future_result(Deserializer &derez, Runtime *rt);
    public:
      // These three fields are only valid on the owner node
      TaskOp *const task;
      const GenerationID task_gen;
      const bool predicated;
    private:
      FRIEND_ALL_RUNTIME_CLASSES
      UserEvent ready_event;
      void *result; 
      size_t result_size;
      volatile bool empty;
      volatile bool sampled;
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
      Impl(SingleTask *ctx, TaskOp *task, 
           size_t future_size, Runtime *rt);
      Impl(SingleTask *ctx, Runtime *rt); // empty map
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
    public:
      SingleTask *const context;
      TaskOp *const task;
      const GenerationID task_gen;
      const size_t future_size;
      const bool predicated;
      const bool valid;
      Runtime *const runtime;
    private:
      Event ready_event;
      std::map<DomainPoint,Future,DomainPoint::STLComparator> futures;
      // Unlike futures, the future map is never used remotely
      // so it can create and destroy its own lock.
      Lock lock;
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
      Impl(const RegionRequirement &req, Event ready_event,
           bool mapped, SingleTask *ctx, MapperID mid,
           MappingTagID tag, Runtime *rt);
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
      void reset_reference(const InstanceRef &ref, 
                           UserEvent term_event);
      Event get_ready_event(void) const;
      const InstanceRef& get_reference(void) const;
    public:
      Runtime *const runtime;
      SingleTask *const context;
      const MapperID map_id;
      const MappingTagID tag;
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
    };

    /**
     * \class ProcessorManager
     * This class manages all the state for a single processor
     * within a given instance of the Runtime.  It keeps
     * queues for each of the different stages that operations
     * undergo and also tracks when the scheduling task needs
     * to be run for a processor.
     */
    class ProcessorManager {
    public:
      ProcessorManager(Processor proc, Processor::Kind proc_kind,
                       Runtime *rt, unsigned min_out,
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
      // Functions that perform mapping calls
      void invoke_mapper_set_task_options(TaskOp *task);
      void invoke_mapper_select_variant(TaskOp *task);
      bool invoke_mapper_pre_map_task(TaskOp *task);
      bool invoke_mapper_map_task(TaskOp *task);
      void invoke_mapper_failed_mapping(Mappable *mappable);
      void invoke_mapper_notify_result(Mappable *mappable);
      void invoke_mapper_slice_domain(TaskOp *task,
                                      std::vector<Mapper::DomainSplit> &splits);
      bool invoke_mapper_map_inline(Inline *op);
      bool invoke_mapper_map_copy(Copy *op);
      bool invoke_mapper_speculate(TaskOp *op, bool &value); 
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
    public:
      void perform_scheduling(void);
      void process_steal_request(Processor thief, 
                                 const std::vector<MapperID> &thieves);
      void process_advertisement(Processor advertiser, MapperID mid);
    public:
      void increment_pending(void);
      void decrement_pending(void);
      void start_execution(void);
      void pause_execution(void);
    public:
      void add_to_dependence_queue(Operation *op);
      void add_to_ready_queue(TaskOp *op, bool previous_failure);
      void add_to_local_ready_queue(Operation *op, bool previous_failure);
    public:
      Event find_gc_epoch_event(void);
    protected:
      bool perform_dependence_checks(void);
      bool perform_other_operations(void);
      void perform_mapping_operations(void);
      void issue_advertisements(MapperID mid);
    protected:
      // Immutable state
      Runtime *const runtime;
      const Processor local_proc;
      const Processor::Kind proc_kind;
      const Processor utility_proc;
      // Is our utility processor different from the local proc
      const bool explicit_utility_proc;
      // Effective super-scalar width of the runtime
      const unsigned superscalar_width;
      // Minimum number of tasks that must be scheduled to disable idle task
      const unsigned min_outstanding;
      // Is stealing disabled 
      const bool stealing_disabled;
      // Maximum number of outstanding steals permitted by any mapper
      const unsigned max_outstanding_steals;
    protected:
      // Mutable state
      unsigned current_pending;
      bool current_executing;
      bool idle_task_enabled;
    protected:
      // Since dependence analysis is usually on the critical path,
      // we allow the processor managers to create garbage collection
      // epoch events which ensure garbage collections will only be
      // done when there are no dependence analyses to perform.  While
      // this works for dependence analysis, it is generally unsafe for
      // the garbage collector to wait on mapping analyses to be done
      // since some mapping analyses may require the garbage collector
      // to run and free-up valuable memory resources in order for 
      // mappings to succeed.  These values are protected by the 
      // dependence lock.
      UserEvent gc_epoch_trigger;
      Event gc_epoch_event;
    protected:
      // Note locks are declaraed in the order in which they
      // must be taken 
      Lock idle_lock;
      // Lock for protecting the dependence queue
      Lock dependence_lock;
      // Lock for protecting the ready queues and
      // the other ready queue
      Lock queue_lock;
      // A list of operations needing dependence analysis.  We keep
      // a seperate list for each depth.  This allows us to pull
      // from deeper lists first which is a performance optimization
      // to prevent operations from deeper levels of the task tree
      // from having to wait for operations at shallower levels.
      std::deque<std::deque<Operation*> > dependence_queues;
      // For each mapper, a list of tasks that are ready to map
      std::vector<std::list<TaskOp*> > ready_queues;
      // All the local operations that are ready to be performed
      // which we will do in batches as they become ready
      std::list<Operation*> local_ready_queue;
      // Mapper objects
      std::vector<Mapper*> mapper_objects;
      // Mapper locks
      std::vector<Lock> mapper_locks;
      // For each mapper, the set of processors to which it
      // has outstanding steal requests
      std::map<MapperID,std::set<Processor> > outstanding_steal_requests;
      // Failed thiefs to notify when tasks become available
      std::multimap<MapperID,Processor> failed_thiefs;
      // Locks for stealing and thieving
      Lock stealing_lock;
      Lock thieving_lock;
    };

    /**
     * \class MessageManager
     * This class manages sending and receiving of message between
     * instances of the Runtime residing on different nodes.
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
      enum MessageKind {
        TASK_MESSAGE,
        STEAL_MESSAGE,
        ADVERTISEMENT_MESSAGE,
        SEND_INDEX_SPACE_NODE,
        SEND_INDEX_PARTITION_NODE,
        SEND_FIELD_SPACE_NODE,
        SEND_LOGICAL_REGION_NODE,
        INDEX_SPACE_DESTRUCTION_MESSAGE,
        INDEX_PARTITION_DESTRUCTION_MESSAGE,
        FIELD_SPACE_DESTRUCTION_MESSAGE,
        LOGICAL_REGION_DESTRUCTION_MESSAGE,
        LOGICAL_PARTITION_DESTRUCTION_MESSAGE,
        FIELD_ALLOCATION_MESSAGE,
        FIELD_DESTRUCTION_MESSAGE,
        INDIVIDUAL_REMOTE_MAPPED,
        INDIVIDUAL_REMOTE_COMPLETE,
        INDIVIDUAL_REMOTE_COMMIT,
        SLICE_REMOTE_MAPPED,
        SLICE_REMOTE_COMPLETE,
        SLICE_REMOTE_COMMIT,
        DISTRIBUTED_REMOVE_RESOURCE,
        DISTRIBUTED_REMOVE_REMOTE,
        DISTRIBUTED_ADD_REMOTE,
        HIERARCHICAL_REMOVE_RESOURCE,
        HIERARCHICAL_REMOVE_REMOTE,
        SEND_BACK_USER,
        SEND_USER,
        SEND_INSTANCE_VIEW,
        SEND_BACK_INSTANCE_VIEW,
        SEND_REDUCTION_VIEW,
        SEND_BACK_REDUCTION_VIEW,
        SEND_INSTANCE_MANAGER,
        SEND_REDUCTION_MANAGER,
        SEND_REGION_STATE,
        SEND_PARTITION_STATE,
        SEND_BACK_REGION_STATE,
        SEND_BACK_PARTITION_STATE,
        SEND_REMOTE_REFERENCES,
        SEND_INDIVIDUAL_REQUEST,
        SEND_INDIVIDUAL_RETURN,
        SEND_SLICE_REQUEST,
        SEND_SLICE_RETURN,
        SEND_FUTURE,
        SEND_FUTURE_RESULT,
      };
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
      MessageManager(AddressSpaceID remote, 
                     Runtime *rt, size_t max,
                     const std::set<Processor> &procs);
      MessageManager(const MessageManager &rhs);
      ~MessageManager(void);
    public:
      MessageManager& operator=(const MessageManager &rhs);
    public:
      // Methods for sending tasks
      void send_task(Serializer &rez, bool flush);
      void send_steal_request(Serializer &rez, bool flush);
      void send_advertisement(Serializer &rez, bool flush);
      void send_index_space_node(Serializer &rez, bool flush);
      void send_index_partition_node(Serializer &rez, bool flush);
      void send_field_space_node(Serializer &rez, bool flush);
      void send_logical_region_node(Serializer &rez, bool flush);
      void send_index_space_destruction(Serializer &rez, bool flush);
      void send_index_partition_destruction(Serializer &rez, bool flush);
      void send_field_space_destruction(Serializer &rez, bool flush);
      void send_logical_region_destruction(Serializer &rez, bool flush);
      void send_logical_partition_destruction(Serializer &rez, bool flush);
      void send_field_allocation(Serializer &rez, bool flush);
      void send_field_destruction(Serializer &rez, bool flush);
      void send_individual_remote_mapped(Serializer &rez, bool flush);
      void send_individual_remote_complete(Serializer &rez, bool flush);
      void send_individual_remote_commit(Serializer &rez, bool flush);
      void send_slice_remote_mapped(Serializer &rez, bool flush);
      void send_slice_remote_complete(Serializer &rez, bool flush);
      void send_slice_remote_commit(Serializer &rez, bool flush);
      void send_remove_distributed_resource(Serializer &rez, bool flush);
      void send_remove_distributed_remote(Serializer &rez, bool flush);
      void send_add_distributed_remote(Serializer &rez, bool flush);
      void send_remove_hierarchical_resource(Serializer &rez, bool flush);
      void send_remove_hierarchical_remote(Serializer &rez, bool flush);
      void send_back_user(Serializer &rez, bool flush);
      void send_user(Serializer &rez, bool flush);
      void send_instance_view(Serializer &rez, bool flush);
      void send_back_instance_view(Serializer &rez, bool flush);
      void send_reduction_view(Serializer &rez, bool flush);
      void send_back_reduction_view(Serializer &rez, bool flush);
      void send_instance_manager(Serializer &rez, bool flush);
      void send_reduction_manager(Serializer &rez, bool flush);
      void send_region_state(Serializer &rez, bool flush);
      void send_partition_state(Serializer &rez, bool flush);
      void send_back_region_state(Serializer &rez, bool flush);
      void send_back_partition_state(Serializer &rez, bool flush);
      void send_remote_references(Serializer &rez, bool flush);
      void send_individual_request(Serializer &rez, bool flush);
      void send_individual_return(Serializer &rez, bool flush);
      void send_slice_request(Serializer &rez, bool flush);
      void send_slice_return(Serializer &rez, bool flush);
      void send_future(Serializer &rez, bool flush);
      void send_future_result(Serializer &rez, bool flush);
    public:
      // Receiving message method
      void process_message(const void *args, size_t arglen);
    private:
      void package_message(Serializer &rez, MessageKind k, bool flush);
      void send_message(bool complete);
      void handle_messages(unsigned num_messages, 
                           const char *args, size_t arglen);
      void buffer_messages(unsigned num_messages,
                           const void *args, size_t arglen);
    public:
      const AddressSpaceID local_address_space;
      const AddressSpaceID remote_address_space;
      const std::set<Processor> remote_address_procs;
    private:
      Runtime *const runtime;
      // State for sending messages
      Processor target;
      Lock send_lock;
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
    private:
      int ctx;
    };

    /**
     * \class Runtime
     * This is the actual implementation of the Legion runtime functionality
     * that implements the underlying interface for the HighLevelRuntime
     * objects.  Most of the calls in the HighLevelRuntime class translate
     * directly to calls to this interface.  Unfortunately this adds
     * an extra function call overhead to every runtime call because C++
     * is terrible and doesn't have mix-in classes.
     */
    class Runtime {
    public:
      Runtime(Machine *m, AddressSpaceID space_id,
              const std::set<Processor> &local_procs,
              const std::set<AddressSpaceID> &address_spaces,
              const std::map<Processor,AddressSpaceID> &proc_spaces);
      Runtime(const Runtime &rhs);
      ~Runtime(void);
    public:
      Runtime& operator=(const Runtime &rhs);
    public:
      void launch_top_level_task(Processor proc);
      void perform_one_time_logging(void);
    public:
      IndexSpace create_index_space(Context ctx, size_t max_num_elmts);
      IndexSpace create_index_space(Context ctx, Domain domain);
      void destroy_index_space(Context ctx, IndexSpace handle);
      // Called from deletion op
      bool finalize_index_space_destroy(IndexSpace handle);
    public:
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                            const Coloring &coloring,
                                            bool disjoint,
                                            int part_color);
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                            Domain color_space,
                                            const DomainColoring &coloring,
                                            bool disjoint,
                                            int part_color);
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
      Accessor::RegionAccessor<Accessor::AccessorType::Generic> field_accessor,
                                            int part_color);
      void destroy_index_partition(Context ctx, IndexPartition handle);
      // Called from deletion op
      void finalize_index_partition_destroy(IndexPartition handle);
    public:
      IndexPartition get_index_partition(Context ctx, IndexSpace parent, 
                                         Color color);
      IndexSpace get_index_subspace(Context ctx, IndexPartition p, 
                                    Color color); 
      Domain get_index_space_domain(Context ctx, IndexSpace handle);
      Domain get_index_partition_color_space(Context ctx, IndexPartition p);
      Color get_index_space_color(Context ctx, IndexSpace handle);
      Color get_index_partition_color(Context ctx, IndexPartition handle);
    public:
      ptr_t safe_cast(Context ctx, ptr_t pointer, LogicalRegion region);
      DomainPoint safe_cast(Context ctx, DomainPoint point, 
                            LogicalRegion region);
    public:
      FieldSpace create_field_space(Context ctx);
      void destroy_field_space(Context ctx, FieldSpace handle);
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
      LogicalPartition get_logical_partition_by_color(Context ctx, 
                                                      LogicalRegion parent, 
                                                      Color c);
      LogicalPartition get_logical_partition_by_tree(Context ctx, 
                                                     IndexPartition handle, 
                                                     FieldSpace fspace, 
                                                     RegionTreeID tid); 
      LogicalRegion get_logical_subregion(Context ctx, LogicalPartition parent, 
                                          IndexSpace handle);
      LogicalRegion get_logical_subregion_by_color(Context ctx, 
                                                   LogicalPartition parent, 
                                                   Color c);
      LogicalRegion get_logical_subregion_by_tree(Context ctx, 
                                                  IndexSpace handle, 
                                                  FieldSpace fspace, 
                                                  RegionTreeID tid);
      Color get_logical_region_color(Context ctx, LogicalRegion handle);
      Color get_logical_partition_color(Context ctx, LogicalPartition handle);
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
      void map_all_regions(Context ctx);
      void unmap_all_regions(Context ctx);
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
      Reservation create_reservation(Context ctx);
      void destroy_reservation(Context ctx, Reservation r);
      PhaseBarrier create_phase_barrier(Context ctx, unsigned participants);
      void destroy_phase_barrier(Context ctx, PhaseBarrier pb);
      PhaseBarrier advance_phase_barrier(Context ctx, PhaseBarrier pb);
      void issue_legion_mapping_fence(Context ctx);
      void issue_legion_execution_fence(Context ctx);
    public:
      Mapper* get_mapper(Context ctx, MapperID id);
      Processor get_executing_processor(Context ctx);
      void raise_region_exception(Context ctx, PhysicalRegion region, 
                                  bool nuclear);
    public:
      void add_mapper(MapperID map_id, Mapper *mapper, Processor proc);
      void replace_default_mapper(Mapper *mapper, Processor proc);
    public:
      FieldID allocate_field(Context ctx, FieldSpace space, 
                             size_t field_size, FieldID fid, bool local);
      void free_field(Context ctx, FieldSpace space, FieldID fid);
      void allocate_fields(Context ctx, FieldSpace space, 
                           const std::vector<size_t> &sizes,
                           std::vector<FieldID> &resulting_fields, bool local);
      void free_fields(Context ctx, FieldSpace space, 
                       const std::set<FieldID> &to_free);
    public:
      const std::vector<PhysicalRegion>& begin_task(Context ctx);
      void end_task(Context ctx, const void *result, size_t result_size,
                    bool owned);
      const void* get_local_args(Context ctx, DomainPoint &point, 
                                 size_t &local_size);
    public:
      // Messaging functions
      MessageManager* find_messenger(AddressSpaceID sid) const;
      MessageManager* find_messenger(Processor target) const;
      AddressSpaceID find_address_space(Processor target) const;
      void send_task(Processor target, TaskOp *task);
      void send_tasks(Processor target, const std::set<TaskOp*> &tasks);
      void send_steal_request(const std::multimap<Processor,MapperID> &targets,
                              Processor thief);
      void send_advertisements(const std::set<Processor> &targets,
                              MapperID map_id, Processor source);
      void send_index_space_node(AddressSpaceID target, Serializer &rez);
      void send_index_partition_node(AddressSpaceID target, Serializer &rez);
      void send_field_space_node(AddressSpaceID target, Serializer &rez);
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
      void send_field_allocation(FieldSpace space, FieldID fid, 
                                 size_t size, unsigned idx, 
                                 AddressSpaceID target);
      void send_field_destruction(FieldSpace space, FieldID fid, 
                                  AddressSpaceID target); 
      void send_individual_remote_mapped(Processor target, 
                                         Serializer &rez, bool flush = true);
      void send_individual_remote_complete(Processor target, Serializer &rez);
      void send_individual_remote_commit(Processor target, Serializer &rez);
      void send_slice_remote_mapped(Processor target, Serializer &rez);
      void send_slice_remote_complete(Processor target, Serializer &rez);
      void send_slice_remote_commit(Processor target, Serializer &rez);
      void send_remove_distributed_resource(AddressSpaceID target,
                                            Serializer &rez);
      void send_remove_distributed_remote(AddressSpaceID target,
                                          Serializer &rez);
      void send_add_distributed_remote(AddressSpaceID target, Serializer &rez);
      void send_remove_hierarchical_resource(AddressSpaceID target,
                                             Serializer &rez);
      void send_remove_hierarchical_remote(AddressSpaceID target, 
                                           Serializer &rez);
      void send_back_user(AddressSpaceID target, Serializer &rez);
      void send_user(AddressSpaceID target, Serializer &rez);
      void send_instance_view(AddressSpaceID target, Serializer &rez);
      void send_back_instance_view(AddressSpaceID target, Serializer &rez);
      void send_reduction_view(AddressSpaceID target, Serializer &rez);
      void send_back_reduction_view(AddressSpaceID target, Serializer &rez);
      void send_instance_manager(AddressSpaceID target, Serializer &rez);
      void send_reduction_manager(AddressSpaceID target, Serializer &rez);
      void send_region_state(AddressSpaceID target, Serializer &rez);
      void send_partition_state(AddressSpaceID target, Serializer &rez);
      void send_back_region_state(AddressSpaceID target, Serializer &rez);
      void send_back_partition_state(AddressSpaceID target, Serializer &rez);
      void send_remote_references(AddressSpaceID target, Serializer &rez);
      void send_individual_request(AddressSpaceID target, Serializer &rez);
      void send_individual_return(AddressSpaceID target, Serializer &rez);
      void send_slice_request(AddressSpaceID target, Serializer &rez);
      void send_slice_return(AddressSpaceID target, Serializer &rez);
      void send_future(AddressSpaceID target, Serializer &rez);
      void send_future_result(AddressSpaceID target, Serializer &rez);
    public:
      // Complementary tasks for handling messages
      void handle_task(Deserializer &derez);
      void handle_steal(Deserializer &derez);
      void handle_advertisement(Deserializer &derez);
      void handle_index_space_node(Deserializer &derez, AddressSpaceID source);
      void handle_index_partition_node(Deserializer &derez,
                                       AddressSpaceID source);
      void handle_field_space_node(Deserializer &derez, AddressSpaceID source);
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
      void handle_field_allocation(Deserializer &derez, AddressSpaceID source);
      void handle_field_destruction(Deserializer &derez, AddressSpaceID source);
      void handle_individual_remote_mapped(Deserializer &derez);
      void handle_individual_remote_complete(Deserializer &derez);
      void handle_individual_remote_commit(Deserializer &derez);
      void handle_slice_remote_mapped(Deserializer &derez);
      void handle_slice_remote_complete(Deserializer &derez);
      void handle_slice_remote_commit(Deserializer &derez);
      void handle_distributed_remove_resource(Deserializer &derez);
      void handle_distributed_remove_remote(Deserializer &derez,
                                            AddressSpaceID source);
      void handle_distributed_add_remote(Deserializer &derez);
      void handle_hierarchical_remove_resource(Deserializer &derez);
      void handle_hierarchical_remove_remote(Deserializer &derez);
      void handle_send_back_user(Deserializer &derez, AddressSpaceID source);
      void handle_send_user(Deserializer &derez, AddressSpaceID source);
      void handle_send_instance_view(Deserializer &derez, 
                                     AddressSpaceID source);
      void handle_send_back_instance_view(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_send_reduction_view(Deserializer &derez,
                                      AddressSpaceID source);
      void handle_send_back_reduction_view(Deserializer &derez,
                                           AddressSpaceID source);
      void handle_send_instance_manager(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_send_reduction_manager(Deserializer &derez,
                                         AddressSpaceID source);
      void handle_send_region_state(Deserializer &derez, AddressSpaceID source);
      void handle_send_partition_state(Deserializer &derez, 
                                       AddressSpaceID source);
      void handle_send_back_region_state(Deserializer &derez, 
                                         AddressSpaceID source);
      void handle_send_back_partition_state(Deserializer &derez, 
                                            AddressSpaceID source);
      void handle_send_remote_references(Deserializer &derez);
      void handle_individual_request(Deserializer &derez, 
                                     AddressSpaceID source);
      void handle_individual_return(Deserializer &derez);
      void handle_slice_request(Deserializer &derez, AddressSpaceID source);
      void handle_slice_return(Deserializer &derez);
      void handle_future_send(Deserializer &derez, AddressSpaceID source);
      void handle_future_result(Deserializer &derez);
    public:
      // Helper methods for the RegionTreeForest
      inline unsigned get_context_count(void) { return total_contexts; }
      inline unsigned get_start_color(void) const { return address_space; }
      inline unsigned get_color_modulus(void) const { return runtime_stride; }
    public:
      void increment_pending(Processor p);
      void decrement_pending(Processor p);
      void start_execution(Processor p);
      void pause_execution(Processor p);
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
      void invoke_mapper_failed_mapping(Processor target, Mappable *mappable);
      void invoke_mapper_notify_result(Processor target, Mappable *mappable);
      void invoke_mapper_slice_domain(Processor target, MultiTask *task,
                                      std::vector<Mapper::DomainSplit> &splits);
      bool invoke_mapper_map_inline(Processor target, Inline *op);
      bool invoke_mapper_map_copy(Processor target, Copy *op);
      bool invoke_mapper_speculate(Processor target, TaskOp *task, bool &value);
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
    public:
      void allocate_context(SingleTask *task);
      void free_context(SingleTask *task);
    public:
      DistributedID get_available_distributed_id(void);
      void free_distributed_id(DistributedID did);
    public:
      void register_distributed_collectable(DistributedID did,
                                            DistributedCollectable *dc);
      void unregister_distributed_collectable(DistributedID did);
      DistributedCollectable* find_distributed_collectable(DistributedID did);
    public:
      void register_hierarchical_collectable(DistributedID did,
                                             HierarchicalCollectable *hc);
      void unregister_hierarchical_collectable(DistributedID did);
      HierarchicalCollectable* find_hierarchical_collectable(DistributedID did);
    public:
      void register_future(DistributedID did, Future::Impl *impl);
      void unregister_future(DistributedID did);
      bool has_future(DistributedID did);
      Future::Impl* find_future(DistributedID did);
    public:
      Event find_gc_epoch_event(Processor local_proc);
    public:
      IndividualTask* get_available_individual_task(void);
      PointTask*      get_available_point_task(void);
      IndexTask*      get_available_index_task(void);
      SliceTask*      get_available_slice_task(void);
      RemoteTask*     get_available_remote_task(void);
      InlineTask*     get_available_inline_task(void);
      MapOp*          get_available_map_op(void);
      CopyOp*         get_available_copy_op(void);
      FenceOp*        get_available_fence_op(void);
      DeletionOp*     get_available_deletion_op(void);
      CloseOp*        get_available_close_op(void);
      FuturePredOp*   get_available_future_pred_op(void);
      NotPredOp*      get_available_not_pred_op(void);
      AndPredOp*      get_available_and_pred_op(void);
      OrPredOp*       get_available_or_pred_op(void);
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
      void free_deletion_op(DeletionOp *op);
      void free_close_op(CloseOp *op); 
      void free_future_predicate_op(FuturePredOp *op);
      void free_not_predicate_op(NotPredOp *op);
      void free_and_predicate_op(AndPredOp *op);
      void free_or_predicate_op(OrPredOp *op);
    public:
      RemoteTask* find_or_init_remote_context(UniqueID uid); 
      bool is_local(Processor proc) const;
    public:
      IndexPartition  get_unique_partition_id(void);
      FieldSpaceID    get_unique_field_space_id(void);
      RegionTreeID    get_unique_tree_id(void);
      UniqueID        get_unique_operation_id(void);
      FieldID         get_unique_field_id(void);
    public:
      // Verify that a region requirement is valid
      LegionErrorType verify_requirement(const RegionRequirement &req,
                                         FieldID &bad_field);
    public:
      // Methods for helping with dumb nested class scoping problems
      Future help_create_future(TaskOp *task = NULL);
      void help_complete_future(const Future &f);
      bool help_reset_future(const Future &f);
#ifdef DYNAMIC_TESTS
    public:
      bool perform_dynamic_independence_tests(void);
#endif
    public:
      // These are the static methods that become the meta-tasks
      // for performing all the needed runtime operations
      static void initialize_runtime(
                          const void *args, size_t arglen, Processor p);
      static void shutdown_runtime(
                          const void *args, size_t arglen, Processor p);
      static void schedule_runtime(
                          const void *args, size_t arglen, Processor p);
      static void message_task(
                          const void *args, size_t arglen, Processor p);
      static void post_end_task(
                          const void *args, size_t arglen, Processor p);
      static void copy_complete_task(
                          const void *args, size_t arglen, Processor p);
      static void fence_complete_task(
                          const void *args, size_t arglen, Processor p);
      static void close_complete_task(
                          const void *args, size_t arglen, Processor p);
      static void reclaim_local_field_task(
                          const void *args, size_t arglen, Processor p);
      static void deferred_collect_task(
                          const void *args, size_t arglen, Processor p);
    protected:
      // Internal runtime methods invoked by the above static methods
      // after the find the right runtime instance to call
      void process_schedule_request(Processor p);
      void process_message_task(const void *args, size_t arglen);
    public:
      // The HighLevelRuntime wrapper for this class
      HighLevelRuntime *const high_level;
      // The machine object for this runtime
      Machine *const machine;
      const AddressSpaceID address_space; 
      const unsigned runtime_stride; // stride for uniqueness
      RegionTreeForest *const forest;
    protected:
      // Internal runtime state 
      // The local processor managed by this runtime
      const std::set<Processor> local_procs;
      // Processor managers for each of the local processors
      std::map<Processor,ProcessorManager*> proc_managers;
      // Message managers for each of the other runtimes
      std::map<AddressSpaceID,MessageManager*> message_managers;
      // For every processor map it to its address space
      const std::map<Processor,AddressSpaceID> proc_spaces;
#ifdef DEBUG_HIGH_LEVEL
      friend class TreeStateLogger;
      TreeStateLogger *get_tree_state_logger(void) { return tree_state_logger; }
#endif
    protected:
      unsigned unique_partition_id;
      unsigned unique_field_space_id;
      unsigned unique_tree_id;
      unsigned unique_operation_id;
      unsigned unique_field_id;
    protected:
      Lock available_lock;
      unsigned total_contexts;
      std::deque<RegionTreeContext> available_contexts;
    protected:
      Lock distributed_id_lock;
      DistributedID unique_distributed_id;
      std::deque<DistributedID> available_distributed_ids;
    protected:
      // Garbage collection data structures
      Lock distributed_collectable_lock;
      std::map<DistributedID,DistributedCollectable*> dist_collectables;
      Lock hierarchical_collectable_lock;
      std::map<DistributedID,HierarchicalCollectable*> hier_collectables;
    protected:
      // Keep track of futures
      Lock future_lock;
      std::map<DistributedID,Future::Impl*> local_futures;
    protected:
      // The runtime keeps track of remote contexts so they
      // can be re-used by multiple tasks that get sent remotely
      Lock remote_lock;
      std::map<UniqueID,RemoteTask*> remote_contexts;
    protected:
      Lock individual_task_lock;
      Lock point_task_lock;
      Lock index_task_lock;
      Lock slice_task_lock;
      Lock remote_task_lock;
      Lock inline_task_lock;
      Lock map_op_lock;
      Lock copy_op_lock;
      Lock fence_op_lock;
      Lock deletion_op_lock;
      Lock close_op_lock;
      Lock future_pred_op_lock;
      Lock not_pred_op_lock;
      Lock and_pred_op_lock;
      Lock or_pred_op_lock;
    protected:
      std::deque<IndividualTask*> available_individual_tasks;
      std::deque<PointTask*>      available_point_tasks;
      std::deque<IndexTask*>      available_index_tasks;
      std::deque<SliceTask*>      available_slice_tasks;
      std::deque<RemoteTask*>     available_remote_tasks;
      std::deque<InlineTask*>     available_inline_tasks;
      std::deque<MapOp*>          available_map_ops;
      std::deque<CopyOp*>         available_copy_ops;
      std::deque<FenceOp*>        available_fence_ops;
      std::deque<DeletionOp*>     available_deletion_ops;
      std::deque<CloseOp*>        available_close_ops;
      std::deque<FuturePredOp*>   available_future_pred_ops;
      std::deque<NotPredOp*>      available_not_pred_ops;
      std::deque<AndPredOp*>      available_and_pred_ops;
      std::deque<OrPredOp*>       available_or_pred_ops;
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger *tree_state_logger;
      // For debugging purposes keep track of
      // some of the outstanding tasks
      std::set<IndividualTask*> out_individual_tasks;
      std::set<PointTask*>      out_point_tasks;
      std::set<IndexTask*>      out_index_tasks;
      std::set<SliceTask*>      out_slice_tasks;
#endif
    public:
      // Static methods for start-up and callback phases
      static int start(int argc, char **argv, bool background);
      static void wait_for_shutdown(void);
      static void set_top_level_task_id(Processor::TaskFuncID top_id);
      static const ReductionOp* get_reduction_op(ReductionOpID redop_id);
      static void set_registration_callback(RegistrationCallbackFnptr callback);
      static InputArgs& get_input_args(void);
      static Runtime* get_runtime(Processor p);
      static LowLevel::ReductionOpTable& get_reduction_table(void);
      static ProjectionID register_region_projection_function(
                                    ProjectionID handle, void *func_ptr);
      static ProjectionID register_partition_projection_function(
                                    ProjectionID handle, void *func_ptr);
      static TaskID update_collection_table(
                      LowLevelFnptr low_level_ptr, InlineFnptr inline_ptr,
                      TaskID uid, Processor::Kind proc_kind, 
                      bool single_task, bool index_space_task,
                      VariantID vid, size_t return_size,
                      const TaskConfigOptions &options);
      static TaskVariantCollection* get_variant_collection(
                      Processor::TaskFuncID tid);
      static PartitionProjectionFnptr 
                    find_partition_projection_function(ProjectionID pid);
      static RegionProjectionFnptr
                    find_region_projection_function(ProjectionID pid);
      static InlineFnptr find_inline_function(Processor::TaskFuncID fid);
    private:
      static Processor::TaskIDTable& get_task_table(
                                          bool add_runtime_tasks = true);
      static std::map<Processor::TaskFuncID,InlineFnptr>& 
                                            get_inline_table(void);
      static std::map<Processor::TaskFuncID,TaskVariantCollection*>& 
                                            get_collection_table(void);
      static RegionProjectionTable& get_region_projection_table(void);
      static PartitionProjectionTable& get_partition_projection_table(void);
      static void register_runtime_tasks(Processor::TaskIDTable &table);
      static Processor::TaskFuncID get_next_available_id(void);
      static void log_machine(Machine *machine);
    public:
      // Static member variables
      static Runtime *runtime_map[(MAX_NUM_PROCS+1/*+1 for NO_PROC*/)];
      static unsigned startup_arrivals;
      static volatile RegistrationCallbackFnptr registration_callback;
      static Processor::TaskFuncID legion_main_id;
      static unsigned max_task_window_per_context;
      static unsigned min_tasks_to_schedule;
      static unsigned superscalar_width;
      static unsigned max_message_size;
      static unsigned max_filter_size;
      static bool separate_runtime_instances;
      static bool stealing_disabled;
      static bool resilient_mode;
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
#ifdef DYNAMIC_TESTS
    public:
      static bool dynamic_independence_tests;
#endif 
    public:
      // The baseline time for profiling
      static const long long init_time;
    };

  }; // namespace HighLevel
}; // namespace LegionRuntime

#endif // __RUNTIME_H__

// EOF

