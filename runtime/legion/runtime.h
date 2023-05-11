/* Copyright 2023 Stanford University, NVIDIA Corporation
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
      void set_point(const DomainPoint &point, const UntypedBuffer &arg,
                     bool replace);
      void set_point(const DomainPoint &point, const Future &f, bool replace);
      bool remove_point(const DomainPoint &point);
      UntypedBuffer get_point(const DomainPoint &point);
    public:
      FutureMap freeze(TaskContext *ctx, Provenance *provenance);
      void unfreeze(void);
    public:
      Runtime *const runtime;
    private:
      FutureMap future_map;
      std::map<DomainPoint,Future> arguments;
      std::set<RtEvent> point_set_deletion_preconditions;
      IndexSpaceNode *point_set;
      unsigned dimensionality;
      unsigned dependent_futures; // number of futures with producer ops
      bool update_point_set;
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
      FieldAllocatorImpl(FieldSpaceNode *node, 
                         TaskContext *context, RtEvent ready);
      FieldAllocatorImpl(const FieldAllocatorImpl &rhs);
      ~FieldAllocatorImpl(void);
    public:
      FieldAllocatorImpl& operator=(const FieldAllocatorImpl &rhs);
    public:
      inline FieldSpace get_field_space(void) const { return field_space; }
    public:
      FieldID allocate_field(size_t field_size, 
                             FieldID desired_fieldid,
                             CustomSerdezID serdez_id, bool local,
                             Provenance *provenance);
      FieldID allocate_field(const Future &field_size, 
                             FieldID desired_fieldid,
                             CustomSerdezID serdez_id, bool local,
                             Provenance *provenance);
      void free_field(FieldID fid, const bool unordered,
                      Provenance *provenance);
    public:
      void allocate_fields(const std::vector<size_t> &field_sizes,
                           std::vector<FieldID> &resulting_fields,
                           CustomSerdezID serdez_id, bool local,
                           Provenance *provenance);
      void allocate_fields(const std::vector<Future> &field_sizes,
                           std::vector<FieldID> &resulting_fields,
                           CustomSerdezID serdez_id, bool local,
                           Provenance *provenance);
      void free_fields(const std::set<FieldID> &to_free, const bool unordered,
                       Provenance *provenance = NULL);
    public:
      inline void free_from_runtime(void) { free_from_application = false; }
    public:
      FieldSpace field_space;
      FieldSpaceNode *const node;
      TaskContext *const context;
      const RtEvent ready_event;
    protected:
      bool free_from_application;
    };

    /**
     * \class PredicateImpl
     * This class provides the base support for a predicate and
     * any state needed to manage the mapping of things that 
     * depend on a predicate value
     */
    class PredicateImpl : public Collectable {
    public:
      PredicateImpl(Operation *creator);
      PredicateImpl(const PredicateImpl &rhs) = delete;
      virtual ~PredicateImpl(void);
    public:
      PredicateImpl& operator=(const PredicateImpl &rhs) = delete;
    public:
      // This returns the predicate value if it is set or returns the
      // names of the guards to use if has not been set
      virtual bool get_predicate(size_t context_index,
          PredEvent &true_guard, PredEvent &false_guard);
      bool get_predicate(RtEvent &ready);
      virtual void set_predicate(bool value);
    public:
      InnerContext *const context;
      Operation *const creator;
      const GenerationID creator_gen;
      const UniqueID creator_uid;
      const size_t creator_ctx_index;
    protected:
      mutable LocalLock predicate_lock;
      PredUserEvent true_guard, false_guard;
      RtUserEvent ready_event;
      int value; // <0 is unset, 0 is false, >0 is true
    };

    /**
     * \class ReplPredicateImpl
     * This is a predicate implementation for control replication
     * contexts. It provides the same functionality as the normal
     * version, but it also has one extra invariant, which is that
     * it guarantees that it will not return a false predicate
     * result until it guarantees that all the shards will return
     * the same false result for all equivalent operations.
     */
    class ReplPredicateImpl : public PredicateImpl {
    public:
      ReplPredicateImpl(Operation *creator, CollectiveID id);
      ReplPredicateImpl(const ReplPredicateImpl &rhs) = delete;
      virtual ~ReplPredicateImpl(void);
    public:
      ReplPredicateImpl& operator=(const ReplPredicateImpl &rhs) = delete;
    public:
      virtual bool get_predicate(size_t context_index,
          PredEvent &true_guard, PredEvent &false_guard);
      virtual void set_predicate(bool value);
    protected:
      const CollectiveID collective_id;
      size_t max_observed_index;
      PredicateCollective *collective;
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
      struct FutureCallbackArgs : public LgTaskArgs<FutureCallbackArgs> {
      public:
        static const LgTaskID TASK_ID = LG_FUTURE_CALLBACK_TASK_ID;
      public:
        FutureCallbackArgs(FutureImpl *i);
      public:
        FutureImpl *const impl;
      };
      struct CallbackReleaseArgs : public LgTaskArgs<CallbackReleaseArgs> {
      public:
        static const LgTaskID TASK_ID = LG_CALLBACK_RELEASE_TASK_ID;
      public:
        CallbackReleaseArgs(FutureFunctor *functor, bool own_functor);
      public:
        FutureFunctor *const functor;
        const bool own_functor;
      };
      struct FutureBroadcastArgs : public LgTaskArgs<FutureBroadcastArgs> {
      public:
        static const LgTaskID TASK_ID = LG_FUTURE_BROADCAST_TASK_ID;
      public:
        FutureBroadcastArgs(FutureImpl *i);
      public:
        FutureImpl *const impl;
      };
      struct PendingInstance {
      public:
        PendingInstance(void)
          : instance(NULL), op(NULL), uid(0), eager(false) { }
        PendingInstance(FutureInstance *i, ApUserEvent r)
          : instance(i), op(NULL), uid(0), inst_ready(r), eager(false) { }
        PendingInstance(Operation *o, UniqueID id, ApUserEvent r, 
                        RtUserEvent a, bool e)
          : instance(NULL), op(o), uid(id), inst_ready(r),
            alloc_ready(a), eager(e) { }
      public:
        FutureInstance *instance;
        Operation *op;
        UniqueID uid;
        ApUserEvent inst_ready;
        RtUserEvent alloc_ready;
        std::set<AddressSpaceID> remote_requests;
        bool eager;
      };
    public:
      // This constructor provides the complete size and effects event
      // at the point the future is constructed so they don't need to
      // be provided later with set_future_result_size 
      FutureImpl(TaskContext *ctx, Runtime *rt, bool register_future,
                 DistributedID did, Provenance *provenance,
                 Operation *op = NULL);
      // This constructor is for futures made by tasks or other operations
      // which do not know the size or effects for the operation until later
      FutureImpl(TaskContext *ctx, Runtime *rt, bool register_future, 
                 DistributedID did, Operation *op, GenerationID gen,
                 size_t op_ctx_index, const DomainPoint &op_point,
#ifdef LEGION_SPY
                 UniqueID op_uid,
#endif
                 int op_depth, Provenance *provenance,
                 CollectiveMapping *mapping = NULL);
      FutureImpl(const FutureImpl &rhs);
      virtual ~FutureImpl(void);
    public:
      FutureImpl& operator=(const FutureImpl &rhs);
    public:
      // Finalize the future before everything shuts down
      void prepare_for_shutdown(void);
      // Wait without subscribing to the payload
      void wait(bool silence_warnings, const char *warning_string);
      const void* get_buffer(Processor proc, Memory::Kind memory,
                             size_t *extent_in_bytes = NULL, 
                             bool check_extent = false,
                             bool silence_warnings = false, 
                             const char *warning_string = NULL);
      const void* get_buffer(Memory memory,
                             size_t *extent_in_bytes = NULL, 
                             bool check_extent = false,
                             bool silence_warnings = false, 
                             const char *warning_string = NULL);
      PhysicalInstance get_instance(Memory::Kind kind,
                             size_t extent_in_bytes, bool check_extent,
                             bool silence_warnings, const char *warning_string);
      void report_incompatible_accessor(const char *accessor_kind,
                                        PhysicalInstance instance);
      bool find_or_create_application_instance(Memory target, UniqueID uid);
      RtEvent request_application_instance(Memory target, SingleTask *task,
                       UniqueID uid, AddressSpaceID source,
                       ApUserEvent ready_event = ApUserEvent::NO_AP_USER_EVENT,
                       size_t upper_bound_size = SIZE_MAX);
      ApEvent find_application_instance_ready(Memory target, SingleTask *task);
      // The return event for this method indicates when the resources have
      // been allocated for the instance and we can consider it mapped
      RtEvent request_internal_buffer(Operation *op, bool eager);
      const void *find_internal_buffer(TaskContext *ctx, size_t &expected_size);
      FutureInstance* get_canonical_instance(void);
      ApEvent reduce_from_canonical(FutureInstance *target, AllReduceOp *op,
                          const ReductionOpID redop_id,
                          const ReductionOp *redop, bool exclusive,
                          ApEvent precondition = ApEvent::NO_AP_EVENT);
      bool is_empty(bool block, bool silence_warnings = true,
                    const char *warning_string = NULL,
                    bool internal = false);
      size_t get_untyped_size(void);
      const void *get_metadata(size_t *metasize);
      ApEvent get_ready_event(bool need_lock = true);
      // A special function for predicates to peek
      // at the boolean value of a future if it is set
      // Must have called request internal buffer first and event must trigger
      bool get_boolean_value(TaskContext *ctx);
    public:
      // This will simply save the value of the future
      void set_result(ApEvent complete, FutureInstance *instance, 
                      void *metadata = NULL, size_t metasize = 0);
      void set_results(ApEvent complete,
                      const std::vector<FutureInstance*> &instances,
                      void *metadata = NULL, size_t metasize = 0);
      void set_result(ApEvent complete, FutureFunctor *callback_functor,
                      bool own, Processor functor_proc);
      // This is the same as above but for data that we know is visible
      // in the system memory and should always make a local FutureInstance
      // and for which we know that there is no completion effects
      void set_local(const void *value, size_t size, bool own = false);
      // This will save the value of the future locally
      void unpack_result(Deserializer &derez);
      void unpack_instances(Deserializer &derez);
      // Reset the future in case we need to restart the
      // computation for resiliency reasons
      bool reset_future(void);
      // Request that we get meta data for the future on this node
      // The return event here will indicate when we have local data
      // that is valid to access for this particular future
      RtEvent subscribe(bool need_lock = true);
      size_t get_upper_bound_size(void);
      void get_future_coordinates(TaskTreeCoordinates &coordinates) const;
      void pack_future(Serializer &rez, AddressSpaceID target);
      static Future unpack_future(Runtime *runtime, 
          Deserializer &derez, Operation *op = NULL, GenerationID op_gen = 0,
#ifdef LEGION_SPY
          UniqueID op_uid = 0,
#endif
          int op_depth = 0);
    public:
      virtual void notify_local(void);
    public:
      void register_dependence(Operation *consumer_op);
      void register_remote(AddressSpaceID sid);
      void set_future_result_size(size_t size, AddressSpaceID source);
    protected:
      void finish_set_future(ApEvent complete); // must be holding lock
      void create_pending_instances(void); // must be holding lock
      FutureInstance* find_or_create_instance(Memory memory, Operation *op,
                        UniqueID op_uid, bool eager, bool need_lock = true,
                        ApUserEvent inst_ready = ApUserEvent::NO_AP_USER_EVENT,
                        FutureInstance *existing = NULL);
      void mark_sampled(void);
      void broadcast_result(void); // must be holding lock
      void record_subscription(AddressSpaceID subscriber, bool need_lock);
    protected:
      RtEvent invoke_callback(void); // must be holding lock
      void perform_callback(void);
      void perform_broadcast(void);
      // must be holding lock
      void pack_future_result(Serializer &rez) const;
    public:
      RtEvent record_future_registered(void);
      static void handle_future_result(Deserializer &derez, Runtime *rt);
      static void handle_future_result_size(Deserializer &derez,
                                  Runtime *runtime, AddressSpaceID source);
      static void handle_future_subscription(Deserializer &derez, Runtime *rt,
                                             AddressSpaceID source);
      static void handle_future_create_instance_request(Deserializer &derez,
                                                        Runtime *runtime);
      static void handle_future_create_instance_response(Deserializer &derez,
                                                         Runtime *runtime);
    public:
      void contribute_to_collective(const DynamicCollective &dc,unsigned count);
      static void handle_contribute_to_collective(const void *args);
      static void handle_callback(const void *args);
      static void handle_release(const void *args);
      static void handle_broadcast(const void *args);
    public:
      TaskContext *const context;
      // These three fields are only valid on the owner node
      Operation *const producer_op;
      const GenerationID op_gen;
      // The depth of the context in which this was made
      const int producer_depth;
#ifdef LEGION_SPY
      const UniqueID producer_uid;
#endif
      const size_t producer_context_index;
      const DomainPoint producer_point;
      Provenance *const provenance;
    private:
      mutable LocalLock future_lock;
      RtUserEvent subscription_event;
      AddressSpaceID result_set_space;
      // On the owner node, keep track of the registered waiters
      std::set<AddressSpaceID> subscribers;
      std::map<Memory,FutureInstance*> instances;
      FutureInstance *canonical_instance;
    private:
      void *metadata;
      size_t metasize;
    private:
      // The determined size of this future to this point
      // This is only an upper bound until it is solidifed
      size_t future_size;
      // This is the upper bound size prior to being refined
      // down to a precise size when the future is finally set
      size_t upper_bound_size; 
      // The event denoting when all the effects represented by
      // this future are actually complete
      ApEvent future_complete;
    private:
      // Instances that need to be made once canonical instance is set
      std::map<Memory,PendingInstance> pending_instances;
      // Requests to create instances on remote nodes
      // First event is the mapped event, second is ready event
      std::map<Memory,std::pair<RtUserEvent,ApUserEvent> > pending_requests;
    private:
      Processor callback_proc;
      FutureFunctor *callback_functor;
      bool own_callback_functor;
    private:
      // Whether this future has a size set yet
      bool future_size_set;
    private:
      std::atomic<bool> empty;
      std::atomic<bool> sampled;
    };

    /**
     * \class FutureInstance
     * A future instance represents the data for a single copy of
     * the future in a memory somewhere. It has a duality to it that
     * is likely confusing at first. It can either be an external 
     * allocation which may or may not have an external realm instance
     * associated with it. Or it could be a normal realm instance for
     * which we have extracted the pointer and size for it. Furthermore
     * when moving these from one node to another, sometimes we pass
     * them by-value if they can be cheaply copied, other times we
     * will move the just the references to the instances and allocations.
     * You'll have to look into the implementation to discover which
     * is happening, but when you get an unpacked copy on the remote
     * side it is a valid future instance that can you use regardless.
     * Each future instance has a concept of instance ownership which 
     * exists with exactly one copy of each future instance. If a future
     * instance is packed and moved to a remote node then it can only be
     * read from so we can track the appropriate read effects.
     * Current future instances are immutable after they are initially 
     * written, but are designed so that we might easily be able to relax
     * that later so we can support mutable future values.
     * Note that none of the methods in this class are thread safe so
     * atomicity needs to come from the caller.
     */
    class FutureInstance {
    public:
      struct FreeExternalArgs : public LgTaskArgs<FreeExternalArgs> {
      public:
        static const LgTaskID TASK_ID = LG_FREE_EXTERNAL_TASK_ID;
      public:
        FreeExternalArgs(const Realm::ExternalInstanceResource *r,
            void (*func)(const Realm::ExternalInstanceResource&),
            PhysicalInstance inst);
      public:
        const Realm::ExternalInstanceResource *const resource;
        void (*const freefunc)(const Realm::ExternalInstanceResource&);
        const PhysicalInstance instance;
      };
    public:
      FutureInstance(const void *data, size_t size,
                     ApEvent ready_event, Runtime *runtime, bool eager, 
                     bool external, bool own_allocation = true,
                     PhysicalInstance inst = PhysicalInstance::NO_INST,
                     Processor free_proc = Processor::NO_PROC,
                     RtEvent use_event = RtEvent::NO_RT_EVENT,
                     ApUserEvent remote_read = ApUserEvent::NO_AP_USER_EVENT);
      FutureInstance(const void *data, size_t size,
                     ApEvent ready_event, Runtime *runtime, bool own,
                     const Realm::ExternalInstanceResource *allocation,
                     void (*freefunc)(
                       const Realm::ExternalInstanceResource&) = NULL,
                     Processor free_proc = Processor::NO_PROC,
                     PhysicalInstance inst = PhysicalInstance::NO_INST,
                     RtEvent use_event = RtEvent::NO_RT_EVENT,
                     ApUserEvent remote_read = ApUserEvent::NO_AP_USER_EVENT);
      FutureInstance(const FutureInstance &rhs) = delete;
      ~FutureInstance(void);
    public:
      FutureInstance& operator=(const FutureInstance &rhs) = delete;
    public:
      ApEvent initialize(const ReductionOp *redop, Operation *op);
      ApEvent copy_from(FutureInstance *source, Operation *op,
                        ApEvent precondition = ApEvent::NO_AP_EVENT,
                        bool check_source_ready = true);
      ApEvent reduce_from(FutureInstance *source, Operation *op,
                          const ReductionOpID redop_id,
                          const ReductionOp *redop, bool exclusive,
                          ApEvent precondition = ApEvent::NO_AP_EVENT);
      void record_read_event(ApEvent read_event);
    public:
      // This method can be called concurrently from different threads
      const void* get_data(void);
      bool is_ready(bool check_ready_event = true) const;
      ApEvent get_ready(bool check_ready_event = true) const;
      ApEvent collapse_reads(void);
      // This method will return an instance that represents the
      // data for this future instance of a given size, if the needed size
      // does not match the base size then a fresh instance will be returned
      // which will be the responsibility of the caller to destroy
      PhysicalInstance get_instance(size_t needed_size, bool &own_inst);
    public:
      bool can_pack_by_value(void) const;
      bool pack_instance(Serializer &rez, bool pack_ownership, 
                         bool other_ready = false,
                         ApEvent ready = ApEvent::NO_AP_EVENT);
      static FutureInstance* unpack_instance(Deserializer &derez, Runtime *rt);
    public:
      static ApEvent init_ready(ApEvent r, Runtime *rt, PhysicalInstance inst);
      static bool check_meta_visible(Runtime *runtime, Memory memory,
                                     bool has_freefunc = false);
      static FutureInstance* create_local(const void *value, size_t size, 
                                          bool own, Runtime *runtime);
      static void handle_free_external(Deserializer &derez, Runtime *runtime);
      static void handle_free_external(const void *args);
      static void free_host_memory(const Realm::ExternalInstanceResource &mem);
    public:
      Runtime *const runtime;
    public:
      const size_t size;
      const Memory memory;
      const ApEvent ready_event;
      const Realm::ExternalInstanceResource *const resource;
      void (*const freefunc)(const Realm::ExternalInstanceResource&);
      const Processor freeproc;
      const bool eager_allocation;
      const bool external_allocation;
      const bool is_meta_visible;
    protected:
      bool own_allocation;
      std::atomic<const void*> data;
      // This instance always has a domain of [0,0] and a field
      // size == `size` for the future instance
      PhysicalInstance instance;
      // Event for when it is safe to use the instance
      RtEvent use_event;
      // Events for operations reading from this instance
      std::vector<ApEvent> read_events;
      // If we don't own our instance then we have an event to trigger
      // when all our read events are done
      ApUserEvent remote_reads_done;
      // Whether we own this instance
      // Note if we own the allocation then we must own the instance as well
      // We can own the instance without owning the allocation in the case
      // of external allocations that we don't own but make an instance later
      bool own_instance;
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
      FutureMapImpl(TaskContext *ctx, Operation *op, IndexSpaceNode *domain,
                    Runtime *rt, DistributedID did, Provenance *provenance,
                    bool register_now = true, 
                    CollectiveMapping *mapping = NULL);
      FutureMapImpl(TaskContext *ctx, Runtime *rt, IndexSpaceNode *domain,
                    DistributedID did, size_t index,
                    ApEvent completion, Provenance *provenance,
                    bool register_now = true, 
                    CollectiveMapping *mapping = NULL); // remote
      FutureMapImpl(TaskContext *ctx, Operation *op, size_t index,
                    GenerationID gen, int depth, 
#ifdef LEGION_SPY
                    UniqueID uid,
#endif
                    IndexSpaceNode *domain, Runtime *rt, DistributedID did,
                    ApEvent completion, Provenance *provenance);
      FutureMapImpl(const FutureMapImpl &rhs);
      virtual ~FutureMapImpl(void);
    public:
      FutureMapImpl& operator=(const FutureMapImpl &rhs);
    public:
      virtual bool is_replicate_future_map(void) const { return false; }
    public:
      virtual void notify_local(void);
    public:
      Domain get_domain(void) const;
      virtual Future get_future(const DomainPoint &point, 
                                bool internal_only,
                                RtEvent *wait_on = NULL); 
      void set_future(const DomainPoint &point, FutureImpl *impl);
      void get_void_result(const DomainPoint &point, 
                            bool silence_warnings = true,
                            const char *warning_string = NULL);
      virtual void wait_all_results(bool silence_warnings = true,
                                    const char *warning_string = NULL);
      bool reset_all_futures(void);
    public:
      void pack_future_map(Serializer &rez, AddressSpaceID target);
      static FutureMap unpack_future_map(Runtime *runtime,
          Deserializer &derez, TaskContext *ctx);
    public:
      virtual void get_all_futures(std::map<DomainPoint,FutureImpl*> &futures);
      void set_all_futures(const std::map<DomainPoint,Future> &futures);
    public:
      // Will return NULL if it does not exist
      virtual FutureImpl* find_shard_local_future(ShardID shard,
                                                  const DomainPoint &point);
      virtual void get_shard_local_futures(ShardID shard,
                                    std::map<DomainPoint,FutureImpl*> &futures);
    public:
      void register_dependence(Operation *consumer_op);
      void process_future_response(Deserializer &derez);
    public:
      RtEvent record_future_map_registered(void);
      static void handle_future_map_future_request(Deserializer &derez,
                              Runtime *runtime, AddressSpaceID source);
      static void handle_future_map_future_response(Deserializer &derez,
                                                    Runtime *runtime);
    public:
      TaskContext *const context;
      // Either an index space task or a must epoch op
      Operation *const op;
      const size_t op_ctx_index;
      const GenerationID op_gen;
      const int op_depth;
#ifdef LEGION_SPY
      const UniqueID op_uid;
#endif
      Provenance *const provenance;
      IndexSpaceNode *const future_map_domain;
      const ApEvent completion_event;
    protected:
      mutable LocalLock future_map_lock;
      std::map<DomainPoint,FutureImpl*> futures;
    };

    /**
     * \class TransformFutureMapImpl
     * This class is a wrapper around a future map implementation that
     * will transform the points being accessed on to a previous future map
     */
    class TransformFutureMapImpl : public FutureMapImpl {
    public:
      typedef DomainPoint (*PointTransformFnptr)(const DomainPoint& point,
                                                 const Domain &domain,
                                                 const Domain &range);
      TransformFutureMapImpl(FutureMapImpl *previous, IndexSpaceNode *domain,
                             PointTransformFnptr fnptr, Provenance *provenance);
      TransformFutureMapImpl(FutureMapImpl *previous, IndexSpaceNode *domain,
                             PointTransformFunctor *functor, bool own_functor,
                             Provenance *provenance);
      TransformFutureMapImpl(const TransformFutureMapImpl &rhs);
      virtual ~TransformFutureMapImpl(void);
    public:
      TransformFutureMapImpl& operator=(const TransformFutureMapImpl &rhs);
    public:
      virtual bool is_replicate_future_map(void) const;
      virtual Future get_future(const DomainPoint &point, 
                                bool internal_only,
                                RtEvent *wait_on = NULL);
      virtual void get_all_futures(std::map<DomainPoint,FutureImpl*> &futures);
      virtual void wait_all_results(bool silence_warnings = true,
                                    const char *warning_string = NULL);
    public:
      // Will return NULL if it does not exist
      virtual FutureImpl* find_shard_local_future(ShardID shard,
                                                  const DomainPoint &point);
      virtual void get_shard_local_futures(ShardID shard,
                                    std::map<DomainPoint,FutureImpl*> &futures);
    public:
      FutureMapImpl *const previous;
      const bool own_functor;
      const bool is_functor;
    protected:
      union {
        PointTransformFnptr fnptr;
        PointTransformFunctor *functor; 
      } transform;
    };

    /**
     * \class ReplFutureMapImpl
     * This a special kind of future map that is created
     * in control replication contexts
     */
    class ReplFutureMapImpl : public FutureMapImpl {
    public:
      ReplFutureMapImpl(TaskContext *ctx, ShardManager *man, Operation *op,
                        IndexSpaceNode *domain, IndexSpaceNode *shard_domain,
                        Runtime *rt, DistributedID did, Provenance *provenance,
                        CollectiveMapping *collective_mapping);
      ReplFutureMapImpl(TaskContext *ctx, ShardManager *man, Runtime *rt,
                        IndexSpaceNode *domain, IndexSpaceNode *shard_domain,
                        DistributedID did, size_t index,
                        ApEvent completion, Provenance *provenance,
                        CollectiveMapping *collective_mapping);
      ReplFutureMapImpl(const ReplFutureMapImpl &rhs) = delete;
      virtual ~ReplFutureMapImpl(void);
    public:
      ReplFutureMapImpl& operator=(const ReplFutureMapImpl &rhs) = delete;
    public:
      virtual bool is_replicate_future_map(void) const { return true; }
    public:
      virtual Future get_future(const DomainPoint &point,
                                bool internal, RtEvent *wait_on = NULL);
      virtual void get_all_futures(std::map<DomainPoint,FutureImpl*> &futures);
      virtual void wait_all_results(bool silence_warnings = true,
                                    const char *warning_string = NULL);
    public:
      // Will return NULL if it does not exist
      virtual FutureImpl* find_shard_local_future(ShardID shard,
                                                  const DomainPoint &point);
      virtual void get_shard_local_futures(ShardID shard,
                                    std::map<DomainPoint,FutureImpl*> &futures);
    public:
      bool set_sharding_function(ShardingFunction *function, bool own = false);
      RtEvent get_sharding_function_ready(void);
    public:
      ShardManager *const shard_manager;
      IndexSpaceNode *const shard_domain;
      // Unlike normal future maps, we know these only ever exist on the
      // node where they are made so we store their producer op information
      // in case they have to make futures from remote shards
      const int op_depth; 
      const UniqueID op_uid;
    protected:
      RtUserEvent sharding_function_ready;
      std::atomic<ShardingFunction*> sharding_function;
      // Whether the future map owns the sharding function
      bool own_sharding_function;
      bool collective_performed;
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
      PhysicalRegionImpl(const RegionRequirement &req, RtEvent mapped_event,
            ApEvent ready_event, ApUserEvent term_event, bool mapped, 
            TaskContext *ctx, MapperID mid, MappingTagID tag, bool leaf, 
            bool virt, bool collective, Runtime *rt);
      PhysicalRegionImpl(const PhysicalRegionImpl &rhs) = delete;
      ~PhysicalRegionImpl(void);
    public:
      PhysicalRegionImpl& operator=(const PhysicalRegionImpl &rhs) = delete;
    public:
      inline bool created_accessor(void) const { return made_accessor; }
    public:
      void wait_until_valid(bool silence_warnings, const char *warning_string, 
                            bool warn = false, const char *src = NULL);
      bool is_valid(void) const;
      bool is_mapped(void) const;
      LogicalRegion get_logical_region(void) const;
      PrivilegeMode get_privilege(void) const;
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(bool silence_warnings = true);
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic> 
          get_field_accessor(FieldID field, bool silence_warnings = true);
    public:
      void unmap_region(void);
      ApEvent remap_region(ApEvent new_ready_event);
      const RegionRequirement& get_requirement(void) const;
      void add_padded_field(FieldID fid);
      void set_reference(const InstanceRef &references, bool safe = false);
      void set_references(const InstanceSet &instances, bool safe = false);
      bool has_references(void) const;
      void get_references(InstanceSet &instances) const;
      void get_memories(std::set<Memory>& memories, 
          bool silence_warnings, const char *warning_string) const;
      void get_fields(std::vector<FieldID>& fields) const;
#if defined(LEGION_PRIVILEGE_CHECKS) || defined(LEGION_BOUNDS_CHECKS)
    public:
      const char* get_task_name(void) const;
#endif
#ifdef LEGION_BOUNDS_CHECKS
    public:
      bool contains_ptr(ptr_t ptr);
      bool contains_point(const DomainPoint &dp);
#endif
    public:
      void get_bounds(void *realm_is, TypeTag type_tag);
      PieceIteratorImpl* get_piece_iterator(FieldID fid, bool privilege_only,
                          bool silence_warnings, const char *warning_string);
      PhysicalInstance get_instance_info(PrivilegeMode mode, 
                                         FieldID fid, size_t field_size, 
                                         void *realm_is, TypeTag type_tag,
                                         const char *warning_string,
                                         bool silence_warnings, 
                                         bool generic_accessor,
                                         bool check_field_size,
                                         ReductionOpID redop);
      PhysicalInstance get_padding_info(FieldID fid, size_t field_size,
                                        Domain *inner, Domain &outer,
                                        const char *warning_string,
                                        bool silence_warnings,
                                        bool generic_accessor,
                                        bool check_field_size);
      void report_incompatible_accessor(const char *accessor_kind,
                             PhysicalInstance instance, FieldID fid);
      void report_incompatible_multi_accessor(unsigned index, FieldID fid,
                           PhysicalInstance inst1, PhysicalInstance inst2);
      void report_colocation_violation(const char *accessor_kind,
                           FieldID fid, PhysicalInstance inst1,
                           PhysicalInstance ins2, const PhysicalRegion &other,
                           bool reduction);
      static void empty_colocation_regions(const char *accessor_kind,
                                           FieldID fid, bool reduction);
      static void fail_bounds_check(DomainPoint p, FieldID fid, 
                                    PrivilegeMode mode, bool multi);
      static void fail_bounds_check(Domain d, FieldID fid, 
                                    PrivilegeMode mode, bool multi);
      static void fail_privilege_check(DomainPoint p, FieldID fid, 
                                    PrivilegeMode mode);
      static void fail_privilege_check(Domain d, FieldID fid, 
                                    PrivilegeMode mode);
      static void fail_padding_check(DomainPoint d, FieldID fid);
    public:
      Runtime *const runtime;
      TaskContext *const context;
      const MapperID map_id;
      const MappingTagID tag;
      const bool leaf_region;
      const bool virtual_mapped;
      // Whether this physical region represents a collectively
      // created group of instances or not (e.g. ReplAttachOp)
      const bool collective;
      const bool replaying;
    private:
      const RegionRequirement req;
      // Event for when the 'references' are set by the producer op
      // can only be accessed in "application" side code
      // There should only be one of these triggered by the producer
      const RtEvent mapped_event;
      // Event for when it is safe to use the physical instances
      // can only be accessed in "application" side code
      // triggered by mapping stage code
      ApEvent ready_event;
      // Event for when the mapped application code is done accessing
      // the physical region, set in "application" side code 
      // should only be accessed there as well
      ApUserEvent termination_event;
      // Physical instances for this mapping
      // written by the "mapping stage" code of whatever operation made this
      // can be accessed in "application" side code after 'mapped' triggers
      InstanceSet references;
      // Any fields which we have privileges on the padded space (sorted)
      // This enables us to access the padded space for this field
      std::vector<FieldID> padded_fields;
      // "appliciation side" state
      // whether it is currently mapped
      bool mapped; 
      // whether it is currently valid -> mapped and ready_event has triggered
      bool valid; 
      bool made_accessor;
#ifdef LEGION_BOUNDS_CHECKS
    private:
      Domain bounds;
#endif
    };

    /**
     * \class OutputRegionImpl
     * The base implementation of an output region object.
     *
     * Just like physical region impls, we don't need to make
     * output region impls thread safe, because they are accessed
     * exclusively by a single task.
     */
    class OutputRegionImpl : public Collectable,
                             public LegionHeapify<OutputRegionImpl> {
    public:
      static const AllocationType alloc_type = OUTPUT_REGION_ALLOC;
    private:
      struct LayoutCreator {
      public:
        LayoutCreator(Realm::InstanceLayoutGeneric* &l,
                      const Domain & d,
                      const Realm::InstanceLayoutConstraints &c,
                      const std::vector<int32_t> &d_order)
          : layout(l), domain(d), constraints(c), dim_order(d_order)
        { }
        template<typename DIM, typename COLOR_T>
        static inline void demux(LayoutCreator *creator)
        {
#ifdef DEBUG_LEGION
          assert(creator->dim_order.size() == DIM::N);
#endif
          const DomainT<DIM::N, COLOR_T> bounds =
            Rect<DIM::N, COLOR_T>(creator->domain);
          creator->layout =
            Realm::InstanceLayoutGeneric::choose_instance_layout(
                bounds, creator->constraints, creator->dim_order.data());
        }
      private:
        Realm::InstanceLayoutGeneric* &layout;
        const Domain &domain;
        const Realm::InstanceLayoutConstraints &constraints;
        const std::vector<int32_t> &dim_order;
      };
    public:
      OutputRegionImpl(unsigned index,
                       const OutputRequirement &req,
                       InstanceSet instance_set,
                       TaskContext *ctx,
                       Runtime *rt,
                       const bool global_indexing,
                       const bool valid);
      OutputRegionImpl(const OutputRegionImpl &rhs);
      ~OutputRegionImpl(void);
    public:
      OutputRegionImpl& operator=(const OutputRegionImpl &rhs);
    public:
      Memory target_memory(void) const;
    public:
      LogicalRegion get_logical_region(void) const;
      bool is_valid_output_region(void) const;
    public:
      void check_type_tag(TypeTag type_tag) const;
      void check_field_size(FieldID field_id, size_t field_size) const;
      void get_layout(FieldID field_id,
                      std::vector<DimensionKind> &ordering,
                      size_t &alignment) const;
      size_t get_field_size(FieldID field_id) const;
    public:
      void return_data(const DomainPoint &extents,
                       FieldID field_id,
                       PhysicalInstance instance,
                       const LayoutConstraintSet *constraints,
                       bool check_constraints);
    private:
      void return_data(const DomainPoint &extents,
                       FieldID field_id,
                       uintptr_t ptr,
                       size_t alignment);
    private:
      struct FinalizeOutputArgs : public LgTaskArgs<FinalizeOutputArgs> {
      public:
        static const LgTaskID TASK_ID = LG_FINALIZE_OUTPUT_ID;
      public:
        FinalizeOutputArgs(OutputRegionImpl *r)
          : LgTaskArgs<FinalizeOutputArgs>(implicit_provenance),
            region(r) { }
        OutputRegionImpl *region;
      };
    public:
      void finalize(bool defer = true);
    public:
      static void handle_finalize_output(const void *args);
    public:
      bool is_complete(FieldID &unbound_field) const;
    public:
      const OutputRequirement &get_requirement(void) const { return req; }
      DomainPoint get_extents(void) const { return extents; }
    protected:
      PhysicalManager *get_manager(FieldID field_id) const;
    public:
      Runtime *const runtime;
      TaskContext *const context;
    private:
      struct ReturnedInstanceInfo {
        uintptr_t ptr;
        size_t alignment;
      };
    private:
      OutputRequirement req;
      InstanceSet instance_set;
      // Output data batched during task execution
      std::map<FieldID,ReturnedInstanceInfo> returned_instances;
      std::map<FieldID,size_t> field_sizes;
      std::map<FieldID,PhysicalManager*> managers;
      std::vector<PhysicalInstance> escaped_instances;
      DomainPoint extents;
      const unsigned index;
      const bool created_region;
      const bool global_indexing;
    };

    /**
     * \class ExternalResourcesImpl
     * This class provides the backing data structure for a collection of
     * physical regions that represent external data that have been attached
     * to logical regions in the same region tree
     */
    class ExternalResourcesImpl : public Collectable,
                                  public LegionHeapify<ExternalResourcesImpl> {
    public:
      static const AllocationType alloc_type = EXTERNAL_RESOURCES_ALLOC;
    public:
      ExternalResourcesImpl(InnerContext *context, size_t num_regions,
                            RegionTreeNode *upper, IndexSpaceNode *launch,
                            LogicalRegion parent,
                            const std::set<FieldID> &privilege_fields);
      ExternalResourcesImpl(const ExternalResourcesImpl &rhs);
      ~ExternalResourcesImpl(void);
    public:
      ExternalResourcesImpl& operator=(const ExternalResourcesImpl &rhs);
    public:
      size_t size(void) const;
      void set_region(unsigned index, PhysicalRegionImpl *region);
      PhysicalRegion get_region(unsigned index) const;
      void set_projection(ProjectionID pid);
      inline ProjectionID get_projection(void) const { return pid; }
      Future detach(InnerContext *context, IndexDetachOp *op, 
                    const bool flush, const bool unordered,
                    Provenance *provenance);
    public:
      InnerContext *const context;
      // Save these for when we go to do the detach
      RegionTreeNode *const upper_bound;
      IndexSpaceNode *const launch_bounds;
      const std::vector<FieldID> privilege_fields;
      const LogicalRegion parent;
    protected:
      std::vector<PhysicalRegion> regions;
      ProjectionID pid;
      bool detached;
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
      bool participating;
    public:
      std::map<int,AddressSpace> forward_mapping;
      std::map<AddressSpace,int> reverse_mapping;
    protected:
      mutable LocalLock reservation;
      RtUserEvent done_event;
      std::vector<int> stage_notifications;
      std::vector<bool> sent_stages;
    protected:
      int collective_radix;
      int collective_log_radix;
      int collective_stages;
      int collective_participating_spaces;
      int collective_last_radix;
      // Handle a small race on deciding who gets to
      // trigger the done event
      bool done_triggered;
    }; 

    /**
     * \class ImplicitShardManager
     * This is a class for helping to construct implicitly 
     * control replicated top-level tasks from external threads.
     * It helps to setup tasks just as though they had been 
     * control replicated, except everything was already control
     * replicated remotely.
     */
    class ImplicitShardManager : public Collectable {
    public:
      ImplicitShardManager(Runtime *rt, TaskID tid, MapperID mid, 
           Processor::Kind k, unsigned shards_per_address_space);
      ImplicitShardManager(const ImplicitShardManager &rhs) = delete;
      ~ImplicitShardManager(void);
    public:
      ImplicitShardManager& operator=(const ImplicitShardManager &rhs) = delete;
    public:
      bool record_arrival(bool local);
      ShardTask* create_shard(int shard_id, const DomainPoint &shard_point,
                              Processor proxy, const char *task_name);
    protected:
      void create_shard_manager(void);
      void request_shard_manager(void);
    public:
      void process_implicit_request(Deserializer &derez, AddressSpaceID source);
      RtUserEvent process_implicit_response(ShardManager *manager,
                                            InnerContext *context);
    public:
      static void handle_remote_request(Deserializer &derez, Runtime *runtime, 
                                        AddressSpaceID remote_space);
      static void handle_remote_response(Deserializer &derez, Runtime *runtime);
    public:
      Runtime *const runtime;
      const TaskID task_id;
      const MapperID mapper_id;
      const Processor::Kind kind;
      const unsigned shards_per_address_space;
    protected:
      mutable LocalLock manager_lock;
      unsigned remaining_create_arrivals;
      unsigned expected_local_arrivals;
      unsigned expected_remote_arrivals;
      unsigned local_shard_id;
      InnerContext *top_context;
      ShardManager *shard_manager;
      RtUserEvent manager_ready;
      Processor local_proxy;
      const char *local_task_name;
      std::map<DomainPoint,ShardID> shard_points;
      std::vector<std::pair<AddressSpaceID,void*> > remote_spaces;
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
    public:
      inline bool is_visible_memory(Memory memory) const
        { return (visible_memories.find(memory) != visible_memories.end()); }
      void find_visible_memories(std::set<Memory> &visible) const;
      Memory find_best_visible_memory(Memory::Kind kind) const;
    public:
      ApEvent find_concurrent_fence_event(ApEvent next);
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
      std::map<Memory,size_t/*bandwidth affinity*/> visible_memories;
    protected:
      // Keep track of the termination event for the previous 
      // concurrently executed task on this processor
      ApEvent previous_concurrent_execution;
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
        FIND_MANY_CONSTRAINTS,
        FIND_MANY_LAYOUT,
      };
    public:
      struct FreeEagerInstanceArgs : public LgTaskArgs<FreeEagerInstanceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_FREE_EAGER_INSTANCE_TASK_ID;
      public:
        FreeEagerInstanceArgs(MemoryManager *m, PhysicalInstance i)
          : LgTaskArgs<FreeEagerInstanceArgs>(implicit_provenance),
            manager(m), inst(i) { }
      public:
        MemoryManager *const manager;
        const PhysicalInstance inst;
      };
      class FutureInstanceAllocator : public ProfilingResponseHandler {
      public:
        FutureInstanceAllocator(void);
      public:
        virtual void handle_profiling_response(
                const ProfilingResponseBase *base,
                const Realm::ProfilingResponse &response,
                const void *orig, size_t orig_length);
        inline bool succeeded(void) const
        {
          if (!ready.has_triggered())
            ready.wait();
          return success.load();
        }
      private:
        const RtUserEvent ready;
        std::atomic<bool> success;
      };
#ifdef LEGION_MALLOC_INSTANCES
    public:
      struct MallocInstanceArgs : public LgTaskArgs<MallocInstanceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_MALLOC_INSTANCE_TASK_ID;
      public:
        MallocInstanceArgs(MemoryManager *m, Realm::InstanceLayoutGeneric *l, 
                     const Realm::ProfilingRequestSet *r, PhysicalInstance *i)
          : LgTaskArgs<MallocInstanceArgs>(implicit_provenance), 
            manager(m), layout(l), requests(r), instance(i) { }
      public:
        MemoryManager *const manager;
        Realm::InstanceLayoutGeneric *const layout;
        const Realm::ProfilingRequestSet *const requests;
        PhysicalInstance *const instance;
      };
      struct FreeInstanceArgs : public LgTaskArgs<FreeInstanceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_FREE_INSTANCE_TASK_ID;
      public:
        FreeInstanceArgs(MemoryManager *m, PhysicalInstance i)
          : LgTaskArgs<FreeInstanceArgs>(implicit_provenance), 
            manager(m), instance(i) { }
      public:
        MemoryManager *const manager;
        const PhysicalInstance instance;
      };
#endif
    public:
      MemoryManager(Memory mem, Runtime *rt);
      MemoryManager(const MemoryManager &rhs) = delete;
      ~MemoryManager(void);
    public:
      MemoryManager& operator=(const MemoryManager &rhs) = delete;
    public:
#if defined(LEGION_USE_CUDA) || defined(LEGION_USE_HIP)
      inline Processor get_local_gpu(void) const { return local_gpu; }
#endif
      static inline bool is_owner_memory(Memory m, AddressSpace space)
        {
          if (m.address_space() == space)
            return true;
          const Memory::Kind kind = m.kind();
          // File system memories are "local" everywhere
          return ((kind == Memory::HDF_MEM) || (kind == Memory::FILE_MEM));
        }
    public:
      void find_shutdown_preconditions(std::set<ApEvent> &preconditions);
      void prepare_for_shutdown(void);
      void finalize(void);
    public:
      void register_remote_instance(PhysicalManager *manager);
      void unregister_remote_instance(PhysicalManager *manager);
      void unregister_deleted_instance(PhysicalManager *manager);
    public:
      bool create_physical_instance(const LayoutConstraintSet &contraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result,
                                    Processor processor, bool acquire, 
                                    GCPriority priority, bool tight_bounds,
                                    LayoutConstraintKind *unsat_kind, 
                                    unsigned *unsat_index, size_t *footprint, 
                                    UniqueID creator_id, bool remote = false);
      bool create_physical_instance(LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result,
                                    Processor processor, bool acquire, 
                                    GCPriority priority, bool tight_bounds,
                                    LayoutConstraintKind *unsat_kind,
                                    unsigned *unsat_index, size_t *footprint, 
                                    UniqueID creator_id, bool remote = false);
      bool find_or_create_physical_instance(
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    Processor processor,
                                    bool acquire, GCPriority priority, 
                                    bool tight_region_bounds, 
                                    LayoutConstraintKind *unsat_kind, 
                                    unsigned *unsat_index, size_t *footprint, 
                                    UniqueID creator_id, bool remote = false);
      bool find_or_create_physical_instance(
                                    LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    Processor processor,
                                    bool acquire, GCPriority priority, 
                                    bool tight_region_bounds, 
                                    LayoutConstraintKind *unsat_kind,
                                    unsigned *unsat_index, size_t *footprint, 
                                    UniqueID creator_id, bool remote = false);
      bool find_physical_instance(  const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_bounds, bool remote = false);
      bool find_physical_instance(  LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_bounds, bool remote = false);
      void find_physical_instances( const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    std::vector<MappingInstance> &results, 
                                    bool acquire, bool tight_bounds, 
                                    bool remote = false);
      void find_physical_instances( LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    std::vector<MappingInstance> &results, 
                                    bool acquire, bool tight_bounds, 
                                    bool remote = false);
      void release_tree_instances(RegionTreeID tid);
      void set_garbage_collection_priority(PhysicalManager *manager,
                                    GCPriority priority);
      void record_created_instance( PhysicalManager *manager, bool acquire,
                                    GCPriority priority);
      FutureInstance* create_future_instance(Operation *op, UniqueID creator_id,
                                  ApEvent ready_event, size_t size, bool eager);
      void free_future_instance(PhysicalInstance inst, size_t size, 
                                RtEvent free_event, bool eager);
    public:
      void process_instance_request(Deserializer &derez, AddressSpaceID source);
      void process_instance_response(Deserializer &derez,AddressSpaceID source);
    protected:
      bool find_satisfying_instance(const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire, 
                                    bool tight_region_bounds, bool remote);
      void find_satisfying_instances(const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    std::vector<MappingInstance> &results, 
                                    bool acquire, bool tight_region_bounds, 
                                    bool remote);
      bool find_valid_instance(     const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire, 
                                    bool tight_region_bounds, bool remote);
      void release_candidate_references(const std::deque<PhysicalManager*>
                                                        &candidates) const;
    public:
      PhysicalManager* create_unbound_instance(LogicalRegion region,
                                               LayoutConstraintSet &constraints,
                                               ApEvent ready_event,
                                               MapperID mapper_id,
                                               Processor target_proc,
                                               GCPriority priority);
      void check_instance_deletions(const std::vector<PhysicalManager*> &del);
    protected:
      // We serialize all allocation attempts in a memory in order to 
      // ensure find_and_create calls will remain atomic
      RtEvent acquire_allocation_privilege(void);
      void release_allocation_privilege(void);
      PhysicalManager* allocate_physical_instance(InstanceBuilder &builder,
                                          size_t *footprint,
                                          LayoutConstraintKind *unsat_kind,
                                          unsigned *unsat_index);
    public:
      void remove_collectable(GCPriority priority, PhysicalManager *manager);
    public:
      RtEvent attach_external_instance(PhysicalManager *manager);
      void detach_external_instance(PhysicalManager *manager);
    public:
      bool is_visible_memory(Memory other);
    public:
      RtEvent create_eager_instance(PhysicalInstance &instance, LgEvent unique,
                                    Realm::InstanceLayoutGeneric *layout);
      // Create an external instance that is a view to the eager pool instance
      RtEvent create_sub_eager_instance(PhysicalInstance &instance,
                                        uintptr_t ptr, size_t size,
                                        Realm::InstanceLayoutGeneric *layout,
                                        LgEvent unique_event);
      void free_eager_instance(PhysicalInstance instance, RtEvent defer);
      static void handle_free_eager_instance(const void *args);
    public:
      void free_external_allocation(uintptr_t ptr, size_t size);
#ifdef LEGION_MALLOC_INSTANCES
    public:
      RtEvent allocate_legion_instance(Realm::InstanceLayoutGeneric *layout,
                                     const Realm::ProfilingRequestSet &requests,
                                     PhysicalInstance &inst,
                                     bool needs_defer = true);
      void record_legion_instance(InstanceManager *manager, 
                                  PhysicalInstance instance);
      void free_legion_instance(InstanceManager *manager, RtEvent deferred);
      void free_legion_instance(RtEvent deferred, PhysicalInstance inst,
                                bool needs_defer = true);
      static void handle_malloc_instance(const void *args);
      static void handle_free_instance(const void *args);
#endif
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
    public:
      // Realm instance backin the eager pool
      // Must be allocated at the start-up time
      PhysicalInstance eager_pool_instance;
      uintptr_t eager_pool;
      // Allocator object for eager allocations
      typedef BasicRangeAllocator<size_t, size_t> EagerAllocator;
      EagerAllocator *eager_allocator;
      size_t eager_remaining_capacity;
      // Allocation counter
      std::atomic<size_t> next_allocation_id;
      // Mapping from pointers to their allocation ids
      std::map<uintptr_t,size_t> eager_allocations;
    protected:
      // Lock for controlling access to the data
      // structures in this memory manager
      mutable LocalLock manager_lock;
      // Lock for ordering garbage collection
      // This lock should always be taken before the manager lock
      mutable LocalLock collection_lock;
      // We maintain several sets of instances here
      // This is a generic list that tracks all the allocated instances
      // For collectable instances they have non-NULL GCHole that 
      // represents a range of memory that can be collected
      // This data structure is protected by the manager_lock
      typedef LegionMap<PhysicalManager*,GCPriority,
                        MEMORY_INSTANCES_ALLOC> TreeInstances;
      std::map<RegionTreeID,TreeInstances> current_instances;
      // Keep track of all groupings of instances based on their 
      // garbage collection priorities and placement in memory
      std::map<GCPriority,std::set<PhysicalManager*>,
               std::greater<GCPriority> > collectable_instances;
      // Keep track of outstanding requuests for allocations which
      // will be tried in the order that they arrive
      std::deque<RtUserEvent> pending_allocation_attempts;
    protected:
      std::set<Memory> visible_memories;
    protected:
#ifdef LEGION_MALLOC_INSTANCES
      std::map<InstanceManager*,PhysicalInstance> legion_instances;
      std::map<PhysicalInstance,size_t> allocations;
      std::map<RtEvent,PhysicalInstance> pending_collectables;
#endif
#if defined(LEGION_USE_CUDA) || defined(LEGION_USE_HIP)
      Processor local_gpu;
#endif
    protected:
      class GarbageCollector {
      public:
        GarbageCollector(LocalLock &collection_lock, LocalLock &manager_lock,
                         AddressSpaceID local, Memory memory, size_t needed,
                         std::map<GCPriority,std::set<PhysicalManager*>,
                                 std::greater<GCPriority> > &collectables);
        GarbageCollector(const GarbageCollector &rhs) = delete;
        ~GarbageCollector(void);
      public:
        GarbageCollector& operator=(const GarbageCollector &rhs) = delete;
      public:
        RtEvent perform_collection(void);
        inline bool collection_complete(void) const 
          { return (current_priority == LEGION_GC_NEVER_PRIORITY); }
      protected:
        void sort_next_priority_holes(bool advance = true);
      protected:
        struct Range {
        public:
          Range(void) : size(0) { }
          Range(PhysicalManager *m); 
          std::vector<PhysicalManager*> managers;
          size_t size;
        };
      protected:
        // Note this makes sure there is only one collection at a time
        AutoLock collection_lock;
        LocalLock &manager_lock;
        std::map<GCPriority,std::set<PhysicalManager*>,
                 std::greater<GCPriority> > &collectable_instances;
        const Memory memory;
        const AddressSpaceID local_space;
        const size_t needed_size;
      protected:
        std::vector<PhysicalManager*> small_holes, perfect_holes;
        std::map<size_t,std::vector<PhysicalManager*> > large_holes;
        std::map<uintptr_t,Range> ranges;
        GCPriority current_priority;
      };
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
        size_t index;
        unsigned messages;
        unsigned total;
      };
    public:
      VirtualChannel(VirtualChannelKind kind,AddressSpaceID local_address_space,
               size_t max_message_size, bool profile, LegionProfiler *profiler);
      VirtualChannel(const VirtualChannel &rhs);
      ~VirtualChannel(void);
    public:
      VirtualChannel& operator=(const VirtualChannel &rhs);
    public:
      void package_message(Serializer &rez, MessageKind k, bool flush,
                           RtEvent flush_precondition,
                           Runtime *runtime, Processor target, 
                           bool response, bool shutdown);
      void process_message(const void *args, size_t arglen, 
                        Runtime *runtime, AddressSpaceID remote_address_space);
      void confirm_shutdown(ShutdownManager *shutdown_manager, bool phase_one);
    private:
      void send_message(bool complete, Runtime *runtime, Processor target, 
                        MessageKind kind, bool response, bool shutdown,
                        RtEvent send_precondition);
      bool handle_messages(unsigned num_messages, Runtime *runtime, 
                           AddressSpaceID remote_address_space,
                           const char *args, size_t arglen) const;
      static void buffer_messages(unsigned num_messages,
                                  const void *args, size_t arglen,
                                  char *&receiving_buffer,
                                  size_t &receiving_buffer_size,
                                  size_t &receiving_index,
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
      const bool profile_outgoing_messages;
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
      size_t receiving_index;
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
      template<MessageKind M>
      inline void send_message(Serializer &rez, bool flush, 
                        bool response = false, bool shutdown = false,
                        RtEvent flush_precondition = RtEvent::NO_RT_EVENT);
      void receive_message(const void *args, size_t arglen);
      void confirm_shutdown(ShutdownManager *shutdown_manager,
                            bool phase_one);
      // Maintain a static-mapping between message kinds and virtual channels
      static inline VirtualChannelKind find_message_vc(MessageKind kind);
    private:
      VirtualChannel *const channels;
    public:
      Runtime *const runtime;
      // State for sending messages
      const AddressSpaceID remote_address_space;
      const Processor target;
      const bool always_flush;
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
      bool handle_response(int code, bool success, 
                           const std::set<RtEvent> &to_add);
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
      int return_code;
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
      PendingVariantRegistration(VariantID vid, size_t return_type_size,
                                 bool has_return_type_size,
                                 const TaskVariantRegistrar &registrar,
                                 const void *user_data, size_t user_data_size,
                                 const CodeDescriptor &realm_desc, 
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
      size_t return_type_size;
      bool has_return_type_size;
      TaskVariantRegistrar registrar;
      void *user_data;
      size_t user_data_size;
      CodeDescriptor realm_desc; 
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
                  const TaskVariantRegistrar &registrar, 
                  size_t return_type_size, bool has_return_type_size,
                  const CodeDescriptor &realm_desc,
                  const void *user_data = NULL, size_t user_data_size = 0);
      VariantImpl(const VariantImpl &rhs) = delete;
      ~VariantImpl(void);
    public:
      VariantImpl& operator=(const VariantImpl &rhs) = delete;
    public:
      inline bool is_leaf(void) const { return leaf_variant; }
      inline bool is_inner(void) const { return inner_variant; }
      inline bool is_idempotent(void) const { return idempotent_variant; }
      inline bool is_replicable(void) const { return replicable_variant; }
      inline bool is_concurrent(void) const { return concurrent_variant; }
      inline const char* get_name(void) const { return variant_name; }
      inline const ExecutionConstraintSet&
        get_execution_constraints(void) const { return execution_constraints; }
      inline const TaskLayoutConstraintSet& 
        get_layout_constraints(void) const { return layout_constraints; } 
    public:
      bool is_no_access_region(unsigned idx) const;
    public:
      ApEvent dispatch_task(Processor target, SingleTask *task, 
          TaskContext *ctx, ApEvent precondition,
          int priority, Realm::ProfilingRequestSet &requests);
      void dispatch_inline(Processor current, TaskContext *ctx);
    public:
      bool can_use(Processor::Kind kind, bool warn) const;
    public:
      void broadcast_variant(RtUserEvent done, AddressSpaceID origin,
                             AddressSpaceID local);
      void find_padded_locks(SingleTask *task, 
                    const std::vector<RegionRequirement> &regions,
                    const std::deque<InstanceSet> &physical_instances) const;
      void record_padded_fields(const std::vector<RegionRequirement> &regions,
                    const std::vector<PhysicalRegion> &physical_regions) const;
    public:
      static void handle_variant_broadcast(Runtime *runtime, 
                                           Deserializer &derez);
      static bool check_padding(Runtime *runtime,
                                const TaskLayoutConstraintSet &constraints);
    public:
      const VariantID vid;
      TaskImpl *const owner;
      Runtime *const runtime;
      const bool global; // globally valid variant
      const bool needs_padding;
      const bool has_return_type_size;
      const size_t return_type_size;
    public:
      const CodeDescriptorID descriptor_id;
      CodeDescriptor realm_descriptor;
    public:
      const ExecutionConstraintSet execution_constraints;
      const TaskLayoutConstraintSet   layout_constraints;
    private:
      void *user_data;
      size_t user_data_size;
      ApEvent ready_event;
    private: // properties
      const bool leaf_variant;
      const bool inner_variant;
      const bool idempotent_variant;
      const bool replicable_variant;
      const bool concurrent_variant;
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
                        bool inter, DistributedID did = 0,
                        CollectiveMapping *collective_mapping = NULL);
      LayoutConstraints(LayoutConstraintID layout_id,
                        Runtime *runtime, const LayoutConstraintSet &cons,
                        FieldSpace handle, bool inter);
      LayoutConstraints(const LayoutConstraints &rhs);
      virtual ~LayoutConstraints(void);
    public:
      LayoutConstraints& operator=(const LayoutConstraints &rhs);
      bool operator==(const LayoutConstraints &rhs) const;
      bool operator==(const LayoutConstraintSet &rhs) const;
    public:
      virtual void notify_local(void);
    public:
      inline FieldSpace get_field_space(void) const { return handle; }
      inline const char* get_name(void) const { return constraints_name; }
    public:
      void send_constraint_response(AddressSpaceID source,
                                    RtUserEvent done_event);
      void update_constraints(Deserializer &derez);
    public:
      bool entails(LayoutConstraints *other_constraints, unsigned total_dims,
                   const LayoutConstraint **failed_constraint, 
                   bool test_pointer = true);
      bool entails(const LayoutConstraintSet &other, unsigned total_dims,
                   const LayoutConstraint **failed_constraint, 
                   bool test_pointer = true) const;
      bool conflicts(LayoutConstraints *other_constraints, unsigned total_dims,
                     const LayoutConstraint **conflict_constraint);
      bool conflicts(const LayoutConstraintSet &other, unsigned total_dims,
                     const LayoutConstraint **conflict_constraint) const;
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
      using ProjectionFunctor::project;
      virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                    LogicalRegion upper_bound,
                                    const DomainPoint &point);
      virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                    LogicalPartition upper_bound,
                                    const DomainPoint &point);
      virtual LogicalRegion project(LogicalRegion upper_bound,
                                    const DomainPoint &point,
                                    const Domain &launch_domain);
      virtual LogicalRegion project(LogicalPartition upper_bound,
                                    const DomainPoint &point,
                                    const Domain &launch_domain);
      virtual void invert(LogicalRegion region, LogicalRegion upper_bound,
                          const Domain &launch_domain,
                          std::vector<DomainPoint> &ordered_points);
      virtual void invert(LogicalRegion region, LogicalPartition upper_bound,
                          const Domain &launch_domain,
                          std::vector<DomainPoint> &ordered_points);
      virtual bool is_functional(void) const;
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
      virtual void record_intra_space_dependences(unsigned idx,
                               const std::vector<DomainPoint> &region_deps) = 0;
      virtual const Mappable* as_mappable(void) const = 0;
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
      void prepare_for_shutdown(void);
    public:
      // The old path explicitly for tasks
      LogicalRegion project_point(Task *task, unsigned idx, Runtime *runtime,
                       const Domain &launch_domain, const DomainPoint &point);
      void project_points(const RegionRequirement &req, unsigned idx,
                          Runtime *runtime, const Domain &launch_domain,
                          const std::vector<PointTask*> &point_tasks);
      // Generalized and annonymized
      void project_points(Operation *op, unsigned idx, 
                          const RegionRequirement &req, 
                          Runtime *runtime, const Domain &launch_domain,
                          const std::vector<ProjectionPoint*> &points);
      // Projection for refinements
      void project_refinement(IndexSpaceNode *domain, RegionTreeNode *node,
                              std::vector<RegionNode*> &regions) const;
    protected:
      // Old checking code explicitly for tasks
      void check_projection_region_result(LogicalRegion upper_bound,
                                          const Task *task, unsigned idx,
                                          LogicalRegion result, 
                                          Runtime *runtime) const;
      void check_projection_partition_result(LogicalPartition upper_bound,
                                             const Task *task, unsigned idx,
                                             LogicalRegion result,
                                             Runtime *runtime) const;
      // Annonymized checking code
      void check_projection_region_result(LogicalRegion upper_bound,
                                          Operation *op, unsigned idx,
                                          LogicalRegion result,
                                          Runtime *runtime) const;
      void check_projection_partition_result(LogicalPartition upper_bound,
                                          Operation *op, unsigned idx,
                                          LogicalRegion result,
                                          Runtime *runtime) const;
      // Checking for inversion
      void check_inversion(const Task *task, unsigned idx,
                           const std::vector<DomainPoint> &ordered_points);
      void check_containment(const Task *task, unsigned idx,
                             const std::vector<DomainPoint> &ordered_points);
      void check_inversion(const Mappable *mappable, unsigned idx,
                           const std::vector<DomainPoint> &ordered_points);
      void check_containment(const Mappable *mappable, unsigned idx,
                             const std::vector<DomainPoint> &ordered_points);
    public:
      bool is_complete(RegionTreeNode *node, Operation *op, 
                       unsigned index, IndexSpaceNode *projection_space) const;
      ProjectionNode* construct_projection_tree(Operation *op, unsigned index,
                        const RegionRequirement &req, ShardID local_shard,
                        RegionTreeNode *root, const ProjectionInfo &proj_info);
      static void add_to_projection_tree(LogicalRegion region,
                  RegionTreeNode *root, RegionTreeForest *context, 
                  std::map<RegionTreeNode*,ProjectionNode*> &node_map,
                  ShardID owner_shard);
#if 0
    public: 
      // From scratch
      ProjectionTree* construct_projection_tree(Operation *op, unsigned index,
                  ShardID local_shard, RegionTreeNode *root,
                  IndexSpaceNode *launch_domain, ShardingFunction *sharding, 
                  IndexSpaceNode *shard_domain) const;
      // Contribute to an existing tree
      void construct_projection_tree(Operation *op, unsigned index,
                  ShardID local_shard, RegionTreeNode *root, 
                  IndexSpaceNode *launch_domain, ShardingFunction *sharding,
                  IndexSpaceNode *sharding_domain,
                  std::map<IndexTreeNode*,ProjectionTree*> &node_map) const;
      static void add_to_projection_tree(LogicalRegion region,
                  IndexTreeNode *root, RegionTreeForest *context, 
                  std::map<IndexTreeNode*,ProjectionTree*> &node_map,
                  ShardID owner_shard = 0); 
#endif
    public:
      const unsigned depth; 
      const bool is_exclusive;
      const bool is_functional;
      const bool is_invertible;
      const ProjectionID projection_id;
      ProjectionFunctor *const functor;
    protected:
      mutable LocalLock projection_reservation;  
    }; 

    /**
     * \class CyclicShardingFunctor
     * The cyclic sharding functor just round-robins the points
     * onto the available set of shards
     */
    class CyclicShardingFunctor : public ShardingFunctor {
    public:
      CyclicShardingFunctor(void);
      CyclicShardingFunctor(const CyclicShardingFunctor &rhs) = delete;
      virtual ~CyclicShardingFunctor(void);
    public:
      CyclicShardingFunctor& operator=(
          const CyclicShardingFunctor &rhs) = delete;
    public:
      template<int DIM>
      size_t linearize_point(const Realm::IndexSpace<DIM,coord_t> &is,
                              const Realm::Point<DIM,coord_t> &point) const;
    public:
      virtual ShardID shard(const DomainPoint &point,
                            const Domain &full_space,
                            const size_t total_shards);
    };

    /**
     * \class ShardingFunction
     * The sharding function class wraps a sharding functor and will
     * cache results for queries so that we don't need to constantly
     * be inverting the results of the sharding functor.
     */
    class ShardingFunction {
    public:
      struct ShardKey {
      public:
        ShardKey(void) 
          : sid(0), full_space(IndexSpace::NO_SPACE), 
            shard_space(IndexSpace::NO_SPACE) { }
        ShardKey(ShardID s, IndexSpace f, IndexSpace sh)
          : sid(s), full_space(f), shard_space(sh) { }
      public:
        inline bool operator<(const ShardKey &rhs) const
        {
          if (sid < rhs.sid)
            return true;
          if (sid > rhs.sid)
            return false;
          if (full_space < rhs.full_space)
            return true;
          if (full_space > rhs.full_space)
            return false;
          return shard_space < rhs.shard_space;
        }
        inline bool operator==(const ShardKey &rhs) const
        {
          if (sid != rhs.sid)
            return false;
          if (full_space != rhs.full_space)
            return false;
          return shard_space == rhs.shard_space;
        }
      public:
        ShardID sid;
        IndexSpace full_space, shard_space;
      };
    public:
      ShardingFunction(ShardingFunctor *functor, RegionTreeForest *forest,
                       ShardManager *manager, ShardingID sharding_id, 
                       bool skip_checks = false, bool own_functor = false);
      ShardingFunction(const ShardingFunction &rhs) = delete;
      virtual ~ShardingFunction(void);
    public:
      ShardingFunction& operator=(const ShardingFunction &rhs) = delete;
    public:
      ShardID find_owner(const DomainPoint &point,
                         const Domain &sharding_space);
      IndexSpace find_shard_space(ShardID shard, IndexSpaceNode *full_space,
          IndexSpace sharding_space, Provenance *provenance);
      bool find_shard_participants(IndexSpaceNode *full_space,
          IndexSpace sharding_space, std::vector<ShardID> &participants);
    public:
      ShardingFunctor *const functor;
      RegionTreeForest *const forest;
      ShardManager *const manager;
      const ShardingID sharding_id;
      const bool use_points;
      const bool skip_checks;
      const bool own_functor;
    protected:
      mutable LocalLock sharding_lock;
      std::map<ShardKey,IndexSpace/*result*/> shard_index_spaces;
      std::map<std::pair<IndexSpace,IndexSpace>,
               std::vector<ShardID> > shard_participants;
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
    class Runtime : public LegionHeapify<Runtime> {
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
            eager_alloc_percentage(LEGION_DEFAULT_EAGER_ALLOC_PERCENTAGE),
            eager_alloc_percentage_overrides({}),
            max_message_size(LEGION_DEFAULT_MAX_MESSAGE_SIZE),
            gc_epoch_size(LEGION_DEFAULT_GC_EPOCH_SIZE),
            max_control_replication_contexts(
                        LEGION_DEFAULT_MAX_CONTROL_REPLICATION_CONTEXTS),
            max_local_fields(LEGION_DEFAULT_LOCAL_FIELDS),
            max_replay_parallelism(LEGION_DEFAULT_MAX_REPLAY_PARALLELISM),
            safe_control_replication(0),
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
#ifdef LEGION_SPY
            legion_spy_enabled(true),
#else
            legion_spy_enabled(false),
#endif
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
            dump_free_ranges(false),
            num_profiling_nodes(0),
            serializer_type("binary"),
            prof_footprint_threshold(128 << 20),
            prof_target_latency(100) { }
      public:
        int delay_start;
        int legion_collective_radix;
        int initial_task_window_size;
        unsigned initial_task_window_hysteresis;
        unsigned initial_tasks_to_schedule;
        unsigned initial_meta_task_vector_width;
        unsigned eager_alloc_percentage;
        std::map<Realm::Memory::Kind, unsigned> eager_alloc_percentage_overrides;
        unsigned max_message_size;
        unsigned gc_epoch_size;
        unsigned max_control_replication_contexts;
        unsigned max_local_fields;
        unsigned max_replay_parallelism;
        unsigned safe_control_replication;
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
        bool dump_free_ranges;
      public:
        unsigned num_profiling_nodes;
        std::string serializer_type;
        std::string prof_logfile;
        size_t prof_footprint_threshold;
        size_t prof_target_latency;
      public:
        bool parse_alloc_percentage_override_argument(const std::string& s);
      };
    public:
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
      struct DeferConcurrentAnalysisArgs :
        public LgTaskArgs<DeferConcurrentAnalysisArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_CONCURRENT_ANALYSIS_TASK_ID;
      public:
        DeferConcurrentAnalysisArgs(ProcessorManager *man, ApEvent n,
                                    ApUserEvent r)
          : LgTaskArgs<DeferConcurrentAnalysisArgs>(implicit_provenance),
            manager(man), next(n), result(r) { }
      public:
        ProcessorManager *const manager;
        const ApEvent next;
        const ApUserEvent result;
      };
    public:
      struct ProcessorGroupInfo {
      public:
        ProcessorGroupInfo(void)
          : processor_group(ProcessorGroup::NO_PROC_GROUP) { }
        ProcessorGroupInfo(ProcessorGroup p, const ProcessorMask &m)
          : processor_group(p), processor_mask(m) { }
      public:
        ProcessorGroup      processor_group;
        ProcessorMask       processor_mask;
      };
    public:
      Runtime(Machine m, const LegionConfiguration &config,
              bool background, InputArgs input_args, 
              AddressSpaceID space_id, Memory sysmem,
              const std::set<Processor> &local_procs,
              const std::set<Processor> &local_util_procs,
              const std::set<AddressSpaceID> &address_spaces,
              const std::map<Processor,AddressSpaceID> &proc_spaces,
              bool supply_default_mapper);
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
      const Memory runtime_system_memory;
      const AddressSpaceID address_space; 
      const unsigned total_address_spaces;
      // stride for uniqueness, may or may not be the same depending
      // on the number of available control replication contexts
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
      const unsigned eager_alloc_percentage;
      const std::map<Realm::Memory::Kind, unsigned> eager_alloc_percentage_overrides;
      const unsigned max_message_size;
      const unsigned gc_epoch_size;
      const unsigned max_control_replication_contexts;
      const unsigned max_local_fields;
      const unsigned max_replay_parallelism;
      const unsigned safe_control_replication;
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
      const bool supply_default_mapper;
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
      const bool dump_free_ranges;
    public:
      const unsigned num_profiling_nodes;
    public:
      const int legion_collective_radix;
      MPIRankTable *const mpi_rank_table;
    public:
      void register_static_variants(void);
      void register_static_constraints(void);
      void register_static_projections(void);
      void register_static_sharding_functors(void);
      void initialize_legion_prof(const LegionConfiguration &config);
      void log_machine(Machine machine) const;
      void initialize_mappers(void);
      void initialize_virtual_manager(void);
      void initialize_runtime(void);
#ifdef LEGION_USE_LIBDL
      void send_registration_callback(AddressSpaceID space,
                                      Realm::DSOReferenceImplementation *impl,
                                      RtEvent done, std::set<RtEvent> &applied,
                                      const void *buffer, size_t buffer_size,
                                      bool withargs, bool deduplicate,
                                      size_t dedup_tag);
#endif
      RtEvent perform_registration_callback(void *callback, const void *buffer,
          size_t size, bool withargs, bool global, bool preregistered,
          bool deduplicate, size_t dedup_tag);
      void startup_runtime(void);
      void finalize_runtime(void);
      ApEvent launch_mapper_task(Mapper *mapper, Processor proc, 
                                 TaskID tid,
                                 const UntypedBuffer &arg, MapperID map_id);
      void process_mapper_task_result(const MapperTaskArgs *args); 
    public:
      void create_shared_ownership(IndexSpace handle, 
              const bool total_sharding_collective = false,
              const bool unpack_reference = false);
      void create_shared_ownership(IndexPartition handle,
              const bool total_sharding_collective = false,
              const bool unpack_reference = false);
      void create_shared_ownership(FieldSpace handle,
              const bool total_sharding_collective = false,
              const bool unpack_reference = false);
      void create_shared_ownership(LogicalRegion handle,
              const bool total_sharding_collective = false,
              const bool unpack_reference = false);
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
      size_t get_field_size(Context ctx, FieldSpace handle, FieldID fid);
      size_t get_field_size(FieldSpace handle, FieldID fid);
      void get_field_space_fields(Context ctx, FieldSpace handle,
                                  std::vector<FieldID> &fields);
      void get_field_space_fields(FieldSpace handle, 
                                  std::vector<FieldID> &fields);
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
      ArgumentMap create_argument_map(void);
    public:
      Future execute_task(Context ctx,
                          const TaskLauncher &launcher,
                          std::vector<OutputRequirement> *outputs);
      FutureMap execute_index_space(Context ctx,
                                    const IndexTaskLauncher &launcher,
                                    std::vector<OutputRequirement> *outputs);
      Future execute_index_space(Context ctx, const IndexTaskLauncher &launcher,
                                 ReductionOpID redop, bool deterministic,
                                 std::vector<OutputRequirement> *outputs);
    public:
      PhysicalRegion map_region(Context ctx, 
                                const InlineLauncher &launcher);
      PhysicalRegion map_region(Context ctx, unsigned idx, 
                                MapperID id, MappingTagID tag,
                                Provenance *provenance);
      void remap_region(Context ctx, const PhysicalRegion &region,
                        Provenance *provenance = NULL);
      void unmap_region(Context ctx, PhysicalRegion region);
    public:
      void fill_fields(Context ctx, const FillLauncher &launcher);
      void fill_fields(Context ctx, const IndexFillLauncher &launcher);
      void issue_copy_operation(Context ctx, const CopyLauncher &launcher);
      void issue_copy_operation(Context ctx, const IndexCopyLauncher &launcher);
    public:
      void issue_acquire(Context ctx, const AcquireLauncher &launcher);
      void issue_release(Context ctx, const ReleaseLauncher &launcher);
      TraceID generate_dynamic_trace_id(bool check_context = true);
      TraceID generate_library_trace_ids(const char *name, size_t count);
      static TraceID& get_current_static_trace_id(void);
      static TraceID generate_static_trace_id(void);
      FutureMap execute_must_epoch(Context ctx, 
                                   const MustEpochLauncher &launcher);
      Future issue_timing_measurement(Context ctx,
                                      const TimingLauncher &launcher);
    public:
      void* get_local_task_variable(Context ctx, LocalVariableID id);
      void set_local_task_variable(Context ctx, LocalVariableID id,
                      const void *value, void (*destructor)(void*));
    public:
      Mapper* get_mapper(Context ctx, MapperID id, Processor target);
      MappingCallInfo* begin_mapper_call(Context ctx, MapperID id, 
                                         Processor target);
      void end_mapper_call(MappingCallInfo *info);
    public:
      void print_once(Context ctx, FILE *f, const char *message);
      void log_once(Context ctx, Realm::LoggerMessage &message);
    public:
      bool is_MPI_interop_configured(void);
      const std::map<int,AddressSpace>& find_forward_MPI_mapping(void);
      const std::map<AddressSpace,int>& find_reverse_MPI_mapping(void);
      int find_local_MPI_rank(void);
    public:
      Mapping::MapperRuntime* get_mapper_runtime(void);
      MapperID generate_dynamic_mapper_id(bool check_context = true);
      MapperID generate_library_mapper_ids(const char *name, size_t count);
      static MapperID& get_current_static_mapper_id(void);
      static MapperID generate_static_mapper_id(void);
      void add_mapper(MapperID map_id, Mapper *mapper, Processor proc);
      void replace_default_mapper(Mapper *mapper, Processor proc);
      MapperManager* find_mapper(MapperID map_id);
      MapperManager* find_mapper(Processor target, MapperID map_id);
      static MapperManager* wrap_mapper(Runtime *runtime, Mapper *mapper,
                MapperID map_id, Processor proc, bool is_default = false);
    public:
      ProjectionID generate_dynamic_projection_id(bool check_context = true);
      ProjectionID generate_library_projection_ids(const char *name,size_t cnt);
      static ProjectionID& get_current_static_projection_id(void);
      static ProjectionID generate_static_projection_id(void);
      void register_projection_functor(ProjectionID pid, 
                                       ProjectionFunctor *func,
                                       bool need_zero_check = true,
                                       bool silence_warnings = false,
                                       const char *warning_string = NULL,
                                       bool preregistered = false);
      static void preregister_projection_functor(ProjectionID pid,
                                       ProjectionFunctor *func);
      ProjectionFunction* find_projection_function(ProjectionID pid,
                                                   bool can_fail = false);
      static ProjectionFunctor* get_projection_functor(ProjectionID pid);
      void unregister_projection_functor(ProjectionID pid);
    public:
      ShardingID generate_dynamic_sharding_id(bool check_context = true);
      ShardingID generate_library_sharding_ids(const char *name, size_t count);
      static ShardingID& get_current_static_sharding_id(void);
      static ShardingID generate_static_sharding_id(void);
      void register_sharding_functor(ShardingID sid,
                                     ShardingFunctor *func,
                                     bool need_zero_check = true,
                                     bool silence_warnings= false,
                                     const char *warning_string = NULL,
                                     bool preregistered = false);
      static void preregister_sharding_functor(ShardingID sid,
                                     ShardingFunctor *func);
      ShardingFunctor* find_sharding_functor(ShardingID sid, 
                                             bool can_fail = false);
      static ShardingFunctor* get_sharding_functor(ShardingID sid);
    public:
      void register_reduction(ReductionOpID redop_id,
                              ReductionOp *redop,
                              SerdezInitFnptr init_fnptr,
                              SerdezFoldFnptr fold_fnptr,
                              bool permit_duplicates,
                              bool preregistered);
      void register_serdez(CustomSerdezID serdez_id,
                           SerdezOp *serdez_op,
                           bool permit_duplicates,
                           bool preregistered);
      const ReductionOp* get_reduction(ReductionOpID redop_id);
      FillView* find_or_create_reduction_fill_view(ReductionOpID redop_id);
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
      TaskID generate_dynamic_task_id(bool check_context = true);
      TaskID generate_library_task_ids(const char *name, size_t count);
      VariantID register_variant(const TaskVariantRegistrar &registrar,
                                 const void *user_data, size_t user_data_size,
                                 const CodeDescriptor &realm_desc,
                                 size_t return_type_size,
                                 bool has_return_type_size,
                                 VariantID vid = LEGION_AUTO_GENERATE_ID,
                                 bool check_task_id = true,
                                 bool check_context = true,
                                 bool preregistered = false);
      TaskImpl* find_or_create_task_impl(TaskID task_id);
      TaskImpl* find_task_impl(TaskID task_id);
      VariantImpl* find_variant_impl(TaskID task_id, VariantID variant_id,
                                     bool can_fail = false);
    public:
      ReductionOpID generate_dynamic_reduction_id(bool check_context = true);
      ReductionOpID generate_library_reduction_ids(const char *name, 
                                                   size_t count);
    public:
      CustomSerdezID generate_dynamic_serdez_id(bool check_context = true);
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
      void send_remote_task_profiling_response(Processor tar, Serializer &rez);
      void send_shared_ownership(AddressSpaceID target, Serializer &rez);
      void send_index_space_request(AddressSpaceID target, Serializer &rez);
      void send_index_space_response(AddressSpaceID target, Serializer &rez);
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
      void send_index_space_generate_color_request(AddressSpaceID target,
                                                   Serializer &rez);
      void send_index_space_generate_color_response(AddressSpaceID target,
                                                    Serializer &rez);
      void send_index_space_release_color(AddressSpaceID target, 
                                          Serializer &rez);
      void send_index_partition_notification(AddressSpaceID target, 
                                             Serializer &rez);
      void send_index_partition_request(AddressSpaceID target, Serializer &rez);
      void send_index_partition_response(AddressSpaceID target,Serializer &rez);
      void send_index_partition_return(AddressSpaceID target, Serializer &rez);
      void send_index_partition_child_request(AddressSpaceID target,
                                              Serializer &rez);
      void send_index_partition_child_response(AddressSpaceID target,
                                               Serializer &rez);
      void send_index_partition_disjoint_update(AddressSpaceID target,
                                                Serializer &rez);
      void send_index_partition_shard_rects_request(AddressSpaceID target,
                                                    Serializer &rez);
      void send_index_partition_shard_rects_response(AddressSpaceID target,
                                                     Serializer &rez);
      void send_index_partition_remote_interference_request(
                                    AddressSpaceID target, Serializer &rez);
      void send_index_partition_remote_interference_response(
                                    AddressSpaceID target, Serializer &rez);
      void send_field_space_node(AddressSpaceID target, Serializer &rez);
      void send_field_space_request(AddressSpaceID target, Serializer &rez);
      void send_field_space_return(AddressSpaceID target, Serializer &rez);
      void send_field_space_allocator_request(AddressSpaceID target, 
                                              Serializer &rez);
      void send_field_space_allocator_response(AddressSpaceID target, 
                                               Serializer &rez);
      void send_field_space_allocator_invalidation(AddressSpaceID, 
                                                   Serializer &rez);
      void send_field_space_allocator_flush(AddressSpaceID target, 
                                            Serializer &rez);
      void send_field_space_allocator_free(AddressSpaceID target, 
                                           Serializer &rez);
      void send_field_space_infos_request(AddressSpaceID, Serializer &rez);
      void send_field_space_infos_response(AddressSpaceID, Serializer &rez);
      void send_field_alloc_request(AddressSpaceID target, Serializer &rez);
      void send_field_size_update(AddressSpaceID target, Serializer &rez);
      void send_field_free(AddressSpaceID target, Serializer &rez);
      void send_field_free_indexes(AddressSpaceID target, Serializer &rez);
      void send_field_space_layout_invalidation(AddressSpaceID target, 
                                                Serializer &rez);
      void send_local_field_alloc_request(AddressSpaceID target, 
                                          Serializer &rez);
      void send_local_field_alloc_response(AddressSpaceID target,
                                           Serializer &rez);
      void send_local_field_free(AddressSpaceID target, Serializer &rez);
      void send_local_field_update(AddressSpaceID target, Serializer &rez);
      void send_top_level_region_request(AddressSpaceID target,Serializer &rez);
      void send_top_level_region_return(AddressSpaceID target, Serializer &rez);
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
                                           std::set<RtEvent> &applied);
      void send_individual_remote_future_size(Processor target,Serializer &rez);
      void send_individual_remote_complete(Processor target, Serializer &rez);
      void send_individual_remote_commit(Processor target, Serializer &rez);
      void send_slice_remote_mapped(Processor target, Serializer &rez);
      void send_slice_remote_complete(Processor target, Serializer &rez);
      void send_slice_remote_commit(Processor target, Serializer &rez);
      void send_slice_verify_concurrent_execution(Processor target,
                                                  Serializer &rez);
      void send_slice_find_intra_space_dependence(Processor target, 
                                                  Serializer &rez);
      void send_slice_record_intra_space_dependence(Processor target,
                                                    Serializer &rez);
      void send_slice_remote_rendezvous(Processor target, Serializer &rez);
      void send_did_remote_registration(AddressSpaceID target, Serializer &rez);
      void send_did_downgrade_request(AddressSpaceID target, Serializer &rez);
      void send_did_downgrade_response(AddressSpaceID target, Serializer &rez);
      void send_did_downgrade_success(AddressSpaceID target, Serializer &rez);
      void send_did_downgrade_update(AddressSpaceID target, Serializer &rez);
      void send_did_acquire_global_request(AddressSpaceID target, 
                                           Serializer &rez);
      void send_did_acquire_global_response(AddressSpaceID target,
                                            Serializer &rez);
      void send_did_acquire_valid_request(AddressSpaceID target,
                                          Serializer &rez);
      void send_did_acquire_valid_response(AddressSpaceID target,
                                           Serializer &rez);
      void send_created_region_contexts(AddressSpaceID target, Serializer &rez);
      void send_back_atomic(AddressSpaceID target, Serializer &rez);
      void send_atomic_reservation_request(AddressSpaceID target, 
                                           Serializer &rez);
      void send_atomic_reservation_response(AddressSpaceID target, 
                                            Serializer &rez);
      void send_padded_reservation_request(AddressSpaceID target, 
                                           Serializer &rez);
      void send_padded_reservation_response(AddressSpaceID target, 
                                            Serializer &rez);
      void send_materialized_view(AddressSpaceID target, Serializer &rez);
      void send_fill_view(AddressSpaceID target, Serializer &rez);
      void send_fill_view_value(AddressSpaceID target, Serializer &rez);
      void send_phi_view(AddressSpaceID target, Serializer &rez);
      void send_reduction_view(AddressSpaceID target, Serializer &rez);
      void send_replicated_view(AddressSpaceID target, Serializer &rez);
      void send_allreduce_view(AddressSpaceID target, Serializer &rez);
      void send_instance_manager(AddressSpaceID target, Serializer &rez);
      void send_manager_update(AddressSpaceID target, Serializer &rez);
      void send_collective_distribute_fill(AddressSpaceID target,
                                           Serializer &rez);
      void send_collective_distribute_point(AddressSpaceID target,
                                            Serializer &rez);
      void send_collective_distribute_pointwise(AddressSpaceID target,
                                                Serializer &rez);
      void send_collective_distribute_reduction(AddressSpaceID target,
                                                Serializer &rez);
      void send_collective_distribute_broadcast(AddressSpaceID target,
                                                Serializer &rez);
      void send_collective_distribute_reducecast(AddressSpaceID target,
                                                 Serializer &rez);
      void send_collective_distribute_hourglass(AddressSpaceID target,
                                                Serializer &rez);
      void send_collective_distribute_allreduce(AddressSpaceID target,
                                                Serializer &rez);
      void send_collective_hammer_reduction(AddressSpaceID target,
                                            Serializer &rez);
      void send_collective_fuse_gather(AddressSpaceID target, Serializer &rez);
      void send_collective_register_user_request(AddressSpaceID target,
                                                 Serializer &rez);
      void send_collective_register_user_response(AddressSpaceID target,
                                                  Serializer &rez);
      void send_collective_individual_register_user(AddressSpaceID target,
                                                    Serializer &rez);
      void send_collective_point_request(AddressSpaceID target,Serializer &rez);
      void send_collective_point_response(AddressSpaceID target,
                                          Serializer &rez);
      void send_collective_remote_instances_request(AddressSpaceID target,
                                                    Serializer &rez);
      void send_collective_remote_instances_response(AddressSpaceID target,
                                                     Serializer &rez);
      void send_collective_nearest_instances_request(AddressSpaceID target,
                                                     Serializer &rez);
      void send_collective_nearest_instances_response(AddressSpaceID target,
                                                      Serializer &rez);
      void send_collective_remote_registration(AddressSpaceID target,
                                               Serializer &rez);
      void send_collective_deletion(AddressSpaceID target, Serializer &rez);
      void send_collective_finalize_mapping(AddressSpaceID target,
                                            Serializer &rez);
      void send_collective_view_creation(AddressSpaceID target,Serializer &rez);
      void send_collective_view_deletion(AddressSpaceID target,Serializer &rez);
      void send_collective_view_release(AddressSpaceID target, Serializer &rez);
      void send_collective_view_notification(AddressSpaceID target,
                                             Serializer &rez);
      void send_collective_view_make_valid(AddressSpaceID target, 
                                           Serializer &rez);
      void send_collective_view_make_invalid(AddressSpaceID target, 
                                             Serializer &rez);
      void send_collective_view_invalidate_request(AddressSpaceID target,
                                                   Serializer &rez);
      void send_collective_view_invalidate_response(AddressSpaceID target,
                                                    Serializer &rez);
      void send_collective_view_add_remote_reference(AddressSpaceID target,
                                                     Serializer &rez);
      void send_collective_view_remove_remote_reference(AddressSpaceID target,
                                                        Serializer &rez);
      void send_create_top_view_request(AddressSpaceID target, Serializer &rez);
      void send_create_top_view_response(AddressSpaceID target,Serializer &rez);
      void send_view_register_user(AddressSpaceID target, Serializer &rez);
      void send_view_find_copy_preconditions_request(AddressSpaceID target,
                                                     Serializer &rez);
      void send_view_add_copy_user(AddressSpaceID target, Serializer &rez);
      void send_view_find_last_users_request(AddressSpaceID target,
                                             Serializer &rez);
      void send_view_find_last_users_response(AddressSpaceID target,
                                              Serializer &rez);
#ifdef ENABLE_VIEW_REPLICATION
      void send_view_replication_request(AddressSpaceID target,Serializer &rez);
      void send_view_replication_response(AddressSpaceID target,
                                          Serializer &rez);
      void send_view_replication_removal(AddressSpaceID target,Serializer &rez);
#endif
      void send_future_result(AddressSpaceID target, Serializer &rez);
      void send_future_result_size(AddressSpaceID target, Serializer &rez);
      void send_future_subscription(AddressSpaceID target, Serializer &rez);
      void send_future_create_instance_request(AddressSpaceID target,
                                               Serializer &rez);
      void send_future_create_instance_response(AddressSpaceID target,
                                                Serializer &rez);
      void send_future_map_request_future(AddressSpaceID target, 
                                          Serializer &rez);
      void send_future_map_response_future(AddressSpaceID target,
                                           Serializer &rez);
      void send_control_replicate_compute_equivalence_sets(
                                        AddressSpaceID target, Serializer &rez);
      void send_control_replicate_equivalence_set_notification(
                                        AddressSpaceID target, Serializer &rez);
      void send_control_replicate_intra_space_dependence(AddressSpaceID target,
                                                         Serializer &rez);
      void send_control_replicate_broadcast_update(AddressSpaceID target,
                                                   Serializer &rez);
      void send_control_replicate_created_regions(AddressSpaceID target,
                                                  Serializer &rez);
      void send_control_replicate_trace_event_request(AddressSpaceID target,
                                                      Serializer &rez);
      void send_control_replicate_trace_event_response(AddressSpaceID target,
                                                       Serializer &rez);
      void send_control_replicate_trace_frontier_request(AddressSpaceID target,
                                                      Serializer &rez);
      void send_control_replicate_trace_frontier_response(AddressSpaceID target,
                                                       Serializer &rez);
      void send_control_replicate_trace_update(AddressSpaceID target,
                                               Serializer &rez);
      void send_control_replicate_implicit_request(AddressSpaceID target,
                                                   Serializer &rez);
      void send_control_replicate_implicit_response(AddressSpaceID target,
                                                    Serializer &rez);
      void send_control_replicate_find_collective_view(AddressSpaceID target,
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
      void send_remote_context_physical_request(AddressSpaceID target, 
                                                Serializer &rez);
      void send_remote_context_physical_response(AddressSpaceID target,
                                                 Serializer &rez);
      void send_remote_context_find_collective_view_request(
                                                  AddressSpaceID target,
                                                  Serializer &rez);
      void send_remote_context_find_collective_view_response(
                                                  AddressSpaceID target,
                                                  Serializer &rez);
      void send_remote_context_collective_rendezvous(AddressSpaceID target,
                                                     Serializer &rez);
      void send_compute_equivalence_sets_request(AddressSpaceID target, 
                                                 Serializer &rez);
      void send_compute_equivalence_sets_response(AddressSpaceID target,
                                                  Serializer &rez);
      void send_cancel_equivalence_sets_subscription(AddressSpaceID target,
                                                     Serializer &rez);
      void send_finish_equivalence_sets_subscription(AddressSpaceID target,
                                                     Serializer &rez);
      void send_equivalence_set_response(AddressSpaceID target,Serializer &rez);
      void send_equivalence_set_replication_request(AddressSpaceID target,
                                                    Serializer &rez);
      void send_equivalence_set_replication_response(AddressSpaceID target,
                                                     Serializer &rez);
      void send_equivalence_set_replication_invalidation(AddressSpaceID target,
                                                         Serializer &rez);
      void send_equivalence_set_migration(AddressSpaceID target, 
                                          Serializer &rez);
      void send_equivalence_set_owner_update(AddressSpaceID target,
                                             Serializer &rez);
      void send_equivalence_set_make_owner(AddressSpaceID target,
                                           Serializer &rez);
      void send_equivalence_set_clone_request(AddressSpaceID target,
                                              Serializer &rez);
      void send_equivalence_set_clone_response(AddressSpaceID target,
                                               Serializer &rez);
      void send_equivalence_set_capture_request(AddressSpaceID target,
                                                Serializer &rez);
      void send_equivalence_set_capture_response(AddressSpaceID target,
                                                 Serializer &rez);
      void send_equivalence_set_remote_request_instances(AddressSpaceID target,
                                                         Serializer &rez);
      void send_equivalence_set_remote_request_invalid(AddressSpaceID target,
                                                       Serializer &rez);
      void send_equivalence_set_remote_request_antivalid(AddressSpaceID target,
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
      void send_equivalence_set_remote_clones(AddressSpaceID target,
                                              Serializer &rez);
      void send_equivalence_set_remote_instances(AddressSpaceID target,
                                                 Serializer &rez);
      void send_instance_request(AddressSpaceID target, Serializer &rez);
      void send_instance_response(AddressSpaceID target, Serializer &rez);
      void send_external_create_request(AddressSpaceID target, Serializer &rez);
      void send_external_create_response(AddressSpaceID target,Serializer &rez);
      void send_external_attach(AddressSpaceID target, Serializer &rez);
      void send_external_detach(AddressSpaceID target, Serializer &rez);
      void send_gc_priority_update(AddressSpaceID target, Serializer &rez);
      void send_gc_request(AddressSpaceID target, Serializer &rez);
      void send_gc_response(AddressSpaceID target, Serializer &rez);
      void send_gc_acquire(AddressSpaceID target, Serializer &rez);
      void send_gc_failed(AddressSpaceID target, Serializer &rez);
      void send_gc_mismatch(AddressSpaceID target, Serializer &rez);
      void send_gc_notify(AddressSpaceID target, Serializer &rez);
      void send_gc_debug_request(AddressSpaceID target, Serializer &rez);
      void send_gc_debug_response(AddressSpaceID target, Serializer &rez);
      void send_gc_record_event(AddressSpaceID target, Serializer &rez);
      void send_acquire_request(AddressSpaceID target, Serializer &rez);
      void send_acquire_response(AddressSpaceID target, Serializer &rez);
      void send_variant_broadcast(AddressSpaceID target, Serializer &rez);
      void send_constraint_request(AddressSpaceID target, Serializer &rez);
      void send_constraint_response(AddressSpaceID target, Serializer &rez);
      void send_constraint_release(AddressSpaceID target, Serializer &rez);
      void send_mpi_rank_exchange(AddressSpaceID target, Serializer &rez);
      void send_replicate_launch(AddressSpaceID target, Serializer &rez);
      void send_replicate_post_mapped(AddressSpaceID target, Serializer &rez);
      void send_replicate_post_execution(AddressSpaceID target,
                                         Serializer &rez);
      void send_replicate_trigger_complete(AddressSpaceID target, 
                                           Serializer &rez);
      void send_replicate_trigger_commit(AddressSpaceID target,
                                         Serializer &rez);
      void send_control_replicate_collective_message(AddressSpaceID target,
                                                     Serializer &rez);
      void send_control_replicate_rendezvous_message(AddressSpaceID target,
                                                     Serializer &rez);
      void send_library_mapper_request(AddressSpaceID target, Serializer &rez);
      void send_library_mapper_response(AddressSpaceID target, Serializer &rez);
      void send_library_trace_request(AddressSpaceID target, Serializer &rez);
      void send_library_trace_response(AddressSpaceID target, Serializer &rez);
      void send_library_projection_request(AddressSpaceID target, 
                                           Serializer &rez);
      void send_library_projection_response(AddressSpaceID target,
                                            Serializer &rez);
      void send_library_sharding_request(AddressSpaceID target,Serializer &rez);
      void send_library_sharding_response(AddressSpaceID target, 
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
      void send_remote_op_completion_effect(AddressSpaceID target,
                                            Serializer &rez);
      void send_remote_trace_update(AddressSpaceID target, Serializer &rez);
      void send_remote_trace_response(AddressSpaceID target, Serializer &rez);
      void send_free_external_allocation(AddressSpaceID target,Serializer &rez);
      void send_create_future_instance_request(AddressSpaceID target,
                                               Serializer &rez);
      void send_create_future_instance_response(AddressSpaceID target,
                                                Serializer &rez);
      void send_free_future_instance(AddressSpaceID target, Serializer &rez);
      void send_shutdown_notification(AddressSpaceID target, Serializer &rez);
      void send_shutdown_response(AddressSpaceID target, Serializer &rez);
    public:
      // Complementary tasks for handling messages
      void handle_task(Deserializer &derez);
      void handle_steal(Deserializer &derez);
      void handle_advertisement(Deserializer &derez);
#ifdef LEGION_USE_LIBDL
      void handle_registration_callback(Deserializer &derez);
#endif
      void handle_remote_task_replay(Deserializer &derez);
      void handle_remote_task_profiling_response(Deserializer &derez);
      void handle_shared_ownership(Deserializer &derez);
      void handle_index_space_request(Deserializer &derez, 
                                      AddressSpaceID source);
      void handle_index_space_response(Deserializer &derez,
                                       AddressSpaceID source);
      void handle_index_space_return(Deserializer &derez,
                                     AddressSpaceID source); 
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
      void handle_index_space_generate_color_request(Deserializer &derez,
                                                     AddressSpaceID source);
      void handle_index_space_generate_color_response(Deserializer &derez);
      void handle_index_space_release_color(Deserializer &derez);
      void handle_index_partition_notification(Deserializer &derez);
      void handle_index_partition_request(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_index_partition_response(Deserializer &derez,
                                           AddressSpaceID source);
      void handle_index_partition_return(Deserializer &derez,
                                         AddressSpaceID source);
      void handle_index_partition_child_request(Deserializer &derez,
                                                AddressSpaceID source);
      void handle_index_partition_child_response(Deserializer &derez);
      void handle_index_partition_disjoint_update(Deserializer &derez);
      void handle_index_partition_shard_rects_request(Deserializer &derez);
      void handle_index_partition_shard_rects_response(Deserializer &derez,
                                                       AddressSpaceID source);
      void handle_index_partition_remote_interference_request(
                                   Deserializer &derez, AddressSpaceID source);
      void handle_index_partition_remote_interference_response(
                                   Deserializer &derez);
      void handle_field_space_node(Deserializer &derez, AddressSpaceID source);
      void handle_field_space_request(Deserializer &derez,
                                      AddressSpaceID source);
      void handle_field_space_return(Deserializer &derez);
      void handle_field_space_allocator_request(Deserializer &derez,
                                                AddressSpaceID source);
      void handle_field_space_allocator_response(Deserializer &derez);
      void handle_field_space_allocator_invalidation(Deserializer &derez);
      void handle_field_space_allocator_flush(Deserializer &derez);
      void handle_field_space_allocator_free(Deserializer &derez, 
                                             AddressSpaceID source);
      void handle_field_space_infos_request(Deserializer &derez);
      void handle_field_space_infos_response(Deserializer &derez);
      void handle_field_alloc_request(Deserializer &derez);
      void handle_field_size_update(Deserializer &derez, AddressSpaceID source);
      void handle_field_free(Deserializer &derez, AddressSpaceID source);
      void handle_field_free_indexes(Deserializer &derez);
      void handle_field_space_layout_invalidation(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_local_field_alloc_request(Deserializer &derez,
                                            AddressSpaceID source);
      void handle_local_field_alloc_response(Deserializer &derez);
      void handle_local_field_free(Deserializer &derez);
      void handle_local_field_update(Deserializer &derez);
      void handle_top_level_region_request(Deserializer &derez,
                                           AddressSpaceID source);
      void handle_top_level_region_return(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_index_space_destruction(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_index_partition_destruction(Deserializer &derez);
      void handle_field_space_destruction(Deserializer &derez);
      void handle_logical_region_destruction(Deserializer &derez);
      void handle_individual_remote_future_size(Deserializer &derez);
      void handle_individual_remote_complete(Deserializer &derez);
      void handle_individual_remote_commit(Deserializer &derez);
      void handle_slice_remote_mapped(Deserializer &derez, 
                                      AddressSpaceID source);
      void handle_slice_remote_complete(Deserializer &derez);
      void handle_slice_remote_commit(Deserializer &derez);
      void handle_slice_verify_concurrent_execution(Deserializer &derez);
      void handle_slice_find_intra_dependence(Deserializer &derez);
      void handle_slice_record_intra_dependence(Deserializer &derez);
      void handle_slice_remote_collective_rendezvous(Deserializer &derez,
                                                     AddressSpaceID source);
      void handle_did_remote_registration(Deserializer &derez, 
                                          AddressSpaceID source);
      void handle_did_downgrade_request(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_did_downgrade_response(Deserializer &derez);
      void handle_did_downgrade_success(Deserializer &derez);
      void handle_did_downgrade_update(Deserializer &derez);
      void handle_did_global_acquire_request(Deserializer &derez);
      void handle_did_global_acquire_response(Deserializer &derez);
      void handle_did_valid_acquire_request(Deserializer &derez);
      void handle_did_valid_acquire_response(Deserializer &derez);
      void handle_created_region_contexts(Deserializer &derez);  
      void handle_send_atomic_reservation_request(Deserializer &derez);
      void handle_send_atomic_reservation_response(Deserializer &derez);
      void handle_send_padded_reservation_request(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_send_padded_reservation_response(Deserializer &derez);
      void handle_send_materialized_view(Deserializer &derez); 
      void handle_send_fill_view(Deserializer &derez);
      void handle_send_fill_view_value(Deserializer &derez);
      void handle_send_phi_view(Deserializer &derez);
      void handle_send_reduction_view(Deserializer &derez);
      void handle_send_replicated_view(Deserializer &derez);
      void handle_send_allreduce_view(Deserializer &derez);
      void handle_send_instance_manager(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_send_manager_update(Deserializer &derez,
                                      AddressSpaceID source);
      void handle_collective_distribute_fill(Deserializer &derez,
                                             AddressSpaceID source);
      void handle_collective_distribute_point(Deserializer &derez,
                                              AddressSpaceID source);
      void handle_collective_distribute_pointwise(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_collective_distribute_reduction(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_collective_distribute_broadcast(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_collective_distribute_reducecast(Deserializer &derez,
                                                   AddressSpaceID source);
      void handle_collective_distribute_hourglass(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_collective_distribute_allreduce(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_collective_hammer_reduction(Deserializer &derez,
                                              AddressSpaceID source);
      void handle_collective_fuse_gather(Deserializer &derez,
                                         AddressSpaceID source);
      void handle_collective_user_request(Deserializer &derez);
      void handle_collective_user_response(Deserializer &derez);
      void handle_collective_user_registration(Deserializer &derez);
      void handle_collective_remote_instances_request(Deserializer &derez,
                                                    AddressSpaceID source);
      void handle_collective_remote_instances_response(Deserializer &derez,
                                                    AddressSpaceID source);
      void handle_collective_nearest_instances_request(Deserializer &derez);
      void handle_collective_nearest_instances_response(Deserializer &derez);
      void handle_collective_remote_registration(Deserializer &derez);
      void handle_collective_finalize_mapping(Deserializer &derez);
      void handle_collective_view_creation(Deserializer &derez);
      void handle_collective_view_deletion(Deserializer &derez);
      void handle_collective_view_release(Deserializer &derez);
      void handle_collective_view_notification(Deserializer &derez);
      void handle_collective_view_make_valid(Deserializer &derez);
      void handle_collective_view_make_invalid(Deserializer &derez);
      void handle_collective_view_invalidate_request(Deserializer &derez);
      void handle_collective_view_invalidate_response(Deserializer &derez);
      void handle_collective_view_add_remote_reference(Deserializer &derez);
      void handle_collective_view_remove_remote_reference(Deserializer &derez);
      void handle_create_top_view_request(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_create_top_view_response(Deserializer &derez);
      void handle_view_request(Deserializer &derez, AddressSpaceID source);
      void handle_view_register_user(Deserializer &derez,AddressSpaceID source);
      void handle_view_copy_pre_request(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_view_add_copy_user(Deserializer &derez,AddressSpaceID source);
      void handle_view_find_last_users_request(Deserializer &derez,
                                               AddressSpaceID source);
      void handle_view_find_last_users_response(Deserializer &derez);
#ifdef ENABLE_VIEW_REPLICATION
      void handle_view_replication_request(Deserializer &derez,
                                           AddressSpaceID source);
      void handle_view_replication_response(Deserializer &derez);
      void handle_view_replication_removal(Deserializer &derez, 
                                           AddressSpaceID source);
#endif
      void handle_manager_request(Deserializer &derez, AddressSpaceID source);
      void handle_future_result(Deserializer &derez);
      void handle_future_result_size(Deserializer &derez,
                                     AddressSpaceID source);
      void handle_future_subscription(Deserializer &derez, 
                                      AddressSpaceID source);
      void handle_future_create_instance_request(Deserializer &derez);
      void handle_future_create_instance_response(Deserializer &derez);
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
      void handle_remote_context_physical_request(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_remote_context_physical_response(Deserializer &derez);
      void handle_remote_context_find_collective_view_request(
                                                      Deserializer &derez,
                                                      AddressSpaceID source);
      void handle_remote_context_find_collective_view_response(
                                                      Deserializer &derez);
      void handle_compute_equivalence_sets_request(Deserializer &derez, 
                                                   AddressSpaceID source);
      void handle_compute_equivalence_sets_response(Deserializer &derez,
                                                    AddressSpaceID source);
      void handle_cancel_equivalence_sets_subscription(Deserializer &derez,
                                                       AddressSpaceID source);
      void handle_finish_equivalence_sets_subscription(Deserializer &derez,
                                                       AddressSpaceID source);
      void handle_equivalence_set_request(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_equivalence_set_response(Deserializer &derez);
      void handle_equivalence_set_invalidate_trackers(Deserializer &derez);
      void handle_equivalence_set_replication_request(Deserializer &derez,
                                                      AddressSpaceID source);
      void handle_equivalence_set_replication_response(Deserializer &derez);
      void handle_equivalence_set_replication_invalidation(Deserializer &derez);
      void handle_equivalence_set_migration(Deserializer &derez,
                                            AddressSpaceID source);
      void handle_equivalence_set_owner_update(Deserializer &derez);
      void handle_equivalence_set_make_owner(Deserializer &derez);
      void handle_equivalence_set_clone_request(Deserializer &derez);
      void handle_equivalence_set_clone_response(Deserializer &derez);
      void handle_equivalence_set_capture_request(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_equivalence_set_capture_response(Deserializer &derez,
                                                   AddressSpaceID source);
      void handle_equivalence_set_remote_request_instances(Deserializer &derez, 
                                                         AddressSpaceID srouce);
      void handle_equivalence_set_remote_request_invalid(Deserializer &derez, 
                                                         AddressSpaceID srouce);
      void handle_equivalence_set_remote_request_antivalid(Deserializer &derez,
                                                         AddressSpaceID source);
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
      void handle_equivalence_set_remote_clones(Deserializer &derez,
                                                AddressSpaceID source);
      void handle_equivalence_set_remote_instances(Deserializer &derez);
      void handle_instance_request(Deserializer &derez, AddressSpaceID source);
      void handle_instance_response(Deserializer &derez,AddressSpaceID source);
      void handle_external_create_request(Deserializer &derez,
                                          AddressSpaceID source);
      void handle_external_create_response(Deserializer &derez);
      void handle_external_attach(Deserializer &derez);
      void handle_external_detach(Deserializer &derez);
      void handle_gc_priority_update(Deserializer &derez,AddressSpaceID source);
      void handle_gc_request(Deserializer &derez, AddressSpaceID source);
      void handle_gc_response(Deserializer &derez);
      void handle_gc_acquire(Deserializer &derez);
      void handle_gc_failed(Deserializer &derez);
      void handle_gc_mismatch(Deserializer &derez);
      void handle_gc_notify(Deserializer &derez);
      void handle_gc_debug_request(Deserializer &derez, AddressSpaceID source);
      void handle_gc_debug_response(Deserializer &derez);
      void handle_gc_record_event(Deserializer &derez);
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
      void handle_replicate_launch(Deserializer &derez,AddressSpaceID source);
      void handle_replicate_post_mapped(Deserializer &derez);
      void handle_replicate_post_execution(Deserializer &derez);
      void handle_replicate_trigger_complete(Deserializer &derez);
      void handle_replicate_trigger_commit(Deserializer &derez);
      void handle_control_replicate_collective_message(Deserializer &derez);
      void handle_control_replicate_rendezvous_message(Deserializer &derez);
      void handle_control_replicate_compute_equivalence_sets(
                                                           Deserializer &derez);
      void handle_control_replicate_equivalence_set_notification(
                                                           Deserializer &derez);
      void handle_control_replicate_intra_space_dependence(Deserializer &derez);
      void handle_control_replicate_broadcast_update(Deserializer &derez);
      void handle_control_replicate_created_regions(Deserializer &derez);
      void handle_control_replicate_trace_event_request(Deserializer &derez,
                                                        AddressSpaceID source);
      void handle_control_replicate_trace_event_response(Deserializer &derez);
      void handle_control_replicate_trace_frontier_request(Deserializer &derez,
                                                        AddressSpaceID source);
      void handle_control_replicate_trace_frontier_response(
                                                        Deserializer &derez);
      void handle_control_replicate_trace_update(Deserializer &derez,
                                                 AddressSpaceID source);
      void handle_control_replicate_implicit_request(Deserializer &derez,
                                                     AddressSpaceID source);
      void handle_control_replicate_implicit_response(Deserializer &derez);
      void handle_control_replicate_find_collective_view(Deserializer &derez);
      void handle_library_mapper_request(Deserializer &derez,
                                         AddressSpaceID source);
      void handle_library_mapper_response(Deserializer &derez);
      void handle_library_trace_request(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_library_trace_response(Deserializer &derez);
      void handle_library_projection_request(Deserializer &derez,
                                             AddressSpaceID source);
      void handle_library_projection_response(Deserializer &derez);
      void handle_library_sharding_request(Deserializer &derez,
                                           AddressSpaceID source);
      void handle_library_sharding_response(Deserializer &derez);
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
      void handle_remote_op_completion_effect(Deserializer &derez);
      void handle_remote_tracing_update(Deserializer &derez,
                                        AddressSpaceID source);
      void handle_remote_tracing_response(Deserializer &derez);
      void handle_free_external_allocation(Deserializer &derez);
      void handle_create_future_instance_request(Deserializer &derez,
                                                 AddressSpaceID source);
      void handle_create_future_instance_response(Deserializer &derez);
      void handle_free_future_instance(Deserializer &derez);
      void handle_concurrent_reservation_creation(Deserializer &derez,
                                                  AddressSpaceID source);
      void handle_concurrent_execution_analysis(Deserializer &derez);
      void handle_shutdown_notification(Deserializer &derez, 
                                        AddressSpaceID source);
      void handle_shutdown_response(Deserializer &derez);
    public: // Calls to handle mapper requests
      bool create_physical_instance(Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result,
                                    Processor processor, bool acquire, 
                                    GCPriority priority, bool tight_bounds,
                                    const LayoutConstraint **unsat,
                                    size_t *footprint, UniqueID creator_id);
      bool create_physical_instance(Memory target_memory, 
                                    LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result,
                                    Processor processor, bool acquire, 
                                    GCPriority priority, bool tight_bounds,
                                    const LayoutConstraint **unsat,
                                    size_t *footprint, UniqueID creator_id);
      bool find_or_create_physical_instance(Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    Processor processor,
                                    bool acquire, GCPriority priority,
                                    bool tight_bounds, 
                                    const LayoutConstraint **unsat,
                                    size_t *footprint, UniqueID creator_id);
      bool find_or_create_physical_instance(Memory target_memory,
                                    LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    Processor processor,
                                    bool acquire, GCPriority priority,
                                    bool tight_bounds, 
                                    const LayoutConstraint **unsat,
                                    size_t *footprint, UniqueID creator_id);
      bool find_physical_instance(Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_region_bounds);
      bool find_physical_instance(Memory target_memory,
                                    LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_region_bounds);
      void find_physical_instances(Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    std::vector<MappingInstance> &results, 
                                    bool acquire, bool tight_region_bounds);
      void find_physical_instances(Memory target_memory,
                                    LayoutConstraints *constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    std::vector<MappingInstance> &result, 
                                    bool acquire, bool tight_region_bounds);
      void release_tree_instances(RegionTreeID tid);
    public:
      // Manage the execution of tasks within a context
      void activate_context(InnerContext *context);
      void deactivate_context(InnerContext *context);
    public:
      void add_to_ready_queue(Processor p, TaskOp *task_op);
    public:
      inline Processor find_utility_group(void) { return utility_group; }
      Processor find_processor_group(const std::vector<Processor> &procs);
      ProcessorMask find_processor_mask(const std::vector<Processor> &procs);
      template<typename T>
      inline RtEvent issue_runtime_meta_task(const LgTaskArgs<T> &args,
                                             LgPriority lg_priority,
                                   RtEvent precondition = RtEvent::NO_RT_EVENT,
                                   Processor proc = Processor::NO_PROC);
      template<typename T>
      inline RtEvent issue_application_processor_task(const LgTaskArgs<T> &args,
                                   LgPriority lg_priority, const Processor proc,
                                   RtEvent precondition = RtEvent::NO_RT_EVENT);
    public:
      // Support for concurrent index task execution 
      RtEvent acquire_concurrent_reservation(RtEvent release_event,
                        RtEvent precondition = RtEvent::NO_RT_EVENT);
      Reservation find_or_create_concurrent_reservation(void);
      RtEvent find_concurrent_fence_event(Processor target, ApEvent next,
                                ApEvent &previous, RtEvent precondition);
      static void handle_concurrent_analysis(const void *args);
    public:
      DistributedID get_available_distributed_id(void); 
      DistributedID get_remote_distributed_id(AddressSpaceID from);
      void handle_remote_distributed_id_request(Deserializer &derez,
                                                AddressSpaceID source);
      void handle_remote_distributed_id_response(Deserializer &derez);
      AddressSpaceID determine_owner(DistributedID did) const;
      size_t find_distance(AddressSpaceID src, AddressSpaceID dst) const;
    public:
      void register_distributed_collectable(DistributedID did,
                                            DistributedCollectable *dc);
      void unregister_distributed_collectable(DistributedID did);
      bool has_distributed_collectable(DistributedID did);
      DistributedCollectable* find_distributed_collectable(DistributedID did, 
                                                           bool wait = false);
      DistributedCollectable* find_distributed_collectable(DistributedID did,
                                                           RtEvent &ready, 
                                                           bool wait = false);
      DistributedCollectable* weak_find_distributed_collectable(
                                                           DistributedID did);
      bool find_pending_collectable_location(DistributedID did,void *&location);
      void* find_or_create_pending_collectable_location(DistributedID did, 
                                                        size_t size);
      void record_pending_distributed_collectable(DistributedID did);
      void revoke_pending_distributed_collectable(DistributedID did);
      bool find_or_create_distributed_collectable(DistributedID did,
          DistributedCollectable *&collectable, RtEvent &ready, void *buffer);
    public:
      LogicalView* find_or_request_logical_view(DistributedID did,
                                                RtEvent &ready);
      PhysicalManager* find_or_request_instance_manager(DistributedID did, 
                                                        RtEvent &ready);
      EquivalenceSet* find_or_request_equivalence_set(DistributedID did,
                                                      RtEvent &ready);
      InnerContext* find_or_request_inner_context(DistributedID did,
                                                  RtEvent &ready);
      ShardManager* find_shard_manager(DistributedID did, bool can_fail=false);
    protected:
      template<typename T, MessageKind MK>
      DistributedCollectable* find_or_request_distributed_collectable(
                                            DistributedID did, RtEvent &ready);
    public:
      FutureImpl* find_or_create_future(DistributedID did,
                                        DistributedID ctx_did,
                                        size_t op_ctx_index,
                                        const DomainPoint &point,
                                        Provenance *provenance,
                                        Operation *op = NULL,
                                        GenerationID op_gen = 0, 
#ifdef LEGION_SPY
                                        UniqueID op_uid = 0,
#endif
                                        int op_depth = 0,
                                        CollectiveMapping *mapping = NULL);
      FutureMapImpl* find_or_create_future_map(DistributedID did, 
                          TaskContext *ctx, size_t index, IndexSpace domain,
                          ApEvent completion, Provenance *provenance);
      IndexSpace find_or_create_index_slice_space(const Domain &launch_domain,
                                    TypeTag type_tag, Provenance *provenance);
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
      void finalize_runtime_shutdown(int exit_code);
    public:
      bool has_outstanding_tasks(void);
#ifdef DEBUG_LEGION
      void increment_total_outstanding_tasks(unsigned tid, bool meta);
      void decrement_total_outstanding_tasks(unsigned tid, bool meta);
#else
      inline void increment_total_outstanding_tasks(void)
        { total_outstanding_tasks.fetch_add(1); }
      inline void decrement_total_outstanding_tasks(void)
        { total_outstanding_tasks.fetch_sub(1); }
#endif
    public:
      template<typename T>
      inline T* get_available(LocalLock &local_lock, std::deque<T*> &queue);
      template<typename T, typename WRAP>
      inline T* get_available(LocalLock &local_lock, std::deque<T*> &queue);
      template<typename T>
      inline void free_available(std::deque<T*> &queue);
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
      RefinementOp*         get_available_refinement_op(void);
      AdvisementOp*         get_available_advisement_op(void);
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
      DiscardOp*            get_available_discard_op(void);
      AttachOp*             get_available_attach_op(void);
      IndexAttachOp*        get_available_index_attach_op(void);
      PointAttachOp*        get_available_point_attach_op(void);
      DetachOp*             get_available_detach_op(void);
      IndexDetachOp*        get_available_index_detach_op(void);
      PointDetachOp*        get_available_point_detach_op(void);
      TimingOp*             get_available_timing_op(void);
      TunableOp*            get_available_tunable_op(void);
      AllReduceOp*          get_available_all_reduce_op(void);
    public: // Control replication operations
      ReplIndividualTask*   get_available_repl_individual_task(void);
      ReplIndexTask*        get_available_repl_index_task(void);
      ReplMergeCloseOp*     get_available_repl_merge_close_op(void);
      ReplVirtualCloseOp*   get_available_repl_virtual_close_op(void);
      ReplRefinementOp*     get_available_repl_refinement_op(void);
      ReplFillOp*           get_available_repl_fill_op(void);
      ReplIndexFillOp*      get_available_repl_index_fill_op(void);
      ReplDiscardOp*        get_available_repl_discard_op(void);
      ReplCopyOp*           get_available_repl_copy_op(void);
      ReplIndexCopyOp*      get_available_repl_index_copy_op(void);
      ReplDeletionOp*       get_available_repl_deletion_op(void);
      ReplPendingPartitionOp* get_available_repl_pending_partition_op(void);
      ReplDependentPartitionOp* get_available_repl_dependent_partition_op(void);
      ReplMustEpochOp*      get_available_repl_epoch_op(void);
      ReplTimingOp*         get_available_repl_timing_op(void);
      ReplTunableOp*        get_available_repl_tunable_op(void);
      ReplAllReduceOp*      get_available_repl_all_reduce_op(void);
      ReplFenceOp*          get_available_repl_fence_op(void);
      ReplMapOp*            get_available_repl_map_op(void);
      ReplAttachOp*         get_available_repl_attach_op(void);
      ReplIndexAttachOp*    get_available_repl_index_attach_op(void);
      ReplDetachOp*         get_available_repl_detach_op(void);
      ReplIndexDetachOp*    get_available_repl_index_detach_op(void);
      ReplAcquireOp*        get_available_repl_acquire_op(void);
      ReplReleaseOp*        get_available_repl_release_op(void);
      ReplTraceCaptureOp*   get_available_repl_capture_op(void);
      ReplTraceCompleteOp*  get_available_repl_trace_op(void);
      ReplTraceReplayOp*    get_available_repl_replay_op(void);
      ReplTraceBeginOp*     get_available_repl_begin_op(void);
      ReplTraceSummaryOp*   get_available_repl_summary_op(void);
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
      void free_refinement_op(RefinementOp *op);
      void free_advisement_op(AdvisementOp *op);
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
      void free_discard_op(DiscardOp *op);
      void free_attach_op(AttachOp *op);
      void free_index_attach_op(IndexAttachOp *op);
      void free_point_attach_op(PointAttachOp *op);
      void free_detach_op(DetachOp *op);
      void free_index_detach_op(IndexDetachOp *op);
      void free_point_detach_op(PointDetachOp *op);
      void free_timing_op(TimingOp *op);
      void free_tunable_op(TunableOp *op);
      void free_all_reduce_op(AllReduceOp *op);
    public: // Control replication operations
      void free_repl_individual_task(ReplIndividualTask *task);
      void free_repl_index_task(ReplIndexTask *task);
      void free_repl_merge_close_op(ReplMergeCloseOp *op);
      void free_repl_virtual_close_op(ReplVirtualCloseOp *op);
      void free_repl_refinement_op(ReplRefinementOp *op);
      void free_repl_fill_op(ReplFillOp *op);
      void free_repl_index_fill_op(ReplIndexFillOp *op);
      void free_repl_discard_op(ReplDiscardOp *op);
      void free_repl_copy_op(ReplCopyOp *op);
      void free_repl_index_copy_op(ReplIndexCopyOp *op);
      void free_repl_deletion_op(ReplDeletionOp *op);
      void free_repl_pending_partition_op(ReplPendingPartitionOp *op);
      void free_repl_dependent_partition_op(ReplDependentPartitionOp *op);
      void free_repl_epoch_op(ReplMustEpochOp *op);
      void free_repl_timing_op(ReplTimingOp *op);
      void free_repl_tunable_op(ReplTunableOp *op);
      void free_repl_all_reduce_op(ReplAllReduceOp *op);
      void free_repl_fence_op(ReplFenceOp *op);
      void free_repl_map_op(ReplMapOp *op);
      void free_repl_attach_op(ReplAttachOp *op);
      void free_repl_index_attach_op(ReplIndexAttachOp *op);
      void free_repl_detach_op(ReplDetachOp *op);
      void free_repl_index_detach_op(ReplIndexDetachOp *op);
      void free_repl_acquire_op(ReplAcquireOp *op);
      void free_repl_release_op(ReplReleaseOp *op);
      void free_repl_capture_op(ReplTraceCaptureOp *op);
      void free_repl_trace_op(ReplTraceCompleteOp *op);
      void free_repl_replay_op(ReplTraceReplayOp *op);
      void free_repl_begin_op(ReplTraceBeginOp *op);
      void free_repl_summary_op(ReplTraceSummaryOp *op);
    public:
      RegionTreeContext allocate_region_tree_context(void);
      void free_region_tree_context(RegionTreeContext tree_ctx); 
      inline AddressSpaceID get_runtime_owner(UniqueID uid) const
        { return (uid % total_address_spaces); } 
    public:
      bool is_local(Processor proc) const;
      bool is_visible_memory(Processor proc, Memory mem);
      void find_visible_memories(Processor proc, std::set<Memory> &visible);
      Memory find_local_memory(Processor proc, Memory::Kind mem_kind);
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
#ifdef LEGION_SPY
      unsigned           get_unique_indirections_id(void);
#endif
    public:
      // Verify that a region requirement is valid
      LegionErrorType verify_requirement(const RegionRequirement &req,
                                         FieldID &bad_field);
    public:
      // Methods for helping with dumb nested class scoping problems
      IndexSpace help_create_index_space_handle(TypeTag type_tag);
    public:
      unsigned generate_random_integer(void);
#ifdef LEGION_TRACE_ALLOCATION
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
      static void application_processor_runtime_task(
                          const void *args, size_t arglen, 
			  const void *userdata, size_t userlen,
			  Processor p);
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
      unsigned total_outstanding_tasks;
#else
      std::atomic<unsigned> total_outstanding_tasks;
#endif
      std::atomic<unsigned> outstanding_top_level_tasks;
#ifdef DEBUG_SHUTDOWN_HANG
    public:
      std::vector<std::atomic<int> > outstanding_counts;
#endif
      // To support concurrent index task launches we need to have a
      // global reservation that any node can ask for in order to know
      // that it is safe to perform collective analysis. This reservation
      // is made on demand on node 0 and gradually spread to other nodes
      std::atomic<Reservation> concurrent_reservation;
    public:
      // Internal runtime state 
      // The local processor managed by this runtime
      const std::set<Processor> local_procs;
    protected:
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
      std::atomic<MessageManager*> message_managers[LEGION_MAX_NUM_NODES];
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
      std::atomic<unsigned> unique_index_space_id;
      std::atomic<unsigned> unique_index_partition_id;
      std::atomic<unsigned> unique_field_space_id;
      std::atomic<unsigned> unique_index_tree_id;
      std::atomic<unsigned> unique_region_tree_id;
      std::atomic<unsigned> unique_field_id; 
      std::atomic<unsigned long long> unique_operation_id;
      std::atomic<unsigned long long> unique_code_descriptor_id;
      std::atomic<unsigned long long> unique_constraint_id;
      std::atomic<unsigned long long> unique_is_expr_id;
#ifdef LEGION_SPY
      std::atomic<unsigned> unique_indirections_id;
#endif
      std::atomic<unsigned> unique_task_id;
      std::atomic<unsigned> unique_mapper_id;
      std::atomic<unsigned> unique_trace_id;
      std::atomic<unsigned> unique_projection_id;
      std::atomic<unsigned> unique_sharding_id;
      std::atomic<unsigned> unique_redop_id;
      std::atomic<unsigned> unique_serdez_id;
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
      struct LibraryShardingIDs {
      public:
        ShardingID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string,LibraryShardingIDs> library_sharding_ids;
      // This is only valid on node 0
      unsigned unique_library_sharding_id;
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
      mutable LocalLock callback_lock;
#ifdef LEGION_USE_LIBDL
      // Have this be a member variable so that it keeps references
      // to all the dynamic objects that we load
      Realm::DSOCodeTranslator callback_translator;
#endif
      std::map<void*,RtEvent> local_callbacks_done;
    public:
      struct RegistrationKey {
        inline RegistrationKey(void) : tag(0) { }
        inline RegistrationKey(size_t t, const std::string &dso, 
                               const std::string &symbol)
          : tag(t), dso_name(dso), symbol_name(symbol) { }
        inline bool operator<(const RegistrationKey &rhs) const
        { 
          if (tag < rhs.tag) return true;
          if (tag > rhs.tag) return false;
          if (dso_name < rhs.dso_name) return true;
          if (dso_name > rhs.dso_name) return false;
          return symbol_name < rhs.symbol_name; 
        }
        size_t tag;
        std::string dso_name;
        std::string symbol_name;
      };
    protected:
      std::map<RegistrationKey,RtEvent>                global_callbacks_done;
      std::map<RegistrationKey,RtEvent>                global_local_done;
      std::map<RegistrationKey,std::set<RtUserEvent> > pending_remote_callbacks;
    protected:
      mutable LocalLock redop_lock;
      std::map<ReductionOpID,FillView*> redop_fill_views;
      mutable LocalLock serdez_lock;
    protected:
      mutable LocalLock projection_lock;
      std::map<ProjectionID,ProjectionFunction*> projection_functions;
    protected:
      mutable LocalLock sharding_lock;
      std::map<ShardingID,ShardingFunctor*> sharding_functors;
    protected:
      mutable LocalLock group_lock;
      LegionMap<uint64_t,LegionDeque<ProcessorGroupInfo>,
                PROCESSOR_GROUP_ALLOC> processor_groups;
    protected:
      mutable LocalLock processor_mapping_lock;
      std::map<Processor,unsigned> processor_mapping;
    protected:
      std::atomic<DistributedID> unique_distributed_id;
    protected:
      mutable LocalLock distributed_collectable_lock;
      LegionMap<DistributedID,DistributedCollectable*,
                RUNTIME_DIST_COLLECT_ALLOC> dist_collectables;
      std::map<DistributedID,
        std::pair<DistributedCollectable*,RtUserEvent> > pending_collectables;
    protected:
      mutable LocalLock is_slice_lock;
      std::map<std::pair<Domain,TypeTag>,IndexSpace> index_slice_spaces;
    protected:
      // The runtime keeps track of remote contexts so they
      // can be re-used by multiple tasks that get sent remotely
      mutable LocalLock context_lock;
      unsigned total_contexts;
      std::deque<RegionTreeContext> available_contexts;
    protected:
      // Keep track of managers for control replication execution
      mutable LocalLock shard_lock;
      std::map<TaskID,ImplicitShardManager*> implicit_shard_managers;
    protected:
      // For generating random numbers
      mutable LocalLock random_lock;
      unsigned short random_state[3];
#ifdef LEGION_TRACE_ALLOCATION
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
      std::atomic<unsigned long long> allocation_tracing_count;
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
      mutable LocalLock refinement_op_lock;
      mutable LocalLock advisement_op_lock;
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
      mutable LocalLock discard_op_lock;
      mutable LocalLock attach_op_lock;
      mutable LocalLock detach_op_lock;
      mutable LocalLock timing_op_lock;
      mutable LocalLock tunable_op_lock;
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
      std::deque<RefinementOp*>         available_refinement_ops;
      std::deque<AdvisementOp*>         available_advisement_ops;
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
      std::deque<DiscardOp*>            available_discard_ops;
      std::deque<AttachOp*>             available_attach_ops;
      std::deque<IndexAttachOp*>        available_index_attach_ops;
      std::deque<PointAttachOp*>        available_point_attach_ops;
      std::deque<DetachOp*>             available_detach_ops;
      std::deque<IndexDetachOp*>        available_index_detach_ops;
      std::deque<PointDetachOp*>        available_point_detach_ops;
      std::deque<TimingOp*>             available_timing_ops;
      std::deque<TunableOp*>            available_tunable_ops;
      std::deque<AllReduceOp*>          available_all_reduce_ops;
    protected: // Control replication operations
      std::deque<ReplIndividualTask*>   available_repl_individual_tasks;
      std::deque<ReplIndexTask*>        available_repl_index_tasks;
      std::deque<ReplMergeCloseOp*>     available_repl_merge_close_ops;
      std::deque<ReplVirtualCloseOp*>   available_repl_virtual_close_ops;
      std::deque<ReplRefinementOp*>     available_repl_refinement_ops;
      std::deque<ReplFillOp*>           available_repl_fill_ops;
      std::deque<ReplIndexFillOp*>      available_repl_index_fill_ops;
      std::deque<ReplDiscardOp*>        available_repl_discard_ops;
      std::deque<ReplCopyOp*>           available_repl_copy_ops;
      std::deque<ReplIndexCopyOp*>      available_repl_index_copy_ops;
      std::deque<ReplDeletionOp*>       available_repl_deletion_ops;
      std::deque<ReplPendingPartitionOp*> 
                                        available_repl_pending_partition_ops;
      std::deque<ReplDependentPartitionOp*> 
                                        available_repl_dependent_partition_ops;
      std::deque<ReplMustEpochOp*>      available_repl_must_epoch_ops;
      std::deque<ReplTimingOp*>         available_repl_timing_ops;
      std::deque<ReplTunableOp*>        available_repl_tunable_ops;
      std::deque<ReplAllReduceOp*>      available_repl_all_reduce_ops;
      std::deque<ReplFenceOp*>          available_repl_fence_ops;
      std::deque<ReplMapOp*>            available_repl_map_ops;
      std::deque<ReplAttachOp*>         available_repl_attach_ops;
      std::deque<ReplIndexAttachOp*>    available_repl_index_attach_ops;
      std::deque<ReplDetachOp*>         available_repl_detach_ops;
      std::deque<ReplIndexDetachOp*>    available_repl_index_detach_ops;
      std::deque<ReplAcquireOp*>        available_repl_acquire_ops;
      std::deque<ReplReleaseOp*>        available_repl_release_ops;
      std::deque<ReplTraceCaptureOp*>   available_repl_capture_ops;
      std::deque<ReplTraceCompleteOp*>  available_repl_trace_ops;
      std::deque<ReplTraceReplayOp*>    available_repl_replay_ops;
      std::deque<ReplTraceBeginOp*>     available_repl_begin_ops;
      std::deque<ReplTraceSummaryOp*>   available_repl_summary_ops;
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
          LayoutConstraintID id, DistributedID did = 0,
          CollectiveMapping *collective_mapping = NULL);
      LayoutConstraints* register_layout(FieldSpace handle,
               const LayoutConstraintSet &cons, bool internal);
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
                                                 bool can_fail = false,
                                                 RtEvent *wait_for = NULL);
    public:
      // Static methods for start-up and callback phases
      static int start(int argc, char **argv, bool background, bool def_mapper);
      static void register_builtin_reduction_operators(void);
      static const LegionConfiguration& initialize(int *argc, char ***argv, 
                                                   bool filter);
      static LegionConfiguration parse_arguments(int argc, char **argv);
      static void perform_slow_config_checks(const LegionConfiguration &config);
      static void configure_interoperability(bool separate_runtimes);
      static RtEvent configure_runtime(int argc, char **argv,
          const LegionConfiguration &config, RealmRuntime &realm,
          Processor::Kind &startup_kind, bool background, bool default_mapper);
      static int wait_for_shutdown(void);
      static void set_return_code(int return_code);
      Future launch_top_level_task(const TaskLauncher &launcher);
      IndividualTask* create_implicit_top_level(TaskID top_task_id,
                                                MapperID top_mapper_id,
                                                Processor proxy,
                                                const char *task_name);
      ImplicitShardManager* find_implicit_shard_manager(TaskID top_task_id,
                                                MapperID top_mapper_id,
                                                Processor::Kind kind,
                                                unsigned shards_per_space,
                                                bool local);
      Context begin_implicit_task(TaskID top_task_id,
                                  MapperID top_mapper_id,
                                  Processor::Kind proc_kind,
                                  const char *task_name,
                                  bool control_replicable,
                                  unsigned shard_per_address_space,
                                  int shard_id, const DomainPoint &point);
      void unbind_implicit_task_from_external_thread(Context ctx);
      void bind_implicit_task_to_external_thread(Context ctx);
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
      static void add_registration_callback(RegistrationCallbackFnptr callback,
                                            bool dedup, size_t dedup_tag);
      static void add_registration_callback(
       RegistrationWithArgsCallbackFnptr callback, const UntypedBuffer &buffer,
                                            bool dedup, size_t dedup_tag);
      static void perform_dynamic_registration_callback(
                               RegistrationCallbackFnptr callback, bool global,
                               bool deduplicate, size_t dedup_tag);
      static void perform_dynamic_registration_callback(
                               RegistrationWithArgsCallbackFnptr callback,
                               const UntypedBuffer &buffer, bool global,
                               bool deduplicate, size_t dedup_tag);
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
      static std::map<ShardingID,ShardingFunctor*>&
                                get_pending_sharding_table(void);
      static std::vector<LegionHandshake>&
                                get_pending_handshake_table(void);
      struct RegistrationCallback {
        union {
          RegistrationCallbackFnptr withoutargs;
          RegistrationWithArgsCallbackFnptr withargs;
        } callback;
        UntypedBuffer buffer;
        size_t dedup_tag;
        bool deduplicate;
        bool has_args;
      };
      static std::vector<RegistrationCallback>&
                                get_pending_registration_callbacks(void);
      static TaskID& get_current_static_task_id(void);
      static TaskID generate_static_task_id(void);
      static VariantID preregister_variant(
                      const TaskVariantRegistrar &registrar,
                      const void *user_data, size_t user_data_size,
                      const CodeDescriptor &realm_desc, size_t return_type_size,
                      bool has_return_type_size, const char *task_name,
                      VariantID vid, bool check_id = true);
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
#if defined(LEGION_PRIVILEGE_CHECKS) || defined(LEGION_BOUNDS_CHECKS)
    public:
      static const char* find_privilege_task_name(void *impl);
#endif
#ifdef LEGION_BOUNDS_CHECKS
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
      static std::atomic<int> background_waits;
      // Shutdown error condition
      static int return_code;
      // Static member variables for MPI interop
      static int mpi_rank;
    public:
      static inline ApEvent merge_events(const TraceInfo *info,
                                         ApEvent e1, ApEvent e2);
      static inline ApEvent merge_events(const TraceInfo *info,
                                         ApEvent e1, ApEvent e2, ApEvent e3);
      static inline ApEvent merge_events(const TraceInfo *info,
                                         const std::set<ApEvent> &events);
      static inline ApEvent merge_events(const TraceInfo *info,
                                         const std::vector<ApEvent> &events);
    public:
      static inline RtEvent merge_events(RtEvent e1, RtEvent e2);
      static inline RtEvent merge_events(RtEvent e1, RtEvent e2, RtEvent e3);
      static inline RtEvent merge_events(const std::set<RtEvent> &events);
      static inline RtEvent merge_events(const std::vector<RtEvent> &events);
    public:
      static inline ApUserEvent create_ap_user_event(const TraceInfo *info);
      static inline void trigger_event(const TraceInfo *info, 
          ApUserEvent to_trigger, ApEvent precondition = ApEvent::NO_AP_EVENT);
      static inline void poison_event(ApUserEvent to_poison);
    public:
      static inline RtUserEvent create_rt_user_event(void);
      static inline void trigger_event(RtUserEvent to_trigger,
                                   RtEvent precondition = RtEvent::NO_RT_EVENT);
      static inline void poison_event(RtUserEvent to_poison);
    public:
      static inline PredUserEvent create_pred_event(void);
      static inline void trigger_event(PredUserEvent to_trigger);
      static inline void poison_event(PredUserEvent to_poison);
      static inline PredEvent merge_events(const TraceInfo *info,
                                           PredEvent e1, PredEvent e2);
    public:
      static inline ApEvent ignorefaults(ApEvent e);
      static inline RtEvent protect_event(ApEvent to_protect);
      static inline RtEvent protect_merge_events(
                                          const std::set<ApEvent> &events);
    public:
      static inline ApBarrier get_previous_phase(const PhaseBarrier &bar);
      static inline void phase_barrier_arrive(const PhaseBarrier &bar, 
                unsigned cnt, ApEvent precondition = ApEvent::NO_AP_EVENT,
                const void *reduce_value = NULL, size_t reduce_value_size = 0);
      static inline void advance_barrier(PhaseBarrier &bar);
      static inline void alter_arrival_count(PhaseBarrier &bar, int delta);
    public:
      static inline ApBarrier get_previous_phase(const ApBarrier &bar);
      static inline void phase_barrier_arrive(const ApBarrier &bar, 
                unsigned cnt, ApEvent precondition = ApEvent::NO_AP_EVENT,
                const void *reduce_value = NULL, size_t reduce_value_size = 0);
      static inline void advance_barrier(ApBarrier &bar);
      static inline bool get_barrier_result(ApBarrier bar, void *result,
                                            size_t result_size);
    public:
      static inline RtBarrier get_previous_phase(const RtBarrier &bar);
      static inline void phase_barrier_arrive(const RtBarrier &bar,
                unsigned cnt, RtEvent precondition = RtEvent::NO_RT_EVENT,
                const void *reduce_value = NULL, size_t reduce_value_size = 0);
      static inline void advance_barrier(RtBarrier &bar);
      static inline bool get_barrier_result(RtBarrier bar, void *result,
                                            size_t result_size);
      static inline void alter_arrival_count(RtBarrier &bar, int delta);
    public:
      static inline ApEvent acquire_ap_reservation(Reservation r,bool exclusive,
                                   ApEvent precondition = ApEvent::NO_AP_EVENT);
      static inline RtEvent acquire_rt_reservation(Reservation r,bool exclusive,
                                   RtEvent precondition = RtEvent::NO_RT_EVENT);
      static inline void release_reservation(Reservation r,
                                   LgEvent precondition = LgEvent::NO_LG_EVENT);
    };

    // This is a small helper class for converting realm index spaces when
    // the types don't naturally align with the underlying index space type
    template<int DIM, typename TYPELIST>
    struct RealmSpaceConverter {
      static inline void convert_to(const Domain &domain, void *realm_is, 
                                    const TypeTag type_tag, const char *context)
      {
        // Compute the type tag for this particular type with the same DIM
        const TypeTag tag =
          NT_TemplateHelper::encode_tag<DIM,typename TYPELIST::HEAD>();
        if (tag == type_tag)
        {
          Realm::IndexSpace<DIM,typename TYPELIST::HEAD> *target =
            static_cast<Realm::IndexSpace<DIM,typename TYPELIST::HEAD>*>(
                                                                realm_is);
          *target = domain;
        }
        else
          RealmSpaceConverter<DIM,typename TYPELIST::TAIL>::convert_to(domain,
                                                  realm_is, type_tag, context);
      }
    };

    // Specialization for end-of-list cases
    template<int DIM>
    struct RealmSpaceConverter<DIM,Realm::DynamicTemplates::TypeListTerm> {
      static inline void convert_to(const Domain &domain, void *realm_is, 
                                    const TypeTag type_tag, const char *context)
      {
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
          "Dynamic type mismatch in '%s'", context)
      }
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
      {
#ifdef LEGION_TRACE_ALLOCATION
        HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
        void *ptr = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
        result = new(ptr) T(this);
      }
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename WRAP>
    inline T* Runtime::get_available(LocalLock &local_lock, 
                                     std::deque<T*> &queue)
    //--------------------------------------------------------------------------
    {
      static_assert(sizeof(T) == sizeof(WRAP), "wrapper sizes should match");
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
      {
#ifdef LEGION_TRACE_ALLOCATION
        HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
        void *ptr = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
        result = new(ptr) WRAP(this);
      }
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Runtime::free_available(std::deque<T*> &queue)
    //--------------------------------------------------------------------------
    {
      for (typename std::deque<T*>::const_iterator it = 
            queue.begin(); it != queue.end(); it++)
      {
#ifdef LEGION_TRACE_ALLOCATION
        HandleAllocation<T,HasAllocType<T>::value>::trace_free();
#endif
        // Do explicit deletion to keep valgrind happy
        (*it)->~T();
        free(*it);
      }
      queue.clear();
    }

    //--------------------------------------------------------------------------
    template<bool CAN_BE_DELETED, typename T>
    inline void Runtime::release_operation(std::deque<T*> &queue, T* operation)
    //--------------------------------------------------------------------------
    {
      if (CAN_BE_DELETED && (queue.size() == LEGION_MAX_RECYCLABLE_OBJECTS))
      {
#ifdef LEGION_TRACE_ALLOCATION
        HandleAllocation<T,HasAllocType<T>::value>::trace_free();
#endif
        // Do explicit deletion to keep valgrind happy
        operation->~T();
        free(operation);
      }
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
      if (T::TASK_ID < LG_BEGIN_SHUTDOWN_TASK_IDS)
        increment_total_outstanding_tasks(args.lg_task_id, true/*meta*/);
#else
      if (T::TASK_ID < LG_BEGIN_SHUTDOWN_TASK_IDS)
        increment_total_outstanding_tasks();
#endif
#ifdef DEBUG_SHUTDOWN_HANG
      outstanding_counts[T::TASK_ID].fetch_add(1);
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
      if ((T::TASK_ID < LG_BEGIN_SHUTDOWN_TASK_IDS) && (profiler != NULL))
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
    template<typename T>
    inline RtEvent Runtime::issue_application_processor_task(
                                 const LgTaskArgs<T> &args, LgPriority priority,
                                 const Processor target, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(T::TASK_ID < LG_BEGIN_SHUTDOWN_TASK_IDS,
          "Shutdown tasks should never be run directly on application procs");
      // If this is not a task directly related to shutdown or is a message, 
      // to a remote node then increment the number of outstanding tasks
#ifdef DEBUG_LEGION
      assert(target.exists());
      assert(target.kind() != Processor::UTIL_PROC);
      increment_total_outstanding_tasks(args.lg_task_id, true/*meta*/);
#else
      increment_total_outstanding_tasks();
#endif
#ifdef DEBUG_SHUTDOWN_HANG
      outstanding_counts[T::TASK_ID].fetch_add(1);
#endif
      DETAILED_PROFILER(this, REALM_SPAWN_META_CALL);
      if (profiler != NULL)
      {
        Realm::ProfilingRequestSet requests;
        profiler->add_meta_request(requests, T::TASK_ID, args.provenance);
#ifdef LEGION_SEPARATE_META_TASKS
        return RtEvent(target.spawn(LG_APP_PROC_TASK_ID + T::TASK_ID, &args,
                              sizeof(T), requests, precondition, priority));
#else
        return RtEvent(target.spawn(LG_APP_PROC_TASK_ID, &args, sizeof(T),
                                    requests, precondition, priority));
#endif
      }
      else
#ifdef LEGION_SEPARATE_META_TASKS
        return RtEvent(target.spawn(LG_APP_PROC_TASK_ID + T::TASK_ID, &args,
                                    sizeof(T), precondition, priority));
#else
        return RtEvent(target.spawn(LG_APP_PROC_TASK_ID, &args, sizeof(T), 
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
      // Fuck C++
      const std::set<ApEvent> *legion_events = &events;
      const std::set<Realm::Event> *realm_events;
      static_assert(sizeof(legion_events) == sizeof(realm_events), "Fuck C++");
      memcpy(&realm_events, &legion_events, sizeof(legion_events));
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
    /*static*/ inline ApEvent Runtime::merge_events(
                      const TraceInfo *info, const std::vector<ApEvent> &events)
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
          ApEvent result = events.front();
          info->record_merge_events(result, events);
          return result;
        }
        else
          return events.front();
      }
#endif
      // Fuck C++
      const std::vector<ApEvent> *legion_events = &events;
      const std::vector<Realm::Event> *realm_events;
      static_assert(sizeof(legion_events) == sizeof(realm_events), "Fuck C++");
      memcpy(&realm_events, &legion_events, sizeof(legion_events));
      ApEvent result(Realm::Event::merge_events(*realm_events));
#ifdef LEGION_DISABLE_EVENT_PRUNING 
      if (!result.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        result = ApEvent(rename);
      }
      else
      {
        // Check to make sure it isn't a rename
        for (unsigned idx = 0; idx < events.size(); idx++)
        {
          if (events[idx] != result)
            continue;
          Realm::UserEvent rename(Realm::UserEvent::create_user_event());
          rename.trigger(result);
          result = ApEvent(rename);
          break;
        }
      }
#endif
#ifdef LEGION_SPY
      for (std::vector<ApEvent>::const_iterator it = events.begin();
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
      // Fuck C++
      const std::set<RtEvent> *legion_events = &events;
      const std::set<Realm::Event> *realm_events;
      static_assert(sizeof(legion_events) == sizeof(realm_events), "Fuck C++");
      memcpy(&realm_events, &legion_events, sizeof(legion_events));
      // No logging for runtime operations currently
      return RtEvent(Realm::Event::merge_events(*realm_events));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtEvent Runtime::merge_events(
                                             const std::vector<RtEvent> &events)
    //--------------------------------------------------------------------------
    {
#ifndef LEGION_DISABLE_EVENT_PRUNING
      if (events.empty())
        return RtEvent::NO_RT_EVENT;
      if (events.size() == 1)
        return events.front();
#endif
      // Fuck C++
      const std::vector<RtEvent> *legion_events = &events;
      const std::vector<Realm::Event> *realm_events;
      static_assert(sizeof(legion_events) == sizeof(realm_events), "Fuck C++");
      memcpy(&realm_events, &legion_events, sizeof(legion_events));
      // No logging for runtime operations currently
      return RtEvent(Realm::Event::merge_events(*realm_events));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApUserEvent Runtime::create_ap_user_event(
                                                          const TraceInfo *info)
    //--------------------------------------------------------------------------
    {
      ApUserEvent result;
      if ((info == NULL) || !info->recording)
      {
        result = ApUserEvent(Realm::UserEvent::create_user_event());
#ifdef LEGION_SPY
        LegionSpy::log_ap_user_event(result);
#endif
      }
      else
        info->record_create_ap_user_event(result);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::trigger_event(const TraceInfo *info,
                                   ApUserEvent to_trigger, ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      Realm::UserEvent copy = to_trigger;
      copy.trigger(precondition);
#ifdef LEGION_SPY
      LegionSpy::log_ap_user_event_trigger(to_trigger);
      if (precondition.exists())
        LegionSpy::log_event_dependence(precondition, to_trigger);
#endif
      if ((info != NULL) && info->recording)
        info->record_trigger_event(to_trigger, precondition);
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
    /*static*/ inline PredUserEvent Runtime::create_pred_event(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      PredUserEvent result(Realm::UserEvent::create_user_event());
      LegionSpy::log_pred_event(result);
      return result;
#else
      return PredUserEvent(Realm::UserEvent::create_user_event());
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::trigger_event(PredUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      Realm::UserEvent copy = to_trigger;
      copy.trigger();
#ifdef LEGION_SPY
      LegionSpy::log_pred_event_trigger(to_trigger);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::poison_event(PredUserEvent to_poison)
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
    /*static*/ inline PredEvent Runtime::merge_events(
                              const TraceInfo *info, PredEvent e1, PredEvent e2)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(e1.exists());
      assert(e2.exists());
#endif
      PredEvent result(Realm::Event::merge_events(e1, e2));
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
        result = PredEvent(rename);
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
    /*static*/ inline ApEvent Runtime::ignorefaults(ApEvent e)
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
      return result;
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
      const std::set<ApEvent> *ptr = &events;
      const std::set<Realm::Event> *realm_events = NULL;
      static_assert(sizeof(realm_events) == sizeof(ptr), "Fuck c++");
      memcpy(&realm_events, &ptr, sizeof(realm_events));
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
    /*static*/ inline ApBarrier Runtime::get_previous_phase(
                                                           const ApBarrier &bar)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      return ApBarrier(copy.get_previous_phase());
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::phase_barrier_arrive(
                  const ApBarrier &bar, unsigned count, ApEvent precondition,
                  const void *reduce_value, size_t reduce_value_size)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      copy.arrive(count, precondition, reduce_value, reduce_value_size);
#ifdef LEGION_SPY
      if (precondition.exists())
        LegionSpy::log_event_dependence(precondition, bar);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::advance_barrier(ApBarrier &bar)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      bar = ApBarrier(copy.advance_barrier());
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
    /*static*/ inline RtBarrier Runtime::get_previous_phase(const RtBarrier &b)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = b;
      return RtBarrier(copy.get_previous_phase());
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::phase_barrier_arrive(const RtBarrier &bar,
           unsigned count, RtEvent precondition, const void *value, size_t size)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      copy.arrive(count, precondition, value, size); 
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::advance_barrier(RtBarrier &bar)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      bar = RtBarrier(copy.advance_barrier());
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool Runtime::get_barrier_result(RtBarrier bar, 
                                               void *result, size_t result_size)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      return copy.get_result(result, result_size);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::alter_arrival_count(RtBarrier &b, int delta)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = b;
      b = RtBarrier(copy.alter_arrival_count(delta));
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
      LegionSpy::log_reservation_acquire(r, precondition, result);
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

    //--------------------------------------------------------------------------
    /*static*/ inline VirtualChannelKind MessageManager::find_message_vc(
                                                               MessageKind kind)
    //--------------------------------------------------------------------------
    {
      switch (kind)
      {
        case TASK_MESSAGE:
          return TASK_VIRTUAL_CHANNEL;
        case STEAL_MESSAGE:
          return MAPPER_VIRTUAL_CHANNEL;
        case ADVERTISEMENT_MESSAGE:
          return MAPPER_VIRTUAL_CHANNEL;
        case SEND_REGISTRATION_CALLBACK:
          break;
        case SEND_REMOTE_TASK_REPLAY:
          break;
        case SEND_REMOTE_TASK_PROFILING_RESPONSE:
          break;
        case SEND_SHARED_OWNERSHIP:
          break;
        case SEND_INDEX_SPACE_REQUEST:
          break;
        case SEND_INDEX_SPACE_RESPONSE:
          return INDEX_SPACE_VIRTUAL_CHANNEL;
        case SEND_INDEX_SPACE_RETURN:
          return INDEX_SPACE_VIRTUAL_CHANNEL;
        case SEND_INDEX_SPACE_SET:
          break;
        case SEND_INDEX_SPACE_CHILD_REQUEST:
          break;
        case SEND_INDEX_SPACE_CHILD_RESPONSE:
          break;
        case SEND_INDEX_SPACE_COLORS_REQUEST:
          break;
        case SEND_INDEX_SPACE_COLORS_RESPONSE:
          break;
        case SEND_INDEX_SPACE_REMOTE_EXPRESSION_REQUEST:
          break;
        case SEND_INDEX_SPACE_REMOTE_EXPRESSION_RESPONSE:
          return EXPRESSION_VIRTUAL_CHANNEL;
        case SEND_INDEX_SPACE_GENERATE_COLOR_REQUEST:
          break;
        case SEND_INDEX_SPACE_GENERATE_COLOR_RESPONSE:
          break;
        case SEND_INDEX_SPACE_RELEASE_COLOR:
          break;
        case SEND_INDEX_PARTITION_NOTIFICATION:
          break;
        case SEND_INDEX_PARTITION_REQUEST:
          break;
        case SEND_INDEX_PARTITION_RESPONSE:
          return INDEX_SPACE_VIRTUAL_CHANNEL;
        case SEND_INDEX_PARTITION_RETURN:
          return INDEX_SPACE_VIRTUAL_CHANNEL;
        case SEND_INDEX_PARTITION_CHILD_REQUEST:
          break;
        case SEND_INDEX_PARTITION_CHILD_RESPONSE:
          break;
        case SEND_INDEX_PARTITION_DISJOINT_UPDATE:
          break;
        case SEND_INDEX_PARTITION_SHARD_RECTS_REQUEST:
          break;
        case SEND_INDEX_PARTITION_SHARD_RECTS_RESPONSE:
          break;
        case SEND_INDEX_PARTITION_REMOTE_INTERFERENCE_REQUEST:
          break;
        case SEND_INDEX_PARTITION_REMOTE_INTERFERENCE_RESPONSE:
          break;
        case SEND_FIELD_SPACE_NODE:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_REQUEST:
          break;
        case SEND_FIELD_SPACE_RETURN:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_ALLOCATOR_REQUEST:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_ALLOCATOR_RESPONSE:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_ALLOCATOR_INVALIDATION:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_ALLOCATOR_FLUSH:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_ALLOCATOR_FREE:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_INFOS_REQUEST:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_INFOS_RESPONSE:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_ALLOC_REQUEST:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SIZE_UPDATE:
          break;
        case SEND_FIELD_FREE:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_FREE_INDEXES:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_LAYOUT_INVALIDATION:
          break;
        case SEND_LOCAL_FIELD_ALLOC_REQUEST:
          break;
        case SEND_LOCAL_FIELD_ALLOC_RESPONSE:
          break;
        case SEND_LOCAL_FIELD_FREE:
          break;
        case SEND_LOCAL_FIELD_UPDATE:
          break;
        case SEND_TOP_LEVEL_REGION_REQUEST:
          break;
        case SEND_TOP_LEVEL_REGION_RETURN:
          break;
        case INDEX_SPACE_DESTRUCTION_MESSAGE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case INDEX_PARTITION_DESTRUCTION_MESSAGE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case FIELD_SPACE_DESTRUCTION_MESSAGE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case LOGICAL_REGION_DESTRUCTION_MESSAGE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case INDIVIDUAL_REMOTE_FUTURE_SIZE:
          return TASK_VIRTUAL_CHANNEL;
        case INDIVIDUAL_REMOTE_COMPLETE:
          return TASK_VIRTUAL_CHANNEL;
        case INDIVIDUAL_REMOTE_COMMIT:
          return TASK_VIRTUAL_CHANNEL;
        case SLICE_REMOTE_MAPPED:
          return TASK_VIRTUAL_CHANNEL;
        case SLICE_REMOTE_COMPLETE:
          return TASK_VIRTUAL_CHANNEL;
        case SLICE_REMOTE_COMMIT:
          return TASK_VIRTUAL_CHANNEL;
        case SLICE_VERIFY_CONCURRENT_EXECUTION:
          break;
        case SLICE_FIND_INTRA_DEP:
          break;
        case SLICE_RECORD_INTRA_DEP:
          break;
        case SLICE_REMOTE_COLLECTIVE_RENDEZVOUS:
          break;
        case DISTRIBUTED_REMOTE_REGISTRATION:
          break;
        // Low priority so reference counting doesn't starve
        // out the rest of our work
        case DISTRIBUTED_DOWNGRADE_REQUEST:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case DISTRIBUTED_DOWNGRADE_RESPONSE:
          break;
        case DISTRIBUTED_DOWNGRADE_SUCCESS:
          break;
        // Put downgrade updates and acquire requests
        // on same ordered virtual channel so that 
        // acquire requests cannot starve out an owner
        // update while it is in flight by circling
        // around and around
        case DISTRIBUTED_DOWNGRADE_UPDATE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case DISTRIBUTED_GLOBAL_ACQUIRE_REQUEST:
          return REFERENCE_VIRTUAL_CHANNEL;
        case DISTRIBUTED_GLOBAL_ACQUIRE_RESPONSE:
          break;
        case DISTRIBUTED_VALID_ACQUIRE_REQUEST:
          break;
        case DISTRIBUTED_VALID_ACQUIRE_RESPONSE:
          break;
        case SEND_ATOMIC_RESERVATION_REQUEST:
          break;
        case SEND_ATOMIC_RESERVATION_RESPONSE:
          break;
        case SEND_PADDED_RESERVATION_REQUEST:
          break;
        case SEND_PADDED_RESERVATION_RESPONSE:
          break;
        case SEND_CREATED_REGION_CONTEXTS:
          break;
        case SEND_MATERIALIZED_VIEW:
          break;
        case SEND_FILL_VIEW:
          break;
        case SEND_FILL_VIEW_VALUE:
          break;
        case SEND_PHI_VIEW:
          break;
        case SEND_REDUCTION_VIEW:
          break;
        case SEND_REPLICATED_VIEW:
          break;
        case SEND_ALLREDUCE_VIEW:
          break;
        case SEND_INSTANCE_MANAGER:
          break;
        case SEND_MANAGER_UPDATE:
          break;
        // Only collective operations apply to destinations need to be
        // on the ordered virtual channel since they need to be ordered
        // with respect to the same CopyFillAggregator, there's no need
        // to do the same thing read-only collectives since they can 
        // never be read more than once by each CopyFillAggregator
        case SEND_COLLECTIVE_DISTRIBUTE_FILL:
          return COLLECTIVE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_DISTRIBUTE_POINT:
          break; // read-only
        case SEND_COLLECTIVE_DISTRIBUTE_POINTWISE:
          return COLLECTIVE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_DISTRIBUTE_REDUCTION:
          break; // read-only
        case SEND_COLLECTIVE_DISTRIBUTE_BROADCAST:
          return COLLECTIVE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_DISTRIBUTE_REDUCECAST:
          return COLLECTIVE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_DISTRIBUTE_HOURGLASS:
          return COLLECTIVE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_DISTRIBUTE_ALLREDUCE:
          break; // no views involved so effectively read-only
        case SEND_COLLECTIVE_HAMMER_REDUCTION:
          break; // read-only
        case SEND_COLLECTIVE_FUSE_GATHER:
          return COLLECTIVE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_USER_REQUEST:
          break;
        case SEND_COLLECTIVE_USER_RESPONSE:
          break;
        case SEND_COLLECTIVE_REGISTER_USER:
          break;
        case SEND_COLLECTIVE_REMOTE_INSTANCES_REQUEST:
          break;
        case SEND_COLLECTIVE_REMOTE_INSTANCES_RESPONSE:
          break;
        case SEND_COLLECTIVE_NEAREST_INSTANCES_REQUEST:
          break;
        case SEND_COLLECTIVE_NEAREST_INSTANCES_RESPONSE:
          break;
          // These messages need to be ordered with respect to
          // register_user messages so they go on the same VC
        case SEND_COLLECTIVE_REMOTE_REGISTRATION:
          return UPDATE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_FINALIZE_MAPPING:
          break;
        case SEND_COLLECTIVE_VIEW_CREATION:
          break;
        case SEND_COLLECTIVE_VIEW_DELETION:
          break;
        case SEND_COLLECTIVE_VIEW_RELEASE:
          break;
        case SEND_COLLECTIVE_VIEW_NOTIFICATION:
          break;
        // All these collective messages need to go on the same
        // virtual channel since they all need to be ordered 
        // with respect to each other
        case SEND_COLLECTIVE_VIEW_MAKE_VALID:
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_VIEW_MAKE_INVALID:
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_VIEW_INVALIDATE_REQUEST:
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_VIEW_INVALIDATE_RESPONSE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_VIEW_ADD_REMOTE_REFERENCE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_VIEW_REMOVE_REMOTE_REFERENCE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_CREATE_TOP_VIEW_REQUEST:
          break;
        case SEND_CREATE_TOP_VIEW_RESPONSE:
          break;
        case SEND_VIEW_REQUEST:
          break;
        case SEND_VIEW_REGISTER_USER:
          return UPDATE_VIRTUAL_CHANNEL;
        case SEND_VIEW_FIND_COPY_PRE_REQUEST:
          return UPDATE_VIRTUAL_CHANNEL;
        case SEND_VIEW_ADD_COPY_USER:
          return UPDATE_VIRTUAL_CHANNEL;
        case SEND_VIEW_FIND_LAST_USERS_REQUEST:
          break;
        case SEND_VIEW_FIND_LAST_USERS_RESPONSE:
          break;
        case SEND_VIEW_REPLICATION_REQUEST:
          return UPDATE_VIRTUAL_CHANNEL;
        case SEND_VIEW_REPLICATION_RESPONSE:
          return UPDATE_VIRTUAL_CHANNEL;
        case SEND_VIEW_REPLICATION_REMOVAL:
          return UPDATE_VIRTUAL_CHANNEL;
        case SEND_MANAGER_REQUEST:
          break;
        case SEND_FUTURE_RESULT:
          break;
        case SEND_FUTURE_RESULT_SIZE:
          break;
        case SEND_FUTURE_SUBSCRIPTION:
          break;
        case SEND_FUTURE_CREATE_INSTANCE_REQUEST:
          break;
        case SEND_FUTURE_CREATE_INSTANCE_RESPONSE:
          break;
        case SEND_FUTURE_MAP_REQUEST:
          break;
        case SEND_FUTURE_MAP_RESPONSE:
          break;
        case SEND_REPL_COMPUTE_EQUIVALENCE_SETS:
          break;
        case SEND_REPL_EQUIVALENCE_SET_NOTIFICATION:
          break;
        case SEND_REPL_INTRA_SPACE_DEP:
          break;
        case SEND_REPL_BROADCAST_UPDATE:
          break;
        case SEND_REPL_CREATED_REGIONS:
          break;
        case SEND_REPL_TRACE_EVENT_REQUEST:
          break;
        case SEND_REPL_TRACE_EVENT_RESPONSE:
          break;
        case SEND_REPL_TRACE_FRONTIER_REQUEST:
          break;
        case SEND_REPL_TRACE_FRONTIER_RESPONSE:
          break;
        case SEND_REPL_TRACE_UPDATE:
          break;
        case SEND_REPL_IMPLICIT_REQUEST:
          break;
        // This has to go on the task virtual channel so that it is ordered
        // with respect to any distributions
        // See Runtime::send_replicate_launch
        case SEND_REPL_IMPLICIT_RESPONSE:
          return TASK_VIRTUAL_CHANNEL;
        case SEND_REPL_FIND_COLLECTIVE_VIEW:
          break;
        case SEND_MAPPER_MESSAGE:
          return MAPPER_VIRTUAL_CHANNEL;
        case SEND_MAPPER_BROADCAST:
          return MAPPER_VIRTUAL_CHANNEL;
        case SEND_TASK_IMPL_SEMANTIC_REQ:
          break;
        case SEND_INDEX_SPACE_SEMANTIC_REQ:
          break;
        case SEND_INDEX_PARTITION_SEMANTIC_REQ:
          break;
        case SEND_FIELD_SPACE_SEMANTIC_REQ:
          break;
        case SEND_FIELD_SEMANTIC_REQ:
          break;
        case SEND_LOGICAL_REGION_SEMANTIC_REQ:
          break;
        case SEND_LOGICAL_PARTITION_SEMANTIC_REQ:
          break;
        case SEND_TASK_IMPL_SEMANTIC_INFO:
          break;
        case SEND_INDEX_SPACE_SEMANTIC_INFO:
          break;
        case SEND_INDEX_PARTITION_SEMANTIC_INFO:
          break;
        case SEND_FIELD_SPACE_SEMANTIC_INFO:
          break;
        case SEND_FIELD_SEMANTIC_INFO:
          break;
        case SEND_LOGICAL_REGION_SEMANTIC_INFO:
          break;
        case SEND_LOGICAL_PARTITION_SEMANTIC_INFO:
          break;
        case SEND_REMOTE_CONTEXT_REQUEST:
          break;
        case SEND_REMOTE_CONTEXT_RESPONSE:
          break;
        case SEND_REMOTE_CONTEXT_PHYSICAL_REQUEST:
          break;
        case SEND_REMOTE_CONTEXT_PHYSICAL_RESPONSE:
          break;
        case SEND_REMOTE_CONTEXT_FIND_COLLECTIVE_VIEW_REQUEST:
          break;
        case SEND_REMOTE_CONTEXT_FIND_COLLECTIVE_VIEW_RESPONSE:
          break;
        case SEND_COMPUTE_EQUIVALENCE_SETS_REQUEST:
          break;
        case SEND_COMPUTE_EQUIVALENCE_SETS_RESPONSE:
          break;
        case SEND_CANCEL_EQUIVALENCE_SETS_SUBSCRIPTION:
          break;
        case SEND_FINISH_EQUIVALENCE_SETS_SUBSCRIPTION:
          break;
        case SEND_EQUIVALENCE_SET_REQUEST:
          break;
        case SEND_EQUIVALENCE_SET_RESPONSE:
          break;
        case SEND_EQUIVALENCE_SET_REPLICATION_REQUEST:
          break;
        case SEND_EQUIVALENCE_SET_REPLICATION_RESPONSE:
          break;
        case SEND_EQUIVALENCE_SET_REPLICATION_INVALIDATION:
          break;
        case SEND_EQUIVALENCE_SET_MIGRATION:
          return MIGRATION_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_OWNER_UPDATE:
          return MIGRATION_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_MAKE_OWNER:
          break;
        case SEND_EQUIVALENCE_SET_CLONE_REQUEST:
          break;
        case SEND_EQUIVALENCE_SET_CLONE_RESPONSE:
          break;
        case SEND_EQUIVALENCE_SET_CAPTURE_REQUEST:
          break;
        case SEND_EQUIVALENCE_SET_CAPTURE_RESPONSE:
          break;
        case SEND_EQUIVALENCE_SET_REMOTE_REQUEST_INSTANCES:
          break;
        case SEND_EQUIVALENCE_SET_REMOTE_REQUEST_INVALID:
          break;
        case SEND_EQUIVALENCE_SET_REMOTE_REQUEST_ANTIVALID:
          break;
        case SEND_EQUIVALENCE_SET_REMOTE_UPDATES:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_REMOTE_ACQUIRES:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_REMOTE_RELEASES:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_REMOTE_COPIES_ACROSS:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_REMOTE_OVERWRITES:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_REMOTE_FILTERS:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_REMOTE_CLONES:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_REMOTE_INSTANCES:
          break;
        case SEND_INSTANCE_REQUEST:
          break;
        case SEND_INSTANCE_RESPONSE:
          break;
        case SEND_EXTERNAL_CREATE_REQUEST:
          break;
        case SEND_EXTERNAL_CREATE_RESPONSE:
          break;
        case SEND_EXTERNAL_ATTACH:
          break;
        case SEND_EXTERNAL_DETACH:
          break;
        case SEND_GC_PRIORITY_UPDATE:
          break;
        case SEND_GC_REQUEST:
          break;
        case SEND_GC_RESPONSE:
          break;
        case SEND_GC_ACQUIRE:
          break;
        case SEND_GC_FAILED:
          break;
        case SEND_GC_MISMATCH:
          break;
        case SEND_GC_NOTIFY:
          // This one goes on the resource virtual channel because there
          // is nothing else preventing the deletion of the managers
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_GC_DEBUG_REQUEST:
          break;
        case SEND_GC_DEBUG_RESPONSE:
          break;
        case SEND_GC_RECORD_EVENT:
          break;
        case SEND_ACQUIRE_REQUEST:
          break;
        case SEND_ACQUIRE_RESPONSE:
          break;
        case SEND_VARIANT_BROADCAST:
          break;
        case SEND_CONSTRAINT_REQUEST:
          return LAYOUT_CONSTRAINT_VIRTUAL_CHANNEL;
        case SEND_CONSTRAINT_RESPONSE:
          return LAYOUT_CONSTRAINT_VIRTUAL_CHANNEL;
        case SEND_CONSTRAINT_RELEASE:
          return LAYOUT_CONSTRAINT_VIRTUAL_CHANNEL;
        case SEND_TOP_LEVEL_TASK_REQUEST:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_TOP_LEVEL_TASK_COMPLETE:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_MPI_RANK_EXCHANGE:
          break;
        case SEND_REPLICATE_LAUNCH:
          return TASK_VIRTUAL_CHANNEL;
        case SEND_REPLICATE_POST_MAPPED:
          break;
        case SEND_REPLICATE_POST_EXECUTION:
          break;
        case SEND_REPLICATE_TRIGGER_COMPLETE:
          break;
        case SEND_REPLICATE_TRIGGER_COMMIT:
          break;
        case SEND_CONTROL_REPLICATE_COLLECTIVE_MESSAGE:
          break;
        // All rendezvous messages need to be ordered
        case SEND_CONTROL_REPLICATE_RENDEZVOUS_MESSAGE:
          return RENDEZVOUS_VIRTUAL_CHANNEL;
        case SEND_LIBRARY_MAPPER_REQUEST:
          break;
        case SEND_LIBRARY_MAPPER_RESPONSE:
          break;
        case SEND_LIBRARY_TRACE_REQUEST:
          break;
        case SEND_LIBRARY_TRACE_RESPONSE:
          break;
        case SEND_LIBRARY_PROJECTION_REQUEST:
          break;
        case SEND_LIBRARY_PROJECTION_RESPONSE:
          break;
        case SEND_LIBRARY_SHARDING_REQUEST:
          break;
        case SEND_LIBRARY_SHARDING_RESPONSE:
          break;
        case SEND_LIBRARY_TASK_REQUEST:
          break;
        case SEND_LIBRARY_TASK_RESPONSE:
          break;
        case SEND_LIBRARY_REDOP_REQUEST:
          break;
        case SEND_LIBRARY_REDOP_RESPONSE:
          break;
        case SEND_LIBRARY_SERDEZ_REQUEST:
          break;
        case SEND_LIBRARY_SERDEZ_RESPONSE:
          break;
        case SEND_REMOTE_OP_REPORT_UNINIT:
          break;
        case SEND_REMOTE_OP_PROFILING_COUNT_UPDATE:
          break;
        case SEND_REMOTE_OP_COMPLETION_EFFECT:
          break;
        case SEND_REMOTE_TRACE_UPDATE:
          return TRACING_VIRTUAL_CHANNEL;
        case SEND_REMOTE_TRACE_RESPONSE:
          break;
        case SEND_FREE_EXTERNAL_ALLOCATION:
          break;
        case SEND_CREATE_FUTURE_INSTANCE_REQUEST:
          break;
        case SEND_CREATE_FUTURE_INSTANCE_RESPONSE:
          break;
        case SEND_FREE_FUTURE_INSTANCE:
          break;
        case SEND_REMOTE_DISTRIBUTED_ID_REQUEST:
          break;
        case SEND_REMOTE_DISTRIBUTED_ID_RESPONSE:
          break;
        case SEND_CONCURRENT_RESERVATION_CREATION:
          break;
        case SEND_CONCURRENT_EXECUTION_ANALYSIS:
          break;
        case SEND_SHUTDOWN_NOTIFICATION:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_SHUTDOWN_RESPONSE:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case LAST_SEND_KIND:
          assert(false);
      }
      return DEFAULT_VIRTUAL_CHANNEL;
    }

  }; // namespace Internal 
}; // namespace Legion 

#endif // __RUNTIME_H__

// EOF

