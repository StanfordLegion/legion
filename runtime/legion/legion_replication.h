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

#ifndef __LEGION_REPLICATION_H__
#define __LEGION_REPLICATION_H__

#include "legion/legion_ops.h"
#include "legion/legion_tasks.h"
#include "legion/legion_trace.h"
#include "legion/legion_context.h"

namespace Legion {
  namespace Internal { 

#ifdef DEBUG_LEGION_COLLECTIVES
    /**
     * \class CollectiveCheckReduction
     * A small helper reduction for use with checking that 
     * Legion collectives are properly aligned across all shards
     */
    class CollectiveCheckReduction {
    public:
      typedef long RHS;
      typedef long LHS;
      static const long IDENTITY;
      static const long identity;
      static const long BAD;
      static const ReductionOpID REDOP;

      template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
      template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
    }; 

    /**
     * \class CloseCheckReduction
     * Another helper reduction for comparing the phase barriers
     * used by close operations which should be ordered
     */
    class CloseCheckReduction {
    public:
      struct CloseCheckValue {
      public:
        CloseCheckValue(void);
        CloseCheckValue(const LogicalUser &user, RtBarrier barrier,
                        RegionTreeNode *node, bool read_only);
      public:
        bool operator==(const CloseCheckValue &rhs) const;
      public:
        unsigned operation_index;
        unsigned region_requirement_index;
        RtBarrier barrier;
        LogicalRegion region;
        LogicalPartition partition;
        bool is_region;
        bool read_only;
      };
    public:
      typedef CloseCheckValue RHS;
      typedef CloseCheckValue LHS;
      static const CloseCheckValue IDENTITY;
      static const CloseCheckValue identity;
      static const ReductionOpID REDOP;

      template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
      template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
    };
#endif

    struct SelectShardingFunctorOutput :
      public Mapper::SelectShardingFunctorOutput {
      inline SelectShardingFunctorOutput(void)
        { chosen_functor = UINT_MAX; slice_recurse = true; }
    };

    /**
     * \class ShardCollective
     * The shard collective is the base class for performing
     * collective operations between shards
     */
    class ShardCollective {
    public:
      struct DeferCollectiveArgs : public LgTaskArgs<DeferCollectiveArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_COLLECTIVE_TASK_ID;
      public:
        DeferCollectiveArgs(ShardCollective *c)
          : LgTaskArgs(implicit_provenance), collective(c) { }
      public:
        ShardCollective *const collective;
      };
    public:
      ShardCollective(CollectiveIndexLocation loc, ReplicateContext *ctx);
      ShardCollective(ReplicateContext *ctx, CollectiveID id);
      virtual ~ShardCollective(void);
    public:
      virtual void perform_collective_async(
                     RtEvent precondition = RtEvent::NO_RT_EVENT) = 0;
      virtual RtEvent perform_collective_wait(bool block = false) = 0;
      virtual void handle_collective_message(Deserializer &derez) = 0;
      void perform_collective_sync(RtEvent pre = RtEvent::NO_RT_EVENT);
      static void handle_deferred_collective(const void *args);
    protected:
      bool defer_collective_async(RtEvent precondition);
      int convert_to_index(ShardID id, ShardID origin) const;
      ShardID convert_to_shard(int index, ShardID origin) const;
    public:
      ShardManager *const manager;
      ReplicateContext *const context;
      const ShardID local_shard;
      const CollectiveID collective_index;
    protected:
      mutable LocalLock collective_lock;
    };

    /**
     * \class BroadcastCollective
     * This shard collective has equivalent functionality to 
     * MPI Broadcast in that it will transmit some data on one
     * shard to all the other shards.
     */
    class BroadcastCollective : public ShardCollective {
    public:
      BroadcastCollective(CollectiveIndexLocation loc,
                          ReplicateContext *ctx, ShardID origin);
      BroadcastCollective(ReplicateContext *ctx, 
                          CollectiveID id, ShardID origin); 
      virtual ~BroadcastCollective(void);
    public:
      // We guarantee that these methods will be called atomically
      virtual void pack_collective(Serializer &rez) const = 0;
      virtual void unpack_collective(Deserializer &derez) = 0;
    public:
      virtual void perform_collective_async(RtEvent pre = RtEvent::NO_RT_EVENT);
      virtual RtEvent perform_collective_wait(bool block = true);
      virtual void handle_collective_message(Deserializer &derez);
      virtual RtEvent post_broadcast(void) { return RtEvent::NO_RT_EVENT; }
    public:
      RtEvent get_done_event(void) const;
      inline bool is_origin(void) const
        { return (origin == local_shard); }
    protected:
      void send_messages(void) const;
    public:
      const ShardID origin;
      const int shard_collective_radix;
    private:
      RtUserEvent done_event; // valid on all shards except origin
    };

    /**
     * \class GatherCollective
     * This shard collective has equivalent functionality to
     * MPI Gather in that it will ensure that data from all
     * the shards are reduced down to a single shard.
     */
    class GatherCollective : public ShardCollective {
    public:
      GatherCollective(CollectiveIndexLocation loc,
                       ReplicateContext *ctx, ShardID target);
      virtual ~GatherCollective(void);
    public:
      // We guarantee that these methods will be called atomically
      virtual void pack_collective(Serializer &rez) const = 0;
      virtual void unpack_collective(Deserializer &derez) = 0;
    public:
      virtual void perform_collective_async(RtEvent pre = RtEvent::NO_RT_EVENT);
      // Make sure to call this in the destructor of anything not the target
      virtual RtEvent perform_collective_wait(bool block = true);
      virtual void handle_collective_message(Deserializer &derez);
      virtual RtEvent post_gather(void) { return RtEvent::NO_RT_EVENT; }
      inline bool is_target(void) const { return (target == local_shard); }
      inline RtEvent get_done_event(void) const { return done_event; }
      // Use this method in case we don't actually end up using the collective
      void elide_collective(void);
    protected:
      void send_message(void);
      int compute_expected_notifications(void) const;
    public:
      const ShardID target;
      const int shard_collective_radix;
      const int expected_notifications;
    private:
      RtUserEvent done_event; // only valid on owner shard
      int received_notifications;
    };

    /**
     * \class AllGatherCollective
     * This shard collective has equivalent functionality to
     * MPI All Gather in that it will ensure that all shards
     * see the value data from all other shards.
     */
    template<bool INORDER>
    class AllGatherCollective : public ShardCollective {
    public:
      // Inorder says whether we need to see messages for stages inorder,
      // e.g. do we need to see all stage 0 messages before stage 1
      AllGatherCollective(CollectiveIndexLocation loc, ReplicateContext *ctx);
      AllGatherCollective(ReplicateContext *ctx, CollectiveID id);
      virtual ~AllGatherCollective(void);
    public:
      // We guarantee that these methods will be called atomically
      virtual void pack_collective_stage(Serializer &rez, int stage) = 0;
      virtual void unpack_collective_stage(Deserializer &derez, int stage) = 0;
    public:
      virtual void perform_collective_async(RtEvent pre = RtEvent::NO_RT_EVENT);
      virtual RtEvent perform_collective_wait(bool block = true);
      virtual void handle_collective_message(Deserializer &derez);
      // Use this method in case we don't actually end up using the collective
      void elide_collective(void);
    protected:
      void initialize_collective(void);
      void construct_message(ShardID target, int stage, Serializer &rez);
      bool initiate_collective(void);
      void send_remainder_stage(void);
      bool send_ready_stages(const int start_stage=1);
      void unpack_stage(int stage, Deserializer &derez);
      void complete_exchange(void);
      virtual RtEvent post_complete_exchange(void) 
        { return RtEvent::NO_RT_EVENT; }
    public: 
      const int shard_collective_radix;
      const int shard_collective_log_radix;
      const int shard_collective_stages;
      const int shard_collective_participating_shards;
      const int shard_collective_last_radix;
      const bool participating; 
    private:
      RtUserEvent done_event;
      std::vector<int> stage_notifications;
      std::vector<bool> sent_stages;
      std::map<int,std::vector<std::pair<void*,size_t> > > *reorder_stages;
      // Handle a small race on deciding who gets to
      // trigger the done event, only the last one of these
      // will get to do the trigger to avoid any races
      unsigned pending_send_ready_stages;
#ifdef DEBUG_LEGION
      bool done_triggered;
#endif
    };

    /**
     * \class FutureAllReduceCollective
     * This collective will build a butterfly network for reducing
     * future instance values. Note that execution will not resume
     * until the precondition event for each future instance triggers
     * so this collective can be used to build the Realm event graph
     * in advance of actual execution.
     */
    class FutureAllReduceCollective : public AllGatherCollective<false> {
    public:
      struct PendingReduce {
      public:
        PendingReduce(void) : instance(NULL) { }
        PendingReduce(FutureInstance *inst, ApUserEvent post)
          : instance(inst), postcondition(post) { }
      public:
        FutureInstance *instance;
        ApUserEvent postcondition;
      };
    public:
      FutureAllReduceCollective(Operation *op, CollectiveIndexLocation loc, 
          ReplicateContext *ctx, ReductionOpID redop_id,
          const ReductionOp *redop, bool deterministic);
      FutureAllReduceCollective(Operation *op, ReplicateContext *ctx, 
          CollectiveID id, ReductionOpID redop_id, 
          const ReductionOp *redop, bool deterministic);
      virtual ~FutureAllReduceCollective(void);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void set_shadow_instance(FutureInstance *shadow);
      RtEvent async_reduce(FutureInstance *instance, ApEvent &ready_event);
    protected:
      ApEvent perform_reductions(const std::map<ShardID,PendingReduce> &pend);
      void create_shadow_instance(void);
      void finalize(void);
    public:
      Operation *const op;
      const ReductionOp *const redop;
      const ReductionOpID redop_id;
      const bool deterministic;
    protected:
      const ApUserEvent finished;
      std::map<int,std::map<ShardID,PendingReduce> > pending_reductions;
      std::set<ApEvent> shadow_postconditions;
      FutureInstance *instance;
      FutureInstance *shadow_instance;
      ApEvent instance_ready;
      ApEvent shadow_ready;
      int last_stage_sends;
      int current_stage;
      bool pack_shadow;
    };

    /**
     * \class AllReduceCollective
     * This shard collective has equivalent functionality to 
     * MPI All Reduce in that it will take a value from each
     * shard and reduce it down to a final value using a 
     * Legion reduction operator. We'll build this on top
     * of the AllGatherCollective
     */
    template<typename REDOP>
    class AllReduceCollective : public AllGatherCollective<false> {
    public:
      AllReduceCollective(CollectiveIndexLocation loc, ReplicateContext *ctx);
      AllReduceCollective(ReplicateContext *ctx, CollectiveID id);
      virtual ~AllReduceCollective(void);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void async_all_reduce(typename REDOP::RHS value);
      RtEvent wait_all_reduce(bool block = true);
      typename REDOP::RHS sync_all_reduce(typename REDOP::RHS value);
      typename REDOP::RHS get_result(void);
    protected:
      typename REDOP::RHS value;
      int current_stage;
      std::map<int,std::vector<typename REDOP::RHS> > future_values;
    };

    /**
     * \class ValueBroadcast
     * This will broadcast a value of any type that can be 
     * trivially serialized to all the shards.
     */
    template<typename T>
    class ValueBroadcast : public BroadcastCollective {
    public:
      ValueBroadcast(ReplicateContext *ctx, CollectiveIndexLocation loc)
        : BroadcastCollective(loc, ctx, ctx->owner_shard->shard_id) { }
      ValueBroadcast(ReplicateContext *ctx, ShardID origin,
                     CollectiveIndexLocation loc)
        : BroadcastCollective(loc, ctx, origin) { }
      ValueBroadcast(CollectiveID id, ReplicateContext *ctx, ShardID origin)
        : BroadcastCollective(ctx, id, origin) { }
      ValueBroadcast(const ValueBroadcast &rhs) 
        : BroadcastCollective(rhs) { assert(false); }
      virtual ~ValueBroadcast(void) { }
    public:
      ValueBroadcast& operator=(const ValueBroadcast &rhs)
        { assert(false); return *this; }
      inline void broadcast(const T &v) 
        { value = v; perform_collective_async(); }
      inline T get_value(bool wait = true)
        { if (wait) perform_collective_wait(); return value; }
    public:
      virtual void pack_collective(Serializer &rez) const 
        { rez.serialize(value); }
      virtual void unpack_collective(Deserializer &derez)
        { derez.deserialize(value); }
    protected:
      T value;
    };

    /**
     * \class ValueExchange
     * This class will exchange a value of any type that can be
     * trivially serialized to all the shards
     */
    template<typename T>
    class ValueExchange : public AllGatherCollective<false> { 
    public:
      ValueExchange(CollectiveIndexLocation loc, ReplicateContext *ctx)
        : AllGatherCollective(loc, ctx) { }
      ValueExchange(ReplicateContext *ctx, CollectiveID id)
        : AllGatherCollective(ctx, id) { }
      virtual ~ValueExchange(void) { }
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage)
      {
        rez.serialize<size_t>(values.size());
        for (typename std::set<T>::const_iterator it = values.begin();
              it != values.end(); it++)
          rez.serialize(*it);
      }
      virtual void unpack_collective_stage(Deserializer &derez, int stage)
      {
        size_t num_values;
        derez.deserialize(num_values);
        for (unsigned idx = 0; idx < num_values; idx++)
        {
          T value;
          derez.deserialize(value);
          values.insert(value);
        }
      }
    public:
      const std::set<T>& exchange_values(T value)
      {
        values.insert(value);
        perform_collective_sync();
        return values;
      }
    protected:
      std::set<T> values;
    };

    /**
     * \class BufferBroadcast
     * Broadcast out a binary buffer out to all the shards
     */
    class BufferBroadcast : public BroadcastCollective {
    public:
      BufferBroadcast(ReplicateContext *ctx, CollectiveIndexLocation loc)
        : BroadcastCollective(loc, ctx, ctx->owner_shard->shard_id),
          buffer(NULL), size(0), own(false) { }
      BufferBroadcast(ReplicateContext *ctx, ShardID origin,
                     CollectiveIndexLocation loc)
        : BroadcastCollective(loc, ctx, origin),
          buffer(NULL), size(0), own(false) { }
      BufferBroadcast(const BufferBroadcast &rhs) 
        : BroadcastCollective(rhs) { assert(false); }
      virtual ~BufferBroadcast(void) { if (own) free(buffer); }
    public:
      BufferBroadcast& operator=(const BufferBroadcast &rhs)
        { assert(false); return *this; }
      void broadcast(void *buffer, size_t size, bool copy = true);
      const void* get_buffer(size_t &size, bool wait = true);
    public:
      virtual void pack_collective(Serializer &rez) const; 
      virtual void unpack_collective(Deserializer &derez);
    protected:
      void *buffer;
      size_t size;
      bool own;
    };

    /**
     * \class ShardSyncTree
     * A synchronization tree allows one shard to be notified when
     * all the other shards have reached a certain point in the 
     * execution of the program.
     */
    class ShardSyncTree : public GatherCollective {
    public:
      ShardSyncTree(ReplicateContext *ctx, ShardID origin, 
                    CollectiveIndexLocation loc);
      ShardSyncTree(const ShardSyncTree &rhs) = delete;
      virtual ~ShardSyncTree(void);
    public:
      ShardSyncTree& operator=(const ShardSyncTree &rhs) = delete; 
    public:
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
      virtual RtEvent post_gather(void);
    protected:
      std::vector<RtEvent> postconditions;
    };

    /**
     * \class ShardEventTree
     * This collective will construct an event broadcast tree
     * so that one shard can notify all the other shards once
     * an event has triggered
     */
    class ShardEventTree : public BroadcastCollective {
    public:
      ShardEventTree(ReplicateContext *ctx, ShardID origin, 
                     CollectiveID id);
      ShardEventTree(const ShardEventTree &rhs) = delete; 
      virtual ~ShardEventTree(void);
    public:
      ShardEventTree& operator=(const ShardEventTree &rhs) = delete; 
    public:
      void signal_tree(RtEvent precondition); // origin
      RtEvent get_local_event(void);
    public:
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
      virtual RtEvent post_broadcast(void) { return postcondition; }
    protected:
      RtEvent precondition, postcondition;
    };

    /**
     * \class SingleTaskTree
     * This collective is an extension of ShardEventTree that also
     * provides a broadcasting mechanism for the size of the future
     */
    class SingleTaskTree : public ShardEventTree {
    public:
      SingleTaskTree(ReplicateContext *ctx, ShardID origin, 
                     CollectiveID id, FutureImpl *impl);
      SingleTaskTree(const SingleTaskTree &rhs) = delete;
      virtual ~SingleTaskTree(void);
    public:
      SingleTaskTree & operator=(const SingleTaskTree &rhs) = delete;
    public:
      void broadcast_future_size(RtEvent precondition, 
          size_t future_size, bool has_size);
    public:
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
    protected:
      FutureImpl *const future;
      size_t future_size;
      bool has_future_size;
    };

    /**
     * \class CrossProductExchange
     * A class for exchanging the names of partitions created by
     * a call for making cross-product partitions
     */
    class CrossProductCollective : public AllGatherCollective<false> {
    public:
      CrossProductCollective(ReplicateContext *ctx,
                             CollectiveIndexLocation loc);
      CrossProductCollective(const CrossProductCollective &rhs);
      virtual ~CrossProductCollective(void);
    public:
      CrossProductCollective& operator=(const CrossProductCollective &rhs);
    public:
      void exchange_partitions(std::map<IndexSpace,IndexPartition> &handles);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    protected:
      std::map<IndexSpace,IndexPartition> non_empty_handles;
    };

    /**
     * \class ShardingGatherCollective
     * A class for gathering all the names of the ShardingIDs chosen
     * by different mappers to confirm that they are all the same.
     * This is primarily only used in debug mode.
     */
    class ShardingGatherCollective : public GatherCollective {
    public:
      ShardingGatherCollective(ReplicateContext *ctx, ShardID target,
                               CollectiveIndexLocation loc);
      ShardingGatherCollective(const ShardingGatherCollective &rhs);
      virtual ~ShardingGatherCollective(void);
    public:
      ShardingGatherCollective& operator=(const ShardingGatherCollective &rhs);
    public:
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
    public:
      void contribute(ShardingID value);
      bool validate(ShardingID value);
    protected:
      std::map<ShardID,ShardingID> results;
    };

    /**
     * \class IndirectRecordExchange
     * A class for doing an all-gather of indirect records for 
     * doing gather/scatter/full-indirect copy operations.
     */
    class IndirectRecordExchange : public AllGatherCollective<true> {
    public:
      IndirectRecordExchange(ReplicateContext *ctx, CollectiveID id);
      IndirectRecordExchange(const IndirectRecordExchange &rhs) = delete;
      virtual ~IndirectRecordExchange(void);
    public:
      IndirectRecordExchange& operator=(
          const IndirectRecordExchange &rhs) = delete;
    public:
      RtEvent exchange_records(
          std::vector<std::vector<IndirectRecord>*> &targets,
          std::vector<IndirectRecord> &local_records);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
      virtual RtEvent post_complete_exchange(void);
    protected:
      std::vector<std::vector<IndirectRecord>*> local_targets;
      std::vector<IndirectRecord> all_records;
    };
    
    /**
     * \class FieldDescriptorExchange
     * A class for doing an all-gather of field descriptors for 
     * doing dependent partitioning operations. This will also build
     * a butterfly tree of user events that will be used to know when
     * all of the constituent shards are done with the operation they
     * are collectively performing together.
     */
    class FieldDescriptorExchange : public AllGatherCollective<false> {
    public:
      FieldDescriptorExchange(ReplicateContext *ctx,
                              CollectiveIndexLocation loc);
      FieldDescriptorExchange(const FieldDescriptorExchange &rhs);
      virtual ~FieldDescriptorExchange(void);
    public:
      FieldDescriptorExchange& operator=(const FieldDescriptorExchange &rhs);
    public:
      ApEvent exchange_descriptors(ApEvent ready_event,
                                 const std::vector<FieldDataDescriptor> &desc);
      // Have to call this with the completion event
      ApEvent exchange_completion(ApEvent complete_event);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      std::set<ApEvent> ready_events;
      std::vector<FieldDataDescriptor> descriptors;
    public:
      // Use these for building the butterfly network of user events for
      // knowing when everything is done on all the nodes. 
      // This vector is of the number of stages and tracks the incoming
      // set of remote complete events for a stage, in the case of a 
      // remainder stage it is of size 1
      std::vector<std::set<ApUserEvent> > remote_to_trigger; // stages
      // This vector is the number of stages+1 to capture the ready
      // event for each of the different stages as well as the event
      // for when the entire collective is done
      mutable std::vector<std::set<ApEvent> > local_preconditions; 
    };

    /**
     * \class FieldDescriptorGather
     * A class for doing a gather of field descriptors to a specific
     * node for doing dependent partitioning operations. This collective
     * also will construct an event broadcast tree to inform all the 
     * constituent shards about when the operation is done with the 
     * instances which are being gathered.
     */
    class FieldDescriptorGather : public GatherCollective {
    public:
      FieldDescriptorGather(ReplicateContext *ctx, ShardID target,
                            CollectiveIndexLocation loc);
      FieldDescriptorGather(const FieldDescriptorGather &rhs);
      virtual ~FieldDescriptorGather(void);
    public:
      FieldDescriptorGather& operator=(const FieldDescriptorGather &rhs);
    public:
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
    public:
      void contribute(ApEvent ready_event,
                      const std::vector<FieldDataDescriptor> &descriptors);
      const std::vector<FieldDataDescriptor>& 
           get_full_descriptors(ApEvent &ready);
      // Owner shard only
      void notify_remote_complete(ApEvent precondition);
      // Non-owner shard only
      ApEvent get_complete_event(void) const;
    protected:
      std::set<ApEvent> ready_events;
      std::vector<FieldDataDescriptor> descriptors;
      std::set<ApUserEvent> remote_complete_events;
      ApUserEvent complete_event;
      bool used;
    };

    /**
     * \class FutureBroadcast
     * A class for broadcasting a future result to all the shards
     */
    class FutureBroadcast : public BroadcastCollective {
    public:
      FutureBroadcast(ReplicateContext *ctx, CollectiveID id, 
                      ShardID source, FutureImpl *impl);
      FutureBroadcast(const FutureBroadcast &rhs);
      virtual ~FutureBroadcast(void);
    public:
      FutureBroadcast& operator=(const FutureBroadcast &rhs);
    public:
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
    public:
      void broadcast_future(void);
    protected:
      FutureImpl *const impl;
      RtEvent ready;
    };

    /**
     * \class BufferExchange
     * A class for doing an all-to-all exchange of byte buffers
     */
    class BufferExchange : public AllGatherCollective<false> {
    public:
      BufferExchange(ReplicateContext *ctx,
                     CollectiveIndexLocation loc);
      BufferExchange(const BufferExchange &rhs);
      virtual ~BufferExchange(void);
    public:
      BufferExchange& operator=(const BufferExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      const std::map<ShardID,std::pair<void*,size_t> >&
        exchange_buffers(void *value, size_t size, bool keep_self = false);
      RtEvent exchange_buffers_async(void *value, size_t size, 
                                     bool keep_self = false);
      const std::map<ShardID,std::pair<void*,size_t> >& sync_buffers(bool keep);
    protected:
      std::map<ShardID,std::pair<void*,size_t> > results;
    };

    /**
     * \class FutureNameExchange
     * A class for doing an all-to-all exchange of future names
     */
    class FutureNameExchange : public AllGatherCollective<false> {
    public:
      FutureNameExchange(ReplicateContext *ctx, CollectiveID id, 
                         ReplFutureMapImpl *future_map,
                         ReferenceMutator *mutator);
      FutureNameExchange(const FutureNameExchange &rhs);
      virtual ~FutureNameExchange(void);
    public:
      FutureNameExchange& operator=(const FutureNameExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void exchange_future_names(std::map<DomainPoint,Future> &futures);
    public:
      ReplFutureMapImpl *const future_map;
      ReferenceMutator *const mutator;
    protected:
      std::map<DomainPoint,Future> results;
    };

    /**
     * \class MustEpochMappingBroadcast 
     * A class for broadcasting the results of the mapping decisions
     * for a map must epoch call on a single node
     */
    class MustEpochMappingBroadcast : public BroadcastCollective {
    public:
      MustEpochMappingBroadcast(ReplicateContext *ctx, ShardID origin,
                                CollectiveID collective_id);
      MustEpochMappingBroadcast(const MustEpochMappingBroadcast &rhs);
      virtual ~MustEpochMappingBroadcast(void);
    public:
      MustEpochMappingBroadcast& operator=(
                                  const MustEpochMappingBroadcast &rhs);
    public:
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
    public:
      void broadcast(const std::vector<Processor> &processor_mapping,
         const std::vector<std::vector<Mapping::PhysicalInstance> > &mappings);
      void receive_results(std::vector<Processor> &processor_mapping,
               const std::vector<unsigned> &constraint_indexes,
               std::vector<std::vector<Mapping::PhysicalInstance> > &mappings,
               std::map<PhysicalManager*,unsigned> &acquired);
    protected:
      std::vector<Processor> processors;
      std::vector<std::vector<DistributedID> > instances;
    protected:
      RtUserEvent local_done_event;
      mutable std::set<RtEvent> done_events;
      std::set<PhysicalManager*> held_references;
    };

    /**
     * \class MustEpochMappingExchange
     * A class for exchanging the mapping decisions for
     * specific constraints for a must epoch launch
     */
    class MustEpochMappingExchange : public AllGatherCollective<false> {
    public:
      struct ConstraintInfo {
        std::vector<DistributedID> instances;
        ShardID                    origin_shard;
        int                        weight;
      };
    public:
      MustEpochMappingExchange(ReplicateContext *ctx,
                               CollectiveID collective_id);
      MustEpochMappingExchange(const MustEpochMappingExchange &rhs);
      virtual ~MustEpochMappingExchange(void);
    public:
      MustEpochMappingExchange& operator=(const MustEpochMappingExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void exchange_must_epoch_mappings(ShardID shard_id, 
              size_t total_shards, size_t total_constraints,
              const std::vector<const Task*> &local_tasks,
              const std::vector<const Task*> &all_tasks,
                    std::vector<Processor> &processor_mapping,
              const std::vector<unsigned> &constraint_indexes,
              std::vector<std::vector<Mapping::PhysicalInstance> > &mappings,
              const std::vector<int> &mapping_weights,
              std::map<PhysicalManager*,unsigned> &acquired);
    protected:
      std::map<DomainPoint,Processor> processors;
      std::map<unsigned/*constraint index*/,ConstraintInfo> constraints;
    protected:
      RtUserEvent local_done_event;
      std::set<RtEvent> done_events;
      std::set<PhysicalManager*> held_references;
    };

    /**
     * \class MustEpochDependenceExchange
     * A class for exchanging the mapping dependence events for all 
     * the single tasks in a must epoch launch so we can know which
     * order the point tasks are being mapped in.
     */
    class MustEpochDependenceExchange : public AllGatherCollective<false> {
    public:
      MustEpochDependenceExchange(ReplicateContext *ctx, 
                                  CollectiveIndexLocation loc);
      MustEpochDependenceExchange(const MustEpochDependenceExchange &rhs);
      virtual ~MustEpochDependenceExchange(void);
    public:
      MustEpochDependenceExchange& operator=(
                                  const MustEpochDependenceExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void exchange_must_epoch_dependences(
                            std::map<DomainPoint,RtUserEvent> &mapped_events);
    protected:
      std::map<DomainPoint,RtUserEvent> mapping_dependences;
    };

    /**
     * \class MustEpochCompletionExchange
     * A class for exchanging the local mapping and completion events
     * for all the tasks in a must epoch operation
     */
    class MustEpochCompletionExchange : public AllGatherCollective<false> {
    public:
      MustEpochCompletionExchange(ReplicateContext *ctx,
                                  CollectiveIndexLocation loc);
      MustEpochCompletionExchange(const MustEpochCompletionExchange &rhs);
      virtual ~MustEpochCompletionExchange(void);
    public:
      MustEpochCompletionExchange& operator=(
                                    const MustEpochCompletionExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void exchange_must_epoch_completion(RtEvent mapped, ApEvent complete,
                                          std::set<RtEvent> &tasks_mapped,
                                          std::set<ApEvent> &tasks_complete);
    protected:
      std::set<RtEvent> tasks_mapped;
      std::set<ApEvent> tasks_complete;
    }; 

    /**
     * \class ShardedMappingExchange
     * A class for exchanging the names of instances and mapping dependence
     * events for sharded mapping operations.
     */
    class ShardedMappingExchange : public AllGatherCollective<false> {
    public:
      ShardedMappingExchange(CollectiveIndexLocation loc, ReplicateContext *ctx,
                             ShardID shard_id, bool check_mappings);
      ShardedMappingExchange(const ShardedMappingExchange &rhs);
      virtual ~ShardedMappingExchange(void);
    public:
      ShardedMappingExchange& operator=(const ShardedMappingExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void initiate_exchange(const InstanceSet &mappings,
                             const std::vector<InstanceView*> &views);
      void complete_exchange(Operation *op, ShardedView *sharded_view,
                             const InstanceSet &mappings,
                             std::set<RtEvent> &map_applied_events);
    public:
      const ShardID shard_id;
      const bool check_mappings;
    protected:
      std::map<DistributedID,LegionMap<ShardID,FieldMask> > mappings;
      LegionMap<DistributedID,FieldMask> global_views;
    };

    /**
     * \class TemplateIndexExchange
     * A class for exchanging proposed templates for trace replay
     */
    class TemplateIndexExchange : public AllGatherCollective<true> {
    public:
      TemplateIndexExchange(ReplicateContext *ctx, CollectiveID id);
      TemplateIndexExchange(const TemplateIndexExchange &rhs);
      virtual ~TemplateIndexExchange(void);
    public:
      TemplateIndexExchange& operator=(const TemplateIndexExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void initiate_exchange(const std::vector<int> &indexes);
      void complete_exchange(std::map<int,unsigned> &index_counts);
    protected:
      int current_stage;
      std::map<int,unsigned> index_counts;
    };

    /**
     * \class UnorderedExchange
     * This is a class that exchanges information about unordered operations
     * that are ready to execute on each shard so that we can determine which
     * operations can be inserted into a task stream
     */
    class UnorderedExchange : public AllGatherCollective<true> {
    public:
      UnorderedExchange(ReplicateContext *ctx, CollectiveIndexLocation loc);
      UnorderedExchange(const UnorderedExchange &rhs);
      virtual ~UnorderedExchange(void);
    public:
      UnorderedExchange& operator=(const UnorderedExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      bool exchange_unordered_ops(const std::list<Operation*> &unordered_ops,
                                  std::vector<Operation*> &ready_ops);
    protected:
      template<typename T>
      void update_future_counts(const int stage,
          std::map<int,std::map<T,unsigned> > &future_counts,
          std::map<T,unsigned> &counts);
      template<typename T>
      void pack_counts(Serializer &rez, const std::map<T,unsigned> &counts);
      template<typename T>
      void unpack_counts(const int stage, Deserializer &derez, 
                         std::map<T,unsigned> &future_counts);
      template<typename T>
      void pack_field_counts(Serializer &rez, 
          const std::map<std::pair<T,FieldID>,unsigned> &counts);
      template<typename T>
      void unpack_field_counts(const int stage, Deserializer &derez, 
          std::map<std::pair<T,FieldID>,unsigned> &future_counts);
      template<typename T, typename OP>
      void initialize_counts(const std::map<T,OP*> &ops,
                             std::map<T,unsigned> &counts);
      template<typename T, typename OP>
      void find_ready_ops(const size_t total_shards,
          const std::map<T,unsigned> &final_counts,
          const std::map<T,OP*> &ops, std::vector<Operation*> &ready_ops);
    protected:
      std::map<IndexSpace,unsigned> index_space_counts;
      std::map<IndexPartition,unsigned> index_partition_counts;
      std::map<FieldSpace,unsigned> field_space_counts;
      // Use the lowest field ID here as the key
      std::map<std::pair<FieldSpace,FieldID>,unsigned> field_counts;
      std::map<LogicalRegion,unsigned> logical_region_counts;
      // Use the lowest field ID here as the key
      std::map<std::pair<LogicalRegion,FieldID>,unsigned> detach_counts;
    protected:
      std::map<IndexSpace,ReplDeletionOp*> index_space_deletions;
      std::map<IndexPartition,ReplDeletionOp*> index_partition_deletions;
      std::map<FieldSpace,ReplDeletionOp*> field_space_deletions;
      // Use the lowest field ID here as the key
      std::map<std::pair<FieldSpace,FieldID>,ReplDeletionOp*> field_deletions;
      std::map<LogicalRegion,ReplDeletionOp*> logical_region_deletions;
      // Use the lowest field ID here as the key
      std::map<std::pair<LogicalRegion,FieldID>,ReplDetachOp*> detachments;
    };

    /**
     * \class ConsensusMatchBase
     * A base class for consensus match
     */
    class ConsensusMatchBase : public AllGatherCollective<true> {
    public:
      struct ConsensusMatchArgs : public LgTaskArgs<ConsensusMatchArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_CONSENSUS_MATCH_TASK_ID;
      public:
        ConsensusMatchArgs(ConsensusMatchBase *b, UniqueID uid)
          : LgTaskArgs(uid), base(b) { }
      public:
        ConsensusMatchBase *const base;
      };
    public:
      ConsensusMatchBase(ReplicateContext *ctx, CollectiveIndexLocation loc);
      ConsensusMatchBase(const ConsensusMatchBase &rhs);
      virtual ~ConsensusMatchBase(void);
    public:
      virtual void complete_exchange(void) = 0;
    public:
      static void handle_consensus_match(const void *args);
    };

    /**
     * \class ConsensusMatchExchange
     * This is collective for performing a consensus exchange between 
     * the shards for a collection of values.
     */
    template<typename T>
    class ConsensusMatchExchange : ConsensusMatchBase {
    public:
      ConsensusMatchExchange(ReplicateContext *ctx, CollectiveIndexLocation loc,
                      Future to_complete, void *output, ApUserEvent to_trigger);
      ConsensusMatchExchange(const ConsensusMatchExchange &rhs);
      virtual ~ConsensusMatchExchange(void);
    public:
      ConsensusMatchExchange& operator=(const ConsensusMatchExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      bool match_elements_async(const void *input, size_t num_elements);
      virtual void complete_exchange(void);
    protected:
      Future to_complete;
      T *const output;
      const ApUserEvent to_trigger;
      std::map<T,size_t> element_counts;
#ifdef DEBUG_LEGION
      size_t max_elements;
#endif
    };

    /**
     * \class VerifyReplicableExchange
     * This class exchanges hash values of all the inputs for calls
     * into control replication contexts in order to ensure that they 
     * all are the same.
     */
    class VerifyReplicableExchange : public AllGatherCollective<false> {
    public:
      VerifyReplicableExchange(CollectiveIndexLocation loc, 
                               ReplicateContext *ctx);
      VerifyReplicableExchange(const VerifyReplicableExchange &rhs);
      virtual ~VerifyReplicableExchange(void);
    public:
      VerifyReplicableExchange& operator=(const VerifyReplicableExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      typedef std::map<std::pair<uint64_t,uint64_t>,ShardID> ShardHashes;
      const ShardHashes& exchange(const uint64_t hash[2]);
    public:
      ShardHashes unique_hashes;
    };

    /**
     * \class OutputSizeExchange
     * This class exchanges sizes of output subregions that are globally
     * indexed.
     */
    class OutputSizeExchange : public AllGatherCollective<false> {
    public:
      typedef std::map<DomainPoint,DomainPoint> SizeMap;
    public:
      OutputSizeExchange(ReplicateContext *ctx,
                         CollectiveIndexLocation loc,
                         std::map<unsigned,SizeMap> &all_output_sizes);
      OutputSizeExchange(const OutputSizeExchange &rhs);
      virtual ~OutputSizeExchange(void);
    public:
      OutputSizeExchange& operator=(const OutputSizeExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      RtEvent exchange_output_sizes(void);
    public:
      std::map<unsigned,SizeMap> &all_output_sizes;
    };

    /**
     * \class IndexAttachLaunchSpace
     * This collective computes the number of points in each
     * shard of a replicated index attach collective in order
     * to help compute the index launch space
     */
    class IndexAttachLaunchSpace : public AllGatherCollective<false> {
    public:
      IndexAttachLaunchSpace(ReplicateContext *ctx,
                             CollectiveIndexLocation loc);
      IndexAttachLaunchSpace(const IndexAttachLaunchSpace &rhs);
      virtual ~IndexAttachLaunchSpace(void);
    public:
      IndexAttachLaunchSpace& operator=(const IndexAttachLaunchSpace &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void exchange_counts(size_t count);
      IndexSpaceNode* get_launch_space(Provenance *provenance);
    protected:
      std::vector<size_t> sizes;
      unsigned nonzeros;
    };

    /**
     * \class IndexAttachUpperBound
     * This computes the upper bound node in the region
     * tree for an index space attach operation
     */
    class IndexAttachUpperBound : public AllGatherCollective<false> {
    public:
      IndexAttachUpperBound(ReplicateContext *ctx,
                            CollectiveIndexLocation loc,
                            RegionTreeForest *forest);
      IndexAttachUpperBound(const IndexAttachUpperBound &rhs);
      virtual ~IndexAttachUpperBound(void);
    public:
      IndexAttachUpperBound& operator=(const IndexAttachUpperBound &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      RegionTreeNode* find_upper_bound(RegionTreeNode *node);
    public:
      RegionTreeForest *const forest;
    protected:
      RegionTreeNode *node;
    };

    /**
     * \class IndexAttachExchange
     * This class is used to exchange the needed metadata for
     * replicated index space attach operations
     */
    class IndexAttachExchange : public AllGatherCollective<false> {
    public:
      IndexAttachExchange(ReplicateContext *ctx,
                          CollectiveIndexLocation loc);
      IndexAttachExchange(const IndexAttachExchange &rhs);
      virtual ~IndexAttachExchange(void);
    public:
      IndexAttachExchange& operator=(const IndexAttachExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void exchange_spaces(std::vector<IndexSpace> &spaces);
      size_t get_spaces(std::vector<IndexSpace> &spaces, unsigned &local_start);
      IndexSpaceNode* get_launch_space(void); 
    protected:
      std::map<ShardID,std::vector<IndexSpace> > shard_spaces;
    };

    /**
     * \class IndexAttachCoregions
     * Exchange the information about coregions between the different
     * shards to ensure that only a single point will perform the 
     * mapping if multiple points map to the same region
     */
    class IndexAttachCoregions : public AllGatherCollective<false> {
    public:
      struct PendingPoint {
      public:
        PendingPoint(void)
          : region(LogicalRegion::NO_REGION),
            instances(NULL), attached_event(NULL) { }
        PendingPoint(LogicalRegion r, InstanceSet &s, ApUserEvent &e)
          : region(r), instances(&s), attached_event(&e) { }
      public:
        LogicalRegion region;
        InstanceSet *instances;
        ApUserEvent *attached_event;
      };
      struct RegionPoints {
      public:
        std::map<ShardID,ApUserEvent> shard_events;
        std::set<DistributedID> managers;
      };
    public:
      IndexAttachCoregions(ReplicateContext *ctx,
                           CollectiveIndexLocation loc, size_t points);
      IndexAttachCoregions(const IndexAttachCoregions &rhs);
      virtual ~IndexAttachCoregions(void);
    public:
      IndexAttachCoregions& operator=(const IndexAttachCoregions &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
      virtual RtEvent post_complete_exchange(void);
    public:
      bool record_point(PointAttachOp *point, LogicalRegion region,
              InstanceSet &instances, ApUserEvent &attached_event);
    public:
      const size_t total_points;
    protected:
      std::map<PointAttachOp*,PendingPoint> pending_points;
      std::map<LogicalRegion,RegionPoints> region_points;
    };

    /**
     * \class ImplicitShardingFunctor
     * Support the computation of an implicit sharding function for 
     * the creation of replicated future maps
     */
    class ImplicitShardingFunctor : public AllGatherCollective<false>,
                                    public ShardingFunctor {
    public:
      ImplicitShardingFunctor(ReplicateContext *ctx,
                              CollectiveIndexLocation loc,
                              ReplFutureMapImpl *map);
      ImplicitShardingFunctor(const ImplicitShardingFunctor &rhs);
      virtual ~ImplicitShardingFunctor(void);
    public:
      ImplicitShardingFunctor& operator=(const ImplicitShardingFunctor &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      virtual ShardID shard(const DomainPoint &point,
                            const Domain &full_space,
                            const size_t total_shards);
    protected:
      virtual RtEvent post_complete_exchange(void);
    public:
      template<typename T>
      void compute_sharding(const std::map<DomainPoint,T> &points)
      {
        for (typename std::map<DomainPoint,T>::const_iterator it =
              points.begin(); it != points.end(); it++)
          implicit_sharding[it->first] = local_shard; 
        this->perform_collective_async();
      }
    public:
      ReplFutureMapImpl *const map;
    protected:
      std::map<DomainPoint,ShardID> implicit_sharding;
    };

    /**
     * \class ConcurrentExecutionValidator
     * This collective helps to validate the safety of the execution of
     * concurrent index space task launches to ensure that all the point
     * tasks have been mapped to different processors.
     */
    class ConcurrentExecutionValidator : public GatherCollective {
    public:
      ConcurrentExecutionValidator(ReplIndexTask *owner,
          CollectiveIndexLocation loc, ReplicateContext *ctx, ShardID target);
      virtual ~ConcurrentExecutionValidator(void) { }
    public:
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
      virtual RtEvent post_gather(void);
    public:
      void perform_validation(std::map<DomainPoint,Processor> &processors);
    public:
      ReplIndexTask *const owner;
    protected:
      std::map<DomainPoint,Processor> concurrent_processors;
    };

    /**
     * \class SlowBarrier
     * This class creates a collective that behaves like a barrier, but is
     * probably slower than Realm phase barriers. It's useful for cases
     * where we may not know whether we are going to perform a barrier or
     * not so we grab a collective ID. We can throw away collective IDs
     * for free, but in the rare case we actually do need to perform
     * the barrier then this class will handle the implementation.
     */
    class SlowBarrier : public AllGatherCollective<false> {
    public:
      SlowBarrier(ReplicateContext *ctx, CollectiveID id);
      SlowBarrier(const SlowBarrier &rhs);
      virtual ~SlowBarrier(void);
    public:
      SlowBarrier& operator=(const SlowBarrier &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage) { }
      virtual void unpack_collective_stage(Deserializer &derez, int stage) { }
    };

    /**
     * \class ReplIndividualTask
     * An individual task that is aware that it is 
     * being executed in a control replication context.
     */
    class ReplIndividualTask : public IndividualTask {
    public:
      ReplIndividualTask(Runtime *rt);
      ReplIndividualTask(const ReplIndividualTask &rhs);
      virtual ~ReplIndividualTask(void);
    public:
      ReplIndividualTask& operator=(const ReplIndividualTask &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual void resolve_false(bool speculated, bool launched);
      virtual void shard_off(RtEvent mapped_precondition);
      virtual void prepare_map_must_epoch(void);
      virtual void handle_future_size(size_t return_type_size,
          bool has_return_type_size, std::set<RtEvent> &applied_events);
    public:
      // Override these so we can broadcast the future result
      virtual void trigger_task_complete(void);
    public:
      void initialize_replication(ReplicateContext *ctx);
      void set_sharding_function(ShardingID functor,ShardingFunction *function);
    protected:
      ShardID owner_shard;
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
      CollectiveID mapped_collective_id; // id for mapped event broadcast
      CollectiveID future_collective_id; // id for the future broadcast 
      SingleTaskTree *mapped_collective;
      FutureBroadcast *future_collective;
#ifdef DEBUG_LEGION
    public:
      inline void set_sharding_collective(ShardingGatherCollective *collective)
        { sharding_collective = collective; }
    protected:
      ShardingGatherCollective *sharding_collective;
#endif
    };

    /**
     * \class ReplIndexTask
     * An individual task that is aware that it is 
     * being executed in a control replication context.
     */
    class ReplIndexTask : public IndexTask {
    public:
      ReplIndexTask(Runtime *rt);
      ReplIndexTask(const ReplIndexTask &rhs);
      virtual ~ReplIndexTask(void);
    public:
      ReplIndexTask& operator=(const ReplIndexTask &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
    protected:
      virtual void create_future_instances(std::vector<Memory> &target_mems);
      virtual void finish_index_task_reduction(void);
      virtual RtEvent finish_index_task_complete(void);
    public:
      // Have to override this too for doing output in the
      // case that we misspeculate
      virtual void resolve_false(bool speculated, bool launched);
      virtual void prepare_map_must_epoch(void);
    public:
      void initialize_replication(ReplicateContext *ctx);
      void set_sharding_function(ShardingID functor,ShardingFunction *function);
      virtual FutureMapImpl* create_future_map(TaskContext *ctx,
                    IndexSpace launch_space, IndexSpace shard_space);
      virtual void initialize_concurrent_analysis(void);
      virtual RtEvent verify_concurrent_execution(const DomainPoint &point,
                                                  Processor target);
      void select_sharding_function(ReplicateContext *repl_ctx);
    public:
      // Methods for supporting intra-index-space mapping dependences
      virtual RtEvent find_intra_space_dependence(const DomainPoint &point);
      virtual void record_intra_space_dependence(const DomainPoint &point,
                                                 const DomainPoint &next,
                                                 RtEvent point_mapped);
    protected:
      virtual void finalize_output_regions(void);
    protected:
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
      BufferExchange *serdez_redop_collective;
      FutureAllReduceCollective *all_reduce_collective;
      OutputSizeExchange *output_size_collective;
    protected:
      // Map of output sizes collected by this shard
      std::map<unsigned,SizeMap> local_output_sizes;
    protected:
      std::set<std::pair<DomainPoint,ShardID> > unique_intra_space_deps;
    protected:
      // For setting up concurrent execution
      RtBarrier concurrent_prebar, concurrent_postbar;
      ConcurrentExecutionValidator *concurrent_validator;
#ifdef DEBUG_LEGION
    public:
      inline void set_sharding_collective(ShardingGatherCollective *collective)
        { sharding_collective = collective; }
    protected:
      ShardingGatherCollective *sharding_collective;
#endif
    protected:
      bool slice_sharding_output;
    };

    /**
     * \class ReplMergeCloseOp
     * A close operation that is aware that it is being
     * executed in a control replication context.
     */
    class ReplMergeCloseOp : public MergeCloseOp {
    public:
      ReplMergeCloseOp(Runtime *runtime);
      ReplMergeCloseOp(const ReplMergeCloseOp &rhs);
      virtual ~ReplMergeCloseOp(void);
    public:
      ReplMergeCloseOp& operator=(const ReplMergeCloseOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      void set_repl_close_info(RtBarrier mapped_barrier);
      virtual void record_refinements(const FieldMask &refinement_mask,
                                      const bool overwrite);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void); 
    protected:
      RtBarrier mapped_barrier;
      RtBarrier refinement_barrier;
      ValueBroadcast<DistributedID> *did_collective;
    };

    /**
     * \class ReplRefinementOp
     * A refinement operatoin that is aware that it is being
     * executed ina  control replication context.
     */
    class ReplRefinementOp : public RefinementOp {
    public:
      ReplRefinementOp(Runtime *runtime);
      ReplRefinementOp(const ReplRefinementOp &rhs);
      virtual ~ReplRefinementOp(void);
    public:
      ReplRefinementOp& operator=(const ReplRefinementOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      void set_repl_refinement_info(RtBarrier mapped_barrier, 
                                    RtBarrier refinement_barrier);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void); 
    protected:
      void initialize_replicated_set(EquivalenceSet *set,
          const FieldMask &mask, std::set<RtEvent> &applied_events) const;
    protected:
      RtBarrier mapped_barrier;
      RtBarrier refinement_barrier;
      std::vector<ValueBroadcast<DistributedID>*> collective_dids;
      // Note that this data structure ensures that we do things
      // for these partitions in a order that is consistent across
      // shards because all shards will sort the keys the same way
      std::map<LogicalPartition,PartitionNode*> replicated_partitions;
      // Same thing for one-off regions
      std::map<LogicalRegion,RegionNode*> replicated_regions;
      // Version information objects for each of our local regions
      // that we are own after sharding non-replicated partitions
      LegionMap<RegionNode*,VersionInfo> sharded_region_version_infos;
      // Regions for which we need to propagate refinements for
      // non-replicated partition refinements
      std::map<PartitionNode*,std::vector<RegionNode*> > sharded_regions;
      // Fields for partitions that have refinement regions
      FieldMaskSet<PartitionNode> sharded_partitions;
    };

    /**
     * \class ReplFillOp
     * A copy operation that is aware that it is being
     * executed in a control replication context.
     */
    class ReplFillOp : public FillOp {
    public:
      ReplFillOp(Runtime *rt);
      ReplFillOp(const ReplFillOp &rhs);
      virtual ~ReplFillOp(void);
    public:
      ReplFillOp& operator=(const ReplFillOp &rhs);
    public:
      void initialize_replication(ReplicateContext *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual void resolve_false(bool speculated, bool launched);
    protected:
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
      MapperManager *mapper;
    public:
      CollectiveID mapped_collective_id;
      ShardEventTree *mapped_collective;
#ifdef DEBUG_LEGION
    public:
      inline void set_sharding_collective(ShardingGatherCollective *collective)
        { sharding_collective = collective; }
    protected:
      ShardingGatherCollective *sharding_collective;
#endif
    };

    /**
     * \class ReplIndexFillOp
     * An index fill operation that is aware that it is 
     * being executed in a control replication context.
     */
    class ReplIndexFillOp : public IndexFillOp {
    public:
      ReplIndexFillOp(Runtime *rt);
      ReplIndexFillOp(const ReplIndexFillOp &rhs);
      virtual ~ReplIndexFillOp(void);
    public:
      ReplIndexFillOp& operator=(const ReplIndexFillOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual void resolve_false(bool speculated, bool launched);
    public:
      void initialize_replication(ReplicateContext *ctx);
    protected:
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
      MapperManager *mapper;
#ifdef DEBUG_LEGION
    public:
      inline void set_sharding_collective(ShardingGatherCollective *collective)
        { sharding_collective = collective; }
    protected:
      ShardingGatherCollective *sharding_collective;
#endif
    };

    /**
     * \class ReplCopyOp
     * A copy operation that is aware that it is being
     * executed in a control replication context.
     */
    class ReplCopyOp : public CopyOp {
    public:
      ReplCopyOp(Runtime *rt);
      ReplCopyOp(const ReplCopyOp &rhs);
      virtual ~ReplCopyOp(void);
    public:
      ReplCopyOp& operator=(const ReplCopyOp &rhs);
    public:
      void initialize_replication(ReplicateContext *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual void resolve_false(bool speculated, bool launched);
    protected:
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
    public:
      CollectiveID mapped_collective_id;
      ShardEventTree *mapped_collective; 
#ifdef DEBUG_LEGION
    public:
      inline void set_sharding_collective(ShardingGatherCollective *collective)
        { sharding_collective = collective; }
    protected:
      ShardingGatherCollective *sharding_collective;
#endif
    };

    /**
     * \class ReplIndexCopyOp
     * An index fill operation that is aware that it is 
     * being executed in a control replication context.
     */
    class ReplIndexCopyOp : public IndexCopyOp {
    public:
      ReplIndexCopyOp(Runtime *rt);
      ReplIndexCopyOp(const ReplIndexCopyOp &rhs);
      virtual ~ReplIndexCopyOp(void);
    public:
      ReplIndexCopyOp& operator=(const ReplIndexCopyOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual void resolve_false(bool speculated, bool launched);
      virtual RtEvent exchange_indirect_records(
          const unsigned index, const ApEvent local_pre, 
          const ApEvent local_post, ApEvent &collective_pre,
          ApEvent &collective_post, const TraceInfo &trace_info,
          const InstanceSet &instances, const RegionRequirement &req,
          const DomainPoint &key,
          std::vector<IndirectRecord> &records, const bool sources);
      virtual RtEvent finalize_exchange(const unsigned index,const bool source);
    public:
      virtual RtEvent find_intra_space_dependence(const DomainPoint &point);
      virtual void record_intra_space_dependence(const DomainPoint &point,
                                                 const DomainPoint &next,
                                                 RtEvent point_mapped);
    public:
      void initialize_replication(ReplicateContext *ctx);
    protected:
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
      std::vector<ApBarrier> pre_indirection_barriers;
      std::vector<ApBarrier> post_indirection_barriers;
      std::vector<IndirectRecordExchange*> src_collectives;
      std::vector<IndirectRecordExchange*> dst_collectives;
      std::set<std::pair<DomainPoint,ShardID> > unique_intra_space_deps;
#ifdef DEBUG_LEGION
    public:
      inline void set_sharding_collective(ShardingGatherCollective *collective)
        { sharding_collective = collective; }
    protected:
      ShardingGatherCollective *sharding_collective;
#endif
    };

    /**
     * \class ReplDeletionOp
     * A deletion operation that is aware that it is
     * being executed in a control replication context.
     */
    class ReplDeletionOp : public DeletionOp {
    public:
      ReplDeletionOp(Runtime *rt);
      ReplDeletionOp(const ReplDeletionOp &rhs);
      virtual ~ReplDeletionOp(void);
    public:
      ReplDeletionOp& operator=(const ReplDeletionOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void trigger_complete(void);
    public:
      void initialize_replication(ReplicateContext *ctx, 
                                  bool is_total, bool is_first,
                                  RtBarrier *ready_barrier = NULL,
                                  RtBarrier *mapping_barrier = NULL,
                                  RtBarrier *execution_barrier = NULL);
      // Help for handling unordered deletions 
      void record_unordered_kind(
       std::map<IndexSpace,ReplDeletionOp*> &index_space_deletions,
       std::map<IndexPartition,ReplDeletionOp*> &index_partition_deletions,
       std::map<FieldSpace,ReplDeletionOp*> &field_space_deletions,
       std::map<std::pair<FieldSpace,FieldID>,ReplDeletionOp*> &field_deletions,
       std::map<LogicalRegion,ReplDeletionOp*> &logical_region_deletions);
    protected:
      RtBarrier ready_barrier;
      RtBarrier mapping_barrier;
      RtBarrier execution_barrier;
      bool is_total_sharding;
      bool is_first_local_shard;
    };

    /**
     * \class ReplPendingPartitionOp
     * A pending partition operation that knows that its
     * being executed in a control replication context
     */
    class ReplPendingPartitionOp : public PendingPartitionOp {
    public:
      ReplPendingPartitionOp(Runtime *rt);
      ReplPendingPartitionOp(const ReplPendingPartitionOp &rhs);
      virtual ~ReplPendingPartitionOp(void);
    public:
      ReplPendingPartitionOp& operator=(const ReplPendingPartitionOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void populate_sources(const FutureMap &fm);
      virtual void trigger_execution(void);
    };

    /**
     * \class ReplDependentPartitionOp
     * A dependent partitioning operation that knows that it
     * is being executed in a control replication context
     */
    class ReplDependentPartitionOp : public DependentPartitionOp {
    public:
      class ReplByFieldThunk : public ByFieldThunk {
      public:
        ReplByFieldThunk(ReplicateContext *ctx,
                         ShardID target, IndexPartition p);
      public:
        virtual ApEvent perform(DependentPartitionOp *op,
            RegionTreeForest *forest, ApEvent instances_ready,
            const std::vector<FieldDataDescriptor> &instances);
        virtual void elide_collectives(void) 
          { gather_collective.elide_collective(); }
      protected:
        FieldDescriptorGather gather_collective;
      };
      class ReplByImageThunk : public ByImageThunk {
      public:
#ifdef SHARD_BY_IMAGE
        ReplByImageThunk(ReplicateContext *ctx,
                         IndexPartition p, IndexPartition proj,
                         ShardID shard_id, size_t total);
#else
        ReplByImageThunk(ReplicateContext *ctx, ShardID target,
                         IndexPartition p, IndexPartition proj,
                         ShardID shard_id, size_t total);
#endif
      public:
        virtual ApEvent perform(DependentPartitionOp *op,
            RegionTreeForest *forest, ApEvent instances_ready,
            const std::vector<FieldDataDescriptor> &instances);
        virtual void elide_collectives(void) { collective.elide_collective(); }
      protected:
#ifdef SHARD_BY_IMAGE
        FieldDescriptorExchange collective;
#else
        FieldDescriptorGather collective;
#endif
        const ShardID shard_id;
        const size_t total_shards;
      };
      class ReplByImageRangeThunk : public ByImageRangeThunk {
      public:
#ifdef SHARD_BY_IMAGE
        ReplByImageRangeThunk(ReplicateContext *ctx,
                              IndexPartition p, IndexPartition proj,
                              ShardID shard_id, size_t total);
#else
        ReplByImageRangeThunk(ReplicateContext *ctx, ShardID target, 
                              IndexPartition p, IndexPartition proj,
                              ShardID shard_id, size_t total);
#endif
      public:
        virtual ApEvent perform(DependentPartitionOp *op,
            RegionTreeForest *forest, ApEvent instances_ready,
            const std::vector<FieldDataDescriptor> &instances);
        virtual void elide_collectives(void) { collective.elide_collective(); }
      protected:
#ifdef SHARD_BY_IMAGE
        FieldDescriptorExchange collective;
#else
        FieldDescriptorGather collective;
#endif
        const ShardID shard_id;
        const size_t total_shards;
      };
      class ReplByPreimageThunk : public ByPreimageThunk {
      public:
        ReplByPreimageThunk(ReplicateContext *ctx, ShardID target,
                            IndexPartition p, IndexPartition proj);
      public:
        virtual ApEvent perform(DependentPartitionOp *op,
            RegionTreeForest *forest, ApEvent instances_ready,
            const std::vector<FieldDataDescriptor> &instances);
        virtual void elide_collectives(void) 
          { gather_collective.elide_collective(); }
      protected:
        FieldDescriptorGather gather_collective;
      };
      class ReplByPreimageRangeThunk : public ByPreimageRangeThunk {
      public:
        ReplByPreimageRangeThunk(ReplicateContext *ctx, ShardID target,
                                 IndexPartition p, IndexPartition proj);
      public:
        virtual ApEvent perform(DependentPartitionOp *op,
            RegionTreeForest *forest, ApEvent instances_ready,
            const std::vector<FieldDataDescriptor> &instances);
        virtual void elide_collectives(void) 
          { gather_collective.elide_collective(); }
      protected:
        FieldDescriptorGather gather_collective;
      };
      // Nothing special about association for control replication
    public:
      ReplDependentPartitionOp(Runtime *rt);
      ReplDependentPartitionOp(const ReplDependentPartitionOp &rhs);
      virtual ~ReplDependentPartitionOp(void);
    public:
      ReplDependentPartitionOp& operator=(const ReplDependentPartitionOp &rhs);
    public:
      void initialize_by_field(ReplicateContext *ctx, ShardID target,
                               ApEvent ready_event, IndexPartition pid,
                               LogicalRegion handle, LogicalRegion parent,
                               IndexSpace color_space, FieldID fid, 
                               MapperID id, MappingTagID tag,
                               const UntypedBuffer &marg,
                               Provenance *provenance);
      void initialize_by_image(ReplicateContext *ctx,
#ifndef SHARD_BY_IMAGE
                               ShardID target,
#endif
                               ApEvent ready_event, IndexPartition pid,
                               IndexSpace handle, LogicalPartition projection,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag,
                               const UntypedBuffer &marg,
                               ShardID shard, size_t total_shards,
                               Provenance *provenance);
      void initialize_by_image_range(ReplicateContext *ctx,
#ifndef SHARD_BY_IMAGE
                               ShardID target,
#endif
                               ApEvent ready_event, IndexPartition pid,
                               IndexSpace handle, LogicalPartition projection,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag,
                               const UntypedBuffer &marg,
                               ShardID shard, size_t total_shards,
                               Provenance *provenance);
      void initialize_by_preimage(ReplicateContext *ctx, ShardID target,
                               ApEvent ready_event, IndexPartition pid,
                               IndexPartition projection, LogicalRegion handle,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag,
                               const UntypedBuffer &marg,
                               Provenance *provenance);
      void initialize_by_preimage_range(ReplicateContext *ctx, ShardID target, 
                               ApEvent ready_event, IndexPartition pid,
                               IndexPartition projection, LogicalRegion handle,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag,
                               const UntypedBuffer &marg,
                               Provenance *provenance);
      void initialize_by_association(ReplicateContext *ctx,LogicalRegion domain,
                               LogicalRegion domain_parent, FieldID fid,
                               IndexSpace range, MapperID id, MappingTagID tag,
                               const UntypedBuffer &marg,
                               Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      // Need to pick our sharding functor
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);  
      virtual void finalize_mapping(void);
      virtual void select_partition_projection(void);
    protected:
      void select_sharding_function(void);
    protected:
      ShardingFunction *sharding_function;
      RtBarrier mapping_barrier;
#ifdef DEBUG_LEGION
    public:
      inline void set_sharding_collective(ShardingGatherCollective *collective)
        { sharding_collective = collective; }
    protected:
      ShardingGatherCollective *sharding_collective;
#endif
    };

    /**
     * \class ReplMustEpochOp
     * A must epoch operation that is aware that it is 
     * being executed in a control replication context
     */
    class ReplMustEpochOp : public MustEpochOp {
    public:
      ReplMustEpochOp(Runtime *rt);
      ReplMustEpochOp(const ReplMustEpochOp &rhs);
      virtual ~ReplMustEpochOp(void);
    public:
      ReplMustEpochOp& operator=(const ReplMustEpochOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual FutureMapImpl* create_future_map(TaskContext *ctx,
                      IndexSpace domain, IndexSpace shard_space);
      virtual RtEvent get_concurrent_analysis_precondition(void);
      virtual void instantiate_tasks(InnerContext *ctx,
                                     const MustEpochLauncher &launcher);
      virtual MapperManager* invoke_mapper(void);
      virtual void map_and_distribute(std::set<RtEvent> &tasks_mapped,
                                      std::set<ApEvent> &tasks_complete);
      virtual bool has_prepipeline_stage(void) const { return true; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_commit(void);
      virtual void receive_resources(size_t return_index,
              std::map<LogicalRegion,unsigned> &created_regions,
              std::vector<DeletedRegion> &deleted_regions,
              std::set<std::pair<FieldSpace,FieldID> > &created_fields,
              std::vector<DeletedField> &deleted_fields,
              std::map<FieldSpace,unsigned> &created_field_spaces,
              std::map<FieldSpace,std::set<LogicalRegion> > &latent_spaces,
              std::vector<DeletedFieldSpace> &deleted_field_spaces,
              std::map<IndexSpace,unsigned> &created_index_spaces,
              std::vector<DeletedIndexSpace> &deleted_index_spaces,
              std::map<IndexPartition,unsigned> &created_partitions,
              std::vector<DeletedPartition> &deleted_partitions,
              std::set<RtEvent> &preconditions);
    public:
      void map_replicate_tasks(void) const;
      void distribute_replicate_tasks(void);
    public:
      void initialize_replication(ReplicateContext *ctx);
      Domain get_shard_domain(void) const;
    protected:
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
      CollectiveID mapping_collective_id;
      bool collective_map_must_epoch_call;
      MustEpochMappingBroadcast *mapping_broadcast;
      MustEpochMappingExchange *mapping_exchange;
      MustEpochDependenceExchange *dependence_exchange;
      MustEpochCompletionExchange *completion_exchange;
      std::set<SingleTask*> shard_single_tasks;
      RtBarrier resource_return_barrier;
      RtBarrier concurrent_prebar, concurrent_postbar;
#ifdef DEBUG_LEGION
    public:
      inline void set_sharding_collective(ShardingGatherCollective *collective)
        { sharding_collective = collective; }
    protected:
      ShardingGatherCollective *sharding_collective;
#endif
    };

    /**
     * \class ReplTimingOp
     * A timing operation that is aware that it is 
     * being executed in a control replication context
     */
    class ReplTimingOp : public TimingOp {
    public:
      ReplTimingOp(Runtime *rt);
      ReplTimingOp(const ReplTimingOp &rhs);
      virtual ~ReplTimingOp(void);
    public:
      ReplTimingOp& operator=(const ReplTimingOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_mapping(void);
      virtual void trigger_execution(void);
    public:
      inline void set_timing_collective(ValueBroadcast<long long> *collective) 
        { timing_collective = collective; }
    protected:
      ValueBroadcast<long long> *timing_collective;
    }; 

    /**
     * \class ReplTunableOp
     * A tunable operation that is aware that it is
     * being executed in a control replicated context
     */
    class ReplTunableOp : public TunableOp {
    public:
      ReplTunableOp(Runtime *rt);
      ReplTunableOp(const ReplTunableOp &rhs);
      virtual ~ReplTunableOp(void);
    public:
      ReplTunableOp& operator=(const ReplTunableOp &rhs);
    public:
      void initialize_replication(ReplicateContext *context);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual void process_result(MapperManager *mapper, 
                                  void *buffer, size_t size) const;
    protected:
      BufferBroadcast *value_broadcast;       
    };

    /**
     * \class ReplAllReduceOp
     * An all-reduce operation that is aware that it is
     * being executed in a control replication context
     */
    class ReplAllReduceOp : public AllReduceOp {
    public:
      ReplAllReduceOp(Runtime *rt);
      ReplAllReduceOp(const ReplAllReduceOp &rhs);
      virtual ~ReplAllReduceOp(void);
    public:
      ReplAllReduceOp& operator=(const ReplAllReduceOp &rhs);
    public:
      void initialize_replication(ReplicateContext *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    protected:
      virtual void populate_sources(void);
      virtual void create_future_instances(std::vector<Memory> &target_mems);
      virtual void all_reduce_serdez(void);
      virtual RtEvent all_reduce_redop(void);
    protected:
      BufferExchange *serdez_redop_collective;
      FutureAllReduceCollective *all_reduce_collective;
    };

    /**
     * \class ReplFenceOp
     * A fence operation that is aware that it is being 
     * executed in a control replicated context. Currently
     * this only applies to mixed and execution fences.
     */
    class ReplFenceOp : public FenceOp {
    public:
      ReplFenceOp(Runtime *rt);
      ReplFenceOp(const ReplFenceOp &rhs);
      virtual ~ReplFenceOp(void);
    public:
      ReplFenceOp& operator=(const ReplFenceOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void trigger_replay(void);
      virtual void complete_replay(ApEvent complete_event);
    protected:
      void initialize_fence_barriers(ReplicateContext *repl_ctx = NULL);
    protected:
      RtBarrier mapping_fence_barrier;
      ApBarrier execution_fence_barrier;
    };

    /**
     * \class ReplMapOp
     * An inline mapping operation that is aware that it is being
     * executed in a control replicated context. We require that
     * any inline mapping be mapped on all shards before we consider
     * it mapped on any shard. The reason for this is that inline
     * mappings can act like a kind of communication between shards
     * where they are all reading/writing to the same logical region.
     */
    class ReplMapOp : public MapOp {
    public:
      ReplMapOp(Runtime *rt);
      ReplMapOp(const ReplMapOp &rhs);
      virtual ~ReplMapOp(void);
    public:
      ReplMapOp& operator=(const ReplMapOp &rhs);
    public:
      void initialize_replication(ReplicateContext *ctx);
      RtEvent complete_inline_mapping(RtEvent mapping_applied);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void); 
    protected:
      RtBarrier inline_barrier;
      ShardedMappingExchange *exchange; 
      ValueBroadcast<DistributedID> *view_did_broadcast;
      ShardedView *sharded_view;
    };

    /**
     * \class ReplAttachOp
     * An attach operation that is aware that it is being
     * executed in a control replicated context.
     */
    class ReplAttachOp : public AttachOp {
    public:
      ReplAttachOp(Runtime *rt);
      ReplAttachOp(const ReplAttachOp &rhs);
      virtual ~ReplAttachOp(void);
    public:
      ReplAttachOp& operator=(const ReplAttachOp &rhs);
    public:
      void initialize_replication(ReplicateContext *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
    protected:
      RtBarrier resource_barrier;
      ApBarrier broadcast_barrier;
      ApBarrier reduce_barrier;
      RtUserEvent repl_mapping_applied;
      InstanceRef external_instance;
      ShardedMappingExchange *exchange; 
      ValueBroadcast<DistributedID> *did_broadcast;
      ShardedView *sharded_view;
      RtEvent all_mapped_event;
      bool exchange_complete;
    };

    /**
     * \class ReplIndexAttachOp
     * An index space attach operation that is aware
     * that it is executing in a control replicated context
     */
    class ReplIndexAttachOp : public IndexAttachOp {
    public:
      ReplIndexAttachOp(Runtime *rt);
      ReplIndexAttachOp(const ReplIndexAttachOp &rhs);
      virtual ~ReplIndexAttachOp(void);
    public:
      ReplIndexAttachOp& operator=(const ReplIndexAttachOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void check_point_requirements(
                    const std::vector<IndexSpace> &spaces);
      virtual bool are_all_direct_children(bool local);
      virtual RtEvent find_coregions(PointAttachOp *point, LogicalRegion region,
          InstanceSet &instances, ApUserEvent &attached_event);
    public:
      void initialize_replication(ReplicateContext *ctx);
    protected:
      IndexAttachExchange *collective;
      ShardingFunction *sharding_function;
      IndexAttachCoregions *attach_coregions_collective;
    };

    /**
     * \class ReplDetachOp
     * An detach operation that is aware that it is being
     * executed in a control replicated context.
     */
    class ReplDetachOp : public DetachOp {
    public:
      ReplDetachOp(Runtime *rt);
      ReplDetachOp(const ReplDetachOp &rhs);
      virtual ~ReplDetachOp(void);
    public:
      ReplDetachOp& operator=(const ReplDetachOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
    public:
      // Help for unordered detachments
      void record_unordered_kind(
        std::map<std::pair<LogicalRegion,FieldID>,ReplDetachOp*> &detachments);
    public:
      RtBarrier resource_barrier;
    };

    /**
     * \class ReplIndexDetachOp
     * An index space detach operation that is aware
     * that it is executing in a control replicated context
     */
    class ReplIndexDetachOp : public IndexDetachOp {
    public:
      ReplIndexDetachOp(Runtime *rt);
      ReplIndexDetachOp(const ReplIndexDetachOp &rhs);
      virtual ~ReplIndexDetachOp(void);
    public:
      ReplIndexDetachOp& operator=(const ReplIndexDetachOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
    protected:
      ShardingFunction *sharding_function;
    };

    /**
     * \class ReplTraceOp
     * Base class for all replicated trace operations
     */
    class ReplTraceOp : public ReplFenceOp {
    public:
      ReplTraceOp(Runtime *rt);
      ReplTraceOp(const ReplTraceOp &rhs);
      virtual ~ReplTraceOp(void);
    public:
      ReplTraceOp& operator=(const ReplTraceOp &rhs);
    public:
      virtual void execute_dependence_analysis(void);
      virtual void sync_for_replayable_check(void);
      virtual bool exchange_replayable(ReplicateContext *ctx, bool replayable);
      virtual void sync_compute_frontiers(RtEvent precondition);
    protected:
      LegionTrace *local_trace;
    };
    
    /**
     * \class ReplTraceCaptureOp
     * Control replicated version of the TraceCaptureOp
     */
    class ReplTraceCaptureOp : public ReplTraceOp {
    public:
      static const AllocationType alloc_type = TRACE_CAPTURE_OP_ALLOC;
    public:
      ReplTraceCaptureOp(Runtime *rt);
      ReplTraceCaptureOp(const ReplTraceCaptureOp &rhs);
      virtual ~ReplTraceCaptureOp(void);
    public:
      ReplTraceCaptureOp& operator=(const ReplTraceCaptureOp &rhs);
    public:
      void initialize_capture(ReplicateContext *ctx, Provenance *provenance,
          bool has_blocking_call, bool remove_trace_reference);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void sync_for_replayable_check(void);
      virtual bool exchange_replayable(ReplicateContext *ctx, bool replayable);
      virtual void sync_compute_frontiers(RtEvent precondition);
    protected:
      PhysicalTemplate *current_template;
      RtBarrier recording_fence;
      CollectiveID replayable_collective_id;
      CollectiveID replay_sync_collective_id;
      CollectiveID sync_compute_frontiers_collective_id;
      bool has_blocking_call;
      bool remove_trace_reference;
      bool is_recording;
    };

    /**
     * \class ReplTraceCompleteOp
     * Control replicated version of TraceCompleteOp
     */
    class ReplTraceCompleteOp : public ReplTraceOp {
    public:
      static const AllocationType alloc_type = TRACE_COMPLETE_OP_ALLOC;
    public:
      ReplTraceCompleteOp(Runtime *rt);
      ReplTraceCompleteOp(const ReplTraceCompleteOp &rhs);
      virtual ~ReplTraceCompleteOp(void);
    public:
      ReplTraceCompleteOp& operator=(const ReplTraceCompleteOp &rhs);
    public:
      void initialize_complete(ReplicateContext *ctx, Provenance *provenance,
                               bool has_blocking_call);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void sync_for_replayable_check(void);
      virtual bool exchange_replayable(ReplicateContext *ctx, bool replayable);
      virtual void sync_compute_frontiers(RtEvent precondition);
    protected:
      PhysicalTemplate *current_template;
      ApEvent template_completion;
      RtBarrier recording_fence;
      CollectiveID replayable_collective_id;
      CollectiveID replay_sync_collective_id;
      CollectiveID sync_compute_frontiers_collective_id;
      bool replayed;
      bool has_blocking_call;
      bool is_recording;
    };

    /**
     * \class ReplTraceReplayOp
     * Control replicated version of TraceReplayOp
     */
    class ReplTraceReplayOp : public ReplTraceOp {
    public:
      static const AllocationType alloc_type = TRACE_REPLAY_OP_ALLOC;
    public:
      ReplTraceReplayOp(Runtime *rt);
      ReplTraceReplayOp(const ReplTraceReplayOp &rhs);
      virtual ~ReplTraceReplayOp(void);
    public:
      ReplTraceReplayOp& operator=(const ReplTraceReplayOp &rhs);
    public:
      void initialize_replay(ReplicateContext *ctx, LegionTrace *trace,
                             Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
    protected:
      // This a parameter that controls how many rounds of template
      // selection we want shards to go through before giving up
      // and doing a capture. The trick is to get all the shards to
      // agree on the template. Each round will have each shard 
      // propose twice as many viable traces the previous round so
      // we get some nice exponential back-off properties. Increase
      // the number of rounds if you want them to try for longer.
      static const int TRACE_SELECTION_ROUNDS = 2;
      CollectiveID trace_selection_collective_ids[TRACE_SELECTION_ROUNDS];
    };

    /**
     * \class ReplTraceBeginOp
     * Control replicated version of trace begin op
     */
    class ReplTraceBeginOp : public ReplTraceOp {
    public:
      static const AllocationType alloc_type = TRACE_BEGIN_OP_ALLOC;
    public:
      ReplTraceBeginOp(Runtime *rt);
      ReplTraceBeginOp(const ReplTraceBeginOp &rhs);
      virtual ~ReplTraceBeginOp(void);
    public:
      ReplTraceBeginOp& operator=(const ReplTraceBeginOp &rhs);
    public:
      void initialize_begin(ReplicateContext *ctx, LegionTrace *trace,
                            Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    };

    /**
     * \class ReplTraceSummaryOp
     * Control replicated version of TraceSummaryOp
     */
    class ReplTraceSummaryOp : public ReplTraceOp {
    public:
      static const AllocationType alloc_type = TRACE_SUMMARY_OP_ALLOC;
    public:
      ReplTraceSummaryOp(Runtime *rt);
      ReplTraceSummaryOp(const ReplTraceSummaryOp &rhs);
      virtual ~ReplTraceSummaryOp(void);
    public:
      ReplTraceSummaryOp& operator=(const ReplTraceSummaryOp &rhs);
    public:
      void initialize_summary(ReplicateContext *ctx,
                              ShardedPhysicalTemplate *tpl,
                              Operation *invalidator,
                              Provenance *provenance);
      void perform_logging(void);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
    protected:
      ShardedPhysicalTemplate *current_template;
    };

    /**
     * \class ShardMapping
     * A mapping from the shard IDs to their address spaces
     */
    class ShardMapping : public Collectable {
    public:
      ShardMapping(void);
      ShardMapping(const ShardMapping &rhs);
      ShardMapping(const std::vector<AddressSpaceID> &spaces);
      ~ShardMapping(void);
    public:
      ShardMapping& operator=(const ShardMapping &rhs);
      AddressSpaceID operator[](unsigned idx) const;
      AddressSpaceID& operator[](unsigned idx);
    public:
      inline size_t size(void) const { return address_spaces.size(); }
      inline void resize(size_t size) { address_spaces.resize(size); }
    public:
      void pack_mapping(Serializer &rez) const;
      void unpack_mapping(Deserializer &derez);
    protected:
      std::vector<AddressSpaceID> address_spaces;
    };

    /**
     * \class CollectiveMapping
     * A collective mapping is an ordering of unique address spaces
     * and can be used to construct broadcast and reduction trees.
     */
    class CollectiveMapping : public Collectable {
    public:
      CollectiveMapping(const std::vector<AddressSpaceID> &spaces,size_t radix);
      CollectiveMapping(const ShardMapping &shard_mapping, size_t radix);
      CollectiveMapping(Deserializer &derez, size_t total_spaces);
    public:
      inline AddressSpaceID operator[](unsigned idx) const
        { return unique_sorted_spaces.get_index(idx); }
      inline size_t size(void) const { return total_spaces; }
      bool operator==(const CollectiveMapping &rhs) const;
      bool operator!=(const CollectiveMapping &rhs) const;
    public:
      AddressSpaceID get_parent(const AddressSpaceID origin, 
                                const AddressSpaceID local) const;
      void get_children(const AddressSpaceID origin, const AddressSpaceID local,
                        std::vector<AddressSpaceID> &children) const;
      inline bool contains(const AddressSpaceID space) const
        { return unique_sorted_spaces.contains(space); }
      AddressSpaceID find_nearest(AddressSpaceID space) const;
    public:
      void pack(Serializer &rez) const;
    protected:
      inline unsigned find_index(const AddressSpaceID space) const
        { return unique_sorted_spaces.find_index(space); }
      unsigned convert_to_offset(unsigned index, unsigned origin) const;
      unsigned convert_to_index(unsigned offset, unsigned origin) const;
    protected:
      NodeSet unique_sorted_spaces;
      size_t total_spaces;
      size_t radix;
    };

    /**
     * \class ShardManager
     * This is a class that manages the execution of one or
     * more shards for a given control replication context on
     * a single node. It provides support for doing broadcasts,
     * reductions, and exchanges of information between the 
     * variaous shard tasks.
     */
    class ShardManager : public Mapper::SelectShardingFunctorInput, 
                          public Collectable {
    public:
      struct ShardManagerLaunchArgs :
        public LgTaskArgs<ShardManagerLaunchArgs> {
      public:
        static const LgTaskID TASK_ID = LG_CONTROL_REP_LAUNCH_TASK_ID;
      public:
        ShardManagerLaunchArgs(ShardTask *s)
          : LgTaskArgs<ShardManagerLaunchArgs>(s->get_unique_op_id()), 
            shard(s) { }
      public:
        ShardTask *const shard;
      };
      struct ShardManagerDeleteArgs :
        public LgTaskArgs<ShardManagerDeleteArgs> {
      public:
        static const LgTaskID TASK_ID = LG_CONTROL_REP_DELETE_TASK_ID;
      public:
        ShardManager *manager;
      };
    public:
      enum BroadcastMessageKind {
        RESOURCE_UPDATE_KIND,
        CREATED_REGION_UPDATE_KIND,
      };
    public:
      struct AttachDeduplication {
      public:
        AttachDeduplication(void) : done_count(0) { }
      public:
        RtUserEvent pending;
        std::vector<const IndexAttachLauncher*> launchers; 
        std::map<LogicalRegion,const IndexAttachLauncher*> owners;
        unsigned done_count;
      };
    public:
      ShardManager(Runtime *rt, ReplicationID repl_id, 
                   bool control, bool top, bool isomorphic_points,
                   const Domain &shard_domain,
                   std::vector<DomainPoint> &&shard_points,
                   std::vector<DomainPoint> &&sorted_points,
                   std::vector<ShardID> &&shard_lookup,
                   AddressSpaceID owner_space, SingleTask *original = NULL,
                   RtBarrier shard_task_barrier = RtBarrier::NO_RT_BARRIER);
      ShardManager(const ShardManager &rhs) = delete;
      ~ShardManager(void);
    public:
      ShardManager& operator=(const ShardManager &rhs) = delete;
    public:
      inline RtBarrier get_shard_task_barrier(void) const
        { return shard_task_barrier; }
      inline ShardMapping& get_mapping(void) const
        { return *address_spaces; }
      inline CollectiveMapping& get_collective_mapping(void) const
        { return *collective_mapping; }
      inline AddressSpaceID get_shard_space(ShardID sid) const
        { return (*address_spaces)[sid]; }    
      inline bool is_first_local_shard(ShardTask *task) const
        { return (local_shards[0] == task); }
      inline const std::set<AddressSpace>& get_unique_shard_spaces(void) const
        { return unique_shard_spaces; }
      inline ReplicateContext* find_local_context(void) const
        { return local_shards[0]->get_shard_execution_context(); }
    public:
      void set_shard_mapping(const std::vector<Processor> &shard_mapping);
      void set_address_spaces(const std::vector<AddressSpaceID> &spaces);
      void create_callback_barrier(size_t arrival_count);
      ShardTask* create_shard(ShardID id, Processor target);
      void extract_event_preconditions(const std::deque<InstanceSet> &insts);
      void launch(const std::vector<bool> &virtual_mapped);
      void distribute_shards(AddressSpaceID target,
                             const std::vector<ShardTask*> &shards);
      void unpack_shards_and_launch(Deserializer &derez);
      void launch_shard(ShardTask *task,
                        RtEvent precondition = RtEvent::NO_RT_EVENT) const;
      EquivalenceSet* get_initial_equivalence_set(unsigned idx) const;
      EquivalenceSet* deduplicate_equivalence_set_creation(RegionNode *node,
                      const FieldMask &mask, DistributedID did, bool &first);
      void deduplicate_attaches(const IndexAttachLauncher &launcher,
                                std::vector<unsigned> &indexes);
      // Return true if we have a shard on every address space
      bool is_total_sharding(void);
    public:
      void handle_post_mapped(bool local, RtEvent precondition);
      void handle_post_execution(FutureInstance *instance, void *metadata,
                                 size_t metasize, bool local);
      RtEvent trigger_task_complete(bool local, ApEvent effects_done);
      void trigger_task_commit(bool local);
    public:
      void send_collective_message(ShardID target, Serializer &rez);
      void handle_collective_message(Deserializer &derez);
    public:
      void send_future_map_request(ShardID target, Serializer &rez);
      void handle_future_map_request(Deserializer &derez);
    public:
      void send_disjoint_complete_request(ShardID target, Serializer &rez);
      void handle_disjoint_complete_request(Deserializer &derez);
    public:
      void send_intra_space_dependence(ShardID target, Serializer &rez);
      void handle_intra_space_dependence(Deserializer &derez);
    public:
      void broadcast_resource_update(ShardTask *source, Serializer &rez,
                                     std::set<RtEvent> &applied_events);
    public:
      void broadcast_created_region_contexts(ShardTask *source, Serializer &rez,
                                             std::set<RtEvent> &applied_events);
    protected:
      void broadcast_message(ShardTask *source, Serializer &rez,
                BroadcastMessageKind kind, std::set<RtEvent> &applied_events);
      void handle_broadcast(Deserializer &derez);
    public:
      void send_trace_event_request(ShardedPhysicalTemplate *physical_template,
                          ShardID shard_source, AddressSpaceID template_source, 
                          size_t template_index, ApEvent event, 
                          AddressSpaceID event_space, RtUserEvent done_event);
      void send_trace_event_response(ShardedPhysicalTemplate *physical_template,
                          AddressSpaceID template_source, ApEvent event,
                          ApBarrier result, RtUserEvent done_event);
      void send_trace_frontier_request(ShardedPhysicalTemplate *physical_template,
                          ShardID shard_source, AddressSpaceID template_source, 
                          size_t template_index, ApEvent event, 
                          AddressSpaceID event_space, unsigned frontier,
                          RtUserEvent done_event);
      void send_trace_frontier_response(ShardedPhysicalTemplate *physical_template,
                          AddressSpaceID template_source, unsigned frontier,
                          ApBarrier result, RtUserEvent done_event);
      void send_trace_update(ShardID target, Serializer &rez);
      void handle_trace_update(Deserializer &derez, AddressSpaceID source);
    public:
      static void handle_launch(const void *args);
      static void handle_delete(const void *args);
    public:
      static void handle_launch(Deserializer &derez, Runtime *rt, 
                                AddressSpaceID source);
      static void handle_delete(Deserializer &derez, Runtime *rt);
      static void handle_post_mapped(Deserializer &derez, Runtime *rt);
      static void handle_post_execution(Deserializer &derez, Runtime *rt);
      static void handle_trigger_complete(Deserializer &derez, Runtime *rt);
      static void handle_trigger_commit(Deserializer &derez, Runtime *rt);
      static void handle_collective_message(Deserializer &derez, Runtime *rt);
      static void handle_future_map_request(Deserializer &derez, Runtime *rt);
      static void handle_disjoint_complete_request(Deserializer &derez, 
                                                   Runtime *rt);
      static void handle_intra_space_dependence(Deserializer &derez, 
                                                Runtime *rt);
      static void handle_broadcast_update(Deserializer &derez, Runtime *rt);
      static void handle_trace_event_request(Deserializer &derez, Runtime *rt,
                                             AddressSpaceID request_source);
      static void handle_trace_event_response(Deserializer &derez);
      static void handle_trace_frontier_request(Deserializer &derez,Runtime *rt,
                                                AddressSpaceID request_source);
      static void handle_trace_frontier_response(Deserializer &derez);
      static void handle_trace_update(Deserializer &derez, Runtime *rt,
                                      AddressSpaceID source);
      static void handle_barrier_refresh(Deserializer &derez, Runtime *rt);
    public:
      ShardingFunction* find_sharding_function(ShardingID sid);
    public:
#ifdef LEGION_USE_LIBDL
      void perform_global_registration_callbacks(
                     Realm::DSOReferenceImplementation *dso, const void *buffer,
                     size_t buffer_size, bool withargs, size_t dedup_tag,
                     RtEvent local_done, RtEvent global_done,
                     std::set<RtEvent> &preconditions);
#endif
      bool perform_semantic_attach(void);
    public:
      Runtime *const runtime;
      const ReplicationID repl_id;
      const AddressSpaceID owner_space;
      const std::vector<DomainPoint> shard_points;
      const std::vector<DomainPoint> sorted_points;
      const std::vector<ShardID> shard_lookup;
      const Domain shard_domain;
      const size_t total_shards;
      SingleTask *const original_task;
      const bool control_replicated;
      const bool top_level_task;
      const bool isomorphic_points;
    protected:
      mutable LocalLock                manager_lock;
      // Inheritted from Mapper::SelectShardingFunctorInput
      // std::vector<Processor>        shard_mapping;
      ShardMapping*                    address_spaces;
      CollectiveMapping*               collective_mapping;
      std::vector<ShardTask*>          local_shards;
      std::vector<EquivalenceSet*>     mapped_equivalence_sets;
    protected:
      // There are four kinds of signals that come back from 
      // the execution of the shards:
      // - mapping complete
      // - future result
      // - task complete
      // - task commit
      // The owner applies these to the original task object only
      // after they have occurred for all the shards
      unsigned    local_mapping_complete, remote_mapping_complete;
      unsigned    local_execution_complete, remote_execution_complete;
      unsigned    trigger_local_complete, trigger_remote_complete;
      unsigned    trigger_local_commit,   trigger_remote_commit;
      unsigned    remote_constituents;
      unsigned    semantic_attach_counter;
      FutureInstance *local_future_result;
      std::set<RtEvent> mapping_preconditions;
    protected:
      RtBarrier shard_task_barrier;
      RtBarrier callback_barrier;
    protected:
      std::map<ShardingID,ShardingFunction*> sharding_functions;
    protected:
      std::map<DistributedID,std::pair<EquivalenceSet*,size_t> > 
                                        created_equivalence_sets;
      // ApEvents describing the completion of each shard
      std::set<ApEvent> shard_effects;
    protected:
      // A unique set of address spaces on which shards exist 
      std::set<AddressSpaceID> unique_shard_spaces;
#ifdef LEGION_USE_LIBDL
      std::set<Runtime::RegistrationKey> unique_registration_callbacks;
#endif
    protected:
      AttachDeduplication *attach_deduplication;
    };

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_REPLICATION_H__
