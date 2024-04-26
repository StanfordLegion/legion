/* Copyright 2024 Stanford University, NVIDIA Corporation
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
        CloseCheckValue(Operation *op, RtBarrier barrier,
                        RegionTreeNode *node, bool read_only);
      public:
        bool operator==(const CloseCheckValue &rhs) const;
      public:
        unsigned operation_index;
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
      virtual MessageKind get_message_kind(void) const = 0;
      // We guarantee that these methods will be called atomically
      virtual void pack_collective(Serializer &rez) const = 0;
      virtual void unpack_collective(Deserializer &derez) = 0;
    public:
      virtual void perform_collective_async(RtEvent pre = RtEvent::NO_RT_EVENT);
      virtual RtEvent perform_collective_wait(bool block = true);
      virtual void handle_collective_message(Deserializer &derez);
      virtual RtEvent post_broadcast(void) { return RtEvent::NO_RT_EVENT; }
      // Use this method in case we don't actually end up using the collective
      virtual void elide_collective(void);
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
      GatherCollective(ReplicateContext *ctx, 
                       CollectiveID id, ShardID origin);
      virtual ~GatherCollective(void);
    public:
      virtual MessageKind get_message_kind(void) const = 0;
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
      RtEvent get_done_event(void);
      // Use this method in case we don't actually end up using the collective
      virtual void elide_collective(void);
    protected:
      void send_message(void);
      int compute_expected_notifications(void) const;
    public:
      const ShardID target;
      const int shard_collective_radix;
      const int expected_notifications;
    private:
      union DoneEvent {
        DoneEvent(void) : to_trigger(RtUserEvent::NO_RT_USER_EVENT) { }
        RtUserEvent to_trigger;
        RtEvent postcondition;
      } done_event;
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
      virtual MessageKind get_message_kind(void) const = 0;
      // We guarantee that these methods will be called atomically
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage) = 0;
      virtual void unpack_collective_stage(Deserializer &derez, int stage) = 0;
    public:
      virtual void perform_collective_async(RtEvent pre = RtEvent::NO_RT_EVENT);
      virtual RtEvent perform_collective_wait(bool block = true);
      virtual void handle_collective_message(Deserializer &derez);
      // Use this method in case we don't actually end up using the collective
      virtual void elide_collective(void);
      inline RtEvent get_done_event(void) const { return done_event; }
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
    protected:
      struct PendingReduction {
        FutureInstance *instance;
        ApEvent precondition;
        ApUserEvent postcondition;
      };
    public:
      FutureAllReduceCollective(Operation *op, CollectiveIndexLocation loc, 
          ReplicateContext *ctx, ReductionOpID redop_id,
          const ReductionOp *redop);
      FutureAllReduceCollective(Operation *op, ReplicateContext *ctx, 
          CollectiveID id, ReductionOpID redop_id, const ReductionOp *redop);
      virtual ~FutureAllReduceCollective(void);
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_FUTURE_ALLREDUCE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
      virtual RtEvent post_complete_exchange(void);
      virtual void elide_collective(void);
    public:
      void set_shadow_instance(FutureInstance *shadow);
      RtEvent async_reduce(FutureInstance *instance, ApEvent &ready_event);
    protected:
      ApEvent perform_reductions(const std::map<ShardID,PendingReduction> &red);
      void create_shadow_instance(void);
    public:
      Operation *const op;
      const ReductionOp *const redop;
      const ReductionOpID redop_id;
    protected:
      const ApUserEvent finished; 
      std::map<int,std::map<ShardID,PendingReduction> > pending_reductions;
      FutureInstance *instance;
      FutureInstance *shadow_instance;
      ApEvent instance_ready;
      ApEvent shadow_ready;
      std::vector<ApEvent> shadow_reads;
      int current_stage;
      bool pack_shadow;
    };

    /**
     * \class FutureBroadcast
     * This class will broadast a future result out to all all shards
     */
    class FutureBroadcastCollective : public BroadcastCollective {
    public:
      FutureBroadcastCollective(ReplicateContext *ctx, 
          CollectiveIndexLocation loc, ShardID origin, Operation *op);
      FutureBroadcastCollective(const FutureBroadcastCollective &rhs) = delete;
      virtual ~FutureBroadcastCollective(void);
    public:
      FutureBroadcastCollective& operator=(
                                const FutureBroadcastCollective &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_FUTURE_ALLREDUCE; }
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
      virtual void elide_collective(void);
      virtual RtEvent post_broadcast(void);
    public:
      RtEvent async_broadcast(FutureInstance *instance, 
          ApEvent precondition = ApEvent::NO_AP_EVENT,
          RtEvent postcondition = RtEvent::NO_RT_EVENT);
    public:
      Operation *const op;
      const ApUserEvent finished;
    protected:
      FutureInstance *instance;
      ApEvent write_event;
      mutable std::vector<ApEvent> read_events;
      RtEvent postcondition;
    };

    /**
     * \class FutureReduction
     * This class builds a reduction tree of futures down to a single
     * future value.
     */
    class FutureReductionCollective : public GatherCollective {
    public:
      FutureReductionCollective(ReplicateContext *ctx,
          CollectiveIndexLocation loc, ShardID origin,
          Operation *op, FutureBroadcastCollective *broadcast,
          const ReductionOp *redop, ReductionOpID redop_id);
      FutureReductionCollective(const FutureReductionCollective &rhs) = delete;
      virtual ~FutureReductionCollective(void);
    public:
      FutureReductionCollective& operator=(
                                const FutureReductionCollective &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_FUTURE_REDUCTION; }
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
      virtual RtEvent post_gather(void);
    public:
      void async_reduce(FutureInstance *instance, ApEvent precondition);
    protected:
      void perform_reductions(void) const;
    public:
      Operation *const op;
      FutureBroadcastCollective *const broadcast;
      const ReductionOp *const redop;
      const ReductionOpID redop_id;
    protected:
      FutureInstance *instance;
      mutable ApEvent ready;
      std::map<ShardID,std::pair<FutureInstance*,ApEvent> > pending_reductions;
    };

    /**
     * \class AllReduceCollective
     * This shard collective has equivalent functionality to 
     * MPI All Reduce in that it will take a value from each
     * shard and reduce it down to a final value using a 
     * Legion reduction operator. We'll build this on top
     * of the AllGatherCollective
     */
    template<typename REDOP, bool INORDER>
    class AllReduceCollective : public AllGatherCollective<INORDER> {
    public:
      AllReduceCollective(CollectiveIndexLocation loc, ReplicateContext *ctx);
      AllReduceCollective(ReplicateContext *ctx, CollectiveID id);
      virtual ~AllReduceCollective(void);
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_VALUE_ALLREDUCE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void async_all_reduce(typename REDOP::RHS value);
      RtEvent wait_all_reduce(bool block = true);
      typename REDOP::RHS sync_all_reduce(typename REDOP::RHS value);
      typename REDOP::RHS get_result(void);
    protected:
      typename REDOP::RHS value;
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_VALUE_BROADCAST; }
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_VALUE_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage)
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
      BufferBroadcast(CollectiveID id, ReplicateContext *ctx)
        : BroadcastCollective(ctx, id, ctx->owner_shard->shard_id),
          buffer(NULL), size(0), own(false) { }
      BufferBroadcast(CollectiveID id, ShardID origin,
                      ReplicateContext *ctx)
        : BroadcastCollective(ctx, id, origin),
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_BUFFER_BROADCAST; }
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_SHARD_SYNC_TREE; }
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
      virtual RtEvent post_gather(void);
    protected:
      std::vector<RtEvent> postconditions;
      const RtEvent done;
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_SHARD_EVENT_TREE; }
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_SINGLE_TASK_TREE; }
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
    protected:
      FutureImpl *const future;
      size_t future_size;
      bool has_future_size;
    };

    /**
     * \class CrossProductCollective
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_CROSS_PRODUCT_PARTITION; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_SHARDING_GATHER_COLLECTIVE; }
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_INDIRECT_COPY_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
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
    class FieldDescriptorExchange : public AllGatherCollective<true> {
    public:
      FieldDescriptorExchange(ReplicateContext *ctx, CollectiveID id,
                              std::vector<FieldDataDescriptor> &descriptors);
      FieldDescriptorExchange(const FieldDescriptorExchange &rhs) = delete;
      virtual ~FieldDescriptorExchange(void);
    public:
      FieldDescriptorExchange& operator=(
                              const FieldDescriptorExchange &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_FIELD_DESCRIPTOR_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    protected:
      std::vector<FieldDataDescriptor> &descriptors;
    };

    /**
     * \class FieldDescriptorGather
     * Gather all of the field descriptors to a particular shard and
     * track the merge of all the ready events
     */
    class FieldDescriptorGather : public GatherCollective {
    public:
      FieldDescriptorGather(ReplicateContext *ctx, CollectiveID id,
                            std::vector<FieldDataDescriptor> &descriptors,
                            std::map<DomainPoint,Domain> &remote_targets);
      FieldDescriptorGather(const FieldDescriptorGather &rhs) = delete;
      virtual ~FieldDescriptorGather(void);
    public:
      FieldDescriptorGather& operator=(
                            const FieldDescriptorGather &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_FIELD_DESCRIPTOR_GATHER; }
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
    public:
      void contribute_instances(ApEvent instances_ready);
      ApEvent get_ready_event(void);
    protected:
      std::vector<FieldDataDescriptor> &descriptors;
      std::map<DomainPoint,Domain> &remote_targets;
      std::vector<ApEvent> ready_events;
    };

    /**
     * \class DeppartResultScatter
     * Scatter the results of a dependent partitioning operation
     * back across the shards so they can fill in their nodes
     */
    class DeppartResultScatter : public BroadcastCollective {
    public:
      DeppartResultScatter(ReplicateContext *ctx, CollectiveID id,
                           std::vector<DeppartResult> &results);
      DeppartResultScatter(const DeppartResultScatter &rhs) = delete;
      virtual ~DeppartResultScatter(void);
    public:
      DeppartResultScatter& operator=(const DeppartResultScatter &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_DEPPART_RESULT_SCATTER; }
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
    public:
      void broadcast_results(ApEvent done_event);
      inline ApEvent get_done_event(void) { return done_event; }
    public:
      std::vector<DeppartResult> &results;
      const ApUserEvent done_event;
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_BUFFER_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
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
      FutureNameExchange(ReplicateContext *ctx, CollectiveIndexLocation loc);
      FutureNameExchange(const FutureNameExchange &rhs);
      virtual ~FutureNameExchange(void);
    public:
      FutureNameExchange& operator=(const FutureNameExchange &rhs);
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_FUTURE_NAME_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void exchange_future_names(std::map<DomainPoint,FutureImpl*> &futures);
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_MUST_EPOCH_MAPPING_BROADCAST; }
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_MUST_EPOCH_MAPPING_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_MUST_EPOCH_DEPENDENCE_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_MUST_EPOCH_COMPLETION_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
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
     * \class CheckCollectiveMapping
     * A class for exchanging the names of instances used for collective mapping
     */
    class CheckCollectiveMapping : public AllGatherCollective<true/*inorder*/> {
    public:
      CheckCollectiveMapping(ReplicateContext *ctx, CollectiveID id);
      CheckCollectiveMapping(const CheckCollectiveMapping&) = delete;
      virtual ~CheckCollectiveMapping(void);
    public:
      CheckCollectiveMapping& operator=(const CheckCollectiveMapping&) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_CHECK_COLLECTIVE_MAPPING; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void verify(const InstanceSet &instances, MapperManager *mapper);
    protected:
      typedef LegionVector<std::pair<ShardID,FieldMask > > ShardFields;
      std::map<PhysicalInstance,ShardFields> mapped_instances;
    };

    /**
     * \class CheckCollectiveSources
     * A class for exchanging the names of source instances for confirming
     * that all shards have listed the same instances for mapping
     */
    class CheckCollectiveSources : public BroadcastCollective {
    public:
      CheckCollectiveSources(ReplicateContext *ctx, CollectiveID id);
      CheckCollectiveSources(const CheckCollectiveSources&) = delete;
      virtual ~CheckCollectiveSources(void);
    public:
      CheckCollectiveSources& operator=(const CheckCollectiveSources&) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_CHECK_COLLECTIVE_SOURCES; }
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
    public:
      bool verify(const std::vector<PhysicalManager*> &instances);
    protected:
      std::vector<DistributedID> source_instances;
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_TEMPLATE_INDEX_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_UNORDERED_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void start_unordered_exchange(const std::vector<Operation*> &operations);
      void find_ready_operations(std::vector<Operation*> &ready_operations);
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
      std::map<std::pair<LogicalRegion,FieldID>,unsigned> region_detach_counts;
      std::map<
        std::pair<LogicalPartition,FieldID>,unsigned> partition_detach_counts;
    protected:
      std::map<IndexSpace,ReplDeletionOp*> index_space_deletions;
      std::map<IndexPartition,ReplDeletionOp*> index_partition_deletions;
      std::map<FieldSpace,ReplDeletionOp*> field_space_deletions;
      // Use the lowest field ID here as the key
      std::map<std::pair<FieldSpace,FieldID>,ReplDeletionOp*> field_deletions;
      std::map<LogicalRegion,ReplDeletionOp*> logical_region_deletions;
      // Use the lowest field ID here as the key
      std::map<std::pair<LogicalRegion,FieldID>,Operation*> region_detachments;
      std::map<
        std::pair<LogicalPartition,FieldID>,Operation*> partition_detachments;
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
                             Future to_complete, void *output);
      ConsensusMatchExchange(const ConsensusMatchExchange &rhs);
      virtual ~ConsensusMatchExchange(void);
    public:
      ConsensusMatchExchange& operator=(const ConsensusMatchExchange &rhs);
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_CONSENSUS_MATCH; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      bool match_elements_async(const void *input, size_t num_elements);
      virtual void complete_exchange(void);
    protected:
      Future to_complete;
      T *const output;
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_VERIFY_CONTROL_REPLICATION_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      typedef std::map<std::pair<uint64_t,uint64_t>,ShardID> ShardHashes;
      const ShardHashes& exchange(const uint64_t hash[2]);
    public:
      ShardHashes unique_hashes;
    };

    /**
     * \class OutputExtentExchange
     * This class exchanges sizes of output subregions that are globally
     * indexed.
     */
    class OutputExtentExchange : public AllGatherCollective<false> {
    public:
      typedef std::map<DomainPoint,DomainPoint> OutputExtentMap;
    public:
      OutputExtentExchange(ReplicateContext *ctx, ReplIndexTask *owner,
                           CollectiveIndexLocation loc,
                           std::vector<OutputExtentMap> &all_output_extents);
      OutputExtentExchange(const OutputExtentExchange &rhs) = delete;
      virtual ~OutputExtentExchange(void);
    public:
      OutputExtentExchange& operator=(const OutputExtentExchange &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_OUTPUT_SIZE_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
      virtual RtEvent post_complete_exchange(void);
    public:
      ReplIndexTask *const owner;
      std::vector<OutputExtentMap> &all_output_extents;
    };

    /**
     as \class IndexAttachLaunchSpace
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_INDEX_ATTACH_LAUNCH_SPACE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_INDEX_ATTACH_UPPER_BOUND; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_INDEX_ATTACH_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void exchange_spaces(std::vector<IndexSpace> &spaces);
      size_t get_spaces(std::vector<IndexSpace> &spaces, unsigned &local_start);
      IndexSpaceNode* get_launch_space(void); 
    protected:
      std::map<ShardID,std::vector<IndexSpace> > shard_spaces;
    };

    /**
     * \class ShardParticipants
     * Find the shard participants in a replicated context
     */
    class ShardParticipantsExchange : public AllGatherCollective<false> {
    public:
      ShardParticipantsExchange(ReplicateContext *ctx,
                                CollectiveIndexLocation loc);
      ShardParticipantsExchange(const ShardParticipantsExchange &rhs) = delete;
      virtual ~ShardParticipantsExchange(void);
    public:
      ShardParticipantsExchange& operator=(
                                const ShardParticipantsExchange &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_SHARD_PARTICIPANTS_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void exchange(bool participating);
      bool find_shard_participants(std::vector<ShardID> &shards);
    protected:
      std::set<ShardID> participants;
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_IMPLICIT_SHARDING_FUNCTOR; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
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
     * \class CreateCollectiveFillView
     * This collective checks to see if all the shards picked the
     * same fill view, and if not, then will make a new collective
     * fill view for the shards to use
     */
    class CreateCollectiveFillView : public AllGatherCollective<false> {
    public:
      CreateCollectiveFillView(ReplicateContext *ctx, CollectiveID id,
                               FillOp *op, DistributedID fill_view,
                               DistributedID fresh_did);
      CreateCollectiveFillView(const CreateCollectiveFillView &rhs) = delete;
      virtual ~CreateCollectiveFillView(void) { }
    public:
      CreateCollectiveFillView& operator=(
                               const CreateCollectiveFillView &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_CREATE_FILL_VIEW; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
      virtual RtEvent post_complete_exchange(void);
    protected:
      FillOp *const fill_op;
      const DistributedID fresh_did;
      std::set<DistributedID> selected_views;
    };

    /**
     * \class CollectiveViewRendezvous
     * A gather collective for performing the rendezvous for the creation
     * of collective views across all the shards
     */
    class CollectiveViewRendezvous : public GatherCollective {
    public:
      typedef CollectiveViewCreatorBase::RendezvousKey RendezvousKey;
      typedef CollectiveViewCreatorBase::RendezvousResult RendezvousResult;
      typedef CollectiveViewCreatorBase::CollectiveRendezvous 
        CollectiveRendezvous;
      class Finalizer {
      public:
        virtual void finalize_collective_mapping(const RendezvousKey &key,
                std::map<LogicalRegion,CollectiveRendezvous> &rendezvous) = 0;
      };
    public:
      CollectiveViewRendezvous(CollectiveID, ReplicateContext *ctx, 
          Operation *op, Finalizer *finalizer, 
          const RendezvousKey &key, RegionTreeID tid);
      CollectiveViewRendezvous(const CollectiveViewRendezvous &rhs) = delete;
      virtual ~CollectiveViewRendezvous(void);
    public:
      CollectiveViewRendezvous& operator=(
          const CollectiveViewRendezvous &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_VIEW_RENDEZVOUS; }
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
      virtual RtEvent post_gather(void);
    public:
      void perform_rendezvous(
          std::map<LogicalRegion,CollectiveRendezvous> &rendezvous);
    public:
      const RendezvousKey key;
      Operation *const op;
      Finalizer *const finalizer;
    protected:
      std::map<LogicalRegion,CollectiveRendezvous> rendezvous;
    };

    /**
     * \class CollectiveVersioningRendezvous
     * A gather collective to perform the rendezvous for performing
     * versioning analysis for collective region requirements
     */
    class CollectiveVersioningRendezvous : public GatherCollective {
    public:
      typedef CollectiveVersioningBase::RegionVersioning RegionVersioning;
      class Finalizer {
      public:
        virtual void finalize_collective_versioning(unsigned index,
            unsigned parent_req_index,
            LegionMap<LogicalRegion,RegionVersioning> &pending_versions) = 0;
      };
    public:
      CollectiveVersioningRendezvous(CollectiveID, ReplicateContext *ctx, 
          Operation *op, Finalizer *finalizer, ShardID owner, unsigned index);
      CollectiveVersioningRendezvous(
          const CollectiveVersioningRendezvous &rhs) = delete;
      virtual ~CollectiveVersioningRendezvous(void);
    public:
      CollectiveVersioningRendezvous& operator=(
          const CollectiveVersioningRendezvous &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_VERSIONING_RENDEZVOUS; }
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
      virtual RtEvent post_gather(void);
    public:
      void perform_rendezvous(unsigned parent_req_index,
          LegionMap<LogicalRegion,RegionVersioning> &pending_versions);
    public:
      Operation *const op;
      Finalizer *const finalizer;
      const unsigned index;
    protected:
      LegionMap<LogicalRegion,RegionVersioning> pending_versions;
      unsigned parent_req_index;
    };

    /**
     * \class ConcurrentMappingRendezvous
     * This collective helps to validate the safety of the execution of
     * concurrent index space task launches to ensure that all the point
     * tasks have been mapped to different processors.
     */
    class ConcurrentMappingRendezvous : public GatherCollective {
    public:
      ConcurrentMappingRendezvous(ReplIndexTask *owner,
          CollectiveIndexLocation loc, ReplicateContext *ctx,
          ShardID target, size_t expected_points);
      virtual ~ConcurrentMappingRendezvous(void) { }
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_CONCURRENT_MAPPING_RENDEZVOUS; }
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
    public:
      void perform_rendezvous(std::map<Processor,DomainPoint> &processors,
                              ApUserEvent all_mapped_event);
    public:
      ReplIndexTask *const owner;
      const size_t expected_points;
    protected:
      std::map<Processor,DomainPoint> concurrent_processors;
      ApUserEvent all_mapped_event;
    };
    
    /**
     * \class ConcurrentAllreduce
     * This class helps with the process of performing an all-reduce between
     * the shards of a concurrent index space task launch to get their 
     * maximum lamport clock and whether any inputs were poisoned
     * Do this in order so we can count the total points we've seen and
     * send out the results as soon as possible in the case that we can
     * short-circuit the result when all points come from just a subset
     * of the shards.
     */
    class ConcurrentAllreduce : public AllGatherCollective<true> {
    public:
      ConcurrentAllreduce(CollectiveIndexLocation loc, ReplicateContext *ctx,
                          size_t expected_points);
      ConcurrentAllreduce(const ConcurrentAllreduce &rhs) = delete;
      virtual ~ConcurrentAllreduce(void);
    public:
      ConcurrentAllreduce& operator=(const ConcurrentAllreduce &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_CONCURRENT_ALLREDUCE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void exchange(std::vector<std::pair<SliceTask*,AddressSpaceID> > &slices,
        uint64_t lamport_clock, bool poisoned, RtBarrier collective_kernel_bar,
        VariantID variant, size_t points);
    protected:
      void notify_concurrent_slices(void);
    public:
      const size_t expected_points;
    protected:
      std::vector<std::pair<SliceTask*,AddressSpaceID> > concurrent_slices;
      RtBarrier collective_kernel_barrier;
      uint64_t concurrent_lamport_clock;
      VariantID concurrent_variant;
      size_t total_points;
      bool concurrent_poisoned;
    };

    /**
     * \class ProjectionTreeExchange
     * This class provides a way of exchanging the projection trees 
     * data structures between the shards in a way that is memory 
     * efficient and won't involve materializing all the data for
     * each subtree on every node.
     */
    class ProjectionTreeExchange : public AllGatherCollective<false> {
    public:
      ProjectionTreeExchange(ProjectionNode *n, ReplicateContext *ctx,
          CollectiveIndexLocation loc, bool &disjoint, 
          bool &permits_name_based_self_analysis, bool &unique_shards);
      ProjectionTreeExchange(const ProjectionTreeExchange &rhs) = delete;
      ~ProjectionTreeExchange(void);
    public:
      ProjectionTreeExchange& operator=(const ProjectionTreeExchange&) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_PROJECTION_TREE_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target, 
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
      virtual RtEvent post_complete_exchange(void);
    public:
      ProjectionNode *const node;
      bool &disjoint;
      bool &permits_name_based_self_analysis;
      bool &unique_shards;
      bool leaves_only;
    protected:
#ifdef LEGION_NAME_BASED_CHILDREN_SHARDS
      typedef ProjectionNode::ShardSet ShardSet;
#endif
      typedef ProjectionNode::RegionSummary RegionSummary; 
      typedef ProjectionNode::PartitionSummary PartitionSummary; 
      std::map<LogicalRegion,RegionSummary> region_summaries;
      std::map<LogicalPartition,PartitionSummary> partition_summaries;
    };

    /**
     * \class TimeoutMatchExchange
     * This class helps perform all all-reduce exchange between the shards
     * to see which logical users that have timed out on their analyses
     * can be collected across all the shards. To be collected all the 
     * shards must agree on what they are pruning.
     */
    class TimeoutMatchExchange : public AllGatherCollective<false> {
    public:
      TimeoutMatchExchange(ReplicateContext *ctx, CollectiveIndexLocation loc);
      TimeoutMatchExchange(const TimeoutMatchExchange &rhs) = delete;
      virtual ~TimeoutMatchExchange(void);
    public:
      TimeoutMatchExchange& operator=(const TimeoutMatchExchange &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_TIMEOUT_MATCH_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target, 
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void perform_exchange(std::vector<LogicalUser*> &timeouts, bool ready);
      bool complete_exchange(std::vector<LogicalUser*> &to_delete);
    protected:
      std::vector<LogicalUser*> timeout_users;
      // Pair represents <context index,region requirement index> for each user
      std::vector<std::pair<size_t,unsigned> > all_timeouts;
      bool double_latency;
    };

    /**
     * \class MaskExchange
     * This class will perform an all-reduce of a field mask between
     * the shards so that all participants either get the union or
     * intersection of all the fields in the field mask
     */
    template<bool UNION>
    class MaskExchange : public AllGatherCollective<false> {
    public:
      MaskExchange(FieldMask &m, ReplicateContext *ctx, CollectiveID id)
        : AllGatherCollective<false>(ctx, id), mask(m) { }
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_MASK_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage) 
        { rez.serialize(mask); }
      virtual void unpack_collective_stage(Deserializer &derez, int stage)
        { FieldMask m; derez.deserialize(m);
          if (UNION) mask |= m; else mask &= m; }
    private:
      FieldMask &mask;
    };

    /**
     * \class PredicateCollective
     * A class for performing all-reduce of the maximum observed indexes
     * for a replicated predicate impl
     */
    class PredicateCollective : 
      public AllReduceCollective<MaxReduction<uint64_t>,false> {
    public:
      PredicateCollective(ReplPredicateImpl *predicate, 
          ReplicateContext *ctx, CollectiveID id);
      PredicateCollective(const PredicateCollective &rhs) = delete;
      virtual ~PredicateCollective(void) { }
    public:
      PredicateCollective& operator=(const PredicateCollective &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_PREDICATE_EXCHANGE; }
      virtual RtEvent post_complete_exchange(void);
    public:
      ReplPredicateImpl *const predicate;
    };

    /**
     * \class CrossProductExchange
     * This all-gather exchanges IDs for the creation of replicated
     * partitions when performing a cross-product partition
     */
    class CrossProductExchange : public AllGatherCollective<false> {
    public:
      CrossProductExchange(ReplicateContext *ctx, CollectiveIndexLocation loc);
      CrossProductExchange(const CrossProductExchange &rhs) = delete;
      virtual ~CrossProductExchange(void) { }
    public:
      CrossProductExchange& operator=(const CrossProductExchange &rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_CROSS_PRODUCT_EXCHANGE; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void exchange_ids(LegionColor color,DistributedID did,IndexPartition pid);
      void sync_child_ids(LegionColor color, DistributedID &did, 
                          IndexPartition &pid);
    protected:
      std::map<LegionColor,std::pair<IndexPartition,DistributedID> > child_ids;
    };

    /**
     * \class TracingSetDeduplication
     * This class performs an all-gather on the names of equivalence sets
     * which might be replicated across the shards and therefore we want 
     * to efficiently deduplicate which of them will be captured by 
     * different nodes for establishing tracing pre/post-conditions
     */
    class TracingSetDeduplication : public AllGatherCollective<false> {
    public:
      TracingSetDeduplication(ReplicateContext *ctx, CollectiveID id);
      TracingSetDeduplication(const TracingSetDeduplication &rhs) = delete;
      virtual ~TracingSetDeduplication(void);
    public:
      TracingSetDeduplication& operator=(
          const TracingSetDeduplication &rhs) = delete;
    public:
       virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_TRACING_SET_DEDUPLICATION; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void record_set(DistributedID did, unsigned parent_req_index);
      const std::map<DistributedID,unsigned>& all_gather_collective_sets(void);
    private:
      std::map<DistributedID,unsigned> collective_sets;
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
      virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_SLOW_BARRIER; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage) { }
      virtual void unpack_collective_stage(Deserializer &derez, int stage) { }
    };

    /**
     * \class ShardRendezvous
     * A sharded rendezvous class is similar to a shard collective, but it
     * instead of performing collective operations between all the shards
     * in a control replicated context, it will support doing parallel
     * rendezvous between a subset of shards in the context. Callers must
     * provide a unique key for performing the rendezvous.
     */
    class ShardRendezvous {
    public:
      ShardRendezvous(ReplicateContext *ctx, ShardID origin,
                      const std::vector<ShardID> &participants);
      ShardRendezvous(const ShardRendezvous &rhs) = delete;
      virtual ~ShardRendezvous(void) { }
    public:
      ShardRendezvous& operator=(const ShardRendezvous &rhs) = delete;
    public:
      virtual bool receive_message(Deserializer &derez) = 0;
    public:
      void prefix_message(Serializer &rez, ShardID target) const;
      void register_rendezvous(void);
      size_t get_total_participants(void) const;
      ShardID get_parent(void) const;
      size_t count_children(void) const;
      void get_children(std::vector<ShardID> &children) const;
    protected:
      unsigned find_index(ShardID shard) const;
      ShardID get_index(unsigned index) const;
      unsigned convert_to_offset(unsigned index, unsigned origin) const;
      unsigned convert_to_index(unsigned offset, unsigned origin) const;
    public:
      ReplicateContext *const context;
      const ShardID origin_shard;
      const ShardID local_shard;
      const std::vector<ShardID> &participants;
      const bool all_shards_participating;
    protected:
      mutable LocalLock rendezvous_lock;
    };

    /**
     * \class ShardedChildrenBroadcast
     * Broadcast out a set of child nodes across the set of shards
     */
    class ShardedChildrenBroadcast : public ShardRendezvous {
    public:
      ShardedChildrenBroadcast(ReplicateContext *ctx,
          ShardID source, const std::vector<ShardID> &participants,
          std::vector<IndexPartNode*> &children);
      ShardedChildrenBroadcast(const ShardedChildrenBroadcast &rhs) = delete;
      virtual ~ShardedChildrenBroadcast(void);
    public:
      ShardedChildrenBroadcast& operator=(
          const ShardedChildrenBroadcast &rhs) = delete;
    public:
      virtual bool receive_message(Deserializer &derez);
      void broadcast(void);
      void receive(void);
    protected:
      std::vector<IndexPartNode*> &children;
      RtUserEvent received_event;
      bool received;
    };

    /**
     * \class TotalLeavesRendezvous
     * Perform an all-reduce to count the total number of leaves
     * in a region-subtree for a subsets of shards in a control
     * replicated context.
     */
    class TotalLeavesRendezvous : public ShardRendezvous {
    public:
      TotalLeavesRendezvous(ReplicateContext *ctx,
          ShardID root, const std::vector<ShardID> &participants);
      TotalLeavesRendezvous(const TotalLeavesRendezvous &rhs) = delete;
      virtual ~TotalLeavesRendezvous(void) { }
    public:
      TotalLeavesRendezvous& operator=(
          const TotalLeavesRendezvous &rhs) = delete;
    public:
      virtual bool receive_message(Deserializer &derez);
      size_t accumulate(size_t leaves);
    protected:
      size_t total_leaves;
      RtUserEvent received_event;
      int remaining;
    };

    /**
     * \class ShardedMapping
     * This class mirrors the CollectiveMapping class and provides helper
     * methods for doing collective-style tree broadcasts and reductions
     * on a unique set of shards.
     */
    class ShardedMapping : public Collectable {
    public:
      ShardedMapping(const std::set<ShardID> &shards, size_t radix);
      ShardedMapping(const std::vector<ShardID> &shards, size_t radix);
    public:
      inline ShardID operator[](unsigned idx) const
        { return unique_sorted_shards[idx]; }
      inline size_t size(void) const { return unique_sorted_shards.size(); }
      bool operator==(const ShardedMapping &rhs) const;
      bool operator!=(const ShardedMapping &rhs) const;
    public:
      ShardID get_parent(const ShardID origin, const ShardID local) const;
      void get_children(const ShardID origin, const ShardID local,
                        std::vector<ShardID> &children) const;
      size_t count_children(const ShardID origin, const ShardID local) const;
      bool contains(const ShardID space) const;
    protected:
      unsigned find_index(const ShardID space) const;
      unsigned convert_to_offset(unsigned index, unsigned origin) const;
      unsigned convert_to_index(unsigned offset, unsigned origin) const;
    protected:
      std::vector<ShardID> unique_sorted_shards;
      size_t radix;
    };

    /**
     * \class ReplCollectiveVersioning
     */
    template<typename OP>
    class ReplCollectiveVersioning : public OP,
      public CollectiveVersioningRendezvous::Finalizer{
    public:
      typedef typename OP::RegionVersioning RegionVersioning;
    public:
      ReplCollectiveVersioning(Runtime *rt);
      ReplCollectiveVersioning(
          const ReplCollectiveVersioning<OP> &rhs) = delete;
    public:
      ReplCollectiveVersioning<OP>& operator=(
          const ReplCollectiveVersioning<OP> &rhs) = delete;
    public:
      virtual void deactivate(bool free = true);
      virtual void finalize_collective_versioning_analysis(unsigned index,
          unsigned parent_req_index,
          LegionMap<LogicalRegion,RegionVersioning> &to_perform);
      virtual void finalize_collective_versioning(unsigned index,
            unsigned parent_req_index,
            LegionMap<LogicalRegion,RegionVersioning> &pending_versions);
    public:
      void create_collective_rendezvous(unsigned requirement_index);
      virtual void shard_off_collective_rendezvous(
          std::set<RtEvent> &done_events);
      virtual void elide_collective_rendezvous(void);
    protected:
      std::map<unsigned,
             CollectiveVersioningRendezvous*> collective_versioning_rendezvous;
    };

    /**
     * \class ReplCollectiveViewCreator
     * This class provides additional functionality for creating collective
     * views in control replication contexts by helping to manage the 
     * rendezvous between the shards.
     */
    template<typename OP>
    class ReplCollectiveViewCreator : public ReplCollectiveVersioning<OP>, 
      public CollectiveViewRendezvous::Finalizer {
    public:
      typedef typename OP::RendezvousKey RendezvousKey;
      typedef typename OP::CollectiveRendezvous CollectiveRendezvous;
    public:
      ReplCollectiveViewCreator(Runtime *rt);
      ReplCollectiveViewCreator(
          const ReplCollectiveViewCreator<OP> &rhs) = delete;
    public:
      ReplCollectiveViewCreator<OP>& operator=(
          const ReplCollectiveViewCreator<OP> &rhs) = delete;
    public:
      virtual void deactivate(bool free = true); 
      virtual void construct_collective_mapping(const RendezvousKey &key,
                      std::map<LogicalRegion,CollectiveRendezvous> &rendezvous);
      virtual void finalize_collective_mapping(const RendezvousKey &key,
                      std::map<LogicalRegion,CollectiveRendezvous> &rendezvous);
      void create_collective_rendezvous(RegionTreeID tid,
          unsigned requirement_index, unsigned analysis_index = 0);
      virtual void shard_off_collective_rendezvous(
          std::set<RtEvent> &done_events);
      virtual void elide_collective_rendezvous(void);
    protected: 
      std::map<RendezvousKey,
               CollectiveViewRendezvous*> collective_view_rendezvous;
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
      virtual void deactivate(bool free = true);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual void predicate_false(void);
      virtual void shard_off(RtEvent mapped_precondition);
      virtual void prepare_map_must_epoch(void);
    public:
      // Override this so it can broadcast the future result
      virtual Future create_future(void);
    public:
      // Override for saying when it is safe to use output region trees
      virtual void record_output_registered(RtEvent registered,
                                      std::set<RtEvent> &applied_events);
    public:
      void initialize_replication(ReplicateContext *ctx);
      void set_sharding_function(ShardingID functor,ShardingFunction *function);
    protected:
      ShardID owner_shard;
      IndexSpaceNode *launch_space;
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
      RtBarrier output_bar;
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
    class ReplIndexTask : public ReplCollectiveViewCreator<IndexTask> {
    public:
      ReplIndexTask(Runtime *rt);
      ReplIndexTask(const ReplIndexTask &rhs) = delete;
      virtual ~ReplIndexTask(void);
    public:
      ReplIndexTask& operator=(const ReplIndexTask &rhs) = delete;
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
    protected:
      virtual void create_future_instances(std::vector<Memory> &target_mems);
      virtual void finish_index_task_reduction(void);
    public:
      // Have to override this too for doing output in the
      // case that we misspeculate
      virtual void predicate_false(void);
      virtual void prepare_map_must_epoch(void);
    public:
      void initialize_replication(ReplicateContext *ctx);
      void set_sharding_function(ShardingID functor,ShardingFunction *function);
      virtual FutureMap create_future_map(TaskContext *ctx,
                    IndexSpace launch_space, IndexSpace shard_space);
      virtual ApEvent rendezvous_concurrent_mapped(
          const DomainPoint &point, Processor target);
      virtual ApEvent rendezvous_concurrent_mapped(
          std::vector<std::pair<Processor,DomainPoint> > &targets);
      virtual void concurrent_allreduce(SliceTask *slice,
          AddressSpaceID slice_space, size_t points, uint64_t lamport_clock,
          VariantID vid, bool poisoned);
      void select_sharding_function(ReplicateContext *repl_ctx);
    public:
      // Methods for supporting intra-index-space mapping dependences
      virtual RtEvent find_intra_space_dependence(const DomainPoint &point);
      virtual void record_intra_space_dependence(const DomainPoint &point,
                                                 const DomainPoint &next,
                                                 RtEvent point_mapped);
    public:
      // Output regions
      virtual void record_output_registered(RtEvent registered);
      virtual void finalize_output_regions(bool first_invocation);
    public:
      virtual size_t get_collective_points(void) const;
      virtual bool find_shard_participants(std::vector<ShardID> &shards);
    protected:
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
      BufferExchange *serdez_redop_collective;
      FutureAllReduceCollective *all_reduce_collective;
      FutureReductionCollective *reduction_collective;
      FutureBroadcastCollective *broadcast_collective;
      OutputExtentExchange *output_size_collective;
      CollectiveID collective_check_id;
      RtBarrier output_bar;
    protected:
      // Map of output sizes collected by this shard
      std::vector<OutputExtentMap> local_output_extents;
    protected:
      std::set<std::pair<DomainPoint,ShardID> > unique_intra_space_deps;
    protected:
      // For setting up concurrent execution
      ConcurrentMappingRendezvous *concurrent_mapping_rendezvous;
      ConcurrentAllreduce *concurrent_exchange;
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
      virtual void deactivate(bool free = true);
    public:
      void set_repl_close_info(RtBarrier mapped_barrier);
      virtual void trigger_ready(void);
    protected:
      RtBarrier mapped_barrier;
    };

    /**
     * \class ReplRefinementOp
     * A refinement operation that is aware that it is being
     * executed in a control replication context.
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
      virtual void deactivate(bool free = true);
    public:
      void set_repl_refinement_info(RtBarrier mapped_barrier, 
                                    RtBarrier refinement_barrier);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void); 
    protected:
      RtBarrier mapped_barrier;
      RtBarrier refinement_barrier;
    };

    /**
     * \class ReplResetOp
     * A reset operation that is aware it is being executed
     * in a control replicated context
     */
    class ReplResetOp : public ResetOp {
    public:
      ReplResetOp(Runtime *runtime);
      ReplResetOp(const ReplResetOp &rhs) = delete;
      virtual ~ReplResetOp(void);
    public:
      ReplResetOp& operator=(const ReplResetOp &rhs) = delete;
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
    protected:
      RtBarrier reset_barrier;
    };

    /**
     * \class ReplFillOp
     * A copy operation that is aware that it is being
     * executed in a control replication context.
     */
    class ReplFillOp : 
      public ReplCollectiveVersioning<CollectiveVersioning<FillOp> > {
    public:
      ReplFillOp(Runtime *rt);
      ReplFillOp(const ReplFillOp &rhs) = delete;
      virtual ~ReplFillOp(void);
    public:
      ReplFillOp& operator=(const ReplFillOp &rhs) = delete;
    public:
      void initialize_replication(ReplicateContext *ctx,
                                  DistributedID fresh_did,
                                  bool is_first_local);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual bool is_collective_first_local_shard(void) const
        { return is_first_local_shard; }
      virtual RtEvent finalize_complete_mapping(RtEvent event);
      virtual bool perform_collective_analysis(CollectiveMapping *&mapping,
                                               bool &first_local);
      virtual RtEvent perform_collective_versioning_analysis(unsigned index,
                       LogicalRegion handle, EqSetTracker *tracker,
                       const FieldMask &mask, unsigned parent_req_index);
      virtual RtEvent initialize_fill_view(void);
      virtual void predicate_false(void);
    public:
      RtBarrier collective_map_barrier;
      CreateCollectiveFillView *collective;
      CollectiveID collective_id;
      DistributedID fresh_did;
      bool is_first_local_shard;
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
      virtual void deactivate(bool free = true);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual RtEvent initialize_fill_view(void);
      virtual IndexSpaceNode* get_shard_points(void) const 
        { return shard_points; }
      virtual bool find_shard_participants(std::vector<ShardID> &shards);
    public:
      void initialize_replication(ReplicateContext *ctx,
                                  DistributedID fresh_did);
    protected:
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
      IndexSpaceNode *shard_points;
      MapperManager *mapper;
      CreateCollectiveFillView *collective;
      CollectiveID collective_id;
      DistributedID fresh_did;
#ifdef DEBUG_LEGION
    public:
      inline void set_sharding_collective(ShardingGatherCollective *collective)
        { sharding_collective = collective; }
    protected:
      ShardingGatherCollective *sharding_collective;
#endif
    };

    /**
     * \class ReplDiscardOp
     * A discard operation that is aware that it is being
     * exected in a control replication context.
     */
    class ReplDiscardOp : 
      public ReplCollectiveVersioning<CollectiveVersioning<DiscardOp> > {
    public:
      ReplDiscardOp(Runtime *rt);
      ReplDiscardOp(const ReplDiscardOp &rhs) = delete;
      virtual ~ReplDiscardOp(void);
    public:
      ReplDiscardOp& operator=(const ReplDiscardOp &rhs) = delete;
    public:
      void initialize_replication(ReplicateContext *ctx, bool is_first_local);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual RtEvent finalize_complete_mapping(RtEvent event);
      virtual bool perform_collective_analysis(CollectiveMapping *&mapping,
                                               bool &first_local);
      virtual RtEvent perform_collective_versioning_analysis(unsigned index,
                       LogicalRegion handle, EqSetTracker *tracker,
                       const FieldMask &mask, unsigned parent_req_index);
    protected:
      RtBarrier collective_map_barrier;
      bool is_first_local_shard;
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
      virtual void deactivate(bool free = true);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
    protected:
      IndexSpaceNode *launch_space;
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
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
      virtual void deactivate(bool free = true);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual IndexSpaceNode* get_shard_points(void) const 
        { return shard_points; }
      virtual bool find_shard_participants(std::vector<ShardID> &shards);
    protected:
      virtual RtEvent exchange_indirect_records(
          const unsigned index, const ApEvent local_pre, 
          const ApEvent local_post, ApEvent &collective_pre,
          ApEvent &collective_post, const TraceInfo &trace_info,
          const InstanceSet &instances, const RegionRequirement &req,
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
      IndexSpaceNode *shard_points;
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
    class ReplDeletionOp : 
      public ReplCollectiveVersioning<CollectiveVersioning<DeletionOp> > {
    public:
      ReplDeletionOp(Runtime *rt);
      ReplDeletionOp(const ReplDeletionOp &rhs) = delete;
      virtual ~ReplDeletionOp(void);
    public:
      ReplDeletionOp& operator=(const ReplDeletionOp &rhs) = delete;
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void trigger_commit(void);
    public:
      void initialize_replication(ReplicateContext *ctx, bool is_first,
                                  RtBarrier *ready_barrier = NULL,
                                  RtBarrier *mapping_barrier = NULL,
                                  RtBarrier *commit_barrier = NULL);
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
      RtBarrier commit_barrier;
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
      virtual void deactivate(bool free = true);
    public:
      virtual void populate_sources(const FutureMap &fm,
          IndexPartition pid, bool needs_all_futures);
      virtual void trigger_execution(void);
    };

    /**
     * \class ReplDependentPartitionOp
     * A dependent partitioning operation that knows that it
     * is being executed in a control replication context
     */
    class ReplDependentPartitionOp : public ReplCollectiveViewCreator<
                            CollectiveViewCreator<DependentPartitionOp> > {
    public:
      ReplDependentPartitionOp(Runtime *rt);
      ReplDependentPartitionOp(const ReplDependentPartitionOp &rhs) = delete;
      virtual ~ReplDependentPartitionOp(void);
    public:
      ReplDependentPartitionOp& operator=(
                               const ReplDependentPartitionOp &rhs) = delete;
    public:
      void initialize_replication(ReplicateContext *context);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      // Need to pick our sharding functor
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);  
      virtual void finalize_mapping(void);
      virtual ApEvent trigger_thunk(IndexSpace handle, ApEvent insts_ready,
                                    const InstanceSet &mapped_instances,
                                    const PhysicalTraceInfo &info,
                                    const DomainPoint &color);
      virtual void trigger_execution(void);
      virtual void select_partition_projection(void);
      virtual IndexSpaceNode* get_shard_points(void) const 
        { return shard_points; }
      virtual bool find_shard_participants(std::vector<ShardID> &shards);
      virtual bool perform_collective_analysis(CollectiveMapping *&mapping,
                                               bool &first_local);
      virtual RtEvent perform_collective_versioning_analysis(unsigned index,
                       LogicalRegion handle, EqSetTracker *tracker,
                       const FieldMask &mask, unsigned parent_req_index);
    protected:
      void select_sharding_function(void);
      void find_remote_targets(std::vector<ApEvent> &preconditions);
    protected:
      ShardingFunction *sharding_function;
      IndexSpaceNode *shard_points;
      RtBarrier mapping_barrier;
      FieldDescriptorGather *gather;
      DeppartResultScatter *scatter;
      FieldDescriptorExchange *exchange;
      ApBarrier collective_ready;
      ApBarrier collective_done;
      std::map<DomainPoint,Domain> remote_targets;
      std::vector<DeppartResult> deppart_results;
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
      struct DeferMustEpochReturnResourcesArgs : 
        public LgTaskArgs<DeferMustEpochReturnResourcesArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_MUST_EPOCH_RETURN_TASK_ID;
      public:
        DeferMustEpochReturnResourcesArgs(ReplMustEpochOp *o)
          : LgTaskArgs<DeferMustEpochReturnResourcesArgs>(
              o->get_unique_op_id()), op(o), 
            done(Runtime::create_rt_user_event()) { }
      public:
        ReplMustEpochOp *const op;
        const RtUserEvent done;
      };
    public:
      ReplMustEpochOp(Runtime *rt);
      ReplMustEpochOp(const ReplMustEpochOp &rhs);
      virtual ~ReplMustEpochOp(void);
    public:
      ReplMustEpochOp& operator=(const ReplMustEpochOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual FutureMap create_future_map(TaskContext *ctx,
                      IndexSpace domain, IndexSpace shard_space);
      virtual void instantiate_tasks(InnerContext *ctx,
                                     const MustEpochLauncher &launcher);
      virtual MapperManager* invoke_mapper(void);
      virtual void map_and_distribute(std::set<RtEvent> &tasks_mapped,
                                      std::set<ApEvent> &tasks_complete);
      virtual bool has_prepipeline_stage(void) const { return true; }
      virtual void trigger_prepipeline_stage(void);
      virtual void receive_resources(uint64_t return_index,
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
      void map_replicate_tasks(void);
      void distribute_replicate_tasks(void);
    public:
      void initialize_replication(ReplicateContext *ctx);
      Domain get_shard_domain(void) const;
      size_t count_shard_local_points(IndexSpaceNode *launch_domain);
    public:
      bool has_return_resources(void) const;
      static void handle_defer_return_resources(const void *args);
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
#ifdef DEBUG_LEGION
    public:
      inline void set_sharding_collective(ShardingGatherCollective *collective)
        { sharding_collective = collective; }
    protected:
      ShardingGatherCollective *sharding_collective;
#endif
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
      virtual void deactivate(bool free = true);
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
      ReplAllReduceOp(const ReplAllReduceOp &rhs) = delete;
      virtual ~ReplAllReduceOp(void);
    public:
      ReplAllReduceOp& operator=(const ReplAllReduceOp &rhs) = delete;
    public:
      void initialize_replication(ReplicateContext *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    protected:
      virtual void populate_sources(void);
      virtual void create_future_instances(void);
      virtual void all_reduce_serdez(void);
      virtual ApEvent all_reduce_redop(RtEvent &executed);
    protected:
      BufferExchange *serdez_redop_collective;
      FutureAllReduceCollective *all_reduce_collective;
      FutureReductionCollective *reduction_collective;
      FutureBroadcastCollective *broadcast_collective;
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
      virtual void deactivate(bool free = true);
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
     * \class ReplTimingOp
     * A timing operation that is aware that it is 
     * being executed in a control replication context
     */
    class ReplTimingOp : public ReplFenceOp {
    public:
      ReplTimingOp(Runtime *rt);
      ReplTimingOp(const ReplTimingOp &rhs) = delete;
      virtual ~ReplTimingOp(void);
    public:
      ReplTimingOp& operator=(const ReplTimingOp &rhs) = delete;
    public:
      Future initialize(InnerContext *ctx, const TimingLauncher &launcher,
                        Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual bool invalidates_physical_trace_template(bool &exec_fence) const
        { return false; }
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void trigger_complete(ApEvent complete); 
      virtual void trigger_commit(void);
      virtual void perform_measurement(void);
    public:
      inline void set_timing_collective(ValueBroadcast<long long> *collective) 
        { timing_collective = collective; }
    protected:
      TimingMeasurement measurement;
      std::set<Future> preconditions;
      RtEvent measured;
      ValueBroadcast<long long> *timing_collective;
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
    class ReplMapOp : 
      public ReplCollectiveViewCreator<CollectiveViewCreator<MapOp> > {
    public:
      ReplMapOp(Runtime *rt);
      ReplMapOp(const ReplMapOp &rhs) = delete;
      virtual ~ReplMapOp(void);
    public:
      ReplMapOp& operator=(const ReplMapOp &rhs) = delete;
    public:
      void initialize_replication(ReplicateContext *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual bool invoke_mapper(InstanceSet &mapped_instances,
              std::vector<PhysicalManager*> &source_instances);
      virtual bool supports_collective_instances(void) const { return true; }
      virtual RtEvent finalize_complete_mapping(RtEvent precondition);
      virtual bool perform_collective_analysis(CollectiveMapping *&mapping,
                                               bool &first_local);
      virtual RtEvent perform_collective_versioning_analysis(unsigned index,
                       LogicalRegion handle, EqSetTracker *tracker,
                       const FieldMask &mask, unsigned parent_req_index);
      virtual bool find_shard_participants(std::vector<ShardID> &shards);
    protected:
      CollectiveID mapping_check, sources_check;
      RtBarrier collective_map_barrier; 
    };

    /**
     * \class ReplAttachOp
     * An attach operation that is aware that it is being
     * executed in a control replicated context.
     */
    class ReplAttachOp : 
      public ReplCollectiveViewCreator<CollectiveViewCreator<AttachOp> > {
    public:
      ReplAttachOp(Runtime *rt);
      ReplAttachOp(const ReplAttachOp &rhs) = delete;
      virtual ~ReplAttachOp(void);
    public:
      ReplAttachOp& operator=(const ReplAttachOp &rhs) = delete;
    public:
      void initialize_replication(ReplicateContext *ctx,
                                  bool collective_instances,
                                  bool deduplicate_across_shards,
                                  bool first_local_shard);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
    public:
      virtual PhysicalManager* create_manager(RegionNode *node,
                                   const std::vector<FieldID> &field_set,
                                   const std::vector<size_t> &field_sizes,
                                   const std::vector<unsigned> &mask_index_map,
                                   const std::vector<CustomSerdezID> &serez,
                                              const FieldMask &external_mask);
      virtual RtEvent finalize_complete_mapping(RtEvent event);
      virtual bool perform_collective_analysis(CollectiveMapping *&mapping,
                                               bool &first_local);
      virtual RtEvent perform_collective_versioning_analysis(unsigned index,
                       LogicalRegion handle, EqSetTracker *tracker,
                       const FieldMask &mask, unsigned parent_req_index);
      virtual bool find_shard_participants(std::vector<ShardID> &shards);
    protected:
      RtBarrier collective_map_barrier;
      size_t exchange_index;
      bool collective_instances;
      bool deduplicate_across_shards;
      bool is_first_local_shard;
      // individual insts: whether at least one shard lives on the local process
      bool contains_individual; 
    protected:
      RtBarrier resource_barrier;
      ValueBroadcast<DistributedID> *did_broadcast;
      // Need this because std::pair<PhysicalInstance,ApEvent> is not
      // trivially copyable for reasons passing understanding
      struct InstanceEvents {
        PhysicalInstance instance;
        ApEvent ready_event;
        LgEvent unique_event;
      };
      ValueBroadcast<InstanceEvents> *single_broadcast;
    };

    /**
     * \class ReplIndexAttachOp
     * An index space attach operation that is aware
     * that it is executing in a control replicated context
     */
    class ReplIndexAttachOp : public ReplCollectiveViewCreator<IndexAttachOp> {
    public:
      ReplIndexAttachOp(Runtime *rt);
      ReplIndexAttachOp(const ReplIndexAttachOp &rhs) = delete;
      virtual ~ReplIndexAttachOp(void);
    public:
      ReplIndexAttachOp& operator=(const ReplIndexAttachOp &rhs) = delete;
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void check_point_requirements(
                    const std::vector<IndexSpace> &spaces);
      virtual bool are_all_direct_children(bool local);
      virtual bool find_shard_participants(std::vector<ShardID> &shards);
    public:
      void initialize_replication(ReplicateContext *ctx);
    protected:
      IndexAttachExchange *collective;
      ShardParticipantsExchange *participants;
      ShardingFunction *sharding_function;
    };

    /**
     * \class ReplDetachOp
     * An detach operation that is aware that it is being
     * executed in a control replicated context.
     */
    class ReplDetachOp : 
      public ReplCollectiveViewCreator<CollectiveViewCreator<DetachOp> > {
    public:
      ReplDetachOp(Runtime *rt);
      ReplDetachOp(const ReplDetachOp &rhs) = delete;
      virtual ~ReplDetachOp(void);
    public:
      ReplDetachOp& operator=(const ReplDetachOp &rhs) = delete;
    public:
      void initialize_replication(ReplicateContext *ctx,
                                  bool collective_instances,
                                  bool first_local_shard);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual RtEvent finalize_complete_mapping(RtEvent event);
      virtual void detach_external_instance(PhysicalManager *manager);
      virtual bool perform_collective_analysis(CollectiveMapping *&mapping,
                                               bool &first_local);
      virtual RtEvent perform_collective_versioning_analysis(unsigned index,
                       LogicalRegion handle, EqSetTracker *tracker,
                       const FieldMask &mask, unsigned parent_req_index);
      virtual bool find_shard_participants(std::vector<ShardID> &shards);
    public:
      // Help for unordered detachments
      void record_unordered_kind(
        std::map<std::pair<LogicalRegion,FieldID>,Operation*> &detachments);
    protected:
      RtBarrier collective_map_barrier;
      ApBarrier effects_barrier;
      size_t exchange_index;
      bool collective_instances;
      bool is_first_local_shard;
    };

    /**
     * \class ReplIndexDetachOp
     * An index space detach operation that is aware
     * that it is executing in a control replicated context
     */
    class ReplIndexDetachOp : public ReplCollectiveViewCreator<IndexDetachOp> {
    public:
      ReplIndexDetachOp(Runtime *rt);
      ReplIndexDetachOp(const ReplIndexDetachOp &rhs) = delete;
      virtual ~ReplIndexDetachOp(void);
    public:
      ReplIndexDetachOp& operator=(const ReplIndexDetachOp &rhs) = delete;
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual bool find_shard_participants(std::vector<ShardID> &shards);
    public:
      void initialize_replication(ReplicateContext *ctx);
      void record_unordered_kind(std::map<
          std::pair<LogicalRegion,FieldID>,Operation*> &region_detachments,
          std::map<std::pair<
            LogicalPartition,FieldID>,Operation*> &partition_detachments);
    protected:
      ShardingFunction *sharding_function;
      ShardParticipantsExchange *participants;
      ApBarrier effects_barrier;
    };

    /**
     * \class ReplAcquireOp
     * An acquire op that is aware that it is
     * executing in a control replicated context
     */
    class ReplAcquireOp : 
      public ReplCollectiveViewCreator<CollectiveViewCreator<AcquireOp> > {
    public:
      ReplAcquireOp(Runtime *rt);
      ReplAcquireOp(const ReplAcquireOp &rhs) = delete;
      virtual ~ReplAcquireOp(void);
    public:
      ReplAcquireOp& operator=(const ReplAcquireOp &rhs) = delete;
    public:
      void initialize_replication(ReplicateContext *context,
                                  bool first_local_shard);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual void predicate_false(void);
      virtual RtEvent finalize_complete_mapping(RtEvent precondition);
      virtual bool perform_collective_analysis(CollectiveMapping *&mapping,
                                               bool &first_local);
      virtual RtEvent perform_collective_versioning_analysis(unsigned index,
                       LogicalRegion handle, EqSetTracker *tracker,
                       const FieldMask &mask, unsigned parent_req_index);
    protected:
      RtBarrier collective_map_barrier;
      bool is_first_local_shard;
    };

    /**
     * \class ReplReleaseOp
     * A release op that is aware that it 
     */
    class ReplReleaseOp : 
      public ReplCollectiveViewCreator<CollectiveViewCreator<ReleaseOp> > {
    public:
      ReplReleaseOp(Runtime *rt);
      ReplReleaseOp(const ReplReleaseOp &rhs) = delete;
      virtual ~ReplReleaseOp(void);
    public:
      ReplReleaseOp& operator=(const ReplReleaseOp &rhs) = delete;
    public:
      void initialize_replication(ReplicateContext *context,
                                  bool first_local_shard);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual void predicate_false(void);
      virtual RtEvent finalize_complete_mapping(RtEvent event);
      virtual void invoke_mapper(std::vector<PhysicalManager*> &src_instances); 
      virtual bool perform_collective_analysis(CollectiveMapping *&mapping,
                                               bool &first_local);
      virtual RtEvent perform_collective_versioning_analysis(unsigned index,
                       LogicalRegion handle, EqSetTracker *tracker,
                       const FieldMask &mask, unsigned parent_req_index);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    protected:
      CollectiveID sources_check;
      RtBarrier collective_map_barrier;
      bool is_first_local_shard;
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
      virtual bool is_tracing_fence(void) const override { return true; }
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                     std::set<RtEvent> &applied) const override;
    protected:
      struct TemplateStatus {
        bool all_valid;
        bool any_not_acquired;
      };
      struct StatusReduction {
        typedef TemplateStatus RHS;
        template<bool EXCLUSIVE>
        static inline void fold(RHS &rhs1, RHS rhs2) 
        {
          if (!rhs2.all_valid)
            rhs1.all_valid = false;
          if (rhs2.any_not_acquired)
            rhs1.any_not_acquired = true;
        }
      };
    };
    
#if 0
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
      virtual void deactivate(bool free = true);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void sync_for_replayable_check(void);
      virtual bool exchange_replayable(ReplicateContext *ctx, bool replayable);
      virtual void sync_for_idempotent_check(void);
      virtual bool exchange_idempotent(ReplicateContext *ctx, bool idempotent);
      virtual void sync_compute_frontiers(RtEvent precondition);
    protected:
      PhysicalTemplate *current_template;
      RtBarrier recording_fence;
      CollectiveID replayable_collective_id;
      CollectiveID replay_sync_collective_id;
      CollectiveID idempotent_collective_id;
      CollectiveID idempotent_sync_collective_id;
      CollectiveID sync_compute_frontiers_collective_id;
      bool has_blocking_call;
      bool remove_trace_reference;
      bool is_recording;
    };
#endif

    class ReplCompleteOp : public ReplTraceOp, public CompleteOp {
    public:
      ReplCompleteOp(Runtime *rt) : ReplTraceOp(rt) { }
      virtual ~ReplCompleteOp(void) { }
    };

    // Mixin class for adding support for replicated complete interfaces
    template<typename OP>
    class ReplTraceComplete : public OP {
    public:
      ReplTraceComplete(Runtime *rt);
      virtual ~ReplTraceComplete(void) { }
    protected:
      void initialize_complete(ReplicateContext *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      virtual FenceOp* get_complete_operation(void) { return this; }
      virtual void begin_replayable_exchange(ReplayableStatus status);
      virtual void end_replayable_exchange(ReplayableStatus &status);
      virtual void begin_idempotent_exchange(IdempotencyStatus idempotent);
      virtual void end_idempotent_exchange(IdempotencyStatus &idempotent);
      virtual void sync_compute_frontiers(RtEvent event);
      virtual void deduplicate_condition_sets(
          std::map<EquivalenceSet*,unsigned> &condition_sets);
    private:
      CollectiveID replayable_collective_id;
      CollectiveID idempotent_collective_id;
      CollectiveID sync_compute_frontiers_collective_id;
      CollectiveID deduplication_collective_id;
      AllReduceCollective<ProdReduction<bool>,false > *replayable_collective;
      AllReduceCollective<ProdReduction<bool>,false > *idempotent_collective;
    };

    /**
     * \class ReplTraceCompleteOp
     * Control replicated version of TraceCompleteOp
     */
    class ReplTraceCompleteOp : public ReplTraceComplete<ReplCompleteOp> {
    public:
      static const AllocationType alloc_type = TRACE_COMPLETE_OP_ALLOC;
    public:
      ReplTraceCompleteOp(Runtime *rt);
      ReplTraceCompleteOp(const ReplTraceCompleteOp &rhs) = delete;
      virtual ~ReplTraceCompleteOp(void);
    public:
      ReplTraceCompleteOp& operator=(const ReplTraceCompleteOp &rhs) = delete;
    public:
      void initialize_complete(ReplicateContext *ctx, LogicalTrace *trace,
                               Provenance *provenance, bool remove_reference);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
    protected:
      bool has_blocking_call;
      bool remove_trace_reference;
    };

#if 0
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
      void initialize_replay(ReplicateContext *ctx, LogicalTrace *trace,
                             Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
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
#endif

    class ReplBeginOp : public ReplTraceOp, public BeginOp {
    public:
      ReplBeginOp(Runtime *rt) : ReplTraceOp(rt) { }
      virtual ~ReplBeginOp(void) { }
    };

    // Mixin class for adding support for replicated complete interfaces
    template<typename OP>
    class ReplTraceBegin : public OP {
    public:
      ReplTraceBegin(Runtime *rt);
      virtual ~ReplTraceBegin(void) { }
    protected:
      void initialize_begin(ReplicateContext *ctx, LogicalTrace *trace);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      virtual ApEvent get_begin_completion(void) 
        { return this->get_completion_event(); }
      virtual FenceOp* get_begin_operation(void) { return this; }
      virtual PhysicalTemplate* create_fresh_template(PhysicalTrace *trace);
      virtual bool allreduce_template_status(bool &valid, bool acquired);
    private:
      // If we do back-to-back executions of different traces
      // then we fuse the invalidation of the previous trace into the
      // begin operation of the next trace
      std::vector<CollectiveID> status_collective_ids;
    };

    /**
     * \class ReplTraceBeginOp
     * Control replicated version of trace begin op
     */
    class ReplTraceBeginOp : public ReplTraceBegin<ReplBeginOp> {
    public:
      static const AllocationType alloc_type = TRACE_BEGIN_OP_ALLOC;
    public:
      ReplTraceBeginOp(Runtime *rt);
      ReplTraceBeginOp(const ReplTraceBeginOp &rhs) = delete;
      virtual ~ReplTraceBeginOp(void);
    public:
      ReplTraceBeginOp& operator=(const ReplTraceBeginOp &rhs) = delete;
    public:
      void initialize_begin(ReplicateContext *ctx, LogicalTrace *trace,
                            Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
    };

    class ReplRecurrentOp : public ReplTraceOp, public RecurrentOp {
    public:
      ReplRecurrentOp(Runtime *rt) : ReplTraceOp(rt) { }
      virtual ~ReplRecurrentOp(void) { }
    };

    /**
     * \class ReplTraceRecurrentOp
     * Control replicated version of TraceRecurrentOp
     */
    class ReplTraceRecurrentOp : 
      public ReplTraceBegin<ReplTraceComplete<ReplRecurrentOp> > {
    public:
      static const AllocationType alloc_type = TRACE_RECURRENT_OP_ALLOC;
    public:
      ReplTraceRecurrentOp(Runtime *rt);
      ReplTraceRecurrentOp(const ReplTraceRecurrentOp &rhs) = delete;
      virtual ~ReplTraceRecurrentOp(void);
    public:
      ReplTraceRecurrentOp& operator=(
          const ReplTraceRecurrentOp &rhs) = delete;
    public:
      void initialize_recurrent(ReplicateContext *ctx, LogicalTrace *trace,
         LogicalTrace *previous, Provenance *provenance, bool remove_reference);
      void perform_logging(void);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
    protected:
      void perform_template_creation_barrier(void);
    protected:
      LogicalTrace *previous;
      SlowBarrier *slow_barrier;
      CollectiveID slow_barrier_id;
      bool has_blocking_call;
      bool has_intermediate_fence;
      bool remove_trace_reference;
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
      inline bool empty(void) const { return address_spaces.empty(); }
      inline size_t size(void) const { return address_spaces.size(); }
      inline void resize(size_t size) { address_spaces.resize(size); }
    public:
      void pack_mapping(Serializer &rez) const;
      void unpack_mapping(Deserializer &derez);
      static void pack_empty(Serializer &rez);
    protected:
      std::vector<AddressSpaceID> address_spaces;
    }; 

    /**
     * \class ShardManager
     * This is a class that manages the execution of one or
     * more shards for a given control replication context on
     * a single node. It provides support for doing broadcasts,
     * reductions, and exchanges of information between the 
     * variaous shard tasks.
     */
    class ShardManager : public CollectiveViewCreator<CollectiveHelperOp>,
                         public Mapper::SelectShardingFunctorInput {
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
      ShardManager(Runtime *rt, DistributedID did,
                   CollectiveMapping *mapping, unsigned local_count,
                   bool top, bool isomorphic_points, bool cr,
                   const Domain &shard_domain,
                   std::vector<DomainPoint> &&shard_points,
                   std::vector<DomainPoint> &&sorted_points,
                   std::vector<ShardID> &&shard_lookup,
                   SingleTask *original = NULL,
                   RtBarrier shard_task_bar = RtBarrier::NO_RT_BARRIER,
                   RtBarrier callback_bar = RtBarrier::NO_RT_BARRIER);
      ShardManager(const ShardManager &rhs) = delete;
      ~ShardManager(void);
    public:
      ShardManager& operator=(const ShardManager &rhs) = delete;
    public:
      void notify_local(void);
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
      inline ReplicateContext* find_local_context(void) const
        { return local_shards[0]->get_replicate_context(); }
      inline size_t count_local_shards(void) const 
        { return local_shards.size(); }
      inline unsigned find_local_index(ShardTask *task) const
        {
          for (unsigned idx = 0; idx < local_shards.size(); idx++)
            if (local_shards[idx] == task)
              return idx;
          assert(false);
          return 0;
        }
      inline ContextID get_first_shard_tree_context(void) const
        { return local_shards.front()->
          get_replicate_context()->get_logical_tree_context(); }
    public: // From CollectiveHelperOp
      virtual InnerContext* get_context(void);
      virtual InnerContext* find_physical_context(unsigned index);
      virtual size_t get_collective_points(void) const;
    public:
      void distribute_explicit(SingleTask *task, VariantID chosen_variant,
                               std::vector<Processor> &target_processors,
                               std::vector<VariantID> &leaf_variants);
      void distribute_implicit(TaskID top_task_id, MapperID mapper_id,
                               Processor::Kind kind, unsigned shards_per_space,
                               TopLevelContext *top_context);
      void pack_shard_manager(Serializer &rez) const;
      void set_shard_mapping(std::vector<Processor> &shard_mapping);
      ShardTask* create_shard(ShardID id, Processor target,
          VariantID variant, InnerContext *parent_ctx, SingleTask *source);
      ShardTask* create_shard(ShardID id, Processor target,
          VariantID variant, InnerContext *parent_ctx, Deserializer &derez);
    public:
      virtual void finalize_collective_versioning_analysis(unsigned index,
          unsigned parent_req_index,
          LegionMap<LogicalRegion,RegionVersioning> &to_perform);
      virtual void construct_collective_mapping(const RendezvousKey &key,
                      std::map<LogicalRegion,CollectiveRendezvous> &rendezvous);
      void finalize_replicate_collective_versioning(unsigned index,
          unsigned parent_req_index, LegionMap<LogicalRegion,
            CollectiveVersioningBase::RegionVersioning> &to_perform);
      void finalize_replicate_collective_views(
          const CollectiveViewCreatorBase::RendezvousKey &key,
          std::map<LogicalRegion,
            CollectiveViewCreatorBase::CollectiveRendezvous> &rendezvous);
      void rendezvous_check_virtual_mappings(ShardID shard,
          MapperManager *mapper, const std::vector<bool> &virtual_mappings);
    public:
      EquivalenceSet* get_initial_equivalence_set(unsigned idx,
              LogicalRegion region, InnerContext *context, bool first_shard);
      // If the creating shards are NULL we'll assume that they are all
      // participating in the creation of the index space
      EquivalenceSet* deduplicate_equivalence_set_creation(RegionNode *node,
          size_t op_ctx_index, unsigned refinement_number, 
          InnerContext *context, bool first_shard,
          const std::vector<ShardID> *creating_shards = NULL);
      FillView* deduplicate_fill_view_creation(DistributedID did, FillOp *op,
                                               bool &set_view);
      void deduplicate_attaches(const IndexAttachLauncher &launcher,
                                std::vector<unsigned> &indexes);
      Future deduplicate_future_creation(ReplicateContext *ctx,
          DistributedID did, Operation *op, const DomainPoint &point);
      FutureMap deduplicate_future_map_creation(ReplicateContext *ctx,
          Operation *op, IndexSpaceNode *domain, IndexSpaceNode *shard_domain,
          DistributedID did, Provenance *provenance); 
      FutureMap deduplicate_future_map_creation(ReplicateContext *ctx,
          IndexSpaceNode *domain, IndexSpaceNode *shard_domain,
          DistributedID did, Provenance *provenance);
      // Return true if we have a shard on every address space
      bool is_total_sharding(void);
      template<typename T>
      inline void exchange_shard_local_op_data(uint64_t context_index,
                                               size_t exchange_index,
                                               const T &data)
      {
        static_assert(std::is_trivially_copyable<T>());
        exchange_shard_local_op_data(context_index, exchange_index,
                                     &data, sizeof(data));
      }
      void exchange_shard_local_op_data(uint64_t context_index,
                                        size_t exchange_index,
                                        const void *data, size_t size);
      template<typename T>
      inline T find_shard_local_op_data(uint64_t context_index,
                                        size_t exchange_index)
      {
        static_assert(std::is_trivially_copyable<T>());
        T result;
        find_shard_local_op_data(context_index, exchange_index, 
                                 &result, sizeof(result));
        return result;
      }
      void find_shard_local_op_data(uint64_t context_index,
                                    size_t exchange_index,
                                    void *data, size_t size);
      void barrier_shard_local(uint64_t context_index, size_t exchange_index);
    public:
      void handle_post_mapped(bool local, RtEvent precondition);
      bool handle_future(ApEvent effects, FutureInstance *instance,
                         const void *metadata, size_t metasize);
      void trigger_task_complete(bool local, ApEvent effects_done);
      void trigger_task_commit(bool local, RtEvent precondition);
    public:
      void send_collective_message(MessageKind message, ShardID target, 
                                   Serializer &rez);
      void handle_collective_message(Deserializer &derez);
    public:
      void send_rendezvous_message(ShardID target, Serializer &rez);
      void handle_rendezvous_message(Deserializer &derez);
    public:
      void send_compute_equivalence_sets(ShardID target, Serializer &rez);
      void handle_compute_equivalence_sets(Deserializer &derez);
      void handle_equivalence_set_notification(Deserializer &derez);
    public:
      void send_output_equivalence_set(ShardID target, Serializer &rez);
      void handle_output_equivalence_set(Deserializer &derez);
    public:
      void send_refine_equivalence_sets(ShardID target, Serializer &rez);
      void handle_refine_equivalence_sets(Deserializer &derez);
    public:
      void send_intra_space_dependence(ShardID target, Serializer &rez);
      void handle_intra_space_dependence(Deserializer &derez);
    public:
      void broadcast_resource_update(ShardTask *source, Serializer &rez,
                                     std::set<RtEvent> &applied_events);
    public:
      void broadcast_created_region_contexts(ShardTask *source, Serializer &rez,
                                             std::set<RtEvent> &applied_events);
      void send_created_region_contexts(ShardID target, Serializer &rez,
                                        std::set<RtEvent> &applied_events);
      void handle_created_region_contexts(Deserializer &derez,
                                          std::set<RtEvent> &applied_events);
    public:
      bool has_empty_shard_subtree(AddressSpaceID space, 
          ShardingFunction *sharding, IndexSpaceNode *full_space,
          IndexSpace sharding_space);
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
      RtEvent send_trace_event_trigger(TraceID trace_id, AddressSpaceID target,
          ApUserEvent lhs, ApEvent rhs, const TraceLocalID &tlid);
      void send_trace_frontier_request(
                          ShardedPhysicalTemplate *physical_template,
                          ShardID shard_source, AddressSpaceID template_source, 
                          size_t template_index, ApEvent event, 
                          AddressSpaceID event_space, unsigned frontier,
                          RtUserEvent done_event);
      void send_trace_frontier_response(
                          ShardedPhysicalTemplate *physical_template,
                          AddressSpaceID template_source, unsigned frontier,
                          ApBarrier result, RtUserEvent done_event);
      void send_trace_update(ShardID target, Serializer &rez);
      void handle_trace_update(Deserializer &derez, AddressSpaceID source); 
      void send_find_trace_local_sets(ShardID target, Serializer &rez);
      void handle_find_trace_local_sets(Deserializer &derez, 
                                        AddressSpaceID source);
    public:
      ShardID find_collective_owner(RegionTreeID tid) const;
      void send_find_or_create_collective_view(ShardID target, Serializer &rez);
      void handle_find_or_create_collective_view(Deserializer &derez);
    public:
      static void handle_distribution(Deserializer &derez, Runtime *rt); 
      static void handle_collective_versioning(Deserializer &derez,Runtime *rt);
      static void handle_collective_mapping(Deserializer &derez, Runtime *rt);
      static void handle_virtual_rendezvous(Deserializer &derez, Runtime *rt);
      static void handle_post_mapped(Deserializer &derez, Runtime *rt);
      static void handle_trigger_complete(Deserializer &derez, Runtime *rt);
      static void handle_trigger_commit(Deserializer &derez, Runtime *rt);
      static void handle_collective_message(Deserializer &derez, Runtime *rt);
      static void handle_rendezvous_message(Deserializer &derez, Runtime *rt);
      static void handle_compute_equivalence_sets(Deserializer &derez,
                                                  Runtime *runtime);
      static void handle_output_equivalence_set(Deserializer &derez,
                                                Runtime *runtime);
      static void handle_refine_equivalence_sets(Deserializer &derez,
                                                 Runtime *runtime);
      static void handle_equivalence_set_notification(Deserializer &derez, 
                                                      Runtime *rt);
      static void handle_intra_space_dependence(Deserializer &derez, 
                                                Runtime *rt);
      static void handle_broadcast_update(Deserializer &derez, Runtime *rt);
      static void handle_created_regions(Deserializer &derez, Runtime *rt);
      static void handle_trace_event_request(Deserializer &derez, Runtime *rt,
                                             AddressSpaceID request_source);
      static void handle_trace_event_response(Deserializer &derez);
      static void handle_trace_event_trigger(Deserializer &derez, Runtime *rt);
      static void handle_trace_frontier_request(Deserializer &derez,Runtime *rt,
                                                AddressSpaceID request_source);
      static void handle_trace_frontier_response(Deserializer &derez);
      static void handle_trace_update(Deserializer &derez, Runtime *rt,
                                      AddressSpaceID source);
      static void handle_find_trace_local_sets(Deserializer &derez,
          Runtime *runtime, AddressSpaceID source);
      static void handle_find_collective_view(Deserializer &derez, Runtime *rt);
    public:
      ShardingFunction* find_sharding_function(ShardingID sid, 
                                               bool skip_check = false);
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
      const std::vector<DomainPoint> shard_points;
      const std::vector<DomainPoint> sorted_points;
      const std::vector<ShardID> shard_lookup;
      const Domain shard_domain;
      const size_t total_shards;
      SingleTask *const original_task;
      const unsigned local_constituents;
      const unsigned remote_constituents;
      const bool top_level_task;
      const bool isomorphic_points;
      const bool control_replicated; 
    protected:
      mutable LocalLock                manager_lock;
      // Inheritted from Mapper::SelectShardingFunctorInput
      // std::vector<Processor>        shard_mapping;
      ShardMapping*                    address_spaces;
      std::vector<ShardTask*>          local_shards;
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
      unsigned    trigger_local_complete, trigger_remote_complete;
      unsigned    trigger_local_commit,   trigger_remote_commit;
      unsigned    semantic_attach_counter;
      size_t      future_size;
      std::set<RtEvent> mapping_preconditions;
    protected:
      // These barriers only are needed for control replicated tasks
      RtBarrier shard_task_barrier;
      RtBarrier callback_barrier;
    protected:
      std::map<ShardingID,ShardingFunction*> sharding_functions;
    protected:
      // We need a triple here to uniquely identify creations and
      // make sure the right equivalence set gets hooked up with
      // in the right way.
      // 1. index of the creator op in the context
      // 2. number of the refinement for that creator op
      // 3. logical region being refined
      struct EquivalenceSetKey {
      public:
        inline EquivalenceSetKey(void)
          : handle(LogicalRegion::NO_REGION), op_ctx_index(0), 
            refinement_number(0) { }
        inline EquivalenceSetKey(size_t op, unsigned number, LogicalRegion h)
          : handle(h), op_ctx_index(op), refinement_number(number) { }
      public:
        inline bool operator<(const EquivalenceSetKey &rhs) const
        {
          if (op_ctx_index < rhs.op_ctx_index) return true;
          if (op_ctx_index > rhs.op_ctx_index) return false;
          if (refinement_number < rhs.refinement_number) return true;
          if (refinement_number > rhs.refinement_number) return false;
          return (handle < rhs.handle);
        }
      public:
        LogicalRegion handle;
        size_t op_ctx_index;
        unsigned refinement_number;
      };
      struct NewEquivalenceSet {
      public:
        EquivalenceSet *new_set;
        DistributedID did;
        CollectiveMapping *mapping;
        RtUserEvent ready_event;
        size_t remaining;
      };
      std::map<EquivalenceSetKey,NewEquivalenceSet>
                                        created_equivalence_sets;
      std::map<DistributedID,std::pair<FutureImpl*,size_t> > created_futures;
      std::map<DistributedID,std::pair<ReplFutureMapImpl*,size_t> >
                                        created_future_maps;
      std::map<DistributedID,std::pair<FillView*,size_t> > 
                                        created_fill_views;
      // ApEvents describing the completion of each shard
      std::set<ApEvent> shard_effects;
      std::set<RtEvent> commit_preconditions;
#ifdef LEGION_USE_LIBDL
    protected:
      std::set<Runtime::RegistrationKey> unique_registration_callbacks;
#endif
    protected:
      struct ShardLocalData {
      public:
        ShardLocalData(void) : buffer(NULL), size(0), remaining(0) { }
      public:
        void *buffer;
        size_t size;
        RtUserEvent pending;
        unsigned remaining;
      };
      std::map<std::pair<size_t,size_t>,ShardLocalData> shard_local_data;
    protected:
      AttachDeduplication *attach_deduplication;
    protected:
      struct VirtualMappingRendezvous {
        std::vector<bool> virtual_mappings;
        MapperManager *mapper;
        ShardID shard;
        unsigned remaining_arrivals;
      };
      VirtualMappingRendezvous *virtual_mapping_rendezvous;
    };

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_REPLICATION_H__
