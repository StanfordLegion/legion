/* Copyright 2019 Stanford University, NVIDIA Corporation
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

    /**
     * \class ShardCollective
     * The shard collective is the base class for performing
     * collective operations between shards
     */
    class ShardCollective {
    public:
      ShardCollective(CollectiveIndexLocation loc, ReplicateContext *ctx);
      ShardCollective(ReplicateContext *ctx, CollectiveID id);
      virtual ~ShardCollective(void);
    public:
      virtual void handle_collective_message(Deserializer &derez) = 0;
    protected:
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
      void perform_collective_async(void);
      RtEvent perform_collective_wait(bool block = true);
      virtual void handle_collective_message(Deserializer &derez);
    public:
      RtEvent get_done_event(void) const;
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
      void perform_collective_async(void);
      // Make sure to call this in the destructor of anything not the target
      RtEvent perform_collective_wait(bool block = true);
      virtual void handle_collective_message(Deserializer &derez);
      inline bool is_target(void) const { return (target == local_shard); }
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
    class AllGatherCollective : public ShardCollective {
    public:
      AllGatherCollective(CollectiveIndexLocation loc, ReplicateContext *ctx);
      AllGatherCollective(ReplicateContext *ctx, CollectiveID id);
      virtual ~AllGatherCollective(void);
    public:
      // We guarantee that these methods will be called atomically
      virtual void pack_collective_stage(Serializer &rez, int stage) const = 0;
      virtual void unpack_collective_stage(Deserializer &derez, int stage) = 0;
    public:
      void perform_collective_sync(void);
      void perform_collective_async(void);
      RtEvent perform_collective_wait(bool block = true);
      virtual void handle_collective_message(Deserializer &derez);
      // Use this method in case we don't actually end up using the collective
      void elide_collective(void);
    protected:
      void initialize_collective(void);
      void construct_message(ShardID target, int stage, Serializer &rez) const;
      bool initiate_collective(void);
      void send_remainder_stage(void);
      bool send_ready_stages(const int start_stage=1);
      void unpack_stage(int stage, Deserializer &derez);
      void complete_exchange(void);
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
      // Handle a small race on deciding who gets to
      // trigger the done event, only the last one of these
      // will get to do the trigger to avoid any races
      unsigned pending_send_ready_stages;
#ifdef DEBUG_LEGION
      bool done_triggered;
#endif
    };

    /**
     * \class BarrierExchangeCollective
     * A class for exchanging sets of barriers between shards
     */
    template<typename BAR>
    class BarrierExchangeCollective : public AllGatherCollective {
    public:
      BarrierExchangeCollective(ReplicateContext *ctx, size_t window_size, 
                                typename std::vector<BAR> &barriers,
                                CollectiveIndexLocation loc);
      BarrierExchangeCollective(const BarrierExchangeCollective &rhs);
      virtual ~BarrierExchangeCollective(void);
    public:
      BarrierExchangeCollective& operator=(const BarrierExchangeCollective &rs);
    public:
      void exchange_barriers_async(void);
      void wait_for_barrier_exchange(void);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage) const;
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    protected:
      const size_t window_size;
      std::vector<BAR> &barriers;
      std::map<unsigned,BAR> local_barriers;
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
    class ValueExchange : public AllGatherCollective { 
    public:
      ValueExchange(CollectiveIndexLocation loc, ReplicateContext *ctx)
        : AllGatherCollective(loc, ctx) { }
      ValueExchange(ReplicateContext *ctx, CollectiveID id)
        : AllGatherCollective(ctx, id) { }
      virtual ~ValueExchange(void) { }
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage) const
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
    class ShardSyncTree : public BroadcastCollective {
    public:
      ShardSyncTree(ReplicateContext *ctx, ShardID origin, 
                    CollectiveIndexLocation loc);
      ShardSyncTree(const ShardSyncTree &rhs) 
        : BroadcastCollective(rhs), is_origin(false) 
        { assert(false); }
      virtual ~ShardSyncTree(void);
    public:
      ShardSyncTree& operator=(const ShardSyncTree &rhs) 
        { assert(false); return *this; }
    public:
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
    protected:
      RtUserEvent done_event;
      mutable std::set<RtEvent> done_preconditions;
      const bool is_origin;
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
      ShardEventTree(const ShardEventTree &rhs) 
        : BroadcastCollective(rhs), is_origin(false) { assert(false); }
      virtual ~ShardEventTree(void);
    public:
      ShardEventTree& operator=(const ShardEventTree &rhs) 
        { assert(false); return *this; }
    public:
      void signal_tree(RtEvent precondition); // origin
      RtEvent get_local_event(void);
    public:
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
    protected:
      RtUserEvent local_event;
      RtEvent trigger_event;
      RtEvent finished_event;
      const bool is_origin;
    };

    /**
     * \class CrossProductExchange
     * A class for exchanging the names of partitions created by
     * a call for making cross-product partitions
     */
    class CrossProductCollective : public AllGatherCollective {
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
      virtual void pack_collective_stage(Serializer &rez, int stage) const;
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
    class IndirectRecordExchange : public AllGatherCollective {
    public:
      struct IndirectKey {
      public:
        IndirectKey(void) { }
        IndirectKey(PhysicalInstance i, ApEvent e, const Domain &d)
          : inst(i), ready_event(e), domain(d) { }
      public:
        inline bool operator<(const IndirectKey &rhs) const 
        {
          if (inst.id < rhs.inst.id)
            return true;
          if (inst.id > rhs.inst.id)
            return false;
          if (ready_event.id < rhs.ready_event.id)
            return true;
          if (ready_event.id > rhs.ready_event.id)
            return false;
          return (domain < rhs.domain);
        }
        inline bool operator==(const IndirectKey &rhs) const
        {
          if (inst.id != rhs.inst.id)
            return false;
          if (ready_event.id != rhs.ready_event.id)
            return false;
          return (domain == rhs.domain);
        }
      public:
        PhysicalInstance inst;
        ApEvent ready_event;
        Domain domain;
      };
    public:
      IndirectRecordExchange(ReplicateContext *ctx,
                             CollectiveIndexLocation loc);
      IndirectRecordExchange(const IndirectRecordExchange &rhs);
      virtual ~IndirectRecordExchange(void);
    public:
      IndirectRecordExchange& operator=(const IndirectRecordExchange &rhs);
    public:
      void exchange_records(LegionVector<IndirectRecord>::aligned &records);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage) const;
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    protected:
      LegionMap<IndirectKey,FieldMask>::aligned records;
    };
    
    /**
     * \class FieldDescriptorExchange
     * A class for doing an all-gather of field descriptors for 
     * doing dependent partitioning operations. This will also build
     * a butterfly tree of user events that will be used to know when
     * all of the constituent shards are done with the operation they
     * are collectively performing together.
     */
    class FieldDescriptorExchange : public AllGatherCollective {
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
      virtual void pack_collective_stage(Serializer &rez, int stage) const;
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
      FutureBroadcast(ReplicateContext *ctx, CollectiveID id, ShardID source);
      FutureBroadcast(const FutureBroadcast &rhs);
      virtual ~FutureBroadcast(void);
    public:
      FutureBroadcast& operator=(const FutureBroadcast &rhs);
    public:
      virtual void pack_collective(Serializer &rez) const;
      virtual void unpack_collective(Deserializer &derez);
    public:
      void broadcast_future(const void *result, size_t result_size);
      void receive_future(FutureImpl *f);
    protected:
      void *result;
      size_t result_size;
    };

    /**
     * \class FutureExchange
     * A class for doing an all-to-all exchange of future values
     */
    class FutureExchange : public AllGatherCollective {
    public:
      FutureExchange(ReplicateContext *ctx, size_t future_size,
                     CollectiveIndexLocation loc);
      FutureExchange(const FutureExchange &rhs);
      virtual ~FutureExchange(void);
    public:
      FutureExchange& operator=(const FutureExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage) const;
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      // This takes ownership of the buffer
      RtEvent exchange_futures(void *value);
      void reduce_futures(ReplIndexTask *target);
    public:
      const size_t future_size;
    protected:
      std::map<ShardID,void*> results;
    };

    /**
     * \class FutureNameExchange
     * A class for doing an all-to-all exchange of future names
     */
    class FutureNameExchange : public AllGatherCollective {
    public:
      FutureNameExchange(ReplicateContext *ctx, CollectiveID id);
      FutureNameExchange(const FutureNameExchange &rhs);
      virtual ~FutureNameExchange(void);
    public:
      FutureNameExchange& operator=(const FutureNameExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage) const;
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void exchange_future_names(std::map<DomainPoint,Future> &futures);
    protected:
      std::map<DomainPoint,Future> results;
      LocalReferenceMutator mutator;
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
               std::map<PhysicalManager*,std::pair<unsigned,bool> > &acquired);
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
    class MustEpochMappingExchange : public AllGatherCollective {
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
      virtual void pack_collective_stage(Serializer &rez, int stage) const;
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
              std::map<PhysicalManager*,std::pair<unsigned,bool> > &acquired);
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
    class MustEpochDependenceExchange : public AllGatherCollective {
    public:
      MustEpochDependenceExchange(ReplicateContext *ctx, 
                                  CollectiveIndexLocation loc);
      MustEpochDependenceExchange(const MustEpochDependenceExchange &rhs);
      virtual ~MustEpochDependenceExchange(void);
    public:
      MustEpochDependenceExchange& operator=(
                                  const MustEpochDependenceExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage) const;
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
    class MustEpochCompletionExchange : public AllGatherCollective {
    public:
      MustEpochCompletionExchange(ReplicateContext *ctx,
                                  CollectiveIndexLocation loc);
      MustEpochCompletionExchange(const MustEpochCompletionExchange &rhs);
      virtual ~MustEpochCompletionExchange(void);
    public:
      MustEpochCompletionExchange& operator=(
                                    const MustEpochCompletionExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage) const;
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
    class ShardedMappingExchange : public AllGatherCollective {
    public:
      ShardedMappingExchange(CollectiveIndexLocation loc, ReplicateContext *ctx,
                             ShardID shard_id, bool check_mappings);
      ShardedMappingExchange(const ShardedMappingExchange &rhs);
      virtual ~ShardedMappingExchange(void);
    public:
      ShardedMappingExchange& operator=(const ShardedMappingExchange &rhs);
    public:
      virtual void pack_collective_stage(Serializer &rez, int stage) const;
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
      std::map<DistributedID,LegionMap<ShardID,FieldMask>::aligned> mappings;
      LegionMap<DistributedID,FieldMask>::aligned global_views;
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
    public:
      // Override these so we can broadcast the future result
      virtual void handle_future(const void *res, size_t res_size, bool owned);
      virtual void trigger_task_complete(bool deferred = false);
    public:
      void initialize_replication(ReplicateContext *ctx);
      void set_sharding_function(ShardingID functor,ShardingFunction *function);
    protected:
      ShardID owner_shard;
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
      CollectiveID mapped_collective_id; // id for mapped event broadcast
      CollectiveID future_collective_id; // id for the future broadcast 
      ShardEventTree *mapped_collective;
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
    public:
      // Override this so we can exchange reduction results
      virtual void trigger_task_complete(bool deferred = false);
      // Have to override this too for doing output in the
      // case that we misspeculate
      virtual void resolve_false(bool speculated, bool launched);
    public:
      void initialize_replication(ReplicateContext *ctx);
      void set_sharding_function(ShardingID functor,ShardingFunction *function);
      virtual FutureMapImpl* create_future_map(TaskContext *ctx,
                    IndexSpace launch_space, IndexSpace shard_space);
    protected:
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
      FutureExchange *reduction_collective;
#ifdef DEBUG_LEGION
    public:
      inline void set_sharding_collective(ShardingGatherCollective *collective)
        { sharding_collective = collective; }
    protected:
      ShardingGatherCollective *sharding_collective;
#endif
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
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void); 
    protected:
      RtBarrier mapped_barrier;
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
      virtual ApEvent exchange_indirect_records(const unsigned index,
          const ApEvent local_done, const PhysicalTraceInfo &trace_info,
          const InstanceSet &instances, const IndexSpace space,
          LegionVector<IndirectRecord>::aligned &records, const bool sources);
    public:
      void initialize_replication(ReplicateContext *ctx,
                                  std::vector<ApBarrier> &indirection_bars,
                                  unsigned &next_indirection_index);
    protected:
      ShardingID sharding_functor;
      ShardingFunction *sharding_function;
      std::vector<ApBarrier> indirection_barriers;
      std::vector<IndirectRecordExchange*> src_collectives;
      std::vector<IndirectRecordExchange*> dst_collectives;
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
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void trigger_complete(void);
    public:
      void initialize_replication(ReplicateContext *ctx, 
          RtBarrier &deletion_barrier, bool is_total, bool is_first);
    protected:
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
      virtual void trigger_mapping(void);
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
                               FieldID fid, MapperID id, MappingTagID tag);
      void initialize_by_image(ReplicateContext *ctx,
#ifndef SHARD_BY_IMAGE
                               ShardID target,
#endif
                               ApEvent ready_event, IndexPartition pid,
                               LogicalPartition projection,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag,
                               ShardID shard, size_t total_shards);
      void initialize_by_image_range(ReplicateContext *ctx,
#ifndef SHARD_BY_IMAGE
                               ShardID target,
#endif
                               ApEvent ready_event, IndexPartition pid,
                               LogicalPartition projection,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag,
                               ShardID shard, size_t total_shards);
      void initialize_by_preimage(ReplicateContext *ctx, ShardID target,
                               ApEvent ready_event, IndexPartition pid,
                               IndexPartition projection, LogicalRegion handle,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag);
      void initialize_by_preimage_range(ReplicateContext *ctx, ShardID target, 
                               ApEvent ready_event, IndexPartition pid,
                               IndexPartition projection, LogicalRegion handle,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag);
      // nothing special about association for control replication
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      // Need to pick our sharding functor
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_ready(void);  
    protected:
      ShardingID sharding_functor;
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
                 IndexSpace launch_space, IndexSpace shard_space);
      virtual void instantiate_tasks(InnerContext *ctx,
                                     const MustEpochLauncher &launcher);
      virtual MapperManager* invoke_mapper(void);
      virtual void map_and_distribute(std::set<RtEvent> &tasks_mapped,
                                      std::set<ApEvent> &tasks_complete);
      virtual bool has_prepipeline_stage(void) const { return true; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_commit(void);
      void map_replicate_tasks(void) const;
      void distribute_replicate_tasks(void) const;
    public:
      void initialize_replication(ReplicateContext *ctx);
      Domain get_shard_domain(void) const;
    public:
      static IndexSpace create_temporary_launch_space(Runtime *runtime,
                                  RegionTreeForest *forest, Context ctx, 
                                  const MustEpochLauncher &launcher);
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
      virtual void deferred_execute(void);
    public:
      inline void set_timing_collective(ValueBroadcast<long long> *collective) 
        { timing_collective = collective; }
    protected:
      ValueBroadcast<long long> *timing_collective;
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
      void initialize_repl_fence(ReplicateContext *ctx, FenceKind kind);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_mapping(void);
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
      void initialize_replication(ReplicateContext *ctx, RtBarrier &inline_bar);
      RtEvent complete_inline_mapping(RtEvent mapping_applied,
                                      const InstanceSet &mapped_insts);
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
      void initialize_replication(ReplicateContext *ctx,
                                  RtBarrier &resource_bar);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
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
      void initialize_replication(ReplicateContext *ctx,
                                  RtBarrier &resource_bar);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
    public:
      RtBarrier resource_barrier;
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
      ShardManager(Runtime *rt, ReplicationID repl_id, 
                   bool control, bool top, size_t total_shards,
                   AddressSpaceID owner_space, SingleTask *original = NULL,
                   RtBarrier startup_barrier = RtBarrier::NO_RT_BARRIER);
      ShardManager(const ShardManager &rhs);
      ~ShardManager(void);
    public:
      ShardManager& operator=(const ShardManager &rhs);
    public:
      inline ApBarrier get_pending_partition_barrier(void) const
        { return pending_partition_barrier; }
      inline ApBarrier get_future_map_barrier(void) const
        { return future_map_barrier; }
      inline RtBarrier get_creation_barrier(void) const
        { return creation_barrier; }
      inline RtBarrier get_deletion_barrier(void) const
        { return deletion_barrier; }
      inline RtBarrier get_inline_mapping_barrier(void) const
        { return inline_mapping_barrier; }
      inline RtBarrier get_external_resource_barrier(void) const
        { return external_resource_barrier; }
      inline RtBarrier get_mapping_fence_barrier(void) const
        { return mapping_fence_barrier; }
      inline ApBarrier get_execution_fence_barrier(void) const
        { return execution_fence_barrier; }
#ifdef DEBUG_LEGION_COLLECTIVES
      inline RtBarrier get_collective_check_barrier(void) const
        { return collective_check_barrier; }
      inline RtBarrier get_close_check_barrier(void) const
        { return close_check_barrier; }
#endif
    public:
      inline ShardMapping& get_mapping(void) const
        { return *address_spaces; }
      inline AddressSpaceID get_shard_space(ShardID sid) const
        { return (*address_spaces)[sid]; }    
      inline bool is_first_local_shard(ShardTask *task) const
        { return (local_shards[0] == task); }
    public:
      void set_shard_mapping(const std::vector<Processor> &shard_mapping);
      ShardTask* create_shard(ShardID id, Processor target);
      void extract_event_preconditions(const std::deque<InstanceSet> &insts);
      void launch(void);
      void distribute_shards(AddressSpaceID target,
                             const std::vector<ShardTask*> &shards);
      void unpack_shards_and_launch(Deserializer &derez);
      void launch_shard(ShardTask *task,
                        RtEvent precondition = RtEvent::NO_RT_EVENT) const;
      void complete_startup_initialization(void) const;
      // Return true if we have a shard on every address space
      bool is_total_sharding(void);
    public:
      void handle_post_mapped(bool local, RtEvent precondition);
      void handle_post_execution(const void *res, size_t res_size, 
                                 bool owned, bool local);
      void trigger_task_complete(bool local);
      void trigger_task_commit(bool local);
    public:
      void send_collective_message(ShardID target, Serializer &rez);
      void handle_collective_message(Deserializer &derez);
    public:
      void send_future_map_request(ShardID target, Serializer &rez);
      void handle_future_map_request(Deserializer &derez);
    public:
      void send_equivalence_set_request(ShardID target, Serializer &rez);
      void handle_equivalence_set_request(Deserializer &derez);
    public:
      RtEvent broadcast_resource_update(ShardTask *source, Serializer &rez);
      void handle_resource_update(Deserializer &derez);
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
      static void handle_top_view_request(Deserializer &derez, Runtime *rt,
                                          AddressSpaceID request_source);
      static void handle_top_view_response(Deserializer &derez, Runtime *rt);
      static void handle_eq_request(Deserializer &derez, Runtime *rt);
      static void handle_resource_update(Deserializer &derez, Runtime *rt);
    public:
      ShardingFunction* find_sharding_function(ShardingID sid);
    public:
      void create_instance_top_view(PhysicalManager *manager, 
                                    AddressSpaceID source, 
                                    ReplicateContext *request_context,
                                    AddressSpaceID request_source,
                                    bool handle_now = false);
    public:
      Runtime *const runtime;
      const ReplicationID repl_id;
      const AddressSpaceID owner_space;
      const size_t total_shards;
      SingleTask *const original_task;
      const bool control_replicated;
      const bool top_level_task;
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
      unsigned    local_execution_complete, remote_execution_complete;
      unsigned    trigger_local_complete, trigger_remote_complete;
      unsigned    trigger_local_commit,   trigger_remote_commit;
      unsigned    remote_constituents;
      void*       local_future_result; size_t local_future_size;
      bool        local_future_set;
      std::set<RtEvent> mapping_preconditions;
    protected:
      RtBarrier startup_barrier;
      ApBarrier pending_partition_barrier;
      ApBarrier future_map_barrier;
      RtBarrier creation_barrier;
      RtBarrier deletion_barrier;
      RtBarrier inline_mapping_barrier;
      RtBarrier external_resource_barrier;
      RtBarrier mapping_fence_barrier;
      ApBarrier execution_fence_barrier;
      ApBarrier attach_broadcast_barrier;
      ApBarrier attach_reduce_barrier;
#ifdef DEBUG_LEGION_COLLECTIVES
      RtBarrier collective_check_barrier;
      RtBarrier close_check_barrier;
#endif
    protected:
      std::map<ShardingID,ShardingFunction*> sharding_functions;
    protected:
      // A unique set of address spaces on which shards exist 
      std::set<AddressSpaceID> unique_shard_spaces;
    };

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_REPLICATION_H__
