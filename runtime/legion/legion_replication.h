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

#ifndef __LEGION_REPLICATION_H__
#define __LEGION_REPLICATION_H__

#include "legion_ops.h"
#include "legion_tasks.h"

namespace Legion {
  namespace Internal {

    /**
     * \class IndexSpaceReduction
     * A class for performing reductions of index spaces
     */
    class IndexSpaceReduction {
    public:
      typedef IndexSpace LHS;
      typedef IndexSpace RHS;
      static const IndexSpace identity;

      template<bool EXCLUSIVE>
      static inline void apply(LHS &lhs, RHS rhs)
      {
#ifdef DEBUG_LEGION
        assert((lhs.exists() && !rhs.exists()) ||
               (!lhs.exists() && rhs.exists()) ||
               (lhs.exists() && (lhs == rhs)));
#endif
        if (rhs.exists())
          lhs = rhs;
      }

      template<bool EXCLUSIVE>
      static inline void fold(RHS &rhs1, RHS rhs2)
      {
#ifdef DEBUG_LEGION
        assert((rhs1.exists() && !rhs2.exists()) ||
               (!rhs1.exists() && rhs2.exists()) ||
               (rhs1.exists() && (rhs1 == rhs2)));
#endif
        if (rhs2.exists())
          rhs1 = rhs2;
      }
    };

    /**
     * \class IndexPartitionReduction
     * A class for performing reductions of index partition IDs
     */
    class IndexPartitionReduction {
    public:
      typedef IndexPartitionID LHS;
      typedef IndexPartitionID RHS;
      static const IndexPartitionID identity;

      template<bool EXCLUSIVE>
      static inline void apply(LHS &lhs, RHS rhs)
      {
#ifdef DEBUG_LEGION
        assert(((lhs != 0) && (rhs == 0)) ||
               ((lhs == 0) && (rhs != 0)) ||
               ((lhs != 0) && (lhs == rhs)));
#endif
        if (rhs != 0)
          lhs = rhs;
      }

      template<bool EXCLUSIVE>
      static inline void fold(RHS &rhs1, RHS rhs2)
      {
#ifdef DEBUG_LEGION
        assert(((rhs1 != 0) && (rhs2 == 0)) ||
               ((rhs1 == 0) && (rhs2 != 0)) ||
               ((rhs1 != 0) && (rhs1 == rhs2)));
#endif
        if (rhs2 != 0)
          rhs1 = rhs2;
      }
    };

    /**
     * \class ColorReduction
     * A class for performing reductions of legion colors
     */
    class ColorReduction {
    public:
      typedef LegionColor LHS;
      typedef LegionColor RHS;
      static const LegionColor identity;

      template<bool EXCLUSIVE>
      static inline void apply(LHS &lhs, RHS rhs)
      {
#ifdef DEBUG_LEGION
        assert(((lhs != INVALID_COLOR) && (rhs == INVALID_COLOR)) ||
               ((lhs == INVALID_COLOR) && (rhs != INVALID_COLOR)) ||
               ((lhs != INVALID_COLOR) && (lhs == rhs)));
#endif
        if (rhs != INVALID_COLOR)
          lhs = rhs;
      }

      template<bool EXCLUSIVE>
      static inline void fold(RHS &rhs1, RHS rhs2)
      {
#ifdef DEBUG_LEGION
        assert(((rhs1 != INVALID_COLOR) && (rhs2 == INVALID_COLOR)) ||
               ((rhs1 == INVALID_COLOR) && (rhs2 != INVALID_COLOR)) ||
               ((rhs1 != INVALID_COLOR) && (rhs1 == rhs2)));
#endif
        if (rhs2 != INVALID_COLOR)
          rhs1 = rhs2;
      }
    };

    /**
     * \class FieldSpaceReduction
     * A class for performing reductions of field spaces
     */
    class FieldSpaceReduction {
    public:
      typedef FieldSpace LHS;
      typedef FieldSpace RHS;
      static const FieldSpace identity;

      template<bool EXCLUSIVE>
      static inline void apply(LHS &lhs, RHS rhs)
      {
#ifdef DEBUG_LEGION
        assert((lhs.exists() && !rhs.exists()) ||
               (!lhs.exists() && rhs.exists()) ||
               (lhs.exists() && (lhs == rhs)));
#endif
        if (rhs.exists())
          lhs = rhs;
      }

      template<bool EXCLUSIVE>
      static inline void fold(RHS &rhs1, RHS rhs2)
      {
#ifdef DEBUG_LEGION
        assert((rhs1.exists() && !rhs2.exists()) ||
               (!rhs1.exists() && rhs2.exists()) ||
               (rhs1.exists() && (rhs1 == rhs2)));
#endif
        if (rhs2.exists())
          rhs1 = rhs2;
      }
    };

    /**
     * \class LogicalRegionReduction
     * A class for performing reductions of region tree IDs
     */
    class LogicalRegionReduction {
    public:
      typedef RegionTreeID LHS;
      typedef RegionTreeID RHS;
      static const RegionTreeID identity;

      template<bool EXCLUSIVE>
      static inline void apply(LHS &lhs, RHS rhs)
      {
#ifdef DEBUG_LEGION
        assert(((lhs != 0) && (rhs == 0)) ||
               ((lhs == 0) && (rhs != 0)) ||
               ((lhs != 0) && (lhs == rhs)));
#endif
        if (rhs != 0)
          lhs = rhs;
      }

      template<bool EXCLUSIVE>
      static inline void fold(RHS &rhs1, RHS rhs2)
      {
#ifdef DEBUG_LEGION
        assert(((rhs1 != 0) && (rhs2 == 0)) ||
               ((rhs1 == 0) && (rhs2 != 0)) ||
               ((rhs1 != 0) && (rhs1 == rhs2)));
#endif
        if (rhs2 != 0)
          rhs1 = rhs2;
      }
    };

    /**
     * \class FieldReduction
     * A class for performing reductions of field IDs
     */
    class FieldReduction {
    public:
      typedef FieldID LHS;
      typedef FieldID RHS;
      static const FieldID identity;

      template<bool EXCLUSIVE>
      static inline void apply(LHS &lhs, RHS rhs)
      {
#ifdef DEBUG_LEGION
        assert(((lhs != 0) && (rhs == 0)) ||
               ((lhs == 0) && (rhs != 0)) ||
               ((lhs != 0) && (lhs == rhs)));
#endif
        if (rhs != 0)
          lhs = rhs;
      }

      template<bool EXCLUSIVE>
      static inline void fold(RHS &rhs1, RHS rhs2)
      {
#ifdef DEBUG_LEGION
        assert(((rhs1 != 0) && (rhs2 == 0)) ||
               ((rhs1 == 0) && (rhs2 != 0)) ||
               ((rhs1 != 0) && (rhs1 == rhs2)));
#endif
        if (rhs2 != 0)
          rhs1 = rhs2;
      }
    };

    /**
     * \class TimingReduction
     * A class for reducing a broadcast of timing measurements
     */
    class TimingReduction {
    public:
      typedef long long LHS;
      typedef long long RHS;
      static const long long identity;

      template<bool EXCLUSIVE>
      static inline void apply(LHS &lhs, RHS rhs)
      {
#ifdef DEBUG_LEGION
        assert(((lhs != 0) && (rhs == 0)) ||
               ((lhs == 0) && (rhs != 0)) ||
               ((lhs != 0) && (lhs == rhs)));
#endif
        if (rhs != 0)
          lhs = rhs;
      }

      template<bool EXCLUSIVE>
      static inline void fold(RHS &rhs1, RHS rhs2)
      {
#ifdef DEBUG_LEGION
        assert(((rhs1 != 0) && (rhs2 == 0)) ||
               ((rhs1 == 0) && (rhs2 != 0)) ||
               ((rhs1 != 0) && (rhs1 == rhs2)));
#endif
        if (rhs2 != 0)
          rhs1 = rhs2;
      }
    };

    /**
     * \class TrueReduction
     * A class for reduction a broadcast of a true value
     */
    class TrueReduction {
    public:
      typedef bool LHS;
      typedef bool RHS;
      static const bool identity;

      template<bool EXCLUSIVE>
      static inline void apply(LHS &lhs, RHS rhs)
      {
        if (rhs)
          lhs = true;
      }

      template<bool EXCLUSIVE>
      static inline void fold(RHS &rhs1, RHS rhs2)
      {
        if (rhs2)
          rhs1 = true;
      }
    };

    /**
     * \class FalseReduction
     * A class for reduction a broadcast of a false value
     */
    class FalseReduction {
    public:
      typedef bool LHS;
      typedef bool RHS;
      static const bool identity;

      template<bool EXCLUSIVE>
      static inline void apply(LHS &lhs, RHS rhs)
      {
        if (!rhs)
          lhs = false;
      }

      template<bool EXCLUSIVE>
      static inline void fold(RHS &rhs1, RHS rhs2)
      {
        if (!rhs2)
          rhs1 = false;
      }
    };

#ifdef DEBUG_LEGION
    /**
     * \class ShardingReduction
     * A class for performing reductions of ShardingIDs
     * down to a single ShardingID. This is used in debug
     * mode to determine if the mappers across all shards
     * chose the same sharding ID for a given operation.
     * This is only used in debug mode
     */
    class ShardingReduction {
    public:
      typedef ShardingID LHS;
      typedef ShardingID RHS;
      static const ShardingID identity;

      template<bool EXCLUSIVE>
      static inline void apply(LHS &lhs, RHS rhs)
      {
#ifdef DEBUG_LEGION
        assert((lhs < UINT_MAX) || (rhs < UINT_MAX));
#endif
        if (lhs == UINT_MAX)
          lhs = rhs;
      }

      template<bool EXCLUSIVE>
      static inline void fold(RHS &rhs1, RHS rhs2)
      {
#ifdef DEBUG_LEGION
        assert((rhs1 < UINT_MAX) || (rhs2 < UINT_MAX));
#endif
        if (rhs1 == UINT_MAX)
          rhs1 = rhs2;
      }
    };
#endif

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
    protected:
      ShardingID sharding_functor;
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
      virtual void trigger_ready(void);
    protected:
      ShardingID sharding_functor;
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
      virtual void trigger_ready(void);
    protected:
      ShardingID sharding_functor;
      MapperManager *mapper;
    };

    /**
     * \class ReplCopyOp
     * A fill operation that is aware that it is being
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
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_ready(void);
    protected:
      ShardingID sharding_functor;
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
      virtual void trigger_ready(void);
    protected:
      ShardingID sharding_functor;
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
      ReplDependentPartitionOp(Runtime *rt);
      ReplDependentPartitionOp(const ReplDependentPartitionOp &rhs);
      virtual ~ReplDependentPartitionOp(void);
    public:
      ReplDependentPartitionOp& operator=(const ReplDependentPartitionOp &rhs);
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
    class ShardManager : public Mapper::SelectShardingFunctorInput {
    public:
      struct ShardManagerCloneArgs :
        public LgTaskArgs<ShardManagerCloneArgs> {
      public:
        static const LgTaskID TASK_ID = LG_CONTROL_REP_CLONE_TASK_ID;
      public:
        ShardManager *manager;
        RtEvent ready_event;
        RtUserEvent to_trigger;
        ShardTask *first_shard;
      };
      struct ShardManagerLaunchArgs :
        public LgTaskArgs<ShardManagerLaunchArgs> {
      public:
        static const LgTaskID TASK_ID = LG_CONTROL_REP_LAUNCH_TASK_ID;
      public:
        ShardManager *manager;
      };
      struct ShardManagerDeleteArgs :
        public LgTaskArgs<ShardManagerDeleteArgs> {
      public:
        static const LgTaskID TASK_ID = LG_CONTROL_REP_DELETE_TASK_ID;
      public:
        ShardManager *manager;
      };
    public:
      ShardManager(Runtime *rt, ControlReplicationID repl_id, size_t total,
                   unsigned address_space_index, AddressSpaceID owner_space,
                   SingleTask *original = NULL);
      ShardManager(const ShardManager &rhs);
      ~ShardManager(void);
    public:
      ShardManager& operator=(const ShardManager &rhs);
    public:
      inline ApBarrier get_pending_partition_barrier(void) const
        { return pending_partition_barrier; }
    public:
      inline ShardMapping* get_mapping(void) const
        { return address_spaces; }
    public:
      void launch(const std::vector<AddressSpaceID> &spaces,
                  const std::map<ShardID,Processor> &shard_mapping);
      void unpack_launch(Deserializer &derez);
      void clone_and_launch(RtEvent ready, RtUserEvent to_trigger, 
                            ShardTask *first_shard);
      void create_shards(void);
      void launch_shards(void) const;
    public:
      void broadcast_launch(RtEvent start, RtUserEvent to_trigger,
                            SingleTask *to_clone);
      bool broadcast_delete(
              RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT);
    public:
      void handle_post_mapped(bool local);
      void handle_future(const void *res, size_t res_size, bool owned);
      void trigger_task_complete(bool local);
      void trigger_task_commit(bool local);
    public:
      void send_collective_message(ShardID target, Serializer &rez);
      void handle_collective_message(Deserializer &derez);
    public:
      static void handle_clone(const void *args);
      static void handle_launch(const void *args);
      static void handle_delete(const void *args);
    public:
      static void handle_launch(Deserializer &derez, Runtime *rt, 
                                AddressSpaceID source);
      static void handle_delete(Deserializer &derez, Runtime *rt);
      static void handle_post_mapped(Deserializer &derez, Runtime *rt);
      static void handle_trigger_complete(Deserializer &derez, Runtime *rt);
      static void handle_trigger_commit(Deserializer &derez, Runtime *rt);
      static void handle_collective_stage(Deserializer &derez, Runtime *rt);
    public:
      ShardingFunction* find_sharding_function(ShardingID sid);
    public:
      Runtime *const runtime;
      const ControlReplicationID repl_id;
      const size_t total_shards;
      const unsigned address_space_index;
      const AddressSpaceID owner_space;
      SingleTask *const original_task;
    protected:
      Reservation                      manager_lock;
      // Inheritted from Mapper::SelectShardingFunctorInput
      // std::map<ShardID,Processor>   shard_mapping;
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
      unsigned    remote_constituents;
      bool        first_future;
    protected:
      ApBarrier pending_partition_barrier;
    protected:
      std::map<ShardingID,ShardingFunction*> sharding_functions;
    };

    /**
     * \class ShardCollective
     * The shard collective is the base class for performing
     * collective operations between shards
     */
    class ShardCollective {
    public:
      ShardCollective(ReplicateContext *ctx);
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
      Reservation collective_lock;
    };

    /**
     * \class BroadcastCollective
     * This shard collective has equivalent functionality to 
     * MPI Broadcast in that it will transmit some data on one
     * shard to all the other shards.
     */
    class BroadcastCollective : public ShardCollective {
    public:
      BroadcastCollective(ReplicateContext *ctx, ShardID origin);
      virtual ~BroadcastCollective(void);
    public:
      // We guarantee that these methods will be called atomically
      virtual void pack_collective(Serializer &rez) const = 0;
      virtual void unpack_collective(Deserializer &derez) = 0;
    public:
      void perform_collective_async(void) const;
      void perform_collective_wait(void) const;
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
      GatherCollective(ReplicateContext *ctx, ShardID target);
      virtual ~GatherCollective(void);
    public:
      // We guarantee that these methods will be called atomically
      virtual void pack_collective(Serializer &rez) const = 0;
      virtual void unpack_collective(Deserializer &derez) = 0;
    public:
      void perform_collective_async(void);
      void perform_collective_wait(void);
      virtual void handle_collective_message(Deserializer &derez);
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
      AllGatherCollective(ReplicateContext *ctx);
      virtual ~AllGatherCollective(void);
    public:
      // We guarantee that these methods will be called atomically
      virtual void pack_collective_stage(Serializer &rez, int stage) const = 0;
      virtual void unpack_collective_stage(Deserializer &derez, int stage) = 0;
    public:
      void perform_collective_sync(void);
      void perform_collective_async(void);
      void perform_collective_wait(void);
      virtual void handle_collective_message(Deserializer &derez);
    protected:
      void send_explicit_stage(int stage);
      bool send_ready_stages(void);
      void construct_message(ShardID target, int stage, Serializer &rez) const;
      void unpack_stage(int stage, Deserializer &derez);
    public: 
      const int shard_collective_radix;
      const int shard_collective_log_radix;
      const int shard_collective_stages;
      const int shard_collective_participating_shards;
      const int shard_collective_last_radix;
      const int shard_collective_last_log_radix;
      const bool participating; 
    private:
      RtUserEvent done_event;
      std::vector<int> stage_notifications;
      std::vector<bool> sent_stages;
    };

    /**
     * \class BarrierExchangeCollective
     * A class for exchanging sets of barriers between shards
     */
    class BarrierExchangeCollective : public AllGatherCollective {
    public:
      BarrierExchangeCollective(ReplicateContext *ctx, size_t window_size, 
                                std::vector<RtBarrier> &barriers);
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
      std::vector<RtBarrier> &barriers;
      std::map<unsigned,RtBarrier> local_barriers;
    };

    /**
     * \class ValueBroadcast
     * This will broadcast a value of any type that can be 
     * trivially serialized to all the shards.
     */
    template<typename T>
    class ValueBroadcast : public BroadcastCollective {
    public:
      ValueBroadcast(ReplicateContext *ctx)
        : BroadcastCollective(ctx, ctx->owner_shard->shard_id) { }
      ValueBroadcast(ReplicateContext *ctx, ShardID origin)
        : BroadcastCollective(ctx, origin) { }
      ValueBroadcast(const ValueBroadcast &rhs) { assert(false); }
      virtual ~ValueBroadcast(void) { }
    public:
      ValueBroadcast& operator=(const ValueBroadcast &rhs)
        { assert(false); return *this; }
      inline void broadcast(const T &v) 
        { value = v; perform_collective_async(); }
      inline operator T(void) const { perform_collective_wait(); return value; }
    public:
      virtual void pack_collective(Serializer &rez) const 
        { rez.serialize(value); }
      virtual void unpack_collective(Deserializer &derez)
        { derez.deserialize(value); }
    protected:
      T value;
    };

    /**
     * \class CrossProductExchange
     * A class for exchanging the names of partitions created by
     * a call for making cross-product partitions
     */
    class CrossProductCollective : public AllGatherCollective {
    public:
      CrossProductCollective(ReplicateContext *ctx);
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

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_REPLICATION_H__
