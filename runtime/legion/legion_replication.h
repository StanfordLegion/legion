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
      virtual void trigger_ready(void);
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
      inline RtBarrier get_index_space_allocator_barrier(void) const
        { return index_space_allocator_barrier; }
      inline RtBarrier get_index_partition_allocator_barrier(void) const
        { return index_partition_allocator_barrier; }
      inline RtBarrier get_color_partition_allocator_barrier(void) const
        { return color_partition_allocator_barrier; }
      inline RtBarrier get_field_space_allocator_barrier(void) const
        { return field_space_allocator_barrier; }
      inline RtBarrier get_field_allocator_barrier(void) const
        { return field_allocator_barrier; }
      inline RtBarrier get_logical_region_allocator_barrier(void) const
        { return logical_region_allocator_barrier; }
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
      void send_shard_collective_stage(ShardID target, Serializer &rez);
      void notify_collective_stage(Deserializer &derez);
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
      std::vector<AddressSpaceID>      address_spaces;
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
    protected: // Allocation barriers to be passed to shards
      RtBarrier index_space_allocator_barrier;
      RtBarrier index_partition_allocator_barrier;
      RtBarrier color_partition_allocator_barrier;
      RtBarrier field_space_allocator_barrier;
      RtBarrier field_allocator_barrier;
      RtBarrier logical_region_allocator_barrier;
    protected:
      std::map<ShardingID,ShardingFunction*> sharding_functions;
    };

    /**
     * \class ShardCollective
     * The shard collective is used for performing all-to-all
     * exchanges amongst the shards of a control replicated task
     */
    class ShardCollective {
    public:
      ShardCollective(ReplicateContext *ctx);
      virtual ~ShardCollective(void);
    public:
      // We guarantee that these methods will be called atomically
      virtual void pack_collective_stage(Serializer &rez, int stage) = 0;
      virtual void unpack_collective_stage(Deserializer &derez, int stage) = 0;
    public:
      void perform_collective_sync(void);
      void perform_collective_async(void);
      void perform_collective_wait(void);
      void handle_unpack_stage(Deserializer &derez);
    protected:
      void send_explicit_stage(int stage);
      bool send_ready_stages(void);
      void unpack_stage(int stage, Deserializer &derez);
    public:
      ShardManager *const manager;
      ReplicateContext *const context;
      const ShardID local_shard;
      const CollectiveID collective_index;
      const int shard_collective_radix;
      const int shard_collective_log_radix;
      const int shard_collective_stages;
      const int shard_collective_participating_shards;
      const int shard_collective_last_radix;
      const int shard_collective_last_log_radix;
      const bool participating;
    protected:
      Reservation collective_lock;
    private:
      RtUserEvent done_event;
      std::vector<int> stage_notifications;
      std::vector<bool> sent_stages;
    };

    class BarrierExchangeCollective : public ShardCollective {
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
      virtual void pack_collective_stage(Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    protected:
      const size_t window_size;
      std::vector<RtBarrier> &barriers;
      std::map<unsigned,RtBarrier> local_barriers;
    };

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_REPLICATION_H__
