/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_COLLECTIVES_H__
#define __LEGION_COLLECTIVES_H__

#include "legion/kernel/garbage_collection.h"
#include "legion/kernel/metatask.h"
#include "legion/api/data.h"
#include "legion/managers/message.h"
#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    static inline bool configure_collective_settings(
        const int participants, const int local_space, int& collective_radix,
        int& collective_log_radix, int& collective_stages,
        int& participating_spaces, int& collective_last_radix)
    //--------------------------------------------------------------------------
    {
      legion_assert(participants > 0);
      legion_assert(collective_radix > 1);
      const int MultiplyDeBruijnBitPosition[32] = {
          0, 9,  1,  10, 13, 21, 2,  29, 11, 14, 16, 18, 22, 25, 3, 30,
          8, 12, 20, 28, 15, 17, 24, 7,  19, 27, 23, 6,  26, 5,  4, 31};
      // First adjust the radix based on the number of nodes if necessary
      if (collective_radix > participants)
      {
        if (participants == 1)
        {
          // Handle the unsual case of a single participant
          collective_radix = 0;
          collective_log_radix = 0;
          collective_stages = 0;
          participating_spaces = 1;
          collective_last_radix = 0;
          return (local_space == 0);
        }
        else
          collective_radix = participants;
      }
      // Adjust the radix to the next smallest power of 2
      uint32_t radix_copy = collective_radix;
      for (int i = 0; i < 5; i++) radix_copy |= radix_copy >> (1 << i);
      collective_log_radix = MultiplyDeBruijnBitPosition
          [(uint32_t)(radix_copy * 0x07C4ACDDU) >> 27];
      if (collective_radix != (1 << collective_log_radix))
        collective_radix = (1 << collective_log_radix);

      // Compute the number of stages
      uint32_t node_copy = participants;
      for (int i = 0; i < 5; i++) node_copy |= node_copy >> (1 << i);
      // Now we have it log 2
      int log_nodes = MultiplyDeBruijnBitPosition
          [(uint32_t)(node_copy * 0x07C4ACDDU) >> 27];

      // Stages round up in case of incomplete stages
      collective_stages =
          (log_nodes + collective_log_radix - 1) / collective_log_radix;
      int log_remainder = log_nodes % collective_log_radix;
      if (log_remainder > 0)
      {
        // We have an incomplete last stage
        collective_last_radix = 1 << log_remainder;
        // Now we can compute the number of participating stages
        participating_spaces = 1
                               << ((collective_stages - 1) *
                                       collective_log_radix +
                                   log_remainder);
      }
      else
      {
        collective_last_radix = collective_radix;
        participating_spaces = 1 << (collective_stages * collective_log_radix);
      }
      legion_assert((participating_spaces % collective_radix) == 0);
      const bool participant = (local_space < participating_spaces);
      return participant;
    }

    // Static locations for where collectives are allocated
    // These are just arbitrary numbers but they should appear
    // with at most one logical static collective kind
    // Ones that have been commented out are free to be reused
    enum CollectiveIndexLocation {
      // COLLECTIVE_LOC_0 = 0,
      COLLECTIVE_LOC_1 = 1,
      COLLECTIVE_LOC_2 = 2,
      COLLECTIVE_LOC_3 = 3,
      COLLECTIVE_LOC_4 = 4,
      COLLECTIVE_LOC_5 = 5,
      COLLECTIVE_LOC_6 = 6,
      COLLECTIVE_LOC_7 = 7,
      COLLECTIVE_LOC_8 = 8,
      COLLECTIVE_LOC_9 = 9,
      COLLECTIVE_LOC_10 = 10,
      COLLECTIVE_LOC_11 = 11,
      COLLECTIVE_LOC_12 = 12,
      COLLECTIVE_LOC_13 = 13,
      COLLECTIVE_LOC_14 = 14,
      COLLECTIVE_LOC_15 = 15,
      COLLECTIVE_LOC_16 = 16,
      COLLECTIVE_LOC_17 = 17,
      COLLECTIVE_LOC_18 = 18,
      COLLECTIVE_LOC_19 = 19,
      COLLECTIVE_LOC_20 = 20,
      COLLECTIVE_LOC_21 = 21,
      COLLECTIVE_LOC_22 = 22,
      COLLECTIVE_LOC_23 = 23,
      COLLECTIVE_LOC_24 = 24,
      COLLECTIVE_LOC_25 = 25,
      COLLECTIVE_LOC_26 = 26,
      COLLECTIVE_LOC_27 = 27,
      COLLECTIVE_LOC_28 = 28,
      COLLECTIVE_LOC_29 = 29,
      COLLECTIVE_LOC_30 = 30,
      COLLECTIVE_LOC_31 = 31,
      COLLECTIVE_LOC_32 = 32,
      COLLECTIVE_LOC_33 = 33,
      COLLECTIVE_LOC_34 = 34,
      COLLECTIVE_LOC_35 = 35,
      COLLECTIVE_LOC_36 = 36,
      COLLECTIVE_LOC_37 = 37,
      COLLECTIVE_LOC_38 = 38,
      COLLECTIVE_LOC_39 = 39,
      COLLECTIVE_LOC_40 = 40,
      COLLECTIVE_LOC_41 = 41,
      COLLECTIVE_LOC_42 = 42,
      COLLECTIVE_LOC_43 = 43,
      COLLECTIVE_LOC_44 = 44,
      COLLECTIVE_LOC_45 = 45,
      COLLECTIVE_LOC_46 = 46,
      COLLECTIVE_LOC_47 = 47,
      COLLECTIVE_LOC_48 = 48,
      COLLECTIVE_LOC_49 = 49,
      COLLECTIVE_LOC_50 = 50,
      COLLECTIVE_LOC_51 = 51,
      COLLECTIVE_LOC_52 = 52,
      COLLECTIVE_LOC_53 = 53,
      COLLECTIVE_LOC_54 = 54,
      COLLECTIVE_LOC_55 = 55,
      COLLECTIVE_LOC_56 = 56,
      COLLECTIVE_LOC_57 = 57,
      COLLECTIVE_LOC_58 = 58,
      COLLECTIVE_LOC_59 = 59,
      COLLECTIVE_LOC_60 = 60,
      COLLECTIVE_LOC_61 = 61,
      COLLECTIVE_LOC_62 = 62,
      COLLECTIVE_LOC_63 = 63,
      COLLECTIVE_LOC_64 = 64,
      COLLECTIVE_LOC_65 = 65,
      COLLECTIVE_LOC_66 = 66,
      COLLECTIVE_LOC_67 = 67,
      COLLECTIVE_LOC_68 = 68,
      COLLECTIVE_LOC_69 = 69,
      COLLECTIVE_LOC_70 = 70,
      COLLECTIVE_LOC_71 = 71,
      COLLECTIVE_LOC_72 = 72,
      COLLECTIVE_LOC_73 = 73,
      COLLECTIVE_LOC_74 = 74,
      COLLECTIVE_LOC_75 = 75,
      COLLECTIVE_LOC_76 = 76,
      COLLECTIVE_LOC_77 = 77,
      COLLECTIVE_LOC_78 = 78,
      COLLECTIVE_LOC_79 = 79,
      COLLECTIVE_LOC_80 = 80,
      COLLECTIVE_LOC_81 = 81,
      COLLECTIVE_LOC_82 = 82,
      COLLECTIVE_LOC_83 = 83,
      COLLECTIVE_LOC_84 = 84,
      COLLECTIVE_LOC_85 = 85,
      COLLECTIVE_LOC_86 = 86,
      COLLECTIVE_LOC_87 = 87,
      COLLECTIVE_LOC_88 = 88,
      COLLECTIVE_LOC_89 = 89,
      COLLECTIVE_LOC_90 = 90,
      COLLECTIVE_LOC_91 = 91,
      COLLECTIVE_LOC_92 = 92,
      COLLECTIVE_LOC_93 = 93,
      COLLECTIVE_LOC_94 = 94,
      COLLECTIVE_LOC_95 = 95,
      COLLECTIVE_LOC_96 = 96,
      COLLECTIVE_LOC_97 = 97,
      COLLECTIVE_LOC_98 = 98,
      COLLECTIVE_LOC_99 = 99,
      COLLECTIVE_LOC_100 = 100,
      COLLECTIVE_LOC_101 = 101,
      COLLECTIVE_LOC_102 = 102,
      COLLECTIVE_LOC_103 = 103,
      COLLECTIVE_LOC_104 = 104,
      COLLECTIVE_LOC_105 = 105,
      COLLECTIVE_LOC_106 = 106,
      COLLECTIVE_LOC_107 = 107,
      COLLECTIVE_LOC_108 = 108,
      COLLECTIVE_LOC_109 = 109,
      COLLECTIVE_LOC_110 = 110,
    };

#ifdef LEGION_DEBUG_COLLECTIVES
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

      template<bool EXCLUSIVE>
      static void apply(LHS& lhs, RHS rhs);
      template<bool EXCLUSIVE>
      static void fold(RHS& rhs1, RHS rhs2);
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
        CloseCheckValue(
            Operation* op, RtBarrier barrier, RegionTreeNode* node,
            bool read_only);
      public:
        bool operator==(const CloseCheckValue& rhs) const;
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

      template<bool EXCLUSIVE>
      static void apply(LHS& lhs, RHS rhs);
      template<bool EXCLUSIVE>
      static void fold(RHS& rhs1, RHS rhs2);
    };
#endif

    /**
     * \class ShardCollective
     * The shard collective is the base class for performing
     * collective operations between shards
     */
    class ShardCollective {
    public:
      struct DeferCollectiveArgs : public LgTaskArgs<DeferCollectiveArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_COLLECTIVE_TASK_ID;
      public:
        DeferCollectiveArgs(void) = default;
        DeferCollectiveArgs(ShardCollective* c)
          : LgTaskArgs(false, false), collective(c)
        { }
        void execute(void) const;
      public:
        ShardCollective* collective;
      };
    public:
      ShardCollective(CollectiveIndexLocation loc, ReplicateContext* ctx);
      ShardCollective(ReplicateContext* ctx, CollectiveID id);
      virtual ~ShardCollective(void);
    public:
      virtual void perform_collective_async(
          RtEvent precondition = RtEvent::NO_RT_EVENT) = 0;
      virtual RtEvent perform_collective_wait(bool block = false) = 0;
      virtual void handle_collective_message(Deserializer& derez) = 0;
      void perform_collective_sync(RtEvent pre = RtEvent::NO_RT_EVENT);
    protected:
      bool defer_collective_async(RtEvent precondition);
      int convert_to_index(ShardID id, ShardID origin) const;
      ShardID convert_to_shard(int index, ShardID origin) const;
    public:
      ShardManager* const manager;
      ReplicateContext* const context;
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
      BroadcastCollective(
          CollectiveIndexLocation loc, ReplicateContext* ctx, ShardID origin);
      BroadcastCollective(
          ReplicateContext* ctx, CollectiveID id, ShardID origin);
      virtual ~BroadcastCollective(void);
    public:
      virtual MessageKind get_message_kind(void) const = 0;
      // We guarantee that these methods will be called atomically
      virtual void pack_collective(Serializer& rez) const = 0;
      virtual void unpack_collective(Deserializer& derez) = 0;
    public:
      virtual void perform_collective_async(
          RtEvent pre = RtEvent::NO_RT_EVENT) override;
      virtual RtEvent perform_collective_wait(bool block = true) override;
      virtual void handle_collective_message(Deserializer& derez) override;
      virtual RtEvent post_broadcast(void) { return RtEvent::NO_RT_EVENT; }
      // Use this method in case we don't actually end up using the collective
      virtual void elide_collective(void);
    public:
      RtEvent get_done_event(void) const;
      inline bool is_origin(void) const { return (origin == local_shard); }
    protected:
      void send_messages(void) const;
    public:
      const ShardID origin;
      const int shard_collective_radix;
    private:
      RtUserEvent done_event;  // valid on all shards except origin
    };

    /**
     * \class GatherCollective
     * This shard collective has equivalent functionality to
     * MPI Gather in that it will ensure that data from all
     * the shards are reduced down to a single shard.
     */
    class GatherCollective : public ShardCollective {
    public:
      GatherCollective(
          CollectiveIndexLocation loc, ReplicateContext* ctx, ShardID target);
      GatherCollective(ReplicateContext* ctx, CollectiveID id, ShardID origin);
      virtual ~GatherCollective(void);
    public:
      virtual MessageKind get_message_kind(void) const = 0;
      // We guarantee that these methods will be called atomically
      virtual void pack_collective(Serializer& rez) const = 0;
      virtual void unpack_collective(Deserializer& derez) = 0;
    public:
      virtual void perform_collective_async(
          RtEvent pre = RtEvent::NO_RT_EVENT) override;
      // Make sure to call this in the destructor of anything not the target
      virtual RtEvent perform_collective_wait(bool block = true) override;
      virtual void handle_collective_message(Deserializer& derez) override;
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
      AllGatherCollective(CollectiveIndexLocation loc, ReplicateContext* ctx);
      AllGatherCollective(ReplicateContext* ctx, CollectiveID id);
      AllGatherCollective(
          ReplicateContext* ctx, CollectiveID id,
          const std::vector<ShardID>& participants);
      virtual ~AllGatherCollective(void);
    public:
      virtual MessageKind get_message_kind(void) const = 0;
      // We guarantee that these methods will be called atomically
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) = 0;
      virtual void unpack_collective_stage(Deserializer& derez, int stage) = 0;
    public:
      virtual void perform_collective_async(
          RtEvent pre = RtEvent::NO_RT_EVENT) override;
      virtual RtEvent perform_collective_wait(bool block = true) override;
      virtual void handle_collective_message(Deserializer& derez) override;
      // Use this method in case we don't actually end up using the collective
      virtual void elide_collective(void);
      inline RtEvent get_done_event(void) const { return done_event; }
    protected:
      void initialize_collective(void);
      void construct_message(ShardID target, int stage, Serializer& rez);
      bool initiate_collective(void);
      void send_remainder_stage(void);
      bool send_ready_stages(const int start_stage = 1);
      void unpack_stage(int stage, Deserializer& derez);
      void complete_exchange(void);
      virtual RtEvent post_complete_exchange(void)
      {
        return RtEvent::NO_RT_EVENT;
      }
    protected:
      const std::vector<ShardID>* const participants;  // can be nullptr
      const size_t total_shards;
      int local_index;
      int shard_collective_radix;
      int shard_collective_log_radix;
      int shard_collective_stages;
      int shard_collective_participating_shards;
      int shard_collective_last_radix;
      bool participating;
    private:
      RtUserEvent done_event;
      std::vector<int> stage_notifications;
      std::vector<bool> sent_stages;
      std::map<int, std::vector<std::pair<void*, size_t> > >* reorder_stages;
      // Handle a small race on deciding who gets to
      // trigger the done event, only the last one of these
      // will get to do the trigger to avoid any races
      unsigned pending_send_ready_stages;
      bool done_triggered;
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
      AllReduceCollective(CollectiveIndexLocation loc, ReplicateContext* ctx);
      AllReduceCollective(ReplicateContext* ctx, CollectiveID id);
      virtual ~AllReduceCollective(void);
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_VALUE_ALLREDUCE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      void async_all_reduce(typename REDOP::RHS value);
      RtEvent wait_all_reduce(bool block = true);
      typename REDOP::RHS sync_all_reduce(typename REDOP::RHS value);
      typename REDOP::RHS get_result(void);
    protected:
      typename REDOP::RHS value;
    };

    /**
     * \class BufferBroadcast
     * Broadcast out a binary buffer out to all the shards
     */
    class BufferBroadcast : public BroadcastCollective {
    public:
      BufferBroadcast(ReplicateContext* ctx, CollectiveIndexLocation loc);
      BufferBroadcast(
          ReplicateContext* ctx, ShardID origin, CollectiveIndexLocation loc)
        : BroadcastCollective(loc, ctx, origin), buffer(nullptr), size(0),
          own(false)
      { }
      BufferBroadcast(CollectiveID id, ReplicateContext* ctx);
      BufferBroadcast(CollectiveID id, ShardID origin, ReplicateContext* ctx)
        : BroadcastCollective(ctx, id, origin), buffer(nullptr), size(0),
          own(false)
      { }
      BufferBroadcast(const BufferBroadcast& rhs) = delete;
      virtual ~BufferBroadcast(void)
      {
        if (own)
          free(buffer);
      }
    public:
      BufferBroadcast& operator=(const BufferBroadcast& rhs) = delete;
      void broadcast(void* buffer, size_t size, bool copy = true);
      const void* get_buffer(size_t& size, bool wait = true);
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_BUFFER_BROADCAST;
      }
      virtual void pack_collective(Serializer& rez) const override;
      virtual void unpack_collective(Deserializer& derez) override;
    protected:
      void* buffer;
      size_t size;
      bool own;
    };

    /**
     * \class BufferExchange
     * A class for doing an all-to-all exchange of byte buffers
     */
    class BufferExchange : public AllGatherCollective<false> {
    public:
      BufferExchange(ReplicateContext* ctx, CollectiveIndexLocation loc);
      BufferExchange(const BufferExchange& rhs) = delete;
      virtual ~BufferExchange(void);
    public:
      BufferExchange& operator=(const BufferExchange& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_BUFFER_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      const std::map<ShardID, std::pair<void*, size_t> >& exchange_buffers(
          void* value, size_t size, bool keep_self = false);
      RtEvent exchange_buffers_async(
          void* value, size_t size, bool keep_self = false);
      const std::map<ShardID, std::pair<void*, size_t> >& sync_buffers(
          bool keep);
    protected:
      std::map<ShardID, std::pair<void*, size_t> > results;
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
        FutureInstance* instance;
        ApEvent precondition;
        ApUserEvent postcondition;
      };
    public:
      FutureAllReduceCollective(
          Operation* op, CollectiveIndexLocation loc, ReplicateContext* ctx,
          ReductionOpID redop_id, const ReductionOp* redop);
      FutureAllReduceCollective(
          Operation* op, ReplicateContext* ctx, CollectiveID id,
          ReductionOpID redop_id, const ReductionOp* redop);
      virtual ~FutureAllReduceCollective(void);
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_FUTURE_ALLREDUCE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
      virtual RtEvent post_complete_exchange(void) override;
      virtual void elide_collective(void) override;
    public:
      void set_shadow_instance(FutureInstance* shadow);
      RtEvent async_reduce(FutureInstance* instance, ApEvent& ready_event);
    protected:
      ApEvent perform_reductions(
          const std::map<ShardID, PendingReduction>& red);
      void create_shadow_instance(void);
    public:
      Operation* const op;
      const ReductionOp* const redop;
      const ReductionOpID redop_id;
    protected:
      const ApUserEvent finished;
      std::map<int, std::map<ShardID, PendingReduction> > pending_reductions;
      FutureInstance* instance;
      FutureInstance* shadow_instance;
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
      FutureBroadcastCollective(
          ReplicateContext* ctx, CollectiveIndexLocation loc, ShardID origin,
          Operation* op);
      FutureBroadcastCollective(const FutureBroadcastCollective& rhs) = delete;
      virtual ~FutureBroadcastCollective(void);
    public:
      FutureBroadcastCollective& operator=(
          const FutureBroadcastCollective& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_FUTURE_ALLREDUCE;
      }
      virtual void pack_collective(Serializer& rez) const override;
      virtual void unpack_collective(Deserializer& derez) override;
      virtual void elide_collective(void) override;
      virtual RtEvent post_broadcast(void) override;
    public:
      RtEvent async_broadcast(
          FutureInstance* instance, ApEvent precondition = ApEvent::NO_AP_EVENT,
          RtEvent postcondition = RtEvent::NO_RT_EVENT);
    public:
      Operation* const op;
      const ApUserEvent finished;
    protected:
      FutureInstance* instance;
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
      FutureReductionCollective(
          ReplicateContext* ctx, CollectiveIndexLocation loc, ShardID origin,
          Operation* op, FutureBroadcastCollective* broadcast,
          const ReductionOp* redop, ReductionOpID redop_id);
      FutureReductionCollective(const FutureReductionCollective& rhs) = delete;
      virtual ~FutureReductionCollective(void);
    public:
      FutureReductionCollective& operator=(
          const FutureReductionCollective& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_FUTURE_REDUCTION;
      }
      virtual void pack_collective(Serializer& rez) const override;
      virtual void unpack_collective(Deserializer& derez) override;
      virtual RtEvent post_gather(void) override;
    public:
      void async_reduce(FutureInstance* instance, ApEvent precondition);
    protected:
      void perform_reductions(void) const;
    public:
      Operation* const op;
      FutureBroadcastCollective* const broadcast;
      const ReductionOp* const redop;
      const ReductionOpID redop_id;
    protected:
      FutureInstance* instance;
      mutable ApEvent ready;
      std::map<ShardID, std::pair<FutureInstance*, ApEvent> >
          pending_reductions;
    };

    /**
     * \class ShardParticipants
     * Find the shard participants in a replicated context
     */
    class ShardParticipantsExchange : public AllGatherCollective<false> {
    public:
      ShardParticipantsExchange(
          ReplicateContext* ctx, CollectiveIndexLocation loc);
      ShardParticipantsExchange(const ShardParticipantsExchange& rhs) = delete;
      virtual ~ShardParticipantsExchange(void);
    public:
      ShardParticipantsExchange& operator=(
          const ShardParticipantsExchange& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_SHARD_PARTICIPANTS_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      void exchange(bool participating);
      bool find_shard_participants(std::vector<ShardID>& shards);
    protected:
      std::set<ShardID> participants;
    };

    /**
     * \class InterferingPointExchange
     *
     */
    template<typename T>
    class InterferingPointExchange : public AllGatherCollective<true> {
    public:
      InterferingPointExchange(ReplicateContext* ctx, CollectiveID id, T* op);
      InterferingPointExchange(const InterferingPointExchange& rhs) = delete;
      virtual ~InterferingPointExchange(void);
    public:
      InterferingPointExchange& operator=(const InterferingPointExchange& rhs) =
          delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_INTERFERING_POINT_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
      virtual RtEvent post_complete_exchange(void) override;
    public:
      void exchange_domain_points(
          std::map<unsigned, std::vector<std::pair<DomainPoint, Domain> > >&
              points);
    public:
      T* const op;
    protected:
      std::map<unsigned, std::vector<std::pair<DomainPoint, Domain> > >
          domain_points;
    };

    /**
     * \class ShardingGatherCollective
     * A class for gathering all the names of the ShardingIDs chosen
     * by different mappers to confirm that they are all the same.
     * This is primarily only used in debug mode.
     */
    class ShardingGatherCollective : public GatherCollective {
    public:
      ShardingGatherCollective(
          ReplicateContext* ctx, ShardID target, CollectiveIndexLocation loc);
      ShardingGatherCollective(const ShardingGatherCollective& rhs) = delete;
      virtual ~ShardingGatherCollective(void);
    public:
      ShardingGatherCollective& operator=(const ShardingGatherCollective& rhs) =
          delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_SHARDING_GATHER_COLLECTIVE;
      }
      virtual void pack_collective(Serializer& rez) const override;
      virtual void unpack_collective(Deserializer& derez) override;
    public:
      void contribute(ShardingID value);
      bool validate(ShardingID value);
    protected:
      std::map<ShardID, ShardingID> results;
    };

    /**
     * \class ValueBroadcast
     * This will broadcast a value of any type that can be
     * trivially serialized to all the shards.
     */
    template<typename T>
    class ValueBroadcast : public BroadcastCollective {
    public:
      // ValueBroadcast(ReplicateContext *ctx, CollectiveIndexLocation loc)
      //   : BroadcastCollective(loc, ctx, ctx->owner_shard->shard_id) { }
      ValueBroadcast(
          ReplicateContext* ctx, ShardID origin, CollectiveIndexLocation loc)
        : BroadcastCollective(loc, ctx, origin)
      { }
      ValueBroadcast(CollectiveID id, ReplicateContext* ctx, ShardID origin)
        : BroadcastCollective(ctx, id, origin)
      { }
      ValueBroadcast(const ValueBroadcast& rhs) = delete;
      virtual ~ValueBroadcast(void) { }
    public:
      ValueBroadcast& operator=(const ValueBroadcast& rhs) = delete;
      inline void broadcast(const T& v)
      {
        value = v;
        perform_collective_async();
      }
      inline T get_value(bool wait = true)
      {
        if (wait)
          perform_collective_wait();
        return value;
      }
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_VALUE_BROADCAST;
      }
      virtual void pack_collective(Serializer& rez) const override
      {
        rez.serialize(value);
      }
      virtual void unpack_collective(Deserializer& derez) override
      {
        derez.deserialize(value);
      }
    protected:
      T value;
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
      SlowBarrier(ReplicateContext* ctx, CollectiveID id)
        : AllGatherCollective<false>(ctx, id)
      { }
      SlowBarrier(const SlowBarrier& rhs) = delete;
      virtual ~SlowBarrier(void) { }
    public:
      SlowBarrier& operator=(const SlowBarrier& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_SLOW_BARRIER;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override
      { }
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override
      { }
    };

    /**
     * \class CollectiveMapping
     * A collective mapping is an ordering of unique address spaces
     * and can be used to construct broadcast and reduction trees.
     * This is especialy useful for collective instances and for
     * parts of control replication.
     */
    class CollectiveMapping : public Collectable {
    public:
      CollectiveMapping(
          const std::vector<AddressSpaceID>& spaces, size_t radix);
      CollectiveMapping(const ShardMapping& shard_mapping, size_t radix);
      CollectiveMapping(Deserializer& derez, size_t total_spaces);
      CollectiveMapping(const CollectiveMapping& rhs);
    public:
      inline AddressSpaceID operator[](unsigned idx) const
      {
        legion_assert(idx < size());
        return unique_sorted_spaces.get_index(idx);
      }
      inline unsigned find_index(const AddressSpaceID space) const
      {
        return unique_sorted_spaces.find_index(space);
      }
      inline size_t size(void) const { return total_spaces; }
      inline AddressSpaceID get_origin(void) const
      {
        legion_assert(size() > 0);
        return unique_sorted_spaces.find_first_set();
      }
      bool operator==(const CollectiveMapping& rhs) const;
      bool operator!=(const CollectiveMapping& rhs) const;
    public:
      AddressSpaceID get_parent(
          const AddressSpaceID origin, const AddressSpaceID local) const;
      size_t count_children(
          const AddressSpaceID origin, const AddressSpaceID local) const;
      void get_children(
          const AddressSpaceID origin, const AddressSpaceID local,
          std::vector<AddressSpaceID>& children) const;
      AddressSpaceID find_nearest(AddressSpaceID start) const;
      inline bool contains(const AddressSpaceID space) const
      {
        return unique_sorted_spaces.contains(space);
      }
      bool contains(const CollectiveMapping& rhs) const;
      CollectiveMapping* clone_with(AddressSpace space) const;
      void pack(Serializer& rez) const;
      static void pack_null(Serializer& rez);
    protected:
      unsigned convert_to_offset(unsigned index, unsigned origin) const;
      unsigned convert_to_index(unsigned offset, unsigned origin) const;
    protected:
      NodeSet<TODO_LIFETIME> unique_sorted_spaces;
      size_t total_spaces;
      size_t radix;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_COLLECTIVES_H__
