/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_FUTURE_MAP_IMPL_H__
#define __LEGION_FUTURE_MAP_IMPL_H__

#include "legion/api/future_map.h"
#include "legion/kernel/garbage_collection.h"
#include "legion/operations/operation.h"
#include "legion/utilities/collectives.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /**
     * \class FutureMapImpl
     * The base implementation of a future map object. Note
     * that this is now a distributed collectable object too
     * that can be used to find the name of a future for a
     * given point anywhere in the machine.
     */
    class FutureMapImpl : public DistributedCollectable,
                          public Heapify<FutureMapImpl, SHORT_LIFETIME> {
    public:
      FutureMapImpl(
          TaskContext* ctx, Operation* op, IndexSpaceNode* domain,
          DistributedID did, Provenance* provenance, bool register_now = true,
          CollectiveMapping* mapping = nullptr);
      FutureMapImpl(
          TaskContext* ctx, IndexSpaceNode* domain, DistributedID did,
          uint64_t blocking_index, const std::optional<uint64_t>& context_index,
          Provenance* provenance, bool register_now = true,
          CollectiveMapping* mapping = nullptr);  // remote
      FutureMapImpl(
          TaskContext* ctx, Operation* op, uint64_t blocking_index,
          GenerationID gen, int depth, UniqueID uid, IndexSpaceNode* domain,
          DistributedID did, Provenance* provenance);
      FutureMapImpl(const FutureMapImpl& rhs) = delete;
      virtual ~FutureMapImpl(void);
    public:
      FutureMapImpl& operator=(const FutureMapImpl& rhs) = delete;
    public:
      virtual bool is_replicate_future_map(void) const { return false; }
    public:
      virtual void notify_local(void) override;
    public:
      Domain get_domain(void) const;
      std::optional<uint64_t> get_context_index(void) const;
      virtual Future get_future(
          const DomainPoint& point, bool internal_only,
          RtEvent* wait_on = nullptr);
      void set_future(const DomainPoint& point, FutureImpl* impl);
      void get_void_result(
          const DomainPoint& point, bool silence_warnings = true,
          const char* warning_string = nullptr);
      virtual void wait_all_results(
          bool silence_warnings = true, const char* warning_string = nullptr);
      bool reset_all_futures(void);
    public:
      void pack_future_map(Serializer& rez, AddressSpaceID target);
      static FutureMap unpack_future_map(
          Deserializer& derez, AddressSpaceID source, TaskContext* ctx);
    public:
      virtual void get_all_futures(std::map<DomainPoint, FutureImpl*>& futures);
      void set_all_futures(const std::map<DomainPoint, Future>& futures);
    public:
      virtual FutureImpl* find_local_future(const DomainPoint& point);
      virtual void get_shard_local_futures(
          ShardID shard, std::map<DomainPoint, FutureImpl*>& futures);
    public:
      void register_dependence(Operation* consumer_op);
      virtual RtEvent find_pointwise_dependence(
          const DomainPoint& point, int depth,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT);
      void process_future_response(Deserializer& derez);
    public:
      void record_future_map_registered(void);
    public:
      TaskContext* const context;
      // Either an index space task or a must epoch op
      Operation* const op;
      const GenerationID op_gen;
      const int op_depth;
      const UniqueID op_uid;
      const uint64_t blocking_index;
      Provenance* const provenance;
      IndexSpaceNode* const future_map_domain;
    private:
      // This field is only set on remote nodes that are not the owner
      // of the future map, invoke get_context_index to get the
      // right context index for the operation that produced this
      const std::optional<uint64_t> remote_context_index;
    protected:
      mutable LocalLock future_map_lock;
      std::map<DomainPoint, FutureImpl*> futures;
    };

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const FutureMapImpl& fm)
    //--------------------------------------------------------------------------
    {
      if (fm.provenance != nullptr)
        os << "future map from " << fm.provenance->human;
      else if (fm.op != nullptr)
        os << "future map from " << fm.op->get_logging_name()
           << " (UID: " << fm.op_uid << ")";
      else if (fm.op_uid > 0)
        os << "future map from unknown op (UID: " << fm.op_uid << ")";
      else
        os << "external future map";
      return os;
    }

    /**
     * \class TransformFutureMapImpl
     * This class is a wrapper around a future map implementation that
     * will transform the points being accessed on to a previous future map
     */
    class TransformFutureMapImpl : public FutureMapImpl {
    public:
      TransformFutureMapImpl(
          FutureMapImpl* previous, IndexSpaceNode* domain,
          PointTransformFunctor* functor, bool own_functor,
          Provenance* provenance);
      TransformFutureMapImpl(const TransformFutureMapImpl& rhs) = delete;
      virtual ~TransformFutureMapImpl(void);
    public:
      TransformFutureMapImpl& operator=(const TransformFutureMapImpl& rhs) =
          delete;
    public:
      virtual bool is_replicate_future_map(void) const override;
      virtual Future get_future(
          const DomainPoint& point, bool internal_only,
          RtEvent* wait_on = nullptr) override;
      virtual void get_all_futures(
          std::map<DomainPoint, FutureImpl*>& futures) override;
      virtual void wait_all_results(
          bool silence_warnings = true,
          const char* warning_string = nullptr) override;
    public:
      virtual FutureImpl* find_local_future(const DomainPoint& point) override;
      virtual void get_shard_local_futures(
          ShardID shard, std::map<DomainPoint, FutureImpl*>& futures) override;
      virtual RtEvent find_pointwise_dependence(
          const DomainPoint& point, int depth,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT) override;
    public:
      FutureMapImpl* const previous;
      PointTransformFunctor* const functor;
      const bool own_functor;
    };

    /**
     * \class ReplFutureMapImpl
     * This a special kind of future map that is created
     * in control replication contexts
     */
    class ReplFutureMapImpl : public FutureMapImpl {
    public:
      ReplFutureMapImpl(
          TaskContext* ctx, ShardManager* man, Operation* op,
          IndexSpaceNode* domain, IndexSpaceNode* shard_domain,
          DistributedID did, Provenance* provenance,
          CollectiveMapping* collective_mapping);
      ReplFutureMapImpl(
          TaskContext* ctx, ShardManager* man, IndexSpaceNode* domain,
          IndexSpaceNode* shard_domain, DistributedID did, uint64_t index,
          std::optional<uint64_t> ctx_idx, Provenance* provenance,
          CollectiveMapping* collective_mapping);
      ReplFutureMapImpl(const ReplFutureMapImpl& rhs) = delete;
      virtual ~ReplFutureMapImpl(void);
    public:
      ReplFutureMapImpl& operator=(const ReplFutureMapImpl& rhs) = delete;
    public:
      virtual bool is_replicate_future_map(void) const override { return true; }
    public:
      virtual Future get_future(
          const DomainPoint& point, bool internal,
          RtEvent* wait_on = nullptr) override;
      virtual void get_all_futures(
          std::map<DomainPoint, FutureImpl*>& futures) override;
    public:
      // Will return nullptr if it does not exist
      virtual void get_shard_local_futures(
          ShardID shard, std::map<DomainPoint, FutureImpl*>& futures) override;
      virtual RtEvent find_pointwise_dependence(
          const DomainPoint& point, int depth,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT) override;
    public:
      bool set_sharding_function(ShardingFunction* function, bool own = false);
      RtEvent get_sharding_function_ready(void);
    public:
      ShardManager* const shard_manager;
      IndexSpaceNode* const shard_domain;
      // Unlike normal future maps, we know these only ever exist on the
      // node where they are made so we store their producer op information
      // in case they have to make futures from remote shards
      const int op_depth;
    protected:
      RtUserEvent sharding_function_ready;
      std::atomic<ShardingFunction*> sharding_function;
      // Whether the future map owns the sharding function
      bool own_sharding_function;
      bool collective_performed;
    };

    /**
     * \class FutureNameExchange
     * A class for doing an all-to-all exchange of future names
     */
    class FutureNameExchange : public AllGatherCollective<false> {
    public:
      FutureNameExchange(ReplicateContext* ctx, CollectiveIndexLocation loc);
      FutureNameExchange(const FutureNameExchange& rhs) = delete;
      virtual ~FutureNameExchange(void);
    public:
      FutureNameExchange& operator=(const FutureNameExchange& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_FUTURE_NAME_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      void exchange_future_names(std::map<DomainPoint, FutureImpl*>& futures);
    protected:
      std::map<DomainPoint, Future> results;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_FUTURE_MAP_IMPL_H__
