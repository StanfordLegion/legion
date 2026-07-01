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

#ifndef __LEGION_PREDICATE_IMPL_H__
#define __LEGION_PREDICATE_IMPL_H__

#include "legion/api/predicate.h"
#include "legion/api/redop.h"
#include "legion/kernel/garbage_collection.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    /**
     * \class PredicateImpl
     * This class provides the base support for a predicate and
     * any state needed to manage the mapping of things that
     * depend on a predicate value
     */
    class PredicateImpl : public Collectable {
    public:
      PredicateImpl(Operation* creator);
      PredicateImpl(const PredicateImpl& rhs) = delete;
      virtual ~PredicateImpl(void);
    public:
      PredicateImpl& operator=(const PredicateImpl& rhs) = delete;
    public:
      // This returns the predicate value if it is set or returns the
      // names of the guards to use if has not been set
      virtual bool get_predicate(
          uint64_t context_index, PredEvent& true_guard,
          PredEvent& false_guard);
      bool get_predicate(RtEvent& ready);
      virtual void set_predicate(bool value);
    public:
      InnerContext* const context;
      Operation* const creator;
      const GenerationID creator_gen;
      const UniqueID creator_uid;
    protected:
      mutable LocalLock predicate_lock;
      PredUserEvent true_guard, false_guard;
      RtUserEvent ready_event;
      int value;  // <0 is unset, 0 is false, >0 is true
    };

    /**
     * \class PredicateCollective
     * A class for performing all-reduce of the maximum observed indexes
     * for a replicated predicate impl
     */
    class PredicateCollective
      : public AllReduceCollective<MaxReduction<uint64_t>, false> {
    public:
      PredicateCollective(
          ReplPredicateImpl* predicate, ReplicateContext* ctx, CollectiveID id);
      PredicateCollective(const PredicateCollective& rhs) = delete;
      virtual ~PredicateCollective(void) { }
    public:
      PredicateCollective& operator=(const PredicateCollective& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_PREDICATE_EXCHANGE;
      }
      virtual RtEvent post_complete_exchange(void) override;
    public:
      ReplPredicateImpl* const predicate;
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
      ReplPredicateImpl(
          Operation* creator, uint64_t coordinate, CollectiveID id);
      ReplPredicateImpl(const ReplPredicateImpl& rhs) = delete;
      virtual ~ReplPredicateImpl(void);
    public:
      ReplPredicateImpl& operator=(const ReplPredicateImpl& rhs) = delete;
    public:
      virtual bool get_predicate(
          uint64_t context_index, PredEvent& true_guard,
          PredEvent& false_guard) override;
      virtual void set_predicate(bool value) override;
    public:
      const uint64_t predicate_coordinate;
    protected:
      const CollectiveID collective_id;
      size_t max_observed_index;
      PredicateCollective* collective;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_PREDICATE_IMPL_H__
