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

#ifndef __LEGION_REMOTE_CONTEXT_H__
#define __LEGION_REMOTE_CONTEXT_H__

#include "legion/contexts/inner.h"

namespace Legion {
  namespace Internal {

    /**
     * \class RemoteTask
     * A small helper class for giving application
     * visibility to this remote context
     */
    class RemoteTask : public ExternalTask {
    public:
      RemoteTask(RemoteContext* owner);
      RemoteTask(const RemoteTask& rhs) = delete;
      virtual ~RemoteTask(void);
    public:
      RemoteTask& operator=(const RemoteTask& rhs) = delete;
    public:
      virtual int get_depth(void) const override;
      virtual UniqueID get_unique_id(void) const override;
      virtual Domain get_slice_domain(void) const override;
      virtual uint64_t get_context_index(void) const override;
      virtual void set_context_index(uint64_t index) override;
      virtual bool has_parent_task(void) const override;
      virtual const Task* get_parent_task(void) const override;
      virtual const char* get_task_name(void) const override;
      virtual ShardID get_shard_id(void) const override;
      virtual size_t get_total_shards(void) const override;
      virtual DomainPoint get_shard_point(void) const override;
      virtual Domain get_shard_domain(void) const override;
      virtual bool has_trace(void) const;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const override;
    public:
      RemoteContext* const owner;
      uint64_t context_index;
    };

    /**
     * \class RemoteContext
     * A remote copy of a TaskContext for the
     * execution of sub-tasks on remote notes.
     */
    class RemoteContext
      : public HeapifyMixin<RemoteContext, InnerContext, CONTEXT_LIFETIME> {
    public:
      RemoteContext(DistributedID did, CollectiveMapping* mapping = nullptr);
      RemoteContext(const RemoteContext& rhs) = delete;
      virtual ~RemoteContext(void);
    public:
      RemoteContext& operator=(const RemoteContext& rhs) = delete;
    public:
      static Mapper::ContextConfigOutput configure_remote_context(void);
    public:
      virtual const Task* get_task(void) const override;
      virtual UniqueID get_unique_id(void) const override;
      virtual ShardID get_shard_id(void) const override { return shard_id; }
      virtual DistributedID get_replication_id(void) const override
      {
        return repl_id;
      }
      virtual size_t get_total_shards(void) const override
      {
        return total_shards;
      }
      void unpack_remote_context(Deserializer& derez);
      virtual InnerContext* find_parent_context(void) override;
    public:
      virtual InnerContext* find_top_context(
          InnerContext* previous = nullptr) override;
    public:
      virtual RtEvent compute_equivalence_sets(
          unsigned req_index, const std::vector<EqSetTracker*>& targets,
          const std::vector<AddressSpaceID>& target_spaces,
          AddressSpaceID creation_target_space, IndexSpaceExpression* expr,
          const FieldMask& mask) override;
      virtual RtEvent record_output_equivalence_set(
          EqSetTracker* source, AddressSpaceID source_space, unsigned req_index,
          EquivalenceSet* set, const FieldMask& mask) override;
      virtual InnerContext* find_parent_physical_context(
          unsigned index) override;
      virtual void pack_inner_context(Serializer& rez) const override;
      virtual CollectiveResult* find_or_create_collective_view(
          RegionTreeID tid, const std::vector<DistributedID>& instances,
          RtEvent& ready) override;
      virtual void refine_equivalence_sets(
          unsigned req_index, IndexSpaceNode* node,
          const FieldMask& refinement_mask,
          std::vector<RtEvent>& applied_events, bool sharded = false,
          bool first = true,
          const CollectiveMapping* mapping = nullptr) override;
      virtual RtEvent find_pointwise_dependence(
          uint64_t context_index, const DomainPoint& point, ShardID shard,
          bool intra_space,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT,
          std::optional<unsigned> output_index =
              std::optional<unsigned>()) override;
      virtual void find_trace_local_sets(
          unsigned req_index, const FieldMask& mask,
          std::map<EquivalenceSet*, unsigned>& current_sets,
          IndexSpaceNode* node = nullptr,
          const CollectiveMapping* mapping = nullptr) override;
      virtual void invalidate_logical_context(void) override;
      virtual void invalidate_region_tree_contexts(
          const bool is_top_level_task, std::set<RtEvent>& applied,
          const ShardMapping* shard_mapping = nullptr,
          ShardID source_shard = 0) override;
      virtual void receive_created_region_contexts(
          const std::vector<RegionNode*>& created_regions,
          const std::vector<EqKDTree*>& created_trees,
          std::set<RtEvent>& applied_events, const ShardMapping* mapping,
          ShardID source_shard) override;
    public:
      const Task* get_parent_task(void);
      inline Provenance* get_provenance(void) { return provenance; }
    public:
      void unpack_local_field_update(Deserializer& derez);
      void set_physical_context_result(unsigned index, InnerContext* result);
    protected:
      DistributedID parent_context_did;
      std::atomic<InnerContext*> parent_ctx;
      ShardManager* shard_manager;  // if we're lucky and one is already here
      Provenance* provenance;
    protected:
      bool top_level_context;
      RemoteTask remote_task;
      UniqueID remote_uid;
    protected:
      std::vector<unsigned> local_parent_req_indexes;
      std::vector<bool> local_virtual_mapped;
    protected:
      // Cached physical contexts recorded from the owner
      mutable LocalLock remote_lock;
      std::map<unsigned /*index*/, InnerContext*> physical_contexts;
      std::map<unsigned, RtEvent> pending_physical_contexts;
    protected:
      // For remote replicate contexts
      friend class RemoteTask;
      ShardID shard_id;
      size_t total_shards;
      DomainPoint shard_point;
      Domain shard_domain;
      DistributedID repl_id;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_REMOTE_CONTEXT_H__
