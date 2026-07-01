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

#ifndef __LEGION_TOPLEVEL_CONTEXT_H__
#define __LEGION_TOPLEVEL_CONTEXT_H__

#include "legion/contexts/inner.h"

namespace Legion {
  namespace Internal {

    /**
     * \class TopLevelContext
     * This is the top-level task context that
     * exists at the root of a task tree. In
     * general there will only be one of these
     * per application unless mappers decide to
     * create their own tasks for performing
     * computation.
     */
    class TopLevelContext
      : public HeapifyMixin<TopLevelContext, InnerContext, CONTEXT_LIFETIME> {
    public:
      TopLevelContext(
          Processor executing, coord_t normal_id, coord_t implicit_id,
          DistributedID id = 0, CollectiveMapping* mapping = nullptr);
      TopLevelContext(const TopLevelContext& rhs) = delete;
      virtual ~TopLevelContext(void);
    public:
      TopLevelContext& operator=(const TopLevelContext& rhs) = delete;
    public:
      static Mapper::ContextConfigOutput configure_toplevel_context(void);
    public:
      virtual void pack_remote_context(
          Serializer& rez, AddressSpaceID target,
          bool replicate = false) override;
      virtual InnerContext* find_parent_context(void) override;
      virtual UniqueID get_unique_id(void) const override { return root_uid; }
    public:
      virtual InnerContext* find_top_context(
          InnerContext* previous = nullptr) override;
    public:
      virtual void receive_created_region_contexts(
          const std::vector<RegionNode*>& created_regions,
          const std::vector<EqKDTree*>& created_trees,
          std::set<RtEvent>& applied_events, const ShardMapping* mapping,
          ShardID source_shard) override;
      virtual RtEvent compute_equivalence_sets(
          unsigned req_index, const std::vector<EqSetTracker*>& targets,
          const std::vector<AddressSpaceID>& target_spaces,
          AddressSpaceID creation_target_space, IndexSpaceExpression* expr,
          const FieldMask& mask) override;
      virtual RtEvent record_output_equivalence_set(
          EqSetTracker* source, AddressSpaceID source_space, unsigned req_index,
          EquivalenceSet* set, const FieldMask& mask) override;
    public:
      const UniqueID root_uid;
    protected:
      std::vector<RegionRequirement> dummy_requirements;
      std::vector<OutputRequirement> dummy_output_requirements;
      std::vector<unsigned> dummy_indexes;
      std::vector<bool> dummy_mapped;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_TOPLEVEL_CONTEXT_H__
