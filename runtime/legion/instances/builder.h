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

#ifndef __LEGION_INSTANCE_BUILDER_H__
#define __LEGION_INSTANCE_BUILDER_H__

#include "legion/api/constraints.h"
#include "legion/tools/profiler.h"

namespace Legion {
  namespace Internal {

    /**
     * \class InstanceBuilder
     * A helper for building physical instances of logical regions
     */
    class InstanceBuilder : public ProfilingResponseHandler {
    public:
      InstanceBuilder(
          const std::vector<LogicalRegion>& regs,
          const LayoutConstraintSet& cons, MemoryManager* memory = nullptr,
          UniqueID cid = 0)
        : regions(regs), constraints(cons), memory_manager(memory),
          creator_id(cid), instance(PhysicalInstance::NO_INST),
          field_space_node(nullptr), instance_domain(nullptr), tree_id(0),
          redop_id(0), reduction_op(nullptr), realm_layout(nullptr),
          piece_list(nullptr), piece_list_size(0), valid(false),
          allocated(false)
      { }
      virtual ~InstanceBuilder(void);
    public:
      [[nodiscard]] bool initialize(void);
      PhysicalManager* create_physical_instance(
          LayoutConstraintKind* unsat_kind, unsigned* unsat_index,
          size_t* footprint = nullptr,
          RtEvent collection_done = RtEvent::NO_RT_EVENT,
          PhysicalInstance hole = PhysicalInstance::NO_INST,
          LgEvent hole_unique_event = LgEvent::NO_LG_EVENT);
    public:
      virtual bool handle_profiling_response(
          const Realm::ProfilingResponse& response, const void* orig,
          size_t orig_length, LgEvent& fevent, bool& failed_alloc) override;
    protected:
      bool compute_space_and_domain(void);
    protected:
      void compute_layout_parameters(void);
    protected:
      const std::vector<LogicalRegion>& regions;
      LayoutConstraintSet constraints;
      MemoryManager* const memory_manager;
      const UniqueID creator_id;
    protected:
      PhysicalInstance instance;
      RtUserEvent profiling_ready;
    protected:
      FieldSpaceNode* field_space_node;
      IndexSpaceExpression* instance_domain;
      RegionTreeID tree_id;
      // Mapping from logical field order to layout order
      std::vector<unsigned> mask_index_map;
      std::vector<size_t> field_sizes;
      std::vector<CustomSerdezID> serdez;
      FieldMask instance_mask;
      ReductionOpID redop_id;
      const ReductionOp* reduction_op;
      Realm::InstanceLayoutGeneric* realm_layout;
      void* piece_list;
      size_t piece_list_size;
    public:
      LgEvent caller_fevent;
      LgEvent current_unique_event;
      bool valid;
      bool allocated;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_INSTANCE_BUILDER_H__
