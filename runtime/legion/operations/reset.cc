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

#include "legion/operations/reset.h"
#include "legion/contexts/replicate.h"
#include "legion/nodes/region.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Reset Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ResetOp::ResetOp(void) : Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ResetOp::~ResetOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ResetOp::initialize(
        InnerContext* ctx, LogicalRegion parent, LogicalRegion region,
        const std::set<FieldID>& fids)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx);
      requirement = RegionRequirement(
          region, LEGION_READ_WRITE, LEGION_EXCLUSIVE, parent);
      requirement.privilege_fields = fids;
      if (runtime->safe_model)
        verify_requirement(requirement);
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      LegionSpy::log_reset_operation(parent_ctx->get_unique_id(), unique_op_id);
      LegionSpy::log_logical_requirement(
          unique_op_id, 0 /*idx*/, true /*region*/,
          requirement.region.index_space.get_id(),
          requirement.region.field_space.get_id(),
          requirement.region.get_tree_id(), requirement.privilege,
          requirement.prop, requirement.redop,
          requirement.parent.index_space.get_id());
      LegionSpy::log_requirement_fields(unique_op_id, 0 /*idx*/, fids);
    }

    //--------------------------------------------------------------------------
    void ResetOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
      parent_req_index = 0;
    }

    //--------------------------------------------------------------------------
    void ResetOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      requirement.privilege_fields.clear();
      Operation::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* ResetOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[RESET_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind ResetOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return RESET_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t ResetOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    void ResetOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(requirement.handle_type == LEGION_SINGULAR_PROJECTION);
      analyze_region_requirements();
    }

    //--------------------------------------------------------------------------
    void ResetOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      std::vector<RtEvent> map_applied_conditions;
      RegionNode* node = runtime->get_node(requirement.region);
      FieldMask refinement_mask =
          node->column_source->get_field_mask(requirement.privilege_fields);
      parent_ctx->refine_equivalence_sets(
          parent_req_index, node->row_source, refinement_mask,
          map_applied_conditions);
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      complete_execution();
    }

    //--------------------------------------------------------------------------
    unsigned ResetOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      legion_assert(parent_req_index != TRACED_PARENT_INDEX);
      return parent_req_index;
    }

    /////////////////////////////////////////////////////////////
    // Repl Reset Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplResetOp::ReplResetOp(void) : ResetOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplResetOp::~ReplResetOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplResetOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ResetOp::activate();
      reset_barrier = RtBarrier::NO_RT_BARRIER;
    }

    //--------------------------------------------------------------------------
    void ReplResetOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      ResetOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplResetOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      reset_barrier = repl_ctx->get_next_collective_map_barriers();
      ResetOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void ReplResetOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      runtime->phase_barrier_arrive(reset_barrier, 1 /*count*/);
      const RtEvent precondition = reset_barrier;
      Runtime::advance_barrier(reset_barrier);
      enqueue_ready_operation(precondition);
    }

    //--------------------------------------------------------------------------
    void ReplResetOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      std::vector<RtEvent> map_applied_conditions;
      RegionNode* node = runtime->get_node(requirement.region);
      FieldMask refinement_mask =
          node->column_source->get_field_mask(requirement.privilege_fields);
      parent_ctx->refine_equivalence_sets(
          parent_req_index, node->row_source, refinement_mask,
          map_applied_conditions);
      if (!map_applied_conditions.empty())
        runtime->phase_barrier_arrive(
            reset_barrier, 1 /*count*/,
            Runtime::merge_events(map_applied_conditions));
      else
        runtime->phase_barrier_arrive(reset_barrier, 1 /*count*/);
      complete_mapping(reset_barrier);
      complete_execution();
    }

  }  // namespace Internal
}  // namespace Legion
