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

#include "legion/operations/refinement.h"
#include "legion/contexts/replicate.h"
#include "legion/nodes/region.h"
#include "legion/tracing/logical.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Refinment Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RefinementOp::RefinementOp(void) : InternalOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RefinementOp::~RefinementOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void RefinementOp::activate(void)
    //--------------------------------------------------------------------------
    {
      InternalOp::activate();
      refinement_node = nullptr;
      parent_req_index = 0;
      refinement_number = 0;
    }

    //--------------------------------------------------------------------------
    void RefinementOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      refinement_mask.clear();
      InternalOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* RefinementOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[REFINEMENT_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind RefinementOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return REFINEMENT_OP_KIND;
    }

    //--------------------------------------------------------------------------
    const FieldMask& RefinementOp::get_internal_mask(void) const
    //--------------------------------------------------------------------------
    {
      return refinement_mask;
    }

    //--------------------------------------------------------------------------
    void RefinementOp::initialize(
        Operation* creator, unsigned index, LogicalRegion parent,
        RegionTreeNode* refine, unsigned parent_index)
    //--------------------------------------------------------------------------
    {
      legion_assert(refinement_node == nullptr);
      initialize_internal(creator, index);
      refinement_node = refine;
      parent_req_index = parent_index;
      if (tracing)
        trace->register_internal(this);
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        LegionSpy::log_refinement_operation(
            parent_ctx->get_unique_id(), unique_op_id);
        if (refinement_node->is_region())
        {
          RegionNode* root = refinement_node->as_region_node();
          LegionSpy::log_logical_requirement(
              unique_op_id, 0 /*idx*/, true /*region*/,
              root->handle.index_space.get_id(),
              root->handle.field_space.get_id(), root->handle.get_tree_id(),
              LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0 /*redop*/,
              parent.index_space.get_id());
        }
        else
        {
          PartitionNode* root = refinement_node->as_partition_node();
          LegionSpy::log_logical_requirement(
              unique_op_id, 0 /*idx*/, false /*region*/,
              root->handle.index_partition.get_id(),
              root->handle.field_space.get_id(), root->handle.get_tree_id(),
              LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0 /*redop*/,
              parent.index_space.get_id());
        }
        LegionSpy::log_internal_op_creator(
            unique_op_id, creator->get_unique_op_id(), index);
      }
    }

    //--------------------------------------------------------------------------
    void RefinementOp::record_refinement_mask(
        unsigned number, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(!refinement_mask);
      legion_assert(refinement_node != nullptr);
      refinement_mask = mask;
      refinement_number = number;
      if ((spy_logging_level > NO_SPY_LOGGING) && !!mask)
      {
        std::set<FieldID> fields;
        refinement_node->column_source->get_field_set(mask, parent_ctx, fields);
        LegionSpy::log_requirement_fields(unique_op_id, 0 /*idx*/, fields);
      }
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* RefinementOp::get_refinement_node(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(refinement_node != nullptr);
      return refinement_node;
    }

    //--------------------------------------------------------------------------
    void RefinementOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Set the must epoch op back to nullptr since we know our dependence
      // analysis is now complete and we want to run through the rest
      // of the pipeline without being impeded by the must epoch op
      must_epoch = nullptr;
    }

    //--------------------------------------------------------------------------
    void RefinementOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!!refinement_mask);
      legion_assert(refinement_node != nullptr);
      std::vector<RtEvent> map_applied_conditions;
      // Check to see if this is a region or a parttiion
      if (refinement_node->is_region())
      {
        RegionNode* region = refinement_node->as_region_node();
        parent_ctx->refine_equivalence_sets(
            parent_req_index, region->row_source, refinement_mask,
            map_applied_conditions);
      }
      else
      {
        IndexPartNode* partition =
            refinement_node->as_partition_node()->row_source;
        if (partition->is_disjoint() && !partition->is_complete())
        {
          // For disjoint and incomplete partitions we can traverse
          // each of their children individually and refine them
          for (ColorSpaceIterator itr(partition); itr; itr++)
          {
            IndexSpaceNode* child = partition->get_child(*itr);
            parent_ctx->refine_equivalence_sets(
                parent_req_index, child, refinement_mask,
                map_applied_conditions);
          }
        }
        else
        {
          // For complete partitions we refine from the root since it will
          // have the same impact as if we did it by individual subregions
          // For aliased but incomplete partitions we just do it from the
          // root as well since we can't compute the overlapping parts
          parent_ctx->refine_equivalence_sets(
              parent_req_index, partition->parent, refinement_mask,
              map_applied_conditions);
        }
      }
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      complete_execution();
    }

    /////////////////////////////////////////////////////////////
    // Repl Refinement Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplRefinementOp::ReplRefinementOp(void) : RefinementOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplRefinementOp::~ReplRefinementOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplRefinementOp::activate(void)
    //--------------------------------------------------------------------------
    {
      RefinementOp::activate();
      mapped_barrier = RtBarrier::NO_RT_BARRIER;
      refinement_barrier = RtBarrier::NO_RT_BARRIER;
    }

    //--------------------------------------------------------------------------
    void ReplRefinementOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      RefinementOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplRefinementOp::set_repl_refinement_info(
        RtBarrier mapped_bar, RtBarrier refinement_bar)
    //--------------------------------------------------------------------------
    {
      legion_assert(!mapped_barrier.exists());
      legion_assert(!refinement_barrier.exists());
      mapped_barrier = mapped_bar;
      refinement_barrier = refinement_bar;
    }

    //--------------------------------------------------------------------------
    void ReplRefinementOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(refinement_barrier.exists());
      runtime->phase_barrier_arrive(refinement_barrier, 1 /*count*/);
      enqueue_ready_operation(refinement_barrier);
    }

    //--------------------------------------------------------------------------
    void ReplRefinementOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(mapped_barrier.exists());
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      std::vector<RtEvent> map_applied_conditions;
      // Check to see if this is a region or a parttiion
      if (refinement_node->is_region())
      {
        RegionNode* region = refinement_node->as_region_node();
        // Replicated so no need to do sharding
        parent_ctx->refine_equivalence_sets(
            parent_req_index, region->row_source, refinement_mask,
            map_applied_conditions);
      }
      else
      {
        IndexPartNode* partition =
            refinement_node->as_partition_node()->row_source;
        if (partition->is_disjoint() && !partition->is_complete())
        {
          // We have a small heuristic here, if the partition has at least
          // half as many children as there are shards then we will shard
          // the traversal, otherwise we can replicate the refinements and
          // do them without any communication
          if (repl_ctx->total_shards <= uint64_t(2 * partition->total_children))
          {
            for (ColorSpaceIterator itr(
                     partition, repl_ctx->owner_shard->shard_id,
                     repl_ctx->total_shards);
                 itr; itr++)
            {
              IndexSpaceNode* child = partition->get_child(*itr);
              parent_ctx->refine_equivalence_sets(
                  parent_req_index, child, refinement_mask,
                  map_applied_conditions, true /*sharded*/);
            }
          }
          else
          {
            // Not sharded path
            for (ColorSpaceIterator itr(partition); itr; itr++)
            {
              IndexSpaceNode* child = partition->get_child(*itr);
              parent_ctx->refine_equivalence_sets(
                  parent_req_index, child, refinement_mask,
                  map_applied_conditions);
            }
          }
        }
        else
        {
          // For complete partitions we refine from the root since it will
          // have the same impact as if we did it by individual subregions
          // For aliased but incomplete partitions we just do it from the
          // root as well since we can't compute the overlapping parts
          // This is replicated so no need to do sharding
          parent_ctx->refine_equivalence_sets(
              parent_req_index, partition->parent, refinement_mask,
              map_applied_conditions);
        }
      }
      if (!map_applied_conditions.empty())
        runtime->phase_barrier_arrive(
            mapped_barrier, 1 /*count*/,
            Runtime::merge_events(map_applied_conditions));
      else
        runtime->phase_barrier_arrive(mapped_barrier, 1 /*count*/);
      complete_mapping(mapped_barrier);
      complete_execution();
    }

  }  // namespace Internal
}  // namespace Legion
