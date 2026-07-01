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

#include "legion/operations/deletion.h"
#include "legion/analysis/overwrite.h"
#include "legion/analysis/versioning.h"
#include "legion/contexts/replicate.h"
#include "legion/api/data_impl.h"
#include "legion/managers/shard.h"
#include "legion/nodes/index.h"
#include "legion/utilities/privileges.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Deletion Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeletionOp::DeletionOp(void) : Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    DeletionOp::~DeletionOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void DeletionOp::set_deletion_preconditions(
        const std::map<Operation*, GenerationID>& deps)
    //--------------------------------------------------------------------------
    {
      legion_assert(!has_preconditions);
      dependences = deps;
      has_preconditions = true;
      create_deletion_requirements();
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_index_space_deletion(
        InnerContext* ctx, IndexSpace handle, std::vector<IndexPartition>& subs,
        const bool unordered, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      kind = INDEX_SPACE_DELETION;
      index_space = handle;
      sub_partitions.swap(subs);
      LegionSpy::log_deletion_operation(
          parent_ctx->get_unique_id(), unique_op_id, unordered);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_index_part_deletion(
        InnerContext* ctx, IndexPartition handle,
        std::vector<IndexPartition>& subs, const bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      kind = INDEX_PARTITION_DELETION;
      index_part = handle;
      sub_partitions.swap(subs);
      LegionSpy::log_deletion_operation(
          parent_ctx->get_unique_id(), unique_op_id, unordered);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_field_space_deletion(
        InnerContext* ctx, FieldSpace handle, const bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      kind = FIELD_SPACE_DELETION;
      field_space = handle;
      LegionSpy::log_deletion_operation(
          parent_ctx->get_unique_id(), unique_op_id, unordered);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_field_deletion(
        InnerContext* ctx, FieldSpace handle, FieldID fid, const bool unordered,
        FieldAllocatorImpl* impl, Provenance* provenance,
        const bool non_owner_shard)
    //--------------------------------------------------------------------------
    {
      legion_assert(impl != nullptr);
      legion_assert(allocator == nullptr);
      initialize_operation(ctx, provenance);
      kind = FIELD_DELETION;
      field_space = handle;
      free_fields.insert(fid);
      // Hold a reference to the allocator to keep it alive until
      // we are done performing the field deletion
      allocator = impl;
      allocator->add_reference();
      // Wait for the allocator to be ready before doing this
      // next part if we have to
      if (allocator->ready_event.exists() &&
          !allocator->ready_event.has_triggered())
        allocator->ready_event.wait();
      // Free up the indexes for these fields since we know that they
      // will be deleted at a finite time in the future
      const std::vector<FieldID> field_vec(1, fid);
      runtime->free_field_indexes(
          handle, field_vec, get_mapped_event(), non_owner_shard);
      LegionSpy::log_deletion_operation(
          parent_ctx->get_unique_id(), unique_op_id, unordered);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_field_deletions(
        InnerContext* ctx, FieldSpace handle, const std::set<FieldID>& to_free,
        const bool unordered, FieldAllocatorImpl* impl, Provenance* provenance,
        const bool non_owner_shard)
    //--------------------------------------------------------------------------
    {
      legion_assert(impl != nullptr);
      legion_assert(allocator == nullptr);
      initialize_operation(ctx, provenance);
      kind = FIELD_DELETION;
      field_space = handle;
      free_fields = to_free;
      // Hold a reference to the allocator to keep it alive until
      // we are done performing the field deletion
      allocator = impl;
      allocator->add_reference();
      // Wait for the allocator to be ready before doing this
      // next part if we have to
      if (allocator->ready_event.exists() &&
          !allocator->ready_event.has_triggered())
        allocator->ready_event.wait();
      // Free up the indexes for these fields since we know that they
      // will be deleted at a finite time in the future
      const std::vector<FieldID> field_vec(to_free.begin(), to_free.end());
      runtime->free_field_indexes(
          handle, field_vec, get_mapped_event(), non_owner_shard);
      LegionSpy::log_deletion_operation(
          parent_ctx->get_unique_id(), unique_op_id, unordered);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_logical_region_deletion(
        InnerContext* ctx, LogicalRegion handle, const bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      kind = LOGICAL_REGION_DELETION;
      logical_region = handle;
      LegionSpy::log_deletion_operation(
          parent_ctx->get_unique_id(), unique_op_id, unordered);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
      allocator = nullptr;
      has_preconditions = false;
    }

    //--------------------------------------------------------------------------
    void DeletionOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // We can remove the reference to the allocator once we are
      // done with all of our free operations
      if ((allocator != nullptr) && allocator->remove_reference())
        delete allocator;
      sub_partitions.clear();
      free_fields.clear();
      local_fields.clear();
      global_fields.clear();
      local_field_indexes.clear();
      parent_req_indexes.clear();
      deletion_req_indexes.clear();
      returnable_privileges.clear();
      deletion_requirements.clear();
      version_infos.clear();
      map_applied_conditions.clear();
      dependences.clear();
      Operation::deactivate(false /*free*/);
      // Return this to the available deletion ops on the queue
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* DeletionOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DELETION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind DeletionOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DELETION_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void DeletionOp::create_deletion_requirements(void)
    //--------------------------------------------------------------------------
    {
      switch (kind)
      {
        // These cases do not need any kind of analysis to construct
        // any region requirements
        case INDEX_SPACE_DELETION:
        case INDEX_PARTITION_DELETION:
        case FIELD_SPACE_DELETION:
          break;
        case FIELD_DELETION:
          {
            parent_ctx->analyze_destroy_fields(
                field_space, free_fields, deletion_requirements,
                parent_req_indexes, global_fields, local_fields,
                local_field_indexes, deletion_req_indexes);
            break;
          }
        case LOGICAL_REGION_DELETION:
          {
            parent_ctx->analyze_destroy_logical_region(
                logical_region, deletion_requirements, parent_req_indexes,
                returnable_privileges);
            break;
          }
        default:
          std::abort();
      }
      legion_assert(deletion_requirements.size() == parent_req_indexes.size());
    }

    //--------------------------------------------------------------------------
    void DeletionOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (has_preconditions)
      {
        for (const std::pair<Operation* const, GenerationID>& dit : dependences)
          register_dependence(dit.first, dit.second);
        // We still need to perform the invalidations in this path as well
        const ContextID ctx = parent_ctx->get_logical_tree_context();
        for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
          runtime->invalidate_region_tree_context(
              ctx, deletion_requirements[idx], (kind == FIELD_DELETION));
        return;
      }
      create_deletion_requirements();
      // Even though we're going to do a full fence analysis after this,
      // we still need to do this call so we register ourselves in the
      // region tree to serve as mapping dependences on things that might
      // use these data structures in the case of recycling, e.g. in the
      // case that we recycle a field index
      analyze_region_requirements();
      // Deletions act as upwards acting fences so that we can record
      // dependences on all the operations in front of us in program
      // order and thereby ensure we only commit after they have all
      // committed so deletion effects don't show up out of order.
      parent_ctx->perform_mapping_fence_analysis(this);
      // Now we can invalidate the context since all internal operations
      // have been recorded in the tree
      const ContextID ctx = parent_ctx->get_logical_tree_context();
      for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
        runtime->invalidate_region_tree_context(
            ctx, deletion_requirements[idx], (kind == FIELD_DELETION));
      log_deletion_requirements();
    }

    //--------------------------------------------------------------------------
    void DeletionOp::log_deletion_requirements(void)
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level == NO_SPY_LOGGING)
        return;
      for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
      {
        const RegionRequirement& req = deletion_requirements[idx];
        if (req.handle_type != LEGION_PARTITION_PROJECTION)
          LegionSpy::log_logical_requirement(
              unique_op_id, idx, true /*region*/,
              req.region.index_space.get_id(), req.region.field_space.get_id(),
              req.region.get_tree_id(), req.privilege, req.prop, req.redop,
              req.parent.index_space.get_id());
        else
          LegionSpy::log_logical_requirement(
              unique_op_id, idx, false /*region*/,
              req.partition.index_partition.get_id(),
              req.partition.field_space.get_id(), req.partition.get_tree_id(),
              req.privilege, req.prop, req.redop,
              req.parent.index_space.get_id());
        LegionSpy::log_requirement_fields(
            unique_op_id, idx, req.privilege_fields);
      }
    }

    //--------------------------------------------------------------------------
    void DeletionOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (kind == FIELD_DELETION)
      {
        // Field deletions need to compute their version infos
        std::set<RtEvent> preconditions;
        version_infos.resize(deletion_requirements.size());
        for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
          perform_versioning_analysis(
              idx, deletion_requirements[idx], version_infos[idx],
              preconditions);
        if (!preconditions.empty())
        {
          enqueue_ready_operation(Runtime::merge_events(preconditions));
          return;
        }
      }
      enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void DeletionOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (kind == FIELD_DELETION)
      {
        // For this case we actually need to go through and prune out any
        // valid instances for these fields in the equivalence sets in order
        // to be able to free up the resources.
        const TraceInfo trace_info(this);
        for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
        {
          const VersionInfo& version_info = version_infos[idx];
          invalidate_fields(
              idx, deletion_requirements[idx], version_info,
              PhysicalTraceInfo(trace_info, idx), nullptr /*no collective map*/,
              false /*not collective*/);
          // Make sure we keep the equivalence sets alive while the
          // invalidation analysis is running since we're about to
          // invalidate the equivalence sets in the next step
          const op::FieldMaskMap<EquivalenceSet>& eq_sets =
              version_info.get_equivalence_sets();
          for (op::FieldMaskMap<EquivalenceSet>::const_iterator it =
                   eq_sets.begin();
               it != eq_sets.end(); it++)
            it->first->add_base_gc_ref(FIELD_ALLOCATOR_REF);
        }
        // make sure that we don't try to do the deletion calls until
        // after the allocator is ready
        if (allocator->ready_event.exists())
          map_applied_conditions.insert(allocator->ready_event);
      }
      // Clean out the physical state for these operations once we know that
      // all prior operations that needed the state have been done
      for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
      {
        const RegionRequirement& req = deletion_requirements[idx];
        parent_ctx->invalidate_region_tree_context(
            req, find_parent_index(idx), map_applied_conditions,
            (kind == FIELD_DELETION));
      }
      // Mark that we're done mapping and defer the execution as appropriate
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void DeletionOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      switch (kind)
      {
        case INDEX_SPACE_DELETION:
          {
            legion_assert(deletion_req_indexes.empty());
            runtime->destroy_index_space(
                index_space, runtime->address_space, preconditions);
            if (!sub_partitions.empty())
            {
              for (const IndexPartition& part : sub_partitions)
                runtime->destroy_index_partition(part, preconditions);
            }
            break;
          }
        case INDEX_PARTITION_DELETION:
          {
            legion_assert(deletion_req_indexes.empty());
            runtime->destroy_index_partition(index_part, preconditions);
            if (!sub_partitions.empty())
            {
              for (const IndexPartition& part : sub_partitions)
                runtime->destroy_index_partition(part, preconditions);
            }
            break;
          }
        case FIELD_SPACE_DELETION:
          {
            legion_assert(deletion_req_indexes.empty());
            runtime->destroy_field_space(field_space, preconditions);
            break;
          }
        case FIELD_DELETION:
          {
            if (!local_fields.empty())
              runtime->free_local_fields(
                  field_space, local_fields, local_field_indexes);
            if (!global_fields.empty())
              runtime->free_fields(field_space, global_fields, preconditions);
            if (!local_fields.empty())
              parent_ctx->remove_deleted_local_fields(
                  field_space, local_fields);
            break;
          }
        case LOGICAL_REGION_DELETION:
          {
            runtime->destroy_logical_region(logical_region, preconditions);
            break;
          }
        default:
          std::abort();
      }
      // Remove any references that we added to the equivalence sets
      for (unsigned idx = 0; idx < version_infos.size(); idx++)
      {
        const op::FieldMaskMap<EquivalenceSet>& eq_sets =
            version_infos[idx].get_equivalence_sets();
        for (op::FieldMaskMap<EquivalenceSet>::const_iterator it =
                 eq_sets.begin();
             it != eq_sets.end(); it++)
          if (it->first->remove_base_gc_ref(FIELD_ALLOCATOR_REF))
            delete it->first;
      }
      if (!preconditions.empty())
        commit_operation(
            true /*deactivate*/, Runtime::merge_events(preconditions));
      else
        commit_operation(true /*deactivate*/);
    }

    //--------------------------------------------------------------------------
    unsigned DeletionOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx < parent_req_indexes.size());
      legion_assert(parent_req_indexes[idx] != TRACED_PARENT_INDEX);
      return parent_req_indexes[idx];
    }

    //--------------------------------------------------------------------------
    void DeletionOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::invalidate_fields(
        unsigned index, const RegionRequirement& req,
        const VersionInfo& version_info, const PhysicalTraceInfo& trace_info,
        CollectiveMapping* collective_mapping,
        const bool collective_first_local)
    //--------------------------------------------------------------------------
    {
      legion_assert(req.handle_type == LEGION_SINGULAR_PROJECTION);

      const RegionUsage usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
      IndexSpaceNode* local_node =
          runtime->get_node(req.region.get_index_space());
      OverwriteAnalysis* analysis = new OverwriteAnalysis(
          this, index, usage, local_node, nullptr /*view*/,
          version_info.get_valid_mask(), trace_info, collective_mapping,
          ApEvent::NO_AP_EVENT, PredEvent::NO_PRED_EVENT,
          PredEvent::NO_PRED_EVENT, false /*add restriction*/,
          collective_first_local);
      analysis->add_reference();
      const RtEvent traversal_done = analysis->perform_traversal(
          RtEvent::NO_RT_EVENT, version_info, map_applied_conditions);
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, map_applied_conditions);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Repl Deletion Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplDeletionOp::ReplDeletionOp(void)
      : ReplCollectiveVersioning<CollectiveVersioning<DeletionOp> >()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplDeletionOp::~ReplDeletionOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveVersioning<CollectiveVersioning<DeletionOp> >::activate();
      ready_barrier = RtBarrier::NO_RT_BARRIER;
      mapping_barrier = RtBarrier::NO_RT_BARRIER;
      commit_barrier = RtBarrier::NO_RT_BARRIER;
      is_first_local_shard = false;
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveVersioning<CollectiveVersioning<DeletionOp> >::deactivate(
          false /*freeop*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Do the base call
      DeletionOp::trigger_dependence_analysis();
      // Then get any barriers that we need for our execution
      // We might have already received our barriers
      if (commit_barrier.exists())
        return;
      legion_assert(!mapping_barrier.exists());
      legion_assert(!commit_barrier.exists());
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // Only field and region deletions need a ready barrier since they
      // will be touching the physical states of the region tree
      if ((kind == LOGICAL_REGION_DELETION) || (kind == FIELD_DELETION))
      {
        ready_barrier = repl_ctx->get_next_deletion_ready_barrier();
        mapping_barrier = repl_ctx->get_next_deletion_mapping_barrier();
        if (kind == FIELD_DELETION)
          create_collective_rendezvous(0 /*requirement index*/);
      }
      // All deletion kinds need an execution barrier
      commit_barrier = repl_ctx->get_next_deletion_execution_barrier();
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if ((kind == FIELD_DELETION) || (kind == LOGICAL_REGION_DELETION))
        runtime->phase_barrier_arrive(ready_barrier, 1 /*count*/);
      if (kind == FIELD_DELETION)
      {
        // Field deletions need to compute their version infos
        std::set<RtEvent> preconditions;
        version_infos.resize(deletion_requirements.size());
        for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
          perform_versioning_analysis(
              idx, deletion_requirements[idx], version_infos[idx],
              preconditions, nullptr /*output region*/,
              true /*collective rendezvous*/);
        if (!preconditions.empty())
        {
          preconditions.insert(ready_barrier);
          enqueue_ready_operation(Runtime::merge_events(preconditions));
          return;
        }
      }
      enqueue_ready_operation(ready_barrier);
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      if (kind == FIELD_DELETION)
      {
        legion_assert(mapping_barrier.exists());
        if (is_first_local_shard)
        {
          // For this case we actually need to go through and prune out any
          // valid instances for these fields in the equivalence sets in order
          // to be able to free up the resources.
          const TraceInfo trace_info(this);
          for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
          {
            const VersionInfo& version_info = version_infos[idx];
            invalidate_fields(
                idx, deletion_requirements[idx], version_info,
                PhysicalTraceInfo(trace_info, idx),
                &repl_ctx->shard_manager->get_collective_mapping(),
                is_first_local_shard);
            // Make sure we keep the equivalence sets alive while the
            // invalidation analysis is running since we're about to
            // invalidate the equivalence sets in the next step
            const op::FieldMaskMap<EquivalenceSet>& eq_sets =
                version_info.get_equivalence_sets();
            for (op::FieldMaskMap<EquivalenceSet>::const_iterator it =
                     eq_sets.begin();
                 it != eq_sets.end(); it++)
              it->first->add_base_gc_ref(FIELD_ALLOCATOR_REF);
          }
        }
        // make sure that we don't try to do the deletion calls until
        // after the allocator is ready
        if (allocator->ready_event.exists())
          map_applied_conditions.insert(allocator->ready_event);
      }
      // Clean out the physical state for these operations once we know that
      // all prior operations that needed the state have been done
      for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
      {
        const RegionRequirement& req = deletion_requirements[idx];
        parent_ctx->invalidate_region_tree_context(
            req, find_parent_index(idx), map_applied_conditions,
            (kind == FIELD_DELETION));
      }
      if (mapping_barrier.exists())
      {
        if (!map_applied_conditions.empty())
          runtime->phase_barrier_arrive(
              mapping_barrier, 1 /*count*/,
              Runtime::merge_events(map_applied_conditions));
        else
          runtime->phase_barrier_arrive(mapping_barrier, 1 /*count*/);
        complete_mapping(mapping_barrier);
      }
      else if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(commit_barrier.exists());
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      if (!commit_barrier.has_triggered())
      {
        // We need to make sure all the operations across all the shards
        // have committed before we actually do this deletion on every
        // shard. If we ever move to a mode where we do a commit barrier
        // for every operation in a control replicated context then we can
        // get rid of this but for now it is absolutely necessary
        runtime->phase_barrier_arrive(commit_barrier, 1 /*count*/);
        if (!commit_barrier.has_triggered())
        {
          DeferDeletionCommitArgs args(this);
          runtime->issue_runtime_meta_task(
              args, LG_THROUGHPUT_DEFERRED_PRIORITY, commit_barrier);
          return;
        }
      }
      std::set<RtEvent> applied;
      const CollectiveMapping& mapping =
          repl_ctx->shard_manager->get_collective_mapping();
      if (is_first_local_shard)
      {
        switch (kind)
        {
          case INDEX_SPACE_DELETION:
            {
              legion_assert(deletion_req_indexes.empty());
              runtime->destroy_index_space(
                  index_space, runtime->address_space, applied, &mapping);
              if (!sub_partitions.empty())
              {
                for (const IndexPartition& part : sub_partitions)
                  runtime->destroy_index_partition(part, applied, &mapping);
              }
              break;
            }
          case INDEX_PARTITION_DELETION:
            {
              legion_assert(deletion_req_indexes.empty());
              runtime->destroy_index_partition(index_part, applied, &mapping);
              if (!sub_partitions.empty())
              {
                for (const IndexPartition& part : sub_partitions)
                  runtime->destroy_index_partition(part, applied, &mapping);
              }
              break;
            }
          case FIELD_SPACE_DELETION:
            {
              legion_assert(deletion_req_indexes.empty());
              runtime->destroy_field_space(field_space, applied, &mapping);
              break;
            }
          case FIELD_DELETION:
            // Everyone is going to do the same thing for field deletions
            break;
          case LOGICAL_REGION_DELETION:
            {
              runtime->destroy_logical_region(
                  logical_region, applied, &mapping);
              break;
            }
          default:
            std::abort();
        }
      }
      // If this is a field deletion then everyone does the same thing
      if (kind == FIELD_DELETION)
      {
        if (!local_fields.empty())
          runtime->free_local_fields(
              field_space, local_fields, local_field_indexes, &mapping);
        if (!global_fields.empty())
          runtime->free_fields(
              field_space, global_fields, applied,
              (repl_ctx->owner_shard->shard_id != 0));
        if (!local_fields.empty())
          parent_ctx->remove_deleted_local_fields(field_space, local_fields);
      }
      // Remove any references that we added to the equivalence sets
      for (unsigned idx = 0; idx < version_infos.size(); idx++)
      {
        const op::FieldMaskMap<EquivalenceSet>& eq_sets =
            version_infos[idx].get_equivalence_sets();
        for (op::FieldMaskMap<EquivalenceSet>::const_iterator it =
                 eq_sets.begin();
             it != eq_sets.end(); it++)
          if (it->first->remove_base_gc_ref(FIELD_ALLOCATOR_REF))
            delete it->first;
      }
      // Still have to do this for legion spy
      LegionSpy::log_operation_events(
          unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
      // commit once all the shards are done
      if (!applied.empty())
        commit_operation(true /*deactivate*/, Runtime::merge_events(applied));
      else
        commit_operation(true /*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::DeferDeletionCommitArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      op->trigger_commit();
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::initialize_replication(
        ReplicateContext* ctx, bool is_first, RtBarrier* ready_bar,
        RtBarrier* mapping_bar, RtBarrier* commit_bar)
    //--------------------------------------------------------------------------
    {
      legion_assert(!ready_barrier.exists());
      legion_assert(!mapping_barrier.exists());
      legion_assert(!commit_barrier.exists());
      is_first_local_shard = is_first;
      if (commit_bar != nullptr)
      {
        // Get our barriers now
        if ((kind == LOGICAL_REGION_DELETION) || (kind == FIELD_DELETION))
        {
          ready_barrier = *ready_bar;
          Runtime::advance_barrier(*ready_bar);
          mapping_barrier = *mapping_bar;
          Runtime::advance_barrier(*mapping_bar);
        }
        // All deletion kinds need an execution barrier
        commit_barrier = *commit_bar;
        Runtime::advance_barrier(*commit_bar);
      }
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::record_unordered_kind(
        std::map<IndexSpace, ReplDeletionOp*>& index_space_deletions,
        std::map<IndexPartition, ReplDeletionOp*>& index_partition_deletions,
        std::map<FieldSpace, ReplDeletionOp*>& field_space_deletions,
        std::map<std::pair<FieldSpace, FieldID>, ReplDeletionOp*>&
            field_deletions,
        std::map<LogicalRegion, ReplDeletionOp*>& logical_region_deletions)
    //--------------------------------------------------------------------------
    {
      switch (kind)
      {
        case INDEX_SPACE_DELETION:
          {
            legion_assert(
                index_space_deletions.find(index_space) ==
                index_space_deletions.end());
            index_space_deletions[index_space] = this;
            break;
          }
        case INDEX_PARTITION_DELETION:
          {
            legion_assert(
                index_partition_deletions.find(index_part) ==
                index_partition_deletions.end());
            index_partition_deletions[index_part] = this;
            break;
          }
        case FIELD_SPACE_DELETION:
          {
            legion_assert(
                field_space_deletions.find(field_space) ==
                field_space_deletions.end());
            field_space_deletions[field_space] = this;
            break;
          }
        case FIELD_DELETION:
          {
            legion_assert(!free_fields.empty());
            const std::pair<FieldSpace, FieldID> key(
                field_space, *(free_fields.begin()));
            legion_assert(field_deletions.find(key) == field_deletions.end());
            field_deletions[key] = this;
            break;
          }
        case LOGICAL_REGION_DELETION:
          {
            legion_assert(
                logical_region_deletions.find(logical_region) ==
                logical_region_deletions.end());
            logical_region_deletions[logical_region] = this;
            break;
          }
        default:
          std::abort();  // should never get here
      }
    }

    /////////////////////////////////////////////////////////////
    // Remote Deletion Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteDeletionOp::RemoteDeletionOp(Operation* ptr, AddressSpaceID src)
      : RemoteOp(ptr, src)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteDeletionOp::~RemoteDeletionOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemoteDeletionOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteDeletionOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteDeletionOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteDeletionOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const char* RemoteDeletionOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DELETION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind RemoteDeletionOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DELETION_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteDeletionOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
    }

    //--------------------------------------------------------------------------
    void RemoteDeletionOp::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      // Nothing for the moment
    }

  }  // namespace Internal
}  // namespace Legion
