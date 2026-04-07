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

#include "legion/contexts/inner.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/analysis/projection.h"
#include "legion/api/data_impl.h"
#include "legion/api/functors_impl.h"
#include "legion/api/future_impl.h"
#include "legion/api/predicate_impl.h"
#include "legion/api/redop.h"
#include "legion/managers/shard.h"
#include "legion/nodes/kdtree.h"
#include "legion/nodes/region.h"
#include "legion/operations/acquire.h"
#include "legion/operations/allreduce.h"
#include "legion/operations/attach.h"
#include "legion/operations/begin.h"
#include "legion/operations/boolean.h"
#include "legion/operations/close.h"
#include "legion/operations/complete.h"
#include "legion/operations/copy.h"
#include "legion/operations/creation.h"
#include "legion/operations/deletion.h"
#include "legion/operations/dependent.h"
#include "legion/operations/detach.h"
#include "legion/operations/discard.h"
#include "legion/operations/dynamic.h"
#include "legion/operations/fence.h"
#include "legion/operations/fill.h"
#include "legion/operations/frame.h"
#include "legion/operations/mapping.h"
#include "legion/operations/mustepoch.h"
#include "legion/operations/partition.h"
#include "legion/operations/recurrent.h"
#include "legion/operations/refinement.h"
#include "legion/operations/release.h"
#include "legion/operations/reset.h"
#include "legion/operations/timing.h"
#include "legion/operations/tunable.h"
#include "legion/tasks/index.h"
#include "legion/tasks/individual.h"
#include "legion/tracing/logical.h"
#include "legion/utilities/privileges.h"
#include "legion/utilities/provenance.h"
#include "legion/views/allreduce.h"
#include "legion/views/fill.h"
#include "legion/views/individual.h"
#include "legion/views/replicate.h"

#define SWAP_PART_KINDS(k1, k2) std::swap(k1, k2)

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Inner Context
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InnerContext::InnerContext(
        const Mapper::ContextConfigOutput& config, SingleTask* owner, int d,
        bool finner, const std::vector<RegionRequirement>& reqs,
        const std::vector<OutputRequirement>& out_reqs,
        const std::vector<unsigned>& parent_indexes,
        const std::vector<bool>& virt_mapped, TaskPriority task_priority,
        ApEvent exec_fence, DistributedID id, bool inline_task,
        bool implicit_task, bool concurrent, CollectiveMapping* mapping)
      : TaskContext(
            owner, d, reqs, out_reqs,
            LEGION_DISTRIBUTED_HELP_ENCODE(
                (id > 0) ? id : runtime->get_available_distributed_id(),
                INNER_CONTEXT_DC),
            (id == 0) /*register if not remote*/, inline_task, implicit_task,
            mapping),
        tree_context(runtime->allocate_region_tree_context()),
        full_inner_context(finner), concurrent_context(concurrent),
        finished_execution(false), has_inline_accessor(false),
        next_created_index(0), context_configuration(config),
        parent_req_indexes(parent_indexes), virtual_mapped(virt_mapped),
        total_children_count(0), next_blocking_index(0),
        outstanding_prepipeline_tasks(0),
        enqueue_task_comp_queue(CompletionQueue::NO_QUEUE),
        trigger_execution_comp_queue(CompletionQueue::NO_QUEUE),
        deferred_execution_comp_queue(CompletionQueue::NO_QUEUE),
        deferred_mapped_comp_queue(CompletionQueue::NO_QUEUE),
        deferred_completion_comp_queue(CompletionQueue::NO_QUEUE),
        deferred_commit_comp_queue(CompletionQueue::NO_QUEUE),
        current_trace(nullptr), previous_trace(nullptr),
        physical_trace_replay_status(0), outstanding_subtasks(0),
        pending_subtasks(0), pending_frames(0), currently_active_context(false),
        outstanding_commit_task(false), current_mapping_fence(nullptr),
        current_mapping_fence_gen(0), current_mapping_fence_index(0),
        current_execution_fence_event(exec_fence),
        current_execution_fence_index(0), last_implicit_creation(nullptr),
        last_implicit_creation_gen(0), current_priority(task_priority)
    //--------------------------------------------------------------------------
    {
      mutable_priority = context_configuration.mutable_priority;
      // If we have an owner, clone our local fields from its context
      // and also compute the coordinates for this context in the task tree
      if (owner != nullptr)
      {
        TaskContext* owner_ctx = owner_task->get_context();
        InnerContext* parent_ctx = legion_safe_cast<InnerContext*>(owner_ctx);
        parent_ctx->clone_local_fields(local_field_infos);
        // Get the coordinates for the parent task
        parent_ctx->compute_task_tree_coordinates(context_coordinates);
        // Then add our coordinates for our task
        context_coordinates.emplace_back(ContextCoordinate(
            owner_task->get_context_index(), owner_task->index_point));
        // Fill in the next created index based on the requirements
        // Only safe to do this if we've got an owner task and therefore
        // we know the regions vector is initialized
        next_created_index = reqs.size();
      }
#ifdef LEGION_GC
      log_garbage.info(
          "GC Inner Context %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
      current_fence_uid = 0;
    }

    //--------------------------------------------------------------------------
    InnerContext::~InnerContext(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(deletion_counts.empty());
      // At this point we can free our region tree context
      runtime->free_region_tree_context(tree_context);
      if (enqueue_task_comp_queue.exists())
        enqueue_task_comp_queue.destroy();
      if (trigger_execution_comp_queue.exists())
        trigger_execution_comp_queue.destroy();
      if (deferred_execution_comp_queue.exists())
        deferred_execution_comp_queue.destroy();
      if (deferred_mapped_comp_queue.exists())
        deferred_mapped_comp_queue.destroy();
      if (deferred_completion_comp_queue.exists())
        deferred_completion_comp_queue.destroy();
      if (deferred_commit_comp_queue.exists())
        deferred_commit_comp_queue.destroy();
      // Clean up any locks and barriers that the user
      // asked us to destroy
      while (!context_locks.empty())
      {
        context_locks.back().destroy_reservation();
        context_locks.pop_back();
      }
      while (!context_barriers.empty())
      {
        Realm::Barrier bar = context_barriers.back();
        bar.destroy_barrier();
        context_barriers.pop_back();
      }
      if (window_wait.exists())
        Runtime::trigger_event(window_wait);
      // No need for the lock here since we're being cleaned up
      if (!local_field_infos.empty())
        local_field_infos.clear();
      if (!attach_functions.empty())
      {
        for (const std::map<
                 IndexTreeNode*,
                 std::vector<AttachProjectionFunctor*>>::value_type& fit :
             attach_functions)
        {
          for (AttachProjectionFunctor* const & it : fit.second)
          {
            // Unregister it with the runtime if it is not the identity
            // The runtime will delete the functor for us
            if (it->pid > 0)
              runtime->unregister_projection_functor(it->pid);
            else  // This is the identity so we can just delete it ourself
              delete it;
          }
        }
        attach_functions.clear();
      }
      legion_assert(pending_top_views.empty());
      legion_assert(outstanding_subtasks == 0);
      legion_assert(pending_subtasks == 0);
      legion_assert(pending_frames == 0);
    }

    //--------------------------------------------------------------------------
    void InnerContext::notify_local(void)
    //--------------------------------------------------------------------------
    {
      // Remove any references that we are holding on instance top views
      std::map<PhysicalManager*, IndividualView*> to_unregister;
      {
        AutoLock inst_lock(instance_view_lock);
        to_unregister.swap(instance_top_views);
      }
      for (const std::map<PhysicalManager*, IndividualView*>::value_type& it :
           to_unregister)
      {
        it.first->unregister_active_context(this);
        if (it.second->remove_nested_gc_ref(did))
          delete it.second;
      }
      // Remove any global references that we are holding on collective views
      std::map<RegionTreeID, std::vector<CollectiveResult*>> to_release;
      {
        AutoLock c_lock(collective_lock);
        to_release.swap(collective_results);
      }
      for (const std::map<
               RegionTreeID, std::vector<CollectiveResult*>>::value_type& rit :
           to_release)
      {
        for (CollectiveResult* const & it : rit.second)
        {
          release_collective_view(did, it->collective_did);
          delete it;
        }
      }
      // Shouldn't need any lock for these as the context is not longer
      // valid and there shouldn't be any races
      while (!fill_view_cache.empty())
      {
        FillView* next = fill_view_cache.front().view;
        if (next->remove_nested_valid_ref(did))
          delete next;
        fill_view_cache.pop_front();
      }
      // Traces can refer back to us so make sure we remove our references
      // to them here so they can clean up their resource referenes to us
      for (const std::map<TraceID, LogicalTrace*>::value_type& it : traces)
        if (it.second->remove_reference())
          delete it.second;
      traces.clear();
    }

    //--------------------------------------------------------------------------
    void InnerContext::receive_resources(
        uint64_t return_index, std::map<LogicalRegion, unsigned>& created_regs,
        std::vector<DeletedRegion>& deleted_regs,
        std::set<std::pair<FieldSpace, FieldID>>& created_fids,
        std::vector<DeletedField>& deleted_fids,
        std::map<FieldSpace, unsigned>& created_fs,
        std::map<FieldSpace, std::set<LogicalRegion>>& latent_fs,
        std::vector<DeletedFieldSpace>& deleted_fs,
        std::map<IndexSpace, unsigned>& created_is,
        std::vector<DeletedIndexSpace>& deleted_is,
        std::map<IndexPartition, unsigned>& created_partitions,
        std::vector<DeletedPartition>& deleted_partitions,
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      bool need_deletion_dependences = true;
      std::map<Operation*, GenerationID> dependences;
      if (!created_regs.empty())
        register_region_creations(created_regs);
      if (!deleted_regs.empty())
      {
        compute_return_deletion_dependences(return_index, dependences);
        need_deletion_dependences = false;
        register_region_deletions(dependences, deleted_regs, preconditions);
      }
      if (!created_fids.empty())
        register_field_creations(created_fids);
      if (!deleted_fids.empty())
      {
        if (need_deletion_dependences)
        {
          compute_return_deletion_dependences(return_index, dependences);
          need_deletion_dependences = false;
        }
        register_field_deletions(dependences, deleted_fids, preconditions);
      }
      if (!created_fs.empty())
        register_field_space_creations(created_fs);
      if (!latent_fs.empty())
        register_latent_field_spaces(latent_fs);
      if (!deleted_fs.empty())
      {
        if (need_deletion_dependences)
        {
          compute_return_deletion_dependences(return_index, dependences);
          need_deletion_dependences = false;
        }
        register_field_space_deletions(dependences, deleted_fs, preconditions);
      }
      if (!created_is.empty())
        register_index_space_creations(created_is);
      if (!deleted_is.empty())
      {
        if (need_deletion_dependences)
        {
          compute_return_deletion_dependences(return_index, dependences);
          need_deletion_dependences = false;
        }
        register_index_space_deletions(dependences, deleted_is, preconditions);
      }
      if (!created_partitions.empty())
        register_index_partition_creations(created_partitions);
      if (!deleted_partitions.empty())
      {
        if (need_deletion_dependences)
        {
          compute_return_deletion_dependences(return_index, dependences);
          need_deletion_dependences = false;
        }
        register_index_partition_deletions(
            dependences, deleted_partitions, preconditions);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_region_creations(
        std::map<LogicalRegion, unsigned>& regions)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      if (!latent_field_spaces.empty())
      {
        for (const std::map<LogicalRegion, unsigned>::value_type& it : regions)
        {
          std::map<FieldSpace, std::set<LogicalRegion>>::iterator finder =
              latent_field_spaces.find(it.first.get_field_space());
          if (finder != latent_field_spaces.end())
            finder->second.insert(it.first);
        }
      }
      if (!created_regions.empty())
      {
        for (const std::pair<const LogicalRegion, unsigned>& it : regions)
        {
          std::map<LogicalRegion, unsigned>::iterator finder =
              created_regions.find(it.first);
          if (finder == created_regions.end())
          {
            created_regions.insert(it);
            add_created_region(it.first, false /*task local*/);
          }
          else
            finder->second += it.second;
        }
      }
      else
      {
        created_regions.swap(regions);
        for (const std::map<LogicalRegion, unsigned>::value_type& it :
             created_regions)
          add_created_region(it.first, false /*task local*/);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_region_deletions(
        const std::map<Operation*, GenerationID>& dependences,
        std::vector<DeletedRegion>& regions, std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      std::vector<DeletedRegion> delete_now;
      {
        AutoLock priv_lock(privilege_lock);
        for (const DeletedRegion& rit : regions)
        {
          std::map<LogicalRegion, unsigned>::iterator region_finder =
              created_regions.find(rit.region);
          if (region_finder == created_regions.end())
          {
            if (local_regions.find(rit.region) != local_regions.end())
            {
              Error err(LEGION_PROGRAMMING_MODEL_EXCEPTION);
              err << "Local logical region " << rit.region << " in task "
                  << get_task_name() << " (UID " << get_unique_id()
                  << ") was not deleted by this task. Local regions can only "
                     "be "
                  << "deleted by the task that made them.";
              err.raise();
            }
            // Deletion keeps going up
            deleted_regions.emplace_back(rit);
          }
          else
          {
            // One of ours to delete
            legion_assert(region_finder->second > 0);
            if (--region_finder->second == 0)
            {
              // No need to delete this here, it will be deleted by the op
              // Check to see if we have any latent field spaces to clean up
              if (!latent_field_spaces.empty())
              {
                std::map<FieldSpace, std::set<LogicalRegion>>::iterator finder =
                    latent_field_spaces.find(rit.region.get_field_space());
                if (finder != latent_field_spaces.end())
                {
                  std::set<LogicalRegion>::iterator latent_finder =
                      finder->second.find(rit.region);
                  legion_assert(latent_finder != finder->second.end());
                  finder->second.erase(latent_finder);
                  if (finder->second.empty())
                  {
                    // Now that all the regions using this field space have
                    // been deleted we can clean up all the created_fields
                    for (std::set<std::pair<FieldSpace, FieldID>>::iterator it =
                             created_fields.begin();
                         it != created_fields.end();
                         /*nothing*/)
                    {
                      if (it->first == finder->first)
                      {
                        std::set<std::pair<FieldSpace, FieldID>>::iterator
                            to_delete = it++;
                        created_fields.erase(to_delete);
                      }
                      else
                        it++;
                    }
                    latent_field_spaces.erase(finder);
                  }
                }
              }
              delete_now.emplace_back(rit);
            }
          }
        }
      }
      if (!delete_now.empty())
      {
        for (const DeletedRegion& it : delete_now)
        {
          DeletionOp* op = runtime->get_operation<DeletionOp>();
          op->initialize_logical_region_deletion(
              this, it.region, true /*unordered*/, it.provenance);
          if (!add_to_dependence_queue(
                  op, nullptr /*deps*/, true /*unordered*/))
          {
            // We're past the execution of the parent task so we need
            // to run this manually and capture its effects ourselves
            preconditions.insert(op->get_commit_event());
            op->set_deletion_preconditions(dependences);
            op->execute_dependence_analysis();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_field_creations(
        std::set<std::pair<FieldSpace, FieldID>>& fields)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      if (!created_fields.empty())
      {
        for (const std::pair<FieldSpace, FieldID>& field : fields)
        {
          legion_assert(created_fields.find(field) == created_fields.end());
          created_fields.insert(field);
        }
      }
      else
        created_fields.swap(fields);
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_field_deletions(
        const std::map<Operation*, GenerationID>& dependences,
        std::vector<DeletedField>& fields, std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      std::map<std::pair<FieldSpace, Provenance*>, std::set<FieldID>>
          delete_now;
      {
        AutoLock priv_lock(privilege_lock);
        for (const DeletedField& fit : fields)
        {
          const std::pair<FieldSpace, FieldID> key(fit.space, fit.fid);
          std::set<std::pair<FieldSpace, FieldID>>::const_iterator
              field_finder = created_fields.find(key);
          if (field_finder == created_fields.end())
          {
            std::map<std::pair<FieldSpace, FieldID>, bool>::iterator
                local_finder = local_fields.find(key);
            if (local_finder != local_fields.end())
            {
              Error err(LEGION_PROGRAMMING_MODEL_EXCEPTION);
              err << "Local field " << fit.fid << " in field space "
                  << fit.space << " in task " << get_task_name() << " (UID "
                  << get_unique_id()
                  << ") was not deleted by this task. Local fields can only be "
                  << "deleted by the task that made them.";
              err.raise();
            }
            deleted_fields.emplace_back(fit);
          }
          else
          {
            // One of ours to delete
            std::pair<FieldSpace, Provenance*> now_key(
                fit.space, fit.provenance);
            delete_now[now_key].insert(fit.fid);
            // No need to delete this now, it will be deleted
            // when the deletion op makes its region requirements
          }
        }
      }
      if (!delete_now.empty())
      {
        for (const std::pair<
                 const std::pair<FieldSpace, Provenance*>, std::set<FieldID>>&
                 it : delete_now)
        {
          DeletionOp* op = runtime->get_operation<DeletionOp>();
          FieldAllocatorImpl* allocator =
              create_field_allocator(it.first.first, true /*unordered*/);
          op->initialize_field_deletions(
              this, it.first.first, it.second, true /*unordered*/, allocator,
              it.first.second, false /*non owner shard*/);
          if (!add_to_dependence_queue(
                  op, nullptr /*deps*/, true /*unordered*/))
          {
            // We're past the execution of the parent task so we need
            // to run this manually and capture its effects ourselves
            preconditions.insert(op->get_commit_event());
            op->set_deletion_preconditions(dependences);
            op->execute_dependence_analysis();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_field_space_creations(
        std::map<FieldSpace, unsigned>& spaces)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      if (!latent_field_spaces.empty())
      {
        // Remove any latent field spaces we have ownership for
        for (const std::map<FieldSpace, unsigned>::value_type& it : spaces)
        {
          std::map<FieldSpace, std::set<LogicalRegion>>::iterator finder =
              latent_field_spaces.find(it.first);
          if (finder != latent_field_spaces.end())
            latent_field_spaces.erase(finder);
        }
      }
      if (!created_field_spaces.empty())
      {
        for (const std::pair<const FieldSpace, unsigned>& space : spaces)
        {
          std::map<FieldSpace, unsigned>::iterator finder =
              created_field_spaces.find(space.first);
          if (finder == created_field_spaces.end())
            created_field_spaces.insert(space);
          else
            finder->second += space.second;
        }
      }
      else
        created_field_spaces.swap(spaces);
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_latent_field_spaces(
        std::map<FieldSpace, std::set<LogicalRegion>>& spaces)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(privilege_lock);
      if (!created_field_spaces.empty())
      {
        // Remote any latent field spaces we already have ownership on
        for (std::map<FieldSpace, std::set<LogicalRegion>>::iterator it =
                 spaces.begin();
             it != spaces.end();
             /*nothing*/)
        {
          if (created_field_spaces.find(it->first) !=
              created_field_spaces.end())
          {
            std::map<FieldSpace, std::set<LogicalRegion>>::iterator to_delete =
                it++;
            spaces.erase(to_delete);
          }
          else
            it++;
        }
        if (spaces.empty())
          return;
      }
      if (!created_regions.empty())
      {
        // See if any of these regions are copies of our latent spaces
        for (const std::map<LogicalRegion, unsigned>::value_type& it :
             created_regions)
        {
          std::map<FieldSpace, std::set<LogicalRegion>>::iterator finder =
              spaces.find(it.first.get_field_space());
          if (finder != spaces.end())
            finder->second.insert(it.first);
        }
      }
      // Now we can do the merge
      if (!latent_field_spaces.empty())
      {
        for (const std::map<FieldSpace, std::set<LogicalRegion>>::value_type&
                 it : spaces)
        {
          std::map<FieldSpace, std::set<LogicalRegion>>::iterator finder =
              latent_field_spaces.find(it.first);
          if (finder != latent_field_spaces.end())
            finder->second.insert(it.second.begin(), it.second.end());
          else
            latent_field_spaces.insert(it);
        }
      }
      else
        latent_field_spaces.swap(spaces);
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_field_space_deletions(
        const std::map<Operation*, GenerationID>& dependences,
        std::vector<DeletedFieldSpace>& spaces,
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      std::vector<DeletedFieldSpace> delete_now;
      {
        AutoLock priv_lock(privilege_lock);
        for (const DeletedFieldSpace& fit : spaces)
        {
          std::map<FieldSpace, unsigned>::iterator finder =
              created_field_spaces.find(fit.space);
          if (finder != created_field_spaces.end())
          {
            legion_assert(finder->second > 0);
            if (--finder->second == 0)
            {
              delete_now.emplace_back(fit);
              created_field_spaces.erase(finder);
              // Count how many regions are still using this field space
              // that still need to be deleted before we can remove the
              // list of created fields
              std::set<LogicalRegion> remaining_regions;
              for (const std::pair<const LogicalRegion, unsigned>& it :
                   created_regions)
                if (it.first.get_field_space() == fit.space)
                  remaining_regions.insert(it.first);
              for (const std::pair<const LogicalRegion, bool>& it :
                   local_regions)
                if (it.first.get_field_space() == fit.space)
                  remaining_regions.insert(it.first);
              if (remaining_regions.empty())
              {
                // No remaining regions so we can remove any created fields now
                for (std::set<std::pair<FieldSpace, FieldID>>::iterator it =
                         created_fields.begin();
                     it != created_fields.end();
                     /*nothing*/)
                {
                  if (it->first == fit.space)
                  {
                    std::set<std::pair<FieldSpace, FieldID>>::iterator
                        to_delete = it++;
                    created_fields.erase(to_delete);
                  }
                  else
                    it++;
                }
              }
              else
                latent_field_spaces[fit.space] = remaining_regions;
            }
          }
          else
            // If we didn't make this field space, record the deletion
            // and keep going. It will be handled by the context that
            // made the field space
            deleted_field_spaces.emplace_back(fit);
        }
      }
      if (!delete_now.empty())
      {
        for (const DeletedFieldSpace& it : delete_now)
        {
          DeletionOp* op = runtime->get_operation<DeletionOp>();
          op->initialize_field_space_deletion(
              this, it.space, true /*unordered*/, it.provenance);
          if (!add_to_dependence_queue(
                  op, nullptr /*deps*/, true /*unordered*/))
          {
            // We're past the execution of the parent task so we need
            // to run this manually and capture its effects ourselves
            preconditions.insert(op->get_commit_event());
            op->set_deletion_preconditions(dependences);
            op->execute_dependence_analysis();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_index_space_creations(
        std::map<IndexSpace, unsigned>& spaces)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      if (!created_index_spaces.empty())
      {
        for (const std::pair<const IndexSpace, unsigned>& it : spaces)
        {
          std::map<IndexSpace, unsigned>::iterator finder =
              created_index_spaces.find(it.first);
          if (finder == created_index_spaces.end())
            created_index_spaces.insert(it);
          else
            finder->second += it.second;
        }
      }
      else
        created_index_spaces.swap(spaces);
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_index_space_deletions(
        const std::map<Operation*, GenerationID>& dependences,
        std::vector<DeletedIndexSpace>& spaces,
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      std::vector<DeletedIndexSpace> delete_now;
      std::vector<std::vector<IndexPartition>> sub_partitions;
      {
        AutoLock priv_lock(privilege_lock);
        for (const DeletedIndexSpace& sit : spaces)
        {
          std::map<IndexSpace, unsigned>::iterator finder =
              created_index_spaces.find(sit.space);
          if (finder != created_index_spaces.end())
          {
            legion_assert(finder->second > 0);
            if (--finder->second == 0)
            {
              delete_now.emplace_back(sit);
              sub_partitions.resize(sub_partitions.size() + 1);
              created_index_spaces.erase(finder);
              if (sit.recurse)
              {
                std::vector<IndexPartition>& subs = sub_partitions.back();
                // Also remove any index partitions for this index space tree
                for (std::map<IndexPartition, unsigned>::iterator it =
                         created_index_partitions.begin();
                     it != created_index_partitions.end();
                     /*nothing*/)
                {
                  if (it->first.get_tree_id() == sit.space.get_tree_id())
                  {
                    legion_assert(it->second > 0);
                    if (--it->second == 0)
                    {
                      subs.emplace_back(it->first);
                      std::map<IndexPartition, unsigned>::iterator to_delete =
                          it++;
                      created_index_partitions.erase(to_delete);
                    }
                    else
                      it++;
                  }
                  else
                    it++;
                }
              }
            }
          }
          else
            // If we didn't make the index space in this context, just
            // record it and keep going, it will get handled later
            deleted_index_spaces.emplace_back(sit);
        }
      }
      if (!delete_now.empty())
      {
        legion_assert(delete_now.size() == sub_partitions.size());
        for (unsigned idx = 0; idx < delete_now.size(); idx++)
        {
          DeletionOp* op = runtime->get_operation<DeletionOp>();
          op->initialize_index_space_deletion(
              this, delete_now[idx].space, sub_partitions[idx],
              true /*unordered*/, delete_now[idx].provenance);
          if (!add_to_dependence_queue(
                  op, nullptr /*deps*/, true /*unordered*/))
          {
            // We're past the execution of the parent task so we need
            // to run this manually and capture its effects ourselves
            preconditions.insert(op->get_commit_event());
            op->set_deletion_preconditions(dependences);
            op->execute_dependence_analysis();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_index_partition_creations(
        std::map<IndexPartition, unsigned>& parts)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      if (!created_index_partitions.empty())
      {
        for (const std::pair<const IndexPartition, unsigned>& part : parts)
        {
          std::map<IndexPartition, unsigned>::iterator finder =
              created_index_partitions.find(part.first);
          if (finder == created_index_partitions.end())
            created_index_partitions.insert(part);
          else
            finder->second += part.second;
        }
      }
      else
        created_index_partitions.swap(parts);
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_index_partition_deletions(
        const std::map<Operation*, GenerationID>& dependences,
        std::vector<DeletedPartition>& parts, std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      std::vector<DeletedPartition> delete_now;
      std::vector<std::vector<IndexPartition>> sub_partitions;
      {
        AutoLock priv_lock(privilege_lock);
        for (const DeletedPartition& pit : parts)
        {
          std::map<IndexPartition, unsigned>::iterator finder =
              created_index_partitions.find(pit.partition);
          if (finder != created_index_partitions.end())
          {
            legion_assert(finder->second > 0);
            if (--finder->second == 0)
            {
              delete_now.emplace_back(pit);
              sub_partitions.resize(sub_partitions.size() + 1);
              created_index_partitions.erase(finder);
              if (pit.recurse)
              {
                std::vector<IndexPartition>& subs = sub_partitions.back();
                // Remove any other partitions that this partition dominates
                for (std::map<IndexPartition, unsigned>::iterator it =
                         created_index_partitions.begin();
                     it != created_index_partitions.end();
                     /*nothing*/)
                {
                  if ((pit.partition.get_tree_id() ==
                       it->first.get_tree_id()) &&
                      runtime->is_dominated_tree_only(it->first, pit.partition))
                  {
                    legion_assert(it->second > 0);
                    if (--it->second == 0)
                    {
                      subs.emplace_back(it->first);
                      std::map<IndexPartition, unsigned>::iterator to_delete =
                          it++;
                      created_index_partitions.erase(to_delete);
                    }
                    else
                      it++;
                  }
                  else
                    it++;
                }
              }
            }
          }
          else
            // If we didn't make the partition, record it and keep going
            deleted_index_partitions.emplace_back(pit);
        }
      }
      if (!delete_now.empty())
      {
        legion_assert(delete_now.size() == sub_partitions.size());
        for (unsigned idx = 0; idx < delete_now.size(); idx++)
        {
          DeletionOp* op = runtime->get_operation<DeletionOp>();
          op->initialize_index_part_deletion(
              this, delete_now[idx].partition, sub_partitions[idx],
              true /*unordered*/, delete_now[idx].provenance);
          if (!add_to_dependence_queue(
                  op, nullptr /*deps*/, true /*unordered*/))
          {
            // We're past the execution of the parent task so we need
            // to run this manually and capture its effects ourselves
            preconditions.insert(op->get_commit_event());
            op->set_deletion_preconditions(dependences);
            op->execute_dependence_analysis();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::compute_return_deletion_dependences(
        uint64_t return_index, std::map<Operation*, GenerationID>& dependences)
    //--------------------------------------------------------------------------
    {
      // This is a mixed mapping and execution fence analysis
      AutoLock child_lock(child_op_lock, false /*exclusive*/);
      for (const ReorderBufferEntry& entry : reorder_buffer)
      {
        // If it's younger than our deletion we don't care
        if (entry.operation_index >= return_index)
          continue;
        if (!entry.complete)
          dependences[entry.operation] = entry.operation->get_generation();
      }
    }

    //--------------------------------------------------------------------------
    ContextID InnerContext::get_logical_tree_context(void) const
    //--------------------------------------------------------------------------
    {
      return tree_context;
    }

    //--------------------------------------------------------------------------
    ContextID InnerContext::get_physical_tree_context(void) const
    //--------------------------------------------------------------------------
    {
      return tree_context;
    }

    //--------------------------------------------------------------------------
    bool InnerContext::is_inner_context(void) const
    //--------------------------------------------------------------------------
    {
      return full_inner_context;
    }

    //--------------------------------------------------------------------------
    RtEvent InnerContext::compute_equivalence_sets(
        unsigned req_index, const std::vector<EqSetTracker*>& targets,
        const std::vector<AddressSpaceID>& target_spaces,
        AddressSpaceID creation_target_space, IndexSpaceExpression* expr,
        const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(targets.size() == target_spaces.size());
      legion_assert(std::is_sorted(target_spaces.begin(), target_spaces.end()));
      legion_assert(std::binary_search(
          target_spaces.begin(), target_spaces.end(), creation_target_space));
      // If this is virtual mapped, then continue up to the parent
      if ((req_index < regions.size()) && virtual_mapped[req_index])
        return find_parent_context()->compute_equivalence_sets(
            parent_req_indexes[req_index], targets, target_spaces,
            creation_target_space, expr, mask);
      // Find the equivalence set tree for this region requirement
      LocalLock* tree_lock = nullptr;
      EqKDTree* tree = find_equivalence_set_kd_tree(req_index, tree_lock);
      // Then ask the index space expression to traverse the tree for
      // all of its rectangles and find the equivalence sets that are needed
      op::FieldMaskMap<EqKDTree> to_create, new_subscriptions;
      op::FieldMaskMap<EquivalenceSet> eq_sets;
      std::vector<RtEvent> pending_sets;
      op::map<EqKDTree*, Domain> creation_rects;
      op::map<EquivalenceSet*, op::map<Domain, FieldMask>> creation_srcs;
      op::map<ShardID, op::map<Domain, FieldMask>> remote_shard_rects;
      std::vector<unsigned> new_target_references(targets.size(), 0);
      expr->compute_equivalence_sets(
          tree, tree_lock, mask, targets, target_spaces, new_target_references,
          eq_sets, pending_sets, new_subscriptions, to_create, creation_rects,
          creation_srcs, remote_shard_rects);
      legion_assert(remote_shard_rects.empty());
      const CollectiveMapping target_mapping(
          target_spaces, runtime->legion_collective_radix);
      return report_equivalence_sets(
          req_index, target_mapping, targets, creation_target_space, mask,
          new_target_references, eq_sets, new_subscriptions, to_create,
          creation_rects, creation_srcs, 1 /*expected responses*/,
          pending_sets);
    }

    //--------------------------------------------------------------------------
    RtEvent InnerContext::report_equivalence_sets(
        unsigned req_index, const CollectiveMapping& target_mapping,
        const std::vector<EqSetTracker*>& targets,
        const AddressSpaceID creation_target_space, const FieldMask& mask,
        std::vector<unsigned>& new_target_references,
        op::FieldMaskMap<EquivalenceSet>& eq_sets,
        op::FieldMaskMap<EqKDTree>& new_subscriptions,
        op::FieldMaskMap<EqKDTree>& to_create,
        op::map<EqKDTree*, Domain>& creation_rects,
        op::map<EquivalenceSet*, op::map<Domain, FieldMask>>& creation_srcs,
        size_t expected_responses, std::vector<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(targets.size() == target_mapping.size());
      legion_assert(targets.size() == new_target_references.size());
      legion_assert(to_create.size() == creation_rects.size());
      // Figure out where the origin of the target mapping should be
      const AddressSpaceID local_space = runtime->address_space;
      const AddressSpaceID origin_space =
          target_mapping.contains(local_space) ?
              local_space :
              target_mapping.find_nearest(local_space);
      std::vector<AddressSpaceID> children;
      if (origin_space == local_space)
        target_mapping.get_children(origin_space, local_space, children);
      else
        children.emplace_back(origin_space);
      AddressSpaceID creation_child = origin_space;
      if (!to_create.empty() && !children.empty() &&
          (creation_target_space != origin_space))
      {
        std::sort(children.begin(), children.end());
        creation_child = creation_target_space;
        while (!std::binary_search(
            children.begin(), children.end(), creation_child))
        {
          legion_assert(creation_child != origin_space);
          creation_child =
              target_mapping.get_parent(origin_space, creation_child);
        }
      }
      InnerContext* outermost = find_parent_physical_context(req_index);
      for (const AddressSpaceID& cit : children)
      {
        // Send a message back to the child node with the results
        const RtUserEvent ready_event = Runtime::create_rt_user_event();
        ComputeEquivalenceSetsResponse rez;
        {
          RezCheck z(rez);
          outermost->pack_inner_context(rez);
          rez.serialize(runtime->address_space);
          target_mapping.pack(rez);
          for (unsigned idx = 0; idx < targets.size(); idx++)
          {
            rez.serialize(targets[idx]);
            rez.serialize(new_target_references[idx]);
          }
          rez.serialize(creation_target_space);
          rez.serialize(mask);
          rez.serialize<size_t>(eq_sets.size());
          for (op::FieldMaskMap<EquivalenceSet>::const_iterator it =
                   eq_sets.begin();
               it != eq_sets.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize<size_t>(new_subscriptions.size());
          for (op::FieldMaskMap<EqKDTree>::const_iterator it =
                   new_subscriptions.begin();
               it != new_subscriptions.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
          }
          // We only need to send the creation sets along the path to
          // creation_target_space
          if (cit == creation_child)
          {
            rez.serialize<size_t>(to_create.size());
            for (op::FieldMaskMap<EqKDTree>::const_iterator it =
                     to_create.begin();
                 it != to_create.end(); it++)
            {
              legion_assert(
                  creation_rects.find(it->first) != creation_rects.end());
              rez.serialize(it->first);
              rez.serialize(it->second);
              rez.serialize(creation_rects[it->first]);
            }
            rez.serialize<size_t>(creation_srcs.size());
            for (op::map<EquivalenceSet*, op::map<Domain, FieldMask>>::
                     const_iterator sit = creation_srcs.begin();
                 sit != creation_srcs.end(); sit++)
            {
              // No need to pack a reference since we know that that
              // EqKDTree is still holding references to the creation_srcs
              // until we're done making the new equivalence sets
              rez.serialize(sit->first->did);
              rez.serialize<size_t>(sit->second.size());
              for (op::map<Domain, FieldMask>::const_iterator it =
                       sit->second.begin();
                   it != sit->second.end(); it++)
              {
                rez.serialize(it->first);
                rez.serialize(it->second);
              }
            }
          }
          else
          {
            rez.serialize<size_t>(0);
            rez.serialize<size_t>(0);
          }
          rez.serialize(expected_responses);
          rez.serialize(ready_event);
        }
        rez.dispatch(cit);
        ready_events.emplace_back(ready_event);
      }
      if (origin_space == local_space)
      {
        // We can report the results back immediately
        const unsigned target_index = target_mapping.find_index(local_space);
        legion_assert(target_index < targets.size());
        targets[target_index]->record_equivalence_sets(
            outermost, mask, eq_sets, to_create, creation_rects, creation_srcs,
            new_subscriptions, new_target_references[target_index], local_space,
            expected_responses, ready_events, target_mapping, targets,
            creation_target_space);
      }
      if (!ready_events.empty())
        return Runtime::merge_events(ready_events);
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ComputeEquivalenceSetsResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      InnerContext* context = InnerContext::unpack_inner_context(derez);
      AddressSpaceID source_space;
      derez.deserialize(source_space);
      size_t total_spaces;
      derez.deserialize(total_spaces);
      const CollectiveMapping target_mapping(derez, total_spaces);
      const AddressSpaceID origin_space =
          target_mapping.contains(source_space) ?
              source_space :
              target_mapping.find_nearest(source_space);
      // Check to see if there are any children to continue sending this to
      std::vector<AddressSpaceID> children;
      target_mapping.get_children(
          origin_space, runtime->address_space, children);
      std::vector<EqSetTracker*> targets(target_mapping.size());
      std::vector<unsigned> new_target_references(target_mapping.size());
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        derez.deserialize(targets[idx]);
        derez.deserialize(new_target_references[idx]);
      }
      AddressSpaceID creation_target_space;
      derez.deserialize(creation_target_space);
      FieldMask mask;
      derez.deserialize(mask);
      size_t num_sets;
      derez.deserialize(num_sets);
      op::FieldMaskMap<EquivalenceSet> eq_sets;
      std::map<EquivalenceSet*, DistributedID> did_map;
      std::vector<RtEvent> done_events;
      for (unsigned idx = 0; idx < num_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        EquivalenceSet* set =
            runtime->find_or_request_equivalence_set(did, ready);
        if (ready.exists())
          done_events.emplace_back(ready);
        FieldMask set_mask;
        derez.deserialize(set_mask);
        eq_sets.insert(set, set_mask);
        if (!children.empty())
          did_map.emplace(std::make_pair(set, did));
      }
      size_t num_subscriptions;
      derez.deserialize(num_subscriptions);
      op::FieldMaskMap<EqKDTree> new_subscriptions;
      for (unsigned idx = 0; idx < num_subscriptions; idx++)
      {
        EqKDTree* tree;
        derez.deserialize(tree);
        FieldMask mask;
        derez.deserialize(mask);
        new_subscriptions.insert(tree, mask);
      }
      size_t num_creations;
      derez.deserialize(num_creations);
      op::FieldMaskMap<EqKDTree> to_create;
      op::map<EqKDTree*, Domain> creation_rects;
      for (unsigned idx = 0; idx < num_creations; idx++)
      {
        EqKDTree* tree;
        derez.deserialize(tree);
        FieldMask tree_mask;
        derez.deserialize(tree_mask);
        to_create.insert(tree, tree_mask);
        derez.deserialize(creation_rects[tree]);
      }
      local::map<DistributedID, local::map<Domain, FieldMask>> temporary_srcs;
      op::map<EquivalenceSet*, op::map<Domain, FieldMask>> creation_srcs;
      size_t num_sources;
      derez.deserialize(num_sources);
      std::vector<RtEvent> ready_events;
      if (creation_target_space == runtime->address_space)
      {
        for (unsigned idx1 = 0; idx1 < num_sources; idx1++)
        {
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          EquivalenceSet* set =
              runtime->find_or_request_equivalence_set(did, ready);
          if (ready.exists())
            ready_events.emplace_back(ready);
          op::map<Domain, FieldMask>& rects = creation_srcs[set];
          size_t num_rects;
          derez.deserialize(num_rects);
          for (unsigned idx2 = 0; idx2 < num_rects; idx2++)
          {
            Domain rect;
            derez.deserialize(rect);
            derez.deserialize(rects[rect]);
          }
        }
      }
      else
      {
        for (unsigned idx1 = 0; idx1 < num_sources; idx1++)
        {
          DistributedID did;
          derez.deserialize(did);
          local::map<Domain, FieldMask>& rects = temporary_srcs[did];
          size_t num_rects;
          derez.deserialize(num_rects);
          for (unsigned idx2 = 0; idx2 < num_rects; idx2++)
          {
            Domain rect;
            derez.deserialize(rect);
            derez.deserialize(rects[rect]);
          }
        }
      }
      size_t expected_responses;
      derez.deserialize(expected_responses);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      AddressSpaceID creation_child = origin_space;
      if (!to_create.empty() && !children.empty() &&
          (creation_target_space != runtime->address_space))
      {
        std::sort(children.begin(), children.end());
        creation_child = creation_target_space;
        while (!std::binary_search(
            children.begin(), children.end(), creation_child))
        {
          legion_assert(creation_child != origin_space);
          creation_child =
              target_mapping.get_parent(origin_space, creation_child);
        }
      }
      // Send off any messages to children
      for (const AddressSpaceID& cit : children)
      {
        // Send a message back to the child node with the results
        const RtUserEvent child_event = Runtime::create_rt_user_event();
        ComputeEquivalenceSetsResponse rez;
        {
          RezCheck z(rez);
          context->pack_inner_context(rez);
          rez.serialize(source_space);
          target_mapping.pack(rez);
          for (unsigned idx = 0; idx < targets.size(); idx++)
          {
            rez.serialize(targets[idx]);
            rez.serialize(new_target_references[idx]);
          }
          rez.serialize(creation_target_space);
          rez.serialize(mask);
          rez.serialize<size_t>(eq_sets.size());
          for (op::FieldMaskMap<EquivalenceSet>::const_iterator it =
                   eq_sets.begin();
               it != eq_sets.end(); it++)
          {
            legion_assert(did_map.find(it->first) != did_map.end());
            rez.serialize(did_map[it->first]);
            rez.serialize(it->second);
          }
          rez.serialize<size_t>(new_subscriptions.size());
          for (op::FieldMaskMap<EqKDTree>::const_iterator it =
                   new_subscriptions.begin();
               it != new_subscriptions.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
          }
          // We only need to send the creation sets along the path to
          // creation_target_space
          if (cit == creation_child)
          {
            legion_assert(creation_srcs.empty());
            rez.serialize<size_t>(to_create.size());
            for (op::FieldMaskMap<EqKDTree>::const_iterator it =
                     to_create.begin();
                 it != to_create.end(); it++)
            {
              legion_assert(
                  creation_rects.find(it->first) != creation_rects.end());
              rez.serialize(it->first);
              rez.serialize(it->second);
              rez.serialize(creation_rects[it->first]);
            }
            rez.serialize<size_t>(temporary_srcs.size());
            for (local::map<DistributedID, local::map<Domain, FieldMask>>::
                     const_iterator sit = temporary_srcs.begin();
                 sit != temporary_srcs.end(); sit++)
            {
              // No need to pack a reference since we know that that
              // EqKDTree is still holding references to the creation_srcs
              // until we're done making the new equivalence sets
              rez.serialize(sit->first);
              rez.serialize<size_t>(sit->second.size());
              for (local::map<Domain, FieldMask>::const_iterator it =
                       sit->second.begin();
                   it != sit->second.end(); it++)
              {
                rez.serialize(it->first);
                rez.serialize(it->second);
              }
            }
          }
          else
          {
            rez.serialize<size_t>(0);
            rez.serialize<size_t>(0);
          }
          rez.serialize(expected_responses);
          rez.serialize(child_event);
        }
        rez.dispatch(cit);
        done_events.emplace_back(child_event);
      }
      // Wait for any ready events to be complete
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      // Find the local target
      legion_assert(target_mapping.contains(runtime->address_space));
      unsigned target_index = target_mapping.find_index(runtime->address_space);
      legion_assert(target_index < targets.size());
      targets[target_index]->record_equivalence_sets(
          context, mask, eq_sets, to_create, creation_rects, creation_srcs,
          new_subscriptions, new_target_references[target_index], source_space,
          expected_responses, done_events, target_mapping, targets,
          creation_target_space);
      if (!done_events.empty())
        Runtime::trigger_event(done_event, Runtime::merge_events(done_events));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    RtEvent InnerContext::record_output_equivalence_set(
        EqSetTracker* source, AddressSpaceID source_space, unsigned req_index,
        EquivalenceSet* set, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(regions.size() <= req_index);
      // Be very careful, you can't use find_equivalence_set_kd_tree here
      // because the tree will not be marked ready until after all the
      // output equivalence sets have registered themselves, so it's up
      // to us to make an equivalence set tree if one doesn't already
      // exist and then register ourselves with it
      LocalLock* tree_lock = nullptr;
      EqKDTree* tree = find_or_create_output_set_kd_tree(req_index, tree_lock);
      local::FieldMaskMap<EqKDTree> new_subscriptions;
      op::map<ShardID, op::map<Domain, FieldMask>> remote_shard_rects;
      unsigned references = set->set_expr->record_output_equivalence_set(
          tree, tree_lock, set, mask, source, source_space, new_subscriptions,
          remote_shard_rects);
      legion_assert(remote_shard_rects.empty());
      return report_output_registrations(
          source, source_space, references, new_subscriptions);
    }

    //--------------------------------------------------------------------------
    /*static*/ void OutputEquivalenceSetRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      InnerContext* context = InnerContext::unpack_inner_context(derez);
      EqSetTracker* source;
      derez.deserialize(source);
      AddressSpaceID source_space;
      derez.deserialize(source_space);
      unsigned req_index;
      derez.deserialize(req_index);
      DistributedID set_did;
      derez.deserialize(set_did);
      RtEvent set_ready;
      EquivalenceSet* set =
          runtime->find_or_request_equivalence_set(set_did, set_ready);
      FieldMask mask;
      derez.deserialize(mask);
      RtUserEvent recorded;
      derez.deserialize(recorded);
      if (set_ready.exists() && !set_ready.has_triggered())
        set_ready.wait();
      Runtime::trigger_event(
          recorded, context->record_output_equivalence_set(
                        source, source_space, req_index, set, mask));
      set->unpack_global_ref();
    }

    //--------------------------------------------------------------------------
    RtEvent InnerContext::report_output_registrations(
        EqSetTracker* target, AddressSpaceID target_space, unsigned references,
        local::FieldMaskMap<EqKDTree>& new_subscriptions)
    //--------------------------------------------------------------------------
    {
      if (new_subscriptions.empty())
      {
        legion_assert(references == 0);
        return RtEvent::NO_RT_EVENT;
      }
      if (target_space != runtime->address_space)
      {
        const RtUserEvent reported = Runtime::create_rt_user_event();
        OutputEquivalenceSetResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(target);
          rez.serialize(references);
          rez.serialize<size_t>(new_subscriptions.size());
          for (local::FieldMaskMap<EqKDTree>::const_iterator it =
                   new_subscriptions.begin();
               it != new_subscriptions.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
          }
          rez.serialize(reported);
        }
        rez.dispatch(target_space);
        return reported;
      }
      else
      {
        if (references > 0)
          target->add_subscription_reference(references);
        target->record_output_subscriptions(
            runtime->address_space, new_subscriptions);
        return RtEvent::NO_RT_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void OutputEquivalenceSetResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      EqSetTracker* tracker;
      derez.deserialize(tracker);
      unsigned references;
      derez.deserialize(references);
      if (references > 0)
        tracker->add_subscription_reference(references);
      size_t num_subscriptions;
      derez.deserialize(num_subscriptions);
      local::FieldMaskMap<EqKDTree> new_subscriptions;
      for (unsigned idx = 0; idx < num_subscriptions; idx++)
      {
        EqKDTree* tree;
        derez.deserialize(tree);
        FieldMask mask;
        derez.deserialize(mask);
        new_subscriptions.insert(tree, mask);
      }
      tracker->record_output_subscriptions(source, new_subscriptions);
      RtUserEvent reported;
      derez.deserialize(reported);
      Runtime::trigger_event(reported);
    }

    //--------------------------------------------------------------------------
    InnerContext::EqKDRoot::EqKDRoot(void) : tree(nullptr), lock(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    InnerContext::EqKDRoot::EqKDRoot(EqKDTree* t)
      : tree(t), lock(new LocalLock())
    //--------------------------------------------------------------------------
    {
      legion_assert(tree != nullptr);
      tree->add_reference();
    }

    //--------------------------------------------------------------------------
    InnerContext::EqKDRoot::EqKDRoot(EqKDRoot&& rhs) noexcept
      : tree(rhs.tree), lock(rhs.lock)
    //--------------------------------------------------------------------------
    {
      rhs.tree = nullptr;
      rhs.lock = nullptr;
    }

    //--------------------------------------------------------------------------
    InnerContext::EqKDRoot::~EqKDRoot(void)
    //--------------------------------------------------------------------------
    {
      // Either both nullptr or both not nullptr
      legion_assert((tree == nullptr) == (lock == nullptr));
      if (tree != nullptr)
      {
        if (tree->remove_reference())
          delete tree;
        delete lock;
      }
    }

    //--------------------------------------------------------------------------
    InnerContext::EqKDRoot& InnerContext::EqKDRoot::operator=(
        EqKDRoot&& rhs) noexcept
    //--------------------------------------------------------------------------
    {
      if (tree == nullptr)
      {
        legion_assert(lock == nullptr);
        tree = rhs.tree;
        lock = rhs.lock;
        rhs.tree = nullptr;
        rhs.lock = nullptr;
      }
      else
      {
        legion_assert(lock != nullptr);
        // Should never be overwriting one tree with another
        legion_assert(rhs.tree == nullptr);
        legion_assert(rhs.lock == nullptr);
        if (tree->remove_reference())
          delete tree;
        delete lock;
        tree = nullptr;
        lock = nullptr;
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::find_root_index_space(unsigned req_index) const
    //--------------------------------------------------------------------------
    {
      // Already holding the privilege lock from the caller
      if (req_index < regions.size())
      {
        legion_assert(
            regions[req_index].handle_type == LEGION_SINGULAR_PROJECTION);
        return regions[req_index].region.get_index_space();
      }
      else
      {
        std::map<unsigned, std::pair<LogicalRegion, bool>>::const_iterator
            finder = returnable_privileges.find(req_index);
        legion_assert(finder != returnable_privileges.end());
        return finder->second.first.get_index_space();
      }
    }

    //--------------------------------------------------------------------------
    EqKDTree* InnerContext::find_equivalence_set_kd_tree(
        unsigned req_index, LocalLock*& tree_lock,
        bool return_null_if_doesnt_exist)
    //--------------------------------------------------------------------------
    {
      // Use the privilege lock since we also need to access the created
      // requirements data structure as well in this routine
      RtEvent wait_on;
      {
        AutoLock priv_lock(privilege_lock, false /*exclusive*/);
        // If there is a guard we always need to wait on it
        std::map<unsigned, RtUserEvent>::const_iterator finder =
            pending_equivalence_set_trees.find(req_index);
        if (finder != pending_equivalence_set_trees.end())
          wait_on = finder->second;
        else
        {
          std::map<unsigned, EqKDRoot>::const_iterator finder =
              equivalence_set_trees.find(req_index);
          if (finder != equivalence_set_trees.end())
          {
            tree_lock = finder->second.lock;
            return finder->second.tree;
          }
          else if (return_null_if_doesnt_exist)
            return nullptr;
        }
      }
      IndexSpace root_space = IndexSpace::NO_SPACE;
      if (!wait_on.exists())
      {
        // Make sure we didn't lose the race
        AutoLock priv_lock(privilege_lock);
        std::map<unsigned, RtUserEvent>::iterator finder =
            pending_equivalence_set_trees.find(req_index);
        // If there's a guard always make sure we wait on it
        if (finder != pending_equivalence_set_trees.end())
        {
          // There's already a guard so someone else is making it
          if (!finder->second.exists())
            finder->second = Runtime::create_rt_user_event();
          wait_on = finder->second;
        }
        else
        {
          std::map<unsigned, EqKDRoot>::const_iterator finder =
              equivalence_set_trees.find(req_index);
          if (finder != equivalence_set_trees.end())
          {
            tree_lock = finder->second.lock;
            return finder->second.tree;
          }
          // save a guard that we're making this
          pending_equivalence_set_trees[req_index] =
              RtUserEvent::NO_RT_USER_EVENT;
          root_space = find_root_index_space(req_index);
        }
      }
      if (!wait_on.exists())
      {
        legion_assert(root_space.exists());
        // Create the equivalence set tree
        IndexSpaceNode* root = runtime->get_node(root_space);
        // Normal contexts and control replication contexts will do
        // different things here while creating the equivalence set tree
        EqKDTree* tree = create_equivalence_set_kd_tree(root);
        // Now we can save it and wake up anyone looking for it
        AutoLock priv_lock(privilege_lock);
        std::map<unsigned, EqKDRoot>::const_iterator it =
            equivalence_set_trees.emplace(req_index, EqKDRoot(tree)).first;
        tree_lock = it->second.lock;
        std::map<unsigned, RtUserEvent>::iterator finder =
            pending_equivalence_set_trees.find(req_index);
        legion_assert(finder != pending_equivalence_set_trees.end());
        if (finder->second.exists())
          Runtime::trigger_event(finder->second);
        pending_equivalence_set_trees.erase(finder);
        return tree;
      }
      else
      {
        wait_on.wait();
        AutoLock priv_lock(privilege_lock, false /*exclusive*/);
        std::map<unsigned, EqKDRoot>::const_iterator finder =
            equivalence_set_trees.find(req_index);
        legion_assert(finder != equivalence_set_trees.end());
        tree_lock = finder->second.lock;
        return finder->second.tree;
      }
    }

    //--------------------------------------------------------------------------
    EqKDTree* InnerContext::find_or_create_output_set_kd_tree(
        unsigned req_index, LocalLock*& tree_lock)
    //--------------------------------------------------------------------------
    {
      IndexSpace root_space = IndexSpace::NO_SPACE;
      {
        AutoLock priv_lock(privilege_lock, false /*exclusive*/);
        // Should always have a guard here
        legion_assert(
            pending_equivalence_set_trees.find(req_index) !=
            pending_equivalence_set_trees.end());
        std::map<unsigned, EqKDRoot>::const_iterator finder =
            equivalence_set_trees.find(req_index);
        if (finder != equivalence_set_trees.end())
        {
          tree_lock = finder->second.lock;
          return finder->second.tree;
        }
        root_space = find_root_index_space(req_index);
      }
      legion_assert(root_space.exists());
      IndexSpaceNode* root = runtime->get_node(root_space);
      EqKDTree* tree = create_equivalence_set_kd_tree(root);
      AutoLock priv_lock(privilege_lock);
      std::map<unsigned, EqKDRoot>::const_iterator finder =
          equivalence_set_trees.find(req_index);
      if (finder == equivalence_set_trees.end())
        finder = equivalence_set_trees.emplace(req_index, EqKDRoot(tree)).first;
      else
        delete tree;
      tree_lock = finder->second.lock;
      return finder->second.tree;
    }

    //--------------------------------------------------------------------------
    void InnerContext::finalize_output_eqkd_tree(unsigned req_index)
    //--------------------------------------------------------------------------
    {
      IndexSpace root_space = IndexSpace::NO_SPACE;
      {
        AutoLock priv_lock(privilege_lock);
        std::map<unsigned, RtUserEvent>::iterator finder =
            pending_equivalence_set_trees.find(req_index);
        // Should always find an existing guard for outputs
        legion_assert(finder != pending_equivalence_set_trees.end());
        // If there are no waiters or the equivalence set tree has
        // already been made then we are just done
        if (equivalence_set_trees.find(req_index) !=
            equivalence_set_trees.end())
        {
          if (finder->second.exists())
            Runtime::trigger_event(finder->second);
          pending_equivalence_set_trees.erase(finder);
          return;
        }
        // If we get here there is someone waiting for us to make
        // the tree and it hasn't been made yet, so do that now
        root_space = find_root_index_space(req_index);
      }
      // If we get here then we need to make the new tree
      IndexSpaceNode* root = runtime->get_node(root_space);
      // Normal contexts and control replication contexts will do
      // different things here while creating the equivalence set tree
      EqKDTree* tree = create_equivalence_set_kd_tree(root);
      AutoLock priv_lock(privilege_lock);
      // No one else should have made it in the interim
      legion_assert(
          equivalence_set_trees.find(req_index) == equivalence_set_trees.end());
      equivalence_set_trees.emplace(req_index, EqKDRoot(tree));
      std::map<unsigned, RtUserEvent>::iterator finder =
          pending_equivalence_set_trees.find(req_index);
      legion_assert(finder != pending_equivalence_set_trees.end());
      if (finder->second.exists())
        Runtime::trigger_event(finder->second);
      pending_equivalence_set_trees.erase(finder);
    }

    //--------------------------------------------------------------------------
    EqKDTree* InnerContext::create_equivalence_set_kd_tree(IndexSpaceNode* node)
    //--------------------------------------------------------------------------
    {
      // We can just construct this like normal
      return node->create_equivalence_set_kd_tree();
    }

    //--------------------------------------------------------------------------
    void InnerContext::refine_equivalence_sets(
        unsigned req_index, IndexSpaceNode* node,
        const FieldMask& refinement_mask, std::vector<RtEvent>& applied_events,
        bool sharded, bool first, const CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      legion_assert(!sharded);
      if ((req_index < regions.size()) && virtual_mapped[req_index])
      {
        find_parent_context()->refine_equivalence_sets(
            parent_req_indexes[req_index], node, refinement_mask,
            applied_events, sharded, false /*first*/, mapping);
        return;
      }
      LocalLock* tree_lock = nullptr;
      EqKDTree* tree = find_equivalence_set_kd_tree(req_index, tree_lock);
      node->invalidate_equivalence_set_kd_tree(
          tree, tree_lock, refinement_mask, applied_events,
          true /*move to previous*/);
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_trace_local_sets(
        unsigned req_index, const FieldMask& mask,
        std::map<EquivalenceSet*, unsigned>& current_sets, IndexSpaceNode* node,
        const CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      if (node == nullptr)
      {
        LogicalRegion region = find_logical_region(req_index);
        node = runtime->get_node(region.get_index_space());
      }
      if ((req_index < regions.size()) && virtual_mapped[req_index])
      {
        find_parent_context()->find_trace_local_sets(
            parent_req_indexes[req_index], mask, current_sets, node, mapping);
        return;
      }
      // Find the equivalence set tree for this region requirement
      LocalLock* tree_lock = nullptr;
      EqKDTree* tree = find_equivalence_set_kd_tree(req_index, tree_lock);
      node->find_trace_local_sets_kd_tree(
          tree, tree_lock, mask, req_index, get_shard_id(), current_sets);
    }

    //--------------------------------------------------------------------------
    unsigned InnerContext::find_parent_region_index(
        Operation* op, const RegionRequirement& req, unsigned index,
        bool skip_privileges, bool force_compute)
    //--------------------------------------------------------------------------
    {
      // If we're in a fixed trace that means we've already done this check
      // before and we know we've got a parent region so we'll defer actually
      // doing it unless it is really necessary
      if (!force_compute && (current_trace != nullptr) &&
          current_trace->is_fixed())
        return Operation::TRACED_PARENT_INDEX;
      // We can check the fixed region requirements without the lock
      bool found_parent = false;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        const RegionRequirement& our_req = regions[idx];
        legion_assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
        if (our_req.region != req.parent)
          continue;
        found_parent = true;
        // Check to see if the privileges cover
        if (!skip_privileges && (PRIV_ONLY(req) & (~(our_req.privilege))))
          continue;
        // Check to see if all the fields are represented
        if (our_req.privilege_fields.size() < req.privilege_fields.size())
          continue;
        bool contained = true;
        for (FieldID fid : req.privilege_fields)
        {
          if (our_req.privilege_fields.find(fid) !=
              our_req.privilege_fields.end())
            continue;
          contained = false;
          break;
        }
        if (contained)
          return idx;
      }
      const FieldSpace fs = req.parent.get_field_space();
      // Need the lock because deletions can be coming back asynchronously
      // and mutating the create requirements data structure
      AutoLock priv_lock(privilege_lock);
      std::map<LogicalRegion, std::pair<RegionRequirement, unsigned>>::iterator
          finder = created_requirements.find(req.parent);
      if (finder != created_requirements.end())
      {
        RegionRequirement& our_req = finder->second.first;
        legion_assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
        found_parent = true;
        // Check to see if the privileges cover
        if (skip_privileges || !((PRIV_ONLY(req) & (~(our_req.privilege)))))
        {
          std::map<unsigned, std::pair<LogicalRegion, bool>>::const_iterator
              returnable_finder =
                  returnable_privileges.find(finder->second.second);
          legion_assert(returnable_finder != returnable_privileges.end());
          // If this is a returnable privilege requiremnt that means
          // that we made this region so we always have privileges
          // on any fields for that region, just add them and be done
          if (returnable_finder->second.second)
          {
            our_req.privilege_fields.insert(
                req.privilege_fields.begin(), req.privilege_fields.end());
            return finder->second.second;
          }
          bool dominated = true;
          for (FieldID fid : req.privilege_fields)
          {
            if (our_req.privilege_fields.find(fid) !=
                our_req.privilege_fields.end())
              continue;
            // Check to see if this is a field we made
            // and haven't destroyed yet
            std::pair<FieldSpace, FieldID> key(fs, fid);
            if (created_fields.find(key) != created_fields.end())
            {
              // We made it so we can add it to the requirement
              // and continue on our way
              our_req.privilege_fields.insert(fid);
              continue;
            }
            if (local_fields.find(key) != local_fields.end())
            {
              // We made it so we can add it to the requirement
              // and continue on our way
              our_req.privilege_fields.insert(fid);
              continue;
            }
            // Otherwise we don't have privileges
            dominated = false;
            break;
          }
          if (dominated)
            return finder->second.second;
        }
      }
      if (!found_parent)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Unable to find a 'parent' region " << req.parent
              << " in any of the region requirements of parent task "
              << get_task_name() << " (UID: " << get_unique_id()
              << ") to serve as the origin "
              << "of privileges for region requirement " << index << " of "
              << op->get_logging_name() << " (UID: " << op->get_unique_op_id()
              << "). All region requirements have to derive their privileges "
              << "from one region requirement of the parent task.";
        error.raise();
      }
      // Method of last resort, check to see if we made all the fields
      // if we did, then we can make a new requirement for all the fields
      for (const FieldID& it : req.privilege_fields)
      {
        const std::pair<FieldSpace, FieldID> key(fs, it);
        if ((created_fields.find(key) == created_fields.end()) &&
            (local_fields.find(key) == local_fields.end()))
        {
          // Raise an exception
          FieldSpaceNode* node = runtime->get_node(fs);
          const void* name = nullptr;
          size_t name_size = 0;
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          if (node->retrieve_semantic_information(
                  it, LEGION_NAME_SEMANTIC_TAG, name, name_size,
                  true /*can fail*/, false /*wait until*/))
          {
            std::string_view field_name((const char*)name, name_size);
            error << "Unable to find privileges for field " << field_name
                  << " of region requirement " << index << " of "
                  << op->get_logging_name()
                  << " (UID: " << op->get_unique_op_id() << ") in parent task "
                  << get_task_name() << " (UID: " << get_unique_id() << "). "
                  << "All fields must derive their region requirements from "
                  << "one of the region requirements of the parent task or "
                  << "from the creation of a region or allocation of a field "
                  << "(without a corresponding deletion) in the parent task.";
          }
          else
            error << "Unable to find privileges for field " << it
                  << " of region requirement " << index << " of "
                  << op->get_logging_name()
                  << " (UID: " << op->get_unique_op_id() << ") in parent task "
                  << get_task_name() << " (UID: " << get_unique_id() << "). "
                  << "All fields must derive their region requirements from "
                  << "one of the region requirements of the parent task or "
                  << "from the creation of a region or allocation of a field "
                  << "(without a corresponding deletion) in the parent task.";
          error.raise();
        }
      }
      // If we get here then we can make a new requirement
      // which has non-returnable privileges
      // Get the top level region for the region tree
      RegionNode* top = runtime->get_tree(req.parent.tree_did);
      const unsigned result = next_created_index++;
      RegionRequirement& new_req =
          created_requirements
              .emplace(
                  top->handle, std::make_pair(
                                   RegionRequirement(
                                       top->handle, LEGION_READ_WRITE,
                                       LEGION_EXCLUSIVE, top->handle),
                                   result))
              .first->second.first;
      TaskOp::log_requirement(get_unique_id(), result, new_req);
      // Add our fields
      new_req.privilege_fields.insert(
          req.privilege_fields.begin(), req.privilege_fields.end());
      // This is not a returnable privilege requirement
      returnable_privileges.emplace(result, std::make_pair(top->handle, false));
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion InnerContext::find_logical_region(unsigned index) const
    //--------------------------------------------------------------------------
    {
      if (index < regions.size())
        return regions[index].region;
      AutoLock priv_lock(privilege_lock, false /*exclusive*/);
      std::map<unsigned, std::pair<LogicalRegion, bool>>::const_iterator
          finder = returnable_privileges.find(index);
      legion_assert(finder != returnable_privileges.end());
      return finder->second.first;
    }

    //--------------------------------------------------------------------------
    InnerContext* InnerContext::find_parent_physical_context(unsigned index)
    //--------------------------------------------------------------------------
    {
      legion_assert(regions.size() <= virtual_mapped.size());
      legion_assert(regions.size() <= parent_req_indexes.size());
      if (index < regions.size())
      {
        // See if it is virtual mapped
        if (virtual_mapped[index])
          return find_parent_context()->find_parent_physical_context(
              parent_req_indexes[index]);
        else  // We mapped a physical instance so we're it
          return this;
      }
      else  // We created it
      {
        // Check to see if this has returnable privileges or not
        // If they are not returnable, then we can just be the
        // context for the handling the meta-data management,
        // otherwise if they are returnable then the top-level
        // context has to provide global guidance about which
        // node manages the meta-data.
        AutoLock priv_lock(privilege_lock, false /*exclusive*/);
        std::map<unsigned, std::pair<LogicalRegion, bool>>::const_iterator
            finder = returnable_privileges.find(index);
        if ((finder != returnable_privileges.end()) && !finder->second.second)
          return this;
      }
      return find_top_context();
    }

    //--------------------------------------------------------------------------
    InnerContext* InnerContext::find_top_context(InnerContext* previous)
    //--------------------------------------------------------------------------
    {
      TaskContext* parent = find_parent_context();
      if (parent != nullptr)
        return parent->find_top_context(this);
      legion_assert(previous != nullptr);
      return previous;
    }

    //--------------------------------------------------------------------------
    void InnerContext::pack_remote_context(
        Serializer& rez, AddressSpaceID target, bool replicate)
    //--------------------------------------------------------------------------
    {
      legion_assert(owner_task != nullptr);
      rez.serialize(depth);
      // See if we need to pack up base task information
      owner_task->pack_external_task(rez, target);
      legion_assert(regions.size() <= parent_req_indexes.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        rez.serialize(parent_req_indexes[idx]);
      // Pack up our virtual mapping information
      std::vector<unsigned> virtual_indexes;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (virtual_mapped[idx])
          virtual_indexes.emplace_back(idx);
      }
      rez.serialize<size_t>(virtual_indexes.size());
      for (unsigned idx = 0; idx < virtual_indexes.size(); idx++)
        rez.serialize(virtual_indexes[idx]);
      rez.serialize(find_parent_context()->did);
      context_coordinates.serialize(rez);
      Provenance* provenance = owner_task->get_provenance();
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
      rez.serialize(get_unique_id());
      // Finally pack the local field infos
      AutoLock local_lock(local_field_lock, false /*exclusive*/);
      rez.serialize<size_t>(local_field_infos.size());
      for (const std::map<FieldSpace, std::vector<LocalFieldInfo>>::value_type&
               it : local_field_infos)
      {
        rez.serialize(it.first);
        rez.serialize<size_t>(it.second.size());
        for (unsigned idx = 0; idx < it.second.size(); idx++)
          rez.serialize(it.second[idx]);
      }
      rez.serialize<bool>(concurrent_context);
      rez.serialize<bool>(replicate);
    }

    //--------------------------------------------------------------------------
    void InnerContext::compute_task_tree_coordinates(
        TaskTreeCoordinates& coordinates) const
    //--------------------------------------------------------------------------
    {
      // Reserve an extra level for the common case
      coordinates.reserve(context_coordinates.size() + 1);
      coordinates = context_coordinates;
    }

    //--------------------------------------------------------------------------
    void InnerContext::return_resources(
        ResourceTracker* target, uint64_t return_index,
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      if (created_regions.empty() && deleted_regions.empty() &&
          created_fields.empty() && deleted_fields.empty() &&
          created_field_spaces.empty() && latent_field_spaces.empty() &&
          deleted_field_spaces.empty() && created_index_spaces.empty() &&
          deleted_index_spaces.empty() && created_index_partitions.empty() &&
          deleted_index_partitions.empty())
        return;
      target->receive_resources(
          return_index, created_regions, deleted_regions, created_fields,
          deleted_fields, created_field_spaces, latent_field_spaces,
          deleted_field_spaces, created_index_spaces, deleted_index_spaces,
          created_index_partitions, deleted_index_partitions, preconditions);
      created_regions.clear();
      deleted_regions.clear();
      created_fields.clear();
      deleted_fields.clear();
      created_field_spaces.clear();
      latent_field_spaces.clear();
      deleted_field_spaces.clear();
      created_index_spaces.clear();
      deleted_index_spaces.clear();
      created_index_partitions.clear();
      deleted_index_partitions.clear();
    }

    //--------------------------------------------------------------------------
    void InnerContext::pack_return_resources(
        Serializer& rez, uint64_t return_index)
    //--------------------------------------------------------------------------
    {
      pack_resources_return(rez, return_index);
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space(
        const Domain& bounds, bool take_ownership, TypeTag type_tag,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      return create_index_space_internal(
          bounds, type_tag, provenance, take_ownership);
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space(
        const std::vector<DomainPoint>& points, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      switch (points[0].get_dim())
      {
#define DIMFUNC(DIM)                                                         \
  case DIM:                                                                  \
    {                                                                        \
      std::vector<Realm::Point<DIM, coord_t>> realm_points(points.size());   \
      for (unsigned idx = 0; idx < points.size(); idx++)                     \
        realm_points[idx] = Point<DIM, coord_t>(points[idx]);                \
      const DomainT<DIM, coord_t> realm_is(                                  \
          (Realm::IndexSpace<DIM, coord_t>(realm_points)));                  \
      const Domain bounds(realm_is);                                         \
      return create_index_space_internal(                                    \
          bounds, NT_TemplateHelper::encode_tag<DIM, coord_t>(), provenance, \
          true);                                                             \
    }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          std::abort();
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space(
        const std::vector<Domain>& rects, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      switch (rects[0].get_dim())
      {
#define DIMFUNC(DIM)                                                         \
  case DIM:                                                                  \
    {                                                                        \
      std::vector<Realm::Rect<DIM, coord_t>> realm_rects(rects.size());      \
      for (unsigned idx = 0; idx < rects.size(); idx++)                      \
        realm_rects[idx] = Rect<DIM, coord_t>(rects[idx]);                   \
      const DomainT<DIM, coord_t> realm_is(                                  \
          (Realm::IndexSpace<DIM, coord_t>(realm_rects)));                   \
      const Domain bounds(realm_is);                                         \
      return create_index_space_internal(                                    \
          bounds, NT_TemplateHelper::encode_tag<DIM, coord_t>(), provenance, \
          true);                                                             \
    }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          std::abort();
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_internal(
        const Domain& bounds, TypeTag type_tag, Provenance* provenance,
        bool take_ownership)
    //--------------------------------------------------------------------------
    {
      IndexSpace handle(
          runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), type_tag);
      LegionSpy::log_top_index_space(
          handle.get_id(), runtime->address_space,
          (provenance == nullptr) ? std::string_view() : provenance->human);
      runtime->create_index_space(handle, bounds, take_ownership, provenance);
      register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::find_index_launch_space(
        const Domain& domain, Provenance* provenance, const bool take_ownership)
    //--------------------------------------------------------------------------
    {
      std::map<Domain, IndexSpace>::const_iterator finder =
          index_launch_spaces.find(domain);
      if (finder != index_launch_spaces.end())
      {
        if (take_ownership && !domain.dense())
        {
          Domain copy = domain;
          copy.destroy();
        }
        return finder->second;
      }
      IndexSpace result;
      switch (domain.get_dim())
      {
#define DIMFUNC(DIM)                                                         \
  case DIM:                                                                  \
    {                                                                        \
      result = create_index_space_internal(                                  \
          domain, NT_TemplateHelper::encode_tag<DIM, coord_t>(), provenance, \
          take_ownership);                                                   \
      break;                                                                 \
    }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          std::abort();
      }
      index_launch_spaces[domain] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_unbound_index_space(
        TypeTag type_tag, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      return create_index_space_internal(
          Domain::NO_DOMAIN, type_tag, provenance, true);
    }

    //--------------------------------------------------------------------------
    void InnerContext::create_shared_ownership(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return;
      // Check to see if this is a top-level index space, if not then
      // we shouldn't even be destroying it
      if (!runtime->is_top_level_index_space(handle))
      {
        Error err(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        err << "Illegal call to create shared ownership for index space "
            << handle << " in task " << get_task_name() << " (UID "
            << get_unique_id()
            << ") which is not a top-level index space. Legion only permits "
            << "top-level index spaces to have shared ownership.";
        err.raise();
      }
      runtime->create_shared_ownership(handle);
      AutoLock priv_lock(privilege_lock);
      std::map<IndexSpace, unsigned>::iterator finder =
          created_index_spaces.find(handle);
      if (finder != created_index_spaces.end())
        finder->second++;
      else
        created_index_spaces[handle] = 1;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::union_index_spaces(
        const std::vector<IndexSpace>& spaces, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (spaces.empty())
        return IndexSpace::NO_SPACE;
      bool none_exists = true;
      for (const IndexSpace& it : spaces)
      {
        if (none_exists && it.exists())
          none_exists = false;
        if (spaces[0].get_type_tag() != it.get_type_tag())
        {
          Error err(LEGION_DYNAMIC_TYPE_EXCEPTION);
          err << "Dynamic type mismatch in 'union_index_spaces' performed in "
                 "task "
              << get_task_name() << " (UID " << get_unique_id() << ")";
          err.raise();
        }
      }
      if (none_exists)
        return IndexSpace::NO_SPACE;
      const IndexSpace handle(
          runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), spaces[0].get_type_tag());
      runtime->create_union_space(handle, provenance, spaces);
      register_index_space_creation(handle);
      LegionSpy::log_top_index_space(
          handle.get_id(), runtime->address_space,
          (provenance == nullptr) ? std::string_view() : provenance->human);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::intersect_index_spaces(
        const std::vector<IndexSpace>& spaces, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (spaces.empty())
        return IndexSpace::NO_SPACE;
      bool none_exists = true;
      for (const IndexSpace& it : spaces)
      {
        if (none_exists && it.exists())
          none_exists = false;
        if (spaces[0].get_type_tag() != it.get_type_tag())
        {
          Error err(LEGION_DYNAMIC_TYPE_EXCEPTION);
          err << "Dynamic type mismatch in 'intersect_index_spaces' performed "
                 "in task "
              << get_task_name() << " (UID " << get_unique_id() << ")";
          err.raise();
        }
      }
      if (none_exists)
        return IndexSpace::NO_SPACE;
      const IndexSpace handle(
          runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), spaces[0].get_type_tag());
      runtime->create_intersection_space(handle, provenance, spaces);
      register_index_space_creation(handle);
      LegionSpy::log_top_index_space(
          handle.get_id(), runtime->address_space,
          (provenance == nullptr) ? std::string_view() : provenance->human);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::subtract_index_spaces(
        IndexSpace left, IndexSpace right, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (!left.exists())
        return IndexSpace::NO_SPACE;
      if (right.exists() && left.get_type_tag() != right.get_type_tag())
      {
        Error err(LEGION_DYNAMIC_TYPE_EXCEPTION);
        err << "Dynamic type mismatch in 'create_difference_spaces' performed "
               "in task "
            << get_task_name() << " (UID " << get_unique_id() << ")";
        err.raise();
      }
      const IndexSpace handle(
          runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), left.get_type_tag());
      runtime->create_difference_space(handle, provenance, left, right);
      register_index_space_creation(handle);
      LegionSpy::log_top_index_space(
          handle.get_id(), runtime->address_space,
          (provenance == nullptr) ? std::string_view() : provenance->human);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space(
        const Future& future, TypeTag type_tag, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      IndexSpace handle(
          runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), type_tag);
      LegionSpy::log_top_index_space(
          handle.get_id(), runtime->address_space,
          (provenance == nullptr) ? std::string_view() : provenance->human);
      // Get a new creation operation
      CreationOp* creator_op = runtime->get_operation<CreationOp>();
      const ApEvent ready = creator_op->get_completion_event();
      IndexSpaceNode* node = runtime->create_node(
          handle, Domain::NO_DOMAIN, true /*task ownership*/,
          nullptr /*parent*/, 0 /*color*/, RtEvent::NO_RT_EVENT, provenance,
          ready, 0 /*expr id*/, nullptr /*collective mapping*/,
          true /*add root reference*/);
      creator_op->initialize_index_space(this, node, future, provenance);
      register_index_space_creation(handle);
      add_to_dependence_queue(creator_op);
      return handle;
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_index_space(
        IndexSpace handle, const bool unordered, const bool recurse,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return;
      // Check to see if this is a top-level index space, if not then
      // we shouldn't even be destroying it
      if (!runtime->is_top_level_index_space(handle))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal call to destroy index space " << handle << " in task "
              << get_task_name() << " (UID " << get_unique_id()
              << ") which is not a top-level index space. Legion only permits "
              << "top-level index spaces to be destroyed.";
        error.raise();
      }
      // Check to see if this is one that we should be allowed to destory
      std::vector<IndexPartition> sub_partitions;
      {
        AutoLock priv_lock(privilege_lock);
        std::map<IndexSpace, unsigned>::iterator finder =
            created_index_spaces.find(handle);
        if (finder == created_index_spaces.end())
        {
          // If we didn't make the index space in this context, just
          // record it and keep going, it will get handled later
          deleted_index_spaces.emplace_back(
              DeletedIndexSpace(handle, recurse, provenance));
          return;
        }
        else
        {
          legion_assert(finder->second > 0);
          if (--finder->second == 0)
            created_index_spaces.erase(finder);
          else
            return;
        }
        if (recurse)
        {
          // Also remove any index partitions for this index space tree
          for (std::map<IndexPartition, unsigned>::iterator it =
                   created_index_partitions.begin();
               it != created_index_partitions.end();
               /*nothing*/)
          {
            if (it->first.get_tree_id() == handle.get_tree_id())
            {
              sub_partitions.emplace_back(it->first);
              legion_assert(it->second > 0);
              if (--it->second == 0)
              {
                std::map<IndexPartition, unsigned>::iterator to_delete = it++;
                created_index_partitions.erase(to_delete);
              }
              else
                it++;
            }
            else
              it++;
          }
        }
      }
      DeletionOp* op = runtime->get_operation<DeletionOp>();
      op->initialize_index_space_deletion(
          this, handle, sub_partitions, unordered, provenance);
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error
              << "Illegal unordered index space deletion performed after task "
              << get_task_name() << " (UID " << get_unique_id()
              << ") has finished executing. All unordered operations must "
              << "be performed before the end of the execution of the parent "
                 "task.";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::create_shared_ownership(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return;
      runtime->create_shared_ownership(handle);
      AutoLock priv_lock(privilege_lock);
      std::map<IndexPartition, unsigned>::iterator finder =
          created_index_partitions.find(handle);
      if (finder != created_index_partitions.end())
        finder->second++;
      else
        created_index_partitions[handle] = 1;
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_index_partition(
        IndexPartition handle, const bool unordered, const bool recurse,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return;
      std::vector<IndexPartition> sub_partitions;
      {
        AutoLock priv_lock(privilege_lock);
        std::map<IndexPartition, unsigned>::iterator finder =
            created_index_partitions.find(handle);
        if (finder != created_index_partitions.end())
        {
          legion_assert(finder->second > 0);
          if (--finder->second == 0)
            created_index_partitions.erase(finder);
          else
            return;
          if (recurse)
          {
            // Remove any other partitions that this partition dominates
            for (std::map<IndexPartition, unsigned>::iterator it =
                     created_index_partitions.begin();
                 it != created_index_partitions.end();
                 /*nothing*/)
            {
              if ((handle.get_tree_id() == it->first.get_tree_id()) &&
                  runtime->is_dominated_tree_only(it->first, handle))
              {
                sub_partitions.emplace_back(it->first);
                legion_assert(it->second > 0);
                if (--it->second == 0)
                {
                  std::map<IndexPartition, unsigned>::iterator to_delete = it++;
                  created_index_partitions.erase(to_delete);
                }
                else
                  it++;
              }
              else
                it++;
            }
          }
        }
        else
        {
          // If we didn't make the partition, record it and keep going
          deleted_index_partitions.emplace_back(
              DeletedPartition(handle, recurse, provenance));
          return;
        }
      }
      DeletionOp* op = runtime->get_operation<DeletionOp>();
      op->initialize_index_part_deletion(
          this, handle, sub_partitions, unordered, provenance);
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Illegal unordered index partition deletion performed after "
                   "task "
                << get_task_name() << " (UID " << get_unique_id()
                << ") has finished executing. All unordered operations must "
                << "be performed before the end of the execution of the parent "
                   "task.";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_equal_partition(
        IndexSpace parent, IndexSpace color_space, size_t granularity,
        Color color, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(
          runtime->get_unique_index_partition_id(), parent.get_tree_id(),
          parent.get_type_tag());
      LegionColor partition_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      part_op->initialize_equal_partition(this, pid, granularity, provenance);
      RtEvent safe = create_pending_partition_internal(
          pid, parent, color_space, partition_color,
          LEGION_DISJOINT_COMPLETE_KIND, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_weights(
        IndexSpace parent, const FutureMap& weights, IndexSpace color_space,
        size_t granularity, Color color, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      const IndexPartition pid(
          runtime->get_unique_index_partition_id(), parent.get_tree_id(),
          parent.get_type_tag());
      LegionColor partition_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      part_op->initialize_weight_partition(
          this, pid, weights, granularity, provenance);
      const RtEvent safe = create_pending_partition_internal(
          pid, parent, color_space, partition_color,
          LEGION_DISJOINT_COMPLETE_KIND, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_union(
        IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
        IndexSpace color_space, PartitionKind kind, Color color,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind);
      IndexPartition pid(
          runtime->get_unique_index_partition_id(), parent.get_tree_id(),
          parent.get_type_tag());
      if (parent.get_tree_id() != handle1.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "IndexPartition " << handle1.get_id()
              << " is not part of the same "
              << "index tree as IndexSpace " << parent.get_id() << " in create "
              << "partition by union!";
        error.raise();
      }
      if (parent.get_tree_id() != handle2.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "IndexPartition " << handle2.get_id()
              << " is not part of the same "
              << "index tree as IndexSpace " << parent.get_id() << " in create "
              << "partition by union!";
        error.raise();
      }
      LegionColor partition_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      part_op->initialize_union_partition(
          this, pid, handle1, handle2, provenance);
      // If either partition is aliased the result is aliased
      if ((kind == LEGION_COMPUTE_KIND) ||
          (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        // If one of these partitions is aliased then the result is aliased
        IndexPartNode* p1 = runtime->get_node(handle1);
        if (p1->is_disjoint(true /*from app*/))
        {
          IndexPartNode* p2 = runtime->get_node(handle2);
          if (!p2->is_disjoint(true /*from app*/))
          {
            if (kind == LEGION_COMPUTE_KIND)
              kind = LEGION_ALIASED_KIND;
            else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
              kind = LEGION_ALIASED_COMPLETE_KIND;
            else
              kind = LEGION_ALIASED_INCOMPLETE_KIND;
          }
        }
        else
        {
          if (kind == LEGION_COMPUTE_KIND)
            kind = LEGION_ALIASED_KIND;
          else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
            kind = LEGION_ALIASED_COMPLETE_KIND;
          else
            kind = LEGION_ALIASED_INCOMPLETE_KIND;
        }
      }
      RtEvent safe = create_pending_partition_internal(
          pid, parent, color_space, partition_color, kind, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_intersection(
        IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
        IndexSpace color_space, PartitionKind kind, Color color,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind);
      IndexPartition pid(
          runtime->get_unique_index_partition_id(), parent.get_tree_id(),
          parent.get_type_tag());
      if (parent.get_tree_id() != handle1.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "IndexPartition " << handle1.get_id()
              << " is not part of the same "
              << "index tree as IndexSpace " << parent.get_id()
              << " in create partition by "
              << "intersection!";
        error.raise();
      }
      if (parent.get_tree_id() != handle2.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "IndexPartition " << handle2.get_id()
              << " is not part of the same "
              << "index tree as IndexSpace " << parent.get_id()
              << " in create partition by "
              << "intersection!";
        error.raise();
      }
      LegionColor partition_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      part_op->initialize_intersection_partition(
          this, pid, handle1, handle2, provenance);
      // If either partition is disjoint then the result is disjoint
      if ((kind == LEGION_COMPUTE_KIND) ||
          (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        IndexPartNode* p1 = runtime->get_node(handle1);
        if (!p1->is_disjoint(true /*from app*/))
        {
          IndexPartNode* p2 = runtime->get_node(handle2);
          if (p2->is_disjoint(true /*from app*/))
          {
            if (kind == LEGION_COMPUTE_KIND)
              kind = LEGION_DISJOINT_KIND;
            else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
              kind = LEGION_DISJOINT_COMPLETE_KIND;
            else
              kind = LEGION_DISJOINT_INCOMPLETE_KIND;
          }
        }
        else
        {
          if (kind == LEGION_COMPUTE_KIND)
            kind = LEGION_DISJOINT_KIND;
          else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
            kind = LEGION_DISJOINT_COMPLETE_KIND;
          else
            kind = LEGION_DISJOINT_INCOMPLETE_KIND;
        }
      }
      RtEvent safe = create_pending_partition_internal(
          pid, parent, color_space, partition_color, kind, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_intersection(
        IndexSpace parent, IndexPartition partition, PartitionKind kind,
        Color color, bool dominates, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind);
      IndexPartition pid(
          runtime->get_unique_index_partition_id(), parent.get_tree_id(),
          parent.get_type_tag());
      if (parent.get_type_tag() != partition.get_type_tag())
      {
        Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
        error << "IndexPartition " << partition.get_id()
              << " does not have the same type as the parent index space "
              << parent.get_id() << " in task " << get_task_name() << " (UID "
              << get_unique_id() << ")";
        error.raise();
      }
      LegionColor partition_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      part_op->initialize_intersection_partition(
          this, pid, partition, dominates, provenance);
      IndexPartNode* part_node = runtime->get_node(partition);
      // See if we can determine disjointness if we weren't told
      if ((kind == LEGION_COMPUTE_KIND) ||
          (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        if (part_node->is_disjoint(true /*from app*/))
        {
          if (kind == LEGION_COMPUTE_KIND)
            kind = LEGION_DISJOINT_KIND;
          else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
            kind = LEGION_DISJOINT_COMPLETE_KIND;
          else
            kind = LEGION_DISJOINT_INCOMPLETE_KIND;
        }
      }
      RtEvent safe = create_pending_partition_internal(
          pid, parent, part_node->color_space->handle, partition_color, kind,
          provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_difference(
        IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
        IndexSpace color_space, PartitionKind kind, Color color,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind);
      IndexPartition pid(
          runtime->get_unique_index_partition_id(), parent.get_tree_id(),
          parent.get_type_tag());
      if (parent.get_tree_id() != handle1.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "IndexPartition " << handle1.get_id()
              << " is not part of the same "
              << "index tree as IndexSpace " << parent.get_id() << " in create "
              << "partition by difference!";
        error.raise();
      }
      if (parent.get_tree_id() != handle2.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "IndexPartition " << handle2.get_id()
              << " is not part of the same "
              << "index tree as IndexSpace " << parent.get_id() << " in create "
              << "partition by difference!";
        error.raise();
      }
      LegionColor partition_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      part_op->initialize_difference_partition(
          this, pid, handle1, handle2, provenance);
      // If the left-hand-side is disjoint the result is disjoint
      if ((kind == LEGION_COMPUTE_KIND) ||
          (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        IndexPartNode* p1 = runtime->get_node(handle1);
        if (p1->is_disjoint(true /*from app*/))
        {
          if (kind == LEGION_COMPUTE_KIND)
            kind = LEGION_DISJOINT_KIND;
          else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
            kind = LEGION_DISJOINT_COMPLETE_KIND;
          else
            kind = LEGION_DISJOINT_INCOMPLETE_KIND;
        }
      }
      RtEvent safe = create_pending_partition_internal(
          pid, parent, color_space, partition_color, kind, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    Color InnerContext::create_cross_product_partitions(
        IndexPartition handle1, IndexPartition handle2,
        std::map<IndexSpace, IndexPartition>& handles, PartitionKind kind,
        Color color, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (handle1.get_tree_id() != handle2.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "IndexPartition " << handle1.get_id()
              << " is not part of the same "
              << "index tree as IndexPartition " << handle2.get_id()
              << " in create "
              << "cross product partitions!";
        error.raise();
      }
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind);
      LegionColor partition_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      std::set<RtEvent> safe_events;
      create_pending_cross_product_internal(
          handle1, handle2, handles, kind, provenance, partition_color,
          safe_events);
      part_op->initialize_cross_product(
          this, handle1, handle2, partition_color, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      if (!safe_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(safe_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      if (runtime->verify_partitions)
      {
        Domain color_space = runtime->get_index_partition_color_space(handle1);
        // This code will only work if the color space has type coord_t
        TypeTag type_tag;
        switch (color_space.get_dim())
        {
#define DIMFUNC(DIM)                                            \
  case DIM:                                                     \
    {                                                           \
      type_tag = NT_TemplateHelper::encode_tag<DIM, coord_t>(); \
      break;                                                    \
    }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            std::abort();
        }
        for (Domain::DomainPointIterator itr(color_space); itr; itr++)
        {
          IndexSpace subspace;
          switch (color_space.get_dim())
          {
#define DIMFUNC(DIM)                                                 \
  case DIM:                                                          \
    {                                                                \
      const Point<DIM, coord_t> p(itr.p);                            \
      subspace = runtime->get_index_subspace(handle1, &p, type_tag); \
      break;                                                         \
    }
            LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
            default:
              std::abort();
          }
          IndexPartition part =
              runtime->get_index_partition(subspace, partition_color);
          verify_partition(part, verify_kind, __func__);
        }
      }
      return partition_color;
    }

    //--------------------------------------------------------------------------
    void InnerContext::create_association(
        LogicalRegion domain, LogicalRegion domain_parent, FieldID domain_fid,
        IndexSpace range, MapperID id, MappingTagID tag,
        const UntypedBuffer& marg, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      DependentPartitionOp* part_op =
          runtime->get_operation<DependentPartitionOp>();
      part_op->initialize_by_association(
          this, domain, domain_parent, domain_fid, range, id, tag, marg,
          provenance);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
        {
          Warning warning;
          warning
              << "Runtime is unmapping and remapping physical regions around "
              << "create_association call in task " << get_task_name()
              << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_restricted_partition(
        IndexSpace parent, IndexSpace color_space, const void* transform,
        size_t transform_size, const void* extent, size_t extent_size,
        PartitionKind part_kind, Color color, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      IndexPartition pid(
          runtime->get_unique_index_partition_id(), parent.get_tree_id(),
          parent.get_type_tag());
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      part_op->initialize_restricted_partition(
          this, pid, transform, transform_size, extent, extent_size,
          provenance);
      RtEvent safe = create_pending_partition_internal(
          pid, parent, color_space, part_color, part_kind, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_domain(
        IndexSpace parent, const FutureMap& domains, IndexSpace color_space,
        bool perform_intersections, PartitionKind part_kind, Color color,
        Provenance* provenance, bool skip_check)
    //--------------------------------------------------------------------------
    {
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      IndexPartition pid(
          runtime->get_unique_index_partition_id(), parent.get_tree_id(),
          parent.get_type_tag());
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      part_op->initialize_by_domain(
          this, pid, domains, perform_intersections, provenance);
      RtEvent safe = create_pending_partition_internal(
          pid, parent, color_space, part_color, part_kind, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_field(
        LogicalRegion handle, LogicalRegion parent_priv, FieldID fid,
        IndexSpace color_space, Color color, MapperID id, MappingTagID tag,
        PartitionKind part_kind, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // Partition by field is disjoint by construction
      PartitionKind verify_kind = LEGION_DISJOINT_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      IndexSpace parent = handle.get_index_space();
      IndexPartition pid(
          runtime->get_unique_index_partition_id(), parent.get_tree_id(),
          parent.get_type_tag());
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp* part_op =
          runtime->get_operation<DependentPartitionOp>();
      RtEvent safe = create_pending_partition_internal(
          pid, parent, color_space, part_color, part_kind, provenance);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_field(
          this, pid, handle, parent_priv, color_space, fid, id, tag, marg,
          provenance);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
        {
          Warning warning;
          warning
              << "Runtime is unmapping and remapping physical regions around "
              << "create_partition_by_field call in task " << get_task_name()
              << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_image(
        IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
        FieldID fid, IndexSpace color_space, PartitionKind part_kind,
        Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      IndexPartition pid(
          runtime->get_unique_index_partition_id(), handle.get_tree_id(),
          handle.get_type_tag());
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp* part_op =
          runtime->get_operation<DependentPartitionOp>();
      RtEvent safe = create_pending_partition_internal(
          pid, handle, color_space, part_color, part_kind, provenance);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_image(
          this, pid, handle, projection, parent, fid, id, tag, marg,
          provenance);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
        {
          Warning warning;
          warning
              << "Runtime is unmapping and remapping physical regions around "
              << "create_partition_by_image call in task " << get_task_name()
              << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_image_range(
        IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
        FieldID fid, IndexSpace color_space, PartitionKind part_kind,
        Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      IndexPartition pid(
          runtime->get_unique_index_partition_id(), handle.get_tree_id(),
          handle.get_type_tag());
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp* part_op =
          runtime->get_operation<DependentPartitionOp>();
      RtEvent safe = create_pending_partition_internal(
          pid, handle, color_space, part_color, part_kind, provenance);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_image_range(
          this, pid, handle, projection, parent, fid, id, tag, marg,
          provenance);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
        {
          Warning warning;
          warning
              << "Runtime is unmapping and remapping physical regions around "
              << "create_partition_by_image_range call in task "
              << get_task_name() << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_preimage(
        IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
        FieldID fid, IndexSpace color_space, PartitionKind part_kind,
        Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      IndexPartition pid(
          runtime->get_unique_index_partition_id(),
          handle.get_index_space().get_tree_id(), handle.get_type_tag());
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp* part_op =
          runtime->get_operation<DependentPartitionOp>();
      // If the source of the preimage is disjoint then the result is disjoint
      // Note this only applies here and not to range
      if ((part_kind == LEGION_COMPUTE_KIND) ||
          (part_kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        IndexPartNode* p = runtime->get_node(projection);
        if (p->is_disjoint(true /*from app*/))
        {
          if (part_kind == LEGION_COMPUTE_KIND)
            part_kind = LEGION_DISJOINT_KIND;
          else if (part_kind == LEGION_COMPUTE_COMPLETE_KIND)
            part_kind = LEGION_DISJOINT_COMPLETE_KIND;
          else
            part_kind = LEGION_DISJOINT_INCOMPLETE_KIND;
        }
      }
      RtEvent safe = create_pending_partition_internal(
          pid, handle.get_index_space(), color_space, part_color, part_kind,
          provenance);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_preimage(
          this, pid, projection, handle, parent, fid, id, tag, marg,
          provenance);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
        {
          Warning warning;
          warning
              << "Runtime is unmapping and remapping physical regions around "
              << "create_partition_by_preimage call in task " << get_task_name()
              << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_preimage_range(
        IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
        FieldID fid, IndexSpace color_space, PartitionKind part_kind,
        Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      IndexPartition pid(
          runtime->get_unique_index_partition_id(),
          handle.get_index_space().get_tree_id(), handle.get_type_tag());
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp* part_op =
          runtime->get_operation<DependentPartitionOp>();
      RtEvent safe = create_pending_partition_internal(
          pid, handle.get_index_space(), color_space, part_color, part_kind,
          provenance);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_preimage_range(
          this, pid, projection, handle, parent, fid, id, tag, marg,
          provenance);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
        {
          Warning warning;
          warning
              << "Runtime is unmapping and remapping physical regions around "
              << "create_partition_by_preimage_range call in task "
              << get_task_name() << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_pending_partition(
        IndexSpace parent, IndexSpace color_space, PartitionKind part_kind,
        Color color, Provenance* provenance, bool trust)
    //--------------------------------------------------------------------------
    {
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions && !trust)
        SWAP_PART_KINDS(verify_kind, part_kind);
      IndexPartition pid(
          runtime->get_unique_index_partition_id(), parent.get_tree_id(),
          parent.get_type_tag());
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      RtEvent safe = create_pending_partition_internal(
          pid, parent, color_space, part_color, part_kind, provenance);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions && !trust)
      {
        // We can't block to check this here because the user needs
        // control back in order to fill in the pieces of the partitions
        // so just launch a meta-task to check it when we can
        VerifyPartitionArgs args(this, pid, verify_kind, __func__);
        runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY);
      }
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_union(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, const std::vector<IndexSpace>& handles,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      IndexSpace result = instantiate_subspace(parent, realm_color, type_tag);
      part_op->initialize_index_space_union(this, result, handles, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_union(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, IndexPartition handle, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      IndexSpace result = instantiate_subspace(parent, realm_color, type_tag);
      part_op->initialize_index_space_union(this, result, handle, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_intersection(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, const std::vector<IndexSpace>& handles,
        Provenance* prov)
    //--------------------------------------------------------------------------
    {
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      IndexSpace result = instantiate_subspace(parent, realm_color, type_tag);
      part_op->initialize_index_space_intersection(this, result, handles, prov);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_intersection(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, IndexPartition handle, Provenance* prov)
    //--------------------------------------------------------------------------
    {
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      IndexSpace result = instantiate_subspace(parent, realm_color, type_tag);
      part_op->initialize_index_space_intersection(this, result, handle, prov);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_difference(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, IndexSpace initial,
        const std::vector<IndexSpace>& handles, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      PendingPartitionOp* part_op =
          runtime->get_operation<PendingPartitionOp>();
      IndexSpace result = instantiate_subspace(parent, realm_color, type_tag);
      part_op->initialize_index_space_difference(
          this, result, initial, handles, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::verify_partition(
        IndexPartition pid, PartitionKind kind, const char* function_name)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* node = runtime->get_node(pid);
      // Check containment first because our implementation of the algorithms
      // for disjointnss and completeness rely upon it.
      for (ColorSpaceIterator itr(node); itr; itr++)
      {
        IndexSpaceNode* child_node = node->get_child(*itr);
        IndexSpaceExpression* diff =
            runtime->subtract_index_spaces(child_node, node->parent);
        if (!diff->is_empty())
        {
          const DomainPoint bad =
              node->color_space->delinearize_color_to_point(*itr);
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Call to partition function " << function_name << " in "
                << *this << " has non-dominated child sub-region at color "
                << bad << ".";
          error.raise();
        }
      }
      // Check disjointness
      if ((kind == LEGION_DISJOINT_KIND) ||
          (kind == LEGION_DISJOINT_COMPLETE_KIND) ||
          (kind == LEGION_DISJOINT_INCOMPLETE_KIND))
      {
        if (!node->is_disjoint(true /*from application*/))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Call to partitioning function " << function_name << " in "
                << get_task_name() << " (UID " << get_unique_id()
                << ") specified partition was "
                << ((kind == LEGION_DISJOINT_KIND) ? "DISJOINT_KIND" :
                    (kind == LEGION_DISJOINT_COMPLETE_KIND) ?
                                                     "DISJOINT_COMPLETE_KIND" :
                                                     "DISJOINT_INCOMPLETE_KIND")
                << " but the partition is aliased.";
          error.raise();
        }
      }
      else if (
          (kind == LEGION_ALIASED_KIND) ||
          (kind == LEGION_ALIASED_COMPLETE_KIND) ||
          (kind == LEGION_ALIASED_INCOMPLETE_KIND))
      {
        if (node->is_disjoint(true /*from application*/))
        {
          Warning warning;
          warning << "Call to partitioning function " << function_name << " in "
                  << get_task_name() << " (UID " << get_unique_id()
                  << ") specified partition was "
                  << ((kind == LEGION_ALIASED_KIND) ? "ALIASED_KIND" :
                      (kind == LEGION_ALIASED_COMPLETE_KIND) ?
                                                      "ALIASED_COMPLETE_KIND" :
                                                      "ALIASED_INCOMPLETE_KIND")
                  << " but the partition is disjoint. This could "
                  << "lead to a performance bug.";
          warning.raise();
        }
      }
      // Check completeness
      if ((kind == LEGION_DISJOINT_COMPLETE_KIND) ||
          (kind == LEGION_ALIASED_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_COMPLETE_KIND))
      {
        if (!node->is_complete(true /*from application*/))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Call to partitioning function " << function_name << " in "
                << get_task_name() << " (UID " << get_unique_id()
                << ") specified partition was "
                << ((kind == LEGION_DISJOINT_COMPLETE_KIND) ?
                        "DISJOINT_COMPLETE_KIND" :
                    (kind == LEGION_ALIASED_COMPLETE_KIND) ?
                        "ALIASED_COMPLETE_KIND" :
                        "COMPUTE_COMPLETE_KIND")
                << " but the partition is incomplete.";
          error.raise();
        }
      }
      else if (
          (kind == LEGION_DISJOINT_INCOMPLETE_KIND) ||
          (kind == LEGION_ALIASED_INCOMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        if (node->is_complete(true /*from application*/))
        {
          Warning warning;
          warning << "Call to partitioning function " << function_name << " in "
                  << get_task_name() << " (UID " << get_unique_id()
                  << ") specified partition was "
                  << ((kind == LEGION_DISJOINT_INCOMPLETE_KIND) ?
                          "DISJOINT_INCOMPLETE_KIND" :
                      (kind == LEGION_ALIASED_INCOMPLETE_KIND) ?
                          "ALIASED_INCOMPLETE_KIND" :
                          "COMPUTE_INCOMPLETE_KIND")
                  << " but the partition is complete. This could "
                  << "lead to a performance bug.";
          warning.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::VerifyPartitionArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      proxy_this->verify_partition(pid, kind, func);
    }

    //--------------------------------------------------------------------------
    FieldSpace InnerContext::create_field_space(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      FieldSpace space(runtime->get_unique_field_space_id());
      LegionSpy::log_field_space(
          space.get_id(), runtime->address_space,
          (provenance == nullptr) ? std::string_view() : provenance->human);
      runtime->create_node(space, RtEvent::NO_RT_EVENT, provenance);
      register_field_space_creation(space);
      return space;
    }

    //--------------------------------------------------------------------------
    FieldSpace InnerContext::create_field_space(
        const std::vector<size_t>& sizes,
        std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      FieldSpace space(runtime->get_unique_field_space_id());
      LegionSpy::log_field_space(
          space.get_id(), runtime->address_space,
          (provenance == nullptr) ? std::string_view() : provenance->human);
      FieldSpaceNode* node =
          runtime->create_node(space, RtEvent::NO_RT_EVENT, provenance);
      register_field_space_creation(space);
      if (resulting_fields.size() < sizes.size())
        resulting_fields.resize(sizes.size(), LEGION_AUTO_GENERATE_ID);
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == LEGION_AUTO_GENERATE_ID)
          resulting_fields[idx] = runtime->get_unique_field_id();
        else if (resulting_fields[idx] >= LEGION_MAX_APPLICATION_FIELD_ID)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Task " << *this << " attempted to allocate a field "
                << "with ID " << resulting_fields[idx] << "which exceeds the "
                << "LEGION_MAX_APPLICATION_FIELD_ID bound set in "
                   "legion_config.h.";
          error.raise();
        }
        LegionSpy::log_field_creation(
            space.get_id(), resulting_fields[idx], sizes[idx],
            (provenance == nullptr) ? std::string_view() : provenance->human);
      }
      node->initialize_fields(sizes, resulting_fields, serdez_id, provenance);
      register_all_field_creations(space, false /*local*/, resulting_fields);
      return space;
    }

    //--------------------------------------------------------------------------
    FieldSpace InnerContext::create_field_space(
        const std::vector<Future>& sizes,
        std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      const FieldSpace space = create_field_space(provenance);
      FieldSpaceNode* node = runtime->get_node(space);
      if (resulting_fields.size() < sizes.size())
        resulting_fields.resize(sizes.size(), LEGION_AUTO_GENERATE_ID);
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == LEGION_AUTO_GENERATE_ID)
          resulting_fields[idx] = runtime->get_unique_field_id();
        else if (resulting_fields[idx] >= LEGION_MAX_APPLICATION_FIELD_ID)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Task " << *this << " attempted to allocate a field "
                << "with ID " << resulting_fields[idx] << " which exceeds the "
                << "LEGION_MAX_APPLICATION_FIELD_ID bound set in "
                   "legion_config.h.";
          error.raise();
        }
      }
      for (unsigned idx = 0; idx < sizes.size(); idx++)
        if (sizes[idx].impl == nullptr)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Invalid empty future passed to field allocation for "
                << "field " << resulting_fields[idx] << " in task " << *this
                << ".";
          error.raise();
        }
      // Get a new creation operation
      CreationOp* creator_op = runtime->get_operation<CreationOp>();
      const ApEvent ready = creator_op->get_completion_event();
      creator_op->initialize_fields(
          this, node, resulting_fields, sizes, provenance);
      node->initialize_fields(
          ready, resulting_fields, serdez_id, creator_op->get_provenance());
      register_all_field_creations(space, false /*local*/, resulting_fields);
      add_to_dependence_queue(creator_op);
      return space;
    }

    //--------------------------------------------------------------------------
    void InnerContext::create_shared_ownership(FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return;
      runtime->create_shared_ownership(handle);
      AutoLock priv_lock(privilege_lock);
      std::map<FieldSpace, unsigned>::iterator finder =
          created_field_spaces.find(handle);
      if (finder != created_field_spaces.end())
        finder->second++;
      else
        created_field_spaces[handle] = 1;
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_field_space(
        FieldSpace handle, const bool unordered, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return;
      // Check to see if this is one that we should be allowed to destory
      {
        AutoLock priv_lock(privilege_lock);
        std::map<FieldSpace, unsigned>::iterator finder =
            created_field_spaces.find(handle);
        if (finder != created_field_spaces.end())
        {
          legion_assert(finder->second > 0);
          if (--finder->second == 0)
            created_field_spaces.erase(finder);
          else
            return;
          // Count how many regions are still using this field space
          // that still need to be deleted before we can remove the
          // list of created fields
          std::set<LogicalRegion> latent_regions;
          for (const std::pair<const LogicalRegion, unsigned>& it :
               created_regions)
            if (it.first.get_field_space() == handle)
              latent_regions.insert(it.first);
          for (const std::pair<const LogicalRegion, bool>& it : local_regions)
            if (it.first.get_field_space() == handle)
              latent_regions.insert(it.first);
          if (latent_regions.empty())
          {
            // No remaining regions so we can remove any created fields now
            for (std::set<std::pair<FieldSpace, FieldID>>::iterator it =
                     created_fields.begin();
                 it != created_fields.end();
                 /*nothing*/)
            {
              if (it->first == handle)
              {
                std::set<std::pair<FieldSpace, FieldID>>::iterator to_delete =
                    it++;
                created_fields.erase(to_delete);
              }
              else
                it++;
            }
          }
          else
            latent_field_spaces[handle] = latent_regions;
        }
        else
        {
          // If we didn't make this field space, record the deletion
          // and keep going. It will be handled by the context that
          // made the field space
          deleted_field_spaces.emplace_back(
              DeletedFieldSpace(handle, provenance));
          return;
        }
      }
      DeletionOp* op = runtime->get_operation<DeletionOp>();
      op->initialize_field_space_deletion(this, handle, unordered, provenance);
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error
              << "Illegal unordered field space deletion performed after task "
              << get_task_name() << " (UID " << get_unique_id()
              << ") has finished executing. All unordered operations must "
              << "be performed before the end of the execution of the parent "
                 "task.";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    FieldAllocatorImpl* InnerContext::create_field_allocator(
        FieldSpace handle, bool unordered)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock priv_lock(privilege_lock, false /*exclusive*/);
        std::map<FieldSpace, FieldAllocatorImpl*>::const_iterator finder =
            field_allocators.find(handle);
        if (finder != field_allocators.end())
          return finder->second;
      }
      // Didn't find it, so have to make, retake the lock in exclusive mode
      FieldSpaceNode* node = runtime->get_node(handle);
      AutoLock priv_lock(privilege_lock);
      // Check to see if we lost the race
      std::map<FieldSpace, FieldAllocatorImpl*>::const_iterator finder =
          field_allocators.find(handle);
      if (finder != field_allocators.end())
        return finder->second;
      // Don't have one so make a new one
      const RtEvent ready = node->create_allocator(runtime->address_space);
      FieldAllocatorImpl* result = new FieldAllocatorImpl(node, this, ready);
      // Save it for later
      field_allocators[handle] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_field_allocator(FieldSpaceNode* node)
    //--------------------------------------------------------------------------
    {
      const RtEvent ready = node->destroy_allocator(runtime->address_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      AutoLock priv_lock(privilege_lock);
      std::map<FieldSpace, FieldAllocatorImpl*>::iterator finder =
          field_allocators.find(node->handle);
      legion_assert(finder != field_allocators.end());
      field_allocators.erase(finder);
    }

    //--------------------------------------------------------------------------
    FieldID InnerContext::allocate_field(
        FieldSpace space, size_t field_size, FieldID fid, bool local,
        CustomSerdezID serdez_id, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (fid == LEGION_AUTO_GENERATE_ID)
        fid = runtime->get_unique_field_id();
      else if (fid >= LEGION_MAX_APPLICATION_FIELD_ID)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Task " << *this << " attempted to allocate a field with ID "
              << fid << " which exceeds the LEGION_MAX_APPLICATION_FIELD_ID "
              << "bound set in legion_config.h.";
        error.raise();
      }
      LegionSpy::log_field_creation(
          space.get_id(), fid, field_size,
          (provenance == nullptr) ? std::string_view() : provenance->human);
      std::set<RtEvent> done_events;
      if (local)
        allocate_local_field(
            space, field_size, fid, serdez_id, done_events, provenance);
      else
        runtime->allocate_field(space, field_size, fid, serdez_id, provenance);
      register_field_creation(space, fid, local);
      if (!done_events.empty())
      {
        RtEvent wait_on = Runtime::merge_events(done_events);
        wait_on.wait();
      }
      return fid;
    }

    //--------------------------------------------------------------------------
    void InnerContext::allocate_fields(
        FieldSpace space, const std::vector<size_t>& sizes,
        std::vector<FieldID>& resulting_fields, bool local,
        CustomSerdezID serdez_id, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (resulting_fields.size() < sizes.size())
        resulting_fields.resize(sizes.size(), LEGION_AUTO_GENERATE_ID);
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == LEGION_AUTO_GENERATE_ID)
          resulting_fields[idx] = runtime->get_unique_field_id();
        else if (resulting_fields[idx] >= LEGION_MAX_APPLICATION_FIELD_ID)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Task " << *this << " attempted to allocate a field "
                << "with ID " << resulting_fields[idx] << " which exceeds the "
                << "LEGION_MAX_APPLICATION_FIELD_ID bound set in "
                   "legion_config.h.";
          error.raise();
        }
        LegionSpy::log_field_creation(
            space.get_id(), resulting_fields[idx], sizes[idx],
            (provenance == nullptr) ? std::string_view() : provenance->human);
      }
      std::set<RtEvent> done_events;
      if (local)
        allocate_local_fields(
            space, sizes, resulting_fields, serdez_id, done_events, provenance);
      else
        runtime->allocate_fields(
            space, sizes, resulting_fields, serdez_id, provenance);
      register_all_field_creations(space, local, resulting_fields);
      if (!done_events.empty())
      {
        RtEvent wait_on = Runtime::merge_events(done_events);
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    FieldID InnerContext::allocate_field(
        FieldSpace space, const Future& field_size, FieldID fid, bool local,
        CustomSerdezID serdez_id, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (local)
      {
        Fatal fatal;
        fatal << "Local fields do no support allocation with future sizes yet.";
        fatal.raise();
      }
      if (fid == LEGION_AUTO_GENERATE_ID)
        fid = runtime->get_unique_field_id();
      else if (fid >= LEGION_MAX_APPLICATION_FIELD_ID)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Task " << get_task_name() << " (ID " << get_unique_id()
              << ") attempted to allocate a field with ID " << fid
              << " which exceeds the LEGION_MAX_APPLICATION_FIELD_ID bound set "
              << "in legion_config.h";
        error.raise();
      }
      if (field_size.impl == nullptr)
      {
        Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
        error << "Invalid empty future passed to field allocation for field "
              << fid << " in task " << get_task_name() << " (UID "
              << get_unique_id() << ")";
        error.raise();
      }
      // Get a new creation operation
      CreationOp* creator_op = runtime->get_operation<CreationOp>();
      const ApEvent ready = creator_op->get_completion_event();
      // Tell the node that we're allocating a field of size zero
      // which will indicate that we'll fill in the size later
      RtEvent precondition;
      FieldSpaceNode* node = runtime->allocate_field(
          space, ready, fid, serdez_id, provenance, precondition);
      creator_op->initialize_field(this, node, fid, field_size, provenance);
      register_field_creation(space, fid, local);
      // Make sure the IDs are valid for the user
      if (precondition.exists() && !precondition.has_triggered())
        precondition.wait();
      add_to_dependence_queue(creator_op);
      return fid;
    }

    //--------------------------------------------------------------------------
    void InnerContext::allocate_local_field(
        FieldSpace space, size_t field_size, FieldID fid,
        CustomSerdezID serdez_id, std::set<RtEvent>& done_events,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // See if we've exceeded our local field allocations
      // for this field space
      AutoLock local_lock(local_field_lock);
      std::vector<LocalFieldInfo>& infos = local_field_infos[space];
      if (infos.size() == runtime->max_local_fields)
      {
        Error error(LEGION_RESOURCE_EXCEPTION);
        error << "Exceeded maximum number of local fields in context of task "
              << get_task_name() << " (UID " << get_unique_id()
              << "). The maximum is currently set to "
              << runtime->max_local_fields
              << ", but can be modified with the -lg:local flag.";
        error.raise();
      }
      std::set<unsigned> current_indexes;
      for (const LocalFieldInfo& it : infos) current_indexes.insert(it.index);
      std::vector<FieldID> fields(1, fid);
      std::vector<size_t> sizes(1, field_size);
      std::vector<unsigned> new_indexes;
      if (!runtime->allocate_local_fields(
              space, fields, sizes, serdez_id, current_indexes, new_indexes,
              provenance))
      {
        Error error(LEGION_RESOURCE_EXCEPTION);
        error << "Unable to allocate local field in context of task "
              << get_task_name() << " (UID " << get_unique_id()
              << ") due to local field size fragmentation. This situation can "
              << "be improved by increasing the maximum number of permitted "
              << "local fields in a context with the -lg:local flag.";
        error.raise();
      }
      legion_assert(new_indexes.size() == 1);
      // Only need the lock here when modifying since all writes
      // to this data structure are serialized
      infos.emplace_back(
          LocalFieldInfo(fid, field_size, serdez_id, new_indexes[0], false));
      struct Functor {
      public:
        Functor(
            DistributedID id, FieldSpace sp, Provenance* prov,
            const LocalFieldInfo& in, std::set<RtEvent>& done)
          : did(id), space(sp), provenance(prov), info(in), done_events(done),
            count(0)
        { }
        void apply(AddressSpaceID target)
        {
          RtUserEvent done_event = Runtime::create_rt_user_event();
          LocalFieldUpdateMessage rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize<size_t>(1);  // field space count
            rez.serialize(space);
            if (provenance != nullptr)
              provenance->serialize(rez);
            else
              Provenance::serialize_null(rez);
            rez.serialize<size_t>(1);  // field count
            rez.serialize(info);
            rez.serialize(done_event);
          }
          rez.dispatch(target);
          done_events.insert(done_event);
          count++;
        };
      public:
        DistributedID did;
        FieldSpace space;
        Provenance* provenance;
        const LocalFieldInfo& info;
        std::set<RtEvent>& done_events;
        unsigned count;
      };
      Functor functor(did, space, provenance, infos.back(), done_events);
      map_over_remote_instances(functor);
      if (functor.count > 0)
        pack_global_ref(functor.count);
    }

    //--------------------------------------------------------------------------
    void InnerContext::allocate_fields(
        FieldSpace space, const std::vector<Future>& sizes,
        std::vector<FieldID>& resulting_fields, bool local,
        CustomSerdezID serdez_id, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (local)
      {
        Fatal fatal;
        fatal << "Local fields do no support allocation with future sizes yet.";
        fatal.raise();
      }
      if (resulting_fields.size() < sizes.size())
        resulting_fields.resize(sizes.size(), LEGION_AUTO_GENERATE_ID);
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == LEGION_AUTO_GENERATE_ID)
          resulting_fields[idx] = runtime->get_unique_field_id();
        else if (resulting_fields[idx] >= LEGION_MAX_APPLICATION_FIELD_ID)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error
              << "Task " << get_task_name() << " (ID " << get_unique_id()
              << ") attempted to allocate a field with ID "
              << resulting_fields[idx]
              << " which exceeds the LEGION_MAX_APPLICATION_FIELD_ID bound set "
              << "in legion_config.h";
          error.raise();
        }
      }
      for (unsigned idx = 0; idx < sizes.size(); idx++)
        if (sizes[idx].impl == nullptr)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Invalid empty future passed to field allocation for field "
                << resulting_fields[idx] << " in task " << get_task_name()
                << " (UID " << get_unique_id() << ")";
          error.raise();
        }
      // Get a new creation operation
      CreationOp* creator_op = runtime->get_operation<CreationOp>();
      const ApEvent ready = creator_op->get_completion_event();
      // Tell the node that we're allocating a field of size zero
      // which will indicate that we'll fill in the size later
      RtEvent precondition;
      FieldSpaceNode* node = runtime->allocate_fields(
          space, ready, resulting_fields, serdez_id, provenance, precondition);
      creator_op->initialize_fields(
          this, node, resulting_fields, sizes, provenance);
      register_all_field_creations(space, local, resulting_fields);
      // Need to make sure that field IDs are valid for users
      if (precondition.exists() && !precondition.has_triggered())
        precondition.wait();
      add_to_dependence_queue(creator_op);
    }

    //--------------------------------------------------------------------------
    void InnerContext::allocate_local_fields(
        FieldSpace space, const std::vector<size_t>& sizes,
        const std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
        std::set<RtEvent>& done_events, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // See if we've exceeded our local field allocations
      // for this field space
      AutoLock local_lock(local_field_lock);
      std::vector<LocalFieldInfo>& infos = local_field_infos[space];
      if ((infos.size() + sizes.size()) > runtime->max_local_fields)
      {
        Error error(LEGION_RESOURCE_EXCEPTION);
        error << "Exceeded maximum number of local fields in context of task "
              << get_task_name() << " (UID " << get_unique_id()
              << "). The maximum is currently set to "
              << runtime->max_local_fields
              << ", but can be modified with the -lg:local flag.";
        error.raise();
      }
      std::set<unsigned> current_indexes;
      for (const LocalFieldInfo& it : infos) current_indexes.insert(it.index);
      std::vector<unsigned> new_indexes;
      if (!runtime->allocate_local_fields(
              space, resulting_fields, sizes, serdez_id, current_indexes,
              new_indexes, provenance))
      {
        Error error(LEGION_RESOURCE_EXCEPTION);
        error << "Unable to allocate local field in context of task "
              << get_task_name() << " (UID " << get_unique_id()
              << ") due to local field size fragmentation. This situation can "
              << "be improved by increasing the maximum number of permitted "
              << "local fields in a context with the -lg:local flag.";
        error.raise();
      }
      legion_assert(new_indexes.size() == resulting_fields.size());
      // Only need the lock here when writing since we know all writes
      // are serialized and we only need to worry about interfering readers
      const unsigned offset = infos.size();
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
        infos.emplace_back(LocalFieldInfo(
            resulting_fields[idx], sizes[idx], serdez_id, new_indexes[idx],
            false));
      // Have to send notifications to any remote nodes
      struct Functor {
      public:
        Functor(
            DistributedID id, FieldSpace sp, Provenance* prov, size_t s,
            unsigned off, const std::vector<LocalFieldInfo>& in,
            std::set<RtEvent>& done)
          : did(id), space(sp), provenance(prov), size(s), offset(off),
            infos(in), done_events(done), count(0)
        { }
        void apply(AddressSpaceID target)
        {
          RtUserEvent done_event = Runtime::create_rt_user_event();
          LocalFieldUpdateMessage rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize<size_t>(1);  // field space count
            rez.serialize(space);
            if (provenance != nullptr)
              provenance->serialize(rez);
            else
              Provenance::serialize_null(rez);
            rez.serialize<size_t>(size);  // field count
            for (unsigned idx = 0; idx < size; idx++)
              rez.serialize(infos[offset + idx]);
            rez.serialize(done_event);
          }
          rez.dispatch(target);
          done_events.insert(done_event);
          count++;
        }
      public:
        DistributedID did;
        FieldSpace space;
        Provenance* provenance;
        size_t size;
        unsigned offset;
        const std::vector<LocalFieldInfo>& infos;
        std::set<RtEvent>& done_events;
        unsigned count;
      };
      Functor functor(
          did, space, provenance, resulting_fields.size(), offset, infos,
          done_events);
      map_over_remote_instances(functor);
      if (functor.count > 0)
        pack_global_ref(functor.count);
    }

    //--------------------------------------------------------------------------
    void InnerContext::free_field(
        FieldAllocatorImpl* allocator, FieldSpace space, FieldID fid,
        const bool unordered, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock priv_lock(privilege_lock, false /*exclusive*/);
        const std::pair<FieldSpace, FieldID> key(space, fid);
        // This field will actually be removed in analyze_destroy_fields
        std::set<std::pair<FieldSpace, FieldID>>::const_iterator finder =
            created_fields.find(key);
        if (finder == created_fields.end())
        {
          std::map<std::pair<FieldSpace, FieldID>, bool>::iterator
              local_finder = local_fields.find(key);
          if (local_finder == local_fields.end())
          {
            // If we didn't make this field, record the deletion and
            // then have a later context handle it
            deleted_fields.emplace_back(DeletedField(space, fid, provenance));
            return;
          }
          else
            local_finder->second = true;
        }
        // Don't remove anything from created fields yet, we still might
        // need it as part of the logical dependence analysis for earlier ops
      }
      // Launch off the deletion operation
      DeletionOp* op = runtime->get_operation<DeletionOp>();
      op->initialize_field_deletion(
          this, space, fid, unordered, allocator, provenance,
          false /*non owner shard*/);
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Illegal unordered field free performed after task "
                << get_task_name() << " (UID " << get_unique_id()
                << ") has finished executing. All unordered operations must "
                << "be performed before the end of the execution of the parent "
                   "task.";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::free_fields(
        FieldAllocatorImpl* allocator, FieldSpace space,
        const std::set<FieldID>& to_free, const bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      std::set<FieldID> free_now;
      {
        AutoLock priv_lock(privilege_lock, false /*exclusive*/);
        // These fields will actually be removed in analyze_destroy_fields
        for (const FieldID& it : to_free)
        {
          const std::pair<FieldSpace, FieldID> key(space, it);
          std::set<std::pair<FieldSpace, FieldID>>::const_iterator finder =
              created_fields.find(key);
          if (finder == created_fields.end())
          {
            std::map<std::pair<FieldSpace, FieldID>, bool>::iterator
                local_finder = local_fields.find(key);
            if (local_finder != local_fields.end())
            {
              local_finder->second = true;
              free_now.insert(it);
            }
            else
              deleted_fields.emplace_back(DeletedField(space, it, provenance));
          }
          else
          {
            // Don't remove anything from created fields yet,
            // we still might need need it as part of the logical
            // dependence analysis for earlier ops
            free_now.insert(it);
          }
        }
      }
      if (free_now.empty())
        return;
      DeletionOp* op = runtime->get_operation<DeletionOp>();
      op->initialize_field_deletions(
          this, space, free_now, unordered, allocator, provenance,
          false /*non owner shard*/);
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Illegal unordered free fields performed after task "
                << get_task_name() << " (UID " << get_unique_id()
                << ") has finished executing. All unordered operations must "
                << "be performed before the end of the execution of the parent "
                   "task.";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    LogicalRegion InnerContext::create_logical_region(
        IndexSpace index_space, FieldSpace field_space, const bool task_local,
        Provenance* provenance, const bool output_region)
    //--------------------------------------------------------------------------
    {
      RegionTreeID tid = runtime->get_unique_region_tree_id();
      LogicalRegion region(tid, index_space, field_space);
      LegionSpy::log_top_region(
          index_space.get_id(), field_space.get_id(), tid,
          runtime->address_space,
          (provenance == nullptr) ? std::string_view() : provenance->human);
      const DistributedID did = runtime->get_available_distributed_id();
      runtime->create_node(
          region, nullptr /*parent*/, RtEvent::NO_RT_EVENT, did, provenance);
      // Register the creation of a top-level region with the context
      const unsigned created_index =
          register_region_creation(region, task_local, output_region);
      if (output_region)
      {
        // If this is an output region make sure nobody tries to compute
        // the equivalence sets for it until we know it is ready
        AutoLock priv_lock(privilege_lock);
        legion_assert(
            equivalence_set_trees.find(created_index) ==
            equivalence_set_trees.end());
        legion_assert(
            pending_equivalence_set_trees.find(created_index) ==
            pending_equivalence_set_trees.end());
        // Put in a guard so that nobody else tries to make it
        pending_equivalence_set_trees[created_index] =
            RtUserEvent::NO_RT_USER_EVENT;
      }
      return region;
    }

    //--------------------------------------------------------------------------
    void InnerContext::create_shared_ownership(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return;
      if (!runtime->is_top_level_region(handle))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Illegal call to create shared ownership for logical region ("
            << handle.index_space.get_id() << "," << handle.field_space.get_id()
            << "," << handle.get_tree_id() << ") in task " << get_task_name()
            << " (UID " << get_unique_id()
            << ") which is not a top-level logical "
            << "region. Legion only permits top-level logical regions to have "
            << "shared ownerships.";
        error.raise();
      }
      runtime->create_shared_ownership(handle);
      AutoLock priv_lock(privilege_lock);
      std::map<LogicalRegion, unsigned>::iterator finder =
          created_regions.find(handle);
      if (finder != created_regions.end())
        finder->second++;
      else
        created_regions[handle] = 1;
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_logical_region(
        LogicalRegion handle, const bool unordered, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return;
      // Check to see if this is a top-level logical region, if not then
      // we shouldn't even be destroying it
      if (!runtime->is_top_level_region(handle))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal call to destroy logical region ("
              << handle.index_space.get_id() << ","
              << handle.field_space.get_id() << "," << handle.get_tree_id()
              << ") in task " << get_task_name() << " (UID " << get_unique_id()
              << ") which is not a top-level logical "
              << "region. Legion only permits top-level logical regions to be "
                 "destroyed.";
        error.raise();
      }
      // Check to see if this is one that we should be allowed to destory
      {
        AutoLock priv_lock(privilege_lock, false /*exclusive*/);
        std::map<LogicalRegion, unsigned>::iterator finder =
            created_regions.find(handle);
        if (finder == created_regions.end())
        {
          // Check to see if it is a local region
          std::map<LogicalRegion, bool>::iterator local_finder =
              local_regions.find(handle);
          // Mark that this region is deleted, safe even though this
          // is a read-only lock because we're not changing the structure
          // of the map
          if (local_finder == local_regions.end())
          {
            // Record the deletion for later and propagate it up
            deleted_regions.emplace_back(DeletedRegion(handle, provenance));
            return;
          }
          else
            local_finder->second = true;
        }
        else
        {
          if (finder->second == 0)
          {
            Warning warning;
            warning << "Duplicate deletions were performed for region "
                    << handle << " in task tree rooted by " << *this << ".";
            warning.raise();
            return;
          }
          if (--finder->second > 0)
            return;
          // Don't remove anything from created regions yet, we still might
          // need it as part of the logical dependence analysis for earlier
          // operations, but the reference count is zero so we're protected
        }
      }
      DeletionOp* op = runtime->get_operation<DeletionOp>();
      op->initialize_logical_region_deletion(
          this, handle, unordered, provenance);
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Illegal unordered logical region deletion performed after "
                   "task "
                << get_task_name() << " (UID " << get_unique_id()
                << ") has finished executing. All unordered operations must "
                << "be performed before the end of the execution of the parent "
                   "task.";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::reset_equivalence_sets(
        LogicalRegion parent, LogicalRegion region,
        const std::set<FieldID>& fields)
    //--------------------------------------------------------------------------
    {
      // Ignore reset calls inside of traces
      if ((current_trace != nullptr) && current_trace->is_fixed())
      {
        Warning warning;
        warning << "Ignoring equivalence sets reset in " << get_task_name()
                << " (UID " << get_unique_id()
                << ") because it was made inside "
                << "of a trace.";
        warning.raise();
      }
      if (fields.empty())
      {
        Warning warning;
        warning << "Ignoring equivalence sets reset in " << get_task_name()
                << " (UID " << get_unique_id()
                << ") because it contains no fields.";
        warning.raise();
        return;
      }
      ResetOp* reset = runtime->get_operation<ResetOp>();
      reset->initialize(this, parent, region, fields);
      add_to_dependence_queue(reset);
    }

    //--------------------------------------------------------------------------
    void InnerContext::get_local_field_set(
        const FieldSpace handle, const std::set<unsigned>& indexes,
        std::set<FieldID>& to_set) const
    //--------------------------------------------------------------------------
    {
      AutoLock lf_lock(local_field_lock, false /*exclusive*/);
      std::map<FieldSpace, std::vector<LocalFieldInfo>>::const_iterator finder =
          local_field_infos.find(handle);
      legion_assert(finder != local_field_infos.end());
      [[maybe_unused]] unsigned found = 0;
      for (const LocalFieldInfo& it : finder->second)
      {
        if (indexes.find(it.index) != indexes.end())
        {
          found++;
          to_set.insert(it.fid);
        }
      }
      legion_assert(found == indexes.size());
    }

    //--------------------------------------------------------------------------
    void InnerContext::get_local_field_set(
        const FieldSpace handle, const std::set<unsigned>& indexes,
        std::vector<FieldID>& to_set) const
    //--------------------------------------------------------------------------
    {
      AutoLock lf_lock(local_field_lock, false /*exclusive*/);
      std::map<FieldSpace, std::vector<LocalFieldInfo>>::const_iterator finder =
          local_field_infos.find(handle);
      legion_assert(finder != local_field_infos.end());
      [[maybe_unused]] unsigned found = 0;
      for (const LocalFieldInfo& it : finder->second)
      {
        if (indexes.find(it.index) != indexes.end())
        {
          found++;
          to_set.emplace_back(it.fid);
        }
      }
      legion_assert(found == indexes.size());
    }

    //--------------------------------------------------------------------------
    unsigned InnerContext::add_created_region(
        LogicalRegion handle, bool task_local, bool output_region)
    //--------------------------------------------------------------------------
    {
      // Already hold the lock from the caller
      if (!task_local && !output_region)
      {
        // There's a race here with created region tree contexts coming back
        // and making these requirements for themselves so we check for
        // duplications here in that case
        std::map<LogicalRegion, std::pair<RegionRequirement, unsigned>>::
            const_iterator finder = created_requirements.find(handle);
        if (finder != created_requirements.end())
          return finder->second.second;
      }
      RegionRequirement new_req(
          handle, LEGION_READ_WRITE, LEGION_EXCLUSIVE, handle);
      if (output_region)
        new_req.flags |= LEGION_CREATED_OUTPUT_REQUIREMENT_FLAG;
      TaskOp::log_requirement(get_unique_id(), next_created_index, new_req);
      // Put a region requirement with no fields in the list of
      // created requirements, we know we can add any fields for
      // this field space in the future since we own all privileges
      created_requirements.emplace(
          handle, std::make_pair(std::move(new_req), next_created_index));
      // Created regions always return privileges that they make
      returnable_privileges.emplace(
          next_created_index, std::make_pair(handle, !task_local));
      return next_created_index++;
    }

    //--------------------------------------------------------------------------
    void InnerContext::log_created_requirements(void) const
    //--------------------------------------------------------------------------
    {
      for (const std::pair<
               const LogicalRegion, std::pair<RegionRequirement, unsigned>>&
               req : created_requirements)
      {
        // We already logged the requirement when we made it
        // Skip it if there are no privilege fields
        if (req.second.first.privilege_fields.empty())
          continue;
        owner_task->log_virtual_mapping(req.second.second, req.second.first);
      }
    }

    //--------------------------------------------------------------------------
    unsigned InnerContext::register_region_creation(
        LogicalRegion handle, bool task_local, bool output_region)
    //--------------------------------------------------------------------------
    {
      // Create a new logical region
      // Hold the operation lock when doing this since children could
      // be returning values from the utility processor
      AutoLock priv_lock(privilege_lock);
      legion_assert(local_regions.find(handle) == local_regions.end());
      legion_assert(created_regions.find(handle) == created_regions.end());
      if (task_local)
      {
        local_regions[handle] = false /*not deleted*/;
      }
      else
      {
        legion_assert(created_regions.find(handle) == created_regions.end());
        created_regions[handle] = 1;
      }
      return add_created_region(handle, task_local, output_region);
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_field_creation(
        FieldSpace handle, FieldID fid, bool local)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      std::pair<FieldSpace, FieldID> key(handle, fid);
      legion_assert(local_fields.find(key) == local_fields.end());
      legion_assert(created_fields.find(key) == created_fields.end());
      if (!local)
      {
        legion_assert(created_fields.find(key) == created_fields.end());
        created_fields.insert(key);
      }
      else
        local_fields[key] = false /*deleted*/;
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_all_field_creations(
        FieldSpace handle, bool local, const std::vector<FieldID>& fields)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      if (local)
      {
        for (unsigned idx = 0; idx < fields.size(); idx++)
        {
          std::pair<FieldSpace, FieldID> key(handle, fields[idx]);
          legion_assert(local_fields.find(key) == local_fields.end());
          local_fields[key] = false /*deleted*/;
        }
      }
      else
      {
        for (unsigned idx = 0; idx < fields.size(); idx++)
        {
          std::pair<FieldSpace, FieldID> key(handle, fields[idx]);
          legion_assert(created_fields.find(key) == created_fields.end());
          created_fields.insert(key);
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_field_space_creation(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      legion_assert(
          created_field_spaces.find(space) == created_field_spaces.end());
      created_field_spaces[space] = 1;
    }

    //--------------------------------------------------------------------------
    bool InnerContext::has_created_index_space(IndexSpace space) const
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      return (created_index_spaces.find(space) != created_index_spaces.end());
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_index_space_creation(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      legion_assert(
          created_index_spaces.find(space) == created_index_spaces.end());
      created_index_spaces[space] = 1;
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_index_partition_creation(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      legion_assert(
          created_index_partitions.find(handle) ==
          created_index_partitions.end());
      created_index_partitions[handle] = 1;
    }

    //--------------------------------------------------------------------------
    void InnerContext::report_leaks_and_duplicates(
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      if (!deleted_regions.empty())
      {
        for (const DeletedRegion& it : deleted_regions)
        {
          Warning warning;
          warning << "Duplicate deletions were performed for region "
                  << it.region << " in task tree rooted by " << get_task_name()
                  << " (provenance "
                  << ((it.provenance == nullptr) ?
                          7 :
                          int(it.provenance->human.length()))
                  << ":"
                  << ((it.provenance == nullptr) ? "unknown" :
                                                   it.provenance->human.data())
                  << ")";
          warning.raise();
        }
        deleted_regions.clear();
      }
      if (!deleted_fields.empty())
      {
        for (const DeletedField& it : deleted_fields)
        {
          Warning warning;
          warning << "Duplicate deletions were performed on field " << it.fid
                  << " of field space " << it.space
                  << " in task tree rooted by " << get_task_name()
                  << " (provenance "
                  << ((it.provenance != nullptr) ?
                          int(it.provenance->human.length()) :
                          7)
                  << ":"
                  << ((it.provenance != nullptr) ? it.provenance->human.data() :
                                                   "unknown")
                  << ")";
          warning.raise();
        }
        deleted_fields.clear();
      }
      if (!deleted_field_spaces.empty())
      {
        for (const DeletedFieldSpace& it : deleted_field_spaces)
        {
          Warning warning;
          warning << "Duplicate deletions were performed on field space "
                  << it.space << " in task tree rooted by " << get_task_name()
                  << " (provenance "
                  << ((it.provenance == nullptr) ?
                          7 :
                          int(it.provenance->human.length()))
                  << ":"
                  << ((it.provenance == nullptr) ? "unknown" :
                                                   it.provenance->human.data())
                  << ")";
          warning.raise();
        }
        deleted_field_spaces.clear();
      }
      if (!deleted_index_spaces.empty())
      {
        for (const DeletedIndexSpace& it : deleted_index_spaces)
        {
          Warning warning;
          warning << "Duplicate deletions were performed on index space "
                  << it.space << " in task tree rooted by " << get_task_name()
                  << " (provenance "
                  << ((it.provenance == nullptr) ?
                          7 :
                          int(it.provenance->human.length()))
                  << ":"
                  << ((it.provenance == nullptr) ? "unknown" :
                                                   it.provenance->human.data())
                  << ")";
          warning.raise();
        }
        deleted_index_spaces.clear();
      }
      if (!deleted_index_partitions.empty())
      {
        for (const DeletedPartition& it : deleted_index_partitions)
        {
          Warning warning;
          warning << "Duplicate deletions were performed on index partition "
                  << it.partition << " in task tree rooted by "
                  << get_task_name() << " (provenance "
                  << ((it.provenance == nullptr) ?
                          7 :
                          int(it.provenance->human.length()))
                  << ":"
                  << ((it.provenance == nullptr) ? "unknown" :
                                                   it.provenance->human.data())
                  << ")";
          warning.raise();
        }
        deleted_index_partitions.clear();
      }
      // Now we go through and delete anything that the user leaked
      if (!created_regions.empty())
      {
        for (const std::pair<const LogicalRegion, unsigned>& rit :
             created_regions)
        {
          if (runtime->report_leaks)
          {
            Warning warning;
            warning << "Logical region " << rit.first
                    << " was leaked out of task tree rooted by task "
                    << get_task_name();
            warning.raise();
          }
          runtime->destroy_logical_region(rit.first, preconditions);
          // Remove any latent field spaces and therefore any created fields
          // since they might not be able to be cleaned up after this since
          // this region might be holding the last reference to the field space
          if (!latent_field_spaces.empty())
          {
            std::map<FieldSpace, std::set<LogicalRegion>>::iterator finder =
                latent_field_spaces.find(rit.first.get_field_space());
            if (finder != latent_field_spaces.end())
            {
              std::set<LogicalRegion>::iterator latent_finder =
                  finder->second.find(rit.first);
              legion_assert(latent_finder != finder->second.end());
              finder->second.erase(latent_finder);
              if (finder->second.empty())
              {
                // Now that all the regions using this field space have
                // been deleted we can clean up all the created_fields
                for (std::set<std::pair<FieldSpace, FieldID>>::iterator it =
                         created_fields.begin();
                     it != created_fields.end();
                     /*nothing*/)
                {
                  if (it->first == finder->first)
                  {
                    std::set<std::pair<FieldSpace, FieldID>>::iterator
                        to_delete = it++;
                    created_fields.erase(to_delete);
                  }
                  else
                    it++;
                }
                latent_field_spaces.erase(finder);
              }
            }
          }
        }
        created_regions.clear();
      }
      if (!created_fields.empty())
      {
        std::map<FieldSpace, FieldAllocatorImpl*> leak_allocators;
        for (const std::pair<FieldSpace, FieldID>& it : created_fields)
        {
          if (runtime->report_leaks)
          {
            Warning warning;
            warning << "Field " << it.second << " of field space " << it.first
                    << " was leaked out of task tree rooted by task "
                    << get_task_name();
            warning.raise();
          }
          std::map<FieldSpace, FieldAllocatorImpl*>::const_iterator finder =
              leak_allocators.find(it.first);
          if (finder == leak_allocators.end())
          {
            FieldAllocatorImpl* allocator =
                create_field_allocator(it.first, true /*unordered*/);
            allocator->add_reference();
            leak_allocators[it.first] = allocator;
            allocator->ready_event.wait();
          }
          else
            finder->second->ready_event.wait();
          runtime->free_field(it.first, it.second, preconditions);
        }
        for (const std::pair<const FieldSpace, FieldAllocatorImpl*>& it :
             leak_allocators)
          if (it.second->remove_reference())
            delete it.second;
        created_fields.clear();
      }
      if (!created_field_spaces.empty())
      {
        for (const std::pair<const FieldSpace, unsigned>& it :
             created_field_spaces)
        {
          if (runtime->report_leaks)
          {
            Warning warning;
            warning << "Field space " << it.first
                    << " was leaked out of task tree rooted by task "
                    << get_task_name();
            warning.raise();
          }
          runtime->destroy_field_space(it.first, preconditions);
        }
        created_field_spaces.clear();
      }
      if (!created_index_partitions.empty())
      {
        for (const std::pair<const IndexPartition, unsigned>& it :
             created_index_partitions)
        {
          if (runtime->report_leaks)
          {
            Warning warning;
            warning << "Index partition " << it.first
                    << " was leaked out of task tree rooted by task "
                    << get_task_name();
            warning.raise();
          }
          runtime->destroy_index_partition(it.first, preconditions);
        }
        created_index_partitions.clear();
      }
      if (!created_index_spaces.empty())
      {
        for (const std::pair<const IndexSpace, unsigned>& it :
             created_index_spaces)
        {
          if (runtime->report_leaks)
          {
            Warning warning;
            warning << "Index space " << it.first
                    << " was leaked out of task tree rooted by task "
                    << get_task_name();
            warning.raise();
          }
          runtime->destroy_index_space(
              it.first, runtime->address_space, preconditions);
        }
        created_index_spaces.clear();
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::analyze_destroy_fields(
        FieldSpace handle, const std::set<FieldID>& to_delete,
        std::vector<RegionRequirement>& delete_reqs,
        std::vector<unsigned>& parent_req_indexes,
        std::vector<FieldID>& global_to_free,
        std::vector<FieldID>& local_to_free,
        std::vector<unsigned>& local_field_indexes,
        std::vector<unsigned>& deletion_indexes)
    //--------------------------------------------------------------------------
    {
      {
        // We can't destroy any fields from our original regions because we
        // were not the ones that made them.
        AutoLock priv_lock(privilege_lock);
        // We can actually remove the fields from the data structure now
        for (const FieldID& it : to_delete)
        {
          const std::pair<FieldSpace, FieldID> key(handle, it);
          std::set<std::pair<FieldSpace, FieldID>>::iterator finder =
              created_fields.find(key);
          if (finder == created_fields.end())
          {
            std::map<std::pair<FieldSpace, FieldID>, bool>::iterator
                local_finder = local_fields.find(key);
            legion_assert(local_finder != local_fields.end());
            legion_assert(local_finder->second);
            local_fields.erase(local_finder);
            local_to_free.emplace_back(it);
          }
          else
          {
            created_fields.erase(finder);
            global_to_free.emplace_back(it);
          }
        }
        // Now figure out which region requirements can be destroyed
        for (std::pair<
                 const LogicalRegion, std::pair<RegionRequirement, unsigned>>&
                 it : created_requirements)
        {
          if (it.second.first.region.get_field_space() != handle)
            continue;
          std::set<FieldID> overlapping_fields;
          for (const FieldID& fit : to_delete)
          {
            std::set<FieldID>::iterator finder =
                it.second.first.privilege_fields.find(fit);
            if (finder != it.second.first.privilege_fields.end())
            {
              overlapping_fields.insert(fit);
              // Remove this from the created requirements fields
              it.second.first.privilege_fields.erase(finder);
            }
          }
          if (overlapping_fields.empty())
            continue;
          delete_reqs.resize(delete_reqs.size() + 1);
          RegionRequirement& req = delete_reqs.back();
          req.region = it.second.first.region;
          req.parent = it.second.first.region;
          req.privilege = LEGION_READ_WRITE;
          req.prop = LEGION_EXCLUSIVE;
          req.privilege_fields.swap(overlapping_fields);
          req.handle_type = LEGION_SINGULAR_PROJECTION;
          parent_req_indexes.emplace_back(it.second.second);
          // We need some extra logging for legion spy
          LegionSpy::log_requirement_fields(
              get_unique_id(), it.second.second, req.privilege_fields);
          owner_task->log_virtual_mapping(it.second.second, req);
        }
      }
      if (!local_to_free.empty())
        analyze_free_local_fields(handle, local_to_free, local_field_indexes);
    }

    //--------------------------------------------------------------------------
    void InnerContext::analyze_destroy_logical_region(
        LogicalRegion handle, std::vector<RegionRequirement>& delete_reqs,
        std::vector<unsigned>& parent_req_indexes,
        std::vector<bool>& returnable)
    //--------------------------------------------------------------------------
    {
      // If we're deleting a field space then we can't be deleting any of the
      // original requirements, only requirements that we created
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        // We need some extra logging for legion spy
        std::vector<MappingInstance> instances(
            1, Mapping::PhysicalInstance::get_virtual_instance());
        AutoLock priv_lock(privilege_lock);
        for (std::pair<
                 const LogicalRegion, std::pair<RegionRequirement, unsigned>>&
                 it : created_requirements)
        {
          // Has to match precisely
          if (handle.get_tree_id() == it.second.first.region.get_tree_id())
          {
            // Should be the same region
            legion_assert(handle == it.second.first.region);
            legion_assert(
                returnable_privileges.find(it.second.second) !=
                returnable_privileges.end());
            // Do extra logging for legion spy
            owner_task->log_virtual_mapping(it.second.second, it.second.first);
            // Then do the result of the normal operations
            delete_reqs.resize(delete_reqs.size() + 1);
            RegionRequirement& req = delete_reqs.back();
            req.region = it.second.first.region;
            req.parent = it.second.first.region;
            req.privilege = LEGION_READ_WRITE;
            req.prop = LEGION_EXCLUSIVE;
            // Swap the privilege fields so that nothing else tries
            // to delete those particular fields
            req.privilege_fields.swap(it.second.first.privilege_fields);
            req.handle_type = LEGION_SINGULAR_PROJECTION;
            req.flags = it.second.first.flags;
            parent_req_indexes.emplace_back(it.second.second);
            returnable.emplace_back(
                returnable_privileges[it.second.second].second);
          }
        }
        // Remove the region from the created set
        {
          std::map<LogicalRegion, unsigned>::iterator finder =
              created_regions.find(handle);
          if (finder == created_regions.end())
          {
            std::map<LogicalRegion, bool>::iterator local_finder =
                local_regions.find(handle);
            legion_assert(local_finder != local_regions.end());
            legion_assert(local_finder->second);
            local_regions.erase(local_finder);
          }
          else
          {
            legion_assert(finder->second == 0);
            created_regions.erase(finder);
          }
        }
        // Check to see if we have any latent field spaces to clean up
        if (!latent_field_spaces.empty())
        {
          std::map<FieldSpace, std::set<LogicalRegion>>::iterator finder =
              latent_field_spaces.find(handle.get_field_space());
          if (finder != latent_field_spaces.end())
          {
            std::set<LogicalRegion>::iterator region_finder =
                finder->second.find(handle);
            legion_assert(region_finder != finder->second.end());
            finder->second.erase(region_finder);
            if (finder->second.empty())
            {
              // Now that all the regions using this field space have
              // been deleted we can clean up all the created_fields
              for (std::set<std::pair<FieldSpace, FieldID>>::iterator fit =
                       created_fields.begin();
                   fit != created_fields.end();
                   /*nothing*/)
              {
                if (fit->first == finder->first)
                {
                  std::set<std::pair<FieldSpace, FieldID>>::iterator to_delete =
                      fit++;
                  created_fields.erase(to_delete);
                }
                else
                  fit++;
              }
              latent_field_spaces.erase(finder);
            }
          }
        }
      }
      else
      {
        AutoLock priv_lock(privilege_lock);
        for (std::pair<
                 const LogicalRegion, std::pair<RegionRequirement, unsigned>>&
                 it : created_requirements)
        {
          // Has to match precisely
          if (handle.get_tree_id() == it.second.first.region.get_tree_id())
          {
            // Should be the same region
            legion_assert(handle == it.second.first.region);
            legion_assert(
                returnable_privileges.find(it.second.second) !=
                returnable_privileges.end());
            delete_reqs.resize(delete_reqs.size() + 1);
            RegionRequirement& req = delete_reqs.back();
            req.region = it.second.first.region;
            req.parent = it.second.first.region;
            req.privilege = LEGION_READ_WRITE;
            req.prop = LEGION_EXCLUSIVE;
            // Swap the privilege fields so that nothing else tries
            // to delete those particular fields
            req.privilege_fields.swap(it.second.first.privilege_fields);
            req.handle_type = LEGION_SINGULAR_PROJECTION;
            parent_req_indexes.emplace_back(it.second.second);
            returnable.emplace_back(
                returnable_privileges[it.second.second].second);
          }
        }
        // Remove the region from the created set
        {
          std::map<LogicalRegion, unsigned>::iterator finder =
              created_regions.find(handle);
          if (finder == created_regions.end())
          {
            std::map<LogicalRegion, bool>::iterator local_finder =
                local_regions.find(handle);
            legion_assert(local_finder != local_regions.end());
            legion_assert(local_finder->second);
            local_regions.erase(local_finder);
          }
          else
          {
            legion_assert(finder->second == 0);
            created_regions.erase(finder);
          }
        }
        // Check to see if we have any latent field spaces to clean up
        if (!latent_field_spaces.empty())
        {
          std::map<FieldSpace, std::set<LogicalRegion>>::iterator finder =
              latent_field_spaces.find(handle.get_field_space());
          if (finder != latent_field_spaces.end())
          {
            std::set<LogicalRegion>::iterator region_finder =
                finder->second.find(handle);
            legion_assert(region_finder != finder->second.end());
            finder->second.erase(region_finder);
            if (finder->second.empty())
            {
              // Now that all the regions using this field space have
              // been deleted we can clean up all the created_fields
              for (std::set<std::pair<FieldSpace, FieldID>>::iterator fit =
                       created_fields.begin();
                   fit != created_fields.end();
                   /*nothing*/)
              {
                if (fit->first == finder->first)
                {
                  std::set<std::pair<FieldSpace, FieldID>>::iterator to_delete =
                      fit++;
                  created_fields.erase(to_delete);
                }
                else
                  fit++;
              }
              latent_field_spaces.erase(finder);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_physical_region(
        const RegionRequirement& req, bool mapped, MapperID mid,
        MappingTagID tag, ApUserEvent& unmap_event, bool is_virtual_mapped,
        const InstanceSet& physical_instances)
    //--------------------------------------------------------------------------
    {
      legion_assert(!unmap_event.exists());
      if (!is_virtual_mapped)
        unmap_event = Runtime::create_ap_user_event(nullptr);
      PhysicalRegionImpl* impl = new PhysicalRegionImpl(
          req, RtEvent::NO_RT_EVENT, ApEvent::NO_AP_EVENT,
          mapped ? unmap_event : ApUserEvent::NO_AP_USER_EVENT, mapped, this,
          mid, tag, false /*leaf region*/, is_virtual_mapped,
          false /*never collective*/, NO_BLOCKING_INDEX,
          physical_region_count++);
      physical_regions.emplace_back(PhysicalRegion(impl));
      if (!is_virtual_mapped)
      {
#ifdef LEGION_DEBUG
        if (owner_task->is_remote())
        {
          // If the owner task is remote, then we need to acquire the
          // valid references first since the valid references are held
          // on the owner node and the checking code wants to see that
          // these instances are already valid when adding the references
          if (!physical_instances.acquire_valid_references(CONTEXT_REF))
          {
            Fatal fatal;
            fatal << "Found an internal garbage collection race. Please "
                     "run with -lg:safe_mapper and see if it reports any "
                     "errors. If not, then please report this as a bug.";
            fatal.raise();
          }
          impl->set_references(physical_instances, true /*safe*/);
          // Remove the references we acquired after they've been added
          // by the physical region
          physical_instances.remove_valid_references(CONTEXT_REF);
        }
        else
#endif
          impl->set_references(physical_instances, true /*safe*/);
      }
    }

    //--------------------------------------------------------------------------
    Future InnerContext::execute_task(
        const TaskLauncher& launcher, std::vector<OutputRequirement>* outputs,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
        return predicate_task_false(launcher, provenance);
      IndividualTask* task = runtime->get_operation<IndividualTask>();
      Future result = task->initialize_task(
          this, launcher, provenance, false /*top level*/, false /*must epoch*/,
          outputs);
      execute_task_launch(
          task, false /*index*/, launcher.static_dependences, provenance,
          launcher.silence_warnings, launcher.enable_inlining);
      return result;
    }

    //--------------------------------------------------------------------------
    uint64_t InnerContext::get_next_blocking_index(void)
    //--------------------------------------------------------------------------
    {
      return next_blocking_index++;
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::execute_index_space(
        const IndexTaskLauncher& launcher,
        std::vector<OutputRequirement>* outputs, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.must_parallelism)
      {
        // Turn around and use a must epoch launcher
        MustEpochLauncher epoch_launcher(launcher.map_id, launcher.tag);
        epoch_launcher.add_index_task(launcher);
        epoch_launcher.provenance = launcher.provenance;
        FutureMap result = execute_must_epoch(epoch_launcher, provenance);
        return result;
      }
      if (launcher.launch_domain.exists() &&
          (launcher.launch_domain.get_volume() == 0))
      {
        Warning warning;
        warning << "Ignoring empty index task launch in task "
                << get_task_name() << " (ID " << get_unique_id() << ")";
        warning.raise();
        return FutureMap();
      }
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
        launch_space =
            find_index_launch_space(launcher.launch_domain, provenance);
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
        return predicate_index_task_false(launch_space, launcher, provenance);
      IndexTask* task = runtime->get_operation<IndexTask>();
      FutureMap result = task->initialize_task(
          this, launcher, launch_space, provenance, true /*track*/, outputs);
      execute_task_launch(
          task, true /*index*/, launcher.static_dependences, provenance,
          launcher.silence_warnings, launcher.enable_inlining);
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::execute_index_space(
        const IndexTaskLauncher& launcher, ReductionOpID redop,
        bool deterministic, std::vector<OutputRequirement>* outputs,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.must_parallelism)
      {
        // Turn around and use a must epoch launcher
        MustEpochLauncher epoch_launcher(launcher.map_id, launcher.tag);
        epoch_launcher.add_index_task(launcher);
        epoch_launcher.provenance = launcher.provenance;
        FutureMap result = execute_must_epoch(epoch_launcher, provenance);
        return reduce_future_map(
            result, redop, deterministic, launcher.map_id, launcher.tag,
            provenance, launcher.initial_value);
      }
      if (launcher.launch_domain.exists() &&
          (launcher.launch_domain.get_volume() == 0))
      {
        Warning warning;
        warning << "Ignoring empty index task launch in task "
                << get_task_name() << " (ID " << get_unique_id() << ")";
        warning.raise();
        if (!launcher.initial_value.is_empty())
          return launcher.initial_value;

        // Else return the reduction operation's identity value
        const ReductionOp* reduction_op = runtime->get_reduction(redop);
        FutureImpl* result = new FutureImpl(
            this, true /*register*/, runtime->get_available_distributed_id(),
            provenance);
        result->set_local(
            reduction_op->identity, reduction_op->sizeof_rhs, false /*own*/);
        return Future(result);
      }
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
        launch_space =
            find_index_launch_space(launcher.launch_domain, provenance);
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
        return predicate_index_task_reduce_false(
            launcher, launch_space, redop, provenance);
      IndexTask* task = runtime->get_operation<IndexTask>();
      Future result = task->initialize_task(
          this, launcher, launch_space, provenance, redop, deterministic,
          true /*track*/, outputs);
      execute_task_launch(
          task, true /*index*/, launcher.static_dependences, provenance,
          launcher.silence_warnings, launcher.enable_inlining);
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::reduce_future_map(
        const FutureMap& future_map, ReductionOpID redop, bool deterministic,
        MapperID mapper_id, MappingTagID tag, Provenance* prov,
        Future initial_value)
    //--------------------------------------------------------------------------
    {
      if (future_map.impl == nullptr)
      {
        const ReductionOp* reduction_op = runtime->get_reduction(redop);
        FutureImpl* result = new FutureImpl(
            this, true /*register*/, runtime->get_available_distributed_id(),
            prov);
        result->set_local(
            reduction_op->identity, reduction_op->sizeof_rhs, false /*own*/);
        return Future(result);
      }
      AllReduceOp* all_reduce_op = runtime->get_operation<AllReduceOp>();
      Future result = all_reduce_op->initialize(
          this, future_map, redop, deterministic, mapper_id, tag, prov,
          initial_value);
      add_to_dependence_queue(all_reduce_op);
      return result;
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::construct_future_map(
        IndexSpace space, const std::map<DomainPoint, UntypedBuffer>& data,
        Provenance* provenance, bool collective, ShardingID sid, bool implicit,
        bool check_space)
    //--------------------------------------------------------------------------
    {
      Domain domain;
      runtime->find_domain(space, domain);
      if (data.size() != domain.get_volume())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "The number of buffers passed into a future map construction ("
              << data.size() << ") does not match the volume of the domain ("
              << domain.get_volume() << ") for the future map in task "
              << get_task_name() << " (UID " << get_unique_id() << ").";
        error.raise();
      }
      const DistributedID did = runtime->get_available_distributed_id();
      IndexSpaceNode* launch_node = runtime->get_node(space);
      FutureMapImpl* impl = new FutureMapImpl(
          this, launch_node, did, NO_BLOCKING_INDEX, std::optional<uint64_t>(),
          provenance);
      for (const std::pair<const DomainPoint, UntypedBuffer>& it : data)
      {
        if (!domain.contains(it.first))
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Point passed into future map construction is not contained "
                   "within the bounds of the domain in task "
                << get_task_name() << " (UID " << get_unique_id() << ")";
          error.raise();
        }
        const size_t future_size = it.second.get_size();
        FutureImpl* future = new FutureImpl(
            this, true /*register*/, runtime->get_available_distributed_id(),
            provenance);
        future->set_local(it.second.get_ptr(), future_size);
        impl->set_future(it.first, future);
      }
      return FutureMap(impl);
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::construct_future_map(
        const Domain& domain, const std::map<DomainPoint, UntypedBuffer>& data,
        bool collective, ShardingID sid, bool implicit)
    //--------------------------------------------------------------------------
    {
      // this method is deprecated so don't care about provenance
      // make sure we don't do any control replication checks on the
      // space since we can't guarantee it is the same across the shards
      return construct_future_map(
          find_index_launch_space(domain, nullptr /*prov*/), data,
          nullptr /*deprecated so no provenance*/, collective, sid, implicit,
          false /*check space*/);
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::construct_future_map(
        IndexSpace space, const std::map<DomainPoint, Future>& futures,
        Provenance* provenance, bool collective, ShardingID sid, bool implicit,
        bool check_space)
    //--------------------------------------------------------------------------
    {
      CreationOp* creation_op = runtime->get_operation<CreationOp>();
      creation_op->initialize_map(this, provenance, futures);
      const DistributedID did = runtime->get_available_distributed_id();
      IndexSpaceNode* launch_node = runtime->get_node(space);
      if (futures.size() != launch_node->get_volume())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "The number of futures passed into a future map construction ("
              << futures.size() << ") does not match the volume of the domain ("
              << launch_node->get_volume() << ") for the future map in task "
              << get_task_name() << " (UID " << get_unique_id() << ").";
        error.raise();
      }
      FutureMapImpl* impl =
          new FutureMapImpl(this, creation_op, launch_node, did, provenance);
      add_to_dependence_queue(creation_op);
      impl->set_all_futures(futures);
      return FutureMap(impl);
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::construct_future_map(
        const Domain& domain, const std::map<DomainPoint, Future>& futures,
        bool collective, ShardingID sid, bool implicit)
    //--------------------------------------------------------------------------
    {
      // Make sure we don't do any control replication checks on the
      // space here since it might not be the same across the shards
      return construct_future_map(
          find_index_launch_space(domain, nullptr), futures,
          nullptr /*deprecated so no provenance*/, collective, sid, implicit,
          false /*check space*/);
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::transform_future_map(
        const FutureMap& fm, IndexSpace new_domain,
        PointTransformFunctor* functor, bool own_func, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (fm.impl == nullptr)
        return fm;
      IndexSpaceNode* new_node = runtime->get_node(new_domain);
      return FutureMap(new TransformFutureMapImpl(
          fm.impl, new_node, functor, own_func, provenance));
    }

    //--------------------------------------------------------------------------
    PhysicalRegion InnerContext::map_region(
        const InlineLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (IS_NO_ACCESS(launcher.requirement))
        return PhysicalRegion();
      MapOp* map_op = runtime->get_operation<MapOp>();
      PhysicalRegion result = map_op->initialize(
          this, launcher, provenance, physical_region_count++);
      if (current_trace != nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Attempted an inline mapping of region ("
              << launcher.requirement.region.index_space.get_id() << ","
              << launcher.requirement.region.field_space.get_id() << ","
              << launcher.requirement.region.get_tree_id()
              << ") inside of trace " << current_trace->tid
              << " of parent task " << get_task_name() << " (ID "
              << get_unique_id()
              << "). It is illegal to perform inline mapping operations inside "
                 "of traces.";
        error.raise();
      }
      bool parent_conflict = false, inline_conflict = false;
      const int index =
          has_conflicting_regions(map_op, parent_conflict, inline_conflict);
      if (parent_conflict)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Attempted an inline mapping of region ("
              << launcher.requirement.region.index_space.get_id() << ","
              << launcher.requirement.region.field_space.get_id() << ","
              << launcher.requirement.region.get_tree_id()
              << ") that conflicts with mapped region ("
              << regions[index].region.index_space.get_id() << ","
              << regions[index].region.field_space.get_id() << ","
              << regions[index].region.get_tree_id() << ") at index " << index
              << " of parent task " << get_task_name() << " (ID "
              << get_unique_id()
              << ") that would ultimately result in deadlock. Instead you "
                 "receive this error message.";
        error.raise();
      }
      if (inline_conflict)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Attempted an inline mapping of region ("
              << launcher.requirement.region.index_space.get_id() << ","
              << launcher.requirement.region.field_space.get_id() << ","
              << launcher.requirement.region.get_tree_id()
              << ") that conflicts with previous inline mapping in task "
              << get_task_name() << " (ID " << get_unique_id()
              << ") that would ultimately result in deadlock.  Instead you "
                 "receive this error message.";
        error.raise();
      }
      register_inline_mapped_region(result);
      add_to_dependence_queue(map_op, launcher.static_dependences);
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent InnerContext::remap_region(
        const PhysicalRegion& region, Provenance* provenance, bool internal)
    //--------------------------------------------------------------------------
    {
      // Check to see if the region is already mapped,
      // if it is then we are done
      if (region.is_mapped())
        return ApEvent::NO_AP_EVENT;
      if (current_trace != nullptr)
      {
        const RegionRequirement& req = region.impl->get_requirement();
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Attempted an inline mapping of region ("
              << req.region.index_space.get_id() << ","
              << req.region.field_space.get_id() << ","
              << req.region.get_tree_id() << ") inside of trace "
              << current_trace->tid << " of parent task " << get_task_name()
              << " (ID " << get_unique_id()
              << "). It is illegal to perform inline mapping operations inside "
                 "of traces.";
        error.raise();
      }
      MapOp* map_op = runtime->get_operation<MapOp>();
      map_op->initialize(this, region, provenance);
      register_inline_mapped_region(region);
      const ApEvent result = map_op->get_completion_event();
      add_to_dependence_queue(map_op);
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::unmap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      if (!region.is_mapped())
        return;
      region.impl->unmap_region(false /*escaped*/);
      unregister_inline_mapped_region(region);
    }

    //--------------------------------------------------------------------------
    void InnerContext::unmap_all_regions(bool external)
    //--------------------------------------------------------------------------
    {
      for (const PhysicalRegion& region : physical_regions)
      {
        if (region.is_mapped())
          region.impl->unmap_region(!external);
      }
      // Also unmap any of our inline mapped physical regions
      AutoLock i_lock(inline_lock);
      for (const std::pair<const RegionTreeID, ctx::list<PhysicalRegion>>&
               tree_regions : inline_regions)
      {
        for (const PhysicalRegion& region : tree_regions.second)
        {
          if (region.is_mapped())
            region.impl->unmap_region(!external);
        }
      }
      if (!external)
        inline_regions.clear();
    }

    //--------------------------------------------------------------------------
    void InnerContext::fill_fields(
        const FillLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.fields.empty())
      {
        Warning warning;
        warning << "Ignoring fill request with no fields in task "
                << get_task_name() << " (UID " << get_unique_id() << ")";
        warning.raise();
        return;
      }
      FillOp* fill_op = runtime->get_operation<FillOp>();
      fill_op->initialize(this, launcher, provenance);
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(fill_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          Warning warning;
          warning << "Runtime is unmapping and remapping physical regions "
                     "around fill_fields call in task "
                  << get_task_name() << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(fill_op, launcher.static_dependences);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    void InnerContext::fill_fields(
        const IndexFillLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.fields.empty())
      {
        Warning warning;
        warning << "Ignoring index fill request with no fields in task "
                << get_task_name() << " (UID " << get_unique_id() << ")";
        warning.raise();
        return;
      }
      if (launcher.launch_domain.exists() &&
          (launcher.launch_domain.get_volume() == 0))
      {
        Warning warning;
        warning << "Ignoring empty index space fill in task " << get_task_name()
                << " (ID " << get_unique_id() << ")";
        warning.raise();
        return;
      }
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
        launch_space =
            find_index_launch_space(launcher.launch_domain, provenance);
      IndexFillOp* fill_op = runtime->get_operation<IndexFillOp>();
      fill_op->initialize(this, launcher, launch_space, provenance);
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(fill_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          Warning warning;
          warning << "Runtime is unmapping and remapping physical regions "
                     "around fill_fields call in task "
                  << get_task_name() << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(fill_op, launcher.static_dependences);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    void InnerContext::discard_fields(
        const DiscardLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.fields.empty())
      {
        Warning warning;
        warning << "Ignoring discard request with no fields in task "
                << get_task_name() << " (UID " << get_unique_id() << ")";
        warning.raise();
        return;
      }
      DiscardOp* discard_op = runtime->get_operation<DiscardOp>();
      discard_op->initialize(this, launcher, provenance);
      // We still unamp conflicting regions for discard, but we wil never
      // remap them afterwards since this is invalidating the data
      if (!runtime->unsafe_launch)
      {
        std::vector<PhysicalRegion> unmapped_regions;
        find_conflicting_regions(discard_op, unmapped_regions);
        if (!unmapped_regions.empty())
        {
          if (runtime->runtime_warnings && !launcher.silence_warnings)
          {
            Warning warning;
            warning << "Runtime is unmapping and remapping physical regions "
                       "around discard_fields call in task "
                    << get_task_name() << " (UID " << get_unique_id() << ").";
            warning.raise();
          }
          // Unmap any regions which are conflicting
          for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
            unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
        }
      }
      add_to_dependence_queue(discard_op, launcher.static_dependences);
      // Do not remap the previously mapped regions, they are uninitialized
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_copy(
        const CopyLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      CopyOp* copy_op = runtime->get_operation<CopyOp>();
      copy_op->initialize(this, launcher, provenance);
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(copy_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          Warning warning;
          warning << "Runtime is unmapping and remapping physical regions "
                     "around issue_copy_operation call in task "
                  << get_task_name() << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(copy_op, launcher.static_dependences);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_copy(
        const IndexCopyLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.launch_domain.exists() &&
          (launcher.launch_domain.get_volume() == 0))
      {
        Warning warning;
        warning << "Ignoring empty index space copy in task " << get_task_name()
                << " (ID " << get_unique_id() << ")";
        warning.raise();
        return;
      }
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
        launch_space =
            find_index_launch_space(launcher.launch_domain, provenance);
      IndexCopyOp* copy_op = runtime->get_operation<IndexCopyOp>();
      copy_op->initialize(this, launcher, launch_space, provenance);
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(copy_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          Warning warning;
          warning << "Runtime is unmapping and remapping physical regions "
                     "around issue_copy_operation call in task "
                  << get_task_name() << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(copy_op, launcher.static_dependences);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_acquire(
        const AcquireLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      AcquireOp* acquire_op = runtime->get_operation<AcquireOp>();
      acquire_op->initialize(this, launcher, provenance);
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this acquire operation.
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(acquire_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          Warning warning;
          warning << "Runtime is unmapping and remapping physical regions "
                     "around issue_acquire call in task "
                  << get_task_name() << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the acquire operation
      add_to_dependence_queue(acquire_op, launcher.static_dependences);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_release(
        const ReleaseLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      ReleaseOp* release_op = runtime->get_operation<ReleaseOp>();
      release_op->initialize(this, launcher, provenance);
      // Check to see if we need to do any unmappings and remappings
      // before we can issue the release operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(release_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          Warning warning;
          warning << "Runtime is unmapping and remapping physical regions "
                     "around issue_release call in task "
                  << get_task_name() << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the release operation
      add_to_dependence_queue(release_op, launcher.static_dependences);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion InnerContext::attach_resource(
        const AttachLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      AttachOp* attach_op = runtime->get_operation<AttachOp>();
      PhysicalRegion result = attach_op->initialize(
          this, launcher, provenance, physical_region_count++);
      bool parent_conflict = false, inline_conflict = false;
      int index =
          has_conflicting_regions(attach_op, parent_conflict, inline_conflict);
      if (parent_conflict)
      {
        Fatal fatal;
        fatal << "Attempted an external attach operation on region "
              << launcher.handle << " that conflicts with mapped region "
              << regions[index].region << " at index " << index
              << " of parent task " << get_task_name() << " (ID "
              << get_unique_id()
              << ") that would ultimately result in deadlock. Instead you "
                 "receive this error message. Try unmapping the region before "
                 "invoking 'attach_external_resource'.";
        fatal.raise();
      }
      if (inline_conflict)
      {
        Fatal fatal;
        fatal << "Attempted an external attach operation on region "
              << launcher.handle
              << " that conflicts with previous inline mapping in task "
              << get_task_name() << " (ID " << get_unique_id()
              << ") that would ultimately result in deadlock. Instead you "
                 "receive this error message. Try unmapping the region before "
                 "invoking 'attach_external_resource'.";
        fatal.raise();
      }
      // Add this region to the list of inline mapped regions if it is mapped
      if (result.is_mapped())
        register_inline_mapped_region(result);
      add_to_dependence_queue(attach_op, launcher.static_dependences);
      return result;
    }

    //--------------------------------------------------------------------------
    ExternalResources InnerContext::attach_resources(
        const IndexAttachLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.handles.empty())
        return ExternalResources();
      // This is not control replicated so no need to deduplicate anything
      std::vector<unsigned> indexes(launcher.handles.size());
      for (unsigned idx = 0; idx < indexes.size(); idx++) indexes[idx] = idx;
      // Compute the upper bound partition node from this launcher
      RegionTreeNode* node =
          compute_index_attach_upper_bound(launcher, indexes);
      IndexSpaceNode* launch_space = runtime->get_node(find_index_launch_space(
          Domain(Point<1>(0), Point<1>(indexes.size() - 1)), provenance));
      IndexAttachOp* attach_op = runtime->get_operation<IndexAttachOp>();
      ExternalResources result = attach_op->initialize(
          this, node, launch_space, launcher, indexes, provenance,
          false /*replicated*/, physical_region_count);
      physical_region_count += indexes.size();
      const RegionRequirement& req = attach_op->get_requirement();
      bool parent_conflict = false, inline_conflict = false;
      int index =
          has_conflicting_internal(req, parent_conflict, inline_conflict);
      if (parent_conflict)
      {
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
        {
          Fatal fatal;
          fatal << "Attempted an index attach operation with upper bound "
                   "partition "
                << req.partition << " that conflicts with mapped region "
                << regions[index].region << " at index " << index
                << " of parent task " << get_task_name() << " (ID "
                << get_unique_id()
                << ") that would ultimately result in deadlock. Instead you "
                   "receive this error message. Try unmapping the region "
                   "before invoking 'attach_external_resources'.";
          fatal.raise();
        }
        else
        {
          Fatal fatal;
          fatal
              << "Attempted an index attach operation with upper bound region "
              << req.region << " that conflicts with mapped region "
              << regions[index].region << " at index " << index
              << " of parent task " << get_task_name() << " (ID "
              << get_unique_id()
              << ") that would ultimately result in deadlock. Instead you "
                 "receive this error message. Try unmapping the region before "
                 "invoking 'attach_external_resources'.";
          fatal.raise();
        }
      }
      if (inline_conflict)
      {
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
        {
          Fatal fatal;
          fatal << "Attempted an index attach operation with upper bound "
                   "partition "
                << req.partition
                << " that conflicts with previous inline mapping in task "
                << get_task_name() << " (ID " << get_unique_id()
                << ") that would ultimately result in deadlock. Instead you "
                   "receive this error message. Try unmapping the region "
                   "before invoking 'attach_external_resources'.";
          fatal.raise();
        }
        else
        {
          Fatal fatal;
          fatal
              << "Attempted an index attach operation with upper bound region "
              << req.region
              << " that conflicts with previous inline mapping in task "
              << get_task_name() << " (ID " << get_unique_id()
              << ") that would ultimately result in deadlock. Instead you "
                 "receive this error message. Try unmapping the region before "
                 "invoking 'attach_external_resources'.";
          fatal.raise();
        }
      }
      add_to_dependence_queue(attach_op, launcher.static_dependences);
      return result;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* InnerContext::compute_index_attach_upper_bound(
        const IndexAttachLauncher& launcher,
        const std::vector<unsigned>& indexes)
    //--------------------------------------------------------------------------
    {
      std::vector<RegionTreeNode*> previous_nodes(indexes.size());
      std::vector<unsigned> depths(indexes.size());
      unsigned max_depth = 0;
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        const unsigned index = indexes[idx];
        LogicalRegion handle = launcher.handles[index];
        if (handle.get_tree_id() != launcher.parent.get_tree_id())
        {
          Fatal fatal;
          fatal << "Handle " << handle
                << " of index attach operation in parent task "
                << get_task_name() << " (UID " << get_unique_id()
                << ") does not come from the same region tree as the parent "
                   "region "
                << launcher.parent
                << ". All regions for an index space attach must be from the "
                   "same tree.";
          fatal.raise();
        }
        previous_nodes[idx] = runtime->get_node(handle);
        depths[idx] = previous_nodes[idx]->get_depth();
        if (max_depth < depths[idx])
          max_depth = depths[idx];
      }
      // Walk all the nodes up from the bottom until they arrive at a
      // common ancestor, along the way check to make sure that any nodes
      // that arrive at a common join point from two different paths do
      // so at a disjoint partition
      std::vector<RegionTreeNode*> next_nodes(indexes.size());
      while (max_depth > 0)
      {
        std::map<RegionTreeNode*, std::vector<unsigned>> next_to_previous;
        bool all_same = true;
        for (unsigned idx = 0; idx < indexes.size(); idx++)
        {
          if (depths[idx] == max_depth)
          {
            depths[idx]--;
            next_nodes[idx] = previous_nodes[idx]->get_parent();
            next_to_previous[next_nodes[idx]].emplace_back(idx);
            if (all_same && (idx > 0) &&
                (next_nodes[idx - 1] != next_nodes[idx]))
              all_same = false;
          }
          else
          {
            next_nodes[idx] = previous_nodes[idx];
            all_same = false;
          }
        }
        // check to see if all the next to previous cases play by the rules
        for (const std::map<RegionTreeNode*, std::vector<unsigned>>::value_type&
                 it : next_to_previous)
        {
          if (it.second.size() == 1)
            continue;
          // Can skip any disjoint partitions since it doesn't matter where
          // their children came from
          if (!it.first->is_region() &&
              it.first->as_partition_node()->row_source->is_disjoint())
            continue;
          // Otherwise check to see that they all came from the same child
          // If they didn't, then we can't prove tree disjointness
          RegionTreeNode* previous = previous_nodes[it.second.front()];
          for (unsigned idx = 1; idx < it.second.size(); idx++)
          {
            if (previous == previous_nodes[it.second[idx]])
              continue;
            const LogicalRegion h1 = launcher.handles[it.second.front()];
            const LogicalRegion h2 = launcher.handles[it.second[idx]];
            Fatal fatal;
            fatal << "Logical region handle " << h1 << " from index "
                  << it.second.front()
                  << " of index attach operation in parent task "
                  << get_task_name() << " (UID " << get_unique_id()
                  << ") is not region-tree disjoint with logical region handle "
                  << h2 << " from index " << it.second[idx]
                  << ". All regions in index space attach operations must be "
                     "region-tree disjoint.";
            fatal.raise();
          }
        }
        previous_nodes.swap(next_nodes);
        if (all_same)
          break;
        max_depth--;
      }
      // At this point all the previous nodes should be the same
      return previous_nodes.back();
    }

    //--------------------------------------------------------------------------
    ProjectionID InnerContext::compute_index_attach_projection(
        IndexTreeNode* upper_bound, IndexAttachOp* op, unsigned local_start,
        size_t local_size, std::vector<IndexSpace>& spaces,
        const bool can_use_identity)
    //--------------------------------------------------------------------------
    {
      std::map<IndexTreeNode*, std::vector<AttachProjectionFunctor*>>::iterator
          finder = attach_functions.find(upper_bound);
      if (finder != attach_functions.end())
      {
        for (AttachProjectionFunctor* const & it : finder->second)
        {
          if (it->handles.size() != spaces.size())
            continue;
          bool equal = true;
          for (unsigned idx = 0; idx < spaces.size(); idx++)
          {
            if (it->handles[idx] == spaces[idx])
              continue;
            equal = false;
            break;
          }
          if (equal)
            return it->pid;
        }
      }
      else  // instantiate the entry in the map
        finder = attach_functions
                     .insert(std::make_pair(
                         upper_bound, std::vector<AttachProjectionFunctor*>()))
                     .first;
      // If the upper bound is a partition, do a quick check to see if
      // all the spaces are immediate children of the upper bound, if
      // so then we can use projection function 0
      if (!upper_bound->is_index_space_node() && can_use_identity)
      {
        bool all_children = true;
        IndexPartNode* parent = upper_bound->as_index_part_node();
        for (unsigned idx = 0; idx < local_size; idx++)
        {
          IndexSpaceNode* child = runtime->get_node(spaces[local_start + idx]);
          if (child->parent == parent)
            continue;
          all_children = false;
          break;
        }
        // Bounce this off the operation in case we are control replicated
        all_children = op->are_all_direct_children(all_children);
        if (all_children)
        {
          // We can use the identity projection in this case
          // so just make it, but no need to register it with the runtime
          finder->second.emplace_back(
              new AttachProjectionFunctor(0 /*identity*/, std::move(spaces)));
          return 0;
        }
      }
      // If we get here then we need to make it
      // Generate a fresh dynamic ID and store it
      const ProjectionID result =
          runtime->generate_dynamic_projection_id(false /*check context*/);
      AttachProjectionFunctor* functor =
          new AttachProjectionFunctor(result, std::move(spaces));
      runtime->register_projection_functor(
          result, functor, false /*check*/, true /*silence warnings*/);
      finder->second.emplace_back(functor);
      LegionSpy::log_projection_function(
          result, functor->get_depth(), functor->is_invertible());
      return result;
    }

    //--------------------------------------------------------------------------
    InnerContext::AttachProjectionFunctor::AttachProjectionFunctor(
        ProjectionID p, std::vector<IndexSpace>&& spaces)
      : ProjectionFunctor(Internal::runtime->external), handles(spaces), pid(p)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    LogicalRegion InnerContext::AttachProjectionFunctor::project(
        LogicalRegion upper_bound, const DomainPoint& point,
        const Domain& launch)
    //--------------------------------------------------------------------------
    {
      const unsigned offset = compute_offset(point, launch);
      legion_assert(offset < handles.size());
      return runtime->get_logical_subregion_by_tree(
          handles[offset], upper_bound.get_field_space(),
          upper_bound.get_tree_id());
    }

    //--------------------------------------------------------------------------
    LogicalRegion InnerContext::AttachProjectionFunctor::project(
        LogicalPartition upper, const DomainPoint& point, const Domain& launch)
    //--------------------------------------------------------------------------
    {
      const unsigned offset = compute_offset(point, launch);
      legion_assert(offset < handles.size());
      return runtime->get_logical_subregion_by_tree(
          handles[offset], upper.get_field_space(), upper.get_tree_id());
    }

    //--------------------------------------------------------------------------
    /*static*/ unsigned InnerContext::AttachProjectionFunctor::compute_offset(
        const DomainPoint& point, const Domain& launch)
    //--------------------------------------------------------------------------
    {
      if (point.get_dim() == 2)
      {
        const Point<2> p = point;
        // Control replication case, see if we're compacted or not
        if (launch.dense() && (launch.lo()[0] == 0))
        {
          // Dense means that all the shards had the same number of points
          // so we can compute where our offset is based on that
          const Rect<2> bounds = launch;
          return p[0] * (bounds.hi[1] - bounds.lo[1] + 1) + p[1];
        }
        else
        {
          // We computed prefix sums for the non-dense case so whatever
          // the second dimension is is the one that says which space we are
          return p[1];
        }
      }
      else
      {
        const Point<1> p = point;
        return p[0];
      }
    }

    //--------------------------------------------------------------------------
    Future InnerContext::detach_resource(
        PhysicalRegion region, const bool flush, const bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // Unmap the region here so that it is safe for re-use
      if (region.impl->is_mapped())
      {
        region.impl->unmap_region(false /*escaped*/);
        // Remove this region from the list of inline regions if it is mapped
        unregister_inline_mapped_region(region);
      }
      DetachOp* op = runtime->get_operation<DetachOp>();
      Future result =
          op->initialize_detach(this, region, flush, unordered, provenance);
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        Fatal fatal;
        fatal
            << "Illegal unordered detach operation performed after task "
            << get_task_name() << " (UID " << get_unique_id()
            << ") has finished executing. All unordered operations must be "
               "performed before the end of the execution of the parent task.";
        fatal.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::detach_resources(
        ExternalResources resources, const bool flush, const bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (resources.impl == nullptr)
        return Future();
      IndexDetachOp* op = runtime->get_operation<IndexDetachOp>();
      Future result =
          resources.impl->detach(this, op, flush, unordered, provenance);
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        Fatal fatal;
        fatal
            << "Illegal unordered index detach operation performed after task "
            << get_task_name() << " (UID " << get_unique_id()
            << ") has finished executing. All unordered operations must be "
               "performed before the end of the execution of the parent task.";
        fatal.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::progress_unordered_operations(bool end_task)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(dependence_lock);
      if (end_task)
      {
        // This is the end of this parent task so mark that we're done
        legion_assert(!finished_execution);
        legion_assert(current_trace == nullptr);
        finished_execution = true;
      }
      // No progress can occur inside of a trace
      else if (current_trace != nullptr)
        return;
      insert_unordered_ops(d_lock);
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::execute_must_epoch(
        const MustEpochLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      MustEpochOp* epoch_op = runtime->get_operation<MustEpochOp>();
      FutureMap result = epoch_op->initialize(this, launcher, provenance);
      // Now find all the parent task regions we need to invalidate
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        epoch_op->find_conflicted_regions(unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          Warning warning;
          warning << "Runtime is unmapping and remapping physical regions "
                     "around issue_release call in task "
                  << get_task_name() << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Now we can issue the must epoch
      add_to_dependence_queue(epoch_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::issue_timing_measurement(
        const TimingLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      TimingOp* timing_op = runtime->get_operation<TimingOp>();
      Future result = timing_op->initialize(this, launcher, provenance);
      add_to_dependence_queue(timing_op);
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::select_tunable_value(
        const TunableLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      TunableOp* tunable_op = runtime->get_operation<TunableOp>();
      Future result = tunable_op->initialize(this, launcher, provenance);
      add_to_dependence_queue(tunable_op);
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::issue_mapping_fence(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      FenceOp* fence_op = runtime->get_operation<FenceOp>();
      Future f = fence_op->initialize(
          this, FenceOp::MAPPING_FENCE, true /*return future*/, provenance);
      add_to_dependence_queue(fence_op);
      return f;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::issue_execution_fence(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      FenceOp* fence_op = runtime->get_operation<FenceOp>();
      Future f = fence_op->initialize(
          this, FenceOp::EXECUTION_FENCE, true /*return future*/, provenance);
      add_to_dependence_queue(fence_op);
      return f;
    }

    //--------------------------------------------------------------------------
    void InnerContext::complete_frame(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      FrameOp* frame_op = runtime->get_operation<FrameOp>();
      frame_op->initialize(this, provenance);
      RtEvent wait_on;
      if (context_configuration.max_outstanding_frames > 0)
      {
        AutoLock child_lock(child_op_lock);
        frame_ops.emplace_back(frame_op);
        legion_assert(
            frame_ops.size() <=
            (size_t)context_configuration.max_outstanding_frames);
        if (frame_ops.size() ==
            (size_t)context_configuration.max_outstanding_frames)
          wait_on = frame_ops.front()->get_commit_event();
      }
      add_to_dependence_queue(frame_op);
      if (wait_on.exists())
        wait_on.wait();
    }

    //--------------------------------------------------------------------------
    Predicate InnerContext::create_predicate(
        const Future& f, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (f.impl == nullptr)
      {
        Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
        error << "Illegal predicate creation performed on "
                 "empty future inside of task "
              << *this << ".";
        error.raise();
      }
      FuturePredOp* pred_op = runtime->get_operation<FuturePredOp>();
      // Hold a reference before initialization
      Predicate result = pred_op->initialize(this, f, provenance);
      add_to_dependence_queue(pred_op);
      return result;
    }

    //--------------------------------------------------------------------------
    Predicate InnerContext::predicate_not(
        const Predicate& p, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (p == Predicate::TRUE_PRED)
        return Predicate::FALSE_PRED;
      if (p == Predicate::FALSE_PRED)
        return Predicate::TRUE_PRED;
      NotPredOp* pred_op = runtime->get_operation<NotPredOp>();
      // Hold a reference before initialization
      Predicate result = pred_op->initialize(this, p, provenance);
      add_to_dependence_queue(pred_op);
      return result;
    }

    //--------------------------------------------------------------------------
    Predicate InnerContext::create_predicate(
        const PredicateLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.predicates.empty())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal predicate creation performed on a "
              << "set of empty previous predicates in task " << *this << ".";
        error.raise();
      }
      else if (launcher.predicates.size() == 1)
        return launcher.predicates[0];
      if (launcher.and_op)
      {
        // Check for short circuit cases
        std::vector<Predicate> actual_predicates;
        for (const Predicate& it : launcher.predicates)
        {
          if (it == Predicate::FALSE_PRED)
            return Predicate::FALSE_PRED;
          else if (it == Predicate::TRUE_PRED)
            continue;
          actual_predicates.emplace_back(it);
        }
        if (actual_predicates.empty())  // they were all true
          return Predicate::TRUE_PRED;
        else if (actual_predicates.size() == 1)
          return actual_predicates[0];
        AndPredOp* pred_op = runtime->get_operation<AndPredOp>();
        // Hold a reference before initialization
        Predicate result =
            pred_op->initialize(this, actual_predicates, provenance);
        add_to_dependence_queue(pred_op);
        return result;
      }
      else
      {
        // Check for short circuit cases
        std::vector<Predicate> actual_predicates;
        for (const Predicate& it : launcher.predicates)
        {
          if (it == Predicate::TRUE_PRED)
            return Predicate::TRUE_PRED;
          else if (it == Predicate::FALSE_PRED)
            continue;
          actual_predicates.emplace_back(it);
        }
        if (actual_predicates.empty())  // they were all false
          return Predicate::FALSE_PRED;
        else if (actual_predicates.size() == 1)
          return actual_predicates[0];
        OrPredOp* pred_op = runtime->get_operation<OrPredOp>();
        // Hold a reference before initialization
        Predicate result =
            pred_op->initialize(this, actual_predicates, provenance);
        add_to_dependence_queue(pred_op);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    Future InnerContext::get_predicate_future(
        const Predicate& p, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (p == Predicate::TRUE_PRED)
      {
        Future result(new FutureImpl(
            this, true /*register*/, runtime->get_available_distributed_id(),
            provenance));
        const bool value = true;
        result.impl->set_local(&value, sizeof(value));
        return result;
      }
      else if (p == Predicate::FALSE_PRED)
      {
        Future result(new FutureImpl(
            this, true /*register*/, runtime->get_available_distributed_id(),
            provenance));
        const bool value = false;
        result.impl->set_local(&value, sizeof(value));
        return result;
      }
      else
      {
        legion_assert(p.impl != nullptr);
        FuturePredOp* pred_op = runtime->get_operation<FuturePredOp>();
        // Hold a reference before initialization
        Future result = pred_op->initialize(this, p, provenance);
        add_to_dependence_queue(pred_op);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    PredicateImpl* InnerContext::create_predicate_impl(Operation* op)
    //--------------------------------------------------------------------------
    {
      return new PredicateImpl(op);
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_prepipeline_queue(Operation* op)
    //--------------------------------------------------------------------------
    {
      bool issue_task = false;
      const GenerationID gen = op->get_generation();
      {
        AutoLock p_lock(prepipeline_lock);
        prepipeline_queue.emplace_back(std::make_pair(op, gen));
        // Cap the number of outstanding prepipeline tasks as no more than
        // the number of utility processors that we're running on
        if (outstanding_prepipeline_tasks < runtime->num_utility_procs)
        {
          const unsigned max_tasks =
              (prepipeline_queue.size() +
               context_configuration.meta_task_vector_width - 1) /
              context_configuration.meta_task_vector_width;
          if (outstanding_prepipeline_tasks < max_tasks)
          {
            issue_task = true;
            outstanding_prepipeline_tasks++;
          }
        }
      }
      if (issue_task)
      {
        add_base_resource_ref(META_TASK_REF);
        PrepipelineArgs args(this);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY);
      }
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      std::vector<std::pair<Operation*, GenerationID>> to_perform;
      to_perform.reserve(context_configuration.meta_task_vector_width);
      {
        AutoLock p_lock(prepipeline_lock);
        for (unsigned idx = 0;
             idx < context_configuration.meta_task_vector_width; idx++)
        {
          if (prepipeline_queue.empty())
            break;
          to_perform.emplace_back(prepipeline_queue.front());
          prepipeline_queue.pop_front();
        }
      }
      // No need for a sort here, these operations are already in program order
      // Perform our prepipeline tasks
      for (const std::pair<Operation*, GenerationID>& it : to_perform)
        it.first->execute_prepipeline_stage(it.second, false /*need wait*/);
      AutoLock p_lock(prepipeline_lock);
      legion_assert(outstanding_prepipeline_tasks > 0);
      const unsigned max_tasks =
          (prepipeline_queue.size() +
           context_configuration.meta_task_vector_width - 1) /
          context_configuration.meta_task_vector_width;
      if (max_tasks < outstanding_prepipeline_tasks)
      {
        outstanding_prepipeline_tasks--;
        return true;
      }
      else
      {
        PrepipelineArgs args(this);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY);
        // Reference keeps flowing with the continuation
        return false;
      }
    }

    //--------------------------------------------------------------------------
    bool InnerContext::add_to_dependence_queue(
        Operation* op, const std::vector<StaticDependence>* dependences,
        bool unordered, bool outermost)
    //--------------------------------------------------------------------------
    {
      GenerationID pointwise_generation = 0;
      std::map<DomainPoint, RtUserEvent> pending_pointwise;
      LgPriority priority = LG_THROUGHPUT_WORK_PRIORITY;
      // If this is ordered, we need to record this in the reorder buffer
      // and determine if we need to perform a window wait or not
      if (!unordered)
      {
        // Update any previous trace with state of the new operation
        if (previous_trace != nullptr)
        {
          bool execution_fence = false;
          if (op->invalidates_physical_trace_template(execution_fence))
          {
            if (!execution_fence)
            {
              // Issue a trace invalidation operation to free up
              // resources associated with the template
              FenceOp* complete =
                  initialize_trace_completion(op->get_provenance());
              // The previous trace is no longer valid
              previous_trace = nullptr;
              // We can safely recurse here since cleared the previous
              InnerContext::add_to_dependence_queue(
                  complete, nullptr /*deps*/, false /*unordered*/,
                  false /*outermost*/);
            }
            else
              previous_trace->record_intermediate_fence();
          }
        }
        // Set the trace for the operation
        if (current_trace != nullptr)
          current_trace->initialize_operation(op, dependences);
        // Enqueue this in the reorder buffer and then see if we need to
        // perform a window wait because there are too many outstanding ops
        AutoLock child_lock(child_op_lock);
        // Get the context index for this new operation
        const size_t context_index = total_children_count++;
        op->set_context_index(context_index, get_current_exception_handler());
        legion_assert(
            reorder_buffer.empty() ||
            ((reorder_buffer.back().operation_index + 1) == context_index));
        reorder_buffer.emplace_back(ReorderBufferEntry(op, context_index));
        if (!pending_pointwise_dependences.empty())
        {
          std::map<uint64_t, std::map<DomainPoint, RtUserEvent>>::iterator
              finder = pending_pointwise_dependences.find(context_index);
          if (finder != pending_pointwise_dependences.end())
          {
            pending_pointwise.swap(finder->second);
            pending_pointwise_dependences.erase(finder);
            pointwise_generation = op->get_generation();
          }
        }
        // Check to see if we need to perform a window wait
        // Only need to check if we are not tracing by frames
        // and not inside of a trace that might be replayed
        if ((context_configuration.min_frames_to_schedule == 0) &&
            (context_configuration.max_window_size > 0) &&
            (reorder_buffer.size() > context_configuration.max_window_size) &&
            ((current_trace == nullptr) || !current_trace->is_fixed()))
        {
          legion_assert(!window_wait.exists());
          window_wait = Runtime::create_rt_user_event();
          const RtEvent wait_event = window_wait;
          child_lock.release();
          if (!wait_event.has_triggered())
            wait_event.wait();
          child_lock.reacquire();
        }
        // Bump our priority if the context is not active as it means
        // that the runtime is currently not ahead of execution
        if (!currently_active_context)
          priority = LG_THROUGHPUT_DEFERRED_PRIORITY;
      }
      // Launch the task to perform the prepipeline stage for the operation
      if (op->has_prepipeline_stage())
        add_to_prepipeline_queue(op);
      // Trigger any pending pointwise dependences
      if (!pending_pointwise.empty())
      {
        for (const std::pair<const DomainPoint, RtUserEvent>& it :
             pending_pointwise)
          op->find_pointwise_dependence(
              it.first, pointwise_generation, it.second);
      }
      RtEvent precondition;
      RtEvent commit_event;
      bool issue_task = false;
      // We disable program order execution when we are replaying a
      // physical trace since it might not be sound to block
      if (runtime->program_order_execution && !unordered && outermost &&
          !is_replaying_physical_trace())
        commit_event = op->get_commit_event();
      {
        AutoLock d_lock(dependence_lock);
        if (unordered)
        {
          if (finished_execution)
            return false;
          // If this is unordered, stick it on the list of
          // unordered ops to be added later and then we're done
          unordered_ops.emplace_back(op);
          return true;
        }
        if (dependence_queue.empty())
        {
          issue_task = true;
          precondition = dependence_precondition;
        }
        dependence_queue.emplace_back(op);
        // Insert any unordered operations now as long as we aren't
        // doing program order execution, if we're doing program order
        // execution then we'll do that after running this operation
        if (!commit_event.exists())
          insert_unordered_ops(d_lock);
      }
      if (issue_task)
      {
        DependenceArgs args(this);
        runtime->issue_runtime_meta_task(args, priority, precondition);
      }
      if (commit_event.exists())
      {
        commit_event.wait();
        // Now do our insertion of any unordered operations
        AutoLock d_lock(dependence_lock);
        insert_unordered_ops(d_lock);
      }
      return true;
    }

    //--------------------------------------------------------------------------
    FenceOp* InnerContext::initialize_trace_completion(Provenance* prov)
    //--------------------------------------------------------------------------
    {
      legion_assert(previous_trace != nullptr);
      TraceCompleteOp* op = runtime->get_operation<TraceCompleteOp>();
      op->initialize_complete(
          this, previous_trace, prov,
          (traces.find(previous_trace->tid) == traces.end()));
      return op;
    }

    //--------------------------------------------------------------------------
    void InnerContext::process_dependence_stage(void)
    //--------------------------------------------------------------------------
    {
      std::vector<Operation*> to_perform;
      to_perform.reserve(context_configuration.meta_task_vector_width);
      Operation* launch_next_op = nullptr;
      {
        AutoLock d_lock(dependence_lock);
        for (unsigned idx = 0;
             idx < context_configuration.meta_task_vector_width; idx++)
        {
          if (dependence_queue.empty())
            break;
          to_perform.emplace_back(dependence_queue.front());
          dependence_queue.pop_front();
        }
        if (dependence_queue.empty())
          // Guard ourselves against tasks running after us
          dependence_precondition =
              RtEvent(Processor::get_current_finish_event());
        else
          launch_next_op = dependence_queue.front();
      }
      // No need for a sort these operations are already in program order
      // Perform our operations
      for (Operation* const & it : to_perform)
      {
        implicit_operation = it;
        it->execute_dependence_analysis();
      }
      implicit_operation = nullptr;
      // Then launch the next task if needed
      if (launch_next_op != nullptr)
      {
        DependenceArgs args(this);
        // Sample currently_active without the lock to try to get our priority
        runtime->issue_runtime_meta_task(
            args, !currently_active_context ? LG_THROUGHPUT_DEFERRED_PRIORITY :
                                              LG_THROUGHPUT_WORK_PRIORITY);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, typename ARGS, bool HAS_BOUND>
    void InnerContext::add_to_queue(
        QueueEntry<T> entry, LocalLock& lock, std::list<QueueEntry<T>>& queue,
        CompletionQueue& comp_queue)
    //--------------------------------------------------------------------------
    {
      legion_assert(entry.ready.exists());
      bool issue_task = false;
      RtEvent precondition;
      long long performed = 0;
      {
        AutoLock l_lock(lock);
        // Issue a task if there isn't one running right now
        if (queue.empty())
        {
          issue_task = true;
          // Make the queue the first time if necessary
          if (!comp_queue.exists())
            // We can put an upper bound on the number of operations as long
            // as we aren't using frames to runahead, if we're using frames
            // to run ahead then we can't know the maximum run ahead size
            comp_queue = CompletionQueue::create_completion_queue(
                (HAS_BOUND &&
                 (context_configuration.min_frames_to_schedule == 0)) ?
                    context_configuration.max_window_size :
                    0);
        }
        queue.emplace_back(entry);
        comp_queue.add_event(entry.ready);
        if (issue_task)
        {
          if (implicit_profiler != nullptr)
            performed = Realm::Clock::current_time_in_nanoseconds();
          precondition = RtEvent(comp_queue.get_nonempty_event());
        }
      }
      if (issue_task)
      {
        // Add a reference to the context the first time we defer this
        add_base_resource_ref(META_TASK_REF);
        ARGS args(this, precondition, performed);
        runtime->issue_runtime_meta_task(
            args, LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_ready_queue(Operation* op)
    //--------------------------------------------------------------------------
    {
      bool issue_task = false;
      {
        AutoLock r_lock(ready_lock);
        issue_task = ready_queue.empty();
        ready_queue.emplace_back(op);
      }
      if (issue_task)
      {
        add_base_resource_ref(META_TASK_REF);
        TriggerReadyArgs args(this);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    T InnerContext::process_queue(
        LocalLock& lock, RtEvent& precondition, std::list<QueueEntry<T>>& queue,
        CompletionQueue& comp_queue, std::vector<T>& to_perform,
        LgEvent previous_fevent, long long& performed) const
    //--------------------------------------------------------------------------
    {
      T next{};
      const size_t vector_width = context_configuration.meta_task_vector_width;
      to_perform.reserve(vector_width);
      AutoLock l_lock(lock);
      std::vector<RtEvent> ready_events(vector_width);
      const size_t num_ready =
          comp_queue.pop_events(&ready_events.front(), vector_width);
      legion_assert(num_ready > 0);
      ready_events.resize(num_ready);
      std::sort(ready_events.begin(), ready_events.end());
      if (precondition.exists() && (implicit_profiler != nullptr))
        implicit_profiler->record_completion_queue_event(
            precondition, previous_fevent, performed, &ready_events.front(),
            num_ready);
      // Find the entries
      for (typename std::list<QueueEntry<T>>::iterator it = queue.begin();
           it != queue.end();
           /*nothing*/)
      {
        std::vector<RtEvent>::iterator finder = std::lower_bound(
            ready_events.begin(), ready_events.end(), it->ready);
        if ((finder != ready_events.end()) && (*finder == it->ready))
        {
          to_perform.emplace_back(it->op);
          it = queue.erase(it);
          ready_events.erase(finder);
          if (ready_events.empty())
            break;
        }
        else
          it++;
      }
      legion_assert(ready_events.empty());
      if (!queue.empty())
      {
        if (implicit_profiler != nullptr)
          performed = Realm::Clock::current_time_in_nanoseconds();
        precondition = RtEvent(comp_queue.get_nonempty_event());
        next = queue.front().op;
      }
      return next;
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_ready_queue(void)
    //--------------------------------------------------------------------------
    {
      Operation* next = nullptr;
      std::vector<Operation*> to_perform;
      {
        AutoLock r_lock(ready_lock);
        legion_assert(!ready_queue.empty());
        for (unsigned idx = 0;
             idx < context_configuration.meta_task_vector_width; idx++)
        {
          to_perform.emplace_back(ready_queue.front());
          if (spy_logging_level > LIGHT_SPY_LOGGING)
          {
            previous_completion_events.insert(
                to_perform.back()->get_completion_event());
            // Periodically merge these to keep this data structure from
            // exploding when we have a long-running task, although don't do
            // this for fence operations in case we have to prune ourselves out
            // of the set
            if (previous_completion_events.size() >=
                LEGION_DEFAULT_MAX_TASK_WINDOW)
            {
              // Only merge ones that we know are completed
              std::vector<ApEvent> triggered;
              for (std::set<ApEvent>::const_iterator pit =
                       previous_completion_events.begin();
                   pit != previous_completion_events.end();
                   /*nothing*/)
              {
                if (pit->has_triggered_faultignorant())
                {
                  triggered.emplace_back(*pit);
                  std::set<ApEvent>::const_iterator delete_it = pit++;
                  previous_completion_events.erase(delete_it);
                }
                else
                  pit++;
              }
              if (!triggered.empty())
                previous_completion_events.insert(
                    Runtime::merge_events(nullptr, triggered));
            }
          }
          ready_queue.pop_front();
          if (ready_queue.empty())
            break;
        }
        if (!ready_queue.empty())
          next = ready_queue.front();
      }
      // Sort these into program order to guarantee forward progress
      std::sort(
          to_perform.begin(), to_perform.end(),
          [](Operation* lhs, Operation* rhs) {
            return lhs->get_context_index() < rhs->get_context_index();
          });
      for (Operation* const & it : to_perform)
      {
        implicit_enclosing_context = did;
        implicit_operation = it;
        Provenance* provenance = it->get_provenance();
        implicit_provenance = (provenance == nullptr) ? 0 : provenance->pid;
        implicit_unique_op_id = it->get_unique_op_id();
        it->set_execution_fence_event(current_execution_fence_event);
        it->trigger_ready();
      }
      implicit_operation = nullptr;
      if (next != nullptr)
      {
        TriggerReadyArgs args(this);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY);
        return false;
      }
      else
        return true;
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_task_queue(SingleTask* task, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<SingleTask*, DeferredEnqueueTaskArgs, false /*has bounds*/>(
          QueueEntry<SingleTask*>(task, ready), enqueue_task_lock,
          enqueue_task_queue, enqueue_task_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_enqueue_task_queue(
        RtEvent precondition, LgEvent previous_fevent, long long performed)
    //--------------------------------------------------------------------------
    {
      std::vector<SingleTask*> to_perform;
      SingleTask* next = process_queue<SingleTask*>(
          enqueue_task_lock, precondition, enqueue_task_queue,
          enqueue_task_comp_queue, to_perform, previous_fevent, performed);
      // No need to sort here, we're just adding these to the ready queue
      for (SingleTask* const & it : to_perform)
      {
        implicit_enclosing_context = did;
        implicit_operation = it;
        Provenance* provenance = it->get_provenance();
        implicit_provenance = (provenance == nullptr) ? 0 : provenance->pid;
        implicit_unique_op_id = it->get_unique_op_id();
        it->enqueue_ready_task(false /*use target*/);
      }
      implicit_operation = nullptr;
      if (next != nullptr)
      {
        DeferredEnqueueTaskArgs args(this, precondition, performed);
        runtime->issue_runtime_meta_task(
            args, LG_THROUGHPUT_WORK_PRIORITY, precondition);
        return false;
      }
      else
        return true;
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_trigger_execution_queue(
        Operation* op, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<Operation*, TriggerExecutionArgs, false /*has bounds*/>(
          QueueEntry<Operation*>(op, ready), trigger_execution_lock,
          trigger_execution_queue, trigger_execution_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_trigger_execution_queue(
        RtEvent precondition, LgEvent previous_fevent, long long performed)
    //--------------------------------------------------------------------------
    {
      std::vector<Operation*> to_perform;
      Operation* next = process_queue<Operation*>(
          trigger_execution_lock, precondition, trigger_execution_queue,
          trigger_execution_comp_queue, to_perform, previous_fevent, performed);
      // Sort these in program order to guarantee forward progress
      std::sort(
          to_perform.begin(), to_perform.end(),
          [](Operation* lhs, Operation* rhs) {
            return lhs->get_context_index(), rhs->get_context_index();
          });
      for (Operation* const & it : to_perform)
      {
        implicit_enclosing_context = did;
        implicit_operation = it;
        Provenance* provenance = it->get_provenance();
        implicit_provenance = (provenance == nullptr) ? 0 : provenance->pid;
        implicit_unique_op_id = it->get_unique_op_id();
        it->trigger_execution();
      }
      implicit_operation = nullptr;
      if (next != nullptr)
      {
        TriggerExecutionArgs args(this, precondition, performed);
        runtime->issue_runtime_meta_task(
            args, LG_THROUGHPUT_WORK_PRIORITY, precondition);
        return false;
      }
      else
        return true;
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_deferred_execution_queue(
        Operation* op, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<Operation*, DeferredExecutionArgs, false /*has bounds*/>(
          QueueEntry<Operation*>(op, ready), deferred_execution_lock,
          deferred_execution_queue, deferred_execution_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_deferred_execution_queue(
        RtEvent precondition, LgEvent previous_fevent, long long performed)
    //--------------------------------------------------------------------------
    {
      std::vector<Operation*> to_perform;
      Operation* next = process_queue<Operation*>(
          deferred_execution_lock, precondition, deferred_execution_queue,
          deferred_execution_comp_queue, to_perform, previous_fevent,
          performed);
      // Sort these into program order to guarantee forward progress
      std::sort(
          to_perform.begin(), to_perform.end(),
          [](Operation* lhs, Operation* rhs) {
            return lhs->get_context_index(), rhs->get_context_index();
          });
      for (Operation* const & it : to_perform)
      {
        implicit_enclosing_context = did;
        implicit_operation = it;
        Provenance* provenance = it->get_provenance();
        implicit_provenance = (provenance == nullptr) ? 0 : provenance->pid;
        implicit_unique_op_id = it->get_unique_op_id();
        it->complete_execution();
      }
      implicit_operation = nullptr;
      if (next != nullptr)
      {
        DeferredExecutionArgs args(this, precondition, performed);
        runtime->issue_runtime_meta_task(
            args, LG_THROUGHPUT_WORK_PRIORITY, precondition);
        return false;
      }
      else
        return true;
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_deferred_mapped_queue(
        Operation* op, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<Operation*, DeferredMappedArgs, false /*has bounds*/>(
          QueueEntry<Operation*>(op, ready), deferred_mapped_lock,
          deferred_mapped_queue, deferred_mapped_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_deferred_mapped_queue(
        RtEvent precondition, LgEvent previous_fevent, long long performed)
    //--------------------------------------------------------------------------
    {
      std::vector<Operation*> to_perform;
      Operation* next = process_queue<Operation*>(
          deferred_mapped_lock, precondition, deferred_mapped_queue,
          deferred_mapped_comp_queue, to_perform, previous_fevent, performed);
      // Sort these into program order to guarantee forward progress
      std::sort(
          to_perform.begin(), to_perform.end(),
          [](Operation* lhs, Operation* rhs) {
            return lhs->get_context_index() < rhs->get_context_index();
          });
      for (Operation* const & it : to_perform)
      {
        implicit_enclosing_context = did;
        implicit_operation = it;
        Provenance* provenance = it->get_provenance();
        implicit_provenance = (provenance == nullptr) ? 0 : provenance->pid;
        implicit_unique_op_id = it->get_unique_op_id();
        it->complete_mapping();
      }
      implicit_operation = nullptr;
      if (next != nullptr)
      {
        DeferredMappedArgs args(this, precondition, performed);
        runtime->issue_runtime_meta_task(
            args, LG_THROUGHPUT_WORK_PRIORITY, precondition);
        return false;
      }
      else
        return true;
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_deferred_completion_queue(
        Operation* op, ApEvent effects, bool tracked)
    //--------------------------------------------------------------------------
    {
      legion_assert(effects.exists());
      bool issue_task = false;
      long long performed = 0;
      RtEvent precondition;
      {
        AutoLock child_lock(child_op_lock);
        if (tracked)
        {
          ReorderBufferEntry& entry = find_rob_entry(op);
          legion_assert(!entry.complete);
          entry.complete_event = effects;
          entry.complete = true;
        }
        // Issue a task if there isn't one running right now
        if (deferred_completion_queue.empty())
        {
          issue_task = true;
          // Make the queue the first time if necessary
          if (!deferred_completion_comp_queue.exists())
            // No idea how many outstanding point tasks can be coming
            // through here so we can't put a bound on it
            deferred_completion_comp_queue =
                CompletionQueue::create_completion_queue(0);
        }
        deferred_completion_queue.emplace_back(CompletionEntry(op, effects));
        deferred_completion_comp_queue.add_event_faultaware(effects);
        if (issue_task)
        {
          if (implicit_profiler != nullptr)
            performed = Realm::Clock::current_time_in_nanoseconds();
          precondition =
              RtEvent(deferred_completion_comp_queue.get_nonempty_event());
        }
      }
      if (issue_task)
      {
        // Add a reference to the context the first time we defer this
        add_base_resource_ref(META_TASK_REF);
        DeferredCompletionArgs args(this, precondition, performed);
        runtime->issue_runtime_meta_task(
            args, LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_deferred_completion_queue(
        RtEvent precondition, LgEvent previous_fevent, long long performed)
    //--------------------------------------------------------------------------
    {
      Operation* next = nullptr;
      std::vector<CompletionEntry> to_perform;
      {
        const size_t vector_width =
            context_configuration.meta_task_vector_width;
        to_perform.reserve(vector_width);
        AutoLock child_lock(child_op_lock);
        std::vector<ApEvent> ready_events(vector_width);
        const size_t num_ready = deferred_completion_comp_queue.pop_events(
            &ready_events.front(), vector_width);
        legion_assert(num_ready > 0);
        ready_events.resize(num_ready);
        std::sort(ready_events.begin(), ready_events.end());
        if (precondition.exists() && (implicit_profiler != nullptr))
          implicit_profiler->record_completion_queue_event(
              precondition, previous_fevent, performed, &ready_events.front(),
              num_ready);
        // Find the entries
        for (std::list<CompletionEntry>::iterator it =
                 deferred_completion_queue.begin();
             it != deferred_completion_queue.end();
             /*nothing*/)
        {
          std::vector<ApEvent>::iterator finder = std::lower_bound(
              ready_events.begin(), ready_events.end(), it->effects);
          if ((finder != ready_events.end()) && (*finder == it->effects))
          {
            to_perform.emplace_back(*it);
            it = deferred_completion_queue.erase(it);
            ready_events.erase(finder);
            if (ready_events.empty())
              break;
          }
          else
            it++;
        }
        legion_assert(ready_events.empty());
        if (!deferred_completion_queue.empty())
        {
          if (implicit_profiler != nullptr)
            performed = Realm::Clock::current_time_in_nanoseconds();
          precondition =
              RtEvent(deferred_completion_comp_queue.get_nonempty_event());
          next = deferred_completion_queue.front().op;
        }
      }
      // Sort these based on program order to guarantee forward progress
      std::sort(
          to_perform.begin(), to_perform.end(),
          [](const CompletionEntry& lhs, const CompletionEntry& rhs) {
            return lhs.op->get_context_index() < rhs.op->get_context_index();
          });
      for (const CompletionEntry& it : to_perform)
      {
        bool poisoned = false;
        // TODO: do something with poisoned completion events and resilience
        if (!it.effects.has_triggered_faultaware(poisoned) || poisoned)
          std::abort();
        implicit_enclosing_context = did;
        implicit_operation = it.op;
        Provenance* provenance = it.op->get_provenance();
        implicit_provenance = (provenance == nullptr) ? 0 : provenance->pid;
        implicit_unique_op_id = it.op->get_unique_op_id();
        it.op->complete_operation(it.effects, false /*first*/);
      }
      implicit_operation = nullptr;
      if (next != nullptr)
      {
        DeferredCompletionArgs args(this, precondition, performed);
        runtime->issue_runtime_meta_task(
            args, LG_THROUGHPUT_WORK_PRIORITY, precondition);
        return false;
      }
      else
        return true;
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_trigger_commit_queue(Operation* op)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(trigger_commit_lock);
      legion_no_skip_assert(
          commit_priority_queue
              .emplace(std::make_pair(op->get_context_index(), op))
              .second);
      if (!outstanding_commit_task)
      {
        TriggerCommitArgs args(this);
        add_base_resource_ref(META_TASK_REF);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY);
        outstanding_commit_task = true;
      }
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_trigger_commit_queue(void)
    //--------------------------------------------------------------------------
    {
      bool trigger_next = false;
      std::vector<Operation*> to_perform;
      {
        AutoLock c_lock(trigger_commit_lock);
        legion_assert(outstanding_commit_task);
        to_perform.reserve(std::min<size_t>(
            commit_priority_queue.size(),
            context_configuration.meta_task_vector_width));
        while (
            !commit_priority_queue.empty() &&
            (to_perform.size() < context_configuration.meta_task_vector_width))
        {
          std::map<uint64_t, Operation*>::iterator it =
              commit_priority_queue.begin();
          to_perform.push_back(it->second);
          commit_priority_queue.erase(it);
        }
        if (!commit_priority_queue.empty())
          trigger_next = true;
        else
          outstanding_commit_task = false;
      }
      // Sort these based on program order to guarantee forward progress
      std::sort(
          to_perform.begin(), to_perform.end(),
          [](Operation* lhs, Operation* rhs) {
            return lhs->get_context_index() < rhs->get_context_index();
          });
      for (std::vector<Operation*>::const_iterator it = to_perform.begin();
           it != to_perform.end(); it++)
      {
        implicit_enclosing_context = did;
        implicit_operation = (*it);
        Provenance* provenance = (*it)->get_provenance();
        implicit_provenance = (provenance == nullptr) ? 0 : provenance->pid;
        implicit_unique_op_id = (*it)->get_unique_op_id();
        (*it)->trigger_commit();
      }
      implicit_operation = nullptr;
      if (trigger_next)
      {
        TriggerCommitArgs args(this);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY);
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_deferred_commit_queue(
        Operation* op, RtEvent ready, bool deactivate)
    //--------------------------------------------------------------------------
    {
      add_to_queue<
          std::pair<Operation*, bool>, DeferredCommitArgs,
          false /*has bounds*/>(
          QueueEntry<std::pair<Operation*, bool>>(
              std::pair<Operation*, bool>(op, deactivate), ready),
          deferred_commit_lock, deferred_commit_queue,
          deferred_commit_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_deferred_commit_queue(
        RtEvent precondition, LgEvent previous_fevent, long long performed)
    //--------------------------------------------------------------------------
    {
      std::vector<std::pair<Operation*, bool>> to_perform;
      std::pair<Operation*, bool> next =
          process_queue<std::pair<Operation*, bool>>(
              deferred_commit_lock, precondition, deferred_commit_queue,
              deferred_commit_comp_queue, to_perform, previous_fevent,
              performed);
      // Sort these based on program order to guarantee forward progress
      std::sort(
          to_perform.begin(), to_perform.end(),
          [](const std::pair<Operation*, bool>& lhs,
             const std::pair<Operation*, bool>& rhs) {
            return lhs.first->get_context_index() <
                   rhs.first->get_context_index();
          });
      for (const std::pair<Operation*, bool>& it : to_perform)
      {
        implicit_enclosing_context = did;
        implicit_operation = it.first;
        Provenance* provenance = it.first->get_provenance();
        implicit_provenance = (provenance == nullptr) ? 0 : provenance->pid;
        implicit_unique_op_id = it.first->get_unique_op_id();
        it.first->commit_operation(it.second);
      }
      implicit_operation = nullptr;
      if (next.first != nullptr)
      {
        DeferredCommitArgs args(this, precondition, performed);
        runtime->issue_runtime_meta_task(
            args, LG_THROUGHPUT_WORK_PRIORITY, precondition);
        return false;
      }
      else
        return true;
    }

    //--------------------------------------------------------------------------
    void InnerContext::insert_unordered_ops(AutoLock& d_lock)
    //--------------------------------------------------------------------------
    {
      // No progressing of unordered operations inside a trace
      if (current_trace != nullptr)
        return;
      // If we don't have any unordered operations then we are done
      if (unordered_ops.empty())
        return;
      issue_unordered_operations(d_lock, unordered_ops);
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_unordered_operations(
        AutoLock& d_lock, std::vector<Operation*>& ready_operations)
    //--------------------------------------------------------------------------
    {
      if ((previous_trace != nullptr) && !ready_operations.empty())
      {
        // Make an invalidation operation and add it to the list of
        // operations to add to the queue
        FenceOp* complete = initialize_trace_completion(
            ready_operations.front()->get_provenance());
        // This actually needs to be pushed onto the front of the queue to
        // make sure that it is done before any of the unordered operations
        ready_operations.insert(ready_operations.begin(), complete);
        previous_trace = nullptr;
      }
      std::map<Operation*, GenerationID> pending_generations;
      std::map<Operation*, std::map<DomainPoint, RtUserEvent>>
          pending_pointwise;
      if (runtime->program_order_execution)
      {
        while (!ready_operations.empty())
        {
          Operation* op = ready_operations.back();
          ready_operations.pop_back();
          // Record it in the reorder buffer
          {
            AutoLock child_lock(child_op_lock);
            const size_t context_index = total_children_count++;
            op->set_context_index(
                context_index, get_current_exception_handler());
            legion_assert(
                reorder_buffer.empty() ||
                ((reorder_buffer.back().operation_index + 1) == context_index));
            reorder_buffer.emplace_back(ReorderBufferEntry(op, context_index));
            if (!pending_pointwise_dependences.empty())
            {
              std::map<uint64_t, std::map<DomainPoint, RtUserEvent>>::iterator
                  finder = pending_pointwise_dependences.find(context_index);
              if (finder != pending_pointwise_dependences.end())
              {
                pending_pointwise[op].swap(finder->second);
                pending_pointwise_dependences.erase(finder);
                pending_generations[op] = op->get_generation();
              }
            }
          }
          legion_assert(dependence_queue.empty());
          dependence_queue.emplace_back(op);
          RtEvent precondition = dependence_precondition;
          // Release the lock and launch the meta-task
          d_lock.release();
          const RtEvent commit_event = op->get_commit_event();
          DependenceArgs args(this);
          const LgPriority priority = LG_THROUGHPUT_WORK_PRIORITY;
          runtime->issue_runtime_meta_task(args, priority, precondition);
          commit_event.wait();
          // Reacquire the lock before doing the next operation
          d_lock.reacquire();
        }
      }
      else
      {
        // Common path for normal execution where we don't need to
        // execute each of these things in program order
        // We need the child op lock here so we can add these to this
        // list of executing children as well
        AutoLock child_lock(child_op_lock);
        for (Operation* op : ready_operations)
        {
          const size_t context_index = total_children_count++;
          op->set_context_index(context_index, get_current_exception_handler());
          legion_assert(
              reorder_buffer.empty() ||
              ((reorder_buffer.back().operation_index + 1) == context_index));
          reorder_buffer.emplace_back(ReorderBufferEntry(op, context_index));
          if (!pending_pointwise_dependences.empty())
          {
            std::map<uint64_t, std::map<DomainPoint, RtUserEvent>>::iterator
                finder = pending_pointwise_dependences.find(context_index);
            if (finder != pending_pointwise_dependences.end())
            {
              pending_pointwise[op].swap(finder->second);
              pending_pointwise_dependences.erase(finder);
              pending_generations[op] = op->get_generation();
            }
          }
          if (dependence_queue.empty())
          {
            DependenceArgs args(this);
            const LgPriority priority = LG_THROUGHPUT_WORK_PRIORITY;
            runtime->issue_runtime_meta_task(
                args, priority, dependence_precondition);
          }
          dependence_queue.emplace_back(op);
        }
        ready_operations.clear();
      }
      if (!pending_pointwise.empty())
      {
        d_lock.release();
        for (const std::pair<
                 Operation* const, std::map<DomainPoint, RtUserEvent>>& pit :
             pending_pointwise)
        {
          std::map<Operation*, GenerationID>::const_iterator finder =
              pending_generations.find(pit.first);
          legion_assert(finder != pending_generations.end());
          for (const std::pair<const DomainPoint, RtUserEvent>& it : pit.second)
            pit.first->find_pointwise_dependence(
                it.first, finder->second, it.second);
        }
        d_lock.reacquire();
      }
    }

    //--------------------------------------------------------------------------
    unsigned InnerContext::minimize_repeat_results(
        unsigned ready, bool& double_wait_interval)
    //--------------------------------------------------------------------------
    {
      double_wait_interval = false;
      return ready;
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_executing_child(Operation* op)
    //--------------------------------------------------------------------------
    {
      AutoLock child_lock(child_op_lock);
      const size_t context_index = total_children_count++;
      op->set_context_index(context_index, get_current_exception_handler());
      legion_assert(
          reorder_buffer.empty() ||
          ((reorder_buffer.back().operation_index + 1) == context_index));
      reorder_buffer.emplace_back(ReorderBufferEntry(op, context_index));
      if (!pending_pointwise_dependences.empty())
      {
        std::map<uint64_t, std::map<DomainPoint, RtUserEvent>>::iterator
            finder = pending_pointwise_dependences.find(context_index);
        if (finder != pending_pointwise_dependences.end())
        {
          std::map<DomainPoint, RtUserEvent> to_trigger;
          to_trigger.swap(finder->second);
          pending_pointwise_dependences.erase(finder);
          const GenerationID generation = op->get_generation();
          child_lock.release();
          for (const std::pair<const DomainPoint, RtUserEvent>& it : to_trigger)
            op->find_pointwise_dependence(it.first, generation, it.second);
        }
      }
    }

    //--------------------------------------------------------------------------
    InnerContext::ReorderBufferEntry& InnerContext::find_rob_entry(
        Operation* op)
    //--------------------------------------------------------------------------
    {
      legion_assert(!reorder_buffer.empty());
      ReorderBufferEntry& head = reorder_buffer.front();
      legion_assert(head.operation_index <= op->get_context_index());
      uint64_t offset = op->get_context_index() - head.operation_index;
      legion_assert(offset < reorder_buffer.size());
      ReorderBufferEntry& entry = reorder_buffer[offset];
      legion_assert(entry.operation == op);
      return entry;
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_child_complete(Operation* op)
    //--------------------------------------------------------------------------
    {
      AutoLock child_lock(child_op_lock);
      ReorderBufferEntry& entry = find_rob_entry(op);
      entry.complete = true;
      if (!task_executed && (spy_logging_level > LIGHT_SPY_LOGGING))
      {
        if (entry.complete_event.exists())
          cummulative_child_completion_events.emplace_back(
              entry.complete_event);
        // Make sure this vector doesn't grow too large for long-running tasks
        constexpr size_t MAX_SIZE = 32;
        if (cummulative_child_completion_events.size() == MAX_SIZE)
        {
          const ApEvent merged = Runtime::merge_events(
              nullptr, cummulative_child_completion_events);
          cummulative_child_completion_events.clear();
          cummulative_child_completion_events.emplace_back(merged);
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_child_commit(Operation* op)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      bool needs_trigger = false;
      {
        AutoLock child_lock(child_op_lock);
        legion_assert(!reorder_buffer.empty());
        // Mark that this operation is commited
        ReorderBufferEntry& entry = find_rob_entry(op);
        legion_assert(!entry.committed);
        entry.committed = true;
        // Pop as many committed entries off the front fo the ROB as possible
        while (!reorder_buffer.empty())
        {
          const ReorderBufferEntry& front = reorder_buffer.front();
          if (!front.committed)
            break;
          reorder_buffer.pop_front();
        }
        // Check to see if we need to wake up a window waiter
        // Add some hysteresis here so that we have some runway for when
        // the paused task resumes it can run for a little while.
        if (window_wait.exists() &&
            (context_configuration.max_window_size > 0) &&
            (reorder_buffer.size() <=
             ((100 - context_configuration.hysteresis_percentage) *
              context_configuration.max_window_size / 100)))
        {
          to_trigger = window_wait;
          window_wait = RtUserEvent::NO_RT_USER_EVENT;
        }
        // See if we need to trigger the all children commited call
        if (reorder_buffer.empty() && task_executed)
          needs_trigger = true;
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
      if (needs_trigger && (owner_task != nullptr))
        owner_task->trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    int InnerContext::has_conflicting_regions(
        MapOp* op, bool& parent_conflict, bool& inline_conflict)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement& req = op->get_requirement();
      return has_conflicting_internal(req, parent_conflict, inline_conflict);
    }

    //--------------------------------------------------------------------------
    int InnerContext::has_conflicting_regions(
        AttachOp* attach, bool& parent_conflict, bool& inline_conflict)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement& req = attach->get_requirement();
      return has_conflicting_internal(req, parent_conflict, inline_conflict);
    }

    //--------------------------------------------------------------------------
    int InnerContext::has_conflicting_internal(
        const RegionRequirement& req, bool& parent_conflict,
        bool& inline_conflict)
    //--------------------------------------------------------------------------
    {
      parent_conflict = false;
      inline_conflict = false;
      // No need to hold our lock here because we are the only ones who
      // could possibly be doing any mutating of the physical_regions data
      // structure but we are here so we aren't mutating
      for (unsigned our_idx = 0; our_idx < physical_regions.size(); our_idx++)
      {
        // skip any regions which are not mapped
        if (!physical_regions[our_idx].is_mapped())
          continue;
        const RegionRequirement& our_req =
            physical_regions[our_idx].impl->get_requirement();
        // This better be true for a single task
        legion_assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        if (check_region_dependence(
                our_tid, our_space, our_req, our_usage, req))
        {
          parent_conflict = true;
          return our_idx;
        }
      }
      // Need lock here because of unordered detach operations
      AutoLock i_lock(inline_lock, false /*exclusive*/);
      ctx::map<RegionTreeID, ctx::list<PhysicalRegion>>::const_iterator finder =
          inline_regions.find(req.parent.get_tree_id());
      if (finder == inline_regions.end())
        return -1;
      for (const PhysicalRegion& region : finder->second)
      {
        if (!region.is_mapped())
          continue;
        const RegionRequirement& our_req = region.impl->get_requirement();
        // This better be true for a single task
        legion_assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        if (check_region_dependence(
                our_tid, our_space, our_req, our_usage, req))
        {
          inline_conflict = true;
          // No index for inline conflicts
          return -1;
        }
      }
      return -1;
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_conflicting_regions(
        TaskOp* task, std::vector<PhysicalRegion>& conflicting)
    //--------------------------------------------------------------------------
    {
      for (const RegionRequirement& req : task->regions)
        find_conflicting_internal(req, conflicting);
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_conflicting_regions(
        CopyOp* copy, std::vector<PhysicalRegion>& conflicting)
    //--------------------------------------------------------------------------
    {
      for (const RegionRequirement& req : copy->src_requirements)
        find_conflicting_internal(req, conflicting);
      for (const RegionRequirement& req : copy->dst_requirements)
        find_conflicting_internal(req, conflicting);
      for (const RegionRequirement& req : copy->src_indirect_requirements)
        find_conflicting_internal(req, conflicting);
      for (const RegionRequirement& req : copy->dst_indirect_requirements)
        find_conflicting_internal(req, conflicting);
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_conflicting_regions(
        AcquireOp* acquire, std::vector<PhysicalRegion>& conflicting)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement& req = acquire->get_requirement();
      find_conflicting_internal(req, conflicting);
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_conflicting_regions(
        ReleaseOp* release, std::vector<PhysicalRegion>& conflicting)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement& req = release->get_requirement();
      find_conflicting_internal(req, conflicting);
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_conflicting_regions(
        DependentPartitionOp* partition,
        std::vector<PhysicalRegion>& conflicting)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement& req = partition->get_requirement();
      find_conflicting_internal(req, conflicting);
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_conflicting_internal(
        const RegionRequirement& req, std::vector<PhysicalRegion>& conflicting)
    //--------------------------------------------------------------------------
    {
      // No need to hold our lock here because we are the only ones who
      // could possibly be doing any mutating of the physical_regions data
      // structure but we are here so we aren't mutating
      for (unsigned our_idx = 0; our_idx < physical_regions.size(); our_idx++)
      {
        // skip any regions which are not mapped
        if (!physical_regions[our_idx].is_mapped())
          continue;
        const RegionRequirement& our_req =
            physical_regions[our_idx].impl->get_requirement();
        // This better be true for a single task
        legion_assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        if (check_region_dependence(
                our_tid, our_space, our_req, our_usage, req))
        {
          // Check for duplicates here
          std::vector<PhysicalRegion>::iterator it = std::lower_bound(
              conflicting.begin(), conflicting.end(),
              physical_regions[our_idx]);
          if ((it == conflicting.end()) || ((*it) != physical_regions[our_idx]))
            conflicting.insert(it, physical_regions[our_idx]);
        }
      }
      // Need lock here because of unordered detach operations
      AutoLock i_lock(inline_lock, false /*exclusive*/);
      ctx::map<RegionTreeID, ctx::list<PhysicalRegion>>::const_iterator finder =
          inline_regions.find(req.parent.get_tree_id());
      if (finder != inline_regions.end())
      {
        for (const PhysicalRegion& region : finder->second)
        {
          if (!region.is_mapped())
            continue;
          const RegionRequirement& our_req = region.impl->get_requirement();
          // This better be true for a single task
          legion_assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
          RegionTreeID our_tid = our_req.region.get_tree_id();
          IndexSpace our_space = our_req.region.get_index_space();
          RegionUsage our_usage(our_req);
          if (check_region_dependence(
                  our_tid, our_space, our_req, our_usage, req))
          {
            // Check for duplicates here
            std::vector<PhysicalRegion>::iterator it = std::lower_bound(
                conflicting.begin(), conflicting.end(), region);
            if ((it == conflicting.end()) || ((*it) != region))
              conflicting.insert(it, region);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_conflicting_regions(
        FillOp* fill, std::vector<PhysicalRegion>& conflicting)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement& req = fill->get_requirement();
      find_conflicting_internal(req, conflicting);
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_conflicting_regions(
        DiscardOp* discard, std::vector<PhysicalRegion>& conflicting)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement& req = discard->get_requirement();
      find_conflicting_internal(req, conflicting);
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_inline_mapped_region(
        const PhysicalRegion& region)
    //--------------------------------------------------------------------------
    {
      // Because of 'remap_region', this method can be called
      // both for inline regions as well as regions which were
      // initally mapped for the task.  Do a quick check to see
      // if it was an original region.  If it was then we're done.
      for (unsigned idx = 0; idx < physical_regions.size(); idx++)
      {
        if (physical_regions[idx].impl == region.impl)
          return;
      }
      // Need lock because of unordered detach operations
      const RegionTreeID tid =
          region.impl->get_requirement().region.get_tree_id();
      AutoLock i_lock(inline_lock);
      inline_regions[tid].emplace_back(region);
    }

    //--------------------------------------------------------------------------
    void InnerContext::unregister_inline_mapped_region(
        const PhysicalRegion& region)
    //--------------------------------------------------------------------------
    {
      const RegionTreeID tid =
          region.impl->get_requirement().region.get_tree_id();
      // Need lock because of unordered detach operations
      AutoLock i_lock(inline_lock);
      ctx::map<RegionTreeID, ctx::list<PhysicalRegion>>::iterator finder =
          inline_regions.find(tid);
      if (finder != inline_regions.end())
      {
        for (std::list<PhysicalRegion>::iterator it = finder->second.begin();
             it != finder->second.end(); it++)
        {
          if (it->impl == region.impl)
          {
            if (runtime->runtime_warnings && !has_inline_accessor)
              has_inline_accessor = region.impl->created_accessor();
            finder->second.erase(it);
            if (finder->second.empty())
              inline_regions.erase(finder);
            return;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::print_children(void)
    //--------------------------------------------------------------------------
    {
      // Don't both taking the lock since this is for debugging
      // and isn't actually called anywhere
      for (const ReorderBufferEntry& entry : reorder_buffer)
      {
        if (entry.complete)
          printf("Completed Child %p\n", entry.operation);
        else
          printf("Executing Child %p\n", entry.operation);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_implicit_dependences(Operation* op)
    //--------------------------------------------------------------------------
    {
      // If there are any outstanding unmapped dependent partition operations
      // outstanding then we might have an implicit dependence on its execution
      // so we always record a dependence on it
      if (last_implicit_creation != nullptr)
      {
        if (op->register_dependence(
                last_implicit_creation, last_implicit_creation_gen) &&
            // Can't prune when doing legion spy
            (spy_logging_level <= LIGHT_SPY_LOGGING))
          last_implicit_creation = nullptr;
      }
      if (current_mapping_fence != nullptr)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
        {
          if (current_fence_uid > 0)
          {
            unsigned num_regions = op->get_region_count();
            if (num_regions > 0)
            {
              for (unsigned idx = 0; idx < num_regions; idx++)
              {
                LegionSpy::log_mapping_dependence(
                    get_unique_id(), current_fence_uid, 0,
                    op->get_unique_op_id(), idx, TRUE_DEPENDENCE);
              }
            }
            else
              LegionSpy::log_mapping_dependence(
                  get_unique_id(), current_fence_uid, 0, op->get_unique_op_id(),
                  0, TRUE_DEPENDENCE);
          }
          // Have to record this operation in case there is a fence later
          ops_since_last_fence.emplace_back(op->get_unique_op_id());
          // Cannot prune because of Legion Spy
          op->register_dependence(
              current_mapping_fence, current_mapping_fence_gen);
        }
        else if (op->register_dependence(
                     current_mapping_fence, current_mapping_fence_gen))
          current_mapping_fence = nullptr;
      }
      else if (spy_logging_level > LIGHT_SPY_LOGGING)
        ops_since_last_fence.emplace_back(op->get_unique_op_id());
    }

    //--------------------------------------------------------------------------
    void InnerContext::perform_mapping_fence_analysis(Operation* op)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      {
        const OpKind op_kind = op->get_operation_kind();
        // It's alright if you hit this assertion for a new operation kind
        // Just add the new operation kind here and then update the check
        // in register_implicit_dependences that looks for all these kinds too
        // so that we do not run into trouble when running with Legion Spy.
        legion_assert(
            (op_kind == FENCE_OP_KIND) || (op_kind == FRAME_OP_KIND) ||
            (op_kind == TIMING_OP_KIND) || (op_kind == DELETION_OP_KIND) ||
            (op_kind == TRACE_BEGIN_OP_KIND) ||
            (op_kind == TRACE_RECURRENT_OP_KIND) ||
            (op_kind == TRACE_COMPLETE_OP_KIND));
      }
#endif
      std::vector<std::pair<Operation*, GenerationID>> previous_operations;
      // Take the lock and iterate through our current pending
      // operations and find all the ones with a context index
      // that is less than the index for the fence operation
      const uint64_t next_fence_index = op->get_context_index();
      {
        // Mapping analysis only
        AutoLock child_lock(child_op_lock, false /*exclusive*/);
        for (std::deque<ReorderBufferEntry>::const_reverse_iterator it =
                 reorder_buffer.crbegin();
             it != reorder_buffer.crend(); it++)
        {
          if (it->operation_index < current_mapping_fence_index)
            break;
          // If it came after this fence we skip it
          if (next_fence_index <= it->operation_index)
            continue;
          if (it->complete)
            continue;
          previous_operations.emplace_back(
              std::make_pair(it->operation, it->operation->get_generation()));
        }
      }
      // Now record the dependences
      if (!previous_operations.empty())
      {
        for (const std::pair<Operation*, GenerationID>& it :
             previous_operations)
          op->register_dependence(it.first, it.second);
      }
      if (spy_logging_level > LIGHT_SPY_LOGGING)
      {
        // Record a dependence on the previous fence
        if (current_fence_uid > 0)
          LegionSpy::log_mapping_dependence(
              get_unique_id(), current_fence_uid, 0 /*index*/,
              op->get_unique_op_id(), 0 /*index*/, TRUE_DEPENDENCE);
        for (UniqueID uid : ops_since_last_fence)
        {
          // Skip ourselves if we are here
          if (uid == op->get_unique_op_id())
            continue;
          LegionSpy::log_mapping_dependence(
              get_unique_id(), uid, 0 /*index*/, op->get_unique_op_id(),
              0 /*index*/, TRUE_DEPENDENCE);
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::perform_execution_fence_analysis(
        Operation* op, std::set<ApEvent>& previous_events)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      {
        const OpKind op_kind = op->get_operation_kind();
        // It's alright if you hit this assertion for a new operation kind
        // Just add the new operation kind here and then update the check
        // in register_implicit_dependences that looks for all these kinds too
        // so that we do not run into trouble when running with Legion Spy.
        legion_assert(
            (op_kind == FENCE_OP_KIND) || (op_kind == FRAME_OP_KIND) ||
            (op_kind == TIMING_OP_KIND) || (op_kind == TRACE_BEGIN_OP_KIND) ||
            (op_kind == TRACE_RECURRENT_OP_KIND) ||
            (op_kind == TRACE_COMPLETE_OP_KIND));
      }
#endif
      // Take the lock and iterate through our current pending
      // operations and find all the ones with a context index
      // that is less than the index for the fence operation
      const uint64_t next_fence_index = op->get_context_index();
      {
        // Execution analysis only
        AutoLock child_lock(child_op_lock, false /*exclusive*/);
        for (std::deque<ReorderBufferEntry>::const_reverse_iterator it =
                 reorder_buffer.crbegin();
             it != reorder_buffer.crend(); it++)
        {
          if (it->operation_index < current_execution_fence_index)
            break;
          if (it->committed || (next_fence_index <= it->operation_index))
            continue;
          previous_events.insert(it->operation->get_completion_event());
        }
      }
      if (spy_logging_level > LIGHT_SPY_LOGGING)
      {
        // If we're doing execution record dependence on all previous operations
        // We can do this without the lock here because this is a fence and we
        // know that no other operations can be mapping in parallel with it
        previous_events.insert(
            previous_completion_events.begin(),
            previous_completion_events.end());
        // Don't include ourselves though
        previous_events.erase(op->get_completion_event());
      }
      // Also include the current execution fence in case the operation
      // already completed and wasn't in the set, make sure to do this
      // before we update the current fence
      if (current_execution_fence_event.exists())
        previous_events.insert(current_execution_fence_event);
    }

    //--------------------------------------------------------------------------
#ifdef LEGION_DEBUG_COLLECTIVES
    RefinementOp* InnerContext::get_refinement_op(
        Operation* op, RegionTreeNode* node)
#else
    RefinementOp* InnerContext::get_refinement_op(void)
#endif
    //--------------------------------------------------------------------------
    {
      return runtime->get_operation<RefinementOp>();
    }

    //--------------------------------------------------------------------------
    void InnerContext::update_current_mapping_fence(FenceOp* op)
    //--------------------------------------------------------------------------
    {
      current_mapping_fence = op;
      current_mapping_fence_gen = op->get_generation();
      current_mapping_fence_index = op->get_context_index();
      if (spy_logging_level > LIGHT_SPY_LOGGING)
      {
        current_fence_uid = op->get_unique_op_id();
        ops_since_last_fence.clear();
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::update_current_execution_fence(
        FenceOp* op, ApEvent fence_event)
    //--------------------------------------------------------------------------
    {
      // Only update the current fence event if we're actually an
      // execution fence, otherwise by definition we need the previous event
      current_execution_fence_event = fence_event;
      current_execution_fence_index = op->get_context_index();
    }

    //--------------------------------------------------------------------------
    void InnerContext::update_current_implicit_creation(Operation* op)
    //--------------------------------------------------------------------------
    {
      // Just overwrite since we know we already recorded a dependence
      // between this operation and the previous last deppart op
      last_implicit_creation = op;
      last_implicit_creation_gen = op->get_generation();
    }

    //--------------------------------------------------------------------------
    ApEvent InnerContext::get_current_execution_fence_event(void)
    //--------------------------------------------------------------------------
    {
      return current_execution_fence_event;
    }

    //--------------------------------------------------------------------------
    void InnerContext::begin_trace(
        TraceID tid, bool logical_only, bool static_trace,
        const std::set<RegionTreeID>* trees, bool deprecated,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (runtime->no_tracing)
        return;
      if (runtime->no_physical_tracing)
        logical_only = true;

      // No need to hold the lock here, this is only ever called
      // by the one thread that is running the task.
      if (current_trace != nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal nested trace with ID " << tid << " attempted in task "
              << get_task_name() << " (UID " << get_unique_id() << ").";
        error.raise();
      }

      std::map<TraceID, LogicalTrace*>::const_iterator finder =
          traces.find(tid);
      LogicalTrace* trace = nullptr;
      if (finder == traces.end())
      {
        // Trace does not exist yet, so make one and record it
        trace = new LogicalTrace(
            this, tid, logical_only, static_trace, provenance, trees);
        if (!deprecated)
          traces[tid] = trace;
        trace->add_reference();
      }
      else
        trace = finder->second;
      legion_assert(trace != nullptr);
      TraceOp* trace_op = nullptr;
      if (previous_trace == nullptr)
      {
        TraceBeginOp* begin = runtime->get_operation<TraceBeginOp>();
        begin->initialize_begin(this, trace, provenance);
        trace_op = begin;
      }
      else
      {
        TraceRecurrentOp* recurrent =
            runtime->get_operation<TraceRecurrentOp>();
        recurrent->initialize_recurrent(
            this, trace, previous_trace, provenance,
            (traces.find(previous_trace->tid) == traces.end()));
        trace_op = recurrent;
        previous_trace = nullptr;
      }
      if (trace->is_fixed() && trace->has_physical_trace())
      {
        // Record the event for when the trace replay is ready
        physical_trace_replay_status.store(trace_op->get_mapped_event().id);
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          tracing_replay_event = trace_op->get_completion_event();
      }
      add_to_dependence_queue(trace_op);
      // Now mark that we are starting a trace
      current_trace = trace;
      current_trace_blocking_index = next_blocking_index;
    }

    //--------------------------------------------------------------------------
    void InnerContext::record_physical_trace_replay(RtEvent ready, bool replay)
    //--------------------------------------------------------------------------
    {
      physical_trace_replay_status.compare_exchange_strong(
          ready.id, replay ? TRACE_REPLAYING : TRACE_NOT_REPLAYING);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::is_replaying_physical_trace(void)
    //--------------------------------------------------------------------------
    {
      if (current_trace == nullptr)
        return false;
      if (!current_trace->is_fixed())
        return false;
      realm_id_t status = physical_trace_replay_status.load();
      if (status > TRACE_REPLAYING)
      {
        // Result is not ready yet so wait until it is
        RtEvent ready;
        ready.id = status;
        if (!ready.has_triggered())
          ready.wait();
        status = physical_trace_replay_status.load();
        // No need to spin again because there won't be anymore outstanding
        // trace capture ops to be setting this
        legion_assert(status <= TRACE_REPLAYING);
      }
      return (status == TRACE_REPLAYING);
    }

    //--------------------------------------------------------------------------
    void InnerContext::end_trace(
        TraceID tid, bool deprecated, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (runtime->no_tracing)
        return;

      if (current_trace == nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Unmatched end trace for ID " << tid << " in task "
              << get_task_name() << " (UID " << get_unique_id() << ").";
        error.raise();
      }
      else if (!deprecated && (current_trace->tid != tid))
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal end trace call on trace ID " << tid
              << " that does not match the current trace ID "
              << current_trace->tid << " in task " << get_task_name()
              << " (UID " << get_unique_id() << ").";
        error.raise();
      }
      // Mark that the current trace is now fixed
      if (!current_trace->is_fixed())
        current_trace->fix_trace(provenance);
      else if (runtime->safe_tracing)
        current_trace->check_operation_count();
      current_trace->reset_intermediate_fence();
      previous_trace = current_trace;
      current_trace = nullptr;
      if (spy_logging_level > LIGHT_SPY_LOGGING)
        tracing_replay_event = ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    void InnerContext::record_blocking_call(
        uint64_t blocking_index, bool invalidate_trace)
    //--------------------------------------------------------------------------
    {
      // It's only a blocking call if the wait occurs from an operation
      // inside the trace so we can eliminate any waits from futures that
      // were produced before the trace or in the case of inline mappings
      // we know those operations are not traceable so they had to be
      // issued before we started capturing the trace
      if ((current_trace != nullptr) && invalidate_trace &&
          (blocking_index != NO_BLOCKING_INDEX) &&
          (current_trace_blocking_index <= blocking_index))
      {
        if (is_replaying_physical_trace())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Physical tracing violation! Trace "
                << current_trace->get_trace_id() << " in task "
                << get_task_name() << " (UID " << get_unique_id()
                << ") encountered a blocking API call that was unseen when it "
                   "was "
                << "recorded. It is required that traces do not change their "
                   "behavior.";
          error.raise();
        }
        else
          current_trace->record_blocking_call();
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::wait_on_future(FutureImpl* future, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      // This may look really bad that we're waiting for the producer
      // operation to commit, but that is absolutely necessary
      // to make sure all the region tree changes are captured and
      // propagated back up to the parent task which cannot happen until
      // we know the operation is not going to be restarted. This is why
      // it is so bad to wait on futures and we strongly discourage it.
      ready.wait();
    }

    //--------------------------------------------------------------------------
    void InnerContext::wait_on_future_map(FutureMapImpl* map, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      ready.wait();
    }

    //--------------------------------------------------------------------------
    void InnerContext::finish_frame(FrameOp* frame)
    //--------------------------------------------------------------------------
    {
      // Pull off all the frame events until we reach ours
      if (context_configuration.max_outstanding_frames > 0)
      {
        AutoLock child_lock(child_op_lock);
        legion_assert(!frame_ops.empty());
        legion_assert(frame_ops.front() == frame);
        frame_ops.pop_front();
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::increment_outstanding(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          (context_configuration.min_tasks_to_schedule == 0) ||
          (context_configuration.min_frames_to_schedule == 0));
      legion_assert(
          (context_configuration.min_tasks_to_schedule > 0) ||
          (context_configuration.min_frames_to_schedule > 0));
      AutoLock child_lock(child_op_lock);
      if (!currently_active_context && (outstanding_subtasks == 0) &&
          (((context_configuration.min_tasks_to_schedule > 0) &&
            (pending_subtasks < context_configuration.min_tasks_to_schedule)) ||
           ((context_configuration.min_frames_to_schedule > 0) &&
            (pending_frames < context_configuration.min_frames_to_schedule))))
      {
        currently_active_context = true;
        runtime->activate_context(this);
      }
      outstanding_subtasks++;
    }

    //--------------------------------------------------------------------------
    void InnerContext::decrement_outstanding(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          (context_configuration.min_tasks_to_schedule == 0) ||
          (context_configuration.min_frames_to_schedule == 0));
      legion_assert(
          (context_configuration.min_tasks_to_schedule > 0) ||
          (context_configuration.min_frames_to_schedule > 0));
      AutoLock child_lock(child_op_lock);
      legion_assert(outstanding_subtasks > 0);
      outstanding_subtasks--;
      if (currently_active_context && (outstanding_subtasks == 0) &&
          (((context_configuration.min_tasks_to_schedule > 0) &&
            (pending_subtasks < context_configuration.min_tasks_to_schedule)) ||
           ((context_configuration.min_frames_to_schedule > 0) &&
            (pending_frames < context_configuration.min_frames_to_schedule))))
      {
        currently_active_context = false;
        runtime->deactivate_context(this);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::increment_pending(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped frames
      if (context_configuration.min_tasks_to_schedule == 0)
        return;
      AutoLock child_lock(child_op_lock);
      pending_subtasks++;
      if (currently_active_context && (outstanding_subtasks > 0) &&
          (pending_subtasks == context_configuration.min_tasks_to_schedule))
      {
        currently_active_context = false;
        runtime->deactivate_context(this);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::decrement_pending(TaskOp* child)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduled by frames
      if (context_configuration.min_tasks_to_schedule > 0)
        decrement_pending(true /*need deferral*/);
    }

    //--------------------------------------------------------------------------
    void InnerContext::decrement_pending(bool need_deferral)
    //--------------------------------------------------------------------------
    {
      AutoLock child_lock(child_op_lock);
      legion_assert(pending_subtasks > 0);
      pending_subtasks--;
      if (!currently_active_context && (outstanding_subtasks > 0) &&
          (pending_subtasks < context_configuration.min_tasks_to_schedule))
      {
        currently_active_context = true;
        runtime->activate_context(this);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::increment_frame(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped tasks
      if (context_configuration.min_frames_to_schedule == 0)
        return;
      AutoLock child_lock(child_op_lock);
      pending_frames++;
      if (currently_active_context && (outstanding_subtasks > 0) &&
          (pending_frames == context_configuration.min_frames_to_schedule))
      {
        currently_active_context = false;
        runtime->deactivate_context(this);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::decrement_frame(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped tasks
      if (context_configuration.min_frames_to_schedule == 0)
        return;
      AutoLock child_lock(child_op_lock);
      legion_assert(pending_frames > 0);
      pending_frames--;
      if (!currently_active_context && (outstanding_subtasks > 0) &&
          (pending_frames < context_configuration.min_frames_to_schedule))
      {
        currently_active_context = true;
        runtime->activate_context(this);
      }
    }

    //--------------------------------------------------------------------------
#ifdef LEGION_DEBUG_COLLECTIVES
    MergeCloseOp* InnerContext::get_merge_close_op(
        Operation* op, RegionTreeNode* node)
#else
    MergeCloseOp* InnerContext::get_merge_close_op(void)
#endif
    //--------------------------------------------------------------------------
    {
      return runtime->get_operation<MergeCloseOp>();
    }

    //--------------------------------------------------------------------------
    void InnerContext::pack_inner_context(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(did);               // pack our distributed ID
      rez.serialize<DistributedID>(0);  // no shard manager
    }

    //--------------------------------------------------------------------------
    /*static*/ InnerContext* InnerContext::unpack_inner_context(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DistributedID ctx_did, man_did;
      derez.deserialize(ctx_did);
      derez.deserialize(man_did);
      if ((runtime->determine_owner(ctx_did) != runtime->address_space) &&
          (man_did > 0))
      {
        ShardManager* manager =
            runtime->find_shard_manager(man_did, true /*can fail*/);
        if (manager != nullptr)
          return manager->find_local_context();
      }
      return runtime->find_or_request_inner_context(ctx_did);
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_lock(Lock l)
    //--------------------------------------------------------------------------
    {
      // Can only be called from user land so no need to hold the lock
      context_locks.emplace_back(l.reservation_lock);
    }

    //--------------------------------------------------------------------------
    Grant InnerContext::acquire_grant(const std::vector<LockRequest>& requests)
    //--------------------------------------------------------------------------
    {
      // Kind of annoying, but we need to unpack and repack the
      // Lock type here to build new requests because the C++
      // type system is dumb with nested classes.
      std::vector<GrantImpl::ReservationRequest> unpack_requests(
          requests.size());
      for (unsigned idx = 0; idx < requests.size(); idx++)
      {
        unpack_requests[idx] = GrantImpl::ReservationRequest(
            requests[idx].lock.reservation_lock, requests[idx].mode,
            requests[idx].exclusive);
      }
      return Grant(new GrantImpl(unpack_requests));
    }

    //--------------------------------------------------------------------------
    void InnerContext::release_grant(Grant grant)
    //--------------------------------------------------------------------------
    {
      grant.impl->release_grant();
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_phase_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      // Can only be called from user land so no need to hold the lock
      context_barriers.emplace_back(pb.phase_barrier);
    }

    //--------------------------------------------------------------------------
    DynamicCollective InnerContext::create_dynamic_collective(
        unsigned arrivals, ReductionOpID redop, const void* init_value,
        size_t init_size)
    //--------------------------------------------------------------------------
    {
      if ((runtime->profiler != nullptr) &&
          !runtime->profiler->no_critical_paths &&
          !runtime->profiler->all_critical_arrivals)
      {
        Fatal fatal;
        fatal << "Task " << *this
              << " requested the creation of a dynamic collective while "
                 "profiling "
              << "for critical paths without recording all critical barrier "
                 "arrivals. "
              << "Critical path analysis with dynamic collectives requires "
                 "that you "
              << "use the '-lg:prof_all_critical_arrivals' flag.";
        fatal.raise();
      }
      return DynamicCollective(
          ApBarrier(Realm::Barrier::create_barrier(
              arrivals, redop, init_value, init_size)),
          redop);
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_dynamic_collective(DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      // Can only be called from user land so no need to hold the lock
      context_barriers.emplace_back(dc.phase_barrier);
    }

    //--------------------------------------------------------------------------
    void InnerContext::arrive_dynamic_collective(
        DynamicCollective dc, const void* buffer, size_t size, unsigned count)
    //--------------------------------------------------------------------------
    {
      runtime->phase_barrier_arrive(
          dc, count, ApEvent::NO_AP_EVENT, buffer, size);
    }

    //--------------------------------------------------------------------------
    void InnerContext::defer_dynamic_collective_arrival(
        DynamicCollective dc, const Future& future, unsigned count)
    //--------------------------------------------------------------------------
    {
      future.impl->contribute_to_collective(dc, count);
      // No need to register anything if this future is an application future
      // or it was made in a context above this in the region tree
      if ((future.impl->producer_op == nullptr) ||
          (future.impl->producer_depth < get_depth()))
        return;
      // Record this future as a contribution to the collective
      // for future dependence analysis
      const size_t barrier_gen =
          Realm::ID(dc.phase_barrier.id).event_generation();
      const size_t barrier_name = dc.phase_barrier.id - barrier_gen;
      AutoLock pb_lock(phase_barrier_lock);
      barrier_contributions[barrier_name].emplace_back(BarrierContribution(
          future.impl->producer_op, future.impl->op_gen,
          future.impl->producer_uid, 0 /*no muid*/, barrier_gen));
    }

    //--------------------------------------------------------------------------
    void InnerContext::perform_barrier_dependence_analysis(
        Operation* op, const std::vector<PhaseBarrier>& wait_barriers,
        const std::vector<PhaseBarrier>& arrive_barriers,
        MustEpochOp* must_epoch)
    //--------------------------------------------------------------------------
    {
      AutoLock pb_lock(phase_barrier_lock);
      if (!wait_barriers.empty())
        analyze_barrier_dependences(op, wait_barriers, must_epoch, true);
      if (!arrive_barriers.empty())
        analyze_barrier_dependences(op, arrive_barriers, must_epoch, false);
    }

    //--------------------------------------------------------------------------
    void InnerContext::analyze_barrier_dependences(
        Operation* op, const std::vector<PhaseBarrier>& barriers,
        MustEpochOp* must_epoch_op, bool previous_gen)
    //--------------------------------------------------------------------------
    {
      const UniqueID uid = op->get_unique_op_id();
      const GenerationID gen = op->get_generation();
      const UniqueID muid =
          (must_epoch_op == nullptr) ? 0 : must_epoch_op->get_unique_op_id();
      // Record our barriers for future uses
      for (const PhaseBarrier& ait : barriers)
      {
        // Figure out the generic barrier ID
        const ApBarrier barrier =
            previous_gen ? ait.phase_barrier :
                           Runtime::get_previous_phase(ait.phase_barrier);
        const size_t barrier_gen = Realm::ID(barrier.id).event_generation();
        const size_t barrier_name = barrier.id - barrier_gen;
        std::list<BarrierContribution>& previous =
            barrier_contributions[barrier_name];
        for (std::list<BarrierContribution>::iterator it = previous.begin();
             it != previous.end();
             /*nothing*/)
        {
          // skip anything with a larger barrier generation
          if (it->bargen >= barrier_gen)
          {
            it++;
            continue;
          }
          // If must epoch and same uid then skip it
          if ((muid > 0) && (muid == it->muid))
          {
            it++;
            continue;
          }
          LegionSpy::log_mapping_dependence(
              get_unique_id(), it->uid, 0, uid, 0, TRUE_DEPENDENCE);
          if (op->register_dependence(it->op, it->gen) ||
              // No pruning for Legion Spy
              (spy_logging_level > LIGHT_SPY_LOGGING))
            it++;
          else
            it = previous.erase(it);
        }
        previous.emplace_back(
            BarrierContribution(op, gen, uid, muid, barrier_gen));
      }
    }

    //--------------------------------------------------------------------------
    Future InnerContext::get_dynamic_collective_result(
        DynamicCollective dc, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      DynamicCollectiveOp* collective =
          runtime->get_operation<DynamicCollectiveOp>();
      Future result = collective->initialize(this, dc, provenance);
      add_to_dependence_queue(collective);
      return result;
    }

    //--------------------------------------------------------------------------
    DynamicCollective InnerContext::advance_dynamic_collective(
        DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      DynamicCollective result = dc;
      Runtime::advance_barrier(result);
      LegionSpy::log_event_dependence(dc.phase_barrier, result.phase_barrier);
      return result;
    }

    //--------------------------------------------------------------------------
    TaskPriority InnerContext::get_current_priority(void) const
    //--------------------------------------------------------------------------
    {
      return current_priority;
    }

    //--------------------------------------------------------------------------
    void InnerContext::set_current_priority(TaskPriority priority)
    //--------------------------------------------------------------------------
    {
      legion_assert(mutable_priority);
      legion_assert(realm_done_event.exists());
      // This can be racy but that is the mappers problem
      realm_done_event.set_operation_priority(priority);
      current_priority = priority;
    }

    //--------------------------------------------------------------------------
    void InnerContext::initialize_region_tree_contexts(
        const std::vector<RegionRequirement>& clone_requirements,
        const std::vector<ApUserEvent>& unmap_events)
    //--------------------------------------------------------------------------
    {
      // Save to cast to single task here because this will never
      // happen during inlining of index space tasks
      legion_assert(owner_task != nullptr);
      const std::deque<InstanceSet>& physical_instances =
          owner_task->get_physical_instances();
      const std::vector<bool>& no_access_regions =
          owner_task->get_no_access_regions();
      legion_assert(regions.size() <= physical_instances.size());
      legion_assert(regions.size() <= virtual_mapped.size());
      legion_assert(regions.size() <= no_access_regions.size());
      // Initialize all of the logical contexts no matter what
      //
      // For all of the physical contexts that were mapped, initialize them
      // with a specified reference to the current instance, otherwise
      // they were a virtual reference and we can ignore it.
      const ShardID local_shard = get_shard_id();
      const UniqueID context_uid = get_unique_id();
      std::map<PhysicalManager*, IndividualView*> top_views;
      for (unsigned idx1 = 0; idx1 < regions.size(); idx1++)
      {
        // this better be true for single tasks
        legion_assert(regions[idx1].handle_type == LEGION_SINGULAR_PROJECTION);
        // If this is a NO_ACCESS or had no privilege fields we can skip this
        if (no_access_regions[idx1])
          continue;
        const RegionRequirement& req = clone_requirements[idx1];
        if (virtual_mapped[idx1])
        {
          // If we're read-only or reduce-only we disallow refinements
          if (!IS_WRITE(req))
          {
            // In this case we also tell the region tree that this is
            // already refined so that no read or reduce refinements can
            // be performed in this context
            RegionNode* region_node = runtime->get_node(req.region);
            const FieldMask user_mask =
                region_node->column_source->get_field_mask(
                    req.privilege_fields);
            region_node->initialize_no_refine_fields(tree_context, user_mask);
          }
          continue;
        }
        RegionUsage usage(req);
        // Make this read-write so that users always pick up a dependence on it
        // for Legion Spy. This is a major hack and should be removed eventually
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          usage.privilege = LEGION_READ_WRITE;
        legion_assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
        // Make our equivalence set kd tree for look-ups
        RegionNode* region_node = runtime->get_node(req.region);
        EqKDTree* tree =
            region_node->row_source->create_equivalence_set_kd_tree(
                get_total_shards());
        equivalence_set_trees.emplace(idx1, EqKDRoot(tree));
        const FieldMask user_mask =
            region_node->column_source->get_field_mask(req.privilege_fields);
        EquivalenceSet* eq_set = create_initial_equivalence_set(idx1, req);
        const InstanceSet& sources = physical_instances[idx1];
        legion_assert(!sources.empty());
        // Find or make views for each of our instances and then
        // add initial users for each of them
        std::vector<IndividualView*> corresponding(sources.size());
        // Build our set of corresponding views
        for (unsigned idx2 = 0; idx2 < sources.size(); idx2++)
        {
          const InstanceRef& src_ref = sources[idx2];
          PhysicalManager* manager = src_ref.get_physical_manager();
          const FieldMask& view_mask = src_ref.get_valid_fields();
          legion_assert(!(view_mask - user_mask));  // should be dominated
          // Check to see if the view exists yet or not
          std::map<PhysicalManager*, IndividualView*>::const_iterator finder =
              top_views.find(manager);
          if (finder == top_views.end())
          {
            IndividualView* new_view =
                create_instance_top_view(manager, runtime->address_space);
            top_views[manager] = new_view;
            corresponding[idx2] = new_view;
            // Record the initial user for the instance
            new_view->add_initial_user(
                unmap_events[idx1], usage, view_mask, region_node->row_source,
                context_uid, idx1);
          }
          else
          {
            corresponding[idx2] = finder->second;
            // Record the initial user for the instance
            finder->second->add_initial_user(
                unmap_events[idx1], usage, view_mask, region_node->row_source,
                context_uid, idx1);
          }
        }
        // Restore the normal usage now that we've added the users
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          usage = RegionUsage(req);
        // Only need to do the initialization if we're the logical owner
        if (eq_set->is_logical_owner())
        {
          // The parent region requirement is restricted if it is
          // simultaneous or it is reduce-only. Simultaneous is
          // restricted because of normal Legion coherence semantics.
          // Reduce-only is restricted because we don't issue close
          // operations at the end of a context for reduce-only cases
          // right now so by making it restricted things are eagerly
          // flushed out to the parent task's instance.
          const bool restricted =
              IS_SIMULT(regions[idx1]) || IS_REDUCE(regions[idx1]);
          eq_set->initialize_set(
              usage, user_mask, restricted, sources, corresponding);
        }
        region_node->row_source->initialize_equivalence_set_kd_tree(
            tree, eq_set, user_mask, local_shard, false /*current*/);
        // Each equivalence set here comes with a reference that we
        // need to remove after we've registered it
        if (eq_set->remove_base_gc_ref(CONTEXT_REF))
          std::abort();  // should never hit this
      }
    }

    //--------------------------------------------------------------------------
    EquivalenceSet* InnerContext::create_initial_equivalence_set(
        unsigned idx, const RegionRequirement& req)
    //--------------------------------------------------------------------------
    {
      // This is the normal equivalence set creation pathway for single tasks
      IndexSpaceNode* node = runtime->get_node(req.region.index_space);
      EquivalenceSet* result = new EquivalenceSet(
          runtime->get_available_distributed_id(), runtime->address_space, node,
          req.region.get_tree_id(), find_parent_physical_context(idx),
          true /*register now*/);
      // Add a context ref that will be removed after this is registered
      result->add_base_gc_ref(CONTEXT_REF);
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::invalidate_logical_context(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (IS_NO_ACCESS(regions[idx]))
          continue;
        runtime->invalidate_region_tree_context(
            tree_context, regions[idx], false /*filter specific fields*/);
      }
      std::set<LogicalRegion> invalidated_regions;
      for (const std::pair<
               const LogicalRegion, std::pair<RegionRequirement, unsigned>>&
               req : created_requirements)
      {
        // Little tricky here, this is safe to invaliate the whole
        // tree even if we only had privileges on a field because
        // if we had privileges on the whole region in this context
        // it would have merged the created_requirement and we wouldn't
        // have a non returnable privilege requirement in this context
        runtime->invalidate_region_tree_context(
            tree_context, req.second.first, false /*filter specific fields*/);
        invalidated_regions.insert(req.second.first.region);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::invalidate_region_tree_contexts(
        const bool is_top_level_task, std::set<RtEvent>& applied,
        const ShardMapping* mapping, ShardID source_shard)
    //--------------------------------------------------------------------------
    {
      // Invalidate all our region contexts
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (IS_NO_ACCESS(regions[idx]))
          continue;
        RegionNode* node = runtime->get_node(regions[idx].region);
        std::map<unsigned, EqKDRoot>::iterator finder =
            equivalence_set_trees.find(idx);
        // Should not have any equivalence set trees for virtual mappings
        legion_assert(
            !virtual_mapped[idx] || (finder == equivalence_set_trees.end()));
        if (finder == equivalence_set_trees.end())
          continue;
        const FieldMask close_mask =
            node->column_source->get_field_mask(regions[idx].privilege_fields);
        std::vector<RtEvent> applied_events;
        node->row_source->invalidate_equivalence_set_kd_tree(
            finder->second.tree, finder->second.lock, close_mask,
            applied_events, false /*move to previous*/);
        equivalence_set_trees.erase(finder);
        if (!applied_events.empty())
          applied.insert(applied_events.begin(), applied_events.end());
      }
      // Also tell any traces to invalidate their references to the
      // equivalence set tree data structures
      for (const std::map<TraceID, LogicalTrace*>::value_type& it : traces)
        it.second->invalidate_equivalence_sets();
      if (!created_requirements.empty())
        invalidate_created_requirement_contexts(
            is_top_level_task, applied, mapping, source_shard);
    }

    //--------------------------------------------------------------------------
    void InnerContext::invalidate_created_requirement_contexts(
        const bool is_top, std::set<RtEvent>& applied,
        const ShardMapping* shard_mapping, ShardID source_shard)
    //--------------------------------------------------------------------------
    {
      std::map<RegionNode*, EqKDTree*> return_regions;
      for (const std::pair<
               const LogicalRegion, std::pair<RegionRequirement, unsigned>>&
               it : created_requirements)
      {
        RegionNode* node = runtime->get_node(it.second.first.region);
        legion_assert(
            returnable_privileges.find(it.second.second) !=
            returnable_privileges.end());
        std::map<unsigned, EqKDRoot>::iterator finder =
            equivalence_set_trees.find(it.second.second);
        if (finder == equivalence_set_trees.end())
          continue;
        // See if we're a returnable privilege or not
        if (returnable_privileges[it.second.second].second && !is_top)
        {
          legion_assert(return_regions.find(node) == return_regions.end());
          finder->second.tree->add_reference();
          return_regions[node] = finder->second.tree;
          equivalence_set_trees.erase(finder);
        }
        else
        {
          // Not returning so just remove it which will delete the tree
          legion_assert(return_regions.find(node) == return_regions.end());
          const FieldMask close_mask = node->column_source->get_field_mask(
              it.second.first.privilege_fields);
          std::vector<RtEvent> applied_events;
          node->row_source->invalidate_equivalence_set_kd_tree(
              finder->second.tree, finder->second.lock, close_mask,
              applied_events, false /*move to previous*/);
          equivalence_set_trees.erase(finder);
          if (!applied_events.empty())
            applied.insert(applied_events.begin(), applied_events.end());
        }
      }
      if (!return_regions.empty())
      {
        std::vector<RegionNode*> created_nodes;
        created_nodes.reserve(return_regions.size());
        std::vector<EqKDTree*> created_trees;
        created_trees.reserve(return_regions.size());
        for (const std::pair<RegionNode* const, EqKDTree*>& it : return_regions)
        {
          created_nodes.emplace_back(it.first);
          created_trees.emplace_back(it.second);
        }
        InnerContext* parent_ctx = find_parent_context();
        parent_ctx->receive_created_region_contexts(
            created_nodes, created_trees, applied, shard_mapping, source_shard);
        for (EqKDTree* const & tree : created_trees)
          if ((tree != nullptr) && tree->remove_reference())
            delete tree;
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::receive_created_region_contexts(
        const std::vector<RegionNode*>& created_nodes,
        const std::vector<EqKDTree*>& created_trees,
        std::set<RtEvent>& applied_events, const ShardMapping* mapping,
        ShardID source_shard)
    //--------------------------------------------------------------------------
    {
      legion_assert(created_nodes.size() == created_trees.size());
      AutoLock priv_lock(privilege_lock);
      for (unsigned idx = 0; idx < created_trees.size(); idx++)
      {
        unsigned index = add_created_region(
            created_nodes[idx]->handle, false /*task local*/,
            false /*output region*/);
        if (created_trees[idx] == nullptr)
          continue;
        // Check to see if we're the first one or whether we're merging
        std::map<unsigned, EqKDRoot>::const_iterator finder =
            equivalence_set_trees.find(index);
        if (finder != equivalence_set_trees.end())
        {
          // This happens when we're merging multiple trees coming back
          // from a sub-task that was control replicated, so we extract
          // the equivalence sets and add them into our tree
          local::FieldMaskMap<EquivalenceSet> eq_sets;
          created_trees[idx]->find_local_equivalence_sets(
              eq_sets, source_shard);
          const ShardID local_shard = get_shard_id();
          for (local::FieldMaskMap<EquivalenceSet>::const_iterator it =
                   eq_sets.begin();
               it != eq_sets.end(); it++)
            it->first->set_expr->initialize_equivalence_set_kd_tree(
                finder->second.tree, it->first, it->second, local_shard,
                true /*current*/);
        }
        else
        {
          finder =
              equivalence_set_trees.emplace(index, EqKDRoot(created_trees[idx]))
                  .first;
          // Filter all the current equivalence sets on to the previous
          const FieldMask all_ones_mask(LEGION_FIELD_MASK_FIELD_ALL_ONES);
          std::vector<RtEvent> applied;
          created_nodes[idx]->row_source->invalidate_equivalence_set_kd_tree(
              finder->second.tree, finder->second.lock, all_ones_mask, applied,
              true /*move to previous*/);
          if (!applied.empty())
            applied_events.insert(applied.begin(), applied.end());
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::invalidate_region_tree_context(
        const RegionRequirement& req, unsigned req_index,
        std::set<RtEvent>& applied_events, bool filter_specific_fields)
    //--------------------------------------------------------------------------
    {
      if (!req.privilege_fields.empty())
      {
        LocalLock* tree_lock = nullptr;
        EqKDTree* tree = find_equivalence_set_kd_tree(
            req_index, tree_lock, true /*null if doesn't exist*/);
        if (tree != nullptr)
        {
          std::vector<RtEvent> applied;
          RegionNode* node = runtime->get_node(req.region);
          const FieldMask invalidate_mask =
              node->column_source->get_field_mask(req.privilege_fields);
          node->row_source->invalidate_equivalence_set_kd_tree(
              tree, tree_lock, invalidate_mask, applied,
              false /*move to previous*/);
          if (!applied.empty())
            applied_events.insert(applied.begin(), applied.end());
        }
      }
      // Check to see if we should actually invalidate this tree
      if (!filter_specific_fields)
      {
        // Need the lock before doing this invalidation in case the
        // equivalence set trees data structure resizes
        AutoLock priv_lock(privilege_lock);
        equivalence_set_trees.erase(req_index);
        // Also need to remove the returnable privileges information
        // and any created region requirements
        std::map<unsigned, std::pair<LogicalRegion, bool>>::iterator finder =
            returnable_privileges.find(req_index);
        legion_assert(finder != returnable_privileges.end());
        created_requirements.erase(finder->second.first);
        returnable_privileges.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    ProjectionSummary* InnerContext::construct_projection_summary(
        Operation* op, unsigned index, const RegionRequirement& req,
        LogicalState* state, const ProjectionInfo& proj_info)
    //--------------------------------------------------------------------------
    {
      ProjectionNode* tree = proj_info.projection->construct_projection_tree(
          op, index, req, 0 /*local shard*/, state->owner, proj_info);
      return new ProjectionSummary(proj_info, tree, op, index, req, state);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::has_interfering_shards(
        ProjectionSummary* one, ProjectionSummary* two, bool& dominates)
    //--------------------------------------------------------------------------
    {
      legion_assert(dominates);
      return one->get_tree()->interferes(
          two->get_tree(), 0 /*local shard*/, dominates);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::match_timeouts(
        std::vector<LogicalUser*>& timeouts, TimeoutMatchExchange*& exchange)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here, we can invalidate all timeouts
      legion_assert(exchange == nullptr);
      return false;
    }

    //--------------------------------------------------------------------------
    std::pair<bool, bool> InnerContext::has_pointwise_dominance(
        ProjectionSummary* one, ProjectionSummary* two)
    //--------------------------------------------------------------------------
    {
      // Do the analysis in both directions
      ProjectionNode* t1 = one->get_tree();
      ProjectionNode* t2 = two->get_tree();
      return std::make_pair(
          t1->pointwise_dominates(t2), t2->pointwise_dominates(t1));
    }

    //--------------------------------------------------------------------------
    RtEvent InnerContext::find_pointwise_dependence(
        uint64_t context_index, const DomainPoint& point, ShardID shard,
        RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      Operation* op;
      GenerationID gen;
      {
        // We're just reading so only need the lock in read-only mode
        AutoLock child_lock(child_op_lock, false /*exclusive*/);
        // If the context index is less than what is at the front of the
        // reorder buffer then this operation was already retired
        if (reorder_buffer.empty() ||
            (context_index < reorder_buffer.front().operation_index))
        {
          if (to_trigger.exists())
            Runtime::trigger_event(to_trigger);
          return RtEvent::NO_RT_EVENT;
        }
        legion_assert(context_index <= reorder_buffer.back().operation_index);
        size_t offset = context_index - reorder_buffer.front().operation_index;
        const ReorderBufferEntry& entry = reorder_buffer[offset];
        if (entry.complete)
        {
          if (to_trigger.exists())
            Runtime::trigger_event(to_trigger);
          return RtEvent::NO_RT_EVENT;
        }
        legion_assert(entry.operation_index == context_index);
        op = entry.operation;
        // Have to do this while holding the lock to ensure it isn't
        // committed while we're getting the generation
        gen = op->get_generation();
      }
      // Now we can do the base call to get the operation
      return op->find_pointwise_dependence(point, gen, to_trigger);
    }

    //--------------------------------------------------------------------------
    void InnerContext::convert_individual_views(
        const std::vector<PhysicalManager*>& sources,
        std::vector<IndividualView*>& source_views, CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      source_views.resize(sources.size());
      std::vector<unsigned> still_needed;
      {
        AutoLock inst_lock(instance_view_lock, false /*exclusive*/);
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          // See if we can find it
          PhysicalManager* manager = sources[idx];
          std::map<PhysicalManager*, IndividualView*>::const_iterator finder =
              instance_top_views.find(manager);
          if (finder != instance_top_views.end())
          {
            // A little sanity check that the mappings match, if they don't
            // then that will lead to bigger problems
            legion_assert(
                (mapping == nullptr) ||
                (mapping == finder->second->collective_mapping) ||
                (*mapping == *(finder->second->collective_mapping)));
            source_views[idx] = finder->second;
          }
          else
            still_needed.emplace_back(idx);
        }
      }
      if (!still_needed.empty())
      {
        const AddressSpaceID local_space = runtime->address_space;
        for (const unsigned& it : still_needed)
        {
          PhysicalManager* manager = sources[it];
          source_views[it] =
              create_instance_top_view(manager, local_space, mapping);
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::convert_individual_views(
        const InstanceSet& sources, std::vector<IndividualView*>& source_views,
        CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      source_views.resize(sources.size());
      std::vector<unsigned> still_needed;
      {
        AutoLock inst_lock(instance_view_lock, false /*exclusive*/);
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          // See if we can find it
          PhysicalManager* manager = sources[idx].get_physical_manager();
          std::map<PhysicalManager*, IndividualView*>::const_iterator finder =
              instance_top_views.find(manager);
          if (finder != instance_top_views.end())
          {
            // A little sanity check that the mappings match, if they don't
            // then that will lead to bigger problems
            legion_assert(
                (mapping == nullptr) ||
                (mapping == finder->second->collective_mapping) ||
                (*mapping == *(finder->second->collective_mapping)));
            source_views[idx] = finder->second;
          }
          else
            still_needed.emplace_back(idx);
        }
      }
      if (!still_needed.empty())
      {
        const AddressSpaceID local_space = runtime->address_space;
        for (const unsigned& it : still_needed)
        {
          PhysicalManager* manager = sources[it].get_physical_manager();
          source_views[it] =
              create_instance_top_view(manager, local_space, mapping);
        }
      }
      // No need to invalidate the collective views here, we know that
      // source views are never going to be registered with the physical
      // analysis state so we can safely give out the individual views
    }

    //--------------------------------------------------------------------------
    void InnerContext::send_context(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      update_remote_instances(target);
      RemoteContextResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        pack_remote_context(rez, target);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ComputeEquivalenceSetsRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID context_did;
      derez.deserialize(context_did);
      // This should always be coming back to the owner node so there's no
      // need to defer this is at should always be here
      InnerContext* local_ctx = static_cast<InnerContext*>(
          runtime->find_distributed_collectable(context_did));
      size_t num_targets;
      derez.deserialize(num_targets);
      std::vector<EqSetTracker*> targets(num_targets);
      std::vector<AddressSpaceID> target_spaces(num_targets);
      for (unsigned idx = 0; idx < num_targets; idx++)
      {
        derez.deserialize(targets[idx]);
        derez.deserialize(target_spaces[idx]);
      }
      AddressSpaceID creation_target_space;
      derez.deserialize(creation_target_space);
      IndexSpaceExpression* expr =
          IndexSpaceExpression::unpack_expression(derez, source);
      FieldMask mask;
      derez.deserialize(mask);
      unsigned req_index;
      derez.deserialize(req_index);
      RtUserEvent ready_event;
      derez.deserialize(ready_event);

      const RtEvent done = local_ctx->compute_equivalence_sets(
          req_index, targets, target_spaces, creation_target_space, expr, mask);
      Runtime::trigger_event(ready_event, done);
    }

    //--------------------------------------------------------------------------
    void InnerContext::convert_analysis_views(
        const InstanceSet& targets,
        op::vector<op::FieldMaskMap<InstanceView>>& target_views)
    //--------------------------------------------------------------------------
    {
      target_views.resize(targets.size());
      std::vector<unsigned> still_needed;
      {
        AutoLock inst_lock(instance_view_lock, false /*exclusive*/);
        for (unsigned idx = 0; idx < targets.size(); idx++)
        {
          // See if we can find it
          const InstanceRef& ref = targets[idx];
          PhysicalManager* manager = ref.get_physical_manager();
          std::map<PhysicalManager*, IndividualView*>::const_iterator finder =
              instance_top_views.find(manager);
          if (finder != instance_top_views.end())
            target_views[idx].insert(finder->second, ref.get_valid_fields());
          else
            still_needed.emplace_back(idx);
        }
      }
      if (!still_needed.empty())
      {
        const AddressSpaceID local_space = runtime->address_space;
        for (const unsigned& it : still_needed)
        {
          const InstanceRef& ref = targets[it];
          PhysicalManager* manager = ref.get_physical_manager();
          target_views[it].insert(
              create_instance_top_view(manager, local_space),
              ref.get_valid_fields());
        }
      }
    }

    //--------------------------------------------------------------------------
    InnerContext::CollectiveResult*
        InnerContext::find_or_create_collective_view(
            RegionTreeID tid, const std::vector<DistributedID>& instances,
            RtEvent& ready)
    //--------------------------------------------------------------------------
    {
      // Just ignore the ready event since the result will be ready now
      return find_or_create_collective_view(tid, instances);
    }

    //--------------------------------------------------------------------------
    InnerContext::CollectiveResult*
        InnerContext::find_or_create_collective_view(
            RegionTreeID tid, const std::vector<DistributedID>& instances)
    //--------------------------------------------------------------------------
    {
      legion_assert(instances.size() > 1);
      AutoLock c_lock(collective_lock);
      std::vector<CollectiveResult*>& collectives = collective_results[tid];
      for (CollectiveResult* result : collectives)
      {
        if (result->matches(instances))
        {
          result->add_reference();
          return result;
        }
      }
      // If we get here then we need to make it
      std::vector<AddressSpaceID> spaces(instances.size());
      for (unsigned idx = 0; idx < spaces.size(); idx++)
        spaces[idx] = runtime->determine_owner(instances[idx]);
      std::sort(spaces.begin(), spaces.end());
      std::vector<AddressSpaceID>::iterator end =
          std::unique(spaces.begin(), spaces.end());
      spaces.resize(std::distance(spaces.begin(), end));
      CollectiveMapping* mapping =
          new CollectiveMapping(spaces, runtime->legion_collective_radix);
      mapping->add_reference();
      DistributedID collective_did =
          mapping->contains(runtime->address_space) ?
              runtime->get_available_distributed_id() :
              runtime->get_remote_distributed_id(
                  mapping->find_nearest(runtime->address_space));
      const RtEvent ready =
          create_collective_view(did, collective_did, mapping, instances);
      // This is a bit subtle, we need to encode the right kind of the
      // distributed ID (e.g. whether it is just replicated or allreduce)
      // The way we determine that is by looking at the distributed IDs of
      // the instances which also encode whether they are for reductions
      // instances or not
      const bool redop = InstanceManager::is_reduction_did(instances.back());
      if (redop)
        collective_did = LogicalView::encode_allreduce_did(collective_did);
      else
        collective_did = LogicalView::encode_replicated_did(collective_did);
      CollectiveResult* result =
          new CollectiveResult(instances, collective_did, ready);
      result->add_reference(2 /*one for us and one for result*/);
      collectives.emplace_back(result);
      if (mapping->remove_reference())
        delete mapping;
      return result;
    }

    //--------------------------------------------------------------------------
    RtEvent InnerContext::create_collective_view(
        DistributedID creator_did, DistributedID collective_did,
        CollectiveMapping* mapping,
        const std::vector<DistributedID>& individual_dids)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner_space =
          runtime->determine_owner(collective_did);
      if (mapping->contains(runtime->address_space))
      {
        // Send the result out to any children and then make our local copy
        std::vector<AddressSpaceID> children;
        mapping->get_children(owner_space, runtime->address_space, children);
        std::vector<RtEvent> done_events(children.size());
        for (unsigned idx = 0; idx < children.size(); idx++)
        {
          const RtUserEvent done = Runtime::create_rt_user_event();
          CollectiveViewCreation rez;
          {
            RezCheck z(rez);
            pack_inner_context(rez);
            rez.serialize(creator_did);
            rez.serialize(collective_did);
            mapping->pack(rez);
            rez.serialize<size_t>(individual_dids.size());
            for (DistributedID idv_did : individual_dids)
              rez.serialize(idv_did);
            rez.serialize(done);
          }
          rez.dispatch(children[idx]);
          done_events[idx] = done;
        }
        std::vector<IndividualView*> local_views;
        for (DistributedID idv_did : individual_dids)
        {
          if (runtime->determine_owner(idv_did) != runtime->address_space)
            continue;
          // Should always be able to find it since we're on the owner node
          PhysicalManager* manager = static_cast<PhysicalManager*>(
              runtime->find_distributed_collectable(idv_did));
          local_views.emplace_back(
              create_instance_top_view(manager, runtime->address_space));
        }
        legion_assert(!local_views.empty());
        ReductionOpID redop = local_views.back()->get_redop();
        CollectiveView* view = nullptr;
        if (redop > 0)
          view = new AllreduceView(
              collective_did, creator_did, local_views, individual_dids,
              false /*register now*/, mapping, redop);
        else
          view = new ReplicatedView(
              collective_did, creator_did, local_views, individual_dids,
              false /*register now*/, mapping);
        if (view->is_owner())
          view->add_nested_gc_ref(creator_did);
        view->register_with_runtime();
        if (!done_events.empty())
          return Runtime::merge_events(done_events);
        else
          return RtEvent::NO_RT_EVENT;
      }
      else
      {
        // Send this to the owner node to start the broadcast tree
        const RtUserEvent done = Runtime::create_rt_user_event();
        CollectiveViewCreation rez;
        {
          RezCheck z(rez);
          pack_inner_context(rez);
          rez.serialize(creator_did);
          rez.serialize(collective_did);
          mapping->pack(rez);
          rez.serialize<size_t>(individual_dids.size());
          for (DistributedID idv_did : individual_dids) rez.serialize(idv_did);
          rez.serialize(done);
        }
        rez.dispatch(owner_space);
        return done;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewCreation::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      InnerContext* context = InnerContext::unpack_inner_context(derez);
      DistributedID creator_did, collective_did;
      derez.deserialize(creator_did);
      derez.deserialize(collective_did);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping* mapping = new CollectiveMapping(derez, num_spaces);
      mapping->add_reference();
      size_t num_dids;
      derez.deserialize(num_dids);
      std::vector<DistributedID> individual_dids(num_dids);
      for (unsigned idx = 0; idx < num_dids; idx++)
        derez.deserialize(individual_dids[idx]);
      RtUserEvent done;
      derez.deserialize(done);
      Runtime::trigger_event(
          done, context->create_collective_view(
                    creator_did, collective_did, mapping, individual_dids));
      if (mapping->remove_reference())
        delete mapping;
    }

    //--------------------------------------------------------------------------
    void InnerContext::notify_collective_deletion(
        RegionTreeID tid, DistributedID collective_did)
    //--------------------------------------------------------------------------
    {
      bool found = false;
      {
        AutoLock c_lock(collective_lock);
        std::map<RegionTreeID, std::vector<CollectiveResult*>>::iterator
            finder = collective_results.find(tid);
        if (finder == collective_results.end())
          return;
        for (std::vector<CollectiveResult*>::iterator it =
                 finder->second.begin();
             it != finder->second.end(); it++)
        {
          if ((*it)->collective_did != collective_did)
            continue;
          found = true;
          delete (*it);
          finder->second.erase(it);
          break;
        }
      }
      if (found)
        release_collective_view(did, collective_did);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewDeletion::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID collective_did;
      derez.deserialize(collective_did);
      RegionTreeID tid;
      derez.deserialize(tid);
      DistributedID context_did;
      derez.deserialize(context_did);
      // The context might already be deleted so do a weak find
      InnerContext* context = static_cast<InnerContext*>(
          runtime->weak_find_distributed_collectable(context_did));
      if (context == nullptr)
        return;
      context->notify_collective_deletion(tid, collective_did);
      if (context->remove_base_resource_ref(RUNTIME_REF))
        delete context;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::release_collective_view(
        DistributedID context_did, DistributedID collective_did)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner = runtime->determine_owner(collective_did);
      if (owner != runtime->address_space)
      {
        CollectiveViewRelease rez;
        rez.serialize(context_did);
        rez.serialize(collective_did);
        rez.dispatch(owner);
      }
      else
      {
        // Better be able to find it since we know that we're still holding
        // a resource reference to it
        CollectiveView* view = static_cast<CollectiveView*>(
            runtime->find_distributed_collectable(collective_did));
        // Now remove the resource reference that was added by the
        // constructor for the collective view
        if (view->remove_nested_gc_ref(context_did))
          delete view;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewRelease::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID context_did, collective_did;
      derez.deserialize(context_did);
      derez.deserialize(collective_did);
      InnerContext::release_collective_view(context_did, collective_did);
    }

    //--------------------------------------------------------------------------
    IndividualView* InnerContext::create_instance_top_view(
        PhysicalManager* manager, AddressSpaceID request_source,
        CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      // Check to see if we already have the
      // instance, if we do, return it, otherwise make it and save it
      RtEvent wait_on;
      {
        AutoLock inst_lock(instance_view_lock);
        std::map<PhysicalManager*, IndividualView*>::const_iterator finder =
            instance_top_views.find(manager);
        if (finder != instance_top_views.end())
          // We've already got the view, so we are done
          return finder->second;
        // See if someone else is already making it
        std::map<PhysicalManager*, RtUserEvent>::iterator pending_finder =
            pending_top_views.find(manager);
        if (pending_finder == pending_top_views.end())
          // mark that we are making it
          pending_top_views[manager] = RtUserEvent::NO_RT_USER_EVENT;
        else
        {
          // See if we are the first one to follow
          if (!pending_finder->second.exists())
            pending_finder->second = Runtime::create_rt_user_event();
          wait_on = pending_finder->second;
        }
      }
      if (wait_on.exists())
      {
        // Someone else is making it so we just have to wait for it
        wait_on.wait();
        // Retake the lock and read out the result
        AutoLock inst_lock(instance_view_lock, false /*exclusive*/);
        std::map<PhysicalManager*, IndividualView*>::const_iterator finder =
            instance_top_views.find(manager);
        legion_assert(finder != instance_top_views.end());
        return finder->second;
      }
      IndividualView* result = manager->find_or_create_instance_top_view(
          this, request_source, mapping);
      // Use a gc reference here to ensure that the view is remains alive
      // everywhere until the instance is deleted or the context ends
      result->add_nested_gc_ref(did);
      // Record the result and trigger any user event to signal that the
      // view is ready
      RtUserEvent to_trigger;
      {
        AutoLock inst_lock(instance_view_lock);
        legion_assert(
            instance_top_views.find(manager) == instance_top_views.end());
        instance_top_views[manager] = result;
        std::map<PhysicalManager*, RtUserEvent>::iterator pending_finder =
            pending_top_views.find(manager);
        legion_assert(pending_finder != pending_top_views.end());
        to_trigger = pending_finder->second;
        pending_top_views.erase(pending_finder);
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
      return result;
    }

    //--------------------------------------------------------------------------
    FillView* InnerContext::find_or_create_fill_view(
        FillOp* op, const void* value, size_t value_size, RtEvent& ready)
    //--------------------------------------------------------------------------
    {
      // Two versions of this method depending on whether we are doing
      // Legion Spy or not, Legion Spy wants to know exactly which op
      // made each fill view so we can't cache them
      if (spy_logging_level <= LIGHT_SPY_LOGGING)
      {
        // See if we can find this in the cache first
        // No need for a lock if we're not control replicated
        for (std::list<FillViewEntry>::iterator it = fill_view_cache.begin();
             it != fill_view_cache.end(); it++)
        {
          // Skip comparing against fill views from futures
          if (it->future_did > 0)
            continue;
          // Safe to do this since we know we're only comparing against other
          // fill views that were also made with eager values
          if (!it->view->matches(value, value_size))
            continue;
          // Record a reference on it and then return
          FillView* result = it->view;
          result->add_base_valid_ref(MAPPING_ACQUIRE_REF);
          // Move it back to the front of the list
          fill_view_cache.splice(fill_view_cache.begin(), fill_view_cache, it);
          return result;
        }
        // At this point we have to make it since we couldn't find it
        FillView* fill_view = new FillView(
            runtime->get_available_distributed_id(), op->get_unique_op_id(),
            value, value_size, true /*register now*/);
        fill_view->add_base_valid_ref(MAPPING_ACQUIRE_REF);
        // Add it to the cache since we're not doing Legion Spy
        fill_view->add_nested_valid_ref(did);
        fill_view_cache.emplace_front(
            FillViewEntry{fill_view, 0 /*did*/, nullptr, 0});
        if (fill_view_cache.size() > MAX_FILL_VIEW_CACHE_SIZE)
        {
          FillView* oldest = fill_view_cache.back().view;
          if (oldest->remove_nested_valid_ref(did))
            delete oldest;
          fill_view_cache.pop_back();
        }
        return fill_view;
      }
      else
      {
        FillView* fill_view = new FillView(
            runtime->get_available_distributed_id(), op->get_unique_op_id(),
            value, value_size, true /*register now*/);
        fill_view->add_base_valid_ref(MAPPING_ACQUIRE_REF);
        return fill_view;
      }
    }

    //--------------------------------------------------------------------------
    FillView* InnerContext::find_or_create_fill_view(
        FillOp* op, const Future& future, bool& set_view, RtEvent& ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(!set_view);
      legion_assert(future.impl != nullptr);
      // Two versions of this method depending on whether we are doing
      // Legion Spy or not, Legion Spy wants to know exactly which op
      // made each fill view so we can't cache them
      if (spy_logging_level <= LIGHT_SPY_LOGGING)
      {
        const DistributedID future_did = future.impl->did;
        legion_assert(future_did != 0);
        // See if we can find this in the cache first
        // No need for a lock if we're not control replicated
        for (std::list<FillViewEntry>::iterator it = fill_view_cache.begin();
             it != fill_view_cache.end(); it++)
        {
          if (it->future_did != future_did)
            continue;
          // Record a reference on it and then return
          FillView* result = it->view;
          result->add_base_valid_ref(MAPPING_ACQUIRE_REF);
          // Move it back to the front of the list
          fill_view_cache.splice(fill_view_cache.begin(), fill_view_cache, it);
          return result;
        }
        // We're going to need to set the value for this view
        set_view = true;
        FillView* fill_view = new FillView(
            runtime->get_available_distributed_id(), op->get_unique_op_id(),
            true /*register now*/);
        fill_view->add_base_valid_ref(MAPPING_ACQUIRE_REF);
        // Add it to the cache since we're not doing Legion Spy
        fill_view->add_nested_valid_ref(did);
        fill_view_cache.emplace_front(
            FillViewEntry{fill_view, future_did, nullptr, 0});
        if (fill_view_cache.size() > MAX_FILL_VIEW_CACHE_SIZE)
        {
          FillView* oldest = fill_view_cache.back().view;
          if (oldest->remove_nested_valid_ref(did))
            delete oldest;
          fill_view_cache.pop_back();
        }
        return fill_view;
      }
      else
      {
        set_view = true;
        FillView* fill_view = new FillView(
            runtime->get_available_distributed_id(), op->get_unique_op_id(),
            true /*register now*/);
        fill_view->add_base_valid_ref(MAPPING_ACQUIRE_REF);
        return fill_view;
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::notify_instance_deletion(PhysicalManager* deleted)
    //--------------------------------------------------------------------------
    {
      InstanceView* removed = nullptr;
      {
        AutoLock inst_lock(instance_view_lock);
        std::map<PhysicalManager*, IndividualView*>::iterator finder =
            instance_top_views.find(deleted);
        if (finder == instance_top_views.end())
          return;
        removed = finder->second;
        instance_top_views.erase(finder);
      }
      if (removed->remove_nested_gc_ref(did))
        delete removed;
    }

    //--------------------------------------------------------------------------
    FutureInstance* InnerContext::create_task_local_future(
        Memory memory, size_t size, bool silence_warnings,
        const char* warning_string)
    //--------------------------------------------------------------------------
    {
      MemoryManager* manager = runtime->find_memory_manager(memory);
      // Safe to block indefinitely waiting on unbounded pools
      FutureInstance* instance = manager->create_future_instance(
          get_unique_id(), context_coordinates, size,
          nullptr /*safe_for_unbounded_pools*/);
      if (instance == nullptr)
      {
        const size_t remaining = manager->query_available_memory();
        if (size <= remaining)
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error << "Failed to allocate space for a future for task "
                << get_task_name() << " (UID " << get_unique_id() << ") in "
                << manager->get_name() << " memory of size " << size
                << " bytes. There are still " << remaining
                << " bytes free in the memory, but they are fragmented such "
                   "that a hole "
                << "of " << size
                << " bytes could not be found. We recommend you check the "
                << "order of allocations and alignment requirements to try to "
                   "minimize the "
                << "amount of padding between instances. Otherwise you will "
                   "need to increase "
                << "the size of the memory.";
          error.raise();
        }
        else
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error << "Failed to allocate space for a future for task "
                << get_task_name() << " (UID " << get_unique_id() << ") in "
                << manager->get_name() << " memory of size " << size
                << " bytes. If you receive this error then "
                << "you really are out of memory. You have two options: either "
                   "increase "
                << "the size of this memory when configuring Realm, or find a "
                   "bigger machine.";
          error.raise();
        }
      }
      return instance;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance InnerContext::create_task_local_instance(
        Memory memory, const Realm::InstanceLayoutGeneric& layout,
        bool can_fail, bool escaping, RtEvent& use_event)
    //--------------------------------------------------------------------------
    {
      LgEvent unique_event;
      // If we're profiling then each of these needs a unique event
      if (runtime->profiler != nullptr)
        Runtime::rename_event(unique_event);
      MemoryManager* manager = runtime->find_memory_manager(memory);
      PhysicalInstance instance = manager->create_task_local_instance(
          get_unique_id(), context_coordinates, unique_event, layout, use_event,
          nullptr /*safe_for_unbounded_pools*/);
      if (!instance.exists())
      {
        if (can_fail)
          return instance;
        const size_t remaining = manager->query_available_memory();
        if (layout.bytes_used <= remaining)
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error << "Failed to allocate DeferredBuffer/Value/Reduction for task "
                << get_task_name() << " (UID " << get_unique_id() << ") in "
                << manager->get_name() << " memory of size "
                << layout.bytes_used << " bytes. There are still " << remaining
                << " bytes free in the memory, "
                << "but they are fragmented such that a hole of "
                << layout.bytes_used << " bytes aligned on a "
                << layout.alignment_reqd << " byte boundary "
                << "could not be found. We recommend you check the order of "
                   "allocations "
                << "and alignment requirements to try to minimize the amount "
                   "of padding "
                << "between instances. Otherwise you will need to increase the "
                   "size of "
                << "the memory.";
          error.raise();
        }
        else
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error
              << "Failed to allocate DeferredBuffer/Value/Reduction for task "
              << get_task_name() << " (UID " << get_unique_id() << ") in "
              << manager->get_name() << " memory of size " << layout.bytes_used
              << " bytes. If you receive this error then you really are out of "
                 "memory. "
              << "You have two options: increase the size of this memory when "
              << "configuring Realm, or find a bigger machine.";
          error.raise();
        }
      }
      // We support multiple (OpenMP) threads calling in here simultaneously
      // so we need to protect this data structure, we co-opt the inline-Lock
      // here since we're unlikely to be contending with inline mappings
      AutoLock i_lock(inline_lock);
      task_local_instances[instance] =
          std::make_pair(unique_event, true /*escaping*/);
      return instance;
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_task_local_instance(
        PhysicalInstance instance, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock i_lock(inline_lock);
        std::map<PhysicalInstance, std::pair<LgEvent, bool>>::iterator finder =
            task_local_instances.find(instance);
        if (finder == task_local_instances.end())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Detected double deletion of deferred buffer " << instance
                << " in parent task " << get_task_name() << " (UID "
                << get_unique_id() << ").";
          error.raise();
        }
        task_local_instances.erase(finder);
      }
      MemoryManager* manager =
          runtime->find_memory_manager(instance.get_location());
#ifdef LEGION_MALLOC_INSTANCES
      manager->free_legion_instance(precondition, instance);
#else
      manager->free_task_local_instance(instance, precondition);
#endif
    }

    //--------------------------------------------------------------------------
    size_t InnerContext::query_available_memory(Memory memory, bool escaping)
    //--------------------------------------------------------------------------
    {
      MemoryManager* manager = runtime->find_memory_manager(memory);
      return manager->query_available_memory();
    }

    //--------------------------------------------------------------------------
    void InnerContext::release_memory_pool(Memory target)
    //--------------------------------------------------------------------------
    {
      // Nothing to do for inner tasks since they do not have memory pools
    }

    //--------------------------------------------------------------------------
    void InnerContext::end_task(
        const void* res, size_t res_size, bool owned,
        PhysicalInstance deferred_result_instance,
        FutureFunctor* callback_functor,
        const Realm::ExternalInstanceResource* resource,
        void (*freefunc)(const Realm::ExternalInstanceResource&),
        const void* metadataptr, size_t metadatasize, ApEvent effects)
    //--------------------------------------------------------------------------
    {
      if (overhead_profiler != nullptr)
      {
        const long long current = Realm::Clock::current_time_in_nanoseconds();
        const long long diff =
            current - overhead_profiler->previous_profiling_time;
        overhead_profiler->application_time += diff;
      }
      if (realm_done_event.exists())
      {
        // Case of a normal task
        legion_assert(!effects.exists());
        effects = realm_done_event;
        if (owner_task->is_concurrent())
          runtime->end_concurrent_task(executing_processor);
      }
      else  // implicit task
        realm_done_event = effects;
      // Check to see if we have any unordered operations that we need to inject
      // This has to be done before we do any deletions to make sure that all
      // these unordered operations are actually issued before deletions
      progress_unordered_operations(true /*end task*/);
      // Quick check to make sure the user didn't forget to end a trace
      if (current_trace != nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Task " << get_task_name() << " (UID " << get_unique_id()
              << ") failed to end trace before exiting!";
        error.raise();
      }
      // See if we have an outstanding previous trace to clean up
      else if (previous_trace != nullptr)
      {
        FenceOp* complete = initialize_trace_completion(nullptr /*provenance*/);
        // No longer have a previous trace
        previous_trace = nullptr;
        add_to_dependence_queue(complete);
      }
      // See if we have any local regions or fields that need to be deallocated
      std::vector<LogicalRegion> local_regions_to_delete;
      std::map<FieldSpace, std::set<FieldID>> local_fields_to_delete;
      {
        AutoLock priv_lock(privilege_lock, false /*exclusive*/);
        for (const std::pair<const LogicalRegion, bool>& it : local_regions)
          if (!it.second)
            local_regions_to_delete.emplace_back(it.first);
        for (const std::pair<const std::pair<FieldSpace, FieldID>, bool>& it :
             local_fields)
          if (!it.second)
            local_fields_to_delete[it.first.first].insert(it.first.second);
      }
      if (!local_regions_to_delete.empty())
      {
        for (const LogicalRegion& it : local_regions_to_delete)
          destroy_logical_region(
              it, false /*unordered*/, nullptr /*provenace*/);
      }
      if (!local_fields_to_delete.empty())
      {
        for (const std::pair<const FieldSpace, std::set<FieldID>>& it :
             local_fields_to_delete)
        {
          FieldAllocatorImpl* allocator =
              create_field_allocator(it.first, false /*unordered*/);
          allocator->add_reference();
          free_fields(
              allocator, it.first, it.second, false /*unordered*/,
              nullptr /*provenance*/);
          if (allocator->remove_reference())
            delete allocator;
        }
      }
      if (!index_launch_spaces.empty())
      {
        // These index spaces are now local to this context so we only
        // want to invoke our local deletion and not the global deletion
        // across all the shards in the case of control replication
        for (const std::pair<const Domain, IndexSpace>& it :
             index_launch_spaces)
          InnerContext::destroy_index_space(
              it.second, false /*unordered*/, true /*recurse*/,
              nullptr /*provenance*/);
      }
      // See if there are any runtime warnings to issue
      if (runtime->runtime_warnings)
      {
        if (total_children_count == 0)
        {
          // If there were no sub operations and this wasn't marked a
          // leaf task then signal a warning
          VariantImpl* impl = runtime->find_variant_impl(
              owner_task->task_id, owner_task->get_selected_variant());
          {
            Warning warning;
            warning << "Variant " << impl->get_name() << " of task "
                    << get_task_name() << " (UID " << get_unique_id()
                    << ") was not marked as a 'leaf' variant "
                    << "but it didn't execute any operations. Did you forget "
                       "the 'leaf' annotation?";
            warning.raise();
          }
        }
        else if (!owner_task->is_inner())
        {
          // If this task had sub operations and wasn't marked as inner
          // and made no accessors warn about missing 'inner' annotation
          // First check for any inline accessors that were made
          bool has_accessor = has_inline_accessor;
          if (!has_accessor)
          {
            for (unsigned idx = 0; idx < physical_regions.size(); idx++)
            {
              if (!physical_regions[idx].impl->created_accessor())
                continue;
              has_accessor = true;
              break;
            }
          }
          if (!has_accessor)
          {
            VariantImpl* impl = runtime->find_variant_impl(
                owner_task->task_id, owner_task->get_selected_variant());
            {
              Warning warning;
              warning << "Variant " << impl->get_name() << " of task "
                      << get_task_name() << " (UID " << get_unique_id()
                      << ") was not marked as an 'inner' variant "
                      << "but it only launched operations and did not make any "
                         "accessors. Did you "
                      << "forget the 'inner' annotation?";
              warning.raise();
            }
          }
        }
      }
      // Unmap any of our mapped regions before issuing any close operations
      unmap_all_regions(false /*external*/);
      const std::deque<InstanceSet>& physical_instances =
          owner_task->get_physical_instances();
      // Note that this loop doesn't handle create regions
      // we deal with that case below
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (virtual_mapped[idx])
          continue;
        // We also don't need to close up read-only instances
        // or reduction-only instances (because they are restricted)
        // so all changes have already been propagated
        if (!IS_WRITE(regions[idx]))
          continue;
        legion_assert(!physical_instances[idx].empty());
        PostCloseOp* close_op = runtime->get_operation<PostCloseOp>();
        close_op->initialize(this, idx, physical_instances[idx]);
        add_to_dependence_queue(close_op);
      }
      // At this point we should have grabbed any references to these
      // physical regions so we can clear them at this point
      physical_regions.clear();
      TaskContext::end_task(
          res, res_size, owned, deferred_result_instance, callback_functor,
          resource, freefunc, metadataptr, metadatasize, effects);
    }

    //--------------------------------------------------------------------------
    void InnerContext::post_end_task(void)
    //--------------------------------------------------------------------------
    {
      // If we weren't a leaf task, compute the conditions for being mapped
      // which is that all of our children are now mapped
      // Also test for whether we need to trigger any of our child
      // complete or committed operations before marking that we
      // are done executing
      bool need_commit = false;
      std::vector<RtEvent> preconditions;
      std::vector<ApEvent> completion_events;
      {
        AutoLock child_lock(child_op_lock);
        // Only need to do this for executing and executed children
        // We know that any complete children are done
        for (const ReorderBufferEntry& entry : reorder_buffer)
        {
          if (entry.complete)
            continue;
          RtEvent mapped = entry.operation->get_mapped_event();
          if (mapped.exists())
            preconditions.emplace_back(mapped);
        }
        legion_assert(!task_executed);
        // Now that we know the last registration has taken place we
        // can mark that we are done executing
        task_executed = true;
        if (spy_logging_level > LIGHT_SPY_LOGGING)
        {
          completion_events.insert(
              completion_events.end(),
              cummulative_child_completion_events.begin(),
              cummulative_child_completion_events.end());
          cummulative_child_completion_events.clear();
        }
        if (!reorder_buffer.empty())
        {
          for (const ReorderBufferEntry& entry : reorder_buffer)
            if (!entry.complete)
              completion_events.emplace_back(
                  entry.operation->get_completion_event());
            else if (entry.complete_event.exists())
              completion_events.emplace_back(entry.complete_event);
        }
        else
          need_commit = true;
      }
      if (!completion_events.empty())
      {
        completion_events.emplace_back(realm_done_event);
        owner_task->record_inner_termination(
            Runtime::merge_events(nullptr, completion_events));
      }
      else
        owner_task->record_inner_termination(realm_done_event);
      if (!preconditions.empty())
        owner_task->handle_post_mapped(Runtime::merge_events(preconditions));
      else
        owner_task->handle_post_mapped();
      if (need_commit)
        owner_task->trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void InnerContext::handle_mispredication(void)
    //--------------------------------------------------------------------------
    {
      owner_task->handle_post_mapped();
      owner_task->record_inner_termination(ApEvent::NO_AP_EVENT);
      unmap_all_regions(false /*external*/);
      TaskContext::handle_mispredication();
    }

    //--------------------------------------------------------------------------
    void InnerContext::PrepipelineArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      if (context->process_prepipeline_stage() &&
          context->remove_base_resource_ref(META_TASK_REF))
        delete context;
    }

    //--------------------------------------------------------------------------
    void InnerContext::DependenceArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      context->process_dependence_stage();
    }

    //--------------------------------------------------------------------------
    void InnerContext::TriggerReadyArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      if (context->process_ready_queue() &&
          context->remove_base_resource_ref(META_TASK_REF))
        delete context;
    }

    //--------------------------------------------------------------------------
    void InnerContext::DeferredEnqueueTaskArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      if (context->process_enqueue_task_queue(
              precondition, previous_fevent, performed) &&
          context->remove_base_resource_ref(META_TASK_REF))
        delete context;
    }

    //--------------------------------------------------------------------------
    void InnerContext::TriggerExecutionArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      if (context->process_trigger_execution_queue(
              precondition, previous_fevent, performed) &&
          context->remove_base_resource_ref(META_TASK_REF))
        delete context;
    }

    //--------------------------------------------------------------------------
    void InnerContext::DeferredExecutionArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      if (context->process_deferred_execution_queue(
              precondition, previous_fevent, performed) &&
          context->remove_base_resource_ref(META_TASK_REF))
        delete context;
    }

    //--------------------------------------------------------------------------
    void InnerContext::DeferredMappedArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      if (context->process_deferred_mapped_queue(
              precondition, previous_fevent, performed) &&
          context->remove_base_resource_ref(META_TASK_REF))
        delete context;
    }

    //--------------------------------------------------------------------------
    void InnerContext::DeferredCompletionArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      if (context->process_deferred_completion_queue(
              precondition, previous_fevent, performed) &&
          context->remove_base_resource_ref(META_TASK_REF))
        delete context;
    }

    //--------------------------------------------------------------------------
    void InnerContext::TriggerCommitArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      if (context->process_trigger_commit_queue() &&
          context->remove_base_resource_ref(META_TASK_REF))
        delete context;
    }

    //--------------------------------------------------------------------------
    void InnerContext::DeferredCommitArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      if (context->process_deferred_commit_queue(
              precondition, previous_fevent, performed) &&
          context->remove_base_resource_ref(META_TASK_REF))
        delete context;
    }

    //--------------------------------------------------------------------------
    bool InnerContext::inline_child_task(TaskOp* child)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_inline_task(child->get_unique_id());
      // Check to see if the child is predicated
      // If it is wait for it to resolve
      if (child->is_predicated_op())
      {
        // See if the predicate speculates false, if so return false
        // and then we are done.
        if (!child->get_predicate_value(total_children_count))
          return true;
      }
      // Find the mapped physical regions associated with each of the
      // child task's region requirements. If we don't have one then
      // it's not legal to inline the child task
      std::vector<PhysicalRegion> child_regions(child->regions.size());
      for (unsigned childidx = 0; childidx < child_regions.size(); childidx++)
      {
        const RegionRequirement& child_req = child->regions[childidx];
        bool found = false;
        for (unsigned our_idx = 0; our_idx < physical_regions.size(); our_idx++)
        {
          if (!physical_regions[our_idx].is_mapped())
            continue;
          const RegionRequirement& our_req = regions[our_idx];
          const RegionTreeID our_tid = our_req.region.get_tree_id();
          const IndexSpace our_space = our_req.region.get_index_space();
          const RegionUsage our_usage(our_req);
          if (!check_region_dependence(
                  our_tid, our_space, our_req, our_usage, child_req,
                  false /*ignore privileges*/))
            continue;
          child_regions[childidx] = physical_regions[our_idx];
          found = true;
          break;
        }
        if (found)
          continue;
        // Need the lock here because of unordered detach operations
        AutoLock i_lock(inline_lock, false /*exclusive*/);
        ctx::map<RegionTreeID, ctx::list<PhysicalRegion>>::const_iterator
            finder = inline_regions.find(child_req.parent.get_tree_id());
        if (finder != inline_regions.end())
        {
          for (const PhysicalRegion& region : finder->second)
          {
            legion_assert(region.is_mapped());
            const RegionRequirement& our_req = region.impl->get_requirement();
            const RegionTreeID our_tid = our_req.region.get_tree_id();
            const IndexSpace our_space = our_req.region.get_index_space();
            const RegionUsage our_usage(our_req);
            if (!check_region_dependence(
                    our_tid, our_space, our_req, our_usage, child_req,
                    false /*ignore privileges*/))
              continue;
            child_regions[childidx] = region;
            found = true;
            break;
          }
        }
        // If we didn't find any physical region then report the warning
        // and return because we couldn't find a mapped physical region
        if (!found)
        {
          Warning warning;
          warning << "Failed to inline task " << child->get_task_name()
                  << " (UID " << child->get_unique_id() << ") into parent task "
                  << owner_task->get_task_name() << " (UID "
                  << owner_task->get_unique_id()
                  << ") because there was no mapped "
                  << "region for region requirement " << childidx
                  << " to use. Currently all "
                  << "regions must be mapped in the parent task in order to "
                     "allow for inlining. "
                  << "If you believe you have a compelling use case for inline "
                     "a task with "
                  << "virtually mapped regions then please contact the Legion "
                     "developers.";
          warning.raise();
          return false;
        }
      }
      register_executing_child(child);
      // Now select the variant for task based on the regions
      std::deque<InstanceSet> physical_instances(child_regions.size());
      VariantImpl* variant =
          select_inline_variant(child, child_regions, physical_instances);
      child->perform_inlining(variant, physical_instances);
      // Finish the inlining of the child task to execute, note this doesn't
      // wait for the effects of the children to be done, it just blocks to
      // make sure the code for the children are done running on this processor
      wait_for_inlined();
      return true;
    }

    //--------------------------------------------------------------------------
    void InnerContext::analyze_free_local_fields(
        FieldSpace handle, const std::vector<FieldID>& local_to_free,
        std::vector<unsigned>& local_field_indexes)
    //--------------------------------------------------------------------------
    {
      AutoLock local_lock(local_field_lock, false /*exclusive*/);
      std::map<FieldSpace, std::vector<LocalFieldInfo>>::const_iterator finder =
          local_field_infos.find(handle);
      legion_assert(finder != local_field_infos.end());
      for (unsigned idx = 0; idx < local_to_free.size(); idx++)
      {
        [[maybe_unused]] bool found = false;
        for (const LocalFieldInfo& info : finder->second)
        {
          if (info.fid == local_to_free[idx])
          {
            // Can't remove it yet
            local_field_indexes.emplace_back(info.index);
            found = true;
            break;
          }
        }
        legion_assert(found);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::remove_deleted_local_fields(
        FieldSpace space, const std::vector<FieldID>& to_remove)
    //--------------------------------------------------------------------------
    {
      AutoLock local_lock(local_field_lock);
      std::map<FieldSpace, std::vector<LocalFieldInfo>>::iterator finder =
          local_field_infos.find(space);
      legion_assert(finder != local_field_infos.end());
      for (unsigned idx = 0; idx < to_remove.size(); idx++)
      {
        [[maybe_unused]] bool found = false;
        for (std::vector<LocalFieldInfo>::iterator it = finder->second.begin();
             it != finder->second.end(); it++)
        {
          if (it->fid == to_remove[idx])
          {
            finder->second.erase(it);
            found = true;
            break;
          }
        }
        legion_assert(found);
      }
      if (finder->second.empty())
        local_field_infos.erase(finder);
    }

    //--------------------------------------------------------------------------
    void InnerContext::execute_task_launch(
        TaskOp* task, bool index,
        const std::vector<StaticDependence>* dependences,
        Provenance* provenance, bool silence_warnings, bool inlining_enabled)
    //--------------------------------------------------------------------------
    {
      bool perform_inlining = false;
      if (inlining_enabled)
        perform_inlining = task->select_task_options(true /*prioritize*/);
      // Now check to see if we're inling the task or just performing
      // a normal asynchronous task launch
      if (!perform_inlining || !inline_child_task(task))
      {
        // Normal task launch, iterate over the context task's
        // regions and see if we need to unmap any of them
        std::vector<PhysicalRegion> unmapped_regions;
        if (!runtime->unsafe_launch)
          find_conflicting_regions(task, unmapped_regions);
        if (!unmapped_regions.empty())
        {
          if (runtime->runtime_warnings && !silence_warnings)
          {
            if (index)
            {
              Warning warning;
              warning << "WARNING: Runtime is unmapping and remapping physical "
                         "regions around "
                      << "execute_index_space call in task " << get_task_name()
                      << " (UID " << get_unique_id() << ").";
              warning.raise();
            }
            else
            {
              Warning warning;
              warning << "WARNING: Runtime is unmapping and remapping physical "
                         "regions around "
                      << "execute_task call in task " << get_task_name()
                      << " (UID " << get_unique_id() << ").";
              warning.raise();
            }
          }
          for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
            unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
        }
        // Issue the task call
        add_to_dependence_queue(task, dependences);
        // Remap any unmapped regions
        if (!unmapped_regions.empty())
          remap_unmapped_regions(current_trace, unmapped_regions, provenance);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::remap_unmapped_regions(
        LogicalTrace* trace,
        const std::vector<PhysicalRegion>& unmapped_regions,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      legion_assert(!unmapped_regions.empty());
      if (trace != nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal runtime remapping in trace " << trace->tid
              << " inside of task " << *this << ". Traces must perfectly "
              << "manage their physical mappings with no runtime help.";
        error.raise();
      }
      std::set<ApEvent> mapped_events;
      for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
      {
        const ApEvent ready =
            remap_region(unmapped_regions[idx], provenance, true /*internal*/);
        if (ready.exists())
          mapped_events.insert(ready);
      }
      // Wait for all the re-mapping operations to complete
      const ApEvent mapped_event =
          Runtime::merge_events(nullptr, mapped_events);
      bool poisoned = false;
      if (mapped_event.has_triggered_faultaware(poisoned))
      {
        if (poisoned)
          raise_poison_exception();
        return;
      }
      mapped_event.wait_faultaware(poisoned);
      if (poisoned)
        raise_poison_exception();
    }

    //--------------------------------------------------------------------------
    void InnerContext::clone_local_fields(
        std::map<FieldSpace, std::vector<LocalFieldInfo>>& child_local) const
    //--------------------------------------------------------------------------
    {
      legion_assert(child_local.empty());
      AutoLock local_lock(local_field_lock, false /*exclusive*/);
      if (local_field_infos.empty())
        return;
      for (const std::pair<const FieldSpace, std::vector<LocalFieldInfo>>& fit :
           local_field_infos)
      {
        std::vector<LocalFieldInfo>& child = child_local[fit.first];
        child.resize(fit.second.size());
        for (unsigned idx = 0; idx < fit.second.size(); idx++)
        {
          LocalFieldInfo& field = child[idx];
          field = fit.second[idx];
          field.ancestor = true;  // mark that this is an ancestor field
        }
      }
    }

    //--------------------------------------------------------------------------
    Operation* InnerContext::get_earliest(void) const
    //--------------------------------------------------------------------------
    {
      if (reorder_buffer.empty())
        return nullptr;
      return reorder_buffer.front().operation;
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_implicit_replay_dependence(Operation* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_mapping_dependence(
          get_unique_id(), current_fence_uid, 0 /*idx*/, op->get_unique_op_id(),
          0 /*idx*/, LEGION_TRUE_DEPENDENCE);
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::instantiate_subspace(
        IndexPartition parent, const void* realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* parent_node = runtime->get_node(parent);
      LegionColor child_color =
          parent_node->color_space->linearize_color(realm_color, type_tag);
      IndexSpaceNode* child_node = parent_node->get_child(child_color);
      return child_node->handle;
    }

    //--------------------------------------------------------------------------
    RtEvent InnerContext::create_pending_partition_internal(
        IndexPartition pid, IndexSpace parent, IndexSpace color_space,
        LegionColor& partition_color, PartitionKind part_kind,
        Provenance* provenance, CollectiveMapping* mapping, RtEvent initialized)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* parent_node = runtime->get_node(parent);
      IndexSpaceNode* color_node = runtime->get_node(color_space);
      if (partition_color == INVALID_COLOR)
        partition_color = parent_node->generate_color();
      // If we are making this partition on a different node than the
      // owner node of the parent index space then we have to tell that
      // owner node about the existence of this partition
      RtEvent parent_notified;
      const AddressSpaceID parent_owner = parent_node->get_owner_space();
      if ((parent_owner != runtime->address_space) &&
          ((mapping == nullptr) || !mapping->contains(parent_owner)) &&
          ((mapping == nullptr) ||
           (mapping->find_nearest(parent_owner) == runtime->address_space)))
      {
        RtUserEvent notified_event = Runtime::create_rt_user_event();
        IndexPartitionNotification rez;
        {
          RezCheck z(rez);
          rez.serialize(pid);
          rez.serialize(parent);
          rez.serialize(partition_color);
          rez.serialize(notified_event);
        }
        rez.dispatch(parent_owner);
        parent_notified = notified_event;
      }
      if ((part_kind == LEGION_COMPUTE_KIND) ||
          (part_kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        // Use 1 if we know it's complete, 0 if it's not,
        // otherwise -1 since we don't know
        const int complete = (part_kind == LEGION_COMPUTE_COMPLETE_KIND)   ? 1 :
                             (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND) ? 0 :
                                                                             -1;
        runtime->create_node(
            pid, parent_node, color_node, partition_color, complete, provenance,
            initialized, mapping);
        LegionSpy::log_index_partition(
            parent.get_id(), pid.get_id(), -1 /*unknown*/, complete,
            partition_color, runtime->address_space,
            (provenance == nullptr) ? std::string_view() : provenance->human);
      }
      else
      {
        const bool disjoint = (part_kind == LEGION_DISJOINT_KIND) ||
                              (part_kind == LEGION_DISJOINT_COMPLETE_KIND) ||
                              (part_kind == LEGION_DISJOINT_INCOMPLETE_KIND);
        // Use 1 if we know it's complete, 0 if it's not,
        // otherwise -1 since we don't know
        const int complete = ((part_kind == LEGION_DISJOINT_COMPLETE_KIND) ||
                              (part_kind == LEGION_ALIASED_COMPLETE_KIND)) ?
                                 1 :
                             ((part_kind == LEGION_DISJOINT_INCOMPLETE_KIND) ||
                              (part_kind == LEGION_ALIASED_INCOMPLETE_KIND)) ?
                                 0 :
                                 -1;
        runtime->create_node(
            pid, parent_node, color_node, partition_color, disjoint, complete,
            provenance, initialized, mapping);
        LegionSpy::log_index_partition(
            parent.get_id(), pid.get_id(), disjoint ? 1 : 0, complete,
            partition_color, runtime->address_space,
            (provenance == nullptr) ? std::string_view() : provenance->human);
      }
      register_index_partition_creation(pid);
      return parent_notified;
    }

    //--------------------------------------------------------------------------
    void InnerContext::create_pending_cross_product_internal(
        IndexPartition handle1, IndexPartition handle2,
        std::map<IndexSpace, IndexPartition>& user_handles, PartitionKind kind,
        Provenance* provenance, LegionColor& part_color,
        std::set<RtEvent>& safe_events, ShardID local_shard,
        const ShardMapping* shard_mapping,
        ValueBroadcast<LegionColor>* color_broadcast)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* base = runtime->get_node(handle1);
      IndexPartNode* source = runtime->get_node(handle2);
      // If we're supposed to compute this, but we already know that
      // the source is disjoint then we can also conlclude the the
      // resulting partitions will also be disjoint under intersection
      if (((kind == LEGION_COMPUTE_KIND) ||
           (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
           (kind == LEGION_COMPUTE_INCOMPLETE_KIND)) &&
          source->is_disjoint(true /*from app*/))
      {
        if (kind == LEGION_COMPUTE_KIND)
          kind = LEGION_DISJOINT_KIND;
        else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
          kind = LEGION_DISJOINT_COMPLETE_KIND;
        else
          kind = LEGION_DISJOINT_INCOMPLETE_KIND;
      }
      // If the source dominates the base then we know that all the
      // partitions that we are about to make will be complete
      if (((kind == LEGION_DISJOINT_KIND) || (kind == LEGION_ALIASED_KIND) ||
           (kind == LEGION_COMPUTE_KIND)) &&
          source->is_complete() && source->parent->dominates(base->parent))
      {
        if (kind == LEGION_DISJOINT_KIND)
          kind = LEGION_DISJOINT_COMPLETE_KIND;
        else if (kind == LEGION_ALIASED_KIND)
          kind = LEGION_ALIASED_COMPLETE_KIND;
        else
          kind = LEGION_COMPUTE_COMPLETE_KIND;
      }
      // If we haven't been given a color yet, we need to find
      // one that will be valid for all the child partitions
      // We don't have any way to atomically test and set a color
      // for all the partitions we're about to make so instead
      // we'll do this iteratively until we succeed, which
      // hopefully will not be too many iterations
      LegionColor lower_bound = 0;
      std::set<LegionColor> existing_colors;
      std::vector<IndexSpaceNode*> children_nodes;
      while (part_color == INVALID_COLOR)
      {
        // If this is the first time through populate the existing colors
        if ((lower_bound == 0) && existing_colors.empty())
        {
          for (ColorSpaceIterator itr(base); itr; itr++)
          {
            IndexSpaceNode* child_node = base->get_child(*itr);
            children_nodes.emplace_back(child_node);
            std::vector<LegionColor> colors;
            LegionColor bound = child_node->get_colors(colors);
            if (!colors.empty())
              existing_colors.insert(colors.begin(), colors.end());
            if (bound > lower_bound)
              lower_bound = bound;
          }
        }
        // Prune out any colors below the lower bound, we know they are never
        // going to be something that we can use across all the children
        while (!existing_colors.empty())
        {
          std::set<LegionColor>::iterator next = existing_colors.begin();
          if ((*next) <= lower_bound)
          {
            if ((*next) == lower_bound)
              lower_bound++;
            existing_colors.erase(next);
          }
          else
            break;
        }
        // Find the next available color
        part_color = lower_bound++;
        legion_assert(part_color != INVALID_COLOR);
        // Now confirm that we can reserve this color in all our subregions
        for (std::vector<IndexSpaceNode*>::const_iterator it =
                 children_nodes.begin();
             it != children_nodes.end(); it++)
        {
          LegionColor result = (*it)->generate_color(part_color);
          if (result == part_color)
            continue;
          // If we failed we need to remove all the failed colors
          for (std::vector<IndexSpaceNode*>::const_iterator it2 =
                   children_nodes.begin();
               it2 != it; it2++)
            (*it)->release_color(part_color);
          // Record that this is an existing color to skip
          existing_colors.insert(part_color);
          part_color = INVALID_COLOR;
          break;
        }
      }
      if (color_broadcast != nullptr)
        color_broadcast->broadcast(part_color);
      // Iterate over all our sub-regions and generate partitions
      if (shard_mapping == nullptr)
      {
        for (ColorSpaceIterator itr(base); itr; itr++)
        {
          IndexSpaceNode* child_node = base->get_child(*itr);
          IndexPartition pid(
              runtime->get_unique_index_partition_id(), handle1.get_tree_id(),
              handle1.get_type_tag());
          const RtEvent safe = create_pending_partition_internal(
              pid, child_node->handle, source->color_space->handle, part_color,
              kind, provenance);
          // If the user requested the handle for this point return it
          user_handles[child_node->handle] = pid;
          if (safe.exists())
            safe_events.insert(safe);
        }
      }
      else if (((LegionColor)shard_mapping->size()) <= base->total_children)
      {
        // There are more subregions than shards so we can shard the
        // children over all the shards to make the partitions
        for (ColorSpaceIterator itr(base, local_shard, shard_mapping->size());
             itr; itr++)
        {
          IndexSpaceNode* child_node = base->get_child(*itr);
          IndexPartition pid(
              runtime->get_unique_index_partition_id(), handle1.get_tree_id(),
              handle1.get_type_tag());
          const RtEvent safe = create_pending_partition_internal(
              pid, child_node->handle, source->color_space->handle, part_color,
              kind, provenance);
          // If the user requested the handle for this point return it
          user_handles[child_node->handle] = pid;
          if (safe.exists())
            safe_events.insert(safe);
        }
      }
      else
      {
        // There are fewer subregions than shards, so we can actually
        // have multiple shards collaborating to create each partition
        // Round-robin the shards over the children partitions to compute
        const unsigned color_index = local_shard % base->total_children;
        LegionColor child_color = color_index;
        if (base->total_children < base->max_linearized_color)
        {
          unsigned index = 0;
          for (ColorSpaceIterator itr(base); itr; itr++, index++)
          {
            if (index != color_index)
              continue;
            child_color = *itr;
            break;
          }
        }
        IndexSpaceNode* child_node = base->get_child(child_color);
        // Figure out how many shards are participating on this child
        // and what their address spaces are so we can make a collective
        // mapping for the new partition. Also tracke if we're the first
        // shard on this address space.
        bool first_local_shard = true;
        std::vector<AddressSpaceID> child_spaces;
        for (ShardID shard = color_index; shard < shard_mapping->size();
             shard += base->total_children)
        {
          const AddressSpaceID space = (*shard_mapping)[shard];
          if (std::binary_search(
                  child_spaces.begin(), child_spaces.end(), space))
          {
            if (shard == local_shard)
              first_local_shard = false;
            continue;
          }
          child_spaces.emplace_back(space);
          std::sort(child_spaces.begin(), child_spaces.end());
        }
        legion_assert(!child_spaces.empty());
        ReplicateContext* repl_ctx = legion_safe_cast<ReplicateContext*>(this);
        CrossProductExchange exchange(repl_ctx, COLLECTIVE_LOC_50);
        if (first_local_shard)
        {
          // If we're the first space for this child then make the
          // distributed ID and index partition name for the new partition
          // and then exchange in the collective
          if (child_spaces.front() == runtime->address_space)
          {
            IndexPartition pid(
                runtime->get_unique_index_partition_id(), handle1.get_tree_id(),
                handle1.get_type_tag());
            exchange.exchange_ids(child_color, pid);
            user_handles[child_node->handle] = pid;
          }
          else
            exchange.perform_collective_async();
          IndexPartition child_pid = IndexPartition::NO_PART;
          exchange.sync_child_ids(child_color, child_pid);
          const RtEvent safe = create_pending_partition_internal(
              child_pid, child_node->handle, source->color_space->handle,
              part_color, kind, provenance,
              new CollectiveMapping(
                  child_spaces, runtime->legion_collective_radix));
          if (safe.exists())
            safe_events.insert(safe);
        }
        else
        {
          // Still need to participate in the collective exchange
          exchange.perform_collective_sync();
        }
      }
    }

  }  // namespace Internal
}  // namespace Legion
