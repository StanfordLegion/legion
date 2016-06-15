/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "runtime.h"
#include "legion_ops.h"
#include "legion_tasks.h"
#include "region_tree.h"
#include "legion_spy.h"
#include "legion_trace.h"
#include "legion_profiling.h"
#include "legion_instances.h"
#include "legion_views.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Operation 
    /////////////////////////////////////////////////////////////

    const char *const 
      Operation::op_names[Operation::LAST_OP_KIND] = OPERATION_NAMES;

    //--------------------------------------------------------------------------
    Operation::Operation(Runtime *rt)
      : runtime(rt), op_lock(Reservation::create_reservation()), 
        gen(0), unique_op_id(0), 
        outstanding_mapping_references(0),
        hardened(false), parent_ctx(NULL)
    //--------------------------------------------------------------------------
    {
      dependence_tracker.mapping = NULL;
      if (!Runtime::resilient_mode)
        commit_event = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    Operation::~Operation(void)
    //--------------------------------------------------------------------------
    {
      op_lock.destroy_reservation();
      op_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    /*static*/ const char* Operation::get_string_rep(OpKind kind)
    //--------------------------------------------------------------------------
    {
      return op_names[kind];
    }

    //--------------------------------------------------------------------------
    void Operation::activate_operation(void)
    //--------------------------------------------------------------------------
    {
      // Get a new unique ID for this operation
      unique_op_id = runtime->get_unique_operation_id();
      outstanding_mapping_references = 0;
#ifdef DEBUG_LEGION
      mapped = false;
      executed = false;
      resolved = false;
#endif
      completed = false;
      committed = false;
      hardened = false;
      trigger_commit_invoked = true;
      early_commit_request = false;
      track_parent = false;
      parent_ctx = NULL;
      need_completion_trigger = true;
      mapped_event = Runtime::create_rt_user_event();
      resolved_event = Runtime::create_rt_user_event();
      completion_event = Runtime::create_ap_user_event();
      if (Runtime::resilient_mode)
        commit_event = Runtime::create_rt_user_event();
      trace = NULL;
      tracing = false;
      must_epoch = NULL;
#ifdef DEBUG_LEGION
      assert(mapped_event.exists());
      assert(resolved_event.exists());
      assert(completion_event.exists());
      if (Runtime::resilient_mode)
        assert(commit_event.exists());
#endif
      if (runtime->profiler != NULL)
        runtime->profiler->register_operation(this);
    }
    
    //--------------------------------------------------------------------------
    void Operation::deactivate_operation(void)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock o_lock(op_lock);
        gen++;
      }
      incoming.clear();
      outgoing.clear();
      unverified_regions.clear();
      verify_regions.clear();
      logical_records.clear();
      if (dependence_tracker.commit != NULL)
      {
        delete dependence_tracker.commit;
        dependence_tracker.commit = NULL;
      }
      if (!mapped_event.has_triggered())
        Runtime::trigger_event(mapped_event);
      if (!resolved_event.has_triggered())
        Runtime::trigger_event(resolved_event);
      if (need_completion_trigger && !completion_event.has_triggered())
        Runtime::trigger_event(completion_event);
      if (!commit_event.has_triggered())
        Runtime::trigger_event(commit_event);
    }

    //--------------------------------------------------------------------------
    size_t Operation::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 0;
    }

    //--------------------------------------------------------------------------
    Mappable* Operation::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return parent_ctx;
    }

    //--------------------------------------------------------------------------
    unsigned Operation::get_operation_depth(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent_ctx != NULL);
#endif
      return (parent_ctx->get_depth()+1);
    }

    //--------------------------------------------------------------------------
    void Operation::initialize_privilege_path(RegionTreePath &path,
                                              const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      if ((req.handle_type == SINGULAR) || (req.handle_type == REG_PROJECTION))
      {
        runtime->forest->initialize_path(req.region.get_index_space(),
                                         req.parent.get_index_space(), path);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(req.handle_type == PART_PROJECTION);
#endif
        runtime->forest->initialize_path(req.partition.get_index_partition(),
                                         req.parent.get_index_space(), path);
      }
    }

    //--------------------------------------------------------------------------
    void Operation::initialize_mapping_path(RegionTreePath &path,
                                            const RegionRequirement &req,
                                            LogicalRegion start_node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
#endif
      runtime->forest->initialize_path(req.region.get_index_space(),
                                       start_node.get_index_space(), path);
    }

    //--------------------------------------------------------------------------
    void Operation::initialize_mapping_path(RegionTreePath &path,
                                            const RegionRequirement &req,
                                            LogicalPartition start_node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
#endif
      runtime->forest->initialize_path(req.region.get_index_space(),
                                       start_node.get_index_partition(), path);
    }

    //--------------------------------------------------------------------------
    void Operation::set_trace(LegionTrace *t, bool is_tracing)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace == NULL);
      assert(t != NULL);
#endif
      trace = t; 
      tracing = is_tracing;
    }

    //--------------------------------------------------------------------------
    void Operation::set_must_epoch(MustEpochOp *epoch, bool do_registration)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(must_epoch == NULL);
      assert(epoch != NULL);
#endif
      must_epoch = epoch;
      if (do_registration)
        must_epoch->register_subop(this);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::localize_region_requirement(RegionRequirement &r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(r.handle_type == SINGULAR);
#endif
      r.parent = r.region;
      r.prop = EXCLUSIVE;
    }

    //--------------------------------------------------------------------------
    void Operation::release_acquired_instances(
       std::map<PhysicalManager*,std::pair<unsigned,bool> > &acquired_instances)
    //--------------------------------------------------------------------------
    {
      for (std::map<PhysicalManager*,std::pair<unsigned,bool> >::iterator it = 
            acquired_instances.begin(); it != acquired_instances.end(); it++)
      {
        if (it->first->remove_base_valid_ref(MAPPING_ACQUIRE_REF, this, 
                                             it->second.first))
          PhysicalManager::delete_physical_manager(it->first);
      }
      acquired_instances.clear();
    }

    //--------------------------------------------------------------------------
    void Operation::initialize_operation(SingleTask *ctx, bool track, 
                                         unsigned regs/*= 0*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
      assert(completion_event.exists());
#endif
      parent_ctx = ctx;
      track_parent = track;
      if (track_parent)
        parent_ctx->register_new_child_operation(this);
      for (unsigned idx = 0; idx < regs; idx++)
        unverified_regions.insert(idx);
    }

    //--------------------------------------------------------------------------
    void Operation::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      // should be overwridden by inheriting classes
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      begin_dependence_analysis();
      end_dependence_analysis();
      // Since there are no registered dependences here
      // then trigger mapping will occur immediately
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Then put this thing on the ready queue
      runtime->add_to_local_queue(parent_ctx->get_executing_processor(),
                                  this, false/*prev fail*/);
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_remote_state_analysis(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      // We have nothing to check for so just trigger the event
      Runtime::trigger_event(ready_event);
    }

    //--------------------------------------------------------------------------
    bool Operation::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      // Mark that we finished mapping
      complete_mapping();
      // If we have nothing to do also mark that we have completed execution
      complete_execution();
      // Return true indicating we successfully triggered
      return true;
    }
    
    //--------------------------------------------------------------------------
    void Operation::trigger_resolution(void)
    //--------------------------------------------------------------------------
    {
      resolve_speculation();
    } 

    //--------------------------------------------------------------------------
    void Operation::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void Operation::deferred_execute(void)
    //--------------------------------------------------------------------------
    {
      // should only be called if overridden
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Operation::deferred_commit_trigger(GenerationID our_gen)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(our_gen <= gen); // better not be ahead of where we are now
#endif
        if ((our_gen == gen) && !trigger_commit_invoked)
        {
          trigger_commit_invoked = true;
          need_trigger = true;
        }
      }
      if (need_trigger)
        trigger_commit();
    }

    //--------------------------------------------------------------------------
    void Operation::report_interfering_requirements(unsigned idx1,unsigned idx2)
    //--------------------------------------------------------------------------
    {
      // should only be called if overridden
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Operation::report_interfering_close_requirement(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // should only be called if overridden
      assert(false);
    }

    //--------------------------------------------------------------------------
    unsigned Operation::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    FatTreePath* Operation::compute_fat_path(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    void Operation::select_sources(const InstanceRef &target,
                                   const InstanceSet &sources,
                                   std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      assert(false);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                                     Operation::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      // should only be called for inherited types
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    void Operation::update_atomic_locks(Reservation lock, bool exclusive)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Operation::complete_mapping(RtEvent wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      {
        AutoLock o_lock(op_lock);
        assert(!mapped);
        mapped = true;
      }
#endif
      Runtime::trigger_event(mapped_event, wait_on);
    }

    //--------------------------------------------------------------------------
    void Operation::complete_execution(RtEvent wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
      if (wait_on.exists() && !wait_on.has_triggered())
      {
        // We have to defer the execution of this operation
        DeferredExecuteArgs args;
        args.hlr_id = HLR_DEFERRED_EXECUTE_ID;
        args.proxy_this = this;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_DEFERRED_EXECUTE_ID,
                                         HLR_LATENCY_PRIORITY,
                                         this, wait_on);
        return;
      }
      // Tell our parent context that we are done mapping
      // It's important that this is done before we mark that we
      // are executed to avoid race conditions
      if (track_parent)
        parent_ctx->register_child_executed(this);
#ifdef DEBUG_LEGION
      {
        AutoLock o_lock(op_lock);
        assert(!executed);
        executed = true;
      }
#endif
      // Now see if we are ready to complete this operation
      if (!mapped_event.has_triggered() || !resolved_event.has_triggered())
      {
        RtEvent trigger_pre = 
          Runtime::merge_events(mapped_event, resolved_event);
        TriggerCompleteArgs args;
        args.hlr_id = HLR_TRIGGER_COMPLETE_ID;
        args.proxy_this = this;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_TRIGGER_COMPLETE_ID,
                                         HLR_LATENCY_PRIORITY,
                                         this, trigger_pre);
      }
      else // Do the trigger now
        trigger_complete();
    }

    //--------------------------------------------------------------------------
    void Operation::resolve_speculation(RtEvent wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      {
        AutoLock o_lock(op_lock);
        assert(!resolved);
        resolved = true;
      }
#endif
      Runtime::trigger_event(resolved_event, wait_on);
    }

    //--------------------------------------------------------------------------
    void Operation::complete_operation(RtEvent wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
      if (wait_on.exists() && !wait_on.has_triggered())
      {
        DeferredCompleteArgs args;
        args.hlr_id = HLR_DEFERRED_COMPLETE_ID;
        args.proxy_this = this;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_DEFERRED_COMPLETE_ID,
                                         HLR_LATENCY_PRIORITY,
                                         this, wait_on);
        return;
      }
      bool need_trigger = false;
      // Tell our parent that we are complete
      // It's important that we do this before we mark ourselves
      // completed in order to avoid race conditions
      if (track_parent)
        parent_ctx->register_child_complete(this);
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(mapped);
        assert(executed);
        assert(resolved);
        assert(!completed);
#endif
        completed = true;
        // Now that we have done the completion stage, we can 
        // mark trigger commit to false which will open all the
        // different path ways for doing commit, this also
        // means we need to check all the ways here because they
        // have been disable previously
        trigger_commit_invoked = false;
        // Check to see if we need to trigger commit
        if ((!Runtime::resilient_mode) || early_commit_request ||
            ((hardened && unverified_regions.empty())))
        {
          trigger_commit_invoked = true;
          need_trigger = true;
        }
        else if (outstanding_mapping_references == 0)
        {
#ifdef DEBUG_LEGION
          assert(dependence_tracker.commit != NULL);
#endif
          CommitDependenceTracker *tracker = dependence_tracker.commit;
          need_trigger = tracker->issue_commit_trigger(this, runtime);
          if (need_trigger)
            trigger_commit_invoked = true;
        }
      }
      if (need_completion_trigger)
        Runtime::trigger_event(completion_event);
      // finally notify all the operations we dependended on
      // that we validated their regions note we don't need
      // the lock since this was all set when we did our mapping analysis
      for (std::map<Operation*,std::set<unsigned> >::const_iterator it =
            verify_regions.begin(); it != verify_regions.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(incoming.find(it->first) != incoming.end());
#endif
        GenerationID ver_gen = incoming[it->first];
        it->first->notify_regions_verified(it->second, ver_gen);
      } 
      // If we're not in resilient mode, then we can now
      // commit this operation
      if (need_trigger)
        trigger_commit();
    }

    //--------------------------------------------------------------------------
    void Operation::commit_operation(bool do_deactivate,
                                     RtEvent wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
      if (wait_on.exists() && !wait_on.has_triggered())
      {
        DeferredCommitArgs args;
        args.hlr_id = HLR_DEFERRED_COMMIT_ID;
        args.proxy_this = this;
        args.deactivate = do_deactivate;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_DEFERRED_COMMIT_ID,
                                         HLR_LATENCY_PRIORITY,
                                         this, wait_on);
        return;
      }
      // Tell our parent context that we are committed
      // Do this before actually committing to avoid race conditions
      if (track_parent)
        parent_ctx->register_child_commit(this);
      // Mark that we are committed 
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(mapped);
        assert(executed);
        assert(resolved);
        assert(completed);
        assert(!committed);
#endif
        committed = true;
      } 
      // Trigger the commit event
      if (Runtime::resilient_mode)
        Runtime::trigger_event(commit_event);
      if (do_deactivate)
        deactivate();
    }

    //--------------------------------------------------------------------------
    void Operation::harden_operation(void)
    //--------------------------------------------------------------------------
    {
      // Mark that this operation is now hardened against failures
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(!hardened);
#endif
        hardened = true;
        if (unverified_regions.empty() && !trigger_commit_invoked)
        {
          trigger_commit_invoked = true;
          need_trigger = true;
        }
      }
      if (need_trigger)
        trigger_commit();
    }

    //--------------------------------------------------------------------------
    void Operation::quash_operation(GenerationID gen, bool restart)
    //--------------------------------------------------------------------------
    {
      // TODO: actually handle quashing of operations
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Operation::request_early_commit(void)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      early_commit_request = true;
    }

    //--------------------------------------------------------------------------
    void Operation::begin_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dependence_tracker.mapping == NULL);
#endif
      // Make a dependence tracker
      dependence_tracker.mapping = new MappingDependenceTracker();
      // Register ourselves with our trace if there is one
      // This will also add any necessary dependences
      if (trace != NULL)
        trace->register_operation(this, gen);
      // See if we have any fence dependences
      parent_ctx->register_fence_dependence(this);
    }

    //--------------------------------------------------------------------------
    void Operation::end_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dependence_tracker.mapping != NULL);
#endif
      MappingDependenceTracker *tracker = dependence_tracker.mapping;
      // Now make a commit tracker
      dependence_tracker.commit = new CommitDependenceTracker();
      // Cannot touch anything not on our stack after this call
      tracker->issue_stage_triggers(this, runtime, must_epoch);
      delete tracker;
    }

    //--------------------------------------------------------------------------
    bool Operation::register_dependence(Operation *target, 
                                        GenerationID target_gen)
    //--------------------------------------------------------------------------
    {
      if (must_epoch != NULL)
        must_epoch->verify_dependence(this, gen, target, target_gen);
      // The rest of this method is the same as the one below
      if (target == this)
      {
        // Can't remove this if we are tracing
        if (tracing)
        {
          // Don't forget to record the dependence
#ifdef DEBUG_LEGION
          assert(trace != NULL);
#endif
          if (target_gen < gen)
            trace->record_dependence(this, target_gen, this, gen);
          return false;
        }
        else
          return (target_gen < gen);
      }
      bool registered_dependence = false;
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(dependence_tracker.mapping != NULL);
#endif
      bool prune = target->perform_registration(target_gen, this, gen,
                                                registered_dependence,
                                                dependence_tracker.mapping,
                                                commit_event);
      if (registered_dependence)
        incoming[target] = target_gen;
      if (tracing)
      {
#ifdef DEBUG_LEGION
        assert(trace != NULL);
#endif
        trace->record_dependence(target, target_gen, this, gen);
        // Unsound to prune when tracing
        prune = false;
      }
      return prune;
    }

    //--------------------------------------------------------------------------
    bool Operation::register_region_dependence(unsigned idx, Operation *target,
                                          GenerationID target_gen, 
                                          unsigned target_idx,
                                          DependenceType dtype, bool validates,
                                          const FieldMask &dependent_mask)
    //--------------------------------------------------------------------------
    {
      bool do_registration = true;
      if (must_epoch != NULL)
      {
        do_registration = 
          must_epoch->record_dependence(this, gen, target, target_gen, 
                                        idx, target_idx, dtype);
      }
      // Can never register a dependence on ourself since it means
      // that the target was recycled and will never register. Return
      // true if the generation is older than our current generation.
      if (target == this)
      {
        if (target_gen == gen)
          report_interfering_requirements(target_idx, idx);
        // Can't remove this if we are tracing
        if (tracing)
        {
          // Don't forget to record the dependence
#ifdef DEBUG_LEGION
          assert(trace != NULL);
#endif
          if (target_gen < gen)
            trace->record_region_dependence(this, target_gen, 
                                            this, gen, target_idx, 
                                            idx, dtype, validates,
                                            dependent_mask);
          return false;
        }
        else
          return (target_gen < gen);
      }
      bool registered_dependence = false;
      AutoLock o_lock(op_lock);
      bool prune = false;
      if (do_registration)
      {
#ifdef DEBUG_LEGION
        assert(dependence_tracker.mapping != NULL);
#endif
        prune = target->perform_registration(target_gen, this, gen,
                                                registered_dependence,
                                                dependence_tracker.mapping,
                                                commit_event);
      }
      if (registered_dependence)
      {
        incoming[target] = target_gen;
        // If we registered a mapping dependence then we can verify
        if (validates)
          verify_regions[target].insert(idx);
      }
      if (tracing)
      {
#ifdef DEBUG_LEGION
        assert(trace != NULL);
#endif
        trace->record_region_dependence(target, target_gen, 
                                        this, gen, target_idx, 
                                        idx, dtype, validates,
                                        dependent_mask);
        // Unsound to prune when tracing
        prune = false;
      }
      return prune;
    }

    //--------------------------------------------------------------------------
    bool Operation::perform_registration(GenerationID our_gen, 
                                         Operation *op, GenerationID op_gen,
                                         bool &registered_dependence,
                                         MappingDependenceTracker *tracker,
                                         RtEvent other_commit_event)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(our_gen <= gen); // better not be ahead of where we are now
#endif
      // If the generations match and we haven't committed yet, 
      // register an outgoing dependence
      if ((our_gen == gen) && !committed)
      {
#ifdef DEBUG_LEGION
        // should still have some mapping references
        // if other operations are trying to register dependences
        assert(outstanding_mapping_references > 0);
#endif
        // Check to see if we've already recorded this dependence
        std::map<Operation*,GenerationID>::const_iterator finder = 
          outgoing.find(op);
        if (finder == outgoing.end())
        {
          outgoing[op] = op_gen;
          // Record that the operation has a mapping dependence
          // on us as long as we haven't mapped
          tracker->add_mapping_dependence(mapped_event);
          tracker->add_resolution_dependence(resolved_event);
          // Record that we have a commit dependence on the
          // registering operation
#ifdef DEBUG_LEGION
          assert(dependence_tracker.commit != NULL);
#endif
          dependence_tracker.commit->add_commit_dependence(other_commit_event);
          registered_dependence = true;
        }
        else
        {
          // We already registered it
          registered_dependence = false;
        }
        // Cannot prune this operation from the list since it
        // is still not committed
        return false;
      }
      // We already committed so we're done and this
      // operation can be pruned from the list of users
      registered_dependence = false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool Operation::is_operation_committed(GenerationID our_gen)
    //--------------------------------------------------------------------------
    {
      // If we're on an old generation then it's definitely committed
      return (our_gen < gen);
    }

    //--------------------------------------------------------------------------
    void Operation::add_mapping_reference(GenerationID our_gen)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(our_gen <= gen); // better not be ahead of where we are now
#endif
      if (our_gen == gen)
        outstanding_mapping_references++;
    }

    //--------------------------------------------------------------------------
    void Operation::remove_mapping_reference(GenerationID our_gen)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(our_gen <= gen); // better not be ahead of where we are now
#endif
        if ((our_gen == gen) && !committed)
        {
#ifdef DEBUG_LEGION
          assert(outstanding_mapping_references > 0);
#endif
          outstanding_mapping_references--;
          // If we've completed and we have no mapping references
          // and we have no outstanding commit dependences then 
          // we can commit this operation
          if ((outstanding_mapping_references == 0) && !trigger_commit_invoked)
          {
#ifdef DEBUG_LEGION
            assert(dependence_tracker.commit != NULL);
#endif
            CommitDependenceTracker *tracker = dependence_tracker.commit;
            need_trigger = tracker->issue_commit_trigger(this, runtime);
            if (need_trigger)
              trigger_commit_invoked = true;
          }
        }
        // otherwise we were already recycled and are no longer valid
      }
      if (need_trigger)
        trigger_commit();
    }

    //--------------------------------------------------------------------------
    RtEvent Operation::invoke_state_analysis(void)
    //--------------------------------------------------------------------------
    {
      // First check to see if the parent context has remote state
      if ((parent_ctx != NULL) && parent_ctx->has_remote_state())
      {
        // This can be an expensive operation so defer it, but give it
        // a slight priority boost because we know that it will likely
        // involve some inter-node communication and we want to get 
        // that in flight quickly to help hide latency.
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        StateAnalysisArgs args;
        args.hlr_id = HLR_STATE_ANALYSIS_ID;
        args.proxy_op = this;
        args.ready_event = ready_event;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_STATE_ANALYSIS_ID, 
                                         HLR_LATENCY_PRIORITY, this);
        return ready_event;
      }
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void Operation::record_logical_dependence(const LogicalUser &user)
    //--------------------------------------------------------------------------
    {
      logical_records.push_back(user);
    }

    //--------------------------------------------------------------------------
    LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned&
                                            Operation::get_logical_records(void)
    //--------------------------------------------------------------------------
    {
      return logical_records;
    }

    //--------------------------------------------------------------------------
    void Operation::clear_logical_records(void)
    //--------------------------------------------------------------------------
    {
      logical_records.clear();
    }

    //--------------------------------------------------------------------------
    void Operation::notify_regions_verified(const std::set<unsigned> &regions,
                                            GenerationID our_gen)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(our_gen <= gen); // better not be ahead of where we are now
#endif
        if ((our_gen == gen) && !trigger_commit_invoked)
        {
          for (std::set<unsigned>::const_iterator it = regions.begin();
                it != regions.end(); it++)
          {
            unverified_regions.erase(*it);
          }
          if (hardened && unverified_regions.empty()
              && !trigger_commit_invoked)
          {
            need_trigger = true;
            trigger_commit_invoked = true;
          }
        }
      }
      if (need_trigger)
        trigger_commit();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::prepare_for_mapping(const InstanceRef &ref,
                                                   MappingInstance &instance) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!ref.is_composite_ref());
#endif
      instance = ref.get_mapping_instance();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::prepare_for_mapping(const InstanceSet &valid,
                                      std::vector<MappingInstance> &input_valid)
    //--------------------------------------------------------------------------
    {
      input_valid.resize(valid.size());
      for (unsigned idx = 0; idx < valid.size(); idx++)
      {
        const InstanceRef &ref = valid[idx];
#ifdef DEBUG_LEGION
        assert(!ref.is_composite_ref());
#endif
        MappingInstance &inst = input_valid[idx];
        inst = ref.get_mapping_instance();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::prepare_for_mapping(const InstanceSet &valid,
                                      const std::set<Memory> &visible_filter,
                                      std::vector<MappingInstance> &input_valid)
    //--------------------------------------------------------------------------
    {
      input_valid.reserve(valid.size());
      unsigned next_index = 0;
      for (unsigned idx = 0; idx < valid.size(); idx++)
      {
        const InstanceRef &ref = valid[idx];
#ifdef DEBUG_LEGION
        assert(!ref.is_composite_ref());
#endif
        if (visible_filter.find(ref.get_manager()->get_memory()) == 
            visible_filter.end())
          continue;
        input_valid.resize(next_index+1);
        MappingInstance &inst = input_valid[next_index++];
        inst = ref.get_mapping_instance();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::compute_ranking(
                              const std::deque<MappingInstance> &output,
                              const InstanceSet &sources,
                              std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      ranking.reserve(output.size());
      for (std::deque<MappingInstance>::const_iterator it = 
            output.begin(); it != output.end(); it++)
      {
        const PhysicalManager *manager = it->impl;
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          if (manager == sources[idx].get_manager())
          {
            ranking.push_back(idx);
            break;
          }
        }
        // Ignore any instances which are not in the original set of sources
      }
    }

    //--------------------------------------------------------------------------
    void Operation::MappingDependenceTracker::issue_stage_triggers(
                      Operation *op, Runtime *runtime, MustEpochOp *must_epoch)
    //--------------------------------------------------------------------------
    {
      bool map_now = true;
      bool resolve_now = true;
      if (!mapping_dependences.empty())
      {
        RtEvent map_precondition = Runtime::merge_events(mapping_dependences);
        if (!map_precondition.has_triggered())
        {
          if (must_epoch == NULL)
          {
            DeferredMappingArgs args;
            args.hlr_id = HLR_DEFERRED_MAPPING_TRIGGER_ID;
            args.proxy_this = op;
            runtime->issue_runtime_meta_task(&args, sizeof(args),
                                             HLR_DEFERRED_MAPPING_TRIGGER_ID,
                                             HLR_LATENCY_PRIORITY,
                                             op, map_precondition);
          }
          else
            must_epoch->add_mapping_dependence(map_precondition);  
          map_now = false;
        }
      }
      if (!resolution_dependences.empty())
      {
        RtEvent resolve_precondition = 
          Runtime::merge_events(resolution_dependences);
        if (!resolve_precondition.has_triggered())
        {
          DeferredResolutionArgs args;
          args.hlr_id = HLR_DEFERRED_RESOLUTION_TRIGGER_ID;
          args.proxy_this = op;
          runtime->issue_runtime_meta_task(&args, sizeof(args),
                                           HLR_DEFERRED_RESOLUTION_TRIGGER_ID,
                                           HLR_LATENCY_PRIORITY,
                                           op, resolve_precondition);
          resolve_now = false;
        }
      }
      if (map_now && (must_epoch == NULL))
        op->trigger_mapping();
      if (resolve_now)
        op->trigger_resolution();
    }
    
    //--------------------------------------------------------------------------
    bool Operation::CommitDependenceTracker::issue_commit_trigger(Operation *op,
                                                               Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      if (!commit_dependences.empty())
      {
        RtEvent commit_precondition = Runtime::merge_events(commit_dependences);
        if (!commit_precondition.has_triggered())
        {
          DeferredCommitTriggerArgs args;
          args.hlr_id = HLR_DEFERRED_COMMIT_TRIGGER_ID;
          args.proxy_this = op;
          args.gen = op->get_generation();
          runtime->issue_runtime_meta_task(&args, sizeof(args),
                                           HLR_DEFERRED_COMMIT_TRIGGER_ID,
                                           HLR_LATENCY_PRIORITY,
                                           op, commit_precondition);
          return false;
        }
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // Predicate Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PredicateImpl::PredicateImpl(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::activate_predicate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      predicate_resolved = false;
      predicate_references = 0;
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::deactivate_predicate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
#ifdef DEBUG_LEGION
      assert(predicate_references == 0);
#endif
      waiters.clear();
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::add_predicate_reference(void)
    //--------------------------------------------------------------------------
    {
      bool add_map_reference;
      {
        AutoLock o_lock(op_lock);
        add_map_reference = (predicate_references == 0);
        predicate_references++;
      }
      if (add_map_reference)
        add_mapping_reference(get_generation());
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::remove_predicate_reference(void)
    //--------------------------------------------------------------------------
    {
      bool need_trigger;
      bool remove_reference;
      GenerationID task_gen = 0;  // initialization to make gcc happy
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(predicate_references > 0);
#endif
        predicate_references--;
        remove_reference = (predicate_references == 0);
        if (remove_reference)
        {
          // Get the task generation before things can be cleaned up
          task_gen = get_generation();
          need_trigger = predicate_resolved;
        }
        else
          need_trigger = false;
      }
      if (need_trigger)
        complete_execution();
      if (remove_reference)
        remove_mapping_reference(task_gen);
    }

    //--------------------------------------------------------------------------
    bool PredicateImpl::register_waiter(PredicateWaiter *waiter,
                                          GenerationID waiter_gen, bool &value)
    //--------------------------------------------------------------------------
    {
      bool valid;
      AutoLock o_lock(op_lock);
      if (predicate_resolved)
      {
        value = predicate_value;
        valid = true;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(waiters.find(waiter) == waiters.end());
#endif
        waiters[waiter] = waiter_gen;
        valid = false;
      }
      return valid;
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::set_resolved_value(GenerationID pred_gen, bool value)
    //--------------------------------------------------------------------------
    {
      bool need_trigger;
      // Make a copy of the waiters since we could get cleaned up in parallel
      std::map<PredicateWaiter*,GenerationID> copy_waiters;
      {
        AutoLock o_lock(op_lock);
        if ((pred_gen == get_generation()) && !predicate_resolved)
        {
          predicate_resolved = true;
          predicate_value = value;
          copy_waiters = waiters;
          need_trigger = (predicate_references == 0);
        }
        else
          need_trigger= false;
      }
      // Notify any waiters, no need to hold the lock since waiters can't
      // be added after we set the state to resolved
      for (std::map<PredicateWaiter*,GenerationID>::const_iterator it = 
            copy_waiters.begin(); it != copy_waiters.end(); it++)
      {
        it->first->notify_predicate_value(it->second, value);
      }
      // Now see if we need to indicate we are done executing
      if (need_trigger)
        complete_execution();
    }

    /////////////////////////////////////////////////////////////
    // Speculative Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SpeculativeOp::SpeculativeOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void SpeculativeOp::activate_speculative(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      speculation_state = RESOLVE_TRUE_STATE;
      predicate = NULL;
      received_trigger_resolution = false;
      predicate_waiter = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void SpeculativeOp::deactivate_speculative(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
    }

    //--------------------------------------------------------------------------
    void SpeculativeOp::initialize_speculation(SingleTask *ctx, bool track,
                                               unsigned regions,
                                               const Predicate &p)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, track, regions);
      if (p == Predicate::TRUE_PRED)
      {
        speculation_state = RESOLVE_TRUE_STATE;
        predicate = NULL;
      }
      else if (p == Predicate::FALSE_PRED)
      {
        speculation_state = RESOLVE_FALSE_STATE;
        predicate = NULL;
      }
      else
      {
        speculation_state = PENDING_MAP_STATE;
        predicate = p.impl;
        predicate->add_predicate_reference();
      }
    }

    //--------------------------------------------------------------------------
    void SpeculativeOp::register_predicate_dependence(void)
    //--------------------------------------------------------------------------
    {
      if (predicate != NULL)
        register_dependence(predicate, predicate->get_generation());
    }

    //--------------------------------------------------------------------------
    bool SpeculativeOp::is_predicated(void) const
    //--------------------------------------------------------------------------
    {
      return (predicate != NULL);
    }

    //--------------------------------------------------------------------------
    bool SpeculativeOp::get_predicate_value(Processor proc)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_event = RtEvent::NO_RT_EVENT;
      // this is actually set on all paths, but the compiler can't see it
      bool result = false; 
      {
        AutoLock o_lock(op_lock);
        if (speculation_state == RESOLVE_TRUE_STATE)
          result = true;
        else if (speculation_state == RESOLVE_FALSE_STATE)
          result = false;
        else
        {
#ifdef DEBUG_LEGION
          assert(predicate != NULL);
#endif
          predicate_waiter = Runtime::create_rt_user_event();
          wait_event = predicate_waiter;
        }
      }
      if (wait_event.exists())
      {
        if (!wait_event.has_triggered())
          wait_event.wait();
        // Might be a little bit of a race here with cleanup
#ifdef DEBUG_LEGION
        assert((speculation_state == RESOLVE_TRUE_STATE) ||
               (speculation_state == RESOLVE_FALSE_STATE));
#endif
        if (speculation_state == RESOLVE_TRUE_STATE)
          result = true;
        else
          result = false;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void SpeculativeOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Quick out
      if (predicate == NULL)
      {
#ifdef DEBUG_LEGION
        assert((speculation_state == RESOLVE_TRUE_STATE) ||
               (speculation_state == RESOLVE_FALSE_STATE));
#endif
        if (speculation_state == RESOLVE_TRUE_STATE)
          resolve_true();
        else
          resolve_false();
        return;
      }
#ifdef DEBUG_LEGION
      assert(predicate != NULL);
#endif
      // Register ourselves as a waiter on the predicate value
      // If the predicate hasn't resolved yet, then we can ask the
      // mapper if it would like us to speculate on the value.
      // Then take the lock and set up our state.
      bool value, speculated = false;
      bool valid = predicate->register_waiter(this, get_generation(), value);
      // Now that we've attempted to register ourselves with the
      // predicate we can remove the predicate reference
      predicate->remove_predicate_reference();
      if (!valid)
        speculated = speculate(value);
      // Now hold the lock and figure out what we should do
      bool continue_true = false;
      bool continue_false = false;
      bool need_resolution = false;
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        switch (speculation_state)
        {
          case PENDING_MAP_STATE:
            {
              if (valid)
              {
                if (value)
                {
                  speculation_state = RESOLVE_TRUE_STATE;
                  continue_true = true;
                }
                else
                {
                  speculation_state = RESOLVE_FALSE_STATE;
                  continue_false = true;
                }
                need_resolution = received_trigger_resolution;
                need_trigger = predicate_waiter.exists();
              }
              else if (speculated)
              {
                if (value)
                {
                  speculation_state = SPECULATE_TRUE_STATE;
                  continue_true = true;
                }
                else
                {
                  speculation_state = SPECULATE_FALSE_STATE;
                  continue_false = true;
                }
              }
              // Otherwise just stay in pending map state
              break;
            }
          case RESOLVE_TRUE_STATE:
            {
              // Someone else has already resolved us to true and
              // triggered the appropriate method
              break;
            }
          case RESOLVE_FALSE_STATE:
            {
              // Someone else has already resolved us to false and
              // triggered the appropriate method
              break;
            }
          default:
            assert(false); // shouldn't be in the other states
        }
      }
      // Now do what we need to do
      if (need_trigger)
        Runtime::trigger_event(predicate_waiter);
      if (continue_true)
        resolve_true();
      if (continue_false)
        resolve_false();
      if (need_resolution)
        resolve_speculation(); 
    }

    //--------------------------------------------------------------------------
    void SpeculativeOp::trigger_resolution(void)
    //--------------------------------------------------------------------------
    {
      // Quick out
      if (predicate == NULL)
      {
        resolve_speculation();
        return;
      }
      bool need_trigger;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(!received_trigger_resolution);
#endif
        received_trigger_resolution = true;
        need_trigger = (speculation_state == RESOLVE_TRUE_STATE) ||
                        (speculation_state == RESOLVE_FALSE_STATE);
      }
      if (need_trigger)
        resolve_speculation();
    }

    //--------------------------------------------------------------------------
    void SpeculativeOp::notify_predicate_value(GenerationID pred_gen,bool value)
    //--------------------------------------------------------------------------
    {
      bool continue_true = false;
      bool continue_false = false;
      bool need_mispredict = false;
      bool restart = false;
      bool need_resolve = false;
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(pred_gen == get_generation());
#endif
        switch (speculation_state)
        {
          case PENDING_MAP_STATE:
            {
              if (value)
              {
                speculation_state = RESOLVE_TRUE_STATE;
                continue_true = true;
              }
              else
              {
                speculation_state = RESOLVE_FALSE_STATE;
                continue_false = true;
              }
              need_resolve = received_trigger_resolution;
              need_trigger = predicate_waiter.exists();
              break;
            }
          case SPECULATE_TRUE_STATE:
            {
              if (value) // We guessed right
              {
                speculation_state = RESOLVE_TRUE_STATE;
                need_resolve = received_trigger_resolution;
              }
              else
              {
                // We guessed wrong
                speculation_state = RESOLVE_FALSE_STATE;
                need_mispredict = true;
                restart = false;
              }
              break;
            }
          case SPECULATE_FALSE_STATE:
            {
              if (value)
              {
                speculation_state = RESOLVE_TRUE_STATE;
                need_mispredict = true;
                restart = true;
              }
              else
              {
                speculation_state = RESOLVE_FALSE_STATE;
                need_resolve = received_trigger_resolution;  
              }
              break;
            }
          default:
            assert(false); // shouldn't be in any of the other states
        }
      }
      if (need_trigger)
        Runtime::trigger_event(predicate_waiter);
      if (continue_true)
        resolve_true();
      if (continue_false)
        resolve_false();
      if (need_mispredict)
        quash_operation(get_generation(), restart);
      if (need_resolve)
        resolve_speculation();
    }

    //--------------------------------------------------------------------------
    void SpeculativeOp::deferred_execute(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    /////////////////////////////////////////////////////////////
    // Map Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MapOp::MapOp(Runtime *rt)
      : InlineMapping(), Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MapOp::MapOp(const MapOp &rhs)
      : InlineMapping(), Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MapOp::~MapOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MapOp& MapOp::operator=(const MapOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    PhysicalRegion MapOp::initialize(SingleTask *ctx, 
                                     const InlineLauncher &launcher,
                                     bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx;
      initialize_operation(ctx, true/*track*/);
      if (launcher.requirement.privilege_fields.empty())
      {
        log_task.warning("WARNING: REGION REQUIREMENT OF INLINE MAPPING "
                               "IN TASK %s (ID %lld) HAS NO PRIVILEGE "
                               "FIELDS! DID YOU FORGET THEM?!?",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id());
      }
      requirement = launcher.requirement;
      map_id = launcher.map_id;
      tag = launcher.tag;
      layout_constraint_id = launcher.layout_constraint_id;
      termination_event = Runtime::create_ap_user_event();
      region = PhysicalRegion(legion_new<PhysicalRegionImpl>(requirement,
                              completion_event, true/*mapped*/, ctx, 
                              map_id, tag, false/*leaf*/, runtime));
      if (check_privileges)
        check_privilege();
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_mapping_operation(parent_ctx->get_unique_id(),
                                         unique_op_id);
        LegionSpy::log_logical_requirement(unique_op_id,0/*index*/,
                                           true/*region*/,
                                           requirement.region.index_space.id,
                                           requirement.region.field_space.id,
                                           requirement.region.tree_id,
                                           requirement.privilege,
                                           requirement.prop,
                                           requirement.redop,
                                           requirement.parent.index_space.id);
        LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                          requirement.privilege_fields);
      }
      return region;
    }

    //--------------------------------------------------------------------------
    PhysicalRegion MapOp::initialize(SingleTask *ctx,
                                     const RegionRequirement &req,
                                     MapperID id, MappingTagID t,
                                     bool check_privileges)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      parent_task = ctx;
      requirement = req;
      if (requirement.privilege_fields.empty())
      {
        log_task.warning("WARNING: REGION REQUIREMENT OF INLINE MAPPING "
                               "IN TASK %s (ID %lld) HAS NO PRIVILEGE "
                               "FIELDS! DID YOU FORGET THEM?!?",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id());
      }
      requirement = req;
      map_id = id;
      tag = t;
      parent_task = ctx;
      termination_event = Runtime::create_ap_user_event();
      region = PhysicalRegion(legion_new<PhysicalRegionImpl>(requirement,
                              completion_event, true/*mapped*/, ctx, 
                              map_id, tag, false/*leaf*/, runtime));
      if (check_privileges)
        check_privilege();
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_mapping_operation(parent_ctx->get_unique_id(),
                                         unique_op_id);
        LegionSpy::log_logical_requirement(unique_op_id,0/*index*/,
                                           true/*region*/,
                                           requirement.region.index_space.id,
                                           requirement.region.field_space.id,
                                           requirement.region.tree_id,
                                           requirement.privilege,
                                           requirement.prop,
                                           requirement.redop,
                                           requirement.parent.index_space.id);
        LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                          requirement.privilege_fields);
      }
      return region;
    }

    //--------------------------------------------------------------------------
    void MapOp::initialize(SingleTask *ctx, const PhysicalRegion &reg)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      parent_task = ctx;
      requirement = reg.impl->get_requirement();
      map_id = reg.impl->map_id;
      tag = reg.impl->tag;
      parent_task = ctx;
      termination_event = Runtime::create_ap_user_event();
      region = reg;
      region.impl->remap_region(completion_event);
      // We're only really remapping it if it already had a physical
      // instance that we can use to make a valid value
      remap_region = region.impl->has_references();
      // No need to check the privileges here since we know that we have
      // them from the first time that we made this physical region
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_mapping_operation(parent_ctx->get_unique_id(), 
                                         unique_op_id);
        LegionSpy::log_logical_requirement(unique_op_id,0/*index*/,
                                           true/*region*/,
                                           requirement.region.index_space.id,
                                           requirement.region.field_space.id,
                                           requirement.region.tree_id,
                                           requirement.privilege,
                                           requirement.prop,
                                           requirement.redop,
                                           requirement.parent.index_space.id);
        LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                          requirement.privilege_fields);
      }
    }

    //--------------------------------------------------------------------------
    void MapOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      parent_ctx = NULL;
      remap_region = false;
      mapper = NULL;
      layout_constraint_id = 0;
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void MapOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      // Remove our reference to the region
      region = PhysicalRegion();
      privilege_path.clear();
      version_info.clear();
#ifdef DEBUG_LEGION
      assert(acquired_instances.empty());
#endif
      acquired_instances.clear();
      atomic_locks.clear();
      map_applied_conditions.clear();
      profiling_results = Mapper::InlineProfilingInfo();
      // Now return this operation to the queue
      runtime->free_map_op(this);
    } 

    //--------------------------------------------------------------------------
    const char* MapOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[MAP_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind MapOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return MAP_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t MapOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    Mappable* MapOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    void MapOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // First compute our parent region requirement
      compute_parent_index();  
      begin_dependence_analysis();
      RestrictInfo restrict_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   version_info,
                                                   restrict_info,
                                                   privilege_path);
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void MapOp::trigger_remote_state_analysis(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;  
      version_info.make_local(preconditions, this, runtime->forest);
      if (preconditions.empty())
        Runtime::trigger_event(ready_event);
      else
        Runtime::trigger_event(ready_event,
                               Runtime::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    bool MapOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      InstanceSet mapped_instances;
      // If we are restricted we know the answer
      if (requirement.is_restricted())
      {
        parent_ctx->get_physical_references(parent_req_index, mapped_instances);
        runtime->forest->traverse_and_register(physical_ctx, privilege_path,
                                               requirement, version_info, 
                                               this, 0/*idx*/, 
                                               termination_event, 
                                               false/*defer add users*/,
                                               map_applied_conditions,
                                               mapped_instances
#ifdef DEBUG_LEGION
                                               , get_logging_name()
                                               , unique_op_id
#endif
                                               );
      }
      // If we are remapping then we also know the answer
      else if (remap_region)
      {
        region.impl->get_references(mapped_instances);
        runtime->forest->traverse_and_register(physical_ctx, privilege_path,
                                               requirement, version_info,
                                               this, 0/*idx*/, 
                                               termination_event,
                                               false/*defer add users*/,
                                               map_applied_conditions,
                                               mapped_instances
#ifdef DEBUG_LEGION
                                               , get_logging_name()
                                               , unique_op_id
#endif
                                               );
      }
      else
      {
        // We're going to need to invoke the mapper, find the set of valid
        // instances as part of our traversal
        InstanceSet valid_instances;
        runtime->forest->physical_traverse_path(physical_ctx, privilege_path,
                                                requirement, version_info, 
                                                this, 0/*idx*/,
                                                true/*find valid*/,
                                                map_applied_conditions,
                                                valid_instances
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
        // Now we've got the valid instances so invoke the mapper
        invoke_mapper(valid_instances, mapped_instances);
        // Then we can register our mapped instances
        runtime->forest->physical_register_only(physical_ctx, requirement,
                                                version_info, this, 0/*idx*/,
                                                termination_event, 
                                                false/*defer add users*/,
                                                map_applied_conditions,
                                                mapped_instances
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
      }
#ifdef DEBUG_LEGION
      assert(!mapped_instances.empty());
#endif 
      // We're done so apply our mapping changes
      version_info.apply_mapping(physical_ctx.get_id(), 
                               runtime->address_space, map_applied_conditions);
      // Update our physical instance with the newly mapped instances
      // Have to do this before triggering the mapped event
      region.impl->reset_references(mapped_instances, termination_event);
      ApEvent map_complete_event = ApEvent::NO_AP_EVENT;
      if (mapped_instances.size() > 1)
      {
        std::set<ApEvent> mapped_events;
        for (unsigned idx = 0; idx < mapped_instances.size(); idx++)
          mapped_events.insert(mapped_instances[idx].get_ready_event());
        map_complete_event = Runtime::merge_events(mapped_events);
      }
      else
        map_complete_event = mapped_instances[0].get_ready_event();
      if (Runtime::legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, 0/*idx*/,
                                              requirement,
                                              mapped_instances);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, map_complete_event,
                                        termination_event);
#endif
      }
      // See if we have any reservations to take as part of this map
      if (!atomic_locks.empty())
      {
        // They've already been sorted in order 
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          map_complete_event = 
                Runtime::acquire_ap_reservation(it->first, it->second,
                                                map_complete_event);
          // We can also issue the release condition on our termination
          Runtime::release_reservation(it->first, termination_event);
        }
      }
      // Now we can trigger the mapping event and indicate
      // to all our mapping dependences that we are mapped.
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      
      if (!map_complete_event.has_triggered())
      {
        // Issue a deferred trigger on our completion event
        // and mark that we are no longer responsible for 
        // triggering our completion event
        Runtime::trigger_event(completion_event, map_complete_event);
        need_completion_trigger = false;
        DeferredExecuteArgs deferred_execute_args;
        deferred_execute_args.hlr_id = HLR_DEFERRED_EXECUTION_TRIGGER_ID;
        deferred_execute_args.proxy_this = this;
        runtime->issue_runtime_meta_task(&deferred_execute_args,
                                         sizeof(deferred_execute_args),
                                         HLR_DEFERRED_EXECUTION_TRIGGER_ID,
                                         HLR_LATENCY_PRIORITY, this, 
                               Runtime::protect_event(map_complete_event));
      }
      else
        deferred_execute();
      // return true since we succeeded
      return true;
    }

    //--------------------------------------------------------------------------
    void MapOp::deferred_execute(void)
    //--------------------------------------------------------------------------
    { 
      // Note that completing mapping and execution should
      // be enough to trigger the completion operation call
      // Trigger an early commit of this operation
      // Note that a mapping operation terminates as soon as it
      // is done mapping reflecting that after this happens, information
      // has flowed back out into the application task's execution.
      // Therefore mapping operations cannot be restarted because we
      // cannot track how the application task uses their data.
      // This means that any attempts to restart an inline mapping
      // will result in the entire task needing to be restarted.
      request_early_commit();
      // Mark that we are done executing
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void MapOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      // Don't commit this operation until we've reported our profiling
      commit_operation(true/*deactivate*/, profiling_reported); 
    }

    //--------------------------------------------------------------------------
    unsigned MapOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void MapOp::select_sources(const InstanceRef &target,
                               const InstanceSet &sources,
                               std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      Mapper::SelectInlineSrcInput input;
      Mapper::SelectInlineSrcOutput output;
      prepare_for_mapping(sources, input.source_instances); 
      prepare_for_mapping(target, input.target);
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_select_inline_sources(this, &input, &output);
      compute_ranking(output.chosen_ranking, sources, ranking);
    } 

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                                         MapOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void MapOp::update_atomic_locks(Reservation lock, bool exclusive)
    //--------------------------------------------------------------------------
    {
      std::map<Reservation,bool>::iterator finder = atomic_locks.find(lock);
      if (finder != atomic_locks.end())
      {
        if (!finder->second && exclusive)
          finder->second = true;
      }
      else
        atomic_locks[lock] = exclusive;
    }

    //--------------------------------------------------------------------------
    void MapOp::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    UniqueID MapOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    } 

    //--------------------------------------------------------------------------
    int MapOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    void MapOp::check_privilege(void)
    //--------------------------------------------------------------------------
    { 
      if ((requirement.handle_type == PART_PROJECTION) || 
          (requirement.handle_type == REG_PROJECTION))
      {
        log_region.error("Projection region requirements are not "
                               "permitted for inline mappings (in task %s)",
                               parent_ctx->get_task_name());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_BAD_PROJECTION_USE);
      }
      FieldID bad_field;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, then check the privileges with the parent context
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field);
      switch (et)
      {
        case NO_ERROR:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            log_region.error("Requirest for invalid region handle "
                                   "(%x,%d,%d) for inline mapping "
                                   "(ID %lld)",
                                   requirement.region.index_space.id, 
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id, 
                                   unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_REGION_HANDLE);
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) || 
                            (requirement.handle_type == REG_PROJECTION)
                             ? requirement.region.field_space : 
                               requirement.partition.field_space;
            log_region.error("Field %d is not a valid field of field "
                                   "space %d for inline mapping (ID %lld)",
                                   bad_field, sp.id, unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
          }
        case ERROR_INVALID_INSTANCE_FIELD:
          {
            log_region.error("Instance field %d is not one of the "
                                   "privilege fields for inline mapping "
                                   "(ID %lld)",
                                    bad_field, unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_INSTANCE_FIELD);
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            log_region.error("Instance field %d is a duplicate for "
                                    "inline mapping (ID %lld)",
                                    bad_field, unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_DUPLICATE_INSTANCE_FIELD);
          }
        case ERROR_BAD_PARENT_REGION:
          {
            log_region.error("Parent task %s (ID %lld) of inline mapping "
                                   "(ID %lld) does not have a region "
                                   "requirement for region (%x,%x,%x) "
                                   "as a parent of region requirement",
                                   parent_ctx->get_task_name(), 
                                   parent_ctx->get_unique_id(),
                                   unique_op_id, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_PARENT_REGION);
          }
        case ERROR_BAD_REGION_PATH:
          {
            log_region.error("Region (%x,%x,%x) is not a "
                             "sub-region of parent region "
   		             "(%x,%x,%x) for region requirement of inline "
                             "mapping (ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id, 
                             requirement.region.tree_id,
                             requirement.parent.index_space.id,
                             requirement.parent.field_space.id,
                             requirement.parent.tree_id,
                             unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_PATH);
          }
        case ERROR_BAD_REGION_TYPE:
          {
            log_region.error("Region requirement of inline mapping "
                                   "(ID %lld) cannot find privileges for field "
                                   "%d in parent task",
                                   unique_op_id, bad_field);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_TYPE);
          }
        case ERROR_BAD_REGION_PRIVILEGES:
          {
            log_region.error("Privileges %x for region " 
                                   "(%x,%x,%x) are not a subset of privileges "
                                   "of parent task's privileges for region "
                                   "requirement of inline mapping (ID %lld)",
                                   requirement.privilege, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id, 
                                   unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_PRIVILEGES);
          }
        // this should never happen with an inline mapping
        case ERROR_NON_DISJOINT_PARTITION: 
        default:
          assert(false); // Should never happen
      }
    }

    //--------------------------------------------------------------------------
    void MapOp::compute_parent_index(void)
    //--------------------------------------------------------------------------
    {
      int parent_index = parent_ctx->find_parent_region_req(requirement);
      if (parent_index < 0)
      {
        log_region.error("Parent task %s (ID %lld) of inline mapping "
                                   "(ID %lld) does not have a region "
                                   "requirement for region (%x,%x,%x) "
                                   "as a parent of region requirement.",
                                   parent_ctx->get_task_name(), 
                                   parent_ctx->get_unique_id(),
                                   unique_op_id, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id);
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_BAD_PARENT_REGION);
      }
      else
        parent_req_index = unsigned(parent_index);
    }

    //--------------------------------------------------------------------------
    void MapOp::invoke_mapper(const InstanceSet &valid_instances,
                                    InstanceSet &chosen_instances)
    //--------------------------------------------------------------------------
    {
      Mapper::MapInlineInput input;
      Mapper::MapInlineOutput output;
      if (!requirement.is_no_access())
      {
        std::set<Memory> visible_memories;
        runtime->find_visible_memories(parent_ctx->get_executing_processor(),
                                       visible_memories);
        prepare_for_mapping(valid_instances, visible_memories, 
                            input.valid_instances);
      }
      else
        prepare_for_mapping(valid_instances, input.valid_instances);
      // Invoke the mapper
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_map_inline(this, &input, &output);
      // Now we have to validate the output
      // Go through the instances and make sure we got one for every field
      // Also check to make sure that none of them are composite instances
      RegionTreeID bad_tree = 0;
      std::vector<FieldID> missing_fields;
      std::vector<PhysicalManager*> unacquired;
      int composite_index = runtime->forest->physical_convert_mapping(this,
                                requirement, output.chosen_instances, 
                                chosen_instances, bad_tree, missing_fields,
                                &acquired_instances, unacquired, 
                                !Runtime::unsafe_mapper);
      if (bad_tree > 0)
      {
        log_run.error("Invalid mapper output from invocation of 'map_inline' "
                      "on mapper %s. Mapper selected instance from region "
                      "tree %d to satisfy a region requirement for an inline "
                      "mapping in task %s (ID %lld) whose logical region is "
                      "from region tree %d.", mapper->get_mapper_name(),
                      bad_tree, parent_ctx->get_task_name(), 
                      parent_ctx->get_unique_id(), 
                      requirement.region.get_tree_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      if (!missing_fields.empty())
      {
        log_run.error("Invalid mapper output from invocation of 'map_inline' "
                      "on mapper %s. Mapper failed to specify a physical "
                      "instance for %ld fields of the region requirement to "
                      "an inline mapping in task %s (ID %lld). The missing "
                      "fields are listed below.", mapper->get_mapper_name(),
                      missing_fields.size(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id());
        for (std::vector<FieldID>::const_iterator it = missing_fields.begin();
              it != missing_fields.end(); it++)
        {
          const void *name; size_t name_size;
          runtime->retrieve_semantic_information(
              requirement.region.get_field_space(), *it, NAME_SEMANTIC_TAG,
              name, name_size, false, false);
          log_run.error("Missing instance for field %s (FieldID: %d)",
                        static_cast<const char*>(name), *it);
        }
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      if (!unacquired.empty())
      {
        for (std::vector<PhysicalManager*>::const_iterator it = 
              unacquired.begin(); it != unacquired.end(); it++)
        {
          if (acquired_instances.find(*it) == acquired_instances.end())
          {
            log_run.error("Invalid mapper output from 'map_inline' invocation "
                        "on mapper %s. Mapper selected physical instance for "
                        "inline mapping in task %s (ID %lld) which has already "
                        "been collected. If the mapper had properly acquired "
                        "this instance as part of the mapper call it would "
                        "have detected this. Please update the mapper to abide "
                        "by proper mapping conventions.", 
                        mapper->get_mapper_name(), parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
        }
        // If we did successfully acquire them, still issue the warning
        log_run.warning("WARNING: mapper %s faield to acquire instance "
                        "for inline mapping operation in task %s (ID %lld) "
                        "in 'map_inline' call. You may experience undefined "
                        "behavior as a consequence.", mapper->get_mapper_name(),
                        parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id());
      }
      if (composite_index >= 0)
      {
        log_run.error("Invalid mapper output from invocation of 'map_inline' "
                      "on mapper %s. Mapper requested creation of a composite "
                      "instance for inline mapping in task %s (ID %lld).",
                      mapper->get_mapper_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      } 
      // If we are doing unsafe mapping, then we can return
      if (Runtime::unsafe_mapper)
        return;
      // If this requirement doesn't have a no access flag then we
      // need to check to make sure that the instances are visible
      if (!requirement.is_no_access())
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        std::set<Memory> visible_memories;
        runtime->find_visible_memories(exec_proc, visible_memories);
        for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
        {
          Memory mem = chosen_instances[idx].get_memory();   
          if (visible_memories.find(mem) == visible_memories.end())
          {
            log_run.error("Invalid mapper output from invocation of "
                          "'map_inline' on mapper %s. Mapper selected a "
                          "physical instance in memory " IDFMT " which is "
                          "not visible from processor " IDFMT ". The inline "
                          "mapping operation was issued in task %s (ID %lld).",
                          mapper->get_mapper_name(), mem.id, exec_proc.id,
                          parent_ctx->get_task_name(), 
                          parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
        }
      }
      // Iterate over the instances and make sure they are all valid
      // for the given logical region which we are mapping
      std::vector<LogicalRegion> regions_to_check(1, requirement.region);
      for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
      {
        if (!chosen_instances[idx].get_manager()->meets_regions(
                                                        regions_to_check))
        {
          log_run.error("Invalid mapper output from invocation of 'map_inline' "
                        "on mapper %s. Mapper specified an instance that does "
                        "not meet the logical region requirement. The inline "
                        "mapping operation was issued in task %s (ID %lld).",
                        mapper->get_mapper_name(), parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_OUTPUT);
        }
      }
      // If this is a reduction region requirement, make sure all the
      // chosen instances are specialized reduction instances
      if (IS_REDUCE(requirement))
      {
        for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
        {
          if (!chosen_instances[idx].get_manager()->is_reduction_manager())
          {
            log_run.error("Invalid mapper output from invocation of "
                          "'map_inline' on mapper %s. Mapper failed to select "
                          "specialized reduction instances for region "
                          "requirement with reduction-only privileges for "
                          "inline mapping operation in task %s (ID %lld).",
                          mapper->get_mapper_name(),parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
          std::map<PhysicalManager*,std::pair<unsigned,bool> >::const_iterator 
            finder = acquired_instances.find(
                chosen_instances[idx].get_manager());
#ifdef DEBUG_LEGION
          assert(finder != acquired_instances.end());
#endif
          if (!finder->second.second)
          {
            log_run.error("Invalid mapper output from invocatino of "
                          "'map_inline' on mapper %s. Mapper made an illegal "
                          "decision to re-use a reduction instance for an "
                          "inline mapping in task %s (ID %lld). Reduction "
                          "instances are not currently permitted to be "
                          "recycled.", mapper->get_mapper_name(),
                          parent_ctx->get_task_name(), 
                          parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
        }
      }
      else
      {
        for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
        {
          if (!chosen_instances[idx].get_manager()->is_instance_manager())
          {
            log_run.error("Invalid mapper output from invocation of "
                          "'map_inline' on mapper %s. Mapper selected an "
                          "illegal specialized reduction instance for region "
                          "requirement without reduction privileges for "
                          "inline mapping operation in task %s (ID %lld).",
                          mapper->get_mapper_name(),parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
        }
      }
      if (layout_constraint_id > 0)
      {
        // Check the layout constraints are valid
        LayoutConstraints *constraints = 
          runtime->find_layout_constraints(layout_constraint_id);
        for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
        {
          PhysicalManager *manager = chosen_instances[idx].get_manager();
          if (manager->conflicts(constraints))
          {
            log_run.error("Invalid mapper output. Mapper %s selected "
                          "instance for inline mapping (ID %lld) in task %s "
                          "(ID %lld) which failed to satisfy the corresponding "
                          "layout constraints.", 
                          mapper->get_mapper_name(), get_unique_op_id(),
                          parent_ctx->get_task_name(), 
                          parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void MapOp::report_profiling_results(void)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_inline_report_profiling(this, &profiling_results);
#ifdef DEBUG_LEGION
      assert(profiling_reported.exists());
#endif
      // Trigger the event indicating we are done reporting profiling
      Runtime::trigger_event(profiling_reported);
    }

    /////////////////////////////////////////////////////////////
    // Copy Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyOp::CopyOp(Runtime *rt)
      : Copy(), SpeculativeOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CopyOp::CopyOp(const CopyOp &rhs)
      : Copy(), SpeculativeOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CopyOp::~CopyOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CopyOp& CopyOp::operator=(const CopyOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void CopyOp::initialize(SingleTask *ctx,
                            const CopyLauncher &launcher, bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx;
      initialize_speculation(ctx, true/*track*/, 
                             launcher.src_requirements.size() + 
                               launcher.dst_requirements.size(), 
                             launcher.predicate);
      src_requirements.resize(launcher.src_requirements.size());
      dst_requirements.resize(launcher.dst_requirements.size());
      src_versions.resize(launcher.src_requirements.size());
      dst_versions.resize(launcher.dst_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        if (launcher.src_requirements[idx].privilege_fields.empty())
        {
          log_task.warning("WARNING: SOURCE REGION REQUIREMENT %d OF "
                           "COPY (ID %lld) IN TASK %s (ID %lld) HAS NO "
                           "PRIVILEGE FIELDS! DID YOU FORGET THEM?!?",
                           idx, get_unique_op_id(),
                           parent_ctx->get_task_name(), 
                           parent_ctx->get_unique_id());
        }
        src_requirements[idx] = launcher.src_requirements[idx];
        src_requirements[idx].flags |= NO_ACCESS_FLAG;
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        if (launcher.src_requirements[idx].privilege_fields.empty())
        {
          log_task.warning("WARNING: DESTINATION REGION REQUIREMENT %d OF"
                           " COPY (ID %lld) IN TASK %s (ID %lld) HAS NO "
                           "PRIVILEGE FIELDS! DID YOU FORGET THEM?!?",
                           idx, get_unique_op_id(),
                           parent_ctx->get_task_name(), 
                           parent_ctx->get_unique_id());
        }
        dst_requirements[idx] = launcher.dst_requirements[idx];
        dst_requirements[idx].flags |= NO_ACCESS_FLAG;
        // If our privilege is not reduce, then shift it to write discard
        // since we are going to write all over the region
        if (dst_requirements[idx].privilege != REDUCE)
          dst_requirements[idx].privilege = WRITE_DISCARD;
      }
      grants = launcher.grants;
      // Register ourselves with all the grants
      for (unsigned idx = 0; idx < grants.size(); idx++)
        grants[idx].impl->register_operation(completion_event);
      wait_barriers = launcher.wait_barriers;
#ifdef LEGION_SPY
      for (std::vector<PhaseBarrier>::const_iterator it = 
            launcher.arrive_barriers.begin(); it != 
            launcher.arrive_barriers.end(); it++)
      {
        arrive_barriers.push_back(*it);
        LegionSpy::log_event_dependence(it->phase_barrier,
            arrive_barriers.back().phase_barrier);
      }
#else
      arrive_barriers = launcher.arrive_barriers;
#endif
      map_id = launcher.map_id;
      tag = launcher.tag;
      if (check_privileges)
      {
        if (src_requirements.size() != dst_requirements.size())
        {
          log_run.error("Number of source requirements (%ld) does not "
                        "match number of destination requirements (%ld) "
                        "for copy operation (ID %lld) with parent "
                        "task %s (ID %lld)",
                        src_requirements.size(), dst_requirements.size(),
                        get_unique_id(), parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_COPY_REQUIREMENTS_MISMATCH);
        }
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        {
          if (src_requirements[idx].privilege_fields.size() != 
              src_requirements[idx].instance_fields.size())
          {
            log_run.error("Copy source requirement %d for copy operation "
                          "(ID %lld) in parent task %s (ID %lld) has %ld "
                          "privilege fields and %ld instance fields.  "
                          "Copy requirements must have exactly the same "
                          "number of privilege and instance fields.",
                          idx, get_unique_id(), 
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id(),
                          src_requirements[idx].privilege_fields.size(),
                          src_requirements[idx].instance_fields.size());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_COPY_FIELDS_SIZE);
          }
          if (!IS_READ_ONLY(src_requirements[idx]))
          {
            log_run.error("Copy source requirement %d for copy operation "
                          "(ID %lld) in parent task %s (ID %lld) must "
                          "be requested with a read-only privilege.",
                          idx, get_unique_id(),
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_COPY_PRIVILEGE);
          }
          check_copy_privilege(src_requirements[idx], idx, true/*src*/);
        }
        for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
        {
          if (dst_requirements[idx].privilege_fields.size() != 
              dst_requirements[idx].instance_fields.size())
          {
            log_run.error("Copy destination requirement %d for copy "
                          "operation (ID %lld) in parent task %s "
                          "(ID %lld) has %ld privilege fields and %ld "
                          "instance fields.  Copy requirements must "
                          "have exactly the same number of privilege "
                          "and instance fields.", idx, 
                          get_unique_id(), 
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id(),
                          dst_requirements[idx].privilege_fields.size(),
                          dst_requirements[idx].instance_fields.size());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_COPY_FIELDS_SIZE);
          }
          if (!HAS_WRITE(dst_requirements[idx]))
          {
            log_run.error("Copy destination requirement %d for copy "
                          "operation (ID %lld) in parent task %s "
                          "(ID %lld) must be requested with a "
                          "read-write or write-discard privilege.",
                          idx, get_unique_id(),
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_COPY_PRIVILEGE);
          }
          check_copy_privilege(dst_requirements[idx], idx, false/*src*/);
        }
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        {
          IndexSpace src_space = src_requirements[idx].region.get_index_space();
          IndexSpace dst_space = dst_requirements[idx].region.get_index_space();
          if (!runtime->forest->are_compatible(src_space, dst_space))
          {
            log_run.error("Copy launcher index space mismatch at index "
                          "%d of cross-region copy (ID %lld) in task %s "
                          "(ID %lld). Source requirement with index "
                          "space %x and destination requirement "
                          "with index space %x do not have the "
                          "same number of dimensions or the same number "
                          "of elements in their element masks.",
                          idx, get_unique_id(),
                          parent_ctx->get_task_name(), 
                          parent_ctx->get_unique_id(),
                          src_space.id, dst_space.id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_COPY_SPACE_MISMATCH);
          }
          else if (!runtime->forest->is_dominated(src_space, dst_space))
          {
            log_run.error("Destination index space %x for "
                          "requirement %d of cross-region copy "
                          "(ID %lld) in task %s (ID %lld) is not "
                          "a sub-region of the source index space %x.", 
                          dst_space.id, idx, get_unique_id(),
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id(),
                          src_space.id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_COPY_SPACE_MISMATCH);
          }
        }
      }
      // Initialize the privilege and mapping paths for all of the
      // region requirements that we have
      src_privilege_paths.resize(src_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        initialize_privilege_path(src_privilege_paths[idx],
                                  src_requirements[idx]);
      }
      dst_privilege_paths.resize(dst_requirements.size());
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        initialize_privilege_path(dst_privilege_paths[idx],
                                  dst_requirements[idx]);
      }
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_copy_operation(parent_ctx->get_unique_id(),
                                      unique_op_id);
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        {
          const RegionRequirement &req = src_requirements[idx];
          LegionSpy::log_logical_requirement(unique_op_id, idx, true/*region*/,
                                             req.region.index_space.id,
                                             req.region.field_space.id,
                                             req.region.tree_id,
                                             req.privilege,
                                             req.prop, req.redop,
                                             req.parent.index_space.id);
          LegionSpy::log_requirement_fields(unique_op_id, idx, 
                                            req.instance_fields);
        }
        for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
        {
          const RegionRequirement &req = dst_requirements[idx];
          LegionSpy::log_logical_requirement(unique_op_id, 
                                             src_requirements.size()+idx, 
                                             true/*region*/,
                                             req.region.index_space.id,
                                             req.region.field_space.id,
                                             req.region.tree_id,
                                             req.privilege,
                                             req.prop, req.redop,
                                             req.parent.index_space.id);
          LegionSpy::log_requirement_fields(unique_op_id, 
                                            src_requirements.size()+idx, 
                                            req.instance_fields);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative();
      mapper = NULL;
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void CopyOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_speculative();
      // Clear out our region tree state
      src_requirements.clear();
      dst_requirements.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      src_privilege_paths.clear();
      dst_privilege_paths.clear();
      src_parent_indexes.clear();
      dst_parent_indexes.clear();
      src_versions.clear();
      dst_versions.clear();
#ifdef DEBUG_LEGION
      assert(acquired_instances.empty());
#endif
      acquired_instances.clear();
      atomic_locks.clear();
      map_applied_conditions.clear();
      profiling_results = Mapper::CopyProfilingInfo();
      // Return this operation to the runtime
      runtime->free_copy_op(this);
    }

    //--------------------------------------------------------------------------
    const char* CopyOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[COPY_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind CopyOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return COPY_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t CopyOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return src_requirements.size() + dst_requirements.size();
    }

    //--------------------------------------------------------------------------
    Mappable* CopyOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent indexes
      compute_parent_indexes(); 
      begin_dependence_analysis();
      // Register a dependence on our predicate
      register_predicate_dependence();
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        RestrictInfo restrict_info;
        runtime->forest->perform_dependence_analysis(this, idx, 
                                                     src_requirements[idx],
                                                     src_versions[idx],
                                                     restrict_info,
                                                     src_privilege_paths[idx]);
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        unsigned index = src_requirements.size()+idx;
        RestrictInfo restrict_info;
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        if (is_reduce_req)
          dst_requirements[idx].privilege = READ_WRITE;
        runtime->forest->perform_dependence_analysis(this, index, 
                                                     dst_requirements[idx],
                                                     dst_versions[idx],
                                                     restrict_info,
                                                     dst_privilege_paths[idx]);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = REDUCE;
      }
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_remote_state_analysis(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      for (unsigned idx = 0; idx < src_versions.size(); idx++)
        src_versions[idx].make_local(preconditions, this, runtime->forest); 
      for (unsigned idx = 0; idx < dst_versions.size(); idx++)
        dst_versions[idx].make_local(preconditions, this, runtime->forest); 
      if (preconditions.empty())
        Runtime::trigger_event(ready_event);
      else
        Runtime::trigger_event(ready_event,
                               Runtime::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    void CopyOp::resolve_true(void)
    //--------------------------------------------------------------------------
    {
      // Put this on the queue of stuff to do
      runtime->add_to_local_queue(parent_ctx->get_executing_processor(),
                                  this, false/*prev fail*/);
    }

    //--------------------------------------------------------------------------
    void CopyOp::resolve_false(void)
    //--------------------------------------------------------------------------
    {
      // Mark that this operation has completed both
      // execution and mapping indicating that we are done
      // Do it in this order to avoid calling 'execute_trigger'
      complete_execution();
      assert(0 && "TODO: advance mapping states if you care");
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    bool CopyOp::speculate(bool &value)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      Mapper::SpeculativeOutput output;
      output.speculate = false;
      mapper->invoke_copy_speculate(this, &output);
      if (output.speculate)
      {
        value = output.speculative_value;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool CopyOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      std::vector<RegionTreeContext> src_contexts(src_requirements.size());
      std::vector<RegionTreeContext> dst_contexts(dst_requirements.size());
      std::vector<InstanceSet> valid_src_instances(src_requirements.size());
      std::vector<InstanceSet> valid_dst_instances(dst_requirements.size());
      Mapper::MapCopyInput input;
      Mapper::MapCopyOutput output;
      input.src_instances.resize(src_requirements.size());
      input.dst_instances.resize(dst_requirements.size());
      output.src_instances.resize(src_requirements.size());
      output.dst_instances.resize(dst_requirements.size());
      // First go through and do the traversals to find the valid instances
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        src_contexts[idx] = parent_ctx->find_enclosing_context(
                                                    src_parent_indexes[idx]);
        // No need to find the valid instances if this is restricted
        if (src_requirements[idx].is_restricted())
          continue;
        InstanceSet &valid_instances = valid_src_instances[idx];
        runtime->forest->physical_traverse_path(src_contexts[idx],
                                                src_privilege_paths[idx],
                                                src_requirements[idx],
                                                src_versions[idx],
                                                this, idx, true/*find valid*/,
                                                map_applied_conditions,
                                                valid_instances
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
        // Convert these to the valid set of mapping instances
        // No need to filter for copies
        prepare_for_mapping(valid_instances, input.src_instances[idx]);
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        dst_contexts[idx] = parent_ctx->find_enclosing_context(
                                                    dst_parent_indexes[idx]);
        // No need to find the valid instances if this is restricted
        if (dst_requirements[idx].is_restricted())
          continue;
        InstanceSet &valid_instances = valid_dst_instances[idx];
        runtime->forest->physical_traverse_path(dst_contexts[idx],
                                                dst_privilege_paths[idx],
                                                dst_requirements[idx],
                                                dst_versions[idx],
                                                this, idx, true/*find valid*/,
                                                map_applied_conditions,
                                                valid_instances
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
        // No need to filter for copies
        prepare_for_mapping(valid_instances, input.dst_instances[idx]);
      }
      // Now we can ask the mapper what to do
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_map_copy(this, &input, &output);
      // Now we can carry out the mapping requested by the mapper
      // and issue the across copies, first set up the sync precondition
      ApEvent sync_precondition = ApEvent::NO_AP_EVENT;
      if (!wait_barriers.empty() || !grants.empty())
      {
        std::set<ApEvent> preconditions;
        for (std::vector<PhaseBarrier>::const_iterator it = 
              wait_barriers.begin(); it != wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(*it); 
          preconditions.insert(e);
        }
        for (std::vector<Grant>::const_iterator it = grants.begin();
              it != grants.end(); it++)
        {
          ApEvent e = it->impl->acquire_grant();
          preconditions.insert(e);
        }
        sync_precondition = Runtime::merge_events(preconditions);
      }
      // Register the source and destination regions
      std::set<ApEvent> copy_complete_events;
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        InstanceSet src_targets, dst_targets;
        // The common case 
        int src_composite = -1;
        // Make a user event for when this copy across is done
        // and add it to the set of copy complete events
        ApUserEvent local_completion = Runtime::create_ap_user_event();
        copy_complete_events.insert(local_completion);
        if (!src_requirements[idx].is_restricted())
        {
          // Do the conversion and check for errors
          src_composite = 
            perform_conversion<true/*src*/>(idx, src_requirements[idx],
                                            output.src_instances[idx],
                                            src_targets,
                                            IS_REDUCE(dst_requirements[idx]));
          if (Runtime::legion_spy_enabled)
            runtime->forest->log_mapping_decision(unique_op_id, idx, 
                                                  src_requirements[idx],
                                                  src_targets);
          // If we have a compsite reference, we need to map it
          // as a virtual region
          if (src_composite >= 0)
          {
            // No need to do the registration here
            runtime->forest->map_virtual_region(src_contexts[idx],
                                                src_requirements[idx],
                                                src_targets[src_composite],
                                                src_versions[idx],
                                                parent_ctx, this,
                                                false/*needs fields*/
#ifdef DEBUG_LEGION
                                                , idx, get_logging_name()
                                                , unique_op_id
#endif
                                                );
          }
          else
          {
            // Now do the registration
            set_mapping_state(idx, true/*src*/);
            runtime->forest->physical_register_only(src_contexts[idx],
                                                    src_requirements[idx],
                                                    src_versions[idx],
                                                    this, idx, local_completion,
                                                    false/*defer add users*/,
                                                    map_applied_conditions,
                                                    src_targets
#ifdef DEBUG_LEGION
                                                    , get_logging_name()
                                                    , unique_op_id
#endif
                                                    );
          }
        }
        else
        {
          // Restricted case, get the instances from the parent
          parent_ctx->get_physical_references(src_parent_indexes[idx],
                                              src_targets);
          if (Runtime::legion_spy_enabled)
            runtime->forest->log_mapping_decision(unique_op_id, idx, 
                                                  src_requirements[idx],
                                                  src_targets);
          set_mapping_state(idx, true/*src*/);
          runtime->forest->traverse_and_register(src_contexts[idx],
                                                 src_privilege_paths[idx],
                                                 src_requirements[idx],
                                                 src_versions[idx], this, idx,
                                                 local_completion, 
                                                 false/*defer add users*/,
                                                 map_applied_conditions,
                                                 src_targets
#ifdef DEBUG_LEGION
                                                 , get_logging_name()
                                                 , unique_op_id
#endif
                                                 );
        }
        // Little bit of a hack here, if we are going to do a reduction
        // explicit copy, switch the privileges to read-write when doing
        // the registration since we know we are using normal instances
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        if (is_reduce_req)
          dst_requirements[idx].privilege = READ_WRITE;
        // Common case is not restricted
        if (!dst_requirements[idx].is_restricted())
        {
          perform_conversion<false/*src*/>(idx, dst_requirements[idx],
                                           output.dst_instances[idx],
                                           dst_targets);
          // Now do the registration
          set_mapping_state(idx, false/*src*/);
          runtime->forest->physical_register_only(dst_contexts[idx],
                                                  dst_requirements[idx],
                                                  dst_versions[idx],
                                                  this, idx, local_completion,
                                                  false/*defer add users*/,
                                                  map_applied_conditions,
                                                  dst_targets
#ifdef DEBUG_LEGION
                                                  , get_logging_name()
                                                  , unique_op_id
#endif
                                                  );
        }
        else
        {
          // Restricted case, get the instances from the parent
          parent_ctx->get_physical_references(dst_parent_indexes[idx],
                                              dst_targets);
          set_mapping_state(idx, false/*src*/);
          runtime->forest->traverse_and_register(dst_contexts[idx],
                                                 dst_privilege_paths[idx],
                                                 dst_requirements[idx],
                                                 dst_versions[idx], this, idx,
                                                 local_completion, 
                                                 false/*defer add users*/, 
                                                 map_applied_conditions,
                                                 dst_targets
#ifdef DEBUG_LEGION
                                                 , get_logging_name()
                                                 , unique_op_id
#endif
                                                 );
        }
        if (Runtime::legion_spy_enabled)
          runtime->forest->log_mapping_decision(unique_op_id, 
             idx + src_requirements.size(), dst_requirements[idx], dst_targets);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = REDUCE;
        ApEvent local_sync_precondition = sync_precondition;
        // See if we have any atomic locks we have to acquire
        if ((idx < atomic_locks.size()) && !atomic_locks[idx].empty())
        {
          // Issue the acquires and releases for the reservations
          // necessary for performing this across operation
          const std::map<Reservation,bool> &local_locks = atomic_locks[idx];
          for (std::map<Reservation,bool>::const_iterator it = 
                local_locks.begin(); it != local_locks.end(); it++)
          {
            local_sync_precondition = 
              Runtime::acquire_ap_reservation(it->first, it->second,
                                              local_sync_precondition);
            // We can also issue the release here too
            Runtime::release_reservation(it->first, local_completion);
          }
        }
        // If we made it here, we passed all our error-checking so
        // now we can issue the copy/reduce across operation
        // Trigger our local completion event contingent upon 
        // the copy/reduce across being done
        if (!IS_REDUCE(dst_requirements[idx]))
        {
          ApEvent across_done = 
            runtime->forest->copy_across(src_contexts[idx], dst_contexts[idx], 
                                  src_requirements[idx], dst_requirements[idx],
                                  src_targets, dst_targets, src_versions[idx], 
                                  src_composite, this, idx, 
                                  local_sync_precondition, 
                                  map_applied_conditions);
          Runtime::trigger_event(local_completion, across_done);
        }
        else
        {
          // Composite instances are not valid sources for reductions across
#ifdef DEBUG_LEGION
          assert(src_composite == -1);
#endif
          ApEvent across_done = 
            runtime->forest->reduce_across(src_contexts[idx], dst_contexts[idx],
                                  src_requirements[idx], dst_requirements[idx],
                                  src_targets, dst_targets, src_versions[idx], 
                                  this, local_sync_precondition);
          Runtime::trigger_event(local_completion, across_done);
        }
        // Apply our changes to the version states
        src_versions[idx].apply_mapping(src_contexts[idx].get_id(),
                            runtime->address_space, map_applied_conditions);
        dst_versions[idx].apply_mapping(dst_contexts[idx].get_id(),
                            runtime->address_space, map_applied_conditions); 
      }
      ApEvent copy_complete_event = Runtime::merge_events(copy_complete_events);
#ifdef LEGION_SPY
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_operation_events(unique_op_id, copy_complete_event,
                                        completion_event);
#endif
      // Chain all the unlock and barrier arrivals off of the
      // copy complete event
      if (!arrive_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::iterator it = 
              arrive_barriers.begin(); it != arrive_barriers.end(); it++)
        {
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        completion_event);    
        }
      }
      // Mark that we completed mapping
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      // Handle the case for marking when the copy completes
      Runtime::trigger_event(completion_event, copy_complete_event);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(copy_complete_event));
      // We succeeded mapping
      return true;
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      for (std::vector<VersionInfo>::iterator it = src_versions.begin();
            it != src_versions.end(); it++)
      {
        it->release();
      }
      for (std::vector<VersionInfo>::iterator it = dst_versions.begin();
            it != dst_versions.end(); it++)
      {
        it->release();
      }
      // Don't commit this operation until we've reported our profiling
      commit_operation(true/*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    void CopyOp::report_interfering_requirements(unsigned idx1, unsigned idx2)
    //--------------------------------------------------------------------------
    {
      bool is_src1 = idx1 < src_requirements.size();
      bool is_src2 = idx2 < src_requirements.size();
      unsigned actual_idx1 = is_src1 ? idx1 : (idx1 - src_requirements.size());
      unsigned actual_idx2 = is_src2 ? idx2 : (idx2 - src_requirements.size());
      log_run.error("Aliased region requirements for copy operations "
                    "are not permitted. Region requirement %d of %s "
                    "requirements and %d of %s requirements interfering for "
                    "copy operation (UID %lld) in task %s (UID %lld).",
                    actual_idx1, is_src1 ? "source" : "destination",
                    actual_idx2, is_src2 ? "source" : "destination",
                    unique_op_id, parent_ctx->get_task_name(),
                    parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_ALIASED_REGION_REQUIREMENTS);
    }

    //--------------------------------------------------------------------------
    void CopyOp::report_interfering_close_requirement(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here, we can skip these since it won't impact anything
    }

    //--------------------------------------------------------------------------
    unsigned CopyOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      if (idx >= src_parent_indexes.size())
      {
        idx -= src_parent_indexes.size();
#ifdef DEBUG_LEGION
        assert(idx < dst_parent_indexes.size());
#endif
        return dst_parent_indexes[idx];
      }
      else
        return src_parent_indexes[idx];
    }

    //--------------------------------------------------------------------------
    void CopyOp::select_sources(const InstanceRef &target,
                                const InstanceSet &sources,
                                std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      Mapper::SelectCopySrcInput input;
      Mapper::SelectCopySrcOutput output;
      prepare_for_mapping(sources, input.source_instances);
      prepare_for_mapping(target, input.target);
      input.is_src = current_src;
      input.region_req_index = current_index;
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_select_copy_sources(this, &input, &output);
      // Fill in the ranking based on the output
      compute_ranking(output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                                        CopyOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void CopyOp::update_atomic_locks(Reservation lock, bool exclusive)
    //--------------------------------------------------------------------------
    {
      // We should only be doing analysis on one region requirement
      // at a time so we don't need to hold the operation lock when 
      // updating this data structure
      if (current_index >= atomic_locks.size())
        atomic_locks.resize(current_index+1);
      std::map<Reservation,bool> &local_locks = atomic_locks[current_index];
      std::map<Reservation,bool>::iterator finder = local_locks.find(lock);
      if (finder != local_locks.end())
      {
        if (!finder->second && exclusive)
          finder->second = true;
      }
      else
        local_locks[lock] = exclusive;
    }

    //--------------------------------------------------------------------------
    void CopyOp::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    UniqueID CopyOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id; 
    }

    //--------------------------------------------------------------------------
    int CopyOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    void CopyOp::check_copy_privilege(const RegionRequirement &requirement, 
                                      unsigned idx, bool src)
    //--------------------------------------------------------------------------
    {
      if ((requirement.handle_type == PART_PROJECTION) ||
          (requirement.handle_type == REG_PROJECTION))
      {
        log_region.error("Projection region requirements are not "
                               "permitted for copy operations (in task %s)",
                               parent_ctx->get_task_name());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_BAD_PROJECTION_USE);
      }
      FieldID bad_field;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, then check the privileges with the parent context
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field);
      switch (et)
      {
        case NO_ERROR:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            log_region.error("Requirest for invalid region handle "
                                   "(%x,%d,%d) for index %d of %s "
                                   "requirements of copy operation (ID %lld)",
                                   requirement.region.index_space.id, 
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id, 
                                   idx, (src ? "source" : "destination"),
                                   unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_REGION_HANDLE);
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) || 
                            (requirement.handle_type == REG_PROJECTION)
                             ? requirement.region.field_space : 
                               requirement.partition.field_space;
            log_region.error("Field %d is not a valid field of field "
                                   "space %d for index %d of %s requirements "
                                   "of copy operation (ID %lld)",
                                   bad_field, sp.id, idx, 
                                   (src ? "source" : "destination"),
                                   unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
          }
        case ERROR_INVALID_INSTANCE_FIELD:
          {
            log_region.error("Instance field %d is not one of the "
                                   "privilege fields for index %d of %s "
                                   "requirements of copy operation (ID %lld)",
                                    bad_field, idx, 
                                    (src ? "source" : "destination"),
                                    unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_INSTANCE_FIELD);
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            log_region.error("Instance field %d is a duplicate for "
                                    "index %d of %s requirements of copy "
                                    "operation (ID %lld)",
                                    bad_field, idx,
                                    (src ? "source" : "destination"),
                                    unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_DUPLICATE_INSTANCE_FIELD);
          }
        case ERROR_BAD_PARENT_REGION:
          {
            log_region.error("Parent task %s (ID %lld) of copy operation "
                                   "(ID %lld) does not have a region "
                                   "requirement for region (%x,%x,%x) "
                                   "as a parent of index %d of %s region "
                                   "requirements",
                                   parent_ctx->get_task_name(), 
                                   parent_ctx->get_unique_id(),
                                   unique_op_id, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id,
                                   idx, (src ? "source" : "destination"));
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_PARENT_REGION);
          }
        case ERROR_BAD_REGION_PATH:
          {
            log_region.error("Region (%x,%x,%x) is not a "
                                   "sub-region of parent region "
                                   "(%x,%x,%x) for index %d of "
                                   "%s region requirements of copy "
                                   "operation (ID %lld)",
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id,
                                   requirement.parent.index_space.id,
                                   requirement.parent.field_space.id,
                                   requirement.parent.tree_id,
                                   idx, (src ? "source" : "destination"),
                                   unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_PATH);
          }
        case ERROR_BAD_REGION_TYPE:
          {
            log_region.error("Region requirement of copy operation "
                                   "(ID %lld) cannot find privileges for field "
                                   "%d in parent task from index %d of %s "
                                   "region requirements",
                                   unique_op_id, bad_field, idx,
                                   (src ? "source" : "destination"));
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_TYPE);
          }
        case ERROR_BAD_REGION_PRIVILEGES:
          {
            log_region.error("Privileges %x for region (%x,%x,%x) are "
                                   "not a subset of privileges of parent "
                                   "task's privileges for index %d of %s "
                                   "region requirements for copy "
                                   "operation (ID %lld)",
                                   requirement.privilege, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id, 
                                   idx, (src ? "source" : "destination"),
                                   unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_PRIVILEGES);
          }
        // this should never happen with an inline mapping
        case ERROR_NON_DISJOINT_PARTITION: 
        default:
          assert(false); // Should never happen
      }
    }

    //--------------------------------------------------------------------------
    void CopyOp::compute_parent_indexes(void)
    //--------------------------------------------------------------------------
    {
      src_parent_indexes.resize(src_requirements.size());
      dst_parent_indexes.resize(dst_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        int parent_index = 
          parent_ctx->find_parent_region_req(src_requirements[idx]);
        if (parent_index < 0)
        {
          log_region.error("Parent task %s (ID %lld) of copy operation "
                                   "(ID %lld) does not have a region "
                                   "requirement for region (%x,%x,%x) "
                                   "as a parent of index %d of source region "
                                   "requirements",
                                   parent_ctx->get_task_name(), 
                                   parent_ctx->get_unique_id(),
                                   unique_op_id, 
                                   src_requirements[idx].region.index_space.id,
                                   src_requirements[idx].region.field_space.id, 
                                   src_requirements[idx].region.tree_id, idx);
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_BAD_PARENT_REGION);
        }
        else
          src_parent_indexes[idx] = unsigned(parent_index);
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        int parent_index = 
          parent_ctx->find_parent_region_req(dst_requirements[idx]);
        if (parent_index < 0)
        {
          log_region.error("Parent task %s (ID %lld) of copy operation "
                                   "(ID %lld) does not have a region "
                                   "requirement for region (%x,%x,%x) "
                                   "as a parent of index %d of destination "
                                   "region requirements",
                                   parent_ctx->get_task_name(), 
                                   parent_ctx->get_unique_id(),
                                   unique_op_id, 
                                   dst_requirements[idx].region.index_space.id,
                                   dst_requirements[idx].region.field_space.id, 
                                   dst_requirements[idx].region.tree_id, idx);
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_BAD_PARENT_REGION);
        }
        else
          dst_parent_indexes[idx] = unsigned(parent_index);
      }
    }

    //--------------------------------------------------------------------------
    template<bool IS_SRC>
    int CopyOp::perform_conversion(unsigned idx, const RegionRequirement &req,
                                   std::vector<MappingInstance> &output,
                                   InstanceSet &targets, bool is_reduce)
    //--------------------------------------------------------------------------
    {
      RegionTreeID bad_tree = 0;
      std::vector<FieldID> missing_fields;
      std::vector<PhysicalManager*> unacquired;
      int composite_idx = runtime->forest->physical_convert_mapping(this,
                              req, output, targets, bad_tree, missing_fields,
                              &acquired_instances, unacquired, 
                              !Runtime::unsafe_mapper);
      if (bad_tree > 0)
      {
        log_run.error("Invalid mapper output from invocation of 'map_copy' "
                      "on mapper %s. Mapper selected an instance from "
                      "region tree %d to satisfy %s region requirement %d "
                      "for explicit region-to_region copy in task %s (ID %lld) "
                      "but the logical region for this requirement is from "
                      "region tree %d.", mapper->get_mapper_name(), bad_tree,
                      IS_SRC ? "source" : "destination", idx, 
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id(),
                      req.region.get_tree_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      if (!missing_fields.empty())
      {
        log_run.error("Invalid mapper output from invocation of 'map_copy' "
                      "on mapper %s. Mapper failed to specify a physical "
                      "instance for %ld fields of the region requirement %d "
                      "of explicit region-to-region copy in task %s (ID %lld). "
                      "Ths missing fields are listed below.",
                      mapper->get_mapper_name(), missing_fields.size(), idx,
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());

        for (std::vector<FieldID>::const_iterator it = missing_fields.begin();
              it != missing_fields.end(); it++)
        {
          const void *name; size_t name_size;
          runtime->retrieve_semantic_information(
              req.region.get_field_space(), *it, NAME_SEMANTIC_TAG,
              name, name_size, false, false);
          log_run.error("Missing instance for field %s (FieldID: %d)",
                        static_cast<const char*>(name), *it);
        }
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      if (!unacquired.empty())
      {
        for (std::vector<PhysicalManager*>::const_iterator it = 
              unacquired.begin(); it != unacquired.end(); it++)
        {
          if (acquired_instances.find(*it) == acquired_instances.end())
          {
            log_run.error("Invalid mapper output from 'map_copy' invocation "
                          "on mapper %s. Mapper selected physical instance "
                          "for %s region requirement %d of explicit region-to-"
                          "region copy in task %s (ID %lld) which has already "
                          "been collected. If the mapper had properly acquired "
                          "this instance as part of the mapper call it would "
                          "have detected this. Please update the mapper to "
                          "abide by proper mapping conventions.",
                          mapper->get_mapper_name(), 
                          IS_SRC ? "source" : "destination", idx,
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
        }
        // If we did successfully acquire them, still issue the warning
        log_run.warning("WARNING: mapper %s failed to acquire instances "
                        "for %s region requirement %d of explicit region-to-"
                        "region copy in task %s (ID %lld) in 'map_copy' call. "
                        "You may experience undefined behavior as a "
                        "consequence.", mapper->get_mapper_name(),
                        IS_SRC ? "source" : "destination", idx,
                        parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id());
      }
      // Destination is not allowed to have composite instances
      if (!IS_SRC && (composite_idx >= 0))
      {
        log_run.error("Invalid mapper output from invocation of 'map_copy' "
                      "on mapper %s. Mapper requested the creation of a "
                      "composite instance for destination region requiremnt "
                      "%d. Only source region requirements are permitted to "
                      "be composite instances for explicit region-to-region "
                      "copy operations. Operation was issued in task %s "
                      "(ID %lld).", mapper->get_mapper_name(), idx,
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT); 
      } 
      if (IS_SRC && (composite_idx >= 0) && is_reduce)
      {
        log_run.error("Invalid mapper output from invocation of 'map_copy' "
                      "on mapper %s. Mapper requested the creation of a "
                      "composite instance for the source requirement %d of "
                      "an explicit region-to-region reduction. Only real "
                      "physical instances are permitted to be sources of "
                      "explicit region-to-region reductions. Operation was "
                      "issued in task %s (ID %lld).", mapper->get_mapper_name(),
                      idx, parent_ctx->get_task_name(), 
                      parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      if (Runtime::unsafe_mapper)
        return composite_idx;
      std::vector<LogicalRegion> regions_to_check(1, req.region);
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        const InstanceRef &ref = targets[idx];
        PhysicalManager *manager = ref.get_manager();
        if (manager->is_virtual_instance())
          continue;
        if (!manager->meets_regions(regions_to_check))
        {
          log_run.error("Invalid mapper output from invocation of 'map_copy' "
                        "on mapper %s. Mapper specified an instance for %s "
                        "region requirement at index %d that does not meet "
                        "the logical region requirement. The copy operation "
                        "was issued in task %s (ID %lld).",
                        mapper->get_mapper_name(), 
                        IS_SRC ? "source" : "destination", idx,
                        parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_OUTPUT);
        }
      }
      // Make sure all the destinations are real instances, this has
      // to be true for all kinds of explicit copies including reductions
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        if (IS_SRC && (int(idx) == composite_idx))
          continue;
        if (!targets[idx].get_manager()->is_instance_manager())
        {
          log_run.error("Invalid mapper output from invocation of 'map_copy' "
                        "on mapper %s. Mapper specified an illegal "
                        "specialized instance as the target for %s "
                        "region requirement %d of an explicit copy operation "
                        "in task %s (ID %lld).", mapper->get_mapper_name(),
                        IS_SRC ? "source" : "destination", idx, 
                        parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_OUTPUT);
        }
      }
      return composite_idx;
    }

    //--------------------------------------------------------------------------
    void CopyOp::report_profiling_results(void)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_copy_report_profiling(this, &profiling_results);
#ifdef DEBUG_LEGION
      assert(profiling_reported.exists());
#endif
      Runtime::trigger_event(profiling_reported);
    }

    /////////////////////////////////////////////////////////////
    // Fence Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FenceOp::FenceOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FenceOp::FenceOp(const FenceOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FenceOp::~FenceOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FenceOp& FenceOp::operator=(const FenceOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FenceOp::initialize(SingleTask *ctx, FenceKind kind)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      fence_kind = kind;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_fence_operation(parent_ctx->get_unique_id(),
                                       unique_op_id);
    }

    //--------------------------------------------------------------------------
    void FenceOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void FenceOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      runtime->free_fence_op(this);
    }

    //--------------------------------------------------------------------------
    const char* FenceOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[FENCE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind FenceOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return FENCE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void FenceOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      begin_dependence_analysis();
      // Register this fence with all previous users in the parent's context
      parent_ctx->perform_fence_analysis(this);
      // Now update the parent context with this fence
      // before we can complete the dependence analysis
      // and possibly be deactivated
      parent_ctx->update_current_fence(this);
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    bool FenceOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      switch (fence_kind)
      {
        case MAPPING_FENCE:
          {
            complete_mapping();
            complete_execution();
            break;
          }
        case MIXED_FENCE:
          {
            // Mark that we finished our mapping now
            complete_mapping();
            // Intentionally fall through
          }
        case EXECUTION_FENCE:
          {
            // Go through and launch a completion task dependent upon
            // all the completion events of our incoming dependences.
            // Make sure that the events that we pulled out our still valid.
            // Note since we are performing this operation, then we know
            // that we are mapped and therefore our set of input dependences
            // have been fixed so we can read them without holding the lock.
            std::set<ApEvent> trigger_events;
            for (std::map<Operation*,GenerationID>::const_iterator it = 
                  incoming.begin(); it != incoming.end(); it++)
            {
              ApEvent complete = it->first->get_completion_event();
              if (it->second == it->first->get_generation())
                trigger_events.insert(complete);
            }
            RtEvent wait_on = Runtime::protect_merge_events(trigger_events);
            if (!wait_on.has_triggered())
            {
              DeferredExecuteArgs deferred_execute_args;
              deferred_execute_args.hlr_id = HLR_DEFERRED_EXECUTION_TRIGGER_ID;
              deferred_execute_args.proxy_this = this;
              runtime->issue_runtime_meta_task(&deferred_execute_args,
                                               sizeof(deferred_execute_args),
                                              HLR_DEFERRED_EXECUTION_TRIGGER_ID,
                                               HLR_LATENCY_PRIORITY,
                                               this, wait_on);
            }
            else
              deferred_execute();
            break;
          }
        default:
          assert(false); // should never get here
      }
      // If we successfully performed the operation return true
      return true;
    }

    //--------------------------------------------------------------------------
    void FenceOp::deferred_execute(void)
    //--------------------------------------------------------------------------
    {
      switch (fence_kind)
      {
        case EXECUTION_FENCE:
          {
            complete_mapping();
            // Intentionally fall through
          }
        case MIXED_FENCE:
          {
            complete_execution();
            break;
          }
        default:
          assert(false); // should never get here
      }
    } 

    /////////////////////////////////////////////////////////////
    // Frame Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FrameOp::FrameOp(Runtime *rt)
      : FenceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FrameOp::FrameOp(const FrameOp &rhs)
      : FenceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FrameOp::~FrameOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FrameOp& FrameOp::operator=(const FrameOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FrameOp::initialize(SingleTask *ctx)
    //--------------------------------------------------------------------------
    {
      FenceOp::initialize(ctx, MIXED_FENCE);
      parent_ctx->issue_frame(this, completion_event); 
    }

    //--------------------------------------------------------------------------
    void FrameOp::set_previous(ApEvent previous)
    //--------------------------------------------------------------------------
    {
      previous_completion = previous;
    }

    //--------------------------------------------------------------------------
    void FrameOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      previous_completion = ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    void FrameOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      runtime->free_frame_op(this);
    }

    //--------------------------------------------------------------------------
    const char* FrameOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[FRAME_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind FrameOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return FRAME_OP_KIND;
    }

    //--------------------------------------------------------------------------
    bool FrameOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      // Increment the number of mapped frames
      parent_ctx->increment_frame();
      // Mark that we finished our mapping now
      complete_mapping();
      // Go through and launch a completion task dependent upon
      // all the completion events of our incoming dependences.
      // Make sure that the events that we pulled out our still valid.
      // Note since we are performing this operation, then we know
      // that we are mapped and therefore our set of input dependences
      // have been fixed so we can read them without holding the lock.
      std::set<ApEvent> trigger_events;
      // Include our previous completion event if necessary
      if (previous_completion.exists())
        trigger_events.insert(previous_completion);
      for (std::map<Operation*,GenerationID>::const_iterator it = 
            incoming.begin(); it != incoming.end(); it++)
      {
        ApEvent complete = it->first->get_completion_event();
        if (it->second == it->first->get_generation())
          trigger_events.insert(complete);
      }
      RtEvent wait_on = Runtime::protect_merge_events(trigger_events);
      if (!wait_on.has_triggered())
      {
        DeferredExecuteArgs deferred_execute_args;
        deferred_execute_args.hlr_id = HLR_DEFERRED_EXECUTION_TRIGGER_ID;
        deferred_execute_args.proxy_this = this;
        runtime->issue_runtime_meta_task(&deferred_execute_args,
                                         sizeof(deferred_execute_args),
                                         HLR_DEFERRED_EXECUTION_TRIGGER_ID,
                                         HLR_LATENCY_PRIORITY,
                                         this, wait_on);
      }
      else
        deferred_execute();
      return true;
    }

    //--------------------------------------------------------------------------
    void FrameOp::deferred_execute(void)
    //--------------------------------------------------------------------------
    {
      // This frame has finished executing so it is no longer mapped
      parent_ctx->decrement_frame();
      // This frame is also finished so we can tell the context
      parent_ctx->finish_frame(completion_event);
      complete_execution();
    }

    /////////////////////////////////////////////////////////////
    // Deletion Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeletionOp::DeletionOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeletionOp::DeletionOp(const DeletionOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DeletionOp::~DeletionOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeletionOp& DeletionOp::operator=(const DeletionOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_index_space_deletion(SingleTask *ctx,
                                                     IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      kind = INDEX_SPACE_DELETION;
      index_space = handle;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_deletion_operation(parent_ctx->get_unique_id(),
                                          unique_op_id);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_index_part_deletion(SingleTask *ctx,
                                                    IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      kind = INDEX_PARTITION_DELETION;
      index_part = handle;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_deletion_operation(parent_ctx->get_unique_id(),
                                          unique_op_id);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_field_space_deletion(SingleTask *ctx,
                                                     FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      kind = FIELD_SPACE_DELETION;
      field_space = handle;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_deletion_operation(parent_ctx->get_unique_id(),
                                          unique_op_id);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_field_deletion(SingleTask *ctx, 
                                                FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      kind = FIELD_DELETION;
      field_space = handle;
      free_fields.insert(fid);
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_deletion_operation(parent_ctx->get_unique_id(),
                                          unique_op_id);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_field_deletions(SingleTask *ctx,
                            FieldSpace handle, const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      kind = FIELD_DELETION;
      field_space = handle;
      free_fields = to_free;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_deletion_operation(parent_ctx->get_unique_id(),
                                          unique_op_id);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_logical_region_deletion(SingleTask *ctx,
                                                        LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      kind = LOGICAL_REGION_DELETION;
      logical_region = handle;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_deletion_operation(parent_ctx->get_unique_id(),
                                          unique_op_id);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_logical_partition_deletion(SingleTask *ctx,
                                                       LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      kind = LOGICAL_PARTITION_DELETION;
      logical_part = handle;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_deletion_operation(parent_ctx->get_unique_id(),
                                          unique_op_id);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void DeletionOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      free_fields.clear();
      parent_req_indexes.clear();
      // Return this to the available deletion ops on the queue
      runtime->free_deletion_op(this);
    }

    //--------------------------------------------------------------------------
    const char* DeletionOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[DELETION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind DeletionOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return DELETION_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void DeletionOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      std::vector<RegionRequirement> deletion_requirements;
      switch (kind)
      {
        case INDEX_SPACE_DELETION:
          {
            parent_ctx->analyze_destroy_index_space(index_space, 
                                                    deletion_requirements,
                                                    parent_req_indexes);
            break;
          }
        case INDEX_PARTITION_DELETION:
          {
            parent_ctx->analyze_destroy_index_partition(index_part, 
                                                        deletion_requirements,
                                                        parent_req_indexes);
            break;
          }
        case FIELD_SPACE_DELETION:
          {
            parent_ctx->analyze_destroy_field_space(field_space, 
                                                    deletion_requirements,
                                                    parent_req_indexes);
            break;
          }
        case FIELD_DELETION:
          {
            parent_ctx->analyze_destroy_fields(field_space, free_fields, 
                                               deletion_requirements,
                                               parent_req_indexes);
            break;
          }
        case LOGICAL_REGION_DELETION:
          {
            parent_ctx->analyze_destroy_logical_region(logical_region,
                                                       deletion_requirements,
                                                       parent_req_indexes);
            break;
          }
        case LOGICAL_PARTITION_DELETION:
          {
            parent_ctx->analyze_destroy_logical_partition(logical_part, 
                                                         deletion_requirements,
                                                         parent_req_indexes);
            break;
          }
        default:
          // should never get here
          assert(false);
      }
#ifdef DEBUG_LEGION
      assert(deletion_requirements.size() == parent_req_indexes.size());
#endif
      if (Runtime::legion_spy_enabled)
      {
        for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
        {
          const RegionRequirement &req = deletion_requirements[idx];
          if (req.handle_type != PART_PROJECTION)
            LegionSpy::log_logical_requirement(unique_op_id, idx,true/*region*/,
                                               req.region.index_space.id,
                                               req.region.field_space.id,
                                               req.region.tree_id,
                                               req.privilege,
                                               req.prop, req.redop,
                                               req.parent.index_space.id);
          else
            LegionSpy::log_logical_requirement(unique_op_id,idx,false/*region*/,
                                               req.partition.index_partition.id,
                                               req.partition.field_space.id,
                                               req.partition.tree_id,
                                               req.privilege,
                                               req.prop, req.redop,
                                               req.parent.index_space.id);
          LegionSpy::log_requirement_fields(unique_op_id, idx, 
                                            req.privilege_fields);
        }
      }
      begin_dependence_analysis();
      for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
      {
        RegionRequirement &req = deletion_requirements[idx];
        // Perform the normal region requirement analysis
        VersionInfo version_info;
        RestrictInfo restrict_info;
        RegionTreePath privilege_path;
        initialize_privilege_path(privilege_path, req);
        runtime->forest->perform_deletion_analysis(this, idx, req, 
                                                   version_info,
                                                   restrict_info,
                                                   privilege_path);
        version_info.release();
      }
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    bool DeletionOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      switch (kind)
      {
        case INDEX_SPACE_DELETION:
          {
            // Only need to tell our parent if it is a top-level index space
            if (runtime->forest->is_top_level_index_space(index_space))
              parent_ctx->register_index_space_deletion(index_space);
            break;
          }
        case INDEX_PARTITION_DELETION:
          {
            parent_ctx->register_index_partition_deletion(index_part);
            break;
          }
        case FIELD_SPACE_DELETION:
          {
            parent_ctx->register_field_space_deletion(field_space);
            break;
          }
        case FIELD_DELETION:
          {
            parent_ctx->register_field_deletions(field_space, free_fields);
            break;
          }
        case LOGICAL_REGION_DELETION:
          {
            // Only need to tell our parent if it is a top-level region
            if (runtime->forest->is_top_level_region(logical_region))
              parent_ctx->register_region_deletion(logical_region);
            break;
          }
        case LOGICAL_PARTITION_DELETION:
          {
            // We don't need to register partition deletions explicitly
            break;
          }
        default:
          assert(false); // should never get here
      }
      complete_mapping();
      complete_execution();
      return true;
    }

    //--------------------------------------------------------------------------
    unsigned DeletionOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < parent_req_indexes.size());
#endif
      return parent_req_indexes[idx];
    }

    /////////////////////////////////////////////////////////////
    // Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CloseOp::CloseOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CloseOp::CloseOp(const CloseOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CloseOp::~CloseOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CloseOp& CloseOp::operator=(const CloseOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    } 

    //--------------------------------------------------------------------------
    UniqueID CloseOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    int CloseOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    size_t CloseOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    void CloseOp::initialize_close(SingleTask *ctx,
                                   const RegionRequirement &req, bool track)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(completion_event.exists());
#endif
      initialize_operation(ctx, track);
      parent_task = ctx;
      requirement = req;
      initialize_privilege_path(privilege_path, requirement);
    } 

    //--------------------------------------------------------------------------
    void CloseOp::initialize_close(SingleTask *ctx, unsigned idx, bool track)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(completion_event.exists());
#endif
      initialize_operation(ctx, track);
      parent_task = ctx;
      ctx->clone_requirement(idx, requirement);
      initialize_privilege_path(privilege_path, requirement);
    }

    //--------------------------------------------------------------------------
    void CloseOp::perform_logging(bool is_intermediate_close_op, bool read_only)
    //--------------------------------------------------------------------------
    {
      if (!Runtime::legion_spy_enabled)
        return;
      LegionSpy::log_close_operation(parent_ctx->get_unique_id(),
                                     unique_op_id,
                                     is_intermediate_close_op, read_only);
      if (requirement.handle_type == PART_PROJECTION)
        LegionSpy::log_logical_requirement(unique_op_id, 0/*idx*/,
                                  false/*region*/,
                                  requirement.partition.index_partition.id,
                                  requirement.partition.field_space.id,
                                  requirement.partition.tree_id,
                                  requirement.privilege,
                                  requirement.prop,
                                  requirement.redop,
                                  requirement.parent.index_space.id);
      else
        LegionSpy::log_logical_requirement(unique_op_id, 0/*idx*/,
                                  true/*region*/,
                                  requirement.region.index_space.id,
                                  requirement.region.field_space.id,
                                  requirement.region.tree_id,
                                  requirement.privilege,
                                  requirement.prop,
                                  requirement.redop,
                                  requirement.parent.index_space.id);
      LegionSpy::log_requirement_fields(unique_op_id, 0/*idx*/,
                                requirement.privilege_fields);
    } 

    //--------------------------------------------------------------------------
    void CloseOp::activate_close(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
#ifdef DEBUG_LEGION
      assert(completion_event.exists());
#endif
    }

    //--------------------------------------------------------------------------
    void CloseOp::deactivate_close(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      privilege_path.clear();
      version_info.clear();
      restrict_info.clear();
    } 

    //--------------------------------------------------------------------------
    void CloseOp::trigger_remote_state_analysis(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      version_info.make_local(preconditions, this, runtime->forest);
      if (preconditions.empty())
        Runtime::trigger_event(ready_event);
      else
        Runtime::trigger_event(ready_event,
                               Runtime::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    void CloseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      commit_operation(true/*deactivate*/);
    }

    /////////////////////////////////////////////////////////////
    // Trace Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceCloseOp::TraceCloseOp(Runtime *runtime)
      : CloseOp(runtime)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceCloseOp::~TraceCloseOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void TraceCloseOp::initialize_trace_close_op(SingleTask *ctx, 
                                                 const RegionRequirement &req,
                        const LegionMap<ColorPoint,FieldMask>::aligned &targets,
                                                 LegionTrace *trace, int close, 
                                                 const FieldMask &close_m,
                                                 Operation *create)
    //--------------------------------------------------------------------------
    {
      // Don't track these kinds of closes
      // We don't want to be stalling in the analysis pipeline
      // because we ran out of slots to issue
      initialize_close(ctx, req, false/*track*/);
      // Since we didn't register with our parent, we need to set
      // any trace that we might have explicitly. Get whether we
      // are tracing from the creation operation.
      if (trace != NULL)
        set_trace(trace, create->is_tracing());
      requirement = req;
      initialize_privilege_path(privilege_path, requirement);
      target_children = targets;
      close_idx = close;
      close_mask = close_m;
      create_op = create;
      create_gen = create_op->get_generation(); 
    }

    //--------------------------------------------------------------------------
    void TraceCloseOp::activate_trace_close(void)
    //--------------------------------------------------------------------------
    {
      activate_close();
      close_idx = -1;
      create_op = NULL;
      create_gen = 0;
    }

    //--------------------------------------------------------------------------
    void TraceCloseOp::deactivate_trace_close(void)
    //--------------------------------------------------------------------------
    {
      deactivate_close();
      target_children.clear();
      next_children.clear();
      close_mask.clear();
    }

    //--------------------------------------------------------------------------
    void TraceCloseOp::record_trace_dependence(Operation *target, 
                                               GenerationID target_gen,
                                               int target_idx,
                                               int source_idx, 
                                               DependenceType dtype,
                                               const FieldMask &dependent_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(close_idx >= 0);
#endif
      // Check to see if the target is also our creator
      // in which case we can skip it
      if ((target == create_op) && (target_gen == create_gen))
        return;
      // Check to see if the source is our source
      if (source_idx != close_idx)
        return;
      FieldMask overlap = close_mask & dependent_mask;
      // If the fields also don't overlap then we are done
      if (!overlap)
        return;
      // Otherwise do the registration
      register_region_dependence(0/*idx*/, target, target_gen,
                               target_idx, dtype, false/*validates*/, overlap);
    }

    //--------------------------------------------------------------------------
    void TraceCloseOp::add_next_child(const ColorPoint &next_child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(next_child.is_valid());
#endif
      next_children.insert(next_child);
    }

    /////////////////////////////////////////////////////////////
    // Inter Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InterCloseOp::InterCloseOp(Runtime *runtime)
      : TraceCloseOp(runtime)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InterCloseOp::InterCloseOp(const InterCloseOp &rhs)
      : TraceCloseOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    InterCloseOp::~InterCloseOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InterCloseOp& InterCloseOp::operator=(const InterCloseOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::initialize(SingleTask *ctx, const RegionRequirement &req,
                        const LegionMap<ColorPoint,FieldMask>::aligned &targets,
                                  LegionTrace *trace, int close, 
                                  const VersionInfo &close_info,
                                  const VersionInfo &ver_info,
                                  const RestrictInfo &res_info,
                                  const FieldMask &close_m, Operation *create)
    //--------------------------------------------------------------------------
    {
      initialize_trace_close_op(ctx, req, targets,
                                trace, close, close_m, create);
      // Merge in the two different version informations
      version_info.merge(close_info, close_m);
      version_info.merge(ver_info, close_m);
      restrict_info.merge(res_info, close_m);
      parent_req_index = create->find_parent_index(close_idx);
      if (Runtime::legion_spy_enabled)
      {
        perform_logging(true/*is intermediate close op*/, false/*read only*/);
        LegionSpy::log_close_op_creator(unique_op_id,
                                        create->get_unique_op_id(),
                                        close_idx);
      }
    } 

    //--------------------------------------------------------------------------
    void InterCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_trace_close();  
      mapper = NULL;
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_trace_close();
#ifdef DEBUG_LEGION
      assert(acquired_instances.empty());
#endif
      acquired_instances.clear();
      map_applied_conditions.clear();
      profiling_results = Mapper::CloseProfilingInfo();
      runtime->free_inter_close_op(this);
    }

    //--------------------------------------------------------------------------
    const char* InterCloseOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[INTER_CLOSE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind InterCloseOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return INTER_CLOSE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    bool InterCloseOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(completion_event.exists());
#endif
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      // See if we are restricted or not and if not find our valid instances 
      InstanceSet chosen_instances;
      int composite_idx = -1;
      if (!restrict_info.has_restrictions())
      {
        InstanceSet valid_instances;
        runtime->forest->physical_traverse_path(physical_ctx, privilege_path,
                                                requirement, version_info,
                                                this, 0/*idx*/, 
                                                true/*find valid*/,
                                                map_applied_conditions,
                                                valid_instances
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
        // now invoke the mapper to find the instances to use
        composite_idx = invoke_mapper(valid_instances, chosen_instances);
      }
      else
      {
        parent_ctx->get_physical_references(parent_req_index, chosen_instances);
      } 
      // Now we can perform our close operation
      ApEvent close_event = 
        runtime->forest->physical_perform_close(physical_ctx, requirement,
                                                version_info, this, 0/*idx*/, 
                                                composite_idx, target_children,
                                                next_children, completion_event,
                                                map_applied_conditions,
                                                chosen_instances
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
      if (Runtime::legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, 0/*idx*/,
                                              requirement,
                                              chosen_instances);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, close_event, 
                                        completion_event);
#endif
      }
      version_info.apply_close(physical_ctx.get_id(), runtime->address_space,
                               target_children, map_applied_conditions);
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      complete_execution(Runtime::protect_event(close_event));
      // This should always succeed
      return true;
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      // Don't commit this operation until the profiling information is reported
      commit_operation(true/*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    unsigned InterCloseOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
      assert(create_op != NULL);
#endif
      return create_op->find_parent_index(close_idx);
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::select_sources(const InstanceRef &target,
                                      const InstanceSet &sources,
                                      std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      Mapper::SelectCloseSrcInput input;
      Mapper::SelectCloseSrcOutput output;
      prepare_for_mapping(target, input.target);
      prepare_for_mapping(sources, input.source_instances);
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_select_close_sources(this, &input, &output);
      compute_ranking(output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                                  InterCloseOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    int InterCloseOp::invoke_mapper(const InstanceSet &valid_instances,
                                          InstanceSet &chosen_instances)
    //--------------------------------------------------------------------------
    {
      Mapper::MapCloseInput input;
      Mapper::MapCloseOutput output;
      // No need to filter for close operations
      prepare_for_mapping(valid_instances, input.valid_instances);
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      if (requirement.handle_type == PART_PROJECTION)
      {
        // Always tell the mapper that we are closing to the parent region
        // if we are actually closing to a partition
        LogicalPartition partition = requirement.partition;
        requirement.handle_type = SINGULAR;
        requirement.region = 
          runtime->forest->get_parent_logical_region(partition); 
        mapper->invoke_map_close(this, &input, &output);
        requirement.handle_type = PART_PROJECTION;
        requirement.partition = partition;
      }
      else // This is the common case
        mapper->invoke_map_close(this, &input, &output);
      // Now we have to validate the output
      // Make sure we have at least one instance for every field
      RegionTreeID bad_tree = 0;
      std::vector<FieldID> missing_fields;
      std::vector<PhysicalManager*> unacquired;
      int composite_index = runtime->forest->physical_convert_mapping(this,
                                  requirement, output.chosen_instances, 
                                  chosen_instances, bad_tree, missing_fields,
                                  &acquired_instances, unacquired, 
                                  !Runtime::unsafe_mapper);
      if (bad_tree > 0)
      {
        log_run.error("Invalid mapper output from invocation of 'map_close' "
                      "on mapper %s. Mapper selected a physical instance from "
                      "region tree %d to satisfy region requirement from "
                      "close operation in task %s (ID %lld) whose logical "
                      "region is from region tree %d",mapper->get_mapper_name(),
                      bad_tree, parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id(), 
                      requirement.region.get_tree_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit (ERROR_INVALID_MAPPER_OUTPUT);
      }
      if (!missing_fields.empty())
      {
        log_run.error("Invalid mapper output from invocation of 'map_close' "
                      "on mapper %s. Mapper failed to specify a physical "
                      "instance for %ld fields fo the region requirement to "
                      "a close operation in task %s (ID %lld). The missing "
                      "fields are listed below.", mapper->get_mapper_name(),
                      missing_fields.size(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id());
        for (std::vector<FieldID>::const_iterator it = missing_fields.begin();
              it != missing_fields.end(); it++)
        {
          const void *name; size_t name_size;
          runtime->retrieve_semantic_information(
              requirement.region.get_field_space(), *it, NAME_SEMANTIC_TAG,
              name, name_size, false, false);
          log_run.error("Missing instance for field %s (FieldID: %d)",
                        static_cast<const char*>(name), *it);
        }
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      if (!unacquired.empty())
      {
        for (std::vector<PhysicalManager*>::const_iterator it = 
              unacquired.begin(); it != unacquired.end(); it++)
        {
          if (acquired_instances.find(*it) == acquired_instances.end())
          { 
            log_run.error("Invalid mapper output from 'map_close' invocation "
                          "on mapper %s. Mapper selected physical instance for "
                          "close operation in task %s (ID %lld) which has "
                          "already been collected. If the mapper had properly "
                          "acquired this instance as part of the mapper call "
                          "it would have detected this. Please update the "
                          "mapper to abide by proper mapping conventions.",
                          mapper->get_mapper_name(),parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
        }
        // If we did successfully acquire them, still issue the warning
        log_run.warning("WARNING: mapper %s failed to acquire instance "
                        "for close operation in task %s (ID %lld) in "
                        "'map_close' call. You may experience undefined "
                        "behavior as a consequence.", mapper->get_mapper_name(),
                        parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id());
      } 
      if (Runtime::unsafe_mapper)
        return composite_index;
      std::vector<LogicalRegion> regions_to_check(1, requirement.region);
      for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
      {
        if (int(idx) == composite_index)
          continue;
        const InstanceRef &ref = chosen_instances[idx];
        if (!ref.get_manager()->meets_regions(regions_to_check))
        {
          log_run.error("Invalid mapper output from invocation of 'map_close' "
                        "on mapper %s. Mapper specified an instance which does "
                        "not meet the logical region requirement. The close "
                        "operation was issued in task %s (ID %lld).",
                        mapper->get_mapper_name(), parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_OUTPUT);
        }
      }
      return composite_index;
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::report_profiling_results(void)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_close_report_profiling(this, &profiling_results);
#ifdef DEBUG_LEGION
      assert(profiling_reported.exists());
#endif
      Runtime::trigger_event(profiling_reported);
    }

    /////////////////////////////////////////////////////////////
    // Read Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReadCloseOp::ReadCloseOp(Runtime *rt)
      : TraceCloseOp(rt)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    ReadCloseOp::ReadCloseOp(const ReadCloseOp &rhs)
      : TraceCloseOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReadCloseOp::~ReadCloseOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReadCloseOp& ReadCloseOp::operator=(const ReadCloseOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReadCloseOp::initialize(SingleTask *ctx, const RegionRequirement &req,
                        const LegionMap<ColorPoint,FieldMask>::aligned &targets,
                                 LegionTrace *trace, int close,
                                 const FieldMask &close_m, Operation *create)
    //--------------------------------------------------------------------------
    {
      initialize_trace_close_op(ctx, req, targets, 
                                trace, close, close_m, create);
      parent_req_index = create->find_parent_index(close_idx);
      if (Runtime::legion_spy_enabled)
      {
        perform_logging(true/*is intermediate close op*/, true/*read only*/);
        LegionSpy::log_close_op_creator(unique_op_id,
                                        create->get_unique_op_id(),
                                        close_idx);
      }
    }

    //--------------------------------------------------------------------------
    void ReadCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_trace_close();
    }
    
    //--------------------------------------------------------------------------
    void ReadCloseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_trace_close();
      runtime->free_read_close_op(this);
    }

    //--------------------------------------------------------------------------
    const char* ReadCloseOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[READ_CLOSE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind ReadCloseOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return READ_CLOSE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    unsigned ReadCloseOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    /////////////////////////////////////////////////////////////
    // Post Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PostCloseOp::PostCloseOp(Runtime *runtime)
      : CloseOp(runtime)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PostCloseOp::PostCloseOp(const PostCloseOp &rhs)
      : CloseOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PostCloseOp::~PostCloseOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PostCloseOp& PostCloseOp::operator=(const PostCloseOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::initialize(SingleTask *ctx, unsigned idx) 
    //--------------------------------------------------------------------------
    {
      initialize_close(ctx, idx, true/*track*/);
      // If it was write-discard from the task's perspective, make it
      // read-write within the task's context
      if (requirement.privilege == WRITE_DISCARD)
        requirement.privilege = READ_WRITE;
      parent_idx = idx;
      localize_region_requirement(requirement);
      if (Runtime::legion_spy_enabled)
      {
        perform_logging(false/*intermediate close op*/, false/*read only*/);
        LegionSpy::log_close_op_creator(unique_op_id,
                                        ctx->get_unique_op_id(),
                                        parent_idx);
      }
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_close();
      mapper = NULL;
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_close();
#ifdef DEBUG_LEGION
      assert(acquired_instances.empty());
#endif
      acquired_instances.clear();
      map_applied_conditions.clear();
      profiling_results = Mapper::CloseProfilingInfo();
      runtime->free_post_close_op(this);
    }

    //--------------------------------------------------------------------------
    const char* PostCloseOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[POST_CLOSE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind PostCloseOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return POST_CLOSE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(completion_event.exists());
#endif
      // This stage is only done for close operations issued
      // at the end of the task as dependence analysis for other
      // close operations is done inline in the region tree traversal
      // for other kinds of operations 
      // see RegionTreeNode::register_logical_node
      begin_dependence_analysis();
      // Handle a special case that involves closing to a reduction instance
      if (requirement.privilege == REDUCE)
        runtime->forest->perform_reduction_close_analysis(this, 0/*idx*/,
                                                          requirement,
                                                          version_info);
      else
        runtime->forest->perform_dependence_analysis(this, 0/*idx*/,
                                                     requirement,
                                                     version_info,
                                                     restrict_info,
                                                     privilege_path);
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    bool PostCloseOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(completion_event.exists());
#endif
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_idx);
      // We already know the instances that we are going to need
      InstanceSet chosen_instances;
      parent_ctx->get_physical_references(parent_idx, chosen_instances);
      ApEvent close_event = 
        runtime->forest->physical_close_context(physical_ctx, requirement,
                                                version_info, this, 0/*idx*/,
                                                map_applied_conditions,
                                                chosen_instances
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
      if (Runtime::legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, 0/*idx*/,
                                              requirement,
                                              chosen_instances);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, close_event, 
                                        completion_event);
#endif
      }
      // No need to apply our mapping because we are done!
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      Runtime::trigger_event(completion_event, close_event);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(close_event));
      // This should always succeed
      return true;
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      // Only commit this operation if we are done profiling
      commit_operation(true/*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    unsigned PostCloseOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return parent_idx;
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::select_sources(const InstanceRef &target,
                                     const InstanceSet &sources,
                                     std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      Mapper::SelectCloseSrcInput input;
      Mapper::SelectCloseSrcOutput output;
      prepare_for_mapping(target, input.target);
      prepare_for_mapping(sources, input.source_instances);
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_select_close_sources(this, &input, &output);
      compute_ranking(output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                                   PostCloseOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::report_profiling_results(void)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_close_report_profiling(this, &profiling_results);
#ifdef DEBUG_LEGION
      assert(profiling_reported.exists());
#endif
      Runtime::trigger_event(profiling_reported);
    }

    /////////////////////////////////////////////////////////////
    // Virtual Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VirtualCloseOp::VirtualCloseOp(Runtime *rt)
      : CloseOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VirtualCloseOp::VirtualCloseOp(const VirtualCloseOp &rhs) 
      : CloseOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    VirtualCloseOp::~VirtualCloseOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VirtualCloseOp& VirtualCloseOp::operator=(const VirtualCloseOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void VirtualCloseOp::initialize(SingleTask *ctx, unsigned index)
    //--------------------------------------------------------------------------
    {
      initialize_close(ctx, index, true/*track*/);
      // Make this read-write to pick up the earlier changes
      if (requirement.privilege == WRITE_DISCARD)
        requirement.privilege = READ_WRITE;
      parent_idx = index;
      localize_region_requirement(requirement);
      if (Runtime::legion_spy_enabled)
      {
        perform_logging(false/*intermediate close op*/, false/*read only*/);
        LegionSpy::log_close_op_creator(unique_op_id,
                                        ctx->get_unique_op_id(),
                                        parent_idx);
      }
    }
    
    //--------------------------------------------------------------------------
    void VirtualCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_close();
    }

    //--------------------------------------------------------------------------
    void VirtualCloseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_close();
      map_applied_conditions.clear();
      runtime->free_virtual_close_op(this);
    }

    //--------------------------------------------------------------------------
    const char* VirtualCloseOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[VIRTUAL_CLOSE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind VirtualCloseOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return VIRTUAL_CLOSE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void VirtualCloseOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      begin_dependence_analysis();
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   version_info,
                                                   restrict_info,
                                                   privilege_path);
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    bool VirtualCloseOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_idx);
      InstanceSet composite_refs(1);
      composite_refs[0] = InstanceRef(true/*composite*/);
      // Always eagerly translate composite instances back out into
      // the parent context
      runtime->forest->map_virtual_region(physical_ctx, requirement,
                                          composite_refs[0], version_info,
                                          parent_ctx->get_parent(), this,
                                          true/*needs fields*/
#ifdef DEBUG_LEGION
                                          , 0/*idx*/, get_logging_name()
                                          , unique_op_id
#endif
                                          );
      // Pass the reference back to the parent task
      parent_ctx->return_virtual_instance(parent_idx, composite_refs); 
      // Then we can mark that we are mapped and executed
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      complete_execution();
      return true;
    }

    //--------------------------------------------------------------------------
    unsigned VirtualCloseOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return parent_idx;
    }

    //--------------------------------------------------------------------------
    void VirtualCloseOp::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    /////////////////////////////////////////////////////////////
    // Acquire Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AcquireOp::AcquireOp(Runtime *rt)
      : Acquire(), SpeculativeOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AcquireOp::AcquireOp(const AcquireOp &rhs)
      : Acquire(), SpeculativeOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    AcquireOp::~AcquireOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AcquireOp& AcquireOp::operator=(const AcquireOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::initialize(Context ctx, 
                               const AcquireLauncher &launcher,
                               bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx;
      initialize_speculation(ctx, true/*track*/,
                             1/*num region requirements*/,
                             launcher.predicate);
      // Note we give it READ WRITE EXCLUSIVE to make sure that nobody
      // can be re-ordered around this operation for mapping or
      // normal dependences.  We won't actually read or write anything.
      requirement = RegionRequirement(launcher.logical_region, READ_WRITE,
                                      EXCLUSIVE, launcher.parent_region); 
      // Do a little bit of error checking
      {
        const RegionRequirement &physical_req = 
          launcher.physical_region.impl->get_requirement();
        if (!runtime->forest->is_subregion(launcher.logical_region, 
                                           physical_req.region))
        {
          log_task.error("ERROR: Acquire operation requested privileges "
                         "on logical region (%x,%d,%d) which is "
                         "not a subregion of the physical instance "
                         "region (%x,%d,%d)",
                         launcher.logical_region.index_space.id,
                         launcher.logical_region.field_space.id,
                         launcher.logical_region.get_tree_id(),
                         physical_req.region.index_space.id,
                         physical_req.region.field_space.id,
                         physical_req.region.get_tree_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_ACQUIRE_MISMATCH);
        }
        for (std::set<FieldID>::const_iterator it = launcher.fields.begin();
              it != launcher.fields.end(); it++)
        {
          if (physical_req.privilege_fields.find(*it) == 
              physical_req.privilege_fields.end())
          {
            log_task.error("ERROR: Acquire operation requested on "
                           "field %d which is not contained in the "
                           "requested physical instance", *it);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_ACQUIRE_MISMATCH);
          }
        }
      }
      if (launcher.fields.empty())
      {
        log_task.warning("WARNING: PRIVILEGE FIELDS OF ACQUIRE OPERATION"
                         "IN TASK %s (ID %lld) HAS NO PRIVILEGE "
                         "FIELDS! DID YOU FORGET THEM?!?",
                         parent_ctx->get_task_name(), 
                         parent_ctx->get_unique_id());
      }
      requirement.privilege_fields = launcher.fields;
      logical_region = launcher.logical_region;
      parent_region = launcher.parent_region;
      fields = launcher.fields; 
      // Mark the requirement restricted
      region = launcher.physical_region;
      grants = launcher.grants;
      // Register ourselves with all the grants
      for (unsigned idx = 0; idx < grants.size(); idx++)
        grants[idx].impl->register_operation(completion_event);
      wait_barriers = launcher.wait_barriers;
#ifdef LEGION_SPY
      for (std::vector<PhaseBarrier>::const_iterator it = 
            launcher.arrive_barriers.begin(); it != 
            launcher.arrive_barriers.end(); it++)
      {
        arrive_barriers.push_back(*it);
        LegionSpy::log_event_dependence(it->phase_barrier,
                                arrive_barriers.back().phase_barrier);
      }
#else
      arrive_barriers = launcher.arrive_barriers;
#endif
      map_id = launcher.map_id;
      tag = launcher.tag;
      if (check_privileges)
        check_acquire_privilege();
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_acquire_operation(parent_ctx->get_unique_id(),
                                         unique_op_id);
        LegionSpy::log_logical_requirement(unique_op_id,0/*index*/,
                                           true/*region*/,
                                           requirement.region.index_space.id,
                                           requirement.region.field_space.id,
                                           requirement.region.tree_id,
                                           requirement.privilege,
                                           requirement.prop,
                                           requirement.redop,
                                           requirement.parent.index_space.id);
        LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                          requirement.privilege_fields);
      }
    }

    //--------------------------------------------------------------------------
    void AcquireOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative();
      mapper = NULL;
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_speculative();  
      // Remove our reference to the physical region
      region = PhysicalRegion();
      privilege_path.clear();
      fields.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      version_info.clear();
#ifdef DEBUG_LEGION
      assert(acquired_instances.empty());
#endif
      acquired_instances.clear();
      map_applied_conditions.clear();
      profiling_results = Mapper::AcquireProfilingInfo();
      // Return this operation to the runtime
      runtime->free_acquire_op(this);
    }

    //--------------------------------------------------------------------------
    const char* AcquireOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[ACQUIRE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind AcquireOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return ACQUIRE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t AcquireOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    Mappable* AcquireOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    { 
      // First compute the parent index
      compute_parent_index();
      begin_dependence_analysis();
      // Register a dependence on our predicate
      register_predicate_dependence();
      // First register any mapping dependences that we have
      RestrictInfo restrict_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   version_info,
                                                   restrict_info,
                                                   privilege_path);
      // Now tell the forest that we have user-level coherence
      RegionTreeContext ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      runtime->forest->acquire_user_coherence(ctx, requirement.region,
                                              requirement.privilege_fields);
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_remote_state_analysis(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;  
      version_info.make_local(preconditions, this, runtime->forest);
      if (preconditions.empty())
        Runtime::trigger_event(ready_event);
      else
        Runtime::trigger_event(ready_event,
                               Runtime::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    void AcquireOp::resolve_true(void)
    //--------------------------------------------------------------------------
    {
      // Put this on the queue of stuff to do
      runtime->add_to_local_queue(parent_ctx->get_executing_processor(),
                                  this, false/*prev fail*/);
    }

    //--------------------------------------------------------------------------
    void AcquireOp::resolve_false(void)
    //--------------------------------------------------------------------------
    {
      // Clean up this operation
      complete_execution();
      assert(0 && "TODO: advance mapping states if you care");
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    bool AcquireOp::speculate(bool &value)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      Mapper::SpeculativeOutput output;
      output.speculate = false;
      mapper->invoke_acquire_speculate(this, &output);
      if (output.speculate)
      {
        value = output.speculative_value;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool AcquireOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      // Map this is a restricted region. We already know the 
      // physical region that we want to map.
      InstanceSet mapped_instances;
      parent_ctx->get_physical_references(parent_req_index, mapped_instances); 
      // Invoke the mapper before doing anything else 
      invoke_mapper();
      // Now we can map the operation
      runtime->forest->traverse_and_register(physical_ctx, privilege_path,
                                             requirement, version_info, 
                                             this, 0/*idx*/, completion_event, 
                                             false/*defer add users*/,
                                             map_applied_conditions,
                                             mapped_instances
#ifdef DEBUG_LEGION
                                             , get_logging_name()
                                             , unique_op_id
#endif
                                             );
      version_info.apply_mapping(physical_ctx.get_id(),
                                 runtime->address_space,map_applied_conditions);
      // Get all the events that need to happen before we can consider
      // ourselves acquired: reference ready and all synchronization
      std::set<ApEvent> acquire_preconditions;
      for (unsigned idx = 0; idx < mapped_instances.size(); idx++)
        acquire_preconditions.insert(mapped_instances[idx].get_ready_event());
      if (!wait_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::const_iterator it = 
              wait_barriers.begin(); it != wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(*it);
          acquire_preconditions.insert(e);
        }
      }
      if (!grants.empty())
      {
        for (std::vector<Grant>::const_iterator it = grants.begin();
              it != grants.end(); it++)
        {
          ApEvent e = it->impl->acquire_grant();
          acquire_preconditions.insert(e);
        }
      }
      ApEvent acquire_complete = Runtime::merge_events(acquire_preconditions);
      if (Runtime::legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, 0/*idx*/,
                                              requirement,
                                              mapped_instances);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, acquire_complete,
                                        completion_event);
#endif
      }
      // Chain any arrival barriers
      if (!arrive_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::iterator it = 
              arrive_barriers.begin(); it != arrive_barriers.end(); it++)
        {
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        completion_event);
        }
      }
      // Mark that we completed mapping
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      Runtime::trigger_event(completion_event, acquire_complete);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(acquire_complete));
      // we succeeded in mapping
      return true;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      // Don't commit thisoperation until we've reported profiling information
      commit_operation(true/*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    unsigned AcquireOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                                     AcquireOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    UniqueID AcquireOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    int AcquireOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& AcquireOp::get_requirement(void) const
    //--------------------------------------------------------------------------
    {
      return requirement;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::check_acquire_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, check the privileges, but only check the
      // data and not the actual privilege values since we're
      // using psuedo-read-write-exclusive
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, true/*skip*/);
      switch (et)
      {
        case NO_ERROR:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            log_region.error("Requirest for invalid region handle "
                                   "(%x,%d,%d) of requirement for "
                                   "acquire operation (ID %lld)",
                                   requirement.region.index_space.id, 
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id, 
                                   unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_REGION_HANDLE);
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) || 
                            (requirement.handle_type == REG_PROJECTION)
                             ? requirement.region.field_space : 
                               requirement.partition.field_space;
            log_region.error("Field %d is not a valid field of field "
                                   "space %d of requirement for acquire "
                                   "operation (ID %lld)",
                                   bad_field, sp.id, unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
          }
        case ERROR_BAD_PARENT_REGION:
          {
            log_region.error("Parent task %s (ID %lld) of acquire "
                             "operation (ID %lld) does not have a region "
                             "requirement for region (%x,%x,%x) as a parent",
                             parent_ctx->get_task_name(), 
                             parent_ctx->get_unique_id(),
                             unique_op_id, 
                             requirement.region.index_space.id,
                             requirement.region.field_space.id, 
                             requirement.region.tree_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_PARENT_REGION);
          }
        case ERROR_BAD_REGION_PATH:
          {
            log_region.error("Region (%x,%x,%x) is not a "
                             "sub-region of parent region (%x,%x,%x) of "
                             "requirement for acquire operation (ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id, 
                             requirement.region.tree_id,
                             requirement.parent.index_space.id,
                             requirement.parent.field_space.id,
                             requirement.parent.tree_id, unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_PATH);
          }
        case ERROR_BAD_REGION_TYPE:
          {
            log_region.error("Region requirement of acquire operation "
                                   "(ID %lld) cannot find privileges for field "
                                   "%d in parent task",
                                   unique_op_id, bad_field);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_TYPE);
          }
        // these should never happen with an acquire operation 
        case ERROR_INVALID_INSTANCE_FIELD:
        case ERROR_DUPLICATE_INSTANCE_FIELD:
        case ERROR_BAD_REGION_PRIVILEGES:
        case ERROR_NON_DISJOINT_PARTITION: 
        default:
          assert(false); // Should never happen
      }
    }

    //--------------------------------------------------------------------------
    void AcquireOp::compute_parent_index(void)
    //--------------------------------------------------------------------------
    {
      int parent_index = parent_ctx->find_parent_region_req(requirement,
                                                    false/*check privilege*/);
      if (parent_index < 0)
      {
        log_region.error("Parent task %s (ID %lld) of acquire "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id);
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_BAD_PARENT_REGION);
      }
      else
        parent_req_index = unsigned(parent_index);
    }

    //--------------------------------------------------------------------------
    void AcquireOp::invoke_mapper(void)
    //--------------------------------------------------------------------------
    {
      Mapper::MapAcquireInput input;
      Mapper::MapAcquireOutput output;
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_map_acquire(this, &input, &output);
    }

    //--------------------------------------------------------------------------
    void AcquireOp::report_profiling_results(void)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_acquire_report_profiling(this, &profiling_results);
#ifdef DEBUG_LEGION
      assert(profiling_reported.exists());
#endif
      Runtime::trigger_event(profiling_reported);
    }

    /////////////////////////////////////////////////////////////
    // Release Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReleaseOp::ReleaseOp(Runtime *rt)
      : Release(), SpeculativeOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReleaseOp::ReleaseOp(const ReleaseOp &rhs)
      : Release(), SpeculativeOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReleaseOp::~ReleaseOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReleaseOp& ReleaseOp::operator=(const ReleaseOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::initialize(Context ctx, 
                               const ReleaseLauncher &launcher, 
                               bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx;
      initialize_speculation(ctx, true/*track*/, 
                             1/*num region requirements*/,
                             launcher.predicate);
      // Note we give it READ WRITE EXCLUSIVE to make sure that nobody
      // can be re-ordered around this operation for mapping or
      // normal dependences.  We won't actually read or write anything.
      requirement = RegionRequirement(launcher.logical_region, READ_WRITE, 
                                      EXCLUSIVE, launcher.parent_region); 
      // Do a little bit of error checking
      {
        const RegionRequirement &physical_req = 
          launcher.physical_region.impl->get_requirement();
        if (!runtime->forest->is_subregion(launcher.logical_region, 
                                           physical_req.region))
        {
          log_task.error("ERROR: Release operation requested privileges "
                               "on logical region (%x,%d,%d) which is "
                               "not a subregion of the physical instance "
                               "region (%x,%d,%d)",
                               launcher.logical_region.index_space.id,
                               launcher.logical_region.field_space.id,
                               launcher.logical_region.get_tree_id(),
                               physical_req.region.index_space.id,
                               physical_req.region.field_space.id,
                               physical_req.region.get_tree_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_RELEASE_MISMATCH);
        }
        for (std::set<FieldID>::const_iterator it = launcher.fields.begin();
              it != launcher.fields.end(); it++)
        {
          if (physical_req.privilege_fields.find(*it) == 
              physical_req.privilege_fields.end())
          {
            log_task.error("ERROR: Release operation requested on "
                                 "field %d which is not contained in the "
                                 "requested physical instance", *it);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_RELEASE_MISMATCH);
          }
        }
      }
      if (launcher.fields.empty())
      {
        log_task.warning("WARNING: PRIVILEGE FIELDS OF RELEASE OPERATION"
                               "IN TASK %s (ID %lld) HAS NO PRIVILEGE "
                               "FIELDS! DID YOU FORGET THEM?!?",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id());
      }
      requirement.privilege_fields = launcher.fields;
      logical_region = launcher.logical_region;
      parent_region = launcher.parent_region;
      fields = launcher.fields; 
      region = launcher.physical_region;
      grants = launcher.grants;
      // Register ourselves with all the grants
      for (unsigned idx = 0; idx < grants.size(); idx++)
        grants[idx].impl->register_operation(completion_event);
      wait_barriers = launcher.wait_barriers;
#ifdef LEGION_SPY
      for (std::vector<PhaseBarrier>::const_iterator it = 
            launcher.arrive_barriers.begin(); it != 
            launcher.arrive_barriers.end(); it++)
      {
        arrive_barriers.push_back(*it);
        LegionSpy::log_event_dependence(it->phase_barrier,
                                arrive_barriers.back().phase_barrier);
      }
#else
      arrive_barriers = launcher.arrive_barriers;
#endif
      map_id = launcher.map_id;
      tag = launcher.tag;
      if (check_privileges)
        check_release_privilege();
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_release_operation(parent_ctx->get_unique_id(),
                                         unique_op_id);
        LegionSpy::log_logical_requirement(unique_op_id,0/*index*/,
                                           true/*region*/,
                                           requirement.region.index_space.id,
                                           requirement.region.field_space.id,
                                           requirement.region.tree_id,
                                           requirement.privilege,
                                           requirement.prop,
                                           requirement.redop,
                                           requirement.parent.index_space.id);
        LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                          requirement.privilege_fields);
      }
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative(); 
      mapper = NULL;
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_speculative();
      // Remove our reference to the physical region
      region = PhysicalRegion();
      privilege_path.clear();
      fields.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      version_info.clear();
#ifdef DEBUG_LEGION
      assert(acquired_instances.empty());
#endif
      acquired_instances.clear();
      map_applied_conditions.clear();
      profiling_results = Mapper::ReleaseProfilingInfo();
      // Return this operation to the runtime
      runtime->free_release_op(this);
    }

    //--------------------------------------------------------------------------
    const char* ReleaseOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[RELEASE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind ReleaseOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return RELEASE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t ReleaseOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    Mappable* ReleaseOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    { 
      // First compute the parent index
      compute_parent_index();
      begin_dependence_analysis();
      // Register a dependence on our predicate
      register_predicate_dependence();
      // First register any mapping dependences that we have
      RestrictInfo restrict_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   version_info,
                                                   restrict_info,
                                                   privilege_path);
      // Now tell the forest that we are relinquishing user-level coherence
      RegionTreeContext ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      runtime->forest->restrict_user_coherence(ctx, parent_ctx, 
                                               requirement.region,
                                               requirement.privilege_fields);
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::trigger_remote_state_analysis(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;  
      version_info.make_local(preconditions, this, runtime->forest);
      if (preconditions.empty())
        Runtime::trigger_event(ready_event);
      else
        Runtime::trigger_event(ready_event,
                               Runtime::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::resolve_true(void)
    //--------------------------------------------------------------------------
    {
      // Put this on the queue of stuff to do
      runtime->add_to_local_queue(parent_ctx->get_executing_processor(),
                                  this, false/*prev fail*/);
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::resolve_false(void)
    //--------------------------------------------------------------------------
    {
      // Clean up this operation
      complete_execution();
      assert(0 && "TODO: advance mapping states if you care");
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    bool ReleaseOp::speculate(bool &value)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      Mapper::SpeculativeOutput output;
      output.speculate = false;
      mapper->invoke_release_speculate(this, &output);
      if (output.speculate)
      {
        value = output.speculative_value;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool ReleaseOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      // We already know what the answer has to be here 
      InstanceSet mapped_instances;
      parent_ctx->get_physical_references(parent_req_index, mapped_instances); 
      // Invoke the mapper before doing anything else 
      invoke_mapper();
      // Now we can map the operation
      runtime->forest->traverse_and_register(physical_ctx, privilege_path,
                                             requirement, version_info, 
                                             this, 0/*idx*/, completion_event, 
                                             false/*defer add users*/,
                                             map_applied_conditions,
                                             mapped_instances
#ifdef DEBUG_LEGION
                                             , get_logging_name()
                                             , unique_op_id
#endif
                                             );
      version_info.apply_mapping(physical_ctx.get_id(),
                                 runtime->address_space,map_applied_conditions);
      std::set<ApEvent> release_preconditions;
      for (unsigned idx = 0; idx < mapped_instances.size(); idx++)
        release_preconditions.insert(mapped_instances[idx].get_ready_event());
      if (!wait_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::const_iterator it = 
              wait_barriers.begin(); it != wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(*it);
          release_preconditions.insert(e);
        }
      }
      if (!grants.empty())
      {
        for (std::vector<Grant>::const_iterator it = grants.begin();
              it != grants.end(); it++)
        {
          ApEvent e = it->impl->acquire_grant();
          release_preconditions.insert(e);
        }
      }
      ApEvent release_complete = Runtime::merge_events(release_preconditions);
      if (Runtime::legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, 0/*idx*/,
                                              requirement,
                                              mapped_instances);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, release_complete,
                                        completion_event);
#endif
      }
      // Chain any arrival barriers
      if (!arrive_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::const_iterator it = 
              arrive_barriers.begin(); it != arrive_barriers.end(); it++)
        {
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        completion_event);
        }
      }
      // Mark that we completed mapping
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      Runtime::trigger_event(completion_event, release_complete);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(release_complete));
      // We succeeded in mapping
      return true;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      // Don't commit this operation until the profiling is done
      commit_operation(true/*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    unsigned ReleaseOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::select_sources(const InstanceRef &target,
                                   const InstanceSet &sources,
                                   std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      Mapper::SelectReleaseSrcInput input;
      Mapper::SelectReleaseSrcOutput output;
      prepare_for_mapping(target, input.target);
      prepare_for_mapping(sources, input.source_instances);
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_select_release_sources(this, &input, &output);
      compute_ranking(output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                                     ReleaseOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    UniqueID ReleaseOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    int ReleaseOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& ReleaseOp::get_requirement(void) const
    //--------------------------------------------------------------------------
    {
      return requirement;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::check_release_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, check the privileges, but only check the
      // data and not the actual privilege values since we're
      // using psuedo-read-write-exclusive
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, true/*skip*/);
      switch (et)
      {
        // There is no such thing as bad privileges for release operations
        // because we control what they are doing
        case NO_ERROR:
        case ERROR_BAD_REGION_PRIVILEGES:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            log_region.error("Requirest for invalid region handle "
                                   "(%x,%d,%d) of requirement for "
                                   "release operation (ID %lld)",
                                   requirement.region.index_space.id, 
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id, 
                                   unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_REGION_HANDLE);
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) || 
                            (requirement.handle_type == REG_PROJECTION)
                             ? requirement.region.field_space : 
                               requirement.partition.field_space;
            log_region.error("Field %d is not a valid field of field "
                                   "space %d of requirement for release "
                                   "operation (ID %lld)",
                                   bad_field, sp.id, unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
          }
        case ERROR_BAD_PARENT_REGION:
          {
            log_region.error("Parent task %s (ID %lld) of release "
                             "operation (ID %lld) does not have a region "
                             "requirement for region (%x,%x,%x) as a parent",
                             parent_ctx->get_task_name(), 
                             parent_ctx->get_unique_id(),
                             unique_op_id, 
                             requirement.region.index_space.id,
                             requirement.region.field_space.id, 
                             requirement.region.tree_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_PARENT_REGION);
          }
        case ERROR_BAD_REGION_PATH:
          {
            log_region.error("Region (%x,%x,%x) is not a "
                                   "sub-region of parent region (%x,%x,%x) "
			           "of requirement for release "
                                   "operation (ID %lld)",
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id,
                                   requirement.parent.index_space.id,
                                   requirement.parent.field_space.id,
                                   requirement.parent.tree_id, unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_PATH);
          }
        case ERROR_BAD_REGION_TYPE:
          {
            log_region.error("Region requirement of release operation "
                                   "(ID %lld) cannot find privileges for field "
                                   "%d in parent task",
                                   unique_op_id, bad_field);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_TYPE);
          }
        // these should never happen with an release operation 
        case ERROR_INVALID_INSTANCE_FIELD:
        case ERROR_DUPLICATE_INSTANCE_FIELD:
        case ERROR_NON_DISJOINT_PARTITION: 
        default:
          assert(false); // Should never happen
      }
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::compute_parent_index(void)
    //--------------------------------------------------------------------------
    {
      int parent_index = parent_ctx->find_parent_region_req(requirement,
                                                    false/*check privilege*/);
      if (parent_index < 0)
      {
        log_region.error("Parent task %s (ID %lld) of release "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id);
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_BAD_PARENT_REGION);
      }
      else
        parent_req_index = unsigned(parent_index);
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::invoke_mapper(void)
    //--------------------------------------------------------------------------
    {
      Mapper::MapReleaseInput input;
      Mapper::MapReleaseOutput output;
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_map_release(this, &input, &output);
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::report_profiling_results(void)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_release_report_profiling(this, &profiling_results);
#ifdef DEBUG_LEGION
      assert(profiling_reported.exists());
#endif
      Runtime::trigger_event(profiling_reported);
    }

    /////////////////////////////////////////////////////////////
    // Dynamic Collective Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DynamicCollectiveOp::DynamicCollectiveOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DynamicCollectiveOp::DynamicCollectiveOp(const DynamicCollectiveOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DynamicCollectiveOp::~DynamicCollectiveOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DynamicCollectiveOp& DynamicCollectiveOp::operator=(
                                                const DynamicCollectiveOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    Future DynamicCollectiveOp::initialize(SingleTask *ctx, 
                                           const DynamicCollective &dc)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      future = Future(legion_new<FutureImpl>(runtime, true/*register*/,
            runtime->get_available_distributed_id(true), runtime->address_space,
            runtime->address_space, RtUserEvent::NO_RT_USER_EVENT, this));
      collective = dc;
      return future;
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      // Free the future
      future = Future();
      deactivate_operation();
      runtime->free_dynamic_collective_op(this);
    }

    //--------------------------------------------------------------------------
    const char* DynamicCollectiveOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[DYNAMIC_COLLECTIVE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind DynamicCollectiveOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return DYNAMIC_COLLECTIVE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    bool DynamicCollectiveOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      ApEvent barrier = Runtime::get_previous_phase(collective.phase_barrier);
      if (!barrier.has_triggered())
      {
        DeferredExecuteArgs deferred_execute_args;
        deferred_execute_args.hlr_id = HLR_DEFERRED_EXECUTION_TRIGGER_ID;
        deferred_execute_args.proxy_this = this;
        runtime->issue_runtime_meta_task(&deferred_execute_args,
                                         sizeof(deferred_execute_args),
                                         HLR_DEFERRED_EXECUTION_TRIGGER_ID,
                                         HLR_LATENCY_PRIORITY, this, 
                                         Runtime::protect_event(barrier));
      }
      else
        deferred_execute();
      complete_mapping();
      return true;
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::deferred_execute(void)
    //--------------------------------------------------------------------------
    {
      const ReductionOp *redop = Runtime::get_reduction_op(collective.redop);
      const size_t result_size = redop->sizeof_lhs;
      void *result_buffer = legion_malloc(FUTURE_RESULT_ALLOC, result_size);
      ApBarrier prev = Runtime::get_previous_phase(collective.phase_barrier);
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      bool result = 
#endif
#endif
      Runtime::get_barrier_result(prev, result_buffer, result_size);
#ifdef DEBUG_LEGION
      assert(result);
#endif
      future.impl->set_result(result_buffer, result_size, true/*own*/);
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      future.impl->complete_future();
      complete_operation();
    }

    /////////////////////////////////////////////////////////////
    // Future Predicate Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FuturePredOp::FuturePredOp(Runtime *rt)
      : PredicateOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FuturePredOp::FuturePredOp(const FuturePredOp &rhs)
      : PredicateOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never happen
      assert(false);
    }

    //--------------------------------------------------------------------------
    FuturePredOp::~FuturePredOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FuturePredOp& FuturePredOp::operator=(const FuturePredOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FuturePredOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_predicate();
    }

    //--------------------------------------------------------------------------
    void FuturePredOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_predicate();
      future = Future();
      runtime->free_future_predicate_op(this);
    }

    //--------------------------------------------------------------------------
    const char* FuturePredOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[FUTURE_PRED_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind FuturePredOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return FUTURE_PRED_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void FuturePredOp::initialize(SingleTask *ctx, Future f)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
      assert(f.impl != NULL);
#endif
      // Don't track this as it can lead to deadlock because
      // predicates can't complete until all their references from
      // the parent task have been removed.
      initialize_operation(ctx, false/*track*/);
      future = f;
    }

    //--------------------------------------------------------------------------
    void FuturePredOp::resolve_future_predicate(void)
    //--------------------------------------------------------------------------
    {
      bool valid;
      bool value = future.impl->get_boolean_value(valid);
#ifdef DEBUG_LEGION
      assert(valid);
#endif
      set_resolved_value(get_generation(), value);
    }

    //--------------------------------------------------------------------------
    void FuturePredOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(future.impl != NULL);
#endif
      begin_dependence_analysis();
      // Register this operation as dependent on task that
      // generated the future
      future.impl->register_dependence(this);
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void FuturePredOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // See if we have a value
      bool valid;
      bool value = future.impl->get_boolean_value(valid);
      if (valid)
        set_resolved_value(get_generation(), value);
      else
      {
        // Launch a task to get the value
        add_predicate_reference();
        ResolveFuturePredArgs args;
        args.hlr_id = HLR_RESOLVE_FUTURE_PRED_ID;
        args.future_pred_op = this;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_RESOLVE_FUTURE_PRED_ID,
                                         HLR_LATENCY_PRIORITY, this, 
                                         Runtime::protect_event(
                                           future.impl->get_ready_event()));
      }
      // Mark that we completed mapping this operation
      complete_mapping();
    } 

    /////////////////////////////////////////////////////////////
    // Not Predicate Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    NotPredOp::NotPredOp(Runtime *rt)
      : PredicateOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    NotPredOp::NotPredOp(const NotPredOp &rhs)
      : PredicateOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never happen
      assert(false);
    }

    //--------------------------------------------------------------------------
    NotPredOp::~NotPredOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    NotPredOp& NotPredOp::operator=(const NotPredOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void NotPredOp::initialize(SingleTask *ctx, const Predicate &p)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
#endif
      // Don't track this as it can lead to deadlock because
      // predicates can't complete until all their references from
      // the parent task have been removed.
      initialize_operation(ctx, false/*track*/);
      // Don't forget to reverse the values
      if (p == Predicate::TRUE_PRED)
        set_resolved_value(get_generation(), false);
      else if (p == Predicate::FALSE_PRED)
        set_resolved_value(get_generation(), true);
      else
      {
#ifdef DEBUG_LEGION
        assert(p.impl != NULL);
#endif
        pred_op = p.impl;
        pred_op->add_predicate_reference();
      }
    }

    //--------------------------------------------------------------------------
    void NotPredOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_predicate();
      pred_op = NULL;
    }

    //--------------------------------------------------------------------------
    void NotPredOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_predicate();
      runtime->free_not_predicate_op(this);
    }

    //--------------------------------------------------------------------------
    const char* NotPredOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[NOT_PRED_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind NotPredOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return NOT_PRED_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void NotPredOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      begin_dependence_analysis();
      if (pred_op != NULL)
        register_dependence(pred_op, pred_op->get_generation());
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void NotPredOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (pred_op != NULL)
      {
        bool prev_value;
        bool valid = pred_op->register_waiter(this, get_generation(),
                                              prev_value);
        // Don't forget to negate 
        if (valid)
          set_resolved_value(get_generation(), !prev_value);
        // Now we can remove the reference we added
        pred_op->remove_predicate_reference();
      }
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    void NotPredOp::notify_predicate_value(GenerationID prev_gen, bool value)
    //--------------------------------------------------------------------------
    {
      // No short circuit in this one
      // We can test this without the lock because 
      // it is monotonically increasing
#ifdef DEBUG_LEGION
      assert(prev_gen == get_generation());
#endif
      // Don't forget to negate the value
      set_resolved_value(prev_gen, !value);
    }

    /////////////////////////////////////////////////////////////
    // And Predicate Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AndPredOp::AndPredOp(Runtime *rt)
      : PredicateOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AndPredOp::AndPredOp(const AndPredOp &rhs)
      : PredicateOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never happen
      assert(false);
    }

    //--------------------------------------------------------------------------
    AndPredOp::~AndPredOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AndPredOp& AndPredOp::operator=(const AndPredOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void AndPredOp::initialize(SingleTask *ctx,
                               const Predicate &p1, 
                               const Predicate &p2)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
#endif
      // Don't track this as it can lead to deadlock because
      // predicates can't complete until all their references from
      // the parent task have been removed.
      initialize_operation(ctx, false/*track*/);
      // Short circuit case
      if ((p1 == Predicate::FALSE_PRED) || 
          (p2 == Predicate::FALSE_PRED))
      {
        set_resolved_value(get_generation(), false);
        return;
      }
      if (p1 == Predicate::TRUE_PRED)
      {
        left_value = true;
        left_valid = true;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(p1.impl != NULL);
#endif
        left_valid = false;
        left = p1.impl;
        left->add_predicate_reference();
      }
      if (p2 == Predicate::TRUE_PRED)
      {
        right_value = true;
        right_valid = true;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(p2.impl != NULL);
#endif
        right_valid = false;
        right = p2.impl;
        right->add_predicate_reference();
      }
    }

    //--------------------------------------------------------------------------
    void AndPredOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_predicate();
      left = NULL;
      right = NULL;
      left_valid = false;
      right_valid = false;
    }

    //--------------------------------------------------------------------------
    void AndPredOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_predicate();
      runtime->free_and_predicate_op(this);
    }

    //--------------------------------------------------------------------------
    const char* AndPredOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[AND_PRED_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind AndPredOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return AND_PRED_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void AndPredOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      begin_dependence_analysis();
      if (left != NULL)
        register_dependence(left, left->get_generation());
      if (left != NULL)
        register_dependence(right, right->get_generation());
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void AndPredOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Hold the lock when doing this to prevent 
      // any triggers from interfering with the analysis
      bool need_resolve = false;
      bool resolve_value;
      GenerationID local_gen = get_generation();
      {
        AutoLock o_lock(op_lock);
        if (!predicate_resolved)
        {
          if (left != NULL)
            left_valid = left->register_waiter(this, get_generation(),
                                               left_value);
          if (right != NULL)
            right_valid = right->register_waiter(this, get_generation(),
                                                 right_value);
          // Both valid
          if (left_valid && right_valid)
          {
            need_resolve = true;
            resolve_value = (left_value && right_value);
          }
          // Left short circuit
          else if (left_valid && !left_value)
          {
            need_resolve = true;
            resolve_value = false;
          }
          // Right short circuit
          else if (right_valid && !right_value) 
          {
            need_resolve = true;
            resolve_value = false;
          }
        }
      }
      if (need_resolve)
        set_resolved_value(local_gen, resolve_value);
      // Clean up any references that we have
      if (left != NULL)
        left->remove_predicate_reference();
      if (right != NULL)
        right->remove_predicate_reference();
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    void AndPredOp::notify_predicate_value(GenerationID pred_gen, bool value)
    //--------------------------------------------------------------------------
    {
      bool need_resolve = false, resolve_value = false;
      if (pred_gen == get_generation())
      {
        AutoLock o_lock(op_lock);
        // Check again to make sure we didn't lose the race
        if ((pred_gen == get_generation()) && !predicate_resolved)
        {
          if (!value)
          {
            need_resolve = true;
            resolve_value = false;
          }
          else
          {
            // Figure out which of the two values to fill in
#ifdef DEBUG_LEGION
            assert(!left_valid || !right_valid);
#endif
            if (!left_valid)
            {
              left_value = value;
              left_valid = true;
            }
            else
            {
              right_value = value;
              right_valid = true;
            }
            if (left_valid && right_valid)
            {
              need_resolve = true;
              resolve_value = (left_value && right_value);
            }
          }
        }
        else
          need_resolve = false;
      }
      else
        need_resolve = false;
      if (need_resolve)
        set_resolved_value(pred_gen, resolve_value);
    }

    /////////////////////////////////////////////////////////////
    // Or Predicate Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OrPredOp::OrPredOp(Runtime *rt)
      : PredicateOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OrPredOp::OrPredOp(const OrPredOp &rhs)
      : PredicateOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never happen
      assert(false);
    }

    //--------------------------------------------------------------------------
    OrPredOp::~OrPredOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OrPredOp& OrPredOp::operator=(const OrPredOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void OrPredOp::initialize(SingleTask *ctx,
                              const Predicate &p1, 
                              const Predicate &p2)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
#endif
      // Don't track this as it can lead to deadlock because
      // predicates can't complete until all their references from
      // the parent task have been removed.
      initialize_operation(ctx, false/*track*/);
      // Short circuit case
      if ((p1 == Predicate::TRUE_PRED) || 
          (p2 == Predicate::TRUE_PRED))
      {
        set_resolved_value(get_generation(), true);
        return;
      }
      if (p1 == Predicate::FALSE_PRED)
      {
        left_value = false;
        left_valid = true;
      }
      else
      {
        left = p1.impl;
        left_valid = false;
        left->add_predicate_reference();
      }
      if (p2 == Predicate::FALSE_PRED)
      {
        right_value = false;
        right_valid = true;
      }
      else
      {
        right = p2.impl;
        right_valid = false;
        right->add_predicate_reference();
      }
    }

    //--------------------------------------------------------------------------
    void OrPredOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_predicate();
      left = NULL;
      right = NULL;
      left_valid = false;
      right_valid = false;
    }

    //--------------------------------------------------------------------------
    void OrPredOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_predicate();
      runtime->free_or_predicate_op(this);
    }

    //--------------------------------------------------------------------------
    const char* OrPredOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[OR_PRED_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind OrPredOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return OR_PRED_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void OrPredOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      begin_dependence_analysis();
      if (left != NULL)
        register_dependence(left, left->get_generation());
      if (right != NULL)
        register_dependence(right, right->get_generation());
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void OrPredOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Hold the lock when doing this to prevent 
      // any triggers from interfering with the analysis
      bool need_resolve = false;
      bool resolve_value;
      GenerationID local_gen = get_generation();
      {
        AutoLock o_lock(op_lock);
        if (!predicate_resolved)
        {
          if (left != NULL)
            left_valid = left->register_waiter(this, get_generation(),
                                               left_value);
          if (right != NULL)
            right_valid = right->register_waiter(this, get_generation(),
                                                 right_value);
          // Both valid
          if (left_valid && right_valid)
          {
            need_resolve = true;
            resolve_value = (left_value || right_value);
          }
          // Left short circuit
          else if (left_valid && left_value)
          {
            need_resolve = true;
            resolve_value = true;
          }
          // Right short circuit
          else if (right_valid && right_value) 
          {
            need_resolve = true;
            resolve_value = true;
          }
        }
      }
      if (need_resolve)
        set_resolved_value(local_gen, resolve_value);
      // Clean up any references that we have
      if (left != NULL)
        left->remove_predicate_reference();
      if (right != NULL)
        right->remove_predicate_reference();
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    void OrPredOp::notify_predicate_value(GenerationID pred_gen, bool value)
    //--------------------------------------------------------------------------
    {
      bool need_resolve = false, resolve_value = false;
      if (pred_gen == get_generation())
      {
        AutoLock o_lock(op_lock);
        // Check again to make sure we didn't lose the race
        if ((pred_gen == get_generation()) && !predicate_resolved)
        {
          if (value)
          {
            need_resolve = true;
            resolve_value = true;
          }
          else
          {
            // Figure out which of the two values to fill in
#ifdef DEBUG_LEGION
            assert(!left_valid || !right_valid);
#endif
            if (!left_valid)
            {
              left_value = value;
              left_valid = true;
            }
            else
            {
              right_value = value;
              right_valid = true;
            }
            if (left_valid && right_valid)
            {
              need_resolve = true;
              resolve_value = (left_value || right_value);
            }
          }
        }
        else
          need_resolve = false;
      }
      else
        need_resolve = false;
      if (need_resolve)
        set_resolved_value(pred_gen, resolve_value);
    }


    /////////////////////////////////////////////////////////////
    // Must Epoch Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochOp::MustEpochOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MustEpochOp::MustEpochOp(const MustEpochOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MustEpochOp::~MustEpochOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MustEpochOp& MustEpochOp::operator=(const MustEpochOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    FutureMap MustEpochOp::initialize(SingleTask *ctx,
                                              const MustEpochLauncher &launcher,
                                              bool check_privileges)
    //--------------------------------------------------------------------------
    {
      // Initialize this operation
      initialize_operation(ctx, true/*track*/);
      // Initialize operations for everything in the launcher
      // Note that we do not track these operations as we want them all to
      // appear as a single operation to the parent context in order to
      // avoid deadlock with the maximum window size.
      indiv_tasks.resize(launcher.single_tasks.size());
      for (unsigned idx = 0; idx < launcher.single_tasks.size(); idx++)
      {
        indiv_tasks[idx] = runtime->get_available_individual_task(true);
        indiv_tasks[idx]->initialize_task(ctx, launcher.single_tasks[idx],
                                          check_privileges, false/*track*/);
        indiv_tasks[idx]->set_must_epoch(this, idx, true/*register*/);
        // If we have a trace, set it for this operation as well
        if (trace != NULL)
          indiv_tasks[idx]->set_trace(trace, !trace->is_fixed());
        indiv_tasks[idx]->must_epoch_task = true;
      }
      indiv_triggered.resize(indiv_tasks.size(), false);
      index_tasks.resize(launcher.index_tasks.size());
      for (unsigned idx = 0; idx < launcher.index_tasks.size(); idx++)
      {
        index_tasks[idx] = runtime->get_available_index_task(true);
        index_tasks[idx]->initialize_task(ctx, launcher.index_tasks[idx],
                                          check_privileges, false/*track*/);
        index_tasks[idx]->set_must_epoch(this, indiv_tasks.size()+idx, 
                                         true/*register*/);
        if (trace != NULL)
          index_tasks[idx]->set_trace(trace, !trace->is_fixed());
        index_tasks[idx]->must_epoch_task = true;
      }
      index_triggered.resize(index_tasks.size(), false);
      mapper_id = launcher.map_id;
      mapper_tag = launcher.mapping_tag;
      // Make a new future map for storing our results
      // We'll fill it in later
      result_map = FutureMap(legion_new<FutureMapImpl>(ctx, 
                                             get_completion_event(), runtime));
#ifdef DEBUG_LEGION
      size_t total_points = 0;
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
      {
        result_map.impl->add_valid_point(indiv_tasks[idx]->index_point);
        total_points++;
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        result_map.impl->add_valid_domain(index_tasks[idx]->index_domain);
        total_points += index_tasks[idx]->index_domain.get_volume();
      }
      // Assume for now that all msut epoch launches have to be
      // mapped to CPUs
      Machine::ProcessorQuery all_cpus(runtime->machine);
      all_cpus.only_kind(Processor::LOC_PROC); 
      if (total_points > all_cpus.count())
      {
        log_run.error("Illegal must epoch launch in task %s (UID %lld). "
            "Must epoch launch requested %ld tasks, but only %ld CPUs "
            "exist in this machine.", parent_ctx->get_task_name(),
            parent_ctx->get_unique_id(), total_points, all_cpus.count());
        assert(false);
      }
#endif
      return result_map;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::find_conflicted_regions(
                                 std::vector<PhysicalRegion> &conflicts)
    //--------------------------------------------------------------------------
    {
      // Dump them all into a set when they are done to deduplicate them
      // This is not the most optimized way to do this, but it will work for now
      std::set<PhysicalRegion> temp_conflicts;
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
      {
        std::vector<PhysicalRegion> temp;
        parent_ctx->find_conflicting_regions(indiv_tasks[idx], temp);
        temp_conflicts.insert(temp.begin(),temp.end());
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        std::vector<PhysicalRegion> temp;
        parent_ctx->find_conflicting_regions(index_tasks[idx], temp);
        temp_conflicts.insert(temp.begin(),temp.end());
      }
      conflicts.insert(conflicts.end(),
                       temp_conflicts.begin(),temp_conflicts.end());
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      mapper_id = 0;
      mapper_tag = 0;
      // Set to 1 to include the triggers we get for our operation
      remaining_subop_completes = 1;
      remaining_subop_commits = 1;
      triggering_complete = false;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      // All the sub-operations we have will deactivate themselves
      indiv_tasks.clear();
      indiv_triggered.clear();
      index_tasks.clear();
      index_triggered.clear();
      slice_tasks.clear();
      single_tasks.clear();
      // Remove our reference on the future map
      result_map = FutureMap();
      constraints.clear();
      task_sets.clear();
#ifdef DEBUG_LEGION
      assert(acquired_instances.empty());
#endif
      acquired_instances.clear();
      dependence_map.clear();
      for (std::vector<DependenceRecord*>::iterator it = dependences.begin();
            it != dependences.end(); it++)
      {
        delete (*it);
      }
      dependences.clear();
      single_task_map.clear();
      input.tasks.clear();
      input.constraints.clear();
      output.task_processors.clear();
      output.constraint_mappings.clear();
      // Return this operation to the free list
      runtime->free_epoch_op(this);
    }

    //--------------------------------------------------------------------------
    const char* MustEpochOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[MUST_EPOCH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind MustEpochOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return MUST_EPOCH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t MustEpochOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      for (std::vector<IndividualTask*>::const_iterator it = 
            indiv_tasks.begin(); it != indiv_tasks.end(); it++)
      {
        result += (*it)->get_region_count();
      }
      for (std::vector<IndexTask*>::const_iterator it = 
            index_tasks.begin(); it != index_tasks.end(); it++)
      {
        result += (*it)->get_region_count();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      begin_dependence_analysis();
      // For every one of our sub-operations, add an additional mapping 
      // dependence.  When our sub-operations map, they will trigger these
      // mapping dependences which guarantees that we will not be able to
      // map until all of the sub-operations are ready to map.
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
        indiv_tasks[idx]->trigger_dependence_analysis();
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
        index_tasks[idx]->trigger_dependence_analysis();
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::trigger_remote_state_analysis(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
      {
        RtUserEvent indiv_event = Runtime::create_rt_user_event();
        indiv_tasks[idx]->trigger_remote_state_analysis(indiv_event);
        preconditions.insert(indiv_event);
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        RtUserEvent index_event = Runtime::create_rt_user_event();
        index_tasks[idx]->trigger_remote_state_analysis(index_event);
        preconditions.insert(index_event);
      }
      Runtime::trigger_event(ready_event,
                             Runtime::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    bool MustEpochOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      // First mark that each of the tasks will be locally mapped
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
        indiv_tasks[idx]->set_locally_mapped(true);
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
        index_tasks[idx]->set_locally_mapped(true);
      // Call trigger execution on each of our sub-operations, since they
      // each have marked that they have a must_epoch owner, they will
      // not actually map and launch, but instead will register all the base
      // operations with us.  Note this step requires that we mark everything
      // as needing to locally map in the 'initialize' method.  Check for
      // error codes indicating failed pre-mapping.
      if (!triggering_complete)
      {
        task_sets.resize(indiv_tasks.size()+index_tasks.size());
        MustEpochTriggerer triggerer(this);
        if (!triggerer.trigger_tasks(indiv_tasks, indiv_triggered,
                                     index_tasks, index_triggered))
          return false;

#ifdef DEBUG_LEGION
        assert(!single_tasks.empty());
#endif 
        // Next build the set of single tasks and all their constraints.
        // Iterate over all the recorded dependences
        std::vector<Mapper::MappingConstraint> &constraints = input.constraints;
        constraints.resize(dependences.size());
        // Clear the dependence map now, we'll fill it in again
        // with a different set of points
        dependence_map.clear();
        unsigned constraint_idx = 0;
        for (std::vector<DependenceRecord*>::const_iterator it = 
              dependences.begin(); it != dependences.end(); 
              it++, constraint_idx++)
        {
          Mapper::MappingConstraint &constraint = constraints[constraint_idx];
#ifdef DEBUG_LEGION
          assert((*it)->op_indexes.size() == (*it)->req_indexes.size());
#endif
          // Add constraints for all the different elements
          for (unsigned idx = 0; idx < (*it)->op_indexes.size(); idx++)
          {
            unsigned req_index = (*it)->req_indexes[idx];
            const std::set<SingleTask*> &task_set = 
                        task_sets[(*it)->op_indexes[idx]];
            for (std::set<SingleTask*>::const_iterator sit = task_set.begin();
                  sit != task_set.end(); sit++)
            {
              constraint.constrained_tasks.push_back(*sit);
              constraint.requirement_indexes.push_back(req_index);
#ifdef DEBUG_LEGION
              assert(single_task_map.find(*sit) != single_task_map.end());
#endif
              // Update the dependence map
              std::pair<unsigned,unsigned> key(single_task_map[*sit],req_index);
              dependence_map[key] = constraint_idx;
            }
          }
        }
        // Clear this eagerly to save space
        for (std::vector<DependenceRecord*>::const_iterator it = 
              dependences.begin(); it != dependences.end(); it++)
        {
          delete (*it);
        }
        dependences.clear();
        // Mark that we have finished building all the constraints so
        // we don't have to redo it if we end up failing a mapping.
        triggering_complete = true;
      }
      // Fill in the rest of the inputs to the mapper call
      input.mapping_tag = mapper_tag;
      input.tasks.insert(input.tasks.end(), single_tasks.begin(),
                                            single_tasks.end());
      // Also resize the outputs so the mapper knows what it is doing
      output.constraint_mappings.resize(input.constraints.size());
      output.task_processors.resize(single_tasks.size(), Processor::NO_PROC);
      Processor mapper_proc = parent_ctx->get_executing_processor();
      MapperManager *mapper = runtime->find_mapper(mapper_proc, mapper_id);
      // We've got all our meta-data set up so go ahead and issue the call
      mapper->invoke_map_must_epoch(this, &input, &output);
      // Check that all the tasks have been assigned to different processors
      {
        std::map<Processor,SingleTask*> target_procs;
        for (unsigned idx = 0; idx < single_tasks.size(); idx++)
        {
          Processor proc = output.task_processors[idx];
          SingleTask *task = single_tasks[idx];
          if (!proc.exists())
          {
            log_run.error("Invalid mapper output from invocation of "
                "'map_must_epoch' on mapper %s. Mapper failed to specify "
                "a valid processor for task %s (ID %lld) at index %d. Call "
                "occurred in parent task %s (ID %lld).", 
                mapper->get_mapper_name(), task->get_task_name(),
                task->get_unique_id(), idx, parent_ctx->get_task_name(),
                parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
          if (target_procs.find(proc) != target_procs.end())
          {
            SingleTask *other = target_procs[proc];
            log_run.error("Invalid mapper output from invocation of "
                "'map_must_epoch' on mapper %s. Mapper requests both tasks "
                "%s (ID %lld) and %s (ID %lld) be mapped to the same "
                "processor (" IDFMT ") which is illegal in a must epoch "
                "launch. Must epoch was launched inside of task %s (ID %lld).",
                mapper->get_mapper_name(), other->get_task_name(),
                other->get_unique_id(), task->get_task_name(),
                task->get_unique_id(), proc.id, parent_ctx->get_task_name(),
                parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          } 
          target_procs[proc] = task;
          task->target_proc = proc;
        }
      }
      // Then we need to actually perform the mapping
      {
        MustEpochMapper mapper(this); 
        if (!mapper.map_tasks(single_tasks))
          return false;
      }

      // Once all the tasks have been initialized we can defer
      // our all mapped event on all their all mapped events
      std::set<RtEvent> tasks_all_mapped;
      std::set<ApEvent> tasks_all_complete;
      for (std::vector<IndividualTask*>::const_iterator it = 
            indiv_tasks.begin(); it != indiv_tasks.end(); it++)
      {
        tasks_all_mapped.insert((*it)->get_mapped_event());
        tasks_all_complete.insert((*it)->get_completion_event());
      }
      for (std::vector<IndexTask*>::const_iterator it = 
            index_tasks.begin(); it != index_tasks.end(); it++)
      {
        tasks_all_mapped.insert((*it)->get_mapped_event());
        tasks_all_complete.insert((*it)->get_completion_event());
      }
      // If we passed all the constraints, then kick everything off
      MustEpochDistributor distributor(this);
      distributor.distribute_tasks(runtime, indiv_tasks, slice_tasks); 
      
      // Mark that we are done mapping and executing this operation
      RtEvent all_mapped = Runtime::merge_events(tasks_all_mapped);
      RtEvent all_complete = Runtime::protect_merge_events(tasks_all_complete);
      complete_mapping(all_mapped);
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      complete_execution(all_complete);
      return true;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      bool need_complete;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(remaining_subop_completes > 0);
#endif
        remaining_subop_completes--;
        need_complete = (remaining_subop_completes == 0);
      }
      if (need_complete)
      {
        result_map.impl->complete_all_futures();
        complete_operation();
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      bool need_commit;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(remaining_subop_commits > 0);
#endif
        remaining_subop_commits--;
        need_commit = (remaining_subop_commits == 0);
      }
      if (need_commit)
        commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::verify_dependence(Operation *src_op, GenerationID src_gen,
                                        Operation *dst_op, GenerationID dst_gen)
    //--------------------------------------------------------------------------
    {
      // If they are the same, then we can ignore them
      if ((src_op == dst_op) && (src_gen == dst_gen))
        return;
      // Check to see if the source is one of our operations, if it is
      // then we have an actual dependence which is an error.
      int src_index = find_operation_index(src_op, src_gen);
      if (src_index >= 0)
      {
        int dst_index = find_operation_index(dst_op, dst_gen);
#ifdef DEBUG_LEGION
        assert(dst_index >= 0);
#endif
        TaskOp *src_task = find_task_by_index(src_index);
        TaskOp *dst_task = find_task_by_index(dst_index);
        log_run.error("MUST EPOCH ERROR: dependence between task "
            "%s (ID %lld) and task %s (ID %lld)\n",
            src_task->get_task_name(), src_task->get_unique_id(),
            dst_task->get_task_name(), dst_task->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_MUST_EPOCH_FAILURE);
      }
    }
    
    //--------------------------------------------------------------------------
    bool MustEpochOp::record_dependence(Operation *src_op, GenerationID src_gen,
                                        Operation *dst_op, GenerationID dst_gen,
                                        unsigned src_idx, unsigned dst_idx,
                                        DependenceType dtype)
    //--------------------------------------------------------------------------
    {
      // If they are the same we can ignore them 
      if ((src_op == dst_op) && (src_gen == dst_gen))
        return true;
      // Check to see if the source is one of our operations
      int src_index = find_operation_index(src_op, src_gen);
      int dst_index = find_operation_index(dst_op, dst_gen);
      if ((src_index >= 0) && (dst_index >= 0))
      {
        // If it is, see what kind of dependence we have
        if ((dtype == TRUE_DEPENDENCE) || (dtype == ANTI_DEPENDENCE) ||
            (dtype == ATOMIC_DEPENDENCE))
        {
          TaskOp *src_task = find_task_by_index(src_index);
          TaskOp *dst_task = find_task_by_index(dst_index);
          log_run.error("MUST EPOCH ERROR: dependence between region %d "
              "of task %s (ID %lld) and region %d of task %s (ID %lld) of "
              " type %s", src_idx, src_task->get_task_name(),
              src_task->get_unique_id(), dst_idx, 
              dst_task->get_task_name(), dst_task->get_unique_id(),
              (dtype == TRUE_DEPENDENCE) ? "TRUE DEPENDENCE" :
                (dtype == ANTI_DEPENDENCE) ? "ANTI DEPENDENCE" :
                "ATOMIC DEPENDENCE");
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_MUST_EPOCH_FAILURE);
        }
        else if (dtype == SIMULTANEOUS_DEPENDENCE)
        {
          // Record the dependence kind
          int dst_index = find_operation_index(dst_op, dst_gen);
#ifdef DEBUG_LEGION
          assert(dst_index >= 0);
#endif
          // See if the dependence record already exists
          std::pair<unsigned,unsigned> src_key(src_index,src_idx);
          std::pair<unsigned,unsigned> dst_key(dst_index,dst_idx);
          std::map<std::pair<unsigned,unsigned>,unsigned>::iterator
            record_finder = dependence_map.find(dst_key);
          if (record_finder == dependence_map.end())
          {
#ifdef DEBUG_LEGION
            assert(dependence_map.find(dst_key) == dependence_map.end());
#endif
            // We have to make new record
            DependenceRecord *new_record = new DependenceRecord();
            new_record->add_entry(src_index, src_idx);
            new_record->add_entry(dst_index, dst_idx);
            unsigned record_index = dependences.size();
            dependence_map[src_key] = record_index;
            dependence_map[dst_key] = record_index;
            dependences.push_back(new_record);
          }
          else
          {
            // Just add the source to the collection
            dependences[record_finder->second]->add_entry(src_index, src_idx);
            dependence_map[src_key] = record_finder->second;
          }
          return false;
        }
        // NO_DEPENDENCE and PROMOTED_DEPENDENCE are not errors
        // and do not need to be recorded
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::must_epoch_map_task_callback(SingleTask *task,
                                              Mapper::MapTaskInput &map_input,
                                              Mapper::MapTaskOutput &map_output)
    //--------------------------------------------------------------------------
    {
      // We have to do three things here
      // 1. Update the target processor
      // 2. Mark as inputs and outputs any regions which we know
      //    the results for as a result of our must epoch mapping
      // 3. Record that we premapped those regions
      // First find the index for this task
#ifdef DEBUG_LEGION
      assert(single_task_map.find(task) != single_task_map.end());
#endif
      unsigned index = single_task_map[task];
      // Set the target processor by the index 
      task->target_proc = output.task_processors[index]; 
      // Now iterate over the constraints figure out which ones
      // apply to this task
      std::pair<unsigned,unsigned> key(index,0);
      for (unsigned idx = 0; idx < task->regions.size(); idx++)
      {
        key.second = idx;
        std::map<std::pair<unsigned,unsigned>,unsigned>::const_iterator
          record_finder = dependence_map.find(key);
        if (record_finder != dependence_map.end())
        {
          map_input.valid_instances[idx] = 
            output.constraint_mappings[record_finder->second];
          map_output.chosen_instances[idx] = 
            output.constraint_mappings[record_finder->second];
          // Also record that we premapped this
          map_input.premapped_regions.push_back(idx);
        }
      }
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                                   MustEpochOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances; 
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::add_mapping_dependence(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dependence_tracker.mapping != NULL);
#endif
      dependence_tracker.mapping->add_mapping_dependence(precondition);
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::register_single_task(SingleTask *single, unsigned index)
    //--------------------------------------------------------------------------
    {
      // Can do the first part without the lock 
#ifdef DEBUG_LEGION
      assert(index < task_sets.size());
#endif
      task_sets[index].insert(single);
      AutoLock o_lock(op_lock);
      const unsigned single_task_index = single_tasks.size();
      single_tasks.push_back(single);
      single_task_map[single] = single_task_index;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::register_slice_task(SliceTask *slice)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      slice_tasks.insert(slice);
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::set_future(const DomainPoint &point, const void *result, 
                                 size_t result_size, bool owner)
    //--------------------------------------------------------------------------
    {
      Future f = result_map.impl->get_future(point);
      f.impl->set_result(result, result_size, owner);
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::unpack_future(const DomainPoint &point, 
                                    Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      Future f = result_map.impl->get_future(point);
      f.impl->unpack_future(derez);
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::register_subop(Operation *op)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      remaining_subop_completes++;
      remaining_subop_commits++;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::notify_subop_complete(Operation *op)
    //--------------------------------------------------------------------------
    {
      bool need_complete;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(remaining_subop_completes > 0);
#endif
        remaining_subop_completes--;
        need_complete = (remaining_subop_completes == 0);
      }
      if (need_complete)
      {
        // Complete all our futures
        result_map.impl->complete_all_futures();
        complete_operation();
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::notify_subop_commit(Operation *op)
    //--------------------------------------------------------------------------
    {
      bool need_commit;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(remaining_subop_commits > 0);
#endif
        remaining_subop_commits--;
        need_commit = (remaining_subop_commits == 0);
      }
      if (need_commit)
        commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    int MustEpochOp::find_operation_index(Operation *op, GenerationID op_gen)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
      {
        if ((indiv_tasks[idx] == op) && 
            (indiv_tasks[idx]->get_generation() == op_gen))
          return idx;
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        if ((index_tasks[idx] == op) &&
            (index_tasks[idx]->get_generation() == op_gen))
          return (idx+indiv_tasks.size());
      }
      return -1;
    }

    //--------------------------------------------------------------------------
    TaskOp* MustEpochOp::find_task_by_index(int index)
    //--------------------------------------------------------------------------
    {
      assert(index >= 0);
      if ((size_t)index < indiv_tasks.size())
        return indiv_tasks[index];
      index -= indiv_tasks.size();
      if ((size_t)index < index_tasks.size())
        return index_tasks[index];
      assert(false);
      return NULL;
    }

    /////////////////////////////////////////////////////////////
    // Must Epoch Triggerer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochTriggerer::MustEpochTriggerer(MustEpochOp *own)
      : owner(own)
    //--------------------------------------------------------------------------
    {
      trigger_lock = Reservation::create_reservation();
    }

    //--------------------------------------------------------------------------
    MustEpochTriggerer::MustEpochTriggerer(const MustEpochTriggerer &rhs)
      : owner(rhs.owner)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MustEpochTriggerer::~MustEpochTriggerer(void)
    //--------------------------------------------------------------------------
    {
      trigger_lock.destroy_reservation();
      trigger_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    MustEpochTriggerer& MustEpochTriggerer::operator=(
                                                  const MustEpochTriggerer &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool MustEpochTriggerer::trigger_tasks(
                                const std::vector<IndividualTask*> &indiv_tasks,
                                std::vector<bool> &indiv_triggered,
                                const std::vector<IndexTask*> &index_tasks,
                                std::vector<bool> &index_triggered)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> wait_events;
      for (unsigned idx = 0; idx < indiv_triggered.size(); idx++)
      {
        if (!indiv_triggered[idx])
        {
          MustEpochIndivArgs args;
          args.hlr_id = HLR_MUST_INDIV_ID;
          args.triggerer = this;
          args.task = indiv_tasks[idx];
          RtEvent wait = 
            owner->runtime->issue_runtime_meta_task(&args, sizeof(args), 
                                                    HLR_MUST_INDIV_ID, 
                                                    HLR_THROUGHPUT_PRIORITY,
                                                    owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        if (!index_triggered[idx])
        {
          MustEpochIndexArgs args;
          args.hlr_id = HLR_MUST_INDEX_ID;
          args.triggerer = this;
          args.task = index_tasks[idx];
          RtEvent wait = 
            owner->runtime->issue_runtime_meta_task(&args, sizeof(args), 
                                                    HLR_MUST_INDEX_ID, 
                                                    HLR_THROUGHPUT_PRIORITY,
                                                    owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
      }
      // Wait for all of the launches to be done
      // We can safely block to free up the utility processor
      if (!wait_events.empty())
      {
        RtEvent trigger_event = Runtime::merge_events(wait_events);
        trigger_event.wait();
      }
      // Now see if any failed
      // Otherwise mark which ones succeeded
      if (!failed_individual_tasks.empty())
      {
        for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
        {
          if (indiv_triggered[idx])
            continue;
          if (failed_individual_tasks.find(indiv_tasks[idx]) ==
              failed_individual_tasks.end())
            indiv_triggered[idx] = true;
        }
      }
      if (!failed_index_tasks.empty())
      {
        for (unsigned idx = 0; idx < index_tasks.size(); idx++)
        {
          if (index_triggered[idx])
            continue;
          if (failed_index_tasks.find(index_tasks[idx]) ==
              failed_index_tasks.end())
            index_triggered[idx] = true;
        }
      }
      return (failed_individual_tasks.empty() && failed_index_tasks.empty());
    }

    //--------------------------------------------------------------------------
    void MustEpochTriggerer::trigger_individual(IndividualTask *task)
    //--------------------------------------------------------------------------
    {
      if (!task->trigger_execution())
      {
        AutoLock t_lock(trigger_lock);
        failed_individual_tasks.insert(task);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochTriggerer::trigger_index(IndexTask *task)
    //--------------------------------------------------------------------------
    {
      if (!task->trigger_execution())
      {
        AutoLock t_lock(trigger_lock);
        failed_index_tasks.insert(task);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void MustEpochTriggerer::handle_individual(const void *args)
    //--------------------------------------------------------------------------
    {
      const MustEpochIndivArgs *indiv_args = (const MustEpochIndivArgs*)args;
      indiv_args->triggerer->trigger_individual(indiv_args->task);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MustEpochTriggerer::handle_index(const void *args)
    //--------------------------------------------------------------------------
    {
      const MustEpochIndexArgs *index_args = (const MustEpochIndexArgs*)args;
      index_args->triggerer->trigger_index(index_args->task);
    }

    /////////////////////////////////////////////////////////////
    // Must Epoch Mapper 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochMapper::MustEpochMapper(MustEpochOp *own)
      : owner(own), success(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MustEpochMapper::MustEpochMapper(const MustEpochMapper &rhs)
      : owner(rhs.owner)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MustEpochMapper::~MustEpochMapper(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MustEpochMapper& MustEpochMapper::operator=(const MustEpochMapper &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool MustEpochMapper::map_tasks(const std::deque<SingleTask*> &single_tasks)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> wait_events;   
      MustEpochMapArgs args;
      args.hlr_id = HLR_MUST_MAP_ID;
      args.mapper = this;
      for (std::deque<SingleTask*>::const_iterator it = single_tasks.begin();
            it != single_tasks.end(); it++)
      {
        args.task = *it;
        RtEvent wait = 
          owner->runtime->issue_runtime_meta_task(&args, sizeof(args), 
                                                  HLR_MUST_MAP_ID, 
                                                  HLR_THROUGHPUT_PRIORITY,
                                                  owner);
        if (wait.exists())
          wait_events.insert(wait);
      }
      if (!wait_events.empty())
      {
        RtEvent mapped_event = Runtime::merge_events(wait_events);
        mapped_event.wait();
      }
#ifdef DEBUG_LEGION
      assert(success); // should always succeed now
#endif
      return success;
    }

    //--------------------------------------------------------------------------
    void MustEpochMapper::map_task(SingleTask *task)
    //--------------------------------------------------------------------------
    {
      // Note we don't need to hold a lock here because this is
      // a monotonic change.  Once it fails for anyone then it
      // fails for everyone.
      if (!task->perform_mapping(owner))
        success = false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void MustEpochMapper::handle_map_task(const void *args)
    //--------------------------------------------------------------------------
    {
      const MustEpochMapArgs *map_args = (const MustEpochMapArgs*)args;
      map_args->mapper->map_task(map_args->task);
    }

    /////////////////////////////////////////////////////////////
    // Must Epoch Distributor 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochDistributor::MustEpochDistributor(MustEpochOp *own)
      : owner(own)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MustEpochDistributor::MustEpochDistributor(const MustEpochDistributor &rhs)
      : owner(rhs.owner)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MustEpochDistributor::~MustEpochDistributor(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MustEpochDistributor& MustEpochDistributor::operator=(
                                                const MustEpochDistributor &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void MustEpochDistributor::distribute_tasks(Runtime *runtime,
                                const std::vector<IndividualTask*> &indiv_tasks,
                                const std::set<SliceTask*> &slice_tasks)
    //--------------------------------------------------------------------------
    {
      MustEpochDistributorArgs dist_args;
      dist_args.hlr_id = HLR_MUST_DIST_ID;
      MustEpochLauncherArgs launch_args;
      launch_args.hlr_id = HLR_MUST_LAUNCH_ID;
      std::set<RtEvent> wait_events;
      for (std::vector<IndividualTask*>::const_iterator it = 
            indiv_tasks.begin(); it != indiv_tasks.end(); it++)
      {
        if (!runtime->is_local((*it)->target_proc))
        {
          dist_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(&dist_args, sizeof(dist_args), 
                                             HLR_MUST_DIST_ID, 
                                             HLR_THROUGHPUT_PRIORITY, owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
        else
        {
          launch_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(&launch_args, sizeof(launch_args), 
                                             HLR_MUST_LAUNCH_ID, 
                                             HLR_THROUGHPUT_PRIORITY, owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
      }
      for (std::set<SliceTask*>::const_iterator it = 
            slice_tasks.begin(); it != slice_tasks.end(); it++)
      {
        if (!runtime->is_local((*it)->target_proc))
        {
          dist_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(&dist_args, sizeof(dist_args), 
                                             HLR_MUST_DIST_ID, 
                                             HLR_THROUGHPUT_PRIORITY, owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
        else
        {
          launch_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(&launch_args, sizeof(launch_args), 
                                             HLR_MUST_LAUNCH_ID, 
                                             HLR_THROUGHPUT_PRIORITY, owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
      }
      if (!wait_events.empty())
      {
        RtEvent dist_event = Runtime::merge_events(wait_events);
        dist_event.wait();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void MustEpochDistributor::handle_distribute_task(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const MustEpochDistributorArgs *dist_args = 
        (const MustEpochDistributorArgs*)args;
      dist_args->task->distribute_task();
    }

    //--------------------------------------------------------------------------
    /*static*/ void MustEpochDistributor::handle_launch_task(const void *args)
    //--------------------------------------------------------------------------
    {
      const MustEpochLauncherArgs *launch_args = 
        (const MustEpochLauncherArgs *)args;
      launch_args->task->launch_task();
    }

    /////////////////////////////////////////////////////////////
    // Pending Partition Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PendingPartitionOp::PendingPartitionOp(Runtime *rt)
      : Operation(rt), thunk(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PendingPartitionOp::PendingPartitionOp(const PendingPartitionOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PendingPartitionOp::~PendingPartitionOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PendingPartitionOp& PendingPartitionOp::operator=(
                                                  const PendingPartitionOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_equal_partition(SingleTask *ctx,
                                                        IndexPartition pid, 
                                                        size_t granularity)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new EqualPartitionThunk(pid, granularity);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_weighted_partition(SingleTask *ctx, 
                                                           IndexPartition pid, 
                                                           size_t granularity,
                                       const std::map<DomainPoint,int> &weights)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new WeightedPartitionThunk(pid, granularity, weights);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_union_partition(SingleTask *ctx,
                                                        IndexPartition pid,
                                                        IndexPartition h1,
                                                        IndexPartition h2)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new UnionPartitionThunk(pid, h1, h2);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_intersection_partition(SingleTask *ctx,
                                                            IndexPartition pid,
                                                            IndexPartition h1,
                                                            IndexPartition h2)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new IntersectionPartitionThunk(pid, h1, h2);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_difference_partition(SingleTask *ctx,
                                                             IndexPartition pid,
                                                             IndexPartition h1,
                                                             IndexPartition h2)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new DifferencePartitionThunk(pid, h1, h2);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_cross_product(SingleTask *ctx,
                                                      IndexPartition base,
                                                      IndexPartition source,
                                  std::map<DomainPoint,IndexPartition> &handles)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new CrossProductThunk(base, source, handles);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_union(SingleTask *ctx,
                                                          IndexSpace target,
                                         const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingSpace(target, true/*union*/, handles);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_union(SingleTask *ctx,
                                                          IndexSpace target,
                                                          IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingSpace(target, true/*union*/, handle);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_intersection(
     SingleTask *ctx, IndexSpace target, const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingSpace(target, false/*union*/, handles);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_intersection(
                      SingleTask *ctx, IndexSpace target, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingSpace(target, false/*union*/, handle);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_difference(SingleTask *ctx,
                                         IndexSpace target, IndexSpace initial, 
                                         const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingDifference(target, initial, handles);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::perform_logging()
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_pending_partition_operation(
          parent_ctx->get_unique_id(),
          unique_op_id);
      thunk->perform_logging(this);
    }

    //--------------------------------------------------------------------------
    bool PendingPartitionOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      // Perform the partitioning operation
      ApEvent ready_event = thunk->perform(runtime->forest);
      // We can trigger the handle ready event now
      Runtime::trigger_event(handle_ready);
      complete_mapping();
      Runtime::trigger_event(completion_event, ready_event);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(ready_event));
      // Return true since we succeeded
      return true;
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      handle_ready = Runtime::create_ap_user_event();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      if (thunk != NULL)
        delete thunk;
      thunk = NULL;
      runtime->free_pending_partition_op(this);
    }

    //--------------------------------------------------------------------------
    const char* PendingPartitionOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[PENDING_PARTITION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind PendingPartitionOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return PENDING_PARTITION_OP_KIND;
    }

    /////////////////////////////////////////////////////////////
    // Dependent Partition Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DependentPartitionOp::DependentPartitionOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DependentPartitionOp::DependentPartitionOp(const DependentPartitionOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DependentPartitionOp::~DependentPartitionOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DependentPartitionOp& DependentPartitionOp::operator=(
                                                const DependentPartitionOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_field(SingleTask *ctx, 
                                                   IndexPartition pid,
                                    LogicalRegion handle, LogicalRegion parent,
                                    const Domain &space, FieldID fid)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/); 
      partition_kind = BY_FIELD;
      // Projection region requirement since we need the whole sub-tree
      requirement = RegionRequirement(handle, 0/*idx*/, READ_ONLY, 
                                      EXCLUSIVE, parent);
      requirement.add_field(fid);
      partition_handle = pid;
      color_space = space;
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_image(SingleTask *ctx, 
                                                   IndexPartition pid,
                                          LogicalPartition projection,
                                          LogicalRegion parent, FieldID fid, 
                                          const Domain &space)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      partition_kind = BY_IMAGE;
      // Projection region requirement since we need the whole sub-tree
      requirement = RegionRequirement(projection, 0/*id*/, READ_ONLY,
                                      EXCLUSIVE, parent);
      requirement.add_field(fid);
      partition_handle = pid;
      color_space = space;
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_preimage(SingleTask *ctx,
                                    IndexPartition pid, IndexPartition proj,
                                    LogicalRegion handle, LogicalRegion parent,
                                    FieldID fid, const Domain &space)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      partition_kind = BY_PREIMAGE;
      // Projection region requirement since we need the whole sub-tree
      requirement = RegionRequirement(handle, 0/*idx*/, READ_ONLY, 
                                      EXCLUSIVE, parent);
      requirement.add_field(fid);
      partition_handle = pid;
      color_space = space;
      projection = proj;
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::perform_logging()
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_dependent_partition_operation(
          parent_ctx->get_unique_id(),
          unique_op_id,
          partition_handle.id,
          partition_kind);
      if (requirement.handle_type == PART_PROJECTION)
        LegionSpy::log_logical_requirement(unique_op_id, 0/*idx*/,
                                  false/*region*/,
                                  requirement.partition.index_partition.id,
                                  requirement.partition.field_space.id,
                                  requirement.partition.tree_id,
                                  requirement.privilege,
                                  requirement.prop,
                                  requirement.redop,
                                  requirement.parent.index_space.id);
      else
        LegionSpy::log_logical_requirement(unique_op_id, 0/*idx*/,
                                  true/*region*/,
                                  requirement.region.index_space.id,
                                  requirement.region.field_space.id,
                                  requirement.region.tree_id,
                                  requirement.privilege,
                                  requirement.prop,
                                  requirement.redop,
                                  requirement.parent.index_space.id);
      LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                        requirement.privilege_fields);
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& DependentPartitionOp::get_requirement(void) const
    //--------------------------------------------------------------------------
    {
      return requirement;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      compute_parent_index();
      initialize_privilege_path(privilege_path, requirement);
      begin_dependence_analysis();
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   version_info,
                                                   restrict_info,
                                                   privilege_path);
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_remote_state_analysis(
                                                        RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;  
      version_info.make_local(preconditions, this, runtime->forest);
      if (preconditions.empty())
        Runtime::trigger_event(ready_event);
      else
        Runtime::trigger_event(ready_event,
                               Runtime::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    bool DependentPartitionOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      // We still have to do the traversal to flush any reductions
      InstanceSet empty_instances;
      runtime->forest->physical_traverse_path(physical_ctx, privilege_path,
                                              requirement, version_info, 
                                              this, 0/*idx*/, 
                                              false/*find valid*/,
                                              map_applied_conditions,
                                              empty_instances
#ifdef DEBUG_LEGION
                                              , get_logging_name()
                                              , unique_op_id
#endif
                                              );
      ApEvent ready_event = ApEvent::NO_AP_EVENT;
      switch (partition_kind)
      {
        case BY_FIELD:
          {
            ready_event = 
              runtime->forest->create_partition_by_field(physical_ctx, 
                this, 0/*idx*/, requirement, partition_handle, color_space,
                completion_event, version_info, map_applied_conditions);
            break;
          }
        case BY_IMAGE:
          {
            ready_event = 
              runtime->forest->create_partition_by_image(physical_ctx, 
                this, 0/*idx*/, requirement, partition_handle, color_space,
                completion_event, version_info, map_applied_conditions);
            break;
          }
        case BY_PREIMAGE:
          {
            ready_event = 
              runtime->forest->create_partition_by_preimage(physical_ctx, 
                this, 0/*idx*/, requirement, projection, partition_handle,
                color_space, completion_event, version_info, 
                map_applied_conditions);
            break;
          }
        default:
          assert(false); // should never get here
      }
      // Once we are done running these routines, we can mark
      // that the handles have all been completed
#ifdef DEBUG_LEGION
      assert(handle_ready.exists() && !handle_ready.has_triggered());
#endif
      Runtime::trigger_event(handle_ready);
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      Runtime::trigger_event(completion_event, ready_event);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(ready_event));
      // return true since we succeeded
      return true;
    }

    //--------------------------------------------------------------------------
    unsigned DependentPartitionOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    FatTreePath* DependentPartitionOp::compute_fat_path(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      if (requirement.handle_type == PART_PROJECTION)
      {
        IndexPartition handle = requirement.partition.get_index_partition();
        return runtime->forest->compute_full_fat_path(handle);
      }
      else
      {
        IndexSpace handle = requirement.region.get_index_space();
        return runtime->forest->compute_full_fat_path(handle);
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      handle_ready = Runtime::create_ap_user_event();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      privilege_path = RegionTreePath();
      if (!handle_ready.has_triggered())
        Runtime::trigger_event(handle_ready);
      version_info.clear();
      restrict_info.clear();
      map_applied_conditions.clear();
      runtime->free_dependent_partition_op(this);
    }

    //--------------------------------------------------------------------------
    const char* DependentPartitionOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[DEPENDENT_PARTITION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind DependentPartitionOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return DEPENDENT_PARTITION_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t DependentPartitionOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::compute_parent_index(void)
    //--------------------------------------------------------------------------
    {
      int parent_index = parent_ctx->find_parent_region_req(requirement);
      if (parent_index < 0)
      {
        log_region.error("Parent task %s (ID %lld) of partition "
                                   "operation (ID %lld) does not have a region "
                                   "requirement for region (%x,%x,%x) "
                                   "as a parent of region requirement.",
                                   parent_ctx->get_task_name(), 
                                   parent_ctx->get_unique_id(),
                                   unique_op_id, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id);
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_BAD_PARENT_REGION);
      }
      else
        parent_req_index = unsigned(parent_index);
    }

    enum PendingPartitionKind
    {
      EQUAL_PARTITION = 0,
      WEIGHTED_PARTITION,
      UNION_PARTITION,
      INTERSECTION_PARTITION,
      DIFFERENCE_PARTITION,
    };
    //--------------------------------------------------------------------------
    void PendingPartitionOp::EqualPartitionThunk::perform_logging(
                                                         PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(op->unique_op_id, pid.id,
          EQUAL_PARTITION);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::WeightedPartitionThunk::perform_logging(
                                                         PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(op->unique_op_id, pid.id,
          WEIGHTED_PARTITION);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::UnionPartitionThunk::perform_logging(
                                                         PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(op->unique_op_id, pid.id,
          UNION_PARTITION);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::IntersectionPartitionThunk::perform_logging(
                                                         PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(op->unique_op_id, pid.id,
          INTERSECTION_PARTITION);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::DifferencePartitionThunk::perform_logging(
                                                         PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(op->unique_op_id, pid.id,
          DIFFERENCE_PARTITION);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::CrossProductThunk::perform_logging(
                                                         PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::ComputePendingSpace::perform_logging(
                                                         PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::ComputePendingDifference::perform_logging(
                                                         PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
    }

    ///////////////////////////////////////////////////////////// 
    // Fill Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FillOp::FillOp(Runtime *rt)
      : SpeculativeOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FillOp::FillOp(const FillOp &rhs)
      : SpeculativeOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FillOp::~FillOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FillOp& FillOp::operator=(const FillOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FillOp::initialize(SingleTask *ctx, LogicalRegion handle,
                            LogicalRegion parent, FieldID fid,
                            const void *ptr, size_t size,
                            const Predicate &pred,bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      initialize_speculation(ctx, true/*track*/, 1, pred);
      requirement = RegionRequirement(handle, WRITE_DISCARD, EXCLUSIVE, parent);
      requirement.privilege_fields.insert(fid);
      value_size = size;
      value = malloc(value_size);
      memcpy(value, ptr, value_size);
      if (check_privileges)
        check_fill_privilege();
      initialize_privilege_path(privilege_path, requirement);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void FillOp::initialize(SingleTask *ctx, LogicalRegion handle,
                            LogicalRegion parent, FieldID fid, 
                            const Future &f,
                            const Predicate &pred,bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      initialize_speculation(ctx, true/*track*/, 1, pred);
      requirement = RegionRequirement(handle, WRITE_DISCARD, EXCLUSIVE, parent);
      requirement.privilege_fields.insert(fid);
      future = f;
      if (check_privileges)
        check_fill_privilege();
      initialize_privilege_path(privilege_path, requirement);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void FillOp::initialize(SingleTask *ctx, LogicalRegion handle,
                            LogicalRegion parent,
                            const std::set<FieldID> &fields,
                            const void *ptr, size_t size,
                            const Predicate &pred,bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      initialize_speculation(ctx, true/*track*/, 1, pred);
      requirement = RegionRequirement(handle, WRITE_DISCARD, EXCLUSIVE, parent);
      requirement.privilege_fields = fields;
      value_size = size;
      value = malloc(value_size);
      memcpy(value, ptr, size);
      if (check_privileges)
        check_fill_privilege();
      initialize_privilege_path(privilege_path, requirement);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void FillOp::initialize(SingleTask *ctx, LogicalRegion handle,
                            LogicalRegion parent,
                            const std::set<FieldID> &fields, 
                            const Future &f,
                            const Predicate &pred,bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      initialize_speculation(ctx, true/*track*/, 1, pred);
      requirement = RegionRequirement(handle, WRITE_DISCARD, EXCLUSIVE, parent);
      requirement.privilege_fields = fields;
      future = f;
      if (check_privileges)
        check_fill_privilege();
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void FillOp::initialize(SingleTask *ctx, const FillLauncher &launcher,
                            bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      initialize_speculation(ctx, true/*track*/, 1, launcher.predicate);
      requirement = RegionRequirement(launcher.handle, WRITE_DISCARD,
                                      EXCLUSIVE, launcher.parent);
      requirement.privilege_fields = launcher.fields;
      value_size = launcher.argument.get_size();
      if (value_size > 0)
      {
        value = malloc(value_size);
        memcpy(value, launcher.argument.get_ptr(), value_size);
      }
      else
        future = launcher.future;
      grants = launcher.grants;
      wait_barriers = launcher.wait_barriers;
      arrive_barriers = launcher.arrive_barriers;
      if (check_privileges)
        check_fill_privilege();
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void FillOp::perform_logging(void)
    //--------------------------------------------------------------------------
    {
      if (!Runtime::legion_spy_enabled)
	return;
      LegionSpy::log_fill_operation(parent_ctx->get_unique_id(), 
                                    unique_op_id);
      LegionSpy::log_logical_requirement(unique_op_id, 0/*index*/,
                                         true/*region*/,
                                         requirement.region.index_space.id,
                                         requirement.region.field_space.id,
                                         requirement.region.tree_id,
                                         requirement.privilege,
                                         requirement.prop,
                                         requirement.redop,
                                         requirement.parent.index_space.id);
      LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                        requirement.privilege_fields);
    }

    //--------------------------------------------------------------------------
    void FillOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative();
      value = NULL;
      value_size = 0;
    }

    //--------------------------------------------------------------------------
    void FillOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_speculative();
      privilege_path.clear();
      if (value != NULL) 
      {
        free(value);
        value = NULL;
      }
      version_info.clear();
      future = Future();
      restrict_info.clear();
      map_applied_conditions.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      runtime->free_fill_op(this);
    }

    //--------------------------------------------------------------------------
    const char* FillOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[FILL_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind FillOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return FILL_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t FillOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_dependence_analysis(void) 
    //--------------------------------------------------------------------------
    {
      // First compute the parent index
      compute_parent_index();
      begin_dependence_analysis();
      // Register a dependence on our predicate
      register_predicate_dependence();
      // If we are waiting on a future register a dependence
      if (future.impl != NULL)
        future.impl->register_dependence(this);
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   version_info,
                                                   restrict_info,
                                                   privilege_path);
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_remote_state_analysis(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;  
      version_info.make_local(preconditions, this, runtime->forest);
      if (preconditions.empty())
        Runtime::trigger_event(ready_event);
      else
        Runtime::trigger_event(ready_event,
                               Runtime::merge_events(preconditions));
    }
    
    //--------------------------------------------------------------------------
    void FillOp::resolve_true(void)
    //--------------------------------------------------------------------------
    {
      // Put this on the queue of stuff to do
      runtime->add_to_local_queue(parent_ctx->get_executing_processor(),
                                  this, false/*prev fail*/);
    }

    //--------------------------------------------------------------------------
    void FillOp::resolve_false(void)
    //--------------------------------------------------------------------------
    {
      // Mark that this operation has completed both
      // execution and mapping indicating that we are done
      // Do it in this order to avoid calling 'execute_trigger'
      complete_execution();
      assert(0 && "TODO: advance mapping states if you care");
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    bool FillOp::speculate(bool &value)
    //--------------------------------------------------------------------------
    {
      // We never speculate on fill ops since they are lazy anyway
      return false;
    }

    //--------------------------------------------------------------------------
    bool FillOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      // Still have to walk the path
      InstanceSet empty_instances;
      runtime->forest->physical_traverse_path(physical_ctx, privilege_path,
                                              requirement, version_info, 
                                              this, 0/*idx*/, 
                                              false/*find valid*/,
                                              map_applied_conditions,
                                              empty_instances
#ifdef DEBUG_LEGION
                                              , get_logging_name()
                                              , unique_op_id
#endif
                                              );
      // Tell the region tree forest to fill in this field
      // Note that the forest takes ownership of the value buffer
      if (future.impl == NULL)
      {
#ifdef DEBUG_LEGION
        assert(value != NULL);
#endif
        InstanceSet mapped_instances;
        if (restrict_info.has_restrictions())
        {
          parent_ctx->get_physical_references(parent_req_index, 
                                              mapped_instances);
          runtime->forest->traverse_and_register(physical_ctx, privilege_path,
                                                 requirement, version_info, 
                                                 this, 0/*idx*/, 
                                                 ApEvent::NO_AP_EVENT,
                                                 false/*defer add users*/,
                                                 map_applied_conditions,
                                                 mapped_instances
#ifdef DEBUG_LEGION
                                                 , get_logging_name()
                                                 , unique_op_id
#endif
                                                 );
        }
        ApEvent sync_precondition = compute_sync_precondition();
        ApEvent done_event = 
          runtime->forest->fill_fields(physical_ctx, this, requirement, 
                                       0/*idx*/, value, value_size, 
                                       version_info, restrict_info, 
                                       mapped_instances, sync_precondition,
                                       map_applied_conditions);
        if (!mapped_instances.empty() && Runtime::legion_spy_enabled)
        {
          runtime->forest->log_mapping_decision(unique_op_id, 0/*idx*/,
                                                requirement,
                                                mapped_instances);
#ifdef LEGION_SPY
          LegionSpy::log_operation_events(unique_op_id, done_event,
                                          completion_event);
#endif
        }
        version_info.apply_mapping(physical_ctx.get_id(),
                               runtime->address_space, map_applied_conditions);
        // Clear value and value size since the forest ended up 
        // taking ownership of them
        value = NULL;
        value_size = 0;
        if (!map_applied_conditions.empty())
          complete_mapping(Runtime::merge_events(map_applied_conditions));
        else
          complete_mapping();
        // See if we have any arrivals to trigger
        if (!arrive_barriers.empty())
        {
          for (std::vector<PhaseBarrier>::const_iterator it = 
                arrive_barriers.begin(); it != arrive_barriers.end(); it++)
          {
            Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                          completion_event);
          }
        }
        complete_execution(Runtime::protect_event(done_event));
      }
      else
      {
        // If we have a future value see if its event has triggered
        ApEvent future_ready_event = future.impl->get_ready_event();
        if (!future_ready_event.has_triggered())
        {
          // Launch a task to handle the deferred complete
          DeferredExecuteArgs deferred_execute_args;
          deferred_execute_args.hlr_id = HLR_DEFERRED_EXECUTION_TRIGGER_ID;
          deferred_execute_args.proxy_this = this;
          runtime->issue_runtime_meta_task(&deferred_execute_args,
                                           sizeof(deferred_execute_args),
                                           HLR_DEFERRED_EXECUTION_TRIGGER_ID,
                                           HLR_LATENCY_PRIORITY, this, 
                                 Runtime::protect_event(future_ready_event));
        }
        else
          deferred_execute(); // can do the completion now
      }
      // This should never fail
      return true;
    }

    //--------------------------------------------------------------------------
    void FillOp::deferred_execute(void)
    //--------------------------------------------------------------------------
    {
      // Make a copy of the future value since the region tree
      // will want to take ownership of the buffer
      size_t result_size = future.impl->get_untyped_size();
      void *result = malloc(result_size);
      memcpy(result, future.impl->get_untyped_result(), result_size);
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      InstanceSet mapped_instances;
      if (restrict_info.has_restrictions())
      {
        parent_ctx->get_physical_references(parent_req_index, mapped_instances);
        runtime->forest->traverse_and_register(physical_ctx, privilege_path,
                                               requirement, version_info, 
                                               this, 0/*idx*/, 
                                               ApEvent::NO_AP_EVENT, 
                                               false/*defer add users*/,
                                               map_applied_conditions,
                                               mapped_instances
#ifdef DEBUG_LEGION
                                               , get_logging_name()
                                               , unique_op_id
#endif
                                               );
      }
      ApEvent sync_precondition = compute_sync_precondition();
      ApEvent done_event = 
          runtime->forest->fill_fields(physical_ctx, this, requirement, 
                                       0/*idx*/, value, value_size, 
                                       version_info, restrict_info, 
                                       mapped_instances, sync_precondition,
                                       map_applied_conditions);
      if (!mapped_instances.empty() && Runtime::legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, 0/*idx*/,
                                              requirement,
                                              mapped_instances);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, done_event,
                                        completion_event);
#endif
      }
      version_info.apply_mapping(physical_ctx.get_id(),
                              runtime->address_space, map_applied_conditions);
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      // See if we have any arrivals to trigger
      if (!arrive_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::const_iterator it = 
              arrive_barriers.begin(); it != arrive_barriers.end(); it++)
        {
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        completion_event);
        }
      }
      complete_execution(Runtime::protect_event(done_event));
    }
    
    //--------------------------------------------------------------------------
    unsigned FillOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void FillOp::check_fill_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field);
      switch (et)
      {
        case NO_ERROR:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            log_region.error("Requirest for invalid region handle "
                                   "(%x,%d,%d) for fill operation"
                                   "(ID %lld)",
                                   requirement.region.index_space.id, 
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id, 
                                   unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_REGION_HANDLE);
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) || 
                            (requirement.handle_type == REG_PROJECTION)
                             ? requirement.region.field_space : 
                               requirement.partition.field_space;
            log_region.error("Field %d is not a valid field of field "
                                   "space %d for fill operation (ID %lld)",
                                   bad_field, sp.id, unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
          }
        case ERROR_INVALID_INSTANCE_FIELD:
          {
            log_region.error("Instance field %d is not one of the "
                                   "privilege fields for fill operation"
                                   "(ID %lld)",
                                    bad_field, unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_INSTANCE_FIELD);
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            log_region.error("Instance field %d is a duplicate for "
                                    "fill operation (ID %lld)",
                                    bad_field, unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_DUPLICATE_INSTANCE_FIELD);
          }
        case ERROR_BAD_PARENT_REGION:
          {
            log_region.error("Parent task %s (ID %lld) of fill operation "
                                   "(ID %lld) does not have a region "
                                   "requirement for region (%x,%x,%x) "
                                   "as a parent of region requirement",
                                   parent_ctx->get_task_name(), 
                                   parent_ctx->get_unique_id(),
                                   unique_op_id, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_PARENT_REGION);
          }
        case ERROR_BAD_REGION_PATH:
          {
            log_region.error("Region (%x,%x,%x) is not a "
                                   "sub-region of parent region "
                                   "(%x,%x,%x) for region requirement of fill "
                                   "operation (ID %lld)",
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id,
                                   requirement.parent.index_space.id,
                                   requirement.parent.field_space.id,
                                   requirement.parent.tree_id,
                                   unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_PATH);
          }
        case ERROR_BAD_REGION_TYPE:
          {
            log_region.error("Region requirement of fill operation "
                                   "(ID %lld) cannot find privileges for field "
                                   "%d in parent task",
                                   unique_op_id, bad_field);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_TYPE);
          }
        case ERROR_BAD_REGION_PRIVILEGES:
          {
            log_region.error("Privileges %x for region "
                                   "(%x,%x,%x) are not a subset of privileges "
                                   "of parent task's privileges for region "
                                   "requirement of fill operation (ID %lld)",
                                   requirement.privilege, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id, 
                                   unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_PRIVILEGES);
          }
        // this should never happen with a fill operation 
        case ERROR_NON_DISJOINT_PARTITION: 
        default:
          assert(false); // Should never happen
      }
    }

    //--------------------------------------------------------------------------
    void FillOp::compute_parent_index(void)
    //--------------------------------------------------------------------------
    {
      int parent_index = parent_ctx->find_parent_region_req(requirement);
      if (parent_index < 0)
      {
        log_region.error("Parent task %s (ID %lld) of fill "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id);
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_BAD_PARENT_REGION);
      }
      else
        parent_req_index = unsigned(parent_index);
    }

    //--------------------------------------------------------------------------
    ApEvent FillOp::compute_sync_precondition(void) const
    //--------------------------------------------------------------------------
    {
      if (wait_barriers.empty() && grants.empty())
        return ApEvent::NO_AP_EVENT;
      std::set<ApEvent> sync_preconditions;
      if (!wait_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::const_iterator it = 
              wait_barriers.begin(); it != wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
          sync_preconditions.insert(e);
        }
      }
      if (!grants.empty())
      {
        for (std::vector<Grant>::const_iterator it = grants.begin();
              it != grants.end(); it++)
        {
          ApEvent e = it->impl->acquire_grant();
          sync_preconditions.insert(e);
        }
      }
      return Runtime::merge_events(sync_preconditions);
    }

    ///////////////////////////////////////////////////////////// 
    // Attach Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AttachOp::AttachOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AttachOp::AttachOp(const AttachOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    AttachOp::~AttachOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AttachOp& AttachOp::operator=(const AttachOp &rhs)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    PhysicalRegion AttachOp::initialize_hdf5(SingleTask *ctx,
                                             const char *name,
                                             LogicalRegion handle, 
                                             LogicalRegion parent,
                                      const std::map<FieldID,const char*> &fmap,
                                             LegionFileMode mode,
                                             bool check_privileges)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      if (fmap.empty())
      {
        log_run.warning("WARNING: HDF5 ATTACH OPERATION ISSUED WITH NO "
                        "FIELD MAPPINGS IN TASK %s (ID %lld)! DID YOU "
                        "FORGET THEM?!?", parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id());

      }
      file_type = HDF5_FILE;
      file_name = strdup(name);
      // Construct the region requirement for this task
      requirement = RegionRequirement(handle, WRITE_DISCARD, EXCLUSIVE, parent);
      for (std::map<FieldID,const char*>::const_iterator it = fmap.begin();
            it != fmap.end(); it++)
      {
        requirement.add_field(it->first);
        field_map[it->first] = strdup(it->second);
      }
      file_mode = mode;
      region = PhysicalRegion(legion_new<PhysicalRegionImpl>(requirement,
                              completion_event, true/*mapped*/, ctx,
                              0/*map id*/, 0/*tag*/, false/*leaf*/, runtime));
      if (check_privileges)
        check_privilege();
      initialize_privilege_path(privilege_path, requirement);
      return region;
    }

    //--------------------------------------------------------------------------
    PhysicalRegion AttachOp::initialize_file(SingleTask *ctx,
                                             const char *name,
                                             LogicalRegion handle,
                                             LogicalRegion parent,
                                      const std::vector<FieldID> &fvec,
                                             LegionFileMode mode,
                                             bool check_privileges)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      if (fvec.empty())
      {
        log_run.warning("WARNING: FILE ATTACH OPERATION ISSUED WITH NO "
                        "FIELD MAPPINGS IN TASK %s (ID %lld)! DID YOU "
                        "FORGET THEM?!?", parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id());

      }
      file_type = NORMAL_FILE;
      file_name = strdup(name);
      // Construct the region requirement for this task
      requirement = RegionRequirement(handle, WRITE_DISCARD, EXCLUSIVE, parent);
      for (std::vector<FieldID>::const_iterator it = fvec.begin();
            it != fvec.end(); it++)
      {
        requirement.add_field(*it);
      }
      file_mode = mode;
      region = PhysicalRegion(legion_new<PhysicalRegionImpl>(requirement,
                              completion_event, true/*mapped*/, ctx,
                              0/*map id*/, 0/*tag*/, false/*leaf*/, runtime));
      if (check_privileges)
        check_privilege();
      initialize_privilege_path(privilege_path, requirement);
      return region;
    }

    //--------------------------------------------------------------------------
    void AttachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      file_name = NULL;
    }

    //--------------------------------------------------------------------------
    void AttachOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      if (file_name != NULL)
      {
        free(const_cast<char*>(file_name));
        file_name = NULL;
      }
      for (std::map<FieldID,const char*>::const_iterator it = field_map.begin();
            it != field_map.end(); it++)
      {
        free(const_cast<char*>(it->second));
      }
      field_map.clear();
      region = PhysicalRegion();
      privilege_path.clear();
      version_info.clear();
      restrict_info.clear();
      map_applied_conditions.clear();
      runtime->free_attach_op(this);
    }

    //--------------------------------------------------------------------------
    const char* AttachOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[ATTACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind AttachOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return ATTACH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t AttachOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }
    
    //--------------------------------------------------------------------------
    void AttachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent index
      compute_parent_index();
      begin_dependence_analysis();
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   version_info,
                                                   restrict_info, 
                                                   privilege_path);
      // If we have any restriction on ourselves, that is very bad
      if (restrict_info.has_restrictions())
      {
        log_run.error("Illegal file attachment for file %s performed on "
                      "logical region (%x,%x,%x) which is under "
                      "restricted coherence! User coherence must first "
                      "be acquired with an acquire operation before "
                      "attachment can be performed.", file_name,
                      requirement.region.index_space.id,
                      requirement.region.field_space.id,
                      requirement.region.tree_id);
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_ILLEGAL_FILE_ATTACH);
      }
      // After we are done with our dependence analysis, then we 
      // need to add restricted coherence on the logical region 
      RegionTreeContext ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      runtime->forest->restrict_user_coherence(ctx, parent_ctx, 
                                               requirement.region,
                                               requirement.privilege_fields);
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void AttachOp::trigger_remote_state_analysis(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;  
      version_info.make_local(preconditions, this, runtime->forest);
      if (preconditions.empty())
        Runtime::trigger_event(ready_event);
      else
        Runtime::trigger_event(ready_event,
                               Runtime::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    bool AttachOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      // We still have to do a traversal to make sure everything is open
      InstanceSet empty_instances;
      runtime->forest->physical_traverse_path(physical_ctx, privilege_path,
                                              requirement, version_info, 
                                              this, 0/*idx*/,
                                              false/*find valid*/,
                                              map_applied_conditions,
                                              empty_instances
#ifdef DEBUG_LEGION
                                              , get_logging_name()
                                              , unique_op_id
#endif
                                              );
      InstanceRef result = runtime->forest->attach_file(physical_ctx,
                                                        requirement, this,
                                                        version_info);
#ifdef DEBUG_LEGION
      assert(result.has_ref());
#endif
      version_info.apply_mapping(physical_ctx.get_id(),
                                 runtime->address_space,map_applied_conditions);
      // This operation is ready once the file is attached
      region.impl->set_reference(result);
      // Once we have created the instance, then we are done
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      ApEvent attach_event = result.get_ready_event();
      Runtime::trigger_event(completion_event, attach_event);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(attach_event));
      // Should always succeed
      return true;
    }

    //--------------------------------------------------------------------------
    unsigned AttachOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void AttachOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void AttachOp::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance AttachOp::create_instance(const Domain &dom,
             const std::vector<size_t> &sizes, LayoutConstraintSet &constraints)
    //--------------------------------------------------------------------------
    {
      // TODO: Update attach operation to fill in 
      // constraints for different file types
      assert(false);
      if (file_type == HDF5_FILE) {
        // First build the set of field paths
        std::vector<const char*> field_files(field_map.size());
        unsigned idx = 0;
        for (std::map<FieldID,const char*>::const_iterator it = 
              field_map.begin(); it != field_map.end(); it++, idx++)
        {
          field_files[idx] = it->second;
        }
        // Now ask the low-level runtime to create the instance
        PhysicalInstance result = dom.create_hdf5_instance(file_name, sizes,
                             field_files, (file_mode == LEGION_FILE_READ_ONLY));
#ifdef DEBUG_LEGION
      assert(result.exists());
#endif
        return result;
      } else if (file_type == NORMAL_FILE) {
        PhysicalInstance result = 
          dom.create_file_instance(file_name, sizes, file_mode);
        return result;
      } else {
        assert(0);
        return PhysicalInstance::NO_INST;
      }
    }

    //--------------------------------------------------------------------------
    void AttachOp::check_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, then check the privileges with the parent context
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field);
      switch (et)
      {
        // Not there is no such things as bad privileges for 
        // acquires and releases because they are controlled by the runtime
        case NO_ERROR:
        case ERROR_BAD_REGION_PRIVILEGES:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            log_region.error("Requirest for invalid region handle "
                                   "(%x,%d,%d) for attach operation "
                                   "(ID %lld)",
                                   requirement.region.index_space.id, 
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id, 
                                   unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_REGION_HANDLE);
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) || 
                            (requirement.handle_type == REG_PROJECTION)
                             ? requirement.region.field_space : 
                               requirement.partition.field_space;
            log_region.error("Field %d is not a valid field of field "
                                   "space %d for attach operation (ID %lld)",
                                   bad_field, sp.id, unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
          }
        case ERROR_INVALID_INSTANCE_FIELD:
          {
            log_region.error("Instance field %d is not one of the "
                                   "privilege fields for attach operation "
                                   "(ID %lld)",
                                    bad_field, unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_INSTANCE_FIELD);
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            log_region.error("Instance field %d is a duplicate for "
                                    "attach operation (ID %lld)",
                                    bad_field, unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_DUPLICATE_INSTANCE_FIELD);
          }
        case ERROR_BAD_PARENT_REGION:
          {
            log_region.error("Parent task %s (ID %lld) of attach operation "
                                   "(ID %lld) does not have a region "
                                   "requirement for region (%x,%x,%x) "
                                   "as a parent of region requirement",
                                   parent_ctx->get_task_name(), 
                                   parent_ctx->get_unique_id(),
                                   unique_op_id, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_PARENT_REGION);
          }
        case ERROR_BAD_REGION_PATH:
          {
            log_region.error("Region (%x,%x,%x) is not a "
                             "sub-region of parent region "
                             "(%x,%x,%x) for region requirement of attach "
                             "operation (ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id, 
                             requirement.region.tree_id,
                             requirement.parent.index_space.id,
                             requirement.parent.field_space.id,
                             requirement.parent.tree_id,
                             unique_op_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_PATH);
          }
        case ERROR_BAD_REGION_TYPE:
          {
            log_region.error("Region requirement of attach operation "
                                   "(ID %lld) cannot find privileges for field "
                                   "%d in parent task",
                                   unique_op_id, bad_field);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_BAD_REGION_TYPE);
          }
        // this should never happen with an inline mapping
        case ERROR_NON_DISJOINT_PARTITION: 
        default:
          assert(false); // Should never happen
      }
    }

    //--------------------------------------------------------------------------
    void AttachOp::compute_parent_index(void)
    //--------------------------------------------------------------------------
    {
      int parent_index = parent_ctx->find_parent_region_req(requirement);
      if (parent_index < 0)
      {
        log_region.error("Parent task %s (ID %lld) of attach "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id);
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_BAD_PARENT_REGION);
      }
      else
        parent_req_index = unsigned(parent_index);
    }

    ///////////////////////////////////////////////////////////// 
    // Detach Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DetachOp::DetachOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DetachOp::DetachOp(const DetachOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DetachOp::~DetachOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DetachOp& DetachOp::operator=(const DetachOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void DetachOp::initialize_detach(SingleTask *ctx, 
                                     PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      // No need to check privileges because we never would have been
      // able to attach in the first place anyway.
      requirement = region.impl->get_requirement();
      initialize_privilege_path(privilege_path, requirement);
      // Delay getting a reference until trigger_execution().  This means we
      //  have to keep region
      this->region = region;
    }

    //--------------------------------------------------------------------------
    void DetachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void DetachOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      region = PhysicalRegion();
      privilege_path.clear();
      version_info.clear();
      restrict_info.clear();
      runtime->free_detach_op(this);
    }

    //--------------------------------------------------------------------------
    const char* DetachOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[DETACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind DetachOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return DETACH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t DetachOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent index
      compute_parent_index();
      begin_dependence_analysis();
      // Before we do our dependence analysis, we can remove the 
      // restricted coherence on the logical region
      RegionTreeContext ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      runtime->forest->acquire_user_coherence(ctx, requirement.region,
                                              requirement.privilege_fields);
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement, 
                                                   version_info,
                                                   restrict_info,
                                                   privilege_path);
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_remote_state_analysis(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;  
      version_info.make_local(preconditions, this, runtime->forest);
      if (preconditions.empty())
        Runtime::trigger_event(ready_event);
      else
        Runtime::trigger_event(ready_event,
                               Runtime::merge_events(preconditions));
    }
    
    //--------------------------------------------------------------------------
    bool DetachOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      // Actual unmap of an inline mapped region was deferred to here
      if (region.impl->is_mapped())
        region.impl->unmap_region();
      // Now we can get the reference we need for the detach operation
      InstanceSet references;
      region.impl->get_references(references);
#ifdef DEBUG_LEGION
      assert(references.size() == 1);
#endif
      InstanceRef reference = references[0];
      // Check that this is actually a file
      PhysicalManager *manager = reference.get_manager();
#ifdef DEBUG_LEGION
      assert(!manager->is_reduction_manager()); 
#endif
      InstanceManager *inst_manager = manager->as_instance_manager(); 
      if (!inst_manager->is_attached_file())
      {
        log_run.error("Illegal detach operation on a physical region which "
                      "was not attached!");
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_ILLEGAL_DETACH_OPERATION);
      }

      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      ApEvent detach_event = 
        runtime->forest->detach_file(physical_ctx, requirement, 
                                     this, version_info, reference);
      std::set<RtEvent> applied_conditions;
      version_info.apply_mapping(physical_ctx.get_id(),
                                 runtime->address_space, applied_conditions);
      if (!applied_conditions.empty())
        complete_mapping(Runtime::merge_events(applied_conditions));
      else
        complete_mapping();
      Runtime::trigger_event(completion_event, detach_event);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(detach_event));
      // This should always succeed
      return true;
    }

    //--------------------------------------------------------------------------
    unsigned DetachOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void DetachOp::compute_parent_index(void)
    //--------------------------------------------------------------------------
    {
      int parent_index = parent_ctx->find_parent_region_req(requirement);
      if (parent_index < 0)
      {
        log_region.error("Parent task %s (ID %lld) of detach "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id);
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_BAD_PARENT_REGION);
      }
      else
        parent_req_index = unsigned(parent_index);
    }

    ///////////////////////////////////////////////////////////// 
    // Timing Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TimingOp::TimingOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TimingOp::TimingOp(const TimingOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TimingOp::~TimingOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TimingOp& TimingOp::operator=(const TimingOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    Future TimingOp::initialize(SingleTask *ctx, const Future &pre)
    //--------------------------------------------------------------------------
    {
      kind = ABSOLUTE_MEASUREMENT;
      precondition = pre; 
      result = Future(legion_new<FutureImpl>(runtime, true/*register*/,
                  runtime->get_available_distributed_id(true),
                  runtime->address_space, runtime->address_space, 
                  RtUserEvent::NO_RT_USER_EVENT, this));
      return result;
    }

    //--------------------------------------------------------------------------
    Future TimingOp::initialize_microseconds(SingleTask *ctx, const Future &pre)
    //--------------------------------------------------------------------------
    {
      kind = MICROSECOND_MEASUREMENT;
      precondition = pre;
      result = Future(legion_new<FutureImpl>(runtime, true/*register*/,
                  runtime->get_available_distributed_id(true),
                  runtime->address_space, runtime->address_space, 
                  RtUserEvent::NO_RT_USER_EVENT, this));
      return result;
    }

    //--------------------------------------------------------------------------
    Future TimingOp::initialize_nanoseconds(SingleTask *ctx, const Future &pre)
    //--------------------------------------------------------------------------
    {
      kind = NANOSECOND_MEASUREMENT;
      precondition = pre;
      result = Future(legion_new<FutureImpl>(runtime, true/*register*/,
                  runtime->get_available_distributed_id(true),
                  runtime->address_space, runtime->address_space, 
                  RtUserEvent::NO_RT_USER_EVENT, this));
      return result;
    }

    //--------------------------------------------------------------------------
    void TimingOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation(); 
    }

    //--------------------------------------------------------------------------
    void TimingOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      // Remove our references
      precondition = Future();
      result = Future();
      runtime->free_timing_op(this);
    }

    //--------------------------------------------------------------------------
    const char* TimingOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[TIMING_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TimingOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return TIMING_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TimingOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      begin_dependence_analysis();
      if (precondition.impl != NULL)
        precondition.impl->register_dependence(this);
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    bool TimingOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      complete_mapping();
      if ((precondition.impl != NULL) && 
          !(precondition.impl->ready_event.has_triggered()))
      {
        ApEvent wait_on = precondition.impl->get_ready_event();
        DeferredExecuteArgs args;
        args.hlr_id = HLR_DEFERRED_EXECUTE_ID;
        args.proxy_this = this;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_DEFERRED_EXECUTE_ID,
                                         HLR_LATENCY_PRIORITY, this, 
                                         Runtime::protect_event(wait_on));
      }
      else
        deferred_execute();
      return true;
    }

    //--------------------------------------------------------------------------
    void TimingOp::deferred_execute(void)
    //--------------------------------------------------------------------------
    {
      switch (kind)
      {
        case ABSOLUTE_MEASUREMENT:
          {
            double value = Realm::Clock::current_time();
            result.impl->set_result(&value, sizeof(value), false);
            break;
          }
        case MICROSECOND_MEASUREMENT:
          {
            long long value = Realm::Clock::current_time_in_microseconds();
            result.impl->set_result(&value, sizeof(value), false);
            break;
          }
        case NANOSECOND_MEASUREMENT:
          {
            long long value = Realm::Clock::current_time_in_nanoseconds();
            result.impl->set_result(&value, sizeof(value), false);
            break;
          }
        default:
          assert(false); // should never get here
      }
      // Complete the future
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void TimingOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      result.impl->complete_future(); 
      complete_operation();
    }
 
  }; // namespace Internal 
}; // namespace Legion 

// EOF

