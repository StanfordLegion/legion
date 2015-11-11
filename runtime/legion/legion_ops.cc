/* Copyright 2015 Stanford University, NVIDIA Corporation
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
#include "legion_logging.h"
#include "legion_profiling.h"
#include "legion_instances.h"
#include "legion_views.h"

namespace LegionRuntime {
  namespace HighLevel {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Operation 
    /////////////////////////////////////////////////////////////

    const char *const 
      Operation::op_names[Operation::LAST_OP_KIND] = OPERATION_NAMES;

    //--------------------------------------------------------------------------
    Operation::Operation(Internal *rt)
      : runtime(rt), op_lock(Reservation::create_reservation()), 
        gen(0), unique_op_id(0), 
        outstanding_mapping_references(0),
        hardened(false), parent_ctx(NULL)
    //--------------------------------------------------------------------------
    {
      dependence_tracker.mapping = NULL;
      if (!Internal::resilient_mode)
        commit_event = UserEvent::NO_USER_EVENT;
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
#ifdef DEBUG_HIGH_LEVEL
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
      mapped_event = UserEvent::create_user_event();
      resolved_event = UserEvent::create_user_event();
      completion_event = UserEvent::create_user_event();
      if (Internal::resilient_mode)
        commit_event = UserEvent::create_user_event();
      trace = NULL;
      tracing = false;
      must_epoch = NULL;
      must_epoch_index = 0;
#ifdef DEBUG_HIGH_LEVEL
      assert(mapped_event.exists());
      assert(resolved_event.exists());
      assert(completion_event.exists());
      if (Internal::resilient_mode)
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
        mapped_event.trigger();
      if (!resolved_event.has_triggered())
        resolved_event.trigger();
      if (need_completion_trigger && !completion_event.has_triggered())
        completion_event.trigger();
      if (!commit_event.has_triggered())
        commit_event.trigger();
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
#ifdef DEBUG_HIGH_LEVEL
      assert(parent_ctx != NULL);
#endif
      return (parent_ctx->depth+1);
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
      assert(req.handle_type == SINGULAR);
#endif
      runtime->forest->initialize_path(req.region.get_index_space(),
                                       start_node.get_index_partition(), path);
    }

    //--------------------------------------------------------------------------
    void Operation::set_trace(LegionTrace *t)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(trace == NULL);
      assert(t != NULL);
#endif
      trace = t; 
      tracing = !trace->is_fixed();
    }

    //--------------------------------------------------------------------------
    void Operation::set_must_epoch(MustEpochOp *epoch, unsigned index)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(must_epoch == NULL);
      assert(epoch != NULL);
#endif
      must_epoch = epoch;
      must_epoch_index = index;
      must_epoch->register_subop(this);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::localize_region_requirement(RegionRequirement &r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(r.handle_type == SINGULAR);
#endif
      r.parent = r.region;
      r.prop = EXCLUSIVE;
    }

    //--------------------------------------------------------------------------
    void Operation::initialize_operation(SingleTask *ctx, bool track, 
                                         unsigned regs/*= 0*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
    void Operation::trigger_remote_state_analysis(UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      // We have nothing to check for so just trigger the event
      ready_event.trigger();
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
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, COMPLETE_OPERATION);
#endif
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation();
      // Once we're done with this, we can deactivate the object
      deactivate();
    }

    //--------------------------------------------------------------------------
    void Operation::deferred_execute(void)
    //--------------------------------------------------------------------------
    {
      // should only be called if overridden
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Operation::deferred_commit(GenerationID our_gen)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
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
    void Operation::complete_mapping(Event wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      {
        AutoLock o_lock(op_lock);
        assert(!mapped);
        mapped = true;
      }
#endif
      mapped_event.trigger(wait_on);
    }

    //--------------------------------------------------------------------------
    void Operation::complete_execution(Event wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
      if (!wait_on.has_triggered())
      {
        // We have to defer the execution of this operation
        DeferredExecuteArgs args;
        args.hlr_id = HLR_DEFERRED_EXECUTE_ID;
        args.proxy_this = this;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_DEFERRED_EXECUTE_ID,
                                         this, wait_on);
        return;
      }
      // Tell our parent context that we are done mapping
      // It's important that this is done before we mark that we
      // are executed to avoid race conditions
      if (track_parent)
        parent_ctx->register_child_executed(this);
#ifdef DEBUG_HIGH_LEVEL
      {
        AutoLock o_lock(op_lock);
        assert(!executed);
        executed = true;
      }
#endif
      // Now see if we are ready to complete this operation
      if (!mapped_event.has_triggered() || !resolved_event.has_triggered())
      {
        Event trigger_pre = 
          Event::merge_events(mapped_event, resolved_event);
        DeferredCompleteArgs args;
        args.hlr_id = HLR_DEFERRED_COMPLETE_ID;
        args.proxy_this = this;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_DEFERRED_COMPLETE_ID,
                                         this, trigger_pre);
      }
      else // Do the trigger now
        trigger_complete();
    }

    //--------------------------------------------------------------------------
    void Operation::resolve_speculation(Event wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      {
        AutoLock o_lock(op_lock);
        assert(!resolved);
        resolved = true;
      }
#endif
      resolved_event.trigger(wait_on);
    }

    //--------------------------------------------------------------------------
    void Operation::complete_operation(void)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = false;
      // Tell our parent that we are complete
      // It's important that we do this before we mark ourselves
      // completed in order to avoid race conditions
      if (track_parent)
        parent_ctx->register_child_complete(this);
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
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
        if ((!Internal::resilient_mode) || early_commit_request ||
            ((hardened && unverified_regions.empty())))
        {
          trigger_commit_invoked = true;
          need_trigger = true;
        }
        else if (outstanding_mapping_references == 0)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(dependence_tracker.commit != NULL);
#endif
          CommitDependenceTracker *tracker = dependence_tracker.commit;
          need_trigger = tracker->issue_commit_trigger(this, runtime);
          if (need_trigger)
            trigger_commit_invoked = true;
        }
      }
      if (need_completion_trigger)
        completion_event.trigger(); 
      if (must_epoch != NULL)
        must_epoch->notify_subop_complete(this);
      // finally notify all the operations we dependended on
      // that we validated their regions note we don't need
      // the lock since this was all set when we did our mapping analysis
      for (std::map<Operation*,std::set<unsigned> >::const_iterator it =
            verify_regions.begin(); it != verify_regions.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
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
    void Operation::commit_operation(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, COMMIT_OPERATION);
#endif
      // Tell our parent context that we are committed
      // Do this before actually committing to avoid race conditions
      if (track_parent)
        parent_ctx->register_child_commit(this);
      // Mark that we are committed 
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(mapped);
        assert(executed);
        assert(resolved);
        assert(completed);
        assert(!committed);
#endif
        committed = true;
      } 
      if (must_epoch != NULL)
        must_epoch->notify_subop_commit(this);
      // Trigger the commit event
      if (Internal::resilient_mode)
        commit_event.trigger();
    }

    //--------------------------------------------------------------------------
    void Operation::harden_operation(void)
    //--------------------------------------------------------------------------
    {
      // Mark that this operation is now hardened against failures
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
                                         Event other_commit_event)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(our_gen <= gen); // better not be ahead of where we are now
#endif
      // If the generations match and we haven't committed yet, 
      // register an outgoing dependence
      if ((our_gen == gen) && !committed)
      {
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
        assert(our_gen <= gen); // better not be ahead of where we are now
#endif
        if ((our_gen == gen) && !committed)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(outstanding_mapping_references > 0);
#endif
          outstanding_mapping_references--;
          // If we've completed and we have no mapping references
          // and we have no outstanding commit dependences then 
          // we can commit this operation
          if ((outstanding_mapping_references == 0) && !trigger_commit_invoked)
          {
#ifdef DEBUG_HIGH_LEVEL
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
    Event Operation::invoke_state_analysis(void)
    //--------------------------------------------------------------------------
    {
      // First check to see if the parent context has remote state
      if ((parent_ctx != NULL) && parent_ctx->has_remote_state())
      {
        // This can be an expensive operation so defer it, but give it
        // a slight priority boost because we know that it will likely
        // involve some inter-node communication and we want to get 
        // that in flight quickly to help hide latency.
        UserEvent ready_event = UserEvent::create_user_event();
        StateAnalysisArgs args;
        args.hlr_id = HLR_STATE_ANALYSIS_ID;
        args.proxy_op = this;
        args.ready_event = ready_event;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_STATE_ANALYSIS_ID, this,
                                         Event::NO_EVENT, 1/*priority*/);
        return ready_event;
      }
      else
        return Event::NO_EVENT;
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
#ifdef DEBUG_HIGH_LEVEL
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
    void Operation::MappingDependenceTracker::issue_stage_triggers(
                      Operation *op, Internal *runtime, MustEpochOp *must_epoch)
    //--------------------------------------------------------------------------
    {
      bool map_now = true;
      bool resolve_now = true;
      if (!mapping_dependences.empty())
      {
        Event map_precondition = Event::merge_events(mapping_dependences);
        if (!map_precondition.has_triggered())
        {
          if (must_epoch == NULL)
          {
            DeferredMappingArgs args;
            args.hlr_id = HLR_DEFERRED_MAPPING_TRIGGER_ID;
            args.proxy_this = op;
            runtime->issue_runtime_meta_task(&args, sizeof(args),
                                             HLR_DEFERRED_MAPPING_TRIGGER_ID,
                                             op, map_precondition);
          }
          else
            must_epoch->add_mapping_dependence(map_precondition);  
          map_now = false;
        }
      }
      if (!resolution_dependences.empty())
      {
        Event resolve_precondition = 
          Event::merge_events(resolution_dependences);
        if (!resolve_precondition.has_triggered())
        {
          DeferredResolutionArgs args;
          args.hlr_id = HLR_DEFERRED_RESOLUTION_TRIGGER_ID;
          args.proxy_this = op;
          runtime->issue_runtime_meta_task(&args, sizeof(args),
                                           HLR_DEFERRED_RESOLUTION_TRIGGER_ID,
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
                                                              Internal *runtime)
    //--------------------------------------------------------------------------
    {
      if (!commit_dependences.empty())
      {
        Event commit_precondition = Event::merge_events(commit_dependences);
        if (!commit_precondition.has_triggered())
        {
          DeferredCommitArgs args;
          args.hlr_id = HLR_DEFERRED_COMMIT_ID;
          args.proxy_this = op;
          args.gen = op->get_generation();
          runtime->issue_runtime_meta_task(&args, sizeof(args),
                                           HLR_DEFERRED_COMMIT_ID,
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
    Predicate::Impl::Impl(Internal *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void Predicate::Impl::activate_predicate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      predicate_resolved = false;
      predicate_references = 0;
    }

    //--------------------------------------------------------------------------
    void Predicate::Impl::deactivate_predicate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
#ifdef DEBUG_HIGH_LEVEL
      assert(predicate_references == 0);
#endif
      waiters.clear();
    }

    //--------------------------------------------------------------------------
    void Predicate::Impl::add_predicate_reference(void)
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
    void Predicate::Impl::remove_predicate_reference(void)
    //--------------------------------------------------------------------------
    {
      bool need_trigger;
      bool remove_reference;
      GenerationID task_gen = 0;  // initialization to make gcc happy
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
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
    bool Predicate::Impl::register_waiter(PredicateWaiter *waiter,
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
#ifdef DEBUG_HIGH_LEVEL
        assert(waiters.find(waiter) == waiters.end());
#endif
        waiters[waiter] = waiter_gen;
        valid = false;
      }
      return valid;
    }

    //--------------------------------------------------------------------------
    void Predicate::Impl::set_resolved_value(GenerationID pred_gen, bool value)
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
    SpeculativeOp::SpeculativeOp(Internal *rt)
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
      predicate_waiter = UserEvent::NO_USER_EVENT;
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
      Event wait_event = Event::NO_EVENT;
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
#ifdef DEBUG_HIGH_LEVEL
          assert(predicate != NULL);
#endif
          predicate_waiter = UserEvent::create_user_event();
          wait_event = predicate_waiter;
        }
      }
      if (wait_event.exists())
      {
        if (!wait_event.has_triggered())
        {
          runtime->pre_wait(proc);
          wait_event.wait();
          runtime->post_wait(proc);
        }
        // Might be a little bit of a race here with cleanup
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
        assert((speculation_state == RESOLVE_TRUE_STATE) ||
               (speculation_state == RESOLVE_FALSE_STATE));
#endif
        if (speculation_state == RESOLVE_TRUE_STATE)
          resolve_true();
        else
          resolve_false();
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
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
        predicate_waiter.trigger();
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
        predicate_waiter.trigger();
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
    MapOp::MapOp(Internal *rt)
      : Inline(), Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MapOp::MapOp(const MapOp &rhs)
      : Inline(), Operation(NULL)
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
                               parent_ctx->variants->name, 
                               parent_ctx->get_unique_task_id());
      }
      requirement.copy_without_mapping_info(launcher.requirement);
      requirement.initialize_mapping_fields(); 
      map_id = launcher.map_id;
      tag = launcher.tag;
      termination_event = UserEvent::create_user_event();
      region = PhysicalRegion(legion_new<PhysicalRegion::Impl>(requirement,
                              completion_event, true/*mapped*/, ctx, 
                              map_id, tag, false/*leaf*/, runtime));
      if (check_privileges)
        check_privilege();
      initialize_privilege_path(privilege_path, requirement);
#ifdef LEGION_LOGGING
      LegionLogging::log_mapping_operation(
          parent_ctx->get_executing_processor(),
          parent_ctx->get_unique_op_id(),
          unique_op_id);
      LegionLogging::log_logical_requirement(
                                         parent_ctx->get_executing_processor(),
                                         unique_op_id, 0/*idx*/, true/*region*/,
                                         requirement.region.index_space.id,
                                         requirement.region.field_space.id,
                                         requirement.region.tree_id,
                                         requirement.privilege,
                                         requirement.prop,
                                         requirement.redop,
                                         requirement.privilege_fields);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_mapping_operation(parent_ctx->get_unique_task_id(),
                                       unique_op_id);
      LegionSpy::log_logical_requirement(unique_op_id,0/*index*/,
                                         true/*region*/,
                                         requirement.region.index_space.id,
                                         requirement.region.field_space.id,
                                         requirement.region.tree_id,
                                         requirement.privilege,
                                         requirement.prop,
                                         requirement.redop);
      LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                        requirement.privilege_fields);
#endif
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
                               parent_ctx->variants->name, 
                               parent_ctx->get_unique_task_id());
      }
      requirement.copy_without_mapping_info(req);
      requirement.initialize_mapping_fields();
      map_id = id;
      tag = t;
      parent_task = ctx;
      termination_event = UserEvent::create_user_event();
      region = PhysicalRegion(legion_new<PhysicalRegion::Impl>(requirement,
                              completion_event, true/*mapped*/, ctx, 
                              map_id, tag, false/*leaf*/, runtime));
      if (check_privileges)
        check_privilege();
      initialize_privilege_path(privilege_path, requirement);
#ifdef LEGION_LOGGING
      LegionLogging::log_mapping_operation(
          parent_ctx->get_executing_processor(),
          parent_ctx->get_unique_op_id(),
          unique_op_id);
      LegionLogging::log_logical_requirement(
                                         parent_ctx->get_executing_processor(),
                                         unique_op_id, 0/*idx*/, true/*region*/,
                                         requirement.region.index_space.id,
                                         requirement.region.field_space.id,
                                         requirement.region.tree_id,
                                         requirement.privilege,
                                         requirement.prop,
                                         requirement.redop,
                                         requirement.privilege_fields);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_mapping_operation(parent_ctx->get_unique_task_id(),
                                       unique_op_id);
      LegionSpy::log_logical_requirement(unique_op_id,0/*index*/,
                                         true/*region*/,
                                         requirement.region.index_space.id,
                                         requirement.region.field_space.id,
                                         requirement.region.tree_id,
                                         requirement.privilege,
                                         requirement.prop,
                                         requirement.redop);
      LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                        requirement.privilege_fields);
#endif
      return region;
    }

    //--------------------------------------------------------------------------
    void MapOp::initialize(SingleTask *ctx, const PhysicalRegion &reg)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      parent_task = ctx;
      requirement.copy_without_mapping_info(reg.impl->get_requirement());
      requirement.initialize_mapping_fields();
      map_id = reg.impl->map_id;
      tag = reg.impl->tag;
      parent_task = ctx;
      termination_event = UserEvent::create_user_event();
      region = reg;
      region.impl->remap_region(completion_event);
      // We're only really remapping it if it already had a physical
      // instance that we can use to make a valid value
      remap_region = region.impl->get_reference().has_ref();
      // No need to check the privileges here since we know that we have
      // them from the first time that we made this physical region
      initialize_privilege_path(privilege_path, requirement);
#ifdef LEGION_LOGGING
      LegionLogging::log_mapping_operation(
          parent_ctx->get_executing_processor(),
          parent_ctx->get_unique_op_id(),
          unique_op_id);
      LegionLogging::log_logical_requirement(
                                         parent_ctx->get_executing_processor(),
                                         unique_op_id, 0/*idx*/, true/*region*/,
                                         requirement.region.index_space.id,
                                         requirement.region.field_space.id,
                                         requirement.region.tree_id,
                                         requirement.privilege,
                                         requirement.prop,
                                         requirement.redop,
                                         requirement.privilege_fields);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_mapping_operation(parent_ctx->get_unique_task_id(), 
                                       unique_op_id);
      LegionSpy::log_logical_requirement(unique_op_id,0/*index*/,
                                         true/*region*/,
                                         requirement.region.index_space.id,
                                         requirement.region.field_space.id,
                                         requirement.region.tree_id,
                                         requirement.privilege,
                                         requirement.prop,
                                         requirement.redop);
      LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                        requirement.privilege_fields);
#endif
    }

    //--------------------------------------------------------------------------
    void MapOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      parent_ctx = NULL;
      remap_region = false;
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
      restrict_info.clear();
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
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, BEGIN_DEPENDENCE_ANALYSIS); 
#endif
      // First compute our parent region requirement
      compute_parent_index();  
      begin_dependence_analysis();
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   version_info,
                                                   restrict_info,
                                                   privilege_path);
      end_dependence_analysis();
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, END_DEPENDENCE_ANALYSIS);
#endif
    }

    //--------------------------------------------------------------------------
    void MapOp::trigger_remote_state_analysis(UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      std::set<Event> preconditions;  
      version_info.make_local(preconditions, runtime->forest,
                              physical_ctx.get_id());
      if (preconditions.empty())
        ready_event.trigger();
      else
        ready_event.trigger(Event::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    bool MapOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, BEGIN_MAPPING);
#endif
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      Processor local_proc = parent_ctx->get_executing_processor();
      // If we haven't already premapped the path, then do so now
      if (!requirement.premapped)
      {
        requirement.premapped = runtime->forest->premap_physical_region(
                  physical_ctx, privilege_path, requirement, version_info, 
                  this, parent_ctx, local_proc
#ifdef DEBUG_HIGH_LEVEL
                  , 0/*idx*/, get_logging_name(), unique_op_id
#endif
                  );
#ifdef DEBUG_HIGH_LEVEL
        assert(requirement.premapped);
#endif
      }
      MappingRef map_ref;
      bool notify = false;
      // If we are restricted we know the answer
      if (restrict_info.has_restrictions())
      {
        map_ref = runtime->forest->map_restricted_region(physical_ctx,
                                                         requirement,
                                                         0/*idx*/,
                                                         version_info,
                                                         local_proc
#ifdef DEBUG_HIGH_LEVEL
                                                         , get_logging_name()
                                                         , unique_op_id
#endif
                                                         );
#ifdef DEBUG_HIGH_LEVEL
        assert(map_ref.has_ref());
#endif
      }
      // If we are remapping then we also know the answer
      else if (remap_region)
      {
        InstanceRef target = region.impl->get_reference();
        // We're remapping the region so we don't actually
        // need to ask the mapper about anything when doing the remapping
        map_ref = runtime->forest->remap_physical_region(physical_ctx,
                                                         requirement,
                                                         0/*idx*/,
                                                         version_info,
                                                         target
#ifdef DEBUG_HIGH_LEVEL
                                                         , get_logging_name()
                                                         , unique_op_id
#endif
                                                         );
#ifdef DEBUG_HIGH_LEVEL
        assert(map_ref.has_ref());
#endif

      }
      else
      {
        // Now ask the mapper how to map this inline mapping operation 
        notify = runtime->invoke_mapper_map_inline(local_proc, this);
        // Do the mapping and see if it succeeded 
        map_ref = runtime->forest->map_physical_region(physical_ctx,
                                                       requirement,
                                                       0/*idx*/,
                                                       version_info,
                                                       this,
                                                       local_proc,
                                                       local_proc
#ifdef DEBUG_HIGH_LEVEL
                                                       , get_logging_name()
                                                       , unique_op_id
#endif
                                                       );
        if (!map_ref.has_ref())
        {
          requirement.mapping_failed = true;
          requirement.selected_memory = Memory::NO_MEMORY;
          // Tell the mapper that we failed to map
          runtime->invoke_mapper_failed_mapping(local_proc, this);
          // clear the version info
          version_info.reset();
          return false;
        }
      }
      InstanceRef result = runtime->forest->register_physical_region(
                                                                physical_ctx,
                                                                map_ref,
                                                                requirement,
                                                                0/*idx*/,
                                                                version_info,
                                                                this,
                                                                local_proc,
                                                            termination_event
#ifdef DEBUG_HIGH_LEVEL
                                                          , get_logging_name()
                                                          , unique_op_id
#endif
                                                            );
#ifdef DEBUG_HIGH_LEVEL
      assert(result.has_ref());
#endif
      // We're done so apply our mapping changes
      std::set<Event> applied_conditions;
      version_info.apply_mapping(physical_ctx.get_id(), 
                                 runtime->address_space, applied_conditions);
      // We succeeded in mapping, so set up our physical region with
      // the reference information.  Note that the physical region
      // becomes responsible for triggering the termination event
      // when it unmaps so we can recycle this mapping operation 
      // immediately.
      if (notify)
      {
        requirement.mapping_failed = false;
        requirement.selected_memory = result.get_memory();
        runtime->invoke_mapper_notify_result(local_proc, this);
      }
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, END_MAPPING);
      LegionLogging::log_operation_events(
          Processor::get_executing_processor(),
          unique_op_id, Event::NO_EVENT, result.get_ready_event());
      LegionLogging::log_event_dependence(
          Processor::get_executing_processor(),
          termination_event, parent_ctx->get_task_completion());
      LegionLogging::log_event_dependence(
          Processor::get_executing_processor(),
          result.get_ready_event(),
          completion_event);
      LegionLogging::log_event_dependence(
          Processor::get_executing_processor(),
          completion_event,
          termination_event);
      LegionLogging::log_physical_user(
          Processor::get_executing_processor(),
          result.get_manager()->get_instance(),
          unique_op_id, 0/*idx*/);
#endif
#ifdef LEGION_SPY
      // Log an implicit dependence on the parent's start event
      LegionSpy::log_implicit_dependence(parent_ctx->get_start_event(),
                                         result.get_ready_event());
      LegionSpy::log_op_events(unique_op_id, result.get_ready_event(),
                                termination_event);
      // Log an implicit dependence on the parent's term event
      LegionSpy::log_implicit_dependence(termination_event, 
                                         parent_ctx->get_task_completion());
      LegionSpy::log_op_user(unique_op_id, 0/*idx*/, 
          result.get_manager()->get_instance().id);
      {
        Processor proc = Processor::get_executing_processor();
        LegionSpy::log_op_proc_user(unique_op_id, proc.id);
      }
#endif
      // Have to do this before triggering the mapped event
      region.impl->reset_reference(result, termination_event);
      // Now we can trigger the mapping event and indicate
      // to all our mapping dependences that we are mapped.
      if (!applied_conditions.empty())
        complete_mapping(Event::merge_events(applied_conditions));
      else
        complete_mapping();
      
      Event map_complete_event = result.get_ready_event(); 
      if (!map_complete_event.has_triggered())
      {
        // Issue a deferred trigger on our completion event
        // and mark that we are no longer responsible for 
        // triggering our completion event
        completion_event.trigger(map_complete_event);
        need_completion_trigger = false;
        DeferredExecuteArgs deferred_execute_args;
        deferred_execute_args.hlr_id = HLR_DEFERRED_EXECUTION_TRIGGER_ID;
        deferred_execute_args.proxy_this = this;
        runtime->issue_runtime_meta_task(&deferred_execute_args,
                                         sizeof(deferred_execute_args),
                                         HLR_DEFERRED_EXECUTION_TRIGGER_ID,
                                         this, map_complete_event);
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
      Operation::trigger_commit();
    }

    //--------------------------------------------------------------------------
    unsigned MapOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    Mappable::MappableKind MapOp::get_mappable_kind(void) const
    //--------------------------------------------------------------------------
    {
      return INLINE_MAPPABLE;
    }

    //--------------------------------------------------------------------------
    Task* MapOp::as_mappable_task(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Copy* MapOp::as_mappable_copy(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Inline* MapOp::as_mappable_inline(void) const
    //--------------------------------------------------------------------------
    {
      MapOp *proxy_this = const_cast<MapOp*>(this);
      return proxy_this;
    }

    //--------------------------------------------------------------------------
    Acquire* MapOp::as_mappable_acquire(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Release* MapOp::as_mappable_release(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    UniqueID MapOp::get_unique_mappable_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
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
                               parent_ctx->variants->name);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_INVALID_INSTANCE_FIELD);
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            log_region.error("Instance field %d is a duplicate for "
                                    "inline mapping (ID %lld)",
                                    bad_field, unique_op_id);
#ifdef DEBUG_HIGH_LEVEL
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
                                   parent_ctx->variants->name, 
                                   parent_ctx->get_unique_task_id(),
                                   unique_op_id, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
                                   parent_ctx->variants->name, 
                                   parent_ctx->get_unique_task_id(),
                                   unique_op_id, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_BAD_PARENT_REGION);
      }
      else
        parent_req_index = unsigned(parent_index);
    }

    /////////////////////////////////////////////////////////////
    // Copy Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyOp::CopyOp(Internal *rt)
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
      src_restrictions.resize(launcher.src_requirements.size());
      dst_restrictions.resize(launcher.dst_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        if (launcher.src_requirements[idx].privilege_fields.empty())
        {
          log_task.warning("WARNING: SOURCE REGION REQUIREMENT %d OF "
                                 "COPY (ID %lld) IN TASK %s (ID %lld) HAS NO "
                                 "PRIVILEGE FIELDS! DID YOU FORGET THEM?!?",
                                 idx, get_unique_op_id(),
                                 parent_ctx->variants->name, 
                                 parent_ctx->get_unique_task_id());
        }
        src_requirements[idx].copy_without_mapping_info(
            launcher.src_requirements[idx]);
        src_requirements[idx].initialize_mapping_fields();
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
                                 parent_ctx->variants->name, 
                                 parent_ctx->get_unique_task_id());
        }
        dst_requirements[idx].copy_without_mapping_info(
            launcher.dst_requirements[idx]);
        dst_requirements[idx].initialize_mapping_fields();
        dst_requirements[idx].flags |= NO_ACCESS_FLAG;
      }
      grants = launcher.grants;
      // Register ourselves with all the grants
      for (unsigned idx = 0; idx < grants.size(); idx++)
        grants[idx].impl->register_operation(completion_event);
      wait_barriers = launcher.wait_barriers;
      for (std::vector<PhaseBarrier>::const_iterator it = 
            launcher.arrive_barriers.begin(); it != 
            launcher.arrive_barriers.end(); it++)
      {
        arrive_barriers.push_back(*it);
#ifdef LEGION_LOGGING
        LegionLogging::log_event_dependence(
            Processor::get_executing_processor(),
            it->phase_barrier, arrive_barriers.back().phase_barrier);
#endif
#ifdef LEGION_SPY
        LegionSpy::log_event_dependence(it->phase_barrier,
                                arrive_barriers.back().phase_barrier);
#endif
      }
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
                              get_unique_copy_id(), parent_ctx->variants->name,
                              parent_ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
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
                                idx, get_unique_copy_id(), 
                                parent_ctx->variants->name,
                                parent_ctx->get_unique_task_id(),
                                src_requirements[idx].privilege_fields.size(),
                                src_requirements[idx].instance_fields.size());
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_INVALID_COPY_FIELDS_SIZE);
          }
          if (!IS_READ_ONLY(src_requirements[idx]))
          {
            log_run.error("Copy source requirement %d for copy operation "
                                "(ID %lld) in parent task %s (ID %lld) must "
                                "be requested with a read-only privilege.",
                                idx, get_unique_copy_id(),
                                parent_ctx->variants->name,
                                parent_ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
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
                                get_unique_copy_id(), 
                                parent_ctx->variants->name,
                                parent_ctx->get_unique_task_id(),
                                dst_requirements[idx].privilege_fields.size(),
                                dst_requirements[idx].instance_fields.size());
#ifdef DEBUG_HIGH_LEVEL
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
                                idx, get_unique_copy_id(),
                                parent_ctx->variants->name,
                                parent_ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
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
                                idx, get_unique_copy_id(),
                                parent_ctx->variants->name, 
                                parent_ctx->get_unique_task_id(),
                                src_space.id, dst_space.id);
#ifdef DEBUG_HIGH_LEVEL
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
                                dst_space.id, idx, get_unique_copy_id(),
                                parent_ctx->variants->name,
                                parent_ctx->get_unique_task_id(),
                                src_space.id);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef LEGION_LOGGING
      LegionLogging::log_copy_operation(parent_ctx->get_executing_processor(),
                                        parent_ctx->get_unique_op_id(),
                                        unique_op_id);
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        const RegionRequirement &req = src_requirements[idx];
        LegionLogging::log_logical_requirement(
                                        parent_ctx->get_executing_processor(),
                                        unique_op_id, idx, true/*region*/,
                                        req.region.index_space.id,
                                        req.region.field_space.id,
                                        req.region.tree_id,
                                        req.privilege,
                                        req.prop,
                                        req.redop,
                                        req.privilege_fields);
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        const RegionRequirement &req = dst_requirements[idx];
        LegionLogging::log_logical_requirement(
                                        parent_ctx->get_executing_processor(),
                                        unique_op_id, 
                                        src_requirements.size()+idx, 
                                        true/*region*/,
                                        req.region.index_space.id,
                                        req.region.field_space.id,
                                        req.region.tree_id,
                                        req.privilege,
                                        req.prop,
                                        req.redop,
                                        req.privilege_fields);
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_copy_operation(parent_ctx->get_unique_task_id(),
                                    unique_op_id);
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        const RegionRequirement &req = src_requirements[idx];
        LegionSpy::log_logical_requirement(unique_op_id, idx, true/*region*/,
                                           req.region.index_space.id,
                                           req.region.field_space.id,
                                           req.region.tree_id,
                                           req.privilege,
                                           req.prop, req.redop);
        LegionSpy::log_requirement_fields(unique_op_id, idx, 
                                          req.privilege_fields);
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
                                           req.prop, req.redop);
        LegionSpy::log_requirement_fields(unique_op_id, 
                                          src_requirements.size()+idx, 
                                          req.privilege_fields);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void CopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative();
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
      src_restrictions.clear();
      dst_restrictions.clear();
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
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, BEGIN_DEPENDENCE_ANALYSIS);
#endif
      // First compute the parent indexes
      compute_parent_indexes(); 
      begin_dependence_analysis();
      // Register a dependence on our predicate
      register_predicate_dependence();
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        runtime->forest->perform_dependence_analysis(this, idx, 
                                                     src_requirements[idx],
                                                     src_versions[idx],
                                                     src_restrictions[idx],
                                                     src_privilege_paths[idx]);
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        unsigned index = src_requirements.size()+idx;
        runtime->forest->perform_dependence_analysis(this, index, 
                                                     dst_requirements[idx],
                                                     dst_versions[idx],
                                                     dst_restrictions[idx],
                                                     dst_privilege_paths[idx]);
      }
      end_dependence_analysis();
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, END_DEPENDENCE_ANALYSIS);
#endif
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_remote_state_analysis(UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<Event> preconditions;
      for (unsigned idx = 0; idx < src_versions.size(); idx++)
      {
        RegionTreeContext physical_ctx =
          parent_ctx->find_enclosing_context(src_parent_indexes[idx]);
        src_versions[idx].make_local(preconditions, runtime->forest, 
                                     physical_ctx.get_id());
      }
      for (unsigned idx = 0; idx < dst_versions.size(); idx++)
      {
        RegionTreeContext physical_ctx =
          parent_ctx->find_enclosing_context(dst_parent_indexes[idx]);
        dst_versions[idx].make_local(preconditions, runtime->forest, 
                                     physical_ctx.get_id());
      }
      if (preconditions.empty())
        ready_event.trigger();
      else
        ready_event.trigger(Event::merge_events(preconditions));
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
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    bool CopyOp::speculate(bool &value)
    //--------------------------------------------------------------------------
    {
      Processor exec_proc = parent_ctx->get_executing_processor();
      return runtime->invoke_mapper_speculate(exec_proc, this, value);
    }

    //--------------------------------------------------------------------------
    bool CopyOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, BEGIN_MAPPING);
#endif
      bool map_success = true;
      std::vector<RegionTreeContext> src_contexts(src_requirements.size());
      std::vector<RegionTreeContext> dst_contexts(dst_requirements.size());
      // Premap all the regions if we haven't already done so
      bool premapped = true;
      Processor local_proc = parent_ctx->get_executing_processor();
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        src_contexts[idx] = parent_ctx->find_enclosing_context(
                                                  src_parent_indexes[idx]);
        if (!src_requirements[idx].premapped)
        {
          src_requirements[idx].premapped = 
            runtime->forest->premap_physical_region(
                  src_contexts[idx],src_privilege_paths[idx],
                  src_requirements[idx], src_versions[idx],
                  this, parent_ctx, local_proc
#ifdef DEBUG_HIGH_LEVEL
                  , idx, get_logging_name(), unique_op_id
#endif
                  );
#ifdef DEBUG_HIGH_LEVEL
          assert(src_requirements[idx].premapped);
#endif
        }
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        dst_contexts[idx] = parent_ctx->find_enclosing_context(
                                                  dst_parent_indexes[idx]);
        if (!dst_requirements[idx].premapped)
        {
          dst_requirements[idx].premapped = 
            runtime->forest->premap_physical_region(
                  dst_contexts[idx],dst_privilege_paths[idx],
                  dst_requirements[idx], dst_versions[idx],
                  this, parent_ctx, local_proc
#ifdef DEBUG_HIGH_LEVEL
                  , src_requirements.size()+idx
                  , get_logging_name(), unique_op_id
#endif
                  );
#ifdef DEBUG_HIGH_LEVEL
          assert(dst_requirements[idx].premapped);
#endif
        }
      }
      // If we couldn't premap, then we need to try again later
      if (!premapped)
        return false;
      // Now ask the mapper how to map this copy operation
      bool notify = runtime->invoke_mapper_map_copy(local_proc, this);
      // Map all the destination instances
      LegionVector<MappingRef>::aligned 
                            src_mapping_refs(src_requirements.size());
      for (unsigned idx = 0; (idx < src_requirements.size()) && 
            map_success; idx++)
      {
        // If there is no ranking for this instance then we can skip
        // it as we will just issue copies from the existing valid instances
        if (src_requirements[idx].target_ranking.empty())
          continue;
        // If this is a restricted instance then we need to
        // map wherever the existing physical instance was
        if (src_restrictions[idx].has_restrictions())
        {
          src_mapping_refs[idx] = runtime->forest->map_restricted_region(
                                                        src_contexts[idx],
                                                        src_requirements[idx],
                                                        idx, 
                                                        src_versions[idx],
                                                        local_proc
#ifdef DEBUG_HIGH_LEVEL
                                                        , get_logging_name()
                                                        , unique_op_id
#endif
                                                        );
#ifdef DEBUG_HIGH_LEVEL
          assert(src_mapping_refs[idx].has_ref());
#endif
          continue;
        }
        src_mapping_refs[idx] = runtime->forest->map_physical_region(
                                                        src_contexts[idx],
                                                        src_requirements[idx],
                                                        idx,
                                                        src_versions[idx],
                                                        this,
                                                        local_proc,
                                                        local_proc
#ifdef DEBUG_HIGH_LEVEL
                                                        , get_logging_name()
                                                        , unique_op_id
#endif
                                                        );
        if (!src_mapping_refs[idx].has_ref())
        {
          map_success = false;
          src_requirements[idx].mapping_failed = true;
          src_requirements[idx].selected_memory = Memory::NO_MEMORY;
          break;
        }
      }
      LegionVector<MappingRef>::aligned 
                          dst_mapping_refs(dst_requirements.size());
      for (unsigned idx = 0; (idx < dst_requirements.size()) && 
            map_success; idx++)
      {
        // If this is a restricted instance, then we need to map
        // wherever the existing physical instance was
        if (dst_restrictions[idx].has_restrictions())
        {
          // Little bit of a hack here: if this is a restricted reduction,
          // we actually want to map to a normal instance, so make it look
          // like the privileges are read-write while selecting the instance
          // and then switch back after we are done
          if (IS_REDUCE(dst_requirements[idx]))
          {
            dst_requirements[idx].privilege = READ_WRITE;
            dst_mapping_refs[idx] = runtime->forest->map_restricted_region(
                                                      dst_contexts[idx],
                                                      dst_requirements[idx],
                                                      src_requirements.size()+idx,
                                                      dst_versions[idx],
                                                      local_proc
#ifdef DEBUG_HIGH_LEVEL
                                                      , get_logging_name()
                                                      , unique_op_id
#endif
                                                      );
            // Switch the privileges back
            dst_requirements[idx].privilege = REDUCE;
          }
          else // The normal thing
            dst_mapping_refs[idx] = runtime->forest->map_restricted_region(
                                                      dst_contexts[idx],
                                                      dst_requirements[idx],
                                                      src_requirements.size()+idx,
                                                      dst_versions[idx],
                                                      local_proc
#ifdef DEBUG_HIGH_LEVEL
                                                      , get_logging_name()
                                                      , unique_op_id
#endif
                                                      );
#ifdef DEBUG_HIGH_LEVEL
          assert(dst_mapping_refs[idx].has_ref());
#endif
          continue;
        }
        dst_mapping_refs[idx] = runtime->forest->map_physical_region(
                                                    dst_contexts[idx],
                                                    dst_requirements[idx],
                                                    src_requirements.size()+idx,
                                                    dst_versions[idx],
                                                    this,
                                                    local_proc,
                                                    local_proc
#ifdef DEBUG_HIGH_LEVEL
                                                    , get_logging_name()
                                                    , unique_op_id
#endif
                                                    );
        if (!dst_mapping_refs[idx].has_ref())
        {
          map_success = false;
          dst_requirements[idx].mapping_failed = true;
          dst_requirements[idx].selected_memory = Memory::NO_MEMORY;
          // Break out since we failed
          break;
        }
      }

      // If we successfully mapped, then we can issue the copies
      // These should be guaranteed to succeed since no new 
      // regions will need to be created because we already
      // made all the destination regions.
      if (map_success)
      {
        // First get the set of barriers and grants we need to wait 
        // on before we can start running this copy.
        Event sync_precondition = Event::NO_EVENT;
        if (!wait_barriers.empty() || !grants.empty())
        {
          std::set<Event> preconditions;
          for (std::vector<PhaseBarrier>::const_iterator it = 
                wait_barriers.begin(); it != wait_barriers.end(); it++)
          {
            Event e = it->phase_barrier.get_previous_phase(); 
            preconditions.insert(e);
          }
          for (std::vector<Grant>::const_iterator it = grants.begin();
                it != grants.end(); it++)
          {
            Event e = it->impl->acquire_grant();
            preconditions.insert(e);
          }
          sync_precondition = Event::merge_events(preconditions);
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
          if (!sync_precondition.exists())
          {
            UserEvent new_pre = UserEvent::create_user_event();
            new_pre.trigger();
            sync_precondition = new_pre;
          }
#endif
#ifdef LEGION_LOGGING
          LegionLogging::log_event_dependences(
              Processor::get_executing_processor(), preconditions,
                                              sync_precondition);
#endif
#ifdef LEGION_SPY
          LegionSpy::log_event_dependences(preconditions,
                                           sync_precondition);
#endif
        }
#ifdef LEGION_SPY
        std::set<Event> start_events;
        start_events.insert(sync_precondition);
#endif
        std::set<Event> applied_conditions;
        std::set<Event> copy_complete_events;
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        {
          InstanceRef dst_ref = runtime->forest->register_physical_region(
                                                        dst_contexts[idx],
                                                        dst_mapping_refs[idx],
                                                        dst_requirements[idx],
                                                        idx,
                                                        dst_versions[idx],
                                                        this,
                                                        local_proc,
                                                        completion_event
#ifdef DEBUG_HIGH_LEVEL
                                                        , get_logging_name()
                                                        , unique_op_id
#endif
                                                        );
#ifdef DEBUG_HIGH_LEVEL
          assert(dst_ref.has_ref());
#endif
          if (notify)
          {
            dst_requirements[idx].mapping_failed = false;
            dst_requirements[idx].selected_memory = dst_ref.get_memory();
          }
#ifdef LEGION_SPY
          start_events.insert(dst_ref.get_ready_event());
          LegionSpy::log_op_user(unique_op_id, src_requirements.size()+idx,
              dst_ref.get_manager()->get_instance().id);
#endif
          if (!src_mapping_refs[idx].has_ref())
          {
            // In this case, there is no source instance so we need
            // to issue copies from the valid set
            if (!IS_REDUCE(dst_requirements[idx]))
              copy_complete_events.insert(runtime->forest->copy_across(this, 
                                          parent_ctx->get_executing_processor(),
                                                   src_contexts[idx],
                                                   dst_contexts[idx],
                                                   src_requirements[idx],
                                                   src_versions[idx],
                                                   dst_requirements[idx],
                                                   dst_ref, sync_precondition));
            else
              copy_complete_events.insert(runtime->forest->reduce_across(this,
                                          parent_ctx->get_executing_processor(),
                                                   src_contexts[idx],
                                                   dst_contexts[idx],
                                                   src_requirements[idx],
                                                   src_versions[idx],
                                                   dst_requirements[idx],
                                                   dst_ref, sync_precondition));
          }
          else
          {
            InstanceRef src_ref = runtime->forest->register_physical_region(
                                                        src_contexts[idx],
                                                        src_mapping_refs[idx],
                                                        src_requirements[idx],
                                                        idx,
                                                        src_versions[idx],
                                                        this,
                                                        local_proc,
                                                        completion_event
#ifdef DEBUG_HIGH_LEVEL
                                                        , get_logging_name()
                                                        , unique_op_id
#endif
                                                        );
            
#ifdef DEBUG_HIGH_LEVEL
            assert(src_ref.has_ref());
#endif
            // Apply our changes to the state
            src_versions[idx].apply_mapping(src_contexts[idx].get_id(),
                                runtime->address_space, applied_conditions);
            dst_versions[idx].apply_mapping(dst_contexts[idx].get_id(),
                                runtime->address_space, applied_conditions);
            if (notify)
            {
              src_requirements[idx].mapping_failed = false;
              src_requirements[idx].selected_memory = src_ref.get_memory();
            }
            // Now issue the copies from source to destination
            if (!IS_REDUCE(dst_requirements[idx]))
              copy_complete_events.insert(
                  runtime->forest->copy_across(this, src_contexts[idx],
                                          dst_contexts[idx],
                                          src_requirements[idx],
                                          dst_requirements[idx],
                                          src_ref, dst_ref, sync_precondition));
            else
              copy_complete_events.insert(
                  runtime->forest->reduce_across(this, dst_contexts[idx],
                                          dst_contexts[idx],
                                          src_requirements[idx],
                                          dst_requirements[idx],
                                          src_ref, dst_ref, sync_precondition));
#ifdef LEGION_SPY
            start_events.insert(src_ref.get_ready_event());
            LegionSpy::log_op_user(unique_op_id, idx,
                src_ref.get_manager()->get_instance().id);
#endif
          }
        }
#ifdef LEGION_LOGGING
        LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                        unique_op_id, END_MAPPING);
#endif
        // Launch the complete task if necessary 
        Event copy_complete_event = 
          Event::merge_events(copy_complete_events);
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
        if (!copy_complete_event.exists())
        {
          UserEvent new_copy_complete = UserEvent::create_user_event();
          new_copy_complete.trigger();
          copy_complete_event = new_copy_complete;
        }
#endif
#ifdef LEGION_LOGGING
        LegionLogging::log_event_dependences(
                                    Processor::get_executing_processor(),
                                    copy_complete_events, copy_complete_event);
#endif
#ifdef LEGION_SPY
        Event start_event = Event::merge_events(start_events);
        if (!start_event.exists())
        {
          UserEvent new_start_event = UserEvent::create_user_event();
          new_start_event.trigger();
          start_event = new_start_event;
        }
        LegionSpy::log_event_dependences(start_events, start_event);
        LegionSpy::log_op_events(unique_op_id, start_event,
                                 completion_event);
        LegionSpy::log_event_dependences(copy_complete_events, 
                                         copy_complete_event);
        LegionSpy::log_event_dependence(copy_complete_event,
                                        completion_event);
        {
          Processor proc = Processor::get_executing_processor();
          LegionSpy::log_op_proc_user(unique_op_id, proc.id);
        }
#endif
        // Chain all the unlock and barrier arrivals off of the
        // copy complete event
        if (!arrive_barriers.empty())
        {
          for (std::vector<PhaseBarrier>::const_iterator it = 
                arrive_barriers.begin(); it != arrive_barriers.end(); it++)
          {
            it->phase_barrier.arrive(1/*count*/, copy_complete_event);    
#ifdef LEGION_LOGGING
            LegionLogging::log_event_dependence(
                Processor::get_executing_processor(),       
                copy_complete_event, it->phase_barrier);
#endif
#ifdef LEGION_SPY
            LegionSpy::log_event_dependence(completion_event, 
                                            it->phase_barrier);
#endif
          }
        }

        // Mark that we completed mapping
        if (!applied_conditions.empty())
          complete_mapping(Event::merge_events(applied_conditions));
        else
          complete_mapping();
        // Notify the mapper if it wanted to be notified
        if (notify)
          runtime->invoke_mapper_notify_result(local_proc, this);

#ifdef LEGION_LOGGING
        LegionLogging::log_event_dependence(
                                        Processor::get_executing_processor(),
                                        copy_complete_event,
                                        completion_event);
#endif
#ifdef LEGION_SPY
        LegionSpy::log_event_dependence(copy_complete_event,
                                        completion_event);
#endif
        // Handle the case for marking when the copy completes
        completion_event.trigger(copy_complete_event);
        need_completion_trigger = false;
        complete_execution(copy_complete_event);
      }
      else
      {
        // We failed to map, so notify the mapper
        runtime->invoke_mapper_failed_mapping(local_proc, this);
        // Clear our our instances that did map so we start
        // again next time.
        src_mapping_refs.clear();
        dst_mapping_refs.clear();
        for (unsigned idx = 0; idx < src_versions.size(); idx++)
          src_versions[idx].reset();
        for (unsigned idx = 0; idx < dst_versions.size(); idx++)
          dst_versions[idx].reset();
      }
      return map_success;
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
      Operation::trigger_commit();
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
                          "requirements and %d of %s requirements aliased for "
                          "copy operation (UID %lld) in task %s (UID %lld).",
                          actual_idx1, is_src1 ? "source" : "destination",
                          actual_idx2, is_src2 ? "source" : "destination",
                          unique_op_id, parent_ctx->variants->name,
                          parent_ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
        assert(idx < dst_parent_indexes.size());
#endif
        return dst_parent_indexes[idx];
      }
      else
        return src_parent_indexes[idx];
    }

    //--------------------------------------------------------------------------
    Mappable::MappableKind CopyOp::get_mappable_kind(void) const
    //--------------------------------------------------------------------------
    {
      return COPY_MAPPABLE;
    }

    //--------------------------------------------------------------------------
    Task* CopyOp::as_mappable_task(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Copy* CopyOp::as_mappable_copy(void) const
    //--------------------------------------------------------------------------
    {
      CopyOp *proxy_this = const_cast<CopyOp*>(this);
      return proxy_this;
    }
    
    //--------------------------------------------------------------------------
    Inline* CopyOp::as_mappable_inline(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }
    
    //--------------------------------------------------------------------------
    Acquire* CopyOp::as_mappable_acquire(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Release* CopyOp::as_mappable_release(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    UniqueID CopyOp::get_unique_mappable_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id; 
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
                               parent_ctx->variants->name);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
                                   "space %d for index %d of %s requirements"
                                   "of copy operation (ID %lld)",
                                   bad_field, sp.id, idx, 
                                   (src ? "source" : "destination"),
                                   unique_op_id);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
                                   parent_ctx->variants->name, 
                                   parent_ctx->get_unique_task_id(),
                                   unique_op_id, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id,
                                   idx, (src ? "source" : "destination"));
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
                                   parent_ctx->variants->name, 
                                   parent_ctx->get_unique_task_id(),
                                   unique_op_id, 
                                   src_requirements[idx].region.index_space.id,
                                   src_requirements[idx].region.field_space.id, 
                                   src_requirements[idx].region.tree_id, idx);
#ifdef DEBUG_HIGH_LEVEL
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
                                   parent_ctx->variants->name, 
                                   parent_ctx->get_unique_task_id(),
                                   unique_op_id, 
                                   dst_requirements[idx].region.index_space.id,
                                   dst_requirements[idx].region.field_space.id, 
                                   dst_requirements[idx].region.tree_id, idx);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_BAD_PARENT_REGION);
        }
        else
          dst_parent_indexes[idx] = unsigned(parent_index);
      }
    }

    /////////////////////////////////////////////////////////////
    // Fence Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FenceOp::FenceOp(Internal *rt)
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
#ifdef LEGION_LOGGING
      LegionLogging::log_fence_operation(parent_ctx->get_executing_processor(),
                                         parent_ctx->get_unique_op_id(),
                                         unique_op_id);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_fence_operation(parent_ctx->get_unique_task_id(),
                                     unique_op_id);
#endif
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
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, BEGIN_DEPENDENCE_ANALYSIS);
#endif
      begin_dependence_analysis();
      // Register this fence with all previous users in the parent's context
      RegionTreeContext ctx = parent_ctx->get_context();
      for (unsigned idx = 0; idx < parent_ctx->regions.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(parent_ctx->regions[idx].handle_type == SINGULAR);
#endif
        runtime->forest->perform_fence_analysis(ctx, this, 
                          parent_ctx->regions[idx].region, true/*dominate*/);
      }
      // Now update the parent context with this fence
      // before we can complete the dependence analysis
      // and possibly be deactivated
      parent_ctx->update_current_fence(this);
      end_dependence_analysis();
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, END_DEPENDENCE_ANALYSIS);
#endif
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
            std::set<Event> trigger_events;
            for (std::map<Operation*,GenerationID>::const_iterator it = 
                  incoming.begin(); it != incoming.end(); it++)
            {
              Event complete = it->first->get_completion_event();
              if (it->second == it->first->get_generation())
                trigger_events.insert(complete);
            }
            Event wait_on = Event::merge_events(trigger_events);
            if (!wait_on.has_triggered())
            {
              DeferredExecuteArgs deferred_execute_args;
              deferred_execute_args.hlr_id = HLR_DEFERRED_EXECUTION_TRIGGER_ID;
              deferred_execute_args.proxy_this = this;
              runtime->issue_runtime_meta_task(&deferred_execute_args,
                                               sizeof(deferred_execute_args),
                                              HLR_DEFERRED_EXECUTION_TRIGGER_ID,
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
    FrameOp::FrameOp(Internal *rt)
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
    void FrameOp::set_previous(Event previous)
    //--------------------------------------------------------------------------
    {
      previous_completion = previous;
    }

    //--------------------------------------------------------------------------
    void FrameOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      previous_completion = Event::NO_EVENT;
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
      std::set<Event> trigger_events;
      // Include our previous completion event if necessary
      if (previous_completion.exists())
        trigger_events.insert(previous_completion);
      for (std::map<Operation*,GenerationID>::const_iterator it = 
            incoming.begin(); it != incoming.end(); it++)
      {
        Event complete = it->first->get_completion_event();
        if (it->second == it->first->get_generation())
          trigger_events.insert(complete);
      }
      Event wait_on = Event::merge_events(trigger_events);
      if (!wait_on.has_triggered())
      {
        DeferredExecuteArgs deferred_execute_args;
        deferred_execute_args.hlr_id = HLR_DEFERRED_EXECUTION_TRIGGER_ID;
        deferred_execute_args.proxy_this = this;
        runtime->issue_runtime_meta_task(&deferred_execute_args,
                                         sizeof(deferred_execute_args),
                                         HLR_DEFERRED_EXECUTION_TRIGGER_ID,
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
    DeletionOp::DeletionOp(Internal *rt)
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
#ifdef LEGION_LOGGING
      LegionLogging::log_deletion_operation(
          parent_ctx->get_executing_processor(),
          parent_ctx->get_unique_op_id(), unique_op_id);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deletion_operation(parent_ctx->get_unique_task_id(),
                                        unique_op_id);
#endif
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_index_part_deletion(SingleTask *ctx,
                                                    IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      kind = INDEX_PARTITION_DELETION;
      index_part = handle;
#ifdef LEGION_LOGGING
      LegionLogging::log_deletion_operation(
          parent_ctx->get_executing_processor(),
          parent_ctx->get_unique_op_id(), unique_op_id);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deletion_operation(parent_ctx->get_unique_task_id(),
                                        unique_op_id);
#endif
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_field_space_deletion(SingleTask *ctx,
                                                     FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      kind = FIELD_SPACE_DELETION;
      field_space = handle;
#ifdef LEGION_LOGGING
      LegionLogging::log_deletion_operation(
          parent_ctx->get_executing_processor(),
          parent_ctx->get_unique_op_id(), unique_op_id);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deletion_operation(parent_ctx->get_unique_task_id(),
                                        unique_op_id);
#endif
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
#ifdef LEGION_LOGGING
      LegionLogging::log_deletion_operation(
          parent_ctx->get_executing_processor(),
          parent_ctx->get_unique_op_id(), unique_op_id);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deletion_operation(parent_ctx->get_unique_task_id(),
                                        unique_op_id);
#endif
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
#ifdef LEGION_LOGGING
      LegionLogging::log_deletion_operation(
          parent_ctx->get_executing_processor(),
          parent_ctx->get_unique_op_id(), unique_op_id);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deletion_operation(parent_ctx->get_unique_task_id(),
                                        unique_op_id);
#endif
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_logical_region_deletion(SingleTask *ctx,
                                                        LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      kind = LOGICAL_REGION_DELETION;
      logical_region = handle;
#ifdef LEGION_LOGGING
      LegionLogging::log_deletion_operation(
          parent_ctx->get_executing_processor(),
          parent_ctx->get_unique_op_id(), unique_op_id);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deletion_operation(parent_ctx->get_unique_task_id(),
                                        unique_op_id);
#endif
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_logical_partition_deletion(SingleTask *ctx,
                                                       LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      kind = LOGICAL_PARTITION_DELETION;
      logical_part = handle;
#ifdef LEGION_LOGGING
      LegionLogging::log_deletion_operation(
          parent_ctx->get_executing_processor(),
          parent_ctx->get_unique_op_id(), unique_op_id);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deletion_operation(parent_ctx->get_unique_task_id(),
                                        unique_op_id);
#endif
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
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, BEGIN_DEPENDENCE_ANALYSIS);
#endif
      begin_dependence_analysis();
      switch (kind)
      {
        case INDEX_SPACE_DELETION:
          {
            parent_ctx->analyze_destroy_index_space(index_space, this);
            break;
          }
        case INDEX_PARTITION_DELETION:
          {
            parent_ctx->analyze_destroy_index_partition(index_part, this);
            break;
          }
        case FIELD_SPACE_DELETION:
          {
            parent_ctx->analyze_destroy_field_space(field_space, this);
            break;
          }
        case FIELD_DELETION:
          {
            parent_ctx->analyze_destroy_fields(field_space, this, free_fields);
            break;
          }
        case LOGICAL_REGION_DELETION:
          {
            parent_ctx->analyze_destroy_logical_region(logical_region, this);
            break;
          }
        case LOGICAL_PARTITION_DELETION:
          {
            parent_ctx->analyze_destroy_logical_partition(logical_part, this);
            break;
          }
        default:
          // should never get here
          assert(false);
      }
      end_dependence_analysis();
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, END_DEPENDENCE_ANALYSIS);
#endif
    }

    //--------------------------------------------------------------------------
    void DeletionOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // By not actually doing any operations until commit time we ensure
      // that the deletions don't actually impact the region tree state
      // until we know that it is safe to do so.
      switch (kind)
      {
        case INDEX_SPACE_DELETION:
          {
            bool top_level = runtime->finalize_index_space_destroy(index_space);
            if (top_level)
              parent_ctx->register_index_space_deletion(index_space);
            break;
          }
        case INDEX_PARTITION_DELETION:
          {
            runtime->finalize_index_partition_destroy(index_part);
            break;
          }
        case FIELD_SPACE_DELETION:
          {
            runtime->finalize_field_space_destroy(field_space);
            parent_ctx->register_field_space_deletion(field_space);
            break;
          }
        case FIELD_DELETION:
          {
            runtime->finalize_field_destroy(field_space, free_fields);
            parent_ctx->register_field_deletions(field_space, free_fields);
            break;
          }
        case LOGICAL_REGION_DELETION:
          {
            bool top_level = runtime->finalize_logical_region_destroy(
                                                        logical_region);
            // If this was a top-level region destruction
            // tell the enclosing parent task it has lost privileges
            if (top_level)
              parent_ctx->register_region_deletion(logical_region);
            break;
          }
        case LOGICAL_PARTITION_DELETION:
          {
            runtime->finalize_logical_partition_destroy(logical_part);
            break;
          }
        default:
          assert(false); // should never get here
      }
#ifdef LEGION_LOGGING
      LegionLogging::log_operation_events(
          Processor::get_executing_processor(),
          unique_op_id, Event::NO_EVENT, completion_event);
#endif
      // Commit this operation
      commit_operation();
      // Then deactivate it
      deactivate();
    }

    /////////////////////////////////////////////////////////////
    // Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CloseOp::CloseOp(Internal *rt)
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
#ifdef DEBUG_HIGH_LEVEL
      assert(completion_event.exists());
#endif
      initialize_operation(ctx, track);
      requirement.copy_without_mapping_info(req);
      requirement.initialize_mapping_fields();
      initialize_privilege_path(privilege_path, requirement);
    } 

    //--------------------------------------------------------------------------
    void CloseOp::perform_logging(unsigned is_inter_close_op /* = 0*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_LOGGING
      LegionLogging::log_close_operation(parent_ctx->get_executing_processor(),
                                         parent_ctx->get_unique_op_id(),
                                         unique_op_id);
      LegionLogging::log_logical_requirement(
                                         parent_ctx->get_executing_processor(),
                                         unique_op_id, 0/*idx*/, true/*region*/,
                                         requirement.region.index_space.id,
                                         requirement.region.field_space.id,
                                         requirement.region.tree_id,
                                         requirement.privilege,
                                         requirement.prop,
                                         requirement.redop,
                                         requirement.privilege_fields);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_close_operation(parent_ctx->get_unique_task_id(),
                                     unique_op_id,
                                     is_inter_close_op);
      if (requirement.handle_type == PART_PROJECTION)
        LegionSpy::log_logical_requirement(unique_op_id, 0/*idx*/,
                                  false/*region*/,
                                  requirement.partition.index_partition.id,
                                  requirement.partition.field_space.id,
                                  requirement.partition.tree_id,
                                  requirement.privilege,
                                  requirement.prop,
                                  requirement.redop);
      else
        LegionSpy::log_logical_requirement(unique_op_id, 0/*idx*/,
                                  true/*region*/,
                                  requirement.region.index_space.id,
                                  requirement.region.field_space.id,
                                  requirement.region.tree_id,
                                  requirement.privilege,
                                  requirement.prop,
                                  requirement.redop);
      LegionSpy::log_requirement_fields(unique_op_id, 0/*idx*/,
                                requirement.privilege_fields);
#endif
    } 

    //--------------------------------------------------------------------------
    void CloseOp::activate_close(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
#ifdef DEBUG_HIGH_LEVEL
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
    void CloseOp::trigger_remote_state_analysis(UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(find_parent_index(0));
      std::set<Event> preconditions;
      version_info.make_local(preconditions, runtime->forest,
                              physical_ctx.get_id());
      if (preconditions.empty())
        ready_event.trigger();
      else
        ready_event.trigger(Event::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    void CloseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      Operation::trigger_commit();
    }

    /////////////////////////////////////////////////////////////
    // Inter Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InterCloseOp::InterCloseOp(Internal *runtime)
      : CloseOp(runtime)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InterCloseOp::InterCloseOp(const InterCloseOp &rhs)
      : CloseOp(NULL)
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
                                  const std::set<ColorPoint> &targets,bool open,
                                  LegionTrace *trace, int close, 
                                  const VersionInfo &close_info,
                                  const VersionInfo &ver_info,
                                  const RestrictInfo &res_info,
                                  const FieldMask &close_m, Operation *create)
    //--------------------------------------------------------------------------
    {
      // Don't track these kinds of closes
      // We don't want to be stalling in the analysis pipeline
      // because we ran out of slots to issue
      initialize_close(ctx, req, false/*track*/);
      // Since we didn't register with our parent, we need to set
      // any trace that we might have explicitly
      if (trace != NULL)
        set_trace(trace);
      requirement.copy_without_mapping_info(req);
      requirement.initialize_mapping_fields();
      initialize_privilege_path(privilege_path, requirement);
      // Merge in the two different version informations
      version_info.merge(close_info, close_m);
      version_info.merge(ver_info, close_m);
      restrict_info.merge(res_info, close_m);
      target_children = targets;
      leave_open = open;
      close_idx = close;
      close_mask = close_m;
      create_op = create;
      create_gen = create_op->get_generation();
      parent_req_index = create->find_parent_index(close_idx);
      perform_logging(1/*is inter close op*/);
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::add_next_child(const ColorPoint &next_child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(next_child.is_valid());
#endif
      next_children.insert(next_child);
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_close();
      leave_open = false;
      close_idx = -1;
      create_op = NULL;
      create_gen = 0;
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_close();
      target_children.clear();
      close_mask.clear();
      next_children.clear();
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
    const RegionRequirement& InterCloseOp::get_region_requirement(void) const
    //--------------------------------------------------------------------------
    {
      return requirement;
    }

    //--------------------------------------------------------------------------
    const std::set<ColorPoint>& InterCloseOp::get_target_children(void) const
    //--------------------------------------------------------------------------
    {
      return target_children;
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::record_trace_dependence(Operation *target, 
                                               GenerationID target_gen,
                                               int target_idx,
                                               int source_idx, 
                                               DependenceType dtype,
                                               const FieldMask &dependent_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
    bool InterCloseOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(completion_event.exists());
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, BEGIN_MAPPING);
#endif
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      Processor local_proc = parent_ctx->get_executing_processor(); 
      // Never need to premap close operations
      Event close_event = Event::NO_EVENT;
      // If our requirement is restricted, then we already know what
      // our target should be.
      MappingRef target;
      if (restrict_info.has_restrictions())
      {
        target = runtime->forest->map_restricted_region(physical_ctx,
                                                        requirement,
                                                        0/*idx*/,
                                                        version_info,
                                                        local_proc
#ifdef DEBUG_HIGH_LEVEL
                                                        , get_logging_name()
                                                        , unique_op_id
#endif
                                                        );
      }
      // For partition operations that don't have a next child we
      // always want to make a composite instance because the low-level
      // runtime knows how to deal with lots of small instances
      bool force_composite = next_children.empty() && 
                             create_op->is_partition_op();
      bool success = runtime->forest->perform_close_operation(physical_ctx,
                                              requirement, parent_ctx,
                                              local_proc, target_children,
                                              leave_open, next_children, 
                                              close_event, target,
                                              version_info, force_composite
#ifdef DEBUG_HIGH_LEVEL
                                              , 0 /*idx*/ 
                                              , get_logging_name()
                                              , unique_op_id
#endif
                                              );
      // If we didn't succeed, then return
      if (!success)
      {
        version_info.reset();
        return false;
      }
      std::set<Event> applied_conditions;
      version_info.apply_close(physical_ctx.get_id(), leave_open,
                               runtime->address_space, applied_conditions);
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, END_MAPPING);
      LegionLogging::log_operation_events(
          Processor::get_executing_processor(),
          unique_op_id, Event::NO_EVENT, close_event);
      LegionLogging::log_physical_user(
          Processor::get_executing_processor(),
          reference.get_manager()->get_instance(), unique_op_id, 0/*idx*/);
#endif
#ifdef LEGION_SPY
      if (target.has_ref())
        LegionSpy::log_op_user(unique_op_id, 0/*idx*/, 
          target.get_view()->get_manager()->get_instance().id);
      {
        Processor proc = Processor::get_executing_processor();
        LegionSpy::log_op_proc_user(unique_op_id, proc.id);
      }
#endif
      if (!applied_conditions.empty())
        complete_mapping(Event::merge_events(applied_conditions));
      else
        complete_mapping();
#ifdef LEGION_LOGGING
      LegionLogging::log_event_dependence(Processor::get_executing_processor(),
                                          close_event,
                                          completion_event);
#endif
      complete_execution(close_event);
      // This should always succeed
      return true;
    }

    //--------------------------------------------------------------------------
    unsigned InterCloseOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
      assert(create_op != NULL);
#endif
      return create_op->find_parent_index(close_idx);
    }

    /////////////////////////////////////////////////////////////
    // Post Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PostCloseOp::PostCloseOp(Internal *runtime)
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
    void PostCloseOp::initialize(SingleTask *ctx, unsigned idx, 
                                 const InstanceRef &ref)
    //--------------------------------------------------------------------------
    {
      initialize_close(ctx, ctx->regions[idx], true/*track*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(ref.has_ref());
#endif
      reference = ref;
      // If it was write-discard from the task's perspective, make it
      // read-write within the task's context
      if (requirement.privilege == WRITE_DISCARD)
        requirement.privilege = READ_WRITE;
      parent_idx = idx;
      localize_region_requirement(requirement);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_close();
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_close();
      reference = InstanceRef();
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
#ifdef DEBUG_HIGH_LEVEL
      assert(completion_event.exists());
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, BEGIN_DEPENDENCE_ANALYSIS);
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
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, END_DEPENDENCE_ANALYSIS);
#endif
    }

    //--------------------------------------------------------------------------
    bool PostCloseOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(completion_event.exists());
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, BEGIN_MAPPING);
#endif
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_idx);
      Processor local_proc = parent_ctx->get_executing_processor();
      // We never need to premap close operations 

      // If we have a reference then we know we are closing a context
      // to a specific physical instance, so we can issue that without
      // worrying about failing.
      Event close_event = runtime->forest->close_physical_context(physical_ctx,
                                            requirement, version_info, this,
                                            local_proc, reference
#ifdef DEBUG_HIGH_LEVEL
                                            , 0 /*idx*/ 
                                            , get_logging_name()
                                            , unique_op_id
#endif
                                            );
      // No need to apply our mapping because we are done!
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, END_MAPPING);
      LegionLogging::log_operation_events(
          Processor::get_executing_processor(),
          unique_op_id, Event::NO_EVENT, close_event);
      LegionLogging::log_physical_user(
          Processor::get_executing_processor(),
          reference.get_manager()->get_instance(), unique_op_id, 0/*idx*/);
#endif
#ifdef LEGION_SPY
      // Log an implicit dependence on the parent's start event
      LegionSpy::log_implicit_dependence(parent_ctx->get_start_event(), 
                                         close_event);
      // Note this gives us a dependence to the parent's termination event
      // We log this only when close operations are used for closing contexts
      LegionSpy::log_op_events(unique_op_id, close_event, 
                             parent_ctx->get_task_completion());
      LegionSpy::log_op_user(unique_op_id, 0/*idx*/, 
          reference.get_manager()->get_instance().id);
      {
        Processor proc = Processor::get_executing_processor();
        LegionSpy::log_op_proc_user(unique_op_id, proc.id);
      }
#endif
      complete_mapping();
#ifdef LEGION_LOGGING
      LegionLogging::log_event_dependence(Processor::get_executing_processor(),
                                          close_event,
                                          completion_event);
#endif
      completion_event.trigger(close_event);
      need_completion_trigger = false;
      complete_execution(close_event);
      // This should always succeed
      return true;
    }

    //--------------------------------------------------------------------------
    unsigned PostCloseOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      return parent_idx;
    }

    /////////////////////////////////////////////////////////////
    // Virtual Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VirtualCloseOp::VirtualCloseOp(Internal *rt)
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
      initialize_close(ctx, ctx->regions[index], true/*track*/);
      // If it was write-discard from the task's perspective, make it
      // read-write within the task's context
      if (requirement.privilege == WRITE_DISCARD)
        requirement.privilege = READ_WRITE;
      parent_idx = index;
      localize_region_requirement(requirement);
      perform_logging();
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
      CompositeRef virtual_ref = 
        runtime->forest->map_virtual_region(physical_ctx, requirement, 
                                            0/*idx*/, version_info
#ifdef DEBUG_HIGH_LEVEL
                                            , get_logging_name()
                                            , unique_op_id
#endif
                                            );
      // Pass the reference back to the parent task
      parent_ctx->return_virtual_instance(parent_idx, virtual_ref);
      // Then we can mark that we are mapped and executed
      complete_mapping();
      complete_execution();
      return true;
    }

    //--------------------------------------------------------------------------
    unsigned VirtualCloseOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      return parent_idx;
    }

    /////////////////////////////////////////////////////////////
    // Acquire Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AcquireOp::AcquireOp(Internal *rt)
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
      requirement.initialize_mapping_fields();
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
                               parent_ctx->variants->name, 
                               parent_ctx->get_unique_task_id());
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
      for (std::vector<PhaseBarrier>::const_iterator it = 
            launcher.arrive_barriers.begin(); it != 
            launcher.arrive_barriers.end(); it++)
      {
        arrive_barriers.push_back(*it);
#ifdef LEGION_SPY
        LegionSpy::log_event_dependence(it->phase_barrier,
                                arrive_barriers.back().phase_barrier);
#endif
      }
      map_id = launcher.map_id;
      tag = launcher.tag;
      if (check_privileges)
        check_acquire_privilege();
      initialize_privilege_path(privilege_path, requirement);
#ifdef LEGION_SPY
      LegionSpy::log_acquire_operation(parent_ctx->get_unique_task_id(),
                                       unique_op_id);
      LegionSpy::log_logical_requirement(unique_op_id,0/*index*/,
                                         true/*region*/,
                                         requirement.region.index_space.id,
                                         requirement.region.field_space.id,
                                         requirement.region.tree_id,
                                         requirement.privilege,
                                         requirement.prop,
                                         requirement.redop);
      LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                        requirement.privilege_fields);
#endif
    }

    //--------------------------------------------------------------------------
    void AcquireOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative();
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
      restrict_info.clear();
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
    void AcquireOp::trigger_remote_state_analysis(UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      std::set<Event> preconditions;  
      version_info.make_local(preconditions, runtime->forest,
                              physical_ctx.get_id());
      if (preconditions.empty())
        ready_event.trigger();
      else
        ready_event.trigger(Event::merge_events(preconditions));
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
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    bool AcquireOp::speculate(bool &value)
    //--------------------------------------------------------------------------
    {
      Processor exec_proc = parent_ctx->get_executing_processor();
      return runtime->invoke_mapper_speculate(exec_proc, this, value);
    }

    //--------------------------------------------------------------------------
    bool AcquireOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      Processor local_proc = parent_ctx->get_executing_processor();
      // If we haven't already premapped the path, then do so now
      if (!requirement.premapped)
      {
        // Use our parent_ctx as the mappable since technically
        // we aren't a mappable.  Technically this shouldn't do anything
        // because we've marked ourselves as being restricted.
        requirement.premapped = runtime->forest->premap_physical_region(
                  physical_ctx, privilege_path, requirement, version_info,
                  this, parent_ctx, local_proc
#ifdef DEBUG_HIGH_LEVEL
                  , 0/*idx*/, get_logging_name(), unique_op_id
#endif
                  );
#ifdef DEBUG_HIGH_LEVEL
        assert(requirement.premapped);
#endif
      }
      
      // Map this is a restricted region. We already know the 
      // physical region that we want to map.
      MappingRef map_ref = runtime->forest->map_restricted_region(physical_ctx,
                                                                  requirement,
                                                                  0/*idx*/,
                                                                  version_info,
                                                                  local_proc
#ifdef DEBUG_HIGH_LEVEL
                                                          , get_logging_name()
                                                          , unique_op_id
#endif
                                                                  );
#ifdef DEBUG_HIGH_LEVEL
      assert(map_ref.has_ref());
#endif
      InstanceRef result = runtime->forest->register_physical_region(
                                                          physical_ctx,
                                                          map_ref,
                                                          requirement,
                                                          0/*idx*/,
                                                          version_info,
                                                          this,
                                                          local_proc,
                                                          completion_event
#ifdef DEBUG_HIGH_LEVEL
                                                          , get_logging_name()
                                                          , unique_op_id
#endif
                                                          );
#ifdef DEBUG_HIGH_LEVEL
      assert(result.has_ref());
#endif
      std::set<Event> applied_conditions;
      version_info.apply_mapping(physical_ctx.get_id(),
                                 runtime->address_space, applied_conditions);
      // Get all the events that need to happen before we can consider
      // ourselves acquired: reference ready and all synchronization
      std::set<Event> acquire_preconditions;
#ifdef LEGION_SPY
      std::set<Event> acquire_preconditions_spy;
      acquire_preconditions.insert(result.get_ready_event());
#endif
      acquire_preconditions.insert(result.get_ready_event());
      if (!wait_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::const_iterator it = 
              wait_barriers.begin(); it != wait_barriers.end(); it++)
        {
          Event e = it->phase_barrier.get_previous_phase();
          acquire_preconditions.insert(e);
#ifdef LEGION_SPY
          acquire_preconditions_spy.insert(
              it->phase_barrier.get_previous_phase());
#endif
        }
      }
      if (!grants.empty())
      {
        for (std::vector<Grant>::const_iterator it = grants.begin();
              it != grants.end(); it++)
        {
          Event e = it->impl->acquire_grant();
          acquire_preconditions.insert(e);
#ifdef LEGION_SPY
          acquire_preconditions_spy.insert(e);
#endif
        }
      }
      Event acquire_complete = Event::merge_events(acquire_preconditions);
#ifdef LEGION_SPY
      if (!acquire_complete.exists())
      {
        UserEvent new_acquire_complete = UserEvent::create_user_event();
        new_acquire_complete.trigger();
        acquire_complete = new_acquire_complete;
      }
      LegionSpy::log_event_dependences(acquire_preconditions_spy,
          acquire_complete);
      LegionSpy::log_implicit_dependence(parent_ctx->get_start_event(),
          acquire_complete);
      LegionSpy::log_op_events(unique_op_id, acquire_complete,
          completion_event);
      LegionSpy::log_implicit_dependence(acquire_complete,
          parent_ctx->get_task_completion());
      LegionSpy::log_op_user(unique_op_id, 0,
          result.get_manager()->get_instance().id);
      LegionSpy::log_event_dependence(acquire_complete, completion_event);
      {
        Processor proc = Processor::get_executing_processor();
        LegionSpy::log_op_proc_user(unique_op_id, proc.id);
      }
#endif
      // Chain any arrival barriers
      if (!arrive_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::const_iterator it = 
              arrive_barriers.begin(); it != arrive_barriers.end(); it++)
        {
          it->phase_barrier.arrive(1/*count*/, acquire_complete);
#ifdef LEGION_SPY
          LegionSpy::log_event_dependence(completion_event,
              it->phase_barrier);
#endif
        }
      }
      
      // Mark that we completed mapping
      if (!applied_conditions.empty())
        complete_mapping(Event::merge_events(applied_conditions));
      else
        complete_mapping();
      completion_event.trigger(acquire_complete);
      need_completion_trigger = false;
      complete_execution(acquire_complete);
      // we succeeded in mapping
      return true;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      Operation::trigger_commit();
    }

    //--------------------------------------------------------------------------
    unsigned AcquireOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    Mappable::MappableKind AcquireOp::get_mappable_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ACQUIRE_MAPPABLE;
    }

    //--------------------------------------------------------------------------
    Task* AcquireOp::as_mappable_task(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Copy* AcquireOp::as_mappable_copy(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Inline* AcquireOp::as_mappable_inline(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Acquire* AcquireOp::as_mappable_acquire(void) const
    //--------------------------------------------------------------------------
    {
      AcquireOp *proxy_this = const_cast<AcquireOp*>(this);
      return proxy_this;
    }

    //--------------------------------------------------------------------------
    Release* AcquireOp::as_mappable_release(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    UniqueID AcquireOp::get_unique_mappable_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
          }
        case ERROR_BAD_PARENT_REGION:
          {
            log_region.error("Parent task %s (ID %lld) of acquire "
                             "operation (ID %lld) does not have a region "
                             "requirement for region (%x,%x,%x) as a parent",
                             parent_ctx->variants->name, 
                             parent_ctx->get_unique_task_id(),
                             unique_op_id, 
                             requirement.region.index_space.id,
                             requirement.region.field_space.id, 
                             requirement.region.tree_id);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
                               parent_ctx->variants->name, 
                               parent_ctx->get_unique_task_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_BAD_PARENT_REGION);
      }
      else
        parent_req_index = unsigned(parent_index);
    }

    /////////////////////////////////////////////////////////////
    // Release Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReleaseOp::ReleaseOp(Internal *rt)
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
      requirement.initialize_mapping_fields();
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
                               parent_ctx->variants->name, 
                               parent_ctx->get_unique_task_id());
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
      for (std::vector<PhaseBarrier>::const_iterator it = 
            launcher.arrive_barriers.begin(); it != 
            launcher.arrive_barriers.end(); it++)
      {
        arrive_barriers.push_back(*it);
#ifdef LEGION_SPY
        LegionSpy::log_event_dependence(it->phase_barrier,
                                arrive_barriers.back().phase_barrier);
#endif
      }
      map_id = launcher.map_id;
      tag = launcher.tag;
      if (check_privileges)
        check_release_privilege();
      initialize_privilege_path(privilege_path, requirement);
#ifdef LEGION_SPY
      LegionSpy::log_release_operation(parent_ctx->get_unique_task_id(),
                                       unique_op_id);
      LegionSpy::log_logical_requirement(unique_op_id,0/*index*/,
                                         true/*region*/,
                                         requirement.region.index_space.id,
                                         requirement.region.field_space.id,
                                         requirement.region.tree_id,
                                         requirement.privilege,
                                         requirement.prop,
                                         requirement.redop);
      LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                        requirement.privilege_fields);
#endif
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative(); 
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
      restrict_info.clear();
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
    void ReleaseOp::trigger_remote_state_analysis(UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      std::set<Event> preconditions;  
      version_info.make_local(preconditions, runtime->forest,
                              physical_ctx.get_id());
      if (preconditions.empty())
        ready_event.trigger();
      else
        ready_event.trigger(Event::merge_events(preconditions));
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
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    bool ReleaseOp::speculate(bool &value)
    //--------------------------------------------------------------------------
    {
      Processor exec_proc = parent_ctx->get_executing_processor();
      return runtime->invoke_mapper_speculate(exec_proc, this, value);
    }

    //--------------------------------------------------------------------------
    bool ReleaseOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      Processor local_proc = parent_ctx->get_executing_processor();
      // If we haven't already premapped the path, then do so now
      if (!requirement.premapped)
      {
        // Use our parent_ctx as the mappable since technically
        // we aren't a mappable.  Technically this shouldn't do anything
        // because we've marked ourselves as being restricted.
        requirement.premapped = runtime->forest->premap_physical_region(
                  physical_ctx, privilege_path, requirement, version_info, 
                  this, parent_ctx, local_proc
#ifdef DEBUG_HIGH_LEVEL
                  , 0/*idx*/, get_logging_name(), unique_op_id
#endif
                  );
#ifdef DEBUG_HIGH_LEVEL
        assert(requirement.premapped);
#endif
      }
      
#ifdef LEGION_SPY
      LegionSpy::IDType inst_id;
      {
        const InstanceRef& ref = region.impl->get_reference();
        inst_id = ref.get_manager()->get_instance().id;
      }
#endif
      // Map this is a restricted region and then register it. The process
      // of registering it will close up any open children to this instance.
      MappingRef map_ref = runtime->forest->remap_physical_region(physical_ctx,
                                                                  requirement,
                                                                  0/*idx*/,
                                                                  version_info,
                                                  region.impl->get_reference()
#ifdef DEBUG_HIGH_LEVEL
                                                            , get_logging_name()
                                                            , unique_op_id
#endif
                                                                  );
#ifdef DEBUG_HIGH_LEVEL
      assert(map_ref.has_ref());
#endif
      InstanceRef result = runtime->forest->register_physical_region(
                                                            physical_ctx,
                                                            map_ref,
                                                            requirement,
                                                            0/*idx*/,
                                                            version_info,
                                                            this,
                                                            local_proc,
                                                            completion_event
#ifdef DEBUG_HIGH_LEVEL
                                                            , get_logging_name()
                                                            , unique_op_id
#endif
                                                            );
#ifdef DEBUG_HIGH_LEVEL
      assert(result.has_ref());
#endif
      std::set<Event> applied_conditions;
      version_info.apply_mapping(physical_ctx.get_id(),
                                 runtime->address_space, applied_conditions);

      Event release_event = result.get_ready_event();
      std::set<Event> release_preconditions;
      release_preconditions.insert(release_event);
#ifdef LEGION_SPY
      std::set<Event> release_preconditions_spy;
      release_preconditions_spy.insert(release_event);
#endif
      if (!wait_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::const_iterator it = 
              wait_barriers.begin(); it != wait_barriers.end(); it++)
        {
          Event e = it->phase_barrier.get_previous_phase();
          release_preconditions.insert(e);
#ifdef LEGION_SPY
          release_preconditions_spy.insert(
              it->phase_barrier.get_previous_phase());
#endif
        }
      }
      if (!grants.empty())
      {
        for (std::vector<Grant>::const_iterator it = grants.begin();
              it != grants.end(); it++)
        {
          Event e = it->impl->acquire_grant();
          release_preconditions.insert(e);
#ifdef LEGION_SPY
          release_preconditions_spy.insert(e);
#endif
        }
      }
      Event release_complete = Event::merge_events(release_preconditions);
#ifdef LEGION_SPY
      if (!release_complete.exists())
      {
        UserEvent new_release_complete = UserEvent::create_user_event();
        new_release_complete.trigger();
        release_complete = new_release_complete;
      }
      LegionSpy::log_event_dependences(release_preconditions_spy,
          release_complete);
      LegionSpy::log_implicit_dependence(parent_ctx->get_start_event(),
          release_complete);
      LegionSpy::log_op_events(unique_op_id, release_complete,
          completion_event);
      LegionSpy::log_op_user(unique_op_id, 0, inst_id);
      LegionSpy::log_implicit_dependence(release_complete,
          parent_ctx->get_task_completion());
      LegionSpy::log_event_dependence(release_complete, completion_event);
      {
        Processor proc = Processor::get_executing_processor();
        LegionSpy::log_op_proc_user(unique_op_id, proc.id);
      }
#endif
      // Chain any arrival barriers
      if (!arrive_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::const_iterator it = 
              arrive_barriers.begin(); it != arrive_barriers.end(); it++)
        {
#ifdef LEGION_SPY
          LegionSpy::log_event_dependence(completion_event,
              it->phase_barrier);
#endif
          it->phase_barrier.arrive(1/*count*/, release_complete);
        }
      }
      
      // Mark that we completed mapping
      if (!applied_conditions.empty())
        complete_mapping(Event::merge_events(applied_conditions));
      else
        complete_mapping();
      completion_event.trigger(release_complete);
      need_completion_trigger = false;
      complete_execution(release_complete);
      // We succeeded in mapping
      return true;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      Operation::trigger_commit();
    }

    //--------------------------------------------------------------------------
    unsigned ReleaseOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    Mappable::MappableKind ReleaseOp::get_mappable_kind(void) const
    //--------------------------------------------------------------------------
    {
      return RELEASE_MAPPABLE;
    }

    //--------------------------------------------------------------------------
    Task* ReleaseOp::as_mappable_task(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Copy* ReleaseOp::as_mappable_copy(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Inline* ReleaseOp::as_mappable_inline(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Acquire* ReleaseOp::as_mappable_acquire(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Release* ReleaseOp::as_mappable_release(void) const
    //--------------------------------------------------------------------------
    {
      ReleaseOp *proxy_this = const_cast<ReleaseOp*>(this);
      return proxy_this;
    }

    //--------------------------------------------------------------------------
    UniqueID ReleaseOp::get_unique_mappable_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
          }
        case ERROR_BAD_PARENT_REGION:
          {
            log_region.error("Parent task %s (ID %lld) of release "
                             "operation (ID %lld) does not have a region "
                             "requirement for region (%x,%x,%x) as a parent",
                             parent_ctx->variants->name, 
                             parent_ctx->get_unique_task_id(),
                             unique_op_id, 
                             requirement.region.index_space.id,
                             requirement.region.field_space.id, 
                             requirement.region.tree_id);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
                               parent_ctx->variants->name, 
                               parent_ctx->get_unique_task_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_BAD_PARENT_REGION);
      }
      else
        parent_req_index = unsigned(parent_index);
    }

    /////////////////////////////////////////////////////////////
    // Dynamic Collective Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DynamicCollectiveOp::DynamicCollectiveOp(Internal *rt)
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
      future = Future(legion_new<Future::Impl>(runtime, true/*register*/,
            runtime->get_available_distributed_id(true), runtime->address_space,
            runtime->address_space, this));
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
      Barrier barrier = collective.phase_barrier.get_previous_phase();
      if (!barrier.has_triggered())
      {
        DeferredExecuteArgs deferred_execute_args;
        deferred_execute_args.hlr_id = HLR_DEFERRED_EXECUTION_TRIGGER_ID;
        deferred_execute_args.proxy_this = this;
        runtime->issue_runtime_meta_task(&deferred_execute_args,
                                         sizeof(deferred_execute_args),
                                         HLR_DEFERRED_EXECUTION_TRIGGER_ID,
                                         this, barrier);
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
      const ReductionOp *redop = Internal::get_reduction_op(collective.redop);
      const size_t result_size = redop->sizeof_lhs;
      void *result_buffer = legion_malloc(FUTURE_RESULT_ALLOC, result_size);
#ifdef DEBUG_HIGH_LEVEL
      bool result = 
#endif
      collective.phase_barrier.get_previous_phase().get_result(result_buffer,
							       result_size);
#ifdef DEBUG_HIGH_LEVEL
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
    FuturePredOp::FuturePredOp(Internal *rt)
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
      assert(valid);
#endif
      set_resolved_value(get_generation(), value);
    }

    //--------------------------------------------------------------------------
    void FuturePredOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
                                         this, future.impl->get_ready_event());
      }
      // Mark that we completed mapping this operation
      complete_mapping();
    } 

    /////////////////////////////////////////////////////////////
    // Not Predicate Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    NotPredOp::NotPredOp(Internal *rt)
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
      assert(prev_gen == get_generation());
#endif
      // Don't forget to negate the value
      set_resolved_value(prev_gen, !value);
    }

    /////////////////////////////////////////////////////////////
    // And Predicate Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AndPredOp::AndPredOp(Internal *rt)
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
                               const Predicate &p1, const Predicate &p2)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx != NULL);
#endif
      // Don't track this as it can lead to deadlock because
      // predicates can't complete until all their references from
      // the parent task have been removed.
      initialize_operation(ctx, false/*track*/);
      // Short circuit case
      if ((p1 == Predicate::FALSE_PRED) || (p2 == Predicate::FALSE_PRED))
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
    OrPredOp::OrPredOp(Internal *rt)
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
                              const Predicate &p1, const Predicate &p2)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx != NULL);
#endif
      // Don't track this as it can lead to deadlock because
      // predicates can't complete until all their references from
      // the parent task have been removed.
      initialize_operation(ctx, false/*track*/);
      // Short circuit case
      if ((p1 == Predicate::TRUE_PRED) || (p2 == Predicate::TRUE_PRED))
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
#ifdef DEBUG_HIGH_LEVEL
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
    MustEpochOp::MustEpochOp(Internal *rt)
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
        indiv_tasks[idx]->set_must_epoch(this, idx);
        // If we have a trace, set it for this operation as well
        if (trace != NULL)
          indiv_tasks[idx]->set_trace(trace);
        indiv_tasks[idx]->must_parallelism = true;
      }
      indiv_triggered.resize(indiv_tasks.size(), false);
      index_tasks.resize(launcher.index_tasks.size());
      for (unsigned idx = 0; idx < launcher.index_tasks.size(); idx++)
      {
        index_tasks[idx] = runtime->get_available_index_task(true);
        index_tasks[idx]->initialize_task(ctx, launcher.index_tasks[idx],
                                          check_privileges, false/*track*/);
        index_tasks[idx]->set_must_epoch(this, indiv_tasks.size()+idx);
        if (trace != NULL)
          index_tasks[idx]->set_trace(trace);
        index_tasks[idx]->must_parallelism = true;
      }
      index_triggered.resize(index_tasks.size(), false);
      mapper_id = launcher.map_id;
      mapper_tag = launcher.mapping_tag;
      // Make a new future map for storing our results
      // We'll fill it in later
      result_map = legion_new<FutureMap::Impl>(ctx, 
                                               get_completion_event(), runtime);
#ifdef DEBUG_HIGH_LEVEL
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
      {
        result_map.impl->add_valid_point(indiv_tasks[idx]->index_point);
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        result_map.impl->add_valid_domain(index_tasks[idx]->index_domain);
      }
#endif
      return result_map;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::set_task_options(ProcessorManager *manager)
    //--------------------------------------------------------------------------
    {
      // Mark that all these operations will map locally because we
      // have to do a single mapping call for all our operations.
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
      {
        manager->invoke_mapper_set_task_options(indiv_tasks[idx]);
        indiv_tasks[idx]->map_locally = true;
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        manager->invoke_mapper_set_task_options(index_tasks[idx]);
        index_tasks[idx]->map_locally = true;
      }
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
      dependences.clear();
      mapping_dependences.clear();
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
      unsigned prev_count = 0;
      dependence_count.resize(indiv_tasks.size() + index_tasks.size());
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
      {
        indiv_tasks[idx]->trigger_dependence_analysis();
        unsigned next_count = dependences.size();
        dependence_count[idx] = next_count - prev_count;
        prev_count = next_count;
      }
      unsigned offset = indiv_tasks.size();
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        index_tasks[idx]->trigger_dependence_analysis();
        unsigned next_count = dependences.size();
        dependence_count[offset+idx] = next_count - prev_count;
        prev_count = next_count;
      }
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::trigger_remote_state_analysis(UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<Event> preconditions;
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
      {
        UserEvent indiv_event = UserEvent::create_user_event();
        indiv_tasks[idx]->trigger_remote_state_analysis(indiv_event);
        preconditions.insert(indiv_event);
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        UserEvent index_event = UserEvent::create_user_event();
        index_tasks[idx]->trigger_remote_state_analysis(index_event);
        preconditions.insert(index_event);
      }
      ready_event.trigger(Event::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    bool MustEpochOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
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
                                     index_tasks, index_triggered,
                                     dependences, dependence_count))
          return false;

#ifdef DEBUG_HIGH_LEVEL
        assert(!single_tasks.empty());
#endif 
        // Next build the set of single tasks and all their constraints.
        // Iterate over all the recorded dependences
        constraints.reserve(dependences.size());
        for (std::deque<DependenceRecord>::const_iterator it = 
              dependences.begin(); it != dependences.end(); it++)
        {
          // Add constraints for all the different elements
          const std::set<SingleTask*> &s1 = task_sets[it->op1_idx];
          const std::set<SingleTask*> &s2 = task_sets[it->op2_idx];
          for (std::set<SingleTask*>::const_iterator it1 = s1.begin();
                it1 != s1.end(); it1++)
          {
            for (std::set<SingleTask*>::const_iterator it2 = s2.begin();
                  it2 != s2.end(); it2++)
            {
              constraints.push_back(Mapper::MappingConstraint(
                                                       *it1, it->reg1_idx,
                                                       *it2, it->reg2_idx,
                                                       it->dtype));
              mapping_dependences[*it1].push_back(*it2);
              mapping_dependences[*it2].push_back(*it1);
              // Tell the tasks they will need to refetch physical
              // state when mapping these regions so we can make
              // sure that they see other mapped regions from other
              // tasks in this must epoch launch
              (*it1)->recapture_version_info(it->reg1_idx);
              (*it2)->recapture_version_info(it->reg2_idx);
            }
          }
        }
        // Clear this eagerly to save space
        dependences.clear();
        // Mark that we have finished building all the constraints so
        // we don't have to redo it if we end up failing a mapping.
        triggering_complete = true;
      }
      // Next make the mapper call to perform the mapping for all the tasks
      std::vector<Task*> copy_single_tasks(single_tasks.begin(),
                                           single_tasks.end());
      Processor mapper_proc = parent_ctx->get_executing_processor();
      bool notify = runtime->invoke_mapper_map_must_epoch(
          mapper_proc, copy_single_tasks, constraints, mapper_id, mapper_tag);

      // Check that all the tasks have been assigned to different processors
      std::map<Processor,SingleTask*> target_procs;
      for (std::deque<SingleTask*>::const_iterator it = 
            single_tasks.begin(); it != single_tasks.end(); it++)
      {
        if (target_procs.find((*it)->target_proc) != target_procs.end())
        {
          SingleTask *other = target_procs[(*it)->target_proc];
          log_run.error("MUST EPOCH ERROR: Task %s (ID %lld) and "
              "task %s (ID %lld) both requested to be run on processor "
              IDFMT "!",
              (*it)->variants->name, (*it)->get_unique_task_id(),
              other->variants->name, other->get_unique_task_id(),
              other->target_proc.id);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_MUST_EPOCH_FAILURE);
        }
        target_procs[(*it)->target_proc] = *it;
      }

      // Then we need to actually perform the mapping
      {
        MustEpochMapper mapper(this); 
        if (!mapper.map_tasks(single_tasks, mapping_dependences))
          return false;
      }

      // Everybody successfully mapped so now check that all
      // of the constraints have been satisfied
      for (std::vector<Mapper::MappingConstraint>::const_iterator it = 
            constraints.begin(); it != constraints.end(); it++)
      {
        // We know that all these tasks are single tasks
        // so doing static casts are safe
        SingleTask *t1 = static_cast<SingleTask*>(const_cast<Task*>(it->t1));
        SingleTask *t2 = static_cast<SingleTask*>(const_cast<Task*>(it->t2));
        PhysicalManager *inst1 = t1->get_instance(it->idx1);
        PhysicalManager *inst2 = t2->get_instance(it->idx2);
        // Check to make sure they selected the same instance 
        if (inst1 != inst2)
        {
          log_run.error("MUST EPOCH ERROR: failed constraint! "
              "Task %s (ID %lld) mapped region %d to instance " IDFMT " in "
              "memory " IDFMT " , but task %s (ID %lld) mapped region %d to "
              "instance " IDFMT " in memory " IDFMT ".",
              t1->variants->name, t1->get_unique_task_id(), it->idx1,
              inst1->get_instance().id, inst1->memory.id,
              t2->variants->name, t2->get_unique_task_id(), it->idx2,
              inst2->get_instance().id, inst2->memory.id);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_MUST_EPOCH_FAILURE);
        }
      }

      // If the mapper wanted to know on success, then tell it
      if (notify)
      {
        // Notify the mappers that the tasks successfully mapped
        for (std::deque<SingleTask*>::const_iterator it = 
              single_tasks.begin(); it != single_tasks.end(); it++)
        {
          runtime->invoke_mapper_notify_result(mapper_proc, *it);
        }
      }

      // Once all the tasks have been initialized we can defer
      // our all mapped event on all their all mapped events
      std::set<Event> tasks_all_mapped;
      std::set<Event> tasks_all_complete;
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
      Event all_mapped = Event::merge_events(tasks_all_mapped);
      Event all_complete = Event::merge_events(tasks_all_complete);
      complete_mapping(all_mapped);
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
        assert(remaining_subop_commits > 0);
#endif
        remaining_subop_commits--;
        need_commit = (remaining_subop_commits == 0);
      }
      if (need_commit)
        commit_operation();
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
#ifdef DEBUG_HIGH_LEVEL
        assert(dst_index >= 0);
#endif
        TaskOp *src_task = find_task_by_index(src_index);
        TaskOp *dst_task = find_task_by_index(dst_index);
        log_run.error("MUST EPOCH ERROR: dependence between task "
            "%s (ID %lld) and task %s (ID %lld)\n",
            src_task->variants->name, src_task->get_unique_task_id(),
            dst_task->variants->name, dst_task->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
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
              " type %s", src_idx, src_task->variants->name,
              src_task->get_unique_task_id(), dst_idx, 
              dst_task->variants->name, dst_task->get_unique_task_id(),
              (dtype == TRUE_DEPENDENCE) ? "TRUE DEPENDENCE" :
                (dtype == ANTI_DEPENDENCE) ? "ANTI DEPENDENCE" :
                "ATOMIC DEPENDENCE");
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_MUST_EPOCH_FAILURE);
        }
        else if (dtype == SIMULTANEOUS_DEPENDENCE)
        {
          // Record the dependence kind
          int dst_index = find_operation_index(dst_op, dst_gen);
#ifdef DEBUG_HIGH_LEVEL
          assert(dst_index >= 0);
#endif
          dependences.push_back(DependenceRecord(src_index, dst_index,
                                                 src_idx, dst_idx, dtype)); 
          return false;
        }
        // NO_DEPENDENCE and PROMOTED_DEPENDENCE are not errors
        // and do not need to be recorded
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::add_mapping_dependence(Event precondition)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(dependence_tracker.mapping != NULL);
#endif
      dependence_tracker.mapping->add_mapping_dependence(precondition);
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::register_single_task(SingleTask *single, unsigned index)
    //--------------------------------------------------------------------------
    {
      // Can do the first part without the lock 
#ifdef DEBUG_HIGH_LEVEL
      assert(index < task_sets.size());
#endif
      task_sets[index].insert(single);
      AutoLock o_lock(op_lock);
      single_tasks.push_back(single);
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
        assert(remaining_subop_commits > 0);
#endif
        remaining_subop_commits--;
        need_commit = (remaining_subop_commits == 0);
      }
      if (need_commit)
        commit_operation();
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
                                std::vector<bool> &index_triggered,
                          const std::deque<MustEpochOp::DependenceRecord> &deps,
                                const std::vector<unsigned> &dep_counts)
    //--------------------------------------------------------------------------
    {
      std::set<Event> wait_events;
      unsigned dep_offset = 0; 
      std::vector<Event> triggered_events(indiv_tasks.size() + 
                            index_tasks.size(), Event::NO_EVENT);
      for (unsigned idx = 0; idx < indiv_triggered.size(); idx++)
      {
        if (!indiv_triggered[idx])
        {
          std::set<Event> preconditions;
          // Figure out the event preconditiions
          for (unsigned dep_idx = 0; dep_idx < dep_counts[idx]; dep_idx++)
          {
            const MustEpochOp::DependenceRecord &record = 
              deps[dep_offset + dep_idx]; 
#ifdef DEBUG_HIGH_LEVEL
            assert(idx == record.op1_idx);
            assert(record.op1_idx < triggered_events.size());
#endif
            Event pre = triggered_events[record.op2_idx]; 
            if (pre.exists())
              preconditions.insert(pre);
          }
          Event precondition;
          if (!preconditions.empty())
            precondition = Event::merge_events(preconditions);
          else
            precondition = Event::NO_EVENT;
          MustEpochIndivArgs args;
          args.hlr_id = HLR_MUST_INDIV_ID;
          args.triggerer = this;
          args.task = indiv_tasks[idx];
          Event wait = owner->runtime->issue_runtime_meta_task(&args, 
                  sizeof(args), HLR_MUST_INDIV_ID, owner, precondition);
          if (wait.exists())
          {
            wait_events.insert(wait);
            triggered_events[idx] = wait;
          }
        }
        dep_offset += dep_counts[idx];
      }
      const unsigned op_offset = indiv_tasks.size();
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        if (!index_triggered[idx])
        {
          std::set<Event> preconditions;
          // Figure out the event preconditiions
          for (unsigned dep_idx = 0; 
                dep_idx < dep_counts[op_offset + idx]; dep_idx++)
          {
            const MustEpochOp::DependenceRecord &record = 
              deps[dep_offset + dep_idx]; 
#ifdef DEBUG_HIGH_LEVEL
            assert(idx == record.op1_idx);
            assert(record.op1_idx < triggered_events.size());
#endif
            Event pre = triggered_events[record.op2_idx]; 
            if (pre.exists())
              preconditions.insert(pre);
          }
          Event precondition;
          if (!preconditions.empty())
            precondition = Event::merge_events(preconditions);
          else
            precondition = Event::NO_EVENT;
          MustEpochIndexArgs args;
          args.hlr_id = HLR_MUST_INDEX_ID;
          args.triggerer = this;
          args.task = index_tasks[idx];
          Event wait = owner->runtime->issue_runtime_meta_task(&args,
                sizeof(args), HLR_MUST_INDEX_ID, owner, precondition);
          if (wait.exists())
          {
            wait_events.insert(wait);
            triggered_events[op_offset + idx] = wait;
          }
        }
        dep_offset += dep_counts[op_offset + idx];
      }

      // Wait for all of the launches to be done
      // We can safely block to free up the utility processor
      if (!wait_events.empty())
      {
        Event trigger_event = Event::merge_events(wait_events);
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
    bool MustEpochMapper::map_tasks(const std::deque<SingleTask*> &single_tasks,
      const std::map<SingleTask*,std::deque<SingleTask*> > &mapping_dependences)
    //--------------------------------------------------------------------------
    {
      std::set<Event> wait_events;   
      MustEpochMapArgs args;
      args.hlr_id = HLR_MUST_MAP_ID;
      args.mapper = this;
      std::map<SingleTask*,Event> mapping_events;
      for (std::deque<SingleTask*>::const_iterator it = single_tasks.begin();
            it != single_tasks.end(); it++)
      {
        args.task = *it;
        // Compute the preconditions
        std::set<Event> preconditions; 
        std::map<SingleTask*,std::deque<SingleTask*> >::const_iterator 
          dep_finder = mapping_dependences.find(*it);
        if (dep_finder != mapping_dependences.end())
        {
          const std::deque<SingleTask*> &deps = dep_finder->second;
          for (std::deque<SingleTask*>::const_iterator dit = 
                deps.begin(); dit != deps.end(); dit++)
          {
            std::map<SingleTask*,Event>::const_iterator finder = 
              mapping_events.find(*dit);
            if (finder != mapping_events.end())
              preconditions.insert(finder->second);
          }
        }
        Event precondition = Event::NO_EVENT;
        if (!preconditions.empty())
          precondition = Event::merge_events(preconditions);
        Event wait = owner->runtime->issue_runtime_meta_task(&args, 
                            sizeof(args), HLR_MUST_MAP_ID, owner, precondition);
        if (wait.exists())
        {
          mapping_events[*it] = wait;
          wait_events.insert(wait);
        }
      }
      
      if (!wait_events.empty())
      {
        Event mapped_event = Event::merge_events(wait_events);
        mapped_event.wait();
      }

      // If we failed to map then unmap all the tasks 
      if (!success)
      {
        for (std::deque<SingleTask*>::const_iterator it = single_tasks.begin();
              it != single_tasks.end(); it++)
        {
          (*it)->unmap_all_regions();
        }
      }
      return success;
    }

    //--------------------------------------------------------------------------
    void MustEpochMapper::map_task(SingleTask *task)
    //--------------------------------------------------------------------------
    {
      // Note we don't need to hold a lock here because this is
      // a monotonic change.  Once it fails for anyone then it
      // fails for everyone.
      if (!task->perform_mapping(true/*mapper invoked already*/))
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
    void MustEpochDistributor::distribute_tasks(Internal *runtime,
                                const std::vector<IndividualTask*> &indiv_tasks,
                                const std::set<SliceTask*> &slice_tasks)
    //--------------------------------------------------------------------------
    {
      MustEpochDistributorArgs dist_args;
      dist_args.hlr_id = HLR_MUST_DIST_ID;
      MustEpochLauncherArgs launch_args;
      launch_args.hlr_id = HLR_MUST_LAUNCH_ID;
      std::set<Event> wait_events;
      for (std::vector<IndividualTask*>::const_iterator it = 
            indiv_tasks.begin(); it != indiv_tasks.end(); it++)
      {
        if (!runtime->is_local((*it)->target_proc))
        {
          dist_args.task = *it;
          Event wait = runtime->issue_runtime_meta_task(&dist_args,
                          sizeof(dist_args), HLR_MUST_DIST_ID, owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
        else
        {
          launch_args.task = *it;
          Event wait = runtime->issue_runtime_meta_task(&launch_args,
                          sizeof(launch_args), HLR_MUST_LAUNCH_ID, owner);
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
          Event wait = runtime->issue_runtime_meta_task(&dist_args,
                          sizeof(dist_args), HLR_MUST_DIST_ID, owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
        else
        {
          launch_args.task = *it;
          Event wait = runtime->issue_runtime_meta_task(&launch_args,
                          sizeof(launch_args), HLR_MUST_LAUNCH_ID, owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
      }
      if (!wait_events.empty())
      {
        Event dist_event = Event::merge_events(wait_events);
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
    PendingPartitionOp::PendingPartitionOp(Internal *rt)
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
#ifdef DEBUG_HIGH_LEVEL
      assert(thunk == NULL);
#endif
      thunk = new EqualPartitionThunk(pid, granularity);
#ifdef LEGION_SPY
      perform_logging();
#endif
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_weighted_partition(SingleTask *ctx, 
                                                           IndexPartition pid, 
                                                           size_t granularity,
                                       const std::map<DomainPoint,int> &weights)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(thunk == NULL);
#endif
      thunk = new WeightedPartitionThunk(pid, granularity, weights);
#ifdef LEGION_SPY
      perform_logging();
#endif
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_union_partition(SingleTask *ctx,
                                                        IndexPartition pid,
                                                        IndexPartition h1,
                                                        IndexPartition h2)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(thunk == NULL);
#endif
      thunk = new UnionPartitionThunk(pid, h1, h2);
#ifdef LEGION_SPY
      perform_logging();
#endif
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_intersection_partition(SingleTask *ctx,
                                                            IndexPartition pid,
                                                            IndexPartition h1,
                                                            IndexPartition h2)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(thunk == NULL);
#endif
      thunk = new IntersectionPartitionThunk(pid, h1, h2);
#ifdef LEGION_SPY
      perform_logging();
#endif
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_difference_partition(SingleTask *ctx,
                                                             IndexPartition pid,
                                                             IndexPartition h1,
                                                             IndexPartition h2)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(thunk == NULL);
#endif
      thunk = new DifferencePartitionThunk(pid, h1, h2);
#ifdef LEGION_SPY
      perform_logging();
#endif
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_cross_product(SingleTask *ctx,
                                                      IndexPartition base,
                                                      IndexPartition source,
                                  std::map<DomainPoint,IndexPartition> &handles)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(thunk == NULL);
#endif
      thunk = new CrossProductThunk(base, source, handles);
#ifdef LEGION_SPY
      perform_logging();
#endif
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_union(SingleTask *ctx,
                                                          IndexSpace target,
                                         const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingSpace(target, true/*union*/, handles);
#ifdef LEGION_SPY
      perform_logging();
#endif
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_union(SingleTask *ctx,
                                                          IndexSpace target,
                                                          IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingSpace(target, true/*union*/, handle);
#ifdef LEGION_SPY
      perform_logging();
#endif
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_intersection(
     SingleTask *ctx, IndexSpace target, const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingSpace(target, false/*union*/, handles);
#ifdef LEGION_SPY
      perform_logging();
#endif
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_intersection(
                      SingleTask *ctx, IndexSpace target, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingSpace(target, false/*union*/, handle);
#ifdef LEGION_SPY
      perform_logging();
#endif
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_difference(SingleTask *ctx,
                                         IndexSpace target, IndexSpace initial, 
                                         const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingDifference(target, initial, handles);
#ifdef LEGION_SPY
      perform_logging();
#endif
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::perform_logging()
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      LegionSpy::log_pending_partition_operation(
          parent_ctx->get_unique_task_id(),
          unique_op_id);
      thunk->perform_logging(this);
#endif
    }

    //--------------------------------------------------------------------------
    bool PendingPartitionOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      // Perform the partitioning operation
      Event ready_event = thunk->perform(runtime->forest);
      // We can trigger the handle ready event now
      handle_ready.trigger();
#ifdef LEGION_SPY
      LegionSpy::log_implicit_dependence(parent_ctx->get_start_event(),
          ready_event);
      LegionSpy::log_op_events(unique_op_id, ready_event,
          completion_event);
      LegionSpy::log_implicit_dependence(completion_event,
          parent_ctx->get_task_completion());
      LegionSpy::log_event_dependence(handle_ready, ready_event);
      LegionSpy::log_event_dependence(ready_event, completion_event);
      {
        Processor local_proc = Processor::get_executing_processor();
        LegionSpy::log_op_proc_user(unique_op_id, local_proc.id);
      }
#endif
      complete_mapping();
      completion_event.trigger(ready_event);
      need_completion_trigger = false;
      complete_execution(ready_event);
      // Return true since we succeeded
      return true;
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      handle_ready = UserEvent::create_user_event();
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
    DependentPartitionOp::DependentPartitionOp(Internal *rt)
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
      requirement.initialize_mapping_fields();
      partition_handle = pid;
      color_space = space;
#ifdef LEGION_SPY
      perform_logging();
#endif
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
      requirement.initialize_mapping_fields();
      partition_handle = pid;
      color_space = space;
#ifdef LEGION_SPY
      perform_logging();
#endif
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
      requirement.initialize_mapping_fields();
      partition_handle = pid;
      color_space = space;
      projection = proj;
#ifdef LEGION_SPY
      perform_logging();
#endif
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::perform_logging()
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      LegionSpy::log_dependent_partition_operation(
          parent_ctx->get_unique_task_id(),
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
                                  requirement.redop);
      else
        LegionSpy::log_logical_requirement(unique_op_id, 0/*idx*/,
                                  true/*region*/,
                                  requirement.region.index_space.id,
                                  requirement.region.field_space.id,
                                  requirement.region.tree_id,
                                  requirement.privilege,
                                  requirement.prop,
                                  requirement.redop);
      LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                        requirement.privilege_fields);
#endif
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
                                                          UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      std::set<Event> preconditions;  
      version_info.make_local(preconditions, runtime->forest,
                              physical_ctx.get_id());
      if (preconditions.empty())
        ready_event.trigger();
      else
        ready_event.trigger(Event::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    bool DependentPartitionOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      Processor local_proc = parent_ctx->get_executing_processor();
      // If we haven't already premapped the path, then do so now
      if (!requirement.premapped)
      {
        requirement.premapped = runtime->forest->premap_physical_region(
                  physical_ctx, privilege_path, requirement, version_info,
                  this, parent_ctx, local_proc
#ifdef DEBUG_HIGH_LEVEL
                  , 0/*idx*/, get_logging_name(), unique_op_id
#endif
                  );
#ifdef DEBUG_HIGH_LEVEL
        assert(requirement.premapped);
#endif
      }

      Event ready_event = Event::NO_EVENT;
      switch (partition_kind)
      {
        case BY_FIELD:
          {
            ready_event = 
              runtime->forest->create_partition_by_field(physical_ctx,
                                                         local_proc,
                                                         requirement,
                                                         partition_handle,
                                                         color_space,
                                                         completion_event,
                                                         version_info);
            break;
          }
        case BY_IMAGE:
          {
            ready_event = 
              runtime->forest->create_partition_by_image(physical_ctx,
                                                         local_proc,
                                                         requirement,
                                                         partition_handle,
                                                         color_space,
                                                         completion_event,
                                                         version_info);
            break;
          }
        case BY_PREIMAGE:
          {
            ready_event = 
              runtime->forest->create_partition_by_preimage(physical_ctx,
                                                            local_proc,
                                                            requirement,
                                                            projection,
                                                            partition_handle,
                                                            color_space,
                                                            completion_event,
                                                            version_info);
            break;
          }
        default:
          assert(false); // should never get here
      }
      // Once we are done running these routines, we can mark
      // that the handles have all been completed
#ifdef DEBUG_HIGH_LEVEL
      assert(handle_ready.exists() && !handle_ready.has_triggered());
#endif
      handle_ready.trigger();
#ifdef LEGION_SPY
      LegionSpy::log_implicit_dependence(parent_ctx->get_start_event(),
          ready_event);
      LegionSpy::log_op_events(unique_op_id, ready_event,
          completion_event);
      LegionSpy::log_implicit_dependence(completion_event,
          parent_ctx->get_task_completion());
      LegionSpy::log_op_proc_user(unique_op_id, local_proc.id);
      LegionSpy::log_event_dependence(handle_ready, ready_event);
      LegionSpy::log_event_dependence(ready_event, completion_event);
#endif
      complete_mapping();
      completion_event.trigger(ready_event);
      need_completion_trigger = false;
      complete_execution(ready_event);
      // return true since we succeeded
      return true;
    }

    //--------------------------------------------------------------------------
    unsigned DependentPartitionOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    FatTreePath* DependentPartitionOp::compute_fat_path(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
      handle_ready = UserEvent::create_user_event();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      privilege_path = RegionTreePath();
      if (!handle_ready.has_triggered())
        handle_ready.trigger();
      version_info.clear();
      restrict_info.clear();
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
      Operation::trigger_commit();
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
                                   parent_ctx->variants->name, 
                                   parent_ctx->get_unique_task_id(),
                                   unique_op_id, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_BAD_PARENT_REGION);
      }
      else
        parent_req_index = unsigned(parent_index);
    }


#ifdef LEGION_SPY
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

#endif
    ///////////////////////////////////////////////////////////// 
    // Fill Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FillOp::FillOp(Internal *rt)
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
                            const Predicate &pred, bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      initialize_speculation(ctx, true/*track*/, 1, pred);
      requirement = RegionRequirement(handle, WRITE_DISCARD, EXCLUSIVE, parent);
      requirement.privilege_fields.insert(fid);
      requirement.initialize_mapping_fields();
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
                            LogicalRegion parent, FieldID fid, const Future &f,
                            const Predicate &pred, bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      initialize_speculation(ctx, true/*track*/, 1, pred);
      requirement = RegionRequirement(handle, WRITE_DISCARD, EXCLUSIVE, parent);
      requirement.privilege_fields.insert(fid);
      requirement.initialize_mapping_fields();
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
                            const Predicate &pred, bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      initialize_speculation(ctx, true/*track*/, 1, pred);
      requirement = RegionRequirement(handle, WRITE_DISCARD, EXCLUSIVE, parent);
      requirement.privilege_fields = fields;
      requirement.initialize_mapping_fields();
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
                            const std::set<FieldID> &fields, const Future &f,
                            const Predicate &pred, bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      initialize_speculation(ctx, true/*track*/, 1, pred);
      requirement = RegionRequirement(handle, WRITE_DISCARD, EXCLUSIVE, parent);
      requirement.privilege_fields = fields;
      requirement.initialize_mapping_fields();
      future = f;
      if (check_privileges)
        check_fill_privilege();
      initialize_privilege_path(privilege_path, requirement);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void FillOp::perform_logging(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      LegionSpy::log_fill_operation(parent_ctx->get_unique_task_id(), 
                                    unique_op_id);
      LegionSpy::log_logical_requirement(unique_op_id, 0/*index*/,
                                         true/*region*/,
                                         requirement.region.index_space.id,
                                         requirement.region.field_space.id,
                                         requirement.region.tree_id,
                                         requirement.privilege,
                                         requirement.prop,
                                         requirement.redop);
      LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                        requirement.privilege_fields);
#endif
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
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, BEGIN_DEPENDENCE_ANALYSIS);
#endif
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
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, END_DEPENDENCE_ANALYSIS);
#endif
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_remote_state_analysis(UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      std::set<Event> preconditions;  
      version_info.make_local(preconditions, runtime->forest,
                              physical_ctx.get_id());
      if (preconditions.empty())
        ready_event.trigger();
      else
        ready_event.trigger(Event::merge_events(preconditions));
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
      if (!requirement.premapped)
      {
        Processor local_proc = parent_ctx->get_executing_processor();
        requirement.premapped = runtime->forest->premap_physical_region(
                  physical_ctx, privilege_path, requirement, version_info,
                  this, parent_ctx, local_proc
#ifdef DEBUG_HIGH_LEVEL
                  , 0/*idx*/, get_logging_name(), unique_op_id
#endif
                  );
#ifdef DEBUG_HIGH_LEVEL
        assert(requirement.premapped);
#endif
      }
      
      // Tell the region tree forest to fill in this field
      // Note that the forest takes ownership of the value buffer
      if (future.impl == NULL)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(value != NULL);
#endif
        runtime->forest->fill_fields(physical_ctx, requirement,
                                     value, value_size, version_info);
        std::set<Event> applied_conditions;
        version_info.apply_mapping(physical_ctx.get_id(),
                                   runtime->address_space, applied_conditions);
        // Clear value and value size since the forest ended up 
        // taking ownership of them
        value = NULL;
        value_size = 0;
        if (!applied_conditions.empty())
          complete_mapping(Event::merge_events(applied_conditions));
        else
          complete_mapping();
        complete_execution();
      }
      else
      {
        // If we have a future value see if its event has triggered
        Event future_ready_event = future.impl->get_ready_event();
        if (!future_ready_event.has_triggered())
        {
          // Launch a task to handle the deferred complete
          DeferredExecuteArgs deferred_execute_args;
          deferred_execute_args.hlr_id = HLR_DEFERRED_EXECUTION_TRIGGER_ID;
          deferred_execute_args.proxy_this = this;
          runtime->issue_runtime_meta_task(&deferred_execute_args,
                                           sizeof(deferred_execute_args),
                                           HLR_DEFERRED_EXECUTION_TRIGGER_ID,
                                           this, future_ready_event);
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
      runtime->forest->fill_fields(physical_ctx, requirement, 
                                   result, result_size, version_info);
      std::set<Event> applied_conditions;
      version_info.apply_mapping(physical_ctx.get_id(),
                                 runtime->address_space, applied_conditions);
      if (!applied_conditions.empty())
        complete_mapping(Event::merge_events(applied_conditions));
      else
        complete_mapping();
      complete_execution();
    }
    
    //--------------------------------------------------------------------------
    unsigned FillOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      Operation::trigger_commit();
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_INVALID_INSTANCE_FIELD);
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            log_region.error("Instance field %d is a duplicate for "
                                    "fill operation (ID %lld)",
                                    bad_field, unique_op_id);
#ifdef DEBUG_HIGH_LEVEL
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
                                   parent_ctx->variants->name, 
                                   parent_ctx->get_unique_task_id(),
                                   unique_op_id, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
                               parent_ctx->variants->name, 
                               parent_ctx->get_unique_task_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_BAD_PARENT_REGION);
      }
      else
        parent_req_index = unsigned(parent_index);
    }

    ///////////////////////////////////////////////////////////// 
    // Attach Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AttachOp::AttachOp(Internal *rt)
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
                        "FORGET THEM?!?", parent_ctx->variants->name,
                        parent_ctx->get_unique_task_id());

      }
      file_name = strdup(name);
      // Construct the region requirement for this task
      requirement = RegionRequirement(handle, WRITE_DISCARD, EXCLUSIVE, parent);
      requirement.initialize_mapping_fields();
      for (std::map<FieldID,const char*>::const_iterator it = fmap.begin();
            it != fmap.end(); it++)
      {
        requirement.add_field(it->first);
        field_map[it->first] = strdup(it->second);
      }
      file_mode = mode;
      region = PhysicalRegion(legion_new<PhysicalRegion::Impl>(requirement,
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
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, BEGIN_DEPENDENCE_ANALYSIS);
#endif
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, END_DEPENDENCE_ANALYSIS);
#endif
    }

    //--------------------------------------------------------------------------
    void AttachOp::trigger_remote_state_analysis(UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      std::set<Event> preconditions;  
      version_info.make_local(preconditions, runtime->forest,
                              physical_ctx.get_id());
      if (preconditions.empty())
        ready_event.trigger();
      else
        ready_event.trigger(Event::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    bool AttachOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      if (!requirement.premapped)
      {
        Processor local_proc = parent_ctx->get_executing_processor();
        requirement.premapped = runtime->forest->premap_physical_region(
                  physical_ctx, privilege_path, requirement, version_info,
                  this, parent_ctx, local_proc
#ifdef DEBUG_HIGH_LEVEL
                  , 0/*idx*/, get_logging_name(), unique_op_id
#endif
                  );
#ifdef DEBUG_HIGH_LEVEL
        assert(requirement.premapped);
#endif
      }
      
      InstanceRef result = runtime->forest->attach_file(physical_ctx,
                                                        requirement, this,
                                                        version_info);
#ifdef DEBUG_HIGH_LEVEL
      assert(result.has_ref());
#endif
      std::set<Event> applied_conditions;
      version_info.apply_mapping(physical_ctx.get_id(),
                                 runtime->address_space, applied_conditions);
      // This operation is ready once the file is attached
      region.impl->set_reference(result);
      // Once we have created the instance, then we are done
      if (!applied_conditions.empty())
        complete_mapping(Event::merge_events(applied_conditions));
      else
        complete_mapping();
      Event acquired_event = result.get_ready_event();
      completion_event.trigger(acquired_event);
      need_completion_trigger = false;
      complete_execution(acquired_event);
      // Should always succeed
      return true;
    }

    //--------------------------------------------------------------------------
    unsigned AttachOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void AttachOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      Operation::trigger_commit();
    }

    //--------------------------------------------------------------------------
    PhysicalInstance AttachOp::create_instance(const Domain &dom,
                                               const std::vector<size_t> &sizes)
    //--------------------------------------------------------------------------
    {
      // First build the set of field paths
      std::vector<const char*> field_files(field_map.size());
      unsigned idx = 0;
      for (std::map<FieldID,const char*>::const_iterator it = field_map.begin();
            it != field_map.end(); it++, idx++)
      {
        field_files[idx] = it->second;
      }
      // Now ask the low-level runtime to create the instance  
      PhysicalInstance result = dom.create_hdf5_instance(file_name, sizes,
                             field_files, (file_mode == LEGION_FILE_READ_ONLY));
#ifdef DEBUG_HIGH_LEVEL
      assert(result.exists());
#endif
      return result;
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_INVALID_INSTANCE_FIELD);
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            log_region.error("Instance field %d is a duplicate for "
                                    "attach operation (ID %lld)",
                                    bad_field, unique_op_id);
#ifdef DEBUG_HIGH_LEVEL
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
                                   parent_ctx->variants->name, 
                                   parent_ctx->get_unique_task_id(),
                                   unique_op_id, 
                                   requirement.region.index_space.id,
                                   requirement.region.field_space.id, 
                                   requirement.region.tree_id);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
                               parent_ctx->variants->name, 
                               parent_ctx->get_unique_task_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id);
#ifdef DEBUG_HIGH_LEVEL
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
    DetachOp::DetachOp(Internal *rt)
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
    void DetachOp::initialize_detach(SingleTask *ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      // No need to check privileges because we never would have been
      // able to attach in the first place anyway.
      requirement.copy_without_mapping_info(region.impl->get_requirement());
      requirement.initialize_mapping_fields();
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
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, BEGIN_DEPENDENCE_ANALYSIS);
#endif
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
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      unique_op_id, END_DEPENDENCE_ANALYSIS);
#endif
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_remote_state_analysis(UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      std::set<Event> preconditions;  
      version_info.make_local(preconditions, runtime->forest,
                              physical_ctx.get_id());
      if (preconditions.empty())
        ready_event.trigger();
      else
        ready_event.trigger(Event::merge_events(preconditions));
    }
    
    //--------------------------------------------------------------------------
    bool DetachOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      // Now we can get the reference we need for the detach operation
      InstanceRef reference = region.impl->get_reference();
      // Check that this is actually a file
      PhysicalManager *manager = reference.get_manager();
#ifdef DEBUG_HIGH_LEVEL
      assert(!manager->is_reduction_manager()); 
#endif
      InstanceManager *inst_manager = manager->as_instance_manager(); 
      if (!inst_manager->is_attached_file())
      {
        log_run.error("Illegal detach operation on a physical region which "
                      "was not attached!");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_ILLEGAL_DETACH_OPERATION);
      }

      RegionTreeContext physical_ctx = 
        parent_ctx->find_enclosing_context(parent_req_index);
      if (!requirement.premapped)
      {
        Processor local_proc = parent_ctx->get_executing_processor();
        requirement.premapped = runtime->forest->premap_physical_region(
                  physical_ctx, privilege_path, requirement, version_info,
                  this, parent_ctx, local_proc
#ifdef DEBUG_HIGH_LEVEL
                  , 0/*idx*/, get_logging_name(), unique_op_id
#endif
                  );
#ifdef DEBUG_HIGH_LEVEL
        assert(requirement.premapped);
#endif
      }
      Event detach_event = 
        runtime->forest->detach_file(physical_ctx, requirement, this,
                                     version_info, reference);
      std::set<Event> applied_conditions;
      version_info.apply_mapping(physical_ctx.get_id(),
                                 runtime->address_space, applied_conditions);
      if (!applied_conditions.empty())
        complete_mapping(Event::merge_events(applied_conditions));
      else
        complete_mapping();
      completion_event.trigger(detach_event);
      need_completion_trigger = false;
      complete_execution(detach_event);
      // This should always succeed
      return true;
    }

    //--------------------------------------------------------------------------
    unsigned DetachOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
      Operation::trigger_commit();
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
                               parent_ctx->variants->name, 
                               parent_ctx->get_unique_task_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id);
#ifdef DEBUG_HIGH_LEVEL
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
    TimingOp::TimingOp(Internal *rt)
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
      result = Future(legion_new<Future::Impl>(runtime, true/*register*/,
                  runtime->get_available_distributed_id(true),
                  runtime->address_space, runtime->address_space, this));
      return result;
    }

    //--------------------------------------------------------------------------
    Future TimingOp::initialize_microseconds(SingleTask *ctx, const Future &pre)
    //--------------------------------------------------------------------------
    {
      kind = MICROSECOND_MEASUREMENT;
      precondition = pre;
      result = Future(legion_new<Future::Impl>(runtime, true/*register*/,
                  runtime->get_available_distributed_id(true),
                  runtime->address_space, runtime->address_space, this));
      return result;
    }

    //--------------------------------------------------------------------------
    Future TimingOp::initialize_nanoseconds(SingleTask *ctx, const Future &pre)
    //--------------------------------------------------------------------------
    {
      kind = NANOSECOND_MEASUREMENT;
      precondition = pre;
      result = Future(legion_new<Future::Impl>(runtime, true/*register*/,
                  runtime->get_available_distributed_id(true),
                  runtime->address_space, runtime->address_space, this));
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
        Event wait_on = precondition.impl->get_ready_event();
        DeferredExecuteArgs args;
        args.hlr_id = HLR_DEFERRED_EXECUTE_ID;
        args.proxy_this = this;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_DEFERRED_EXECUTE_ID,
                                         this, wait_on);
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
 
  }; // namespace LegionRuntime
}; // namespace HighLevel

// EOF

