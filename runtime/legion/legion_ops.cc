/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#include "legion/runtime.h"
#include "legion/legion_ops.h"
#include "legion/legion_tasks.h"
#include "legion/region_tree.h"
#include "legion/legion_spy.h"
#include "legion/legion_trace.h"
#include "legion/legion_context.h"
#include "legion/legion_profiling.h"
#include "legion/legion_instances.h"
#include "legion/legion_views.h"

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
      : runtime(rt), gen(0), unique_op_id(0), context_index(0), 
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
      context_index = 0;
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
      execution_fence_event = ApEvent::NO_AP_EVENT;
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
      // Generation is bumped when we committed
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
      if ((trace != NULL) && (trace->remove_reference()))
        delete trace;
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
      return parent_ctx->get_task();
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
        if (!req.region.exists())
          return;
        runtime->forest->initialize_path(req.region.get_index_space(),
                                         req.parent.get_index_space(), path);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(req.handle_type == PART_PROJECTION);
#endif
        if (!req.partition.exists())
          return;
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
    void Operation::set_trace(LegionTrace *t, bool is_tracing,
                              const std::vector<StaticDependence> *dependences)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace == NULL);
      assert(t != NULL);
#endif
      trace = t; 
      trace->add_reference();
      tracing = is_tracing;
      trace->record_static_dependences(this, dependences);
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
      // Write discard privileges become read-write inside the operation
      if (r.privilege == WRITE_DISCARD)
        r.privilege = READ_WRITE;
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
          delete it->first;
      }
      acquired_instances.clear();
    }

    //--------------------------------------------------------------------------
    void Operation::initialize_operation(TaskContext *ctx, bool track, 
                                         unsigned regs/*= 0*/,
                      const std::vector<StaticDependence> *dependences/*=NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
      assert(completion_event.exists());
#endif
      parent_ctx = ctx;
      track_parent = track;
      if (track_parent)
        context_index = 
          parent_ctx->register_new_child_operation(this, dependences);
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
    void Operation::execute_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Always wrap this call with calls to begin/end dependence analysis
      begin_dependence_analysis();
      trigger_dependence_analysis();
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    RtEvent Operation::issue_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      if (has_prepipeline_stage())
      {
        PrepipelineArgs args;
        args.proxy_this = this;
        // Give this deferred throughput priority so that it is always
        // ahead of the logical analysis
        return runtime->issue_runtime_meta_task(args, 
                LG_THROUGHPUT_DEFERRED_PRIORITY, this);
      }
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    bool Operation::has_prepipeline_stage(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // Should only be called by inherited types
      assert(false); 
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do in the base case
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Put this thing on the ready queue
      enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Mark that we finished mapping
      complete_mapping();
      // If we have nothing to do also mark that we have completed execution
      complete_execution();
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
          // Bump the generation
          gen++;
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
    unsigned Operation::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return 0;
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
    ApEvent Operation::get_restrict_precondition(void) const
    //--------------------------------------------------------------------------
    {
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent Operation::merge_restrict_preconditions(
                                const std::vector<Grant> &grants, 
                                const std::vector<PhaseBarrier> &wait_barriers)
    //--------------------------------------------------------------------------
    {
      if (!grants.empty())
        assert(false); // Figure out how to deduplicate grant acquires
      if (wait_barriers.empty())
        return ApEvent::NO_AP_EVENT;
      if (wait_barriers.size() == 1)
        return Runtime::get_previous_phase(wait_barriers[0].phase_barrier);
      std::set<ApEvent> wait_events;
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        wait_events.insert(
            Runtime::get_previous_phase(wait_barriers[idx].phase_barrier));
      return Runtime::merge_events(wait_events);
    }

    //--------------------------------------------------------------------------
    void Operation::record_restrict_postcondition(ApEvent postcondition)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Operation:: add_copy_profiling_request(
                                           Realm::ProfilingRequestSet &requests)
    //--------------------------------------------------------------------------
    {
      // Do nothing
    }

    //--------------------------------------------------------------------------
    void Operation::handle_profiling_response(
                                       const Realm::ProfilingResponse &response)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Operation::filter_copy_request_kinds(MapperManager *mapper,
        const std::set<ProfilingMeasurementID> &requests,
        std::vector<ProfilingMeasurementID> &results, bool warn_if_not_copy)
    //--------------------------------------------------------------------------
    {
      for (std::set<ProfilingMeasurementID>::const_iterator it = 
            requests.begin(); it != requests.end(); it++)
      {
        switch ((Realm::ProfilingMeasurementID)*it)
        {
          case Realm::PMID_OP_STATUS:
          case Realm::PMID_OP_BACKTRACE:
          case Realm::PMID_OP_TIMELINE:
          case Realm::PMID_OP_MEM_USAGE:
            {
              results.push_back(*it);
              break;
            }
          default:
            {
              if (warn_if_not_copy) 
              {
                REPORT_LEGION_WARNING(LEGION_WARNING_NOT_COPY,
                            "Mapper %s requested a profiling "
                            "measurement of type %d which is not applicable to "
                            "operation %s (UID %lld) and will be ignored.",
                            mapper->get_mapper_name(), *it, get_logging_name(),
                            get_unique_op_id());
              }
            }
        }
      }
    }

    //--------------------------------------------------------------------------
    MaterializedView* Operation::create_temporary_instance(PhysicalManager *dst,
                                 unsigned index, const FieldMask &needed_fields)
    //--------------------------------------------------------------------------
    {
      PhysicalManager *result= 
        select_temporary_instance(dst, index, needed_fields);
#ifdef DEBUG_LEGION
      assert(result->is_instance_manager());
#endif
      InstanceView *view = parent_ctx->create_instance_top_view(result, 
                                                runtime->address_space);
      return view->as_materialized_view();
    }

    //--------------------------------------------------------------------------
    PhysicalManager* Operation::select_temporary_instance(PhysicalManager *dst,
                                 unsigned index, const FieldMask &needed_fields)
    //--------------------------------------------------------------------------
    {
      // Should only be called for interhited types
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    void Operation::validate_temporary_instance(PhysicalManager *result,
                                  std::set<PhysicalManager*> &previous_managers,
           const std::map<PhysicalManager*,std::pair<unsigned,bool> > &acquired,
                                  const FieldMask &needed_fields,
                                  LogicalRegion needed_region,
                                  MapperManager *mapper,
                                  const char *mapper_call_name) const 
    //--------------------------------------------------------------------------
    {
      if (!!(needed_fields - result->layout->allocated_fields))
        // Doesn't have all the fields
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of '%s' on "
                      "mapper %s. The temporary instance selected for %s "
                      "(UID %lld) did not have space for all the necessary "
                      "fields.", mapper_call_name, mapper->get_mapper_name(),
                      get_logging_name(), unique_op_id)
      std::vector<LogicalRegion> needed_regions(1, needed_region);
      if (!result->meets_regions(needed_regions))
        // Doesn't meet the needed region
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of '%s' on "
                      "mapper %s. The temporary instance selected for %s "
                      "(UID %lld) is not large enough for the necessary "
                      "logical region.", mapper_call_name,
                      mapper->get_mapper_name(), get_logging_name(),
                      unique_op_id)
      std::map<PhysicalManager*,std::pair<unsigned,bool> >::const_iterator
        finder = acquired.find(result);
      if (finder == acquired.end())
        // Not acquired, these must be acquired so we can properly
        // check that it is a fresh instance
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of '%s' on "
                      "mapper %s. The temporary instance selected for %s "
                      "(UID %lld) was not properly acquired.",
                      mapper_call_name, mapper->get_mapper_name(),
                      get_logging_name(), unique_op_id)
      // Little hack: permit this if we are doing replay mapping
      if ((Runtime::replay_file == NULL) && (!finder->second.second || 
          (previous_managers.find(result) != previous_managers.end())))
        // Not a fresh instance
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of '%s' on "
                      "mapper %s. The temporary instance selected for %s "
                      "(UID %lld) is not a freshly created instance.",
                      mapper_call_name, mapper->get_mapper_name(),
                      get_logging_name(), unique_op_id)
    }

    //--------------------------------------------------------------------------
    void Operation::log_temporary_instance(PhysicalManager *result, 
                           unsigned index, const FieldMask &needed_fields) const
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID> fields;
      result->region_node->column_source->get_field_set(needed_fields, fields); 
      for (std::vector<FieldID>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        LegionSpy::log_temporary_instance(unique_op_id, index, 
                                          *it, result->get_use_event());
      }
    }

    //--------------------------------------------------------------------------
    void Operation::enqueue_ready_operation(RtEvent wait_on/*=Event::NO_EVENT*/,
                           LgPriority priority/*= LG_THROUGHPUT_WORK_PRIORITY*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Only index space tasks should come through this path
      if (get_operation_kind() == TASK_OP_KIND)
        assert(dynamic_cast<IndexTask*>(this) != NULL);
#endif
      Processor p = parent_ctx->get_executing_processor();
      runtime->add_to_local_queue(p, this, priority, wait_on);
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
        DeferredExecArgs args;
        args.proxy_this = this;
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_DEFERRED_PRIORITY,
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
        args.proxy_this = this;
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_DEFERRED_PRIORITY,
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
        args.proxy_this = this;
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_DEFERRED_PRIORITY,
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
          gen++;
        }
        else if (outstanding_mapping_references == 0)
        {
#ifdef DEBUG_LEGION
          assert(dependence_tracker.commit != NULL);
#endif
          CommitDependenceTracker *tracker = dependence_tracker.commit;
          need_trigger = tracker->issue_commit_trigger(this, runtime);
          if (need_trigger)
          {
            trigger_commit_invoked = true;
            gen++;
          }
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
        args.proxy_this = this;
        args.deactivate = do_deactivate;
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_DEFERRED_PRIORITY,
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
          gen++;
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
      execution_fence_event = parent_ctx->register_fence_dependence(this);
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
#ifdef LEGION_SPY
      // Enforce serial mapping dependences for validating runtime analysis
      // Don't record it though to avoid confusing legion spy
      // Do this after all the rest of the dependence analysis to catch
      // dependences on any close operations that were generated
      RtEvent previous_mapped = 
        parent_ctx->update_previous_mapped_event(mapped_event);
      if (previous_mapped.exists())
        dependence_tracker.mapping->add_mapping_dependence(previous_mapped);
#endif
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
#ifdef DEBUG_LEGION
      assert(our_gen <= gen); // better not be ahead of where we are now
#endif
      // If the generations match and we haven't committed yet, 
      // register an outgoing dependence
      if (our_gen == gen)
      {
        AutoLock o_lock(op_lock);
        // Retest generation to see if we lost the race
        if ((our_gen == gen) && !committed)
        {
#ifdef DEBUG_LEGION
          // should still have some mapping references
          // if other operations are trying to register dependences
          // This assertion no longer holds because of how we record
          // fence dependences from context operation lists which 
          // don't track mapping dependences
          //assert(outstanding_mapping_references > 0);
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
            dependence_tracker.commit->add_commit_dependence(
                                          other_commit_event);
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
    bool Operation::add_mapping_reference(GenerationID our_gen)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(our_gen <= gen); // better not be ahead of where we are now
#endif
      if (our_gen < gen)
        return false;
      outstanding_mapping_references++;
      return true;
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
            {
              trigger_commit_invoked = true;
              gen++;
            }
          }
        }
        // otherwise we were already recycled and are no longer valid
      }
      if (need_trigger)
        trigger_commit();
    }

    //--------------------------------------------------------------------------
    void Operation::record_logical_dependence(const LogicalUser &user)
    //--------------------------------------------------------------------------
    {
      // Record the advance operations separately, in many cases we don't
      // need to include them in our analysis of above users, but in the case
      // of creating new advance operations below in the tree we do
      if (user.op->get_operation_kind() == ADVANCE_OP_KIND)
        logical_advances.push_back(user);
      else
        logical_records.push_back(user);
    }

    //--------------------------------------------------------------------------
    void Operation::clear_logical_records(void)
    //--------------------------------------------------------------------------
    {
      logical_records.clear();
      logical_advances.clear();
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
            gen++;
          }
        }
      }
      if (need_trigger)
        trigger_commit();
    }

    //--------------------------------------------------------------------------
    InnerContext* Operation::find_logical_context(unsigned index)
    //--------------------------------------------------------------------------
    {
      return parent_ctx->find_parent_logical_context(find_parent_index(index));
    }

    //--------------------------------------------------------------------------
    InnerContext* Operation::find_physical_context(unsigned index)
    //--------------------------------------------------------------------------
    {
      return parent_ctx->find_parent_physical_context(find_parent_index(index));
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::prepare_for_mapping(const InstanceRef &ref,
                                                   MappingInstance &instance) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!ref.is_virtual_ref());
#endif
      instance = ref.get_mapping_instance();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::prepare_for_mapping(const InstanceSet &valid,
                                      std::vector<MappingInstance> &input_valid)
    //--------------------------------------------------------------------------
    {
      unsigned offset = input_valid.size();
      input_valid.resize(offset + valid.size());
      for (unsigned idx = 0; idx < valid.size(); idx++)
      {
        const InstanceRef &ref = valid[idx];
#ifdef DEBUG_LEGION
        assert(!ref.is_virtual_ref());
#endif
        MappingInstance &inst = input_valid[offset+idx];
        inst = ref.get_mapping_instance();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::prepare_for_mapping(const InstanceSet &valid,
                                      const std::set<Memory> &visible_filter,
                                      std::vector<MappingInstance> &input_valid)
    //--------------------------------------------------------------------------
    {
      unsigned offset = input_valid.size();
      input_valid.reserve(offset+valid.size());
      unsigned next_index = offset;
      for (unsigned idx = 0; idx < valid.size(); idx++)
      {
        const InstanceRef &ref = valid[idx];
#ifdef DEBUG_LEGION
        assert(!ref.is_virtual_ref());
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
    void Operation::perform_projection_version_analysis(
                                const ProjectionInfo &proj_info, 
                                const RegionRequirement &owner_req,
                                const RegionRequirement &local_req, 
                                const unsigned idx,
                                const UniqueID logical_context_uid, 
                                VersionInfo &version_info, 
                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      RegionTreeNode *parent_node;
      if (owner_req.handle_type == PART_PROJECTION)
        parent_node = runtime->forest->get_node(owner_req.partition);
      else
        parent_node = runtime->forest->get_node(owner_req.region);
#ifdef DEBUG_LEGION
      assert(local_req.handle_type == SINGULAR);
#endif
      RegionTreeNode *child_node = 
        runtime->forest->get_node(local_req.region);
      // If they are the same node, we are already done
      if (child_node == parent_node)
        return;
      // Compute our privilege full projection path 
      RegionTreePath projection_path;
      runtime->forest->initialize_path(child_node->get_row_source(),
                     parent_node->get_row_source(), projection_path);
      // Any opens/advances have already been generated to the
      // upper bound node, so we don't have to handle that node, 
      // therefore all our paths must start at one node below the
      // upper bound node
      RegionTreeNode *one_below = parent_node->get_tree_child(
              projection_path.get_child(parent_node->get_depth()));
      RegionTreePath one_below_path;
      one_below_path.initialize(projection_path.get_min_depth()+1, 
                                projection_path.get_max_depth());
      for (unsigned idx2 = projection_path.get_min_depth()+1; 
            idx2 < projection_path.get_max_depth(); idx2++)
        one_below_path.register_child(idx2, projection_path.get_child(idx2));
      const LegionMap<ProjectionEpochID,FieldMask>::aligned &proj_epochs = 
        proj_info.get_projection_epochs();
      const LegionMap<unsigned,FieldMask>::aligned empty_dirty_previous;
      // Do the analysis to see if we've opened all the nodes to the child
      {
        for (LegionMap<ProjectionEpochID,FieldMask>::aligned::const_iterator
              it = proj_epochs.begin(); it != proj_epochs.end(); it++)
        {
          // Advance version numbers from one below the upper bound
          // all the way down to the child
          runtime->forest->advance_version_numbers(this, idx, 
              false/*update parent state*/, false/*doesn't matter*/,
              logical_context_uid, true/*dedup opens*/, 
              false/*dedup advance*/, it->first, 0/*id*/, one_below, 
              one_below_path, it->second, empty_dirty_previous, ready_events);
        }
      }
      // If we're doing something other than reading, we need
      // to also do the advance for anything open below, we do
      // this from the one below node to the node above the child node
      // The exception is if we are reducing in which case we go from
      // the all the way to the bottom so that the first reduction
      // point bumps the version number appropriately. Another exception is 
      // for dirty reductions where we know that there is already a write 
      // at the base level so we don't need to do an advance to get our 
      // reduction registered with the parent VersionState object

      if (!IS_READ_ONLY(local_req) && 
          ((one_below != child_node) || 
           (IS_REDUCE(local_req) && !proj_info.is_dirty_reduction())))
      {
        RegionTreePath advance_path;
        // If we're a reduction we go all the way to the bottom
        // otherwise if we're read-write we go to the level above
        // because our version_analysis call will do the advance
        // at the destination node.           
        if (IS_REDUCE(local_req) && !proj_info.is_dirty_reduction())
        {
#ifdef DEBUG_LEGION
          assert((one_below->get_depth() < child_node->get_depth()) ||
                 (one_below == child_node)); 
#endif
          advance_path = one_below_path;
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(one_below->get_depth() < child_node->get_depth()); 
#endif
          advance_path.initialize(one_below_path.get_min_depth(), 
                                  one_below_path.get_max_depth()-1);
          for (unsigned idx2 = one_below_path.get_min_depth(); 
                idx2 < (one_below_path.get_max_depth()-1); idx2++)
            advance_path.register_child(idx2, one_below_path.get_child(idx2));
        }
        const bool parent_is_upper_bound = 
          (owner_req.handle_type != PART_PROJECTION) && 
          (owner_req.region == owner_req.parent);
        for (LegionMap<ProjectionEpochID,FieldMask>::aligned::const_iterator
              it = proj_epochs.begin(); it != proj_epochs.end(); it++)
        {
          // Advance version numbers from the upper bound to one above
          // the target child for split version numbers
          runtime->forest->advance_version_numbers(this, idx, 
              true/*update parent state*/, parent_is_upper_bound,
              logical_context_uid, false/*dedup opens*/, 
              true/*dedup advances*/, 0/*id*/, it->first, one_below, 
              advance_path, it->second, empty_dirty_previous, ready_events);
        }
      }
      // Now we can record our version numbers just like everyone else
      // We can skip the check for virtual version information because
      // our owner slice already did it
      runtime->forest->perform_versioning_analysis(this, idx, local_req,
                                    one_below_path, version_info, 
                                    ready_events, false/*partial*/, 
                                    false/*disjoint close*/, NULL/*filter*/,
                                    one_below, logical_context_uid, 
                                    &proj_epochs, true/*skip parent check*/);
    }

    //--------------------------------------------------------------------------
    void Operation::MappingDependenceTracker::issue_stage_triggers(
                      Operation *op, Runtime *runtime, MustEpochOp *must_epoch)
    //--------------------------------------------------------------------------
    {
      bool resolve_now = true;
      RtEvent map_precondition;
      if (!mapping_dependences.empty())
        map_precondition = Runtime::merge_events(mapping_dependences);
      if (must_epoch == NULL)
      {
        // We always launch the task to avoid expensive recursive calls
        DeferredReadyArgs args;
        args.proxy_this = op;
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_DEFERRED_PRIORITY,
                                         op, map_precondition);
      }
      else if (!map_precondition.has_triggered())
        must_epoch->add_mapping_dependence(map_precondition);

      if (!resolution_dependences.empty())
      {
        RtEvent resolve_precondition = 
          Runtime::merge_events(resolution_dependences);
        if (!resolve_precondition.has_triggered())
        {
          DeferredResolutionArgs args;
          args.proxy_this = op;
          runtime->issue_runtime_meta_task(args,LG_THROUGHPUT_DEFERRED_PRIORITY,
                                           op, resolve_precondition);
          resolve_now = false;
        }
      }
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
          args.proxy_this = op;
          args.gen = op->get_generation();
          runtime->issue_runtime_meta_task(args,LG_THROUGHPUT_DEFERRED_PRIORITY,
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
      collect_predicate = RtUserEvent::NO_RT_USER_EVENT;
      predicate_references = 0;
      true_guard = PredEvent::NO_PRED_EVENT;
      false_guard = PredEvent::NO_PRED_EVENT;
      can_result_future_complete = false;
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
      result_future = Future();
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
      bool remove_reference;
      GenerationID task_gen = 0;  // initialization to make gcc happy
      RtUserEvent to_trigger;
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
          to_trigger = collect_predicate;
        }
      }
      if (remove_reference)
        remove_mapping_reference(task_gen);
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      // Also check to see if we need to complete our future
      bool complete_future = false;
      {
        AutoLock o_lock(op_lock);
        if (result_future.impl != NULL)
          complete_future = true;
        else
          can_result_future_complete = true;
      }
      if (complete_future)
      {
        result_future.impl->set_result(&predicate_value, 
                                       sizeof(predicate_value), false/*own*/);
        result_future.impl->complete_future();
      }
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      {
        AutoLock o_lock(op_lock);
        // See if we have any outstanding references, if so make a precondition
        if (predicate_references > 0)
        {
          collect_predicate = Runtime::create_rt_user_event();
          precondition = collect_predicate;
        }
        
      } 
      commit_operation(true/*deactivate*/, precondition);
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
    PredEvent PredicateImpl::get_true_guard(void)
    //--------------------------------------------------------------------------
    {
      bool trigger = false;
      bool poison = false;
      PredEvent result;
      {
        AutoLock o_lock(op_lock);
        if (!true_guard.exists())
          true_guard = Runtime::create_pred_event();
        result = true_guard;
        if (predicate_resolved)
        {
          if (predicate_value)
            trigger = true;
          else
            poison = true;
        }
      }
      if (trigger)
        Runtime::trigger_event(result);
      else if (poison)
        Runtime::poison_event(result);
      return result;
    }

    //--------------------------------------------------------------------------
    PredEvent PredicateImpl::get_false_guard(void)
    //--------------------------------------------------------------------------
    {
      bool trigger = false;
      bool poison = false;
      PredEvent result;
      {
        AutoLock o_lock(op_lock);
        if (!false_guard.exists())
          false_guard = Runtime::create_pred_event();
        result = false_guard;
        if (predicate_resolved)
        {
          if (predicate_value)
            poison = true;
          else
            trigger = true;
        }
      }
      if (trigger)
        Runtime::trigger_event(result);
      else if (poison)
        Runtime::poison_event(result);
      return result;
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::get_predicate_guards(PredEvent &true_result,
                                             PredEvent &false_result)
    //--------------------------------------------------------------------------
    {
      bool handle_true = false;
      bool handle_false = false;
      {
        AutoLock o_lock(op_lock);
        if (!true_guard.exists())
          true_guard = Runtime::create_pred_event();
        true_result = true_guard;
        if (!false_guard.exists())
          false_guard = Runtime::create_pred_event();
        false_result = false_guard;
        if (predicate_resolved)
        {
          if (predicate_value)
            handle_true = true;
          else
            handle_false = true;
        }
      }
      if (handle_true)
      {
        Runtime::trigger_event(true_result);
        Runtime::poison_event(false_result);
      }
      else if (handle_false)
      {
        Runtime::poison_event(true_result);
        Runtime::trigger_event(false_result);
      }
    }

    //--------------------------------------------------------------------------
    Future PredicateImpl::get_future_result(void)
    //--------------------------------------------------------------------------
    {
      bool complete_future = false;
      if (result_future.impl == NULL)
      {
        Future temp = Future(
              new FutureImpl(runtime, true/*register*/,
                runtime->get_available_distributed_id(),
                runtime->address_space, this));
        AutoLock o_lock(op_lock);
        // See if we lost the race
        if (result_future.impl == NULL)
        {
          result_future = temp; 
          // if the predicate is complete we can complete the future
          complete_future = can_result_future_complete; 
        }
      }
      if (complete_future)
      {
        result_future.impl->set_result(&predicate_value, 
                                sizeof(predicate_value), false/*owned*/);
        result_future.impl->complete_future();
      }
      return result_future;
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::set_resolved_value(GenerationID pred_gen, bool value)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = true;
      // Make a copy of the waiters since we could get cleaned up in parallel
      std::map<PredicateWaiter*,GenerationID> copy_waiters;
      PredEvent to_trigger, to_poison;
      {
        AutoLock o_lock(op_lock);
        if ((pred_gen == get_generation()) && !predicate_resolved)
        {
          predicate_resolved = true;
          predicate_value = value;
          copy_waiters = waiters;
          if (predicate_value)
          {
            to_trigger = true_guard;
            to_poison = false_guard;
          }
          else
          {
            to_poison = true_guard;
            to_trigger = false_guard;
          }
        }
        else
          need_trigger = false;
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
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
      if (to_poison.exists())
        Runtime::poison_event(to_poison);
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
      speculation_state = PENDING_ANALYSIS_STATE;
      predicate = NULL;
      speculate_mapping_only = false;
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
    void SpeculativeOp::initialize_speculation(TaskContext *ctx, bool track,
        unsigned regions, const std::vector<StaticDependence> *dependences,
        const Predicate &p)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, track, regions, dependences);
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
        speculation_state = PENDING_ANALYSIS_STATE;
        predicate = p.impl;
        predicate->add_predicate_reference();
        if (Runtime::legion_spy_enabled)
          LegionSpy::log_predicate_use(unique_op_id, 
                                       predicate->get_unique_op_id());
      }
    }

    //--------------------------------------------------------------------------
    void SpeculativeOp::register_predicate_dependence(void)
    //--------------------------------------------------------------------------
    {
      if (predicate != NULL)
      {
        register_dependence(predicate, predicate->get_generation());
        // Now we can remove our predicate reference
        predicate->remove_predicate_reference();
      }
    }

    //--------------------------------------------------------------------------
    bool SpeculativeOp::is_predicated_op(void) const
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
        wait_event.lg_wait();
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
    void SpeculativeOp::execute_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Quick out
      if (predicate == NULL)
      {
#ifdef DEBUG_LEGION
        assert((speculation_state == RESOLVE_TRUE_STATE) ||
               (speculation_state == RESOLVE_FALSE_STATE));
#endif
        if (speculation_state == RESOLVE_FALSE_STATE)
        {
          if (Runtime::legion_spy_enabled)
            LegionSpy::log_predicated_false_op(unique_op_id);
          // Still need a commit tracker here
          dependence_tracker.commit = new CommitDependenceTracker();
          resolve_false(false/*speculated*/, false/*launched*/);
        }
        else
        {
          resolve_true(false/*speculated*/, false/*launched*/);
          Operation::execute_dependence_analysis();
        }
        return;
      }
      // Register ourselves as a waiter on the predicate value
      // If the predicate hasn't resolved yet, then we can ask the
      // mapper if it would like us to speculate on the value.
      // Then take the lock and set up our state.
      bool value, speculated = false;
      bool valid = predicate->register_waiter(this, get_generation(), value);
      // We don't support speculation for legion spy validation runs
      // as it doesn't really understand the event graphs that get
      // generated because of the predication events
#ifndef LEGION_SPY
      if (!valid)
        speculated = query_speculate(value, speculate_mapping_only);
#endif
      // Now hold the lock and figure out what we should do
      bool continue_true = false;
      bool continue_false = false;
      bool launch_speculation = false;
      RtEvent wait_on;
      {
        AutoLock o_lock(op_lock);
        switch (speculation_state)
        {
          case PENDING_ANALYSIS_STATE:
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
              }
              else if (speculated)
              {
                // Always launch in the speculated state
                launch_speculation = true;
                if (value)
                  speculation_state = SPECULATE_TRUE_STATE;
                else
                  speculation_state = SPECULATE_FALSE_STATE;
              }
              // Otherwise just stay in pending analysis state
              // and wait for the result of the predicate
              else
              {
                if (!predicate_waiter.exists())
                  predicate_waiter = Runtime::create_rt_user_event();
                wait_on = predicate_waiter;
              }
              break;
            }
          case RESOLVE_TRUE_STATE:
            {
              // Someone else has already resolved us to true so
              // we are good to go
              continue_true = true;
              break;
            }
          case RESOLVE_FALSE_STATE:
            {
              // Someone else has already resolved us to false so
              // do the opposite thing
              continue_false = true;
              break;
            }
          default:
            assert(false); // shouldn't be in the other states
        }
      }
      // Handle the waiting case if necessary
      if (wait_on.exists())
      {
        wait_on.lg_wait();
        // Now retake the lock and see if anything changed
        AutoLock o_lock(op_lock);
        switch (speculation_state)
        {
          case RESOLVE_TRUE_STATE:
            {
              continue_true = true;
              break;
            }
          case RESOLVE_FALSE_STATE:
            {
              continue_false = true;
              break;
            }
          default:
            assert(false); // should not be in any other states
        }
      }
      // At most one of these should be true
#ifdef DEBUG_LEGION
      assert(!continue_true || !continue_false);
#endif
      if (continue_true)
        resolve_true(speculated, false/*launched*/);
      else if (continue_false)
      {
        if (Runtime::legion_spy_enabled)
          LegionSpy::log_predicated_false_op(unique_op_id);
        // Can remove our predicate reference since we don't need it anymore
        predicate->remove_predicate_reference();
        // Still need a commit tracker here
        dependence_tracker.commit = new CommitDependenceTracker();
        resolve_false(speculated, false/*launched*/);
      }
#ifdef DEBUG_LEGION
      else
        assert(launch_speculation);
#endif
      if (continue_true || launch_speculation)
        Operation::execute_dependence_analysis();
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
      bool need_trigger = false;
      bool need_resolve = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(pred_gen == get_generation());
#endif
        need_trigger = predicate_waiter.exists();
        need_resolve = received_trigger_resolution;
        switch (speculation_state)
        {
          case PENDING_ANALYSIS_STATE:
            {
              if (value)
                speculation_state = RESOLVE_TRUE_STATE;
              else
                speculation_state = RESOLVE_FALSE_STATE;
              break;
            }
          case SPECULATE_TRUE_STATE:
            {
              if (value) // We guessed right
              {
                speculation_state = RESOLVE_TRUE_STATE;
                continue_true = true;
              }
              else
              {
                // We guessed wrong
                speculation_state = RESOLVE_FALSE_STATE;
                continue_false = true;
              }
              break;
            }
          case SPECULATE_FALSE_STATE:
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
              break;
            }
          default:
            assert(false); // shouldn't be in any of the other states
        }
      }
      if (need_trigger)
        Runtime::trigger_event(predicate_waiter);
      if (continue_true)
        resolve_true(true/*speculated*/, true/*launched*/);
      else if (continue_false)
      {
        if (Runtime::legion_spy_enabled)
          LegionSpy::log_predicated_false_op(unique_op_id);
        resolve_false(true/*speculated*/, true/*launched*/);
      }
      if (need_resolve)
        resolve_speculation();
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
    PhysicalRegion MapOp::initialize(TaskContext *ctx, 
                                     const InlineLauncher &launcher,
                                     bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/, 1/*regions*/, 
                           launcher.static_dependences);
      if (launcher.requirement.privilege_fields.empty())
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_REGION_REQUIREMENT_INLINE,
                         "REGION REQUIREMENT OF INLINE MAPPING "
                         "IN TASK %s (ID %lld) HAS NO PRIVILEGE "
                         "FIELDS! DID YOU FORGET THEM?!?",
                         parent_ctx->get_task_name(),
                         parent_ctx->get_unique_id());
      }
      requirement = launcher.requirement;
      termination_event = Runtime::create_ap_user_event();
      grants = launcher.grants;
      // Register ourselves with all the grants
      for (unsigned idx = 0; idx < grants.size(); idx++)
        grants[idx].impl->register_operation(termination_event);
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
      layout_constraint_id = launcher.layout_constraint_id;
      region = PhysicalRegion(new PhysicalRegionImpl(requirement,
                              completion_event, true/*mapped*/, ctx, 
                              map_id, tag, false/*leaf*/, 
                              false/*virtual mapped*/, runtime));
      if (check_privileges)
        check_privilege();
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_mapping_operation(parent_ctx->get_unique_id(),
                                         unique_op_id);
      return region;
    }

    //--------------------------------------------------------------------------
    void MapOp::initialize(TaskContext *ctx, const PhysicalRegion &reg)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 1/*regions*/);
      parent_task = ctx->get_task();
      requirement = reg.impl->get_requirement();
      map_id = reg.impl->map_id;
      tag = reg.impl->tag;
      region = reg;
      termination_event = Runtime::create_ap_user_event();
      region.impl->remap_region(completion_event);
      // We're only really remapping it if it already had a physical
      // instance that we can use to make a valid value
      remap_region = region.impl->has_references();
      // No need to check the privileges here since we know that we have
      // them from the first time that we made this physical region
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_mapping_operation(parent_ctx->get_unique_id(),
                                         unique_op_id);
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
      outstanding_profiling_requests = 1; // start at 1 to guard
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
    }

    //--------------------------------------------------------------------------
    void MapOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      // Remove our reference to the region
      region = PhysicalRegion();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      privilege_path.clear();
      version_info.clear();
      restrict_info.clear();
#ifdef DEBUG_LEGION
      assert(acquired_instances.empty());
#endif
      acquired_instances.clear();
      atomic_locks.clear();
      map_applied_conditions.clear();
      mapped_preconditions.clear();
      profiling_requests.clear();
      if (mapper_data != NULL)
      {
        free(mapper_data);
        mapper_data = NULL;
        mapper_data_size = 0;
      }
      // Now return this operation to the queue
      runtime->free_map_op(this);
    } 

    //--------------------------------------------------------------------------
    const char* MapOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[MAP_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind MapOp::get_operation_kind(void) const
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
    void MapOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute our parent region requirement
      compute_parent_index();
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
      { 
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
    void MapOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      ProjectionInfo projection_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   restrict_info,
                                                   version_info,
                                                   projection_info,
                                                   privilege_path);
    }

    //--------------------------------------------------------------------------
    void MapOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Compute the version numbers for this mapping operation
      std::set<RtEvent> preconditions;
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement, 
                                                   privilege_path,
                                                   version_info,
                                                   preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void MapOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      InstanceSet mapped_instances;
      // If we are remapping then we know the answer
      // so we don't need to do any premapping
      if (remap_region)
      {
        region.impl->get_references(mapped_instances);
        runtime->forest->physical_register_only(requirement,
                                                version_info, restrict_info,
                                                this, 0/*idx*/, 
                                                termination_event,
                                                false/*defer add users*/,
                                                true/*read only locks*/,
                                                map_applied_conditions,
                                                mapped_instances, 
                                                NULL/*advance projections*/
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
        runtime->forest->physical_premap_only(this, 0/*idx*/, requirement,
                                              version_info, valid_instances);
        // Now we've got the valid instances so invoke the mapper
        invoke_mapper(valid_instances, mapped_instances);
        // Then we can register our mapped instances
        runtime->forest->physical_register_only(requirement,
                                                version_info, restrict_info,
                                                this, 0/*idx*/,
                                                termination_event, 
                                                false/*defer add users*/,
                                                true/*read only locks*/,
                                                map_applied_conditions,
                                                mapped_instances,
                                                NULL/*advance projections*/
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
      version_info.apply_mapping(map_applied_conditions);
      // If we have any wait preconditions from phase barriers or 
      // grants then we can add them to the mapping preconditions
      if (!wait_barriers.empty() || !grants.empty())
      {
        for (std::vector<PhaseBarrier>::const_iterator it = 
              wait_barriers.begin(); it != wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(*it); 
          mapped_preconditions.insert(e);
          if (Runtime::legion_spy_enabled)
            LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
        for (std::vector<Grant>::const_iterator it = grants.begin();
              it != grants.end(); it++)
        {
          ApEvent e = it->impl->acquire_grant();
          mapped_preconditions.insert(e);
        }
      }
      // Update our physical instance with the newly mapped instances
      // Have to do this before triggering the mapped event
      if (!mapped_preconditions.empty())
      {
        // If we have restricted postconditions, tell the physical instance
        // that it has an event to wait for before it is unmapped
        ApEvent wait_for = Runtime::merge_events(mapped_preconditions);
        region.impl->reset_references(mapped_instances, 
                                      termination_event, wait_for);
      }
      else // The normal path here
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
      // Chain all the unlock arrivals off the termination event
      if (!arrive_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::iterator it = 
              arrive_barriers.begin(); it != arrive_barriers.end(); it++)
        {
          if (Runtime::legion_spy_enabled)
            LegionSpy::log_phase_barrier_arrival(unique_op_id, 
                                                 it->phase_barrier);
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        termination_event);    
        }
      }
      // Remove profiling our guard and trigger the profiling event if necessary
      if ((__sync_add_and_fetch(&outstanding_profiling_requests, -1) == 0) &&
          profiling_reported.exists())
        Runtime::trigger_event(profiling_reported);
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
        deferred_execute_args.proxy_this = this;
        runtime->issue_runtime_meta_task(deferred_execute_args,
                                         LG_THROUGHPUT_DEFERRED_PRIORITY, this,
                                   Runtime::protect_event(map_complete_event));
      }
      else
        deferred_execute();
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
      version_info.clear();
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
    PhysicalManager* MapOp::select_temporary_instance(PhysicalManager *dst,
                                 unsigned index, const FieldMask &needed_fields)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      Mapper::CreateInlineTemporaryInput input;
      Mapper::CreateInlineTemporaryOutput output;
      input.destination_instance = MappingInstance(dst);
      if (!Runtime::unsafe_mapper)
      {
        // Fields and regions must both be met
        // The instance must be freshly created
        // Instance must be acquired
        std::set<PhysicalManager*> previous_managers;
        // Get the set of previous managers we've made
        for (std::map<PhysicalManager*,std::pair<unsigned,bool> >::
              const_iterator it = acquired_instances.begin(); it !=
              acquired_instances.end(); it++)
          previous_managers.insert(it->first);
        // Do the mapper call now
        mapper->invoke_inline_create_temporary(this, &input, &output);
        validate_temporary_instance(output.temporary_instance.impl,
            previous_managers, acquired_instances, needed_fields,
            requirement.region, mapper, "create_inline_temporary_instance");
      }
      else
        mapper->invoke_inline_create_temporary(this, &input, &output);
      if (Runtime::legion_spy_enabled)
        log_temporary_instance(output.temporary_instance.impl, 
                               index, needed_fields);
      return output.temporary_instance.impl;
    }

    //--------------------------------------------------------------------------
    void MapOp::record_restrict_postcondition(ApEvent restrict_postcondition)
    //--------------------------------------------------------------------------
    {
      mapped_preconditions.insert(restrict_postcondition);
    }

    //--------------------------------------------------------------------------
    UniqueID MapOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    } 

    //--------------------------------------------------------------------------
    unsigned MapOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
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
        REPORT_LEGION_ERROR(ERROR_PROJECTION_REGION_REQUIREMENTS,
                         "Projection region requirements are not "
                         "permitted for inline mappings (in task %s)",
                         parent_ctx->get_task_name())
      FieldID bad_field = AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, then check the privileges with the parent context
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, bad_index);
      switch (et)
      {
        case NO_ERROR:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            REPORT_LEGION_ERROR(ERROR_REQUIREMENTS_INVALID_REGION,
                             "Requirements for invalid region handle "
                             "(%x,%d,%d) for inline mapping "
                             "(ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             unique_op_id);
            break;
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) ||
            (requirement.handle_type == REG_PROJECTION)
            ? requirement.region.field_space :
            requirement.partition.field_space;
            REPORT_LEGION_ERROR(ERROR_FIELD_NOT_VALID_FIELD,
                            "Field %d is not a valid field of field "
                             "space %d for inline mapping (ID %lld)",
                             bad_field, sp.id, unique_op_id)
            break;
          }
        case ERROR_INVALID_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_PRIVILEGE,
                             "Instance field %d is not one of the "
                             "privilege fields for inline mapping "
                             "(ID %lld)",
                             bad_field, unique_op_id)
            break;
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_PRIVILEGE,
                             "Instance field %d is a duplicate for "
                             "inline mapping (ID %lld)",
                             bad_field, unique_op_id)
            break;
          }
        case ERROR_BAD_PARENT_REGION:
          {
            if (bad_index < 0) 
            {
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_INLINE,
                               "Parent task %s (ID %lld) of inline mapping "
                               "(ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) "
                               "as a parent of region requirement because "
                               "no 'parent' region had that name.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id);
            } 
            else if (bad_field == AUTO_GENERATE_ID) 
            {
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_INLINE,
                               "Parent task %s (ID %lld) of inline mapping "
                               "(ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) "
                               "as a parent of region requirement because "
                               "parent requirement %d did not have "
                               "sufficent privileges.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id, bad_index);
            } 
            else 
            {
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_INLINE,
                               "Parent task %s (ID %lld) of inline mapping "
                               "(ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) "
                               "as a parent of region requirement because "
                               "region requirement %d was missing field %d.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id,
                               bad_index, bad_field);
            }
            break;
          }
        case ERROR_BAD_REGION_PATH:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_NOT_SUBREGION,
                             "Region (%x,%x,%x) is not a "
                             "sub-region of parent region "
                             "(%x,%x,%x) for region requirement of inline "
                             "mapping (ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             requirement.parent.index_space.id,
                             requirement.parent.field_space.id,
                             requirement.parent.tree_id,
                             unique_op_id)
            break;
          }
        case ERROR_BAD_REGION_TYPE:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_REQUIREMENT_INLINE,
                             "Region requirement of inline mapping "
                             "(ID %lld) cannot find privileges for field "
                             "%d in parent task",
                             unique_op_id, bad_field)
            break;
          }
        case ERROR_BAD_REGION_PRIVILEGES:
          {
            REPORT_LEGION_ERROR(ERROR_PRIVILEGES_FOR_REGION,
                             "Privileges %x for region "
                             "(%x,%x,%x) are not a subset of privileges "
                             "of parent task's privileges for region "
                             "requirement of inline mapping (ID %lld)",
                             requirement.privilege,
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             unique_op_id)
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
        REPORT_LEGION_ERROR(ERROR_PARENT_TASK_INLINE,
                         "Parent task %s (ID %lld) of inline mapping "
                         "(ID %lld) does not have a region "
                         "requirement for region (%x,%x,%x) "
                         "as a parent of region requirement.",
                         parent_ctx->get_task_name(),
                         parent_ctx->get_unique_id(),
                         unique_op_id,
                         requirement.region.index_space.id,
                         requirement.region.field_space.id,
                         requirement.region.tree_id)
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
      output.profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      if (restrict_info.has_restrictions())
      {
        prepare_for_mapping(restrict_info.get_instances(), 
                            input.valid_instances);
      }
      else if (!requirement.is_no_access())
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
      if (!output.profiling_requests.empty())
      {
        filter_copy_request_kinds(mapper,
            output.profiling_requests.requested_measurements,
            profiling_requests, true/*warn*/);
        profiling_priority = output.profiling_priority;
      }
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
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'map_inline' "
                      "on mapper %s. Mapper selected instance from region "
                      "tree %d to satisfy a region requirement for an inline "
                      "mapping in task %s (ID %lld) whose logical region is "
                      "from region tree %d.", mapper->get_mapper_name(),
                      bad_tree, parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id(),
                      requirement.region.get_tree_id())
      if (!missing_fields.empty())
      {
        for (std::vector<FieldID>::const_iterator it = missing_fields.begin();
              it != missing_fields.end(); it++)
        {
          const void *name; size_t name_size;
          if (!runtime->retrieve_semantic_information(
               requirement.region.get_field_space(), *it, NAME_SEMANTIC_TAG,
               name, name_size, true, false))
            name = "(no name)";
          log_run.error("Missing instance for field %s (FieldID: %d)",
                        static_cast<const char*>(name), *it);
        }
        REPORT_LEGION_ERROR(ERROR_MISSING_INSTANCE_FIELD,
                      "Invalid mapper output from invocation of 'map_inline' "
                      "on mapper %s. Mapper failed to specify a physical "
                      "instance for %zd fields of the region requirement to "
                      "an inline mapping in task %s (ID %lld). The missing "
                      "fields are listed below.", mapper->get_mapper_name(),
                      missing_fields.size(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
        
      }
      if (!unacquired.empty())
      {
        for (std::vector<PhysicalManager*>::const_iterator it = 
              unacquired.begin(); it != unacquired.end(); it++)
        {
          if (acquired_instances.find(*it) == acquired_instances.end())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from 'map_inline' invocation "
                        "on mapper %s. Mapper selected physical instance for "
                        "inline mapping in task %s (ID %lld) which has already "
                        "been collected. If the mapper had properly acquired "
                        "this instance as part of the mapper call it would "
                        "have detected this. Please update the mapper to abide "
                        "by proper mapping conventions.", 
                        mapper->get_mapper_name(), parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
        }
        // If we did successfully acquire them, still issue the warning
        REPORT_LEGION_WARNING(ERROR_MAPPER_FAILED_ACQUIRE,
                        "mapper %s faield to acquire instance "
                        "for inline mapping operation in task %s (ID %lld) "
                        "in 'map_inline' call. You may experience undefined "
                        "behavior as a consequence.", mapper->get_mapper_name(),
                        parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id())
      }
      if (composite_index >= 0)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'map_inline' "
                      "on mapper %s. Mapper requested creation of a composite "
                      "instance for inline mapping in task %s (ID %lld).",
                      mapper->get_mapper_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
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
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of "
                          "'map_inline' on mapper %s. Mapper selected a "
                          "physical instance in memory " IDFMT " which is "
                          "not visible from processor " IDFMT ". The inline "
                          "mapping operation was issued in task %s (ID %lld).",
                          mapper->get_mapper_name(), mem.id, exec_proc.id,
                          parent_ctx->get_task_name(), 
                          parent_ctx->get_unique_id())
        }
      }
      // Iterate over the instances and make sure they are all valid
      // for the given logical region which we are mapping
      std::vector<LogicalRegion> regions_to_check(1, requirement.region);
      for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
      {
        if (!chosen_instances[idx].get_manager()->meets_regions(
                                                        regions_to_check))
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'map_inline' "
                        "on mapper %s. Mapper specified an instance that does "
                        "not meet the logical region requirement. The inline "
                        "mapping operation was issued in task %s (ID %lld).",
                        mapper->get_mapper_name(), parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
      }
      // If this is a reduction region requirement, make sure all the
      // chosen instances are specialized reduction instances
      if (IS_REDUCE(requirement))
      {
        for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
        {
          if (!chosen_instances[idx].get_manager()->is_reduction_manager())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of "
                          "'map_inline' on mapper %s. Mapper failed to select "
                          "specialized reduction instances for region "
                          "requirement with reduction-only privileges for "
                          "inline mapping operation in task %s (ID %lld).",
                          mapper->get_mapper_name(),parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id())
          std::map<PhysicalManager*,std::pair<unsigned,bool> >::const_iterator 
            finder = acquired_instances.find(
                chosen_instances[idx].get_manager());
#ifdef DEBUG_LEGION
          assert(finder != acquired_instances.end());
#endif
          if (!finder->second.second)
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocatino of "
                          "'map_inline' on mapper %s. Mapper made an illegal "
                          "decision to re-use a reduction instance for an "
                          "inline mapping in task %s (ID %lld). Reduction "
                          "instances are not currently permitted to be "
                          "recycled.", mapper->get_mapper_name(),
                          parent_ctx->get_task_name(), 
                          parent_ctx->get_unique_id())
        }
      }
      else
      {
        for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
        {
          if (!chosen_instances[idx].get_manager()->is_instance_manager())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of "
                          "'map_inline' on mapper %s. Mapper selected an "
                          "illegal specialized reduction instance for region "
                          "requirement without reduction privileges for "
                          "inline mapping operation in task %s (ID %lld).",
                          mapper->get_mapper_name(),parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id())
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
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output. Mapper %s selected "
                          "instance for inline mapping (ID %lld) in task %s "
                          "(ID %lld) which failed to satisfy the corresponding "
                          "layout constraints.", 
                          mapper->get_mapper_name(), get_unique_op_id(),
                          parent_ctx->get_task_name(), 
                          parent_ctx->get_unique_id())
        }
      }
    }

    //--------------------------------------------------------------------------
    void MapOp::add_copy_profiling_request(Realm::ProfilingRequestSet &requests)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      ProfilingResponseBase base(this);
      Realm::ProfilingRequest &request = requests.add_request( 
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
          &base, sizeof(base), profiling_priority);
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      int previous = __sync_fetch_and_add(&outstanding_profiling_requests, 1);
      if ((previous == 1) && !profiling_reported.exists())
        profiling_reported = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    void MapOp::handle_profiling_response(
                                       const Realm::ProfilingResponse &response)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapper != NULL);
#endif
      Mapping::Mapper::InlineProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      mapper->invoke_inline_report_profiling(this, &info);
#ifdef DEBUG_LEGION
      assert(outstanding_profiling_requests > 0);
      assert(profiling_reported.exists());
#endif
      int remaining = __sync_add_and_fetch(&outstanding_profiling_requests, -1);
      // If this was the last one, we can trigger our events
      if (remaining == 0)
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
      this->is_index_space = false;
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
    void CopyOp::initialize(TaskContext *ctx,
                            const CopyLauncher &launcher, bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_speculation(ctx, true/*track*/, 
                             launcher.src_requirements.size() + 
                               launcher.dst_requirements.size(), 
                             launcher.static_dependences,
                             launcher.predicate);
      src_requirements.resize(launcher.src_requirements.size());
      dst_requirements.resize(launcher.dst_requirements.size());
      src_versions.resize(launcher.src_requirements.size());
      dst_versions.resize(launcher.dst_requirements.size());
      src_restrict_infos.resize(launcher.src_requirements.size());
      dst_restrict_infos.resize(launcher.dst_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        if (launcher.src_requirements[idx].privilege_fields.empty())
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_SOURCE_REGION_REQUIREMENT,
                           "SOURCE REGION REQUIREMENT %d OF "
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
          REPORT_LEGION_WARNING(LEGION_WARNING_DESTINATION_REGION_REQUIREMENT,
                           "DESTINATION REGION REQUIREMENT %d OF"
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
      index_point = launcher.point;
      if (check_privileges)
      {
        if (src_requirements.size() != dst_requirements.size())
          REPORT_LEGION_ERROR(ERROR_NUMBER_SOURCE_REQUIREMENTS,
                        "Number of source requirements (%zd) does not "
                        "match number of destination requirements (%zd) "
                        "for copy operation (ID %lld) with parent "
                        "task %s (ID %lld)",
                        src_requirements.size(), dst_requirements.size(),
                        get_unique_id(), parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        {
          if (src_requirements[idx].privilege_fields.size() != 
              src_requirements[idx].instance_fields.size())
            REPORT_LEGION_ERROR(ERROR_COPY_SOURCE_REQUIREMENTS,
                          "Copy source requirement %d for copy operation "
                          "(ID %lld) in parent task %s (ID %lld) has %zd "
                          "privilege fields and %zd instance fields.  "
                          "Copy requirements must have exactly the same "
                          "number of privilege and instance fields.",
                          idx, get_unique_id(), 
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id(),
                          src_requirements[idx].privilege_fields.size(),
                          src_requirements[idx].instance_fields.size())
          if (!IS_READ_ONLY(src_requirements[idx]))
            REPORT_LEGION_ERROR(ERROR_COPY_SOURCE_REQUIREMENTS,
                          "Copy source requirement %d for copy operation "
                          "(ID %lld) in parent task %s (ID %lld) must "
                          "be requested with a read-only privilege.",
                          idx, get_unique_id(),
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id())
          check_copy_privilege(src_requirements[idx], idx, true/*src*/);
        }
        for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
        {
          if (dst_requirements[idx].privilege_fields.size() != 
              dst_requirements[idx].instance_fields.size())
            REPORT_LEGION_ERROR(ERROR_COPY_DESTINATION_REQUIREMENT,
                          "Copy destination requirement %d for copy "
                          "operation (ID %lld) in parent task %s "
                          "(ID %lld) has %zd privilege fields and %zd "
                          "instance fields.  Copy requirements must "
                          "have exactly the same number of privilege "
                          "and instance fields.", idx, 
                          get_unique_id(), 
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id(),
                          dst_requirements[idx].privilege_fields.size(),
                          dst_requirements[idx].instance_fields.size())
          if (!HAS_WRITE(dst_requirements[idx]))
            REPORT_LEGION_ERROR(ERROR_COPY_DESTINATION_REQUIREMENT,
                          "Copy destination requirement %d for copy "
                          "operation (ID %lld) in parent task %s "
                          "(ID %lld) must be requested with a "
                          "read-write or write-discard privilege.",
                          idx, get_unique_id(),
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id())
          check_copy_privilege(dst_requirements[idx], idx, false/*src*/);
        }
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        {
          IndexSpace src_space = src_requirements[idx].region.get_index_space();
          IndexSpace dst_space = dst_requirements[idx].region.get_index_space();
          if (!runtime->forest->are_compatible(src_space, dst_space))
            REPORT_LEGION_ERROR(ERROR_COPY_LAUNCHER_INDEX,
                          "Copy launcher index space mismatch at index "
                          "%d of cross-region copy (ID %lld) in task %s "
                          "(ID %lld). Source requirement with index "
                          "space %x and destination requirement "
                          "with index space %x do not have the "
                          "same number of dimensions or the same number "
                          "of elements in their element masks.",
                          idx, get_unique_id(),
                          parent_ctx->get_task_name(), 
                          parent_ctx->get_unique_id(),
                          src_space.id, dst_space.id)
          else if (!runtime->forest->is_dominated(src_space, dst_space))
            REPORT_LEGION_ERROR(ERROR_DESTINATION_INDEX_SPACE,
                          "Destination index space %x for "
                          "requirement %d of cross-region copy "
                          "(ID %lld) in task %s (ID %lld) is not "
                          "a sub-region of the source index space %x.", 
                          dst_space.id, idx, get_unique_id(),
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id(),
                          src_space.id)
        }
      }
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_copy_operation(parent_ctx->get_unique_id(),
                                      unique_op_id);
    }

    //--------------------------------------------------------------------------
    void CopyOp::activate_copy(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative();
      mapper = NULL;
      outstanding_profiling_requests = 1; // start at 1 to guard
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      predication_guard = PredEvent::NO_PRED_EVENT;
    }

    //--------------------------------------------------------------------------
    void CopyOp::deactivate_copy(void)
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
      src_restrict_infos.clear();
      dst_restrict_infos.clear();
#ifdef DEBUG_LEGION
      assert(acquired_instances.empty());
#endif
      acquired_instances.clear();
      atomic_locks.clear();
      map_applied_conditions.clear();
      restrict_postconditions.clear();
      profiling_requests.clear();
      if (mapper_data != NULL)
      {
        free(mapper_data);
        mapper_data = NULL;
        mapper_data_size = 0;
      }
    }

    //--------------------------------------------------------------------------
    void CopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_copy(); 
    }

    //--------------------------------------------------------------------------
    void CopyOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_copy(); 
      // Return this operation to the runtime
      runtime->free_copy_op(this);
    }

    //--------------------------------------------------------------------------
    const char* CopyOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[COPY_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind CopyOp::get_operation_kind(void) const
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
    void CopyOp::log_copy_requirements(void) const
    //--------------------------------------------------------------------------
    {
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

    //--------------------------------------------------------------------------
    void CopyOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent indexes
      compute_parent_indexes();
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
        log_copy_requirements();
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Register a dependence on our predicate
      register_predicate_dependence();
      ProjectionInfo projection_info;
      src_versions.resize(src_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        runtime->forest->perform_dependence_analysis(this, idx, 
                                                     src_requirements[idx],
                                                     src_restrict_infos[idx],
                                                     src_versions[idx],
                                                     projection_info,
                                                     src_privilege_paths[idx]);
      dst_versions.resize(dst_requirements.size());
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        unsigned index = src_requirements.size()+idx;
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        if (is_reduce_req)
          dst_requirements[idx].privilege = READ_WRITE;
        runtime->forest->perform_dependence_analysis(this, index, 
                                                     dst_requirements[idx],
                                                     dst_restrict_infos[idx],
                                                     dst_versions[idx],
                                                     projection_info,
                                                     dst_privilege_paths[idx]);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = REDUCE;
      }
    }

    //--------------------------------------------------------------------------
    bool CopyOp::query_speculate(bool &value, bool &mapping_only)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      Mapper::SpeculativeOutput output;
      output.speculate = false;
      output.speculate_mapping_only = true;
      mapper->invoke_copy_speculate(this, &output);
      if (!output.speculate)
        return false;
      value = output.speculative_value;
      mapping_only = output.speculate_mapping_only;
      // Make our predicate guard
#ifdef DEBUG_LEGION
      assert(!predication_guard.exists());
#endif
      // Make the copy across precondition guard 
      predication_guard = predicate->get_true_guard();
      // If we're speculating then we make all the destination
      // privileges that are write-discard read-write instead so
      // that we get the earlier version of the data in case we
      // actually are predicated false
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        RegionRequirement &req = dst_requirements[idx];
        if (IS_WRITE_ONLY(req))
          req.privilege = READ_WRITE;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void CopyOp::resolve_true(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void CopyOp::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // If we already launched then we are done
      if (launched)
        return;
      // Otherwise we need to do the things to clean up this operation
      // Mark that this operation has completed both
      // execution and mapping indicating that we are done
      // Do it in this order to avoid calling 'execute_trigger'
      complete_execution();
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      resolve_speculation();
    } 

    //--------------------------------------------------------------------------
    void CopyOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Do our versioning analysis and then add it to the ready queue
      std::set<RtEvent> preconditions;
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        runtime->forest->perform_versioning_analysis(this, idx,
                                                     src_requirements[idx],
                                                     src_privilege_paths[idx],
                                                     src_versions[idx],
                                                     preconditions);
      const unsigned offset = src_requirements.size();
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        if (is_reduce_req)
          dst_requirements[idx].privilege = READ_WRITE;
        runtime->forest->perform_versioning_analysis(this, offset + idx,
                                                     dst_requirements[idx],
                                                     dst_privilege_paths[idx],
                                                     dst_versions[idx],
                                                     preconditions);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = REDUCE;
      }
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      std::vector<InstanceSet> valid_src_instances(src_requirements.size());
      std::vector<InstanceSet> valid_dst_instances(dst_requirements.size());
      Mapper::MapCopyInput input;
      Mapper::MapCopyOutput output;
      input.src_instances.resize(src_requirements.size());
      input.dst_instances.resize(dst_requirements.size());
      output.src_instances.resize(src_requirements.size());
      output.dst_instances.resize(dst_requirements.size());
      output.profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      // First go through and do the traversals to find the valid instances
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        InstanceSet &valid_instances = valid_src_instances[idx];
        runtime->forest->physical_premap_only(this, idx, 
                                              src_requirements[idx],
                                              src_versions[idx],
                                              valid_instances);
        // Convert these to the valid set of mapping instances
        // No need to filter for copies
        if (src_restrict_infos[idx].has_restrictions())
          prepare_for_mapping(src_restrict_infos[idx].get_instances(), 
                              input.src_instances[idx]);
        else
          prepare_for_mapping(valid_instances, input.src_instances[idx]);
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        InstanceSet &valid_instances = valid_dst_instances[idx];
        runtime->forest->physical_premap_only(this, idx+src_requirements.size(),
                                              dst_requirements[idx],
                                              dst_versions[idx],
                                              valid_instances);
        // No need to filter for copies
        if (dst_restrict_infos[idx].has_restrictions())
        {
          prepare_for_mapping(dst_restrict_infos[idx].get_instances(), 
                              input.dst_instances[idx]);
          assert(!input.dst_instances[idx].empty());
        }
        else
          prepare_for_mapping(valid_instances, input.dst_instances[idx]);
      }
      // Now we can ask the mapper what to do
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_map_copy(this, &input, &output);
      if (!output.profiling_requests.empty())
      {
        filter_copy_request_kinds(mapper,
            output.profiling_requests.requested_measurements,
            profiling_requests, true/*warn*/);
        profiling_priority = output.profiling_priority;
      }
      // Now we can carry out the mapping requested by the mapper
      // and issue the across copies, first set up the sync precondition
      ApEvent sync_precondition;
      if (!wait_barriers.empty() || !grants.empty())
      {
        std::set<ApEvent> preconditions;
        for (std::vector<PhaseBarrier>::const_iterator it = 
              wait_barriers.begin(); it != wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(*it); 
          preconditions.insert(e);
          if (Runtime::legion_spy_enabled)
            LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
        for (std::vector<Grant>::const_iterator it = grants.begin();
              it != grants.end(); it++)
        {
          ApEvent e = it->impl->acquire_grant();
          preconditions.insert(e);
        }
        if (sync_precondition.exists())
          preconditions.insert(sync_precondition);
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
          // Clear out the target views, the copy_across call will
          // find the proper valid views
          src_targets.clear();
        }
        else
        {
          // Now do the registration
          set_mapping_state(idx, true/*src*/);
          runtime->forest->physical_register_only(src_requirements[idx],
                                                  src_versions[idx],
                                                  src_restrict_infos[idx],
                                                  this, idx, local_completion,
                                                  false/*defer add users*/,
                                                  true/*read only locks*/,
                                                  map_applied_conditions,
                                                  src_targets,
                                                  get_projection_info(idx, true)
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
        perform_conversion<false/*src*/>(idx, dst_requirements[idx],
                                         output.dst_instances[idx],
                                         dst_targets);
        // Now do the registration
        set_mapping_state(idx, false/*src*/);
        runtime->forest->physical_register_only(dst_requirements[idx],
                                                dst_versions[idx],
                                                dst_restrict_infos[idx],
                                                this, 
                                                idx + src_requirements.size(),
                                                local_completion,
                                                false/*defer add users*/,
                                                false/*not read only*/,
                                                map_applied_conditions,
                                                dst_targets,
                                                get_projection_info(idx, false)
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
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
            runtime->forest->copy_across( 
                                  src_requirements[idx], dst_requirements[idx],
                                  src_targets, dst_targets, 
                                  src_versions[idx], dst_versions[idx], 
                                  local_completion, this, idx,
                                  idx + src_requirements.size(),
                                  local_sync_precondition, predication_guard, 
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
            runtime->forest->reduce_across(
                                  src_requirements[idx], dst_requirements[idx],
                                  src_targets, dst_targets, this, 
                                  local_sync_precondition, predication_guard);
          Runtime::trigger_event(local_completion, across_done);
        }
        // Apply our changes to the version states
        // Don't apply changes to the source if we have a composite instance
        // because it is unsound to mutate the region tree that way
        if (src_composite == -1)
          src_versions[idx].apply_mapping(map_applied_conditions);
        dst_versions[idx].apply_mapping(map_applied_conditions); 
      }
      ApEvent copy_complete_event = Runtime::merge_events(copy_complete_events);
      if (!restrict_postconditions.empty())
      {
        restrict_postconditions.insert(copy_complete_event);
        copy_complete_event = Runtime::merge_events(restrict_postconditions);
      }
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
          if (Runtime::legion_spy_enabled)
            LegionSpy::log_phase_barrier_arrival(unique_op_id, 
                                                 it->phase_barrier);
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        completion_event);    
        }
      }
      // Remove our profiling guard and trigger the profiling event if necessary
      if ((__sync_add_and_fetch(&outstanding_profiling_requests, -1) == 0) &&
          profiling_reported.exists())
        Runtime::trigger_event(profiling_reported);
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
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      for (std::vector<VersionInfo>::iterator it = src_versions.begin();
            it != src_versions.end(); it++)
      {
        it->clear();
      }
      for (std::vector<VersionInfo>::iterator it = dst_versions.begin();
            it != dst_versions.end(); it++)
      {
        it->clear();
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
      REPORT_LEGION_ERROR(ERROR_ALIASED_REQION_REQUIREMENTS,
                    "Aliased region requirements for copy operations "
                    "are not permitted. Region requirement %d of %s "
                    "requirements and %d of %s requirements interfering for "
                    "copy operation (UID %lld) in task %s (UID %lld).",
                    actual_idx1, is_src1 ? "source" : "destination",
                    actual_idx2, is_src2 ? "source" : "destination",
                    unique_op_id, parent_ctx->get_task_name(),
                    parent_ctx->get_unique_id())
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
    PhysicalManager* CopyOp::select_temporary_instance(PhysicalManager *dst,
                                 unsigned index, const FieldMask &needed_fields)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      Mapper::CreateCopyTemporaryInput input;
      Mapper::CreateCopyTemporaryOutput output;
      input.region_requirement_index = index;
      input.src_requirement = current_src;
      input.destination_instance = MappingInstance(dst);
      if (!Runtime::unsafe_mapper)
      {
        // Fields and regions must both be met
        // The instance must be freshly created
        // Instance must be acquired
        std::set<PhysicalManager*> previous_managers;
        // Get the set of previous managers we've made
        for (std::map<PhysicalManager*,std::pair<unsigned,bool> >::
              const_iterator it = acquired_instances.begin(); it !=
              acquired_instances.end(); it++)
          previous_managers.insert(it->first);
        mapper->invoke_copy_create_temporary(this, &input, &output);
        validate_temporary_instance(output.temporary_instance.impl,
            previous_managers, acquired_instances, needed_fields,
            current_src ? src_requirements[index].region :
                          dst_requirements[index].region, mapper,
            "create_copy_temporary_instance");
      }
      else
        mapper->invoke_copy_create_temporary(this, &input, &output);
      if (Runtime::legion_spy_enabled)
        log_temporary_instance(output.temporary_instance.impl, 
                               index, needed_fields);
      return output.temporary_instance.impl;
    }

    //--------------------------------------------------------------------------
    ApEvent CopyOp::get_restrict_precondition(void) const
    //--------------------------------------------------------------------------
    {
      return merge_restrict_preconditions(grants, wait_barriers);
    }

    //--------------------------------------------------------------------------
    void CopyOp::record_restrict_postcondition(ApEvent postcondition)
    //--------------------------------------------------------------------------
    {
      restrict_postconditions.insert(postcondition);
    }

    //--------------------------------------------------------------------------
    UniqueID CopyOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id; 
    }

    //--------------------------------------------------------------------------
    unsigned CopyOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    int CopyOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const ProjectionInfo* CopyOp::get_projection_info(unsigned idx, bool src)
    //--------------------------------------------------------------------------
    {
      // No advance projection epochs for normal copy operations
      return NULL;
    }

    //--------------------------------------------------------------------------
    void CopyOp::check_copy_privilege(const RegionRequirement &requirement, 
                                      unsigned idx, bool src, bool permit_proj)
    //--------------------------------------------------------------------------
    {
      if (!permit_proj && ((requirement.handle_type == PART_PROJECTION) ||
          (requirement.handle_type == REG_PROJECTION)))
        REPORT_LEGION_ERROR(ERROR_PROJECTION_REGION_REQUIREMENTS,
                         "Projection region requirements are not "
                               "permitted for copy operations (in task %s)",
                               parent_ctx->get_task_name())
      FieldID bad_field = AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, then check the privileges with the parent context
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, bad_index);
      switch (et)
      {
        case NO_ERROR:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            REPORT_LEGION_ERROR(ERROR_REQUEST_INVALID_REGION,
                             "Requirements for invalid region handle "
                             "(%x,%d,%d) for index %d of %s "
                             "requirements of copy operation (ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             idx, (src ? "source" : "destination"),
                             unique_op_id)
            break;
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) ||
            (requirement.handle_type == REG_PROJECTION)
            ? requirement.region.field_space :
            requirement.partition.field_space;
            REPORT_LEGION_ERROR(ERROR_FIELD_NOT_VALID,
                             "Field %d is not a valid field of field "
                             "space %d for index %d of %s requirements "
                             "of copy operation (ID %lld)",
                             bad_field, sp.id, idx,
                             (src ? "source" : "destination"),
                             unique_op_id)
            break;
          }
        case ERROR_INVALID_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_PRIVILEGE,
                             "Instance field %d is not one of the "
                             "privilege fields for index %d of %s "
                             "requirements of copy operation (ID %lld)",
                             bad_field, idx,
                             (src ? "source" : "destination"),
                             unique_op_id)
            break;
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_DUPLICATE,
                             "Instance field %d is a duplicate for "
                             "index %d of %s requirements of copy "
                             "operation (ID %lld)",
                             bad_field, idx,
                             (src ? "source" : "destination"),
                             unique_op_id)
            break;
          }
        case ERROR_BAD_PARENT_REGION:
          {
            if (bad_index < 0) 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_COPY,
                               "Parent task %s (ID %lld) of copy operation "
                               "(ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) "
                               "as a parent of index %d of %s region "
                               "requirements because there was no "
                               "'parent' region had that name.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id,
                               idx, (src ? "source" : "destination"))
            else if (bad_field == AUTO_GENERATE_ID) 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_COPY,
                               "Parent task %s (ID %lld) of copy operation "
                               "(ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) "
                               "as a parent of index %d of %s region "
                               "requirements because parent requirement %d "
                               "did not have sufficient privileges.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id,
                               idx, (src ? "source" : "destination"),
                               bad_index)
            else 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_COPY,
                               "Parent task %s (ID %lld) of copy operation "
                               "(ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) "
                               "as a parent of index %d of %s region "
                               "requirements because region requirement %d "
                               "was missing field %d.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id,
                               idx, (src ? "source" : "destination"),
                               bad_index, bad_field)
            break;
          }
        case ERROR_BAD_REGION_PATH:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_NOT_SUBREGION,
                             "Region (%x,%x,%x) is not a "
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
                             unique_op_id)
            break;
          }
        case ERROR_BAD_REGION_TYPE:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_REQUIREMENT_COPY,
                             "Region requirement of copy operation "
                             "(ID %lld) cannot find privileges for field "
                             "%d in parent task from index %d of %s "
                             "region requirements",
                             unique_op_id, bad_field, idx,
                             (src ? "source" : "destination"))
            break;
          }
        case ERROR_BAD_REGION_PRIVILEGES:
          {
            REPORT_LEGION_ERROR(ERROR_PRIVILEGES_FOR_REGION,
                             "Privileges %x for region (%x,%x,%x) are "
                             "not a subset of privileges of parent "
                             "task's privileges for index %d of %s "
                             "region requirements for copy "
                             "operation (ID %lld)",
                             requirement.privilege,
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             idx, (src ? "source" : "destination"),
                             unique_op_id)
            break;
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
          REPORT_LEGION_ERROR(ERROR_PARENT_TASK_COPY,
                           "Parent task %s (ID %lld) of copy operation "
                                   "(ID %lld) does not have a region "
                                   "requirement for region (%x,%x,%x) "
                                   "as a parent of index %d of source region "
                                   "requirements",
                                   parent_ctx->get_task_name(), 
                                   parent_ctx->get_unique_id(),
                                   unique_op_id, 
                                   src_requirements[idx].region.index_space.id,
                                   src_requirements[idx].region.field_space.id, 
                                   src_requirements[idx].region.tree_id, idx)
        else
          src_parent_indexes[idx] = unsigned(parent_index);
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        int parent_index = 
          parent_ctx->find_parent_region_req(dst_requirements[idx]);
        if (parent_index < 0)
          REPORT_LEGION_ERROR(ERROR_PARENT_TASK_COPY,
                           "Parent task %s (ID %lld) of copy operation "
                                   "(ID %lld) does not have a region "
                                   "requirement for region (%x,%x,%x) "
                                   "as a parent of index %d of destination "
                                   "region requirements",
                                   parent_ctx->get_task_name(), 
                                   parent_ctx->get_unique_id(),
                                   unique_op_id, 
                                   dst_requirements[idx].region.index_space.id,
                                   dst_requirements[idx].region.field_space.id, 
                                   dst_requirements[idx].region.tree_id, idx)
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
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'map_copy' "
                      "on mapper %s. Mapper selected an instance from "
                      "region tree %d to satisfy %s region requirement %d "
                      "for explicit region-to_region copy in task %s (ID %lld) "
                      "but the logical region for this requirement is from "
                      "region tree %d.", mapper->get_mapper_name(), bad_tree,
                      IS_SRC ? "source" : "destination", idx, 
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id(),
                      req.region.get_tree_id())
      if (!missing_fields.empty())
      {
        for (std::vector<FieldID>::const_iterator it = missing_fields.begin();
              it != missing_fields.end(); it++)
        {
          const void *name; size_t name_size;
          if (!runtime->retrieve_semantic_information(
               req.region.get_field_space(), *it, NAME_SEMANTIC_TAG,
               name, name_size, true, false))
            name = "(no name)";
          log_run.error("Missing instance for field %s (FieldID: %d)",
                        static_cast<const char*>(name), *it);
        }
        REPORT_LEGION_ERROR(ERROR_MISSING_INSTANCE_FIELD,
                      "Invalid mapper output from invocation of 'map_copy' "
                      "on mapper %s. Mapper failed to specify a physical "
                      "instance for %zd fields of the region requirement %d "
                      "of explicit region-to-region copy in task %s (ID %lld). "
                      "The missing fields are listed below.",
                      mapper->get_mapper_name(), missing_fields.size(), idx,
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id())
      }
      if (!unacquired.empty())
      {
        for (std::vector<PhysicalManager*>::const_iterator it = 
              unacquired.begin(); it != unacquired.end(); it++)
        {
          if (acquired_instances.find(*it) == acquired_instances.end())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from 'map_copy' invocation "
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
                          parent_ctx->get_unique_id())
        }
        // If we did successfully acquire them, still issue the warning
        REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_FAILED_ACQUIRE,
                        "mapper %s failed to acquire instances "
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
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'map_copy' "
                      "on mapper %s. Mapper requested the creation of a "
                      "composite instance for destination region requiremnt "
                      "%d. Only source region requirements are permitted to "
                      "be composite instances for explicit region-to-region "
                      "copy operations. Operation was issued in task %s "
                      "(ID %lld).", mapper->get_mapper_name(), idx,
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id())
      if (IS_SRC && (composite_idx >= 0) && is_reduce)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'map_copy' "
                      "on mapper %s. Mapper requested the creation of a "
                      "composite instance for the source requirement %d of "
                      "an explicit region-to-region reduction. Only real "
                      "physical instances are permitted to be sources of "
                      "explicit region-to-region reductions. Operation was "
                      "issued in task %s (ID %lld).", mapper->get_mapper_name(),
                      idx, parent_ctx->get_task_name(), 
                      parent_ctx->get_unique_id())
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
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'map_copy' "
                        "on mapper %s. Mapper specified an instance for %s "
                        "region requirement at index %d that does not meet "
                        "the logical region requirement. The copy operation "
                        "was issued in task %s (ID %lld).",
                        mapper->get_mapper_name(), 
                        IS_SRC ? "source" : "destination", idx,
                        parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
      }
      // Make sure all the destinations are real instances, this has
      // to be true for all kinds of explicit copies including reductions
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        if (IS_SRC && (int(idx) == composite_idx))
          continue;
        if (!targets[idx].get_manager()->is_instance_manager())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'map_copy' "
                        "on mapper %s. Mapper specified an illegal "
                        "specialized instance as the target for %s "
                        "region requirement %d of an explicit copy operation "
                        "in task %s (ID %lld).", mapper->get_mapper_name(),
                        IS_SRC ? "source" : "destination", idx, 
                        parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id())
      }
      return composite_idx;
    }

    //--------------------------------------------------------------------------
    void CopyOp::add_copy_profiling_request(
                                           Realm::ProfilingRequestSet &requests)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      ProfilingResponseBase base(this);
      Realm::ProfilingRequest &request = requests.add_request( 
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
          &base, sizeof(base), profiling_priority);
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      int previous = __sync_fetch_and_add(&outstanding_profiling_requests, 1);
      if ((previous == 1) && !profiling_reported.exists())
        profiling_reported = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    void CopyOp::handle_profiling_response(
                                       const Realm::ProfilingResponse &response)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapper != NULL);
#endif
      Mapping::Mapper::CopyProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      mapper->invoke_copy_report_profiling(this, &info);
#ifdef DEBUG_LEGION
      assert(outstanding_profiling_requests > 0);
      assert(profiling_reported.exists());
#endif
      int remaining = __sync_add_and_fetch(&outstanding_profiling_requests, -1);
      // If we're the last one then we trigger the result
      if (remaining == 0)
        Runtime::trigger_event(profiling_reported);
    }

    /////////////////////////////////////////////////////////////
    // Index Copy Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexCopyOp::IndexCopyOp(Runtime *rt)
      : CopyOp(rt)
    //--------------------------------------------------------------------------
    {
      this->is_index_space = true;
    }

    //--------------------------------------------------------------------------
    IndexCopyOp::IndexCopyOp(const IndexCopyOp &rhs)
      : CopyOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndexCopyOp::~IndexCopyOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexCopyOp& IndexCopyOp::operator=(const IndexCopyOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::initialize(TaskContext *ctx, 
                                 const IndexCopyLauncher &launcher,
                                 IndexSpace launch_sp, bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_speculation(ctx, true/*track*/, 
                             launcher.src_requirements.size() + 
                               launcher.dst_requirements.size(), 
                             launcher.static_dependences,
                             launcher.predicate);
#ifdef DEBUG_LEGION
      assert(launch_sp.exists());
#endif
      launch_space = launch_sp;
      if (!launcher.launch_domain.exists())
        runtime->forest->find_launch_space_domain(launch_space, index_domain);
      else
        index_domain = launcher.launch_domain;
      src_requirements.resize(launcher.src_requirements.size());
      dst_requirements.resize(launcher.dst_requirements.size());
      src_versions.resize(launcher.src_requirements.size());
      dst_versions.resize(launcher.dst_requirements.size());
      src_restrict_infos.resize(launcher.src_requirements.size());
      dst_restrict_infos.resize(launcher.dst_requirements.size());
      src_projection_infos.resize(launcher.src_requirements.size());
      dst_projection_infos.resize(launcher.dst_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        if (launcher.src_requirements[idx].privilege_fields.empty())
        {
          REPORT_LEGION_WARNING(ERROR_SOURCE_REGION_REQUIREMENT,
                           "SOURCE REGION REQUIREMENT %d OF "
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
          REPORT_LEGION_WARNING(ERROR_DESTINATION_REGION_REQUIREMENT,
                           "DESTINATION REGION REQUIREMENT %d OF"
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
          REPORT_LEGION_ERROR(ERROR_NUMBER_SOURCE_REQUIREMENTS,
                        "Number of source requirements (%zd) does not "
                        "match number of destination requirements (%zd) "
                        "for copy operation (ID %lld) with parent "
                        "task %s (ID %lld)",
                        src_requirements.size(), dst_requirements.size(),
                        get_unique_id(), parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        {
          if (src_requirements[idx].privilege_fields.size() != 
              src_requirements[idx].instance_fields.size())
            REPORT_LEGION_ERROR(ERROR_COPY_SOURCE_REQUIREMENT,
                          "Copy source requirement %d for copy operation "
                          "(ID %lld) in parent task %s (ID %lld) has %zd "
                          "privilege fields and %zd instance fields.  "
                          "Copy requirements must have exactly the same "
                          "number of privilege and instance fields.",
                          idx, get_unique_id(), 
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id(),
                          src_requirements[idx].privilege_fields.size(),
                          src_requirements[idx].instance_fields.size())
          if (!IS_READ_ONLY(src_requirements[idx]))
            REPORT_LEGION_ERROR(ERROR_COPY_SOURCE_REQUIREMENT,
                          "Copy source requirement %d for copy operation "
                          "(ID %lld) in parent task %s (ID %lld) must "
                          "be requested with a read-only privilege.",
                          idx, get_unique_id(),
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id())
          check_copy_privilege(src_requirements[idx], idx, 
                               true/*src*/, true/*permit projection*/);
        }
        for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
        {
          if (dst_requirements[idx].privilege_fields.size() != 
              dst_requirements[idx].instance_fields.size())
            REPORT_LEGION_ERROR(ERROR_COPY_DESTINATION_REQUIREMENT,
                          "Copy destination requirement %d for copy "
                          "operation (ID %lld) in parent task %s "
                          "(ID %lld) has %zd privilege fields and %zd "
                          "instance fields.  Copy requirements must "
                          "have exactly the same number of privilege "
                          "and instance fields.", idx, 
                          get_unique_id(), 
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id(),
                          dst_requirements[idx].privilege_fields.size(),
                          dst_requirements[idx].instance_fields.size())
          if (!HAS_WRITE(dst_requirements[idx]))
            REPORT_LEGION_ERROR(ERROR_COPY_DESTINATION_REQUIREMENT,
                          "Copy destination requirement %d for copy "
                          "operation (ID %lld) in parent task %s "
                          "(ID %lld) must be requested with a "
                          "read-write or write-discard privilege.",
                          idx, get_unique_id(),
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id())
          check_copy_privilege(dst_requirements[idx], idx, 
                               false/*src*/, true/*permit projection*/);
        }
      }
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_copy_operation(parent_ctx->get_unique_id(),
                                      unique_op_id);
        runtime->forest->log_launch_space(launch_space, unique_op_id);
      }
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_copy();
      index_domain = Domain::NO_DOMAIN;
      launch_space = IndexSpace::NO_SPACE;
      points_committed = 0;
      commit_request = false;
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_copy();
      src_projection_infos.clear();
      dst_projection_infos.clear();
      // We can deactivate all of our point operations
      for (std::vector<PointCopyOp*>::const_iterator it = points.begin();
            it != points.end(); it++)
        (*it)->deactivate();
      points.clear();
      commit_preconditions.clear();
      // Return this operation to the runtime
      runtime->free_index_copy_op(this);
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent indexes
      compute_parent_indexes();
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
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        {
          const RegionRequirement &req = src_requirements[idx];
          const bool reg = (req.handle_type == SINGULAR) ||
                           (req.handle_type == REG_PROJECTION);
          const bool proj = (req.handle_type == REG_PROJECTION) ||
                            (req.handle_type == PART_PROJECTION); 

          LegionSpy::log_logical_requirement(unique_op_id, idx, reg,
              reg ? req.region.index_space.id :
                    req.partition.index_partition.id,
              reg ? req.region.field_space.id :
                    req.partition.field_space.id,
              reg ? req.region.tree_id : 
                    req.partition.tree_id,
              req.privilege, req.prop, req.redop, req.parent.index_space.id);
          LegionSpy::log_requirement_fields(unique_op_id, idx, 
                                            req.instance_fields);
          if (proj)
            LegionSpy::log_requirement_projection(unique_op_id, idx, 
                                                  req.projection);
        }
        for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
        {
          const RegionRequirement &req = dst_requirements[idx];
          const bool reg = (req.handle_type == SINGULAR) ||
                           (req.handle_type == REG_PROJECTION);
          const bool proj = (req.handle_type == REG_PROJECTION) ||
                            (req.handle_type == PART_PROJECTION); 

          LegionSpy::log_logical_requirement(unique_op_id, 
              src_requirements.size() + idx, reg,
              reg ? req.region.index_space.id :
                    req.partition.index_partition.id,
              reg ? req.region.field_space.id :
                    req.partition.field_space.id,
              reg ? req.region.tree_id : 
                    req.partition.tree_id,
              req.privilege, req.prop, req.redop, req.parent.index_space.id);
          LegionSpy::log_requirement_fields(unique_op_id, 
                                            src_requirements.size()+idx, 
                                            req.instance_fields);
          if (proj)
            LegionSpy::log_requirement_projection(unique_op_id,
                src_requirements.size() + idx, req.projection);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Register a dependence on our predicate
      register_predicate_dependence();
      src_versions.resize(src_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        src_projection_infos[idx] = 
          ProjectionInfo(runtime, src_requirements[idx], launch_space);
        runtime->forest->perform_dependence_analysis(this, idx, 
                                                     src_requirements[idx],
                                                     src_restrict_infos[idx],
                                                     src_versions[idx],
                                                     src_projection_infos[idx],
                                                     src_privilege_paths[idx]);
      }
      dst_versions.resize(dst_requirements.size());
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        dst_projection_infos[idx] = 
          ProjectionInfo(runtime, dst_requirements[idx], launch_space);
        unsigned index = src_requirements.size()+idx;
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        if (is_reduce_req)
          dst_requirements[idx].privilege = READ_WRITE;
        runtime->forest->perform_dependence_analysis(this, index, 
                                                     dst_requirements[idx],
                                                     dst_restrict_infos[idx],
                                                     dst_versions[idx],
                                                     dst_projection_infos[idx],
                                                     dst_privilege_paths[idx]);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = REDUCE;
      }
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Do the upper bound version analysis first
      std::set<RtEvent> preconditions;
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        VersionInfo &version_info = src_versions[idx];
        // If we already have physical state for it then we've 
        // done this before so there is no need to do it again
        if (version_info.has_physical_states())
          continue;
        ProjectionInfo &proj_info = src_projection_infos[idx];
        const bool partial_traversal = 
          (proj_info.projection_type == PART_PROJECTION) ||
          ((proj_info.projection_type != SINGULAR) && 
           (proj_info.projection->depth > 0));
        runtime->forest->perform_versioning_analysis(this, idx, 
                                                     src_requirements[idx],
                                                     src_privilege_paths[idx],
                                                     version_info,
                                                     preconditions,
                                                     partial_traversal);
      }
      const unsigned offset = src_requirements.size();
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        VersionInfo &version_info = dst_versions[idx];
        // If we already have physical state for it then we've 
        // done this before so there is no need to do it again
        if (version_info.has_physical_states())
          continue;
        ProjectionInfo &proj_info = dst_projection_infos[idx];
        const bool partial_traversal = 
          (proj_info.projection_type == PART_PROJECTION) ||
          ((proj_info.projection_type != SINGULAR) && 
           (proj_info.projection->depth > 0));
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        if (is_reduce_req)
          dst_requirements[idx].privilege = READ_WRITE;
        runtime->forest->perform_versioning_analysis(this, offset + idx,
                                                     dst_requirements[idx],
                                                     dst_privilege_paths[idx],
                                                     version_info,
                                                     preconditions,
                                                     partial_traversal);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = REDUCE;
      }
      // Now enumerate the points
      size_t num_points = index_domain.get_volume();
#ifdef DEBUG_LEGION
      assert(num_points > 0);
#endif
      unsigned point_idx = 0;
      points.resize(num_points);
      for (Domain::DomainPointIterator itr(index_domain); 
            itr; itr++, point_idx++)
      {
        PointCopyOp *point = runtime->get_available_point_copy_op();
        point->initialize(this, itr.p);
        points[point_idx] = point;
      }
      // Perform the projections
      std::vector<ProjectionPoint*> projection_points(points.begin(),
                                                      points.end());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        if (src_requirements[idx].handle_type == SINGULAR)
          continue;
        ProjectionFunction *function = 
          runtime->find_projection_function(src_requirements[idx].projection);
        function->project_points(this, idx, src_requirements[idx],
                                 runtime, projection_points);
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        if (dst_requirements[idx].handle_type == SINGULAR)
          continue;
        ProjectionFunction *function = 
          runtime->find_projection_function(dst_requirements[idx].projection);
        function->project_points(this, src_requirements.size() + idx, 
                                 dst_requirements[idx], runtime, 
                                 projection_points);
      }
#ifdef DEBUG_LEGION
      // Check for interfering point requirements in debug mode
      check_point_requirements();
      // Also check to make sure source requirements dominate
      // the destination requirements for each point
      for (std::vector<PointCopyOp*>::const_iterator it = points.begin();
            it != points.end(); it++)
        (*it)->check_domination();
#endif
      if (Runtime::legion_spy_enabled)
      {
        for (std::vector<PointCopyOp*>::const_iterator it = points.begin();
              it != points.end(); it++) 
          (*it)->log_copy_requirements();
      }
      // Launch the points
      std::set<RtEvent> mapped_preconditions;
      std::set<ApEvent> executed_preconditions;
      for (std::vector<PointCopyOp*>::const_iterator it = points.begin();
            it != points.end(); it++)
      {
        mapped_preconditions.insert((*it)->get_mapped_event());
        executed_preconditions.insert((*it)->get_completion_event());
        (*it)->launch(preconditions);
      }
#ifdef LEGION_SPY
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      completion_event);
#endif
      // Record that we are mapped when all our points are mapped
      // and we are executed when all our points are executed
      complete_mapping(Runtime::merge_events(mapped_preconditions));
      ApEvent done = Runtime::merge_events(executed_preconditions);
      Runtime::trigger_event(completion_event, done); 
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(done));
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // This should never be called as this operation doesn't
      // go through the rest of the queue normally
      assert(false);
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      bool commit_now = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(!commit_request);
#endif
        commit_request = true;
        commit_now = (points.size() == points_committed);
      }
      if (commit_now)
        commit_operation(true/*deactivate*/, 
                          Runtime::merge_events(commit_preconditions));
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::handle_point_commit(RtEvent point_committed)
    //--------------------------------------------------------------------------
    {
      bool commit_now = false;
      RtEvent commit_pre;
      {
        AutoLock o_lock(op_lock);
        points_committed++;
        if (point_committed.exists())
          commit_preconditions.insert(point_committed);
        commit_now = commit_request && (points.size() == points_committed);
      }
      if (commit_now)
        commit_operation(true/*deactivate*/,
                          Runtime::merge_events(commit_preconditions));
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::report_interfering_requirements(unsigned idx1,
                                                      unsigned idx2)
    //--------------------------------------------------------------------------
    {
      bool is_src1 = idx1 < src_requirements.size();
      bool is_src2 = idx2 < src_requirements.size();
      unsigned actual_idx1 = is_src1 ? idx1 : (idx1 - src_requirements.size());
      unsigned actual_idx2 = is_src2 ? idx2 : (idx2 - src_requirements.size());
      REPORT_LEGION_WARNING(LEGION_WARNING_REGION_REQUIREMENTS_INDEX,
                      "Region requirements %d and %d of index copy %lld in "
                      "parent task %s (UID %lld) are potentially interfering. "
                      "It's possible that this is a false positive if there "
                      "are projection region requirements and each of the "
                      "point copies are non-interfering. If the runtime is "
                      "built in debug mode then it will check that the region "
                      "requirements of all points are actually "
                      "non-interfering. If you see no further error messages "
                      "for this index task launch then everything is good.",
                      actual_idx1, actual_idx2, unique_op_id, 
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
      interfering_requirements.insert(std::pair<unsigned,unsigned>(idx1,idx2));
#endif
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    void IndexCopyOp::check_point_requirements(void)
    //--------------------------------------------------------------------------
    {
      // Handle any region requirements which can interfere with itself
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        if (!IS_WRITE(dst_requirements[idx]))
          continue;
        const unsigned index = src_requirements.size() + idx;
        interfering_requirements.insert(
            std::pair<unsigned,unsigned>(index,index));
      }
      // Nothing to do if there are no interfering requirements
      if (interfering_requirements.empty())
        return;
      std::map<DomainPoint,std::vector<LogicalRegion> > point_requirements;
      for (std::vector<PointCopyOp*>::const_iterator pit = points.begin();
            pit != points.end(); pit++)
      {
        const DomainPoint &current_point = (*pit)->get_domain_point();
        std::vector<LogicalRegion> &point_reqs = 
          point_requirements[current_point];
        point_reqs.resize(src_requirements.size() + dst_requirements.size());
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
          point_reqs[idx] = (*pit)->src_requirements[idx].region;
        for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
          point_reqs[src_requirements.size() + idx] = 
            (*pit)->dst_requirements[idx].region;
        // Check against all the prior points
        for (std::map<DomainPoint,std::vector<LogicalRegion> >::const_iterator
              oit = point_requirements.begin(); 
              oit != point_requirements.end(); oit++)
        {
          const bool same_point = (current_point == oit->first);
          const std::vector<LogicalRegion> &other_reqs = oit->second;
          // Now check for interference with any other points
          for (std::set<std::pair<unsigned,unsigned> >::const_iterator it =
                interfering_requirements.begin(); it !=
                interfering_requirements.end(); it++)
          {
            // Can skip comparing against ourself
            if (same_point && (it->first == it->second))
              continue;
            if (!runtime->forest->are_disjoint(
                  point_reqs[it->first].get_index_space(), 
                  other_reqs[it->second].get_index_space()))
            {
              if (current_point.get_dim() <= 1) {
                REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_COPY,
                              "Index space copy launch has intefering "
                              "region requirements %d of point %lld and region "
                              "requirement %d of point %lld of %s (UID %lld) "
                              "in parent task %s (UID %lld) are interfering.",
                              it->first, current_point[0], it->second,
                              oit->first[0], get_logging_name(),
                              get_unique_id(), parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
              } else if (current_point.get_dim() == 2) {
                REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_COPY,
                              "Index space copy launch has intefering "
                              "region requirements %d of point (%lld,%lld) and "
                              "region requirement %d of point (%lld,%lld) of "
                              "%s (UID %lld) in parent task %s (UID %lld) are "
                              "interfering.", it->first, current_point[0],
                              current_point[1], it->second, oit->first[0],
                              oit->first[1], get_logging_name(),
                              get_unique_id(), parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
              } else if (current_point.get_dim() == 3) {
                REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_COPY,
                              "Index space copy launch has intefering "
                              "region requirements %d of point (%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld) of %s (UID %lld) in parent "
                              "task %s (UID %lld) are interfering.", it->first,
                              current_point[0], current_point[1],
                              current_point[2], it->second, oit->first[0],
                              oit->first[1], oit->first[2], get_logging_name(),
                              get_unique_id(), parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
              }
              assert(false);
            }
          }
        }
      }
    }
#endif

    //--------------------------------------------------------------------------
    const ProjectionInfo* IndexCopyOp::get_projection_info(unsigned idx, 
                                                           bool src)
    //--------------------------------------------------------------------------
    {
      if (src)
      {
#ifdef DEBUG_LEGION
        assert(idx < src_projection_infos.size());
#endif
        return &src_projection_infos[idx];
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(idx < dst_projection_infos.size());
#endif
        return &dst_projection_infos[idx]; 
      }
    }

    /////////////////////////////////////////////////////////////
    // Point Copy Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointCopyOp::PointCopyOp(Runtime *rt)
      : CopyOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointCopyOp::PointCopyOp(const PointCopyOp &rhs)
      : CopyOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PointCopyOp::~PointCopyOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointCopyOp& PointCopyOp::operator=(const PointCopyOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::initialize(IndexCopyOp *own, const DomainPoint &p)
    //--------------------------------------------------------------------------
    {
      // Initialize the operation
      initialize_operation(own->get_context(), false/*track*/, 
          own->src_requirements.size() + own->dst_requirements.size());
      index_point = p;
      owner = own;
      execution_fence_event = own->get_execution_fence_event();
      // From Copy
      src_requirements   = owner->src_requirements;
      dst_requirements   = owner->dst_requirements;
      grants             = owner->grants;
      wait_barriers      = owner->wait_barriers;
      arrive_barriers    = owner->arrive_barriers;
      parent_task        = owner->parent_task;
      map_id             = owner->map_id;
      tag                = owner->tag;
      // From CopyOp
      src_parent_indexes = owner->src_parent_indexes;
      dst_parent_indexes = owner->dst_parent_indexes;
      src_restrict_infos = owner->src_restrict_infos;
      dst_restrict_infos = owner->dst_restrict_infos;
      predication_guard  = owner->predication_guard;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_index_point(owner->get_unique_op_id(), unique_op_id, p);
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    void PointCopyOp::check_domination(void) const
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        IndexSpace src_space = src_requirements[idx].region.get_index_space();
        IndexSpace dst_space = dst_requirements[idx].region.get_index_space();
        if (!runtime->forest->are_compatible(src_space, dst_space))
          REPORT_LEGION_ERROR(ERROR_COPY_LAUNCHER_INDEX,
                        "Copy launcher index space mismatch at index "
                        "%d of cross-region copy (ID %lld) in task %s "
                        "(ID %lld). Source requirement with index "
                        "space %x and destination requirement "
                        "with index space %x do not have the "
                        "same number of dimensions or the same number "
                        "of elements in their element masks.",
                        idx, get_unique_id(),
                        parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id(),
                        src_space.id, dst_space.id)
        else if (!runtime->forest->is_dominated(src_space, dst_space))
          REPORT_LEGION_ERROR(ERROR_DESTINATION_INDEX_SPACE2,
                        "Destination index space %x for "
                        "requirement %d of cross-region copy "
                        "(ID %lld) in task %s (ID %lld) is not "
                        "a sub-region of the source index space %x.", 
                        dst_space.id, idx, get_unique_id(),
                        parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id(),
                        src_space.id)
      }
    }
#endif

    //--------------------------------------------------------------------------
    void PointCopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_copy();
      owner = NULL;
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_copy();
      runtime->free_point_copy_op(this);
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::launch(const std::set<RtEvent> &index_preconditions)
    //--------------------------------------------------------------------------
    {
      // Copy over the version infos from our owner
      src_versions = owner->src_versions;
      dst_versions = owner->dst_versions;
      // Perform the version analysis
      std::set<RtEvent> preconditions(index_preconditions);
      const UniqueID logical_context_uid = parent_ctx->get_context_uid();
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        perform_projection_version_analysis(owner->src_projection_infos[idx],
                  owner->src_requirements[idx], src_requirements[idx],
                  idx, logical_context_uid, src_versions[idx], preconditions); 
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        if (is_reduce_req)
          dst_requirements[idx].privilege = READ_WRITE;
        perform_projection_version_analysis(owner->dst_projection_infos[idx],
                  owner->dst_requirements[idx], dst_requirements[idx],
                  src_requirements.size() + idx, logical_context_uid,
                  dst_versions[idx], preconditions);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = REDUCE;
      }
      // Then put ourselves in the queue of operations ready to map
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
      // We can also mark this as having our resolved any predication
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      for (std::vector<VersionInfo>::iterator it = src_versions.begin();
            it != src_versions.end(); it++)
        it->clear();
      for (std::vector<VersionInfo>::iterator it = dst_versions.begin();
            it != dst_versions.end(); it++)
        it->clear();
      // Tell our owner that we are done
      owner->handle_point_commit(profiling_reported);
      // Don't commit this operation until we've reported our profiling
      // Out index owner will deactivate the operation
      commit_operation(false/*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    const DomainPoint& PointCopyOp::get_domain_point(void) const
    //--------------------------------------------------------------------------
    {
      return index_point;
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::set_projection_result(unsigned idx, LogicalRegion result)
    //--------------------------------------------------------------------------
    {
      if (idx < src_requirements.size())
      {
#ifdef DEBUG_LEGION
        assert(src_requirements[idx].handle_type != SINGULAR);
#endif
        src_requirements[idx].region = result;
        src_requirements[idx].handle_type = SINGULAR;
      }
      else
      {
        idx -= src_requirements.size();
#ifdef DEBUG_LEGION
        assert(idx < dst_requirements.size());
        assert(dst_requirements[idx].handle_type != SINGULAR);
#endif
        dst_requirements[idx].region = result;
        dst_requirements[idx].handle_type = SINGULAR;
      }
    }

    //--------------------------------------------------------------------------
    const ProjectionInfo* PointCopyOp::get_projection_info(unsigned idx, 
                                                           bool src)
    //--------------------------------------------------------------------------
    {
      return owner->get_projection_info(idx, src);
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
    void FenceOp::initialize(TaskContext *ctx, FenceKind kind)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      execution_precondition = ctx->get_fence_precondition();
#endif
      initialize_operation(ctx, true/*track*/);
      fence_kind = kind;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_fence_operation(parent_ctx->get_unique_id(),
                                       unique_op_id);
    }

    //--------------------------------------------------------------------------
    bool FenceOp::is_execution_fence(void) const
    //--------------------------------------------------------------------------
    {
      return (fence_kind != MAPPING_FENCE);
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
    const char* FenceOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[FENCE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind FenceOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return FENCE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void FenceOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Register this fence with all previous users in the parent's context
      parent_ctx->perform_fence_analysis(this);
      // Now update the parent context with this fence
      // before we can complete the dependence analysis
      // and possibly be deactivated
      parent_ctx->update_current_fence(this);
    }

    //--------------------------------------------------------------------------
    void FenceOp::trigger_mapping(void)
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
#ifdef LEGION_SPY
            // If we're doing Legion Spy verification, we also need to 
            // validate that we have all the completion events from ALL
            // the previous events in the context since the last fence
            trigger_events.insert(execution_precondition);   
#endif
            ApEvent done = Runtime::merge_events(trigger_events);
            // We can always trigger the completion event when these are done
            Runtime::trigger_event(completion_event, done);
            need_completion_trigger = false;
            if (!done.has_triggered())
            {
              RtEvent wait_on = Runtime::protect_event(done);
              // Was already handled above
              if (fence_kind != MIXED_FENCE)
                complete_mapping(wait_on);
              complete_execution(wait_on);
            }
            else
            {
              // Was already handled above
              if (fence_kind != MIXED_FENCE)
                complete_mapping();
              complete_execution();
            }
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
    void FrameOp::initialize(TaskContext *ctx)
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
    const char* FrameOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[FRAME_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind FrameOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return FRAME_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void FrameOp::trigger_mapping(void)
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
#ifdef LEGION_SPY
      // If we're doing Legion Spy verification, we also need to 
      // validate that we have all the completion events from ALL
      // the previous events in the context since the last fence
      trigger_events.insert(execution_precondition);   
#endif
      ApEvent done = Runtime::merge_events(trigger_events);
      // We can always trigger the completion event when these are done
      Runtime::trigger_event(completion_event, done);
      need_completion_trigger = false;
      if (!done.has_triggered())
      {
        RtEvent wait_on = Runtime::protect_event(done);
        DeferredExecuteArgs deferred_execute_args;
        deferred_execute_args.proxy_this = this;
        runtime->issue_runtime_meta_task(deferred_execute_args,
                                         LG_THROUGHPUT_DEFERRED_PRIORITY,
                                         this, wait_on);
      }
      else
        deferred_execute();
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
    void DeletionOp::initialize_index_space_deletion(TaskContext *ctx,
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
    void DeletionOp::initialize_index_part_deletion(TaskContext *ctx,
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
    void DeletionOp::initialize_field_space_deletion(TaskContext *ctx,
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
    void DeletionOp::initialize_field_deletion(TaskContext *ctx, 
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
    void DeletionOp::initialize_field_deletions(TaskContext *ctx,
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
    void DeletionOp::initialize_logical_region_deletion(TaskContext *ctx,
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
    void DeletionOp::initialize_logical_partition_deletion(TaskContext *ctx,
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
    const char* DeletionOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DELETION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind DeletionOp::get_operation_kind(void) const
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
        // No analysis for these since they don't need to defer anything
        case INDEX_SPACE_DELETION:
        case INDEX_PARTITION_DELETION:
        case FIELD_SPACE_DELETION:
          break;
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
      for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
      {
        RegionRequirement &req = deletion_requirements[idx];
        // Perform the normal region requirement analysis
        VersionInfo version_info;
        RestrictInfo restrict_info;
        RegionTreePath privilege_path;
        initialize_privilege_path(privilege_path, req);
        runtime->forest->perform_deletion_analysis(this, idx, req, 
                                                   restrict_info,
                                                   privilege_path);
        version_info.clear();
      }
    }

    //--------------------------------------------------------------------------
    void DeletionOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Iterate over our incoming operations and find the completion 
      // operations that we need to wait to be done executing before
      // we can actually perform the deletion
      std::set<ApEvent> completion_events;
      for (std::map<Operation*,GenerationID>::const_iterator it = 
            incoming.begin(); it != incoming.end(); it++)
      {
        ApEvent complete = it->first->get_completion_event();
        if (it->second == it->first->get_generation())
          completion_events.insert(complete);
      }
      // Mark that we're done mapping and defer the execution as appropriate
      complete_mapping();
      if (!completion_events.empty())
      {
        ApEvent completion_ready = Runtime::merge_events(completion_events);
        complete_execution(Runtime::protect_event(completion_ready));
      }
      else
        complete_execution();
    }

    //--------------------------------------------------------------------------
    void DeletionOp::trigger_complete(void)
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
      complete_operation();
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
    // Internal Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InternalOp::InternalOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InternalOp::~InternalOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void InternalOp::initialize_internal(Operation *creator, int intern_idx,
                                         const TraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(creator != NULL);
#endif
      // We never track internal operations
      initialize_operation(creator->get_context(), false/*track*/);
#ifdef DEBUG_LEGION
      assert(creator_req_idx == -1);
      assert(create_op == NULL);
#endif
      create_op = creator;
      create_gen = creator->get_generation();
      creator_req_idx = intern_idx;
      if (trace_info.trace != NULL)
        set_trace(trace_info.trace, !trace_info.already_traced, NULL); 
    }

    //--------------------------------------------------------------------------
    void InternalOp::activate_internal(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      creator_req_idx = -1;
      create_op = NULL;
      create_gen = 0;
    }

    //--------------------------------------------------------------------------
    void InternalOp::deactivate_internal(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
    }

    //--------------------------------------------------------------------------
    void InternalOp::record_trace_dependence(Operation *target, 
                                             GenerationID target_gen,
                                             int target_idx,
                                             int source_idx, 
                                             DependenceType dtype,
                                             const FieldMask &dependent_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(creator_req_idx >= 0);
#endif
      // Check to see if the target is also our creator
      // in which case we can skip it
      if ((target == create_op) && (target_gen == create_gen))
        return;
      // Check to see if the source is our source
      if (source_idx != creator_req_idx)
        return;
      FieldMask overlap = get_internal_mask() & dependent_mask;
      // If the fields also don't overlap then we are done
      if (!overlap)
        return;
      // Otherwise do the registration
      register_region_dependence(0/*idx*/, target, target_gen,
                               target_idx, dtype, false/*validates*/, overlap);
    }

    //--------------------------------------------------------------------------
    unsigned InternalOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return create_op->find_parent_index(creator_req_idx);
    }

    /////////////////////////////////////////////////////////////
    // Open Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OpenOp::OpenOp(Runtime *rt)
      : InternalOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OpenOp::OpenOp(const OpenOp &rhs)
      : InternalOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    OpenOp::~OpenOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OpenOp& OpenOp::operator=(const OpenOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void OpenOp::initialize(const FieldMask &mask, RegionTreeNode *start, 
                        const RegionTreePath &path, const TraceInfo &trace_info,
                        Operation *creator, int req_idx)
    //--------------------------------------------------------------------------
    {
      initialize_internal(creator, req_idx, trace_info);
#ifdef DEBUG_LEGION
      assert(start_node == NULL);
#endif
      start_node = start;
      open_path = path;
      open_mask = mask;
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_open_operation(creator->get_context()->get_unique_id(),
                                      unique_op_id);
        LegionSpy::log_internal_op_creator(unique_op_id, 
                                           creator->get_unique_op_id(), 
                                           req_idx);
        unsigned parent_index = find_parent_index(0);
        IndexSpace parent_space = 
          parent_ctx->find_logical_region(parent_index).get_index_space();
        if (start_node->is_region())
        {
          const LogicalRegion &handle = start_node->as_region_node()->handle;
          LegionSpy::log_logical_requirement(unique_op_id, 0/*idx*/,
                                    true/*region*/, handle.index_space.id,
                                    handle.field_space.id, handle.tree_id,
                                    READ_WRITE, EXCLUSIVE, 0/*redop*/,
                                    parent_space.id);
        }
        else
        {
          const LogicalPartition &handle = 
            start_node->as_partition_node()->handle;
          LegionSpy::log_logical_requirement(unique_op_id, 0/*idx*/,
                                    false/*region*/, handle.index_partition.id,
                                    handle.field_space.id, handle.tree_id,
                                    READ_WRITE, EXCLUSIVE, 0/*redop*/,
                                    parent_space.id);
        }
        std::set<FieldID> fields;
        start_node->column_source->get_field_set(open_mask, fields);
        LegionSpy::log_requirement_fields(unique_op_id, 0/*idx*/, fields);
      }
    }

    //--------------------------------------------------------------------------
    void OpenOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_internal();
      start_node = NULL;
    }

    //--------------------------------------------------------------------------
    void OpenOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_internal();
      open_path.clear();
      open_mask.clear();
      runtime->free_open_op(this);
    }

    //--------------------------------------------------------------------------
    const char* OpenOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[OPEN_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind OpenOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return OPEN_OP_KIND;
    }

    //--------------------------------------------------------------------------
    const FieldMask& OpenOp::get_internal_mask(void) const
    //--------------------------------------------------------------------------
    {
      return open_mask;
    }

    //--------------------------------------------------------------------------
    void OpenOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> open_events;
      const LegionMap<unsigned,FieldMask>::aligned empty_dirty_previous;
      runtime->forest->advance_version_numbers(this, 0/*idx*/,
                                               false/*update parent state*/,
                                               parent_ctx->get_context_uid(),
                                               false/*doesn't matter*/,
                                               false/*dedup opens*/,
                                               false/*dedup advances*/, 
                                               0/*open id*/, 0/*advance id*/,
                                               start_node, open_path, 
                                               open_mask, empty_dirty_previous,
                                               open_events);
      // Deviate from the normal pipeline and don't even put this on the
      // ready queue, we are done executing and can be considered mapped
      // once all the open events have triggered
      if (!open_events.empty())
        complete_mapping(Runtime::merge_events(open_events));
      else
        complete_mapping();
      complete_execution();
    } 

    /////////////////////////////////////////////////////////////
    // Advance Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AdvanceOp::AdvanceOp(Runtime *rt)
      : InternalOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AdvanceOp::AdvanceOp(const AdvanceOp &rhs)
      : InternalOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    AdvanceOp::~AdvanceOp(void)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    AdvanceOp& AdvanceOp::operator=(const AdvanceOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void AdvanceOp::initialize(RegionTreeNode *parent, const FieldMask &advance,
                               const TraceInfo &trace_info, Operation *creator, 
                               int req_idx, bool is_upper)
    //--------------------------------------------------------------------------
    {
      initialize_internal(creator, req_idx, trace_info);
#ifdef DEBUG_LEGION
      assert(parent_node == NULL);
#endif
      parent_node = parent;
      advance_mask = advance;
      parent_is_upper_bound = is_upper;
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_advance_operation(
            creator->get_context()->get_unique_id(), unique_op_id);
        LegionSpy::log_internal_op_creator(unique_op_id, 
                                           creator->get_unique_op_id(), 
                                           req_idx);
        unsigned parent_index = find_parent_index(0);
        IndexSpace parent_space = 
          parent_ctx->find_logical_region(parent_index).get_index_space();
        if (parent_node->is_region())
        {
          const LogicalRegion &handle = parent_node->as_region_node()->handle;
          LegionSpy::log_logical_requirement(unique_op_id, 0/*idx*/,
                                    true/*region*/, handle.index_space.id,
                                    handle.field_space.id, handle.tree_id,
                                    READ_WRITE, EXCLUSIVE, 0/*redop*/,
                                    parent_space.id);
        }
        else
        {
          const LogicalPartition &handle = 
            parent_node->as_partition_node()->handle;
          LegionSpy::log_logical_requirement(unique_op_id, 0/*idx*/,
                                    false/*region*/, handle.index_partition.id,
                                    handle.field_space.id, handle.tree_id,
                                    READ_WRITE, EXCLUSIVE, 0/*redop*/,
                                    parent_space.id);
        }
        std::set<FieldID> fields;
        parent_node->column_source->get_field_set(advance_mask, fields);
        LegionSpy::log_requirement_fields(unique_op_id, 0/*idx*/, fields);
      }
    }

    //--------------------------------------------------------------------------
    void AdvanceOp::set_child_node(RegionTreeNode *child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(child_node == NULL);
      assert(parent_node->get_depth() <= child->get_depth());
#endif
      child_node = child;
    }

    //--------------------------------------------------------------------------
    void AdvanceOp::set_split_child_mask(const FieldMask &split_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!(split_mask - advance_mask)); // should be dominated
      assert(split_mask != advance_mask); // should not be the same
#endif
      split_child_mask = split_mask;
    }

    //--------------------------------------------------------------------------
    void AdvanceOp::record_dirty_previous(unsigned depth, 
                                          const FieldMask &dirty_mask)
    //--------------------------------------------------------------------------
    {
      LegionMap<unsigned,FieldMask>::aligned::iterator finder = 
        dirty_previous.find(depth);
      if (finder == dirty_previous.end())
        dirty_previous[depth] = dirty_mask;
      else
        finder->second |= dirty_mask;
    }

    //--------------------------------------------------------------------------
    void AdvanceOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_internal();
      parent_node = NULL;
      child_node = NULL;
      parent_is_upper_bound = false;
    }

    //--------------------------------------------------------------------------
    void AdvanceOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_internal();
      advance_mask.clear();
      split_child_mask.clear();
      dirty_previous.clear();
      runtime->free_advance_op(this);
    }

    //--------------------------------------------------------------------------
    const char* AdvanceOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[ADVANCE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind AdvanceOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ADVANCE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    const FieldMask& AdvanceOp::get_internal_mask(void) const
    //--------------------------------------------------------------------------
    {
      return advance_mask;
    }

    //--------------------------------------------------------------------------
    void AdvanceOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Compute the path to use
      RegionTreePath path; 
      runtime->forest->initialize_path(child_node->get_row_source(),
                                       parent_node->get_row_source(), path);
      // Do the advance
      std::set<RtEvent> advance_events;
      if (!split_child_mask)
      {
        // Common case
        runtime->forest->advance_version_numbers(this, 0/*idx*/, 
                                                 true/*update parent state*/,
                                                 parent_is_upper_bound,
                                                 parent_ctx->get_context_uid(),
                                                 false/*dedup opens*/,
                                                 false/*dedup advance*/,
                                                 0/*open id*/, 0/*advance id*/,
                                                 parent_node, path,
                                                 advance_mask, dirty_previous,
                                                 advance_events);
      }
      else
      {
        // This only happens with reductions of multiple fields
        runtime->forest->advance_version_numbers(this, 0/*idx*/,
                                                 true/*update parent state*/,
                                                 parent_is_upper_bound,
                                                 parent_ctx->get_context_uid(),
                                                 false/*dedup opens*/,
                                                 false/*dedup advance*/,
                                                 0/*open id*/, 0/*advance id*/,
                                                 parent_node, path,
                                                 split_child_mask, 
                                                 dirty_previous,
                                                 advance_events);
        RegionTreePath one_up_path;
        runtime->forest->initialize_path(
                                 child_node->get_parent()->get_row_source(),
                                 parent_node->get_row_source(), one_up_path);
        runtime->forest->advance_version_numbers(this, 0/*idx*/,
                                                 true/*update parent state*/,
                                                 parent_is_upper_bound,
                                                 parent_ctx->get_context_uid(),
                                                 false/*dedup opens*/,
                                                 false/*dedup advance*/,
                                                 0/*open id*/, 0/*advance id*/,
                                                 parent_node, one_up_path,
                                                 advance_mask-split_child_mask,
                                                 dirty_previous,
                                                 advance_events);
      }
      // Deviate from the normal pipeline and don't even put this
      // on the ready queue, we are done executing and can be considered
      // mapped once all the advance events have triggered
      if (!advance_events.empty())
        complete_mapping(Runtime::merge_events(advance_events));
      else
        complete_mapping();
      complete_execution();
    }

    /////////////////////////////////////////////////////////////
    // Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CloseOp::CloseOp(Runtime *rt)
      : InternalOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CloseOp::CloseOp(const CloseOp &rhs)
      : InternalOp(NULL)
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
    unsigned CloseOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    int CloseOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    Mappable* CloseOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    size_t CloseOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    const FieldMask& CloseOp::get_internal_mask(void) const
    //--------------------------------------------------------------------------
    {
      // should only be called by inherited classes
      assert(false);
      return *(new FieldMask());
    }

    //--------------------------------------------------------------------------
    void CloseOp::initialize_close(TaskContext *ctx,
                                   const RegionRequirement &req, bool track)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(completion_event.exists());
#endif
      // Only initialize the operation here, this is not a trace-able op
      initialize_operation(ctx, track);
      // Never track this so don't get the close index
      parent_task = ctx->get_task();
      requirement = req;
      initialize_privilege_path(privilege_path, requirement);
    } 

    //--------------------------------------------------------------------------
    void CloseOp::initialize_close(Operation *creator, unsigned idx,
                                   unsigned parent_req_index,
                                   const RegionRequirement &req,
                                   const TraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(completion_event.exists());
#endif
      initialize_internal(creator, idx, trace_info);
      // We always track this so get the close index
      context_index = parent_ctx->register_new_close_operation(this);
      parent_task = parent_ctx->get_task();
      requirement = req;
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_internal_op_creator(unique_op_id, 
                                           creator->get_unique_op_id(), idx);
    }

    //--------------------------------------------------------------------------
    void CloseOp::perform_logging(void)
    //--------------------------------------------------------------------------
    {
      if (!Runtime::legion_spy_enabled)
        return; 
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
      activate_internal();
#ifdef DEBUG_LEGION
      assert(completion_event.exists());
#endif
    }

    //--------------------------------------------------------------------------
    void CloseOp::deactivate_close(void)
    //--------------------------------------------------------------------------
    {
      deactivate_internal();
      privilege_path.clear();
      version_info.clear();
      restrict_info.clear();
      if (mapper_data != NULL)
      {
        free(mapper_data);
        mapper_data = NULL;
        mapper_data_size = 0;
      }
    } 

    //--------------------------------------------------------------------------
    void CloseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.clear();
      commit_operation(true/*deactivate*/);
    }

    /////////////////////////////////////////////////////////////
    // Inter Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InterCloseOp::InterCloseOp(Runtime *runtime)
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
    void InterCloseOp::initialize(TaskContext *ctx,const RegionRequirement &req,
                              ClosedNode *closed_t, const TraceInfo &trace_info,
                              int close_idx, const VersionInfo &clone_info,
                              const FieldMask &close_m, Operation *creator)
    //--------------------------------------------------------------------------
    {
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_close_operation(ctx->get_unique_id(), unique_op_id,
                                       true/*inter close*/, false/*read only*/);
      parent_req_index = creator->find_parent_index(close_idx);
      initialize_close(creator, close_idx, parent_req_index, req, trace_info);
      close_mask = close_m;
      closed_tree = closed_t;
      version_info.clone_logical(clone_info, close_m, closed_t->node);
      if (parent_ctx->has_restrictions())
        parent_ctx->perform_restricted_analysis(requirement, restrict_info);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    } 

    //--------------------------------------------------------------------------
    ProjectionInfo& InterCloseOp::initialize_disjoint_close(
                        const FieldMask &disjoint_mask, IndexSpace launch_space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(requirement.handle_type == PART_PROJECTION);
      assert(!(disjoint_mask - close_mask)); // better dominate
#endif
      // Always the default projection function
      requirement.projection = 0;
      projection_info = ProjectionInfo(runtime, requirement, launch_space);
      disjoint_close_mask = disjoint_mask;
#ifdef LEGION_SPY
      if (Runtime::legion_spy_enabled)
      {
        std::set<FieldID> disjoint_close_fields;
#ifdef DEBUG_LEGION
        assert(closed_tree != NULL);
#endif
        closed_tree->node->column_source->get_field_set(disjoint_close_mask,
            requirement.privilege_fields, disjoint_close_fields);
        for (std::set<FieldID>::const_iterator it = 
              disjoint_close_fields.begin(); it != 
              disjoint_close_fields.end(); it++)
          LegionSpy::log_disjoint_close_field(unique_op_id, *it);
      }
#endif
      return projection_info;
    }

    //--------------------------------------------------------------------------
    InterCloseOp::DisjointCloseInfo* InterCloseOp::find_disjoint_close_child(
                                          unsigned index, RegionTreeNode *child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index == 0); // should always be zero for now
#endif
      // See if it exists already, if not we need to clone the version info
      LegionMap<RegionTreeNode*,DisjointCloseInfo>::aligned::iterator finder = 
        children_to_close.find(child);
      if (finder == children_to_close.end())
      {
        DisjointCloseInfo &info = children_to_close[child];
        // Set the depth for the version info now for when we record info
        const unsigned child_depth = child->get_depth();
        info.version_info.resize(child_depth);
        return &info;
      }
      else
        return &(finder->second);
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::perform_disjoint_close(RegionTreeNode *child_to_close,
                                              DisjointCloseInfo &close_info,
                                              InnerContext *context,
                                              std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      // Clone any version info that we need from the original version info
      const unsigned child_depth = child_to_close->get_depth();
#ifdef DEBUG_LEGION
      assert(child_depth > 0);
#endif
      version_info.clone_to_depth(child_depth-1, close_info.close_mask,
                        context, close_info.version_info, ready_events);
      runtime->forest->physical_perform_close(requirement,
                                              close_info.version_info,
                                              this, 0/*idx*/, 
                                              close_info.close_node,
                                              child_to_close, 
                                              close_info.close_mask,
                                              ready_events,
                                              restrict_info,
                                              chosen_instances, 
                                              &projection_info
#ifdef DEBUG_LEGION
                                              , get_logging_name()
                                              , unique_op_id
#endif
                                              );
      // It is important that we apply our version info when we are done
      close_info.version_info.apply_mapping(ready_events); 
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_close();  
      closed_tree = NULL;
      mapper = NULL;
      outstanding_profiling_requests = 1; // start at 1 to guard
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_close();
#ifdef DEBUG_LEGION
      assert(acquired_instances.empty());
#endif
      acquired_instances.clear();
      map_applied_conditions.clear();
      close_mask.clear();
      if (closed_tree != NULL)
      {
        delete closed_tree;
        closed_tree = NULL;
      }
      chosen_instances.clear();
      disjoint_close_mask.clear();
      projection_info.clear();
      children_to_close.clear();
      profiling_requests.clear();
      runtime->free_inter_close_op(this);
    }

    //--------------------------------------------------------------------------
    const char* InterCloseOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[INTER_CLOSE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind InterCloseOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return INTER_CLOSE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    const FieldMask& InterCloseOp::get_internal_mask(void) const
    //--------------------------------------------------------------------------
    {
      return close_mask;
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      if (!!disjoint_close_mask)
      {
        runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                     requirement,
                                                     privilege_path,
                                                     version_info,
                                                     preconditions,
                                                     false/*partial*/,
                                                     true/*disjoint close*/,
                                                     &disjoint_close_mask);
        FieldMask non_disjoint = close_mask - disjoint_close_mask;
        if (!!non_disjoint) // handle any remaining fields
          runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                       requirement,
                                                       privilege_path,
                                                       version_info,
                                                       preconditions,
                                                       false/*partial*/,
                                                       false/*disjoint close*/,
                                                       &non_disjoint);
      }
      else // the normal path
        runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                     requirement,
                                                     privilege_path,
                                                     version_info,
                                                     preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(completion_event.exists());
      assert(chosen_instances.empty());
#endif
      // See if we are restricted or not and if not find our valid instances 
      if (!restrict_info.has_restrictions())
      {
        InstanceSet valid_instances;
        runtime->forest->physical_premap_only(this, 0/*idx*/, requirement,
                                              version_info, valid_instances);
        // now invoke the mapper to find the instances to use
        invoke_mapper(valid_instances);
      }
      else
        chosen_instances = restrict_info.get_instances();
#ifdef DEBUG_LEGION
      assert(closed_tree != NULL);
#endif
      RegionTreeNode *close_node = (
          requirement.handle_type == PART_PROJECTION) ?
              static_cast<RegionTreeNode*>(
                  runtime->forest->get_node(requirement.partition)) :
              static_cast<RegionTreeNode*>(
                  runtime->forest->get_node(requirement.region));
      // First see if we have any disjoint close fields
      if (!!disjoint_close_mask)
      {
        // Perform the disjoint close, any disjoint children that come
        // back to us will be recorded
        runtime->forest->physical_disjoint_close(this, 0/*idx*/, close_node,
                                         disjoint_close_mask, version_info);
        // These fields have been closed by a disjoint close
        close_mask -= disjoint_close_mask;
        // See if we have any chlidren to close, either handle them now
        // or issue tasks to perform the close operations
        if (!children_to_close.empty())
        {
          for (LegionMap<RegionTreeNode*,DisjointCloseInfo>::aligned::iterator 
                it = children_to_close.begin(); it != 
                children_to_close.end(); it++)
          {
            // Make our copies of the closed tree now
            it->second.close_node = closed_tree->clone_disjoint_projection(
                                          it->first, it->second.close_mask);
            // See if we need to defer it
            if (!it->second.ready_events.empty())
            {
              RtEvent ready_event = 
                Runtime::merge_events(it->second.ready_events);
              if (ready_event.exists() && !ready_event.has_triggered())
              {
                // Now we have to defer it
                DisjointCloseArgs args;
                args.proxy_this = this;
                args.child_node = it->first;
                args.context = find_physical_context(0/*idx*/);
                RtEvent done_event = runtime->issue_runtime_meta_task(args,
                    LG_THROUGHPUT_DEFERRED_PRIORITY, this, ready_event);
                // Add the done event to the map applied events
                map_applied_conditions.insert(done_event);
                continue;
              }
            }
            // If we make it here, we can do the close of the child immediately
            perform_disjoint_close(it->first, it->second,
                     find_physical_context(0/*idx*/), map_applied_conditions);
          }
        }
      }
      // See if we have any remaining fields for which to do a normal close
      if (!!close_mask)
      {
        // Now we can perform our close operation
        runtime->forest->physical_perform_close(requirement,
                                                version_info, this, 0/*idx*/,
                                                closed_tree, 
                                                close_node, close_mask,
                                                map_applied_conditions,
                                                restrict_info,
                                                chosen_instances, NULL
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
        // The physical perform close call took ownership
        closed_tree = NULL;
      }
      if (Runtime::legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, 0/*idx*/,
                                              requirement,
                                              chosen_instances);
      }
      version_info.apply_mapping(map_applied_conditions);
      // Remove profiling our guard and trigger the profiling event if necessary
      if ((__sync_add_and_fetch(&outstanding_profiling_requests, -1) == 0) &&
          profiling_reported.exists())
        Runtime::trigger_event(profiling_reported);
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.clear();
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
      return create_op->find_parent_index(creator_req_idx);
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
    PhysicalManager* InterCloseOp::select_temporary_instance(
           PhysicalManager *dst, unsigned index, const FieldMask &needed_fields)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      Mapper::CreateCloseTemporaryInput input;
      Mapper::CreateCloseTemporaryOutput output;
      input.destination_instance = MappingInstance(dst);
      if (!Runtime::unsafe_mapper)
      {
        // Fields and regions must both be met
        // The instance must be freshly created
        // Instance must be acquired
        std::set<PhysicalManager*> previous_managers;
        // Get the set of previous managers we've made
        for (std::map<PhysicalManager*,std::pair<unsigned,bool> >::
              const_iterator it = acquired_instances.begin(); it !=
              acquired_instances.end(); it++)
          previous_managers.insert(it->first);
        mapper->invoke_close_create_temporary(this, &input, &output);
        validate_temporary_instance(output.temporary_instance.impl,
            previous_managers, acquired_instances, needed_fields,
            requirement.region, mapper, "create_close_temporary_instance");
      }
      else
        mapper->invoke_close_create_temporary(this, &input, &output);
      if (Runtime::legion_spy_enabled)
        log_temporary_instance(output.temporary_instance.impl, 
                               index, needed_fields);
      return output.temporary_instance.impl;
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::invoke_mapper(const InstanceSet &valid_instances)
    //--------------------------------------------------------------------------
    {
      Mapper::MapCloseInput input;
      Mapper::MapCloseOutput output;
      output.profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      // No need to filter for close operations
      if (restrict_info.has_restrictions())
        prepare_for_mapping(restrict_info.get_instances(), 
                            input.valid_instances);
      else
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
        requirement.projection = 0; // always default
      }
      else // This is the common case
        mapper->invoke_map_close(this, &input, &output);
      if (!output.profiling_requests.empty())
      {
        filter_copy_request_kinds(mapper,
            output.profiling_requests.requested_measurements,
            profiling_requests, true/*warn*/);
        profiling_priority = output.profiling_priority;
      }
      // Now we have to validate the output
      // Make sure we have at least one instance for every field
      RegionTreeID bad_tree = 0;
      std::vector<FieldID> missing_fields;
      std::vector<PhysicalManager*> unacquired;
      runtime->forest->physical_convert_mapping(this,
                                  requirement, output.chosen_instances, 
                                  chosen_instances, bad_tree, missing_fields,
                                  &acquired_instances, unacquired, 
                                  !Runtime::unsafe_mapper);
      if (bad_tree > 0)
      {
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'map_close' "
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
        for (std::vector<FieldID>::const_iterator it = missing_fields.begin();
              it != missing_fields.end(); it++)
        {
          const void *name; size_t name_size;
          if (!runtime->retrieve_semantic_information(
               requirement.region.get_field_space(), *it, NAME_SEMANTIC_TAG,
               name, name_size, true, false))
            name = "(no name)";
          log_run.error("Missing instance for field %s (FieldID: %d)",
                        static_cast<const char*>(name), *it);
        }
        REPORT_LEGION_ERROR(ERROR_MISSING_INSTANCE_FIELD,
                     "Invalid mapper output from invocation of 'map_close' "
                      "on mapper %s. Mapper failed to specify a physical "
                      "instance for %zd fields for the region requirement to "
                      "a close operation in task %s (ID %lld). The missing "
                      "fields are listed below.", mapper->get_mapper_name(),
                      missing_fields.size(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
        
      }
      if (!unacquired.empty())
      {
        for (std::vector<PhysicalManager*>::const_iterator it = 
              unacquired.begin(); it != unacquired.end(); it++)
        {
          if (acquired_instances.find(*it) == acquired_instances.end())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from 'map_close' invocation "
                          "on mapper %s. Mapper selected physical instance for "
                          "close operation in task %s (ID %lld) which has "
                          "already been collected. If the mapper had properly "
                          "acquired this instance as part of the mapper call "
                          "it would have detected this. Please update the "
                          "mapper to abide by proper mapping conventions.",
                          mapper->get_mapper_name(),parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id())
        }
        // If we did successfully acquire them, still issue the warning
        REPORT_LEGION_WARNING(ERROR_MAPPER_FAILED_ACQUIRE,
                        "mapper %s failed to acquire instance "
                        "for close operation in task %s (ID %lld) in "
                        "'map_close' call. You may experience undefined "
                        "behavior as a consequence.", mapper->get_mapper_name(),
                        parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id());
      } 
      if (Runtime::unsafe_mapper)
        return;
      std::vector<LogicalRegion> regions_to_check(1, requirement.region);
      for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
      {
        const InstanceRef &ref = chosen_instances[idx];
        if (!ref.has_ref() || ref.is_virtual_ref())
          continue;
        if (!ref.get_manager()->meets_regions(regions_to_check))
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'map_close' "
                        "on mapper %s. Mapper specified an instance which does "
                        "not meet the logical region requirement. The close "
                        "operation was issued in task %s (ID %lld).",
                        mapper->get_mapper_name(), parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::add_copy_profiling_request(
                                           Realm::ProfilingRequestSet &requests)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      ProfilingResponseBase base(this);
      Realm::ProfilingRequest &request = requests.add_request( 
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
          &base, sizeof(base), profiling_priority);
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      int previous = __sync_fetch_and_add(&outstanding_profiling_requests, 1);
      if ((previous == 1) && !profiling_reported.exists())
        profiling_reported = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    void InterCloseOp::handle_profiling_response(
                                       const Realm::ProfilingResponse &response)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapper != NULL);
#endif
      Mapping::Mapper::CloseProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      mapper->invoke_close_report_profiling(this, &info);
#ifdef DEBUG_LEGION
      assert(profiling_reported.exists());
      assert(outstanding_profiling_requests > 0);
#endif
      int remaining = __sync_add_and_fetch(&outstanding_profiling_requests, -1);
      if (remaining == 0)
        Runtime::trigger_event(profiling_reported);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InterCloseOp::handle_disjoint_close(const void *args)
    //--------------------------------------------------------------------------
    {
      const DisjointCloseArgs *close_args = (const DisjointCloseArgs*)args;
      DisjointCloseInfo *close_info = 
        close_args->proxy_this->find_disjoint_close_child(0/*index*/, 
                                                   close_args->child_node);
      std::set<RtEvent> done_events;
      close_args->proxy_this->perform_disjoint_close(close_args->child_node,
                              *close_info, close_args->context, done_events);
      // We actually have to wait for these events to be done
      // since our completion event was put in the map_applied conditions
      if (!done_events.empty())
      {
        RtEvent wait_on = Runtime::merge_events(done_events);
        wait_on.lg_wait();
      }
    }

    /////////////////////////////////////////////////////////////
    // Read Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReadCloseOp::ReadCloseOp(Runtime *rt)
      : CloseOp(rt)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    ReadCloseOp::ReadCloseOp(const ReadCloseOp &rhs)
      : CloseOp(NULL)
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
    void ReadCloseOp::initialize(TaskContext *ctx, const RegionRequirement &req,
                                 const TraceInfo &trace_info, int close_idx,
                                 const FieldMask &close_m, Operation *creator)
    //--------------------------------------------------------------------------
    {
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_close_operation(ctx->get_unique_id(), unique_op_id,
                                       true/*inter close*/, true/*read only*/);
      parent_req_index = creator->find_parent_index(close_idx);
      initialize_close(creator, close_idx, parent_req_index, req, trace_info);
      close_mask = close_m;
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void ReadCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_close();
    }
    
    //--------------------------------------------------------------------------
    void ReadCloseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_close();
      close_mask.clear();
      runtime->free_read_close_op(this);
    }

    //--------------------------------------------------------------------------
    const char* ReadCloseOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[READ_CLOSE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind ReadCloseOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return READ_CLOSE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    const FieldMask& ReadCloseOp::get_internal_mask(void) const
    //--------------------------------------------------------------------------
    {
      return close_mask;
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
    void PostCloseOp::initialize(TaskContext *ctx, unsigned idx,
                                 const InstanceSet &targets) 
    //--------------------------------------------------------------------------
    {
      initialize_close(ctx, ctx->regions[idx], true/*track*/);
      parent_idx = idx;
      target_instances = targets;
      localize_region_requirement(requirement);
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_close_operation(ctx->get_unique_id(), unique_op_id,
                                       false/*inter*/, false/*read only*/);
        perform_logging();
        LegionSpy::log_internal_op_creator(unique_op_id,
                                           ctx->get_unique_id(),
                                           parent_idx);
      }
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_close();
      mapper = NULL;
      outstanding_profiling_requests = 1; // start at 1 to guard
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
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
      profiling_requests.clear();
      target_instances.clear();
      runtime->free_post_close_op(this);
    }

    //--------------------------------------------------------------------------
    const char* PostCloseOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[POST_CLOSE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind PostCloseOp::get_operation_kind(void) const
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
      ProjectionInfo projection_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   restrict_info,
                                                   version_info,
                                                   projection_info,
                                                   privilege_path);
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   privilege_path,
                                                   version_info,
                                                   preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(completion_event.exists());
#endif
      RegionTreeContext physical_ctx = parent_ctx->get_context(); 
      ApEvent close_event = 
        runtime->forest->physical_close_context(physical_ctx, requirement,
                                                version_info, this, 0/*idx*/,
                                                map_applied_conditions,
                                                target_instances 
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
      if (Runtime::legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, 0/*idx*/,
                                              requirement,
                                              target_instances);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, close_event, 
                                        completion_event);
#endif
      }
      // Remove profiling our guard and trigger the profiling event if necessary
      if ((__sync_add_and_fetch(&outstanding_profiling_requests, -1) == 0) &&
          profiling_reported.exists())
        Runtime::trigger_event(profiling_reported);
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
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.clear();
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
    PhysicalManager* PostCloseOp::select_temporary_instance(
           PhysicalManager *dst, unsigned index, const FieldMask &needed_fields)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      Mapper::CreateCloseTemporaryInput input;
      Mapper::CreateCloseTemporaryOutput output;
      input.destination_instance = MappingInstance(dst);
      if (!Runtime::unsafe_mapper)
      {
        // Fields and regions must both be met
        // The instance must be freshly created
        // Instance must be acquired
        std::set<PhysicalManager*> previous_managers;
        // Get the set of previous managers we've made
        for (std::map<PhysicalManager*,std::pair<unsigned,bool> >::
              const_iterator it = acquired_instances.begin(); it !=
              acquired_instances.end(); it++)
          previous_managers.insert(it->first);
        mapper->invoke_close_create_temporary(this, &input, &output);
        validate_temporary_instance(output.temporary_instance.impl,
            previous_managers, acquired_instances, needed_fields,
            requirement.region, mapper, "create_close_temporary_instance");
      }
      else
        mapper->invoke_close_create_temporary(this, &input, &output);
      if (Runtime::legion_spy_enabled)
        log_temporary_instance(output.temporary_instance.impl, 
                               index, needed_fields);
      return output.temporary_instance.impl;
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::add_copy_profiling_request(
                                           Realm::ProfilingRequestSet &requests)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      ProfilingResponseBase base(this);
      Realm::ProfilingRequest &request = requests.add_request( 
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
          &base, sizeof(base), profiling_priority);
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      int previous = __sync_fetch_and_add(&outstanding_profiling_requests, 1);
      if ((previous == 1) && !profiling_reported.exists())
        profiling_reported = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::handle_profiling_response(
                                       const Realm::ProfilingResponse &response)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapper != NULL);
#endif
      Mapping::Mapper::CloseProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      mapper->invoke_close_report_profiling(this, &info);
#ifdef DEBUG_LEGION
      assert(outstanding_profiling_requests > 0);
      assert(profiling_reported.exists());
#endif
      int remaining = __sync_add_and_fetch(&outstanding_profiling_requests, -1);
      if (remaining == 0)
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
    void VirtualCloseOp::initialize(TaskContext *ctx, unsigned index,
                                    const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      initialize_close(ctx, req, true/*track*/);
      parent_idx = index;
      localize_region_requirement(requirement);
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_close_operation(ctx->get_unique_id(), unique_op_id,
                                       false/*inter*/, false/*read only*/);
        perform_logging();
        LegionSpy::log_internal_op_creator(unique_op_id,
                                           ctx->get_unique_id(),
                                           parent_idx);
        for (std::set<FieldID>::const_iterator it = 
              requirement.privilege_fields.begin(); it !=
              requirement.privilege_fields.end(); it++)
          LegionSpy::log_mapping_decision(unique_op_id, 0/*idx*/, *it,
                                          ApEvent::NO_AP_EVENT/*inst event*/);
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
      runtime->free_virtual_close_op(this);
    }

    //--------------------------------------------------------------------------
    const char* VirtualCloseOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[VIRTUAL_CLOSE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind VirtualCloseOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return VIRTUAL_CLOSE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void VirtualCloseOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Just doing the dependence analysis will precipitate any
      // close operations necessary for the virtual close op to
      // do its job, so it needs to do nothing else
      ProjectionInfo projection_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   restrict_info,
                                                   version_info,
                                                   projection_info,
                                                   privilege_path);
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
      parent_task = ctx->get_task();
      initialize_speculation(ctx, true/*track*/,
                             1/*num region requirements*/,
                             launcher.static_dependences,
                             launcher.predicate);
      // Note we give it READ WRITE EXCLUSIVE to make sure that nobody
      // can be re-ordered around this operation for mapping or
      // normal dependences.  We won't actually read or write anything.
      requirement = RegionRequirement(launcher.logical_region, READ_WRITE,
                                      EXCLUSIVE, launcher.parent_region); 
      if (launcher.fields.empty())
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_PRIVILEGE_FIELDS_ACQUIRE,
                         "PRIVILEGE FIELDS OF ACQUIRE OPERATION"
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
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_acquire_operation(parent_ctx->get_unique_id(),
                                         unique_op_id);
    }

    //--------------------------------------------------------------------------
    void AcquireOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative();
      mapper = NULL;
      outstanding_profiling_requests = 1; // start at 1 to guard
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_speculative();  
      privilege_path.clear();
      fields.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      version_info.clear();
      restrict_info.clear();
#ifdef DEBUG_LEGION
      assert(acquired_instances.empty());
#endif
      acquired_instances.clear();
      map_applied_conditions.clear();
      profiling_requests.clear();
      if (mapper_data != NULL)
      {
        free(mapper_data);
        mapper_data = NULL;
        mapper_data_size = 0;
      }
      // Return this operation to the runtime
      runtime->free_acquire_op(this);
    }

    //--------------------------------------------------------------------------
    const char* AcquireOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[ACQUIRE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind AcquireOp::get_operation_kind(void) const
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
    void AcquireOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent index
      compute_parent_index();
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
      { 
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
    void AcquireOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {  
      // Register a dependence on our predicate
      register_predicate_dependence();
      // First register any mapping dependences that we have
      ProjectionInfo projection_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   restrict_info,
                                                   version_info,
                                                   projection_info,
                                                   privilege_path);
      // Tell the parent that we've done an acquisition
      parent_ctx->add_acquisition(this, requirement);
    }

    //--------------------------------------------------------------------------
    bool AcquireOp::query_speculate(bool &value, bool &mapping_only)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      Mapper::SpeculativeOutput output;
      output.speculate = false;
      output.speculate_mapping_only = true;
      mapper->invoke_acquire_speculate(this, &output);
      if (output.speculate)
      {
        value = output.speculative_value;
        mapping_only = output.speculate_mapping_only;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::resolve_true(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // nothing for speculation currently
    }

    //--------------------------------------------------------------------------
    void AcquireOp::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // If we launched there is nothing to do
      if (launched)
        return;
      // Otherwise do the things needed to clean up this operation
      complete_execution();
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      resolve_speculation();
    } 

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;  
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   privilege_path,
                                                   version_info,
                                                   preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Map this is a restricted region. We already know the 
      // physical region that we want to map.
      InstanceSet mapped_instances = restrict_info.get_instances();
      // Invoke the mapper before doing anything else 
      invoke_mapper();
      // Now we can map the operation
      runtime->forest->physical_register_only(requirement,
                                              version_info, restrict_info,
                                              this, 0/*idx*/, completion_event,
                                              false/*defer add users*/,
                                              false/*not read only*/,
                                              map_applied_conditions,
                                              mapped_instances,
                                              NULL/*advance projections*/
#ifdef DEBUG_LEGION
                                              , get_logging_name()
                                              , unique_op_id
#endif
                                              );
      version_info.apply_mapping(map_applied_conditions);
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
          if (Runtime::legion_spy_enabled)
            LegionSpy::log_phase_barrier_wait(unique_op_id, e);
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
          if (Runtime::legion_spy_enabled)
            LegionSpy::log_phase_barrier_arrival(unique_op_id, 
                                                 it->phase_barrier);
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        completion_event);
        }
      }
      // Remove profiling our guard and trigger the profiling event if necessary
      if ((__sync_add_and_fetch(&outstanding_profiling_requests, -1) == 0) &&
          profiling_reported.exists())
        Runtime::trigger_event(profiling_reported);
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
    }

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.clear();
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
    ApEvent AcquireOp::get_restrict_precondition(void) const
    //--------------------------------------------------------------------------
    {
      return merge_restrict_preconditions(grants, wait_barriers);
    }

    //--------------------------------------------------------------------------
    UniqueID AcquireOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    unsigned AcquireOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
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
      FieldID bad_field = AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, check the privileges, but only check the
      // data and not the actual privilege values since we're
      // using psuedo-read-write-exclusive
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, 
                                         bad_index, true/*skip*/);
      switch (et)
      {
        case NO_ERROR:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            REPORT_LEGION_ERROR(ERROR_REQUEST_INVALID_REGION,
                             "Requirements for invalid region handle "
                             "(%x,%d,%d) of requirement for "
                             "acquire operation (ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             unique_op_id)
            break;
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) ||
            (requirement.handle_type == REG_PROJECTION)
            ? requirement.region.field_space :
            requirement.partition.field_space;
            REPORT_LEGION_ERROR(ERROR_FIELD_NOT_VALID,
                             "Field %d is not a valid field of field "
                             "space %d of requirement for acquire "
                             "operation (ID %lld)",
                             bad_field, sp.id, unique_op_id)
            break;
          }
        case ERROR_BAD_PARENT_REGION:
          {
            if (bad_index < 0) 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ACQUIRE,
                               "Parent task %s (ID %lld) of acquire "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent "
                               "because no 'parent' region had that name.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id)
            else if (bad_field == AUTO_GENERATE_ID) 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ACQUIRE,
                               "Parent task %s (ID %lld) of acquire "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent "
                               "because parent requirement %d did not have "
                               "sufficient privileges.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id, bad_index)
            else 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ACQUIRE,
                               "Parent task %s (ID %lld) of acquire "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent "
                               "because region requirement %d was missing "
                               "field %d.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id,
                               bad_index, bad_field)
            break;
          }
        case ERROR_BAD_REGION_PATH:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_NOT_SUBREGION,
                             "Region (%x,%x,%x) is not a "
                             "sub-region of parent region (%x,%x,%x) of "
                             "requirement for acquire operation (ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             requirement.parent.index_space.id,
                             requirement.parent.field_space.id,
                             requirement.parent.tree_id, unique_op_id)
            break;
          }
        case ERROR_BAD_REGION_TYPE:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_REQUIREMENT_ACQUIRE,
                             "Region requirement of acquire operation "
                             "(ID %lld) cannot find privileges for field "
                             "%d in parent task",
                             unique_op_id, bad_field)
            break;
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
        REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ACQUIRE,
                         "Parent task %s (ID %lld) of acquire "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id)
      else
        parent_req_index = unsigned(parent_index);
    }

    //--------------------------------------------------------------------------
    void AcquireOp::invoke_mapper(void)
    //--------------------------------------------------------------------------
    {
      Mapper::MapAcquireInput input;
      Mapper::MapAcquireOutput output;
      output.profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_map_acquire(this, &input, &output);
      if (!output.profiling_requests.empty())
      {
        filter_copy_request_kinds(mapper,
            output.profiling_requests.requested_measurements,
            profiling_requests, true/*warn*/);
        profiling_priority = output.profiling_priority;
      }
    }

    //--------------------------------------------------------------------------
    void AcquireOp::add_copy_profiling_request(
                                           Realm::ProfilingRequestSet &requests)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      ProfilingResponseBase base(this);
      Realm::ProfilingRequest &request = requests.add_request( 
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
          &base, sizeof(base), profiling_priority);
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      int previous = __sync_fetch_and_add(&outstanding_profiling_requests, 1);
      if ((previous == 1) && !profiling_reported.exists())
        profiling_reported = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    void AcquireOp::handle_profiling_response(
                                       const Realm::ProfilingResponse &response)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapper != NULL);
#endif
      Mapping::Mapper::AcquireProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      mapper->invoke_acquire_report_profiling(this, &info);
#ifdef DEBUG_LEGION
      assert(outstanding_profiling_requests > 0);
      assert(profiling_reported.exists());
#endif
      int remaining = __sync_add_and_fetch(&outstanding_profiling_requests, -1);
      if (remaining == 0)
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
      parent_task = ctx->get_task();
      initialize_speculation(ctx, true/*track*/, 
                             1/*num region requirements*/,
                             launcher.static_dependences,
                             launcher.predicate);
      // Note we give it READ WRITE EXCLUSIVE to make sure that nobody
      // can be re-ordered around this operation for mapping or
      // normal dependences.  We won't actually read or write anything.
      requirement = RegionRequirement(launcher.logical_region, READ_WRITE, 
                                      EXCLUSIVE, launcher.parent_region); 
      if (launcher.fields.empty())
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_PRIVILEGE_FIELDS_RELEASE,
                         "PRIVILEGE FIELDS OF RELEASE OPERATION"
                               "IN TASK %s (ID %lld) HAS NO PRIVILEGE "
                               "FIELDS! DID YOU FORGET THEM?!?",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id());
      }
      requirement.privilege_fields = launcher.fields;
      logical_region = launcher.logical_region;
      parent_region = launcher.parent_region;
      fields = launcher.fields; 
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
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_release_operation(parent_ctx->get_unique_id(),
                                         unique_op_id);
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative(); 
      mapper = NULL;
      outstanding_profiling_requests = 1; // start at 1 to guard
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_speculative();
      privilege_path.clear();
      fields.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      version_info.clear();
      restrict_info.clear();
#ifdef DEBUG_LEGION
      assert(acquired_instances.empty());
#endif
      acquired_instances.clear();
      map_applied_conditions.clear();
      profiling_requests.clear();
      if (mapper_data != NULL)
      {
        free(mapper_data);
        mapper_data = NULL;
        mapper_data_size = 0;
      }
      // Return this operation to the runtime
      runtime->free_release_op(this);
    }

    //--------------------------------------------------------------------------
    const char* ReleaseOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[RELEASE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind ReleaseOp::get_operation_kind(void) const
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
    void ReleaseOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent index
      compute_parent_index();
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
      { 
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
    void ReleaseOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {  
      // Register a dependence on our predicate
      register_predicate_dependence();
      // First register any mapping dependences that we have
      ProjectionInfo projection_info;
      // Tell the parent that we did the release
      parent_ctx->remove_acquisition(this, requirement);
      // Register any mapping dependences that we have
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   restrict_info,
                                                   version_info,
                                                   projection_info,
                                                   privilege_path);
    }

    //--------------------------------------------------------------------------
    bool ReleaseOp::query_speculate(bool &value, bool &mapping_only)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      Mapper::SpeculativeOutput output;
      output.speculate = false;
      output.speculate_mapping_only = true;
      mapper->invoke_release_speculate(this, &output);
      if (output.speculate)
      {
        value = output.speculative_value;
        mapping_only = output.speculate_mapping_only;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::resolve_true(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // nothing for speculation right now
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // If we launched then there is nothing to do
      if (launched)
        return;
      // Do the things needed to clean up this operation
      complete_execution();
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      resolve_speculation();
    } 

    //--------------------------------------------------------------------------
    void ReleaseOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   privilege_path,
                                                   version_info,
                                                   preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // We already know what the answer has to be here 
      InstanceSet mapped_instances = restrict_info.get_instances();
      // Invoke the mapper before doing anything else 
      invoke_mapper();
      // Now we can map the operation
      runtime->forest->physical_register_only(requirement,
                                              version_info, restrict_info,
                                              this, 0/*idx*/, completion_event,
                                              false/*defer add users*/,
                                              false/*not read only*/,
                                              map_applied_conditions,
                                              mapped_instances,
                                              NULL/*advance projections*/
#ifdef DEBUG_LEGION
                                              , get_logging_name()
                                              , unique_op_id
#endif
                                              );
      version_info.apply_mapping(map_applied_conditions);
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
          if (Runtime::legion_spy_enabled)
            LegionSpy::log_phase_barrier_wait(unique_op_id, e);
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
          if (Runtime::legion_spy_enabled)
            LegionSpy::log_phase_barrier_arrival(unique_op_id, 
                                                 it->phase_barrier);
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        completion_event);
        }
      }
      // Remove profiling our guard and trigger the profiling event if necessary
      if ((__sync_add_and_fetch(&outstanding_profiling_requests, -1) == 0) &&
          profiling_reported.exists())
        Runtime::trigger_event(profiling_reported);
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
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.clear();
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
    PhysicalManager* ReleaseOp::select_temporary_instance(PhysicalManager *dst,
                                 unsigned index, const FieldMask &needed_fields)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      Mapper::CreateReleaseTemporaryInput input;
      Mapper::CreateReleaseTemporaryOutput output;
      input.destination_instance = MappingInstance(dst);
      if (!Runtime::unsafe_mapper)
      {
        // Fields and regions must both be met
        // The instance must be freshly created
        // Instance must be acquired
        std::set<PhysicalManager*> previous_managers;
        // Get the set of previous managers we've made
        for (std::map<PhysicalManager*,std::pair<unsigned,bool> >::
              const_iterator it = acquired_instances.begin(); it !=
              acquired_instances.end(); it++)
          previous_managers.insert(it->first);
        mapper->invoke_release_create_temporary(this, &input, &output);
        validate_temporary_instance(output.temporary_instance.impl,
            previous_managers, acquired_instances, needed_fields,
            logical_region, mapper, "create_release_temporary_instance");
      }
      else
        mapper->invoke_release_create_temporary(this, &input, &output);
      if (Runtime::legion_spy_enabled)
        log_temporary_instance(output.temporary_instance.impl, 
                               index, needed_fields);
      return output.temporary_instance.impl;
    }

    //--------------------------------------------------------------------------
    ApEvent ReleaseOp::get_restrict_precondition(void) const
    //--------------------------------------------------------------------------
    {
      return merge_restrict_preconditions(grants, wait_barriers);
    }

    //--------------------------------------------------------------------------
    UniqueID ReleaseOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    unsigned ReleaseOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
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
      FieldID bad_field = AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, check the privileges, but only check the
      // data and not the actual privilege values since we're
      // using psuedo-read-write-exclusive
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, 
                                         bad_index, true/*skip*/);
      switch (et)
      {
          // There is no such thing as bad privileges for release operations
          // because we control what they are doing
        case NO_ERROR:
        case ERROR_BAD_REGION_PRIVILEGES:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            REPORT_LEGION_ERROR(ERROR_REQUEST_INVALID_REGION,
                             "Requirements for invalid region handle "
                             "(%x,%d,%d) of requirement for "
                             "release operation (ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             unique_op_id)
            break;
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) ||
            (requirement.handle_type == REG_PROJECTION)
            ? requirement.region.field_space :
            requirement.partition.field_space;
            REPORT_LEGION_ERROR(ERROR_FIELD_NOT_VALID,
                             "Field %d is not a valid field of field "
                             "space %d of requirement for release "
                             "operation (ID %lld)",
                             bad_field, sp.id, unique_op_id)
            break;
          }
        case ERROR_BAD_PARENT_REGION:
          {
            if (bad_index < 0) 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_RELEASE,
                               "Parent task %s (ID %lld) of release "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent "
                               "because no 'parent' region had that name.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id)
            else if (bad_field == AUTO_GENERATE_ID) 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_RELEASE,
                               "Parent task %s (ID %lld) of release "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent "
                               "because parent requirement %d did not have "
                               "sufficient privileges.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id, bad_index)
            else 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_RELEASE,
                               "Parent task %s (ID %lld) of release "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent "
                               "because region requirement %d was missing "
                               "field %d.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id,
                               bad_index, bad_field)
            break;
          }
        case ERROR_BAD_REGION_PATH:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_NOT_SUBREGION,
                             "Region (%x,%x,%x) is not a "
                             "sub-region of parent region (%x,%x,%x) "
                             "of requirement for release "
                             "operation (ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             requirement.parent.index_space.id,
                             requirement.parent.field_space.id,
                             requirement.parent.tree_id, unique_op_id)
            break;
          }
        case ERROR_BAD_REGION_TYPE:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_REQUIREMENT_RELEASE,
                             "Region requirement of release operation "
                             "(ID %lld) cannot find privileges for field "
                             "%d in parent task",
                             unique_op_id, bad_field)
            break;
          }
        // these should never happen with a release operation
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
        REPORT_LEGION_ERROR(ERROR_PARENT_TASK_RELEASE,
                         "Parent task %s (ID %lld) of release "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id)
      else
        parent_req_index = unsigned(parent_index);
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::invoke_mapper(void)
    //--------------------------------------------------------------------------
    {
      Mapper::MapReleaseInput input;
      Mapper::MapReleaseOutput output;
      output.profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_map_release(this, &input, &output);
      if (!output.profiling_requests.empty())
      {
        filter_copy_request_kinds(mapper,
            output.profiling_requests.requested_measurements,
            profiling_requests, true/*warn*/);
        profiling_priority = output.profiling_priority;
      }
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::add_copy_profiling_request(
                                           Realm::ProfilingRequestSet &requests)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      ProfilingResponseBase base(this);
      Realm::ProfilingRequest &request = requests.add_request( 
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
          &base, sizeof(base), profiling_priority);
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      int previous = __sync_fetch_and_add(&outstanding_profiling_requests, 1);
      if ((previous == 1) && !profiling_reported.exists())
        profiling_reported = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::handle_profiling_response(
                                       const Realm::ProfilingResponse &response)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapper != NULL);
#endif
      Mapping::Mapper::ReleaseProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      mapper->invoke_release_report_profiling(this, &info);
#ifdef DEBUG_LEGION
      assert(outstanding_profiling_requests > 0);
      assert(profiling_reported.exists());
#endif
      int remaining = __sync_add_and_fetch(&outstanding_profiling_requests, -1);
      if (remaining == 0)
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
    Future DynamicCollectiveOp::initialize(TaskContext *ctx, 
                                           const DynamicCollective &dc)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      future = Future(new FutureImpl(runtime, true/*register*/,
            runtime->get_available_distributed_id(), 
            runtime->address_space, this));
      collective = dc;
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_dynamic_collective(ctx->get_unique_id(), unique_op_id);
        DomainPoint empty_point;
        LegionSpy::log_future_creation(unique_op_id, 
                                 future.impl->get_ready_event(), empty_point);
      }
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
    const char* DynamicCollectiveOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DYNAMIC_COLLECTIVE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind DynamicCollectiveOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DYNAMIC_COLLECTIVE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // See if we had any contributions for this dynamic collective
      std::vector<Future> contributions;
      parent_ctx->find_collective_contributions(collective, contributions);
      for (std::vector<Future>::const_iterator it = contributions.begin();
            it != contributions.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->impl != NULL);
#endif
        it->impl->register_dependence(this);
#ifdef LEGION_SPY
        if (it->impl->producer_op != NULL)
          LegionSpy::log_mapping_dependence(
              parent_ctx->get_unique_id(), it->impl->producer_uid, 0,
              get_unique_op_id(), 0, TRUE_DEPENDENCE);
#endif
      }
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      ApEvent barrier = Runtime::get_previous_phase(collective.phase_barrier);
      if (!barrier.has_triggered())
      {
        DeferredExecuteArgs deferred_execute_args;
        deferred_execute_args.proxy_this = this;
        runtime->issue_runtime_meta_task(deferred_execute_args,
                                         LG_THROUGHPUT_DEFERRED_PRIORITY, this,
                                         Runtime::protect_event(barrier));
      }
      else
        deferred_execute();
      complete_mapping();
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
    const char* FuturePredOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[FUTURE_PRED_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind FuturePredOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return FUTURE_PRED_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void FuturePredOp::initialize(TaskContext *ctx, Future f)
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
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_predicate_operation(ctx->get_unique_id(), unique_op_id);
        if ((future.impl != NULL) && future.impl->get_ready_event().exists())
          LegionSpy::log_future_use(unique_op_id, 
                                    future.impl->get_ready_event());
      }
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
      // Register this operation as dependent on task that
      // generated the future
      future.impl->register_dependence(this);
    }

    //--------------------------------------------------------------------------
    void FuturePredOp::trigger_ready(void)
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
        args.future_pred_op = this;
        runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, this,
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
    void NotPredOp::initialize(TaskContext *ctx, const Predicate &p)
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
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_predicate_operation(ctx->get_unique_id(), unique_op_id);
        if ((p != Predicate::TRUE_PRED) && (p != Predicate::FALSE_PRED))
          LegionSpy::log_predicate_use(unique_op_id, 
                                       pred_op->get_unique_op_id());
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
    const char* NotPredOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[NOT_PRED_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind NotPredOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return NOT_PRED_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void NotPredOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (pred_op != NULL)
        register_dependence(pred_op, pred_op->get_generation());
    }

    //--------------------------------------------------------------------------
    void NotPredOp::trigger_ready(void)
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
    void AndPredOp::initialize(TaskContext *ctx, 
                               const std::vector<Predicate> &predicates)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
#endif
      // Don't track this as it can lead to deadlock because
      // predicates can't complete until all their references from
      // the parent task have been removed.
      initialize_operation(ctx, false/*track*/);
      // Now do the registration
      for (std::vector<Predicate>::const_iterator it = predicates.begin();
            it != predicates.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->impl != NULL);
#endif
        previous.push_back(it->impl);
        it->impl->add_predicate_reference();
      }
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_predicate_operation(ctx->get_unique_id(), unique_op_id);
        for (std::vector<PredicateOp*>::const_iterator it = previous.begin();
              it != previous.end(); it++)
          LegionSpy::log_predicate_use(unique_op_id, 
                                       (*it)->get_unique_op_id());
      }
    }

    //--------------------------------------------------------------------------
    void AndPredOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_predicate();
      true_count = 0;
      false_short = false;
    }

    //--------------------------------------------------------------------------
    void AndPredOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_predicate();
      previous.clear();
      runtime->free_and_predicate_op(this);
    }

    //--------------------------------------------------------------------------
    const char* AndPredOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[AND_PRED_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind AndPredOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return AND_PRED_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void AndPredOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      for (std::vector<PredicateOp*>::const_iterator it = previous.begin();
            it != previous.end(); it++)
        register_dependence(*it, (*it)->get_generation());
    }

    //--------------------------------------------------------------------------
    void AndPredOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Hold the lock when doing this to prevent 
      // any triggers from interfering with the analysis
      bool need_resolve = false;
      GenerationID local_gen = get_generation();
      {
        AutoLock o_lock(op_lock);
        if (!predicate_resolved)
        {
          for (std::vector<PredicateOp*>::const_iterator it = previous.begin();
                it != previous.end(); it++)
          {
            bool value = false;
            bool valid = (*it)->register_waiter(this, get_generation(), value);
            if (valid)
            {
              if (!value)
              {
                false_short = true;
                break;
              }
              else
                true_count++;
            }
          }
          need_resolve = false_short || (true_count == previous.size());
        }
      }
      if (need_resolve)
        set_resolved_value(local_gen, !false_short);
      // Clean up any references that we have
      for (std::vector<PredicateOp*>::const_iterator it = previous.begin();
            it != previous.end(); it++)
        (*it)->remove_predicate_reference();
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    void AndPredOp::notify_predicate_value(GenerationID pred_gen, bool value)
    //--------------------------------------------------------------------------
    {
      bool need_resolve = false;
      if (pred_gen == get_generation())
      {
        AutoLock o_lock(op_lock);
        // Check again to make sure we didn't lose the race
        if ((pred_gen == get_generation()) && !predicate_resolved)
        {
          if (!value)
          {
            false_short = true;
            need_resolve = true;
          }
          else
          {
            true_count++;
            need_resolve = !false_short && (true_count == previous.size());
          }
        }
      }
      if (need_resolve)
        set_resolved_value(pred_gen, !false_short);
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
    void OrPredOp::initialize(TaskContext *ctx, 
                              const std::vector<Predicate> &predicates)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
#endif
      // Don't track this as it can lead to deadlock because
      // predicates can't complete until all their references from
      // the parent task have been removed.
      initialize_operation(ctx, false/*track*/);
      // Now do the registration
      for (std::vector<Predicate>::const_iterator it = predicates.begin();
            it != predicates.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->impl != NULL);
#endif
        previous.push_back(it->impl);
        it->impl->add_predicate_reference();
      }
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_predicate_operation(ctx->get_unique_id(), unique_op_id);
        for (std::vector<PredicateOp*>::const_iterator it = previous.begin();
              it != previous.end(); it++)
          LegionSpy::log_predicate_use(unique_op_id, 
                                       (*it)->get_unique_op_id());
      }
    }

    //--------------------------------------------------------------------------
    void OrPredOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_predicate();
      false_count = 0;
      true_short = false;
    }

    //--------------------------------------------------------------------------
    void OrPredOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_predicate();
      previous.clear();
      runtime->free_or_predicate_op(this);
    }

    //--------------------------------------------------------------------------
    const char* OrPredOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[OR_PRED_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind OrPredOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return OR_PRED_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void OrPredOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      for (std::vector<PredicateOp*>::const_iterator it = previous.begin();
            it != previous.end(); it++)
        register_dependence(*it, (*it)->get_generation());
    }

    //--------------------------------------------------------------------------
    void OrPredOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Hold the lock when doing this to prevent 
      // any triggers from interfering with the analysis
      bool need_resolve = false;
      GenerationID local_gen = get_generation();
      {
        AutoLock o_lock(op_lock);
        if (!predicate_resolved)
        {
          for (std::vector<PredicateOp*>::const_iterator it = previous.begin();
                it != previous.end(); it++)
          {
            bool value = false;
            bool valid = (*it)->register_waiter(this, get_generation(), value);
            if (valid)
            {
              if (value)
              {
                true_short = true;
                break;
              }
              else
                false_count++;
            }
          }
          need_resolve = true_short || (false_count == previous.size());
        }
      }
      if (need_resolve)
        set_resolved_value(local_gen, true_short);
      // Clean up any references that we have
      for (std::vector<PredicateOp*>::const_iterator it = previous.begin();
            it != previous.end(); it++)
        (*it)->remove_predicate_reference();
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    void OrPredOp::notify_predicate_value(GenerationID pred_gen, bool value)
    //--------------------------------------------------------------------------
    {
      bool need_resolve = false;
      if (pred_gen == get_generation())
      {
        AutoLock o_lock(op_lock);
        // Check again to make sure we didn't lose the race
        if ((pred_gen == get_generation()) && !predicate_resolved)
        {
          if (value)
          {
            true_short = true;
            need_resolve = true;
          }
          else
          {
            false_count++;
            need_resolve = !true_short && (false_count == previous.size());
          }
        }
      }
      if (need_resolve)
        set_resolved_value(pred_gen, true_short);
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
    FutureMap MustEpochOp::initialize(TaskContext *ctx,
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
        indiv_tasks[idx] = runtime->get_available_individual_task();
        indiv_tasks[idx]->initialize_task(ctx, launcher.single_tasks[idx],
                                          check_privileges, false/*track*/);
        indiv_tasks[idx]->set_must_epoch(this, idx, true/*register*/);
        // If we have a trace, set it for this operation as well
        if (trace != NULL)
          indiv_tasks[idx]->set_trace(trace, !trace->is_fixed(), NULL);
        indiv_tasks[idx]->must_epoch_task = true;
      }
      indiv_triggered.resize(indiv_tasks.size(), false);
      index_tasks.resize(launcher.index_tasks.size());
      for (unsigned idx = 0; idx < launcher.index_tasks.size(); idx++)
      {
        IndexSpace launch_space = launcher.index_tasks[idx].launch_space;
        if (!launch_space.exists())
          launch_space = runtime->find_or_create_index_launch_space(
                      launcher.index_tasks[idx].launch_domain);
        index_tasks[idx] = runtime->get_available_index_task();
        index_tasks[idx]->initialize_task(ctx, launcher.index_tasks[idx],
                          launch_space, check_privileges, false/*track*/);
        index_tasks[idx]->set_must_epoch(this, indiv_tasks.size()+idx, 
                                         true/*register*/);
        if (trace != NULL)
          index_tasks[idx]->set_trace(trace, !trace->is_fixed(), NULL);
        index_tasks[idx]->must_epoch_task = true;
      }
      index_triggered.resize(index_tasks.size(), false);
      mapper_id = launcher.map_id;
      mapper_tag = launcher.mapping_tag;
      // Make a new future map for storing our results
      // We'll fill it in later
      result_map = FutureMap(new FutureMapImpl(ctx, this, runtime,
            runtime->get_available_distributed_id(),
            runtime->address_space));
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
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_MUST_EPOCH,
                      "Illegal must epoch launch in task %s (UID %lld). "
            "Must epoch launch requested %zd tasks, but only %zd CPUs "
            "exist in this machine.", parent_ctx->get_task_name(),
            parent_ctx->get_unique_id(), total_points, all_cpus.count());
        assert(false);
      }
#endif
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_must_epoch_operation(ctx->get_unique_id(), unique_op_id);
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
      mapping_dependences.clear();
      input.tasks.clear();
      input.constraints.clear();
      output.task_processors.clear();
      output.constraint_mappings.clear();
      // Return this operation to the free list
      runtime->free_epoch_op(this);
    }

    //--------------------------------------------------------------------------
    const char* MustEpochOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[MUST_EPOCH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind MustEpochOp::get_operation_kind(void) const
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
    void MustEpochOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
        if (indiv_tasks[idx]->has_prepipeline_stage())
          indiv_tasks[idx]->trigger_prepipeline_stage();
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
        if (index_tasks[idx]->has_prepipeline_stage())
          index_tasks[idx]->trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // For every one of our sub-operations, add an additional mapping 
      // dependence.  When our sub-operations map, they will trigger these
      // mapping dependences which guarantees that we will not be able to
      // map until all of the sub-operations are ready to map.
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
        indiv_tasks[idx]->execute_dependence_analysis();
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
        index_tasks[idx]->execute_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // First mark that each of the tasks will be origin mapped
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
        indiv_tasks[idx]->set_origin_mapped(true);
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
        index_tasks[idx]->set_origin_mapped(true);
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
        triggerer.trigger_tasks(indiv_tasks, indiv_triggered,
                                     index_tasks, index_triggered);
#ifdef DEBUG_LEGION
        assert(!single_tasks.empty());
#endif 
        // Next build the set of single tasks and all their constraints.
        // Iterate over all the recorded dependences
        std::vector<Mapper::MappingConstraint> &constraints = input.constraints;
        constraints.resize(dependences.size());
        mapping_dependences.resize(single_tasks.size());
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
          std::set<unsigned> single_indexes;
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
              single_indexes.insert(key.first);
            }
          }
          // Record the mapping dependences
          for (std::set<unsigned>::const_iterator it1 = 
                single_indexes.begin(); it1 != single_indexes.end(); it1++)
          {
            for (std::set<unsigned>::const_iterator it2 = 
                  single_indexes.begin(); it2 != it1; it2++)
            {
              mapping_dependences[*it1].insert(*it2);
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
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of "
                "'map_must_epoch' on mapper %s. Mapper failed to specify "
                "a valid processor for task %s (ID %lld) at index %d. Call "
                "occurred in parent task %s (ID %lld).", 
                mapper->get_mapper_name(), task->get_task_name(),
                task->get_unique_id(), idx, parent_ctx->get_task_name(),
                parent_ctx->get_unique_id())
          if (target_procs.find(proc) != target_procs.end())
          {
            SingleTask *other = target_procs[proc];
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of "
                "'map_must_epoch' on mapper %s. Mapper requests both tasks "
                "%s (ID %lld) and %s (ID %lld) be mapped to the same "
                "processor (" IDFMT ") which is illegal in a must epoch "
                "launch. Must epoch was launched inside of task %s (ID %lld).",
                mapper->get_mapper_name(), other->get_task_name(),
                other->get_unique_id(), task->get_task_name(),
                task->get_unique_id(), proc.id, parent_ctx->get_task_name(),
                parent_ctx->get_unique_id())
          }
          target_procs[proc] = task;
          task->target_proc = proc;
        }
      }
      // Then we need to actually perform the mapping
      {
        MustEpochMapper mapper(this); 
        mapper.map_tasks(single_tasks, mapping_dependences);
        mapping_dependences.clear();
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
        if (dst_index >= 0)
        {
          TaskOp *src_task = find_task_by_index(src_index);
          TaskOp *dst_task = find_task_by_index(dst_index);
          REPORT_LEGION_ERROR(ERROR_MUST_EPOCH_DEPENDENCE,
                        "MUST EPOCH ERROR: dependence between task "
              "%s (ID %lld) and task %s (ID %lld)\n",
              src_task->get_task_name(), src_task->get_unique_id(),
              dst_task->get_task_name(), dst_task->get_unique_id())
        }
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
          REPORT_LEGION_ERROR(ERROR_MUST_EPOCH_DEPENDENCE,
                        "MUST EPOCH ERROR: dependence between region %d "
              "of task %s (ID %lld) and region %d of task %s (ID %lld) of "
              " type %s", src_idx, src_task->get_task_name(),
              src_task->get_unique_id(), dst_idx, 
              dst_task->get_task_name(), dst_task->get_unique_id(),
              (dtype == TRUE_DEPENDENCE) ? "TRUE DEPENDENCE" :
                (dtype == ANTI_DEPENDENCE) ? "ANTI DEPENDENCE" :
                "ATOMIC DEPENDENCE")
        }
        else if (dtype == SIMULTANEOUS_DEPENDENCE)
        {
          // Record the dependence kind
          int dst_index = find_operation_index(dst_op, dst_gen);
#ifdef DEBUG_LEGION
          assert(dst_index >= 0);
#endif
          // See if the dependence record already exists
          const std::pair<unsigned,unsigned> src_key(src_index,src_idx);
          const std::pair<unsigned,unsigned> dst_key(dst_index,dst_idx);
          std::map<std::pair<unsigned,unsigned>,unsigned>::iterator
            src_record_finder = dependence_map.find(src_key);
          if (src_record_finder != dependence_map.end())
          {
            // Already have a source record, see if we have 
            // a destination record too
            std::map<std::pair<unsigned,unsigned>,unsigned>::iterator
              dst_record_finder = dependence_map.find(dst_key); 
            if (dst_record_finder == dependence_map.end())
            {
              // Update the destination record entry
              dependence_map[dst_key] = src_record_finder->second;
              dependences[src_record_finder->second]->add_entry(dst_index, 
                                                                dst_idx);
            }
#ifdef DEBUG_LEGION
            else // both already there so just assert they are the same
              assert(src_record_finder->second == dst_record_finder->second);
#endif
          }
          else
          {
            // No source record
            // See if we have a destination record entry
            std::map<std::pair<unsigned,unsigned>,unsigned>::iterator
              dst_record_finder = dependence_map.find(dst_key);
            if (dst_record_finder == dependence_map.end())
            {
              // Neither source nor destination have an entry so
              // make a new record
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
              // Have a destination but no source, so update the source
              dependence_map[src_key] = dst_record_finder->second;
              dependences[dst_record_finder->second]->add_entry(src_index,
                                                                src_idx);
            }
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
    RtUserEvent MustEpochOp::find_slice_versioning_event(UniqueID slice_id,
                                                         bool &first)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      std::map<UniqueID,RtUserEvent>::const_iterator finder = 
        slice_version_events.find(slice_id);
      if (finder == slice_version_events.end())
      {
        first = true; 
        RtUserEvent result = Runtime::create_rt_user_event();
        slice_version_events[slice_id] = result;
        return result;
      }
      else
      {
        first = false;
        return finder->second;
      }
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
      : current_proc(own->get_context()->get_executing_processor()), owner(own)
    //--------------------------------------------------------------------------
    {
      trigger_lock = Reservation::create_reservation();
    }

    //--------------------------------------------------------------------------
    MustEpochTriggerer::MustEpochTriggerer(const MustEpochTriggerer &rhs)
      : current_proc(rhs.current_proc), owner(rhs.owner)
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
    void MustEpochTriggerer::trigger_tasks(
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
          args.triggerer = this;
          args.task = indiv_tasks[idx];
          RtEvent wait = 
            owner->runtime->issue_runtime_meta_task(args, 
                  LG_THROUGHPUT_DEFERRED_PRIORITY, owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        if (!index_triggered[idx])
        {
          MustEpochIndexArgs args;
          args.triggerer = this;
          args.task = index_tasks[idx];
          RtEvent wait = 
            owner->runtime->issue_runtime_meta_task(args,
                  LG_THROUGHPUT_DEFERRED_PRIORITY, owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
      }
      // Wait for all of the launches to be done
      // We can safely block to free up the utility processor
      if (!wait_events.empty())
      {
        RtEvent trigger_event = Runtime::merge_events(wait_events);
        trigger_event.lg_wait();
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochTriggerer::trigger_individual(IndividualTask *task)
    //--------------------------------------------------------------------------
    {
      task->set_target_proc(current_proc);
      task->trigger_mapping();
    }

    //--------------------------------------------------------------------------
    void MustEpochTriggerer::trigger_index(IndexTask *task)
    //--------------------------------------------------------------------------
    {
      task->set_target_proc(current_proc);
      task->trigger_mapping();
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
      : owner(own)
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
    void MustEpochMapper::map_tasks(const std::deque<SingleTask*> &single_tasks,
                            const std::vector<std::set<unsigned> > &dependences)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(single_tasks.size() == dependences.size());
#endif
      MustEpochMapArgs args;
      args.mapper = this;
      // For correctness we still have to abide by the mapping dependences
      // computed on the individual tasks while we are mapping them
      std::vector<RtEvent> mapped_events(single_tasks.size());
      for (unsigned idx = 0; idx < single_tasks.size(); idx++)
      {
        // Figure out our preconditions
        std::set<RtEvent> preconditions;
        for (std::set<unsigned>::const_iterator it = 
              dependences[idx].begin(); it != dependences[idx].end(); it++)
        {
#ifdef DEBUG_LEGION
          assert((*it) < idx);
#endif
          preconditions.insert(mapped_events[*it]);          
        }
        args.task = single_tasks[idx];
        if (!preconditions.empty())
        {
          RtEvent precondition = Runtime::merge_events(preconditions);
          mapped_events[idx] = 
            owner->runtime->issue_runtime_meta_task(args, 
                LG_THROUGHPUT_DEFERRED_PRIORITY, owner, precondition); 
        }
        else
          mapped_events[idx] = 
            owner->runtime->issue_runtime_meta_task(args,
                  LG_THROUGHPUT_DEFERRED_PRIORITY, owner);
      }
      std::set<RtEvent> wait_events(mapped_events.begin(), mapped_events.end());
      if (!wait_events.empty())
      {
        RtEvent mapped_event = Runtime::merge_events(wait_events);
        mapped_event.lg_wait();
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochMapper::map_task(SingleTask *task)
    //--------------------------------------------------------------------------
    {
      // Before we can actually map, we have to perform our versioning analysis
      RtEvent versions_ready = task->perform_must_epoch_version_analysis(owner);
      if (versions_ready.exists())
        versions_ready.lg_wait();
      // Note we don't need to hold a lock here because this is
      // a monotonic change.  Once it fails for anyone then it
      // fails for everyone.
      RtEvent done_mapping = task->perform_mapping(owner);
      if (done_mapping.exists())
        done_mapping.lg_wait();
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
      MustEpochLauncherArgs launch_args;
      std::set<RtEvent> wait_events;
      for (std::vector<IndividualTask*>::const_iterator it = 
            indiv_tasks.begin(); it != indiv_tasks.end(); it++)
      {
        if (!runtime->is_local((*it)->target_proc))
        {
          dist_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(dist_args, 
                LG_THROUGHPUT_DEFERRED_PRIORITY, owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
        else
        {
          launch_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(launch_args,
                  LG_THROUGHPUT_DEFERRED_PRIORITY, owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
      }
      for (std::set<SliceTask*>::const_iterator it = 
            slice_tasks.begin(); it != slice_tasks.end(); it++)
      {
        (*it)->update_target_processor();
        if (!runtime->is_local((*it)->target_proc))
        {
          dist_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(dist_args, 
                LG_THROUGHPUT_DEFERRED_PRIORITY, owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
        else
        {
          launch_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(launch_args,
                 LG_THROUGHPUT_DEFERRED_PRIORITY, owner);
          if (wait.exists())
            wait_events.insert(wait);
        }
      }
      if (!wait_events.empty())
      {
        RtEvent dist_event = Runtime::merge_events(wait_events);
        dist_event.lg_wait();
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
    void PendingPartitionOp::initialize_equal_partition(TaskContext *ctx,
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
    void PendingPartitionOp::initialize_union_partition(TaskContext *ctx,
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
    void PendingPartitionOp::initialize_intersection_partition(TaskContext *ctx,
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
    void PendingPartitionOp::initialize_difference_partition(TaskContext *ctx,
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
    void PendingPartitionOp::initialize_restricted_partition(TaskContext *ctx,
                                                          IndexPartition pid,
                                                          const void *transform,
                                                          size_t transform_size,
                                                          const void *extent,
                                                          size_t extent_size)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new RestrictedPartitionThunk(pid, transform, transform_size,
                                           extent, extent_size);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_cross_product(TaskContext *ctx,
                                                      IndexPartition base,
                                                      IndexPartition source,
                                                      LegionColor part_color)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new CrossProductThunk(base, source, part_color);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_union(TaskContext *ctx,
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
    void PendingPartitionOp::initialize_index_space_union(TaskContext *ctx,
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
    TaskContext *ctx, IndexSpace target, const std::vector<IndexSpace> &handles)
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
                     TaskContext *ctx, IndexSpace target, IndexPartition handle)
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
    void PendingPartitionOp::initialize_index_space_difference(TaskContext *ctx,
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
    void PendingPartitionOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Give these slightly higher priority since they are likely
      // needed by later operations
      enqueue_ready_operation(RtEvent::NO_RT_EVENT, 
                              LG_THROUGHPUT_DEFERRED_PRIORITY);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Perform the partitioning operation
      ApEvent ready_event = thunk->perform(this, runtime->forest);
      complete_mapping();
      Runtime::trigger_event(completion_event, ready_event);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(ready_event));
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
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
    const char* PendingPartitionOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[PENDING_PARTITION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind PendingPartitionOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return PENDING_PARTITION_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::EqualPartitionThunk::perform_logging(
                                                         PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(op->unique_op_id, pid.id,
          EQUAL_PARTITION);
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
    void PendingPartitionOp::RestrictedPartitionThunk::perform_logging(
                                                         PendingPartitionOp *op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(op->unique_op_id, pid.id,
          RESTRICTED_PARTITION);
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
    // Dependent Partition Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DependentPartitionOp::DependentPartitionOp(Runtime *rt)
      : Operation(rt), thunk(NULL)
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
    void DependentPartitionOp::initialize_by_field(TaskContext *ctx, 
                                                   IndexPartition pid,
                                                   LogicalRegion handle, 
                                                   LogicalRegion parent,
                                                   FieldID fid,
                                                   MapperID id, MappingTagID t)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (!runtime->forest->check_partition_by_field_size(pid, 
            handle.get_field_space(), fid, false/*range*/, 
            true/*use color space*/))
      {
        log_run.error("ERROR: Field size of field %d does not match the size "
                      "of the color space elements for 'partition_by_field' "
                      "call in task %s (UID %lld)", fid, ctx->get_task_name(),
                      ctx->get_unique_id());
        assert(false);
      }
#endif
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/); 
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      requirement = RegionRequirement(handle, READ_ONLY, EXCLUSIVE, parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ByFieldThunk(pid);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_image(TaskContext *ctx, 
                                                   IndexPartition pid,
                                          LogicalPartition projection,
                                          LogicalRegion parent, FieldID fid,
                                          MapperID id, MappingTagID t) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (!runtime->forest->check_partition_by_field_size(pid, 
            projection.get_field_space(), fid, false/*range*/))
      {
        log_run.error("ERROR: Field size of field %d does not match the size "
                      "of the destination index space elements for "
                      "'partition_by_image' call in task %s (UID %lld)",
                      fid, ctx->get_task_name(), ctx->get_unique_id());
        assert(false);
      }
#endif
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      LogicalRegion proj_parent = 
        runtime->forest->get_parent_logical_region(projection);
      requirement = RegionRequirement(proj_parent, READ_ONLY, EXCLUSIVE,parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ByImageThunk(pid, projection.get_index_partition());
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_image_range(TaskContext *ctx, 
                                                         IndexPartition pid,
                                                LogicalPartition projection,
                                                LogicalRegion parent,
                                                FieldID fid, MapperID id,
                                                MappingTagID t) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (!runtime->forest->check_partition_by_field_size(pid, 
            projection.get_field_space(), fid, true/*range*/))
      {
        log_run.error("ERROR: Field size of field %d does not match the size "
                      "of the destination index space elements for "
                      "'partition_by_image_range' call in task %s (UID %lld)",
                      fid, ctx->get_task_name(), ctx->get_unique_id());
        assert(false);
      }
#endif
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      LogicalRegion proj_parent = 
        runtime->forest->get_parent_logical_region(projection);
      requirement = RegionRequirement(proj_parent, READ_ONLY, EXCLUSIVE,parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ByImageRangeThunk(pid, projection.get_index_partition());
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_preimage(TaskContext *ctx,
                                    IndexPartition pid, IndexPartition proj,
                                    LogicalRegion handle, LogicalRegion parent,
                                    FieldID fid, MapperID id, MappingTagID t)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (!runtime->forest->check_partition_by_field_size(proj,
            handle.get_field_space(), fid, false/*range*/))
      {
        log_run.error("ERROR: Field size of field %d does not match the size "
                      "of the range index space elements for "
                      "'partition_by_preimage' call in task %s (UID %lld)",
                      fid, ctx->get_task_name(), ctx->get_unique_id());
        assert(false);
      }
#endif
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      requirement = RegionRequirement(handle, READ_ONLY, EXCLUSIVE, parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ByPreimageThunk(pid, proj);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_preimage_range(TaskContext *ctx,
                                    IndexPartition pid, IndexPartition proj,
                                    LogicalRegion handle, LogicalRegion parent,
                                    FieldID fid, MapperID id, MappingTagID t)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (!runtime->forest->check_partition_by_field_size(proj,
            handle.get_field_space(), fid, true/*range*/))
      {
        log_run.error("ERROR: Field size of field %d does not match the size "
                     "of the range index space elements for "
                     "'partition_by_preimage_range' call in task %s (UID %lld)",
                     fid, ctx->get_task_name(), ctx->get_unique_id());
        assert(false);
      }
#endif
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      requirement = RegionRequirement(handle, READ_ONLY, EXCLUSIVE, parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ByPreimageRangeThunk(pid, proj);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_association(TaskContext *ctx,
                        LogicalRegion domain, LogicalRegion domain_parent, 
                        FieldID fid, IndexSpace range, 
                        MapperID id, MappingTagID t)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (!runtime->forest->check_association_field_size(range,
            domain.get_field_space(), fid))
      {
        log_run.error("ERROR: Field size of field %d does not match the size "
                      "of the range index space elements for "
                      "'create_association' call in task %s (UID %lld)",
                      fid, ctx->get_task_name(), ctx->get_unique_id());
        assert(false);
      }
#endif
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/);
      // start-off with non-projection requirement
      requirement = RegionRequirement(domain, READ_WRITE, 
                                      EXCLUSIVE, domain_parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new AssociationThunk(domain.get_index_space(), range);
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::perform_logging(void) const
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_dependent_partition_operation(
          parent_ctx->get_unique_id(), unique_op_id, 
          thunk->get_partition().get_id(), thunk->get_kind());
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::log_requirement(void) const
    //--------------------------------------------------------------------------
    {
      if (requirement.handle_type == PART_PROJECTION)
      {
        LegionSpy::log_logical_requirement(unique_op_id, 0/*idx*/,
                                  false/*region*/,
                                  requirement.partition.index_partition.id,
                                  requirement.partition.field_space.id,
                                  requirement.partition.tree_id,
                                  requirement.privilege,
                                  requirement.prop,
                                  requirement.redop,
                                  requirement.parent.index_space.id);
        LegionSpy::log_requirement_projection(unique_op_id, 0/*idx*/, 
                                              requirement.projection);
        runtime->forest->log_launch_space(launch_space, unique_op_id);
      }
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
    void DependentPartitionOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      compute_parent_index();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Before doing the dependence analysis we have to ask the
      // mapper whether it would like to make this an index space
      // operation or a single operation
      select_partition_projection();
      // Do thise now that we've picked our region requirement
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
        log_requirement();
      if (is_index_space)
        projection_info = ProjectionInfo(runtime, requirement, launch_space);
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   restrict_info,
                                                   version_info,
                                                   projection_info,
                                                   privilege_path);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::select_partition_projection(void)
    //--------------------------------------------------------------------------
    {
      Mapper::SelectPartitionProjectionInput input;
      Mapper::SelectPartitionProjectionOutput output;
      // Find the open complete projections, and then invoke the mapper call
      runtime->forest->find_open_complete_partitions(this, 0/*idx*/, 
                        requirement, input.open_complete_partitions);
      // Invoke the mapper
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_select_partition_projection(this, &input, &output);
      // Check the output
      if (output.chosen_partition == LogicalPartition::NO_PART)
        return;
      IndexPartNode *partition_node = 
       runtime->forest->get_node(output.chosen_partition.get_index_partition());
      // Make sure that it is complete, and then update our information
      if (!Runtime::unsafe_mapper && !partition_node->is_complete(false))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of "
                      "'select_partition_projection' on mapper %s."
                      "Mapper selected a logical partition that is "
                      "not complete for dependent partitioning operation "
                      "in task %s (UID %lld).", mapper->get_mapper_name(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id())
      // Update the region requirement and other information
      requirement.partition = output.chosen_partition;
      requirement.handle_type = PART_PROJECTION;
      requirement.projection = 0; // always default
      launch_space = partition_node->color_space->handle;
      index_domain = partition_node->color_space->get_color_space_domain();
      is_index_space = true;
#ifdef LEGION_SPY
      intermediate_index_event = Runtime::create_ap_user_event();
#endif
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      // See if this is an index space operation
      if (is_index_space)
      {
#ifdef DEBUG_LEGION
        assert(requirement.handle_type == PART_PROJECTION);
#endif
        // Perform the partial versioning analysis
        runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                     requirement,
                                                     privilege_path,
                                                     version_info,
                                                     preconditions,
                                                     true/*partial*/);
        // Now enumerate the points and kick them off
        size_t num_points = index_domain.get_volume();
#ifdef DEBUG_LEGION
        assert(num_points > 0);
#endif
        unsigned point_idx = 0;
        points.resize(num_points);
        for (Domain::DomainPointIterator itr(index_domain); 
              itr; itr++, point_idx++)
        {
          PointDepPartOp *point = 
            runtime->get_available_point_dep_part_op();
          point->initialize(this, itr.p);
          points[point_idx] = point;
        }
        // Perform the projections
        ProjectionFunction *function = 
          runtime->find_projection_function(requirement.projection);
        std::vector<ProjectionPoint*> projection_points(points.begin(),
                                                        points.end());
        function->project_points(this, 0/*idx*/, requirement,
                                 runtime, projection_points);
        // No need to check the validity of the points, we know they are good
        if (Runtime::legion_spy_enabled)
        {
          for (std::vector<PointDepPartOp*>::const_iterator it = 
                points.begin(); it != points.end(); it++)
            (*it)->log_requirement();
        }
        // Launch the points
        std::set<RtEvent> mapped_preconditions;
        for (std::vector<PointDepPartOp*>::const_iterator it = 
              points.begin(); it != points.end(); it++)
        {
          mapped_preconditions.insert((*it)->get_mapped_event());
          (*it)->launch(preconditions);
        }
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                        completion_event);
#endif
        // We are mapped when all our points are mapped
        complete_mapping(Runtime::merge_events(mapped_preconditions));
      }
      else
      {
        // Path for a non-index space implementation
        runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                     requirement,
                                                     privilege_path,
                                                     version_info,
                                                     preconditions);
        // Give these operations slightly higher priority since
        // they are likely needed for other operations
        if (!preconditions.empty())
          enqueue_ready_operation(Runtime::merge_events(preconditions),
                                  LG_THROUGHPUT_DEFERRED_PRIORITY);
        else
          enqueue_ready_operation(RtEvent::NO_RT_EVENT, 
                                  LG_THROUGHPUT_DEFERRED_PRIORITY);
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(requirement.handle_type == SINGULAR);
#endif
      // Perform the mapping call to get the physical isntances
      InstanceSet valid_instances, mapped_instances;
      runtime->forest->physical_premap_only(this, 0/*idx*/, requirement,
                                            version_info, valid_instances);
      // We have the valid instances so invoke the mapper
      invoke_mapper(valid_instances, mapped_instances);
      if (Runtime::legion_spy_enabled)
        runtime->forest->log_mapping_decision(unique_op_id, 0/*idx*/,
                                              requirement,
                                              mapped_instances);
#ifdef DEBUG_LEGION
      assert(!mapped_instances.empty()); 
#endif
      // Then we can register our mapped_instances
      runtime->forest->physical_register_only(requirement, 
                                              version_info, restrict_info,
                                              this, 0/*idx*/,
                                              completion_event,
                                              false/*defer add users*/,
                                              true/*read only locks*/,
                                              map_applied_conditions,
                                              mapped_instances,
                                              NULL/*no projection info*/
#ifdef DEBUG_LEGION
                                              , get_logging_name()
                                              , unique_op_id
#endif
                                              );
      ApEvent done_event = trigger_thunk(requirement.region.get_index_space(),
                                         mapped_instances);
      // Apply our changes to the version state
      version_info.apply_mapping(map_applied_conditions);
      // Once we are done running these routines, we can mark
      // that the handles have all been completed
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      if (!restricted_postconditions.empty())
      {
        restricted_postconditions.insert(done_event);
        done_event = Runtime::merge_events(restricted_postconditions);
      }
#ifdef LEGION_SPY
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_operation_events(unique_op_id, done_event,
                                        completion_event);
#endif
      Runtime::trigger_event(completion_event, done_event);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(done_event));
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::trigger_thunk(IndexSpace handle,
                                                const InstanceSet &mapped_insts)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(requirement.privilege_fields.size() == 1);
      assert(mapped_insts.size() == 1);
#endif
      if (is_index_space)
      {
        // Update our data structure and see if we are the ones
        // to perform the operation
        bool ready = false;
        {
          AutoLock o_lock(op_lock);
          instances.resize(instances.size() + 1);
          FieldDataDescriptor &desc = instances.back();
          const InstanceRef &ref = mapped_insts[0];
          PhysicalManager *manager = ref.get_manager();
          desc.index_space = handle;
          desc.inst = manager->get_instance();
          desc.field_offset = manager->layout->find_field_info(
                        *(requirement.privilege_fields.begin())).field_id;
          index_preconditions.insert(ref.get_ready_event());
#ifdef DEBUG_LEGION
          assert(!points.empty());
#endif
          ready = (instances.size() == points.size());
        }
        if (ready)
        {
          ApEvent done_event = thunk->perform(this, runtime->forest,
              Runtime::merge_events(index_preconditions), instances);
          Runtime::trigger_event(completion_event, done_event);
          need_completion_trigger = false;
#ifdef LEGION_SPY
          Runtime::trigger_event(intermediate_index_event, done_event);
#endif
          complete_execution(Runtime::protect_event(done_event));
        }
#ifdef LEGION_SPY
        return intermediate_index_event;
#else
        return completion_event;
#endif
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(instances.empty());
#endif
        instances.resize(1);
        FieldDataDescriptor &desc = instances[0];
        const InstanceRef &ref = mapped_insts[0];
        PhysicalManager *manager = ref.get_manager();
        desc.index_space = handle;
        desc.inst = manager->get_instance();
        desc.field_offset = manager->layout->find_field_info(
                      *(requirement.privilege_fields.begin())).field_id;
        ApEvent ready_event = ref.get_ready_event();
        return thunk->perform(this, runtime->forest, 
                              ready_event, instances);
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::invoke_mapper(const InstanceSet &valid_instances,
                                             InstanceSet &mapped_instances)
    //--------------------------------------------------------------------------
    {
      Mapper::MapPartitionInput input;
      Mapper::MapPartitionOutput output;
      if (restrict_info.has_restrictions())
      {
        prepare_for_mapping(restrict_info.get_instances(), 
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
      mapper->invoke_map_partition(this, &input, &output);
      if (!output.profiling_requests.empty())
        filter_copy_request_kinds(mapper,
            output.profiling_requests.requested_measurements,
            profiling_requests, true/*warn*/);
      // Now we have to validate the output
      // Go through the instances and make sure we got one for every field
      // Also check to make sure that none of them are composite instances
      RegionTreeID bad_tree = 0;
      std::vector<FieldID> missing_fields;
      std::vector<PhysicalManager*> unacquired;
      int composite_index = runtime->forest->physical_convert_mapping(this,
                                requirement, output.chosen_instances, 
                                mapped_instances, bad_tree, missing_fields,
                                &acquired_instances, unacquired, 
                                !Runtime::unsafe_mapper);
      if (bad_tree > 0)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'map_partition'"
                      " on mapper %s. Mapper selected instance from region "
                      "tree %d to satisfy a region requirement for a partition "
                      "mapping in task %s (ID %lld) whose logical region is "
                      "from region tree %d.", mapper->get_mapper_name(),
                      bad_tree, parent_ctx->get_task_name(), 
                      parent_ctx->get_unique_id(), 
                      requirement.region.get_tree_id())
      if (!missing_fields.empty())
      {
        for (std::vector<FieldID>::const_iterator it = missing_fields.begin();
              it != missing_fields.end(); it++)
        {
          const void *name; size_t name_size;
          if (!runtime->retrieve_semantic_information(
               requirement.region.get_field_space(), *it, NAME_SEMANTIC_TAG,
               name, name_size, true, false))
            name = "(no name)";
          log_run.error("Missing instance for field %s (FieldID: %d)",
                        static_cast<const char*>(name), *it);
        }
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'map_partition'"
                      " on mapper %s. Mapper failed to specify a physical "
                      "instance for %zd fields of the region requirement to "
                      "a partition mapping in task %s (ID %lld). The missing "
                      "fields are listed below.", mapper->get_mapper_name(),
                      missing_fields.size(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id());
      }
      if (!unacquired.empty())
      {
        for (std::vector<PhysicalManager*>::const_iterator it = 
              unacquired.begin(); it != unacquired.end(); it++)
        {
          if (acquired_instances.find(*it) == acquired_instances.end())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from 'map_partition' "
                        "invocation on mapper %s. Mapper selected physical "
                        "instance for partition mapping in task %s (ID %lld) "
                        "which has already been collected. If the mapper had "
                        "properly acquired this instance as part of the mapper "
                        "call it would have detected this. Please update the "
                        "mapper to abide by proper mapping conventions.", 
                        mapper->get_mapper_name(), parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
        }
        // If we did successfully acquire them, still issue the warning
        REPORT_LEGION_WARNING(ERROR_MAPPER_FAILED_ACQUIRE,
                        "WARNING: mapper %s faield to acquire instance "
                        "for partition mapping operation in task %s (ID %lld) "
                        "in 'map_partition' call. You may experience undefined "
                        "behavior as a consequence.", mapper->get_mapper_name(),
                        parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id())
      }
      if (composite_index >= 0)
      {
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'map_partition'"
                      " on mapper %s. Mapper requested creation of a composite "
                      "instance for partition mapping in task %s (ID %lld).",
                      mapper->get_mapper_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
      } 
      // If we are doing unsafe mapping, then we can return
      if (Runtime::unsafe_mapper)
        return;
      // Iterate over the instances and make sure they are all valid
      // for the given logical region which we are mapping
      std::vector<LogicalRegion> regions_to_check(1, requirement.region);
      for (unsigned idx = 0; idx < mapped_instances.size(); idx++)
      {
        if (!mapped_instances[idx].get_manager()->meets_regions(
                                                        regions_to_check))
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of "
                        "'map_partition' on mapper %s. Mapper specified an "
                        "instance that does not meet the logical region "
                        "requirement. The inline mapping operation was issued "
                        "in task %s (ID %lld).", mapper->get_mapper_name(), 
                        parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
      }
      for (unsigned idx = 0; idx < mapped_instances.size(); idx++)
      {
        if (!mapped_instances[idx].get_manager()->is_instance_manager())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of "
                        "'map_partition' on mapper %s. Mapper selected an "
                        "illegal specialized reduction instance for "
                        "partition operation in task %s (ID %lld).",
                        mapper->get_mapper_name(),parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.clear();
      bool commit_now = false;
      if (is_index_space)
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(!commit_request);
#endif
        commit_request = true;
        commit_now = (points.size() == points_committed);
      }
      else
        commit_now = true;
      if (commit_now)
        commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::handle_point_commit(RtEvent point_committed)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_index_space);
#endif
      bool commit_now = false;
      RtEvent commit_pre;
      {
        AutoLock o_lock(op_lock);
        points_committed++;
        if (point_committed.exists())
          commit_preconditions.insert(point_committed);
        commit_now = commit_request && (points.size() == points_committed);
      }
      if (commit_now)
        commit_operation(true/*deactivate*/,
                          Runtime::merge_events(commit_preconditions));
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::ByFieldThunk::perform(
     DependentPartitionOp *op, RegionTreeForest *forest,
     ApEvent instances_ready, const std::vector<FieldDataDescriptor> &instances)
    //--------------------------------------------------------------------------
    {
      return forest->create_partition_by_field(op, pid, 
                                               instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::ByImageThunk::perform(
     DependentPartitionOp *op, RegionTreeForest *forest,
     ApEvent instances_ready, const std::vector<FieldDataDescriptor> &instances)
    //--------------------------------------------------------------------------
    {
      return forest->create_partition_by_image(op, pid, projection, 
                                               instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::ByImageRangeThunk::perform(
     DependentPartitionOp *op, RegionTreeForest *forest,
     ApEvent instances_ready, const std::vector<FieldDataDescriptor> &instances)
    //--------------------------------------------------------------------------
    {
      return forest->create_partition_by_image_range(op, pid, projection, 
                                                     instances,instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::ByPreimageThunk::perform(
     DependentPartitionOp *op, RegionTreeForest *forest,
     ApEvent instances_ready, const std::vector<FieldDataDescriptor> &instances)
    //--------------------------------------------------------------------------
    {
      return forest->create_partition_by_preimage(op, pid, projection, 
                                                  instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::ByPreimageRangeThunk::perform(
     DependentPartitionOp *op, RegionTreeForest *forest,
     ApEvent instances_ready, const std::vector<FieldDataDescriptor> &instances)
    //--------------------------------------------------------------------------
    {
      return forest->create_partition_by_preimage_range(op, pid, projection, 
                                                  instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::AssociationThunk::perform(
     DependentPartitionOp *op, RegionTreeForest *forest,
     ApEvent instances_ready, const std::vector<FieldDataDescriptor> &instances)
    //--------------------------------------------------------------------------
    {
      return forest->create_association(op, domain, range, 
                                        instances, instances_ready);
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
    Partition::PartitionKind DependentPartitionOp::get_partition_kind(void) 
                                                                           const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(thunk != NULL);
#endif
      return thunk->get_kind();
    }

    //--------------------------------------------------------------------------
    UniqueID DependentPartitionOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    unsigned DependentPartitionOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    int DependentPartitionOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_dependent_op(); 
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::activate_dependent_op(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      is_index_space = false;
      launch_space = IndexSpace::NO_SPACE;
      index_domain = Domain::NO_DOMAIN;
      parent_req_index = 0;
      mapper = NULL;
      points_committed = 0;
      commit_request = false;
      outstanding_profiling_requests = 1; // start at 1 to guard
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_dependent_op(); 
      if (mapper_data != NULL)
      {
        free(mapper_data);
        mapper_data = NULL;
        mapper_data_size = 0;
      }
      runtime->free_dependent_partition_op(this);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::deactivate_dependent_op(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      if (thunk != NULL)
      {
        delete thunk;
        thunk = NULL;
      }
      privilege_path = RegionTreePath();
      projection_info.clear();
      version_info.clear();
      restrict_info.clear();
      map_applied_conditions.clear();
      acquired_instances.clear();
      restricted_postconditions.clear();
      // We deactivate all of our point operations
      for (std::vector<PointDepPartOp*>::const_iterator it = 
            points.begin(); it != points.end(); it++)
        (*it)->deactivate();
      points.clear();
      instances.clear();
      index_preconditions.clear();
      commit_preconditions.clear();
      profiling_requests.clear();
    }

    //--------------------------------------------------------------------------
    const char* DependentPartitionOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DEPENDENT_PARTITION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind DependentPartitionOp::get_operation_kind(void) const
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
    void DependentPartitionOp::select_sources(const InstanceRef &target,
                                              const InstanceSet &sources,
                                              std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      Mapper::SelectPartitionSrcInput input;
      Mapper::SelectPartitionSrcOutput output;
      prepare_for_mapping(sources, input.source_instances);
      prepare_for_mapping(target, input.target);
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_select_partition_sources(this, &input, &output);
      compute_ranking(output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                          DependentPartitionOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    PhysicalManager* DependentPartitionOp::select_temporary_instance(
           PhysicalManager *dst, unsigned index, const FieldMask &needed_fields)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      Mapper::CreatePartitionTemporaryInput input;
      Mapper::CreatePartitionTemporaryOutput output;
      input.destination_instance = MappingInstance(dst);
      if (!Runtime::unsafe_mapper)
      {
        // Fields and regions must both be met
        // The instance must be freshly created
        // Instance must be acquired
        std::set<PhysicalManager*> previous_managers;
        // Get the set of previous managers we've made
        for (std::map<PhysicalManager*,std::pair<unsigned,bool> >::
              const_iterator it = acquired_instances.begin(); it !=
              acquired_instances.end(); it++)
          previous_managers.insert(it->first);
        // Do the mapper call now
        mapper->invoke_partition_create_temporary(this, &input, &output);
        validate_temporary_instance(output.temporary_instance.impl,
            previous_managers, acquired_instances, needed_fields,
            requirement.region, mapper, "create_partition_temporary_instance");
      }
      else
        mapper->invoke_partition_create_temporary(this, &input, &output);
      if (Runtime::legion_spy_enabled)
        log_temporary_instance(output.temporary_instance.impl, 
                               index, needed_fields);
      return output.temporary_instance.impl;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::record_restrict_postcondition(
                                                          ApEvent postcondition)
    //--------------------------------------------------------------------------
    {
      restricted_postconditions.insert(postcondition);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::add_copy_profiling_request(
                                           Realm::ProfilingRequestSet &requests)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      ProfilingResponseBase base(this);
      Realm::ProfilingRequest &request = requests.add_request( 
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
          &base, sizeof(base));
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      int previous = __sync_fetch_and_add(&outstanding_profiling_requests, 1);
      if ((previous == 1) && !profiling_reported.exists())
        profiling_reported = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::handle_profiling_response(
                                       const Realm::ProfilingResponse &response)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapper != NULL);
#endif
      Mapping::Mapper::PartitionProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      mapper->invoke_partition_report_profiling(this, &info);
#ifdef DEBUG_LEGION
      assert(outstanding_profiling_requests > 0);
      assert(profiling_reported.exists());
#endif
      int remaining = __sync_add_and_fetch(&outstanding_profiling_requests, -1);
      // If this was the last one, we can trigger our events
      if (remaining == 0)
        Runtime::trigger_event(profiling_reported);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::compute_parent_index(void)
    //--------------------------------------------------------------------------
    {
      int parent_index = parent_ctx->find_parent_region_req(requirement);
      if (parent_index < 0)
        REPORT_LEGION_ERROR(ERROR_PARENT_TASK_PARTITION,
                         "Parent task %s (ID %lld) of partition "
                         "operation (ID %lld) does not have a region "
                         "requirement for region (%x,%x,%x) "
                         "as a parent of region requirement.",
                         parent_ctx->get_task_name(),
                         parent_ctx->get_unique_id(),
                         unique_op_id,
                         requirement.region.index_space.id,
                         requirement.region.field_space.id,
                         requirement.region.tree_id)
      else
        parent_req_index = unsigned(parent_index);
    } 

    ///////////////////////////////////////////////////////////// 
    // Point Dependent Partition Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointDepPartOp::PointDepPartOp(Runtime *rt)
      : DependentPartitionOp(rt), owner(NULL)
    //--------------------------------------------------------------------------
    {
      is_index_space = false;
    }

    //--------------------------------------------------------------------------
    PointDepPartOp::PointDepPartOp(const PointDepPartOp &rhs)
      : DependentPartitionOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PointDepPartOp::~PointDepPartOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointDepPartOp& PointDepPartOp::operator=(const PointDepPartOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::initialize(DependentPartitionOp *own, 
                                    const DomainPoint &p)
    //--------------------------------------------------------------------------
    {
      initialize_operation(own->get_context(), false/*track*/, 1/*size*/);
      index_point = p;
      owner = own;
      requirement = owner->requirement;
      parent_task = owner->parent_task;
      map_id      = owner->map_id;
      tag         = owner->tag;
      parent_req_index = owner->parent_req_index;
      restrict_info = owner->restrict_info;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_index_point(own->get_unique_op_id(), unique_op_id, p);
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::launch(const std::set<RtEvent> &launch_preconditions)
    //--------------------------------------------------------------------------
    {
      // Copy over the version infos from our owner
      version_info = owner->version_info;
      // Perform the version analysis for our point
      std::set<RtEvent> preconditions(launch_preconditions);
      const UniqueID logical_context_uid = parent_ctx->get_context_uid();
      perform_projection_version_analysis(owner->projection_info,
          owner->requirement, requirement, 0/*idx*/, 
          logical_context_uid, version_info, preconditions);
      // Then put ourselves in the queue of operations ready to map
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
      // We can also mark this as having our resolved any predication
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_dependent_op();
      owner = NULL;
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_dependent_op();
      runtime->free_point_dep_part_op(this);
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ApEvent PointDepPartOp::trigger_thunk(IndexSpace handle,
                                          const InstanceSet &mapped_instances)
    //--------------------------------------------------------------------------
    {
      return owner->trigger_thunk(handle, mapped_instances);
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // Tell our owner that we are done
      owner->handle_point_commit(profiling_reported);
      // Don't commit this operation until we've reported our profiling
      // Out index owner will deactivate the operation
      commit_operation(false/*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    const DomainPoint& PointDepPartOp::get_domain_point(void) const
    //--------------------------------------------------------------------------
    {
      return index_point;
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::set_projection_result(unsigned idx, 
                                               LogicalRegion result)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
      assert(requirement.handle_type == PART_PROJECTION);
#endif
      requirement.region = result;
      requirement.handle_type = SINGULAR;
    }

    ///////////////////////////////////////////////////////////// 
    // Fill Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FillOp::FillOp(Runtime *rt)
      : SpeculativeOp(rt), Fill()
    //--------------------------------------------------------------------------
    {
      this->is_index_space = false;
    }

    //--------------------------------------------------------------------------
    FillOp::FillOp(const FillOp &rhs)
      : SpeculativeOp(NULL), Fill()
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
    void FillOp::initialize(TaskContext *ctx, const FillLauncher &launcher,
                            bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      parent_task = ctx->get_task();
      initialize_speculation(ctx, true/*track*/, 1, 
                             launcher.static_dependences, launcher.predicate);
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
      map_id = launcher.map_id;
      tag = launcher.tag;
      if (check_privileges)
        check_fill_privilege();
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_fill_operation(parent_ctx->get_unique_id(), 
                                      unique_op_id);
        if ((value_size == 0) && (future.impl != NULL) &&
            future.impl->get_ready_event().exists())
          LegionSpy::log_future_use(unique_op_id, 
                                    future.impl->get_ready_event());
      }
    }

    //--------------------------------------------------------------------------
    void FillOp::activate_fill(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative();
      value = NULL;
      value_size = 0;
      true_guard = PredEvent::NO_PRED_EVENT;
      false_guard = PredEvent::NO_PRED_EVENT;
    }

    //--------------------------------------------------------------------------
    void FillOp::deactivate_fill(void)
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
      if (mapper_data != NULL)
      {
        free(mapper_data);
        mapper_data = NULL;
        mapper_data_size = 0;
      }
    }

    //--------------------------------------------------------------------------
    void FillOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_fill(); 
    }

    //--------------------------------------------------------------------------
    void FillOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_fill(); 
      runtime->free_fill_op(this);
    }

    //--------------------------------------------------------------------------
    const char* FillOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[FILL_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind FillOp::get_operation_kind(void) const
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
    Mappable* FillOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    UniqueID FillOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    unsigned FillOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index; 
    }

    //--------------------------------------------------------------------------
    int FillOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    void FillOp::log_fill_requirement(void) const
    //--------------------------------------------------------------------------
    {
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
    void FillOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent index
      compute_parent_index();
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
        log_fill_requirement();
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_dependence_analysis(void) 
    //--------------------------------------------------------------------------
    {
      // Register a dependence on our predicate
      register_predicate_dependence();
      // If we are waiting on a future register a dependence
      if (future.impl != NULL)
        future.impl->register_dependence(this);
      ProjectionInfo projection_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   restrict_info,
                                                   version_info,
                                                   projection_info,
                                                   privilege_path);
    }

    //--------------------------------------------------------------------------
    bool FillOp::query_speculate(bool &value, bool &mapping_only)
    //--------------------------------------------------------------------------
    {
      // Always speculate on fill ops, but mapping only since
      // we know that there is an easy way to defer them
#if 1
      value = true;
      mapping_only = true;
#ifdef DEBUG_LEGION
      assert(!true_guard.exists());
      assert(!false_guard.exists());
#endif
      // Make the copy across precondition guard 
      predicate->get_predicate_guards(true_guard, false_guard);
      return true;
#else
      return false;
#endif
    }

    //--------------------------------------------------------------------------
    void FillOp::resolve_true(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void FillOp::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // If we already launched then there is nothing to do
      if (launched)
        return;
      // Otherwise do the work to clean up this operation
      // Mark that this operation has completed both
      // execution and mapping indicating that we are done
      // Do it in this order to avoid calling 'execute_trigger'
      complete_execution();
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      resolve_speculation();
    } 

    //--------------------------------------------------------------------------
    void FillOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   privilege_path,
                                                   version_info,
                                                   preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
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
          mapped_instances = restrict_info.get_instances();
          runtime->forest->physical_register_only(requirement,
                                                  version_info, restrict_info,
                                                  this, 0/*idx*/,
                                                  ApEvent::NO_AP_EVENT,
                                                  false/*defer add users*/,
                                                  false/*not read only*/,
                                                  map_applied_conditions,
                                                  mapped_instances,
                                                  get_projection_info()
#ifdef DEBUG_LEGION
                                                  , get_logging_name()
                                                  , unique_op_id
#endif
                                                  );
        }
        ApEvent sync_precondition = compute_sync_precondition();
        ApEvent done_event = 
          runtime->forest->fill_fields(this, requirement, 
                                       0/*idx*/, value, value_size, 
                                       version_info, restrict_info, 
                                       mapped_instances, sync_precondition,
                                       map_applied_conditions, 
                                       true_guard, false_guard);
        if (!mapped_instances.empty())
          runtime->forest->log_mapping_decision(unique_op_id, 0/*idx*/,
                                                requirement,
                                                mapped_instances);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, done_event, 
                                        completion_event);
#endif
        version_info.apply_mapping(map_applied_conditions);
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
            if (Runtime::legion_spy_enabled)
              LegionSpy::log_phase_barrier_arrival(unique_op_id, 
                                                   it->phase_barrier);
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
          deferred_execute_args.proxy_this = this;
          runtime->issue_runtime_meta_task(deferred_execute_args,
                                           LG_THROUGHPUT_DEFERRED_PRIORITY,this,
                                    Runtime::protect_event(future_ready_event));
        }
        else
          deferred_execute(); // can do the completion now
      }
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
      InstanceSet mapped_instances;
      if (restrict_info.has_restrictions())
      {
        mapped_instances = restrict_info.get_instances();
        runtime->forest->physical_register_only(requirement,
                                                version_info, restrict_info,
                                                this, 0/*idx*/,
                                                ApEvent::NO_AP_EVENT,
                                                false/*defer add users*/,
                                                false/*not read only*/,
                                                map_applied_conditions,
                                                mapped_instances,
                                                get_projection_info()
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
      }
      ApEvent sync_precondition = compute_sync_precondition();
      ApEvent done_event = 
          runtime->forest->fill_fields(this, requirement, 
                                       0/*idx*/, result, result_size, 
                                       version_info, restrict_info, 
                                       mapped_instances, sync_precondition,
                                       map_applied_conditions,
                                       true_guard, false_guard);
      if (!mapped_instances.empty())
        runtime->forest->log_mapping_decision(unique_op_id, 0/*idx*/,
                                              requirement,
                                              mapped_instances);
#ifdef LEGION_SPY
      LegionSpy::log_operation_events(unique_op_id, done_event,
                                      completion_event);
#endif
      version_info.apply_mapping(map_applied_conditions);
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
          if (Runtime::legion_spy_enabled)
            LegionSpy::log_phase_barrier_arrival(unique_op_id, 
                                                 it->phase_barrier);
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
      version_info.clear();
      commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    ApEvent FillOp::get_restrict_precondition(void) const
    //--------------------------------------------------------------------------
    {
      return merge_restrict_preconditions(grants, wait_barriers);
    }

    //--------------------------------------------------------------------------
    const ProjectionInfo* FillOp::get_projection_info(void)
    //--------------------------------------------------------------------------
    {
      // No advance projection info for normal fills
      return NULL;
    }

    //--------------------------------------------------------------------------
    void FillOp::check_fill_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field = AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, bad_index);
      switch (et)
      {
        case NO_ERROR:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            REPORT_LEGION_ERROR(ERROR_REQUEST_INVALID_REGION,
                             "Requirements for invalid region handle "
                             "(%x,%d,%d) for fill operation"
                             "(ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             unique_op_id)
            break;
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) ||
            (requirement.handle_type == REG_PROJECTION)
            ? requirement.region.field_space :
            requirement.partition.field_space;
            REPORT_LEGION_ERROR(ERROR_FIELD_NOT_VALID,
                             "Field %d is not a valid field of field "
                             "space %d for fill operation (ID %lld)",
                             bad_field, sp.id, unique_op_id)
            break;
          }
        case ERROR_INVALID_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_PRIVILEGE,
                             "Instance field %d is not one of the "
                             "privilege fields for fill operation"
                             "(ID %lld)",
                             bad_field, unique_op_id)
            break;
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_DUPLICATE,
                             "Instance field %d is a duplicate for "
                             "fill operation (ID %lld)",
                             bad_field, unique_op_id)
            break;
          }
        case ERROR_BAD_PARENT_REGION:
          {
            if (bad_index < 0) 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_FILL,
                               "Parent task %s (ID %lld) of fill operation "
                               "(ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) "
                               "as a parent of region requirement because "
                               "no 'parent' region had that name.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id)
            else if (bad_field == AUTO_GENERATE_ID) 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_FILL,
                               "Parent task %s (ID %lld) of fill operation "
                               "(ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) "
                               "as a parent of region requirement because "
                               "parent requirement %d did not have "
                               "sufficient privileges.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id, bad_index)
            else 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_FILL,
                               "Parent task %s (ID %lld) of fill operation "
                               "(ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) "
                               "as a parent of region requirement because "
                               "region requirement %d was missing field %d.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id,
                               bad_index, bad_field)
            break;
          }
        case ERROR_BAD_REGION_PATH:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_NOT_SUBREGION,
                             "Region (%x,%x,%x) is not a "
                             "sub-region of parent region "
                             "(%x,%x,%x) for region requirement of fill "
                             "operation (ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             requirement.parent.index_space.id,
                             requirement.parent.field_space.id,
                             requirement.parent.tree_id,
                             unique_op_id)
            break;
          }
        case ERROR_BAD_REGION_TYPE:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_REQUIREMENT_FILL,
                             "Region requirement of fill operation "
                             "(ID %lld) cannot find privileges for field "
                             "%d in parent task",
                             unique_op_id, bad_field)
            break;
          }
        case ERROR_BAD_REGION_PRIVILEGES:
          {
            REPORT_LEGION_ERROR(ERROR_PRIVILEGES_REGION_SUBSET,
                             "Privileges %x for region "
                             "(%x,%x,%x) are not a subset of privileges "
                             "of parent task's privileges for region "
                             "requirement of fill operation (ID %lld)",
                             requirement.privilege,
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             unique_op_id)
            break;
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
        REPORT_LEGION_ERROR(ERROR_PARENT_TASK_FILL,
                         "Parent task %s (ID %lld) of fill "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id(),
                               unique_op_id, 
                               requirement.parent.index_space.id,
                               requirement.parent.field_space.id, 
                               requirement.parent.tree_id)
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
          if (Runtime::legion_spy_enabled)
            LegionSpy::log_phase_barrier_wait(unique_op_id, e);
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
    // Index Fill Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexFillOp::IndexFillOp(Runtime *rt)
      : FillOp(rt)
    //--------------------------------------------------------------------------
    {
      this->is_index_space = true;
    }

    //--------------------------------------------------------------------------
    IndexFillOp::IndexFillOp(const IndexFillOp &rhs)
      : FillOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndexFillOp::~IndexFillOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexFillOp& IndexFillOp::operator=(const IndexFillOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::initialize(TaskContext *ctx,
                                 const IndexFillLauncher &launcher,
                                 IndexSpace launch_sp, bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      parent_task = ctx->get_task();
      initialize_speculation(ctx, true/*track*/, 1, 
                             launcher.static_dependences, launcher.predicate);
#ifdef DEBUG_LEGION
      assert(launch_sp.exists());
#endif
      launch_space = launch_sp;
      if (!launcher.launch_domain.exists())
        runtime->forest->find_launch_space_domain(launch_space, index_domain);
      else
        index_domain = launcher.launch_domain;
      if (launcher.region.exists())
      {
#ifdef DEBUG_LEGION
        assert(!launcher.partition.exists());
#endif
        requirement = RegionRequirement(launcher.region, launcher.projection,
                                        WRITE_DISCARD, EXCLUSIVE,
                                        launcher.parent);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(launcher.partition.exists());
#endif
        requirement = RegionRequirement(launcher.partition, launcher.projection,
                                        WRITE_DISCARD, EXCLUSIVE,
                                        launcher.parent);
      }
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
      map_id = launcher.map_id;
      tag = launcher.tag;
      if (check_privileges)
        check_fill_privilege();
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_fill_operation(parent_ctx->get_unique_id(), 
                                      unique_op_id);
        if ((value_size == 0) && (future.impl != NULL) &&
            future.impl->get_ready_event().exists())
          LegionSpy::log_future_use(unique_op_id, 
                                    future.impl->get_ready_event());
        runtime->forest->log_launch_space(launch_space, unique_op_id);
      }
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_fill();
      index_domain = Domain::NO_DOMAIN;
      launch_space = IndexSpace::NO_SPACE;
      points_committed = 0;
      commit_request = false;
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_fill();
      projection_info.clear();
      // We can deactivate our point operations
      for (std::vector<PointFillOp*>::const_iterator it = points.begin();
            it != points.end(); it++)
        (*it)->deactivate();
      points.clear();
      // Return the operation to the runtime
      runtime->free_index_fill_op(this);
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent index
      compute_parent_index();
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
      { 
        const bool reg = (requirement.handle_type == SINGULAR) ||
                         (requirement.handle_type == REG_PROJECTION);
        const bool proj = (requirement.handle_type == REG_PROJECTION) ||
                          (requirement.handle_type == PART_PROJECTION); 

        LegionSpy::log_logical_requirement(unique_op_id, 0/*idx*/, reg,
            reg ? requirement.region.index_space.id :
                  requirement.partition.index_partition.id,
            reg ? requirement.region.field_space.id :
                  requirement.partition.field_space.id,
            reg ? requirement.region.tree_id : 
                  requirement.partition.tree_id,
            requirement.privilege, requirement.prop, 
            requirement.redop, requirement.parent.index_space.id);
        LegionSpy::log_requirement_fields(unique_op_id, 0/*idx*/, 
                                          requirement.privilege_fields);
        if (proj)
          LegionSpy::log_requirement_projection(unique_op_id, 0/*idx*/, 
                                                requirement.projection);
      }
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Register a dependence on our predicate
      register_predicate_dependence();
      // If we are waiting on a future register a dependence
      if (future.impl != NULL)
        future.impl->register_dependence(this);
      projection_info = ProjectionInfo(runtime, requirement, launch_space);
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   restrict_info,
                                                   version_info,
                                                   projection_info,
                                                   privilege_path);
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Do the upper bound version analysis first
      std::set<RtEvent> preconditions;
      if (!version_info.has_physical_states())
      {
        const bool partial_traversal = 
          (projection_info.projection_type == PART_PROJECTION) ||
          ((projection_info.projection_type != SINGULAR) && 
           (projection_info.projection->depth > 0));
        runtime->forest->perform_versioning_analysis(this, 0/*idx*/, 
                                                     requirement,
                                                     privilege_path,
                                                     version_info,
                                                     preconditions,
                                                     partial_traversal);
      }
      // Now enumerate the points
      size_t num_points = index_domain.get_volume();
#ifdef DEBUG_LEGION
      assert(num_points > 0);
#endif
      unsigned point_idx = 0;
      points.resize(num_points);
      for (Domain::DomainPointIterator itr(index_domain); 
            itr; itr++, point_idx++)
      {
        PointFillOp *point = runtime->get_available_point_fill_op();
        point->initialize(this, itr.p);
        points[point_idx] = point;
      }
      // Now we have to do the projection
      ProjectionFunction *function = 
        runtime->find_projection_function(requirement.projection);
      std::vector<ProjectionPoint*> projection_points(points.begin(),
                                                      points.end());
      function->project_points(this, 0/*idx*/, requirement,
                               runtime, projection_points);
#ifdef DEBUG_LEGION
      // Check for interfering point requirements in debug mode
      check_point_requirements();
#endif
      if (Runtime::legion_spy_enabled)
      {
        for (std::vector<PointFillOp*>::const_iterator it = points.begin();
              it != points.end(); it++)
          (*it)->log_fill_requirement();
      }
      // Launch the points
      std::set<RtEvent> mapped_preconditions;
      std::set<ApEvent> executed_preconditions;
      for (std::vector<PointFillOp*>::const_iterator it = points.begin();
            it != points.end(); it++)
      {
        mapped_preconditions.insert((*it)->get_mapped_event());
        executed_preconditions.insert((*it)->get_completion_event());
        (*it)->launch(preconditions);
      }
#ifdef LEGION_SPY
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      completion_event);
#endif
      // Record that we are mapped when all our points are mapped
      // and we are executed when all our points are executed
      complete_mapping(Runtime::merge_events(mapped_preconditions));
      ApEvent done = Runtime::merge_events(executed_preconditions);
      Runtime::trigger_event(completion_event, done);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(done));
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // This should never be called as this operation doesn't
      // go through the rest of the queue normally
      assert(false);
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      bool commit_now = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(!commit_request);
#endif
        commit_request = true;
        commit_now = (points.size() == points_committed);
      }
      if (commit_now)
        commit_operation(true/*deactivate*/); 
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::handle_point_commit(void)
    //--------------------------------------------------------------------------
    {
      bool commit_now = false;
      {
        AutoLock o_lock(op_lock);
        points_committed++;
        commit_now = commit_request && (points.size() == points_committed);
      }
      if (commit_now)
        commit_operation(true/*deactivate*/);
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    void IndexFillOp::check_point_requirements(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx1 = 0; idx1 < points.size(); idx1++)
      {
        const RegionRequirement &req1 = points[idx1]->get_requirement();
        for (unsigned idx2 = 0; idx2 < idx1; idx2++)
        {
          const RegionRequirement &req2 = points[idx2]->get_requirement();
          if (!runtime->forest->are_disjoint(req1.region.get_index_space(), 
                                             req2.region.get_index_space()))
          {
            const DomainPoint &p1 = points[idx1]->get_domain_point();
            const DomainPoint &p2 = points[idx2]->get_domain_point();
            if (p1.get_dim() <= 1) 
            {
              REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_FILL,
                            "Index space fill launch has intefering "
                            "region requirements 0 of point %lld and region "
                            "requirement 0 of point %lld of %s (UID %lld) "
                            "in parent task %s (UID %lld) are interfering.",
                            p1[0], p2[0], get_logging_name(),
                            get_unique_op_id(), parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id());
            } 
            else if (p1.get_dim() == 2) 
            {
              REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_FILL,
                            "Index space fill launch has intefering "
                            "region requirements 0 of point (%lld,%lld) and "
                            "region requirement 0 of point (%lld,%lld) of "
                            "%s (UID %lld) in parent task %s (UID %lld) are "
                            "interfering.", p1[0], p1[1], p2[0], p2[1],
                            get_logging_name(), get_unique_op_id(),
                            parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id());
            } 
            else if (p1.get_dim() == 3) 
            {
              REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_FILL,
                            "Index space fill launch has intefering "
                            "region requirements 0 of point (%lld,%lld,%lld)"
                            " and region requirement 0 of point "
                            "(%lld,%lld,%lld) of %s (UID %lld) in parent "
                            "task %s (UID %lld) are interfering.",
                            p1[0], p1[1], p1[2], p2[0], p2[1], p2[2],
                            get_logging_name(), get_unique_op_id(),
                            parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id());
            }
            assert(false);
          }
        }
      }
    }
#endif

    //--------------------------------------------------------------------------
    const ProjectionInfo* IndexFillOp::get_projection_info(void)
    //--------------------------------------------------------------------------
    {
      return &projection_info;
    }

    ///////////////////////////////////////////////////////////// 
    // Point Fill Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointFillOp::PointFillOp(Runtime *rt)
      : FillOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointFillOp::PointFillOp(const PointFillOp &rhs)
      : FillOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PointFillOp::~PointFillOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointFillOp& PointFillOp::operator=(const PointFillOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PointFillOp::initialize(IndexFillOp *own, const DomainPoint &p)
    //--------------------------------------------------------------------------
    {
      // Initialize the operation
      initialize_operation(own->get_context(), false/*track*/, 1/*regions*/); 
      index_point = p;
      owner = own;
      execution_fence_event = own->get_execution_fence_event();
      // From Fill
      requirement        = owner->get_requirement();
      grants             = owner->grants;
      wait_barriers      = owner->wait_barriers;
      arrive_barriers    = owner->arrive_barriers;
      parent_task        = owner->parent_task;
      map_id             = owner->map_id;
      tag                = owner->tag;
      // From FillOp
      parent_req_index   = owner->parent_req_index;
      restrict_info      = owner->restrict_info;
      true_guard         = owner->true_guard;
      false_guard        = owner->false_guard;
      future             = owner->future;
      value_size         = owner->value_size;
      if (value_size > 0)
      {
        value = malloc(value_size);
        memcpy(value, owner->value, value_size);
      }
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_index_point(owner->get_unique_op_id(), unique_op_id, p);
    }

    //--------------------------------------------------------------------------
    void PointFillOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_fill();
      owner = NULL;
    }

    //--------------------------------------------------------------------------
    void PointFillOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_fill();
      runtime->free_point_fill_op(this);
    }

    //--------------------------------------------------------------------------
    void PointFillOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointFillOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointFillOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointFillOp::launch(const std::set<RtEvent> &index_preconditions)
    //--------------------------------------------------------------------------
    {
      version_info = owner->version_info;
      // Perform the version info
      std::set<RtEvent> preconditions(index_preconditions);
      const UniqueID logical_context_uid = parent_ctx->get_context_uid();
      perform_projection_version_analysis(owner->projection_info, 
          owner->get_requirement(), requirement, 0/*idx*/, 
          logical_context_uid, version_info, preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
      // We can also mark this as having our resolved any predication
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    void PointFillOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      version_info.clear();
      // Tell our owner that we are done
      owner->handle_point_commit();
      // Don't commit this operation until we've reported our profiling
      // Out index owner will deactivate the operation
      commit_operation(false/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    const DomainPoint& PointFillOp::get_domain_point(void) const
    //--------------------------------------------------------------------------
    {
      return index_point;
    }

    //--------------------------------------------------------------------------
    void PointFillOp::set_projection_result(unsigned idx, LogicalRegion result)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      requirement.region = result;
      requirement.handle_type = SINGULAR;
    }

    //--------------------------------------------------------------------------
    const ProjectionInfo* PointFillOp::get_projection_info(void)
    //--------------------------------------------------------------------------
    {
      return owner->get_projection_info();
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
    PhysicalRegion AttachOp::initialize(TaskContext *ctx,
                          const AttachLauncher &launcher, bool check_privileges)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 1/*regions*/, 
                           launcher.static_dependences);
      resource = launcher.resource;
      switch (resource)
      {
        case EXTERNAL_POSIX_FILE:
          {
            if (launcher.file_fields.empty()) 
            {
              REPORT_LEGION_WARNING(LEGION_WARNING_FILE_ATTACH_OPERATION,
                              "FILE ATTACH OPERATION ISSUED WITH NO "
                              "FIELD MAPPINGS IN TASK %s (ID %lld)! DID YOU "
                              "FORGET THEM?!?", parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
            }
            file_name = strdup(launcher.file_name);
            // Construct the region requirement for this task
            requirement = RegionRequirement(launcher.handle, WRITE_DISCARD, 
                                            EXCLUSIVE, launcher.parent);
            for (std::vector<FieldID>::const_iterator it = 
                  launcher.file_fields.begin(); it != 
                  launcher.file_fields.end(); it++)
              requirement.add_field(*it);
            file_mode = launcher.mode;       
            break;
          }
        case EXTERNAL_HDF5_FILE:
          {
            if (launcher.field_files.empty()) 
            {
              REPORT_LEGION_WARNING(LEGION_WARNING_HDF5_ATTACH_OPERATION,
                            "HDF5 ATTACH OPERATION ISSUED WITH NO "
                            "FIELD MAPPINGS IN TASK %s (ID %lld)! DID YOU "
                            "FORGET THEM?!?", parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id());
            }
            file_name = strdup(launcher.file_name);
            // Construct the region requirement for this task
            requirement = RegionRequirement(launcher.handle, WRITE_DISCARD, 
                                            EXCLUSIVE, launcher.parent);
            for (std::map<FieldID,const char*>::const_iterator it = 
                  launcher.field_files.begin(); it != 
                  launcher.field_files.end(); it++)
            {
              requirement.add_field(it->first);
              field_map[it->first] = strdup(it->second);
            }
            file_mode = launcher.mode;
            break;
          }
        case EXTERNAL_INSTANCE:
          {
            layout_constraint_set = launcher.constraints;  
            const std::set<FieldID> &fields = launcher.privilege_fields;
            if (fields.empty())
              REPORT_LEGION_WARNING(LEGION_WARNING_EXTERNAL_ATTACH_OPERATION,
                            "EXTERNAL ARRAY ATTACH OPERATION ISSUED WITH NO "
                            "PRIVILEGE FIELDS IN TASK %s (ID %lld)! DID YOU "
                            "FORGET THEM?!?", parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id())
            if (!layout_constraint_set.pointer_constraint.is_valid)
              REPORT_LEGION_ERROR(ERROR_ATTACH_OPERATION_MISSING_POINTER,
                            "EXTERNAL ARRAY ATTACH OPERATION ISSUED WITH NO "
                            "POINTER CONSTRAINT IN TASK %s (ID %lld)!",
                            parent_ctx->get_task_name(), 
                            parent_ctx->get_unique_id())
            // Construct the region requirement for this task
            requirement = RegionRequirement(launcher.handle, WRITE_DISCARD, 
                                            EXCLUSIVE, launcher.parent);
            for (std::set<FieldID>::const_iterator it = 
                  fields.begin(); it != fields.end(); it++)
              requirement.add_field(*it);
            break;
          }
        default:
          assert(false); // should never get here
      }
      region = PhysicalRegion(new PhysicalRegionImpl(requirement,
                              completion_event, true/*mapped*/, ctx,
                              0/*map id*/, 0/*tag*/, false/*leaf*/, 
                              false/*virtual mapped*/, runtime));
      if (check_privileges)
        check_privilege();
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_attach_operation(parent_ctx->get_unique_id(),
                                        unique_op_id);
      return region;
    }

    //--------------------------------------------------------------------------
    void AttachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      file_name = NULL;
      external_instance = NULL;
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
      field_pointers_map.clear();
      region = PhysicalRegion();
      privilege_path.clear();
      version_info.clear();
      restrict_info.clear();
      map_applied_conditions.clear();
      runtime->free_attach_op(this);
    }

    //--------------------------------------------------------------------------
    const char* AttachOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[ATTACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind AttachOp::get_operation_kind(void) const
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
    void AttachOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent index
      compute_parent_index();
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
      { 
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
    void AttachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      ProjectionInfo projection_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   restrict_info, 
                                                   version_info,
                                                   projection_info,
                                                   privilege_path);
      // If we have any restriction on ourselves, that is very bad
      if (restrict_info.has_restrictions())
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_FILE_ATTACHMENT,
                      "Illegal file attachment for file %s performed on "
                      "logical region (%x,%x,%x) which is under "
                      "restricted coherence! User coherence must first "
                      "be acquired with an acquire operation before "
                      "attachment can be performed.", file_name,
                      requirement.region.index_space.id,
                      requirement.region.field_space.id,
                      requirement.region.tree_id)
      switch (resource)
      {
        case EXTERNAL_POSIX_FILE:
        case EXTERNAL_HDF5_FILE:
          {
            external_instance = 
              runtime->forest->create_external_instance(this, requirement, 
                                              requirement.instance_fields);
            break;
          }
        case EXTERNAL_INSTANCE:
          {
            external_instance = 
              runtime->forest->create_external_instance(this, requirement,
                        layout_constraint_set.field_constraint.field_set);
            break;
          }
        default:
          assert(false);
      }
      external_instance->memory_manager->record_created_instance(
        external_instance, false/*acquire*/, 0/*mapper id*/, 
        parent_ctx->get_executing_processor(), 0/*priority*/, false/*remote*/);
      // Tell the parent that we added the restriction
      parent_ctx->add_restriction(this, external_instance, requirement);
    }

    //--------------------------------------------------------------------------
    void AttachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;  
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   privilege_path,
                                                   version_info,
                                                   preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void AttachOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      InstanceRef result = runtime->forest->attach_external(this, 0/*idx*/,
                                                        requirement,
                                                        external_instance,
                                                        version_info,
                                                        map_applied_conditions);
#ifdef DEBUG_LEGION
      assert(result.has_ref());
#endif
      version_info.apply_mapping(map_applied_conditions);
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
      version_info.clear();
      commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void AttachOp::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance AttachOp::create_instance(IndexSpaceNode *node,
                                         const std::vector<FieldID> &field_set,
                                         const std::vector<size_t> &sizes, 
                                               LayoutConstraintSet &constraints,
                                               ApEvent &ready_event)
    //--------------------------------------------------------------------------
    {
      PhysicalInstance result = PhysicalInstance::NO_INST;
      switch (resource)
      {
        case EXTERNAL_POSIX_FILE:
          {
            std::vector<Realm::FieldID> field_ids(field_set.size());
            unsigned idx = 0;
            for (std::vector<FieldID>::const_iterator it = 
                  field_set.begin(); it != field_set.end(); it++, idx++)
            {
	      field_ids[idx] = *it;
            }
            result = node->create_file_instance(file_name, field_ids, sizes, 
                                                file_mode, ready_event);
            constraints.specialized_constraint = 
              SpecializedConstraint(GENERIC_FILE_SPECIALIZE);           
            constraints.field_constraint = 
              FieldConstraint(requirement.privilege_fields, 
                              false/*contiguous*/, false/*inorder*/);
            constraints.memory_constraint = 
              MemoryConstraint(result.get_location().kind());
            // TODO: Fill in the other constraints: 
            // OrderingConstraint, SplittingConstraints DimensionConstraints,
            // AlignmentConstraints, OffsetConstraints
            break;
          }
        case EXTERNAL_HDF5_FILE:
          {
            // First build the set of field paths
            std::vector<Realm::FieldID> field_ids(field_map.size());
            std::vector<const char*> field_files(field_map.size());
            unsigned idx = 0;
            for (std::map<FieldID,const char*>::const_iterator it = 
                  field_map.begin(); it != field_map.end(); it++, idx++)
            {
	      field_ids[idx] = it->first;
              field_files[idx] = it->second;
            }
            // Now ask the low-level runtime to create the instance
            result = node->create_hdf5_instance(file_name,
					field_ids, sizes, field_files,
                                        (file_mode == LEGION_FILE_READ_ONLY),
                                        ready_event);
            constraints.specialized_constraint = 
              SpecializedConstraint(HDF5_FILE_SPECIALIZE);
            constraints.field_constraint = 
              FieldConstraint(requirement.privilege_fields, 
                              false/*contiguous*/, false/*inorder*/);
            constraints.memory_constraint = 
              MemoryConstraint(result.get_location().kind());
            // TODO: Fill in the other constraints: 
            // OrderingConstraint, SplittingConstraints DimensionConstraints,
            // AlignmentConstraints, OffsetConstraints
            break;
          }
        case EXTERNAL_INSTANCE:
          {
            // Create the Instance Layout Generic object for realm
            Realm::InstanceLayoutConstraints realm_constraints;
            // Get some help from the instance builder to make this
            InstanceBuilder::convert_layout_constraints(layout_constraint_set,
                                          field_set, sizes, realm_constraints);
            const PointerConstraint &pointer = 
                                      layout_constraint_set.pointer_constraint;
#ifdef DEBUG_LEGION
            assert(pointer.is_valid);
#endif
            Realm::InstanceLayoutGeneric *ilg = 
              node->create_layout(realm_constraints, 
                  layout_constraint_set.ordering_constraint);
            result = node->create_external_instance(pointer.memory, pointer.ptr,
                                                    ilg, ready_event);
            constraints = layout_constraint_set;
            constraints.specialized_constraint = 
              SpecializedConstraint(NORMAL_SPECIALIZE);
            break;
          }
        default:
          assert(false);
      }
      if (Runtime::legion_spy_enabled)
      {
        // We always need a unique ready event for Legion Spy
        if (!ready_event.exists())
        {
          ApUserEvent rename_ready = Runtime::create_ap_user_event();
          Runtime::trigger_event(rename_ready);
          ready_event = rename_ready;
        }
        for (std::set<FieldID>::const_iterator it = 
              requirement.privilege_fields.begin(); it !=
              requirement.privilege_fields.end(); it++)
          LegionSpy::log_mapping_decision(unique_op_id, 0/*idx*/, 
                                          *it, ready_event);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void AttachOp::check_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field = AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, then check the privileges with the parent context
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, bad_index);
      switch (et)
      {
          // Not there is no such things as bad privileges for
          // acquires and releases because they are controlled by the runtime
        case NO_ERROR:
        case ERROR_BAD_REGION_PRIVILEGES:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            REPORT_LEGION_ERROR(ERROR_REQUIREMENTS_INVALID_REGION,
                             "Requirements for invalid region handle "
                             "(%x,%d,%d) for attach operation "
                             "(ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             unique_op_id)
            break;
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) ||
              (requirement.handle_type == REG_PROJECTION) ? 
                requirement.region.field_space :
                requirement.partition.field_space;
            REPORT_LEGION_ERROR(ERROR_FIELD_NOT_VALID,
                             "Field %d is not a valid field of field "
                             "space %d for attach operation (ID %lld)",
                             bad_field, sp.id, unique_op_id)
            break;
          }
        case ERROR_INVALID_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_PRIVILEGE,
                             "Instance field %d is not one of the "
                             "privilege fields for attach operation "
                             "(ID %lld)",
                             bad_field, unique_op_id)
            break;
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_DUPLICATE,
                             "Instance field %d is a duplicate for "
                             "attach operation (ID %lld)",
                             bad_field, unique_op_id)
            break;
          }
        case ERROR_BAD_PARENT_REGION:
          {
            if (bad_index > 0) 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ATTACH,
                               "Parent task %s (ID %lld) of attach operation "
                               "(ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) "
                               "as a parent of region requirement because "
                               "no 'parent' region had that name.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id)
            else if (bad_field == AUTO_GENERATE_ID) 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ATTACH,
                               "Parent task %s (ID %lld) of attach operation "
                               "(ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) "
                               "as a parent of region requirement because "
                               "parent requirement %d did not have "
                               "sufficient privileges.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id, bad_index)
            else 
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ATTACH,
                               "Parent task %s (ID %lld) of attach operation "
                               "(ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) "
                               "as a parent of region requirement because "
                               "region requirement %d was missing field %d.",
                               parent_ctx->get_task_name(),
                               parent_ctx->get_unique_id(),
                               unique_op_id,
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id,
                               bad_index, bad_field)
            break;
          }
        case ERROR_BAD_REGION_PATH:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_NOT_SUBREGION,
                             "Region (%x,%x,%x) is not a "
                             "sub-region of parent region "
                             "(%x,%x,%x) for region requirement of attach "
                             "operation (ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             requirement.parent.index_space.id,
                             requirement.parent.field_space.id,
                             requirement.parent.tree_id,
                             unique_op_id)
            break;
          }
        case ERROR_BAD_REGION_TYPE:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_REQUIREMENT_ATTACH,
                             "Region requirement of attach operation "
                             "(ID %lld) cannot find privileges for field "
                             "%d in parent task",
                             unique_op_id, bad_field)
            break;
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
        REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ATTACH,
                         "Parent task %s (ID %lld) of attach "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id)
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
    void DetachOp::initialize_detach(TaskContext *ctx, 
                                     PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      // No need to check privileges because we never would have been
      // able to attach in the first place anyway.
      requirement = region.impl->get_requirement();
      // Delay getting a reference until trigger_mapping().  This means we
      //  have to keep region
      if (!region.is_valid())
        region.wait_until_valid();
      this->region = region; 
      // Check to see if this is a valid detach operation
      if (!region.impl->is_external_region())
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_DETACH_OPERATION,
          "Illegal detach operation (ID %lld) performed in "
                      "task %s (ID %lld). Detach was performed on an region "
                      "that had not previously been attached.",
                      get_unique_op_id(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_detach_operation(parent_ctx->get_unique_id(),
                                        unique_op_id);
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
    const char* DetachOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DETACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind DetachOp::get_operation_kind(void) const
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
    void DetachOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent index
      compute_parent_index();
      initialize_privilege_path(privilege_path, requirement);
      if (Runtime::legion_spy_enabled)
      { 
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
    void DetachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Before we do our dependence analysis, we can remove the 
      // restricted coherence on the logical region
      ProjectionInfo projection_info;
      // Tell the parent that we've release the restriction
      parent_ctx->remove_restriction(this, requirement);
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement, 
                                                   restrict_info,
                                                   version_info,
                                                   projection_info,
                                                   privilege_path);
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement, 
                                                   privilege_path,
                                                   version_info,
                                                   preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_mapping(void)
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
      if (!inst_manager->is_external_instance())
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_DETACH_OPERATION,
                      "Illegal detach operation on a physical region which "
                      "was not attached!")
      std::set<RtEvent> applied_conditions;
      ApEvent detach_event = 
        runtime->forest->detach_external(requirement, this, 0/*idx*/, 
                                     version_info,reference,applied_conditions);
      version_info.apply_mapping(applied_conditions);
      if (!applied_conditions.empty())
        complete_mapping(Runtime::merge_events(applied_conditions));
      else
        complete_mapping();

      // Now remove the valid reference added by the attach operation
      manager->memory_manager->set_garbage_collection_priority(manager,
        0, parent_ctx->get_executing_processor(), GC_MAX_PRIORITY);

      Runtime::trigger_event(completion_event, detach_event);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(detach_event));
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
      version_info.clear();
      commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void DetachOp::compute_parent_index(void)
    //--------------------------------------------------------------------------
    {
      int parent_index = parent_ctx->find_parent_region_req(requirement);
      if (parent_index < 0)
        REPORT_LEGION_ERROR(ERROR_PARENT_TASK_DETACH,
                         "Parent task %s (ID %lld) of detach "
                               "operation (ID %lld) does not have a region "
                               "requirement for region (%x,%x,%x) as a parent",
                               parent_ctx->get_task_name(), 
                               parent_ctx->get_unique_id(),
                               unique_op_id, 
                               requirement.region.index_space.id,
                               requirement.region.field_space.id, 
                               requirement.region.tree_id)
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
    Future TimingOp::initialize(TaskContext *ctx,const TimingLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
      measurement = launcher.measurement;
      preconditions = launcher.preconditions;
      result = Future(new FutureImpl(runtime, true/*register*/,
                  runtime->get_available_distributed_id(),
                  runtime->address_space, this));
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_timing_operation(ctx->get_unique_id(), unique_op_id);
        DomainPoint empty_point;
        LegionSpy::log_future_creation(unique_op_id, 
            result.impl->get_ready_event(), empty_point);
        for (std::set<Future>::const_iterator it = preconditions.begin();
              it != preconditions.end(); it++)
        {
          if ((it->impl != NULL) && it->impl->get_ready_event().exists())
            LegionSpy::log_future_use(unique_op_id,it->impl->get_ready_event());
        }
      }
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
      preconditions.clear();
      result = Future();
      runtime->free_timing_op(this);
    }

    //--------------------------------------------------------------------------
    const char* TimingOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TIMING_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TimingOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TIMING_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TimingOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      for (std::set<Future>::const_iterator it = preconditions.begin();
            it != preconditions.end(); it++)
      {
        if (it->impl != NULL)
          it->impl->register_dependence(this);
      }
    }

    //--------------------------------------------------------------------------
    void TimingOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      complete_mapping();
      std::set<ApEvent> pre_events;
      for (std::set<Future>::const_iterator it = preconditions.begin();
            it != preconditions.end(); it++)
      {
        if ((it->impl != NULL) && !it->impl->ready_event.has_triggered())
          pre_events.insert(it->impl->get_ready_event());
      }
      // Also make sure we wait for any execution fences that we have
      if (execution_fence_event.exists())
        pre_events.insert(execution_fence_event);
      RtEvent wait_on;
      if (!pre_events.empty())
        wait_on = Runtime::protect_event(Runtime::merge_events(pre_events));
      if (wait_on.exists() && !wait_on.has_triggered())
      {
        DeferredExecuteArgs args;
        args.proxy_this = this;
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_DEFERRED_PRIORITY,
                                         this, wait_on);
      }
      else
        deferred_execute();
    }

    //--------------------------------------------------------------------------
    void TimingOp::deferred_execute(void)
    //--------------------------------------------------------------------------
    {
      switch (measurement)
      {
        case MEASURE_SECONDS:
          {
            double value = Realm::Clock::current_time();
            result.impl->set_result(&value, sizeof(value), false);
            break;
          }
        case MEASURE_MICRO_SECONDS:
          {
            long long value = Realm::Clock::current_time_in_microseconds();
            result.impl->set_result(&value, sizeof(value), false);
            break;
          }
        case MEASURE_NANO_SECONDS:
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

