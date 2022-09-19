/* Copyright 2022 Stanford University, NVIDIA Corporation
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
    // Provenance
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void Provenance::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      const size_t strlen = provenance.length();
      rez.serialize(strlen);
      if (strlen > 0)
        rez.serialize(provenance.c_str(), strlen);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Provenance::serialize_null(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(0);
    }

    //--------------------------------------------------------------------------
    /*static*/ Provenance* Provenance::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t length;
      derez.deserialize(length);
      if (length > 0)
      {
        Provenance *result = new Provenance(derez.get_current_pointer(),length);
        derez.advance_pointer(length);
        return result;
      }
      else
        return NULL;
    }

    /////////////////////////////////////////////////////////////
    // Operation 
    /////////////////////////////////////////////////////////////

    const char *const 
      Operation::op_names[Operation::LAST_OP_KIND] = OPERATION_NAMES;

    //--------------------------------------------------------------------------
    Operation::Operation(Runtime *rt)
      : runtime(rt), gen(0), unique_op_id(0), context_index(0), 
        outstanding_mapping_references(0), activated(false), hardened(false), 
        parent_ctx(NULL), mapping_tracker(NULL), commit_tracker(NULL)
    //--------------------------------------------------------------------------
    {
      if (!runtime->resilient_mode)
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
#ifdef DEBUG_LEGION
      assert(!activated);
#endif
      // Get a new unique ID for this operation
      unique_op_id = runtime->get_unique_operation_id();
      context_index = 0;
      outstanding_mapping_references = 0;
      activated = true;
      prepipelined = false;
      mapped = false;
      executed = false;
      resolved = false;
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
      completion_event = Runtime::create_ap_user_event(NULL);
      if (runtime->resilient_mode)
        commit_event = Runtime::create_rt_user_event(); 
      execution_fence_event = ApEvent::NO_AP_EVENT;
      trace = NULL;
      tracing = false;
      trace_local_id = (unsigned)-1;
      must_epoch = NULL;
      provenance = NULL;
#ifdef DEBUG_LEGION
      assert(mapped_event.exists());
      assert(resolved_event.exists());
      assert(completion_event.exists());
      if (runtime->resilient_mode)
        assert(commit_event.exists());
#endif 
    }
    
    //--------------------------------------------------------------------------
    void Operation::deactivate_operation(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(activated);
#endif
      activated = false;
      // Generation is bumped when we committed
      incoming.clear();
      outgoing.clear();
      unverified_regions.clear();
      verify_regions.clear();
      logical_records.clear();
      if (mapping_tracker != NULL)
      {
        delete mapping_tracker;
        mapping_tracker = NULL;
      }
      if (commit_tracker != NULL)
      {
        delete commit_tracker;
        commit_tracker = NULL;
      }
      if (!mapped)
        Runtime::trigger_event(mapped_event);
      if (!resolved)
        Runtime::trigger_event(resolved_event);
      if (need_completion_trigger)
        Runtime::trigger_event(NULL, completion_event);
      if (!commit_event.has_triggered())
        Runtime::trigger_event(commit_event);
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
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
      if ((req.handle_type == LEGION_SINGULAR_PROJECTION) || 
          (req.handle_type == LEGION_REGION_PROJECTION))
      {
        if (!req.region.exists())
          return;
        runtime->forest->initialize_path(req.region.get_index_space(),
                                         req.parent.get_index_space(), path);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(req.handle_type == LEGION_PARTITION_PROJECTION);
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
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
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
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      runtime->forest->initialize_path(req.region.get_index_space(),
                                       start_node.get_index_partition(), path);
    }

    //--------------------------------------------------------------------------
    void Operation::set_tracking_parent(size_t index)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!track_parent);
#endif
      track_parent = true;
      context_index = index;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_unordered_operation(parent_ctx->get_unique_id(),
                                           unique_op_id, context_index);
    }

    //--------------------------------------------------------------------------
    void Operation::set_trace(LegionTrace *t,
                              const std::vector<StaticDependence> *dependences,
                              const LogicalTraceInfo *trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace == NULL);
      assert(t != NULL);
#endif
      trace = t; 
      tracing = trace->initialize_op_tracing(this, dependences, trace_info);
    }

    //--------------------------------------------------------------------------
    void Operation::set_trace_local_id(unsigned id)
    //--------------------------------------------------------------------------
    {
      trace_local_id = id;
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
      assert(r.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      r.parent = r.region;
      r.prop = LEGION_EXCLUSIVE;
      // If we're doing a write discard, then we can add read privileges
      // inside our task since it is safe to read what we wrote
      if (HAS_WRITE_DISCARD(r))
      {
        r.privilege |= (LEGION_READ_PRIV | LEGION_REDUCE_PRIV);
        // Then remove any discard masks from the privileges
        r.privilege &= ~LEGION_DISCARD_MASK;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent Operation::release_nonempty_acquired_instances(RtEvent perform,
                        std::map<PhysicalManager*,unsigned> &acquired_instances)
    //--------------------------------------------------------------------------
    {
      if (perform.exists() && !perform.has_triggered())
      {
        std::vector<std::pair<PhysicalManager*,unsigned> > *to_release = NULL;
        for (std::map<PhysicalManager*,unsigned>::iterator it = 
              acquired_instances.begin(); it != acquired_instances.end(); )
        {
          if (it->first->instance_footprint > 0)
          {
            if (to_release == NULL)
              to_release = 
                new std::vector<std::pair<PhysicalManager*,unsigned> >();
            to_release->push_back(std::make_pair(it->first, it->second));
            std::map<PhysicalManager*,unsigned>::iterator to_delete = it++;
            acquired_instances.erase(to_delete);
          }
          else
            it++;
        }
        if (to_release != NULL)
        {
          DeferReleaseAcquiredArgs args(this, to_release);
          return runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_DEFERRED_PRIORITY, perform); 
        }
        else
          return perform;
      }
      for (std::map<PhysicalManager*,unsigned>::iterator it = 
            acquired_instances.begin(); it != acquired_instances.end(); )
      {
        if (it->first->instance_footprint > 0)
        {
          if (it->first->remove_base_valid_ref(MAPPING_ACQUIRE_REF, 
                                NULL/*mutator*/, it->second))
            delete it->first;
          std::map<PhysicalManager*,unsigned>::iterator to_delete = it++;
          acquired_instances.erase(to_delete);
        }
        else
          it++;
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::release_acquired_instances(
                        std::map<PhysicalManager*,unsigned> &acquired_instances)
    //--------------------------------------------------------------------------
    {
      for (std::map<PhysicalManager*,unsigned>::iterator it = 
            acquired_instances.begin(); it != acquired_instances.end(); it++)
        if (it->first->remove_base_valid_ref(MAPPING_ACQUIRE_REF, 
                              NULL/*mutator*/, it->second))
          delete it->first;
      acquired_instances.clear();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::handle_deferred_release(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferReleaseAcquiredArgs *dargs = 
        (const DeferReleaseAcquiredArgs*)args;
      for (std::vector<std::pair<PhysicalManager*,unsigned> >::const_iterator 
            it = dargs->instances->begin(); it != dargs->instances->end(); it++)
      {
        if (it->first->remove_base_valid_ref(MAPPING_ACQUIRE_REF, 
                                    NULL/*mutator*/, it->second))
          delete it->first;
      }
      delete dargs->instances;
    }

    //--------------------------------------------------------------------------
    void Operation::initialize_operation(InnerContext *ctx, bool track, 
                                         unsigned regs/*= 0*/,
                                         const char *prov/*= NULL*/,
                      const std::vector<StaticDependence> *dependences/*=NULL*/)
    //--------------------------------------------------------------------------
    {
      if ((prov != NULL) && (strlen(prov) > 0))
        initialize_operation(ctx, track, regs,new Provenance(prov),dependences);
      else
        initialize_operation(ctx, track, regs, (Provenance*)NULL, dependences);
    }

    //--------------------------------------------------------------------------
    void Operation::initialize_operation(InnerContext *ctx, bool track, 
                                         unsigned regs/*= 0*/,
                                         Provenance *prov/*= NULL*/,
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
      provenance = prov;
      if (provenance != NULL)
      {
        provenance->add_reference();
        if (runtime->legion_spy_enabled)
          LegionSpy::log_operation_provenance(unique_op_id,
                                  prov->provenance.c_str());
      }
      if (runtime->profiler != NULL)
        runtime->profiler->register_operation(this);
    }

    //--------------------------------------------------------------------------
    void Operation::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      // should be overwridden by inheriting classes
      assert(false);
    }

    //--------------------------------------------------------------------------
    RtEvent Operation::execute_prepipeline_stage(GenerationID generation, 
                                                 bool from_logical_analysis)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock op(op_lock);
#ifdef DEBUG_LEGION
        assert(generation <= gen);
#endif
        if (generation < gen)
          return RtEvent::NO_RT_EVENT;
        // Check to see if we've already started the analysis
        if (prepipelined)
        {
          // Someone else already started, figure out if we need to wait
          if (from_logical_analysis)
            return prepipelined_event;
          else
            return RtEvent::NO_RT_EVENT;
        }
        else
        {
          // We got here first, mark that we're doing it
          prepipelined = true;
          // If we're not the logical analysis, make a wait event in
          // case the logical analysis comes along and needs to wait
          if (!from_logical_analysis)
          {
#ifdef DEBUG_LEGION
            assert(!prepipelined_event.exists());
#endif
            prepipelined_event = Runtime::create_rt_user_event(); 
          }
        }
      }
      trigger_prepipeline_stage();
      // Trigger any guard events we might have
      if (!from_logical_analysis)
      {
        AutoLock op(op_lock);
#ifdef DEBUG_LEGION
        assert(prepipelined_event.exists());
#endif
        Runtime::trigger_event(prepipelined_event);
        prepipelined_event = RtUserEvent::NO_RT_USER_EVENT;
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void Operation::execute_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Check to make sure our prepipeline stage is done if we have one
      if (has_prepipeline_stage())
      {
        RtEvent wait_on = execute_prepipeline_stage(gen, true/*need wait*/);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      // Always wrap this call with calls to begin/end dependence analysis
      begin_dependence_analysis();
      trigger_dependence_analysis();
      end_dependence_analysis();
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
      // The execution stage only gets invoked if you call it explicitly
      // We do so here to ensure that we call the complete execution method
      trigger_execution();
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      // Mark that we finished execution
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
    void Operation::select_sources(const unsigned index,
                                   const InstanceRef &target,
                                   const InstanceSet &sources,
                                   std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      assert(false);
    }

    //--------------------------------------------------------------------------
    CollectiveManager* Operation::find_or_create_collective_instance(
                                  MappingCallKind call_kind, unsigned index,
                                  const LayoutConstraintSet &constraints,
                                  const std::vector<LogicalRegion> &regions,
                                  Memory::Kind kind, size_t *footprint,
                                  LayoutConstraintKind *unsat_kind,
                                  unsigned *unsat_index, 
                                  DomainPoint &collective_point)
    //--------------------------------------------------------------------------
    {
      // we do not do anything here in this case
      return NULL;
    }

    //--------------------------------------------------------------------------
    bool Operation::finalize_collective_instance(MappingCallKind call,
                                              unsigned total_calls, bool succes)
    //--------------------------------------------------------------------------
    {
      // should only be called for inherited types
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void Operation::report_total_collective_instance_calls(
                              MappingCallKind mapper_call, unsigned total_calls)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here for this case
    }

    //--------------------------------------------------------------------------
    void Operation::report_uninitialized_usage(const unsigned index,
                                 LogicalRegion handle, const RegionUsage usage, 
                                 const char *field_string, RtUserEvent reported)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reported.exists());
      assert(!reported.has_triggered());
#endif
      // Read-only or reduction usage of uninitialized data is always an error
      if (IS_READ_ONLY(usage))
        REPORT_LEGION_ERROR(ERROR_UNINITIALIZED_USE,
                      "Region requirement %d of operation %s (UID %lld) in "
                      "parent task %s (UID %lld) is using uninitialized data "
                      "for field(s) %s of logical region (%d,%d,%d) with "
                      "read-only privileges", index, get_logging_name(), 
                      get_unique_op_id(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id(), field_string, 
                      handle.get_index_space().get_id(),
                      handle.get_field_space().get_id(), 
                      handle.get_tree_id())
      else if (IS_REDUCE(usage))
        REPORT_LEGION_ERROR(ERROR_UNINITIALIZED_USE,
                      "Region requirement %d of operation %s (UID %lld) in "
                      "parent task %s (UID %lld) is using uninitialized data "
                      "for field(s) %s of logical region (%d,%d,%d) with "
                      "reduction privileges", index, get_logging_name(), 
                      get_unique_op_id(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id(), field_string, 
                      handle.get_index_space().get_id(),
                      handle.get_field_space().get_id(), 
                      handle.get_tree_id())
      else // Read-write usage is just a warning
        REPORT_LEGION_WARNING(LEGION_WARNING_UNINITIALIZED_USE,
                      "Region requirement %d of operation %s (UID %lld) in "
                      "parent task %s (UID %lld) is using uninitialized data "
                      "for field(s) %s of logical region (%d,%d,%d)", index, 
                      get_logging_name(), get_unique_op_id(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id(),
                      field_string, handle.get_index_space().get_id(),
                      handle.get_field_space().get_id(), 
                      handle.get_tree_id())
      Runtime::trigger_event(reported);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,unsigned>* 
                                     Operation::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      // should only be called for inherited types
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    void Operation::update_atomic_locks(const unsigned index,
                                        Reservation lock, bool exclusive)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      assert(false);
    }

    //--------------------------------------------------------------------------
    /*static*/ ApEvent Operation::merge_sync_preconditions(
                                 const TraceInfo &trace_info,
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
      return Runtime::merge_events(&trace_info, wait_events);
    }

    //--------------------------------------------------------------------------
    void Operation::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &requests, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Operation::handle_profiling_response(const ProfilingResponseBase *base,
                                       const Realm::ProfilingResponse &response,
                                       const void *orig, size_t orig_length)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Operation::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      assert(false);
    }

    //--------------------------------------------------------------------------
    ApEvent Operation::compute_init_precondition(const TraceInfo &info)
    //--------------------------------------------------------------------------
    {
      return execution_fence_event;
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
    void Operation::enqueue_ready_operation(RtEvent wait_on/*=Event::NO_EVENT*/,
                           LgPriority priority/*= LG_THROUGHPUT_WORK_PRIORITY*/)
    //--------------------------------------------------------------------------
    {
      TriggerOpArgs args(this);
      runtime->issue_runtime_meta_task(args, priority, wait_on); 
    }

    //--------------------------------------------------------------------------
    void Operation::complete_mapping(RtEvent wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!mapped);
#endif
      mapped = true;
      Runtime::trigger_event(mapped_event, wait_on);
    }

    //--------------------------------------------------------------------------
    void Operation::complete_execution(RtEvent wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
      if (wait_on.exists() && !wait_on.has_triggered())
      {
        // We have to defer the execution of this operation
        parent_ctx->add_to_deferred_execution_queue(this, wait_on);
        return;
      }
      // Tell our parent context that we are done mapping
      // It's important that this is done before we mark that we
      // are executed to avoid race conditions
      if (track_parent)
        parent_ctx->register_child_executed(this);
#ifdef DEBUG_LEGION
      assert(!executed);
#endif
      executed = true;
      // Now see if we are ready to complete this operation
      if (!mapped_event.has_triggered() || !resolved_event.has_triggered())
      {
        RtEvent trigger_pre = 
          Runtime::merge_events(mapped_event, resolved_event);
        if (trigger_pre.exists() && !trigger_pre.has_triggered())
          parent_ctx->add_to_trigger_completion_queue(this, trigger_pre);
        else
          trigger_complete();
      }
      else // Do the trigger now
        trigger_complete();
    }

    //--------------------------------------------------------------------------
    void Operation::resolve_speculation(RtEvent wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!resolved);
#endif
      resolved = true;
      Runtime::trigger_event(resolved_event, wait_on);
    }

    //--------------------------------------------------------------------------
    void Operation::complete_operation(RtEvent wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
      if (wait_on.exists() && !wait_on.has_triggered())
      {
        parent_ctx->add_to_deferred_completion_queue(this, wait_on);
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
        if ((!runtime->resilient_mode) || early_commit_request ||
            ((hardened && unverified_regions.empty())))
        {
          trigger_commit_invoked = true;
          need_trigger = true;
        }
        else if (outstanding_mapping_references == 0)
        {
          if (commit_tracker != NULL)
          {
            CommitDependenceTracker *tracker = commit_tracker;
            commit_tracker = NULL;
            need_trigger = tracker->issue_commit_trigger(this, runtime);
            delete tracker;
          }
          else
            need_trigger = true;
          if (need_trigger)
            trigger_commit_invoked = true;
        }
      }
      if (need_completion_trigger)
      {
        Runtime::trigger_event(NULL, completion_event);
        need_completion_trigger = false;
      }
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
        parent_ctx->add_to_deferred_commit_queue(this, wait_on, do_deactivate);
        return;
      }
      // Tell our parent context that we are committed
      // Do this before actually committing to avoid race conditions
      if (track_parent)
      {
        // Do a check here to make sure the completion event has triggered
        // before we record that this operation is commited. This is crucial
        // to ensuring that fence operations are working correctly in the
        // parent context. If not triggered, then defer this until it does.
        // Inner task completion also relies upon this to work correctly
        if (!completion_event.has_triggered_faultignorant())
        {
          const RtEvent safe = Runtime::protect_event(completion_event);
          if (safe.exists() && !safe.has_triggered())
          {
            parent_ctx->add_to_deferred_commit_queue(this, safe, do_deactivate);
            return;
          }
        }
        parent_ctx->register_child_commit(this);
      }
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
        // At this point we bumb the generation as we can never roll back
        // after we have committed the operation
        gen++;
      } 
      // Trigger the commit event
      if (runtime->resilient_mode)
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
      assert(mapping_tracker == NULL);
#endif
      // Make a dependence tracker
      mapping_tracker = new MappingDependenceTracker();
      // Register ourselves with our trace if there is one
      // This will also add any necessary dependences
      if (trace != NULL)
        trace->register_operation(this, gen);
      parent_ctx->invalidate_trace_cache(trace, this);

      // See if we have any fence dependences
      execution_fence_event = parent_ctx->register_implicit_dependences(this);
    }

    //--------------------------------------------------------------------------
    void Operation::end_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapping_tracker != NULL);
#endif
      // Cannot touch anything not on our stack after this call
      MappingDependenceTracker *tracker = mapping_tracker;
      mapping_tracker = NULL;
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
      assert(mapping_tracker != NULL);
#endif
      bool prune = target->perform_registration(target_gen, this, gen,
                                                registered_dependence,
                                                mapping_tracker,
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
        assert(mapping_tracker != NULL);
#endif
        prune = target->perform_registration(target_gen, this, gen,
                                                registered_dependence,
                                                mapping_tracker,
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
    void Operation::register_no_dependence(unsigned idx, Operation *target,
            GenerationID target_gen, unsigned target_idx, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tracing);
      assert(trace != NULL);
      assert(!is_internal_op());
#endif
      if (target == this)
        return;
      trace->record_no_dependence(target, target_gen, this, gen, 
                                  target_idx, idx, mask);
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
        if (our_gen == gen)
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
            if (runtime->resilient_mode)
            {
              if (commit_tracker == NULL)
                commit_tracker = new CommitDependenceTracker();
              commit_tracker->add_commit_dependence(other_commit_event);
            }
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
            if (commit_tracker != NULL)
            {
              CommitDependenceTracker *tracker = commit_tracker;
              commit_tracker = NULL;
              need_trigger = tracker->issue_commit_trigger(this,runtime);
              delete tracker;
            }
            else
              need_trigger = true;
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
    void Operation::record_logical_dependence(const LogicalUser &user)
    //--------------------------------------------------------------------------
    {
      // Record the advance operations separately, in many cases we don't
      // need to include them in our analysis of above users, but in the case
      // of creating new advance operations below in the tree we do
      logical_records.push_back(user);
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
    InnerContext* Operation::find_logical_context(unsigned index)
    //--------------------------------------------------------------------------
    {
      return parent_ctx->find_parent_logical_context(find_parent_index(index));
    }

    //--------------------------------------------------------------------------
    InnerContext* Operation::find_physical_context(unsigned index,
                                                   const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      return parent_ctx->find_parent_physical_context(
                find_parent_index(index), req.parent);
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
        PhysicalManager *manager = ref.get_physical_manager();
        if (!manager->has_visible_from(visible_filter))
          continue;
        input_valid.resize(next_index+1);
        MappingInstance &inst = input_valid[next_index++];
        inst = ref.get_mapping_instance();
      }
    }

    //--------------------------------------------------------------------------
    void Operation::compute_ranking(MapperManager *mapper,
                              const std::deque<MappingInstance> &output,
                              const InstanceSet &sources,
                              std::vector<unsigned> &ranking) const
    //--------------------------------------------------------------------------
    {
      ranking.reserve(output.size());
      for (std::deque<MappingInstance>::const_iterator it = 
            output.begin(); it != output.end(); it++)
      {
        const InstanceManager *manager = it->impl;
        bool found = false;
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          if (manager == sources[idx].get_manager())
          {
            found = true;
            ranking.push_back(idx);
            break;
          }
        }
        // Ignore any instances which are not in the original set of sources
        if (!found)
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_INVALID_INSTANCE,
              "Ignoring invalid instance output from mapper %s for "
              "select copy sources call on Operation %s (UID %lld)",
              mapper->get_mapper_name(), get_logging_name(), get_unique_op_id())
      }
    }

    //--------------------------------------------------------------------------
    void Operation::pack_remote_operation(Serializer &rez,AddressSpaceID target,
                                        std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      // should only be called on derived classes
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Operation::pack_local_remote_operation(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent_ctx != NULL);
#endif
      rez.serialize(get_operation_kind());
      rez.serialize(this);
      rez.serialize(runtime->address_space);
      rez.serialize(unique_op_id);
      rez.serialize(parent_ctx->get_unique_id());
      rez.serialize<bool>(tracing);
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    void Operation::dump_physical_state(RegionRequirement *req, unsigned idx,
                                        bool before, bool closing)
    //--------------------------------------------------------------------------
    {
      if ((req->handle_type == LEGION_PARTITION_PROJECTION &&
           req->partition == LogicalPartition::NO_PART) ||
          (req->handle_type != LEGION_PARTITION_PROJECTION &&
           req->region == LogicalRegion::NO_REGION))
        return;
      InnerContext *context = find_physical_context(idx, *req);
      RegionTreeContext ctx = context->get_context();
      RegionTreeNode *child_node = req->handle_type == 
        LEGION_PARTITION_PROJECTION ?
        static_cast<RegionTreeNode*>(runtime->forest->get_node(req->partition)) :
        static_cast<RegionTreeNode*>(runtime->forest->get_node(req->region));
      FieldMask user_mask =
        child_node->column_source->get_field_mask(req->privilege_fields);
      TreeStateLogger::capture_state(runtime, req, idx,
                                     get_logging_name(), unique_op_id,
                                     child_node, ctx.get_id(),
                                     before/*before*/, false/*premap*/,
                                     closing/*closing*/, false/*logical*/,
                   FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
    }
#endif

    //--------------------------------------------------------------------------
    void Operation::MappingDependenceTracker::issue_stage_triggers(
                      Operation *op, Runtime *runtime, MustEpochOp *must_epoch)
    //--------------------------------------------------------------------------
    {
      bool resolve_now = true;
      bool trigger_now = false;
      RtEvent map_precondition;
      if (!mapping_dependences.empty())
        map_precondition = Runtime::merge_events(mapping_dependences);
      if (must_epoch == NULL)
      {
        if (map_precondition.exists() && !map_precondition.has_triggered())
          op->get_context()->add_to_ready_queue(op, map_precondition);
        else
          trigger_now = true;
      }
      else if (!map_precondition.has_triggered())
        must_epoch->add_mapping_dependence(map_precondition);

      if (!resolution_dependences.empty())
      {
        RtEvent resolve_precondition = 
          Runtime::merge_events(resolution_dependences);
        if (!resolve_precondition.has_triggered())
        {
          op->get_context()->add_to_resolution_queue(op, resolve_precondition);
          resolve_now = false;
        }
      }
      if (resolve_now)
        op->trigger_resolution();
      if (trigger_now)
        op->trigger_ready();
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
          op->get_context()->add_to_trigger_commit_queue(op,
                                        commit_precondition);
          return false;
        }
      }
      return true;
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Memoizable
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    void Memoizable::pack_remote_memoizable(Serializer &rez, 
                                            AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<Memoizable*>(const_cast<Memoizable*>(this));
      const AddressSpaceID origin_space = get_origin_space();
#ifdef DEBUG_LEGION
      assert(origin_space != target);
#endif
      rez.serialize(origin_space);
      rez.serialize<Operation::OpKind>(get_memoizable_kind());
      TraceLocalID tid = get_trace_local_id();
      rez.serialize(tid.first);
      rez.serialize(tid.second);
      rez.serialize(get_memo_completion());
      rez.serialize<bool>(is_memoizable_task());
      rez.serialize<bool>(is_memoizing());
    }

    //--------------------------------------------------------------------------
    RemoteMemoizable::RemoteMemoizable(Operation *o, Memoizable *orig,
                                       AddressSpaceID orgn, Operation::OpKind k,
                                       TraceLocalID tid, ApEvent completion,
                                       bool is_mem, bool is_m)
      : op(o), original(orig), origin(orgn), kind(k), trace_local_id(tid),
        completion_event(completion), is_mem_task(is_mem), is_memo(is_m)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteMemoizable::~RemoteMemoizable(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool RemoteMemoizable::is_memoizable_task(void) const
    //--------------------------------------------------------------------------
    {
      return is_mem_task;
    }

    //--------------------------------------------------------------------------
    bool RemoteMemoizable::is_recording(void) const
    //--------------------------------------------------------------------------
    {
      // Has to be true if we made this
      return true;
    }

    //--------------------------------------------------------------------------
    bool RemoteMemoizable::is_memoizing(void) const
    //--------------------------------------------------------------------------
    {
      return is_memo;
    }

    //--------------------------------------------------------------------------
    AddressSpaceID RemoteMemoizable::get_origin_space(void) const
    //--------------------------------------------------------------------------
    {
      return origin;
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate* RemoteMemoizable::get_template(void) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    ApEvent RemoteMemoizable::get_memo_completion(void) const
    //--------------------------------------------------------------------------
    {
      return completion_event;
    }

    //--------------------------------------------------------------------------
    void RemoteMemoizable::replay_mapping_output(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    Operation* RemoteMemoizable::get_operation(void) const
    //--------------------------------------------------------------------------
    {
      return op;
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemoteMemoizable::get_memoizable_kind(void) const
    //--------------------------------------------------------------------------
    {
      return kind;
    }

    //--------------------------------------------------------------------------
    TraceLocalID RemoteMemoizable::get_trace_local_id(void) const
    //--------------------------------------------------------------------------
    {
      return trace_local_id;
    }

    //--------------------------------------------------------------------------
    ApEvent RemoteMemoizable::compute_sync_precondition(
                                              const TraceInfo *trace_info) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    void RemoteMemoizable::set_effects_postcondition(ApEvent postcondition)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteMemoizable::complete_replay(ApEvent complete_event)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteMemoizable::find_equivalence_sets(Runtime *runtime, unsigned idx,
              const FieldMask &mask, FieldMaskSet<EquivalenceSet> &target) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(origin != runtime->address_space);
#endif
      RtUserEvent done = Runtime::create_rt_user_event();
      // Send the request back to the origin node
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(original);
        rez.serialize(idx);
        rez.serialize(mask);
        rez.serialize(&target);
        rez.serialize(done);
      }
      runtime->send_remote_trace_equivalence_sets_request(origin, rez);
      done.wait();
    }

    //--------------------------------------------------------------------------
    const VersionInfo& RemoteMemoizable::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *new VersionInfo();
    }

    //--------------------------------------------------------------------------
    void RemoteMemoizable::pack_remote_memoizable(Serializer &rez,
                                                  AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(original);
      rez.serialize(origin);
      if (origin == target)
        return;
      rez.serialize(kind);
      rez.serialize(trace_local_id.first);
      rez.serialize(trace_local_id.second);
      rez.serialize(completion_event);
      rez.serialize<bool>(is_mem_task);
      rez.serialize<bool>(is_memo);
    }

    //--------------------------------------------------------------------------
    Memoizable* RemoteMemoizable::clone(Operation *newop)
    //--------------------------------------------------------------------------
    {
      return new RemoteMemoizable(newop, original, origin, kind, trace_local_id,
                                  completion_event, is_mem_task, is_memo);
    }

    //--------------------------------------------------------------------------
    /*static*/ Memoizable* RemoteMemoizable::unpack_remote_memoizable(
                           Deserializer &derez, Operation *op, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      Memoizable *original;
      derez.deserialize(original);
      AddressSpaceID origin;
      derez.deserialize(origin);
      if (origin == runtime->address_space)
        return original;
      Operation::OpKind kind;
      derez.deserialize(kind);
      TraceLocalID tid;
      derez.deserialize(tid.first);
      derez.deserialize(tid.second);
      ApEvent completion_event;
      derez.deserialize(completion_event);
      bool is_mem_task, is_memo;
      derez.deserialize<bool>(is_mem_task);
      derez.deserialize<bool>(is_memo);
      return new RemoteMemoizable(op, original, origin, kind, tid,
                                  completion_event, is_mem_task, is_memo);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteMemoizable::handle_eq_request(Deserializer &derez,
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memoizable *memo;
      derez.deserialize(memo);
      unsigned index;
      derez.deserialize(index);
      FieldMask mask;
      derez.deserialize(mask);
      FieldMaskSet<EquivalenceSet> *target;
      derez.deserialize(target);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      
      FieldMaskSet<EquivalenceSet> result;
      memo->find_equivalence_sets(runtime, index, mask, result);
      if (!result.empty())
      {
        Serializer rez;
        {
          RezCheck z2(rez);
          rez.serialize(target);
          rez.serialize<size_t>(result.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                result.begin(); it != result.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize(done_event);
        }
        runtime->send_remote_trace_equivalence_sets_response(source, rez);
      }
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteMemoizable::handle_eq_response(Deserializer &derez,
                                                         Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldMaskSet<EquivalenceSet> *target;
      derez.deserialize(target);
      size_t num_sets;
      derez.deserialize(num_sets);
      std::set<RtEvent> ready_events;
      for (unsigned idx = 0; idx < num_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        EquivalenceSet *set = 
          runtime->find_or_request_equivalence_set(did, ready);
        FieldMask mask;
        derez.deserialize(mask);
        target->insert(set, mask);
        if (ready.exists() && !ready.has_triggered())
          ready_events.insert(ready);
      }
      RtUserEvent done_event;
      derez.deserialize(done_event);

      if (!ready_events.empty())
        Runtime::trigger_event(done_event, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(done_event);
    }

    ///////////////////////////////////////////////////////////// 
    // External Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::pack_mappable(const Mappable &mappable,
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(mappable.map_id);
      rez.serialize(mappable.tag);
      rez.serialize(mappable.mapper_data_size);
      if (mappable.mapper_data_size > 0)
        rez.serialize(mappable.mapper_data, mappable.mapper_data_size);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::pack_index_space_requirement(
                              const IndexSpaceRequirement &req, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(req.handle);
      rez.serialize(req.privilege);
      rez.serialize(req.parent);
      // no need to send verified
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::unpack_index_space_requirement(
                                IndexSpaceRequirement &req, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(req.handle);
      derez.deserialize(req.privilege);
      derez.deserialize(req.parent);
      req.verified = true;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::pack_region_requirement(
                                  const RegionRequirement &req, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(req.region);
      rez.serialize(req.partition);
      rez.serialize(req.privilege_fields.size());
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
            it != req.privilege_fields.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(req.instance_fields.size());
      for (std::vector<FieldID>::const_iterator it = 
            req.instance_fields.begin(); it != req.instance_fields.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(req.privilege);
      rez.serialize(req.prop);
      rez.serialize(req.parent);
      rez.serialize(req.redop);
      rez.serialize(req.tag);
      rez.serialize(req.flags);
      rez.serialize(req.handle_type);
      rez.serialize(req.projection);
      size_t projection_size = 0;
      const void *projection_args = req.get_projection_args(&projection_size);
      rez.serialize(projection_size);
      if (projection_size > 0)
        rez.serialize(projection_args, projection_size);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::unpack_mappable(Mappable &mappable,
                                                      Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(mappable.map_id);
      derez.deserialize(mappable.tag);
      derez.deserialize(mappable.mapper_data_size);
      if (mappable.mapper_data_size > 0)
      {
        // If we already have mapper data, then we are going to replace it
        if (mappable.mapper_data != NULL)
          free(mappable.mapper_data);
        mappable.mapper_data = malloc(mappable.mapper_data_size);
        derez.deserialize(mappable.mapper_data, mappable.mapper_data_size);
      }
      else if (mappable.mapper_data != NULL)
      {
        // If we freed it remotely then we can free it here too
        free(mappable.mapper_data);
        mappable.mapper_data = NULL;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::unpack_region_requirement(
                                    RegionRequirement &req, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(req.region);
      derez.deserialize(req.partition);
      size_t num_privilege_fields;
      derez.deserialize(num_privilege_fields);
      for (unsigned idx = 0; idx < num_privilege_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        req.privilege_fields.insert(fid);
      }
      size_t num_instance_fields;
      derez.deserialize(num_instance_fields);
      for (unsigned idx = 0; idx < num_instance_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        req.instance_fields.push_back(fid);
      }
      derez.deserialize(req.privilege);
      derez.deserialize(req.prop);
      derez.deserialize(req.parent);
      derez.deserialize(req.redop);
      derez.deserialize(req.tag);
      derez.deserialize(req.flags);
      derez.deserialize(req.handle_type);
      derez.deserialize(req.projection);
      req.flags |= LEGION_VERIFIED_FLAG;
      size_t projection_size;
      derez.deserialize(projection_size);
      if (projection_size > 0)
      {
        void *projection_ptr = malloc(projection_size);
        derez.deserialize(projection_ptr, projection_size);
        req.set_projection_args(projection_ptr, projection_size, true/*own*/);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::pack_grant(const Grant &grant, 
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      grant.impl->pack_grant(rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::unpack_grant(Grant &grant, 
                                                   Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Create a new grant impl object to perform the unpack
      grant = Grant(new GrantImpl());
      grant.impl->unpack_grant(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::pack_phase_barrier(
                                  const PhaseBarrier &barrier, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(barrier.phase_barrier);
    }  

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::unpack_phase_barrier(
                                    PhaseBarrier &barrier, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(barrier.phase_barrier);
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
      bool set_future = false;
      {
        AutoLock o_lock(op_lock);
        if (result_future.impl != NULL)
          set_future = true;
        else
          can_result_future_complete = true;
      }
      if (set_future)
        result_future.impl->set_result(&predicate_value, 
                                       sizeof(predicate_value), false/*own*/);
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
      bool set_future = false;
      if (result_future.impl == NULL)
      {
        Future temp = Future(
              new FutureImpl(parent_ctx, runtime, true/*register*/,
                runtime->get_available_distributed_id(),
                runtime->address_space, get_completion_event(), this));
        AutoLock o_lock(op_lock);
        // See if we lost the race
        if (result_future.impl == NULL)
        {
          result_future = temp; 
          // if the predicate is complete we can complete the future
          set_future = can_result_future_complete; 
        }
      }
      if (set_future)
        result_future.impl->set_result(&predicate_value, 
                                sizeof(predicate_value), false/*owned*/);
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
    void SpeculativeOp::initialize_speculation(InnerContext *ctx, bool track,
        unsigned regions, const std::vector<StaticDependence> *dependences,
        const Predicate &p, const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, track, regions, provenance, dependences);
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
        if (runtime->legion_spy_enabled)
          LegionSpy::log_predicate_use(unique_op_id, 
                                       predicate->get_unique_op_id());
      }
    }

    //--------------------------------------------------------------------------
    void SpeculativeOp::initialize_speculation(InnerContext *ctx, bool track,
        unsigned regions, const std::vector<StaticDependence> *dependences,
        const Predicate &p, Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, track, regions, provenance, dependences);
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
        if (runtime->legion_spy_enabled)
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
          if (runtime->legion_spy_enabled)
            LegionSpy::log_predicated_false_op(unique_op_id);
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
        wait_on.wait();
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
        if (runtime->legion_spy_enabled)
          LegionSpy::log_predicated_false_op(unique_op_id);
        // Can remove our predicate reference since we don't need it anymore
        predicate->remove_predicate_reference();
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
        if (runtime->legion_spy_enabled)
          LegionSpy::log_predicated_false_op(unique_op_id);
        resolve_false(true/*speculated*/, true/*launched*/);
      }
      if (need_resolve)
        resolve_speculation();
    }

    /////////////////////////////////////////////////////////////
    // External Mapping
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalMapping::ExternalMapping(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ExternalMapping::pack_external_mapping(Serializer &rez, 
                                                AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_region_requirement(requirement, rez);
      rez.serialize(grants.size());
      for (unsigned idx = 0; idx < grants.size(); idx++)
        pack_grant(grants[idx], rez);
      rez.serialize(wait_barriers.size());
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        pack_phase_barrier(wait_barriers[idx], rez);
      rez.serialize(arrive_barriers.size());
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        pack_phase_barrier(arrive_barriers[idx], rez);
      rez.serialize(layout_constraint_id);
      pack_mappable(*this, rez);
      rez.serialize<size_t>(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalMapping::unpack_external_mapping(Deserializer &derez,
                                                  Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_region_requirement(requirement, derez);
      size_t num_grants;
      derez.deserialize(num_grants);
      grants.resize(num_grants);
      for (unsigned idx = 0; idx < grants.size(); idx++)
        unpack_grant(grants[idx], derez);
      size_t num_wait_barriers;
      derez.deserialize(num_wait_barriers);
      wait_barriers.resize(num_wait_barriers);
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        unpack_phase_barrier(wait_barriers[idx], derez);
      size_t num_arrive_barriers;
      derez.deserialize(num_arrive_barriers);
      arrive_barriers.resize(num_arrive_barriers);
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        unpack_phase_barrier(arrive_barriers[idx], derez);
      derez.deserialize(layout_constraint_id);
      unpack_mappable(*this, derez);
      size_t index;
      derez.deserialize(index);
      set_context_index(index);
    }

    /////////////////////////////////////////////////////////////
    // Map Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MapOp::MapOp(Runtime *rt)
      : ExternalMapping(), Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MapOp::MapOp(const MapOp &rhs)
      : ExternalMapping(), Operation(NULL)
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
    PhysicalRegion MapOp::initialize(InnerContext *ctx, 
                                     const InlineLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/, 1/*regions*/,
          launcher.provenance.c_str(), launcher.static_dependences);
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
      const ApUserEvent term_event = Runtime::create_ap_user_event(NULL);
      region = PhysicalRegion(new PhysicalRegionImpl(requirement,
            mapped_event, ready_event, term_event, true/*mapped*/, ctx,
            map_id, tag, false/*leaf*/, false/*virtual mapped*/, runtime));
      termination_event = term_event;
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
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      layout_constraint_id = launcher.layout_constraint_id;
      
      if (runtime->legion_spy_enabled)
        LegionSpy::log_mapping_operation(parent_ctx->get_unique_id(),
                                         unique_op_id, context_index);
      return region;
    }

    //--------------------------------------------------------------------------
    void MapOp::initialize(InnerContext *ctx, const PhysicalRegion &reg,
                           const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 1/*regions*/, provenance);
      parent_task = ctx->get_task();
      requirement = reg.impl->get_requirement();
      // If this was a write-discard privilege, change it to read-write
      // so that we don't lose any data
      if (HAS_WRITE_DISCARD(requirement))
        requirement.privilege = LEGION_READ_WRITE;
      map_id = reg.impl->map_id;
      tag = reg.impl->tag;
      region = reg;
      termination_event = region.impl->remap_region(ready_event);
      remap_region = true;
      // No need to check the privileges here since we know that we have
      // them from the first time that we made this physical region
      if (runtime->legion_spy_enabled)
        LegionSpy::log_mapping_operation(parent_ctx->get_unique_id(),
                                         unique_op_id, context_index);
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
      ready_event = Runtime::create_ap_user_event(NULL);
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
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
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      atomic_locks.clear();
      map_applied_conditions.clear();
      profiling_requests.clear();
      if (!profiling_info.empty())
      {
        for (unsigned idx = 0; idx < profiling_info.size(); idx++)
          free(profiling_info[idx].buffer);
        profiling_info.clear();
      }
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
      if (runtime->legion_spy_enabled)
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
      if (runtime->check_privileges)
        check_privilege();
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(this, 
                              wait_barriers, arrive_barriers);
      ProjectionInfo projection_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions);
    }

    //--------------------------------------------------------------------------
    void MapOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Compute the version numbers for this mapping operation
      std::set<RtEvent> preconditions;
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement, 
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
      const PhysicalTraceInfo trace_info(this, 0/*index*/, true/*init*/);
      // If we have any wait preconditions from phase barriers or 
      // grants then we use them to compute a precondition for doing
      // any copies or anything else for this operation
      ApEvent init_precondition = execution_fence_event;
      if (!wait_barriers.empty() || !grants.empty())
      {
        ApEvent sync_precondition = 
          merge_sync_preconditions(trace_info, grants, wait_barriers);
        if (sync_precondition.exists())
        {
          if (init_precondition.exists())
            init_precondition = Runtime::merge_events(&trace_info, 
                                  init_precondition, sync_precondition); 
          else
            init_precondition = sync_precondition;
        }
      }
      InstanceSet mapped_instances;
      // If we are remapping then we know the answer
      // so we don't need to do any premapping
      ApEvent effects_done;
      if (remap_region)
      {
        region.impl->get_references(mapped_instances);
        effects_done = 
          runtime->forest->physical_perform_updates_and_registration(
                                                requirement, version_info,
                                                this, 0/*idx*/, 
                                                init_precondition,
                                                termination_event,
                                                mapped_instances, 
                                                trace_info,
                                                map_applied_conditions
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
      }
      else
      { 
        // Now we've got the valid instances so invoke the mapper
        const bool record_valid = invoke_mapper(mapped_instances); 
        // First mapping so set the references now
        region.impl->set_references(mapped_instances);
        // Then we can register our mapped instances
        effects_done = 
          runtime->forest->physical_perform_updates_and_registration(
                                                requirement, version_info,
                                                this, 0/*idx*/,
                                                init_precondition,
                                                termination_event, 
                                                mapped_instances,
                                                trace_info,
                                                map_applied_conditions,
#ifdef DEBUG_LEGION
                                                get_logging_name(),
                                                unique_op_id,
#endif
                                                record_valid);
      }
#ifdef DEBUG_LEGION
      if (!IS_NO_ACCESS(requirement) && !requirement.privilege_fields.empty())
      {
        assert(!mapped_instances.empty());
        dump_physical_state(&requirement, 0);
      } 
#endif
      ApEvent map_complete_event = ApEvent::NO_AP_EVENT;
      if (mapped_instances.size() > 1)
      {
        std::set<ApEvent> mapped_events;
        for (unsigned idx = 0; idx < mapped_instances.size(); idx++)
          mapped_events.insert(mapped_instances[idx].get_ready_event());
        map_complete_event = Runtime::merge_events(&trace_info, mapped_events);
      }
      else if (!mapped_instances.empty())
        map_complete_event = mapped_instances[0].get_ready_event();
      if (runtime->legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, parent_ctx, 
                                              0/*idx*/, requirement,
                                              mapped_instances);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, map_complete_event,
                                        termination_event);
#endif
      }
      if (!effects_done.exists())
        effects_done = termination_event; 
      if (!atomic_locks.empty() || !arrive_barriers.empty())
      {
        // They've already been sorted in order 
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          map_complete_event = 
                Runtime::acquire_ap_reservation(it->first, it->second,
                                                map_complete_event);
          // We can also issue the release condition on our termination
          Runtime::release_reservation(it->first, effects_done);
        }
        for (std::vector<PhaseBarrier>::iterator it = 
              arrive_barriers.begin(); it != arrive_barriers.end(); it++)
        {
          if (runtime->legion_spy_enabled)
            LegionSpy::log_phase_barrier_arrival(unique_op_id, 
                                                 it->phase_barrier);
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        effects_done);    
        }
      }
      // We can trigger the ready event now that we know its precondition
      Runtime::trigger_event(NULL, ready_event, map_complete_event);
      // Now we can trigger the mapping event and indicate
      // to all our mapping dependences that we are mapped.
      RtEvent mapping_applied;
      if (!map_applied_conditions.empty())
        mapping_applied = Runtime::merge_events(map_applied_conditions);
      if (!acquired_instances.empty())
        mapping_applied = release_nonempty_acquired_instances(mapping_applied, 
                                                          acquired_instances);
      complete_mapping(mapping_applied);
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
      // If we have any copy-out effects from this inline mapping, we'll
      // need to keep it around long enough for the parent task in case
      // it decides that it needs to
      if (!request_early_complete(effects_done))
        complete_execution(Runtime::protect_event(effects_done));
      else
        complete_execution();
    }

    //--------------------------------------------------------------------------
    void MapOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we need to do a profiling response
      if (profiling_reported.exists())
      {
        if (outstanding_profiling_requests.load() > 0)
        {
#ifdef DEBUG_LEGION
          assert(mapped_event.has_triggered());
#endif
          std::vector<MapProfilingInfo> to_perform;
          {
            AutoLock o_lock(op_lock);
            to_perform.swap(profiling_info);
          }
          if (!to_perform.empty())
          {
            for (unsigned idx = 0; idx < to_perform.size(); idx++)
            {
              MapProfilingInfo &info = to_perform[idx];
              const Realm::ProfilingResponse resp(info.buffer,info.buffer_size);
              info.total_reports = outstanding_profiling_requests.load();
              info.profiling_responses.attach_realm_profiling_response(resp);
              mapper->invoke_inline_report_profiling(this, &info);
              free(info.buffer);
            }
            const int count = to_perform.size() + 
              outstanding_profiling_reported.fetch_add(to_perform.size());
#ifdef DEBUG_LEGION
            assert(count <= outstanding_profiling_requests.load());
#endif
            if (count == outstanding_profiling_requests.load())
              Runtime::trigger_event(profiling_reported);
          }
        }
        else
        {
          // We're not expecting any profiling callbacks so we need to
          // do one ourself to inform the mapper that there won't be any
          Mapping::Mapper::InlineProfilingInfo info;
          info.total_reports = 0;
          info.fill_response = false; // make valgrind happy
          mapper->invoke_inline_report_profiling(this, &info);    
          Runtime::trigger_event(profiling_reported);
        }
      }
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
    void MapOp::select_sources(const unsigned index,
                               const InstanceRef &target,
                               const InstanceSet &sources,
                               std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index == 0);
#endif
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
      compute_ranking(mapper, output.chosen_ranking, sources, ranking);
    } 

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,unsigned>* MapOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void MapOp::update_atomic_locks(const unsigned index,
                                    Reservation lock, bool exclusive)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index == 0);
#endif
      AutoLock o_lock(op_lock);
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
    ApEvent MapOp::get_program_order_event(void) const
    //--------------------------------------------------------------------------
    {
      return ready_event;
    }

    //--------------------------------------------------------------------------
    UniqueID MapOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    } 

    //--------------------------------------------------------------------------
    size_t MapOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void MapOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int MapOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* MapOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    void MapOp::check_privilege(void)
    //--------------------------------------------------------------------------
    { 
      if ((requirement.handle_type == LEGION_PARTITION_PROJECTION) || 
          (requirement.handle_type == LEGION_REGION_PROJECTION))
        REPORT_LEGION_ERROR(ERROR_PROJECTION_REGION_REQUIREMENTS,
                         "Projection region requirements are not "
                         "permitted for inline mappings (in task %s)",
                         parent_ctx->get_task_name())
      FieldID bad_field = LEGION_AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, then check the privileges with the parent context
      if (et == LEGION_NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, bad_index);
      switch (et)
      {
        case LEGION_NO_ERROR:
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
            FieldSpace sp = 
              (requirement.handle_type == LEGION_SINGULAR_PROJECTION) ||
              (requirement.handle_type == LEGION_REGION_PROJECTION)
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
            else if (bad_field == LEGION_AUTO_GENERATE_ID) 
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
    bool MapOp::invoke_mapper(InstanceSet &chosen_instances)
    //--------------------------------------------------------------------------
    {
      Mapper::MapInlineInput input;
      Mapper::MapInlineOutput output;
      output.profiling_priority = LG_THROUGHPUT_WORK_PRIORITY; 
      output.track_valid_region = true;
      // Invoke the mapper
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      if (mapper->request_valid_instances)
      {
        InstanceSet valid_instances;
        runtime->forest->physical_premap_region(this, 0/*idx*/, requirement,
                      version_info, valid_instances, map_applied_conditions);
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
      }
      mapper->invoke_map_inline(this, &input, &output);
      if (!output.profiling_requests.empty())
      {
        filter_copy_request_kinds(mapper,
            output.profiling_requests.requested_measurements,
            profiling_requests, true/*warn*/);
        profiling_priority = output.profiling_priority;
#ifdef DEBUG_LEGION
        assert(!profiling_reported.exists());
#endif
        profiling_reported = Runtime::create_rt_user_event();
      }
      // Now we have to validate the output
      // Go through the instances and make sure we got one for every field
      // Also check to make sure that none of them are composite instances
      RegionTreeID bad_tree = 0;
      std::vector<FieldID> missing_fields;
      std::vector<PhysicalManager*> unacquired;
      int virtual_index = runtime->forest->physical_convert_mapping(this,
                                requirement, output.chosen_instances, 
                                chosen_instances, bad_tree, missing_fields,
                                &acquired_instances, unacquired, 
                                !runtime->unsafe_mapper);
      if (bad_tree > 0)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'map_inline' "
                      "on mapper %s. Mapper selected instance from region "
                      "tree %d to satisfy a region requirement for an inline "
                      "mapping in task %s (ID %lld) whose region tree is %d.", 
                      mapper->get_mapper_name(),
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
               requirement.region.get_field_space(), *it, 
               LEGION_NAME_SEMANTIC_TAG, name, name_size, true, false))
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
      if (virtual_index >= 0)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'map_inline' "
                      "on mapper %s. Mapper requested creation of a composite "
                      "instance for inline mapping in task %s (ID %lld).",
                      mapper->get_mapper_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
      if (!output.track_valid_region && !IS_READ_ONLY(requirement))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_NON_READ_ONLY_UNTRACK_VALID,
            "Ignoring request by mapper %s to not track valid instances "
            "for inline mapping %lld in parent task %s (UID %lld) because "
            "the region requirement does not have read-only privileges.",
            mapper->get_mapper_name(), unique_op_id, 
            parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        output.track_valid_region = true;
      }
      // If we are doing unsafe mapping, then we can return
      if (runtime->unsafe_mapper)
        return output.track_valid_region;
      // If this requirement doesn't have a no access flag then we
      // need to check to make sure that the instances are visible
      if (!requirement.is_no_access())
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        std::set<Memory> visible_memories;
        runtime->find_visible_memories(exec_proc, visible_memories);
        for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
        {
          const Memory mem = chosen_instances[idx].get_memory();
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
        PhysicalManager *manager = chosen_instances[idx].get_physical_manager();
        if (!manager->meets_regions(regions_to_check))
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
          if (!chosen_instances[idx].get_manager()->is_reduction_manager())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of "
                          "'map_inline' on mapper %s. Mapper failed to select "
                          "specialized reduction instances for region "
                          "requirement with reduction-only privileges for "
                          "inline mapping operation in task %s (ID %lld).",
                          mapper->get_mapper_name(),parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id())
      }
      else
      {
        for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
        {
          if (chosen_instances[idx].get_manager()->is_reduction_manager())
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
          InstanceManager *manager = chosen_instances[idx].get_manager();
          const LayoutConstraint *conflict_constraint = NULL;
          if (manager->conflicts(constraints, get_shard_point(), 
                                 &conflict_constraint))
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
      return output.track_valid_region;
    }

    //--------------------------------------------------------------------------
    void MapOp::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &requests, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest &request = requests.add_request( 
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
          &response, sizeof(response), profiling_priority);
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      handle_profiling_update(count);
    }

    //--------------------------------------------------------------------------
    void MapOp::handle_profiling_response(const ProfilingResponseBase *base,
                                       const Realm::ProfilingResponse &response,
                                       const void *orig, size_t orig_length)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapper != NULL);
#endif
      const OpProfilingResponse *op_info = 
        static_cast<const OpProfilingResponse*>(base);
      // Check to see if we are done mapping, if not then we need to defer
      // this until we are done mapping so we know how many reports to expect
      if (!mapped_event.has_triggered())
      {
        // Take the lock and see if we lost the race
        AutoLock o_lock(op_lock);
        if (!mapped_event.has_triggered())
        {
          // Save this profiling response for later until we know the
          // full count of profiling responses
          profiling_info.resize(profiling_info.size() + 1);
          MapProfilingInfo &info = profiling_info.back();
          info.fill_response = op_info->fill;
          info.buffer_size = orig_length;
          info.buffer = malloc(orig_length);
          memcpy(info.buffer, orig, orig_length);
          return;
        }
      }
      // If we get here then we can handle the response now
      Mapping::Mapper::InlineProfilingInfo info; 
      info.profiling_responses.attach_realm_profiling_response(response);
      info.total_reports = outstanding_profiling_requests.load();
      info.fill_response = op_info->fill;
      mapper->invoke_inline_report_profiling(this, &info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(count <= outstanding_profiling_requests.load());
#endif
      if (count == outstanding_profiling_requests.load())
        Runtime::trigger_event(profiling_reported);
    }

    //--------------------------------------------------------------------------
    void MapOp::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count > 0);
      assert(!mapped_event.has_triggered());
#endif
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void MapOp::pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                      std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_mapping(rez, target);
      rez.serialize<size_t>(profiling_requests.size());
      if (!profiling_requests.empty())
      {
        for (unsigned idx = 0; idx < profiling_requests.size(); idx++)
          rez.serialize(profiling_requests[idx]);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Create a user event for this response
        const RtUserEvent response = Runtime::create_rt_user_event();
        rez.serialize(response);
        applied_events.insert(response);
      }
    }

    //--------------------------------------------------------------------------
    DomainPoint MapOp::get_shard_point(void) const
    //--------------------------------------------------------------------------
    {
      // We have no shard point in the non control replicated case
      return DomainPoint();
    }

    /////////////////////////////////////////////////////////////
    // External Copy 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalCopy::ExternalCopy(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ExternalCopy::pack_external_copy(Serializer &rez,
                                          AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize<size_t>(src_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        pack_region_requirement(src_requirements[idx], rez);
      rez.serialize<size_t>(dst_requirements.size());
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
        pack_region_requirement(dst_requirements[idx], rez);
      rez.serialize<size_t>(src_indirect_requirements.size());
      for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        pack_region_requirement(src_indirect_requirements[idx], rez);
      rez.serialize<size_t>(dst_indirect_requirements.size());
      for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        pack_region_requirement(dst_indirect_requirements[idx], rez);
      rez.serialize(grants.size());
      for (unsigned idx = 0; idx < grants.size(); idx++)
        pack_grant(grants[idx], rez);
      rez.serialize(wait_barriers.size());
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        pack_phase_barrier(wait_barriers[idx], rez);
      rez.serialize(arrive_barriers.size());
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        pack_phase_barrier(arrive_barriers[idx], rez);
      rez.serialize<bool>(is_index_space);
      rez.serialize(index_domain);
      rez.serialize(index_point);
      pack_mappable(*this, rez);
      rez.serialize<size_t>(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalCopy::unpack_external_copy(Deserializer &derez,
                                            Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t num_srcs;
      derez.deserialize(num_srcs);
      src_requirements.resize(num_srcs);
      for (unsigned idx = 0; idx < num_srcs; idx++)
        unpack_region_requirement(src_requirements[idx], derez);
      size_t num_dsts;
      derez.deserialize(num_dsts);
      dst_requirements.resize(num_dsts);
      for (unsigned idx = 0; idx < num_dsts; idx++)
        unpack_region_requirement(dst_requirements[idx], derez);
      size_t num_indirect_srcs;
      derez.deserialize(num_indirect_srcs);
      src_indirect_requirements.resize(num_indirect_srcs);
      for (unsigned idx = 0; idx < num_indirect_srcs; idx++)
        unpack_region_requirement(src_indirect_requirements[idx], derez);
      size_t num_indirect_dsts;
      derez.deserialize(num_indirect_dsts);
      dst_indirect_requirements.resize(num_indirect_dsts);
      for (unsigned idx = 0; idx < num_indirect_dsts; idx++)
        unpack_region_requirement(dst_indirect_requirements[idx], derez);
      size_t num_grants;
      derez.deserialize(num_grants);
      grants.resize(num_grants);
      for (unsigned idx = 0; idx < grants.size(); idx++)
        unpack_grant(grants[idx], derez);
      size_t num_wait_barriers;
      derez.deserialize(num_wait_barriers);
      wait_barriers.resize(num_wait_barriers);
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        unpack_phase_barrier(wait_barriers[idx], derez);
      size_t num_arrive_barriers;
      derez.deserialize(num_arrive_barriers);
      arrive_barriers.resize(num_arrive_barriers);
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        unpack_phase_barrier(arrive_barriers[idx], derez);
      derez.deserialize<bool>(is_index_space);
      derez.deserialize(index_domain);
      derez.deserialize(index_point);
      unpack_mappable(*this, derez);
      size_t index;
      derez.deserialize(index);
      set_context_index(index);
    }

    /////////////////////////////////////////////////////////////
    // Copy Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyOp::CopyOp(Runtime *rt)
      : ExternalCopy(), MemoizableOp<SpeculativeOp>(rt)
    //--------------------------------------------------------------------------
    {
      this->is_index_space = false;
    }

    //--------------------------------------------------------------------------
    CopyOp::CopyOp(const CopyOp &rhs)
      : ExternalCopy(), MemoizableOp<SpeculativeOp>(NULL)
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
    void CopyOp::initialize(InnerContext *ctx, const CopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_speculation(ctx, true/*track*/, 
                             launcher.src_requirements.size() + 
                               launcher.dst_requirements.size(), 
                             launcher.static_dependences,
                             launcher.predicate, launcher.provenance.c_str());
      initialize_memoizable();
      src_requirements.resize(launcher.src_requirements.size());
      dst_requirements.resize(launcher.dst_requirements.size());
      src_versions.resize(launcher.src_requirements.size());
      dst_versions.resize(launcher.dst_requirements.size());
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
        src_requirements[idx].flags |= LEGION_NO_ACCESS_FLAG;
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
        dst_requirements[idx].flags |= LEGION_NO_ACCESS_FLAG;
      }
      if (!launcher.src_indirect_requirements.empty())
      {
        const size_t gather_size = launcher.src_indirect_requirements.size();
        src_indirect_requirements.resize(gather_size);
        gather_versions.resize(gather_size);
        for (unsigned idx = 0; idx < gather_size; idx++)
        {
          RegionRequirement &req = src_indirect_requirements[idx];
          req = launcher.src_indirect_requirements[idx];
          req.flags |= LEGION_NO_ACCESS_FLAG;
          if (req.privilege_fields.size() != 1)
            REPORT_LEGION_ERROR(ERROR_COPY_GATHER_REQUIREMENT, 
                "Source indirect region requirement %d for copy op in "
                "parent task %s (ID %lld) has %zd fields, but exactly one "
                "field is required.", idx, parent_ctx->get_task_name(),
                parent_ctx->get_unique_id(), req.privilege_fields.size())
        }
        if (launcher.src_indirect_is_range.size() != gather_size)
          REPORT_LEGION_ERROR(ERROR_COPY_GATHER_REQUIREMENT,
              "Invalid 'src_indirect_is_range' size in launcher. The "
              "number of entries (%zd) does not match the number of "
              "'src_indirect_requirments' (%zd) for copy operation in "
              "parent task %s (ID %lld)", 
              launcher.src_indirect_is_range.size(), gather_size, 
              parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        gather_is_range = launcher.src_indirect_is_range;
        for (unsigned idx = 0; idx < gather_size; idx++)
        {
          if (!gather_is_range[idx])
            continue;
          // For anything that is a gather by range we either need 
          // it also to be a scatter by range or we need a reduction
          // on the destination region requirement so we know how
          // to handle reducing down all the values
          if ((idx < launcher.dst_indirect_is_range.size()) &&
              launcher.dst_indirect_is_range[idx])
            continue;
          if (dst_requirements[idx].privilege != LEGION_REDUCE)
            REPORT_LEGION_ERROR(ERROR_DESTINATION_REGION_REQUIREMENT,
                "Invalid privileges for destination region requirement %d "
                " for copy across in parent task %s (ID %lld). Destination "
                "region requirements must use reduction privileges when "
                "there is a range-based source indirection field and there "
                "is no corresponding range indirection on the destination.",
                idx, parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        }
        possible_src_indirect_out_of_range = 
          launcher.possible_src_indirect_out_of_range;
      }
      if (!launcher.dst_indirect_requirements.empty())
      {
        const size_t scatter_size = launcher.dst_indirect_requirements.size();
        dst_indirect_requirements.resize(scatter_size);
        scatter_versions.resize(scatter_size);
        for (unsigned idx = 0; idx < scatter_size; idx++)
        {
          RegionRequirement &req = dst_indirect_requirements[idx];
          req = launcher.dst_indirect_requirements[idx];
          req.flags |= LEGION_NO_ACCESS_FLAG;
          if (req.privilege_fields.size() != 1)
            REPORT_LEGION_ERROR(ERROR_COPY_GATHER_REQUIREMENT, 
                "Destination indirect region requirement %d for copy op in "
                "parent task %s (ID %lld) has %zd fields, but exactly one "
                "field is required.", idx, parent_ctx->get_task_name(),
                parent_ctx->get_unique_id(), req.privilege_fields.size())
        }
        if (launcher.dst_indirect_is_range.size() != scatter_size)
          REPORT_LEGION_ERROR(ERROR_COPY_GATHER_REQUIREMENT,
              "Invalid 'dst_indirect_is_range' size in launcher. The "
              "number of entries (%zd) does not match the number of "
              "'dst_indirect_requirments' (%zd) for copy operation in "
              "parent task %s (ID %lld)", 
              launcher.dst_indirect_is_range.size(), scatter_size, 
              parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        scatter_is_range = launcher.dst_indirect_is_range;
        if (!src_indirect_requirements.empty())
        {
          // Full indirections need to have the same index space
          for (unsigned idx = 0; (idx < src_indirect_requirements.size()) &&
                (idx < dst_indirect_requirements.size()); idx++)
          {
            const IndexSpace src_space = 
              src_indirect_requirements[idx].region.get_index_space();
            const IndexSpace dst_space = 
              dst_indirect_requirements[idx].region.get_index_space();
            if (src_space != dst_space)
              REPORT_LEGION_ERROR(ERROR_COPY_SCATTER_REQUIREMENT,
                  "Mismatch between source indirect and destination indirect "
                  "index spaces for requirement %d for copy operation "
                  "(ID %lld) in parent task %s (ID %lld)",
                  idx, get_unique_id(), parent_ctx->get_task_name(),
                  parent_ctx->get_unique_id())
          }
        }
        possible_dst_indirect_out_of_range = 
          launcher.possible_dst_indirect_out_of_range;
        possible_dst_indirect_aliasing = 
          launcher.possible_dst_indirect_aliasing;
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
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      index_point = launcher.point; 
      if (runtime->legion_spy_enabled)
      {
        const unsigned copy_kind = (src_indirect_requirements.empty() ? 0 : 1) +
          (dst_indirect_requirements.empty() ? 0 : 2);
        LegionSpy::log_copy_operation(parent_ctx->get_unique_id(), unique_op_id,
            copy_kind, context_index, false, false);
      }
      if (runtime->check_privileges)
      {
        perform_type_checking();
      }
    }

    //--------------------------------------------------------------------------
    void CopyOp::perform_type_checking(void) const
    //--------------------------------------------------------------------------
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
      if (!src_indirect_requirements.empty() && 
          (src_indirect_requirements.size() != src_requirements.size()))
        REPORT_LEGION_ERROR(ERROR_NUMBER_SRC_INDIRECT_REQUIREMENTS,
                      "Number of source indirect requirements (%zd) does not "
                      "match number of source requirements (%zd) "
                      "for copy operation (ID %lld) with parent "
                      "task %s (ID %lld)", src_indirect_requirements.size(),
                      src_requirements.size(),
                      get_unique_id(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
      if (!dst_indirect_requirements.empty() &&
          (dst_indirect_requirements.size() != src_requirements.size()))
        REPORT_LEGION_ERROR(ERROR_NUMBER_DST_INDIRECT_REQUIREMENTS,
                      "Number of destination indirect requirements (%zd) "
                      "does not match number of source requriements (%zd) "
                      "for copy operation ID (%lld) with parent "
                      "task %s (ID %lld)", dst_indirect_requirements.size(),
                      src_requirements.size(),
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
      }
      if (!src_indirect_requirements.empty())
      {
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          if (src_indirect_requirements[idx].privilege_fields.size() != 1)
            REPORT_LEGION_ERROR(ERROR_COPY_GATHER_REQUIREMENT,
                      "Copy source indirect requirement %d for copy "
                      "operation (ID %lld) in parent task %s "
                      "(ID %lld) has %zd privilege fields but "
                      "source indirect requirements are only permitted "
                      "to have one privilege field.", idx,
                      get_unique_id(), parent_task->get_task_name(),
                      parent_task->get_unique_id(),
                      src_indirect_requirements[idx].privilege_fields.size())
          if (!IS_READ_ONLY(src_indirect_requirements[idx]))
            REPORT_LEGION_ERROR(ERROR_COPY_GATHER_REQUIREMENT,
                      "Copy source indirect requirement %d for copy "
                      "operation (ID %lld) in parent task %s "
                      "(ID %lld) must be requested with a "
                      "read-only privilege.", idx,
                      get_unique_id(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
        }
      }
      if (!dst_indirect_requirements.empty())
      {
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          if (dst_indirect_requirements[idx].privilege_fields.size() != 1)
            REPORT_LEGION_ERROR(ERROR_COPY_SCATTER_REQUIREMENT,
                      "Copy destination indirect requirement %d for copy "
                      "operation (ID %lld) in parent task %s "
                      "(ID %lld) has %zd privilege fields but "
                      "destination indirect requirements are only permitted "
                      "to have one privilege field.", idx,
                      get_unique_id(), parent_task->get_task_name(),
                      parent_task->get_unique_id(),
                      dst_indirect_requirements[idx].privilege_fields.size())
          if (!IS_READ_ONLY(dst_indirect_requirements[idx]))
            REPORT_LEGION_ERROR(ERROR_COPY_SCATTER_REQUIREMENT,
                      "Copy destination indirect requirement %d for copy "
                      "operation (ID %lld) in parent task %s "
                      "(ID %lld) must be requested with a "
                      "read-only privilege.", idx,
                      get_unique_id(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
        } 
      }
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        // Check that the source and destination field sizes are the same
        const std::vector<FieldID> &src_fields = 
          src_requirements[idx].instance_fields;
        const std::vector<FieldID> &dst_fields =
          dst_requirements[idx].instance_fields;
        const FieldSpace src_space = 
          src_requirements[idx].parent.get_field_space();
        const FieldSpace dst_space =
          dst_requirements[idx].parent.get_field_space();
        for (unsigned fidx = 0; fidx < src_fields.size(); fidx++)
        {
          const size_t src_size = 
            runtime->forest->get_field_size(src_space, src_fields[fidx]);
          const size_t dst_size = 
            runtime->forest->get_field_size(dst_space, dst_fields[fidx]);
          if (src_size != dst_size)
            REPORT_LEGION_ERROR(ERROR_FIELD_SIZE_MISMATCH,
                "Different field sizes are not permitted for region-to-region "
                "copy operations. Fields %d and %d of region requirement %d "
                "have different sizes (%zd bytes and %zd bytes respectively) "
                "in copy operation (UID %lld) in parent task %s (UID %lld).",
                src_fields[fidx], dst_fields[fidx], idx, src_size, dst_size,
                get_unique_id(), parent_ctx->get_task_name(),
                parent_ctx->get_unique_id())
          const CustomSerdezID src_serdez =
            runtime->forest->get_field_serdez(src_space, src_fields[fidx]);
          const CustomSerdezID dst_serdez =
            runtime->forest->get_field_serdez(dst_space, dst_fields[fidx]);
           if (src_serdez != dst_serdez)
            REPORT_LEGION_ERROR(ERROR_FIELD_SERDEZ_MISMATCH,
                "Field with different serdez modes are not permitted for "
                "region-to-region copy operations. Fields %d and %d of "
                "region requirement %d have different serdez modes "
                "(%d and %d respectively) in copy operation (UID %lld) in "
                "parent task %s (UID %lld).",
                src_fields[fidx], dst_fields[fidx], idx, src_serdez, 
                dst_serdez, get_unique_id(), parent_ctx->get_task_name(),
                parent_ctx->get_unique_id())
        }
        if (idx < src_indirect_requirements.size())
        {
          // Check that the size of the source indirect field is same 
          // as the size of the source coordinate type
          const RegionRequirement &src_idx_req = src_indirect_requirements[idx];
          const FieldID fid = *src_idx_req.privilege_fields.begin();
          const size_t idx_size = runtime->forest->get_field_size(
              src_idx_req.parent.get_field_space(), fid);
          const IndexSpace src_space =
            src_requirements[idx].parent.get_index_space();
          const size_t coord_size = 
            runtime->forest->get_coordinate_size(src_space, false/*range*/);
          if (idx_size != coord_size)
            REPORT_LEGION_ERROR(ERROR_TYPE_FIELD_MISMATCH,
                "The source indirect field for a copy operation has the "
                "incorrect size for the source region coordinate space. "
                "Field %d of source indirect region requirement %d is "
                "%zd bytes but the coordinate types of the source space "
                "is %zd bytes for copy operation (UID %lld) in parent task "
                "%s (UID %lld).", fid, idx, idx_size, coord_size,
                get_unique_id(), parent_ctx->get_task_name(),
                parent_ctx->get_unique_id())
          const CustomSerdezID idx_serdez = runtime->forest->get_field_serdez(
              src_idx_req.parent.get_field_space(), fid);
          if (idx_serdez != 0)
            REPORT_LEGION_ERROR(ERROR_SERDEZ_FIELD_DISALLOWED,
                "Serdez fields are not permitted to be used as "
                "indirection fields for copy operations. Field %d "
                "of source indirect region requirement %d in copy "
                "(UID %lld) in parent task %s (UID %lld) has serdez "
                "function %d.", fid, idx, get_unique_id(),
                parent_ctx->get_task_name(), 
                parent_ctx->get_unique_id(), idx_serdez)
        }
        if (idx >= dst_indirect_requirements.size())
        {
          if (idx >= src_indirect_requirements.size())
          {
            // Normal copy
            IndexSpace src_space = 
              src_requirements[idx].parent.get_index_space();
            IndexSpace dst_space = 
              dst_requirements[idx].parent.get_index_space();
            bool diff_dims = false;
            if (!runtime->forest->check_types(src_space.get_type_tag(),
                                  dst_space.get_type_tag(), diff_dims))
              REPORT_LEGION_ERROR(ERROR_TYPE_MISMATCH,
                            "Copy launcher index space mismatch at index "
                            "%d of cross-region copy (ID %lld) in task %s "
                            "(ID %lld). The index spaces of the source "
                            "and destination requirements have incompatible "
                            "types because they have different %s.",
                            idx, get_unique_id(),
                            parent_ctx->get_task_name(), 
                            parent_ctx->get_unique_id(), diff_dims ? 
                              "numbers of dimensions" : "coordinate types")
          }
          else
          {
            // Gather copy
            IndexSpace src_indirect_space = 
              src_indirect_requirements[idx].parent.get_index_space();
            IndexSpace dst_space = 
              dst_requirements[idx].parent.get_index_space();
            bool diff_dims = false;
            if (!runtime->forest->check_types(src_indirect_space.get_type_tag(),
                                           dst_space.get_type_tag(), diff_dims))
              REPORT_LEGION_ERROR(ERROR_TYPE_MISMATCH,
                            "Copy launcher index space mismatch at index "
                            "%d of cross-region copy (ID %lld) in task %s "
                            "(ID %lld). The index spaces of the source "
                            "indirect requirement and the destination "
                            "requirement have incompatible types because "
                            "they have different %s.",
                            idx, get_unique_id(),
                            parent_ctx->get_task_name(), 
                            parent_ctx->get_unique_id(), diff_dims ?
                              "numbers of dimensions" : "coordinate types")
          }
        }
        else
        {
          // Check that the size of the source indirect field is same 
          // as the size of the source coordinate type
          const RegionRequirement &dst_idx_req = dst_indirect_requirements[idx];
          const FieldID fid = *dst_idx_req.privilege_fields.begin();
          const size_t idx_size = runtime->forest->get_field_size(
              dst_idx_req.parent.get_field_space(), fid);
          const IndexSpace dst_space =
            dst_requirements[idx].parent.get_index_space();
          const size_t coord_size = 
            runtime->forest->get_coordinate_size(dst_space, false/*range*/);
          if (idx_size != coord_size)
            REPORT_LEGION_ERROR(ERROR_TYPE_FIELD_MISMATCH,
                "The destination indirect field for a copy operation has the "
                "incorrect size for the destination region coordinate space. "
                "Field %d of destination indirect region requirement %d is "
                "%zd bytes but the coordinate types of the destination space "
                "is %zd bytes for copy operation (UID %lld) in parent task "
                "%s (UID %lld).", fid, idx, idx_size, coord_size,
                get_unique_id(), parent_ctx->get_task_name(),
                parent_ctx->get_unique_id())
          const CustomSerdezID idx_serdez = runtime->forest->get_field_serdez(
              dst_idx_req.parent.get_field_space(), fid);
          if (idx_serdez != 0)
            REPORT_LEGION_ERROR(ERROR_SERDEZ_FIELD_DISALLOWED,
                "Serdez fields are not permitted to be used as "
                "indirection fields for copy operations. Field %d "
                "of destination indirect region requirement %d in copy "
                "(UID %lld) in parent task %s (UID %lld) has serdez "
                "function %d.", fid, idx, get_unique_id(),
                parent_ctx->get_task_name(), 
                parent_ctx->get_unique_id(), idx_serdez)
          if (idx >= src_indirect_requirements.size())
          {
            // Scatter copy
            IndexSpace src_space = 
              src_requirements[idx].parent.get_index_space();
            IndexSpace dst_indirect_space= 
              dst_indirect_requirements[idx].parent.get_index_space();
            // Just check compatibility here since it's really hard to
            // prove that we're actually going to write everything
            bool diff_dims = false;
            if (!runtime->forest->check_types(src_space.get_type_tag(), 
                          dst_indirect_space.get_type_tag(), diff_dims))
              REPORT_LEGION_ERROR(ERROR_TYPE_MISMATCH,
                            "Copy launcher index space mismatch at index "
                            "%d of cross-region copy (ID %lld) in task %s "
                            "(ID %lld). The index spaces of the source "
                            "requirement and the destination indirect "
                            "requirement have incompatible types because " 
                            "they have different %s.",
                            idx, get_unique_id(),
                            parent_ctx->get_task_name(), 
                            parent_ctx->get_unique_id(), diff_dims ?
                              "numbers of dimensions" : "coordinate types ")
          }
          else
          {
            // Indirect copy
            IndexSpace src_indirect_space = 
              src_indirect_requirements[idx].parent.get_index_space();
            IndexSpace dst_indirect_space= 
              dst_indirect_requirements[idx].parent.get_index_space();
            // Just check compatibility here since it's really hard to
            // prove that we're actually going to write everything
            bool diff_dims = false;
            if (!runtime->forest->check_types(src_indirect_space.get_type_tag(),
                                  dst_indirect_space.get_type_tag(), diff_dims))
              REPORT_LEGION_ERROR(ERROR_TYPE_MISMATCH,
                            "Copy launcher index space mismatch at index "
                            "%d of cross-region copy (ID %lld) in task %s "
                            "(ID %lld). The index spaces of the source "
                            "indirect requirement and the destination indirect "
                            "requirement have incompatible types because "
                            "they have different %s.",
                            idx, get_unique_id(),
                            parent_ctx->get_task_name(), 
                            parent_ctx->get_unique_id(), diff_dims ?
                              "numbers of dimensions" : "coordinate types")
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyOp::activate_copy(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative();
      activate_memoizable();
      mapper = NULL;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
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
      src_indirect_requirements.clear();
      dst_indirect_requirements.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      src_privilege_paths.clear();
      dst_privilege_paths.clear();
      gather_privilege_paths.clear();
      scatter_privilege_paths.clear();
      src_parent_indexes.clear();
      dst_parent_indexes.clear();
      gather_parent_indexes.clear();
      scatter_parent_indexes.clear();
      src_versions.clear();
      dst_versions.clear();
      gather_versions.clear();
      scatter_versions.clear();
      src_indirect_records.clear();
      dst_indirect_records.clear();
      gather_is_range.clear();
      scatter_is_range.clear();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      atomic_locks.clear();
      map_applied_conditions.clear();
      profiling_requests.clear();
      if (!profiling_info.empty())
      {
        for (unsigned idx = 0; idx < profiling_info.size(); idx++)
          free(profiling_info[idx].buffer);
        profiling_info.clear();
      }
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
      if (!src_indirect_requirements.empty())
      {
        const size_t offset = src_requirements.size() + dst_requirements.size();
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          const RegionRequirement &req = src_indirect_requirements[idx];
#ifdef DEBUG_LEGION
          assert(req.privilege_fields.size() == 1);
#endif
          LegionSpy::log_logical_requirement(unique_op_id, offset + idx,
                                             true/*region*/,
                                             req.region.index_space.id,
                                             req.region.field_space.id,
                                             req.region.tree_id,
                                             req.privilege,
                                             req.prop, req.redop,
                                             req.parent.index_space.id);
          LegionSpy::log_requirement_fields(unique_op_id, offset + idx, 
                                            req.privilege_fields);
        }
      }
      if (!dst_indirect_requirements.empty())
      {
        const size_t offset = src_requirements.size() + 
          dst_requirements.size() + src_indirect_requirements.size();
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          const RegionRequirement &req = dst_indirect_requirements[idx];
#ifdef DEBUG_LEGION
          assert(req.privilege_fields.size() == 1);
#endif
          LegionSpy::log_logical_requirement(unique_op_id, offset + idx,
                                             true/*region*/,
                                             req.region.index_space.id,
                                             req.region.field_space.id,
                                             req.region.tree_id,
                                             req.privilege,
                                             req.prop, req.redop,
                                             req.parent.index_space.id);
          LegionSpy::log_requirement_fields(unique_op_id, offset + idx, 
                                            req.privilege_fields);
        }
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
      if (!src_indirect_requirements.empty())
      {
        gather_privilege_paths.resize(src_indirect_requirements.size());
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          initialize_privilege_path(gather_privilege_paths[idx],
                                    src_indirect_requirements[idx]);
        }
      }
      if (!dst_indirect_requirements.empty())
      {
        scatter_privilege_paths.resize(dst_indirect_requirements.size());
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          initialize_privilege_path(scatter_privilege_paths[idx],
                                    dst_indirect_requirements[idx]);
        }
      } 
      if (runtime->legion_spy_enabled)
        log_copy_requirements();
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->check_privileges)
        check_copy_privileges(false/*permit projection*/);
      // Register a dependence on our predicate
      register_predicate_dependence();
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(this, 
                              wait_barriers, arrive_barriers);
      ProjectionInfo projection_info;
      src_versions.resize(src_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        runtime->forest->perform_dependence_analysis(this, idx, 
                                                     src_requirements[idx],
                                                     projection_info,
                                                     src_privilege_paths[idx],
                                                     map_applied_conditions);
      dst_versions.resize(dst_requirements.size());
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        unsigned index = src_requirements.size()+idx;
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_READ_WRITE;
        runtime->forest->perform_dependence_analysis(this, index, 
                                                     dst_requirements[idx],
                                                     projection_info,
                                                     dst_privilege_paths[idx],
                                                     map_applied_conditions);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_REDUCE;
      }
      if (!src_indirect_requirements.empty())
      {
        gather_versions.resize(src_indirect_requirements.size());
        const size_t offset = src_requirements.size() + dst_requirements.size();
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
          runtime->forest->perform_dependence_analysis(this, offset + idx, 
                                                 src_indirect_requirements[idx],
                                                 projection_info,
                                                 gather_privilege_paths[idx],
                                                 map_applied_conditions);
      }
      if (!dst_indirect_requirements.empty())
      {
        scatter_versions.resize(dst_indirect_requirements.size());
        const size_t offset = src_requirements.size() +
          dst_requirements.size() + src_indirect_requirements.size();
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
          runtime->forest->perform_dependence_analysis(this, offset + idx, 
                                                 dst_indirect_requirements[idx],
                                                 projection_info,
                                                 scatter_privilege_paths[idx],
                                                 map_applied_conditions);
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
        if (HAS_WRITE_DISCARD(req))
          req.privilege &= ~LEGION_DISCARD_MASK;
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
      if (is_replaying())
      {
        enqueue_ready_operation();
        return;
      }

      // Do our versioning analysis and then add it to the ready queue
      std::set<RtEvent> preconditions;
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        runtime->forest->perform_versioning_analysis(this, idx,
                                                     src_requirements[idx],
                                                     src_versions[idx],
                                                     preconditions);
      unsigned offset = src_requirements.size();
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_READ_WRITE;
        runtime->forest->perform_versioning_analysis(this, offset + idx,
                                                     dst_requirements[idx],
                                                     dst_versions[idx],
                                                     preconditions);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_REDUCE;
      }
      offset += dst_requirements.size();
      if (!src_indirect_requirements.empty())
      {
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
          runtime->forest->perform_versioning_analysis(this, offset + idx,
                                                 src_indirect_requirements[idx],
                                                 gather_versions[idx],
                                                 preconditions);
        offset += src_indirect_requirements.size();
      }
      if (!dst_indirect_requirements.empty())
      {
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
          runtime->forest->perform_versioning_analysis(this, offset + idx,
                                                 dst_indirect_requirements[idx],
                                                 scatter_versions[idx],
                                                 preconditions);
      }
      if (!preconditions.empty())
      {
        const RtEvent ready = Runtime::merge_events(preconditions);
        enqueue_ready_operation(ready);
      }
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      const TraceInfo trace_info(this, true/*initialize*/);
      std::vector<InstanceSet> valid_src_instances(src_requirements.size());
      std::vector<InstanceSet> valid_dst_instances(dst_requirements.size());
      std::vector<InstanceSet> valid_gather_instances(
                                          src_indirect_requirements.size());
      std::vector<InstanceSet> valid_scatter_instances(
                                          dst_indirect_requirements.size());
      Mapper::MapCopyInput input;
      Mapper::MapCopyOutput output;
      input.src_instances.resize(src_requirements.size());
      input.dst_instances.resize(dst_requirements.size());
      input.src_indirect_instances.resize(src_indirect_requirements.size());
      input.dst_indirect_instances.resize(dst_indirect_requirements.size());
      output.src_instances.resize(src_requirements.size());
      output.dst_instances.resize(dst_requirements.size());
      output.src_indirect_instances.resize(src_indirect_requirements.size());
      output.dst_indirect_instances.resize(dst_indirect_requirements.size());
      atomic_locks.resize(dst_requirements.size());
      output.profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      output.compute_preimages = false;
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      if (mapper->request_valid_instances)
      {
        // First go through and do the traversals to find the valid instances
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        {
          InstanceSet &valid_instances = valid_src_instances[idx];
          runtime->forest->physical_premap_region(this, idx, 
                                                  src_requirements[idx],
                                                  src_versions[idx],
                                                  valid_instances,
                                                  map_applied_conditions);
          // Convert these to the valid set of mapping instances
          // No need to filter for copies
          prepare_for_mapping(valid_instances, input.src_instances[idx]);
        }
        for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
        {
          InstanceSet &valid_instances = valid_dst_instances[idx];
          // Little bit of a hack here, if we are going to do a reduction
          // explicit copy, switch the privileges to read-write when doing
          // the registration since we know we are using normal instances
          const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
          if (is_reduce_req)
            dst_requirements[idx].privilege = LEGION_READ_WRITE;
          runtime->forest->physical_premap_region(this, 
                                                  idx+src_requirements.size(),
                                                  dst_requirements[idx],
                                                  dst_versions[idx],
                                                  valid_instances,
                                                  map_applied_conditions);
          // No need to filter for copies
          prepare_for_mapping(valid_instances, input.dst_instances[idx]);
          // Switch the privileges back when we are done
          if (is_reduce_req)
            dst_requirements[idx].privilege = LEGION_REDUCE;
        }
        if (!src_indirect_requirements.empty())
        {
          const unsigned offset = 
            src_requirements.size() + dst_requirements.size();
          for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
          {
            InstanceSet &valid_instances = valid_gather_instances[idx];
            runtime->forest->physical_premap_region(this, offset+idx, 
                                                src_indirect_requirements[idx],
                                                gather_versions[idx],
                                                valid_instances,
                                                map_applied_conditions);
            // Convert these to the valid set of mapping instances
            // No need to filter for copies
            prepare_for_mapping(valid_instances, 
                                input.src_indirect_instances[idx]);
          }
        }
        if (!dst_indirect_requirements.empty())
        {
          const unsigned offset = src_requirements.size() + 
            dst_requirements.size() + src_indirect_requirements.size();
          for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
          {
            InstanceSet &valid_instances = valid_scatter_instances[idx];
            runtime->forest->physical_premap_region(this, offset+idx, 
                                                dst_indirect_requirements[idx],
                                                scatter_versions[idx],
                                                valid_instances,
                                                map_applied_conditions);
            // Convert these to the valid set of mapping instances
            // No need to filter for copies
            prepare_for_mapping(valid_instances, 
                                input.dst_indirect_instances[idx]);
          }
        }
      }
      // Now we can ask the mapper what to do 
      mapper->invoke_map_copy(this, &input, &output);
      if (!output.profiling_requests.empty())
      {
        filter_copy_request_kinds(mapper,
            output.profiling_requests.requested_measurements,
            profiling_requests, true/*warn*/);
        profiling_priority = output.profiling_priority;
#ifdef DEBUG_LEGION
        assert(!profiling_reported.exists());
#endif
        profiling_reported = Runtime::create_rt_user_event();
      }
      // Resize these now so they don't change later
      if (!src_indirect_requirements.empty())
        src_indirect_records.resize(src_indirect_requirements.size());
      if (!dst_indirect_requirements.empty())
        dst_indirect_records.resize(dst_indirect_requirements.size());
      // Now we can carry out the mapping requested by the mapper
      // and issue the across copies, first set up the sync precondition
      ApEvent init_precondition = compute_init_precondition(trace_info);
      // Register the source and destination regions
      std::set<ApEvent> copy_complete_events;
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        InstanceSet src_targets, dst_targets, gather_targets, scatter_targets;
        // Make a user event for when this copy across is done
        // and add it to the set of copy complete events
        const ApUserEvent local_postcondition = 
          Runtime::create_ap_user_event(&trace_info);
        copy_complete_events.insert(local_postcondition); 
        // Convert the src_targets and dst_targets first so we can do any
        // exchanges for collective points
        // The common case 
        int src_virtual = -1;
        // Do the conversion and check for errors
        src_virtual = 
          perform_conversion<SRC_REQ>(idx, src_requirements[idx],
                                      output.src_instances[idx],
                                      src_targets,
                                      IS_REDUCE(dst_requirements[idx]));
        if (runtime->legion_spy_enabled)
          runtime->forest->log_mapping_decision(unique_op_id, parent_ctx, 
                                                idx, src_requirements[idx],
                                                src_targets);
        const size_t dst_idx = src_requirements.size() + idx;
        // Little bit of a hack here, if we are going to do a reduction
        // explicit copy, switch the privileges to read-write when doing
        // the registration since we know we are using normal instances
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_READ_WRITE;
        perform_conversion<DST_REQ>(idx, dst_requirements[idx],
                                    output.dst_instances[idx], dst_targets);
        if (runtime->legion_spy_enabled)
          runtime->forest->log_mapping_decision(unique_op_id, parent_ctx,
              dst_idx, dst_requirements[idx], dst_targets);
        // Do any exchanges needed for collective cooperation
        const bool src_indirect = (idx < src_indirect_requirements.size());
        const bool dst_indirect = (idx < dst_indirect_requirements.size());
        const ApUserEvent local_precondition = (src_indirect || dst_indirect) ?
          Runtime::create_ap_user_event(&trace_info) : 
          ApUserEvent::NO_AP_USER_EVENT;
        ApEvent collective_precondition, collective_postcondition;
        // Track applied conditions special for copy-across
        std::set<RtEvent> perform_ready_events;
        if (src_indirect)
        {
          // Do the exchange to get it in flight
          RtEvent exchange_done = exchange_indirect_records(idx, 
              local_precondition, local_postcondition, collective_precondition,
              collective_postcondition, trace_info, src_targets,
              src_requirements[idx], index_point, 
              src_indirect_records[idx], true/*source*/);
          if (exchange_done.exists())
            perform_ready_events.insert(exchange_done);
        }
        if (dst_indirect)
        {
          // It's ok to overwrite the collective postcondition because we 
          // guarantee that they will be the same for multiple calls
          // to exchange for the same operation
          RtEvent exchange_done = exchange_indirect_records(idx, 
              local_precondition, local_postcondition, collective_precondition,
              collective_postcondition, trace_info, dst_targets,
              dst_requirements[idx], index_point, 
              dst_indirect_records[idx], false/*source*/);
          if (exchange_done.exists())
            perform_ready_events.insert(exchange_done);
        }
        if (src_virtual < 0)
        {
          // Don't track source views of copy across operations here,
          // as they will do later when the realm copies are recorded.
          PhysicalTraceInfo src_info(trace_info, idx, false/*update validity*/);
          const bool record_valid = (output.untracked_valid_srcs.find(idx) ==
                                     output.untracked_valid_srcs.end());
          runtime->forest->physical_perform_updates_and_registration(
                                              src_requirements[idx],
                                              src_versions[idx],
                                              this, idx,
                                              init_precondition,
                                              src_indirect ? 
                                                collective_postcondition :
                                                (ApEvent)local_postcondition,
                                              src_targets,
                                              src_info,
                                              map_applied_conditions,
#ifdef DEBUG_LEGION
                                              get_logging_name(),
                                              unique_op_id,
#endif
                                              record_valid);
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(src_targets.size() == 1);
          assert(src_targets[0].is_virtual_ref());
#endif
          src_targets.clear();
          // This is a bit weird but we don't currently have any mechanism
          // for passing the reservations that we find in these cases through
          // to the CopyAcrossAnalysis and through the CopyFillAggregator so
          // for now we're just going to promote privileges on any source and
          // destination requirements to exclusive which is sound with the 
          // logical dependence analysis since we're not changing privileges
          if (IS_ATOMIC(src_requirements[idx]))
            src_requirements[idx].prop = LEGION_EXCLUSIVE;
          if (IS_ATOMIC(dst_requirements[idx]))
            dst_requirements[idx].prop = LEGION_EXCLUSIVE;
        } 
        // Don't track target views of copy across operations here,
        // as they will do later when the realm copies are recorded.
        PhysicalTraceInfo dst_info(trace_info,dst_idx,false/*update_validity*/); 
        ApEvent effects_done = 
          runtime->forest->physical_perform_updates_and_registration(
                                          dst_requirements[idx],
                                          dst_versions[idx], this,
                                          dst_idx,
                                          init_precondition,
                                          dst_indirect ? 
                                            collective_postcondition :
                                            (ApEvent)local_postcondition,
                                          dst_targets,
                                          dst_info,
                                          (src_virtual >= 0) ?
                                            perform_ready_events :
                                            map_applied_conditions,
#ifdef DEBUG_LEGION
                                          get_logging_name(),
                                          unique_op_id,
#endif
                                          true/*record valid*/,
                      // Only check initialized if we don't have an indirection.
                      // If we have an indirection then it is impossible to know
                      // if we writing everything or not
                                    (idx >= src_indirect_requirements.size()) &&
                                     (idx >= dst_indirect_requirements.size()));
        if (effects_done.exists())
          copy_complete_events.insert(effects_done);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_REDUCE; 
        if (idx < src_indirect_requirements.size())
        { 
          std::vector<MappingInstance> gather_instances(1);
          if (idx < output.src_indirect_instances.size())
            gather_instances[0] = output.src_indirect_instances[idx];
          else
            gather_instances.clear();
          perform_conversion<GATHER_REQ>(idx, src_indirect_requirements[idx],
                                         gather_instances, gather_targets);
          // Now do the registration
          const size_t gather_idx = src_requirements.size() + 
            dst_requirements.size() + idx;
          PhysicalTraceInfo gather_info(trace_info, gather_idx);
          const bool record_valid = (output.untracked_valid_ind_srcs.find(idx) 
                                    == output.untracked_valid_ind_srcs.end());
          ApEvent effects_done = 
            runtime->forest->physical_perform_updates_and_registration(
                                       src_indirect_requirements[idx],
                                       gather_versions[idx], this,
                                       gather_idx,
                                       init_precondition,
                                       local_postcondition,
                                       gather_targets,
                                       gather_info,
                                       map_applied_conditions,
#ifdef DEBUG_LEGION
                                       get_logging_name(),
                                       unique_op_id,
#endif
                                       record_valid);
          if (effects_done.exists())
            copy_complete_events.insert(effects_done);
          if (runtime->legion_spy_enabled)
            runtime->forest->log_mapping_decision(unique_op_id, parent_ctx,
                gather_idx, src_indirect_requirements[idx], gather_targets);
        }
        if (idx < dst_indirect_requirements.size())
        { 
          std::vector<MappingInstance> scatter_instances(1);
          if (idx < output.dst_indirect_instances.size())
            scatter_instances[0] = output.dst_indirect_instances[idx];
          else
            scatter_instances.clear();
          perform_conversion<SCATTER_REQ>(idx, dst_indirect_requirements[idx],
                                          scatter_instances, scatter_targets);
          // Now do the registration
          const size_t scatter_idx = src_requirements.size() + 
            dst_requirements.size() + src_indirect_requirements.size() + idx;
          const bool record_valid = (output.untracked_valid_ind_dsts.find(idx) 
                                    == output.untracked_valid_ind_dsts.end());
          PhysicalTraceInfo scatter_info(trace_info, scatter_idx);
          ApEvent effects_done = 
            runtime->forest->physical_perform_updates_and_registration(
                                      dst_indirect_requirements[idx],
                                      scatter_versions[idx], this,
                                      scatter_idx,
                                      init_precondition,
                                      local_postcondition,
                                      scatter_targets,
                                      scatter_info,
                                      map_applied_conditions,
#ifdef DEBUG_LEGION
                                      get_logging_name(),
                                      unique_op_id,
#endif
                                      record_valid);
          if (effects_done.exists())
            copy_complete_events.insert(effects_done);
          if (runtime->legion_spy_enabled)
            runtime->forest->log_mapping_decision(unique_op_id, parent_ctx, 
                scatter_idx, dst_indirect_requirements[idx], scatter_targets);
        }
        // If we made it here, we passed all our error-checking so
        // now we can issue the copy/reduce across operation
        // If we have local completion events then we need to make
        // sure that all those effects have been applied before we
        // can perform the copy across operation, so defer it if necessary
        PhysicalTraceInfo physical_trace_info(idx, trace_info,
                                idx + src_requirements.size());
        RtEvent perform_precondition;
        if (!perform_ready_events.empty())
          perform_precondition = Runtime::merge_events(perform_ready_events);
        if (perform_precondition.exists() &&
            !perform_precondition.has_triggered())
        {
          InstanceSet *deferred_src = new InstanceSet();
          deferred_src->swap(src_targets);
          InstanceSet *deferred_dst = new InstanceSet();
          deferred_dst->swap(dst_targets);
          InstanceSet *deferred_gather = NULL;
          if (!gather_targets.empty())
          {
            deferred_gather = new InstanceSet();
            deferred_gather->swap(gather_targets);
          }
          InstanceSet *deferred_scatter = NULL;
          if (!scatter_targets.empty())
          {
            deferred_scatter = new InstanceSet();
            deferred_scatter->swap(scatter_targets);
          }
          RtUserEvent deferred_applied = Runtime::create_rt_user_event();
          DeferredCopyAcross args(this, physical_trace_info, idx, 
                                  init_precondition, local_precondition,
                                  local_postcondition, collective_precondition,
                                  collective_postcondition, predication_guard,
                                  deferred_applied, deferred_src, deferred_dst,
                                  deferred_gather, deferred_scatter,
                                  output.compute_preimages);
          runtime->issue_runtime_meta_task(args, 
              LG_THROUGHPUT_DEFERRED_PRIORITY, perform_precondition);
          map_applied_conditions.insert(deferred_applied);
        }
        else
          perform_copy_across(idx, init_precondition, local_precondition,
                              local_postcondition, collective_precondition,
                              collective_postcondition,
                              predication_guard, src_targets, dst_targets, 
                              gather_targets.empty() ? NULL : &gather_targets,
                              scatter_targets.empty() ? NULL : &scatter_targets,
                              physical_trace_info, map_applied_conditions,
                              output.compute_preimages);
      }
      ApEvent copy_complete_event = 
        Runtime::merge_events(&trace_info, copy_complete_events);
#ifdef LEGION_SPY
      if (runtime->legion_spy_enabled)
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
          if (runtime->legion_spy_enabled)
            LegionSpy::log_phase_barrier_arrival(unique_op_id, 
                                                 it->phase_barrier);
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        completion_event);    
        }
      }
      if (is_recording())
        tpl->record_complete_replay(this, copy_complete_event);
      // Mark that we completed mapping
      RtEvent mapping_applied;
      if (!map_applied_conditions.empty())
        mapping_applied = Runtime::merge_events(map_applied_conditions);
      if (!acquired_instances.empty())
        mapping_applied = release_nonempty_acquired_instances(mapping_applied, 
                                                          acquired_instances);
      complete_mapping(mapping_applied);
      // Handle the case for marking when the copy completes
      if (!request_early_complete(copy_complete_event))
        complete_execution(Runtime::protect_event(copy_complete_event));
      else
        complete_execution();
    }

    //--------------------------------------------------------------------------
    void CopyOp::perform_copy_across(const unsigned index, 
                                     const ApEvent init_precondition,
                                     const ApUserEvent local_precondition,
                                     const ApUserEvent local_postcondition,
                                     const ApEvent collective_precondition,
                                     const ApEvent collective_postcondition,
                                     const PredEvent predication_guard,
                                     const InstanceSet &src_targets,
                                     const InstanceSet &dst_targets,
                                     const InstanceSet *gather_targets,
                                     const InstanceSet *scatter_targets,
                                     const PhysicalTraceInfo &trace_info,
                                     std::set<RtEvent> &applied_conditions,
                                     const bool compute_preimages)
    //--------------------------------------------------------------------------
    {
      ApEvent copy_post;
      if (scatter_targets == NULL)
      {
        if (gather_targets == NULL)
        {
#ifdef DEBUG_LEGION
          assert(!local_precondition.exists());
#endif
          // Normal copy across
          copy_post = runtime->forest->copy_across( 
              src_requirements[index], dst_requirements[index],
              src_versions[index], dst_versions[index],
              src_targets, dst_targets, this, index, trace_info.dst_index,
              init_precondition, predication_guard, 
              atomic_locks[index], trace_info, applied_conditions);
        }
        else
        {
          // Gather copy
#ifdef DEBUG_LEGION
          assert(index < src_indirect_records.size());
          assert(!src_indirect_records[index].empty());
#endif
          copy_post = runtime->forest->gather_across(
              src_requirements[index], src_indirect_requirements[index],
              dst_requirements[index], src_indirect_records[index], 
              src_targets, (*gather_targets), dst_targets, this, index, 
              src_requirements.size() + dst_requirements.size() + index,
              src_requirements.size() + index, gather_is_range[index],
              init_precondition, predication_guard, collective_precondition,
              collective_postcondition, local_precondition, 
              atomic_locks[index], trace_info, applied_conditions, 
              possible_src_indirect_out_of_range, compute_preimages);
        }
      }
      else
      {
        if (gather_targets == NULL)
        {
          // Scatter copy
#ifdef DEBUG_LEGION
          assert(index < dst_indirect_records.size());
          assert(!dst_indirect_records[index].empty());
#endif
          copy_post = runtime->forest->scatter_across(
              src_requirements[index], dst_indirect_requirements[index],
              dst_requirements[index], src_targets, (*scatter_targets),
              dst_targets, dst_indirect_records[index], this, index, 
              src_requirements.size() + dst_requirements.size() + index,
              src_requirements.size() + index, scatter_is_range[index],
              init_precondition, predication_guard, collective_precondition,
              collective_postcondition, local_precondition, 
              atomic_locks[index], trace_info, applied_conditions, 
              possible_dst_indirect_out_of_range, 
              possible_dst_indirect_aliasing, compute_preimages);
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(gather_is_range[index] == scatter_is_range[index]);
          assert(index < src_indirect_records.size());
          assert(!src_indirect_records[index].empty());
          assert(index < dst_indirect_records.size());
          assert(!dst_indirect_records[index].empty());
#endif
          // Full indirection copy
          copy_post = runtime->forest->indirect_across(
              src_requirements[index], src_indirect_requirements[index],
              dst_requirements[index], dst_indirect_requirements[index],
              src_targets, dst_targets, src_indirect_records[index], 
              (*gather_targets), dst_indirect_records[index], 
              (*scatter_targets), this, index,
              src_requirements.size() + index,
              src_requirements.size() + dst_requirements.size() + index,
              src_requirements.size() + dst_requirements.size() +
              src_indirect_requirements.size() + index, gather_is_range[index],
              init_precondition, predication_guard, collective_precondition,
              collective_postcondition, local_precondition, atomic_locks[index],
              trace_info,applied_conditions, possible_src_indirect_out_of_range,
              possible_dst_indirect_out_of_range,
              possible_dst_indirect_aliasing, compute_preimages);
        }
      }
      if (is_recording())
      {
#ifdef DEBUG_LEGION
        assert((tpl != NULL) && tpl->is_recording());
#endif
        // This can happen in cases when the copy index space is empty
        if (!copy_post.exists())
          copy_post = execution_fence_event;
      }
      Runtime::trigger_event(&trace_info, local_postcondition, copy_post);
#ifdef DEBUG_LEGION
      dump_physical_state(&src_requirements[index], index);
      dump_physical_state(&dst_requirements[index], 
                          index+ src_requirements.size());
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void CopyOp::handle_deferred_across(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferredCopyAcross *dargs = (const DeferredCopyAcross*)args;
      std::set<RtEvent> applied_conditions;
      dargs->copy->perform_copy_across(dargs->index, dargs->init_precondition,
                            dargs->local_precondition, 
                            dargs->local_postcondition,
                            dargs->collective_precondition,
                            dargs->collective_postcondition, dargs->guard, 
                            *dargs->src_targets, *dargs->dst_targets, 
                            dargs->gather_targets, dargs->scatter_targets,
                            *dargs,applied_conditions,dargs->compute_preimages);
      if (!applied_conditions.empty())
        Runtime::trigger_event(dargs->applied, 
            Runtime::merge_events(applied_conditions));
      else
        Runtime::trigger_event(dargs->applied);
      delete dargs->src_targets;
      delete dargs->dst_targets;
      if (dargs->gather_targets != NULL)
        delete dargs->gather_targets;
      if (dargs->scatter_targets != NULL)
        delete dargs->scatter_targets;
      dargs->remove_recorder_reference();
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      if (profiling_reported.exists())
        finalize_copy_profiling();
      commit_operation(true/*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    void CopyOp::finalize_copy_profiling(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(profiling_reported.exists());
#endif
      if (outstanding_profiling_requests.load() > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapped_event.has_triggered());
#endif
        std::vector<CopyProfilingInfo> to_perform;
        {
          AutoLock o_lock(op_lock);
          to_perform.swap(profiling_info);
        }
        if (!to_perform.empty())
        {
          for (unsigned idx = 0; idx < to_perform.size(); idx++)
          {
            CopyProfilingInfo &info = to_perform[idx];
            const Realm::ProfilingResponse resp(info.buffer, info.buffer_size);
            info.total_reports = outstanding_profiling_requests.load();
            info.profiling_responses.attach_realm_profiling_response(resp);
            mapper->invoke_copy_report_profiling(this, &info);
            free(info.buffer);
          }
          const int count = to_perform.size() +
              outstanding_profiling_reported.fetch_add(to_perform.size());
#ifdef DEBUG_LEGION
          assert(count <= outstanding_profiling_requests.load());
#endif
          if (count == outstanding_profiling_requests.load())
            Runtime::trigger_event(profiling_reported);
        }
      }
      else
      {
        // We're not expecting any profiling callbacks so we need to
        // do one ourself to inform the mapper that there won't be any
        Mapping::Mapper::CopyProfilingInfo info;
        info.total_reports = 0;
        info.src_index = 0;
        info.dst_index = 0;
        info.fill_response = false; // make valgrind happy
        mapper->invoke_copy_report_profiling(this, &info);    
        Runtime::trigger_event(profiling_reported);
      }
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
    RtEvent CopyOp::exchange_indirect_records(
        const unsigned index, const ApEvent local_pre, const ApEvent local_post,
        ApEvent &collective_pre, ApEvent &collective_post,
        const TraceInfo &trace_info, const InstanceSet &insts,
        const RegionRequirement &req, const DomainPoint &key,
        std::vector<IndirectRecord> &records, const bool sources)
    //--------------------------------------------------------------------------
    {
      collective_pre = local_pre;
      collective_post = local_post;
      records.emplace_back(IndirectRecord(runtime->forest, req, insts, key));
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    unsigned CopyOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      if (idx < src_parent_indexes.size())
        return src_parent_indexes[idx];
      idx -= src_parent_indexes.size();
      if (idx < dst_parent_indexes.size())
        return dst_parent_indexes[idx];
      idx -= dst_parent_indexes.size();
      if (idx < gather_parent_indexes.size())
        return gather_parent_indexes[idx];
      idx -= gather_parent_indexes.size();
#ifdef DEBUG_LEGION
      assert(idx < scatter_parent_indexes.size());
#endif
      return scatter_parent_indexes[idx];
    }

    //--------------------------------------------------------------------------
    void CopyOp::select_sources(const unsigned index,
                                const InstanceRef &target,
                                const InstanceSet &sources,
                                std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      Mapper::SelectCopySrcInput input;
      Mapper::SelectCopySrcOutput output;
      prepare_for_mapping(sources, input.source_instances);
      prepare_for_mapping(target, input.target);
      input.is_src = false;
      input.is_dst = false;
      input.is_src_indirect = false;
      input.is_dst_indirect = false;
      unsigned mod_index = index;
      if (mod_index < src_requirements.size())
      {
        input.region_req_index = mod_index;
        input.is_src = true;
      }
      else
      {
        mod_index -= src_requirements.size();
        if (mod_index < dst_requirements.size())
        {
          input.region_req_index = mod_index;
          input.is_dst = true;
        }
        else
        {
          mod_index -= dst_requirements.size();
          if (mod_index < src_indirect_requirements.size())
          {
            input.region_req_index = mod_index;
            input.is_src_indirect = true;
          }
          else
          {
            mod_index -= src_indirect_requirements.size();
#ifdef DEBUG_LEGION
            assert(mod_index < dst_indirect_requirements.size());
#endif
            input.is_dst_indirect = true;
          }
        }
      }
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_select_copy_sources(this, &input, &output);
      // Fill in the ranking based on the output
      compute_ranking(mapper, output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,unsigned>* 
                                        CopyOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void CopyOp::update_atomic_locks(const unsigned index,
                                     Reservation lock, bool exclusive)
    //--------------------------------------------------------------------------
    {
      // Figure out which index this should actually be for
      unsigned mod_index = index;
      if (mod_index >= src_requirements.size())
        mod_index -= src_requirements.size();
      if (mod_index >= dst_requirements.size())
        mod_index -= dst_requirements.size();
      if (mod_index >= src_indirect_requirements.size())
        mod_index -= src_indirect_requirements.size();
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(mod_index < atomic_locks.size());
#endif
      std::map<Reservation,bool> &local_locks = atomic_locks[mod_index];
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
    size_t CopyOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void CopyOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int CopyOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* CopyOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    void CopyOp::check_copy_privileges(const bool permit_projection) const 
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        check_copy_privilege(src_requirements[idx], idx, permit_projection);
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
        check_copy_privilege(dst_requirements[idx], 
                        src_requirements.size() + idx, permit_projection);
      if (!src_indirect_requirements.empty())
      {
        const size_t offset = 
          src_requirements.size() + dst_requirements.size();
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
          check_copy_privilege(src_indirect_requirements[idx], 
                               offset + idx, permit_projection);
      }
      if (!dst_indirect_requirements.empty())
      {
        const size_t offset = src_requirements.size() + 
          dst_requirements.size() + src_indirect_requirements.size();
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
          check_copy_privilege(dst_indirect_requirements[idx], 
                               offset + idx, permit_projection);
      }
    }

    //--------------------------------------------------------------------------
    void CopyOp::check_copy_privilege(const RegionRequirement &requirement, 
                                     unsigned idx, const bool permit_proj) const
    //--------------------------------------------------------------------------
    {
      if (!permit_proj && 
          ((requirement.handle_type == LEGION_PARTITION_PROJECTION) ||
           (requirement.handle_type == LEGION_REGION_PROJECTION)))
        REPORT_LEGION_ERROR(ERROR_PROJECTION_REGION_REQUIREMENTS,
                         "Projection region requirements are not "
                               "permitted for copy operations (in task %s)",
                               parent_ctx->get_task_name())
      FieldID bad_field = LEGION_AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, then check the privileges with the parent context
      if (et == LEGION_NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, bad_index);
      const char *req_kind = (idx < src_requirements.size()) ? "source" : 
        (idx < (src_requirements.size() + 
                dst_requirements.size())) ? "destination" : 
        (idx < (src_requirements.size() + dst_requirements.size() + 
                src_indirect_requirements.size())) ? "gather" : "scatter";
      switch (et)
      {
        case LEGION_NO_ERROR:
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
                             idx, req_kind, unique_op_id)
            break;
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = 
              (requirement.handle_type == LEGION_SINGULAR_PROJECTION) ||
              (requirement.handle_type == LEGION_REGION_PROJECTION)
            ? requirement.region.field_space :
            requirement.partition.field_space;
            REPORT_LEGION_ERROR(ERROR_FIELD_NOT_VALID,
                             "Field %d is not a valid field of field "
                             "space %d for index %d of %s requirements "
                             "of copy operation (ID %lld)",
                             bad_field, sp.id, idx, req_kind, unique_op_id)
            break;
          }
        case ERROR_INVALID_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_PRIVILEGE,
                             "Instance field %d is not one of the "
                             "privilege fields for index %d of %s "
                             "requirements of copy operation (ID %lld)",
                             bad_field, idx, req_kind, unique_op_id)
            break;
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_DUPLICATE,
                             "Instance field %d is a duplicate for "
                             "index %d of %s requirements of copy "
                             "operation (ID %lld)",
                             bad_field, idx, req_kind, unique_op_id)
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
                               requirement.region.tree_id, idx, req_kind)
            else if (bad_field == LEGION_AUTO_GENERATE_ID) 
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
                               idx, req_kind, bad_index)
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
                               idx, req_kind, bad_index, bad_field)
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
                             idx, req_kind, unique_op_id)
            break;
          }
        case ERROR_BAD_REGION_TYPE:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_REQUIREMENT_COPY,
                             "Region requirement of copy operation "
                             "(ID %lld) cannot find privileges for field "
                             "%d in parent task from index %d of %s "
                             "region requirements",
                             unique_op_id, bad_field, idx, req_kind)
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
                             idx, req_kind, unique_op_id)
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
      if (!src_indirect_requirements.empty())
      {
        gather_parent_indexes.resize(src_indirect_requirements.size());
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          int parent_index = 
            parent_ctx->find_parent_region_req(src_indirect_requirements[idx]);
          if (parent_index < 0)
            REPORT_LEGION_ERROR(ERROR_PARENT_TASK_COPY,
                           "Parent task %s (ID %lld) of copy operation "
                           "(ID %lld) does not have a region "
                           "requirement for region (%x,%x,%x) "
                           "as a parent of index %d of gather region "
                           "requirements",
                           parent_ctx->get_task_name(), 
                           parent_ctx->get_unique_id(),
                           unique_op_id, 
                           src_indirect_requirements[idx].region.index_space.id,
                           src_indirect_requirements[idx].region.field_space.id, 
                           src_indirect_requirements[idx].region.tree_id, idx)
          else
            gather_parent_indexes[idx] = unsigned(parent_index);
        }
      }
      if (!dst_indirect_requirements.empty())
      {
        scatter_parent_indexes.resize(dst_indirect_requirements.size());
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          int parent_index = 
            parent_ctx->find_parent_region_req(dst_indirect_requirements[idx]);
          if (parent_index < 0)
            REPORT_LEGION_ERROR(ERROR_PARENT_TASK_COPY,
                           "Parent task %s (ID %lld) of copy operation "
                           "(ID %lld) does not have a region "
                           "requirement for region (%x,%x,%x) "
                           "as a parent of index %d of scatter region "
                           "requirements",
                           parent_ctx->get_task_name(), 
                           parent_ctx->get_unique_id(),
                           unique_op_id, 
                           dst_indirect_requirements[idx].region.index_space.id,
                           dst_indirect_requirements[idx].region.field_space.id, 
                           dst_indirect_requirements[idx].region.tree_id, idx)
          else
            scatter_parent_indexes[idx] = unsigned(parent_index);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->legion_spy_enabled && !need_prepipeline_stage)
        log_copy_requirements();
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      tpl->register_operation(this);
      complete_mapping();
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    ApEvent CopyOp::compute_sync_precondition(const TraceInfo *info) const
    //--------------------------------------------------------------------------
    {
      ApEvent result;
      if (!wait_barriers.empty() || !grants.empty())
      {
        std::set<ApEvent> sync_preconditions;
        if (!wait_barriers.empty())
        {
          for (std::vector<PhaseBarrier>::const_iterator it = 
                wait_barriers.begin(); it != wait_barriers.end(); it++)
          {
            ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
            sync_preconditions.insert(e);
            if (runtime->legion_spy_enabled)
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
        // For some reason we don't trace these, not sure why
        result = Runtime::merge_events(NULL, sync_preconditions);
      }
      if ((info != NULL) && info->recording)
        info->record_op_sync_event(result);
      return result;
    }

    //--------------------------------------------------------------------------
    void CopyOp::complete_replay(ApEvent copy_complete_event)
    //--------------------------------------------------------------------------
    {
      // Chain all the unlock and barrier arrivals off of the
      // copy complete event
      if (!arrive_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::iterator it = 
              arrive_barriers.begin(); it != arrive_barriers.end(); it++)
        {
          if (runtime->legion_spy_enabled)
            LegionSpy::log_phase_barrier_arrival(unique_op_id, 
                                                 it->phase_barrier);
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        completion_event);    
        }
      }

      // Handle the case for marking when the copy completes
      Runtime::trigger_event(NULL, completion_event, copy_complete_event);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(completion_event));
    }

    //--------------------------------------------------------------------------
    const VersionInfo& CopyOp::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      if (idx >= src_versions.size())
      {
        idx -= src_versions.size();
        if (idx >= dst_versions.size())
        {
          idx -= dst_versions.size();
          if (idx >= gather_versions.size())
            return scatter_versions[idx - gather_versions.size()];
          else
            return gather_versions[idx];
        }
        else
          return dst_versions[idx];
      }
      else
        return src_versions[idx];
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& CopyOp::get_requirement(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      if (idx >= src_requirements.size())
      {
        idx -= src_requirements.size();
        if (idx >= dst_requirements.size())
        {
          idx -= dst_requirements.size();
          if (idx >= src_indirect_requirements.size())
            return dst_indirect_requirements[
              idx - src_indirect_requirements.size()];
          else
            return src_indirect_requirements[idx];
        }
        else
          return dst_requirements[idx];
      }
      else
        return src_requirements[idx];
    }

    //--------------------------------------------------------------------------
    template<CopyOp::ReqType REQ_TYPE>
    /*static*/ const char* CopyOp::get_req_type_name(void)
    //--------------------------------------------------------------------------
    {
      const char *req_type_names[4] = {
        "source", "destination", "source indirect", "destination indirect",
      };
      return req_type_names[REQ_TYPE];
    }

    //--------------------------------------------------------------------------
    template<CopyOp::ReqType REQ_TYPE>
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
                              !runtime->unsafe_mapper);
      if (bad_tree > 0)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'map_copy' "
                      "on mapper %s. Mapper selected an instance from "
                      "region tree %d to satisfy %s region requirement %d "
                      "for explicit region-to_region copy in task %s (ID %lld) "
                      "but the logical region for this requirement is from "
                      "region tree %d.", mapper->get_mapper_name(), 
                      bad_tree, get_req_type_name<REQ_TYPE>(), idx,
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id(),
                      req.region.get_tree_id())
      if (!missing_fields.empty())
      {
        for (std::vector<FieldID>::const_iterator it = missing_fields.begin();
              it != missing_fields.end(); it++)
        {
          const void *name; size_t name_size;
          if (!runtime->retrieve_semantic_information(
               req.region.get_field_space(), *it, LEGION_NAME_SEMANTIC_TAG,
               name, name_size, true, false))
            name = "(no name)";
          log_run.error("Missing instance for field %s (FieldID: %d)",
                        static_cast<const char*>(name), *it);
        }
        REPORT_LEGION_ERROR(ERROR_MISSING_INSTANCE_FIELD,
                      "Invalid mapper output from invocation of 'map_copy' "
                      "on mapper %s. Mapper failed to specify a physical "
                      "instance for %zd fields of the %s region requirement %d "
                      "of explicit region-to-region copy in task %s (ID %lld). "
                      "The missing fields are listed below.",
                      mapper->get_mapper_name(), missing_fields.size(), 
                      get_req_type_name<REQ_TYPE>(), idx,
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
                          get_req_type_name<REQ_TYPE>(), idx,
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
                        get_req_type_name<REQ_TYPE>(), idx,
                        parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id());
      }
      if (composite_idx >= 0)
      {
        // Destination is not allowed to have composite instances
        if (REQ_TYPE != SRC_REQ)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'map_copy' "
                        "on mapper %s. Mapper requested the creation of a "
                        "virtual instance for %s region requiremnt "
                        "%d. Only source region requirements are permitted to "
                        "be virtual instances for explicit region-to-region "
                        "copy operations. Operation was issued in task %s "
                        "(ID %lld).", mapper->get_mapper_name(), 
                        get_req_type_name<REQ_TYPE>(), idx,
                        parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        if (is_reduce)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'map_copy' "
                        "on mapper %s. Mapper requested the creation of a "
                        "virtual instance for the %s requirement %d of "
                        "an explicit region-to-region reduction. Only real "
                        "physical instances are permitted to be sources of "
                        "explicit region-to-region reductions. Operation was "
                        "issued in task %s (ID %lld).", mapper->get_mapper_name(),
                        get_req_type_name<REQ_TYPE>(), idx, 
                        parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        if (idx < src_indirect_requirements.size())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'map_copy' "
                        "on mapper %s. Mapper requested the creation of a "
                        "virtual instance for %s region requiremnt "
                        "%d. Only source region requirements without source "
                        "indirection requirements are permitted to "
                        "be virtual instances for explicit region-to-region "
                        "copy operations. Operation was issued in task %s "
                        "(ID %lld).", mapper->get_mapper_name(), 
                        get_req_type_name<REQ_TYPE>(), idx,
                        parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        if (idx < dst_indirect_requirements.size())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'map_copy' "
                        "on mapper %s. Mapper requested the creation of a "
                        "virtual instance for %s region requiremnt %d. "
                        "Only source region requirements without destination "
                        "indirection requirements are permitted to "
                        "be virtual instances for explicit region-to-region "
                        "copy operations. Operation was issued in task %s "
                        "(ID %lld).", mapper->get_mapper_name(), 
                        get_req_type_name<REQ_TYPE>(), idx,
                        parent_ctx->get_task_name(), parent_ctx->get_unique_id())
      }
      if (runtime->unsafe_mapper)
        return composite_idx;
      std::vector<LogicalRegion> regions_to_check(1, req.region);
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        const InstanceRef &ref = targets[idx];
        InstanceManager *man = ref.get_manager();
        if (man->is_virtual_manager())
          continue;
        PhysicalManager *manager = man->as_physical_manager();
        if (!manager->meets_regions(regions_to_check))
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'map_copy' "
                        "on mapper %s. Mapper specified an instance for %s "
                        "region requirement at index %d that does not meet "
                        "the logical region requirement. The copy operation "
                        "was issued in task %s (ID %lld).",
                        mapper->get_mapper_name(), 
                        get_req_type_name<REQ_TYPE>(), idx, 
                        parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
        if (manager->is_collective_manager())
        {
          CollectiveManager *collective_manager = 
            manager->as_collective_manager();
          if (!collective_manager->point_space->contains_point(index_point))
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'map_copy' "
                        "on mapper %s. Mapper selected a collective instance "
                        "for %s region requirement at index %d but point copy "
                        "(ID %lld) launched in task %s (ID %lld) is not "
                        "contained within the point space for the collective "
                        "instace.", mapper->get_mapper_name(), 
                        get_req_type_name<REQ_TYPE>(), idx,
                        get_unique_op_id(), parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
        }
      }
      // Make sure all the destinations are real instances, this has
      // to be true for all kinds of explicit copies including reductions
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        if ((REQ_TYPE == SRC_REQ) && (int(idx) == composite_idx))
          continue;
        if (!targets[idx].get_manager()->is_physical_manager())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'map_copy' "
                        "on mapper %s. Mapper specified an illegal "
                        "specialized instance as the target for %s "
                        "region requirement %d of an explicit copy operation "
                        "in task %s (ID %lld).", mapper->get_mapper_name(),
                        get_req_type_name<REQ_TYPE>(), idx,
                        parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id())
      }
      return composite_idx;
    }

    //--------------------------------------------------------------------------
    void CopyOp::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &requests, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest &request = requests.add_request( 
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
          &response, sizeof(response), profiling_priority);
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      handle_profiling_update(count);
    }

    //--------------------------------------------------------------------------
    void CopyOp::handle_profiling_response(const ProfilingResponseBase *base,
                                       const Realm::ProfilingResponse &response,
                                       const void *orig, size_t orig_length)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapper != NULL);
#endif
      const OpProfilingResponse *op_info = 
        static_cast<const OpProfilingResponse*>(base);
      // Check to see if we are done mapping, if not then we need to defer
      // this until we are done mapping so we know how many
      if (!mapped_event.has_triggered())
      {
        // Take the lock and see if we lost the race
        AutoLock o_lock(op_lock);
        if (!mapped_event.has_triggered())
        {
          // Save this profiling response for later until we know the
          // full count of profiling responses
          profiling_info.resize(profiling_info.size() + 1);
          CopyProfilingInfo &info = profiling_info.back();
          info.src_index = op_info->src;
          info.dst_index = op_info->dst;
          info.fill_response = op_info->fill;
          info.buffer_size = orig_length;
          info.buffer = malloc(orig_length);
          memcpy(info.buffer, orig, orig_length);
          return;
        }
      }
      // If we get here then we can handle the response now
      Mapping::Mapper::CopyProfilingInfo info; 
      info.profiling_responses.attach_realm_profiling_response(response);
      info.src_index = op_info->src;
      info.dst_index = op_info->dst;
      info.total_reports = outstanding_profiling_requests.load();
      info.fill_response = op_info->fill;
      mapper->invoke_copy_report_profiling(this, &info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(count <= outstanding_profiling_requests.load());
#endif
      if (count == outstanding_profiling_requests.load())
        Runtime::trigger_event(profiling_reported);
    }

    //--------------------------------------------------------------------------
    void CopyOp::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count > 0);
      assert(!mapped_event.has_triggered());
#endif
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void CopyOp::pack_remote_operation(Serializer &rez, AddressSpaceID target,  
                                       std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_copy(rez, target);
      rez.serialize<size_t>(profiling_requests.size());
      if (!profiling_requests.empty())
      {
        for (unsigned idx = 0; idx < profiling_requests.size(); idx++)
          rez.serialize(profiling_requests[idx]);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Create a user event for this response
        const RtUserEvent response = Runtime::create_rt_user_event();
        rez.serialize(response);
        applied_events.insert(response);
      }
    }

    /////////////////////////////////////////////////////////////
    // Index Copy Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexCopyOp::IndexCopyOp(Runtime *rt)
      : CollectiveInstanceCreator<CopyOp>(rt)
    //--------------------------------------------------------------------------
    {
      this->is_index_space = true;
    }

    //--------------------------------------------------------------------------
    IndexCopyOp::IndexCopyOp(const IndexCopyOp &rhs)
      : CollectiveInstanceCreator<CopyOp>(rhs)
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
    void IndexCopyOp::initialize(InnerContext *ctx, 
                                 const IndexCopyLauncher &launcher,
                                 IndexSpace launch_sp)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_speculation(ctx, true/*track*/, 
                             launcher.src_requirements.size() + 
                               launcher.dst_requirements.size(), 
                             launcher.static_dependences,
                             launcher.predicate, launcher.provenance.c_str());
#ifdef DEBUG_LEGION
      assert(launch_sp.exists());
#endif
      launch_space = runtime->forest->get_node(launch_sp);
      add_launch_space_reference(launch_space);
      if (!launcher.launch_domain.exists())
        launch_space->get_launch_space_domain(index_domain);
      else
        index_domain = launcher.launch_domain;
      src_requirements.resize(launcher.src_requirements.size());
      dst_requirements.resize(launcher.dst_requirements.size());
      src_versions.resize(launcher.src_requirements.size());
      dst_versions.resize(launcher.dst_requirements.size());
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
        src_requirements[idx].flags |= LEGION_NO_ACCESS_FLAG;
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
        if (!HAS_WRITE(launcher.dst_requirements[idx]))
          REPORT_LEGION_ERROR(ERROR_DESTINATION_REGION_REQUIREMENT,
                          "Destination region requirement %d of "
                          "copy (ID %lld) in task %s (ID %lld) does "
                          "not have a write or a reduce privilege.",
                          idx, get_unique_op_id(),
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id())
        dst_requirements[idx] = launcher.dst_requirements[idx];
        dst_requirements[idx].flags |= LEGION_NO_ACCESS_FLAG;
      }
      if (!launcher.src_indirect_requirements.empty())
      {
        const size_t gather_size = launcher.src_indirect_requirements.size();
        src_indirect_requirements.resize(gather_size);
        gather_versions.resize(gather_size);
        for (unsigned idx = 0; idx < gather_size; idx++)
        {
          RegionRequirement &req = src_indirect_requirements[idx];
          req = launcher.src_indirect_requirements[idx];
          req.flags |= LEGION_NO_ACCESS_FLAG;
          if (req.privilege_fields.size() != 1)
            REPORT_LEGION_ERROR(ERROR_COPY_GATHER_REQUIREMENT, 
                "Source indirect region requirement %d for copy op in "
                "parent task %s (ID %lld) has %zd fields, but exactly one "
                "field is required.", idx, parent_ctx->get_task_name(),
                parent_ctx->get_unique_id(), req.privilege_fields.size())
        }
        if (launcher.src_indirect_is_range.size() != gather_size)
          REPORT_LEGION_ERROR(ERROR_COPY_GATHER_REQUIREMENT,
              "Invalid 'src_indirect_is_range' size in launcher. The "
              "number of entries (%zd) does not match the number of "
              "'src_indirect_requirments' (%zd) for copy operation in "
              "parent task %s (ID %lld)", 
              launcher.src_indirect_is_range.size(), gather_size, 
              parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        gather_is_range = launcher.src_indirect_is_range;
        for (unsigned idx = 0; idx < gather_size; idx++)
        {
          if (!gather_is_range[idx])
            continue;
          // For anything that is a gather by range we either need 
          // it also to be a scatter by range or we need a reduction
          // on the destination region requirement so we know how
          // to handle reducing down all the values
          if ((idx < launcher.dst_indirect_is_range.size()) &&
              launcher.dst_indirect_is_range[idx])
            continue;
          if (dst_requirements[idx].privilege != LEGION_REDUCE)
            REPORT_LEGION_ERROR(ERROR_DESTINATION_REGION_REQUIREMENT,
                "Invalid privileges for destination region requirement %d "
                " for copy across in parent task %s (ID %lld). Destination "
                "region requirements must use reduction privileges when "
                "there is a range-based source indirection field and there "
                "is no corresponding range indirection on the destination.",
                idx, parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        }
        src_indirect_records.resize(gather_size);
        collective_exchanges.resize(gather_size);
        possible_src_indirect_out_of_range =
          launcher.possible_src_indirect_out_of_range;
      }
      if (!launcher.dst_indirect_requirements.empty())
      {
        const size_t scatter_size = launcher.dst_indirect_requirements.size();
        dst_indirect_requirements.resize(scatter_size);
        scatter_versions.resize(scatter_size);
        for (unsigned idx = 0; idx < scatter_size; idx++)
        {
          RegionRequirement &req = dst_indirect_requirements[idx];
          req = launcher.dst_indirect_requirements[idx];
          req.flags |= LEGION_NO_ACCESS_FLAG;
          if (req.privilege_fields.size() != 1)
            REPORT_LEGION_ERROR(ERROR_COPY_GATHER_REQUIREMENT, 
                "Destination indirect region requirement %d for copy op in "
                "parent task %s (ID %lld) has %zd fields, but exactly one "
                "field is required.", idx, parent_ctx->get_task_name(),
                parent_ctx->get_unique_id(), req.privilege_fields.size())
        }
        if (launcher.dst_indirect_is_range.size() != scatter_size)
          REPORT_LEGION_ERROR(ERROR_COPY_GATHER_REQUIREMENT,
              "Invalid 'dst_indirect_is_range' size in launcher. The "
              "number of entries (%zd) does not match the number of "
              "'dst_indirect_requirments' (%zd) for copy operation in "
              "parent task %s (ID %lld)", 
              launcher.dst_indirect_is_range.size(), scatter_size, 
              parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        scatter_is_range = launcher.dst_indirect_is_range;
        dst_indirect_records.resize(scatter_size);
        collective_exchanges.resize(scatter_size);
        possible_dst_indirect_out_of_range = 
          launcher.possible_dst_indirect_out_of_range;
        possible_dst_indirect_aliasing = 
          launcher.possible_dst_indirect_aliasing;
      }
      collective_src_indirect_points = launcher.collective_src_indirect_points;
      collective_dst_indirect_points = launcher.collective_dst_indirect_points;
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
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      if (runtime->legion_spy_enabled)
      {
        const unsigned copy_kind = (src_indirect_requirements.empty() ? 0 : 1) +
          (dst_indirect_requirements.empty() ? 0 : 2);
        LegionSpy::log_copy_operation(parent_ctx->get_unique_id(),
                                      unique_op_id, copy_kind, context_index,
                                      collective_src_indirect_points,
                                      collective_dst_indirect_points);
        runtime->forest->log_launch_space(launch_space->handle, unique_op_id);
      }
      if (runtime->check_privileges)
        perform_type_checking();
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_copy();
      index_domain = Domain::NO_DOMAIN;
      launch_space = NULL;
      points_committed = 0;
      commit_request = false;
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_copy();
      // We can deactivate all of our point operations
      for (std::vector<PointCopyOp*>::const_iterator it = points.begin();
            it != points.end(); it++)
        (*it)->deactivate();
      points.clear();
      collective_exchanges.clear();
      commit_preconditions.clear();
      interfering_requirements.clear();
      intra_space_dependences.clear();
      pending_intra_space_dependences.clear();
      if (remove_launch_space_reference(launch_space))
        delete launch_space;
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
        RegionRequirement &req = src_requirements[idx];
        // Promote any singular region requirements to projection
        if (req.handle_type == LEGION_SINGULAR_PROJECTION)
        {
          req.handle_type = LEGION_REGION_PROJECTION;
          req.projection = 0;
        }
        initialize_privilege_path(src_privilege_paths[idx], req);
      }
      dst_privilege_paths.resize(dst_requirements.size());
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        RegionRequirement &req = dst_requirements[idx];
        // Promote any singular region requirements to projection
        if (req.handle_type == LEGION_SINGULAR_PROJECTION)
        {
          req.handle_type = LEGION_REGION_PROJECTION;
          req.projection = 0;
        }
        initialize_privilege_path(dst_privilege_paths[idx], req);
      }
      if (!src_indirect_requirements.empty())
      {
        gather_privilege_paths.resize(src_indirect_requirements.size());
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          RegionRequirement &req = src_indirect_requirements[idx];
          // Promote any singular region requirements to projection
          if (req.handle_type == LEGION_SINGULAR_PROJECTION)
          {
            req.handle_type = LEGION_REGION_PROJECTION;
            req.projection = 0;
          }
          initialize_privilege_path(gather_privilege_paths[idx], req);
        }
      }
      if (!dst_indirect_requirements.empty())
      {
        scatter_privilege_paths.resize(dst_indirect_requirements.size());
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          RegionRequirement &req = dst_indirect_requirements[idx];
          // Promote any singular region requirements to projection
          if (req.handle_type == LEGION_SINGULAR_PROJECTION)
          {
            req.handle_type = LEGION_REGION_PROJECTION;
            req.projection = 0;
          }
          initialize_privilege_path(scatter_privilege_paths[idx], req);
        }
      } 
      if (runtime->legion_spy_enabled)
        log_index_copy_requirements();
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::log_index_copy_requirements(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        const RegionRequirement &req = src_requirements[idx];
        const bool reg = (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
                         (req.handle_type == LEGION_REGION_PROJECTION);
        const bool proj = (req.handle_type == LEGION_REGION_PROJECTION) ||
                          (req.handle_type == LEGION_PARTITION_PROJECTION); 

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
        const bool reg = (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
                         (req.handle_type == LEGION_REGION_PROJECTION);
        const bool proj = (req.handle_type == LEGION_REGION_PROJECTION) ||
                          (req.handle_type == LEGION_PARTITION_PROJECTION);

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
      if (!src_indirect_requirements.empty())
      {
        const size_t offset = 
          src_requirements.size() + dst_requirements.size();
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          const RegionRequirement &req = src_indirect_requirements[idx];
#ifdef DEBUG_LEGION
          assert(req.privilege_fields.size() == 1);
#endif
          const bool reg = (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
                           (req.handle_type == LEGION_REGION_PROJECTION);
          const bool proj = (req.handle_type == LEGION_REGION_PROJECTION) ||
                            (req.handle_type == LEGION_PARTITION_PROJECTION);

          LegionSpy::log_logical_requirement(unique_op_id, offset + idx, reg,
              reg ? req.region.index_space.id :
                    req.partition.index_partition.id,
              reg ? req.region.field_space.id :
                    req.partition.field_space.id,
              reg ? req.region.tree_id : 
                    req.partition.tree_id,
              req.privilege, req.prop, req.redop, req.parent.index_space.id);
          LegionSpy::log_requirement_fields(unique_op_id, offset + idx, 
                                            req.privilege_fields);
          if (proj)
            LegionSpy::log_requirement_projection(unique_op_id, offset + idx,
                                                  req.projection);
        }
      }
      if (!dst_indirect_requirements.empty())
      {
        const size_t offset = src_requirements.size() + 
          dst_requirements.size() + src_indirect_requirements.size();
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          const RegionRequirement &req = dst_indirect_requirements[idx];
#ifdef DEBUG_LEGION
          assert(req.privilege_fields.size() == 1);
#endif
          const bool reg = (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
                           (req.handle_type == LEGION_REGION_PROJECTION);
          const bool proj = (req.handle_type == LEGION_REGION_PROJECTION) ||
                            (req.handle_type == LEGION_PARTITION_PROJECTION);

          LegionSpy::log_logical_requirement(unique_op_id, offset + idx, reg,
              reg ? req.region.index_space.id :
                    req.partition.index_partition.id,
              reg ? req.region.field_space.id :
                    req.partition.field_space.id,
              reg ? req.region.tree_id : 
                    req.partition.tree_id,
              req.privilege, req.prop, req.redop, req.parent.index_space.id);
          LegionSpy::log_requirement_fields(unique_op_id, offset + idx, 
                                            req.privilege_fields);
          if (proj)
            LegionSpy::log_requirement_projection(unique_op_id, offset + idx,
                                                  req.projection);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->check_privileges)
        check_copy_privileges(true/*permit projection*/);
      // Register a dependence on our predicate
      register_predicate_dependence();
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(this, 
                              wait_barriers, arrive_barriers);
      src_versions.resize(src_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        ProjectionInfo src_info(runtime, src_requirements[idx], launch_space); 
        runtime->forest->perform_dependence_analysis(this, idx, 
                                                     src_requirements[idx],
                                                     src_info,
                                                     src_privilege_paths[idx],
                                                     map_applied_conditions);
      }
      dst_versions.resize(dst_requirements.size());
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        ProjectionInfo dst_info(runtime, dst_requirements[idx], launch_space); 
        unsigned index = src_requirements.size()+idx;
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_READ_WRITE;
        runtime->forest->perform_dependence_analysis(this, index, 
                                                     dst_requirements[idx],
                                                     dst_info,
                                                     dst_privilege_paths[idx],
                                                     map_applied_conditions);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_REDUCE;
      }
      if (!src_indirect_requirements.empty())
      {
        gather_versions.resize(src_indirect_requirements.size());
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          ProjectionInfo gather_info(runtime, src_indirect_requirements[idx], 
                                     launch_space);
          runtime->forest->perform_dependence_analysis(this, idx, 
                                                 src_indirect_requirements[idx],
                                                 gather_info,
                                                 gather_privilege_paths[idx],
                                                 map_applied_conditions);
        }
      }
      if (!dst_indirect_requirements.empty())
      {
        scatter_versions.resize(dst_indirect_requirements.size());
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          ProjectionInfo scatter_info(runtime, dst_indirect_requirements[idx],
                                      launch_space);
          runtime->forest->perform_dependence_analysis(this, idx, 
                                                 dst_indirect_requirements[idx],
                                                 scatter_info,
                                                 scatter_privilege_paths[idx],
                                                 map_applied_conditions);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Enumerate the points
      enumerate_points(false/*replaying*/); 
      // Check for interfering point requirements in debug mode
      if (runtime->check_privileges)
        check_point_requirements();
      // Launch the points
      std::set<RtEvent> mapped_preconditions;
      std::set<ApEvent> executed_preconditions;
      for (std::vector<PointCopyOp*>::const_iterator it = points.begin();
            it != points.end(); it++)
      {
        mapped_preconditions.insert((*it)->get_mapped_event());
        executed_preconditions.insert((*it)->get_completion_event());
        (*it)->launch();
      }
#ifdef LEGION_SPY
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      completion_event);
#endif
      // Record that we are mapped when all our points are mapped
      // and we are executed when all our points are executed
      complete_mapping(Runtime::merge_events(mapped_preconditions));
      ApEvent done = Runtime::merge_events(NULL, executed_preconditions);
      if (!request_early_complete(done))
        complete_execution(Runtime::protect_event(done));
      else
        complete_execution();
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
    void IndexCopyOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_replaying());
#endif
      if (runtime->legion_spy_enabled && !need_prepipeline_stage)
        log_index_copy_requirements();
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      // Enumerate the points
      enumerate_points(true/*replaying*/);
      // Then call replay analysis on all of them
      for (std::vector<PointCopyOp*>::const_iterator it = 
            points.begin(); it != points.end(); it++)
        (*it)->trigger_replay();
      complete_mapping();
      complete_execution();
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::enumerate_points(bool replaying)
    //--------------------------------------------------------------------------
    {
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
        if (src_requirements[idx].handle_type == LEGION_SINGULAR_PROJECTION)
          continue;
        ProjectionFunction *function = 
          runtime->find_projection_function(src_requirements[idx].projection);
        function->project_points(this, idx, src_requirements[idx],
                                 runtime, projection_points, launch_space);
      }
      unsigned offset = src_requirements.size();
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        if (dst_requirements[idx].handle_type == LEGION_SINGULAR_PROJECTION)
          continue;
        ProjectionFunction *function = 
          runtime->find_projection_function(dst_requirements[idx].projection);
        function->project_points(this, offset + idx, 
                                 dst_requirements[idx], runtime, 
                                 projection_points, launch_space);
      }
      offset += dst_requirements.size();
      if (!src_indirect_requirements.empty())
      {
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          if (src_indirect_requirements[idx].handle_type == 
              LEGION_SINGULAR_PROJECTION)
            continue;
          ProjectionFunction *function = 
            runtime->find_projection_function(
                src_indirect_requirements[idx].projection);
          function->project_points(this, offset + idx,
                                   src_indirect_requirements[idx], runtime,
                                   projection_points, launch_space);
        }
        offset += src_indirect_requirements.size();
      }
      if (!dst_indirect_requirements.empty())
      {
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          if (dst_indirect_requirements[idx].handle_type == 
              LEGION_SINGULAR_PROJECTION)
            continue;
          ProjectionFunction *function = 
            runtime->find_projection_function(
                dst_indirect_requirements[idx].projection);
          function->project_points(this, offset + idx,
                                   dst_indirect_requirements[idx], runtime,
                                   projection_points, launch_space);
        }
      }
      if (runtime->legion_spy_enabled && !replaying)
      {
        for (std::vector<PointCopyOp*>::const_iterator it = points.begin();
              it != points.end(); it++) 
          (*it)->log_copy_requirements();
      }
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
      interfering_requirements.insert(std::pair<unsigned,unsigned>(idx1,idx2));
    }

    //--------------------------------------------------------------------------
    RtEvent IndexCopyOp::exchange_indirect_records(
        const unsigned index, const ApEvent local_pre, const ApEvent local_post,
        ApEvent &collective_pre, ApEvent &collective_post,
        const TraceInfo &trace_info, const InstanceSet &insts,
        const RegionRequirement &req, const DomainPoint &key,
        std::vector<IndirectRecord> &records, const bool sources)
    //--------------------------------------------------------------------------
    {
      if (sources && !collective_src_indirect_points)
        return CopyOp::exchange_indirect_records(index, local_pre, local_post,
                              collective_pre, collective_post, trace_info, 
                              insts, req, key, records, sources);
      if (!sources && !collective_dst_indirect_points)
        return CopyOp::exchange_indirect_records(index, local_pre, local_post,
                              collective_pre, collective_post, trace_info,
                              insts, req, key, records, sources);
#ifdef DEBUG_LEGION
      assert(local_pre.exists());
      assert(local_post.exists());
#endif
      // Take the lock and record our sets and instances
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(index < collective_exchanges.size());
#endif
      IndirectionExchange &exchange = collective_exchanges[index];
      if (sources)
      {
        if (!exchange.collective_pre.exists())
        {
          exchange.collective_pre = 
            Runtime::create_ap_user_event(&trace_info);
          exchange.collective_post =
            Runtime::create_ap_user_event(&trace_info);
        }
        collective_pre = exchange.collective_pre;
        collective_post = exchange.collective_post;
        if (!exchange.src_ready.exists())
          exchange.src_ready = Runtime::create_rt_user_event();
        if (exchange.local_preconditions.size() < points.size())
        {
          exchange.local_preconditions.insert(local_pre);
          if (exchange.local_preconditions.size() == points.size())
            Runtime::trigger_event(&trace_info, exchange.collective_pre, 
              Runtime::merge_events(&trace_info, exchange.local_preconditions));
        }
        if (exchange.local_postconditions.size() < points.size())
        {
          exchange.local_postconditions.insert(local_post);
          if (exchange.local_postconditions.size() == points.size())
            Runtime::trigger_event(&trace_info, exchange.collective_post, 
             Runtime::merge_events(&trace_info, exchange.local_postconditions));
        }
#ifdef DEBUG_LEGION
        assert(index < src_indirect_records.size());
        assert(src_indirect_records[index].size() < points.size());
#endif
        src_indirect_records[index].emplace_back(
            IndirectRecord(runtime->forest, req, insts, key));
        exchange.src_records.push_back(&records);
        if (src_indirect_records[index].size() == points.size())
        {
          for (unsigned idx = 0; idx < exchange.src_records.size(); idx++)
            *exchange.src_records[idx] = src_indirect_records[index];
          Runtime::trigger_event(exchange.src_ready);
        }
        return exchange.src_ready;
      }
      else
      {
        if (!exchange.collective_pre.exists())
        {
          exchange.collective_pre = 
            Runtime::create_ap_user_event(&trace_info);
          exchange.collective_post =
            Runtime::create_ap_user_event(&trace_info);
        }
        collective_pre = exchange.collective_pre;
        collective_post = exchange.collective_post;
        if (!exchange.dst_ready.exists())
          exchange.dst_ready = Runtime::create_rt_user_event();
        if (exchange.local_preconditions.size() < points.size())
        {
          exchange.local_preconditions.insert(local_pre);
          if (exchange.local_preconditions.size() == points.size())
            Runtime::trigger_event(&trace_info, exchange.collective_pre,
              Runtime::merge_events(&trace_info, exchange.local_preconditions));
        }
        if (exchange.local_postconditions.size() < points.size())
        {
          exchange.local_postconditions.insert(local_post);
          if (exchange.local_postconditions.size() == points.size())
            Runtime::trigger_event(&trace_info, exchange.collective_post,
             Runtime::merge_events(&trace_info, exchange.local_postconditions));
        }
#ifdef DEBUG_LEGION
        assert(index < dst_indirect_records.size());
        assert(dst_indirect_records[index].size() < points.size());
#endif
        dst_indirect_records[index].emplace_back(
            IndirectRecord(runtime->forest, req, insts, key));
        exchange.dst_records.push_back(&records);
        if (dst_indirect_records[index].size() == points.size())
        {
          for (unsigned idx = 0; idx < exchange.dst_records.size(); idx++)
            *exchange.dst_records[idx] = dst_indirect_records[index];
          Runtime::trigger_event(exchange.dst_ready);
        }
        return exchange.dst_ready;
      }
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::check_point_requirements(void)
    //--------------------------------------------------------------------------
    {
      // Handle any region requirements which can interfere with itself
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        const RegionRequirement &req = dst_requirements[idx];
        if (!IS_WRITE(req))
          continue;
        // If the projection functions are invertible then we don't have to 
        // worry about interference because the runtime knows how to hook
        // up those kinds of dependences
        if (req.handle_type != LEGION_SINGULAR_PROJECTION)
        {
          ProjectionFunction *func = 
            runtime->find_projection_function(req.projection);   
          if (func->is_invertible)
            continue;
        }
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
              switch (current_point.get_dim())
              {
                case 1:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_COPY,
                              "Index space copy launch has intefering "
                              "region requirements %d of point %lld and region "
                              "requirement %d of point %lld of %s (UID %lld) "
                              "in parent task %s (UID %lld) are interfering.",
                              it->first, current_point[0], it->second,
                              oit->first[0], get_logging_name(),
                              get_unique_id(), parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#if LEGION_MAX_DIM > 1
                case 2:
                  {
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
                    break;
                  }
#endif
#if LEGION_MAX_DIM > 2
                case 3:
                  {
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
                    break;
                  }
#endif
#if LEGION_MAX_DIM > 3
                case 4:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_COPY,
                              "Index space copy launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld) and region "
                              "requirement %d of point (%lld,%lld,%lld,%lld) "
                              "of %s (UID %lld) in parent task %s (UID %lld) "
                              "are interfering.", it->first, current_point[0], 
                              current_point[1], current_point[2], 
                              current_point[3], it->second,
                              oit->first[0], oit->first[1], oit->first[2],
                              oit->first[3], get_logging_name(),
                              get_unique_id(), parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM > 4
                case 5:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_COPY,
                              "Index space copy launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld) and "
                              "region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld) of "
                              "%s (UID %lld) in parent task %s (UID %lld) are "
                              "interfering.", it->first, current_point[0],
                              current_point[1], current_point[2], 
                              current_point[3], current_point[4],
                              it->second, oit->first[0], oit->first[1], 
                              oit->first[2], oit->first[3], oit->first[4],
                              get_logging_name(), get_unique_id(), 
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM > 5
                case 6:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_COPY,
                              "Index space copy launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld) of %s (UID %lld)"
                              " in parent task %s (UID %lld) are interfering.",
                              it->first, current_point[0], current_point[1],
                              current_point[2], current_point[3], 
                              current_point[4], current_point[5], 
                              it->second, oit->first[0], oit->first[1], 
                              oit->first[2], oit->first[3], oit->first[4],
                              oit->first[5], get_logging_name(),
                              get_unique_id(), parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM > 6
                case 7:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_COPY,
                              "Index space copy launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld)"
                              "and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld) "
                              "of %s (UID %lld) in parent task %s (UID %lld) "
                              "are interfering.", it->first, current_point[0], 
                              current_point[1], current_point[2], 
                              current_point[3], current_point[4], 
                              current_point[5], current_point[6], it->second,
                              oit->first[0], oit->first[1], oit->first[2],
                              oit->first[3], oit->first[4], oit->first[5],
                              oit->first[6], get_logging_name(),
                              get_unique_id(), parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM > 7
                case 8:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_COPY,
                              "Index space copy launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld) and "
                              "region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld) of "
                              "%s (UID %lld) in parent task %s (UID %lld) are "
                              "interfering.", it->first, current_point[0],
                              current_point[1], current_point[2], 
                              current_point[3], current_point[4],
                              current_point[5], current_point[6],
                              current_point[7], it->second, oit->first[0],
                              oit->first[1], oit->first[2], oit->first[3],
                              oit->first[4], oit->first[5], oit->first[6],
                              oit->first[7], get_logging_name(),
                              get_unique_id(), parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM > 8
                case 9:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_COPY,
                              "Index space copy launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld) "
                              "and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld) "
                              "of %s (UID %lld) in parent task %s (UID %lld) "
                              "are interfering.",
                              it->first, current_point[0], current_point[1],
                              current_point[2], current_point[3], 
                              current_point[4], current_point[5], 
                              current_point[6], current_point[7],
                              current_point[8], it->second, oit->first[0],
                              oit->first[1], oit->first[2], oit->first[3],
                              oit->first[4], oit->first[5], oit->first[6],
                              oit->first[7], oit->first[8], get_logging_name(),
                              get_unique_id(), parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
                default:
                  assert(false);
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    RtEvent IndexCopyOp::find_intra_space_dependence(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      // Check to see if we already have it
      std::map<DomainPoint,RtEvent>::const_iterator finder = 
        intra_space_dependences.find(point);
      if (finder != intra_space_dependences.end())
        return finder->second;
      // Otherwise make a temporary one and record it for now
      const RtUserEvent pending_event = Runtime::create_rt_user_event();
      intra_space_dependences[point] = pending_event;
      pending_intra_space_dependences[point] = pending_event;
      return pending_event;
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::record_intra_space_dependence(const DomainPoint &point,
                                                    RtEvent point_mapped)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      std::map<DomainPoint,RtEvent>::iterator finder = 
        intra_space_dependences.find(point);
      if (finder != intra_space_dependences.end())
      {
#ifdef DEBUG_LEGION
        assert(finder->second != point_mapped);
#endif
        std::map<DomainPoint,RtUserEvent>::iterator pending_finder = 
          pending_intra_space_dependences.find(point);
#ifdef DEBUG_LEGION
        assert(pending_finder != pending_intra_space_dependences.end());
#endif
        Runtime::trigger_event(pending_finder->second, point_mapped);
        pending_intra_space_dependences.erase(pending_finder);
        finder->second = point_mapped;
      }
      else
        intra_space_dependences[point] = point_mapped;
    }

    /////////////////////////////////////////////////////////////
    // Point Copy Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointCopyOp::PointCopyOp(Runtime *rt)
      : CopyOp(rt)
    //--------------------------------------------------------------------------
    {
      this->is_index_space = true;
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
          own->src_requirements.size() + own->dst_requirements.size(),
          own->get_provenance());
      index_point = p;
      index_domain = own->index_domain;
      owner = own;
      context_index = own->get_ctx_index();
      execution_fence_event = own->get_execution_fence_event();
      // From Memoizable
      trace_local_id            = owner->get_trace_local_id().first;
      tpl                       = owner->get_template();
      if (tpl != NULL)
        memo_state              = owner->get_memoizable_state();
      // From Copy
      src_requirements          = owner->src_requirements;
      dst_requirements          = owner->dst_requirements;
      src_indirect_requirements = owner->src_indirect_requirements;
      dst_indirect_requirements = owner->dst_indirect_requirements;
      grants                    = owner->grants;
      wait_barriers             = owner->wait_barriers;
      arrive_barriers           = owner->arrive_barriers;
      parent_task               = owner->parent_task;
      map_id                    = owner->map_id;
      tag                       = owner->tag;
      mapper_data_size          = owner->mapper_data_size;
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, owner->mapper_data, mapper_data_size);
      }
      // From CopyOp
      src_parent_indexes        = owner->src_parent_indexes;
      dst_parent_indexes        = owner->dst_parent_indexes;
      gather_parent_indexes     = owner->gather_parent_indexes;
      scatter_parent_indexes    = owner->scatter_parent_indexes;
      gather_is_range           = owner->gather_is_range;
      scatter_is_range          = owner->scatter_is_range;
      predication_guard         = owner->predication_guard;
      possible_src_indirect_out_of_range 
                                = owner->possible_src_indirect_out_of_range;
      possible_dst_indirect_out_of_range
                                = owner->possible_dst_indirect_out_of_range;
      possible_dst_indirect_aliasing
                                = owner->possible_dst_indirect_aliasing;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_index_point(owner->get_unique_op_id(), unique_op_id, p);
    }

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
      intra_space_mapping_dependences.clear();
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
    void PointCopyOp::launch(void)
    //--------------------------------------------------------------------------
    {
      // Perform the version analysis
      std::set<RtEvent> preconditions;
      if (!intra_space_mapping_dependences.empty())
        preconditions.swap(intra_space_mapping_dependences);
      src_versions.resize(src_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        runtime->forest->perform_versioning_analysis(this, idx,
            src_requirements[idx], src_versions[idx], preconditions);
      dst_versions.resize(dst_requirements.size());
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_READ_WRITE;
        runtime->forest->perform_versioning_analysis(this,
            src_requirements.size() + idx, dst_requirements[idx],
            dst_versions[idx], preconditions);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_REDUCE;
      }
      if (!src_indirect_requirements.empty())
      {
        gather_versions.resize(src_indirect_requirements.size());
        const size_t offset = src_requirements.size() + dst_requirements.size();
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
          runtime->forest->perform_versioning_analysis(this, offset + idx, 
           src_indirect_requirements[idx], gather_versions[idx], preconditions);
      }
      if (!dst_indirect_requirements.empty())
      {
        scatter_versions.resize(dst_indirect_requirements.size());
        const size_t offset = src_requirements.size() + 
          dst_requirements.size() + src_indirect_requirements.size();
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
          runtime->forest->perform_versioning_analysis(this, offset + idx, 
           dst_indirect_requirements[idx], scatter_versions[idx],preconditions);
        if (!src_indirect_requirements.empty())
        {
          // Full indirections need to have the same index space
          for (unsigned idx = 0; (idx < src_indirect_requirements.size()) &&
                (idx < dst_indirect_requirements.size()); idx++)
          {
            const IndexSpace src_space = 
              src_indirect_requirements[idx].region.get_index_space();
            const IndexSpace dst_space = 
              dst_indirect_requirements[idx].region.get_index_space();
            if (src_space != dst_space)
              REPORT_LEGION_ERROR(ERROR_COPY_SCATTER_REQUIREMENT,
                  "Mismatch between source indirect and destination indirect "
                  "index spaces for requirement %d for copy operation "
                  "(ID %lld) in parent task %s (ID %lld)",
                  idx, get_unique_id(), parent_ctx->get_task_name(),
                  parent_ctx->get_unique_id())
          }
        }
      }
      // We can also mark this as having our resolved any predication
      resolve_speculation();
      // Then put ourselves in the queue of operations ready to map
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      if (profiling_reported.exists())
        finalize_copy_profiling();
      // Don't commit this operation until we've reported our profiling
      // Out index owner will deactivate the operation
      commit_operation(false/*deactivate*/, profiling_reported);
      // Tell our owner that we are done, they will do the deactivate
      owner->handle_point_commit(profiling_reported);
    }

    //--------------------------------------------------------------------------
    RtEvent PointCopyOp::exchange_indirect_records(
        const unsigned index, const ApEvent local_pre, const ApEvent local_post,
        ApEvent &collective_pre, ApEvent &collective_post,
        const TraceInfo &trace_info, const InstanceSet &insts,
        const RegionRequirement &req, const DomainPoint &key,
        std::vector<IndirectRecord> &records, const bool sources)
    //--------------------------------------------------------------------------
    {
      // Exchange via the owner
      return owner->exchange_indirect_records(index, local_pre, local_post,
                                collective_pre, collective_post, trace_info,
                                insts, req, index_point, records, sources);
    }

    //--------------------------------------------------------------------------
    CollectiveManager* PointCopyOp::find_or_create_collective_instance(
                                  MappingCallKind mapper_call, unsigned index,
                                  const LayoutConstraintSet &constraints,
                                  const std::vector<LogicalRegion> &regions,
                                  Memory::Kind kind, size_t *footprint,
                                  LayoutConstraintKind *unsat_kind,
                                  unsigned *unsat_index,
                                  DomainPoint &collective_point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index_point.get_dim() > 0);
#endif
      collective_point = index_point;
      return owner->find_or_create_collective_instance(mapper_call, index,
          constraints, regions, kind, footprint, unsat_kind, unsat_index, 
          collective_point);
    }

    //--------------------------------------------------------------------------
    bool PointCopyOp::finalize_collective_instance(MappingCallKind call_kind,
                                                   unsigned index, bool success)
    //--------------------------------------------------------------------------
    {
      return owner->finalize_collective_instance(call_kind, index, success);
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::report_total_collective_instance_calls(
                              MappingCallKind mapper_call, unsigned total_calls)
    //--------------------------------------------------------------------------
    {
      owner->report_total_collective_instance_calls(mapper_call, total_calls);
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
        assert(src_requirements[idx].handle_type != LEGION_SINGULAR_PROJECTION);
#endif
        src_requirements[idx].region = result;
        src_requirements[idx].handle_type = LEGION_SINGULAR_PROJECTION;
      }
      else if (idx < (src_requirements.size() + dst_requirements.size()))
      {
        idx -= src_requirements.size();
#ifdef DEBUG_LEGION
        assert(dst_requirements[idx].handle_type != LEGION_SINGULAR_PROJECTION);
#endif
        dst_requirements[idx].region = result;
        dst_requirements[idx].handle_type = LEGION_SINGULAR_PROJECTION;
      }
      else if (idx < (src_requirements.size() + 
            dst_requirements.size() + src_indirect_requirements.size()))
      {
        idx -= (src_requirements.size() + dst_requirements.size());
#ifdef DEBUG_LEGION
        assert(src_indirect_requirements[idx].handle_type != 
                LEGION_SINGULAR_PROJECTION);
#endif
        src_indirect_requirements[idx].region = result;
        src_indirect_requirements[idx].handle_type = LEGION_SINGULAR_PROJECTION;
      }
      else
      {
        idx -= (src_requirements.size() + dst_requirements.size() + 
                src_indirect_requirements.size());
#ifdef DEBUG_LEGION
        assert(idx < dst_indirect_requirements.size());
        assert(dst_indirect_requirements[idx].handle_type != 
                LEGION_SINGULAR_PROJECTION);
#endif
        dst_indirect_requirements[idx].region = result;
        dst_indirect_requirements[idx].handle_type = LEGION_SINGULAR_PROJECTION;
      }
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::record_intra_space_dependences(unsigned index,
                                    const std::vector<DomainPoint> &dependences)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(src_requirements.size() <= index); 
#endif
      index -= src_requirements.size();
#ifdef DEBUG_LEGION
      assert(index < dst_requirements.size());
#endif
      for (unsigned idx = 0; idx < dependences.size(); idx++)
      {
        if (dependences[idx] == index_point)
        {
          // If we've got a prior dependence then record it
          if (idx > 0)
          {
            const DomainPoint &prev = dependences[idx-1];
            const RtEvent pre = owner->find_intra_space_dependence(prev);
            intra_space_mapping_dependences.insert(pre);
            if (runtime->legion_spy_enabled)
            {
              // We know we only need a dependence on the previous point but
              // Legion Spy is stupid, so log everything we have a
              // precondition on even if it is transitively implied
              for (unsigned idx2 = 0; idx2 < idx; idx2++)
                LegionSpy::log_intra_space_dependence(unique_op_id,
                                                      dependences[idx2]);
            }
          }
          // If we're not the last dependence, then send our mapping event
          // so that others can record a dependence on us
          if (idx < (dependences.size()-1))
            owner->record_intra_space_dependence(index_point,
                                                 get_mapped_event());
          return;
        }
      }
      // We should never get here
      assert(false);
    }

    //--------------------------------------------------------------------------
    TraceLocalID PointCopyOp::get_trace_local_id(void) const
    //--------------------------------------------------------------------------
    {
      return TraceLocalID(trace_local_id, index_point);
    }

    /////////////////////////////////////////////////////////////
    // Fence Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FenceOp::FenceOp(Runtime *rt)
      : MemoizableOp<Operation>(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FenceOp::FenceOp(const FenceOp &rhs)
      : MemoizableOp<Operation>(NULL)
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
    Future FenceOp::initialize(InnerContext *ctx, FenceKind kind,
                               bool need_future, const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
      initialize_memoizable();
      fence_kind = kind;
      if (need_future)
      {
        result = Future(new FutureImpl(parent_ctx, runtime, true/*register*/,
              runtime->get_available_distributed_id(),
              runtime->address_space, completion_event, this));
        // We can set the future result right now because we know that it
        // will not be complete until we are complete ourselves
        result.impl->set_result(NULL, 0, true/*own*/); 
      }
      if (runtime->legion_spy_enabled)
        LegionSpy::log_fence_operation(parent_ctx->get_unique_id(),
                                       unique_op_id, context_index);
      return result;
    }

    //--------------------------------------------------------------------------
    Future FenceOp::initialize(InnerContext *ctx, FenceKind kind,
                               bool need_future, Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
      initialize_memoizable();
      fence_kind = kind;
      if (need_future)
      {
        result = Future(new FutureImpl(parent_ctx, runtime, true/*register*/,
              runtime->get_available_distributed_id(),
              runtime->address_space, completion_event, this));
        // We can set the future result right now because we know that it
        // will not be complete until we are complete ourselves
        result.impl->set_result(NULL, 0, true/*own*/); 
      }
      if (runtime->legion_spy_enabled)
        LegionSpy::log_fence_operation(parent_ctx->get_unique_id(),
                                       unique_op_id, context_index);
      return result;
    }

    //--------------------------------------------------------------------------
    void FenceOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_fence(); 
    }

    //--------------------------------------------------------------------------
    void FenceOp::activate_fence(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      activate_memoizable();
    }

    //--------------------------------------------------------------------------
    void FenceOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_fence(); 
      runtime->free_fence_op(this);
    }

    //--------------------------------------------------------------------------
    void FenceOp::deactivate_fence(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      map_applied_conditions.clear();
      execution_preconditions.clear();
      result = Future(); // clear out our future reference
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
      // Perform fence analysis for the given fence in the context
      // Depending on the kind of fence we will either do mapping
      // analysis or execution analysis or both
      perform_fence_analysis(true/*register fence also*/);       
    }

    //--------------------------------------------------------------------------
    void FenceOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      switch (fence_kind)
      {
        case MAPPING_FENCE:
          {
            if (!map_applied_conditions.empty())
              complete_mapping(Runtime::merge_events(map_applied_conditions));
            else
              complete_mapping();
            complete_execution();
            break;
          }
        case EXECUTION_FENCE:
          {
            // If we're recording find all the prior event dependences
            if (is_recording())
              tpl->find_execution_fence_preconditions(execution_preconditions);
            const PhysicalTraceInfo trace_info(this, 0/*index*/, true/*init*/);
            // We can always trigger the completion event when these are done
            ApEvent execution_precondition;
            if (!execution_preconditions.empty())
              execution_precondition = 
                Runtime::merge_events(&trace_info, execution_preconditions);
            if (is_recording())
              tpl->record_complete_replay(this, execution_precondition);
            // Mark that we finished our mapping now
            if (!map_applied_conditions.empty())
              complete_mapping(Runtime::merge_events(map_applied_conditions));
            else
              complete_mapping();
            if (!request_early_complete(execution_precondition))
              complete_execution(
                  Runtime::protect_event(execution_precondition));
            else
              complete_execution();
            break;
          }
        default:
          assert(false); // should never get here
      }
    }

    //--------------------------------------------------------------------------
    void FenceOp::perform_fence_analysis(bool update_fence)
    //--------------------------------------------------------------------------
    {
      switch (fence_kind) 
      {
        case MAPPING_FENCE:
          {
            parent_ctx->perform_fence_analysis(this, execution_preconditions,
                            true/*mapping fence*/, false/*execution fence*/);
#ifdef DEBUG_LEGION
            assert(execution_preconditions.empty());
#endif
            if (update_fence)
              parent_ctx->update_current_fence(this, true, false);
            break;
          }
        case EXECUTION_FENCE:
          {
            parent_ctx->perform_fence_analysis(this, execution_preconditions,
                true/*mapping fence*/, !is_replaying()/*execution fence*/);
            if (update_fence)
              parent_ctx->update_current_fence(this, true, !is_replaying());
            break;
          }
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    void FenceOp::update_current_fence(void)
    //--------------------------------------------------------------------------
    {
      switch (fence_kind) 
      {
        case MAPPING_FENCE:
          {
            parent_ctx->update_current_fence(this, true, false);
            break;
          }
        case EXECUTION_FENCE:
          {
            parent_ctx->update_current_fence(this, true, true);
            break;
          }
        default:
          assert(false);
      }
    }

#ifdef LEGION_SPY
    //--------------------------------------------------------------------------
    void FenceOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      // Still need this to record that this operation is done for LegionSpy
      LegionSpy::log_operation_events(unique_op_id, 
          ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
      complete_operation();
    }
#endif

    //--------------------------------------------------------------------------
    void FenceOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      if (fence_kind == EXECUTION_FENCE)
        tpl->register_operation(this);
      else
        complete_execution();
      complete_mapping();
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    void FenceOp::complete_replay(ApEvent fence_complete_event)
    //--------------------------------------------------------------------------
    {
      // Handle the case for marking when the copy completes
      Runtime::trigger_event(NULL, completion_event, fence_complete_event);
      need_completion_trigger = false;
      complete_execution();
    }

    //--------------------------------------------------------------------------
    const VersionInfo& FenceOp::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      assert(false);
      return *new VersionInfo();
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
    void FrameOp::initialize(InnerContext *ctx, const char *provenance)
    //--------------------------------------------------------------------------
    {
      FenceOp::initialize(ctx,EXECUTION_FENCE,false/*need future*/,provenance);
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
      FenceOp::trigger_mapping();
    }

    //--------------------------------------------------------------------------
    void FrameOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      // This frame has finished executing so it is no longer mapped
      parent_ctx->decrement_frame();
      // This frame is also finished so we can tell the context
      parent_ctx->finish_frame(completion_event);
      FenceOp::trigger_complete();
    }

    /////////////////////////////////////////////////////////////
    // Creation Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CreationOp::CreationOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CreationOp::CreationOp(const CreationOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CreationOp::~CreationOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CreationOp& CreationOp::operator=(const CreationOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void CreationOp::initialize_index_space(
        InnerContext *ctx, IndexSpaceNode *n, const Future &f, const char *prov)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index_space_node == NULL);
      assert(futures.empty());
#endif
      initialize_operation(ctx, true/*track*/, 0/*regions*/, prov);
      kind = INDEX_SPACE_CREATION;
      index_space_node = n;
      futures.push_back(f);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_creation_operation(parent_ctx->get_unique_id(),
                                          unique_op_id, context_index);
    }

    //--------------------------------------------------------------------------
    void CreationOp::initialize_field(InnerContext *ctx, FieldSpaceNode *node,
                                      FieldID fid, const Future &field_size,
                                      const char *provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(field_space_node == NULL);
      assert(fields.empty());
      assert(futures.empty());
#endif
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
      kind = FIELD_ALLOCATION;
      field_space_node = node;
      fields.push_back(fid);
      futures.push_back(field_size);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_creation_operation(parent_ctx->get_unique_id(),
                                          unique_op_id, context_index);
    }

    //--------------------------------------------------------------------------
    void CreationOp::initialize_fields(InnerContext *ctx, FieldSpaceNode *node,
                                       const std::vector<FieldID> &fids,
                                       const std::vector<Future> &field_sizes,
                                       const char *provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(field_space_node == NULL);
      assert(fields.empty());
      assert(futures.empty());
      assert(fids.size() == field_sizes.size());
#endif
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
      kind = FIELD_ALLOCATION;
      field_space_node = node;     
      fields = fids;
      futures = field_sizes;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_creation_operation(parent_ctx->get_unique_id(),
                                          unique_op_id, context_index);
    }

    //--------------------------------------------------------------------------
    void CreationOp::initialize_map(InnerContext *ctx, const char *provenance,
                              const std::map<DomainPoint,Future> &future_points)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(futures.empty());
#endif
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
      kind = FUTURE_MAP_CREATION;
      futures.resize(future_points.size());
      unsigned index = 0;
      for (std::map<DomainPoint,Future>::const_iterator it = 
            future_points.begin(); it != future_points.end(); it++, index++)
        futures[index] = it->second;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_creation_operation(parent_ctx->get_unique_id(),
                                          unique_op_id, context_index);
    }

    //--------------------------------------------------------------------------
    void CreationOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      index_space_node = NULL;
      field_space_node = NULL;
    }

    //--------------------------------------------------------------------------
    void CreationOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      futures.clear();
      fields.clear();
      runtime->free_creation_op(this);
    }

    //--------------------------------------------------------------------------
    const char* CreationOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[CREATION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind CreationOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return CREATION_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void CreationOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!futures.empty());
#endif
      for (std::vector<Future>::const_iterator it = 
            futures.begin(); it != futures.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->impl != NULL);
#endif
        // Register this operation as dependent on task that
        // generated the future
        it->impl->register_dependence(this);
      }
      // Record this with the context as an implicit dependence for all
      // later operations which may rely on this index space for mapping
      if ((kind == INDEX_SPACE_CREATION) || (kind == FIELD_ALLOCATION))
        parent_ctx->update_current_implicit(this);
    }

    //--------------------------------------------------------------------------
    void CreationOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      complete_mapping();
      switch (kind)
      {
        case INDEX_SPACE_CREATION:
          {
#ifdef DEBUG_LEGION
            assert(futures.size() == 1);
#endif
            const ApEvent ready = futures[0].impl->subscribe();
            complete_execution(Runtime::protect_event(ready)); 
            break;
          }
        case FIELD_ALLOCATION:
          {
            std::set<ApEvent> ready_events;
            for (unsigned idx = 0; idx < futures.size(); idx++)
              ready_events.insert(futures[idx].impl->subscribe());
            if (!ready_events.empty())
              complete_execution(Runtime::protect_merge_events(ready_events));
            else
              complete_execution();
            break;
          }
        case FUTURE_MAP_CREATION:
          {
            complete_execution();
            break;
          }
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    void CreationOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> complete_preconditions;
      switch (kind)
      {
        case INDEX_SPACE_CREATION:
          {
            // Pull the pointer for the domain out of the future and assign 
            // it to the index space node
            FutureImpl *impl = futures[0].impl;
            const size_t future_size = impl->get_untyped_size(true/*internal*/);
            if (future_size != sizeof(Domain))
              REPORT_LEGION_ERROR(ERROR_CREATION_FUTURE_TYPE_MISMATCH,
                  "Future for index space creation in task %s (UID %lld) does "
                  "not have the same size as sizeof(Domain) (e.g. %zd bytes). "
                  "The type of futures for index space domains must be a "
                  "Domain.", parent_ctx->get_task_name(), 
                  parent_ctx->get_unique_id(), sizeof(Domain))
            const Domain *domain = static_cast<Domain*>(
                impl->get_untyped_result(true,NULL,true/*internal*/));
            if (index_space_node->set_domain(*domain, runtime->address_space))
              delete index_space_node;
            break;      
          }
        case FIELD_ALLOCATION:
          {
            for (unsigned idx = 0; idx < futures.size(); idx++)
            {
              FutureImpl *impl = futures[idx].impl;
               const size_t future_size = 
                 impl->get_untyped_size(true/*internal*/);
              if (future_size != sizeof(size_t))
                REPORT_LEGION_ERROR(ERROR_FUTURE_SIZE_MISMATCH,
                    "Size of future passed into dynamic field allocation for "
                    "field %d is %zd bytes which not the same as sizeof(size_t)"
                    " (%zd bytes). Futures passed into field allocation calls "
                    "must contain data of the type size_t.",
                    fields[idx], future_size, sizeof(size_t))
              const size_t field_size = 
                  *((const size_t*)impl->get_untyped_result(true, NULL, true));
              field_space_node->update_field_size(fields[idx], field_size,
                          complete_preconditions, runtime->address_space);
              if (runtime->legion_spy_enabled)
                LegionSpy::log_field_creation(field_space_node->handle.id,
                    fields[idx], field_size, (provenance == NULL) ? NULL :
                    provenance->provenance.c_str());
            }
            break;
          }
        case FUTURE_MAP_CREATION:
          // Nothing to do here
          break;
        default:
          assert(false);
      }
#ifdef LEGION_SPY
      // Still have to do this call to let Legion Spy know we're done
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      ApEvent::NO_AP_EVENT);
#endif
      if (!complete_preconditions.empty())
        complete_operation(Runtime::merge_events(complete_preconditions));
      else
        complete_operation();
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
    void DeletionOp::set_deletion_preconditions(ApEvent precondition,
                                  const std::map<Operation*,GenerationID> &deps)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!has_preconditions);
#endif
      dependences = deps;
      execution_precondition = precondition;
      has_preconditions = true;
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_index_space_deletion(InnerContext *ctx,
                           IndexSpace handle, std::vector<IndexPartition> &subs,
                           const bool unordered, Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, !unordered/*track*/, 0/*regions*/, provenance);
      kind = INDEX_SPACE_DELETION;
      index_space = handle;
      sub_partitions.swap(subs);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_deletion_operation(parent_ctx->get_unique_id(),
                              unique_op_id, context_index, unordered);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_index_part_deletion(InnerContext *ctx,
                       IndexPartition handle, std::vector<IndexPartition> &subs,
                       const bool unordered, Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, !unordered/*track*/, 0/*regions*/, provenance);
      kind = INDEX_PARTITION_DELETION;
      index_part = handle;
      sub_partitions.swap(subs);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_deletion_operation(parent_ctx->get_unique_id(),
                              unique_op_id, context_index, unordered);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_field_space_deletion(InnerContext *ctx,
                FieldSpace handle, const bool unordered, Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, !unordered/*track*/, 0/*regions*/, provenance);
      kind = FIELD_SPACE_DELETION;
      field_space = handle;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_deletion_operation(parent_ctx->get_unique_id(),
                              unique_op_id, context_index, unordered);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_field_deletion(InnerContext *ctx, 
                                               FieldSpace handle, FieldID fid, 
                                               const bool unordered,
                                               FieldAllocatorImpl *impl,
                                               Provenance *provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
      assert(allocator == NULL);
#endif
      initialize_operation(ctx, !unordered/*track*/, 0/*regions*/, provenance);
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
      const std::vector<FieldID> field_vec(1,fid);
      runtime->forest->free_field_indexes(handle, field_vec,
          Runtime::protect_event(completion_event));
      if (runtime->legion_spy_enabled)
        LegionSpy::log_deletion_operation(parent_ctx->get_unique_id(),
                              unique_op_id, context_index, unordered);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_field_deletions(InnerContext *ctx,
                            FieldSpace handle, const std::set<FieldID> &to_free,
                            const bool unordered, FieldAllocatorImpl *impl,
                            Provenance *provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
      assert(allocator == NULL);
#endif
      initialize_operation(ctx, !unordered/*track*/, 0/*regions*/, provenance);
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
      runtime->forest->free_field_indexes(handle, field_vec,
          Runtime::protect_event(completion_event));
      if (runtime->legion_spy_enabled)
        LegionSpy::log_deletion_operation(parent_ctx->get_unique_id(),
                              unique_op_id, context_index, unordered);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::initialize_logical_region_deletion(InnerContext *ctx,
             LogicalRegion handle, const bool unordered, Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, !unordered/*track*/, 0/*regions*/, provenance);
      kind = LOGICAL_REGION_DELETION;
      logical_region = handle; 
      if (runtime->legion_spy_enabled)
        LegionSpy::log_deletion_operation(parent_ctx->get_unique_id(),
                              unique_op_id, context_index, unordered);
    }

    //--------------------------------------------------------------------------
    void DeletionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      allocator = NULL;
      has_preconditions = false;
    }

    //--------------------------------------------------------------------------
    void DeletionOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      // We can remove the reference to the allocator once we are
      // done with all of our free operations
      if ((allocator != NULL) && allocator->remove_reference())
        delete allocator;
      deactivate_operation();
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
      if (has_preconditions)
      {
        for (std::map<Operation*,GenerationID>::const_iterator dit = 
              dependences.begin(); dit != dependences.end(); dit++)
          register_dependence(dit->first, dit->second);
        return;
      }
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
            parent_ctx->analyze_destroy_fields(field_space, free_fields,
                              deletion_requirements, parent_req_indexes,
                              global_fields, local_fields, 
                              local_field_indexes, deletion_req_indexes);
            break;
          }
        case LOGICAL_REGION_DELETION:
          {
            parent_ctx->analyze_destroy_logical_region(logical_region,
                                  deletion_requirements, parent_req_indexes, 
                                  returnable_privileges);
            break;
          }
        default:
          assert(false);
      }
#ifdef DEBUG_LEGION
      assert(deletion_requirements.size() == parent_req_indexes.size());
#endif
      // Even though we're going to do a full fence analysis after this,
      // we still need to do this call so we register ourselves in the 
      // region tree to serve as mapping dependences on things that might
      // use these data structures in the case of recycling, e.g. in the
      // case that we recycle a field index
      for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
      {
        RegionRequirement &req = deletion_requirements[idx];
        // Perform the normal region requirement analysis
        RegionTreePath privilege_path;
        initialize_privilege_path(privilege_path, req);
        runtime->forest->perform_deletion_analysis(this, idx, req, 
                           privilege_path, map_applied_conditions,
                           (kind == LOGICAL_REGION_DELETION));
      }
      // Now pretend like this is going to be a mapping fence on everyone
      // who came before, although we will never actually record ourselves
      // as a mapping fence since we want operations that come after us to
      // be re-ordered up above us. We need this upward facing fence though
      // to ensure that all tasks are done above us before we do delete
      // any internal data structures associated with these resources
      std::set<ApEvent> execution_preconditions;
      parent_ctx->perform_fence_analysis(this, execution_preconditions, 
                                    true/*mapping*/, true/*execution*/);
      if (!execution_preconditions.empty())
        execution_precondition = 
          Runtime::merge_events(NULL, execution_preconditions);
      if (runtime->legion_spy_enabled)
      {
        for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
        {
          const RegionRequirement &req = deletion_requirements[idx];
          if (req.handle_type != LEGION_PARTITION_PROJECTION)
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
          runtime->forest->perform_versioning_analysis(this, idx,
                                            deletion_requirements[idx],
                                            version_infos[idx],
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
      // Clean out the physical state for these operations once we know that  
      // all prior operations that needed the state have been done
      if (kind == LOGICAL_REGION_DELETION)
      {
        // Just need to clean out the version managers which will free
        // all the equivalence sets and allow the reference counting to
        // clean everything up
        bool has_outermost = false;
        RegionTreeContext outermost_ctx;
        const RegionTreeContext tree_context = parent_ctx->get_context();
        for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
        {
          const RegionRequirement &req = deletion_requirements[idx];
          if (returnable_privileges[idx])
          {
            if (!has_outermost)
            {
              TaskContext *outermost = 
                parent_ctx->find_outermost_local_context();
              outermost_ctx = outermost->get_context();
              has_outermost = true;
            }
            runtime->forest->invalidate_versions(outermost_ctx, req.region);
          }
          else
            runtime->forest->invalidate_versions(tree_context, req.region);
        }
      }
      else if (kind == FIELD_DELETION)
      {
        // For this case we actually need to go through and prune out any
        // valid instances for these fields in the equivalence sets in order
        // to be able to free up the resources.
        const PhysicalTraceInfo trace_info(this, -1U, false/*init*/);
        for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
          runtime->forest->invalidate_fields(this, idx, version_infos[idx],
                                             trace_info,map_applied_conditions);
        // make sure that we don't try to do the deletion calls until
        // after the allocator is ready
        if (allocator->ready_event.exists())
          map_applied_conditions.insert(allocator->ready_event);
      }
      // Mark that we're done mapping and defer the execution as appropriate
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      // Wait for all the operations on which this deletion depends on to
      // complete before we are officially considered done execution
      if (execution_precondition.exists())
        complete_execution(Runtime::protect_event(execution_precondition));
      else
        complete_execution();
    }

    //--------------------------------------------------------------------------
    void DeletionOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      std::vector<LogicalRegion> regions_to_destroy;
      switch (kind)
      {
        case INDEX_SPACE_DELETION:
          {
#ifdef DEBUG_LEGION
            assert(deletion_req_indexes.empty());
#endif
            runtime->forest->destroy_index_space(index_space, 
                        runtime->address_space, preconditions);
            if (!sub_partitions.empty())
            {
              for (std::vector<IndexPartition>::const_iterator it = 
                    sub_partitions.begin(); it != sub_partitions.end(); it++)
                runtime->forest->destroy_index_partition(*it, preconditions);
            }
            break;
          }
        case INDEX_PARTITION_DELETION:
          {
#ifdef DEBUG_LEGION
            assert(deletion_req_indexes.empty());
#endif
            runtime->forest->destroy_index_partition(index_part, preconditions);
            if (!sub_partitions.empty())
            {
              for (std::vector<IndexPartition>::const_iterator it = 
                    sub_partitions.begin(); it != sub_partitions.end(); it++)
                runtime->forest->destroy_index_partition(*it, preconditions);
            }
            break;
          }
        case FIELD_SPACE_DELETION:
          {
#ifdef DEBUG_LEGION
            assert(deletion_req_indexes.empty());
#endif
            runtime->forest->destroy_field_space(field_space, preconditions);
            break;
          }
        case FIELD_DELETION:
          {
            if (!local_fields.empty())
              runtime->forest->free_local_fields(field_space, 
                            local_fields, local_field_indexes);
            if (!global_fields.empty())
              runtime->forest->free_fields(field_space, global_fields, 
                                           preconditions);
            parent_ctx->remove_deleted_fields(free_fields, parent_req_indexes);
            if (!local_fields.empty())
              parent_ctx->remove_deleted_local_fields(field_space,local_fields);
            if (!deletion_req_indexes.empty())
              parent_ctx->remove_deleted_requirements(deletion_req_indexes,
                                                      regions_to_destroy); 
            break;
          }
        case LOGICAL_REGION_DELETION:
          {
            // In this case we just remove all our regions since we know
            // we put a reference on all of them, if we're the last one
            // to remove the reference on each region then we'll delete them
            if (!parent_req_indexes.empty())
              parent_ctx->remove_deleted_requirements(parent_req_indexes,
                                                      regions_to_destroy);
            else
              // If we had no deletion requirements then we know there is
              // nothing to race with and we can just do our deletion
              runtime->forest->destroy_logical_region(logical_region,
                                                      preconditions);
            break;
          }
        default:
          assert(false);
      }
      if (!regions_to_destroy.empty())
      {
        for (std::vector<LogicalRegion>::const_iterator it =
              regions_to_destroy.begin(); it != regions_to_destroy.end(); it++)
          runtime->forest->destroy_logical_region(*it, preconditions);
      }
#ifdef LEGION_SPY
      // Still have to do this call to let Legion Spy know we're done
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      ApEvent::NO_AP_EVENT);
#endif
      if (!preconditions.empty())
        complete_operation(Runtime::merge_events(preconditions));
      else
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

    //--------------------------------------------------------------------------
    void DeletionOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
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
                                         const LogicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(creator != NULL);
#endif
      // We never track internal operations
      initialize_operation(creator->get_context(), false/*track*/,
                           1/*regions*/, creator->get_provenance());
#ifdef DEBUG_LEGION
      assert(creator_req_idx == -1);
      assert(create_op == NULL);
#endif
      create_op = creator;
      create_gen = creator->get_generation();
      creator_req_idx = intern_idx;
      if (trace_info.trace != NULL)
        set_trace(trace_info.trace, NULL, &trace_info); 
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
    // External Close 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalClose::ExternalClose(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ExternalClose::pack_external_close(Serializer &rez, 
                                            AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      pack_region_requirement(requirement, rez);
      rez.serialize<size_t>(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalClose::unpack_external_close(Deserializer &derez,
                                              Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      unpack_region_requirement(requirement, derez);
      size_t index;
      derez.deserialize(index);
      set_context_index(index);
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
    size_t CloseOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void CloseOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int CloseOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* CloseOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
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
    void CloseOp::initialize_close(InnerContext *ctx,
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
                                   const LogicalTraceInfo &trace_info)
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
      if (runtime->legion_spy_enabled)
        LegionSpy::log_internal_op_creator(unique_op_id, 
                                           creator->get_unique_op_id(), idx);
    }

    //--------------------------------------------------------------------------
    void CloseOp::perform_logging(void)
    //--------------------------------------------------------------------------
    {
      if (!runtime->legion_spy_enabled)
        return; 
      if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
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
      commit_operation(true/*deactivate*/);
    }

    /////////////////////////////////////////////////////////////
    // Inter Close Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MergeCloseOp::MergeCloseOp(Runtime *runtime)
      : CloseOp(runtime)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MergeCloseOp::MergeCloseOp(const MergeCloseOp &rhs)
      : CloseOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MergeCloseOp::~MergeCloseOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MergeCloseOp& MergeCloseOp::operator=(const MergeCloseOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void MergeCloseOp::initialize(InnerContext *ctx,
                              const RegionRequirement &req,
                              const LogicalTraceInfo &trace_info, int close_idx,
                              const FieldMask &close_m, Operation *creator)
    //--------------------------------------------------------------------------
    {
      if (runtime->legion_spy_enabled)
        LegionSpy::log_close_operation(ctx->get_unique_id(), unique_op_id,
                                       context_index, true/*inter close*/);
      parent_req_index = creator->find_parent_index(close_idx);
      initialize_close(creator, close_idx, parent_req_index, req, trace_info);
      close_mask = close_m;
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void MergeCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_close();
    }
    
    //--------------------------------------------------------------------------
    void MergeCloseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_close();
      close_mask.clear();
      runtime->free_merge_close_op(this);
    }

    //--------------------------------------------------------------------------
    const char* MergeCloseOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[MERGE_CLOSE_OP_KIND];
    }

    //-------------------------------------------------------------------------
    Operation::OpKind MergeCloseOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return MERGE_CLOSE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    const FieldMask& MergeCloseOp::get_internal_mask(void) const
    //--------------------------------------------------------------------------
    {
      return close_mask;
    }

    //--------------------------------------------------------------------------
    unsigned MergeCloseOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return parent_req_index;
    }

#ifdef LEGION_SPY
    //--------------------------------------------------------------------------
    void MergeCloseOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      // Still need this to record that this operation is done for LegionSpy
      LegionSpy::log_operation_events(unique_op_id, 
          ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
      complete_operation();
    }
#endif

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
    void PostCloseOp::initialize(InnerContext *ctx, unsigned idx,
                                 const InstanceSet &targets) 
    //--------------------------------------------------------------------------
    {
      initialize_close(ctx, ctx->regions[idx], true/*track*/);
      parent_idx = idx;
      target_instances = targets;
      localize_region_requirement(requirement);
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_close_operation(ctx->get_unique_id(), unique_op_id,
                                       context_index, false/*inter*/);
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
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_close();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      map_applied_conditions.clear();
      profiling_requests.clear();
      if (!profiling_info.empty())
      {
        for (unsigned idx = 0; idx < profiling_info.size(); idx++)
          free(profiling_info[idx].buffer);
        profiling_info.clear();
      }
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
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions);
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement,
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
      const PhysicalTraceInfo trace_info(this, 0/*index*/, false/*init*/);
      ApUserEvent close_event = Runtime::create_ap_user_event(NULL);
      ApEvent effects_done = 
        runtime->forest->physical_perform_updates_and_registration(
                                              requirement, version_info,
                                              this, 0/*idx*/,
                                              ApEvent::NO_AP_EVENT,
                                              close_event,
                                              target_instances, 
                                              trace_info,
                                              map_applied_conditions,
#ifdef DEBUG_LEGION
                                              get_logging_name(),
                                              unique_op_id,
#endif
                                              true/*record valid*/,
                                              false/*check initialized*/);
      std::set<ApEvent> close_preconditions;
      if (effects_done.exists())
        close_preconditions.insert(effects_done);
      for (unsigned idx = 0; idx < target_instances.size(); idx++)
      {
        ApEvent pre = target_instances[idx].get_ready_event();
        if (pre.exists())
          close_preconditions.insert(pre);
      }
      if (!close_preconditions.empty())
        Runtime::trigger_event(NULL, close_event, 
            Runtime::merge_events(&trace_info, close_preconditions));
      else
        Runtime::trigger_event(NULL, close_event);
      if (runtime->legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, parent_ctx,
                                              0/*idx*/, requirement,
                                              target_instances);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, close_event, 
                                        completion_event);
#endif
      }
      // No need to apply our mapping because we are done!
      RtEvent mapping_applied;
      if (!map_applied_conditions.empty())
        mapping_applied = Runtime::merge_events(map_applied_conditions);
      if (!acquired_instances.empty())
        mapping_applied = release_nonempty_acquired_instances(mapping_applied, 
                                                          acquired_instances);
      complete_mapping(mapping_applied);
      if (!request_early_complete(close_event))
        complete_execution(Runtime::protect_event(close_event));
      else
        complete_execution();
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we need to do a profiling response
      if (profiling_reported.exists())
      {
        if (outstanding_profiling_requests.load() > 0)
        {
#ifdef DEBUG_LEGION
          assert(mapped_event.has_triggered());
#endif
          std::vector<CloseProfilingInfo> to_perform;
          {
            AutoLock o_lock(op_lock);
            to_perform.swap(profiling_info);
          }
          if (!to_perform.empty())
          {
            for (unsigned idx = 0; idx < to_perform.size(); idx++)
            {
              CloseProfilingInfo &info = to_perform[idx];
              const Realm::ProfilingResponse resp(info.buffer,info.buffer_size);
              info.total_reports = outstanding_profiling_requests.load();
              info.profiling_responses.attach_realm_profiling_response(resp);
              mapper->invoke_close_report_profiling(this, &info);
              free(info.buffer);
            }
            const int count = to_perform.size() +
                outstanding_profiling_reported.fetch_add(to_perform.size());
#ifdef DEBUG_LEGION
            assert(count <= outstanding_profiling_requests.load());
#endif
            if (count == outstanding_profiling_requests.load())
              Runtime::trigger_event(profiling_reported);
          }
        }
        else
        {
          // We're not expecting any profiling callbacks so we need to
          // do one ourself to inform the mapper that there won't be any
          Mapping::Mapper::CloseProfilingInfo info;
          info.total_reports = 0;
          info.fill_response = false; // make valgrind happy
          mapper->invoke_close_report_profiling(this, &info);    
          Runtime::trigger_event(profiling_reported);
        }
      }
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
    void PostCloseOp::select_sources(const unsigned index,
                                     const InstanceRef &target,
                                     const InstanceSet &sources,
                                     std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index == 0);
#endif
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
      compute_ranking(mapper, output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,unsigned>* 
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
    void PostCloseOp::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &requests, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest &request = requests.add_request( 
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
          &response, sizeof(response), profiling_priority);
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      handle_profiling_update(count);
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::handle_profiling_response(
                                       const ProfilingResponseBase *base,
                                       const Realm::ProfilingResponse &response,
                                       const void *orig, size_t orig_length)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapper != NULL);
#endif
      const OpProfilingResponse *op_info = 
        static_cast<const OpProfilingResponse*>(base);
      // Check to see if we are done mapping, if not then we need to defer
      // this until we are done mapping so we know how many
      if (!mapped_event.has_triggered())
      {
        // Take the lock and see if we lost the race
        AutoLock o_lock(op_lock);
        if (!mapped_event.has_triggered())
        {
          // Save this profiling response for later until we know the
          // full count of profiling responses
          profiling_info.resize(profiling_info.size() + 1);
          CloseProfilingInfo &info = profiling_info.back();
          info.fill_response = op_info->fill;
          info.buffer_size = orig_length;
          info.buffer = malloc(orig_length);
          memcpy(info.buffer, orig, orig_length);
          return;
        }
      }
      // If we get here then we can handle the response now
      Mapping::Mapper::CloseProfilingInfo info; 
      info.profiling_responses.attach_realm_profiling_response(response);
      info.total_reports = outstanding_profiling_requests.load();
      info.fill_response = op_info->fill;
      mapper->invoke_close_report_profiling(this, &info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(count <= outstanding_profiling_requests.load());
#endif
      if (count == outstanding_profiling_requests.load())
        Runtime::trigger_event(profiling_reported);
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count > 0);
      assert(!mapped_event.has_triggered());
#endif
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::pack_remote_operation(Serializer &rez, 
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_close(rez, target);
      rez.serialize<size_t>(profiling_requests.size());
      if (!profiling_requests.empty())
      {
        for (unsigned idx = 0; idx < profiling_requests.size(); idx++)
          rez.serialize(profiling_requests[idx]);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Create a user event for this response
        const RtUserEvent response = Runtime::create_rt_user_event();
        rez.serialize(response);
        applied_events.insert(response);
      }
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
    void VirtualCloseOp::initialize(InnerContext *ctx, unsigned index,
                                    const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      initialize_close(ctx, req, true/*track*/);
      parent_idx = index;
      localize_region_requirement(requirement);
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_close_operation(ctx->get_unique_id(), unique_op_id,
                                       context_index, false/*inter*/);
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
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions);
    }

    //--------------------------------------------------------------------------
    void VirtualCloseOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      complete_execution();
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

#ifdef LEGION_SPY
    //--------------------------------------------------------------------------
    void VirtualCloseOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      // Still need this to record that this operation is done for LegionSpy
      LegionSpy::log_operation_events(unique_op_id, 
          ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
      complete_operation();
    }
#endif

    /////////////////////////////////////////////////////////////
    // External Acquire
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalAcquire::ExternalAcquire(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ExternalAcquire::pack_external_acquire(Serializer &rez,
                                                AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(logical_region);
      rez.serialize(parent_region);
      rez.serialize<size_t>(fields.size());
      for (std::set<FieldID>::const_iterator it = 
            fields.begin(); it != fields.end(); it++)
        rez.serialize(*it);
      rez.serialize(grants.size());
      for (unsigned idx = 0; idx < grants.size(); idx++)
        pack_grant(grants[idx], rez);
      rez.serialize(wait_barriers.size());
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        pack_phase_barrier(wait_barriers[idx], rez);
      rez.serialize(arrive_barriers.size());
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        pack_phase_barrier(arrive_barriers[idx], rez);
      pack_mappable(*this, rez);
      rez.serialize<size_t>(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalAcquire::unpack_external_acquire(Deserializer &derez,
                                                  Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(logical_region);
      derez.deserialize(parent_region);
      size_t num_fields;
      derez.deserialize(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        fields.insert(fid);
      }
      size_t num_grants;
      derez.deserialize(num_grants);
      grants.resize(num_grants);
      for (unsigned idx = 0; idx < grants.size(); idx++)
        unpack_grant(grants[idx], derez);
      size_t num_wait_barriers;
      derez.deserialize(num_wait_barriers);
      wait_barriers.resize(num_wait_barriers);
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        unpack_phase_barrier(wait_barriers[idx], derez);
      size_t num_arrive_barriers;
      derez.deserialize(num_arrive_barriers);
      arrive_barriers.resize(num_arrive_barriers);
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        unpack_phase_barrier(arrive_barriers[idx], derez);
      unpack_mappable(*this, derez);
      size_t index;
      derez.deserialize(index);
      set_context_index(index);
    }

    /////////////////////////////////////////////////////////////
    // Acquire Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AcquireOp::AcquireOp(Runtime *rt)
      : ExternalAcquire(), MemoizableOp<SpeculativeOp>(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AcquireOp::AcquireOp(const AcquireOp &rhs)
      : ExternalAcquire(), MemoizableOp<SpeculativeOp>(NULL)
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
    void AcquireOp::initialize(InnerContext *ctx,
                               const AcquireLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_speculation(ctx, true/*track*/,
                             1/*num region requirements*/,
                             launcher.static_dependences,
                             launcher.predicate, launcher.provenance.c_str());
      initialize_memoizable();
      // Note we give it READ WRITE EXCLUSIVE to make sure that nobody
      // can be re-ordered around this operation for mapping or
      // normal dependences.  We won't actually read or write anything.
      requirement = RegionRequirement(launcher.logical_region, 
          LEGION_READ_WRITE, LEGION_EXCLUSIVE, launcher.parent_region); 
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
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      if (runtime->legion_spy_enabled)
        LegionSpy::log_acquire_operation(parent_ctx->get_unique_id(),
                                         unique_op_id, context_index);
    }

    //--------------------------------------------------------------------------
    void AcquireOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative();
      activate_memoizable();
      mapper = NULL;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_speculative();  
      privilege_path.clear();
      version_info.clear();
      fields.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      map_applied_conditions.clear();
      profiling_requests.clear();
      if (!profiling_info.empty())
      {
        for (unsigned idx = 0; idx < profiling_info.size(); idx++)
          free(profiling_info[idx].buffer);
        profiling_info.clear();
      }
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
      if (runtime->legion_spy_enabled)
        log_acquire_requirement();
    }

    //--------------------------------------------------------------------------
    void AcquireOp::log_acquire_requirement(void)
    //--------------------------------------------------------------------------
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

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {  
      if (runtime->check_privileges)
        check_acquire_privilege();
      // Register a dependence on our predicate
      register_predicate_dependence();
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(this, 
                              wait_barriers, arrive_barriers);
      // First register any mapping dependences that we have
      ProjectionInfo projection_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions);
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
      if (is_replaying())
      {
        enqueue_ready_operation();
        return;
      }

      std::set<RtEvent> preconditions;  
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement,
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
      const PhysicalTraceInfo trace_info(this, 0/*index*/, true/*initialize*/);
      // Invoke the mapper before doing anything else 
      invoke_mapper();
      InstanceSet restricted_instances;
      ApEvent acquire_complete = 
        runtime->forest->acquire_restrictions(requirement, version_info,
                                              this, 0/*idx*/, completion_event,
                                              restricted_instances,
                                              trace_info, 
                                              map_applied_conditions
#ifdef DEBUG_LEGION
                                              , get_logging_name()
                                              , unique_op_id
#endif
                                              );
#ifdef DEBUG_LEGION
      dump_physical_state(&requirement, 0);
#endif
      ApEvent init_precondition = compute_init_precondition(trace_info);
      if (init_precondition.exists())
        acquire_complete = Runtime::merge_events(&trace_info, 
                                   acquire_complete, init_precondition);
      if (runtime->legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, parent_ctx, 
                                              0/*idx*/, requirement,
                                              restricted_instances);
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
          if (runtime->legion_spy_enabled)
            LegionSpy::log_phase_barrier_arrival(unique_op_id, 
                                                 it->phase_barrier);
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        completion_event);
        }
      }
      // Remove profiling our guard and trigger the profiling event if necessary
      if ((outstanding_profiling_requests.fetch_sub(1) == 1) &&
          profiling_reported.exists())
        Runtime::trigger_event(profiling_reported);
      if (is_recording())
        tpl->record_complete_replay(this, acquire_complete);
      // Mark that we completed mapping
      RtEvent mapping_applied;
      if (!map_applied_conditions.empty())
        mapping_applied = Runtime::merge_events(map_applied_conditions);
      if (!acquired_instances.empty())
        mapping_applied = release_nonempty_acquired_instances(mapping_applied, 
                                                          acquired_instances);
      complete_mapping(mapping_applied);
      if (!request_early_complete(acquire_complete))
        complete_execution(Runtime::protect_event(acquire_complete));
      else
        complete_execution();
    }

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we need to do a profiling response
      if (profiling_reported.exists())
      {
        if (outstanding_profiling_requests.load() > 0)
        {
#ifdef DEBUG_LEGION
          assert(mapped_event.has_triggered());
#endif
          std::vector<AcquireProfilingInfo> to_perform;
          {
            AutoLock o_lock(op_lock);
            to_perform.swap(profiling_info);
          }
          if (!to_perform.empty())
          {
            for (unsigned idx = 0; idx < to_perform.size(); idx++)
            {
              AcquireProfilingInfo &info = to_perform[idx];
              const Realm::ProfilingResponse resp(info.buffer,info.buffer_size);
              info.total_reports = outstanding_profiling_requests.load();
              info.profiling_responses.attach_realm_profiling_response(resp);
              mapper->invoke_acquire_report_profiling(this, &info);
              free(info.buffer);
            }
            const int count = to_perform.size() +
                outstanding_profiling_reported.fetch_add(to_perform.size());
#ifdef DEBUG_LEGION
            assert(count <= outstanding_profiling_requests.load());
#endif
            if (count == outstanding_profiling_requests.load())
              Runtime::trigger_event(profiling_reported);
          }
        }
        else
        {
          // We're not expecting any profiling callbacks so we need to
          // do one ourself to inform the mapper that there won't be any
          Mapping::Mapper::AcquireProfilingInfo info;
          info.total_reports = 0;
          info.fill_response = false; // make valgrind happy
          mapper->invoke_acquire_report_profiling(this, &info);    
          Runtime::trigger_event(profiling_reported);
        }
      }
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
    std::map<PhysicalManager*,unsigned>* 
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
    size_t AcquireOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int AcquireOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* AcquireOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& AcquireOp::get_requirement(void) const
    //--------------------------------------------------------------------------
    {
      return requirement;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->legion_spy_enabled && !need_prepipeline_stage)
        log_acquire_requirement();
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      tpl->register_operation(this);
      complete_mapping();
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    ApEvent AcquireOp::compute_sync_precondition(const TraceInfo *info) const
    //--------------------------------------------------------------------------
    {
      ApEvent result;
      if (!wait_barriers.empty() || !grants.empty())
      {
        std::set<ApEvent> sync_preconditions;
        if (!wait_barriers.empty())
        {
          for (std::vector<PhaseBarrier>::const_iterator it =
                wait_barriers.begin(); it != wait_barriers.end(); it++)
          {
            ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
            sync_preconditions.insert(e);
            if (runtime->legion_spy_enabled)
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
        // For some reason we don't trace these, not sure why
        result = Runtime::merge_events(NULL, sync_preconditions);
      }
      if ((info != NULL) && info->recording)
        info->record_op_sync_event(result);
      return result;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::complete_replay(ApEvent acquire_complete_event)
    //--------------------------------------------------------------------------
    {
      // Chain all the unlock and barrier arrivals off of the
      // copy complete event
      if (!arrive_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::iterator it =
              arrive_barriers.begin(); it != arrive_barriers.end(); it++)
        {
          if (runtime->legion_spy_enabled)
            LegionSpy::log_phase_barrier_arrival(unique_op_id,
                                                 it->phase_barrier);
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        completion_event);
        }
      }

      // Handle the case for marking when the copy completes
      Runtime::trigger_event(NULL, completion_event, acquire_complete_event);
      need_completion_trigger = false;
      complete_execution();
    }

    //--------------------------------------------------------------------------
    const VersionInfo& AcquireOp::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return version_info;
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& AcquireOp::get_requirement(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return get_requirement();
    }

    //--------------------------------------------------------------------------
    void AcquireOp::check_acquire_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field = LEGION_AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, check the privileges, but only check the
      // data and not the actual privilege values since we're
      // using psuedo-read-write-exclusive
      if (et == LEGION_NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, 
                                         bad_index, true/*skip*/);
      switch (et)
      {
        case LEGION_NO_ERROR:
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
            FieldSpace sp = 
              (requirement.handle_type == LEGION_SINGULAR_PROJECTION) ||
              (requirement.handle_type == LEGION_REGION_PROJECTION)
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
            else if (bad_field == LEGION_AUTO_GENERATE_ID) 
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
#ifdef DEBUG_LEGION
        assert(!profiling_reported.exists());
#endif
        profiling_reported = Runtime::create_rt_user_event();
      }
    }

    //--------------------------------------------------------------------------
    void AcquireOp::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &requests, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest &request = requests.add_request( 
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
          &response, sizeof(response), profiling_priority);
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      handle_profiling_update(count);
    }

    //--------------------------------------------------------------------------
    void AcquireOp::handle_profiling_response(const ProfilingResponseBase *base,
                                       const Realm::ProfilingResponse &response,
                                       const void *orig, size_t orig_length)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapper != NULL);
#endif
      const OpProfilingResponse *op_info = 
        static_cast<const OpProfilingResponse*>(base);
      // Check to see if we are done mapping, if not then we need to defer
      // this until we are done mapping so we know how many reports to expect
      if (!mapped_event.has_triggered())
      {
        // Take the lock and see if we lost the race
        AutoLock o_lock(op_lock);
        if (!mapped_event.has_triggered())
        {
          // Save this profiling response for later until we know the
          // full count of profiling responses
          profiling_info.resize(profiling_info.size() + 1);
          AcquireProfilingInfo &info = profiling_info.back();
          info.fill_response = op_info->fill;
          info.buffer_size = orig_length;
          info.buffer = malloc(orig_length);
          memcpy(info.buffer, orig, orig_length);
          return;
        }
      }
      // If we get here then we can handle the response now
      Mapping::Mapper::AcquireProfilingInfo info; 
      info.profiling_responses.attach_realm_profiling_response(response);
      info.total_reports = outstanding_profiling_requests.load();
      info.fill_response = op_info->fill;
      mapper->invoke_acquire_report_profiling(this, &info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(count <= outstanding_profiling_requests.load());
#endif
      if (count == outstanding_profiling_requests.load())
        Runtime::trigger_event(profiling_reported);
    }

    //--------------------------------------------------------------------------
    void AcquireOp::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count > 0);
      assert(!mapped_event.has_triggered());
#endif
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void AcquireOp::pack_remote_operation(Serializer &rez,AddressSpaceID target,
                                        std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_acquire(rez, target);
      rez.serialize<size_t>(profiling_requests.size());
      if (!profiling_requests.empty())
      {
        for (unsigned idx = 0; idx < profiling_requests.size(); idx++)
          rez.serialize(profiling_requests[idx]);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Create a user event for this response
        const RtUserEvent response = Runtime::create_rt_user_event();
        rez.serialize(response);
        applied_events.insert(response);
      }
    }

    /////////////////////////////////////////////////////////////
    // External Release 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalRelease::ExternalRelease(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ExternalRelease::pack_external_release(Serializer &rez,
                                                AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(logical_region);
      rez.serialize(parent_region);
      rez.serialize<size_t>(fields.size());
      for (std::set<FieldID>::const_iterator it = 
            fields.begin(); it != fields.end(); it++)
        rez.serialize(*it);
      rez.serialize(grants.size());
      for (unsigned idx = 0; idx < grants.size(); idx++)
        pack_grant(grants[idx], rez);
      rez.serialize(wait_barriers.size());
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        pack_phase_barrier(wait_barriers[idx], rez);
      rez.serialize(arrive_barriers.size());
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        pack_phase_barrier(arrive_barriers[idx], rez);
      pack_mappable(*this, rez);
      rez.serialize<size_t>(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalRelease::unpack_external_release(Deserializer &derez,
                                                  Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(logical_region);
      derez.deserialize(parent_region);
      size_t num_fields;
      derez.deserialize(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        fields.insert(fid);
      }
      size_t num_grants;
      derez.deserialize(num_grants);
      grants.resize(num_grants);
      for (unsigned idx = 0; idx < grants.size(); idx++)
        unpack_grant(grants[idx], derez);
      size_t num_wait_barriers;
      derez.deserialize(num_wait_barriers);
      wait_barriers.resize(num_wait_barriers);
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        unpack_phase_barrier(wait_barriers[idx], derez);
      size_t num_arrive_barriers;
      derez.deserialize(num_arrive_barriers);
      arrive_barriers.resize(num_arrive_barriers);
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        unpack_phase_barrier(arrive_barriers[idx], derez);
      unpack_mappable(*this, derez);
      size_t index;
      derez.deserialize(index);
      set_context_index(index);
    }

    /////////////////////////////////////////////////////////////
    // Release Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReleaseOp::ReleaseOp(Runtime *rt)
      : ExternalRelease(), MemoizableOp<SpeculativeOp>(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReleaseOp::ReleaseOp(const ReleaseOp &rhs)
      : ExternalRelease(), MemoizableOp<SpeculativeOp>(NULL)
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
    void ReleaseOp::initialize(InnerContext *ctx, 
                               const ReleaseLauncher &launcher) 
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_speculation(ctx, true/*track*/, 
                             1/*num region requirements*/,
                             launcher.static_dependences,
                             launcher.predicate, launcher.provenance.c_str());
      initialize_memoizable();
      // Note we give it READ WRITE EXCLUSIVE to make sure that nobody
      // can be re-ordered around this operation for mapping or
      // normal dependences.  We won't actually read or write anything.
      requirement = RegionRequirement(launcher.logical_region, 
          LEGION_READ_WRITE, LEGION_EXCLUSIVE, launcher.parent_region); 
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
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      if (runtime->legion_spy_enabled)
        LegionSpy::log_release_operation(parent_ctx->get_unique_id(),
                                         unique_op_id, context_index);
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative(); 
      activate_memoizable();
      mapper = NULL;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_speculative();
      privilege_path.clear();
      version_info.clear();
      fields.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      map_applied_conditions.clear();
      profiling_requests.clear();
      if (!profiling_info.empty())
      {
        for (unsigned idx = 0; idx < profiling_info.size(); idx++)
          free(profiling_info[idx].buffer);
        profiling_info.clear();
      }
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
      if (runtime->legion_spy_enabled)
        log_release_requirement();
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::log_release_requirement(void)
    //--------------------------------------------------------------------------
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

    //--------------------------------------------------------------------------
    void ReleaseOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {  
      if (runtime->check_privileges)
        check_release_privilege();
      // Register a dependence on our predicate
      register_predicate_dependence();
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(this, 
                              wait_barriers, arrive_barriers);
      // First register any mapping dependences that we have
      ProjectionInfo projection_info;
      // Register any mapping dependences that we have
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions);
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
      if (is_replaying())
      {
        enqueue_ready_operation();
        return;
      }

      std::set<RtEvent> preconditions;
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement,
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
      const PhysicalTraceInfo trace_info(this, 0/*index*/, true/*initialize*/);
      // Invoke the mapper before doing anything else 
      invoke_mapper();
      InstanceSet restricted_instances;
      ApEvent init_precondition = compute_init_precondition(trace_info); 
      ApEvent release_complete = 
        runtime->forest->release_restrictions(requirement, version_info,
                                              this, 0/*idx*/, init_precondition,
                                              completion_event,
                                              restricted_instances, trace_info,
                                              map_applied_conditions
#ifdef DEBUG_LEGION
                                              , get_logging_name()
                                              , unique_op_id
#endif
                                              );
#ifdef DEBUG_LEGION
      dump_physical_state(&requirement, 0);
#endif
      if (init_precondition.exists())
        release_complete = Runtime::merge_events(&trace_info,
                            release_complete, init_precondition);
      if (runtime->legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, parent_ctx, 
                                              0/*idx*/, requirement,
                                              restricted_instances);
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
          if (runtime->legion_spy_enabled)
            LegionSpy::log_phase_barrier_arrival(unique_op_id, 
                                                 it->phase_barrier);
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        completion_event);
        }
      }
      // Remove profiling our guard and trigger the profiling event if necessary
      if ((outstanding_profiling_requests.fetch_sub(1) == 1) &&
          profiling_reported.exists())
        Runtime::trigger_event(profiling_reported);
      if (is_recording())
        tpl->record_complete_replay(this, release_complete);
      // Mark that we completed mapping
      RtEvent mapping_applied;
      if (!map_applied_conditions.empty())
        mapping_applied = Runtime::merge_events(map_applied_conditions);
      if (!acquired_instances.empty())
        mapping_applied = release_nonempty_acquired_instances(mapping_applied, 
                                                          acquired_instances);
      complete_mapping(mapping_applied);
      if (!request_early_complete(release_complete))
        complete_execution(Runtime::protect_event(release_complete));
      else
        complete_execution();
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we need to do a profiling response
      if (profiling_reported.exists())
      {
        if (outstanding_profiling_requests.load() > 0)
        {
#ifdef DEBUG_LEGION
          assert(mapped_event.has_triggered());
#endif
          std::vector<ReleaseProfilingInfo> to_perform;
          {
            AutoLock o_lock(op_lock);
            to_perform.swap(profiling_info);
          }
          if (!to_perform.empty())
          {
            for (unsigned idx = 0; idx < to_perform.size(); idx++)
            {
              ReleaseProfilingInfo &info = to_perform[idx];
              const Realm::ProfilingResponse resp(info.buffer,info.buffer_size);
              info.total_reports = outstanding_profiling_requests.load();
              info.profiling_responses.attach_realm_profiling_response(resp);
              mapper->invoke_release_report_profiling(this, &info);
              free(info.buffer);
            }
            const int count = to_perform.size() +
                outstanding_profiling_reported.fetch_add(to_perform.size());
#ifdef DEBUG_LEGION
            assert(count <= outstanding_profiling_requests.load());
#endif
            if (count == outstanding_profiling_requests.load())
              Runtime::trigger_event(profiling_reported);
          }
        }
        else
        {
          // We're not expecting any profiling callbacks so we need to
          // do one ourself to inform the mapper that there won't be any
          Mapping::Mapper::ReleaseProfilingInfo info;
          info.total_reports = 0;
          info.fill_response = false; // make valgrind happy
          mapper->invoke_release_report_profiling(this, &info);    
          Runtime::trigger_event(profiling_reported);
        }
      }
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
    void ReleaseOp::select_sources(const unsigned index,
                                   const InstanceRef &target,
                                   const InstanceSet &sources,
                                   std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index == 0);
#endif
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
      compute_ranking(mapper, output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,unsigned>* 
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
    size_t ReleaseOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int ReleaseOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* ReleaseOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& ReleaseOp::get_requirement(void) const
    //--------------------------------------------------------------------------
    {
      return requirement;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->legion_spy_enabled && !need_prepipeline_stage)
        log_release_requirement();
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      tpl->register_operation(this);
      complete_mapping();
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    ApEvent ReleaseOp::compute_sync_precondition(const TraceInfo *info) const
    //--------------------------------------------------------------------------
    {
      ApEvent result;
      if (!wait_barriers.empty() || !grants.empty())
      {
        std::set<ApEvent> sync_preconditions;
        if (!wait_barriers.empty())
        {
          for (std::vector<PhaseBarrier>::const_iterator it =
                wait_barriers.begin(); it != wait_barriers.end(); it++)
          {
            ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
            sync_preconditions.insert(e);
            if (runtime->legion_spy_enabled)
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
        // For some reason we don't trace these, not sure why
        result = Runtime::merge_events(NULL, sync_preconditions);
      }
      if ((info != NULL) && info->recording)
        info->record_op_sync_event(result);
      return result;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::complete_replay(ApEvent release_complete_event)
    //--------------------------------------------------------------------------
    {
      // Chain all the unlock and barrier arrivals off of the
      // copy complete event
      if (!arrive_barriers.empty())
      {
        for (std::vector<PhaseBarrier>::iterator it =
              arrive_barriers.begin(); it != arrive_barriers.end(); it++)
        {
          if (runtime->legion_spy_enabled)
            LegionSpy::log_phase_barrier_arrival(unique_op_id,
                                                 it->phase_barrier);
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        completion_event);
        }
      }

      // Handle the case for marking when the copy completes
      Runtime::trigger_event(NULL, completion_event, release_complete_event);
      need_completion_trigger = false;
      complete_execution();
    }

    //--------------------------------------------------------------------------
    const VersionInfo& ReleaseOp::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return version_info;
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& ReleaseOp::get_requirement(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return get_requirement();
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::check_release_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field = LEGION_AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, check the privileges, but only check the
      // data and not the actual privilege values since we're
      // using psuedo-read-write-exclusive
      if (et == LEGION_NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, 
                                         bad_index, true/*skip*/);
      switch (et)
      {
          // There is no such thing as bad privileges for release operations
          // because we control what they are doing
        case LEGION_NO_ERROR:
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
            FieldSpace sp = 
              (requirement.handle_type == LEGION_SINGULAR_PROJECTION) ||
              (requirement.handle_type == LEGION_REGION_PROJECTION)
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
            else if (bad_field == LEGION_AUTO_GENERATE_ID) 
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
#ifdef DEBUG_LEGION
        assert(!profiling_reported.exists());
#endif
        profiling_reported = Runtime::create_rt_user_event();
      }
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &requests, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest &request = requests.add_request( 
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
          &response, sizeof(response), profiling_priority);
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      handle_profiling_update(count);
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::handle_profiling_response(const ProfilingResponseBase *base,
                                       const Realm::ProfilingResponse &response,
                                       const void *orig, size_t orig_length)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapper != NULL);
#endif
      const OpProfilingResponse *op_info = 
        static_cast<const OpProfilingResponse*>(base);
      // Check to see if we are done mapping, if not then we need to defer
      // this until we are done mapping so we know how many reports to expect
      if (!mapped_event.has_triggered())
      {
        // Take the lock and see if we lost the race
        AutoLock o_lock(op_lock);
        if (!mapped_event.has_triggered())
        {
          // Save this profiling response for later until we know the
          // full count of profiling responses
          profiling_info.resize(profiling_info.size() + 1);
          ReleaseProfilingInfo &info = profiling_info.back();
          info.fill_response = op_info->fill;
          info.buffer_size = orig_length;
          info.buffer = malloc(orig_length);
          memcpy(info.buffer, orig, orig_length);
          return;
        }
      }
      // If we get here then we can handle the response now
      Mapping::Mapper::ReleaseProfilingInfo info; 
      info.profiling_responses.attach_realm_profiling_response(response);
      info.total_reports = outstanding_profiling_requests.load();
      info.fill_response = op_info->fill;
      mapper->invoke_release_report_profiling(this, &info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(count <= outstanding_profiling_requests.load());
#endif
      if (count == outstanding_profiling_requests.load())
        Runtime::trigger_event(profiling_reported);
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count > 0);
      assert(!mapped_event.has_triggered());
#endif
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::pack_remote_operation(Serializer &rez,AddressSpaceID target,
                                        std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_release(rez, target);
      rez.serialize<size_t>(profiling_requests.size());
      if (!profiling_requests.empty())
      {
        for (unsigned idx = 0; idx < profiling_requests.size(); idx++)
          rez.serialize(profiling_requests[idx]);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Create a user event for this response
        const RtUserEvent response = Runtime::create_rt_user_event();
        rez.serialize(response);
        applied_events.insert(response);
      }
    }

    /////////////////////////////////////////////////////////////
    // Dynamic Collective Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DynamicCollectiveOp::DynamicCollectiveOp(Runtime *rt)
      : MemoizableOp<Operation>(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DynamicCollectiveOp::DynamicCollectiveOp(const DynamicCollectiveOp &rhs)
      : MemoizableOp<Operation>(NULL)
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
    Future DynamicCollectiveOp::initialize(InnerContext *ctx, 
                            const DynamicCollective &dc, const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
      initialize_memoizable();
      future = Future(new FutureImpl(parent_ctx, runtime, true/*register*/,
            runtime->get_available_distributed_id(), 
            runtime->address_space, get_completion_event(), this));
      collective = dc;
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_dynamic_collective(ctx->get_unique_id(),
                                          unique_op_id, context_index);
        DomainPoint empty_point;
        LegionSpy::log_future_creation(unique_op_id, 
                                 future.impl->get_ready_event(), empty_point);
      }
      return future;
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      trigger_mapping();
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      activate_memoizable();
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
      std::vector<PhaseBarrier> wait_barriers, no_arrival_barriers;
      wait_barriers.push_back(collective);
      parent_ctx->perform_barrier_dependence_analysis(this,
                        wait_barriers, no_arrival_barriers);
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      complete_mapping();
      ApEvent barrier = Runtime::get_previous_phase(collective.phase_barrier);
      if (!barrier.has_triggered_faultignorant())
      {
        const RtEvent safe = Runtime::protect_event(barrier);
        if (safe.exists() && !safe.has_triggered())
          parent_ctx->add_to_trigger_execution_queue(this, safe);
        else
          trigger_execution();
      }
      else
        trigger_execution();
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::trigger_execution(void)
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
#ifdef LEGION_SPY
      LegionSpy::log_operation_events(unique_op_id,
          ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
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
    void FuturePredOp::initialize(InnerContext *ctx, Future f,
                                  const char *provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
      assert(f.impl != NULL);
#endif
      // Don't track this as it can lead to deadlock because
      // predicates can't complete until all their references from
      // the parent task have been removed.
      initialize_operation(ctx, false/*track*/, 0/*regions*/, provenance);
      future = f;
      if (runtime->legion_spy_enabled)
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
      // Mark that we completed mapping this operation
      complete_mapping();
      // See if we have a value
      bool valid;
      bool value = future.impl->get_boolean_value(valid);
#ifdef LEGION_SPY
      // Still have to do this call to let Legion Spy know we're done
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      ApEvent::NO_AP_EVENT);
#endif
      if (valid)
        set_resolved_value(get_generation(), value);
      else
      {
        // Launch a task to get the value
        add_predicate_reference();
        ResolveFuturePredArgs args(this);
        runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY,
                      Runtime::protect_event(future.impl->subscribe()));
      }
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
    void NotPredOp::initialize(InnerContext *ctx, 
                               const Predicate &p, const char *provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
#endif
      // Don't track this as it can lead to deadlock because
      // predicates can't complete until all their references from
      // the parent task have been removed.
      initialize_operation(ctx, false/*track*/, 0/*regions*/, provenance);
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
      if (runtime->legion_spy_enabled)
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
      complete_mapping();
#ifdef LEGION_SPY
      // Still have to do this call to let Legion Spy know we're done
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      ApEvent::NO_AP_EVENT);
#endif
      if (pred_op != NULL)
      {
        bool prev_value;
        bool valid = pred_op->register_waiter(this, get_generation(),
                                              prev_value);
        // Now we can remove the reference we added
        pred_op->remove_predicate_reference();
        // Don't forget to negate 
        if (valid)
          set_resolved_value(get_generation(), !prev_value);
      }
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
    void AndPredOp::initialize(InnerContext *ctx, 
                               const std::vector<Predicate> &predicates,
                               const std::string &provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
#endif
      // Don't track this as it can lead to deadlock because
      // predicates can't complete until all their references from
      // the parent task have been removed.
      initialize_operation(ctx, false/*track*/,0/*regions*/,provenance.c_str());
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
      if (runtime->legion_spy_enabled)
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
      complete_mapping();
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
      // Clean up any references that we have
      for (std::vector<PredicateOp*>::const_iterator it = previous.begin();
            it != previous.end(); it++)
        (*it)->remove_predicate_reference();
#ifdef LEGION_SPY
      // Still have to do this call to let Legion Spy know we're done
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      ApEvent::NO_AP_EVENT);
#endif
      if (need_resolve)
        set_resolved_value(local_gen, !false_short);
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
    void OrPredOp::initialize(InnerContext *ctx, 
                              const std::vector<Predicate> &predicates,
                              const std::string &provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
#endif
      // Don't track this as it can lead to deadlock because
      // predicates can't complete until all their references from
      // the parent task have been removed.
      initialize_operation(ctx, false/*track*/,0/*regions*/,provenance.c_str());
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
      if (runtime->legion_spy_enabled)
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
      complete_mapping();
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
      // Clean up any references that we have
      for (std::vector<PredicateOp*>::const_iterator it = previous.begin();
            it != previous.end(); it++)
        (*it)->remove_predicate_reference();
#ifdef LEGION_SPY
      // Still have to do this call to let Legion Spy know we're done
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      ApEvent::NO_AP_EVENT);
#endif
      if (need_resolve)
        set_resolved_value(local_gen, true_short);
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
    FutureMap MustEpochOp::initialize(InnerContext *ctx,
                                      const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      // Initialize this operation
      initialize_operation(ctx, true/*track*/, 0/*regions*/,
          launcher.provenance.c_str());
      // Make a new future map for storing our results
      // We'll fill it in later
      result_map = FutureMap(new FutureMapImpl(ctx, this, 
            Runtime::protect_event(get_completion_event()), runtime,
            runtime->get_available_distributed_id(),
            runtime->address_space));
      // Initialize operations for everything in the launcher
      // Note that we do not track these operations as we want them all to
      // appear as a single operation to the parent context in order to
      // avoid deadlock with the maximum window size.
      indiv_tasks.resize(launcher.single_tasks.size());
      for (unsigned idx = 0; idx < launcher.single_tasks.size(); idx++)
      {
        indiv_tasks[idx] = runtime->get_available_individual_task();
        indiv_tasks[idx]->initialize_task(ctx, launcher.single_tasks[idx],
                                          false/*track*/);
        indiv_tasks[idx]->initialize_must_epoch(this, idx, true/*register*/);
        // If we have a trace, set it for this operation as well
        if (trace != NULL)
          indiv_tasks[idx]->set_trace(trace, NULL);
      }
      indiv_triggered.resize(indiv_tasks.size(), false);
      index_tasks.resize(launcher.index_tasks.size());
      for (unsigned idx = 0; idx < launcher.index_tasks.size(); idx++)
      {
        IndexSpace launch_space = launcher.index_tasks[idx].launch_space;
        if (!launch_space.exists())
          launch_space = ctx->find_index_launch_space(
              launcher.index_tasks[idx].launch_domain, launcher.provenance);
        index_tasks[idx] = runtime->get_available_index_task();
        index_tasks[idx]->initialize_task(ctx, launcher.index_tasks[idx],
                                          launch_space, false/*track*/);
        index_tasks[idx]->initialize_must_epoch(this, 
            indiv_tasks.size() + idx, true/*register*/);
        if (trace != NULL)
          index_tasks[idx]->set_trace(trace, NULL);
      }
      index_triggered.resize(index_tasks.size(), false);
      mapper_id = launcher.map_id;
      mapper_tag = launcher.mapping_tag; 
#ifdef DEBUG_LEGION
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
        result_map.impl->add_valid_point(indiv_tasks[idx]->index_point);
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
        result_map.impl->add_valid_domain(index_tasks[idx]->index_domain);
#endif
      if (runtime->legion_spy_enabled)
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
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
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
      slice_version_events.clear();
      completion_preconditions.clear();
      commit_preconditions.clear();
      completion_effects.clear();
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
      if (!acquired_instances.empty())
        all_mapped = 
          release_nonempty_acquired_instances(all_mapped, acquired_instances);
      complete_mapping(all_mapped);
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
        ApEvent effects_done;
        if (!completion_effects.empty())
          effects_done = Runtime::merge_events(NULL, completion_effects);
        if (!request_early_complete(effects_done))
          completion_preconditions.insert(Runtime::protect_event(effects_done));
        RtEvent completion_done;
        if (!completion_preconditions.empty())
          completion_done = Runtime::merge_events(completion_preconditions);
#ifdef LEGION_SPY
        // Still need this for Legion Spy
        LegionSpy::log_operation_events(unique_op_id,
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        complete_operation(completion_done);
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
        if ((dtype == LEGION_TRUE_DEPENDENCE) || 
            (dtype == LEGION_ANTI_DEPENDENCE) ||
            (dtype == LEGION_ATOMIC_DEPENDENCE))
        {
          TaskOp *src_task = find_task_by_index(src_index);
          TaskOp *dst_task = find_task_by_index(dst_index);
          REPORT_LEGION_ERROR(ERROR_MUST_EPOCH_DEPENDENCE,
                        "MUST EPOCH ERROR: dependence between region %d "
              "of task %s (ID %lld) and region %d of task %s (ID %lld) of "
              " type %s", src_idx, src_task->get_task_name(),
              src_task->get_unique_id(), dst_idx, 
              dst_task->get_task_name(), dst_task->get_unique_id(),
              (dtype == LEGION_TRUE_DEPENDENCE) ? "TRUE DEPENDENCE" :
                (dtype == LEGION_ANTI_DEPENDENCE) ? "ANTI DEPENDENCE" :
                "ATOMIC DEPENDENCE")
        }
        else if (dtype == LEGION_SIMULTANEOUS_DEPENDENCE)
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
    std::map<PhysicalManager*,unsigned>*
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
      assert(mapping_tracker != NULL);
#endif
      mapping_tracker->add_mapping_dependence(precondition);
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
    void MustEpochOp::register_subop(Operation *op)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      remaining_subop_completes++;
      remaining_subop_commits++;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::notify_subop_complete(Operation *op, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      bool need_complete;
      {
        AutoLock o_lock(op_lock);
        if (precondition.exists())
          completion_preconditions.insert(precondition);
        completion_effects.insert(op->get_completion_event());
#ifdef DEBUG_LEGION
        assert(remaining_subop_completes > 0);
#endif
        remaining_subop_completes--;
        need_complete = (remaining_subop_completes == 0);
      }
      if (need_complete)
      {
        ApEvent effects_done;
        if (!completion_effects.empty())
          effects_done = Runtime::merge_events(NULL, completion_effects);
        if (!request_early_complete(effects_done))
          completion_preconditions.insert(Runtime::protect_event(effects_done));
        RtEvent completion_done;
        if (!completion_preconditions.empty())
          completion_done = Runtime::merge_events(completion_preconditions);
#ifdef LEGION_SPY
        // Still need this for Legion Spy
        LegionSpy::log_operation_events(unique_op_id, 
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        complete_operation(completion_done);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::notify_subop_commit(Operation *op, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      bool need_commit;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(remaining_subop_commits > 0);
#endif
        if (precondition.exists())
          commit_preconditions.insert(precondition);
        remaining_subop_commits--;
        need_commit = (remaining_subop_commits == 0);
      }
      if (need_commit)
      {
        RtEvent commit_precondition;
        if (!commit_preconditions.empty())
          commit_precondition = Runtime::merge_events(commit_preconditions);
        commit_operation(true/*deactivate*/, commit_precondition);
      }
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
          MustEpochIndivArgs args(this, owner, indiv_tasks[idx]);
          RtEvent wait = 
            owner->runtime->issue_runtime_meta_task(args, 
                        LG_THROUGHPUT_DEFERRED_PRIORITY);
          if (wait.exists())
            wait_events.insert(wait);
        }
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        if (!index_triggered[idx])
        {
          MustEpochIndexArgs args(this, owner, index_tasks[idx]);
          RtEvent wait = 
            owner->runtime->issue_runtime_meta_task(args,
                        LG_THROUGHPUT_DEFERRED_PRIORITY);
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
      MustEpochMapArgs args(this, owner);
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
                LG_THROUGHPUT_DEFERRED_PRIORITY, precondition); 
        }
        else
          mapped_events[idx] = 
            owner->runtime->issue_runtime_meta_task(args,
                        LG_THROUGHPUT_DEFERRED_PRIORITY);
      }
      std::set<RtEvent> wait_events(mapped_events.begin(), mapped_events.end());
      if (!wait_events.empty())
      {
        RtEvent mapped_event = Runtime::merge_events(wait_events);
        mapped_event.wait();
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochMapper::map_task(SingleTask *task)
    //--------------------------------------------------------------------------
    {
      // Note we don't need to hold a lock here because this is
      // a monotonic change.  Once it fails for anyone then it
      // fails for everyone.
      RtEvent done_mapping = task->perform_mapping(owner);
      if (done_mapping.exists())
        done_mapping.wait();
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
      MustEpochDistributorArgs dist_args(owner);
      MustEpochLauncherArgs launch_args(owner);
      std::set<RtEvent> wait_events;
      for (std::vector<IndividualTask*>::const_iterator it = 
            indiv_tasks.begin(); it != indiv_tasks.end(); it++)
      {
        if (!runtime->is_local((*it)->target_proc))
        {
          dist_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(dist_args, 
                        LG_THROUGHPUT_DEFERRED_PRIORITY);
          if (wait.exists())
            wait_events.insert(wait);
        }
        else
        {
          launch_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(launch_args,
                          LG_THROUGHPUT_DEFERRED_PRIORITY);
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
                        LG_THROUGHPUT_DEFERRED_PRIORITY);
          if (wait.exists())
            wait_events.insert(wait);
        }
        else
        {
          launch_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(launch_args,
                         LG_THROUGHPUT_DEFERRED_PRIORITY);
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
    void PendingPartitionOp::initialize_equal_partition(InnerContext *ctx,
                                                        IndexPartition pid, 
                                                        size_t granularity,
                                                        const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new EqualPartitionThunk(pid, granularity);
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_weight_partition(InnerContext *ctx,
                                   IndexPartition pid, const FutureMap &weights,
                                   size_t granularity, const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new WeightPartitionThunk(pid, weights, granularity);
      // Also save this locally for analysis
      future_map = weights;
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_union_partition(InnerContext *ctx,
                                                        IndexPartition pid,
                                                        IndexPartition h1,
                                                        IndexPartition h2,
                                                        const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new UnionPartitionThunk(pid, h1, h2);
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_intersection_partition(
                                                            InnerContext *ctx,
                                                            IndexPartition pid,
                                                            IndexPartition h1,
                                                            IndexPartition h2,
                                                            const char *prov)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, prov);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new IntersectionPartitionThunk(pid, h1, h2);
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_intersection_partition(
                                                           InnerContext *ctx,
                                                           IndexPartition pid,
                                                           IndexPartition part,
                                                           const bool dominates,
                                                           const char *prov)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, prov);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new IntersectionWithRegionThunk(pid, part, dominates);
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_difference_partition(InnerContext *ctx,
                                                             IndexPartition pid,
                                                             IndexPartition h1,
                                                             IndexPartition h2,
                                                             const char *prov)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, prov);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new DifferencePartitionThunk(pid, h1, h2);
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_restricted_partition(InnerContext *ctx,
                                                          IndexPartition pid,
                                                          const void *transform,
                                                          size_t transform_size,
                                                          const void *extent,
                                                          size_t extent_size,
                                                          const char *prov)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, prov);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new RestrictedPartitionThunk(pid, transform, transform_size,
                                           extent, extent_size);
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_by_domain(InnerContext *ctx,
                                                  IndexPartition pid,
                                                  const FutureMap &fm,
                                                  bool perform_intersections,
                                                  const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new FutureMapThunk(pid, fm, perform_intersections);
      // Also save this locally for analysis
      future_map = fm;

      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_cross_product(InnerContext *ctx,
                                                      IndexPartition base,
                                                      IndexPartition source,
                                                      LegionColor part_color,
                                                      const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new CrossProductThunk(base, source, part_color);
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_union(InnerContext *ctx,
                                                          IndexSpace target,
                                         const std::vector<IndexSpace> &handles,
                                                         const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingSpace(target, true/*union*/, handles);
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_union(InnerContext *ctx,
                                                         IndexSpace target,
                                                         IndexPartition handle,
                                                         const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingSpace(target, true/*union*/, handle);
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_intersection(
                                         InnerContext *ctx,IndexSpace target,
                                         const std::vector<IndexSpace> &handles,
                                         const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingSpace(target, false/*union*/, handles);
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_intersection(
                                  InnerContext *ctx, IndexSpace target,
                                  IndexPartition handle, const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingSpace(target, false/*union*/, handle);
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_difference(
                                         InnerContext *ctx,
                                         IndexSpace target, IndexSpace initial, 
                                         const std::vector<IndexSpace> &handles,
                                         const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, provenance);
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ComputePendingDifference(target, initial, handles);
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::perform_logging()
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_pending_partition_operation(
          parent_ctx->get_unique_id(), unique_op_id, context_index);
      thunk->perform_logging(this);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if ((future_map.impl != NULL) && (future_map.impl->op != NULL))
        register_dependence(future_map.impl->op, future_map.impl->op_gen);
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
      // Mark that this is mapped right away
      complete_mapping();
      // If necessary defer execution until the future map is ready
      RtEvent future_map_ready;
      if (future_map.impl != NULL)
        future_map_ready = future_map.impl->get_ready_event();
      complete_execution(future_map_ready);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      // Perform the partitioning operation
      const ApEvent ready_event = thunk->perform(this, runtime->forest);
#ifdef LEGION_SPY
      // Still have to do this call to let Legion Spy know we're done
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      ApEvent::NO_AP_EVENT);
#endif
      complete_operation(Runtime::protect_event(ready_event));
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
      future_map = FutureMap(); // clear any references
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
    void PendingPartitionOp::WeightPartitionThunk::perform_logging(
                                                         PendingPartitionOp *op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(op->unique_op_id, pid.id,
          WEIGHT_PARTITION);
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
    void PendingPartitionOp::IntersectionWithRegionThunk::perform_logging(
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
    void PendingPartitionOp::FutureMapThunk::perform_logging(
                                                         PendingPartitionOp *op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(op->unique_op_id, pid.id,
          BY_DOMAIN_PARTITION);
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
    // External Partition
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalPartition::ExternalPartition(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ExternalPartition::pack_external_partition(Serializer &rez,
                                                    AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_region_requirement(requirement, rez);
      rez.serialize<bool>(is_index_space);
      rez.serialize(index_domain);
      rez.serialize(index_point);
      pack_mappable(*this, rez);
      rez.serialize<size_t>(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalPartition::unpack_external_partition(Deserializer &derez,
                                                      Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_region_requirement(requirement, derez);
      derez.deserialize<bool>(is_index_space);
      derez.deserialize(index_domain);
      derez.deserialize(index_point);
      unpack_mappable(*this, derez);
      size_t index;
      derez.deserialize(index);
      set_context_index(index);
    }

    /////////////////////////////////////////////////////////////
    // Dependent Partition Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DependentPartitionOp::DependentPartitionOp(Runtime *rt)
      : ExternalPartition(), CollectiveInstanceCreator<Operation>(rt), 
        thunk(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DependentPartitionOp::DependentPartitionOp(const DependentPartitionOp &rhs)
      : ExternalPartition(), CollectiveInstanceCreator<Operation>(NULL)
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
    void DependentPartitionOp::initialize_by_field(InnerContext *ctx, 
                                                   IndexPartition pid,
                                                   LogicalRegion handle, 
                                                   LogicalRegion parent,
                                                   FieldID fid,
                                                   MapperID id, MappingTagID t,
                                                   const UntypedBuffer &marg,
                                                   const char *prov)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/, 0, prov); 
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      requirement = 
        RegionRequirement(handle, LEGION_READ_ONLY, LEGION_EXCLUSIVE, parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
      mapper_data_size = marg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, marg.get_ptr(), mapper_data_size);
      }
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ByFieldThunk(pid);
      if (runtime->legion_spy_enabled)
        perform_logging();
      if (runtime->check_privileges)
      {
        const size_t field_size = 
          runtime->forest->get_field_size(handle.get_field_space(), fid);
        IndexSpace color_space = 
          runtime->forest->get_index_partition_color_space(pid);
        const size_t coord_size = 
          runtime->forest->get_coordinate_size(color_space, false/*range*/);
        if (field_size != coord_size)
          REPORT_LEGION_ERROR(ERROR_TYPE_FIELD_MISMATCH,
              "The field size for partition-by-field operation does not "
              "match the size of the coordinate type of the color space "
              "of the resulting partition. Field %d has size %zd bytes "
              "but the coordinates of color space %d of partition %d are "
              "%zd bytes for dependent partition operation (UID %lld) in "
              "parent task %s (UID %lld).", fid, field_size, 
              color_space.get_id(), pid.get_id(), coord_size, get_unique_id(),
              parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        const CustomSerdezID serdez = runtime->forest->get_field_serdez(
            handle.get_field_space(), fid);
        if (serdez != 0)
          REPORT_LEGION_ERROR(ERROR_SERDEZ_FIELD_DISALLOWED,
              "Serdez fields are not permitted to be used for any "
              "dependent partitioning calls. Field %d has serdez "
              "function %d and was passed to partition-by-field "
              "operation (UID %lld) in parent task %s (UID %lld).",
              fid, serdez, get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_image(InnerContext *ctx, 
                                                   IndexPartition pid,
                                          LogicalPartition projection,
                                          LogicalRegion parent, FieldID fid,
                                          MapperID id, MappingTagID t,
                                          const UntypedBuffer &marg,
                                          const char *prov)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/, 1/*regions*/, prov);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      LogicalRegion proj_parent = 
        runtime->forest->get_parent_logical_region(projection);
      requirement = 
        RegionRequirement(proj_parent,LEGION_READ_ONLY,LEGION_EXCLUSIVE,parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
      mapper_data_size = marg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, marg.get_ptr(), mapper_data_size);
      }
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ByImageThunk(pid, projection.get_index_partition());
      if (runtime->legion_spy_enabled)
        perform_logging();
      if (runtime->check_privileges)
      {
        const size_t field_size = 
          runtime->forest->get_field_size(projection.get_field_space(), fid);
        const IndexSpace pid_parent = 
          runtime->forest->get_parent_index_space(pid);
        const size_t coord_size = runtime->forest->get_coordinate_size(
                                            pid_parent, false/*range*/);
        if (field_size != coord_size)
          REPORT_LEGION_ERROR(ERROR_TYPE_FIELD_MISMATCH,
              "The field size for partition-by-image operation does not "
              "match the size of the coordinate types of the projection "
              "partition. Field %d has size %zd bytes but the coordinates "
              "of the projection partition %d are %zd bytes for dependent "
              "partition operation (UID %lld) in parent task %s (UID %lld).", 
              fid, field_size, pid.get_id(), coord_size, 
              get_unique_id(), parent_ctx->get_task_name(), 
              parent_ctx->get_unique_id())
        const CustomSerdezID serdez = runtime->forest->get_field_serdez(
            projection.get_field_space(), fid);
        if (serdez != 0)
          REPORT_LEGION_ERROR(ERROR_SERDEZ_FIELD_DISALLOWED,
              "Serdez fields are not permitted to be used for any "
              "dependent partitioning calls. Field %d has serdez "
              "function %d and was passed to partition-by-image "
              "operation (UID %lld) in parent task %s (UID %lld).",
              fid, serdez, get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_image_range(InnerContext *ctx, 
                                                         IndexPartition pid,
                                                LogicalPartition projection,
                                                LogicalRegion parent,
                                                FieldID fid, MapperID id,
                                                MappingTagID t,
                                                const UntypedBuffer &marg,
                                                const char *prov)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/, 1/*regions*/, prov);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      LogicalRegion proj_parent = 
        runtime->forest->get_parent_logical_region(projection);
      requirement = 
        RegionRequirement(proj_parent,LEGION_READ_ONLY,LEGION_EXCLUSIVE,parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
      mapper_data_size = marg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, marg.get_ptr(), mapper_data_size);
      }
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ByImageRangeThunk(pid, projection.get_index_partition());
      if (runtime->legion_spy_enabled)
        perform_logging();
      if (runtime->check_privileges)
      {
        const size_t field_size = 
          runtime->forest->get_field_size(projection.get_field_space(), fid);
        const IndexSpace pid_parent =
          runtime->forest->get_parent_index_space(pid);
        const size_t coord_size = runtime->forest->get_coordinate_size(
                                              pid_parent, true/*range*/);
        if (field_size != coord_size)
          REPORT_LEGION_ERROR(ERROR_TYPE_FIELD_MISMATCH,
              "The field size for partition-by-image-range operation does not "
              "match the size of the coordinate types of the projection "
              "partition. Field %d has size %zd bytes but the coordinates "
              "of the projection partition %d are %zd bytes for dependent "
              "partition operation (UID %lld) in parent task %s (UID %lld).", 
              fid, field_size, pid.get_id(), coord_size, 
              get_unique_id(), parent_ctx->get_task_name(), 
              parent_ctx->get_unique_id())
        const CustomSerdezID serdez = runtime->forest->get_field_serdez(
            projection.get_field_space(), fid);
        if (serdez != 0)
          REPORT_LEGION_ERROR(ERROR_SERDEZ_FIELD_DISALLOWED,
              "Serdez fields are not permitted to be used for any "
              "dependent partitioning calls. Field %d has serdez "
              "function %d and was passed to partition-by-image-range "
              "operation (UID %lld) in parent task %s (UID %lld).",
              fid, serdez, get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_preimage(InnerContext *ctx,
                                    IndexPartition pid, IndexPartition proj,
                                    LogicalRegion handle, LogicalRegion parent,
                                    FieldID fid, MapperID id, MappingTagID t,
                                    const UntypedBuffer &marg, const char *prov)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/, 1/*regions*/, prov);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      requirement = 
        RegionRequirement(handle, LEGION_READ_ONLY, LEGION_EXCLUSIVE, parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
      mapper_data_size = marg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, marg.get_ptr(), mapper_data_size);
      }
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ByPreimageThunk(pid, proj);
      if (runtime->legion_spy_enabled)
        perform_logging();
      if (runtime->check_privileges)
      {
        const size_t field_size = 
          runtime->forest->get_field_size(handle.get_field_space(), fid);
        const IndexSpace proj_parent = 
          runtime->forest->get_parent_index_space(proj);
        const size_t coord_size =
          runtime->forest->get_coordinate_size(proj_parent, false/*range*/);
        if (field_size != coord_size)
          REPORT_LEGION_ERROR(ERROR_TYPE_FIELD_MISMATCH,
              "The field size for partition-by-preimage operation does not "
              "match the size of the coordinate types of the projection "
              "partition. Field %d has size %zd bytes but the coordinates "
              "of the projection partition %d are %zd bytes for dependent "
              "partition operation (UID %lld) in parent task %s (UID %lld).",
              fid, field_size, proj.get_id(), coord_size, 
              get_unique_id(), parent_ctx->get_task_name(), 
              parent_ctx->get_unique_id())
        const CustomSerdezID serdez = runtime->forest->get_field_serdez(
            handle.get_field_space(), fid);
        if (serdez != 0)
          REPORT_LEGION_ERROR(ERROR_SERDEZ_FIELD_DISALLOWED,
              "Serdez fields are not permitted to be used for any "
              "dependent partitioning calls. Field %d has serdez "
              "function %d and was passed to partition-by-preimage "
              "operation (UID %lld) in parent task %s (UID %lld).",
              fid, serdez, get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_preimage_range(InnerContext *ctx,
                                    IndexPartition pid, IndexPartition proj,
                                    LogicalRegion handle, LogicalRegion parent,
                                    FieldID fid, MapperID id, MappingTagID t,
                                    const UntypedBuffer &marg, const char *prov)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/, 1/*regions*/, prov);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      requirement = 
        RegionRequirement(handle, LEGION_READ_ONLY, LEGION_EXCLUSIVE, parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
      mapper_data_size = marg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, marg.get_ptr(), mapper_data_size);
      }
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ByPreimageRangeThunk(pid, proj);
      if (runtime->legion_spy_enabled)
        perform_logging();
      if (runtime->check_privileges)
      {
        const size_t field_size = 
          runtime->forest->get_field_size(handle.get_field_space(), fid);
        const IndexSpace proj_parent = 
          runtime->forest->get_parent_index_space(proj);
        const size_t coord_size = 
          runtime->forest->get_coordinate_size(proj_parent, true/*range*/);
        if (field_size != coord_size)
          REPORT_LEGION_ERROR(ERROR_TYPE_FIELD_MISMATCH,
              "The field size for partition-by-preimage-range operation does "
              "not match the size of the coordinate types of the projection "
              "partition. Field %d has size %zd bytes but the coordinates "
              "of the projection partition %d are %zd bytes for dependent "
              "partition operation (UID %lld) in parent task %s (UID %lld).",
              fid, field_size, proj.get_id(), coord_size, 
              get_unique_id(), parent_ctx->get_task_name(), 
              parent_ctx->get_unique_id())
        const CustomSerdezID serdez = runtime->forest->get_field_serdez(
            handle.get_field_space(), fid);
        if (serdez != 0)
          REPORT_LEGION_ERROR(ERROR_SERDEZ_FIELD_DISALLOWED,
              "Serdez fields are not permitted to be used for any "
              "dependent partitioning calls. Field %d has serdez "
              "function %d and was passed to partition-by-preimage-range "
              "operation (UID %lld) in parent task %s (UID %lld).",
              fid, serdez, get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_association(InnerContext *ctx,
                        LogicalRegion domain, LogicalRegion domain_parent, 
                        FieldID fid, IndexSpace range, MapperID id,
                        MappingTagID t, const UntypedBuffer &marg, 
                        const char *prov)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/, 1/*regions*/, prov);
      // start-off with non-projection requirement
      requirement = RegionRequirement(domain, LEGION_READ_WRITE, 
                                      LEGION_EXCLUSIVE, domain_parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
      mapper_data_size = marg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, marg.get_ptr(), mapper_data_size);
      }
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new AssociationThunk(domain.get_index_space(), range);
      if (runtime->legion_spy_enabled)
        perform_logging();
      if (runtime->check_privileges)
      {
        const size_t field_size =
          runtime->forest->get_field_size(domain.get_field_space(), fid);
        const size_t coord_size = 
          runtime->forest->get_coordinate_size(range, false/*range*/);
        if (field_size != coord_size)
          REPORT_LEGION_ERROR(ERROR_TYPE_FIELD_MISMATCH,
              "The field size for create-by-association operation does not "
              "match the size of the range index space. Field %d has size "
              "%zd bytes but the coordinates of the range index space %d "
              "are %zd bytes for create-by-association operation (UID %lld) "
              "in parent task %s (UID %lld).", fid, field_size,
              range.get_id(), coord_size, get_unique_id(),
              parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        const CustomSerdezID serdez = runtime->forest->get_field_serdez(
            domain.get_field_space(), fid);
        if (serdez != 0)
          REPORT_LEGION_ERROR(ERROR_SERDEZ_FIELD_DISALLOWED,
              "Serdez fields are not permitted to be used for any "
              "dependent partitioning calls. Field %d has serdez "
              "function %d and was passed to create-by-association "
              "operation (UID %lld) in parent task %s (UID %lld).",
              fid, serdez, get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::perform_logging(void) const
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_dependent_partition_operation(
          parent_ctx->get_unique_id(), unique_op_id, 
          thunk->get_partition().get_id(), thunk->get_kind(), context_index);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::log_requirement(void) const
    //--------------------------------------------------------------------------
    {
      if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
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
        runtime->forest->log_launch_space(launch_space->handle, unique_op_id);
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
      // If we're an index space op, promote a singular region requirement
      // up to a projection region requirement for accuracy
      if (is_index_space && 
          (requirement.handle_type == LEGION_SINGULAR_PROJECTION))
      {
        requirement.handle_type = LEGION_REGION_PROJECTION;
        requirement.projection = 0;
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->check_privileges)
        check_privilege();
      // Before doing the dependence analysis we have to ask the
      // mapper whether it would like to make this an index space
      // operation or a single operation
      select_partition_projection();
      // Do thise now that we've picked our region requirement
      initialize_privilege_path(privilege_path, requirement);
      if (runtime->legion_spy_enabled)
        log_requirement();
      ProjectionInfo projection_info;
      if (is_index_space)
        projection_info = ProjectionInfo(runtime, requirement, launch_space); 
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions);
      // Record this dependent partition op with the context so that it 
      // can track implicit dependences on it for later operations
      parent_ctx->update_current_implicit(this);
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
      // We also allow the mapper to pick the same projection partition 
      // if the partition operation is an image or image-range
      if (!runtime->unsafe_mapper && !partition_node->is_complete(false) &&
            !thunk->safe_projection(partition_node->handle))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of "
                      "'select_partition_projection' on mapper %s."
                      "Mapper selected a logical partition that is "
                      "not complete for dependent partitioning operation "
                      "in task %s (UID %lld).", mapper->get_mapper_name(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id())
      // Update the region requirement and other information
      requirement.partition = output.chosen_partition;
      requirement.handle_type = LEGION_PARTITION_PROJECTION;
      requirement.projection = 0; // always default
      launch_space = partition_node->color_space;
      add_launch_space_reference(launch_space);
      index_domain = partition_node->color_space->get_color_space_domain();
      is_index_space = true;
#ifdef LEGION_SPY
      intermediate_index_event = Runtime::create_ap_user_event(NULL);
#endif
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // See if this is an index space operation
      if (is_index_space)
      {
#ifdef DEBUG_LEGION
        assert(requirement.handle_type == LEGION_PARTITION_PROJECTION);
#endif
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
                                 runtime, projection_points, launch_space);
        // No need to check the validity of the points, we know they are good
        if (runtime->legion_spy_enabled)
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
          (*it)->launch();
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
        std::set<RtEvent> preconditions;
        // Path for a non-index space implementation
        runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                     requirement,
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
      assert(requirement.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      const PhysicalTraceInfo trace_info(this, 0/*index*/, true/*init*/);
      // Perform the mapping call to get the physical isntances 
      InstanceSet mapped_instances;
      const bool record_valid = invoke_mapper(mapped_instances);
      if (runtime->legion_spy_enabled)
        runtime->forest->log_mapping_decision(unique_op_id, parent_ctx, 
                                              0/*idx*/, requirement,
                                              mapped_instances);
#ifdef DEBUG_LEGION
      assert(!mapped_instances.empty()); 
#endif
      // Then we can register our mapped_instances
      runtime->forest->physical_perform_updates_and_registration(
                                              requirement, version_info,
                                              this, 0/*idx*/,
                                              ApEvent::NO_AP_EVENT,
                                              completion_event,
                                              mapped_instances,
                                              trace_info,
                                              map_applied_conditions,
#ifdef DEBUG_LEGION
                                              get_logging_name(),
                                              unique_op_id,
#endif
                                              record_valid);
      ApEvent done_event = trigger_thunk(requirement.region.get_index_space(),
                                     mapped_instances, trace_info, index_point);
      // Once we are done running these routines, we can mark
      // that the handles have all been completed
      RtEvent mapping_applied;
      if (!map_applied_conditions.empty())
        mapping_applied = Runtime::merge_events(map_applied_conditions);
      if (!acquired_instances.empty())
        mapping_applied = release_nonempty_acquired_instances(mapping_applied, 
                                                          acquired_instances);
      complete_mapping(mapping_applied);
#ifdef LEGION_SPY
      if (runtime->legion_spy_enabled)
        LegionSpy::log_operation_events(unique_op_id, done_event,
                                        completion_event);
#endif
      if (!request_early_complete(done_event))
        complete_execution(Runtime::protect_event(done_event));
      else
        complete_execution();
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::trigger_thunk(IndexSpace handle,
                                                const InstanceSet &mapped_insts,
                                                const PhysicalTraceInfo &info, 
                                                const DomainPoint &key)
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
          InstanceManager *manager = ref.get_manager();
          desc.index_space = handle;
          desc.inst = manager->get_instance(key);
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
              Runtime::merge_events(&info, index_preconditions), instances);
#ifdef LEGION_SPY
          Runtime::trigger_event(NULL, intermediate_index_event, done_event);
#endif
          if (!request_early_complete(done_event))
            complete_execution(Runtime::protect_event(done_event));
          else
            complete_execution();
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
        InstanceManager *manager = ref.get_manager();
        desc.index_space = handle;
        desc.inst = manager->get_instance(key);
        desc.field_offset = manager->layout->find_field_info(
                      *(requirement.privilege_fields.begin())).field_id;
        ApEvent ready_event = ref.get_ready_event();
        return thunk->perform(this, runtime->forest, 
                              ready_event, instances);
      }
    }

    //--------------------------------------------------------------------------
    bool DependentPartitionOp::invoke_mapper(InstanceSet &mapped_instances)
    //--------------------------------------------------------------------------
    {
      Mapper::MapPartitionInput input;
      Mapper::MapPartitionOutput output;
      output.profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      output.track_valid_region = true;
      // Invoke the mapper
      if (mapper == NULL)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      if (mapper->request_valid_instances)
      {
        InstanceSet valid_instances;
        runtime->forest->physical_premap_region(this, 0/*idx*/, requirement,
                      version_info, valid_instances, map_applied_conditions);
        prepare_for_mapping(valid_instances, input.valid_instances);
      }
      mapper->invoke_map_partition(this, &input, &output);
      if (!output.profiling_requests.empty())
      {
        filter_copy_request_kinds(mapper,
            output.profiling_requests.requested_measurements,
            profiling_requests, true/*warn*/);
        profiling_priority = output.profiling_priority;
#ifdef DEBUG_LEGION
        assert(!profiling_reported.exists());
#endif
        profiling_reported = Runtime::create_rt_user_event();
      }
      // Now we have to validate the output
      // Go through the instances and make sure we got one for every field
      // Also check to make sure that none of them are composite instances
      RegionTreeID bad_tree = 0;
      std::vector<FieldID> missing_fields;
      std::vector<PhysicalManager*> unacquired;
      int virtual_index = runtime->forest->physical_convert_mapping(this,
                                requirement, output.chosen_instances, 
                                mapped_instances, bad_tree, missing_fields,
                                &acquired_instances, unacquired, 
                                !runtime->unsafe_mapper);
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
               requirement.region.get_field_space(), *it, 
               LEGION_NAME_SEMANTIC_TAG, name, name_size, true, false))
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
      if (virtual_index >= 0)
      {
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'map_partition'"
                      " on mapper %s. Mapper requested creation of a composite "
                      "instance for partition mapping in task %s (ID %lld).",
                      mapper->get_mapper_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
      } 
      // If we are doing unsafe mapping, then we can return
      if (runtime->unsafe_mapper)
        return output.track_valid_region;
      // Iterate over the instances and make sure they are all valid
      // for the given logical region which we are mapping
      std::vector<LogicalRegion> regions_to_check(1, requirement.region);
      for (unsigned idx = 0; idx < mapped_instances.size(); idx++)
      {
        PhysicalManager *manager = mapped_instances[idx].get_physical_manager();
        if (!manager->meets_regions(regions_to_check))
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of "
                        "'map_partition' on mapper %s. Mapper specified an "
                        "instance that does not meet the logical region "
                        "requirement. The dependent partition operation was "
                        "issued in task %s (ID %lld).", 
                        mapper->get_mapper_name(),
                        parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
        if (manager->is_reduction_manager())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of "
                        "'map_partition' on mapper %s. Mapper selected an "
                        "illegal specialized reduction instance for dependent "
                        "partition operation in task %s (ID %lld).",
                        mapper->get_mapper_name(),parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
        if (manager->is_collective_manager())
        {
          CollectiveManager *collective_manager = 
            manager->as_collective_manager();
          if (!collective_manager->point_space->contains_point(index_point))
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of "
                        "'map_partition' on mapper %s. Mapper selected a "
                        "collective instance for region requirement %d but "
                        "point partition (ID %lld) launched in task %s "
                        "(ID %lld) is not contained within the point space "
                        "for the collective instace.",
                        mapper->get_mapper_name(), idx, 
                        get_unique_op_id(), parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
        }
        // This is a temporary check to guarantee that instances for 
        // dependent partitioning operations are in memories that 
        // Realm supports for now. In the future this should be fixed
        // so that realm supports all kinds of memories for dependent
        // partitioning operations (see issue #516)
        const Memory::Kind mem_kind = 
          manager->layout->constraints->memory_constraint.get_kind();
        if ((mem_kind != Memory::GLOBAL_MEM) && 
            (mem_kind != Memory::SYSTEM_MEM) &&
            (mem_kind != Memory::REGDMA_MEM) && 
            (mem_kind != Memory::SOCKET_MEM) &&
            (mem_kind != Memory::Z_COPY_MEM))
        {
          const char *mem_names[] = {
#define MEM_NAMES(name, desc) desc,
            REALM_MEMORY_KINDS(MEM_NAMES) 
#undef MEM_NAMES
          };
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of "
                        "'map_partition' on mapper %s for dependent partition "
                        "operation %lld in task %s (UID %lld). Mapper specified"
                        " an instance in memory(memories) with kind %s which is"
                        " not supported for dependent partition operations "
                        "currently (see Legion issue #516). Please pick an "
                        "instance in a CPU-visible memory for now.",
                        mapper->get_mapper_name(), get_unique_op_id(),
                        parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id(), 
                        mem_names[mem_kind])
        }
      }
      return output.track_valid_region;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      if (profiling_reported.exists())
        finalize_partition_profiling();
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
    void DependentPartitionOp::finalize_partition_profiling(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(profiling_reported.exists());
#endif
      if (outstanding_profiling_requests.load() > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapped_event.has_triggered());
#endif
        std::vector<PartitionProfilingInfo> to_perform;
        {
          AutoLock o_lock(op_lock);
          to_perform.swap(profiling_info);
        }
        if (!to_perform.empty())
        {
          for (unsigned idx = 0; idx < to_perform.size(); idx++)
          {
            PartitionProfilingInfo &info = to_perform[idx];
            const Realm::ProfilingResponse resp(info.buffer, info.buffer_size);
            info.total_reports = outstanding_profiling_requests.load();
            info.profiling_responses.attach_realm_profiling_response(resp);
            mapper->invoke_partition_report_profiling(this, &info);
            free(info.buffer);
          }
          const int count = to_perform.size() +
              outstanding_profiling_reported.fetch_add(to_perform.size());
#ifdef DEBUG_LEGION
          assert(count <= outstanding_profiling_requests.load());
#endif
          if (count == outstanding_profiling_requests.load())
            Runtime::trigger_event(profiling_reported);
        }
      }
      else
      {
        // We're not expecting any profiling callbacks so we need to
        // do one ourself to inform the mapper that there won't be any
        Mapping::Mapper::PartitionProfilingInfo info;
        info.total_reports = 0;
        info.fill_response = false; // make valgrind happy
        mapper->invoke_partition_report_profiling(this, &info);    
        Runtime::trigger_event(profiling_reported);
      }
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
    size_t DependentPartitionOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int DependentPartitionOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* DependentPartitionOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    Mappable* DependentPartitionOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
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
      launch_space = NULL;
      index_domain = Domain::NO_DOMAIN;
      parent_req_index = 0;
      mapper = NULL;
      points_committed = 0;
      commit_request = false;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_dependent_op(); 
      if (remove_launch_space_reference(launch_space))
        delete launch_space;
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
      version_info.clear();
      map_applied_conditions.clear();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      // We deactivate all of our point operations
      for (std::vector<PointDepPartOp*>::const_iterator it = 
            points.begin(); it != points.end(); it++)
        (*it)->deactivate();
      points.clear();
      instances.clear();
      index_preconditions.clear();
      commit_preconditions.clear();
      profiling_requests.clear();
      if (!profiling_info.empty())
      {
        for (unsigned idx = 0; idx < profiling_info.size(); idx++)
          free(profiling_info[idx].buffer);
        profiling_info.clear();
      }
      if (mapper_data != NULL)
      {
        free(mapper_data);
        mapper_data = NULL;
        mapper_data_size = 0;
      }
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
    void DependentPartitionOp::select_sources(const unsigned index,
                                              const InstanceRef &target,
                                              const InstanceSet &sources,
                                              std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index == 0);
#endif
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
      compute_ranking(mapper, output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,unsigned>* 
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
    void DependentPartitionOp::add_copy_profiling_request(
                                           const PhysicalTraceInfo &info,
                                           Realm::ProfilingRequestSet &requests,
                                           bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest &request = requests.add_request( 
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
          &response, sizeof(response), profiling_priority);
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      handle_profiling_update(count);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::handle_profiling_response(
                                       const ProfilingResponseBase *base,
                                       const Realm::ProfilingResponse &response,
                                       const void *orig, size_t orig_length)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapper != NULL);
#endif
      const OpProfilingResponse *op_info = 
        static_cast<const OpProfilingResponse*>(base);
      // Check to see if we are done mapping, if not then we need to defer
      // this until we are done mapping so we know how many reports to expect
      if (!mapped_event.has_triggered())
      {
        // Take the lock and see if we lost the race
        AutoLock o_lock(op_lock);
        if (!mapped_event.has_triggered())
        {
          // Save this profiling response for later until we know the
          // full count of profiling responses
          profiling_info.resize(profiling_info.size() + 1);
          PartitionProfilingInfo &info = profiling_info.back();
          info.fill_response = op_info->fill;
          info.buffer_size = orig_length;
          info.buffer = malloc(orig_length);
          memcpy(info.buffer, orig, orig_length);
          return;
        }
      }
      // If we get here then we can handle the response now
      Mapping::Mapper::PartitionProfilingInfo info; 
      info.profiling_responses.attach_realm_profiling_response(response);
      info.total_reports = outstanding_profiling_requests.load();
      info.fill_response = op_info->fill;
      mapper->invoke_partition_report_profiling(this, &info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(count <= outstanding_profiling_requests.load());
#endif
      if (count == outstanding_profiling_requests.load())
        Runtime::trigger_event(profiling_reported);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count > 0);
      assert(!mapped_event.has_triggered());
#endif
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::pack_remote_operation(Serializer &rez, 
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_partition(rez, target);
      rez.serialize<PartitionKind>(get_partition_kind());
      rez.serialize<size_t>(profiling_requests.size());
      if (!profiling_requests.empty())
      {
        for (unsigned idx = 0; idx < profiling_requests.size(); idx++)
          rez.serialize(profiling_requests[idx]);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Create a user event for this response
        const RtUserEvent response = Runtime::create_rt_user_event();
        rez.serialize(response);
        applied_events.insert(response);
      }
    }

    //--------------------------------------------------------------------------
    CollectiveManager* DependentPartitionOp::find_or_create_collective_instance(
                                  MappingCallKind call_kind, unsigned index,
                                  const LayoutConstraintSet &constraints,
                                  const std::vector<LogicalRegion> &regions,
                                  Memory::Kind kind, size_t *footprint,
                                  LayoutConstraintKind *unsat_kind,
                                  unsigned *unsat_index, 
                                  DomainPoint &collective_point)
    //--------------------------------------------------------------------------
    {
      if (points.empty())
        return Operation::find_or_create_collective_instance(call_kind, index,
                            constraints, regions, kind, footprint, unsat_kind, 
                            unsat_index, collective_point);
      else
        return 
          CollectiveInstanceCreator<Operation>::
                find_or_create_collective_instance(
                          call_kind, index, constraints, regions, kind,
                          footprint, unsat_kind, unsat_index, collective_point);
    }

    //--------------------------------------------------------------------------
    bool DependentPartitionOp::finalize_collective_instance(
                        MappingCallKind call_kind, unsigned index, bool success)
    //--------------------------------------------------------------------------
    {
      if (points.empty())
        return Operation::finalize_collective_instance(call_kind,index,success);
      else
        return 
          CollectiveInstanceCreator<Operation>::finalize_collective_instance(
                                                    call_kind, index, success);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::report_total_collective_instance_calls(
                              MappingCallKind mapper_call, unsigned total_calls)
    //--------------------------------------------------------------------------
    {
      if (points.empty())
        Operation::report_total_collective_instance_calls(mapper_call, 
                                                          total_calls);
      else
        CollectiveInstanceCreator<Operation>::
          report_total_collective_instance_calls(mapper_call, total_calls);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::check_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field = LEGION_AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, then check the privileges with the parent context
      if (et == LEGION_NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, bad_index);
      switch (et)
      {
        case LEGION_NO_ERROR:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            REPORT_LEGION_ERROR(ERROR_REQUIREMENTS_INVALID_REGION,
                             "Requirements for invalid region handle "
                             "(%x,%d,%d) for dependent partitioning op "
                             "(ID %lld)",
                             requirement.region.index_space.id,
                             requirement.region.field_space.id,
                             requirement.region.tree_id,
                             unique_op_id);
            break;
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = 
              (requirement.handle_type == LEGION_SINGULAR_PROJECTION) ||
              (requirement.handle_type == LEGION_REGION_PROJECTION)
            ? requirement.region.field_space :
            requirement.partition.field_space;
            REPORT_LEGION_ERROR(ERROR_FIELD_NOT_VALID_FIELD,
                            "Field %d is not a valid field of field "
                             "space %d for dependent partitioning op "
                             "(ID %lld)", bad_field, sp.id, unique_op_id)
            break;
          }
        case ERROR_INVALID_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_PRIVILEGE,
                             "Instance field %d is not one of the "
                             "privilege fields for dependent partitioning "
                             "op (ID %lld)",
                             bad_field, unique_op_id)
            break;
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_PRIVILEGE,
                             "Instance field %d is a duplicate for "
                             "dependent partitioning op (ID %lld)",
                             bad_field, unique_op_id)
            break;
          }
        case ERROR_BAD_PARENT_REGION:
          {
            if (bad_index < 0) 
            {
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_INLINE,
                               "Parent task %s (ID %lld) of dependent "
                               "partitioning op "
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
            else if (bad_field == LEGION_AUTO_GENERATE_ID) 
            {
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_INLINE,
                               "Parent task %s (ID %lld) of dependent "
                               "partitioning op "
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
                               "Parent task %s (ID %lld) of dependent "
                               "partitioning op "
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
                             "(%x,%x,%x) for region requirement of "
                             "dependent partitioning op (ID %lld)",
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
                             "Region requirement of dependent partitioning "
                             "op (ID %lld) cannot find privileges for field "
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
                             "requirement of dependent partitioning op "
                             "(ID %lld)", requirement.privilege,
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
      initialize_operation(own->get_context(), false/*track*/, 1/*size*/,
          own->get_provenance());
      index_point = p;
      owner = own;
      context_index = owner->get_ctx_index();
      index_domain = owner->index_domain;
      requirement = owner->requirement;
      parent_task = owner->parent_task;
      map_id      = owner->map_id;
      tag         = owner->tag;
      mapper_data_size = owner->mapper_data_size;
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, owner->mapper_data, mapper_data_size);
      }
      parent_req_index = owner->parent_req_index;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_index_point(own->get_unique_op_id(), unique_op_id, p);
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::launch(void)
    //--------------------------------------------------------------------------
    {
      // Perform the version analysis for our point
      std::set<RtEvent> preconditions;
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                        requirement, version_info, preconditions);
      // We can also mark this as having our resolved any predication
      resolve_speculation();
      // Then put ourselves in the queue of operations ready to map
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_dependent_op();
      // Reset this to true after it was cleared by the base call
      is_index_space = true;
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
                                          const InstanceSet &mapped_instances,
                                          const PhysicalTraceInfo &trace_info,
                                          const DomainPoint &key)
    //--------------------------------------------------------------------------
    {
      return owner->trigger_thunk(handle, mapped_instances, 
                                  trace_info, index_point);
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      if (profiling_reported.exists())
        finalize_partition_profiling();
      // Don't commit this operation until we've reported our profiling
      // Out index owner will deactivate the operation
      commit_operation(false/*deactivate*/, profiling_reported);
      // Tell our owner that we are done, they will do the deactivate
      owner->handle_point_commit(profiling_reported);
    }

    //--------------------------------------------------------------------------
    CollectiveManager* PointDepPartOp::find_or_create_collective_instance(
                                  MappingCallKind mapper_call, unsigned index,
                                  const LayoutConstraintSet &constraints,
                                  const std::vector<LogicalRegion> &regions,
                                  Memory::Kind kind, size_t *footprint,
                                  LayoutConstraintKind *unsat_kind,
                                  unsigned *unsat_index,
                                  DomainPoint &collective_point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index_point.get_dim() > 0);
#endif
      collective_point = index_point;
      return owner->find_or_create_collective_instance(mapper_call, index,
          constraints, regions, kind, footprint, unsat_kind, unsat_index,
          collective_point);
    }

    //--------------------------------------------------------------------------
    bool PointDepPartOp::finalize_collective_instance(MappingCallKind call_kind,
                                                   unsigned index, bool success)
    //--------------------------------------------------------------------------
    {
      return owner->finalize_collective_instance(call_kind, index, success);
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::report_total_collective_instance_calls(
                              MappingCallKind mapper_call, unsigned total_calls)
    //--------------------------------------------------------------------------
    {
      owner->report_total_collective_instance_calls(mapper_call, total_calls);
    }

    //--------------------------------------------------------------------------
    Partition::PartitionKind PointDepPartOp::get_partition_kind(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(owner != NULL);
#endif
      return owner->get_partition_kind();
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
      assert(requirement.handle_type == LEGION_PARTITION_PROJECTION);
#endif
      requirement.region = result;
      requirement.handle_type = LEGION_SINGULAR_PROJECTION;
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::record_intra_space_dependences(unsigned index,
                                    const std::vector<DomainPoint> &dependences)
    //--------------------------------------------------------------------------
    {
      // Should never get here because our requirements are always read-only
      assert(false);
    }

    /////////////////////////////////////////////////////////////
    // External Fill
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalFill::ExternalFill(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ExternalFill::pack_external_fill(Serializer &rez,
                                          AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_region_requirement(requirement, rez);
      rez.serialize(grants.size());
      for (unsigned idx = 0; idx < grants.size(); idx++)
        pack_grant(grants[idx], rez);
      rez.serialize(wait_barriers.size());
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        pack_phase_barrier(wait_barriers[idx], rez);
      rez.serialize(arrive_barriers.size());
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        pack_phase_barrier(arrive_barriers[idx], rez);
      rez.serialize<bool>(is_index_space);
      rez.serialize(index_domain);
      rez.serialize(index_point);
      pack_mappable(*this, rez);
      rez.serialize<size_t>(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalFill::unpack_external_fill(Deserializer &derez,
                                            Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_region_requirement(requirement, derez);
      size_t num_grants;
      derez.deserialize(num_grants);
      grants.resize(num_grants);
      for (unsigned idx = 0; idx < grants.size(); idx++)
        unpack_grant(grants[idx], derez);
      size_t num_wait_barriers;
      derez.deserialize(num_wait_barriers);
      wait_barriers.resize(num_wait_barriers);
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        unpack_phase_barrier(wait_barriers[idx], derez);
      size_t num_arrive_barriers;
      derez.deserialize(num_arrive_barriers);
      arrive_barriers.resize(num_arrive_barriers);
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        unpack_phase_barrier(arrive_barriers[idx], derez);
      derez.deserialize<bool>(is_index_space);
      derez.deserialize(index_domain);
      derez.deserialize(index_point);
      unpack_mappable(*this, derez);
      size_t index;
      derez.deserialize(index);
      set_context_index(index);
    }

    ///////////////////////////////////////////////////////////// 
    // Fill Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FillOp::FillOp(Runtime *rt)
      : MemoizableOp<SpeculativeOp>(rt), ExternalFill()
    //--------------------------------------------------------------------------
    {
      this->is_index_space = false;
    }

    //--------------------------------------------------------------------------
    FillOp::FillOp(const FillOp &rhs)
      : MemoizableOp<SpeculativeOp>(NULL), ExternalFill()
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
    void FillOp::initialize(InnerContext *ctx, const FillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      parent_task = ctx->get_task();
      initialize_speculation(ctx, true/*track*/, 1, launcher.static_dependences,
                             launcher.predicate, launcher.provenance.c_str());
      initialize_memoizable();
      requirement = RegionRequirement(launcher.handle, LEGION_WRITE_DISCARD,
                                      LEGION_EXCLUSIVE, launcher.parent);
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
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_fill_operation(parent_ctx->get_unique_id(), 
                                      unique_op_id, context_index);
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
      activate_memoizable();
      value = NULL;
      value_size = 0;
      fill_view = NULL;
      true_guard = PredEvent::NO_PRED_EVENT;
      false_guard = PredEvent::NO_PRED_EVENT;
    }

    //--------------------------------------------------------------------------
    void FillOp::deactivate_fill(void)
    //--------------------------------------------------------------------------
    {
      deactivate_speculative();
      privilege_path.clear();
      version_info.clear();
      if (value != NULL) 
      {
        free(value);
        value = NULL;
      }
      future = Future();
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
    size_t FillOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index; 
    }

    //--------------------------------------------------------------------------
    void FillOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int FillOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* FillOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,unsigned>*
                                        FillOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      // Fill Ops should never actually need this, but this method might
      // be called in the process of doing a mapper call
      return NULL;
    }

    //--------------------------------------------------------------------------
    void FillOp::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &reqeusts, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do for the moment
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
      if (runtime->legion_spy_enabled)
        log_fill_requirement();
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_dependence_analysis(void) 
    //--------------------------------------------------------------------------
    {
      if (runtime->check_privileges)
        check_fill_privilege();
      // Register a dependence on our predicate
      register_predicate_dependence();
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(this, 
                              wait_barriers, arrive_barriers);
      // If we are waiting on a future register a dependence
      if (future.impl != NULL)
        future.impl->register_dependence(this);
      ProjectionInfo projection_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions);
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
      if (is_replaying())
      {
        enqueue_ready_operation();
        return;
      }

      std::set<RtEvent> preconditions;
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement,
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
      const PhysicalTraceInfo trace_info(this, 0/*index*/, true/*initialize*/);
      // Tell the region tree forest to fill in this field
      // Note that the forest takes ownership of the value buffer
      if (future.impl == NULL)
      {
#ifdef DEBUG_LEGION
        assert(value != NULL);
        assert(fill_view == NULL);
#endif
        // This is NULL for now until we implement tracing for fills
        ApEvent init_precondition = compute_init_precondition(trace_info);
        // Ask the enclosing context to find or create the fill view
#ifdef DEBUG_LEGION
        InnerContext *context = dynamic_cast<InnerContext*>(parent_ctx);
        assert(context != NULL);
#else
        InnerContext *context = static_cast<InnerContext*>(parent_ctx);
#endif
        // This comes back with a reference to prevent collection
        // that we need to free after we have been mapped
        bool lost_ownership = true;
        fill_view = context->find_or_create_fill_view(this, 
            map_applied_conditions, value, value_size, lost_ownership);
        // If we lost ownership of the buffer to the context then 
        // we no longer need to free it
        if (lost_ownership)
        {
          value = NULL;
          value_size = 0;
        }
        ApEvent done_event = 
          runtime->forest->fill_fields(this, requirement, 0/*idx*/, 
                                       fill_view, version_info, 
                                       init_precondition, true_guard,
                                       trace_info, map_applied_conditions);
        if (runtime->legion_spy_enabled)
        {
#ifdef LEGION_SPY
          LegionSpy::log_operation_events(unique_op_id, done_event, 
                                          completion_event);
#endif
        }
#ifdef DEBUG_LEGION
        dump_physical_state(&requirement, 0);
#endif
        
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
            if (runtime->legion_spy_enabled)
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
        if (!future_ready_event.has_triggered_faultignorant())
        {
          const RtEvent safe = Runtime::protect_event(future_ready_event);
          if (safe.exists() && !safe.has_triggered())
            parent_ctx->add_to_trigger_execution_queue(this, safe);
          else
            trigger_execution();
        }
        else
          trigger_execution(); // can do the completion now
      }
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(fill_view == NULL);
#endif
      const PhysicalTraceInfo trace_info(this, 0/*index*/, false/*init*/);
      // Make a copy of the future value since the region tree
      // will want to take ownership of the buffer
      size_t result_size = future.impl->get_untyped_size();
      void *result = malloc(result_size);
      memcpy(result, future.impl->get_untyped_result(), result_size);
      // This is NULL for now until we implement tracing for fills
      ApEvent init_precondition = compute_init_precondition(trace_info);
      // Ask the enclosing context to find or create the fill view
#ifdef DEBUG_LEGION
      InnerContext *context = dynamic_cast<InnerContext*>(parent_ctx);
      assert(context != NULL);
#else
      InnerContext *context = static_cast<InnerContext*>(parent_ctx);
#endif
      // This comes back with a reference to prevent collection
      // that we need to free after we have been mapped
      bool lost_ownership = false;
      fill_view = context->find_or_create_fill_view(this,
          map_applied_conditions, result, result_size, lost_ownership);
      if (!lost_ownership)
        free(result);
      ApEvent done_event = 
          runtime->forest->fill_fields(this, requirement, 0/*idx*/, 
                                       fill_view, version_info,
                                       init_precondition, true_guard,
                                       trace_info, map_applied_conditions);
#ifdef LEGION_SPY
      LegionSpy::log_operation_events(unique_op_id, done_event,
                                      completion_event);
#endif
#ifdef DEBUG_LEGION
      dump_physical_state(&requirement, 0);
#endif
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
          if (runtime->legion_spy_enabled)
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
    void FillOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      // See if we have a fill view on which we can remove a reference
      if (fill_view != NULL)
      {
        if (fill_view->remove_base_valid_ref(MAPPING_ACQUIRE_REF))
          delete fill_view;
        fill_view = NULL;
      }
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void FillOp::check_fill_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field = LEGION_AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      if (et == LEGION_NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, bad_index);
      switch (et)
      {
        case LEGION_NO_ERROR:
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
            FieldSpace sp = 
              (requirement.handle_type == LEGION_SINGULAR_PROJECTION) ||
              (requirement.handle_type == LEGION_REGION_PROJECTION)
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
            else if (bad_field == LEGION_AUTO_GENERATE_ID) 
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
    ApEvent FillOp::compute_sync_precondition(const TraceInfo *info) const
    //--------------------------------------------------------------------------
    {
      ApEvent result;
      if (!wait_barriers.empty() || !grants.empty())
      {
        std::set<ApEvent> sync_preconditions;
        if (!wait_barriers.empty())
        {
          for (std::vector<PhaseBarrier>::const_iterator it = 
                wait_barriers.begin(); it != wait_barriers.end(); it++)
          {
            ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
            sync_preconditions.insert(e);
            if (runtime->legion_spy_enabled)
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
        // For some reason we don't trace these, not sure why
        result = Runtime::merge_events(NULL, sync_preconditions);
      }
      if ((info != NULL) && info->recording)
        info->record_op_sync_event(result);
      return result;
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->legion_spy_enabled && !need_prepipeline_stage)
        log_fill_requirement();
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      tpl->register_operation(this);
      complete_mapping();
      complete_execution();
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    void FillOp::pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                       std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_fill(rez, target);
    }

    ///////////////////////////////////////////////////////////// 
    // Index Fill Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexFillOp::IndexFillOp(Runtime *rt)
      : CollectiveInstanceCreator<FillOp>(rt)
    //--------------------------------------------------------------------------
    {
      this->is_index_space = true;
    }

    //--------------------------------------------------------------------------
    IndexFillOp::IndexFillOp(const IndexFillOp &rhs)
      : CollectiveInstanceCreator<FillOp>(rhs)
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
    void IndexFillOp::initialize(InnerContext *ctx,
                                 const IndexFillLauncher &launcher,
                                 IndexSpace launch_sp)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      parent_task = ctx->get_task();
      initialize_speculation(ctx, true/*track*/, 1, launcher.static_dependences,
                             launcher.predicate, launcher.provenance.c_str());
#ifdef DEBUG_LEGION
      assert(launch_sp.exists());
#endif
      launch_space = runtime->forest->get_node(launch_sp);
      add_launch_space_reference(launch_space);
      if (!launcher.launch_domain.exists())
        launch_space->get_launch_space_domain(index_domain);
      else
        index_domain = launcher.launch_domain;
      if (launcher.region.exists())
      {
#ifdef DEBUG_LEGION
        assert(!launcher.partition.exists());
#endif
        requirement = RegionRequirement(launcher.region, launcher.projection,
                                        LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
                                        launcher.parent);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(launcher.partition.exists());
#endif
        requirement = RegionRequirement(launcher.partition, launcher.projection,
                                        LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
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
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_fill_operation(parent_ctx->get_unique_id(), 
                                      unique_op_id, context_index);
        if ((value_size == 0) && (future.impl != NULL) &&
            future.impl->get_ready_event().exists())
          LegionSpy::log_future_use(unique_op_id, 
                                    future.impl->get_ready_event());
        runtime->forest->log_launch_space(launch_space->handle, unique_op_id);
      }
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_fill();
      index_domain = Domain::NO_DOMAIN;
      launch_space = NULL;
      points_committed = 0;
      commit_request = false;
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_fill();
      // We can deactivate our point operations
      for (std::vector<PointFillOp*>::const_iterator it = points.begin();
            it != points.end(); it++)
        (*it)->deactivate();
      points.clear();
      if (remove_launch_space_reference(launch_space))
        delete launch_space;
      // Return the operation to the runtime
      runtime->free_index_fill_op(this);
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    { 
      // First compute the parent index
      compute_parent_index();
      // Promote a singular region requirement up to a projection
      if (requirement.handle_type == LEGION_SINGULAR_PROJECTION)
      {
        requirement.handle_type = LEGION_REGION_PROJECTION;
        requirement.projection = 0;
      }
      initialize_privilege_path(privilege_path, requirement);
      if (runtime->legion_spy_enabled)
        log_index_fill_requirement();
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::log_index_fill_requirement(void)
    //--------------------------------------------------------------------------
    {
      const bool reg = (requirement.handle_type == LEGION_SINGULAR_PROJECTION)
                     || (requirement.handle_type == LEGION_REGION_PROJECTION);
      const bool proj = (requirement.handle_type == LEGION_REGION_PROJECTION) 
                  || (requirement.handle_type == LEGION_PARTITION_PROJECTION);

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

    //--------------------------------------------------------------------------
    void IndexFillOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->check_privileges)
        check_fill_privilege();
      // Register a dependence on our predicate
      register_predicate_dependence();
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(this, 
                              wait_barriers, arrive_barriers);
      // If we are waiting on a future register a dependence
      if (future.impl != NULL)
        future.impl->register_dependence(this);
      ProjectionInfo projection_info(runtime, requirement, launch_space);
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions);
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Enumerate the points
      enumerate_points(false/*replaying*/); 
      // Check for interfering point requirements in debug mode
      if (runtime->check_privileges)
        check_point_requirements(); 
      // Launch the points
      std::set<RtEvent> mapped_preconditions;
      std::set<ApEvent> executed_preconditions;
      for (std::vector<PointFillOp*>::const_iterator it = points.begin();
            it != points.end(); it++)
      {
        mapped_preconditions.insert((*it)->get_mapped_event());
        executed_preconditions.insert((*it)->get_completion_event());
        (*it)->launch();
      }
#ifdef LEGION_SPY
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      completion_event);
#endif
      // Record that we are mapped when all our points are mapped
      // and we are executed when all our points are executed
      complete_mapping(Runtime::merge_events(mapped_preconditions));
      ApEvent done = Runtime::merge_events(NULL, executed_preconditions);
      if (!request_early_complete(done))
        complete_execution(Runtime::protect_event(done));
      else
        complete_execution();
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
    void IndexFillOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_replaying());
#endif
      if (runtime->legion_spy_enabled)
        log_index_fill_requirement();
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      // Enumerate the points
      enumerate_points(true/*replaying*/);
      // Then call replay analysis on all of them
      for (std::vector<PointFillOp*>::const_iterator it = 
            points.begin(); it != points.end(); it++)
        (*it)->trigger_replay();
      complete_mapping();
      complete_execution();
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::enumerate_points(bool replaying)
    //--------------------------------------------------------------------------
    {
      // Enumerate the points
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
                               runtime, projection_points, launch_space);
      if (runtime->legion_spy_enabled && !replaying)
      {
        for (std::vector<PointFillOp*>::const_iterator it = points.begin();
              it != points.end(); it++)
          (*it)->log_fill_requirement();
      }
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
            switch (p1.get_dim())
            {
              case 1:
                {
                  REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_FILL,
                            "Index space fill launch has intefering "
                            "region requirements 0 of point %lld and region "
                            "requirement 0 of point %lld of %s (UID %lld) "
                            "in parent task %s (UID %lld) are interfering.",
                            p1[0], p2[0], get_logging_name(),
                            get_unique_op_id(), parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id());
                  break;
                }
#if LEGION_MAX_DIM > 1
              case 2:
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
                  break;
                }
#endif
#if LEGION_MAX_DIM > 2
              case 3:
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
                  break;
                }
#endif
#if LEGION_MAX_DIM > 3
              case 4:
                {
                  REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_FILL,
                            "Index space fill launch has intefering "
                            "region requirements 0 of point "
                            "(%lld,%lld,%lld,%lld) and region "
                            "requirement 0 of point (%lld,%lld,%lld,%lld) of "
                            "%s (UID %lld) in parent task %s (UID %lld) are "
                            "interfering.",
                            p1[0], p1[1], p1[2], p1[3], 
                            p2[0], p2[1], p2[2], p2[3], get_logging_name(),
                            get_unique_op_id(), parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id());
                  break;
                }
#endif
#if LEGION_MAX_DIM > 4
              case 5:
                {
                  REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_FILL,
                            "Index space fill launch has intefering "
                            "region requirements 0 of point "
                            "(%lld,%lld,%lld,%lld,%lld) and "
                            "region requirement 0 of point "
                            "(%lld,%lld,%lld,%lld,%lld) of "
                            "%s (UID %lld) in parent task %s (UID %lld) are "
                            "interfering.", p1[0], p1[1], p1[2], p1[3], p1[4],
                            p2[0], p2[1], p2[2], p2[3], p2[4],
                            get_logging_name(), get_unique_op_id(),
                            parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id());
                  break;
                }
#endif
#if LEGION_MAX_DIM > 5
              case 6:
                {
                  REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_FILL,
                            "Index space fill launch has intefering "
                            "region requirements 0 of point "
                            "(%lld,%lld,%lld,%lld,%lld,%lld)"
                            " and region requirement 0 of point "
                            "(%lld,%lld,%lld,%lld,%lld,%lld) of %s "
                            "(UID %lld) in parent task %s (UID %lld) " 
                            "are interfering.",
                            p1[0], p1[1], p1[2], p1[3], p1[4], p1[5],
                            p2[0], p2[1], p2[2], p2[3], p2[4], p2[5],
                            get_logging_name(), get_unique_op_id(),
                            parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id());
                  break;
                }
#endif
#if LEGION_MAX_DIM > 6
              case 7:
                {
                  REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_FILL,
                            "Index space fill launch has intefering "
                            "region requirements 0 of point "
                            "(%lld,%lld,%lld,%lld,%lld,%lld,%lld) and region "
                            "requirement 0 of point "
                            "(%lld,%lld,%lld,%lld,%lld,%lld,%lld) of "
                            "%s (UID %lld) in parent task %s (UID %lld) are "
                            "interfering.",
                            p1[0], p1[1], p1[2], p1[3], p1[4], p1[5], p1[6],
                            p2[0], p2[1], p2[2], p2[3], p2[4], p2[5], p2[6],
                            get_logging_name(), get_unique_op_id(), 
                            parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id());
                  break;
                }
#endif
#if LEGION_MAX_DIM > 7
              case 8:
                {
                  REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_FILL,
                            "Index space fill launch has intefering "
                            "region requirements 0 of point "
                            "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld) and "
                            "region requirement 0 of point "
                            "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld) of "
                            "%s (UID %lld) in parent task %s (UID %lld) are "
                            "interfering.", p1[0], p1[1], p1[2], p1[3], p1[4],
                            p1[5], p1[6], p1[7], p2[0], p2[1], p2[2], p2[3],
                            p2[4], p2[5], p2[6], p2[7],
                            get_logging_name(), get_unique_op_id(),
                            parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id());
                  break;
                }
#endif
#if LEGION_MAX_DIM > 8
              case 9:
                {
                  REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_FILL,
                            "Index space fill launch has intefering "
                            "region requirements 0 of point "
                            "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld) "
                            "and region requirement 0 of point "
                            "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld) "
                            "of %s (UID %lld) in parent task %s (UID %lld) " 
                            "are interfering.",
                            p1[0], p1[1], p1[2], p1[3], p1[4], p1[5], p1[6],
                            p1[7], p1[8], p2[0], p2[1], p2[2], p2[3], p2[4],
                            p2[5], p2[6], p2[7], p2[8],
                            get_logging_name(), get_unique_op_id(),
                            parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id());
                  break;
                }
#endif
              default:
                assert(false);
            }
          }
        }
      }
    }

    ///////////////////////////////////////////////////////////// 
    // Point Fill Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointFillOp::PointFillOp(Runtime *rt)
      : FillOp(rt)
    //--------------------------------------------------------------------------
    {
      this->is_index_space = true;
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
      initialize_operation(own->get_context(), false/*track*/, 1/*regions*/,
          own->get_provenance());
      index_point = p;
      index_domain = own->index_domain; 
      owner = own;
      context_index = own->get_ctx_index();
      execution_fence_event = own->get_execution_fence_event();
      // From Memoizable
      trace_local_id     = owner->get_trace_local_id().first;
      tpl                = owner->get_template();
      if (tpl != NULL)
        memo_state       = owner->get_memoizable_state();
      // From Fill
      requirement        = owner->get_requirement();
      grants             = owner->grants;
      wait_barriers      = owner->wait_barriers;
      arrive_barriers    = owner->arrive_barriers;
      parent_task        = owner->parent_task;
      map_id             = owner->map_id;
      tag                = owner->tag;
      mapper_data_size   = owner->mapper_data_size;
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, owner->mapper_data, mapper_data_size);
      }
      // From FillOp
      parent_req_index   = owner->parent_req_index;
      true_guard         = owner->true_guard;
      false_guard        = owner->false_guard;
      future             = owner->future;
      value_size         = owner->value_size;
      if (value_size > 0)
      {
        value = malloc(value_size);
        memcpy(value, owner->value, value_size);
      }
      if (runtime->legion_spy_enabled)
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
    void PointFillOp::launch(void)
    //--------------------------------------------------------------------------
    {
      // Perform the version info
      std::set<RtEvent> preconditions;
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                        requirement, version_info, preconditions);
      // We can also mark this as having our resolved any predication
      resolve_speculation();
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void PointFillOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // Don't commit this operation until we've reported our profiling
      // Out index owner will deactivate the operation
      commit_operation(false/*deactivate*/);
      // Tell our owner that we are done, they will do the deactivate
      owner->handle_point_commit();
    }

    //--------------------------------------------------------------------------
    CollectiveManager* PointFillOp::find_or_create_collective_instance(
                                  MappingCallKind mapper_call, unsigned index,
                                  const LayoutConstraintSet &constraints,
                                  const std::vector<LogicalRegion> &regions,
                                  Memory::Kind kind, size_t *footprint,
                                  LayoutConstraintKind *unsat_kind,
                                  unsigned *unsat_index,
                                  DomainPoint &collective_point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index_point.get_dim() > 0);
#endif
      collective_point = index_point;
      return owner->find_or_create_collective_instance(mapper_call, index,
          constraints, regions, kind, footprint, unsat_kind, unsat_index,
          collective_point);
    }

    //--------------------------------------------------------------------------
    bool PointFillOp::finalize_collective_instance(MappingCallKind call_kind,
                                                   unsigned index, bool success)
    //--------------------------------------------------------------------------
    {
      return owner->finalize_collective_instance(call_kind, index, success);
    }

    //--------------------------------------------------------------------------
    void PointFillOp::report_total_collective_instance_calls(
                              MappingCallKind mapper_call, unsigned total_calls)
    //--------------------------------------------------------------------------
    {
      owner->report_total_collective_instance_calls(mapper_call, total_calls);
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
      requirement.handle_type = LEGION_SINGULAR_PROJECTION;
    }

    //--------------------------------------------------------------------------
    void PointFillOp::record_intra_space_dependences(unsigned index,
                                    const std::vector<DomainPoint> &dependences)
    //--------------------------------------------------------------------------
    {
      // Ignore any intra-space requirements on fills, we know that they
      // are all filling the same value so they can be done in any order
    }

    //--------------------------------------------------------------------------
    TraceLocalID PointFillOp::get_trace_local_id(void) const
    //--------------------------------------------------------------------------
    {
      return TraceLocalID(trace_local_id, index_point);
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
    PhysicalRegion AttachOp::initialize(InnerContext *ctx,
                                        const AttachLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 1/*regions*/, 
          launcher.provenance.c_str(), launcher.static_dependences);
      resource = launcher.resource;
      footprint = launcher.footprint;
      restricted = launcher.restricted;
      mapping = launcher.mapped;
      switch (resource)
      {
        case LEGION_EXTERNAL_POSIX_FILE:
          {
            if (launcher.file_fields.empty()) 
              REPORT_LEGION_WARNING(LEGION_WARNING_FILE_ATTACH_OPERATION,
                              "FILE ATTACH OPERATION ISSUED WITH NO "
                              "FIELD MAPPINGS IN TASK %s (ID %lld)! DID YOU "
                              "FORGET THEM?!?", parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id())
            file_name = strdup(launcher.file_name);
            // Construct the region requirement for this task
            requirement = RegionRequirement(launcher.handle, 
                LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, launcher.parent);
            for (std::vector<FieldID>::const_iterator it = 
                  launcher.file_fields.begin(); it != 
                  launcher.file_fields.end(); it++)
              requirement.add_field(*it);
            file_mode = launcher.mode;       
            break;
          }
        case LEGION_EXTERNAL_HDF5_FILE:
          {
#ifndef LEGION_USE_HDF5
            REPORT_LEGION_ERROR(ERROR_ATTACH_HDF5,
                "Invalid attach HDF5 file in parent task %s (UID %lld). "
                "Legion must be built with HDF5 support to attach regions "
                "to HDF5 files", parent_ctx->get_task_name(),
                parent_ctx->get_unique_id())
#endif
            if (launcher.field_files.empty()) 
              REPORT_LEGION_WARNING(LEGION_WARNING_HDF5_ATTACH_OPERATION,
                            "HDF5 ATTACH OPERATION ISSUED WITH NO "
                            "FIELD MAPPINGS IN TASK %s (ID %lld)! DID YOU "
                            "FORGET THEM?!?", parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id())
            file_name = strdup(launcher.file_name);
            // Construct the region requirement for this task
            requirement = RegionRequirement(launcher.handle, 
                LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, launcher.parent);
            for (std::map<FieldID,const char*>::const_iterator it = 
                  launcher.field_files.begin(); it != 
                  launcher.field_files.end(); it++)
            {
              requirement.add_field(it->first);
              field_map[it->first] = strdup(it->second);
            }
            file_mode = launcher.mode;
            // For HDF5 we use the dimension ordering if there is one, 
            // otherwise we'll fill it in ourselves
            const OrderingConstraint &input_constraint = 
              launcher.constraints.ordering_constraint;
            OrderingConstraint &output_constraint = 
              layout_constraint_set.ordering_constraint;
            const int dims = launcher.handle.index_space.get_dim();
            if (!input_constraint.ordering.empty())
            {
              bool has_dimf = false;
              for (std::vector<DimensionKind>::const_iterator it = 
                    input_constraint.ordering.begin(); it !=
                    input_constraint.ordering.end(); it++)
              {
                // dimf should always be the last dimension for HDF5
                if (has_dimf)
                  REPORT_LEGION_ERROR(ERROR_ATTACH_HDF5_CONSTRAINT,
                      "Invalid position of the field dimension for attach "
                      "operation %lld in task %s (UID %lld). The field "
                      "dimension must always be the last dimension for layout "
                      "constraints for HDF5 files.", unique_op_id,
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id())
                else if (*it == LEGION_DIM_F)
                  has_dimf = true;
                else if (int(*it) > dims)
                  REPORT_LEGION_ERROR(ERROR_ATTACH_HDF5_CONSTRAINT,
                      "Invalid dimension %d for ordering constraint of HDF5 "
                      "attach operation %lld in task %s (UID %lld). The "
                      "index space %x only has %d dimensions and split "
                      "dimensions are not permitted.", *it, unique_op_id,
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id(), 
                      launcher.handle.index_space.get_id(), dims)
                output_constraint.ordering.push_back(*it);
              }
              if (int(output_constraint.ordering.size()) != dims)
                REPORT_LEGION_ERROR(ERROR_ATTACH_HDF5_CONSTRAINT,
                    "Ordering constraint for attach %lld in task %s (UID %lld) "
                    "does not contain all the dimensions required for index "
                    "space %x which has %d dimensions.", unique_op_id,
                    parent_ctx->get_task_name(), parent_ctx->get_unique_id(),
                    launcher.handle.index_space.get_id(), dims)
              if (!input_constraint.contiguous)
                REPORT_LEGION_ERROR(ERROR_ATTACH_HDF5_CONSTRAINT,
                    "Ordering constraint for attach %lld in task %s (UID %lld) "
                    "was not marked contiguous. All ordering constraints for "
                    "HDF5 attach operations must be contiguous", unique_op_id,
                    parent_ctx->get_task_name(), parent_ctx->get_unique_id())
              if (!has_dimf)
                output_constraint.ordering.push_back(LEGION_DIM_F);
            }
            else
            {
              // Fill in the ordering constraints for dimensions based
              // on the number of dimensions
              for (int i = 0; i < dims; i++)
                output_constraint.ordering.push_back(
                    (DimensionKind)(LEGION_DIM_X + i)); 
              output_constraint.ordering.push_back(LEGION_DIM_F);
            }
            output_constraint.contiguous = true;
            break;
          }
        case LEGION_EXTERNAL_INSTANCE:
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
            requirement = RegionRequirement(launcher.handle, 
                LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, launcher.parent);
            for (std::set<FieldID>::const_iterator it = 
                  fields.begin(); it != fields.end(); it++)
              requirement.add_field(*it);
            break;
          }
        default:
          assert(false); // should never get here
      }
      if (mapping)
      {
        const ApUserEvent term_event = Runtime::create_ap_user_event(NULL);
        region = PhysicalRegion(new PhysicalRegionImpl(requirement,mapped_event,
              completion_event, term_event, true/*mapped*/, ctx, 0/*map id*/, 
              0/*tag*/, false/*leaf*/, false/*virtual mapped*/, runtime));
        termination_event = term_event;
      }
      else
        region = PhysicalRegion(new PhysicalRegionImpl(requirement,
              mapped_event, completion_event, ApUserEvent::NO_AP_USER_EVENT, 
              false/*mapped*/, ctx, 0/*map id*/, 0/*tag*/, false/*leaf*/, 
              false/*virtual mapped*/, runtime)); 
      if (runtime->legion_spy_enabled)
        LegionSpy::log_attach_operation(parent_ctx->get_unique_id(),
                            unique_op_id, context_index, restricted);
      return region;
    }

    //--------------------------------------------------------------------------
    void AttachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_attach();
    }

    //--------------------------------------------------------------------------
    void AttachOp::activate_attach(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      file_name = NULL;
      footprint = 0;
      attached_event = ApUserEvent::NO_AP_USER_EVENT;
      termination_event = ApEvent::NO_AP_EVENT;
      restricted = true;
    }

    //--------------------------------------------------------------------------
    void AttachOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_attach();
      runtime->free_attach_op(this);
    }

    //--------------------------------------------------------------------------
    void AttachOp::deactivate_attach(void)
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
      map_applied_conditions.clear();
      external_instances.clear();
      layout_constraint_set = LayoutConstraintSet();
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
      create_external_instance();
      if (runtime->legion_spy_enabled)
        log_requirement();
    }

    //--------------------------------------------------------------------------
    LogicalRegion AttachOp::create_external_instance(void)
    //--------------------------------------------------------------------------
    {
      external_instances.resize(1);
      switch (resource)
      {
        case LEGION_EXTERNAL_POSIX_FILE:
        case LEGION_EXTERNAL_HDF5_FILE:
          {
            external_instances[0] = 
              runtime->forest->create_external_instance(this, requirement, 
                                              requirement.instance_fields);
            break;
          }
        case LEGION_EXTERNAL_INSTANCE:
          {
            external_instances[0] = 
              runtime->forest->create_external_instance(this, requirement,
                        layout_constraint_set.field_constraint.field_set);
            break;
          }
        default:
          assert(false);
      }
      return requirement.region;
    }

    //--------------------------------------------------------------------------
    void AttachOp::log_requirement(void)
    //--------------------------------------------------------------------------
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
    
    //--------------------------------------------------------------------------
    void AttachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->check_privileges)
        check_privilege();
      ProjectionInfo projection_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions); 
    }

    //--------------------------------------------------------------------------
    void AttachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    { 
      std::set<RtEvent> preconditions;  
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   version_info,
                                                   preconditions);
      // Register the instance with the memory manager and make sure it is
      // done before we perform our mapping
#ifdef DEBUG_LEGION
      assert(!external_instances.empty());
      assert(external_instances[0].has_ref());
#endif
      PhysicalManager *manager = external_instances[0].get_physical_manager();
      const RtEvent attached = manager->attach_external_instance();
      if (attached.exists())
        preconditions.insert(attached);
      // Perform an exchange looking for any coregions for point AttachOps
      const RtEvent exchanged = check_for_coregions();
      if (exchanged.exists())
        preconditions.insert(exchanged);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    RtEvent AttachOp::check_for_coregions(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here for individual attach ops
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void AttachOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    { 
      const PhysicalTraceInfo trace_info(this, 0/*idx*/, true/*init*/);
      // Check to see if we're going to be the ones performing the attach
      // If someone else has already done the attach for us then the 
      // attached event will be non-trivial
      ApEvent attach_event = attached_event;
      if (!attach_event.exists() || (external_instances.size() > 1))
      {
        InnerContext *context = find_physical_context(0/*index*/, requirement);
        std::vector<InstanceView*> external_views;
        context->convert_target_views(external_instances, external_views);
#ifdef DEBUG_LEGION
        assert(external_views.size() == 1);
#endif
        std::set<LogicalView*> ext_views(external_views.begin(), 
                                         external_views.end());
        attach_event = runtime->forest->attach_external(this, 0/*idx*/,
                                                        requirement,
                                                        external_views,
                                                        ext_views, mapping ?
                                                          termination_event :
                                                          completion_event,
                                                        version_info,
                                                        trace_info,
                                                        map_applied_conditions,
                                                        restricted);
        // Signal to any other point tasks that we performed the attach for them
        if (attached_event.exists())
          Runtime::trigger_event(&trace_info, attached_event, attach_event);
      }
      if (mapping)
        external_instances[0].set_ready_event(attach_event);
      // This operation is ready once the file is attached
      region.impl->set_reference(external_instances[0]);
      // Once we have created the instance, then we are done
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      if (!request_early_complete(attach_event))
        complete_execution(Runtime::protect_event(attach_event));
      else
        complete_execution();
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
      commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void AttachOp::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    void AttachOp::pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                        std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);      
    }

    //--------------------------------------------------------------------------
    PhysicalInstance AttachOp::create_instance(IndexSpaceNode *node,
                                         const std::vector<FieldID> &field_set,
                                         const std::vector<size_t> &sizes,
                                               LayoutConstraintSet &constraints,
                                               ApEvent &ready_event,
                                               size_t &instance_footprint)
    //--------------------------------------------------------------------------
    {
      PhysicalInstance result = PhysicalInstance::NO_INST;
      instance_footprint = footprint;
      switch (resource)
      {
        case LEGION_EXTERNAL_POSIX_FILE:
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
              SpecializedConstraint(LEGION_GENERIC_FILE_SPECIALIZE);           
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
        case LEGION_EXTERNAL_HDF5_FILE:
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
                                      layout_constraint_set.ordering_constraint,
                                      (file_mode == LEGION_FILE_READ_ONLY),
                                      ready_event);
            constraints.specialized_constraint = 
              SpecializedConstraint(LEGION_HDF5_FILE_SPECIALIZE);
            constraints.field_constraint = 
              FieldConstraint(requirement.privilege_fields, 
                              false/*contiguous*/, false/*inorder*/);
            constraints.memory_constraint = 
              MemoryConstraint(result.get_location().kind());
            constraints.ordering_constraint = 
              layout_constraint_set.ordering_constraint;
            break;
          }
        case LEGION_EXTERNAL_INSTANCE:
          {
            Realm::InstanceLayoutGeneric *ilg = node->create_layout(
                layout_constraint_set, field_set, sizes, false/*compact*/);
            const PointerConstraint &pointer = 
                                      layout_constraint_set.pointer_constraint;
#ifdef DEBUG_LEGION
            assert(pointer.is_valid);
#endif
            Realm::ExternalMemoryResource res(pointer.ptr, ilg->bytes_used,
                                              false /*!read_only*/);

            const Memory memory = res.suggested_memory();
            if ((memory != pointer.memory) && pointer.memory.exists())
            {
              const char *mem_names[] = {
#define MEM_NAMES(name, desc) desc,
                REALM_MEMORY_KINDS(MEM_NAMES) 
#undef MEM_NAMES
              };
              REPORT_LEGION_WARNING(LEGION_WARNING_IMPRECISE_ATTACH_MEMORY,
                  "WARNING: %s memory " IDFMT " in pointer constraint for "
                  "attach operation %lld in parent task %s (UID %lld) differs "
                  "from the Realm-suggested %s memory " IDFMT " for the "
                  "external instance. Legion is going to use the more precise "
                  "Realm-specified memory. Please make sure that you do not "
                  "have any code in your application or your mapper that "
                  "relies on the instance being in the originally specified "
                  "memory. To silence this warning you can pass in a NO_MEMORY "
                  "to the pointer constraint.",
                  mem_names[pointer.memory.kind()], pointer.memory.id,
                  unique_op_id, parent_ctx->get_task_name(),
                  parent_ctx->get_unique_id(), mem_names[memory.kind()],
                  memory.id);
            }
            // No profiling for these kinds of instances currently
            Realm::ProfilingRequestSet requests;
            ready_event = ApEvent(PhysicalInstance::create_external_instance(
                                          result, memory, ilg, res, requests));
            constraints = layout_constraint_set;
            constraints.specialized_constraint =
              SpecializedConstraint(LEGION_AFFINE_SPECIALIZE);
            constraints.memory_constraint = MemoryConstraint(memory.kind());
            constraints.pointer_constraint = 
              PointerConstraint(memory, pointer.ptr);
            break;
          }
        default:
          assert(false);
      }
      if (runtime->legion_spy_enabled)
      {
        // We always need a unique ready event for Legion Spy
        if (!ready_event.exists())
        {
          ApUserEvent rename_ready = Runtime::create_ap_user_event(NULL);
          Runtime::trigger_event(NULL, rename_ready);
          ready_event = rename_ready;
        }
        for (std::set<FieldID>::const_iterator it = 
              requirement.privilege_fields.begin(); it !=
              requirement.privilege_fields.end(); it++)
          LegionSpy::log_mapping_decision(unique_op_id, 0/*idx*/, 
                                          *it, ready_event);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                        completion_event);
#endif
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void AttachOp::check_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field = LEGION_AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, then check the privileges with the parent context
      if (et == LEGION_NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, bad_index);
      switch (et)
      {
        // Note there is no such things as bad privileges for
        // acquires and releases because they are controlled by the runtime
        case LEGION_NO_ERROR:
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
            FieldSpace sp = 
              (requirement.handle_type == LEGION_SINGULAR_PROJECTION) ||
              (requirement.handle_type == LEGION_REGION_PROJECTION) ? 
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
            else if (bad_field == LEGION_AUTO_GENERATE_ID) 
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
        // this should never happen with an attach operation
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
    // Index Attach Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexAttachOp::IndexAttachOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAttachOp::IndexAttachOp(const IndexAttachOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndexAttachOp::~IndexAttachOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAttachOp& IndexAttachOp::operator=(const IndexAttachOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_index_attach();
    }
    
    //--------------------------------------------------------------------------
    void IndexAttachOp::activate_index_attach(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      launch_space = NULL;
      points_committed = 0;
      commit_request = false;
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_index_attach();
      runtime->free_index_attach_op(this);
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::deactivate_index_attach(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      resources = ExternalResources();
      privilege_path.clear();
      // We can deactivate all of our point operations
      for (std::vector<PointAttachOp*>::const_iterator it =
            points.begin(); it != points.end(); it++)
        (*it)->deactivate();
      points.clear();
      coregions.clear();
      coregions_attached.clear();
      map_applied_conditions.clear();
    }

    //--------------------------------------------------------------------------
    const char* IndexAttachOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[ATTACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind IndexAttachOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ATTACH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t IndexAttachOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    ExternalResources IndexAttachOp::initialize(InnerContext *ctx,
                                      RegionTreeNode *upper_bound,
                                      IndexSpaceNode *launch_bounds,
                                      const IndexAttachLauncher &launcher,
                                      const std::vector<unsigned> &indexes)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 1/*regions*/,
          launcher.provenance.c_str(), launcher.static_dependences);
      // Construct the region requirement
      // Use a fake projection ID for now, we'll fill it in later during the
      // prepipeline stage before the logical dependence analysis
      // so that our computation of it is off the critical path
      if (upper_bound->is_region())
        requirement = RegionRequirement(upper_bound->as_region_node()->handle,
            0/*fake*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, launcher.parent);
      else
        requirement = 
          RegionRequirement(upper_bound->as_partition_node()->handle,
            0/*fake*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, launcher.parent);
      requirement.privilege_fields = launcher.privilege_fields;
      launch_space = launch_bounds;
      // Do some error checking
      OrderingConstraint output_constraint;
      switch (launcher.resource)
      {
        case LEGION_EXTERNAL_POSIX_FILE:
          {
            if (launcher.file_fields.empty()) 
              REPORT_LEGION_WARNING(LEGION_WARNING_FILE_ATTACH_OPERATION,
                              "FILE INDEX ATTACH OPERATION ISSUED WITH NO "
                              "FIELD MAPPINGS IN TASK %s (ID %lld)! DID YOU "
                              "FORGET THEM?!?", parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id())
            break;
          }
        case LEGION_EXTERNAL_HDF5_FILE:
          {
#ifndef LEGION_USE_HDF5
            REPORT_LEGION_ERROR(ERROR_ATTACH_HDF5,
                "Invalid index attach HDF5 file in parent task %s (UID %lld). "
                "Legion must be built with HDF5 support to attach regions "
                "to HDF5 files", parent_ctx->get_task_name(),
                parent_ctx->get_unique_id())
#endif
            if (launcher.field_files.empty()) 
              REPORT_LEGION_WARNING(LEGION_WARNING_HDF5_ATTACH_OPERATION,
                            "HDF5 INDEX ATTACH OPERATION ISSUED WITH NO "
                            "FIELD MAPPINGS IN TASK %s (ID %lld)! DID YOU "
                            "FORGET THEM?!?", parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id())
            const OrderingConstraint &input_constraint = 
              launcher.constraints.ordering_constraint;
            const int dims = launcher.parent.index_space.get_dim();
            if (!input_constraint.ordering.empty())
            {
              bool has_dimf = false;
              for (std::vector<DimensionKind>::const_iterator it = 
                    input_constraint.ordering.begin(); it !=
                    input_constraint.ordering.end(); it++)
              {
                // dimf should always be the last dimension for HDF5
                if (has_dimf)
                  REPORT_LEGION_ERROR(ERROR_ATTACH_HDF5_CONSTRAINT,
                      "Invalid position of the field dimension for index attach"
                      " operation %lld in task %s (UID %lld). The field "
                      "dimension must always be the last dimension for layout "
                      "constraints for HDF5 files.", unique_op_id,
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id())
                else if (*it == LEGION_DIM_F)
                  has_dimf = true;
                else if (int(*it) > dims)
                  REPORT_LEGION_ERROR(ERROR_ATTACH_HDF5_CONSTRAINT,
                      "Invalid dimension %d for ordering constraint of HDF5 "
                      "index attach operation %lld in task %s (UID %lld). The "
                      "index space %x only has %d dimensions and split "
                      "dimensions are not permitted.", *it, unique_op_id,
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id(), 
                      launcher.parent.index_space.get_id(), dims)
                output_constraint.ordering.push_back(*it);
              }
              if (int(output_constraint.ordering.size()) != dims)
                REPORT_LEGION_ERROR(ERROR_ATTACH_HDF5_CONSTRAINT,
                    "Ordering constraint for index attach %lld in task %s "
                    "(UID %lld) does not contain all the dimensions required "
                    "for index space %x which has %d dimensions.", unique_op_id,
                    parent_ctx->get_task_name(), parent_ctx->get_unique_id(),
                    launcher.parent.index_space.get_id(), dims)
              if (!input_constraint.contiguous)
                REPORT_LEGION_ERROR(ERROR_ATTACH_HDF5_CONSTRAINT,
                    "Ordering constraint for index attach %lld in task %s "
                    "(UID %lld) was not marked contiguous. All ordering "
                    "constraints for HDF5 attach operations must be contiguous",
                    unique_op_id, parent_ctx->get_task_name(), 
                    parent_ctx->get_unique_id())
              if (!has_dimf)
                output_constraint.ordering.push_back(LEGION_DIM_F);
            }
            else
            {
              // Fill in the ordering constraints for dimensions based
              // on the number of dimensions
              for (int i = 0; i < dims; i++)
                output_constraint.ordering.push_back(
                    (DimensionKind)(LEGION_DIM_X + i)); 
              output_constraint.ordering.push_back(LEGION_DIM_F);
            }
            output_constraint.contiguous = true;
            break;
          }
        case LEGION_EXTERNAL_INSTANCE:
          {
            if (launcher.privilege_fields.empty())
              REPORT_LEGION_WARNING(LEGION_WARNING_EXTERNAL_ATTACH_OPERATION,
                            "EXTERNAL ARRAY INDEX ATTACH OPERATION ISSUED WITH "
                            "NO PRIVILEGE FIELDS IN TASK %s (ID %lld)! DID YOU "
                            "FORGET THEM?!?", parent_ctx->get_task_name(),
                            parent_ctx->get_unique_id())
            break;
          }
        default:
          assert(false); // should never get here
      }
      // Create the result and the point attach operations
      ExternalResourcesImpl *result = new ExternalResourcesImpl(ctx,
          indexes.size(), upper_bound, launch_bounds, launcher.parent,
          launcher.privilege_fields);
      points.resize(indexes.size());
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        points[idx] = runtime->get_available_point_attach_op();
        const DomainPoint index_point = Point<1>(indexes[idx]);
        PhysicalRegionImpl *region = points[idx]->initialize(this, ctx,
            launcher, output_constraint, index_point, indexes[idx]);
        result->set_region(idx, region);
      }
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_attach_operation(parent_ctx->get_unique_id(),
                  unique_op_id, context_index, false/*restricted*/);
        runtime->forest->log_launch_space(launch_space->handle, unique_op_id);
      }
      resources = ExternalResources(result);
      return resources;
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent index
      compute_parent_index();
      // Promote a singular region requirement up to a projection
      if (requirement.handle_type == LEGION_SINGULAR_PROJECTION)
      {
        requirement.handle_type = LEGION_REGION_PROJECTION;
        requirement.projection = 0;
      }
      initialize_privilege_path(privilege_path, requirement);
      // Have each of the point tasks create their external instances
      // Keep track of which points share the same logical region
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        PointAttachOp *point = points[idx];
        LogicalRegion region = point->create_external_instance();
        coregions[region].push_back(point);
      }
      if (runtime->check_privileges)
        check_point_requirements();
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Compute the projection function id for this requirement
      std::vector<IndexSpace> spaces(points.size());
      for (unsigned idx = 0; idx < points.size(); idx++)
        spaces[idx] = points[idx]->get_requirement().region.get_index_space();
      if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
        requirement.projection = parent_ctx->compute_index_attach_projection(
            runtime->forest->get_node(requirement.partition.index_partition),
            spaces);
      else
        requirement.projection = parent_ctx->compute_index_attach_projection(
            runtime->forest->get_node(requirement.region.index_space), spaces);
      // Save this for later when we go to detach it
      resources.impl->set_projection(requirement.projection);
      if (runtime->check_privileges)
        check_privilege();
      if (runtime->legion_spy_enabled)
        log_requirement();
      ProjectionInfo projection_info(runtime, requirement, launch_space);
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions);
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> mapped_preconditions;
      std::set<ApEvent> executed_preconditions;
      for (std::vector<PointAttachOp*>::const_iterator it =
            points.begin(); it != points.end(); it++)
      {
        mapped_preconditions.insert((*it)->get_mapped_event());
        executed_preconditions.insert((*it)->get_completion_event());
        (*it)->trigger_ready();
      }
#ifdef LEGION_SPY
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      completion_event);
#endif
      // Record that we are mapped when all our points are mapped
      // and we are executed when all our points are executed
      complete_mapping(Runtime::merge_events(mapped_preconditions));
      ApEvent done = Runtime::merge_events(NULL, executed_preconditions);
      if (!request_early_complete(done))
        complete_execution(Runtime::protect_event(done));
      else
        complete_execution();
    }

    //--------------------------------------------------------------------------
    RtEvent IndexAttachOp::find_coregions(PointAttachOp *point,
         LogicalRegion reg, InstanceSet &instances, ApUserEvent &attached_event)
    //--------------------------------------------------------------------------
    {
      // No need for the lock here since we know this is being done
      // sequentially by IndexAttachOp::trigger_ready
      std::map<LogicalRegion,std::vector<PointAttachOp*> >::iterator
        finder = coregions.find(reg);
#ifdef DEBUG_LEGION
      assert(finder != coregions.end());
      assert(!finder->second.empty());
#endif
      if (finder->second.size() == 1)
      {
        // No co-regions in this case
        coregions.erase(finder);
        return RtEvent::NO_RT_EVENT;
      }
#ifdef DEBUG_LEGION
      assert(!attached_event.exists());
#endif
      // See if we're the first one
      if (finder->second.front() == point)
      {
        // We're the first one
        // Get the instances from all the other points
        const size_t offset = instances.size();
        instances.resize(offset + finder->second.size());
        for (unsigned idx = 0; idx < finder->second.size(); idx++)
          instances[offset+idx] = finder->second[idx]->external_instances[0];
        // Save our attached event for later
        attached_event = Runtime::create_ap_user_event(NULL);
        coregions_attached[reg] = attached_event;  
      }
      else
      {
        // Record the attached event to know that a different
        // operation is going to be performing the attach for this point
        std::map<LogicalRegion,ApUserEvent>::const_iterator event_finder =
          coregions_attached.find(reg);
#ifdef DEBUG_LEGION
        assert(event_finder != coregions_attached.end());
        assert(event_finder->second.exists());
#endif
        attached_event = event_finder->second;
      }
      // Will only have non-trival events here with control replication
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::trigger_commit(void)
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
    void IndexAttachOp::handle_point_commit(void)
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

    //--------------------------------------------------------------------------
    unsigned IndexAttachOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::compute_parent_index(void)
    //--------------------------------------------------------------------------
    {
      int parent_index = parent_ctx->find_parent_region_req(requirement);
      if (parent_index < 0)
        REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ATTACH,
                               "Parent task %s (ID %lld) of index attach "
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
    void IndexAttachOp::check_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field = LEGION_AUTO_GENERATE_ID;
      int bad_index = -1;
      LegionErrorType et = runtime->verify_requirement(requirement, bad_field);
      // If that worked, then check the privileges with the parent context
      if (et == LEGION_NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field, bad_index);
      switch (et)
      {
        // Note there is no such things as bad privileges for
        // acquires and releases because they are controlled by the runtime
        case LEGION_NO_ERROR:
        case ERROR_BAD_REGION_PRIVILEGES:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            REPORT_LEGION_ERROR(ERROR_REQUIREMENTS_INVALID_REGION,
                               "Requirements for invalid region handle (%x,%d,"
                               "%d) for index attach operation (ID %lld)",
                               requirement.region.index_space.id,
                               requirement.region.field_space.id,
                               requirement.region.tree_id,
                               unique_op_id)
            break;
          }
        case ERROR_INVALID_PARTITION_HANDLE:
          {
            REPORT_LEGION_ERROR(ERROR_REQUIREMENTS_INVALID_REGION,
                               "Requirements for invalid partition handle (%x,"
                               "%d,%d) for index attach operation (ID %lld)",
                               requirement.partition.index_partition.id,
                               requirement.partition.field_space.id,
                               requirement.partition.tree_id,
                               unique_op_id)
            break;
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = 
              (requirement.handle_type == LEGION_SINGULAR_PROJECTION) ||
              (requirement.handle_type == LEGION_REGION_PROJECTION) ? 
                requirement.region.field_space :
                requirement.partition.field_space;
            REPORT_LEGION_ERROR(ERROR_FIELD_NOT_VALID,
                             "Field %d is not a valid field of field "
                             "space %d for index attach operation (ID %lld)",
                             bad_field, sp.id, unique_op_id)
            break;
          }
        case ERROR_INVALID_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_PRIVILEGE,
                             "Instance field %d is not one of the privilege "
                             "fields for index attach operation (ID %lld)",
                             bad_field, unique_op_id)
            break;
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_DUPLICATE,
                             "Instance field %d is a duplicate for "
                             "index attach operation (ID %lld)",
                             bad_field, unique_op_id)
            break;
          }
        case ERROR_BAD_PARENT_REGION:
          {
            if (bad_index > 0) 
            {
              if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
                REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ATTACH,
                                 "Parent task %s (ID %lld) of index attach "
                                 "operation (ID %lld) does not have a region "
                                 "requirement for partition (%x,%x,%x) "
                                 "as a parent of region requirement because "
                                 "no 'parent' region had that name.",
                                 parent_ctx->get_task_name(),
                                 parent_ctx->get_unique_id(),
                                 unique_op_id,
                                 requirement.partition.index_partition.id,
                                 requirement.partition.field_space.id,
                                 requirement.partition.tree_id)
              else
                REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ATTACH,
                                 "Parent task %s (ID %lld) of index attach "
                                 "operation (ID %lld) does not have a region "
                                 "requirement for region (%x,%x,%x) "
                                 "as a parent of region requirement because "
                                 "no 'parent' region had that name.",
                                 parent_ctx->get_task_name(),
                                 parent_ctx->get_unique_id(),
                                 unique_op_id,
                                 requirement.region.index_space.id,
                                 requirement.region.field_space.id,
                                 requirement.region.tree_id)
            }
            else if (bad_field == LEGION_AUTO_GENERATE_ID) 
            {
              if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
                REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ATTACH,
                                 "Parent task %s (ID %lld) of index attach "
                                 "operation (ID %lld) does not have a region "
                                 "requirement for partition (%x,%x,%x) "
                                 "as a parent of region requirement because "
                                 "parent requirement %d did not have "
                                 "sufficient privileges.",
                                 parent_ctx->get_task_name(),
                                 parent_ctx->get_unique_id(),
                                 unique_op_id,
                                 requirement.partition.index_partition.id,
                                 requirement.partition.field_space.id,
                                 requirement.partition.tree_id, bad_index)
              else
                REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ATTACH,
                                 "Parent task %s (ID %lld) of index attach "
                                 "operation (ID %lld) does not have a region "
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
            }
            else 
            {
              if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
                REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ATTACH,
                                 "Parent task %s (ID %lld) of index attach "
                                 "operation (ID %lld) does not have a region "
                                 "requirement for partition (%x,%x,%x) "
                                 "as a parent of region requirement because "
                                 "region requirement %d was missing field %d.",
                                 parent_ctx->get_task_name(),
                                 parent_ctx->get_unique_id(),
                                 unique_op_id,
                                 requirement.partition.index_partition.id,
                                 requirement.partition.field_space.id,
                                 requirement.partition.tree_id,
                                 bad_index, bad_field)
              else
                REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ATTACH,
                                 "Parent task %s (ID %lld) of index attach "
                                 "operation (ID %lld) does not have a region "
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
            }
            break;
          }
        case ERROR_BAD_REGION_TYPE:
          {
            REPORT_LEGION_ERROR(ERROR_REGION_REQUIREMENT_ATTACH,
                             "Region requirement of index attach operation "
                             "(ID %lld) cannot find privileges for field "
                             "%d in parent task",
                             unique_op_id, bad_field)
            break;
          }
        // this should never happen with an index attach operation
        case ERROR_BAD_REGION_PATH:
        case ERROR_NON_DISJOINT_PARTITION:
        default:
          assert(false); // Should never happen
      }
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::check_point_requirements(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx1 = 1; idx1 < points.size(); idx1++)
      {
        PointAttachOp *p1 = points[idx1];
        const RegionRequirement &r1 = p1->get_requirement();
#ifdef DEBUG_LEGION
        assert(r1.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
        for (unsigned idx2 = 0; idx2 < idx1; idx2++)
        {
          PointAttachOp *p2 = points[idx2];
          const RegionRequirement &r2 = p2->get_requirement();
#ifdef DEBUG_LEGION
          assert(r2.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
          if (!runtime->forest->are_disjoint(r1.region.get_index_space(),
                                             r2.region.get_index_space()))
            REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_ATTACH,
                "Index attach operation (UID %lld) in parent task %s "
                "(UID %lld) has interfering attachments to regions (%d,%d,%d) "
                "and (%d,%d,%d). All regions must be non-interfering",
                unique_op_id, parent_ctx->get_task_name(),
                parent_ctx->get_unique_id(), r1.region.index_space.id,
                r1.region.field_space.id, r1.region.tree_id,
                r2.region.index_space.id, r2.region.field_space.id,
                r2.region.tree_id)
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::log_requirement(void)
    //--------------------------------------------------------------------------
    {
      if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
        LegionSpy::log_logical_requirement(unique_op_id,
                                       0/*index*/, false/*region*/,
                                       requirement.partition.index_partition.id,
                                       requirement.partition.field_space.id,
                                       requirement.partition.tree_id,
                                       requirement.privilege,
                                       requirement.prop,
                                       requirement.redop,
                                       requirement.parent.index_space.id);
      else
        LegionSpy::log_logical_requirement(unique_op_id, 0/*index*/,
                                           true/*region*/,
                                           requirement.region.index_space.id,
                                           requirement.region.field_space.id,
                                           requirement.region.tree_id,
                                           requirement.privilege,
                                           requirement.prop,
                                           requirement.redop,
                                           requirement.parent.index_space.id);
      LegionSpy::log_requirement_projection(unique_op_id, 0/*index*/,
                                            requirement.projection);
      LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                        requirement.privilege_fields);
    }

    ///////////////////////////////////////////////////////////// 
    // Point Attach Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointAttachOp::PointAttachOp(Runtime *rt)
      : AttachOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointAttachOp::PointAttachOp(const PointAttachOp &rhs)
      : AttachOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PointAttachOp::~PointAttachOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointAttachOp& PointAttachOp::operator=(const PointAttachOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PointAttachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_attach();
      owner = NULL;
    }

    //--------------------------------------------------------------------------
    void PointAttachOp::deactivate(void) 
    //--------------------------------------------------------------------------
    {
      deactivate_attach();
      runtime->free_point_attach_op(this);
    }

    //--------------------------------------------------------------------------
    PhysicalRegionImpl* PointAttachOp::initialize(IndexAttachOp *own,
                         InnerContext *ctx, const IndexAttachLauncher &launcher,
                         const OrderingConstraint &ordering_constraint,
                         const DomainPoint &point, unsigned index)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index < launcher.handles.size());
#endif
      initialize_operation(ctx, false/*track*/, 1/*regions*/,
          own->get_provenance());
      owner = own;
      index_point = point;
      context_index = own->get_ctx_index();
      resource = launcher.resource;
      if (index < launcher.footprint.size())
        footprint = launcher.footprint[index];
      restricted = launcher.restricted;
      mapping = false;
      switch (resource)
      {
        case LEGION_EXTERNAL_POSIX_FILE:
          {
            if (index >= launcher.file_names.size())
              REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_ATTACH,
                  "Insufficient 'file_names' provided by index attach launch "
                  "in parent task %s (UID %lld). Launcher has %zd logical "
                  "regions but only %zd POSIX file names.", 
                  parent_ctx->get_task_name(), parent_ctx->get_unique_id(),
                  launcher.handles.size(), launcher.file_names.size())
            file_name = strdup(launcher.file_names[index]);
            // Construct the region requirement for this task
            requirement = RegionRequirement(launcher.handles[index], 
                LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, launcher.parent);
            for (std::vector<FieldID>::const_iterator it = 
                  launcher.file_fields.begin(); it != 
                  launcher.file_fields.end(); it++)
              requirement.add_field(*it);
            file_mode = launcher.mode;
            break;
          }
        case LEGION_EXTERNAL_HDF5_FILE:
          {
            if (index >= launcher.file_names.size())
              REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_ATTACH,
                  "Insufficient 'file_names' provided by index attach launch "
                  "in parent task %s (UID %lld). Launcher has %zd logical "
                  "regions but only %zd HDF5 file names.", 
                  parent_ctx->get_task_name(), parent_ctx->get_unique_id(),
                  launcher.handles.size(), launcher.file_names.size())
            file_name = strdup(launcher.file_names[index]);
            // Construct the region requirement for this task
            requirement = RegionRequirement(launcher.handles[index], 
                LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, launcher.parent);
            for (std::map<FieldID,std::vector<const char*> >::const_iterator
                  it = launcher.field_files.begin();
                  it != launcher.field_files.end(); it++)
            {
              if (index >= it->second.size())
                REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_ATTACH,
                    "Insufficient 'field_files' provided by index attach launch"
                    " for field %d in parent task %s (UID %lld). Launcher has "
                    "%zd logical regions but only %zd field file names.",
                    it->first, parent_ctx->get_task_name(),
                    parent_ctx->get_unique_id(), launcher.handles.size(), 
                    it->second.size())
              requirement.add_field(it->first);
              field_map[it->first] = strdup(it->second[index]);
            }
            file_mode = launcher.mode;
            layout_constraint_set.ordering_constraint = ordering_constraint;
            break;
          }
        case LEGION_EXTERNAL_INSTANCE:
          {
            layout_constraint_set = launcher.constraints;  
            if (index >= launcher.pointers.size())
              REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_ATTACH,
                  "Insufficient 'pointers' provided by index attach launch "
                  "in parent task %s (UID %lld). Launcher has %zd logical "
                  "regions but only %zd pointers names.", 
                  parent_ctx->get_task_name(), parent_ctx->get_unique_id(),
                  launcher.handles.size(), launcher.pointers.size())
            layout_constraint_set.pointer_constraint = launcher.pointers[index];
            // Construct the region requirement for this task
            requirement = RegionRequirement(launcher.handles[index], 
                LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, launcher.parent);
            const std::set<FieldID> &fields = launcher.privilege_fields;
            for (std::set<FieldID>::const_iterator it = 
                  fields.begin(); it != fields.end(); it++)
              requirement.add_field(*it);
            break;
          }
        default:
          assert(false); // should never get here
      }
      region = PhysicalRegion(new PhysicalRegionImpl(requirement,
            mapped_event, completion_event, ApUserEvent::NO_AP_USER_EVENT, 
            false/*mapped*/, ctx, 0/*map id*/, 0/*tag*/, false/*leaf*/, 
            false/*virtual mapped*/, runtime)); 
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_index_point(owner->get_unique_op_id(), 
                                   unique_op_id, point);
        log_requirement();
      }
      return region.impl;
    }

    //--------------------------------------------------------------------------
    RtEvent PointAttachOp::check_for_coregions(void)
    //--------------------------------------------------------------------------
    {
      owner->find_coregions(this, requirement.region, 
                  external_instances, attached_event); 
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void PointAttachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Resolve our speculation then do the base case
      resolve_speculation();
      AttachOp::trigger_ready();
    }

    //--------------------------------------------------------------------------
    void PointAttachOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(false/*deactivate*/);
      // Tell our owner that we are done, they will do the deactivate
      owner->handle_point_commit();
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
    Future DetachOp::initialize_detach(InnerContext *ctx, PhysicalRegion region,
                  const bool flsh, const bool unordered, const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, !unordered/*track*/, 0, provenance);
      flush = flsh;
      // Get a reference to the region to keep it alive
      this->region = region; 
      requirement = region.impl->get_requirement();
      // Make sure that the privileges are read-write so that we wait for
      // all prior users of this particular region unless we're not flushing
      // in which case we can make the privileges write-discard
      requirement.privilege = flush ? LEGION_READ_WRITE : LEGION_WRITE_DISCARD;
      requirement.prop = LEGION_EXCLUSIVE;
      // Create the future result that we will complete when we're done
      result = Future(new FutureImpl(parent_ctx, runtime, true/*register*/,
                  runtime->get_available_distributed_id(),
                  runtime->address_space, get_completion_event(), this));
      if (runtime->legion_spy_enabled)
        LegionSpy::log_detach_operation(parent_ctx->get_unique_id(),
                            unique_op_id, context_index, unordered);
      return result;
    }

    //--------------------------------------------------------------------------
    void DetachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_detach();
    }

    //--------------------------------------------------------------------------
    void DetachOp::activate_detach(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      flush = true;
    }

    //--------------------------------------------------------------------------
    void DetachOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_detach();
      runtime->free_detach_op(this);
    }

    //--------------------------------------------------------------------------
    void DetachOp::deactivate_detach(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      region = PhysicalRegion();
      privilege_path.clear();
      version_info.clear();
      map_applied_conditions.clear();
      result = Future(); // clear any references on the future
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
      if (runtime->legion_spy_enabled)
        log_requirement();
    }

    //--------------------------------------------------------------------------
    void DetachOp::log_requirement(void)
    //--------------------------------------------------------------------------
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

    //--------------------------------------------------------------------------
    void DetachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Before we do our dependence analysis, we can remove the 
      // restricted coherence on the logical region
      ProjectionInfo projection_info;
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement, 
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions);
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement, 
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
      // Logical dependence analysis should guarantee that we are valid
      // by the time we get here because the inline mapping/attach op
      // that made the physical region should have mapped before us
#ifdef DEBGU_LEGION
      assert(region.impl->get_mapped_event().has_triggered());
#endif
      // Now we can get the reference we need for the detach operation
      InstanceSet references;
      region.impl->get_references(references);
#ifdef DEBUG_LEGION
      assert(references.size() == 1);
#endif
      const InstanceRef &reference = references[0]; 
      // Add a valid reference to the instances to act as an acquire to keep
      // them valid through the end of mapping them, we'll release the valid
      // references when we are done mapping
      WrapperReferenceMutator mutator(map_applied_conditions);
      PhysicalManager *manager = reference.get_physical_manager();
      if (!manager->is_external_instance())
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_DETACH_OPERATION,
                      "Illegal detach operation (ID %lld) performed in "
                      "task %s (ID %lld). Detach was performed on an region "
                      "that had not previously been attached.",
                      get_unique_op_id(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
#ifdef DEBUG_LEGION
      assert(!manager->is_reduction_manager()); 
#endif
      manager->add_base_valid_ref(MAPPING_ACQUIRE_REF, &mutator);
      const PhysicalTraceInfo trace_info(this, 0/*idx*/, true/*init*/);
      // If we need to flush then register this operation to bring the
      // data that it has up to date, use READ-ONLY privileges since we're
      // not going to invalidate the existing data. Don't register ourselves
      // either since we'll get all the preconditions for detaching it
      // as part of the detach_external call
      ApEvent effects_done;
      if (flush)
      {
        requirement.privilege = LEGION_READ_ONLY;
        effects_done = 
          runtime->forest->physical_perform_updates_and_registration(
                                                  requirement, version_info,
                                                  this, 0/*idx*/, 
                                                  ApEvent::NO_AP_EVENT, 
                                                  ApEvent::NO_AP_EVENT,
                                                  references, trace_info,
                                                  map_applied_conditions,
#ifdef DEBUG_LEGION
                                                  get_logging_name(),
                                                  unique_op_id,
#endif
                                                  false/*record valid*/,
                                                  false/*check initialized*/);
        requirement.privilege = LEGION_READ_WRITE;
      }
      InnerContext *context = find_physical_context(0/*index*/, requirement);
      std::vector<InstanceView*> external_views;
      context->convert_target_views(references, external_views);
#ifdef DEBUG_LEGION
      assert(external_views.size() == 1);
#endif
      InstanceView *ext_view = external_views[0];
      ApEvent detach_event = 
        runtime->forest->detach_external(requirement, this, 0/*idx*/,
                                         version_info, ext_view,
                                         trace_info, map_applied_conditions);
      if (detach_event.exists() && effects_done.exists())
        detach_event = 
          Runtime::merge_events(&trace_info, detach_event, effects_done);
      else if (effects_done.exists())
        detach_event = effects_done;
      if (runtime->legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, parent_ctx,0/*idx*/,
                                              requirement, references);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, detach_event,
                                        completion_event);
#endif
      }
      // Also tell the runtime to detach the external instance from memory
      // This has to be done before we can consider this mapped
      RtEvent detached_event = manager->detach_external_instance();
      if (detached_event.exists())
        map_applied_conditions.insert(detached_event);
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping(); 
      if (!request_early_complete(detach_event))
        complete_execution(Runtime::protect_event(detach_event));
      else
        complete_execution();
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
    void DetachOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      // Can be NULL if this is a PointDetachOp
      if (result.impl != NULL)
        result.impl->set_result(NULL, 0, true/*own*/);
      InstanceSet references;
      region.impl->get_references(references);
#ifdef DEBUG_LEGION
      assert(references.size() == 1);
#endif
      const InstanceRef &reference = references[0];
      PhysicalManager *manager = reference.get_physical_manager();
      // We can remove the acquire reference that we added after we're mapped
      if (manager->remove_base_valid_ref(MAPPING_ACQUIRE_REF))
        delete manager;
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void DetachOp::select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index == 0);
#endif
      // TODO: invoke the mapper
    }

    //--------------------------------------------------------------------------
    void DetachOp::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &reqeusts, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void DetachOp::pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                        std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
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
    // Index Detach Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexDetachOp::IndexDetachOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexDetachOp::IndexDetachOp(const IndexDetachOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndexDetachOp::~IndexDetachOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexDetachOp& IndexDetachOp::operator=(const IndexDetachOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_index_detach();
    }
    
    //--------------------------------------------------------------------------
    void IndexDetachOp::activate_index_detach(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      launch_space = NULL;
      points_completed = 0;
      points_committed = 0;
      complete_request = false;
      commit_request = false;
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_index_detach();
      runtime->free_index_detach_op(this);
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::deactivate_index_detach(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      resources = ExternalResources();
      privilege_path.clear();
      // We can deactivate all of our point operations
      for (std::vector<PointDetachOp*>::const_iterator it =
            points.begin(); it != points.end(); it++)
        (*it)->deactivate();
      points.clear();
      map_applied_conditions.clear();
      result = Future();
    }

    //--------------------------------------------------------------------------
    const char* IndexDetachOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DETACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind IndexDetachOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DETACH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t IndexDetachOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    Future IndexDetachOp::initialize_detach(InnerContext *ctx,
                                   LogicalRegion parent,
                                   RegionTreeNode *upper_bound,
                                   IndexSpaceNode *launch_bounds,
                                   ExternalResourcesImpl *external,
                                   const std::vector<FieldID> &privilege_fields,
                                   const std::vector<PhysicalRegion> &regions,
                                   bool flush, bool unordered,
                                   const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, !unordered/*track*/, 0, provenance);
      // Construct the region requirement
      // We'll get the projection later after we know its been written
      // in the dependence analysis stage of the pipeline
      if (upper_bound->is_region())
        requirement = RegionRequirement(upper_bound->as_region_node()->handle,
            0/*fake*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, parent);
      else
        requirement = 
          RegionRequirement(upper_bound->as_partition_node()->handle,
            0/*fake*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, parent);
      for (std::vector<FieldID>::const_iterator it =
            privilege_fields.begin(); it != privilege_fields.end(); it++)
        requirement.add_field(*it);
      resources = ExternalResources(external);
      launch_space = launch_bounds;
      points.reserve(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        PointDetachOp *point = runtime->get_available_point_detach_op();
        const DomainPoint index_point = Point<1>(idx);
        point->initialize_detach(this, ctx, regions[idx], index_point, flush); 
        points.push_back(point);
      }
      // Create the future result that we will complete when we're done
      result = Future(new FutureImpl(parent_ctx, runtime, true/*register*/,
                  runtime->get_available_distributed_id(),
                  runtime->address_space, get_completion_event(), this));
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_detach_operation(parent_ctx->get_unique_id(),
                            unique_op_id, context_index, unordered);
        runtime->forest->log_launch_space(launch_space->handle, unique_op_id);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent index
      compute_parent_index();
      // Promote a singular region requirement up to a projection
      if (requirement.handle_type == LEGION_SINGULAR_PROJECTION)
      {
        requirement.handle_type = LEGION_REGION_PROJECTION;
        requirement.projection = 0;
      }
      initialize_privilege_path(privilege_path, requirement);
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Get the projection ID which we know is valid on the external resources
      requirement.projection = resources.impl->get_projection();
      if (runtime->legion_spy_enabled)
        log_requirement();
      ProjectionInfo projection_info(runtime, requirement, launch_space);
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/, 
                                                   requirement,
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions);
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> mapped_preconditions;
      std::set<ApEvent> executed_preconditions;
      for (std::vector<PointDetachOp*>::const_iterator it =
            points.begin(); it != points.end(); it++)
      {
        mapped_preconditions.insert((*it)->get_mapped_event());
        executed_preconditions.insert((*it)->get_completion_event());
        (*it)->trigger_ready();
      }
#ifdef LEGION_SPY
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      completion_event);
#endif
      // Record that we are mapped when all our points are mapped
      // and we are executed when all our points are executed
      complete_mapping(Runtime::merge_events(mapped_preconditions));
      ApEvent done = Runtime::merge_events(NULL, executed_preconditions);
      if (!request_early_complete(done))
        complete_execution(Runtime::protect_event(done));
      else
        complete_execution();
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      bool complete_now = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(!complete_request);
#endif
        complete_request = true;
        complete_now = (points.size() == points_completed);
      }
      if (complete_now)
        complete_detach();
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::handle_point_complete(void)
    //--------------------------------------------------------------------------
    {
      bool complete_now = false;
      {
        AutoLock o_lock(op_lock);
        points_completed++;
        complete_now = complete_request && (points.size() == points_completed);
      }
      if (complete_now)
        complete_detach();
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::complete_detach(void)
    //--------------------------------------------------------------------------
    {
      result.impl->set_result(NULL, 0, true/*own*/);
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::trigger_commit(void)
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
    void IndexDetachOp::handle_point_commit(void)
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

    //--------------------------------------------------------------------------
    unsigned IndexDetachOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::compute_parent_index(void)
    //--------------------------------------------------------------------------
    {
      int parent_index = parent_ctx->find_parent_region_req(requirement);
      if (parent_index < 0)
        REPORT_LEGION_ERROR(ERROR_PARENT_TASK_ATTACH,
                               "Parent task %s (ID %lld) of index attach "
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
    void IndexDetachOp::log_requirement(void)
    //--------------------------------------------------------------------------
    {
      if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
        LegionSpy::log_logical_requirement(unique_op_id,
                                       0/*index*/, false/*region*/,
                                       requirement.partition.index_partition.id,
                                       requirement.partition.field_space.id,
                                       requirement.partition.tree_id,
                                       requirement.privilege,
                                       requirement.prop,
                                       requirement.redop,
                                       requirement.parent.index_space.id);
      else
        LegionSpy::log_logical_requirement(unique_op_id, 0/*index*/,
                                           true/*region*/,
                                           requirement.region.index_space.id,
                                           requirement.region.field_space.id,
                                           requirement.region.tree_id,
                                           requirement.privilege,
                                           requirement.prop,
                                           requirement.redop,
                                           requirement.parent.index_space.id);
      LegionSpy::log_requirement_projection(unique_op_id, 0/*index*/,
                                            requirement.projection);
      LegionSpy::log_requirement_fields(unique_op_id, 0/*index*/,
                                        requirement.privilege_fields);
    }

    ///////////////////////////////////////////////////////////// 
    // Point Detach Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointDetachOp::PointDetachOp(Runtime *rt)
      : DetachOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointDetachOp::PointDetachOp(const PointDetachOp &rhs)
      : DetachOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PointDetachOp::~PointDetachOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointDetachOp& PointDetachOp::operator=(const PointDetachOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PointDetachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_detach();
      owner = NULL;
    }

    //--------------------------------------------------------------------------
    void PointDetachOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_detach();
      runtime->free_point_detach_op(this);
    }

    //--------------------------------------------------------------------------
    void PointDetachOp::initialize_detach(IndexDetachOp *own, InnerContext *ctx,
              const PhysicalRegion &region, const DomainPoint &point, bool flsh)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, false/*track*/, 1/*regions*/,
          own->get_provenance());
      index_point = point;
      owner = own;
      flush = flsh;
      context_index = own->get_ctx_index();
      // Get a reference to the region to keep it alive
      this->region = region; 
      requirement = region.impl->get_requirement();
      // Make sure that the privileges are read-write so that we wait for
      // all prior users of this particular region
      requirement.privilege = LEGION_READ_WRITE;
      requirement.prop = LEGION_EXCLUSIVE;
      // No need for a future here
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_index_point(owner->get_unique_op_id(), 
                                   unique_op_id, point);
        log_requirement();
      }
    }

    //--------------------------------------------------------------------------
    void PointDetachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Resolve our speculation then do the base case
      resolve_speculation();
      DetachOp::trigger_ready();
    }

    //--------------------------------------------------------------------------
    void PointDetachOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      owner->handle_point_complete();
      DetachOp::trigger_complete();
    }

    //--------------------------------------------------------------------------
    void PointDetachOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(false/*deactivate*/);
      // Tell our owner that we are done, they will do the deactivate
      owner->handle_point_commit(); 
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
    Future TimingOp::initialize(InnerContext *ctx,
                                const TimingLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/,
          launcher.provenance.c_str());
      measurement = launcher.measurement;
      // Only allow non-empty futures 
      if (!launcher.preconditions.empty())
      {
        for (std::set<Future>::const_iterator it =
              launcher.preconditions.begin(); it != 
              launcher.preconditions.end(); it++)
          if (it->impl != NULL)
            preconditions.insert(*it);
      }
      result = Future(new FutureImpl(parent_ctx, runtime, true/*register*/,
                  runtime->get_available_distributed_id(),
                  runtime->address_space, get_completion_event(), this));
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_timing_operation(ctx->get_unique_id(),
                                        unique_op_id, context_index);
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
        it->impl->register_dependence(this);
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
        const ApEvent ready = it->impl->get_ready_event();
        if (ready.exists())
          pre_events.insert(ready);
      }
      // Also make sure we wait for any execution fences that we have
      if (execution_fence_event.exists())
        pre_events.insert(execution_fence_event);
      RtEvent wait_on;
      if (!pre_events.empty())
        wait_on = Runtime::protect_event(
            Runtime::merge_events(NULL, pre_events));
      if (wait_on.exists() && !wait_on.has_triggered())
        parent_ctx->add_to_trigger_execution_queue(this, wait_on);
      else
        trigger_execution();
    }

    //--------------------------------------------------------------------------
    void TimingOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      switch (measurement)
      {
        case LEGION_MEASURE_SECONDS:
          {
            double value = Realm::Clock::current_time();
            result.impl->set_result(&value, sizeof(value), false);
            break;
          }
        case LEGION_MEASURE_MICRO_SECONDS:
          {
            long long value = Realm::Clock::current_time_in_microseconds();
            result.impl->set_result(&value, sizeof(value), false);
            break;
          }
        case LEGION_MEASURE_NANO_SECONDS:
          {
            long long value = Realm::Clock::current_time_in_nanoseconds();
            result.impl->set_result(&value, sizeof(value), false);
            break;
          }
        default:
          assert(false); // should never get here
      }
#ifdef LEGION_SPY
      // Still have to do this call to let Legion Spy know we're done
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      ApEvent::NO_AP_EVENT);
#endif
      // Complete the future
      complete_execution();
    }

    ///////////////////////////////////////////////////////////// 
    // Tunable Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TunableOp::TunableOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TunableOp::TunableOp(const TunableOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TunableOp::~TunableOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TunableOp& TunableOp::operator=(const TunableOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TunableOp::activate_tunable(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      tunable_id = 0;
      mapper_id = 0;
      tag = 0;
      arg = NULL;
      argsize = 0;
      tunable_index = 0;
    }

    //--------------------------------------------------------------------------
    void TunableOp::deactivate_tunable(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      if (arg != NULL)
        free(arg);
      result = Future();
      futures.clear();
    }

    //--------------------------------------------------------------------------
    Future TunableOp::initialize(InnerContext *ctx, 
                                 const TunableLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/,
          launcher.provenance.c_str());
      tunable_id = launcher.tunable;
      mapper_id = launcher.mapper;
      tag = launcher.tag;
      futures = launcher.futures;
      argsize = launcher.arg.get_size();
      if (argsize > 0)
      {
        arg = malloc(argsize);
        memcpy(arg, launcher.arg.get_ptr(), argsize);
      }
      result = Future(new FutureImpl(parent_ctx, runtime, true/*register*/,
            runtime->get_available_distributed_id(),
            runtime->address_space, get_completion_event(), this));
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_tunable_operation(ctx->get_unique_id(),
                                         unique_op_id, context_index);
        const DomainPoint empty_point;
        LegionSpy::log_future_creation(unique_op_id,
            result.impl->get_ready_event(), empty_point);
        tunable_index = parent_ctx->get_tunable_index();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void TunableOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_tunable();
    }

    //--------------------------------------------------------------------------
    void TunableOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_tunable();
      runtime->free_tunable_op(this);
    }

    //--------------------------------------------------------------------------
    const char* TunableOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TUNABLE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TunableOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TUNABLE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TunableOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      for (std::vector<Future>::const_iterator it =
            futures.begin(); it != futures.end(); it++)
        it->impl->register_dependence(this);
    }

    //--------------------------------------------------------------------------
    void TunableOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      complete_mapping();
      std::set<ApEvent> pre_events;
      for (std::vector<Future>::const_iterator it =
            futures.begin(); it != futures.end(); it++)
      {
        const ApEvent ready = it->impl->get_ready_event();
        if (ready.exists())
          pre_events.insert(ready);
      }
      // Also make sure we wait for any execution fences that we have
      if (execution_fence_event.exists())
        pre_events.insert(execution_fence_event);
      RtEvent ready;
      if (!pre_events.empty())
        ready = Runtime::protect_event(
            Runtime::merge_events(NULL, pre_events));
      if (ready.exists() && !ready.has_triggered())
        parent_ctx->add_to_trigger_execution_queue(this, ready);
      else
        trigger_execution();
    }

    //--------------------------------------------------------------------------
    void TunableOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      MapperManager *mapper =
        runtime->find_mapper(parent_ctx->get_executing_processor(), mapper_id);
      Mapper::SelectTunableInput input;
      Mapper::SelectTunableOutput output;
      input.tunable_id = tunable_id;
      input.mapping_tag = tag;
      input.futures = futures;
      input.args = arg;
      input.size = argsize;
      output.value = NULL;
      output.size = 0;
      output.take_ownership = true;
      mapper->invoke_select_tunable_value(parent_ctx->get_owner_task(), 
                                          &input, &output);
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_tunable_value(parent_ctx->get_unique_id(), 
                        tunable_index, output.value, output.size);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                        completion_event); 
#endif
      }
      // Set and complete the future
      result.impl->set_result(output.value, output.size, output.take_ownership);
      complete_execution();
    }

    ///////////////////////////////////////////////////////////// 
    // All Reduce Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AllReduceOp::AllReduceOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AllReduceOp::AllReduceOp(const AllReduceOp &rhs)
      : Operation(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    AllReduceOp::~AllReduceOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AllReduceOp& AllReduceOp::operator=(const AllReduceOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    Future AllReduceOp::initialize(InnerContext *ctx, const FutureMap &fm, 
                ReductionOpID redop_id, bool is_deterministic, const char *prov)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/, 0/*regions*/, prov);
      future_map = fm;
      redop = runtime->get_reduction(redop_id);
      result = Future(new FutureImpl(parent_ctx, runtime, true/*register*/,
                  runtime->get_available_distributed_id(),
                  runtime->address_space, get_completion_event(), this));
      deterministic = is_deterministic;
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_all_reduce_operation(ctx->get_unique_id(),
                                            unique_op_id, context_index);
        const DomainPoint empty_point;
        LegionSpy::log_future_creation(unique_op_id,
            result.impl->get_ready_event(), empty_point);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation(); 
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      future_map = FutureMap();
      result = Future();
      runtime->free_all_reduce_op(this);
    }

    //--------------------------------------------------------------------------
    const char* AllReduceOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[ALL_REDUCE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind AllReduceOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ALL_REDUCE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      future_map.impl->register_dependence(this);
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      complete_mapping();
      const RtEvent ready = future_map.impl->get_ready_event();
      if (ready.exists() && !ready.has_triggered())
        parent_ctx->add_to_trigger_execution_queue(this, ready);
      else
        trigger_execution();
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      std::map<DomainPoint,Future> futures;
      future_map.impl->get_all_futures(futures);
      void *result_buffer = malloc(redop->sizeof_rhs);
      memcpy(result_buffer, redop->identity, redop->sizeof_rhs);
      for (std::map<DomainPoint,Future>::const_iterator it = 
            futures.begin(); it != futures.end(); it++)
      {
        FutureImpl *impl = it->second.impl;
        const size_t future_size = impl->get_untyped_size(true/*internal*/);
        if (future_size != redop->sizeof_rhs)
          REPORT_LEGION_ERROR(ERROR_FUTURE_MAP_REDOP_TYPE_MISMATCH,
              "Future in future map reduction in task %s (UID %lld) does not "
              "have the right input size for the given reduction operator. "
              "Future has size %zd bytes but reduction operator expects "
              "RHS inputs of %zd bytes.", parent_ctx->get_task_name(),
              parent_ctx->get_unique_id(), future_size, redop->sizeof_rhs)
        const void *data = impl->get_untyped_result(true,NULL,true/*internal*/);
        (redop->cpu_fold_excl_fn)(result_buffer, 0, data, 0,
                                  1, redop->userdata);
      }
      if (runtime->legion_spy_enabled)
      {
        for (std::map<DomainPoint,Future>::const_iterator it = 
              futures.begin(); it != futures.end(); it++)
        {
          FutureImpl *impl = it->second.impl;
          const ApEvent ready_event = impl->get_ready_event();
          if (ready_event.exists())
            LegionSpy::log_future_use(unique_op_id, ready_event);
        }
      }
      // Tell the future about the final result which it will own
      result.impl->set_result(result_buffer, redop->sizeof_rhs, true/*own*/);
#ifdef LEGION_SPY
      // Still have to do this call to let Legion Spy know we're done
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      ApEvent::NO_AP_EVENT);
#endif
      // Mark that we are done executing which will complete the future
      // as soon as this operation is complete
      complete_execution();
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteOp::RemoteOp(Runtime *rt, Operation *ptr, AddressSpaceID src)
      : Operation(rt), remote_ptr(ptr), source(src), mapper(NULL),
        profiling_reports(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteOp::RemoteOp(const RemoteOp &rhs)
      : Operation(rhs), remote_ptr(NULL), source(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteOp::~RemoteOp(void)
    //--------------------------------------------------------------------------
    {
      if (!profiling_requests.empty())
      {
#ifdef DEBUG_LEGION
        assert(profiling_response.exists());
#endif
        if (profiling_reports.load() > 0)
        {
          Serializer rez;
          rez.serialize(remote_ptr);
          rez.serialize(profiling_reports.load());
          rez.serialize(profiling_response);
          runtime->send_remote_op_profiling_count_update(source, rez);
        }
        else
          Runtime::trigger_event(profiling_response);
      }
    }

    //--------------------------------------------------------------------------
    RemoteOp& RemoteOp::operator=(const RemoteOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    } 

    //--------------------------------------------------------------------------
    void RemoteOp::defer_deletion(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      DeferRemoteOpDeletionArgs args(this);
      runtime->issue_runtime_meta_task(args, 
          LG_THROUGHPUT_WORK_PRIORITY, precondition);
    }

    //--------------------------------------------------------------------------
    void RemoteOp::pack_remote_base(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent_ctx != NULL);
#endif
      rez.serialize(get_operation_kind());
      rez.serialize(remote_ptr);
      rez.serialize(source);
      rez.serialize(unique_op_id);
      rez.serialize(parent_ctx->get_unique_id());
      rez.serialize<bool>(tracing);
    }

    //--------------------------------------------------------------------------
    void RemoteOp::unpack_remote_base(Deserializer &derez, Runtime *runtime,
                                      std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(unique_op_id);
      UniqueID parent_uid;
      derez.deserialize(parent_uid);
      RtEvent ready;
      parent_ctx = runtime->find_context(parent_uid, false, &ready);
      if (ready.exists())
        ready_events.insert(ready);
      derez.deserialize<bool>(tracing);
    }

    //--------------------------------------------------------------------------
    void RemoteOp::pack_profiling_requests(Serializer &rez,
                                           std::set<RtEvent> &applied) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(profiling_requests.size());
      if (profiling_requests.empty())
        return;
      for (unsigned idx = 0; idx < profiling_requests.size(); idx++)
        rez.serialize(profiling_requests[idx]);
      rez.serialize(profiling_priority);
      rez.serialize(profiling_target);
      // Send a message to the owner with an update for the extra counts
      const RtUserEvent done_event = Runtime::create_rt_user_event();
      rez.serialize<RtEvent>(done_event);
      applied.insert(done_event);
    }

    //--------------------------------------------------------------------------
    void RemoteOp::unpack_profiling_requests(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_requests;
      derez.deserialize(num_requests);
      if (num_requests == 0)
        return;
      profiling_requests.resize(num_requests);
      for (unsigned idx = 0; idx < num_requests; idx++)
        derez.deserialize(profiling_requests[idx]);
      derez.deserialize(profiling_priority);
      derez.deserialize(profiling_target);
      derez.deserialize(profiling_response);
#ifdef DEBUG_LEGION
      assert(profiling_response.exists());
#endif
    }

    //--------------------------------------------------------------------------
    void RemoteOp::activate(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteOp::report_uninitialized_usage(const unsigned index,
                                              LogicalRegion handle,
                                              const RegionUsage usage,
                                              const char *field_string,
                                              RtUserEvent reported)
    //--------------------------------------------------------------------------
    {
      if (source == runtime->address_space)
      {
        // If we're on the owner node we can just do this
        remote_ptr->report_uninitialized_usage(index, handle, usage,
                                               field_string, reported);
        return;
      }
      // Ship this back to the owner node to report it there 
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_ptr);
        rez.serialize(reported);
        rez.serialize(index);
        rez.serialize(handle);
        rez.serialize(usage);
        // Include the null terminator character
        const size_t length = strlen(field_string) + 1;
        rez.serialize<size_t>(length);
        rez.serialize(field_string, length);
      }
      // Send the message and wait for it to be received
      runtime->send_remote_op_report_uninitialized(source, rez);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,unsigned>*
                                      RemoteOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      // We shouldn't actually be acquiring anything here so we just
      // need to make sure that we don't assert
      return NULL;
    }

    //--------------------------------------------------------------------------
    void RemoteOp::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &requests, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return;
      OpProfilingResponse response(remote_ptr, info.index, info.dst_index,fill);
      // Send the result back to the owner node
      Realm::ProfilingRequest &request = requests.add_request( 
          profiling_target, LG_LEGION_PROFILING_ID, 
          &response, sizeof(response), profiling_priority);
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      profiling_reports.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteOp::handle_deferred_deletion(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferRemoteOpDeletionArgs *dargs = 
        (const DeferRemoteOpDeletionArgs*)args;
      delete dargs->op;
    }

    //--------------------------------------------------------------------------
    /*static*/ RemoteOp* RemoteOp::unpack_remote_operation(Deserializer &derez,
                              Runtime *runtime, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      Operation::OpKind kind;
      derez.deserialize(kind);
      Operation *remote_ptr;
      derez.deserialize(remote_ptr);
      AddressSpaceID source;
      derez.deserialize(source);
      RemoteOp *result = NULL;
      switch (kind)
      {
        case TASK_OP_KIND:
          {
            result = new RemoteTaskOp(runtime, remote_ptr, source);
            break;
          }
        case MAP_OP_KIND:
          {
            result = new RemoteMapOp(runtime, remote_ptr, source);
            break;
          }
        case COPY_OP_KIND:
          {
            result = new RemoteCopyOp(runtime, remote_ptr, source);
            break;
          }
        case POST_CLOSE_OP_KIND:
          {
            result = new RemoteCloseOp(runtime, remote_ptr, source);
            break;
          }
        case ACQUIRE_OP_KIND:
          {
            result = new RemoteAcquireOp(runtime, remote_ptr, source);
            break;
          }
        case RELEASE_OP_KIND:
          {
            result = new RemoteReleaseOp(runtime, remote_ptr, source);
            break;
          }
        case DEPENDENT_PARTITION_OP_KIND:
          {
            result = new RemotePartitionOp(runtime, remote_ptr, source);
            break;
          }
        case FILL_OP_KIND:
          {
            result = new RemoteFillOp(runtime, remote_ptr, source);
            break;
          }
        case ATTACH_OP_KIND:
          {
            result = new RemoteAttachOp(runtime, remote_ptr, source);
            break;
          }
        case DETACH_OP_KIND:
          {
            result = new RemoteDetachOp(runtime, remote_ptr, source);
            break;
          }
        case DELETION_OP_KIND:
          {
            result = new RemoteDeletionOp(runtime, remote_ptr, source);
            break;
          }
        case TRACE_REPLAY_OP_KIND:
          {
            result = new RemoteReplayOp(runtime, remote_ptr, source);
            break;
          }
        case TRACE_SUMMARY_OP_KIND:
          {
            result = new RemoteSummaryOp(runtime, remote_ptr, source);
            break;
          }
        default:
          assert(false);
      }
      // Do the rest of the unpack
      result->unpack_remote_base(derez, runtime, ready_events);
      WrapperReferenceMutator mutator(ready_events);
      result->unpack(derez, mutator);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteOp::handle_report_uninitialized(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Operation *op;
      derez.deserialize(op);
      RtUserEvent reported;
      derez.deserialize(reported);
      unsigned index;
      derez.deserialize(index);
      LogicalRegion handle;
      derez.deserialize(handle);
      RegionUsage usage;
      derez.deserialize(usage);
      size_t length;
      derez.deserialize(length);
      const char *field_string = (const char*)derez.get_current_pointer();
      derez.advance_pointer(length);
      op->report_uninitialized_usage(index, handle, usage, 
                                     field_string, reported);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteOp::handle_report_profiling_count_update(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      Operation *op;
      derez.deserialize(op);
      int update_count;
      derez.deserialize(update_count);
      RtUserEvent done_event;
      derez.deserialize(done_event);
#ifdef DEBUG_LEGION
      assert(done_event.exists());
#endif
      op->handle_profiling_update(update_count);
      Runtime::trigger_event(done_event);
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Map Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteMapOp::RemoteMapOp(Runtime *rt, Operation *ptr, AddressSpaceID src)
      : ExternalMapping(), RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteMapOp::RemoteMapOp(const RemoteMapOp &rhs)
      : ExternalMapping(), RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteMapOp::~RemoteMapOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteMapOp& RemoteMapOp::operator=(const RemoteMapOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteMapOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t RemoteMapOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteMapOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteMapOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteMapOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const char* RemoteMapOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[MAP_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemoteMapOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return MAP_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteMapOp::select_sources(const unsigned index,
                                     const InstanceRef &target,
                                     const InstanceSet &sources,
                                     std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      if (source == runtime->address_space)
      {
        // If we're on the owner node we can just do this
        remote_ptr->select_sources(index, target, sources, ranking);
        return;
      }
#ifdef DEBUG_LEGION
      assert(index == 0);
#endif
      Mapper::SelectInlineSrcInput input;
      Mapper::SelectInlineSrcOutput output;
      prepare_for_mapping(sources, input.source_instances); 
      prepare_for_mapping(target, input.target);
      if (mapper == NULL)
        mapper = runtime->find_mapper(map_id);
      mapper->invoke_select_inline_sources(this, &input, &output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    void RemoteMapOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_mapping(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteMapOp::unpack(Deserializer &derez, ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      unpack_external_mapping(derez, runtime);
      unpack_profiling_requests(derez);
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Copy Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteCopyOp::RemoteCopyOp(Runtime *rt, Operation *ptr, AddressSpaceID src)
      : ExternalCopy(), RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteCopyOp::RemoteCopyOp(const RemoteCopyOp &rhs)
      : ExternalCopy(), RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteCopyOp::~RemoteCopyOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteCopyOp& RemoteCopyOp::operator=(const RemoteCopyOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteCopyOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t RemoteCopyOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteCopyOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteCopyOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteCopyOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const char* RemoteCopyOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[COPY_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemoteCopyOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return COPY_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteCopyOp::select_sources(const unsigned index,
                                      const InstanceRef &target,
                                      const InstanceSet &sources,
                                      std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      if (source == runtime->address_space)
      {
        // If we're on the owner node we can just do this
        remote_ptr->select_sources(index, target, sources, ranking);
        return;
      }
      Mapper::SelectCopySrcInput input;
      Mapper::SelectCopySrcOutput output;
      prepare_for_mapping(sources, input.source_instances); 
      prepare_for_mapping(target, input.target);
      input.is_src = false;
      input.is_dst = false;
      input.is_src_indirect = false;
      input.is_dst_indirect = false;
      unsigned mod_index = index;
      if (mod_index < src_requirements.size())
      {
        input.region_req_index = mod_index;
        input.is_src = true;
      }
      else
      {
        mod_index -= src_requirements.size();
        if (mod_index < dst_requirements.size())
        {
          input.region_req_index = mod_index;
          input.is_dst = true;
        }
        else
        {
          mod_index -= dst_requirements.size();
          if (mod_index < src_indirect_requirements.size())
          {
            input.region_req_index = mod_index;
            input.is_src_indirect = true;
          }
          else
          {
            mod_index -= src_indirect_requirements.size();
#ifdef DEBUG_LEGION
            assert(mod_index < dst_indirect_requirements.size());
#endif
            input.is_dst_indirect = true;
          }
        }
      }
      if (mapper == NULL)
        mapper = runtime->find_mapper(map_id);
      mapper->invoke_select_copy_sources(this, &input, &output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    void RemoteCopyOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_copy(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteCopyOp::unpack(Deserializer &derez, ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      unpack_external_copy(derez, runtime);
      unpack_profiling_requests(derez);
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Close Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteCloseOp::RemoteCloseOp(Runtime *rt, Operation *ptr,AddressSpaceID src)
      : ExternalClose(), RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteCloseOp::RemoteCloseOp(const RemoteCloseOp &rhs)
      : ExternalClose(), RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteCloseOp::~RemoteCloseOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteCloseOp& RemoteCloseOp::operator=(const RemoteCloseOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteCloseOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t RemoteCloseOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteCloseOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteCloseOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteCloseOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const char* RemoteCloseOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[POST_CLOSE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemoteCloseOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return POST_CLOSE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteCloseOp::select_sources(const unsigned index,
                                       const InstanceRef &target,
                                       const InstanceSet &sources,
                                       std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      if (source == runtime->address_space)
      {
        // If we're on the owner node we can just do this
        remote_ptr->select_sources(index, target, sources, ranking);
        return;
      }
#ifdef DEBUG_LEGION
      assert(index == 0);
#endif
      Mapper::SelectCloseSrcInput input;
      Mapper::SelectCloseSrcOutput output;
      prepare_for_mapping(sources, input.source_instances); 
      prepare_for_mapping(target, input.target);
      if (mapper == NULL)
        mapper = runtime->find_mapper(map_id);
      mapper->invoke_select_close_sources(this, &input, &output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    void RemoteCloseOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_close(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteCloseOp::unpack(Deserializer &derez, ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      unpack_external_close(derez, runtime);
      unpack_profiling_requests(derez);
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Acquire Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteAcquireOp::RemoteAcquireOp(Runtime *rt, 
                                     Operation *ptr, AddressSpaceID src)
      : ExternalAcquire(), RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteAcquireOp::RemoteAcquireOp(const RemoteAcquireOp &rhs)
      : ExternalAcquire(), RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteAcquireOp::~RemoteAcquireOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteAcquireOp& RemoteAcquireOp::operator=(const RemoteAcquireOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteAcquireOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t RemoteAcquireOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteAcquireOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteAcquireOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteAcquireOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const char* RemoteAcquireOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[ACQUIRE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemoteAcquireOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ACQUIRE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteAcquireOp::select_sources(const unsigned index,
                                         const InstanceRef &target,
                                         const InstanceSet &sources,
                                         std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteAcquireOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_acquire(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteAcquireOp::unpack(Deserializer &derez, ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      unpack_external_acquire(derez, runtime);
      unpack_profiling_requests(derez);
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Release Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteReleaseOp::RemoteReleaseOp(Runtime *rt, 
                                     Operation *ptr, AddressSpaceID src)
      : ExternalRelease(), RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteReleaseOp::RemoteReleaseOp(const RemoteReleaseOp &rhs)
      : ExternalRelease(), RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteReleaseOp::~RemoteReleaseOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteReleaseOp& RemoteReleaseOp::operator=(const RemoteReleaseOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteReleaseOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t RemoteReleaseOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteReleaseOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteReleaseOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteReleaseOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const char* RemoteReleaseOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[RELEASE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemoteReleaseOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return RELEASE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteReleaseOp::select_sources(const unsigned index,
                                         const InstanceRef &target,
                                         const InstanceSet &sources,
                                         std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      if (source == runtime->address_space)
      {
        // If we're on the owner node we can just do this
        remote_ptr->select_sources(index, target, sources, ranking);
        return;
      }
#ifdef DEBUG_LEGION
      assert(index == 0);
#endif
      Mapper::SelectReleaseSrcInput input;
      Mapper::SelectReleaseSrcOutput output;
      prepare_for_mapping(sources, input.source_instances); 
      prepare_for_mapping(target, input.target);
      if (mapper == NULL)
        mapper = runtime->find_mapper(map_id);
      mapper->invoke_select_release_sources(this, &input, &output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    void RemoteReleaseOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_release(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteReleaseOp::unpack(Deserializer &derez, ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      unpack_external_release(derez, runtime);
      unpack_profiling_requests(derez);
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Fill Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteFillOp::RemoteFillOp(Runtime *rt, Operation *ptr, AddressSpaceID src)
      : ExternalFill(), RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteFillOp::RemoteFillOp(const RemoteFillOp &rhs)
      : ExternalFill(), RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteFillOp::~RemoteFillOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteFillOp& RemoteFillOp::operator=(const RemoteFillOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteFillOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t RemoteFillOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteFillOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteFillOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteFillOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const char* RemoteFillOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[FILL_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemoteFillOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return FILL_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteFillOp::select_sources(const unsigned index,
                                      const InstanceRef &target,
                                      const InstanceSet &sources,
                                      std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteFillOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_fill(rez, target);
    }

    //--------------------------------------------------------------------------
    void RemoteFillOp::unpack(Deserializer &derez, ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      unpack_external_fill(derez, runtime);
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Partition Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemotePartitionOp::RemotePartitionOp(Runtime *rt, 
                                     Operation *ptr, AddressSpaceID src)
      : ExternalPartition(), RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemotePartitionOp::RemotePartitionOp(const RemotePartitionOp &rhs)
      : ExternalPartition(), RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemotePartitionOp::~RemotePartitionOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemotePartitionOp& RemotePartitionOp::operator=(
                                                   const RemotePartitionOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemotePartitionOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t RemotePartitionOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemotePartitionOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemotePartitionOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemotePartitionOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    Partition::PartitionKind RemotePartitionOp::get_partition_kind(void) const
    //--------------------------------------------------------------------------
    {
      return part_kind;
    }

    //--------------------------------------------------------------------------
    const char* RemotePartitionOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DEPENDENT_PARTITION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemotePartitionOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DEPENDENT_PARTITION_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemotePartitionOp::select_sources(const unsigned index,
                                           const InstanceRef &target,
                                           const InstanceSet &sources,
                                           std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      if (source == runtime->address_space)
      {
        // If we're on the owner node we can just do this
        remote_ptr->select_sources(index, target, sources, ranking);
        return;
      }
#ifdef DEBUG_LEGION
      assert(index == 0);
#endif
      Mapper::SelectPartitionSrcInput input;
      Mapper::SelectPartitionSrcOutput output;
      prepare_for_mapping(sources, input.source_instances); 
      prepare_for_mapping(target, input.target);
      if (mapper == NULL)
        mapper = runtime->find_mapper(map_id);
      mapper->invoke_select_partition_sources(this, &input, &output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    void RemotePartitionOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_partition(rez, target);
      rez.serialize(part_kind);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemotePartitionOp::unpack(Deserializer &derez,
                                   ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      unpack_external_partition(derez, runtime);
      derez.deserialize(part_kind);
      unpack_profiling_requests(derez);
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Attach Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteAttachOp::RemoteAttachOp(Runtime *rt, 
                                   Operation *ptr, AddressSpaceID src)
      : RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteAttachOp::RemoteAttachOp(const RemoteAttachOp &rhs)
      : RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteAttachOp::~RemoteAttachOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteAttachOp& RemoteAttachOp::operator=(const RemoteAttachOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteAttachOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t RemoteAttachOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteAttachOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteAttachOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const char* RemoteAttachOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[ATTACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemoteAttachOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ATTACH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteAttachOp::select_sources(const unsigned index,
                                        const InstanceRef &target,
                                        const InstanceSet &sources,
                                        std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteAttachOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
    }

    //--------------------------------------------------------------------------
    void RemoteAttachOp::unpack(Deserializer &derez, ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      // Nothing for the moment
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Detach Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteDetachOp::RemoteDetachOp(Runtime *rt, 
                                   Operation *ptr, AddressSpaceID src)
      : RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteDetachOp::RemoteDetachOp(const RemoteDetachOp &rhs)
      : RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteDetachOp::~RemoteDetachOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteDetachOp& RemoteDetachOp::operator=(const RemoteDetachOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteDetachOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t RemoteDetachOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteDetachOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteDetachOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const char* RemoteDetachOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DETACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemoteDetachOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DETACH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteDetachOp::select_sources(const unsigned index,
                                        const InstanceRef &target,
                                        const InstanceSet &sources,
                                        std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      // TODO: invoke the mapper
    }

    //--------------------------------------------------------------------------
    void RemoteDetachOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
    }

    //--------------------------------------------------------------------------
    void RemoteDetachOp::unpack(Deserializer &derez, ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      // Nothing for the moment
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Deletion Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteDeletionOp::RemoteDeletionOp(Runtime *rt, 
                                       Operation *ptr, AddressSpaceID src)
      : RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteDeletionOp::RemoteDeletionOp(const RemoteDeletionOp &rhs)
      : RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteDeletionOp::~RemoteDeletionOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteDeletionOp& RemoteDeletionOp::operator=(const RemoteDeletionOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteDeletionOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t RemoteDeletionOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteDeletionOp::set_context_index(size_t index)
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
    Operation::OpKind RemoteDeletionOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DELETION_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteDeletionOp::select_sources(const unsigned index, 
                                          const InstanceRef &target,
                                          const InstanceSet &sources,
                                          std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteDeletionOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
    }

    //--------------------------------------------------------------------------
    void RemoteDeletionOp::unpack(Deserializer &derez,ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      // Nothing for the moment
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Replay Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteReplayOp::RemoteReplayOp(Runtime *rt,
                                   Operation *ptr, AddressSpaceID src)
      : RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteReplayOp::RemoteReplayOp(const RemoteReplayOp &rhs)
      : RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteReplayOp::~RemoteReplayOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteReplayOp& RemoteReplayOp::operator=(const RemoteReplayOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteReplayOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t RemoteReplayOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteReplayOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteReplayOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const char* RemoteReplayOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_REPLAY_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemoteReplayOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_REPLAY_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteReplayOp::select_sources(const unsigned index, 
                                        const InstanceRef &target,
                                        const InstanceSet &sources,
                                        std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteReplayOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
    }

    //--------------------------------------------------------------------------
    void RemoteReplayOp::unpack(Deserializer &derez, ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      // Nothing for the moment
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Summary Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteSummaryOp::RemoteSummaryOp(Runtime *rt,
                                     Operation *ptr, AddressSpaceID src)
      : RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteSummaryOp::RemoteSummaryOp(const RemoteSummaryOp &rhs)
      : RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteSummaryOp::~RemoteSummaryOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteSummaryOp& RemoteSummaryOp::operator=(const RemoteSummaryOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteSummaryOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t RemoteSummaryOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteSummaryOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteSummaryOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const char* RemoteSummaryOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_SUMMARY_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemoteSummaryOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_SUMMARY_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteSummaryOp::select_sources(const unsigned index, 
                                         const InstanceRef &target,
                                         const InstanceSet &sources,
                                         std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteSummaryOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
    }

    //--------------------------------------------------------------------------
    void RemoteSummaryOp::unpack(Deserializer &derez, ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      // Nothing for the moment
    }
 
  }; // namespace Internal 
}; // namespace Legion 

// EOF

