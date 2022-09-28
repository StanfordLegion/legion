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


#include "legion.h"
#include "legion/legion_ops.h"
#include "legion/legion_spy.h"
#include "legion/legion_trace.h"
#include "legion/legion_tasks.h"
#include "legion/legion_instances.h"
#include "legion/legion_views.h"
#include "legion/legion_context.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Utility functions
    /////////////////////////////////////////////////////////////

    std::ostream& operator<<(std::ostream &out, const TraceLocalID &key)
    {
      out << "(" << key.first << ",";
      if (key.second.dim > 1) out << "(";
      for (int dim = 0; dim < key.second.dim; ++dim)
      {
        if (dim > 0) out << ",";
        out << key.second[dim];
      }
      if (key.second.dim > 1) out << ")";
      out << ")";
      return out;
    }

    std::ostream& operator<<(std::ostream &out,
                             const PhysicalTemplate::Replayable &r)
    {
      if (r.replayable)
        out << "Replayable";
      else
      {
        out << "Non-replayable (" << r.message << ")";
      }
      return out;
    }

    /////////////////////////////////////////////////////////////
    // LegionTrace 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LegionTrace::LegionTrace(InnerContext *c, TraceID t, 
                             bool logical_only, Provenance *p)
      : ctx(c), tid(t), begin_provenance(p), end_provenance(NULL),
        last_memoized(0), physical_op_count(0), blocking_call_observed(false), 
        has_intermediate_ops(false), fixed(false)
    //--------------------------------------------------------------------------
    {
      state.store(LOGICAL_ONLY);
      physical_trace = logical_only ? NULL : 
        new PhysicalTrace(c->owner_task->runtime, this);
      if (begin_provenance != NULL)
        begin_provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    LegionTrace::~LegionTrace(void)
    //--------------------------------------------------------------------------
    {
      if (physical_trace != NULL)
        delete physical_trace;
      if ((begin_provenance != NULL) && begin_provenance->remove_reference())
        delete begin_provenance;
      if ((end_provenance != NULL) && end_provenance->remove_reference())
        delete end_provenance;
    }

    //--------------------------------------------------------------------------
    void LegionTrace::fix_trace(Provenance *provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fixed);
      assert(end_provenance == NULL);
#endif
      fixed = true;
      end_provenance = provenance;
      if (end_provenance != NULL)
        end_provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    void LegionTrace::register_physical_only(Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      std::pair<Operation*,GenerationID> key(op, op->get_generation());
      const unsigned index = operations.size();
      op->set_trace_local_id(index);
      op->add_mapping_reference(key.second);
      operations.push_back(key);
      current_uids[key] = op->get_unique_op_id();
#else
      op->set_trace_local_id(physical_op_count++);
#endif
    }

    //--------------------------------------------------------------------------
    void LegionTrace::replay_aliased_children(
                             std::vector<RegionTreePath> &privilege_paths) const
    //--------------------------------------------------------------------------
    {
      unsigned index = operations.size() - 1;
      std::map<unsigned,LegionVector<AliasChildren> >::const_iterator
        finder = aliased_children.find(index);
      if (finder == aliased_children.end())
        return;
      for (LegionVector<AliasChildren>::const_iterator it = 
            finder->second.begin(); it != finder->second.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->req_index < privilege_paths.size());
#endif
        privilege_paths[it->req_index].record_aliased_children(it->depth,
                                                               it->mask);
      }
    }

    //--------------------------------------------------------------------------
    void LegionTrace::end_trace_execution(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      if (is_replaying())
      {
#ifdef LEGION_SPY
        for (std::vector<std::pair<Operation*,GenerationID> >::const_iterator
              it = operations.begin(); it != operations.end(); it++)
          it->first->remove_mapping_reference(it->second);
        operations.clear();
        current_uids.clear();
#else
#ifdef DEBUG_LEGION
        assert(operations.empty());
#endif
        // Reset the physical op count for the next replay
        physical_op_count = 0;
#endif
        return;
      }

      // Register for this fence on every one of the operations in
      // the trace and then clear out the operations data structure
      for (std::set<std::pair<Operation*,GenerationID> >::iterator it =
           frontiers.begin(); it != frontiers.end(); ++it)
      {
        const std::pair<Operation*,GenerationID> &target = *it;
#ifdef DEBUG_LEGION
        assert(!target.first->is_internal_op());
#endif
        op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
        for (unsigned req_idx = 0; req_idx < num_regions[target]; req_idx++)
        {
          LegionSpy::log_mapping_dependence(
              op->get_context()->get_unique_id(), current_uids[target], req_idx,
              op->get_unique_op_id(), 0, LEGION_TRUE_DEPENDENCE);
        }
#endif
        // Remove any mapping references that we hold
        target.first->remove_mapping_reference(target.second);
      }
      operations.clear();
      last_memoized = 0;
      frontiers.clear();
#ifdef LEGION_SPY
      current_uids.clear();
      num_regions.clear();
#endif
    }

#ifdef LEGION_SPY
    //--------------------------------------------------------------------------
    UniqueID LegionTrace::get_current_uid_by_index(unsigned op_idx) const
    //--------------------------------------------------------------------------
    {
      assert(op_idx < operations.size());
      const std::pair<Operation*,GenerationID> &key = operations[op_idx];
      std::map<std::pair<Operation*,GenerationID>,UniqueID>::const_iterator
        finder = current_uids.find(key);
      assert(finder != current_uids.end());
      return finder->second;
    }
#endif

    //--------------------------------------------------------------------------
    void LegionTrace::invalidate_trace_cache(Operation *invalidator)
    //--------------------------------------------------------------------------
    {
      if (physical_trace == NULL)
        return;
      PhysicalTemplate *current_template = 
        physical_trace->get_current_template();
      if (current_template == NULL)
        return;
      bool execution_fence = false;
      if (invalidator->invalidates_physical_trace_template(execution_fence))
      {
        // Check to see if this is an execution fence or not
        if (execution_fence)
        {
          // If it is an execution fence we need to record the previous
          // templates completion event as a precondition for the fence
#ifdef DEBUG_LEGION
          FenceOp *fence_op = dynamic_cast<FenceOp*>(invalidator);
          assert(fence_op != NULL);
#else
          FenceOp *fence_op = static_cast<FenceOp*>(invalidator);
#endif
          // Record that we had an intermediate execution fence between replays
          // Update the execution event for the trace to be this fence event
          physical_trace->record_intermediate_execution_fence(fence_op);
        }
        else
        {
          physical_trace->clear_cached_template();
          current_template->issue_summary_operations(ctx, invalidator, 
                                                     end_provenance);
          has_intermediate_ops = false;
        }
      }
      else
        has_intermediate_ops = true;
    }

    /////////////////////////////////////////////////////////////
    // StaticTrace
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    StaticTrace::StaticTrace(TraceID t, InnerContext *c, bool logical_only,
                             Provenance *p, const std::set<RegionTreeID> *trees)
      : LegionTrace(c, t, logical_only, p)
    //--------------------------------------------------------------------------
    {
      if (trees != NULL)
        application_trees.insert(trees->begin(), trees->end());
    }
    
    //--------------------------------------------------------------------------
    StaticTrace::StaticTrace(const StaticTrace &rhs)
      : LegionTrace(NULL, 0, true, NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    StaticTrace::~StaticTrace(void)
    //--------------------------------------------------------------------------
    {
      // Remove our mapping references and then clear the operations
      for (std::vector<std::pair<Operation*,GenerationID> >::const_iterator it =
            operations.begin(); it != operations.end(); it++)
        it->first->remove_mapping_reference(it->second);
      operations.clear();
    }

    //--------------------------------------------------------------------------
    StaticTrace& StaticTrace::operator=(const StaticTrace &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool StaticTrace::handles_region_tree(RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      if (application_trees.empty())
        return true;
      return (application_trees.find(tid) != application_trees.end());
    }

    //--------------------------------------------------------------------------
    bool StaticTrace::initialize_op_tracing(Operation *op,
                               const std::vector<StaticDependence> *dependences,
                               const LogicalTraceInfo *trace_info)
    //--------------------------------------------------------------------------
    {
      // If we've already recorded all these, there's no need to do it again
      if (fixed)
        return false;
      // Internal operations get to skip this
      if (op->is_internal_op())
        return false;
      // All other operations have to add something to the list
      if (dependences == NULL)
        static_dependences.resize(static_dependences.size() + 1);
      else // Add it to the list of static dependences
        static_dependences.push_back(*dependences);
      return false;
    }

    //--------------------------------------------------------------------------
    void StaticTrace::register_operation(Operation *op, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      std::pair<Operation*,GenerationID> key(op,gen);
      const unsigned index = operations.size();
      if (!ctx->runtime->no_physical_tracing &&
          op->is_memoizing() && !op->is_internal_op())
      {
        if (index != last_memoized)
          REPORT_LEGION_ERROR(ERROR_INCOMPLETE_PHYSICAL_TRACING,
              "Invalid memoization request. A trace cannot be partially "
              "memoized. Please change the mapper to request memoization "
              "for all the operations in the trace");
        op->set_trace_local_id(index);
        last_memoized = index + 1;
      }

      if (!op->is_internal_op())
      {
        frontiers.insert(key);
        const LegionVector<DependenceRecord> &deps = 
          translate_dependence_records(op, index); 
        operations.push_back(key);
#ifdef LEGION_SPY
        current_uids[key] = op->get_unique_op_id();
        num_regions[key] = op->get_region_count();
#endif
        // Add a mapping reference since people will be 
        // registering dependences
        op->add_mapping_reference(gen);  
        // Then compute all the dependences on this operation from
        // our previous recording of the trace
        for (LegionVector<DependenceRecord>::const_iterator it = 
              deps.begin(); it != deps.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert((it->operation_idx >= 0) &&
                 ((size_t)it->operation_idx < operations.size()));
#endif
          const std::pair<Operation*,GenerationID> &target = 
                                                operations[it->operation_idx];
          std::set<std::pair<Operation*,GenerationID> >::iterator finder =
            frontiers.find(target);
          if (finder != frontiers.end())
          {
            finder->first->remove_mapping_reference(finder->second);
            frontiers.erase(finder);
          }

          if ((it->prev_idx == -1) || (it->next_idx == -1))
          {
            op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
            LegionSpy::log_mapping_dependence(
               op->get_context()->get_unique_id(),
               get_current_uid_by_index(it->operation_idx),
               (it->prev_idx == -1) ? 0 : it->prev_idx,
               op->get_unique_op_id(), 
               (it->next_idx == -1) ? 0 : it->next_idx, LEGION_TRUE_DEPENDENCE);
#endif
          }
          else
          {
            op->register_region_dependence(it->next_idx, target.first,
                                           target.second, it->prev_idx,
                                           it->dtype, it->validates,
                                           it->dependent_mask);
#ifdef LEGION_SPY
            LegionSpy::log_mapping_dependence(
                op->get_context()->get_unique_id(),
                get_current_uid_by_index(it->operation_idx), it->prev_idx,
                op->get_unique_op_id(), it->next_idx, it->dtype);
#endif
          }
        }
      }
      else
      {
        // We already added our creator to the list of operations
        // so the set of dependences is index-1
#ifdef DEBUG_LEGION
        assert(index > 0);
#endif
        const LegionVector<DependenceRecord> &deps = 
          translate_dependence_records(operations[index-1].first, index-1);
        // Special case for internal operations
        // Internal operations need to register transitive dependences
        // on all the other operations with which it interferes.
        // We can get this from the set of operations on which the
        // operation we are currently performing dependence analysis
        // has dependences.
        InternalOp *internal_op = static_cast<InternalOp*>(op);
#ifdef DEBUG_LEGION
        assert(internal_op == dynamic_cast<InternalOp*>(op));
#endif
        int internal_index = internal_op->get_internal_index();
        for (LegionVector<DependenceRecord>::const_iterator it = 
              deps.begin(); it != deps.end(); it++)
        {
          // We only record dependences for this internal operation on
          // the indexes for which this internal operation is being done
          if (internal_index != it->next_idx)
            continue;
#ifdef DEBUG_LEGION
          assert((it->operation_idx >= 0) &&
                 ((size_t)it->operation_idx < operations.size()));
#endif
          const std::pair<Operation*,GenerationID> &target = 
                                                operations[it->operation_idx];
          // If this is the case we can do the normal registration
          if ((it->prev_idx == -1) || (it->next_idx == -1))
          {
            internal_op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
            LegionSpy::log_mapping_dependence(
               op->get_context()->get_unique_id(),
               get_current_uid_by_index(it->operation_idx),
               (it->prev_idx == -1) ? 0 : it->prev_idx,
               op->get_unique_op_id(), 
               (it->next_idx == -1) ? 0 : it->next_idx, LEGION_TRUE_DEPENDENCE);
#endif
          }
          else
          {
            internal_op->record_trace_dependence(target.first, target.second,
                                               it->prev_idx, it->next_idx,
                                               it->dtype, it->dependent_mask);
#ifdef LEGION_SPY
            LegionSpy::log_mapping_dependence(
                internal_op->get_context()->get_unique_id(),
                get_current_uid_by_index(it->operation_idx), it->prev_idx,
                internal_op->get_unique_op_id(), 0, it->dtype);
#endif
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void StaticTrace::record_dependence(
                                     Operation *target, GenerationID target_gen,
                                     Operation *source, GenerationID source_gen)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }
    
    //--------------------------------------------------------------------------
    void StaticTrace::record_region_dependence(
                                    Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    DependenceType dtype, bool validates,
                                    const FieldMask &dependent_mask)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void StaticTrace::record_no_dependence(
                                    Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    const FieldMask &dependent_mask)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void StaticTrace::record_aliased_children(unsigned req_index,unsigned depth,
                                              const FieldMask &aliase_mask)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void StaticTrace::end_trace_capture(void)
    //--------------------------------------------------------------------------
    {
      // Remove mapping fences on the frontiers which haven't been removed yet
      for (std::set<std::pair<Operation*,GenerationID> >::const_iterator it =
            frontiers.begin(); it != frontiers.end(); it++)
        it->first->remove_mapping_reference(it->second);
      operations.clear();
      frontiers.clear();
      last_memoized = 0;
#ifdef LEGION_SPY
      current_uids.clear();
      num_regions.clear();
#endif
    }

#ifdef LEGION_SPY
    //--------------------------------------------------------------------------
    void StaticTrace::perform_logging(
                               UniqueID prev_fence_uid, UniqueID curr_fence_uid)
    //--------------------------------------------------------------------------
    {
      // TODO: Implement this if someone wants to memoize static traces
      assert(false);
    }
#endif

    //--------------------------------------------------------------------------
    const LegionVector<LegionTrace::DependenceRecord>& 
        StaticTrace::translate_dependence_records(Operation *op, unsigned index)
    //--------------------------------------------------------------------------
    {
      // If we already translated it then we are done
      if (index < translated_deps.size())
        return translated_deps[index];
      const unsigned start_idx = translated_deps.size();
      translated_deps.resize(index+1);
      RegionTreeForest *forest = ctx->runtime->forest;
      for (unsigned op_idx = start_idx; op_idx <= index; op_idx++)
      {
        const std::vector<StaticDependence> &static_deps = 
          static_dependences[op_idx];
        LegionVector<DependenceRecord> &translation = 
          translated_deps[op_idx];
        for (std::vector<StaticDependence>::const_iterator it = 
              static_deps.begin(); it != static_deps.end(); it++)
        {
          // Convert the previous offset into an absoluate offset    
          // If the previous offset is larger than the index then 
          // this dependence doesn't matter
          if (it->previous_offset > index)
            continue;
          // Compute the field mask by getting the parent region requirement
          unsigned parent_index = op->find_parent_index(it->current_req_index);
          FieldSpace field_space =  
            ctx->find_logical_region(parent_index).get_field_space();
          const FieldMask dependence_mask = 
            forest->get_node(field_space)->get_field_mask(it->dependent_fields);
          translation.push_back(DependenceRecord(index - it->previous_offset, 
                it->previous_req_index, it->current_req_index,
                it->validates, it->dependence_type, dependence_mask));
        }
      }
      return translated_deps[index];
    }

    /////////////////////////////////////////////////////////////
    // DynamicTrace 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DynamicTrace::DynamicTrace(TraceID t, InnerContext *c, bool logical_only,
                               Provenance *p)
      : LegionTrace(c, t, logical_only, p), tracing(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DynamicTrace::DynamicTrace(const DynamicTrace &rhs)
      : LegionTrace(NULL, 0, true, NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DynamicTrace::~DynamicTrace(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DynamicTrace& DynamicTrace::operator=(const DynamicTrace &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    } 

    //--------------------------------------------------------------------------
    void DynamicTrace::end_trace_capture(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tracing);
#endif
      // We don't record mapping dependences when tracing so we don't need
      // to remove them when we are here
      operations.clear();
      last_memoized = 0;
      op_map.clear();
      internal_dependences.clear();
      tracing = false;
#ifdef LEGION_SPY
      current_uids.clear();
      num_regions.clear();
#endif
    } 

    //--------------------------------------------------------------------------
    bool DynamicTrace::handles_region_tree(RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      // Always handles all of them
      return true;
    }

    //--------------------------------------------------------------------------
    bool DynamicTrace::initialize_op_tracing(Operation *op,
                               const std::vector<StaticDependence> *dependences,
                               const LogicalTraceInfo *trace_info)
    //--------------------------------------------------------------------------
    {
      if (trace_info != NULL) // happens for internal operations
        return !trace_info->already_traced;
      else
        return !is_fixed();
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::register_operation(Operation *op, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      std::pair<Operation*,GenerationID> key(op,gen);
      const unsigned index = operations.size();
      if (!ctx->runtime->no_physical_tracing &&
          op->is_memoizing() && !op->is_internal_op())
      {
        if (index != last_memoized)
        {
          for (unsigned i = 0; i < operations.size(); ++i)
          {
            Operation *op = operations[i].first;
            if (!op->is_internal_op() && op->get_memoizable() == NULL)
              REPORT_LEGION_ERROR(ERROR_PHYSICAL_TRACING_UNSUPPORTED_OP,
                  "Invalid memoization request. Operation of type %s (UID %lld)"
                  " at index %d in trace %d requested memoization, but physical"
                  " tracing does not support this operation type yet.",
                  Operation::get_string_rep(op->get_operation_kind()),
                  op->get_unique_op_id(), i, tid);
          }
          REPORT_LEGION_ERROR(ERROR_INCOMPLETE_PHYSICAL_TRACING,
              "Invalid memoization request. A trace cannot be partially "
              "memoized. Please change the mapper to request memoization "
              "for all the operations in the trace");
        }
        op->set_trace_local_id(index);
        last_memoized = index + 1;
      }
      if (has_physical_trace() &&
          !op->is_internal_op() && op->get_memoizable() == NULL)
        REPORT_LEGION_ERROR(ERROR_PHYSICAL_TRACING_UNSUPPORTED_OP,
            "Invalid memoization request. Operation of type %s (UID %lld) "
            "at index %d in trace %d requested memoization, but physical "
            "tracing does not support this operation type yet.",
            Operation::get_string_rep(op->get_operation_kind()),
            op->get_unique_op_id(), index, tid);

      // Only need to save this in the map if we are not done tracing
      if (tracing)
      {
        // This is the normal case
        if (!op->is_internal_op())
        {
          operations.push_back(key);
          op_map[key] = index;
          // Add a new vector for storing dependences onto the back
          dependences.push_back(LegionVector<DependenceRecord>());
          // Record meta-data about the trace for verifying that
          // it is being replayed correctly
          op_info.push_back(OperationInfo(op));
        }
        else // Otherwise, track internal operations separately
        {
          std::pair<InternalOp*,GenerationID> 
            local_key(static_cast<InternalOp*>(op),gen);
          internal_dependences[local_key] = LegionVector<DependenceRecord>();
        }
      }
      else
      {
        if (!op->is_internal_op())
        {
          frontiers.insert(key);
          // Check for exceeding the trace size
          if (index >= dependences.size())
            REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_RECORDED,
                          "Trace violation! Recorded %zd operations in trace "
                          "%d in task %s (UID %lld) but %d operations have "
                          "now been issued!", dependences.size(), tid,
                          ctx->get_task_name(), ctx->get_unique_id(), index+1)
          // Check to see if the meta-data alignes
          const OperationInfo &info = op_info[index];
          // Check that they are the same kind of operation
          if (info.kind != op->get_operation_kind())
            REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                          "Trace violation! Operation at index %d of trace %d "
                          "in task %s (UID %lld) was recorded as having type "
                          "%s but instead has type %s in replay.",
                          index, tid, ctx->get_task_name(),ctx->get_unique_id(),
                          Operation::get_string_rep(info.kind),
                          Operation::get_string_rep(op->get_operation_kind()))
          // Check that they have the same number of region requirements
          if (info.count != op->get_region_count())
            REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                          "Trace violation! Operation at index %d of trace %d "
                          "in task %s (UID %lld) was recorded as having %d "
                          "regions, but instead has %zd regions in replay.",
                          index, tid, ctx->get_task_name(),
                          ctx->get_unique_id(), info.count,
                          op->get_region_count())
          // If we make it here, everything is good
          const LegionVector<DependenceRecord> &deps = dependences[index];
          operations.push_back(key);
#ifdef LEGION_SPY
          current_uids[key] = op->get_unique_op_id();
          num_regions[key] = op->get_region_count();
#endif
          // Add a mapping reference since people will be 
          // registering dependences
          op->add_mapping_reference(gen);  
          // Then compute all the dependences on this operation from
          // our previous recording of the trace
          for (LegionVector<DependenceRecord>::const_iterator it = 
                deps.begin(); it != deps.end(); it++)
          {
            // Skip any no-dependences since they are still no-deps here
            if (it->dtype == LEGION_NO_DEPENDENCE)
              continue;
#ifdef DEBUG_LEGION
            assert((it->operation_idx >= 0) &&
                   ((size_t)it->operation_idx < operations.size()));
#endif
            const std::pair<Operation*,GenerationID> &target = 
                                                  operations[it->operation_idx];
            std::set<std::pair<Operation*,GenerationID> >::iterator finder =
              frontiers.find(target);
            if (finder != frontiers.end())
            {
              finder->first->remove_mapping_reference(finder->second);
              frontiers.erase(finder);
            }

            if ((it->prev_idx == -1) || (it->next_idx == -1))
            {
              op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
              LegionSpy::log_mapping_dependence(
               op->get_context()->get_unique_id(),
               get_current_uid_by_index(it->operation_idx),
               (it->prev_idx == -1) ? 0 : it->prev_idx,
               op->get_unique_op_id(), 
               (it->next_idx == -1) ? 0 : it->next_idx, LEGION_TRUE_DEPENDENCE);
#endif
            }
            else
            {
              op->register_region_dependence(it->next_idx, target.first,
                                             target.second, it->prev_idx,
                                             it->dtype, it->validates,
                                             it->dependent_mask);
#ifdef LEGION_SPY
              LegionSpy::log_mapping_dependence(
                  op->get_context()->get_unique_id(),
                  get_current_uid_by_index(it->operation_idx), it->prev_idx,
                  op->get_unique_op_id(), it->next_idx, it->dtype);
#endif
            }
          }
        }
        else
        {
          // We already added our creator to the list of operations
          // so the set of dependences is index-1
#ifdef DEBUG_LEGION
          assert(index > 0);
#endif
          const LegionVector<DependenceRecord> &deps = dependences[index-1];
          // Special case for internal operations
          // Internal operations need to register transitive dependences
          // on all the other operations with which it interferes.
          // We can get this from the set of operations on which the
          // operation we are currently performing dependence analysis
          // has dependences.
          InternalOp *internal_op = static_cast<InternalOp*>(op);
#ifdef DEBUG_LEGION
          assert(internal_op == dynamic_cast<InternalOp*>(op));
#endif
          int internal_index = internal_op->get_internal_index();
          for (LegionVector<DependenceRecord>::const_iterator it = 
                deps.begin(); it != deps.end(); it++)
          {
            // We only record dependences for this internal operation on
            // the indexes for which this internal operation is being done
            if (internal_index != it->next_idx)
              continue;
#ifdef DEBUG_LEGION
            assert((it->operation_idx >= 0) &&
                   ((size_t)it->operation_idx < operations.size()));
#endif
            const std::pair<Operation*,GenerationID> &target = 
                                                  operations[it->operation_idx];

            // If this is the case we can do the normal registration
            if ((it->prev_idx == -1) || (it->next_idx == -1))
            {
              internal_op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
              LegionSpy::log_mapping_dependence(
               op->get_context()->get_unique_id(),
               get_current_uid_by_index(it->operation_idx),
               (it->prev_idx == -1) ? 0 : it->prev_idx,
               op->get_unique_op_id(), 
               (it->next_idx == -1) ? 0 : it->next_idx, LEGION_TRUE_DEPENDENCE);
#endif
            }
            else
            {
              // Promote no-dependence cases to full dependences here
              internal_op->record_trace_dependence(target.first, target.second,
                                                   it->prev_idx, it->next_idx,
                                                   LEGION_TRUE_DEPENDENCE, 
                                                   it->dependent_mask);
#ifdef LEGION_SPY
              LegionSpy::log_mapping_dependence(
                  internal_op->get_context()->get_unique_id(),
                  get_current_uid_by_index(it->operation_idx), it->prev_idx,
                  internal_op->get_unique_op_id(), 0, it->dtype);
#endif
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::record_dependence(Operation *target,GenerationID tar_gen,
                                         Operation *source,GenerationID src_gen)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tracing);
      if (!source->is_internal_op())
      {
        assert(operations.back().first == source);
        assert(operations.back().second == src_gen);
      }
#endif
      std::pair<Operation*,GenerationID> target_key(target, tar_gen);
      std::map<std::pair<Operation*,GenerationID>,unsigned>::const_iterator
        finder = op_map.find(target_key);
      // We only need to record it if it falls within our trace
      if (finder != op_map.end())
      {
        // Two cases here
        if (!source->is_internal_op())
        {
          // Normal case
          insert_dependence(DependenceRecord(finder->second));
        }
        else
        {
          // Otherwise this is an internal op so record it special
          // Don't record dependences on our creator
          if (target_key != operations.back())
          {
            std::pair<InternalOp*,GenerationID> 
              src_key(static_cast<InternalOp*>(source), src_gen);
#ifdef DEBUG_LEGION
            assert(internal_dependences.find(src_key) != 
                   internal_dependences.end());
#endif
            insert_dependence(src_key, DependenceRecord(finder->second));
          }
        }
      }
      else if (target->is_internal_op())
      {
        // They shouldn't both be internal operations, if they are, then
        // they should be going through the other path that tracks
        // dependences based on specific region requirements
#ifdef DEBUG_LEGION
        assert(!source->is_internal_op());
#endif
        // First check to see if the internal op is one of ours
        std::pair<InternalOp*,GenerationID> 
          local_key(static_cast<InternalOp*>(target),tar_gen);
        std::map<std::pair<InternalOp*,GenerationID>,
                LegionVector<DependenceRecord>>::const_iterator
          internal_finder = internal_dependences.find(local_key);
        if (internal_finder != internal_dependences.end())
        {
          const LegionVector<DependenceRecord> &internal_deps = 
                                                        internal_finder->second;
          for (LegionVector<DependenceRecord>::const_iterator it = 
                internal_deps.begin(); it != internal_deps.end(); it++)
            insert_dependence(DependenceRecord(it->operation_idx)); 
        }
      }
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::record_region_dependence(Operation *target, 
                                                GenerationID tar_gen,
                                                Operation *source, 
                                                GenerationID src_gen,
                                                unsigned target_idx, 
                                                unsigned source_idx,
                                                DependenceType dtype,
                                                bool validates,
                                                const FieldMask &dep_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tracing);
      if (!source->is_internal_op())
      {
        assert(operations.back().first == source);
        assert(operations.back().second == src_gen);
      }
#endif
      std::pair<Operation*,GenerationID> target_key(target, tar_gen);
      std::map<std::pair<Operation*,GenerationID>,unsigned>::const_iterator
        finder = op_map.find(target_key);
      // We only need to record it if it falls within our trace
      if (finder != op_map.end())
      {
        // Two cases here, 
        if (!source->is_internal_op())
        {
          // Normal case
          insert_dependence(DependenceRecord(finder->second, target_idx, 
                                source_idx, validates, dtype, dep_mask));
        }
        else
        {
          // Otherwise this is a internal op so record it special
          // Don't record dependences on our creator
          if (target_key != operations.back())
          { 
            std::pair<InternalOp*,GenerationID> 
              src_key(static_cast<InternalOp*>(source), src_gen);
#ifdef DEBUG_LEGION
            assert(internal_dependences.find(src_key) != 
                   internal_dependences.end());
#endif
            insert_dependence(src_key, 
                DependenceRecord(finder->second, target_idx, source_idx,
                                 validates, dtype, dep_mask));
          }
        }
      }
      else if (target->is_internal_op())
      {
        // First check to see if the internal op is one of ours
        std::pair<InternalOp*,GenerationID> 
          local_key(static_cast<InternalOp*>(target), tar_gen);
        std::map<std::pair<InternalOp*,GenerationID>,
                 LegionVector<DependenceRecord>>::const_iterator
          internal_finder = internal_dependences.find(local_key);
        if (internal_finder != internal_dependences.end())
        {
          // It is one of ours, so two cases
          if (!source->is_internal_op())
          {
            // Iterate over the internal operation dependences and 
            // translate them to our dependences
            for (LegionVector<DependenceRecord>::const_iterator
                  it = internal_finder->second.begin(); 
                  it != internal_finder->second.end(); it++)
            {
              FieldMask overlap = it->dependent_mask & dep_mask;
              if (!overlap)
                continue;
              insert_dependence(DependenceRecord(it->operation_idx, 
                  it->prev_idx, source_idx, it->validates, it->dtype, overlap));
            }
          }
          else
          {
            // Iterate over the internal operation dependences
            // and translate them to our dependences
            std::pair<InternalOp*,GenerationID> 
              src_key(static_cast<InternalOp*>(source), src_gen);
#ifdef DEBUG_LEGION
            assert(internal_dependences.find(src_key) != 
                   internal_dependences.end());
#endif
            for (LegionVector<DependenceRecord>::const_iterator
                  it = internal_finder->second.begin(); 
                  it != internal_finder->second.end(); it++)
            {
              FieldMask overlap = it->dependent_mask & dep_mask;
              if (!overlap)
                continue;
              insert_dependence(src_key, 
                  DependenceRecord(it->operation_idx, it->prev_idx,
                    source_idx, it->validates, it->dtype, overlap));
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::record_no_dependence(Operation *target, 
                                            GenerationID tar_gen,
                                            Operation *source, 
                                            GenerationID src_gen,
                                            unsigned target_idx, 
                                            unsigned source_idx,
                                            const FieldMask &dep_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tracing);
      assert(!target->is_internal_op());
      assert(!source->is_internal_op());
#endif
      std::pair<Operation*,GenerationID> target_key(target, tar_gen);
      std::map<std::pair<Operation*,GenerationID>,unsigned>::const_iterator
        finder = op_map.find(target_key);
      // We only need to record it if it falls within our trace
      if (finder != op_map.end())
      {
        insert_dependence(DependenceRecord(finder->second, target_idx, 
                  source_idx, false, LEGION_NO_DEPENDENCE, dep_mask));
      }
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::record_aliased_children(unsigned req_index,
                                          unsigned depth, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      unsigned index = operations.size() - 1;
      aliased_children[index].push_back(AliasChildren(req_index, depth, mask));
    } 

#ifdef LEGION_SPY
    //--------------------------------------------------------------------------
    void DynamicTrace::perform_logging(
                               UniqueID prev_fence_uid, UniqueID curr_fence_uid)
    //--------------------------------------------------------------------------
    {
      UniqueID context_uid = ctx->get_unique_id();
      for (unsigned idx = 0; idx < operations.size(); ++idx)
      {
        const UniqueID uid = get_current_uid_by_index(idx);
        const LegionVector<DependenceRecord> &deps = dependences[idx];
        for (LegionVector<DependenceRecord>::const_iterator it =
             deps.begin(); it != deps.end(); it++)
        {
          if ((it->prev_idx == -1) || (it->next_idx == -1))
          {
            LegionSpy::log_mapping_dependence(
               context_uid, get_current_uid_by_index(it->operation_idx),
               (it->prev_idx == -1) ? 0 : it->prev_idx,
               uid,
               (it->next_idx == -1) ? 0 : it->next_idx, LEGION_TRUE_DEPENDENCE);
          }
          else
          {
            LegionSpy::log_mapping_dependence(
                context_uid, get_current_uid_by_index(it->operation_idx),
                it->prev_idx, uid, it->next_idx, it->dtype);
          }
        }
        LegionSpy::log_mapping_dependence(
            context_uid, prev_fence_uid, 0, uid, 0, LEGION_TRUE_DEPENDENCE);
        LegionSpy::log_mapping_dependence(
            context_uid, uid, 0, curr_fence_uid, 0, LEGION_TRUE_DEPENDENCE);
      }
    }
#endif

    //--------------------------------------------------------------------------
    void DynamicTrace::insert_dependence(const DependenceRecord &record)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!dependences.empty());
#endif
      LegionVector<DependenceRecord> &deps = dependences.back();
      // Try to merge it with an existing dependence
      for (unsigned idx = 0; idx < deps.size(); idx++)
        if (deps[idx].merge(record))
          return;
      // If we make it here, we couldn't merge it so just add it
      deps.push_back(record);
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::insert_dependence(
                                 const std::pair<InternalOp*,GenerationID> &key,
                                 const DependenceRecord &record)
    //--------------------------------------------------------------------------
    {
      LegionVector<DependenceRecord> &deps = internal_dependences[key];
      // Try to merge it with an existing dependence
      for (unsigned idx = 0; idx < deps.size(); idx++)
        if (deps[idx].merge(record))
          return;
      // If we make it here, we couldn't merge it so just add it
      deps.push_back(record);
    }

    /////////////////////////////////////////////////////////////
    // TraceOp 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceOp::TraceOp(Runtime *rt)
      : FenceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceOp::TraceOp(const TraceOp &rhs)
      : FenceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TraceOp::~TraceOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceOp& TraceOp::operator=(const TraceOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TraceOp::execute_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapping_tracker == NULL);
#endif
      // Make a dependence tracker
      mapping_tracker = new MappingDependenceTracker();
      // See if we have any fence dependences
      execution_fence_event = parent_ctx->register_implicit_dependences(this);
      parent_ctx->invalidate_trace_cache(local_trace, this);

      trigger_dependence_analysis();
      end_dependence_analysis();
    }

    /////////////////////////////////////////////////////////////
    // TraceCaptureOp 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceCaptureOp::TraceCaptureOp(Runtime *rt)
      : TraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceCaptureOp::TraceCaptureOp(const TraceCaptureOp &rhs)
      : TraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TraceCaptureOp::~TraceCaptureOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceCaptureOp& TraceCaptureOp::operator=(const TraceCaptureOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::initialize_capture(InnerContext *ctx, bool has_block,
                                  bool remove_trace_ref, const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, EXECUTION_FENCE, false/*need future*/, provenance);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
      // Now mark our trace as NULL to avoid registering this operation
      trace = NULL;
      tracing = false;
      current_template = NULL;
      has_blocking_call = has_block;
      is_recording = false;
      remove_trace_reference = remove_trace_ref;
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_fence();
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_fence();
      runtime->free_capture_op(this);
    }

    //--------------------------------------------------------------------------
    const char* TraceCaptureOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_CAPTURE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TraceCaptureOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_CAPTURE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace == NULL);
      assert(local_trace != NULL);
#endif
      // Indicate that we are done capturing this trace
      local_trace->end_trace_capture();
      // Register this fence with all previous users in the parent's context
      FenceOp::trigger_dependence_analysis();
      parent_ctx->record_previous_trace(local_trace);
      if (local_trace->is_recording())
      {
        PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(physical_trace != NULL);
#endif
        physical_trace->record_previous_template_completion(
            get_completion_event());
        current_template = physical_trace->get_current_template();
        physical_trace->clear_cached_template();
        // Save this since we can't read it later in the mapping stage
        is_recording = true;
      }
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Now finish capturing the physical trace
      if (is_recording)
      {
        PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(current_template != NULL);
        assert(physical_trace != NULL);
#endif
        ApEvent pending_deletion = physical_trace->fix_trace(current_template, 
                              this, map_applied_conditions, has_blocking_call);
        if (pending_deletion.exists())
          execution_preconditions.insert(pending_deletion);
        local_trace->initialize_tracing_state();
      }
      if (remove_trace_reference && local_trace->remove_reference())
        delete local_trace;
      FenceOp::trigger_mapping();
    }

    /////////////////////////////////////////////////////////////
    // TraceCompleteOp 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceCompleteOp::TraceCompleteOp(Runtime *rt)
      : TraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceCompleteOp::TraceCompleteOp(const TraceCompleteOp &rhs)
      : TraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TraceCompleteOp::~TraceCompleteOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceCompleteOp& TraceCompleteOp::operator=(const TraceCompleteOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::initialize_complete(InnerContext *ctx, bool has_block,
                                              const char *provenance)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, EXECUTION_FENCE, false/*need future*/, provenance);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
      // Now mark our trace as NULL to avoid registering this operation
      trace = NULL;
      tracing = false;
      current_template = NULL;
      replayed = false;
      has_blocking_call = has_block;
      is_recording = false;
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_fence();
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_fence();
      runtime->free_trace_op(this);
    }

    //--------------------------------------------------------------------------
    const char* TraceCompleteOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_COMPLETE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TraceCompleteOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_COMPLETE_OP_KIND; 
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace == NULL);
      assert(local_trace != NULL);
#endif
#ifdef LEGION_SPY
      if (local_trace->is_replaying())
      {
        PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(physical_trace != NULL);
#endif
        local_trace->perform_logging(
         physical_trace->get_current_template()->get_fence_uid(), unique_op_id);
      }
#endif
      local_trace->end_trace_execution(this);
      parent_ctx->record_previous_trace(local_trace);

      if (local_trace->is_replaying())
      {
        if (has_blocking_call)
          REPORT_LEGION_ERROR(ERROR_INVALID_PHYSICAL_TRACING,
            "Physical tracing violation! Trace %d in task %s (UID %lld) "
            "encountered a blocking API call that was unseen when it was "
            "recorded. It is required that traces do not change their "
            "behavior.", local_trace->get_trace_id(),
            parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(physical_trace != NULL);
#endif 
        current_template = physical_trace->get_current_template();
#ifdef DEBUG_LEGION
        assert(current_template != NULL);
#endif
        parent_ctx->update_current_fence(this, true, true);
        // This is where we make sure that replays are done in order
        // We need to do this because we're not registering this as
        // a fence with the context
        physical_trace->chain_replays(this);
        physical_trace->record_previous_template_completion(completion_event);
        local_trace->initialize_tracing_state();
        replayed = true;
        return;
      }
      else if (local_trace->is_recording())
      {
        PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(physical_trace != NULL);
#endif
        physical_trace->record_previous_template_completion(completion_event);
        current_template = physical_trace->get_current_template();
        physical_trace->clear_cached_template();
        // Save this for later since we can't read it safely in mapping stage
        is_recording = true;
      }
      FenceOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (replayed)
      {
        // Having all our mapping dependences satisfied means that the previous 
        // replay of this template is done so we can start ours now
        std::set<RtEvent> replayed_events;
        current_template->perform_replay(runtime, replayed_events);
        if (!replayed_events.empty())
        {
          enqueue_ready_operation(Runtime::merge_events(replayed_events));
          return;
        }
      }
      enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Now finish capturing the physical trace
      if (is_recording)
      {
#ifdef DEBUG_LEGION
        assert(current_template != NULL);
        assert(local_trace->get_physical_trace() != NULL);
#endif
        const ApEvent pending_deletion =
          local_trace->get_physical_trace()->fix_trace(current_template, 
              this, map_applied_conditions, has_blocking_call);
        if (pending_deletion.exists())
          execution_preconditions.insert(pending_deletion);
        local_trace->initialize_tracing_state();
      }
      else if (replayed)
      {
#ifdef DEBUG_LEGION
        assert(current_template != NULL);
#endif
        std::set<ApEvent> template_postconditions;
        current_template->finish_replay(template_postconditions);
        complete_mapping();
        if (!template_postconditions.empty())
          Runtime::trigger_event(NULL, completion_event, 
              Runtime::merge_events(NULL, template_postconditions));
        else
          Runtime::trigger_event(NULL, completion_event);
        need_completion_trigger = false;
        complete_execution();
        return;
      }
      FenceOp::trigger_mapping();
    }

    /////////////////////////////////////////////////////////////
    // TraceReplayOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceReplayOp::TraceReplayOp(Runtime *rt)
      : TraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceReplayOp::TraceReplayOp(const TraceReplayOp &rhs)
      : TraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TraceReplayOp::~TraceReplayOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceReplayOp& TraceReplayOp::operator=(const TraceReplayOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::initialize_replay(InnerContext *ctx, LegionTrace *trace,
                                          Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, EXECUTION_FENCE, false/*need future*/, provenance);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_fence();
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_fence();
      runtime->free_replay_op(this);
    }

    //--------------------------------------------------------------------------
    const char* TraceReplayOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_REPLAY_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TraceReplayOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_REPLAY_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace == NULL);
      assert(local_trace != NULL);
#endif
      PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
      assert(physical_trace != NULL);
#endif
      bool recurrent = true;
      bool fence_registered = false;
      bool is_recording = local_trace->is_recording();
      if ((physical_trace->get_current_template() == NULL) || is_recording)
      {
        recurrent = false;
        {
          // Wait for the previous recordings to be done before checking
          // template preconditions, otherwise no template would exist.
          RtEvent mapped_event = parent_ctx->get_current_mapping_fence_event();
          if (mapped_event.exists())
            mapped_event.wait();
        }
#ifdef DEBUG_LEGION
        assert(!(local_trace->is_recording() || local_trace->is_replaying()));
#endif

        if (physical_trace->get_current_template() == NULL)
          physical_trace->check_template_preconditions(this, 
                                      map_applied_conditions);
#ifdef DEBUG_LEGION
        assert(physical_trace->get_current_template() == NULL ||
               !physical_trace->get_current_template()->is_recording());
#endif
        parent_ctx->perform_fence_analysis(this, execution_preconditions,
                                           true/*mapping*/, true/*execution*/);
        physical_trace->set_current_execution_fence_event(
            get_completion_event());
        fence_registered = true;
      }

      const bool replaying = (physical_trace->get_current_template() != NULL);
      // Tell the parent context about the physical trace replay result
      parent_ctx->record_physical_trace_replay(mapped_event, replaying);
      if (replaying)
      {
        // If we're recurrent, then check to see if we had any intermeidate
        // ops for which we still need to perform the fence analysis
        // If there were no intermediate dependences then we can just
        // record a dependence on the previous fence
        const ApEvent fence_completion = (recurrent &&
          !local_trace->has_intermediate_operations()) ?
            physical_trace->get_previous_template_completion()
                    : get_completion_event();
        if (recurrent && local_trace->has_intermediate_operations())
        {
          parent_ctx->perform_fence_analysis(this, execution_preconditions,
                                       true/*mapping*/, true/*execution*/);
          local_trace->reset_intermediate_operations();
        }
        if (!fence_registered)
          execution_preconditions.insert(
              parent_ctx->get_current_execution_fence_event());
        physical_trace->initialize_template(fence_completion, recurrent);
        local_trace->set_state_replay();
#ifdef LEGION_SPY
        physical_trace->get_current_template()->set_fence_uid(unique_op_id);
#endif
      }
      else if (!fence_registered)
      {
        parent_ctx->perform_fence_analysis(this, execution_preconditions,
                                           true/*mapping*/, true/*execution*/);
        physical_trace->set_current_execution_fence_event(
            get_completion_event());
      }

      // Now update the parent context with this fence before we can complete
      // the dependence analysis and possibly be deactivated
      parent_ctx->update_current_fence(this, true, true);
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::pack_remote_operation(Serializer &rez, 
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
    }

    /////////////////////////////////////////////////////////////
    // TraceBeginOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceBeginOp::TraceBeginOp(Runtime *rt)
      : TraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceBeginOp::TraceBeginOp(const TraceBeginOp &rhs)
      : TraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TraceBeginOp::~TraceBeginOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceBeginOp& TraceBeginOp::operator=(const TraceBeginOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::initialize_begin(InnerContext *ctx, LegionTrace *trace,
                                        Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, MAPPING_FENCE, false/*need future*/, provenance);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
      trace = NULL;
      tracing = false;
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_fence();
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_fence();
      runtime->free_begin_op(this);
    }

    //--------------------------------------------------------------------------
    const char* TraceBeginOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_BEGIN_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TraceBeginOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_BEGIN_OP_KIND;
    }

    /////////////////////////////////////////////////////////////
    // TraceSummaryOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceSummaryOp::TraceSummaryOp(Runtime *rt)
      : TraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceSummaryOp::TraceSummaryOp(const TraceSummaryOp &rhs)
      : TraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TraceSummaryOp::~TraceSummaryOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceSummaryOp& TraceSummaryOp::operator=(const TraceSummaryOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::initialize_summary(InnerContext *ctx,
                                            PhysicalTemplate *tpl,
                                            Operation *invalidator,
                                            Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, false/*track*/, 0/*regions*/, provenance);
      fence_kind = MAPPING_FENCE;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_fence_operation(parent_ctx->get_unique_id(),
                                       unique_op_id, context_index);
      context_index = invalidator->get_ctx_index();
      current_template = tpl;
      // The summary could have been marked as being traced,
      // so here we forcibly clear them out.
      trace = NULL;
      tracing = false;
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_fence();
      current_template = NULL;
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_fence();
      runtime->free_summary_op(this);
    }

    //--------------------------------------------------------------------------
    const char* TraceSummaryOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_SUMMARY_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TraceSummaryOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_SUMMARY_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_fence_analysis(true/*register fence also*/);
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_template->is_replayable());
#endif
      current_template->apply_postcondition(this, map_applied_conditions);
      FenceOp::trigger_mapping();
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTrace
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTrace::PhysicalTrace(Runtime *rt, LegionTrace *lt)
      : runtime(rt), logical_trace(lt), perform_fence_elision(
          !(runtime->no_trace_optimization || runtime->no_fence_elision)),
        previous_replay(NULL), current_template(NULL),
        nonreplayable_count(0), new_template_count(0), 
        previous_template_completion(ApEvent::NO_AP_EVENT),
        execution_fence_event(ApEvent::NO_AP_EVENT),
        intermediate_execution_fence(false)
    //--------------------------------------------------------------------------
    {
      if (runtime->replay_on_cpus)
      {
        Machine::ProcessorQuery local_procs(runtime->machine);
        local_procs.local_address_space();
        for (Machine::ProcessorQuery::iterator it =
             local_procs.begin(); it != local_procs.end(); it++)
          if (it->kind() == Processor::LOC_PROC)
            replay_targets.push_back(*it);
      }
      else
        replay_targets.push_back(runtime->utility_group);
    }

    //--------------------------------------------------------------------------
    PhysicalTrace::~PhysicalTrace()
    //--------------------------------------------------------------------------
    {
      for (std::vector<PhysicalTemplate*>::iterator it =
           templates.begin(); it != templates.end(); ++it)
        delete (*it);
      templates.clear();
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalTrace::fix_trace(PhysicalTemplate *tpl, Operation *op, 
                      std::set<RtEvent> &applied_events, bool has_blocking_call)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tpl->is_recording());
#endif
      tpl->finalize(op, has_blocking_call);
      ApEvent pending_deletion = ApEvent::NO_AP_EVENT;
      if (!tpl->is_replayable())
      {
        if (!tpl->defer_template_deletion(pending_deletion, applied_events))
          delete tpl;
        if (++nonreplayable_count > LEGION_NON_REPLAYABLE_WARNING)
        {
          const std::string &message = tpl->get_replayable_message();
          const char *message_buffer = message.c_str();
          REPORT_LEGION_WARNING(LEGION_WARNING_NON_REPLAYABLE_COUNT_EXCEEDED,
              "WARNING: The runtime has failed to memoize the trace more than "
              "%u times, due to the absence of a replayable template. It is "
              "highly likely that trace %u will not be memoized for the rest "
              "of execution. The most recent template was not replayable "
              "for the following reason: %s. Please change the mapper to stop "
              "making memoization requests.", LEGION_NON_REPLAYABLE_WARNING,
              logical_trace->get_trace_id(), message_buffer)
          nonreplayable_count = 0;
        }
      }
      else
      {
        // Reset the nonreplayable count when we find a replayable template
        nonreplayable_count = 0;
        // See if we're going to exceed the maximum number of templates
        if (templates.size() == logical_trace->ctx->get_max_trace_templates())
        {
#ifdef DEBUG_LEGION
          assert(!templates.empty());
#endif
          PhysicalTemplate *to_delete = templates.front();
          if (!to_delete->defer_template_deletion(pending_deletion, 
                                                  applied_events))
            delete to_delete;
          // Remove the least recently used (first) one from the vector
          // shift it to the back first though, should be fast
          if (templates.size() > 1)
            std::rotate(templates.begin(),templates.begin()+1,templates.end());
          templates.pop_back();
        }
        templates.push_back(tpl);
        if (++new_template_count > LEGION_NEW_TEMPLATE_WARNING_COUNT)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_NEW_TEMPLATE_COUNT_EXCEEDED,
              "WARNING: The runtime has created %d new replayable templates "
              "for trace %u without replaying any existing templates. This "
              "may mean that your mapper is not making mapper decisions "
              "conducive to replaying templates. Please check that your "
              "mapper is making decisions that align with prior templates. "
              "If you believe that this number of templates is reasonable "
              "please adjust the settings for LEGION_NEW_TEMPLATE_WARNING_COUNT"
              " in legion_config.h.", LEGION_NEW_TEMPLATE_WARNING_COUNT, 
              logical_trace->get_trace_id())
          new_template_count = 0;
        }
      }
      return pending_deletion;
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::check_template_preconditions(TraceReplayOp *op,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      current_template = NULL;
      // Scan backwards since more recently used templates are likely
      // to be the ones that best match what we are executing
      for (int idx = templates.size() - 1; idx >= 0; idx--)
      {
        PhysicalTemplate *tpl = templates[idx];
        if (tpl->check_preconditions(op, applied_events))
        {
#ifdef DEBUG_LEGION
          assert(tpl->is_replayable());
#endif
          // Reset the nonreplayable count when a replayable template satisfies
          // the precondition
          nonreplayable_count = 0;
          // Also reset the new template count as we found a replay
          new_template_count = 0;
          current_template = tpl;
          // Move the template to the end of the vector as most-recently used
          if (idx < int(templates.size() - 1))
            std::rotate(templates.begin()+idx, 
                        templates.begin()+idx+1, templates.end());
          return;
        }
      }
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate* PhysicalTrace::start_new_template(void)
    //--------------------------------------------------------------------------
    {
      current_template = new PhysicalTemplate(this, execution_fence_event);
      return current_template;
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::record_intermediate_execution_fence(FenceOp *fence)
    //--------------------------------------------------------------------------
    {
      if (!intermediate_execution_fence)
        fence->record_execution_precondition(previous_template_completion);
      previous_template_completion = fence->get_completion_event();
      intermediate_execution_fence = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::chain_replays(FenceOp *replay_op)
    //--------------------------------------------------------------------------
    {
      if (previous_replay != NULL)
      {
#ifdef LEGION_SPY
        // Can't prune when doing legion spy
        replay_op->register_dependence(previous_replay, previous_replay_gen);
#else
        if (replay_op->register_dependence(previous_replay,previous_replay_gen))
          previous_replay = NULL;
#endif
      }
      previous_replay = replay_op;
      previous_replay_gen = replay_op->get_generation();
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::initialize_template(
                                       ApEvent fence_completion, bool recurrent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_template != NULL);
#endif
      // If we had an intermeidate execution fence between replays then
      // we should no longer be considered recurrent when we replay the trace
      // We're also not going to be considered recurrent here if we didn't
      // do fence elision since since we'll still need to track the fence
      current_template->initialize_replay(fence_completion, 
          recurrent && perform_fence_elision && !intermediate_execution_fence);
      // Reset this for the next replay
      intermediate_execution_fence = false;
    }

    /////////////////////////////////////////////////////////////
    // TraceViewSet
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    std::string TraceViewSet::FailedPrecondition::to_string(
                                                         TaskContext *ctx) const
    //--------------------------------------------------------------------------
    {
      const char *mem_names[] = {
#define MEM_NAMES(name, desc) #name,
          REALM_MEMORY_KINDS(MEM_NAMES) 
#undef MEM_NAMES
        };
      IndividualManager *manager = view->get_manager()->as_individual_manager();
      FieldSpaceNode *field_space = manager->field_space_node;
      Memory memory = manager->memory_manager->memory;
      char *m = mask.to_string();
      std::vector<FieldID> fields;
      field_space->get_field_set(mask, ctx, fields);

      std::stringstream ss;
      ss << "view: " << view << " in " << mem_names[memory.kind()]
         << " memory " << std::hex << memory.id << std::dec
         << ", Index expr: " << eq->set_expr->expr_id
         << ", Field Mask: " << m << ", Fields: ";
      for (std::vector<FieldID>::const_iterator it =
            fields.begin(); it != fields.end(); it++)
      {
        if (it != fields.begin())
          ss << ", ";
        const void *name = NULL;
        size_t name_size = 0;
        if (field_space->retrieve_semantic_information(LEGION_NAME_SEMANTIC_TAG,
              name, name_size, true/*can fail*/, false/*wait until*/))
        {
          ss << ((const char*)name) << " (" << *it << ")";
        }
        else
          ss << *it;
      }
      return ss.str();
    }

    //--------------------------------------------------------------------------
    TraceViewSet::TraceViewSet(RegionTreeForest *f)
      : forest(f), view_references(f->runtime->dump_physical_traces)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceViewSet::~TraceViewSet(void)
    //--------------------------------------------------------------------------
    {
      if (view_references)
      {
        for (ViewSet::const_iterator it = conditions.begin();
              it != conditions.end(); it++)
          if (it->first->remove_base_resource_ref(TRACE_REF))
            delete it->first;
      }
      conditions.clear();
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::insert(
                  InstanceView *view, EquivalenceSet *eq, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      if (view_references && (conditions.find(view) == conditions.end()))
        view->add_base_resource_ref(TRACE_REF);
      conditions[view].insert(eq, mask);
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::invalidate(
                  InstanceView *view, EquivalenceSet *eq, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      ViewSet::iterator finder = conditions.find(view);
      if (finder == conditions.end())
        return;

      FieldMaskSet<EquivalenceSet> to_delete;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            finder->second.begin(); it != finder->second.end(); ++it)
      {
        FieldMask overlap = mask & it->second;
        if (!overlap)
          continue;

        IndexSpaceExpression *expr1 = eq->set_expr;
        IndexSpaceExpression *expr2 = it->first->set_expr;
        if (expr1 == expr2)
        {
          to_delete.insert(it->first, overlap);
        }
        else if (expr1->get_volume() >= expr2->get_volume())
        {
          IndexSpaceExpression *diff =
            forest->subtract_index_spaces(expr2, expr1);
          if (diff->is_empty())
            to_delete.insert(it->first, overlap);
        }
      }
      for (FieldMaskSet<EquivalenceSet>::iterator it = to_delete.begin();
           it != to_delete.end(); ++it)
      {
        FieldMaskSet<EquivalenceSet>::iterator eit =
          finder->second.find(it->first);
#ifdef DEBUG_LEGION
        assert(eit != finder->second.end());
#endif
        eit.filter(it->second);
        if (!eit->second)
          finder->second.erase(eit);
      }
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::dominates(
         InstanceView *view, EquivalenceSet *eq, FieldMask &non_dominated) const
    //--------------------------------------------------------------------------
    {
      // If this is for an empty equivalence set then it doesn't matter
      if (eq->set_expr->is_empty())
        return true;
      ViewSet::const_iterator finder = conditions.find(view);
      if (finder == conditions.end())
        return false;

      LegionList<FieldSet<EquivalenceSet*> > field_sets;
      finder->second.compute_field_sets(non_dominated, field_sets);
      for (LegionList<FieldSet<EquivalenceSet*> >::const_iterator it =
            field_sets.begin(); it != field_sets.end(); it++)
      {
        if (it->elements.empty())
          continue;
        std::set<IndexSpaceExpression*> exprs;
        for (std::set<EquivalenceSet*>::const_iterator eit = 
              it->elements.begin(); eit != it->elements.end(); eit++)
          exprs.insert((*eit)->set_expr);
        IndexSpaceExpression *union_expr = forest->union_index_spaces(exprs);
        IndexSpaceExpression *expr = eq->set_expr;
        if (expr == union_expr)
          non_dominated -= it->set_mask;
        // Can only dominate if we have enough points
        else if (expr->get_volume() <= union_expr->get_volume())
        {
          IndexSpaceExpression *diff_expr =
            forest->subtract_index_spaces(expr, union_expr);
          if (diff_expr->is_empty())
            non_dominated -= it->set_mask;
        }
      }
      // If there are no fields left then we dominated
      return !non_dominated;
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::subsumed_by(const TraceViewSet &set, 
                                   FailedPrecondition *condition) const
    //--------------------------------------------------------------------------
    {
      for (ViewSet::const_iterator it = conditions.begin();
           it != conditions.end(); ++it)
        for (FieldMaskSet<EquivalenceSet>::const_iterator eit =
             it->second.begin(); eit != it->second.end(); ++eit)
        {
          FieldMask mask = eit->second;
          if (!set.dominates(it->first, eit->first, mask))
          {
            if (condition != NULL)
            {
              condition->view = it->first;
              condition->eq = eit->first;
              condition->mask = mask;
            }
            return false;
          }
        }

      return true;
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::has_refinements(void) const
    //--------------------------------------------------------------------------
    {
      for (ViewSet::const_iterator it = conditions.begin();
           it != conditions.end(); ++it)
        for (FieldMaskSet<EquivalenceSet>::const_iterator eit =
             it->second.begin(); eit != it->second.end(); ++eit)
          if (eit->first->has_refinements(eit->second))
            return true;

      return false;
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::empty(void) const
    //--------------------------------------------------------------------------
    {
      return conditions.empty();
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::dump(void) const
    //--------------------------------------------------------------------------
    {
      for (ViewSet::const_iterator it = conditions.begin();
           it != conditions.end(); ++it)
      {
        InstanceView *view = it->first;
        for (FieldMaskSet<EquivalenceSet>::const_iterator eit =
             it->second.begin(); eit != it->second.end(); ++eit)
        {
          char *mask = eit->second.to_string();
          LogicalRegion lr =
            forest->get_tree(view->get_manager()->tree_id)->handle;
          const void *name = NULL; size_t name_size = 0;
          forest->runtime->retrieve_semantic_information(lr, 
              LEGION_NAME_SEMANTIC_TAG, name, name_size, true, true);
          log_tracing.info() << "  "
                    <<(view->is_reduction_view() ? "Reduction" : "Materialized")
                    << " view: " << view << ", Inst: " << std::hex
                    << view->get_manager()->get_instance(DomainPoint()).id 
                    << std::dec
                    << ", Index expr: " << eit->first->set_expr->expr_id
                    << ", Name: " << (name_size > 0 ? (const char*)name : "")
                    << ", Field Mask: " << mask;
          free(mask);
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // TraceConditionSet
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceConditionSet::TraceConditionSet(RegionTreeForest *f)
      : TraceViewSet(f), cached(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceConditionSet::~TraceConditionSet(void)
    //--------------------------------------------------------------------------
    {
      views.clear();
      version_infos.clear();
      if (!valid_instances.empty())
      {
        for (std::vector<PhysicalManager*>::const_iterator it =
              valid_instances.begin(); it != valid_instances.end(); it++)
          if ((*it)->remove_base_valid_ref(TRACE_REF))
            delete (*it);
      }
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::make_ready(bool postcondition)
    //--------------------------------------------------------------------------
    {
      if (cached)
        return;
      cached = true;

      typedef std::pair<RegionTreeID,EquivalenceSet*> Key;
      LegionMap<Key,FieldMaskSet<InstanceView> > views_by_regions;

      for (ViewSet::iterator it = conditions.begin(); it != conditions.end();
           ++it)
      {
        RegionTreeID tid = it->first->get_manager()->tree_id;
        for (FieldMaskSet<EquivalenceSet>::iterator eit = it->second.begin();
             eit != it->second.end(); ++eit)
        {
          EquivalenceSet *eq = eit->first;
          Key key(tid, eq);
          FieldMaskSet <InstanceView> &vset = views_by_regions[key];
          vset.insert(it->first, eit->second);
        }
        // If we're capturing a postcondition, then we need a valie
        // reference on each of the physical managers
        if (postcondition)
        {
          PhysicalManager *manager = it->first->manager;
          manager->add_base_valid_ref(TRACE_REF);
          valid_instances.push_back(manager);
        }
      }

      // Filter out views that overlap with some restricted views
      if (postcondition)
      {
        for (LegionMap<Key,FieldMaskSet<InstanceView> >::iterator it =
             views_by_regions.begin(); it != views_by_regions.end(); ++it)
        {
          EquivalenceSet *eq = it->first.second;
          FieldMaskSet<InstanceView> &all_views = it->second;
          if (!eq->has_restrictions(all_views.get_valid_mask())) continue;

          FieldMaskSet<InstanceView> restricted_views;
          FieldMask restricted_mask;
          for (FieldMaskSet<InstanceView>::iterator vit = all_views.begin();
               vit != all_views.end(); ++vit)
          {
            FieldMask restricted = eq->is_restricted(vit->first);
            FieldMask overlap = restricted & vit->second;
            if (!!overlap)
            {
              restricted_views.insert(vit->first, overlap);
              restricted_mask |= overlap;
            }
          }

          std::vector<InstanceView*> to_delete;
          for (FieldMaskSet<InstanceView>::iterator vit = all_views.begin();
               vit != all_views.end(); ++vit)
          {
            vit.filter(restricted_mask);
            if (!vit->second)
              to_delete.push_back(vit->first);
          }

          for (std::vector<InstanceView*>::iterator vit = to_delete.begin();
               vit != to_delete.end(); ++vit)
            all_views.erase(*vit);

          for (FieldMaskSet<InstanceView>::iterator vit =
               restricted_views.begin(); vit != restricted_views.end(); ++vit)
            all_views.insert(vit->first, vit->second);
        }
      }

      unsigned idx = 0;
      version_infos.resize(views_by_regions.size());
      for (LegionMap<Key,FieldMaskSet<InstanceView> >::iterator it =
           views_by_regions.begin(); it != views_by_regions.end(); ++it)
      {
        views.push_back(it->second);
        version_infos[idx++].record_equivalence_set(NULL,
            it->first.second, it->second.get_valid_mask());
      }
    }

    //--------------------------------------------------------------------------
    bool TraceConditionSet::require(Operation *op, 
                                    std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cached);
#endif
      for (unsigned idx = 0; idx < views.size(); ++idx)
      {
        FieldMaskSet<InstanceView> invalid_views;
        forest->find_invalid_instances(op, idx, version_infos[idx], views[idx],
                                       invalid_views, applied_events);
        if (!invalid_views.empty())
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::ensure(Operation *op, 
                                   std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cached);
#endif
      const TraceInfo trace_info(op, false/*init*/);
      for (unsigned idx = 0; idx < views.size(); ++idx)
        forest->update_valid_instances(op, idx, version_infos[idx], views[idx],
            PhysicalTraceInfo(trace_info, idx), applied_events);
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTemplate
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTemplate::PhysicalTemplate(PhysicalTrace *t, ApEvent fence_event)
      : trace(t), recording(true), replayable(false, "uninitialized"),
        fence_completion_id(0),
        replay_parallelism(t->runtime->max_replay_parallelism),
        has_virtual_mapping(false), last_fence(NULL),
        recording_done(Runtime::create_rt_user_event()),
        pre(t->runtime->forest), post(t->runtime->forest),
        pre_reductions(t->runtime->forest), post_reductions(t->runtime->forest),
        consumed_reductions(t->runtime->forest)
    //--------------------------------------------------------------------------
    {
      events.push_back(fence_event);
      event_map[fence_event] = fence_completion_id;
      instructions.push_back(
         new AssignFenceCompletion(*this, fence_completion_id, TraceLocalID()));
      // always want at least one set of operations ready for recording
      operations.emplace_back(std::map<TraceLocalID,Memoizable*>());
      pending_inv_topo_order.store(NULL);
      pending_transitive_reduction.store(NULL);
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate::PhysicalTemplate(const PhysicalTemplate &rhs)
      : trace(NULL), recording(true), replayable(false, "uninitialized"),
        fence_completion_id(0),
        replay_parallelism(1), recording_done(RtUserEvent::NO_RT_USER_EVENT),
        pre(NULL), post(NULL), pre_reductions(NULL), post_reductions(NULL),
        consumed_reductions(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate::~PhysicalTemplate(void)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock tpl_lock(template_lock);
        for (std::set<ViewUser*>::iterator it = all_users.begin();
             it != all_users.end(); ++it)
          delete (*it);
        for (std::vector<Instruction*>::iterator it = instructions.begin();
             it != instructions.end(); ++it)
          delete *it;
        // Relesae references to instances
        for (CachedMappings::iterator it = cached_mappings.begin();
            it != cached_mappings.end(); ++it)
        {
          for (std::deque<InstanceSet>::iterator pit =
              it->second.physical_instances.begin(); pit !=
              it->second.physical_instances.end(); pit++)
          {
            for (unsigned idx = 0; idx < pit->size(); idx++)
            {
              const InstanceRef &ref = (*pit)[idx];
              if (!ref.is_virtual_ref())
                ref.remove_valid_reference(MAPPING_ACQUIRE_REF,NULL/*mutator*/);
            }
            pit->clear();
          }
        }
        cached_mappings.clear();
        if (!remote_memos.empty())
          release_remote_memos();
      }
      std::vector<unsigned> *inv_topo_order = pending_inv_topo_order.load();
      if (inv_topo_order != NULL)
        delete inv_topo_order;
      std::vector<std::vector<unsigned> > *transitive_reduction =
        pending_transitive_reduction.load();
      if (transitive_reduction != NULL)
        delete transitive_reduction;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::initialize_replay(ApEvent completion, bool recurrent)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(template_lock);
      operations.emplace_back(std::map<TraceLocalID,Memoizable*>());
      pending_replays.emplace_back(std::make_pair(completion, recurrent));
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalTemplate::get_completion_for_deletion(void) const
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> all_events;
      for (std::map<ApEvent, unsigned>::const_iterator it = event_map.begin();
           it != event_map.end(); ++it)
        all_events.insert(it->first);
      return Runtime::merge_events(NULL, all_events);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::find_execution_fence_preconditions(
                                               std::set<ApEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      // No need for a lock here, the mapping fence protects us
#ifdef DEBUG_LEGION
      assert(!events.empty());
      assert(events.size() == instructions.size());
#endif
      // Scan backwards until we find the previous execution fence (if any)
      for (unsigned idx = events.size() - 1; idx > 0; idx--)
      {
        preconditions.insert(events[idx]);
        if (instructions[idx] == last_fence)
          return;
      }
      preconditions.insert(events.front());
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::check_preconditions(TraceReplayOp *op,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      return pre.require(op, applied_events);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::apply_postcondition(TraceSummaryOp *op,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      post.ensure(op, applied_events);
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate::Replayable PhysicalTemplate::check_replayable(
                                                   bool has_blocking_call) const
    //--------------------------------------------------------------------------
    {
      if (has_blocking_call)
        return Replayable(false, "blocking call");

      if (has_virtual_mapping)
        return Replayable(false, "virtual mapping");

      if (!pre_fill_views.empty())
        return Replayable(false, "external fill views");

      if (!pre_reductions.empty())
        return Replayable(false, "external reduction views");

      if (pre.has_refinements() || post.has_refinements())
        return Replayable(false, "found refined equivalence sets");

      TraceViewSet::FailedPrecondition condition;
      if (!post_reductions.subsumed_by(consumed_reductions, &condition))
      {
        if (trace->runtime->dump_physical_traces)
        {
          return Replayable(
              false, "escaping reduction view: " +
                condition.to_string(trace->logical_trace->ctx));
        }
        else
          return Replayable(false, "escaping reduction views");
      }

      if (!pre.subsumed_by(post, &condition))
      {
        if (trace->runtime->dump_physical_traces)
        {
          return Replayable(
              false, "precondition not subsumed: " +
                condition.to_string(trace->logical_trace->ctx));
        }
        else
          return Replayable(
              false, "precondition not subsumed by postcondition");
      }

      return Replayable(true);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::register_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      Memoizable *memoizable = op->get_memoizable();
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      const TraceLocalID tid = memoizable->get_trace_local_id();
      // Should be able to call back() without the lock even when
      // operations are being removed from the front
      std::map<TraceLocalID,Memoizable*> &ops = operations.back();
#ifdef DEBUG_LEGION
      assert(ops.find(tid) == ops.end());
      assert(memo_entries.find(tid) != memo_entries.end());
#endif
      ops[tid] = memoizable;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::execute_slice(unsigned slice_idx,
                                         bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(slice_idx < slices.size());
#endif
      std::map<TraceLocalID,Memoizable*> &ops = operations.front();
      std::vector<Instruction*> &instructions = slices[slice_idx];
      for (std::vector<Instruction*>::const_iterator it = instructions.begin();
           it != instructions.end(); ++it)
        (*it)->execute(events, user_events, ops, recurrent_replay);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::issue_summary_operations(
          InnerContext* context, Operation *invalidator, Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      TraceSummaryOp *op = trace->runtime->get_available_summary_op();
      op->initialize_summary(context, this, invalidator, provenance);
#ifdef LEGION_SPY
      LegionSpy::log_summary_op_creator(op->get_unique_op_id(),
                                        invalidator->get_unique_op_id());
#endif
      op->execute_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::finalize(Operation *op, bool has_blocking_call)
    //--------------------------------------------------------------------------
    {
      if (!recording_done.has_triggered())
        Runtime::trigger_event(recording_done);
      recording = false;
      replayable = check_replayable(has_blocking_call);

      if (!replayable)
      {
        if (trace->runtime->dump_physical_traces)
        {
          optimize(true/*do transitive reduction inline*/);
          dump_template();
        }
        if (!remote_memos.empty())
          release_remote_memos();
        return;
      }
      generate_conditions();
      // Most of the optimizations are O(N) in the number of instructions
      // so we can do them here without much overhead, but transitive
      // reduction can O(N^3) in the worst case, so we do that with a
      // meta-task in the background in case it is expensive
      optimize(false/*do transitive reduction inline*/);
      size_t num_events = events.size();
      events.clear();
      events.resize(num_events);
      event_map.clear();
      if (!remote_memos.empty())
        release_remote_memos();
      // Defer performing the transitive reduction because it might
      // be expensive (see comment above)
      if (!trace->runtime->no_trace_optimization)
      {
        TransitiveReductionArgs args(this);
        transitive_reduction_done = trace->runtime->issue_runtime_meta_task(
                                          args, LG_THROUGHPUT_WORK_PRIORITY);
      }
      // Can dump now if we're not deferring the transitive reduction
      else if (trace->runtime->dump_physical_traces)
        dump_template();
      operations.pop_front();
      op_views.clear();
      copy_views.clear();
      src_indirect_views.clear();
      dst_indirect_views.clear();
      across_copies.clear();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::generate_conditions(void)
    //--------------------------------------------------------------------------
    {
      pre.make_ready(false /*postcondition*/);
      post.make_ready(true /*postcondition*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::optimize(bool do_transitive_reduction)
    //--------------------------------------------------------------------------
    {
      std::vector<unsigned> gen;
      if (trace->perform_fence_elision)
        elide_fences(gen);
      else
      {
#ifdef DEBUG_LEGION
        assert(instructions.size() == events.size());
#endif
        gen.resize(events.size());
        for (unsigned idx = 0; idx < events.size(); ++idx)
          gen[idx] = idx;
      }
      if (!trace->runtime->no_trace_optimization)
      {
        propagate_merges(gen);
        if (do_transitive_reduction)
          transitive_reduction(false/*deferred*/);
        propagate_copies(&gen);
        eliminate_dead_code(gen);
      }
      prepare_parallel_replay(gen);
      push_complete_replays();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::elide_fences(std::vector<unsigned> &gen)
    //--------------------------------------------------------------------------
    {
      // Reserve some events for merges to be added during fence elision
      unsigned num_merges = 0;
      for (std::vector<Instruction*>::iterator it = instructions.begin();
           it != instructions.end(); ++it)
        switch ((*it)->get_kind())
        {
          case ISSUE_COPY:
            {
              unsigned precondition_idx =
                (*it)->as_issue_copy()->precondition_idx;
              InstructionKind generator_kind =
                instructions[precondition_idx]->get_kind();
              num_merges += (generator_kind != MERGE_EVENT) ? 1 : 0;
              break;
            }
          case ISSUE_FILL:
            {
              unsigned precondition_idx =
                (*it)->as_issue_fill()->precondition_idx;
              InstructionKind generator_kind =
                instructions[precondition_idx]->get_kind();
              num_merges += (generator_kind != MERGE_EVENT) ? 1 : 0;
              break;
            }
          case ISSUE_ACROSS:
            {
              IssueAcross *across = (*it)->as_issue_across();
              if (across->collective_precondition == 0)
              {
                InstructionKind generator_kind = 
                  instructions[across->copy_precondition]->get_kind();
                num_merges += (generator_kind != MERGE_EVENT) ? 1 : 0;
              }
              else
              {
                InstructionKind generator_kind = 
                  instructions[across->collective_precondition]->get_kind();
                num_merges += (generator_kind != MERGE_EVENT) ? 1 : 0;
              }
              if (across->src_indirect_precondition != 0)
              {
                InstructionKind generator_kind = 
                  instructions[across->src_indirect_precondition]->get_kind();
                num_merges += (generator_kind != MERGE_EVENT) ? 1 : 0;
              }
              if (across->dst_indirect_precondition != 0)
              {
                InstructionKind generator_kind = 
                  instructions[across->dst_indirect_precondition]->get_kind();
                num_merges += (generator_kind != MERGE_EVENT) ? 1 : 0;
              }
              break;
            }
          case COMPLETE_REPLAY:
            {
              unsigned completion_event_idx =
                (*it)->as_complete_replay()->rhs;
              InstructionKind generator_kind =
                instructions[completion_event_idx]->get_kind();
              num_merges += generator_kind != MERGE_EVENT ? 1 : 0;
              break;
            }
          default:
            {
              break;
            }
        }

      unsigned merge_starts = events.size();
      events.resize(events.size() + num_merges);

      // We are now going to break the invariant that
      // the generator of events[idx] is instructions[idx].
      // After fence elision, the generator of events[idx] is
      // instructions[gen[idx]].
      gen.resize(events.size());
      std::vector<Instruction*> new_instructions;

      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        InstructionKind kind = inst->get_kind();
        switch (kind)
        {
          case COMPLETE_REPLAY:
            {
              CompleteReplay *replay = inst->as_complete_replay();
              std::map<TraceLocalID, ViewExprs>::iterator finder =
                op_views.find(replay->owner);
              if (finder == op_views.end()) break;
              std::set<unsigned> users;
              find_all_last_users(finder->second, users);
              rewrite_preconditions(replay->rhs, users,
                  instructions, new_instructions, gen, merge_starts);
              break;
            }
          case ISSUE_COPY:
            {
              IssueCopy *copy = inst->as_issue_copy();
              std::map<unsigned, ViewExprs>::iterator finder =
                copy_views.find(copy->lhs);
#ifdef DEBUG_LEGION
              assert(finder != copy_views.end());
#endif
              std::set<unsigned> users;
              find_all_last_users(finder->second, users);
              rewrite_preconditions(copy->precondition_idx, users,
                  instructions, new_instructions, gen, merge_starts);
              break;
            }
          case ISSUE_FILL:
            {
              IssueFill *fill = inst->as_issue_fill();
              std::map<unsigned, ViewExprs>::iterator finder =
                copy_views.find(fill->lhs);
#ifdef DEBUG_LEGION
              assert(finder != copy_views.end());
#endif
              std::set<unsigned> users;
              find_all_last_users(finder->second, users);
              rewrite_preconditions(fill->precondition_idx, users,
                  instructions, new_instructions, gen, merge_starts);
              break;
            }
          case ISSUE_ACROSS:
            {
              IssueAcross *across = inst->as_issue_across();
              std::map<unsigned, ViewExprs>::iterator finder =
                copy_views.find(across->lhs);
#ifdef DEBUG_LEGION
              assert(finder != copy_views.end());
#endif
              std::set<unsigned> users;
              find_all_last_users(finder->second, users);
              // This is super subtle: for indirections that are
              // working collectively together on a set of indirect
              // source or destination instances, we actually have
              // a fan-in event construction. The indirect->copy_precondition
              // contains the result of that fan-in tree which is not
              // what we want to update here. We instead want to update
              // the set of preconditions to that collective fan-in for this
              // part of the indirect which feed into the collective event
              // tree construction. The local fan-in event is stored at
              // indirect->collective_precondition so use that instead for this
              if (across->collective_precondition == 0)
                rewrite_preconditions(across->copy_precondition, users,
                    instructions, new_instructions, gen, merge_starts);
              else
                rewrite_preconditions(across->collective_precondition, users,
                    instructions, new_instructions, gen, merge_starts);
              // Also do the rewrites for any indirection preconditions
              if (across->src_indirect_precondition != 0)
              {
                users.clear();
                finder = src_indirect_views.find(across->lhs);
#ifdef DEBUG_LEGION
                assert(finder != src_indirect_views.end());
#endif
                find_all_last_users(finder->second, users);
                rewrite_preconditions(across->src_indirect_precondition, users,
                    instructions, new_instructions, gen, merge_starts);
              }
              if (across->dst_indirect_precondition != 0)
              {
                users.clear();
                finder = dst_indirect_views.find(across->lhs);
#ifdef DEBUG_LEGION
                assert(finder != dst_indirect_views.end());
#endif
                find_all_last_users(finder->second, users);
                rewrite_preconditions(across->dst_indirect_precondition, users,
                    instructions, new_instructions, gen, merge_starts);
              }
              break;
            }
          default:
            {
              break;
            }
        }
        gen[idx] = new_instructions.size();
        new_instructions.push_back(inst);
      }
      instructions.swap(new_instructions);
      new_instructions.clear();
      // If we added events for fence elision then resize events so that
      // all the new events from a previous trace are generated by the 
      // fence instruction at the beginning of the template
      if (events.size() > gen.size())
        gen.resize(events.size(), 0/*fence instruction*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::rewrite_preconditions(
                            unsigned &precondition, std::set<unsigned> &users,
                            const std::vector<Instruction*> &instructions,
                            std::vector<Instruction*> &new_instructions,
                            std::vector<unsigned> &gen, unsigned &merge_starts)
    //--------------------------------------------------------------------------
    {
      if (users.empty())
        return;
      Instruction *generator_inst = instructions[precondition];
      if (generator_inst->get_kind() == MERGE_EVENT)
      {
        MergeEvent *merge = generator_inst->as_merge_event();
        merge->rhs.insert(users.begin(), users.end());
      }
      else
      {
        unsigned merging_event_idx = merge_starts++;
        gen[merging_event_idx] = new_instructions.size();
        if (precondition != fence_completion_id)
          users.insert(precondition);
        new_instructions.push_back(
            new MergeEvent(*this, merging_event_idx, users,
                           generator_inst->owner));
        precondition = merging_event_idx;
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::propagate_merges(std::vector<unsigned> &gen)
    //--------------------------------------------------------------------------
    {
      std::vector<Instruction*> new_instructions;
      std::vector<bool> used(instructions.size(), false);

      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        InstructionKind kind = inst->get_kind();
        used[idx] = kind != MERGE_EVENT;
        switch (kind)
        {
          case MERGE_EVENT:
            {
              MergeEvent *merge = inst->as_merge_event();
              std::set<unsigned> new_rhs;
              bool changed = false;
              for (std::set<unsigned>::iterator it = merge->rhs.begin();
                   it != merge->rhs.end(); ++it)
              {
                Instruction *generator = instructions[gen[*it]];
                if (generator ->get_kind() == MERGE_EVENT)
                {
                  MergeEvent *to_splice = generator->as_merge_event();
                  new_rhs.insert(to_splice->rhs.begin(), to_splice->rhs.end());
                  changed = true;
                }
                else
                  new_rhs.insert(*it);
              }
              if (changed)
                merge->rhs.swap(new_rhs);
              break;
            }
          case TRIGGER_EVENT:
            {
              TriggerEvent *trigger = inst->as_trigger_event();
              used[gen[trigger->rhs]] = true;
              break;
            }
          case ISSUE_COPY:
            {
              IssueCopy *copy = inst->as_issue_copy();
              used[gen[copy->precondition_idx]] = true;
              break;
            }
          case ISSUE_FILL:
            {
              IssueFill *fill = inst->as_issue_fill();
              used[gen[fill->precondition_idx]] = true;
              break;
            }
          case ISSUE_ACROSS:
            {
              IssueAcross *across = inst->as_issue_across();
              used[gen[across->copy_precondition]] = true;
              if (across->collective_precondition != 0)
                used[gen[across->collective_precondition]] = true;
              if (across->src_indirect_precondition != 0)
                used[gen[across->src_indirect_precondition]] = true;
              if (across->dst_indirect_precondition != 0)
                used[gen[across->dst_indirect_precondition]] = true;
              break;
            }
          case SET_EFFECTS:
            {
              SetEffects *effects = inst->as_set_effects();
              used[gen[effects->rhs]] = true;
              break;
            }
          case COMPLETE_REPLAY:
            {
              CompleteReplay *complete = inst->as_complete_replay();
              used[gen[complete->rhs]] = true;
              break;
            }
          case GET_TERM_EVENT:
          case CREATE_AP_USER_EVENT:
          case SET_OP_SYNC_EVENT:
          case ASSIGN_FENCE_COMPLETION:
            {
              break;
            }
          default:
            {
              // unreachable
              assert(false);
            }
        }
      }

      std::vector<unsigned> inv_gen(instructions.size(), -1U);
      for (unsigned idx = 0; idx < gen.size(); ++idx)
      {
        unsigned g = gen[idx];
#ifdef DEBUG_LEGION
        assert(inv_gen[g] == -1U || g == fence_completion_id);
#endif
        if (g != -1U && g < instructions.size() && inv_gen[g] == -1U)
          inv_gen[g] = idx;
      }
      std::vector<Instruction*> to_delete;
      std::vector<unsigned> new_gen(gen.size(), -1U);
      for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
          it != frontiers.end(); ++it)
        new_gen[it->second] = 0;
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
        if (used[idx])
        {
          Instruction *inst = instructions[idx];
          if (!trace->runtime->no_fence_elision)
          {
            if (inst->get_kind() == MERGE_EVENT)
            {
              MergeEvent *merge = inst->as_merge_event();
              if (merge->rhs.size() > 1)
                merge->rhs.erase(fence_completion_id);
            }
          }
          unsigned e = inv_gen[idx];
#ifdef DEBUG_LEGION
          assert(e == -1U || (e < new_gen.size() && new_gen[e] == -1U));
#endif
          if (e != -1U)
            new_gen[e] = new_instructions.size();
          new_instructions.push_back(inst);
        }
        else
          to_delete.push_back(instructions[idx]);
      instructions.swap(new_instructions);
      gen.swap(new_gen);
      for (unsigned idx = 0; idx < to_delete.size(); ++idx)
        delete to_delete[idx];
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::prepare_parallel_replay(
                                               const std::vector<unsigned> &gen)
    //--------------------------------------------------------------------------
    {
      slices.resize(replay_parallelism);
      std::map<TraceLocalID, unsigned> slice_indices_by_owner;
      std::vector<unsigned> slice_indices_by_inst;
      slice_indices_by_inst.resize(instructions.size());

#ifdef DEBUG_LEGION
      for (unsigned idx = 1; idx < instructions.size(); ++idx)
        slice_indices_by_inst[idx] = -1U;
#endif
      bool round_robin_for_tasks = false;

      std::set<Processor> distinct_targets;
      for (CachedMappings::iterator it = cached_mappings.begin(); it !=
           cached_mappings.end(); ++it)
        distinct_targets.insert(it->second.target_procs[0]);
      round_robin_for_tasks = distinct_targets.size() < replay_parallelism;

      unsigned next_slice_id = 0;
      for (std::map<TraceLocalID,std::pair<unsigned,bool> >::const_iterator 
            it = memo_entries.begin(); it != memo_entries.end(); ++it)
      {
        unsigned slice_index = -1U;
        if (!round_robin_for_tasks && it->second.second)
        {
          CachedMappings::iterator finder = cached_mappings.find(it->first);
#ifdef DEBUG_LEGION
          assert(finder != cached_mappings.end());
          assert(finder->second.target_procs.size() > 0);
#endif
          slice_index =
            finder->second.target_procs[0].id % replay_parallelism;
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(slice_indices_by_owner.find(it->first) ==
              slice_indices_by_owner.end());
#endif
          slice_index = next_slice_id;
          next_slice_id = (next_slice_id + 1) % replay_parallelism;
        }

#ifdef DEBUG_LEGION
        assert(slice_index != -1U);
#endif
        slice_indices_by_owner[it->first] = slice_index;
      }
      // Make sure that event creations and triggers are in the same slice
      std::map<unsigned/*user event*/,unsigned/*slice*/> user_event_slices;
      // Keep track of these so that we don't end up leaking them
      std::vector<Instruction*> crossing_instructions;
      std::map<unsigned,std::pair<unsigned,unsigned> > crossing_counts;
      for (unsigned idx = 1; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        const TraceLocalID &owner = inst->owner;
        std::map<TraceLocalID, unsigned>::iterator finder =
          slice_indices_by_owner.find(owner);
        unsigned slice_index = -1U;
        const InstructionKind kind = inst->get_kind();
        if (finder != slice_indices_by_owner.end())
          slice_index = finder->second;
        else if (kind == TRIGGER_EVENT)
        {
          // Find the slice where the event creation was assigned
          // and make sure that we end up on the same slice
          TriggerEvent *trigger = inst->as_trigger_event(); 
          std::map<unsigned,unsigned>::iterator finder = 
            user_event_slices.find(trigger->lhs);
#ifdef DEBUG_LEGION
          assert(finder != user_event_slices.end());
#endif
          slice_index = finder->second;
          user_event_slices.erase(finder);
        }
        else
        {
          slice_index = next_slice_id;
          next_slice_id = (next_slice_id + 1) % replay_parallelism;
          if (kind == CREATE_AP_USER_EVENT)
          {
            // Save which slice this is on so the later trigger will
            // get recorded on the same slice
            CreateApUserEvent *create = inst->as_create_ap_user_event();
#ifdef DEBUG_LEGION
            assert(user_event_slices.find(create->lhs) ==
                    user_event_slices.end());
#endif
            user_event_slices[create->lhs] = slice_index;
          }
        }
        slices[slice_index].push_back(inst);
        slice_indices_by_inst[idx] = slice_index;

        if (kind == MERGE_EVENT)
        {
          MergeEvent *merge = inst->as_merge_event();
          unsigned crossing_found = false;
          std::set<unsigned> new_rhs;
          for (std::set<unsigned>::iterator it = merge->rhs.begin();
               it != merge->rhs.end(); ++it)
          {
            unsigned rh = *it;
            if (gen[rh] == 0)
              new_rhs.insert(rh);
            else
            {
#ifdef DEBUG_LEGION
              assert(gen[rh] != -1U);
#endif
              unsigned generator_slice = slice_indices_by_inst[gen[rh]];
#ifdef DEBUG_LEGION
              assert(generator_slice != -1U);
#endif
              if (generator_slice != slice_index)
              {
                crossing_found = true;
                std::map<unsigned, std::pair<unsigned,unsigned> >::iterator
                  finder = crossing_counts.find(rh);
                if (finder != crossing_counts.end())
                {
                  new_rhs.insert(finder->second.first);
                  finder->second.second += 1;
                }
                else
                {
                  unsigned new_crossing_event = events.size();
                  events.resize(events.size() + 1);
                  crossing_counts[rh] = 
                    std::pair<unsigned,unsigned>(new_crossing_event,1/*count*/);
                  new_rhs.insert(new_crossing_event);
                  TriggerEvent *crossing = new TriggerEvent(*this,
                      new_crossing_event, rh, instructions[gen[rh]]->owner);
                  slices[generator_slice].push_back(crossing);
                  crossing_instructions.push_back(crossing);
                }
              }
              else
                new_rhs.insert(rh);
            }
          }

          if (crossing_found)
            merge->rhs.swap(new_rhs);
        }
        else
        {
          switch (kind)
          {
            case TRIGGER_EVENT:
              {
                parallelize_replay_event(inst->as_trigger_event()->rhs,
                    slice_index, gen, slice_indices_by_inst,
                    crossing_counts, crossing_instructions);
                break;
              }
            case ISSUE_COPY:
              {
                parallelize_replay_event(
                    inst->as_issue_copy()->precondition_idx,
                    slice_index, gen, slice_indices_by_inst,
                    crossing_counts, crossing_instructions);
                break;
              }
            case ISSUE_FILL:
              {
                parallelize_replay_event(
                    inst->as_issue_fill()->precondition_idx,
                    slice_index, gen, slice_indices_by_inst,
                    crossing_counts, crossing_instructions);
                break;
              }
            case ISSUE_ACROSS:
              {
                IssueAcross *across = inst->as_issue_across();
                parallelize_replay_event(across->copy_precondition,
                    slice_index, gen, slice_indices_by_inst,
                    crossing_counts, crossing_instructions);
                if (across->collective_precondition != 0)
                  parallelize_replay_event(across->collective_precondition,
                      slice_index, gen, slice_indices_by_inst,
                      crossing_counts, crossing_instructions);
                if (across->src_indirect_precondition != 0)
                  parallelize_replay_event(across->src_indirect_precondition,
                      slice_index, gen, slice_indices_by_inst,
                      crossing_counts, crossing_instructions);
                if (across->dst_indirect_precondition != 0)
                  parallelize_replay_event(across->dst_indirect_precondition,
                      slice_index, gen, slice_indices_by_inst,
                      crossing_counts, crossing_instructions);
                break;
              }
            case SET_EFFECTS:
              {
                parallelize_replay_event(inst->as_set_effects()->rhs,
                    slice_index, gen, slice_indices_by_inst,
                    crossing_counts, crossing_instructions);
                break;
              }
            case COMPLETE_REPLAY:
              {
                parallelize_replay_event(inst->as_complete_replay()->rhs,
                    slice_index, gen, slice_indices_by_inst,
                    crossing_counts, crossing_instructions);
                break;
              }
            default:
              {
                break;
              }
          }
        }
      }
#ifdef DEBUG_LEGION
      assert(user_event_slices.empty());
#endif
      // Update the crossing events and their counts
      if (!crossing_counts.empty())
      {
        for (std::map<unsigned,std::pair<unsigned,unsigned> >::const_iterator
              it = crossing_counts.begin(); it != crossing_counts.end(); it++)
          crossing_events.insert(it->second);
      }
      // Append any new crossing instructions to the list of instructions
      // so that they will still be deleted when the template is
      if (!crossing_instructions.empty())
        instructions.insert(instructions.end(),
            crossing_instructions.begin(), crossing_instructions.end());
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::parallelize_replay_event(unsigned &event_to_check,
              unsigned slice_index, const std::vector<unsigned> &gen,
              const std::vector<unsigned> &slice_indices_by_inst,
              std::map<unsigned,std::pair<unsigned,unsigned> > &crossing_counts,
              std::vector<Instruction*> &crossing_instructions)
    //--------------------------------------------------------------------------
    {
      unsigned g = gen[event_to_check];
#ifdef DEBUG_LEGION
      assert(g != -1U && g < instructions.size());
#endif
      unsigned generator_slice = slice_indices_by_inst[g];
#ifdef DEBUG_LEGION
      assert(generator_slice != -1U);
#endif
      if (generator_slice != slice_index)
      {
        std::map<unsigned, std::pair<unsigned,unsigned> >::iterator
          finder = crossing_counts.find(event_to_check);
        if (finder != crossing_counts.end())
        {
          event_to_check = finder->second.first;
          finder->second.second += 1;
        }
        else
        {
          unsigned new_crossing_event = events.size();
          events.resize(events.size() + 1);
          crossing_counts[event_to_check] =
            std::pair<unsigned,unsigned>(new_crossing_event, 1/*count*/);
          TriggerEvent *crossing = new TriggerEvent(*this,
              new_crossing_event, event_to_check, instructions[g]->owner); 
          event_to_check = new_crossing_event;
          slices[generator_slice].push_back(crossing);
          crossing_instructions.push_back(crossing);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::transitive_reduction(bool deferred)
    //--------------------------------------------------------------------------
    {
      // Transitive reduction inspired by Klaus Simon,
      // "An improved algorithm for transitive closure on acyclic digraphs"

      // First, build a DAG and find nodes with no incoming edges
      std::vector<unsigned> topo_order;
      topo_order.reserve(instructions.size());
      std::vector<unsigned> inv_topo_order(events.size(), -1U);
      std::vector<std::vector<unsigned> > incoming(events.size());
      std::vector<std::vector<unsigned> > outgoing(events.size());

      for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
           it != frontiers.end(); ++it)
      {
        inv_topo_order[it->second] = topo_order.size();
        topo_order.push_back(it->second);
      }

      std::map<TraceLocalID, GetTermEvent*> term_insts;
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        switch (inst->get_kind())
        {
          // Pass these instructions as their events will be added later
          case GET_TERM_EVENT :
            {
#ifdef DEBUG_LEGION
              assert(inst->as_get_term_event() != NULL);
#endif
              term_insts[inst->owner] = inst->as_get_term_event();
              break;
            }
          case CREATE_AP_USER_EVENT :
            {
              break;
            }
          case TRIGGER_EVENT :
            {
              TriggerEvent *trigger = inst->as_trigger_event();
              incoming[trigger->lhs].push_back(trigger->rhs);
              outgoing[trigger->rhs].push_back(trigger->lhs);
              break;
            }
          case MERGE_EVENT :
            {
              MergeEvent *merge = inst->as_merge_event();
              for (std::set<unsigned>::iterator it = merge->rhs.begin();
                   it != merge->rhs.end(); ++it)
              {
                incoming[merge->lhs].push_back(*it);
                outgoing[*it].push_back(merge->lhs);
              }
              break;
            }
          case ISSUE_COPY :
            {
              IssueCopy *copy = inst->as_issue_copy();
              incoming[copy->lhs].push_back(copy->precondition_idx);
              outgoing[copy->precondition_idx].push_back(copy->lhs);
              break;
            }
          case ISSUE_FILL :
            {
              IssueFill *fill = inst->as_issue_fill();
              incoming[fill->lhs].push_back(fill->precondition_idx);
              outgoing[fill->precondition_idx].push_back(fill->lhs);
              break;
            }
          case ISSUE_ACROSS:
            {
              IssueAcross *across = inst->as_issue_across();
              incoming[across->lhs].push_back(across->copy_precondition);
              outgoing[across->copy_precondition].push_back(across->lhs);
              if (across->collective_precondition != 0)
              {
                incoming[across->lhs].push_back(
                    across->collective_precondition);
                outgoing[across->collective_precondition].push_back(
                    across->lhs);
              }
              if (across->src_indirect_precondition != 0)
              {
                incoming[across->lhs].push_back(
                    across->src_indirect_precondition);
                outgoing[across->src_indirect_precondition].push_back(
                    across->lhs);
              }
              if (across->dst_indirect_precondition != 0)
              {
                incoming[across->lhs].push_back(
                    across->dst_indirect_precondition);
                outgoing[across->dst_indirect_precondition].push_back(
                    across->lhs);
              }
              break;
            }
          case SET_OP_SYNC_EVENT :
            {
              SetOpSyncEvent *sync = inst->as_set_op_sync_event();
              inv_topo_order[sync->lhs] = topo_order.size();
              topo_order.push_back(sync->lhs);
              break;
            }
          case SET_EFFECTS :
            {
              break;
            }
          case ASSIGN_FENCE_COMPLETION :
            {
              inv_topo_order[fence_completion_id] = topo_order.size();
              topo_order.push_back(fence_completion_id);
              break;
            }
          case COMPLETE_REPLAY :
            {
              CompleteReplay *replay = inst->as_complete_replay();
#ifdef DEBUG_LEGION
              assert(term_insts.find(replay->owner) != term_insts.end());
#endif
              GetTermEvent *term = term_insts[replay->owner];
              unsigned lhs = term->lhs;
#ifdef DEBUG_LEGION
              assert(lhs != -1U);
#endif
              incoming[lhs].push_back(replay->rhs);
              outgoing[replay->rhs].push_back(lhs);
              break;
            }
          default:
            {
              assert(false);
              break;
            }
        }
      }

      // Second, do a toposort on nodes via BFS
      std::vector<unsigned> remaining_edges(incoming.size());
      for (unsigned idx = 0; idx < incoming.size(); ++idx)
        remaining_edges[idx] = incoming[idx].size();

      unsigned idx = 0;
      while (idx < topo_order.size())
      {
        unsigned node = topo_order[idx];
#ifdef DEBUG_LEGION
        assert(remaining_edges[node] == 0);
#endif
        const std::vector<unsigned> &out = outgoing[node];
        for (unsigned oidx = 0; oidx < out.size(); ++oidx)
        {
          unsigned next = out[oidx];
          if (--remaining_edges[next] == 0)
          {
            inv_topo_order[next] = topo_order.size();
            topo_order.push_back(next);
          }
        }
        ++idx;
      }

      // Third, construct a chain decomposition
      unsigned num_chains = 0;
      std::vector<unsigned> chain_indices(topo_order.size(), -1U);

      int pos = chain_indices.size() - 1;
      while (true)
      {
        while (pos >= 0 && chain_indices[pos] != -1U)
          --pos;
        if (pos < 0) break;
        unsigned curr = topo_order[pos];
        while (incoming[curr].size() > 0)
        {
          chain_indices[inv_topo_order[curr]] = num_chains;
          const std::vector<unsigned> &in = incoming[curr];
          bool found = false;
          for (unsigned iidx = 0; iidx < in.size(); ++iidx)
          {
            unsigned next = in[iidx];
            if (chain_indices[inv_topo_order[next]] == -1U)
            {
              found = true;
              curr = next;
              chain_indices[inv_topo_order[curr]] = num_chains;
              break;
            }
          }
          if (!found) break;
        }
        chain_indices[inv_topo_order[curr]] = num_chains;
        ++num_chains;
      }

      // Fourth, find the frontiers of chains that are connected to each node
      std::vector<std::vector<int> > all_chain_frontiers(topo_order.size());
      std::vector<std::vector<unsigned> > incoming_reduced(topo_order.size());
      for (unsigned idx = 0; idx < topo_order.size(); ++idx)
      {
        std::vector<int> chain_frontiers(num_chains, -1);
        const std::vector<unsigned> &in = incoming[topo_order[idx]];
        std::vector<unsigned> &in_reduced = incoming_reduced[idx];
        for (unsigned iidx = 0; iidx < in.size(); ++iidx)
        {
          int rank = inv_topo_order[in[iidx]];
#ifdef DEBUG_LEGION
          assert((unsigned)rank < idx);
#endif
          const std::vector<int> &pred_chain_frontiers =
            all_chain_frontiers[rank];
          for (unsigned k = 0; k < num_chains; ++k)
            chain_frontiers[k] =
              std::max(chain_frontiers[k], pred_chain_frontiers[k]);
        }
        for (unsigned iidx = 0; iidx < in.size(); ++iidx)
        {
          int rank = inv_topo_order[in[iidx]];
          unsigned chain_idx = chain_indices[rank];
          if (chain_frontiers[chain_idx] < rank)
          {
            in_reduced.push_back(in[iidx]);
            chain_frontiers[chain_idx] = rank;
          }
        }
#ifdef DEBUG_LEGION
        assert(in.size() == 0 || in_reduced.size() > 0);
#endif
        all_chain_frontiers[idx].swap(chain_frontiers);
      }

      // Lastly, suppress transitive dependences using chains
      if (deferred)
      {
        // Save the data structures for finalizing the transitive
        // reduction for later, the next replay will incorporate them
        std::vector<unsigned> *inv_topo_order_copy = 
          new std::vector<unsigned>();
        inv_topo_order_copy->swap(inv_topo_order);
        std::vector<std::vector<unsigned> > *in_reduced_copy = 
          new std::vector<std::vector<unsigned> >();
        in_reduced_copy->swap(incoming_reduced);
        // Write them to the members
        pending_inv_topo_order.store(inv_topo_order_copy);
        // Need memory fence so writes happen in this order
        __sync_synchronize();
        pending_transitive_reduction.store(in_reduced_copy);
      }
      else
        finalize_transitive_reduction(inv_topo_order, incoming_reduced);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::finalize_transitive_reduction(
                    const std::vector<unsigned> &inv_topo_order,
                    const std::vector<std::vector<unsigned> > &incoming_reduced)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
        if (instructions[idx]->get_kind() == MERGE_EVENT)
        {
          MergeEvent *merge = instructions[idx]->as_merge_event();
          unsigned order = inv_topo_order[merge->lhs];
#ifdef DEBUG_LEGION
          assert(order != -1U);
#endif
          const std::vector<unsigned> &in_reduced = incoming_reduced[order];
          if (in_reduced.size() == merge->rhs.size())
          {
#ifdef DEBUG_LEGION
            for (unsigned iidx = 0; iidx < in_reduced.size(); ++iidx)
              assert(merge->rhs.find(in_reduced[iidx]) != merge->rhs.end());
#endif
            continue;
          }
#ifdef DEBUG_LEGION
          std::set<unsigned> new_rhs;
          for (unsigned iidx = 0; iidx < in_reduced.size(); ++iidx)
          {
            assert(merge->rhs.find(in_reduced[iidx]) != merge->rhs.end());
            new_rhs.insert(in_reduced[iidx]);
          }
#else
          std::set<unsigned> new_rhs(in_reduced.begin(), in_reduced.end());
#endif
          // Remove any references to crossing events which are no longer needed
          if (!crossing_events.empty())
          {
            for (std::set<unsigned>::const_iterator it =
                  merge->rhs.begin(); it != merge->rhs.end(); it++)
            {
              std::map<unsigned,unsigned>::iterator finder =
                crossing_events.find(*it);
              if ((finder != crossing_events.end()) &&
                  (new_rhs.find(*it) == new_rhs.end()))
              {
#ifdef DEBUG_LEGION
                assert(finder->second > 0);
#endif
                finder->second--;
              }
            }
          }
          merge->rhs.swap(new_rhs);
        }
      // Remove any crossing instructions from the slices that are no
      // longer needed because the transitive reduction eliminated the
      // need for the edge
      for (std::map<unsigned,unsigned>::iterator it =
            crossing_events.begin(); it != crossing_events.end(); /*nothing*/)
      {
        if (it->second == 0)
        {
          // No more references to this crossing instruction so remove it
          bool found = false;
          for (std::vector<std::vector<Instruction*> >::iterator sit =
                slices.begin(); sit != slices.end(); sit++)
          {
            for (std::vector<Instruction*>::iterator iit =
                  sit->begin(); iit != sit->end(); iit++)
            {
              TriggerEvent *trigger = (*iit)->as_trigger_event();
              if (trigger == NULL)
                continue;
              if (trigger->lhs == it->first)
              {
                sit->erase(iit);
                found = true;
                break;
              }
            }
            if (found)
              break;
          }
          std::map<unsigned,unsigned>::iterator to_delete = it++;
          crossing_events.erase(to_delete);
        }
        else
          it++;
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::propagate_copies(std::vector<unsigned> *gen)
    //--------------------------------------------------------------------------
    {
      std::vector<int> substs(events.size(), -1);
      std::vector<Instruction*> new_instructions;
      new_instructions.reserve(instructions.size());
      std::set<Instruction*> to_prune;
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        if (instructions[idx]->get_kind() == MERGE_EVENT)
        {
          MergeEvent *merge = instructions[idx]->as_merge_event();
#ifdef DEBUG_LEGION
          assert(merge->rhs.size() > 0);
#endif
          if (merge->rhs.size() == 1)
          {
            substs[merge->lhs] = *merge->rhs.begin();
#ifdef DEBUG_LEGION
            assert(merge->lhs != (unsigned)substs[merge->lhs]);
#endif
            if (gen == NULL)
              to_prune.insert(inst);
            else
              delete inst;
          }
          else
            new_instructions.push_back(inst);
        }
        else
          new_instructions.push_back(inst);
      }

      if (instructions.size() == new_instructions.size()) return;

      instructions.swap(new_instructions);

      std::vector<unsigned> new_gen((gen == NULL) ? 0 : gen->size(), -1U);
      if (gen != NULL)
      {
        for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
            it != frontiers.end(); ++it)
          new_gen[it->second] = 0;
      }

      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        int lhs = -1;
        switch (inst->get_kind())
        {
          case GET_TERM_EVENT:
            {
              GetTermEvent *term = inst->as_get_term_event();
              lhs = term->lhs;
              break;
            }
          case CREATE_AP_USER_EVENT:
            {
              CreateApUserEvent *create = inst->as_create_ap_user_event();
              lhs = create->lhs;
              break;
            }
          case TRIGGER_EVENT:
            {
              TriggerEvent *trigger = inst->as_trigger_event();
              int subst = substs[trigger->rhs];
              if (subst >= 0) trigger->rhs = (unsigned)subst;
              break;
            }
          case MERGE_EVENT:
            {
              MergeEvent *merge = inst->as_merge_event();
              std::set<unsigned> new_rhs;
              for (std::set<unsigned>::iterator it = merge->rhs.begin();
                   it != merge->rhs.end(); ++it)
              {
                int subst = substs[*it];
                if (subst >= 0) new_rhs.insert((unsigned)subst);
                else new_rhs.insert(*it);
              }
              merge->rhs.swap(new_rhs);
              lhs = merge->lhs;
              break;
            }
          case ISSUE_COPY:
            {
              IssueCopy *copy = inst->as_issue_copy();
              int subst = substs[copy->precondition_idx];
              if (subst >= 0) copy->precondition_idx = (unsigned)subst;
              lhs = copy->lhs;
              break;
            }
          case ISSUE_FILL:
            {
              IssueFill *fill = inst->as_issue_fill();
              int subst = substs[fill->precondition_idx];
              if (subst >= 0) fill->precondition_idx = (unsigned)subst;
              lhs = fill->lhs;
              break;
            }
          case ISSUE_ACROSS:
            {
              IssueAcross *across = inst->as_issue_across();
              int subst = substs[across->copy_precondition];
              if (subst >= 0) across->copy_precondition= (unsigned)subst;
              if (across->collective_precondition != 0)
              {
                int subst = substs[across->collective_precondition];
                if (subst >= 0) 
                  across->collective_precondition = (unsigned)subst;
              }
              if (across->src_indirect_precondition != 0)
              {
                int subst = substs[across->src_indirect_precondition];
                if (subst >= 0) 
                  across->src_indirect_precondition = (unsigned)subst;
              }
              if (across->dst_indirect_precondition != 0)
              {
                int subst = substs[across->dst_indirect_precondition];
                if (subst >= 0) 
                  across->dst_indirect_precondition = (unsigned)subst;
              }
              lhs = across->lhs;
              break;
            }
          case SET_EFFECTS:
            {
              SetEffects *effects = inst->as_set_effects();
              int subst = substs[effects->rhs];
              if (subst >= 0) effects->rhs = (unsigned)subst;
              break;
            }
          case SET_OP_SYNC_EVENT:
            {
              SetOpSyncEvent *sync = inst->as_set_op_sync_event();
              lhs = sync->lhs;
              break;
            }
          case ASSIGN_FENCE_COMPLETION:
            {
              lhs = fence_completion_id;
              break;
            }
          case COMPLETE_REPLAY:
            {
              CompleteReplay *replay = inst->as_complete_replay();
              int subst = substs[replay->rhs];
              if (subst >= 0) replay->rhs = (unsigned)subst;
              break;
            }
          default:
            {
              break;
            }
        }
        if ((lhs != -1) && (gen != NULL))
          new_gen[lhs] = idx;
      }
      if (gen != NULL)
        gen->swap(new_gen);
      if (!to_prune.empty())
      {
#ifdef DEBUG_LEGION
        assert(!slices.empty());
#endif
        // Remove these instructions from any slices and then delete them
        for (unsigned idx = 0; idx < slices.size(); idx++)
        {
          std::vector<Instruction*> &slice = slices[idx];
          for (std::vector<Instruction*>::iterator it =
                slice.begin(); it != slice.end(); /*nothing*/)
          {
            std::set<Instruction*>::iterator finder =
              to_prune.find(*it);
            if (finder != to_prune.end())
            {
              it = slice.erase(it);
              delete *finder;
              to_prune.erase(finder);
              if (to_prune.empty())
                break;
            }
            else
              it++;
          }
          if (to_prune.empty())
            break;
        }
#ifdef DEBUG_LEGION
        assert(to_prune.empty());
#endif
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::eliminate_dead_code(std::vector<unsigned> &gen)
    //--------------------------------------------------------------------------
    {
      std::vector<bool> used(instructions.size(), false);
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        InstructionKind kind = inst->get_kind();
        // We only eliminate two kinds of instructions:
        // GetTermEvent and SetOpSyncEvent
        used[idx] = kind != SET_OP_SYNC_EVENT;
        switch (kind)
        {
          case MERGE_EVENT:
            {
              MergeEvent *merge = inst->as_merge_event();
              for (std::set<unsigned>::iterator it = merge->rhs.begin();
                   it != merge->rhs.end(); ++it)
              {
#ifdef DEBUG_LEGION
                assert(gen[*it] != -1U);
#endif
                used[gen[*it]] = true;
              }
              break;
            }
          case TRIGGER_EVENT:
            {
              TriggerEvent *trigger = inst->as_trigger_event();
#ifdef DEBUG_LEGION
              assert(gen[trigger->rhs] != -1U);
#endif
              used[gen[trigger->rhs]] = true;
              break;
            }
          case ISSUE_COPY:
            {
              IssueCopy *copy = inst->as_issue_copy();
#ifdef DEBUG_LEGION
              assert(gen[copy->precondition_idx] != -1U);
#endif
              used[gen[copy->precondition_idx]] = true;
              break;
            }
          case ISSUE_FILL:
            {
              IssueFill *fill = inst->as_issue_fill();
#ifdef DEBUG_LEGION
              assert(gen[fill->precondition_idx] != -1U);
#endif
              used[gen[fill->precondition_idx]] = true;
              break;
            }
          case ISSUE_ACROSS:
            {
              IssueAcross *across = inst->as_issue_across();
#ifdef DEBUG_LEGION
              assert(gen[across->copy_precondition] != -1U);
#endif
              used[gen[across->copy_precondition]] = true;
              if (across->collective_precondition != 0)
              {
#ifdef DEBUG_LEGION
                assert(gen[across->collective_precondition] != -1U);
#endif
                used[gen[across->collective_precondition]] = true;
              }
              if (across->src_indirect_precondition!= 0)
              {
#ifdef DEBUG_LEGION
                assert(gen[across->src_indirect_precondition] != -1U);
#endif
                used[gen[across->src_indirect_precondition]] = true;
              }
              if (across->dst_indirect_precondition!= 0)
              {
#ifdef DEBUG_LEGION
                assert(gen[across->dst_indirect_precondition] != -1U);
#endif
                used[gen[across->dst_indirect_precondition]] = true;
              }
              break;
            }
          case SET_EFFECTS:
            {
              SetEffects *effects = inst->as_set_effects();
#ifdef DEBUG_LEGION
              assert(gen[effects->rhs] != -1U);
#endif
              used[gen[effects->rhs]] = true;
              break;
            }
          case COMPLETE_REPLAY:
            {
              CompleteReplay *complete = inst->as_complete_replay();
#ifdef DEBUG_LEGION
              assert(gen[complete->rhs] != -1U);
#endif
              used[gen[complete->rhs]] = true;
              break;
            }
          case GET_TERM_EVENT:
          case CREATE_AP_USER_EVENT:
          case SET_OP_SYNC_EVENT:
          case ASSIGN_FENCE_COMPLETION:
            {
              break;
            }
          default:
            {
              // unreachable
              assert(false);
            }
        }
      }
      for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
          it != frontiers.end(); ++it)
      {
        unsigned g = gen[it->first];
        if (g != -1U && g < instructions.size())
          used[g] = true;
      }

      std::vector<unsigned> inv_gen(instructions.size(), -1U);
      for (unsigned idx = 0; idx < gen.size(); ++idx)
      {
        unsigned g = gen[idx];
        if (g != -1U && g < instructions.size() && inv_gen[g] == -1U)
          inv_gen[g] = idx;
      }

      std::vector<Instruction*> new_instructions;
      std::vector<Instruction*> to_delete;
      std::vector<unsigned> new_gen(gen.size(), -1U);
      for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
          it != frontiers.end(); ++it)
        new_gen[it->second] = 0;
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        if (used[idx])
        {
          unsigned e = inv_gen[idx];
#ifdef DEBUG_LEGION
          assert(e == -1U || (e < new_gen.size() && new_gen[e] == -1U));
#endif
          if (e != -1U)
            new_gen[e] = new_instructions.size();
          new_instructions.push_back(instructions[idx]);
        }
        else
          to_delete.push_back(instructions[idx]);
      }

      instructions.swap(new_instructions);
      gen.swap(new_gen);
      for (unsigned idx = 0; idx < to_delete.size(); ++idx)
        delete to_delete[idx];
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::push_complete_replays(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < slices.size(); ++idx)
      {
        std::vector<Instruction*> &instructions = slices[idx];
        std::vector<Instruction*> new_instructions;
        new_instructions.reserve(instructions.size());
        std::vector<Instruction*> complete_replays;
        for (unsigned iidx = 0; iidx < instructions.size(); ++iidx)
        {
          Instruction *inst = instructions[iidx];
          if (inst->get_kind() == COMPLETE_REPLAY)
            complete_replays.push_back(inst);
          else
            new_instructions.push_back(inst);
        }
        new_instructions.insert(new_instructions.end(),
                                complete_replays.begin(),
                                complete_replays.end());
        instructions.swap(new_instructions);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::dump_template(void)
    //--------------------------------------------------------------------------
    {
      InnerContext *ctx = trace->logical_trace->ctx;
      log_tracing.info() << "#### " << replayable << " " << this << " Trace "
        << trace->logical_trace->tid << " for " << ctx->get_task_name()
        << " (UID " << ctx->get_unique_id() << ") ####";
      for (unsigned sidx = 0; sidx < replay_parallelism; ++sidx)
      {
        log_tracing.info() << "[Slice " << sidx << "]";
        dump_instructions(slices[sidx]);
      }
      for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
           it != frontiers.end(); ++it)
        log_tracing.info() << "  events[" << it->second << "] = events["
                           << it->first << "]";

      log_tracing.info() << "[Precondition]";
      pre.dump();
      for (FieldMaskSet<FillView>::const_iterator vit = pre_fill_views.begin();
           vit != pre_fill_views.end(); ++vit)
      {
        char *mask = vit->second.to_string();
        log_tracing.info() << "  Fill view: " << vit->first
                           << ", Field Mask: " << mask;
        free(mask);
      }
      pre_reductions.dump();

      log_tracing.info() << "[Postcondition]";
      post.dump();
      post_reductions.dump();

      log_tracing.info() << "[Consumed Reductions]";
      consumed_reductions.dump();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::dump_instructions(
                                  const std::vector<Instruction*> &instructions)
    //--------------------------------------------------------------------------
    {
      for (std::vector<Instruction*>::const_iterator it = instructions.begin();
           it != instructions.end(); ++it)
        log_tracing.info() << "  " << (*it)->to_string(operations.front());
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::pack_recorder(Serializer &rez,
                 std::set<RtEvent> &applied_events, const AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      rez.serialize(trace->runtime->address_space);
      rez.serialize(target);
      rez.serialize(this);
      RtUserEvent remote_applied = Runtime::create_rt_user_event();
      rez.serialize(remote_applied);
      rez.serialize(recording_done);
      applied_events.insert(remote_applied);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_mapper_output(const TraceLocalID &tlid,
                                            const Mapper::MapTaskOutput &output,
                              const std::deque<InstanceSet> &physical_instances,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
      assert(cached_mappings.find(tlid) == cached_mappings.end());
#endif
      CachedMapping &mapping = cached_mappings[tlid];
      // If you change the things recorded from output here then
      // you also need to change RemoteTraceRecorder::record_mapper_output
      mapping.target_procs = output.target_procs;
      mapping.chosen_variant = output.chosen_variant;
      mapping.task_priority = output.task_priority;
      mapping.postmap_task = output.postmap_task;
      mapping.physical_instances = physical_instances;
      WrapperReferenceMutator mutator(applied_events);
      for (std::deque<InstanceSet>::iterator it =
           mapping.physical_instances.begin(); it !=
           mapping.physical_instances.end(); ++it)
      {
        for (unsigned idx = 0; idx < it->size(); idx++)
        {
          const InstanceRef &ref = (*it)[idx];
          if (ref.is_virtual_ref())
            has_virtual_mapping = true;
          else
            ref.add_valid_reference(MAPPING_ACQUIRE_REF, &mutator);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::get_mapper_output(SingleTask *task,
                                             VariantID &chosen_variant,
                                             TaskPriority &task_priority,
                                             bool &postmap_task,
                              std::vector<Processor> &target_procs,
                              std::deque<InstanceSet> &physical_instances) const
    //--------------------------------------------------------------------------
    {
      const TraceLocalID op_key = task->get_trace_local_id();
      AutoLock t_lock(template_lock, 1, false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(is_replaying());
#endif
      CachedMappings::const_iterator finder = cached_mappings.find(op_key);
#ifdef DEBUG_LEGION
      assert(finder != cached_mappings.end());
#endif
      chosen_variant = finder->second.chosen_variant;
      task_priority = finder->second.task_priority;
      postmap_task = finder->second.postmap_task;
      target_procs = finder->second.target_procs;
      physical_instances = finder->second.physical_instances;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_get_term_event(Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo != NULL);
#endif
      const ApEvent lhs = memo->get_memo_completion();
      const bool fence = 
        (memo->get_memoizable_kind() == Operation::FENCE_OP_KIND);
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      unsigned lhs_ = convert_event(lhs);
      TraceLocalID key = record_memo_entry(memo, lhs_);
      insert_instruction(new GetTermEvent(*this, lhs_, key, fence));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_create_ap_user_event(
                                              ApUserEvent lhs, Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs.exists());
      assert(memo != NULL);
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      unsigned lhs_ = convert_event(lhs);
      user_events[lhs_] = lhs;
      insert_instruction(
          new CreateApUserEvent(*this, lhs_, find_trace_local_id(memo)));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_trigger_event(ApUserEvent lhs, ApEvent rhs,
                                                Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs.exists());
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      unsigned lhs_ = find_event(lhs);
      events.push_back(ApEvent());
      insert_instruction(new TriggerEvent(*this, lhs_, 
            rhs.exists() ? find_event(rhs) : fence_completion_id,
            find_trace_local_id(memo)));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs, ApEvent rhs_,
                                               Memoizable *memo)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> rhs;
      rhs.insert(rhs_);
      record_merge_events(lhs, rhs, memo);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs, ApEvent e1,
                                               ApEvent e2, Memoizable *memo)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> rhs;
      rhs.insert(e1);
      rhs.insert(e2);
      record_merge_events(lhs, rhs, memo);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs, ApEvent e1,
                                               ApEvent e2, ApEvent e3,
                                               Memoizable *memo)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> rhs;
      rhs.insert(e1);
      rhs.insert(e2);
      rhs.insert(e3);
      record_merge_events(lhs, rhs, memo);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs,
                                               const std::set<ApEvent>& rhs,
                                               Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo != NULL);
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      std::set<unsigned> rhs_;
      for (std::set<ApEvent>::const_iterator it = rhs.begin(); it != rhs.end();
           it++)
      {
        std::map<ApEvent, unsigned>::iterator finder = event_map.find(*it);
        if (finder != event_map.end())
          rhs_.insert(finder->second);
      }
      if (rhs_.size() == 0)
        rhs_.insert(fence_completion_id);

#ifndef LEGION_DISABLE_EVENT_PRUNING
      if (!lhs.exists() || (rhs.find(lhs) != rhs.end()))
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        if (rhs.find(lhs) != rhs.end())
          rename.trigger(lhs);
        else
          rename.trigger();
        lhs = ApEvent(rename);
      }
#endif

      insert_instruction(new MergeEvent(*this, convert_event(lhs), rhs_,
            memo->get_trace_local_id()));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs,
                                               const std::vector<ApEvent>& rhs,
                                               Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo != NULL);
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      std::set<unsigned> rhs_;
      for (std::vector<ApEvent>::const_iterator it =
            rhs.begin(); it != rhs.end(); it++)
      {
        std::map<ApEvent, unsigned>::iterator finder = event_map.find(*it);
        if (finder != event_map.end())
          rhs_.insert(finder->second);
      }
      if (rhs_.size() == 0)
        rhs_.insert(fence_completion_id);

#ifndef LEGION_DISABLE_EVENT_PRUNING
      if (!lhs.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        lhs = ApEvent(rename);
      }
      else
      {
        // Check for reuse
        for (unsigned idx = 0; idx < rhs.size(); idx++)
        {
          if (lhs != rhs[idx])
            continue;
          Realm::UserEvent rename(Realm::UserEvent::create_user_event());
          rename.trigger(lhs);
          lhs = ApEvent(rename);
          break;
        }
      }
#endif

      insert_instruction(new MergeEvent(*this, convert_event(lhs), rhs_,
            memo->get_trace_local_id()));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_copy(Memoizable *memo, ApEvent &lhs,
                                             IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField>& src_fields,
                                 const std::vector<CopySrcDstField>& dst_fields,
                                 const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                                             RegionTreeID src_tree_id,
                                             RegionTreeID dst_tree_id,
#endif
                                             ApEvent precondition,
                                             PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo != NULL);
#endif
      if (!lhs.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        lhs = ApEvent(rename);
      } 

      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      unsigned lhs_ = convert_event(lhs);
      insert_instruction(new IssueCopy(
            *this, lhs_, expr, find_trace_local_id(memo),
            src_fields, dst_fields, reservations,
#ifdef LEGION_SPY
            src_tree_id, dst_tree_id,
#endif
            find_event(precondition))); 
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_across(Memoizable *memo, ApEvent &lhs,
                                              ApEvent collective_precondition,
                                              ApEvent copy_precondition,
                                              ApEvent src_indirect_precondition,
                                              ApEvent dst_indirect_precondition,
                                              CopyAcrossExecutor *executor)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo != NULL);
#endif
      if (!lhs.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        lhs = ApEvent(rename);
      } 

      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      unsigned lhs_ = convert_event(lhs);
      unsigned copy_pre = find_event(copy_precondition);
      unsigned collective_pre = 0, src_indirect_pre = 0, dst_indirect_pre = 0;
      if (collective_precondition.exists())
        collective_pre = find_event(collective_precondition);
      if (src_indirect_precondition.exists())
        src_indirect_pre = find_event(src_indirect_precondition);
      if (dst_indirect_precondition.exists())
        dst_indirect_pre = find_event(dst_indirect_precondition);
      IssueAcross *across = new IssueAcross(*this, lhs_,copy_pre,collective_pre,
       src_indirect_pre, dst_indirect_pre, find_trace_local_id(memo), executor);
      across_copies.push_back(across);
      insert_instruction(across);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_fill(Memoizable *memo, ApEvent &lhs,
                                             IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField> &fields,
                                             const void *fill_value, 
                                             size_t fill_size,
#ifdef LEGION_SPY
                                             FieldSpace handle,
                                             RegionTreeID tree_id,
#endif
                                             ApEvent precondition,
                                             PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo != NULL);
#endif
      if (!lhs.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        lhs = ApEvent(rename);
      }

      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      unsigned lhs_ = convert_event(lhs);
      insert_instruction(new IssueFill(*this, lhs_, expr,
                                       find_trace_local_id(memo),
                                       fields, fill_value, fill_size, 
#ifdef LEGION_SPY
                                       handle, tree_id,
#endif
                                       find_event(precondition))); 
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_op_view(Memoizable *memo,
                                          unsigned idx,
                                          InstanceView *view,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          bool update_validity)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo != NULL);
#endif
      // Do this part before we take the lock
      LegionList<FieldSet<EquivalenceSet*> > eqs;
      if (update_validity)
      {
        FieldMaskSet<EquivalenceSet> eq_sets;
        memo->find_equivalence_sets(trace->runtime, idx, user_mask, eq_sets);
        eq_sets.compute_field_sets(user_mask, eqs);
      }

      AutoLock tpl_lock(template_lock);
      TraceLocalID op_key = find_trace_local_id(memo);
      unsigned entry = find_memo_entry(memo);

      FieldMaskSet<IndexSpaceExpression> &views = op_views[op_key][view];
      for (LegionList<FieldSet<EquivalenceSet*> >::iterator it =
           eqs.begin(); it != eqs.end(); ++it)
      {
        FieldMask mask = it->set_mask & user_mask;
        for (std::set<EquivalenceSet*>::iterator eit = it->elements.begin();
             eit != it->elements.end(); ++eit)
        {
          IndexSpaceExpression *expr = (*eit)->set_expr;
          views.insert(expr, mask);
          if (update_validity)
          {
            update_valid_views(view, *eit, usage, mask, true);
            add_view_user(view, usage, entry, expr, mask);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_post_fill_view(
                                     FillView *view, const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      post_fill_views.insert(view, user_mask);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_fill_views(ApEvent lhs, Memoizable *memo,
                                 unsigned idx, IndexSpaceExpression *expr,
                                 const FieldMaskSet<FillView> &tracing_srcs,
                                 const FieldMaskSet<InstanceView> &tracing_dsts,
                                 std::set<RtEvent> &applied_events,
                                 bool reduction_initialization)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo != NULL);
#endif
      // Do this before we take the lock
      LegionList<FieldSet<EquivalenceSet*> > eqs;
      {
        FieldMaskSet<EquivalenceSet> eq_sets;
        const FieldMask &dst_mask = tracing_dsts.get_valid_mask();
        memo->find_equivalence_sets(trace->runtime, idx, dst_mask, eq_sets);
        eq_sets.compute_field_sets(dst_mask, eqs);
      }

      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      const unsigned lhs_ = find_event(lhs);
      // Don't record fill views for initializing reduction 
      // istances since since we don't need to track them
      if (!reduction_initialization)
        record_fill_views(tracing_srcs);
      record_views(lhs_, expr, RegionUsage(LEGION_WRITE_ONLY, 
            LEGION_EXCLUSIVE, 0), tracing_dsts, eqs);
      record_expression_views(copy_views[lhs_], expr, tracing_dsts);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_copy_views(ApEvent lhs, Memoizable *memo,
                                 unsigned src_idx, unsigned dst_idx,
                                 IndexSpaceExpression *expr,
                                 const FieldMaskSet<InstanceView> &tracing_srcs,
                                 const FieldMaskSet<InstanceView> &tracing_dsts,
                                 PrivilegeMode src_mode, PrivilegeMode dst_mode,
                                 bool src_indirect, bool dst_indirect,
                                 std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo != NULL);
#endif
      LegionList<FieldSet<EquivalenceSet*> > src_eqs, dst_eqs;
      // Get these before we take the lock
      {
        FieldMaskSet<EquivalenceSet> eq_sets;
        const FieldMask &src_mask = tracing_srcs.get_valid_mask();
        memo->find_equivalence_sets(trace->runtime, src_idx, src_mask, eq_sets);
        eq_sets.compute_field_sets(src_mask, src_eqs);
      }
      {
        FieldMaskSet<EquivalenceSet> eq_sets;
        const FieldMask &dst_mask = tracing_dsts.get_valid_mask();
        memo->find_equivalence_sets(trace->runtime, dst_idx, dst_mask, eq_sets);
        eq_sets.compute_field_sets(dst_mask, dst_eqs);
      }

      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      const unsigned lhs_ = find_event(lhs);
      record_views(lhs_, expr, RegionUsage(src_mode, 
            LEGION_EXCLUSIVE, 0), tracing_srcs, src_eqs);
      record_expression_views(src_indirect ? 
          src_indirect_views[lhs_] : copy_views[lhs_], expr, tracing_srcs);
      record_views(lhs_, expr, RegionUsage(dst_mode, 
            LEGION_EXCLUSIVE, 0), tracing_dsts, dst_eqs);
      record_expression_views(dst_indirect ?
          dst_indirect_views[lhs_] : copy_views[lhs_], expr, tracing_dsts);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_indirect_views(ApEvent indirect_done,
                            ApEvent all_done, Memoizable *memo, unsigned index,
                            IndexSpaceExpression *expr,
                            const FieldMaskSet<InstanceView> &tracing_views,
                            std::set<RtEvent> &applied, PrivilegeMode privilege)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo != NULL);
#endif
      LegionList<FieldSet<EquivalenceSet*> > eqs;
      // Get these before we take the lock
      {
        FieldMaskSet<EquivalenceSet> eq_sets;
        const FieldMask &view_mask = tracing_views.get_valid_mask();
        memo->find_equivalence_sets(trace->runtime, index, view_mask, eq_sets);
        eq_sets.compute_field_sets(view_mask, eqs);
      }

      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      const unsigned indirect = find_event(indirect_done);
      const unsigned all = find_event(all_done);
      // The thing about indirect views is that the event for which they
      // are done being used is not always the same as the indirect copy
      // that generated them because they can be collective, so we need to
      // record the summary event for when all the indirect copies are done
      // for their view user
      record_views(all, expr, RegionUsage(privilege,
            LEGION_EXCLUSIVE, 0), tracing_views, eqs);
      record_expression_views(copy_views[indirect], expr, tracing_views);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_views(unsigned entry,
                                        IndexSpaceExpression *expr,
                                        const RegionUsage &usage,
                                        const FieldMaskSet<InstanceView> &views,
                              const LegionList<FieldSet<EquivalenceSet*> > &eqs)
    //--------------------------------------------------------------------------
    {
      RegionTreeForest *forest = trace->runtime->forest;
      for (FieldMaskSet<InstanceView>::const_iterator vit = views.begin();
            vit != views.end(); ++vit)
      {
        for (LegionList<FieldSet<EquivalenceSet*> >::const_iterator 
              it = eqs.begin(); it != eqs.end(); ++it)
        {
          const FieldMask mask = it->set_mask & vit->second;
          if (!mask)
            continue;
          for (std::set<EquivalenceSet*>::const_iterator eit = 
                it->elements.begin(); eit != it->elements.end(); ++eit)
          {
            // Test for intersection here
            IndexSpaceExpression *intersect =
              forest->intersect_index_spaces((*eit)->set_expr, expr);
            if (intersect->is_empty())
              continue;
            update_valid_views(vit->first, *eit, usage, mask, false);
            add_view_user(vit->first, usage, entry, intersect, mask);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::update_valid_views(InstanceView *view,
                                              EquivalenceSet *eq,
                                              const RegionUsage &usage,
                                              const FieldMask &user_mask,
                                              bool invalidates)
    //--------------------------------------------------------------------------
    {
      std::set<InstanceView*> &views= view_groups[view->get_manager()->tree_id];
      views.insert(view);

      if (view->is_reduction_view())
      {
        if (invalidates)
        {
#ifdef DEBUG_LEGION
          assert(IS_REDUCE(usage));
#endif
          post_reductions.insert(view, eq, user_mask);
          if (eq->set_expr->is_empty())
            consumed_reductions.insert(view, eq, user_mask);
        }
        else
        {
          if (HAS_READ(usage))
          {
            FieldMask non_dominated = user_mask;
            if (!post_reductions.dominates(view, eq, non_dominated))
              pre_reductions.insert(view, eq, non_dominated);
            else
              consumed_reductions.insert(view, eq, user_mask);
          }
        }
      }
      else
      {
        if (HAS_READ(usage))
        {
          FieldMask non_dominated = user_mask;
          bool is_dominated = post.dominates(view, eq, non_dominated);
          if (!is_dominated)
            pre.insert(view, eq, non_dominated);
        }
        if (invalidates && HAS_WRITE(usage))
        {
          for (std::set<InstanceView*>::iterator vit = views.begin();
               vit != views.end(); ++vit)
          {
            post.invalidate(*vit, eq, user_mask);
          }
        }
        post.insert(view, eq, user_mask);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::add_view_user(InstanceView *view,
                                         const RegionUsage &usage,
                                         unsigned user_index,
                                         IndexSpaceExpression *user_expr,
                                         const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      ViewUser *user = new ViewUser(usage, user_index, user_expr);
      all_users.insert(user);
      RegionTreeForest *forest = trace->runtime->forest;
      FieldMaskSet<ViewUser> &users = view_users[view];
      FieldMaskSet<ViewUser> to_delete;
      for (FieldMaskSet<ViewUser>::iterator it = users.begin();
           it != users.end(); ++it)
      {
        FieldMask overlap = user_mask & it->second;
        if (!overlap)
          continue;

        IndexSpaceExpression *expr1 = user->expr;
        IndexSpaceExpression *expr2 = it->first->expr;
        if (forest->intersect_index_spaces(expr1, expr2)->is_empty())
          continue;

        DependenceType dep =
          check_dependence_type<false>(it->first->usage, user->usage);
        if (dep == LEGION_NO_DEPENDENCE)
          continue;

        to_delete.insert(it->first, overlap);
      }

      for (FieldMaskSet<ViewUser>::iterator it = to_delete.begin();
           it != to_delete.end(); ++it)
      {
        FieldMaskSet<ViewUser>::iterator finder = users.find(it->first);
#ifdef DEBUG_LEGION
        assert(finder != users.end());
#endif
        finder.filter(it->second);
        if (!finder->second)
          users.erase(finder);
      }

      users.insert(user, user_mask);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalTemplate::record_expression_views(ViewExprs &cviews,
                                        IndexSpaceExpression *expr,
                                        const FieldMaskSet<InstanceView> &views)
    //--------------------------------------------------------------------------
    {
      for (FieldMaskSet<InstanceView>::const_iterator it = views.begin();
           it != views.end(); ++it)
        cviews[it->first].insert(expr, it->second);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_fill_views(const FieldMaskSet<FillView>&views)
    //--------------------------------------------------------------------------
    {
      for (FieldMaskSet<FillView>::const_iterator it = views.begin();
           it != views.end(); ++it)
      {
        FieldMaskSet<FillView>::iterator finder =
          post_fill_views.find(it->first);
        if (finder == post_fill_views.end())
          pre_fill_views.insert(it->first, it->second);
        else
        {
          FieldMask non_dominated = it->second - finder->second;
          if (!!non_dominated)
            pre_fill_views.insert(it->first, non_dominated);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_set_op_sync_event(ApEvent &lhs, 
                                                    Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo != NULL);
      assert(memo->is_memoizing());
#endif
      if (!lhs.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        lhs = ApEvent(rename);
      }
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      insert_instruction(new SetOpSyncEvent(*this, convert_event(lhs),
            find_trace_local_id(memo)));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_set_effects(Memoizable *memo, ApEvent &rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo != NULL);
      assert(memo->is_memoizing());
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      events.push_back(ApEvent());
      insert_instruction(new SetEffects(*this, find_trace_local_id(memo),
            find_event(rhs)));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_complete_replay(Memoizable* memo, ApEvent rhs)
    //--------------------------------------------------------------------------
    {
      const TraceLocalID lhs = find_trace_local_id(memo);
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      events.push_back(ApEvent());
      insert_instruction(new CompleteReplay(*this, lhs, find_event(rhs)));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_reservations(const TraceLocalID &tlid, 
                                const std::map<Reservation,bool> &reservations,
                                std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
      assert(cached_reservations.find(tlid) == cached_reservations.end());
#endif
      cached_reservations[tlid] = reservations;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::get_task_reservations(SingleTask *task,
                                 std::map<Reservation,bool> &reservations) const
    //--------------------------------------------------------------------------
    {
      const TraceLocalID key = task->get_trace_local_id();
      AutoLock t_lock(template_lock, 1, false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(is_replaying());
#endif
      std::map<TraceLocalID,std::map<Reservation,bool> >::const_iterator
        finder = cached_reservations.find(key);
#ifdef DEBUG_LEGION
      assert(finder != cached_reservations.end());
#endif
      reservations = finder->second;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::perform_replay(Runtime *runtime, 
                                          std::set<RtEvent> &replayed_events)
    //--------------------------------------------------------------------------
    {
      ApEvent completion;
      bool recurrent;
      {
        AutoLock t_lock(template_lock);
#ifdef DEBUG_LEGION
        assert(!pending_replays.empty());
#endif
        completion = pending_replays.front().first;
        recurrent = pending_replays.front().second;
        pending_replays.pop_front();
      }
      // Check to see if we have a pending transitive reduction result
      std::vector<std::vector<unsigned> > *transitive_reduction = 
        pending_transitive_reduction.load();
      if (transitive_reduction != NULL)
      {
        std::vector<unsigned> *inv_topo_order = pending_inv_topo_order.load();
#ifdef DEBUG_LEGION
        assert(inv_topo_order != NULL);
#endif
        finalize_transitive_reduction(*inv_topo_order, *transitive_reduction);
        delete inv_topo_order;
        pending_inv_topo_order.store(NULL);
        delete transitive_reduction;
        pending_transitive_reduction.store(NULL);
        // We also need to rerun the propagate copies analysis to
        // remove any mergers which contain only a single input
        propagate_copies(NULL/*don't need the gen out*/);
        if (runtime->dump_physical_traces)
          dump_template();
      }
      if (recurrent)
      {
        fence_completion = ApEvent::NO_AP_EVENT;
        for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
            it != frontiers.end(); ++it)
          events[it->second] = events[it->first];
      }
      else
      {
        fence_completion = completion;
        for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
            it != frontiers.end(); ++it)
          events[it->second] = completion;
      }
      events[fence_completion_id] = fence_completion;

      for (std::map<unsigned, unsigned>::iterator it =
            crossing_events.begin(); it != crossing_events.end(); ++it)
      {
        ApUserEvent ev = Runtime::create_ap_user_event(NULL);
        events[it->first] = ev;
        user_events[it->first] = ev;
      }

      const std::vector<Processor> &replay_targets = 
        trace->get_replay_targets();
      for (unsigned idx = 0; idx < replay_parallelism; ++idx)
      {
        ReplaySliceArgs args(this, idx, recurrent);
        const RtEvent done = runtime->replay_on_cpus ?
          runtime->issue_application_processor_task(args, LG_LOW_PRIORITY,
            replay_targets[idx % replay_targets.size()]) :
          runtime->issue_runtime_meta_task(args,LG_THROUGHPUT_DEFERRED_PRIORITY,
            RtEvent::NO_RT_EVENT, replay_targets[idx % replay_targets.size()]);
        replayed_events.insert(done);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::finish_replay(std::set<ApEvent> &postconditions)
    //--------------------------------------------------------------------------
    {
      for (ViewUsers::const_iterator it = view_users.begin();
            it != view_users.end(); ++it)
        for (FieldMaskSet<ViewUser>::const_iterator uit = it->second.begin();
              uit != it->second.end(); ++uit)
          postconditions.insert(events[uit->first->user]);
      if (last_fence != NULL)
        postconditions.insert(events[last_fence->lhs]);
      // Now we can remove the operations as well
      AutoLock t_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(!operations.empty());
#endif
      operations.pop_front();
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::defer_template_deletion(ApEvent &pending_deletion,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      pending_deletion = get_completion_for_deletion();
      if (!pending_deletion.exists())
        return false;
      RtEvent precondition = Runtime::protect_event(pending_deletion);
      if (transitive_reduction_done.exists() && 
          !transitive_reduction_done.has_triggered())
      {
        if (precondition.exists())
          precondition = 
            Runtime::merge_events(precondition, transitive_reduction_done);
        else
          precondition = transitive_reduction_done;
      }
      if (precondition.exists() && !precondition.has_triggered())
      {
        DeleteTemplateArgs args(this);
        applied_events.insert(trace->runtime->issue_runtime_meta_task(args,
                                            LG_LOW_PRIORITY, precondition));
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalTemplate::handle_replay_slice(const void *args)
    //--------------------------------------------------------------------------
    {
      const ReplaySliceArgs *pargs = (const ReplaySliceArgs*)args;
      pargs->tpl->execute_slice(pargs->slice_index, pargs->recurrent_replay);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalTemplate::handle_transitive_reduction(
                                             const void *args, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const TransitiveReductionArgs *targs = 
        (const TransitiveReductionArgs*)args;
      targs->tpl->transitive_reduction(true/*deferred*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalTemplate::handle_delete_template(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeleteTemplateArgs *pargs = (const DeleteTemplateArgs*)args;
      delete pargs->tpl;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::trigger_recording_done(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!recording_done.has_triggered());
#endif
      Runtime::trigger_event(recording_done);
    }

    //--------------------------------------------------------------------------
    TraceLocalID PhysicalTemplate::find_trace_local_id(Memoizable *memo)
    //--------------------------------------------------------------------------
    {
      TraceLocalID op_key = memo->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.front().find(op_key) != operations.front().end());
#endif
      return op_key;
    }

    //--------------------------------------------------------------------------
    unsigned PhysicalTemplate::find_memo_entry(Memoizable *memo)
    //--------------------------------------------------------------------------
    {
      TraceLocalID op_key = find_trace_local_id(memo);
      std::map<TraceLocalID,std::pair<unsigned,bool> >::iterator entry_finder =
        memo_entries.find(op_key);
#ifdef DEBUG_LEGION
      assert(entry_finder != memo_entries.end());
#endif
      return entry_finder->second.first;
    }

    //--------------------------------------------------------------------------
    TraceLocalID PhysicalTemplate::record_memo_entry(Memoizable *memo,
                                                     unsigned entry)
    //--------------------------------------------------------------------------
    {
      TraceLocalID key = memo->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.front().find(key) == operations.front().end());
      assert(memo_entries.find(key) == memo_entries.end());
#endif
      operations.front()[key] = memo;
      const bool is_task = memo->is_memoizable_task();
      memo_entries[key] = std::pair<unsigned,bool>(entry,is_task);
      return key;
    }

    //--------------------------------------------------------------------------
    inline unsigned PhysicalTemplate::convert_event(const ApEvent &event)
    //--------------------------------------------------------------------------
    {
      unsigned event_ = events.size();
      events.push_back(event);
#ifdef DEBUG_LEGION
      assert(event_map.find(event) == event_map.end());
#endif
      event_map[event] = event_;
      return event_;
    }

    //--------------------------------------------------------------------------
    inline unsigned PhysicalTemplate::find_event(const ApEvent &event) const
    //--------------------------------------------------------------------------
    {
      std::map<ApEvent, unsigned>::const_iterator finder= event_map.find(event);
#ifdef DEBUG_LEGION
      assert(finder != event_map.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    inline void PhysicalTemplate::insert_instruction(Instruction *inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instructions.size() + 1 == events.size());
#endif
      instructions.push_back(inst);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::find_all_last_users(ViewExprs &view_exprs,
                                               std::set<unsigned> &users)
    //--------------------------------------------------------------------------
    {
      for (ViewExprs::iterator it = view_exprs.begin(); it != view_exprs.end();
           ++it)
        for (FieldMaskSet<IndexSpaceExpression>::iterator eit =
             it->second.begin(); eit != it->second.end(); ++eit)
          find_last_users(it->first, eit->first, eit->second, users);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::find_last_users(InstanceView *view,
                                           IndexSpaceExpression *expr,
                                           const FieldMask &mask,
                                           std::set<unsigned> &users)
    //--------------------------------------------------------------------------
    {
      if (expr->is_empty()) return;

      ViewUsers::iterator finder = view_users.find(view);
      if (finder == view_users.end()) return;

      RegionTreeForest *forest = trace->runtime->forest;
      for (FieldMaskSet<ViewUser>::iterator uit = finder->second.begin(); uit !=
           finder->second.end(); ++uit)
        if (!!(uit->second & mask))
        {
          ViewUser *user = uit->first;
          IndexSpaceExpression *intersect =
            forest->intersect_index_spaces(expr, user->expr);
          if (!intersect->is_empty())
          {
            std::map<unsigned,unsigned>::const_iterator finder =
              frontiers.find(user->user);
            // See if we have recorded this frontier yet or not
            if (finder == frontiers.end())
            {
              const unsigned next_event_id = events.size();
              frontiers[user->user] = next_event_id;
              events.resize(next_event_id + 1);
              users.insert(next_event_id);
            }
            else
              users.insert(finder->second);
          }
        }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_remote_memoizable(Memoizable *memo)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      remote_memos.push_back(memo);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::release_remote_memos(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!remote_memos.empty());
#endif
      for (std::vector<Memoizable*>::const_iterator it = 
            remote_memos.begin(); it != remote_memos.end(); it++)
        delete (*it);
      remote_memos.clear();
    }

    /////////////////////////////////////////////////////////////
    // Instruction
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Instruction::Instruction(PhysicalTemplate& tpl, const TraceLocalID &o)
      : owner(o)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // GetTermEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GetTermEvent::GetTermEvent(PhysicalTemplate& tpl, unsigned l,
                               const TraceLocalID& r, bool fence)
      : Instruction(tpl, r), lhs(l)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
      assert(tpl.operations.front().find(owner) != 
              tpl.operations.front().end());
#endif
      if (fence)
        tpl.update_last_fence(this);
    }

    //--------------------------------------------------------------------------
    void GetTermEvent::execute(std::vector<ApEvent> &events,
                                 std::map<unsigned,ApUserEvent> &user_events,
                                 std::map<TraceLocalID,Memoizable*> &operations,
                                 const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      operations[owner]->replay_mapping_output();
      events[lhs] = operations[owner]->get_memo_completion();
    }

    //--------------------------------------------------------------------------
    std::string GetTermEvent::to_string(
                                 std::map<TraceLocalID,Memoizable*> &operations)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = operations[" << owner
         << "].get_completion_event()    (op kind: "
         << Operation::op_names[operations[owner]->get_memoizable_kind()]
         << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // CreateApUserEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CreateApUserEvent::CreateApUserEvent(PhysicalTemplate& tpl, unsigned l,
                                         const TraceLocalID &o)
      : Instruction(tpl, o), lhs(l)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
      assert(tpl.user_events.find(lhs) != tpl.user_events.end());
#endif
    }

    //--------------------------------------------------------------------------
    void CreateApUserEvent::execute(std::vector<ApEvent> &events,
                                 std::map<unsigned,ApUserEvent> &user_events,
                                 std::map<TraceLocalID,Memoizable*> &operations,
                                 const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(user_events.find(lhs) != user_events.end());
#endif
      ApUserEvent ev = Runtime::create_ap_user_event(NULL);
      events[lhs] = ev;
      user_events[lhs] = ev;
    }

    //--------------------------------------------------------------------------
    std::string CreateApUserEvent::to_string(
                                 std::map<TraceLocalID,Memoizable*> &operations)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = Runtime::create_ap_user_event()    "
         << "(owner: " << owner << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // TriggerEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TriggerEvent::TriggerEvent(PhysicalTemplate& tpl, unsigned l, unsigned r,
                               const TraceLocalID &o)
      : Instruction(tpl, o), lhs(l), rhs(r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
      assert(rhs < tpl.events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void TriggerEvent::execute(std::vector<ApEvent> &events,
                               std::map<unsigned,ApUserEvent> &user_events,
                               std::map<TraceLocalID,Memoizable*> &operations,
                               const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(events[lhs].exists());
      assert(user_events[lhs].exists());
      assert(events[lhs].id == user_events[lhs].id);
#endif
      Runtime::trigger_event(NULL, user_events[lhs], events[rhs]);
    }

    //--------------------------------------------------------------------------
    std::string TriggerEvent::to_string(
                                 std::map<TraceLocalID,Memoizable*> &operations)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "Runtime::trigger_event(events[" << lhs
         << "], events[" << rhs << "])    (owner: " << owner << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // MergeEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MergeEvent::MergeEvent(PhysicalTemplate& tpl, unsigned l,
                           const std::set<unsigned>& r, const TraceLocalID &o)
      : Instruction(tpl, o), lhs(l), rhs(r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
      assert(rhs.size() > 0);
      for (std::set<unsigned>::iterator it = rhs.begin(); it != rhs.end();
           ++it)
        assert(*it < tpl.events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void MergeEvent::execute(std::vector<ApEvent> &events,
                             std::map<unsigned,ApUserEvent> &user_events,
                             std::map<TraceLocalID,Memoizable*> &operations,
                             const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      std::vector<ApEvent> to_merge;
      to_merge.reserve(rhs.size());
      for (std::set<unsigned>::const_iterator it =
            rhs.begin(); it != rhs.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(*it < events.size());
#endif
        to_merge.push_back(events[*it]);
      }
      ApEvent result = Runtime::merge_events(NULL, to_merge);
      events[lhs] = result;
    }

    //--------------------------------------------------------------------------
    std::string MergeEvent::to_string(
                                 std::map<TraceLocalID,Memoizable*> &operations)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = Runtime::merge_events(";
      unsigned count = 0;
      for (std::set<unsigned>::iterator it = rhs.begin(); it != rhs.end();
           ++it)
      {
        if (count++ != 0) ss << ",";
        ss << "events[" << *it << "]";
      }
      ss << ")    (owner: " << owner << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // AssignFenceCompletion
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AssignFenceCompletion::AssignFenceCompletion(
                       PhysicalTemplate& t, unsigned l, const TraceLocalID &o)
      : Instruction(t, o), tpl(t), lhs(l)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void AssignFenceCompletion::execute(std::vector<ApEvent> &events,
                                 std::map<unsigned,ApUserEvent> &user_events,
                                 std::map<TraceLocalID,Memoizable*> &operations,
                                 const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      events[lhs] = tpl.get_fence_completion();
    }

    //--------------------------------------------------------------------------
    std::string AssignFenceCompletion::to_string(
                                 std::map<TraceLocalID,Memoizable*> &operations)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = fence_completion";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // IssueCopy
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IssueCopy::IssueCopy(PhysicalTemplate& tpl,
                         unsigned l, IndexSpaceExpression *e,
                         const TraceLocalID& key,
                         const std::vector<CopySrcDstField>& s,
                         const std::vector<CopySrcDstField>& d,
                         const std::vector<Reservation>& r,
#ifdef LEGION_SPY
                         RegionTreeID src_tid, RegionTreeID dst_tid,
#endif
                         unsigned pi)
      : Instruction(tpl, key), lhs(l), expr(e), src_fields(s), dst_fields(d), 
        reservations(r),
#ifdef LEGION_SPY
        src_tree_id(src_tid), dst_tree_id(dst_tid),
#endif
        precondition_idx(pi)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
      assert(tpl.operations.front().find(owner) != 
              tpl.operations.front().end());
      assert(src_fields.size() > 0);
      assert(dst_fields.size() > 0);
      assert(precondition_idx < tpl.events.size());
      assert(expr != NULL);
#endif
      expr->add_base_expression_reference(TRACE_REF);
    }

    //--------------------------------------------------------------------------
    IssueCopy::~IssueCopy(void)
    //--------------------------------------------------------------------------
    {
      if (expr->remove_base_expression_reference(TRACE_REF))
        delete expr;
    }

    //--------------------------------------------------------------------------
    void IssueCopy::execute(std::vector<ApEvent> &events,
                            std::map<unsigned,ApUserEvent> &user_events,
                            std::map<TraceLocalID,Memoizable*> &operations,
                            const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      Memoizable *memo = operations[owner];
      ApEvent precondition = events[precondition_idx];
      const PhysicalTraceInfo trace_info(memo->get_operation(), -1U, false);
      events[lhs] = expr->issue_copy(trace_info, dst_fields,
                                     src_fields, reservations,
#ifdef LEGION_SPY
                                     src_tree_id, dst_tree_id,
#endif
                                     precondition, PredEvent::NO_PRED_EVENT);
    }

    //--------------------------------------------------------------------------
    std::string IssueCopy::to_string(
                                 std::map<TraceLocalID,Memoizable*> &operations)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = copy(operations[" << owner << "], "
         << "Index expr: " << expr->expr_id << ", {";
      for (unsigned idx = 0; idx < src_fields.size(); ++idx)
      {
        ss << "(" << std::hex << src_fields[idx].inst.id
           << "," << std::dec << src_fields[idx].subfield_offset
           << "," << src_fields[idx].size
           << "," << src_fields[idx].field_id
           << "," << src_fields[idx].serdez_id << ")";
        if (idx != src_fields.size() - 1) ss << ",";
      }
      ss << "}, {";
      for (unsigned idx = 0; idx < dst_fields.size(); ++idx)
      {
        ss << "(" << std::hex << dst_fields[idx].inst.id
           << "," << std::dec << dst_fields[idx].subfield_offset
           << "," << dst_fields[idx].size
           << "," << dst_fields[idx].field_id
           << "," << dst_fields[idx].serdez_id << ")";
        if (idx != dst_fields.size() - 1) ss << ",";
      }
      ss << "}, events[" << precondition_idx << "]";

      ss << ")";

      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // IssueAcross
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IssueAcross::IssueAcross(PhysicalTemplate& tpl, unsigned l, unsigned copy,
                             unsigned collective, unsigned src_indirect,
                             unsigned dst_indirect, const TraceLocalID& key,
                             CopyAcrossExecutor *exec)
      : Instruction(tpl, key), lhs(l), copy_precondition(copy), 
        collective_precondition(collective), 
        src_indirect_precondition(src_indirect),
        dst_indirect_precondition(dst_indirect), executor(exec)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
      assert(tpl.operations.front().find(owner) != 
              tpl.operations.front().end());
#endif
      executor->add_reference();
    }

    //--------------------------------------------------------------------------
    IssueAcross::~IssueAcross(void)
    //--------------------------------------------------------------------------
    {
      if (executor->remove_reference())
        delete executor;
    }

    //--------------------------------------------------------------------------
    void IssueAcross::execute(std::vector<ApEvent> &events,
                              std::map<unsigned,ApUserEvent> &user_events,
                              std::map<TraceLocalID,Memoizable*> &operations,
                              const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      Memoizable *memo = operations[owner];
      Operation *op = memo->get_operation();
      ApEvent copy_pre = events[copy_precondition];
      ApEvent src_indirect_pre = events[src_indirect_precondition];
      ApEvent dst_indirect_pre = events[dst_indirect_precondition];
      const PhysicalTraceInfo trace_info(op, -1U, false);
      events[lhs] = executor->execute(op, PredEvent::NO_PRED_EVENT,
                                      copy_pre, src_indirect_pre,
                                      dst_indirect_pre, trace_info,
                                      recurrent_replay);
    }

    //--------------------------------------------------------------------------
    std::string IssueAcross::to_string(
                                 std::map<TraceLocalID,Memoizable*> &operations)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = indirect(operations[" << owner << "], "
         << "Copy Across Executor: " << executor << ", {";
      ss << ", TODO: indirections";
      ss << "}, events[" << copy_precondition << "]";
      ss << ", events[" << collective_precondition << "]";
      ss << ", events[" << src_indirect_precondition << "]";
      ss << ", events[" << dst_indirect_precondition << "]";
      ss << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // IssueFill
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IssueFill::IssueFill(PhysicalTemplate& tpl, unsigned l, 
                         IndexSpaceExpression *e, const TraceLocalID &key,
                         const std::vector<CopySrcDstField> &f,
                         const void *value, size_t size, 
#ifdef LEGION_SPY
                         FieldSpace h, RegionTreeID tid,
#endif
                         unsigned pi)
      : Instruction(tpl, key), lhs(l), expr(e), fields(f), fill_size(size),
#ifdef LEGION_SPY
        handle(h), tree_id(tid),
#endif
        precondition_idx(pi)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
      assert(tpl.operations.front().find(owner) != 
              tpl.operations.front().end());
      assert(fields.size() > 0);
      assert(precondition_idx < tpl.events.size());
#endif
      expr->add_base_expression_reference(TRACE_REF);
      fill_value = malloc(fill_size);
      memcpy(fill_value, value, fill_size);
    }

    //--------------------------------------------------------------------------
    IssueFill::~IssueFill(void)
    //--------------------------------------------------------------------------
    {
      if (expr->remove_base_expression_reference(TRACE_REF))
        delete expr;
      free(fill_value);
    }

    //--------------------------------------------------------------------------
    void IssueFill::execute(std::vector<ApEvent> &events,
                            std::map<unsigned,ApUserEvent> &user_events,
                            std::map<TraceLocalID,Memoizable*> &operations,
                            const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      Memoizable *memo = operations[owner];
      ApEvent precondition = events[precondition_idx];
      const PhysicalTraceInfo trace_info(memo->get_operation(), -1U, false);
      events[lhs] = expr->issue_fill(trace_info, fields, 
                                     fill_value, fill_size,
#ifdef LEGION_SPY
                                     trace_info.op->get_unique_op_id(),
                                     handle, tree_id,
#endif
                                     precondition, PredEvent::NO_PRED_EVENT);
    }

    //--------------------------------------------------------------------------
    std::string IssueFill::to_string(
                                 std::map<TraceLocalID,Memoizable*> &operations)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = fill(Index expr: " << expr->expr_id
         << ", {";
      for (unsigned idx = 0; idx < fields.size(); ++idx)
      {
        ss << "(" << std::hex << fields[idx].inst.id
           << "," << std::dec << fields[idx].subfield_offset
           << "," << fields[idx].size
           << "," << fields[idx].field_id
           << "," << fields[idx].serdez_id << ")";
        if (idx != fields.size() - 1) ss << ",";
      }
      ss << "}, events[" << precondition_idx << "])    (owner: "
         << owner << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // SetOpSyncEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SetOpSyncEvent::SetOpSyncEvent(PhysicalTemplate& tpl, unsigned l,
                                       const TraceLocalID& r)
      : Instruction(tpl, r), lhs(l)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
      assert(tpl.operations.front().find(owner) != 
              tpl.operations.front().end());
#endif
    }

    //--------------------------------------------------------------------------
    void SetOpSyncEvent::execute(std::vector<ApEvent> &events,
                                 std::map<unsigned,ApUserEvent> &user_events,
                                 std::map<TraceLocalID,Memoizable*> &operations,
                                 const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      Memoizable *memoizable = operations[owner];
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      ApEvent sync_condition = memoizable->compute_sync_precondition(NULL);
      events[lhs] = sync_condition;
    }

    //--------------------------------------------------------------------------
    std::string SetOpSyncEvent::to_string(
                                 std::map<TraceLocalID,Memoizable*> &operations)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = operations[" << owner
         << "].compute_sync_precondition()    (op kind: "
         << Operation::op_names[operations[owner]->get_memoizable_kind()] 
         << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // SetEffects
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SetEffects::SetEffects(PhysicalTemplate& tpl, const TraceLocalID& l,
                           unsigned r)
      : Instruction(tpl, l), rhs(r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs < tpl.events.size());
      assert(tpl.operations.front().find(owner) != 
              tpl.operations.front().end());
#endif
    }

    //--------------------------------------------------------------------------
    void SetEffects::execute(std::vector<ApEvent> &events,
                             std::map<unsigned,ApUserEvent> &user_events,
                             std::map<TraceLocalID,Memoizable*> &operations,
                             const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      Memoizable *memoizable = operations[owner];
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      memoizable->set_effects_postcondition(events[rhs]);
    }

    //--------------------------------------------------------------------------
    std::string SetEffects::to_string(
                                 std::map<TraceLocalID,Memoizable*> &operations)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "operations[" << owner << "].set_effects_postcondition(events["
         << rhs << "])    (op kind: "
         << Operation::op_names[operations[owner]->get_memoizable_kind()]
         << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // CompleteReplay
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompleteReplay::CompleteReplay(PhysicalTemplate& tpl,
                                              const TraceLocalID& l, unsigned r)
      : Instruction(tpl, l), rhs(r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tpl.operations.front().find(owner) != 
              tpl.operations.front().end());
      assert(rhs < tpl.events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void CompleteReplay::execute(std::vector<ApEvent> &events,
                                 std::map<unsigned,ApUserEvent> &user_events,
                                 std::map<TraceLocalID,Memoizable*> &operations,
                                 const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      Memoizable *memoizable = operations[owner];
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      memoizable->complete_replay(events[rhs]);
    }

    //--------------------------------------------------------------------------
    std::string CompleteReplay::to_string(
                                 std::map<TraceLocalID,Memoizable*> &operations)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "operations[" << owner
         << "].complete_replay(events[" << rhs << "])    (op kind: "
         << Operation::op_names[operations[owner]->get_memoizable_kind()] 
         << ")";
      return ss.str();
    }

  }; // namespace Internal 
}; // namespace Legion

