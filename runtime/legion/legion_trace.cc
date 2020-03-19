/* Copyright 2020 Stanford University, NVIDIA Corporation
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
    LegionTrace::LegionTrace(InnerContext *c, bool logical_only)
      : ctx(c), state(LOGICAL_ONLY), last_memoized(0),
        blocking_call_observed(false)
    //--------------------------------------------------------------------------
    {
      physical_trace = logical_only ? NULL
                                    : new PhysicalTrace(c->owner_task->runtime,
                                                        this);
    }

    //--------------------------------------------------------------------------
    LegionTrace::~LegionTrace(void)
    //--------------------------------------------------------------------------
    {
      if (physical_trace != NULL)
        delete physical_trace;
    }

    //--------------------------------------------------------------------------
    void LegionTrace::register_physical_only(Operation *op, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      if (has_blocking_call())
        REPORT_LEGION_ERROR(ERROR_INVALID_PHYSICAL_TRACING,
            "Physical tracing violation! The trace has a blocking API call "
            "that was unseen when it was recorded. Please make sure that "
            "the trace does not change its behavior.");
      std::pair<Operation*,GenerationID> key(op,gen);
      const unsigned index = operations.size();
      op->set_trace_local_id(index);
      op->add_mapping_reference(gen);
      operations.push_back(key);
#ifdef LEGION_SPY
      current_uids[key] = op->get_unique_op_id();
#endif
    }

    //--------------------------------------------------------------------------
    void LegionTrace::replay_aliased_children(
                             std::vector<RegionTreePath> &privilege_paths) const
    //--------------------------------------------------------------------------
    {
      unsigned index = operations.size() - 1;
      std::map<unsigned,LegionVector<AliasChildren>::aligned>::const_iterator
        finder = aliased_children.find(index);
      if (finder == aliased_children.end())
        return;
      for (LegionVector<AliasChildren>::aligned::const_iterator it = 
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
        for (unsigned idx = 0; idx < operations.size(); ++idx)
          operations[idx].first->remove_mapping_reference(
              operations[idx].second);
        operations.clear();
#ifdef LEGION_SPY
        current_uids.clear();
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
              op->get_unique_op_id(), 0, TRUE_DEPENDENCE);
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
    void LegionTrace::record_blocking_call(void)
    //--------------------------------------------------------------------------
    {
      blocking_call_observed = true;
      if (physical_trace == NULL)
        return;
      PhysicalTemplate *tpl = physical_trace->get_current_template();
      if (tpl != NULL)
        tpl->trigger_recording_done();
      if (is_replaying())
        REPORT_LEGION_ERROR(ERROR_INVALID_PHYSICAL_TRACING,
            "Physical tracing violation! The trace has a blocking API call "
            "that was unseen when it was recorded. Please make sure that "
            "the trace does not change its behavior.");
    }

    //--------------------------------------------------------------------------
    void LegionTrace::invalidate_trace_cache(Operation *invalidator)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate *invalidated_template = NULL;
      if (physical_trace != NULL)
      {
        invalidated_template = physical_trace->get_current_template();
        physical_trace->clear_cached_template();
      }
      if (invalidated_template != NULL)
        invalidated_template->issue_summary_operations(ctx, invalidator);
    }

    /////////////////////////////////////////////////////////////
    // StaticTrace
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    StaticTrace::StaticTrace(InnerContext *c,
                             const std::set<RegionTreeID> *trees)
      : LegionTrace(c, true)
    //--------------------------------------------------------------------------
    {
      if (trees != NULL)
        application_trees.insert(trees->begin(), trees->end());
    }
    
    //--------------------------------------------------------------------------
    StaticTrace::StaticTrace(const StaticTrace &rhs)
      : LegionTrace(NULL, true)
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
    bool StaticTrace::is_fixed(void) const
    //--------------------------------------------------------------------------
    {
      // Static traces are always fixed
      return true;
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
    void StaticTrace::record_static_dependences(Operation *op,
                               const std::vector<StaticDependence> *dependences)
    //--------------------------------------------------------------------------
    {
      // Internal operations get to skip this
      if (op->is_internal_op())
        return;
      // All other operations have to add something to the list
      if (dependences == NULL)
        static_dependences.resize(static_dependences.size() + 1);
      else // Add it to the list of static dependences
        static_dependences.push_back(*dependences);
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
        const LegionVector<DependenceRecord>::aligned &deps = 
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
        for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
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
                (it->next_idx == -1) ? 0 : it->next_idx, TRUE_DEPENDENCE);
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
        const LegionVector<DependenceRecord>::aligned &deps = 
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
        for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
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
                (it->next_idx == -1) ? 0 : it->next_idx, TRUE_DEPENDENCE);
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
    void StaticTrace::record_aliased_children(unsigned req_index,unsigned depth,
                                              const FieldMask &aliase_mask)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

#ifdef LEGION_SPY
    //--------------------------------------------------------------------------
    void StaticTrace::perform_logging(
                               UniqueID prev_fence_uid, UniqueID curr_fence_uid)
    //--------------------------------------------------------------------------
    {
      // TODO: Implement this if someone wants to memoize static traces
    }
#endif

    //--------------------------------------------------------------------------
    const LegionVector<LegionTrace::DependenceRecord>::aligned& 
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
        LegionVector<DependenceRecord>::aligned &translation = 
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
    DynamicTrace::DynamicTrace(TraceID t, InnerContext *c, bool logical_only)
      : LegionTrace(c, logical_only), tid(t), fixed(false), tracing(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DynamicTrace::DynamicTrace(const DynamicTrace &rhs)
      : LegionTrace(NULL, true), tid(0)
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
    void DynamicTrace::fix_trace(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fixed);
#endif
      fixed = true;
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::end_trace_capture(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tracing);
#endif
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
    void DynamicTrace::record_static_dependences(Operation *op,
                               const std::vector<StaticDependence> *dependences)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
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
      if ((is_recording() || is_replaying()) &&
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
          dependences.push_back(LegionVector<DependenceRecord>::aligned());
          // Record meta-data about the trace for verifying that
          // it is being replayed correctly
          op_info.push_back(OperationInfo(op));
        }
        else // Otherwise, track internal operations separately
        {
          std::pair<InternalOp*,GenerationID> 
            local_key(static_cast<InternalOp*>(op),gen);
          internal_dependences[local_key] = 
            LegionVector<DependenceRecord>::aligned();
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
          const LegionVector<DependenceRecord>::aligned &deps = 
                                                          dependences[index];
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
          for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
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
                  (it->next_idx == -1) ? 0 : it->next_idx, TRUE_DEPENDENCE);
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
          const LegionVector<DependenceRecord>::aligned &deps = 
                                                        dependences[index-1];
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
          for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
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
                  (it->next_idx == -1) ? 0 : it->next_idx, TRUE_DEPENDENCE);
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
                LegionVector<DependenceRecord>::aligned>::const_iterator
          internal_finder = internal_dependences.find(local_key);
        if (internal_finder != internal_dependences.end())
        {
          const LegionVector<DependenceRecord>::aligned &internal_deps = 
                                                        internal_finder->second;
          for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
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
                 LegionVector<DependenceRecord>::aligned>::const_iterator
          internal_finder = internal_dependences.find(local_key);
        if (internal_finder != internal_dependences.end())
        {
          // It is one of ours, so two cases
          if (!source->is_internal_op())
          {
            // Iterate over the internal operation dependences and 
            // translate them to our dependences
            for (LegionVector<DependenceRecord>::aligned::const_iterator
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
            for (LegionVector<DependenceRecord>::aligned::const_iterator
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
        UniqueID uid = get_current_uid_by_index(idx);
        const LegionVector<DependenceRecord>::aligned &deps = dependences[idx];
        for (LegionVector<DependenceRecord>::aligned::const_iterator it =
             deps.begin(); it != deps.end(); it++)
        {
          if ((it->prev_idx == -1) || (it->next_idx == -1))
          {
            LegionSpy::log_mapping_dependence(
                context_uid,
                operations[it->operation_idx].first->get_unique_op_id(),
                (it->prev_idx == -1) ? 0 : it->prev_idx,
                uid,
                (it->next_idx == -1) ? 0 : it->next_idx, TRUE_DEPENDENCE);
          }
          else
          {
            LegionSpy::log_mapping_dependence(
                context_uid,
                operations[it->operation_idx].first->get_unique_op_id(),
                it->prev_idx, uid, it->next_idx, it->dtype);
          }
        }
        LegionSpy::log_mapping_dependence(
            context_uid, prev_fence_uid, 0, uid, 0, TRUE_DEPENDENCE);
        LegionSpy::log_mapping_dependence(
            context_uid, uid, 0, curr_fence_uid, 0, TRUE_DEPENDENCE);
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
      LegionVector<DependenceRecord>::aligned &deps = dependences.back();
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
      LegionVector<DependenceRecord>::aligned &deps = internal_dependences[key];
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
    void TraceCaptureOp::initialize_capture(InnerContext *ctx, bool has_block)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, EXECUTION_FENCE, false/*need future*/);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
      assert(trace->is_dynamic_trace());
#endif
      dynamic_trace = trace->as_dynamic_trace();
      local_trace = dynamic_trace;
      // Now mark our trace as NULL to avoid registering this operation
      trace = NULL;
      tracing = false;
      current_template = NULL;
      has_blocking_call = has_block;
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
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
      assert(local_trace == dynamic_trace);
#endif
      // Indicate that we are done capturing this trace
      dynamic_trace->end_trace_capture();
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
      }
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Now finish capturing the physical trace
      if (local_trace->is_recording())
      {
        PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(current_template != NULL);
        assert(physical_trace != NULL);
#endif
        RtEvent pending_deletion =
          physical_trace->fix_trace(current_template, this, has_blocking_call);
        if (pending_deletion.exists())
          execution_precondition = Runtime::merge_events(NULL,
              execution_precondition, ApEvent(pending_deletion));
        local_trace->initialize_tracing_state();
      }
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
    void TraceCompleteOp::initialize_complete(InnerContext *ctx, bool has_block)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, EXECUTION_FENCE, false/*need future*/);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
      // Now mark our trace as NULL to avoid registering this operation
      trace = NULL;
      current_template = NULL;
      template_completion = ApEvent::NO_AP_EVENT;
      replayed = false;
      has_blocking_call = has_block;
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
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
      local_trace->end_trace_execution(this);
      parent_ctx->record_previous_trace(local_trace);

      if (local_trace->is_replaying())
      {
        PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(physical_trace != NULL);
#endif
        PhysicalTemplate *to_replay = physical_trace->get_current_template();
#ifdef DEBUG_LEGION
        assert(to_replay != NULL);
#endif
#ifdef LEGION_SPY
        local_trace->perform_logging(to_replay->get_fence_uid(), unique_op_id);
#endif
        to_replay->execute_all();
        template_completion = to_replay->get_completion();
        Runtime::trigger_event(completion_event, template_completion);
        parent_ctx->update_current_fence(this, true, true);
        physical_trace->record_previous_template_completion(
            template_completion);
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
        physical_trace->record_previous_template_completion(
            get_completion_event());
        current_template = physical_trace->get_current_template();
        physical_trace->clear_cached_template();

      }

      // If this is a static trace, then we remove our reference when we're done
      if (local_trace->is_static_trace())
      {
        StaticTrace *static_trace = static_cast<StaticTrace*>(local_trace);
        if (static_trace->remove_reference())
          delete static_trace;
      }
      FenceOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Now finish capturing the physical trace
      if (local_trace->is_recording())
      {
#ifdef DEBUG_LEGION
        assert(current_template != NULL);
        assert(local_trace->get_physical_trace() != NULL);
#endif
        RtEvent pending_deletion =
          local_trace->get_physical_trace()->fix_trace(current_template, this,
              has_blocking_call);
        if (pending_deletion.exists())
          execution_precondition = Runtime::merge_events(NULL,
              execution_precondition, ApEvent(pending_deletion));
        local_trace->initialize_tracing_state();
      }
      else if (replayed)
      {
        complete_mapping();
        need_completion_trigger = false;
        if (!template_completion.has_triggered())
        {
          RtEvent wait_on = Runtime::protect_event(template_completion);
          complete_execution(wait_on);
        }
        else
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
    void TraceReplayOp::initialize_replay(InnerContext *ctx, LegionTrace *trace)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, EXECUTION_FENCE, false/*need future*/);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
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
      if (physical_trace->get_current_template() == NULL || is_recording)
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
        execution_precondition =
          parent_ctx->perform_fence_analysis(this, true, true);
        physical_trace->set_current_execution_fence_event(
            get_completion_event());
        fence_registered = true;
      }

      if (physical_trace->get_current_template() != NULL)
      {
        if (!fence_registered)
          execution_precondition =
            parent_ctx->get_current_execution_fence_event();
        ApEvent fence_completion =
          recurrent ? physical_trace->get_previous_template_completion()
                    : get_completion_event();
        physical_trace->initialize_template(fence_completion, recurrent);
        local_trace->set_state_replay();
#ifdef LEGION_SPY
        physical_trace->get_current_template()->set_fence_uid(unique_op_id);
#endif
      }
      else if (!fence_registered)
      {
        execution_precondition =
          parent_ctx->perform_fence_analysis(this, true, true);
        physical_trace->set_current_execution_fence_event(
            get_completion_event());
      }

      // Now update the parent context with this fence before we can complete
      // the dependence analysis and possibly be deactivated
      parent_ctx->update_current_fence(this, true, true);
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::pack_remote_operation(Serializer &rez, 
                                              AddressSpaceID target) const
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
    void TraceBeginOp::initialize_begin(InnerContext *ctx, LegionTrace *trace)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, MAPPING_FENCE, false/*need future*/);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
      trace = NULL;
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
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
                                            Operation *invalidator)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, false/*track*/);
      fence_kind = MAPPING_FENCE;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_fence_operation(parent_ctx->get_unique_id(),
                                       unique_op_id);
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
      activate_operation();
      current_template = NULL;
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
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
                                               AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTrace
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTrace::PhysicalTrace(Runtime *rt, LegionTrace *lt)
      : runtime(rt), logical_trace(lt), current_template(NULL),
        nonreplayable_count(0), new_template_count(0),
        previous_template_completion(ApEvent::NO_AP_EVENT),
        execution_fence_event(ApEvent::NO_AP_EVENT)
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
    PhysicalTrace::PhysicalTrace(const PhysicalTrace &rhs)
      : runtime(NULL), logical_trace(NULL), current_template(NULL),
        nonreplayable_count(0), new_template_count(0),
        previous_template_completion(ApEvent::NO_AP_EVENT),
        execution_fence_event(ApEvent::NO_AP_EVENT)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalTrace::~PhysicalTrace()
    //--------------------------------------------------------------------------
    {
      for (LegionVector<PhysicalTemplate*>::aligned::iterator it =
           templates.begin(); it != templates.end(); ++it)
        delete (*it);
      templates.clear();
    }

    //--------------------------------------------------------------------------
    PhysicalTrace& PhysicalTrace::operator=(const PhysicalTrace &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalTrace::fix_trace(
                   PhysicalTemplate *tpl, Operation *op, bool has_blocking_call)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tpl->is_recording());
#endif
      tpl->finalize(op, has_blocking_call);
      RtEvent pending_deletion = RtEvent::NO_RT_EVENT;
      if (!tpl->is_replayable())
      {
        pending_deletion = tpl->defer_template_deletion();
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
      for (LegionVector<PhysicalTemplate*>::aligned::reverse_iterator it =
           templates.rbegin(); it != templates.rend(); ++it)
      {
        if ((*it)->check_preconditions(op, applied_events))
        {
#ifdef DEBUG_LEGION
          assert((*it)->is_replayable());
#endif
          // Reset the nonreplayable count when a replayable template satisfies
          // the precondition
          nonreplayable_count = 0;
          // Also reset the new template count as we found a replay
          new_template_count = 0;
          current_template = *it;
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
    void PhysicalTrace::initialize_template(
                                       ApEvent fence_completion, bool recurrent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_template != NULL);
#endif
      current_template->initialize(runtime, fence_completion, recurrent);
    }

    /////////////////////////////////////////////////////////////
    // TraceViewSet
    /////////////////////////////////////////////////////////////

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

      for (FieldMaskSet<EquivalenceSet>::const_iterator it =
           finder->second.begin(); it != finder->second.end(); ++it)
      {
        FieldMask overlap = non_dominated & it->second;
        if (!overlap)
          continue;

        IndexSpaceExpression *expr1 = eq->set_expr;
        IndexSpaceExpression *expr2 = it->first->set_expr;
        if (expr1 == expr2)
        {
          non_dominated -= overlap;
        }
        else if (expr1->get_volume() <= expr2->get_volume())
        {
          IndexSpaceExpression *diff =
            forest->subtract_index_spaces(expr1, expr2);
          if (diff->is_empty())
            non_dominated -= overlap;
        }
        if (!non_dominated)
          return true;
      }
#ifdef DEBUG_LEGION
      assert(!!non_dominated);
#endif
      return false;
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::subsumed_by(const TraceViewSet &set) const
    //--------------------------------------------------------------------------
    {
      for (ViewSet::const_iterator it = conditions.begin();
           it != conditions.end(); ++it)
        for (FieldMaskSet<EquivalenceSet>::const_iterator eit =
             it->second.begin(); eit != it->second.end(); ++eit)
      {
        FieldMask mask = eit->second;
        if (!set.dominates(it->first, eit->first, mask))
          return false;
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
          forest->runtime->retrieve_semantic_information(lr, NAME_SEMANTIC_TAG,
              name, name_size, true, true);
          log_tracing.info() << "  "
                    <<(view->is_reduction_view() ? "Reduction" : "Materialized")
                    << " view: " << view << ", Inst: " << std::hex
                    << view->get_manager()->get_instance().id << std::dec
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
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::make_ready(bool postcondition)
    //--------------------------------------------------------------------------
    {
      if (cached)
        return;
      cached = true;

      typedef std::pair<RegionTreeID,EquivalenceSet*> Key;
      LegionMap<Key,FieldMaskSet<InstanceView> >::aligned views_by_regions;

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
      }

      // Filter out views that overlap with some restricted views
      if (postcondition)
      {
        for (LegionMap<Key,FieldMaskSet<InstanceView> >::aligned::iterator it =
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
      for (LegionMap<Key,FieldMaskSet<InstanceView> >::aligned::iterator it =
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
        has_virtual_mapping(false),
        recording_done(RtUserEvent::NO_RT_USER_EVENT),
        pre(t->runtime->forest), post(t->runtime->forest),
        pre_reductions(t->runtime->forest), post_reductions(t->runtime->forest),
        consumed_reductions(t->runtime->forest)
    //--------------------------------------------------------------------------
    {
      events.push_back(fence_event);
      event_map[fence_event] = fence_completion_id;
      instructions.push_back(
         new AssignFenceCompletion(*this, fence_completion_id, TraceLocalID()));
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
                ref.remove_valid_reference(MAPPING_ACQUIRE_REF);
            }
            pit->clear();
          }
        }
        cached_mappings.clear();
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::initialize(
                           Runtime *runtime, ApEvent completion, bool recurrent)
    //--------------------------------------------------------------------------
    {
      fence_completion = completion;
      if (recurrent)
        for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
            it != frontiers.end(); ++it)
          events[it->second] = events[it->first];
      else
        for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
            it != frontiers.end(); ++it)
          events[it->second] = completion;

      events[fence_completion_id] = fence_completion;

      for (std::map<unsigned, unsigned>::iterator it = crossing_events.begin();
           it != crossing_events.end(); ++it)
      {
        ApUserEvent ev = Runtime::create_ap_user_event();
        events[it->second] = ev;
        user_events[it->second] = ev;
      }

      replay_ready = Runtime::create_rt_user_event();
      std::set<RtEvent> replay_done_events;
      const std::vector<Processor> &replay_targets =
        trace->get_replay_targets();
      for (unsigned idx = 0; idx < replay_parallelism; ++idx)
      {
        ReplaySliceArgs args(this, idx);
        RtEvent done =
          runtime->issue_runtime_meta_task(args,
            runtime->replay_on_cpus ? LG_LOW_PRIORITY
                                    : LG_THROUGHPUT_WORK_PRIORITY,
            replay_ready, replay_targets[idx % replay_targets.size()]);
        replay_done_events.insert(done);
      }
      replay_done = Runtime::merge_events(replay_done_events);

#ifdef DEBUG_LEGION
      for (std::map<TraceLocalID, Memoizable*>::iterator it =
           operations.begin(); it != operations.end(); ++it)
        it->second = NULL;
#endif
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalTemplate::get_completion(void) const
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> to_merge;
      for (ViewUsers::const_iterator it = view_users.begin();
           it != view_users.end(); ++it)
        for (FieldMaskSet<ViewUser>::const_iterator uit = it->second.begin();
             uit != it->second.end(); ++uit)
          to_merge.insert(events[uit->first->user]);
      return Runtime::merge_events(NULL, to_merge);
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

      if (!post_reductions.subsumed_by(consumed_reductions))
        return Replayable(false, "escaping reduction views");

      if (pre.has_refinements() || post.has_refinements())
        return Replayable(false, "found refined equivalence sets");

      if (!pre.subsumed_by(post))
        return Replayable(false, "precondition not subsumed by postcondition");

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
      std::map<TraceLocalID, Memoizable*>::iterator op_finder =
        operations.find(memoizable->get_trace_local_id());
#ifdef DEBUG_LEGION
      assert(op_finder != operations.end());
      assert(op_finder->second == NULL);
#endif
      op_finder->second = memoizable;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::execute_all(void)
    //--------------------------------------------------------------------------
    {
      Runtime::trigger_event(replay_ready);
      replay_done.wait();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::execute_slice(unsigned slice_idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(slice_idx < slices.size());
#endif
      ApUserEvent fence = Runtime::create_ap_user_event();
      const std::vector<TraceLocalID> &tasks = slice_tasks[slice_idx];
      for (unsigned idx = 0; idx < tasks.size(); ++idx)
        operations[tasks[idx]]
          ->get_operation()->set_execution_fence_event(fence);
      std::vector<Instruction*> &instructions = slices[slice_idx];
      for (std::vector<Instruction*>::const_iterator it = instructions.begin();
           it != instructions.end(); ++it)
        (*it)->execute();
      Runtime::trigger_event(fence);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::issue_summary_operations(
                                  InnerContext* context, Operation *invalidator)
    //--------------------------------------------------------------------------
    {
      TraceSummaryOp *op = trace->runtime->get_available_summary_op();
      op->initialize_summary(context, this, invalidator);
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
          optimize();
          dump_template();
        }
        return;
      }
      generate_conditions();
      optimize();
      if (trace->runtime->dump_physical_traces) dump_template();
      size_t num_events = events.size();
      events.clear();
      events.resize(num_events);
      event_map.clear();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::generate_conditions(void)
    //--------------------------------------------------------------------------
    {
      pre.make_ready(false /*postcondition*/);
      post.make_ready(true /*postcondition*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::optimize(void)
    //--------------------------------------------------------------------------
    {
      std::vector<unsigned> gen;
      if (!(trace->runtime->no_trace_optimization ||
            trace->runtime->no_fence_elision))
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
        transitive_reduction();
        propagate_copies(gen);
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
              num_merges += generator_kind != MERGE_EVENT;
              break;
            }
          case ISSUE_FILL:
            {
              unsigned precondition_idx =
                (*it)->as_issue_fill()->precondition_idx;
              InstructionKind generator_kind =
                instructions[precondition_idx]->get_kind();
              num_merges += generator_kind != MERGE_EVENT;
              break;
            }
          case COMPLETE_REPLAY:
            {
              unsigned completion_event_idx =
                (*it)->as_complete_replay()->rhs;
              InstructionKind generator_kind =
                instructions[completion_event_idx]->get_kind();
              num_merges += generator_kind != MERGE_EVENT;
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
        std::set<unsigned> users;
        unsigned *precondition_idx = NULL;
        switch (kind)
        {
          case COMPLETE_REPLAY:
            {
              CompleteReplay *replay = inst->as_complete_replay();
              std::map<TraceLocalID, ViewExprs>::iterator finder =
                op_views.find(replay->owner);
              if (finder == op_views.end()) break;
              find_all_last_users(finder->second, users);
              precondition_idx = &replay->rhs;
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
              find_all_last_users(finder->second, users);
              precondition_idx = &copy->precondition_idx;
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
              find_all_last_users(finder->second, users);
              precondition_idx = &fill->precondition_idx;
              break;
            }
          default:
            {
              break;
            }
        }

        if (users.size() > 0)
        {
          Instruction *generator_inst = instructions[*precondition_idx];
          if (generator_inst->get_kind() == MERGE_EVENT)
          {
            MergeEvent *merge = generator_inst->as_merge_event();
            merge->rhs.insert(users.begin(), users.end());
          }
          else
          {
            unsigned merging_event_idx = merge_starts++;
            if (*precondition_idx != fence_completion_id)
              users.insert(*precondition_idx);
            gen[merging_event_idx] = new_instructions.size();
            new_instructions.push_back(
                new MergeEvent(*this, merging_event_idx, users,
                               generator_inst->owner));
            *precondition_idx = merging_event_idx;
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

      std::vector<unsigned> inv_gen;
      inv_gen.resize(instructions.size());
      for (unsigned idx = 0; idx < gen.size(); ++idx)
        inv_gen[gen[idx]] = idx;
      std::vector<Instruction*> to_delete;
      std::vector<unsigned> new_gen;
      new_gen.resize(gen.size());
      new_gen[fence_completion_id] = 0;
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
          new_gen[inv_gen[idx]] = new_instructions.size();
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
      slice_tasks.resize(replay_parallelism);
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
        if (it->second.second)
          slice_tasks[slice_index].push_back(it->first);
      }
      for (unsigned idx = 1; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        const TraceLocalID &owner = inst->owner;
        std::map<TraceLocalID, unsigned>::iterator finder =
          slice_indices_by_owner.find(owner);
        unsigned slice_index = -1U;
        if (finder != slice_indices_by_owner.end())
          slice_index = finder->second;
        else
        {
          slice_index = next_slice_id;
          next_slice_id = (next_slice_id + 1) % replay_parallelism;
        }
        slices[slice_index].push_back(inst);
        slice_indices_by_inst[idx] = slice_index;

        if (inst->get_kind() == MERGE_EVENT)
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
              unsigned generator_slice = slice_indices_by_inst[gen[rh]];
#ifdef DEBUG_LEGION
              assert(generator_slice != -1U);
#endif
              if (generator_slice != slice_index)
              {
                crossing_found = true;
                std::map<unsigned, unsigned>::iterator finder =
                  crossing_events.find(rh);
                if (finder != crossing_events.end())
                  new_rhs.insert(finder->second);
                else
                {
                  unsigned new_crossing_event = events.size();
                  events.resize(events.size() + 1);
                  user_events.resize(events.size());
                  crossing_events[rh] = new_crossing_event;
                  new_rhs.insert(new_crossing_event);
                  slices[generator_slice].push_back(
                      new TriggerEvent(*this, new_crossing_event, rh,
                        instructions[gen[rh]]->owner));
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
          unsigned *event_to_check = NULL;
          switch (inst->get_kind())
          {
            case TRIGGER_EVENT :
              {
                event_to_check = &inst->as_trigger_event()->rhs;
                break;
              }
            case ISSUE_COPY :
              {
                event_to_check = &inst->as_issue_copy()->precondition_idx;
                break;
              }
            case ISSUE_FILL :
              {
                event_to_check = &inst->as_issue_fill()->precondition_idx;
                break;
              }
            case SET_EFFECTS :
              {
                event_to_check = &inst->as_set_effects()->rhs;
                break;
              }
            case COMPLETE_REPLAY :
              {
                event_to_check = &inst->as_complete_replay()->rhs;
                break;
              }
            default:
              {
                break;
              }
          }
          if (event_to_check != NULL)
          {
            unsigned ev = *event_to_check;
            unsigned generator_slice = slice_indices_by_inst[gen[ev]];
#ifdef DEBUG_LEGION
            assert(generator_slice != -1U);
#endif
            if (generator_slice != slice_index)
            {
              std::map<unsigned, unsigned>::iterator finder =
                crossing_events.find(ev);
              if (finder != crossing_events.end())
                *event_to_check = finder->second;
              else
              {
                unsigned new_crossing_event = events.size();
                events.resize(events.size() + 1);
                user_events.resize(events.size());
                crossing_events[ev] = new_crossing_event;
                *event_to_check = new_crossing_event;
                slices[generator_slice].push_back(
                    new TriggerEvent(*this, new_crossing_event, ev,
                      instructions[gen[ev]]->owner));
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::transitive_reduction(void)
    //--------------------------------------------------------------------------
    {
      // Transitive reduction inspired by Klaus Simon,
      // "An improved algorithm for transitive closure on acyclic digraphs"

      // First, build a DAG and find nodes with no incoming edges
      std::vector<unsigned> topo_order;
      topo_order.reserve(instructions.size());
      std::vector<unsigned> inv_topo_order(events.size(), -1U);
      std::vector<std::vector<unsigned> > incoming;
      std::vector<std::vector<unsigned> > outgoing;
      incoming.resize(events.size());
      outgoing.resize(events.size());

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
      std::vector<unsigned> remaining_edges;
      remaining_edges.resize(incoming.size());
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
      std::vector<std::vector<int> > all_chain_frontiers;
      std::vector<std::vector<unsigned> > incoming_reduced;
      all_chain_frontiers.resize(topo_order.size());
      incoming_reduced.resize(topo_order.size());
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
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
        if (instructions[idx]->get_kind() == MERGE_EVENT)
        {
          MergeEvent *merge = instructions[idx]->as_merge_event();
          unsigned order = inv_topo_order[merge->lhs];
#ifdef DEBUG_LEGION
          assert(order != -1U);
#endif
          const std::vector<unsigned> &in_reduced = incoming_reduced[order];
          std::set<unsigned> new_rhs;
          for (unsigned iidx = 0; iidx < in_reduced.size(); ++iidx)
          {
#ifdef DEBUG_LEGION
            assert(merge->rhs.find(in_reduced[iidx]) != merge->rhs.end());
#endif
            new_rhs.insert(in_reduced[iidx]);
          }
          if (new_rhs.size() < merge->rhs.size())
            merge->rhs.swap(new_rhs);
        }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::propagate_copies(std::vector<unsigned> &gen)
    //--------------------------------------------------------------------------
    {
      std::vector<int> substs(events.size(), -1);
      std::vector<Instruction*> new_instructions;
      new_instructions.reserve(instructions.size());
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

      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        int lhs = -1;
        switch (inst->get_kind())
        {
          case GET_TERM_EVENT :
            {
              GetTermEvent *term = inst->as_get_term_event();
              lhs = term->lhs;
              break;
            }
          case CREATE_AP_USER_EVENT :
            {
              CreateApUserEvent *create = inst->as_create_ap_user_event();
              lhs = create->lhs;
              break;
            }
          case TRIGGER_EVENT :
            {
              TriggerEvent *trigger = inst->as_trigger_event();
              int subst = substs[trigger->rhs];
              if (subst >= 0) trigger->rhs = (unsigned)subst;
              break;
            }
          case MERGE_EVENT :
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
          case ISSUE_COPY :
            {
              IssueCopy *copy = inst->as_issue_copy();
              int subst = substs[copy->precondition_idx];
              if (subst >= 0) copy->precondition_idx = (unsigned)subst;
              lhs = copy->lhs;
              break;
            }
          case ISSUE_FILL :
            {
              IssueFill *fill = inst->as_issue_fill();
              int subst = substs[fill->precondition_idx];
              if (subst >= 0) fill->precondition_idx = (unsigned)subst;
              lhs = fill->lhs;
              break;
            }
          case SET_EFFECTS :
            {
              SetEffects *effects = inst->as_set_effects();
              int subst = substs[effects->rhs];
              if (subst >= 0) effects->rhs = (unsigned)subst;
              break;
            }
          case SET_OP_SYNC_EVENT :
            {
              SetOpSyncEvent *sync = inst->as_set_op_sync_event();
              lhs = sync->lhs;
              break;
            }
          case ASSIGN_FENCE_COMPLETION :
            {
              lhs = fence_completion_id;
              break;
            }
          case COMPLETE_REPLAY :
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
        if (lhs != -1)
          gen[lhs] = idx;
      }
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
      log_tracing.info() << "#### " << replayable << " " << this << " ####";
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
        log_tracing.info() << "  " << (*it)->to_string();
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
      applied_events.insert(remote_applied);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_mapper_output(Memoizable *memo,
                                            const Mapper::MapTaskOutput &output,
                              const std::deque<InstanceSet> &physical_instances)
    //--------------------------------------------------------------------------
    {
      const TraceLocalID op_key = memo->get_trace_local_id();
      AutoLock t_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
      assert(cached_mappings.find(op_key) == cached_mappings.end());
#endif
      CachedMapping &mapping = cached_mappings[op_key];
      // If you change the things recorded from output here then
      // you also need to change RemoteTraceRecorder::record_mapper_output
      mapping.target_procs = output.target_procs;
      mapping.chosen_variant = output.chosen_variant;
      mapping.task_priority = output.task_priority;
      mapping.postmap_task = output.postmap_task;
      mapping.physical_instances = physical_instances;
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
            ref.add_valid_reference(MAPPING_ACQUIRE_REF);
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
      AutoLock t_lock(template_lock, 1, false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(is_replaying());
#endif

      TraceLocalID op_key = task->get_trace_local_id();
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
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      unsigned lhs_ = convert_event(lhs);
      TraceLocalID key = record_memo_entry(memo, lhs_);
      insert_instruction(new GetTermEvent(*this, lhs_, key));
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
      user_events.resize(events.size());
      user_events.push_back(lhs);

      insert_instruction(new CreateApUserEvent(*this, lhs_,
            find_trace_local_id(memo)));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_trigger_event(ApUserEvent lhs, ApEvent rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs.exists());
      assert(rhs.exists());
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      events.push_back(ApEvent());
      unsigned lhs_ = find_event(lhs);
      insert_instruction(new TriggerEvent(*this, lhs_, find_event(rhs),
            instructions[lhs_]->owner));
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
    void PhysicalTemplate::record_issue_copy(Memoizable *memo,
                                             unsigned src_idx,
                                             unsigned dst_idx,
                                             ApEvent &lhs,
                                             IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField>& src_fields,
                                 const std::vector<CopySrcDstField>& dst_fields,
#ifdef LEGION_SPY
                                             FieldSpace handle,
                                             RegionTreeID src_tree_id,
                                             RegionTreeID dst_tree_id,
#endif
                                             ApEvent precondition,
                                             ReductionOpID redop,
                                             bool reduction_fold,
                                 const FieldMaskSet<InstanceView> &tracing_srcs,
                                 const FieldMaskSet<InstanceView> &tracing_dsts)
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

      LegionList<FieldSet<EquivalenceSet*> >::aligned src_eqs, dst_eqs;
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

      unsigned lhs_ = convert_event(lhs);
      insert_instruction(new IssueCopy(
            *this, lhs_, expr, find_trace_local_id(memo),
            src_fields, dst_fields,
#ifdef LEGION_SPY
            handle, src_tree_id, dst_tree_id,
#endif
            find_event(precondition), redop, reduction_fold));

      record_views(lhs_, expr, RegionUsage(READ_ONLY, EXCLUSIVE, 0), 
                   tracing_srcs, src_eqs);
      record_copy_views(lhs_, expr, tracing_srcs);
      record_views(lhs_, expr, RegionUsage(WRITE_ONLY, EXCLUSIVE, 0), 
                   tracing_dsts, dst_eqs);
      record_copy_views(lhs_, expr, tracing_dsts);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_indirect(Memoizable *memo, ApEvent &lhs,
                             IndexSpaceExpression *expr,
                             const std::vector<CopySrcDstField>& src_fields,
                             const std::vector<CopySrcDstField>& dst_fields,
                             const std::vector<void*> &indirections,
                             ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      // TODO: support for tracing of gather/scatter/indirect operations
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_fill(Memoizable *memo,
                                             unsigned idx,
                                             ApEvent &lhs,
                                             IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField> &fields,
                                             const void *fill_value, 
                                             size_t fill_size,
#ifdef LEGION_SPY
                                             FieldSpace handle,
                                             RegionTreeID tree_id,
#endif
                                             ApEvent precondition,
                                 const FieldMaskSet<FillView> &tracing_srcs,
                                 const FieldMaskSet<InstanceView> &tracing_dsts)
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

      // Do this before we take the lock
      LegionList<FieldSet<EquivalenceSet*> >::aligned eqs;
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

      unsigned lhs_ = convert_event(lhs);
      insert_instruction(new IssueFill(*this, lhs_, expr,
                                       find_trace_local_id(memo),
                                       fields, fill_value, fill_size, 
#ifdef LEGION_SPY
                                       handle, tree_id,
#endif
                                       find_event(precondition)));

      record_fill_views(tracing_srcs);
      record_views(lhs_, expr, RegionUsage(WRITE_ONLY, EXCLUSIVE, 0), 
                   tracing_dsts, eqs);
      record_copy_views(lhs_, expr, tracing_dsts);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_fill_for_reduction(Memoizable *memo,
                                                           unsigned idx,
                                                           InstanceView *view,
                                                     const FieldMask &user_mask,
                                                     IndexSpaceExpression *expr)
    //--------------------------------------------------------------------------
    {
      if (expr->is_empty()) return;

      TraceLocalID op_key = find_trace_local_id(memo);
      ReductionView *reduction_view = view->as_reduction_view();
      ReductionManager *manager = reduction_view->manager;
      LayoutDescription *const layout = manager->layout;
      const ReductionOp *reduction_op = manager->op;

      std::vector<CopySrcDstField> fields;
      std::vector<FieldID> fill_fields;
      manager->field_space_node->get_field_set(user_mask,
          trace->logical_trace->ctx, fill_fields);
      layout->compute_copy_offsets(fill_fields, manager, fields);

      size_t fill_size = reduction_op->sizeof_rhs;
      void *fill_value = malloc(fill_size);
      reduction_op->init(fill_value, 1);

      ApEvent lhs;
      {
        Realm::UserEvent e(Realm::UserEvent::create_user_event());
        e.trigger();
        lhs = ApEvent(e);
      }
      unsigned lhs_ = convert_event(lhs);
      insert_instruction(new IssueFill(*this, lhs_, expr, op_key,
                                       fields, fill_value, fill_size,
#ifdef LEGION_SPY
                                       manager->field_space_node->handle,
                                       manager->tree_id,
#endif
                                       fence_completion_id));
      reduction_ready_events[op_key].insert(lhs);

      FieldMaskSet<InstanceView> views;
      views.insert(view, user_mask);
      record_copy_views(lhs_, expr, views);

      const RegionUsage usage(WRITE_ONLY, EXCLUSIVE, 0);
      add_view_user(view, usage, lhs_, expr, user_mask);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::get_reduction_ready_events(
                              Memoizable *memo, std::set<ApEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(template_lock, 1, false/*exclusive*/);
      std::map<TraceLocalID,std::set<ApEvent> >::iterator finder =
        reduction_ready_events.find(find_trace_local_id(memo));
      if (finder != reduction_ready_events.end())
        ready_events.insert(finder->second.begin(), finder->second.end());
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
      LegionList<FieldSet<EquivalenceSet*> >::aligned eqs;
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
      for (LegionList<FieldSet<EquivalenceSet*> >::aligned::iterator it =
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
            if (view->is_reduction_view() && IS_REDUCE(usage))
              record_issue_fill_for_reduction(memo, idx, view, mask, expr);
            update_valid_views(view, *eit, usage, mask, true);
            add_view_user(view, usage, entry, expr, mask);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_fill_view(
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
    void PhysicalTemplate::record_views(unsigned entry,
                                        IndexSpaceExpression *expr,
                                        const RegionUsage &usage,
                                        const FieldMaskSet<InstanceView> &views,
                     const LegionList<FieldSet<EquivalenceSet*> >::aligned &eqs)
    //--------------------------------------------------------------------------
    {
      RegionTreeForest *forest = trace->runtime->forest;
      for (FieldMaskSet<InstanceView>::const_iterator vit = views.begin();
            vit != views.end(); ++vit)
      {
        for (LegionList<FieldSet<EquivalenceSet*> >::aligned::const_iterator 
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
          check_dependence_type(it->first->usage, user->usage);
        if (dep == NO_DEPENDENCE)
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
    void PhysicalTemplate::record_copy_views(unsigned copy_id,
                                             IndexSpaceExpression *expr,
                                        const FieldMaskSet<InstanceView> &views)
    //--------------------------------------------------------------------------
    {
      ViewExprs &cviews = copy_views[copy_id];
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
    RtEvent PhysicalTemplate::defer_template_deletion(void)
    //--------------------------------------------------------------------------
    {
      ApEvent wait_on = get_completion_for_deletion();
      DeleteTemplateArgs args(this);
      return trace->runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY,
          Runtime::protect_event(wait_on));
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalTemplate::handle_replay_slice(const void *args)
    //--------------------------------------------------------------------------
    {
      const ReplaySliceArgs *pargs = (const ReplaySliceArgs*)args;
      pargs->tpl->execute_slice(pargs->slice_index);
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
      if (!recording_done.has_triggered())
        Runtime::trigger_event(recording_done);
    }

    //--------------------------------------------------------------------------
    TraceLocalID PhysicalTemplate::find_trace_local_id(Memoizable *memo)
    //--------------------------------------------------------------------------
    {
      TraceLocalID op_key = memo->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.find(op_key) != operations.end());
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
      assert(operations.find(key) == operations.end());
      assert(memo_entries.find(key) == memo_entries.end());
#endif
      operations[key] = memo;
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

    /////////////////////////////////////////////////////////////
    // Instruction
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Instruction::Instruction(PhysicalTemplate& tpl, const TraceLocalID &o)
      : operations(tpl.operations), events(tpl.events),
        user_events(tpl.user_events), owner(o)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // GetTermEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GetTermEvent::GetTermEvent(PhysicalTemplate& tpl, unsigned l,
                               const TraceLocalID& r)
      : Instruction(tpl, r), lhs(l)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
      assert(operations.find(owner) != operations.end());
#endif
    }

    //--------------------------------------------------------------------------
    void GetTermEvent::execute(void)
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
    std::string GetTermEvent::to_string(void)
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
      assert(lhs < events.size());
      assert(lhs < user_events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void CreateApUserEvent::execute(void)
    //--------------------------------------------------------------------------
    {
      ApUserEvent ev = Runtime::create_ap_user_event();
      events[lhs] = ev;
      user_events[lhs] = ev;
    }

    //--------------------------------------------------------------------------
    std::string CreateApUserEvent::to_string(void)
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
      assert(lhs < events.size());
      assert(lhs < user_events.size());
      assert(rhs < events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void TriggerEvent::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(events[lhs].exists());
      assert(user_events[lhs].exists());
      assert(events[lhs].id == user_events[lhs].id);
#endif
      Runtime::trigger_event(user_events[lhs], events[rhs]);
    }

    //--------------------------------------------------------------------------
    std::string TriggerEvent::to_string(void)
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
      assert(lhs < events.size());
      assert(rhs.size() > 0);
      for (std::set<unsigned>::iterator it = rhs.begin(); it != rhs.end();
           ++it)
        assert(*it < events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void MergeEvent::execute(void)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> to_merge;
      for (std::set<unsigned>::iterator it = rhs.begin(); it != rhs.end();
           ++it)
      {
#ifdef DEBUG_LEGION
        assert(*it < events.size());
#endif
        to_merge.insert(events[*it]);
      }
      ApEvent result = Runtime::merge_events(NULL, to_merge);
      events[lhs] = result;
    }

    //--------------------------------------------------------------------------
    std::string MergeEvent::to_string(void)
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
      assert(lhs < events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void AssignFenceCompletion::execute(void)
    //--------------------------------------------------------------------------
    {
      events[lhs] = tpl.get_fence_completion();
    }

    //--------------------------------------------------------------------------
    std::string AssignFenceCompletion::to_string(void)
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
#ifdef LEGION_SPY
                         FieldSpace h,RegionTreeID src_tid,RegionTreeID dst_tid,
#endif
                         unsigned pi, ReductionOpID ro, bool rf)
      : Instruction(tpl, key), lhs(l), expr(e), src_fields(s), dst_fields(d), 
#ifdef LEGION_SPY
        handle(h), src_tree_id(src_tid), dst_tree_id(dst_tid),
#endif
        precondition_idx(pi), redop(ro), reduction_fold(rf)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
      assert(operations.find(owner) != operations.end());
      assert(src_fields.size() > 0);
      assert(dst_fields.size() > 0);
      assert(precondition_idx < events.size());
      assert(expr != NULL);
#endif
      expr->add_expression_reference();
    }

    //--------------------------------------------------------------------------
    IssueCopy::~IssueCopy(void)
    //--------------------------------------------------------------------------
    {
      if (expr->remove_expression_reference())
        delete expr;
    }

    //--------------------------------------------------------------------------
    void IssueCopy::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      Memoizable *memo = operations[owner];
      ApEvent precondition = events[precondition_idx];
      const PhysicalTraceInfo trace_info(memo->get_operation(), -1U, false);
      events[lhs] = expr->issue_copy(trace_info, dst_fields, src_fields,
#ifdef LEGION_SPY
                                     handle, src_tree_id, dst_tree_id,
#endif
                                     precondition, PredEvent::NO_PRED_EVENT,
                                     redop, reduction_fold, NULL, NULL);
    }

    //--------------------------------------------------------------------------
    std::string IssueCopy::to_string(void)
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

      if (redop != 0) ss << ", " << redop;
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
      assert(lhs < events.size());
      assert(operations.find(owner) != operations.end());
      assert(fields.size() > 0);
      assert(precondition_idx < events.size());
#endif
      expr->add_expression_reference();
      fill_value = malloc(fill_size);
      memcpy(fill_value, value, fill_size);
    }

    //--------------------------------------------------------------------------
    IssueFill::~IssueFill(void)
    //--------------------------------------------------------------------------
    {
      if (expr->remove_expression_reference())
        delete expr;
      free(fill_value);
    }

    //--------------------------------------------------------------------------
    void IssueFill::execute(void)
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
                                     precondition, PredEvent::NO_PRED_EVENT,
                                     NULL, NULL);
    }

    //--------------------------------------------------------------------------
    std::string IssueFill::to_string(void)
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
      assert(lhs < events.size());
      assert(operations.find(owner) != operations.end());
#endif
    }

    //--------------------------------------------------------------------------
    void SetOpSyncEvent::execute(void)
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
    std::string SetOpSyncEvent::to_string(void)
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
      assert(rhs < events.size());
      assert(operations.find(owner) != operations.end());
#endif
    }

    //--------------------------------------------------------------------------
    void SetEffects::execute(void)
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
    std::string SetEffects::to_string(void)
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
      assert(operations.find(owner) != operations.end());
      assert(rhs < events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void CompleteReplay::execute(void)
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
    std::string CompleteReplay::to_string(void)
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

