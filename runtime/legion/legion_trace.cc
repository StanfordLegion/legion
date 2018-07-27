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

    /////////////////////////////////////////////////////////////
    // LegionTrace 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LegionTrace::LegionTrace(TaskContext *c, bool logical_only)
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
    StaticTrace::StaticTrace(TaskContext *c,const std::set<RegionTreeID> *trees)
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
      if (!implicit_runtime->no_physical_tracing &&
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
    DynamicTrace::DynamicTrace(TraceID t, TaskContext *c, bool logical_only)
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
      if (!implicit_runtime->no_physical_tracing &&
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
      execution_fence_event = parent_ctx->register_fence_dependence(this);
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
    void TraceCaptureOp::initialize_capture(TaskContext *ctx, bool has_block)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, MIXED_FENCE);
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
#ifdef DEBUG_LEGION
        assert(local_trace->get_physical_trace() != NULL);
#endif
        current_template =
          local_trace->get_physical_trace()->get_current_template();
        local_trace->get_physical_trace()->record_previous_template_completion(
            get_completion_event());
      }
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::trigger_mapping(void)
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
          local_trace->get_physical_trace()->fix_trace(
              current_template, has_blocking_call);
        if (pending_deletion.exists())
          execution_precondition = Runtime::merge_events(
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
    void TraceCompleteOp::initialize_complete(TaskContext *ctx, bool has_block)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, MIXED_FENCE);
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
      if (local_trace->is_replaying())
      {
#ifdef DEBUG_LEGION
        assert(local_trace->get_physical_trace() != NULL);
#endif
        PhysicalTemplate *current_template =
          local_trace->get_physical_trace()->get_current_template();
#ifdef DEBUG_LEGION
        assert(current_template != NULL);
#endif
#ifdef LEGION_SPY
        local_trace->perform_logging(
            current_template->get_fence_uid(), unique_op_id);
#endif
        current_template->execute_all();
        template_completion = current_template->get_completion();
        Runtime::trigger_event(completion_event, template_completion);
        local_trace->end_trace_execution(this);
        parent_ctx->update_current_fence(this, true, true);
        parent_ctx->record_previous_trace(local_trace);
        local_trace->get_physical_trace()->record_previous_template_completion(
            template_completion);
        local_trace->initialize_tracing_state();
        replayed = true;
        return;
      }
      else if (local_trace->is_recording())
      {
#ifdef DEBUG_LEGION
        assert(local_trace->get_physical_trace() != NULL);
#endif
        current_template =
          local_trace->get_physical_trace()->get_current_template();
        local_trace->get_physical_trace()->record_previous_template_completion(
            get_completion_event());
      }

      // Indicate that this trace is done being captured
      // This also registers that we have dependences on all operations
      // in the trace.
      local_trace->end_trace_execution(this);

      // We always need to run the full fence analysis, otherwise
      // the operations replayed in the following trace will race
      // with those in the current trace
      execution_precondition =
        parent_ctx->perform_fence_analysis(this, true, true);

      // Now update the parent context with this fence before we can complete
      // the dependence analysis and possibly be deactivated
      parent_ctx->update_current_fence(this, true, true);

      // If this is a static trace, then we remove our reference when we're done
      if (local_trace->is_static_trace())
      {
        StaticTrace *static_trace = static_cast<StaticTrace*>(local_trace);
        if (static_trace->remove_reference())
          delete static_trace;
      }
      parent_ctx->record_previous_trace(local_trace);
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
          local_trace->get_physical_trace()->fix_trace(
              current_template, has_blocking_call);
        if (pending_deletion.exists())
          execution_precondition = Runtime::merge_events(
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
    void TraceReplayOp::initialize_replay(TaskContext *ctx, LegionTrace *trace)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, EXECUTION_FENCE);
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
        if (physical_trace->has_any_templates() || is_recording)
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
          physical_trace->check_template_preconditions();
#ifdef DEBUG_LEGION
        assert(physical_trace->get_current_template() == NULL ||
               !physical_trace->get_current_template()->is_recording());
#endif

        // Register this fence with all previous users in the parent's context
#ifdef LEGION_SPY
        execution_precondition = 
          parent_ctx->perform_fence_analysis(this, true, true);
#else
        execution_precondition = 
          parent_ctx->perform_fence_analysis(this, false, true);
#endif
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
#ifdef LEGION_SPY
        execution_precondition = 
          parent_ctx->perform_fence_analysis(this, true, true);
#else
        execution_precondition = 
          parent_ctx->perform_fence_analysis(this, false, true);
#endif
      }

      // Now update the parent context with this fence before we can complete
      // the dependence analysis and possibly be deactivated
#ifdef LEGION_SPY
      parent_ctx->update_current_fence(this, true, true);
#else
      parent_ctx->update_current_fence(this, false, true);
#endif
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
    void TraceBeginOp::initialize_begin(TaskContext *ctx, LegionTrace *trace)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, MAPPING_FENCE);
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
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceSummaryOp::TraceSummaryOp(const TraceSummaryOp &rhs)
      : Operation(NULL)
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
    void TraceSummaryOp::initialize_summary(
                                  TaskContext *ctx,
                                  UniqueID creator_op_id,
                                  const std::vector<RegionRequirement> &reqs,
                                  const std::vector<InstanceSet> &insts,
                                  const std::vector<unsigned> &indices)
    //--------------------------------------------------------------------------
    {
      size_t num_requirements = reqs.size();
      initialize_operation(ctx, false, num_requirements);
      // We actually want to track summary operations
      track_parent = true;
      context_index = ctx->register_new_summary_operation(this);
      requirements = reqs;
      instances = insts;
      parent_indices = indices;
      privilege_paths.resize(num_requirements);
      for (unsigned idx = 0; idx < num_requirements; ++idx)
        initialize_privilege_path(privilege_paths[idx], requirements[idx]);
      version_infos.resize(num_requirements);
      restrict_infos.resize(num_requirements);
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_summary_operation(parent_ctx->get_unique_id(),
                                         unique_op_id);
        LegionSpy::log_summary_op_creator(unique_op_id, creator_op_id);
        perform_logging();
      }
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::perform_logging(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < requirements.size(); ++idx)
      {
        const RegionRequirement &requirement = requirements[idx];
        if (requirement.handle_type == PART_PROJECTION)
          LegionSpy::log_logical_requirement(unique_op_id, idx,
                                    false/*region*/,
                                    requirement.partition.index_partition.id,
                                    requirement.partition.field_space.id,
                                    requirement.partition.tree_id,
                                    requirement.privilege,
                                    requirement.prop,
                                    requirement.redop,
                                    requirement.parent.index_space.id);
        else
          LegionSpy::log_logical_requirement(unique_op_id, idx,
                                    true/*region*/,
                                    requirement.region.index_space.id,
                                    requirement.region.field_space.id,
                                    requirement.region.tree_id,
                                    requirement.privilege,
                                    requirement.prop,
                                    requirement.redop,
                                    requirement.parent.index_space.id);
        LegionSpy::log_requirement_fields(unique_op_id, idx,
                                  requirement.privilege_fields);
        runtime->forest->log_mapping_decision(unique_op_id, idx,
            requirement, instances[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      requirements.clear();
      instances.clear();
      parent_indices.clear();
      privilege_paths.clear();
      version_infos.clear();
      restrict_infos.clear();
      map_applied_conditions.clear();
      mapped_preconditions.clear();
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
      ProjectionInfo projection_info;
      for (unsigned idx = 0; idx < requirements.size(); ++idx)
      {
        runtime->forest->perform_dependence_analysis(this, idx,
                                                     requirements[idx],
                                                     restrict_infos[idx],
                                                     version_infos[idx],
                                                     projection_info,
                                                     privilege_paths[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Compute the version numbers for this mapping operation
      std::set<RtEvent> preconditions;
      for (unsigned idx = 0; idx < requirements.size(); ++idx)
        runtime->forest->perform_versioning_analysis(this, idx,
                                                     requirements[idx],
                                                     privilege_paths[idx],
                                                     version_infos[idx],
                                                     preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      PhysicalTraceInfo trace_info;
      for (unsigned idx = 0; idx < requirements.size(); ++idx)
        runtime->forest->physical_register_only(requirements[idx],
                                                version_infos[idx],
                                                restrict_infos[idx],
                                                this, idx,
                                                completion_event,
                                                false/*defer add users*/,
                                                false/*read only locks*/,
                                                map_applied_conditions,
                                                instances[idx],
                                                NULL/*advance projections*/,
                                                trace_info
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
      for (unsigned idx = 0; idx < requirements.size(); ++idx)
        version_infos[idx].apply_mapping(map_applied_conditions);
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();

      std::set<ApEvent> wait_events;
      if (execution_fence_event.exists())
        wait_events.insert(execution_fence_event);
      for (unsigned idx = 0; idx < instances.size(); ++idx)
        instances[idx].update_wait_on_events(wait_events);
      ApEvent wait_event = Runtime::merge_events(wait_events);
#ifdef LEGION_SPY
      LegionSpy::log_operation_events(unique_op_id, wait_event,
                                      completion_event);
#endif
      complete_execution(Runtime::protect_event(wait_event));
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      for (std::vector<VersionInfo>::iterator it = version_infos.begin();
           it != version_infos.end(); it++)
        it->clear();
      commit_operation(true);
    }

    //--------------------------------------------------------------------------
    unsigned TraceSummaryOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      return parent_indices[idx];
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTrace
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTrace::PhysicalTrace(Runtime *rt, LegionTrace *lt)
      : runtime(rt), logical_trace(lt), current_template(NULL),
        nonreplayable_count(0)
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
        nonreplayable_count(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalTrace::~PhysicalTrace()
    //--------------------------------------------------------------------------
    {
      for (std::vector<PhysicalTemplate*>::iterator it = templates.begin();
           it != templates.end(); ++it)
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
                                  PhysicalTemplate *tpl, bool has_blocking_call)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tpl->is_recording());
#endif
      tpl->finalize(has_blocking_call);
      RtEvent pending_deletion = RtEvent::NO_RT_EVENT;
      if (!tpl->is_replayable())
      {
        pending_deletion = tpl->defer_template_deletion();
        current_template = NULL;
        if (++nonreplayable_count > LEGION_NON_REPLAYABLE_WARNING)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_NON_REPLAYABLE_COUNT_EXCEEDED,
              "WARNING: The runtime has failed to memoize the trace more than "
              "%u times, due to the absence of a replayable template. It is "
              "highly likely that trace %u will not be memoized for the rest "
              "of execution. Please change the mapper to stop making "
              "memoization requests.", LEGION_NON_REPLAYABLE_WARNING,
              logical_trace->get_trace_id())
          nonreplayable_count = 0;
        }
      }
      else
      {
        // Reset the nonreplayable count when we find a replayable template
        nonreplayable_count = 0;
        templates.push_back(tpl);
      }
      return pending_deletion;
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::check_template_preconditions(void)
    //--------------------------------------------------------------------------
    {
      current_template = NULL;
      for (std::vector<PhysicalTemplate*>::reverse_iterator it =
           templates.rbegin(); it != templates.rend(); ++it)
        if ((*it)->check_preconditions())
        {
#ifdef DEBUG_LEGION
          assert((*it)->is_replayable());
#endif
          // Reset the nonreplayable count when a replayable template satisfies
          // the precondition
          nonreplayable_count = 0;
          current_template = *it;
          return;
        }
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate* PhysicalTrace::start_new_template(ApEvent fence_event)
    //--------------------------------------------------------------------------
    {
      current_template = new PhysicalTemplate(this, fence_event);
#ifdef DEBUG_LEGION
      assert(fence_event.exists());
#endif
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
    // PhysicalTemplate
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTemplate::PhysicalTemplate(PhysicalTrace *t, ApEvent fence_event)
      : trace(t), recording(true), replayable(true), fence_completion_id(0),
        replay_parallelism(implicit_runtime->max_replay_parallelism)
    //--------------------------------------------------------------------------
    {
      events.push_back(fence_event);
      event_map[fence_event] = fence_completion_id;
      instructions.push_back(
         new AssignFenceCompletion(*this, fence_completion_id, TraceLocalID()));
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate::PhysicalTemplate(const PhysicalTemplate &rhs)
      : trace(NULL), recording(true), replayable(true), fence_completion_id(0),
        replay_parallelism(1)
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
            pit->remove_valid_references(MAPPING_ACQUIRE_REF);
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
        {
          if (events[it->first].exists())
            events[it->second] = events[it->first];
          else
            events[it->second] = completion;
        }
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
      std::vector<Processor> &replay_targets = trace->replay_targets;
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
      for (std::map<TraceLocalID, Operation*>::iterator it =
           operations.begin(); it != operations.end(); ++it)
        it->second = NULL;
#endif
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalTemplate::get_completion(void) const
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> to_merge;
      for (std::map<InstanceAccess, UserInfos>::const_iterator it =
           last_users.begin(); it != last_users.end(); ++it)
        for (UserInfos::const_iterator iit = it->second.begin();
             iit != it->second.end(); ++iit)
          for (std::set<unsigned>::const_iterator uit = iit->users.begin();
               uit != iit->users.end(); ++uit)
            to_merge.insert(events[*uit]);
      return Runtime::merge_events(to_merge);
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalTemplate::get_completion_for_deletion(void) const
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> all_events;
      for (std::map<ApEvent, unsigned>::const_iterator it = event_map.begin();
           it != event_map.end(); ++it)
        all_events.insert(it->first);
      return Runtime::merge_events(all_events);
    }

    //--------------------------------------------------------------------------
    /*static*/ bool PhysicalTemplate::check_logical_open(RegionTreeNode *node,
                                                         ContextID ctx,
                                                         FieldMask fields)
    //--------------------------------------------------------------------------
    {
      {
        const LogicalState &state = node->get_logical_state(ctx);
        fields -= state.dirty_fields;
        if (!fields) return true;
      }

      RegionTreeNode *parent_node = node->get_parent();
      if (parent_node != NULL)
      {
#ifdef DEBUG_LEGION
        assert(!parent_node->is_region());
#endif
        const LogicalState &state = parent_node->get_logical_state(ctx);
#ifdef DEBUG_LEGION
        assert(!!fields);
#endif
        for (LegionList<FieldState>::aligned::const_iterator fit =
             state.field_states.begin(); fit !=
             state.field_states.end(); ++fit)
        {
          if (fit->open_state == NOT_OPEN)
            continue;
          FieldMask overlap = fit->valid_fields & fields;
          if (!overlap)
            continue;
          // FIXME: This code will not work as expected if the projection
          //        goes deeper than one level
          const LegionColor &color = node->get_row_source()->color;
          if ((fit->projection != 0 &&
               fit->projection_space->contains_color(color)) ||
              fit->open_children.find(color) != fit->open_children.end())
            fields -= overlap;
        }
      }

      const LogicalState &state = node->get_logical_state(ctx);
      for (LegionList<FieldState>::aligned::const_iterator fit =
           state.field_states.begin(); fit !=
           state.field_states.end(); ++fit)
      {
        if (fit->open_state == NOT_OPEN)
          continue;
        FieldMask overlap = fit->valid_fields & fields;
        if (!overlap)
          continue;
        fields -= overlap;
      }
      return !fields;
    }

    //--------------------------------------------------------------------------
    /*static*/ bool PhysicalTemplate::check_logical_open(RegionTreeNode *node,
                                                         ContextID ctx,
                           LegionMap<IndexSpaceNode*, FieldMask>::aligned projs)
    //--------------------------------------------------------------------------
    {
      const LogicalState &state = node->get_logical_state(ctx);
      for (LegionList<FieldState>::aligned::const_iterator fit =
           state.field_states.begin(); fit !=
           state.field_states.end(); ++fit)
      {
        if (fit->open_state == NOT_OPEN)
          continue;
        if (fit->projection != 0)
        {
          LegionMap<IndexSpaceNode*, FieldMask>::aligned::iterator finder =
            projs.find(fit->projection_space);
          if (finder != projs.end())
          {
            FieldMask overlap = finder->second & fit->valid_fields;
            if (!overlap)
              continue;
            finder->second -= overlap;
            if (!finder->second)
              projs.erase(finder);
          }
        }
      }
      return projs.size() == 0;
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::check_preconditions(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<std::pair<RegionTreeNode*, ContextID>,
                     FieldMask>::aligned::iterator it =
           previous_open_nodes.begin(); it !=
           previous_open_nodes.end(); ++it)
        if (!check_logical_open(it->first.first, it->first.second, it->second))
          return false;

      for (std::map<std::pair<RegionTreeNode*, ContextID>,
           LegionMap<IndexSpaceNode*, FieldMask>::aligned>::iterator it =
           previous_projections.begin(); it !=
           previous_projections.end(); ++it)
        if (!check_logical_open(it->first.first, it->first.second, it->second))
          return false;

      for (LegionMap<InstanceView*, FieldMask>::aligned::iterator it =
           previous_valid_views.begin(); it !=
           previous_valid_views.end(); ++it)
      {
#ifdef DEBUG_LEGION
        assert(logical_contexts.find(it->first) != logical_contexts.end());
        assert(physical_contexts.find(it->first) != physical_contexts.end());
#endif
        RegionTreeNode *logical_node = it->first->logical_node;
        ContextID logical_ctx = logical_contexts[it->first];
        std::pair<RegionTreeNode*, ContextID> key(logical_node, logical_ctx);

        if (previous_open_nodes.find(key) == previous_open_nodes.end() &&
            !check_logical_open(logical_node, logical_ctx, it->second))
          return false;

        ContextID physical_ctx = physical_contexts[it->first];
        PhysicalState *state = new PhysicalState(logical_node, false);
        VersionManager &manager =
          logical_node->get_current_version_manager(physical_ctx);
        manager.update_physical_state(state);
        state->capture_state();

        bool found = false;
        if (it->first->is_materialized_view())
        {
          for (LegionMap<LogicalView*, FieldMask,
                         VALID_VIEW_ALLOC>::track_aligned::iterator vit =
               state->valid_views.begin(); vit !=
               state->valid_views.end(); ++vit)
          {
            if (vit->first->is_materialized_view() &&
                it->first == vit->first && !(it->second - vit->second))
            {
              found = true;
              break;
            }
          }
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(it->first->is_reduction_view());
#endif
          for (LegionMap<ReductionView*, FieldMask,
                         VALID_VIEW_ALLOC>::track_aligned::iterator vit =
               state->reduction_views.begin(); vit !=
               state->reduction_views.end(); ++vit)
          {
            if (it->first == vit->first && !(it->second - vit->second))
            {
              found = true;
              break;
            }
          }
        }
        if (!found)
          return false;
      }

      return true;
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::check_replayable(void) const
    //--------------------------------------------------------------------------
    {
      if (untracked_fill_views.size() > 0)
        return false;
      for (LegionMap<InstanceView*, FieldMask>::aligned::const_iterator it =
           reduction_views.begin(); it !=
           reduction_views.end(); ++it)
      {
        if (it->first->get_manager()->instance_domain->get_volume() > 0)
          return false;
      }
      for (LegionMap<InstanceView*, FieldMask>::aligned::const_iterator it =
           previous_valid_views.begin(); it !=
           previous_valid_views.end(); ++it)
      {
        LegionMap<InstanceView*, FieldMask>::aligned::const_iterator finder =
          valid_views.find(it->first);
        if (finder == valid_views.end() || !!(it->second - finder->second))
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::register_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      Memoizable *memoizable = op->get_memoizable();
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      std::map<TraceLocalID, Operation*>::iterator op_finder =
        operations.find(memoizable->get_trace_local_id());
#ifdef DEBUG_LEGION
      assert(op_finder != operations.end());
      assert(op_finder->second == NULL);
#endif
      op_finder->second = op;
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
        operations[tasks[idx]]->set_execution_fence_event(fence);
      std::vector<Instruction*> &instructions = slices[slice_idx];
      for (std::vector<Instruction*>::const_iterator it = instructions.begin();
           it != instructions.end(); ++it)
        (*it)->execute();
      Runtime::trigger_event(fence);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::issue_summary_operations(
                                   TaskContext* context, Operation *invalidator)
    //--------------------------------------------------------------------------
    {
      Runtime *runtime = trace->runtime;
      for (std::vector<SummaryOpInfo>::iterator it = dedup_summary_ops.begin();
           it != dedup_summary_ops.end(); ++it)
      {
        TraceSummaryOp *op = runtime->get_available_summary_op();
        op->initialize_summary(context, invalidator->get_unique_op_id(),
            it->requirements, it->instances, it->parent_indices);
        context->register_executing_child(op);
        op->execute_dependence_analysis();
        op->add_mapping_reference(op->get_generation());
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::finalize(bool has_blocking_call)
    //--------------------------------------------------------------------------
    {
      recording = false;
      replayable = !has_blocking_call && check_replayable();
      if (outstanding_gc_events.size() > 0)
        for (std::map<InstanceView*, std::set<ApEvent> >::iterator it =
             outstanding_gc_events.begin(); it !=
             outstanding_gc_events.end(); ++it)
        {
          it->first->update_gc_events(it->second);
          it->first->collect_users(it->second);
        }
      if (!replayable)
      {
        if (implicit_runtime->dump_physical_traces)
        {
          optimize();
          dump_template();
        }
        return;
      }
      optimize();
      generate_summary_operations();
      if (implicit_runtime->dump_physical_traces) dump_template();
      size_t num_events = events.size();
      events.clear();
      events.resize(num_events);
      event_map.clear();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::optimize(void)
    //--------------------------------------------------------------------------
    {
      std::vector<unsigned> gen;
      if (!(implicit_runtime->no_trace_optimization ||
            implicit_runtime->no_fence_elision))
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
      if (!implicit_runtime->no_trace_optimization)
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

      // Reserve space for completion events of the previously replayed trace
      // - frontiers[idx] == (event idx from the previous trace)
      // - after each replay, we do assignment events[frontiers[idx]] = idx
      // Note that 'frontiers' is used in 'find_last_users()'
      for (std::map<InstanceAccess,UserInfos>::iterator it = last_users.begin();
           it != last_users.end(); ++it)
        for (UserInfos::iterator iit = it->second.begin(); iit !=
             it->second.end(); ++iit)
          for (std::set<unsigned>::iterator uit = iit->users.begin(); uit !=
               iit->users.end(); ++uit)
          {
            unsigned frontier = *uit;
            if (frontiers.find(frontier) == frontiers.end())
            {
              unsigned next_event_id = events.size();
              frontiers[frontier] = next_event_id;
              events.resize(next_event_id + 1);
            }
          }

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
              std::map<TraceLocalID, std::vector<InstanceReq> >::iterator
                finder = op_reqs.find(replay->owner);
              if (finder == op_reqs.end())
                break;
              const std::vector<InstanceReq> &reqs = finder->second;
              for (std::vector<InstanceReq>::const_iterator it = reqs.begin();
                   it != reqs.end(); ++it)
                for (std::vector<FieldID>::const_iterator fit =
                     it->fields.begin(); fit != it->fields.end(); ++fit)
                  find_last_users(it->instance, it->node, *fit, users);
              precondition_idx = &replay->rhs;
              break;
            }
          case ISSUE_COPY:
            {
              IssueCopy *copy = inst->as_issue_copy();
              for (unsigned idx = 0; idx < copy->src_fields.size(); ++idx)
              {
                const CopySrcDstField &field = copy->src_fields[idx];
                find_last_users(field.inst, copy->node, field.field_id, users);
              }
              for (unsigned idx = 0; idx < copy->dst_fields.size(); ++idx)
              {
                const CopySrcDstField &field = copy->dst_fields[idx];
                find_last_users(field.inst, copy->node, field.field_id, users);
              }
              precondition_idx = &copy->precondition_idx;
              break;
            }
          case ISSUE_FILL:
            {
              IssueFill *fill = inst->as_issue_fill();
              for (unsigned idx = 0; idx < fill->fields.size(); ++idx)
              {
                const CopySrcDstField &field = fill->fields[idx];
                find_last_users(field.inst, fill->node, field.field_id, users);
              }
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
          case COMPLETE_REPLAY:
            {
              CompleteReplay *complete = inst->as_complete_replay();
              used[gen[complete->rhs]] = true;
              break;
            }
          case GET_TERM_EVENT:
          case GET_OP_TERM_EVENT:
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
          if (inst->get_kind() == MERGE_EVENT)
          {
            MergeEvent *merge = inst->as_merge_event();
            if (merge->rhs.size() > 1)
              merge->rhs.erase(fence_completion_id);
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
      for (std::map<TraceLocalID, Operation*>::iterator it =
           operations.begin(); it != operations.end(); ++it)
      {
        unsigned slice_index = -1U;
        if (!round_robin_for_tasks &&
            it->second->get_operation_kind() == Operation::TASK_OP_KIND)
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
        if (it->second->get_operation_kind() == Operation::TASK_OP_KIND)
          slice_tasks[slice_index].push_back(it->first);
      }
      for (unsigned idx = 1; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        const TraceLocalID &owner = inst->owner;
        std::map<TraceLocalID, unsigned>::iterator finder =
          slice_indices_by_owner.find(owner);
#ifdef DEBUG_LEGION
        assert(finder != slice_indices_by_owner.end());
#endif
        unsigned slice_index = finder->second;
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

      std::map<TraceLocalID, Instruction*> term_insts;
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        switch (inst->get_kind())
        {
          // Pass these instructions as their events will be added later
          case GET_TERM_EVENT :
          case GET_OP_TERM_EVENT :
            {
              term_insts[inst->owner] = inst;
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
              Instruction *term_inst = term_insts[replay->owner];
              unsigned lhs = -1U;
              switch (term_inst->get_kind())
              {
                case GET_TERM_EVENT :
                  {
                    GetTermEvent *term = term_inst->as_get_term_event();
                    lhs = term->lhs;
                    break;
                  }
                case GET_OP_TERM_EVENT :
                  {
                    GetOpTermEvent *term = term_inst->as_get_op_term_event();
                    lhs = term->lhs;
                    break;
                  }
                default:
                  {
                    assert(false);
                    break;
                  }
              }
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
        while (chain_indices[pos] != -1U && pos >= 0)
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
          const std::vector<unsigned> &in_reduced =
            incoming_reduced[inv_topo_order[merge->lhs]];
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
          case GET_OP_TERM_EVENT :
            {
              GetOpTermEvent *term = inst->as_get_op_term_event();
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
    void PhysicalTemplate::push_complete_replays()
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
    void PhysicalTemplate::generate_summary_operations(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeForest *forest = trace->runtime->forest;

      typedef std::pair<RegionTreeNode*, PhysicalManager*> DedupKey;
      LegionMap<DedupKey, FieldMask>::aligned covered_fields;
      std::vector<SummaryOpInfo> summary_ops;

      for (int idx = summary_info.size() - 1; idx >= 0; --idx)
      {
        const std::pair<RegionRequirement, InstanceSet> &pair =
          summary_info[idx];
        const RegionRequirement &req = pair.first;
        const InstanceSet &insts = pair.second;

        // We do not need to bump the version number for read-only regions.
        // We can also ignore reductions because we reject traces that end with
        // reduction tasks.
        if (!HAS_WRITE(req) || IS_REDUCE(req)) continue;

        RegionTreeNode *region_node = forest->get_node(req.region);
        FieldSpaceNode *field_node = region_node->get_column_source();
        FieldMask fields = field_node->get_field_mask(req.privilege_fields);

        summary_ops.push_back(SummaryOpInfo());
        SummaryOpInfo &summary_op = summary_ops.back();
        bool dedup = true;
        FieldMask all_uncovered;
        InstanceSet new_insts;
        for (unsigned iidx = 0; iidx < insts.size(); ++iidx)
        {
          const InstanceRef &ref = insts[iidx];
#ifdef DEBUG_LEGION
          assert(!(ref.get_valid_fields() - fields));
#endif
          FieldMask uncovered = fields & ref.get_valid_fields();
          // We only need to consider fields that are not yet analyzed.
          // If all the fields are covered by the summary operations generated
          // so far, we simply ignore the region requirement.
          // We also rememeber whether any of the fields are deduplicated so that
          // we know later whether we should match the set of fields in the
          // requirement with those actually summarized.
          DedupKey key(region_node, ref.get_manager());
          LegionMap<DedupKey, FieldMask>::aligned::iterator cf_finder =
            covered_fields.find(key);
          bool fields_narrowed = false;
          if (cf_finder == covered_fields.end())
            covered_fields[key] = uncovered;
          else
          {
            FieldMask fields = uncovered;
            uncovered -= cf_finder->second;
            if (!uncovered) continue;
            cf_finder->second |= uncovered;
            fields_narrowed = !!(fields - uncovered);
          }
          all_uncovered |= uncovered;

          dedup = false;
          if (fields_narrowed)
            new_insts.add_instance(InstanceRef(ref.get_manager(), uncovered));
          else
            new_insts.add_instance(ref);
        }

        if (!dedup)
        {
          summary_op.requirements.push_back(req);
          summary_op.parent_indices.push_back(parent_indices[idx]);
          RegionRequirement &req_copy = summary_op.requirements.back();
          req_copy.privilege = WRITE_DISCARD;
          if (!!(fields - all_uncovered))
          {
            req_copy.privilege_fields.clear();
            req_copy.instance_fields.clear();
            field_node->get_field_set(all_uncovered, req_copy.privilege_fields);
            field_node->get_field_set(all_uncovered, req_copy.instance_fields);
          }
#ifdef DEBUG_LEGION
          assert(new_insts.size() > 0);
#endif
          summary_op.instances.push_back(new_insts);
        }
        else
          summary_ops.pop_back();
      }
      std::reverse(summary_ops.begin(), summary_ops.end());
      dedup_summary_ops.swap(summary_ops);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline std::string PhysicalTemplate::view_to_string(
                                                       const InstanceView *view)
    //--------------------------------------------------------------------------
    {
      assert(view->logical_node->is_region());
      std::stringstream ss;
      LogicalRegion handle = view->logical_node->as_region_node()->handle;
      ss << "pointer: " << std::hex << view
         << ", instance: " << std::hex << view->get_manager()->get_instance().id
         << ", kind: "
         << (view->is_materialized_view() ? "   normal" : "reduction")
         << ", domain: "
         << view->get_manager()->instance_domain->handle.get_id()
         << ", region: " << "(" << handle.get_index_space().get_id()
         << "," << handle.get_field_space().get_id()
         << "," << handle.get_tree_id()
         << ")";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    /*static*/ inline std::string PhysicalTemplate::view_to_string(
                                                       const FillView *view)
    //--------------------------------------------------------------------------
    {
      assert(view->logical_node->is_region());
      std::stringstream ss;
      LogicalRegion handle = view->logical_node->as_region_node()->handle;
      ss << "pointer: " << std::hex << view
         << ", region: " << "(" << handle.get_index_space().get_id()
         << "," << handle.get_field_space().get_id()
         << "," << handle.get_tree_id()
         << ")";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::dump_template(void)
    //--------------------------------------------------------------------------
    {
      std::cerr << "#### " << (replayable ? "Replayable" : "Non-replayable")
                << " Template " << this << " ####" << std::endl;
      for (unsigned sidx = 0; sidx < replay_parallelism; ++sidx)
      {
        std::cerr << "[Slice " << sidx << "]" << std::endl;
        dump_instructions(slices[sidx]);
      }
      for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
           it != frontiers.end(); ++it)
        std::cerr << "  events[" << it->second << "] = events["
                  << it->first << "]" << std::endl;
      std::cerr << "[Previous Valid Views]" << std::endl;
      for (LegionMap<InstanceView*, FieldMask>::aligned::iterator it =
           previous_valid_views.begin(); it !=
           previous_valid_views.end(); ++it)
      {
        char *mask = it->second.to_string();
        std::cerr << "  " << view_to_string(it->first) << " " << mask
                  << " logical ctx: " << logical_contexts[it->first]
                  << " physical ctx: " << physical_contexts[it->first]
                  << std::endl;
        free(mask);
      }

      std::cerr << "[Previous Fill Views]" << std::endl;
      for (LegionMap<FillView*, FieldMask>::aligned::iterator it =
           untracked_fill_views.begin(); it != untracked_fill_views.end(); ++it)
      {
        char *mask = it->second.to_string();
        std::cerr << "  " << view_to_string(it->first) << " " << mask
                  << " " << std::endl;
        free(mask);
      }

      std::cerr << "[Valid Views]" << std::endl;
      for (LegionMap<InstanceView*, FieldMask>::aligned::iterator it =
           valid_views.begin(); it != valid_views.end(); ++it)
      {
        char *mask = it->second.to_string();
        std::cerr << "  " << view_to_string(it->first) << " " << mask
                  << " logical ctx: " << logical_contexts[it->first]
                  << " physical ctx: " << physical_contexts[it->first]
                  << std::endl;
        free(mask);
      }

      std::cerr << "[Pending Reductions]" << std::endl;
      for (LegionMap<InstanceView*, FieldMask>::aligned::iterator it =
           reduction_views.begin(); it != reduction_views.end(); ++it)
      {
        char *mask = it->second.to_string();
        std::cerr << "  " << view_to_string(it->first) << " " << mask
                  << " logical ctx: " << logical_contexts[it->first]
                  << " physical ctx: " << physical_contexts[it->first]
                  << std::endl;
        free(mask);
      }

      std::cerr << "[Fill Views]" << std::endl;
      for (LegionMap<FillView*, FieldMask>::aligned::iterator it =
           fill_views.begin(); it != fill_views.end(); ++it)
      {
        char *mask = it->second.to_string();
        std::cerr << "  " << view_to_string(it->first) << " " << mask
                  << " " << std::endl;
        free(mask);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::dump_instructions(
                                  const std::vector<Instruction*> &instructions)
    //--------------------------------------------------------------------------
    {
      for (std::vector<Instruction*>::const_iterator it = instructions.begin();
           it != instructions.end(); ++it)
        std::cerr << "  " << (*it)->to_string() << std::endl;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_mapper_output(SingleTask *task,
                                            const Mapper::MapTaskOutput &output,
                              const std::deque<InstanceSet> &physical_instances)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      TraceLocalID op_key = task->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(cached_mappings.find(op_key) == cached_mappings.end());
#endif
      CachedMapping &mapping = cached_mappings[op_key];
      mapping.target_procs = output.target_procs;
      mapping.chosen_variant = output.chosen_variant;
      mapping.task_priority = output.task_priority;
      mapping.postmap_task = output.postmap_task;
      mapping.physical_instances = physical_instances;
      for (std::deque<InstanceSet>::iterator it =
           mapping.physical_instances.begin(); it !=
           mapping.physical_instances.end(); ++it)
      {
        it->add_valid_references(MAPPING_ACQUIRE_REF);
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
    void PhysicalTemplate::record_get_term_event(ApEvent lhs, SingleTask* task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(task->is_memoizing());
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      unsigned lhs_ = events.size();
      events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;

      TraceLocalID key = task->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.find(key) == operations.end());
      assert(task_entries.find(key) == task_entries.end());
#endif
      operations[key] = task;
      task_entries[key] = instructions.size();

      instructions.push_back(new GetTermEvent(*this, lhs_, key));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_create_ap_user_event(
                                              ApUserEvent lhs, Operation *owner)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs.exists());
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      unsigned lhs_ = events.size();
      user_events.resize(events.size());
      events.push_back(lhs);
      user_events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;

      Memoizable *memoizable = owner->get_memoizable();
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      instructions.push_back(new CreateApUserEvent(*this, lhs_,
            memoizable->get_trace_local_id()));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
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
      std::map<ApEvent, unsigned>::iterator lhs_finder = event_map.find(lhs);
      std::map<ApEvent, unsigned>::iterator rhs_finder = event_map.find(rhs);
#ifdef DEBUG_LEGION
      assert(lhs_finder != event_map.end());
      assert(rhs_finder != event_map.end());
#endif
      unsigned lhs_ = lhs_finder->second;
      unsigned rhs_ = rhs_finder->second;
      instructions.push_back(new TriggerEvent(*this, lhs_, rhs_,
            instructions[lhs_]->owner));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs, ApEvent rhs_,
                                               Operation *owner)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> rhs;
      rhs.insert(rhs_);
      record_merge_events(lhs, rhs, owner);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs, ApEvent e1,
                                               ApEvent e2, Operation *owner)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> rhs;
      rhs.insert(e1);
      rhs.insert(e2);
      record_merge_events(lhs, rhs, owner);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs, ApEvent e1,
                                               ApEvent e2, ApEvent e3,
                                               Operation *owner)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> rhs;
      rhs.insert(e1);
      rhs.insert(e2);
      rhs.insert(e3);
      record_merge_events(lhs, rhs, owner);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs,
                                               const std::set<ApEvent>& rhs,
                                               Operation *owner)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      std::set<unsigned> rhs_;
      for (std::set<ApEvent>::const_iterator it = rhs.begin(); it != rhs.end();
           it++)
      {
        std::map<ApEvent, unsigned>::iterator finder = event_map.find(*it);
        if (finder != event_map.end() && finder->second != fence_completion_id)
          rhs_.insert(finder->second);
      }
      if (rhs_.size() == 0)
        rhs_.insert(fence_completion_id);

#ifndef LEGION_SPY
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

      unsigned lhs_ = events.size();
      events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;

      Memoizable *memoizable = owner->get_memoizable();
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      instructions.push_back(new MergeEvent(*this, lhs_, rhs_,
            memoizable->get_trace_local_id()));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_copy_views(InstanceView *src,
                                             const FieldMask &src_mask,
                                             ContextID src_logical_ctx,
                                             ContextID src_physical_ctx,
                                             InstanceView *dst,
                                             const FieldMask &dst_mask,
                                             ContextID dst_logical_ctx,
                                             ContextID dst_physical_ctx)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      update_valid_view(
          false, true, false, src, src_mask, src_logical_ctx, src_physical_ctx);
      update_valid_view(
          false, false, false, dst, dst_mask, dst_logical_ctx, dst_physical_ctx);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_copy(Operation* op, ApEvent &lhs,
                                             RegionNode *node,
                                 const std::vector<CopySrcDstField>& src_fields,
                                 const std::vector<CopySrcDstField>& dst_fields,
                                             ApEvent precondition,
                                             PredEvent predicate_guard,
                                             RegionTreeNode *intersect,
                                             ReductionOpID redop,
                                             bool reduction_fold)
    //--------------------------------------------------------------------------
    {
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

      unsigned lhs_ = events.size();
      events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;

      Memoizable *memoizable = op->get_memoizable();
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      TraceLocalID op_key = memoizable->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.find(op_key) != operations.end());
#endif

      std::map<ApEvent, unsigned>::iterator pre_finder =
        event_map.find(precondition);
#ifdef DEBUG_LEGION
      assert(pre_finder != event_map.end());
#endif

      for (unsigned idx = 0; idx < src_fields.size(); ++idx)
      {
        const CopySrcDstField &field = src_fields[idx];
        record_last_user(field.inst, node, field.field_id, lhs_, true);
      }
      for (unsigned idx = 0; idx < dst_fields.size(); ++idx)
      {
        const CopySrcDstField &field = dst_fields[idx];
        record_last_user(field.inst, node, field.field_id, lhs_, false);
      }

      unsigned precondition_idx = pre_finder->second;
      instructions.push_back(new IssueCopy(
            *this, lhs_, node, op_key, src_fields, dst_fields,
            precondition_idx, predicate_guard,
            intersect, redop, reduction_fold));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_empty_copy(CompositeView *src,
                                             const FieldMask &src_mask,
                                             MaterializedView *dst,
                                             const FieldMask &dst_mask,
                                             ContextID logical_ctx)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      // FIXME: Nested composite views potentially make the check expensive.
      //        Here we simply handle the case we know can be done efficiently.
      if (!src->has_nested_views())
      {
        bool already_valid = false;
        LegionMap<InstanceView*, FieldMask>::aligned::iterator finder =
          valid_views.find(dst);
        if (finder == valid_views.end())
          valid_views[dst] = dst_mask;
        else
        {
          already_valid = !(dst_mask - finder->second);
          finder->second |= dst_mask;
        }
        if (already_valid) return;

        src->closed_tree->record_closed_tree(src_mask, logical_ctx,
            previous_open_nodes, previous_projections);
      }
    }

    //--------------------------------------------------------------------------
    inline void PhysicalTemplate::update_valid_view(bool is_reduction,
                                                    bool has_read,
                                                    bool has_write,
                                                    InstanceView *view,
                                                    const FieldMask &fields,
                                                    ContextID logical_ctx,
                                                    ContextID physical_ctx)
    //--------------------------------------------------------------------------
    {
      if (is_reduction)
      {
#ifdef DEBUG_LEGION
        assert(view->is_reduction_view());
        assert(reduction_views.find(view) == reduction_views.end());
        assert(valid_views.find(view) == valid_views.end());
#endif
        reduction_views[view] = fields;
        valid_views[view] = fields;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(view->is_materialized_view() || view->is_reduction_view());
#endif
        if (has_read)
        {
          FieldMask invalid_fields = fields;

          LegionMap<InstanceView*, FieldMask>::aligned::iterator finder =
            valid_views.find(view);
          if (finder != valid_views.end())
            invalid_fields -= finder->second;

          if (!!invalid_fields && view->is_materialized_view())
            for (LegionMap<InstanceView*, FieldMask>::aligned::iterator vit =
                valid_views.begin(); vit != valid_views.end(); ++vit)
            {
              if (vit->first->get_manager() != view->get_manager()) continue;
              LogicalView *target = vit->first;
              LogicalView *parent = view->get_parent();
              while (parent != NULL)
              {
                if (parent == target)
                  invalid_fields -= vit->second;
                if (!invalid_fields)
                  break;
                parent = parent->get_parent();
              }
              if (!invalid_fields)
                break;
            }

          if (!!invalid_fields)
            previous_valid_views[view] |= invalid_fields;

          if (view->is_reduction_view())
          {
            LegionMap<InstanceView*, FieldMask>::aligned::iterator finder =
              reduction_views.find(view);
            if (finder != reduction_views.end())
            {
              finder->second -= fields;
              if (!finder->second)
                reduction_views.erase(finder);
            }
          }
        }

        if (has_write)
        {
          RegionTreeNode *node = view->logical_node;
          std::vector<InstanceView*> to_delete;
          for (LegionMap<InstanceView*, FieldMask>::aligned::iterator vit =
               valid_views.begin(); vit != valid_views.end(); ++vit)
          {
            if (vit->first->get_manager() == view->get_manager()) continue;
            RegionTreeNode *other = vit->first->logical_node;
            if (node->get_tree_id() != other->get_tree_id()) continue;
            if (!!(fields & vit->second) &&
                node->intersects_with(other, false))
            {
              vit->second = vit->second - fields;
              if (!vit->second)
                to_delete.push_back(vit->first);
            }
          }
          for (unsigned idx = 0; idx < to_delete.size(); ++idx)
            valid_views.erase(to_delete[idx]);
        }

        valid_views[view] |= fields;
      }
#ifdef DEBUG_LEGION
      assert(logical_contexts.find(view) == logical_contexts.end() ||
             logical_contexts[view] == logical_ctx);
      assert(physical_contexts.find(view) == physical_contexts.end() ||
             physical_contexts[view] == physical_ctx);
#endif
      logical_contexts[view] = logical_ctx;
      physical_contexts[view] = physical_ctx;
    }


    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_summary_info(const RegionRequirement &region,
                                               const InstanceSet &instance_set,
                                               unsigned parent_idx)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      summary_info.resize(summary_info.size() + 1);
      summary_info.back().first = region;
      summary_info.back().second = instance_set;
      parent_indices.push_back(parent_idx);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_set_ready_event(Operation *op,
                                                  unsigned region_idx,
                                                  unsigned inst_idx,
                                                  ApEvent &ready_event,
                                                  const RegionRequirement &req,
                                                  RegionNode *region_node,
                                                  InstanceView *view,
                                                  const FieldMask &fields,
                                                  ContextID logical_ctx,
                                                  ContextID physical_ctx)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      Memoizable *memoizable = op->get_memoizable();
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      TraceLocalID op_key = memoizable->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.find(op_key) != operations.end());
#endif

      if (view->is_reduction_view())
      {
        ReductionView *reduction_view = view->as_reduction_view();
        PhysicalManager *manager = reduction_view->get_manager();
        LayoutDescription *const layout = manager->layout;
        const ReductionOp *reduction_op =
          Runtime::get_reduction_op(reduction_view->get_redop());

        std::vector<CopySrcDstField> fields;
        {
          std::vector<FieldID> fill_fields;
          layout->get_fields(fill_fields);
          layout->compute_copy_offsets(fill_fields, manager, fields);
        }

        void *fill_buffer = malloc(reduction_op->sizeof_rhs);
        reduction_op->init(fill_buffer, 1);
#ifdef DEBUG_LEGION
        assert(view->logical_node->is_region());
#endif

        std::map<ApEvent, unsigned>::iterator ready_finder =
          event_map.find(ready_event);
#ifdef DEBUG_LEGION
        assert(ready_finder != event_map.end());
#endif
        unsigned ready_event_idx = ready_finder->second;

        ApUserEvent lhs = Runtime::create_ap_user_event();
        unsigned lhs_ = events.size();
        events.push_back(lhs);
        event_map[lhs] = lhs_;
        Runtime::trigger_event(lhs, ready_event);
        ready_event = lhs;

        instructions.push_back(
            new IssueFill(*this, lhs_, region_node,
                          op_key, fields, fill_buffer, reduction_op->sizeof_rhs,
                          ready_event_idx, PredEvent::NO_PRED_EVENT,
#ifdef LEGION_SPY
                          0,
#endif
                          NULL));
        for (unsigned idx = 0; idx < fields.size(); ++idx)
        {
          const CopySrcDstField &field = fields[idx];
          record_last_user(field.inst, region_node, field.field_id, lhs_, true);
        }
        free(fill_buffer);
      }

      update_valid_view(IS_REDUCE(req), HAS_READ(req), HAS_WRITE(req),
                        view, fields, logical_ctx, physical_ctx);

      std::map<TraceLocalID, unsigned>::iterator finder =
        task_entries.find(op_key);
#ifdef DEBUG_LEGION
      assert(finder != task_entries.end());
#endif
      InstanceReq inst_req;
      inst_req.instance = view->get_manager()->get_instance();
      inst_req.node = region_node;
      region_node->get_column_source()->get_field_set(fields, inst_req.fields);
      inst_req.read = IS_READ_ONLY(req);
      op_reqs[op_key].push_back(inst_req);
      for (std::vector<FieldID>::iterator it = inst_req.fields.begin(); it !=
           inst_req.fields.end(); ++it)
        record_last_user(inst_req.instance, region_node, *it, finder->second,
                         inst_req.read);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_get_op_term_event(ApEvent lhs, Operation* op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(op->is_memoizing());
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      unsigned lhs_ = events.size();
      events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;

#ifdef DEBUG_LEGION
      assert(op->get_memoizable() != NULL);
#endif
      TraceLocalID key = op->get_memoizable()->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.find(key) == operations.end());
      assert(task_entries.find(key) == task_entries.end());
#endif
      operations[key] = op;
      task_entries[key] = instructions.size();

      instructions.push_back(new GetOpTermEvent(*this, lhs_, key));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_set_op_sync_event(ApEvent &lhs, Operation* op)
    //--------------------------------------------------------------------------
    {
      if (!lhs.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        lhs = ApEvent(rename);
      }
#ifdef DEBUG_LEGION
      assert(op->is_memoizing());
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      unsigned lhs_ = events.size();
      events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;

#ifdef DEBUG_LEGION
      assert(op->get_memoizable() != NULL);
#endif
      TraceLocalID key = op->get_memoizable()->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.find(key) != operations.end());
      assert(task_entries.find(key) != task_entries.end());
#endif
      instructions.push_back(new SetOpSyncEvent(*this, lhs_, key));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_complete_replay(Operation* op, ApEvent rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(op->is_memoizing());
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      events.push_back(ApEvent());
#ifdef DEBUG_LEGION
      assert(op->get_memoizable() != NULL);
#endif
      TraceLocalID lhs_ = op->get_memoizable()->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(event_map.find(rhs) != event_map.end());
#endif
      unsigned rhs_ = event_map[rhs];

#ifdef DEBUG_LEGION
      assert(operations.find(lhs_) != operations.end());
      assert(task_entries.find(lhs_) != task_entries.end());
#endif
      instructions.push_back(new CompleteReplay(*this, lhs_, rhs_));

#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_fill(Operation *op, ApEvent &lhs,
                                             RegionNode *node,
                                     const std::vector<CopySrcDstField> &fields,
                                             const void *fill_buffer,
                                             size_t fill_size,
                                             ApEvent precondition,
                                             PredEvent predicate_guard,
#ifdef LEGION_SPY
                                             UniqueID fill_uid,
#endif
                                             RegionTreeNode *intersect)
    //--------------------------------------------------------------------------
    {
      if (!lhs.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        lhs = ApEvent(rename);
      }
#ifdef DEBUG_LEGION
      assert(op->is_memoizing());
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      unsigned lhs_ = events.size();
      events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;

      Memoizable *memoizable = op->get_memoizable();
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      TraceLocalID key = memoizable->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.find(key) != operations.end());
      assert(task_entries.find(key) != task_entries.end());
#endif

      std::map<ApEvent, unsigned>::iterator pre_finder =
        event_map.find(precondition);
#ifdef DEBUG_LEGION
      assert(pre_finder != event_map.end());
#endif
      unsigned precondition_idx = pre_finder->second;

      instructions.push_back(new IssueFill(*this, lhs_, node, key,
                             fields, fill_buffer, fill_size, precondition_idx,
                             predicate_guard,
#ifdef LEGION_SPY
                             fill_uid,
#endif
                             intersect));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_fill_view(
                                FillView *fill_view, const FieldMask &fill_mask)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(fill_views.find(fill_view) == fill_views.end());
      assert(is_recording());
#endif
      fill_views[fill_view] = fill_mask;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_deferred_copy_from_fill_view(
                                                            FillView *fill_view,
                                                         InstanceView* dst_view,
                                                     const FieldMask &copy_mask,
                                                          ContextID logical_ctx,
                                                         ContextID physical_ctx)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      LegionMap<FillView*, FieldMask>::aligned::iterator finder =
        fill_views.find(fill_view);
      if (finder == fill_views.end())
      {
        finder = untracked_fill_views.find(fill_view);
        if (finder == untracked_fill_views.end())
          untracked_fill_views[fill_view] = copy_mask;
        else
          finder->second |= copy_mask;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!(copy_mask - finder->second));
#endif
        LegionMap<InstanceView*, FieldMask>::aligned::iterator finder =
          valid_views.find(dst_view);
        if (finder == valid_views.end())
          valid_views[dst_view] = copy_mask;
        else
          finder->second |= copy_mask;

#ifdef DEBUG_LEGION
        assert(logical_contexts.find(dst_view) == logical_contexts.end() ||
               logical_contexts[dst_view] == logical_ctx);
        assert(physical_contexts.find(dst_view) == physical_contexts.end() ||
               physical_contexts[dst_view] == physical_ctx);
#endif
        logical_contexts[dst_view] = logical_ctx;
        physical_contexts[dst_view] = physical_ctx;
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_empty_copy_from_fill_view(
                                                         InstanceView* dst_view,
                                                     const FieldMask &copy_mask,
                                                          ContextID logical_ctx,
                                                         ContextID physical_ctx)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      LegionMap<InstanceView*, FieldMask>::aligned::iterator finder =
        valid_views.find(dst_view);
      if (finder == valid_views.end())
        valid_views[dst_view] = copy_mask;
      else
        finder->second |= copy_mask;

#ifdef DEBUG_LEGION
      assert(logical_contexts.find(dst_view) == logical_contexts.end() ||
          logical_contexts[dst_view] == logical_ctx);
      assert(physical_contexts.find(dst_view) == physical_contexts.end() ||
          physical_contexts[dst_view] == physical_ctx);
#endif
      logical_contexts[dst_view] = logical_ctx;
      physical_contexts[dst_view] = physical_ctx;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_outstanding_gc_event(
                                         InstanceView *view, ApEvent term_event)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      outstanding_gc_events[view].insert(term_event);
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalTemplate::defer_template_deletion(void)
    //--------------------------------------------------------------------------
    {
      ApEvent wait_on = get_completion_for_deletion();
      DeleteTemplateArgs args(this);
      return implicit_runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY,
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
    inline void PhysicalTemplate::record_last_user(const PhysicalInstance &inst,
                                                   RegionNode *node,
                                                   unsigned field,
                                                   unsigned user, bool read)
    //--------------------------------------------------------------------------
    {
      InstanceAccess key(inst, field);
      std::map<InstanceAccess, UserInfos>::iterator finder =
        last_users.find(key);
      if (finder == last_users.end())
      {
        UserInfos &infos = last_users[key];
        infos.push_back(UserInfo(read, user, node));
      }
      else
      {
        bool joined = false;
        for (UserInfos::iterator it = finder->second.begin();
             it != finder->second.end();)
        {
          if ((read && it->read) || !it->node->intersects_with(node, false))
          {
            if (it->node == node)
            {
#ifdef DEBUG_LEGION
              assert(!joined);
#endif
              it->users.insert(user);
              joined = true;
            }
            ++it;
          }
          else
            it = finder->second.erase(it);
        }
        if (!joined)
          finder->second.push_back(UserInfo(read, user, node));
      }
    }

    //--------------------------------------------------------------------------
    inline void PhysicalTemplate::find_last_users(const PhysicalInstance &inst,
                                                  RegionNode *node,
                                                  unsigned field,
                                                  std::set<unsigned> &users)
    //--------------------------------------------------------------------------
    {
      InstanceAccess key(inst, field);
      std::map<InstanceAccess, UserInfos>::iterator finder =
        last_users.find(key);
#ifdef DEBUG_LEGION
      assert(finder != last_users.end());
#endif
      for (UserInfos::iterator uit = finder->second.begin();
           uit != finder->second.end(); ++uit)
        for (std::set<unsigned>::iterator it = uit->users.begin(); it !=
             uit->users.end(); ++it)
          if (node->intersects_with(uit->node, false))
          {
#ifdef DEBUG_LEGION
            assert(frontiers.find(*it) != frontiers.end());
#endif
            users.insert(frontiers[*it]);
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

      SingleTask *task = dynamic_cast<SingleTask*>(operations[owner]);
      assert(task != NULL);
#else
      SingleTask *task = static_cast<SingleTask*>(operations[owner]);
#endif
      ApEvent completion_event = task->get_task_completion();
      events[lhs] = completion_event;
      task->replay_map_task_output();
    }

    //--------------------------------------------------------------------------
    std::string GetTermEvent::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = operations[" << owner 
         << "].get_task_termination()";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* GetTermEvent::clone(PhysicalTemplate& tpl,
                                  const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator finder = rewrite.find(lhs);
#ifdef DEBUG_LEGION
      assert(finder != rewrite.end());
#endif
      return new GetTermEvent(tpl, finder->second, owner);
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

    //--------------------------------------------------------------------------
    Instruction* CreateApUserEvent::clone(PhysicalTemplate& tpl,
                                    const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator lhs_finder =
        rewrite.find(lhs);
#ifdef DEBUG_LEGION
      assert(lhs_finder != rewrite.end());
#endif
      return new CreateApUserEvent(tpl, lhs_finder->second, owner);
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

    //--------------------------------------------------------------------------
    Instruction* TriggerEvent::clone(PhysicalTemplate& tpl,
                                    const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator lhs_finder =
        rewrite.find(lhs);
      std::map<unsigned, unsigned>::const_iterator rhs_finder =
        rewrite.find(rhs);
#ifdef DEBUG_LEGION
      assert(lhs_finder != rewrite.end());
      assert(rhs_finder != rewrite.end());
#endif
      return new TriggerEvent(tpl, lhs_finder->second, rhs_finder->second,
                              owner);
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
      ApEvent result = Runtime::merge_events(to_merge);
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

    //--------------------------------------------------------------------------
    Instruction* MergeEvent::clone(PhysicalTemplate& tpl,
                                   const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return NULL;
    }

    /////////////////////////////////////////////////////////////
    // AssignFenceCompletion
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AssignFenceCompletion::AssignFenceCompletion(
                       PhysicalTemplate& tpl, unsigned l, const TraceLocalID &o)
      : Instruction(tpl, o), fence_completion(tpl.fence_completion), lhs(l)
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
      events[lhs] = fence_completion;
    }

    //--------------------------------------------------------------------------
    std::string AssignFenceCompletion::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = fence_completion";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* AssignFenceCompletion::clone(PhysicalTemplate& tpl,
                                  const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator finder = rewrite.find(lhs);
#ifdef DEBUG_LEGION
      assert(finder != rewrite.end());
#endif
      return new AssignFenceCompletion(tpl, finder->second, owner);
    }

    /////////////////////////////////////////////////////////////
    // IssueCopy
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IssueCopy::IssueCopy(PhysicalTemplate& tpl,
                         unsigned l, RegionNode *n,
                         const TraceLocalID& key,
                         const std::vector<CopySrcDstField>& s,
                         const std::vector<CopySrcDstField>& d,
                         unsigned pi, PredEvent pg, RegionTreeNode *i,
                         ReductionOpID ro, bool rf)
      : Instruction(tpl, key), lhs(l), node(n), src_fields(s),
        dst_fields(d), precondition_idx(pi), predicate_guard(pg),
        intersect(i), redop(ro), reduction_fold(rf)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
      assert(operations.find(owner) != operations.end());
      assert(src_fields.size() > 0);
      assert(dst_fields.size() > 0);
      assert(precondition_idx < events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void IssueCopy::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      Operation *op = operations[owner];
      ApEvent precondition = events[precondition_idx];
      PhysicalTraceInfo trace_info;
      events[lhs] = node->issue_copy(op, src_fields, dst_fields, precondition,
          predicate_guard, trace_info, intersect, redop, reduction_fold);
    }

    //--------------------------------------------------------------------------
    std::string IssueCopy::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = copy(operations[" << owner << "], {";
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

    //--------------------------------------------------------------------------
    Instruction* IssueCopy::clone(PhysicalTemplate& tpl,
                                  const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator lfinder = rewrite.find(lhs);
      std::map<unsigned, unsigned>::const_iterator pfinder =
        rewrite.find(precondition_idx);
#ifdef DEBUG_LEGION
      assert(lfinder != rewrite.end());
      assert(pfinder != rewrite.end());
#endif
      return new IssueCopy(tpl, lfinder->second, node, owner, src_fields,
        dst_fields, pfinder->second, predicate_guard, intersect, redop,
        reduction_fold);
    }

    /////////////////////////////////////////////////////////////
    // IssueFill
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IssueFill::IssueFill(PhysicalTemplate& tpl, unsigned l, RegionNode *n,
                         const TraceLocalID &key,
                         const std::vector<CopySrcDstField> &f,
                         const void *fb, size_t fs, unsigned pi,
                         PredEvent pg,
#ifdef LEGION_SPY
                         UniqueID u,
#endif
                         RegionTreeNode *i)
      : Instruction(tpl, key), lhs(l), node(n), fields(f),
        precondition_idx(pi), predicate_guard(pg),
#ifdef LEGION_SPY
        fill_uid(u),
#endif
        intersect(i)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
      assert(operations.find(owner) != operations.end());
      assert(fields.size() > 0);
      assert(precondition_idx < events.size());
#endif
      fill_size = fs;
      fill_buffer = malloc(fs);
      memcpy(fill_buffer, fb, fs);
    }

    //--------------------------------------------------------------------------
    IssueFill::~IssueFill(void)
    //--------------------------------------------------------------------------
    {
      free(fill_buffer);
    }

    //--------------------------------------------------------------------------
    void IssueFill::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      Operation *op = operations[owner];
      ApEvent precondition = events[precondition_idx];

      PhysicalTraceInfo trace_info;
      events[lhs] = node->issue_fill(op, fields, fill_buffer,
          fill_size, precondition, predicate_guard,
#ifdef LEGION_SPY
          fill_uid,
#endif
          trace_info, intersect);
    }

    //--------------------------------------------------------------------------
    std::string IssueFill::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = fill({";
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

    //--------------------------------------------------------------------------
    Instruction* IssueFill::clone(PhysicalTemplate& tpl,
                                  const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator lfinder = rewrite.find(lhs);
      std::map<unsigned, unsigned>::const_iterator pfinder =
        rewrite.find(precondition_idx);
#ifdef DEBUG_LEGION
      assert(lfinder != rewrite.end());
      assert(pfinder != rewrite.end());
#endif
      return new IssueFill(tpl, lfinder->second, node, owner, fields,
          fill_buffer, fill_size, pfinder->second, predicate_guard,
#ifdef LEGION_SPY
          fill_uid,
#endif
          intersect);
    }

    /////////////////////////////////////////////////////////////
    // GetOpTermEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GetOpTermEvent::GetOpTermEvent(PhysicalTemplate& tpl, unsigned l,
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
    void GetOpTermEvent::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      events[lhs] = operations[owner]->get_completion_event();
    }

    //--------------------------------------------------------------------------
    std::string GetOpTermEvent::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = operations[" << owner
         << "].get_completion_event()    (op kind: "
         << Operation::op_names[operations[owner]->get_operation_kind()] << ")";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* GetOpTermEvent::clone(PhysicalTemplate& tpl,
                                    const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator finder = rewrite.find(lhs);
#ifdef DEBUG_LEGION
      assert(finder != rewrite.end());
#endif
      return new GetOpTermEvent(tpl, finder->second, owner);
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
      Memoizable *memoizable = operations[owner]->get_memoizable();
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      ApEvent sync_condition = memoizable->compute_sync_precondition();
      events[lhs] = sync_condition;
    }

    //--------------------------------------------------------------------------
    std::string SetOpSyncEvent::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = operations[" << owner
         << "].compute_sync_precondition()    (op kind: "
         << Operation::op_names[operations[owner]->get_operation_kind()] << ")";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* SetOpSyncEvent::clone(PhysicalTemplate& tpl,
                                    const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator finder = rewrite.find(lhs);
#ifdef DEBUG_LEGION
      assert(finder != rewrite.end());
#endif
      return new SetOpSyncEvent(tpl, finder->second, owner);
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
      Memoizable *memoizable = operations[owner]->get_memoizable();
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
         << Operation::op_names[operations[owner]->get_operation_kind()] << ")";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* CompleteReplay::clone(PhysicalTemplate& tpl,
                                    const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator finder = rewrite.find(rhs);
#ifdef DEBUG_LEGION
      assert(finder != rewrite.end());
#endif
      return new CompleteReplay(tpl, owner, finder->second);
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTraceInfo
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(void)
    //--------------------------------------------------------------------------
      : recording(false), op(NULL), tpl(NULL)
    {
    }

  }; // namespace Internal 
}; // namespace Legion

