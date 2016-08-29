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


#include "legion.h"
#include "legion_ops.h"
#include "legion_spy.h"
#include "legion_trace.h"
#include "legion_tasks.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // LegionTrace 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LegionTrace::LegionTrace(TraceID t, SingleTask *c)
      : tid(t), ctx(c), fixed(false), tracing(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LegionTrace::LegionTrace(const LegionTrace &rhs)
      : tid(0), ctx(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LegionTrace::~LegionTrace(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LegionTrace& LegionTrace::operator=(const LegionTrace &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void LegionTrace::fix_trace(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fixed);
#endif
      fixed = true;
    }

    //--------------------------------------------------------------------------
    void LegionTrace::end_trace_capture(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tracing);
#endif
      operations.clear();
      op_map.clear();
      close_dependences.clear();
      tracing = false;
#ifdef LEGION_SPY
      current_uids.clear();
      num_regions.clear();
#endif
    }

    //--------------------------------------------------------------------------
    void LegionTrace::end_trace_execution(Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!tracing);
#endif
      // Register for this fence on every one of the operations in
      // the trace and then clear out the operations data structure
      for (unsigned idx = 0; idx < operations.size(); idx++)
      {
        const std::pair<Operation*,GenerationID> &target = operations[idx];
        op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
        for (unsigned req_idx = 0; req_idx < num_regions[idx]; req_idx++)
        {
          LegionSpy::log_mapping_dependence(
              op->get_parent()->get_unique_op_id(), current_uids[idx], req_idx,
              op->get_unique_op_id(), 0, TRUE_DEPENDENCE);
        }
#endif
        // Remove any mapping references that we hold
        target.first->remove_mapping_reference(target.second);
      }
      operations.clear();
#ifdef LEGION_SPY
      current_uids.clear();
      num_regions.clear();
#endif
    }

    //--------------------------------------------------------------------------
    void LegionTrace::register_operation(Operation *op, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      std::pair<Operation*,GenerationID> key(op,gen);
      const unsigned index = operations.size();
      // Only need to save this in the map if we are not done tracing
      if (tracing)
      {
        // This is the normal case
        if (!op->is_close_op())
        {
          operations.push_back(key);
          op_map[key] = index;
          // Add a new vector for storing dependences onto the back
          dependences.push_back(LegionVector<DependenceRecord>::aligned());
          // Record meta-data about the trace for verifying that
          // it is being replayed correctly
          op_info.push_back(OperationInfo(op));
        }
        else // Otherwise, track close operations separately
          close_dependences[key] = LegionVector<DependenceRecord>::aligned();
      }
      else
      {
        if (!op->is_close_op())
        {
          // Check for exceeding the trace size
          if (index >= dependences.size())
          {
            log_run.error("Trace violation! Recorded %ld operations in trace "
                          "%d in task %s (UID %lld) but %d operations have "
                          "now been issued!", dependences.size(), tid,
                          ctx->get_task_name(), ctx->get_unique_id(), index+1);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_TRACE_VIOLATION);
          }
          // Check to see if the meta-data alignes
          const OperationInfo &info = op_info[index];
          // Check that they are the same kind of operation
          if (info.kind != op->get_operation_kind())
          {
            log_run.error("Trace violation! Operation at index %d of trace %d "
                          "in task %s (UID %lld) was recorded as having type "
                          "%s but instead has type %s in replay.",
                          index, tid, ctx->get_task_name(),ctx->get_unique_id(),
                          Operation::get_string_rep(info.kind),
                          Operation::get_string_rep(op->get_operation_kind()));
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_TRACE_VIOLATION);
          }
          // Check that they have the same number of region requirements
          if (info.count != op->get_region_count())
          {
            log_run.error("Trace violation! Operation at index %d of trace %d "
                          "in task %s (UID %lld) was recorded as having %d "
                          "regions, but instead has %ld regions in replay.",
                          index, tid, ctx->get_task_name(),
                          ctx->get_unique_id(), info.count,
                          op->get_region_count());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_TRACE_VIOLATION);
          }
          // If we make it here, everything is good
          const LegionVector<DependenceRecord>::aligned &deps = 
                                                          dependences[index];
          operations.push_back(key);
#ifdef LEGION_SPY
          current_uids.push_back(op->get_unique_op_id());
          num_regions.push_back(op->get_region_count());
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

            if ((it->prev_idx == -1) || (it->next_idx == -1))
            {
              op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
              LegionSpy::log_mapping_dependence(
                  op->get_parent()->get_unique_op_id(),
                  current_uids[it->operation_idx], 
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
                  op->get_parent()->get_unique_op_id(),
                  current_uids[it->operation_idx], it->prev_idx,
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
          // Special case for close operations
          // Close operations need to register transitive dependences
          // on all the other operations with which it interferes.
          // We can get this from the set of operations on which the
          // operation we are currently performing dependence analysis
          // has dependences.
          TraceCloseOp *close_op = static_cast<TraceCloseOp*>(op);
#ifdef DEBUG_LEGION
          assert(close_op == dynamic_cast<TraceCloseOp*>(op));
#endif
          int closing_index = close_op->get_close_index();
          for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
                deps.begin(); it != deps.end(); it++)
          {
            // We only record dependences for this close operation on
            // the indexes for which this close operation is being done
            if (closing_index != it->next_idx)
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
              close_op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
              LegionSpy::log_mapping_dependence(
                  op->get_parent()->get_unique_op_id(),
                  current_uids[it->operation_idx], 
                  (it->prev_idx == -1) ? 0 : it->prev_idx,
                  op->get_unique_op_id(), 
                  (it->next_idx == -1) ? 0 : it->next_idx, TRUE_DEPENDENCE);
#endif
            }
            else
            {
              close_op->record_trace_dependence(target.first, target.second,
                                                it->prev_idx, it->next_idx,
                                                it->dtype, it->dependent_mask);
#ifdef LEGION_SPY
              LegionSpy::log_mapping_dependence(
                  close_op->get_parent()->get_unique_op_id(),
                  current_uids[it->operation_idx], it->prev_idx,
                  close_op->get_unique_op_id(), 0, it->dtype);
#endif
            }
          }
          // Also see if we have any aliased region requirements that we
          // need to notify the generating task of
          std::map<unsigned,std::vector<std::pair<unsigned,unsigned> > >::
            const_iterator finder = alias_reqs.find(index-1);
          if (finder != alias_reqs.end())
          {
            Operation *create_op = operations.back().first;
            unsigned close_idx = close_op->get_close_index();
            for (std::vector<std::pair<unsigned,unsigned> >::const_iterator
                  it = finder->second.begin(); it != finder->second.end(); it++)
            {
              if (it->second == close_idx)
                create_op->report_interfering_close_requirement(it->first);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void LegionTrace::record_dependence(Operation *target, GenerationID tar_gen,
                                        Operation *source, GenerationID src_gen)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tracing);
      if (!source->is_close_op())
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
        if (!source->is_close_op())
        {
          // Normal case
          dependences.back().push_back(DependenceRecord(finder->second));
        }
        else
        {
          // Otherwise this is a close op so record it special
          // Don't record dependences on our creator
          if (target_key != operations.back())
          {
            std::pair<Operation*,GenerationID> src_key(source, src_gen);
#ifdef DEBUG_LEGION
            assert(close_dependences.find(src_key) != close_dependences.end());
#endif
            close_dependences[src_key].push_back(
                DependenceRecord(finder->second));
          }
        }
      }
      else if (target->is_close_op())
      {
        // They shouldn't both be close operations, if they are, then
        // they should be going through the other path that tracks
        // dependences based on specific region requirements
#ifdef DEBUG_LEGION
        assert(!source->is_close_op());
#endif
        // First check to see if the close op is one of ours
        std::map<std::pair<Operation*,GenerationID>,
                LegionVector<DependenceRecord>::aligned>::const_iterator
          close_finder = close_dependences.find(target_key);
        if (close_finder != close_dependences.end())
        {
          LegionVector<DependenceRecord>::aligned &target_deps = 
                                                        dependences.back();
          const LegionVector<DependenceRecord>::aligned &close_deps = 
                                                        close_finder->second;
          for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
                close_deps.begin(); it != close_deps.end(); it++)
          {
            target_deps.push_back(DependenceRecord(it->operation_idx)); 
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void LegionTrace::record_region_dependence(Operation *target, 
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
      if (!source->is_close_op())
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
        if (!source->is_close_op())
        {
          // Normal case
          dependences.back().push_back(
              DependenceRecord(finder->second, target_idx, source_idx,
                               validates, dtype, dep_mask));
        }
        else
        {
          // Otherwise this is a close op so record it special
          // Don't record dependences on our creator
          if (target_key != operations.back())
          { 
            std::pair<Operation*,GenerationID> src_key(source, src_gen);
#ifdef DEBUG_LEGION
            assert(close_dependences.find(src_key) != close_dependences.end());
#endif
            close_dependences[src_key].push_back(
                DependenceRecord(finder->second, target_idx, source_idx,
                                 validates, dtype, dep_mask));
          }
        }
      }
      else if (target->is_close_op())
      {
        // First check to see if the close op is one of ours
        std::map<std::pair<Operation*,GenerationID>,
                 LegionVector<DependenceRecord>::aligned>::const_iterator
          close_finder = close_dependences.find(target_key);
        if (close_finder != close_dependences.end())
        {
          // It is one of ours, so two cases
          if (!source->is_close_op())
          {
            // Iterate over the close operation dependences and 
            // translate them to our dependences
            for (LegionVector<DependenceRecord>::aligned::const_iterator
                  it = close_finder->second.begin(); 
                  it != close_finder->second.end(); it++)
            {
              FieldMask overlap = it->dependent_mask & dep_mask;
              if (!overlap)
                continue;
              dependences.back().push_back(
                  DependenceRecord(it->operation_idx, it->prev_idx,
                     source_idx, it->validates, it->dtype, overlap));
            }
          }
          else
          {
            // Iterate over the close operation dependences
            // and translate them to our dependences
            std::pair<Operation*,GenerationID> src_key(source, src_gen);
#ifdef DEBUG_LEGION
            assert(close_dependences.find(src_key) != close_dependences.end());
#endif
            LegionVector<DependenceRecord>::aligned &close_deps = 
                                                    close_dependences[src_key];
            for (LegionVector<DependenceRecord>::aligned::const_iterator
                  it = close_finder->second.begin(); 
                  it != close_finder->second.end(); it++)
            {
              FieldMask overlap = it->dependent_mask & dep_mask;
              if (!overlap)
                continue;
              close_deps.push_back(
                  DependenceRecord(it->operation_idx, it->prev_idx,
                    source_idx, it->validates, it->dtype, overlap));
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void LegionTrace::record_aliased_requirements(unsigned idx1, unsigned idx2)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx1 < idx2);
#endif
      unsigned index = operations.size() - 1;
      alias_reqs[index].push_back(std::pair<unsigned,unsigned>(idx1,idx2));
    }

    /////////////////////////////////////////////////////////////
    // TraceCaptureOp 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceCaptureOp::TraceCaptureOp(Runtime *rt)
      : Operation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceCaptureOp::TraceCaptureOp(const TraceCaptureOp &rhs)
      : Operation(NULL)
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
    void TraceCaptureOp::initialize_capture(SingleTask *ctx)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, true/*track*/);
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
      assert(trace != NULL);
#endif
      LegionTrace *local_trace = trace;
      // Now mark our trace as NULL to avoid registering this operation
      trace = NULL;
      tracing = false;
      begin_dependence_analysis();
      // Indicate that we are done capturing this trace
      local_trace->end_trace_capture();
      end_dependence_analysis();
    }

    /////////////////////////////////////////////////////////////
    // TraceCompleteOp 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceCompleteOp::TraceCompleteOp(Runtime *rt)
      : FenceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceCompleteOp::TraceCompleteOp(const TraceCompleteOp &rhs)
      : FenceOp(NULL)
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
    void TraceCompleteOp::initialize_complete(SingleTask *ctx)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, MIXED_FENCE);
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
      assert(trace != NULL);
#endif
      LegionTrace *local_trace = trace;
      // Now mark our trace as NULL to avoid registering this operation
      trace = NULL;
      begin_dependence_analysis();
      // Indicate that this trace is done being captured
      // This also registers that we have dependences on all operations
      // in the trace.
      local_trace->end_trace_execution(this);
      // Now update the parent context with this fence before we can complete
      // the dependence analysis and possibly be deactivated
      parent_ctx->update_current_fence(this);
      end_dependence_analysis();
    }
    
  }; // namespace Internal 
}; // namespace Legion

