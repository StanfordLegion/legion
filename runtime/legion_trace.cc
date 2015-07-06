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


#include "legion.h"
#include "legion_ops.h"
#include "legion_trace.h"
#include "legion_tasks.h"

namespace LegionRuntime {
  namespace HighLevel {

    // Extern declarations for loggers
    extern Logger::Category log_run;

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
#ifdef DEBUG_HIGH_LEVEL
      assert(!fixed);
#endif
      fixed = true;
    }

    //--------------------------------------------------------------------------
    void LegionTrace::end_trace_capture(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(tracing);
#endif
      operations.clear();
      op_map.clear();
      close_dependences.clear();
      tracing = false;
    }

    //--------------------------------------------------------------------------
    void LegionTrace::end_trace_execution(Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!tracing);
#endif
      // Register for this fence on every one of the operations in
      // the trace and then clear out the operations data structure
      for (unsigned idx = 0; idx < operations.size(); idx++)
      {
        const std::pair<Operation*,GenerationID> &target = operations[idx];
        op->register_dependence(target.first, target.second);
        // Remove any mapping references that we hold
        target.first->remove_mapping_reference(target.second);
      }
      operations.clear();
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
        }
        else // Otherwise, track close operations separately
          close_dependences[key] = LegionVector<DependenceRecord>::aligned();
      }
      else
      {
        if (!op->is_close_op())
        {
          const LegionVector<DependenceRecord>::aligned &deps = 
                                                          dependences[index];
          operations.push_back(key);
          // Add a mapping reference since people will be 
          // registering dependences
          op->add_mapping_reference(gen);  
          // Then compute all the dependences on this operation from
          // our previous recording of the trace
          
          for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
                deps.begin(); it != deps.end(); it++)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert((it->operation_idx >= 0) &&
                   ((size_t)it->operation_idx < operations.size()));
#endif
            const std::pair<Operation*,GenerationID> &target = 
                                                  operations[it->operation_idx];
            if ((it->prev_idx == -1) || (it->next_idx == -1))
              op->register_dependence(target.first, target.second);
            else
              op->register_region_dependence(it->next_idx, target.first,
                                             target.second, it->prev_idx,
                                             it->dtype, it->validates,
                                             it->dependent_mask);
          }
        }
        else
        {
          // We already added our creator to the list of operations
          // so the set of dependences is index-1
#ifdef DEBUG_HIGH_LEVEL
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
          InterCloseOp *close_op = static_cast<InterCloseOp*>(op);
#ifdef DEBUG_HIGH_LEVEL
          assert(close_op == dynamic_cast<InterCloseOp*>(op));
#endif
          for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
                deps.begin(); it != deps.end(); it++)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert((it->operation_idx >= 0) &&
                   ((size_t)it->operation_idx < operations.size()));
#endif
            const std::pair<Operation*,GenerationID> &target = 
                                                  operations[it->operation_idx];
            // If this is the case we can do the normal registration
            if ((it->prev_idx == -1) || (it->next_idx == -1))
              close_op->register_dependence(target.first, target.second);
            else
              close_op->record_trace_dependence(target.first, target.second,
                                                it->prev_idx, it->next_idx,
                                                it->dtype, it->dependent_mask);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void LegionTrace::record_dependence(Operation *target, GenerationID tar_gen,
                                        Operation *source, GenerationID src_gen)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(tracing);
      assert(operations.back().first == source);
      assert(operations.back().second == src_gen);
      assert(!source->is_close_op());
#endif
      std::pair<Operation*,GenerationID> target_key(target, tar_gen);
      std::map<std::pair<Operation*,GenerationID>,unsigned>::const_iterator
        finder = op_map.find(target_key);
      // We only need to record it if it falls within our trace
      if (finder != op_map.end())
        dependences.back().push_back(DependenceRecord(finder->second));
      else if (target->is_close_op())
      {
        // See if the target is a close operation on which 
        // we need to check for transitive dependences
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
        // First check to see if the close ops is one of ours
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
#ifdef DEBUG_HIGH_LEVEL
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
      initialize_operation(ctx, true/*track*/, Event::NO_EVENT);
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
    const char* TraceCaptureOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_CAPTURE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TraceCaptureOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return TRACE_CAPTURE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(trace != NULL);
#endif
      LegionTrace *local_trace = trace;
      // Now mark our trace as NULL to avoid registering this operation
      trace = NULL;
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
    const char* TraceCompleteOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_COMPLETE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TraceCompleteOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return TRACE_COMPLETE_OP_KIND; 
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
    
  }; // namespace HighLevel
}; // namespace LegionRuntime

