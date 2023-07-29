/* Copyright 2023 Stanford University, NVIDIA Corporation
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
#include "legion/legion_replication.h"

#include "realm/id.h" // TODO: remove this hackiness

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Utility functions
    /////////////////////////////////////////////////////////////

    std::ostream& operator<<(std::ostream &out, const TraceLocalID &key)
    {
      out << "(" << key.context_index << ",";
      if (key.index_point.dim > 1) out << "(";
      for (int dim = 0; dim < key.index_point.dim; ++dim)
      {
        if (dim > 0) out << ",";
        out << key.index_point[dim];
      }
      if (key.index_point.dim > 1) out << ")";
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
    // LogicalTrace 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalTrace::LogicalTrace(InnerContext *c, TraceID t, bool logical_only,
                               bool static_trace, Provenance *p,
                               const std::set<RegionTreeID> *trees)
      : context(c), tid(t), begin_provenance(p), end_provenance(NULL),
        blocking_call_observed(false),
        has_intermediate_ops(false), fixed(false),
        recording(true), trace_fence(NULL),
        static_translator(static_trace ? new StaticTranslator(trees) : NULL)
    //--------------------------------------------------------------------------
    {
      state.store(LOGICAL_ONLY);
      physical_trace = logical_only ? NULL : 
        new PhysicalTrace(c->owner_task->runtime, this);
      if (begin_provenance != NULL)
        begin_provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    LogicalTrace::~LogicalTrace(void)
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
    void LogicalTrace::fix_trace(Provenance *provenance)
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
    bool LogicalTrace::initialize_op_tracing(Operation *op,
                               const std::vector<StaticDependence> *dependences)
    //--------------------------------------------------------------------------
    {
      if (op->is_internal_op())
      {
        if (!recording)
          return false;
      }
      else
      {
        if (fixed)
          return false;
      }
      if (static_translator != NULL)
        static_translator->push_dependences(dependences);
      return true;
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::skip_analysis(RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      if (!recording)
        return true;
      if (static_translator == NULL)
        return false;
      return static_translator->skip_analysis(tid);
    }

    //--------------------------------------------------------------------------
    size_t LogicalTrace::register_operation(Operation *op, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      const std::pair<Operation*,GenerationID> key(op,gen);
#ifdef LEGION_SPY
      current_uids[key] = op->get_unique_op_id();
      num_regions[key] = op->get_region_count();
#endif
      if (recording)
      {
        // Recording
        if (op->is_internal_op())
          // We don't need to register internal operations
          return SIZE_MAX;
        const size_t index = replay_info.size();
        if (has_physical_trace() && op->get_memoizable() == NULL)
          REPORT_LEGION_ERROR(ERROR_PHYSICAL_TRACING_UNSUPPORTED_OP,
              "Invalid memoization request. Operation of type %s (UID %lld) "
              "at index %zd in trace %d requested memoization, but physical "
              "tracing does not support this operation type yet.",
              Operation::get_string_rep(op->get_operation_kind()),
              op->get_unique_op_id(), index, tid);
        const size_t op_index = operations.size();
        op_map[key] = op_index;
        operations.push_back(key);
        replay_info.push_back(OperationInfo(op));
        if (static_translator != NULL)
        {
          // Add a mapping reference since we might need to refer to it later
          op->add_mapping_reference(gen);
          // Recording a static trace so see if we have 
          // dependences to translate
          std::vector<StaticDependence> to_translate;
          static_translator->pop_dependences(to_translate);
          translate_dependence_records(op, op_index, to_translate);
        }
        return index;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!op->is_internal_op());
#endif
        // Replaying
        const size_t index = replay_index++;
        if (index >= replay_info.size())
          REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_RECORDED,
                        "Trace violation! Recorded %zd operations in trace "
                        "%d in task %s (UID %lld) but %zd operations have "
                        "now been issued!", replay_info.size(), tid,
                        context->get_task_name(), 
                        context->get_unique_id(), index+1)
        // Check to see if the meta-data alignes
        const OperationInfo &info = replay_info[index];
        // Check that they are the same kind of operation
        if (info.kind != op->get_operation_kind())
          REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                        "Trace violation! Operation at index %zd of trace %d "
                        "in task %s (UID %lld) was recorded as having type "
                        "%s but instead has type %s in replay.",
                        index, tid, context->get_task_name(),
                        context->get_unique_id(),
                        Operation::get_string_rep(info.kind),
                        Operation::get_string_rep(op->get_operation_kind()))
        // Check that they have the same number of region requirements
        if (info.region_count != op->get_region_count())
          REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                        "Trace violation! Operation at index %zd of trace %d "
                        "in task %s (UID %lld) was recorded as having %d "
                        "regions, but instead has %zd regions in replay.",
                        index, tid, context->get_task_name(),
                        context->get_unique_id(), info.region_count,
                        op->get_region_count())
        // Add a mapping reference since ops will be registering dependences
        op->add_mapping_reference(gen);
        operations.push_back(key);
        frontiers.insert(key);
        // First make any close operations needed for this operation and
        // register their dependences
        for (LegionVector<CloseInfo>::const_iterator cit = 
              info.closes.begin(); cit != info.closes.end(); cit++)
        {
#ifdef DEBUG_LEGION_COLLECTIVES
          MergeCloseOp *close_op = context->get_merge_close_op(op, cit->node);
#else
          MergeCloseOp *close_op = context->get_merge_close_op();
#endif
          close_op->initialize(context, cit->requirement, cit->creator_idx, op);
          close_op->update_close_mask(cit->close_mask);
          const GenerationID close_gen = close_op->get_generation();
          const std::pair<Operation*,GenerationID> close_key(close_op, 
                                                             close_gen);
          close_op->add_mapping_reference(close_gen);
          operations.push_back(close_key);
#ifdef LEGION_SPY
          current_uids[close_key] = close_op->get_unique_op_id();
          num_regions[close_key] = close_op->get_region_count();
#endif
          close_op->begin_dependence_analysis();
          close_op->trigger_dependence_analysis();
          replay_operation_dependences(close_op, cit->dependences);
          close_op->end_dependence_analysis();
        }
        // Then register the dependences for this operation
        if (!info.dependences.empty())
          replay_operation_dependences(op, info.dependences);
        else // need to at least record a dependence on the fence event
          op->register_dependence(trace_fence, trace_fence_gen);
        return index;
      }
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::register_internal(InternalOp *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
#endif
      const std::pair<Operation*,GenerationID> key(op, op->get_generation());
      const std::pair<Operation*,GenerationID> creator_key(
          op->get_creator_op(), op->get_creator_gen());
      std::map<std::pair<Operation*,GenerationID>,unsigned>::const_iterator
        finder = op_map.find(creator_key);
#ifdef DEBUG_LEGION
      assert(finder != op_map.end());
#endif
      // Record that they have the same entry so that we can detect that
      // they are the same when recording dependences. We do this for all
      // internal operations which won't be replayed and for which we will
      // need to collapse their dependences back onto their creator
      op_map[key] = finder->second;
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::register_close(MergeCloseOp *op, unsigned creator_idx,
#ifdef DEBUG_LEGION_COLLECTIVES
                                      RegionTreeNode *node,
#endif
                                      const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(!replay_info.empty());
#endif
      std::pair<Operation*,GenerationID> key(op, op->get_generation());
      const size_t index = operations.size();
      operations.push_back(key);
      op_map[key] = index;
      OperationInfo &info = replay_info.back();
      info.closes.emplace_back(CloseInfo(op, creator_idx,
#ifdef DEBUG_LEGION_COLLECTIVES
                                         node,
#endif
                                         req));
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::replay_operation_dependences(Operation *op,
                              const LegionVector<DependenceRecord> &dependences)
    //--------------------------------------------------------------------------
    {
      for (LegionVector<DependenceRecord>::const_iterator it =
            dependences.begin(); it != dependences.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->operation_idx >= 0);
        assert(((size_t)it->operation_idx) < operations.size());
        assert(it->dtype != LEGION_NO_DEPENDENCE);
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

    //--------------------------------------------------------------------------
    bool LogicalTrace::record_dependence(Operation *target,
            GenerationID target_gen, Operation *source, GenerationID source_gen)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
#endif
      const std::pair<Operation*,GenerationID> target_key(target, target_gen);
      std::map<std::pair<Operation*,GenerationID>,unsigned>::const_iterator
        target_finder = op_map.find(target_key);
      // The target is not part of the trace so there's no need to record it
      if (target_finder == op_map.end())
        return false;
      const std::pair<Operation*,GenerationID> source_key(source, source_gen);
      std::map<std::pair<Operation*,GenerationID>,unsigned>::const_iterator
        source_finder = op_map.find(source_key);
#ifdef DEBUG_LEGION
      assert(!replay_info.empty());
      assert(source_finder != op_map.end());
#endif
      // In the case of operations recording dependences on internal operations
      // such as refinement operations then we don't need to record those as
      // the refinement operations won't be in the replay
      if (source_finder->second == target_finder->second)
      {
#ifdef DEBUG_LEGION
        assert(target->get_operation_kind() == Operation::REFINEMENT_OP_KIND);
#endif
        return true;
      }
      OperationInfo &info = replay_info.back();
      DependenceRecord record(target_finder->second);
      if (source->get_operation_kind() == Operation::MERGE_CLOSE_OP_KIND)
      {
#ifdef DEBUG_LEGION
        bool found = false;
        assert(!info.closes.empty());
#endif
        // Find the right close info and record the dependence 
        for (unsigned idx = 0; idx < info.closes.size(); idx++)
        {
          CloseInfo &close = info.closes[idx];
          if (close.close_op != source)
            continue;
#ifdef DEBUG_LEGION
          found = true;
#endif
          for (LegionVector<DependenceRecord>::iterator it =
                close.dependences.begin(); it != close.dependences.end(); it++)
            if (it->merge(record))
              return true;
          close.dependences.emplace_back(std::move(record));
          break;
        }
#ifdef DEBUG_LEGION
        assert(found);
#endif
      }
      else
      {
        // Note that if the source is a non-close internal operation then
        // we also come through this pathway so that we record dependences
        // on anything that the operation records any transitive dependences
        // on things that its internal operations dependended on
        for (LegionVector<DependenceRecord>::iterator it =
              info.dependences.begin(); it != info.dependences.end(); it++)
          if (it->merge(record))
            return true;
        info.dependences.emplace_back(std::move(record));
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::record_region_dependence(Operation *target, 
                                                GenerationID target_gen,
                                                Operation *source, 
                                                GenerationID source_gen,
                                                unsigned target_idx, 
                                                unsigned source_idx,
                                                DependenceType dtype,
                                                bool validates,
                                                const FieldMask &dep_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
#endif
      const std::pair<Operation*,GenerationID> target_key(target, target_gen);
      std::map<std::pair<Operation*,GenerationID>,unsigned>::const_iterator
        target_finder = op_map.find(target_key);
      // The target is not part of the trace so there's no need to record it
      if (target_finder == op_map.end())
      {
        // If this is a close operation then we still need to update the mask
        if (source->get_operation_kind() == Operation::MERGE_CLOSE_OP_KIND)
        {
#ifdef DEBUG_LEGION
          assert(!replay_info.empty());
          assert(op_map.find(std::make_pair(source, source_gen)) != 
              op_map.end());
#endif
          OperationInfo &info = replay_info.back();
#ifdef DEBUG_LEGION
          bool found = false;
          assert(!info.closes.empty());
#endif
          // Find the right close info and record the dependence 
          for (unsigned idx = 0; idx < info.closes.size(); idx++)
          {
            CloseInfo &close = info.closes[idx];
            if (close.close_op != source)
              continue;
#ifdef DEBUG_LEGION
            found = true;
#endif
            close.close_mask |= dep_mask;
            break;
          }
#ifdef DEBUG_LEGION
          assert(found);
#endif
        }
        return false;
      }
      const std::pair<Operation*,GenerationID> source_key(source, source_gen);
      std::map<std::pair<Operation*,GenerationID>,unsigned>::const_iterator
        source_finder = op_map.find(source_key);
#ifdef DEBUG_LEGION
      assert(!replay_info.empty());
      assert(source_finder != op_map.end());
#endif
      // In the case of operations recording dependences on internal operations
      // such as refinement operations then we don't need to record those as
      // the refinement operations won't be in the replay
      if (source_finder->second == target_finder->second)
      {
#ifdef DEBUG_LEGION
        assert(target->get_operation_kind() == Operation::REFINEMENT_OP_KIND);
#endif
        return true;
      }
      OperationInfo &info = replay_info.back();
      DependenceRecord record(target_finder->second, target_idx, source_idx,
                              validates, dtype, dep_mask);
      if (source->get_operation_kind() == Operation::MERGE_CLOSE_OP_KIND)
      {
#ifdef DEBUG_LEGION
        bool found = false;
        assert(!info.closes.empty());
#endif
        // Find the right close info and record the dependence 
        for (unsigned idx = 0; idx < info.closes.size(); idx++)
        {
          CloseInfo &close = info.closes[idx];
          if (close.close_op != source)
            continue;
#ifdef DEBUG_LEGION
          found = true;
#endif
          close.close_mask |= dep_mask;
          for (LegionVector<DependenceRecord>::iterator it =
                close.dependences.begin(); it != close.dependences.end(); it++)
            if (it->merge(record))
              return true;
          close.dependences.emplace_back(std::move(record));
          break;
        }
#ifdef DEBUG_LEGION
        assert(found);
#endif
      }
      else
      {
        // Note that if the source is a non-close internal operation then
        // we also come through this pathway so that we record dependences
        // on anything that the operation records any transitive dependences
        // on things that its internal operations dependended on
        for (LegionVector<DependenceRecord>::iterator it =
              info.dependences.begin(); it != info.dependences.end(); it++)
          if (it->merge(record))
            return true;
        info.dependences.emplace_back(std::move(record));
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::begin_trace_execution(FenceOp *fence_op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace_fence == NULL);
#endif
      if (!recording)
      {
        trace_fence = fence_op;
        trace_fence_gen = fence_op->get_generation();
        fence_op->add_mapping_reference(trace_fence_gen);
        replay_index = 0;
      }
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::end_trace_execution(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      if (!recording)
      {
#ifdef DEBUG_LEGION
        assert(trace_fence != NULL);
#endif
        if (replay_index != replay_info.size())
          REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_RECORDED,
                        "Trace violation! Recorded %zd operations in trace "
                        "%d in task %s (UID %lld) but only %zd operations "
                        "have been issued at the end of the trace!", 
                        replay_info.size(), tid,
                        context->get_task_name(), 
                        context->get_unique_id(), replay_index)
        op->register_dependence(trace_fence, trace_fence_gen);
        trace_fence->remove_mapping_reference(trace_fence_gen);
        trace_fence = NULL;
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
                op->get_context()->get_unique_id(), current_uids[target],
                req_idx, op->get_unique_op_id(), 0, LEGION_TRUE_DEPENDENCE);
          }
#endif
          // Remove any mapping references that we hold
          target.first->remove_mapping_reference(target.second);
        }
      }
      else // Finished the recording so we are done
      {
        recording = false;
        op_map.clear();
        if (static_translator != NULL)
        {
#ifdef DEBUG_LEGION
          assert(static_translator->dependences.empty());
#endif
          delete static_translator;
          static_translator = NULL;
          // Also remove the mapping references from all the operations
          for (std::vector<std::pair<Operation*,GenerationID> >::const_iterator
                it = operations.begin(); it != operations.end(); it++)
            it->first->remove_mapping_reference(it->second);
          // Remove mapping fences on the frontiers which haven't been removed 
          for (std::set<std::pair<Operation*,GenerationID> >::const_iterator 
                it = frontiers.begin(); it != frontiers.end(); it++)
            it->first->remove_mapping_reference(it->second);
        }
      }
      operations.clear();
      if (physical_trace != NULL)
        physical_trace->reset_last_memoized();
      frontiers.clear();
#ifdef LEGION_SPY
      current_uids.clear();
      num_regions.clear();
#endif
    }

#ifdef LEGION_SPY
    //--------------------------------------------------------------------------
    UniqueID LogicalTrace::get_current_uid_by_index(unsigned op_idx) const
    //--------------------------------------------------------------------------
    {
      assert(op_idx < operations.size());
      const std::pair<Operation*,GenerationID> &key = operations[op_idx];
      std::map<std::pair<Operation*,GenerationID>,UniqueID>::const_iterator
        finder = current_uids.find(key);
      assert(finder != current_uids.end());
      return finder->second;
    }

#if 0
    //--------------------------------------------------------------------------
    void LogicalTrace::perform_logging(
                               UniqueID prev_fence_uid, UniqueID curr_fence_uid)
    //--------------------------------------------------------------------------
    {
      // This function is really a hack because we need to pretend like 
      // we performed the logical analysis for physical trace replays
      // when at the moment we currently don't, so this does the logging
      // the same as if we did perform the replay in program order, and 
      // even then we don't quite do this fully correctly since we don't 
      // have any close ops which will get the cross-shard analysis wrong
      UniqueID context_uid = context->get_unique_id();
      for (unsigned idx = 0; idx < operations.size(); idx++)
      {
        const UniqueID uid = get_current_uid_by_index(idx);
        if (idx == 0)
        {
          LegionSpy::log_mapping_dependence(context_uid, prev_fence_uid,
              0/*prev index*/, uid, 0/*next index*/, LEGION_TRUE_DEPENDENCE);
        }
        else
        {
          const UniqueID prev = get_current_uid_by_index(idx - 1);
          LegionSpy::log_mapping_dependence(context_uid, prev,
              0/*prev index*/, uid, 0/*next index*/, LEGION_TRUE_DEPENDENCE);
        }
      }
      if (!operations.empty())
      {
        const UniqueID prev = get_current_uid_by_index(operations.size() - 1);
        LegionSpy::log_mapping_dependence(context_uid, prev, 0, 
            curr_fence_uid, 0, LEGION_TRUE_DEPENDENCE);
      }
      else
        LegionSpy::log_mapping_dependence(context_uid, prev_fence_uid, 0,
            curr_fence_uid, 0, LEGION_TRUE_DEPENDENCE);
    }
#endif
#endif

    //--------------------------------------------------------------------------
    void LogicalTrace::invalidate_trace_cache(Operation *invalidator)
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
          current_template->issue_summary_operations(context, invalidator, 
                                                     end_provenance);
          has_intermediate_ops = false;
        }
      }
      else
        has_intermediate_ops = true;
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::translate_dependence_records(Operation *op,
         const unsigned index, const std::vector<StaticDependence> &dependences)
    //--------------------------------------------------------------------------
    {
      RegionTreeForest *forest = context->runtime->forest;
      const bool is_replicated = (context->get_replication_id() > 0);
      for (std::vector<StaticDependence>::const_iterator it =
            dependences.begin(); it != dependences.end(); it++)
      {
        if (it->dependence_type == LEGION_NO_DEPENDENCE)
          continue;
#ifdef DEBUG_LEGION
        assert(it->previous_offset <= index);
#endif
        const std::pair<Operation*,GenerationID> &prev =
            operations[index - it->previous_offset];
        unsigned parent_index = op->find_parent_index(it->current_req_index);
        LogicalRegion root_region = context->find_logical_region(parent_index);
        FieldSpaceNode *fs = forest->get_node(root_region.get_field_space());
        const FieldMask mask = fs->get_field_mask(it->dependent_fields);
        if (is_replicated && !it->shard_only)
        {
          // Need a merge close op to mediate the dependence
          RegionRequirement req(root_region, 
              LEGION_READ_WRITE, LEGION_EXCLUSIVE, root_region);
          req.privilege_fields = it->dependent_fields;
#ifdef DEBUG_LEGION_COLLECTIVES
          MergeCloseOp *close_op = context->get_merge_close_op(op,
                                    forest->get_node(root_region));
#else
          MergeCloseOp *close_op = context->get_merge_close_op();
#endif
          close_op->initialize(context, req, it->current_req_index, op);
          close_op->update_close_mask(mask);
          register_close(close_op, it->current_req_index,
#ifdef DEBUG_LEGION_COLLECTIVES
                         forest->get_node(root_region),
#endif
                         req);
          // Mark that we are starting our dependence analysis
          close_op->begin_dependence_analysis();
          // Do any other work for the dependence analysis
          close_op->trigger_dependence_analysis();
          // Record the dependence of the close on the previous op
          close_op->register_region_dependence(0/*close index*/,
              prev.first, prev.second, it->previous_req_index,
              LEGION_TRUE_DEPENDENCE, false/*validates*/, mask);
          // Then record our dependence on the close operation
          op->register_region_dependence(it->current_req_index,
              close_op, close_op->get_generation(), 0/*close index*/,
              LEGION_TRUE_DEPENDENCE, false/*validates*/, mask);
          // Dispatch this close op
          close_op->end_dependence_analysis();
        }
        else
        {
          // Can just record a normal dependence
          op->register_region_dependence(it->current_req_index,
              prev.first, prev.second, it->previous_req_index,
              it->dependence_type, it->validates, mask);
        }
      }
    }

#if 0
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
                               const std::vector<StaticDependence> *dependences)
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
    size_t StaticTrace::register_operation(Operation *op, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      std::pair<Operation*,GenerationID> key(op,gen);
      const size_t index = operations.size();
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
      return index;
    }

    //--------------------------------------------------------------------------
    bool StaticTrace::record_dependence(
                                     Operation *target, GenerationID target_gen,
                                     Operation *source, GenerationID source_gen)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }
    
    //--------------------------------------------------------------------------
    bool StaticTrace::record_region_dependence(
                                    Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    DependenceType dtype, bool validates,
                                    const FieldMask &dependent_mask)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
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
      if (physical_trace != NULL)
        physical_trace->reset_last_memoized();
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
                it->previous_req_index, it->current_req_index, it->validates, 
                it->dependence_type, dependence_mask));
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
      if (physical_trace != NULL)
        physical_trace->reset_last_memoized();
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
                               const std::vector<StaticDependence> *dependences)
    //--------------------------------------------------------------------------
    {
      return !is_fixed();
    }

    //--------------------------------------------------------------------------
    size_t DynamicTrace::register_operation(Operation *op, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      std::pair<Operation*,GenerationID> key(op,gen);
      const size_t index = operations.size();
      if (has_physical_trace() &&
          !op->is_internal_op() && op->get_memoizable() == NULL)
        REPORT_LEGION_ERROR(ERROR_PHYSICAL_TRACING_UNSUPPORTED_OP,
            "Invalid memoization request. Operation of type %s (UID %lld) "
            "at index %zd in trace %d requested memoization, but physical "
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
                          "%d in task %s (UID %lld) but %zd operations have "
                          "now been issued!", dependences.size(), tid,
                          ctx->get_task_name(), ctx->get_unique_id(), index+1)
          // Check to see if the meta-data alignes
          const OperationInfo &info = op_info[index];
          // Check that they are the same kind of operation
          if (info.kind != op->get_operation_kind())
            REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                          "Trace violation! Operation at index %zd of trace %d "
                          "in task %s (UID %lld) was recorded as having type "
                          "%s but instead has type %s in replay.",
                          index, tid, ctx->get_task_name(),ctx->get_unique_id(),
                          Operation::get_string_rep(info.kind),
                          Operation::get_string_rep(op->get_operation_kind()))
          // Check that they have the same number of region requirements
          if (info.count != op->get_region_count())
            REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                          "Trace violation! Operation at index %zd of trace %d "
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
      return index;
    }

#if 0
    //--------------------------------------------------------------------------
    bool DynamicTrace::record_dependence(Operation *target,GenerationID tar_gen,
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
          insert_dependence(
              DependenceRecord(finder->second, target_idx, source_idx,
                   validates, dtype, dep_mask));
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
              insert_dependence(
                  DependenceRecord(it->operation_idx, it->prev_idx,
                     source_idx, it->validates, it->dtype, overlap));
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
#endif

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
#endif

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
                                  bool remove_trace_ref, Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, EXECUTION_FENCE, false/*need future*/, provenance);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
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
      TraceOp::activate();
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      TraceOp::deactivate(false/*free*/);
      if (freeop)
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
      // Indicate that we are done capturing this trace
      trace->end_trace_execution(this);
      // Register this fence with all previous users in the parent's context
      FenceOp::trigger_dependence_analysis();
      parent_ctx->record_previous_trace(trace);
      if (trace->is_recording())
      {
        PhysicalTrace *physical_trace = trace->get_physical_trace();
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
        PhysicalTrace *physical_trace = trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(physical_trace != NULL);
        assert(current_template != NULL);
        assert(current_template->is_recording());
#endif
        current_template->finalize(parent_ctx, unique_op_id, has_blocking_call);
        if (!current_template->is_replayable())
        {
          physical_trace->record_failed_capture(current_template);
          ApEvent pending_deletion;
          if (!current_template->defer_template_deletion(pending_deletion,
                                                  map_applied_conditions))
            delete current_template;
          if (pending_deletion.exists())
            execution_preconditions.insert(pending_deletion);
        }
        else
        {
          ApEvent pending_deletion = physical_trace->record_replayable_capture(
                                      current_template, map_applied_conditions);
          if (pending_deletion.exists())
            execution_preconditions.insert(pending_deletion);
        }
        // Reset the local trace
        trace->initialize_tracing_state();
      }
      if (remove_trace_reference && trace->remove_reference())
        delete trace;
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
                                              Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, EXECUTION_FENCE, false/*need future*/, provenance);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
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
      TraceOp::activate();
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      TraceOp::deactivate(false/*free*/);
      if (freeop)
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
      trace->end_trace_execution(this);
      parent_ctx->record_previous_trace(trace);

      if (trace->is_replaying())
      {
        if (has_blocking_call)
          REPORT_LEGION_ERROR(ERROR_INVALID_PHYSICAL_TRACING,
            "Physical tracing violation! Trace %d in task %s (UID %lld) "
            "encountered a blocking API call that was unseen when it was "
            "recorded. It is required that traces do not change their "
            "behavior.", trace->get_trace_id(),
            parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        PhysicalTrace *physical_trace = trace->get_physical_trace();
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
        physical_trace->record_previous_template_completion(
                                      get_completion_event());
        trace->initialize_tracing_state();
        replayed = true;
        return;
      }
      else if (trace->is_recording())
      {
        PhysicalTrace *physical_trace = trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(physical_trace != NULL);
#endif
        physical_trace->record_previous_template_completion(
                                      get_completion_event());
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
        assert(trace->get_physical_trace() != NULL);
        assert(current_template->is_recording());
#endif
        current_template->finalize(parent_ctx, unique_op_id, has_blocking_call);
        PhysicalTrace *physical_trace = trace->get_physical_trace();
        if (!current_template->is_replayable())
        {
          physical_trace->record_failed_capture(current_template);
          ApEvent pending_deletion;
          if (!current_template->defer_template_deletion(pending_deletion,
                                                  map_applied_conditions))
            delete current_template;
          if (pending_deletion.exists())
            execution_preconditions.insert(pending_deletion);
        }
        else
        {
          ApEvent pending_deletion = physical_trace->record_replayable_capture(
                                      current_template, map_applied_conditions);
          if (pending_deletion.exists())
            execution_preconditions.insert(pending_deletion);
        }
        trace->initialize_tracing_state();
      }
      else if (replayed)
      {
#ifdef DEBUG_LEGION
        assert(current_template != NULL);
#endif
        std::set<ApEvent> template_postconditions;
        current_template->finish_replay(template_postconditions);
        complete_mapping();
        record_completion_effects(template_postconditions);
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
    void TraceReplayOp::initialize_replay(InnerContext *ctx, 
                                       LogicalTrace *tr, Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, EXECUTION_FENCE, false/*need future*/, provenance);
      trace = tr;
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::activate(void)
    //--------------------------------------------------------------------------
    {
      TraceOp::activate();
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      TraceOp::deactivate(false/*free*/);
      if (freeop)
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
      PhysicalTrace *physical_trace = trace->get_physical_trace();
#ifdef DEBUG_LEGION
      assert(physical_trace != NULL);
#endif
      bool recurrent = true;
      bool fence_registered = false;
      bool is_recording = trace->is_recording();
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
        assert(!(trace->is_recording() || trace->is_replaying()));
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
          !trace->has_intermediate_operations()) ?
            physical_trace->get_previous_template_completion()
                    : get_completion_event();
        if (recurrent && trace->has_intermediate_operations())
        {
          parent_ctx->perform_fence_analysis(this, execution_preconditions,
                                       true/*mapping*/, true/*execution*/);
          trace->reset_intermediate_operations();
        }
        if (!fence_registered)
          execution_preconditions.insert(
              parent_ctx->get_current_execution_fence_event());
        physical_trace->initialize_template(fence_completion, recurrent);
        trace->set_state_replay();
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
    void TraceBeginOp::initialize_begin(InnerContext *ctx, LogicalTrace *tr,
                                        Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, MAPPING_FENCE, false/*need future*/, provenance);
      trace = tr;
      tracing = false;
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::activate(void)
    //--------------------------------------------------------------------------
    {
      TraceOp::activate();
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      TraceOp::deactivate(false/*free*/);
      if (freeop)
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

    //--------------------------------------------------------------------------
    void TraceBeginOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      trace->begin_trace_execution(this);
      TraceOp::trigger_dependence_analysis();
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
      context_index = invalidator->get_ctx_index();
      if (runtime->legion_spy_enabled)
        LegionSpy::log_fence_operation(parent_ctx->get_unique_id(),
            unique_op_id, context_index, false/*execution fence*/);
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
      TraceOp::activate();
      current_template = NULL;
    }

    //--------------------------------------------------------------------------
    void TraceSummaryOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      TraceOp::deactivate(false/*free*/);
      if (freeop)
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
    PhysicalTrace::PhysicalTrace(Runtime *rt, LogicalTrace *lt)
      : runtime(rt), logical_trace(lt), perform_fence_elision(
          !(runtime->no_trace_optimization || runtime->no_fence_elision)),
        repl_ctx(dynamic_cast<ReplicateContext*>(lt->context)),
        previous_replay(NULL), current_template(NULL), nonreplayable_count(0),
        new_template_count(0), last_memoized(0),
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
    ApEvent PhysicalTrace::record_replayable_capture(PhysicalTemplate *tpl,
                                      std::set<RtEvent> &map_applied_conditions)
    //--------------------------------------------------------------------------
    {
      ApEvent pending_deletion;
      // See if we're going to exceed the maximum number of templates
      if (templates.size() == logical_trace->context->get_max_trace_templates())
      {
#ifdef DEBUG_LEGION
        assert(!templates.empty());
#endif
        PhysicalTemplate *to_delete = templates.front();
        if (!to_delete->defer_template_deletion(pending_deletion, 
                                                map_applied_conditions))
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
        InnerContext *ctx = logical_trace->context;
        REPORT_LEGION_WARNING(LEGION_WARNING_NEW_TEMPLATE_COUNT_EXCEEDED,
            "WARNING: The runtime has created %d new replayable templates "
            "for trace %u in task %s (UID %lld) without replaying any "
            "existing templates. This may mean that your mapper is not "
            "making mapper decisions conducive to replaying templates. Please "
            "check that your mapper is making decisions that align with prior "
            "templates. If you believe that this number of templates is "
            "reasonable please adjust the settings for "
            "LEGION_NEW_TEMPLATE_WARNING_COUNT in legion_config.h.",
            LEGION_NEW_TEMPLATE_WARNING_COUNT, logical_trace->get_trace_id(),
            ctx->get_task_name(), ctx->get_unique_id())
        new_template_count = 0;
      }
      // Reset the nonreplayable count when we find a replayable template
      nonreplayable_count = 0;
      current_template = NULL;
      return pending_deletion;
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::record_failed_capture(PhysicalTemplate *tpl)
    //--------------------------------------------------------------------------
    {
      if ((last_memoized > 0) && 
          (++nonreplayable_count > LEGION_NON_REPLAYABLE_WARNING))
      {
        const std::string &message = tpl->get_replayable_message();
        const char *message_buffer = message.c_str();
        InnerContext *ctx = logical_trace->context;
        REPORT_LEGION_WARNING(LEGION_WARNING_NON_REPLAYABLE_COUNT_EXCEEDED,
            "WARNING: The runtime has failed to memoize the trace more than "
            "%u times, due to the absence of a replayable template. It is "
            "highly likely that trace %u in task %s (UID %lld) will not be "
            "memoized for the rest of execution. The most recent template was "
            "not replayable for the following reason: %s. Please change the "
            "mapper to stop making memoization requests.",
            LEGION_NON_REPLAYABLE_WARNING, logical_trace->get_trace_id(),
            ctx->get_task_name(), ctx->get_unique_id(), message_buffer)
        nonreplayable_count = 0;
      }
      current_template = NULL;
    }

    //--------------------------------------------------------------------------
    bool PhysicalTrace::check_memoize_consensus(size_t index)
    //--------------------------------------------------------------------------
    {
      if (index == last_memoized)
      {
        last_memoized = index + 1;
        return true;
      }
      else
      {
        last_memoized = 0;
        return false;
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::reset_last_memoized(void)
    //--------------------------------------------------------------------------
    {
      last_memoized = 0;
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
    bool PhysicalTrace::find_viable_templates(ReplTraceReplayOp *op,
                                             std::set<RtEvent> &applied_events,
                                             unsigned templates_to_find,
                                             std::vector<int> &viable_templates)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(templates_to_find > 0);
#endif
      for (int index = viable_templates.empty() ? templates.size() - 1 : 
            viable_templates.back() - 1; index >= 0; index--)
      {
        PhysicalTemplate *tpl = templates[index];
        if (tpl->check_preconditions(op, applied_events))
        {
          // A good tmplate so add it to the list
          viable_templates.push_back(index);
          // If we've found all our templates then we're done
          if (--templates_to_find == 0)
            return (index == 0); // whether we are done
        }
      }
      return true; // Iterated over all the templates
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::select_template(unsigned index)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index < templates.size());
      assert(templates[index]->is_replayable());
#endif
      // Reset the nonreplayable count when a replayable template satisfies
      // the precondition
      nonreplayable_count = 0;
      // Also reset the new template count as we found a replay
      new_template_count = 0;
      current_template = templates[index]; 
      // Move this one to the back of the line since we all agreed to replay it
      // This way the most recently used on is the one at the end of the vector
      if (index < (templates.size() - 1))
        std::rotate(templates.begin()+index, 
                    templates.begin()+index+1, templates.end());
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate* PhysicalTrace::start_new_template(
                                              TaskTreeCoordinates &&coordinates)
    //--------------------------------------------------------------------------
    {
      // If we have a replicated context then we are making sharded templates
      if (repl_ctx != NULL)
        current_template = new ShardedPhysicalTemplate(this, 
            execution_fence_event, std::move(coordinates), repl_ctx);
      else
        current_template = new PhysicalTemplate(this, execution_fence_event,
                                                std::move(coordinates));
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
      std::stringstream ss;
      char *m = mask.to_string();
      if (view->is_fill_view())
      {
        ss << "fill view: " << view
           << ", Index expr: " << expr->expr_id
           << ", Field Mask: " << m;
      }
      else if (view->is_collective_view())
      {
        ss << "collective view: " << view
           << ", Index expr: " << expr->expr_id
           << ", Field Mask: " << m;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(view->is_individual_view());
#endif
        const char *mem_names[] = {
#define MEM_NAMES(name, desc) #name,
            REALM_MEMORY_KINDS(MEM_NAMES) 
#undef MEM_NAMES
          };
        PhysicalManager *manager =
          view->as_individual_view()->get_manager();
        FieldSpaceNode *field_space = manager->field_space_node;
        Memory memory = manager->memory_manager->memory;

        std::vector<FieldID> fields;
        field_space->get_field_set(mask, ctx, fields);

        ss << "view: " << view << " in " << mem_names[memory.kind()]
           << " memory " << std::hex << memory.id << std::dec
           << ", Index expr: " << expr->expr_id
           << ", Field Mask: " << m << ", Fields: ";
        for (std::vector<FieldID>::const_iterator it =
              fields.begin(); it != fields.end(); it++)
        {
          if (it != fields.begin())
            ss << ", ";
          const void *name = NULL;
          size_t name_size = 0;
          if (field_space->retrieve_semantic_information(
                LEGION_NAME_SEMANTIC_TAG, name, name_size,
                true/*can fail*/, false/*wait until*/))
            ss << ((const char*)name) << " (" << *it << ")";
          else
            ss << *it;
        }
      }
      return ss.str();
    }

    //--------------------------------------------------------------------------
    TraceViewSet::TraceViewSet(InnerContext *ctx, DistributedID own_did, 
                               IndexSpaceExpression *expr, RegionTreeID tid)
      : context(ctx), expression(expr), tree_id(tid), owner_did(
          (own_did > 0) ? own_did : ctx->did), has_collective_views(false)
    //--------------------------------------------------------------------------
    {
      expression->add_nested_expression_reference(owner_did);
      if (owner_did == ctx->did)
        context->add_base_resource_ref(TRACE_REF);
      else
        context->add_nested_resource_ref(owner_did);
    }

    //--------------------------------------------------------------------------
    TraceViewSet::~TraceViewSet(void)
    //--------------------------------------------------------------------------
    {
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); vit++)
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
          if (it->first->remove_nested_expression_reference(owner_did))
            delete it->first;
        if (vit->first->remove_nested_valid_ref(owner_did))
          delete vit->first;
      }
      if (owner_did == context->did)
      {
        if (context->remove_base_resource_ref(TRACE_REF))
          delete context;
      }
      else
      {
        if (context->remove_nested_resource_ref(owner_did))
          delete context;
      }
      if (expression->remove_nested_expression_reference(owner_did))
        delete expression;
      conditions.clear();
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::insert(LogicalView *view, IndexSpaceExpression *expr, 
                              const FieldMask &mask, bool antialiased)
    //--------------------------------------------------------------------------
    {
      ViewExprs::iterator finder = conditions.find(view);
      IndexSpaceExpression *const total_expr = expression; 
      const size_t expr_volume = expr->get_volume();
      if (expr != total_expr)
      {
#ifdef DEBUG_LEGION
        // This is a necessary but not sufficient condition for dominance
        // If we need to we can put in the full intersection test later
        assert(expr_volume <= total_expr->get_volume());
#endif
        // Recognize total expressions when they get here
        if (expr_volume == total_expr->get_volume())
          expr = total_expr;
      }
      // We need to enforce the invariant that there is at most one 
      // expression for field in this function
      if (finder != conditions.end())
      {
        FieldMask set_overlap = mask & finder->second.get_valid_mask();
        if (!!set_overlap)
        {
          if (set_overlap != mask)
          {
            // Handle the difference fields first before we mutate set_overlap
            FieldMask diff = mask - set_overlap;
            if (finder->second.insert(expr, mask))
              expr->add_nested_expression_reference(owner_did);
          }
          FieldMaskSet<IndexSpaceExpression> to_add;
          std::vector<IndexSpaceExpression*> to_delete;
          RegionTreeForest *forest = context->runtime->forest;
          for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                finder->second.begin(); it != finder->second.end(); it++)
          {
            const FieldMask overlap = set_overlap & it->second;
            if (!overlap)
              continue;
            if (it->first != total_expr)
            {
              if (it->first != expr)
              {
                // Not the same expression, so compute the union
                IndexSpaceExpression *union_expr = 
                  forest->union_index_spaces(it->first, expr);
                const size_t union_volume = union_expr->get_volume();
                if (it->first->get_volume() < union_volume)
                {
                  if (expr_volume < union_volume)
                    to_add.insert(union_expr, overlap);
                  else
                    to_add.insert(expr, overlap);
                  it.filter(overlap);
                  if (!it->second)
                    to_delete.push_back(it->first);
                }
                else
                  it.merge(overlap);
              }
              else
                it.merge(overlap);
            }
            set_overlap -= overlap;
            if (!set_overlap)
              break;
          }
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                to_add.begin(); it != to_add.end(); it++)
            if (finder->second.insert(it->first, it->second))
              it->first->add_nested_expression_reference(owner_did);
          for (std::vector<IndexSpaceExpression*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            if (to_add.find(*it) != to_add.end())
              continue;
            finder->second.erase(*it);
            if ((*it)->remove_nested_expression_reference(owner_did))
              delete (*it);
          }
        }
        else if (finder->second.insert(expr, mask))
          expr->add_nested_expression_reference(owner_did);
      }
      else
      {
        if (!antialiased)
        {
          if (view->is_collective_view())
          {
            FieldMaskSet<InstanceView> antialiased_views;
            antialias_collective_view(view->as_collective_view(), mask, 
                                      antialiased_views);
            // Now we can insert all the antialiased 
            for (FieldMaskSet<InstanceView>::const_iterator it =
                 antialiased_views.begin(); it != antialiased_views.end(); it++)
              insert(it->first, expr, it->second, true/*antialiased*/);
            return;
          }
          else if (has_collective_views && view->is_instance_view())
            antialias_individual_view(view->as_individual_view(), mask);
        }
        view->add_nested_valid_ref(owner_did);
        expr->add_nested_expression_reference(owner_did);
        conditions[view].insert(expr, mask);
        if (view->is_collective_view())
          has_collective_views = true;
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::invalidate(
       LogicalView *view, IndexSpaceExpression *expr, const FieldMask &mask,
       std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove,
       std::map<LogicalView*,unsigned> *view_refs_to_remove, bool antialiased)
    //--------------------------------------------------------------------------
    {
      ViewExprs::iterator finder = conditions.find(view);
      if ((finder == conditions.end()) || 
          (finder->second.get_valid_mask() * mask))
      {
        if (!antialiased)
        {
          if (view->is_collective_view())
          {
            FieldMaskSet<InstanceView> antialiased_views;
            antialias_collective_view(view->as_collective_view(), mask, 
                                      antialiased_views);
            // Now we can insert all the antialiased 
            for (FieldMaskSet<InstanceView>::const_iterator it =
                 antialiased_views.begin(); it != antialiased_views.end(); it++)
              invalidate(it->first, expr, it->second, expr_refs_to_remove,
                  view_refs_to_remove, true/*antialiased*/);
          }
          else if (has_collective_views && view->is_instance_view())
          {
            antialias_individual_view(view->as_individual_view(), mask);
            invalidate(view, expr, mask, expr_refs_to_remove, 
                view_refs_to_remove, true/*antialiased*/);
          }
        }
        return;
      }
      const size_t expr_volume = expr->get_volume();
      IndexSpaceExpression *const total_expr = expression; 
#ifdef DEBUG_LEGION
      // This is a necessary but not sufficient condition for dominance
      // If we need to we can put in the full intersection test later
      assert(expr_volume <= total_expr->get_volume());
#endif
      if ((expr == total_expr) || (expr_volume == total_expr->get_volume()))
      {
        // Expr covers the whole instance so no need to do intersections
        if (!(finder->second.get_valid_mask() - mask))
        {
          // Dominate all fields so just filter everything
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                finder->second.begin(); it != finder->second.end(); it++)
          {
            if (expr_refs_to_remove != NULL)
            {
              std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                expr_refs_to_remove->find(it->first);
              if (finder == expr_refs_to_remove->end())
                (*expr_refs_to_remove)[it->first] = 1;
              else
                finder->second += 1;
            }
            else if (it->first->remove_nested_expression_reference(owner_did))
              delete it->first;
          }
          if (view_refs_to_remove != NULL)
          {
            std::map<LogicalView*,unsigned>::iterator finder = 
              view_refs_to_remove->find(view);
            if (finder == view_refs_to_remove->end())
              (*view_refs_to_remove)[view] = 1;
            else
              finder->second += 1;
          }
          else if (view->remove_nested_valid_ref(owner_did))
            delete view;
          conditions.erase(finder);
        }
        else
        {
          // Filter on fields
          std::vector<IndexSpaceExpression*> to_delete;
          for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                finder->second.begin(); it != finder->second.end(); it++)
          {
            it.filter(mask);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          for (std::vector<IndexSpaceExpression*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            finder->second.erase(*it);
            if (expr_refs_to_remove != NULL)
            {
              std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                expr_refs_to_remove->find(*it);
              if (finder == expr_refs_to_remove->end())
                (*expr_refs_to_remove)[*it] = 1;
              else
                finder->second += 1;
            }
            else if ((*it)->remove_nested_expression_reference(owner_did))
              delete (*it);
          }
          if (finder->second.empty())
          {
            if (view_refs_to_remove != NULL)
            {
              std::map<LogicalView*,unsigned>::iterator finder = 
                view_refs_to_remove->find(view);
              if (finder == view_refs_to_remove->end())
                (*view_refs_to_remove)[view] = 1;
              else
                finder->second += 1;
            }
            else if (view->remove_nested_valid_ref(owner_did))
              delete view;
            conditions.erase(finder);
          }
          else
            finder->second.tighten_valid_mask();
        }
      }
      else
      {
        // We need intersection tests as part of filtering
        FieldMaskSet<IndexSpaceExpression> to_add;
        std::vector<IndexSpaceExpression*> to_delete;
        RegionTreeForest *forest = context->runtime->forest;
        for (FieldMaskSet<IndexSpaceExpression>::iterator it =
              finder->second.begin(); it != finder->second.end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          IndexSpaceExpression *intersection = expr;
          if (it->first != total_expr)
          {
            intersection = forest->intersect_index_spaces(it->first, expr);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            if (volume == expr_volume)
              intersection = expr;
            else if (volume == it->first->get_volume())
              intersection = it->first;
          }
          if (intersection->get_volume() < it->first->get_volume())
          {
            // Only dominated part of it so compute the difference
            IndexSpaceExpression *diff = 
              forest->subtract_index_spaces(it->first, intersection);
            to_add.insert(diff, overlap);
          }
          // No matter what we're removing these fields for this expr
          it.filter(overlap);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              to_add.begin(); it != to_add.end(); it++)
          if (finder->second.insert(it->first, it->second))
            it->first->add_nested_expression_reference(owner_did);
        for (std::vector<IndexSpaceExpression*>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
        {
          if (to_add.find(*it) != to_add.end())
            continue;
          finder->second.erase(*it);
          if (expr_refs_to_remove != NULL)
          {
            std::map<IndexSpaceExpression*,unsigned>::iterator finder =
              expr_refs_to_remove->find(*it);
            if (finder == expr_refs_to_remove->end())
              (*expr_refs_to_remove)[*it] = 1;
            else
              finder->second += 1;
          }
          else if ((*it)->remove_nested_expression_reference(owner_did))
            delete (*it);
        }
        if (finder->second.empty())
        {
          if (view_refs_to_remove != NULL)
          {
            std::map<LogicalView*,unsigned>::iterator finder = 
              view_refs_to_remove->find(view);
            if (finder == view_refs_to_remove->end())
              (*view_refs_to_remove)[view] = 1;
            else
              finder->second += 1;
          }
          else if (view->remove_nested_valid_ref(owner_did))
            delete view;
          conditions.erase(finder);
        }
        else
          finder->second.tighten_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::invalidate_all_but(LogicalView *except,
                              IndexSpaceExpression *expr, const FieldMask &mask,
         std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove,
         std::map<LogicalView*,unsigned> *view_refs_to_remove, bool antialiased)
    //--------------------------------------------------------------------------
    {
      if (!antialiased && (except != NULL))
      {
        if (except->is_collective_view())
        {
          FieldMaskSet<InstanceView> antialiased_views;
          antialias_collective_view(except->as_collective_view(), mask, 
                                    antialiased_views);
          // Now we can insert all the antialiased 
          for (FieldMaskSet<InstanceView>::const_iterator it =
               antialiased_views.begin(); it != antialiased_views.end(); it++)
            invalidate_all_but(it->first, expr, it->second, expr_refs_to_remove,
                view_refs_to_remove, true/*antialiased*/);
          return;
        }
        else if (has_collective_views && except->is_instance_view())
          antialias_individual_view(except->as_individual_view(), mask);
      }
      std::vector<LogicalView*> to_invalidate;
      for (ViewExprs::const_iterator it = 
            conditions.begin(); it != conditions.end(); it++)
      {
        if (it->first == except)
          continue;
        if (it->second.get_valid_mask() * mask)
          continue;
        to_invalidate.push_back(it->first);
      }
      for (std::vector<LogicalView*>::const_iterator it = 
            to_invalidate.begin(); it != to_invalidate.end(); it++)
        invalidate(*it, expr, mask, expr_refs_to_remove, 
                   view_refs_to_remove, true/*antialiased*/);
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::dominates(LogicalView *view,
                     IndexSpaceExpression *expr, FieldMask &non_dominated) const
    //--------------------------------------------------------------------------
    {
      // If this is for an empty equivalence set then it doesn't matter
      if (expr->is_empty())
        return true;
      const size_t expr_volume = expr->get_volume();
      IndexSpaceExpression *const total_expr = expression;
#ifdef DEBUG_LEGION
      // This is a necessary but not sufficient condition for dominance
      // If we need to we can put in the full intersection test later
      assert(expr_volume <= total_expr->get_volume());
#endif
      if (expr_volume == total_expr->get_volume())
        expr = total_expr;
      RegionTreeForest *forest = context->runtime->forest;
      ViewExprs::const_iterator finder = conditions.find(view);
      if (finder == conditions.end())
      {
        // If we couldn't find it directly then we need to deal with aliasing
        if (view->is_collective_view())
        {
          CollectiveAntiAlias alias_analysis(view->as_collective_view());
          for (ViewExprs::const_iterator vit =
                conditions.begin(); vit != conditions.end(); vit++)
          {
            if (!vit->first->is_instance_view())
              continue;
            if (vit->second.get_valid_mask() * non_dominated)
              continue;
            InstanceView *inst_view = vit->first->as_instance_view();
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  vit->second.begin(); it != vit->second.end(); it++)
            {
              const FieldMask overlap = it->second & non_dominated;
              if (!overlap)
                continue;
              alias_analysis.traverse(inst_view, overlap, it->first);
            }
          }
          FieldMask dominated = non_dominated;
          FieldMaskSet<IndexSpaceExpression> empty_exprs;
          alias_analysis.visit_leaves(non_dominated, dominated,
                                      expr, forest, empty_exprs);
          if (!!dominated)
            non_dominated -= dominated;
        }
        else if (has_collective_views && view->is_instance_view())
        {
          IndividualView *individual_view = view->as_individual_view();
          for (ViewExprs::const_iterator vit =
                conditions.begin(); vit != conditions.end(); vit++)
          {
            if (!vit->first->is_collective_view())
              continue;
            if (vit->second.get_valid_mask() * non_dominated)
              continue;
            if (!individual_view->aliases(vit->first->as_collective_view()))
              continue;
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  vit->second.begin(); it != vit->second.end(); it++)
            {
              const FieldMask overlap = non_dominated & it->second;
              if (!overlap)
                continue;
              if ((it->first != total_expr) && (it->first != expr))
              {
                IndexSpaceExpression *intersection = 
                  forest->intersect_index_spaces(it->first, expr);
                const size_t volume = intersection->get_volume();
                if (volume == 0)
                  continue;
                // Can only dominate if we have enough points
                if (volume < expr->get_volume())
                  continue;
              }
              // If we get here we were dominated
              non_dominated -= overlap;
              if (!non_dominated)
                break;
            }
          }
        }
      } 
      else
      {
        if ((expr == total_expr) || (expr_volume == total_expr->get_volume()))
        {
          // Expression is for the whole view, so will only be dominated
          // by the expression for the full view
          FieldMaskSet<IndexSpaceExpression>::const_iterator expr_finder =
            finder->second.find(total_expr);
          if (expr_finder != finder->second.end())
          {
            non_dominated -= expr_finder->second;
            if (!non_dominated)
              return true;
          }
        }
        // There is at most one expression per field so just iterate and compare
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              finder->second.begin(); it != finder->second.end(); it++)
        {
          const FieldMask overlap = non_dominated & it->second;
          if (!overlap)
            continue;
          if ((it->first != total_expr) && (it->first != expr))
          {
            IndexSpaceExpression *intersection = 
              forest->intersect_index_spaces(it->first, expr);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            // Can only dominate if we have enough points
            if (volume < expr->get_volume())
              continue;
          }
          // If we get here we were dominated
          non_dominated -= overlap;
          if (!non_dominated)
            break;
        }
      }
      // If there are no fields left then we dominated
      return !non_dominated;
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::dominates(LogicalView *view, 
                            IndexSpaceExpression *expr, FieldMask mask,
                            FieldMaskSet<IndexSpaceExpression> &non_dominated,
                            FieldMaskSet<IndexSpaceExpression> *dominated) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(non_dominated.empty());
#endif
      // If this is for an empty equivalence set then it doesn't matter
      if (expr->is_empty())
      {
        if (dominated != NULL)
          dominated->insert(expr, mask);
        return;
      }
      const size_t expr_volume = expr->get_volume();
      IndexSpaceExpression *const total_expr = expression;
#ifdef DEBUG_LEGION
      // This is a necessary but not sufficient condition for dominance
      // If we need to we can put in the full intersection test later
      assert(expr_volume <= total_expr->get_volume());
#endif
      if (expr_volume == total_expr->get_volume())
        expr = total_expr;
      RegionTreeForest *forest = context->runtime->forest;
      ViewExprs::const_iterator finder = conditions.find(view);
      if (finder == conditions.end())
      {
        if (view->is_collective_view())
        {
          CollectiveAntiAlias alias_analysis(view->as_collective_view());
          for (ViewExprs::const_iterator vit =
                conditions.begin(); vit != conditions.end(); vit++)
          {
            if (!vit->first->is_instance_view())
              continue;
            if (vit->second.get_valid_mask() * mask)
              continue;
            InstanceView *inst_view = vit->first->as_instance_view();
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  vit->second.begin(); it != vit->second.end(); it++)
            {
              const FieldMask overlap = it->second & mask;
              if (!overlap)
                continue;
              alias_analysis.traverse(inst_view, overlap, it->first);
            }
          } 
          FieldMask dominated_mask = mask;
          alias_analysis.visit_leaves(mask, dominated_mask,
                                      non_dominated, expr, forest);
          // Group the expressions across fields so there is exactly
          // one non-dominated expression for each field
          if (!non_dominated.empty())
          {
            LegionList<FieldSet<IndexSpaceExpression*> > field_sets;
            non_dominated.compute_field_sets(FieldMask(), field_sets);
            non_dominated.clear();
            for (LegionList<FieldSet<IndexSpaceExpression*> >::const_iterator 
                  it = field_sets.begin(); it != field_sets.end(); it++)
            {
#ifdef DEBUG_LEGION
              assert(!it->elements.empty());
#endif
              IndexSpaceExpression *non_dominated_expr =
                (it->elements.size() == 1) ? *(it->elements.begin()) :
                forest->union_index_spaces(it->elements);
              non_dominated.insert(non_dominated_expr, it->set_mask);
            }
          }
          if (!!dominated_mask && (dominated != NULL))
            dominated->insert(expr, dominated_mask);
        }
        else if (has_collective_views && view->is_instance_view())
        {
          IndividualView *individual_view = view->as_individual_view();
          for (ViewExprs::const_iterator vit =
                conditions.begin(); vit != conditions.end(); vit++)
          {
            if (!vit->first->is_collective_view())
              continue;
            if (vit->second.get_valid_mask() * mask)
              continue;
            if (!individual_view->aliases(vit->first->as_collective_view()))
              continue;
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  vit->second.begin(); it != vit->second.end(); it++)
            {
              const FieldMask overlap = mask & it->second;
              if (!overlap)
                continue;
              if ((it->first != total_expr) && (it->first != expr))
              {
                IndexSpaceExpression *difference = 
                  forest->subtract_index_spaces(expr, it->first);
                if (!difference->is_empty())
                  non_dominated.insert(difference, overlap);
                else if (dominated != NULL)
                  dominated->insert(expr, overlap);
              }
              // If we get here we were dominated
              else if (dominated != NULL)
                dominated->insert(expr, overlap);
            }
          }
        }
        // If we get here then these fields are definitely not dominated
#ifdef DEBUG_LEGION
        assert(!!mask);
#endif
        non_dominated.insert(expr, mask);
      }
      else if (finder->second.get_valid_mask() * mask)
        non_dominated.insert(expr, mask);
      else
      {
        if ((expr == total_expr) || (expr_volume == total_expr->get_volume()))
        {
          // Expression is for the whole view, so will only be dominated
          // for the full view
          FieldMaskSet<IndexSpaceExpression>::const_iterator expr_finder =
            finder->second.find(total_expr);
          if (expr_finder != finder->second.end())
          {
            const FieldMask overlap = mask & expr_finder->second;
            if (!!overlap)
            {
              if (dominated != NULL)
                dominated->insert(expr, overlap); 
              mask -= overlap;
              if (!mask)
                return;
            }
          }
        }
        // There is at most one expression per field so just iterate and compare
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              finder->second.begin(); it != finder->second.end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          if ((it->first != total_expr) && (it->first != expr))
          {
            IndexSpaceExpression *intersection = 
              forest->intersect_index_spaces(it->first, expr);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            // Can only dominate if we have enough points
            if (volume < expr->get_volume())
            {
              if (dominated != NULL)
                dominated->insert(intersection, overlap);
              IndexSpaceExpression *diff = 
                forest->subtract_index_spaces(expr, intersection);
              non_dominated.insert(diff, overlap);
            }
            else if (dominated != NULL)
              dominated->insert(expr, overlap);
          } // total expr dominates everything
          else if (dominated != NULL)
            dominated->insert(expr, overlap);
          mask -= overlap;
          if (!mask)
            return;
        }
        // If we get here then these fields are definitely not dominated
#ifdef DEBUG_LEGION
        assert(!!mask);
#endif
        non_dominated.insert(expr, mask);
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::filter_independent_fields(IndexSpaceExpression *expr,
                                                 FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      FieldMask independent = mask;
      RegionTreeForest *forest = context->runtime->forest;
      for (ViewExprs::const_iterator vit =
            conditions.begin(); vit != conditions.end(); vit++)
      {
        if (independent * vit->second.get_valid_mask())
          continue;
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
        {
          const FieldMask overlap = it->second & independent;
          if (!overlap)
            continue;
          IndexSpaceExpression *overlap_expr = 
            forest->intersect_index_spaces(it->first, expr);
          if (!overlap_expr->is_empty())
          {
            independent -= overlap;
            if (!independent)
              break;
          }
        }
        if (!independent)
          break;
      }
      if (!!independent)
        mask -= independent;
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::subsumed_by(const TraceViewSet &set, 
                    bool allow_independent, FailedPrecondition *condition) const
    //--------------------------------------------------------------------------
    {
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); ++vit)
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); ++it)
        {
          if (allow_independent)
          {
            // If we're allowing independent views, that means the set
            // does not need to dominate the view as long as there are no
            // views in the set that overlap logically with the test view
            // This allows us to handle the read-only precondition case
            // where we have read-only views that show up in the preconditions
            // but do not appear logically anywhere in the postconditions
            FieldMaskSet<IndexSpaceExpression> non_dominated;
            set.dominates(vit->first, it->first, it->second, non_dominated);
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator nit =
                  non_dominated.begin(); nit != non_dominated.end(); nit++)
            {
              // If all the fields are independent from anything that was
              // written in the postcondition then we know this is a
              // read-only precondition that does not need to be subsumed
              FieldMask mask = nit->second;
              set.filter_independent_fields(nit->first, mask);
              if (!mask)
                continue;
              if (condition != NULL)
              {
                condition->view = vit->first;
                condition->expr = nit->first;
                condition->mask = mask;
              }
              return false;
            }
          }
          else
          {
            FieldMask mask = it->second;
            if (!set.dominates(vit->first, it->first, mask))
            {
              if (condition != NULL)
              {
                condition->view = vit->first;
                condition->expr = it->first;
                condition->mask = mask;
              }
              return false;
            }
          }
        }

      return true;
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::independent_of(const TraceViewSet &set,
                                      FailedPrecondition *condition) const
    //--------------------------------------------------------------------------
    {
      if (conditions.size() > set.conditions.size())
        return set.independent_of(*this, condition);
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); ++vit)
      {
        ViewExprs::const_iterator finder = set.conditions.find(vit->first);
        if (finder == set.conditions.end())
        {
          if (vit->first->is_collective_view())
          {
            CollectiveView *collective = vit->first->as_collective_view();
            for (ViewExprs::const_iterator sit = 
                  set.conditions.begin(); sit != set.conditions.end(); sit++)
            {
              if (!sit->first->is_instance_view())
                continue;
              if (vit->second.get_valid_mask() * sit->second.get_valid_mask())
                continue;
              if (!collective->aliases(sit->first->as_instance_view()))
                continue;
              if (has_overlapping_expressions(collective, vit->second, 
                                              sit->second, condition))
                return false;
            }
          }
          else if (set.has_collective_views && vit->first->is_instance_view())
          {
            IndividualView *view = vit->first->as_individual_view();
            for (ViewExprs::const_iterator sit =
                  set.conditions.begin(); sit != set.conditions.end(); sit++)
            {
              if (!sit->first->is_collective_view())
                continue;
              if (vit->second.get_valid_mask() * sit->second.get_valid_mask())
                continue;
              if (!view->aliases(sit->first->as_collective_view()))
                continue;
              if (has_overlapping_expressions(view, vit->second, 
                                              sit->second, condition))
                return false;
            }
          }
          continue;
        }
        if (vit->second.get_valid_mask() * finder->second.get_valid_mask())
          continue;
        if (has_overlapping_expressions(vit->first, vit->second, 
                                        finder->second, condition))
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::has_overlapping_expressions(LogicalView *view,
        const FieldMaskSet<IndexSpaceExpression> &left_exprs,
        const FieldMaskSet<IndexSpaceExpression> &right_exprs,
        FailedPrecondition *condition) const
    //--------------------------------------------------------------------------
    {
      LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
                FieldMask> overlaps;
      unique_join_on_field_mask_sets(left_exprs, right_exprs, overlaps);
      RegionTreeForest *forest = context->runtime->forest;
      for (LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
                     FieldMask>::const_iterator it = 
            overlaps.begin(); it != overlaps.end(); it++)
      {
        IndexSpaceExpression *overlap = 
          forest->intersect_index_spaces(it->first.first, it->first.second);
        if (!overlap->is_empty())
        {
          if (condition != NULL)
          {
            condition->view = view;
            condition->expr = overlap;
            condition->mask = it->second;
          }
          return true;
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::record_first_failed(FailedPrecondition *condition) const
    //--------------------------------------------------------------------------
    {
      ViewExprs::const_iterator vit = conditions.begin();
      FieldMaskSet<IndexSpaceExpression>::const_iterator it =
        vit->second.begin();
      condition->view = vit->first;
      condition->expr = it->first;
      condition->mask = it->second;
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::transpose_uniquely(
            LegionMap<IndexSpaceExpression*,FieldMaskSet<LogicalView> > &target,
            std::set<IndexSpaceExpression*> &unique_exprs) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target.empty());
#endif
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); ++vit)
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
        {
          target[it->first].insert(vit->first, it->second);
          // Track the unique expressions
          if (unique_exprs.insert(it->first).second)
            it->first->add_base_expression_reference(TRACE_REF);
        }
      if (target.size() == 1)
        return;
      // Now for the hard part, we need to compare any expresions that overlap
      // and have overlapping fields so we can uniquify them, this reduces the
      // number of analyses in the precondition/anticondition cases, and is 
      // necessary for correctness in the postcondition case where we cannot
      // have multiple overwrites for the same fields and index expressions
      FieldMaskSet<IndexSpaceExpression> expr_fields;
      LegionMap<IndexSpaceExpression*,
                FieldMaskSet<LogicalView> > intermediate;
      intermediate.swap(target);
      for (LegionMap<IndexSpaceExpression*,
            FieldMaskSet<LogicalView> >::const_iterator it =
            intermediate.begin(); it != intermediate.end(); it++)
        expr_fields.insert(it->first, it->second.get_valid_mask());
      LegionList<FieldSet<IndexSpaceExpression*> > field_exprs;
      expr_fields.compute_field_sets(FieldMask(), field_exprs);
      for (LegionList<FieldSet<IndexSpaceExpression*> >::const_iterator
            eit = field_exprs.begin(); eit != field_exprs.end(); eit++)
      {
        if (eit->elements.size() == 1)
        {
          IndexSpaceExpression *expr = *(eit->elements.begin());
          FieldMaskSet<LogicalView> &src_views = intermediate[expr];
          FieldMaskSet<LogicalView> &dst_views = target[expr];
          // No chance of overlapping so just move everything over
          if (eit->set_mask != src_views.get_valid_mask())
          {
            // Move over the relevant expressions
            for (FieldMaskSet<LogicalView>::const_iterator it = 
                  src_views.begin(); it != src_views.end(); it++)
            {
              const FieldMask overlap = eit->set_mask & it->second;
              if (!overlap)
                continue;
              dst_views.insert(it->first, overlap);
            }
          }
          else if (!dst_views.empty())
          {
            for (FieldMaskSet<LogicalView>::const_iterator it = 
                  src_views.begin(); it != src_views.end(); it++)
              dst_views.insert(it->first, it->second);
          }
          else
            dst_views.swap(src_views);
          continue;
        }
        RegionTreeForest *forest = context->runtime->forest;
        // Do pair-wise intersection tests for overlapping of the expressions
        std::vector<IndexSpaceExpression*> disjoint_expressions;
        std::vector<std::vector<IndexSpaceExpression*> > disjoint_components;
        for (std::set<IndexSpaceExpression*>::const_iterator isit = 
              eit->elements.begin(); isit != eit->elements.end(); isit++)
        {
          IndexSpaceExpression *current = *isit;
          const size_t num_expressions = disjoint_expressions.size();
          for (unsigned idx = 0; idx < num_expressions; idx++)
          {
            IndexSpaceExpression *expr = disjoint_expressions[idx];
            // Compute the intersection
            IndexSpaceExpression *intersection =
              forest->intersect_index_spaces(expr, current);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            if (volume == current->get_volume())
            {
              // this one dominates us, see if we need to split ourself off
              if (volume < expr->get_volume())
              {
                disjoint_expressions.push_back(intersection);
                disjoint_components.resize(disjoint_components.size() + 1);
                std::vector<IndexSpaceExpression*> &components =
                  disjoint_components.back();
                components.insert(components.end(),
                    disjoint_components[idx].begin(), 
                    disjoint_components[idx].end());
                components.push_back(*isit);
                disjoint_expressions[idx] =
                  forest->subtract_index_spaces(expr, intersection);
              }
              else // Congruent so we are done
                disjoint_components[idx].push_back(*isit);
              current = NULL;
              break;
            }
            else if (volume == expr->get_volume())
            {
              // We dominate the expression so add ourselves and compute diff
              disjoint_components[idx].push_back(*isit); 
              current = forest->subtract_index_spaces(current, intersection);
#ifdef DEBUG_LEGION
              assert(!current->is_empty());
#endif
            }
            else
            {
              // Split into the three parts and keep going
              disjoint_expressions.push_back(intersection);
              disjoint_components.resize(disjoint_components.size() + 1);
              std::vector<IndexSpaceExpression*> &components = 
                disjoint_components.back();
              components.insert(components.end(),
                  disjoint_components[idx].begin(), 
                  disjoint_components[idx].end());
              components.push_back(*isit);
              disjoint_expressions[idx] =
                forest->subtract_index_spaces(expr, intersection);
              current = forest->subtract_index_spaces(current, intersection);
#ifdef DEBUG_LEGION
              assert(!current->is_empty());
#endif
            }
          }
          if (current != NULL)
          {
            disjoint_expressions.push_back(current);
            disjoint_components.resize(disjoint_components.size() + 1);
            disjoint_components.back().push_back(*isit);
          }
        }
        // Now we have overlapping expressions and constituents for
        // each of what used to be the old equivalence sets, so we
        // can now build the actual output target
        for (unsigned idx = 0; idx < disjoint_expressions.size(); idx++)
        {
          FieldMaskSet<LogicalView> &dst_views =
            target[disjoint_expressions[idx]];
          for (std::vector<IndexSpaceExpression*>::const_iterator sit =
                disjoint_components[idx].begin(); sit !=
                disjoint_components[idx].end(); sit++)
          {
#ifdef DEBUG_LEGION
            assert(intermediate.find(*sit) != intermediate.end());
#endif
            const FieldMaskSet<LogicalView> &src_views = intermediate[*sit];
            for (FieldMaskSet<LogicalView>::const_iterator it =
                  src_views.begin(); it != src_views.end(); it++)
            {
              const FieldMask overlap = it->second & eit->set_mask;
              if (!overlap)
                continue;
              dst_views.insert(it->first, overlap);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::find_overlaps(TraceViewSet &target, 
                                     IndexSpaceExpression *expr, 
                                     const bool expr_covers, 
                                     const FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      if (expr_covers)
      {
        for (ViewExprs::const_iterator vit = 
              conditions.begin(); vit != conditions.end(); vit++)
        {
          if (!(vit->second.get_valid_mask() - mask))
          {
            // sending everything
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  vit->second.begin(); it != vit->second.end(); it++)
              target.insert(vit->first, it->first, it->second);
          }
          else
          {
            // filtering on fields
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  vit->second.begin(); it != vit->second.end(); it++)
            {
              const FieldMask overlap = mask & it->second;
              if (!overlap)
                continue;
              target.insert(vit->first, it->first, overlap);
            }
          }
        }
      }
      else
      {
        RegionTreeForest *forest = context->runtime->forest;
        for (ViewExprs::const_iterator vit = 
              conditions.begin(); vit != conditions.end(); vit++)
        {
          FieldMask view_overlap = vit->second.get_valid_mask() & mask;
          if (!view_overlap)
            continue;
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                vit->second.begin(); it != vit->second.end(); it++)
          {
            const FieldMask overlap = it->second & view_overlap;
            if (!overlap)
              continue;
            IndexSpaceExpression *expr_overlap = 
              forest->intersect_index_spaces(it->first, expr); 
            const size_t volume = expr_overlap->get_volume();
            if (volume > 0)
            {
              if (volume == expr->get_volume())
                target.insert(vit->first, expr, overlap);
              else if (volume == it->first->get_volume())
                target.insert(vit->first, it->first, overlap);
              else
                target.insert(vit->first, expr_overlap, overlap);
            }
            view_overlap -= overlap;
            if (!view_overlap)
              break;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::empty(void) const
    //--------------------------------------------------------------------------
    {
      return conditions.empty();
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::merge(TraceViewSet &target) const
    //--------------------------------------------------------------------------
    {
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); ++vit)
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
          target.insert(vit->first, it->first, it->second);
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::pack(Serializer &rez, AddressSpaceID target,
                            const bool pack_references) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(conditions.size());
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); ++vit)
      {
        rez.serialize(vit->first->did);
        rez.serialize<size_t>(vit->second.size());
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
        {
          it->first->pack_expression(rez, target);
          rez.serialize(it->second);
        }
        if (pack_references)
          vit->first->pack_valid_ref();
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::unpack(Deserializer &derez, size_t num_views,
                         AddressSpaceID source, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      RegionTreeForest *forest = context->runtime->forest;
      for (unsigned idx1 = 0; idx1 < num_views; idx1++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView *view =
          forest->runtime->find_or_request_logical_view(did, ready);
        size_t num_exprs;
        derez.deserialize(num_exprs);
        FieldMaskSet<IndexSpaceExpression> &exprs = conditions[view];
        for (unsigned idx2 = 0; idx2 < num_exprs; idx2++)
        {
          IndexSpaceExpression *expr = 
            IndexSpaceExpression::unpack_expression(derez, forest, source);
          FieldMask mask;
          derez.deserialize(mask);
          if (exprs.insert(expr, mask))
            expr->add_nested_expression_reference(owner_did);
        }
        if (ready.exists() && !ready.has_triggered())
          ready_events.insert(ready);
        if (LogicalView::is_collective_did(did))
          has_collective_views = true;
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::unpack_references(void) const
    //--------------------------------------------------------------------------
    {
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); vit++)
      {
        vit->first->add_nested_valid_ref(owner_did);
        vit->first->unpack_valid_ref();
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::dump(void) const
    //--------------------------------------------------------------------------
    {
      RegionTreeForest *forest = context->runtime->forest;
      RegionNode *region = forest->get_tree(tree_id);
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); ++vit)
      {
        LogicalView *view = vit->first;
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); ++it)
        {
          char *mask = region->column_source->to_string(it->second, context);
          const void *name = NULL; size_t name_size = 0;
          if (view->is_fill_view())
          {
            log_tracing.info() << "  "
                      << "Fill view: " << view
                      << ", Index expr: " << it->first->expr_id
                      << ", Name: " << (name_size > 0 ? (const char*)name : "")
                      << ", Fields: " << mask;
          }
          else if (view->is_collective_view())
          {
            log_tracing.info() << "  Collective "
                      << (view->is_reduction_kind() ? "Reduction " : "")
                      << "view: " << view
                      << ", Index expr: " << it->first->expr_id
                      << ", Name: " << (name_size > 0 ? (const char*)name : "")
                      << ", Fields: " << mask;
          }
          else
          {
            PhysicalManager *manager = 
              view->as_individual_view()->get_manager();
            log_tracing.info() << "  "
                      << (view->is_reduction_view() ? 
                          "Reduction" : "Materialized")
                      << " view: " << view << ", Inst: "
                      << std::hex << manager->get_instance().id << std::dec
                      << ", Index expr: " << it->first->expr_id
                      << ", Name: " << (name_size > 0 ? (const char*)name : "")
                      << ", Fields: " << mask;
          }
          free(mask);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::antialias_individual_view(IndividualView *view,
                                                 FieldMask mask)
    //--------------------------------------------------------------------------
    {
      if (!has_collective_views)
        return;
      // See if we can find it in which case we know that it doesn't alias
      // with anything else so there is nothing to split
      ViewExprs::const_iterator finder = conditions.find(view);
      if (finder != conditions.end())
      {
        mask -= finder->second.get_valid_mask();
        if (!mask)
          return;
      }
      FieldMaskSet<CollectiveView> to_refine;
      for (ViewExprs::const_iterator it = 
            conditions.begin(); it != conditions.end(); it++)
      {
        if (!it->first->is_collective_view())
          continue;
        const FieldMask overlap = mask & it->second.get_valid_mask();
        if (!overlap)
          continue;
        CollectiveView *collective = it->first->as_collective_view();
        if (!collective->aliases(view))
          continue;
        to_refine.insert(collective, overlap);  
        mask -= overlap;
        if (!mask)
          break;
      }
      Runtime *runtime = context->runtime;
      // We've got the names of any collective views that need to be
      // refined to not include this individual view, so go ahead and
      // ask the context to make that collective view for us
      std::vector<RtEvent> views_ready;
      std::map<CollectiveView*,PhysicalManager*> individual_results;
      std::map<CollectiveView*,InnerContext::CollectiveResult*> results;
      for (FieldMaskSet<CollectiveView>::const_iterator it = 
            to_refine.begin(); it != to_refine.end(); it++)
      {
        std::vector<DistributedID> dids = it->first->instances;
        std::vector<DistributedID>::iterator finder = 
          std::find(dids.begin(), dids.end(), view->manager->did);
#ifdef DEBUG_LEGION
        assert(finder != dids.end());
#endif
        dids.erase(finder);
        RtEvent ready;
        if (dids.size() > 1)
        {
          InnerContext::CollectiveResult *result =
            context->find_or_create_collective_view(tree_id, dids, ready);
          results[it->first] = result;
        }
        else
        {
          // Just making a single view at this point
          PhysicalManager *manager = 
            runtime->find_or_request_instance_manager(dids.back(), ready);
          individual_results[it->first] = manager;
        }
        if (ready.exists())
          views_ready.push_back(ready);
      }
      if (!views_ready.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(views_ready);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      for (FieldMaskSet<CollectiveView>::const_iterator rit =
            to_refine.begin(); rit != to_refine.end(); rit++)
      {
        RtEvent ready;
        InstanceView *view = NULL;
        std::map<CollectiveView*,PhysicalManager*>::const_iterator
          individual_finder = individual_results.find(rit->first);
        if (individual_finder == individual_results.end())
        {
#ifdef DEBUG_LEGION
          assert(results.find(rit->first) != results.end());
#endif
          // Common case
          InnerContext::CollectiveResult *result = results[rit->first];
          // Then wait for the collective view to be registered
          if (result->ready_event.exists() && 
              !result->ready_event.has_triggered())
            result->ready_event.wait();
          view = static_cast<InstanceView*>(
              runtime->find_or_request_logical_view(
                result->collective_did, ready));
          if (result->remove_reference())
            delete result;
        }
        else // Unusual case of an downgrading to an individual view
          view = context->create_instance_top_view(individual_finder->second,
                                                   runtime->address_space);
        ViewExprs::iterator finder = conditions.find(rit->first);
        if (finder->second.get_valid_mask() == rit->second)
        {
          // Can just swap expressions over in this particular case
          conditions[view].swap(finder->second);
          // Remove the reference if we have one
          if (finder->first->remove_nested_valid_ref(owner_did))
            delete finder->first;
          conditions.erase(finder);
        }
        else
        {
          // Need to filter over specific expression in this case
          FieldMaskSet<IndexSpaceExpression> &to_add = conditions[view];
          std::vector<IndexSpaceExpression*> to_delete; 
          for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                finder->second.begin(); it != finder->second.end(); it++)
          {
            const FieldMask overlap = rit->second & it->second;
            if (!overlap)
              continue;
            to_add.insert(it->first, overlap);
            it.filter(overlap);
            if (!it->second) // reference flows back
              to_delete.push_back(it->first);
            else
              it->first->add_nested_expression_reference(owner_did);
          }
          for (std::vector<IndexSpaceExpression*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
            finder->second.erase(*it);
          finder->second.tighten_valid_mask();
        }
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        view->add_nested_valid_ref(owner_did);
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::antialias_collective_view(CollectiveView *collective,
                  FieldMask mask, FieldMaskSet<InstanceView> &alternative_views)
    //--------------------------------------------------------------------------
    {
      ViewExprs::const_iterator collective_finder = conditions.find(collective);
      if (collective_finder != conditions.end())
      {
        // If we can already find it then it is already anti-aliased so
        // there's no need to do the rest of this work for those fields
        FieldMask overlap = mask & collective_finder->second.get_valid_mask();
        if (!!overlap)
        {
          alternative_views.insert(collective, overlap);
          mask -= overlap;
          if (!mask)
            return;
        }
      }
      ViewExprs to_add;
      CollectiveAntiAlias alias_analysis(collective);
      for (ViewExprs::iterator vit = conditions.begin(); 
            vit != conditions.end(); /*nothing*/)
      {
        if (!vit->first->is_instance_view())
        {
          vit++;
          continue;
        }
        const FieldMask view_overlap = mask & vit->second.get_valid_mask();
        if (!view_overlap)
        {
          vit++;
          continue;
        }
        if (vit->first->is_collective_view())
        {
          CollectiveView *current = vit->first->as_collective_view();
          // See how the instances overlap
          // First get the intersection
          std::vector<DistributedID> intersection;
          if (current->instances.size() < collective->instances.size())
          {
            for (std::vector<DistributedID>::const_iterator it =
                  current->instances.begin(); it !=
                  current->instances.end(); it++)
              if (std::binary_search(collective->instances.begin(),
                    collective->instances.end(), *it))
                intersection.push_back(*it);
          }
          else
          {
            for (std::vector<DistributedID>::const_iterator it =
                  collective->instances.begin(); it !=
                  collective->instances.end(); it++)
              if (std::binary_search(current->instances.begin(),
                    current->instances.end(), *it))
                intersection.push_back(*it);
          }
          // If they don't overlap at all then there's nothing to do
          if (intersection.empty())
          {
            vit++;
            continue;
          }
          // Don't care about expressions for this analysis
          // but we're reusing an exisint alias so we have to
          // conform to get the linker to work
          IndexSpaceExpression *null_expr = NULL;
          alias_analysis.traverse(current, view_overlap, null_expr);
          if (intersection.size() == current->instances.size())
          {
#ifdef DEBUG_LEGION
            assert(intersection.size() < collective->instances.size());
#endif
            vit++;
          }
          else
          {
            // Otherwise, if vit->first is not covered by the intersection
            // then we need to do two things
            // 1. Create a new instance for the difference and record
            //    any overlapping expressions for that in to_add
            std::vector<DistributedID> difference;
            for (std::vector<DistributedID>::const_iterator it =
                  current->instances.begin(); it != 
                  current->instances.end(); it++)
              if (!std::binary_search(collective->instances.begin(),
                    collective->instances.end(), *it))
                difference.push_back(*it);
            InstanceView *diff_view = find_instance_view(difference);
            if (to_add.find(diff_view) == to_add.end())
              diff_view->add_nested_valid_ref(owner_did);
            // 2. Make a new instance for the intersection, analyze it
            //    and record any overlapping expressions in to_add
            InstanceView *inter_view = 
              (intersection.size() == collective->instances.size()) ?
              collective : find_instance_view(intersection);
            if (to_add.find(inter_view) == to_add.end())
              inter_view->add_nested_valid_ref(owner_did);
            std::vector<IndexSpaceExpression*> to_delete;
            for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                  vit->second.begin(); it != vit->second.end(); it++)
            {
              const FieldMask overlap = view_overlap & it->second;
              if (!overlap)
                continue;
              if (to_add[diff_view].insert(it->first, overlap))
                it->first->add_nested_expression_reference(owner_did);
              to_add[inter_view].insert(it->first, overlap);
              it.filter(overlap);
              if (!it->second) // reference flows back
                to_delete.push_back(it->first);
              else
                it->first->add_nested_expression_reference(owner_did);
            }
            if (to_delete.size() < vit->second.size())
            {
              for (std::vector<IndexSpaceExpression*>::const_iterator it =
                    to_delete.begin(); it != to_delete.end(); it++)
                vit->second.erase(*it);
              vit->second.tighten_valid_mask();
              vit++;
            }
            else
            {
              vit->second.clear();
              if (vit->first->remove_nested_valid_ref(owner_did))
                delete vit->first;
              ViewExprs::iterator to_delete = vit++;
              conditions.erase(to_delete);
            }
          }
        }
        else // just an individual view, so we can just traverse it
        {
          IndividualView *individual = vit->first->as_individual_view();
          // Check to see if it they alias
          if (std::binary_search(collective->instances.begin(),
                collective->instances.end(), individual->manager->did))
          {
            // Don't care about expressions for this analysis
            // but we're reusing an exisint alias so we have to
            // conform to get the linker to work
            IndexSpaceExpression *null_expr = NULL;
            alias_analysis.traverse(individual, view_overlap, null_expr);
          }
          vit++;
        }
      }
      // Now traverse the alias analysis and record the alternate views
      // and their index space expressions in to_add
      FieldMask allvalid_mask = mask;
      alias_analysis.visit_leaves(mask, allvalid_mask, 
                                  *this, alternative_views);
      if (!!allvalid_mask)
        alternative_views.insert(collective, allvalid_mask);
      if (!to_add.empty())
      {
        for (ViewExprs::iterator vit = to_add.begin(); 
              vit != to_add.end(); vit++)
        {
          ViewExprs::iterator finder = conditions.find(vit->first);
          if (finder != conditions.end())
          {
            // Remove duplicate view reference
            vit->first->remove_nested_valid_ref(owner_did);
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  vit->second.begin(); it != vit->second.end(); it++)
              // Remove duplicate references
              if (!finder->second.insert(it->first, it->second))
                it->first->remove_nested_expression_reference(owner_did);
          }
          else
          {
            // Already have a reference to the view so pass it here
            // Also have references on the expression so the swap is enough
            conditions[vit->first].swap(vit->second);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    InstanceView* TraceViewSet::find_instance_view(
                                        const std::vector<DistributedID> &dids)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!dids.empty());
#endif
      if (dids.size() > 1)
      {
        RtEvent ready;
        InnerContext::CollectiveResult *result =
          context->find_or_create_collective_view(tree_id, dids, ready);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        // Then wait for the collective view to be registered
        if (result->ready_event.exists() && 
            !result->ready_event.has_triggered())
          result->ready_event.wait();
        InstanceView *view = static_cast<InstanceView*>(
          context->runtime->find_or_request_logical_view(
            result->collective_did, ready));
        if (result->remove_reference())
          delete result;
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        return view;
      }
      else
      {
        RtEvent ready;
        PhysicalManager *manager =
          context->runtime->find_or_request_instance_manager(
              dids.back(), ready);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        return context->create_instance_top_view(manager,
                        context->runtime->address_space);
      }
    }

    /////////////////////////////////////////////////////////////
    // TraceConditionSet
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceConditionSet::TraceConditionSet(PhysicalTrace *trace,
                   RegionTreeForest *f, unsigned parent_req_idx, 
                   IndexSpaceExpression *expr,
                   const FieldMask &mask, RegionTreeID tid)
      : EqSetTracker(set_lock), context(trace->logical_trace->context),
        forest(f), condition_expr(expr), condition_mask(mask), tree_id(tid),
        parent_req_index(parent_req_idx), precondition_views(NULL),
        anticondition_views(NULL), postcondition_views(NULL)
    //--------------------------------------------------------------------------
    {
      condition_expr->add_base_expression_reference(TRACE_REF);
    }

    //--------------------------------------------------------------------------
    TraceConditionSet::~TraceConditionSet(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(equivalence_sets.empty());
#endif
      for (LegionMap<IndexSpaceExpression*,
                     FieldMaskSet<LogicalView> >::const_iterator eit =
            preconditions.begin(); eit != preconditions.end(); eit++)
      {
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              eit->second.begin(); it != eit->second.end(); it++)
          if (it->first->remove_base_valid_ref(TRACE_REF))
            delete it->first;
        if (eit->first->remove_base_expression_reference(TRACE_REF))
          delete eit->first;
      }
      for (LegionMap<IndexSpaceExpression*,
                     FieldMaskSet<LogicalView> >::const_iterator eit =
            anticonditions.begin(); eit != anticonditions.end(); eit++)
      {
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              eit->second.begin(); it != eit->second.end(); it++)
          if (it->first->remove_base_valid_ref(TRACE_REF))
            delete it->first;
        if (eit->first->remove_base_expression_reference(TRACE_REF))
          delete eit->first;
      }
      for (LegionMap<IndexSpaceExpression*,
                     FieldMaskSet<LogicalView> >::const_iterator eit =
            postconditions.begin(); eit != postconditions.end(); eit++)
      {
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              eit->second.begin(); it != eit->second.end(); it++)
          if (it->first->remove_base_valid_ref(TRACE_REF))
            delete it->first;
        if (eit->first->remove_base_expression_reference(TRACE_REF))
          delete eit->first;
      }
      for (std::set<IndexSpaceExpression*>::const_iterator it =
            unique_view_expressions.begin(); it != 
            unique_view_expressions.end(); it++) 
        if ((*it)->remove_base_expression_reference(TRACE_REF))
          delete (*it);
      if (condition_expr->remove_base_expression_reference(TRACE_REF))
        delete condition_expr;
      if (precondition_views != NULL)
        delete precondition_views;
      if (anticondition_views != NULL)
        delete anticondition_views;
      if (postcondition_views != NULL)
        delete postcondition_views;
    }

#if 0
    //--------------------------------------------------------------------------
    void TraceConditionSet::record_subscription(VersionManager *owner,
                                                AddressSpaceID space)
    //--------------------------------------------------------------------------
    {
      const std::pair<VersionManager*,AddressSpaceID> key(owner,space);
      AutoLock s_lock(set_lock);
      if (subscription_owners.empty())
        add_reference();
      std::map<std::pair<VersionManager*,AddressSpaceID>,unsigned>::iterator
        finder = subscription_owners.find(key);
      if (finder == subscription_owners.end())
        subscription_owners[key] = 1;
      else
        finder->second++;
    }

    //--------------------------------------------------------------------------
    bool TraceConditionSet::finish_subscription(EqKDTree *owner,
                                                AddressSpaceID space)
    //--------------------------------------------------------------------------
    {
      const std::pair<EqKDTree*,AddressSpaceID> key(owner,space);
      AutoLock s_lock(set_lock);
      std::map<std::pair<EqKDTree*,AddressSpaceID>,unsigned>::iterator
        finder = subscription_owners.find(key);
#ifdef DEBUG_LEGION
      assert(finder != subscription_owners.end());
      assert(finder->second > 0);
#endif
      if (--finder->second == 0)
        subscription_owners.erase(finder);
      if (!subscription_owners.empty())
        return false;
      return remove_reference();
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::record_equivalence_set(EquivalenceSet *set,
                                                   const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(set_lock);
      if (current_sets.insert(set, mask))
        set->add_base_resource_ref(TRACE_REF);
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::record_pending_equivalence_set(EquivalenceSet *set,
                                                          const FieldMask &mask) 
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(set_lock);
      pending_sets.insert(set, mask);
    }

    //--------------------------------------------------------------------------
    bool TraceConditionSet::invalidate_equivalence_sets(const FieldMask &mask,
                                 const std::vector<RtEvent> &invalidated_events)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(set_lock);
      if (!(mask - invalid_mask))
        return;
      invalid_mask |= mask;
      std::vector<EquivalenceSet*> to_delete;
      for (FieldMaskSet<EquivalenceSet>::iterator it =
            equivalence_sets.begin(); it != equivalence_sets.end(); it++)
      {
        it.filter(mask);
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (std::vector<EquivalenceSet*>::const_iterator it =
            to_delete.begin(); it != to_delete.end(); it++)
      {
        equivalence_sets.erase(*it);
        if ((*it)->remove_base_resource_ref(TRACE_REF))
          assert(false); // should never end up deleting this here
      }
      equivalence_sets.tighten_valid_mask();
    }
#endif

    //--------------------------------------------------------------------------
    void TraceConditionSet::invalidate_equivalence_sets(void)
    //--------------------------------------------------------------------------
    {
      FieldMaskSet<EquivalenceSet> to_remove;
      LegionMap<AddressSpaceID,TreeInvalidations> to_cancel;
      {
        AutoLock s_lock(set_lock);
        if (current_subscriptions.empty())
        {
#ifdef DEBUG_LEGION
          assert(equivalence_sets.empty());
#endif
          return;
        }
        // Copy and not remove since we need to see the acknowledgement
        // before we know when it is safe to remove our references
        to_remove.swap(equivalence_sets);
        for (LegionMap<AddressSpaceID,FieldMaskSet<EqKDTree> >::iterator it =
              current_subscriptions.begin(); it != 
              current_subscriptions.end(); it++)
        {
          TreeInvalidations &invalidations = to_cancel[it->first];
          invalidations.subscribers.swap(it->second);
          invalidations.all_subscribers_finished = true;
        }
        current_subscriptions.clear();
      }
      cancel_subscriptions(context->runtime, to_cancel);
      for (FieldMaskSet<EquivalenceSet>::const_iterator it =
            to_remove.begin(); it != to_remove.end(); it++)
        if (it->first->remove_base_gc_ref(TRACE_REF))
          delete it->first;
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::capture(EquivalenceSet *set, const FieldMask &mask,
                                    std::vector<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(precondition_views == NULL);
      assert(anticondition_views == NULL);
      assert(postcondition_views == NULL);
#endif
      const RtEvent ready_event = 
        set->capture_trace_conditions(this, set->local_space,
            condition_expr, mask, RtUserEvent::NO_RT_USER_EVENT);
      if (ready_event.exists() && !ready_event.has_triggered())
        ready_events.push_back(ready_event);
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::receive_capture(TraceViewSet *pre, 
               TraceViewSet *anti, TraceViewSet *post, std::set<RtEvent> &ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(precondition_views == NULL);
      assert(anticondition_views == NULL);
      assert(postcondition_views == NULL);
#endif
      precondition_views = pre;
      anticondition_views = anti;
      postcondition_views = post;
      if (precondition_views != NULL)
      {
        precondition_views->transpose_uniquely(preconditions,
                                               unique_view_expressions);
        for (LegionMap<IndexSpaceExpression*,
                       FieldMaskSet<LogicalView> >::const_iterator 
              eit = preconditions.begin(); eit != preconditions.end(); eit++)
        {
          eit->first->add_base_expression_reference(TRACE_REF);
          for (FieldMaskSet<LogicalView>::const_iterator it = 
                eit->second.begin(); it != eit->second.end(); it++)
            it->first->add_base_valid_ref(TRACE_REF);
        }
      }
      if (anticondition_views != NULL)
      {
        anticondition_views->transpose_uniquely(anticonditions,
                                                unique_view_expressions);
        for (LegionMap<IndexSpaceExpression*,
                       FieldMaskSet<LogicalView> >::const_iterator 
              eit = anticonditions.begin(); eit != anticonditions.end(); eit++)
        {
          eit->first->add_base_expression_reference(TRACE_REF);
          for (FieldMaskSet<LogicalView>::const_iterator it = 
                eit->second.begin(); it != eit->second.end(); it++)
            it->first->add_base_valid_ref(TRACE_REF);
        }
      }
      if (postcondition_views != NULL)
      {
        postcondition_views->transpose_uniquely(postconditions,
                                                unique_view_expressions);
        for (LegionMap<IndexSpaceExpression*,
                       FieldMaskSet<LogicalView> >::const_iterator 
              eit = postconditions.begin(); eit != postconditions.end(); eit++)
        {
          eit->first->add_base_expression_reference(TRACE_REF);
          for (FieldMaskSet<LogicalView>::const_iterator it = 
                eit->second.begin(); it != eit->second.end(); it++)
            it->first->add_base_valid_ref(TRACE_REF);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool TraceConditionSet::is_empty(void) const
    //--------------------------------------------------------------------------
    {
      if (precondition_views != NULL)
        return false;
      if (anticondition_views != NULL)
        return false;
      if (postcondition_views != NULL)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool TraceConditionSet::is_replayable(bool &not_subsumed,
                                       TraceViewSet::FailedPrecondition *failed)
    //--------------------------------------------------------------------------
    {
      bool replayable = true;
      // Note that it is ok to have precondition views and no postcondition
      // views because that means that everything was read-only and therefore
      // still idempotent and replayable
      if ((precondition_views != NULL) && (postcondition_views != NULL) &&
          !precondition_views->subsumed_by(*postcondition_views, true, failed))
      {
        if ((failed != NULL) && (postcondition_views == NULL))
          precondition_views->record_first_failed(failed);
        replayable = false;
        not_subsumed = true;
      }
      if (replayable && 
          (postcondition_views != NULL) && (anticondition_views != NULL) &&
          !postcondition_views->independent_of(*anticondition_views, failed))
      {
        replayable = false;
        not_subsumed = false;
      }
      // Clean up our view objects since we no longer need them
      if (precondition_views != NULL)
      {
        delete precondition_views;
        precondition_views = NULL;
      }
      if (anticondition_views != NULL)
      {
        delete anticondition_views;
        anticondition_views = NULL;
      }
      if (postcondition_views != NULL)
      {
        delete postcondition_views;
        postcondition_views = NULL;
      }
      return replayable;
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::dump_preconditions(void) const
    //--------------------------------------------------------------------------
    {
      TraceViewSet dump_view_set(context, 0/*owner did*/,
                                 condition_expr, tree_id);
      for (ExprViews::const_iterator eit = 
            preconditions.begin(); eit != preconditions.end(); eit++)
        for (FieldMaskSet<LogicalView>::const_iterator it =
              eit->second.begin(); it != eit->second.end(); it++)
          dump_view_set.insert(it->first, eit->first, it->second);
      dump_view_set.dump();
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::dump_anticonditions(void) const
    //--------------------------------------------------------------------------
    {
      TraceViewSet dump_view_set(context, 0/*owner did*/,
                                 condition_expr, tree_id);
      for (ExprViews::const_iterator eit = 
            anticonditions.begin(); eit != anticonditions.end(); eit++)
        for (FieldMaskSet<LogicalView>::const_iterator it =
              eit->second.begin(); it != eit->second.end(); it++)
          dump_view_set.insert(it->first, eit->first, it->second);
      dump_view_set.dump();
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::dump_postconditions(void) const
    //--------------------------------------------------------------------------
    {
      TraceViewSet dump_view_set(context, 0/*owner did*/,
                                 condition_expr, tree_id);
      for (ExprViews::const_iterator eit = 
            postconditions.begin(); eit != postconditions.end(); eit++)
        for (FieldMaskSet<LogicalView>::const_iterator it =
              eit->second.begin(); it != eit->second.end(); it++)
          dump_view_set.insert(it->first, eit->first, it->second);
      dump_view_set.dump();
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::test_require(Operation *op, 
             std::set<RtEvent> &ready_events, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // We should not need the lock here because the trace should be 
      // blocking all other operations from running and changing the 
      // equivalence sets while we are here
      // First check to see if we need to recompute our equivalence sets
      const FieldMask invalid_mask = 
        condition_mask - equivalence_sets.get_valid_mask();
      if (!!invalid_mask)
      {
        const UniqueID opid = op->get_unique_op_id();
        const RtEvent ready = recompute_equivalence_sets(opid, invalid_mask);
        if (ready.exists() && !ready.has_triggered())
        {
          const RtUserEvent tested = Runtime::create_rt_user_event();
          const RtUserEvent applied = Runtime::create_rt_user_event();
          DeferTracePreconditionTestArgs args(this, op, tested, applied);
          forest->runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_DEFERRED_PRIORITY, ready);
          ready_events.insert(tested);
          applied_events.insert(applied);
          return;
        }
      }
#ifdef DEBUG_LEGION
      assert(precondition_analyses.empty());
      assert(anticondition_analyses.empty());
#endif
      // Make analyses for the precondition and anticondition tests
      for (ExprViews::const_iterator eit = 
            preconditions.begin(); eit != preconditions.end(); eit++)
      {
        InvalidInstAnalysis *analysis = new InvalidInstAnalysis(forest->runtime,  
            op, precondition_analyses.size(), eit->first, eit->second);
        analysis->add_reference();
        precondition_analyses.push_back(analysis);
        std::set<RtEvent> deferral_events;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it =
              equivalence_sets.begin(); it != equivalence_sets.end(); it++)
        {
          const FieldMask overlap = eit->second.get_valid_mask() & it->second;
          if (!overlap)
            continue;
          analysis->analyze(it->first, overlap, deferral_events,applied_events);
        }
        const RtEvent traversal_done = deferral_events.empty() ?
          RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
        if (traversal_done.exists() || analysis->has_remote_sets())
        {
          const RtEvent ready = 
            analysis->perform_remote(traversal_done, applied_events);
          if (ready.exists() && !ready.has_triggered())
            ready_events.insert(ready);
        }
      }
      for (ExprViews::const_iterator eit =
            anticonditions.begin(); eit != anticonditions.end(); eit++)
      {
        AntivalidInstAnalysis *analysis = 
          new AntivalidInstAnalysis(forest->runtime, op, 
              anticondition_analyses.size(), eit->first, eit->second);
        analysis->add_reference();
        anticondition_analyses.push_back(analysis);
        std::set<RtEvent> deferral_events;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it =
              equivalence_sets.begin(); it != equivalence_sets.end(); it++)
        {
          const FieldMask overlap = eit->second.get_valid_mask() & it->second;
          if (!overlap)
            continue;
          analysis->analyze(it->first, overlap, deferral_events,applied_events);
        }
        const RtEvent traversal_done = deferral_events.empty() ?
          RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
        if (traversal_done.exists() || analysis->has_remote_sets())
        {
          const RtEvent ready = 
            analysis->perform_remote(traversal_done, applied_events);
          if (ready.exists() && !ready.has_triggered())
            ready_events.insert(ready);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void TraceConditionSet::handle_precondition_test(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferTracePreconditionTestArgs *dargs = 
        (const DeferTracePreconditionTestArgs*)args;
      std::set<RtEvent> ready_events, applied_events;
      dargs->set->test_require(dargs->op, ready_events, applied_events);
      if (!ready_events.empty())
        Runtime::trigger_event(dargs->done_event, 
            Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(dargs->done_event);
      if (!applied_events.empty())
        Runtime::trigger_event(dargs->applied_event,
            Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(dargs->applied_event);
    }

    //--------------------------------------------------------------------------
    bool TraceConditionSet::check_require(void)
    //--------------------------------------------------------------------------
    {
      bool satisfied = true;
      for (std::vector<InvalidInstAnalysis*>::const_iterator it =
            precondition_analyses.begin(); it != 
            precondition_analyses.end(); it++)
      {
        if ((*it)->has_invalid())
          satisfied = false;
        if ((*it)->remove_reference())
          delete (*it);
      }
      precondition_analyses.clear();
      for (std::vector<AntivalidInstAnalysis*>::const_iterator it =
            anticondition_analyses.begin(); it != 
            anticondition_analyses.end(); it++)
      {
        if ((*it)->has_antivalid())
          satisfied = false;
        if ((*it)->remove_reference())
          delete (*it);
      }
      anticondition_analyses.clear();
      return satisfied;
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::ensure(Operation *op, 
                                   std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // We should not need the lock here because the trace should be 
      // blocking all other operations from running and changing the 
      // equivalence sets while we are here
      // First check to see if we need to recompute our equivalence sets
      const FieldMask invalid_mask = 
        condition_mask - equivalence_sets.get_valid_mask();
      if (!!invalid_mask)
      {
        const UniqueID opid = op->get_unique_op_id();
        const RtEvent ready = recompute_equivalence_sets(opid, invalid_mask);
        if (ready.exists() && !ready.has_triggered())
        {
          const RtUserEvent applied= Runtime::create_rt_user_event();
          DeferTracePostconditionTestArgs args(this, op, applied);
          forest->runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_DEFERRED_PRIORITY, ready);
          applied_events.insert(applied);
          return;
        }
      }
      // Perform an overwrite analysis for each of the postconditions
      unsigned index = 0;
      const TraceInfo trace_info(op);
      const RegionUsage usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
      for (ExprViews::const_iterator eit = 
            postconditions.begin(); eit != postconditions.end(); eit++, index++)
      {
        OverwriteAnalysis *analysis = new OverwriteAnalysis(forest->runtime,
            op, index, usage, eit->first, eit->second, 
            PhysicalTraceInfo(trace_info, index), ApEvent::NO_AP_EVENT);
        analysis->add_reference();
        std::set<RtEvent> deferral_events;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it =
              equivalence_sets.begin(); it != equivalence_sets.end(); it++)
        {
          const FieldMask overlap = eit->second.get_valid_mask() & it->second;
          if (!overlap)
            continue;
          analysis->analyze(it->first, overlap, deferral_events,applied_events);
        }
        const RtEvent traversal_done = deferral_events.empty() ?
          RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
        if (traversal_done.exists() || analysis->has_remote_sets())
          analysis->perform_remote(traversal_done, applied_events);
        if (analysis->remove_reference())
          delete analysis;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void TraceConditionSet::handle_postcondition_test(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferTracePostconditionTestArgs *dargs = 
        (const DeferTracePostconditionTestArgs*)args;
      std::set<RtEvent> ready_events;
      dargs->set->ensure(dargs->op, ready_events);
      if (!ready_events.empty())
        Runtime::trigger_event(dargs->done_event, 
            Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(dargs->done_event);
    }

    //--------------------------------------------------------------------------
    RtEvent TraceConditionSet::recompute_equivalence_sets(UniqueID opid,
                                                  const FieldMask &invalid_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!invalid_mask);
#endif
      AddressSpaceID space = forest->runtime->address_space;
      // Create a user event and store it in the equivalence set ready structure
      const RtUserEvent compute_event = Runtime::create_rt_user_event();
      {
        AutoLock s_lock(set_lock);
#ifdef DEBUG_LEGION
        assert(equivalence_sets_ready == NULL);
#endif
        equivalence_sets_ready = new LegionMap<RtUserEvent,FieldMask>();
        equivalence_sets_ready->insert(
            std::make_pair(compute_event, invalid_mask));
      }
      RtEvent ready = context->compute_equivalence_sets(this, space,
                      parent_req_index, condition_expr, invalid_mask);
      if (ready.exists() && !ready.has_triggered())
      {
        // Launch a meta-task to finalize this trace condition set
        LgFinalizeEqSetsArgs args(this, compute_event, opid,
            context, parent_req_index, condition_expr);
        return forest->runtime->issue_runtime_meta_task(args, 
                        LG_LATENCY_DEFERRED_PRIORITY, ready);
      }
      else
        finalize_equivalence_sets(compute_event, context, forest->runtime,
            parent_req_index, condition_expr, opid);
      return compute_event;
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTemplate
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTemplate::PhysicalTemplate(PhysicalTrace *t, ApEvent fence_event,
                                       TaskTreeCoordinates &&coords)
      : trace(t), coordinates(std::move(coords)), total_replays(1),
        replayable(false, "uninitialized"), fence_completion_id(0),
        replay_parallelism(t->runtime->max_replay_parallelism),
        has_virtual_mapping(false), has_no_consensus(false), last_fence(NULL)
    //--------------------------------------------------------------------------
    {
      recording.store(true);
      events.push_back(fence_event);
      event_map[fence_event] = fence_completion_id;
      pending_inv_topo_order.store(NULL);
      pending_transitive_reduction.store(NULL);
      instructions.push_back(
         new AssignFenceCompletion(*this, fence_completion_id, TraceLocalID()));
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate::~PhysicalTemplate(void)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock tpl_lock(template_lock);
        for (std::vector<TraceConditionSet*>::const_iterator it =
              conditions.begin(); it != conditions.end(); it++)
        {
          (*it)->invalidate_equivalence_sets();
          if ((*it)->remove_reference())
            delete (*it);
        }
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
      std::vector<unsigned> *inv_topo_order = pending_inv_topo_order.load();
      if (inv_topo_order != NULL)
        delete inv_topo_order;
      std::vector<std::vector<unsigned> > *transitive_reduction =
        pending_transitive_reduction.load();
      if (transitive_reduction != NULL)
        delete transitive_reduction;
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalTemplate::get_completion_for_deletion(void) const
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> all_events;
      std::set<ApEvent> local_barriers;
      for (std::map<ApEvent,BarrierAdvance*>::const_iterator it = 
            managed_barriers.begin(); it != managed_barriers.end(); it++)
        local_barriers.insert(it->second->get_current_barrier());
      for (std::map<ApEvent, unsigned>::const_iterator it = event_map.begin();
           it != event_map.end(); ++it)
        // If this is one of our local barriers then don't use it
        if (local_barriers.find(it->first) == local_barriers.end())
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
      // Skip the most recent one as that is going to be our term event
      for (int idx = events.size() - 2; idx > 0; idx--)
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
      std::set<RtEvent> ready_events;
      for (std::vector<TraceConditionSet*>::const_iterator it = 
            conditions.begin(); it != conditions.end(); it++)
        (*it)->test_require(op, ready_events, applied_events);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      bool result = true;
      for (std::vector<TraceConditionSet*>::const_iterator it = 
            conditions.begin(); it != conditions.end(); it++)
        if (!(*it)->check_require())
          result = false;
      return result;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::apply_postcondition(TraceSummaryOp *op,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      for (std::vector<TraceConditionSet*>::const_iterator it = 
            conditions.begin(); it != conditions.end(); it++)
        (*it)->ensure(op, applied_events);
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::check_preconditions(ReplTraceReplayOp *op,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> ready_events;
      for (std::vector<TraceConditionSet*>::const_iterator it = 
            conditions.begin(); it != conditions.end(); it++)
        (*it)->test_require(op, ready_events, applied_events);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      bool result = true;
      for (std::vector<TraceConditionSet*>::const_iterator it = 
            conditions.begin(); it != conditions.end(); it++)
        if (!(*it)->check_require())
          result = false;
      return result;
    } 

    //--------------------------------------------------------------------------
    void PhysicalTemplate::apply_postcondition(ReplTraceSummaryOp *op,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      for (std::vector<TraceConditionSet*>::const_iterator it = 
            conditions.begin(); it != conditions.end(); it++)
        (*it)->ensure(op, applied_events);
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate::Replayable PhysicalTemplate::check_replayable(
                                         ReplTraceOp *op, InnerContext *context,
                                         UniqueID opid, bool has_blocking_call) 
    //--------------------------------------------------------------------------
    {
      if (has_blocking_call)
        return Replayable(false, "blocking call");

      if (has_virtual_mapping)
        return Replayable(false, "virtual mapping");

      if (has_no_consensus)
        return Replayable(false, "no recording consensus");
      
      // First let's get the equivalence sets with data for these regions
      // We'll use the result to get guide the creation of the trace condition
      // sets. Note we're going to end up recomputing the equivalence sets
      // inside the trace condition sets but that is a small price to pay to
      // minimize the number of conditions that we need to do the capture
      // Next we need to compute the equivalence sets for all these regions
      FieldMaskSet<EquivalenceSet> current_sets;
      std::map<EquivalenceSet*,unsigned> parent_req_indexes;
      {
        unsigned index = 0;
        std::set<RtEvent> eq_events;
        const ContextID ctx = context->get_context().get_id();
        LegionVector<VersionInfo> version_infos(trace_regions.size());
        std::map<RegionNode*,unsigned>::const_iterator req_it =
          trace_region_parent_req_indexes.begin();
        for (FieldMaskSet<RegionNode>::const_iterator it =
              trace_regions.begin(); it != trace_regions.end(); 
              it++, req_it++, index++)
        {
#ifdef DEBUG_LEGION
          // Make sure the parent_req_indexes zip with the trace_regions
          assert(req_it->first == it->first);
#endif
          it->first->perform_versioning_analysis(ctx, context, 
            &version_infos[index], it->second, opid, req_it->second, eq_events);
        }
#ifdef DEBUG_LEGION
        assert(req_it == trace_region_parent_req_indexes.end());
#endif
        // Reset in debug mode since we traversed to check
        req_it = trace_region_parent_req_indexes.begin();
        trace_regions.clear();
        if (!eq_events.empty())
        {
          const RtEvent wait_on = Runtime::merge_events(eq_events);
          if (wait_on.exists() && !wait_on.has_triggered())
            wait_on.wait();
        }
        for (unsigned idx = 0; idx < version_infos.size(); idx++, req_it++)
        {
          const FieldMaskSet<EquivalenceSet> &region_sets = 
              version_infos[idx].get_equivalence_sets();
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                region_sets.begin(); it != region_sets.end(); it++)
          {
            current_sets.insert(it->first, it->second);
            parent_req_indexes[it->first] = req_it->second;
          }
          if (req_it->first->remove_base_resource_ref(TRACE_REF))
            delete req_it->first;
        }
        trace_region_parent_req_indexes.clear();
      }
      // Make a trace condition set for each one of them
      // Note for control replication, we're just letting multiple shards 
      // race to their equivalence sets, whichever one gets there first for
      // their fields will be the one to own the preconditions
      std::vector<RtEvent> ready_events;
      conditions.reserve(current_sets.size()); 
      RegionTreeForest *forest = trace->runtime->forest;
      std::map<EquivalenceSet*,unsigned>::const_iterator req_it =
        parent_req_indexes.begin();
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            current_sets.begin(); it != current_sets.end(); it++, req_it++)
      {
#ifdef DEBUG_LEGION
        assert(req_it->first == it->first);
#endif
        TraceConditionSet *condition = new TraceConditionSet(trace, forest,
           req_it->second, it->first->set_expr, it->second, it->first->tree_id);
        condition->add_reference();
        // This looks redundant because it is a bit since we're just going
        // to compute the single equivalence set we already have here but
        // really what we're doing here is registering the condition with 
        // the VersionManager that owns this equivalence set which is a
        // necessary thing for us to do
        const RtEvent ready = 
          condition->recompute_equivalence_sets(opid, it->second);
        if (ready.exists())
          ready_events.push_back(ready);
        condition->capture(it->first, it->second, ready_events);
        conditions.push_back(condition);
      }
      // Wait for the conditions to be ready and then test them for subsumption
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        ready_events.clear();
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      TraceViewSet::FailedPrecondition condition;
      // Need this lock in case we invalidate empty conditions
      AutoLock tpl_lock(template_lock);
      for (std::vector<TraceConditionSet*>::iterator it =
            conditions.begin(); it != conditions.end(); /*nothing*/)
      {
        if ((*it)->is_empty())
        {
          (*it)->invalidate_equivalence_sets();
          if ((*it)->remove_reference())
            delete (*it);
          it = conditions.erase(it);
          continue;
        }
        bool not_subsumed = true;
        if (!(*it)->is_replayable(not_subsumed, &condition))
        {
          if (trace->runtime->dump_physical_traces)
          {
            if (not_subsumed)
              return Replayable(
                  false, "precondition not subsumed: " +
                    condition.to_string(trace->logical_trace->context));
            else
              return Replayable(
               false, "postcondition anti dependent: " +
                 condition.to_string(trace->logical_trace->context));
          }
          else
          {
            if (not_subsumed)
              return Replayable(
                  false, "precondition not subsumed by postcondition");
            else
              return Replayable(
                  false, "postcondition anti dependent");
          }
        }
        it++;
      }
      return Replayable(true);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::register_operation(MemoizableOp *memoizable)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      const TraceLocalID tid = memoizable->get_trace_local_id();
      // Should be able to call back() without the lock even when
      // operations are being removed from the front
      std::map<TraceLocalID,MemoizableOp*> &ops = operations.back();
#ifdef DEBUG_LEGION
      assert(ops.find(tid) == ops.end());
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
      // should be able to read front() even while new maps for operations 
      // are begin appended to the back of 'operations'
      std::map<TraceLocalID,MemoizableOp*> &ops = operations.front();
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
    void PhysicalTemplate::finalize(InnerContext *context, UniqueID opid,
                                    bool has_blocking_call, ReplTraceOp *op)
    //--------------------------------------------------------------------------
    {
      recording = false;
      replayable = check_replayable(op, context, opid, has_blocking_call);

      if (!replayable)
      {
        if (trace->runtime->dump_physical_traces)
        {
          optimize(op, true/*do transitive reduction inline*/);
          dump_template();
        }
        return;
      }
      optimize(op, false/*do transitive reduction inline*/);
      std::fill(events.begin(), events.end(), ApEvent::NO_AP_EVENT);
      event_map.clear();
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
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::optimize(ReplTraceOp *op,
                                    bool do_transitive_reduction)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
      std::vector<RtEvent> frontier_events;
      find_all_last_instance_user_events(frontier_events);
      std::vector<unsigned> gen;
      if (!trace->perform_fence_elision)
      {
        compute_frontiers(frontier_events);
        gen.resize(events.size(), 0/*fence instruction*/);
        for (unsigned idx = 0; idx < instructions.size(); ++idx)
          gen[idx] = idx;
      }
      else
        elide_fences(gen, frontier_events);
      // Check to see if the indirection fields for any across copies are
      // mutated during the execution of the trace. If they aren't then we
      // know that we don't need to recompute preimages on back-to-back replays
      // Do this here so we can also use the 'sync_compute_frontiers' barrier
      // to know that all these analyses are done as well
      if (!across_copies.empty())
      {
        for (std::vector<IssueAcross*>::const_iterator it =
              across_copies.begin(); it != across_copies.end(); it++)
        {
          std::map<unsigned,InstUsers>::iterator finder =
            src_indirect_insts.find((*it)->lhs);
          if ((finder != src_indirect_insts.end()) &&
              are_read_only_users(finder->second))
            (*it)->executor->record_trace_immutable_indirection(true/*src*/);
          finder = dst_indirect_insts.find((*it)->lhs);
          if ((finder != dst_indirect_insts.end()) &&
              are_read_only_users(finder->second))
            (*it)->executor->record_trace_immutable_indirection(false/*dst*/);
        }
        across_copies.clear();
      }
      // Sync the frontier computation so we know that all our frontier data
      // structures such as 'local_frontiers' and 'remote_frontiers' are ready
      sync_compute_frontiers(op, frontier_events);
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
      // After elide fences we can clear these views
      op_insts.clear();
      copy_insts.clear();
      mutated_insts.clear();
      src_indirect_insts.clear();
      dst_indirect_insts.clear();
      instance_last_users.clear();
      // We don't need the expression or view references anymore
      for (std::map<DistributedID,IndividualView*>::const_iterator it =
            recorded_views.begin(); it != recorded_views.end(); it++)
        if (it->second->remove_base_valid_ref(TRACE_REF))
          delete it->second;
      recorded_views.clear();
      for (std::set<IndexSpaceExpression*>::const_iterator it =
           recorded_expressions.begin(); it != recorded_expressions.end(); it++)
        if ((*it)->remove_base_expression_reference(TRACE_REF))
          delete (*it);
      recorded_expressions.clear();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::find_all_last_instance_user_events(
                                          std::vector<RtEvent> &frontier_events)
    //--------------------------------------------------------------------------
    {
      for (std::map<TraceLocalID,InstUsers>::const_iterator it =
            op_insts.begin(); it != op_insts.end(); it++)
        find_last_instance_events(it->second, frontier_events);
      for (std::map<unsigned,InstUsers>::const_iterator it =
            copy_insts.begin(); it != copy_insts.end(); it++)
        find_last_instance_events(it->second, frontier_events);
      for (std::map<unsigned,InstUsers>::const_iterator it =
            src_indirect_insts.begin(); it != src_indirect_insts.end(); it++)
        find_last_instance_events(it->second, frontier_events);
      for (std::map<unsigned,InstUsers>::const_iterator it =
            dst_indirect_insts.begin(); it != dst_indirect_insts.end(); it++)
        find_last_instance_events(it->second, frontier_events);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::find_last_instance_events(const InstUsers &users,
                                          std::vector<RtEvent> &frontier_events)
    //--------------------------------------------------------------------------
    {
      for (InstUsers::const_iterator uit =
            users.begin(); uit != users.end(); uit++)
      {
        std::deque<LastUserResult> &results =
          instance_last_users[uit->instance];
        // Scan through all the queries we've done so far for this instance
        // and see if we've already done one for these parameters
        bool found = false;
        for (std::deque<LastUserResult>::const_iterator it =
              results.begin(); it != results.end(); it++)
        {
          if (!it->user.matches(*uit))
            continue;
          found = true;
          break;
        }
        if (!found)
        {
          results.emplace_back(LastUserResult(*uit));
          LastUserResult &result = results.back();
          std::map<DistributedID,IndividualView*>::const_iterator finder =
            recorded_views.find(uit->instance.view_did);
          RtEvent ready;
          PhysicalManager *manager = 
            trace->runtime->find_or_request_instance_manager(
                                uit->instance.inst_did, ready);
#ifdef DEBUG_LEGION
          assert(finder != recorded_views.end());
#endif
          if (ready.exists() && !ready.has_triggered())
            ready.wait();
          // Query the view for the events that it needs
          // Note that if we're not performing actual fence elision
          // we switch the usage to full read-write privileges so 
          // that we can capture all dependences for the end of the trace
          if (!trace->perform_fence_elision)
          {
            const RegionUsage usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
            finder->second->find_last_users(manager, result.events, usage,
                uit->mask, uit->expr, frontier_events);
          }
          else
            finder->second->find_last_users(manager, result.events,
                uit->usage, uit->mask, uit->expr, frontier_events);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::compute_frontiers(
                                          std::vector<RtEvent> &frontier_events)
    //--------------------------------------------------------------------------
    {
      // We need to wait for all the last user instance events to be ready
      if (!frontier_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(frontier_events);
        frontier_events.clear();
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      // Now we can convert all the results to frontiers
      std::map<ApEvent,unsigned> frontier_map;
      for (std::map<UniqueInst,std::deque<LastUserResult> >::iterator lit =
           instance_last_users.begin(); lit != instance_last_users.end(); lit++)
      {
        for (std::deque<LastUserResult>::iterator uit =
              lit->second.begin(); uit != lit->second.end(); uit++)
        {
          // For each event convert it into a frontier
          for (std::set<ApEvent>::const_iterator it =
                uit->events.begin(); it != uit->events.end(); it++)
          {
            std::map<ApEvent,unsigned>::const_iterator finder =
              frontier_map.find(*it);
            if (finder == frontier_map.end())
            {
              unsigned index = find_frontier_event(*it, frontier_events);
              uit->frontiers.push_back(index);
              frontier_map[*it] = index;
            }
            else
              uit->frontiers.push_back(finder->second);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    unsigned PhysicalTemplate::find_frontier_event(ApEvent event,
                                             std::vector<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      // Check to see if it is an event we know about
      std::map<ApEvent,unsigned>::const_iterator finder = event_map.find(event);
      // If it's not an event we recognize we can just return the start event
      if (finder == event_map.end())
        return 0;
#ifdef DEBUG_LEGION
      assert(frontiers.find(finder->second) == frontiers.end());
#endif
      // Make a new frontier event
      const unsigned next_event_id = events.size();
      frontiers[finder->second] = next_event_id;
      events.resize(next_event_id + 1);
      return next_event_id;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::elide_fences(std::vector<unsigned> &gen,
                                        std::vector<RtEvent> &ready_events) 
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
              CompleteReplay *complete = (*it)->as_complete_replay();
              InstructionKind generator_kind =
                instructions[complete->pre]->get_kind();
              num_merges += (generator_kind != MERGE_EVENT) ? 1 : 0;
              generator_kind = instructions[complete->post]->get_kind(); 
              num_merges += (generator_kind != MERGE_EVENT) ? 1 : 0;
              break;
            }
          default:
            {
              break;
            }
        }

      unsigned merge_starts = events.size();
      events.resize(events.size() + num_merges);

      compute_frontiers(ready_events);

      // We are now going to break the invariant that
      // the generator of events[idx] is instructions[idx].
      // After fence elision, the generator of events[idx] is
      // instructions[gen[idx]].
      gen.resize(events.size(), 0/*fence instruction*/);
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
              std::map<TraceLocalID, InstUsers>::iterator finder =
                op_insts.find(replay->owner);
              if (finder == op_insts.end()) break;
              std::set<unsigned> users;
              find_all_last_users(finder->second, users);
              rewrite_preconditions(replay->pre, users, instructions, 
                  new_instructions, gen, merge_starts);
              rewrite_preconditions(replay->post, users, instructions, 
                  new_instructions, gen, merge_starts);
              break;
            }
          case ISSUE_COPY:
            {
              IssueCopy *copy = inst->as_issue_copy();
              std::map<unsigned, InstUsers>::iterator finder =
                copy_insts.find(copy->lhs);
#ifdef DEBUG_LEGION
              assert(finder != copy_insts.end());
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
              std::map<unsigned, InstUsers>::iterator finder =
                copy_insts.find(fill->lhs);
#ifdef DEBUG_LEGION
              assert(finder != copy_insts.end());
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
              std::map<unsigned, InstUsers>::iterator finder =
                copy_insts.find(across->lhs);
#ifdef DEBUG_LEGION
              assert(finder != copy_insts.end());
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
                finder = src_indirect_insts.find(across->lhs);
#ifdef DEBUG_LEGION
                assert(finder != src_indirect_insts.end());
#endif
                find_all_last_users(finder->second, users);
                rewrite_preconditions(across->src_indirect_precondition, users,
                    instructions, new_instructions, gen, merge_starts);
              }
              if (across->dst_indirect_precondition != 0)
              {
                users.clear();
                finder = dst_indirect_insts.find(across->lhs);
#ifdef DEBUG_LEGION
                assert(finder != dst_indirect_insts.end());
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
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::rewrite_preconditions(
                              unsigned &precondition, std::set<unsigned> &users,
                              const std::vector<Instruction*> &instructions,
                              std::vector<Instruction*> &new_instructions,
                              std::vector<unsigned> &gen,
                              unsigned &merge_starts)
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
          case BARRIER_ARRIVAL:
            {
              BarrierArrival *arrival = inst->as_barrier_arrival();
              used[gen[arrival->rhs]] = true;
              break;
            }
          case ISSUE_COPY:
            {
              IssueCopy *copy = inst->as_issue_copy();
              used[gen[copy->precondition_idx]] = true;
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
          case ISSUE_FILL:
            {
              IssueFill *fill = inst->as_issue_fill();
              used[gen[fill->precondition_idx]] = true;
              break;
            }
          case COMPLETE_REPLAY:
            {
              CompleteReplay *complete = inst->as_complete_replay();
              used[gen[complete->pre]] = true;
              used[gen[complete->post]] = true;
              break;
            }
          case GET_TERM_EVENT:
          case REPLAY_MAPPING:
          case CREATE_AP_USER_EVENT:
          case SET_OP_SYNC_EVENT:
          case ASSIGN_FENCE_COMPLETION:
          case BARRIER_ADVANCE:
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
      record_used_frontiers(used, gen); 

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
      initialize_generators(new_gen);
      std::vector<Instruction*> new_instructions;
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
        if (used[idx])
        {
          Instruction *inst = instructions[idx];
          if (trace->perform_fence_elision)
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
    void PhysicalTemplate::record_used_frontiers(std::vector<bool> &used,
                                         const std::vector<unsigned> &gen) const
    //--------------------------------------------------------------------------
    {
      for (std::map<unsigned,unsigned>::const_iterator it =
            frontiers.begin(); it != frontiers.end(); it++)
        used[gen[it->first]] = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::sync_compute_frontiers(ReplTraceOp *op,
                                    const std::vector<RtEvent> &frontier_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(op == NULL);
      assert(frontier_events.empty());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::initialize_generators(std::vector<unsigned> &new_gen)
    //--------------------------------------------------------------------------
    {
      for (std::map<unsigned, unsigned>::iterator it = 
            frontiers.begin(); it != frontiers.end(); ++it)
        new_gen[it->second] = 0;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::initialize_eliminate_dead_code_frontiers(
                      const std::vector<unsigned> &gen, std::vector<bool> &used)
    //--------------------------------------------------------------------------
    {
      for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
          it != frontiers.end(); ++it)
      {
        unsigned g = gen[it->first];
        if (g != -1U && g < instructions.size())
          used[g] = true;
      }
      // Don't eliminate the last fence instruction
      if (last_fence != NULL)
        used[gen[last_fence->lhs]] = true;
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
      for (std::map<TraceLocalID,std::pair<unsigned,unsigned> >::const_iterator
            it = memo_entries.begin(); it != memo_entries.end(); ++it)
      {
        unsigned slice_index = -1U;
        if (!round_robin_for_tasks && 
            (it->second.second == Operation::TASK_OP_KIND) &&
            (it->first.index_point.get_dim() > 0))
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

        if (inst->get_kind() == MERGE_EVENT)
        {
          MergeEvent *merge = inst->as_merge_event();
          unsigned crossing_found = false;
          std::set<unsigned> new_rhs;
          for (std::set<unsigned>::iterator it = merge->rhs.begin();
               it != merge->rhs.end(); ++it)
          {
            unsigned rh = *it;
            // Don't need to worry about crossing events for the fence
            // initialization as we know it's always set before any 
            // slices executes (rh == 0)
            if ((rh == 0) || (gen[rh] == 0))
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
          switch (inst->get_kind())
          {
            case TRIGGER_EVENT:
              {
                parallelize_replay_event(inst->as_trigger_event()->rhs,
                    slice_index, gen, slice_indices_by_inst,
                    crossing_counts, crossing_instructions);
                break;
              }
            case BARRIER_ARRIVAL:
              {
                parallelize_replay_event(inst->as_barrier_arrival()->rhs,
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
            case COMPLETE_REPLAY:
              {
                parallelize_replay_event(inst->as_complete_replay()->pre,
                    slice_index, gen, slice_indices_by_inst,
                    crossing_counts, crossing_instructions);
                parallelize_replay_event(inst->as_complete_replay()->post,
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
      // If this is the zero event, then don't even bother, we know the 
      // fence event is set before all the slices replay anyway
      if (event_to_check == 0)
        return;
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
    void PhysicalTemplate::initialize_transitive_reduction_frontiers(
       std::vector<unsigned> &topo_order, std::vector<unsigned> &inv_topo_order)
    //--------------------------------------------------------------------------
    {
      for (std::map<unsigned, unsigned>::iterator it = 
            frontiers.begin(); it != frontiers.end(); ++it)
      {
        inv_topo_order[it->second] = topo_order.size();
        topo_order.push_back(it->second);
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

      initialize_transitive_reduction_frontiers(topo_order, inv_topo_order);

      std::map<TraceLocalID, GetTermEvent*> term_insts;
      std::map<TraceLocalID, ReplayMapping*> replay_insts;
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        switch (inst->get_kind())
        {
          // Pass these instructions as their events will be added later
          case GET_TERM_EVENT :
            {
              GetTermEvent *term = inst->as_get_term_event();
              term_insts[term->owner] = term; 
              break;
            }
          case REPLAY_MAPPING:
            {
              ReplayMapping *replay = inst->as_replay_mapping();
              replay_insts[inst->owner] = replay;
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
          case BARRIER_ARRIVAL:
            {
              BarrierArrival *arrival = inst->as_barrier_arrival();
              incoming[arrival->lhs].push_back(arrival->rhs);
              outgoing[arrival->rhs].push_back(arrival->lhs);
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
          case BARRIER_ADVANCE:
            {
              BarrierAdvance *advance = inst->as_barrier_advance();
              inv_topo_order[advance->lhs] = topo_order.size();
              topo_order.push_back(advance->lhs);
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
              // Check to see if we can find a replay instruction to match 
              std::map<TraceLocalID,ReplayMapping*>::iterator replay_finder =
                replay_insts.find(replay->owner);
              if (replay_finder != replay_insts.end())
              {
                incoming[replay_finder->second->lhs].push_back(replay->pre);
                outgoing[replay->pre].push_back(replay_finder->second->lhs);
                replay_insts.erase(replay_finder);
              }
              // Lastly check to see if we can find a term inst to match
              std::map<TraceLocalID,GetTermEvent*>::iterator term_finder =
                term_insts.find(replay->owner);
              if (term_finder != term_insts.end())
              {
                if (replay->post != 0)
                {
                  incoming[term_finder->second->lhs].push_back(replay->post);
                  outgoing[replay->post].push_back(term_finder->second->lhs);
                  term_insts.erase(term_finder);
                }
                else if (replay->pre != 0)
                {
                  incoming[term_finder->second->lhs].push_back(replay->pre);
                  outgoing[replay->pre].push_back(term_finder->second->lhs);
                  term_insts.erase(term_finder);
                }
              }
              break;
            }
          default:
            {
              assert(false);
              break;
            }
        }
      }
#ifdef DEBUG_LEGION
      // should have seen a complete replay instruction for every replay mapping
      assert(replay_insts.empty());
#endif
      if (!term_insts.empty())
      {
        // Any term instructions that don't match with a complete replay
        // need to be recorded as sources for the BFS
        for (std::map<TraceLocalID,GetTermEvent*>::const_iterator it =
              term_insts.begin(); it != term_insts.end(); it++)
        {
          inv_topo_order[it->second->lhs] = topo_order.size();
          topo_order.push_back(it->second->lhs);
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
#ifdef DEBUG_LEGION
      for (unsigned idx = 0; idx < incoming.size(); idx++)
        assert(remaining_edges[idx] == 0);
#endif

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
        // Write them to the members (in this order)
        pending_inv_topo_order.store(inv_topo_order_copy);
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
      std::map<unsigned,unsigned> substitutions;
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
            substitutions[merge->lhs] = *merge->rhs.begin();
#ifdef DEBUG_LEGION
            assert(merge->lhs != substitutions[merge->lhs]);
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

      // Rewrite the frontiers first
      rewrite_frontiers(substitutions); 

      // Then rewrite the instructions
      instructions.swap(new_instructions);

      std::vector<unsigned> new_gen((gen == NULL) ? 0 : gen->size(), -1U);
      if (gen != NULL)
        initialize_generators(new_gen);

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
          case REPLAY_MAPPING:
            {
              ReplayMapping *replay = inst->as_replay_mapping();
              lhs = replay->lhs;
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
              std::map<unsigned,unsigned>::const_iterator finder =
                substitutions.find(trigger->rhs);
              if (finder != substitutions.end())
                trigger->rhs = finder->second;
              break;
            }
          case BARRIER_ARRIVAL:
            {
              BarrierArrival *arrival = inst->as_barrier_arrival();
              std::map<unsigned,unsigned>::const_iterator finder =
                substitutions.find(arrival->rhs);
              if (finder != substitutions.end())
                arrival->rhs = finder->second;
              lhs = arrival->lhs;
              break;
            }
          case MERGE_EVENT:
            {
              MergeEvent *merge = inst->as_merge_event();
              std::set<unsigned> new_rhs;
              for (std::set<unsigned>::iterator it = merge->rhs.begin();
                   it != merge->rhs.end(); ++it)
              {
                std::map<unsigned,unsigned>::const_iterator finder =
                  substitutions.find(*it);
                if (finder != substitutions.end())
                  new_rhs.insert(finder->second);
                else
                  new_rhs.insert(*it);
              }
              merge->rhs.swap(new_rhs);
              lhs = merge->lhs;
              break;
            }
          case ISSUE_COPY:
            {
              IssueCopy *copy = inst->as_issue_copy();
              std::map<unsigned,unsigned>::const_iterator finder =
                substitutions.find(copy->precondition_idx);
              if (finder != substitutions.end())
                copy->precondition_idx = finder->second;
              lhs = copy->lhs;
              break;
            }
          case ISSUE_FILL:
            {
              IssueFill *fill = inst->as_issue_fill();
              std::map<unsigned,unsigned>::const_iterator finder =
                substitutions.find(fill->precondition_idx);
              if (finder != substitutions.end())
                fill->precondition_idx = finder->second;
              lhs = fill->lhs;
              break;
            }
          case ISSUE_ACROSS:
            {
              IssueAcross *across = inst->as_issue_across();
              std::map<unsigned,unsigned>::const_iterator finder =
                substitutions.find(across->copy_precondition);
              if (finder != substitutions.end())
                across->copy_precondition = finder->second;
              if (across->collective_precondition != 0)
              {
                finder = substitutions.find(across->collective_precondition);
                if (finder != substitutions.end())
                  across->collective_precondition = finder->second;
              }
              if (across->src_indirect_precondition != 0)
              {
                finder = substitutions.find(across->src_indirect_precondition);
                if (finder != substitutions.end())
                  across->src_indirect_precondition = finder->second;
              }
              if (across->dst_indirect_precondition != 0)
              {
                finder = substitutions.find(across->dst_indirect_precondition);
                if (finder != substitutions.end())
                  across->dst_indirect_precondition = finder->second;
              }
              lhs = across->lhs;
              break;
            }
          case SET_OP_SYNC_EVENT:
            {
              SetOpSyncEvent *sync = inst->as_set_op_sync_event();
              lhs = sync->lhs;
              break;
            }
          case BARRIER_ADVANCE:
            {
              BarrierAdvance *advance = inst->as_barrier_advance();
              lhs = advance->lhs;
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
              std::map<unsigned,unsigned>::const_iterator finder =
                substitutions.find(replay->pre);
              if (finder != substitutions.end())
                replay->pre = finder->second;
              finder = substitutions.find(replay->post);
              if (finder != substitutions.end())
                replay->post = finder->second;
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
    void PhysicalTemplate::rewrite_frontiers(
                                     std::map<unsigned,unsigned> &substitutions)
    //--------------------------------------------------------------------------
    {
      std::vector<std::pair<unsigned,unsigned> > to_add;
      for (std::map<unsigned,unsigned>::iterator it =
            frontiers.begin(); it != frontiers.end(); /*nothing*/)
      {
        std::map<unsigned,unsigned>::const_iterator finder =
          substitutions.find(it->first);
        if (finder != substitutions.end())
        {
          to_add.emplace_back(std::make_pair(finder->second,it->second));
          std::map<unsigned,unsigned>::iterator to_delete = it++;
          frontiers.erase(to_delete);
        }
        else
          it++;
      }
      for (std::vector<std::pair<unsigned,unsigned> >::const_iterator it =
            to_add.begin(); it != to_add.end(); it++)
      {
        std::map<unsigned,unsigned>::const_iterator finder =
          frontiers.find(it->first);
        if (finder != frontiers.end())
        {
          // Handle the case where we recorded two different frontiers
          // but they are now being merged together from the same source
          // and we can therefore substitute the first one for the second
#ifdef DEBUG_LEGION
          assert(substitutions.find(it->second) == substitutions.end());
#endif
          substitutions[it->second] = finder->second;
        }
        else
          frontiers.insert(*it);
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
        // We only eliminate two kinds of instructions currently:
        // GetTermEvent and SetOpSyncEvent
        used[idx] = (kind != SET_OP_SYNC_EVENT) && (kind != GET_TERM_EVENT);
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
          case COMPLETE_REPLAY:
            {
              CompleteReplay *complete = inst->as_complete_replay();
#ifdef DEBUG_LEGION
              assert(gen[complete->pre] != -1U);
              assert(gen[complete->post] != -1U);
#endif
              used[gen[complete->pre]] = true;
              used[gen[complete->post]] = true;
              break;
            }
          case BARRIER_ARRIVAL:
            {
              BarrierArrival *arrival = inst->as_barrier_arrival();
#ifdef DEBUG_LEGION
              assert(gen[arrival->rhs] != -1U);
#endif
              used[gen[arrival->rhs]] = true;
              break;
            }
          case GET_TERM_EVENT:
          case REPLAY_MAPPING:
          case CREATE_AP_USER_EVENT:
          case SET_OP_SYNC_EVENT:
          case ASSIGN_FENCE_COMPLETION:
          case BARRIER_ADVANCE:
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
      initialize_eliminate_dead_code_frontiers(gen, used);

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
      initialize_generators(new_gen);
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
      InnerContext *ctx = trace->logical_trace->context;
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
      dump_sharded_template();

      log_tracing.info() << "[Precondition]";
      for (std::vector<TraceConditionSet*>::const_iterator it =
            conditions.begin(); it != conditions.end(); it++)
        (*it)->dump_preconditions();

      log_tracing.info() << "[Anticondition]";
      for (std::vector<TraceConditionSet*>::const_iterator it =
            conditions.begin(); it != conditions.end(); it++)
        (*it)->dump_anticonditions();

      log_tracing.info() << "[Postcondition]";
      for (std::vector<TraceConditionSet*>::const_iterator it =
            conditions.begin(); it != conditions.end(); it++)
        (*it)->dump_postconditions();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::dump_instructions(
                                  const std::vector<Instruction*> &instructions)
    //--------------------------------------------------------------------------
    {
      for (std::vector<Instruction*>::const_iterator it = instructions.begin();
           it != instructions.end(); ++it)
        log_tracing.info() << "  " << (*it)->to_string(memo_entries);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::pack_recorder(Serializer &rez,
                                         std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      rez.serialize(trace->runtime->address_space);
      rez.serialize(this);
      RtUserEvent remote_applied = Runtime::create_rt_user_event();
      rez.serialize(remote_applied);
      applied_events.insert(remote_applied);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_premap_output(MemoizableOp *memo,
                                         const Mapper::PremapTaskOutput &output,
                                         std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      const TraceLocalID op_key = memo->get_trace_local_id();
      AutoLock t_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
      assert(!output.reduction_futures.empty());
      assert(cached_premappings.find(op_key) == cached_premappings.end());
#endif
      CachedPremapping &premapping = cached_premappings[op_key];
      premapping.future_locations = output.reduction_futures;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::get_premap_output(IndexTask *task,
                                          std::vector<Memory> &future_locations)
    //--------------------------------------------------------------------------
    {
      TraceLocalID op_key = task->get_trace_local_id();
      AutoLock t_lock(template_lock, 1, false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(is_replaying());
#endif
      CachedPremappings::const_iterator finder = 
        cached_premappings.find(op_key);
#ifdef DEBUG_LEGION
      assert(finder != cached_premappings.end());
#endif
      future_locations = finder->second.future_locations;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_mapper_output(const TraceLocalID &tlid,
                                            const Mapper::MapTaskOutput &output,
                              const std::deque<InstanceSet> &physical_instances,
                              const std::vector<size_t> &future_size_bounds,
                              const std::vector<TaskTreeCoordinates> &coords,
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
      mapping.future_locations = output.future_locations;
      mapping.future_size_bounds = future_size_bounds;
      // Check to see if the future coordinates are inside of our trace
      // They have to be inside of our trace in order for it to be safe
      // for use to be able to re-use their upper bound sizes (because
      // we know those tasks are reusing the same variants)
      for (unsigned idx = 0; idx < future_size_bounds.size(); idx++)
      {
        // If there's no upper bound then no need to check if the
        // future is inside 
        if (future_size_bounds[idx] == SIZE_MAX)
          continue;
        const TaskTreeCoordinates &future_coords = coords[idx];
#ifdef DEBUG_LEGION
        assert(future_coords.size() <= coordinates.size()); 
#endif
        if (future_coords.empty() ||
            (future_coords.size() < coordinates.size()))
        {
          mapping.future_size_bounds[idx] = SIZE_MAX;
          continue;
        }
#ifdef DEBUG_LEGION
#ifndef NDEBUG
        // If the size of the coordinates are the same we better
        // be inside the same parent task or something is really wrong
        for (unsigned idx2 = 0; idx2 < (future_coords.size()-1); idx2++)
          assert(future_coords[idx2] == coordinates[idx2]);
#endif
#endif
        // check to see if it came after the start of the trace
        unsigned last = future_coords.size() - 1;
        if (coordinates[last].context_index <=future_coords[last].context_index)
          continue;
        // Otherwise not inside the trace and therefore we cannot
        // record the bounds for the future
        mapping.future_size_bounds[idx] = SIZE_MAX;
      }
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
                              std::vector<Memory> &future_locations,
                              std::vector<size_t> &future_size_bounds,
                              std::deque<InstanceSet> &physical_instances) const
    //--------------------------------------------------------------------------
    {
      TraceLocalID op_key = task->get_trace_local_id();
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
      future_locations = finder->second.future_locations;
      future_size_bounds = finder->second.future_size_bounds;
      physical_instances = finder->second.physical_instances;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_completion_event(ApEvent lhs,
                                     unsigned op_kind, const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      const bool fence = (op_kind == Operation::FENCE_OP_KIND);
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      unsigned lhs_ = convert_event(lhs);
      record_memo_entry(tlid, lhs_, op_kind);
      insert_instruction(new GetTermEvent(*this, lhs_, tlid, fence));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_replay_mapping(ApEvent lhs,
                 unsigned op_kind, const TraceLocalID &tlid, bool register_memo)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      unsigned lhs_ = convert_event(lhs);
      if (register_memo)
        record_memo_entry(tlid, lhs_, op_kind);
      insert_instruction(new ReplayMapping(*this, lhs_, tlid));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::request_term_event(ApUserEvent &term_event)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!term_event.exists() || term_event.has_triggered());
#endif
      term_event = Runtime::create_ap_user_event(NULL);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_create_ap_user_event(
                                     ApUserEvent &lhs, const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      // Make the event here so it is on our local node
      // Note this is important for control replications where the
      // convert_event method will check this property
      lhs = Runtime::create_ap_user_event(NULL);
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      unsigned lhs_ = convert_event(lhs);
      user_events[lhs_] = lhs;
      insert_instruction(new CreateApUserEvent(*this, lhs_, tlid));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_trigger_event(ApUserEvent lhs, ApEvent rhs,
                                                const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs.exists());
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      // Do this first in case it gets pre-empted
      const unsigned rhs_ = 
        rhs.exists() ? find_event(rhs, tpl_lock) : fence_completion_id;
#ifdef DEBUG_LEGION
      // Make sure we're always recording user events on the same shard
      // where the create user event is recorded
      unsigned lhs_ = UINT_MAX;
      for (std::map<unsigned,ApUserEvent>::const_iterator it =
            user_events.begin(); it != user_events.end(); it++)
      {
        if (it->second != lhs)
          continue;
        lhs_ = it->first;
        break;
      }
      assert(lhs_ != UINT_MAX);
#else
      unsigned lhs_ = find_event(lhs, tpl_lock);
#endif
      events.push_back(ApEvent());
      insert_instruction(new TriggerEvent(*this, lhs_, rhs_, tlid));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs, ApEvent rhs_,
                                               const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      std::vector<ApEvent> rhs(1, rhs_);
      record_merge_events(lhs, rhs, tlid);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs, ApEvent e1,
                                           ApEvent e2, const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      std::vector<ApEvent> rhs(2);
      rhs[0] = e1;
      rhs[1] = e2;
      record_merge_events(lhs, rhs, tlid);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs, ApEvent e1,
                                               ApEvent e2, ApEvent e3,
                                               const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      std::vector<ApEvent> rhs(3);
      rhs[0] = e1;
      rhs[1] = e2;
      rhs[2] = e3;
      record_merge_events(lhs, rhs, tlid);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs,
                                               const std::set<ApEvent>& rhs,
                                               const TraceLocalID &tlid)
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
        std::map<ApEvent,unsigned>::const_iterator finder = event_map.find(*it);
        if (finder != event_map.end())
          rhs_.insert(finder->second);
      }
      if (rhs_.size() == 0)
        rhs_.insert(fence_completion_id);

#ifndef LEGION_DISABLE_EVENT_PRUNING
      if (!lhs.exists() || (rhs.find(lhs) != rhs.end()))
      {
        ApUserEvent rename = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, rename, lhs);
        lhs = rename;
      }
#endif

      insert_instruction(new MergeEvent(*this, convert_event(lhs), rhs_, tlid));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs,
                                               const std::vector<ApEvent>& rhs,
                                               const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      std::set<unsigned> rhs_;
      for (std::vector<ApEvent>::const_iterator it =
            rhs.begin(); it != rhs.end(); it++)
      {
        std::map<ApEvent,unsigned>::const_iterator finder = event_map.find(*it);
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

      insert_instruction(new MergeEvent(*this, convert_event(lhs), rhs_, tlid));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(PredEvent &lhs, PredEvent e1,
                                         PredEvent e2, const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      // need support for predicated execution with tracing
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_collective_barrier(ApBarrier bar, 
              ApEvent pre, const std::pair<size_t,size_t> &key, size_t arrivals)
    //--------------------------------------------------------------------------
    {
      // should only be called on sharded physical templates
      assert(false);
    }

    //--------------------------------------------------------------------------
    ShardID PhysicalTemplate::record_managed_barrier(ApBarrier bar,
                                                     size_t total_arrivals)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(bar.exists());
      assert(is_recording());
      assert(event_map.find(bar) == event_map.end());
      assert(managed_barriers.find(bar) == managed_barriers.end());
      const unsigned lhs = convert_event(bar, false/*check*/);
#else
      const unsigned lhs = convert_event(bar);
#endif
      BarrierAdvance *advance =
        new BarrierAdvance(*this, bar, lhs, total_arrivals, true/*owner*/);
      insert_instruction(advance);
      // Save this as one of the barriers that we're managing
      managed_barriers[bar] = advance;
      return 0; // No bothering with shards here
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_barrier_arrival(ApBarrier bar, ApEvent pre,
        size_t arrivals, std::set<RtEvent> &applied_events, ShardID owner_shard)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Should only be seeing things from ourself here
      assert(owner_shard == 0);
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(bar.exists());
      assert(is_recording());
#endif
      const unsigned rhs = 
        pre.exists() ? find_event(pre, tpl_lock) : fence_completion_id;
      const unsigned lhs = events.size();
      events.push_back(ApEvent());
      BarrierArrival *arrival =
          new BarrierArrival(*this, bar, lhs, rhs, arrivals, true/*managed*/);
      insert_instruction(arrival);
      managed_arrivals[bar].push_back(arrival);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_copy(const TraceLocalID &tlid, 
                                 ApEvent &lhs, IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField>& src_fields,
                                 const std::vector<CopySrcDstField>& dst_fields,
                                 const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                                             RegionTreeID src_tree_id,
                                             RegionTreeID dst_tree_id,
#endif
                                             ApEvent precondition,
                                             PredEvent pred_guard,
                                             LgEvent src_unique,
                                             LgEvent dst_unique,
                                             int priority,
                                             CollectiveKind collective)
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
      // Do this first in case it gets preempted
      const unsigned rhs_ = find_event(precondition, tpl_lock);
      unsigned lhs_ = convert_event(lhs);
      insert_instruction(new IssueCopy(
            *this, lhs_, expr, tlid,
            src_fields, dst_fields, reservations,
#ifdef LEGION_SPY
            src_tree_id, dst_tree_id,
#endif
            rhs_, src_unique, dst_unique, priority, collective)); 
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_fill(const TraceLocalID &tlid, 
                                             ApEvent &lhs,
                                             IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField> &fields,
                                             const void *fill_value, 
                                             size_t fill_size,
#ifdef LEGION_SPY
                                             UniqueID fill_uid,
                                             FieldSpace handle,
                                             RegionTreeID tree_id,
#endif
                                             ApEvent precondition,
                                             PredEvent pred_guard,
                                             LgEvent unique_event,
                                             int priority,
                                             CollectiveKind collective)
    //--------------------------------------------------------------------------
    {
      if (!lhs.exists())
      {
        ApUserEvent rename = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, rename);
        lhs = rename;
      }

      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      // Do this first in case it gets preempted
      const unsigned rhs_ = find_event(precondition, tpl_lock);
      unsigned lhs_ = convert_event(lhs);
      insert_instruction(new IssueFill(*this, lhs_, expr, tlid,
                                       fields, fill_value, fill_size, 
#ifdef LEGION_SPY
                                       fill_uid, handle, tree_id,
#endif
                                       rhs_, unique_event,
                                       priority, collective));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_across(const TraceLocalID &tlid, 
                                              ApEvent &lhs,
                                              ApEvent collective_precondition,
                                              ApEvent copy_precondition,
                                              ApEvent src_indirect_precondition,
                                              ApEvent dst_indirect_precondition,
                                              CopyAcrossExecutor *executor)
    //--------------------------------------------------------------------------
    {
      if (!lhs.exists())
      {
        ApUserEvent rename = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, rename);
        lhs = rename;
      }

      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      unsigned copy_pre = find_event(copy_precondition, tpl_lock);
      unsigned collective_pre = 0, src_indirect_pre = 0, dst_indirect_pre = 0;
      if (collective_precondition.exists())
        collective_pre = find_event(collective_precondition, tpl_lock);
      if (src_indirect_precondition.exists())
        src_indirect_pre = find_event(src_indirect_precondition, tpl_lock);
      if (dst_indirect_precondition.exists())
        dst_indirect_pre = find_event(dst_indirect_precondition, tpl_lock);
      unsigned lhs_ = convert_event(lhs);
      IssueAcross *across = new IssueAcross(*this, lhs_,copy_pre,collective_pre,
       src_indirect_pre, dst_indirect_pre, tlid, executor);
      across_copies.push_back(across);
      insert_instruction(across);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_op_inst(const TraceLocalID &tlid,
                                          unsigned parent_req_index,
                                          const UniqueInst &inst,
                                          RegionNode *node,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          bool update_validity,
                                          std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      if (trace_regions.insert(node, user_mask))
      {
        trace_region_parent_req_indexes[node] = parent_req_index;
        node->add_base_resource_ref(TRACE_REF);
      }
      if (update_validity)
        record_instance_user(op_insts[tlid], inst, usage, 
                             node->row_source, user_mask, applied);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_instance_user(InstUsers &users,
                                                const UniqueInst &instance,
                                                const RegionUsage &usage,
                                                IndexSpaceExpression *expr,
                                                const FieldMask &mask,
                                                std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      if (!IS_READ_ONLY(usage))
        record_mutated_instance(instance, expr, mask, applied);
      for (InstUsers::iterator it = users.begin(); it != users.end(); it++)
      {
        if (!it->matches(instance, usage, expr))
          continue;
        it->mask |= mask;
        return;
      }
      users.emplace_back(InstanceUser(instance, usage, expr, mask));
      if (recorded_views.find(instance.view_did) == recorded_views.end())
      {
        RtEvent ready;
        IndividualView *view = static_cast<IndividualView*>(
            trace->runtime->find_or_request_logical_view(
                                instance.view_did, ready));
        recorded_views[instance.view_did] = view;
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        view->add_base_valid_ref(TRACE_REF);
      }
      if (recorded_expressions.insert(expr).second)
        expr->add_base_expression_reference(TRACE_REF);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_mutated_instance(const UniqueInst &inst,
                              IndexSpaceExpression *expr, const FieldMask &mask,
                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      FieldMaskSet<IndexSpaceExpression> &insts = mutated_insts[inst];
      if (insts.empty() &&
          (recorded_views.find(inst.view_did) == recorded_views.end()))
      {
        RtEvent ready;
        IndividualView *view = static_cast<IndividualView*>(
            trace->runtime->find_or_request_logical_view(inst.view_did, ready));
        recorded_views[inst.view_did] = view;
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        view->add_base_valid_ref(TRACE_REF);
      }
      if (insts.insert(expr,mask) && recorded_expressions.insert(expr).second)
        expr->add_base_expression_reference(TRACE_REF);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_fill_inst(ApEvent lhs,
                                 IndexSpaceExpression *expr,
                                 const UniqueInst &inst,
                                 const FieldMask &fill_mask,
                                 std::set<RtEvent> &applied_events,
                                 const bool reduction_initialization)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      const unsigned lhs_ = find_event(lhs, tpl_lock);
      const RegionUsage usage(LEGION_WRITE_ONLY, LEGION_EXCLUSIVE, 0);
      record_instance_user(copy_insts[lhs_], inst, usage, expr, 
                           fill_mask, applied_events);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_copy_insts(ApEvent lhs, 
                         const TraceLocalID &tlid,
                         unsigned src_idx, unsigned dst_idx,
                         IndexSpaceExpression *expr,
                         const UniqueInst &src_inst, const UniqueInst &dst_inst,
                         const FieldMask &src_mask, const FieldMask &dst_mask,
                         PrivilegeMode src_mode, PrivilegeMode dst_mode,
                         ReductionOpID redop, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      const unsigned lhs_ = find_event(lhs, tpl_lock);
      const RegionUsage src_usage(src_mode, LEGION_EXCLUSIVE, 0);
      const RegionUsage dst_usage(dst_mode, LEGION_EXCLUSIVE, redop);
      record_instance_user(copy_insts[lhs_], src_inst, src_usage,
                           expr, src_mask, applied_events);
      record_instance_user(copy_insts[lhs_], dst_inst, dst_usage,
                           expr, dst_mask, applied_events);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_across_insts(ApEvent lhs, 
                                 const TraceLocalID &tlid,
                                 unsigned src_idx, unsigned dst_idx,
                                 IndexSpaceExpression *expr,
                                 const AcrossInsts &src_insts,
                                 const AcrossInsts &dst_insts,
                                 PrivilegeMode src_mode, PrivilegeMode dst_mode,
                                 bool src_indirect, bool dst_indirect,
                                 std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      const unsigned lhs_ = find_event(lhs, tpl_lock);
      const RegionUsage src_usage(src_mode, LEGION_EXCLUSIVE, 0);
      for (AcrossInsts::const_iterator it =
            src_insts.begin(); it != src_insts.end(); it++)
        record_instance_user(src_indirect ? 
            src_indirect_insts[lhs_] : copy_insts[lhs_],
            it->first, src_usage, expr, it->second, applied_events);
      const RegionUsage dst_usage(dst_mode, LEGION_EXCLUSIVE, 0);
      for (AcrossInsts::const_iterator it =
            dst_insts.begin(); it != dst_insts.end(); it++)
        record_instance_user(dst_indirect ?
            dst_indirect_insts[lhs_] : copy_insts[lhs_],
            it->first, dst_usage, expr, it->second, applied_events);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_indirect_insts(ApEvent indirect_done,
                            ApEvent all_done, IndexSpaceExpression *expr,
                            const AcrossInsts &insts,
                            std::set<RtEvent> &applied, PrivilegeMode privilege)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      const unsigned indirect = find_event(indirect_done, tpl_lock);
      const RegionUsage usage(privilege, LEGION_EXCLUSIVE, 0);
      for (AcrossInsts::const_iterator it = 
            insts.begin(); it != insts.end(); it++)
        record_instance_user(copy_insts[indirect], it->first, 
                             usage, expr, it->second, applied);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_set_op_sync_event(ApEvent &lhs, 
                                                    const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      // Always make a fresh event here for these
      ApUserEvent rename = Runtime::create_ap_user_event(NULL);
      Runtime::trigger_event(NULL, rename, lhs);
      lhs = rename;
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif

      insert_instruction(new SetOpSyncEvent(*this, convert_event(lhs), tlid));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_complete_replay(const TraceLocalID &tlid,
                   ApEvent pre, ApEvent post, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      // Do this first in case it gets preempted
      const unsigned pre_ = pre.exists() ? find_event(pre, tpl_lock) : 0;
      const unsigned post_ = post.exists() ? find_event(post, tpl_lock) : 0;
      events.push_back(ApEvent());
      insert_instruction(new CompleteReplay(*this, tlid, pre_, post_));
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
    void PhysicalTemplate::record_owner_shard(unsigned tid, ShardID owner)
    //--------------------------------------------------------------------------
    {
      // Only called on sharded physical template
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_local_space(unsigned tid, IndexSpace sp)
    //--------------------------------------------------------------------------
    {
      // Only called on sharded physical template
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_sharding_function(unsigned tid, 
                                                    ShardingFunction *function)
    //--------------------------------------------------------------------------
    {
      // Only called on sharded physical template
      assert(false);
    }

    //--------------------------------------------------------------------------
    ShardID PhysicalTemplate::find_owner_shard(unsigned tid)
    //--------------------------------------------------------------------------
    {
      // Only called on sharded physical template
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    IndexSpace PhysicalTemplate::find_local_space(unsigned tid)
    //--------------------------------------------------------------------------
    {
      // Only called on sharded physical template
      assert(false);
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    ShardingFunction* PhysicalTemplate::find_sharding_function(unsigned tid)
    //--------------------------------------------------------------------------
    {
      // Only called on sharded physical template
      assert(false);
      return NULL;
    } 

    //--------------------------------------------------------------------------
    void PhysicalTemplate::initialize_replay(ApEvent completion, 
                                             bool recurrent, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock t_lock(template_lock);
        initialize_replay(completion, recurrent, false/*need lock*/);
        return;
      }
      operations.emplace_back(std::map<TraceLocalID,MemoizableOp*>());
      pending_replays.emplace_back(std::make_pair(completion, recurrent));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::perform_replay(Runtime *runtime, 
                                          std::set<RtEvent> &replayed_events)
    //--------------------------------------------------------------------------
    {
      RtEvent replay_precondition;
      if (total_replays++ == Realm::Barrier::MAX_PHASES)
      {
        replay_precondition = refresh_managed_barriers();
        // Reset it back to one after updating our barriers
        total_replays = 1;
      }
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
        // If it was requested that we dump the traces do that now
        if (runtime->dump_physical_traces)
          dump_template();
      }

      if (recurrent)
      {
        if (last_fence == NULL)
          fence_completion = ApEvent::NO_AP_EVENT;
        else
          fence_completion = completion;
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
            replay_targets[idx % replay_targets.size()], replay_precondition) :
          runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY,
            replay_precondition, replay_targets[idx % replay_targets.size()]);
        replayed_events.insert(done);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalTemplate::refresh_managed_barriers(void)
    //--------------------------------------------------------------------------
    {
      std::map<ShardID,std::map<ApEvent,ApBarrier> > notifications;
      for (std::map<ApEvent,BarrierAdvance*>::const_iterator it =
            managed_barriers.begin(); it != managed_barriers.end(); it++)
        it->second->refresh_barrier(it->first, notifications);
      if (!notifications.empty())
      {
#ifdef DEBUG_LEGION
        assert(notifications.size() == 1);
#endif
        std::map<ShardID,std::map<ApEvent,ApBarrier> >::const_iterator local =
          notifications.begin();
#ifdef DEBUG_LEGION
        assert(local->first == 0);
        assert(local->second.size() == managed_arrivals.size());
#endif
        for (std::map<ApEvent,ApBarrier>::const_iterator it =
              local->second.begin(); it != local->second.end(); it++)
        {
          std::map<ApEvent,std::vector<BarrierArrival*> >::iterator finder =
            managed_arrivals.find(it->first);
#ifdef DEBUG_LEGION
          assert(finder != managed_arrivals.end());
#endif
          for (unsigned idx = 0; idx < finder->second.size(); idx++)
            finder->second[idx]->set_managed_barrier(it->second);
        }
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::finish_replay(std::set<ApEvent> &postconditions)
    //--------------------------------------------------------------------------
    {
      for (std::map<unsigned,unsigned>::const_iterator it =
            frontiers.begin(); it != frontiers.end(); it++)
        postconditions.insert(events[it->first]);
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
                                                               const void *args)
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
    void PhysicalTemplate::record_memo_entry(const TraceLocalID &tlid,
                                             unsigned entry, unsigned op_kind)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo_entries.find(tlid) == memo_entries.end());
#endif
      memo_entries[tlid] = std::pair<unsigned,unsigned>(entry, op_kind);
    }

    //--------------------------------------------------------------------------
#ifdef DEBUG_LEGION
    unsigned PhysicalTemplate::convert_event(const ApEvent &event, bool check)
#else
    inline unsigned PhysicalTemplate::convert_event(const ApEvent &event)
#endif
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
    inline unsigned PhysicalTemplate::find_event(const ApEvent &event, 
                                                 AutoLock &tpl_lock)
    //--------------------------------------------------------------------------
    {
      std::map<ApEvent,unsigned>::const_iterator finder = event_map.find(event);
#ifdef DEBUG_LEGION
      assert(finder != event_map.end());
      assert(finder->second != NO_INDEX);
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
    void PhysicalTemplate::find_all_last_users(const InstUsers &inst_users,
                                               std::set<unsigned> &users) const
    //--------------------------------------------------------------------------
    {
      for (InstUsers::const_iterator uit =
            inst_users.begin(); uit != inst_users.end(); uit++)
      {
        std::map<UniqueInst,std::deque<LastUserResult> >::const_iterator
          finder = instance_last_users.find(uit->instance);
#ifdef DEBUG_LEGION
        assert(finder != instance_last_users.end());
#endif
        for (std::deque<LastUserResult>::const_iterator it =
              finder->second.begin(); it != finder->second.end(); it++)
        {
          if (!it->user.matches(*uit))
            continue;
#ifdef DEBUG_LEGION
          assert(it->events.size() == it->frontiers.size());
#endif
          users.insert(it->frontiers.begin(), it->frontiers.end());
          break;
        }
      }
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::are_read_only_users(InstUsers &inst_users)
    //--------------------------------------------------------------------------
    {
      RegionTreeForest *forest = trace->runtime->forest;
      for (InstUsers::const_iterator vit = 
            inst_users.begin(); vit != inst_users.end(); vit++)
      {
        // Scan through the other users and look for anything overlapping
        LegionMap<UniqueInst,
                  FieldMaskSet<IndexSpaceExpression> >::const_iterator
          finder = mutated_insts.find(vit->instance);
        if (finder == mutated_insts.end())
          continue;
        if (vit->mask * finder->second.get_valid_mask())
          continue;
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              finder->second.begin(); it != finder->second.end(); it++)
        {
          if (vit->mask * it->second)
            continue;
          IndexSpaceExpression *intersect = 
            forest->intersect_index_spaces(vit->expr, it->first);
          if (intersect->is_empty())
            continue;
          // Not immutable
          return false;
        }
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // ShardedPhysicalTemplate
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardedPhysicalTemplate::ShardedPhysicalTemplate(PhysicalTrace *trace,
       ApEvent fence_event, TaskTreeCoordinates &&coords, ReplicateContext *ctx)
      : PhysicalTemplate(trace, fence_event, std::move(coords)), repl_ctx(ctx),
        local_shard(repl_ctx->owner_shard->shard_id), 
        total_shards(repl_ctx->shard_manager->total_shards),
        template_index(repl_ctx->register_trace_template(this)),
        refreshed_barriers(0), next_deferral_precondition(0), 
        recurrent_replays(0), updated_frontiers(0)
    //--------------------------------------------------------------------------
    {
      repl_ctx->add_base_resource_ref(TRACE_REF);
    }

    //--------------------------------------------------------------------------
    ShardedPhysicalTemplate::~ShardedPhysicalTemplate(void)
    //--------------------------------------------------------------------------
    {
      for (std::map<unsigned,ApBarrier>::iterator it = 
            local_frontiers.begin(); it != local_frontiers.end(); it++)
        it->second.destroy_barrier();
      // Unregister ourselves from the context and then remove our reference
      repl_ctx->unregister_trace_template(template_index);
      if (repl_ctx->remove_base_resource_ref(TRACE_REF))
        delete repl_ctx;
    } 

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_merge_events(ApEvent &lhs,
                         const std::set<ApEvent> &rhs, const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      std::set<unsigned> rhs_;
      std::set<RtEvent> wait_for;
      std::vector<ApEvent> pending_events;
      std::map<ApEvent,RtUserEvent> request_events;
      for (std::set<ApEvent>::const_iterator it =
            rhs.begin(); it != rhs.end(); it++)
      {
        if (!it->exists())
          continue;
        std::map<ApEvent,unsigned>::const_iterator finder = event_map.find(*it);
        if (finder == event_map.end())
        {
          // We're going to need to check this event later
          pending_events.push_back(*it);
          // See if anyone else has requested this event yet 
          std::map<ApEvent,RtEvent>::const_iterator request_finder = 
            pending_event_requests.find(*it);
          if (request_finder == pending_event_requests.end())
          {
            const RtUserEvent request_event = Runtime::create_rt_user_event();
            pending_event_requests[*it] = request_event;
            wait_for.insert(request_event);
            request_events[*it] = request_event;
          }
          else
            wait_for.insert(request_finder->second);
        }
        else if (finder->second != NO_INDEX)
          rhs_.insert(finder->second);
      }
      // If we have anything to wait for we need to do that
      if (!wait_for.empty())
      {
        tpl_lock.release();
        // Send any request messages first
        if (!request_events.empty())
        {
          for (std::map<ApEvent,RtUserEvent>::const_iterator it = 
                request_events.begin(); it != request_events.end(); it++)
            request_remote_shard_event(it->first, it->second);
        }
        // Do the wait
        const RtEvent wait_on = Runtime::merge_events(wait_for);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
        tpl_lock.reacquire();
        // All our pending events should be here now
        for (std::vector<ApEvent>::const_iterator it = 
              pending_events.begin(); it != pending_events.end(); it++)
        {
          std::map<ApEvent,unsigned>::const_iterator finder =
            event_map.find(*it);
#ifdef DEBUG_LEGION
          assert(finder != event_map.end());
#endif
          if (finder->second != NO_INDEX)
            rhs_.insert(finder->second);
        }
      }
      if (rhs_.size() == 0)
        rhs_.insert(fence_completion_id);
      
      // If the lhs event wasn't made on this node then we need to rename it
      // because we need all events to go back to a node where we know that
      // we have a shard that can answer queries about it
      const AddressSpaceID event_space = find_event_space(lhs);
      if (event_space != repl_ctx->runtime->address_space)
      {
        ApUserEvent rename = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, rename, lhs);
        lhs = rename;
      }
#ifndef LEGION_DISABLE_EVENT_PRUNING
      else if (!lhs.exists() || (rhs.find(lhs) != rhs.end()))
      {
        ApUserEvent rename = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, rename, lhs);
        lhs = rename;
      }
#endif
      insert_instruction(
          new MergeEvent(*this, convert_event(lhs), rhs_, tlid));
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_merge_events(ApEvent &lhs,
                      const std::vector<ApEvent> &rhs, const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      std::set<unsigned> rhs_;
      std::set<RtEvent> wait_for;
      std::vector<ApEvent> pending_events;
      std::map<ApEvent,RtUserEvent> request_events;
      for (std::vector<ApEvent>::const_iterator it =
            rhs.begin(); it != rhs.end(); it++)
      {
        if (!it->exists())
          continue;
        std::map<ApEvent,unsigned>::const_iterator finder = event_map.find(*it);
        if (finder == event_map.end())
        {
          // We're going to need to check this event later
          pending_events.push_back(*it);
          // See if anyone else has requested this event yet 
          std::map<ApEvent,RtEvent>::const_iterator request_finder = 
            pending_event_requests.find(*it);
          if (request_finder == pending_event_requests.end())
          {
            const RtUserEvent request_event = Runtime::create_rt_user_event();
            pending_event_requests[*it] = request_event;
            wait_for.insert(request_event);
            request_events[*it] = request_event;
          }
          else
            wait_for.insert(request_finder->second);
        }
        else if (finder->second != NO_INDEX)
          rhs_.insert(finder->second);
      }
      // If we have anything to wait for we need to do that
      if (!wait_for.empty())
      {
        tpl_lock.release();
        // Send any request messages first
        if (!request_events.empty())
        {
          for (std::map<ApEvent,RtUserEvent>::const_iterator it = 
                request_events.begin(); it != request_events.end(); it++)
            request_remote_shard_event(it->first, it->second);
        }
        // Do the wait
        const RtEvent wait_on = Runtime::merge_events(wait_for);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
        tpl_lock.reacquire();
        // All our pending events should be here now
        for (std::vector<ApEvent>::const_iterator it = 
              pending_events.begin(); it != pending_events.end(); it++)
        {
          std::map<ApEvent,unsigned>::const_iterator finder =
            event_map.find(*it);
#ifdef DEBUG_LEGION
          assert(finder != event_map.end());
#endif
          if (finder->second != NO_INDEX)
            rhs_.insert(finder->second);
        }
      }
      if (rhs_.size() == 0)
        rhs_.insert(fence_completion_id);
      
      // If the lhs event wasn't made on this node then we need to rename it
      // because we need all events to go back to a node where we know that
      // we have a shard that can answer queries about it
      const AddressSpaceID event_space = find_event_space(lhs);
      if (event_space != repl_ctx->runtime->address_space)
      {
        ApUserEvent rename = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, rename, lhs);
        lhs = rename;
      }
#ifndef LEGION_DISABLE_EVENT_PRUNING
      else if (!lhs.exists())
      {
        ApUserEvent rename = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, rename);
        lhs = rename;
      }
      else
      {
        for (unsigned idx = 0; idx < rhs.size(); idx++)
        {
          if (lhs != rhs[idx])
            continue;
          ApUserEvent rename = Runtime::create_ap_user_event(NULL);
          Runtime::trigger_event(NULL, rename, lhs);
          lhs = rename;
          break;
        }
      }
#endif
      insert_instruction(
          new MergeEvent(*this, convert_event(lhs), rhs_, tlid));
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    unsigned ShardedPhysicalTemplate::convert_event(const ApEvent &event, 
                                                    bool check)
    //--------------------------------------------------------------------------
    {
      // We should only be recording events made on our node
      assert(!check || 
          (find_event_space(event) == repl_ctx->runtime->address_space));
      return PhysicalTemplate::convert_event(event, check);
    }
#endif

    //--------------------------------------------------------------------------
    unsigned ShardedPhysicalTemplate::find_event(const ApEvent &event,
                                                 AutoLock &tpl_lock)
    //--------------------------------------------------------------------------
    {
      std::map<ApEvent,unsigned>::const_iterator finder = event_map.find(event);
      // If we've already got it then we're done
      if (finder != event_map.end())
      {
#ifdef DEBUG_LEGION
        assert(finder->second != NO_INDEX);
#endif
        return finder->second;
      }
      // If we don't have it then we need to request it
      // See if someone else already sent the request
      RtEvent wait_for;
      RtUserEvent request_event;
      std::map<ApEvent,RtEvent>::const_iterator request_finder = 
        pending_event_requests.find(event);
      if (request_finder == pending_event_requests.end())
      {
        // We're the first ones so send the request
        request_event = Runtime::create_rt_user_event();
        wait_for = request_event;
        pending_event_requests[event] = wait_for;
      }
      else
        wait_for = request_finder->second;
      // Can't be holding the lock while we wait
      tpl_lock.release();
      // Send the request if necessary
      if (request_event.exists())
        request_remote_shard_event(event, request_event);
      if (wait_for.exists())
        wait_for.wait();
      tpl_lock.reacquire();
      // Once we get here then there better be an answer
      finder = event_map.find(event);
#ifdef DEBUG_LEGION
      assert(finder != event_map.end());
      assert(finder->second != NO_INDEX);
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_collective_barrier(ApBarrier bar,
              ApEvent pre, const std::pair<size_t,size_t> &key, size_t arrivals)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bar.exists());
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      const unsigned pre_ = pre.exists() ? find_event(pre, tpl_lock) : 0;
#ifdef DEBUG_LEGION
      const unsigned bar_ = convert_event(bar, false/*check*/);
#else
      const unsigned bar_ = convert_event(bar);
#endif
      BarrierArrival *arrival =
        new BarrierArrival(*this, bar, bar_, pre_, arrivals, false/*managed*/);
      insert_instruction(arrival);
#ifdef DEBUG_LEGION
      assert(collective_barriers.find(key) == collective_barriers.end());
#endif
      // Save this collective barrier
      collective_barriers[key] = arrival;
    }

    //--------------------------------------------------------------------------
    ShardID ShardedPhysicalTemplate::record_managed_barrier(ApBarrier bar,
                                                          size_t total_arrivals)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::record_managed_barrier(bar, total_arrivals);
      return local_shard;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_barrier_arrival(ApBarrier bar,
        ApEvent pre, size_t arrival_count, std::set<RtEvent> &applied,
        ShardID owner_shard)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(bar.exists());
      assert(is_recording());
#endif
      // Find the pre event first
      unsigned rhs = find_event(pre, tpl_lock);
      events.push_back(ApEvent());
      BarrierArrival *arrival = new BarrierArrival(*this, bar,
          events.size() - 1, rhs, arrival_count, true/*managed*/);
      insert_instruction(arrival);
      if (owner_shard != local_shard)
      {
        // Check to see if we've already made a barrier arrival instruction
        // for this barrier or not
        std::map<ApEvent,std::vector<BarrierArrival*> >::iterator finder =
          managed_arrivals.find(bar);
        if (finder == managed_arrivals.end())
        {
          // Need to request a subscription to this barrier on the owner shard
          // We need to tell the owner shard that we are going to 
          // subscribe to its updates for this barrier
          RtEvent subscribed = Runtime::create_rt_user_event();
          ShardManager *manager = repl_ctx->shard_manager;
          Serializer rez;
          rez.serialize(manager->did);
          rez.serialize(owner_shard);
          rez.serialize(template_index);
          rez.serialize(REMOTE_BARRIER_SUBSCRIBE);
          rez.serialize(bar);
          rez.serialize(local_shard);
          rez.serialize(subscribed);
          manager->send_trace_update(owner_shard, rez);
          applied.insert(subscribed); 
          managed_arrivals[bar].push_back(arrival);
        }
        else
          finder->second.push_back(arrival);
      }
      else
        managed_arrivals[bar].push_back(arrival);
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_issue_copy(const TraceLocalID &tlid, 
                                 ApEvent &lhs, IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField>& src_fields,
                                 const std::vector<CopySrcDstField>& dst_fields,
                                 const std::vector<Reservation>& reservations,
#ifdef LEGION_SPY
                                 RegionTreeID src_tree_id,
                                 RegionTreeID dst_tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard,
                                 LgEvent src_unique, LgEvent dst_unique, 
                                 int priority, CollectiveKind collective)
    //--------------------------------------------------------------------------
    {
      // Make sure the lhs event is local to our shard
      if (lhs.exists())
      {
        const AddressSpaceID event_space = find_event_space(lhs);
        if (event_space != repl_ctx->runtime->address_space)
        {
          ApUserEvent rename = Runtime::create_ap_user_event(NULL);
          Runtime::trigger_event(NULL, rename, lhs);
          lhs = rename;
        }
      }
      // Then do the base call
      PhysicalTemplate::record_issue_copy(tlid, lhs, expr, src_fields,
                                          dst_fields, reservations,
#ifdef LEGION_SPY
                                          src_tree_id, dst_tree_id,
#endif
                                          precondition, pred_guard,
                                          src_unique, dst_unique,
                                          priority, collective); 
    } 
    
    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_issue_fill(const TraceLocalID &tlid,
                                 ApEvent &lhs, IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField> &fields,
                                 const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                                 UniqueID fill_uid, FieldSpace handle,
                                 RegionTreeID tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard,
                                 LgEvent unique_event, int priority,
                                 CollectiveKind collective)
    //--------------------------------------------------------------------------
    {
      // Make sure the lhs event is local to our shard
      if (lhs.exists())
      {
        const AddressSpaceID event_space = find_event_space(lhs);
        if (event_space != repl_ctx->runtime->address_space)
        {
          ApUserEvent rename = Runtime::create_ap_user_event(NULL);
          Runtime::trigger_event(NULL, rename, lhs);
          lhs = rename;
        }
      }
      // Then do the base call
      PhysicalTemplate::record_issue_fill(tlid, lhs, expr, fields,
                                          fill_value, fill_size,
#ifdef LEGION_SPY
                                          fill_uid, handle, tree_id,
#endif
                                          precondition, pred_guard, 
                                          unique_event, priority, collective);
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_issue_across(const TraceLocalID &tlid, 
                                             ApEvent &lhs,
                                             ApEvent collective_precondition,
                                             ApEvent copy_precondition,
                                             ApEvent src_indirect_precondition,
                                             ApEvent dst_indirect_precondition,
                                             CopyAcrossExecutor *executor)
    //--------------------------------------------------------------------------
    {
      // Make sure the lhs event is local to our shard
      if (lhs.exists())
      {
        const AddressSpaceID event_space = find_event_space(lhs);
        if (event_space != repl_ctx->runtime->address_space)
        {
          ApUserEvent rename = Runtime::create_ap_user_event(NULL);
          Runtime::trigger_event(NULL, rename, lhs);
          lhs = rename;
        }
      }
      // Then do the base call
      PhysicalTemplate::record_issue_across(tlid, lhs, collective_precondition,
                                            copy_precondition,
                                            src_indirect_precondition,
                                            dst_indirect_precondition,
                                            executor);
    }

    //--------------------------------------------------------------------------
    ApBarrier ShardedPhysicalTemplate::find_trace_shard_event(ApEvent event,
                                                           ShardID remote_shard)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      // Check to see if we made this event
      std::map<ApEvent,unsigned>::const_iterator finder = event_map.find(event);
      // If we didn't make this event then we don't do anything
      if (finder == event_map.end() || (finder->second == NO_INDEX))
        return ApBarrier::NO_AP_BARRIER;
      // If we did make it then see if we have a remote barrier for it yet
      std::map<ApEvent,BarrierAdvance*>::const_iterator barrier_finder = 
        managed_barriers.find(event);
      if (barrier_finder == managed_barriers.end())
      {
        // Make a new barrier and record it in the events
        ApBarrier barrier(Realm::Barrier::create_barrier(1/*arrival count*/));
        // The first generation of each barrier should be triggered when
        // it is recorded in a barrier arrival instruction
        Runtime::phase_barrier_arrive(barrier, 1/*count*/);
        // Record this in the instruction stream
#ifdef DEBUG_LEGION
        const unsigned lhs = convert_event(barrier, false/*check*/);
#else
        const unsigned lhs = convert_event(barrier);
#endif
        // First record the barrier advance for this new barrier
        BarrierAdvance *advance = new BarrierAdvance(*this, barrier,
                            lhs, 1/*arrival count*/, true/*owner*/);
        insert_instruction(advance);
        managed_barriers[event] = advance;
        // Next make the arrival instruction for this barrier
        events.push_back(ApEvent());
        BarrierArrival *arrival = new BarrierArrival(*this, barrier,
            events.size() - 1, finder->second, 1/*count*/, true/*managed*/);
        insert_instruction(arrival);
        managed_arrivals[event].push_back(arrival);
        // Record our local shard too
        advance->record_subscribed_shard(local_shard);
        return advance->record_subscribed_shard(remote_shard);
      }
      else
        return barrier_finder->second->record_subscribed_shard(remote_shard);
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_trace_shard_event(
                                               ApEvent event, ApBarrier barrier)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(event.exists());
      assert(event_map.find(event) == event_map.end());
#endif
      if (barrier.exists())
      {
#ifdef DEBUG_LEGION
        assert(local_advances.find(event) == local_advances.end());
        const unsigned index = convert_event(event, false/*check*/);
#else
        const unsigned index = convert_event(event);
#endif
        BarrierAdvance *advance =
          new BarrierAdvance(*this, barrier, index, 1/*count*/, false/*owner*/);
        insert_instruction(advance); 
        local_advances[event] = advance;
        // Don't remove it, just set it to NO_EVENT so we can tell the names
        // of the remote events that we got from other shards
        // See get_completion_for_deletion for where we use this
        std::map<ApEvent,RtEvent>::iterator finder = 
          pending_event_requests.find(event);
#ifdef DEBUG_LEGION
        assert(finder != pending_event_requests.end());
#endif
        finder->second = RtEvent::NO_RT_EVENT;
      }
      else // no barrier means it's not part of the trace
      {
        event_map[event] = NO_INDEX;
        // In this case we can remove it since we're not tracing it      
#ifdef DEBUG_LEGION
        std::map<ApEvent,RtEvent>::iterator finder = 
          pending_event_requests.find(event);
        assert(finder != pending_event_requests.end());
        pending_event_requests.erase(finder);
#else
        pending_event_requests.erase(event);
#endif
      }
    }

    //--------------------------------------------------------------------------
    ApBarrier ShardedPhysicalTemplate::find_trace_shard_frontier(ApEvent event,
                                                           ShardID remote_shard)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      // Check to see if we made this event
      std::map<ApEvent,unsigned>::const_iterator finder = event_map.find(event);
      // If we didn't make this event then we don't do anything
      if (finder == event_map.end() || (finder->second == NO_INDEX))
        return ApBarrier::NO_AP_BARRIER;
      std::map<unsigned,ApBarrier>::const_iterator barrier_finder =
        local_frontiers.find(finder->second);
      if (barrier_finder == local_frontiers.end())
      {
        // Make a barrier and record it 
        const ApBarrier result(
            Realm::Barrier::create_barrier(1/*arrival count*/));
        barrier_finder = local_frontiers.insert(
            std::make_pair(finder->second, result)).first;
      }
      // Record that this shard depends on this event
      local_subscriptions[finder->second].insert(remote_shard);
      return barrier_finder->second;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_trace_shard_frontier(
                                           unsigned frontier, ApBarrier barrier)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      remote_frontiers.emplace_back(std::make_pair(barrier, frontier));
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::handle_trace_update(Deserializer &derez,
                                                      AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      Runtime *runtime = repl_ctx->runtime;
      UpdateKind kind;
      derez.deserialize(kind);
      RtUserEvent done;
      std::set<RtEvent> applied;
      switch (kind)
      {
        case UPDATE_MUTATED_INST:
          {
            derez.deserialize(done);
            UniqueInst inst;
            inst.deserialize(derez);
            PendingRemoteExpression pending;
            RtEvent expr_ready;
            IndexSpaceExpression *user_expr = 
              IndexSpaceExpression::unpack_expression(derez, runtime->forest, 
                                    source, pending, expr_ready);
            if (expr_ready.exists())
            {
              DeferTraceUpdateArgs args(this, kind, done, inst, derez, pending);
              runtime->issue_runtime_meta_task(args,
                  LG_LATENCY_MESSAGE_PRIORITY, expr_ready);
              return;
            }
            else if (handle_update_mutated_inst(inst, user_expr, 
                                                derez, applied, done))
              return;
            break;
          }
        case READ_ONLY_USERS_REQUEST:
          {
            ShardID source_shard;
            derez.deserialize(source_shard);
#ifdef DEBUG_LEGION
            assert(source_shard != repl_ctx->owner_shard->shard_id);
#endif
            size_t num_users;
            derez.deserialize(num_users);
            InstUsers inst_users(num_users);
            RegionTreeForest *forest = trace->runtime->forest;
            for (unsigned vidx = 0; vidx < num_users; vidx++)
            {
              InstanceUser &user = inst_users[vidx];
              user.instance.deserialize(derez);
              user.expr = 
                 IndexSpaceExpression::unpack_expression(derez, forest, source);
              derez.deserialize(user.mask);
            }
            std::atomic<bool> *result;
            derez.deserialize(result);
            derez.deserialize(done);
            ShardManager *manager = repl_ctx->shard_manager;
            if (!PhysicalTemplate::are_read_only_users(inst_users))
            {
              Serializer rez;
              rez.serialize(manager->did);
              rez.serialize(source_shard);
              rez.serialize(template_index);
              rez.serialize(READ_ONLY_USERS_RESPONSE);
              rez.serialize(result);
              rez.serialize(done);
              manager->send_trace_update(source_shard, rez);
              // Make sure we don't double trigger
              done = RtUserEvent::NO_RT_USER_EVENT;
            }
            // Otherwise we can just fall through and trigger the event
            break;
          }
        case READ_ONLY_USERS_RESPONSE:
          {
            std::atomic<bool> *result;
            derez.deserialize(result);
            result->store(false);
            RtUserEvent done;
            derez.deserialize(done);
            Runtime::trigger_event(done);
            break;
          }
        case TEMPLATE_BARRIER_REFRESH:
          {
            size_t num_barriers;
            derez.deserialize(num_barriers);
            AutoLock tpl_lock(template_lock);
            if (update_advances_ready.exists())
            {
              for (unsigned idx = 0; idx < num_barriers; idx++)
              {
                ApEvent key;
                derez.deserialize(key);
                ApBarrier bar;
                derez.deserialize(bar);
                std::map<ApEvent,BarrierAdvance*>::const_iterator finder = 
                  local_advances.find(key);
                if (finder == local_advances.end())
                {
                  std::map<ApEvent,
                    std::vector<BarrierArrival*> >::const_iterator finder2 =
                    managed_arrivals.find(key);
#ifdef DEBUG_LEGION
                  assert(finder2 != managed_arrivals.end());
#endif
                  for (std::vector<BarrierArrival*>::const_iterator it =
                        finder2->second.begin(); it !=
                        finder2->second.end(); it++)
                    (*it)->set_managed_barrier(bar);
                }
                else
                  finder->second->remote_refresh_barrier(bar);
              }
              refreshed_barriers += num_barriers;
              const size_t expected = 
                local_advances.size() + managed_arrivals.size();
#ifdef DEBUG_LEGION
              assert(refreshed_barriers <= expected);
#endif
              // See if the wait has already been done by the local shard
              // If so, trigger it, otherwise do nothing so it can come
              // along and see that everything is done
              if (refreshed_barriers == expected)
              {
                done = update_advances_ready;
                // We're done so reset everything for the next refresh
                update_advances_ready = RtUserEvent::NO_RT_USER_EVENT;
                refreshed_barriers = 0;
              }
            }
            else
            {
              // Buffer these for later until we know it is safe to apply them
              for (unsigned idx = 0; idx < num_barriers; idx++)
              {
                ApEvent key;
                derez.deserialize(key);
#ifdef DEBUG_LEGION
                assert(pending_refresh_barriers.find(key) ==
                        pending_refresh_barriers.end());
#endif
                derez.deserialize(pending_refresh_barriers[key]); 
              }
            }
            break;
          }
        case FRONTIER_BARRIER_REFRESH:
          {
            size_t num_barriers;
            derez.deserialize(num_barriers);
            AutoLock tpl_lock(template_lock);
            if (update_frontiers_ready.exists())
            {
              // Unpack these barriers and refresh the frontiers
              for (unsigned idx = 0; idx < num_barriers; idx++)
              {
                ApBarrier oldbar, newbar;
                derez.deserialize(oldbar);
                derez.deserialize(newbar);
#ifdef DEBUG_LEGION
                bool found = false;
#endif
                for (std::vector<std::pair<ApBarrier,unsigned> >::iterator it =
                      remote_frontiers.begin(); it != 
                      remote_frontiers.end(); it++) 
                {
                  if (it->first != oldbar)
                    continue;
                  it->first = newbar;
#ifdef DEBUG_LEGION
                  found = true;
#endif
                  break;
                }
#ifdef DEBUG_LEGION
                assert(found);
#endif
              }
              updated_frontiers += num_barriers;
#ifdef DEBUG_LEGION
              assert(updated_frontiers <= remote_frontiers.size());
#endif
              if (updated_frontiers == remote_frontiers.size())
              {
                done = update_frontiers_ready;
                // We're done so reset everything for the next stage
                update_frontiers_ready = RtUserEvent::NO_RT_USER_EVENT;
                updated_frontiers = 0;
              }
            }
            else
            {
              // Buffer these barriers for later until it is safe
              for (unsigned idx = 0; idx < num_barriers; idx++)
              {
                ApBarrier oldbar;
                derez.deserialize(oldbar);
#ifdef DEBUG_LEGION
                assert(pending_refresh_frontiers.find(oldbar) ==
                        pending_refresh_frontiers.end());
#endif
                derez.deserialize(pending_refresh_frontiers[oldbar]);
              }
            }
            break;
          }
        case REMOTE_BARRIER_SUBSCRIBE:
          {
            ApBarrier bar;
            derez.deserialize(bar);
            ShardID remote_shard;
            derez.deserialize(remote_shard);
            derez.deserialize(done);

            AutoLock tpl_lock(template_lock);
            std::map<ApEvent,BarrierAdvance*>::const_iterator finder =
              managed_barriers.find(bar);
#ifdef DEBUG_LEGION
            assert(finder != managed_barriers.end());
#endif
            finder->second->record_subscribed_shard(remote_shard);
            break;
          }
        default:
          assert(false);
      }
      if (done.exists())
      {
        if (!applied.empty())
          Runtime::trigger_event(done, Runtime::merge_events(applied));
        else
          Runtime::trigger_event(done);
      }
    }

    //--------------------------------------------------------------------------
    ShardedPhysicalTemplate::DeferTraceUpdateArgs::DeferTraceUpdateArgs(
     ShardedPhysicalTemplate *t, UpdateKind k, RtUserEvent d, 
     Deserializer &derez, const UniqueInst &i, RtUserEvent u)
      : LgTaskArgs<DeferTraceUpdateArgs>(implicit_provenance), target(t), 
        kind(k), done(d), inst(i), expr(NULL),
        buffer_size(derez.get_remaining_bytes()), buffer(malloc(buffer_size)),
        deferral_event(u)
    //--------------------------------------------------------------------------
    {
      memcpy(buffer, derez.get_current_pointer(), buffer_size);
      derez.advance_pointer(buffer_size);
    }

    //--------------------------------------------------------------------------
    ShardedPhysicalTemplate::DeferTraceUpdateArgs::DeferTraceUpdateArgs(
     ShardedPhysicalTemplate *t, UpdateKind k,RtUserEvent d,const UniqueInst &i,
     Deserializer &derez, IndexSpaceExpression *x, RtUserEvent u)
      : LgTaskArgs<DeferTraceUpdateArgs>(implicit_provenance), target(t),
        kind(k), done(d), inst(i), expr(x),
        buffer_size(derez.get_remaining_bytes()), buffer(malloc(buffer_size)),
        deferral_event(u)
    //--------------------------------------------------------------------------
    {
      memcpy(buffer, derez.get_current_pointer(), buffer_size);
      derez.advance_pointer(buffer_size);
      expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    ShardedPhysicalTemplate::DeferTraceUpdateArgs::DeferTraceUpdateArgs(
     ShardedPhysicalTemplate *t, UpdateKind k,RtUserEvent d,const UniqueInst &i,
     Deserializer &derez, const PendingRemoteExpression &pend)
      : LgTaskArgs<DeferTraceUpdateArgs>(implicit_provenance), target(t), 
        kind(k), done(d), inst(i), expr(NULL),
        pending(pend), buffer_size(derez.get_remaining_bytes()), 
        buffer(malloc(buffer_size))
    //--------------------------------------------------------------------------
    {
      memcpy(buffer, derez.get_current_pointer(), buffer_size);
      derez.advance_pointer(buffer_size);
    }

    //--------------------------------------------------------------------------
    ShardedPhysicalTemplate::DeferTraceUpdateArgs::DeferTraceUpdateArgs(
        const DeferTraceUpdateArgs &rhs, RtUserEvent d, IndexSpaceExpression *e)
      : LgTaskArgs<DeferTraceUpdateArgs>(rhs.provenance), target(rhs.target),
        kind(rhs.kind), done(rhs.done), inst(rhs.inst), expr(e), 
        pending(rhs.pending), buffer_size(rhs.buffer_size), buffer(rhs.buffer),
        deferral_event(d)
    //--------------------------------------------------------------------------
    {
      // Expression reference rolls over unless its new and we need a reference
      if (rhs.expr != expr)
        expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardedPhysicalTemplate::handle_deferred_trace_update(
                                             const void *args, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const DeferTraceUpdateArgs *dargs = (const DeferTraceUpdateArgs*)args;
      std::set<RtEvent> applied;
      Deserializer derez(dargs->buffer, dargs->buffer_size);
      switch (dargs->kind)
      {
        case UPDATE_MUTATED_INST:
          {
            if (dargs->expr != NULL)
            {
              if (dargs->target->handle_update_mutated_inst(dargs->inst,
                        dargs->expr, derez, applied, dargs->done, dargs))
                return;
            }
            else
            {
              IndexSpaceExpression *expr = 
                runtime->forest->find_remote_expression(dargs->pending);
              if (dargs->target->handle_update_mutated_inst(dargs->inst,
                              expr, derez, applied, dargs->done, dargs))
                return;
            }
            break;
          }
        default:
          assert(false); // should never get here
      }
#ifdef DEBUG_LEGION
      assert(dargs->done.exists());
#endif
      if (!applied.empty())
        Runtime::trigger_event(dargs->done, Runtime::merge_events(applied));
      else
        Runtime::trigger_event(dargs->done);
      if (dargs->deferral_event.exists())
        Runtime::trigger_event(dargs->deferral_event);
      if ((dargs->expr != NULL) && 
          dargs->expr->remove_base_expression_reference(META_TASK_REF))
        delete dargs->expr;
      free(dargs->buffer);
    }

    //--------------------------------------------------------------------------
    bool ShardedPhysicalTemplate::handle_update_mutated_inst(
                                              const UniqueInst &inst,
                                              IndexSpaceExpression *user_expr, 
                                              Deserializer &derez, 
                                              std::set<RtEvent> &applied,
                                              RtUserEvent done,
                                              const DeferTraceUpdateArgs *dargs)
    //--------------------------------------------------------------------------
    {
      AutoTryLock tpl_lock(template_lock);
      if (!tpl_lock.has_lock())
      {
        RtUserEvent deferral;
        if (dargs != NULL)
          deferral = dargs->deferral_event;
        RtEvent pre;
        if (!deferral.exists())
        {
          deferral = Runtime::create_rt_user_event();
          pre = chain_deferral_events(deferral);
        }
        else
          pre = tpl_lock.try_next();
        if (dargs == NULL)
        {
          DeferTraceUpdateArgs args(this, UPDATE_MUTATED_INST, done, inst,
                                    derez, user_expr, deferral);
          repl_ctx->runtime->issue_runtime_meta_task(args, 
                  LG_LATENCY_MESSAGE_PRIORITY, pre);
        }
        else
        {
          DeferTraceUpdateArgs args(*dargs, deferral, user_expr);
          repl_ctx->runtime->issue_runtime_meta_task(args, 
                  LG_LATENCY_MESSAGE_PRIORITY, pre);
#ifdef DEBUG_LEGION
          // Keep the deserializer happy since we didn't use it
          derez.advance_pointer(derez.get_remaining_bytes());
#endif
        }
        return true;
      }
      FieldMask user_mask;
      derez.deserialize(user_mask);
      PhysicalTemplate::record_mutated_instance(inst, user_expr,
                                                user_mask, applied);
      return false;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::request_remote_shard_event(ApEvent event,
                                                         RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(event.exists());
#endif
      const AddressSpaceID event_space = find_event_space(event);
      repl_ctx->shard_manager->send_trace_event_request(this, 
          repl_ctx->owner_shard->shard_id, repl_ctx->runtime->address_space, 
          template_index, event, event_space, done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID ShardedPhysicalTemplate::find_event_space(
                                                                  ApEvent event)
    //--------------------------------------------------------------------------
    {
      if (!event.exists())
        return 0;
      // TODO: Remove hack include at top of file when we fix this 
      const Realm::ID id(event.id);
      if (id.is_barrier())
        return id.barrier_creator_node();
#ifdef DEBUG_LEGION
      assert(id.is_event());
#endif
      return id.event_creator_node();
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate::Replayable ShardedPhysicalTemplate::check_replayable(
                                         ReplTraceOp *op, InnerContext *context,
                                         UniqueID opid, bool has_blocking_call)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(op != NULL);
#endif
      // We need everyone else to be done capturing their traces
      // before we can do our own replayable check
      op->sync_for_replayable_check();
      // Do the base call first to determine if our local shard is replayable
      const Replayable result = 
       PhysicalTemplate::check_replayable(op, context, opid, has_blocking_call);
      if (result)
      {
        // Now we can do the exchange
        if (op->exchange_replayable(repl_ctx, true/*replayable*/))
          return result;
        else
          return Replayable(false, "Remote shard not replyable");
      }
      else
      {
        // Still need to do the exchange
        op->exchange_replayable(repl_ctx, false/*replayable*/);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::initialize_replay(
                       ApEvent fence_completion, bool recurrent, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock t_lock(template_lock);
        initialize_replay(fence_completion, recurrent, false/*need lock*/);
        return;
      }
      PhysicalTemplate::initialize_replay(fence_completion, recurrent, false);
      pending_collectives.emplace_back(
          std::map<std::pair<size_t,size_t>,ApBarrier>());
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::perform_replay(Runtime *runtime,
                                             std::set<RtEvent> &replayed_events)
    //--------------------------------------------------------------------------
    {
      ApEvent completion; bool recurrent;
      std::map<std::pair<size_t,size_t>,ApBarrier> collective_updates;
      {
        AutoLock t_lock(template_lock);
#ifdef DEBUG_LEGION
        assert(!pending_replays.empty());
        assert(!pending_collectives.empty());
#endif
        const std::pair<ApEvent,bool> &pending = pending_replays.front();
        completion = pending.first;
        recurrent = pending.second;
        collective_updates.swap(pending_collectives.front());
        pending_collectives.pop_front();
      }
      // Now update all of our barrier information
      if (recurrent)
      {
        // If we've run out of generations update the local barriers and
        // send out the updates to everyone
        if (recurrent_replays == Realm::Barrier::MAX_PHASES)
        {
          std::map<ShardID,std::map<ApBarrier/*old**/,ApBarrier/*new*/> >
            notifications;
          // Update our barriers and record which updates to send out
          for (std::map<unsigned,ApBarrier>::iterator it = 
                local_frontiers.begin(); it != local_frontiers.end(); it++)
          {
            const ApBarrier new_barrier(
                Realm::Barrier::create_barrier(1/*arrival count*/));
#ifdef DEBUG_LEGION
            assert(local_subscriptions.find(it->first) !=
                    local_subscriptions.end());
#endif
            const std::set<ShardID> &shards = local_subscriptions[it->first];
            for (std::set<ShardID>::const_iterator sit = 
                  shards.begin(); sit != shards.end(); sit++)
              notifications[*sit][it->second] = new_barrier;
            // destroy the old barrier and replace it with the new one
            it->second.destroy_barrier();
            it->second = new_barrier;
          }
          // Send out the notifications to all the remote shards
          ShardManager *manager = repl_ctx->shard_manager;
          for (std::map<ShardID,std::map<ApBarrier,ApBarrier> >::const_iterator
                nit = notifications.begin(); nit != notifications.end(); nit++)
          {
            Serializer rez;
            rez.serialize(manager->did);
            rez.serialize(nit->first);
            rez.serialize(template_index);
            rez.serialize(FRONTIER_BARRIER_REFRESH);
            rez.serialize<size_t>(nit->second.size());
            for (std::map<ApBarrier,ApBarrier>::const_iterator it = 
                  nit->second.begin(); it != nit->second.end(); it++)
            {
              rez.serialize(it->first);
              rez.serialize(it->second);
            }
            manager->send_trace_update(nit->first, rez);
          }
          // Now we wait to see that we get all of our remote barriers updated
          RtEvent remote_frontiers_ready;
          {
            AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
            assert(!update_frontiers_ready.exists());
#endif
            // Apply any pending refresh frontiers
            if (!pending_refresh_frontiers.empty())
            {
              for (std::map<ApBarrier,ApBarrier>::const_iterator pit =
                    pending_refresh_frontiers.begin(); pit != 
                    pending_refresh_frontiers.end(); pit++)
              {
#ifdef DEBUG_LEGION
                bool found = false;
#endif
                for (std::vector<std::pair<ApBarrier,unsigned> >::iterator it =
                      remote_frontiers.begin(); it !=
                      remote_frontiers.end(); it++)
                {
                  if (it->first != pit->first)
                    continue;
                  it->first = pit->second;
#ifdef DEBUG_LEGION
                  found = true;
#endif
                  break;
                }
#ifdef DEBUG_LEGION
                assert(found);
#endif
              }
              updated_frontiers += pending_refresh_frontiers.size();
#ifdef DEBUG_LEGION
              assert(updated_frontiers <= remote_frontiers.size());
#endif
              pending_refresh_frontiers.clear();
            }
            if (updated_frontiers < remote_frontiers.size())
            {
              update_frontiers_ready = Runtime::create_rt_user_event();
              remote_frontiers_ready = update_frontiers_ready;
            }
            else // Reset this back to zero for the next round
              updated_frontiers = 0;
          }
          // Wait for the remote frontiers to be updated
          if (remote_frontiers_ready.exists() &&
              !remote_frontiers_ready.has_triggered())
            remote_frontiers_ready.wait();
          // Reset this back to zero after barrier updates
          recurrent_replays = 0;
        }
        // Now we can do the normal update of events based on our barriers
        // Don't advance on last generation to avoid setting barriers back to 0
        const bool advance_barriers =
          ((++recurrent_replays) < Realm::Barrier::MAX_PHASES);
        for (std::map<unsigned,ApBarrier>::iterator it = 
              local_frontiers.begin(); it != local_frontiers.end(); it++)
        {
          Runtime::phase_barrier_arrive(it->second, 1/*count*/, 
                                        events[it->first]);
          if (advance_barriers)
            Runtime::advance_barrier(it->second);
        }
        for (std::vector<std::pair<ApBarrier,unsigned> >::iterator it = 
              remote_frontiers.begin(); it != remote_frontiers.end(); it++)
        {
          events[it->second] = it->first;
          if (advance_barriers)
            Runtime::advance_barrier(it->first);
        }
      }
      else
      {
        for (std::vector<std::pair<ApBarrier,unsigned> >::const_iterator it =
              remote_frontiers.begin(); it != remote_frontiers.end(); it++)
          events[it->second] = completion;
      }
      if (!collective_updates.empty())
      {
        for (std::map<std::pair<size_t,size_t>,ApBarrier>::const_iterator it =
              collective_updates.begin(); it != collective_updates.end(); it++)
        {
          // This data structure should be read-only at this point
          // so we shouldn't need the lock to access it
          std::map<std::pair<size_t,size_t>,BarrierArrival*>::const_iterator
            finder = collective_barriers.find(it->first);
#ifdef DEBUG_LEGION
          assert(finder != collective_barriers.end());
#endif
          finder->second->set_managed_barrier(it->second);
        }
      }
      // Now call the base version of this
      PhysicalTemplate::perform_replay(runtime, replayed_events); 
    }

    //--------------------------------------------------------------------------
    RtEvent ShardedPhysicalTemplate::refresh_managed_barriers(void)
    //--------------------------------------------------------------------------
    {
      std::map<ShardID,std::map<ApEvent,ApBarrier> > notifications;
      // Need to update all our barriers since we're out of generations
      for (std::map<ApEvent,BarrierAdvance*>::const_iterator it = 
            managed_barriers.begin(); it != managed_barriers.end(); it++)
        it->second->refresh_barrier(it->first, notifications);
      // Send out the notifications to all the shards
      ShardManager *manager = repl_ctx->shard_manager;
      size_t local_refreshed = 0;
      for (std::map<ShardID,std::map<ApEvent,ApBarrier> >::const_iterator
            nit = notifications.begin(); nit != notifications.end(); nit++)
      {
        if (nit->first != local_shard)
        {
          Serializer rez;
          rez.serialize(manager->did);
          rez.serialize(nit->first);
          rez.serialize(template_index);
          rez.serialize(TEMPLATE_BARRIER_REFRESH);
          rez.serialize<size_t>(nit->second.size());
          for (std::map<ApEvent,ApBarrier>::const_iterator it = 
                nit->second.begin(); it != nit->second.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
          }
          manager->send_trace_update(nit->first, rez);
        }
        else
        {
          local_refreshed = nit->second.size();
          for (std::map<ApEvent,ApBarrier>::const_iterator it =
                nit->second.begin(); it != nit->second.end(); it++)
          {
            std::map<ApEvent,std::vector<BarrierArrival*> >::iterator finder =
              managed_arrivals.find(it->first);
#ifdef DEBUG_LEGION
            assert(finder != managed_arrivals.end());
#endif
            for (unsigned idx = 0; idx < finder->second.size(); idx++)
              finder->second[idx]->set_managed_barrier(it->second);
          }
        }
      }
      // Then wait for all our advances to be updated from other shards
      RtEvent replay_precondition;
      {
        AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
        assert(!update_advances_ready.exists());
#endif
        if (local_refreshed > 0)
          refreshed_barriers += local_refreshed;
        if (!pending_refresh_barriers.empty())
        {
          for (std::map<ApEvent,ApBarrier>::const_iterator it = 
                pending_refresh_barriers.begin(); it != 
                pending_refresh_barriers.end(); it++)
          {
            std::map<ApEvent,BarrierAdvance*>::const_iterator finder = 
              local_advances.find(it->first);
            if (finder == local_advances.end())
            {
              std::map<ApEvent,std::vector<BarrierArrival*> >::const_iterator 
                finder2 = managed_arrivals.find(it->first);
#ifdef DEBUG_LEGION
              assert(finder2 != managed_arrivals.end());
#endif
              for (unsigned idx = 0; idx < finder2->second.size(); idx++)
                finder2->second[idx]->set_managed_barrier(it->second);
            }
            else
              finder->second->remote_refresh_barrier(it->second);
          }
          refreshed_barriers += pending_refresh_barriers.size();

          pending_refresh_barriers.clear();
        }
        const size_t expected = 
          local_advances.size() + managed_arrivals.size();
#ifdef DEBUG_LEGION
        assert(refreshed_barriers <= expected);
#endif
        if (refreshed_barriers < expected)
        {
          update_advances_ready = Runtime::create_rt_user_event();
          replay_precondition = update_advances_ready;
        }
        else // Reset this back to zero for the next round
          refreshed_barriers = 0;
      }
      return replay_precondition;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::finish_replay(
                                              std::set<ApEvent> &postconditions)
    //--------------------------------------------------------------------------
    {
      for (std::map<unsigned,unsigned>::const_iterator it =
            frontiers.begin(); it != frontiers.end(); it++)
        postconditions.insert(events[it->first]);
      // Also need to do any local frontiers that we have here as well
      for (std::map<unsigned,ApBarrier>::const_iterator it = 
            local_frontiers.begin(); it != local_frontiers.end(); it++)
        postconditions.insert(events[it->first]);
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
    ApEvent ShardedPhysicalTemplate::get_completion_for_deletion(void) const
    //--------------------------------------------------------------------------
    {
      // Skip the any events that are from remote shards since we  
      std::set<ApEvent> all_events;
      std::set<ApEvent> local_barriers;
      for (std::map<ApEvent,BarrierAdvance*>::const_iterator it = 
            managed_barriers.begin(); it != managed_barriers.end(); it++)
        local_barriers.insert(it->second->get_current_barrier());
      for (std::map<ApEvent, unsigned>::const_iterator it = event_map.begin();
           it != event_map.end(); ++it)
      {
        // If this is a remote event or one of our barriers then don't use it
        if ((local_barriers.find(it->first) == local_barriers.end()) &&
            (pending_event_requests.find(it->first) == 
              pending_event_requests.end()))
          all_events.insert(it->first);
      }
      return Runtime::merge_events(NULL, all_events);
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_mutated_instance(
                                                const UniqueInst &inst,
                                                IndexSpaceExpression *user_expr,
                                                const FieldMask &user_mask,
                                                std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      const ShardID target_shard = find_inst_owner(inst); 
      // Check to see if we're on the right shard, if not send the message
      if (target_shard != repl_ctx->owner_shard->shard_id)
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        ShardManager *manager = repl_ctx->shard_manager;
        Serializer rez;
        rez.serialize(manager->did);
        rez.serialize(target_shard);
        rez.serialize(template_index);
        rez.serialize(UPDATE_MUTATED_INST);
        rez.serialize(done);
        inst.serialize(rez);
        user_expr->pack_expression(rez, manager->get_shard_space(target_shard));
        rez.serialize(user_mask);
        manager->send_trace_update(target_shard, rez);
        applied.insert(done);
      }
      else
        PhysicalTemplate::record_mutated_instance(inst, user_expr, 
                                                  user_mask, applied);
    }

    //--------------------------------------------------------------------------
    ShardID ShardedPhysicalTemplate::find_inst_owner(const UniqueInst &inst)
    //--------------------------------------------------------------------------
    {
      // Figure out where the owner for this instance is and then send it to 
      // the appropriate shard trace. The algorithm we use for determining
      // the right shard trace is to send a instance to a shard trace on the 
      // node that owns the instance. If there is no shard on that node we 
      // round-robin views based on their owner node mod the number of nodes
      // where there are shards. Once on the correct node, then we pick the
      // shard corresponding to their instance ID mod the number of shards on
      // that node. This algorithm guarantees that all the related instances
      // end up on the same shard for analysis to determine if the trace is
      // replayable or not.
      const AddressSpaceID inst_owner = inst.get_analysis_space();
      std::vector<ShardID> owner_shards;
      find_owner_shards(inst_owner, owner_shards);
#ifdef DEBUG_LEGION
      assert(!owner_shards.empty());
#endif
      // Round-robin based on the distributed IDs for the views in the
      // case where there are multiple shards, this should relatively
      // balance things out
      if (owner_shards.size() > 1)
        return owner_shards[inst.view_did % owner_shards.size()];
      else // If there's only one shard then there is only one choice
        return owner_shards.front();
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::find_owner_shards(AddressSpaceID owner,
                                                   std::vector<ShardID> &shards)
    //--------------------------------------------------------------------------
    {
      // See if we already computed it or not
      std::map<AddressSpaceID,std::vector<ShardID> >::const_iterator finder = 
        did_shard_owners.find(owner);
      if (finder != did_shard_owners.end())
      {
        shards = finder->second;
        return;
      }
      // If we haven't computed it yet, then we need to do that now
      const ShardMapping &shard_spaces = repl_ctx->shard_manager->get_mapping();
      for (unsigned idx = 0; idx < shard_spaces.size(); idx++)
        if (shard_spaces[idx] == owner)
          shards.push_back(idx);
      // If we didn't find any then take the owner mod the number of total
      // spaces and then send it to the shards on that space
      if (shards.empty())
      {
        std::set<AddressSpaceID> unique_spaces;
        for (unsigned idx = 0; idx < shard_spaces.size(); idx++)
          unique_spaces.insert(shard_spaces[idx]);
        const unsigned count = owner % unique_spaces.size();
        std::set<AddressSpaceID>::const_iterator target_space = 
          unique_spaces.begin();
        for (unsigned idx = 0; idx < count; idx++)
          target_space++;
        for (unsigned idx = 0; idx < shard_spaces.size(); idx++)
          if (shard_spaces[idx] == *target_space)
            shards.push_back(idx);
      }
#ifdef DEBUG_LEGION
      assert(!shards.empty());
#endif
      // Save the result so we don't have to do this again for this space
      did_shard_owners[owner] = shards;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_owner_shard(unsigned tid,ShardID owner)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(owner_shards.find(tid) == owner_shards.end());
#endif
      owner_shards[tid] = owner;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_local_space(unsigned tid,IndexSpace sp)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(local_spaces.find(tid) == local_spaces.end());
#endif
      local_spaces[tid] = sp;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_sharding_function(unsigned tid,
                                                     ShardingFunction *function)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(sharding_functions.find(tid) == sharding_functions.end());
#endif
      sharding_functions[tid] = function;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::issue_summary_operations(
          InnerContext *context, Operation *invalidator, Provenance *provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(context);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(context); 
#endif
      ReplTraceSummaryOp *op = trace->runtime->get_available_repl_summary_op();
      op->initialize_summary(repl_ctx, this, invalidator, provenance);
#ifdef LEGION_SPY
      LegionSpy::log_summary_op_creator(op->get_unique_op_id(),
                                        invalidator->get_unique_op_id());
#endif
      op->execute_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::dump_sharded_template(void)
    //--------------------------------------------------------------------------
    {
      for (std::vector<std::pair<ApBarrier,unsigned> >::const_iterator it =
            remote_frontiers.begin(); it != remote_frontiers.end(); it++)
        log_tracing.info() << "events[" << it->second
                           << "] = Runtime::barrier_advance("
                           << std::hex << it->first.id << std::dec << ")";
      for (std::map<unsigned,ApBarrier>::const_iterator it =
            local_frontiers.begin(); it != local_frontiers.end(); it++)
        log_tracing.info() << "Runtime::phase_barrier_arrive(" 
                           << std::hex << it->second.id << std::dec
                           << ", events[" << it->first << "])";
    }

    //--------------------------------------------------------------------------
    ShardID ShardedPhysicalTemplate::find_owner_shard(unsigned tid)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      std::map<unsigned,ShardID>::const_iterator finder = 
        owner_shards.find(tid);
      assert(finder != owner_shards.end());
      return finder->second;
#else
      return owner_shards[tid];
#endif
    }

    //--------------------------------------------------------------------------
    IndexSpace ShardedPhysicalTemplate::find_local_space(unsigned tid)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      std::map<unsigned,IndexSpace>::const_iterator finder = 
        local_spaces.find(tid);
      assert(finder != local_spaces.end());
      return finder->second;
#else
      return local_spaces[tid];
#endif
    }

    //--------------------------------------------------------------------------
    ShardingFunction* ShardedPhysicalTemplate::find_sharding_function(
                                                                   unsigned tid)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      std::map<unsigned,ShardingFunction*>::const_iterator finder = 
        sharding_functions.find(tid);
      assert(finder != sharding_functions.end());
      return finder->second;
#else
      return sharding_functions[tid];
#endif
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::prepare_collective_barrier_replay(
                          const std::pair<size_t,size_t> &key, ApBarrier newbar)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(!pending_collectives.empty());
#endif
      // Save the barrier until it's safe to update the instruction
      pending_collectives.back()[key] = newbar;
    }

    //--------------------------------------------------------------------------
    unsigned ShardedPhysicalTemplate::find_frontier_event(ApEvent event,
                                             std::vector<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      // Check to see which shard should own this event
      std::map<ApEvent,unsigned>::const_iterator finder = event_map.find(event);
      if (finder != event_map.end())
      {
        if (finder->second == NO_INDEX)
          return 0; // start fence event
        else
          return PhysicalTemplate::find_frontier_event(event, ready_events);
      }
      const AddressSpaceID event_space = find_event_space(event);
      // Allocate a slot for this event though we might not use it 
      const unsigned next_event_id = events.size(); 
      const RtUserEvent done_event = Runtime::create_rt_user_event();
      repl_ctx->shard_manager->send_trace_frontier_request(this,
          repl_ctx->owner_shard->shard_id, repl_ctx->runtime->address_space,
          template_index, event, event_space, next_event_id, done_event);
      events.resize(next_event_id + 1);
      ready_events.push_back(done_event);
      return next_event_id;
    }

    //--------------------------------------------------------------------------
    bool ShardedPhysicalTemplate::are_read_only_users(InstUsers &inst_users)
    //--------------------------------------------------------------------------
    {
      std::map<ShardID,InstUsers> shard_inst_users;
      for (InstUsers::iterator vit = 
            inst_users.begin(); vit != inst_users.end(); vit++)
      {
        const ShardID owner_shard = find_inst_owner(vit->instance); 
        shard_inst_users[owner_shard].push_back(*vit);
      }
      std::atomic<bool> result(true);
      std::vector<RtEvent> done_events;
      ShardManager *manager = repl_ctx->shard_manager;
      const ShardID local_shard = repl_ctx->owner_shard->shard_id;
      for (std::map<ShardID,InstUsers>::iterator sit = 
            shard_inst_users.begin(); sit != shard_inst_users.end(); sit++)
      {
        if (sit->first != local_shard)
        {
          const RtUserEvent done = Runtime::create_rt_user_event();
          const AddressSpaceID target = manager->get_shard_space(sit->first);
          Serializer rez;
          rez.serialize(manager->did);
          rez.serialize(sit->first);
          rez.serialize(template_index);
          rez.serialize(READ_ONLY_USERS_REQUEST);
          rez.serialize(local_shard);
          rez.serialize<size_t>(sit->second.size());
          for (InstUsers::const_iterator vit = 
                sit->second.begin(); vit != sit->second.end(); vit++)
          {
            vit->instance.serialize(rez);
            vit->expr->pack_expression(rez, target);
            rez.serialize(vit->mask);
          }
          rez.serialize(&result);
          rez.serialize(done);
          manager->send_trace_update(sit->first, rez);
          done_events.push_back(done);
        }
        else if (!PhysicalTemplate::are_read_only_users(sit->second))
        {
          // Still need to wait for anyone else to write to result if 
          // they end up finding out that they are not read-only
          result.store(false);
          break;
        }
      }
      if (!done_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(done_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      return result.load();
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::sync_compute_frontiers(ReplTraceOp *op,
                                    const std::vector<RtEvent> &frontier_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(op != NULL);
#endif
      if (!frontier_events.empty())
        op->sync_compute_frontiers(Runtime::merge_events(frontier_events));
      else
        op->sync_compute_frontiers(RtEvent::NO_RT_EVENT);
      // Check for any empty remote frontiers which were not actually
      // contained in the trace and therefore need to be pruned out of
      // any event mergers
      std::vector<unsigned> to_filter;
      for (std::vector<std::pair<ApBarrier,unsigned> >::iterator it =
            remote_frontiers.begin(); it != remote_frontiers.end(); /*nothing*/)
      {
        if (!it->first.exists())
        {
          to_filter.push_back(it->second);
          it = remote_frontiers.erase(it);
        }
        else
          it++;
      }
      if (!to_filter.empty())
      {
        for (std::vector<Instruction*>::const_iterator it = 
              instructions.begin(); it != instructions.end(); it++)
        {
          if ((*it)->get_kind() != MERGE_EVENT)
            continue;
          MergeEvent *merge = (*it)->as_merge_event();
          for (unsigned idx = 0; idx < to_filter.size(); idx++)
          {
            std::set<unsigned>::iterator finder =
              merge->rhs.find(to_filter[idx]);
            if (finder == merge->rhs.end())
              continue;
            // Found one, filter it out from the set
            merge->rhs.erase(finder);
            // Handle a weird case where we pruned them all out
            // Go back to the case of just pointing at the completion event
            if (merge->rhs.empty())
              merge->rhs.insert(0/*fence completion id*/);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::initialize_generators(
                                                 std::vector<unsigned> &new_gen)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::initialize_generators(new_gen);
      for (std::vector<std::pair<ApBarrier,unsigned> >::const_iterator it =
            remote_frontiers.begin(); it != remote_frontiers.end(); it++)
        new_gen[it->second] = 0;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::initialize_eliminate_dead_code_frontiers(
                      const std::vector<unsigned> &gen, std::vector<bool> &used)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::initialize_eliminate_dead_code_frontiers(gen, used);
      for (std::map<unsigned,ApBarrier>::const_iterator it =
            local_frontiers.begin(); it != local_frontiers.end(); it++)
      {
        unsigned g = gen[it->first];
        if (g != -1U && g < instructions.size())
          used[g] = true;
      } 
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::initialize_transitive_reduction_frontiers(
       std::vector<unsigned> &topo_order, std::vector<unsigned> &inv_topo_order)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::initialize_transitive_reduction_frontiers(topo_order,
                                                              inv_topo_order);
      for (std::vector<std::pair<ApBarrier,unsigned> >::const_iterator it = 
            remote_frontiers.begin(); it != remote_frontiers.end(); it++)
      {
        inv_topo_order[it->second] = topo_order.size();
        topo_order.push_back(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_used_frontiers(std::vector<bool> &used,
                                         const std::vector<unsigned> &gen) const
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::record_used_frontiers(used, gen);  
      for (std::map<unsigned,ApBarrier>::const_iterator it =
            local_frontiers.begin(); it != local_frontiers.end(); it++)
        used[gen[it->first]] = true;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::rewrite_frontiers(
                                     std::map<unsigned,unsigned> &substitutions)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::rewrite_frontiers(substitutions);
      std::vector<std::pair<unsigned,ApBarrier> > to_add;
      for (std::map<unsigned,ApBarrier>::iterator it =
            local_frontiers.begin(); it != local_frontiers.end(); /*nothing*/)
      {
        std::map<unsigned,unsigned>::const_iterator finder =
          substitutions.find(it->first);
        if (finder != substitutions.end())
        {
          to_add.emplace_back(std::make_pair(finder->second,it->second));
          // Also need to update the local subscriptions data structure
          std::map<unsigned,std::set<ShardID> >::iterator subscription_finder =
            local_subscriptions.find(it->first);
#ifdef DEBUG_LEGION
          assert(subscription_finder != local_subscriptions.end());
#endif
          std::map<unsigned,std::set<ShardID> >::iterator local_finder =
            local_subscriptions.find(finder->second);
          if (local_finder != local_subscriptions.end())
            local_finder->second.insert(subscription_finder->second.begin(),
                                        subscription_finder->second.end());
          else
            local_subscriptions[finder->second].swap(
                                        subscription_finder->second);
          local_subscriptions.erase(subscription_finder);
          std::map<unsigned,ApBarrier>::iterator to_delete = it++;
          local_frontiers.erase(to_delete);
        }
        else
          it++;
      }
      for (std::vector<std::pair<unsigned,ApBarrier> >::const_iterator it =
            to_add.begin(); it != to_add.end(); it++)
        local_frontiers.insert(*it);
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
#endif
      if (fence)
        tpl.update_last_fence(this);
    }

    //--------------------------------------------------------------------------
    void GetTermEvent::execute(std::vector<ApEvent> &events,
                               std::map<unsigned,ApUserEvent> &user_events,
                               std::map<TraceLocalID,MemoizableOp*> &operations,
                               const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      events[lhs] = operations[owner]->get_completion_event();
    }

    //--------------------------------------------------------------------------
    std::string GetTermEvent::to_string(const MemoEntries &memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      MemoEntries::const_iterator finder = memo_entries.find(owner);
#ifdef DEBUG_LEGION
      assert(finder != memo_entries.end());
#endif
      ss << "events[" << lhs << "] = operations[" << owner
         << "].get_completion_event()    (op kind: "
         << Operation::op_names[finder->second.second]
         << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // ReplayMapping
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplayMapping::ReplayMapping(PhysicalTemplate& tpl, unsigned l,
                                 const TraceLocalID& r)
      : Instruction(tpl, r), lhs(l)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void ReplayMapping::execute(std::vector<ApEvent> &events,
                              std::map<unsigned,ApUserEvent> &user_events,
                              std::map<TraceLocalID,MemoizableOp*> &operations,
                              const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      events[lhs] = operations[owner]->replay_mapping();
    }

    //--------------------------------------------------------------------------
    std::string ReplayMapping::to_string(const MemoEntries &memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      MemoEntries::const_iterator finder = memo_entries.find(owner);
#ifdef DEBUG_LEGION
      assert(finder != memo_entries.end());
#endif
      ss << "events[" << lhs << "] = operations[" << owner
         << "].replay_mapping()    (op kind: "
         << Operation::op_names[finder->second.second]
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
                               std::map<TraceLocalID,MemoizableOp*> &operations,
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
    std::string CreateApUserEvent::to_string(const MemoEntries &memo_entries)
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
                               std::map<TraceLocalID,MemoizableOp*> &operations,
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
    std::string TriggerEvent::to_string(const MemoEntries &memo_entries)
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
                             std::map<TraceLocalID,MemoizableOp*> &operations,
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
    std::string MergeEvent::to_string(const MemoEntries &memo_entries)
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
                               std::map<TraceLocalID,MemoizableOp*> &operations,
                               const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      events[lhs] = tpl.get_fence_completion();
    }

    //--------------------------------------------------------------------------
    std::string AssignFenceCompletion::to_string(
                                                const MemoEntries &memo_entries)
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
                         unsigned pi, LgEvent src_uni, LgEvent dst_uni,
                         int pr, CollectiveKind collect)
      : Instruction(tpl, key), lhs(l), expr(e), src_fields(s), dst_fields(d), 
        reservations(r),
#ifdef LEGION_SPY
        src_tree_id(src_tid), dst_tree_id(dst_tid),
#endif
        precondition_idx(pi), src_unique(src_uni),
        dst_unique(dst_uni), priority(pr), collective(collect)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
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
                            std::map<TraceLocalID,MemoizableOp*> &operations,
                            const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      MemoizableOp *op = operations[owner];
      ApEvent precondition = events[precondition_idx];
      const PhysicalTraceInfo trace_info(op, -1U);
      events[lhs] = expr->issue_copy(op, trace_info, dst_fields, 
                                     src_fields, reservations,
#ifdef LEGION_SPY
                                     src_tree_id, dst_tree_id,
#endif
                                     precondition, PredEvent::NO_PRED_EVENT,
                                     src_unique, dst_unique,
                                     collective, priority, true/*replay*/);
    }

    //--------------------------------------------------------------------------
    std::string IssueCopy::to_string(const MemoEntries &memo_entries)
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
                              std::map<TraceLocalID,MemoizableOp*> &operations,
                              const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      MemoizableOp *op = operations[owner];
      ApEvent copy_pre = events[copy_precondition];
      ApEvent src_indirect_pre = events[src_indirect_precondition];
      ApEvent dst_indirect_pre = events[dst_indirect_precondition];
      const PhysicalTraceInfo trace_info(op, -1U);
      events[lhs] = executor->execute(op, PredEvent::NO_PRED_EVENT,
                                      copy_pre, src_indirect_pre,
                                      dst_indirect_pre, trace_info,
                                      true/*replay*/, recurrent_replay);
    }

    //--------------------------------------------------------------------------
    std::string IssueAcross::to_string(const MemoEntries &memo_entires)
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
                         UniqueID uid, FieldSpace h, RegionTreeID tid,
#endif
                         unsigned pi, LgEvent unique, int pr,
                         CollectiveKind collect)
      : Instruction(tpl, key), lhs(l), expr(e), fields(f), fill_size(size),
#ifdef LEGION_SPY
        fill_uid(uid), handle(h), tree_id(tid),
#endif
        precondition_idx(pi), unique_event(unique), priority(pr),
        collective(collect)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
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
                            std::map<TraceLocalID,MemoizableOp*> &operations,
                            const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      MemoizableOp *op = operations[owner];
      ApEvent precondition = events[precondition_idx];
      const PhysicalTraceInfo trace_info(op, -1U);
      events[lhs] = expr->issue_fill(op, trace_info, fields, 
                                     fill_value, fill_size,
#ifdef LEGION_SPY
                                     fill_uid, handle, tree_id,
#endif
                                     precondition, PredEvent::NO_PRED_EVENT,
                                     unique_event, collective, priority,
                                     true/*replay*/);
    }

    //--------------------------------------------------------------------------
    std::string IssueFill::to_string(const MemoEntries &memo_entries)
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
#endif
    }

    //--------------------------------------------------------------------------
    void SetOpSyncEvent::execute(std::vector<ApEvent> &events,
                               std::map<unsigned,ApUserEvent> &user_events,
                               std::map<TraceLocalID,MemoizableOp*> &operations,
                               const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      MemoizableOp *memoizable = operations[owner];
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      TraceInfo info(memoizable);
      ApEvent sync_condition = memoizable->compute_sync_precondition(info);
      events[lhs] = sync_condition;
    }

    //--------------------------------------------------------------------------
    std::string SetOpSyncEvent::to_string(const MemoEntries &memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      MemoEntries::const_iterator finder = memo_entries.find(owner);
#ifdef DEBUG_LEGION
      assert(finder != memo_entries.end());
#endif
      ss << "events[" << lhs << "] = operations[" << owner
         << "].compute_sync_precondition()    (op kind: "
         << Operation::op_names[finder->second.second]
         << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // CompleteReplay
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompleteReplay::CompleteReplay(PhysicalTemplate& tpl, const TraceLocalID& l,
                                   unsigned pr, unsigned po)
      : Instruction(tpl, l), pre(pr), post(po)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(pre < tpl.events.size());
      assert(post < tpl.events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void CompleteReplay::execute(std::vector<ApEvent> &events,
                               std::map<unsigned,ApUserEvent> &user_events,
                               std::map<TraceLocalID,MemoizableOp*> &operations,
                               const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(owner) != operations.end());
      assert(operations.find(owner)->second != NULL);
#endif
      MemoizableOp *memoizable = operations[owner];
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      memoizable->complete_replay(events[pre], events[post]);
    }

    //--------------------------------------------------------------------------
    std::string CompleteReplay::to_string(const MemoEntries &memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      MemoEntries::const_iterator finder = memo_entries.find(owner);
#ifdef DEBUG_LEGION
      assert(finder != memo_entries.end());
#endif
      ss << "operations[" << owner
         << "].complete_replay(events[" << pre
         << "], events[ " << post << "])    (op kind: "
         << Operation::op_names[finder->second.second]
         << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // BarrierArrival
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    BarrierArrival::BarrierArrival(PhysicalTemplate &tpl, ApBarrier bar,
                     unsigned _lhs, unsigned _rhs, size_t arrivals, bool manage)
      : Instruction(tpl, TraceLocalID(0,DomainPoint())), barrier(bar), 
        lhs(_lhs), rhs(_rhs), total_arrivals(arrivals), managed(manage)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
#endif
      if (managed)
        Runtime::advance_barrier(barrier);
    } 

    //--------------------------------------------------------------------------
    void BarrierArrival::execute(std::vector<ApEvent> &events,
                               std::map<unsigned,ApUserEvent> &user_events,
                               std::map<TraceLocalID,MemoizableOp*> &operations,
                               const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
#endif
      Runtime::phase_barrier_arrive(barrier, total_arrivals, events[rhs]);
      events[lhs] = barrier;
      if (managed)
        Runtime::advance_barrier(barrier);
    }

    //--------------------------------------------------------------------------
    std::string BarrierArrival::to_string(const MemoEntries &memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss; 
      ss << "events[" << lhs << "] = Runtime::phase_barrier_arrive("
         << std::hex << barrier.id << std::dec << ", events["; 
      ss << rhs << "], managed : " << (managed ? "yes" : "no") << ")";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    void BarrierArrival::set_collective_barrier(ApBarrier newbar)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!managed);
#endif
      barrier = newbar;
    }

    //--------------------------------------------------------------------------
    void BarrierArrival::set_managed_barrier(ApBarrier newbar)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(managed);
#endif
      barrier = newbar;
    }

    /////////////////////////////////////////////////////////////
    // BarrierAdvance
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    BarrierAdvance::BarrierAdvance(PhysicalTemplate &tpl, ApBarrier bar, 
                                  unsigned _lhs, size_t arrival_count, bool own) 
      : Instruction(tpl, TraceLocalID(0,DomainPoint())), barrier(bar), 
        lhs(_lhs), total_arrivals(arrival_count), owner(own)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < tpl.events.size());
#endif
      if (owner)
        Runtime::advance_barrier(barrier);
    }

    //--------------------------------------------------------------------------
    BarrierAdvance::~BarrierAdvance(void)
    //--------------------------------------------------------------------------
    {
      // Destroy our barrier if we're managing it
      if (owner)
        barrier.destroy_barrier();
    }

    //--------------------------------------------------------------------------
    void BarrierAdvance::execute(std::vector<ApEvent> &events,
                               std::map<unsigned,ApUserEvent> &user_events,
                               std::map<TraceLocalID,MemoizableOp*> &operations,
                               const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
#endif
      events[lhs] = barrier;
      Runtime::advance_barrier(barrier);
    }

    //--------------------------------------------------------------------------
    std::string BarrierAdvance::to_string(const MemoEntries &memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = Runtime::barrier_advance("
         << std::hex << barrier.id << std::dec << ")";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    ApBarrier BarrierAdvance::record_subscribed_shard(ShardID remote_shard)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(owner);
#endif
      subscribed_shards.push_back(remote_shard);
      return barrier;
    }

    //--------------------------------------------------------------------------
    void BarrierAdvance::refresh_barrier(ApEvent key, 
                  std::map<ShardID,std::map<ApEvent,ApBarrier> > &notifications)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(owner);
#endif
      // Destroy the old barrier
      barrier.destroy_barrier();
      // Make the new barrier
      barrier = ApBarrier(Realm::Barrier::create_barrier(total_arrivals));
      for (std::vector<ShardID>::const_iterator it = 
            subscribed_shards.begin(); it != subscribed_shards.end(); it++)
        notifications[*it][key] = barrier;
    }

    //--------------------------------------------------------------------------
    void BarrierAdvance::remote_refresh_barrier(ApBarrier newbar)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!owner);
      assert(subscribed_shards.empty()); 
#endif
      barrier = newbar;
    }

  }; // namespace Internal 
}; // namespace Legion

