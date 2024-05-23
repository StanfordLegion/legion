/* Copyright 2024 Stanford University, NVIDIA Corporation
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

    std::ostream& operator<<(std::ostream &out, ReplayableStatus status)
    {
      switch (status)
      {
        case REPLAYABLE:
          {
            out << "Yes";
            break;
          }
        case NOT_REPLAYABLE_BLOCKING:
          {
            out << "No (Blocking Call)";
            break;
          }
        case NOT_REPLAYABLE_CONSENSUS:
          {
            out << "No (Mapper Consensus)";
            break;
          }
        case NOT_REPLAYABLE_VIRTUAL:
          {
            out << "No (Virtual Mapping)";
            break;
          }
        case NOT_REPLAYABLE_REMOTE_SHARD:
          {
            out << "No (Remote Shard)";
            break;
          }
        default:
          assert(false);
      }
      return out;
    }

    std::ostream& operator<<(std::ostream &out, IdempotencyStatus status)
    {
      switch (status)
      {
        case IDEMPOTENT:
          {
            out << "Yes";
            break;
          }
        case NOT_IDEMPOTENT_SUBSUMPTION:
          {
            out << "No (Preconditions Not Subsumed by Postconditions)";
            break;
          }
        case NOT_IDEMPOTENT_ANTIDEPENDENT:
          {
            out << "No (Postcondition Anti Dependent)";
            break;
          }
        case NOT_IDEMPOTENT_REMOTE_SHARD:
          {
            out << "No (Remote Shard)";
            break;
          }
        default:
          assert(false);
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
        physical_trace(logical_only ? NULL :
            new PhysicalTrace(c->owner_task->runtime, this)),
        verification_index(0), blocking_call_observed(false), fixed(false),
        intermediate_fence(false), recording(true), trace_fence(NULL),
        static_translator(static_trace ? new StaticTranslator(trees) : NULL)
    //--------------------------------------------------------------------------
    {
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
        if (has_physical_trace() && (op->get_memoizable() == NULL))
          REPORT_LEGION_ERROR(ERROR_PHYSICAL_TRACING_UNSUPPORTED_OP,
              "Illegal operation in physical trace. The application launched "
              "a %s operation inside of physical trace %d of parent task %s "
              "(UID %lld) but this kind of operation is not supported for "
              "physical traces at the moment. You can request support but "
              "we can guarantee support for all kinds of operations in "
              "physical traces.", op->get_logging_name(), tid,
              context->get_task_name(), context->get_unique_id())
        // Check to see if we are doing safe tracing checks or not
        if (context->runtime->safe_tracing)
        {
          // Compute the hash for this operation
          Murmur3Hasher hasher;
          const Operation::OpKind kind = op->get_operation_kind();
          hasher.hash(kind);
          TaskID task_id = 0;
          if (kind == Operation::TASK_OP_KIND)
          {
#ifdef DEBUG_LEGION
            TaskOp *task = dynamic_cast<TaskOp*>(op);
            assert(task != NULL);
#else
            TaskOp *task = dynamic_cast<TaskOp*>(op);
#endif
            task_id = task->task_id;
            hasher.hash(task_id);
          }
          const unsigned num_regions = op->get_region_count();
          for (unsigned idx = 0; idx < num_regions; idx++)
          {
            const RegionRequirement &req = op->get_requirement(idx);
            hasher.hash(req.parent);
            hasher.hash(req.handle_type);
            if (req.handle_type == LEGION_PARTITION_PROJECTION)
              hasher.hash(req.partition);
            else
              hasher.hash(req.region);
            for (std::set<FieldID>::const_iterator it =
                  req.privilege_fields.begin(); it != 
                  req.privilege_fields.end(); it++)
              hasher.hash(*it);
            for (std::vector<FieldID>::const_iterator it =
                  req.instance_fields.begin(); it != 
                  req.instance_fields.end(); it++)
              hasher.hash(*it);
            hasher.hash(req.privilege);
            hasher.hash(req.prop);
            hasher.hash(req.redop);
            hasher.hash(req.tag);
            hasher.hash(req.flags);
            if (req.handle_type != LEGION_SINGULAR_PROJECTION)
              hasher.hash(req.projection);
            size_t projection_size = 0;
            const void *projection_args = 
              req.get_projection_args(&projection_size);
            if (projection_size > 0)
              hasher.hash(projection_args, projection_size);
          }
          uint64_t hash[2];
          hasher.finalize(hash);
          if (fixed)
          {
            if (verification_infos.size() <= verification_index)
              REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                  "Detected %d operations in trace %d of parent task %s "
                  "(UID %lld) which differs from the %zd operations that "
                  "where recorded in the first execution of the trace. "
                  "The number of operations in the trace must always "
                  "be the same across all executions of the trace.",
                  verification_index, tid, context->get_task_name(),
                  context->get_unique_id(), verification_infos.size())
            const VerificationInfo &info = 
              verification_infos[verification_index++];
            if (info.kind != kind)
              REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                  "Operation %s does match the recorded operation kind %s "
                  "for the %d operation in trace %d of parent task %s "
                  "(UID %lld). The same order of operations must be "
                  "issued every time a trace is executed.",
                  Operation::get_string_rep(kind), op->get_logging_name(),
                  verification_index-1, tid, 
                  context->get_task_name(), context->get_unique_id())
            if (info.task_id != task_id)
              REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                  "Task %d does match the recorded task %d for the %d task "
                  "in trace %d of parent task %s (UID %lld). The same order "
                  "of operations must be issued every time a trace is "
                  "executed.", task_id, info.task_id,
                  verification_index-1, tid, 
                  context->get_task_name(), context->get_unique_id())
            if (info.regions != num_regions)
            {
              if (kind == Operation::TASK_OP_KIND)
                REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                    "Task %s recorded %d region requirements for trace "
                    "%d in parent task %s (UID %lld) but was re-executed with "
                    "%d region requirements. The number of region requirements"
                    " recorded must always match the number re-executed for "
                    "each corresponding operation in the trace.",
                    op->get_logging_name(), info.regions, tid,
                    context->get_task_name(), context->get_unique_id(),
                    num_regions)
              else
                REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                    "Operation %s recorded %d region requirements for trace "
                    "%d in parent task %s (UID %lld) but was re-executed with "
                    "%d region requirements. The number of region requirements"
                    " must always match the number re-executed for each "
                    "corresponding operation in the trace.",
                    op->get_logging_name(), info.regions, tid,
                    context->get_task_name(), context->get_unique_id(),
                    num_regions)
            }
            if ((info.hash[0] != hash[0]) || (info.hash[1] != hash[1]))
            {
              if (kind == Operation::TASK_OP_KIND)
                REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                    "Task %s was replayed with different region requirements "
                    "for trace %d in parent task %s (UID %lld) than what it "
                    "had when it was recorded. Region requirement arguments "
                    "must match exactly every time a trace is executed.",
                    op->get_logging_name(), tid, context->get_task_name(),
                    context->get_unique_id())
              else
                REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                    "Operation %s was replayed with different region "
                    "requirements for trace %d in parent task %s (UID %lld) "
                    "than waht it had when it was recorded. Region "
                    "requirement arguments must match exactly every time a "
                    "trace is executed.", op->get_logging_name(),
                    tid, context->get_task_name(), context->get_unique_id())
            }
          }
          else
            verification_infos.emplace_back(
                VerificationInfo(kind, task_id, num_regions, hash));
        }
        if (fixed)
          return false;
      }
      if (static_translator != NULL)
        static_translator->push_dependences(dependences);
      return true;
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::check_operation_count(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(verification_index <= verification_infos.size());
#endif
      if (verification_index < verification_infos.size())
        REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                "Detected %d operations in trace %d of parent task %s "
                "(UID %lld) which differs from the %zd operations that "
                "where recorded in the first execution of the trace. "
                "The number of operations in the trace must always "
                "be the same across all executions of the trace.",
                verification_index, tid, context->get_task_name(),
                context->get_unique_id(), verification_infos.size())
      verification_index = 0;
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::skip_analysis(RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
#endif
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
        const size_t op_index = operations.size();
        op_map[key] = op_index;
        operations.push_back(key);
        replay_info.push_back(OperationInfo());
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
                                         it->dtype, it->dependent_mask);
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
                              dtype, dep_mask);
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
    void LogicalTrace::begin_logical_trace(FenceOp *fence_op)
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
    void LogicalTrace::end_logical_trace(FenceOp *op)
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
#endif

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
              LEGION_TRUE_DEPENDENCE, mask);
          // Then record our dependence on the close operation
          op->register_region_dependence(it->current_req_index,
              close_op, close_op->get_generation(), 0/*close index*/,
              LEGION_TRUE_DEPENDENCE, mask);
          // Dispatch this close op
          close_op->end_dependence_analysis();
        }
        else
        {
          // Can just record a normal dependence
          op->register_region_dependence(it->current_req_index,
              prev.first, prev.second, it->previous_req_index,
              it->dependence_type, mask);
        }
      }
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
    void TraceOp::pack_remote_operation(Serializer &rez, 
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
    }

#if 0
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
      has_blocking_call = has_block;
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
      tracing = false;
      current_template = NULL;
      is_recording = false;
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
        current_template->finalize(parent_ctx, this, has_blocking_call);
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
          ApEvent pending_deletion = physical_trace->record_capture(
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
#endif

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
    void TraceCompleteOp::initialize_complete(InnerContext *ctx,
                LogicalTrace *tr, Provenance *provenance, bool remove_reference)
    //--------------------------------------------------------------------------
    {
      initialize(ctx,tr->has_physical_trace() ? EXECUTION_FENCE : MAPPING_FENCE,
          false/*need future*/, provenance);
      trace = tr;
      tracing = false;
      has_blocking_call = trace->get_and_clear_blocking_call();
      remove_trace_reference = remove_reference;
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
        runtime->free_complete_op(this);
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
      trace->end_logical_trace(this);
      TraceOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (trace->has_physical_trace())
      {
        PhysicalTrace *physical = trace->get_physical_trace();
        physical->complete_physical_trace(this, map_applied_conditions,
            execution_preconditions, has_blocking_call);
      }
      if (remove_trace_reference && trace->remove_reference())
        delete trace;
      TraceOp::trigger_mapping();
    }

#if 0
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
#endif

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
      initialize(ctx,tr->has_physical_trace() ? EXECUTION_FENCE : MAPPING_FENCE,
                  false/*need future*/, provenance);
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
      trace->begin_logical_trace(this);
      TraceOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // All our mapping dependences are satisfied, check to see if we're
      // doing a physical replay, if we are then we need to refresh the 
      // equivalence sets for all the templates
      if (trace->has_physical_trace())
      {
        PhysicalTrace *physical = trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(!physical->has_current_template());
#endif
        std::set<RtEvent> refresh_ready;
        physical->refresh_condition_sets(this, refresh_ready);
        if (!refresh_ready.empty())
        {
          enqueue_ready_operation(Runtime::merge_events(refresh_ready));
          return;
        }
      }
      enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (trace->has_physical_trace())
      {
        PhysicalTrace *physical = trace->get_physical_trace();
        const bool replaying = physical->begin_physical_trace(this,
            map_applied_conditions, execution_preconditions);
        // Tell the parent context whether we are replaying
        parent_ctx->record_physical_trace_replay(mapped_event, replaying);
      }
      TraceOp::trigger_mapping();
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate* TraceBeginOp::create_fresh_template(
                                                        PhysicalTrace *physical)
    //--------------------------------------------------------------------------
    {
      return new PhysicalTemplate(physical, get_completion_event());
    }

    /////////////////////////////////////////////////////////////
    // TraceRecurrentOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceRecurrentOp::TraceRecurrentOp(Runtime *rt)
      : TraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceRecurrentOp::~TraceRecurrentOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void TraceRecurrentOp::initialize_recurrent(InnerContext *ctx,
        LogicalTrace *tr, LogicalTrace *prev, Provenance *prov, bool remove_ref)
    //--------------------------------------------------------------------------
    {
      TraceOp::initialize(ctx, tr->has_physical_trace() || 
          prev->has_physical_trace() ? EXECUTION_FENCE : MAPPING_FENCE,
          false/*need future*/, prov);
      trace = tr;
      tracing = false;
      previous = prev;
      has_blocking_call = previous->get_and_clear_blocking_call();
      if (trace == previous)
        has_intermediate_fence = trace->has_intermediate_fence();
      remove_trace_reference = remove_ref;
    }

    //--------------------------------------------------------------------------
    void TraceRecurrentOp::activate(void)
    //--------------------------------------------------------------------------
    {
      TraceOp::activate();
      previous = NULL;
      has_blocking_call = false;
      has_intermediate_fence = false;
      remove_trace_reference = false;
    }

    //--------------------------------------------------------------------------
    void TraceRecurrentOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      TraceOp::deactivate(false/*free*/);
      if (freeop)
        runtime->free_recurrent_op(this);
    }

    //--------------------------------------------------------------------------
    const char* TraceRecurrentOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_RECURRENT_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TraceRecurrentOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_RECURRENT_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TraceRecurrentOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // We don't optimize for recurrent replays of logical analysis
      // at the moment as it doesn't really seem worth it in most cases
      previous->end_logical_trace(this);
      trace->begin_logical_trace(this);
      TraceOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void TraceRecurrentOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> ready_events;
      if (trace != previous)
      {
        if (previous->has_physical_trace())
        {
          PhysicalTrace *physical = previous->get_physical_trace();
          if (physical->is_replaying())
            physical->complete_physical_trace(this, ready_events,
                execution_preconditions, has_blocking_call);
        }
        if (trace->has_physical_trace())
        {
          PhysicalTrace *physical = trace->get_physical_trace();
          physical->refresh_condition_sets(this, ready_events);
        }
      }
      else if (trace->has_physical_trace())
      {
        PhysicalTrace *physical = trace->get_physical_trace();
        if (physical->is_recording())
          physical->refresh_condition_sets(this, ready_events);
        else if (!physical->get_current_template()->is_idempotent())
        {
          physical->refresh_condition_sets(this, ready_events);
          physical->complete_physical_trace(this, ready_events,
              execution_preconditions, has_blocking_call);
        }
      }
      if (!ready_events.empty())
        enqueue_ready_operation(Runtime::merge_events(ready_events));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void TraceRecurrentOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if this is a true recurrent replay or not
      if (trace != previous)
      {
        // Not recurrent so complete the previous trace and begin the new one
        if (previous->has_physical_trace())
        {
          PhysicalTrace *physical = previous->get_physical_trace();
          if (physical->is_recording())
            physical->complete_physical_trace(this, map_applied_conditions,
                execution_preconditions, has_blocking_call);
        }
        if (trace->has_physical_trace())
        {
          PhysicalTrace *physical = trace->get_physical_trace();
          const bool replaying = physical->begin_physical_trace(this,
              map_applied_conditions, execution_preconditions);
          // Tell the parent whether we are replaying
          parent_ctx->record_physical_trace_replay(mapped_event, replaying);
        }
      }
      else if (trace->has_physical_trace())
      {
        // This is recurrent, so try to do the recurrent replay
        PhysicalTrace *physical = trace->get_physical_trace();
        const bool replaying = physical->replay_physical_trace(this,
            map_applied_conditions, execution_preconditions,
            has_blocking_call, has_intermediate_fence);
        // Tell the parent whether we are replaying
        parent_ctx->record_physical_trace_replay(mapped_event, replaying);
      }
      if (remove_trace_reference && previous->remove_reference())
        delete previous;
      TraceOp::trigger_mapping();
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate* TraceRecurrentOp::create_fresh_template(
                                                        PhysicalTrace *physical)
    //--------------------------------------------------------------------------
    {
      return new PhysicalTemplate(physical, get_completion_event());
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTrace
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTrace::PhysicalTrace(Runtime *rt, LogicalTrace *lt)
      : runtime(rt), logical_trace(lt), perform_fence_elision(
          !(runtime->no_trace_optimization || runtime->no_fence_elision)),
        current_template(NULL), nonreplayable_count(0),
        new_template_count(0), recording(false), recurrent(false)
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
      std::set<RtEvent> deleted_events;
      ApEvent pending_deletion = ApEvent::NO_AP_EVENT;
      for (std::vector<PhysicalTemplate*>::iterator it =
           templates.begin(); it != templates.end(); ++it)
        if (!(*it)->defer_template_deletion(pending_deletion, deleted_events))
          delete (*it);
      templates.clear();
      if (!deleted_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(deleted_events);
        wait_on.wait();
      }
    }

#if 0
    //--------------------------------------------------------------------------
    ApEvent PhysicalTrace::record_capture(PhysicalTemplate *tpl,
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
    bool PhysicalTrace::check_template_preconditions(TraceBeginOp *op,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_template == NULL);
#endif
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
#endif

    //--------------------------------------------------------------------------
    void PhysicalTrace::record_parent_req_fields(unsigned index,
                                                 const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      LegionMap<unsigned,FieldMask>::iterator finder =
        parent_req_fields.find(index);
      if (finder == parent_req_fields.end())
        parent_req_fields[index] = mask;
      else
        finder->second |= mask;
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::find_condition_sets(
                         std::map<EquivalenceSet*,unsigned> &current_sets) const
    //--------------------------------------------------------------------------
    {
      InnerContext *context = logical_trace->context;
      for (LegionMap<unsigned,FieldMask>::const_iterator it =
            parent_req_fields.begin(); it != parent_req_fields.end(); it++)
        context->find_trace_local_sets(it->first, it->second, current_sets);
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::refresh_condition_sets(FenceOp *op,
                                          std::set<RtEvent> &ready_events) const
    //--------------------------------------------------------------------------
    {
      // Make sure all the templates have up-to-date equivalence sets for
      // performing any kind of tests on preconditions/postconditions
      for (std::vector<PhysicalTemplate*>::const_iterator it =
            templates.begin(); it != templates.end(); it++)
        if ((*it) != current_template)
          (*it)->refresh_condition_sets(op, ready_events);
    }

    //--------------------------------------------------------------------------
    bool PhysicalTrace::find_replay_template(BeginOp *op,
                                     std::set<RtEvent> &map_applied_events,
                                     std::set<ApEvent> &execution_preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_template == NULL);
#endif
      if (templates.empty())
        return false;
      // Start the first batch of precondition tests
      RtEvent next_ready;
      RtEvent current_ready = templates.back()->test_preconditions(
          op->get_begin_operation(), map_applied_events);
      // Scan backwards since more recently used templates are likely
      // to be the ones that best match what we are executing
      std::vector<unsigned> to_delete;
      for (int idx = templates.size() - 1; idx >= 0; idx--)
      {
        // If it's not the first or the last iteration then we prefetch
        // the following iteration. On the first iteration we hope that
        // template will be ready right away. On the last iteration then
        // there is nothing to prefetch.
        if ((idx > 0) && (idx < (int(templates.size())-1)))
          next_ready = templates[idx-1]->test_preconditions(
              op->get_begin_operation(), map_applied_events); 
        PhysicalTemplate *current = templates[idx];
        // Wait for the preconditions to be ready
        if (current_ready.exists() && !current_ready.has_triggered())
          current_ready.wait();
        bool valid = current->check_preconditions();
        bool acquired = valid ? current->acquire_instance_references() : false;
        // Now do the exchange between the operations to handle the case
        // of control replication to see if all the shards agree on what
        // to do with the template
        if (op->allreduce_template_status(valid, acquired || !valid))
        {
          // Delete now because couldn't acquire some instances
          if (acquired)
            current->release_instance_references();
          // Now delete this template from the entry since at least one of its
          // instances have been deleted and therefore we'll never be able to
          // replay it
          ApEvent pending_deletion;
          if (!current->defer_template_deletion(pending_deletion,
                                                map_applied_events))
            delete current;
          if (pending_deletion.exists())
            execution_preconditions.insert(pending_deletion);
          to_delete.push_back(idx);
        }
        else if (valid)
        {
          // Valid for everyone
#ifdef DEBUG_LEGION
          assert(acquired);
#endif
          if ((idx > 0) && (idx < (int(templates.size()) - 1)))
          {
            // Wait for the prefetched analyses to finish and clean them up
            if (next_ready.exists() && !next_ready.has_triggered())
              next_ready.wait();
            templates[idx-1]->check_preconditions();
          }
          // Everybody agreed to reuse this template so make it the
          // new current template and shuffle it to the front
          current_template = current;
          // Remove any deleted templates before rearranging, by definition
          // all these will be later in the vector than the current template
          // Note this will delete back to front to avoid invalidating
          // indexes later in the to_delete vector
          for (std::vector<unsigned>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
            templates.erase(templates.begin() + (*it));
          // Move the template to the end of the vector as most-recently used
          if (idx < int(templates.size() - 1))
            std::rotate(templates.begin()+idx, 
                        templates.begin()+idx+1, templates.end());
          return true;
        }
        else if (acquired)
          current->release_instance_references();
        if (idx > 0)
        {
          // If this is the first iteration then we start testing the
          // preconditions for the next iteration now too
          if (idx == (int(templates.size() - 1)))
            current_ready = templates[idx-1]->test_preconditions(
                op->get_begin_operation(), map_applied_events);
          else // Shuffle the ready events
            current_ready = next_ready;
        }
      }
      for (std::vector<unsigned>::const_iterator it =
            to_delete.begin(); it != to_delete.end(); it++)
        templates.erase(templates.begin() + (*it));
      return false;
    }

    //--------------------------------------------------------------------------
    bool PhysicalTrace::begin_physical_trace(BeginOp *op,
        std::set<RtEvent> &map_applied_conditions,
        std::set<ApEvent> &execution_preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_template == NULL);
#endif
      const bool replaying = find_replay_template(op,
            map_applied_conditions, execution_preconditions);
      if (replaying)
      {
        begin_replay(op, false/*recurrent*/, false/*has intermediate fence*/);
      }
      else // Start recording a new template
      {
        current_template = op->create_fresh_template(this);
        recording = true;
        recurrent = false;
      }
      return replaying;
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::complete_physical_trace(CompleteOp *op,
        std::set<RtEvent> &map_applied_conditions,
        std::set<ApEvent> &execution_preconditions, bool has_blocking_call)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_template != NULL);
#endif
      if (recording)
      {
        // Complete the recording and see if we have a new pending
        // deletion event that we need to capture
        if (complete_recording(op, map_applied_conditions,
              execution_preconditions, has_blocking_call))
          templates.push_back(current_template);
      }
      else
      {
        // If this isn't a recurrent replay then we need to apply the
        // postconditions to the equivalence sets, if it is recurrent
        // then we know that the postconditions have already been applied
        if (!recurrent)
          current_template->apply_postconditions(
              op->get_complete_operation(), map_applied_conditions);
        current_template->finish_replay(execution_preconditions);
        current_template->release_instance_references();
      }
      current_template = NULL;
    }

    //--------------------------------------------------------------------------
    bool PhysicalTrace::replay_physical_trace(RecurrentOp *op,
        std::set<RtEvent> &map_applied_conditions,
        std::set<ApEvent> &execution_preconditions, 
        bool has_blocking_call, bool has_intermediate_fence)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_template != NULL);
#endif
      PhysicalTemplate *non_idempotent_template = NULL;
      if (recording)
      {
        // Complete the recording. If we recorded a replayable template
        // and it is idempotent then we can replay it right away
        if (complete_recording(op, map_applied_conditions,
              execution_preconditions, has_blocking_call))
        {
          if (current_template->is_idempotent())
          {
            // Need to check if everyone can acquire all the instances
            bool valid = true;
            bool acquired = current_template->acquire_instance_references();
            if (op->allreduce_template_status(valid, acquired))
            {
              if (acquired)
                current_template->release_instance_references();
              // Now delete this template from the entry since at least one 
              // of its instances have been deleted and therefore we'll never
              // be able to replay it
              ApEvent pending_deletion;
              if (!current_template->defer_template_deletion(pending_deletion,
                                                      map_applied_conditions))
                delete current_template;
              if (pending_deletion.exists())
                execution_preconditions.insert(pending_deletion);
            }
            else
            {
#ifdef DEBUG_LEGION
              assert(valid);
#endif
              // Replaying this right away
              templates.push_back(current_template);
              // Treat the end of the recording as an intermediate fence
              // since we don't actually have events to use for a recurrent
              // replay quite yet since we just did the capture
              // We still set recurrent=true so we don't have to apply
              // the postconditions since we know that is unnecssary
              begin_replay(op,true/*recurrent*/,true/*has intermeidate fence*/);
              return true;
            }
          }
          else
            // Don't add this to the list of templates yet, we know it can't
            // be replayed right away so we don't want to check it
            non_idempotent_template = current_template;
        }
        // If we get here then we can't replay the current template so we
        // can just do a normal begin physical trace
        current_template = NULL;
      }
      else if (current_template != NULL)
      {
#ifdef DEBUG_LEGION
        // We should only be here if we're going to do a recurrent replay
        // If the current template was non-idempotent then it would have been
        // cleared by the TraceRecurrentOp in trigger_ready
        assert(current_template->is_idempotent());
#endif
        // If this isn't a recurrent replay then we need to apply the
        // postconditions to the equivalence sets, if it is recurrent
        // then we know that the postconditions have already been applied
        if (!recurrent)
          current_template->apply_postconditions(
              op->get_complete_operation(), map_applied_conditions);
        current_template->finish_replay(execution_preconditions);
        begin_replay(op, true/*recurrent*/, has_intermediate_fence);
        return true;
      }
      else
      {
        // This case occurs when have a recurrent trace with a non-idempotent
        // template. The TraceRecurrentOp will have completed the prior
        // template so the current template will have been cleared.
        // The most recent replayed template should be at the back of the
        // list of templates and it should be non-idempotent. There's no
        // point in considering it for replay since it is non-idempotent
        // and we know its preconditions aren't going to be satisfied so
        // we pop it off the list of templates and add it back once we've
        // decided what we're going to do.
#ifdef DEBUG_LEGION
        assert(!templates.empty());
        assert(!templates.back()->is_idempotent());
#endif
        non_idempotent_template = templates.back();
        templates.pop_back();
      }
#ifdef DEBUG_LEGION
      assert(current_template == NULL);
#endif
      if (non_idempotent_template != NULL)
      {
        // If we have a non-idempotent template we figure out what kind of
        // replay we're going to do and then put the non-idempotent template
        // in thie right place in the list of templates
        if (begin_physical_trace(op, map_applied_conditions,
              execution_preconditions))
        {
#ifdef DEBUG_LEGION
          assert(!templates.empty());
#endif
          // We found another template to replay so it will be the last
          // one on the list, therefore put the non-idempotent one right
          // before it on the list as the one most recently captured/replayed
          // before we found this new template to replay
          templates.insert(templates.end()-1, non_idempotent_template);  
          return true;
        }
        else
        {
          templates.push_back(non_idempotent_template);
          return false;
        }
      }
      else
        return begin_physical_trace(op, map_applied_conditions,
                                    execution_preconditions);
    }

    //--------------------------------------------------------------------------
    bool PhysicalTrace::complete_recording(CompleteOp *op,
            std::set<RtEvent> &map_applied_conditions,
            std::set<ApEvent> &execution_postconditions, bool has_blocking_call)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(current_template != NULL);
#endif
      // Reset the tracing state for the next time
      recording = false;
      ReplayableStatus status =
        current_template->finalize(op, has_blocking_call);
      if (status == REPLAYABLE)
      {
        // See if we're going to exceed the maximum number of templates
        if (templates.size() == 
            logical_trace->context->get_max_trace_templates())
        {
#ifdef DEBUG_LEGION
          assert(!templates.empty());
#endif
          PhysicalTemplate *to_delete = templates.front();
          ApEvent pending_deletion;
          if (!to_delete->defer_template_deletion(pending_deletion, 
                                            map_applied_conditions))
            delete to_delete;
          else if (pending_deletion.exists())
            execution_postconditions.insert(pending_deletion);
          // Remove the least recently used (first) one from the vector
          // shift it to the back first though, should be fast
          if (templates.size() > 1)
            std::rotate(templates.begin(),templates.begin()+1,templates.end());
          templates.pop_back();
        }
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
        return true;
      }
      else
      {
        // Record failed capture
        // We won't consider failure from mappers refusing to memoize
        // as a warning that gets bubbled up to end users.
        if ((status != NOT_REPLAYABLE_CONSENSUS) &&
            (status != NOT_REPLAYABLE_REMOTE_SHARD) &&
            (++nonreplayable_count > LEGION_NON_REPLAYABLE_WARNING))
        {
          InnerContext *ctx = logical_trace->context;
          REPORT_LEGION_WARNING(LEGION_WARNING_NON_REPLAYABLE_COUNT_EXCEEDED,
              "WARNING: The runtime has failed to memoize the trace more than "
              "%u times, due to the absence of a replayable template. It is "
              "highly likely that trace %u in task %s (UID %lld) will not be "
              "memoized for the rest of execution. The most recent template was "
              "not replayable for the following reason: %s. Please change the "
              "mapper to stop making memoization requests.",
              LEGION_NON_REPLAYABLE_WARNING, logical_trace->get_trace_id(),
              ctx->get_task_name(), ctx->get_unique_id(), 
              (status == NOT_REPLAYABLE_BLOCKING) ?
              "blocking call" : "virtual mapping")
          nonreplayable_count = 0;
        }
        // Defer template deletion
        ApEvent pending_deletion;
        if (!current_template->defer_template_deletion(pending_deletion,
                                                  map_applied_conditions))
          delete current_template;
        else if (pending_deletion.exists())
          execution_postconditions.insert(pending_deletion);
        return false;
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::begin_replay(BeginOp *op, bool recur,
                                     bool has_intermediate_fence)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_template != NULL);
#endif
      recording = false;
      recurrent = recur;
      new_template_count = 0;
      // If we had an intermeidate execution fence between replays then
      // we should no longer be considered recurrent when we replay the trace
      // We're also not going to be considered recurrent here if we didn't
      // do fence elision since since we'll still need to track the fence
      current_template->initialize_replay(op->get_begin_completion(),
          recurrent && perform_fence_elision && !has_intermediate_fence);
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
        ss << "fill view: " << std::hex << view->did << std::dec
           << ", Index expr: " << expr->expr_id
           << ", Field Mask: " << m;
      }
      else if (view->is_collective_view())
      {
        CollectiveView *collective = view->as_collective_view();
        ss << "collective view: " << std::hex << view->did << std::dec
           << ", Index expr: " << expr->expr_id
           << ", Field Mask: " << m;
        const char *mem_names[] = {
#define MEM_NAMES(name, desc) #name,
            REALM_MEMORY_KINDS(MEM_NAMES) 
#undef MEM_NAMES
          };
        bool first = true;
        for (std::vector<DistributedID>::const_iterator it =
              collective->instances.begin(); it != 
              collective->instances.end(); it++)
        {
          RtEvent ready;
          PhysicalManager *manager = 
            ctx->runtime->find_or_request_instance_manager(*it, ready);
          if (ready.exists())
            ready.wait();
          if (first)
          {
            ss << ", Fields: ";
            FieldSpaceNode *field_space = manager->field_space_node;
            std::vector<FieldID> fields;
            field_space->get_field_set(mask, ctx, fields);
            for (std::vector<FieldID>::const_iterator fit =
                  fields.begin(); fit != fields.end(); fit++)
            {
              if (fit != fields.begin())
                ss << ", ";
              const void *name = NULL;
              size_t name_size = 0;
              if (field_space->retrieve_semantic_information(
                    LEGION_NAME_SEMANTIC_TAG, name, name_size,
                    true/*can fail*/, false/*wait until*/))
                ss << ((const char*)name) << " (" << *fit << ")";
              else
                ss << *fit;
            }
            ss << ", Instances: ";
            first = false;
          }
          Memory memory = manager->memory_manager->memory;
          ss << "Instance " << std::hex << *it << std::dec
             << " (" << std::hex << manager->get_instance().id 
             << std::dec << ")"
             << " in " << mem_names[memory.kind()]
             << " Memory " << std::hex << memory.id << std::dec;
        }
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

        ss << "Instance " << std::hex << manager->did << std::dec
           << " (" << std::hex << manager->get_instance().id << std::dec << ")"
           << " in " << mem_names[memory.kind()]
           << " Memory " << std::hex << memory.id << std::dec
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
        if (vit->first->remove_nested_gc_ref(owner_did))
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
        view->add_nested_gc_ref(owner_did);
        expr->add_nested_expression_reference(owner_did);
        conditions[view].insert(expr, mask);
        if (view->is_collective_view())
          has_collective_views = true;
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::insert(LegionMap<LogicalView*,
                  FieldMaskSet<IndexSpaceExpression> > &views, bool antialiased)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMaskSet<IndexSpaceExpression> >::
            const_iterator vit = views.begin(); vit != views.end(); vit++)
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
          insert(vit->first, it->first, it->second);
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
          else if (view->remove_nested_gc_ref(owner_did))
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
            else if (view->remove_nested_gc_ref(owner_did))
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
          else if (view->remove_nested_gc_ref(owner_did))
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
#ifdef DEBUG_LEGION
      assert(!!non_dominated);
#endif
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
      if (finder != conditions.end() && 
          !(finder->second.get_valid_mask() * non_dominated))
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
            return true;
        }
      }
#ifdef DEBUG_LEGION
      assert(!!non_dominated);
#endif
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
            // No need to be precise here since the resulting analysis
            // on the leaves is filtering and not computing a union
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
              return true;
          }
        }
      }
      // If there are no fields left then we dominated
      return !non_dominated;
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::dominates(LogicalView *view, 
                    IndexSpaceExpression *expr, FieldMask mask,
                    LegionMap<LogicalView*,
                      FieldMaskSet<IndexSpaceExpression> > &non_dominated) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(non_dominated.empty());
#endif
      // If this is for an empty equivalence set then it doesn't matter
      if (expr->is_empty())
        return;
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
      if (finder != conditions.end() && 
          !(finder->second.get_valid_mask() * mask))
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
              IndexSpaceExpression *diff = 
                forest->subtract_index_spaces(expr, intersection);
              non_dominated[view].insert(diff, overlap);
            }
          } 
          mask -= overlap;
          // Make sure we keep going if we have non-dominated because
          // we need to check it against any collective aliasing
          if (!mask)
          {
            if (non_dominated.empty() ||
                (!has_collective_views && !view->is_collective_view()))
              return;
            else
              break;
          }
        }
        if (!!mask)
          non_dominated[view].insert(expr, mask);
      }
      else
        non_dominated[view].insert(expr, mask);
#ifdef DEBUG_LEGION
      assert(!non_dominated.empty());
#endif
      FieldMaskSet<IndexSpaceExpression> &non_view = non_dominated[view];
      // Now do the checks for any aliasing with collective views 
      if (view->is_collective_view())
      {
        CollectiveView *collective_view = view->as_collective_view();
        CollectiveAntiAlias alias_analysis(collective_view);
        for (ViewExprs::const_iterator vit =
              conditions.begin(); vit != conditions.end(); vit++)
        {
          if (!vit->first->is_instance_view())
            continue;
          if (vit->second.get_valid_mask() * non_view.get_valid_mask())
            continue;
          InstanceView *inst_view = vit->first->as_instance_view();
          if (!collective_view->aliases(inst_view))
            continue;
          // Only record expressions that are relevant
          LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
            FieldMask> join;
          unique_join_on_field_mask_sets(non_view, vit->second, join);
          for (LegionMap<std::pair<IndexSpaceExpression*,
                IndexSpaceExpression*>,FieldMask>::const_iterator it =
                join.begin(); it != join.end(); it++)
          {
            if (it->first.first != it->first.second)
            {
              IndexSpaceExpression *overlap_expr = 
                forest->intersect_index_spaces(it->first.first,
                                               it->first.second);
              if (overlap_expr->is_empty())
                continue;
              if (it->first.first->get_volume() == overlap_expr->get_volume())
                alias_analysis.traverse(inst_view, it->second, it->first.first);
              else if (it->first.second->get_volume() == 
                        overlap_expr->get_volume())
                alias_analysis.traverse(inst_view, it->second,it->first.second);
              else
                alias_analysis.traverse(inst_view, it->second, overlap_expr);
            }
            else
              alias_analysis.traverse(inst_view, it->second, it->first.first);
          }
        }
        // For each of the non-dominated expressions go through the
        // alias analysis and get new expressions that are still not
        // dominated even after the alias analysis
        std::vector<IndexSpaceExpression*> to_remove;
        for (FieldMaskSet<IndexSpaceExpression>::iterator it =
              non_view.begin(); it != non_view.end(); it++)
        {
          FieldMask dominated_mask; 
          alias_analysis.visit_leaves(it->second, dominated_mask,
              context, tree_id, collective_view, non_dominated, 
              it->first, forest);
          // Remove any fields that were diffed
          if (!!dominated_mask)
          {
            it.filter(dominated_mask);
            if (!it->second)
              to_remove.push_back(it->first);
          }
        }
        for (std::vector<IndexSpaceExpression*>::const_iterator it =
              to_remove.begin(); it != to_remove.end(); it++)
          non_view.erase(*it);
        if (non_view.empty())
          non_dominated.erase(view);
      }
      else if (has_collective_views && view->is_instance_view())
      {
        IndividualView *individual_view = view->as_individual_view();
        for (ViewExprs::const_iterator vit =
              conditions.begin(); vit != conditions.end(); vit++)
        {
          if (!vit->first->is_collective_view())
            continue;
          if (vit->second.get_valid_mask() * non_view.get_valid_mask())
            continue;
          if (!individual_view->aliases(vit->first->as_collective_view()))
            continue;
          // Join on the fields to find expressions that match
          LegionMap<std::pair<IndexSpaceExpression*,
            IndexSpaceExpression*>,FieldMask> join;
          unique_join_on_field_mask_sets(non_view, vit->second, join);
          for (LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
                FieldMask>::const_iterator it = join.begin(); 
                it != join.end(); it++)
          {
            IndexSpaceExpression *difference = 
              forest->subtract_index_spaces(it->first.first, it->first.second);
            if (difference->get_volume() < it->first.first->get_volume())
            {
              FieldMaskSet<IndexSpaceExpression>::iterator finder =
                non_view.find(it->first.first);
              finder.filter(it->second);
              if (!finder->second)
                non_view.erase(finder);
              if (!difference->is_empty())
                non_view.insert(difference, it->second);
            }
          }
        }
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
            LegionMap<LogicalView*,
                      FieldMaskSet<IndexSpaceExpression> > non_dominated;
            set.dominates(vit->first, it->first, it->second, non_dominated);
            for (LegionMap<LogicalView*,
                  FieldMaskSet<IndexSpaceExpression> >::const_iterator dit =
                  non_dominated.begin(); dit != non_dominated.end(); dit++)
            {
              for (FieldMaskSet<IndexSpaceExpression>::const_iterator nit =
                    dit->second.begin(); nit != dit->second.end(); nit++)
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
      LegionMap<IndexSpaceExpression*,FieldMaskSet<LogicalView> > &target) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target.empty());
#endif
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); ++vit)
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
          target[it->first].insert(vit->first, it->second);
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
          vit->first->pack_global_ref();
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
        vit->first->add_nested_gc_ref(owner_did);
        vit->first->unpack_global_ref();
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
          if (view->is_fill_view())
          {
            log_tracing.info() << "  "
                      << "Fill View: " << std::hex << view->did << std::dec
                      << ", Index expr: " << it->first->expr_id
                      << ", Fields: " << mask;
          }
          else if (view->is_collective_view())
          {
            CollectiveView *collective = view->as_collective_view();
            std::stringstream ss;
            for (std::vector<DistributedID>::const_iterator cit =
                  collective->instances.begin(); cit != 
                  collective->instances.end(); cit++)
            {
              RtEvent ready;
              PhysicalManager *manager = 
                context->runtime->find_or_request_instance_manager(*cit, ready);
              if (ready.exists())
                ready.wait();
              ss << " Instance " << std::hex << manager->did << std::dec
                 << "(" << std::hex << manager->get_instance().id 
                 << std::dec << "),";
            }
            log_tracing.info() << "  Collective "
                      << (view->is_reduction_kind() ? "Reduction " : "")
                      << "View: " << std::hex << view->did << std::dec
                      << ", Index expr: " << it->first->expr_id
                      << ", Fields: " << mask
                      << ", Instances:" << ss.str();
          }
          else
          {
            PhysicalManager *manager = 
              view->as_individual_view()->get_manager();
            log_tracing.info() << "  "
                      << (view->is_reduction_view() ? 
                          "Reduction" : "Normal")
                      << " Instance " << std::hex << manager->did << std::dec
                      << "(" << std::hex << manager->get_instance().id 
                      << std::dec << ")"
                      << ", Index expr: " << it->first->expr_id
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
          if (finder->first->remove_nested_gc_ref(owner_did))
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
        view->add_nested_gc_ref(owner_did);
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
              diff_view->add_nested_gc_ref(owner_did);
            // 2. Make a new instance for the intersection, analyze it
            //    and record any overlapping expressions in to_add
            InstanceView *inter_view = 
              (intersection.size() == collective->instances.size()) ?
              collective : find_instance_view(intersection);
            if (to_add.find(inter_view) == to_add.end())
              inter_view->add_nested_gc_ref(owner_did);
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
              if (vit->first->remove_nested_gc_ref(owner_did))
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
            vit->first->remove_nested_gc_ref(owner_did);
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
    TraceConditionSet::TraceConditionSet(PhysicalTemplate *tpl,
                   unsigned req_index, RegionTreeID tid,
                   IndexSpaceExpression *expr,
                   FieldMaskSet<LogicalView> &&vws)
      : EqSetTracker(set_lock), owner(tpl), condition_expr(expr),
        views(vws), tree_id(tid), parent_req_index(req_index), shared(false)
    //--------------------------------------------------------------------------
    {
      condition_expr->add_base_expression_reference(TRACE_REF);
      for (FieldMaskSet<LogicalView>::const_iterator it =
            views.begin(); it != views.end(); it++)
        it->first->add_base_gc_ref(TRACE_REF);
#ifdef DEBUG_LEGION
      analysis.invalid = NULL;
#endif
    }

    //--------------------------------------------------------------------------
    TraceConditionSet::~TraceConditionSet(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(equivalence_sets.empty());
      assert(analysis.invalid == NULL);
#endif
      if (condition_expr->remove_base_expression_reference(TRACE_REF))
        delete condition_expr;
      for (FieldMaskSet<LogicalView>::const_iterator it =
            views.begin(); it != views.end(); it++)
        if (it->first->remove_base_gc_ref(TRACE_REF))
          delete it->first;
    }

    //--------------------------------------------------------------------------
    bool TraceConditionSet::matches(IndexSpaceExpression *other_expr,
                             const FieldMaskSet<LogicalView> &other_views) const
    //--------------------------------------------------------------------------
    {
      if (condition_expr != other_expr)
        return false;
      if (views.size() != other_views.size())
        return false;
      for (FieldMaskSet<LogicalView>::const_iterator it =
            views.begin(); it != views.end(); it++)
      {
        FieldMaskSet<LogicalView>::const_iterator finder = 
          other_views.find(it->first);
        if (finder == other_views.end())
          return false;
        if (it->second != finder->second)
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::invalidate_equivalence_sets(void)
    //--------------------------------------------------------------------------
    {
      FieldMaskSet<EquivalenceSet> to_remove;
      LegionMap<AddressSpaceID,FieldMaskSet<EqKDTree> > to_cancel;
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
        to_cancel.swap(current_subscriptions);
      }
      cancel_subscriptions(owner->trace->runtime, to_cancel);
      for (FieldMaskSet<EquivalenceSet>::const_iterator it =
            to_remove.begin(); it != to_remove.end(); it++)
        if (it->first->remove_base_gc_ref(TRACE_REF))
          delete it->first;
    }

#if 0
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
#endif

    //--------------------------------------------------------------------------
    void TraceConditionSet::dump_conditions(void) const
    //--------------------------------------------------------------------------
    {
      TraceViewSet view_set(owner->trace->logical_trace->context, 0/*did*/,
          condition_expr, tree_id);
      for (FieldMaskSet<LogicalView>::const_iterator it =
            views.begin(); it != views.end(); it++)
        view_set.insert(it->first, condition_expr, it->second);
      view_set.dump();
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::test_preconditions(FenceOp *op, unsigned index, 
          std::vector<RtEvent> &ready_events, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // We should not need the lock here because the fence op should be 
      // blocking all other operations from running and changing the 
      // equivalence sets while we are here
#ifdef DEBUG_LEGION
      // We should already have refreshed the equivalence sets before we
      // get here so that they should all be up to date
      assert(!(views.get_valid_mask() - equivalence_sets.get_valid_mask()));
      assert(analysis.invalid == NULL);
#endif
      analysis.invalid = new InvalidInstAnalysis(op->runtime,  
            op, index, condition_expr, views);
      analysis.invalid->add_reference();
      std::set<RtEvent> deferral_events;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it =
            equivalence_sets.begin(); it != equivalence_sets.end(); it++)
      {
        const FieldMask overlap = views.get_valid_mask() & it->second;
        if (!overlap)
          continue;
        analysis.invalid->analyze(it->first, overlap, 
            deferral_events, applied_events);
      }
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis.invalid->has_remote_sets())
      {
        const RtEvent ready = 
          analysis.invalid->perform_remote(traversal_done, applied_events);
        if (ready.exists() && !ready.has_triggered())
          ready_events.push_back(ready);
      }
    }

    //--------------------------------------------------------------------------
    bool TraceConditionSet::check_preconditions(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(analysis.invalid != NULL);
#endif
      const bool result = !analysis.invalid->has_invalid();
      if (analysis.invalid->remove_reference())
        delete analysis.invalid;
#ifdef DEBUG_LEGION
      analysis.invalid = NULL;
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::test_anticonditions(FenceOp *op, unsigned index, 
          std::vector<RtEvent> &ready_events, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // We should not need the lock here because the fence op should be 
      // blocking all other operations from running and changing the 
      // equivalence sets while we are here
#ifdef DEBUG_LEGION
      // We should already have refreshed the equivalence sets before we
      // get here so that they should all be up to date
      assert(!(views.get_valid_mask() - equivalence_sets.get_valid_mask()));
      assert(analysis.invalid == NULL);
#endif
      analysis.antivalid = new AntivalidInstAnalysis(op->runtime, op, index, 
          condition_expr, views);
      analysis.antivalid->add_reference();
      std::set<RtEvent> deferral_events;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it =
            equivalence_sets.begin(); it != equivalence_sets.end(); it++)
      {
        const FieldMask overlap = views.get_valid_mask() & it->second;
        if (!overlap)
          continue;
        analysis.antivalid->analyze(it->first, overlap, 
            deferral_events, applied_events);
      }
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis.antivalid->has_remote_sets())
      {
        const RtEvent ready = 
          analysis.antivalid->perform_remote(traversal_done, applied_events);
        if (ready.exists() && !ready.has_triggered())
          ready_events.push_back(ready);
      }
    }

    //--------------------------------------------------------------------------
    bool TraceConditionSet::check_anticonditions(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(analysis.antivalid != NULL);
#endif
      const bool result = !analysis.antivalid->has_antivalid();
      if (analysis.antivalid->remove_reference())
        delete analysis.antivalid;
#ifdef DEBUG_LEGION
      analysis.invalid = NULL;
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::apply_postconditions(FenceOp *op, unsigned index, 
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // We should not need the lock here because the fence should be 
      // blocking all other operations from running and changing the 
      // equivalence sets while we are here
#ifdef DEBUG_LEGION
      // We should already have refreshed the equivalence sets before we
      // get here so that they should all be up to date
      assert(!(views.get_valid_mask() - equivalence_sets.get_valid_mask()));
#endif
      // Perform an overwrite analysis for each of the postconditions
      const TraceInfo trace_info(op);
      const RegionUsage usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
      OverwriteAnalysis *analysis = new OverwriteAnalysis(op->runtime,
          op, index, usage, condition_expr, views,
          PhysicalTraceInfo(trace_info, index), ApEvent::NO_AP_EVENT);
      analysis->add_reference();
      std::set<RtEvent> deferral_events;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it =
            equivalence_sets.begin(); it != equivalence_sets.end(); it++)
      {
        const FieldMask overlap = views.get_valid_mask() & it->second;
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

    //--------------------------------------------------------------------------
    void TraceConditionSet::refresh_equivalence_sets(FenceOp *op,
                                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      // We should not need the lock here because the fence op should be 
      // blocking all other operations from running and changing the 
      // equivalence sets while we are here
      const FieldMask invalid_mask = 
        views.get_valid_mask() - equivalence_sets.get_valid_mask();
      if (!!invalid_mask)
      {
        Runtime *runtime = owner->trace->runtime;
        AddressSpaceID space = runtime->address_space;
        // Create a user event and store it in equivalence_sets_ready
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
        std::vector<EqSetTracker*> targets(1, this);
        std::vector<AddressSpaceID> target_spaces(1, space);
        InnerContext *context = owner->trace->logical_trace->context;
        InnerContext *outermost =
          context->find_parent_physical_context(parent_req_index);
        RtEvent ready = context->compute_equivalence_sets(parent_req_index,
            targets, target_spaces, space, condition_expr, invalid_mask);
        if (ready.exists() && !ready.has_triggered())
        {
          // Launch a meta-task to finalize this trace condition set
          LgFinalizeEqSetsArgs args(this, compute_event, op->get_unique_op_id(),
              context, outermost, parent_req_index, condition_expr);
          runtime->issue_runtime_meta_task(args, 
                          LG_LATENCY_DEFERRED_PRIORITY, ready);
        }
        else
          finalize_equivalence_sets(compute_event, context, outermost, runtime,
              parent_req_index, condition_expr, op->get_unique_op_id());
        ready_events.insert(compute_event);
      }
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTemplate
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTemplate::PhysicalTemplate(PhysicalTrace *t, ApEvent fence_event)
      : trace(t), total_replays(1), replayable(REPLAYABLE), 
        idempotency(IDEMPOTENT), fence_completion_id(0),
        has_virtual_mapping(false), has_no_consensus(false), last_fence(NULL),
        remaining_replays(0), total_logical(0)
    //--------------------------------------------------------------------------
    {
      events.push_back(fence_event);
      event_map[fence_event] = fence_completion_id;
      finished_transitive_reduction.store(NULL);
      instructions.push_back(
         new AssignFenceCompletion(*this, fence_completion_id, TraceLocalID()));
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate::~PhysicalTemplate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(failure.view == NULL);
      assert(failure.expr == NULL);
#endif
      {
        AutoLock tpl_lock(template_lock); 
        for (std::vector<TraceConditionSet*>::const_iterator it =
              preconditions.begin(); it != preconditions.end(); it++)
        {
          (*it)->invalidate_equivalence_sets();
          if ((*it)->remove_reference())
            delete (*it);
        }
        for (std::vector<TraceConditionSet*>::const_iterator it =
              anticonditions.begin(); it != anticonditions.end(); it++)
        {
          (*it)->invalidate_equivalence_sets();
          if ((*it)->remove_reference())
            delete (*it);
        }
        for (std::vector<TraceConditionSet*>::const_iterator it =
              postconditions.begin(); it != postconditions.end(); it++)
        {
          (*it)->invalidate_equivalence_sets();
          if ((*it)->remove_reference())
            delete (*it);
        }
        for (std::vector<Instruction*>::iterator it = instructions.begin();
             it != instructions.end(); ++it)
          delete *it;
        cached_mappings.clear();
      }
      TransitiveReductionState *state = finished_transitive_reduction.load();
      if (state != NULL)
        delete state;
      for (std::map<DistributedID,IndividualView*>::const_iterator it =
            recorded_views.begin(); it != recorded_views.end(); it++)
        if (it->second->remove_base_gc_ref(TRACE_REF))
          delete it->second;
      for (std::set<IndexSpaceExpression*>::const_iterator it =
           recorded_expressions.begin(); it != recorded_expressions.end(); it++)
        if ((*it)->remove_base_expression_reference(TRACE_REF))
          delete (*it);
      for (std::vector<PhysicalManager*>::const_iterator it =
            all_instances.begin(); it != all_instances.end(); it++)
        if ((*it)->remove_base_gc_ref(TRACE_REF))
          delete (*it);
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
    void PhysicalTemplate::record_execution_fence(const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
      assert(!events.empty());
      assert(events.size() == instructions.size());
#endif
      // This is dumb, in the future we should find the frontiers
      // Scan backwards until we find the previous execution fence (if any)
      // Skip the most recent one as that is going to be our term event
      std::set<unsigned> preconditions;
      for (int idx = events.size() - 1; idx > 0; idx--)
      {
        if (events[idx].exists())
          preconditions.insert(idx);
        if (instructions[idx] == last_fence)
        {
          preconditions.insert(last_fence->complete);
          break;
        }
      }
      if (last_fence == NULL)
        preconditions.insert(0);
#ifdef DEBUG_LEGION
      assert(!preconditions.empty());
#endif
      unsigned complete = 0;
      if (preconditions.size() > 1)
      {
        // Record a merge event 
        complete = events.size();
        events.push_back(ApEvent());
        insert_instruction(
            new MergeEvent(*this, complete, preconditions, tlid));
      }
      else
        complete = *(preconditions.begin());
      events.push_back(ApEvent());
      CompleteReplay *fence = new CompleteReplay(*this, tlid, complete);
      insert_instruction(fence);
      // update the last fence
      last_fence = fence;
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalTemplate::test_preconditions(FenceOp *op,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      std::vector<RtEvent> ready_events;
      for (unsigned idx = 0; idx < preconditions.size(); idx++)
        preconditions[idx]->test_preconditions(op, idx, ready_events,
                                               applied_events);
      for (unsigned idx = 0; idx < anticonditions.size(); idx++)
        anticonditions[idx]->test_anticonditions(op, idx, ready_events,
                                                 applied_events);
      if (!ready_events.empty())
        return Runtime::merge_events(ready_events);
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::check_preconditions(void)
    //--------------------------------------------------------------------------
    {
      bool result = true;
      for (std::vector<TraceConditionSet*>::const_iterator it = 
            preconditions.begin(); it != preconditions.end(); it++)
        if (!(*it)->check_preconditions())
          result = false;
      for (std::vector<TraceConditionSet*>::const_iterator it = 
            anticonditions.begin(); it != anticonditions.end(); it++)
        if (!(*it)->check_anticonditions())
          result = false;
      return result;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::apply_postconditions(FenceOp *op,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < postconditions.size(); idx++)
        postconditions[idx]->apply_postconditions(op, idx, applied_events);
    }

#if 0
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
    PhysicalTemplate::DetailedBoolean PhysicalTemplate::check_idempotent(
        Operation* op, InnerContext* context)
    //--------------------------------------------------------------------------
    {
      // First let's get the equivalence sets with data for these regions
      // We'll use the result to get guide the creation of the trace condition
      // sets. Note we're going to end up recomputing the equivalence sets
      // inside the trace condition sets but that is a small price to pay to
      // minimize the number of conditions that we need to do the capture
      // Next we need to compute the equivalence sets for all these regions
      FieldMaskSet<EquivalenceSet> current_sets;
      std::map<EquivalenceSet*,unsigned> parent_req_indexes;
      {
        std::set<RtEvent> eq_events;
        const ContextID ctx = context->get_physical_tree_context();
        // Need to count how many version infos there are before we
        // start since we can't resize the vector once we start
        // compute the equivalence sets for any of them
        unsigned index = 0;
        for (LegionVector<FieldMaskSet<RegionNode> >::const_iterator it =
              trace_regions.begin(); it != trace_regions.end(); it++)
          index += it->size();
        LegionVector<VersionInfo> version_infos(index);
        index = 0;
        for (unsigned idx = 0; idx < trace_regions.size(); idx++)
        {
          for (FieldMaskSet<RegionNode>::const_iterator it = 
                trace_regions[idx].begin(); it !=
                trace_regions[idx].end(); it++)
          {
            it->first->perform_versioning_analysis(ctx, context,
                &version_infos[index++], it->second, op, 0/*index*/,
                idx, eq_events);
          }
        }
        index = 0;
        if (!eq_events.empty())
        {
          const RtEvent wait_on = Runtime::merge_events(eq_events);
          if (wait_on.exists() && !wait_on.has_triggered())
            wait_on.wait();
        }
        // Transpose over to equivalence sets
        for (unsigned idx = 0; idx < trace_regions.size(); idx++)
        {
          for (FieldMaskSet<RegionNode>::const_iterator rit = 
                trace_regions[idx].begin(); rit !=
                trace_regions[idx].end(); rit++)
          {
            const FieldMaskSet<EquivalenceSet> &region_sets = 
                version_infos[index++].get_equivalence_sets();
            for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                  region_sets.begin(); it != region_sets.end(); it++)
            {
              if (current_sets.insert(it->first, it->second))
                parent_req_indexes[it->first] = idx; 
#ifdef DEBUG_LEGION
              else
                assert(parent_req_indexes[it->first] == idx);
#endif
            }
            if (rit->first->remove_base_resource_ref(TRACE_REF))
              delete rit->first;
          }
        }
        trace_regions.clear();
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
        const RtEvent ready = condition->recompute_equivalence_sets(
            op->get_unique_op_id(), it->second);
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
        if (!(*it)->is_idempotent(not_subsumed, &condition))
        {
          if (trace->runtime->dump_physical_traces)
          {
            if (not_subsumed)
              return DetailedBoolean(
                  false, "precondition not subsumed: " +
                         condition.to_string(trace->logical_trace->context));
            else
              return DetailedBoolean(
                  false, "postcondition anti dependent: " +
                         condition.to_string(trace->logical_trace->context));
          }
          else
          {
            if (not_subsumed)
              return DetailedBoolean(
                  false, "precondition not subsumed by postcondition");
            else
              return DetailedBoolean(
                  false, "postcondition anti dependent");
          }
        }
        it++;
      }
      return DetailedBoolean(true);
    }
#endif

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::can_start_replay(void)
    //--------------------------------------------------------------------------
    {
      // This might look racy but its not. We only call this method when we
      // are replaying a physical template which by definition cannot happen
      // on the first trace execution. Therefore the entire logical analysis
      // will have finished recording all the operations before we start
      // replaying a single operation in the physical analysis
      const size_t op_count = trace->logical_trace->get_operation_count();
      const unsigned total = total_logical.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(total <= op_count);
#endif
      return (total == op_count);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::register_operation(MemoizableOp *memoizable)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      const TraceLocalID tid = memoizable->get_trace_local_id();
      
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(operations.find(tid) == operations.end());
#endif
      operations[tid] = memoizable;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::execute_slice(unsigned slice_idx,
                                         bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(slice_idx < slices.size());
#endif
      std::vector<Instruction*> &instructions = slices[slice_idx];
      for (std::vector<Instruction*>::const_iterator it = instructions.begin();
           it != instructions.end(); ++it)
        (*it)->execute(events, user_events, operations, recurrent_replay);
      unsigned remaining = remaining_replays.fetch_sub(1);
#ifdef DEBUG_LEGION
      assert(remaining > 0);
#endif
      if (remaining == 1)
      {
        AutoLock tpl_lock(template_lock);
        if (replay_postcondition.exists())
          Runtime::trigger_event(replay_postcondition);
      }
    }

    //--------------------------------------------------------------------------
    ReplayableStatus PhysicalTemplate::finalize(CompleteOp *op,
                                                bool has_blocking_call)
    //--------------------------------------------------------------------------
    {
      if (has_no_consensus.load())
        replayable = NOT_REPLAYABLE_CONSENSUS;
      else if (has_blocking_call)
        replayable = NOT_REPLAYABLE_BLOCKING;
      else if (has_virtual_mapping)
        replayable = NOT_REPLAYABLE_VIRTUAL;
      op->begin_replayable_exchange(replayable);
      idempotency = capture_conditions(op); 
      op->begin_idempotent_exchange(idempotency);
      op->end_replayable_exchange(replayable);
      if (is_replayable())
      {
        // The user can't ask for both no transitive reduction and inlining
        // of the transitive reduction.
        assert(!(trace->runtime->no_transitive_reduction &&
                 trace->runtime->inline_transitive_reduction));
        // Optimize will sync the idempotency computation
        optimize(op, trace->runtime->inline_transitive_reduction);
        std::fill(events.begin(), events.end(), ApEvent::NO_AP_EVENT);
        event_map.clear();
        // Defer performing the transitive reduction because it might
        // be expensive (see comment above). Note you can only kick off
        // the transitive reduction in the background once all the other
        // optimizations are done so that they don't race on mutating
        // the instruction and event data structures
        if (!trace->runtime->no_trace_optimization &&
            !trace->runtime->no_transitive_reduction &&
            !trace->runtime->inline_transitive_reduction)
        {
          TransitiveReductionState *state = 
            new TransitiveReductionState(Runtime::create_rt_user_event());
          transitive_reduction_done = state->done;
          TransitiveReductionArgs args(this, state);
          trace->runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY);
        }
        // Can dump now if we're not deferring the transitive reduction
        else if (trace->runtime->dump_physical_traces)
          dump_template();
      }
      else
      {
        if (trace->runtime->dump_physical_traces)
        {
          // Optimize will sync the idempotency computation
          optimize(op, !trace->runtime->no_transitive_reduction);
          dump_template();
        }
        else
          op->end_idempotent_exchange(idempotency);
      }
      return replayable;
    }

    //--------------------------------------------------------------------------
    IdempotencyStatus PhysicalTemplate::capture_conditions(CompleteOp *op)
    //--------------------------------------------------------------------------
    {
      // First let's get the equivalence sets with data for these regions
      // We'll use the result to get guide the creation of the trace condition
      // sets. Note we're going to end up recomputing the equivalence sets
      // inside the trace condition sets but that is a small price to pay to
      // minimize the number of conditions that we need to do the capture
      // Next we need to compute the equivalence sets for all these regions
      std::map<EquivalenceSet*,unsigned> current_sets;
      trace->find_condition_sets(current_sets);

      // For cases of control replication we need to deduplicate the 
      // condition sets so that each shard only captures one equivalence set
      op->deduplicate_condition_sets(current_sets);

      std::vector<RtEvent> ready_events; 
      std::atomic<unsigned> result(IDEMPOTENT); 
      for (std::map<EquivalenceSet*,unsigned>::const_iterator it =
            current_sets.begin(); it != current_sets.end(); it++)
      {
        RtEvent ready = it->first->capture_trace_conditions(this,
            it->first->local_space, it->second, &result);
        if (ready.exists())
          ready_events.push_back(ready);
      }
#if 0
      // Make a trace condition set for each one of them
      std::vector<RtEvent> ready_events;
      conditions.reserve(current_sets.size());
      RegionTreeForest *forest = trace->runtime->forest;
      for (std::map<EquivalenceSet*,unsigned>::const_iterator it =
            current_sets.begin(); it != current_sets.end(); it++)
      {
        CollectiveMapping *mapping = NULL;
        std::map<EquivalenceSet*,CollectiveMapping*>::const_iterator
          finder = collective_conditions.find(it->first);
        if (finder != collective_conditions.end())
        {
          mapping = finder->second;
          collective_conditions.erase(finder);
        }
        TraceConditionSet *condition = new TraceConditionSet(trace, forest,
            mapping, it->second, it->first->tree_id);
        condition->add_reference();
#if 0
        // This looks redundant because it is a bit since we're just going
        // to compute the single equivalence set we already have here but
        // really what we're doing here is registering the condition with
        // the VersionManager that owns this equivalence set which is a
        // necessary thing for us to do
        const RtEvent ready = condition->recompute_equivalence_sets(
            operation->get_unique_op_id(), it->second);
        if (ready.exists())
          ready_events.push_back(ready);
#endif
        condition->capture(it->first, ready_events);
        conditions.push_back(condition);
      }
#ifdef DEBUG_LEGION
      assert(collective_conditions.empty());
#endif
#endif
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        ready_events.clear();
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
#if 0
      IdempotencyStatus status = IDEMPOTENT;
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
        bool not_subsumed = false;
        // Check all of them so that they each set their states
        if (!(*it)->is_idempotent(not_subsumed))
        {
          if (not_subsumed)
            status = NOT_IDEMPOTENT_SUBSUMPTION;
          else
            status = NOT_IDEMPOTENT_ANTIDEPENDENT;
        }
        it++;
      }
      return status;
#endif
      return static_cast<IdempotencyStatus>(result.load());
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::receive_trace_conditions(TraceViewSet *previews,
                       TraceViewSet *antiviews, TraceViewSet *postviews,
                       unsigned parent_req_index, RegionTreeID tree_id,
                       std::atomic<unsigned> *result)
    //--------------------------------------------------------------------------
    {
      // First check to see if these conditions are idempotent or not  
      TraceViewSet::FailedPrecondition fail;
      if ((previews != NULL) && (postviews != NULL) &&
          !previews->subsumed_by(*postviews, true/*allow independent*/, &fail))
      {
        unsigned initial = IDEMPOTENT;
        if (result->compare_exchange_strong(initial,
              NOT_IDEMPOTENT_SUBSUMPTION) &&
            trace->runtime->dump_physical_traces)
        {
          failure = fail;
          if (failure.view != NULL)
            failure.view->add_base_resource_ref(TRACE_REF);
          if (failure.expr != NULL)
            failure.expr->add_base_expression_reference(TRACE_REF);
        }
      }
      else if ((postviews != NULL) && (antiviews != NULL) &&
          !postviews->independent_of(*antiviews, &fail))
      {
        unsigned initial = IDEMPOTENT;
        if (result->compare_exchange_strong(initial, 
              NOT_IDEMPOTENT_ANTIDEPENDENT) && 
            trace->runtime->dump_physical_traces)
        {
          failure = fail;
          if (failure.view != NULL)
            failure.view->add_base_resource_ref(TRACE_REF);
          if (failure.expr != NULL)
            failure.expr->add_base_expression_reference(TRACE_REF);
        }
      }
      // Now we can convert these views into conditions
      std::vector<TraceConditionSet*> postsets;
      // Create the postconditions first so we can see if we can share
      // them with any of the preconditions or anticonditions
      if (postviews != NULL)
      {
        LegionMap<IndexSpaceExpression*,FieldMaskSet<LogicalView> > expr_views;
        postviews->transpose_uniquely(expr_views);
        postsets.reserve(expr_views.size());
        for (LegionMap<IndexSpaceExpression*,FieldMaskSet<LogicalView> >::
              iterator it = expr_views.begin(); it != expr_views.end(); it++)
        {
          TraceConditionSet *set = new TraceConditionSet(this, parent_req_index,
              tree_id, it->first, std::move(it->second));
          set->add_reference();
          postsets.push_back(set);
        }
        AutoLock tpl_lock(template_lock);
        postconditions.insert(postconditions.end(),
            postsets.begin(), postsets.end());
      }
      // Next do the previews and the antiviews looking for sharing with
      // the postviews so we can minimize the number of EqSetTrackers
      if (previews != NULL)
      {
        LegionMap<IndexSpaceExpression*,FieldMaskSet<LogicalView> > expr_views;
        previews->transpose_uniquely(expr_views);
        for (LegionMap<IndexSpaceExpression*,FieldMaskSet<LogicalView> >::
              iterator eit = expr_views.begin(); eit != expr_views.end(); eit++)
        {
          TraceConditionSet *set = NULL;
          for (std::vector<TraceConditionSet*>::iterator it =
                postsets.begin(); it != postsets.end(); it++)
          {
            if (!(*it)->matches(eit->first, eit->second))
              continue;
            set = *it;
            postsets.erase(it);
            break;
          }
          if (set == NULL)
            set = new TraceConditionSet(this, parent_req_index, tree_id, 
                eit->first, std::move(eit->second));
          else
            set->mark_shared();
          set->add_reference();
          AutoLock tpl_lock(template_lock);
          preconditions.push_back(set);
        }
      }
      if (antiviews != NULL)
      {
        LegionMap<IndexSpaceExpression*,FieldMaskSet<LogicalView> > expr_views;
        antiviews->transpose_uniquely(expr_views);
        for (LegionMap<IndexSpaceExpression*,FieldMaskSet<LogicalView> >::
              iterator eit = expr_views.begin(); eit != expr_views.end(); eit++)
        {
          TraceConditionSet *set = NULL;
          for (std::vector<TraceConditionSet*>::iterator it =
                postsets.begin(); it != postsets.end(); it++)
          {
            if (!(*it)->matches(eit->first, eit->second))
              continue;
            set = *it;
            postsets.erase(it);
            break;
          }
          if (set == NULL)
            set = new TraceConditionSet(this, parent_req_index, tree_id, 
                eit->first, std::move(eit->second));
          else
            set->mark_shared();
          set->add_reference();
          AutoLock tpl_lock(template_lock);
          anticonditions.push_back(set);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::refresh_condition_sets(FenceOp *op,
                                          std::set<RtEvent> &ready_events) const
    //--------------------------------------------------------------------------
    {
      for (std::vector<TraceConditionSet*>::const_iterator it =
            preconditions.begin(); it != preconditions.end(); it++)
        (*it)->refresh_equivalence_sets(op, ready_events);
      for (std::vector<TraceConditionSet*>::const_iterator it =
            anticonditions.begin(); it != anticonditions.end(); it++)
        (*it)->refresh_equivalence_sets(op, ready_events);
      for (std::vector<TraceConditionSet*>::const_iterator it =
            postconditions.begin(); it != postconditions.end(); it++)
        if (!(*it)->is_shared())
          (*it)->refresh_equivalence_sets(op, ready_events);
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::acquire_instance_references(void) const
    //--------------------------------------------------------------------------
    {
      for (std::vector<PhysicalManager*>::const_iterator it =
            all_instances.begin(); it != all_instances.end(); it++)
      {
        if (!(*it)->acquire_instance(TRACE_REF))
        {
          // Remove all the references we already added up to now
          // No need to check for deletion, we stil have gc references
          for (std::vector<PhysicalManager*>::const_iterator it2 =
                all_instances.begin(); it2 != it; it2++)
            (*it2)->remove_base_valid_ref(TRACE_REF);
          return false;
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::release_instance_references(void) const
    //--------------------------------------------------------------------------
    {
      // No need to check for deletions, we stil hold gc references
      for (std::vector<PhysicalManager*>::const_iterator it =
            all_instances.begin(); it != all_instances.end(); it++)
        (*it)->remove_base_valid_ref(TRACE_REF);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::optimize(CompleteOp *op,bool do_transitive_reduction)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
      std::vector<RtEvent> frontier_events;
      find_all_last_instance_user_events(frontier_events);
      compute_frontiers(frontier_events);
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
      std::vector<unsigned> gen;
      // Sync the idempotency computation 
      op->end_idempotent_exchange(idempotency);
      // Fence elision can only be performed if the template is idempotent.
      if (!is_idempotent() || !trace->perform_fence_elision)
      {
        gen.resize(events.size(), 0/*fence instruction*/);
        for (unsigned idx = 0; idx < instructions.size(); ++idx)
          gen[idx] = idx;
      }
      else
        elide_fences(gen, frontier_events);
      // Sync the frontier computation so we know that all our frontier data
      // structures such as 'local_frontiers' and 'remote_frontiers' are ready
      sync_compute_frontiers(op, frontier_events);
      if (!trace->runtime->no_trace_optimization)
      {
        propagate_merges(gen);
        if (do_transitive_reduction)
        {
          TransitiveReductionState state(RtUserEvent::NO_RT_USER_EVENT);
          transitive_reduction(&state, false/*deferred*/);
        }
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
      // We do need to translate the recorded views into a vector of instances
      // that need to be acquired in order for the template to be replayed
      all_instances.reserve(recorded_views.size());
      for (std::map<DistributedID,IndividualView*>::const_iterator it =
            recorded_views.begin(); it != recorded_views.end(); it++)
      {
        PhysicalManager *manager = it->second->get_manager();
        manager->add_base_gc_ref(TRACE_REF);
        all_instances.push_back(manager);
        if (it->second->remove_base_gc_ref(TRACE_REF))
          delete it->second;
      }
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
#ifdef DEBUG_LEGION
          assert(finder != recorded_views.end());
#endif
          PhysicalManager *manager = finder->second->get_manager();
#ifdef DEBUG_LEGION
          assert(manager->did == uit->instance.inst_did);
#endif
          const RegionUsage usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
          finder->second->find_last_users(manager, result.events, usage,
              uit->mask, uit->expr, frontier_events);
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
                instructions[complete->complete]->get_kind();
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
              rewrite_preconditions(replay->complete, users, instructions, 
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
              used[gen[complete->complete]] = true;
              break;
            }
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
          // Fence elision-style operations can only be performed if the
          // trace is idempotent.
          if (trace->perform_fence_elision && is_idempotent())
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
    void PhysicalTemplate::sync_compute_frontiers(CompleteOp *op,
                                    const std::vector<RtEvent> &frontier_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
        used[gen[last_fence->complete]] = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::prepare_parallel_replay(
                                               const std::vector<unsigned> &gen)
    //--------------------------------------------------------------------------
    {
      const size_t replay_parallelism = trace->runtime->max_replay_parallelism;
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
                parallelize_replay_event(inst->as_complete_replay()->complete,
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
    void PhysicalTemplate::transitive_reduction(
                                 TransitiveReductionState *state, bool deferred)
    //--------------------------------------------------------------------------
    {
      // Transitive reduction inspired by Klaus Simon,
      // "An improved algorithm for transitive closure on acyclic digraphs"

      // The transitive reduction can be a really long computation and we
      // don't want it monopolizing an entire processor while it's running
      // so we time-slice as background tasks until it is done. We pick the
      // somewhat arbitrary timeslice of 2ms since that's around the right
      // order of magnitude for other meta-tasks on most machines while still
      // being large enough to warm-up caches and make forward progress
      constexpr long long TIMEOUT = 2000; // in microseconds
      unsigned long long running_time = 0;
      unsigned long long previous_time = 
        Realm::Clock::current_time_in_microseconds();
      std::vector<unsigned> &topo_order = state->topo_order;
      std::vector<unsigned> &inv_topo_order = state->inv_topo_order;
      std::vector<std::vector<unsigned> > &incoming = state->incoming;
      std::vector<std::vector<unsigned> > &outgoing = state->outgoing;
      if (state->stage == 0)
      {
        topo_order.reserve(instructions.size());
        inv_topo_order.resize(events.size(), -1U);
        incoming.resize(events.size());
        outgoing.resize(events.size());

        initialize_transitive_reduction_frontiers(topo_order, inv_topo_order);
        state->stage++;
      }

      // First, build a DAG and find nodes with no incoming edges
      if (state->stage == 1)
      {  
        std::map<TraceLocalID, ReplayMapping*> &replay_insts = 
          state->replay_insts;
        for (unsigned idx = state->iteration; idx < instructions.size(); ++idx)
        {
          // Check for timeout
          if (deferred)
          {
            unsigned long long current_time = 
              Realm::Clock::current_time_in_microseconds();
            running_time += (current_time - previous_time);
            if (TIMEOUT <= running_time)
            {
              // Hit the timeout so launch a continuation
              state->iteration = idx;
              TransitiveReductionArgs args(this, state); 
              trace->runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY);
              return;
            }
            else
              previous_time = current_time;
          }
          Instruction *inst = instructions[idx];
          switch (inst->get_kind())
          {
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
                  incoming[replay_finder->second->lhs].push_back(replay->complete);
                  outgoing[replay->complete].push_back(replay_finder->second->lhs);
                  replay_insts.erase(replay_finder);
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
        state->stage++;
        state->iteration = 0;
        replay_insts.clear();
      }

      // Second, do a toposort on nodes via BFS
      if (state->stage == 2)
      {
        std::vector<unsigned> &remaining_edges = state->remaining_edges;
        if (remaining_edges.empty())
        {
          remaining_edges.resize(incoming.size());
          for (unsigned idx = 0; idx < incoming.size(); ++idx)
            remaining_edges[idx] = incoming[idx].size();
        }

        unsigned idx = state->iteration;
        while (idx < topo_order.size())
        {
          // Check for timeout
          if (deferred)
          {
            unsigned long long current_time = 
              Realm::Clock::current_time_in_microseconds();
            running_time += (current_time - previous_time);
            if (TIMEOUT <= running_time)
            {
              // Hit the timeout so launch a continuation
              state->iteration = idx;
              TransitiveReductionArgs args(this, state); 
              trace->runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY); 
              return;
            }
            else
              previous_time = current_time;
          }
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
        state->stage++;
        state->iteration = 0;
        remaining_edges.clear();
      }

      // Third, construct a chain decomposition
      if (state->stage == 3)
      {
        std::vector<unsigned> &chain_indices = state->chain_indices;
        if (chain_indices.empty())
        {
          chain_indices.resize(topo_order.size(), -1U);
          state->pos = chain_indices.size() - 1;
        }

        int pos = state->pos;
        unsigned num_chains = state->num_chains;
        while (true)
        {
          // Check for timeout
          if (deferred)
          {
            unsigned long long current_time = 
              Realm::Clock::current_time_in_microseconds();
            running_time += (current_time - previous_time);
            if (TIMEOUT <= running_time)
            {
              // Hit the timeout so launch a continuation
              state->pos = pos;
              state->num_chains = num_chains;
              TransitiveReductionArgs args(this, state); 
              trace->runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY); 
              return;
            }
            else
              previous_time = current_time;
          }
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
        state->stage++;
        state->num_chains = num_chains;
      }

      // Fourth, find the frontiers of chains that are connected to each node
      if (state->stage == 4)
      {
        const unsigned num_chains = state->num_chains;
        const std::vector<unsigned> &chain_indices = state->chain_indices;
        std::vector<std::vector<int> > &all_chain_frontiers = 
          state->all_chain_frontiers;
        if (all_chain_frontiers.empty())
          all_chain_frontiers.resize(topo_order.size());
        std::vector<std::vector<unsigned> > &incoming_reduced = 
          state->incoming_reduced;
        if (incoming_reduced.empty())
          incoming_reduced.resize(topo_order.size());
        for (unsigned idx = state->iteration; idx < topo_order.size(); idx++)
        {
          // Check for timeout
          if (deferred)
          {
            unsigned long long current_time = 
              Realm::Clock::current_time_in_microseconds();
            running_time += (current_time - previous_time);
            if (TIMEOUT <= running_time)
            {
              // Hit the timeout so launch a continuation
              state->iteration = idx;
              TransitiveReductionArgs args(this, state); 
              trace->runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY); 
              return;
            }
            else
              previous_time = current_time;
          }
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
        state->stage++;
        state->iteration = 0;
        all_chain_frontiers.clear();
      }

      // Lastly, suppress transitive dependences using chains
      if (deferred)
      {
        const RtUserEvent to_trigger = state->done;
        finished_transitive_reduction.store(state);
        Runtime::trigger_event(to_trigger);
      }
      else
        finalize_transitive_reduction(state->inv_topo_order, 
                                      state->incoming_reduced);
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
    void PhysicalTemplate::check_finalize_transitive_reduction(void)
    //--------------------------------------------------------------------------
    {
      TransitiveReductionState *state =
        finished_transitive_reduction.exchange(NULL);
      if (state != NULL)
      {
        finalize_transitive_reduction(state->inv_topo_order, 
                                      state->incoming_reduced);
        delete state;
        // We also need to rerun the propagate copies analysis to
        // remove any mergers which contain only a single input
        propagate_copies(NULL/*don't need the gen out*/);
        if (trace->runtime->dump_physical_traces)
          dump_template();
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
                substitutions.find(replay->complete);
              if (finder != substitutions.end())
                replay->complete = finder->second;
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
        // SetOpSyncEvent
        used[idx] = (kind != SET_OP_SYNC_EVENT);
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
              assert(gen[complete->complete] != -1U);
#endif
              used[gen[complete->complete]] = true;
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
    void PhysicalTemplate::dump_template(void) const
    //--------------------------------------------------------------------------
    {
      InnerContext *ctx = trace->logical_trace->context;
      log_tracing.info() << "#### Replayable: " << replayable 
        << ", Idempotent: " << idempotency << " " 
        << this << " Trace " << trace->logical_trace->tid << " for " 
        << ctx->get_task_name()
        << " (UID " << ctx->get_unique_id() << ") ####";
      if (idempotency == NOT_IDEMPOTENT_SUBSUMPTION)
      {
        log_tracing.info() << "Non-subsumed condition: "
                           << failure.to_string(trace->logical_trace->context);
        if ((failure.view != NULL) && 
            failure.view->remove_base_resource_ref(TRACE_REF))
          delete failure.view;
        failure.view = NULL;
        if ((failure.expr != NULL) &&
            failure.expr->remove_base_expression_reference(TRACE_REF))
          delete failure.expr;
        failure.expr = NULL;
      }
      else if (idempotency == NOT_IDEMPOTENT_ANTIDEPENDENT)
      {
        log_tracing.info() << "Anti-dependent condition: " 
                           << failure.to_string(trace->logical_trace->context);
        if ((failure.view != NULL) && 
            failure.view->remove_base_resource_ref(TRACE_REF))
          delete failure.view;
        failure.view = NULL;
        if ((failure.expr != NULL) &&
            failure.expr->remove_base_expression_reference(TRACE_REF))
          delete failure.expr;
        failure.expr = NULL;
      }
      const size_t replay_parallelism = trace->get_replay_targets().size();
      for (unsigned sidx = 0; sidx < replay_parallelism; ++sidx)
      {
        log_tracing.info() << "[Slice " << sidx << "]";
        dump_instructions(slices[sidx]);
      }
      for (std::map<unsigned, unsigned>::const_iterator it = 
            frontiers.begin(); it != frontiers.end(); ++it)
        log_tracing.info() << "  events[" << it->second << "] = events["
                           << it->first << "]";
      dump_sharded_template();

      log_tracing.info() << "[Precondition]";
      for (std::vector<TraceConditionSet*>::const_iterator it =
            preconditions.begin(); it != preconditions.end(); it++)
        (*it)->dump_conditions();

      log_tracing.info() << "[Anticondition]";
      for (std::vector<TraceConditionSet*>::const_iterator it =
            anticonditions.begin(); it != anticonditions.end(); it++)
        (*it)->dump_conditions();

      log_tracing.info() << "[Postcondition]";
      for (std::vector<TraceConditionSet*>::const_iterator it =
            postconditions.begin(); it != postconditions.end(); it++)
        (*it)->dump_conditions();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::dump_instructions(
                            const std::vector<Instruction*> &instructions) const
    //--------------------------------------------------------------------------
    {
      for (std::vector<Instruction*>::const_iterator it = instructions.begin();
           it != instructions.end(); ++it)
        log_tracing.info() << "  " << (*it)->to_string(memo_entries);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::pack_recorder(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(trace->runtime->address_space);
      rez.serialize(this);
      rez.serialize<DistributedID>(0); // no coll
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
      physical_instances = finder->second.physical_instances;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::get_allreduce_mapping(AllReduceOp *allreduce,
                      std::vector<Memory> &target_memories, size_t &future_size)
    //--------------------------------------------------------------------------
    {
      TraceLocalID op_key = allreduce->get_trace_local_id();
      AutoLock t_lock(template_lock, 1, false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(is_replaying());
#endif
      std::map<TraceLocalID,CachedAllreduce>::const_iterator finder =
        cached_allreduces.find(op_key);
#ifdef DEBUG_LEGION
      assert(finder != cached_allreduces.end());
#endif
      target_memories = finder->second.target_memories;
      future_size = finder->second.future_size;
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
      assert(!term_event.exists() || term_event.has_triggered_faultignorant());
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
    ShardID PhysicalTemplate::record_barrier_creation(ApBarrier &bar,
                                                      size_t total_arrivals)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!bar.exists());
#endif
      bar = ApBarrier(Realm::Barrier::create_barrier(total_arrivals));
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      const unsigned lhs = convert_event(bar);
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
                                             CollectiveKind collective,
                                             bool record_effect)
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
            rhs_, src_unique, dst_unique, priority, collective,record_effect)); 
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
                                             CollectiveKind collective,
                                             bool record_effect)
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
                                       priority, collective, record_effect));
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
        view->add_base_gc_ref(TRACE_REF);
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
        view->add_base_gc_ref(TRACE_REF);
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
                            ApEvent complete, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      // Do this first in case it gets preempted
      const unsigned complete_ = 
        complete.exists() ? find_event(complete, tpl_lock) : 0;
      events.push_back(ApEvent());
      insert_instruction(new CompleteReplay(*this, tlid, complete_));
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
    void PhysicalTemplate::record_future_allreduce(const TraceLocalID &tlid,
        const std::vector<Memory> &target_memories, size_t future_size)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
      assert(cached_allreduces.find(tlid) == cached_allreduces.end());
#endif
      CachedAllreduce &allreduce = cached_allreduces[tlid];
      allreduce.target_memories = target_memories;
      allreduce.future_size = future_size;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_concurrent_barrier(IndexTask *task,
        RtBarrier barrier, const std::vector<ShardID> &shards, size_t arrivals)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!shards.empty());
      assert(std::is_sorted(shards.begin(), shards.end()));
#endif
      const TraceLocalID tlid = task->get_trace_local_id();
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(concurrent_barriers.find(tlid) == concurrent_barriers.end());
#endif
      ConcurrentBarrier &concurrent = concurrent_barriers[tlid];
      concurrent.barrier = barrier;
      concurrent.shards = shards;
      concurrent.participants = arrivals;
    }

    //--------------------------------------------------------------------------
    RtBarrier PhysicalTemplate::get_concurrent_barrier(IndexTask *task)
    //--------------------------------------------------------------------------
    {
      const TraceLocalID tlid = task->get_trace_local_id();
      AutoLock tpl_lock(template_lock);
      std::map<TraceLocalID,ConcurrentBarrier>::iterator finder =
        concurrent_barriers.find(tlid);
#ifdef DEBUG_LEGION
      assert(finder != concurrent_barriers.end());
#endif
      const RtBarrier result = finder->second.barrier;
      Runtime::advance_barrier(finder->second.barrier);
      return result;
    }

    //--------------------------------------------------------------------------
    const std::vector<ShardID>& PhysicalTemplate::get_concurrent_shards(
                                                            ReplIndexTask *task)
    //--------------------------------------------------------------------------
    {
      const TraceLocalID tlid = task->get_trace_local_id();
      AutoLock tpl_lock(template_lock);
      std::map<TraceLocalID,ConcurrentBarrier>::iterator finder =
        concurrent_barriers.find(tlid);
#ifdef DEBUG_LEGION
      assert(finder != concurrent_barriers.end());
#endif
      return finder->second.shards;
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
    void PhysicalTemplate::initialize_replay(ApEvent completion, bool recurrent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.empty());
      assert(remaining_replays.load() == 0);
      assert(!replay_postcondition.exists());
#endif
      if (total_replays++ == Realm::Barrier::MAX_PHASES)
      {
        replay_precondition = refresh_managed_barriers();
        // Reset it back to one after updating our barriers
        total_replays = 1;
      }
      else
        replay_precondition = RtEvent::NO_RT_EVENT;
      remaining_replays.store(slices.size());
      total_logical.store(0);
      // Check to see if we have a finished transitive reduction result
      check_finalize_transitive_reduction();

      if (recurrent)
      {
        if (last_fence != NULL)
          events[fence_completion_id] = events[last_fence->complete];
        for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
            it != frontiers.end(); ++it)
          events[it->second] = events[it->first];
      }
      else
      {
        events[fence_completion_id] = completion;
        for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
            it != frontiers.end(); ++it)
          events[it->second] = completion;
      }

      for (std::map<unsigned, unsigned>::iterator it =
            crossing_events.begin(); it != crossing_events.end(); ++it)
      {
        ApUserEvent ev = Runtime::create_ap_user_event(NULL);
        events[it->first] = ev;
        user_events[it->first] = ev;
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::start_replay(void)
    //--------------------------------------------------------------------------
    {
      Runtime *runtime = trace->runtime;
      const std::vector<Processor> &replay_targets = 
        trace->get_replay_targets();
#ifdef DEBUG_LEGION
      assert(remaining_replays.load() == slices.size());
#endif
      for (unsigned idx = 0; idx < slices.size(); ++idx)
      {
        ReplaySliceArgs args(this, idx, trace->is_recurrent());
        if (runtime->replay_on_cpus)
          runtime->issue_application_processor_task(args, LG_LOW_PRIORITY,
            replay_targets[idx % replay_targets.size()], replay_precondition);
        else
          runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY,
            replay_precondition, replay_targets[idx % replay_targets.size()]);
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
      for (std::map<TraceLocalID,ConcurrentBarrier>::iterator it =
            concurrent_barriers.begin(); it != concurrent_barriers.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->second.shards.size() == 1);
        assert(it->second.shards.back() == 0);
#endif
        it->second.barrier.destroy_barrier();
        it->second.barrier =
          RtBarrier(Realm::Barrier::create_barrier(it->second.participants));
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::finish_replay(std::set<ApEvent> &postconditions)
    //--------------------------------------------------------------------------
    {
      if (remaining_replays.load() > 0)
      {
        RtEvent wait_on;
        {
          AutoLock tpl_lock(template_lock);
          if (remaining_replays.load() > 0)
          {
#ifdef DEBUG_LEGION
            assert(!replay_postcondition.exists());
#endif
            replay_postcondition = Runtime::create_rt_user_event();
            wait_on = replay_postcondition;
          }
        }
        if (wait_on.exists())
        {
          wait_on.wait();
          replay_postcondition = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
      for (std::map<unsigned,unsigned>::const_iterator it =
            frontiers.begin(); it != frontiers.end(); it++)
        postconditions.insert(events[it->first]);
      if (last_fence != NULL)
        postconditions.insert(events[last_fence->complete]);
      operations.clear();
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::defer_template_deletion(ApEvent &pending_deletion,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      pending_deletion = get_completion_for_deletion();
      if (!pending_deletion.exists() && 
          transitive_reduction_done.has_triggered())
      {
        check_finalize_transitive_reduction();
        return false;
      }
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
      {
        check_finalize_transitive_reduction();
        return false;
      }
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
      targs->tpl->transitive_reduction(targs->state, true/*deferred*/); 
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalTemplate::handle_delete_template(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeleteTemplateArgs *pargs = (const DeleteTemplateArgs*)args;
      pargs->tpl->check_finalize_transitive_reduction();
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
                                    ApEvent fence_event, ReplicateContext *ctx)
      : PhysicalTemplate(trace, fence_event), repl_ctx(ctx),
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
    void ShardedPhysicalTemplate::record_trigger_event(ApUserEvent lhs,
                                          ApEvent rhs, const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs.exists());
#endif
      const AddressSpaceID event_space = find_event_space(lhs);
      if ((event_space == trace->runtime->address_space) &&
          record_shard_event_trigger(lhs, rhs, tlid))
        return;
      RtEvent done = repl_ctx->shard_manager->send_trace_event_trigger(
          trace->logical_trace->tid, event_space, lhs, rhs, tlid);
      if (done.exists())
        done.wait();
    }

    //--------------------------------------------------------------------------
    bool ShardedPhysicalTemplate::record_shard_event_trigger(ApUserEvent lhs,
        ApEvent rhs, const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs.exists());
#endif
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(is_recording());
#endif
      std::map<ApEvent,unsigned>::const_iterator finder = event_map.find(lhs);
      if (finder == event_map.end())
        return false;
#ifdef DEBUG_LEGION
      assert(finder->second != NO_INDEX);
#endif
      const unsigned rhs_ =
        rhs.exists() ? find_event(rhs, tpl_lock) : fence_completion_id;
      events.push_back(ApEvent());
      insert_instruction(new TriggerEvent(*this, finder->second, rhs_, tlid));
      return true;
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
    ShardID ShardedPhysicalTemplate::record_barrier_creation(ApBarrier &bar,
                                                          size_t total_arrivals)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::record_barrier_creation(bar, total_arrivals);
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
                                 int priority, CollectiveKind collective,
                                 bool record_effect)
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
                                          priority, collective, record_effect);
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
                                 CollectiveKind collective, bool record_effect)
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
                                          unique_event, priority,
                                          collective, record_effect);
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
              size_t num_concurrent;
              derez.deserialize(num_concurrent);
              for (unsigned idx = 0; idx < num_concurrent; idx++)
              {
                TraceLocalID tlid;
                tlid.deserialize(derez);
                std::map<TraceLocalID,ConcurrentBarrier>::iterator finder =
                  concurrent_barriers.find(tlid);
#ifdef DEBUG_LEGION
                assert(finder != concurrent_barriers.end());
#endif
                derez.deserialize(finder->second.barrier);
              }
              refreshed_barriers += num_barriers + num_concurrent;
              const size_t expected = local_advances.size() +
                managed_arrivals.size() + concurrent_barriers.size();
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
              size_t num_concurrent;
              derez.deserialize(num_concurrent);
              for (unsigned idx = 0; idx < num_concurrent; idx++)
              {
                TraceLocalID tlid;
                tlid.deserialize(derez);
#ifdef DEBUG_LEGION
                assert(pending_concurrent_barriers.find(tlid) ==
                    pending_concurrent_barriers.end());
#endif
                derez.deserialize(pending_concurrent_barriers[tlid]);
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

#if 0
    //--------------------------------------------------------------------------
    PhysicalTemplate::DetailedBoolean ShardedPhysicalTemplate::check_idempotent(
        Operation *op, InnerContext *context)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(op != NULL);
      ReplTraceOp *repl_op = dynamic_cast<ReplTraceOp*>(op);
      assert(repl_op != NULL);
#else
      ReplTraceOp *repl_op = static_cast<ReplTraceOp*>(op);
#endif
      // We need everyone else to be done capturing their traces
      // before we can do our own idempotence check
      repl_op->sync_for_idempotent_check();
      // Do the base call first to determine if our local shard is replayable
      const DetailedBoolean result =
          PhysicalTemplate::check_idempotent(op, context);
      if (result)
      {
        // Now we can do the exchange
        if (repl_op->exchange_idempotent(repl_ctx, true/*replayable*/))
          return result;
        else
          return DetailedBoolean(false, "Remote shard not replyable");
      }
      else
      {
        // Still need to do the exchange
        repl_op->exchange_idempotent(repl_ctx, false/*replayable*/);
        return result;
      }
    }
#endif

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::pack_recorder(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(trace->runtime->address_space);
      rez.serialize(this);
      rez.serialize(repl_ctx->shard_manager->did);
      rez.serialize(trace->logical_trace->tid);
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::initialize_replay(ApEvent completion,
                                                    bool recurrent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(pending_collectives.empty());
#endif
      PhysicalTemplate::initialize_replay(completion, recurrent);
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
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::start_replay(void)
    //--------------------------------------------------------------------------
    {
      if (!pending_collectives.empty())
      {
        for (std::map<std::pair<size_t,size_t>,ApBarrier>::const_iterator it =
             pending_collectives.begin(); it != pending_collectives.end(); it++)
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
        pending_collectives.clear();
      }
      // Now call the base version of this
      PhysicalTemplate::start_replay();
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
      // Also see if we have any concurrent barriers to update
      size_t local_refreshed = 0;
      std::map<ShardID,std::map<TraceLocalID,RtBarrier> > concurrent_updates;
      for (std::map<TraceLocalID,ConcurrentBarrier>::iterator it =
            concurrent_barriers.begin(); it != concurrent_barriers.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(!it->second.shards.empty());
        assert(std::binary_search(it->second.shards.begin(),
              it->second.shards.end(), local_shard));
#endif
        if (local_shard == it->second.shards.front())
        {
          it->second.barrier.destroy_barrier();
          it->second.barrier = 
            RtBarrier(Realm::Barrier::create_barrier(it->second.participants));
          for (unsigned idx = 1; idx < it->second.shards.size(); idx++)
          {
            ShardID shard = it->second.shards[idx];
            notifications[shard]; // instantiate so it is there
            concurrent_updates[shard][it->first] = it->second.barrier;
          }
          local_refreshed++;
        }
      }
      // Send out the notifications to all the shards
      ShardManager *manager = repl_ctx->shard_manager;
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
          std::map<ShardID,std::map<TraceLocalID,RtBarrier> >::const_iterator
            finder = concurrent_updates.find(nit->first);
          if (finder != concurrent_updates.end())
          {
            rez.serialize<size_t>(finder->second.size());
            for (std::map<TraceLocalID,RtBarrier>::const_iterator it =
                  finder->second.begin(); it != finder->second.end(); it++)
            {
              it->first.serialize(rez);
              rez.serialize(it->second);
            }
          }
          else
            rez.serialize<size_t>(0);
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
        if (!pending_concurrent_barriers.empty())
        {
          for (std::map<TraceLocalID,RtBarrier>::const_iterator it =
                pending_concurrent_barriers.begin(); it !=
                pending_concurrent_barriers.end(); it++)
          {
            std::map<TraceLocalID,ConcurrentBarrier>::iterator finder =
              concurrent_barriers.find(it->first);
#ifdef DEBUG_LEGION
            assert(finder != concurrent_barriers.end()); 
#endif
            finder->second.barrier = it->second;
          }
          refreshed_barriers += pending_concurrent_barriers.size();
          pending_concurrent_barriers.clear();
        }
        const size_t expected = local_advances.size() +
          managed_arrivals.size() + concurrent_barriers.size();
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
      PhysicalTemplate::finish_replay(postconditions);
      // Also need to do any local frontiers that we have here as well
      for (std::map<unsigned,ApBarrier>::const_iterator it = 
            local_frontiers.begin(); it != local_frontiers.end(); it++)
        postconditions.insert(events[it->first]);
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
    void ShardedPhysicalTemplate::dump_sharded_template(void) const
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
      // Save the barrier until it's safe to update the instruction
      pending_collectives[key] = newbar;
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
    void ShardedPhysicalTemplate::sync_compute_frontiers(CompleteOp *op,
                                    const std::vector<RtEvent> &frontier_events)
    //--------------------------------------------------------------------------
    {
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
      : Instruction(t, o), lhs(l)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void AssignFenceCompletion::execute(std::vector<ApEvent> &events,
                               std::map<unsigned,ApUserEvent> &user_events,
                               std::map<TraceLocalID,MemoizableOp*> &operations,
                               const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      // This is a no-op since it gets assigned during initialize replay
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
                         int pr, CollectiveKind collect, bool effect)
      : Instruction(tpl, key), lhs(l), expr(e), src_fields(s), dst_fields(d), 
        reservations(r),
#ifdef LEGION_SPY
        src_tree_id(src_tid), dst_tree_id(dst_tid),
#endif
        precondition_idx(pi), src_unique(src_uni),
        dst_unique(dst_uni), priority(pr), collective(collect),
        record_effect(effect)
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
      std::map<TraceLocalID,MemoizableOp*>::const_iterator finder =
        operations.find(owner);
      if (finder == operations.end())
      {
        // Remote copy, should still be able to find the owner op here
        TraceLocalID local = owner; 
        local.index_point = DomainPoint();
        finder = operations.find(local);
      }
#ifdef DEBUG_LEGION
      assert(finder != operations.end());
      assert(finder->second != NULL);
#endif
      ApEvent precondition = events[precondition_idx];
      const PhysicalTraceInfo trace_info(finder->second, -1U);
      events[lhs] = expr->issue_copy(finder->second, trace_info, dst_fields,
                                     src_fields, reservations,
#ifdef LEGION_SPY
                                     src_tree_id, dst_tree_id,
#endif
                                     precondition, PredEvent::NO_PRED_EVENT,
                                     src_unique, dst_unique,
                                     collective, record_effect,
                                     priority, true/*replay*/);
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
      std::map<TraceLocalID,MemoizableOp*>::const_iterator finder =
        operations.find(owner);
      if (finder == operations.end())
      {
        // Remote copy, should still be able to find the owner op here
        TraceLocalID local = owner; 
        local.index_point = DomainPoint();
        finder = operations.find(local);
      }
#ifdef DEBUG_LEGION
      assert(finder != operations.end());
      assert(finder->second != NULL);
#endif
      ApEvent copy_pre = events[copy_precondition];
      ApEvent src_indirect_pre = events[src_indirect_precondition];
      ApEvent dst_indirect_pre = events[dst_indirect_precondition];
      const PhysicalTraceInfo trace_info(finder->second, -1U);
      events[lhs] = executor->execute(finder->second, PredEvent::NO_PRED_EVENT,
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
                         CollectiveKind collect, bool effect)
      : Instruction(tpl, key), lhs(l), expr(e), fields(f), fill_size(size),
#ifdef LEGION_SPY
        fill_uid(uid), handle(h), tree_id(tid),
#endif
        precondition_idx(pi), unique_event(unique), priority(pr),
        collective(collect), record_effect(effect)
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
      std::map<TraceLocalID,MemoizableOp*>::const_iterator finder =
        operations.find(owner);
      if (finder == operations.end())
      {
        // Remote copy, should still be able to find the owner op here
        TraceLocalID local = owner; 
        local.index_point = DomainPoint();
        finder = operations.find(local);
      }
#ifdef DEBUG_LEGION
      assert(finder != operations.end());
      assert(finder->second != NULL);
#endif
      ApEvent precondition = events[precondition_idx];
      const PhysicalTraceInfo trace_info(finder->second, -1U);
      events[lhs] = expr->issue_fill(finder->second, trace_info, fields,
                                     fill_value, fill_size,
#ifdef LEGION_SPY
                                     fill_uid, handle, tree_id,
#endif
                                     precondition, PredEvent::NO_PRED_EVENT,
                                     unique_event, collective, record_effect,
                                     priority, true/*replay*/);
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
                                   unsigned c)
      : Instruction(tpl, l), complete(c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(complete < tpl.events.size());
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
      memoizable->complete_replay(events[complete]);
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
         << "].complete_replay(events[" << complete 
         << "])    (op kind: "
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

