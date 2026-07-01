/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#include "legion/tracing/logical.h"
#include "legion/contexts/inner.h"
#include "legion/nodes/region.h"
#include "legion/operations/close.h"
#include "legion/operations/fence.h"
#include "legion/operations/pointwise.h"
#include "legion/tasks/index.h"
#include "legion/tracing/physical.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // LogicalTrace
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalTrace::LogicalTrace(
        InnerContext* c, TraceID t, bool logical_only, bool static_trace,
        Provenance* p, const std::set<RegionTreeID>* trees)
      : context(c), tid(t), begin_provenance(p), end_provenance(nullptr),
        physical_trace(logical_only ? nullptr : new PhysicalTrace(this)),
        verification_index(0), blocking_call_observed(false), fixed(false),
        intermediate_fence(false), recording(true), trace_fence(nullptr),
        static_translator(static_trace ? new StaticTranslator(trees) : nullptr)
    //--------------------------------------------------------------------------
    {
      if (begin_provenance != nullptr)
        begin_provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    LogicalTrace::~LogicalTrace(void)
    //--------------------------------------------------------------------------
    {
      if (physical_trace != nullptr)
        delete physical_trace;
      if ((begin_provenance != nullptr) && begin_provenance->remove_reference())
        delete begin_provenance;
      if ((end_provenance != nullptr) && end_provenance->remove_reference())
        delete end_provenance;
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::fix_trace(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      legion_assert(!fixed);
      legion_assert(end_provenance == nullptr);
      fixed = true;
      end_provenance = provenance;
      verification_index = 0;  // reset this back to zero
      if (end_provenance != nullptr)
        end_provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::record_operation_hash(
        Operation* op, Murmur3Hasher& hasher, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      const OpKind kind = op->get_operation_kind();
      TaskID task_id = 0;
      if (kind == TASK_OP_KIND)
      {
        TaskOp* task = legion_safe_cast<TaskOp*>(op);
        task_id = task->task_id;
      }
      const unsigned num_regions = op->get_region_count();
      uint64_t hash[2];
      hasher.finalize(hash);
      if (fixed)
      {
        if (verification_infos.size() <= opidx)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error
              << "Detected " << opidx << " operations in trace " << tid
              << " of parent task " << *context << " which differs from the "
              << verification_infos.size()
              << " operations that were recorded in the first execution "
              << "of the trace. The number of operations in the trace "
              << "must always be the same across all executions of the trace.";
          error.raise();
        }
        const VerificationInfo& info = verification_infos[opidx];
        if (info.kind != kind)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Operation " << *op << " does not match the recorded "
                << "operation kind " << Operation::get_string_rep(info.kind)
                << " for the " << opidx << " operation in trace " << tid
                << " of parent task " << *context << ". The same order of "
                << "operations must be issued every time a trace is executed.";
          error.raise();
        }
        if (info.task_id != task_id)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Task " << task_id << " of " << *op
                << " does not match the recorded task " << info.task_id
                << " for the " << opidx << " task in trace " << tid
                << " of parent task " << *context << ". The same order of "
                << "operations must be issued every time a trace is executed.";
          error.raise();
        }
        if (info.regions != num_regions)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error
              << ((task_id > 0) ? "Task " : "Operation ") << *op << " recorded "
              << info.regions << " region requirements for trace " << tid
              << " in parent task " << *context << " but was re-executed "
              << "with " << num_regions
              << " region requirements. The number of "
              << "region requirements must always match the number re-executed "
              << "for each corresponding operation in the trace.";
          error.raise();
        }
        if ((info.hash[0] != hash[0]) || (info.hash[1] != hash[1]))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << ((task_id > 0) ? "Task " : "Operation ") << *op
                << " was replayed with different region requirements for trace "
                << tid << " in parent task " << *context << " than what it had "
                << "when the trace was recorded. Region requirement arguments "
                << "must match exactly every time a trace is executed.";
          error.raise();
        }
      }
      else
      {
        legion_assert(opidx == verification_infos.size());
        verification_infos.emplace_back(
            VerificationInfo(kind, task_id, num_regions, hash));
      }
      return false;  // return value doesn't matter
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::record_operation_noop(Operation* op, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      const OpKind kind = op->get_operation_kind();
      // Cheat a bit and compute a hash based on any region requirements
      // and then call the method for normal operations
      Murmur3Hasher hasher;
      hasher.hash(kind);
      if (kind == TASK_OP_KIND)
      {
        // This can happen with traces with output regions
        TaskOp* task = legion_safe_cast<TaskOp*>(op);
        hasher.hash(task->task_id);
      }
      const unsigned num_regions = op->get_region_count();
      for (unsigned idx = 0; idx < num_regions; idx++)
        Operation::hash_requirement(hasher, op->get_requirement(idx));
      return record_operation_hash(op, hasher, opidx);
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::record_operation_untraceable(
        Operation* op, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      // Make sure we're not a physical trace
      if (has_physical_trace())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Illegal operation in physical trace. The application launched"
            << " " << *op << " inside of physical trace " << tid
            << " of parent "
            << "task " << *context << " but this kind of operation is not "
            << "supported for physical traces currently. You can request "
               "support "
            << "but we cannot guarantee support for all kinds of operations in "
            << "physical traces.";
        error.raise();
      }
      // Can now call the no-op version of this meethod to compute the hash
      // and record/verify it since logical traces can actually handle
      // untraceable operations which are more for physical analysis
      return record_operation_noop(op, opidx);
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::initialize_operation(
        Operation* op, const std::vector<StaticDependence>* dependences)
    //--------------------------------------------------------------------------
    {
      legion_assert(!op->is_internal_op());
      if (static_translator != nullptr)
        static_translator->push_dependences(dependences);
      op->set_trace(this, !fixed, verification_index++);
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::check_operation_count(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(verification_index <= verification_infos.size());
      if (verification_index < verification_infos.size())
      {
        Error e(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        e << "Detected " << verification_index << " operations in trace " << tid
          << " of parent task " << context->get_task_name() << " (UID "
          << context->get_unique_id() << ") which differs from the "
          << verification_infos.size()
          << " operations that were recorded in the "
          << "first execution of the trace. The number of operations in the "
             "trace "
          << "must always be the same across all executions of the trace.";
        e.raise();
      }
      verification_index = 0;
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::skip_analysis(RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      legion_assert(recording);
      if (static_translator == nullptr)
        return false;
      return static_translator->skip_analysis(tid);
    }

    //--------------------------------------------------------------------------
    size_t LogicalTrace::register_operation(Operation* op, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      const std::pair<Operation*, GenerationID> key(op, gen);
      if (recording)
      {
        // Recording
        if (op->is_internal_op())
          // We don't need to register internal operations
          return std::numeric_limits<size_t>::max();
        const size_t index = replay_info.size();
        const size_t op_index = operations.size();
        op_map[key] = std::make_pair(op_index, index);
        operations.emplace_back(OpInfo(op));
        replay_info.emplace_back(OperationInfo());
        if (static_translator != nullptr)
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
        legion_assert(!op->is_internal_op());
        // Replaying
        const size_t index = replay_index++;
        if (index >= replay_info.size())
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Trace violation! Recorded " << replay_info.size()
                << " operations in trace " << tid << " in " << *context
                << " but " << (index + 1)
                << " operations have now been issued.";
          error.raise();
        }
        // Check to see if the meta-data alignes
        OperationInfo& info = replay_info[index];
        // Add a mapping reference since ops will be registering dependences
        op->add_mapping_reference(gen);
        operations.emplace_back(OpInfo(op));
        frontiers.emplace(std::make_pair(key, op->get_unique_op_id()));
        // First make any close operations needed for this operation and
        // register their dependences
        for (ctx::vector<CloseInfo>::const_iterator cit = info.closes.begin();
             cit != info.closes.end(); cit++)
        {
#ifdef LEGION_DEBUG_COLLECTIVES
          MergeCloseOp* close_op = context->get_merge_close_op(op, cit->node);
#else
          MergeCloseOp* close_op = context->get_merge_close_op();
#endif
          close_op->initialize(context, cit->requirement, cit->creator_idx, op);
          close_op->update_close_mask(cit->close_mask);
          const GenerationID close_gen = close_op->get_generation();
          const std::pair<Operation*, GenerationID> close_key(
              close_op, close_gen);
          close_op->add_mapping_reference(close_gen);
          operations.emplace_back(OpInfo(close_op));
          close_op->begin_dependence_analysis();
          close_op->trigger_dependence_analysis();
          replay_operation_dependences(close_op, cit->dependences);
          close_op->end_dependence_analysis();
        }
        if (!info.pointwise_dependences.empty())
          replay_pointwise_dependences(op, info.pointwise_dependences);
        // Then register the dependences for this operation
        if (!info.dependences.empty())
          replay_operation_dependences(op, info.dependences);
        else  // need to at least record a dependence on the fence event
          op->register_dependence(trace_fence, trace_fence_gen);
        return index;
      }
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::register_internal(InternalOp* op)
    //--------------------------------------------------------------------------
    {
      legion_assert(recording);
      const std::pair<Operation*, GenerationID> key(op, op->get_generation());
      // Note that we don't record the index of the operation here since
      // it has no index, instead we record the index of the replay_info
      // that is associated with this internal operation
      op_map[key] = std::make_pair(-1, replay_info.size() - 1);
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::register_close(
        MergeCloseOp* op, unsigned creator_idx,
#ifdef LEGION_DEBUG_COLLECTIVES
        RegionTreeNode* node,
#endif
        const RegionRequirement& req)
    //--------------------------------------------------------------------------
    {
      legion_assert(recording);
      legion_assert(!replay_info.empty());
      std::pair<Operation*, GenerationID> key(op, op->get_generation());
      const size_t index = operations.size();
      operations.emplace_back(OpInfo(op));
      op_map[key] = std::make_pair(index, replay_info.size());
      OperationInfo& info = replay_info.back();
      info.closes.emplace_back(CloseInfo(
          op, creator_idx,
#ifdef LEGION_DEBUG_COLLECTIVES
          node,
#endif
          req));
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::replay_operation_dependences(
        Operation* op, const ctx::vector<DependenceRecord>& dependences)
    //--------------------------------------------------------------------------
    {
      for (ctx::vector<DependenceRecord>::const_iterator it =
               dependences.begin();
           it != dependences.end(); it++)
      {
        legion_assert(it->operation_idx >= 0);
        legion_assert(((size_t)it->operation_idx) < operations.size());
        legion_assert(it->dtype != LEGION_NO_DEPENDENCE);
        std::pair<Operation*, GenerationID> target = std::make_pair(
            operations[it->operation_idx].op,
            operations[it->operation_idx].gen);
        std::map<std::pair<Operation*, GenerationID>, UniqueID>::iterator
            finder = frontiers.find(target);
        if (finder != frontiers.end())
        {
          finder->first.first->remove_mapping_reference(finder->first.second);
          frontiers.erase(finder);
        }
        if ((it->prev_idx == -1) || (it->next_idx == -1))
        {
          op->register_dependence(target.first, target.second);
          LegionSpy::log_mapping_dependence(
              op->get_context()->get_unique_id(),
              operations[it->operation_idx].unique_id,
              (it->prev_idx == -1) ? 0 : it->prev_idx, op->get_unique_op_id(),
              (it->next_idx == -1) ? 0 : it->next_idx, LEGION_TRUE_DEPENDENCE);
        }
        else
        {
          op->register_region_dependence(
              it->next_idx, target.first, target.second, it->prev_idx,
              it->dtype, it->dependent_mask);
          LegionSpy::log_mapping_dependence(
              op->get_context()->get_unique_id(),
              operations[it->operation_idx].unique_id, it->prev_idx,
              op->get_unique_op_id(), it->next_idx, it->dtype);
        }
      }
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::replay_pointwise_dependences(
        Operation* op,
        const std::map<unsigned, std::vector<PointwiseDependence>>& dependences)
    //--------------------------------------------------------------------------
    {
      // Make a copy of the dependences and then update the context index
      // with the right entry from the list of operations
      std::map<unsigned, std::vector<PointwiseDependence>> copy = dependences;
      for (std::pair<const unsigned, std::vector<PointwiseDependence>>& cit :
           copy)
      {
        for (PointwiseDependence& it : cit.second)
        {
          const OpInfo& info = operations[it.context_index];
          it.context_index = info.context_index;
          it.unique_id = info.unique_id;
          LegionSpy::log_mapping_dependence(
              op->get_context()->get_unique_id(), info.unique_id,
              it.region_index, op->get_unique_op_id(), cit.first,
              LEGION_TRUE_DEPENDENCE, true /*pointwise*/);
        }
      }
      op->replay_pointwise_dependences(copy);
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::record_dependence(
        Operation* target, GenerationID target_gen, Operation* source,
        GenerationID source_gen)
    //--------------------------------------------------------------------------
    {
      legion_assert(recording);
      legion_assert(!target->is_internal_op());
      legion_assert(!source->is_internal_op());
      const std::pair<Operation*, GenerationID> target_key(target, target_gen);
      std::map<
          std::pair<Operation*, GenerationID>,
          std::pair<unsigned, unsigned>>::const_iterator target_finder =
          op_map.find(target_key);
      // The target is not part of the trace so there's no need to record it
      if (target_finder == op_map.end())
        return false;
      legion_assert(!replay_info.empty());
      OperationInfo& info = replay_info.back();
      DependenceRecord record(target_finder->second.first);
      for (ctx::vector<DependenceRecord>::iterator it =
               info.dependences.begin();
           it != info.dependences.end(); it++)
        if (it->merge(record))
          return true;
      info.dependences.emplace_back(std::move(record));
      return true;
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::record_region_dependence(
        Operation* target, GenerationID target_gen, Operation* source,
        GenerationID source_gen, unsigned target_idx, unsigned source_idx,
        DependenceType dtype, const FieldMask& dep_mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(recording);
      const std::pair<Operation*, GenerationID> target_key(target, target_gen);
      std::map<
          std::pair<Operation*, GenerationID>,
          std::pair<unsigned, unsigned>>::const_iterator target_finder =
          op_map.find(target_key);
      // The target is not part of the trace so there's no need to record it
      if (target_finder == op_map.end())
      {
        // If this is a close operation then we still need to update the mask
        if (source->get_operation_kind() == MERGE_CLOSE_OP_KIND)
        {
          legion_assert(!replay_info.empty());
          legion_assert(
              op_map.find(std::make_pair(source, source_gen)) != op_map.end());
          OperationInfo& info = replay_info.back();
          [[maybe_unused]] bool found = false;
          legion_assert(!info.closes.empty());
          // Find the right close info and record the dependence
          for (unsigned idx = 0; idx < info.closes.size(); idx++)
          {
            CloseInfo& close = info.closes[idx];
            if (close.close_op != source)
              continue;
            found = true;
            close.close_mask |= dep_mask;
            break;
          }
          legion_assert(found);
        }
        return false;
      }
      legion_assert(!replay_info.empty());
      OperationInfo& info = replay_info.back();
      if (source->get_operation_kind() == MERGE_CLOSE_OP_KIND)
      {
        [[maybe_unused]] bool found = false;
        legion_assert(!info.closes.empty());
        // Find the right close info and record the dependence
        for (unsigned idx = 0; idx < info.closes.size(); idx++)
        {
          CloseInfo& close = info.closes[idx];
          if (close.close_op != source)
            continue;
          found = true;
          close.close_mask |= dep_mask;
          if (target->is_internal_op() &&
              (target->get_operation_kind() != MERGE_CLOSE_OP_KIND))
          {
            legion_assert(target_finder->second.second < replay_info.size());
            const OperationInfo& target_info =
                replay_info[target_finder->second.second];
            std::map<unsigned, ctx::vector<DependenceRecord>>::const_iterator
                finder = target_info.internal_dependences.find(target_idx);
            if (finder != target_info.internal_dependences.end())
            {
              for (ctx::vector<DependenceRecord>::const_iterator rit =
                       finder->second.begin();
                   rit != finder->second.end(); rit++)
              {
                const FieldMask overlap = dep_mask & rit->dependent_mask;
                if (!overlap)
                  continue;
                DependenceRecord record(
                    rit->operation_idx, rit->prev_idx, source_idx, dtype,
                    overlap);
                bool found2 = false;
                for (ctx::vector<DependenceRecord>::iterator it =
                         close.dependences.begin();
                     it != close.dependences.end(); it++)
                {
                  if (!it->merge(record))
                    continue;
                  found2 = true;
                  break;
                }
                if (!found2)
                  close.dependences.emplace_back(std::move(record));
              }
            }
          }
          else
          {
            DependenceRecord record(
                target_finder->second.first, target_idx, source_idx, dtype,
                dep_mask);
            for (ctx::vector<DependenceRecord>::iterator it =
                     close.dependences.begin();
                 it != close.dependences.end(); it++)
              if (it->merge(record))
                return true;
            close.dependences.emplace_back(std::move(record));
          }
          break;
        }
        legion_assert(found);
      }
      else if (source->is_internal_op())
      {
        // Record the dependence on the correct region requirement
        // of the internal operations
        ctx::vector<DependenceRecord>& internal_dependences =
            info.internal_dependences[source_idx];
        if (target->is_internal_op() &&
            (target->get_operation_kind() != MERGE_CLOSE_OP_KIND))
        {
          legion_assert(target_finder->second.second < replay_info.size());
          const OperationInfo& target_info =
              replay_info[target_finder->second.second];
          std::map<unsigned, ctx::vector<DependenceRecord>>::const_iterator
              finder = target_info.internal_dependences.find(target_idx);
          if (finder != target_info.internal_dependences.end())
          {
            for (ctx::vector<DependenceRecord>::const_iterator rit =
                     finder->second.begin();
                 rit != finder->second.end(); rit++)
            {
              const FieldMask overlap = dep_mask & rit->dependent_mask;
              if (!overlap)
                continue;
              DependenceRecord record(
                  rit->operation_idx, rit->prev_idx, source_idx, dtype,
                  overlap);
              bool found = false;
              for (ctx::vector<DependenceRecord>::iterator it =
                       internal_dependences.begin();
                   it != internal_dependences.end(); it++)
              {
                if (!it->merge(record))
                  continue;
                found = true;
                break;
              }
              if (!found)
                internal_dependences.emplace_back(std::move(record));
            }
          }
        }
        else
        {
          DependenceRecord record(
              target_finder->second.first, target_idx, source_idx, dtype,
              dep_mask);
          for (ctx::vector<DependenceRecord>::iterator it =
                   internal_dependences.begin();
               it != internal_dependences.end(); it++)
            if (it->merge(record))
              return true;
          internal_dependences.emplace_back(std::move(record));
        }
      }
      else if (
          target->is_internal_op() &&
          (target->get_operation_kind() != MERGE_CLOSE_OP_KIND))
      {
        legion_assert(target_finder->second.second < replay_info.size());
        // Figure out which region requirement it was for and look for
        // overlapping dependences on that region requirement and record
        // any of those instead
        const OperationInfo& target_info =
            replay_info[target_finder->second.second];
        std::map<unsigned, ctx::vector<DependenceRecord>>::const_iterator
            finder = target_info.internal_dependences.find(target_idx);
        if (finder != target_info.internal_dependences.end())
        {
          for (ctx::vector<DependenceRecord>::const_iterator rit =
                   finder->second.begin();
               rit != finder->second.end(); rit++)
          {
            const FieldMask overlap = dep_mask & rit->dependent_mask;
            if (!overlap)
              continue;
            DependenceRecord record(
                rit->operation_idx, rit->prev_idx, source_idx, dtype, overlap);
            bool found = false;
            for (ctx::vector<DependenceRecord>::iterator it =
                     info.dependences.begin();
                 it != info.dependences.end(); it++)
            {
              if (!it->merge(record))
                continue;
              found = true;
              break;
            }
            if (!found)
              info.dependences.emplace_back(std::move(record));
          }
        }
      }
      else
      {
        DependenceRecord record(
            target_finder->second.first, target_idx, source_idx, dtype,
            dep_mask);
        for (ctx::vector<DependenceRecord>::iterator it =
                 info.dependences.begin();
             it != info.dependences.end(); it++)
          if (it->merge(record))
            return true;
        info.dependences.emplace_back(std::move(record));
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::record_pointwise_dependence(
        Operation* target, GenerationID target_gen, Operation* source,
        GenerationID source_gen, unsigned idx,
        const PointwiseDependence& dependence)
    //--------------------------------------------------------------------------
    {
      legion_assert(!target->is_internal_op());
      legion_assert(!source->is_internal_op());
      const std::pair<Operation*, GenerationID> target_key(target, target_gen);
      std::map<
          std::pair<Operation*, GenerationID>,
          std::pair<unsigned, unsigned>>::const_iterator target_finder =
          op_map.find(target_key);
      // If the target is not part of the trace then there's nothing to do
      if (target_finder == op_map.end())
        return;
      legion_assert(!replay_info.empty());
      OperationInfo& info = replay_info.back();
      // Append the pointwise record to the
      PointwiseDependence& last =
          info.pointwise_dependences[idx].emplace_back(dependence);
      // Update the context index with the relative context index
      last.context_index = target_finder->second.first;
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::begin_logical_trace(FenceOp* fence_op)
    //--------------------------------------------------------------------------
    {
      legion_assert(trace_fence == nullptr);
      if (!recording)
      {
        trace_fence = fence_op;
        trace_fence_gen = fence_op->get_generation();
        fence_op->add_mapping_reference(trace_fence_gen);
        replay_index = 0;
      }
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::end_logical_trace(FenceOp* op)
    //--------------------------------------------------------------------------
    {
      if (!recording)
      {
        legion_assert(trace_fence != nullptr);
        if (replay_index != replay_info.size())
        {
          Error e(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          e << "Trace violation! Recorded " << replay_info.size()
            << " operations in trace " << tid << " in task "
            << context->get_task_name() << " (UID " << context->get_unique_id()
            << ") but only " << replay_index
            << " operations have been issued at the end of the trace.";
          e.raise();
        }
        op->register_dependence(trace_fence, trace_fence_gen);
        trace_fence->remove_mapping_reference(trace_fence_gen);
        trace_fence = nullptr;
        // Register for this fence on every one of the operations in
        // the trace and then clear out the operations data structure
        for (std::pair<const std::pair<Operation*, GenerationID>, UniqueID>&
                 it : frontiers)
        {
          const std::pair<Operation*, GenerationID>& target = it.first;
          legion_assert(!target.first->is_internal_op());
          op->register_dependence(target.first, target.second);
          LegionSpy::log_mapping_dependence(
              op->get_context()->get_unique_id(), it.second, 0 /*idx*/,
              op->get_unique_op_id(), 0, LEGION_TRUE_DEPENDENCE);
          // Remove any mapping references that we hold
          target.first->remove_mapping_reference(target.second);
        }
      }
      else  // Finished the recording so we are done
      {
        recording = false;
        op_map.clear();
        for (unsigned idx = 0; idx < replay_info.size(); idx++)
          replay_info[idx].internal_dependences.clear();
        if (static_translator != nullptr)
        {
          legion_assert(static_translator->dependences.empty());
          delete static_translator;
          static_translator = nullptr;
          // Also remove the mapping references from all the operations
          for (const OpInfo& info : operations)
            info.op->remove_mapping_reference(info.gen);
          // Remove mapping fences on the frontiers which haven't been removed
          for (const std::pair<
                   const std::pair<Operation*, GenerationID>, UniqueID>& it :
               frontiers)
            it.first.first->remove_mapping_reference(it.second);
        }
      }
      operations.clear();
      frontiers.clear();
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::find_concurrent_colors(
        ReplIndexTask* task,
        std::map<Color, CollectiveID>& concurrent_exchange_colors)
    //--------------------------------------------------------------------------
    {
      std::map<TraceLocalID, std::vector<Color>>::const_iterator finder =
          concurrent_colors.find(task->get_trace_local_id());
      if (finder == concurrent_colors.end())
        return false;
      for (const Color& color : finder->second)
        concurrent_exchange_colors.emplace(
            std::make_pair(color, CollectiveID(0)));
      return true;
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::record_concurrent_colors(
        ReplIndexTask* task,
        const std::map<Color, CollectiveID>& concurrent_exchange_colors)
    //--------------------------------------------------------------------------
    {
      const TraceLocalID tlid = task->get_trace_local_id();
      std::vector<Color>& colors = concurrent_colors[tlid];
      legion_assert(colors.empty());
      colors.reserve(concurrent_exchange_colors.size());
      for (const std::pair<const Color, CollectiveID>& it :
           concurrent_exchange_colors)
        colors.emplace_back(it.first);
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::translate_dependence_records(
        Operation* op, const unsigned index,
        const std::vector<StaticDependence>& dependences)
    //--------------------------------------------------------------------------
    {
      const bool is_replicated = (context->get_replication_id() > 0);
      for (const StaticDependence& it : dependences)
      {
        if (it.dependence_type == LEGION_NO_DEPENDENCE)
          continue;
        legion_assert(it.previous_offset <= index);
        // const std::pair<Operation*,GenerationID> &prev =
        std::pair<Operation*, GenerationID> prev = std::make_pair(
            operations[index - it.previous_offset].op,
            operations[index - it.previous_offset].gen);
        unsigned parent_index = op->find_parent_index(it.current_req_index);
        LogicalRegion root_region = context->find_logical_region(parent_index);
        FieldSpaceNode* fs = runtime->get_node(root_region.get_field_space());
        const FieldMask mask = fs->get_field_mask(it.dependent_fields);
        if (is_replicated && !it.shard_only)
        {
          // Need a merge close op to mediate the dependence
          RegionRequirement req(
              root_region, LEGION_READ_WRITE, LEGION_EXCLUSIVE, root_region);
          req.privilege_fields = it.dependent_fields;
#ifdef LEGION_DEBUG_COLLECTIVES
          MergeCloseOp* close_op =
              context->get_merge_close_op(op, runtime->get_node(root_region));
#else
          MergeCloseOp* close_op = context->get_merge_close_op();
#endif
          close_op->initialize(context, req, it.current_req_index, op);
          close_op->update_close_mask(mask);
          register_close(
              close_op, it.current_req_index,
#ifdef LEGION_DEBUG_COLLECTIVES
              runtime->get_node(root_region),
#endif
              req);
          // Mark that we are starting our dependence analysis
          close_op->begin_dependence_analysis();
          // Do any other work for the dependence analysis
          close_op->trigger_dependence_analysis();
          // Record the dependence of the close on the previous op
          close_op->register_region_dependence(
              0 /*close index*/, prev.first, prev.second, it.previous_req_index,
              LEGION_TRUE_DEPENDENCE, mask);
          // Then record our dependence on the close operation
          op->register_region_dependence(
              it.current_req_index, close_op, close_op->get_generation(),
              0 /*close index*/, LEGION_TRUE_DEPENDENCE, mask);
          // Dispatch this close op
          close_op->end_dependence_analysis();
        }
        else
        {
          // Can just record a normal dependence
          op->register_region_dependence(
              it.current_req_index, prev.first, prev.second,
              it.previous_req_index, it.dependence_type, mask);
        }
      }
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::invalidate_equivalence_sets(void) const
    //--------------------------------------------------------------------------
    {
      if (physical_trace != nullptr)
        physical_trace->invalidate_equivalence_sets();
    }

  }  // namespace Internal
}  // namespace Legion
