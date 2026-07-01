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

#include "legion/tracing/physical.h"
#include "legion/contexts/inner.h"
#include "legion/operations/begin.h"
#include "legion/operations/complete.h"
#include "legion/operations/recurrent.h"
#include "legion/tracing/template.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // PhysicalTrace
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTrace::PhysicalTrace(LogicalTrace* lt)
      : logical_trace(lt),
        perform_fence_elision(
            !(runtime->no_trace_optimization || runtime->no_fence_elision)),
        current_template(nullptr), nonreplayable_count(0),
        new_template_count(0), recording(false), recurrent(false)
    //--------------------------------------------------------------------------
    {
      if (runtime->replay_on_cpus)
      {
        Machine::ProcessorQuery local_procs(runtime->machine);
        local_procs.local_address_space();
        for (Machine::ProcessorQuery::iterator it = local_procs.begin();
             it != local_procs.end(); it++)
          if (it->kind() == Processor::LOC_PROC)
            replay_targets.emplace_back(*it);
      }
      else
        replay_targets.emplace_back(runtime->utility_group);
    }

    //--------------------------------------------------------------------------
    PhysicalTrace::~PhysicalTrace()
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> deleted_events;
      ApEvent pending_deletion = ApEvent::NO_AP_EVENT;
      for (PhysicalTemplate* tmpl : templates)
        if (!tmpl->defer_template_deletion(pending_deletion, deleted_events))
          delete tmpl;
      templates.clear();
      if (!deleted_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(deleted_events);
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::record_parent_req_fields(
        unsigned index, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      ctx::map<unsigned, FieldMask>::iterator finder =
          parent_req_fields.find(index);
      if (finder == parent_req_fields.end())
        parent_req_fields[index] = mask;
      else
        finder->second |= mask;
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::invalidate_equivalence_sets(void) const
    //--------------------------------------------------------------------------
    {
      for (PhysicalTemplate* tmpl : templates)
        tmpl->invalidate_equivalence_sets();
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::find_condition_sets(
        std::map<EquivalenceSet*, unsigned>& current_sets) const
    //--------------------------------------------------------------------------
    {
      InnerContext* context = logical_trace->context;
      for (ctx::map<unsigned, FieldMask>::const_iterator it =
               parent_req_fields.begin();
           it != parent_req_fields.end(); it++)
        context->find_trace_local_sets(it->first, it->second, current_sets);
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::refresh_condition_sets(
        FenceOp* op, std::set<RtEvent>& ready_events) const
    //--------------------------------------------------------------------------
    {
      // Make sure all the templates have up-to-date equivalence sets for
      // performing any kind of tests on preconditions/postconditions
      for (PhysicalTemplate* tmpl : templates)
        if (tmpl != current_template)
          tmpl->refresh_condition_sets(op, ready_events);
    }

    //--------------------------------------------------------------------------
    bool PhysicalTrace::find_replay_template(
        BeginOp* op, std::set<RtEvent>& map_applied_events,
        std::set<ApEvent>& execution_preconditions)
    //--------------------------------------------------------------------------
    {
      legion_assert(current_template == nullptr);
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
        if ((idx > 0) && (idx < (int(templates.size()) - 1)))
          next_ready = templates[idx - 1]->test_preconditions(
              op->get_begin_operation(), map_applied_events);
        PhysicalTemplate* current = templates[idx];
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
            current->release_instance_references(map_applied_events);
          // Now delete this template from the entry since at least one of its
          // instances have been deleted and therefore we'll never be able to
          // replay it
          ApEvent pending_deletion;
          if (!current->defer_template_deletion(
                  pending_deletion, map_applied_events))
            delete current;
          if (pending_deletion.exists())
            execution_preconditions.insert(pending_deletion);
          to_delete.emplace_back(idx);
        }
        else if (valid)
        {
          // Valid for everyone
          legion_assert(acquired);
          if ((idx > 0) && (idx < (int(templates.size()) - 1)))
          {
            // Wait for the prefetched analyses to finish and clean them up
            if (next_ready.exists() && !next_ready.has_triggered())
              next_ready.wait();
            templates[idx - 1]->check_preconditions();
          }
          // Everybody agreed to reuse this template so make it the
          // new current template and shuffle it to the front
          current_template = current;
          // Remove any deleted templates before rearranging, by definition
          // all these will be later in the vector than the current template
          // Note this will delete back to front to avoid invalidating
          // indexes later in the to_delete vector
          for (const unsigned& idx : to_delete)
            templates.erase(templates.begin() + idx);
          // Move the template to the end of the vector as most-recently used
          if (idx < int(templates.size() - 1))
            std::rotate(
                templates.begin() + idx, templates.begin() + idx + 1,
                templates.end());
          return true;
        }
        else if (acquired)
          current->release_instance_references(map_applied_events);
        if (idx > 0)
        {
          // If this is the first iteration then we start testing the
          // preconditions for the next iteration now too
          if (idx == (int(templates.size() - 1)))
            current_ready = templates[idx - 1]->test_preconditions(
                op->get_begin_operation(), map_applied_events);
          else  // Shuffle the ready events
            current_ready = next_ready;
        }
      }
      for (const unsigned& idx : to_delete)
        templates.erase(templates.begin() + idx);
      return false;
    }

    //--------------------------------------------------------------------------
    bool PhysicalTrace::begin_physical_trace(
        BeginOp* op, std::set<RtEvent>& map_applied_conditions,
        std::set<ApEvent>& execution_preconditions)
    //--------------------------------------------------------------------------
    {
      legion_assert(current_template == nullptr);
      const bool replaying = find_replay_template(
          op, map_applied_conditions, execution_preconditions);
      if (replaying)
      {
        begin_replay(op, false /*recurrent*/, false /*has intermediate fence*/);
      }
      else  // Start recording a new template
      {
        current_template = op->create_fresh_template(this);
        recording = true;
        recurrent = false;
      }
      return replaying;
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::complete_physical_trace(
        CompleteOp* op, std::set<RtEvent>& map_applied_conditions,
        std::set<ApEvent>& execution_preconditions, bool has_blocking_call)
    //--------------------------------------------------------------------------
    {
      legion_assert(current_template != nullptr);
      if (recording)
      {
        // Complete the recording and see if we have a new pending
        // deletion event that we need to capture
        if (complete_recording(
                op, map_applied_conditions, execution_preconditions,
                has_blocking_call))
          templates.emplace_back(current_template);
      }
      else
      {
        // If this isn't a recurrent replay then we need to apply the
        // postconditions to the equivalence sets, if it is recurrent
        // then we know that the postconditions have already been applied
        if (!recurrent)
          current_template->apply_postconditions(
              op->get_complete_operation(), map_applied_conditions);
        current_template->finish_replay(
            op->get_complete_operation(), execution_preconditions);
        current_template->release_instance_references(map_applied_conditions);
      }
      current_template = nullptr;
    }

    //--------------------------------------------------------------------------
    bool PhysicalTrace::replay_physical_trace(
        RecurrentOp* op, std::set<RtEvent>& map_applied_conditions,
        std::set<ApEvent>& execution_preconditions, bool has_blocking_call,
        bool has_intermediate_fence)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate* non_idempotent_template = nullptr;
      if (recording)
      {
        legion_assert(current_template != nullptr);
        // Complete the recording. If we recorded a replayable template
        // and it is idempotent then we can replay it right away
        if (complete_recording(
                op, map_applied_conditions, execution_preconditions,
                has_blocking_call))
        {
          if (current_template->is_idempotent())
          {
            // Need to check if everyone can acquire all the instances
            bool valid = true;
            bool acquired = current_template->acquire_instance_references();
            if (op->allreduce_template_status(valid, acquired))
            {
              if (acquired)
                current_template->release_instance_references(
                    map_applied_conditions);
              // Now delete this template from the entry since at least one
              // of its instances have been deleted and therefore we'll never
              // be able to replay it
              ApEvent pending_deletion;
              if (!current_template->defer_template_deletion(
                      pending_deletion, map_applied_conditions))
                delete current_template;
              if (pending_deletion.exists())
                execution_preconditions.insert(pending_deletion);
            }
            else
            {
              legion_assert(valid);
              // Replaying this right away
              templates.emplace_back(current_template);
              // Treat the end of the recording as an intermediate fence
              // since we don't actually have events to use for a recurrent
              // replay quite yet since we just did the capture
              // We still set recurrent=true so we don't have to apply
              // the postconditions since we know that is unnecssary
              begin_replay(
                  op, true /*recurrent*/, true /*has intermeidate fence*/);
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
        current_template = nullptr;
      }
      else if (current_template != nullptr)
      {
        // We should only be here if we're going to do a recurrent replay
        // If the current template was non-idempotent then it would have been
        // cleared by the TraceRecurrentOp in trigger_ready
        legion_assert(current_template->is_idempotent());
        // If this isn't a recurrent replay then we need to apply the
        // postconditions to the equivalence sets, if it is recurrent
        // then we know that the postconditions have already been applied
        if (!recurrent)
          current_template->apply_postconditions(
              op->get_complete_operation(), map_applied_conditions);
        current_template->finish_replay(
            op->get_complete_operation(), execution_preconditions);
        begin_replay(op, true /*recurrent*/, has_intermediate_fence);
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
        legion_assert(!templates.empty());
        legion_assert(!templates.back()->is_idempotent());
        non_idempotent_template = templates.back();
        templates.pop_back();
      }
      legion_assert(current_template == nullptr);
      if (non_idempotent_template != nullptr)
      {
        // If we have a non-idempotent template we figure out what kind of
        // replay we're going to do and then put the non-idempotent template
        // in thie right place in the list of templates
        if (begin_physical_trace(
                op, map_applied_conditions, execution_preconditions))
        {
          legion_assert(!templates.empty());
          // We found another template to replay so it will be the last
          // one on the list, therefore put the non-idempotent one right
          // before it on the list as the one most recently captured/replayed
          // before we found this new template to replay
          templates.insert(templates.end() - 1, non_idempotent_template);
          return true;
        }
        else
        {
          templates.emplace_back(non_idempotent_template);
          return false;
        }
      }
      else
        return begin_physical_trace(
            op, map_applied_conditions, execution_preconditions);
    }

    //--------------------------------------------------------------------------
    bool PhysicalTrace::complete_recording(
        CompleteOp* op, std::set<RtEvent>& map_applied_conditions,
        std::set<ApEvent>& execution_postconditions, bool has_blocking_call)
    //--------------------------------------------------------------------------
    {
      legion_assert(recording);
      legion_assert(current_template != nullptr);
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
          legion_assert(!templates.empty());
          PhysicalTemplate* to_delete = templates.front();
          ApEvent pending_deletion;
          if (!to_delete->defer_template_deletion(
                  pending_deletion, map_applied_conditions))
            delete to_delete;
          else if (pending_deletion.exists())
            execution_postconditions.insert(pending_deletion);
          // Remove the least recently used (first) one from the vector
          // shift it to the back first though, should be fast
          if (templates.size() > 1)
            std::rotate(
                templates.begin(), templates.begin() + 1, templates.end());
          templates.pop_back();
        }
        if (++new_template_count > LEGION_NEW_TEMPLATE_WARNING_COUNT)
        {
          InnerContext* ctx = logical_trace->context;
          {
            Warning warning;
            warning << "The runtime has created "
                    << LEGION_NEW_TEMPLATE_WARNING_COUNT
                    << " new replayable templates for trace "
                    << logical_trace->get_trace_id() << " in task "
                    << ctx->get_task_name() << " (UID " << ctx->get_unique_id()
                    << ") without replaying any existing templates. This may "
                       "mean that your mapper "
                    << "is not making mapper decisions conducive to replaying "
                       "templates. Please "
                    << "check that your mapper is making decisions that align "
                       "with prior templates. "
                    << "If you believe that this number of templates is "
                       "reasonable, please adjust "
                    << "the settings for LEGION_NEW_TEMPLATE_WARNING_COUNT in "
                       "legion_config.h.";
            warning.raise();
          }
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
          InnerContext* ctx = logical_trace->context;
          {
            Warning warning;
            warning << "The runtime has failed to memoize the trace more than "
                    << LEGION_NON_REPLAYABLE_WARNING
                    << " times due to the absence of a replayable "
                    << "template. It is highly likely that trace "
                    << logical_trace->get_trace_id() << " in task "
                    << ctx->get_task_name() << " (UID " << ctx->get_unique_id()
                    << ") will not be memoized for the rest of execution. The "
                       "most recent template "
                    << "was not replayable for the following reason: "
                    << ((status == NOT_REPLAYABLE_BLOCKING) ? "blocking call" :
                                                              "virtual mapping")
                    << ". Please change the mapper to stop making memoization "
                       "requests.";
            warning.raise();
          }
          nonreplayable_count = 0;
        }
        // Defer template deletion
        ApEvent pending_deletion;
        if (!current_template->defer_template_deletion(
                pending_deletion, map_applied_conditions))
          delete current_template;
        else if (pending_deletion.exists())
          execution_postconditions.insert(pending_deletion);
        return false;
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::begin_replay(
        BeginOp* op, bool recur, bool has_intermediate_fence)
    //--------------------------------------------------------------------------
    {
      legion_assert(current_template != nullptr);
      recording = false;
      recurrent = recur;
      new_template_count = 0;
      // If we had an intermeidate execution fence between replays then
      // we should no longer be considered recurrent when we replay the trace
      // We're also not going to be considered recurrent here if we didn't
      // do fence elision since since we'll still need to track the fence
      current_template->initialize_replay(
          op->get_begin_completion(),
          recurrent && perform_fence_elision && !has_intermediate_fence);
    }

  }  // namespace Internal
}  // namespace Legion
