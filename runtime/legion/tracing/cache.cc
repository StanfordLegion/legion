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

#include <cmath>

#include "legion/tracing/cache.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Trace Cache
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceCache::TraceCache(InnerContext* ctx)
      : context(ctx), operation_start_idx(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool TraceCache::record_operation(
        Operation* op, Murmur3Hasher::Hash hash, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      // Short circuit if we don't have any traces recorded yet
      // Technically this is superfluous because we'll end up flushing the
      // buffer later in this function if we don't have any active pointers
      // in the trie, but this saves us some work for a simple test and
      // also makes the it clearer that there is no lazy evaluation until
      // we actually start observing traces.
      if (trie.empty())
      {
        operation_start_idx++;
        context->add_to_dependence_queue(
            op, nullptr, false /*unordered*/, false /*outermost*/);
        return true;
      }
      // We only start lazily
      operations.emplace(op);
      // Update all watching pointers. This is very similar to the advancing
      // of pointers in the TraceOccurrenceWatcher.
      active_watching_pointers.emplace_back(trie.get_root(), opidx);
      // Avoid a reallocation in the same way by copying in place.
      unsigned current_index = 0;
      for (unsigned idx = 0; idx < active_watching_pointers.size(); idx++)
      {
        WatchPointer& pointer = active_watching_pointers[idx];
        if (!pointer.advance(hash))
          continue;
        if (current_index != idx)
          active_watching_pointers[current_index++] = pointer;
        else
          current_index++;
      }
      // Erase the remaining unused elements
      active_watching_pointers.erase(
          active_watching_pointers.begin() + current_index,
          active_watching_pointers.end());

      // Now update all the commit pointers. This process is more tricky, as
      // we have to actually decide to flush operations through the dependence
      // queue as we match traces. Additionally, we have to manage heuristics
      // around which traces we should take
      active_commit_pointers.emplace_back(trie.get_root(), opidx);
      current_index = 0;
      for (unsigned idx = 0; idx < active_commit_pointers.size(); idx++)
      {
        CommitPointer& pointer = active_commit_pointers[idx];
        if (!pointer.advance(hash))
          continue;
        if (pointer.complete())
        {
          // Add the new completed pointer to the vector of
          // completed_commit_pointers. We calculate the score of
          // the pointer at the current opidx and use that score
          // for the rest of the operations on the pointer. We must
          // maintain the sortedness of completed_commit_pointers, so we
          // use upper_bound + insert to insert the pointer into the right
          // place. A future investigation can see if a std::set is a better
          // data structure, but the controlled memory usage + cache behavior
          // of the vector is likely better on the small sizes that
          // completed_commit_pointers should grow to.
          legion_assert(std::is_sorted(
              completed_commit_pointers.begin(),
              completed_commit_pointers.end()));
          FrozenCommitPointer frozen(pointer, opidx);
          completed_commit_pointers.insert(
              std::upper_bound(
                  completed_commit_pointers.begin(),
                  completed_commit_pointers.end(), frozen),
              frozen);
        }
        else if (current_index != idx)
          active_commit_pointers[current_index++] = pointer;
        else
          current_index++;
      }
      // Erase the rest of the active pointers.
      active_commit_pointers.erase(
          active_commit_pointers.begin() + current_index,
          active_commit_pointers.end());

      // Find the minimum opidx of the active and completed pointers.
      uint64_t earliest_active = std::numeric_limits<uint64_t>::max();
      for (const CommitPointer& commit_pointer : active_commit_pointers)
        earliest_active = std::min(earliest_active, commit_pointer.get_opidx());
      uint64_t earliest_completed = std::numeric_limits<uint64_t>::max();
      for (const FrozenCommitPointer& frozen_pointer :
           completed_commit_pointers)
        earliest_completed =
            std::min(earliest_completed, frozen_pointer.get_opidx());
      uint64_t earliest_opidx = std::min(earliest_active, earliest_completed);

      // First, flush all operations until the earliest_opidx, as there is
      // nothing we are considering before there. If there are no active
      // or completed operations at all, then we just flush the entire buffer.
      if (active_commit_pointers.empty() && completed_commit_pointers.empty())
        flush_buffer();
      else
        flush_buffer(earliest_opidx);

      if (!completed_commit_pointers.empty())
      {
        if (active_commit_pointers.empty())
        {
          // In this case, there are only completed pointers. We'll try to flush
          // through as many operations as we can. The heuristic is to take
          // traces ordered by score. This hueuristic can lead to suboptimal
          // trace replay, for example if the operation stream is AB and we have
          // completed traces [A, B] but B has a higher score, we'll issue A
          // without actually replaying that trace. Doing this seems to require
          // some fancier logic with interval trees or something, so we'll stick
          // to the simpler piece. completed_commit_pointers should already be
          // sorted to have the highest scoring traces at the front.
          legion_assert(std::is_sorted(
              completed_commit_pointers.begin(),
              completed_commit_pointers.end()));
          for (FrozenCommitPointer& frozen_pointer : completed_commit_pointers)
          {
            // If we're considering a pointer that starts earlier than the
            // pending set of operations, then that trace is behind us. So
            // we just continue onto the next trace.
            if (frozen_pointer.get_opidx() < operation_start_idx)
              continue;
            // Now, flush the buffer up until the start of this trace.
            flush_buffer(frozen_pointer.get_opidx());
            // Finally, we can issue the trace.
            TraceID tid = frozen_pointer.replay(context);
            replay_trace(
                frozen_pointer.get_opidx() + frozen_pointer.get_length(), tid);
          }
          // The set of completed pointers is now empty.
          completed_commit_pointers.clear();
          // At this point, we don't have any completed or active pointers,
          // so flush any remaining operations.
          flush_buffer();
        }
        else
        {
          // In this case, we have both completed and active pointers.
          // What we actually do will change depending on what the overlaps
          // between our completed and active pointers actually are.
          if (earliest_completed < earliest_active)
          {
            // In this case, we have some completed pointers that we could
            // potentially replay behind our active pointers. We're going to
            // take a heuristic here where we only flush completed pointers
            // that do not overlap with any active pointers. This biases us
            // towards longer traces when possible, and makes the replay
            // resilient against different kinds of traces being inserted,
            // such as AB, when the trie already contains BC.
            legion_assert(std::is_sorted(
                completed_commit_pointers.begin(),
                completed_commit_pointers.end()));
            uint64_t cutoff_opidx = earliest_active;
            uint64_t pending_completion_cutoff =
                std::numeric_limits<uint64_t>::max();
            current_index = 0;
            for (unsigned idx = 0; idx < completed_commit_pointers.size();
                 idx++)
            {
              FrozenCommitPointer& pointer = completed_commit_pointers[idx];
              // If this completed pointer spans into an active pointer,
              // then we need to save it for later.
              if (cutoff_opidx <= (pointer.get_opidx() + pointer.get_length()))
              {
                completed_commit_pointers[current_index++] = pointer;
                // If we decide to skip this completed pointer because it
                // overlaps with an active pointer, we shouldn't replay any
                // completed pointers that overlap with this good pointer, as it
                // scored higher.
                pending_completion_cutoff =
                    std::min(pending_completion_cutoff, pointer.get_opidx());
                continue;
              }
              // As before, any pointers that we are already past can be
              // ignored.
              if (pointer.get_opidx() < operation_start_idx)
                continue;
              // Lastly, make sure that this completed pointer doesn't
              // invalidate a completed pointer that was re-queued with a better
              // score.
              if (pending_completion_cutoff <=
                  (pointer.get_opidx() + pointer.get_length()))
              {
                completed_commit_pointers[current_index++] = pointer;
                continue;
              }
              // Here, we can finally replay the trace.
              flush_buffer(pointer.get_opidx());
              TraceID tid = pointer.replay(context);
              replay_trace(pointer.get_opidx() + pointer.get_length(), tid);
            }
            // Clear the remaining invalid completions.
            completed_commit_pointers.erase(
                completed_commit_pointers.begin() + current_index,
                completed_commit_pointers.end());

            // Since we iterated through completed_commit_pointers in sorted
            // order to construct the new vector, it should still be sorted.
            legion_assert(std::is_sorted(
                completed_commit_pointers.begin(),
                completed_commit_pointers.end()));
#ifdef LEGION_DEBUG
            // Since we waited to not cut off any active pointers, there should
            // not be any invalid active pointers.
            for (const CommitPointer& commit_pointer : active_commit_pointers)
            {
              legion_assert(operation_start_idx <= commit_pointer.get_opidx());
            }
#endif
          }
          else if (earliest_completed == earliest_active)
          {
            // We should never be in the case where an active and completed
            // pointer are starting at the same opidx.
            // TODO (rohany): This can actually happen if we allow for
            // prefixes like the comment earlier above suggests. We'll deal
            // with that when we get there. Right now we don't allow prefixes
            // and instead split longer traces into smaller ones with prefix
            // and postfix parts split into separate traces.
            std::abort();
          }
          // Else There are active pointers behind our earliest completed
          // pointer, so there's no point at even looking at the completed
          // pointers.
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool TraceCache::is_operation_ignorable_in_traces(Operation* op)
    //--------------------------------------------------------------------------
    {
      switch (op->get_operation_kind())
      {
        case DISCARD_OP_KIND:
          return true;
        default:
          return false;
      }
    }

    //--------------------------------------------------------------------------
    bool TraceCache::record_noop(Operation* op)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_operation_ignorable_in_traces(op));
      if (trie.empty())
      {
        operation_start_idx++;
        context->add_to_dependence_queue(
            op, nullptr, false /*unordered*/, false /*outermost*/);
        return true;
      }
      // If the operation is a noop during traces, then the replayer
      // takes a much simpler process. In particular, none of the pointers
      // advance or are cancelled, but their depth increases to account
      // for the extra operation. Do a special advance on each active
      // commit pointer.
      operations.emplace(op);
      for (CommitPointer& commit_pointer : active_commit_pointers)
        commit_pointer.advance_for_trace_noop();
      // Because this operation does not invalidate any pointers,
      // we don't always need to compute a minumum and flush the
      // head of the buffer up until the minimum. This is because
      // a previous operation already did that for the active pointers,
      // and this operation does not create any new active or complete
      // pointers. So the addition of this operation cannot require
      // re-flushing the head of the buffer. However, to make the
      // processing of these operations slightly more eager, we'll
      // flush them through if there aren't any active pointers,
      // rather than waiting for the next operation to come along.
      if (active_commit_pointers.empty() && completed_commit_pointers.empty())
        flush_buffer();
      return true;
    }

    //--------------------------------------------------------------------------
    bool TraceCache::has_prefix(
        const std::vector<Murmur3Hasher::Hash>& hashes) const
    //--------------------------------------------------------------------------
    {
      return trie.prefix(&hashes.front(), hashes.size());
    }

    //--------------------------------------------------------------------------
    void TraceCache::insert(
        std::vector<Murmur3Hasher::Hash>& hashes, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      trie.insert(
          &hashes.front(), hashes.size(), TraceInfo(opidx, hashes.size()));
    }

    //--------------------------------------------------------------------------
    void TraceCache::flush(uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      // We have to flush all active pointers from the trie, and
      // them attempt to launch traces for any completed pointers.
      // First, clear the vectors of active pointers.
      active_watching_pointers.clear();
      active_commit_pointers.clear();
      // If we have no completed pointers, flush all pending operations and
      // early exit.
      if (!completed_commit_pointers.empty())
      {
        legion_assert(std::is_sorted(
            completed_commit_pointers.begin(),
            completed_commit_pointers.end()));
        // Now that we have some (sorted) completed pointers, issue them.
        for (FrozenCommitPointer& frozen_pointer : completed_commit_pointers)
        {
          // If we're considering a pointer that starts earlier than the
          // pending set of operations, then that trace is behind us. So
          // we just continue onto the next trace.
          if (frozen_pointer.get_opidx() < operation_start_idx)
            continue;
          // Now, flush the buffer up until the start of this trace.
          flush_buffer(frozen_pointer.get_opidx());
          // Finally, we can issue the trace.
          TraceID tid = frozen_pointer.replay(context);
          replay_trace(
              frozen_pointer.get_opidx() + frozen_pointer.get_length(), tid);
        }
        completed_commit_pointers.clear();
      }
      // Flush all remaining operations.
      flush_buffer();
    }

    //--------------------------------------------------------------------------
    void TraceCache::replay_trace(uint64_t opidx, TraceID tid)
    //--------------------------------------------------------------------------
    {
      // rohany): I don't think that this should happen when we're
      // actually calling replay, but better safe than sorry.
      legion_assert(operation_start_idx < opidx);
      // Similar logic as flush_buffer, but issue a begin and end trace
      // around the flushed operations.
      context->begin_trace(
          tid, false /*logical*/, false /*static*/, nullptr /*managed*/,
          false /*deprecated*/, nullptr /*provenance*/);
      uint64_t difference = opidx - this->operation_start_idx;
      operation_start_idx += difference;
      unsigned traced_ops = 0;
      for (uint64_t idx = 0; idx < difference; idx++)
      {
        Operation* next = operations.front();
        if (!is_operation_ignorable_in_traces(next))
        {
          traced_ops++;
          context->add_to_dependence_queue(
              next, nullptr /*dependences*/, false /*unordered*/,
              false /*outermost*/);
        }
        else
          next->deactivate();
        operations.pop();
      }
      context->end_trace(tid, false /*deprecated*/, nullptr /*provenance*/);
      log_auto_trace.info() << "Replaying trace " << tid << " of length "
                            << traced_ops << " at opidx: " << opidx;
    }

    //--------------------------------------------------------------------------
    void TraceCache::flush_buffer(void)
    //--------------------------------------------------------------------------
    {
      operation_start_idx += operations.size();
      while (!operations.empty())
      {
        context->add_to_dependence_queue(
            operations.front(), nullptr /*dependences*/, false /*unordered*/,
            false /*outermost*/);
        operations.pop();
      }
    }

    //--------------------------------------------------------------------------
    void TraceCache::flush_buffer(uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      // If we've already advanced beyond this point, then there's nothing to
      // do.
      if (opidx <= operation_start_idx)
        return;
      uint64_t difference = opidx - operation_start_idx;
      operation_start_idx += difference;
      for (uint64_t idx = 0; idx < difference; idx++)
      {
        context->add_to_dependence_queue(
            operations.front(), nullptr /*dependences*/, false /*unordered*/,
            false /*outermost*/);
        operations.pop();
      }
    }

    //--------------------------------------------------------------------------
    bool TraceCache::WatchPointer::advance(Murmur3Hasher::Hash token)
    //--------------------------------------------------------------------------
    {
      // We couldn't advance the pointer. Importantly, we can't check
      // node.end here, as this node could be the prefix of another
      // trace in the trie.
      node = node->find_child(token);
      if (node == nullptr)
        return false;
      // If we've hit the end of a string,
      // mark it as visited and update the score.
      if (node->get_end() && (node->get_value().opidx <= opidx))
        node->get_value().visit(opidx);
      return true;
    }

    //--------------------------------------------------------------------------
    bool TraceCache::CommitPointer::advance(Murmur3Hasher::Hash token)
    //--------------------------------------------------------------------------
    {
      node = node->find_child(token);
      if (node == nullptr)
        return false;
      depth++;
      return true;
    }

    //--------------------------------------------------------------------------
    bool TraceCache::CommitPointer::complete(void) const
    //--------------------------------------------------------------------------
    {
      // By ensuring that we don't insert traces that are superstrings
      // of existing traces in the TraceOccurrenceWatcher, this property
      // should be preserved as traces migrate into the TraceReplayer,
      // which relies on a completed pointer as not having any more
      // pending operations to be waiting for.
      legion_assert(!node->get_end() || node->get_children().empty());
      return (node->get_end() && (node->get_value().opidx <= opidx));
    }

    //--------------------------------------------------------------------------
    TraceID TraceCache::CommitPointer::replay(InnerContext* context)
    //--------------------------------------------------------------------------
    {
      TraceInfo& info = node->get_value();
      if (info.replays++ == 0)
        info.tid = context->generate_dynamic_trace_id();
      return info.tid;
    }

    //--------------------------------------------------------------------------
    double TraceCache::CommitPointer::score(uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      return node->get_value().score(opidx);
    }

    //--------------------------------------------------------------------------
    void TraceCache::TraceInfo::visit(uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      // First, compute the difference in trace lengths that
      // this trace was last visited at.
      uint64_t previous_visit = last_visited_opidx;
      uint64_t diff_in_traces = (opidx - previous_visit) / length;
      // Visits only count if they are at least 1 trace length away.
      if (diff_in_traces == 0)
        return;
      // Compute the new visit count by decaying the old visit count
      // and then adding one.
      decaying_visits = (std::pow(R, diff_in_traces) * decaying_visits) + 1;
      // If we visited this trace exactly len(trace) operations after
      // the previous visit, then we're able to replay this trace back-to-back,
      // which is nice to know for scoring. Check previous_visit != 0 to ensure
      // that we at least have one visit before counting idempotent visits.
      if ((previous_visit != 0) && ((opidx - previous_visit) == length))
      {
        uint64_t previous_idemp_visit = last_idempotent_visit_opidx;
        uint64_t idemp_diff = (opidx - previous_idemp_visit) / length;
        decaying_idempotent_visits =
            (std::pow(R, idemp_diff) * decaying_idempotent_visits) + 1;
        last_idempotent_visit_opidx = opidx;
      }
      last_visited_opidx = opidx;
    }

    //--------------------------------------------------------------------------
    double TraceCache::TraceInfo::score(uint64_t opidx) const
    //--------------------------------------------------------------------------
    {
      // Do a similar calculation as visit, where we decay the score
      // of the trace as if reading from opidx.
      // TODO (rohany): I'm not entirely convinced that this is necessary,
      //  because if we're computing the score of something, then it should
      //  have just been visited, i.e. it's score was just updated. However,
      //  this worked well in the simulator so I'll start with it before
      //  switching it up.
      uint64_t previous_visit = last_visited_opidx;
      uint64_t diff_in_traces = (opidx - previous_visit) / length;
      // Increase the visit count by 1 when computing the score so that
      // traces that haven't been visited before don't have a 0 score.
      // The initial score is num_visits * length.
      double score =
          ((std::pow(R, diff_in_traces) * decaying_visits) + 1) * length;
      // Next, we cap the score so that the first trace that gets replayed
      // doesn't get replayed forever.
      score = std::min(score, SCORE_CAP_MULT * ((double)this->length));
      // Then, increase the score a little bit if a trace has already been
      // replayed to favor replays.
      double capped_replays =
          std::max(std::min(REPLAY_SCALE, (double)replays), (double)1);
      double capped_idemp_visits = std::max(
          std::min(IDEMPOTENT_VISIT_SCALE, decaying_idempotent_visits), 1.0);
      return score * capped_replays * capped_idemp_visits;
    }

  }  // namespace Internal
}  // namespace Legion
