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

#include "legion/legion_auto_trace.h"

#include <cmath>
#include <iterator>
#include <unordered_set>

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Trace Cache
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceCache::TraceCache(InnerContext *ctx)
      : context(ctx), operation_start_idx(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void TraceCache::record_operation(Operation *op, Murmur3Hasher::Hash hash,
                                      uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      operations.emplace(op);
      // Update all watching pointers. This is very similar to the advancing
      // of pointers in the TraceOccurrenceWatcher.
      active_watching_pointers.emplace_back(trie.get_root(), opidx);
      // Avoid a reallocation in the same way by copying in place.
      unsigned current_index = 0;
      for (unsigned idx = 0; idx < active_watching_pointers.size(); idx++)
      {
        WatchPointer &pointer = active_watching_pointers[idx];
        if (!pointer.advance(hash))
          continue;
        if (current_index != idx)
          active_watching_pointers[current_index++] = pointer;
        else
          current_index++;
      }
      // Erase the remaining unused elements
      active_watching_pointers.erase(active_watching_pointers.begin() + 
          current_index, active_watching_pointers.end());

      // Now update all the commit pointers. This process is more tricky, as
      // we have to actually decide to flush operations through the dependence
      // queue as we match traces. Additionally, we have to manage heuristics
      // around which traces we should take
      active_commit_pointers.emplace_back(trie.get_root(), opidx);
      current_index = 0;
      for (unsigned idx = 0; idx < active_commit_pointers.size(); idx++)
      {
        CommitPointer &pointer = active_commit_pointers[idx];
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
#ifdef DEBUG_LEGION
          assert(std::is_sorted(completed_commit_pointers.begin(),
                completed_commit_pointers.end()));
#endif
          FrozenCommitPointer frozen(pointer, opidx);
          completed_commit_pointers.insert(
            std::upper_bound(completed_commit_pointers.begin(),
              completed_commit_pointers.end(), frozen), frozen);
        }
        else if (current_index != idx)
          active_commit_pointers[current_index++] = pointer;
        else
          current_index++;
      }
      // Erase the rest of the active pointers.
      active_commit_pointers.erase(active_commit_pointers.begin() + 
          current_index, active_commit_pointers.end());

      // Find the minimum opidx of the active and completed pointers.
      uint64_t earliest_active = std::numeric_limits<uint64_t>::max();
      for (std::vector<CommitPointer>::const_iterator it =
            active_commit_pointers.begin(); it !=
            active_commit_pointers.end(); it++)
        earliest_active = std::min(earliest_active, it->get_opidx());
      uint64_t earliest_completed = std::numeric_limits<uint64_t>::max();
      for (std::vector<FrozenCommitPointer>::const_iterator it =
            completed_commit_pointers.begin(); it !=
            completed_commit_pointers.end(); it++)
        earliest_completed = std::min(earliest_completed, it->get_opidx());
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
#ifdef DEBUG_LEGION
          assert(std::is_sorted(completed_commit_pointers.begin(),
                completed_commit_pointers.end()));
#endif
          for (std::vector<FrozenCommitPointer>::iterator it =
                completed_commit_pointers.begin(); it !=
                completed_commit_pointers.end(); it++)
          {
            // If we're considering a pointer that starts earlier than the
            // pending set of operations, then that trace is behind us. So
            // we just continue onto the next trace.
            if (it->get_opidx() < operation_start_idx)
              continue;
            // Now, flush the buffer up until the start of this trace.
            flush_buffer(it->get_opidx());
            // Finally, we can issue the trace.
            TraceID tid = it->replay(context);
            replay_trace(it->get_opidx() + it->get_length(), tid);
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
#ifdef DEBUG_LEGION
            assert(std::is_sorted(completed_commit_pointers.begin(),
                  completed_commit_pointers.end()));
#endif
            uint64_t cutoff_opidx = earliest_active;
            uint64_t pending_completion_cutoff = 
              std::numeric_limits<uint64_t>::max();
            current_index = 0;
            for (unsigned idx = 0; idx < completed_commit_pointers.size(); idx++)
            {
              FrozenCommitPointer &pointer = completed_commit_pointers[idx];
              // If this completed pointer spans into an active pointer, 
              // then we need to save it for later.
              if (cutoff_opidx <= (pointer.get_opidx() + pointer.get_length()))
              {
                completed_commit_pointers[current_index++] = pointer;
                // If we decide to skip this completed pointer because it overlaps
                // with an active pointer, we shouldn't replay any completed pointers
                // that overlap with this good pointer, as it scored higher.
                pending_completion_cutoff = 
                  std::min(pending_completion_cutoff, pointer.get_opidx());
                continue;
              }
              // As before, any pointers that we are already past can be ignored.
              if (pointer.get_opidx() < operation_start_idx)
                continue;
              // Lastly, make sure that this completed pointer doesn't invalidate
              // a completed pointer that was re-queued with a better score.
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
            completed_commit_pointers.erase(completed_commit_pointers.begin() + 
                current_index, completed_commit_pointers.end());

#ifdef DEBUG_LEGION
            // Since we iterated through completed_commit_pointers in sorted order
            // to construct the new vector, it should still be sorted.
            assert(std::is_sorted(completed_commit_pointers.begin(), 
                  completed_commit_pointers.end()));
            // Since we waited to not cut off any active pointers, there should not
            // be any invalid active pointers.
            for (std::vector<CommitPointer>::const_iterator it =
                  active_commit_pointers.begin(); it !=
                  active_commit_pointers.end(); it++)
            {
              assert(operation_start_idx <= it->get_opidx());
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
            assert(false);
          }
          // Else There are active pointers behind our earliest completed
          // pointer, so there's no point at even looking at the completed
          // pointers.
        }
      }
    }

    //--------------------------------------------------------------------------
    bool TraceCache::is_operation_ignorable_in_traces(Operation* op)
    //--------------------------------------------------------------------------
    {
      switch (op->get_operation_kind()) 
      {
        case Operation::OpKind::DISCARD_OP_KIND:
          return true;
        default:
          return false;
      }
    }

    //--------------------------------------------------------------------------
    void TraceCache::record_noop(Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_operation_ignorable_in_traces(op));
#endif
      // If the operation is a noop during traces, then the replayer
      // takes a much simpler process. In particular, none of the pointers
      // advance or are cancelled, but their depth increases to account
      // for the extra operation. Do a special advance on each active
      // commit pointer.
      operations.emplace(op);
      for (std::vector<CommitPointer>::iterator it =
            active_commit_pointers.begin(); it !=
            active_commit_pointers.end(); it++)
        it->advance_for_trace_noop();
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
      if (active_commit_pointers.empty() && 
          completed_commit_pointers.empty())
        flush_buffer();
    }

    //--------------------------------------------------------------------------
    bool TraceCache::has_prefix(
                           const std::vector<Murmur3Hasher::Hash> &hashes) const
    //--------------------------------------------------------------------------
    {
      return trie.prefix(&hashes.front(), hashes.size());
    }

    //--------------------------------------------------------------------------
    void TraceCache::insert(
                       std::vector<Murmur3Hasher::Hash> &hashes, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      trie.insert(&hashes.front(), hashes.size(),
          TraceInfo(opidx, hashes.size()));
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
      if (completed_commit_pointers.empty())
      {
        flush_buffer();
        return;
      }

#ifdef DEBUG_LEGION
      assert(std::is_sorted(completed_commit_pointers.begin(), 
            completed_commit_pointers.end()));
#endif
      // Now that we have some (sorted) completed pointers,  issue them.
      for (std::vector<FrozenCommitPointer>::iterator it =
            completed_commit_pointers.begin(); it !=
            completed_commit_pointers.end(); it++)
      {
        // If we're considering a pointer that starts earlier than the
        // pending set of operations, then that trace is behind us. So
        // we just continue onto the next trace.
        if (it->get_opidx() < operation_start_idx)
          continue;
        // Now, flush the buffer up until the start of this trace.
        flush_buffer(it->get_opidx());
        // Finally, we can issue the trace.
        TraceID tid = it->replay(context);
        replay_trace(it->get_opidx() + it->get_length(), tid);
      }
      completed_commit_pointers.clear();
      // Flush all remaining operations.
      flush_buffer();
    }

    //--------------------------------------------------------------------------
    void TraceCache::replay_trace(uint64_t opidx, TraceID tid)
    //--------------------------------------------------------------------------
    {
      // rohany): I don't think that this should happen when we're 
      // actually calling replay, but better safe than sorry.
#ifdef DEBUG_LEGION
      assert(operation_start_idx < opidx);
#endif
      // Similar logic as flush_buffer, but issue a begin and end trace
      // around the flushed operations.
      context->begin_trace(tid, false/*logical*/, false/*static*/,
          NULL/*managed*/, false/*deprecated*/, NULL/*provenance*/,
          false/*from application*/);
      uint64_t difference = opidx - this->operation_start_idx;
      operation_start_idx += difference;
      unsigned traced_ops = 0;
      for (uint64_t idx = 0; idx < difference; idx++)
      {
        if (!is_operation_ignorable_in_traces(operations.front()))
          traced_ops++;
        context->add_to_dependence_queue(operations.front(),
            NULL/*dependences*/, false/*unordered*/, false/*outermost*/);
        operations.pop();
      }
      context->end_trace(tid, false/*deprecated*/, NULL/*provenance*/,
          false/*from application*/);
      log_auto_trace.info() << "Replaying trace " << tid
                            << " of length "
                            << traced_ops
                            << " at opidx: " << opidx;
    }

    //--------------------------------------------------------------------------
    void TraceCache::flush_buffer(void)
    //--------------------------------------------------------------------------
    {
      operation_start_idx += operations.size();
      while (!operations.empty()) 
      {
        context->add_to_dependence_queue(operations.front(),
            NULL/*dependences*/, false/*unordered*/, false/*outermost*/);
        operations.pop();
      }
    }

    //--------------------------------------------------------------------------
    void TraceCache::flush_buffer(uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      // If we've already advanced beyond this point, then there's nothing to do.
      if (opidx <= operation_start_idx)
        return;
      uint64_t difference = opidx - operation_start_idx;
      operation_start_idx += difference;
      for (uint64_t idx = 0; idx < difference; idx++)
      {
        context->add_to_dependence_queue(operations.front(),
            NULL/*dependences*/, false/*unordered*/, false/*outermost*/);
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
      if (node == NULL)
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
      if (node == NULL)
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
#ifdef DEBUG_LEGION
      assert(!node->get_end() || node->get_children().empty());
#endif
      return (node->get_end() && (node->get_value().opidx <= opidx));
    }

    //--------------------------------------------------------------------------
    TraceID TraceCache::CommitPointer::replay(InnerContext *context)
    //--------------------------------------------------------------------------
    {
      TraceInfo &info = node->get_value();
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
      decaying_visits = (pow(R, diff_in_traces) * decaying_visits) + 1;
      // If we visited this trace exactly len(trace) operations after
      // the previous visit, then we're able to replay this trace back-to-back,
      // which is nice to know for scoring. Check previous_visit != 0 to ensure
      // that we at least have one visit before counting idempotent visits.
      if ((previous_visit != 0) && ((opidx - previous_visit) == length))
      {
        uint64_t previous_idemp_visit = last_idempotent_visit_opidx;
        uint64_t idemp_diff = (opidx - previous_idemp_visit) / length;
        decaying_idempotent_visits =
          (pow(R, idemp_diff) * decaying_idempotent_visits) + 1;
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
      double score = ((pow(R, diff_in_traces) * decaying_visits) + 1) * length;
      // Next, we cap the score so that the first trace that gets replayed
      // doesn't get replayed forever.
      score = std::min(score, SCORE_CAP_MULT * ((double)this->length));
      // Then, increase the score a little bit if a trace has already been
      // replayed to favor replays.
      double capped_replays =
        std::max(std::min(REPLAY_SCALE, (double)replays), (double)1);
      double capped_idemp_visits = std::max(std::min(IDEMPOTENT_VISIT_SCALE,
            decaying_idempotent_visits), 1.0);
      return score * capped_replays * capped_idemp_visits;
    }

    /////////////////////////////////////////////////////////////
    // Occurrence Watcher
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OccurrenceWatcher::OccurrenceWatcher(InnerContext *context,
                                      const Mapper::ContextConfigOutput &config)
      : cache(context), visit_threshold(config.auto_tracing_visit_threshold)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void OccurrenceWatcher::record_operation(Operation *op,
                                       Murmur3Hasher::Hash hash, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      // Every new hash chould be the start of a new trace
      active_pointers.emplace_back(trie.get_root(), opidx);
      // We'll copy pointers down if they can't be advanced as we iterate
      unsigned current_index = 0;
      for (unsigned idx = 0; idx < active_pointers.size(); idx++)
      {
        TriePointer &pointer = active_pointers[idx];
        if (!pointer.advance(hash))
          continue;
        active_pointers[current_index++] = pointer;
        // See if we found a trace
        if (pointer.complete())
        {
          // If this pointer corresponds to a completed trace, then we have
          // some work to do. First, increment the number of visits on the node.
          TrieNode<Murmur3Hasher::Hash, TraceCandidate>* node = pointer.node;
          TraceCandidate &candidate = node->get_value();
          // Visits only count if they occur at least len(trace) operations
          // after the previous visit. This avoids overcounting traces that
          // look like ABCABC with a count at each repetition of ABC rather
          // than at each occurrence of ABCABC.
          if (pointer.depth <= (opidx - candidate.previous_visited_opidx))
          {
            candidate.visits++;
            candidate.previous_visited_opidx = opidx;
          }
          if ((visit_threshold <= candidate.visits) && !candidate.completed)
          {
            candidate.completed = true;
            std::vector<Murmur3Hasher::Hash> hashes(pointer.depth);
            for (unsigned j = 0; j < pointer.depth; j++) 
            {
#ifdef DEBUG_LEGION
              assert(node != nullptr);
#endif
              hashes[pointer.depth - j - 1] = node->get_token();
              node = node->get_parent();
            }
            // Check to see if this is a prefix of an existing trace
            // TODO (rohany): Do we need to think about superstrings here?
            if (!cache.has_prefix(hashes))
            {
              log_auto_trace.debug() << "Committing trace: "
                                     << candidate.opidx  << " of length: "
                                     << pointer.depth;
              cache.insert(hashes, opidx);
            }
          }
        }
      }
      // At this point we can shrink down the vector to the remaining size
      active_pointers.erase(
          active_pointers.begin()+current_index, active_pointers.end());
      // Now tell the trace cache to reocrd the operation too
      cache.record_operation(op, hash, opidx);
    }

    //--------------------------------------------------------------------------
    bool OccurrenceWatcher::TriePointer::advance(Murmur3Hasher::Hash token)
    //--------------------------------------------------------------------------
    {
      node = node->find_child(token);
      // We couldn't advance the pointer. Importantly, we can't check
      // node.end here, as this node could be the prefix of another
      // trace in the trie
      if (node == NULL)
        return false;
      // Otherwise, move down to the node's child.
      depth++;
      return true;
    }

    //--------------------------------------------------------------------------
    bool OccurrenceWatcher::TriePointer::complete(void) const
    //--------------------------------------------------------------------------
    {
      return (node->get_end() && (node->get_value().opidx <= opidx));
    }

    //--------------------------------------------------------------------------
    void OccurrenceWatcher::record_noop(Operation *op)
    //--------------------------------------------------------------------------
    {
      cache.record_noop(op);
    }

    //--------------------------------------------------------------------------
    void OccurrenceWatcher::flush(uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      active_pointers.clear();
      cache.flush(opidx);
    } 

    //--------------------------------------------------------------------------
    void OccurrenceWatcher::insert(const Murmur3Hasher::Hash *hashes,
                                                    size_t size, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      trie.insert(hashes, size, TraceCandidate(opidx));
    }

    //--------------------------------------------------------------------------
    TrieQueryResult OccurrenceWatcher::query(const Murmur3Hasher::Hash *hashes,
                                             size_t size) const
    //--------------------------------------------------------------------------
    {
      return trie.query(hashes, size);
    }

    /////////////////////////////////////////////////////////////
    // Trace Recognizer
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceRecognizer::TraceRecognizer(InnerContext *ctx,
                                     const Mapper::ContextConfigOutput &config)
      : context(ctx), batchsize(config.auto_tracing_batchsize),
        multi_scale_factor(config.auto_tracing_multi_scale_factor),
        min_trace_length(config.auto_tracing_min_trace_length),
        max_trace_length(config.auto_tracing_max_trace_length),
        watcher(ctx, config), unique_hash_value(0), wait_interval(1)
    //--------------------------------------------------------------------------
    {
      hashes.reserve(batchsize+1);
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::record_operation_hash(Operation *op,
                                          Murmur3Hasher &hasher, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      Murmur3Hasher::Hash hash;
      hasher.finalize(hash);
      hashes.push_back(hash);
      if (check_for_repeats(opidx))
        update_watcher(opidx);
      watcher.record_operation(op, hash, opidx);
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::record_operation_noop(Operation *op)
    //--------------------------------------------------------------------------
    {
      watcher.record_noop(op);
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::record_operation_untraceable(uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      // When encountering a non-traceable operation, insert a
      // dummy hash value into the trace identifier so that the
      // traces it finds don't span across these operations.
      // Generate a unique hash and enqueue it
      hashes.push_back(get_unique_hash());
      if (check_for_repeats(opidx))
        update_watcher(opidx);
      watcher.flush(opidx);
    }

    //--------------------------------------------------------------------------
    Murmur3Hasher::Hash TraceRecognizer::get_unique_hash(void)
    //--------------------------------------------------------------------------
    {
      Murmur3Hasher hasher;
      hasher.hash(Operation::OpKind::LAST_OP_KIND);
      hasher.hash(unique_hash_value);
      Murmur3Hasher::Hash result;
      hasher.finalize(result);
      unique_hash_value++;
      return result;
    }

    //--------------------------------------------------------------------------
    bool TraceRecognizer::check_for_repeats(uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      if (hashes.size() == batchsize)
      {
        // Insert the sentinel token before launching the meta task.
        hashes.push_back(SENTINEL);
        repeat_results.emplace_back(FindRepeatsResult());
        FindRepeatsResult &repeat = repeat_results.back();
        FindRepeatsTaskArgs args(this, &repeat);
        repeat.start = &hashes.front();
        repeat.size = hashes.size();
        repeat.opidx = opidx;
        hashes.swap(repeat.hashes);
        // Runtime meta-task in program order
        repeat.finish_event = implicit_runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, repeat_results.size() > 1 ?
            repeat_results[repeat_results.size()-2].finish_event :
            RtEvent::NO_RT_EVENT);
        hashes.reserve(batchsize + 1);
        return true;
      }
      else if ((hashes.size() % multi_scale_factor) == 0)
      {
        // Otherwise, we are launching an analysis job on a portion of the
        // buffer, given by 2^(ruler function) of the current buffer size.
        // We can conveniently find this value by using the value of the
        // right-most set bit in the index.
        uint64_t index = hashes.size() / multi_scale_factor;
        uint64_t window_size = (index & ~(index - 1)) * multi_scale_factor;
        uint64_t start = hashes.size() - window_size;
        repeat_results.emplace_back(FindRepeatsResult());
        FindRepeatsResult &repeat = repeat_results.back();
        repeat.start = &hashes[start];
        repeat.size = window_size;
        repeat.opidx = opidx;
        FindRepeatsTaskArgs args(this, &repeat);
        // We're going to be a little sneaky around re-using memory for the
        // async jobs, so we're going to make sure that our processing jobs
        // execute in order, because we'll have earlier jobs point to the
        // same memory that later jobs will also use.
        repeat.finish_event = implicit_runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, repeat_results.size() > 1 ?
            repeat_results[repeat_results.size()-2].finish_event :
            RtEvent::NO_RT_EVENT);
        return true;
      }
      else
        return !repeat_results.empty();
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::update_watcher(uint64_t opidx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!repeat_results.empty());
#endif
      // See if we've exceeded the wait interval since the start operation
      // and if not then we just keep going
      if (opidx < (repeat_results.front().opidx + wait_interval))
        return;
      // Scan through the queue and find out how many results are ready
      unsigned ready = 0;
      for (unsigned idx = 0; idx < repeat_results.size(); idx++)
        if (repeat_results[idx].finish_event.has_triggered())
          ready++;
        else
          break;
      // Ask the context how many we should pop off the queue
      // This is necessary for control replication to ensure that
      // all the shards are aligned on how many results to add to
      // the trie at the same point in the program
      bool double_wait_interval = false;
      ready = context->minimize_repeat_results(ready, double_wait_interval);
      for (unsigned idx = 0; idx < ready; idx++)
      {
        FindRepeatsResult &repeats = repeat_results.front();
        for (std::vector<NonOverlappingRepeatsResult>::const_iterator it =
              repeats.result.begin(); it != repeats.result.end(); it++)
          add_trace(repeats.start+it->start, (it->end - it->start), opidx);
        repeat_results.pop_front();
      }
      if (double_wait_interval)
        wait_interval *= 2;
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::add_trace(const Murmur3Hasher::Hash *hashes,
                                    uint64_t size, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      if (size < min_trace_length) 
        return;
      // Check that we aren't uint64_t max before attempting to do
      // some arithmetic that will overflow.
      if ((max_trace_length != std::numeric_limits<unsigned>::max()) &&
          (size > (max_trace_length + min_trace_length))) 
      {
        // If we're larger than the max trace length (plus a little slack),
        // then break up this trace into smaller pieces that we'll insert
        // into our watched data structures. First, insert a trace of the
        // maximum length, then insert the rest of the trace.
        add_trace(hashes, max_trace_length, opidx);
        add_trace(hashes+max_trace_length, size - max_trace_length, opidx);
        return;
      }
      TrieQueryResult query = watcher.query(hashes, size);
      // If we're trying to insert a trace that is either a prefix
      // of another recorded trace, or is already inside the set of
      // recorded traces, then there's nothing to do.
      if (query.prefix || query.contains)
        return;
      // If the trace we're trying to insert is also not a superstring
      // of an existing trace, then this is the easy case where we can
      // just insert it and move on.
      if (!query.superstring) 
      {
        watcher.insert(hashes, size, opidx);
        return;
      }
#ifdef DEBUG_LEGION
      assert(query.superstring);
#endif
      // If the trace we're inserting is a superstring of another
      // string in the recorded set of traces, then splice out the
      // contained prefix and try to insert the rest of the trace.
      add_trace(hashes + query.superstring_match, 
          size - query.superstring_match, opidx);
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::compute_suffix_array(
        const Murmur3Hasher::Hash *str, size_t n,
        std::vector<size_t> &sarray, std::vector<int64_t> &surrogate)
    //--------------------------------------------------------------------------
    {
      // Suffix array construction in O(n*log n) time.
      // The code has been implemented based on the explanations from here:
      // http://www.cs.cmu.edu/~15451-f20/LectureNotes/lec25-suffarray.pdf,
      // with special treatment of radix sort to make it O(n*log n).
      if (n == 0) return;

      // Define a struct for sorting the input string. To handle an
      // arbitrary type T, we use a boolean `present` to ensure that
      // tokens without a "next" value are sorted before any other tokens.
      struct Key {
        Murmur3Hasher::Hash start;
        bool present;
        Murmur3Hasher::Hash next;
        size_t idx;
        bool operator<(const Key& rhs) const {
          return std::tie(start, present, next, idx) <
            std::tie(rhs.start, rhs.present, rhs.next, rhs.idx);
        }
      };

      // First round - O(n log n) sort. We unroll the loop from the
      // lecture notes above once, as we have to do an O(nlog(n)) sort
      // first before we can transition to the radix sorts below.
      std::vector<Key> w(n);
      size_t v = 0;
      {
        for (size_t i = 0; i < n; i++) {
          w[i] = Key {
            .start = str[i],
            .present = i + 1 < n,
            .next = i + 1 < n ? str[i + 1] : Murmur3Hasher::Hash{},
            .idx = i,
          };
        }
        std::sort(w.begin(), w.end());
        Murmur3Hasher::Hash x0 = w[0].start;
        Murmur3Hasher::Hash x1 = w[0].next;
        surrogate[w[0].idx] = 0;
        for (size_t i = 1; i < n; i++) {
          if (x0 != w[i].start || x1 != w[i].next) v++;
          surrogate[w[i].idx] = v;
          x0 = w[i].start;
          x1 = w[i].next;
        }
        // In case we're done, reconstruct the suffix array directly
        // from the w vector.
        if (v >= n - 1) {
          for (size_t i = 0; i < n; i++) {
            sarray[i] = w[i].idx;
          }
          return;
        }
      }

      // After the first round of sorting, we don't need to
      // look at the string anymore, and can just sort based
      // on surrogates computed by the previous sorting step.
      struct SKey {
        int64_t start;
        int64_t next;
        size_t idx;
        bool operator<(const SKey& rhs) const {
          return std::tie(start, next, idx) <
                 std::tie(rhs.start, rhs.next, rhs.idx);
        }
      };

      // Use the surrogates from the previous iteration to construct
      // a new surrogate that represents larger and larger suffixes of
      // the input string.
      size_t shift = 2;
      std::vector<size_t> count(n + 2);
      std::vector<SKey> tmp(n);
      std::vector<SKey> surrogate_sorter(n);
      while (true) {
        // Update sort table.
        for (size_t i = 0; i < n; i++) {
          surrogate_sorter[i] = SKey {
              .start = surrogate[i],
              .next = (i + shift) < n ? surrogate[i + shift] : -1,
              .idx = i,
          };
        }

        // Radix sort O(n) - rolled out, 2 digits. The index in the third
        // element is not needed to be sorted. The radix sort algorithm
        // sorts two digits corresponding to the first and second element in
        // the triple. See for instance https://hacktechhub.com/radix-sort/ for
        // the general idea of radix sort. First, clear the counts.
        std::fill(count.begin(), count.begin() + v + 2, 0);
        // Next, count the frequency of each occurence.
        for (size_t i = 0; i < n; i++)
          count[surrogate_sorter[i].next + 1]++;
        // Update count to contain actual positions.
        for (size_t i = 1; i < v + 2; i++)
          count[i] += count[i - 1];
        // Construct output array based on second digit.
        for (int64_t i = n - 1; i >= 0; i--)
          tmp[(count[surrogate_sorter[i].next + 1]--) - 1] = surrogate_sorter[i];
        // Clear count. Next, sort on first digit.
        std::fill(count.begin(), count.begin() + v + 2, 0);
        // The source is in tmp. Count freq. on first digit.
        for (size_t i = 0; i < n; i++)
          count[tmp[i].start + 1]++;
        // Update count to contain actual positions.
        for (size_t i = 1; i < v + 2; i++)
          count[i] += count[i - 1];
        // Output to array w from tmp.
        for (int64_t i = n - 1; i >= 0; i--)
          surrogate_sorter[(count[tmp[i].start + 1]--) - 1] = tmp[i];

        v = 0;
        // Construct surrogate array. We have to do an extra case here
        // depending on whether this is the first iteration or not, as
        // the types are not the same.
        int64_t x0 = surrogate_sorter[0].start;
        int64_t x1 = surrogate_sorter[0].next;
        surrogate[surrogate_sorter[0].idx] = 0;
        for (size_t i = 1; i < n; i++) {
          if (x0 != surrogate_sorter[i].start || x1 != surrogate_sorter[i].next) v++;
          surrogate[surrogate_sorter[i].idx] = v;
          x0 = surrogate_sorter[i].start;
          x1 = surrogate_sorter[i].next;
        }

        // End if done.
        if (v >= n-1)
          break;
        shift *= 2;
      }
      // Reconstruct the suffix array.
      for (size_t i = 0; i < n; i++)
        sarray[i] = surrogate_sorter[i].idx;
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::compute_lcp(const Murmur3Hasher::Hash *str, 
        size_t n, const std::vector<size_t> &sarray,
        const std::vector<int64_t> &surrogate, std::vector<size_t> &lcp)
    //--------------------------------------------------------------------------
    {
      // Computes the LCP in O(n) time. This is Kasai's algorithm. See e.g.,
      // http://www.cs.cmu.edu/~15451-f20/LectureNotes/lec25-suffarray.pdf 
      // for an explanation. The original paper can be found here:
      // https://link.springer.com/chapter/10.1007/3-540-48194-X_17
      int k = 0;
      lcp.resize(n, 0);
      for(size_t i = 0; i < n; i++){
        if(surrogate[i] == int(n - 1))
          k = 0;
        else{
          size_t j = sarray[surrogate[i] + 1];
          for(; i + k < n && j + k < n && str[i + k] == str[j + k]; k++);
          lcp[surrogate[i]] = k;
          k = std::max(k - 1, 0);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::quick_matching_of_substrings(size_t min_length,
        const std::vector<size_t> &sarray, const std::vector<size_t> &lcp,
        std::vector<NonOverlappingRepeatsResult> &result)
    //--------------------------------------------------------------------------
    {
      // The function computes non-overlapping matching substrings in 
      // O(n log n) time. This is a new algorithm designed by David Broman 
      // in 2024 specifically for the Legion runtime. Please see the following
      // Git repo for a reference implementation and a short explanation:
      // https://github.com/david-broman/matching-substrings
      size_t le = sarray.size();
      using triple = std::tuple<size_t, size_t, size_t>;
      using pair = std::tuple<size_t, size_t>;

      // Construct tuple array O(n)
      std::vector<triple> a(le * 2 - 2);
      size_t k = 0;
      size_t m = 0;
      size_t pre_l = 0;
      for(size_t i = 0; i < le - 1; i++){
        size_t l1 = lcp[i];
        size_t s1 = sarray[i];
        size_t s2 = sarray[i + 1];
        if(s2 >= s1 + l1 || s2 <= s1 - l1){
          // Non-overlapping
          if(pre_l != l1)
            m += 1;
          a[k++] = std::make_tuple(le - l1, m, s1);
          a[k++] = std::make_tuple(le - l1, m, s2);
          pre_l = l1;
        }
        else if(s2 > s1 && s2 < s1 + l1){
          // Overlapping, increasing index
          size_t d = s2 - s1;
          size_t l3 = (((l1 + d) / 2) / d) * d;
          if(pre_l != l3)
            m += 1;
          a[k++] = std::make_tuple(le - l3, m, s1);
          a[k++] = std::make_tuple(le - l3, m, s1 + l3);
          pre_l = l3;
        }
        else if(s1 > s2 && s1 < s2 + l1){
          // Overlapping, decreasing index
          size_t d = s1 - s2;
          size_t l3 = (((l1 + d) / 2) / d) * d;
          if(pre_l != l3)
            m += 1;
          a[k++] = std::make_tuple(le - l3, m, s2);
          a[k++] = std::make_tuple(le - l3, m, s2 + l3);
          pre_l = l3;
        }
      }
      a.resize(k);

      // Sort tuple vector: O(n log n)
      std::sort(a.begin(), a.end());

      // Construct matching intervals: O(n)
      std::vector<bool> flag(le, false);
      std::vector<pair> r;
      size_t m_pre = 0;
      size_t next_k = 0;
      const size_t min_repeats = 2;
      for(size_t i = 0; i < a.size(); i++){
        int l = std::get<0>(a[i]);
        size_t m = std::get<1>(a[i]);
        size_t k = std::get<2>(a[i]);
        size_t le2 = le - l;
        if(m != m_pre){
          if(r.size() >= min_repeats){
            result.push_back(NonOverlappingRepeatsResult{
                .start = std::get<0>(r[0]),
                .end = std::get<1>(r[0]),
                .repeats = r.size()});
            for(const pair &p : r)
              for(size_t j = std::get<0>(p); j < std::get<1>(p); j++)
                flag[j] = true;
          }
          r.clear();
          next_k = 0;
        }
        m_pre = m;
        if(le2 != 0 && le2 >= min_length && k >= next_k &&
           !(flag[k]) && !(flag[k + le2 - 1])){
          r.push_back(std::make_tuple(k, k + le2));
          next_k = k + le2;
        }
      }
      if(r.size() >= min_repeats){
        result.push_back(NonOverlappingRepeatsResult{
            .start = std::get<0>(r[0]),
            .end = std::get<1>(r[0]),
            .repeats = r.size()});
      }
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::compute_longest_nonoverlapping_repeats(
                                                      FindRepeatsResult &repeat)
    //--------------------------------------------------------------------------
    {
      if (repeat.size < 2)
        return;
      std::vector<size_t> sarray(repeat.size);
      std::vector<int64_t> surrogate(repeat.size);
      compute_suffix_array(repeat.start, repeat.size, sarray, surrogate);
      std::vector<size_t> lcp;
      compute_lcp(repeat.start, repeat.size, sarray, surrogate, lcp);
      quick_matching_of_substrings(min_trace_length, sarray, lcp,repeat.result);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TraceRecognizer::find_repeats(const void *args)
    //--------------------------------------------------------------------------
    {
      const FindRepeatsTaskArgs *fargs = (const FindRepeatsTaskArgs*)args;
      fargs->recognizer->compute_longest_nonoverlapping_repeats(*fargs->result);
    }

    /////////////////////////////////////////////////////////////
    // Auto Tracing
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T>
    bool AutoTracing<T>::add_to_dependence_queue(Operation* op,
        const std::vector<StaticDependence>* dependences,
        bool unordered, bool outermost)
    //--------------------------------------------------------------------------
    {
      // If we're unordered or inside an explicit trace then pass through
      // Note that we might set outermost to false if we're being flushed
      // from the trace cache so set it back to true for the context
      if (unordered || (this->current_trace != NULL) || !outermost)
        return T::add_to_dependence_queue(
            op, dependences, unordered, true/*outermost*/);
      else if (op->record_trace_hash(this->recognizer, this->opidx))
      {
        this->opidx++;
        return true;
      }
      else
      {
        // Increment the current trace blocking index so we know
        // when we need to flush operations under blocking calls
        this->current_trace_blocking_index = this->next_blocking_index;
        return T::add_to_dependence_queue(op, dependences);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void AutoTracing<T>::record_blocking_call(uint64_t blocking_index,
                                              bool invalidate_trace)
    //--------------------------------------------------------------------------
    {
      // Check to see if the blocking operation happens for any operation
      // that occurs inside of the range of operations that we are buffering
      if ((blocking_index != InnerContext::NO_BLOCKING_INDEX) &&
          (this->current_trace == NULL) &&
          (this->current_trace_blocking_index <= blocking_index))
      {
        // Handling waits from the application is very similar
        // to the case in add_to_dependence_queue when we encounter an
        // operation that is not traceable. We interrupt traces in
        // the identifier, and flush the watcher and replayer. We identify
        // whether a wait is coming from the application by seeing if the
        // future being waited on has a valid coordinate.
        this->recognizer.record_operation_untraceable(this->opidx);
        this->current_trace_blocking_index = this->next_blocking_index;
      }
      // Need to also do whatever the base context was going to do.
      T::record_blocking_call(blocking_index, invalidate_trace);
    }

    template class AutoTracing<InnerContext>;
    template class AutoTracing<ReplicateContext>;  

  };
};
