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
#include "legion/legion_auto_trace.inl"
#include "legion/suffix_tree.h"

#include <iterator>
#include <unordered_set>

namespace Legion {
  namespace Internal {

    TraceHashHelper::TraceHashHelper() :
      // TODO (rohany): Does precise need to be true here?
      hasher(false /* precise */) {}

    Murmur3Hasher::Hash TraceHashHelper::hash(Operation* op) {
      Operation::OpKind kind = op->get_operation_kind();
      // First just hash the operation kind.
      hasher.hash(kind);
      switch (kind) {
        case Operation::OpKind::TASK_OP_KIND: {
          TaskOp* task = dynamic_cast<TaskOp*>(op);
          switch (task->get_task_kind()) {
            case TaskOp::TaskKind::INDIVIDUAL_TASK_KIND:
            case TaskOp::TaskKind::INDEX_TASK_KIND: {
              this->hash(task);
              break;
            }
            default: {
              std::cout << "Attempting to has unexpected TaskOp kind: "
                        << int(task->get_task_kind())
                        << std::endl;
              assert(false);
            }
          }
          break;
        }
        case Operation::OpKind::FILL_OP_KIND: {
          this->hash(dynamic_cast<FillOp*>(op));
          break;
        }
        case Operation::OpKind::FENCE_OP_KIND: {
          this->hash(dynamic_cast<FenceOp*>(op));
          break;
        }
        case Operation::OpKind::COPY_OP_KIND: {
          this->hash(dynamic_cast<CopyOp*>(op));
          break;
        }
        case Operation::OpKind::ALL_REDUCE_OP_KIND: {
          this->hash(dynamic_cast<AllReduceOp*>(op));
          break;
        }
        default: {
          std::cout << "Attempting to hash an unsupported operation kind:"
                    << std::string(Operation::get_string_rep(kind))
                    << std::endl;
          assert(false);
        }
      };
      return hasher.get_hash();
    }

    void TraceHashHelper::hash(TaskOp* op) {
      hasher.hash(op->task_id);
      // Choosing to skip the fields indexes, futures, grants, wait_barriers, arrive_barriers, ...
      for (std::vector<RegionRequirement>::const_iterator it = op->regions.begin();
           it != op->regions.end(); it++) {
        this->hash(*it);
      }
      assert(op->output_regions.size() == 0);
      hasher.hash<bool>(op->is_index_space);
      if (op->is_index_space) {
        hasher.hash<bool>(op->concurrent_task);
        hasher.hash<bool>(op->must_epoch_task);
        hasher.hash(op->index_domain);
      }
    }

    void TraceHashHelper::hash(FillOp* op) {
      this->hash(op->requirement);
      // Choosing to skip the fields grants, wait_barriers, arrive_barriers.
      hasher.hash<bool>(op->is_index_space);
      if (op->is_index_space) {
        hasher.hash(op->index_domain);
      }
    }

    void TraceHashHelper::hash(FenceOp* op) {
      hasher.hash(op->get_fence_kind());
    }

    void TraceHashHelper::hash(CopyOp* op) {
      for (std::vector<RegionRequirement>::const_iterator it = op->src_requirements.begin();
           it != op->src_requirements.end(); it++) {
        this->hash(*it);
      }
      for (std::vector<RegionRequirement>::const_iterator it = op->dst_requirements.begin();
           it != op->dst_requirements.end(); it++) {
        this->hash(*it);
      }
      for (std::vector<RegionRequirement>::const_iterator it = op->src_indirect_requirements.begin();
           it != op->src_indirect_requirements.end(); it++) {
        this->hash(*it);
      }
      for (std::vector<RegionRequirement>::const_iterator it = op->dst_indirect_requirements.begin();
           it != op->dst_indirect_requirements.end(); it++) {
        this->hash(*it);
      }
      // Not including the fields grants, wait_barriers, arrive_barriers.
      hasher.hash<bool>(op->is_index_space);
      if (op->is_index_space) {
        hasher.hash(op->index_domain);
      }
    }

    void TraceHashHelper::hash(AllReduceOp* op) {
      // TODO (rohany): I'm not sure if anything else needs to
      //  go into the hash other than just the operation kind.
    }

    void TraceHashHelper::hash(const RegionRequirement& req) {
      if (req.region.exists()) {
        hasher.hash<bool>(true); // is_reg
        this->hash(req.region);
      } else {
        hasher.hash<bool>(false); // is_reg
        hasher.hash(req.partition.get_index_partition().get_id());
        hasher.hash(req.partition.get_field_space().get_id());
        hasher.hash(req.partition.get_tree_id());
      }
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
           it != req.privilege_fields.end(); it++) {
        hasher.hash(*it);
      }
      for (std::vector<FieldID>::const_iterator it = req.instance_fields.begin();
           it != req.instance_fields.end(); it++) {
        hasher.hash(*it);
      }
      hasher.hash(req.privilege);
      hasher.hash(req.prop);
      this->hash(req.parent);
      hasher.hash(req.redop);
      // Excluding the fields: tag and flags.
      hasher.hash(req.handle_type);
      hasher.hash(req.projection);
    }

    void TraceHashHelper::hash(const LogicalRegion& region) {
      hasher.hash(region.get_index_space().get_id());
      hasher.hash(region.get_field_space().get_id());
      hasher.hash(region.get_tree_id());
    }

    bool is_operation_traceable(Operation* op) {
      MemoizableOp* memo = op->get_memoizable();
      if (memo == nullptr) {
        return false;
      }
      switch (memo->get_operation_kind()) {
        case Operation::OpKind::TASK_OP_KIND: {
          TaskOp* task = dynamic_cast<TaskOp*>(memo);
          // Tasks with output regions cannot be traced.
          if (task->output_regions.size() > 0) {
            return false;
          }
          return true;
          break;
        }
        default:
          return true;
      }
    }

    void auto_trace_process_repeats(const void* args_) {
      log_auto_trace.info() << "Executing processing repeats meta task.";
      const AutoTraceProcessRepeatsArgs* args = (const AutoTraceProcessRepeatsArgs*)args_;
      std::vector<NonOverlappingRepeatsResult> result =
          compute_longest_nonoverlapping_repeats(*args->operations, args->min_trace_length);
      // Filter the result to remove substrings of longer repeats from consideration.
      size_t copyidx = 0;
      std::unordered_set<size_t> ends;
      for (auto res : result) {
        if (ends.find(res.end) != ends.end()) {
          continue;
        }
        result[copyidx] = res;
        copyidx++;
        ends.insert(res.end);
      }
      // Erase the unused pieces of result.
      result.erase(result.begin() + copyidx, result.end());
      *args->result = std::move(result);
    }

    BatchedTraceIdentifier::BatchedTraceIdentifier(
        Runtime* runtime_,
        TraceOccurrenceWatcher& watcher_,
        size_t batchsize_,
        size_t max_add_,
        size_t max_inflight_requests_,
        bool wait_on_async_job_,
        size_t min_trace_length_
        )
        : runtime(runtime_),
          watcher(watcher_),
          batchsize(batchsize_),
          max_add(max_add_),
          min_trace_length(min_trace_length_),
          jobs_comp_queue(CompletionQueue::create_completion_queue(max_inflight_requests_)),
          max_in_flight_requests(max_inflight_requests_),
          wait_on_async_job(wait_on_async_job_) {
      // Reserve one extra place so that we can insert the sentinel
      // character at the end of the string.
      this->hashes.reserve(this->batchsize + 1);
    }

    void BatchedTraceIdentifier::process(Murmur3Hasher::Hash hash, size_t opidx) {
      this->hashes.push_back(hash);
      if (this->hashes.size() == this->batchsize) {
        // TODO (rohany): Define this sentinel somewhere else.
        // Insert the sentinel token before sending the string off to the meta task.
        this->hashes.push_back(Murmur3Hasher::Hash{0, 0});
        // Initialize a descriptor for the pending result.
        this->jobs_in_flight.push_back(InFlightProcessingRequest{});
        InFlightProcessingRequest& request = this->jobs_in_flight.back();
        // Move the existing vector of hashes into the descriptor, and
        // allocate a result space for the meta task to write into.
        request.hashes = std::move(this->hashes);
        request.result = new std::vector<NonOverlappingRepeatsResult>();
        AutoTraceProcessRepeatsArgs args(&request.hashes, request.result, this->min_trace_length);
        // Launch the meta task and record the finish event.
        request.finish_event = runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY);
        this->jobs_comp_queue.add_event(request.finish_event);
        // If we're configured to process the job sequentially, then wait on it
        // right after the submission.
        if (this->wait_on_async_job) {
          request.finish_event.wait();
        }
        // Finally, allocate a new vector to accumulate hashes into. As before,
        // allocate one extra spot for the sentinel hash.
        this->hashes = std::vector<Murmur3Hasher::Hash>();
        this->hashes.reserve(this->batchsize + 1);
      }

      // If we've hit the maximum number of inflight requests, then wait
      // on one of them to make sure that it gets processed.
      if (this->jobs_in_flight.size() == this->max_in_flight_requests) {
        this->jobs_in_flight.front().finish_event.wait();
      }

      // Query the completion queue to see if any of our jobs finished.
      size_t num_completed = this->jobs_comp_queue.pop_events(&this->finish_event_alloc, 1);
      assert(num_completed == 0 || num_completed == 1);
      // Mark the corresponding in flight descriptor as completed.
      if (num_completed == 1) {
        for (auto& it : this->jobs_in_flight) {
          if (it.finish_event == this->finish_event_alloc) {
            it.completed = true;
          }
        }
        // We'll process at most one of the completed jobs so that
        // we don't pile up a bunch of work all at once.
        auto& finished = this->jobs_in_flight.front();
        if (finished.completed) {
          // Insert the received traces into the occurrence watcher.
          size_t count = 0;
          for (auto trace : *finished.result) {
            // TODO (rohany): Deal with the maximum number of traces permitted in the trie.
            // TODO (rohany): Do we need to consider superstrings here?
            auto start = finished.hashes.begin() + trace.start;
            auto end = finished.hashes.begin() + trace.end;
            if (!this->watcher.prefix(start, end)) {
              this->watcher.insert(start, end, opidx);
              count++;
              // Only insert max_add traces at a time.
              if (count == this->max_add) {
                break;
              }
            }
          }
          // Free the memory allocated for this descriptor.
          delete finished.result;
          this->jobs_in_flight.pop_front();
        }
      }
    }

    TraceOccurrenceWatcher::TraceOccurrenceWatcher(
        TraceReplayer& replayer_, size_t visit_threshold_)
      : replayer(replayer_), visit_threshold(visit_threshold_) { }

    void TraceOccurrenceWatcher::process(Murmur3Hasher::Hash hash, size_t opidx) {
      this->active_pointers.emplace_back(this->trie.get_root(), opidx);
      // We'll avoid any allocations here by copying in pointers
      // as we process them.
      size_t copyidx = 0;
      for (size_t i = 0; i < this->active_pointers.size(); i++) {
        // Try to advance the pointer.
        TriePointer pointer = this->active_pointers[i];
        if (!pointer.advance(hash)) {
          continue;
        }
        this->active_pointers[copyidx] = pointer;
        copyidx++;
        // Check to see if the pointer completed.
        if (pointer.complete()) {
          // If this pointer corresponds to a completed trace, then we have
          // some work to do. First, increment the number of visits on the node.
          TrieNode<Murmur3Hasher::Hash, TraceMeta>* node = pointer.node;
          auto& value = node->get_value();
          value.visits++;
          if (value.visits >= this->visit_threshold && !value.completed) {
            value.completed = true;
            std::vector<Murmur3Hasher::Hash> trace(pointer.depth);
            for (size_t j = 0; j < pointer.depth; j++) {
              assert(node != nullptr);
              trace[pointer.depth - j - 1] = node->get_token();
              node = node->get_parent();
            }
            // TODO (rohany): Do we need to think about superstrings here?
            if (!this->replayer.prefix(trace.begin(), trace.end())) {
              log_auto_trace.debug() << "Committing trace: "
                                     << value.opidx  << " of length: "
                                     << pointer.depth;
              this->replayer.insert(trace.begin(), trace.end(), opidx);
            }
          }
        }
      }
      // Erase the remaining elements from active_pointers.
      this->active_pointers.erase(this->active_pointers.begin() + copyidx, this->active_pointers.end());
    }

    bool TraceOccurrenceWatcher::TriePointer::advance(Murmur3Hasher::Hash token) {
      // We couldn't advance the pointer. Importantly, we can't check
      // node.end here, as this node could be the prefix of another
      // trace in the trie.
      auto it = this->node->get_children().find(token);
      if (it == this->node->get_children().end()) {
        return false;
      }
      // Otherwise, move down to the node's child.
      this->node = it->second;
      this->depth++;
      return true;
    }

    bool TraceOccurrenceWatcher::TriePointer::complete() {
      return this->node->get_end() && this->opidx >= this->node->get_value().opidx;
    }

    // Have to also provide declarations of the static variables.
    constexpr double TraceReplayer::TraceMeta::R;
    constexpr double TraceReplayer::TraceMeta::SCORE_CAP_MULT;
    constexpr size_t TraceReplayer::TraceMeta::REPLAY_SCALE;

    void TraceReplayer::TraceMeta::visit(size_t opidx) {
      // First, compute the difference in trace lengths that
      // this trace was last visited at.
      size_t previous_visit = this->last_visited_opidx;
      size_t diff_in_traces = (opidx - previous_visit) / this->length;
      // Compute the new visit count by decaying the old visit count
      // and then adding one.
      this->decaying_visits = (pow(R, diff_in_traces) * this->decaying_visits) + 1;
    }

    double TraceReplayer::TraceMeta::score(size_t opidx) const {
      // Do a similar calculation as visit, where we decay the score
      // of the trace as if reading from opidx.
      // TODO (rohany): I'm not entirely convinced that this is necessary,
      //  because if we're computing the score of something, then it should
      //  have just been visited, i.e. it's score was just updated. However,
      //  this worked well in the simulator so I'll start with it before
      //  switching it up.
      size_t previous_visit = this->last_visited_opidx;
      size_t diff_in_traces = (opidx - previous_visit) / this->length;
      // Increase the visit count by 1 when computing the score so that
      // traces that haven't been visited before don't have a 0 score.
      // The initial score is num_visits * length.
      double score = ((pow(R, diff_in_traces) * this->decaying_visits) + 1) * this->length;
      // Next, we cap the score so that the first trace that gets replayed
      // doesn't get replayed forever.
      score = std::min(score, SCORE_CAP_MULT * ((double)this->length));
      // Then, increase the score a little bit if a trace has already been
      // replayed to favor replays.
      size_t capped_replays = std::max(std::min(REPLAY_SCALE, this->replays), (size_t)1);
      return score * (double)capped_replays;
    }

    void TraceReplayer::flush_buffer() {
      size_t num_ops = this->operations.size();
      this->operation_start_idx += num_ops;
      while (!this->operations.empty()) {
        this->executor->issue_operation(this->operations.front());
        this->operations.pop();
      }
    }

    void TraceReplayer::flush_buffer(size_t opidx) {
      // If we've already advanced beyond this point, then there's nothing to do.
      if (this->operation_start_idx > opidx) {
        return;
      }
      size_t difference = opidx - this->operation_start_idx;
      this->operation_start_idx += difference;
      for (size_t i = 0; i < difference; i++) {
        this->executor->issue_operation(this->operations.front());
        this->operations.pop();
      }
    }

    void TraceReplayer::replay_trace(size_t opidx, Legion::TraceID tid) {
      // If we've already advanced beyond this point, then there's nothing to do.
      // TODO (rohany): I don't think that this should happen when we're actually
      //  calling replay, but better safe than sorry.
      if (this->operation_start_idx > opidx) {
        return;
      }
      log_auto_trace.debug() << "Replaying trace " << tid
                             << " of length "
                             << (opidx - this->operation_start_idx);
      // Similar logic as flush_buffer, but issue a begin and end trace
      // around the flushed operations.
      this->executor->issue_begin_trace(tid);
      size_t difference = opidx - this->operation_start_idx;
      this->operation_start_idx += difference;
      for (size_t i = 0; i < difference; i++) {
        this->executor->issue_operation(this->operations.front());
        this->operations.pop();
      }
      this->executor->issue_end_trace(tid);
    }

    bool TraceReplayer::WatchPointer::advance(Murmur3Hasher::Hash token) {
      // We couldn't advance the pointer. Importantly, we can't check
      // node.end here, as this node could be the prefix of another
      // trace in the trie.
      auto it = this->node->get_children().find(token);
      if (it == this->node->get_children().end()) {
        return false;
      }
      // Otherwise, move down to the node's child.
      this->node = it->second;

      // If we've hit the end of a string, mark it as visited and update the score.
      if (this->node->get_end() && this->opidx >= this->node->get_value().opidx) {
        this->node->get_value().visit(opidx);
      }
      return true;
    }

    bool TraceReplayer::CommitPointer::advance(Murmur3Hasher::Hash token) {
      // We couldn't advance the pointer. Importantly, we can't check
      // node.end here, as this node could be the prefix of another
      // trace in the trie.
      auto it = this->node->get_children().find(token);
      if (it == this->node->get_children().end()) {
        return false;
      }
      // Otherwise, move down to the node's child.
      this->node = it->second;
      return true;
    }

    bool TraceReplayer::CommitPointer::complete() {
      return this->node->get_end() && this->opidx >= this->node->get_value().opidx;
    }

    TraceID TraceReplayer::CommitPointer::replay(OperationExecutor* exec) {
      auto& value = this->node->get_value();
      TraceID tid;
      if (value.replays == 0) {
        // If we haven't replayed this trace before, generate a fresh id.
        tid = exec->get_fresh_trace_id();
        value.tid = tid;
      } else {
        tid = value.tid;
      }
      value.replays++;
      return tid;
    }

    double TraceReplayer::CommitPointer::score(size_t opidx) {
      return this->node->get_value().score(opidx);
    }

    void TraceReplayer::flush(size_t opidx) {
      // We have to flush all active pointers from the trie, and
      // them attempt to launch traces for any completed pointers.
      // First, clear the vectors of active pointers.
      this->active_watching_pointers.clear();
      this->active_commit_pointers.clear();
      // If we have no completed pointers, flush all pending operations and
      // early exit.
      if (this->completed_commit_pointers.size() == 0) {
        this->flush_buffer();
        return;
      }

      // Now that we have some completed pointers, sort and issue them.
      std::sort(
          this->completed_commit_pointers.begin(),
          this->completed_commit_pointers.end(),
          [opidx](CommitPointer& left, CommitPointer& right) {
            // We make these sort keys (score, -opidx) so that the highest
            // scoring, earliest opidx is the first entry in the ordering.
            auto ltup = std::make_pair(left.score(opidx), -(int64_t)left.get_opidx());
            auto rtup = std::make_pair(right.score(opidx), -(int64_t)right.get_opidx());
            // Use > instead of < so that we get descending order.
            return ltup > rtup;
          }
      );
      for (auto& pointer : this->completed_commit_pointers) {
        // If we're considering a pointer that starts earlier than the
        // pending set of operations, then that trace is behind us. So
        // we just continue onto the next trace.
        if (pointer.get_opidx() < this->operation_start_idx) {
          continue;
        }
        // Now, flush the buffer up until the start of this trace.
        this->flush_buffer(pointer.get_opidx());
        // Finally, we can issue the trace.
        TraceID tid = pointer.replay(this->executor);
        this->replay_trace(pointer.get_opidx() + pointer.get_length(), tid);
      }
      this->completed_commit_pointers.clear();
      // Flush all remaining operations.
      this->flush_buffer();
    }

    void TraceReplayer::process(Legion::Internal::Operation *op, Murmur3Hasher::Hash hash, size_t opidx) {
      this->operations.push(op);

      // Update all watching pointers. This is very similar to the advancing
      // of pointers in the TraceOccurrenceWatcher.
      this->active_watching_pointers.emplace_back(this->trie.get_root(), opidx);
      // Avoid a reallocation in the same way by copying in place.
      size_t copyidx = 0;
      for (size_t i = 0; i < this->active_watching_pointers.size(); i++) {
        WatchPointer pointer = this->active_watching_pointers[i];
        if (!pointer.advance(hash)) {
          continue;
        }
        this->active_watching_pointers[copyidx] = pointer;
        copyidx++;
      }
      // Erase the remaining unused elements.
      this->active_watching_pointers.erase(this->active_watching_pointers.begin() + copyidx, this->active_watching_pointers.end());

      // Now update all the commit pointers. This process is more tricky, as
      // we have to actually decide to flush operations through the dependence
      // queue as we match traces. Additionally, we have to manage heuristics
      // around which traces we should take.
      this->active_commit_pointers.emplace_back(this->trie.get_root(), opidx);
      copyidx = 0;
      for (size_t i = 0; i < this->active_commit_pointers.size(); i++) {
        CommitPointer pointer = this->active_commit_pointers[i];
        if (!pointer.advance(hash)) {
          continue;
        }
        // TODO (rohany): Right now, we can't handle strings that have prefixes
        //  in the trie in this watcher. However, if needed, we could have this
        //  complete function return a completed pointer that corresponds to
        //  the finished prefix, and then also add current pointer back to
        //  the active pointers so then the heuristics we have work out nicely.
        if (pointer.complete()) {
          this->completed_commit_pointers.push_back(pointer);
        } else {
          this->active_commit_pointers[copyidx] = pointer;
          copyidx++;
        }
      }
      // Erase the rest of the active pointers.
      this->active_commit_pointers.erase(this->active_commit_pointers.begin() + copyidx, this->active_commit_pointers.end());

      // Find the minimum opidx of the active and completed pointers.
      size_t earliest_active = std::numeric_limits<size_t>::max();
      size_t earliest_completed = std::numeric_limits<size_t>::max();
      for (auto& pointer : this->active_commit_pointers) {
        earliest_active = std::min(earliest_active, pointer.get_opidx());
      }
      for (auto& pointer : this->completed_commit_pointers) {
        earliest_completed = std::min(earliest_completed, pointer.get_opidx());
      }
      size_t earliest_opidx = std::min(earliest_active, earliest_completed);

      // First, flush all operations until the earliest_opidx, as there is
      // nothing we are considering before there. If there are no active
      // or completed operations at all, then we just flush the entire buffer.
      if (this->active_commit_pointers.empty() && this->completed_commit_pointers.empty()) {
        this->flush_buffer();
      } else {
        this->flush_buffer(earliest_opidx);
      }

      // Now, we have to case on what combination of pointers are available.
      if (this->active_commit_pointers.empty() && this->completed_commit_pointers.empty()) {
        // There's nothing to do if we don't have any of either pointer left.
        return;
      } else if (this->active_commit_pointers.empty()) {
        // In this case, there are only completed pointers. We'll try to flush
        // through as many operations as we can. The heuristic is to take
        // traces ordered by score. This hueuristic can lead to suboptimal
        // trace replay, for example if the operation stream is AB and we have
        // completed traces [A, B] but B has a higher score, we'll issue A without
        // actually replaying that trace. Doing this seems to require some
        // fancier logic with interval trees or something, so we'll stick to
        // the simpler piece. First, sort the completed pointers by their score
        // so that we have the highest scoring pointers at the front.
        std::sort(
          this->completed_commit_pointers.begin(),
          this->completed_commit_pointers.end(),
          [opidx](CommitPointer& left, CommitPointer& right) {
            // We make these sort keys (score, -opidx) so that the highest
            // scoring, earliest opidx is the first entry in the ordering.
            auto ltup = std::make_pair(left.score(opidx), -(int64_t)left.get_opidx());
            auto rtup = std::make_pair(right.score(opidx), -(int64_t)right.get_opidx());
            // Use > instead of < so that we get descending order.
            return ltup > rtup;
          }
        );
        for (auto& pointer : this->completed_commit_pointers) {
          // If we're considering a pointer that starts earlier than the
          // pending set of operations, then that trace is behind us. So
          // we just continue onto the next trace.
          if (pointer.get_opidx() < this->operation_start_idx) {
            continue;
          }
          // Now, flush the buffer up until the start of this trace.
          this->flush_buffer(pointer.get_opidx());
          // Finally, we can issue the trace.
          TraceID tid = pointer.replay(this->executor);
          this->replay_trace(pointer.get_opidx() + pointer.get_length(), tid);
        }
        // The set of completed pointers is now empty.
        this->completed_commit_pointers.clear();
        // At this point, we don't have any completed or active pointers,
        // so flush any remaining operations.
        this->flush_buffer();
      } else if (this->completed_commit_pointers.empty()) {
        // There are no completed pointers, so there's nothing left to do,
        // as we already flushed all possible pending operations.
        return;
      } else {
        // In this case, we have both completed and active pointers. What we actually
        // do will change depending on what the overlaps between our completed and
        // active pointers actually are.
        if (earliest_completed < earliest_active) {
          // In this case, we have some completed pointers that we could potentially
          // replay behind our active pointers. We're going to take a heuristic here
          // where we only flush completed pointers that do not overlap with any
          // active pointers. This biases us towards longer traces when possible,
          // and makes the replay resilient against different kinds of traces
          // being inserted, such as AB, when the trie already contains BC.
          // Sort the completed pointers again.
          // TODO (rohany): There's a micro-optimization here to avoid re-sorting
          //  the completed pointers so much, we can maintain a bit that holds onto
          //  whether the vector has changed since the last time we sorted it (i.e.
          //  we actually inserted a new pointer into the list).
          std::sort(
            this->completed_commit_pointers.begin(),
            this->completed_commit_pointers.end(),
            [opidx](CommitPointer& left, CommitPointer& right) {
              // We make these sort keys (score, -opidx) so that the highest
              // scoring, earliest opidx is the first entry in the ordering.
              auto ltup = std::make_pair(left.score(opidx), -(int64_t)left.get_opidx());
              auto rtup = std::make_pair(right.score(opidx), -(int64_t)right.get_opidx());
              // Use > instead of < so that we get descending order.
              return ltup > rtup;
            }
          );
          size_t cutoff_opidx = earliest_active;
          size_t pending_completion_cutoff = std::numeric_limits<size_t>::max();
          size_t copyidx = 0;
          for (size_t i = 0; i < this->completed_commit_pointers.size(); i++) {
            auto pointer = this->completed_commit_pointers[i];
            // If this completed pointer spans into an active pointer, then we need
            // to save it for later.
            // TODO (rohany): Not sure if this is supposed to be > or >=.
            if (pointer.get_opidx() + pointer.get_length() >= cutoff_opidx) {
              this->completed_commit_pointers[copyidx] = pointer;
              copyidx++;
              // If we decide to skip this completed pointer because it overlaps
              // with an active pointer, we shouldn't replay any completed pointers
              // that overlap with this good pointer, as it scored higher.
              pending_completion_cutoff = std::min(pending_completion_cutoff, pointer.get_opidx());
              continue;
            }
            // As before, any pointers that we are already past can be ignored.
            if (pointer.get_opidx() < this->operation_start_idx) {
              continue;
            }
            // Lastly, make sure that this completed pointer doesn't invalidate
            // a completed pointer that was re-queued with a better score.
            // TODO (rohany): Not sure if this should be > or >=.
            if (pointer.get_opidx() + pointer.get_length() >= pending_completion_cutoff) {
              this->completed_commit_pointers[copyidx] = pointer;
              copyidx++;
              continue;
            }
            // Here, we can finally replay the trace.
            this->flush_buffer(pointer.get_opidx());
            TraceID tid = pointer.replay(this->executor);
            this->replay_trace(pointer.get_opidx() + pointer.get_length(), tid);
          }
          // Clear the remaining invalid completions.
          this->completed_commit_pointers.erase(this->completed_commit_pointers.begin() + copyidx, this->completed_commit_pointers.end());

#ifdef DEBUG_LEGION
          // Since we waited to not cut off any active pointers, there should not
          // be any invalid active pointers.
          for (auto& pointer : this->active_commit_pointers) {
            assert(pointer.get_opidx() >= this->operation_start_idx);
          }
#endif
        } else if (earliest_completed > earliest_active) {
          // There are active pointers behind our earliest completed
          // pointer, so there's no point at even looking at the completed
          // pointers.
          return;
        } else {
          // We should never be in the case where an active and completed pointer
          // are starting at the same opidx.
          // TODO (rohany): This can actually happen if we allow for prefixes like
          //  the comment earlier above suggests. We'll deal with that when we
          //  get there.
          assert(false);
        }
      }
    }

  };
};
