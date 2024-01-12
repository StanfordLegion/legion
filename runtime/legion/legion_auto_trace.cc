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
      const AutoTraceProcessRepeatsArgs* args = (const AutoTraceProcessRepeatsArgs*)args_;
      // TODO (rohany): Make this a command line parameter.
      std::vector<NonOverlappingRepeatsResult> result = compute_longest_nonoverlapping_repeats(*args->operations, 5);
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
      std::cout << "inside my new meta task!" << std::endl;
    }

    BatchedTraceIdentifier::BatchedTraceIdentifier(
        Runtime* runtime_,
        TraceOccurrenceWatcher* watcher_,
        size_t batchsize_,
        size_t max_add_
        )
        : runtime(runtime_), batchsize(batchsize_), watcher(watcher_), max_add(max_add_) {
      // Reserve one extra place so that we can insert the sentinel
      // character at the end of the string.
      this->hashes.reserve(batchsize_ + 1);
    }

    void BatchedTraceIdentifier::process(Murmur3Hasher::Hash hash, size_t opidx) {
      this->hashes.push_back(hash);
      if (this->hashes.size() == this->batchsize) {
        // TODO (rohany): Define this sentinel somewhere else.
        // Insert the sentinel token before sending the string off to the meta task.
        this->hashes.push_back(Murmur3Hasher::Hash{0, 0});
        std::vector<NonOverlappingRepeatsResult> result;
        AutoTraceProcessRepeatsArgs args(&this->hashes, &result);
        // TODO (rohany): What should the priority be for this?
        // TODO (rohany): Make sure that we don't have too many of these
        //  meta tasks pending at once (should have a fixed amount etc).
        // TODO (rohany): We'll wait on this result for now.
        runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY).wait();

        // Insert the received traces into the occurrence watcher.
        size_t count = 0;
        for (auto trace : result) {
          // TODO (rohany): Deal with the maximum number of traces permitted in the trie.
          // TODO (rohany): Do we need to consider superstrings here?
          auto start = this->hashes.begin() + trace.start;
          auto end = this->hashes.begin() + trace.end;
          if (!this->watcher->prefix(start, end)) {
            this->watcher->insert(start, end, opidx);
            count++;
            // Only insert max_add traces at a time.
            if (count == this->max_add) {
              break;
            }
          }
        }
        std::cout << "Inserted: " << count << " traces" << std::endl;
        // After we're done processing our trace, clear the memory so that we
        // can collect more traces.
        this->hashes.clear();
      }
    }

    TraceOccurrenceWatcher::TraceOccurrenceWatcher(size_t visit_threshold_) : visit_threshold(visit_threshold_) { }

    void TraceOccurrenceWatcher::process(Murmur3Hasher::Hash hash, size_t opidx) {
      this->active_pointers.push_back(TriePointer(this->trie.get_root(), opidx));
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
            std::cout << "PROMOTING TRACE: " << value.opidx << " " << pointer.depth << std::endl;
            value.completed = true;
            std::vector<Murmur3Hasher::Hash> trace(pointer.depth);
            for (size_t j = 0; j < pointer.depth; j++) {
              assert(node != nullptr);
              trace[pointer.depth - j - 1] = node->get_token();
              node = node->get_parent();
            }
            // TODO (rohany): Here, insert the trace into the TraceReplayWatcher.
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
  };
};
