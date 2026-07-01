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

#include "legion/tracing/watcher.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Occurrence Watcher
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OccurrenceWatcher::OccurrenceWatcher(
        InnerContext* context, const Mapper::ContextConfigOutput& config)
      : cache(context), visit_threshold(config.auto_tracing_visit_threshold)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool OccurrenceWatcher::record_operation(
        Operation* op, Murmur3Hasher::Hash hash, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      // Every new hash chould be the start of a new trace
      active_pointers.emplace_back(trie.get_root(), opidx);
      // We'll copy pointers down if they can't be advanced as we iterate
      unsigned current_index = 0;
      for (unsigned idx = 0; idx < active_pointers.size(); idx++)
      {
        TriePointer& pointer = active_pointers[idx];
        if (!pointer.advance(hash))
          continue;
        active_pointers[current_index++] = pointer;
        // See if we found a trace
        if (pointer.complete())
        {
          // If this pointer corresponds to a completed trace, then we have
          // some work to do. First, increment the number of visits on the node.
          TrieNode<Murmur3Hasher::Hash, TraceCandidate>* node = pointer.node;
          TraceCandidate& candidate = node->get_value();
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
              legion_assert(node != nullptr);
              hashes[pointer.depth - j - 1] = node->get_token();
              node = node->get_parent();
            }
            // Check to see if this is a prefix of an existing trace
            // TODO (rohany): Do we need to think about superstrings here?
            if (!cache.has_prefix(hashes))
            {
              log_auto_trace.debug() << "Committing trace: " << candidate.opidx
                                     << " of length: " << pointer.depth;
              cache.insert(hashes, opidx);
            }
          }
        }
      }
      // At this point we can shrink down the vector to the remaining size
      active_pointers.erase(
          active_pointers.begin() + current_index, active_pointers.end());
      // Now tell the trace cache to reocrd the operation too
      return cache.record_operation(op, hash, opidx);
    }

    //--------------------------------------------------------------------------
    bool OccurrenceWatcher::TriePointer::advance(Murmur3Hasher::Hash token)
    //--------------------------------------------------------------------------
    {
      node = node->find_child(token);
      // We couldn't advance the pointer. Importantly, we can't check
      // node.end here, as this node could be the prefix of another
      // trace in the trie
      if (node == nullptr)
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
    bool OccurrenceWatcher::record_noop(Operation* op)
    //--------------------------------------------------------------------------
    {
      return cache.record_noop(op);
    }

    //--------------------------------------------------------------------------
    void OccurrenceWatcher::flush(uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      active_pointers.clear();
      cache.flush(opidx);
    }

    //--------------------------------------------------------------------------
    void OccurrenceWatcher::insert(
        const Murmur3Hasher::Hash* hashes, size_t size, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      trie.insert(hashes, size, TraceCandidate(opidx));
    }

    //--------------------------------------------------------------------------
    TrieQueryResult OccurrenceWatcher::query(
        const Murmur3Hasher::Hash* hashes, size_t size) const
    //--------------------------------------------------------------------------
    {
      return trie.query(hashes, size);
    }

  }  // namespace Internal
}  // namespace Legion
