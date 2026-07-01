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

#ifndef __LEGION_OCCURRENCE_WATCHER_H__
#define __LEGION_OCCURRENCE_WATCHER_H__

#include "legion/tracing/cache.h"

namespace Legion {
  namespace Internal {

    /**
     * \class OccurrenceWatcher
     * The occurrence watcher class maintains a trie of hashes corresponding
     * to candidate traces. These are sequences of hashes that we've observed
     * some number of times. Once these sequences are observed a certain
     * number of times determined by visit_threshold then these candidates
     * are promoted up to the full traces in the trace cache and are
     * eligible for replay. Note that each new hash can be the start of a
     * new trace so we maintain a group of active_pointers corresponding to
     * traces that are still matching in the trie.
     */
    class OccurrenceWatcher {
    public:
      OccurrenceWatcher(
          InnerContext* context, const Mapper::ContextConfigOutput& config);
    public:
      bool record_operation(
          Operation* op, Murmur3Hasher::Hash hash, uint64_t opidx);
      bool record_noop(Operation* op);
      void flush(uint64_t opidx);
      void insert(
          const Murmur3Hasher::Hash* hashes, size_t size, uint64_t opidx);
      TrieQueryResult query(
          const Murmur3Hasher::Hash* hashes, size_t size) const;
    private:
      TraceCache cache;
    private:
      struct TraceCandidate {
        // Needs to be default constructable.
        TraceCandidate() : opidx(0) { }
        TraceCandidate(uint64_t opidx_) : opidx(opidx_) { }
        // The opidx that this trace was inserted at.
        uint64_t opidx;
        // The occurrence watcher will only maintain the number
        // of visits. I don't think that we need to do decaying visits
        // here, though we might want to lower the amount of traces that
        // get committed to the replayer.
        uint64_t visits = 0;
        // completed marks whether this trace has moved
        // from the "watched" state to the "committed" state.
        // Once a trace has been completed, it will not be
        // returned from complete() anymore.
        bool completed = false;
        // The opidx that this trace was previously visited at.
        uint64_t previous_visited_opidx = 0;
      };
      Trie<Murmur3Hasher::Hash, TraceCandidate> trie;
      const uint64_t visit_threshold;
    private:
      // TriePointer maintains an active trace being
      // traversed in the watcher's trie.
      class TriePointer {
      public:
        TriePointer(
            TrieNode<Murmur3Hasher::Hash, TraceCandidate>* node_,
            uint64_t opidx_)
          : node(node_), opidx(opidx_), depth(0)
        { }
        bool advance(Murmur3Hasher::Hash token);
        bool complete(void) const;
      public:
        TrieNode<Murmur3Hasher::Hash, TraceCandidate>* node;
        uint64_t opidx;
        uint64_t depth;
      };
      // All currently active pointers that need advancing.
      std::vector<TriePointer> active_pointers;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_OCCURRENCE_WATCHER_H__
