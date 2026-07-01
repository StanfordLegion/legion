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

#ifndef __LEGION_TRACE_CACHE_H__
#define __LEGION_TRACE_CACHE_H__

#include "legion/tracing/trie.h"
#include "legion/contexts/inner.h"

namespace Legion {
  namespace Internal {

    /**
     * \class TraceCache
     * The trace cache maintains a trie corresponding to traces that have
     * been observed the minimum number of times required for replay at
     * which point once we see them we can start to replay them.
     */
    class TraceCache {
    public:
      TraceCache(InnerContext* context);
    public:
      bool record_operation(
          Operation* op, Murmur3Hasher::Hash hash, uint64_t opidx);
      bool record_noop(Operation* op);
      bool has_prefix(const std::vector<Murmur3Hasher::Hash>& hashes) const;
      void insert(std::vector<Murmur3Hasher::Hash>& hashes, uint64_t opidx);
      void flush(uint64_t opidx);
    private:
      bool is_operation_ignorable_in_traces(Operation* op);
      // flush_buffer executes operations until opidx, or flushes
      // the entire operation buffer if no opidx is provided.
      void flush_buffer(void);
      void flush_buffer(uint64_t opidx);
      // replay_trace executes operations under the trace tid
      // until opidx, after which it inserts an end trace.
      void replay_trace(uint64_t opidx, TraceID tid);
    public:
      InnerContext* const context;
    private:
      std::queue<Operation*> operations;
      uint64_t operation_start_idx;
      struct TraceInfo {
        // TraceInfo's need to be default constructable.
        TraceInfo(void) = default;
        TraceInfo(uint64_t opidx_, uint64_t length_)
          : opidx(opidx_), length(length_), last_visited_opidx(0),
            decaying_visits(0), replays(0), last_idempotent_visit_opidx(0),
            decaying_idempotent_visits(0.0), tid(0)
        { }
        // opidx that this trace was inserted at.
        uint64_t opidx;
        // length of the trace. This is used for scoring only.
        uint64_t length;
        // Fields for maintaining a decaying visit count.
        uint64_t last_visited_opidx;
        double decaying_visits;
        // Number of times the trace has been replayed.
        uint64_t replays;
        // Number of times the trace has been visited in
        // an idempotent manner (tracked in a decaying manner).
        uint64_t last_idempotent_visit_opidx;
        double decaying_idempotent_visits;
        // ID for the trace. It is unset if replays == 0.
        TraceID tid;
        // visit updates the TraceInfo's decaying visit count when visited
        // at opidx.
        void visit(uint64_t opidx);
        // score computes the TraceInfo's score when observed at opidx.
        double score(uint64_t opidx) const;
        // R is the exponential rate of decay for a trace.
        static constexpr double R = 0.99;
        // SCORE_CAP_MULT is the multiplier for how large the score
        // of a particular trace can ever get.
        static constexpr double SCORE_CAP_MULT = 10;
        // REPLAY_SCALE is at most how much a score should be increased
        // to favor replays.
        static constexpr double REPLAY_SCALE = 1.75;
        // IDEMPOTENT_VISIT_SCALE is at most how much a score should
        // be increased to favor idempotent replays.
        static constexpr double IDEMPOTENT_VISIT_SCALE = 2.0;
      };
      Trie<Murmur3Hasher::Hash, TraceInfo> trie;
    private:
      // For watching and maintaining decaying visit counts
      // of pointers for scoring.
      class WatchPointer {
      public:
        WatchPointer(
            TrieNode<Murmur3Hasher::Hash, TraceInfo>* node_, uint64_t opidx_)
          : node(node_), opidx(opidx_)
        { }
        // This pointer only has an advance function, as there's nothing
        // to do on commit.
        bool advance(Murmur3Hasher::Hash token);
        uint64_t get_opidx() const { return opidx; }
      private:
        TrieNode<Murmur3Hasher::Hash, TraceInfo>* node;
        uint64_t opidx;
      };
      std::vector<WatchPointer> active_watching_pointers;
    private:
      // For the actual committed trie.
      class CommitPointer {
      public:
        CommitPointer(
            TrieNode<Murmur3Hasher::Hash, TraceInfo>* node_, uint64_t opidx_)
          : node(node_), opidx(opidx_), depth(0)
        { }
        bool advance(Murmur3Hasher::Hash token);
        void advance_for_trace_noop() { depth++; }
        bool complete(void) const;
        TraceID replay(InnerContext* context);
        double score(uint64_t opidx);
        uint64_t get_opidx(void) const { return opidx; }
        uint64_t get_length(void) const { return depth; }
      private:
        TrieNode<Murmur3Hasher::Hash, TraceInfo>* node;
        uint64_t opidx;
        // depth is the number of operations (traceable and trace no-ops)
        // contained within the trace.
        uint64_t depth;
      };
      std::vector<CommitPointer> active_commit_pointers;
    private:
      // FrozenCommitPointer is a commit pointer with a frozen score
      // so that it can be maintained in-order inside completed_commit_pointers.
      // We use a separate type here so that CommitPointers do not get
      // accidentally ordered by the metric below.
      class FrozenCommitPointer : public CommitPointer {
      public:
        // We make these sort keys (score, -opidx) so that the highest
        // scoring, earliest opidx is the first entry in the ordering.
        FrozenCommitPointer(CommitPointer& p, uint64_t opidx)
          : CommitPointer(p), score(p.score(opidx), -int64_t(p.get_opidx()))
        { }
        friend bool operator<(
            const FrozenCommitPointer& a, const FrozenCommitPointer& b)
        // Use > instead of < so that we get descending order.
        {
          return a.score > b.score;
        }
      private:
        std::pair<double, int64_t> score;
      };
      // completed_commit_pointers is a _sorted_ vector of completed
      // commit pointers. All operations on it must preserve the sortedness.
      std::vector<FrozenCommitPointer> completed_commit_pointers;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_TRACE_CACHE_H__
