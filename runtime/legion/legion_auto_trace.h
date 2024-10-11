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

#ifndef __LEGION_AUTO_TRACE_H__
#define __LEGION_AUTO_TRACE_H__

#include "legion.h"
#include "legion/legion_context.h"
#include "legion/legion_types.h"
#include "legion/legion_utilities.h"
#include "legion/trie.h"

#include <limits>
#include <queue>

template<>
struct std::hash<Legion::Internal::Murmur3Hasher::Hash> {
  std::size_t operator()(const Legion::Internal::Murmur3Hasher::Hash& h) const noexcept {
    return h.x ^ (h.y << 1);
  }
};

template<>
struct std::equal_to<Legion::Internal::Murmur3Hasher::Hash> {
  constexpr bool operator()(const Legion::Internal::Murmur3Hasher::Hash& lhs,
                            const Legion::Internal::Murmur3Hasher::Hash& rhs) const
  {
      return lhs.x == rhs.x && lhs.y == rhs.y;
  }
};

template<>
struct std::less<Legion::Internal::Murmur3Hasher::Hash> {
  constexpr bool operator()(const Legion::Internal::Murmur3Hasher::Hash& lhs,
                            const Legion::Internal::Murmur3Hasher::Hash& rhs) const
  {
    return lhs.x < rhs.x && lhs.y < rhs.y;
  }
};

namespace Legion {
  namespace Internal {

    // Declare all of the necessary loggers.
    LEGION_EXTERN_LOGGER_DECLARATIONS

    /**
     * \class TraceCache
     * The trace cache maintains a trie corresponding to traces that have
     * been observed the minimum number of times required for replay at 
     * which point once we see them we can start to replay them.
     */
    class TraceCache {
    public:
      TraceCache(InnerContext *context);
    public:
      void record_operation(Operation *op,
          Murmur3Hasher::Hash hash, uint64_t opidx);
      void record_noop(Operation *op);
      bool has_prefix(const std::vector<Murmur3Hasher::Hash> &hashes) const;
      void insert(std::vector<Murmur3Hasher::Hash> &hashes, uint64_t opidx);
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
      InnerContext *const context;
    private:
      std::queue<Operation*> operations;
      uint64_t operation_start_idx;
      struct TraceInfo {
        // TraceInfo's need to be default constructable.
        TraceInfo(void) = default;
        TraceInfo(uint64_t opidx_, uint64_t length_)
          : opidx(opidx_), length(length_), last_visited_opidx(0),
            decaying_visits(0), replays(0),
            last_idempotent_visit_opidx(0),
            decaying_idempotent_visits(0.0), tid(0) { }
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
        WatchPointer(TrieNode<Murmur3Hasher::Hash, TraceInfo>* node_,
                     uint64_t opidx_)
            : node(node_), opidx(opidx_) { }
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
        CommitPointer(TrieNode<Murmur3Hasher::Hash, TraceInfo>* node_,
                      uint64_t opidx_)
          : node(node_), opidx(opidx_), depth(0) { }
        bool advance(Murmur3Hasher::Hash token);
        void advance_for_trace_noop() { depth++; }
        bool complete(void) const;
        TraceID replay(InnerContext *context);
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
          : CommitPointer(p),
            score(p.score(opidx), -int64_t(p.get_opidx())) { }
        friend bool operator<(const FrozenCommitPointer& a,
                              const FrozenCommitPointer& b) 
          // Use > instead of < so that we get descending order.
          { return a.score > b.score; }
      private:
        std::pair<double, int64_t> score;
      };
      // completed_commit_pointers is a _sorted_ vector of completed
      // commit pointers. All operations on it must preserve the sortedness.
      std::vector<FrozenCommitPointer> completed_commit_pointers;
    };

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
      OccurrenceWatcher(InnerContext *context,
          const Mapper::ContextConfigOutput &config);
    public:
      void record_operation(Operation *op,
          Murmur3Hasher::Hash hash, uint64_t opidx); 
      void record_noop(Operation *op);
      void flush(uint64_t opidx);
      void insert(const Murmur3Hasher::Hash *hashes, 
                  size_t size, uint64_t opidx);
      TrieQueryResult query(const Murmur3Hasher::Hash *hashes,
                            size_t size) const;
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
        TriePointer(TrieNode<Murmur3Hasher::Hash, TraceCandidate>* node_,
                    uint64_t opidx_)
          : node(node_), opidx(opidx_), depth(0) { }
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

    /**
     * \class TraceRecognizer
     * The trace recognizer class lazily buffers up a sequence of hashes 
     * corresponding  to the sequence of operations and their arguments 
     * and looks for repeats within the sequence for which we can replay. 
     */
    class TraceRecognizer {
    public:
      // Non overlapping repeats implementation.
      struct NonOverlappingRepeatsResult {
        size_t start;
        size_t end;
        size_t repeats;
      };
      struct FindRepeatsResult {
        std::vector<Murmur3Hasher::Hash> hashes; // only for storage
        std::vector<NonOverlappingRepeatsResult> result;
        Murmur3Hasher::Hash *start;
        size_t size;
        uint64_t opidx;
        RtEvent finish_event;
      };
      struct FindRepeatsTaskArgs : public LgTaskArgs<FindRepeatsTaskArgs> {
      public:
        static constexpr LgTaskID TASK_ID = 
          LG_AUTO_TRACE_PROCESS_REPEATS_TASK_ID;
      public:
        FindRepeatsTaskArgs(TraceRecognizer *recog, FindRepeatsResult *res)
          : LgTaskArgs<FindRepeatsTaskArgs>(implicit_provenance),
            recognizer(recog), result(res) { } 
      public:
        TraceRecognizer *const recognizer;
        FindRepeatsResult *const result;
      };
    public:
      TraceRecognizer(InnerContext *context,
          const Mapper::ContextConfigOutput &config);
    public:
      void record_operation_hash(Operation *op, 
          Murmur3Hasher &hasher, uint64_t opidx);
      void record_operation_noop(Operation *op);
      void record_operation_untraceable(uint64_t opidx);
      static void find_repeats(const void *args);
    private:
      bool check_for_repeats(uint64_t opidx);
      void update_watcher(uint64_t opidx);
      void add_trace(const Murmur3Hasher::Hash *hashes, 
                     uint64_t size, uint64_t opidx);
      void compute_suffix_array(const Murmur3Hasher::Hash *hashes, size_t size,
                                std::vector<size_t> &sarray,
                                std::vector<int64_t> &surrogate);
      void compute_lcp(const Murmur3Hasher::Hash *hashes, size_t size,
                       const std::vector<size_t> &sarray,
                       const std::vector<int64_t> &surrogate,
                       std::vector<size_t> &lcp);
      void quick_matching_of_substrings(size_t min_length,
          const std::vector<size_t> &sarray,
          const std::vector<size_t> &lcp,
          std::vector<NonOverlappingRepeatsResult> &result);
      void compute_longest_nonoverlapping_repeats(FindRepeatsResult &result);
      // Generates a hash value that will not repeat. This is used to 
      // represent operations or events that are not traceable, so that 
      // the trace identification analysis does not identify repeats that
      // cross over untraceable operations.
      Murmur3Hasher::Hash get_unique_hash(void);
    public:
      InnerContext *const context;
      const uint64_t batchsize;
      const uint64_t multi_scale_factor;
      const uint64_t min_trace_length;
      const uint64_t max_trace_length;
      static constexpr Murmur3Hasher::Hash SENTINEL = {};
    private:
      OccurrenceWatcher watcher;
      std::vector<Murmur3Hasher::Hash> hashes; 
      std::deque<FindRepeatsResult> repeat_results;
      // unique_hash_value maintains a counter of non-traceable operations
      // seen so far, used to generate unique hashes for those operations.
      uint64_t unique_hash_value;
      unsigned wait_interval;
    };

    /**
     * \class AutoTracing
     * The auto-tracing class provides an overload of the 
     * add_to_dependence_queue method that will hook in the auto tracing
     * infrastructure and use the trace recognizer to see if we can find
     * traces that we will try to replay.
     */
    template<typename T>
    class AutoTracing : public T {
    public:
      template <typename ... Args>
      AutoTracing(const Mapper::ContextConfigOutput &config, Args&& ... args)
        : T(config, std::forward<Args>(args) ... ),
          recognizer(this, config), opidx(0) { }
    public:
      virtual bool add_to_dependence_queue(Operation *op,
          const std::vector<StaticDependence>* dependences = NULL,
          bool unordered = false, bool outermost = true) override;
      // If the application performs a blocking operation, we need to know
      // about that, so override TaskContext::record_blocking_call().
      virtual void record_blocking_call(uint64_t future_coordinate,
                                        bool invalidate_trace = true) override;
    private:
      TraceRecognizer recognizer;
      uint64_t opidx;
    };

  };
};

#endif // __LEGION_AUTO_TRACE_H__
