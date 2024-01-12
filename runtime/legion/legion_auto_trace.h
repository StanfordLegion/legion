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
#include "legion/suffix_tree.h"
#include "legion/trie.h"

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

namespace Legion {
  namespace Internal {

    // Forward declarations.
    class BatchedTraceIdentifier;
    class TraceOccurrenceWatcher;

    // TODO (rohany): Can add hashing methods here for hashing? I don't think that
    //  I want to do this on the Operations themselves because i'm not going to has
    //  everything about each operation, but just fields that are necessary for tracing.
    class TraceHashHelper {
    public:
      TraceHashHelper();
      Murmur3Hasher::Hash hash(Operation* op);
      void hash(TaskOp* op);
      void hash(FillOp* op);
      void hash(FenceOp* op);
      void hash(CopyOp* op);
      // TODO (rohany): AllReduceOp, DiscardOp.
      void hash(const RegionRequirement& req);
      void hash(const LogicalRegion& region);
    private:
      Murmur3Hasher hasher;
    };

    class BatchedTraceIdentifier {
    public:
      BatchedTraceIdentifier(
        Runtime* runtime,
        TraceOccurrenceWatcher* watcher,
        size_t batchsize,  // Number of operations batched at once.
        size_t max_add // Maximum number of traces to add to the watcher at once.
      );
      void process(Murmur3Hasher::Hash hash, size_t opidx);
    private:
      // We need a runtime here in order to launch meta tasks.
      Runtime* runtime;
      std::vector<Murmur3Hasher::Hash> hashes;
      TraceOccurrenceWatcher* watcher;
      size_t batchsize;
      size_t max_add;
    };

    class TraceOccurrenceWatcher {
    public:
      TraceOccurrenceWatcher(size_t visit_threshold);
      void process(Murmur3Hasher::Hash hash, size_t opidx);

      template<typename T>
      void insert(T start, T end, size_t opidx);
      template<typename T>
      bool prefix(T start, T end);
    private:
      struct TraceMeta {
        // Needs to be default constructable.
        TraceMeta() : opidx(0) { }
        TraceMeta(size_t opidx_) : opidx(opidx_) { }
        // The opidx that this trace was inserted at.
        size_t opidx;
        // The occurrence watcher will only maintain the number
        // of visits. I don't think that we need to do decaying visits
        // here, though we might want to lower the amount of traces that
        // get committed to the replayer.
        size_t visits = 0;
        // completed marks whether this trace has moved
        // from the "watched" state to the "committed" state.
        // Once a trace has been completed, it will not be
        // returned from complete() anymore.
        bool completed = false;
      };
      Trie<Murmur3Hasher::Hash, TraceMeta> trie;
      size_t visit_threshold;

      // TriePointer maintains an active trace being
      // traversed in the watcher's trie.
      class TriePointer {
      public:
        TriePointer(TrieNode<Murmur3Hasher::Hash, TraceMeta>* node_, size_t opidx_)
          : node(node_), opidx(opidx_), depth(0) { }
        bool advance(Murmur3Hasher::Hash token);
        bool complete();
      public:
        TrieNode<Murmur3Hasher::Hash, TraceMeta>* node;
        size_t opidx;
        size_t depth;
      };
      // All currently active pointers that need advancing.
      std::vector<TriePointer> active_pointers;
    };

    template <typename T>
    class AutomaticTracingContext : public T {
    public:
      // TODO (rohany): I'm not sure of the C++-ism to declare this constructor
      //  here and then implement it somewhere else.
      template <typename ... Args>
      AutomaticTracingContext(Args&& ... args)
        // TODO (rohany): Make all of these constants command line parameters.
        : T(std::forward<Args>(args) ... ),
          opidx(0),
          identifier(this->runtime, &this->watcher, 100, 10),
          watcher(15)
          {}
    public:
      bool add_to_dependence_queue(Operation *op,
                                   bool unordered = false,
                                   bool outermost = true) override;
    private:
      size_t opidx;
      BatchedTraceIdentifier identifier;
      TraceOccurrenceWatcher watcher;
    };

    // Offline string analysis operations.
    struct AutoTraceProcessRepeatsArgs : public LgTaskArgs<AutoTraceProcessRepeatsArgs> {
    public:
      static const LgTaskID TASK_ID = LG_AUTO_TRACE_PROCESS_REPEATS_TASK_ID;
    public:
      AutoTraceProcessRepeatsArgs(
       std::vector<Murmur3Hasher::Hash>* operations_,
       std::vector<NonOverlappingRepeatsResult>* result_
      ) : LgTaskArgs<AutoTraceProcessRepeatsArgs>(implicit_provenance), operations(operations_), result(result_) {}
    public:
      std::vector<Murmur3Hasher::Hash>* operations;
      std::vector<NonOverlappingRepeatsResult>* result;
    };
    void auto_trace_process_repeats(const void* args);

    // Utility functions.
    bool is_operation_traceable(Operation* op);


    // TODO (rohany): Can we move these declarations to another file?

    template <typename T>
    bool AutomaticTracingContext<T>::add_to_dependence_queue(Operation* op, bool unordered, bool outermost) {
      // TODO (rohany): unordered operations should always be forwarded to the underlying context and not buffered up.
      if (is_operation_traceable(op)) {
        Murmur3Hasher::Hash hash = TraceHashHelper{}.hash(op);
        // TODO (rohany): Have to have a hash value that can be used as the sentinel $
        //  token for the suffix tree processing algorithms.
        assert(!(hash.x == 0 && hash.y == 0));
        this->identifier.process(hash, this->opidx);
        this->watcher.process(hash, this->opidx);

        this->opidx++;
      }

      return T::add_to_dependence_queue(op, unordered, outermost);
    }

    template <typename T>
    void TraceOccurrenceWatcher::insert(T start, T end, size_t opidx) {
      this->trie.insert(start, end, TraceMeta(opidx));
    }

    template <typename T>
    bool TraceOccurrenceWatcher::prefix(T start, T end) {
      return this->trie.prefix(start, end);
    }
  };
};

#endif // __LEGION_AUTO_TRACE_H__
