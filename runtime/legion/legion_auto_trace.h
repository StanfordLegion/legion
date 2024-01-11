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

    template <typename T>
    class AutomaticTracingContext : public T {
    public:
      // TODO (rohany): I'm not sure of the C++-ism to declare this constructor
      //  here and then implement it somewhere else.
      template <typename ... Args>
      AutomaticTracingContext(Args&& ... args) : T(std::forward<Args>(args) ... ) {}
    public:
      bool add_to_dependence_queue(Operation *op,
                                   bool unordered = false,
                                   bool outermost = true) override;
    private:
      std::vector<Murmur3Hasher::Hash> hashes;
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
        auto hash = TraceHashHelper{}.hash(op);
        // TODO (rohany): Have to have a hash value that can be used as the sentinel $
        //  token for the suffix tree processing algorithms.
        assert(!(hash.x == 0 && hash.y == 0));

        this->hashes.push_back(hash);

        // TODO (rohany): Turn this number into a command line parameter.
        if (this->hashes.size() == 1000) {
          // TODO (rohany): Do this allocation so that it doesn't have to resize again.
          // TODO (rohany): Figure out a policy around freeing this allocation.
          auto string = new std::vector<Murmur3Hasher::Hash>(this->hashes.begin(), this->hashes.end());
          string->push_back(Murmur3Hasher::Hash{0, 0});
          auto result = new std::vector<NonOverlappingRepeatsResult>();
          // TODO (rohany): What should the priority be for this?
          // TODO (rohany): Make sure that we don't have too many of these
          //  meta tasks pending at once (should have a fixed amount etc).
          // TODO (rohany): We'll wait on this result for now.
          this->runtime->issue_runtime_meta_task(AutoTraceProcessRepeatsArgs(string, result), LG_LATENCY_WORK_PRIORITY).wait();

          // TODO (rohany): These deletions will need to get handled differently later.
          delete string;
          delete result;
        }
      }

      return T::add_to_dependence_queue(op, unordered, outermost);
    }


  };
};

#endif // __LEGION_AUTO_TRACE_H__
