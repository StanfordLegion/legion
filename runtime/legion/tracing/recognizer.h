/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_TRACE_RECOGNIZER_H__
#define __LEGION_TRACE_RECOGNIZER_H__

#include "legion/tracing/recording.h"
#include "legion/tracing/watcher.h"

namespace Legion {
  namespace Internal {

    /**
     * \class TraceRecognizer
     * The trace recognizer class lazily buffers up a sequence of hashes
     * corresponding to the sequence of operations and their arguments
     * and looks for repeats within the sequence for which we can replay.
     */
    class TraceRecognizer : public TraceHashRecorder {
    public:
      // Non overlapping repeats implementation.
      struct NonOverlappingRepeatsResult {
        size_t start;
        size_t end;
        size_t repeats;
      };
      struct FindRepeatsResult {
      public:
        const Murmur3Hasher::Hash* start;
      private:
        size_t size;
      public:
        uint64_t opidx;
        RtEvent finish_event;
        std::vector<Murmur3Hasher::Hash> hashes;  // only for storage
        std::vector<NonOverlappingRepeatsResult> result;
      public:
        FindRepeatsResult(Murmur3Hasher::Hash* h, size_t s, uint64_t idx)
          : start(h), size(s), opidx(idx)
        { }
        // Make this act like a string view that is always terminated with a
        // sentinel
        inline Murmur3Hasher::Hash operator[](unsigned idx) const
        {
          if (idx < size)
            return start[idx];
          legion_assert(idx == size);
          return SENTINEL;
        }
        inline size_t length(void) const
        {
          // Always allow for the sentinel value
          return size + 1;
        }
      };
      struct FindRepeatsTaskArgs : public LgTaskArgs<FindRepeatsTaskArgs> {
      public:
        static constexpr LgTaskID TASK_ID =
            LG_AUTO_TRACE_PROCESS_REPEATS_TASK_ID;
      public:
        FindRepeatsTaskArgs(void) = default;
        FindRepeatsTaskArgs(TraceRecognizer* recog, FindRepeatsResult* res)
          : LgTaskArgs<FindRepeatsTaskArgs>(false, true), recognizer(recog),
            result(res)
        { }
        void execute(void) const;
      public:
        TraceRecognizer* recognizer;
        FindRepeatsResult* result;
      };
    public:
      TraceRecognizer(
          InnerContext* context, const Mapper::ContextConfigOutput& config);
      TraceRecognizer(const TraceRecognizer& rhs) = delete;
      virtual ~TraceRecognizer(void);
    public:
      TraceRecognizer& operator=(const TraceRecognizer& rhs) = delete;
    public:  // From TraceHashRecorder
      virtual bool record_operation_hash(
          Operation* op, Murmur3Hasher& hasher, uint64_t opidx) override;
      virtual bool record_operation_noop(
          Operation* op, uint64_t opidx) override;
      virtual bool record_operation_untraceable(
          Operation* op, uint64_t opidx) override;
    private:
      bool check_for_repeats(uint64_t opidx);
      void update_watcher(uint64_t opidx);
      void add_trace(
          const Murmur3Hasher::Hash* hashes, uint64_t size, uint64_t opidx);
      void compute_suffix_array(
          const FindRepeatsResult& string, std::vector<size_t>& sarray,
          std::vector<int64_t>& surrogate);
      void compute_lcp(
          const FindRepeatsResult& string, const std::vector<size_t>& sarray,
          const std::vector<int64_t>& surrogate, std::vector<size_t>& lcp);
      void quick_matching_of_substrings(
          size_t min_length, const std::vector<size_t>& sarray,
          const std::vector<size_t>& lcp,
          std::vector<NonOverlappingRepeatsResult>& result);
      void compute_longest_nonoverlapping_repeats(FindRepeatsResult& result);
      // Generates a hash value that will not repeat. This is used to
      // represent operations or events that are not traceable, so that
      // the trace identification analysis does not identify repeats that
      // cross over untraceable operations.
      Murmur3Hasher::Hash get_unique_hash(void);
    public:
      InnerContext* const context;
      const uint64_t batchsize;
      const uint64_t multi_scale_factor;
      const uint64_t min_trace_length;
      const uint64_t max_trace_length;
      const unsigned max_inflight_requests;
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

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_TRACE_RECOGNIZER_H__
