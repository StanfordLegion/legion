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

#include "legion/tracing/recognizer.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Trace Recognizer
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceRecognizer::TraceRecognizer(
        InnerContext* ctx, const Mapper::ContextConfigOutput& config)
      : context(ctx),
        // Double the window size for the batch size since we need to have a
        // batch size that is double to detect repeats as long as the window
        batchsize(2 * config.auto_tracing_window_size),
        multi_scale_factor(config.auto_tracing_ruler_function),
        min_trace_length(config.auto_tracing_min_trace_length),
        max_trace_length(config.auto_tracing_max_trace_length),
        max_inflight_requests(config.auto_tracing_turbo_lag),
        watcher(ctx, config), unique_hash_value(0), wait_interval(1)
    //--------------------------------------------------------------------------
    {
      hashes.reserve(batchsize);
    }

    //--------------------------------------------------------------------------
    TraceRecognizer::~TraceRecognizer(void)
    //--------------------------------------------------------------------------
    {
      // Cancel any outstanding meta-tasks and then wait for them to be done
      // They shoud all be chained off each other so poisoning the first one
      // and waiting for the last one should be sufficient to cancel everything
      if (!repeat_results.empty())
      {
        // TODO: re-enable this once Realm issue #450 is resolved
        // repeat_results.front().finish_event.cancel_operation(nullptr, 0);
        bool do_not_care = false;
        repeat_results.back().finish_event.wait_faultaware(do_not_care, false);
      }
    }

    //--------------------------------------------------------------------------
    bool TraceRecognizer::record_operation_hash(
        Operation* op, Murmur3Hasher& hasher, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      Murmur3Hasher::Hash hash;
      hasher.finalize(hash);
      legion_assert(hashes.size() < hashes.capacity());
      hashes.emplace_back(hash);
      if (check_for_repeats(opidx))
        update_watcher(opidx);
      return watcher.record_operation(op, hash, opidx);
    }

    //--------------------------------------------------------------------------
    bool TraceRecognizer::record_operation_noop(Operation* op, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      return watcher.record_noop(op);
    }

    //--------------------------------------------------------------------------
    bool TraceRecognizer::record_operation_untraceable(
        Operation* op, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      log_auto_trace.debug() << "Encountered untraceable operation: " << *op;
      // When encountering a non-traceable operation, insert a
      // dummy hash value into the trace identifier so that the
      // traces it finds don't span across these operations.
      // Generate a unique hash and enqueue it
      legion_assert(hashes.size() < hashes.capacity());
      hashes.emplace_back(get_unique_hash());
      if (check_for_repeats(opidx))
        update_watcher(opidx);
      watcher.flush(opidx);
      return false;
    }

    //--------------------------------------------------------------------------
    Murmur3Hasher::Hash TraceRecognizer::get_unique_hash(void)
    //--------------------------------------------------------------------------
    {
      Murmur3Hasher hasher;
      hasher.hash(LAST_OP_KIND);
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
        FindRepeatsResult& repeat = repeat_results.emplace_back(
            FindRepeatsResult(&hashes.front(), hashes.size(), opidx));
        hashes.swap(repeat.hashes);
        // Runtime meta-task in program order
        if (max_inflight_requests > 0)
        {
          FindRepeatsTaskArgs args(this, &repeat);
          repeat.finish_event = runtime->issue_runtime_meta_task(
              args, LG_THROUGHPUT_WORK_PRIORITY,
              repeat_results.size() > 1 ?
                  repeat_results[repeat_results.size() - 2].finish_event :
                  RtEvent::NO_RT_EVENT);
        }
        else
          compute_longest_nonoverlapping_repeats(repeat);
        hashes.reserve(batchsize);
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
        FindRepeatsResult& repeat = repeat_results.emplace_back(
            FindRepeatsResult(&hashes[start], window_size, opidx));
        if (max_inflight_requests > 0)
        {
          FindRepeatsTaskArgs args(this, &repeat);
          // We're going to be a little sneaky around re-using memory for the
          // async jobs, so we're going to make sure that our processing jobs
          // execute in order, because we'll have earlier jobs point to the
          // same memory that later jobs will also use.
          repeat.finish_event = runtime->issue_runtime_meta_task(
              args, LG_THROUGHPUT_WORK_PRIORITY,
              repeat_results.size() > 1 ?
                  repeat_results[repeat_results.size() - 2].finish_event :
                  RtEvent::NO_RT_EVENT);
        }
        else
          compute_longest_nonoverlapping_repeats(repeat);
        return true;
      }
      else
        return !repeat_results.empty();
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::update_watcher(uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      legion_assert(!repeat_results.empty());
      // See if we've exceeded the wait interval since the start operation
      // and if not then we just keep going
      if (opidx < (repeat_results.front().opidx + wait_interval))
        return;
      // If we've hit the maximum number of inflight requests, then wait
      // on one of them to make sure that it gets processed.
      if (max_inflight_requests <= repeat_results.size())
        repeat_results.front().finish_event.wait();
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
        FindRepeatsResult& repeats = repeat_results.front();
        for (NonOverlappingRepeatsResult& result : repeats.result)
          add_trace(
              repeats.start + result.start, (result.end - result.start), opidx);
        repeat_results.pop_front();
      }
      if (double_wait_interval)
        wait_interval *= 2;
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::add_trace(
        const Murmur3Hasher::Hash* hashes, uint64_t size, uint64_t opidx)
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
        add_trace(hashes + max_trace_length, size - max_trace_length, opidx);
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
      legion_assert(query.superstring);
      // If the trace we're inserting is a superstring of another
      // string in the recorded set of traces, then splice out the
      // contained prefix and try to insert the rest of the trace.
      add_trace(
          hashes + query.superstring_match, size - query.superstring_match,
          opidx);
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::compute_suffix_array(
        const FindRepeatsResult& str, std::vector<size_t>& sarray,
        std::vector<int64_t>& surrogate)
    //--------------------------------------------------------------------------
    {
      // Suffix array construction in O(n*log n) time.
      // The code has been implemented based on the explanations from here:
      // http://www.cs.cmu.edu/~15451-f20/LectureNotes/lec25-suffarray.pdf,
      // with special treatment of radix sort to make it O(n*log n).
      const size_t n = str.length();
      legion_assert(n > 0);

      // Define a struct for sorting the input string. To handle an
      // arbitrary type T, we use a boolean `present` to ensure that
      // tokens without a "next" value are sorted before any other tokens.
      struct Key {
        Murmur3Hasher::Hash start;
        bool present;
        Murmur3Hasher::Hash next;
        size_t idx;
        bool operator<(const Key& rhs) const
        {
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
        for (size_t i = 0; i < n; i++)
        {
          w[i] = Key{
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
        for (size_t i = 1; i < n; i++)
        {
          if (x0 != w[i].start || x1 != w[i].next)
            v++;
          surrogate[w[i].idx] = v;
          x0 = w[i].start;
          x1 = w[i].next;
        }
        // In case we're done, reconstruct the suffix array directly
        // from the w vector.
        if (v >= n - 1)
        {
          for (size_t i = 0; i < n; i++)
          {
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
        bool operator<(const SKey& rhs) const
        {
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
      while (true)
      {
        // Update sort table.
        for (size_t i = 0; i < n; i++)
        {
          surrogate_sorter[i] = SKey{
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
        for (size_t i = 0; i < n; i++) count[surrogate_sorter[i].next + 1]++;
        // Update count to contain actual positions.
        for (size_t i = 1; i < v + 2; i++) count[i] += count[i - 1];
        // Construct output array based on second digit.
        for (int64_t i = n - 1; i >= 0; i--)
          tmp[(count[surrogate_sorter[i].next + 1]--) - 1] =
              surrogate_sorter[i];
        // Clear count. Next, sort on first digit.
        std::fill(count.begin(), count.begin() + v + 2, 0);
        // The source is in tmp. Count freq. on first digit.
        for (size_t i = 0; i < n; i++) count[tmp[i].start + 1]++;
        // Update count to contain actual positions.
        for (size_t i = 1; i < v + 2; i++) count[i] += count[i - 1];
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
        for (size_t i = 1; i < n; i++)
        {
          if (x0 != surrogate_sorter[i].start || x1 != surrogate_sorter[i].next)
            v++;
          surrogate[surrogate_sorter[i].idx] = v;
          x0 = surrogate_sorter[i].start;
          x1 = surrogate_sorter[i].next;
        }

        // End if done.
        if (v >= n - 1)
          break;
        shift *= 2;
      }
      // Reconstruct the suffix array.
      for (size_t i = 0; i < n; i++) sarray[i] = surrogate_sorter[i].idx;
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::compute_lcp(
        const FindRepeatsResult& str, const std::vector<size_t>& sarray,
        const std::vector<int64_t>& surrogate, std::vector<size_t>& lcp)
    //--------------------------------------------------------------------------
    {
      // Computes the LCP in O(n) time. This is Kasai's algorithm. See e.g.,
      // http://www.cs.cmu.edu/~15451-f20/LectureNotes/lec25-suffarray.pdf
      // for an explanation. The original paper can be found here:
      // https://link.springer.com/chapter/10.1007/3-540-48194-X_17
      const size_t n = str.length();
      legion_assert(n > 0);

      int k = 0;
      lcp.resize(n, 0);
      for (size_t i = 0; i < n; i++)
      {
        if (surrogate[i] == int(n - 1))
          k = 0;
        else
        {
          size_t j = sarray[surrogate[i] + 1];
          for (; i + k < n && j + k < n && str[i + k] == str[j + k]; k++);
          lcp[surrogate[i]] = k;
          k = std::max(k - 1, 0);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::quick_matching_of_substrings(
        size_t min_length, const std::vector<size_t>& sarray,
        const std::vector<size_t>& lcp,
        std::vector<NonOverlappingRepeatsResult>& result)
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
      for (size_t i = 0; i < le - 1; i++)
      {
        size_t l1 = lcp[i];
        size_t s1 = sarray[i];
        size_t s2 = sarray[i + 1];
        if (s2 >= s1 + l1 || s1 >= s2 + l1)
        {
          // Non-overlapping
          if (pre_l != l1)
            m += 1;
          a[k++] = std::make_tuple(le - l1, m, s1);
          a[k++] = std::make_tuple(le - l1, m, s2);
          pre_l = l1;
        }
        else if (s2 > s1 && s2 < s1 + l1)
        {
          // Overlapping, increasing index
          size_t d = s2 - s1;
          size_t l3 = (((l1 + d) / 2) / d) * d;
          if (pre_l != l3)
            m += 1;
          a[k++] = std::make_tuple(le - l3, m, s1);
          a[k++] = std::make_tuple(le - l3, m, s1 + l3);
          pre_l = l3;
        }
        else if (s1 > s2 && s1 < s2 + l1)
        {
          // Overlapping, decreasing index
          size_t d = s1 - s2;
          size_t l3 = (((l1 + d) / 2) / d) * d;
          if (pre_l != l3)
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
      for (size_t i = 0; i < a.size(); i++)
      {
        int l = std::get<0>(a[i]);
        size_t m = std::get<1>(a[i]);
        size_t k = std::get<2>(a[i]);
        size_t le2 = le - l;
        if (m != m_pre)
        {
          if (r.size() >= min_repeats)
          {
            result.emplace_back(NonOverlappingRepeatsResult{
                .start = std::get<0>(r[0]),
                .end = std::get<1>(r[0]),
                .repeats = r.size()});
            for (const pair& p : r)
              for (size_t j = std::get<0>(p); j < std::get<1>(p); j++)
                flag[j] = true;
          }
          r.clear();
          next_k = 0;
        }
        m_pre = m;
        if (le2 != 0 && le2 >= min_length && k >= next_k && !(flag[k]) &&
            !(flag[k + le2 - 1]))
        {
          r.emplace_back(std::make_tuple(k, k + le2));
          next_k = k + le2;
        }
      }
      if (r.size() >= min_repeats)
      {
        result.emplace_back(NonOverlappingRepeatsResult{
            .start = std::get<0>(r[0]),
            .end = std::get<1>(r[0]),
            .repeats = r.size()});
      }
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::compute_longest_nonoverlapping_repeats(
        FindRepeatsResult& repeat)
    //--------------------------------------------------------------------------
    {
      if (repeat.length() <= 2)
        return;
      std::vector<size_t> sarray(repeat.length());
      std::vector<int64_t> surrogate(repeat.length());
      compute_suffix_array(repeat, sarray, surrogate);
      std::vector<size_t> lcp;
      compute_lcp(repeat, sarray, surrogate, lcp);
      quick_matching_of_substrings(
          min_trace_length, sarray, lcp, repeat.result);
    }

    //--------------------------------------------------------------------------
    void TraceRecognizer::FindRepeatsTaskArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      recognizer->compute_longest_nonoverlapping_repeats(*result);
    }

  }  // namespace Internal
}  // namespace Legion
