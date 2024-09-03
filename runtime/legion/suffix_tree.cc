/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#include "legion_defines.h"
#include "legion/suffix_tree.h"

namespace Legion {
  namespace Internal {
    NonOverlappingAlgorithm parse_non_overlapping_algorithm(const std::string& str) {
      if (str == "suffix_tree_walk") {
        return NonOverlappingAlgorithm::SUFFIX_TREE_WALK;
      }
      else if (str == "quick_matching_of_substrings") {
        return NonOverlappingAlgorithm::QUICK_MATCHING_OF_SUBSTRINGS;
      }
      return NonOverlappingAlgorithm::NO_ALGORITHM;
    }

    const char* non_overlapping_algorithm_to_string(NonOverlappingAlgorithm alg) {
      switch (alg) {
        case NonOverlappingAlgorithm::SUFFIX_TREE_WALK: return "suffix_tree_walk";
        case NonOverlappingAlgorithm::QUICK_MATCHING_OF_SUBSTRINGS: return "quick_matching_of_substrings";
        case NonOverlappingAlgorithm::NO_ALGORITHM: return "no_algorithm";
        default: {
          assert(false);
          return nullptr;
        }
      }
    }

    // Please see the following Git repo for a reference implementation and a short explanation:
    // https://github.com/david-broman/matching-substrings
    std::vector<NonOverlappingRepeatsResult>
    quick_matching_of_substrings(size_t min_length,
                                 const std::vector<size_t>& sarray,
                                 const std::vector<size_t>& lcp) {
      std::vector<NonOverlappingRepeatsResult> result;
      size_t le = sarray.size();
      using triple = std::tuple<size_t, size_t, size_t>;
      using pair = std::tuple<size_t, size_t>;

      // Construct tuple array O(n)
      std::vector<triple> a(le * 2 - 2);
      size_t k = 0;
      size_t m = 0;
      size_t pre_l = 0;
      for(size_t i = 0; i < le - 1; i++){
        int l1 = lcp[i];
        int s1 = sarray[i];
        int s2 = sarray[i + 1];
        if(s2 >= s1 + l1 || s2 <= s1 - l1){
          // Non-overlapping
          if(int(pre_l) != l1)
            m += 1;
          a[k++] = std::make_tuple(le - l1, m, s1);
          a[k++] = std::make_tuple(le - l1, m, s2);
          pre_l = l1;
        }
        else if(s2 > s1 && s2 < s1 + l1){
          // Overlapping, increasing index
          size_t d = s2 - s1;
          size_t l3 = (((l1 + d) / 2) / d) * d;
          if(pre_l != l3)
            m += 1;
          a[k++] = std::make_tuple(le - l3, m, s1);
          a[k++] = std::make_tuple(le - l3, m, s1 + l3);
          pre_l = l3;
        }
        else if(s1 > s2 && s1 < s2 + l1){
          // Overlapping, decreasing index
          size_t d = s1 - s2;
          size_t l3 = (((l1 + d) / 2) / d) * d;
          if(pre_l != l3)
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
      for(size_t i = 0; i < a.size(); i++){
        int l = std::get<0>(a[i]);
        size_t m = std::get<1>(a[i]);
        size_t k = std::get<2>(a[i]);
        size_t le2 = le - l;
        if(m != m_pre){
          if(r.size() >= min_repeats){
            result.push_back(NonOverlappingRepeatsResult{
                .start = std::get<0>(r[0]),
                .end = std::get<1>(r[0]),
                .repeats = r.size()});
            for(const pair &p : r)
              for(size_t j = std::get<0>(p); j < std::get<1>(p); j++)
                flag[j] = true;
          }
          r.clear();
          next_k = 0;
        }
        m_pre = m;
        if(le2 != 0 && le2 >= min_length && k >= next_k &&
           !(flag[k]) && !(flag[k + le2 - 1])){
          r.push_back(std::make_tuple(k, k + le2));
          next_k = k + le2;
        }
      }
      if(r.size() >= min_repeats){
        result.push_back(NonOverlappingRepeatsResult{
            .start = std::get<0>(r[0]),
            .end = std::get<1>(r[0]),
            .repeats = r.size()});
      }
      return result;
    }

  };
};
