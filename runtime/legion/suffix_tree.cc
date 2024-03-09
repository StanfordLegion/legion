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

#include "legion/suffix_tree.h"

namespace Legion {
  namespace Internal {
    NonOverlappingAlgorithm parse_non_overlapping_algorithm(const std::string& str) {
      if (str == "suffix_tree_walk") {
        return NonOverlappingAlgorithm::SUFFIX_TREE_WALK;
      }
      else if (str == "quick_matching_substrings") {
        return NonOverlappingAlgorithm::QUICK_MATCHING_SUBSTRINGS;
      }
      return NonOverlappingAlgorithm::NO_ALGORITHM;
    }

    const char* non_overlapping_algorithm_to_string(NonOverlappingAlgorithm alg) {
      switch (alg) {
        case NonOverlappingAlgorithm::SUFFIX_TREE_WALK: return "suffix_tree_walk";
        case NonOverlappingAlgorithm::QUICK_MATCHING_SUBSTRINGS: return "quick_matching_substrings";
        case NonOverlappingAlgorithm::NO_ALGORITHM: return "no_algorithm";
        default: {
          assert(false);
          return nullptr;
        }
      }
    }
  };
};
