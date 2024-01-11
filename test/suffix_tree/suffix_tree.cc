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

#include <legion/suffix_tree.h>

#include <chrono>
#include <cstdlib>
#include <vector>

using namespace Legion;
using namespace Legion::Internal;

using namespace std;

void random_creation_test() {
  size_t strlen = 1000;
  // Brute force test random suffix tree construction. We shouldn't
  // fail any assertions on some random strings.
  for (size_t i = 0; i < 1000; i++) {
    std::vector<int> vec(strlen + 1);
    for (size_t j = 0; j < strlen; j++) {
      vec[j] = 'a' + (rand() % 26);
    }
    // Sentinel guaranteed not to be equal to any of the entries.
    vec[strlen] = 0;
    SuffixTree<int> tree(vec);
  }
}

void random_search_test() {
  size_t strlen = 1000;
  std::vector<int> vec(strlen + 1);
  for (size_t j = 0; j < strlen; j++) {
    vec[j] = 'a' + (rand() % 26);
  }
  // Sentinel guaranteed not to be equal to any of the entries.
  vec[strlen] = 0;
  SuffixTree<int> tree(vec);
  for (size_t i = 0; i < 1000; i++) {
    // Generate a random substring that is gauranteed to be in the tree.
    size_t querylen = (rand() % 100) + 1;
    size_t start = rand() % (strlen - querylen);
    std::vector<int> query(vec.begin() + start, vec.begin() + start + querylen);
    assert(tree.find(vec, query));
    // Generate a random substring that is gauranteed to not be in the tree.
    query[rand() % querylen] = 1;
    assert(!tree.find(vec, query));
  }
}

void basic_repeats_test() {
  auto str = std::vector<int>{'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 0};
  auto result = compute_longest_nonoverlapping_repeats(str);
  assert(result.size() == 3);
  assert((std::vector<int>{'a', 'b'} == std::vector<int>(str.begin() + result[0].start, str.begin() + result[0].end)) && result[0].repeats == 3);
  assert((std::vector<int>{'c', 'a', 'b'} == std::vector<int>(str.begin() + result[1].start, str.begin() + result[1].end)) && result[1].repeats == 2);
  assert((std::vector<int>{'b'} == std::vector<int>(str.begin() + result[2].start, str.begin() + result[2].end)) && result[2].repeats == 3);
}

void benchmark_repeats() {
  size_t strlen = 1000;
  std::vector<int> vec(strlen + 1);
  for (size_t j = 0; j < strlen; j++) {
    vec[j] = 'a' + (rand() % 3);
  }
  // Sentinel guaranteed not to be equal to any of the entries.
  vec[strlen] = 0;
  int nruns = 1000;
  int warmup = 10;

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < nruns + warmup; i++) {
    if (i == warmup) {
      t1 = std::chrono::high_resolution_clock::now();
    }
    compute_longest_nonoverlapping_repeats(vec);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  double us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Repeats took: " << (us / nruns)  << " microseconds." << std::endl;
}

int main(int argc, char **argv) {
  random_creation_test();
  random_search_test();
  basic_repeats_test();
  benchmark_repeats();
  std::cout << "Passed all tests!" << std::endl;
  return 0;
}
