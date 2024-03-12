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
  assert(result.size() == 2);
  assert((std::vector<int>{'a', 'b'} == std::vector<int>(str.begin() + result[0].start, str.begin() + result[0].end)) && result[0].repeats == 3);
  assert((std::vector<int>{'c', 'a', 'b'} == std::vector<int>(str.begin() + result[1].start, str.begin() + result[1].end)) && result[1].repeats == 2);
}

void benchmark_repeats(const char* str, NonOverlappingAlgorithm alg) {

  size_t strlen = 5000;
  std::vector<int> vec(strlen + 1);
  for (size_t j = 0; j < strlen; j++) {
    vec[j] = 'a' + (rand() % 3);
  }

  // Sentinel guaranteed not to be equal to any of the entries.
  vec[strlen] = 0;
  int nruns = 100;
  int warmup = 10;

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < nruns + warmup; i++) {
    if (i == warmup) {
      t1 = std::chrono::high_resolution_clock::now();
    }
    compute_longest_nonoverlapping_repeats(vec, 0, alg);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  double us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << str << "Repeats took: " << (us / nruns)  << " microseconds." << std::endl;
}

auto min_len_tests = std::vector<int>{0,0,0,0,0,0,2,0,0,0,0};

auto str_tests = std::vector<std::vector<int>>{
  std::vector<int>{'b', 'a', 'n', 'a', 'n', 'a'},
  std::vector<int>{'b', 'a', 'n', 'a', 'a','n', 'a', 'n'},
  std::vector<int>{3, 1, 8, 8, 3, 1, 8},
  std::vector<int>{'a','a','a','a','a','a','a','b','b','b','b','b','b','a','a','a','a','a','a'},
  std::vector<int>{'a','a','a','a','a','a'} ,
  std::vector<int>{'a','a','a'} ,
  std::vector<int>{'a','a','a'} ,
  std::vector<int>{'a','a'},
  std::vector<int>{'a'} ,
  std::vector<int>{},
  std::vector<int>{'f','d','d','d','a','b','a','b','c','d','c','d','c','d','e','d','d','d'}
};

auto sarray_tests = std::vector<std::vector<size_t>>{
  std::vector<size_t>{5, 3, 1, 0, 4, 2},
  std::vector<size_t>{3, 6, 1, 4, 0, 7, 2, 5},
  std::vector<size_t>{5, 1, 4, 0, 6, 3, 2},
  std::vector<size_t>{18, 17, 16, 15, 14, 13, 0, 1, 2, 3, 4, 5, 6, 12, 11, 10, 9, 8, 7},
  std::vector<size_t>{5, 4, 3, 2, 1, 0},
  std::vector<size_t>{2, 1, 0},
  std::vector<size_t>{2, 1, 0},
  std::vector<size_t>{1, 0},
  std::vector<size_t>{0},
  std::vector<size_t>{},
  std::vector<size_t>{4, 6, 5, 7, 8, 10, 12, 17, 3, 9, 11, 16, 2, 15, 1, 13, 14, 0}
};

auto lcp_tests = std::vector<std::vector<size_t>>{
  std::vector<size_t>{1, 3, 0, 0, 2, 0},
  std::vector<size_t>{1, 2, 3, 0, 0, 1, 2, 0},
  std::vector<size_t>{2, 0, 3, 0, 1, 1, 0},
  std::vector<size_t>{1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 0},
  std::vector<size_t>{1, 2, 3, 4, 5, 0},
  std::vector<size_t>{1, 2, 0},
  std::vector<size_t>{1, 2, 0},
  std::vector<size_t>{1, 0},
  std::vector<size_t>{0},
  std::vector<size_t>{},
  std::vector<size_t>{2, 0, 1, 0, 4, 2, 0, 1, 1, 3, 1, 2, 2, 3, 1, 0, 0, 0}
};

#define RES(s, e, r) (NonOverlappingRepeatsResult{.start = s, .end = e, .repeats = r})

auto result_tests = std::vector<std::vector<NonOverlappingRepeatsResult>>{
  std::vector<NonOverlappingRepeatsResult>{RES(1, 3, 2)},
  std::vector<NonOverlappingRepeatsResult>{RES(1, 4, 2)},
  std::vector<NonOverlappingRepeatsResult>{RES(0, 3, 2)},
  std::vector<NonOverlappingRepeatsResult>{RES(0, 6, 2), RES(7, 10, 2)},
  std::vector<NonOverlappingRepeatsResult>{RES(0, 3, 2)},
  std::vector<NonOverlappingRepeatsResult>{RES(0, 1, 3)},
  std::vector<NonOverlappingRepeatsResult>{},
  std::vector<NonOverlappingRepeatsResult>{RES(0, 1, 2)},
  std::vector<NonOverlappingRepeatsResult>{},
  std::vector<NonOverlappingRepeatsResult>{},
  std::vector<NonOverlappingRepeatsResult>{RES(1, 4, 2), RES(4, 6, 2), RES(8, 10, 3)}
};

void test_known_str_sarray_lsp() {
  for(size_t i = 0; i < str_tests.size(); i++){
    size_t len = str_tests[i].size();

    // Test expected suffix array results
    std::vector<size_t> sarray(len);
    std::vector<int64_t> surrogate(len);
    suffix_array(str_tests[i], sarray, surrogate);
    assert(sarray == sarray_tests[i]);

    // Test expected LCP results
    auto lcp = compute_lcp(str_tests[i], sarray, surrogate);
    assert(lcp == lcp_tests[i]);
  }
}

void test_known_matching_substrings() {
  for(size_t i = 0; i < str_tests.size(); i++){
    NonOverlappingAlgorithm alg = NonOverlappingAlgorithm::QUICK_MATCHING_OF_SUBSTRINGS;
    auto result = compute_longest_nonoverlapping_repeats(str_tests[i], min_len_tests[i], alg);
    assert(result.size() == result_tests[i].size());
    for(size_t j = 0; j < result.size(); j++){
      assert(result[j].start == result_tests[i][j].start);
      assert(result[j].end == result_tests[i][j].end);
      assert(result[j].repeats == result_tests[i][j].repeats);
    }
  }
}


auto gen_conf_tests = std::vector<std::vector<size_t>>{
  std::vector<size_t>{4, 2, 3, 5},
  std::vector<size_t>{40, 20, 1, 10},
  std::vector<size_t>{3, 50, 10, 3},
  std::vector<size_t>{10, 50, 10, 1},
  std::vector<size_t>{100, 2, 1, 1},
};

auto gen_result_tests = std::vector<std::vector<NonOverlappingRepeatsResult>>{
  std::vector<NonOverlappingRepeatsResult>{RES(0, 24, 5)},
  std::vector<NonOverlappingRepeatsResult>{RES(0, 800, 10)},
  std::vector<NonOverlappingRepeatsResult>{RES(0, 1500, 3)},
  std::vector<NonOverlappingRepeatsResult>{RES(0, 2500, 2)},
  std::vector<NonOverlappingRepeatsResult>{RES(0, 50, 2), RES(100, 150, 2)},
};

std::vector<NonOverlappingRepeatsResult>
test_gen_string_of_substrings(size_t token_repeats, size_t token_kinds,
                           size_t inner_repeats, size_t outer_repeats){
    // Example 0000111100001111000011112
    //         0000111100001111000011113
    //         0000111100001111000011114
    //         0000111100001111000011115
    //         0000111100001111000011116
    // The above has properties
    // token_repeats = 4
    // token_kinds   = 2
    // inner_repeats = 3
    // outer_repeats = 5  (number of unique separators is equal to outer_repeats)

    // Timing 4900 elements  = 0.12 s
    //        49000 elements = 0.12 s
    //        490000 elements = 0.93 s
    //        4900000 elements = 9.93 s


    size_t n = token_repeats * token_kinds * inner_repeats * outer_repeats + outer_repeats;
    std::vector<int> str(n);
    size_t sep = 0;
    for(size_t i = 0; i < outer_repeats; i++){
      for(size_t j = 0; j < inner_repeats; j++)
        for(size_t k = 0; k < token_kinds; k++)
          for(size_t l = 0; l < token_repeats; l++)
            str[l + k * token_repeats + j * token_repeats * token_kinds +
                i * token_repeats * token_kinds * inner_repeats + sep] = k;
      str[token_repeats * token_kinds * inner_repeats * (i + 1) + sep] =
        sep + token_kinds;
      sep++;
    }
    NonOverlappingAlgorithm alg = NonOverlappingAlgorithm::QUICK_MATCHING_OF_SUBSTRINGS;
    auto result = compute_longest_nonoverlapping_repeats(str, 0, alg);
    return result;
}


void test_gen_large_substrings(){
  for(size_t i = 0; i < gen_conf_tests.size(); i++){
    auto r = gen_conf_tests[i];
    auto result = test_gen_string_of_substrings(r[0], r[1], r[2], r[3]);
    assert(result.size() == gen_result_tests[i].size());
    for(size_t j = 0; j < result.size(); j++){
      assert(result[j].start == gen_result_tests[i][j].start);
      assert(result[j].end == gen_result_tests[i][j].end);
      assert(result[j].repeats == gen_result_tests[i][j].repeats);
    }
  }
}


int main(int argc, char **argv) {
  random_creation_test();
  random_search_test();
  basic_repeats_test();
  benchmark_repeats("Suffix tree. ", NonOverlappingAlgorithm::SUFFIX_TREE_WALK);
  benchmark_repeats("Quick substring matching. ",
                    NonOverlappingAlgorithm::QUICK_MATCHING_OF_SUBSTRINGS);

  test_known_str_sarray_lsp();
  test_known_matching_substrings();
  test_gen_large_substrings();

  std::cout << "Passed all tests!" << std::endl;
  return 0;
}
