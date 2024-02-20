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

#include <legion/trie.h>

#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

using namespace Legion;
using namespace Legion::Internal;

using namespace std;

struct empty {};

std::vector<char> random_string(size_t len) {
  std::vector<char> string(len);
  for (int j = 0; j < len; j++) {
    string[j] = 'a' + (rand() % 26);
  }
  return string;
}

void random_test() {
  // Generate N random strings of a random length. Maintain
  // a map of these strings so that we don't double insert.
  std::set<std::vector<char>> strings;
  Trie<char, empty> trie;
  int N = 1000;
  int maxlen = 100;
  for (int i = 0; i < N; i++) {
    while (true) {
      int strlen = (rand() % maxlen) + 1;
      auto string = random_string(strlen);
      if (strings.find(string) == strings.end()) {
        trie.insert(string.begin(), string.end(), empty{});
        strings.insert(string);
        break;
      }
    }
  }

  // All of the strings in our set should be in the trie, and a random
  // prefix of each should be in the trie.
  for (auto string : strings) {
    assert(trie.contains(string.begin(), string.end()));
    int len = rand() % string.size();
    assert(trie.prefix(string.begin(), string.begin() + len));

    // Also test the query API.
    auto result = trie.query(string.begin(), string.end());
    assert(result.contains && result.prefix);
    result = trie.query(string.begin(), string.begin() + len);
    // assert(!result.contains && result.prefix);
    assert(result.contains == trie.contains(string.begin(), string.begin() + len));
    assert(result.prefix);
  }

  // A bunch of random strings should not be in the trie.
  for (int i = 0; i < 100; i++) {
    while (true) {
      int strlen = (rand() % maxlen) + 1;
      auto string = random_string(strlen);
      if (strings.find(string) == strings.end()) {
        assert(!trie.contains(string.begin(), string.end()));
        // Also test the query API.
        auto result = trie.query(string.begin(), string.end());
        assert(!result.contains);
        break;
      }
    }
  }
}

void test_superstring() {
  Trie<char, empty> trie;
  std::vector<std::vector<char>> strings = {
      {'a', 'b', 'c'},
      {'d', 'e'},
      {'f', 'g', 'h', 'i'}
  };
  for (auto& s : strings) {
    trie.insert(s.begin(), s.end(), empty{});
  }
  std::vector<char> q1 = {'a', 'b', 'c', 'd', 'e'};
  assert(trie.query(q1.begin(), q1.end()).superstring &&
         trie.query(q1.begin(), q1.end()).superstring_match == 3);
  std::vector<char> q2 = {'b', 'c', 'd', 'e'};
  assert(!trie.query(q2.begin(), q2.end()).superstring);
  std::vector<char> q3 = {'k', 'l', 'm', 'n'};
  assert(!trie.query(q3.begin(), q3.end()).superstring);
  std::vector<char> q4 = {'d', 'e', 'f'};
  assert(trie.query(q4.begin(), q4.end()).superstring &&
         trie.query(q4.begin(), q4.end()).superstring_match == 2);
}

int main(int argc, char **argv) {
  random_test();
  test_superstring();
  std::cout << "Passed all tests!" << std::endl;
  return 0;
}
