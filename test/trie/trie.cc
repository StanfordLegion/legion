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
        trie.insert(string, empty{});
        strings.insert(string);
        break;
      }
    }
  }

  // All of the strings in our set should be in the trie, and a random
  // prefix of each should be in the trie.
  for (auto string : strings) {
    assert(trie.contains(string));
    int len = rand() % string.size();
    std::vector<char> prefix(string.begin(), string.begin() + len);
    assert(trie.prefix(prefix));
  }

  // A bunch of random strings should not be in the trie.
  for (int i = 0; i < 100; i++) {
    while (true) {
      int strlen = (rand() % maxlen) + 1;
      auto string = random_string(strlen);
      if (strings.find(string) == strings.end()) {
        assert(!trie.contains(string));
        break;
      }
    }
  }
}


int main(int argc, char **argv) {
  random_test();
  std::cout << "Passed all tests!" << std::endl;
  return 0;
}
