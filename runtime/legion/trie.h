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

#ifndef __LEGION_TRIE_H__
#define __LEGION_TRIE_H__

#include <unordered_map>

namespace Legion {
  namespace Internal {

    // Forward declarations.
    template <typename T, typename V>
    class Trie;

    template <typename T, typename V>
    class TrieNode {
    public:
      TrieNode(T token_) : token(token_), end(false) {}
      // TODO (rohany): Implement a recursive destructor.
    private:
      friend class Trie<T, V>;
      T token;
      V value;
      bool end;
      std::unordered_map<T, TrieNode<T, V>*> children;
    };

    // Trie is a mapping of strings of tokens T to values V.
    template <typename T, typename V>
    class Trie {
    public:
      // Initialize the root with an arbitrary token.
      Trie() : root(new TrieNode<T, V>(T{})) {}

      void insert(const std::vector<T>& str, V value) {
        TrieNode<T, V>* node = this->root;

        for (auto token : str) {
          auto it = node->children.find(token);
          if (it != node->children.end()) {
            node = it->second;
          } else {
            TrieNode<T, V>* new_node = new TrieNode<T, V>(token);
            node->children[token] = new_node;
            node = new_node;
          }
        }

        // We aren't supporting insertions of a string multiple times.
        assert(!node->end);
        node->end = true;
        node->value = value;
      }

      bool contains(const std::vector<T>& str) {
        TrieNode<T, V>* node = this->root;
        for (auto token : str) {
          auto it = node->children.find(token);
          if (it != node->children.end()) {
            node = it->second;
          } else {
            return false;
          }
        }

        return node->end;
      }

      bool prefix(const std::vector<T>& str) {
        TrieNode<T, V>* node = this->root;

        for (auto token : str) {
          auto it = node->children.find(token);
          if (it != node->children.end()) {
            node = it->second;
          } else {
            return false;
          }
        }

        return true;
      }

      // TODO (rohany): Implement remove and superstring.

    private:
      // TODO (rohany): Does this actually need to be a pointer?
      TrieNode<T, V>* root;
    };
  };
};

#endif // __LEGION_TRIE_H__