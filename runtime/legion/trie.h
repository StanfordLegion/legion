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
      TrieNode(T token_, TrieNode<T, V>* parent_) : token(token_), end(false), parent(parent_) {}
      ~TrieNode() {
        for (auto it : this->children) {
          delete it.second;
        }
      }
      const std::unordered_map<T, TrieNode<T, V>*>& get_children() const { return this->children; }
      TrieNode<T, V>* get_parent() const { return this->parent; }
      V& get_value() { return this->value; }
      const V& get_value() const { return this->value; }
      bool get_end() const { return this->end; }
      T get_token() const { return this->token; }
    private:
      friend class Trie<T, V>;
      T token;
      V value;
      bool end;
      TrieNode<T, V>* parent;
      std::unordered_map<T, TrieNode<T, V>*> children;
    };

    // TrieQueryResult is a non-template struct that contains
    // return information for the Trie::query method.
    struct TrieQueryResult {
      bool prefix = false;
      bool contains = false;
      bool superstring = false;
      // superstring_match is set only when superstring=true. It
      // returns the length of the matched prefix that the queried
      // string is a superstring of.
      uint64_t superstring_match = 0;
    };

    // Trie is a mapping of strings of tokens T to values V.
    template <typename T, typename V>
    class Trie {
    public:
      // Initialize the root with an arbitrary token.
      Trie() : root(T{}, nullptr) {}

      template<typename ITER>
      void insert(ITER start, ITER end, V value) {
        TrieNode<T, V>* node = &this->root;

        for (auto tokitr = start; tokitr != end; tokitr++) {
          auto token = *tokitr;
          auto it = node->children.find(token);
          if (it != node->children.end()) {
            node = it->second;
          } else {
            TrieNode<T, V>* new_node = new TrieNode<T, V>(token, node);
            node->children[token] = new_node;
            node = new_node;
          }
        }

        // We aren't supporting insertions of a string multiple times.
        assert(!node->end);
        node->end = true;
        node->value = value;
      }

      // query is a method that performs a prefix, containment and
      // superstring query on the trie in a single traversal.
      template<typename ITER>
      TrieQueryResult query(ITER start, ITER end) {
        TrieNode<T, V>* node = &this->root;
        size_t matched_toks = 0;
        for (auto tokitr = start; tokitr != end; tokitr++) {
          auto token = *tokitr;
          auto it = node->children.find(token);
          if (it != node->children.end()) {
            node = it->second;
            matched_toks++;
          } else {
            break;
          }
        }
        TrieQueryResult result;
        // If we matched all of our input string, then we're
        // a prefix of some string in the trie.
        result.prefix = matched_toks == (end - start);
        // If we matched all of the input string and ended on an
        // end node, then we found an exact match.
        result.contains = result.prefix && node->end;
        // If we've ended at a string in the trie but still have
        // tokens left to process in the input string, then our
        // input string is a super-string of a string in the trie.
        result.superstring = node->end && (matched_toks < (end - start));
        result.superstring_match = matched_toks;
        return result;
      }

      template<typename ITER>
      bool contains(ITER start, ITER end) {
        TrieNode<T, V>* node = &this->root;
        for (auto tokitr = start; tokitr != end; tokitr++) {
          auto token = *tokitr;
          auto it = node->children.find(token);
          if (it != node->children.end()) {
            node = it->second;
          } else {
            return false;
          }
        }

        return node->end;
      }

      template<typename ITER>
      bool prefix(ITER start, ITER end) {
        TrieNode<T, V>* node = &this->root;

        for (auto tokitr = start; tokitr != end; tokitr++) {
          auto token = *tokitr;
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

      TrieNode<T, V>* get_root() { return &root; }
    private:
      TrieNode<T, V> root;
    };
  };
};

#endif // __LEGION_TRIE_H__