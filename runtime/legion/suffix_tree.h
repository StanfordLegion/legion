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

#ifndef __LEGION_SUFFIX_TREE_H__
#define __LEGION_SUFFIX_TREE_H__

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <iostream>

namespace Legion {
  namespace Internal {

    // This suffix tree implementation was adapted from the Python implementation
    // available in https://github.com/cceh/suffix-tree/. Note that this code
    // has a GPL license, so not sure where to go from here.
    template<typename T>
    class SuffixTree;

    template<typename T>
    class SuffixTreeNode {
    public:
      SuffixTreeNode(SuffixTreeNode* parent_, size_t start_, size_t end_, bool internal_) :
        parent(parent_), start(start_), end(end_), internal(internal_) {}

      ~SuffixTreeNode() {
        // TODO (rohany): I can't make the C++ type checker happy...
        for (auto it = children.begin(); it != children.end(); it++) {
          delete it->second;
        }
      }

      size_t depth() { return end - start; }
      bool is_internal() { return this->internal; }
      bool is_leaf() { return !this->is_internal(); }
      const std::unordered_map<T, SuffixTreeNode<T>*>& get_children() { return this->children; }
      size_t get_start() { return this->start; }
      size_t get_end() { return this->end; }

    protected:
      SuffixTreeNode<T>* split_edge(const std::vector<T>& str, size_t new_len, SuffixTreeNode<T>* child) {
        assert(this->depth() < new_len && new_len < child->depth());

        size_t new_edge_end = child->start + new_len;
        // It is always safe to shorten a path.
        SuffixTreeNode<T>* new_node = new SuffixTreeNode<T>(this, child->start, new_edge_end, true /* internal */);

        // Substitute new node.
        this->children[
            str[child->start + this->depth()]
        ] = new_node;
        new_node->children[str[new_edge_end]] = child;
        child->parent = new_node;

        return new_node;
      }

      struct FindPathResult {
        SuffixTreeNode<T>* head;
        size_t matched_len;
        SuffixTreeNode<T>* child;
      };
      FindPathResult find_path(const std::vector<T>& base_str, const std::vector<T>& query_str, size_t start, size_t end) {
        /*
        Find a path starting from this node.

        The path is absolute.

        Returns the deepest node on the path, the matched length of the path,
        and also the next deeper node if the matched length is longer than the
        string-depth of the deepest node on the path.
        */
        SuffixTreeNode<T>* node = this;
        size_t matched_len = this->depth();
        size_t max_len = end - start;

        while (matched_len < max_len) {
          // Find the edge to follow.
          assert(node->is_internal());
          // TODO (rohany): Can't figure out the right typename to make C++ happy.
          auto it = node->children.find(query_str[start + matched_len]);
          if (it != node->children.end()) {
            SuffixTreeNode<T>* child = it->second;
            // Follow the edge.
            size_t stop = std::min(child->depth(), max_len);
            while (matched_len < stop) {
              if (!std::equal_to<T>{}(base_str[child->start + matched_len], query_str[start + matched_len])) {
                break;
              }
              matched_len++;
            }
            if (matched_len < child->depth()) {
              // The path ends between node and child.
              return FindPathResult{node, matched_len, child};
            }
            // We reached another node, loop.
            node = child;
          } else {
            // No edge to follow.
            return FindPathResult{node, matched_len, nullptr};
          }
        }
        // Path exhausted.
        return FindPathResult{node, matched_len, nullptr};
      }
    private:
      SuffixTreeNode<T>* parent;
      size_t start;
      size_t end;
      bool internal;
      std::unordered_map<T, SuffixTreeNode<T>*> children;
      // Used by McCreight's algorithm.
      SuffixTreeNode<T>* suffix_link = nullptr;

      // Allow the suffix tree to modify node internal state.
      friend class SuffixTree<T>;
    };

    // Internal nodes and leaf nodes?

    template<typename T1, typename T2>
    void printvec(T1& outs, T2 vec) {
      outs << "'";
      for (auto x : vec) {
        outs << x << " ";
      }
      outs << "'";
    }

    // Suffix Tree class.
    template<typename T>
    class SuffixTree {
    public:
      // TODO (rohany): I don't know a clean way to handle this, but the user of this
      //  method must provide a "properly" formatted string to the SuffixTree. This means
      //  that the final token in `str` must be a token that is not equal to all other
      //  characters in the string.
      SuffixTree(const std::vector<T>& str) {
        #ifdef DEBUG_LEGION
        // Check that the string is properly formatted.
        for (size_t i = 0; i < str.size() - 1; i++) {
          assert(!std::equal_to<T>{}(str[i], str[str.size() - 1]));
        }
        #endif

        this->root = new SuffixTreeNode<T>(nullptr, 0, 0, true /* internal */);
        build(str);
      }

      ~SuffixTree() {
        delete this->root;
      }

      bool find(const std::vector<T>& base_str, const std::vector<T>& query_str) {
        // TODO (rohany): Unable to make C++ happy about the typename.
        auto result = this->root->find_path(base_str, query_str, 0, query_str.size());
        return result.matched_len == query_str.size();
      }

      SuffixTreeNode<T>* get_root() { return this->root; }
    private:
      void build(const std::vector<T>& str) {
        this->root->suffix_link = this->root;
        this->root->parent = this->root;

        size_t end = str.size();
        SuffixTreeNode<T>* head = this->root;
        size_t matched_len = 0;

        for (size_t start = 0; start < end; start++) {
          // Substep a.

          // debug
          // std::cout << "Start of substep A with head " << head->is_internal() << " depth=" << head->depth() << std::endl;
          // debug

          SuffixTreeNode<T>* c = head->suffix_link;
          if (c == nullptr) {
            c = head->parent->suffix_link;
          }

          // debug
          // std::cout << "Start of substep B with c " << c->is_internal() << " depth=" << c->depth() << std::endl;
          // debug

          // Substep b.
          if (matched_len > 1) {
            // We have to examine only the first symbol of each edge because we
            // already know there must be a path.
            size_t depth = matched_len - 1;
            while (c->depth() < depth) {
              assert(c->is_internal());
              size_t idx = head->start + c->depth() + 1;
              c = c->children[str[idx]];
            }
            if (c->depth() > depth) {
              // The path ended in the middle of an edge.
              assert(c->parent != nullptr);
              c = c->parent->split_edge(str, depth, c);
            }

            // debug
            // std::cout << "rescanned to depth of " << depth << std::endl;
            // debug

            assert(c->depth() == depth);
          }

          if (head->suffix_link == nullptr) {
            assert(c->is_internal());
            head->suffix_link = c;
          }

          // debug
          // std::cout << "start of substep C on c " << c->is_internal() << " depth=" << c->depth() << std::endl;
          // debug

          // Substep c.
          // Slow scan from d.
          // TODO (rohany): Can't figure out the right typename to make C++ happy.
          auto find_path_result = c->find_path(str, str, start, end);
          head = find_path_result.head;
          matched_len = find_path_result.matched_len;
          SuffixTreeNode<T>* child = find_path_result.child;

          // debug
          // std::cout << "scanned to depth of " << matched_len << std::endl;
          // debug

          if (child != nullptr) {
            // The path ended in the middle of an edge.
            head = head->split_edge(str, matched_len, child);
          }

          // debug
          // std::cout << "new head is head " << c->is_internal() << " depth=" << head->depth() << std::endl;
          // debug

          head->children[str[start + matched_len]] = new SuffixTreeNode<T>(head, start, end, false /* internal */);

          // debug
          // std::cout << "Added ";
          // printvec(std::cout, std::vector<T>(str.begin() + start, str.begin() + end));
          // std::cout << " to node ";
          // printvec(std::cout, std::vector<T>(str.begin() + head->start, str.begin() + head->end));
          // std::cout << " as " << str[start + head->depth()] << std::endl;
          // debug
        }
      }

      SuffixTreeNode<T>* root;
    };

    // Non overlapping repeats implementation.
    struct NonOverlappingRepeatsResult {
      size_t start;
      size_t end;
      size_t repeats;
    };

    // Helper functions.
    namespace {
      struct RepeatsWalkResult {
        std::vector<size_t> leaf_starts;
        size_t imin;
        size_t imax;

        void combine(const RepeatsWalkResult& other) {
          // TODO (rohany): See if we can do this in place...
          std::vector<size_t> leaf_starts;
          leaf_starts.reserve(this->leaf_starts.size() + other.leaf_starts.size());
          std::merge(
            this->leaf_starts.begin(),
            this->leaf_starts.end(),
            other.leaf_starts.begin(),
            other.leaf_starts.end(),
            std::back_inserter(leaf_starts)
          );
          this->leaf_starts = leaf_starts;
          this->imin = std::min(this->imin, other.imin);
          this->imax = std::max(this->imax, other.imax);
        }
      };
      template<typename T>
      RepeatsWalkResult walk(SuffixTreeNode<T>* node, std::vector<NonOverlappingRepeatsResult>& output, size_t min_length) {
        if (node->is_leaf()) {
          size_t start = node->get_start();
          RepeatsWalkResult result {
            .leaf_starts = std::vector<size_t>(1, start),
            .imin = start,
            .imax = start,
          };
          return result;
        } else {
          RepeatsWalkResult result {
            .leaf_starts = std::vector<size_t>(),
            .imin = std::numeric_limits<size_t>::max(),
            .imax = std::numeric_limits<size_t>::min(),
          };
          // This approach is performing O(n^2) work by computing the leaf starts at each
          // node in the tree. If this becomes expensive, there's a way to slightly optimize
          // this by only collecting all of the leaves of internal nodes that correspond to
          // repeated substrings in the first place. We'll pay extra tree traversals for this,
          // but that cost could be offset if there aren't many repeated substrings.
          for (auto it = node->get_children().begin(); it != node->get_children().end(); it++) {
            result.combine(walk(it->second, output, min_length));
          }
          // We've found a repeat. Now find how many times it actually repeats. Exclude nodes of depth
          // zero to exclude the root node.
          if (result.imin + node->depth() <= result.imax && node->depth() != 0 && node->depth() >= min_length) {
            size_t count = 0;
            assert(result.leaf_starts.size() >= 1);
            size_t current = result.leaf_starts[0];
            for (size_t i = 1; i < result.leaf_starts.size(); i++) {
              size_t next = result.leaf_starts[i];
              if (current + node->depth() <= next) {
                current = next;
                count++;
              }
            }
            // If we found at least one other valid start, then we've found a candidate.
            if (count >= 1) {
              output.push_back(NonOverlappingRepeatsResult{
                .start = node->get_start(),
                .end = node->get_end(),
                .repeats = count + 1,
              });
            }
          }
          return result;
        }
      }
    }

    // The input string must also be formatted correctly for the suffix tree (unique last character).
    template<typename T>
    std::vector<NonOverlappingRepeatsResult> compute_longest_nonoverlapping_repeats(const std::vector<T>& str, size_t min_length = 0) {
      SuffixTree<T> tree(str);
      std::vector<NonOverlappingRepeatsResult> result;
      walk(tree.get_root(), result, min_length);
      std::sort(result.begin(), result.end(), [](const NonOverlappingRepeatsResult& left, const NonOverlappingRepeatsResult& right) {
        // Note: using > instead of < to sort in descending order.
        return std::make_pair((left.end - left.start) * left.repeats, left.repeats) >
               std::make_pair((right.end - right.start) * right.repeats, right.repeats);
      });
      return result;
    }
  };
};

#endif // __LEGION_SUFFIX_TREE_H__