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
#include <cassert>
#include <limits>
#include <unordered_map>
#include <unordered_set>
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
    class SuffixTreeNode;

    // Custom allocator for SuffixTreeNodes.
    template<typename T>
    class SuffixTreeNodeAllocator {
    public:
      virtual SuffixTreeNode<T>* alloc(SuffixTreeNode<T>* parent, size_t start, size_t end, bool internal) = 0;
    };

    template<typename T>
    class SuffixTreeNode {
    public:
      // Default constructor as well, so that it can be put in vectors.
      SuffixTreeNode() : parent(nullptr), start(0), end(0), internal(false) {}
      SuffixTreeNode(SuffixTreeNode* parent_, size_t start_, size_t end_, bool internal_) :
        parent(parent_), start(start_), end(end_), internal(internal_) {}

      size_t depth() { return end - start; }
      bool is_internal() { return this->internal; }
      bool is_leaf() { return !this->is_internal(); }
      const std::unordered_map<T, SuffixTreeNode<T>*>& get_children() { return this->children; }
      size_t get_start() { return this->start; }
      size_t get_end() { return this->end; }

    protected:
      SuffixTreeNode<T>* split_edge(SuffixTreeNodeAllocator<T>& alloc, const std::vector<T>& str, size_t new_len, SuffixTreeNode<T>* child) {
        assert(this->depth() < new_len && new_len < child->depth());

        size_t new_edge_end = child->start + new_len;
        // It is always safe to shorten a path.
        SuffixTreeNode<T>* new_node = alloc.alloc(this, child->start, new_edge_end, true /* internal*/);

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
    class SuffixTree : public SuffixTreeNodeAllocator<T> {
    public:
      // TODO (rohany): I don't know a clean way to handle this, but the user of this
      //  method must provide a "properly" formatted string to the SuffixTree. This means
      //  that the final token in `str` must be a token that is not equal to all other
      //  characters in the string.
      SuffixTree(const std::vector<T>& str)
          // The maximum number of nodes in a suffix tree is 2*N.
        : node_alloc(2 * str.size() + 1),
          node_alloc_idx(0) {
        #ifdef DEBUG_LEGION
        // Check that the string is properly formatted.
        for (size_t i = 0; i < str.size() - 1; i++) {
          assert(!std::equal_to<T>{}(str[i], str[str.size() - 1]));
        }
        #endif

        this->root = this->alloc(nullptr, 0, 0, true /* internal */);
        build(str);
      }

      bool find(const std::vector<T>& base_str, const std::vector<T>& query_str) {
        // TODO (rohany): Unable to make C++ happy about the typename.
        auto result = this->root->find_path(base_str, query_str, 0, query_str.size());
        return result.matched_len == query_str.size();
      }

      SuffixTreeNode<T>* get_root() { return this->root; }

      SuffixTreeNode<T>* alloc(SuffixTreeNode<T>* parent, size_t start, size_t end, bool internal) override {
        assert(this->node_alloc_idx < this->node_alloc.size());
        return new (&this->node_alloc[this->node_alloc_idx++]) SuffixTreeNode<T>(parent, start, end, internal);
      }

    private:
      void build(const std::vector<T>& str) {
        this->root->suffix_link = this->root;
        this->root->parent = this->root;

        size_t end = str.size();
        SuffixTreeNode<T>* head = this->root;
        size_t matched_len = 0;

        for (size_t start = 0; start < end; start++) {
          // Substep a.

          SuffixTreeNode<T>* c = head->suffix_link;
          if (c == nullptr) {
            c = head->parent->suffix_link;
          }

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
              c = c->parent->split_edge(*this, str, depth, c);
            }

            assert(c->depth() == depth);
          }

          if (head->suffix_link == nullptr) {
            assert(c->is_internal());
            head->suffix_link = c;
          }

          // Substep c.
          // Slow scan from d.
          // TODO (rohany): Can't figure out the right typename to make C++ happy.
          auto find_path_result = c->find_path(str, str, start, end);
          head = find_path_result.head;
          matched_len = find_path_result.matched_len;
          SuffixTreeNode<T>* child = find_path_result.child;

          if (child != nullptr) {
            // The path ended in the middle of an edge.
            head = head->split_edge(*this, str, matched_len, child);
          }

          head->children[str[start + matched_len]] = this->alloc(head, start, end, false /* internal */);
        }
      }

      SuffixTreeNode<T>* root;
      // Allocation scratch space for suffix tree nodes.
      std::vector<SuffixTreeNode<T>> node_alloc;
      size_t node_alloc_idx;
    };

    // Non overlapping repeats implementation.
    struct NonOverlappingRepeatsResult {
      size_t start;
      size_t end;
      size_t repeats;
    };

    // The algorithm that should be used to compute the repeats.
    enum NonOverlappingAlgorithm {
      SUFFIX_TREE_WALK = 0,
      QUICK_MATCHING_SUBSTRINGS = 1,
      NO_ALGORITHM,
    };
    NonOverlappingAlgorithm parse_non_overlapping_algorithm(const std::string&);
    const char* non_overlapping_algorithm_to_string(NonOverlappingAlgorithm);

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

    // Suffix array construction in O(n*log n) time.
    // The code has been implemented based on the explanations
    // from here http://www.cs.cmu.edu/~15451-f20/LectureNotes/lec25-suffarray.pdf,
    // with special treatment of radix sort to make it O(n*log n).
    template<typename T>
    void suffix_array(const std::vector<T>& str,
                      std::vector<size_t>& sarray,
                      std::vector<int>& surrogate) {
      size_t n = str.size();
      using triple = std::tuple<int, int, size_t>;

      // First round - O(n log n) sort
      std::vector<triple> w(n);
      for(size_t i = 0; i < n; i++)
        w[i] = std::make_tuple(str[i], i + 1 < n ? str[i + 1] : -1, i);
      std::sort(w.begin(), w.end());

      size_t shift = 1;
      std::vector<size_t> count(n+2);
      std::vector<triple> tmp(n);
      while(true){
        int v = 0;

        // Construct surrogate
        size_t x0 = std::get<0>(w[0]);
        size_t x1 = std::get<1>(w[0]);
        surrogate[std::get<2>(w[0])] = 0;
        for(size_t i = 1; i < n; i++){
          if(x0 != std::get<0>(w[i]) || x1 != std::get<1>(w[i]))
            v++;
          surrogate[std::get<2>(w[i])] = v;
          x0 = std::get<0>(w[i]);
          x1 = std::get<1>(w[i]);
        }

        // End if done
        if(v >= n-1)
          break;
        shift *= 2;

        // Update sort table
        for(size_t i = 0; i < n; i++)
          w[i] = std::make_tuple(surrogate[i],
                                 (i + shift) < n ? surrogate[i + shift] : -1, i);
        // Radix sort O(n) - rolled out, 2 digits. The index in the third
        // element is not needed to be sorted. The radix sort algorithm
        // sorts two digits corresponding to the first and second element in
        // the triple. See for instance https://hacktechhub.com/radix-sort/ for
        // the general idea of radix sort.
        for(size_t i = 0; i < v + 2; i++) // Clear the counts
          count[i] = 0;
        for(size_t i = 0; i < n; i++)     // Count the frequency
          count[std::get<1>(w[i])+1]++;
        for(size_t i = 1; i < v+2; i++)   // Update count to contain actual positions
          count[i] += count[i - 1];
        for(int i = n - 1; i >= 0; i--)   // Construct output array based on second digit
          tmp[(count[std::get<1>(w[i]) + 1]--) - 1] = w[i];
        for(size_t i = 0; i < v+2; i++)   // Clear count. Next, sort on first digit.
          count[i] = 0;
        for(size_t i = 0; i < n; i++)     // The source is in tmp. Count freq. on first digit.
          count[std::get<0>(tmp[i])+1]++;
        for(size_t i = 1; i < v+2; i++)   // Update count to contain actual positions.
          count[i] += count[i - 1];
        for(int i = n - 1; i >= 0; i--)   // Output to array w from tmp.
          w[(count[std::get<0>(tmp[i]) + 1]--) - 1] = tmp[i];
      }
      // Reconstruct the suffix array
      for(size_t i = 0; i < n; i++)
        sarray[i] = std::get<2>(w[i]);
    }

    // Computes the LCP in O(n) time. This is Kasai's algorithm. See e.g.,
    // http://www.cs.cmu.edu/~15451-f20/LectureNotes/lec25-suffarray.pdf for an explanation.
    // The original paper can be found here:
    // https://link.springer.com/chapter/10.1007/3-540-48194-X_17
    template<typename T>
    std::vector<size_t> compute_lcp(const std::vector<T>& str,
                                    const std::vector<size_t>& sarray,
                                    const std::vector<int>& surrogate) {
      size_t n = str.size();
      int k = 0;
      std::vector<size_t> lcp(n, 0);
      for(size_t i = 0; i < n; i++){
        if(surrogate[i] == n - 1)
          k = 0;
        else{
          size_t j = sarray[surrogate[i] + 1];
          for(; i + k < n && j + k < n && str[i + k] == str[j + k]; k++);
          lcp[surrogate[i]] = k;
          k = std::max(k - 1, 0);
        }
      }
      return lcp;
    }

    // The function computes non-overlapping matching substrings in O(n log n) time.
    // This is a new algorithm designed by David Broman in 2024.
    // Please see the following Git repo for a reference implementation and a short explanation:
    // https://github.com/david-broman/matching-substrings
    template<typename T>
    std::vector<NonOverlappingRepeatsResult>
    quick_matching_substrings(const std::vector<T>& str,
                              size_t min_length,
                              const std::vector<size_t>& sarray,
                              const std::vector<size_t>& lcp) {
      std::vector<NonOverlappingRepeatsResult> result;
      size_t le = str.size();
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
          if(pre_l != l1)
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
      for(size_t i = 0; i < le; i++){
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
        if(le2 >= min_length && k >= next_k && !(flag[k]) && !(flag[k + le2 - 1])){
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

    // The input string must also be formatted correctly for the suffix tree (unique last character).
    template<typename T>
    std::vector<NonOverlappingRepeatsResult> compute_longest_nonoverlapping_repeats(
        const std::vector<T>& str,
        size_t min_length = 0,
        NonOverlappingAlgorithm algorithm = NonOverlappingAlgorithm::SUFFIX_TREE_WALK
    ) {
      switch (algorithm) {
        case NonOverlappingAlgorithm::SUFFIX_TREE_WALK: {
          SuffixTree<T> tree(str);
          std::vector<NonOverlappingRepeatsResult> result;
          walk(tree.get_root(), result, min_length);
          std::sort(result.begin(), result.end(), [](const NonOverlappingRepeatsResult& left, const NonOverlappingRepeatsResult& right) {
            // Note: using > instead of < to sort in descending order.
            return std::make_pair((left.end - left.start) * left.repeats, left.repeats) >
                   std::make_pair((right.end - right.start) * right.repeats, right.repeats);
          });
          // Filter the result to remove substrings of longer repeats from consideration.
          size_t copyidx = 0;
          std::unordered_set<size_t> ends;
          for (auto res : result) {
            if (ends.find(res.end) != ends.end()) {
              continue;
            }
            result[copyidx] = res;
            copyidx++;
            ends.insert(res.end);
          }
          // Erase the unused pieces of result.
          result.erase(result.begin() + copyidx, result.end());
          return result;
        }
      case NonOverlappingAlgorithm::QUICK_MATCHING_SUBSTRINGS: {
        if(str.size() < 2)
          return {};
        std::vector<size_t> sarray(str.size());
        std::vector<int> surrogate(str.size());
        suffix_array(str, sarray, surrogate);
        std::vector<size_t> lcp = compute_lcp(str, sarray, surrogate);
        std::vector<NonOverlappingRepeatsResult> result =
          quick_matching_substrings(str, min_length, sarray, lcp);
        return result;
      }
        default:
          assert(false);
      }
      return {};
    }
  };
};

#endif // __LEGION_SUFFIX_TREE_H__
