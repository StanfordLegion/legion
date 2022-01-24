/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// templated interval tree

#ifndef REALM_INTERVAL_TREE_H
#define REALM_INTERVAL_TREE_H

#include <vector>
#include <set>

namespace Realm {

  template <typename IT, typename LT>
  class IntervalTree {
  public:
    IntervalTree(void);
    ~IntervalTree(void);

    bool empty(void) const;
    size_t size(void) const;

    void add_interval(IT iv_start, IT iv_end, LT iv_label, bool defer = true);

    template <typename IR>
    void add_intervals(const IR& iv_ranges, LT iv_label, bool defer = true);

    void remove_interval(IT iv_start, IT iv_end, LT iv_label);

    template <typename IR>
    void remove_intervals(const IR& iv_ranges, LT iv_label);

    void remove_by_label(LT iv_label);

    void construct_tree(bool rebuild_completely = false);

    template <typename MARKER>
    void test_interval(IT iv_start, IT iv_end, MARKER& marker) const;

    void test_interval(IT iv_start, IT iv_end, std::vector<bool>& labels_found) const;
    void test_interval(IT iv_start, IT iv_end, std::set<LT>& labels_found) const;

    template <typename IR, typename MARKER>
    void test_intervals(const IR& iv_ranges, MARKER& marker) const;

    template <typename IR>
    void test_intervals(const IR& iv_ranges, std::vector<bool>& labels_found) const;

    template <typename IR>
    void test_intervals(const IR& iv_ranges, std::set<LT>& labels_found) const;

    template <typename IR, typename MARKER>
    void test_sorted_intervals(const IR& iv_ranges, MARKER& marker) const;

    template <typename IR>
    void test_sorted_intervals(const IR& iv_ranges, std::vector<bool>& labels_found) const;

    template <typename IR>
    void test_sorted_intervals(const IR& iv_ranges, std::set<LT>& labels_found) const;

    struct TreeNode {
      IT split_val;
      TreeNode *left, *right;
      std::vector<IT> starts, ends;
      std::vector<LT> labels;
      std::vector<int> sorted_by_start, sorted_by_end;

      TreeNode(IT _split_val);
      ~TreeNode(void);

      static TreeNode *build_tree(const std::vector<IT>& pending_starts,
				  const std::vector<IT>& pending_ends,
				  const std::vector<LT>& pending_labels);

      template <typename MARKER>
      void test_interval(IT iv_start, IT iv_end, MARKER& marker) const;
				  
      template <typename IR, typename MARKER>
      void test_sorted_intervals(const IR& iv_ranges,
				 int pos, int count,
				 MARKER& marker) const;
				  
      void repopulate_pending(IntervalTree<IT,LT> *tree);
    };

  protected:
    std::vector<IT> pending_starts, pending_ends;
    std::vector<LT> pending_labels;
    TreeNode *root;
    size_t count;
  };
      
};

#include "realm/interval_tree.inl"

#endif // ifndef REALM_INTERVAL_TREE_H

