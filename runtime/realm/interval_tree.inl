/* Copyright 2015 Stanford University, NVIDIA Corporation
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

// nop, but helps IDEs
#include "interval_tree.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class IntervalTree<IT,LT>::TreeNode
  
  template <typename IT, typename LT>
  inline IntervalTree<IT,LT>::TreeNode::TreeNode(IT _split_val)
    : split_val(_split_val), left(0), right(0)
  {}

  template <typename IT, typename LT>
  inline IntervalTree<IT,LT>::TreeNode::~TreeNode(void)
  {
    // recursively delete child subtrees (if any)
    delete left;
    delete right;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class VectorMarker<IT,LT>

  template <typename IT, typename LT>
  class VectorMarker {
  public:
    VectorMarker(std::vector<bool>& _labels_found) : labels_found(_labels_found) {}
    void mark_overlap(IT iv_start, IT iv_end, LT iv_label)
    {
      unsigned idx = iv_label;
      if(idx >= labels_found.size())
	labels_found.resize(idx + 1, false);
      labels_found[idx] = true;
    }
  protected:
    std::vector<bool>& labels_found;
  };


  ////////////////////////////////////////////////////////////////////////
  //
  // class SetMarker<IT,LT>

  template <typename IT, typename LT>
  class SetMarker {
  public:
    SetMarker(std::set<LT>& _labels_found) : labels_found(_labels_found) {}
    void mark_overlap(IT iv_start, IT iv_end, LT iv_label)
    {
      labels_found.insert(iv_label);
    }
  protected:
    std::set<LT>& labels_found;
  };


  ////////////////////////////////////////////////////////////////////////
  //
  // class IntervalTree<IT,LT>

  template <typename IT, typename LT>
  inline IntervalTree<IT,LT>::IntervalTree(void)
    : root(0), count(0)
  {}

  template <typename IT, typename LT>
  inline IntervalTree<IT,LT>::~IntervalTree(void)
  {
    delete root;
  }

  template <typename IT, typename LT>
  inline bool IntervalTree<IT,LT>::empty(void) const
  {
    return count == 0;
  }

  template <typename IT, typename LT>
  inline size_t IntervalTree<IT,LT>::size(void) const
  {
    return count;
  }

  template <typename IT, typename LT>
  inline void IntervalTree<IT,LT>::add_interval(IT iv_start, IT iv_end, LT iv_label, bool defer /*= true*/)
  {
    pending_starts.push_back(iv_start);
    pending_ends.push_back(iv_end);
    pending_labels.push_back(iv_label);
    if(!defer)
      construct_tree(false /* incremental rebuild*/);
  }

  template <typename IT, typename LT>
  template <typename IR>
  inline void IntervalTree<IT,LT>::add_intervals(const IR& iv_ranges, LT iv_label, bool defer /*= true*/)
  {
    size_t old_count = pending_starts.size();
    size_t new_count = iv_ranges.size();
    pending_starts.resize(old_count + new_count);
    pending_ends.resize(old_count + new_count);
    pending_labels.resize(old_count + new_count, iv_label); // fill all with same label
    for(size_t i = 0; i < new_count; i++) {
      pending_starts[old_count + i] = iv_ranges.start(i);
      pending_ends[old_count + i] = iv_ranges.end(i);
    }
    if(!defer)
      construct_tree(false /* incremental rebuild*/);
  }

  template <typename IT, typename LT>
  inline void IntervalTree<IT,LT>::remove_interval(IT iv_start, IT iv_end, LT iv_label)
  {
    assert(0);
  }

  template <typename IT, typename LT>
  template <typename IR>
  inline void IntervalTree<IT,LT>::remove_intervals(const IR& iv_ranges, LT iv_label)
  {
    assert(0);
  }

  template <typename IT, typename LT>
  inline void IntervalTree<IT,LT>::remove_by_label(LT iv_label)
  {
    assert(0);
  }

  template <typename IT, typename LT>
  inline void IntervalTree<IT,LT>::construct_tree(bool rebuild_completely /*= false*/)
  {
    if(rebuild_completely && (root != 0)) {
      root->repopulate_pending(this);
      delete root;
      root = 0;
    }

    // TODO
    assert(0);
  }

  template <typename IT, typename LT>
  template <typename MARKER>
  inline void IntervalTree<IT,LT>::test_interval(IT iv_start, IT iv_end, MARKER& marker) const
  {
    // TODO
    assert(0);
  }
    
  template <typename IT, typename LT>
  inline void IntervalTree<IT,LT>::test_interval(IT iv_start, IT iv_end, std::vector<bool>& labels_found) const
  {
    test_interval(iv_start, iv_end, VectorMarker<IT,LT>(labels_found));
  }

  template <typename IT, typename LT>
  inline void IntervalTree<IT,LT>::test_interval(IT iv_start, IT iv_end, std::set<LT>& labels_found) const
  {
    test_interval(iv_start, iv_end, SetMarker<IT,LT>(labels_found));
  }

  template <typename IT, typename LT>
  template <typename IR, typename MARKER>
  inline void IntervalTree<IT,LT>::test_intervals(const IR& iv_ranges, MARKER& marker) const
  {
    // unsorted (and possibly overlapping) intervals have to be tested separately
    for(size_t i = 0; i < iv_ranges.size(); i++)
      test_interval(iv_ranges.start(i), iv_ranges.end(i), marker);
  }

  template <typename IT, typename LT>
  template <typename IR>
  void IntervalTree<IT,LT>::test_intervals(const IR& iv_ranges, std::vector<bool>& labels_found) const
  {
    test_intervals(iv_ranges, VectorMarker<IT,LT>(labels_found));
  }

  template <typename IT, typename LT>
  template <typename IR>
  void IntervalTree<IT,LT>::test_intervals(const IR& iv_ranges, std::set<LT>& labels_found) const
  {
    test_intervals(iv_ranges, SetMarker<IT,LT>(labels_found));
  }

  template <typename IT, typename LT>
  template <typename IR, typename MARKER>
  inline void IntervalTree<IT,LT>::test_sorted_intervals(const IR& iv_ranges, MARKER& marker) const
  {
    // simultaneous walk of interval tree and implied binary tree of iv_ranges
    assert(0);
  }

  template <typename IT, typename LT>
  template <typename IR>
  void IntervalTree<IT,LT>::test_sorted_intervals(const IR& iv_ranges, std::vector<bool>& labels_found) const
  {
    test_sorted_intervals(iv_ranges, VectorMarker<IT,LT>(labels_found));
  }

  template <typename IT, typename LT>
  template <typename IR>
  void IntervalTree<IT,LT>::test_sorted_intervals(const IR& iv_ranges, std::set<LT>& labels_found) const
  {
    test_sorted_intervals(iv_ranges, SetMarker<IT,LT>(labels_found));
  }


}; // namespace Realm
