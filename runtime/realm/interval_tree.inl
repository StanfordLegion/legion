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

// nop, but helps IDEs
#include "realm/interval_tree.h"

#include <cassert>

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

  namespace {
    template <typename T>
    class StartSorter {
    public:
      StartSorter(const std::vector<T>& _vals) : vals(_vals) {}
      bool operator()(int i, int j) { return vals[i] < vals[j]; }
    protected:
      const std::vector<T>& vals;
    };

    template <typename T>
    class EndSorter {
    public:
      EndSorter(const std::vector<T>& _vals) : vals(_vals) {}
      bool operator()(int i, int j) { return vals[i] > vals[j]; }
    protected:
      const std::vector<T>& vals;
    };
  };

  template <typename IT, typename LT>
  /*static*/ inline typename IntervalTree<IT,LT>::TreeNode *
       IntervalTree<IT,LT>::TreeNode::build_tree(const typename std::vector<IT>& pending_starts,
						 const typename std::vector<IT>& pending_ends,
						 const typename std::vector<LT>& pending_labels)
  {
    size_t count = pending_starts.size();
    if(!count) return 0;

#ifdef DEBUG_INTERVALS
    std::cout << "Pending:";
    for(size_t i = 0; i < count; i++)
      std::cout << " " << pending_starts[i] << ".." << pending_ends[i] << "=" << pending_labels[i];
    std::cout << "\n";
#endif

    // to pick cut points, we work with sorted start points and sorted end points
    std::vector<IT> sorted_starts(pending_starts);
    std::vector<IT> sorted_ends(pending_ends);
    std::sort(sorted_starts.begin(), sorted_starts.end());
    std::sort(sorted_ends.begin(), sorted_ends.end());

    // quick version: pick a value that is sort of in the middle
    size_t j = count >> 1;
    // careful!  can't average by sum/2 without getting overflow
    IT cut = sorted_ends[j] + (sorted_starts[count - 1 - j] - sorted_ends[j]) / 2;
#ifdef DEBUG_INTERVALS
    std::cout << "CUT = " << cut << " (" << j << ")\n";
#endif
#if 0
    // now we want to find a cut point C that results in a roughly even number of intervals
    //  completely below (i.e. sorted_ends[j] < C) and above (sorted_starts[n - 1 - j] >= C)
    // we do this by finding the largest j satisfying both of those above
    // since we must have j < n/2, we do a binary search on just [0,n/2]
    size_t j_lo = 0;
    size_t j_hi = count >> 1;
    while(j_lo < j_hi) {
      size_t j_mid = (j_lo + j_hi) >> 1;
      if(sorted_ends[j_mid] < sorted_starts[count - j_mid - 1]) {
	// current value is ok, try going higher
	j_lo = j_mid;
      } else {
	// current value is not ok, move lower
	j_hi = j_mid - 1;
      }
    }
#ifdef DEBUG_INTERVALS
    std::cout << "cut: " << j_lo << ": " << sorted_ends[j_lo] << " < " << sorted_starts[count - j_lo - 1] << "\n";
#endif
#endif

    TreeNode *node = new TreeNode(cut);

    // walk over the pending intervals and put them on one side or the other or add to our node
    std::vector<IT> left_starts, right_starts, left_ends, right_ends;
    std::vector<LT> left_labels, right_labels;
    size_t num_local = 0;
    for(size_t i = 0; i < count; i++) {
      if(pending_ends[i] < cut) {
	left_starts.push_back(pending_starts[i]);
	left_ends.push_back(pending_ends[i]);
	left_labels.push_back(pending_labels[i]);
	continue;
      }
      if(pending_starts[i] > cut) {
	right_starts.push_back(pending_starts[i]);
	right_ends.push_back(pending_ends[i]);
	right_labels.push_back(pending_labels[i]);
	continue;
      }
      // overlap - add it to this node
      // note that this data structure effectively splits the interval into two pieces -
      //  the part below cut (if any) and the part above (if any)
      node->starts.push_back(pending_starts[i]);
      node->ends.push_back(pending_ends[i]);
      node->labels.push_back(pending_labels[i]);
      num_local++;
    }

    // now build and sort the permutations based on start (or end)
    if(num_local > 0) {
      node->sorted_by_start.resize(num_local);
      node->sorted_by_end.resize(num_local);
      for(size_t i = 0; i < num_local; i++) {
	node->sorted_by_start[i] = i;
	node->sorted_by_end[i] = i;
      }
      std::sort(node->sorted_by_start.begin(), node->sorted_by_start.end(),
		StartSorter<IT>(node->starts));
      std::sort(node->sorted_by_end.begin(), node->sorted_by_end.end(),
		EndSorter<IT>(node->ends));
    }

    node->left = build_tree(left_starts, left_ends, left_labels);
    node->right = build_tree(right_starts, right_ends, right_labels);

    return node;
  }

  template <typename IT, typename LT>
  inline void IntervalTree<IT,LT>::TreeNode::repopulate_pending(IntervalTree<IT,LT> *tree)
  {
    tree->pending_starts.insert(tree->pending_starts.end(), starts.begin(), starts.end());
    tree->pending_ends.insert(tree->pending_ends.end(), ends.begin(), ends.end());
    tree->pending_labels.insert(tree->pending_labels.end(), labels.begin(), labels.end());
    if(left)
      left->repopulate_pending(tree);
    if(right)
      right->repopulate_pending(tree);
  }

  template <typename IT, typename LT>
  template <typename MARKER>
  inline void IntervalTree<IT,LT>::TreeNode::test_interval(IT iv_start, IT iv_end,
							   MARKER& marker) const
  {
#ifdef DEBUG_INTERVALS
    std::cout << "CHECK " << split_val << " " << iv_start << " " << iv_end << " " << left << " " << right << "\n";
#endif

    // three cases to test on our local intervals:
    // 1) interval is entirely below our split point
    if(iv_end < split_val) {
      // walk our intervals by start point, marking things that overlap the end
      for(size_t i = 0; i < sorted_by_start.size(); i++) {
	if(starts[sorted_by_start[i]] > iv_end) 
	  break;  // no more will match either
	marker.mark_overlap(starts[sorted_by_start[i]],
			    ends[sorted_by_start[i]],
			    labels[sorted_by_start[i]]);
      }
    } 
    // 2) interval is entirely above our split point
    else if(iv_start > split_val) {
      // walk our intervals by end point, marking things that overlap the start
      for(size_t i = 0; i < sorted_by_end.size(); i++) {
	if(ends[sorted_by_end[i]] < iv_start) 
	  break;  // no more will match either
	marker.mark_overlap(starts[sorted_by_end[i]],
			    ends[sorted_by_end[i]],
			    labels[sorted_by_end[i]]);
      }
    }
    // 3) interval covers our split point
    else {
      // easy - all intervals overlap
      for(size_t i = 0; i < starts.size(); i++)
	marker.mark_overlap(starts[i], ends[i], labels[i]);
    }

    // now the recursion:

    // 4) left subtree if it exists and any part of the test interval is below our split
    if(left && (iv_start < split_val))
      left->test_interval(iv_start, iv_end, marker);

    // 5) right subtree if it exists and any part of the test interval is above our split
    if(right && (iv_end > split_val))
      right->test_interval(iv_start, iv_end, marker);
  }

  template <typename IT, typename LT>
  template <typename IR, typename MARKER>
  inline void IntervalTree<IT,LT>::TreeNode::test_sorted_intervals(const IR& iv_ranges,
								   int offset, int count,
								   MARKER& marker) const
  {
    assert(count > 0);

#ifdef DEBUG_INTERVALS
    std::cout << "START " << count;
    for(size_t i = 0; i < count; i++)
      std::cout << " " << iv_ranges.start(offset + i) << ".." << iv_ranges.end(offset + i);
    std::cout << "\n";
#endif
    // search to find the first range that includes our split or is above it
    int lo = 0;
    int hi = count - 1;
    while(lo <= hi) {
      int mid = (lo + hi) >> 1;
      if(split_val < iv_ranges.start(mid + offset))
	hi = mid - 1;
      else if(split_val > iv_ranges.end(mid + offset))
	lo = mid + 1;
      else {
	// we have an overlap
#ifdef DEBUG_INTERVALS
	std::cout << "OVERLAP " << lo << " " << mid << " " << hi << " " << split_val << " " << iv_ranges.start(mid + offset) << " " << iv_ranges.end(mid + offset) << "\n";
#endif
	lo = hi = mid;
	break;
      }
    }
#ifdef DEBUG_INTERVALS
    std::cout << "CHECK " << split_val << " / " << hi << " ";
    if(hi >= 0)
      std::cout << iv_ranges.end(hi + offset);
    else
      std::cout << "xx";
    std::cout << " / " << lo << " ";
    if(lo < count)
      std::cout << iv_ranges.start(lo + offset);
    else
      std::cout << "xx";
    std::cout << "\n";
#endif

    if(lo == hi) {
      // one test interval covers the split value, so all of our intervals are overlapped
      for(size_t i = 0; i < starts.size(); i++)
	marker.mark_overlap(starts[i], ends[i], labels[i]);
    } else {
      // our split point isn't covered, but test our intervals against the highest test inverval
      //  below and the lowest test interval above
      if(hi >= 0) {
	IT last_end = iv_ranges.end(hi + offset);
	for(size_t i = 0; i < sorted_by_start.size(); i++) {
	  if(starts[sorted_by_start[i]] > last_end)
	    break;  // no more will match either
	  marker.mark_overlap(starts[sorted_by_start[i]],
			      ends[sorted_by_start[i]],
			      labels[sorted_by_start[i]]);
	}
      }
      if(lo < count) {
	IT first_start = iv_ranges.start(lo + offset);
	for(size_t i = 0; i < sorted_by_end.size(); i++) {
	  if(ends[sorted_by_end[i]] < first_start) 
	    break;  // no more will match either
	  marker.mark_overlap(starts[sorted_by_end[i]],
			      ends[sorted_by_end[i]],
			      labels[sorted_by_end[i]]);
	}
      }
    }

    // if any invervals on-or-below, recurse the left subtree
    if(left && (hi >= 0))
      left->test_sorted_intervals(iv_ranges, offset, hi + 1, marker);

    // if any intervals on-or-above, recurse the right subtree
    if(right && (lo < count))
      right->test_sorted_intervals(iv_ranges, offset + lo, count - lo, marker);
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
#ifdef DEBUG_INTERVALS
      std::cout << "MARK " << iv_start << " " << iv_end << " " << iv_label << "\n";
#endif
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
    // ignore empty intervals
    if(iv_start > iv_end)
      return;
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
    pending_starts.reserve(old_count + new_count);
    pending_ends.reserve(old_count + new_count);
    pending_labels.reserve(old_count + new_count);
    for(size_t i = 0; i < new_count; i++) {
      if(iv_ranges.start(i) > iv_ranges.end(i)) continue;
      pending_starts.push_back(iv_ranges.start(i));
      pending_ends.push_back(iv_ranges.end(i));
      pending_labels.push_back(iv_label);
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

    root = TreeNode::build_tree(pending_starts, pending_ends, pending_labels);
    pending_starts.clear();
    pending_ends.clear();
    pending_labels.clear();
  }

  template <typename IT, typename LT>
  template <typename MARKER>
  inline void IntervalTree<IT,LT>::test_interval(IT iv_start, IT iv_end, MARKER& marker) const
  {
    if(root)
      root->test_interval(iv_start, iv_end, marker);
  }
    
  template <typename IT, typename LT>
  inline void IntervalTree<IT,LT>::test_interval(IT iv_start, IT iv_end, std::vector<bool>& labels_found) const
  {
    VectorMarker<IT,LT> marker(labels_found);
    test_interval(iv_start, iv_end, marker);
  }

  template <typename IT, typename LT>
  inline void IntervalTree<IT,LT>::test_interval(IT iv_start, IT iv_end, std::set<LT>& labels_found) const
  {
    SetMarker<IT,LT> marker(labels_found);
    test_interval(iv_start, iv_end, marker);
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
    VectorMarker<IT,LT> marker(labels_found);
    test_intervals(iv_ranges, marker);
  }

  template <typename IT, typename LT>
  template <typename IR>
  void IntervalTree<IT,LT>::test_intervals(const IR& iv_ranges, std::set<LT>& labels_found) const
  {
    SetMarker<IT,LT> marker(labels_found);
    test_intervals(iv_ranges, marker);
  }

  template <typename IT, typename LT>
  template <typename IR, typename MARKER>
  inline void IntervalTree<IT,LT>::test_sorted_intervals(const IR& iv_ranges, MARKER& marker) const
  {
    // simultaneous walk of interval tree and implied binary tree of iv_ranges
#define SLOW_VERSIONs
#ifdef SLOW_VERSION
    for(size_t i = 0; i < iv_ranges.size(); i++)
      test_interval(iv_ranges.start(i), iv_ranges.end(i), marker);
#else
    if(root && (iv_ranges.size() > 0))
      root->test_sorted_intervals(iv_ranges, 0, iv_ranges.size(), marker);
#endif
  }

  template <typename IT, typename LT>
  template <typename IR>
  void IntervalTree<IT,LT>::test_sorted_intervals(const IR& iv_ranges, std::vector<bool>& labels_found) const
  {
    VectorMarker<IT,LT> marker(labels_found);
    test_sorted_intervals(iv_ranges, marker);
  }

  template <typename IT, typename LT>
  template <typename IR>
  void IntervalTree<IT,LT>::test_sorted_intervals(const IR& iv_ranges, std::set<LT>& labels_found) const
  {
    SetMarker<IT,LT> marker(labels_found);
    test_sorted_intervals(iv_ranges, marker);
  }


}; // namespace Realm
