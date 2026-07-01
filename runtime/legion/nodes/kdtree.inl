/* Copyright 2026 Stanford University, NVIDIA Corporation
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

// Included from kdtree.h - do not include this directly

// Useful for IDEs
#include "legion/nodes/kdtree.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // KD Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline KDNode<DIM, T>* KDTree::as_kdnode(void)
    //--------------------------------------------------------------------------
    {
      return legion_safe_cast<KDNode<DIM, T>*>(this);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, bool BY_RECTS>
    /*static*/ inline bool KDTree::compute_best_splitting_plane(
        const Rect<DIM, T>& bounds, const std::vector<Rect<DIM, T> >& rects,
        Rect<DIM, T>& best_left_bounds, Rect<DIM, T>& best_right_bounds,
        std::vector<Rect<DIM, T> >& best_left_set,
        std::vector<Rect<DIM, T> >& best_right_set)
    //--------------------------------------------------------------------------
    {
      // If we have sub-optimal bad sets we will track them here
      // so we can iterate through other dimensions to look for
      // better splitting planes
      int best_dim = -1;
      float best_cost = 2.f;  // worst possible cost
      for (int d = 0; d < DIM; d++)
      {
        // Try to compute a splitting plane for this dimension
        // Count how many rectangles start and end at each location
        std::map<std::pair<T, bool /*stop*/>, uint64_t> forward_lines;
        std::map<std::pair<T, bool /*start*/>, uint64_t> backward_lines;
        for (unsigned idx = 0; idx < rects.size(); idx++)
        {
          const Rect<DIM, T>& subset_bounds = rects[idx];
          // Start forward
          std::pair<T, bool> start_key(subset_bounds.lo[d], false);
          typename std::map<std::pair<T, bool>, uint64_t>::iterator finder =
              forward_lines.find(start_key);
          if (finder == forward_lines.end())
          {
            if (BY_RECTS)
              forward_lines[start_key] = 1;
            else
              forward_lines[start_key] = subset_bounds.volume();
          }
          else
          {
            if (BY_RECTS)
              finder->second++;
            else
              finder->second += subset_bounds.volume();
          }
          // Start backward
          start_key.second = true;
          finder = backward_lines.find(start_key);
          if (finder == backward_lines.end())
          {
            if (BY_RECTS)
              backward_lines[start_key] = 1;
            else
              backward_lines[start_key] = subset_bounds.volume();
          }
          else
          {
            if (BY_RECTS)
              finder->second++;
            else
              finder->second += subset_bounds.volume();
          }
          // Stop forward
          std::pair<T, uint64_t> stop_key(subset_bounds.hi[d], true);
          finder = forward_lines.find(stop_key);
          if (finder == forward_lines.end())
          {
            if (BY_RECTS)
              forward_lines[stop_key] = 1;
            else
              forward_lines[stop_key] = subset_bounds.volume();
          }
          else
          {
            if (BY_RECTS)
              finder->second += 1;
            else
              finder->second += subset_bounds.volume();
          }
          // Stop backward
          stop_key.second = false;
          finder = backward_lines.find(stop_key);
          if (finder == backward_lines.end())
          {
            if (BY_RECTS)
              backward_lines[stop_key] = 1;
            else
              backward_lines[stop_key] = subset_bounds.volume();
          }
          else
          {
            if (BY_RECTS)
              finder->second++;
            else
              finder->second += subset_bounds.volume();
          }
        }
        // Construct two lists by scanning from left-to-right and
        // from right-to-left of the number of rectangles that would
        // be inlcuded on the left or right side by each splitting plane
        std::map<T, uint64_t> lower_inclusive, upper_exclusive;
        uint64_t total = 0;
        for (typename std::map<std::pair<T, bool>, uint64_t>::const_iterator
                 it = forward_lines.begin();
             it != forward_lines.end(); it++)
        {
          // Increment first for starts for inclusivity
          if (!it->first.second)
            total += it->second;
          // Always record the count for all splits
          lower_inclusive[it->first.first] = total;
        }
        // If all the lines exist at the same value
        // then we'll never have a splitting plane
        if (lower_inclusive.size() == 1)
          continue;
        total = 0;
        for (typename std::map<
                 std::pair<T, bool>, uint64_t>::const_reverse_iterator it =
                 backward_lines.rbegin();
             it != backward_lines.rend(); it++)
        {
          // Always record the count for all splits
          upper_exclusive[it->first.first] = total;
          // Increment last for stops for exclusivity
          if (!it->first.second)
            total += it->second;
        }
        legion_assert(lower_inclusive.size() == upper_exclusive.size());
        // We want to take the mini-max of the two numbers in order
        // to try to balance the splitting plane across the two sets
        T split = 0;
        if (!BY_RECTS)
        {
          total = 0;
          for (unsigned idx = 0; idx < rects.size(); idx++)
            total += rects[idx].volume();
        }
        else
          total = rects.size();
        uint64_t split_max = total;
        for (typename std::map<T, uint64_t>::const_iterator it =
                 lower_inclusive.begin();
             it != lower_inclusive.end(); it++)
        {
          const uint64_t left = it->second;
          const uint64_t right = upper_exclusive[it->first];
          const uint64_t max = (left > right) ? left : right;
          if (max < split_max)
          {
            split_max = max;
            split = it->first;
          }
        }
        // Check for the case where we can't find a splitting plane
        if (split_max == total)
          continue;
        // Sort the subsets into left and right
        Rect<DIM, T> left_bounds(bounds);
        Rect<DIM, T> right_bounds(bounds);
        left_bounds.hi[d] = split;
        right_bounds.lo[d] = split + 1;
        std::vector<Rect<DIM, T> > left_set, right_set;
        for (typename std::vector<Rect<DIM, T> >::const_iterator it =
                 rects.begin();
             it != rects.end(); it++)
        {
          const Rect<DIM, T> left_rect = it->intersection(left_bounds);
          if (!left_rect.empty())
            left_set.emplace_back(left_rect);
          const Rect<DIM, T> right_rect = it->intersection(right_bounds);
          if (!right_rect.empty())
            right_set.emplace_back(right_rect);
        }
        legion_assert(left_set.size() < rects.size());
        legion_assert(right_set.size() < rects.size());
        // Compute the cost of this refinement
        // First get the percentage reductions of both sets
        float cost_left, cost_right;
        if (BY_RECTS)
        {
          cost_left = float(left_set.size()) / float(rects.size());
          cost_right = float(right_set.size()) / float(rects.size());
        }
        else
        {
          uint64_t volume = 0;
          for (typename std::vector<Rect<DIM, T> >::const_iterator it =
                   left_set.begin();
               it != left_set.end(); it++)
            volume += it->volume();
          cost_left = float(volume) / float(total);
          volume = 0;
          for (typename std::vector<Rect<DIM, T> >::const_iterator it =
                   right_set.begin();
               it != right_set.end(); it++)
            volume += it->volume();
          cost_right = float(volume) / float(total);
        }
        // We want to give better scores to sets that are closer together
        // so we'll include the absolute value of the difference in the
        // two costs as part of computing the average cost
        // If the savings are identical then this will be zero extra cost
        // Note this cost metric should always produce values between
        // 1.0 and 2.0, with 1.0 being a perfect 50% reduction on each side
        float cost_diff = (cost_left < cost_right) ? (cost_right - cost_left) :
                                                     (cost_left - cost_right);
        float total_cost = (cost_left + cost_right + cost_diff);
        legion_assert((1.f <= total_cost) && (total_cost <= 2.f));
        // Check to see if the cost is considered to be a "good" refinement
        // For now we'll say that this is a good cost if it is less than
        // or equal to 1.5, halfway between the range of costs from 1.0 to 2.0
        if ((total_cost <= 1.5f) && (total_cost < best_cost))
        {
          best_dim = d;
          best_cost = total_cost;
          best_left_set.swap(left_set);
          best_right_set.swap(right_set);
          best_left_bounds = left_bounds;
          best_right_bounds = right_bounds;
        }
      }
      return (best_dim >= 0);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, typename RT>
    KDNode<DIM, T, RT>::KDNode(
        const Rect<DIM, T>& b,
        std::vector<std::pair<Rect<DIM, T>, RT> >& subrects)
      : bounds(b), left(nullptr), right(nullptr)
    //--------------------------------------------------------------------------
    {
      // This is the base case
      if (subrects.size() <= LEGION_MAX_BVH_FANOUT)
      {
        rects.swap(subrects);
        return;
      }
      // If we have sub-optimal bad sets we will track them here
      // so we can iterate through other dimensions to look for
      // better splitting planes
      int best_dim = -1;
      float best_cost = 2.f;  // worst possible cost
      Rect<DIM, T> best_left_bounds, best_right_bounds;
      std::vector<std::pair<Rect<DIM, T>, RT> > best_left_set, best_right_set;
      for (int d = 0; d < DIM; d++)
      {
        // Try to compute a splitting plane for this dimension
        // Count how many rectangles start and end at each location
        std::map<std::pair<coord_t, bool /*stop*/>, unsigned> forward_lines;
        std::map<std::pair<coord_t, bool /*start*/>, unsigned> backward_lines;
        for (unsigned idx = 0; idx < subrects.size(); idx++)
        {
          const Rect<DIM, T>& subset_bounds = subrects[idx].first;
          // Start forward
          std::pair<coord_t, bool> start_key(subset_bounds.lo[d], false);
          std::map<std::pair<coord_t, bool>, unsigned>::iterator finder =
              forward_lines.find(start_key);
          if (finder == forward_lines.end())
            forward_lines[start_key] = 1;
          else
            finder->second++;
          // Start backward
          start_key.second = true;
          finder = backward_lines.find(start_key);
          if (finder == backward_lines.end())
            backward_lines[start_key] = 1;
          else
            finder->second++;
          // Stop forward
          std::pair<coord_t, bool> stop_key(subset_bounds.hi[d], true);
          finder = forward_lines.find(stop_key);
          if (finder == forward_lines.end())
            forward_lines[stop_key] = 1;
          else
            finder->second += 1;
          // Stop backward
          stop_key.second = false;
          finder = backward_lines.find(stop_key);
          if (finder == backward_lines.end())
            backward_lines[stop_key] = 1;
          else
            finder->second++;
        }
        // Construct two lists by scanning from left-to-right and
        // from right-to-left of the number of rectangles that would
        // be inlcuded on the left or right side by each splitting plane
        std::map<coord_t, unsigned> lower_inclusive, upper_exclusive;
        unsigned count = 0;
        for (typename std::map<
                 std::pair<coord_t, bool>, unsigned>::const_iterator it =
                 forward_lines.begin();
             it != forward_lines.end(); it++)
        {
          // Increment first for starts for inclusivity
          if (!it->first.second)
            count += it->second;
          // Always record the count for all splits
          lower_inclusive[it->first.first] = count;
        }
        // If all the lines exist at the same value
        // then we'll never have a splitting plane
        if (lower_inclusive.size() == 1)
          continue;
        count = 0;
        for (typename std::map<std::pair<coord_t, bool>, unsigned>::
                 const_reverse_iterator it = backward_lines.rbegin();
             it != backward_lines.rend(); it++)
        {
          // Always record the count for all splits
          upper_exclusive[it->first.first] = count;
          // Increment last for stops for exclusivity
          if (!it->first.second)
            count += it->second;
        }
        legion_assert(lower_inclusive.size() == upper_exclusive.size());
        // We want to take the mini-max of the two numbers in order
        // to try to balance the splitting plane across the two sets
        T split = 0;
        unsigned split_max = subrects.size();
        for (const std::pair<const coord_t, unsigned>& it : lower_inclusive)
        {
          const unsigned lower = it.second;
          const unsigned upper = upper_exclusive[it.first];
          const unsigned max = (lower > upper) ? lower : upper;
          if (max < split_max)
          {
            split_max = max;
            split = it.first;
          }
        }
        // Check for the case where we can't find a splitting plane
        if (split_max == subrects.size())
          continue;
        // Sort the subsets into left and right
        Rect<DIM, T> left_bounds(bounds);
        Rect<DIM, T> right_bounds(bounds);
        left_bounds.hi[d] = split;
        right_bounds.lo[d] = split + 1;
        std::vector<std::pair<Rect<DIM, T>, RT> > left_set, right_set;
        for (typename std::vector<std::pair<Rect<DIM, T>, RT> >::const_iterator
                 it = subrects.begin();
             it != subrects.end(); it++)
        {
          const Rect<DIM, T> left_rect = it->first.intersection(left_bounds);
          if (!left_rect.empty())
            left_set.emplace_back(std::make_pair(left_rect, it->second));
          const Rect<DIM, T> right_rect = it->first.intersection(right_bounds);
          if (!right_rect.empty())
            right_set.emplace_back(std::make_pair(right_rect, it->second));
        }
        legion_assert(left_set.size() < subrects.size());
        legion_assert(right_set.size() < subrects.size());
        // Compute the cost of this refinement
        // First get the percentage reductions of both sets
        float cost_left = float(left_set.size()) / float(subrects.size());
        float cost_right = float(right_set.size()) / float(subrects.size());
        // We want to give better scores to sets that are closer together
        // so we'll include the absolute value of the difference in the
        // two costs as part of computing the average cost
        // If the savings are identical then this will be zero extra cost
        // Note this cost metric should always produce values between
        // 1.0 and 2.0, with 1.0 being a perfect 50% reduction on each side
        float cost_diff = (cost_left < cost_right) ? (cost_right - cost_left) :
                                                     (cost_left - cost_right);
        float total_cost = (cost_left + cost_right + cost_diff);
        legion_assert((1.f <= total_cost) && (total_cost <= 2.f));
        // Check to see if the cost is considered to be a "good" refinement
        // For now we'll say that this is a good cost if it is less than
        // or equal to 1.5, halfway between the range of costs from 1.0 to 2.0
        if ((total_cost <= 1.5f) && (total_cost < best_cost))
        {
          best_dim = d;
          best_cost = total_cost;
          best_left_set.swap(left_set);
          best_right_set.swap(right_set);
          best_left_bounds = left_bounds;
          best_right_bounds = right_bounds;
        }
      }
      // See if we had at least one good refinement
      if (best_dim >= 0)
      {
        // Always clear the old-subrects before recursing to reduce memory usage
        {
          std::vector<std::pair<Rect<DIM, T>, RT> > empty;
          empty.swap(subrects);
        }
        left = new KDNode<DIM, T, RT>(best_left_bounds, best_left_set);
        right = new KDNode<DIM, T, RT>(best_right_bounds, best_right_set);
      }
      else
      {
        Warning warn;
        warn << "Failed to find a refinement for KD tree with " << DIM
             << " dimensions "
             << "and " << subrects.size()
             << " rectangles. Please report your application to the "
             << "Legion developers' mailing list.";
        warn.raise();
        // If we make it here then we couldn't find a splitting plane to refine
        // anymore so just record all the subrects as our rects
        rects.swap(subrects);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, typename RT>
    KDNode<DIM, T, RT>::~KDNode(void)
    //--------------------------------------------------------------------------
    {
      if (left != nullptr)
        delete left;
      if (right != nullptr)
        delete right;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, typename RT>
    void KDNode<DIM, T, RT>::find_interfering(
        const Rect<DIM, T>& test, std::set<RT>& interfering) const
    //--------------------------------------------------------------------------
    {
      if ((left != nullptr) && left->bounds.overlaps(test))
        left->find_interfering(test, interfering);
      if ((right != nullptr) && right->bounds.overlaps(test))
        right->find_interfering(test, interfering);
      for (typename std::vector<std::pair<Rect<DIM, T>, RT> >::const_iterator
               it = rects.begin();
           it != rects.end(); it++)
        if (it->first.overlaps(test))
          interfering.insert(it->second);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, typename RT>
    void KDNode<DIM, T, RT>::record_inorder_traversal(
        std::vector<RT>& out) const
    //--------------------------------------------------------------------------
    {
      if (left != nullptr)
        left->record_inorder_traversal(out);
      for (typename std::vector<std::pair<Rect<DIM, T>, RT> >::const_iterator
               it = rects.begin();
           it != rects.end(); it++)
        out.emplace_back(it->second);
      if (right != nullptr)
        right->record_inorder_traversal(out);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, typename RT>
    RT KDNode<DIM, T, RT>::find(const Point<DIM, T>& point) const
    //--------------------------------------------------------------------------
    {
      for (typename std::vector<std::pair<Rect<DIM, T>, RT> >::const_iterator
               it = rects.begin();
           it != rects.end(); it++)
        if (it->first.contains(point))
          return it->second;
      if ((left != nullptr) && left->bounds.contains(point))
        return left->find(point);
      if ((right != nullptr) && right->bounds.contains(point))
        return right->find(point);
      // Should always find it currently
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    KDNode<DIM, T, void>::KDNode(
        const Rect<DIM, T>& b, std::vector<Rect<DIM, T> >& subrects)
      : bounds(b), left(nullptr), right(nullptr)
    //--------------------------------------------------------------------------
    {
      // This is the base case
      if (subrects.size() <= LEGION_MAX_BVH_FANOUT)
      {
        rects.swap(subrects);
        return;
      }
      Rect<DIM, T> best_left_bounds, best_right_bounds;
      std::vector<Rect<DIM, T> > best_left_set, best_right_set;
      bool success = compute_best_splitting_plane<DIM, T>(
          bounds, subrects, best_left_bounds, best_right_bounds, best_left_set,
          best_right_set);
      // See if we had at least one good refinement
      if (success)
      {
        // Always clear the old-subrects before recursing to reduce memory usage
        {
          std::vector<Rect<DIM, T> > empty;
          empty.swap(subrects);
        }
        left = new KDNode<DIM, T, void>(best_left_bounds, best_left_set);
        right = new KDNode<DIM, T, void>(best_right_bounds, best_right_set);
      }
      else
      {
        Warning warn;
        warn << "Failed to find a refinement for KD tree with " << DIM
             << " dimensions "
             << "and " << subrects.size()
             << " rectangles. Please report your application to the "
             << "Legion developers' mailing list.";
        warn.raise();
        // If we make it here then we couldn't find a splitting plane to refine
        // anymore so just record all the subrects as our rects
        rects.swap(subrects);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    KDNode<DIM, T, void>::~KDNode(void)
    //--------------------------------------------------------------------------
    {
      if (left != nullptr)
        delete left;
      if (right != nullptr)
        delete right;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t KDNode<DIM, T, void>::count_rectangles(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = rects.size();
      if (left != nullptr)
        result += left->count_rectangles();
      if (right != nullptr)
        result += right->count_rectangles();
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t KDNode<DIM, T, void>::count_intersecting_points(
        const Rect<DIM, T>& rect) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      for (typename std::vector<Rect<DIM, T> >::const_iterator it =
               rects.begin();
           it != rects.end(); it++)
      {
        const Rect<DIM, T> overlap = it->intersection(rect);
        result += overlap.volume();
      }
      if (left != nullptr)
      {
        Rect<DIM, T> left_overlap = rect.intersection(left->bounds);
        if (!left_overlap.empty())
          result += left->count_intersecting_points(left_overlap);
      }
      if (right != nullptr)
      {
        Rect<DIM, T> right_overlap = rect.intersection(right->bounds);
        if (!right_overlap.empty())
          result += right->count_intersecting_points(right_overlap);
      }
      return result;
    }

#ifdef DEFINE_NT_TEMPLATES
    /////////////////////////////////////////////////////////////
    // Equivalence Set KD Tree
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline EqKDTreeT<DIM, T>* EqKDTree::as_eq_kd_tree(void)
    //--------------------------------------------------------------------------
    {
      return legion_safe_cast<EqKDTreeT<DIM, T>*>(this);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDTreeT<DIM, T>::EqKDTreeT(const Rect<DIM, T>& rect) : bounds(rect)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDTreeT<DIM, T>::compute_shard_equivalence_sets(
        const Domain& domain, const FieldMask& mask,
        const std::vector<EqSetTracker*>& trackers,
        const std::vector<AddressSpaceID>& tracker_spaces,
        std::vector<unsigned>& new_tracker_references,
        op::FieldMaskMap<EquivalenceSet>& eq_sets,
        std::vector<RtEvent>& pending_sets,
        op::FieldMaskMap<EqKDTree>& subscriptions,
        op::FieldMaskMap<EqKDTree>& to_create,
        op::map<EqKDTree*, Domain>& creation_rects,
        op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      const Rect<DIM, T> rect = domain;
      op::map<ShardID, op::map<Domain, FieldMask> > remote_shard_rects;
      compute_equivalence_sets(
          rect, mask, trackers, tracker_spaces, new_tracker_references, eq_sets,
          pending_sets, subscriptions, to_create, creation_rects, creation_srcs,
          remote_shard_rects, local_shard);
      // Should not have any of these at this point
      legion_assert(remote_shard_rects.empty());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDTreeT<DIM, T>::record_shard_output_equivalence_set(
        EquivalenceSet* set, const Domain& domain, const FieldMask& mask,
        EqSetTracker* tracker, AddressSpaceID tracker_space,
        local::FieldMaskMap<EqKDTree>& new_subscriptions, ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      const Rect<DIM, T> rect = domain;
      op::map<ShardID, op::map<Domain, FieldMask> > remote_shard_rects;
      unsigned references = record_output_equivalence_set(
          set, rect, mask, tracker, tracker_space, new_subscriptions,
          remote_shard_rects, local_shard);
      // Should not have any of these at this point
      legion_assert(remote_shard_rects.empty());
      return references;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDTreeT<DIM, T>::invalidate_shard_tree(
        const Domain& domain, const FieldMask& mask,
        std::vector<RtEvent>& invalidated)
    //--------------------------------------------------------------------------
    {
      const Rect<DIM, T> rect = domain;
      invalidate_tree(rect, mask, invalidated, true /*move to previous*/);
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set KD Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDNode<DIM, T>::EqKDNode(const Rect<DIM, T>& rect)
      : EqKDTreeT<DIM, T>(rect), lefts(nullptr), rights(nullptr),
        current_sets(nullptr), previous_sets(nullptr),
        current_set_preconditions(nullptr), pending_set_creations(nullptr),
        pending_postconditions(nullptr), subscriptions(nullptr),
        child_previous_below(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDNode<DIM, T>::~EqKDNode(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(subscriptions == nullptr);
      legion_assert(pending_set_creations == nullptr);
      legion_assert(pending_postconditions == nullptr);
      if (lefts != nullptr)
      {
        for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
                 lefts->begin();
             it != lefts->end(); it++)
          if (it->first->remove_reference())
            delete it->first;
        delete lefts;
      }
      if (rights != nullptr)
      {
        for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
                 rights->begin();
             it != rights->end(); it++)
          if (it->first->remove_reference())
            delete it->first;
        delete rights;
      }
      if (current_sets != nullptr)
      {
        for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
                 current_sets->begin();
             it != current_sets->end(); it++)
          if (it->first->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
            delete it->first;
        delete current_sets;
      }
      if (previous_sets != nullptr)
      {
        for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
                 previous_sets->begin();
             it != previous_sets->end(); it++)
          if (it->first->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
            delete it->first;
        delete previous_sets;
      }
      if (current_set_preconditions != nullptr)
        delete current_set_preconditions;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::initialize_set(
        EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
        ShardID local_shard, bool current)
    //--------------------------------------------------------------------------
    {
      local::FieldMaskMap<EqKDNode<DIM, T> > to_traverse;
      {
        FieldMask remaining, unrefined = mask;
        AutoLock n_lock(node_lock);
        if (lefts != nullptr)
          unrefined -= lefts->get_valid_mask();
        if (!!unrefined)
        {
          if (rect == this->bounds)
          {
            if (current)
            {
              if (current_sets == nullptr)
                current_sets = new lng::FieldMaskMap<EquivalenceSet>();
              if (current_sets->insert(set, unrefined))
                set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
            }
            else
            {
              if (previous_sets == nullptr)
                previous_sets = new lng::FieldMaskMap<EquivalenceSet>();
              if (previous_sets->insert(set, unrefined))
                set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
            }
            remaining = mask - unrefined;
            if (!remaining)
              return;
          }
          else
          {
            // Refine for all the fields which aren't refined yet
            refine_node(rect, unrefined);
            if (!current)
              all_previous_below |= unrefined;
            remaining = mask;
          }
        }
        else
          remaining = mask;
        // If we get here, we're traversing refinements
        if (current && !!all_previous_below)
          all_previous_below -= remaining;
        legion_assert(!!remaining);
        legion_assert((lefts != nullptr) && (rights != nullptr));
        for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
                 lefts->begin();
             it != lefts->end(); it++)
        {
          const FieldMask overlap = remaining & it->second;
          if (!overlap)
            continue;
          Rect<DIM, T> intersection = rect.intersection(it->first->bounds);
          if (intersection.empty())
            continue;
          to_traverse.insert(it->first, overlap);
          if (intersection == rect)
          {
            remaining -= overlap;
            if (!remaining)
              break;
          }
        }
        if (!!remaining)
        {
          for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator
                   it = rights->begin();
               it != rights->end(); it++)
          {
            const FieldMask overlap = remaining & it->second;
            if (!overlap)
              continue;
            legion_assert(rect.overlaps(it->first->bounds));
            to_traverse.insert(it->first, overlap);
            remaining -= overlap;
            if (!remaining)
              break;
          }
        }
      }
      for (typename local::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
               to_traverse.begin();
           it != to_traverse.end(); it++)
      {
        const Rect<DIM, T> overlap = rect.intersection(it->first->bounds);
        legion_assert(!overlap.empty());
        it->first->initialize_set(
            set, overlap, it->second, local_shard, current);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::compute_equivalence_sets(
        const Rect<DIM, T>& rect, const FieldMask& mask,
        const std::vector<EqSetTracker*>& trackers,
        const std::vector<AddressSpaceID>& tracker_spaces,
        std::vector<unsigned>& new_tracker_references,
        op::FieldMaskMap<EquivalenceSet>& eq_sets,
        std::vector<RtEvent>& pending_sets,
        op::FieldMaskMap<EqKDTree>& new_subscriptions,
        op::FieldMaskMap<EqKDTree>& to_create,
        op::map<EqKDTree*, Domain>& creation_rects,
        op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      legion_assert(this->bounds.contains(rect));
      local::FieldMaskMap<EqKDNode<DIM, T> > to_traverse, to_get_previous,
          to_invalidate_previous;
      {
        FieldMask remaining = mask;
        AutoLock n_lock(node_lock);
        // First check to see if we have any current equivalence sets
        // here which means we can just record them
        if ((current_sets != nullptr) &&
            !(remaining * current_sets->get_valid_mask()))
        {
          FieldMask check_preconditions;
          for (lng::FieldMaskMap<EquivalenceSet>::const_iterator cit =
                   current_sets->begin();
               cit != current_sets->end(); cit++)
          {
            const FieldMask overlap = remaining & cit->second;
            if (!overlap)
              continue;
            eq_sets.insert(cit->first, overlap);
            remaining -= overlap;
            new_subscriptions.insert(this, overlap);
            for (unsigned idx = 0; idx < trackers.size(); idx++)
              new_tracker_references[idx] += record_subscription(
                  trackers[idx], tracker_spaces[idx], overlap);
            if (current_set_preconditions != nullptr)
              check_preconditions |= overlap;
            if (!remaining)
              break;
          }
          if (!!check_preconditions)
          {
            // Check to see if there are any pending set creation events
            // still valid for this event, if so we still need to
            // record them to make sure we don't try to use this set
            // until it is actually ready
            for (shrt::map<RtEvent, FieldMask>::iterator it =
                     current_set_preconditions->begin();
                 it != current_set_preconditions->end();
                 /*nothing*/)
            {
              if (!it->first.has_triggered())
              {
                if (!(check_preconditions * it->second))
                {
                  pending_sets.emplace_back(it->first);
                  check_preconditions -= it->second;
                  if (!check_preconditions)
                    break;
                }
                it++;
              }
              else
              {
                // Perform the previous invalidations now that the
                // event has triggered
                invalidate_previous_sets(it->second, to_invalidate_previous);
                shrt::map<RtEvent, FieldMask>::iterator to_delete = it++;
                current_set_preconditions->erase(to_delete);
              }
            }
            if (current_set_preconditions->empty())
            {
              delete current_set_preconditions;
              current_set_preconditions = nullptr;
            }
          }
        }
        if (!!remaining)
        {
          // if we still have remaining fields, check for any pending
          // sets that might be in the process of being made
          if (pending_set_creations != nullptr)
          {
            for (shrt::map<RtUserEvent, FieldMask>::const_iterator it =
                     pending_set_creations->begin();
                 it != pending_set_creations->end(); it++)
            {
              const FieldMask overlap = remaining & it->second;
              if (!overlap)
                continue;
              pending_sets.emplace_back(it->first);
              remaining -= overlap;
              new_subscriptions.insert(this, overlap);
              for (unsigned idx = 0; idx < trackers.size(); idx++)
                new_tracker_references[idx] += record_subscription(
                    trackers[idx], tracker_spaces[idx], overlap);
              if (!remaining)
                break;
            }
          }
          if (!!remaining)
          {
            // Next check to see if we have to traverse below any nodes
            // below because they have been refined. If they're all previous
            // below and we're trying to make equivalence sets here then we
            // can skip traversing below since we'll be able to coarsen
            FieldMask to_coarsen;
            if (!!all_previous_below && (rect == this->bounds))
            {
              to_coarsen = remaining & all_previous_below;
              remaining -= to_coarsen;
            }
            if ((lefts != nullptr) && !(remaining * lefts->get_valid_mask()))
            {
              FieldMask right_mask;
              for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator
                       it = lefts->begin();
                   it != lefts->end(); it++)
              {
                const FieldMask overlap = it->second & remaining;
                if (!overlap)
                  continue;
                if (it->first->bounds.overlaps(rect))
                {
                  to_traverse.insert(it->first, overlap);
                  if (!it->first->bounds.contains(rect))
                    right_mask |= overlap;
                }
                else
                  right_mask |= overlap;
                remaining -= overlap;
                if (!remaining)
                  continue;
              }
              if (!!right_mask)
              {
                for (typename lng::FieldMaskMap<
                         EqKDNode<DIM, T> >::const_iterator it =
                         rights->begin();
                     it != rights->end(); it++)
                {
                  const FieldMask overlap = it->second & right_mask;
                  if (!overlap)
                    continue;
                  if (it->first->bounds.overlaps(rect))
                    to_traverse.insert(it->first, overlap);
                  right_mask -= overlap;
                  if (!right_mask)
                    break;
                }
              }
            }
            // Re-introduce the fields we want to try to refine here
            if (!!to_coarsen)
              remaining |= to_coarsen;
            if (!!remaining)
            {
              // if we still have remaining fields, then we're going to
              // be the ones to make the equivalence set for these fields
              // check to see if the rect is the same as our bounds or
              // whether it is smaller
              if (rect == this->bounds)
              {
                // easy case, we can just record that we're making a
                // new equivalence set at this node
                if (pending_set_creations == nullptr)
                  pending_set_creations =
                      new shrt::map<RtUserEvent, FieldMask>();
                const RtUserEvent ready = Runtime::create_rt_user_event();
                pending_set_creations->insert(std::make_pair(ready, remaining));
                // Record the subscription now so we know whether to
                // add a reference to the tracker or not
                for (unsigned idx = 0; idx < trackers.size(); idx++)
                  new_tracker_references[idx] += record_subscription(
                      trackers[idx], tracker_spaces[idx], remaining);
                to_create.insert(this, remaining);
                creation_rects[this] = Domain(rect);
                // Find any creation sources
                if ((previous_sets != nullptr) &&
                    !(remaining * previous_sets->get_valid_mask()))
                {
                  for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
                           previous_sets->begin();
                       it != previous_sets->end(); it++)
                  {
                    const FieldMask overlap = it->second & remaining;
                    if (!overlap)
                      continue;
                    creation_srcs[it->first][this->bounds] |= overlap;
                    remaining -= overlap;
                    if (!remaining)
                      break;
                  }
                }
                // Check for any previous sets below us to get as well
                if (!!remaining && !!all_previous_below)
                {
                  FieldMask all_prev_below = all_previous_below & remaining;
                  if (!!all_prev_below)
                    // These fields will no longer be all_prevous_below now
                    find_to_get_previous(all_prev_below, to_get_previous);
                }
              }
              else
              {
                refine_node(rect, remaining);
                legion_assert((lefts != nullptr) && (rights != nullptr));
                for (typename lng::FieldMaskMap<
                         EqKDNode<DIM, T> >::const_iterator it = lefts->begin();
                     it != lefts->end(); it++)
                {
                  const FieldMask overlap = remaining & it->second;
                  if (!overlap)
                    continue;
                  const Rect<DIM, T> intersection =
                      rect.intersection(it->first->bounds);
                  if (intersection.empty())
                    continue;
                  to_traverse.insert(it->first, overlap);
                  if (intersection == rect)
                  {
                    remaining -= overlap;
                    if (!remaining)
                      break;
                  }
                }
                if (!!remaining)
                {
                  for (typename lng::FieldMaskMap<
                           EqKDNode<DIM, T> >::const_iterator it =
                           rights->begin();
                       it != rights->end(); it++)
                  {
                    const FieldMask overlap = remaining & it->second;
                    if (!overlap)
                      continue;
                    legion_assert(rect.overlaps(it->first->bounds));
                    to_traverse.insert(it->first, overlap);
                    remaining -= overlap;
                    if (!remaining)
                      break;
                  }
                }
              }
            }
          }
        }
        // If we're traversing for any fields then remove them from the set
        // of all previous below since we know what we'll no longer have
        // all previous below at this point
        if (!to_traverse.empty())
        {
          if (!!all_previous_below)
            all_previous_below -= to_traverse.get_valid_mask();
          // Also filter the individual child previous below
          if ((child_previous_below != nullptr) &&
              !(to_traverse.get_valid_mask() *
                child_previous_below->get_valid_mask()))
          {
            for (typename local::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator
                     it = to_traverse.begin();
                 it != to_traverse.end(); it++)
            {
              typename lng::FieldMaskMap<EqKDNode<DIM, T> >::iterator finder =
                  child_previous_below->find(it->first);
              if (finder == child_previous_below->end())
                continue;
              finder.filter(it->second);
              if (!finder->second)
                child_previous_below->erase(finder);
            }
            if (child_previous_below->empty())
            {
              delete child_previous_below;
              child_previous_below = nullptr;
            }
            else
              child_previous_below->tighten_valid_mask();
          }
        }
      }
      legion_assert(
          to_traverse.get_valid_mask() * to_get_previous.get_valid_mask());
      for (typename local::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
               to_traverse.begin();
           it != to_traverse.end(); it++)
      {
        const Rect<DIM, T> overlap = rect.intersection(it->first->bounds);
        legion_assert(!overlap.empty());
        it->first->compute_equivalence_sets(
            overlap, it->second, trackers, tracker_spaces,
            new_tracker_references, eq_sets, pending_sets, new_subscriptions,
            to_create, creation_rects, creation_srcs, remote_shard_rects,
            local_shard);
      }
      for (typename local::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
               to_get_previous.begin();
           it != to_get_previous.end(); it++)
        it->first->find_all_previous_sets(it->second, creation_srcs);
      for (typename local::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
               to_invalidate_previous.begin();
           it != to_invalidate_previous.end(); it++)
      {
        it->first->invalidate_all_previous_sets(it->second);
        if (it->first->remove_reference())
          delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::find_all_previous_sets(
        FieldMask mask,
        op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs)
    //--------------------------------------------------------------------------
    {
      local::FieldMaskMap<EqKDNode<DIM, T> > to_get_previous;
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        legion_assert(
            (current_sets == nullptr) ||
            (current_sets->get_valid_mask() * mask));
        if (previous_sets != nullptr)
        {
          for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
                   previous_sets->begin();
               it != previous_sets->end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            creation_srcs[it->first][this->bounds] |= overlap;
            mask -= overlap;
            if (!mask)
              return;
          }
        }
        legion_assert(!(mask - all_previous_below));
        find_to_get_previous(mask, to_get_previous);
      }
      for (typename local::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
               to_get_previous.begin();
           it != to_get_previous.end(); it++)
        it->first->find_all_previous_sets(it->second, creation_srcs);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::find_to_get_previous(
        FieldMask& all_prev_below,
        local::FieldMaskMap<EqKDNode<DIM, T> >& to_get_previous) const
    //--------------------------------------------------------------------------
    {
      legion_assert((lefts != nullptr) && (rights != nullptr));
      // We're going to pull these out of the lefts and rights
      // since we're just going to use them to get the sets
      for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
               lefts->begin();
           it != lefts->end(); it++)
      {
        const FieldMask overlap = all_prev_below & it->second;
        if (!overlap)
          continue;
        to_get_previous.insert(it->first, overlap);
      }
      for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
               rights->begin();
           it != rights->end(); it++)
      {
        const FieldMask overlap = all_prev_below & it->second;
        if (!overlap)
          continue;
        to_get_previous.insert(it->first, overlap);
        all_prev_below -= overlap;
        if (!all_prev_below)
          break;
      }
      legion_assert(!all_prev_below);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::refine_node(
        const Rect<DIM, T>& rect, const FieldMask& mask, bool refine_current)
    //--------------------------------------------------------------------------
    {
      // We shouldn't have any existing refinements for these fields
      legion_assert((lefts == nullptr) || (mask * lefts->get_valid_mask()));
      legion_assert((rights == nullptr) || (mask * rights->get_valid_mask()));
      legion_assert(mask * all_previous_below);
      legion_assert(
          (child_previous_below == nullptr) ||
          (mask * child_previous_below->get_valid_mask()));
      // We shouldn't have any current sets either for these fields
      legion_assert(
          (current_sets == nullptr) ||
          (mask * current_sets->get_valid_mask()) || refine_current);
#ifdef LEGION_DEBUG
      if (pending_set_creations != nullptr)
      {
        // Invalidations should never be racing with pending set
        // creations, this should be guaranteed by the logical dependence
        // analysis which ensures refinements are serialized with respect
        // to all other operations
        for (shrt::map<RtUserEvent, FieldMask>::const_iterator it =
                 pending_set_creations->begin();
             it != pending_set_creations->end(); it++)
          legion_assert(mask * it->second);
      }
      if (subscriptions != nullptr)
      {
        // We should never be refining something which has subscribers
        for (lng::map<AddressSpaceID, lng::FieldMaskMap<EqSetTracker> >::
                 const_iterator it = subscriptions->begin();
             it != subscriptions->end(); it++)
          legion_assert(mask * it->second.get_valid_mask());
      }
#endif
      // Need to create a new refinement for these fields
      // to match the rectangle being requested
      // First check to see if we can find a dimension where
      // the half-way cutting plane puts the rectangle on
      // one side or the other, if we can find such a
      // dimension then we'll just split on the biggest one
      // and continue on our way. If we can't find such a
      // dimension we'll split on the dimension that where the
      // edge of the rectangle falls closest to the halfway point
      T split = 0;
      int dim = -1;
      T largest = 0;
      for (int d = 0; d < DIM; d++)
      {
        if (this->bounds.lo[d] == this->bounds.hi[d])
          continue;
        T diff = this->bounds.hi[d] - this->bounds.lo[d];
        T mid = this->bounds.lo[d] + (diff / 2);
        if ((rect.hi[d] <= mid) || (mid < rect.lo[d]))
        {
          if ((dim < 0) || (largest < diff))
          {
            dim = d;
            split = mid;
            largest = diff;
          }
        }
      }
      if (dim < 0)
      {
        // We couldn't find a nice splitting dimension, so
        // we're now going to find the one with an edge on
        // the rectangle that is closest to the middle splitting
        // point. We're guaranteed that such a split must exist
        // because the rect and the bounds are not equal
        T distance = 0;
        for (int d = 0; d < DIM; d++)
        {
          if (this->bounds.lo[d] == this->bounds.hi[d])
            continue;
          T diff = this->bounds.hi[d] - this->bounds.lo[d];
          T mid = this->bounds.lo[d] + (diff / 2);
          if (this->bounds.lo[d] < rect.lo[d])
          {
            T dist = ((rect.lo[d] - 1) <= mid) ? mid - (rect.lo[d] - 1) :
                                                 (rect.lo[d] - 1) - mid;
            if ((dim < 0) || (dist < distance))
            {
              dim = d;
              split = rect.lo[d] - 1;
              distance = dist;
            }
          }
          if (rect.hi[d] < this->bounds.hi[d])
          {
            T dist = (rect.hi[d] <= mid) ? mid - rect.hi[d] : rect.hi[d] - mid;
            if ((dim < 0) || (dist < distance))
            {
              dim = d;
              split = rect.hi[d];
              distance = dist;
            }
          }
        }
        legion_assert(dim >= 0);
      }
      Rect<DIM, T> left_bounds = this->bounds;
      Rect<DIM, T> right_bounds = this->bounds;
      left_bounds.hi[dim] = split;
      right_bounds.lo[dim] = split + 1;
      // See if we can reuse any existing subnodes or whether we
      // need to make new ones
      if (lefts != nullptr)
      {
        EqKDNode<DIM, T>* prior_left = nullptr;
        for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::iterator it =
                 lefts->begin();
             it != lefts->end(); it++)
        {
          if (it->first->bounds != left_bounds)
            continue;
          prior_left = it->first;
          it.merge(mask);
          break;
        }
        if (prior_left != nullptr)
        {
          EqKDNode<DIM, T>* prior_right = nullptr;
          for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::iterator it =
                   rights->begin();
               it != rights->end(); it++)
          {
            if (it->first->bounds != right_bounds)
              continue;
            prior_right = it->first;
            it.merge(mask);
            break;
          }
          legion_assert(prior_right != nullptr);
          legion_assert(
              left_bounds.contains(rect) || right_bounds.contains(rect));
          if (previous_sets != nullptr)
          {
            if (!refine_current)
              all_previous_below |= mask & previous_sets->get_valid_mask();
            clone_sets(prior_left, prior_right, mask, previous_sets, false);
          }
          if (refine_current)
            clone_sets(prior_left, prior_right, mask, current_sets, true);
          return;
        }
      }
      // If we still have remaining fields, then we need to
      // make new left and right nodes
      EqKDNode<DIM, T>* new_left = new EqKDNode<DIM, T>(left_bounds);
      EqKDNode<DIM, T>* new_right = new EqKDNode<DIM, T>(right_bounds);
      if (lefts == nullptr)
        lefts = new lng::FieldMaskMap<EqKDNode<DIM, T> >();
      if (lefts->insert(new_left, mask))
        new_left->add_reference();
      if (rights == nullptr)
        rights = new lng::FieldMaskMap<EqKDNode<DIM, T> >();
      if (rights->insert(new_right, mask))
        new_right->add_reference();
      if (previous_sets != nullptr)
      {
        if (!refine_current)
          all_previous_below |= mask & previous_sets->get_valid_mask();
        clone_sets(new_left, new_right, mask, previous_sets, false /*current*/);
      }
      if (refine_current)
        clone_sets(new_left, new_right, mask, current_sets, true /*current*/);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::record_equivalence_set(
        EquivalenceSet* set, const FieldMask& mask, RtEvent ready,
        const CollectiveMapping& creator_spaces,
        const std::vector<EqSetTracker*>& creators)
    //--------------------------------------------------------------------------
    {
      legion_assert(!creators.empty());
      legion_assert(creator_spaces.size() == creators.size());
      local::FieldMaskMap<EqKDNode<DIM, T> > to_invalidate_previous;
      {
        AutoLock n_lock(node_lock);
        if (current_sets == nullptr)
          current_sets = new lng::FieldMaskMap<EquivalenceSet>();
        legion_assert(mask * current_sets->get_valid_mask());
        if (current_sets->insert(set, mask))
          set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
        // Send notifications to all the subscriptions that are waiting for
        // the set to be sent to them
        if (subscriptions != nullptr)
        {
          for (lng::map<AddressSpaceID, lng::FieldMaskMap<EqSetTracker> >::
                   const_iterator sit = subscriptions->begin();
               sit != subscriptions->end(); sit++)
          {
            if (sit->second.get_valid_mask() * mask)
              continue;
            // See if there is a creator to ignore on this space
            EqSetTracker* creator = nullptr;
            if (creator_spaces.contains(sit->first))
            {
              const unsigned index = creator_spaces.find_index(sit->first);
              creator = creators[index];
            }
            if (sit->first != runtime->address_space)
            {
              local::FieldMaskMap<EqSetTracker> to_notify;
              for (lng::FieldMaskMap<EqSetTracker>::const_iterator it =
                       sit->second.begin();
                   it != sit->second.end(); it++)
              {
                // Skip the creator tracker since it made it
                if (it->first == creator)
                  continue;
                const FieldMask overlap = mask & it->second;
                if (!overlap)
                  continue;
                to_notify.insert(it->first, overlap);
              }
              if (!to_notify.empty())
              {
                // Create an event for when this is triggered
                const RtUserEvent recorded = Runtime::create_rt_user_event();
                ComputeEquivalenceSetsPending rez;
                {
                  RezCheck z(rez);
                  rez.serialize(set->did);
                  rez.serialize<size_t>(to_notify.size());
                  for (local::FieldMaskMap<EqSetTracker>::const_iterator it =
                           to_notify.begin();
                       it != to_notify.end(); it++)
                  {
                    rez.serialize(it->first);
                    rez.serialize(it->second);
                  }
                  rez.serialize(recorded);
                }
                rez.dispatch(sit->first);
                // Save this event as a postcondition for any pending creations
                legion_assert(pending_set_creations != nullptr);
                for (shrt::map<RtUserEvent, FieldMask>::const_iterator it =
                         pending_set_creations->begin();
                     it != pending_set_creations->end(); it++)
                {
                  if (it->second * to_notify.get_valid_mask())
                    continue;
                  if (pending_postconditions == nullptr)
                    pending_postconditions =
                        new std::map<RtUserEvent, std::vector<RtEvent> >();
                  (*pending_postconditions)[it->first].emplace_back(recorded);
                }
              }
            }
            else
            {
              // Local case so we can notify these directly
              for (lng::FieldMaskMap<EqSetTracker>::const_iterator it =
                       sit->second.begin();
                   it != sit->second.end(); it++)
              {
                // Skip the creator tracker since it made it
                if (it->first == creator)
                  continue;
                const FieldMask overlap = mask & it->second;
                if (!overlap)
                  continue;
                it->first->record_pending_equivalence_set(set, overlap);
              }
            }
          }
        }
        legion_assert(pending_set_creations != nullptr);
        // Filter out any pending set creation events
        for (shrt::map<RtUserEvent, FieldMask>::iterator it =
                 pending_set_creations->begin();
             it != pending_set_creations->end();
             /*nothing*/)
        {
          it->second -= mask;
          if (!it->second)
          {
            // Removed all the fields so now we can trigger the event
            // See if it has any postconditions
            if (pending_postconditions != nullptr)
            {
              std::map<RtUserEvent, std::vector<RtEvent> >::iterator finder =
                  pending_postconditions->find(it->first);
              if (finder != pending_postconditions->end())
              {
                if (ready.exists())
                  finder->second.emplace_back(ready);
                Runtime::trigger_event(
                    it->first, Runtime::merge_events(finder->second));
                pending_postconditions->erase(finder);
                if (pending_postconditions->empty())
                {
                  delete pending_postconditions;
                  pending_postconditions = nullptr;
                }
              }
              else
                Runtime::trigger_event(it->first, ready);
            }
            else
              Runtime::trigger_event(it->first, ready);
            shrt::map<RtUserEvent, FieldMask>::iterator to_delete = it++;
            pending_set_creations->erase(to_delete);
          }
          else
          {
            if (ready.exists() && !ready.has_triggered())
            {
              if (pending_postconditions == nullptr)
                pending_postconditions =
                    new std::map<RtUserEvent, std::vector<RtEvent> >();
              (*pending_postconditions)[it->first].emplace_back(ready);
            }
            it++;
          }
        }
        if (pending_set_creations->empty())
        {
          delete pending_set_creations;
          pending_set_creations = nullptr;
        }
        // If the ready event hasn't triggered when need to keep around
        // this event so no one tries to use the new equivalence set or
        // invalidate any of the previous sets it depends on until it
        // is actually ready to be used and all the clones are done
        if (ready.exists() && !ready.has_triggered())
        {
          if (current_set_preconditions == nullptr)
            current_set_preconditions = new shrt::map<RtEvent, FieldMask>();
          current_set_preconditions->insert(std::make_pair(ready, mask));
        }
        else  // we can invalidate the previous sets now
          // we can only do this if the ready event has triggered which
          // indicates that all the clone operations from the previous
          // sets are done and it's safe to remove the references
          invalidate_previous_sets(mask, to_invalidate_previous);
      }
      for (typename local::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
               to_invalidate_previous.begin();
           it != to_invalidate_previous.end(); it++)
      {
        it->first->invalidate_all_previous_sets(it->second);
        if (it->first->remove_reference())
          delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDNode<DIM, T>::record_output_equivalence_set(
        EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
        EqSetTracker* tracker, AddressSpaceID tracker_space,
        local::FieldMaskMap<EqKDTree>& subscriptions,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      unsigned new_subs = 0;
      local::FieldMaskMap<EqKDNode<DIM, T> > to_traverse;
      {
        FieldMask submask = mask;
        AutoLock n_lock(node_lock);
        if (rect == this->bounds)
        {
          FieldMask local_fields = mask;
          if (lefts != nullptr)
            local_fields -= lefts->get_valid_mask();
          if (!!local_fields)
          {
            // Record the set and subscriptions here
            legion_assert(
                (current_sets == nullptr) ||
                (local_fields * current_sets->get_valid_mask()));
            legion_assert(
                (previous_sets == nullptr) ||
                (local_fields * previous_sets->get_valid_mask()));
            if (current_sets == nullptr)
              current_sets = new lng::FieldMaskMap<EquivalenceSet>();
            if (current_sets->insert(set, local_fields))
              set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
            subscriptions.insert(this, local_fields);
            new_subs +=
                record_subscription(tracker, tracker_space, local_fields);
            submask -= local_fields;
          }
        }
        else
        {
          FieldMask to_refine = mask;
          if (lefts != nullptr)
            to_refine -= lefts->get_valid_mask();
          if (!!to_refine)
            refine_node(rect, to_refine);
        }
        // Check to see if there are any already refined nodes to traverse
        if (!!submask)
        {
          legion_assert((lefts != nullptr) && (rights != nullptr));
          for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator
                   it = lefts->begin();
               it != lefts->end(); it++)
          {
            const FieldMask overlap = submask & it->second;
            if (!overlap)
              continue;
            if (!it->first->bounds.overlaps(rect))
              continue;
            to_traverse.insert(it->first, overlap);
            if (it->first->bounds.contains(rect))
            {
              submask -= overlap;
              if (!submask)
                break;
            }
          }
          if (!!submask)
          {
            for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator
                     it = rights->begin();
                 it != rights->end(); it++)
            {
              const FieldMask overlap = submask & it->second;
              if (!overlap)
                continue;
              legion_assert(it->first->bounds.overlaps(rect));
              to_traverse.insert(it->first, overlap);
              submask -= overlap;
              if (!submask)
                break;
            }
          }
        }
      }
      // Continue the traversal for anything below
      for (typename local::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
               to_traverse.begin();
           it != to_traverse.end(); it++)
      {
        const Rect<DIM, T> overlap = it->first->bounds.intersection(rect);
        new_subs += it->first->record_output_equivalence_set(
            set, overlap, it->second, tracker, tracker_space, subscriptions,
            remote_shard_rects, local_shard);
      }
      return new_subs;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::invalidate_all_previous_sets(const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      local::FieldMaskMap<EqKDNode<DIM, T> > to_invalidate_previous;
      {
        AutoLock n_lock(node_lock);
        invalidate_previous_sets(mask, to_invalidate_previous);
      }
      for (typename local::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
               to_invalidate_previous.begin();
           it != to_invalidate_previous.end(); it++)
      {
        it->first->invalidate_all_previous_sets(it->second);
        if (it->first->remove_reference())
          delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::invalidate_previous_sets(
        const FieldMask& mask,
        local::FieldMaskMap<EqKDNode<DIM, T> >& to_invalidate_previous)
    //--------------------------------------------------------------------------
    {
      if ((previous_sets != nullptr) &&
          !(mask * previous_sets->get_valid_mask()))
      {
        std::vector<EquivalenceSet*> to_delete;
        for (lng::FieldMaskMap<EquivalenceSet>::iterator it =
                 previous_sets->begin();
             it != previous_sets->end(); it++)
        {
          it.filter(mask);
          if (!it->second)
            to_delete.emplace_back(it->first);
        }
        for (EquivalenceSet* set : to_delete)
        {
          previous_sets->erase(set);
          if (set->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
            delete set;
        }
        if (previous_sets->empty())
        {
          delete previous_sets;
          previous_sets = nullptr;
        }
        else
          previous_sets->tighten_valid_mask();
      }
      if (!(mask * all_previous_below))
      {
        legion_assert((lefts != nullptr) && (rights != nullptr));
        all_previous_below -= mask;
        std::vector<EqKDNode<DIM, T>*> to_delete;
        for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::iterator it =
                 lefts->begin();
             it != lefts->end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          to_invalidate_previous.insert(it->first, overlap);
          it.filter(overlap);
          if (!it->second)
            // Don't remove the refernce, it's in to_invalidate_previous
            to_delete.emplace_back(it->first);
          else
            it->first->add_reference();
        }
        for (typename std::vector<EqKDNode<DIM, T>*>::const_iterator it =
                 to_delete.begin();
             it != to_delete.end(); it++)
          lefts->erase(*it);
        if (lefts->empty())
        {
          delete lefts;
          lefts = nullptr;
        }
        else
          lefts->tighten_valid_mask();
        to_delete.clear();
        for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::iterator it =
                 rights->begin();
             it != rights->end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          to_invalidate_previous.insert(it->first, overlap);
          it.filter(overlap);
          if (!it->second)
            // Don't remove the refernce, it's in to_invalidate_previous
            to_delete.emplace_back(it->first);
          else
            it->first->add_reference();
        }
        for (typename std::vector<EqKDNode<DIM, T>*>::const_iterator it =
                 to_delete.begin();
             it != to_delete.end(); it++)
          rights->erase(*it);
        if (rights->empty())
        {
          delete rights;
          rights = nullptr;
        }
        else
          rights->tighten_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDNode<DIM, T>::record_subscription(
        EqSetTracker* tracker, AddressSpaceID tracker_space,
        const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      if (subscriptions == nullptr)
        subscriptions =
            new lng::map<AddressSpaceID, lng::FieldMaskMap<EqSetTracker> >();
      lng::FieldMaskMap<EqSetTracker>& trackers =
          (*subscriptions)[tracker_space];
      lng::FieldMaskMap<EqSetTracker>::iterator finder = trackers.find(tracker);
      if (finder != trackers.end())
      {
        FieldMask new_fields = mask - finder->second;
        if (!new_fields)
          return 0;
        trackers.insert(tracker, new_fields);
        const unsigned total_fields = new_fields.pop_count();
        this->add_reference(total_fields);
        return total_fields;
      }
      else
      {
        trackers.insert(tracker, mask);
        // Add a reference for every field
        const unsigned total_fields = mask.pop_count();
        this->add_reference(total_fields);
        return total_fields;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::clone_sets(
        EqKDNode<DIM, T>* left, EqKDNode<DIM, T>* right, FieldMask mask,
        lng::FieldMaskMap<EquivalenceSet>*& sets, bool current)
    //--------------------------------------------------------------------------
    {
      legion_assert(sets != nullptr);
      std::vector<EquivalenceSet*> to_delete;
      for (lng::FieldMaskMap<EquivalenceSet>::iterator it = sets->begin();
           it != sets->end(); it++)
      {
        const FieldMask overlap = it->second & mask;
        if (!overlap)
          continue;
        left->record_set(it->first, overlap, current);
        right->record_set(it->first, overlap, current);
        it.filter(overlap);
        if (!it->second)
          to_delete.emplace_back(it->first);
        mask -= overlap;
        if (!mask)
          break;
      }
      for (EquivalenceSet* const set : to_delete)
      {
        sets->erase(set);
        if (set->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
          delete set;
      }
      if (sets->empty())
      {
        delete sets;
        sets = nullptr;
      }
      else
        sets->tighten_valid_mask();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::record_set(
        EquivalenceSet* set, const FieldMask& mask, bool current)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      if (current)
      {
        if (current_sets == nullptr)
          current_sets = new lng::FieldMaskMap<EquivalenceSet>();
        if (current_sets->insert(set, mask))
          set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
      }
      else
      {
        if (previous_sets == nullptr)
          previous_sets = new lng::FieldMaskMap<EquivalenceSet>();
        if (previous_sets->insert(set, mask))
          set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::find_local_equivalence_sets(
        local::FieldMaskMap<EquivalenceSet>& eq_sets, ShardID local_shard) const
    //--------------------------------------------------------------------------
    {
      // No need for the lock here since this should be done exclusively
      // while nothing else is modifying the state of this tree
      if (current_sets != nullptr)
      {
        for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
                 current_sets->begin();
             it != current_sets->end(); it++)
          eq_sets.insert(it->first, it->second);
      }
      if (previous_sets != nullptr)
      {
        // Only record previous sets if we didn't have current sets
        for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
                 previous_sets->begin();
             it != previous_sets->end(); it++)
        {
          if (current_sets != nullptr)
          {
            FieldMask mask = it->second - current_sets->get_valid_mask();
            if (!!mask)
              eq_sets.insert(it->first, mask);
          }
          else
            eq_sets.insert(it->first, it->second);
        }
      }
      if (lefts != nullptr)
      {
        for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
                 lefts->begin();
             it != lefts->end(); it++)
          it->first->find_local_equivalence_sets(eq_sets, local_shard);
      }
      if (rights != nullptr)
      {
        for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
                 rights->begin();
             it != rights->end(); it++)
          it->first->find_local_equivalence_sets(eq_sets, local_shard);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::find_shard_equivalence_sets(
        local::map<
            ShardID,
            local::map<RegionNode*, local::FieldMaskMap<EquivalenceSet> > >&
            eq_sets,
        ShardID source_shard, ShardID dst_lower_shard, ShardID dst_upper_shard,
        RegionNode* region) const
    //--------------------------------------------------------------------------
    {
      if ((dst_lower_shard == dst_upper_shard) ||
          (this->bounds.volume() <= EqKDSharded<DIM, T>::MIN_SPLIT_SIZE))
        find_local_equivalence_sets(
            eq_sets[dst_lower_shard][region], source_shard);
      else
        // We still need to break up this rectangle the same way that it will
        // be broken up by a EqKDSharded node for these shards
        find_shard_equivalence_sets(
            this->bounds, eq_sets, dst_lower_shard, dst_upper_shard, region);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::find_shard_equivalence_sets(
        const Rect<DIM, T>& rect,
        local::map<
            ShardID,
            local::map<RegionNode*, local::FieldMaskMap<EquivalenceSet> > >&
            eq_sets,
        ShardID dst_lower_shard, ShardID dst_upper_shard,
        RegionNode* region) const
    //--------------------------------------------------------------------------
    {
      legion_assert(dst_lower_shard < dst_upper_shard);
      // Split this the same way that EqKDSharded will split it to find
      // the rectangle that we need to use to search for equivalence sets
      // Check to see if we hit the minimum size
      if (rect.volume() <= EqKDSharded<DIM, T>::MIN_SPLIT_SIZE)
      {
        find_rect_equivalence_sets(rect, eq_sets[dst_lower_shard][region]);
        return;
      }
      // Find the largest dimension and split it in half
      // Note we cannot use the rectangle to guide our splitting plane here
      // like we do with the EqKDNode because this splitting needs to be
      // deterministic across the shards
      T split = 0;
      int dim = -1;
      T largest = 0;
      for (int d = 0; d < DIM; d++)
      {
        T diff = rect.hi[d] - rect.lo[d];
        if (diff <= largest)
          continue;
        largest = diff;
        dim = d;
        split = rect.lo[d] + (diff / 2);
      }
      legion_assert(dim >= 0);
      Rect<DIM, T> left_bounds = rect;
      Rect<DIM, T> right_bounds = rect;
      left_bounds.hi[dim] = split;
      right_bounds.lo[dim] = split + 1;
      // Find the splitting of the shards
      ShardID diff = dst_upper_shard - dst_lower_shard;
      ShardID mid = dst_lower_shard + (diff / 2);
      if (dst_lower_shard == mid)
        find_rect_equivalence_sets(left_bounds, eq_sets[mid][region]);
      else
        find_shard_equivalence_sets(
            left_bounds, eq_sets, dst_lower_shard, mid, region);
      if ((mid + 1) == dst_upper_shard)
        find_rect_equivalence_sets(right_bounds, eq_sets[mid + 1][region]);
      else
        find_shard_equivalence_sets(
            right_bounds, eq_sets, mid + 1, dst_upper_shard, region);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::find_rect_equivalence_sets(
        const Rect<DIM, T>& rect,
        local::FieldMaskMap<EquivalenceSet>& eq_sets) const
    //--------------------------------------------------------------------------
    {
      legion_assert(this->bounds.contains(rect));
      std::vector<EqKDNode<DIM, T>*> to_traverse;
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        if (current_sets != nullptr)
        {
          for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
                   current_sets->begin();
               it != current_sets->end(); it++)
            eq_sets.insert(it->first, it->second);
        }
        if (previous_sets != nullptr)
        {
          FieldMask remaining = previous_sets->get_valid_mask();
          if (current_sets != nullptr)
            remaining -= current_sets->get_valid_mask();
          if (!!remaining)
          {
            for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
                     current_sets->begin();
                 it != current_sets->end(); it++)
            {
              const FieldMask overlap = it->second & remaining;
              if (!overlap)
                continue;
              eq_sets.insert(it->first, overlap);
              remaining -= overlap;
              if (!remaining)
                break;
            }
          }
        }
        FieldMask remaining;
        if (lefts != nullptr)
        {
          for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator
                   it = lefts->begin();
               it != lefts->end(); it++)
          {
            const Rect<DIM, T> overlap = it->first->bounds.intersection(rect);
            if (!overlap.empty())
            {
              to_traverse.emplace_back(it->first);
              if (overlap != rect)
                remaining |= it->second;
            }
            else
              remaining |= it->second;
          }
        }
        if (!!remaining)
        {
          for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator
                   it = rights->begin();
               it != rights->end(); it++)
          {
            const FieldMask overlap = it->second & remaining;
            if (!overlap)
              continue;
            legion_assert(it->first->bounds.overlaps(rect));
            to_traverse.emplace_back(it->first);
            remaining -= overlap;
            if (!remaining)
              break;
          }
        }
      }
      for (typename std::vector<EqKDNode<DIM, T>*>::const_iterator it =
               to_traverse.begin();
           it != to_traverse.end(); it++)
      {
        const Rect<DIM, T> overlap = (*it)->bounds.intersection(rect);
        legion_assert(!overlap.empty());
        (*it)->find_rect_equivalence_sets(overlap, eq_sets);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::invalidate_tree(
        const Rect<DIM, T>& rect, const FieldMask& mask,
        std::vector<RtEvent>& invalidated, bool move_to_previous,
        FieldMask* parent_all_previous)
    //--------------------------------------------------------------------------
    {
      legion_assert(this->bounds.contains(rect));
      // This is very important: invalidations are protected from the root
      // of the equivalence set tree with an exclusive lock, that means that
      // any invalidation traversing the tree does not need to protect against
      // races with compute_equivalence_set traversals since.
      // This property is what allows us to safely propagate the information
      // about the all_previous_below safely up the tree. While it's expensive
      // to have a whole-tree lock like this, invalidations being done by
      // refinements should be pretty rare so it shouldn't impact performance.
      // Note we still need to take the node lock here in this node because
      // calls like record_equivalence_set or cancel_subscription can still
      // be coming back asynchronously to touch data structures in these nodes
      FieldMask remaining = mask;
      local::FieldMaskMap<EqKDNode<DIM, T> > to_traverse;
      local::FieldMaskMap<EqKDNode<DIM, T> > to_invalidate_previous;
      local::map<AddressSpaceID, local::FieldMaskMap<EqSetTracker> >
          to_invalidate;
      {
        // Take the lock to protect against data structures that might
        // be racing with recording equivalence sets or cancelling subscriptions
        AutoLock n_lock(node_lock);
        // First check to see if there are any current sets which
        // haven't had their previous sets filtered yet. We have to
        // do this first to ensure that the lefts and rights and data
        // structures only contain real lefts and rights and not ones
        // that we were just holding on to for previous reasons
        if (current_set_preconditions != nullptr)
        {
          for (shrt::map<RtEvent, FieldMask>::iterator it =
                   current_set_preconditions->begin();
               it != current_set_preconditions->end();
               /*nothing*/)
          {
            if (!(it->second * mask))
            {
              // Better have triggered by the point we're doing
              // this invalidation or something is wrong with the
              // mapping dependences for this refinement operation
              legion_assert(it->first.has_triggered());
              invalidate_previous_sets(it->second, to_invalidate_previous);
              shrt::map<RtEvent, FieldMask>::iterator to_delete = it++;
              current_set_preconditions->erase(to_delete);
            }
            else
              it++;
          }
          if (current_set_preconditions->empty())
          {
            delete current_set_preconditions;
            current_set_preconditions = nullptr;
          }
        }
        // Special case for where we're just invalidating everything so we
        // can filter things from the previous sets eagerly
        if (!move_to_previous && (previous_sets != nullptr) &&
            !(mask * previous_sets->get_valid_mask()))
        {
          std::vector<EquivalenceSet*> to_delete;
          for (lng::FieldMaskMap<EquivalenceSet>::iterator it =
                   previous_sets->begin();
               it != previous_sets->end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          for (EquivalenceSet* const set : to_delete)
          {
            previous_sets->erase(set);
            if (set->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
              delete set;
          }
          if (previous_sets->empty())
          {
            delete previous_sets;
            previous_sets = nullptr;
          }
          else
            previous_sets->tighten_valid_mask();
        }
        // Check to see which fields we have current equivalence sets
        // for on this node as these are the ones that will ultimately
        // have to be invalidated
        FieldMask current_mask;
        if (current_sets != nullptr)
          current_mask = mask & current_sets->get_valid_mask();
        if (!!current_mask)
        {
          // No matter what we're going to do here we need to
          // invalidate the subscriptions so do that first
          if (subscriptions != nullptr)
          {
            for (lng::map<AddressSpaceID, lng::FieldMaskMap<EqSetTracker> >::
                     iterator sit = subscriptions->begin();
                 sit != subscriptions->end();
                 /*nothing*/)
            {
              if (sit->second.get_valid_mask() * current_mask)
              {
                sit++;
                continue;
              }
              local::FieldMaskMap<EqSetTracker>& invalidations =
                  to_invalidate[sit->first];
              if (!!(sit->second.get_valid_mask() - current_mask))
              {
                // Filter out specific subscriptions
                std::vector<EqSetTracker*> to_delete;
                for (lng::FieldMaskMap<EqSetTracker>::iterator it =
                         sit->second.begin();
                     it != sit->second.end(); it++)
                {
                  const FieldMask overlap = current_mask & it->second;
                  if (!overlap)
                    continue;
                  invalidations.insert(it->first, overlap);
                  it.filter(overlap);
                  if (!it->second)
                    to_delete.emplace_back(it->first);
                }
                for (EqSetTracker* tracker : to_delete)
                  sit->second.erase(tracker);
              }
              else  // Invalidating the subscriptions for all fields
              {
                for (lng::FieldMaskMap<EqSetTracker>::iterator it =
                         sit->second.begin();
                     it != sit->second.end(); it++)
                  invalidations.insert(it->first, it->second);
                sit->second.clear();
              }
              if (sit->second.empty())
              {
                lng::map<AddressSpaceID, lng::FieldMaskMap<EqSetTracker> >::
                    iterator to_delete = sit++;
                subscriptions->erase(to_delete);
              }
              else
              {
                sit->second.tighten_valid_mask();
                sit++;
              }
            }
            if (subscriptions->empty())
            {
              delete subscriptions;
              subscriptions = nullptr;
            }
          }
          if (this->bounds == rect)
          {
            // We know these fields no longer need to be filtered
            // since we're going to filter them here
            remaining -= current_mask;
            // We have permissions to perform all the invalidations
            // here without needing to refine. First filter the
            // previous sets for these fields since we know we're
            // going to have current sets to flush back to the
            // previous sets for them
            if (move_to_previous)
            {
              if (previous_sets == nullptr)
                previous_sets = new lng::FieldMaskMap<EquivalenceSet>();
              else if (!(current_mask * previous_sets->get_valid_mask()))
              {
                // Very important that we only filter previous sets
                // if we actually have current sets to replace them with
                std::vector<EquivalenceSet*> to_delete;
                for (lng::FieldMaskMap<EquivalenceSet>::iterator it =
                         previous_sets->begin();
                     it != previous_sets->end(); it++)
                {
                  it.filter(current_mask);
                  if (!it->second)
                    to_delete.emplace_back(it->first);
                }
                for (EquivalenceSet* set : to_delete)
                {
                  previous_sets->erase(set);
                  if (set->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
                    delete set;
                }
                // No need to delete previous sets or tighten its valid
                // mask since we know that there will be current sets
                // getting stored into the previous sets for all those fields
              }
            }
            // Now we can invalidate the current sets and flush them
            // back to the previous sets if we're moving them
            std::vector<EquivalenceSet*> to_delete;
            for (lng::FieldMaskMap<EquivalenceSet>::iterator it =
                     current_sets->begin();
                 it != current_sets->end(); it++)
            {
              const FieldMask overlap = it->second & current_mask;
              if (!overlap)
                continue;
              if (move_to_previous && previous_sets->insert(it->first, overlap))
                it->first->add_base_gc_ref(DISJOINT_COMPLETE_REF);
              it.filter(overlap);
              if (!it->second)
                to_delete.emplace_back(it->first);
              current_mask -= overlap;
              if (!current_mask)
                break;
            }
            // Should have moved something over for all fields
            legion_assert(!current_mask);
            for (EquivalenceSet* set : to_delete)
            {
              current_sets->erase(set);
              if (set->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
                delete set;
            }
            if (current_sets->empty())
            {
              delete current_sets;
              current_sets = nullptr;
            }
            else
              current_sets->tighten_valid_mask();
          }
          else
          {
            // It's unsound for us to partially invalidate this node
            // if the rect is only a subset of the bounds because if
            // we don't have permissions to perform the invalidations
            // for those points then we might end up trying to change
            // the equivalence sets while other operations are trying
            // to mutate those equivalence sets leading to races. To
            // avoid this we'll check to see if we have any current
            // equivalence sets
            if (lefts != nullptr)
            {
              FieldMask refine = current_mask - lefts->get_valid_mask();
              if (!!refine)
                refine_node(rect, refine, true /*refine current*/);
            }
            else
              refine_node(rect, current_mask, true /*refine current*/);
          }
        }
        // Now see if we need to continue the traversal for any remaining fields
        if (!!remaining)
        {
          // We can skip performing invalidations if we know that everything
          // below is already previous-only
          if (!!all_previous_below)
            remaining -= all_previous_below;
          // Find the nodes to traverse below
          if (!!remaining && (lefts != nullptr) &&
              !(remaining * lefts->get_valid_mask()))
          {
            FieldMask right_mask;
            for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator
                     it = lefts->begin();
                 it != lefts->end(); it++)
            {
              const FieldMask overlap = it->second & remaining;
              if (!overlap)
                continue;
              // Compute the overlap
              const Rect<DIM, T> intersection =
                  rect.intersection(it->first->bounds);
              if (!intersection.empty())
              {
                to_traverse.insert(it->first, overlap);
                if (intersection != rect)
                  right_mask |= overlap;
              }
              else
                right_mask |= overlap;
              remaining -= overlap;
              if (!remaining)
                break;
            }
            if (!!right_mask)
            {
              for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator
                       it = rights->begin();
                   it != rights->end(); it++)
              {
                const FieldMask overlap = it->second & right_mask;
                if (!overlap)
                  continue;
                legion_assert(rect.overlaps(it->first->bounds));
                to_traverse.insert(it->first, overlap);
                right_mask -= overlap;
                if (!right_mask)
                  break;
              }
              legion_assert(!right_mask);
            }
          }
        }
      }
      if (!to_invalidate.empty())
        EqSetTracker::invalidate_subscriptions(
            this, to_invalidate, invalidated);
      for (typename local::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
               to_invalidate_previous.begin();
           it != to_invalidate_previous.end(); it++)
      {
        it->first->invalidate_all_previous_sets(it->second);
        if (it->first->remove_reference())
          delete it->first;
      }
      // Now do the traversal for the invalidation below
      bool has_child_previous = false;
      for (typename local::FieldMaskMap<EqKDNode<DIM, T> >::iterator it =
               to_traverse.begin();
           it != to_traverse.end(); it++)
      {
        const Rect<DIM, T> intersection = rect.intersection(it->first->bounds);
        legion_assert(!intersection.empty());
        FieldMask child_previous;
        it->first->invalidate_tree(
            intersection, it->second, invalidated, move_to_previous,
            &child_previous);
        // Clear the fields
        it.clear();
        // Save any below
        if (!!child_previous)
        {
          it.merge(child_previous);
          has_child_previous = true;
        }
      }
      // Record the any all-previous fields at this child
      if (has_child_previous || (parent_all_previous != nullptr))
      {
        // Need to retake the lock here because record_equivalence_set
        // could be calling back in here and mutating the previous sets
        // while we're try to read it which can lead to the wrong set
        // of fields being recorded
        AutoLock n_lock(node_lock);
        if (has_child_previous)
        {
          for (typename local::FieldMaskMap<EqKDNode<DIM, T> >::iterator it =
                   to_traverse.begin();
               it != to_traverse.end(); it++)
            if (!!it->second)
              record_child_all_previous(it->first, it->second);
        }
        if (parent_all_previous != nullptr)
        {
          *parent_all_previous = all_previous_below;
          if (previous_sets != nullptr)
            *parent_all_previous |= previous_sets->get_valid_mask();
          // Only return fields that were invalidated
          *parent_all_previous &= mask;
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::record_child_all_previous(
        EqKDNode<DIM, T>* child, FieldMask mask)
    //--------------------------------------------------------------------------
    {
      if (!!all_previous_below)
      {
        // If the fields are already all-previous below then we're done
        mask -= all_previous_below;
        if (!mask)
          return;
      }
      if (child_previous_below != nullptr)
      {
        // See if the other child is already all-previous for these fields
        if (!(mask * child_previous_below->get_valid_mask()))
        {
          std::vector<EqKDNode<DIM, T>*> to_delete;
          for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::iterator it =
                   child_previous_below->begin();
               it != child_previous_below->end(); it++)
          {
            if (it->first == child)
              continue;
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            // The other child is already all-previous for these fields
            // so we can record them as all previous now
            all_previous_below |= overlap;
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
            mask -= overlap;
            if (!mask)
              break;
          }
          for (typename std::vector<EqKDNode<DIM, T>*>::const_iterator it =
                   to_delete.begin();
               it != to_delete.end(); it++)
            child_previous_below->erase(*it);
          if (!mask)
          {
            if (child_previous_below->empty())
            {
              delete child_previous_below;
              child_previous_below = nullptr;
            }
            else
              child_previous_below->tighten_valid_mask();
            return;
          }
          else
            child_previous_below->tighten_valid_mask();
        }
      }
      else
        child_previous_below = new lng::FieldMaskMap<EqKDNode<DIM, T> >();
      child_previous_below->insert(child, mask);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::invalidate_shard_tree_remote(
        const Rect<DIM, T>& rect, const FieldMask& mask,
        std::vector<RtEvent>& invalidated,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      invalidate_tree(rect, mask, invalidated, true /*move to previous*/);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDNode<DIM, T>::cancel_subscription(
        EqSetTracker* tracker, AddressSpaceID space, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      if (subscriptions == nullptr)
        return 0;
      lng::map<AddressSpaceID, lng::FieldMaskMap<EqSetTracker> >::iterator
          subscription_finder = subscriptions->find(space);
      if (subscription_finder == subscriptions->end())
        return 0;
      lng::FieldMaskMap<EqSetTracker>::iterator finder =
          subscription_finder->second.find(tracker);
      if (finder == subscription_finder->second.end())
        return 0;
      const FieldMask overlap = mask & finder->second;
      if (!overlap)
        return 0;
      finder.filter(overlap);
      if (!finder->second)
      {
        subscription_finder->second.erase(finder);
        if (subscription_finder->second.empty())
        {
          subscriptions->erase(subscription_finder);
          if (subscriptions->empty())
          {
            delete subscriptions;
            subscriptions = nullptr;
          }
        }
        else
          subscription_finder->second.tighten_valid_mask();
      }
      else
        subscription_finder->second.tighten_valid_mask();
      return overlap.pop_count();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::find_trace_local_sets(
        const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
        ShardID local_shard,
        std::map<EquivalenceSet*, unsigned>& local_sets) const
    //--------------------------------------------------------------------------
    {
      if (this->bounds.empty())
        return;
      legion_assert(this->bounds.contains(rect));
      local::FieldMaskMap<EqKDNode<DIM, T> > to_traverse;
      {
        FieldMask remaining = mask;
        AutoLock n_lock(node_lock, false /*exclusive*/);
        if ((current_sets != nullptr) &&
            !(remaining * current_sets->get_valid_mask()))
        {
          for (typename lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
                   current_sets->begin();
               it != current_sets->end(); it++)
          {
            if (mask * it->second)
              continue;
            local_sets[it->first] = req_index;
          }
          remaining -= current_sets->get_valid_mask();
          if (!remaining)
            return;
        }
        if ((previous_sets != nullptr) &&
            !(remaining * previous_sets->get_valid_mask()))
        {
          for (typename lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
                   previous_sets->begin();
               it != previous_sets->end(); it++)
          {
            if (mask * it->second)
              continue;
            local_sets[it->first] = req_index;
          }
          remaining -= previous_sets->get_valid_mask();
          if (!remaining)
            return;
        }
        if ((lefts != nullptr) && !(lefts->get_valid_mask() * remaining))
        {
          legion_assert(rights != nullptr);
          legion_assert(!(rights->get_valid_mask() * remaining));
          for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator
                   it = lefts->begin();
               it != lefts->end(); it++)
          {
            const FieldMask overlap = remaining & it->second;
            if (!overlap)
              continue;
            if (!it->first->bounds.overlaps(rect))
              continue;
            to_traverse.insert(it->first, overlap);
          }
          for (typename lng::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator
                   it = rights->begin();
               it != rights->end(); it++)
          {
            const FieldMask overlap = remaining & it->second;
            if (!overlap)
              continue;
            if (!it->first->bounds.overlaps(rect))
              continue;
            to_traverse.insert(it->first, overlap);
          }
        }
      }
      for (typename local::FieldMaskMap<EqKDNode<DIM, T> >::const_iterator it =
               to_traverse.begin();
           it != to_traverse.end(); it++)
      {
        const Rect<DIM, T> overlap = it->first->bounds.intersection(rect);
        legion_assert(!overlap.empty());
        it->first->find_trace_local_sets(
            overlap, it->second, req_index, local_shard, local_sets);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM, T>::find_shard_trace_local_sets(
        const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
        std::map<EquivalenceSet*, unsigned>& current_sets,
        local::map<ShardID, FieldMask>& remote_shards, ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      find_trace_local_sets(rect, mask, req_index, local_shard, current_sets);
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set KD Sparse
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDSparse<DIM, T>::EqKDSparse(
        const Rect<DIM, T>& bound, const std::vector<Rect<DIM, T> >& rects)
      : EqKDTreeT<DIM, T>(bound)
    //--------------------------------------------------------------------------
    {
      if (rects.size() <= LEGION_MAX_BVH_FANOUT)
      {
        // Base case of a small enough number of children
        children.reserve(rects.size());
        for (typename std::vector<Rect<DIM, T> >::const_iterator it =
                 rects.begin();
             it != rects.end(); it++)
        {
          EqKDNode<DIM, T>* child = new EqKDNode<DIM, T>(*it);
          child->add_reference();
          children.emplace_back(child);
        }
        return;
      }
      // Unlike some of our other KDNode implementations, we know that all of
      // these rectangles are non-overlapping with each other which means that
      // we should always be able to find good splitting planes
      Rect<DIM, T> best_left_bounds, best_right_bounds;
      std::vector<Rect<DIM, T> > best_left_set, best_right_set;
      bool success = KDTree::compute_best_splitting_plane<DIM, T>(
          bound, rects, best_left_bounds, best_right_bounds, best_left_set,
          best_right_set);
      // See if we had at least one good refinement
      if (success)
      {
        EqKDSparse<DIM, T>* left =
            new EqKDSparse<DIM, T>(best_left_bounds, best_left_set);
        left->add_reference();
        children.emplace_back(left);
        EqKDSparse<DIM, T>* right =
            new EqKDSparse<DIM, T>(best_right_bounds, best_right_set);
        right->add_reference();
        children.emplace_back(right);
      }
      else
      {
        Warning warn;
        warn << "Failed to find a refinement for Equivalence Set KD tree with "
             << DIM << " dimensions and " << rects.size()
             << " rectangles. Please report your application to "
             << "the Legion developers' mailing list.";
        warn.raise();
        // If we make it here then we couldn't find a splitting plane to refine
        // anymore so just record all the subrects as our rects
        children.reserve(rects.size());
        for (typename std::vector<Rect<DIM, T> >::const_iterator it =
                 rects.begin();
             it != rects.end(); it++)
        {
          EqKDNode<DIM, T>* child = new EqKDNode<DIM, T>(*it);
          child->add_reference();
          children.emplace_back(child);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDSparse<DIM, T>::~EqKDSparse(void)
    //--------------------------------------------------------------------------
    {
      for (EqKDTreeT<DIM, T>* const child : children)
        if (child->remove_reference())
          delete child;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM, T>::initialize_set(
        EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
        ShardID local_shard, bool current)
    //--------------------------------------------------------------------------
    {
      for (typename std::vector<EqKDTreeT<DIM, T>*>::const_iterator it =
               children.begin();
           it != children.end(); it++)
      {
        const Rect<DIM, T> overlap = rect.intersection((*it)->bounds);
        if (!overlap.empty())
          (*it)->initialize_set(set, overlap, mask, local_shard, current);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM, T>::compute_equivalence_sets(
        const Rect<DIM, T>& rect, const FieldMask& mask,
        const std::vector<EqSetTracker*>& trackers,
        const std::vector<AddressSpaceID>& tracker_spaces,
        std::vector<unsigned>& new_tracker_references,
        op::FieldMaskMap<EquivalenceSet>& eq_sets,
        std::vector<RtEvent>& pending_sets,
        op::FieldMaskMap<EqKDTree>& subscriptions,
        op::FieldMaskMap<EqKDTree>& to_create,
        op::map<EqKDTree*, Domain>& creation_rects,
        op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      for (typename std::vector<EqKDTreeT<DIM, T>*>::const_iterator it =
               children.begin();
           it != children.end(); it++)
      {
        const Rect<DIM, T> overlap = rect.intersection((*it)->bounds);
        if (!overlap.empty())
          (*it)->compute_equivalence_sets(
              overlap, mask, trackers, tracker_spaces, new_tracker_references,
              eq_sets, pending_sets, subscriptions, to_create, creation_rects,
              creation_srcs, remote_shard_rects, local_shard);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM, T>::record_equivalence_set(
        EquivalenceSet* set, const FieldMask& mask, RtEvent ready,
        const CollectiveMapping& creator_spaces,
        const std::vector<EqSetTracker*>& creators)
    //--------------------------------------------------------------------------
    {
      // This should never be called on a sparse tree node
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDSparse<DIM, T>::record_output_equivalence_set(
        EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
        EqSetTracker* tracker, AddressSpaceID tracker_space,
        local::FieldMaskMap<EqKDTree>& subscriptions,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      unsigned new_subs = 0;
      for (typename std::vector<EqKDTreeT<DIM, T>*>::const_iterator it =
               children.begin();
           it != children.end(); it++)
      {
        const Rect<DIM, T> overlap = rect.intersection((*it)->bounds);
        if (!overlap.empty())
          new_subs += (*it)->record_output_equivalence_set(
              set, overlap, mask, tracker, tracker_space, subscriptions,
              remote_shard_rects, local_shard);
      }
      return new_subs;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM, T>::find_local_equivalence_sets(
        local::FieldMaskMap<EquivalenceSet>& eq_sets, ShardID local_shard) const
    //--------------------------------------------------------------------------
    {
      for (typename std::vector<EqKDTreeT<DIM, T>*>::const_iterator it =
               children.begin();
           it != children.end(); it++)
        (*it)->find_local_equivalence_sets(eq_sets, local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM, T>::find_shard_equivalence_sets(
        local::map<
            ShardID,
            local::map<RegionNode*, local::FieldMaskMap<EquivalenceSet> > >&
            eq_sets,
        ShardID source_shard, ShardID dst_lower_shard, ShardID dst_upper_shard,
        RegionNode* region) const
    //--------------------------------------------------------------------------
    {
      // TODO
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM, T>::invalidate_tree(
        const Rect<DIM, T>& rect, const FieldMask& mask,
        std::vector<RtEvent>& invalidated, bool move_to_previous,
        FieldMask* parent_all_previous)
    //--------------------------------------------------------------------------
    {
      for (typename std::vector<EqKDTreeT<DIM, T>*>::const_iterator it =
               children.begin();
           it != children.end(); it++)
      {
        const Rect<DIM, T> overlap = rect.intersection((*it)->bounds);
        if (!overlap.empty())
          (*it)->invalidate_tree(
              overlap, mask, invalidated, move_to_previous,
              parent_all_previous);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM, T>::invalidate_shard_tree_remote(
        const Rect<DIM, T>& rect, const FieldMask& mask,
        std::vector<RtEvent>& invalidated,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      invalidate_tree(rect, mask, invalidated, true /*move to previous*/);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDSparse<DIM, T>::cancel_subscription(
        EqSetTracker* tracker, AddressSpaceID space, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      // should never be called on sparse nodes since they don't track
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM, T>::find_trace_local_sets(
        const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
        ShardID local_shard,
        std::map<EquivalenceSet*, unsigned>& current_sets) const
    //--------------------------------------------------------------------------
    {
      legion_assert(this->bounds.contains(rect));
      for (typename std::vector<EqKDTreeT<DIM, T>*>::const_iterator it =
               children.begin();
           it != children.end(); it++)
      {
        const Rect<DIM, T> overlap = (*it)->bounds.intersection(rect);
        if (overlap.empty())
          continue;
        (*it)->find_trace_local_sets(
            overlap, mask, req_index, local_shard, current_sets);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM, T>::find_shard_trace_local_sets(
        const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
        std::map<EquivalenceSet*, unsigned>& current_sets,
        local::map<ShardID, FieldMask>& remote_shards, ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      find_trace_local_sets(rect, mask, req_index, local_shard, current_sets);
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set KD Sharded
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDSharded<DIM, T>::EqKDSharded(
        const Rect<DIM, T>& rect, ShardID low, ShardID high)
      : EqKDTreeT<DIM, T>(rect), lower(low), upper(high), left(nullptr),
        right(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDSharded<DIM, T>::~EqKDSharded(void)
    //--------------------------------------------------------------------------
    {
      EqKDTreeT<DIM, T>* next = left.load();
      if ((next != nullptr) && next->remove_reference())
        delete next;
      next = right.load();
      if ((next != nullptr) && next->remove_reference())
        delete next;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM, T>::initialize_set(
        EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
        ShardID local_shard, bool current)
    //--------------------------------------------------------------------------
    {
      legion_assert(this->bounds.contains(rect));
      legion_assert(lower <= local_shard);
      legion_assert(local_shard <= upper);
      EqKDTreeT<DIM, T>* next = right.load();
      // Check to see if we've reached the bottom
      if (next == nullptr)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and we can make a local node or node
          if (lower == local_shard)
          {
            EqKDTreeT<DIM, T>* local = left.load();
            if (local == nullptr)
              local = refine_local();
            local->initialize_set(set, rect, mask, local_shard, current);
          }
          return;
        }
        else  // Create the refinement
        {
          refine_node();
          next = right.load();
        }
      }
      legion_assert(next != nullptr);
      legion_assert(lower != upper);
      // We only need to traverse down the child that has our local shard
      ShardID diff = upper - lower;
      ShardID mid = lower + (diff / 2);
      if (local_shard <= mid)
        next = left.load();
      const Rect<DIM, T> overlap = next->bounds.intersection(rect);
      if (!overlap.empty())
        next->initialize_set(set, overlap, mask, local_shard, current);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM, T>::compute_equivalence_sets(
        const Rect<DIM, T>& rect, const FieldMask& mask,
        const std::vector<EqSetTracker*>& trackers,
        const std::vector<AddressSpaceID>& tracker_spaces,
        std::vector<unsigned>& new_tracker_references,
        op::FieldMaskMap<EquivalenceSet>& eq_sets,
        std::vector<RtEvent>& pending_sets,
        op::FieldMaskMap<EqKDTree>& subscriptions,
        op::FieldMaskMap<EqKDTree>& to_create,
        op::map<EqKDTree*, Domain>& creation_rects,
        op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      legion_assert(this->bounds.contains(rect));
      EqKDTreeT<DIM, T>* next = right.load();
      // Check to see if we've reached the bottom
      if (next == nullptr)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and we can make a local node or node
          if (lower == local_shard)
          {
            EqKDTreeT<DIM, T>* local = left.load();
            if (local == nullptr)
              local = refine_local();
            local->compute_equivalence_sets(
                rect, mask, trackers, tracker_spaces, new_tracker_references,
                eq_sets, pending_sets, subscriptions, to_create, creation_rects,
                creation_srcs, remote_shard_rects, local_shard);
          }
          else
            remote_shard_rects[lower][rect] |= mask;
          // We're done
          return;
        }
        else  // Create the refinement
        {
          refine_node();
          next = right.load();
        }
      }
      legion_assert(next != nullptr);
      const Rect<DIM, T> right_overlap = next->bounds.intersection(rect);
      if (!right_overlap.empty())
        next->compute_equivalence_sets(
            right_overlap, mask, trackers, tracker_spaces,
            new_tracker_references, eq_sets, pending_sets, subscriptions,
            to_create, creation_rects, creation_srcs, remote_shard_rects,
            local_shard);
      next = left.load();
      legion_assert(next != nullptr);
      const Rect<DIM, T> left_overlap = next->bounds.intersection(rect);
      if (!left_overlap.empty())
        next->compute_equivalence_sets(
            left_overlap, mask, trackers, tracker_spaces,
            new_tracker_references, eq_sets, pending_sets, subscriptions,
            to_create, creation_rects, creation_srcs, remote_shard_rects,
            local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t EqKDSharded<DIM, T>::get_total_volume(void) const
    //--------------------------------------------------------------------------
    {
      return this->bounds.volume();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDTreeT<DIM, T>* EqKDSharded<DIM, T>::refine_local(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(right.load() == nullptr);
      EqKDNode<DIM, T>* next = new EqKDNode<DIM, T>(this->bounds);
      EqKDTreeT<DIM, T>* expected = nullptr;
      if (left.compare_exchange_strong(expected, next))
      {
        next->add_reference();
        return next;
      }
      else
      {
        delete next;
        return expected;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM, T>::refine_node(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(lower < upper);
      // Find the largest dimension and split it in half
      // Note we cannot use the rectangle to guide our splitting plane here
      // like we do with the EqKDNode because this splitting needs to be
      // deterministic across the shards
      T split = 0;
      int dim = -1;
      T largest = 0;
      for (int d = 0; d < DIM; d++)
      {
        T diff = this->bounds.hi[d] - this->bounds.lo[d];
        if (diff <= largest)
          continue;
        largest = diff;
        dim = d;
        split = this->bounds.lo[d] + (diff / 2);
      }
      legion_assert(dim >= 0);
      Rect<DIM, T> left_bounds = this->bounds;
      Rect<DIM, T> right_bounds = this->bounds;
      left_bounds.hi[dim] = split;
      right_bounds.lo[dim] = split + 1;
      // Find the splitting of the shards
      ShardID diff = upper - lower;
      ShardID mid = lower + (diff / 2);
      // Do left before right so that as soon as right is set then left is too
      EqKDSharded<DIM, T>* next =
          new EqKDSharded<DIM, T>(left_bounds, lower, mid);
      EqKDTreeT<DIM, T>* expected = nullptr;
      if (left.compare_exchange_strong(expected, next))
        next->add_reference();
      else
        delete next;
      next = new EqKDSharded<DIM, T>(right_bounds, mid + 1, upper);
      expected = nullptr;
      if (right.compare_exchange_strong(expected, next))
        next->add_reference();
      else
        delete next;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM, T>::record_equivalence_set(
        EquivalenceSet* set, const FieldMask& mask, RtEvent ready,
        const CollectiveMapping& creator_spaces,
        const std::vector<EqSetTracker*>& creators)
    //--------------------------------------------------------------------------
    {
      // This should never be called on a sharded tree node
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDSharded<DIM, T>::record_output_equivalence_set(
        EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
        EqSetTracker* tracker, AddressSpaceID tracker_space,
        local::FieldMaskMap<EqKDTree>& subscriptions,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      legion_assert(this->bounds.contains(rect));
      EqKDTreeT<DIM, T>* next = right.load();
      // Check to see if we've reached the bottom
      if (next == nullptr)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and we can make a local node or node
          if (lower == local_shard)
          {
            EqKDTreeT<DIM, T>* local = left.load();
            if (local == nullptr)
              local = refine_local();
            return local->record_output_equivalence_set(
                set, rect, mask, tracker, tracker_space, subscriptions,
                remote_shard_rects, local_shard);
          }
          else
          {
            remote_shard_rects[lower][rect] |= mask;
            return 0;
          }
        }
        else  // Create the refinement
        {
          refine_node();
          next = right.load();
        }
      }
      unsigned new_subs = 0;
      legion_assert(next != nullptr);
      const Rect<DIM, T> right_overlap = next->bounds.intersection(rect);
      if (!right_overlap.empty())
        new_subs += next->record_output_equivalence_set(
            set, right_overlap, mask, tracker, tracker_space, subscriptions,
            remote_shard_rects, local_shard);
      next = left.load();
      legion_assert(next != nullptr);
      const Rect<DIM, T> left_overlap = next->bounds.intersection(rect);
      if (!left_overlap.empty())
        new_subs += next->record_output_equivalence_set(
            set, left_overlap, mask, tracker, tracker_space, subscriptions,
            remote_shard_rects, local_shard);
      return new_subs;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM, T>::find_local_equivalence_sets(
        local::FieldMaskMap<EquivalenceSet>& eq_sets, ShardID local_shard) const
    //--------------------------------------------------------------------------
    {
      legion_assert(lower <= local_shard);
      legion_assert(local_shard <= upper);
      EqKDTreeT<DIM, T>* next = right.load();
      // Check to see if we've reached the bottom
      if (next == nullptr)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and we can make a local node or node
          if (lower == local_shard)
          {
            EqKDTreeT<DIM, T>* local = left.load();
            if (local != nullptr)
              local->find_local_equivalence_sets(eq_sets, local_shard);
          }
        }
        return;
      }
      legion_assert(next != nullptr);
      legion_assert(lower != upper);
      // We only need to traverse down the child that has our local shard
      ShardID diff = upper - lower;
      ShardID mid = lower + (diff / 2);
      if (local_shard <= mid)
        next = left.load();
      next->find_local_equivalence_sets(eq_sets, local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM, T>::find_shard_equivalence_sets(
        local::map<
            ShardID,
            local::map<RegionNode*, local::FieldMaskMap<EquivalenceSet> > >&
            eq_sets,
        ShardID source_shard, ShardID dst_lower_shard, ShardID dst_upper_shard,
        RegionNode* region) const
    //--------------------------------------------------------------------------
    {
      // Keep going to the local shard and split the dst shards along the way
      legion_assert(lower <= source_shard);
      legion_assert(source_shard <= upper);
      EqKDTreeT<DIM, T>* next = right.load();
      // Check to see if we've reached the bottom
      if (next == nullptr)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and we can make a local node or node
          if (lower == source_shard)
          {
            EqKDTreeT<DIM, T>* local = left.load();
            if (local != nullptr)
              local->find_shard_equivalence_sets(
                  eq_sets, source_shard, dst_lower_shard, dst_upper_shard,
                  region);
          }
        }
        return;
      }
      legion_assert(next != nullptr);
      legion_assert(lower != upper);
      // We only need to traverse down the child that has our local shard
      ShardID src_diff = upper - lower;
      ShardID src_mid = lower + (src_diff / 2);
      ShardID dst_diff = dst_upper_shard - dst_lower_shard;
      ShardID dst_mid = dst_lower_shard + (dst_diff / 2);
      if (source_shard <= src_mid)
      {
        next = left.load();
        dst_upper_shard = dst_mid;
      }
      else if (dst_lower_shard != dst_upper_shard)
        dst_lower_shard = dst_mid + 1;
      next->find_shard_equivalence_sets(
          eq_sets, source_shard, dst_lower_shard, dst_upper_shard, region);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM, T>::invalidate_tree(
        const Rect<DIM, T>& rect, const FieldMask& mask,
        std::vector<RtEvent>& invalidated_events, bool move_to_previous,
        FieldMask* parent_all_previous)
    //--------------------------------------------------------------------------
    {
      legion_assert(this->bounds.contains(rect));
      // Just traverse everything that is open, no need to do any refinements
      // since the same invalidation is being done on every shard
      EqKDTreeT<DIM, T>* next = left.load();
      if (next != nullptr)
      {
        const Rect<DIM, T> overlap = next->bounds.intersection(rect);
        if (!overlap.empty())
          next->invalidate_tree(
              overlap, mask, invalidated_events, move_to_previous,
              parent_all_previous);
      }
      next = right.load();
      if (next != nullptr)
      {
        const Rect<DIM, T> overlap = next->bounds.intersection(rect);
        if (!overlap.empty())
          next->invalidate_tree(
              overlap, mask, invalidated_events, move_to_previous,
              parent_all_previous);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM, T>::invalidate_shard_tree_remote(
        const Rect<DIM, T>& rect, const FieldMask& mask,
        std::vector<RtEvent>& invalidated,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      // This invalidation is only being done on the local shard so we need
      // to perform any needed refinements so we can send the updates to
      // other shards to perform if necessary
      legion_assert(this->bounds.contains(rect));
      EqKDTreeT<DIM, T>* next = right.load();
      // Check to see if we've reached the bottom
      if (next == nullptr)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and we can make a local node or node
          if (lower == local_shard)
          {
            EqKDTreeT<DIM, T>* local = left.load();
            // Only need to perform the refinement if it already exists
            if (local != nullptr)
              local->invalidate_shard_tree_remote(
                  rect, mask, invalidated, remote_shard_rects, local_shard);
          }
          else
            remote_shard_rects[lower][rect] |= mask;
          return;
        }
        else  // Create the refinement
        {
          refine_node();
          next = right.load();
        }
      }
      legion_assert(next != nullptr);
      const Rect<DIM, T> right_overlap = next->bounds.intersection(rect);
      if (!right_overlap.empty())
        next->invalidate_shard_tree_remote(
            right_overlap, mask, invalidated, remote_shard_rects, local_shard);
      next = left.load();
      legion_assert(next != nullptr);
      const Rect<DIM, T> left_overlap = next->bounds.intersection(rect);
      if (!left_overlap.empty())
        next->invalidate_shard_tree_remote(
            left_overlap, mask, invalidated, remote_shard_rects, local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDSharded<DIM, T>::cancel_subscription(
        EqSetTracker* tracker, AddressSpaceID space, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      // should never be called on sharded nodes since they don't track
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM, T>::find_trace_local_sets(
        const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
        ShardID local_shard,
        std::map<EquivalenceSet*, unsigned>& local_sets) const
    //--------------------------------------------------------------------------
    {
      legion_assert(lower <= local_shard);
      legion_assert(local_shard <= upper);
      EqKDTreeT<DIM, T>* next = right.load();
      // Check to see if we've reached the bottom
      if (next == nullptr)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and if it is whether we have a node to traverse
          if (lower == local_shard)
          {
            EqKDTreeT<DIM, T>* local = left.load();
            if (local != nullptr)
              local->find_trace_local_sets(
                  rect, mask, req_index, local_shard, local_sets);
          }
        }
        // Else no need to create the refinement if it doesn't exist yet
        return;
      }
      legion_assert(next != nullptr);
      legion_assert(lower != upper);
      // We only need to traverse down the child that has our local shard
      ShardID diff = upper - lower;
      ShardID mid = lower + (diff / 2);
      if (local_shard <= mid)
        next = left.load();
      const Rect<DIM, T> next_overlap = next->bounds.intersection(rect);
      if (!next_overlap.empty())
        next->find_trace_local_sets(
            next_overlap, mask, req_index, local_shard, local_sets);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM, T>::find_shard_trace_local_sets(
        const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
        std::map<EquivalenceSet*, unsigned>& local_sets,
        local::map<ShardID, FieldMask>& remote_shards, ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      EqKDTreeT<DIM, T>* next = right.load();
      // Check to see if we've reached the bottom
      if (next == nullptr)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and we can make a local node or node
          if (lower == local_shard)
          {
            EqKDTreeT<DIM, T>* local = left.load();
            if (local == nullptr)
              local = refine_local();
            local->find_shard_trace_local_sets(
                rect, mask, req_index, local_sets, remote_shards, local_shard);
          }
          else
            remote_shards[lower] |= mask;
          // We're done
          return;
        }
        else  // Create the refinement
        {
          refine_node();
          next = right.load();
        }
      }
      legion_assert(next != nullptr);
      const Rect<DIM, T> right_overlap = next->bounds.intersection(rect);
      if (!right_overlap.empty())
        next->find_shard_trace_local_sets(
            right_overlap, mask, req_index, local_sets, remote_shards,
            local_shard);
      next = left.load();
      legion_assert(next != nullptr);
      const Rect<DIM, T> left_overlap = next->bounds.intersection(rect);
      if (!left_overlap.empty())
        next->find_shard_trace_local_sets(
            left_overlap, mask, req_index, local_sets, remote_shards,
            local_shard);
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set KD Sparse Sharded
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    /*static*/ inline bool EqKDSparseSharded<DIM, T>::sort_by_volume(
        const Rect<DIM, T>& r1, const Rect<DIM, T>& r2)
    //--------------------------------------------------------------------------
    {
      return (r1.volume() < r2.volume());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDSparseSharded<DIM, T>::EqKDSparseSharded(
        const Rect<DIM, T>& bound, ShardID low, ShardID high,
        std::vector<Rect<DIM, T> >& rects)
      : EqKDSharded<DIM, T>(bound, low, high)
    //--------------------------------------------------------------------------
    {
      legion_assert(rects.size() > 1);
      rectangles.swap(rects);
      total_volume = 0;
      for (typename std::vector<Rect<DIM, T> >::const_iterator it =
               rectangles.begin();
           it != rectangles.end(); it++)
        total_volume += it->volume();
      // If there's a chance that we might need to refine these then
      // stable sort them so that refine_node can rely on them already
      // being sorted. Note the stable sort! Must maintain deterministic
      // order across the shards
      if (this->MIN_SPLIT_SIZE <= total_volume)
        std::stable_sort(rectangles.begin(), rectangles.end(), sort_by_volume);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDSparseSharded<DIM, T>::~EqKDSparseSharded(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t EqKDSparseSharded<DIM, T>::get_total_volume(void) const
    //--------------------------------------------------------------------------
    {
      return total_volume;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDTreeT<DIM, T>* EqKDSparseSharded<DIM, T>::refine_local(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(this->right.load() == nullptr);
      EqKDSparse<DIM, T>* next =
          new EqKDSparse<DIM, T>(this->bounds, rectangles);
      EqKDTreeT<DIM, T>* expected = nullptr;
      if (this->left.compare_exchange_strong(expected, next))
      {
        next->add_reference();
        return next;
      }
      else
      {
        delete next;
        return expected;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparseSharded<DIM, T>::refine_node(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(this->lower < this->upper);
      // Note here that we don't want to evenly divide the rectangles across
      // the shards, but instead we want to evenly split the points across the
      // shards. We have two ways to do this, the first way is to call
      // compute_best_splitting_plane which will try to find a good splitting
      // plane that maintains the integrity of the spatial locality of all the
      // rectangles. Note that when we call compute_best_splitting_plane we
      // ask it to sort based on the number of points and not by the number of
      // rectangles which should keep the total points balanced across shards
      Rect<DIM, T> left_bounds, right_bounds;
      std::vector<Rect<DIM, T> > left_set, right_set;
      if (!KDTree::compute_best_splitting_plane<DIM, T, false>(
              this->bounds, rectangles, left_bounds, right_bounds, left_set,
              right_set))
      {
        // If we get here, then compute_best_splitting_plane failed to find
        // a splitting plane to split the rectangles nicely, so now we're
        // going to fall back to a dumb and greedy heuristic which will still
        // give us a good distribution of points around the shards which is
        // just to go from the largest rectangles to the smallest and assign
        // them to either the right or the left set depending on which one
        // is larger to get a roughly evenly distributed set of points
        // Note that this is determinisitc because the stable sort done in
        // the constructor of this class maintains the ordering of rectangles
        // with equivalent volumes across the shards.
        uint64_t left_volume = 0, right_volume = 0;
        // Reverse iterator to go from largest to smallest
        for (typename std::vector<Rect<DIM, T> >::const_reverse_iterator it =
                 rectangles.crbegin();
             it != rectangles.crend(); it++)
        {
          if (left_volume <= right_volume)
          {
            left_set.emplace_back(*it);
            left_volume += it->volume();
            left_bounds = left_bounds.union_bbox(*it);
          }
          else
          {
            right_set.emplace_back(*it);
            right_volume += it->volume();
            right_bounds = right_bounds.union_bbox(*it);
          }
        }
      }
      legion_assert(!left_set.empty());
      legion_assert(!right_set.empty());
      // Find the splitting of the shards
      ShardID diff = this->upper - this->lower;
      ShardID mid = this->lower + (diff / 2);
      EqKDSharded<DIM, T>* next = nullptr;
      if (left_set.size() > 1)
        next = new EqKDSparseSharded(left_bounds, this->lower, mid, left_set);
      else
        next = new EqKDSharded<DIM, T>(left_set.back(), this->lower, mid);
      EqKDTreeT<DIM, T>* expected = nullptr;
      if (this->left.compare_exchange_strong(expected, next))
        next->add_reference();
      else
        delete next;
      if (right_set.size() > 1)
        next = new EqKDSparseSharded(
            right_bounds, mid + 1, this->upper, right_set);
      else
        next = new EqKDSharded<DIM, T>(right_set.back(), mid + 1, this->upper);
      expected = nullptr;
      if (this->right.compare_exchange_strong(expected, next))
        next->add_reference();
      else
        delete next;
    }
#endif  // DEFINE_NT_TEMPLATES

  }  // namespace Internal
}  // namespace Legion
