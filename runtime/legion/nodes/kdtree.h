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

#ifndef __LEGION_KDTREE_H__
#define __LEGION_KDTREE_H__

namespace Legion {
  namespace Internal {

    /**
     * \interface KDTree
     * A virtual interface to a KD tree
     */
    class KDTree {
    public:
      virtual ~KDTree(void) { }
    public:
      template<int DIM, typename T>
      inline KDNode<DIM, T>* as_kdnode(void);
      // This method tries to compute a splitting plane either by evenly
      // dividing the rectangles or by evenly dividing the points in the
      // sets of rectangles depending on the BY_RECTS parameter
      template<int DIM, typename T, bool BY_RECTS = true>
      static inline bool compute_best_splitting_plane(
          const Rect<DIM, T>& bounds, const std::vector<Rect<DIM, T> >& rects,
          Rect<DIM, T>& best_left_bounds, Rect<DIM, T>& best_right_bounds,
          std::vector<Rect<DIM, T> >& best_left_set,
          std::vector<Rect<DIM, T> >& best_right_set);
    };

    /**
     * \class EqKDTree
     * This class defines the interface for looking up equivalence
     * sets for any given parent logical region
     */
    class EqKDTree : public Collectable {
    public:
      virtual ~EqKDTree(void) { }
    public:
      virtual void compute_shard_equivalence_sets(
          const Domain& rect, const FieldMask& mask,
          const std::vector<EqSetTracker*>& trackers,
          const std::vector<AddressSpaceID>& tracker_spaces,
          std::vector<unsigned>& new_tracker_references,
          op::FieldMaskMap<EquivalenceSet>& eq_sets,
          std::vector<RtEvent>& pending_sets,
          op::FieldMaskMap<EqKDTree>& subscriptions,
          op::FieldMaskMap<EqKDTree>& to_create,
          op::map<EqKDTree*, Domain>& creation_rects,
          op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs,
          ShardID local_shard) = 0;
      virtual unsigned record_shard_output_equivalence_set(
          EquivalenceSet* set, const Domain& rect, const FieldMask& mask,
          EqSetTracker* tracker, AddressSpaceID tracker_space,
          local::FieldMaskMap<EqKDTree>& new_subscriptions,
          ShardID local_shard) = 0;
      virtual void record_equivalence_set(
          EquivalenceSet* set, const FieldMask& mask, RtEvent ready,
          const CollectiveMapping& creator_spaces,
          const std::vector<EqSetTracker*>& creators) = 0;
      virtual void find_local_equivalence_sets(
          local::FieldMaskMap<EquivalenceSet>& eq_sets,
          ShardID local_shard) const = 0;
      virtual void find_shard_equivalence_sets(
          local::map<
              ShardID,
              local::map<RegionNode*, local::FieldMaskMap<EquivalenceSet> > >&
              eq_sets,
          ShardID source_shard, ShardID dst_lower_shard,
          ShardID dst_upper_shard, RegionNode* region) const = 0;
      virtual void invalidate_shard_tree(
          const Domain& domain, const FieldMask& mask,
          std::vector<RtEvent>& invalidated) = 0;
      // Return true if we should remove the reference on the origin tracker
      virtual unsigned cancel_subscription(
          EqSetTracker* tracker, AddressSpaceID space,
          const FieldMask& mask) = 0;
      // Just use this method of indirecting into template land
      virtual IndexSpaceExpression* create_from_rectangles(
          const std::vector<Domain>& rectangles) const = 0;
    public:
      template<int DIM, typename T>
      inline EqKDTreeT<DIM, T>* as_eq_kd_tree(void);
    };

    /**
     * \class EqKDTreeT
     */
    template<int DIM, typename T>
    class EqKDTreeT : public EqKDTree {
    public:
      EqKDTreeT(const Rect<DIM, T>& rect);
      virtual ~EqKDTreeT(void) { }
    public:
      virtual void initialize_set(
          EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
          ShardID local_shard, bool current) = 0;
      virtual void compute_shard_equivalence_sets(
          const Domain& rect, const FieldMask& mask,
          const std::vector<EqSetTracker*>& trackers,
          const std::vector<AddressSpaceID>& tracker_spaces,
          std::vector<unsigned>& new_tracker_references,
          op::FieldMaskMap<EquivalenceSet>& eq_sets,
          std::vector<RtEvent>& pending_sets,
          op::FieldMaskMap<EqKDTree>& subscriptions,
          op::FieldMaskMap<EqKDTree>& to_create,
          op::map<EqKDTree*, Domain>& creation_rects,
          op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs,
          ShardID local_shard);
      virtual void compute_equivalence_sets(
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
          ShardID local_shard = 0) = 0;
      virtual unsigned record_shard_output_equivalence_set(
          EquivalenceSet* set, const Domain& rect, const FieldMask& mask,
          EqSetTracker* tracker, AddressSpaceID tracker_space,
          local::FieldMaskMap<EqKDTree>& new_subscriptions,
          ShardID local_shard);
      virtual void record_equivalence_set(
          EquivalenceSet* set, const FieldMask& mask, RtEvent ready,
          const CollectiveMapping& creator_spaces,
          const std::vector<EqSetTracker*>& creators) = 0;
      virtual unsigned record_output_equivalence_set(
          EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
          EqSetTracker* tracker, AddressSpaceID tracker_space,
          local::FieldMaskMap<EqKDTree>& subscriptions,
          op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
          ShardID local_shard = 0) = 0;
      virtual void find_shard_equivalence_sets(
          local::map<
              ShardID,
              local::map<RegionNode*, local::FieldMaskMap<EquivalenceSet> > >&
              eq_sets,
          ShardID source_shard, ShardID dst_lower_shard,
          ShardID dst_upper_shard, RegionNode* region) const = 0;
      virtual void invalidate_tree(
          const Rect<DIM, T>& rect, const FieldMask& mask,
          std::vector<RtEvent>& invalidated_events, bool move_to_previous,
          FieldMask* parent_all_previous = nullptr) = 0;
      virtual void invalidate_shard_tree(
          const Domain& domain, const FieldMask& mask,
          std::vector<RtEvent>& invalidated);
      virtual void invalidate_shard_tree_remote(
          const Rect<DIM, T>& rect, const FieldMask& mask,
          std::vector<RtEvent>& invalidated,
          op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
          ShardID local_shard = 0) = 0;
      virtual unsigned cancel_subscription(
          EqSetTracker* tracker, AddressSpaceID space,
          const FieldMask& mask) = 0;
      // Just use this method of indirecting into template land
      virtual IndexSpaceExpression* create_from_rectangles(
          const std::vector<Domain>& rectangles) const;
      virtual void find_trace_local_sets(
          const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
          ShardID local_shard,
          std::map<EquivalenceSet*, unsigned>& current_sets) const = 0;
      virtual void find_shard_trace_local_sets(
          const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
          std::map<EquivalenceSet*, unsigned>& current_sets,
          local::map<ShardID, FieldMask>& remote_shards,
          ShardID local_shard) = 0;
    public:
      const Rect<DIM, T> bounds;
    };

    /**
     * This class provides support for efficient spatial lookup of
     * equivalence sets for a given parent region in a context.
     */
    template<int DIM, typename T>
    class EqKDNode : public EqKDTreeT<DIM, T>,
                     public Heapify<EqKDNode<DIM, T>, CONTEXT_LIFETIME> {
    public:
      EqKDNode(const Rect<DIM, T>& bounds);
      EqKDNode(const EqKDNode& rhs) = delete;
      virtual ~EqKDNode(void);
    public:
      EqKDNode& operator=(const EqKDNode& rhs) = delete;
    public:
      virtual void initialize_set(
          EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
          ShardID local_shard, bool current);
      virtual void compute_equivalence_sets(
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
          ShardID local_shard = 0);
      virtual void record_equivalence_set(
          EquivalenceSet* set, const FieldMask& mask, RtEvent ready,
          const CollectiveMapping& creator_spaces,
          const std::vector<EqSetTracker*>& creators);
      virtual unsigned record_output_equivalence_set(
          EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
          EqSetTracker* tracker, AddressSpaceID tracker_space,
          local::FieldMaskMap<EqKDTree>& subscriptions,
          op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
          ShardID local_shard = 0);
      virtual void find_local_equivalence_sets(
          local::FieldMaskMap<EquivalenceSet>& eq_sets,
          ShardID local_shard) const;
      virtual void find_shard_equivalence_sets(
          local::map<
              ShardID,
              local::map<RegionNode*, local::FieldMaskMap<EquivalenceSet> > >&
              eq_sets,
          ShardID source_shard, ShardID dst_lower_shard,
          ShardID dst_upper_shard, RegionNode* region) const;
      virtual void invalidate_tree(
          const Rect<DIM, T>& rect, const FieldMask& mask,
          std::vector<RtEvent>& invalidated_events, bool move_to_previous,
          FieldMask* parent_all_previous = nullptr);
      virtual void invalidate_shard_tree_remote(
          const Rect<DIM, T>& rect, const FieldMask& mask,
          std::vector<RtEvent>& invalidated,
          op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
          ShardID local_shard = 0);
      virtual unsigned cancel_subscription(
          EqSetTracker* tracker, AddressSpaceID space, const FieldMask& mask);
      virtual void find_trace_local_sets(
          const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
          ShardID local_shard,
          std::map<EquivalenceSet*, unsigned>& current_sets) const;
      virtual void find_shard_trace_local_sets(
          const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
          std::map<EquivalenceSet*, unsigned>& current_sets,
          local::map<ShardID, FieldMask>& remote_shards, ShardID local_shard);
    public:
      void find_all_previous_sets(
          FieldMask mask,
          op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs);
      void invalidate_all_previous_sets(const FieldMask& mask);
      void find_shard_equivalence_sets(
          const Rect<DIM, T>& rect,
          local::map<
              ShardID,
              local::map<RegionNode*, local::FieldMaskMap<EquivalenceSet> > >&
              eq_sets,
          ShardID dst_lower_shard, ShardID dst_upper_shard,
          RegionNode* region) const;
      void find_rect_equivalence_sets(
          const Rect<DIM, T>& rect,
          local::FieldMaskMap<EquivalenceSet>& eq_sets) const;
    protected:
      void refine_node(
          const Rect<DIM, T>& rect, const FieldMask& mask,
          bool refine_current = false);
      unsigned record_subscription(
          EqSetTracker* tracker, AddressSpaceID tracker_space,
          const FieldMask& mask);
      void clone_sets(
          EqKDNode<DIM, T>* left, EqKDNode<DIM, T>* right, FieldMask clone,
          lng::FieldMaskMap<EquivalenceSet>*& sets, bool current);
      void record_set(EquivalenceSet* set, const FieldMask& mask, bool current);
      void find_to_get_previous(
          FieldMask& all_prev_below,
          local::FieldMaskMap<EqKDNode<DIM, T> >& to_get_previous) const;
      void invalidate_previous_sets(
          const FieldMask& mask,
          local::FieldMaskMap<EqKDNode<DIM, T> >& to_invalidate_previous);
      void record_child_all_previous(EqKDNode<DIM, T>* child, FieldMask mask);
    protected:
      mutable LocalLock node_lock;
      // Left and right sub-trees for different fields
      lng::FieldMaskMap<EqKDNode<DIM, T> >*lefts, *rights;
      // Current equivalence sets are the ones that have current data
      // Previous equivalence sets are ones that have been invalidated but
      // we still need to update whatever the new equivalence sets are
      lng::FieldMaskMap<EquivalenceSet>*current_sets, *previous_sets;
      // Events for indicating when the current sets are ready because they
      // might still be in the process of being initialized
      shrt::map<RtEvent, FieldMask>* current_set_preconditions;
      // Equvialence sets that are being made at this level but are not ready
      shrt::map<RtUserEvent, FieldMask>* pending_set_creations;
      // Postconditions for the creation of each of the pending sets
      std::map<RtUserEvent, std::vector<RtEvent> >* pending_postconditions;
      // Trackers on different nodes that are currently tracking the state
      // of this node for different fields
      lng::map<AddressSpaceID, lng::FieldMaskMap<EqSetTracker> >* subscriptions;
      // If we know that only one of the children (either right or left)
      // is all previous below we record that here in case we ultimately
      // end up seeing the other child become all-previous below at which
      // point we can set the all_previous_below mask appropriately
      lng::FieldMaskMap<EqKDNode<DIM, T> >* child_previous_below;
      // Record fields for which all the sub-nodes (in both left and right)
      // only have previous sets and no current sets because we invalidated
      // everything at this node and below
      FieldMask all_previous_below;
    };

    /**
     * \class EqKDSparse
     * In the case of index spaces with sparsity maps, this class helps
     * deal with the tracking of splitting planes for the rectangles until
     * we get down to a single rectangle and can move to EqKDNodes
     */
    template<int DIM, typename T>
    class EqKDSparse : public EqKDTreeT<DIM, T> {
    public:
      EqKDSparse(
          const Rect<DIM, T>& bound, const std::vector<Rect<DIM, T> >& rects);
      EqKDSparse(const EqKDSparse& rhs) = delete;
      virtual ~EqKDSparse(void);
    public:
      EqKDSparse& operator=(const EqKDSparse& rhs) = delete;
    public:
      virtual void initialize_set(
          EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
          ShardID local_shard, bool current);
      virtual void compute_equivalence_sets(
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
          ShardID local_shard = 0);
      virtual void record_equivalence_set(
          EquivalenceSet* set, const FieldMask& mask, RtEvent ready,
          const CollectiveMapping& creator_spaces,
          const std::vector<EqSetTracker*>& creators);
      virtual unsigned record_output_equivalence_set(
          EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
          EqSetTracker* tracker, AddressSpaceID tracker_space,
          local::FieldMaskMap<EqKDTree>& subscriptions,
          op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
          ShardID local_shard = 0);
      virtual void find_local_equivalence_sets(
          local::FieldMaskMap<EquivalenceSet>& eq_sets,
          ShardID local_shard) const;
      virtual void find_shard_equivalence_sets(
          local::map<
              ShardID,
              local::map<RegionNode*, local::FieldMaskMap<EquivalenceSet> > >&
              eq_sets,
          ShardID source_shard, ShardID dst_lower_shard,
          ShardID dst_upper_shard, RegionNode* region) const;
      virtual void invalidate_tree(
          const Rect<DIM, T>& rect, const FieldMask& mask,
          std::vector<RtEvent>& invalidated_events, bool move_to_previous,
          FieldMask* parent_all_previous = nullptr);
      virtual void invalidate_shard_tree_remote(
          const Rect<DIM, T>& rect, const FieldMask& mask,
          std::vector<RtEvent>& invalidated,
          op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
          ShardID local_shard = 0);
      virtual unsigned cancel_subscription(
          EqSetTracker* tracker, AddressSpaceID space, const FieldMask& mask);
      virtual void find_trace_local_sets(
          const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
          ShardID local_shard,
          std::map<EquivalenceSet*, unsigned>& current_sets) const;
      virtual void find_shard_trace_local_sets(
          const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
          std::map<EquivalenceSet*, unsigned>& current_sets,
          local::map<ShardID, FieldMask>& remote_shards, ShardID local_shard);
    protected:
      std::vector<EqKDTreeT<DIM, T>*> children;
    };

    /**
     * \class EqKDSharded
     * For control replicated contexts, this class provides a way of sharding
     * a dense rectangle down into subspaces handled by different shards.
     * We split shards by high order bits first down to low order bits in order
     * to maintain spatial locality between shards
     */
    template<int DIM, typename T>
    class EqKDSharded : public EqKDTreeT<DIM, T> {
    public:
      EqKDSharded(const Rect<DIM, T>& bound, ShardID lower, ShardID upper);
      EqKDSharded(const EqKDSharded& rhs) = delete;
      virtual ~EqKDSharded(void);
    public:
      EqKDSharded& operator=(const EqKDSharded& rhs) = delete;
    public:
      virtual void initialize_set(
          EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
          ShardID local_shard, bool current);
      virtual void compute_equivalence_sets(
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
          ShardID local_shard = 0);
      virtual void record_equivalence_set(
          EquivalenceSet* set, const FieldMask& mask, RtEvent ready,
          const CollectiveMapping& creator_spaces,
          const std::vector<EqSetTracker*>& creators);
      virtual unsigned record_output_equivalence_set(
          EquivalenceSet* set, const Rect<DIM, T>& rect, const FieldMask& mask,
          EqSetTracker* tracker, AddressSpaceID tracker_space,
          local::FieldMaskMap<EqKDTree>& subscriptions,
          op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
          ShardID local_shard = 0);
      virtual void find_local_equivalence_sets(
          local::FieldMaskMap<EquivalenceSet>& eq_sets,
          ShardID local_shard) const;
      virtual void find_shard_equivalence_sets(
          local::map<
              ShardID,
              local::map<RegionNode*, local::FieldMaskMap<EquivalenceSet> > >&
              eq_sets,
          ShardID source_shard, ShardID dst_lower_shard,
          ShardID dst_upper_shard, RegionNode* region) const;
      virtual void invalidate_tree(
          const Rect<DIM, T>& rect, const FieldMask& mask,
          std::vector<RtEvent>& invalidated_events, bool move_to_previous,
          FieldMask* parent_all_previous = nullptr);
      virtual void invalidate_shard_tree_remote(
          const Rect<DIM, T>& rect, const FieldMask& mask,
          std::vector<RtEvent>& invalidated,
          op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
          ShardID local_shard = 0);
      virtual unsigned cancel_subscription(
          EqSetTracker* tracker, AddressSpaceID space, const FieldMask& mask);
      virtual void find_trace_local_sets(
          const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
          ShardID local_shard,
          std::map<EquivalenceSet*, unsigned>& current_sets) const;
      virtual void find_shard_trace_local_sets(
          const Rect<DIM, T>& rect, const FieldMask& mask, unsigned req_index,
          std::map<EquivalenceSet*, unsigned>& current_sets,
          local::map<ShardID, FieldMask>& remote_shards, ShardID local_shard);
    protected:
      // Make these methods virtual so they can be overloaded by the sparse
      // version of this class that inherits from this class as well
      virtual size_t get_total_volume(void) const;
      virtual void refine_node(void);
      virtual EqKDTreeT<DIM, T>* refine_local(void);
    public:
      // Lower bound shard (inclusive)
      const ShardID lower;
      // Upper bound shard (inclusive)
      const ShardID upper;
      // To avoid over-decomposing we specify a minimum split size, as soon
      // as the total number of points represented by this  node are less
      // than this value then we stop splitting and use the smallest shard
      // in the set of shards to handle the results
      static constexpr size_t MIN_SPLIT_SIZE = 4096;
    protected:
      // These are atomic since they are lazily instantiated but once
      // they are instantiated then they don't change so we don't need
      // to have a lock in this node of the tree
      std::atomic<EqKDTreeT<DIM, T>*> left, right;
    };

    /**
     * \class EqKDSparseSharded
     * This class handles the case of splitting sparse index spaces down
     * to subsets of rectangles that can be handled by individual shards.
     */
    template<int DIM, typename T>
    class EqKDSparseSharded : public EqKDSharded<DIM, T> {
    public:
      EqKDSparseSharded(
          const Rect<DIM, T>& bound, ShardID lower, ShardID upper,
          std::vector<Rect<DIM, T> >& rects);
      EqKDSparseSharded(const EqKDSparseSharded& rhs) = delete;
      virtual ~EqKDSparseSharded(void);
    public:
      EqKDSparseSharded& operator=(const EqKDSparseSharded& rhs) = delete;
    protected:
      virtual size_t get_total_volume(void) const;
      virtual void refine_node(void);
      virtual EqKDTreeT<DIM, T>* refine_local(void);
      static inline bool sort_by_volume(
          const Rect<DIM, T>& r1, const Rect<DIM, T>& r2);
    protected:
      std::vector<Rect<DIM, T> > rectangles;
      size_t total_volume;
    };

    /**
     * \class KDNode
     * A KDNode is used for performing fast interference tests for
     * expressions against rectangles from child subregions in a partition.
     */
    template<int DIM, typename T, typename RT>
    class KDNode {
    public:
      KDNode(
          const Rect<DIM, T>& bounds,
          std::vector<std::pair<Rect<DIM, T>, RT> >& subrects);
      KDNode(const KDNode& rhs) = delete;
      ~KDNode(void);
    public:
      KDNode& operator=(const KDNode& rhs) = delete;
    public:
      void find_interfering(
          const Rect<DIM, T>& test, std::set<RT>& interfering) const;
      void record_inorder_traversal(std::vector<RT>& order) const;
      RT find(const Point<DIM, T>& point) const;
    public:
      const Rect<DIM, T> bounds;
    protected:
      KDNode<DIM, T, RT>* left;
      KDNode<DIM, T, RT>* right;
      std::vector<std::pair<Rect<DIM, T>, RT> > rects;
    };

    // Specialization for void case
    template<int DIM, typename T>
    class KDNode<DIM, T, void> : public KDTree {
    public:
      KDNode(const Rect<DIM, T>& bounds, std::vector<Rect<DIM, T> >& subrects);
      KDNode(const KDNode& rhs) = delete;
      virtual ~KDNode(void);
    public:
      KDNode& operator=(const KDNode& rhs) = delete;
    public:
      size_t count_rectangles(void) const;
      size_t count_intersecting_points(const Rect<DIM, T>& rect) const;
    public:
      const Rect<DIM, T> bounds;
    protected:
      KDNode<DIM, T, void>* left;
      KDNode<DIM, T, void>* right;
      std::vector<Rect<DIM, T> > rects;
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/nodes/kdtree.inl"

#endif  // __LEGION_KDTREE_H__
