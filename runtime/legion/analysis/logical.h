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

#ifndef __LEGION_LOGICAL_ANALYSIS_H__
#define __LEGION_LOGICAL_ANALYSIS_H__

#include "legion/kernel/garbage_collection.h"
#include "legion/utilities/dynamic_table.h"
#include "legion/utilities/fieldmask_map.h"
#include "legion/utilities/privileges.h"

namespace Legion {
  namespace Internal {

    /**
     * \struct LogicalUser
     * A class for representing logical users of a logical
     * region including the necessary information to
     * register mapping dependences on the user.
     */
    struct LogicalUser : public Collectable {
    public:
      LogicalUser(
          Operation* o, unsigned id, const RegionUsage& u,
          ProjectionSummary* proj = nullptr,
          unsigned internal_idx = std::numeric_limits<unsigned>::max());
      LogicalUser(const LogicalUser& rhs) = delete;
      ~LogicalUser(void);
    public:
      LogicalUser& operator=(const LogicalUser& rhs) = delete;
    public:
      struct Comparator {
      public:
        inline bool operator()(
            const LogicalUser* lhs, const LogicalUser* rhs) const
        {
          if (lhs->ctx_index < rhs->ctx_index)
            return true;
          if (lhs->ctx_index > rhs->ctx_index)
            return false;
          if (lhs->internal_idx < rhs->internal_idx)
            return true;
          if (lhs->internal_idx > rhs->internal_idx)
            return false;
          return (lhs->idx < rhs->idx);
        }
      };
    public:
      const RegionUsage usage;
      Operation* const op;
      const size_t ctx_index;
      const UniqueID uid;
      // Since internal operations have the same ctx_index as their
      // creator we need a way to distinguish them from the creator
      const unsigned internal_idx;
      const unsigned idx;
      const GenerationID gen;
      ProjectionSummary* const shard_proj;
      const bool pointwise_analyzable;
    };

    enum OpenState {
      NOT_OPEN = 0,
      OPEN_READ_ONLY = 1,
      OPEN_READ_WRITE = 2,  // unknown dirty information below
      OPEN_REDUCE = 3,      // make sure to check reduction value
    };

    /**
     * \struct FieldState
     * Track the field state more accurately
     * for logical traversals to figure out
     * which tasks can run in parallel.
     */
    struct FieldState {
    public:
      FieldState(void);
      FieldState(OpenState state, const FieldMask& m, RegionTreeNode* child);
      FieldState(
          const RegionUsage& usage, const FieldMask& m, RegionTreeNode* child);
      FieldState(const FieldState& rhs);
      FieldState(FieldState&& rhs) noexcept;
      FieldState& operator=(const FieldState& rhs);
      FieldState& operator=(FieldState&& rhs) noexcept;
      ~FieldState(void);
    public:
      inline const FieldMask& valid_fields(void) const
      {
        return open_children.get_valid_mask();
      }
    public:
      bool overlaps(const FieldState& rhs) const;
      void merge(FieldState& rhs, RegionTreeNode* node);
      bool filter(const FieldMask& mask);
      void add_child(RegionTreeNode* child, const FieldMask& mask);
      void remove_child(RegionTreeNode* child);
    public:
      // Template trickery to avoid issues with undefined types
      template<typename T>
      struct NodeComparator {
      public:
        inline bool operator()(const T* lhs, const T* rhs) const
        {
          return lhs->get_color() < rhs->get_color();
        }
      };
      typedef FieldMaskMap<
          RegionTreeNode, SHORT_LIFETIME, NodeComparator<RegionTreeNode> >
          OrderedFieldMaskChildren;
      OrderedFieldMaskChildren open_children;
      OpenState open_state;
      ReductionOpID redop;
    };

    /**
     * \class LogicalState
     * Track all the information about the current state
     * of a logical region from a given context. This
     * is effectively all the information at the analysis
     * wavefront for this particular logical region.
     */
    class LogicalState : public Heapify<LogicalState, CONTEXT_LIFETIME> {
    public:
      LogicalState(RegionTreeNode* owner, ContextID ctx);
      LogicalState(const LogicalState& state) = delete;
      ~LogicalState(void);
    public:
      LogicalState& operator=(const LogicalState& rhs) = delete;
    public:
      void check_init(void);
      void clear(void);
      void clear_deleted_state(ContextID ctx, const FieldMask& deleted_mask);
      ProjectionSummary* find_or_create_projection_summary(
          Operation* op, unsigned index, const RegionRequirement& req,
          LogicalAnalysis& analysis, const ProjectionInfo& proj_info);
      void remove_projection_summary(ProjectionSummary* summary);
      bool has_interfering_shards(
          LogicalAnalysis& analysis, ProjectionSummary* one,
          ProjectionSummary* two, bool& dominates);
      bool record_pointwise_dependence(
          LogicalAnalysis& analysis, const LogicalUser& prev,
          const LogicalUser& next, bool& dominates);
#ifdef LEGION_DEBUG
      void sanity_check(void) const;
#endif
    public:
      void initialize_no_refine_fields(const FieldMask& mask);
      void update_refinement_node(
          ContextID ctx, const RegionUsage& usage,
          const FieldMask& current_mask, FieldMask& updated_refinements);
      void update_refinement_child(
          ContextID ctx, RegionTreeNode* child, const RegionUsage& usage,
          const FieldMask& current_mask, FieldMask& child_mask,
          FieldMask& updated_refinements);
      void update_refinement_projection(
          ContextID ctx, ProjectionSummary* summary, const RegionUsage& usage,
          const FieldMask& proj_mask, FieldMask& updated_refinements);
      void deduplicate_refinements(local::vector<RefinementTracker*>& to_check);
      void invalidate_refinements(ContextID ctx, FieldMask invalidation_mask);
      void record_refinement_dependences(
          ContextID ctx, const LogicalUser& refinement_user,
          const FieldMask& refinement_mask, const ProjectionInfo& proj_info,
          RegionTreeNode* previous_child, LogicalRegion privilege_root,
          LogicalAnalysis& logical_analysis);
      void register_local_user(LogicalUser& user, const FieldMask& mask);
      void filter_current_epoch_users(const FieldMask& field_mask);
      void filter_previous_epoch_users(const FieldMask& field_mask);
      void filter_timeout_users(LogicalAnalysis& logical_analysis);
      void promote_next_child(RegionTreeNode* child, FieldMask mask);
    public:
      RegionTreeNode* const owner;
    public:
      shrt::list<FieldState> field_states;
      // Note that even though these are field mask sets keyed on pointers
      // we mark them as determinsitic so that shards always iterate over
      // these elements in the same order
      typedef FieldMaskMap<LogicalUser, SHORT_LIFETIME, LogicalUser::Comparator>
          OrderedFieldMaskUsers;
      OrderedFieldMaskUsers curr_epoch_users, prev_epoch_users;
    protected:
      // In some cases such as repeated read-only uses of a field we can
      // accumulate unbounded numbers of users in the curr/prev_epoch_users.
      // To combat blow-up in the size of those data structures we check to
      // see if the size of those data structures have grown to be at least
      // MIN_TIMEOUT_CHECK_SIZE. If they've grown that large then we attempt
      // to filter out those users. To avoid doing the filtering on every
      // return to this logical state, we only perform the filters every
      // so many returns to the logical state such that we can hide the
      // latency of the testing those timeouts across the parent task
      // context (can often be a non-trivial latency for control
      // replicated parent task contexts)
      static constexpr unsigned MIN_TIMEOUT_CHECK_SIZE = LEGION_MAX_FIELDS;
      unsigned total_timeout_check_iterations;
      unsigned remaining_timeout_check_iterations;
      TimeoutMatchExchange* timeout_exchange;
    public:
      // Refinement trackers manage the state of refinements for different
      // fields on this particular node of the region tree if we're along
      // a disjoint and complete partition path in this context.
      // We keep refinement trackers coalesced by fields as long as much
      // as possible but diverge them whenever unequal field sets try to
      // access them. They are grouped back together whenever we decide
      // to perform a refinement along this node.
      lng::FieldMaskMap<RefinementTracker> refinement_trackers;
    public:
      static constexpr size_t PROJECTION_CACHE_SIZE = 32;
      // Note that this list can grow bigger than PROJECTION_CACHE_SIZE
      // but we only keep references on the entries within the size of the
      // cache. This allows us to still hit on projections that are still
      // alive from other references, but also allows those entries to
      // be pruned out once they are no longer alive
      std::list<ProjectionSummary*> projection_summary_cache;
      std::unordered_map<
          ProjectionSummary*,
          std::unordered_map<
              ProjectionSummary*,
              std::pair<bool /*interferes*/, bool /*dominates*/> > >
          interfering_shards;
      // Track which pairs of projection summaries have point-wise mapping
      // dependences between them.
      std::unordered_map<
          ProjectionSummary*,
          std::unordered_map<
              ProjectionSummary*,
              std::pair<bool /*pointwise*/, bool /*dominates*/> > >
          pointwise_dependences;
    };

    typedef DynamicTableAllocator<LogicalState, 10, 8> LogicalStateAllocator;

    /**
     * \class LogicalAnalysis
     * The logical analysis helps capture the state of region tree traversals
     * for all region requirements in an operation. This includes capturing
     * the needed refinements to be performed as well as closes that have
     * already been performed for various projection region requirements.
     * At the end of the analysis, it issues all the refinements to be performed
     * by an operation and then performs them after all of the region
     * requirements for that operation are done being analyzed. That ensures
     * that we have at most one refinement change for all region requirements
     * in an operation that touch the same fields of the same region tree.
     */
    class LogicalAnalysis : public NoHeapify {
    public:
      static constexpr unsigned NO_OUTPUT_OFFSET =
          std::numeric_limits<unsigned>::max();
      LogicalAnalysis(Operation* op, unsigned output_offset = NO_OUTPUT_OFFSET);
      LogicalAnalysis(const LogicalAnalysis& rhs) = delete;
      ~LogicalAnalysis(void);
    public:
      LogicalAnalysis& operator=(const LogicalAnalysis& rhs) = delete;
    public:
      // Template trickery to avoid issues with undefined types
      template<typename T>
      struct OpComparator {
      public:
        inline bool operator()(const T* lhs, const T* rhs) const
        {
          return (lhs->get_unique_op_id() < rhs->get_unique_op_id());
        }
      };
      typedef OpComparator<RefinementOp> RefinementComparator;
      typedef FieldMaskMap<
          RefinementOp, TASK_LOCAL_LIFETIME, RefinementComparator>
          OrderedRefinements;
      void record_pending_refinement(
          LogicalRegion privilege, unsigned req_index,
          unsigned parent_req_index, RegionTreeNode* refinement_node,
          const FieldMask& refinement_mask, OrderedRefinements& refinements);
    public:
      // Record a prior operation that we need to depend on with a
      // close operation to group together dependences
      void record_close_dependence(
          LogicalRegion privilege, unsigned req_index,
          RegionTreeNode* path_node, const LogicalUser* user,
          const FieldMask& mask);
    protected:
      void issue_internal_operation(
          RegionTreeNode* node, InternalOp* close_op,
          const FieldMask& internal_mask, const unsigned internal_index) const;
    public:
      Operation* const op;
      InnerContext* const context;
      // Offset for output region requirements which we will use
      // to ignore any refinement requests for them
      const unsigned output_region_offset;
    protected:
      // Keep these ordered by the order in which we make them so that
      // all shards will iterate over them in the same order for
      // control replication cases, we do this by sorting them based
      // on their unique IDs which are monotonically increasing so we
      // know that they will be in order across shards too
      OrderedRefinements pending_refinements;
      std::map<RegionTreeNode*, MergeCloseOp*> pending_closes;
    };

    /**
     * \class RegionTreePath
     * Keep track of the path and states associated with a
     * given region requirement of an operation.
     */
    class RegionTreePath {
    public:
      RegionTreePath(void);
    public:
      void initialize(unsigned min_depth, unsigned max_depth);
      void register_child(unsigned depth, const LegionColor color);
      void clear();
    public:
#ifdef LEGION_DEBUG
      bool has_child(unsigned depth) const;
      LegionColor get_child(unsigned depth) const;
#else
      inline bool has_child(unsigned depth) const
      {
        return path[depth] != INVALID_COLOR;
      }
      inline LegionColor get_child(unsigned depth) const { return path[depth]; }
#endif
      inline unsigned get_path_length(void) const
      {
        return ((max_depth - min_depth) + 1);
      }
      inline unsigned get_min_depth(void) const { return min_depth; }
      inline unsigned get_max_depth(void) const { return max_depth; }
    protected:
      std::vector<LegionColor> path;
      unsigned min_depth;
      unsigned max_depth;
    };

    /**
     * \class PathTraverser
     * An abstract class which provides the needed
     * functionality for walking a path and visiting
     * all the kinds of nodes along the path.
     */
    class PathTraverser {
    public:
      PathTraverser(RegionTreePath& path);
      PathTraverser(const PathTraverser& rhs) = delete;
      virtual ~PathTraverser(void);
    public:
      PathTraverser& operator=(const PathTraverser& rhs) = delete;
    public:
      // Return true if the traversal was successful
      // or false if one of the nodes exit stopped early
      bool traverse(RegionTreeNode* start);
    public:
      virtual bool visit_region(RegionNode* node) = 0;
      virtual bool visit_partition(PartitionNode* node) = 0;
    protected:
      RegionTreePath& path;
    protected:
      // Fields are only valid during traversal
      unsigned depth;
      bool has_child;
      LegionColor next_child;
    };

    /**
     * \class NodeTraverser
     * An abstract class which provides the needed
     * functionality for visiting a node in the tree
     * and all of its sub-nodes.
     */
    class NodeTraverser {
    public:
      NodeTraverser(bool force = false) : force_instantiation(force) { }
    public:
      virtual bool break_early(void) const { return false; }
      virtual bool visit_only_valid(void) const = 0;
      virtual bool visit_region(RegionNode* node) = 0;
      virtual bool visit_partition(PartitionNode* node) = 0;
    public:
      const bool force_instantiation;
    };

    /**
     * \class CurrentInitializer
     * A class for initializing current states
     */
    class CurrentInitializer : public NodeTraverser {
    public:
      CurrentInitializer(ContextID ctx);
      CurrentInitializer(const CurrentInitializer& rhs) = delete;
      ~CurrentInitializer(void);
    public:
      CurrentInitializer& operator=(const CurrentInitializer& rhs) = delete;
    public:
      virtual bool visit_only_valid(void) const override;
      virtual bool visit_region(RegionNode* node) override;
      virtual bool visit_partition(PartitionNode* node) override;
    protected:
      const ContextID ctx;
    };

    /**
     * \class CurrentInvalidator
     * A class for invalidating current states
     */
    class CurrentInvalidator : public NodeTraverser {
    public:
      CurrentInvalidator(ContextID ctx);
      CurrentInvalidator(const CurrentInvalidator& rhs) = delete;
      ~CurrentInvalidator(void);
    public:
      CurrentInvalidator& operator=(const CurrentInvalidator& rhs) = delete;
    public:
      virtual bool visit_only_valid(void) const override;
      virtual bool visit_region(RegionNode* node) override;
      virtual bool visit_partition(PartitionNode* node) override;
    protected:
      const ContextID ctx;
    };

    /**
     * \class DeletionInvalidator
     * A class for invalidating current states for deletions
     */
    class DeletionInvalidator : public NodeTraverser {
    public:
      DeletionInvalidator(ContextID ctx, const FieldMask& deletion_mask);
      DeletionInvalidator(const DeletionInvalidator& rhs) = delete;
      ~DeletionInvalidator(void);
    public:
      DeletionInvalidator& operator=(const DeletionInvalidator& rhs) = delete;
    public:
      virtual bool visit_only_valid(void) const override;
      virtual bool visit_region(RegionNode* node) override;
      virtual bool visit_partition(PartitionNode* node) override;
    protected:
      const ContextID ctx;
      const FieldMask& deletion_mask;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_LOGICAL_ANALYSIS_H__
