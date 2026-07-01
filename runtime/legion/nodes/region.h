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

#ifndef __LEGION_LOGICAL_REGION_H__
#define __LEGION_LOGICAL_REGION_H__

#include "legion/analysis/logical.h"
#include "legion/analysis/refinement.h"
#include "legion/analysis/versioning.h"
#include "legion/nodes/index.h"
#include "legion/nodes/field.h"
#include "legion/utilities/buffers.h"

namespace Legion {
  namespace Internal {

    /**
     * \class RegionTreeNode
     * A generic region tree node from which all
     * other kinds of region tree nodes inherit.  Notice
     * that all important analyses are defined on
     * this kind of node making them general across
     * all kinds of node types.
     */
    class RegionTreeNode : public DistributedCollectable {
    public:
      RegionTreeNode(
          FieldSpaceNode* column, RtEvent initialized, RtEvent tree_init,
          Provenance* provenance = nullptr, DistributedID did = 0,
          CollectiveMapping* mapping = nullptr);
      virtual ~RegionTreeNode(void);
    public:
      static AddressSpaceID get_owner_space(RegionTreeID tid);
    public:
      inline LogicalState& get_logical_state(ContextID ctx)
      {
        return *(logical_states.lookup_entry(ctx, this, ctx));
      }
      inline LogicalState* get_logical_state_ptr(ContextID ctx)
      {
        return logical_states.lookup_entry(ctx, this, ctx);
      }
      inline VersionManager& get_current_version_manager(ContextID ctx)
      {
        return *(current_versions.lookup_entry(ctx, this, ctx));
      }
      inline VersionManager* get_current_version_manager_ptr(ContextID ctx)
      {
        return current_versions.lookup_entry(ctx, this, ctx);
      }
      typedef FieldState::OrderedFieldMaskChildren OrderedFieldMaskChildren;
    public:
      void attach_semantic_information(
          SemanticTag tag, AddressSpaceID source, const void* buffer,
          size_t size, bool is_mutable, bool local_only);
      bool retrieve_semantic_information(
          SemanticTag tag, const void*& result, size_t& size, bool can_fail,
          bool wait_until);
      virtual AddressSpaceID find_semantic_owner(void) const = 0;
      virtual void send_semantic_request(
          AddressSpaceID target, SemanticTag tag, bool can_fail,
          bool wait_until, RtUserEvent ready) = 0;
      virtual void send_semantic_info(
          AddressSpaceID target, SemanticTag tag, const void* buffer,
          size_t size, bool is_mutable, RtUserEvent ready) = 0;
    public:
      // Logical traversal operations
      void register_logical_user(
          LogicalRegion privilege_root, LogicalUser& user,
          const RegionTreePath& path, const LogicalTraceInfo& trace_info,
          const ProjectionInfo& projection_info, const FieldMask& user_mask,
          FieldMask& unopened_field_mask, const FieldMask& refinement_mask,
          LogicalAnalysis& logical_analysis,
          FieldMaskMap<
              RefinementOp, TASK_LOCAL_LIFETIME,
              LogicalAnalysis::RefinementComparator>& refinements);
      void add_open_field_state(
          LogicalState& state, const LogicalUser& user,
          const FieldMask& open_mask, RegionTreeNode* next_child);
      void siphon_interfering_children(
          LogicalState& state, LogicalAnalysis& analysis,
          const FieldMask& closing_mask, const LogicalUser& user,
          LogicalRegion privilege_root, RegionTreeNode* next_child,
          FieldMask& open_below);
      void perform_close_operations(
          const LogicalUser& user, const FieldMask& close_mask,
          OrderedFieldMaskChildren& children, LogicalRegion privilege_root,
          RegionTreeNode* path_node, LogicalAnalysis& analysis,
          FieldMask& open_below, RegionTreeNode* next_child = nullptr,
          FieldMask* next_child_fields = nullptr,
          const bool filter_next_child = false);
      void close_logical_node(
          const LogicalUser& user, const FieldMask& closing_mask,
          LogicalRegion privilege_root, RegionTreeNode* path_node,
          LogicalAnalysis& analysis, FieldMask& still_open);
      ProjectionSummary* compute_projection_summary(
          Operation* op, unsigned index, const RegionRequirement& req,
          LogicalAnalysis& analysis, const ProjectionInfo& info);
      void record_refinement_dependences(
          ContextID ctx, const LogicalUser& refinement_user,
          const FieldMask& refinement_mask, const ProjectionInfo& proj_info,
          RegionTreeNode* previous_child, LogicalRegion privilege_root,
          LogicalAnalysis& logical_analysis);
      void merge_new_field_state(LogicalState& state, FieldState& new_state);
      void filter_prev_epoch_users(LogicalState& state, const FieldMask& mask);
      void filter_curr_epoch_users(LogicalState& state, const FieldMask& mask);
      void report_uninitialized_usage(
          Operation* op, unsigned index, const FieldMask& uninitialized,
          RtUserEvent reported);
      void invalidate_logical_refinement(
          ContextID ctx, const FieldMask& invalidate_mask);
    public:
      void initialize_current_state(ContextID ctx);
      void invalidate_current_state(ContextID ctx);
      void invalidate_deleted_state(
          ContextID ctx, const FieldMask& deleted_mask);
    public:
      virtual unsigned get_depth(void) const = 0;
      virtual LegionColor get_color(void) const = 0;
      virtual IndexTreeNode* get_row_source(void) const = 0;
      virtual RegionTreeID get_tree_id(void) const = 0;
      virtual RegionTreeNode* get_parent(void) const = 0;
      virtual RegionTreeNode* get_tree_child(const LegionColor c) = 0;
      virtual bool is_region(void) const = 0;
#ifdef LEGION_DEBUG
      virtual RegionNode* as_region_node(void) const = 0;
      virtual PartitionNode* as_partition_node(void) const = 0;
#else
      inline RegionNode* as_region_node(void) const;
      inline PartitionNode* as_partition_node(void) const;
#endif
      virtual RefinementTracker* create_refinement_tracker(void) = 0;
      virtual bool visit_node(PathTraverser* traverser) = 0;
      virtual bool visit_node(NodeTraverser* traverser) = 0;
    public:
      virtual bool are_children_disjoint(
          const LegionColor c1, const LegionColor c2) = 0;
      virtual bool are_all_children_disjoint(void) = 0;
      virtual bool is_complete(void) = 0;
      virtual bool intersects_with(
          RegionTreeNode* other, bool compute = true) = 0;
      virtual bool track_refinements(void) const = 0;
    public:
      virtual size_t get_num_children(void) const = 0;
      virtual void send_node(Serializer& rez, AddressSpaceID target) = 0;
    public:
      // Logical helper operations
      typedef FieldMaskMap<LogicalUser, SHORT_LIFETIME, LogicalUser::Comparator>
          OrderedFieldMaskUsers;
      template<bool TRACK_DOM>
      FieldMask perform_dependence_checks(
          LogicalRegion privilege_root, const LogicalUser& user,
          OrderedFieldMaskUsers& users, const FieldMask& check_mask,
          const FieldMask& open_below, const bool arrived,
          const ProjectionInfo& proj_info, LogicalState& state,
          LogicalAnalysis& logical_analysis);
      static void perform_closing_checks(
          LogicalAnalysis& analysis, OrderedFieldMaskUsers& users,
          const LogicalUser& user, const FieldMask& check_mask,
          LogicalRegion root_privilege, RegionTreeNode* path_node,
          FieldMask& still_open);
    public:
      inline FieldSpaceNode* get_column_source(void) const
      {
        return column_source;
      }
    public:
      FieldSpaceNode* const column_source;
      Provenance* const provenance;
      RtEvent initialized;
      const RtEvent tree_initialized;  // top level tree initialization
    public:
      bool registered;
    protected:
      DynamicTable<LogicalStateAllocator> logical_states;
      DynamicTable<VersionManagerAllocator> current_versions;
    protected:
      mutable LocalLock node_lock;
    protected:
      lng::map<SemanticTag, SemanticInfo> semantic_info;
    };

    /**
     * \class RegionNode
     * Represent a region in a region tree
     */
    class RegionNode : public RegionTreeNode,
                       public Heapify<RegionNode, LONG_LIFETIME> {
    public:
      struct SemanticRequestArgs : public LgTaskArgs<SemanticRequestArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_REGION_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        SemanticRequestArgs(void) = default;
        SemanticRequestArgs(
            RegionNode* proxy, SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticRequestArgs>(false, false), proxy_this(proxy),
            tag(t), source(src)
        { }
        void execute(void) const;
      public:
        RegionNode* proxy_this;
        SemanticTag tag;
        AddressSpaceID source;
      };
    public:
      RegionNode(
          LogicalRegion r, PartitionNode* par, IndexSpaceNode* row_src,
          FieldSpaceNode* col_src, DistributedID did, RtEvent initialized,
          RtEvent tree_initialized, CollectiveMapping* mapping,
          Provenance* provenance);
      RegionNode(const RegionNode& rhs) = delete;
      virtual ~RegionNode(void);
    public:
      RegionNode& operator=(const RegionNode& rhs) = delete;
    public:
      virtual void notify_local(void);
    public:
      void record_registered(void);
    public:
      bool has_color(const LegionColor p);
      PartitionNode* get_child(const LegionColor p);
      void add_child(PartitionNode* child);
      void remove_child(const LegionColor p);
      void add_tracker(PartitionTracker* tracker);
      void initialize_no_refine_fields(ContextID ctx, const FieldMask& m);
    public:
      virtual unsigned get_depth(void) const;
      virtual LegionColor get_color(void) const;
      virtual IndexTreeNode* get_row_source(void) const;
      virtual RegionTreeID get_tree_id(void) const;
      virtual RegionTreeNode* get_parent(void) const;
      virtual RegionTreeNode* get_tree_child(const LegionColor c);
      virtual RefinementTracker* create_refinement_tracker(void)
      {
        return new RegionRefinementTracker(this);
      }
    public:
      virtual bool are_children_disjoint(
          const LegionColor c1, const LegionColor c2);
      virtual bool are_all_children_disjoint(void);
      virtual bool is_region(void) const;
#ifdef LEGION_DEBUG
      virtual RegionNode* as_region_node(void) const;
      virtual PartitionNode* as_partition_node(void) const;
#endif
      virtual bool visit_node(PathTraverser* traverser);
      virtual bool visit_node(NodeTraverser* traverser);
      virtual bool is_complete(void);
      virtual bool intersects_with(RegionTreeNode* other, bool compute = true);
      virtual bool track_refinements(void) const;
      virtual size_t get_num_children(void) const;
      virtual void send_node(Serializer& rez, AddressSpaceID target);
      static void handle_node_creation(
          Deserializer& derez, AddressSpaceID source);
    public:
      virtual AddressSpaceID find_semantic_owner(void) const;
      virtual void send_semantic_request(
          AddressSpaceID target, SemanticTag tag, bool can_fail,
          bool wait_until, RtUserEvent ready);
      virtual void send_semantic_info(
          AddressSpaceID target, SemanticTag tag, const void* buffer,
          size_t size, bool is_mutable, RtUserEvent ready);
      void process_semantic_request(
          SemanticTag tag, AddressSpaceID source, bool can_fail,
          bool wait_until, RtUserEvent ready);
    public:
      // Support for refinements and versioning
      void perform_versioning_analysis(
          ContextID ctx, InnerContext* parent_ctx, VersionInfo* version_info,
          const FieldMask& version_mask, Operation* op, unsigned index,
          unsigned parent_req_index, std::set<RtEvent>& ready_events,
          RtEvent* output_region_ready = nullptr,
          bool collective_rendezvous = false);
    public:
      void find_open_complete_partitions(
          ContextID ctx, const FieldMask& mask,
          std::vector<LogicalPartition>& partitions);
    public:
      const LogicalRegion handle;
      PartitionNode* const parent;
      IndexSpaceNode* const row_source;
    protected:
      std::map<LegionColor, PartitionNode*> color_map;
      std::list<PartitionTracker*> partition_trackers;
    };

    /**
     * \class PartitionNode
     * Represent an instance of a partition in a region tree.
     */
    class PartitionNode : public RegionTreeNode,
                          public Heapify<PartitionNode, LONG_LIFETIME> {
    public:
      struct SemanticRequestArgs : public LgTaskArgs<SemanticRequestArgs> {
      public:
        static constexpr LgTaskID TASK_ID =
            LG_PARTITION_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        SemanticRequestArgs(void) = default;
        SemanticRequestArgs(
            PartitionNode* proxy, SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticRequestArgs>(false, false), proxy_this(proxy),
            tag(t), source(src)
        { }
        void execute(void) const;
      public:
        PartitionNode* proxy_this;
        SemanticTag tag;
        AddressSpaceID source;
      };
    public:
      PartitionNode(
          LogicalPartition p, RegionNode* par, IndexPartNode* row_src,
          FieldSpaceNode* col_src, RtEvent init, RtEvent tree);
      PartitionNode(const PartitionNode& rhs) = delete;
      virtual ~PartitionNode(void);
    public:
      PartitionNode& operator=(const PartitionNode& rhs) = delete;
    public:
      virtual void notify_local(void);
    public:
      void record_registered(void);
    public:
      bool has_color(const LegionColor c);
      RegionNode* get_child(const LegionColor c);
      void add_child(RegionNode* child);
    public:
      virtual unsigned get_depth(void) const;
      virtual LegionColor get_color(void) const;
      virtual IndexTreeNode* get_row_source(void) const;
      virtual RegionTreeID get_tree_id(void) const;
      virtual RegionTreeNode* get_parent(void) const;
      virtual RegionTreeNode* get_tree_child(const LegionColor c);
      virtual RefinementTracker* create_refinement_tracker(void)
      {
        return new PartitionRefinementTracker(this);
      }
    public:
      virtual bool are_children_disjoint(
          const LegionColor c1, const LegionColor c2);
      virtual bool are_all_children_disjoint(void);
      virtual bool is_region(void) const;
#ifdef LEGION_DEBUG
      virtual RegionNode* as_region_node(void) const;
      virtual PartitionNode* as_partition_node(void) const;
#endif
      virtual bool visit_node(PathTraverser* traverser);
      virtual bool visit_node(NodeTraverser* traverser);
      virtual bool is_complete(void);
      virtual bool intersects_with(RegionTreeNode* other, bool compute = true);
      virtual bool track_refinements(void) const;
      virtual size_t get_num_children(void) const;
      virtual void send_node(Serializer& rez, AddressSpaceID target);
    public:
      virtual AddressSpaceID find_semantic_owner(void) const;
      virtual void send_semantic_request(
          AddressSpaceID target, SemanticTag tag, bool can_fail,
          bool wait_until, RtUserEvent ready);
      virtual void send_semantic_info(
          AddressSpaceID target, SemanticTag tag, const void* buffer,
          size_t size, bool is_mutable, RtUserEvent ready);
      void process_semantic_request(
          SemanticTag tag, AddressSpaceID source, bool can_fail,
          bool wait_until, RtUserEvent ready);
    public:
      const LogicalPartition handle;
      RegionNode* const parent;
      IndexPartNode* const row_source;
    protected:
      std::map<LegionColor, RegionNode*> color_map;
    };

#ifndef LEGION_DEBUG
    //--------------------------------------------------------------------------
    inline RegionNode* RegionTreeNode::as_region_node(void) const
    //--------------------------------------------------------------------------
    {
      return static_cast<RegionNode*>(const_cast<RegionTreeNode*>(this));
    }

    //--------------------------------------------------------------------------
    inline PartitionNode* RegionTreeNode::as_partition_node(void) const
    //--------------------------------------------------------------------------
    {
      return static_cast<PartitionNode*>(const_cast<RegionTreeNode*>(this));
    }
#endif

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_LOGICAL_REGION_H__
