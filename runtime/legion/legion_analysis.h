/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_ANALYSIS_H__
#define __LEGION_ANALYSIS_H__

#include "legion/legion_types.h"
#include "legion/legion_utilities.h"
#include "legion/legion_allocation.h"
#include "legion/garbage_collection.h"

namespace Legion {
  namespace Internal {

    /**
     * \struct GenericUser
     * A base struct for tracking the user of a logical region
     */
    struct GenericUser {
    public:
      GenericUser(void) { }
      GenericUser(const RegionUsage &u, const FieldMask &m)
        : usage(u), field_mask(m) { }
    public:
      RegionUsage usage;
      FieldMask field_mask;
    };

    /**
     * \struct LogicalUser
     * A class for representing logical users of a logical 
     * region including the necessary information to
     * register mapping dependences on the user.
     */
    struct LogicalUser : public GenericUser {
    public:
      LogicalUser(void);
      LogicalUser(Operation *o, unsigned id, 
                  const RegionUsage &u, const FieldMask &m);
      LogicalUser(Operation *o, GenerationID gen, unsigned id,
                  const RegionUsage &u, const FieldMask &m);
    public:
      Operation *op;
      unsigned idx;
      GenerationID gen;
      // This field addresses a problem regarding when
      // to prune tasks out of logical region tree data
      // structures.  If no later task ever performs a
      // dependence test against this user, we might
      // never prune it from the list.  This timeout
      // prevents that from happening by forcing a
      // test to be performed whenever the timeout
      // reaches zero.
      int timeout;
#ifdef LEGION_SPY
      UniqueID uid;
#endif
    public:
      static const int TIMEOUT = DEFAULT_LOGICAL_USER_TIMEOUT;
    };

    /**
     * \class VersionInfo
     * A class for tracking version information about region usage
     */
    class VersionInfo {
    public:
      struct DeferredVersionFinalizeArgs : 
        public LgTaskArgs<DeferredVersionFinalizeArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_VERSION_FINALIZE_TASK_ID;
      public:
        DeferredVersionFinalizeArgs(VersionInfo *info, UniqueID uid)
          : LgTaskArgs<DeferredVersionFinalizeArgs>(uid),
            proxy_this(info) { } 
      public:
        VersionInfo *const proxy_this;
      };
    public:
      VersionInfo(void);
      VersionInfo(const VersionInfo &rhs);
      virtual ~VersionInfo(void);
    public:
      VersionInfo& operator=(const VersionInfo &rhs);
    public:
      inline bool has_version_info(void) const { return (op != NULL); }
      inline const std::set<EquivalenceSet*>& get_equivalence_sets(void) const
        { return equivalence_sets; }
      inline std::set<RtEvent>& get_applied_events(void) 
        { return applied_events; }
      inline void record_applied_event(RtEvent applied)
        { applied_events.insert(applied); }
    public:
      void initialize_mapping(Operation *op);
      void record_equivalence_sets(VersionManager *owner,
                                   const std::set<EquivalenceSet*> &sets);
      void acquire_equivalence_sets(const RegionRequirement &req,
                                    const FieldMask &acquire_mask,
                                    std::set<Reservation> &needed_reservations);
      // There are very strict requirements for calling this with
      // runtime_relaxed=true. You have to be concurrently mapping actual
      // physical instances (to guarantee reduction applications in parallel)
      // As far as I know this is currently only safe from control replication
      // contexts with things like ReplMapOp and ReplAttachOp
      void make_ready(const RegionRequirement &req, const FieldMask &mask,
                      std::set<RtEvent> &ready_events,
                      const bool runtime_relaxed = false);
      void finalize_mapping(std::set<RtEvent> &map_applied_events, 
                            bool block = false);
    protected:
      void perform_finalize(void);
    public:
      static void handle_defer_finalize(const void *args);
    protected:
      Operation *op;
      VersionManager *owner;
      std::set<RtEvent> applied_events;
      std::set<EquivalenceSet*> equivalence_sets;
    };

    /**
     * \struct LogicalTraceInfo
     * Information about tracing needed for logical
     * dependence analysis.
     */
    struct LogicalTraceInfo {
    public:
      LogicalTraceInfo(bool already_tr,
                       LegionTrace *tr,
                       unsigned idx,
                       const RegionRequirement &r);
    public:
      bool already_traced;
      LegionTrace *trace;
      unsigned req_idx;
      const RegionRequirement &req;
    };

    /**
     * \struct PhysicalTraceInfo
     */
    struct PhysicalTraceInfo {
    public:
      explicit PhysicalTraceInfo(Operation *op, bool initialize = true);
      PhysicalTraceInfo(Operation *op, Memoizable *memo);
    public:
      void record_merge_events(ApEvent &result, ApEvent e1, ApEvent e2) const;
      void record_merge_events(ApEvent &result, ApEvent e1, 
                               ApEvent e2, ApEvent e3) const;
      void record_merge_events(ApEvent &result, 
                               const std::set<ApEvent> &events) const;
      void record_op_sync_event(ApEvent &result) const;
    public:
      void record_issue_copy(ApEvent &result,
                             IndexSpaceExpression *expr,
                             const std::vector<CopySrcDstField>& src_fields,
                             const std::vector<CopySrcDstField>& dst_fields,
#ifdef LEGION_SPY
                             FieldSpace handle,
                             RegionTreeID src_tree_id,
                             RegionTreeID dst_tree_id,
#endif
                             ApEvent precondition,
                             ReductionOpID redop,
                             bool reduction_fold) const;
      void record_issue_fill(ApEvent &result,
                             IndexSpaceExpression *expr,
                             const std::vector<CopySrcDstField> &fields,
                             const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                             UniqueID fill_uid,
                             FieldSpace handle,
                             RegionTreeID tree_id,
#endif
                             ApEvent precondition) const;
      void record_empty_copy(DeferredView *view,
                             const FieldMask &copy_mask,
                             MaterializedView *dst) const;
    public:
      Operation *const op;
      PhysicalTemplate *const tpl;
      const bool recording;
    };

    /**
     * \class ProjectionInfo
     * Projection information for index space requirements
     */
    class ProjectionInfo {
    public:
      ProjectionInfo(void)
        : projection(NULL), projection_type(SINGULAR),
          projection_space(NULL) { }
      ProjectionInfo(Runtime *runtime, const RegionRequirement &req,
                     IndexSpace launch_space, ShardingFunction *func = NULL,
                     IndexSpace shard_space = IndexSpace::NO_SPACE);
    public:
      inline bool is_projecting(void) const { return (projection != NULL); }
    public:
      ProjectionFunction *projection;
      ProjectionType projection_type;
      IndexSpaceNode *projection_space;
      ShardingFunction *sharding_function;
      IndexSpaceNode *sharding_space;
    };

    /**
     * \struct PhysicalUser
     * A class for representing physical users of a logical
     * region including necessary information to 
     * register execution dependences on the user.
     */
    struct PhysicalUser : public Collectable, 
                          public LegionHeapify<PhysicalUser> {
    public:
      static const AllocationType alloc_type = PHYSICAL_USER_ALLOC;
    public:
      PhysicalUser(IndexSpaceExpression *expr);
      PhysicalUser(const RegionUsage &u, IndexSpaceExpression *expr,
                   UniqueID op_id, unsigned index, bool copy);
      PhysicalUser(const PhysicalUser &rhs);
      ~PhysicalUser(void);
    public:
      PhysicalUser& operator=(const PhysicalUser &rhs);
    public:
      void pack_user(Serializer &rez, AddressSpaceID target);
      static PhysicalUser* unpack_user(Deserializer &derez, bool add_reference,
                               RegionTreeForest *forest, AddressSpaceID source);
    public:
      RegionUsage usage;
      IndexSpaceExpression *const expr;
      const size_t expr_volume;
      UniqueID op_id;
      unsigned index; // region requirement index
      bool copy_user; // is this from a copy or an operation
    };  

    /**
     * \struct ChildState
     * Tracks the which fields have open children
     * and then which children are open for each
     * field. We also keep track of the children
     * that are in the process of being closed
     * to avoid races on two different operations
     * trying to close the same child.
     */
    struct ChildState {
    public:
      ChildState(void) { }
      ChildState(const FieldMask &m)
        : valid_fields(m) { }
      ChildState(const ChildState &rhs) 
        : valid_fields(rhs.valid_fields),
          open_children(rhs.open_children) { }
    public:
      ChildState& operator=(const ChildState &rhs)
      {
        valid_fields = rhs.valid_fields;
        open_children = rhs.open_children;
        return *this;
      }
    public:
      FieldMask valid_fields;
      LegionMap<LegionColor,FieldMask>::aligned open_children;
    };

    /**
     * \struct ProjectionSummary
     * A small helper class that tracks the triple that 
     * uniquely defines a set of region requirements
     * for a projection operation
     */
    struct ProjectionSummary {
    public:
      ProjectionSummary(void);
      ProjectionSummary(IndexSpaceNode *is, 
                        ProjectionFunction *p, 
                        ShardingFunction *s,
                        IndexSpaceNode *sd);
      ProjectionSummary(const ProjectionInfo &info);
    public:
      bool operator<(const ProjectionSummary &rhs) const;
      bool operator==(const ProjectionSummary &rhs) const;
      bool operator!=(const ProjectionSummary &rhs) const;
    public:
      void pack_summary(Serializer &rez) const;
      static ProjectionSummary unpack_summary(Deserializer &derez,
                                              RegionTreeForest *context);
    public:
      IndexSpaceNode *domain;
      ProjectionFunction *projection;
      ShardingFunction *sharding;
      IndexSpaceNode *sharding_domain;
    };

    /**
     * \struct FieldState
     * Track the field state more accurately
     * for logical traversals to figure out 
     * which tasks can run in parallel.
     */
    struct FieldState : public ChildState {
    public:
      FieldState(void);
      FieldState(const GenericUser &u, const FieldMask &m, 
                 LegionColor child);
      FieldState(const RegionUsage &u, const FieldMask &m,
                 ProjectionFunction *proj, IndexSpaceNode *proj_space, 
                 ShardingFunction *sharding_function, 
                 IndexSpaceNode *sharding_space,
                 RegionTreeNode *node, bool dirty_reduction = false);
    public:
      inline bool is_projection_state(void) const 
        { return (open_state >= OPEN_READ_ONLY_PROJ); } 
    public:
      bool overlaps(const FieldState &rhs) const;
      bool projections_match(const FieldState &rhs) const;
      void merge(const FieldState &rhs, RegionTreeNode *node);
    public:
      bool can_elide_close_operation(const ProjectionInfo &info,
                     RegionTreeNode *node, bool reduction) const;
      void record_projection_summary(const ProjectionInfo &info,
                                     RegionTreeNode *node);
    protected:
      bool expensive_elide_test(const ProjectionInfo &info,
                RegionTreeNode *node, bool reduction) const;
    public:
      void print_state(TreeStateLogger *logger, 
                       const FieldMask &capture_mask,
                       RegionNode *node) const;
      void print_state(TreeStateLogger *logger, 
                       const FieldMask &capture_mask,
                       PartitionNode *node) const;
    public:
      OpenState open_state;
      ReductionOpID redop;
      std::set<ProjectionSummary> projections;
      unsigned rebuild_timeout;
      bool disjoint_shallow;
    };

    /**
     * \class ProjectionTree
     * This is a tree that stores the summary of a region
     * tree that is accessed by an index launch and which
     * node owns the leaves for in the case of control replication
     */
    class ProjectionTree {
    public:
      ProjectionTree(IndexTreeNode *source,
                     ShardID owner_shard = 0);
      ProjectionTree(const ProjectionTree &rhs);
      ~ProjectionTree(void);
    public:
      ProjectionTree& operator=(const ProjectionTree &rhs);
    public:
      void add_child(ProjectionTree *child);
      bool dominates(const ProjectionTree *other) const;
      bool disjoint(const ProjectionTree *other) const;
      bool all_same_shard(ShardID other_shard) const;
    public:
      IndexTreeNode *const node;
      const ShardID owner_shard;
      std::map<IndexTreeNode*,ProjectionTree*> children;
    };

    /**
     * \class LogicalState
     * Track all the information about the current state
     * of a logical region from a given context. This
     * is effectively all the information at the analysis
     * wavefront for this particular logical region.
     */
    class LogicalState : public LegionHeapify<LogicalState> {
    public:
      static const AllocationType alloc_type = CURRENT_STATE_ALLOC;
    public:
      LogicalState(RegionTreeNode *owner, ContextID ctx);
      LogicalState(const LogicalState &state);
      ~LogicalState(void);
    public:
      LogicalState& operator=(const LogicalState &rhs);
    public:
      void check_init(void);
      void clear_logical_users(void);
      void reset(void);
      void clear_deleted_state(const FieldMask &deleted_mask);
    public:
      RegionTreeNode *const owner;
    public:
      LegionList<FieldState,
                 LOGICAL_FIELD_STATE_ALLOC>::track_aligned field_states;
      LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned 
                                                            curr_epoch_users;
      LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned 
                                                            prev_epoch_users;
    public:
      // Keep track of which fields we've done a reduction to here
      FieldMask reduction_fields;
      LegionMap<ReductionOpID,FieldMask>::aligned outstanding_reductions;
    };

    typedef DynamicTableAllocator<LogicalState,10,8> LogicalStateAllocator;

    /**
     * \class LogicalCloser
     * This structure helps keep track of the state
     * necessary for performing a close operation
     * on the logical region tree.
     */
    class LogicalCloser {
    public:
      LogicalCloser(ContextID ctx, const LogicalUser &u, 
                    RegionTreeNode *root, bool validates);
      LogicalCloser(const LogicalCloser &rhs);
      ~LogicalCloser(void);
    public:
      LogicalCloser& operator=(const LogicalCloser &rhs);
    public:
      inline bool has_close_operations(void) const { return !!close_mask; }
      // Record normal closes like this
      void record_close_operation(const FieldMask &mask);
      void record_closed_user(const LogicalUser &user, const FieldMask &mask);
#ifndef LEGION_SPY
      void pop_closed_user(void);
#endif
      void initialize_close_operations(LogicalState &state, 
                                       Operation *creator,
                                       const LogicalTraceInfo &trace_info);
      void perform_dependence_analysis(const LogicalUser &current,
                                       const FieldMask &open_below,
             LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &cusers,
             LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned &pusers);
      void update_state(LogicalState &state);
      void register_close_operations(
              LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &users);
    protected:
      void register_dependences(CloseOp *close_op, 
                                const LogicalUser &close_user,
                                const LogicalUser &current, 
                                const FieldMask &open_below,
             LegionList<LogicalUser,CLOSE_LOGICAL_ALLOC>::track_aligned &husers,
             LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned &ausers,
             LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &cusers,
             LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned &pusers);
    public:
      ContextID ctx;
      const LogicalUser &user;
      RegionTreeNode *const root_node;
      const bool validates;
      LegionList<LogicalUser,CLOSE_LOGICAL_ALLOC>::track_aligned closed_users;
    protected:
      FieldMask close_mask;
    protected:
      // At most we will ever generate one close operation at a node
      MergeCloseOp *close_op;
    protected:
      // Cache the generation IDs so we can kick off ops before adding users
      GenerationID merge_close_gen;
    }; 

    class KDTree {
    public:
      virtual ~KDTree(void) { }
      virtual bool refine(std::vector<EquivalenceSet*> &subsets) = 0;
    };

    /**
     * \class KDNode
     * A KDNode is used for constructing a KD tree in order to divide up the 
     * sub equivalence sets when there are too many to fan out from just 
     * one parent equivalence set.
     */
    template<int DIM>
    class KDNode : public KDTree {
    public:
      KDNode(IndexSpaceExpression *expr, Runtime *runtime, 
             int refinement_dim, int last_changed_dim = -1); 
      KDNode(const Rect<DIM> &rect, Runtime *runtime,
             int refinement_dim, int last_changed_dim = -1);
      KDNode(const KDNode<DIM> &rhs);
      virtual ~KDNode(void);
    public:
      KDNode<DIM>& operator=(const KDNode<DIM> &rhs);
    public:
      virtual bool refine(std::vector<EquivalenceSet*> &subsets);
    public:
      static Rect<DIM> get_bounds(IndexSpaceExpression *expr);
    public:
      Runtime *const runtime;
      const Rect<DIM> bounds;
      const int refinement_dim;
      // For detecting non convext cases where we kind find a 
      // splitting plane in any dimension
      const int last_changed_dim;
    };

    /**
     * \class CopyFillAggregator
     * The copy aggregator class is one that records the copies
     * that needs to be done for different equivalence classes and
     * then merges them together into the biggest possible copies
     * that can be issued together.
     */
    class CopyFillAggregator : public WrapperReferenceMutator {
    public:
      typedef LegionMap<InstanceView*,
               FieldMaskSet<IndexSpaceExpression> >::aligned InstanceFieldExprs;
      typedef LegionMap<ApEvent,
               FieldMaskSet<IndexSpaceExpression> >::aligned EventFieldExprs;
      class CopyUpdate;
      class FillUpdate;
      class Update {
      public:
        Update(IndexSpaceExpression *exp, const FieldMask &mask,
               CopyAcrossHelper *helper, PredEvent guard)
          : expr(exp), src_mask(mask), across_helper(helper), 
            predicate_guard(guard) { }
        virtual ~Update(void) { }
      public:
        virtual void record_source_expressions(
                        InstanceFieldExprs &src_exprs) const = 0;
        virtual void compute_source_preconditions(RegionTreeForest *forest,
               const std::map<InstanceView*,EventFieldExprs> &src_pre,
               LegionMap<ApEvent,FieldMask>::aligned &preconditions) const = 0;
        virtual void sort_updates(std::map<InstanceView*,
                                           std::vector<CopyUpdate*> > &copies,
                                  std::vector<FillUpdate*> &fills) = 0;
      public:
        IndexSpaceExpression *const expr;
        const FieldMask src_mask;
        CopyAcrossHelper *const across_helper;
        const PredEvent predicate_guard;
      };
      class CopyUpdate : public Update, public LegionHeapify<CopyUpdate> {
      public:
        CopyUpdate(InstanceView *src, const FieldMask &mask,
                   IndexSpaceExpression *expr,
                   ReductionOpID red = 0,
                   CopyAcrossHelper *helper = NULL,
                   PredEvent guard = PredEvent::NO_PRED_EVENT)
          : Update(expr, mask, helper, guard), source(src), redop(red) { }
        virtual ~CopyUpdate(void) { }
      private:
        CopyUpdate(const CopyUpdate &rhs)
          : Update(rhs.expr, rhs.src_mask, 
                   rhs.across_helper, rhs.predicate_guard), 
            source(rhs.source), redop(rhs.redop) { assert(false); }
        CopyUpdate& operator=(const CopyUpdate &rhs)
          { assert(false); return *this; }
      public:
        virtual void record_source_expressions(
                        InstanceFieldExprs &src_exprs) const;
        virtual void compute_source_preconditions(RegionTreeForest *forest,
                   const std::map<InstanceView*,EventFieldExprs> &src_pre,
                   LegionMap<ApEvent,FieldMask>::aligned &preconditions) const;
        virtual void sort_updates(std::map<InstanceView*,
                                           std::vector<CopyUpdate*> > &copies,
                                  std::vector<FillUpdate*> &fills);
      public:
        InstanceView *const source;
        const ReductionOpID redop;
      };
      class FillUpdate : public Update, public LegionHeapify<FillUpdate> {
      public:
        FillUpdate(FillView *src, const FieldMask &mask,
                   IndexSpaceExpression *expr,
                   CopyAcrossHelper *helper = NULL,
                   PredEvent guard = PredEvent::NO_PRED_EVENT)
          : Update(expr, mask, helper, guard), source(src) { }
        virtual ~FillUpdate(void) { }
      private:
        FillUpdate(const FillUpdate &rhs)
          : Update(rhs.expr, rhs.src_mask, rhs.across_helper,
                   rhs.predicate_guard),
            source(rhs.source) { assert(false); }
        FillUpdate& operator=(const FillUpdate &rhs)
          { assert(false); return *this; }
      public:
        virtual void record_source_expressions(
                        InstanceFieldExprs &src_exprs) const;
        virtual void compute_source_preconditions(RegionTreeForest *forest,
                   const std::map<InstanceView*,EventFieldExprs> &src_pre,
                   LegionMap<ApEvent,FieldMask>::aligned &preconditions) const;
        virtual void sort_updates(std::map<InstanceView*,
                                           std::vector<CopyUpdate*> > &copies,
                                  std::vector<FillUpdate*> &fills);
      public:
        FillView *const source;
      };
      typedef LegionMap<ApEvent,
               FieldMaskSet<Update> >::aligned EventFieldUpdates;
    public:
      CopyFillAggregator(RegionTreeForest *forest, Operation *op, unsigned idx,
                         std::set<RtEvent> &applied_events, bool track_events);
      CopyFillAggregator(RegionTreeForest *forest, Operation *op, 
                         unsigned src_idx, unsigned dst_idx,
                         std::set<RtEvent> &applied_events, bool track_events);
      CopyFillAggregator(const CopyFillAggregator &rhs);
      ~CopyFillAggregator(void);
    public:
      CopyFillAggregator& operator=(const CopyFillAggregator &rhs);
    public:
      void record_updates(InstanceView *dst_view, 
                          const FieldMaskSet<LogicalView> &src_views,
                          const FieldMask &src_mask,
                          IndexSpaceExpression *expr,
                          ReductionOpID redop = 0,
                          CopyAcrossHelper *across_helper = NULL);
      // Neither fills nor reductions should have a redop across as they
      // should have been applied an instance directly for across copies
      void record_fill(InstanceView *dst_view,
                       FillView *src_view,
                       const FieldMask &fill_mask,
                       IndexSpaceExpression *expr,
                       CopyAcrossHelper *across_helper = NULL);
      void record_reductions(InstanceView *dst_view,
                             const std::vector<ReductionView*> &src_views,
                             const unsigned src_fidx,
                             const unsigned dst_fidx,
                             IndexSpaceExpression *expr,
                             CopyAcrossHelper *across_helper = NULL);
      // Record preconditions coming back from analysis on views
      void record_preconditions(InstanceView *view, bool reading,
                                EventFieldExprs &preconditions);
      void record_precondition(InstanceView *view, bool reading,
                               ApEvent event, const FieldMask &mask,
                               IndexSpaceExpression *expr);
      inline bool has_updates(void) const
        { return !sources.empty() || !reductions.empty(); }
      void issue_updates(const PhysicalTraceInfo &trace_info, 
                         ApEvent precondition,
                         // Next two flags are used for across-copies
                         // to indicate when we already know preconditions
                         const bool has_src_preconditions = false,
                         const bool has_dst_preconditions = false);
      ApEvent summarize(const PhysicalTraceInfo &trace_info) const;
    protected:
      void record_view(LogicalView *new_view);
      void perform_updates(const LegionMap<InstanceView*,
                            FieldMaskSet<Update> >::aligned &updates,
                           const PhysicalTraceInfo &trace_info,
                           ApEvent precondition,
                           const bool has_src_preconditions,
                           const bool has_dst_preconditions);
      void issue_fills(InstanceView *target,
                       const std::vector<FillUpdate*> &fills,
                       ApEvent precondition, const FieldMask &fill_mask,
                       const PhysicalTraceInfo &trace_info,
                       const bool has_dst_preconditions);
      void issue_copies(InstanceView *target, 
                        const std::map<InstanceView*,
                                       std::vector<CopyUpdate*> > &copies,
                        ApEvent precondition, const FieldMask &copy_mask,
                        const PhysicalTraceInfo &trace_info,
                        const bool has_dst_preconditions);
    public:
      RegionTreeForest *const forest;
      Operation *const op;
      const unsigned src_index;
      const unsigned dst_index;
      const bool track_events;
    protected:
      LegionMap<InstanceView*,FieldMaskSet<Update> >::aligned sources; 
      std::vector</*vector over reduction epochs*/
        LegionMap<InstanceView*,FieldMaskSet<Update> >::aligned> reductions;
      // Figure out the reduction operator is for each epoch of a
      // given destination instance and field
      std::map<std::pair<InstanceView*,unsigned/*dst fidx*/>,
               std::vector<ReductionOpID> > reduction_epochs;
      std::set<LogicalView*> all_views; // used for reference counting
    protected:
      mutable LocalLock pre_lock; 
      std::map<InstanceView*,EventFieldExprs> dst_pre, src_pre;
    protected:
      // Runtime mapping effects that we create
      std::set<RtEvent> &effects;
      // Events for the completion of our copies if we are supposed
      // to be tracking them
      std::set<ApEvent> events;
    protected:
      struct SourceQuery {
      public:
        SourceQuery(void) { }
        SourceQuery(const std::set<InstanceView*> srcs,
                    const FieldMask src_mask,
                    InstanceView *res)
          : sources(srcs), query_mask(src_mask), result(res) { }
      public:
        std::set<InstanceView*> sources;
        FieldMask query_mask;
        InstanceView *result;
      };
      // Cached calls to the mapper for selecting sources
      std::map<InstanceView*,LegionVector<SourceQuery>::aligned> mapper_queries;
    };

    /**
     * \class EquivalenceSet
     * The equivalence set class tracks the physical state of a
     * set of points in a logical region for all the fields. There
     * is an owner node for the equivlance set that uses a ESI
     * protocol in order to manage local and remote copies of 
     * the equivalence set for each of the different fields.
     * It's also possible for the equivalence set to be refined
     * into sub equivalence sets which then subsum it's responsibility.
     */
    class EquivalenceSet : public DistributedCollectable,
                           public LegionHeapify<EquivalenceSet> {
    public:
      static const AllocationType alloc_type = EQUIVALENCE_SET_ALLOC;
    public:
      struct MappingGuard {
      public:
        MappingGuard(void)
          : guard_index(0), count(0) { }
        MappingGuard(const FieldMask &m)
          : guard_mask(m), guard_index(0), count(1) { }
      public:
        FieldMask guard_mask;
        FieldMask mutated_mask;
        RtUserEvent aliased_waiters;
        unsigned long long guard_index;
        unsigned count;
      };
    public:
      struct RefinementTaskArgs : public LgTaskArgs<RefinementTaskArgs> {
      public:
        static const LgTaskID TASK_ID = LG_REFINEMENT_TASK_ID;
      public:
        RefinementTaskArgs(EquivalenceSet *t)
          : LgTaskArgs<RefinementTaskArgs>(implicit_provenance), target(t) { }
      public:
        EquivalenceSet *const target;
      };
      struct RemoteRefTaskArgs : public LgTaskArgs<RemoteRefTaskArgs> {
      public:
        static const LgTaskID TASK_ID = LG_REMOTE_REF_TASK_ID;
      public:
        RemoteRefTaskArgs(EquivalenceSet *s, RtEvent d)
          : LgTaskArgs<RemoteRefTaskArgs>(implicit_provenance), 
            set(s), done(d) { }
      public:
        EquivalenceSet *const set;
        const RtEvent done;
      };
      struct DeferRayTraceArgs : public LgTaskArgs<DeferRayTraceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_RAY_TRACE_TASK_ID;
      public:
        DeferRayTraceArgs(EquivalenceSet *s, VersionManager *t,
                          IndexSpaceExpression *e, IndexSpace h,
                          AddressSpaceID o, RtUserEvent d)
          : LgTaskArgs<DeferRayTraceArgs>(implicit_provenance),
            set(s), target(t), expr(e), handle(h), origin(o), done(d) { }
      public:
        EquivalenceSet *const set;
        VersionManager *const target;
        IndexSpaceExpression *const expr;
        const IndexSpace handle;
        const AddressSpaceID origin;
        const RtUserEvent done;
      };
    protected:
      enum EqState {
        // Owner starts in the mapping state, goes to pending refinement
        // once there are any refinements to be done which will wait for
        // all mappings to finish and then goes to refined once any 
        // refinements have been done
        MAPPING_STATE,
        PENDING_REFINED_STATE, // waiting for mappings to drain
        REFINED_STATE, // subsets is stable and no refinements being performed
        REFINING_STATE, // running the refinement task
        // Remote copies start in the invalid state, go to pending valid
        // while waiting for a lease on the current subsets, valid once they 
        // get a lease, pending invalid once they get an invalid notification
        // but have outsanding mappings, followed by invalid
        INVALID_STATE,
        PENDING_VALID_STATE,
        VALID_STATE,
        PENDING_INVALID_STATE,
      };
    protected:
      class RefinementThunk : public Collectable {
      public:
        RefinementThunk(IndexSpaceExpression *expr, IndexSpaceNode *node,
            EquivalenceSet *owner, AddressSpaceID source);
        RefinementThunk(const RefinementThunk &rhs) 
          : owner(NULL), expr(NULL), refinement(NULL) { assert(false); }
        ~RefinementThunk(void) { }
      public:
        EquivalenceSet* perform_refinement(void);
        EquivalenceSet* get_refinement(void);
      public:
        EquivalenceSet *const owner;
        IndexSpaceExpression *const expr;
        EquivalenceSet *const refinement;
      protected:
        RtUserEvent refinement_ready;
      };
      // Pending requests for updates
      struct PendingRequest : public LegionHeapify<PendingRequest> {
      public:
        PendingRequest(Operation *o, RtUserEvent ready, RtUserEvent applied,
                       const RegionUsage &u, const bool relaxed) 
          : op(o), ready_event(ready), applied_event(applied), usage(u),
            runtime_relaxed(relaxed), remaining_updates(0), 
            remaining_invalidates(0) { }
      public:
        Operation *const op;
        const RtUserEvent ready_event;
        const RtUserEvent applied_event;
        const RegionUsage usage;
        const bool runtime_relaxed;
        std::set<RtEvent> applied_events;
        FieldMask exclusive_mask;
        FieldMask single_redop_mask;
        int remaining_updates;
        int remaining_invalidates;
      };
      // Deferred update reqeusts
      struct DeferredRequest {
      public:
        DeferredRequest(AddressSpaceID invalid, PendingRequest *pending,
                        bool inval, ReductionOpID skip)
          : invalid_space(invalid), pending_request(pending),
            skip_redop(skip), invalidate(inval) { }
      public:
        const AddressSpaceID invalid_space;
        PendingRequest *const pending_request;
        const ReductionOpID skip_redop;
        const bool invalidate;
      };
      struct DisjointPartitionRefinement {
      public:
        DisjointPartitionRefinement(IndexPartNode *p)
          : partition(p) { }
      public:
        IndexPartNode *const partition;
        std::map<IndexSpaceNode*,EquivalenceSet*> children;
      };
    public:
      EquivalenceSet(Runtime *rt, DistributedID did,
                     AddressSpaceID owner_space,
                     AddressSpaceID logical_owner,
                     IndexSpaceExpression *expr, 
                     IndexSpaceNode *index_space_node,
                     Reservation version_lock, bool register_now);
      EquivalenceSet(const EquivalenceSet &rhs);
      virtual ~EquivalenceSet(void);
    public:
      EquivalenceSet& operator=(const EquivalenceSet &rhs);
    public:
      inline bool is_logical_owner(void) const 
        { return (local_space == logical_owner_space); }
      inline void record_mutated_guard(Operation *op, const FieldMask &mask)
        {
#ifdef DEBUG_LEGION
          LegionMap<Operation*,MappingGuard>::aligned::iterator finder = 
            mapping_guards.find(op);
          assert(finder != mapping_guards.end()); 
          finder->second.mutated_mask |= mask;
#else
          mapping_guards[op].mutated_mask |= mask;
#endif
          // Always update the summary
          mutated_guard_summary |= mask;
        }
    public:
      // From distributed collectable
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      void clone_from(EquivalenceSet *parent);
      inline bool has_unrefined_remainder(void) const 
        { return (unrefined_remainder != NULL); }
      void refine_remainder(void);
      bool acquire_mapping_guard(Operation *op,
                                 const FieldMask &guard_mask,
                                 std::set<EquivalenceSet*> &alt_sets,
                                 bool recursed = false);
      void remove_mapping_guard(Operation *op);
      RtEvent ray_trace_equivalence_sets(VersionManager *target,
                                         IndexSpaceExpression *expr, 
                                         IndexSpace handle,
                                         AddressSpaceID source); 
      void request_valid_copy(Operation *op, FieldMask request_mask,
                              const RegionUsage usage,
                              std::set<RtEvent> &ready_events, 
                              std::set<RtEvent> &applied_events,
                              AddressSpaceID request_space,
                              // Parameter for the runtime to allow
                              // relaxed replication under safe circumstances
                              // such as dynamic control replication
                              const bool runtime_relaxed,
                              PendingRequest *pending_request = NULL);
      void record_subset(EquivalenceSet *set);
    public:
      // Analysis methods
      inline bool has_restrictions(const FieldMask &mask) const
        { return !(mask * restricted_fields); }
      void initialize_set(const RegionUsage &usage,
                          const FieldMask &user_mask,
                          const bool restricted,
                          const InstanceSet &sources,
            const std::vector<InstanceView*> &corresponding,
                          std::set<RtEvent> &applied_events);
      bool find_valid_instances(FieldMaskSet<LogicalView> &insts,
                                const FieldMask &user_mask) const;
      bool filter_valid_instances(FieldMaskSet<LogicalView> &insts,
                                  const FieldMask &user_mask) const;
      bool find_reduction_instances(FieldMaskSet<ReductionView> &insts,
                ReductionOpID redop, const FieldMask &user_mask) const;
      bool filter_reduction_instances(FieldMaskSet<ReductionView> &insts,
                  ReductionOpID redop, const FieldMask &user_mask) const;
      void update_set(Operation *op, const RegionUsage &usage, 
                      const FieldMask &user_mask,
                      const InstanceSet &target_instances,
                      const std::vector<InstanceView*> &target_views,
                      CopyFillAggregator &input_aggregator,
                      CopyFillAggregator &output_aggregator,
                      std::set<RtEvent> &applied_events,
                      FieldMask *initialized = NULL);
      void acquire_restrictions(Operation *op, FieldMask acquire_mask,
                                FieldMaskSet<InstanceView> &instances,
          std::map<InstanceView*,std::set<IndexSpaceExpression*> > &inst_exprs);
      void release_restrictions(Operation *op, const FieldMask &release_mask,
                                CopyFillAggregator &release_aggregator,
                                FieldMaskSet<InstanceView> &instances,
          std::map<InstanceView*,std::set<IndexSpaceExpression*> > &inst_exprs,
                                std::set<RtEvent> &ready_events);
      void issue_across_copies(const RegionUsage &usage,
              const FieldMask &src_mask, 
              const InstanceSet &source_instances,
              const InstanceSet &target_instances,
              const std::vector<InstanceView*> &source_views,
              const std::vector<InstanceView*> &target_views,
              IndexSpaceExpression *overlap, CopyFillAggregator &aggregator,
              PredEvent pred_guard, ReductionOpID redop, FieldMask &initialized,
              const std::vector<unsigned> *src_indexes = NULL,
              const std::vector<unsigned> *dst_indexes = NULL,
              const std::vector<CopyAcrossHelper*> *across_helpers = NULL)const;
      void overwrite_set(Operation *op, LogicalView *view,const FieldMask &mask,
                         CopyFillAggregator &output_aggregator,
                         std::set<RtEvent> &ready_events,
                         PredEvent pred_guard = PredEvent::NO_PRED_EVENT,
                         bool add_restriction = false);
      void filter_set(Operation *op, LogicalView *view, const FieldMask &mask,
                      bool remove_restriction = false);
    protected:
      // Help for analysis, all must be called while holding the lock
      void record_instances(const FieldMask &record_mask, 
                            const InstanceSet &target_instances,
                            const std::vector<InstanceView*> &target_views,
                                  ReferenceMutator &mutator);
      void issue_update_copies_and_fills(CopyFillAggregator &aggregator,
                                         FieldMask update_mask,
                                         const InstanceSet &target_instances,
                         const std::vector<InstanceView*> &target_views,
                                         IndexSpaceExpression *expr,
                                         bool skip_check = false) const;
      void filter_valid_instances(const FieldMask &filter_mask);
      void filter_reduction_instances(const FieldMask &filter_mask);
      void apply_reductions(const FieldMask &reduce_mask, 
                            CopyFillAggregator &aggregator);
      void copy_out(const FieldMask &restricted_mask,
                    const InstanceSet &src_instances,
                    const std::vector<InstanceView*> &src_views,
                          CopyFillAggregator &aggregator) const;
      // For the case with a single instance and a logical view
      void copy_out(const FieldMask &restricted_mask, LogicalView *src_view,
                          CopyFillAggregator &aggregator) const;
      void advance_version_numbers(FieldMask advance_mask);
    protected:
      void perform_refinements(void);
      void finalize_disjoint_refinement(void);
      void remove_remote_references(RtEvent done);
      void send_equivalence_set(AddressSpaceID target);
      void pack_initial_reduction_state(Serializer &rez) const;
      void unpack_initial_reduction_state(Deserializer &derez);
      void add_pending_refinement(RefinementThunk *thunk); // call with lock
      void launch_refinement_task(void); // call with lock
      void process_subset_request(AddressSpaceID source,bool needs_lock = true);
      void process_subset_response(Deserializer &derez);
      void process_subset_invalidation(RtUserEvent to_trigger);
      void invalidate_remote_state(RtUserEvent to_trigger);
      RtEvent record_reduction_application(FieldMask reduced_mask,
                                           bool needs_lock = false);
      void pack_state(Serializer &rez, RtUserEvent &handled_event,
                      PendingRequest *pending_request,
                      const FieldMask &pack_mask, 
                      ReductionOpID skip_redop, bool invalidate);
    protected:
      void update_single_copies(FieldMask &to_update,
                                AddressSpaceID request_space,
                                PendingRequest *pending_request,
                                unsigned &pending_updates,
                                unsigned &pending_invalidates,
                                const bool needs_updates,
                                FieldMask &single_fields,
              LegionMap<AddressSpaceID,FieldMask>::aligned &single_copies);
      void upgrade_single_to_multi(FieldMask &to_update,
                                   AddressSpaceID request_space,
                                   PendingRequest *pending_request,
                                   unsigned &pending_updates,
                                   ReductionOpID redop,
                                   FieldMask &single_fields,
              LegionMap<AddressSpaceID,FieldMask>::aligned &single_copies,
                                   FieldMask &multi_fields,
              LegionMap<AddressSpaceID,FieldMask>::aligned &multi_copies);
      void record_exclusive_copy(AddressSpaceID request_space,
                                 const FieldMask &request_mask);
      void record_shared_copy(AddressSpaceID request_space,
                              const FieldMask &request_mask);
      void record_relaxed_copy(AddressSpaceID request_space,
                               const FieldMask &request_mask);
      void record_single_reduce(AddressSpaceID request_space,
                                const FieldMask &request_mask,
                                ReductionOpID redop);
      // For filtering exclusive and single redop
      void filter_single_copies(FieldMask &to_filter, 
                                FieldMask &single_fields,
          LegionMap<AddressSpaceID,FieldMask>::aligned &single_copies,
                                AddressSpaceID request_space,
                                PendingRequest *pending_request,
                                unsigned &pending_updates,
                                unsigned &pending_invalidates,
                                const bool needs_udpates);
      // For filtering shared and multi redop
      void filter_multi_copies(FieldMask &to_filter,
                               FieldMask &multi_fields,
         LegionMap<AddressSpaceID,FieldMask>::aligned &multi_copies,
                               AddressSpaceID request_space,
                               PendingRequest *pending_request,
                               unsigned &pending_updates,
                               unsigned &pending_invalidates,
                               const bool needs_updates,
                               const bool updates_from_all);
      void filter_redop_modes(FieldMask to_filter);
      void request_update(AddressSpaceID valid_space, 
                          AddressSpaceID invalid_space,
                          const FieldMask &update_mask,
                          PendingRequest *pending_request,
                          unsigned &pending_updates,
                          bool invalidate,
                          ReductionOpID skip_redop = 0,
                          bool needs_lock = false);
      void process_update_response(PendingRequest *pending_request, 
                                   Deserializer &derez);
      void request_invalidate(AddressSpaceID target,
                              AddressSpaceID source,
                              const FieldMask &invalidate_mask,
                              PendingRequest *pending_request,
                              unsigned &pending_updates,
                              bool meta_only,
                              bool needs_lock = false);
      void record_pending_counts(PendingRequest *pending_request,
                                 unsigned pending_updates,
                                 unsigned pending_invalidates,
                                 bool needs_lock);
      void record_invalidation(PendingRequest *pending_request,
                               bool meta_only, bool needs_lock = false);
      bool finalize_pending_update(PendingRequest *pending_request);
      void finalize_pending_request(PendingRequest *pending_request);
      void perform_all_deferred_requests(void);
      void perform_ready_deferred_requests(void);
#ifdef DEBUG_LEGION
      void sanity_check(void) const;
#endif
    public:
      static void handle_refinement(const void *args);
      static void handle_remote_references(const void *args);
      static void handle_ray_trace(const void *args);
      static void handle_equivalence_set_request(Deserializer &derez,
                            Runtime *runtime, AddressSpaceID source);
      static void handle_equivalence_set_response(Deserializer &derez,
                            Runtime *runtime, AddressSpaceID source);
      static void handle_subset_request(Deserializer &derez,
                            Runtime *runtime, AddressSpaceID source);
      static void handle_subset_response(Deserializer &derez, Runtime *runtime);
      static void handle_subset_invalidation(Deserializer &derez, Runtime *rt);
      static void handle_ray_trace_request(Deserializer &derez, 
                            Runtime *runtime, AddressSpaceID source);
      static void handle_ray_trace_response(Deserializer &derez, Runtime *rt);
      static void handle_valid_request(Deserializer &derez, 
                            Runtime *runtime, AddressSpaceID source);
      static void handle_valid_response(Deserializer &derez, Runtime *rt);
      static void handle_update_request(Deserializer &derez, Runtime *rt);
      static void handle_update_response(Deserializer &derez, Runtime *rt);
      static void handle_invalidate_request(Deserializer &derez, Runtime *rt);
      static void handle_invalidate_response(Deserializer &derez, Runtime *rt);
      static void handle_reduction_application(Deserializer &derez,Runtime *rt);
    public:
      IndexSpaceExpression *const set_expr;
      IndexSpaceNode *const index_space_node; // can be NULL
      const AddressSpaceID logical_owner_space;
      const Reservation version_lock;
    protected:
      mutable LocalLock eq_lock;
    protected:
      // This is the actual physical state of the equivalence class
      FieldMaskSet<LogicalView>                           valid_instances;
      std::map<unsigned/*field idx*/,
               std::vector<ReductionView*> >              reduction_instances;
      FieldMask                                           reduction_fields;
      FieldMaskSet<InstanceView>                          restricted_instances;
      FieldMask                                           restricted_fields;
      // This is the current version number of the equivalence set
      // Each field should appear in exactly one mask
      LegionMap<VersionID,FieldMask>::aligned             version_numbers;
    protected:
      // Track the current state of this equivalence state
      EqState eq_state;
      // Track the mapping events of the current operations that
      // are using this equivalence class to map, the count 
      // indicates how many times it has been acquired
      LegionMap<Operation*,MappingGuard>::aligned mapping_guards;
      // Keep a summary mask of the fields in the mapping guards
      FieldMask mutated_guard_summary;
      // Keep track of the refinements that need to be done
      std::vector<RefinementThunk*> pending_refinements;
      // Keep an event to track when the refinements are ready
      RtUserEvent transition_event;
      // Keep an order on all operations that attempt to acquire
      // this equivalence class. Hopefully 64 bits is enough that
      // we never have to reset this
      unsigned long long next_guard_index;
    protected:
      // If we have sub sets then we track those here
      // If this data structure is not empty, everything above is invalid
      // except for the remainder expression which is just waiting until
      // someone else decides that they need to access it
      std::vector<EquivalenceSet*> subsets;
      // Set on the owner node for tracking the remote subset leases
      std::set<AddressSpaceID> remote_subsets, pending_subset_requests;
      // Index space expression for unrefined remainder of our set_expr
      // This is only valid on the owner node
      IndexSpaceExpression *unrefined_remainder;
      // For detecting when we are slicing a disjoint partition
      DisjointPartitionRefinement *disjoint_partition_refinement;
    protected:
      // Track which fields we hold in exclusive mode
      FieldMask exclusive_fields;
      // Track which fields we hold in shared mode
      FieldMask shared_fields;
      // Track which fields we hold in single reduction mode
      FieldMask single_redop_fields;
      // Track which fields we hold in multiple reduction mode
      FieldMask multi_redop_fields;
      // Track which fields we hold in a relaxed mode
      FieldMask relaxed_fields;
      // Track the reduction modes for fields
      LegionMap<ReductionOpID,FieldMask>::aligned redop_modes;
      // These members are only valid on the owner node
      // Track which nodes have shared copies of fields
      LegionMap<AddressSpaceID,FieldMask>::aligned exclusive_copies;
      LegionMap<AddressSpaceID,FieldMask>::aligned shared_copies;
      LegionMap<AddressSpaceID,FieldMask>::aligned single_reduction_copies;
      LegionMap<AddressSpaceID,FieldMask>::aligned multi_reduction_copies;
      // Requests that are pending from this node 
      FieldMaskSet<PendingRequest> outstanding_requests;
      // Deferred update requests
      FieldMaskSet<DeferredRequest> deferred_requests;
      // Track references for inflight messages so we only remove them
      // once we know that they have successfully handled 
      std::map<RtEvent,std::vector<LogicalView*> > inflight_references;
    public:
      static const VersionID init_version = 1;
    };

    /**
     * \class VersionManager
     * The VersionManager class tracks the starting equivalence
     * sets for a given node in the logical region tree. Note
     * that its possible that these have since been shattered
     * and we need to traverse them, but it's a cached starting
     * point that doesn't involve tracing the entire tree.
     */
    class VersionManager : public LegionHeapify<VersionManager> {
    public:
      static const AllocationType alloc_type = VERSION_MANAGER_ALLOC;
    public:
      VersionManager(RegionTreeNode *node, ContextID ctx); 
      VersionManager(const VersionManager &manager);
      ~VersionManager(void);
    public:
      VersionManager& operator=(const VersionManager &rhs);
    public:
      inline bool has_versions(void) const { return has_equivalence_sets; }
    public:
      void reset(void);
    public:
      void perform_versioning_analysis(InnerContext *parent_ctx,
                                       VersionInfo *version_info,
                                       RegionNode *region_node);
      void record_equivalence_set(EquivalenceSet *set);
      void update_equivalence_sets(const std::set<EquivalenceSet*> &alt_sets);
    public:
      void print_physical_state(RegionTreeNode *node,
                                const FieldMask &capture_mask,
                                TreeStateLogger *logger);
    public:
      const ContextID ctx;
      RegionTreeNode *const node;
      Runtime *const runtime;
    protected:
      mutable LocalLock manager_lock;
    protected: 
      std::set<EquivalenceSet*> equivalence_sets; 
      RtUserEvent equivalence_sets_ready;
      volatile bool has_equivalence_sets;
    };

    typedef DynamicTableAllocator<VersionManager,10,8> VersionManagerAllocator; 

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
      void record_aliased_children(unsigned depth, const FieldMask &mask);
      void clear();
    public:
#ifdef DEBUG_LEGION 
      bool has_child(unsigned depth) const;
      LegionColor get_child(unsigned depth) const;
#else
      inline bool has_child(unsigned depth) const
        { return path[depth] != INVALID_COLOR; }
      inline LegionColor get_child(unsigned depth) const
        { return path[depth]; }
#endif
      inline unsigned get_path_length(void) const
        { return ((max_depth-min_depth)+1); }
      inline unsigned get_min_depth(void) const { return min_depth; }
      inline unsigned get_max_depth(void) const { return max_depth; }
    public:
      const FieldMask* get_aliased_children(unsigned depth) const;
    protected:
      std::vector<LegionColor> path;
      LegionMap<unsigned/*depth*/,FieldMask>::aligned interfering_children;
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
      PathTraverser(RegionTreePath &path);
      PathTraverser(const PathTraverser &rhs);
      virtual ~PathTraverser(void);
    public:
      PathTraverser& operator=(const PathTraverser &rhs);
    public:
      // Return true if the traversal was successful
      // or false if one of the nodes exit stopped early
      bool traverse(RegionTreeNode *start);
    public:
      virtual bool visit_region(RegionNode *node) = 0;
      virtual bool visit_partition(PartitionNode *node) = 0;
    protected:
      RegionTreePath &path;
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
      NodeTraverser(bool force = false)
        : force_instantiation(force) { }
    public:
      virtual bool break_early(void) const { return false; }
      virtual bool visit_only_valid(void) const = 0;
      virtual bool visit_region(RegionNode *node) = 0;
      virtual bool visit_partition(PartitionNode *node) = 0;
    public:
      const bool force_instantiation;
    };

    /**
     * \class LogicalPathRegistrar
     * A class that registers dependences for an operation
     * against all other operation with an overlapping
     * field mask along a given path
     */
    class LogicalPathRegistrar : public PathTraverser {
    public:
      LogicalPathRegistrar(ContextID ctx, Operation *op,
            const FieldMask &field_mask, RegionTreePath &path);
      LogicalPathRegistrar(const LogicalPathRegistrar &rhs);
      virtual ~LogicalPathRegistrar(void);
    public:
      LogicalPathRegistrar& operator=(const LogicalPathRegistrar &rhs);
    public:
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    public:
      const ContextID ctx;
      const FieldMask field_mask;
      Operation *const op;
    };

    /**
     * \class LogicalRegistrar
     * A class that registers dependences for an operation
     * against all other operations with an overlapping
     * field mask.
     */
    class LogicalRegistrar : public NodeTraverser {
    public:
      LogicalRegistrar(ContextID ctx, Operation *op,
                       const FieldMask &field_mask,
                       bool dom);
      LogicalRegistrar(const LogicalRegistrar &rhs);
      ~LogicalRegistrar(void);
    public:
      LogicalRegistrar& operator=(const LogicalRegistrar &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    public:
      const ContextID ctx;
      const FieldMask field_mask;
      Operation *const op;
      const bool dominate;
    };

    /**
     * \class CurrentInitializer 
     * A class for initializing current states 
     */
    class CurrentInitializer : public NodeTraverser {
    public:
      CurrentInitializer(ContextID ctx);
      CurrentInitializer(const CurrentInitializer &rhs);
      ~CurrentInitializer(void);
    public:
      CurrentInitializer& operator=(const CurrentInitializer &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
    };

    /**
     * \class CurrentInvalidator 
     * A class for invalidating current states 
     */
    class CurrentInvalidator : public NodeTraverser {
    public:
      CurrentInvalidator(ContextID ctx, bool users_only);
      CurrentInvalidator(const CurrentInvalidator &rhs);
      ~CurrentInvalidator(void);
    public:
      CurrentInvalidator& operator=(const CurrentInvalidator &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
      const bool users_only;
    };

    /**
     * \class DeletionInvalidator
     * A class for invalidating current states for deletions
     */
    class DeletionInvalidator : public NodeTraverser {
    public:
      DeletionInvalidator(ContextID ctx, const FieldMask &deletion_mask);
      DeletionInvalidator(const DeletionInvalidator &rhs);
      ~DeletionInvalidator(void);
    public:
      DeletionInvalidator& operator=(const DeletionInvalidator &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
      const FieldMask &deletion_mask;
    };

    /**
     * \class InstanceRef
     * A class for keeping track of references to physical instances
     */
    class InstanceRef : public LegionHeapify<InstanceRef> {
    public:
      InstanceRef(bool composite = false);
      InstanceRef(const InstanceRef &rhs);
      InstanceRef(PhysicalManager *manager, const FieldMask &valid_fields,
                  ApEvent ready_event = ApEvent::NO_AP_EVENT);
      ~InstanceRef(void);
    public:
      InstanceRef& operator=(const InstanceRef &rhs);
    public:
      bool operator==(const InstanceRef &rhs) const;
      bool operator!=(const InstanceRef &rhs) const;
    public:
      inline bool has_ref(void) const { return (manager != NULL); }
      inline ApEvent get_ready_event(void) const { return ready_event; }
      inline void set_ready_event(ApEvent ready) { ready_event = ready; }
      inline PhysicalManager* get_manager(void) const { return manager; }
      inline const FieldMask& get_valid_fields(void) const 
        { return valid_fields; }
      inline void update_fields(const FieldMask &update) 
        { valid_fields |= update; }
    public:
      inline bool is_local(void) const { return local; }
      MappingInstance get_mapping_instance(void) const;
      bool is_virtual_ref(void) const; 
    public:
      // These methods are used by PhysicalRegion::Impl to hold
      // valid references to avoid premature collection
      void add_valid_reference(ReferenceSource source) const;
      void remove_valid_reference(ReferenceSource source) const;
    public:
      Memory get_memory(void) const;
    public:
      bool is_field_set(FieldID fid) const;
      LegionRuntime::Accessor::RegionAccessor<
          LegionRuntime::Accessor::AccessorType::Generic>
            get_accessor(void) const;
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const;
    public:
      void pack_reference(Serializer &rez) const;
      void unpack_reference(Runtime *rt, Deserializer &derez, RtEvent &ready);
    protected:
      FieldMask valid_fields; 
      ApEvent ready_event;
      PhysicalManager *manager;
      bool local;
    };

    /**
     * \class InstanceSet
     * This class is an abstraction for representing one or more
     * instance references. It is designed to be light-weight and
     * easy to copy by value. It maintains an internal copy-on-write
     * data structure to avoid unnecessary premature copies.
     */
    class InstanceSet {
    public:
      struct CollectableRef : public Collectable, public InstanceRef {
      public:
        CollectableRef(void)
          : Collectable(), InstanceRef() { }
        CollectableRef(const InstanceRef &ref)
          : Collectable(), InstanceRef(ref) { }
        CollectableRef(const CollectableRef &rhs)
          : Collectable(), InstanceRef(rhs) { }
        ~CollectableRef(void) { }
      public:
        CollectableRef& operator=(const CollectableRef &rhs);
      };
      struct InternalSet : public Collectable {
      public:
        InternalSet(size_t size = 0)
          { if (size > 0) vector.resize(size); }
        InternalSet(const InternalSet &rhs) : vector(rhs.vector) { }
        ~InternalSet(void) { }
      public:
        InternalSet& operator=(const InternalSet &rhs)
          { assert(false); return *this; }
      public:
        inline bool empty(void) const { return vector.empty(); }
      public:
        LegionVector<InstanceRef>::aligned vector; 
      };
    public:
      InstanceSet(size_t init_size = 0);
      InstanceSet(const InstanceSet &rhs);
      ~InstanceSet(void);
    public:
      InstanceSet& operator=(const InstanceSet &rhs);
      bool operator==(const InstanceSet &rhs) const;
      bool operator!=(const InstanceSet &rhs) const;
    public:
      InstanceRef& operator[](unsigned idx);
      const InstanceRef& operator[](unsigned idx) const;
    public:
      bool empty(void) const;
      size_t size(void) const;
      void resize(size_t new_size);
      void clear(void);
      void add_instance(const InstanceRef &ref);
      bool is_virtual_mapping(void) const;
    public:
      void pack_references(Serializer &rez) const;
      void unpack_references(Runtime *runtime, Deserializer &derez, 
                             std::set<RtEvent> &ready_events);
    public:
      void add_valid_references(ReferenceSource source) const;
      void remove_valid_references(ReferenceSource source) const;
    public:
      void update_wait_on_events(std::set<ApEvent> &wait_on_events) const;
    public:
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const;
    protected:
      void make_copy(void);
    protected:
      union {
        CollectableRef* single;
        InternalSet*     multi;
      } refs;
      bool single;
      mutable bool shared;
    };

    /**
     * \class VersioningInvalidator
     * A class for reseting the versioning managers for 
     * a deleted region (sub)-tree so that version states
     * and the things they point to can be cleaned up
     * by the garbage collector. The better long term
     * answer is to have individual contexts do this.
     */
    class VersioningInvalidator : public NodeTraverser {
    public:
      VersioningInvalidator(void);
      VersioningInvalidator(RegionTreeContext ctx);
    public:
      virtual bool visit_only_valid(void) const { return true; }
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
      const bool invalidate_all;
    };

  }; // namespace Internal 
}; // namespace Legion

#endif // __LEGION_ANALYSIS_H__
