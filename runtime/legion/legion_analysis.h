/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "legion_types.h"
#include "legion_utilities.h"
#include "legion_allocation.h"
#include "garbage_collection.h"

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

    class FieldVersions : public Collectable {
    public:
      FieldVersions(void);
      FieldVersions(const FieldVersions &rhs);
      ~FieldVersions(void);
    public:
      FieldVersions& operator=(const FieldVersions &rhs);
    public:
      inline const LegionMap<VersionID,FieldMask>::aligned& 
            get_field_versions(void) const { return field_versions; }
      inline LegionMap<VersionID,FieldMask>::aligned& 
            get_mutable_field_versions(void) { return field_versions; }
    public:
      void add_field_version(VersionID vid, const FieldMask &mask);
    private:
      LegionMap<VersionID,FieldMask>::aligned field_versions;
    };

    /**
     * \class VersionInfo
     * A class for tracking version information about region usage
     */
    class VersionInfo {
    public:
      struct NodeInfo {
      public:
        NodeInfo(void) : physical_state(NULL), field_versions(NULL), 
          bit_mask(0) { set_needs_capture(); } 
        // Always make deep copies of the physical state
        NodeInfo(const NodeInfo &rhs);
        ~NodeInfo(void);
        NodeInfo& operator=(const NodeInfo &rhs);
      public:
        // Don't forget to update clone_from methods
        // when changing these bit values
        static const unsigned BASE_FIELDS_MASK = 0x3F;
        inline void set_path_only(void) { bit_mask |= 0x1; }
        inline void set_needs_final(void) { bit_mask |= 0x2; }
        inline void set_close_top(void) { bit_mask |= 0x4; }
        inline void set_close_node(void) { bit_mask |= 0x8; }
        inline void set_leave_open(void) { bit_mask |= 0x10; }
        inline void set_split_node(void) { bit_mask |= 0x20; }
        inline void set_needs_capture(void) { bit_mask |= 0x40; }
        inline void unset_needs_capture(void)
        { if (needs_capture()) bit_mask &= BASE_FIELDS_MASK; }
      public:
        inline bool path_only(void) const { return (0x1 & bit_mask); }
        inline bool needs_final(void) const { return (0x2 & bit_mask); }
        inline bool close_top(void) const { return (0x4 & bit_mask); }
        inline bool close_node(void) const { return (0x8 & bit_mask); }
        inline bool leave_open(void) const { return (0x10 & bit_mask); }
        inline bool split_node(void) const { return (0x20 & bit_mask); }
        inline bool needs_capture(void) const { return (0x40 & bit_mask); }
      public:
        PhysicalState *physical_state;
        FieldVersions *field_versions;
        FieldMask        advance_mask;
      public:
        unsigned bit_mask;
      };
    public:
      VersionInfo(void);
      VersionInfo(const VersionInfo &rhs);
      ~VersionInfo(void);
    public:
      VersionInfo& operator=(const VersionInfo &rhs);
    public:
      inline NodeInfo& find_tree_node_info(RegionTreeNode *node)
        { return node_infos[node]; }
    public:
      void set_upper_bound_node(RegionTreeNode *node);
      inline bool is_upper_bound_node(RegionTreeNode *node) const
        { return (node == upper_bound_node); }
      inline RegionTreeNode* get_upper_bound_node(void) const 
        { return upper_bound_node; }
    public:
      void merge(const VersionInfo &rhs, const FieldMask &mask);
      void apply_mapping(ContextID ctx, AddressSpaceID target,
                         std::set<RtEvent> &applied_conditions,
			 bool copy_previous = false);
      void apply_close(ContextID ctx, AddressSpaceID target,
             const LegionMap<ColorPoint,FieldMask>::aligned &closed_children,
                       std::set<RtEvent> &applied_conditions); 
      void reset(void);
      void release(void);
      void clear(void);
      void recapture_state(void);
      void sanity_check(RegionTreeNode *node);
    public:
      PhysicalState* find_physical_state(RegionTreeNode *node, bool capture); 
      FieldVersions* get_versions(RegionTreeNode *node) const;
      const FieldMask& get_advance_mask(RegionTreeNode *node, 
                                        bool &is_split) const;
    public:
      // These pack all the state for the version info
      void pack_version_info(Serializer &rez);
      void unpack_version_info(Deserializer &derez);
    public:
      // These only pack the version numbers
      void pack_version_numbers(Serializer &rez);
      void unpack_version_numbers(Deserializer &derez,RegionTreeForest *forest);
    public:
      void make_local(std::set<RtEvent> &preconditions,
                      Operation *owner_op, RegionTreeForest *forest);
      void clone_version_info(RegionTreeForest *forest, LogicalRegion handle,
                              const VersionInfo &rhs, bool check_below);
      void clone_version_numbers(const VersionInfo &rhs, 
                                 const CompositeCloser &closer);
    public:
      inline bool is_packed(void) const { return packed; }
      void pack_buffer(Serializer &rez);
      void unpack_buffer(Operation *owner_op, RegionTreeForest *forest);
    protected:
      void pack_node_info(Serializer &rez, NodeInfo &info,
                          RegionTreeNode *node);
      void unpack_node_info(Operation *owner_op, RegionTreeNode *node,
                            Deserializer &derez);
    protected:
      void pack_node_version_numbers(Serializer &rez, NodeInfo &info,
                                     RegionTreeNode *node);
      void unpack_node_version_numbers(RegionTreeNode *node, 
                                       Deserializer &derez);
    protected:
      LegionMap<RegionTreeNode*,NodeInfo>::aligned node_infos;
      RegionTreeNode *upper_bound_node;
    protected:
      bool packed;
      void *packed_buffer;
      size_t packed_size;
    }; 

    /**
     * \class Restriction
     * A class for tracking restrictions that occur as part of
     * relaxed coherence and with tracking external resources
     */
    class Restriction {
    public:
      Restriction(RegionNode *node);
      Restriction(const Restriction &rhs);
      ~Restriction(void);
    public:
      Restriction& operator=(const Restriction &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      void add_restricted_instance(InstanceManager *inst, 
                                   const FieldMask &restricted_fields);
    public:
      void find_restrictions(RegionTreeNode *node, 
                             FieldMask &possibly_restricted,
                             RestrictInfo &restrict_info) const;
      bool matches(DetachOp *op, RegionNode *node,
                   FieldMask &remaining_fields); 
      void remove_restricted_fields(FieldMask &remaining_fields) const;
    public:
      void add_acquisition(AcquireOp *op, RegionNode *node,
                           FieldMask &remaining_fields);
      void remove_acquisition(ReleaseOp *op, RegionNode *node,
                              FieldMask &remaining_fields);
    public:
      void add_restriction(AttachOp *op, RegionNode *node,
                InstanceManager *manager, FieldMask &remaining_fields);
      void remove_restriction(DetachOp *op, RegionNode *node,
                              FieldMask &remaining_fields);

    public:
      const RegionTreeID tree_id;
      RegionNode *const local_node;
    protected:
      FieldMask restricted_fields;
      std::set<Acquisition*> acquisitions;
      // We only need garbage collection references on these
      // instances because we know one of two things is always
      // true: either they are attached files so they aren't 
      // subject to memories in which garbage collection will
      // occur, or they are simultaneous restricted, so that
      // enclosing context of the parent task has a valid 
      // reference to them so there is no need for us to 
      // have a valid reference.
      // Same in RestrictInfo
      LegionMap<InstanceManager*,FieldMask>::aligned instances;
    };

    /**
     * \class Acquisition
     * A class for tracking when restrictions are relaxed by
     * explicit acquisitions of a region
     */
    class Acquisition {
    public:
      Acquisition(RegionNode *node, const FieldMask &acquired_fields);
      Acquisition(const Acquisition &rhs);
      ~Acquisition(void);
    public:
      Acquisition& operator=(const Acquisition &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      void find_restrictions(RegionTreeNode *node, 
                             FieldMask &possibly_restricted,
                             RestrictInfo &restrict_info) const;
      bool matches(ReleaseOp *op, RegionNode *node, 
                   FieldMask &remaining_fields);
      void remove_acquired_fields(FieldMask &remaining_fields) const;
    public:
      void add_acquisition(AcquireOp *op, RegionNode *node,
                           FieldMask &remaining_fields);
      void remove_acquisition(ReleaseOp *op, RegionNode *node,
                              FieldMask &remaining_fields);
    public:
      void add_restriction(AttachOp *op, RegionNode *node,
                 InstanceManager *manager, FieldMask &remaining_fields);
      void remove_restriction(DetachOp *op, RegionNode *node,
                              FieldMask &remaining_fields);
    public:
      RegionNode *const local_node;
    protected:
      FieldMask acquired_fields;
      std::set<Restriction*> restrictions;
    };

    /**
     * \struct TracingInfo
     * Information about tracing needed for logical
     * dependence analysis.
     */
    struct TraceInfo {
    public:
      TraceInfo(bool already_tr,
                  LegionTrace *tr,
                  unsigned idx,
                  const RegionRequirement &r)
        : already_traced(already_tr), trace(tr),
          req_idx(idx), req(r) { }
    public:
      bool already_traced;
      LegionTrace *trace;
      unsigned req_idx;
      const RegionRequirement &req;
    };

    /**
     * \struct PhysicalUser
     * A class for representing physical users of a logical
     * region including necessary information to 
     * register execution dependences on the user.
     */
    struct PhysicalUser : public Collectable {
    public:
      static const AllocationType alloc_type = PHYSICAL_USER_ALLOC;
    public:
      PhysicalUser(void);
      PhysicalUser(const RegionUsage &u, const ColorPoint &child, 
                   UniqueID op_id, unsigned index);
      PhysicalUser(const PhysicalUser &rhs);
      ~PhysicalUser(void);
    public:
      PhysicalUser& operator=(const PhysicalUser &rhs);
    public:
      void pack_user(Serializer &rez);
      static PhysicalUser* unpack_user(Deserializer &derez, 
                                       bool add_reference);
    public:
      RegionUsage usage;
      ColorPoint child;
      UniqueID op_id;
      unsigned index; // region requirement index
    }; 

    /**
     * \struct TraversalInfo
     */
    struct TraversalInfo {
    public:
      TraversalInfo(ContextID ctx, Operation *op, unsigned index, 
                    const RegionRequirement &req, VersionInfo &version_info,
                    const FieldMask &traversal_mask, 
                    std::set<RtEvent> &map_applied_events);
    public:
      const ContextID ctx;
      Operation *const op;
      const unsigned index;
      const RegionRequirement &req;
      VersionInfo &version_info;
      const FieldMask traversal_mask;
      const UniqueID context_uid;
      std::set<RtEvent> &map_applied_events;
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
      LegionMap<ColorPoint,FieldMask>::aligned open_children;
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
                 const ColorPoint &child);
    public:
      bool overlaps(const FieldState &rhs) const;
      void merge(const FieldState &rhs, RegionTreeNode *node);
    public:
      void print_state(TreeStateLogger *logger, 
                       const FieldMask &capture_mask) const;
    public:
      OpenState open_state;
      ReductionOpID redop;
      unsigned rebuild_timeout;
    }; 

    /**
     * \struct VersionStateInfo
     * A small helper class for tracking collections of 
     * version state objects and their sets of fields
     */
    struct VersionStateInfo {
    public:
      VersionStateInfo(void)
        : single(true) { states.single_state = NULL; }
#ifdef DEBUG_LEGION
      ~VersionStateInfo(void)
      { if (!single) assert(states.multi_states == NULL); }
#endif
    public:
      FieldMask valid_fields;
      union {
        VersionState *single_state;
        LegionMap<VersionState*,FieldMask>::aligned *multi_states;
      } states;
      bool single;
    };

    /**
     * \class CurrentState 
     * Track all the information about the current state
     * of a logical region from a given context. This
     * is effectively all the information at the analysis
     * wavefront for this particular logical region.
     */
    class CurrentState {
    public:
      static const AllocationType alloc_type = CURRENT_STATE_ALLOC;
      static const VersionID init_version = 1;
    public:
      CurrentState(RegionTreeNode *owner, ContextID ctx);
      CurrentState(const CurrentState &state);
      ~CurrentState(void);
    public:
      CurrentState& operator=(const CurrentState &rhs);
      void* operator new(size_t count);
      void* operator new[](size_t count);
      void operator delete(void *ptr);
      void operator delete[](void *ptr);
    public:
      void check_init(void);
      void clear_logical_users(void);
      void reset(void);
      void clear_deleted_state(const FieldMask &deleted_mask);
      void sanity_check(void);
    public:
      void initialize_state(ApEvent term_event,
                            const RegionUsage &usage,
                            const FieldMask &user_mask,
                            const InstanceSet &targets,
                            SingleTask *context, unsigned init_index,
                            const std::vector<LogicalView*> &corresponding);
      void record_version_numbers(const FieldMask &mask,
                                  const LogicalUser &user,
                                  VersionInfo &version_info,
                                  bool capture_previous, bool path_only,
                                  bool need_final, bool close_top, 
                                  bool report_unversioned,
                                  bool close_node = false,
                                  bool capture_leave_open = false,
                                  bool split_node = false);
      void advance_version_numbers(const FieldMask &mask);
    public:
      void print_physical_state(RegionTreeNode *node,
                                const FieldMask &capture_mask,
                          LegionMap<ColorPoint,FieldMask>::aligned &to_traverse,
                                TreeStateLogger *logger);
    public:
      RegionTreeNode *const owner;
    public:
      LegionList<FieldState,
                 LOGICAL_FIELD_STATE_ALLOC>::track_aligned field_states;
      LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned 
                                                            curr_epoch_users;
      LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned 
                                                            prev_epoch_users;
      LegionMap<VersionID,VersionStateInfo>::aligned current_version_infos;
      LegionMap<VersionID,VersionStateInfo>::aligned previous_version_infos;
      // Fields for which we have outstanding local reductions
      FieldMask outstanding_reduction_fields;
      LegionMap<ReductionOpID,FieldMask>::aligned outstanding_reductions;
      // Fields which we know have been mutated below in the region tree
      FieldMask dirty_below;
      // Fields that have already undergone at least a partial close
      FieldMask partially_closed;
    protected:
      Reservation state_lock; 
    };

    typedef DynamicTableAllocator<CurrentState, 10, 8> CurrentStateAllocator;
 
    /**
     * \struct LogicalCloser
     * This structure helps keep track of the state
     * necessary for performing a close operation
     * on the logical region tree.
     */
    class LogicalCloser {
    public:
      struct ClosingInfo {
      public:
        ClosingInfo(void) { }
        ClosingInfo(const FieldMask &m,
                    const LegionDeque<LogicalUser>::aligned &users)
          : child_fields(m) 
        { child_users.insert(child_users.end(), users.begin(), users.end()); }
      public:
        FieldMask child_fields;
        FieldMask leave_open_mask;
        LegionList<LogicalUser,CLOSE_LOGICAL_ALLOC>::track_aligned child_users;
      };
      struct ClosingSet {
      public:
        ClosingSet(void) { }
        ClosingSet(const FieldMask &m)
          : closing_mask(m) { }
      public:
        inline void add_child(const ColorPoint &key, const FieldMask &open)
        {
          LegionMap<ColorPoint,FieldMask>::aligned::iterator finder = 
            children.find(key);
          if (finder != children.end())
          {
            if (!!open)
              finder->second |= open;
          }
          else
          {
            if (!!open)
              children[key] = open;
            else
              children[key] = FieldMask();
          }
        }
        inline void filter_children(void)
        {
          for (LegionMap<ColorPoint,FieldMask>::aligned::iterator it = 
                children.begin(); it != children.end(); it++)
          {
            it->second &= closing_mask;
          }
        }
      public:
        FieldMask closing_mask;
        // leave open may over-approximate so filter before
        // building the close operations!
        LegionMap<ColorPoint,FieldMask/*leave open*/>::aligned children;
      };
    public:
      LogicalCloser(ContextID ctx, const LogicalUser &u,
                    bool validates, bool captures);
      LogicalCloser(const LogicalCloser &rhs);
      ~LogicalCloser(void);
    public:
      LogicalCloser& operator=(const LogicalCloser &rhs);
    public:
      inline bool has_close_operations(void) const 
        { return (!closed_children.empty() || 
                  !read_only_children.empty() || !!flush_only_fields); }
      inline bool has_closed_fields(void) const { return !!closed_mask; }
      inline const FieldMask& get_closed_fields(void) const 
        { return closed_mask; }
      void record_closed_child(const ColorPoint &child, const FieldMask &mask,
                               bool leave_open, bool read_only_close);
      void record_partial_fields(const FieldMask &skipped_fields);
      void record_flush_only_fields(const FieldMask &flush_only);
      void initialize_close_operations(RegionTreeNode *target, 
                                       Operation *creator,
                                       const VersionInfo &version_info,
                                       const TraceInfo &trace_info);
      void add_next_child(const ColorPoint &next_child);
      void perform_dependence_analysis(const LogicalUser &current,
                                       const FieldMask &open_below,
             LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &cusers,
             LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned &pusers);
      void update_state(CurrentState &state);
      void register_close_operations(
              LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &users);
      void record_version_numbers(RegionTreeNode *node, CurrentState &state,
                                  const FieldMask &local_mask, bool leave_open);
      void record_top_version_numbers(RegionTreeNode *node, CurrentState &state);
      void merge_version_info(VersionInfo &target, const FieldMask &merge_mask);
    protected:
      static void compute_close_sets(
                     const LegionMap<ColorPoint,ClosingInfo>::aligned &children,
                     LegionList<ClosingSet>::aligned &close_sets);
      void create_normal_close_operations(RegionTreeNode *target, 
                          Operation *creator, const VersionInfo &local_info,
                          const VersionInfo &version_info,
                          const TraceInfo &trace_info,
                          LegionList<ClosingSet>::aligned &close_sets);
      void create_read_only_close_operations(RegionTreeNode *target, 
                          Operation *creator, const TraceInfo &trace_info,
                          const LegionList<ClosingSet>::aligned &close_sets);
      void register_dependences(const LogicalUser &current, 
                                const FieldMask &open_below,
             LegionMap<TraceCloseOp*,LogicalUser>::aligned &closes,
             LegionMap<ColorPoint,ClosingInfo>::aligned &children,
             LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned &ausers,
             LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &cusers,
             LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned &pusers);
    public:
      ContextID ctx;
      const LogicalUser &user;
      const bool validates;
      const bool capture_users;
      LegionDeque<LogicalUser>::aligned closed_users;
    protected:
      FieldMask closed_mask, partial_mask;
      LegionMap<ColorPoint,ClosingInfo>::aligned closed_children;
      LegionMap<ColorPoint,ClosingInfo>::aligned read_only_children;
    protected:
      // Use the base TraceCloseOp class so we can call the same
      // register_dependences method on all of them
      LegionMap<TraceCloseOp*,LogicalUser>::aligned normal_closes;
      LegionMap<TraceCloseOp*,LogicalUser>::aligned read_only_closes;
    protected:
      VersionInfo closed_version_info;
      FieldMask flush_only_fields;
    }; 

    /**
     * \struct EventSet 
     * A helper class for building sets of fields with 
     * a common set of preconditions for doing copies.
     */
    struct EventSet {
    public:
      EventSet(void) { }
      EventSet(const FieldMask &m)
        : set_mask(m) { }
    public:
      FieldMask set_mask;
      std::set<ApEvent> preconditions;
    }; 
    
    /**
     * \class PhysicalState
     * A physical state is a temporary buffer for holding a merged
     * group of version state objects which can then be used by 
     * physical traversal routines. Physical state objects track
     * the version state objects that they use and remove references
     * to them when they are done.
     */
    class PhysicalState {
    public:
      static const AllocationType alloc_type = PHYSICAL_STATE_ALLOC;
    public:
      PhysicalState(RegionTreeNode *node);
      PhysicalState(const PhysicalState &rhs);
      ~PhysicalState(void);
    public:
      PhysicalState& operator=(const PhysicalState &rhs);
      void* operator new(size_t count);
      void* operator new[](size_t count);
      void operator delete(void *ptr);
      void operator delete[](void *ptr);
    public:
      void add_version_state(VersionState *state, const FieldMask &mask);
      void add_advance_state(VersionState *state, const FieldMask &mask);
      void capture_state(bool path_only, bool split_node);
      void apply_path_only_state(const FieldMask &advance_mask,
            AddressSpaceID target, std::set<RtEvent> &applied_conditions) const;
      void apply_state(const FieldMask &advance_mask, 
            AddressSpaceID target, std::set<RtEvent> &applied_conditions);
      void filter_and_apply(const FieldMask &advance_mask,AddressSpaceID target,
            bool filter_masks, bool filter_views, bool filter_children,
            const LegionMap<ColorPoint,FieldMask>::aligned *closed_children,
                            std::set<RtEvent> &applied_conditions);
      void reset(void);
      void filter_open_children(const FieldMask &filter_mask);
    public:
      PhysicalState* clone(bool clone_state, bool need_advance) const;
      PhysicalState* clone(const FieldMask &clone_mask, 
                           bool clone_state, bool need_advance) const;
      void make_local(std::set<RtEvent> &preconditions, 
                      bool needs_final, bool needs_advance); 
    public:
      void print_physical_state(const FieldMask &capture_mask,
          LegionMap<ColorPoint,FieldMask>::aligned &to_traverse,
                                TreeStateLogger *logger);
    public:
      RegionTreeNode *const node;
    public:
      // Fields which have dirty data
      FieldMask dirty_mask;
      // Fields with outstanding reductions
      FieldMask reduction_mask;
      // State of any child nodes
      ChildState children;
      // The valid instance views
      LegionMap<LogicalView*, FieldMask,
                VALID_VIEW_ALLOC>::track_aligned valid_views;
      // The valid reduction veiws
      LegionMap<ReductionView*, FieldMask,
                VALID_REDUCTION_ALLOC>::track_aligned reduction_views;
    public:
      LegionMap<VersionID,VersionStateInfo>::aligned version_states;
      LegionMap<VersionID,VersionStateInfo>::aligned advance_states;
    };

    /**
     * \class VersionState
     * This class tracks the physical state information
     * for a particular version number from the persepective
     * of a given logical region.
     */
    class VersionState : public DistributedCollectable {
    public:
      static const AllocationType alloc_type = VERSION_STATE_ALLOC;
    public:
      enum VersionRequestKind {
        PATH_ONLY_VERSION_REQUEST,
        INITIAL_VERSION_REQUEST,
        FINAL_VERSION_REQUEST,
      };
      struct RequestInfo {
      public:
        AddressSpaceID target;
        RtUserEvent to_trigger;
        FieldMask request_mask;
        VersionRequestKind kind;
      };
    public:
      struct SendVersionStateArgs {
      public:
        HLRTaskID hlr_id;
        VersionState *proxy_this;
        AddressSpaceID target;
        VersionRequestKind request_kind;
        FieldMask *request_mask;
        RtUserEvent to_trigger;
      };
      struct UpdateViewReferences {
      public:
        HLRTaskID hlr_id;
        DistributedID did;
        LogicalView *view;
      };
      struct RemoveVersionStateRefArgs {
      public:
        HLRTaskID hlr_id;
        VersionState *proxy_this;
        ReferenceSource ref_kind;
      };
    public:
      VersionState(VersionID vid, Runtime *rt, DistributedID did,
                   AddressSpaceID owner_space, AddressSpaceID local_space, 
                   RegionTreeNode *node, bool register_now);
      VersionState(const VersionState &rhs);
      virtual ~VersionState(void);
    public:
      VersionState& operator=(const VersionState &rhs);
      void* operator new(size_t count);
      void* operator new[](size_t count);
      void operator delete(void *ptr);
      void operator delete[](void *ptr);
    public:
      void initialize(ApEvent term_event, const RegionUsage &usage,
                      const FieldMask &user_mask, const InstanceSet &targets,
                      SingleTask *context, unsigned init_index,
                      const std::vector<LogicalView*> &corresponding);
      void update_split_previous_state(PhysicalState *state,
                                       const FieldMask &update_mask) const;
      void update_split_advance_state(PhysicalState *state,
                                      const FieldMask &update_mask) const;
      void update_path_only_state(PhysicalState *state,
                                  const FieldMask &update_mask) const;
      void update_physical_state(PhysicalState *state, 
                                 const FieldMask &update_mask) const; 
    public: // methods for applying state information
      void merge_path_only_state(const PhysicalState *state,
                                 const FieldMask &merge_mask,
                                 AddressSpaceID target,
                                 std::set<RtEvent> &applied_conditions);
      void merge_physical_state(const PhysicalState *state, 
                                const FieldMask &merge_mask,
                                AddressSpaceID target,
                                std::set<RtEvent> &applied_conditions,
                                bool need_lock = true);
      void filter_and_merge_physical_state(const PhysicalState *state,
                                const FieldMask &merge_mask,
                                AddressSpaceID target, bool filter_masks,
                                bool filter_views, bool filter_children,
            const LegionMap<ColorPoint,FieldMask>::aligned *closed_children,
                                std::set<RtEvent> &applied_conditions);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      void request_initial_version_state(const FieldMask &request_mask,
                                         std::set<RtEvent> &preconditions);
      void request_final_version_state(const FieldMask &request_mask,
                                       std::set<RtEvent> &preconditions);
      void select_initial_targets(AddressSpaceID request_space, 
                                  FieldMask &needed_mask,
                                  LegionDeque<RequestInfo>::aligned &targets,
                                  std::set<RtEvent> &preconditions);
      void select_final_targets(AddressSpaceID request_space,
                                FieldMask &needed_mask,
                                LegionDeque<RequestInfo>::aligned &targets,
                                std::set<RtEvent> &preconditions);
    public:
      void send_version_state(AddressSpaceID target, VersionRequestKind kind,
                         const FieldMask &request_mask, RtUserEvent to_trigger);
      void send_version_state_request(AddressSpaceID target, AddressSpaceID src,
                          RtUserEvent to_trigger, const FieldMask &request_mask,
                          VersionRequestKind request_kind);
      void launch_send_version_state(AddressSpaceID target,
                                     RtUserEvent to_trigger, 
                                     VersionRequestKind request_kind,
                                     const FieldMask &request_mask, 
                                     RtEvent precondition=RtEvent::NO_RT_EVENT);
    public:
      void handle_version_state_path_only(AddressSpaceID source,
                                          FieldMask &path_only_mask);
      void handle_version_state_initialization(AddressSpaceID source,
                                               FieldMask &initial_mask);
      void handle_version_state_request(AddressSpaceID source, 
                                        RtUserEvent to_trigger, 
                                        VersionRequestKind request_kind,
                                        FieldMask &request_mask);
      void handle_version_state_response(AddressSpaceID source,
          RtUserEvent to_trigger, VersionRequestKind kind, Deserializer &derez);
    public:
      static void process_view_references(const void *args);
    public:
      static void process_version_state_path_only(Runtime *rt,
                              Deserializer &derez, AddressSpaceID source);
      static void process_version_state_initialization(Runtime *rt,
                              Deserializer &derez, AddressSpaceID source);
      static void process_version_state_request(Runtime *rt, 
                                                Deserializer &derez);
      static void process_version_state_response(Runtime *rt,
                              Deserializer &derez, AddressSpaceID source);
    public:
      void remove_version_state_ref(ReferenceSource ref_kind, 
                                    RtEvent done_event);
      static void process_remove_version_state_ref(const void *args);
    public:
      const VersionID version_number;
      RegionTreeNode *const logical_node;
    protected:
      Reservation state_lock;
      // Fields which have been directly written to
      FieldMask dirty_mask;
      // Fields which have reductions
      FieldMask reduction_mask;
      // State of any child nodes
      ChildState children;
      // The valid instance views
      LegionMap<LogicalView*, FieldMask,
                VALID_VIEW_ALLOC>::track_aligned valid_views;
      // The valid reduction veiws
      LegionMap<ReductionView*, FieldMask,
                VALID_REDUCTION_ALLOC>::track_aligned reduction_views;
#ifdef DEBUG_LEGION
      // Track our current state 
      bool currently_active;
      bool currently_valid;
#endif
    protected:
      // Fields which are in the path only state
      FieldMask path_only_fields;
      // Fields which are in the initial state
      FieldMask initial_fields;
      // Fields which are in the final state
      FieldMask final_fields;
      // Initial ready events
      LegionMap<RtEvent,FieldMask>::aligned initial_events;
      LegionMap<RtEvent,FieldMask>::aligned final_events;
      // These are valid on the owner node only
      LegionMap<AddressSpaceID,FieldMask>::aligned path_only_nodes;
      LegionMap<AddressSpaceID,FieldMask>::aligned initial_nodes;
      LegionMap<AddressSpaceID,FieldMask>::aligned final_nodes;
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
      void register_child(unsigned depth, const ColorPoint &color);
      void clear();
    public:
#ifdef DEBUG_LEGION 
      bool has_child(unsigned depth) const;
      const ColorPoint& get_child(unsigned depth) const;
#else
      inline bool has_child(unsigned depth) const
        { return path[depth].is_valid(); }
      inline const ColorPoint& get_child(unsigned depth) const
        { return path[depth]; }
#endif
      inline unsigned get_path_length(void) const
        { return ((max_depth-min_depth)+1); }
      inline unsigned get_min_depth(void) const { return min_depth; }
      inline unsigned get_max_depth(void) const { return max_depth; }
    protected:
      std::vector<ColorPoint> path;
      unsigned min_depth;
      unsigned max_depth;
    };

    /**
     * \class FatTreePath
     * A data structure for representing many different
     * paths through a region tree.
     */
    class FatTreePath {
    public:
      FatTreePath(void);
      FatTreePath(const FatTreePath &rhs);
      ~FatTreePath(void);
    public:
      FatTreePath& operator=(const FatTreePath &rhs);
    public:
      inline const std::map<ColorPoint,FatTreePath*>& get_children(void) const
        { return children; }
      void add_child(const ColorPoint &child_color, FatTreePath *child);
      bool add_child(const ColorPoint &child_color, FatTreePath *child,
                     IndexTreeNode *index_tree_node);
    protected:
      std::map<ColorPoint,FatTreePath*> children;
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
      ColorPoint next_child;
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
      CurrentInvalidator(ContextID ctx, bool logical_users_only);
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
      const bool logical_users_only;
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
     * \class ReductionCloser
     * A class for performing reduciton close operations
     */
    class ReductionCloser {
    public:
      ReductionCloser(ContextID ctx, ReductionView *target,
                      const FieldMask &reduc_mask, 
                      VersionInfo &version_info, 
                      Operation *op, unsigned index,
                      std::set<RtEvent> &map_applied_events);
      ReductionCloser(const ReductionCloser &rhs);
      ~ReductionCloser(void);
    public:
      ReductionCloser& operator=(const ReductionCloser &rhs);
      void issue_close_reductions(RegionTreeNode *node, PhysicalState *state);
    public:
      const ContextID ctx;
      ReductionView *const target;
      const FieldMask close_mask;
      VersionInfo &version_info;
      Operation *const op;
      const unsigned index;
      std::set<RtEvent> &map_applied_events;
    protected:
      std::set<ReductionView*> issued_reductions;
    };

    /**
     * \class InstanceRef
     * A class for keeping track of references to physical instances
     */
    class InstanceRef {
    public:
      struct DeferCompositeHandleArgs {
      public:
        HLRTaskID hlr_id;
        CompositeView *view;
      };
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
      inline bool has_ref(void) const 
      {
#ifdef DEBUG_LEGION
        assert(!composite);
#endif
        return (ptr.manager != NULL);
      }
      inline ApEvent get_ready_event(void) const { return ready_event; }
      inline void set_ready_event(ApEvent ready) { ready_event = ready; }
      inline PhysicalManager* get_manager(void) const 
      { 
#ifdef DEBUG_LEGION
        assert(!composite);
#endif
        return ptr.manager; 
      }
      inline const FieldMask& get_valid_fields(void) const 
        { return valid_fields; }
    public:
      inline bool is_composite_ref(void) const { return composite; }
      inline bool is_local(void) const { return local; }
      void set_composite_view(CompositeView *view, ReferenceMutator *mutator);
      CompositeView* get_composite_view(void) const;
      MappingInstance get_mapping_instance(void) const;
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
      void pack_reference(Serializer &rez, AddressSpaceID target);
      void unpack_reference(Runtime *rt, TaskOp *task,
                            Deserializer &derez, RtEvent &ready);
      static void handle_deferred_composite_handle(const void *args);
    protected:
      FieldMask valid_fields; 
      ApEvent ready_event;
      union {
        PhysicalManager *manager;
        CompositeView *view;
      } ptr;
      bool composite;
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
    public:
      bool has_composite_ref(void) const;
      const InstanceRef& get_composite_ref(void) const;
    public:
      void pack_references(Serializer &rez, AddressSpaceID target) const;
      void unpack_references(Runtime *runtime, TaskOp *task,
          Deserializer &derez, std::set<RtEvent> &ready_events);
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
     * \class PhysicalTraverser
     * A class for traversing the physical region tree to open up
     * sub-trees and find valid instances for a given region requirement
     */
    class PhysicalTraverser : public PathTraverser {
    public:
      PhysicalTraverser(RegionTreePath &path, TraversalInfo *info, 
                        InstanceSet *targets);
      PhysicalTraverser(const PhysicalTraverser &rhs);
      ~PhysicalTraverser(void);
    public:
      PhysicalTraverser& operator=(const PhysicalTraverser &rhs);
    public:
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      bool traverse_node(RegionTreeNode *node);
    protected:
      TraversalInfo *const info;
      InstanceSet *const targets;
    };

    /**
     * \struct CompositeCloser
     * Class for helping with closing of physical trees to composite instances
     */
    class CompositeCloser {
    public:
      CompositeCloser(ContextID ctx, 
                      VersionInfo &version_info, SingleTask *target_ctx);
      CompositeCloser(const CompositeCloser &rhs);
      ~CompositeCloser(void);
    public:
      CompositeCloser& operator=(const CompositeCloser &rhs);
    public:
      CompositeNode* get_composite_node(RegionTreeNode *tree_node,
                                        bool root = false);
      CompositeView* create_valid_view(PhysicalState *state,
                                      CompositeNode *root,
                                      const FieldMask &valid_mask);
      void capture_physical_state(CompositeNode *target,
                                  RegionTreeNode *node,
                                  PhysicalState *state,
                                  const FieldMask &close_mask,
                                  const FieldMask &dirty_mask,
                                  const FieldMask &reduc_mask);
    public:
      void update_capture_mask(RegionTreeNode *node,
                               const FieldMask &capture_mask);
      bool filter_capture_mask(RegionTreeNode *node,
                               FieldMask &capture_mask);
    public:
      const ContextID ctx;
      VersionInfo &version_info;
      SingleTask *const target_ctx;
    public:
      std::map<RegionTreeNode*,CompositeNode*> constructed_nodes;
      LegionMap<RegionTreeNode*,FieldMask>::aligned capture_fields;
      LegionMap<ReductionView*,FieldMask>::aligned reduction_views;
    };

    /**
     * \class RestrictInfo
     * A class for tracking mapping restrictions based 
     * on region usage.
     */
    class RestrictInfo {
    public:
      struct DeferRestrictedManagerArgs {
      public:
        HLRTaskID hlr_id;
        InstanceManager *manager;
      };
    public:
      RestrictInfo(void);
      RestrictInfo(const RestrictInfo &rhs); 
      ~RestrictInfo(void);
    public:
      RestrictInfo& operator=(const RestrictInfo &rhs);
    public:
      inline bool has_restrictions(void) const 
        { return !restrictions.empty(); }
    public:
      void record_restriction(InstanceManager *inst, const FieldMask &mask);
      void populate_restrict_fields(FieldMask &to_fill) const;
      void clear(void);
      const InstanceSet& get_instances(void);
    public:
      void pack_info(Serializer &rez);
      void unpack_info(Deserializer &derez, Runtime *runtime,
                       std::set<RtEvent> &ready_events);
      static void handle_deferred_reference(const void *args);
    protected:
      // We only need garbage collection references on these
      // instances because we know one of two things is always
      // true: either they are attached files so they aren't 
      // subject to memories in which garbage collection will
      // occur, or they are simultaneous restricted, so that
      // enclosing context of the parent task has a valid 
      // reference to them so there is no need for us to 
      // have a valid reference.
      // // Same in Restriction
      LegionMap<InstanceManager*,FieldMask>::aligned restrictions;
    protected:
      InstanceSet restricted_instances;
    };

  }; // namespace Internal 
}; // namespace Legion

#endif // __LEGION_ANALYSIS_H__
