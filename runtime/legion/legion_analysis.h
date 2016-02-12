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
        static const unsigned BASE_FIELDS_MASK = 0x7;
        inline void set_path_only(void) { bit_mask |= 1; }
        inline void set_needs_final(void) { bit_mask |= 2; }
        inline void set_close_top(void) { bit_mask |= 4; }
        inline void set_needs_capture(void) { bit_mask |= 8; }
        inline void unset_needs_capture(void)
        { if (needs_capture()) bit_mask -= 8; }
      public:
        inline bool path_only(void) const { return (1 & bit_mask); }
        inline bool needs_final(void) const { return (2 & bit_mask); }
        inline bool close_top(void) const { return (4 & bit_mask); }
        inline bool needs_capture(void) const { return (8 & bit_mask); }
      public:
        PhysicalState *physical_state;
        FieldVersions *field_versions;
        // For nodes in close operations that are not the top node
        // this mask doubles as the set of fields which need to be 
        // applied because the close is leave open. Unfortunately
        // we can't put this in a union to make this more clear
        // because C unions are dumb.
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
                         std::set<Event> &applied_conditions);
      void apply_close(ContextID ctx, AddressSpaceID target,
             const LegionMap<ColorPoint,FieldMask>::aligned &closed_children,
                       std::set<Event> &applied_conditions); 
      void reset(void);
      void release(void);
      void clear(void);
      void recapture_state(void);
      void sanity_check(RegionTreeNode *node);
    public:
      PhysicalState* find_physical_state(RegionTreeNode *node, bool capture); 
      FieldVersions* get_versions(RegionTreeNode *node) const;
    public:
      void pack_version_info(Serializer &rez, AddressSpaceID local_space,
                             ContextID ctx);
      void unpack_version_info(Deserializer &derez);
      void make_local(std::set<Event> &preconditions,
                      RegionTreeForest *forest, ContextID ctx);
      void clone_version_info(RegionTreeForest *forest, LogicalRegion handle,
                              const VersionInfo &rhs, bool check_below);
      void clone_from(const VersionInfo &rhs);
      void clone_from(const VersionInfo &rhs, CompositeCloser &closer);
    protected:
      void pack_buffer(Serializer &rez, 
                       AddressSpaceID local_space, ContextID ctx);
      void unpack_buffer(RegionTreeForest *forest, ContextID ctx);
      void pack_node_info(Serializer &rez, NodeInfo &info,
                          RegionTreeNode *node, ContextID ctx);
      void unpack_node_info(RegionTreeNode *node, ContextID ctx,
                            Deserializer &derez, AddressSpaceID source);
    protected:
      LegionMap<RegionTreeNode*,NodeInfo>::aligned node_infos;
      RegionTreeNode *upper_bound_node;
    protected:
      bool packed;
      void *packed_buffer;
      size_t packed_size;
    };

    /**
     * \class RestrictInfo
     * A class for tracking mapping restrictions based 
     * on region usage.
     */
    class RestrictInfo {
    public:
      RestrictInfo(void);
      RestrictInfo(const RestrictInfo &rhs); 
      ~RestrictInfo(void);
    public:
      RestrictInfo& operator=(const RestrictInfo &rhs);
    public:
      inline bool needs_check(void) const { return perform_check; }
      inline void set_check(void) { perform_check = true; } 
      inline void add_restriction(LogicalRegion handle, const FieldMask &mask)
      {
        LegionMap<LogicalRegion,FieldMask>::aligned::iterator finder = 
          restrictions.find(handle);
        if (finder == restrictions.end())
          restrictions[handle] = mask;
        else
          finder->second |= mask;
      }
      inline bool has_restrictions(void) const { return !restrictions.empty(); }
      bool has_restrictions(LogicalRegion handle, RegionNode *node,
                            const std::set<FieldID> &fields) const;
      inline void clear(void)
      {
        perform_check = false;
        restrictions.clear();
      }
      inline void merge(const RestrictInfo &rhs, const FieldMask &mask)
      {
        perform_check = rhs.perform_check;
        for (LegionMap<LogicalRegion,FieldMask>::aligned::const_iterator it = 
              rhs.restrictions.begin(); it != rhs.restrictions.end(); it++)
        {
          FieldMask overlap = it->second & mask;
          if (!overlap)
            continue;
          restrictions[it->first] = overlap;
        }
      }
    public:
      void pack_info(Serializer &rez);
      void unpack_info(Deserializer &derez, AddressSpaceID source, 
                       RegionTreeForest *forest);
    protected:
      bool perform_check;
      LegionMap<LogicalRegion,FieldMask>::aligned restrictions;
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
      PhysicalUser(const RegionUsage &u, const ColorPoint &child,
                   FieldVersions *versions = NULL);
      PhysicalUser(const PhysicalUser &rhs);
      ~PhysicalUser(void);
    public:
      PhysicalUser& operator=(const PhysicalUser &rhs);
    public:
      bool same_versions(const FieldMask &test_mask, 
                         const FieldVersions *other) const;
    public:
      void pack_user(Serializer &rez);
      static PhysicalUser* unpack_user(Deserializer &derez, 
                                       FieldSpaceNode *node,
                                       AddressSpaceID source,
                                       bool add_reference);
    public:
      RegionUsage usage;
      ColorPoint child;
      FieldVersions *const versions;
    }; 

    /**
     * \struct MappableInfo
     */
    struct MappableInfo {
    public:
      MappableInfo(ContextID ctx, Operation *op,
                   Processor local_proc, RegionRequirement &req,
                   VersionInfo &version_info,
                   const FieldMask &traversal_mask);
    public:
      const ContextID ctx;
      Operation *const op;
      const Processor local_proc;
      RegionRequirement &req;
      VersionInfo &version_info;
      const FieldMask traversal_mask;
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
      FieldMask valid_fields;
      LegionMap<VersionState*,FieldMask>::aligned states;
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
      void sanity_check(void);
    public:
      void initialize_state(LogicalView *view, Event term_event,
                            const RegionUsage &usage,
                            const FieldMask &user_mask);
      void record_version_numbers(const FieldMask &mask,
                                  const LogicalUser &user,
                                  VersionInfo &version_info,
                                  bool capture_previous, bool path_only,
                                  bool need_final, bool close_top, 
                                  bool report_unversioned,
                                  bool capture_leave_open = false);
      void advance_version_numbers(const FieldMask &mask);
    public:
      VersionState* create_new_version_state(VersionID vid); 
      VersionState* create_remote_version_state(VersionID vid, 
                              DistributedID did, AddressSpaceID owner_space);
      VersionState* find_remote_version_state(VersionID vid, DistributedID did,
                                              AddressSpaceID owner_space);
    public:
      inline bool has_persistent_views(void) const { return has_persistent; }
      void add_persistent_view(MaterializedView *view);
      void remove_persistent_view(MaterializedView *view);
      void capture_persistent_views(
                            LegionMap<LogicalView*,FieldMask>::aligned &views,
                                    const FieldMask &capture_mask);
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
      // Fields on which the user has 
      // asked for explicit coherence
      FieldMask restricted_fields;
    protected:
      Reservation state_lock; 
    protected:
      bool has_persistent;
      std::set<MaterializedView*> persistent_views;
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
      inline bool has_closed_fields(void) const { return !!closed_mask; }
      void record_closed_child(const ColorPoint &child, const FieldMask &mask,
                               bool leave_open, bool read_only_close);
      void record_partial_fields(const FieldMask &skipped_fields);
      void record_flush_only_fields(const FieldMask &flush_only);
      void initialize_close_operations(RegionTreeNode *target, 
                                       Operation *creator,
                                       const VersionInfo &version_info,
                                       const RestrictInfo &restrict_info,
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
                          const RestrictInfo &restrict_info, 
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

    struct CopyTracker {
    public:
      CopyTracker(void);
    public:
      inline void add_copy_event(Event e) { copy_events.insert(e); } 
      Event get_termination_event(void) const;
    protected:
      std::set<Event> copy_events;
    };

    /**
     * \struct PhysicalCloser
     * Class for helping with the closing of physical region trees
     */
    class PhysicalCloser : public CopyTracker {
    public:
      PhysicalCloser(const MappableInfo &info,
                     LogicalRegion closing_handle);
      PhysicalCloser(const PhysicalCloser &rhs);
      ~PhysicalCloser(void);
    public:
      PhysicalCloser& operator=(const PhysicalCloser &rhs);
    public:
      bool needs_targets(void) const;
      void add_target(MaterializedView *target);
      void close_tree_node(RegionTreeNode *node, 
                           const FieldMask &closing_mask);
      const std::vector<MaterializedView*>& get_upper_targets(void) const;
      const std::vector<MaterializedView*>& get_lower_targets(void) const;
    public:
      void update_dirty_mask(const FieldMask &mask);
      const FieldMask& get_dirty_mask(void) const;
      void update_node_views(RegionTreeNode *node, PhysicalState *state);
    public:
      inline void set_leave_open_mask(const FieldMask &leave_open)
        { leave_open_mask = leave_open; }
      inline const FieldMask& get_leave_open_mask(void) const 
        { return leave_open_mask; }
    public:
      const MappableInfo &info;
      const LogicalRegion handle;
    protected:
      FieldMask leave_open_mask;
    protected:
      bool targets_selected;
      FieldMask dirty_mask;
      std::vector<MaterializedView*> upper_targets;
      std::vector<MaterializedView*> lower_targets;
      std::set<Event> close_events;
    }; 

    /**
     * \struct CompositeCloser
     * Class for helping with closing of physical trees to composite instances
     */
    class CompositeCloser {
    public:
      CompositeCloser(ContextID ctx, VersionInfo &version_info);
      CompositeCloser(const CompositeCloser &rhs);
      ~CompositeCloser(void);
    public:
      CompositeCloser& operator=(const CompositeCloser &rhs);
    public:
      CompositeNode* get_composite_node(RegionTreeNode *tree_node,
                                        CompositeNode *parent);
      CompositeRef create_valid_view(PhysicalState *state,
                                     CompositeNode *root,
                                     const FieldMask &closed_mask,
                                     bool register_view);
      void capture_physical_state(CompositeNode *target,
                                  RegionTreeNode *node,
                                  PhysicalState *state,
                                  const FieldMask &capture_mask,
                                  FieldMask &dirty_mask);
      void update_capture_mask(RegionTreeNode *node,
                               const FieldMask &capture_mask);
      bool filter_capture_mask(RegionTreeNode *node,
                               FieldMask &capture_mask);
    public:
      inline void set_leave_open_mask(const FieldMask &leave_open)
        { leave_open_mask = leave_open; }
    public:
      const ContextID ctx;
      VersionInfo &version_info;
    public:
      FieldMask leave_open_mask;
    public:
      CompositeVersionInfo *composite_version_info;
      std::map<RegionTreeNode*,CompositeNode*> constructed_nodes;
      LegionMap<RegionTreeNode*,FieldMask>::aligned capture_fields;
      LegionMap<ReductionView*,FieldMask>::aligned reduction_views;
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
      std::set<Event> preconditions;
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
      PhysicalState(CurrentState *manager);
#ifdef DEBUG_HIGH_LEVEL
      PhysicalState(CurrentState *manager, RegionTreeNode *node);
#endif
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
      void capture_state(bool path_only, bool close_top);
      void apply_path_only_state(const FieldMask &advance_mask,
            AddressSpaceID target, std::set<Event> &applied_conditions) const;
      void apply_state(const FieldMask &advance_mask, 
            AddressSpaceID target, std::set<Event> &applied_conditions);
      void filter_and_apply(bool top, AddressSpaceID target,
            const LegionMap<ColorPoint,FieldMask>::aligned &closed_children,
                            std::set<Event> &applied_conditions);
      void release_created_instances(void);
      void reset(void);
      void record_created_instance(InstanceView *view, bool remote);
      void filter_open_children(const FieldMask &filter_mask);
    public:
      PhysicalState* clone(bool clone_state, bool need_advance) const;
      PhysicalState* clone(const FieldMask &clone_mask, 
                           bool clone_state, bool need_advance) const;
      void make_local(std::set<Event> &preconditions, 
                      bool needs_final, bool needs_advance); 
    public:
      void print_physical_state(const FieldMask &capture_mask,
          LegionMap<ColorPoint,FieldMask>::aligned &to_traverse,
                                TreeStateLogger *logger);
    public:
      CurrentState *const manager;
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
      // Any instance views which we created and are therefore holding
      // additional valid references that will need to be removed
      // after our updates have been applied
      std::deque<std::pair<InstanceView*,bool/*remote*/> > created_instances;
    public:
      LegionMap<VersionID,VersionStateInfo>::aligned version_states;
      LegionMap<VersionID,VersionStateInfo>::aligned advance_states;
#ifdef DEBUG_HIGH_LEVEL
    public:
      RegionTreeNode *const node;
#endif
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
        UserEvent to_trigger;
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
        UserEvent to_trigger;
      };
    public:
      VersionState(VersionID vid, Runtime *rt, DistributedID did,
                   AddressSpaceID owner_space, AddressSpaceID local_space, 
                   CurrentState *manager); 
      VersionState(const VersionState &rhs);
      virtual ~VersionState(void);
    public:
      VersionState& operator=(const VersionState &rhs);
      void* operator new(size_t count);
      void* operator new[](size_t count);
      void operator delete(void *ptr);
      void operator delete[](void *ptr);
    public:
      void initialize(LogicalView *view, Event term_event,
                      const RegionUsage &usage,
                      const FieldMask &user_mask);
      void update_close_top_state(PhysicalState *state,
                                  const FieldMask &update_mask) const;
      void update_open_children_state(PhysicalState *state,
                                      const FieldMask &update_mask) const;
      void update_path_only_state(PhysicalState *state,
                                  const FieldMask &update_mask) const;
      void update_physical_state(PhysicalState *state, 
                                 const FieldMask &update_mask) const; 
      void merge_path_only_state(const PhysicalState *state,
                                 const FieldMask &merge_mask,
                                 AddressSpaceID target,
                                 std::set<Event> &applied_conditions);
      void merge_physical_state(const PhysicalState *state, 
                                const FieldMask &merge_mask,
                                AddressSpaceID target,
                                std::set<Event> &applied_conditions,
                                bool need_lock = true);
      void filter_and_merge_physical_state(const PhysicalState *state,
                                const FieldMask &merge_mask, bool top,
                                AddressSpaceID target,
            const LegionMap<ColorPoint,FieldMask>::aligned &closed_children,
                                std::set<Event> &applied_conditions);
    public:
      virtual void notify_active(void);
      virtual void notify_inactive(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
    public:
      void request_initial_version_state(const FieldMask &request_mask,
                                         std::set<Event> &preconditions);
      void request_final_version_state(const FieldMask &request_mask,
                                       std::set<Event> &preconditions);
      void select_initial_targets(AddressSpaceID request_space, 
                                  FieldMask &needed_mask,
                                  LegionDeque<RequestInfo>::aligned &targets,
                                  std::set<Event> &preconditions);
      void select_final_targets(AddressSpaceID request_space,
                                FieldMask &needed_mask,
                                LegionDeque<RequestInfo>::aligned &targets,
                                std::set<Event> &preconditions);
    public:
      void send_version_state(AddressSpaceID target, VersionRequestKind kind,
                           const FieldMask &request_mask, UserEvent to_trigger);
      void send_version_state_request(AddressSpaceID target, AddressSpaceID src,
                            UserEvent to_trigger, const FieldMask &request_mask,
                            VersionRequestKind request_kind);
      void launch_send_version_state(AddressSpaceID target,
                                     UserEvent to_trigger, 
                                     VersionRequestKind request_kind,
                                     const FieldMask &request_mask, 
                                     Event precondition = Event::NO_EVENT);
    public:
      void handle_version_state_path_only(AddressSpaceID source,
                                          FieldMask &path_only_mask);
      void handle_version_state_initialization(AddressSpaceID source,
                                               FieldMask &initial_mask);
      void handle_version_state_request(AddressSpaceID source, 
                                        UserEvent to_trigger, 
                                        VersionRequestKind request_kind,
                                        FieldMask &request_mask);
      void handle_version_state_response(AddressSpaceID source,
            UserEvent to_trigger, VersionRequestKind kind, Deserializer &derez);
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
      const VersionID version_number;
      CurrentState *const manager;
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
#ifdef DEBUG_HIGH_LEVEL
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
      LegionMap<Event,FieldMask>::aligned initial_events;
      LegionMap<Event,FieldMask>::aligned final_events;
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
      bool has_child(unsigned depth) const;
      const ColorPoint& get_child(unsigned depth) const;
      unsigned get_path_length(void) const;
    public:
      InstanceRef translate_ref(const InstanceRef &ref) const;
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
     * \class RestrictionMutator
     * A class for mutating the state of restrction fields
     */
    class RestrictionMutator : public NodeTraverser {
    public:
      RestrictionMutator(ContextID ctx, const FieldMask &mask,
                         bool add_restrict);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
      const FieldMask &restrict_mask;
      const bool add_restrict;
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
                      Processor local_proc, Operation *op);
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
      const Processor local_proc;
      Operation *const op;
    protected:
      std::set<ReductionView*> issued_reductions;
    };

    /**
     * \class MappingRef
     * This class keeps a valid reference to a physical instance that has
     * been allocated and is ready to have dependence analysis performed.
     * Once all the allocations have been performed, then an operation
     * can pass all of the mapping references to the RegionTreeForest
     * to actually perform the operations necessary to make the 
     * region valid and return an InstanceRef.
     */
    class MappingRef {
    public:
      MappingRef(void);
      MappingRef(LogicalView *view, const FieldMask &needed_mask);
      MappingRef(const MappingRef &rhs);
      ~MappingRef(void);
    public:
      MappingRef& operator=(const MappingRef &rhs);
    public:
      inline bool has_ref(void) const { return (view != NULL); }
      inline LogicalView* get_view(void) const { return view; } 
      inline const FieldMask& get_mask(void) const { return needed_fields; }
    private:
      LogicalView *view;
      FieldMask needed_fields;
    };

    /**
     * \class InstanceRef
     * A class for keeping track of references to physical instances
     */
    class InstanceRef {
    public:
      InstanceRef(void);
      InstanceRef(Event ready, InstanceView *view);
      InstanceRef(Event ready, InstanceView *view,
                  const std::vector<Reservation> &locks);
    public:
      bool operator==(const InstanceRef &rhs) const;
      bool operator!=(const InstanceRef &rhs) const;
    public:
      inline bool has_ref(void) const { return (manager != NULL); }
      inline bool has_required_locks(void) const 
                                      { return !needed_locks.empty(); }
      inline Event get_ready_event(void) const { return ready_event; }
      inline void add_reservation(Reservation handle) 
                                  { needed_locks.push_back(handle); }
      inline InstanceManager* get_manager(void) const { return manager; }
      inline InstanceView* get_instance_view(void) const { return view; }
    public:
      bool is_composite_ref(void) const;
      MappingInstance get_mapping_instance(void) const;
    public:
      // These methods are used by PhysicalRegion::Impl to hold
      // valid references to avoid premature collection
      void add_valid_reference(ReferenceSource source) const;
      void remove_valid_reference(ReferenceSource source) const;
    public:
      MaterializedView* get_materialized_view(void) const;
      ReductionView* get_reduction_view(void) const;
    public:
      void update_atomic_locks(std::map<Reservation,bool> &atomic_locks,
                               bool exclusive);
      Memory get_memory(void) const;
      LegionRuntime::Accessor::RegionAccessor<
          LegionRuntime::Accessor::AccessorType::Generic>
            get_accessor(void) const;
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const;
    public:
      void pack_reference(Serializer &rez, AddressSpaceID target);
      void unpack_reference(Runtime *rt, Deserializer &derez);
    private:
      Event ready_event;
      InstanceView *view; // only valid on creation node
      InstanceManager *manager;
      std::vector<Reservation> needed_locks;
    };

    class CompositeRef {
    public:
      CompositeRef(void);
      CompositeRef(CompositeView *view);
      CompositeRef(const CompositeRef &rhs);
      ~CompositeRef(void);
    public:
      CompositeRef& operator=(const CompositeRef &rhs);
    public:
      inline bool has_ref(void) const { return (view != NULL); }
      inline bool is_local(void) const { return local; }
      CompositeView* get_view(void) const { return view; }
    public:
      void pack_reference(Serializer &rez, AddressSpaceID target);
      void unpack_reference(Runtime *rt, Deserializer &derez);
    private:
      CompositeView *view;
      bool local;
    };

    class InstanceSet {
    public:
      class InternalSet : public Collectable {
      public:
        InternalSet(

      };
    public:
      InstanceSet(size_t init_size = 0);
      InstanceSet(InstanceSet &rhs);
      ~InstanceSet(void);
    public:
      InstanceSet& operator=(InstanceSet &rhs);
      bool operator==(const InstanceSet &rhs) const;
      bool operator!=(const InstanceSet &rhs) const;
    public:
      InstanceRef& operator[](unsigned idx);
      const InstanceRef& operator[](unsigned idx) const;
    public:
      bool empty(void) const;
      size_t size(void) const;
      void clear(void);
      void reserve(size_t size);
      void add_instance(const InstanceRef &ref);
    public:
      bool has_composite_ref(void) const;
      const CompositeRef& get_composite_ref(void) const;
    public:
      void pack_references(Serializer &rez, AddressSpaceID target) const;
      void unpack_references(Runtime *runtime, Deserializer &derez);
    public:
      void add_valid_references(ReferenceSource source);
      void remove_valid_references(ReferenceSource source);
    public:
      void update_wait_on_events(std::set<Event> &wait_on_events) const;
      bool has_required_locks(void) const;
      void update_atomic_locks(std::map<Reservation,bool> &lks,bool excl) const;
    public:
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(RegionTreeForest *forest, FieldID fid) const;
    protected:
      union {
        InstanceRef*                        single;
        LegionVector<InstanceRef>::aligned*  multi;
      } refs;
      size_t size;
      bool shared;
    };

    /**
     * \class PremapTraverser
     * A traverser of the physical region tree for
     * performing the premap operation.
     * Keep track of the last node we visited
     */
    class PremapTraverser : public PathTraverser {
    public:
      PremapTraverser(RegionTreePath &path, const MappableInfo &info);  
      PremapTraverser(const PremapTraverser &rhs); 
      ~PremapTraverser(void);
    public:
      PremapTraverser& operator=(const PremapTraverser &rhs);
    public:
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      bool premap_node(RegionTreeNode *node, LogicalRegion closing_handle);
    protected:
      const MappableInfo &info;
    };

    /**
     * \class MappingTraverser
     * A traverser of the physical region tree for
     * performing the mapping operation.
     */
    class MappingTraverser : public PathTraverser {
    public:
      MappingTraverser(RegionTreePath &path, const MappableInfo &info,
                       const RegionUsage &u, const FieldMask &m,
                       Processor proc, unsigned idx, 
                       InstanceView *target = NULL/*for restricted*/);
      MappingTraverser(const MappingTraverser &rhs);
      ~MappingTraverser(void);
    public:
      MappingTraverser& operator=(const MappingTraverser &rhs);
    public:
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    public:
      const MappingRef& get_instance_ref(void) const;
    protected:
      void traverse_node(RegionTreeNode *node);
      bool map_physical_region(RegionNode *node);
      bool map_reduction_region(RegionNode *node);
    public:
      const MappableInfo &info;
      const RegionUsage usage;
      const FieldMask user_mask;
      const Processor target_proc;
      const unsigned index;
    protected:
      MappingRef result;
      LogicalView *target;
    };

  }; // namespace Internal 
}; // namespace Legion

#endif // __LEGION_ANALYSIS_H__
