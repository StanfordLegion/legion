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
     * \struct VersioningSet
     * A small helper class for tracking collections of 
     * version state objects and their sets of fields
     * This is the same as the above class, but specialized
     * for VersionState objects explicitly
     */
    template<ReferenceSource REF_SRC = LAST_SOURCE_REF>
    class VersioningSet {
    public:
      class iterator : public std::iterator<std::input_iterator_tag,
                              std::pair<VersionState*,FieldMask> > {
      public:
        iterator(const VersioningSet *_set, 
            std::pair<VersionState*,FieldMask> *_result, bool _single)
          : set(_set), result(_result), single(_single) { }
      public:
        iterator(const iterator &rhs)
          : set(rhs.set), result(rhs.result), single(rhs.single) { }
        ~iterator(void) { }
      public:
        inline iterator& operator=(const iterator &rhs)
          { set = rhs.set; result = rhs.result; 
            single = rhs.single; return *this; }
      public:
        inline bool operator==(const iterator &rhs) const
          { return (set == rhs.set) && (result == rhs.result) && 
                    (single == rhs.single); }
        inline bool operator!=(const iterator &rhs) const
          { return (set != rhs.set) || (result != rhs.result) || 
                    (single != rhs.single); }
      public:
        inline std::pair<VersionState*,FieldMask> operator*(void) 
          { return *result; }
        inline std::pair<VersionState*,FieldMask>* operator->(void)
          { return result; }
        inline iterator& operator++(/*prefix*/void)
          { if (single) result = NULL; 
            else result = set->next(result->first); return *this; }
        inline iterator operator++(/*postfix*/int)
          { iterator copy(*this); 
            if (single) result = NULL; 
            else result = set->next(result->first); return copy; }
      public:
        inline operator bool(void) const
          { return (result != NULL); }
      private:
        const VersioningSet *set;
        std::pair<VersionState*,FieldMask> *result;
        bool single;
      };
    public:
      VersioningSet(void);
      VersioningSet(const VersioningSet &rhs);
      ~VersioningSet(void);
    public:
      VersioningSet& operator=(const VersioningSet &rhs);
    public:
      inline bool empty(void) const 
        { return single && (versions.single_version == NULL); }
      inline const FieldMask& get_valid_mask(void) const 
        { return valid_fields; }
    public:
      FieldMask& operator[](VersionState *state);
      const FieldMask& operator[](VersionState *state) const;
    public:
      void insert(VersionState *state, const FieldMask &mask, 
                  ReferenceMutator *mutator = NULL); 
      void erase(VersionState *to_erase);
      void clear(void);
      size_t size(void) const;
    public:
      std::pair<VersionState*,FieldMask>* next(VersionState *current) const;
    public:
      void move(VersioningSet &other);
    public:
      iterator begin(void) const;
      inline iterator end(void) const { return iterator(this, NULL, true); }
    public:
      template<ReferenceSource ARG_KIND>
      void reduce(const FieldMask &reduce_mask, 
                  VersioningSet<ARG_KIND> &new_states,
                  ReferenceMutator *mutator);
#ifdef DEBUG_LEGION
      void sanity_check(void) const;
#endif
    protected:
      // Fun with C, keep these two fields first and in this order
      // so that a 
      // VersioningSet of size 1 looks the same as an entry
      // in the STL Map in the multi-version case, 
      // provides goodness for the iterator
      union {
        VersionState *single_version;
        LegionMap<VersionState*,FieldMask>::aligned *multi_versions;
      } versions;
      FieldMask valid_fields;
      bool single;
    };

    /**
     * \class VersionInfo
     * A class for tracking version information about region usage
     */
    class VersionInfo {
    public:
      VersionInfo(void);
      VersionInfo(const VersionInfo &rhs);
      ~VersionInfo(void);
    public:
      VersionInfo& operator=(const VersionInfo &rhs);
    public:
      void record_split_fields(RegionTreeNode *node, const FieldMask &split);
      void add_current_version(VersionState *state, 
                               const FieldMask &state_mask, bool path_only);
      void add_advance_version(VersionState *state, 
                               const FieldMask &state_mask, bool path_only);
      void add_field_versions(unsigned depth, FieldVersions* versions);
    public:
      inline bool is_upper_bound_node(RegionTreeNode *node) const
        { return (node == upper_bound_node); }
      inline RegionTreeNode* get_upper_bound_node(void) const 
        { return upper_bound_node; }
      void set_upper_bound_node(RegionTreeNode *node);
      inline bool has_physical_states(void) const 
        { return !physical_states.empty(); }
    public:
      // The copy through parameter is useful for mis-speculated
      // operations that still need to copy state from one 
      // version number to the next even though they didn't
      // modify the physical state object
      void apply_mapping(AddressSpaceID target,
                         std::set<RtEvent> &applied_conditions,
                         bool copy_through = false);
      void clear(void);
      void sanity_check(unsigned depth);
    public:
      PhysicalState* find_physical_state(RegionTreeNode *node); 
      FieldVersions* get_versions(RegionTreeNode *node) const;
      const FieldMask& get_split_mask(unsigned depth) const;
      const FieldMask& get_split_mask(RegionTreeNode *node, 
                                      bool &is_split) const;
    public:
      void pack_version_info(Serializer &rez);
      void unpack_version_info(Deserializer &derez, RegionTreeForest *forest);
      void pack_version_numbers(Serializer &rez);
      void unpack_version_numbers(Deserializer &derez,RegionTreeForest *forest);
    protected:
      void pack_upper_bound_node(Serializer &rez);
      void unpack_upper_bound_node(Deserializer &derez, 
                                   RegionTreeForest *forest);
    protected:
      RegionTreeNode *upper_bound_node;
      // All of these are indexed by depth in the region tree
      std::vector<PhysicalState*> physical_states;
      std::vector<FieldVersions*> field_versions;
      LegionVector<FieldMask>::aligned split_masks;
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
     * \class ProjectionInfo
     * Projection information for index space requirements
     */
    class ProjectionInfo {
    public:
      ProjectionInfo(void)
        : projection(NULL), projection_domain(Domain::NO_DOMAIN) { }
      ProjectionInfo(Runtime *runtime, const RegionRequirement &req,
                     const Domain &launch_domain);
    public:
      inline bool is_projecting(void) const { return (projection != NULL); }
      inline const LegionMap<ProjectionEpochID,FieldMask>::aligned&
        get_projection_epochs(void) const { return projection_epochs; }
      void record_projection_epoch(ProjectionEpochID epoch,
                                   const FieldMask &epoch_mask);
    public:
      void pack_info(Serializer &rez) const;
      void unpack_info(Deserializer &derez, Runtime *runtime,
          const RegionRequirement &req, const Domain &launch_domain);
    public:
      ProjectionFunction *projection;
      Domain projection_domain;
    protected:
      LegionMap<ProjectionEpochID,FieldMask>::aligned projection_epochs;
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
      FieldState(const GenericUser &u, const FieldMask &m,
                 ProjectionFunction *proj, const Domain &proj_domain, bool dis);
    public:
      inline bool is_projection_state(void) const 
        { return (open_state >= OPEN_READ_ONLY_PROJ); } 
    public:
      bool overlaps(const FieldState &rhs) const;
      void merge(const FieldState &rhs, RegionTreeNode *node);
    public:
      bool projection_domain_dominates(const Domain &next_domain) const;
    public:
      void print_state(TreeStateLogger *logger, 
                       const FieldMask &capture_mask) const;
    public:
      OpenState open_state;
      ReductionOpID redop;
      ProjectionFunction *projection;
      Domain projection_domain;
      unsigned rebuild_timeout;
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
    public:
      void advance_projection_epochs(const FieldMask &advance_mask);
      void capture_projection_epochs(const FieldMask &capture_mask,
                                     ProjectionInfo &info) const;
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
      // Fields for which we have outstanding local reductions
      FieldMask outstanding_reduction_fields;
      LegionMap<ReductionOpID,FieldMask>::aligned outstanding_reductions;
    public:
      // Fields which we know have been mutated below in the region tree
      FieldMask dirty_below;
      // Keep track of the current projection epoch for each field
      LegionMap<ProjectionEpochID,FieldMask>::aligned projection_epochs;
    };

    typedef DynamicTableAllocator<CurrentState,10,8> CurrentStateAllocator;
 
    /**
     * \struct LogicalCloser
     * This structure helps keep track of the state
     * necessary for performing a close operation
     * on the logical region tree.
     */
    class LogicalCloser {
    public:
      LogicalCloser(ContextID ctx, const LogicalUser &u,
                    RegionTreeNode *root, bool validates, bool captures);
      LogicalCloser(const LogicalCloser &rhs);
      ~LogicalCloser(void);
    public:
      LogicalCloser& operator=(const LogicalCloser &rhs);
    public:
      inline bool has_close_operations(void) const 
        { return (!!normal_close_mask) || (!!read_only_close_mask) ||
                  (!!flush_only_close_mask); }
      void record_close_operation(RegionTreeNode *root, const FieldMask &mask, 
                            bool leave_open, bool read_only, bool flush_only);
      void record_closed_child(RegionTreeNode *node, const FieldMask &mask,
                               bool read_only_close);
      void record_closed_user(const LogicalUser &user, 
                              const FieldMask &mask, bool read_only);
      void initialize_close_operations(RegionTreeNode *target, 
                                       Operation *creator,
                                       const TraceInfo &trace_info);
      void perform_dependence_analysis(const LogicalUser &current,
                                       const FieldMask &open_below,
             LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &cusers,
             LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned &pusers);
      void update_state(CurrentState &state);
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
      const bool capture_users;
      LegionList<LogicalUser,CLOSE_LOGICAL_ALLOC>::track_aligned 
                                                      normal_closed_users;
      LegionList<LogicalUser,CLOSE_LOGICAL_ALLOC>::track_aligned 
                                                      read_only_closed_users;
    protected:
      FieldMask normal_close_mask;
      FieldMask leave_open_mask;
      FieldMask read_only_close_mask;
      FieldMask flush_only_close_mask;
      // Normal closed child nodes
      LegionMap<RegionTreeNode*,FieldMask>::aligned closed_nodes;
    protected:
      // At most we will ever generate three close operations at a node
      InterCloseOp *normal_close_op;
      ReadCloseOp *read_only_close_op;
      InterCloseOp *flush_only_close_op;
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
      PhysicalState(RegionTreeNode *node, bool path_only);
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
    public:
      inline bool is_captured(void) const { return captured; }
      void capture_state(void);
      inline bool has_advance_states(void) const 
        { return (!advance_states.empty()); } 
      void apply_state(AddressSpaceID target, 
                       std::set<RtEvent> &applied_conditions) const; 
    public:
      void initialize_composite_instance(CompositeView *view,
                                         const FieldMask &close_mask);
    public:
      PhysicalState* clone(void) const;
    public:
      void print_physical_state(const FieldMask &capture_mask,
          LegionMap<ColorPoint,FieldMask>::aligned &to_traverse,
                                TreeStateLogger *logger);
    public:
      RegionTreeNode *const node;
      const bool path_only;
    public:
      // Fields which have dirty data
      FieldMask dirty_mask;
      // Fields with outstanding reductions
      FieldMask reduction_mask;
      // The valid instance views
      LegionMap<LogicalView*, FieldMask,
                VALID_VIEW_ALLOC>::track_aligned valid_views;
      // The valid reduction veiws
      LegionMap<ReductionView*, FieldMask,
                VALID_REDUCTION_ALLOC>::track_aligned reduction_views;
    protected:
      typedef VersioningSet<PHYSICAL_STATE_REF> PhysicalVersions;
      PhysicalVersions version_states;
      PhysicalVersions advance_states;
    protected:
      bool captured;
    };

    /**
     * \class VersionManager
     * The VersionManager class tracks the current version state
     * objects for a given region tree node in a specific context
     * VersionManager objects are either an owner or remote. 
     * The owner tracks the set of remote managers and invalidates
     * them whenever changes occur to the version state.
     * Owners are assigned by the enclosing task context using
     * a first-touch policy. The first node to ask to be an owner
     * for a given logical region or partition will be assigned
     * to be the owner.
     */
    class VersionManager {
    public:
      static const AllocationType alloc_type = VERSION_MANAGER_ALLOC;
      static const VersionID init_version = 1;
    public:
      VersionManager(RegionTreeNode *node, ContextID ctx); 
      VersionManager(const VersionManager &manager);
      ~VersionManager(void);
    public:
      VersionManager& operator=(const VersionManager &rhs);
      void* operator new(size_t count);
      void* operator new[](size_t count);
      void operator delete(void *ptr);
      void operator delete[](void *ptr);
    public:
      void reset(void);
    public:
      void initialize_state(ApEvent term_event,
                            const RegionUsage &usage,
                            const FieldMask &user_mask,
                            const InstanceSet &targets,
                            SingleTask *context, unsigned init_index,
                            const std::vector<LogicalView*> &corresponding);
    public:
      void record_versions(const FieldMask &version_mask,
                           FieldMask &unversioned_mask,
                           SingleTask *context,
                           Operation *op, unsigned index,
                           const RegionUsage &usage,
                           VersionInfo &version_info,
                           std::set<RtEvent> &ready_events);
      void record_path_only_versions(const FieldMask &version_mask,
                                     const FieldMask &split_mask,
                                     FieldMask &unversioned_mask,
                                     SingleTask *context,
                                     Operation *op, unsigned index,
                                     const RegionUsage &usage,
                                     VersionInfo &version_info,
                                     std::set<RtEvent> &ready_events);
      void advance_versions(FieldMask version_mask, SingleTask *context,
                            bool has_initial_state,AddressSpaceID initial_space,
                            bool update_parent_state,
                            AddressSpaceID source_space,
                            std::set<RtEvent> &applied_events,
                            bool dedup_opens = false,
                            ProjectionEpochID open_epoch = 0,
                            bool dedup_advances = false, 
                            ProjectionEpochID advance_epoch = 0);
      void update_child_versions(SingleTask *context,
                                 const ColorPoint &child_color,
                                 VersioningSet<> &new_states,
                                 std::set<RtEvent> &applied_events);
      void invalidate_version_infos(const FieldMask &invalidate_mask);
      template<ReferenceSource REF_SRC>
      static void filter_version_info(const FieldMask &invalidate_mask,
            typename LegionMap<VersionID,VersioningSet<REF_SRC> >::aligned 
                                                                &to_filter);
    public:
      void print_physical_state(RegionTreeNode *node,
                                const FieldMask &capture_mask,
                          LegionMap<ColorPoint,FieldMask>::aligned &to_traverse,
                                TreeStateLogger *logger);
    protected:
      VersionState* create_new_version_state(VersionID vid,
          bool has_initial_state, AddressSpaceID initial_space);
    public:
      RtEvent send_remote_advance(const FieldMask &advance_mask,
                                  bool has_initial_state,
                                  AddressSpaceID initial_space,
                                  bool update_parent_state,
                                  bool dedup_opens, 
                                  ProjectionEpochID open_epoch,
                                  bool dedup_advances,
                                  ProjectionEpochID advance_epoch);
      static void handle_remote_advance(Deserializer &derez, Runtime *runtime,
                                        AddressSpaceID source_space);
    public:
      RtEvent send_remote_invalidate(AddressSpaceID target,
                                     const FieldMask &invalidate_mask);
      static void handle_remote_invalidate(Deserializer &derez, 
                                           Runtime *runtime);
    public:
      RtEvent send_remote_version_request(FieldMask request_mask,
                                          std::set<RtEvent> &ready_events);
      static void handle_request(Deserializer &derez, Runtime *runtime,
                                 AddressSpaceID source_space);
    public:
      void pack_response(Serializer &rez, AddressSpaceID target,
                         const FieldMask &request_mask);
      template<ReferenceSource REF_SRC>
      static void find_send_infos(
          typename LegionMap<VersionID,VersioningSet<REF_SRC> >::aligned& 
            version_infos, const FieldMask &request_mask, 
          LegionMap<VersionState*,FieldMask>::aligned& send_infos);
      static void pack_send_infos(Serializer &rez, const
          LegionMap<VersionState*,FieldMask>::aligned& send_infos);
    public:
      void unpack_response(Deserializer &derez, RtUserEvent done_event,
                           const FieldMask &update_mask,
                           std::set<RtEvent> *applied_events);
      static void unpack_send_infos(Deserializer &derez,
          LegionMap<VersionState*,FieldMask>::aligned &infos,
          Runtime *runtime, std::set<RtEvent> &preconditions);
      template<ReferenceSource REF_SRC>
      static void merge_send_infos(
          typename LegionMap<VersionID,VersioningSet<REF_SRC> >::aligned 
            &target_infos, 
          const LegionMap<VersionState*,FieldMask>::aligned &source_infos,
          ReferenceMutator *mutator);
      static void handle_response(Deserializer &derez);
    protected:
      void sanity_check(void);
    public:
      const ContextID ctx;
      RegionTreeNode *const node;
      const unsigned depth;
      Runtime *const runtime;
    protected:
      Reservation manager_lock;
    protected:
      SingleTask *current_context;
    protected:
      bool is_owner;
      AddressSpaceID owner_space;
    protected:
      typedef VersioningSet<VERSION_MANAGER_REF> ManagerVersions;
      LegionMap<VersionID,ManagerVersions>::aligned current_version_infos;
      LegionMap<VersionID,ManagerVersions>::aligned previous_version_infos;
    protected:
      // On the owner node this is the set of fields for which there are
      // remote copies. On remote nodes this is the set of fields which
      // are locally valid.
      FieldMask remote_valid_fields;
    protected:
      // Owner information about which nodes have remote copies
      LegionMap<AddressSpaceID,FieldMask>::aligned remote_valid;
      // Information about preivous opens
      LegionMap<ProjectionEpochID,FieldMask>::aligned previous_opens;
      // Information about previous advances
      LegionMap<ProjectionEpochID,FieldMask>::aligned previous_advancers;
      // Remote information about outstanding requests we've made
      LegionMap<RtUserEvent,FieldMask>::aligned outstanding_requests;
    };

    typedef DynamicTableAllocator<VersionManager,10,8> VersionManagerAllocator;

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
        INITIAL_VERSION_REQUEST,
        FINAL_VERSION_REQUEST,
      };
    public:
      struct SendVersionStateArgs {
      public:
        HLRTaskID hlr_id;
        VersionState *proxy_this;
        AddressSpaceID target;
        FieldMask *request_mask;
        RtUserEvent to_trigger;
      };
      struct UpdateStateReduceArgs {
      public:
        HLRTaskID hlr_id;
        VersionState *proxy_this;
        ColorPoint child_color;
        VersioningSet<> *children;
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
      struct FinalRequestFunctor {
      public:
        FinalRequestFunctor(VersionState *proxy, AddressSpaceID r,
                            const FieldMask &m, std::set<RtEvent> &pre)
          : proxy_this(proxy), requestor(r), mask(m), preconditions(pre) { }
      public:
        void apply(AddressSpaceID target);
      private:
        VersionState *proxy_this;
        AddressSpaceID requestor;
        const FieldMask &mask;
        std::set<RtEvent> &preconditions;
      };
    public:
      VersionState(VersionID vid, Runtime *rt, DistributedID did,
                   AddressSpaceID owner_space, AddressSpaceID local_space, 
                   bool has_initial_state, AddressSpaceID initial_space,
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
      void update_path_only_state(PhysicalState *state,
                                  const FieldMask &update_mask) const;
      void update_physical_state(PhysicalState *state, 
                                 const FieldMask &update_mask) const; 
    public: // methods for applying state information
      void merge_physical_state(const PhysicalState *state, 
                                const FieldMask &merge_mask,
                                AddressSpaceID target,
                                std::set<RtEvent> &applied_conditions,
                                bool need_lock = true);
      void reduce_open_children(const ColorPoint &child_color,
                                const FieldMask &update_mask,
                                VersioningSet<> &new_states,
                                ReferenceMutator *mutator,
                                bool need_lock = true);
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
    public:
      void send_version_state_update(AddressSpaceID target,
                         const FieldMask &request_mask, RtUserEvent to_trigger);
      void send_version_state_update_request(AddressSpaceID target, 
                          AddressSpaceID src, RtUserEvent to_trigger, 
                          const FieldMask &request_mask,
                          VersionRequestKind request_kind);
      void launch_send_version_state_update(AddressSpaceID target,
                                     RtUserEvent to_trigger, 
                                     const FieldMask &request_mask, 
                                     RtEvent precondition=RtEvent::NO_RT_EVENT);
    public:
      void send_version_state(AddressSpaceID source);
      static void handle_version_state_request(Deserializer &derez,
                                      Runtime *runtime, AddressSpaceID source);
      static void handle_version_state_response(Deserializer &derez,
                                      Runtime *runtime, AddressSpaceID source);
    public:
      void handle_version_state_update_request(AddressSpaceID source, 
                                        RtUserEvent to_trigger, 
                                        VersionRequestKind request_kind,
                                        FieldMask &request_mask);
      void handle_version_state_update_response(RtUserEvent to_trigger, 
                          Deserializer &derez, const FieldMask &update);
    public:
      static void process_version_state_reduction(const void *args);
    public:
      void remove_version_state_ref(ReferenceSource ref_kind, 
                                     RtEvent done_event);
      static void process_remove_version_state_ref(const void *args);
    public:
      static void process_view_references(const void *args);
    public:
      static void process_version_state_update_request(Runtime *rt, 
                                                Deserializer &derez);
      static void process_version_state_update_response(Runtime *rt,
                                                 Deserializer &derez); 
    public:
      const VersionID version_number;
      RegionTreeNode *const logical_node;
      const bool has_initial_state;
      const AddressSpaceID initial_space;
    protected:
      Reservation state_lock;
      // Fields which have been directly written to
      FieldMask dirty_mask;
      // Fields which have reductions
      FieldMask reduction_mask;
      typedef VersioningSet<VERSION_STATE_TREE_REF> StateVersions;
      LegionMap<ColorPoint,StateVersions>::aligned open_children;
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
      // The fields for which we have seen updates
      FieldMask valid_fields;
      // Track when we have valid data for initial and final fields
      LegionMap<RtEvent,FieldMask>::aligned initial_events;
      LegionMap<RtEvent,FieldMask>::aligned final_events;
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
    class ReductionCloser : public NodeTraverser {
    public:
      ReductionCloser(ContextID ctx, ReductionView *target,
                      const FieldMask &reduc_mask, 
                      VersionInfo &version_info, 
                      Operation *op, unsigned index,
                      std::set<RtEvent> &map_applied_events);
      ReductionCloser(const ReductionCloser &rhs);
      ~ReductionCloser(void);
    public:
      virtual bool visit_only_valid(void) const { return true; }
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    public:
      ReductionCloser& operator=(const ReductionCloser &rhs);
      void issue_close_reductions(RegionTreeNode *node);
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
      virtual bool visit_only_valid(void) const { return true; }
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    };

  }; // namespace Internal 
}; // namespace Legion

#endif // __LEGION_ANALYSIS_H__
