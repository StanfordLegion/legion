/* Copyright 2013 Stanford University
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


#ifndef __REGION_TREE_H__
#define __REGION_TREE_H__

#include "legion_types.h"
#include "legion_utilities.h"

namespace LegionRuntime {
  namespace HighLevel {

    /////////////////////////////////////////////////////////////
    // Region Tree Forest 
    /////////////////////////////////////////////////////////////
    class RegionTreeForest {
    public:
      enum SendingMode {
        PHYSICAL,
        PRIVILEGE,
        DIFF,
      };
    public:
      RegionTreeForest(HighLevelRuntime *rt);
      ~RegionTreeForest(void);
    public:
      void lock_context(bool exclusive = true);
      void unlock_context(void);
#ifdef DEBUG_HIGH_LEVEL
      void assert_locked(void);
      void assert_not_locked(void);
#endif
    public:
      bool compute_index_path(IndexSpace parent, IndexSpace child, std::vector<Color> &path);
      bool compute_partition_path(IndexSpace parent, IndexPartition child, std::vector<Color> &path);
      bool is_disjoint(LogicalPartition partition);
      // Checks for r1 <= r2 or r2 <= r1, but won't catch !(r1 * r2) at the same level
      bool are_overlapping(LogicalRegion h1, LogicalRegion h2);
    public:
      // Index Space operations
      void create_index_space(Domain domain);
      void destroy_index_space(IndexSpace space, bool finalize, const std::vector<ContextID> &deletion_contexts);
      void get_destroyed_regions(IndexSpace space, std::vector<LogicalRegion> &new_deletions);
      Color create_index_partition(IndexPartition pid, IndexSpace parent, bool disjoint, int color,
                                  const std::map<Color,Domain> &coloring, Domain color_space); 
      void destroy_index_partition(IndexPartition pid, bool finalize, const std::vector<ContextID> &deletion_contexts);
      void get_destroyed_partitions(IndexPartition pid, std::vector<LogicalPartition> &new_deletions);
      IndexPartition get_index_partition(IndexSpace parent, Color color, bool can_create);
      IndexSpace get_index_subspace(IndexPartition p, Color color, bool can_create);
      Domain get_index_space_domain(IndexSpace handle, bool can_create);
      Domain get_index_partition_color_space(IndexPartition p, bool can_create);
    public:
      // Field Space operations
      void create_field_space(FieldSpace space);
      void destroy_field_space(FieldSpace space, bool finalize, const std::vector<ContextID> &deletion_contexts);
      void get_destroyed_regions(FieldSpace space, std::vector<LogicalRegion> &new_deletions);
      void allocate_fields(FieldSpace space, const std::map<FieldID,size_t> &field_allocations);
      void free_fields(FieldSpace space, const std::set<FieldID> &to_free);
      bool has_field(FieldSpace space, FieldID fid);
      size_t get_field_size(FieldSpace space, FieldID fid);
    public:
      // Logical Region operations
      void create_region(LogicalRegion handle, ContextID ctx);  
      void destroy_region(LogicalRegion handle, bool finalize, const std::vector<ContextID> &deletion_contexts);
      void destroy_partition(LogicalPartition handle, bool finalize, const std::vector<ContextID> &deletion_contexts);
      LogicalPartition get_region_partition(LogicalRegion parent, IndexPartition handle);
      LogicalRegion get_partition_subregion(LogicalPartition parent, IndexSpace handle);
      LogicalPartition get_region_subcolor(LogicalRegion parent, Color c, bool can_create);
      LogicalRegion get_partition_subcolor(LogicalPartition parent, Color c, bool can_create);
      LogicalPartition get_partition_subtree(IndexPartition handle, FieldSpace space, RegionTreeID tid);
      LogicalRegion get_region_subtree(IndexSpace handle, FieldSpace space, RegionTreeID tid);
      LogicalRegion get_partition_parent(LogicalPartition handle);
    public:
      // Logical Region contexts 
      void initialize_logical_context(LogicalRegion handle, ContextID ctx);
      void analyze_region(RegionAnalyzer &az);
      // Special registrations for deletions
      void analyze_index_space_deletion(ContextID ctx, IndexSpace sp, DeletionOperation *op);
      void analyze_index_part_deletion(ContextID ctx, IndexPartition part, DeletionOperation *op);
      void analyze_field_space_deletion(ContextID ctx, FieldSpace sp, DeletionOperation *op);
      void analyze_field_deletion(ContextID ctx, FieldSpace sp, const std::set<FieldID> &to_free, DeletionOperation *op);
      void analyze_region_deletion(ContextID ctx, LogicalRegion handle, DeletionOperation *op);
      void analyze_partition_deletion(ContextID ctx, LogicalPartition handle, DeletionOperation *op);
    public:
      // Physical Region contexts
      InstanceRef initialize_physical_context(const RegionRequirement &req, unsigned idx, InstanceRef ref, UniqueID uid, 
                                              ContextID ctx, Event term_event, const DomainPoint &point);
      void map_region(RegionMapper &rm, LogicalRegion start_region);
      Event close_to_instance(const InstanceRef &ref, RegionMapper &rm);
      Event close_to_reduction(const InstanceRef &ref, RegionMapper &rm);
      void invalidate_physical_context(const RegionRequirement &req, const std::vector<FieldID> &new_fields, 
                                       ContextID ctx, bool new_only);
      void invalidate_physical_context(LogicalRegion handle, ContextID ctx);
      void invalidate_physical_context(LogicalPartition handle, ContextID ctx);
      void invalidate_physical_context(LogicalRegion handle, ContextID ctx, const std::vector<FieldID> &fields,
                                       bool last_use);
      void merge_field_context(LogicalRegion handle, ContextID outer_ctx, ContextID inner_ctx, const std::vector<FieldID> &fields);
    public:
      // Packing and unpacking send
      size_t compute_region_forest_shape_size(const std::vector<IndexSpaceRequirement> &indexes,
                                            const std::vector<FieldSpaceRequirement> &fields,
                                            const std::vector<RegionRequirement> &regions);
      void pack_region_forest_shape(Serializer &rez);
      void unpack_region_forest_shape(Deserializer &derez);
    private:
      FieldMask compute_field_mask(const RegionRequirement &req, SendingMode mode, FieldSpaceNode *field_node) const;
    public:
      // Packing and unpacking state send
      size_t compute_region_tree_state_size(const RegionRequirement &req, ContextID ctx, SendingMode mode);
      size_t post_compute_region_tree_state_size(void);
      void begin_pack_region_tree_state(Serializer &rez, unsigned long num_ways = 1);
      void pack_region_tree_state(const RegionRequirement &req, ContextID ctx, SendingMode mode, Serializer &rez
#ifdef DEBUG_HIGH_LEVEL
            , unsigned idx, const char *task_name, unsigned uid
#endif
          );
      void begin_unpack_region_tree_state(Deserializer &derez, unsigned long split_factor = 1);
      void unpack_region_tree_state(const RegionRequirement &req, ContextID ctx, SendingMode mode, Deserializer &derez
#ifdef DEBUG_HIGH_LEVEL
            , unsigned idx, const char *task_name, unsigned uid
#endif
          );
    public:
      // Packing and unpacking reference send
      size_t compute_reference_size(InstanceRef ref);
      void pack_reference(const InstanceRef &ref, Serializer &derez);
      InstanceRef unpack_reference(Deserializer &derez);
    public:
      // Packing and unpacking reference return
      size_t compute_reference_size_return(InstanceRef ref);
      void pack_reference_return(InstanceRef ref, Serializer &rez);
      void unpack_and_remove_reference(Deserializer &derez, UniqueID uid); // will unpack and remove reference
    public:
      // Packing and unpacking structure updates return
      size_t compute_region_tree_updates_return(void);
      void pack_region_tree_updates_return(Serializer &rez);
      void unpack_region_tree_updates_return(Deserializer &derez, const std::vector<ContextID> &enclosing_contexts);
    public:
      // Packing and unpacking state return
      size_t compute_region_tree_state_return(const RegionRequirement &req, unsigned idx, 
                                              ContextID ctx, bool overwrite, SendingMode mode 
#ifdef DEBUG_HIGH_LEVEL
                                              , const char *task_name
                                              , unsigned uid
#endif
                                              );
      
      void post_partition_state_return(const RegionRequirement &req, ContextID ctx, SendingMode mode);
      size_t post_compute_region_tree_state_return(bool created_only);
      void begin_pack_region_tree_state_return(Serializer &rez, bool created_only);
      void pack_region_tree_state_return(const RegionRequirement &req, unsigned idx, 
                              ContextID ctx, bool overwrite, SendingMode mode, Serializer &rez);
      
      void post_partition_pack_return(const RegionRequirement &req, ContextID ctx, SendingMode mode);
      void end_pack_region_tree_state_return(Serializer &rez, bool created_only);
      void begin_unpack_region_tree_state_return(Deserializer &derez, bool created_only);
      void unpack_region_tree_state_return(const RegionRequirement &req, ContextID ctx, 
                                            bool overwrite, SendingMode mode, Deserializer &derez
#ifdef DEBUG_HIGH_LEVEL
                                            , unsigned idx, const char *task_name
                                            , unsigned uid
#endif
                                            );
      void end_unpack_region_tree_state_return(Deserializer &derez, bool created_only);
    public:
      // Packing and unpacking for returning created field states for pre-existing region trees
      size_t compute_created_field_state_return(LogicalRegion handle, const std::vector<FieldID> &fields, 
                                                ContextID ctx
#ifdef DEBUG_HIGH_LEVEL
                                              , const char *task_name, unsigned uid
#endif
                                              );
      void pack_created_field_state_return(LogicalRegion handle, const std::vector<FieldID> &fields, 
                                            ContextID ctx, Serializer &rez);
      void unpack_created_field_state_return(LogicalRegion handle, ContextID ctx, Deserializer &derez
#ifdef DEBUG_HIGH_LEVEL
                                            , const char *task_name, unsigned uid      
#endif
                                            );
    public:
      // Packing and unpacking for returning created region trees
      size_t compute_created_state_return(LogicalRegion handle, ContextID ctx
#ifdef DEBUG_HIGH_LEVEL
                                            , const char *task_name, unsigned uid
#endif
                                            );
      void pack_created_state_return(LogicalRegion handle, ContextID ctx, Serializer &rez);
      void unpack_created_state_return(LogicalRegion handle, ContextID ctx, Deserializer &derez
#ifdef DEBUG_HIGH_LEVEL
                                            , const char *task_name, unsigned uid
#endif
                                          );
    public:
      // Packing and unpacking leaked references
      size_t compute_leaked_return_size(void);
      void pack_leaked_return(Serializer &rez);
      void unpack_leaked_return(Deserializer &derez); // will unpack leaked references and remove them
#ifdef DYNAMIC_TESTS
    public:
      // For performing dynamic independence tests
      bool fix_dynamic_test_set(void); // Return true if there are any tests to perform (must hold forest lock)
      void perform_dynamic_tests(void); // Perform the set of dynamic tests (don't need to hold the lock if only on performing
      void publish_dynamic_test_results(void);
#endif
    protected:
      friend class IndexSpaceNode;
      friend class IndexPartNode;
      friend class FieldSpaceNode;
      friend class RegionTreeNode;
      friend class RegionNode;
      friend class PartitionNode;
      friend class InstanceManager;
      friend class PhysicalManager;
      friend class ReductionManager;
      friend class ListReductionManager;
      friend class FoldReducitonManager;
      friend class PhysicalView;
      friend class InstanceView;
      friend class ReductionView;
    public: 
      IndexSpaceNode* create_node(Domain d, IndexPartNode *par, Color c, bool add);
      IndexPartNode* create_node(IndexPartition p, IndexSpaceNode *par, Color c, Domain color_space, bool dis, bool add);
      FieldSpaceNode* create_node(FieldSpace sp);
      RegionNode* create_node(LogicalRegion r, PartitionNode *par, bool add);
      PartitionNode* create_node(LogicalPartition p, RegionNode *par, bool add);
    public:
      void destroy_node(IndexSpaceNode *node, bool top, bool finalize); // (recursive)
      void destroy_node(IndexPartNode *node, bool top, bool finalize); // (recursive)
      void destroy_node(FieldSpaceNode *node);
      void destroy_node(RegionNode *node, bool top, bool finalize, const std::vector<ContextID> &deletion_contexts); // (recursive)
      void destroy_node(PartitionNode *node, bool top, bool finalize, const std::vector<ContextID> &deletion_contexts); // (recursive)
    public:
      IndexSpaceNode* get_node(IndexSpace space);
      IndexPartNode*  get_node(IndexPartition part);
      FieldSpaceNode* get_node(FieldSpace space);
      RegionNode*     get_node(LogicalRegion handle);
      PartitionNode * get_node(LogicalPartition handle);
      RegionNode*     get_top_node(RegionTreeID tid);
    public:
      bool has_node(IndexSpace space) const;
      bool has_node(IndexPartition part) const;
      bool has_node(FieldSpace space) const;
      bool has_node(LogicalRegion handle, bool strict = true) const;
      bool has_node(LogicalPartition handle, bool strict = true) const;
    public:
      InstanceView* create_instance_view(InstanceManager *manager, InstanceView *par, RegionNode *reg, bool made_local);
      InstanceManager* create_instance_manager(Memory location, PhysicalInstance inst, 
                        const std::map<FieldID,Domain::CopySrcDstField> &infos, FieldSpace fsp,
                        const FieldMask &field_mask, bool remote, bool clone, UniqueManagerID mid = 0);
      ReductionView* create_reduction_view(ReductionManager *manager, RegionNode *reg, bool made_local);
      ReductionManager* create_reduction_manager(Memory location, PhysicalInstance inst, ReductionOpID redop, const ReductionOp *op, 
            bool remote, bool clone, Domain domain = Domain::NO_DOMAIN, UniqueManagerID mid = 0);
      PhysicalView* find_view(InstanceKey key) const;
      PhysicalManager* find_manager(UniqueManagerID mid) const;
      InstanceView* find_instance_view(InstanceKey key) const;
      InstanceManager* find_instance_manager(UniqueManagerID mid) const;
      ReductionView* find_reduction_view(InstanceKey key) const;
      ReductionManager* find_reduction_manager(UniqueManagerID mid) const;
      bool has_view(InstanceKey key) const;
      bool has_instance_view(InstanceKey key) const;
      bool has_reduction_view(InstanceKey key) const;
    public:
      template<typename T>
      Color generate_unique_color(const std::map<Color,T> &current_map);
    private:
      HighLevelRuntime *const runtime;
      // There are two primary locks for a RegionTreeForest.  The context_lock
      // is the coarse grained lock that supports mutating the state of the
      // RegionTreeForest.  In general all operations need to hold this lock
      // before performing any operations on the data structure.  The second lock,
      // the creation_lock is used for reading the shape of the forest without
      // holding the context_lock.  This allows for simultaneous mapping and
      // resolving of name queuries (e.g. get subregion with color 'c' or partition P)
      // which means tasks don't need to wait for mapping to finish before
      // continuing.  To update the shape of the structure of the RegionTreeForest
      // you still need to hold the context lock and the creation lock.
#ifdef LOW_LEVEL_LOCKS
      Lock context_lock;
      Lock creation_lock;
#else
      ImmovableLock context_lock;
      ImmovableLock creation_lock;
#endif
#ifdef DEBUG_HIGH_LEVEL
      bool lock_held;
#endif
    private:
      // The creation lock must be held whenever updating one of these
      // five data structures or any of the color_maps
      std::map<IndexSpace,IndexSpaceNode*>     index_nodes;
      std::map<IndexPartition,IndexPartNode*>  index_parts;
      std::map<FieldSpace,FieldSpaceNode*>     field_nodes;
      std::map<LogicalRegion,RegionNode*>     region_nodes;
      std::map<LogicalPartition,PartitionNode*> part_nodes;
    private:
      // References to delete when cleaning up
      std::map<UniqueManagerID,InstanceManager*> managers;
      std::map<InstanceKey,InstanceView*> views;
      std::map<UniqueManagerID,ReductionManager*> reduc_managers;
      std::map<InstanceKey,ReductionView*> reduc_views;
    private: // lists of new things to know what to return
      std::list<IndexSpace> created_index_trees;
      std::list<IndexSpace> deleted_index_spaces;
      std::list<IndexPartition> deleted_index_parts;
    private:
      std::set<FieldSpace> created_field_spaces;
      std::list<FieldSpace> deleted_field_spaces;
    private:
      std::list<LogicalRegion> created_region_trees;
      std::list<LogicalRegion> deleted_regions;
      std::list<LogicalPartition> deleted_partitions;
    private:
      // Data structures for determining what to pack and unpack when moving trees
      std::set<IndexSpaceNode*>     send_index_nodes;
      std::set<FieldSpaceNode*>     send_field_nodes;
      std::set<RegionNode*>         send_logical_nodes;
      std::vector<IndexPartNode*>  new_index_part_nodes;
      std::set<InstanceManager*>        unique_managers;
      std::set<ReductionManager*>       unique_reductions;
      std::map<InstanceView*,FieldMask> unique_views; // points to the top instance view
      std::vector<InstanceView*>        ordered_views;
      std::vector<bool>                 overwrite_views; // for knowing when to overwrite views when returning 
      std::map<unsigned,std::vector<RegionNode*> >  diff_region_maps;
      std::map<unsigned,std::vector<PartitionNode*> > diff_part_maps;
      std::vector<InstanceManager*>     returning_managers;
      std::vector<ReductionManager*>    returning_reductions;
      std::vector<InstanceView*>        returning_views;
      std::vector<ReductionView*>       returning_reduc_views;
      std::map<EscapedUser,unsigned>    escaped_users;
      std::set<EscapedCopy>             escaped_copies;
      std::vector<RegionNode*>          created_field_space_trees;
      std::vector<FieldSpaceNode*>      created_field_nodes;
    private:
      std::list<IndexSpaceNode*> top_index_trees;
      std::list<RegionNode*>   top_logical_trees;
#ifdef DYNAMIC_TESTS
    private:
      class DynamicSpaceTest {
      public:
        DynamicSpaceTest(IndexPartNode *parent, Color c1, IndexSpace left, Color c2, IndexSpace right);
        bool perform_test(void);
        void publish_test(void) const;
      public:
        IndexPartNode *parent;
        Color c1, c2;
        IndexSpace left, right;
      };
      class DynamicPartTest {
      public:
        DynamicPartTest(IndexSpaceNode *parent, Color c1, Color c2);
        void add_child_space(bool left, IndexSpace space);
        bool perform_test(void);
        void publish_test(void) const;
      public:
        IndexSpaceNode *parent;
        Color c1, c2;
        std::vector<IndexSpace> left, right;
      };
    private:
      std::list<DynamicSpaceTest> dynamic_space_tests;
      std::list<DynamicPartTest>  dynamic_part_tests;
      std::list<DynamicSpaceTest> ghost_space_tests;
      std::list<DynamicPartTest>  ghost_part_tests;
#endif
    };

    /////////////////////////////////////////////////////////////
    // Index Space Node 
    /////////////////////////////////////////////////////////////
    /**
     * Capture the relationship of index spaces
     */
    class IndexSpaceNode {
    public:
      friend class RegionTreeForest;
      friend class IndexPartNode;
      friend class RegionNode;
      friend class PartitionNode;
      friend class InstanceManager;
      friend class PhysicalManager;
      friend class ReductionManager;
      friend class ListReductionManager;
      friend class FoldReducitonManager;
      friend class PhysicalView;
      friend class InstanceView;
      friend class ReductionView;
      IndexSpaceNode(Domain d, IndexPartNode *par,
                Color c, bool add, RegionTreeForest *ctx);
      ~IndexSpaceNode(void);
      void mark_destroyed(void);
    public:
      void add_child(IndexPartition handle, IndexPartNode *node);
      void remove_child(Color c);
      bool has_child(Color c) const;
      IndexPartNode* get_child(Color c);
      bool are_disjoint(Color c1, Color c2);
      void add_disjoint(Color c1, Color c2);
      Color generate_color(void);
    public:
      void add_instance(RegionNode *inst);
      void remove_instance(RegionNode *inst);
      RegionNode* instantiate_region(RegionTreeID tid, FieldSpace fid);
    public:
      size_t compute_tree_size(bool returning) const;
      void serialize_tree(Serializer &rez, bool returning);
      static IndexSpaceNode* deserialize_tree(Deserializer &derez, IndexPartNode *parent,
                          RegionTreeForest *context, bool returning);
      void mark_node(bool recurse);
      IndexSpaceNode* find_top_marked(void) const;
      void find_new_partitions(std::vector<IndexPartNode*> &new_parts) const;
    private:
      const Domain domain;
      const IndexSpace handle;
      const unsigned depth;
      const Color color;
      IndexPartNode *const parent;
      RegionTreeForest *const context;
      std::map<Color,IndexPartNode*> color_map; // all children seen locally ever
      std::map<Color,IndexPartNode*> valid_map; // valid children
      std::list<RegionNode*> logical_nodes; // corresponding region nodes
      std::set<std::pair<Color,Color> > disjoint_subsets; // pairs of disjoint subsets
      bool added; // added locally
      bool marked;
      bool destroy_index_space; // destroy index space when deleting node
      bool node_destroyed; // node has been destroyed
    };

    /////////////////////////////////////////////////////////////
    // Index Partition Node 
    /////////////////////////////////////////////////////////////
    class IndexPartNode {
    public:
      friend class RegionTreeForest;
      friend class IndexSpaceNode;
      friend class RegionNode;
      friend class PartitionNode;
      friend class InstanceManager;
      friend class PhysicalManager;
      friend class ReductionManager;
      friend class ListReductionManager;
      friend class FoldReducitonManager;
      friend class PhysicalView;
      friend class InstanceView;
      friend class ReductionView;
      IndexPartNode(IndexPartition p, IndexSpaceNode *par,
                Color c, Domain color_space, bool dis, bool add, RegionTreeForest *ctx);
      ~IndexPartNode(void);
      void mark_destroyed(void);
    public:
      void add_child(IndexSpace handle, IndexSpaceNode *node);
      void remove_child(Color c);
      bool has_child(Color c) const;
      IndexSpaceNode* get_child(Color c);
      bool are_disjoint(Color c1, Color c2);
      void add_disjoint(Color c1, Color c2);
    public:
      void add_instance(PartitionNode *inst);
      void remove_instance(PartitionNode *inst);
      PartitionNode* instantiate_partition(RegionTreeID tid, FieldSpace fid);
    public:
      size_t compute_tree_size(bool returning) const;
      void serialize_tree(Serializer &rez, bool returning);
      static void deserialize_tree(Deserializer &derez, IndexSpaceNode *parent, 
                        RegionTreeForest *context, bool returning);
      void mark_node(bool recurse);
      IndexSpaceNode* find_top_marked(void) const;
      void find_new_partitions(std::vector<IndexPartNode*> &new_parts) const;
    private:
      const IndexPartition handle;
      const unsigned depth;
      const Color color;
      const Domain color_space;
      IndexSpaceNode *const parent;
      RegionTreeForest *const context;
      std::map<Color,IndexSpaceNode*> color_map; // all children seen locally ever
      std::map<Color,IndexSpaceNode*> valid_map; // valid children
      std::list<PartitionNode*> logical_nodes; // corresponding partition nodes
      std::set<std::pair<Color,Color> > disjoint_subspaces; // for non-disjoint partitions
      const bool disjoint;
      bool added; // was the node added locally
      bool marked;
      bool node_destroyed; // has this node been destroyed
    };

    /////////////////////////////////////////////////////////////
    // Field Space Node 
    /////////////////////////////////////////////////////////////
    class FieldSpaceNode {
    public:
      friend class RegionTreeForest;
      friend class InstanceManager;
      friend class PhysicalManager;
      friend class ReductionManager;
      friend class ListReductionManager;
      friend class FoldReducitonManager;
      friend class PhysicalView;
      friend class InstanceView;
      friend class ReductionView;
      FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx);
      ~FieldSpaceNode(void);
      void mark_destroyed(void);
    public:
      struct FieldInfo {
      public:
        FieldInfo(void) : field_size(0), idx(0) { }
        FieldInfo(size_t size, unsigned id)
          : field_size(size), idx(id) { }
      public:
        size_t field_size;
        unsigned idx;
      };
    public:
      void allocate_fields(const std::map<FieldID,size_t> &field_allocations);
      void free_fields(const std::set<FieldID> &to_free);
      bool has_field(FieldID fid);
      size_t get_field_size(FieldID fid);
      bool is_set(FieldID fid, const FieldMask &mask) const;
    public:
      void add_instance(RegionNode *node);
      void remove_instance(RegionNode *node);
      InstanceManager* create_instance(Memory location, Domain domain, 
                    const std::vector<FieldID> &fields, size_t blocking_factor);
    public:
      size_t compute_node_size(void) const;
      void serialize_node(Serializer &rez) const;
      static FieldSpaceNode* deserialize_node(Deserializer &derez, RegionTreeForest *context);
    public:
      bool has_modifications(void) const;
      size_t compute_field_return_size(void) const;
      void serialize_field_return(Serializer &rez);
      void deserialize_field_return(Deserializer &derez);
      unsigned get_shift_amount(void) const { return shift; }
    public:
      FieldMask get_field_mask(const std::vector<FieldID> &fields) const;
      FieldMask get_field_mask(const std::set<FieldID> &fields) const;
      FieldMask get_field_mask(void) const;
      FieldMask get_created_field_mask(void) const;
    public:
      // For debugging return an allocated string for the given mask
      char* to_string(const FieldMask &mask) const;
      // Sanity check debugging
      void sanity_check(void) const;
    private:
      const FieldSpace handle;
      RegionTreeForest *const context;
      // Top nodes in the trees for which this field space is used 
      std::list<RegionNode*> logical_nodes;
      std::map<FieldID,FieldInfo> fields;
      std::list<FieldID> created_fields;
      std::list<FieldID> deleted_fields;
      unsigned total_index_fields;
      // Shift is only valid while the context lock is held during
      // the deserialization phase, once it is released, all bets are off
      unsigned shift;
    };

    /////////////////////////////////////////////////////////////
    // Region Tree Node 
    /////////////////////////////////////////////////////////////
    /**
     * A generic parent class for RegionNode and PartitionNode
     */
    class RegionTreeNode {
    public:
      enum OpenState {
        NOT_OPEN           = 0,
        OPEN_READ_ONLY     = 1,
        OPEN_READ_WRITE    = 2, // unknown dirty information below
        OPEN_SINGLE_REDUCE = 3, // only one open child with only reductions below
        OPEN_MULTI_REDUCE  = 4, // multiple open children with the same reduction
      };
    public:
      struct FieldState {
      public:
        FieldState(void);
        FieldState(const GenericUser &user);
        FieldState(const GenericUser &user, const FieldMask &mask, Color next);
      public:
        bool still_valid(void) const;
        bool overlap(const FieldState &rhs) const;
        void merge(const FieldState &rhs);
        bool upgrade(const FieldState &rhs);
        void clear(const FieldMask &init_mask);
      public:
        size_t compute_state_size(const FieldMask &pack_mask) const;
        void pack_physical_state(const FieldMask &pack_mask, Serializer &rez) const;
        void unpack_physical_state(Deserializer &derez, unsigned shift = 0);
      public:
        void print_state(TreeStateLogger *logger, FieldMask capture_mask) const;
      public:
        FieldMask valid_fields;
        OpenState open_state;
        ReductionOpID redop;
        std::map<Color,FieldMask> open_children;
      };
      struct GenericState {
      public:
        std::list<FieldState> field_states;
        // The following data structure tracks diffs for sending back
        std::list<FieldState> added_states;
      };
      struct LogicalState : public GenericState {
      public:
        std::list<LogicalUser> curr_epoch_users; // Users from the current epoch
        std::list<LogicalUser> prev_epoch_users; // Users from the previous epoch
      };
      struct PhysicalState : public GenericState {
      public:
        PhysicalState(void) { }
        PhysicalState(FieldMask top) : top_mask(top) { }
      public:
        std::map<InstanceView*,FieldMask> valid_views;
        std::map<ReductionView*,FieldMask> reduction_views;
        FieldMask dirty_mask;
        FieldMask reduction_mask;
        // Used for tracking diffs in the physical tree for sending back
        std::map<InstanceView*,FieldMask> added_views;
        std::map<ReductionView*,FieldMask> added_reductions;
        // Mark the top of the context for given fields
        FieldMask top_mask;
      public:
        void clear_state(const FieldMask &init_mask);
      };
    public:
      RegionTreeNode(RegionTreeForest *ctx);
    public:
      void register_logical_region(const LogicalUser &user, RegionAnalyzer &az);
      void open_logical_tree(const LogicalUser &user, RegionAnalyzer &az);
      void open_logical_tree(const LogicalUser &user, const ContextID ctx, std::vector<Color> &path);
      void close_logical_tree(LogicalCloser &closer, const FieldMask &closing_mask);
    public:
      virtual void close_physical_tree(PhysicalCloser &closer, const FieldMask &closing_mask) = 0;
      virtual void close_reduction_tree(ReductionCloser &closer, const FieldMask &closing_mask) = 0;
    protected:
      // Generic operations on the region tree
      //**********************************************************************//
      // HEY YOU! (out there in the cold) YES YOU!
      // PAY ATTENTION RIGHT HERE!!!!!!!!!!!!!!!!!
      // These are the two most important functions in all of Legion!
      // During both logical and physical traversals, for every node that is visited
      // siphon_open_children is invoked to close up any open children that must be
      // closed before the traversal can continue.  To perform the close operation
      // on a FieldState, perform_close_operations is invoked.
      //**********************************************************************//
      bool siphon_open_children(TreeCloser &closer, // Closer that will perform any close operations
                                GenericState &state, // The state of the RegionTreeNode
                                const GenericUser &user, // The user for which we are traversing
                                const FieldMask &current_mask, // The local field mask for which we are traversing
                                int next_child = -1); // The next child we are going to traverse (if any)
      bool perform_close_operations(TreeCloser &closer, // Closer for performing close operations
                                    const GenericUser &user, // The user for which we are performing the close
                                    const FieldMask &closing_mask, // The field mask for the close operation
                                    FieldState &state, // The field state we are closing
                                    int next_child, // The next child we are traversing (or -1 if none)
                                    bool allow_next_child, // Are we allowed to keep the state open if it is the same child
                                    bool upgrade_next_child, // Whether or not the same child will need to upgrade its state
                                    bool permit_leave_open, // Whether or not we are allowed to leave this state open
                                    std::vector<FieldState> &new_states, // Append downgraded states to the set of new states
                                    FieldMask &already_open); // return set of fields which are already open for the next_child
    protected:
      // Logical region helper functions
      FieldMask perform_dependence_checks(const LogicalUser &user, 
                    const std::list<LogicalUser> &users, const FieldMask &user_mask, bool closing_partition = false);
      // This will merge the new state in with existing states 
      void merge_new_field_state(GenericState &gstate, const FieldState &new_state, bool add_state);
      void merge_new_field_states(GenericState &gstate, std::vector<FieldState> &new_states, bool add_states);
      // This will upgrade all the fields to the new state, checking that upgrades are safe
      void upgrade_new_field_state(GenericState &gstate, const FieldState &new_state, bool add_state);
      void sanity_check_field_states(const GenericState &gstate);
      void down_sample_list(std::list<LogicalUser> &users, unsigned max_users);
      void filter_user_list(std::list<LogicalUser> &users);
    protected:
      virtual bool are_children_disjoint(Color c1, Color c2) = 0;
      virtual bool are_closing_partition(void) const = 0;
      virtual RegionTreeNode* get_tree_child(Color c) = 0;
      virtual Color get_color(void) const = 0;
#ifdef DEBUG_HIGH_LEVEL
      virtual bool color_match(Color c) = 0;
#endif
    protected:
      RegionTreeForest *const context;
      std::map<ContextID,LogicalState> logical_states;
      std::map<ContextID,PhysicalState> physical_states;
    };

    /////////////////////////////////////////////////////////////
    // Region Node 
    /////////////////////////////////////////////////////////////
    /**
     * Represents a single logical region
     */
    class RegionNode : public RegionTreeNode {
    public:
      friend class RegionTreeForest;
      friend class IndexSpaceNode;
      friend class PartitionNode;
      friend class InstanceManager;
      friend class PhysicalManager;
      friend class ReductionManager;
      friend class ListReductionManager;
      friend class FoldReducitonManager;
      friend class PhysicalView;
      friend class InstanceView;
      friend class ReductionView;
      friend class TreeStateLogger;
      RegionNode(LogicalRegion r, PartitionNode *par, IndexSpaceNode *row_src,
                 FieldSpaceNode *col_src, bool add, RegionTreeForest *ctx);
      virtual ~RegionNode(void);
      void mark_destroyed(void);
    public:
      void add_child(LogicalPartition handle, PartitionNode *child);
      bool has_child(Color c) const;
      PartitionNode* get_child(Color c);
      void remove_child(Color c);
    public:
      // Logical context operations
      void initialize_logical_context(ContextID ctx);
      void register_deletion_operation(ContextID ctx, DeletionOperation *op, const FieldMask &deletion_mask);
    public:
      // Physical context operations
      void initialize_physical_context(ContextID ctx, bool clear, const FieldMask &init_mask, bool top);
      void fill_exclusive_context(ContextID ctx, const FieldMask &fill_mask, Color open_child);
      void merge_physical_context(ContextID outer_ctx, ContextID inner_ctx, const FieldMask &merge_mask);
      void register_physical_region(const PhysicalUser &user, RegionMapper &rm);
      void open_physical_tree(const PhysicalUser &user, RegionMapper &rm);
      virtual void close_physical_tree(PhysicalCloser &closer, const FieldMask &closing_mask);
      virtual void close_reduction_tree(ReductionCloser &closer, const FieldMask &closing_mask);
    protected:
      virtual bool are_children_disjoint(Color c1, Color c2);
      virtual bool are_closing_partition(void) const;
      virtual RegionTreeNode* get_tree_child(Color c);
      virtual Color get_color(void) const;
#ifdef DEBUG_HIGH_LEVEL
      virtual bool color_match(Color c);
#endif
    public:
      // Physical traversal methods
      InstanceView* map_physical_region(const PhysicalUser &user, RegionMapper &rm);
      ReductionView* map_reduction_region(const PhysicalUser &user, RegionMapper &rm);
      void update_valid_views(ContextID ctx, const FieldMask &valid_mask, bool dirty, InstanceView* new_view);
      void update_valid_views(ContextID ctx, const FieldMask &valid_mask, 
                          const FieldMask &dirty_mask, const std::vector<InstanceView*>& new_views);
      void update_reduction_views(ContextID ctx, const FieldMask &valid_mask, ReductionView *new_view);
      // Issue the necessary copies for each of the fields in the copy mask to make the data in
      // the destination instance up-to-date.
      void issue_update_copies(InstanceView *dst, RegionMapper &rm, FieldMask copy_mask);
      // Issue reductions from the current reduction instances to the target instance to make it valid
      void issue_update_reductions(PhysicalView *target, const FieldMask &mask, RegionMapper &rm);
      void mark_invalid_instance_views(ContextID ctx, const FieldMask &invalid_mask, bool recurse);
      void recursive_invalidate_views(ContextID ctx, const FieldMask &invalid_mask, bool last_use);
      void invalidate_instance_views(PhysicalState &state, const FieldMask &invalid_mask, bool clean);
      void invalidate_reduction_views(PhysicalState &state, const FieldMask &invalid_mask);
      void find_valid_instance_views(ContextID ctx, 
                                     std::list<std::pair<InstanceView*,FieldMask> > &valid_views, 
                                     const FieldMask &valid_mask, const FieldMask &field_mask, bool needs_space);
      void find_valid_reduction_views(ContextID ctx, std::list<ReductionView*> &valid_views, const FieldMask &valid_mask);
      InstanceView* create_instance(Memory location, RegionMapper &rm, const std::vector<FieldID> &new_fields);
      ReductionView* create_reduction(Memory location, RegionMapper &rm);
      void issue_final_close_operation(ContextID ctx, const PhysicalUser &user, TreeCloser &closer);
      void pull_valid_views(ContextID ctx, const FieldMask &field_mask);
      void flush_reductions(const PhysicalUser &user, RegionMapper &rm);
      void update_top_mask(FieldMask allocated_mask);
    public:
      size_t compute_tree_size(bool returning) const;
      void serialize_tree(Serializer &rez, bool returning);
      static RegionNode* deserialize_tree(Deserializer &derez, PartitionNode *parent,
                      RegionTreeForest *context, bool returning);
      void mark_node(bool recurse);
      RegionNode* find_top_marked(void) const;
      void find_new_partitions(std::vector<PartitionNode*> &new_parts) const;
    public:
      size_t compute_state_size(ContextID ctx, const FieldMask &pack_mask,
                                std::set<InstanceManager*> &unique_managers, 
                                std::set<ReductionManager*> &unique_reductions,
                                bool mark_invalid_views, bool recurse); 
      void pack_physical_state(ContextID ctx, const FieldMask &pack_mask,
                                Serializer &rez, bool invalidate_views, bool recurse);
      void unpack_physical_state(ContextID ctx, Deserializer &derez, bool recurse, unsigned shift = 0);
    public:
      size_t compute_diff_state_size(ContextID, const FieldMask &pack_mask,
                                std::set<InstanceManager*> &unique_managers,
                                std::set<ReductionManager*> &unique_reductions,
                                std::vector<RegionNode*> &diff_regions,
                                std::vector<PartitionNode*> &diff_partitions,
                                bool invalidate_views, bool recurse);
      void pack_diff_state(ContextID ctx, const FieldMask &pack_mask, Serializer &rez);
      void unpack_diff_state(ContextID ctx, Deserializer &derez);
    public:
      void print_physical_context(ContextID ctx, TreeStateLogger *logger, FieldMask capture_mask);
    private:
      const LogicalRegion handle;
      PartitionNode *const parent;
      IndexSpaceNode *const row_source;
      FieldSpaceNode *const column_source; // only valid for top of region trees
      std::map<Color,PartitionNode*> color_map; // all children seen locally ever
      std::map<Color,PartitionNode*> valid_map; // only valid children
      bool added;
      bool marked;
    };

    /////////////////////////////////////////////////////////////
    // Partition Node 
    /////////////////////////////////////////////////////////////
    class PartitionNode : public RegionTreeNode {
    public:
      friend class RegionTreeForest;
      friend class IndexPartNode;
      friend class RegionNode;
      friend class InstanceManager;
      friend class PhysicalManager;
      friend class ReductionManager;
      friend class ListReductionManager;
      friend class FoldReducitonManager;
      friend class PhysicalView;
      friend class InstanceView;
      friend class ReducitonView;
      friend class TreeStateLogger;
      PartitionNode(LogicalPartition p, RegionNode *par, IndexPartNode *row_src,
                    bool add, RegionTreeForest *ctx);
      virtual ~PartitionNode(void);
      void mark_destroyed(void);
    public:
      void add_child(LogicalRegion handle, RegionNode *child);
      bool has_child(Color c) const;
      RegionNode* get_child(Color c);
      void remove_child(Color c);
    public:
      // Logical context operations
      void initialize_logical_context(ContextID ctx);
      void register_deletion_operation(ContextID ctx, DeletionOperation *op, const FieldMask &deletion_mask);
    public:
      void initialize_physical_context(ContextID ctx, bool clear, const FieldMask &initialize_mask);
      void fill_exclusive_context(ContextID ctx, const FieldMask &fill_mask, Color open_child);
      void merge_physical_context(ContextID outer_ctx, ContextID inner_ctx, const FieldMask &merge_mask);
      void register_physical_region(const PhysicalUser &user, RegionMapper &rm);
      void open_physical_tree(const PhysicalUser &user, RegionMapper &rm);
      virtual void close_physical_tree(PhysicalCloser &closer, const FieldMask &closing_mask);
      virtual void close_reduction_tree(ReductionCloser &closer, const FieldMask &closing_mask);
    protected:
      virtual bool are_children_disjoint(Color c1, Color c2);
      virtual bool are_closing_partition(void) const;
      virtual RegionTreeNode* get_tree_child(Color c);
      virtual Color get_color(void) const;
#ifdef DEBUG_HIGH_LEVEL
      virtual bool color_match(Color c);
#endif
    public:
      size_t compute_tree_size(bool returning) const;
      void serialize_tree(Serializer &rez, bool returning);
      static void deserialize_tree(Deserializer &derez, RegionNode *parent,
                        RegionTreeForest *context, bool returning);
      void mark_node(bool recurse);
      RegionNode* find_top_marked(void) const;
      void find_new_partitions(std::vector<PartitionNode*> &new_parts) const;
    public:
      size_t compute_state_size(ContextID ctx, const FieldMask &pack_mask,
                                std::set<InstanceManager*> &unique_managers, 
                                std::set<ReductionManager*> &unique_reductions,
                                bool mark_invalid_views, bool recurse);
      void pack_physical_state(ContextID ctx, const FieldMask &mask,
                                Serializer &rez, bool invalidate_views, bool recurse);
      void unpack_physical_state(ContextID ctx, Deserializer &derez, bool recurse, unsigned shift = 0);
    public:
      size_t compute_diff_state_size(ContextID, const FieldMask &pack_mask,
                                std::set<InstanceManager*> &unique_managers,
                                std::set<ReductionManager*> &unique_reductions,
                                std::vector<RegionNode*> &diff_regions,
                                std::vector<PartitionNode*> &diff_partitions,
                                bool invalidate_views, bool recurse);
      void pack_diff_state(ContextID ctx, const FieldMask &pack_mask, Serializer &rez);
      void unpack_diff_state(ContextID ctx, Deserializer &derez);
    public:
      void mark_invalid_instance_views(ContextID ctx, const FieldMask &invalid_mask, bool recurse);
      void recursive_invalidate_views(ContextID ctx, const FieldMask &invalid_mask, bool last_use);
    public:
      void print_physical_context(ContextID ctx, TreeStateLogger *logger, FieldMask capture_mask);
    private:
      const LogicalPartition handle;
      RegionNode *const parent;
      IndexPartNode *const row_source;
      // No column source here
      std::map<Color,RegionNode*> color_map; // all children seen locally ever
      std::map<Color,RegionNode*> valid_map; // only valid children
      const bool disjoint;
      bool added;
      bool marked;
    };

    /////////////////////////////////////////////////////////////
    // Region Usage 
    /////////////////////////////////////////////////////////////
    struct RegionUsage {
    public:
      RegionUsage(void)
        : privilege(NO_ACCESS), prop(EXCLUSIVE), redop(0) { }
      RegionUsage(PrivilegeMode p, CoherenceProperty c, ReductionOpID r)
        : privilege(p), prop(c), redop(r) { }
      RegionUsage(const RegionRequirement &req)
        : privilege(req.privilege), prop(req.prop), redop(req.redop) { }
    public:
      inline bool operator==(const RegionUsage &rhs) const
      { return ((privilege == rhs.privilege) && (prop == rhs.prop) && (redop == rhs.redop)); }
      inline bool operator!=(const RegionUsage &rhs) const
      { return !((*this) == rhs); }
    public:
      PrivilegeMode     privilege;
      CoherenceProperty prop;
      ReductionOpID     redop;
    };

    /////////////////////////////////////////////////////////////
    // Generic User 
    /////////////////////////////////////////////////////////////
    struct GenericUser {
    public:
      GenericUser(void) { }
      GenericUser(const FieldMask &m, const RegionUsage &u);
    public:
      FieldMask field_mask;
      RegionUsage usage;
    };

    /////////////////////////////////////////////////////////////
    // Logical User 
    /////////////////////////////////////////////////////////////
    struct LogicalUser : public GenericUser {
    public:
      LogicalUser(void) : GenericUser(), op(NULL), idx(0), gen(0) { }
      LogicalUser(GeneralizedOperation *o, unsigned id, const FieldMask &m, const RegionUsage &u);
    public:
      bool operator==(const LogicalUser &rhs) const;
      bool operator!=(const LogicalUser &rhs) const { return !((*this) == rhs); }
    public:
      GeneralizedOperation *op;
      unsigned idx;
      GenerationID gen;
    };

    /////////////////////////////////////////////////////////////
    // Physical User 
    /////////////////////////////////////////////////////////////
    struct PhysicalUser : public GenericUser {
    public:
      PhysicalUser(void) : GenericUser(), single_term(Event::NO_EVENT), multi_term(Event::NO_EVENT), idx(0) { }
      PhysicalUser(const FieldMask &m, const RegionUsage &u, Event single, Event multi, unsigned idx);
    public:
      bool operator==(const PhysicalUser &rhs) const;
      bool operator!=(const PhysicalUser &rhs) const { return !((*this) == rhs); }
    public:
      Event single_term;
      Event multi_term;
      unsigned idx;
    };

    /////////////////////////////////////////////////////////////
    // Physical Manager 
    /////////////////////////////////////////////////////////////
    /**
     * This is the base class for all managers, including both
     * instance and reduction managers, and contains all the
     * logic that can be shared between them.
     */
    class PhysicalManager {
    public:
      PhysicalManager(RegionTreeForest *ctx, UniqueManagerID mid, bool remote, bool clone, Memory loc, PhysicalInstance inst);
      virtual ~PhysicalManager(void);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic> get_accessor(void) const = 0;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic> get_field_accessor(FieldID fid) const = 0;
      virtual bool is_reduction_manager(void) const = 0;
      virtual InstanceManager* as_instance_manager(void) const = 0;
      virtual ReductionManager* as_reduction_manager(void) const = 0;
    public:
      inline UniqueManagerID get_unique_id(void) const { return unique_id; }
      inline Memory get_location(void) const { return location; }
      inline bool is_returned(void) const { return remote_frac.is_empty(); }
      inline bool is_clone(void) const { return clone; }
      inline bool is_remote(void) const { return remote; }
      inline PhysicalInstance get_instance(void) const 
      { 
#ifdef DEBUG_HIGH_LEVEL
        assert(instance.exists());
#endif
        return instance; 
      }
    protected:
      RegionTreeForest *const context;
      unsigned references;
      const UniqueManagerID unique_id;
      bool remote;
      const bool clone;
      InstFrac remote_frac; // The fraction we are remote from somewhere else
      InstFrac local_frac; // Fraction of this instance info that is still here
      const Memory location;
      PhysicalInstance instance;
    };

    /////////////////////////////////////////////////////////////
    // Instance Manager 
    /////////////////////////////////////////////////////////////
    /**
     * Instance managers are the objects that represent physical
     * instances and keeps track of when they can be garbage
     * collected.  Once all InstanceViews have removed their
     * reference then they can be collected.
     */
    class InstanceManager : public PhysicalManager {
    public:
      InstanceManager(Memory m, PhysicalInstance inst, const std::map<FieldID,Domain::CopySrcDstField> &infos,
              FieldSpace fsp, const FieldMask &mask, RegionTreeForest *ctx, UniqueManagerID mid, bool rem, bool clone);
      virtual ~InstanceManager(void);
    public:
      inline const FieldMask& get_allocated_fields(void) const { return allocated_fields; }
      inline Lock get_lock(void) const { return lock; }
      inline UniqueManagerID get_unique_id(void) const { return unique_id; }
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic> get_accessor(void) const
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(instance.exists());
#endif
        Accessor::RegionAccessor<Accessor::AccessorType::Generic> result = instance.get_accessor();
        return result;
      }
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic> get_field_accessor(FieldID fid) const
      {
        std::map<FieldID,Domain::CopySrcDstField>::const_iterator finder = field_infos.find(fid);
#ifdef DEBUG_HIGH_LEVEL
        assert(instance.exists());
        assert(finder != field_infos.end());
#endif
        Accessor::RegionAccessor<Accessor::AccessorType::Generic> temp = instance.get_accessor();
        return temp.get_untyped_field_accessor(finder->second.offset, finder->second.size);
      }
      virtual bool is_reduction_manager(void) const { return false; }
      virtual InstanceManager* as_instance_manager(void) const { return const_cast<InstanceManager*>(this); }
      virtual ReductionManager* as_reduction_manager(void) const { return NULL; }
    public:
      void add_reference(void);
      void remove_reference(void);
      void add_view(InstanceView *view);
      void compute_copy_offsets(const FieldMask &mask, std::vector<Domain::CopySrcDstField> &fields);
      void find_info(FieldID fid, std::vector<Domain::CopySrcDstField> &sources);
      InstanceManager* clone_manager(const FieldMask &mask, FieldSpaceNode *node) const;
    public:
      // This function helps to find views that need to be sent from the a specific logical region
      // and any of it's children with an optional filter on the chosen children
      void find_views_from(LogicalRegion handle, std::map<InstanceView*,FieldMask> &unique_views,
                            std::vector<InstanceView*> &ordered_views, const FieldMask &packing_mask, int filter = -1);
    public:
      size_t compute_send_size(void) const;
      void pack_manager_send(Serializer &rez, unsigned long num_ways);
      static void unpack_manager_send(RegionTreeForest *context, Deserializer &derez, unsigned long split_factor);
    public:
      void find_user_returns(std::vector<InstanceView*> &returning_views) const;
      size_t compute_return_size(void) const;
      void pack_manager_return(Serializer &rez);
      static void unpack_manager_return(RegionTreeForest *context, Deserializer &derez); 
    public:
      // For checking if an instance manager no longer has any valid
      // references and can therefore be returned back
      bool is_valid_free(void) const;
    public:
      void pack_remote_fraction(Serializer &rez);
      void unpack_remote_fraction(Deserializer &derez);
    private:
      void garbage_collect(void);
    private:
      friend class RegionTreeForest;
      friend class PhysicalView;
      friend class InstanceView;
      friend class ReductionView;
      friend class RegionNode; // for sanity checks only 
      Lock lock;
      const FieldSpace fspace;
      FieldMask allocated_fields;
      std::map<FieldID,Domain::CopySrcDstField> field_infos;
      // One nice property of this view of views is that they are in
      // a partial order that will allow them to be serialized and deserialized correctly
      std::vector<InstanceView*> all_views;
    };

    /////////////////////////////////////////////////////////////
    // Reduction Manager 
    /////////////////////////////////////////////////////////////
    /**
     * For keeping track of the information for reduction instances
     */
    class ReductionManager : public PhysicalManager {
    public:
      ReductionManager(RegionTreeForest *ctx, UniqueManagerID mid, bool remote, bool clone, Memory loc, 
                        PhysicalInstance inst, ReductionOpID redop, const ReductionOp *op);
      virtual ~ReductionManager(void);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic> get_accessor(void) const = 0;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic> get_field_accessor(FieldID fid) const = 0;
      virtual bool is_reduction_manager(void) const { return true; }
      virtual InstanceManager* as_instance_manager(void) const { return NULL; }
      virtual ReductionManager* as_reduction_manager(void) const { return const_cast<ReductionManager*>(this); }
      virtual ReductionManager* clone_manager(void) const = 0;
      virtual void find_field_offsets(const FieldMask &reduce_mask, std::vector<Domain::CopySrcDstField> &fields) = 0;
      virtual Event issue_reduction(const std::vector<Domain::CopySrcDstField> &src_fields,
                                    const std::vector<Domain::CopySrcDstField> &dst_fields,
                                    Domain space, Event precondition, bool reduction_fold) = 0;
      virtual bool is_foldable(void) const = 0;
      virtual Domain get_pointer_space(void) const = 0;
    public:
      void set_view(ReductionView *v);
      void add_reference(void);
      void remove_reference(void);
    public:
      size_t compute_send_size(void) const;
      void pack_manager_send(Serializer &rez, unsigned long num_ways);
      static void unpack_manager_send(RegionTreeForest *context, Deserializer &derez, unsigned long split_factor);
    public:
      size_t compute_return_size(void) const;
      void pack_manager_return(Serializer &rez);
      static void unpack_manager_return(RegionTreeForest *context, Deserializer &derez);
    public:
      void pack_remote_fraction(Serializer &rez);
      void unpack_remote_fraction(Deserializer &derez);
    public:
      bool is_valid_free(void) const;
      void find_user_returns(std::vector<ReductionView*> &returning_views) const;
    private:
      void garbage_collect(void);
    public:
      const ReductionOpID redop;
      const ReductionOp *op;
    private:
      friend class RegionTreeForest;
      friend class PhysicalView;
      friend class InstanceView;
      friend class ReductionView;
      friend class RegionNode;
      ReductionView *view;
    };

    /////////////////////////////////////////////////////////////
    // List Reduction Manager 
    /////////////////////////////////////////////////////////////
    /**
     * The manager for list reduction instances
     */
    class ListReductionManager : public ReductionManager {
    public:
      ListReductionManager(RegionTreeForest *ctx, UniqueManagerID mid, bool remote, bool clone, Memory loc, 
                        PhysicalInstance inst, ReductionOpID redop, const ReductionOp *op, Domain domain);
      virtual ~ListReductionManager(void);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic> get_accessor(void) const;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic> get_field_accessor(FieldID fid) const;
      virtual ReductionManager* clone_manager(void) const;
      virtual void find_field_offsets(const FieldMask &reduce_mask, std::vector<Domain::CopySrcDstField> &fields);
      virtual Event issue_reduction(const std::vector<Domain::CopySrcDstField> &src_fields,
                                    const std::vector<Domain::CopySrcDstField> &dst_fields,
                                    Domain space, Event precondition, bool reduction_fold);
      virtual bool is_foldable(void) const { return false; }
      virtual Domain get_pointer_space(void) const { return ptr_space; }
    public:
      const Domain ptr_space;
    };

    /////////////////////////////////////////////////////////////
    // Fold Reduction Manager 
    /////////////////////////////////////////////////////////////
    /**
     * The manager for fold reduction instances
     */
    class FoldReductionManager : public ReductionManager {
    public:
      FoldReductionManager(RegionTreeForest *ctx, UniqueManagerID mid, bool remote, bool clone, Memory loc, 
                        PhysicalInstance inst, ReductionOpID redop, const ReductionOp *op);
      virtual ~FoldReductionManager(void);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic> get_accessor(void) const;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic> get_field_accessor(FieldID fid) const;
      virtual ReductionManager* clone_manager(void) const;
      virtual void find_field_offsets(const FieldMask &reduce_mask, std::vector<Domain::CopySrcDstField> &fields);
      virtual Event issue_reduction(const std::vector<Domain::CopySrcDstField> &src_fields,
                                    const std::vector<Domain::CopySrcDstField> &dst_fields,
                                    Domain space, Event precondition, bool reduction_fold);
      virtual bool is_foldable(void) const { return true; }
      virtual Domain get_pointer_space(void) const { return Domain::NO_DOMAIN; }
    };

    /**
     * A class for uniquely identifying a specific InstanceView on
     * a single node in the system.  Primarily this allows us to
     * merge InstanceViews that may represent the same state but
     * were created on different nodes.
     */
    class InstanceKey {
    public:
      InstanceKey(void);
      InstanceKey(UniqueManagerID mid, LogicalRegion handle);
    public:
      bool operator==(const InstanceKey &rhs) const;
      bool operator<(const InstanceKey &rhs) const;
    public:
      UniqueManagerID mid;
      LogicalRegion handle;
    };

    /////////////////////////////////////////////////////////////
    // Physical View 
    /////////////////////////////////////////////////////////////
    /**
     * This is the base class for both instance views and reduction
     * views and contains all the logic that can be shared between them.
     */
    class PhysicalView {
    public:
      PhysicalView(RegionTreeForest *ctx, RegionNode *logical, bool made_local);
      virtual ~PhysicalView(void);
    public:
      virtual bool is_reduction_view(void) const = 0;
      virtual InstanceView* as_instance_view(void) const = 0;
      virtual ReductionView* as_reduction_view(void) const = 0;
      virtual PhysicalManager* get_manager(void) const = 0;
      virtual InstanceRef add_copy_user(ReductionOpID redop, Event copy_term, const FieldMask &mask, bool reading) = 0;
      virtual void remove_user(UniqueID uid, unsigned refs, bool force) = 0;
      virtual void remove_copy(Event copy, bool force) = 0;
      virtual void add_valid_reference(void) = 0;
      virtual void remove_valid_reference(void) = 0;
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask, std::set<Event> &preconditions,
                                            std::vector<Domain::CopySrcDstField> &dst_fields) = 0;
    public:
      RegionTreeForest *const context;
      RegionNode *const logical_region;
    protected:
      unsigned valid_references;
      bool local_view; // true until it is sent back in some form
    };

    /////////////////////////////////////////////////////////////
    // Instance View 
    /////////////////////////////////////////////////////////////
    /**
     * Instance views correspond to the view to a single physical
     * instance from a given location in the region tree.  Instance
     * views also manage the references from that viewpoint.  Once
     * all the references from tasks and the region tree are removed
     * from the instance view, then it will remove its reference
     * from the instance manager which can lead to the instance
     * being garbage collected.
     */
    class InstanceView : public PhysicalView {
    protected:
      struct ReferenceTracker {
      public:
        ReferenceTracker(void)
          : references(0) { }
        ReferenceTracker(unsigned refs, const DomainPoint &point)
          : references(refs) { }
      public:
        bool add_reference(unsigned refs, const DomainPoint &point);
        bool remove_reference(unsigned refs);
        bool use_single_event(void) const;
        unsigned get_reference_count(void) const;
      public:
        size_t compute_tracker_send(void) const;
        void pack_tracker_send(Serializer &rez);
        void unpack_tracker_send(Deserializer &derez);
      public:
        size_t compute_tracker_return(void) const;
        void pack_tracker_return(Serializer &rez);
        void unpack_tracker_return(Deserializer &derez);
      private:
        unsigned references;
        std::set<DomainPoint,DomainPoint::STLComparator> points;
      };
      struct CopyUser {
      public:
        CopyUser(void)
          :  redop(0), references(0) { }
        CopyUser(ReductionOpID op, unsigned ref)
          : redop(op), references(ref) { }
      public:
        ReductionOpID redop;
        unsigned references;
      };
      struct AliasedUser {
      public:
        AliasedUser(void) { }
        AliasedUser(const FieldMask &mask, const PhysicalUser &u, bool single)
          : valid_mask(mask), user(u), use_single(single) { }
      public:
        FieldMask valid_mask;
        PhysicalUser user;
        bool use_single;
      };
      struct AliasedCopyUser {
      public:
        AliasedCopyUser(void) { }
        AliasedCopyUser(const FieldMask &mask, Event ready, ReductionOpID op)
          : valid_mask(mask), ready_event(ready), redop(op) { }
      public:
        FieldMask valid_mask;
        Event ready_event;
        ReductionOpID redop;
      };
      typedef std::pair<Event,FieldMask> AliasedEvent;
    public:
      InstanceView(InstanceManager *man, InstanceView *par,  
                    RegionNode *reg, RegionTreeForest *ctx, bool made_local);
      virtual ~InstanceView(void);
    public:
      InstanceView* get_subview(Color pc, Color rc);
      void add_child_view(Color pc, Color rc, InstanceView *child);
      void remove_child_view(const std::pair<Color,Color> &key);
    public:
      virtual bool is_reduction_view(void) const { return false; } 
      virtual InstanceView* as_instance_view(void) const { return const_cast<InstanceView*>(this); }
      virtual ReductionView* as_reduction_view(void) const { return NULL; } 
      InstanceRef add_user(UniqueID uid, const PhysicalUser &user, const DomainPoint &point);
      InstanceRef add_init_user(UniqueID uid, const PhysicalUser &user, const DomainPoint &point);
      virtual InstanceRef add_copy_user(ReductionOpID redop, Event copy_term, const FieldMask &mask, bool reading);
      virtual void remove_user(UniqueID uid, unsigned refs, bool force);
      virtual void remove_copy(Event copy, bool force);
    public:
      virtual void add_valid_reference(void);
      virtual void remove_valid_reference(void);
      void mark_to_be_invalidated(void);
      bool is_valid_view(void) const;
      bool has_war_dependence(const PhysicalUser &user) const;
    public:
      inline PhysicalInstance get_instance(void) const { return manager->get_instance(); }
      inline Memory get_location(void) const { return manager->get_location(); }
      inline const FieldMask& get_physical_mask(void) const { return manager->get_allocated_fields(); }
      inline InstanceKey get_key(void) const { return InstanceKey(manager->get_unique_id(), logical_region->handle); }
      virtual PhysicalManager* get_manager(void) const { return manager; }
      inline InstanceManager* get_instance_manager(void) const { return manager; }
      inline LogicalRegion get_handle(void) const { return logical_region->handle; }
      Event get_valid_event(const FieldMask &mask);
      void copy_to(RegionMapper &rm, const FieldMask &copy_mask, std::set<Event> &preconditions,
                                            std::vector<Domain::CopySrcDstField> &dst_fields);
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask, std::set<Event> &preconditions,
                                            std::vector<Domain::CopySrcDstField> &dst_fields);
      void copy_from(RegionMapper &rm, const FieldMask &copy_mask, std::set<Event> &preconditions,
                                            std::vector<Domain::CopySrcDstField> &src_fields);
      void find_copy_preconditions(std::set<Event> &wait_on, bool writing, ReductionOpID redop, const FieldMask &mask);
      void update_valid_event(Event new_valid, const FieldMask &mask);
    private:
      void check_state_change(bool adding);
      void find_dependences_above(std::set<Event> &wait_on, const PhysicalUser &user, InstanceView *child);
      void find_dependences_above(std::set<Event> &wait_on, bool writing, ReductionOpID redop, const FieldMask &mask, InstanceView *child);
      bool find_dependences_below(std::set<Event> &wait_on, const PhysicalUser &user);
      bool find_dependences_below(std::set<Event> &wait_on, bool writing, ReductionOpID redop, const FieldMask &mask);
      bool find_local_dependences(std::set<Event> &wait_on, const PhysicalUser &user);
      bool find_local_dependences(std::set<Event> &wait_on, bool writing, ReductionOpID redop, const FieldMask &mask);
      bool has_war_dependence_above(const PhysicalUser &user) const;
      bool has_war_dependence_below(const PhysicalUser &user) const;
      bool has_local_war_dependence(const PhysicalUser &user) const;
      template<typename T>
      void remove_invalid_elements(std::map<T,FieldMask> &elements, const FieldMask &new_mask);
      void remove_invalid_users(std::map<std::pair<UniqueID,unsigned>,PhysicalUser> &users, const FieldMask &new_mask);
    public:
      size_t compute_send_size(const FieldMask &pack_mask);
      void pack_view_send(const FieldMask &pack_mask, Serializer &rez);
      static void unpack_view_send(RegionTreeForest *context, Deserializer &derez);
    public:
      void find_required_views(std::map<InstanceView*,FieldMask> &unique_views,
             std::vector<InstanceView*> &ordered_views, const FieldMask &packing_mask, int filter);
      void find_required_below(std::map<InstanceView*,FieldMask> &unique_views,
             std::vector<InstanceView*> &ordered_views, const FieldMask &packing_mask);
    public:
      void find_aliased_above(std::vector<AliasedUser> &add_aliased_users, 
                              std::vector<AliasedCopyUser> &add_aliased_copies,
                              std::vector<AliasedEvent> &add_aliased_events,
                              const FieldMask &packing_mask, InstanceView *child_source);
      void find_aliased_below(std::vector<AliasedUser> &add_aliased_users, 
                              std::vector<AliasedCopyUser> &add_aliased_copies,
                              std::vector<AliasedEvent> &add_aliased_events,
                              const FieldMask &packing_mask);
      void find_aliased_local(std::vector<AliasedUser> &add_aliased_users, 
                              std::vector<AliasedCopyUser> &add_aliased_copies,
                              std::vector<AliasedEvent> &add_aliased_events,
                              const FieldMask &packing_mask);
    public:
      bool has_added_users(void) const;
      size_t compute_return_state_size(const FieldMask &pack_mask, bool overwrite, 
          std::map<EscapedUser,unsigned> &escaped_users, std::set<EscapedCopy> &escaped_copies);
      size_t compute_return_users_size(std::map<EscapedUser,unsigned> &escaped_users,
                                       std::set<EscapedCopy> &escaped_copies,
                                       bool already_returning, const FieldMask &returning_mask);
      void pack_return_state(const FieldMask &mask, bool overwrite, Serializer &rez);
      void pack_return_users(Serializer &rez);
      static void unpack_return_state(RegionTreeForest *context, Deserializer &derez);
      static void unpack_return_users(RegionTreeForest *context, Deserializer &derez);
    public:
      size_t compute_simple_return(void) const;
      void pack_simple_return(Serializer &rez);
      static void unpack_simple_return(RegionTreeForest *context, Deserializer &derez);
    public:
      // Some helper methods
      template<typename T>
      static void pack_vector(const std::vector<T> &to_pack, Serializer &rez);
      template<typename T>
      static void unpack_vector(std::vector<T> &target, Deserializer &derez);
#ifdef DEBUG_HIGH_LEVEL
    public:
      void sanity_check_state(void) const;
#endif
    public:
      InstanceManager *const manager;
      InstanceView *const parent; 
    private:
      friend class RegionTreeForest;
      friend class InstanceManager; 
      friend class PhysicalManager;
      friend class ReductionManager;
      friend class ListReductionManager;
      friend class FoldReducitonManager;
      typedef std::pair<Color/*part color*/,Color/*region color*/> ChildKey;
      std::map<ChildKey,InstanceView*> children;
      // For each child keep track of other children with which it aliases
      std::map<InstanceView*,std::set<InstanceView*> > aliased_children;
      // The next four members only deal with garbage collection
      // and should be passed back whenever an InstanceView is
      // passed back and is not remote
      std::map<UniqueID,ReferenceTracker> users;
      std::map<UniqueID,ReferenceTracker> added_users;
      std::map<Event,ReductionOpID> copy_users; // if redop > 0 then writing reduction, otherwise just a read
      std::map<Event,ReductionOpID> added_copy_users;
#ifdef LEGION_SPY
      std::map<UniqueID,ReferenceTracker> deleted_users;
      std::map<Event,ReductionOpID> deleted_copy_users;
#endif
      // The next three members deal with dependence analysis
      // and the state of the view, they should always entirely
      // be passed back
      std::map<std::pair<UniqueID,unsigned/*idx*/>,PhysicalUser> epoch_users;
      std::map<Event,FieldMask> epoch_copy_users;
      std::map<Event,FieldMask> valid_events;
      // These are users and valid events that would normally come
      // from doing 'find_dependences_above', but since we are
      // remote we summarize them here.  Note we never have to
      // send these back since they are always only a summary of 
      // things that already exist on the original node.
      std::vector<AliasedUser> aliased_users;
      std::vector<AliasedCopyUser> aliased_copy_users;
      std::vector<AliasedEvent> aliased_valid_events;
      // Everything below here is used for serializing and deserializing instance views
      size_t packing_sizes[7]; // storage for packing instances
      bool to_be_invalidated; // about to be invalidated
      // For packing aliased things to be sent remotely 
      std::vector<AliasedUser> packing_aliased_users;
      std::vector<AliasedCopyUser> packing_aliased_copy_users;
      std::vector<AliasedEvent> packing_aliased_valid_events;
      // For packing trackers
      std::map<UniqueID,int/*location*/> needed_trackers;
    };

    /////////////////////////////////////////////////////////////
    // Reduction View 
    /////////////////////////////////////////////////////////////
    /**
     * For keeping track of views on reduction instances
     */
    class ReductionView : public PhysicalView {
    protected:
      struct TaskUser {
      public:
        TaskUser(void)
          : references(0), use_multi(false) { }
        TaskUser(const PhysicalUser &u, unsigned ref)
          : user(u), mask(u.field_mask), references(ref), use_multi(false) { }
      public:
        PhysicalUser user;
        FieldMask mask;
        unsigned references;
        bool use_multi;
      };
      struct CopyUser {
      public:
        CopyUser(void)
          : reading(true) { }
        CopyUser(bool read, const FieldMask &m)
          : reading(read), mask(m) { }
      public:
        bool reading;
        FieldMask mask;
      };
    public:
      ReductionView(RegionTreeForest *ctx, RegionNode *logical, bool made_local, ReductionManager *man);
      virtual ~ReductionView(void);
    public:
      virtual bool is_reduction_view(void) const { return true; }
      virtual InstanceView* as_instance_view(void) const { return NULL; }
      virtual ReductionView* as_reduction_view(void) const { return const_cast<ReductionView*>(this); } 
      virtual PhysicalManager* get_manager(void) const { return manager; }
      InstanceRef add_user(UniqueID uid, const PhysicalUser &user);
      InstanceRef add_init_user(UniqueID uid, const PhysicalUser &user);
      virtual InstanceRef add_copy_user(ReductionOpID redop, Event copy_term, const FieldMask &mask, bool reading);
      virtual void remove_user(UniqueID uid, unsigned refs, bool force);
      virtual void remove_copy(Event copy, bool force);
      virtual void add_valid_reference(void);
      virtual void remove_valid_reference(void);
    public:
      inline ReductionManager* get_reduction_manager(void) const { return manager; }
      inline Memory get_location(void) const { return manager->get_location(); }
      inline InstanceKey get_key(void) const { return InstanceKey(manager->get_unique_id(), logical_region->handle); }
      inline PhysicalInstance get_instance(void) const { return manager->get_instance(); }
    public:
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask, std::set<Event> &preconditions,
                                            std::vector<Domain::CopySrcDstField> &dst_fields);
      void reduce_from(ReductionOpID redop, const FieldMask &reduce_mask, std::set<Event> &preconditions,
                                            std::vector<Domain::CopySrcDstField> &src_fields);
      void perform_reduction(PhysicalView *dst, const FieldMask &reduce_mask, RegionMapper &rm);
      
      Event get_valid_event(const FieldMask &mask);
      void mark_to_be_invalidated(void);
      bool has_added_users(void) const;
      bool is_valid_view(void) const;
    public:
      size_t compute_send_size(void) const;
      void pack_view_send(Serializer &rez);
      static void unpack_view_send(RegionTreeForest *context, Deserializer &derez);
    public:
      size_t compute_return_size(std::map<EscapedUser,unsigned> &escaped_users, 
                                 std::set<EscapedCopy> &escaped_copies) const;
      void pack_view_return(Serializer &rez);
      static void unpack_view_return(RegionTreeForest *context, Deserializer &derez);
    private:
      void find_preconditions(const std::map<UniqueID,TaskUser> &current_users, 
                    const FieldMask &reduce_mask, std::set<Event> &preconditions);
      void find_read_preconditions(const std::map<Event,CopyUser> &current_users, 
                    const FieldMask &reduce_mask, std::set<Event> &preconditions);
      void find_write_preconditions(const std::map<Event,CopyUser> &current_users, 
                    const FieldMask &reduce_mask, std::set<Event> &preconditions);
      void check_state_change(bool adding);
    public:
      ReductionManager *const manager;
    private:
      friend class RegionTreeForest;
      friend class InstanceManager; 
      friend class PhysicalManager;
      friend class ReductionManager;
      friend class ListReductionManager;
      friend class FoldReducitonManager;
      std::map<UniqueID,TaskUser> users;
      std::map<UniqueID,TaskUser> added_users;
      std::map<Event,CopyUser>    copy_users;
      std::map<Event,CopyUser>    added_copy_users;
#ifdef LEGION_SPY
      std::map<UniqueID,TaskUser> deleted_users;
      std::map<Event,CopyUser>    deleted_copy_users;
#endif
      bool to_be_invalidated;
    };

    /////////////////////////////////////////////////////////////
    // Instance Reference 
    /////////////////////////////////////////////////////////////
    /**
     * Used for passing around references to InstanceViews
     */
    class InstanceRef {
    public:
      InstanceRef(void);
      InstanceRef(Event ready, Memory loc, PhysicalInstance inst, 
                  PhysicalView *v, bool copy = false, Lock lock = Lock::NO_LOCK);
      InstanceRef(PhysicalManager *m, Event ready, Memory loc, PhysicalInstance inst,
                  bool copy = false, Lock lock = Lock::NO_LOCK);
    public:
      inline bool is_virtual_ref(void) const { return (!location.exists()); }
      inline Event get_ready_event(void) const { return ready_event; }
      inline bool has_required_lock(void) const { return required_lock.exists(); }
      inline Lock get_required_lock(void) const { return required_lock; }
      inline PhysicalInstance get_instance(void) const { return instance; }
      inline PhysicalManager* get_manager(void) const { return manager; }
      inline Memory get_memory(void) const { return manager->get_location(); }
      inline bool is_reduction_ref(void) const { return is_reduction; }
      void remove_reference(UniqueID uid, bool strict);
    private:
      friend class RegionTreeForest;
      Event ready_event;
      Lock required_lock;
      Memory location;
      PhysicalInstance instance;
      bool copy;
      bool is_reduction;
      PhysicalView *view;
      PhysicalManager *manager;
    };

    /////////////////////////////////////////////////////////////
    // Escaped User 
    /////////////////////////////////////////////////////////////
    struct EscapedUser {
    public:
      EscapedUser(void)
        : user(0) { }
      EscapedUser(InstanceKey k, UniqueID uid)
        : view_key(k), user(uid) { }
    public:
      bool operator==(const EscapedUser &rhs) const;
      bool operator<(const EscapedUser &rhs) const;
    public:
      InstanceKey view_key;
      UniqueID user;
    };

    /////////////////////////////////////////////////////////////
    // Escaped Copy 
    /////////////////////////////////////////////////////////////
    struct EscapedCopy {
    public:
      EscapedCopy(void)
        : copy_event(Event::NO_EVENT) { }
      EscapedCopy(InstanceKey k, Event copy)
        : view_key(k), copy_event(copy) { }
    public:
      bool operator==(const EscapedCopy &rhs) const;
      bool operator<(const EscapedCopy &rhs) const;
    public:
      InstanceKey view_key;
      Event copy_event;
    };
    
    /////////////////////////////////////////////////////////////
    // Tree Closer 
    /////////////////////////////////////////////////////////////
    class TreeCloser {
    public:
      virtual void pre_siphon(void) = 0;
      virtual void post_siphon(void) = 0; 
      virtual bool closing_state(const RegionTreeNode::FieldState &state) = 0;
      virtual void close_tree_node(RegionTreeNode *node, const FieldMask &closing_mask) = 0;
      virtual bool leave_children_open(void) const = 0;
    };

    /////////////////////////////////////////////////////////////
    // Logical Closer 
    /////////////////////////////////////////////////////////////
    class LogicalCloser : public TreeCloser {
    public:
      LogicalCloser(const LogicalUser &u, ContextID c, std::list<LogicalUser> &users, bool closing_part);
    public:
      virtual void pre_siphon(void);
      virtual void post_siphon(void);
      virtual bool closing_state(const RegionTreeNode::FieldState &state);
      virtual void close_tree_node(RegionTreeNode *node, const FieldMask &closing_mask);
      virtual bool leave_children_open(void) const;
    public:
      const LogicalUser &user;
      ContextID ctx;
      std::list<LogicalUser> &epoch_users;
      bool closing_partition;
    };

    /////////////////////////////////////////////////////////////
    // Physical Closer 
    /////////////////////////////////////////////////////////////
    /**
     * A class for performing a close between physical instances
     * at adjacent (region) levels of the physical region tree.
     * Upper targets are the InstanceViews from the upper level
     * and lower targets are the InstanceViews from the lower level.
     */
    class PhysicalCloser : public TreeCloser {
    public:
      PhysicalCloser(const PhysicalUser &u, RegionMapper &rm, RegionNode *close_target, bool leave_open);
      PhysicalCloser(const PhysicalCloser &rhs, RegionNode *close_target);
      ~PhysicalCloser(void);
    public:
      virtual void pre_siphon(void);
      virtual void post_siphon(void);
      virtual bool closing_state(const RegionTreeNode::FieldState &state);
      virtual void close_tree_node(RegionTreeNode *node, const FieldMask &closing_mask);
      virtual bool leave_children_open(void) const;
    public:
      void pre_region(Color region_color);
      void post_region(void);
    public:
      void pre_partition(Color partition_color);
      void post_partition(void);
    public:
      void add_upper_target(InstanceView *target);
    public:
      const PhysicalUser &user;
      RegionMapper &rm;
      RegionNode *const close_target;
      const bool leave_open;
      bool targets_selected;
      bool success;
      bool partition_valid;
      Color partition_color;
      FieldMask dirty_mask;
      std::vector<InstanceView*> lower_targets;
      std::vector<InstanceView*> upper_targets;
    };

    /////////////////////////////////////////////////////////////
    // Reduction Closer 
    /////////////////////////////////////////////////////////////
    /**
     * A class for closing up all the sub-task reduction instances
     * to a reduction instance class.
     */
    class ReductionCloser : public TreeCloser {
    public:
      ReductionCloser(const PhysicalUser &u, RegionMapper &rm, RegionNode *close_target, ReductionView *target);
      ~ReductionCloser(void);
    public:
      virtual void pre_siphon(void);
      virtual void post_siphon(void);
      virtual bool closing_state(const RegionTreeNode::FieldState &state);
      virtual void close_tree_node(RegionTreeNode *node, const FieldMask &closing_mask);
      virtual bool leave_children_open(void) const;
    public:
      const PhysicalUser &user;
      RegionMapper &rm;
      RegionNode *const close_target;
      ReductionView *const target;
      bool success;
    };

    /////////////////////////////////////////////////////////////
    // Region Analyzer 
    /////////////////////////////////////////////////////////////
    /**
     * Used for doing logical traversals of the region tree.
     */
    class RegionAnalyzer {
    public:
      RegionAnalyzer(ContextID ctx_id, GeneralizedOperation *op, unsigned idx, const RegionRequirement &req);
    public:
      const ContextID ctx;
      GeneralizedOperation *const op;
      const unsigned idx;
      const LogicalRegion start;
      const RegionUsage usage;
      std::vector<FieldID> fields;
      std::vector<Color> path;
    };

    /////////////////////////////////////////////////////////////
    // Region Mapper 
    /////////////////////////////////////////////////////////////
    /**
     * This is the class that is used to do physical traversals
     * of the region tree for both sanitization and mapping
     * physical instances.
     */
    class RegionMapper {
    public:
#ifdef LOW_LEVEL_LOCKS
      RegionMapper(Task *t, UniqueID uid, ContextID id, unsigned idx, const RegionRequirement &req, Mapper *mapper, 
                    Lock mapper_lock, Processor target, Event single, Event multi, 
                    MappingTagID tag, bool sanitizing, bool inline_mapping, 
                    std::vector<InstanceRef> &source_copy);
#else
      RegionMapper(Task *t, UniqueID uid, ContextID id, unsigned idx, const RegionRequirement &req, Mapper *mapper, 
                    ImmovableLock mapper_lock, Processor target, Event single, Event multi, 
                    MappingTagID tag, bool sanitizing, bool inline_mapping, 
                    std::vector<InstanceRef> &source_copy);
#endif
    public:
      ContextID ctx;
      bool sanitizing;
      bool inline_mapping;
      bool success; // for knowing whether a sanitizing walk succeeds or not
      bool final_closing;
      unsigned idx;
      const RegionRequirement &req;
      Task *task;
      UniqueID uid;
#ifdef LOW_LEVEL_LOCKS
      Lock mapper_lock;
#else
      ImmovableLock mapper_lock;
#endif
      Mapper *mapper;
      MappingTagID tag;
      Processor target;
      Event single_term;
      Event multi_term;
      std::vector<unsigned> path;
      // Vector for tracking source copy references, note it's a reference
      std::vector<InstanceRef> &source_copy_instances;
      // The resulting InstanceRef (if any)
      InstanceRef result;
    };

  }; // namespace HighLevel
}; // namespace LegionRuntime

#endif // __REGION_TREE_H__

