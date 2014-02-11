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


#ifndef __LEGION_REGION_TREE_H__
#define __LEGION_REGION_TREE_H__

#include "legion_types.h"
#include "legion_utilities.h"
#include "garbage_collection.h"
#include "field_tree.h"

namespace LegionRuntime {
  namespace HighLevel {
    
    /**
     * \class RegionTreeForest
     * "In the darkness of the forest resides the one true magic..."
     * Most of the magic in Legion is encoded in the RegionTreeForest
     * class and its children.  This class manages both the shape and 
     * states of the region tree.  We use fine-grained locking on 
     * individual nodes and the node look-up tables to enable easy 
     * updates to the shape of the tree.  Each node has a lock that 
     * protects the pointers to its child nodes.  There is a creation 
     * lock that protects the look-up tables.  The logical and physical
     * states of each of the nodes are stored using deques which can
     * be appended to without worrying about resizing so we don't 
     * require any locks for accessing state.  Each logical and physical
     * task context must maintain its own external locking mechanism
     * for serializing access to its logical and physical states.
     *
     * Modifications to the region tree shape are accompanied by a 
     * runtime mask which says which nodes have seen the update.  The
     * forest will record which nodes have sent updates and then 
     * tell the runtime to send updates to the other nodes which
     * have not observed the updates.
     */
    class RegionTreeForest {
    public:
      RegionTreeForest(Runtime *rt);
      RegionTreeForest(const RegionTreeForest &rhs);
      ~RegionTreeForest(void);
    public:
      RegionTreeForest& operator=(const RegionTreeForest &rhs);
    public:
      void create_index_space(const Domain &domain);
      void create_index_partition(IndexPartition pid, IndexSpace parent,
          bool disjoint, int part_color,
          const std::map<Color,Domain> &subspaces, Domain color_space);
      bool destroy_index_space(IndexSpace handle, AddressSpaceID source);
      void destroy_index_partition(IndexPartition handle, 
                                   AddressSpaceID source);
    public:
      IndexPartition get_index_partition(IndexSpace parent, Color color);
      IndexSpace get_index_subspace(IndexPartition parent, Color color);
      Domain get_index_space_domain(IndexSpace handle);
      Domain get_index_partition_color_space(IndexPartition p);
      void get_index_space_partition_colors(IndexSpace sp,
                                            std::set<Color> &colors);
      bool is_index_partition_disjoint(IndexPartition p);
      Color get_index_space_color(IndexSpace handle);
      Color get_index_partition_color(IndexPartition handle);
      IndexSpaceAllocator* get_index_space_allocator(IndexSpace handle);
    public:
      void create_field_space(FieldSpace handle);
      void destroy_field_space(FieldSpace handle, AddressSpaceID source);
      // Return true if local is set to true and we actually performed the 
      // allocation.  It is an error if the field already existed and the
      // allocation was not local.
      bool allocate_field(FieldSpace handle, size_t field_size, 
                          FieldID fid, bool local);
      void free_field(FieldSpace handle, FieldID fid, AddressSpaceID source);
      void allocate_fields(FieldSpace handle, const std::vector<size_t> &sizes,
                           const std::vector<FieldID> &resulting_fields);
      void free_fields(FieldSpace handle, const std::set<FieldID> &to_free,
                       AddressSpaceID source);
      void allocate_field_index(FieldSpace handle, size_t field_size, 
                                FieldID fid, unsigned index, 
                                AddressSpaceID source);
      void allocate_field_indexes(FieldSpace handle, 
                                  const std::vector<FieldID> &resulting_fields,
                                  const std::vector<size_t> &sizes,
                                  const std::vector<unsigned> &indexes,
                                  AddressSpaceID source);
      void get_all_fields(FieldSpace handle, std::set<FieldID> &fields);
      void get_all_regions(FieldSpace handle, std::set<LogicalRegion> &regions);
    public:
      void create_logical_region(LogicalRegion handle);
      bool destroy_logical_region(LogicalRegion handle, 
                                  AddressSpaceID source);
      void destroy_logical_partition(LogicalPartition handle,
                                     AddressSpaceID source);
    public:
      LogicalPartition get_logical_partition(LogicalRegion parent, 
                                             IndexPartition handle);
      LogicalPartition get_logical_partition_by_color(
                                  LogicalRegion parent, Color color);
      LogicalPartition get_logical_partition_by_tree(
          IndexPartition handle, FieldSpace space, RegionTreeID tid);
      LogicalRegion get_logical_subregion(LogicalPartition parent,
                                          IndexSpace handle);
      LogicalRegion get_logical_subregion_by_color(
                              LogicalPartition parent, Color color);
      LogicalRegion get_logical_subregion_by_tree(
            IndexSpace handle, FieldSpace space, RegionTreeID tid);
      Color get_logical_region_color(LogicalRegion handle);
      Color get_logical_partition_color(LogicalPartition handle);
    public:
      // Logical analysis methods
      void perform_dependence_analysis(RegionTreeContext ctx, 
                                       Operation *op, unsigned idx,
                                       RegionRequirement &req,
                                       RegionTreePath &path);
      void perform_fence_analysis(RegionTreeContext ctx, Operation *fence,
                                  LogicalRegion handle);
      void analyze_destroy_index_space(RegionTreeContext ctx, 
                    IndexSpace handle, Operation *op, LogicalRegion region);
      void analyze_destroy_index_partition(RegionTreeContext ctx,
                    IndexPartition handle, Operation *op, LogicalRegion region);
      void analyze_destroy_field_space(RegionTreeContext ctx,
                    FieldSpace handle, Operation *op, LogicalRegion region);
      void analyze_destroy_fields(RegionTreeContext ctx,
            FieldSpace handle, const std::set<FieldID> &fields, 
            Operation *op, LogicalRegion region);
      void analyze_destroy_logical_region(RegionTreeContext ctx,
                  LogicalRegion handle, Operation *op, LogicalRegion region);
      void analyze_destroy_logical_partition(RegionTreeContext ctx,
                  LogicalPartition handle, Operation *op, LogicalRegion region);
      void initialize_logical_context(RegionTreeContext ctx, 
                                      LogicalRegion handle);
      void invalidate_logical_context(RegionTreeContext ctx,
                                      LogicalRegion handle);
      void acquire_user_coherence(RegionTreeContext ctx,
                                  LogicalRegion handle,
                                  const std::set<FieldID> &fields);
      void release_user_coherence(RegionTreeContext ctx,
                                  LogicalRegion handle,
                                  const std::set<FieldID> &fields);
    public:
      // Physical analysis methods
      bool premap_physical_region(RegionTreeContext ctx,
                                  RegionTreePath &path,
                                  RegionRequirement &req,
                                  Mappable *mappable,
                                  SingleTask *parent_ctx,
                                  Processor local_proc
#ifdef DEBUG_HIGH_LEVEL
                                  , unsigned index
                                  , const char *log_name
                                  , UniqueID uid
#endif
                                  );
      MappingRef map_physical_region(RegionTreeContext ctx,
                                     RegionTreePath &path,
                                     RegionRequirement &req,
                                     unsigned idx,
                                     Mappable *mappable,
                                     Processor local_proc,
                                     Processor target_proc
#ifdef DEBUG_HIGH_LEVEL
                                     , const char *log_name
                                     , UniqueID uid
#endif
                                     );
      // Note this works without a path which assumes
      // we are remapping exactly the logical region
      // specified by the region requirement
      MappingRef remap_physical_region(RegionTreeContext ctx,
                                       RegionRequirement &req,
                                       unsigned index,
                                       const InstanceRef &ref
#ifdef DEBUG_HIGH_LEVEL
                                       , const char *log_name
                                       , UniqueID uid
#endif
                                       );
      InstanceRef register_physical_region(RegionTreeContext ctx,
                                           const MappingRef &ref,
                                           RegionRequirement &req,
                                           unsigned idx,
                                           Mappable *mappable,
                                           Processor local_proc,
                                           Event term_event
#ifdef DEBUG_HIGH_LEVEL
                                           , const char *log_name
                                           , UniqueID uid
                                           , RegionTreePath &path
#endif
                                           );
      InstanceRef initialize_physical_context(RegionTreeContext ctx,
                    const RegionRequirement &req, PhysicalManager *manager,
                    Event term_event, Processor local_proc, unsigned depth,
                    std::map<PhysicalManager*,PhysicalView*> &top_views);
      void invalidate_physical_context(RegionTreeContext ctx,
                                       LogicalRegion handle);
      Event close_physical_context(RegionTreeContext ctx,
                                   RegionRequirement &req,
                                   Mappable *mappable,
                                   SingleTask *parent_ctx,
                                   Processor local_proc,
                                   const InstanceRef &ref
#ifdef DEBUG_HIGH_LEVEL
                                   , unsigned index
                                   , const char *log_name
                                   , UniqueID uid
#endif
                                   );
      Event copy_across(RegionTreeContext src_ctx, 
                        RegionTreeContext dst_ctx,
                        const RegionRequirement &src_req,
                        const RegionRequirement &dst_req,
                        const InstanceRef &src_ref,
                        const InstanceRef &dst_ref,
                        Event precondition);
    public:
      // Methods for sending and returning state information
      void send_physical_state(RegionTreeContext ctx,
                               const RegionRequirement &req,
                               UniqueID unique_id,
                               AddressSpaceID target,
                               std::set<PhysicalView*> &needed_views,
                               std::set<PhysicalManager*> &needed_managers);
      void send_tree_shape(const IndexSpaceRequirement &req,
                           AddressSpaceID target);
      void send_tree_shape(const RegionRequirement &req,
                           AddressSpaceID target);
      void send_tree_shape(IndexSpace handle, AddressSpaceID target);
      void send_tree_shape(FieldSpace handle, AddressSpaceID target);
      void send_tree_shape(LogicalRegion handle, AddressSpaceID target);
      void send_back_physical_state(RegionTreeContext ctx,
                               RegionTreeContext remote_ctx,
                               RegionTreePath &path,
                               const RegionRequirement &req,
                               AddressSpaceID target,
                               std::set<PhysicalManager*> &needed_managers);
      void send_remote_references(
          const std::set<PhysicalManager*> &needed_managers,
          AddressSpaceID target);
      void send_remote_references(const std::set<PhysicalView*> &needed_views,
          const std::set<PhysicalManager*> &needed_managers, 
          AddressSpaceID target);
      void handle_remote_references(Deserializer &derez);
    public:
      // Debugging method for checking context state
      void check_context_state(RegionTreeContext ctx);
    public:
      IndexSpaceNode* create_node(Domain d, IndexPartNode *par, Color c);
      IndexPartNode*  create_node(IndexPartition p, IndexSpaceNode *par,
                                 Color c, Domain color_space, bool disjoint);
      FieldSpaceNode* create_node(FieldSpace space);
      RegionNode*     create_node(LogicalRegion r, PartitionNode *par);
      PartitionNode*  create_node(LogicalPartition p, RegionNode *par);
    public:
      IndexSpaceNode* get_node(IndexSpace space);
      IndexPartNode*  get_node(IndexPartition part);
      FieldSpaceNode* get_node(FieldSpace space);
      RegionNode*     get_node(LogicalRegion handle);
      PartitionNode*  get_node(LogicalPartition handle);
      RegionNode*     get_tree(RegionTreeID tid);
    public:
      bool has_node(IndexSpace space) const;
      bool has_node(IndexPartition part) const;
      bool has_node(FieldSpace space) const;
      bool has_node(LogicalRegion handle) const;
      bool has_node(LogicalPartition handle) const;
      bool has_tree(RegionTreeID tid) const;
      bool has_field(FieldSpace space, FieldID fid);
    public:
      bool is_disjoint(IndexPartition handle);
      bool is_disjoint(LogicalPartition handle);
      bool are_disjoint(IndexSpace parent, IndexSpace child);
      bool are_disjoint(IndexSpace parent, IndexPartition child);
      bool compute_index_path(IndexSpace parent, IndexSpace child,
                              std::vector<Color> &path);
      bool compute_partition_path(IndexSpace parent, IndexPartition child,
                                  std::vector<Color> &path); 
      void initialize_path(IndexSpace child, IndexSpace parent,
                           RegionTreePath &path);
      void initialize_path(IndexPartition child, IndexSpace parent,
                           RegionTreePath &path);
      void initialize_path(IndexSpace child, IndexPartition parent,
                           RegionTreePath &path);
      void initialize_path(IndexPartition child, IndexPartition parent,
                           RegionTreePath &path);
    public:
      void register_physical_manager(PhysicalManager *manager);
      void unregister_physical_manager(DistributedID did);
      void register_physical_view(PhysicalView *view);
      void unregister_physical_view(DistributedID did);
    public:
      bool has_manager(DistributedID did) const;
      bool has_view(DistributedID did) const;
      PhysicalManager* find_manager(DistributedID did);
      PhysicalView* find_view(DistributedID did);
    protected:
      void initialize_path(IndexTreeNode* child, IndexTreeNode *parent,
                           RegionTreePath &path);
    public:
      template<typename T>
      Color generate_unique_color(const std::map<Color,T> &current_map);
    public:
      void resize_node_contexts(unsigned total_contexts);
#ifdef DEBUG_HIGH_LEVEL
    public:
      // These are debugging methods and are never called from
      // actual code, therefore they never take locks
      void dump_logical_state(LogicalRegion region, ContextID ctx);
      void dump_physical_state(LogicalRegion region, ContextID ctx);
#endif
    public:
      Runtime *const runtime;
    protected:
      Reservation forest_lock;
      Reservation lookup_lock;
      Reservation distributed_lock;
    private:
      // The lookup lock must be held when accessing these
      // data structures
      std::map<IndexSpace,IndexSpaceNode*>     index_nodes;
      std::map<IndexPartition,IndexPartNode*>  index_parts;
      std::map<FieldSpace,FieldSpaceNode*>     field_nodes;
      std::map<LogicalRegion,RegionNode*>     region_nodes;
      std::map<LogicalPartition,PartitionNode*> part_nodes;
      std::map<RegionTreeID,RegionNode*>        tree_nodes;
    private:
      // References to objects stored in the region forest
      std::map<DistributedID,PhysicalManager*> managers;
      std::map<DistributedID,PhysicalView*> views;
#ifdef DYNAMIC_TESTS
    public:
      class DynamicSpaceTest {
      public:
        DynamicSpaceTest(IndexPartNode *parent, Color c1, 
              IndexSpace left, Color c2, IndexSpace right);
        void perform_test(void) const;
      public:
        IndexPartNode *parent;
        Color c1, c2;
        IndexSpace left, right;
      };
      class DynamicPartTest {
      public:
        DynamicPartTest(IndexSpaceNode *parent, Color c1, Color c2);
        void add_child_space(bool left, IndexSpace space);
        void perform_test(void) const;
      public:
        IndexSpaceNode *parent;
        Color c1, c2;
        std::vector<IndexSpace> left, right;
      };
    private:
      Reservation dynamic_lock;
      std::deque<DynamicSpaceTest> dynamic_space_tests;
      std::deque<DynamicPartTest>  dynamic_part_tests;
    public:
      bool perform_dynamic_tests(unsigned num_tests);
      void add_disjointness_test(const DynamicPartTest &test);
#endif
    };

    /**
     * \class IndexTreeNode
     * The abstract base class for nodes in the index space trees.
     */
    class IndexTreeNode {
    public:
      IndexTreeNode(void);
      IndexTreeNode(Color color, unsigned depth, RegionTreeForest *ctx); 
      virtual ~IndexTreeNode(void);
    public:
      virtual IndexTreeNode* get_parent(void) const = 0;
      virtual void send_node(AddressSpaceID target, bool up, bool down) = 0;
    public:
      const unsigned depth;
      const Color color;
      RegionTreeForest *const context;
    public:
      std::set<AddressSpaceID> creation_set;
      std::set<AddressSpaceID> destruction_set;
    protected:
      Reservation node_lock;
    };

    /**
     * \class IndexSpaceNode
     * A class for representing a generic index space node.
     */
    class IndexSpaceNode : public IndexTreeNode {
    public:
      IndexSpaceNode(Domain d, IndexPartNode *par, Color c,
                     RegionTreeForest *ctx);
      IndexSpaceNode(const IndexSpaceNode &rhs);
      virtual ~IndexSpaceNode(void);
    public:
      IndexSpaceNode& operator=(const IndexSpaceNode &rhs);
    public:
      virtual IndexTreeNode* get_parent(void) const;
    public:
      bool has_child(Color c);
      IndexPartNode* get_child(Color c);
      void add_child(IndexPartNode *child);
      void remove_child(Color c);
    public:
      bool are_disjoint(Color c1, Color c2);
      void add_disjoint(Color c1, Color c2);
      Color generate_color(void);
      void get_colors(std::set<Color> &colors);
    public:
      void add_instance(RegionNode *inst);
      bool has_instance(RegionTreeID tid);
      void add_creation_source(AddressSpaceID source);
      void destroy_node(AddressSpaceID source);
#ifdef DYNAMIC_TESTS
      void add_disjointness_tests(IndexPartNode *child,
                const std::vector<IndexSpaceNode*> &children);
#endif
    public:
      virtual void send_node(AddressSpaceID target, bool up, bool down);
      static void handle_node_creation(RegionTreeForest *context,
                                       Deserializer &derez, 
                                       AddressSpaceID source);
    public:
      IndexSpaceAllocator* get_allocator(void);
    public:
      const Domain domain;
      const IndexSpace handle;
      IndexPartNode *const parent;
    private:
      // Must hold the node lock when accessing the
      // remaining data structures
      // Color map is all children seen ever
      std::map<Color,IndexPartNode*> color_map;
      // Valid map is all chidlren that haven't been deleted
      std::map<Color,IndexPartNode*> valid_map;
      std::set<RegionNode*> logical_nodes;
      std::set<std::pair<Color,Color> > disjoint_subsets;
    private:
      IndexSpaceAllocator *allocator;
    };

    /**
     * \class IndexPartNode
     * A node for representing a generic index partition.
     */
    class IndexPartNode : public IndexTreeNode { 
    public:
      IndexPartNode(IndexPartition p, IndexSpaceNode *par,
                    Color c, Domain color_space, bool dis,
                    RegionTreeForest *ctx);
      IndexPartNode(const IndexPartNode &rhs);
      virtual ~IndexPartNode(void);
    public:
      IndexPartNode& operator=(const IndexPartNode &rhs);
    public:
      virtual IndexTreeNode* get_parent(void) const;
    public:
      bool has_child(Color c);
      IndexSpaceNode* get_child(Color c);
      void add_child(IndexSpaceNode *child);
      void remove_child(Color c);
    public:
      bool are_disjoint(Color c1, Color c2);
      void add_disjoint(Color c1, Color c2);
    public:
      void add_instance(PartitionNode *inst);
      bool has_instance(RegionTreeID tid);
      void add_creation_source(AddressSpaceID source);
      void destroy_node(AddressSpaceID source);
#ifdef DYNAMIC_TESTS
      void add_disjointness_tests(IndexPartNode *child,
              const std::vector<IndexSpaceNode*> &children);
#endif
    public:
      virtual void send_node(AddressSpaceID target, bool up, bool down);
      static void handle_node_creation(RegionTreeForest *context,
                                       Deserializer &derez, 
                                       AddressSpaceID source);
    public:
      const IndexPartition handle;
      const Domain color_space;
      IndexSpaceNode *parent;
      const bool disjoint;
    private:
      // Must hold the node lock when accessing
      // the remaining data structures
      std::map<Color,IndexSpaceNode*> color_map;
      std::map<Color,IndexSpaceNode*> valid_map;
      std::set<PartitionNode*> logical_nodes;
      std::set<std::pair<Color,Color> > disjoint_subspaces;
    };

    /**
     * \class FieldSpaceNode
     * Represent a generic field space that can be
     * pointed at by nodes in the region trees.
     */
    class FieldSpaceNode {
    public:
      struct FieldInfo {
      public:
        FieldInfo(void) : field_size(0), idx(0), 
                          local(false), destroyed(false) { }
        FieldInfo(size_t size, unsigned id, bool loc)
          : field_size(size), idx(id), local(loc), destroyed(false) { }
      public:
        size_t field_size;
        unsigned idx;
        bool local;
        bool destroyed;
      };
    public:
      FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx);
      FieldSpaceNode(const FieldSpaceNode &rhs);
      ~FieldSpaceNode(void);
    public:
      FieldSpaceNode& operator=(const FieldSpaceNode &rhs);
    public:
      void allocate_field(FieldID fid, size_t size, bool local);
      void allocate_field_index(FieldID fid, size_t size, 
                                AddressSpaceID runtime, unsigned index);
      void free_field(FieldID fid, AddressSpaceID source);
      bool has_field(FieldID fid);
      size_t get_field_size(FieldID fid);
      void get_all_fields(std::set<FieldID> &to_set);
      void get_all_regions(std::set<LogicalRegion> &regions);
    public:
      void add_instance(RegionNode *inst);
      bool has_instance(RegionTreeID tid);
      void add_creation_source(AddressSpaceID source);
      void destroy_node(AddressSpaceID source);
    public:
      void transform_field_mask(FieldMask &mask, AddressSpaceID source);
      FieldMask get_field_mask(const std::set<FieldID> &fields) const;
      void get_field_indexes(const std::set<FieldID> &fields,
                             std::map<unsigned,FieldID> &indexes) const;
    public:
      InstanceManager* create_instance(Memory location, Domain dom,
                                       const std::set<FieldID> &fields,
                                       size_t blocking_factor, unsigned depth,
                                       RegionNode *node);
      ReductionManager* create_reduction(Memory location, Domain dom,
                                        FieldID fid, bool reduction_list,
                                        RegionNode *node, ReductionOpID redop);
    public:
      void send_node(AddressSpaceID target);
      static void handle_node_creation(RegionTreeForest *context,
                                       Deserializer &derez, 
                                       AddressSpaceID target);
    public:
      // Help with debug printing
      char* to_string(const FieldMask &mask) const;
      void to_field_set(const FieldMask &mask,
                        std::set<FieldID> &field_set) const;
    protected:
      // Assume we are already holding the node lock
      // when calling these methods
      unsigned allocate_index(bool local, int goal=-1);
      void free_index(unsigned index);
    public:
      const FieldSpace handle;
      RegionTreeForest *const context;
    public:
      std::set<AddressSpaceID> creation_set;
      std::set<AddressSpaceID> destruction_set;
    private:
      Reservation node_lock;
      // Top nodes in the trees for which this field space is used
      std::set<RegionNode*> logical_nodes;
      std::map<FieldID,FieldInfo> fields;
      FieldMask allocated_indexes;
      /*
       * Every field space contains a permutation transformer that
       * can translate a field mask from any other node onto
       * this node.
       */
      std::map<AddressSpaceID,FieldPermutation> transformers;
    };
 
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
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      UniqueID uid;
#endif
    public:
      static const int TIMEOUT = DEFAULT_LOGICAL_USER_TIMEOUT;
    };

    /**
     * \struct PhysicalUser
     * A class for representing physical users of a logical
     * region including necessary information to 
     * register execution dependences on the user.
     */
    struct PhysicalUser : public GenericUser {
    public:
      PhysicalUser(void);
      PhysicalUser(const RegionUsage &u, const FieldMask &m,
                   Event term_event, int child = -1);
    public:
      Event term_event;
      int child;
    };

    /**
     * \struct CloseInfo
     * A struct containing information about how to close
     * a child node including the close mask and whether
     * the child can be kept open in read mode.
     */
    struct CloseInfo {
    public:
      CloseInfo(void)
        : target_child(-1), close_mask(FieldMask()), 
          leave_open(false), allow_next(false) { }
      CloseInfo(int child, const FieldMask &m, bool open, bool allow)
        : target_child(child), close_mask(m), 
          leave_open(open), allow_next(allow) { }
    public:
      int target_child;
      FieldMask close_mask;
      bool leave_open;
      bool allow_next;
    };

    /**
     * \struct MappableInfo
     */
    struct MappableInfo {
    public:
      MappableInfo(ContextID ctx, Mappable *mappable,
                   Processor local_proc, RegionRequirement &req,
                   const FieldMask &traversal_mask);
    public:
      const ContextID ctx;
      Mappable *const mappable;
      const Processor local_proc;
      RegionRequirement &req;
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
      FieldMask valid_fields;
      std::map<Color,FieldMask> open_children;
    };

    /**
     * \struct FieldState
     * Track the field state more accurately
     * for logical traversals to figure out 
     * which tasks can run in parallel.
     */
    struct FieldState : public ChildState {
    public:
      FieldState(const GenericUser &u, const FieldMask &m, Color child);
    public:
      bool overlaps(const FieldState &rhs) const;
      void merge(const FieldState &rhs);
    public:
      void print_state(TreeStateLogger *logger, 
                       const FieldMask &capture_mask) const;
    public:
      OpenState open_state;
      ReductionOpID redop;
    }; 

    /**
     * \struct LogicalState
     * Track the version states for a given logical
     * region as well as the previous and current
     * epoch users and any close operations that
     * needed to be performed.
     */
    struct LogicalState {
    public:
      LogicalState(void);
      LogicalState(const LogicalState &state);
      ~LogicalState(void);
    public:
      LogicalState& operator=(const LogicalState &rhs);
    public:
      void reset(void);
    public:
      std::map<VersionID,FieldMask> field_versions;
      std::list<FieldState> field_states;
#ifndef LOGICAL_FIELD_TREE
      std::list<LogicalUser> curr_epoch_users;
      std::list<LogicalUser> prev_epoch_users;
#else
      FieldTree<LogicalUser> *curr_epoch_users;
      FieldTree<LogicalUser> *prev_epoch_users;
#endif
      std::map<Color,std::list<CloseInfo> > close_operations;
      // Fields on which the user has 
      // asked for explicit coherence
      FieldMask user_level_coherence;
    };

    /**
     * \class LogicalDepAnalyzer
     * A class for use with doing logical dependence
     * analysis on field tree data structures.
     */
    class LogicalDepAnalyzer {
    public:
      LogicalDepAnalyzer(const LogicalUser &user,
                         const FieldMask &check_mask,
                         bool validates_regions);
    public:
      bool analyze(LogicalUser &user);
      FieldMask get_dominator_mask(void) const;
    public:
      inline void begin_node(FieldTree<LogicalUser> *node) { }
      inline void end_node(FieldTree<LogicalUser> *node) { }
    private:
      const LogicalUser user;
      const bool validates_regions;
      FieldMask dominator_mask;
      FieldMask observed_mask;
    };

    class LogicalOpAnalyzer {
    public:
      LogicalOpAnalyzer(Operation *op);
    public:
      bool analyze(LogicalUser &user);
    public:
      inline void begin_node(FieldTree<LogicalUser> *node) { }
      inline void end_node(FieldTree<LogicalUser> *node) { }
    public:
      Operation *const op;
    };

    /**
     * \class LogicalFilter
     * A class for helping with filtering logical users
     * out of a field tree data structure.
     */
    class LogicalFilter {
    public:
      LogicalFilter(const FieldMask &filter_mask,
                    FieldTree<LogicalUser> *target = NULL);
    public:
      bool analyze(LogicalUser &user);
    public:
      void begin_node(FieldTree<LogicalUser> *node);
      void end_node(FieldTree<LogicalUser> *node);
    private:
      const FieldMask filter_mask;
      FieldTree<LogicalUser> *const target;
      std::deque<LogicalUser> reinsert;
      unsigned reinsert_count;
      std::deque<unsigned> reinsert_stack;
    };

    /**
     * \class LogicalFieldInvalidator
     */
    class LogicalFieldInvalidator {
    public:
      LogicalFieldInvalidator(void) { }
    public:
      bool analyze(const LogicalUser &user);
    public:
      inline void begin_node(FieldTree<LogicalUser> *node) { }
      inline void end_node(FieldTree<LogicalUser> *node) { }
    };

    /**
     * \struct LogicalCloser
     * This structure helps keep track of the state
     * necessary for performing a close operation
     * on the logical region tree.
     */
    struct LogicalCloser {
    public:
      LogicalCloser(ContextID ctx, const LogicalUser &u,
                    bool validates);
#ifdef LOGICAL_FIELD_TREE
    public:
      bool analyze(LogicalUser &user);
    public:
      void begin_node(FieldTree<LogicalUser> *node);
      void end_node(FieldTree<LogicalUser> *node);
#endif
    public:
      ContextID ctx;
      const LogicalUser &user;
      bool validates;
      // All the fields that we close for this traversal
      FieldMask closed_mask;
      std::deque<LogicalUser> closed_users;
      std::deque<CloseInfo> close_operations;
#ifdef LOGICAL_FIELD_TREE
    public:
      bool current;
      bool has_non_dominator;
      FieldMask local_closing_mask;
      FieldMask local_non_dominator_mask;
      std::deque<LogicalUser> reinsert;
      unsigned reinsert_count;
      std::deque<unsigned> reinsert_stack;
#endif
    }; 

    /**
     * \struct PhysicalState
     * Track the physical state of a logical region
     * including which fields have dirty data,
     * which children are open, and the valid
     * reduction and instance views.
     */
    struct PhysicalState {
    public:
      PhysicalState(void);
      PhysicalState(ContextID ctx);
#ifdef DEBUG_HIGH_LEVEL
      PhysicalState(ContextID ctx, RegionTreeNode *node);
#endif
    public:
      FieldMask dirty_mask;
      FieldMask reduction_mask;
      ChildState children;
      std::map<InstanceView*,FieldMask> valid_views;
      std::map<ReductionView*,FieldMask> reduction_views;
      std::set<Color> complete_children;
    public:
      // These are used for managing access to the physical state
      unsigned acquired_count;
      bool exclusive;
      std::deque<std::pair<UserEvent,bool/*exclusive*/> > requests;
    public:
      ContextID ctx;
#ifdef DEBUG_HIGH_LEVEL
      RegionTreeNode *node;
#endif
    }; 

    /**
     * \struct PhysicalCloser
     * Class for helping with the closing of physical region trees
     */
    struct PhysicalCloser {
    public:
      PhysicalCloser(MappableInfo *info,
                     bool leave_open,
                     LogicalRegion closing_handle);
      PhysicalCloser(const PhysicalCloser &rhs);
      ~PhysicalCloser(void);
    public:
      PhysicalCloser& operator=(const PhysicalCloser &rhs);
    public:
      bool needs_targets(void) const;
      void add_target(InstanceView *target);
      void close_tree_node(RegionTreeNode *node, 
                           const FieldMask &closing_mask);
      const std::vector<InstanceView*>& get_upper_targets(void) const;
      const std::vector<InstanceView*>& get_lower_targets(void) const;
    public:
      void update_dirty_mask(const FieldMask &mask);
      const FieldMask& get_dirty_mask(void) const;
      void update_node_views(RegionTreeNode *node, PhysicalState &state);
    public:
      MappableInfo *const info;
      const LogicalRegion handle;
      bool permit_leave_open;
    protected:
      bool targets_selected;
      FieldMask dirty_mask;
      std::vector<InstanceView*> upper_targets;
      std::vector<InstanceView*> lower_targets;
    }; 

    /**
     * \class PhysicalDepAnalyzer
     * A class for helping with doing physical dependence 
     * analysis on a physical field tree data structure.
     * In the process it also filters out any users which
     * should be moved back to the next epoch.
     */
    template<bool FILTER>
    class PhysicalDepAnalyzer {
    public:
      PhysicalDepAnalyzer(const PhysicalUser &user,
                          const FieldMask &check_mask,
                          RegionTreeNode *logical_node,
                          std::set<Event> &wait_on);
    public:
      bool analyze(PhysicalUser &user);
      const FieldMask& get_observed_mask(void) const;
      const FieldMask& get_non_dominated_mask(void) const;
    public:
      void begin_node(FieldTree<PhysicalUser> *node);
      void end_node(FieldTree<PhysicalUser> *node);
    public:
      void insert_filtered_users(FieldTree<PhysicalUser> *target);
    private:
      const PhysicalUser user;
      RegionTreeNode *const logical_node;
      std::set<Event> &wait_on;
      FieldMask non_dominated;
      FieldMask observed;
    private:
      std::deque<PhysicalUser> reinsert;
      unsigned reinsert_count;
      std::deque<unsigned> reinsert_stack;
    private:
      std::deque<PhysicalUser> filtered_users;
    };

    /**
     * \class PhysicalFilter
     * A class for helping with doing filtering of 
     * physical users of a physical user field tree.
     */
    class PhysicalFilter {
    public:
      PhysicalFilter(const FieldMask &filter_mask);
    public:
      bool analyze(PhysicalUser &user);
    public:
      void begin_node(FieldTree<PhysicalUser> *node);
      void end_node(FieldTree<PhysicalUser> *node);
    private:
      const FieldMask filter_mask;
      std::deque<PhysicalUser> reinsert;
      unsigned reinsert_count;
      std::deque<unsigned> reinsert_stack;
    };

    /**
     * \class PhysicalEventFilter
     * A class for helping with garbage collection
     * of users from the previous and current lists
     * after they have completed.
     */
    class PhysicalEventFilter {
    public:
      PhysicalEventFilter(Event term)
        : term_event(term) { }
    public:
      inline bool analyze(const PhysicalUser &user)
      {
        if (user.term_event == term_event)
          return false;
        else
          return true;
      }
    public:
      inline void begin_node(FieldTree<PhysicalUser> *node) { }
      inline void end_node(FieldTree<PhysicalUser> *node) { }
    private:
      const Event term_event;
    };

    /**
     * \class PhysicalCopyAnalyzer
     * A class for helping with doing dependence analysis
     * for copy operations in physical user field trees.
     */
    template<bool READING, bool REDUCE, bool TRACK, bool ABOVE>
    class PhysicalCopyAnalyzer {
    public:
      PhysicalCopyAnalyzer(const FieldMask &copy_mask,
                           ReductionOpID redop,
                           std::set<Event> &wait_on, 
                           int color = -1,
                           RegionTreeNode *logical_node = NULL);
    public:
      bool analyze(const PhysicalUser &user);
      inline const FieldMask& get_non_dominated_mask(void) const 
        { return non_dominated; }
    public:
      inline void begin_node(FieldTree<PhysicalUser> *node) { }
      inline void end_node(FieldTree<PhysicalUser> *node) { }
    private:
      const FieldMask copy_mask;
      const ReductionOpID redop;
      const int local_color;
      RegionTreeNode *const logical_node;
    private:
      std::set<Event> &wait_on;
      FieldMask non_dominated;
    };

    /**
     * \class WARAnalyzer
     * This class helps in doing write-after-read
     * checks on the physical field tree data structure
     * that stores the current epoch users.
     */
    template<bool ABOVE>
    class WARAnalyzer {
    public:
      WARAnalyzer(int color = -1, 
                  RegionTreeNode *node = NULL); 
    public:
      inline void begin_node(FieldTree<PhysicalUser> *node) { }
      inline void end_node(FieldTree<PhysicalUser> *node) { }
    public:
      bool analyze(const PhysicalUser &user);
    public:
      inline bool has_war_dependence(void) const { return has_war; }
    private:
      const int local_color;
      RegionTreeNode *const logical_node;
    private:
      bool has_war;
    };

    /**
     * \class PhysicalUnpacker
     * This class helps in restructuring and transforming
     * field trees after they have been unpacked on a 
     * remote node.
     */
    class PhysicalUnpacker {
    public:
      PhysicalUnpacker(FieldSpaceNode *field_node, AddressSpaceID source,
                       std::map<Event,FieldMask> &deferred_events);
    public:
      void begin_node(FieldTree<PhysicalUser> *node);
      void end_node(FieldTree<PhysicalUser> *node);
    public:
      bool analyze(PhysicalUser &user);
    private:
      FieldSpaceNode *const field_node;
      const AddressSpaceID source;
      std::map<Event,FieldMask> &deferred_events;
    private:
      std::deque<PhysicalUser> reinsert;
      unsigned reinsert_count;
      std::deque<unsigned> reinsert_stack;
    };

    /**
     * \class LegionStack
     * A special runtime class for keeping track of both physical
     * and logical states.  The stack objects are never allowed
     * to shrink.  They are always maintained in a consistent state
     * so they can be accessed even when being appended to.  We assume
     * that there is only one appender at a time.  Access time is O(1).
     */
    template<typename T, int MAX_SIZE, int INC_SIZE>
    class LegionStack {
    public:
      LegionStack(void);
      LegionStack(const LegionStack<T,MAX_SIZE,INC_SIZE> &rhs);
      ~LegionStack(void);
    public:
      LegionStack<T,MAX_SIZE,INC_SIZE>& operator=(
                          const LegionStack<T,MAX_SIZE,INC_SIZE> &rhs);
      T& operator[](unsigned int idx);
    public:
      void append(unsigned int append_count);
      size_t size(void) const;
    private:
      T* ptr_buffer[(MAX_SIZE+INC_SIZE-1)/INC_SIZE];
      size_t buffer_size;
      size_t remaining;
    };

    /**
     * \class RegionTreeNode
     * A generic region tree node from which all
     * other kinds of region tree nodes inherit.  Notice
     * that all important analyses are defined on 
     * this kind of node making them general across
     * all kinds of node types.
     */
    class RegionTreeNode {
    public:
      RegionTreeNode(RegionTreeForest *ctx);
      virtual ~RegionTreeNode(void);
    public:
      void reserve_contexts(unsigned num_contexts);
      LogicalState& get_logical_state(ContextID ctx);
      PhysicalState& acquire_physical_state(ContextID ctx, bool exclusive);
      void acquire_physical_state(PhysicalState &state, bool exclusive);
      bool release_physical_state(PhysicalState &state);
    public:
      // Logical traversal operations
      void register_logical_node(ContextID ctx,
                                 const LogicalUser &user,
                                 RegionTreePath &path);
      void open_logical_node(ContextID ctx,
                             const LogicalUser &user,
                             RegionTreePath &path);
      void close_logical_node(LogicalCloser &closer,
                              const FieldMask &closing_mask);
      bool siphon_logical_children(LogicalCloser &closer,
                                   LogicalState &state,
                                   const FieldMask &closing_mask,
                                   bool record_close_operations,
                                   int next_child = -1);
      void perform_close_operations(LogicalCloser &closer,
                                    const FieldMask &closing_mask,
                                    FieldState &closing_state,
                                    int next_child, bool allow_next_child,
                                    bool upgrade_next_child, 
                                    bool permit_leave_open,
                                    bool record_close_operations,
                                    std::deque<FieldState> &new_states,
                                    FieldMask &need_open);
      void merge_new_field_state(LogicalState &state, 
                                 const FieldState &new_state);
      void merge_new_field_states(LogicalState &state, 
                                  const std::deque<FieldState> &new_states);
      void record_field_versions(LogicalState &state, RegionTreePath &path,
                                 const FieldMask &field_mask, 
                                 unsigned depth, bool before);
      void record_close_operations(LogicalState &state, RegionTreePath &path,
                                  const FieldMask &field_mask, unsigned depth);
      void update_close_operations(LogicalState &state, 
                                   const std::deque<CloseInfo> &close_ops);
      void advance_field_versions(LogicalState &state, const FieldMask &mask);
      void filter_prev_epoch_users(LogicalState &state, const FieldMask &mask);
      void filter_curr_epoch_users(LogicalState &state, const FieldMask &mask);
      void filter_close_operations(LogicalState &state, const FieldMask &mask);
      void sanity_check_logical_state(LogicalState &state);
      void initialize_logical_state(ContextID ctx);
      void invalidate_logical_state(ContextID ctx);
      void register_logical_dependences(ContextID ctx, Operation *op,
                                        const FieldMask &field_mask);
      void record_user_coherence(ContextID ctx, FieldMask &coherence_mask);
      void acquire_user_coherence(ContextID ctx, 
                                  const FieldMask &coherence_mask);
      void release_user_coherence(ContextID ctx, 
                                  const FieldMask &coherence_mask);
    public:
      // Physical traversal operations
      // Entry
      void close_physical_node(PhysicalCloser &closer,
                               const FieldMask &closing_mask);
      bool select_close_targets(PhysicalCloser &closer,
                                const FieldMask &closing_mask,
                                bool complete,
                          const std::map<InstanceView*,FieldMask> &valid_views,
                          std::map<InstanceView*,FieldMask> &update_views);
      bool siphon_physical_children(PhysicalCloser &closer,
                                    PhysicalState &state,
                                    const FieldMask &closing_mask,
                                    int next_child, 
                                    bool allow_next);
      bool close_physical_child(PhysicalCloser &closer,
                                PhysicalState &state,
                                const FieldMask &closing_mask,
                                Color target_child,
                                int next_child,
                                bool allow_next);
      void find_valid_instance_views(PhysicalState &state,
                                     const FieldMask &valid_mask,
                                     const FieldMask &space_mask, 
                                     bool needs_space,
                           std::map<InstanceView*,FieldMask> &valid_views);
      void find_valid_reduction_views(PhysicalState &state,
                                      const FieldMask &valid_mask,
                                      std::set<ReductionView*> &valid_views);
      void pull_valid_instance_views(PhysicalState &state,
                                     const FieldMask &mask);
      // Since figuring out how to issue copies is expensive, try not
      // to hold the physical state lock when doing them.
      void issue_update_copies(MappableInfo *info,
                               InstanceView *target, 
                               FieldMask copy_mask,
                      const std::map<InstanceView*,FieldMask> &valid_instances);
      void issue_update_reductions(PhysicalView *target,
                                   const FieldMask &update_mask,
                                   Processor local_proc,
                    const std::map<ReductionView*,FieldMask> &valid_reductions);
      void invalidate_instance_views(PhysicalState &state,
                                     const FieldMask &invalid_mask, bool clean);
      void invalidate_reduction_views(PhysicalState &state,
                                      const FieldMask &invalid_mask);
      void update_valid_views(PhysicalState &state, const FieldMask &valid_mask,
                              bool dirty, InstanceView *new_view);
      void update_valid_views(PhysicalState &state, const FieldMask &valid_mask,
                              const FieldMask &dirty_mask, 
                              const std::vector<InstanceView*> &new_views);
      void update_reduction_views(PhysicalState &state, 
                                  const FieldMask &valid_mask,
                                  ReductionView *new_view);
      void flush_reductions(const FieldMask &flush_mask,
                            ReductionOpID redop, MappableInfo *info);
      // Entry
      void initialize_physical_state(ContextID ctx);
      // Entry
      void invalidate_physical_state(ContextID ctx);
      // Entry
      void invalidate_physical_state(ContextID ctx, 
                                     const FieldMask &invalid_mask);
    public:
      virtual unsigned get_depth(void) const = 0;
      virtual unsigned get_color(void) const = 0;
      virtual RegionTreeNode* get_parent(void) const = 0;
      virtual RegionTreeNode* get_tree_child(Color c) = 0;
      virtual bool are_children_disjoint(Color c1, Color c2) = 0;
      virtual bool is_region(void) const = 0;
      virtual bool visit_node(PathTraverser *traverser) = 0;
      virtual bool visit_node(NodeTraverser *traverser) = 0;
      virtual Domain get_domain(void) const = 0;
      virtual InstanceView* create_instance(Memory target_mem,
                                            const std::set<FieldID> &fields,
                                            size_t blocking_factor,
                                            unsigned depth) = 0;
      virtual ReductionView* create_reduction(Memory target_mem,
                                              FieldID fid,
                                              bool reduction_list,
                                              ReductionOpID redop) = 0;
      virtual void send_node(AddressSpaceID target) = 0;
      virtual void print_logical_context(ContextID ctx, 
                                         TreeStateLogger *logger,
                                         const FieldMask &mask) = 0;
      virtual void print_physical_context(ContextID ctx, 
                                          TreeStateLogger *logger,
                                          const FieldMask &mask) = 0;
#ifdef DEBUG_HIGH_LEVEL
    public:
      // These methods are only ever called by a debugger
      virtual void dump_logical_context(ContextID ctx, 
                                        TreeStateLogger *logger,
                                        const FieldMask &mask) = 0;
      virtual void dump_physical_context(ContextID ctx, 
                                         TreeStateLogger *logger,
                                         const FieldMask &mask) = 0;
#endif
    public:
      bool pack_send_state(ContextID ctx, Serializer &rez, 
                           AddressSpaceID target,
                           const FieldMask &send_mask,
                           std::set<PhysicalView*> &needed_views,
                           std::set<PhysicalManager*> &needed_managers);
      bool pack_send_back_state(ContextID ctx, Serializer &rez,
                                AddressSpaceID target, const FieldMask &send_mask,
                                std::set<PhysicalManager*> &needed_managers);
      void unpack_send_state(ContextID ctx, Deserializer &derez, 
                             FieldSpaceNode *column, AddressSpaceID source);
    public:
#ifndef LOGICAL_FIELD_TREE
      // Logical helper operations
      static FieldMask perform_dependence_checks(const LogicalUser &user, 
            std::list<LogicalUser> &users, const FieldMask &check_mask, 
            bool validates_regions);
#else
      static FieldMask perform_dependence_checks(const LogicalUser &user,
            FieldTree<LogicalUser> *users, const FieldMask &check_mask,
            bool validates_regions);
#endif
    public:
      RegionTreeForest *const context;
    public:
      std::set<AddressSpaceID> creation_set;
      std::set<AddressSpaceID> destruction_set;
    protected:
      Reservation node_lock;
      LegionStack<LogicalState,MAX_CONTEXTS,DEFAULT_CONTEXTS> logical_states;
      LegionStack<PhysicalState,MAX_CONTEXTS,DEFAULT_CONTEXTS> physical_states;
#ifdef DEBUG_HIGH_LEVEL
      // Uses these for debugging to avoid races accessing
      // the logical and physical deques to check for size
      // when they are possibly growing and in an inconsistent state
      size_t logical_state_size;
      size_t physical_state_size;
#endif
    };

    /**
     * \class RegionNode
     * Represent a region in a region tree
     */
    class RegionNode : public RegionTreeNode {
    public:
      RegionNode(LogicalRegion r, PartitionNode *par, IndexSpaceNode *row_src,
                 FieldSpaceNode *col_src, RegionTreeForest *ctx);
      RegionNode(const RegionNode &rhs);
      virtual ~RegionNode(void);
    public:
      RegionNode& operator=(const RegionNode &rhs);
    public:
      bool has_child(Color c);
      PartitionNode* get_child(Color c);
      void add_child(PartitionNode *child);
      void remove_child(Color c);
      void add_creation_source(AddressSpaceID source);
      void destroy_node(AddressSpaceID source);
    public:
      virtual unsigned get_depth(void) const;
      virtual unsigned get_color(void) const;
      virtual RegionTreeNode* get_parent(void) const;
      virtual RegionTreeNode* get_tree_child(Color c);
      virtual bool are_children_disjoint(Color c1, Color c2);
      virtual bool is_region(void) const;
      virtual bool visit_node(PathTraverser *traverser);
      virtual bool visit_node(NodeTraverser *traverser);
      virtual Domain get_domain(void) const;
      virtual InstanceView* create_instance(Memory target_mem,
                                            const std::set<FieldID> &fields,
                                            size_t blocking_factor,
                                            unsigned depth);
      virtual ReductionView* create_reduction(Memory target_mem,
                                              FieldID fid,
                                              bool reduction_list,
                                              ReductionOpID redop);
      virtual void send_node(AddressSpaceID target);
      static void handle_node_creation(RegionTreeForest *context,
                            Deserializer &derez, AddressSpaceID source);
    public:
      // Logging calls
      virtual void print_logical_context(ContextID ctx, 
                                         TreeStateLogger *logger,
                                         const FieldMask &mask);
      virtual void print_physical_context(ContextID ctx, 
                                          TreeStateLogger *logger,
                                          const FieldMask &mask);
      void print_logical_state(LogicalState &state,
                               const FieldMask &capture_mask,
                               std::map<Color,FieldMask> &to_traverse,
                               TreeStateLogger *logger);
      void print_physical_state(PhysicalState &state,
                                const FieldMask &capture_mask,
                                std::map<Color,FieldMask> &to_traverse,
                                TreeStateLogger *logger);
#ifdef DEBUG_HIGH_LEVEL
    public:
      // These methods are only ever called by a debugger
      virtual void dump_logical_context(ContextID ctx, 
                                        TreeStateLogger *logger,
                                        const FieldMask &mask);
      virtual void dump_physical_context(ContextID ctx, 
                                         TreeStateLogger *logger,
                                         const FieldMask &mask);
#endif
    public:
      void remap_region(ContextID ctx, InstanceView *view, 
                        const FieldMask &user_mask, FieldMask &needed_mask);
      InstanceRef register_region(MappableInfo *info, 
                                  PhysicalUser &user,
                                  PhysicalView *view,
                                  const FieldMask &needed_fields);
      InstanceRef seed_state(ContextID ctx, PhysicalUser &user,
                             PhysicalView *new_view,
                             Processor local_proc);
      Event close_state(MappableInfo *info, PhysicalUser &user,
                        const InstanceRef &target);
    public:
      bool send_state(ContextID ctx, UniqueID uid, AddressSpaceID target,
                      const FieldMask &send_mask, bool invalidate,
                      std::set<PhysicalView*> &needed_views,
                      std::set<PhysicalManager*> &needed_managers);
      static void handle_send_state(RegionTreeForest *context,
                                    Deserializer &derez, 
                                    AddressSpaceID source);
    public:
      bool send_back_state(ContextID ctx, ContextID remote_ctx,
                           AddressSpaceID target,
                           bool invalidate, const FieldMask &send_mask,
                           std::set<PhysicalManager*> &needed_managers);
      static void handle_send_back_state(RegionTreeForest *context,
                           Deserializer &derez, AddressSpaceID source);
    public:
      const LogicalRegion handle;
      PartitionNode *const parent;
      IndexSpaceNode *const row_source;
      FieldSpaceNode *const column_source;
    protected:
      std::map<Color,PartitionNode*> color_map;
      std::map<Color,PartitionNode*> valid_map;
    };

    /**
     * \class PartitionNode
     * Represent an instance of a partition in a region tree.
     */
    class PartitionNode : public RegionTreeNode {
    public:
      PartitionNode(LogicalPartition p, RegionNode *par, 
                    IndexPartNode *row_src, FieldSpaceNode *col_src,
                    RegionTreeForest *ctx);
      PartitionNode(const PartitionNode &rhs);
      ~PartitionNode(void);
    public:
      PartitionNode& operator=(const PartitionNode &rhs);
    public:
      bool has_child(Color c);
      RegionNode* get_child(Color c);
      void add_child(RegionNode *child);
      void remove_child(Color c);
      void add_creation_source(AddressSpaceID source);
      void destroy_node(AddressSpaceID source);
    public:
      virtual unsigned get_depth(void) const;
      virtual unsigned get_color(void) const;
      virtual RegionTreeNode* get_parent(void) const;
      virtual RegionTreeNode* get_tree_child(Color c);
      virtual bool are_children_disjoint(Color c1, Color c2);
      virtual bool is_region(void) const;
      virtual bool visit_node(PathTraverser *traverser);
      virtual bool visit_node(NodeTraverser *traverser);
      virtual Domain get_domain(void) const;
      virtual InstanceView* create_instance(Memory target_mem,
                                            const std::set<FieldID> &fields,
                                            size_t blocking_factor,
                                            unsigned depth);
      virtual ReductionView* create_reduction(Memory target_mem,
                                              FieldID fid,
                                              bool reduction_list,
                                              ReductionOpID redop);
      virtual void send_node(AddressSpaceID target);
    public:
      // Logging calls
      virtual void print_logical_context(ContextID ctx, 
                                         TreeStateLogger *logger,
                                         const FieldMask &mask);
      virtual void print_physical_context(ContextID ctx, 
                                          TreeStateLogger *logger,
                                          const FieldMask &mask);
      void print_logical_state(LogicalState &state,
                               const FieldMask &capture_mask,
                               std::map<Color,FieldMask> &to_traverse,
                               TreeStateLogger *logger);
      void print_physical_state(PhysicalState &state,
                                const FieldMask &capture_mask,
                                std::map<Color,FieldMask> &to_traverse,
                                TreeStateLogger *logger);
#ifdef DEBUG_HIGH_LEVEL
    public:
      // These methods are only ever called by a debugger
      virtual void dump_logical_context(ContextID ctx, 
                                        TreeStateLogger *logger,
                                        const FieldMask &mask);
      virtual void dump_physical_context(ContextID ctx, 
                                         TreeStateLogger *logger,
                                         const FieldMask &mask);
#endif
    public:
      bool send_state(ContextID ctx, UniqueID uid, AddressSpaceID target,
                      const FieldMask &send_mask, bool invalidate,
                      std::set<PhysicalView*> &needed_views,
                      std::set<PhysicalManager*> &needed_managers);
      static void handle_send_state(RegionTreeForest *context,
                                    Deserializer &derez, 
                                    AddressSpaceID source);
    public:
      bool send_back_state(ContextID ctx, ContextID remote_ctx,
                           AddressSpaceID target,
                           bool invalidate, const FieldMask &send_mask,
                           std::set<PhysicalManager*> &needed_managers);
      static void handle_send_back_state(RegionTreeForest *context,
                           Deserializer &derez, AddressSpaceID source);
    public:
      const LogicalPartition handle;
      RegionNode *const parent;
      IndexPartNode *const row_source;
      FieldSpaceNode *const column_source;
      const bool disjoint;
      const bool complete;
    protected:
      std::map<Color,RegionNode*> color_map;
      std::map<Color,RegionNode*> valid_map;
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
      void register_child(unsigned depth, Color color);
    public:
      bool has_child(unsigned depth) const;
      Color get_child(unsigned depth) const;
      unsigned get_path_length(void) const;
    public:
      void record_close_operation(unsigned depth, const CloseInfo &info,
                                  const FieldMask &close_mask);
      void record_before_version(unsigned depth, VersionID vid,
                                 const FieldMask &version_mask);
      void record_after_version(unsigned depth, VersionID vid,
                                const FieldMask &version_mask);
    public:
      const std::deque<CloseInfo>& get_close_operations(unsigned depth) const;
    protected:
      std::vector<int> path;
      std::vector<std::deque<CloseInfo> > close_operations;
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
      Color next_child;
    };

    /**
     * \class NodeTraverser
     * An abstract class which provides the needed
     * functionality for visiting a node in the tree
     * and all of its sub-nodes.
     */
    class NodeTraverser {
    public:
      virtual bool visit_only_valid(void) const = 0;
      virtual bool visit_region(RegionNode *node) = 0;
      virtual bool visit_partition(PartitionNode *node) = 0;
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
                       const FieldMask &field_mask);
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
    };

    /**
     * \class LogicalInitializer
     * A class for initializing logical contexts
     */
    class LogicalInitializer : public NodeTraverser {
    public:
      LogicalInitializer(ContextID ctx);
      LogicalInitializer(const LogicalInitializer &rhs);
      ~LogicalInitializer(void);
    public:
      LogicalInitializer& operator=(const LogicalInitializer &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
    };

    /**
     * \class LogicalInvalidator
     * A class for invalidating logical contexts
     */
    class LogicalInvalidator : public NodeTraverser {
    public:
      LogicalInvalidator(ContextID ctx);
      LogicalInvalidator(const LogicalInvalidator &rhs);
      ~LogicalInvalidator(void);
    public:
      LogicalInvalidator& operator=(const LogicalInvalidator &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
    };

    /**
     * \class RestrictedTraverser
     * A class for checking for user-level software coherence
     * on restricted logical regions.
     */
    class RestrictedTraverser : public PathTraverser {
    public:
      RestrictedTraverser(ContextID ctx, RegionTreePath &path);
      RestrictedTraverser(const RestrictedTraverser &rhs);
      virtual ~RestrictedTraverser(void);
    public:
      RestrictedTraverser& operator=(const RestrictedTraverser &rhs);
    public:
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    public:
      const FieldMask& get_coherence_mask(void) const;
    protected:
      const ContextID ctx;
      FieldMask coherence_mask;
    };

    /**
     * \class PhysicalInitializer
     * A class for initializing physical contexts
     */
    class PhysicalInitializer : public NodeTraverser {
    public:
      PhysicalInitializer(ContextID ctx);
      PhysicalInitializer(const PhysicalInitializer &rhs);
      ~PhysicalInitializer(void);
    public:
      PhysicalInitializer& operator=(const PhysicalInitializer &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
    };

    /**
     * \class PhysicalInvalidator
     * A class for invalidating physical contexts
     */
    class PhysicalInvalidator : public NodeTraverser {
    public:
      PhysicalInvalidator(ContextID ctx);
      PhysicalInvalidator(ContextID ctx, const FieldMask &invalid_mask);
      PhysicalInvalidator(const PhysicalInvalidator &rhs);
      ~PhysicalInvalidator(void);
    public:
      PhysicalInvalidator& operator=(const PhysicalInvalidator &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
      const bool total;
      const FieldMask invalid_mask;
    };

    /**
     * \class ReductionCloser
     * A class for performing reduciton close operations
     */
    class ReductionCloser : public NodeTraverser {
    public:
      ReductionCloser(ContextID ctx, ReductionView *target,
                      const FieldMask &reduc_mask,
                      Processor local_proc);
      ReductionCloser(const ReductionCloser &rhs);
      ~ReductionCloser(void);
    public:
      ReductionCloser& operator=(const ReductionCloser &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
      ReductionView *const target;
      const FieldMask close_mask;
      const Processor local_proc;
    };

    /**
     * \class PremapTraverser
     * A traverser of the physical region tree for
     * performing the premap operation.
     */
    class PremapTraverser : public PathTraverser {
    public:
      PremapTraverser(RegionTreePath &path, MappableInfo *info);  
      PremapTraverser(const PremapTraverser &rhs); 
      ~PremapTraverser(void);
    public:
      PremapTraverser& operator=(const PremapTraverser &rhs);
    public:
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      bool perform_close_operations(RegionTreeNode *node,
                                    LogicalRegion closing_handle);
    protected:
      MappableInfo *const info;
    }; 

    /**
     * \class StateSender
     * This class is used for traversing the region
     * tree to figure out which states need to be sent back
     */
    class StateSender : public NodeTraverser {
    public:
      StateSender(ContextID ctx, UniqueID uid, AddressSpaceID target,
                  std::set<PhysicalView*> &needed_views,
                  std::set<PhysicalManager*> &needed_managers,
                  const FieldMask &send_mask, bool invalidate);
      StateSender(const StateSender &rhs);
      ~StateSender(void);
    public:
      StateSender& operator=(const StateSender &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    public:
      const ContextID ctx;
      const UniqueID uid;
      const AddressSpaceID target;
      std::set<PhysicalView*> &needed_views;
      std::set<PhysicalManager*> &needed_managers;
      const FieldMask send_mask;
      const bool invalidate;
    };

    /**
     * \class PathReturner
     * This class is used for sending back select paths
     * of physical state for merging between where a region
     * mapped and where it's projection requirement initially
     * had privileges.
     */
    class PathReturner : public PathTraverser {
    public:
      PathReturner(RegionTreePath &path, ContextID ctx, 
                   RegionTreeContext remote_ctx, AddressSpaceID target,
                   const FieldMask &return_mask,
                   std::set<PhysicalManager*> &needed_managers);
      PathReturner(const PathReturner &rhs);
      ~PathReturner(void);
    public:
      PathReturner& operator=(const PathReturner &rhs);
    public:
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    public:
      const ContextID ctx;
      const ContextID remote_ctx;
      const AddressSpaceID target;
      const FieldMask return_mask;
      std::set<PhysicalManager*> &needed_managers;
    };

    /**
     * \class StateReturner
     * This class is used for returning state back to a
     * context on the original node for a task.
     */
    class StateReturner : public NodeTraverser {
    public:
      StateReturner(ContextID ctx, RegionTreeContext remote_ctx,
                    AddressSpaceID target, bool invalidate,
                    const FieldMask &return_mask,
                    std::set<PhysicalManager*> &needed_managers);
      StateReturner(const StateReturner &rhs);
      ~StateReturner(void);
    public:
      StateReturner& operator=(const StateReturner &rhs);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    public:
      const ContextID ctx;
      const ContextID remote_ctx;
      const AddressSpaceID target;
      const bool invalidate;
      const FieldMask return_mask;
      std::set<PhysicalManager*> &needed_managers;
    };
 
    /**
     * \class PhysicalManager
     * This class abstracts a physical instance in memory
     * be it a normal instance or a reduction instance.
     */
    class PhysicalManager : public DistributedCollectable {
    public:
      PhysicalManager(RegionTreeForest *ctx, DistributedID did,
                      AddressSpaceID owner_space, AddressSpaceID local_space,
                      Memory mem, PhysicalInstance inst);
      virtual ~PhysicalManager(void);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_accessor(void) const = 0;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_field_accessor(FieldID fid) const = 0;
      virtual bool is_reduction_manager(void) const = 0;
      virtual InstanceManager* as_instance_manager(void) const = 0;
      virtual ReductionManager* as_reduction_manager(void) const = 0;
      virtual size_t get_instance_size(void) const = 0;
      virtual void notify_activate(void);
      virtual void garbage_collect(void) = 0;
      virtual void notify_valid(void);
      virtual void notify_invalid(void) = 0;
      virtual void notify_new_remote(AddressSpaceID sid);
    public:
      inline PhysicalInstance get_instance(void) const
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(instance.exists());
#endif
        return instance;
      }
    public:
      RegionTreeForest *const context;
      const Memory memory;
    protected:
      PhysicalInstance instance;
    };

    /**
     * \class InstanceManager
     * A class for managing normal physical instances
     */
    class InstanceManager : public PhysicalManager {
    public:
      InstanceManager(RegionTreeForest *ctx, DistributedID did,
                      AddressSpaceID owner_space, AddressSpaceID local_space,
                      Memory mem, PhysicalInstance inst, RegionNode *node,
                      const FieldMask &mask, size_t blocking_factor,
                      const std::map<FieldID,Domain::CopySrcDstField> &infos,
                      const std::map<unsigned,FieldID> &indexes,
                      Event use_event, unsigned depth);
      InstanceManager(const InstanceManager &rhs);
      virtual ~InstanceManager(void);
    public:
      InstanceManager& operator=(const InstanceManager &rhs);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_accessor(void) const;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_field_accessor(FieldID fid) const;
      virtual bool is_reduction_manager(void) const;
      virtual InstanceManager* as_instance_manager(void) const;
      virtual ReductionManager* as_reduction_manager(void) const;
      virtual size_t get_instance_size(void) const;
      virtual void garbage_collect(void);
      virtual void notify_invalid(void);
    public:
      inline Event get_use_event(void) const { return use_event; }
    public:
      InstanceView* create_top_view(unsigned depth);
      void compute_copy_offsets(const FieldMask &copy_mask,
                                std::vector<Domain::CopySrcDstField> &fields);
      void compute_copy_offsets(const std::vector<FieldID> &copy_fields,
                                std::vector<Domain::CopySrcDstField> &fields);
    public:
      DistributedID send_manager(AddressSpaceID target, 
                                 std::set<PhysicalManager*> &needed_managers);
    public:
      static void handle_send_manager(RegionTreeForest *context, 
                                      AddressSpaceID source,
                                      Deserializer &derez);
    public:
      void pack_manager(Serializer &rez);
      static InstanceManager* unpack_manager(Deserializer &derez,
                                             RegionTreeForest *context, 
                                             DistributedID did,
                                             bool make = true);
    public:
      void add_valid_view(InstanceView *view);
      void remove_valid_view(InstanceView *view);
      bool match_instance(size_t field_size, const Domain &dom) const;
      bool match_instance(const std::vector<size_t> &fields_sizes,
                          const Domain &dom, const size_t bf) const;
    public:
      RegionNode *const region_node;
      const FieldMask allocated_fields;
      const size_t blocking_factor;
      // Event that needs to trigger before we can start using
      // this physical instance.
      const Event use_event;
      const unsigned depth;
    protected:
      const std::map<FieldID,Domain::CopySrcDstField> field_infos;
      // Remember these indexes are only good on the local node and
      // have to be transformed when the manager is sent remotely
      const std::map<unsigned/*index*/,FieldID> field_indexes;
    protected:
      // Keep track of whether we've recycled this instance or not
      bool recycled;
      // Keep a set of the views we need to see when recycling
      std::set<InstanceView*> valid_views;
    };

    /**
     * \class ReductionManager
     * An abstract class for managing reduction physical instances
     */
    class ReductionManager : public PhysicalManager {
    public:
      ReductionManager(RegionTreeForest *ctx, DistributedID did,
                       AddressSpaceID owner_space, AddressSpaceID local_space,
                       Memory mem, PhysicalInstance inst, 
                       RegionNode *region_node, ReductionOpID redop, 
                       const ReductionOp *op);
      virtual ~ReductionManager(void);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_accessor(void) const = 0;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_field_accessor(FieldID fid) const = 0;
      virtual bool is_reduction_manager(void) const;
      virtual InstanceManager* as_instance_manager(void) const;
      virtual ReductionManager* as_reduction_manager(void) const;
      virtual size_t get_instance_size(void) const = 0;
      virtual void garbage_collect(void);
      virtual void notify_invalid(void);
    public:
      virtual bool is_foldable(void) const = 0;
      virtual void find_field_offsets(const FieldMask &reduce_mask,
          std::vector<Domain::CopySrcDstField> &fields) = 0;
      virtual Event issue_reduction(
          const std::vector<Domain::CopySrcDstField> &src_fields,
          const std::vector<Domain::CopySrcDstField> &dst_fields,
          Domain space, Event precondition, bool reduction_fold) = 0;
      virtual Domain get_pointer_space(void) const = 0;
    public:
      DistributedID send_manager(AddressSpaceID target, 
                        std::set<PhysicalManager*> &needed_managers);
    public:
      static void handle_send_manager(RegionTreeForest *context,
                                      AddressSpaceID source,
                                      Deserializer &derez);
    public:
      void pack_manager(Serializer &rez);
      static ReductionManager* unpack_manager(Deserializer &derez,
                            RegionTreeForest *context, 
                            DistributedID did, bool make = true);
    public:
      ReductionView* create_view(void);
    public:
      const ReductionOp *const op;
      const ReductionOpID redop;
      RegionNode *const region_node;
    };

    /**
     * \class ListReductionManager
     * A class for storing list reduction instances
     */
    class ListReductionManager : public ReductionManager {
    public:
      ListReductionManager(RegionTreeForest *ctx, DistributedID did,
                           AddressSpaceID owner_space, 
                           AddressSpaceID local_space,
                           Memory mem, PhysicalInstance inst, 
                           RegionNode *node, ReductionOpID redop, 
                           const ReductionOp *op, Domain dom);
      ListReductionManager(const ListReductionManager &rhs);
      virtual ~ListReductionManager(void);
    public:
      ListReductionManager& operator=(const ListReductionManager &rhs);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_accessor(void) const;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_field_accessor(FieldID fid) const;
      virtual size_t get_instance_size(void) const;
    public:
      virtual bool is_foldable(void) const;
      virtual void find_field_offsets(const FieldMask &reduce_mask,
          std::vector<Domain::CopySrcDstField> &fields);
      virtual Event issue_reduction(
          const std::vector<Domain::CopySrcDstField> &src_fields,
          const std::vector<Domain::CopySrcDstField> &dst_fields,
          Domain space, Event precondition, bool reduction_fold);
      virtual Domain get_pointer_space(void) const;
    protected:
      const Domain ptr_space;
    };

    /**
     * \class FoldReductionManager
     * A class for representing fold reduction instances
     */
    class FoldReductionManager : public ReductionManager {
    public:
      FoldReductionManager(RegionTreeForest *ctx, DistributedID did,
                           AddressSpaceID owner_space, 
                           AddressSpaceID local_space,
                           Memory mem, PhysicalInstance inst, 
                           RegionNode *node, ReductionOpID redop, 
                           const ReductionOp *op);
      FoldReductionManager(const FoldReductionManager &rhs);
      virtual ~FoldReductionManager(void);
    public:
      FoldReductionManager& operator=(const FoldReductionManager &rhs);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_accessor(void) const;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_field_accessor(FieldID fid) const;
      virtual size_t get_instance_size(void) const;
    public:
      virtual bool is_foldable(void) const;
      virtual void find_field_offsets(const FieldMask &reduce_mask,
          std::vector<Domain::CopySrcDstField> &fields);
      virtual Event issue_reduction(
          const std::vector<Domain::CopySrcDstField> &src_fields,
          const std::vector<Domain::CopySrcDstField> &dst_fields,
          Domain space, Event precondition, bool reduction_fold);
      virtual Domain get_pointer_space(void) const;
    };

    /**
     * \class PhysicalView
     * This class abstracts a view on a physical instance
     * in memory.  Physical views are reference counted
     * and will delete themselves once they no longer have
     * any valid handles.
     */
    class PhysicalView : public HierarchicalCollectable {
    public:
      PhysicalView(RegionTreeForest *ctx, DistributedID did,
                   AddressSpaceID owner_proc, DistributedID own_did,
                   RegionTreeNode *node);
      virtual ~PhysicalView(void);
    public:
      virtual bool is_reduction_view(void) const = 0;
      virtual InstanceView* as_instance_view(void) const = 0;
      virtual ReductionView* as_reduction_view(void) const = 0;
      virtual PhysicalManager* get_manager(void) const = 0;
    public:
      virtual void add_copy_user(ReductionOpID redop, Event copy_term,
                                 const FieldMask &mask, bool reading,
                                 Processor exec_proc) = 0;
      virtual InstanceRef add_user(PhysicalUser &user,
                                   Processor exec_proc) = 0;
      virtual bool reduce_to(ReductionOpID redop, 
                             const FieldMask &reduce_mask,
                             std::set<Event> &preconditions,
                     std::vector<Domain::CopySrcDstField> &src_fields) = 0;
    public:
      virtual void notify_activate(void) = 0;
      virtual void garbage_collect(void) = 0;
      virtual void notify_valid(void) = 0;
      virtual void notify_invalid(void) = 0;
    public:
      void defer_collect_user(Event term_event, const FieldMask &mask,
                              Processor p, bool gc_epoch);
      virtual void collect_user(Event term_event,
                                const FieldMask &term_mask) = 0;
      static void handle_deferred_collect(Deserializer &derez);
    public:
      void send_back_user(const PhysicalUser &user);
      virtual void process_send_back_user(AddressSpaceID source,
                                          PhysicalUser &user) = 0;
      static void handle_send_back_user(RegionTreeForest *context,
                                        Deserializer &derez,
                                        AddressSpaceID source);
    public:
      void send_user(AddressSpaceID target, DistributedID target_did,
                     const PhysicalUser &user);
      virtual void process_send_user(AddressSpaceID source,
                                     PhysicalUser &user) = 0;
      static void handle_send_user(RegionTreeForest *context,
                                   Deserializer &derez,
                                   AddressSpaceID source);
    public:
      RegionTreeForest *const context;
      RegionTreeNode *const logical_node;
    protected:
      Reservation view_lock;
    };

    /**
     * \class InstanceView
     * The InstanceView class is used for providing views
     * onto instance managers from a given logical perspective.
     */
    class InstanceView : public PhysicalView {
    public:
      InstanceView(RegionTreeForest *ctx, DistributedID did,
                   AddressSpaceID owner_proc, DistributedID own_did,
                   RegionTreeNode *node, InstanceManager *manager,
                   InstanceView *parent, unsigned depth);
      InstanceView(const InstanceView &rhs);
      virtual ~InstanceView(void);
    public:
      InstanceView& operator=(const InstanceView &rhs);
    public:
      Memory get_location(void) const;
      size_t get_blocking_factor(void) const;
      InstanceView* get_subview(Color c);
      void add_subview(InstanceView *view, Color c);
      const FieldMask& get_physical_mask(void) const;
    public:
      void copy_to(const FieldMask &copy_mask, 
                   std::set<Event> &preconditions,
                   std::vector<Domain::CopySrcDstField> &dst_fields);
      void copy_from(const FieldMask &copy_mask, 
                     std::set<Event> &preconditions,
                     std::vector<Domain::CopySrcDstField> &src_fields);
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask,
                     std::set<Event> &preconditions,
                     std::vector<Domain::CopySrcDstField> &dst_fields);
      bool has_war_dependence(const RegionUsage &usage, 
                              const FieldMask &user_mask);
      void accumulate_events(std::set<Event> &all_events);
    public:
      virtual bool is_reduction_view(void) const;
      virtual InstanceView* as_instance_view(void) const;
      virtual ReductionView* as_reduction_view(void) const;
      virtual PhysicalManager* get_manager(void) const;
    public:
      virtual void add_copy_user(ReductionOpID redop, Event copy_term,
                                 const FieldMask &mask, bool reading,
                                 Processor exec_proc);
      virtual InstanceRef add_user(PhysicalUser &user,
                                   Processor exec_proc);
    public:
      virtual void notify_activate(void);
      virtual void garbage_collect(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
      virtual void collect_user(Event term_event,
                                const FieldMask &term_mask);
      virtual void process_send_back_user(AddressSpaceID source,
                                          PhysicalUser &user);
      virtual void process_send_user(AddressSpaceID source,
                                     PhysicalUser &user); 
    protected:
      void add_user_above(std::set<Event> &wait_on, PhysicalUser &user);
      template<bool ABOVE>
      void add_local_user(std::set<Event> &wait_on, 
                          const PhysicalUser &user);
    protected:
      void find_copy_preconditions(std::set<Event> &wait_on, 
                                   bool writing, ReductionOpID redop, 
                                   const FieldMask &mask);
      void find_copy_preconditions_above(Color child_color,
                                   std::set<Event> &wait_on,
                                   bool writing, ReductionOpID redop,
                                   const FieldMask &copy_mask);
      template<bool ABOVE>
      void find_local_copy_preconditions(std::set<Event> &wait_on,
                                   bool writing, ReductionOpID redop,
                                   const FieldMask &copy_mask,
                                   int local_color);
      bool has_war_dependence_above(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    Color child_color);
      void update_versions(const FieldMask &update_mask);
      void filter_local_users(Event term_event,
                              const FieldMask &term_mask);
      void notify_subscribers(std::set<AddressSpaceID> &notified, 
                              const PhysicalUser &user);
      void condense_user_list(std::list<PhysicalUser> &users);
    public:
      DistributedID send_state(AddressSpaceID target,
                      std::set<PhysicalView*> &needed_views,
                      std::set<PhysicalManager*> &needed_managers);
      DistributedID send_back_state(AddressSpaceID target,
                      std::set<PhysicalManager*> &needed_managers);
    protected:
      void pack_instance_view(Serializer &rez);
      void unpack_instance_view(Deserializer &derez, AddressSpaceID source);
    public:
      static void handle_send_instance_view(RegionTreeForest *context, 
                                            Deserializer &derez,
                                            AddressSpaceID source);
      static void handle_send_back_instance_view(
                      RegionTreeForest *context, Deserializer &derez,
                      AddressSpaceID source);
    public:
      InstanceManager *const manager;
      InstanceView *const parent;
      const unsigned depth;
    protected:
      // The lock for the instance shared between all views
      // of a physical instance within a context.  The top
      // most view is responsible for deleting the lock
      // when it is reclaimed.
      Reservation inst_lock;
      // Keep track of the child views
      std::map<Color,InstanceView*> children;
      // These are the sets of users in the current and next epochs
      // for performing dependence analysis
#ifndef PHYSICAL_FIELD_TREE
      std::list<PhysicalUser> curr_epoch_users;
      std::list<PhysicalUser> prev_epoch_users;
#else
      FieldTree<PhysicalUser> *const curr_epoch_users;
      FieldTree<PhysicalUser> *const prev_epoch_users;
#endif
      // Keep track of how many outstanding references we have
      // for each of the user events
      std::set<Event> event_references;
      // Version information for each of the fields
      std::map<VersionID,FieldMask> current_versions;
    };

    /**
     * \class ReductionView
     * The ReductionView class is used for providing a view
     * onto reduction physical instances from any logical perspective.
     */
    class ReductionView : public PhysicalView {
    public:
      ReductionView(RegionTreeForest *ctx, DistributedID did,
                    AddressSpaceID owner_proc, DistributedID own_did,
                    RegionTreeNode *node, ReductionManager *manager);
      ReductionView(const ReductionView &rhs);
      virtual ~ReductionView(void);
    public:
      ReductionView& operator=(const ReductionView&rhs);
    public:
      void perform_reduction(PhysicalView *target, 
                             const FieldMask &copy_mask, Processor local_proc);
    public:
      virtual bool is_reduction_view(void) const;
      virtual InstanceView* as_instance_view(void) const;
      virtual ReductionView* as_reduction_view(void) const;
      virtual PhysicalManager* get_manager(void) const;
    public:
      virtual void add_copy_user(ReductionOpID redop, Event copy_term,
                                 const FieldMask &mask, bool reading,
                                 Processor exec_proc);
      virtual InstanceRef add_user(PhysicalUser &user,
                                   Processor exec_proc);
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask,
                     std::set<Event> &preconditions,
                     std::vector<Domain::CopySrcDstField> &dst_fields);
    public:
      void reduce_from(ReductionOpID redop, const FieldMask &reduce_mask,
                       std::set<Event> &preconditions,
                       std::vector<Domain::CopySrcDstField> &src_fields);
      void notify_subscribers(const PhysicalUser &user, 
                              int skip = -1);
    public:
      virtual void notify_activate(void);
      virtual void garbage_collect(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
      virtual void collect_user(Event term_event,
                                const FieldMask &term_mask);
      virtual void process_send_back_user(AddressSpaceID source,
                                          PhysicalUser &user);
      virtual void process_send_user(AddressSpaceID source,
                                     PhysicalUser &user);
    public:
      DistributedID send_state(AddressSpaceID target,
                               std::set<PhysicalView*> &needed_views,
                               std::set<PhysicalManager*> &needed_managers);
      DistributedID send_back_state(AddressSpaceID target,
                               std::set<PhysicalManager*> &needed_managers);
    public:
      void pack_reduction_view(Serializer &rez);
      void unpack_reduction_view(Deserializer &derez, AddressSpaceID source);
    public:
      static void handle_send_reduction_view(RegionTreeForest *context,
                                Deserializer &derez, AddressSpaceID source);
      static void handle_send_back_reduction_view(RegionTreeForest *context,
                                Deserializer &derez, AddressSpaceID source);
    public:
      Memory get_location(void) const;
      ReductionOpID get_redop(void) const;
    public:
      ReductionManager *const manager;
    protected:
      std::list<PhysicalUser> reduction_users;
      std::list<PhysicalUser> reading_users;
      std::set<Event> event_references;
    };

    /**
     * \class ViewHandle
     * The view handle class provides a handle that
     * properly maintains the reference counting property on
     * physical views for garbage collection purposes.
     */
    class ViewHandle {
    public:
      ViewHandle(void);
      ViewHandle(PhysicalView *v);
      ViewHandle(const ViewHandle &rhs);
      ~ViewHandle(void);
    public:
      ViewHandle& operator=(const ViewHandle &rhs);
    public:
      inline bool has_view(void) const { return (view != NULL); }
      inline PhysicalView* get_view(void) const { return view; }
      inline bool is_reduction_view(void) const
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(view != NULL);
#endif
        return view->is_reduction_view();
      }
      inline PhysicalManager* get_manager(void) const
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(view != NULL);
#endif
        return view->get_manager();
      }
    private:
      PhysicalView *view;
    };

    /**
     * \class MappingRef
     * This class keeps a reference to a physical instance that has
     * been allocated and is ready to have dependence analysis performed.
     * Once all the allocations have been performed, then an operation
     * can pass all of the mapping references to the RegionTreeForest
     * to actually perform the operations necessary to make the 
     * region valid and return an InstanceRef.
     */
    class MappingRef {
    public:
      MappingRef(void);
      MappingRef(const ViewHandle &handle, const FieldMask &needed_mask);
    public:
      inline bool has_ref(void) const { return handle.has_view(); }
      inline PhysicalView* get_view(void) const { return handle.get_view(); } 
      inline const FieldMask& get_mask(void) const { return needed_fields; }
    private:
      ViewHandle handle;
      FieldMask needed_fields;
    };

    /**
     * \class InstanceRef
     * A class for keeping track of references to physical instances
     */
    class InstanceRef {
    public:
      InstanceRef(void);
      InstanceRef(Event ready, Reservation lock, const ViewHandle &handle);
    public:
      inline bool has_ref(void) const { return handle.has_view(); }
      inline Event get_ready_event(void) const { return ready_event; }
      inline bool has_required_lock(void) const { return needed_lock.exists(); }
      Reservation get_required_lock(void) const { return needed_lock; }
      const ViewHandle& get_handle(void) const { return handle; }
      Memory get_memory(void) const;
      Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_accessor(void) const;
      Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_field_accessor(FieldID fid) const;
      void add_valid_reference(void);
      void remove_valid_reference(void);
      void pack_reference(Serializer &rez, AddressSpaceID target);
      static InstanceRef unpack_reference(Deserializer &derez,
                                          RegionTreeForest *context,
                                          unsigned depth);
    private:
      Event ready_event;
      Reservation needed_lock;
      ViewHandle handle;
    };

    /**
     * \class MappingTraverser
     * A traverser of the physical region tree for
     * performing the mapping operation.
     */
    class MappingTraverser : public PathTraverser {
    public:
      MappingTraverser(RegionTreePath &path, MappableInfo *info,
                       const RegionUsage &u, const FieldMask &m,
                       Processor target, unsigned idx);
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
      MappableInfo *const info;
      const RegionUsage usage;
      const FieldMask user_mask;
      const Processor target_proc;
      const unsigned index;
    protected:
      MappingRef result;
    }; 

  };
};

#endif // __LEGION_REGION_TREE_H__

// EOF

