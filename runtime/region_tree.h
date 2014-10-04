/* Copyright 2014 Stanford University
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
#include "legion_allocation.h"
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
      void create_index_space(const Domain &hull,
                              const std::set<Domain> &domains);
      void create_index_partition(IndexPartition pid, IndexSpace parent,
          bool disjoint, int part_color,
          const std::map<Color,Domain> &subspaces, Domain color_space);
      void create_index_partition(IndexPartition pid, IndexSpace parent,
          bool disjoint, int part_color,
          const std::map<Color,Domain> &hulls, Domain color_space,
          const std::map<Color,std::set<Domain> > &components);
      bool destroy_index_space(IndexSpace handle, AddressSpaceID source);
      void destroy_index_partition(IndexPartition handle, 
                                   AddressSpaceID source);
    public:
      IndexPartition get_index_partition(IndexSpace parent, Color color);
      IndexSpace get_index_subspace(IndexPartition parent, Color color);
      bool has_multiple_domains(IndexSpace handle);
      Domain get_index_space_domain(IndexSpace handle);
      void get_index_space_domains(IndexSpace handle,
                                   std::vector<Domain> &domains);
      Domain get_index_partition_color_space(IndexPartition p);
      void get_index_space_partition_colors(IndexSpace sp,
                                            std::set<Color> &colors);
      bool is_index_partition_disjoint(IndexPartition p);
      Color get_index_space_color(IndexSpace handle);
      Color get_index_partition_color(IndexPartition handle);
      IndexSpace get_parent_index_space(IndexPartition handle);
      bool has_parent_index_partition(IndexSpace handle);
      IndexPartition get_parent_index_partition(IndexSpace handle);
      IndexSpaceAllocator* get_index_space_allocator(IndexSpace handle);
      size_t get_domain_volume(IndexSpace handle);
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
      size_t get_field_size(FieldSpace handle, FieldID fid);
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
      LogicalRegion get_parent_logical_region(LogicalPartition handle);
      bool has_parent_logical_partition(LogicalRegion handle);
      LogicalPartition get_parent_logical_partition(LogicalRegion handle);
      size_t get_domain_volume(LogicalRegion handle);
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
                    std::map<PhysicalManager*,LogicalView*> &top_views);
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
      Event copy_across(Mappable *mappable,
                        Processor local_proc,
                        RegionTreeContext src_ctx,
                        RegionTreeContext dst_ctx,
                        RegionRequirement &src_req,
                        const RegionRequirement &dst_req,
                        const InstanceRef &dst_ref,
                        Event precondition);
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
                               std::map<LogicalView*,FieldMask> &needed_views,
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
      void send_remote_references(
          const std::map<LogicalView*,FieldMask> &needed_views,
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
      bool are_compatible(IndexSpace left, IndexSpace right);
      bool is_dominated(IndexSpace src, IndexSpace dst);
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
      void register_logical_view(DistributedID did, LogicalView *view);
      void unregister_logical_view(DistributedID did);
    public:
      bool has_manager(DistributedID did) const;
      bool has_view(DistributedID did) const;
      PhysicalManager* find_manager(DistributedID did);
      LogicalView* find_view(DistributedID did);
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
      LegionKeyValue<DistributedID,PhysicalManager*,
                     PHYSICAL_MANAGER_ALLOC>::map managers;
      LegionKeyValue<DistributedID,LogicalView*,
                     LOGICAL_VIEW_ALLOC>::map views;
#ifdef DYNAMIC_TESTS
    public:
      class DynamicSpaceTest {
      public:
        DynamicSpaceTest(IndexPartNode *parent, 
                         IndexSpaceNode *left, 
                         IndexSpaceNode *right);
        void perform_test(void) const;
      public:
        IndexPartNode *parent;
        IndexSpaceNode *left, *right;
      };
      class DynamicPartTest {
      public:
        DynamicPartTest(IndexSpaceNode *parent, 
                        IndexPartNode *left, 
                        IndexPartNode *right);
        void add_child_space(bool left, IndexSpaceNode *node);
        void perform_test(void) const;
      public:
        IndexSpaceNode *parent;
        IndexPartNode *left, *right;
        std::vector<IndexSpaceNode*> left_spaces, right_spaces;
      };
    private:
      Reservation dynamic_lock;
      std::deque<DynamicSpaceTest> dynamic_space_tests;
      std::deque<DynamicPartTest>  dynamic_part_tests;
    public:
      bool perform_dynamic_tests(unsigned num_tests);
      void add_disjointness_test(const DynamicPartTest &test);
      static bool are_disjoint(const Domain &left,
                               const Domain &right);
      static bool are_disjoint(IndexSpaceNode *left,
                               IndexSpaceNode *right);
#endif
#ifdef DEBUG_PERF
    public:
      void record_call(int kind, unsigned long long time);
    protected:
      void begin_perf_trace(int kind);
      void end_perf_trace(unsigned long long tolerance);
    public:
      struct CallRecord {
      public:
        CallRecord(void)
          : kind(0), count(0), total_time(0), max_time(0), min_time(0) { }
        CallRecord(int k)
          : kind(k), count(0), total_time(0), max_time(0), min_time(0) { }
      public:
        inline void record_call(unsigned long long time)
        {
          count++;
          total_time += time;
          if (min_time == 0)
            min_time = time;
          else if (time < min_time)
            min_time = time;
          if (time > max_time)
            max_time = time;
        }
      public:
        int kind;
        int count;
        unsigned long long total_time;
        unsigned long long min_time;
        unsigned long long max_time;
      };
      struct PerfTrace {
      public:
        PerfTrace(void)
          : tracing(false), kind(0) { }
        PerfTrace(int k, unsigned long long start);
      public:
        inline void record_call(int call_kind, unsigned long long time)
        {
          if (tracing)
            records[call_kind].record_call(time);
        }
        void report_trace(unsigned long long diff);
      public:
        bool tracing;
        int kind;
        unsigned long long start;
        std::vector<CallRecord> records;
      };
    protected:
      Reservation perf_trace_lock;
      std::vector<PerfTrace> traces;
#endif
    };

#ifdef DEBUG_PERF
    enum TraceKind {
      REGION_DEPENDENCE_ANALYSIS,  
      PREMAP_PHYSICAL_REGION_ANALYSIS,
      MAP_PHYSICAL_REGION_ANALYSIS,
      REMAP_PHYSICAL_REGION_ANALYSIS,
      REGISTER_PHYSICAL_REGION_ANALYSIS,
      COPY_ACROSS_ANALYSIS,
    };

    enum CallKind {
      CREATE_NODE_CALL,
      GET_NODE_CALL,
      ARE_DISJOINT_CALL,
      COMPUTE_PATH_CALL,
      RESIZE_CONTEXTS_CALL,
      CREATE_INSTANCE_CALL,
      CREATE_REDUCTION_CALL,
      PERFORM_PREMAP_CLOSE_CALL,
      MAPPING_TRAVERSE_CALL,
      MAP_PHYSICAL_REGION_CALL,
      MAP_REDUCTION_REGION_CALL,
      RESERVE_CONTEXTS_CALL,
      ACQUIRE_PHYSICAL_STATE_CALL,
      RELEASE_PHYSICAL_STATE_CALL,
      REGISTER_LOGICAL_NODE_CALL,
      OPEN_LOGICAL_NODE_CALL,
      CLOSE_LOGICAL_NODE_CALL,
      SIPHON_LOGICAL_CHILDREN_CALL,
      PERFORM_LOGICAL_CLOSE_CALL,
      RECORD_CLOSE_CALL,
      UPDATE_CLOSE_CALL,
      ADVANCE_FIELD_CALL,
      FILTER_PREV_EPOCH_CALL,
      FILTER_CURR_EPOCH_CALL,
      FILTER_CLOSE_CALL,
      INITIALIZE_LOGICAL_CALL,
      INVALIDATE_LOGICAL_CALL,
      REGISTER_LOGICAL_DEPS_CALL,
      CLOSE_PHYSICAL_NODE_CALL,
      SELECT_CLOSE_TARGETS_CALL,
      SIPHON_PHYSICAL_CHILDREN_CALL,
      CLOSE_PHYSICAL_CHILD_CALL,
      FIND_VALID_INSTANCE_VIEWS_CALL,
      FIND_VALID_REDUCTION_VIEWS_CALL,
      PULL_VALID_VIEWS_CALL,
      FIND_COPY_ACROSS_INSTANCES_CALL,
      ISSUE_UPDATE_COPIES_CALL,
      ISSUE_UPDATE_REDUCTIONS_CALL,
      PERFORM_COPY_DOMAIN_CALL,
      INVALIDATE_INSTANCE_VIEWS_CALL,
      INVALIDATE_REDUCTION_VIEWS_CALL,
      UPDATE_VALID_VIEWS_CALL,
      UPDATE_REDUCTION_VIEWS_CALL,
      FLUSH_REDUCTIONS_CALL,
      INITIALIZE_PHYSICAL_STATE_CALL,
      INVALIDATE_PHYSICAL_STATE_CALL,
      PERFORM_DEPENDENCE_CHECKS_CALL,
      PERFORM_CLOSING_CHECKS_CALL,
      REMAP_REGION_CALL,
      REGISTER_REGION_CALL,
      CLOSE_PHYSICAL_STATE_CALL,
      GARBAGE_COLLECT_CALL,
      NOTIFY_INVALID_CALL,
      GET_RECYCLE_EVENT_CALL,
      DEFER_COLLECT_USER_CALL,
      GET_SUBVIEW_CALL,
      COPY_FIELD_CALL,
      COPY_TO_CALL,
      REDUCE_TO_CALL,
      COPY_FROM_CALL,
      REDUCE_FROM_CALL,
      HAS_WAR_DEPENDENCE_CALL,
      ACCUMULATE_EVENTS_CALL,
      ADD_COPY_USER_CALL,
      ADD_USER_CALL,
      ADD_USER_ABOVE_CALL,
      ADD_LOCAL_USER_CALL,
      FIND_COPY_PRECONDITIONS_CALL,
      FIND_COPY_PRECONDITIONS_ABOVE_CALL,
      FIND_LOCAL_COPY_PRECONDITIONS_CALL,
      HAS_WAR_DEPENDENCE_ABOVE_CALL,
      UPDATE_VERSIONS_CALL,
      CONDENSE_USER_LIST_CALL,
      PERFORM_REDUCTION_CALL,
      NUM_CALL_KIND,
    };

    class PerfTracer {
    public:
      PerfTracer(RegionTreeForest *f, int k)
        : forest(f), kind(k)
      {
        start = TimeStamp::get_current_time_in_micros();
      }
      ~PerfTracer(void)
      {
        unsigned long long stop = TimeStamp::get_current_time_in_micros();
        unsigned long long diff = stop - start;
        forest->record_call(kind, diff);
      }
    private:
      RegionTreeForest *forest;
      int kind;
      unsigned long long start;
    };
#endif

    /**
     * \class IndexTreeNode
     * The abstract base class for nodes in the index space trees.
     */
    class IndexTreeNode {
    public:
      struct IntersectInfo {
      public:
        IntersectInfo(void)
          : has_intersects(false), own_intersections(false) { }
        IntersectInfo(bool has, bool own)
          : has_intersects(has), own_intersections(own) { }
        IntersectInfo(bool has, bool own, const std::set<Domain> &ds)
          : has_intersects(has), own_intersections(own), intersections(ds) { }
      public:
        bool has_intersects;
        bool own_intersections;
        std::set<Domain> intersections;
      };
    public:
      IndexTreeNode(void);
      IndexTreeNode(Color color, unsigned depth, RegionTreeForest *ctx); 
      virtual ~IndexTreeNode(void);
    public:
      virtual IndexTreeNode* get_parent(void) const = 0;
      virtual void send_node(AddressSpaceID target, bool up, bool down) = 0;
    public:
      static bool compute_intersections(const std::set<Domain> &left,
                                        const std::set<Domain> &right,
                                        std::set<Domain> &result);
      static bool compute_intersections(const std::set<Domain> &left,
                                        const Domain &right,
                                        std::set<Domain> &result);
      static bool compute_intersection(const Domain &left,
                                       const Domain &right,
                                       Domain &result);
      static bool compute_dominates(const std::set<Domain> &left_set,
                                    const std::set<Domain> &right_set);
    public:
      const unsigned depth;
      const Color color;
      RegionTreeForest *const context;
    public:
      NodeMask creation_set;
      NodeMask destruction_set;
    protected:
      Reservation node_lock;
    protected:
      std::map<IndexTreeNode*,IntersectInfo> intersections;
      std::map<IndexTreeNode*,bool> dominators;
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
      size_t get_num_children(void) const;
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
    public:
      bool has_component_domains(void) const;
      void update_component_domains(const std::set<Domain> &domains);
      const std::set<Domain>& get_component_domains(void) const;
      bool intersects_with(IndexSpaceNode *other);
      bool intersects_with(IndexPartNode *other);
      const std::set<Domain>& get_intersection_domains(IndexSpaceNode *other);
      const std::set<Domain>& get_intersection_domains(IndexPartNode *other);
      bool dominates(IndexSpaceNode *other);
      bool dominates(IndexPartNode *other);
#ifdef DYNAMIC_TESTS
    public:
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
      // If we have component domains keep track of those as well
      std::set<Domain> component_domains;
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
      size_t get_num_children(void) const;
    public:
      bool are_disjoint(Color c1, Color c2);
      void add_disjoint(Color c1, Color c2);
      bool is_complete(void);
    public:
      void add_instance(PartitionNode *inst);
      bool has_instance(RegionTreeID tid);
      void add_creation_source(AddressSpaceID source);
      void destroy_node(AddressSpaceID source);
    public:
      void get_subspace_domains(std::set<Domain> &subspaces);
      bool intersects_with(IndexSpaceNode *other);
      bool intersects_with(IndexPartNode *other);
      const std::set<Domain>& get_intersection_domains(IndexSpaceNode *other);
      const std::set<Domain>& get_intersection_domains(IndexPartNode *other);
      bool dominates(IndexSpaceNode *other);
      bool dominates(IndexPartNode *other);
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
      bool has_complete, complete;
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
      unsigned get_field_index(FieldID fid) const;
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
      LayoutDescription* find_layout_description(const FieldMask &mask,
                                                 const Domain &domain,
                                                 size_t blocking_factor);
      LayoutDescription* create_layout_description(const FieldMask &mask,
                                                   const Domain &domain,
                                                   size_t blocking_factor,
                                   const std::set<FieldID> &create_fields,
                                   const std::vector<size_t> &field_sizes,
                                   const std::vector<unsigned> &indexes);
      LayoutDescription* register_layout_description(LayoutDescription *desc);
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
      NodeMask creation_set;
      NodeMask destruction_set;
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
    private:
      // Keep track of the layouts associated with this field space
      // Index them by their hash of their field mask to help
      // differentiate them.
      std::map<FIELD_TYPE,LegionContainer<LayoutDescription*,
                          LAYOUT_DESCRIPTION_ALLOC>::deque> layouts;
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
     * \class TreeCloseImpl
     * The base class for storing information about close
     * operations that need to be performed.
     */
    class TreeCloseImpl : public Collectable {
    public:
      static const AllocationType alloc_type = TREE_CLOSE_IMPL_ALLOC;
    public:
      TreeCloseImpl(int child, const FieldMask &m, bool open);
      TreeCloseImpl(const TreeCloseImpl &rhs);
      ~TreeCloseImpl(void);
    public:
      TreeCloseImpl& operator=(const TreeCloseImpl &rhs);
    public:
      void add_close_op(const FieldMask &close_mask,
                        std::deque<CloseInfo> &needed_ops);
      void return_close_op(const FieldMask &close_mask);
    public:
      const int target_child;
      const bool leave_open;
      // expose this one publicly since we know that all
      // accesses to it will be serialized by the dependence
      // analysis process
      FieldMask remaining_logical;
    private:
      FieldMask remaining_physical;
      Reservation tree_reservation;
    };

    /**
     * \class TreeClose
     * A handle to a close operation that is supports reference
     * counting for knowing when it is safe to delete close
     * operations.
     */
    class TreeClose {
    public:
      TreeClose(void);
      TreeClose(TreeCloseImpl *op);
      TreeClose(const TreeClose &rhs);
      ~TreeClose(void);
    public:
      TreeClose& operator=(const TreeClose &rhs);
    public:
      inline int get_child(void) const { return impl->target_child; }
      inline bool get_leave_open(void) const { return impl->leave_open; }
      inline FieldMask& get_logical_mask(void) const 
      { return impl->remaining_logical; }
    public:
      void add_close_op(const FieldMask &close_mask,
                        std::deque<CloseInfo> &needed_ops);
      void return_close_op(const FieldMask &close_mask);
    private:
      TreeCloseImpl *impl;
    };

    /**
     * \struct CloseInfo
     * A struct containing information about how to close
     * a child node including the close mask and whether
     * the child can be kept open in read mode.
     */
    struct CloseInfo {
    public:
      CloseInfo(void) { }
      CloseInfo(const FieldMask &m, TreeCloseImpl *impl)
        : close_mask(m), close_handle(impl) { }
    public:
      inline int get_child(void) const { return close_handle.get_child(); }
      inline bool get_leave_open(void) const 
        { return close_handle.get_leave_open(); }
      inline void return_close_op(void)
        { close_handle.return_close_op(close_mask); }
    public:
      FieldMask close_mask;
      TreeClose close_handle;
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
      unsigned rebuild_timeout;
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
      LegionKeyValue<VersionID,FieldMask,
                     LOGICAL_FIELD_VERSIONS_ALLOC>::map field_versions;
      LegionContainer<FieldState,LOGICAL_FIELD_STATE_ALLOC>::list field_states;
#ifndef LOGICAL_FIELD_TREE
      LegionContainer<LogicalUser,CURR_LOGICAL_ALLOC>::list curr_epoch_users;
      LegionContainer<LogicalUser,PREV_LOGICAL_ALLOC>::list prev_epoch_users;
#else
      FieldTree<LogicalUser> *curr_epoch_users;
      FieldTree<LogicalUser> *prev_epoch_users;
#endif
      std::map<Color,LegionContainer<TreeClose,TREE_CLOSE_ALLOC>::list> 
                                                            close_operations;
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
                         bool validates_regions,
                         bool tracing);
    public:
      bool analyze(LogicalUser &user);
      FieldMask get_dominator_mask(void) const;
    public:
      inline void begin_node(FieldTree<LogicalUser> *node) { }
      inline void end_node(FieldTree<LogicalUser> *node) { }
    private:
      const LogicalUser user;
      const bool validates_regions;
      const bool tracing;
      FieldMask dominator_mask;
      FieldMask observed_mask;
    };

    template<bool DOMINATE>
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
      std::deque<TreeClose> close_operations;
#ifdef LOGICAL_FIELD_TREE
    public:
      FieldMask local_closing_mask;
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
      LegionKeyValue<InstanceView*, FieldMask,
                     VALID_VIEW_ALLOC>::map valid_views;
      LegionKeyValue<ReductionView*, FieldMask,
                     VALID_REDUCTION_ALLOC>::map reduction_views;
      LegionKeyValue<MaterializedView*, std::map<Event,FieldMask>,
                     PENDING_UPDATES_ALLOC>::map pending_updates;
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
      void add_target(MaterializedView *target);
      void close_tree_node(RegionTreeNode *node, 
                           const FieldMask &closing_mask);
      const std::vector<MaterializedView*>& get_upper_targets(void) const;
      const std::vector<MaterializedView*>& get_lower_targets(void) const;
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
      std::vector<MaterializedView*> upper_targets;
      std::vector<MaterializedView*> lower_targets;
    }; 

    /**
     * \struct CompositeCloser
     * Class for helping with closing of physical trees to composite instances
     */
    struct CompositeCloser {
    public:
      CompositeCloser(ContextID ctx);
      CompositeCloser(const CompositeCloser &rhs);
      ~CompositeCloser(void);
    public:
      CompositeCloser& operator=(const CompositeCloser &rhs);
    public:
      CompositeNode* get_composite_node(RegionTreeNode *tree_node,
                                        CompositeNode *parent);
      void update_reduction_views(ReductionView *view,
                                  const FieldMask &valid_fields);
      void update_valid_views(PhysicalState &state,
                              CompositeNode *root,
                              const FieldMask &closed_mask);
    public:
      const ContextID ctx;
      bool permit_leave_open;
    public:
      std::map<RegionTreeNode*,CompositeNode*> constructed_nodes;
      std::map<CompositeNode*,FieldMask> collapsed_nodes;
      std::map<ReductionView*,FieldMask> reduction_views;
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
     * \struct PreconditionSet
     * A helper class for building sets of fields with 
     * a common set of preconditions for doing copies.
     */
    struct PreconditionSet {
    public:
      PreconditionSet(void) { }
      PreconditionSet(const FieldMask &m)
        : pre_mask(m) { }
    public:
      FieldMask pre_mask;
      std::set<Event> preconditions;
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
      RegionTreeNode(RegionTreeForest *ctx, FieldSpaceNode *column);
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
                                 RegionTreePath &path,
                                 const bool already_traced);
      void open_logical_node(ContextID ctx,
                             const LogicalUser &user,
                             RegionTreePath &path,
                             const bool already_traced);
      void close_logical_node(LogicalCloser &closer,
                              const FieldMask &closing_mask,
                              bool permit_leave_open);
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
                                   const std::deque<TreeClose> &close_ops);
      void advance_field_versions(LogicalState &state, const FieldMask &mask);
      void filter_prev_epoch_users(LogicalState &state, const FieldMask &mask);
      void filter_curr_epoch_users(LogicalState &state, const FieldMask &mask);
      void filter_close_operations(LogicalState &state, const FieldMask &mask);
      void sanity_check_logical_state(LogicalState &state);
      void initialize_logical_state(ContextID ctx);
      void invalidate_logical_state(ContextID ctx);
      template<bool DOMINATE>
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
                          const std::map<InstanceView*,FieldMask> &valid_views,
                          std::map<MaterializedView*,FieldMask> &update_views,
                                bool &create_composite);
      bool siphon_physical_children(PhysicalCloser &closer,
                                    PhysicalState &state,
                                    const FieldMask &closing_mask,
                                    int next_child,
                                    bool &create_composite); 
      bool close_physical_child(PhysicalCloser &closer,
                                PhysicalState &state,
                                const FieldMask &closing_mask,
                                Color target_child,
                                int next_child,
                                bool &create_composite);
      // Analogous methods to those above except for closing to a composite view
      void close_physical_node(CompositeCloser &closer,
                               CompositeNode *node,
                               const FieldMask &closing_mask,
                               FieldMask &dirty_mask,
                               FieldMask &complete_mask);
      void siphon_physical_children(CompositeCloser &closer,
                                    CompositeNode *node,
                                    PhysicalState &state,
                                    const FieldMask &closing_mask,
                                    FieldMask &dirty_mask,
                                    FieldMask &complete_mask);
      void close_physical_child(CompositeCloser &closer,
                                CompositeNode *node,
                                PhysicalState &state,
                                const FieldMask &closing_mask,
                                Color target_child,
                                int next_child,
                                FieldMask &dirty_mask,
                                FieldMask &complete_mask);
      // This method will always add valid references to the set of views
      // that are returned.  It is up to the caller to remove the references.
      void find_valid_instance_views(PhysicalState &state,
                                     const FieldMask &valid_mask,
                                     const FieldMask &space_mask, 
                                     bool needs_space,
                             std::map<InstanceView*,FieldMask> &valid_views);
      static void remove_valid_references(
                       const std::map<InstanceView*,FieldMask> &valid_views);
      void find_valid_reduction_views(PhysicalState &state,
                                      const FieldMask &valid_mask,
                                      std::set<ReductionView*> &valid_views);
      static void remove_valid_references(
                                const std::set<ReductionView*> &valid_views);
      void pull_valid_instance_views(PhysicalState &state,
                                     const FieldMask &mask);
      void find_pending_updates(PhysicalState &state, 
                                MaterializedView *target,
                                FieldMask &needed_fields,
                                std::set<Event> &pending_events);
      void find_copy_across_instances(MappableInfo *info,
                                      MaterializedView *target,
                           std::map<MaterializedView*,FieldMask> &src_instances,
                       std::map<CompositeView*,FieldMask> &composite_instances);
      // Since figuring out how to issue copies is expensive, try not
      // to hold the physical state lock when doing them. NOTE IT IS UNSOUND
      // TO CALL THIS METHOD WITH A SET OF VALID INSTANCES ACQUIRED BY PASSING
      // 'TRUE' TO THE find_valid_instance_views METHOD!!!!!!!!
      void issue_update_copies(MappableInfo *info,
                               MaterializedView *target, 
                               FieldMask copy_mask,
                      const std::map<InstanceView*,FieldMask> &valid_instances);
      void sort_copy_instances(MappableInfo *info,
                               MaterializedView *target,
                               FieldMask &copy_mask,
                              std::map<InstanceView*,FieldMask> &copy_instances,
                           std::map<MaterializedView*,FieldMask> &src_instances,
                       std::map<CompositeView*,FieldMask> &composite_instances);
      // Issue copies for fields with the same event preconditions
      static void issue_grouped_copies(MappableInfo *info,
                                       MaterializedView *dst,
                                       std::map<Event,FieldMask> &preconditions,
                                       const FieldMask &update_mask,
                                       const std::set<Domain> &copy_domains,
                     const std::map<MaterializedView*,FieldMask> &src_instances,
                                     std::map<Event,FieldMask> &postconditions);
      // Note this function can mutate the preconditions set
      static void compute_precondition_sets(FieldMask update_mask,
          std::map<Event,FieldMask> &preconditions,
          std::list<PreconditionSet> &precondition_sets);
      Event perform_copy_operation(Event precondition,
                        const std::vector<Domain::CopySrcDstField> &src_fields,
                        const std::vector<Domain::CopySrcDstField> &dst_fields);
      void issue_update_reductions(LogicalView *target,
                                   const FieldMask &update_mask,
                                   Processor local_proc,
                    const std::map<ReductionView*,FieldMask> &valid_reductions);
      void invalidate_instance_views(PhysicalState &state,
                                     const FieldMask &invalid_mask, 
                                     bool clean, bool force);
      void invalidate_reduction_views(PhysicalState &state,
                                      const FieldMask &invalid_mask);
      void update_valid_views(PhysicalState &state, const FieldMask &valid_mask,
                              bool dirty, InstanceView *new_view);
      void update_valid_views(PhysicalState &state, const FieldMask &valid_mask,
                              const FieldMask &dirty_mask, 
                              const std::vector<InstanceView*> &new_views);
      // I hate the container problem, somebody solve it please
      void update_valid_views(PhysicalState &state, const FieldMask &valid_mask,
                              const FieldMask &dirty,
                              const std::vector<MaterializedView*> &new_views);
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
                                     const FieldMask &invalid_mask,
                                     bool force);
    public:
      virtual unsigned get_depth(void) const = 0;
      virtual unsigned get_color(void) const = 0;
      virtual RegionTreeNode* get_parent(void) const = 0;
      virtual RegionTreeNode* get_tree_child(Color c) = 0;
      virtual bool are_children_disjoint(Color c1, Color c2) = 0;
      virtual bool are_all_children_disjoint(void) = 0;
      virtual bool is_region(void) const = 0;
      virtual RegionNode* as_region_node(void) const = 0;
      virtual PartitionNode* as_partition_node(void) const = 0;
      virtual bool visit_node(PathTraverser *traverser) = 0;
      virtual bool visit_node(NodeTraverser *traverser) = 0;
      virtual bool has_component_domains(void) const = 0;
      virtual const std::set<Domain>& get_component_domains(void) const = 0;
      virtual Domain get_domain(void) const = 0;
      virtual bool is_complete(void) = 0;
      virtual bool intersects_with(RegionTreeNode *other) = 0;
      virtual bool dominates(RegionTreeNode *other) = 0;
      virtual const std::set<Domain>& 
                            get_intersection_domains(RegionTreeNode *other) = 0;
      virtual size_t get_num_children(void) const = 0;
      virtual MaterializedView * create_instance(Memory target_mem,
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
                           std::map<LogicalView*,FieldMask> &needed_views,
                           std::set<PhysicalManager*> &needed_managers);
      bool pack_send_back_state(ContextID ctx, Serializer &rez,
                                AddressSpaceID target, const FieldMask &send_mask,
                                std::set<PhysicalManager*> &needed_managers);
      void unpack_send_state(ContextID ctx, Deserializer &derez, 
                             FieldSpaceNode *column, AddressSpaceID source);
    public:
#ifndef LOGICAL_FIELD_TREE
      // Logical helper operations
      template<typename ALLOC> 
      FieldMask perform_dependence_checks(const LogicalUser &user, 
            std::list<LogicalUser, ALLOC> &users, const FieldMask &check_mask,
            bool validates_regions);
      template<typename ALLOC>
      void perform_closing_checks(LogicalCloser &closer,
            std::list<LogicalUser, ALLOC> &users, const FieldMask &check_mask);
#else
      FieldMask perform_dependence_checks(const LogicalUser &user,
            FieldTree<LogicalUser> *users, const FieldMask &check_mask,
            bool validates_regions);
      void perform_closing_checks(LogicalCloser &closer,
            FieldTree<LogicalUser> *users, const FieldMask &check_mask);
#endif
    public:
      inline FieldSpaceNode* get_column_source(void) const 
      { return column_source; }
    public:
      RegionTreeForest *const context;
      FieldSpaceNode *const column_source;
    public:
      NodeMask creation_set;
      NodeMask destruction_set;
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
      virtual bool are_all_children_disjoint(void);
      virtual bool is_region(void) const;
      virtual RegionNode* as_region_node(void) const;
      virtual PartitionNode* as_partition_node(void) const;
      virtual bool visit_node(PathTraverser *traverser);
      virtual bool visit_node(NodeTraverser *traverser);
      virtual bool has_component_domains(void) const;
      virtual const std::set<Domain>& get_component_domains(void) const;
      virtual Domain get_domain(void) const;
      virtual bool is_complete(void);
      virtual bool intersects_with(RegionTreeNode *other);
      virtual bool dominates(RegionTreeNode *other);
      virtual const std::set<Domain>& 
                                get_intersection_domains(RegionTreeNode *other);
      virtual size_t get_num_children(void) const;
      virtual MaterializedView* create_instance(Memory target_mem,
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
      void remap_region(ContextID ctx, MaterializedView *view, 
                        const FieldMask &user_mask, FieldMask &needed_mask);
      InstanceRef register_region(MappableInfo *info, 
                                  PhysicalUser &user,
                                  LogicalView *view,
                                  const FieldMask &needed_fields);
      InstanceRef seed_state(ContextID ctx, PhysicalUser &user,
                             LogicalView *new_view,
                             Processor local_proc);
      Event close_state(MappableInfo *info, PhysicalUser &user,
                        const InstanceRef &target);
    public:
      bool send_state(ContextID ctx, UniqueID uid, AddressSpaceID target,
                      const FieldMask &send_mask, bool invalidate,
                      std::map<LogicalView*,FieldMask> &needed_views,
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
      virtual bool are_all_children_disjoint(void);
      virtual bool is_region(void) const;
      virtual RegionNode* as_region_node(void) const;
      virtual PartitionNode* as_partition_node(void) const;
      virtual bool visit_node(PathTraverser *traverser);
      virtual bool visit_node(NodeTraverser *traverser);
      virtual bool has_component_domains(void) const;
      virtual const std::set<Domain>& get_component_domains(void) const;
      virtual Domain get_domain(void) const;
      virtual bool is_complete(void);
      virtual bool intersects_with(RegionTreeNode *other);
      virtual bool dominates(RegionTreeNode *other);
      virtual const std::set<Domain>& 
                                get_intersection_domains(RegionTreeNode *other);
      virtual size_t get_num_children(void) const;
      virtual MaterializedView* create_instance(Memory target_mem,
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
                      std::map<LogicalView*,FieldMask> &needed_views,
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
      const bool disjoint;
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
      void record_close_operation(unsigned depth, const TreeClose &info,
                                  const FieldMask &close_mask);
      void record_before_version(unsigned depth, VersionID vid,
                                 const FieldMask &version_mask);
      void record_after_version(unsigned depth, VersionID vid,
                                const FieldMask &version_mask);
    public:
      void get_close_operations(unsigned depth,
                                std::deque<CloseInfo> &needed_ops);
    protected:
      std::vector<int> path;
      std::vector<std::deque<std::pair<TreeClose,FieldMask> > > close_ops;
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
    template<bool DOMINATE>
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
      PhysicalInvalidator(ContextID ctx, const FieldMask &invalid_mask,
                          const bool force);
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
      const bool force;
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
                  std::map<LogicalView*,FieldMask> &needed_views,
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
      std::map<LogicalView*,FieldMask> &needed_views;
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
     * \class LayoutDescription
     * This class is for deduplicating the meta-data
     * associated with describing the layouts of physical
     * instances. Often times this meta data is rather 
     * large (~100K) and since we routinely create up
     * to 100K instances, it is important to deduplicate
     * the data.  Since many instances will have the
     * same layout then they can all share the same
     * description object.
     */
    class LayoutDescription {
    public:
      struct OffsetEntry {
      public:
        OffsetEntry(void) { }
        OffsetEntry(const FieldMask &m,
                    const std::vector<Domain::CopySrcDstField> &f)
          : offset_mask(m), offsets(f) { }
      public:
        FieldMask offset_mask;
        std::vector<Domain::CopySrcDstField> offsets;
      };
    public:
      LayoutDescription(const FieldMask &mask,
                        const Domain &domain,
                        size_t blocking_factor,
                        FieldSpaceNode *owner);
      LayoutDescription(const LayoutDescription &rhs);
      ~LayoutDescription(void);
    public:
      LayoutDescription& operator=(const LayoutDescription &rhs);
    public:
      void compute_copy_offsets(const FieldMask &copy_mask, 
                                PhysicalInstance inst,
                                std::vector<Domain::CopySrcDstField> &fields);
      void compute_copy_offsets(const std::vector<FieldID> &copy_fields,
                                PhysicalInstance inst,
                                std::vector<Domain::CopySrcDstField> &fields);
    public:
      void add_field_info(FieldID fid, unsigned index,
                          size_t offset, size_t field_size);
      const Domain::CopySrcDstField& find_field_info(FieldID fid) const;
      size_t get_layout_size(void) const;
    public:
      bool match_shape(const size_t field_size) const;
      bool match_shape(const std::vector<size_t> &field_sizes, 
                       const size_t bf) const;
    public:
      bool match_layout(const FieldMask &mask, 
                        const size_t vol, const size_t bf) const;
      bool match_layout(const FieldMask &mask,
                        const Domain &d, const size_t bf) const;
      bool match_layout(LayoutDescription *rhs) const;
    public:
      void pack_layout_description(Serializer &rez, AddressSpaceID target);
      void unpack_layout_description(Deserializer &derez);
      void update_known_nodes(AddressSpaceID target);
      static LayoutDescription* handle_unpack_layout_description(
          Deserializer &derez, AddressSpaceID source, RegionNode *node);
    public:
      static size_t compute_layout_volume(const Domain &d);
    public:
      const FieldMask allocated_fields;
      const size_t blocking_factor;
      const size_t volume;
      FieldSpaceNode *const owner;
    protected:
      std::map<FieldID,Domain::CopySrcDstField> field_infos;
      // Remember these indexes are only good on the local node and
      // have to be transformed when the manager is sent remotely
      std::map<unsigned/*index*/,FieldID> field_indexes;
    protected:
      // Memoized value for matching physical instances
      std::map<unsigned/*offset*/,unsigned/*size*/> offset_size_map;
    protected:
      Reservation offset_lock; 
      std::map<FIELD_TYPE,std::vector<OffsetEntry> > memoized_offsets;
    protected:
      NodeMask known_nodes;
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
      static const AllocationType alloc_type = INSTANCE_MANAGER_ALLOC;
    public:
      struct OffsetEntry {
      public:
        OffsetEntry(void) { }
        OffsetEntry(const FieldMask &m,
                    const std::vector<Domain::CopySrcDstField> &f)
          : offset_mask(m), offsets(f) { }
      public:
        FieldMask offset_mask;
        std::vector<Domain::CopySrcDstField> offsets;
      };
    public:
      InstanceManager(RegionTreeForest *ctx, DistributedID did,
                      AddressSpaceID owner_space, AddressSpaceID local_space,
                      Memory mem, PhysicalInstance inst, RegionNode *node,
                      LayoutDescription *desc, Event use_event, 
                      unsigned depth, bool persistent = false);
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
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
    public:
      inline Event get_use_event(void) const { return use_event; }
      Event get_recycle_event(void);
    public:
      MaterializedView* create_top_view(unsigned depth);
      void compute_copy_offsets(const FieldMask &copy_mask,
                                std::vector<Domain::CopySrcDstField> &fields);
      void compute_copy_offsets(const std::vector<FieldID> &copy_fields,
                                std::vector<Domain::CopySrcDstField> &fields);
    public:
      DistributedID send_manager(AddressSpaceID target, 
                                 std::set<PhysicalManager*> &needed_managers);
      static void handle_send_manager(RegionTreeForest *context, 
                                      AddressSpaceID source,
                                      Deserializer &derez);
    public:
      void pack_manager(Serializer &rez, AddressSpaceID target);
      static InstanceManager* unpack_manager(Deserializer &derez,
                                             RegionTreeForest *context, 
                                             DistributedID did,
                                             AddressSpaceID source,
                                             bool make = true);
    public:
      void add_valid_view(MaterializedView *view);
      void remove_valid_view(MaterializedView *view);
      bool match_instance(size_t field_size, const Domain &dom) const;
      bool match_instance(const std::vector<size_t> &fields_sizes,
                          const Domain &dom, const size_t bf) const;
    public:
      bool is_persistent(void) const;
      void make_persistent(AddressSpaceID origin);
      static void handle_make_persistent(Deserializer &derez,
                                         RegionTreeForest *context,
                                         AddressSpaceID source);
    public:
      RegionNode *const region_node;
      LayoutDescription *const layout;
      // Event that needs to trigger before we can start using
      // this physical instance.
      const Event use_event;
      const unsigned depth;
    protected:
      // Keep track of whether we've recycled this instance or not
      bool recycled;
      // Keep a set of the views we need to see when recycling
      std::set<MaterializedView*> valid_views;
    protected:
      // This is monotonic variable that once it becomes true
      // will remain true for the duration of the instance lifetime.
      // If set to true, it should prevent the instance from ever
      // being collected before the context in which it was created
      // is destroyed.
      bool persistent;
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
          Domain space, Event precondition, bool reduction_fold,
          bool precise_domain) = 0;
      virtual Domain get_pointer_space(void) const = 0;
    public:
      virtual bool is_list_manager(void) const = 0;
      virtual ListReductionManager* as_list_manager(void) const = 0;
      virtual FoldReductionManager* as_fold_manager(void) const = 0;
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
      static const AllocationType alloc_type = LIST_MANAGER_ALLOC;
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
          Domain space, Event precondition, bool reduction_fold,
          bool precise_domain);
      virtual Domain get_pointer_space(void) const;
    public:
      virtual bool is_list_manager(void) const;
      virtual ListReductionManager* as_list_manager(void) const;
      virtual FoldReductionManager* as_fold_manager(void) const;
    protected:
      const Domain ptr_space;
    };

    /**
     * \class FoldReductionManager
     * A class for representing fold reduction instances
     */
    class FoldReductionManager : public ReductionManager {
    public:
      static const AllocationType alloc_type = FOLD_MANAGER_ALLOC;
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
          Domain space, Event precondition, bool reduction_fold,
          bool precise_domain);
      virtual Domain get_pointer_space(void) const;
    public:
      virtual bool is_list_manager(void) const;
      virtual ListReductionManager* as_list_manager(void) const;
      virtual FoldReductionManager* as_fold_manager(void) const;
    };

    /**
     * \class Logicalview 
     * This class is the abstract base class for representing
     * the logical view onto one or more physical instances
     * in memory.  Logical views are reference counted
     * and will delete themselves once they no longer have
     * any valid handles.
     */
    class LogicalView : public HierarchicalCollectable {
    public:
      LogicalView(RegionTreeForest *ctx, DistributedID did,
                  AddressSpaceID owner_proc, DistributedID own_did,
                  RegionTreeNode *node);
      virtual ~LogicalView(void);
    public:
      virtual bool is_reduction_view(void) const = 0;
      virtual InstanceView* as_instance_view(void) const = 0;
      virtual ReductionView* as_reduction_view(void) const = 0;
      virtual PhysicalManager* get_manager(void) const = 0;
    public:
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           const FieldMask &copy_mask,
                                 std::map<Event,FieldMask> &preconditions) = 0;
      virtual void add_copy_user(ReductionOpID redop, Event copy_term,
                                 const FieldMask &mask, bool reading,
                                 Processor exec_proc) = 0;
      virtual InstanceRef add_user(PhysicalUser &user,
                                   Processor exec_proc) = 0;
      virtual bool reduce_to(ReductionOpID redop, 
                             const FieldMask &reduce_mask,
                     std::vector<Domain::CopySrcDstField> &src_fields) = 0;
    public:
      virtual void notify_activate(void) = 0;
      virtual void garbage_collect(void) = 0;
      virtual void notify_valid(void) = 0;
      virtual void notify_invalid(void) = 0;
    public:
      void defer_collect_user(Event term_event);
      virtual void collect_users(const std::set<Event> &term_events) = 0;
      static void handle_deferred_collect(LogicalView *view,
                                          const std::set<Event> &term_events);
    public:
      void send_back_user(const PhysicalUser &user);
      virtual void process_send_back_user(AddressSpaceID source,
                                          PhysicalUser &user) = 0;
      static void handle_send_back_user(RegionTreeForest *context,
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
     * The InstanceView class is used for managing the meta-data
     * for one or more physical instances which represent the
     * up-to-date version from a logical region's perspective.
     * The InstaceView class has two sub-classes: materialized
     * views which represent a single physical instance, or
     * composite views which contain multiple physical instances.
     */
    class InstanceView : public LogicalView {
    public:
      InstanceView(RegionTreeForest *ctx, DistributedID did,
                   AddressSpaceID owner_proc, DistributedID own_did,
                   RegionTreeNode *node);
      virtual ~InstanceView(void);
    public:
      virtual bool is_reduction_view(void) const;
      virtual InstanceView* as_instance_view(void) const;
      virtual ReductionView* as_reduction_view(void) const;
      virtual PhysicalManager* get_manager(void) const = 0;
      virtual bool is_persistent(void) const = 0;
    public:
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           const FieldMask &copy_mask,
                                 std::map<Event,FieldMask> &preconditions) = 0;
      virtual void add_copy_user(ReductionOpID redop, Event copy_term,
                                 const FieldMask &mask, bool reading,
                                 Processor exec_proc) = 0;
      virtual InstanceRef add_user(PhysicalUser &user,
                                   Processor exec_proc) = 0;
      virtual bool reduce_to(ReductionOpID redop, 
                             const FieldMask &reduce_mask,
                     std::vector<Domain::CopySrcDstField> &src_fields) = 0;
    public:
      virtual void notify_activate(void) = 0;
      virtual void garbage_collect(void) = 0;
      virtual void notify_valid(void) = 0;
      virtual void notify_invalid(void) = 0;
    public:
      virtual void collect_users(const std::set<Event> &term_events) = 0;
    public:
      virtual void process_send_back_user(AddressSpaceID source,
                                          PhysicalUser &user) = 0;
    public: // Virtual methods specific to InstanceView start here
      virtual bool is_composite_view(void) const = 0;
      virtual MaterializedView* as_materialized_view(void) const = 0;
      virtual CompositeView* as_composite_view(void) const = 0;
    public:
      virtual bool has_parent_view(void) const = 0;
      virtual InstanceView* get_parent_view(void) const = 0;
      virtual InstanceView* get_subview(Color c) = 0;
    public:
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &dst_fields) = 0;
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &src_fields) = 0;
      virtual bool has_war_dependence(const RegionUsage &usage, 
                                      const FieldMask &user_mask) = 0;
    public:
      virtual DistributedID send_state(AddressSpaceID target,
                            const FieldMask &send_mask,
                            std::map<LogicalView*,FieldMask> &needed_views,
                            std::set<PhysicalManager*> &needed_managers) = 0;
      virtual DistributedID send_back_state(AddressSpaceID target,
                            const FieldMask &send_mask,
                            std::set<PhysicalManager*> &needed_managers) = 0;
    public:
      void add_alias_did(DistributedID did);
    public:
      static void handle_send_subscriber(RegionTreeForest *context,
                                         Deserializer &derez,
                                         AddressSpaceID source);
    protected:
      // A list of aliased distributed IDs for this view
      std::set<DistributedID> aliases;
    };

    /**
     * \class MaterializedView 
     * The MaterializedView class is used for representing a given
     * logical view onto a single physical instance.
     */
    class MaterializedView : public InstanceView {
    public:
      static const AllocationType alloc_type = MATERIALIZED_VIEW_ALLOC;
    public:
      MaterializedView(RegionTreeForest *ctx, DistributedID did,
                       AddressSpaceID owner_proc, DistributedID own_did,
                       RegionTreeNode *node, InstanceManager *manager,
                       MaterializedView *parent, unsigned depth);
      MaterializedView(const MaterializedView &rhs);
      virtual ~MaterializedView(void);
    public:
      MaterializedView& operator=(const MaterializedView &rhs);
    public:
      Memory get_location(void) const;
      size_t get_blocking_factor(void) const;
      const FieldMask& get_physical_mask(void) const;
    public:
      virtual bool is_composite_view(void) const;
      virtual MaterializedView* as_materialized_view(void) const;
      virtual CompositeView* as_composite_view(void) const;
    public:
      virtual bool has_parent_view(void) const;
      virtual InstanceView* get_parent_view(void) const;
      virtual InstanceView* get_subview(Color c);
      bool add_subview(MaterializedView *view, Color c);
      MaterializedView* get_materialized_subview(Color c);
      MaterializedView* get_materialized_parent_view(void) const;
    public:
      void copy_field(FieldID fid, std::vector<Domain::CopySrcDstField> &infos);
    public:
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &dst_fields);
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &src_fields);
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask,
                     std::vector<Domain::CopySrcDstField> &dst_fields);
      virtual bool has_war_dependence(const RegionUsage &usage, 
                              const FieldMask &user_mask);
    public:
      void accumulate_events(std::set<Event> &all_events);
    public:
      virtual PhysicalManager* get_manager(void) const;
    public:
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           const FieldMask &copy_mask,
                                   std::map<Event,FieldMask> &preconditions);
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
      virtual void collect_users(const std::set<Event> &term_users);
      virtual void process_send_back_user(AddressSpaceID source,
                                          PhysicalUser &user);
    protected:
      void add_user_above(std::set<Event> &wait_on, PhysicalUser &user);
      template<bool ABOVE>
      void add_local_user(std::set<Event> &wait_on, 
                          const PhysicalUser &user);
      void add_copy_user_above(PhysicalUser &user);
      void add_local_copy_user(PhysicalUser &user);
    protected: 
      void find_copy_preconditions_above(Color child_color,
                                   ReductionOpID redop, bool reading,
                                   const FieldMask &copy_mask,
                                   std::map<Event,FieldMask> &preconditions);
      template<bool ABOVE>
      void find_local_copy_preconditions(int local_color,
                                   ReductionOpID redop, bool reading,
                                   const FieldMask &copy_mask,
                                   std::map<Event,FieldMask> &preconditions);
      bool has_war_dependence_above(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    Color child_color);
      void update_versions(const FieldMask &update_mask);
      void filter_local_users(Event term_event);
      void filter_local_users(const std::set<Event> &term_events);
      template<typename ALLOC>
      void condense_user_list(std::list<PhysicalUser,ALLOC> &users, 
                              bool previous);
    public:
      virtual DistributedID send_state(AddressSpaceID target,
                            const FieldMask &send_mask,
                            std::map<LogicalView*,FieldMask> &needed_views,
                            std::set<PhysicalManager*> &needed_managers);
      virtual DistributedID send_back_state(AddressSpaceID target,
                            const FieldMask &send_mask,
                            std::set<PhysicalManager*> &needed_managers);
    public:
      virtual bool is_persistent(void) const;
      void make_persistent(void);
    protected:
      void pack_materialized_view(Serializer &rez);
      void unpack_materialized_view(Deserializer &derez, 
                                    AddressSpaceID source, bool need_lock);
      void send_updates(DistributedID remote_did, AddressSpaceID target,
                        const FieldMask &update_mask);
      void process_updates(Deserializer &derez, AddressSpaceID source);
    public:
      static void handle_send_materialized_view(RegionTreeForest *context, 
                                                Deserializer &derez,
                                                AddressSpaceID source);
      static void handle_send_back_materialized_view(
                      RegionTreeForest *context, Deserializer &derez,
                      AddressSpaceID source); 
      static void handle_send_updates(RegionTreeForest *context,
                                      Deserializer &derez,
                                      AddressSpaceID source);
      template<typename ALLOC>
      static void filter_list(std::list<PhysicalUser,ALLOC> &user_list,
                              const FieldMask &filter_mask);
    public:
      InstanceManager *const manager;
      MaterializedView *const parent;
      const unsigned depth;
    protected:
      // The lock for the instance shared between all views
      // of a physical instance within a context.  The top
      // most view is responsible for deleting the lock
      // when it is reclaimed.
      Reservation inst_lock;
      // Keep track of the child views
      std::map<Color,MaterializedView*> children;
      // These are the sets of users in the current and next epochs
      // for performing dependence analysis
#ifndef PHYSICAL_FIELD_TREE
      LegionContainer<PhysicalUser,CURR_PHYSICAL_ALLOC>::list curr_epoch_users;
      LegionContainer<PhysicalUser,PREV_PHYSICAL_ALLOC>::list prev_epoch_users;
#else
      FieldTree<PhysicalUser> *const curr_epoch_users;
      FieldTree<PhysicalUser> *const prev_epoch_users;
#endif
      // Keep track of how many outstanding references we have
      // for each of the user events
      LegionContainer<Event,EVENT_REFERENCE_ALLOC>::set event_references;
      // Version information for each of the fields
      LegionKeyValue<VersionID,FieldMask,
                     PHYSICAL_VERSION_ALLOC>::map current_versions;
    };

    /**
     * \class CompositeView
     * The CompositeView class is used for deferring close
     * operations by representing a valid version of a single
     * logical region with a bunch of different instances.
     */
    class CompositeView : public InstanceView {
    public:
      static const AllocationType alloc_type = COMPOSITE_VIEW_ALLOC;
    public:
      struct ReduceInfo {
      public:
        ReduceInfo(void) { }
        ReduceInfo(const FieldMask &valid, const Domain &dom)
          : valid_fields(valid) { intersections.insert(dom); }
        ReduceInfo(const FieldMask &valid, 
                   const std::set<Domain> &inters)
          : valid_fields(valid), intersections(inters) { }
      public:
        FieldMask valid_fields;
        std::set<Domain> intersections;
      };
    public:
      CompositeView(RegionTreeForest *ctx, DistributedID did,
                    AddressSpaceID owner_proc, RegionTreeNode *node, 
                    DistributedID owner_did, const FieldMask &mask,
                    CompositeView *parent = NULL);
      CompositeView(const CompositeView &rhs);
      virtual ~CompositeView(void);
    public:
      CompositeView& operator=(const CompositeView &rhs);
    public:
      virtual PhysicalManager* get_manager(void) const;
      virtual bool is_persistent(void) const;
    public:
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           const FieldMask &copy_mask,
                                 std::map<Event,FieldMask> &preconditions);
      virtual void add_copy_user(ReductionOpID redop, Event copy_term,
                                 const FieldMask &mask, bool reading,
                                 Processor exec_proc);
      virtual InstanceRef add_user(PhysicalUser &user,
                                   Processor exec_proc);
      virtual bool reduce_to(ReductionOpID redop, 
                             const FieldMask &reduce_mask,
                     std::vector<Domain::CopySrcDstField> &src_fields);
    public:
      virtual void notify_activate(void);
      virtual void garbage_collect(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
    public:
      virtual void collect_users(const std::set<Event> &term_events);
    public:
      virtual void process_send_back_user(AddressSpaceID source,
                                          PhysicalUser &user);
    public:
      virtual bool is_composite_view(void) const;
      virtual MaterializedView* as_materialized_view(void) const;
      virtual CompositeView* as_composite_view(void) const;
    public:
      virtual bool has_parent_view(void) const;
      virtual InstanceView* get_parent_view(void) const;
      virtual InstanceView* get_subview(Color c);
      bool add_subview(CompositeView *view, Color c);
      void update_valid_mask(const FieldMask &mask);
    public:
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &dst_fields);
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &src_fields);
      virtual bool has_war_dependence(const RegionUsage &usage, 
                                      const FieldMask &user_mask);
    public:
      virtual DistributedID send_state(AddressSpaceID target,
                            const FieldMask &send_mask,
                            std::map<LogicalView*,FieldMask> &needed_views,
                            std::set<PhysicalManager*> &needed_managers);
      virtual DistributedID send_back_state(AddressSpaceID target,
                            const FieldMask &send_mask,
                            std::set<PhysicalManager*> &needed_managers);
    public:
      void add_root(CompositeNode *root, const FieldMask &valid);
      void update_reduction_views(ReductionView *view, 
                                  const FieldMask &valid_mask);
      void issue_composite_copies(MappableInfo *info,
                                  MaterializedView *dst,
                                  const FieldMask &copy_mask);
      void issue_composite_copies(MappableInfo *info,
                                  MaterializedView *dst,
                                  const FieldMask &copy_mask,
                  const std::map<Event,FieldMask> &preconditions,
                        std::map<Event,FieldMask> &postconditions);
    public:
      // Note that copy-across only works for a single field at a time
      void issue_composite_copies_across(MappableInfo *info,
                                         MaterializedView *dst,
                                         FieldID src_field,
                                         FieldID dst_field,
                                         Event precondition,
                                         std::set<Event> &postconditions);
    protected:
      void flush_reductions(MappableInfo *info,
                            MaterializedView *dst,
                            const FieldMask &event_mask,
                            const std::map<Event,FieldMask> &preconditions,
                            std::set<Event> &event_set);
    public:
      void pack_composite_view(Serializer &rez, bool send_back,
                               AddressSpaceID target,
                               const FieldMask &pack_mask,
                               std::map<LogicalView*,FieldMask> &needed_views,
                               std::set<PhysicalManager*> &needed_managers);
      void unpack_composite_view(Deserializer &derez, 
                                 AddressSpaceID source,
                                 bool send_back, bool need_lock);
      void send_updates(DistributedID remote_did, AddressSpaceID target, 
                        FieldMask send_mask, 
                        std::map<LogicalView*,FieldMask> &needed_views,
                        std::set<PhysicalManager*> &needed_managers);
    public:
      static void handle_send_composite_view(RegionTreeForest *context, 
                                             Deserializer &derez,
                                             AddressSpaceID source);
      static void handle_send_back_composite_view(RegionTreeForest *context,
                                                  Deserializer &derez,
                                                  AddressSpaceID source);
      static void handle_send_composite_update(RegionTreeForest *context,
                                               Deserializer &derez,
                                               AddressSpaceID source);
    public:
      CompositeView *const parent;
    protected:
      // The set of fields represented by this composite view
      FieldMask valid_mask;
      // Keep track of the roots and their field masks
      // There is exactly one root for every field
      std::map<CompositeNode*,FieldMask> roots;
      // Track the set of reductions which need to be applied here
      FieldMask reduction_mask;
      std::map<ReductionView*,ReduceInfo> valid_reductions;
      // Keep track of all the child views
      std::map<Color,CompositeView*> children;
      // Keep track of which fields have been sent remotely
      std::map<AddressSpaceID,FieldMask> remote_state;
    };

    /**
     * \class CompositeNode
     * A helper class for representing the frozen state of a region
     * tree as part of one or more composite views.
     */
    class CompositeNode : public Collectable {
    public:
      static const AllocationType alloc_type = COMPOSITE_NODE_ALLOC;
    public:
      struct ChildInfo {
      public:
        ChildInfo(void)
          : complete(false) { }
        ChildInfo(bool c, const FieldMask &m)
          : complete(c), open_fields(m) { }
      public:
        bool complete;
        FieldMask open_fields;
      };
    public:
      CompositeNode(RegionTreeNode *logical, CompositeNode *parent);
      CompositeNode(const CompositeNode &rhs);
      ~CompositeNode(void);
    public:
      CompositeNode& operator=(const CompositeNode &rhs);
    public:
      void capture_physical_state(RegionTreeNode *tree_node,
                                  PhysicalState &state,
                                  const FieldMask &capture_mask,
                                  CompositeCloser &closer,
                                  FieldMask &global_dirty,
                                  FieldMask &complete_mask);
      void update_parent_info(const FieldMask &mask);
      void update_child_info(CompositeNode *child, const FieldMask &mask);
      void update_instance_views(InstanceView *view,
                                 const FieldMask &valid_mask);
    public:
      void issue_update_copies(MappableInfo *info,
                               MaterializedView *dst,
                               FieldMask traversal_mask,
                               const FieldMask &copy_mask,
                               const std::map<Event,FieldMask> &preconditions,
                               std::map<Event,FieldMask> &postconditions);
      void issue_across_copies(MappableInfo *info,
                               MaterializedView *dst,
                               unsigned src_index,
                               FieldID  src_field,
                               FieldID  dst_field,
                               bool    need_field,
                               std::set<Event> &preconditions,
                               std::set<Event> &postconditions);
    public:
      bool intersects_with(RegionTreeNode *dst);
      const std::set<Domain>& find_intersection_domains(RegionTreeNode *dst);
    public:
      void find_bounding_roots(CompositeView *target, const FieldMask &mask);
    public:
      void add_gc_references(void);
      void remove_gc_references(void);
      void add_valid_references(void);
      void remove_valid_references(void);
    protected:
      bool dominates(RegionTreeNode *dst);
    public:
      void pack_composite_node(Serializer &rez, bool send_back,
                               AddressSpaceID target,
                               const FieldMask &send_mask,
                               std::map<LogicalView*,FieldMask> &needed_views,
                               std::set<PhysicalManager*> &needed_managers);
      void unpack_composite_node(Deserializer &derez, AddressSpaceID source);
    public:
      RegionTreeForest *const context;
      RegionTreeNode *const logical_node;
      CompositeNode *const parent;
    protected:
      FieldMask dirty_mask;
      std::map<CompositeNode*,ChildInfo> open_children;
      std::map<InstanceView*,FieldMask> valid_views;
    };

    /**
     * \class ReductionView
     * The ReductionView class is used for providing a view
     * onto reduction physical instances from any logical perspective.
     */
    class ReductionView : public LogicalView {
    public:
      static const AllocationType alloc_type = REDUCTION_VIEW_ALLOC;
    public:
      ReductionView(RegionTreeForest *ctx, DistributedID did,
                    AddressSpaceID owner_proc, DistributedID own_did,
                    RegionTreeNode *node, ReductionManager *manager);
      ReductionView(const ReductionView &rhs);
      virtual ~ReductionView(void);
    public:
      ReductionView& operator=(const ReductionView&rhs);
    public:
      void perform_reduction(LogicalView *target, 
                             const FieldMask &copy_mask, Processor local_proc);
      Event perform_composite_reduction(MaterializedView *target,
                                        const FieldMask &copy_mask,
                                        Processor local_proc,
                                        const std::set<Event> &preconditions,
                                        const std::set<Domain> &reduce_domains);
      Event perform_composite_across_reduction(MaterializedView *target,
                                               FieldID dst_field,
                                               FieldID src_field,
                                               unsigned src_index,
                                               Processor local_proc,
                                       const std::set<Event> &preconditions,
                                       const std::set<Domain> &reduce_domains);
    public:
      virtual bool is_reduction_view(void) const;
      virtual InstanceView* as_instance_view(void) const;
      virtual ReductionView* as_reduction_view(void) const;
      virtual PhysicalManager* get_manager(void) const;
    public:
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           const FieldMask &copy_mask,
                                   std::map<Event,FieldMask> &preconditions);
      virtual void add_copy_user(ReductionOpID redop, Event copy_term,
                                 const FieldMask &mask, bool reading,
                                 Processor exec_proc);
      virtual InstanceRef add_user(PhysicalUser &user,
                                   Processor exec_proc);
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask,
                     std::vector<Domain::CopySrcDstField> &dst_fields);
    public:
      void reduce_from(ReductionOpID redop, const FieldMask &reduce_mask,
                       std::vector<Domain::CopySrcDstField> &src_fields);
    public:
      virtual void notify_activate(void);
      virtual void garbage_collect(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
      virtual void collect_users(const std::set<Event> &term_events);
      virtual void process_send_back_user(AddressSpaceID source,
                                          PhysicalUser &user);
    public:
      DistributedID send_state(AddressSpaceID target,
                               const FieldMask &send_mask,
                               std::map<LogicalView*,FieldMask> &needed_views,
                               std::set<PhysicalManager*> &needed_managers);
      DistributedID send_back_state(AddressSpaceID target,
                                    const FieldMask &send_mask,
                               std::set<PhysicalManager*> &needed_managers);
    public:
      void pack_reduction_view(Serializer &rez);
      void unpack_reduction_view(Deserializer &derez, AddressSpaceID source);
      void send_updates(DistributedID remote_did, AddressSpaceID target,
                        const FieldMask &update_mask);
      void process_updates(Deserializer &derez, AddressSpaceID source);
    public:
      static void handle_send_reduction_view(RegionTreeForest *context,
                                Deserializer &derez, AddressSpaceID source);
      static void handle_send_back_reduction_view(RegionTreeForest *context,
                                Deserializer &derez, AddressSpaceID source);
      static void handle_send_update(RegionTreeForest *context,
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
      ViewHandle(LogicalView *v);
      ViewHandle(const ViewHandle &rhs);
      ~ViewHandle(void);
    public:
      ViewHandle& operator=(const ViewHandle &rhs);
    public:
      inline bool has_view(void) const { return (view != NULL); }
      inline LogicalView* get_view(void) const { return view; }
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
      LogicalView *view;
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
      InstanceRef(Event ready, Reservation lock, const ViewHandle &handle);
    public:
      bool operator==(const InstanceRef &rhs) const;
      bool operator!=(const InstanceRef &rhs) const;
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

