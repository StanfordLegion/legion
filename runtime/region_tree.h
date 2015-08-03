/* Copyright 2015 Stanford University, NVIDIA Corporation
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
      struct DisjointnessArgs {
        HLRTaskID hlr_id;
        IndexPartition handle;
        UserEvent ready;
      };  
    public:
      RegionTreeForest(Runtime *rt);
      RegionTreeForest(const RegionTreeForest &rhs);
      ~RegionTreeForest(void);
    public:
      RegionTreeForest& operator=(const RegionTreeForest &rhs);
    public:
      void create_index_space(IndexSpace handle, const Domain &domain,
                              IndexSpaceKind kind, AllocateMode mode);
      void create_index_space(IndexSpace handle, const Domain &hull,
                              const std::set<Domain> &domains,
                              IndexSpaceKind kind, AllocateMode mode);
      void create_index_partition(IndexPartition pid, IndexSpace parent,
                                  ColorPoint part_color, 
                                  const std::map<DomainPoint,Domain> &subspaces,
                                  const Domain &color_space, 
                                  PartitionKind part_kind, AllocateMode mode);
      void create_index_partition(IndexPartition pid, IndexSpace parent,
                                  ColorPoint part_color, 
                                  const std::map<DomainPoint,Domain> &hulls, 
                  const std::map<DomainPoint,std::set<Domain> > &components,
                                  const Domain &color_space, 
                                  PartitionKind part_kind, AllocateMode mode);
      void compute_partition_disjointness(IndexPartition handle,
                                          UserEvent ready_event);
      bool destroy_index_space(IndexSpace handle, AddressSpaceID source);
      void destroy_index_partition(IndexPartition handle, 
                                   AddressSpaceID source);
    public:
      Event create_equal_partition(IndexPartition pid, size_t granularity);
      Event create_weighted_partition(IndexPartition pid, size_t granularity,
                                      const std::map<DomainPoint,int> &weights);
    public:
      Event create_partition_by_union(IndexPartition pid,
                                      IndexPartition handle1,
                                      IndexPartition handle2);
      Event create_partition_by_intersection(IndexPartition pid,
                                             IndexPartition handle1,
                                             IndexPartition handle2);
      Event create_partition_by_difference(IndexPartition pid,
                                           IndexPartition handle1,
                                           IndexPartition handle2);
      Event create_cross_product_partitions(IndexPartition base,
                                            IndexPartition source,
                      std::map<DomainPoint,IndexPartition> &handles);
    public:
      void compute_pending_color_space(IndexSpace parent,
                                       IndexPartition handle1,
                                       IndexPartition handle2,
                                       Domain &color_space,
                         LowLevel::IndexSpace::IndexSpaceOperation op);
      void create_pending_partition(IndexPartition pid,
                                    IndexSpace parent,
                                    const Domain &color_space,
                                    ColorPoint partition_color,
                                    PartitionKind part_kind,
                                    bool allocable, 
                                    Event handle_ready,
                                    Event domain_ready,
                                    bool create = false);
      void create_pending_cross_product(IndexPartition handle1,
                                        IndexPartition handle2,
                  std::map<DomainPoint,IndexPartition> &our_handles,
                  std::map<DomainPoint,IndexPartition> &user_handles,
                                           PartitionKind kind,
                                           ColorPoint &part_color,
                                           bool allocable,
                                           Event handle_ready,
                                           Event domain_ready);
      Event create_partition_by_field(RegionTreeContext ctx,
                                      Processor local_proc,
                                      const RegionRequirement &req,
                                      IndexPartition pending,
                                      const Domain &color_space,
                                      Event term_event,
                                      VersionInfo &version_info);
      Event create_partition_by_image(RegionTreeContext ctx,
                                      Processor local_proc,
                                      const RegionRequirement &req,
                                      IndexPartition pending,
                                      const Domain &color_space,
                                      Event term_event,
                                      VersionInfo &version_info);
      Event create_partition_by_preimage(RegionTreeContext ctx,
                                      Processor local_proc,
                                      const RegionRequirement &req,
                                      IndexPartition projection,
                                      IndexPartition pending,
                                      const Domain &color_space,
                                      Event term_event,
                                      VersionInfo &version_info);
    public:
      IndexSpace find_pending_space(IndexPartition parent,
                                    const DomainPoint &color,
                                    UserEvent &handle_ready,
                                    UserEvent &domain_ready);
      Event compute_pending_space(IndexSpace result,
                                  const std::vector<IndexSpace> &handles,
                                  bool is_union);
      Event compute_pending_space(IndexSpace result,
                                  IndexPartition handle,
                                  bool is_union);
      Event compute_pending_space(IndexSpace result,
                                  IndexSpace initial,
                                  const std::vector<IndexSpace> &handles);
    public:
      IndexPartition get_index_partition(IndexSpace parent, 
                                         const ColorPoint &color);
      IndexSpace get_index_subspace(IndexPartition parent, 
                                    const ColorPoint &color);
      bool has_multiple_domains(IndexSpace handle);
      Domain get_index_space_domain(IndexSpace handle);
      void get_index_space_domains(IndexSpace handle,
                                   std::vector<Domain> &domains);
      Domain get_index_partition_color_space(IndexPartition p);
      void get_index_space_partition_colors(IndexSpace sp,
                                            std::set<ColorPoint> &colors);
      ColorPoint get_index_space_color(IndexSpace handle);
      ColorPoint get_index_partition_color(IndexPartition handle);
      IndexSpace get_parent_index_space(IndexPartition handle);
      bool has_parent_index_partition(IndexSpace handle);
      IndexPartition get_parent_index_partition(IndexSpace handle);
      IndexSpaceAllocator* get_index_space_allocator(IndexSpace handle);
      size_t get_domain_volume(IndexSpace handle);
      bool is_index_partition_disjoint(IndexPartition p);
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
      void invalidate_field_index(const std::set<RegionNode*> &regions,
                                  unsigned field_idx);
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
      LogicalPartition get_logical_partition_by_color(LogicalRegion parent, 
                                                      const ColorPoint &color);
      bool has_logical_partition_by_color(LogicalRegion parent,
                                          const ColorPoint &color);
      LogicalPartition get_logical_partition_by_tree(
          IndexPartition handle, FieldSpace space, RegionTreeID tid);
      LogicalRegion get_logical_subregion(LogicalPartition parent,
                                          IndexSpace handle);
      LogicalRegion get_logical_subregion_by_color(
                              LogicalPartition parent, const ColorPoint &color);
      bool has_logical_subregion_by_color(LogicalPartition parent,
                                          const ColorPoint &color);
      LogicalRegion get_logical_subregion_by_tree(
            IndexSpace handle, FieldSpace space, RegionTreeID tid);
      ColorPoint get_logical_region_color(LogicalRegion handle);
      ColorPoint get_logical_partition_color(LogicalPartition handle);
      LogicalRegion get_parent_logical_region(LogicalPartition handle);
      bool has_parent_logical_partition(LogicalRegion handle);
      LogicalPartition get_parent_logical_partition(LogicalRegion handle);
      size_t get_domain_volume(LogicalRegion handle);
    public:
      // Logical analysis methods
      void perform_dependence_analysis(Operation *op, unsigned idx,
                                       RegionRequirement &req,
                                       VersionInfo &version_info,
                                       RestrictInfo &restrict_info,
                                       RegionTreePath &path);
      void perform_reduction_close_analysis(Operation *op, unsigned idx,
                                       RegionRequirement &req,
                                       VersionInfo &version_info);
      void perform_fence_analysis(RegionTreeContext ctx, Operation *fence,
                                  LogicalRegion handle, bool dominate);
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
      void restrict_user_coherence(SingleTask *parent_ctx,
                                   LogicalRegion handle,
                                   const std::set<FieldID> &fields);
      void acquire_user_coherence(SingleTask *parent_ctx,
                                  LogicalRegion handle,
                                  const std::set<FieldID> &fields);
      bool has_restrictions(LogicalRegion handle, const RestrictInfo &info,
                            const std::set<FieldID> &fields);
    public:
      // Physical analysis methods
      bool premap_physical_region(RegionTreeContext ctx,
                                  RegionTreePath &path,
                                  RegionRequirement &req,
                                  VersionInfo &version_info,
                                  Operation *op,
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
                                     VersionInfo &version_info,
                                     Operation *op,
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
                                       VersionInfo &version_info,
                                       const InstanceRef &ref
#ifdef DEBUG_HIGH_LEVEL
                                       , const char *log_name
                                       , UniqueID uid
#endif
                                       );
      // This call will not actually perform a traversal
      // but will instead compute the proper view on which
      // to perform the mapping based on a target instance
      MappingRef map_restricted_region(RegionTreeContext ctx,
                                       RegionRequirement &req,
                                       unsigned index,
                                       VersionInfo &version_info,
                                       Processor target_proc
#ifdef DEBUG_HIGH_LEVEL
                                       , const char *log_name
                                       , UniqueID uid
#endif
                                       );
      // Same as the call above, but with a mapping path
      MappingRef map_restricted_region(RegionTreeContext ctx,
                                       RegionTreePath &path,
                                       RegionRequirement &req,
                                       unsigned index,
                                       VersionInfo &version_info,
                                       Processor target_proc
#ifdef DEBUG_HIGH_LEVEL
                                       , const char *log_name
                                       , UniqueID uid
#endif
                                       );
      InstanceRef register_physical_region(RegionTreeContext ctx,
                                           const MappingRef &ref,
                                           RegionRequirement &req,
                                           unsigned idx,
                                           VersionInfo &version_info,
                                           Operation *op,
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
      bool perform_close_operation(RegionTreeContext ctx,
                                   RegionRequirement &req,
                                   SingleTask *parent_ctx,
                                   Processor local_proc,
                                   const std::set<ColorPoint> &targets,
                                   bool leave_open,
                                   const std::set<ColorPoint> &next_children,
                                   Event &closed,
                                   const MappingRef &target,
                                   VersionInfo &version_info,
                                   bool force_composite
#ifdef DEBUG_HIGH_LEVEL
                                   , unsigned index
                                   , const char *log_name
                                   , UniqueID uid
#endif
                                   );


      Event close_physical_context(RegionTreeContext ctx,
                                   RegionRequirement &req,
                                   VersionInfo &version_info,
                                   Operation *op,
                                   Processor local_proc,
                                   const InstanceRef &ref
#ifdef DEBUG_HIGH_LEVEL
                                   , unsigned index
                                   , const char *log_name
                                   , UniqueID uid
#endif
                                   );
      Event copy_across(Operation *op,
                        Processor local_proc,
                        RegionTreeContext src_ctx,
                        RegionTreeContext dst_ctx,
                        RegionRequirement &src_req,
                        VersionInfo &src_version_info,
                        const RegionRequirement &dst_req,
                        const InstanceRef &dst_ref,
                        Event precondition);
      Event copy_across(Operation *op,
                        RegionTreeContext src_ctx, 
                        RegionTreeContext dst_ctx,
                        const RegionRequirement &src_req,
                        const RegionRequirement &dst_req,
                        const InstanceRef &src_ref,
                        const InstanceRef &dst_ref,
                        Event precondition);
      Event reduce_across(Operation *op,
                        Processor local_proc,
                        RegionTreeContext src_ctx,
                        RegionTreeContext dst_ctx,
                        RegionRequirement &src_req,
                        VersionInfo &version_info,
                        const RegionRequirement &dst_req,
                        const InstanceRef &dst_ref,
                        Event precondition);
      Event reduce_across(Operation *op,
                        RegionTreeContext src_ctx, 
                        RegionTreeContext dst_ctx,
                        const RegionRequirement &src_req,
                        const RegionRequirement &dst_req,
                        const InstanceRef &src_ref,
                        const InstanceRef &dst_ref,
                        Event precondition);
      // This takes ownership of the value buffer
      void fill_fields(RegionTreeContext ctx,
                       const RegionRequirement &req,
                       const void *value, size_t value_size,
                       VersionInfo &version_info);
      InstanceRef attach_file(RegionTreeContext ctx,
                              const RegionRequirement &req,
                              AttachOp *attach_op,
                              VersionInfo &version_info);
      void detach_file(RegionTreeContext ctx, 
                       const RegionRequirement &req,
                       const InstanceRef &ref);
    public:
      // Methods for sending and returning state information
      void send_remote_references(
          const std::set<PhysicalManager*> &needed_managers,
          AddressSpaceID target);
      void send_remote_references(
          const LegionMap<LogicalView*,FieldMask>::aligned &needed_views,
          const std::set<PhysicalManager*> &needed_managers, 
          AddressSpaceID target);
      void handle_remote_references(Deserializer &derez);
    public:
      // Debugging method for checking context state
      void check_context_state(RegionTreeContext ctx);
    public:
      // We know the domain of the index space
      IndexSpaceNode* create_node(IndexSpace is, const Domain &d, 
                                  IndexPartNode *par, ColorPoint color,
                                  IndexSpaceKind kind, AllocateMode mode);
      // Give the event for when the domain is ready
      IndexSpaceNode* create_node(IndexSpace is, const Domain &d, Event ready,
                                  IndexPartNode *par, ColorPoint color,
                                  IndexSpaceKind kind, AllocateMode mode);
      // Give two events for when the domain handle and domain are ready
      IndexSpaceNode* create_node(IndexSpace is, 
                                  Event handle_ready, Event domain_ready,
                                  IndexPartNode *par, ColorPoint color,
                                  IndexSpaceKind kind, AllocateMode mode);
      // We know the disjointness of the index partition
      IndexPartNode*  create_node(IndexPartition p, IndexSpaceNode *par,
                                  ColorPoint color, Domain color_space, 
                                  bool disjoint, AllocateMode mode);
      // Give the event for when the disjointness information is ready
      IndexPartNode*  create_node(IndexPartition p, IndexSpaceNode *par,
                                  ColorPoint color, Domain color_space,
                                  Event ready_event, AllocateMode mode);
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
      bool is_subregion(LogicalRegion child, LogicalRegion parent);
      bool is_disjoint(IndexPartition handle);
      bool is_disjoint(LogicalPartition handle);
      bool are_disjoint(IndexSpace parent, IndexSpace child);
      bool are_disjoint(IndexSpace parent, IndexPartition child);
      bool are_compatible(IndexSpace left, IndexSpace right);
      bool is_dominated(IndexSpace src, IndexSpace dst);
      bool compute_index_path(IndexSpace parent, IndexSpace child,
                              std::vector<ColorPoint> &path);
      bool compute_partition_path(IndexSpace parent, IndexPartition child,
                                  std::vector<ColorPoint> &path); 
      void initialize_path(IndexSpace child, IndexSpace parent,
                           RegionTreePath &path);
      void initialize_path(IndexPartition child, IndexSpace parent,
                           RegionTreePath &path);
      void initialize_path(IndexSpace child, IndexPartition parent,
                           RegionTreePath &path);
      void initialize_path(IndexPartition child, IndexPartition parent,
                           RegionTreePath &path);
      FatTreePath* compute_fat_path(IndexSpace child, IndexSpace parent,
                               std::map<IndexTreeNode*,FatTreePath*> &storage,
                               bool test_overlap, bool &overlap);
      FatTreePath* compute_fat_path(IndexSpace child, 
                                    IndexPartition parent,
                               std::map<IndexTreeNode*,FatTreePath*> &storage,
                               bool test_overlap, bool &overlap);
    protected:
      FatTreePath* compute_fat_path(IndexTreeNode *child, IndexTreeNode *parent,
                                 std::map<IndexTreeNode*,FatTreePath*> &storage,
                                 bool test_overlap, bool &overlap);
    public:
      // Interfaces to the low-level runtime
      Event issue_copy(const Domain &dom, Operation *op,
                       const std::vector<Domain::CopySrcDstField> &src_fields,
                       const std::vector<Domain::CopySrcDstField> &dst_fields,
                       Event precondition = Event::NO_EVENT);
      Event issue_fill(const Domain &dom, Operation *op,
                       const std::vector<Domain::CopySrcDstField> &dst_fields,
                       const void *fill_value, size_t fill_size,
                       Event precondition = Event::NO_EVENT);
      Event issue_reduction_copy(const Domain &dom, Operation *op,
                       ReductionOpID redop, bool reduction_fold,
                       const std::vector<Domain::CopySrcDstField> &src_fields,
                       const std::vector<Domain::CopySrcDstField> &dst_fields,
                       Event precondition = Event::NO_EVENT);
      Event issue_indirect_copy(const Domain &dom, Operation *op,
                       const Domain::CopySrcDstField &idx,
                       ReductionOpID redop, bool reduction_fold,
                       const std::vector<Domain::CopySrcDstField> &src_fields,
                       const std::vector<Domain::CopySrcDstField> &dst_fields,
                       Event precondition = Event::NO_EVENT);
      PhysicalInstance create_instance(const Domain &dom, Memory target, 
                                       size_t field_size, Operation *op);
      PhysicalInstance create_instance(const Domain &dom, Memory target,
                                       const std::vector<size_t> &field_sizes,
                                       size_t blocking_factor, Operation *op);
      PhysicalInstance create_instance(const Domain &dom, Memory target,
                                       size_t field_size, ReductionOpID redop,
                                       Operation *op);
    protected:
      void initialize_path(IndexTreeNode* child, IndexTreeNode *parent,
                           RegionTreePath &path);
    public:
      template<typename T>
      Color generate_unique_color(const std::map<Color,T> &current_map);
#ifdef DEBUG_HIGH_LEVEL
    public:
      // These are debugging methods and are never called from
      // actual code, therefore they never take locks
      void dump_logical_state(LogicalRegion region, ContextID ctx);
      void dump_physical_state(LogicalRegion region, ContextID ctx);
#endif
    public:
      void attach_semantic_information(IndexSpace handle, SemanticTag tag,
                                       const NodeSet &source_mask,
                                       const void *buffer, size_t size);
      void attach_semantic_information(IndexPartition handle, SemanticTag tag,
                                       const NodeSet &source_mask,
                                       const void *buffer, size_t size);
      void attach_semantic_information(FieldSpace handle, SemanticTag tag,
                                       const NodeSet &source_mask,
                                       const void *buffer, size_t size);
      void attach_semantic_information(FieldSpace handle, FieldID fid,
                                       SemanticTag tag, const NodeSet &source,
                                       const void *buffer, size_t size);
      void attach_semantic_information(LogicalRegion handle, SemanticTag tag,
                                       const NodeSet &source_mask,
                                       const void *buffer, size_t size);
      void attach_semantic_information(LogicalPartition handle, SemanticTag tag,
                                       const NodeSet &source_mask,
                                       const void *buffer, size_t size);
    public:
      void retrieve_semantic_information(IndexSpace handle, SemanticTag tag,
                                         const void *&result, size_t &size);
      void retrieve_semantic_information(IndexPartition handle, SemanticTag tag,
                                         const void *&result, size_t &size);
      void retrieve_semantic_information(FieldSpace handle, SemanticTag tag,
                                         const void *&result, size_t &size);
      void retrieve_semantic_information(FieldSpace handle, FieldID fid,
                                         SemanticTag tag,
                                         const void *&result, size_t &size);
      void retrieve_semantic_information(LogicalRegion handle, SemanticTag tag,
                                         const void *&result, size_t &size);
      void retrieve_semantic_information(LogicalPartition part, SemanticTag tag,
                                         const void *&result, size_t &size);
    public:
      Runtime *const runtime;
    protected:
      Reservation lookup_lock;
    private:
      // The lookup lock must be held when accessing these
      // data structures
      std::map<IndexSpace,IndexSpaceNode*>     index_nodes;
      std::map<IndexPartition,IndexPartNode*>  index_parts;
      std::map<FieldSpace,FieldSpaceNode*>     field_nodes;
      std::map<LogicalRegion,RegionNode*>     region_nodes;
      std::map<LogicalPartition,PartitionNode*> part_nodes;
      std::map<RegionTreeID,RegionNode*>        tree_nodes;
    public:
      static bool are_disjoint(const Domain &left,
                               const Domain &right);
      static bool are_disjoint(IndexSpaceNode *left,
                               IndexSpaceNode *right);
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
        unsigned long long max_time;
        unsigned long long min_time;
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
      std::vector<std::vector<PerfTrace> > traces;
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
      PERFORM_CLOSE_OPERATIONS_ANALYSIS,
    };

    enum CallKind {
      CREATE_NODE_CALL,
      GET_NODE_CALL,
      ARE_DISJOINT_CALL,
      COMPUTE_PATH_CALL,
      CREATE_INSTANCE_CALL,
      CREATE_REDUCTION_CALL,
      PERFORM_PREMAP_CLOSE_CALL,
      MAPPING_TRAVERSE_CALL,
      MAP_PHYSICAL_REGION_CALL,
      MAP_REDUCTION_REGION_CALL,
      REGISTER_LOGICAL_NODE_CALL,
      OPEN_LOGICAL_NODE_CALL,
      CLOSE_LOGICAL_NODE_CALL,
      SIPHON_LOGICAL_CHILDREN_CALL,
      PERFORM_LOGICAL_CLOSE_CALL,
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
     * \struct SemanticInfo
     * A struct for storing semantic information for various things
     */
    struct SemanticInfo {
    public:
      SemanticInfo(void)
        : buffer(NULL), size(0) { }  
      SemanticInfo(void *buf, size_t s, const NodeSet &init)
        : buffer(buf), size(s), node_mask(init) { }
    public:
      void *buffer;
      size_t size;
      NodeSet node_mask;
    };

    enum SemanticInfoKind {
      INDEX_SPACE_SEMANTIC,
      INDEX_PARTITION_SEMANTIC,
      FIELD_SPACE_SEMANTIC,
      FIELD_SEMANTIC,
      LOGICAL_REGION_SEMANTIC,
      LOGICAL_PARTITION_SEMANTIC,
    };

    template<SemanticInfoKind KIND>
    struct SendSemanticInfoFunctor {
    public:
      SendSemanticInfoFunctor(Runtime *rt, Serializer &r)
        : runtime(rt), rez(r) { }
    public:
      void apply(AddressSpaceID target);
    private:
      Runtime *runtime;
      Serializer &rez;
    }; 

    /**
     * \class IndexTreeNode
     * The abstract base class for nodes in the index space trees.
     */
    class IndexTreeNode {
    public:
      struct IntersectInfo {
      public:
        IntersectInfo(void)
          : has_intersects(false),
            intersections_valid(false) { }
        IntersectInfo(bool has)
          : has_intersects(has), 
            intersections_valid(!has) { }
        IntersectInfo(const std::set<Domain> &ds)
          : has_intersects(true), intersections_valid(true),
            intersections(ds) { }
      public:
        bool has_intersects;
        bool intersections_valid;
        std::set<Domain> intersections;
      };
    public:
      IndexTreeNode(void);
      IndexTreeNode(ColorPoint color, unsigned depth, RegionTreeForest *ctx); 
      virtual ~IndexTreeNode(void);
    public:
      virtual IndexTreeNode* get_parent(void) const = 0;
      virtual size_t get_num_elmts(void) = 0;
      virtual void send_node(AddressSpaceID target, bool up, bool down) = 0;
    public:
      virtual bool is_index_space_node(void) const = 0;
      virtual IndexSpaceNode* as_index_space_node(void) = 0;
      virtual IndexPartNode* as_index_part_node(void) = 0;
    public:
      void attach_semantic_information(SemanticTag tag, const NodeSet &mask,
                                       const void *buffer, size_t size);
      void retrieve_semantic_information(SemanticTag tag,
                                         const void *&result, size_t &size);
      virtual void send_semantic_info(const NodeSet &targets, SemanticTag tag,
                                      const void *buffer, size_t size, 
                                      const NodeSet &current) = 0;
    public:
      static bool compute_intersections(const std::set<Domain> &left,
                                        const std::set<Domain> &right,
                                        std::set<Domain> &result,
                                        bool compute);
      static bool compute_intersections(const std::set<Domain> &left,
                                        const Domain &right,
                                        std::set<Domain> &result,
                                        bool compute);
      static bool compute_intersection(const Domain &left,
                                       const Domain &right,
                                       Domain &result, bool compute);
      static bool compute_dominates(const std::set<Domain> &left_set,
                                    const std::set<Domain> &right_set);
    public:
      const unsigned depth;
      const ColorPoint color;
      RegionTreeForest *const context;
    public:
      NodeSet creation_set;
      NodeSet child_creation;
      NodeSet destruction_set;
    protected:
      Reservation node_lock;
    protected:
      std::map<IndexTreeNode*,IntersectInfo> intersections;
      std::map<IndexTreeNode*,bool> dominators;
    protected:
      LegionMap<SemanticTag,SemanticInfo>::aligned semantic_info;
    protected:
      std::map<std::pair<ColorPoint,ColorPoint>,Event> pending_tests;
    };

    /**
     * \class IndexSpaceNode
     * A class for representing a generic index space node.
     */
    class IndexSpaceNode : public IndexTreeNode {
    public:
      struct DynamicIndependenceArgs {
        HLRTaskID hlr_id;
        IndexSpaceNode *parent;
        IndexPartNode *left, *right;
      };
    public:
      IndexSpaceNode(IndexSpace handle, const Domain &d, 
                     IndexPartNode *par, ColorPoint c,
                     IndexSpaceKind kind, AllocateMode mode,
                     RegionTreeForest *ctx);
      IndexSpaceNode(IndexSpace handle, const Domain &d, Event ready,
                     IndexPartNode *par, ColorPoint c,
                     IndexSpaceKind kind, AllocateMode mode,
                     RegionTreeForest *ctx);
      IndexSpaceNode(IndexSpace handle, Event handle_ready, Event dom_ready,
                     IndexPartNode *par, ColorPoint c,
                     IndexSpaceKind kind, AllocateMode mode,
                     RegionTreeForest *ctx);
      IndexSpaceNode(const IndexSpaceNode &rhs);
      virtual ~IndexSpaceNode(void);
    public:
      IndexSpaceNode& operator=(const IndexSpaceNode &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      virtual bool is_index_space_node(void) const;
      virtual IndexSpaceNode* as_index_space_node(void);
      virtual IndexPartNode* as_index_part_node(void);
    public:
      virtual IndexTreeNode* get_parent(void) const;
      virtual size_t get_num_elmts(void);
    public:
      virtual void send_semantic_info(const NodeSet &targets, SemanticTag tag,
                                      const void *buffer, size_t size,
                                      const NodeSet &current);
      static void handle_semantic_info(RegionTreeForest *forest,
                                       Deserializer &derez);
    public:
      bool has_child(const ColorPoint &c);
      IndexPartNode* get_child(const ColorPoint &c);
      void add_child(IndexPartNode *child);
      void remove_child(const ColorPoint &c);
      size_t get_num_children(void) const;
      void get_children(std::map<ColorPoint,IndexPartNode*> &children);
      void get_child_colors(std::set<ColorPoint> &colors, bool only_valid);
    public:
      Event get_domain_precondition(void);
      const Domain& get_domain_blocking(void);
      const Domain& get_domain(Event &ready_event);
      const Domain& get_domain_no_wait(void);
      void set_domain(const Domain &dom);
      void get_domains_blocking(std::vector<Domain> &domains);
      void get_domains(std::vector<Domain> &domains, Event &precondition);
      size_t get_domain_volume(bool app_query = false);
    public:
      bool are_disjoint(const ColorPoint &c1, const ColorPoint &c2); 
      void record_disjointness(bool disjoint, 
                               const ColorPoint &c1, const ColorPoint &c2);
      Color generate_color(void);
      void get_colors(std::set<ColorPoint> &colors);
    public:
      void add_instance(RegionNode *inst);
      bool has_instance(RegionTreeID tid);
      void add_creation_source(AddressSpaceID source);
      void destroy_node(AddressSpaceID source);
    public:
      bool has_component_domains(void) const;
      void update_component_domains(const std::set<Domain> &domains);
      const std::set<Domain>& get_component_domains_blocking(void) const;
      const std::set<Domain>& get_component_domains(Event &precondition) const;
      bool intersects_with(IndexSpaceNode *other, bool compute = true);
      bool intersects_with(IndexPartNode *other, bool compute = true);
      const std::set<Domain>& get_intersection_domains(IndexSpaceNode *other);
      const std::set<Domain>& get_intersection_domains(IndexPartNode *other);
      bool dominates(IndexSpaceNode *other);
      bool dominates(IndexPartNode *other);
    public:
      Event create_subspaces_by_field(
          const std::vector<FieldDataDescriptor> &field_data,
          std::map<DomainPoint, LowLevel::IndexSpace> &subspaces,
          bool mutable_results, Event precondition);
      Event create_subspaces_by_image(
          const std::vector<FieldDataDescriptor> &field_data,
          std::map<LowLevel::IndexSpace, LowLevel::IndexSpace> &subpsaces,
          bool mutable_results, Event precondition);
      Event create_subspaces_by_preimage(
          const std::vector<FieldDataDescriptor> &field_data,
          std::map<LowLevel::IndexSpace, LowLevel::IndexSpace> &subspaces,
          bool mutable_results, Event precondition);
    public:
      static void handle_disjointness_test(IndexSpaceNode *parent,
                                           IndexPartNode *left,
                                           IndexPartNode *right);
    public:
      virtual void send_node(AddressSpaceID target, bool up, bool down);
      static void handle_node_creation(RegionTreeForest *context,
                                       Deserializer &derez, 
                                       AddressSpaceID source);
    public:
      static void handle_node_request(RegionTreeForest *context,
                                      Deserializer &derez,
                                      AddressSpaceID source);
      static void handle_node_return(Deserializer &derez);
    public:
      IndexSpaceAllocator* get_allocator(void);
    public:
      const IndexSpace handle;
      IndexPartNode *const parent;
      const IndexSpaceKind kind;
      const AllocateMode mode;
    protected:
      // Track when the domain handle is ready
      Event handle_ready;
      // Track when the domain has actually been computed
      Event domain_ready;
      Domain domain;
    protected:
      // Must hold the node lock when accessing the
      // remaining data structures
      // Color map is all children seen ever
      std::map<ColorPoint,IndexPartNode*> color_map;
      // Valid map is all chidlren that haven't been deleted
      std::map<ColorPoint,IndexPartNode*> valid_map;
      std::set<RegionNode*> logical_nodes;
      std::set<std::pair<ColorPoint,ColorPoint> > disjoint_subsets;
      std::set<std::pair<ColorPoint,ColorPoint> > aliased_subsets;
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
      struct DynamicIndependenceArgs {
        HLRTaskID hlr_id;
        IndexPartNode *parent;
        IndexSpaceNode *left, *right;
      };
      struct PendingChildArgs {
        HLRTaskID hlr_id;
        IndexPartNode *parent;
        ColorPoint pending_child;
      };
    public:
      IndexPartNode(IndexPartition p, IndexSpaceNode *par,
                    ColorPoint c, Domain color_space, 
                    bool disjoint, AllocateMode mode,
                    RegionTreeForest *ctx);
      IndexPartNode(IndexPartition p, IndexSpaceNode *par,
                    ColorPoint c, Domain color_space,
                    Event ready_event, AllocateMode mode,
                    RegionTreeForest *ctx);
      IndexPartNode(const IndexPartNode &rhs);
      virtual ~IndexPartNode(void);
    public:
      IndexPartNode& operator=(const IndexPartNode &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      virtual bool is_index_space_node(void) const;
      virtual IndexSpaceNode* as_index_space_node(void);
      virtual IndexPartNode* as_index_part_node(void);
    public:
      virtual IndexTreeNode* get_parent(void) const;
      virtual size_t get_num_elmts(void);
    public:
      virtual void send_semantic_info(const NodeSet &targets, SemanticTag tag,
                                      const void *buffer, size_t size,
                                      const NodeSet &current);
      static void handle_semantic_info(RegionTreeForest *forest,
                                       Deserializer &derez);
    public:
      bool has_child(const ColorPoint &c);
      IndexSpaceNode* get_child(const ColorPoint &c);
      void add_child(IndexSpaceNode *child);
      void remove_child(const ColorPoint &c);
      size_t get_num_children(void) const;
      void get_children(std::map<ColorPoint,IndexSpaceNode*> &children);
    public:
      void compute_disjointness(UserEvent ready_event);
      bool is_disjoint(bool from_app = false);
      bool are_disjoint(const ColorPoint &c1, const ColorPoint &c2,
                        bool force_compute = false);
      void record_disjointness(bool disjoint,
                               const ColorPoint &c1, const ColorPoint &c2);
      bool is_complete(void);
      void get_colors(std::set<ColorPoint> &colors);
    public:
      void add_instance(PartitionNode *inst);
      bool has_instance(RegionTreeID tid);
      void add_creation_source(AddressSpaceID source);
      void destroy_node(AddressSpaceID source);
    public:
      void add_pending_child(const ColorPoint &child_color,
                             UserEvent handle_ready, UserEvent domain_ready);
      bool get_pending_child(const ColorPoint &child_color,
                             UserEvent &handle_ready, UserEvent &domain_ready);
      void remove_pending_child(const ColorPoint &child_color);
      static void handle_pending_child_task(const void *args);
    public:
      Event create_equal_children(size_t granularity);
      Event create_weighted_children(const std::map<DomainPoint,int> &weights,
                                     size_t granularity);
      Event create_by_operation(IndexPartNode *left, IndexPartNode *right,
                                LowLevel::IndexSpace::IndexSpaceOperation op);
      Event create_by_operation(IndexSpaceNode *left, IndexPartNode *right,
                                LowLevel::IndexSpace::IndexSpaceOperation op);
    public:
      void get_subspace_domain_preconditions(std::set<Event> &preconditions);
      void get_subspace_domains(std::set<Domain> &subspaces);
      bool intersects_with(IndexSpaceNode *other, bool compute = true);
      bool intersects_with(IndexPartNode *other, bool compute = true);
      const std::set<Domain>& get_intersection_domains(IndexSpaceNode *other);
      const std::set<Domain>& get_intersection_domains(IndexPartNode *other);
      bool dominates(IndexSpaceNode *other);
      bool dominates(IndexPartNode *other);
    public:
      static void handle_disjointness_test(IndexPartNode *parent,
                                           IndexSpaceNode *left,
                                           IndexSpaceNode *right);
    public:
      virtual void send_node(AddressSpaceID target, bool up, bool down);
      static void handle_node_creation(RegionTreeForest *context,
                                       Deserializer &derez, 
                                       AddressSpaceID source);
    public:
      static void handle_node_request(RegionTreeForest *context,
                                      Deserializer &derez,
                                      AddressSpaceID source);
      static void handle_node_return(Deserializer &derez);
    public:
      const IndexPartition handle;
      const Domain color_space;
      const AllocateMode mode;
      IndexSpaceNode *const parent;
    protected:
      bool disjoint;
      Event disjoint_ready;
    protected:
      bool has_complete, complete;
    protected:
      // Must hold the node lock when accessing
      // the remaining data structures
      std::map<ColorPoint,IndexSpaceNode*> color_map;
      std::map<ColorPoint,IndexSpaceNode*> valid_map;
      std::set<PartitionNode*> logical_nodes;
      std::set<std::pair<ColorPoint,ColorPoint> > disjoint_subspaces;
      std::set<std::pair<ColorPoint,ColorPoint> > aliased_subspaces;
    protected:
      // Support for pending child spaces that still need to be computed
      std::map<ColorPoint,std::pair<UserEvent,UserEvent> > pending_children;
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
      struct SendFieldAllocationFunctor {
      public:
        SendFieldAllocationFunctor(FieldSpace h, FieldID f, size_t s,
                                   unsigned i, Runtime *rt)
          : handle(h), field(f), size(s), index(i), runtime(rt) { }
      public:
        void apply(AddressSpaceID target);
      private:
        FieldSpace handle;
        FieldID field;
        size_t size;
        unsigned index;
        Runtime *runtime;
      };
      struct SendFieldDestructionFunctor {
      public:
        SendFieldDestructionFunctor(FieldSpace h, FieldID f, Runtime *rt)
          : handle(h), field(f), runtime(rt) { }
      public:
        void apply(AddressSpaceID target);
      private:
        FieldSpace handle;
        FieldID field;
        Runtime *runtime;
      };
    public:
      FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx);
      FieldSpaceNode(const FieldSpaceNode &rhs);
      ~FieldSpaceNode(void);
    public:
      FieldSpaceNode& operator=(const FieldSpaceNode &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      void attach_semantic_information(SemanticTag tag, 
                                       const NodeSet &sources,
                                       const void *buffer, size_t size);
      void attach_semantic_information(FieldID fid, SemanticTag tag,
                                       const NodeSet &sources,
                                       const void *buffer, size_t size);
      void retrieve_semantic_information(SemanticTag tag,
                                         const void *&result, size_t &size);
      void retrieve_semantic_information(FieldID fid, SemanticTag tag,
                                         const void *&result, size_t &size);
      static void handle_semantic_info(RegionTreeForest *forest,
                                       Deserializer &derez);
      static void handle_field_semantic_info(RegionTreeForest *forest,
                                             Deserializer &derez);
    public:
      void allocate_field(FieldID fid, size_t size, bool local);
      void allocate_field_index(FieldID fid, size_t size, 
                                AddressSpaceID runtime, unsigned index);
      void free_field(FieldID fid, AddressSpaceID source);
      bool has_field(FieldID fid);
      size_t get_field_size(FieldID fid);
      void get_all_fields(std::set<FieldID> &to_set);
      void get_all_regions(std::set<LogicalRegion> &regions);
      void get_field_set(const FieldMask &mask, std::set<FieldID> &to_set);
      void get_field_set(const FieldMask &mask, const std::set<FieldID> &basis,
                         std::set<FieldID> &to_set);
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
    protected:
      void compute_create_offsets(const std::set<FieldID> &create_fields,
                                  std::vector<size_t> &field_sizes,
                                  std::vector<unsigned> &indexes);
    public:
      InstanceManager* create_instance(Memory location, Domain dom,
                                       const std::set<FieldID> &fields,
                                       size_t blocking_factor, unsigned depth,
                                       RegionNode *node, Operation *op);
      ReductionManager* create_reduction(Memory location, Domain dom,
                                        FieldID fid, bool reduction_list,
                                        RegionNode *node, ReductionOpID redop,
                                        Operation *op);
    public:
      InstanceManager* create_file_instance(const std::set<FieldID> &fields,
                                            const FieldMask &attach_mask,
                                            RegionNode *node, AttachOp *op);
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
      static void handle_node_request(RegionTreeForest *context,
                                      Deserializer &derez,
                                      AddressSpaceID source);
      static void handle_node_return(Deserializer &derez);
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
      NodeSet creation_set;
      NodeSet destruction_set;
    private:
      Reservation node_lock;
      // Top nodes in the trees for which this field space is used
      std::set<RegionNode*> logical_nodes;
      std::map<FieldID,FieldInfo> fields;
      FieldMask allocated_indexes;
      int next_allocation_index; // for use in the random case
      /*
       * Every field space contains a permutation transformer that
       * can translate a field mask from any other node onto
       * this node.
       */
      LegionMap<AddressSpaceID,FieldPermutation>::aligned transformers;
    private:
      // Keep track of the layouts associated with this field space
      // Index them by their hash of their field mask to help
      // differentiate them.
      std::map<FIELD_TYPE,LegionList<LayoutDescription*,
                          LAYOUT_DESCRIPTION_ALLOC>::tracked> layouts;
    private:
      LegionMap<SemanticTag,SemanticInfo>::aligned semantic_info;
      LegionMap<std::pair<FieldID,SemanticTag>,SemanticInfo>::aligned 
                                                    semantic_field_info;
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
        NodeInfo(void)
          : physical_state(NULL), field_versions(NULL), premap_only(false), 
            path_only(false), advance(false), needs_capture(true) { }
        // Always make deep copies of the physical state
        NodeInfo(const NodeInfo &rhs);
        ~NodeInfo(void);
        NodeInfo& operator=(const NodeInfo &rhs);
      public:
        PhysicalState *physical_state;
        FieldVersions *field_versions;
        bool premap_only; // state needed for premapping only
        bool path_only; // state needed for intermediate path
        bool advance;
        bool needs_capture;
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
      inline void set_advance(void) { advance = true; }
      inline bool will_advance(void) { return advance; }
    public:
      void set_upper_bound_node(RegionTreeNode *node);
      inline bool is_upper_bound_node(RegionTreeNode *node) const
        { return (node == upper_bound_node); }
    public:
      void merge(const VersionInfo &rhs, const FieldMask &mask);
      void apply_premapping(ContextID ctx);
      void apply_mapping(ContextID ctx);
      void reset(void);
      void release(void);
      void clear(void);
      void sanity_check(RegionTreeNode *node);
    public:
      PhysicalState* find_physical_state(RegionTreeNode *node); 
      PhysicalState* create_physical_state(RegionTreeNode *node,
                                           VersionManager *manager,
                                           bool initialize, bool capture);
      FieldVersions* get_versions(RegionTreeNode *node) const;
    public:
      void pack_version_info(Serializer &rez, AddressSpaceID local_space,
                             ContextID ctx);
      void unpack_version_info(Deserializer &derez);
      void make_local(std::set<Event> &preconditions, RegionTreeForest *forest,
                      ContextID ctx, bool path_only = false);
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
      std::map<RegionTreeNode*,NodeInfo> node_infos;
      RegionTreeNode *upper_bound_node;
      bool advance;
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
      ChildState(const ChildState &rhs) 
      {
        valid_fields = rhs.valid_fields;
        open_children = rhs.open_children;
      }
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
     * \struct LogicalState
     * Track the version states for a given logical
     * region as well as the previous and current
     * epoch users and any close operations that
     * needed to be performed.
     */
    struct LogicalState {
    public:
      static const AllocationType alloc_type = LOGICAL_STATE_ALLOC;
    public:
      LogicalState(RegionTreeNode *owner, ContextID ctx);
      LogicalState(const LogicalState &state);
      ~LogicalState(void);
    public:
      LogicalState& operator=(const LogicalState &rhs);
      void* operator new(size_t count);
      void* operator new[](size_t count);
      void operator delete(void *ptr);
      void operator delete[](void *ptr);
    public:
      void reset(void);
    public:
      LegionList<FieldState,
                 LOGICAL_FIELD_STATE_ALLOC>::track_aligned field_states;
      LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned 
                                                            curr_epoch_users;
      LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned 
                                                            prev_epoch_users;
      LegionMap<VersionID,FieldMask,VERSION_ID_ALLOC>::track_aligned
                                                            field_versions;
      // Fields for which we have outstanding local reductions
      FieldMask outstanding_reduction_fields;
      LegionMap<ReductionOpID,FieldMask>::aligned outstanding_reductions;
      // Fields which we know have been mutated below in the region tree
      FieldMask dirty_below;
      // Fields on which the user has 
      // asked for explicit coherence
      FieldMask restricted_fields;
    };

    typedef DynamicTableAllocator<LogicalState, 10, 8> LogicalStateAllocator;
 
    /**
     * \struct LogicalCloser
     * This structure helps keep track of the state
     * necessary for performing a close operation
     * on the logical region tree.
     */
    struct LogicalCloser {
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
        LegionList<LogicalUser,CLOSE_LOGICAL_ALLOC>::track_aligned child_users;
      };
      struct ClosingSet {
      public:
        ClosingSet(void) { }
        ClosingSet(const FieldMask &m)
          : closing_mask(m) { }
      public:
        FieldMask closing_mask;
        std::set<ColorPoint> children;
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
      const FieldMask& get_closed_mask(void) const { return closed_mask; }
      void record_closed_child(const ColorPoint &child, const FieldMask &mask,
                               bool leave_open);
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
      void register_close_operations(
              LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &users);
      void record_version_numbers(RegionTreeNode *node, LogicalState &state,
                                  const FieldMask &local_mask);
      void merge_version_info(VersionInfo &target, const FieldMask &merge_mask);
    protected:
      static void compute_close_sets(
                     const LegionMap<ColorPoint,ClosingInfo>::aligned &children,
                     LegionList<ClosingSet>::aligned &close_sets);
      void create_close_operations(RegionTreeNode *target, Operation *creator,
                          const VersionInfo &version_info,
                          const RestrictInfo &restrict_info, 
                          const TraceInfo &trace_info, bool open,
                          const LegionList<ClosingSet>::aligned &close_sets,
                      LegionMap<InterCloseOp*,LogicalUser>::aligned &close_ops);
      void register_dependences(const LogicalUser &current, 
                                const FieldMask &open_below,
             LegionMap<InterCloseOp*,LogicalUser>::aligned &closes,
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
      FieldMask closed_mask;
      LegionMap<ColorPoint,ClosingInfo>::aligned leave_open_children;
      LegionMap<ColorPoint,ClosingInfo>::aligned force_close_children;
    protected:
      LegionMap<InterCloseOp*,LogicalUser>::aligned leave_open_closes;
      LegionMap<InterCloseOp*,LogicalUser>::aligned force_close_closes;
    protected:
      VersionInfo close_versions;
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
    struct PhysicalCloser : public CopyTracker {
    public:
      PhysicalCloser(const MappableInfo &info,
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
      void update_node_views(RegionTreeNode *node, PhysicalState *state);
    public:
      const MappableInfo &info;
      const LogicalRegion handle;
      const bool permit_leave_open;
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
    struct CompositeCloser {
    public:
      CompositeCloser(ContextID ctx, VersionInfo &version_info,
                      bool permit_leave_open);
      CompositeCloser(const CompositeCloser &rhs);
      ~CompositeCloser(void);
    public:
      CompositeCloser& operator=(const CompositeCloser &rhs);
    public:
      CompositeNode* get_composite_node(RegionTreeNode *tree_node,
                                        CompositeNode *parent);
      void create_valid_view(PhysicalState *state,
                             CompositeNode *root,
                             const FieldMask &closed_mask);
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
      const ContextID ctx;
      const bool permit_leave_open;
      VersionInfo &version_info;
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
      PhysicalState(void);
#ifdef DEBUG_HIGH_LEVEL
      PhysicalState(RegionTreeNode *node);
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
      void capture_state(bool path_only);
      void apply_state(bool advance) const;
      void reset(void);
    public:
      PhysicalState* clone(bool clone_state) const;
      PhysicalState* clone(const FieldMask &clone_mask, bool clone_state) const;
      void make_local(std::set<Event> &preconditions, bool advance);
    public:
      void print_physical_state(const FieldMask &capture_mask,
          LegionMap<ColorPoint,FieldMask>::aligned &to_traverse,
                                TreeStateLogger *logger);
    public:
      // Fields which were closed and can be ignored when applying
      FieldMask closed_mask;
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
      enum VersionMetaState {
        INVALID_VERSION_STATE,
        EVENTUAL_VERSION_STATE, // eventually consistent state
        MERGED_VERSION_STATE, // merged consistent state
      };
    public:
      class BroadcastFunctor {
      public:
        BroadcastFunctor(AddressSpaceID loc, std::deque<AddressSpaceID> &t)
          : local(loc), targets(t) { }
      public:
        inline void apply(AddressSpaceID target)
        {
          if (target != local)
            targets.push_back(target);
        }
      protected:
        AddressSpaceID local;
        std::deque<AddressSpaceID> &targets;
      };
    public:
      struct SendVersionStateArgs {
      public:
        HLRTaskID hlr_id;
        VersionState *proxy_this;
        AddressSpaceID target;
        UserEvent to_trigger;
      };
    public:
      VersionState(VersionID vid, Runtime *rt, DistributedID did,
                   AddressSpaceID owner_space, AddressSpaceID local_space, 
                   VersionManager *manager, bool initialize);
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
      void update_physical_state(PhysicalState *state, 
                                 const FieldMask &update_mask, 
                                 bool path_only) const;
      bool merge_physical_state(const PhysicalState *state, 
                                const FieldMask &merge_mask);
    public:
      virtual void notify_active(void);
      virtual void notify_inactive(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
    public:
      Event request_eventual_version_state(void);
      Event request_merged_version_state(void);
      void send_version_state(AddressSpaceID target, UserEvent to_trigger);
      void send_initialization_notice(void);
      void send_version_state_request(AddressSpaceID target, AddressSpaceID src,
                           UserEvent to_trigger, bool merged_req, bool upgrade);
      void launch_send_version_state(AddressSpaceID target, 
                                 UserEvent to_trigger, Event precondition);
      AddressSpaceID select_next_target(bool eventual, AddressSpaceID source);
    public:
      void handle_version_state_initialization(AddressSpaceID source);
      void handle_version_state_request(AddressSpaceID source, 
                       UserEvent to_trigger, bool merged_req, bool upgrade);
      void handle_version_state_response(AddressSpaceID source,
                             UserEvent to_trigger, Deserializer &derez);
      void handle_version_state_broadcast_response(UserEvent to_trigger,
                                                   Deserializer &derez);
    public:
      static void process_version_state_initialization(Runtime *rt,
                              Deserializer &derez, AddressSpaceID source);
      static void process_version_state_request(Runtime *rt, 
                                                Deserializer &derez);
      static void process_version_state_response(Runtime *rt,
                              Deserializer &derez, AddressSpaceID source);
      static void process_version_state_broadcast_response(Runtime *rt,
                                                     Deserializer &derez);
    public:
      const VersionID version_number;
      VersionManager *const manager;
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
      VersionMetaState meta_state;
      Event eventual_ready, merged_ready;
      // These are valid on the owner node only
      NodeSet eventual_nodes, merged_nodes;
      unsigned eventual_index, merged_index;
    };

    /**
     * \class VersionManager
     * This class tracks all the different versioned physical
     * state objects from the perspective of a logical region.
     * The version manager only keeps track of the most recent
     * versions for each field. Once a version state is no longer
     * valid for any fields then it releases its reference and
     * the version state objection can be collected once no
     * additional operations need to have access to it.
     */
    class VersionManager { 
    public:
      VersionManager(RegionTreeNode *owner);
      VersionManager(const VersionManager &rhs);
      ~VersionManager(void);
    public:
      VersionManager& operator=(const VersionManager &rhs);
    public:
      PhysicalState* construct_state(RegionTreeNode *node,
          const LegionMap<VersionID,FieldMask>::aligned &versions, 
          bool path_only, bool advance, bool initialize, bool capture);
      void initialize_state(LogicalView *view, Event term_event,
                            const RegionUsage &usage,
                            const FieldMask &user_mask);
      void filter_states(const FieldMask &filter_mask);
      void check_init(void);
      void clear(void);
      void sanity_check(void);
      void detach_instance(const FieldMask &mask, PhysicalManager *target);
    public:
      void print_physical_state(RegionTreeNode *node,
                                const FieldMask &capture_mask,
                          LegionMap<ColorPoint,FieldMask>::aligned &to_traverse,
                                TreeStateLogger *logger);
    protected:
      void filter_previous_states(VersionID vid, const FieldMask &filter_mask);
      void filter_current_states(VersionID vid, const FieldMask &filter_mask);
      void capture_previous_states(VersionID vid, const FieldMask &capture_mask,
                                   PhysicalState *state);
      void capture_previous_states(VersionID vid, const FieldMask &capture_mask,
                                   PhysicalState *state, FieldMask &to_create);
      void capture_current_states(VersionID vid, const FieldMask &capture_mask,
                                  PhysicalState *state, FieldMask &to_create,
                                  bool advance);
    protected:
      VersionState* create_new_version_state(VersionID vid, bool initialize);
      VersionState* create_remote_version_state(VersionID vid, 
              DistributedID did, AddressSpaceID owner_space, bool initialize);
    public:
      VersionState* find_remote_version_state(VersionID vid, DistributedID did,
                                      AddressSpaceID source, bool initialize,
                                    bool request_eventual, bool request_merged);
    public:
      RegionTreeNode *const owner;
    protected:
      Reservation version_lock;
      LegionMap<VersionID,VersionStateInfo>::aligned current_version_infos;
      LegionMap<VersionID,VersionStateInfo>::aligned previous_version_infos;
#ifdef DEBUG_HIGH_LEVEL
      // Debug only since this can grow unbounded
      LegionMap<VersionID,FieldMask>::aligned observed;
#endif
    };

    typedef DynamicTableAllocator<VersionManager,10,8> VersionManagerAllocator;

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
      LogicalState& get_logical_state(ContextID ctx);
      void set_restricted_fields(ContextID ctx, FieldMask &child_restricted);
      inline PhysicalState* get_physical_state(ContextID ctx, VersionInfo &info,
                                               bool initialize = true,
                                               bool capture = true)
      {
        // First check to see if the version info already has a state
        PhysicalState *result = info.find_physical_state(this);  
        if (result != NULL)
          return result;
        // If it didn't have it then we need to make it
        VersionManager *manager = version_managers.lookup_entry(ctx, this);
#ifdef DEBUG_HIGH_LEVEL
        assert(manager != NULL);
#endif
        // Now have the version info create a physical state with the manager
        result = info.create_physical_state(this, manager, initialize, capture);
#ifdef DEBUG_HIGH_LEVEL
        assert(result != NULL);
#endif
        return result;
      }
    public:
      void attach_semantic_information(SemanticTag tag, const NodeSet &mask,
                                       const void *buffer, size_t size);
      void retrieve_semantic_information(SemanticTag tag,
                                         const void *&result, size_t &size);
      virtual void send_semantic_info(const NodeSet &targets, SemanticTag tag,
                                      const void *buffer, size_t size,
                                      const NodeSet &current) = 0;
    public:
      // Logical traversal operations
      void register_logical_node(ContextID ctx,
                                 const LogicalUser &user,
                                 RegionTreePath &path,
                                 VersionInfo &version_info,
                                 RestrictInfo &restrict_info,
                                 const TraceInfo &trace_info,
                                 const bool projecting);
      void open_logical_node(ContextID ctx,
                             const LogicalUser &user,
                             RegionTreePath &path,
                             VersionInfo &version_info,
                             RestrictInfo &restrict_info,
                             const bool already_traced,
                             const bool projecting);
      void register_logical_fat_path(ContextID ctx,
                                     const LogicalUser &user,
                                     FatTreePath *fat_path,
                                     VersionInfo &version_info,
                                     RestrictInfo &restrict_info,
                                     const TraceInfo &trace_info);
      void open_logical_fat_path(ContextID ctx,
                                 const LogicalUser &user,
                                 FatTreePath *fat_path,
                                 VersionInfo &version_info,
                                 RestrictInfo &restrict_info);
      void close_reduction_analysis(ContextID ctx,
                                    const LogicalUser &user,
                                    VersionInfo &version_info);
      void close_logical_subtree(LogicalCloser &closer,
                                 const FieldMask &closing_mask);
      void close_logical_node(LogicalCloser &closer,
                              const FieldMask &closing_mask,
                              bool permit_leave_open);
      void siphon_logical_children(LogicalCloser &closer,
                                   LogicalState &state,
                                   const FieldMask &closing_mask,
                                   bool record_close_operations,
                                   const ColorPoint &next_child,
                                   FieldMask &open_below);
      void perform_close_operations(LogicalCloser &closer,
                                    const FieldMask &closing_mask,
                                    FieldState &closing_state,
                                    const ColorPoint &next_child, 
                                    bool allow_next_child,
                                    bool upgrade_next_child, 
                                    bool permit_leave_open,
                                    bool record_close_operations,
                                   LegionDeque<FieldState>::aligned &new_states,
                                    FieldMask &need_open);
      void merge_new_field_state(LogicalState &state, 
                                 const FieldState &new_state);
      void merge_new_field_states(LogicalState &state, 
                            const LegionDeque<FieldState>::aligned &new_states);
      void filter_prev_epoch_users(LogicalState &state, const FieldMask &mask);
      void filter_curr_epoch_users(LogicalState &state, const FieldMask &mask);
      void record_version_numbers(LogicalState &state, const FieldMask &mask,
                                  VersionInfo &info, bool capture_previous,
                                  bool premap_only, bool path_only);
      void advance_version_numbers(LogicalState &state, const FieldMask &mask);
      void record_logical_reduction(LogicalState &state, ReductionOpID redop,
                                    const FieldMask &user_mask);
      void clear_logical_reduction_fields(LogicalState &state,
                                          const FieldMask &cleared_mask);
      void sanity_check_logical_state(LogicalState &state);
      void initialize_logical_state(ContextID ctx);
      void invalidate_logical_state(ContextID ctx);
      template<bool DOMINATE>
      void register_logical_dependences(ContextID ctx, Operation *op,
                                        const FieldMask &field_mask);
      void add_restriction(ContextID ctx, const FieldMask &restricted_mask);
      void release_restriction(ContextID ctx, const FieldMask &restricted_mask);
      void record_logical_restrictions(ContextID ctx, RestrictInfo &info,
                                       const FieldMask &mask);
    public:
      // Physical traversal operations
      // Entry
      void close_physical_node(PhysicalCloser &closer,
                               const FieldMask &closing_mask);
      bool select_close_targets(PhysicalCloser &closer,
                                const FieldMask &closing_mask,
                 const LegionMap<LogicalView*,FieldMask>::aligned &valid_views,
                  LegionMap<MaterializedView*,FieldMask>::aligned &update_views,
                                bool &create_composite);
      bool siphon_physical_children(PhysicalCloser &closer,
                                    PhysicalState *state,
                                    const FieldMask &closing_mask,
                                    const std::set<ColorPoint> &next_children,
                                    bool &create_composite); 
      bool close_physical_child(PhysicalCloser &closer,
                                PhysicalState *state,
                                const FieldMask &closing_mask,
                                const ColorPoint &target_child,
                                FieldMask &child_mask,
                                const std::set<ColorPoint> &next_children,
                                bool &create_composite, bool &changed);
      // Analogous methods to those above except for closing to a composite view
      void create_composite_instance(ContextID ctx_id,
                                     const std::set<ColorPoint> &targets,
                                     bool leave_open, 
                                     const std::set<ColorPoint> &next_children,
                                     const FieldMask &closing_mask,
                                     VersionInfo &version_info); 
      void close_physical_node(CompositeCloser &closer,
                               CompositeNode *node,
                               const FieldMask &closing_mask,
                               FieldMask &dirty_mask,
                               FieldMask &complete_mask);
      void siphon_physical_children(CompositeCloser &closer,
                                    CompositeNode *node,
                                    PhysicalState *state,
                                    const FieldMask &closing_mask,
                                    FieldMask &dirty_mask,
                                    FieldMask &complete_mask);
      void close_physical_child(CompositeCloser &closer,
                                CompositeNode *node,
                                PhysicalState *state,
                                const FieldMask &closing_mask,
                                const ColorPoint &target_child,
                                const std::set<ColorPoint> &next_children,
                                FieldMask &dirty_mask,
                                FieldMask &complete_mask);
      void open_physical_child(ContextID ctx_id,
                               const ColorPoint &child_color,
                               const FieldMask &open_mask,
                               VersionInfo &version_info);
      // A special version of siphon physical region for closing
      // to a reduction instance at the end of a task
      void siphon_physical_children(ReductionCloser &closer,
                                    PhysicalState *state);
      void close_physical_node(ReductionCloser &closer);
      // This method will always add valid references to the set of views
      // that are returned.  It is up to the caller to remove the references.
      void find_valid_instance_views(ContextID ctx,
                                     PhysicalState *state,
                                     const FieldMask &valid_mask,
                                     const FieldMask &space_mask, 
                                     VersionInfo &version_info,
                                     bool needs_space,
                 LegionMap<LogicalView*,FieldMask>::aligned &valid_views);
      void find_valid_reduction_views(ContextID ctx, PhysicalState *state, 
                                      ReductionOpID redop,
                                      const FieldMask &valid_mask,
                                      VersionInfo &version_info,
                                      std::set<ReductionView*> &valid_views);
      void pull_valid_instance_views(ContextID ctx, PhysicalState *state,
                                     const FieldMask &mask, 
                                     VersionInfo &version_info);
      void find_copy_across_instances(const MappableInfo &info,
                                      MaterializedView *target,
                 LegionMap<MaterializedView*,FieldMask>::aligned &src_instances,
               LegionMap<DeferredView*,FieldMask>::aligned &deferred_instances);
      // Since figuring out how to issue copies is expensive, try not
      // to hold the physical state lock when doing them. NOTE IT IS UNSOUND
      // TO CALL THIS METHOD WITH A SET OF VALID INSTANCES ACQUIRED BY PASSING
      // 'TRUE' TO THE find_valid_instance_views METHOD!!!!!!!!
      void issue_update_copies(const MappableInfo &info,
                               MaterializedView *target, 
                               FieldMask copy_mask,
            const LegionMap<LogicalView*,FieldMask>::aligned &valid_instances,
                               CopyTracker *tracker = NULL);
      void sort_copy_instances(const MappableInfo &info,
                               MaterializedView *target,
                               FieldMask &copy_mask,
                     LegionMap<LogicalView*,FieldMask>::aligned &copy_instances,
                 LegionMap<MaterializedView*,FieldMask>::aligned &src_instances,
               LegionMap<DeferredView*,FieldMask>::aligned &deferred_instances);
      // Issue copies for fields with the same event preconditions
      static void issue_grouped_copies(RegionTreeForest *context,
                                       const MappableInfo &info,
                                       MaterializedView *dst,
                             LegionMap<Event,FieldMask>::aligned &preconditions,
                                       const FieldMask &update_mask,
                                       Event copy_domains_precondition,
                                       const std::set<Domain> &copy_domains,
           const LegionMap<MaterializedView*,FieldMask>::aligned &src_instances,
                                       const VersionInfo &src_version_info,
                           LegionMap<Event,FieldMask>::aligned &postconditions,
                                       CopyTracker *tracker = NULL);
      // Note this function can mutate the preconditions set
      static void compute_event_sets(FieldMask update_mask,
          const LegionMap<Event,FieldMask>::aligned &preconditions,
          LegionList<EventSet>::aligned &event_sets);
      Event perform_copy_operation(Operation *op, Event precondition,
                        const std::vector<Domain::CopySrcDstField> &src_fields,
                        const std::vector<Domain::CopySrcDstField> &dst_fields);
      void issue_update_reductions(LogicalView *target,
                                   const FieldMask &update_mask,
                                   const VersionInfo &version_info,
                                   Processor local_proc,
          const LegionMap<ReductionView*,FieldMask>::aligned &valid_reductions,
                                   Operation *op,
                                   CopyTracker *tracker = NULL);
      void invalidate_instance_views(PhysicalState *state,
                                     const FieldMask &invalid_mask); 
      void invalidate_reduction_views(PhysicalState *state,
                                      const FieldMask &invalid_mask);
      void update_valid_views(PhysicalState *state, const FieldMask &valid_mask,
                              bool dirty, LogicalView *new_view);
      void update_valid_views(PhysicalState *state, const FieldMask &valid_mask,
                              const FieldMask &dirty_mask, 
                              const std::vector<LogicalView*> &new_views);
      // I hate the container problem, somebody solve it please
      void update_valid_views(PhysicalState *state, const FieldMask &valid_mask,
                              const FieldMask &dirty,
                              const std::vector<MaterializedView*> &new_views);
      void update_reduction_views(PhysicalState *state, 
                                  const FieldMask &valid_mask,
                                  ReductionView *new_view);
      void flush_reductions(const FieldMask &flush_mask, ReductionOpID redop, 
                            const MappableInfo &info,CopyTracker *tracker=NULL);
      // Entry
      void initialize_physical_state(ContextID ctx);
      // Entry
      void invalidate_physical_state(ContextID ctx);
      // Entry
      void detach_instance_views(ContextID ctx, const FieldMask &detach_mask,
                                 PhysicalManager *target);
      void clear_physical_states(const FieldMask &mask);
    public:
      bool register_logical_view(LogicalView *view);
      void unregister_logical_view(LogicalView *view);
      LogicalView* find_view(DistributedID did);
      VersionState* find_remote_version_state(ContextID ctx, VersionID vid,
                                  DistributedID did, AddressSpaceID source, 
                                  bool initialize, bool request_eventual, 
                                  bool request_merged);
    public:
      bool register_physical_manager(PhysicalManager *manager);
      void unregister_physical_manager(PhysicalManager *manager);
      PhysicalManager* find_manager(DistributedID did);
    public:
      virtual unsigned get_depth(void) const = 0;
      virtual const ColorPoint& get_color(void) const = 0;
      virtual IndexTreeNode *get_row_source(void) const = 0;
      virtual RegionTreeID get_tree_id(void) const = 0;
      virtual RegionTreeNode* get_parent(void) const = 0;
      virtual RegionTreeNode* get_tree_child(const ColorPoint &c) = 0; 
      virtual void instantiate_children(void) = 0;
      virtual bool is_region(void) const = 0;
      virtual RegionNode* as_region_node(void) const = 0;
      virtual PartitionNode* as_partition_node(void) const = 0;
      virtual bool visit_node(PathTraverser *traverser) = 0;
      virtual bool visit_node(NodeTraverser *traverser) = 0;
    public:
      virtual bool are_children_disjoint(const ColorPoint &c1, 
                                         const ColorPoint &c2) = 0;
      virtual bool are_all_children_disjoint(void) = 0;
      virtual bool has_component_domains(void) const = 0;
      virtual const std::set<Domain>&
                        get_component_domains_blocking(void) const = 0;
      virtual const std::set<Domain>& 
                        get_component_domains(Event &ready) const = 0;
      virtual const Domain& get_domain_blocking(void) const = 0;
      virtual const Domain& get_domain(Event &precondition) const = 0;
      virtual const Domain& get_domain_no_wait(void) const = 0;
      virtual bool is_complete(void) = 0;
      virtual bool intersects_with(RegionTreeNode *other) = 0;
      virtual bool dominates(RegionTreeNode *other) = 0;
      virtual const std::set<Domain>& 
                      get_intersection_domains(RegionTreeNode *other) = 0;
    public:
      virtual size_t get_num_children(void) const = 0;
      virtual InterCloseOp* create_close_op(Operation *creator, 
                                            const FieldMask &closing_mask,
                                            bool leave_open,
                                            const std::set<ColorPoint> &targets,
                                            const VersionInfo &close_info,
                                            const VersionInfo &version_info,
                                            const RestrictInfo &res_info,
                                            const TraceInfo &trace_info) = 0;
      virtual bool perform_close_operation(const MappableInfo &info,
                                           const FieldMask &closing_mask,
                                           const std::set<ColorPoint> &targets,
                                           const MappingRef &target_region,
                                           VersionInfo &version_info,
                                           bool leave_open,
                                     const std::set<ColorPoint> &next_children,
                                           Event &closed,
                                           bool &create_composite) = 0;
      virtual MaterializedView * create_instance(Memory target_mem,
                                                const std::set<FieldID> &fields,
                                                size_t blocking_factor,
                                                unsigned depth, 
                                                Operation *op) = 0;
      virtual ReductionView* create_reduction(Memory target_mem,
                                              FieldID fid,
                                              bool reduction_list,
                                              ReductionOpID redop,
                                              Operation *op) = 0;
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
      // Logical helper operations
      template<AllocationType ALLOC, bool RECORD, bool HAS_SKIP, bool TRACK_DOM>
      static FieldMask perform_dependence_checks(const LogicalUser &user, 
          typename LegionList<LogicalUser, ALLOC>::track_aligned &users, 
          const FieldMask &check_mask, const FieldMask &open_below,
          bool validates_regions, Operation *to_skip = NULL, 
          GenerationID skip_gen = 0);
      template<AllocationType ALLOC>
      static void perform_closing_checks(LogicalCloser &closer,
          typename LegionList<LogicalUser, ALLOC>::track_aligned &users, 
          const FieldMask &check_mask);
    public:
      inline FieldSpaceNode* get_column_source(void) const 
      { return column_source; }
    public:
      RegionTreeForest *const context;
      FieldSpaceNode *const column_source;
    public:
      NodeSet creation_set;
      NodeSet destruction_set;
    protected:
      DynamicTable<LogicalStateAllocator> logical_states;
      DynamicTable<VersionManagerAllocator> version_managers;
    protected:
      Reservation node_lock;
      // While logical states and version managers have dense keys
      // within a node, distributed IDs don't so we use a map that
      // should rarely need to be accessed for tracking views
      LegionMap<DistributedID,LogicalView*,
                LOGICAL_VIEW_ALLOC>::tracked logical_views;
      LegionMap<DistributedID,PhysicalManager*,
                PHYSICAL_MANAGER_ALLOC>::tracked physical_managers;
    protected:
      LegionMap<SemanticTag,SemanticInfo>::aligned semantic_info;
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
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      bool has_child(const ColorPoint &p);
      bool has_color(const ColorPoint &p);
      PartitionNode* get_child(const ColorPoint &p);
      void add_child(PartitionNode *child);
      void remove_child(const ColorPoint &p);
      void add_creation_source(AddressSpaceID source);
      void destroy_node(AddressSpaceID source);
    public:
      virtual unsigned get_depth(void) const;
      virtual const ColorPoint& get_color(void) const;
      virtual IndexTreeNode *get_row_source(void) const;
      virtual RegionTreeID get_tree_id(void) const;
      virtual RegionTreeNode* get_parent(void) const;
      virtual RegionTreeNode* get_tree_child(const ColorPoint &c);
      virtual bool are_children_disjoint(const ColorPoint &c1, 
                                         const ColorPoint &c2);
      virtual bool are_all_children_disjoint(void);
      virtual void instantiate_children(void);
      virtual bool is_region(void) const;
      virtual RegionNode* as_region_node(void) const;
      virtual PartitionNode* as_partition_node(void) const;
      virtual bool visit_node(PathTraverser *traverser);
      virtual bool visit_node(NodeTraverser *traverser);
      virtual bool has_component_domains(void) const;
      virtual const std::set<Domain>& 
                                     get_component_domains_blocking(void) const;
      virtual const std::set<Domain>& get_component_domains(Event &ready) const;
      virtual const Domain& get_domain_blocking(void) const;
      virtual const Domain& get_domain(Event &precondition) const;
      virtual const Domain& get_domain_no_wait(void) const;
      virtual bool is_complete(void);
      virtual bool intersects_with(RegionTreeNode *other);
      virtual bool dominates(RegionTreeNode *other);
      virtual const std::set<Domain>& 
                                get_intersection_domains(RegionTreeNode *other);
      virtual size_t get_num_children(void) const;
      virtual InterCloseOp* create_close_op(Operation *creator, 
                                            const FieldMask &closing_mask,
                                            bool leave_open,
                                            const std::set<ColorPoint> &targets,
                                            const VersionInfo &close_info,
                                            const VersionInfo &version_info,
                                            const RestrictInfo &res_info,
                                            const TraceInfo &trace_info);
      virtual bool perform_close_operation(const MappableInfo &info,
                                           const FieldMask &closing_mask,
                                           const std::set<ColorPoint> &targets,
                                           const MappingRef &target_region,
                                           VersionInfo &version_info,
                                           bool leave_open,
                                     const std::set<ColorPoint> &next_children,
                                           Event &closed,
                                           bool &create_composite);
      virtual MaterializedView* create_instance(Memory target_mem,
                                                const std::set<FieldID> &fields,
                                                size_t blocking_factor,
                                                unsigned depth,
                                                Operation *op);
      virtual ReductionView* create_reduction(Memory target_mem,
                                              FieldID fid,
                                              bool reduction_list,
                                              ReductionOpID redop,
                                              Operation *op);
      virtual void send_node(AddressSpaceID target);
      static void handle_node_creation(RegionTreeForest *context,
                            Deserializer &derez, AddressSpaceID source);
    public:
      virtual void send_semantic_info(const NodeSet &targets, SemanticTag tag,
                                      const void *buffer, size_t size,
                                      const NodeSet &current);
      static void handle_semantic_info(RegionTreeForest *forest,
                                       Deserializer &derez);
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
                         LegionMap<ColorPoint,FieldMask>::aligned &to_traverse,
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
                        const FieldMask &user_mask, 
                        VersionInfo &version_info, FieldMask &needed_mask);
      InstanceRef register_region(const MappableInfo &info, Event term_event,
                                  const RegionUsage &usage, 
                                  const FieldMask &user_mask,
                                  LogicalView *view,
                                  const FieldMask &needed_fields);
      InstanceRef seed_state(ContextID ctx, Event term_event,
                             const RegionUsage &usage,
                             const FieldMask &user_mask,
                             LogicalView *new_view,
                             Processor local_proc);
      Event close_state(const MappableInfo &info, Event term_event,
                        RegionUsage &usage, const FieldMask &user_mask,
                        const InstanceRef &target);
      void find_field_descriptors(ContextID ctx, Event term_event,
                                  const RegionUsage &usage,
                                  const FieldMask &user_mask,
                                  unsigned fid_idx, Processor proc, 
                                  std::vector<FieldDataDescriptor> &field_data,
                                  std::set<Event> &preconditions,
                                  VersionInfo &version_info);
      void fill_fields(ContextID ctx, const FieldMask &fill_mask,
                       const void *value, size_t value_size, 
                       VersionInfo &version_info);
      InstanceRef attach_file(ContextID ctx, const FieldMask &attach_mask,
                             const RegionRequirement &req, AttachOp *attach_op,
                             VersionInfo &version_info);
      void detach_file(ContextID ctx, const FieldMask &detach_mask,
                       PhysicalManager *detach_target);
    public:
      const LogicalRegion handle;
      PartitionNode *const parent;
      IndexSpaceNode *const row_source;
    protected:
      std::map<ColorPoint,PartitionNode*> color_map;
      std::map<ColorPoint,PartitionNode*> valid_map;
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
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      bool has_child(const ColorPoint &c);
      bool has_color(const ColorPoint &c);
      RegionNode* get_child(const ColorPoint &c);
      void add_child(RegionNode *child);
      void remove_child(const ColorPoint &c);
      void add_creation_source(AddressSpaceID source);
      void destroy_node(AddressSpaceID source);
    public:
      virtual unsigned get_depth(void) const;
      virtual const ColorPoint& get_color(void) const;
      virtual IndexTreeNode *get_row_source(void) const;
      virtual RegionTreeID get_tree_id(void) const;
      virtual RegionTreeNode* get_parent(void) const;
      virtual RegionTreeNode* get_tree_child(const ColorPoint &c);
      virtual bool are_children_disjoint(const ColorPoint &c1, 
                                         const ColorPoint &c2);
      virtual bool are_all_children_disjoint(void);
      virtual void instantiate_children(void);
      virtual bool is_region(void) const;
      virtual RegionNode* as_region_node(void) const;
      virtual PartitionNode* as_partition_node(void) const;
      virtual bool visit_node(PathTraverser *traverser);
      virtual bool visit_node(NodeTraverser *traverser);
      virtual bool has_component_domains(void) const;
      virtual const std::set<Domain>& 
                                     get_component_domains_blocking(void) const;
      virtual const std::set<Domain>& get_component_domains(Event &ready) const;
      virtual const Domain& get_domain_blocking(void) const;
      virtual const Domain& get_domain(Event &precondition) const;
      virtual const Domain& get_domain_no_wait(void) const;
      virtual bool is_complete(void);
      virtual bool intersects_with(RegionTreeNode *other);
      virtual bool dominates(RegionTreeNode *other);
      virtual const std::set<Domain>& 
                                get_intersection_domains(RegionTreeNode *other);
      virtual size_t get_num_children(void) const;
      virtual InterCloseOp* create_close_op(Operation *creator, 
                                            const FieldMask &closing_mask,
                                            bool leave_open,
                                            const std::set<ColorPoint> &targets,
                                            const VersionInfo &close_info,
                                            const VersionInfo &version_info,
                                            const RestrictInfo &res_info,
                                            const TraceInfo &trace_info);
      virtual bool perform_close_operation(const MappableInfo &info,
                                           const FieldMask &closing_mask,
                                           const std::set<ColorPoint> &targets,
                                           const MappingRef &target_region,
                                           VersionInfo &version_info,
                                           bool leave_open,
                                     const std::set<ColorPoint> &next_children,
                                           Event &closed,
                                           bool &create_composite);
      virtual MaterializedView* create_instance(Memory target_mem,
                                                const std::set<FieldID> &fields,
                                                size_t blocking_factor,
                                                unsigned depth,
                                                Operation *op);
      virtual ReductionView* create_reduction(Memory target_mem,
                                              FieldID fid,
                                              bool reduction_list,
                                              ReductionOpID redop,
                                              Operation *op);
      virtual void send_node(AddressSpaceID target);
    public:
      virtual void send_semantic_info(const NodeSet &targets, SemanticTag tag,
                                      const void *buffer, size_t size,
                                      const NodeSet &current);
      static void handle_semantic_info(RegionTreeForest *forest,
                                       Deserializer &derez);
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
                         LegionMap<ColorPoint,FieldMask>::aligned &to_traverse,
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
      const LogicalPartition handle;
      RegionNode *const parent;
      IndexPartNode *const row_source;
    protected:
      std::map<ColorPoint,RegionNode*> color_map;
      std::map<ColorPoint,RegionNode*> valid_map;
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
      inline const std::map<ColorPoint,FatTreePath*> get_children(void) const
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
     * \class RestrictionMutator
     * A class for mutating the state of restrction fields
     */
    template<bool ADD_RESTRICT>
    class RestrictionMutator : public NodeTraverser {
    public:
      RestrictionMutator(ContextID ctx, const FieldMask &mask);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
      const FieldMask &restrict_mask;
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
    };

    class FieldInvalidator : public NodeTraverser {
    public:
      FieldInvalidator(unsigned idx) { to_clear.set_bit(idx); }
    public:
      virtual bool visit_only_valid(void) const { return false; }
      virtual bool visit_region(RegionNode *node) 
        { node->clear_physical_states(to_clear); return true; }
      virtual bool visit_partition(PartitionNode *node) 
        { node->clear_physical_states(to_clear); return true; }
    protected:
      FieldMask to_clear;
    };

    /**
     * \class PhysicalDetacher
     * A class for detaching physical instances normally associated
     * with files that have been attached.
     */
    class PhysicalDetacher : public NodeTraverser {
    public:
      PhysicalDetacher(ContextID ctx, const FieldMask &detach_mask,
                       PhysicalManager *to_detach);
    public:
      virtual bool visit_only_valid(void) const;
      virtual bool visit_region(RegionNode *node);
      virtual bool visit_partition(PartitionNode *node);
    protected:
      const ContextID ctx;
      const FieldMask &detach_mask;
      PhysicalManager *const target;
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
    class LayoutDescription : public Collectable {
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
      void* operator new(size_t count);
      void operator delete(void *ptr);
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
      void set_descriptor(FieldDataDescriptor &desc, unsigned fid_idx) const;
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
      Reservation layout_lock; 
      std::map<FIELD_TYPE,LegionVector<OffsetEntry>::aligned > 
                                                  memoized_offsets;
      NodeSet known_nodes;
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
                      Memory mem, RegionNode *node,
                      PhysicalInstance inst, bool register_now);
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
      virtual void notify_active(void);
      virtual void notify_inactive(void) = 0;
      virtual void notify_valid(void);
      virtual void notify_invalid(void) = 0;
      virtual DistributedID send_manager(AddressSpaceID target) = 0; 
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
      RegionNode *const region_node;
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
      enum InstanceFlag {
        NO_INSTANCE_FLAG = 0x00000000,
        PERSISTENT_FLAG  = 0x00000001,
        ATTACH_FILE_FLAG = 0x00000002,
      };
    public:
      class PersistenceFunctor {
      public:
        PersistenceFunctor(AddressSpaceID src, Runtime *rt, Serializer &z) 
          : source(src), runtime(rt), rez(z) { }
      public:
        void apply(AddressSpaceID target);
      protected:
        const AddressSpaceID source;
        Runtime *const runtime;
        Serializer &rez;
      };
    public:
      InstanceManager(RegionTreeForest *ctx, DistributedID did,
                      AddressSpaceID owner_space, AddressSpaceID local_space,
                      Memory mem, PhysicalInstance inst, RegionNode *node,
                      LayoutDescription *desc, Event use_event, 
                      unsigned depth, bool register_now,
                      InstanceFlag flag = NO_INSTANCE_FLAG);
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
      virtual void notify_inactive(void);
#ifdef DEBUG_HIGH_LEVEL
      virtual void notify_valid(void);
#endif
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
      void set_descriptor(FieldDataDescriptor &desc, unsigned fid_idx) const;
    public:
      virtual DistributedID send_manager(AddressSpaceID target); 
      static void handle_send_manager(Runtime *runtime, 
                                      AddressSpaceID source,
                                      Deserializer &derez);
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
      bool is_attached_file(void) const;
    public:
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
      InstanceFlag instance_flags;
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
                       const ReductionOp *op, bool register_now);
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
      virtual void notify_inactive(void);
      virtual void notify_invalid(void);
    public:
      virtual bool is_foldable(void) const = 0;
      virtual void find_field_offsets(const FieldMask &reduce_mask,
          std::vector<Domain::CopySrcDstField> &fields) = 0;
      virtual Event issue_reduction(Operation *op,
          const std::vector<Domain::CopySrcDstField> &src_fields,
          const std::vector<Domain::CopySrcDstField> &dst_fields,
          Domain space, Event precondition, bool reduction_fold,
          bool precise_domain) = 0;
      virtual Domain get_pointer_space(void) const = 0;
    public:
      virtual bool is_list_manager(void) const = 0;
      virtual ListReductionManager* as_list_manager(void) const = 0;
      virtual FoldReductionManager* as_fold_manager(void) const = 0;
      virtual Event get_use_event(void) const = 0;
    public:
      virtual DistributedID send_manager(AddressSpaceID target); 
    public:
      static void handle_send_manager(Runtime *runtime,
                                      AddressSpaceID source,
                                      Deserializer &derez);
    public:
      ReductionView* create_view(void);
    public:
      const ReductionOp *const op;
      const ReductionOpID redop;
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
                           const ReductionOp *op, Domain dom, bool reg_now);
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
      virtual Event issue_reduction(Operation *op,
          const std::vector<Domain::CopySrcDstField> &src_fields,
          const std::vector<Domain::CopySrcDstField> &dst_fields,
          Domain space, Event precondition, bool reduction_fold,
          bool precise_domain);
      virtual Domain get_pointer_space(void) const;
    public:
      virtual bool is_list_manager(void) const;
      virtual ListReductionManager* as_list_manager(void) const;
      virtual FoldReductionManager* as_fold_manager(void) const;
      virtual Event get_use_event(void) const;
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
                           const ReductionOp *op, Event use_event,
                           bool register_now);
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
      virtual Event issue_reduction(Operation *op,
          const std::vector<Domain::CopySrcDstField> &src_fields,
          const std::vector<Domain::CopySrcDstField> &dst_fields,
          Domain space, Event precondition, bool reduction_fold,
          bool precise_domain);
      virtual Domain get_pointer_space(void) const;
    public:
      virtual bool is_list_manager(void) const;
      virtual ListReductionManager* as_list_manager(void) const;
      virtual FoldReductionManager* as_fold_manager(void) const;
      virtual Event get_use_event(void) const;
    public:
      const Event use_event;
    };

    /**
     * \class Logicalview 
     * This class is the abstract base class for representing
     * the logical view onto one or more physical instances
     * in memory.  Logical views are reference counted
     * and will delete themselves once they no longer have
     * any valid handles.
     */
    class LogicalView : public DistributedCollectable {
    public:
      LogicalView(RegionTreeForest *ctx, DistributedID did,
                  AddressSpaceID owner_proc, AddressSpaceID local_space,
                  RegionTreeNode *node, bool register_now);
      virtual ~LogicalView(void);
    public:
      static void delete_logical_view(LogicalView *view);
    public:
      virtual void send_remote_registration(void);
      virtual void send_remote_valid_update(AddressSpaceID target, 
                                            unsigned count, bool add);
      virtual void send_remote_gc_update(AddressSpaceID target,
                                         unsigned count, bool add);
      virtual void send_remote_resource_update(AddressSpaceID target,
                                               unsigned count, bool add);
    public:
      virtual bool is_instance_view(void) const = 0;
      virtual bool is_deferred_view(void) const = 0;
      virtual InstanceView* as_instance_view(void) const = 0;
      virtual DeferredView* as_deferred_view(void) const = 0;
      virtual bool has_manager(void) const = 0;
      virtual PhysicalManager* get_manager(void) const = 0;
      virtual bool has_parent(void) const = 0;
      virtual LogicalView* get_parent(void) const = 0;
      virtual LogicalView* get_subview(const ColorPoint &c) = 0;
      virtual bool is_persistent(void) const = 0;
    public:
      static void handle_view_remote_registration(RegionTreeForest *forest,
                                                  Deserializer &derez,
                                                  AddressSpaceID source);
      static void handle_view_remote_valid_update(RegionTreeForest *forest,
                                                  Deserializer &derez);
      static void handle_view_remote_gc_update(RegionTreeForest *forest,
                                               Deserializer &derez);
      static void handle_view_remote_resource_update(RegionTreeForest *forest,
                                                     Deserializer &derez);
    public:
      virtual void notify_active(void) = 0;
      virtual void notify_inactive(void) = 0;
      virtual void notify_valid(void) = 0;
      virtual void notify_invalid(void) = 0;
    public:
      DistributedID send_view(AddressSpaceID target, 
                              const FieldMask &update_mask);
      virtual DistributedID send_view_base(AddressSpaceID target) = 0;
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask) = 0;
    public:
      void defer_collect_user(Event term_event);
      virtual void collect_users(const std::set<Event> &term_events) = 0;
      static void handle_deferred_collect(LogicalView *view,
                                          const std::set<Event> &term_events);
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
                   AddressSpaceID owner_proc, AddressSpaceID local_space,
                   RegionTreeNode *node, bool register_now);
      virtual ~InstanceView(void);
    public:
      virtual bool is_instance_view(void) const;
      virtual bool is_deferred_view(void) const;
      virtual InstanceView* as_instance_view(void) const;
      virtual DeferredView* as_deferred_view(void) const;
      virtual bool is_materialized_view(void) const = 0;
      virtual bool is_reduction_view(void) const = 0;
      virtual MaterializedView* as_materialized_view(void) const = 0;
      virtual ReductionView* as_reduction_view(void) const = 0;
      virtual bool has_manager(void) const = 0;
      virtual PhysicalManager* get_manager(void) const = 0;
      virtual bool has_parent(void) const = 0;
      virtual LogicalView* get_parent(void) const = 0;
      virtual LogicalView* get_subview(const ColorPoint &c) = 0;
      virtual Memory get_location(void) const = 0;
      virtual bool is_persistent(void) const = 0;
    public:
      // Entry point functions for doing physical dependence analysis
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           const FieldMask &copy_mask,
                                           const VersionInfo &version_info,
                     LegionMap<Event,FieldMask>::aligned &preconditions) = 0;
      virtual void add_copy_user(ReductionOpID redop, Event copy_term,
                                 const VersionInfo &version_info,
                                 const FieldMask &mask, bool reading) = 0;
      virtual InstanceRef add_user(const RegionUsage &user, Event term_event,
                                   const FieldMask &user_mask,
                                   const VersionInfo &version_info) = 0;
      virtual void add_initial_user(Event term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask) = 0;
    public:
      // Reference counting state change functions
      virtual void notify_active(void) = 0;
      virtual void notify_inactive(void) = 0;
      virtual void notify_valid(void) = 0;
      virtual void notify_invalid(void) = 0;
    public:
      virtual DistributedID send_view_base(AddressSpaceID target) = 0;
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask) = 0;
    public:
      // Instance recycling
      virtual void collect_users(const std::set<Event> &term_events) = 0;
    public:
      // Getting field information for performing copies
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &dst_fields) = 0;
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &src_fields) = 0;
      virtual bool reduce_to(ReductionOpID redop, 
                             const FieldMask &reduce_mask,
                     std::vector<Domain::CopySrcDstField> &src_fields) = 0;
      virtual void reduce_from(ReductionOpID redop,const FieldMask &reduce_mask,
                       std::vector<Domain::CopySrcDstField> &src_fields) = 0;
      virtual bool has_war_dependence(const RegionUsage &usage, 
                                      const FieldMask &user_mask) = 0;
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
      struct EventUsers {
      public:
        EventUsers(void)
          : single(true) { users.single_user = NULL; }
      public:
        FieldMask user_mask;
        union {
          PhysicalUser *single_user;
          LegionMap<PhysicalUser*,FieldMask>::aligned *multi_users;
        } users;
        bool single;
      };
    public:
      MaterializedView(RegionTreeForest *ctx, DistributedID did,
                       AddressSpaceID owner_proc, AddressSpaceID local_proc,
                       RegionTreeNode *node, InstanceManager *manager,
                       MaterializedView *parent, unsigned depth,
                       bool register_now);
      MaterializedView(const MaterializedView &rhs);
      virtual ~MaterializedView(void);
    public:
      MaterializedView& operator=(const MaterializedView &rhs);
    public:
      size_t get_blocking_factor(void) const;
      const FieldMask& get_physical_mask(void) const;
    public:
      virtual bool is_materialized_view(void) const;
      virtual bool is_reduction_view(void) const;
      virtual MaterializedView* as_materialized_view(void) const;
      virtual ReductionView* as_reduction_view(void) const;
    public:
      MaterializedView* get_materialized_subview(const ColorPoint &c);
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
      virtual void reduce_from(ReductionOpID redop,const FieldMask &reduce_mask,
                       std::vector<Domain::CopySrcDstField> &src_fields);
      virtual bool has_war_dependence(const RegionUsage &usage, 
                              const FieldMask &user_mask);
    public:
      void accumulate_events(std::set<Event> &all_events);
    public:
      virtual bool has_manager(void) const { return true; }
      virtual PhysicalManager* get_manager(void) const { return manager; }
      virtual bool has_parent(void) const { return (parent != NULL); }
      virtual LogicalView* get_parent(void) const { return parent; }
      virtual LogicalView* get_subview(const ColorPoint &c);
      virtual Memory get_location(void) const;
    public:
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           const FieldMask &copy_mask,
                                           const VersionInfo &version_info,
                         LegionMap<Event,FieldMask>::aligned &preconditions);
      virtual void add_copy_user(ReductionOpID redop, Event copy_term,
                                 const VersionInfo &version_info,
                                 const FieldMask &mask, bool reading);
      virtual InstanceRef add_user(const RegionUsage &user, Event term_event,
                                   const FieldMask &user_mask,
                                   const VersionInfo &version_info);
      virtual void add_initial_user(Event term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask);
    public:
      virtual void notify_active(void);
      virtual void notify_inactive(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
      virtual void collect_users(const std::set<Event> &term_users);
    public:
      virtual DistributedID send_view_base(AddressSpaceID target);
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask);
    protected:
      void add_user_above(const RegionUsage &usage, Event term_event,
                          const ColorPoint &child_color,
                          const VersionInfo &version_info,
                          const FieldMask &user_mask,
                          std::set<Event> &preconditions);
      bool add_local_user(const RegionUsage &usage, 
                          Event term_event, bool base_user,
                          const ColorPoint &child_color,
                          const VersionInfo &version_info,
                          const FieldMask &user_mask,
                          std::set<Event> &preconditions);
    protected:
      void add_copy_user_above(const RegionUsage &usage, Event copy_term,
                               const ColorPoint &child_color,
                               const VersionInfo &version_info,
                               const FieldMask &copy_mask);
      void add_local_copy_user(const RegionUsage &usage, 
                               Event copy_term, bool base_user,
                               const ColorPoint &child_color,
                               const VersionInfo &version_info,
                               const FieldMask &copy_mask);
      bool find_current_preconditions(Event test_event,
                                      const PhysicalUser *prev_user,
                                      const FieldMask &prev_mask,
                                      const RegionUsage &next_user,
                                      const FieldMask &next_mask,
                                      const ColorPoint &child_color,
                                      std::set<Event> &preconditions,
                                      FieldMask &observed,
                                      FieldMask &non_dominated);
      bool find_previous_preconditions(Event test_event,
                                       const PhysicalUser *prev_user,
                                       const FieldMask &prev_mask,
                                       const RegionUsage &next_user,
                                       const FieldMask &next_mask,
                                       const ColorPoint &child_color,
                                       std::set<Event> &preconditions);
    protected: 
      void find_copy_preconditions_above(ReductionOpID redop, bool reading,
                                         const FieldMask &copy_mask,
                                         const ColorPoint &child_color,
                                         const VersionInfo &version_info,
                       LegionMap<Event,FieldMask>::aligned &preconditions);
      void find_local_copy_preconditions(ReductionOpID redop, bool reading,
                                         const FieldMask &copy_mask,
                                         const ColorPoint &child_color,
                                         const VersionInfo &version_info,
                           LegionMap<Event,FieldMask>::aligned &preconditions);
      void find_current_copy_preconditions(Event test_event,
                                           const PhysicalUser *user, 
                                           const FieldMask &user_mask,
                                           ReductionOpID redop, bool reading,
                                           const FieldMask &copy_mask,
                                           const ColorPoint &child_color,
                                           const FieldVersions *versions,
                        LegionMap<Event,FieldMask>::aligned &preconditions,
                                           FieldMask &observed,
                                           FieldMask &non_dominated);
      void find_previous_copy_preconditions(Event test_event,
                                            const PhysicalUser *user,
                                            const FieldMask &user_Mask,
                                            ReductionOpID redop, bool reading,
                                            const FieldMask &copy_mask,
                                            const ColorPoint &child_dolor,
                                            const FieldVersions *versions,
                        LegionMap<Event,FieldMask>::aligned &preconditions);
    protected:
      void filter_previous_users(
                    const LegionMap<Event,FieldMask>::aligned &filter_previous);
      void filter_current_users(const FieldMask &dominated);
      void filter_local_users(Event term_event);
      void add_current_user(PhysicalUser *user, Event term_event,
                            const FieldMask &user_mask);
    protected:
      bool has_war_dependence_above(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    const ColorPoint &child_color);
      bool has_local_war_dependence(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    const ColorPoint &child_color,
                                    const ColorPoint &local_color);
    public:
      //void update_versions(const FieldMask &update_mask);
      void find_atomic_reservations(InstanceRef &target, const FieldMask &mask);
    public:
      void set_descriptor(FieldDataDescriptor &desc, unsigned fid_idx) const;
    public:
      virtual bool is_persistent(void) const;
      void make_persistent(void);
    public:
      void send_back_atomic_reservations(
          const std::vector<std::pair<FieldID,Reservation> > &send_back);
      void process_atomic_reservations(Deserializer &derez);
      static void handle_send_back_atomic(RegionTreeForest *ctx,
                                          Deserializer &derez);
    public:
      static void handle_send_materialized_view(Runtime *runtime,
                              Deserializer &derez, AddressSpaceID source);
    public:
      InstanceManager *const manager;
      MaterializedView *const parent;
      const unsigned depth;
    protected:
      // Keep track of the locks used for managing atomic coherence
      // on individual fields of this materialized view. Only the
      // top-level view for an instance needs to track this.
      std::map<FieldID,Reservation> atomic_reservations;
      // Keep track of the child views
      std::map<ColorPoint,MaterializedView*> children;
      // There are three operations that are done on materialized views
      // 1. iterate over all the users for use analysis
      // 2. garbage collection to remove old users for an event
      // 3. send updates for a certain set of fields
      // The first and last both iterate over the current and previous
      // user sets, while the second one needs to find specific events.
      // Therefore we store the current and previous sets as maps to
      // users indexed by events. Iterating over the maps are no worse
      // that iterating over lists (for arbitrary insertion and deletion)
      // and will provide fast indexing for removing items.
      LegionMap<Event,EventUsers>::aligned current_epoch_users;
      LegionMap<Event,EventUsers>::aligned previous_epoch_users;
      // Also keep a set of events for which we have outstanding
      // garbage collection meta-tasks so we don't launch more than one
      // We need this even though we have the data structures above because
      // an event might be filtered out for some fields, so we can't rely
      // on it to detect when we have outstanding gc meta-tasks
      std::set<Event> outstanding_gc_events;
      // TODO: Keep track of the current version numbers for each field
      // This will allow us to detect when physical instances are no
      // longer valid from a particular view when doing rollbacks for
      // resilience or mis-speculation.
      //LegionMap<VersionID,FieldMask,
      //          PHYSICAL_VERSION_ALLOC>::track_aligned current_versions;
    protected:
      // Useful for pruning the initial users at cleanup time
      std::set<Event> initial_user_events;
    };

    /**
     * \class ReductionView
     * The ReductionView class is used for providing a view
     * onto reduction physical instances from any logical perspective.
     */
    class ReductionView : public InstanceView {
    public:
      static const AllocationType alloc_type = REDUCTION_VIEW_ALLOC;
    public:
      struct EventUsers {
      public:
        EventUsers(void)
          : single(true) { users.single_user = NULL; }
      public:
        FieldMask user_mask;
        union {
          PhysicalUser *single_user;
          LegionMap<PhysicalUser*,FieldMask>::aligned *multi_users;
        } users;
        bool single;
      };
    public:
      ReductionView(RegionTreeForest *ctx, DistributedID did,
                    AddressSpaceID owner_proc, AddressSpaceID local_proc,
                    RegionTreeNode *node, ReductionManager *manager,
                    bool register_now);
      ReductionView(const ReductionView &rhs);
      virtual ~ReductionView(void);
    public:
      ReductionView& operator=(const ReductionView&rhs);
    public:
      void perform_reduction(InstanceView *target, const FieldMask &copy_mask, 
                             const VersionInfo &version_info,
                             Processor local_proc, Operation *op,
                             CopyTracker *tracker = NULL);
      Event perform_deferred_reduction(MaterializedView *target,
                                        const FieldMask &copy_mask,
                                        const VersionInfo &version_info,
                                        const std::set<Event> &preconditions,
                                        const std::set<Domain> &reduce_domains,
                                        Event domain_precondition,
                                        Operation *op);
      Event perform_deferred_across_reduction(MaterializedView *target,
                                              FieldID dst_field,
                                              FieldID src_field,
                                              unsigned src_index,
                                       const VersionInfo &version_info,
                                       const std::set<Event> &preconditions,
                                       const std::set<Domain> &reduce_domains,
                                       Event domain_precondition,
                                       Operation *op);
    public:
      virtual bool is_materialized_view(void) const;
      virtual bool is_reduction_view(void) const;
      virtual MaterializedView* as_materialized_view(void) const;
      virtual ReductionView* as_reduction_view(void) const;
      virtual bool has_manager(void) const { return true; } 
      virtual PhysicalManager* get_manager(void) const;
      virtual bool has_parent(void) const { return false; }
      virtual LogicalView* get_parent(void) const 
        { assert(false); return NULL; } 
      virtual LogicalView* get_subview(const ColorPoint &c);
      virtual Memory get_location(void) const;
      virtual bool is_persistent(void) const;
    public:
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           const FieldMask &copy_mask,
                                           const VersionInfo &version_info,
                         LegionMap<Event,FieldMask>::aligned &preconditions);
      virtual void add_copy_user(ReductionOpID redop, Event copy_term,
                                 const VersionInfo &version_info,
                                 const FieldMask &mask, bool reading);
      virtual InstanceRef add_user(const RegionUsage &user, Event term_event,
                                   const FieldMask &user_mask,
                                   const VersionInfo &version_info);
      virtual void add_initial_user(Event term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask);
    public:
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask,
                     std::vector<Domain::CopySrcDstField> &dst_fields);
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &dst_fields);
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &src_fields);
      virtual bool has_war_dependence(const RegionUsage &usage, 
                                      const FieldMask &user_mask);
    public:
      void reduce_from(ReductionOpID redop, const FieldMask &reduce_mask,
                       std::vector<Domain::CopySrcDstField> &src_fields);
    public:
      virtual void notify_active(void);
      virtual void notify_inactive(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
      virtual void collect_users(const std::set<Event> &term_events);
    public:
      virtual DistributedID send_view_base(AddressSpaceID target);
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask);
    protected:
      void add_physical_user(PhysicalUser *user, bool reading,
                             Event term_event, const FieldMask &user_mask);
      void filter_local_users(Event term_event);
    public:
      static void handle_send_reduction_view(Runtime *runtime,
                              Deserializer &derez, AddressSpaceID source);
    public:
      ReductionOpID get_redop(void) const;
    public:
      ReductionManager *const manager;
    protected:
      LegionMap<Event,EventUsers>::aligned reduction_users;
      LegionMap<Event,EventUsers>::aligned reading_users;
      std::set<Event> outstanding_gc_events;
    protected:
      std::set<Event> initial_user_events;
    };

    /**
     * \class DeferredView
     * A DeferredView class is an abstract class the complements
     * the MaterializedView class. While materialized views are 
     * actual views onto a real instance, deferred views are 
     * effectively place holders for non-physical isntances which
     * contain enough information to perform the necessary 
     * operations to bring a materialized view up to date for 
     * specific fields. There are several different flavors of
     * deferred views and this class is the base type.
     */
    class DeferredView : public LogicalView {
    public:
      struct ReductionEpoch {
      public:
        ReductionEpoch(void)
          : redop(0) { }
        ReductionEpoch(ReductionView *v, ReductionOpID r, const FieldMask &m)
          : valid_fields(m), redop(r) { views.insert(v); }
      public:
        FieldMask valid_fields;
        ReductionOpID redop;
        std::set<ReductionView*> views;
      };
    public:
      DeferredView(RegionTreeForest *ctx, DistributedID did,
                   AddressSpaceID owner_space, AddressSpaceID local_space,
                   RegionTreeNode *node, bool register_now);
      virtual ~DeferredView(void);
    public:
      // Deferred views never have managers
      virtual bool has_manager(void) const { return false; }
      virtual PhysicalManager* get_manager(void) const
      { return NULL; }
      virtual bool has_parent(void) const = 0;
      virtual LogicalView* get_parent(void) const = 0;
      virtual LogicalView* get_subview(const ColorPoint &c) = 0;
      // Deferred views are never persistent
      virtual bool is_persistent(void) const { return false; }
    public:
    public:
      virtual void notify_active(void) = 0;
      virtual void notify_inactive(void) = 0;
      virtual void notify_valid(void) = 0;
      virtual void notify_invalid(void) = 0;
    public:
      virtual DistributedID send_view_base(AddressSpaceID target) = 0;
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask) = 0;
    public:
      // Should never be called
      virtual void collect_users(const std::set<Event> &term_events)
        { assert(false); }
    public:
      virtual bool is_instance_view(void) const { return false; }
      virtual bool is_deferred_view(void) const { return true; }
      virtual InstanceView* as_instance_view(void) const { return NULL; }
      virtual DeferredView* as_deferred_view(void) const
        { return const_cast<DeferredView*>(this); }
    public:
      virtual bool is_composite_view(void) const = 0;
      virtual bool is_fill_view(void) const = 0;
      virtual FillView* as_fill_view(void) const = 0;
      virtual CompositeView* as_composite_view(void) const = 0;
    public:
      void update_reduction_views(ReductionView *view, 
                                  const FieldMask &valid_mask,
                                  bool update_parent = true);
      void update_reduction_epochs(const ReductionEpoch &epoch);
    protected:
      void update_reduction_views_above(ReductionView *view,
                                        const FieldMask &valid_mask,
                                        DeferredView *from_child);
      void update_local_reduction_views(ReductionView *view,
                                        const FieldMask &valid_mask);
      virtual void update_child_reduction_views(ReductionView *view,
                                                const FieldMask &valid_mask,
                                                DeferredView *skip = NULL) = 0;
      void flush_reductions(const MappableInfo &info,
                            MaterializedView *dst,
                            const FieldMask &reduce_mask,
                            LegionMap<Event,FieldMask>::aligned &conditions);
      void flush_reductions_across(const MappableInfo &info,
                                   MaterializedView *dst,
                                   FieldID src_field, FieldID dst_field,
                                   Event dst_precondition,
                                   std::set<Event> &conditions);
      Event find_component_domains(ReductionView *reduction_view,
                                   MaterializedView *dst_view,
                                   std::set<Domain> &component_domains);
    protected:
      void activate_deferred(void);
      void deactivate_deferred(void);
      void validate_deferred(void);
      void invalidate_deferred(void);
    public:
      virtual void issue_deferred_copies(const MappableInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                                         CopyTracker *tracker = NULL) = 0;
      virtual void issue_deferred_copies(const MappableInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
             const LegionMap<Event,FieldMask>::aligned &preconditions,
                   LegionMap<Event,FieldMask>::aligned &postconditions,
                                         CopyTracker *tracker = NULL) = 0;
    public:
      virtual void issue_deferred_copies_across(const MappableInfo &info,
                                                MaterializedView *dst,
                                                FieldID src_field,
                                                FieldID dst_field,
                                                Event precondition,
                                          std::set<Event> &postconditions) = 0;
    public:
      virtual void find_field_descriptors(Event term_event,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx,
                                          Processor local_proc,
                                  std::vector<FieldDataDescriptor> &field_data,
                                          std::set<Event> &preconditions) = 0;
      virtual bool find_field_descriptors(Event term_event,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx,
                                          Processor local_proc,
                                          LowLevel::IndexSpace target,
                                          Event target_precondition,
                                  std::vector<FieldDataDescriptor> &field_data,
                                  std::set<Event> &preconditions,
                             std::vector<LowLevel::IndexSpace> &already_handled,
                             std::set<Event> &already_preconditions) = 0;
    protected:
      // Track the set of reduction views which need to be applied here
      FieldMask reduction_mask;
      // We need to keep these in order because there may be multiple
      // generations of reductions applied to this instance
      LegionDeque<ReductionEpoch>::aligned reduction_epochs;
    };

    /**
     * \class CompositeView
     * The CompositeView class is used for deferring close
     * operations by representing a valid version of a single
     * logical region with a bunch of different instances.
     */
    class CompositeView : public DeferredView {
    public:
      static const AllocationType alloc_type = COMPOSITE_VIEW_ALLOC; 
    public:
      CompositeView(RegionTreeForest *ctx, DistributedID did,
                    AddressSpaceID owner_proc, RegionTreeNode *node, 
                    AddressSpaceID local_proc, const FieldMask &mask,
                    bool register_now, CompositeView *parent = NULL);
      CompositeView(const CompositeView &rhs);
      virtual ~CompositeView(void);
    public:
      CompositeView& operator=(const CompositeView &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      virtual bool has_parent(void) const { return (parent != NULL); }
      virtual LogicalView* get_parent(void) const { return parent; }
      virtual LogicalView* get_subview(const ColorPoint &c);
    public:
      virtual void notify_active(void);
      virtual void notify_inactive(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
    public:
      virtual DistributedID send_view_base(AddressSpaceID target);
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask);
      void unpack_composite_view(Deserializer &derez, AddressSpaceID source);
      void make_local(std::set<Event> &preconditions);
    public:
      virtual bool is_composite_view(void) const { return true; }
      virtual bool is_fill_view(void) const { return false; }
      virtual FillView* as_fill_view(void) const
        { return NULL; }
      virtual CompositeView* as_composite_view(void) const
        { return const_cast<CompositeView*>(this); }
    public:
      void update_valid_mask(const FieldMask &mask);
      void flatten_composite_view(FieldMask &global_dirt,
              const FieldMask &flatten_mask, CompositeCloser &closer,
              CompositeNode *target);
    public:
      virtual void find_field_descriptors(Event term_event,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx,
                                          Processor local_proc,
                                  std::vector<FieldDataDescriptor> &field_data,
                                          std::set<Event> &preconditions);
      virtual bool find_field_descriptors(Event term_event,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx,
                                          Processor local_proc,
                                          LowLevel::IndexSpace target,
                                          Event target_precondition,
                                  std::vector<FieldDataDescriptor> &field_data,
                                          std::set<Event> &preconditions,
                             std::vector<LowLevel::IndexSpace> &already_handled,
                                       std::set<Event> &already_preconditions);
    public:
      void add_root(CompositeNode *root, const FieldMask &valid, bool top);
      virtual void update_child_reduction_views(ReductionView *view,
                                                const FieldMask &valid_mask,
                                                DeferredView *skip = NULL);
    public:
      virtual void issue_deferred_copies(const MappableInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                                         CopyTracker *tracker = NULL);
      virtual void issue_deferred_copies(const MappableInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
             const LegionMap<Event,FieldMask>::aligned &preconditions,
                   LegionMap<Event,FieldMask>::aligned &postconditions,
                                         CopyTracker *tracker = NULL);
    public:
      // Note that copy-across only works for a single field at a time
      virtual void issue_deferred_copies_across(const MappableInfo &info,
                                                MaterializedView *dst,
                                                FieldID src_field,
                                                FieldID dst_field,
                                                Event precondition,
                                         std::set<Event> &postconditions);
    public:
      static void handle_send_composite_view(Runtime *runtime, 
                              Deserializer &derez, AddressSpaceID source);
    public:
      CompositeView *const parent;
    protected:
      // The set of fields represented by this composite view
      FieldMask valid_mask;
      // Keep track of the roots and their field masks
      // There is exactly one root for every field
      LegionMap<CompositeNode*,FieldMask>::aligned roots;
      // Keep track of all the child views
      std::map<ColorPoint,CompositeView*> children;
    };

    /**
     * \class CompositeVersionInfo
     * This is a wrapper class for keeping track of the version
     * information for all the composite nodes in a composite instance.
     */
    class CompositeVersionInfo : public Collectable {
    public:
      CompositeVersionInfo(void);
      CompositeVersionInfo(const CompositeVersionInfo &rhs);
      ~CompositeVersionInfo(void);
    public:
      CompositeVersionInfo& operator=(const CompositeVersionInfo &rhs);
    public:
      inline VersionInfo& get_version_info(void)
        { return version_info; }
    protected:
      VersionInfo version_info;
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
      CompositeNode(RegionTreeNode *logical, CompositeNode *parent,
                    CompositeVersionInfo *version_info);
      CompositeNode(const CompositeNode &rhs);
      ~CompositeNode(void);
    public:
      CompositeNode& operator=(const CompositeNode &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      void capture_physical_state(RegionTreeNode *tree_node,
                                  PhysicalState *state,
                                  const FieldMask &capture_mask,
                                  CompositeCloser &closer,
                                  FieldMask &global_dirty,
                                  const FieldMask &other_dirty_mask,
                const LegionMap<LogicalView*,FieldMask,
                        VALID_VIEW_ALLOC>::track_aligned &other_valid_views);
      CompositeNode* flatten(const FieldMask &flatten_mask, 
                             CompositeCloser &closer,
                             CompositeNode *parent,
                             FieldMask &global_dirt,
                             CompositeNode *target);
      CompositeNode* create_clone_node(CompositeNode *parent,
                                       CompositeCloser &closer);
      void update_parent_info(const FieldMask &mask);
      void update_child_info(CompositeNode *child, const FieldMask &mask);
      void update_instance_views(LogicalView *view,
                                 const FieldMask &valid_mask);
    public:
      void issue_update_copies(const MappableInfo &info,
                               MaterializedView *dst,
                               FieldMask traversal_mask,
                               const FieldMask &copy_mask,
                       const LegionMap<Event,FieldMask>::aligned &preconditions,
                           LegionMap<Event,FieldMask>::aligned &postconditions,
                               CopyTracker *tracker = NULL);
      void issue_across_copies(const MappableInfo &info,
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
      void set_owner_did(DistributedID owner_did);
    public:
      bool find_field_descriptors(Event term_event, const RegionUsage &usage,
                                  const FieldMask &user_mask,
                                  unsigned fid_idx, Processor local_proc, 
                                  LowLevel::IndexSpace target, Event target_pre,
                                  std::vector<FieldDataDescriptor> &field_data,
                                  std::set<Event> &preconditions,
                             std::vector<LowLevel::IndexSpace> &already_handled,
                                  std::set<Event> &already_preconditions);
    public:
      void add_gc_references(void);
      void remove_gc_references(void);
      void add_valid_references(void);
      void remove_valid_references(void);
    public:
      void pack_composite_tree(Serializer &rez, AddressSpaceID target);
      void unpack_composite_tree(Deserializer &derez, AddressSpaceID source);
      void make_local(std::set<Event> &preconditions,
                      std::set<DistributedID> &checked_views);
    protected:
      bool dominates(RegionTreeNode *dst);
      template<typename MAP_TYPE>
      void capture_instances(const FieldMask &capture_mask, 
                             FieldMask &need_flatten,
          LegionMap<CompositeView*,FieldMask>::aligned &to_flatten,
          const MAP_TYPE &instances);
    public:
      RegionTreeForest *const context;
      RegionTreeNode *const logical_node;
      CompositeNode *const parent;
      CompositeVersionInfo *const version_info;
    protected:
      DistributedID owner_did;
      FieldMask dirty_mask;
      LegionMap<CompositeNode*,ChildInfo>::aligned open_children;
      LegionMap<LogicalView*,FieldMask,
                VALID_VIEW_ALLOC>::track_aligned valid_views;
    };

    /**
     * \class FillView
     * This is a deferred view that is used for filling in 
     * fields with a default value.
     */
    class FillView : public DeferredView {
    public:
      static const AllocationType alloc_type = FILL_VIEW_ALLOC;
    public:
      class FillViewValue : public Collectable {
      public:
        FillViewValue(const void *v, size_t size)
          : value(v), value_size(size) { }
        FillViewValue(const FillViewValue &rhs)
          : value(NULL), value_size(0) { assert(false); }
        ~FillViewValue(void)
        { free(const_cast<void*>(value)); }
      public:
        FillViewValue& operator=(const FillViewValue &rhs)
        { assert(false); return *this; }
      public:
        const void *const value;
        const size_t value_size;
      };
    public:
      FillView(RegionTreeForest *ctx, DistributedID did,
               AddressSpaceID owner_proc, AddressSpaceID local_proc,
               RegionTreeNode *node, bool reg_now, FillViewValue *value,
               FillView *parent = NULL);
      FillView(const FillView &rhs);
      virtual ~FillView(void);
    public:
      FillView& operator=(const FillView &rhs);
    public:
      virtual bool has_parent(void) const { return (parent != NULL); }
      virtual LogicalView* get_parent(void) const { return parent; }
      virtual LogicalView* get_subview(const ColorPoint &c);
    public:
      virtual void notify_active(void);
      virtual void notify_inactive(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
    public:
      virtual DistributedID send_view_base(AddressSpaceID target);
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask);
    public:
      virtual bool is_composite_view(void) const { return false; }
      virtual bool is_fill_view(void) const { return true; }
      virtual FillView* as_fill_view(void) const
        { return const_cast<FillView*>(this); }
      virtual CompositeView* as_composite_view(void) const { return NULL; }
    public:
      virtual void update_child_reduction_views(ReductionView *view,
                                                const FieldMask &valid_mask,
                                                DeferredView *skip = NULL);
    public:
      virtual void issue_deferred_copies(const MappableInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                                         CopyTracker *tracker = NULL);
      virtual void issue_deferred_copies(const MappableInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
             const LegionMap<Event,FieldMask>::aligned &preconditions,
                   LegionMap<Event,FieldMask>::aligned &postconditions,
                                         CopyTracker *tracker = NULL);
    public:
      virtual void issue_deferred_copies_across(const MappableInfo &info,
                                                MaterializedView *dst,
                                                FieldID src_field,
                                                FieldID dst_field,
                                                Event precondition,
                                          std::set<Event> &postconditions);
    public:
      virtual void find_field_descriptors(Event term_event, 
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx,
                                          Processor local_proc,
                                  std::vector<FieldDataDescriptor> &field_data,
                                          std::set<Event> &preconditions);
      virtual bool find_field_descriptors(Event term_event,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx,
                                          Processor local_proc,
                                          LowLevel::IndexSpace target,
                                          Event target_precondition,
                                  std::vector<FieldDataDescriptor> &field_data,
                                          std::set<Event> &preconditions,
                             std::vector<LowLevel::IndexSpace> &already_handled,
                                       std::set<Event> &already_preconditions);
    public:
      static void handle_send_fill_view(Runtime *runtime, Deserializer &derez,
                                        AddressSpaceID source);
    public:
      FillView *const parent;
      FillViewValue *const value;
    protected:
      // Keep track of the child views
      std::map<ColorPoint,FillView*> children;
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
        if (!view->is_instance_view())
          return false;
        return view->as_instance_view()->is_reduction_view();
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
      InstanceRef(Event ready, const ViewHandle &handle);
      InstanceRef(Event ready, const ViewHandle &handle,
                  const std::vector<Reservation> &locks);
    public:
      bool operator==(const InstanceRef &rhs) const;
      bool operator!=(const InstanceRef &rhs) const;
    public:
      inline bool has_ref(void) const { return handle.has_view(); }
      inline bool has_required_locks(void) const 
                                      { return !needed_locks.empty(); }
      inline Event get_ready_event(void) const { return ready_event; }
      const ViewHandle& get_handle(void) const { return handle; }
      inline void add_reservation(Reservation handle) 
                                  { needed_locks.push_back(handle); }
      void update_atomic_locks(std::map<Reservation,bool> &atomic_locks,
                               bool exclusive) const;
      Memory get_memory(void) const;
      Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_accessor(void) const;
      Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_field_accessor(FieldID fid) const;
      void pack_reference(Serializer &rez, AddressSpaceID target);
      static InstanceRef unpack_reference(Deserializer &derez,
                                          RegionTreeForest *context,
                                          unsigned depth);
    private:
      Event ready_event;
      ViewHandle handle;
      std::vector<Reservation> needed_locks;
    };

    /**
     * \class MappingTraverser
     * A traverser of the physical region tree for
     * performing the mapping operation.
     */
    template<bool RESTRICTED>
    class MappingTraverser : public PathTraverser {
    public:
      MappingTraverser(RegionTreePath &path, const MappableInfo &info,
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
      bool map_restricted_physical(RegionNode *node);
      bool map_restricted_reduction(RegionNode *node);
    public:
      const MappableInfo &info;
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

