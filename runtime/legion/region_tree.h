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


#ifndef __LEGION_REGION_TREE_H__
#define __LEGION_REGION_TREE_H__

#include "legion_types.h"
#include "legion_utilities.h"
#include "legion_allocation.h"
#include "legion_analysis.h"
#include "garbage_collection.h"
#include "field_tree.h"

namespace Legion {
  namespace Internal {
    
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
                         Realm::IndexSpace::IndexSpaceOperation op);
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
                          FieldID fid, bool local, CustomSerdezID serdez_id);
      void free_field(FieldSpace handle, FieldID fid, AddressSpaceID source);
      void allocate_fields(FieldSpace handle, const std::vector<size_t> &sizes,
                           const std::vector<FieldID> &resulting_fields,
                           CustomSerdezID serdez_id);
      void free_fields(FieldSpace handle, const std::set<FieldID> &to_free,
                       AddressSpaceID source);
      void allocate_field_index(FieldSpace handle, size_t field_size, 
                                FieldID fid, unsigned index, 
                                CustomSerdezID serdez_id,
                                AddressSpaceID source);
      void allocate_field_indexes(FieldSpace handle, 
                                  const std::vector<FieldID> &resulting_fields,
                                  const std::vector<size_t> &sizes,
                                  const std::vector<unsigned> &indexes,
                                  CustomSerdezID serdez_id,
                                  AddressSpaceID source);
      void get_all_fields(FieldSpace handle, std::set<FieldID> &fields);
      void get_all_regions(FieldSpace handle, std::set<LogicalRegion> &regions);
      size_t get_field_size(FieldSpace handle, FieldID fid);
      void get_field_space_fields(FieldSpace handle, std::set<FieldID> &fields);
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
      void restrict_user_coherence(RegionTreeContext ctx,
                                   SingleTask *parent_ctx,
                                   LogicalRegion handle,
                                   const std::set<FieldID> &fields);
      void acquire_user_coherence(RegionTreeContext ctx,
                                  LogicalRegion handle,
                                  const std::set<FieldID> &fields);
      bool has_restrictions(LogicalRegion handle, const RestrictInfo &info,
                            const std::set<FieldID> &fields);
    public:
      
      void initialize_current_context(RegionTreeContext ctx,
                                      const RegionRequirement &req,
                                      CompositeView *composite_view);
      void invalidate_current_context(RegionTreeContext ctx,
                                      LogicalRegion handle,
                                      bool logical_users_only);
    public: // Physical analysis methods
#if 1
      void initialize_current_context(RegionTreeContext ctx,
                    const RegionRequirement &req, const InstanceSet &source,
                    Event term_event, unsigned depth,
                    std::map<PhysicalManager*,InstanceView*> &top_views,
                    InstanceSet &target);
      void physical_traverse_path(RegionTreeContext ctx,
                                  RegionTreePath &path,
                                  const RegionRequirement &req,
                                  VersionInfo &version_info,
                                  Operation *op, bool find_valid,
                                  InstanceSet &valid_insts
#ifdef DEBUG_HIGH_LEVEL
                                  , unsigned index
                                  , const char *log_name
                                  , UniqueID uid
#endif
                                  );
      void traverse_and_register(RegionTreeContext ctx,
                                 RegionTreePath &path,
                                 const RegionRequirement &req,
                                 VersionInfo &version_info,
                                 Operation *op, Event term_event, 
                                 const InstanceSet &targets
#ifdef DEBUG_HIGH_LEVEL
                                 , unsigned index
                                 , const char *log_name
                                 , UniqueID uid
#endif
                                 );
      void physical_register_only(RegionTreeContext ctx,
                                  const RegionRequirement &req,
                                  VersionInfo &version_info,
                                  Operation *op, Event term_event,
                                  const InstanceSet &targets
#ifdef DEBUG_HIGH_LEVEL
                                 , unsigned index
                                 , const char *log_name
                                 , UniqueID uid
#endif
                                 );
      Event physical_perform_close(RegionTreeContext ctx,
                                   const RegionRequirement &req,
                                   VersionInfo &version_info,
                                   Operation *op, int composite_index,
                    const LegionMap<ColorPoint,FieldMask>::aligned &to_close,
                    const std::set<ColorPoint> &next_children,
                    const InstanceSet &targets
#ifdef DEBUG_HIGH_LEVEL
                                  , unsigned index
                                  , const char *log_name
                                  , UniqueID uid
#endif
                                  );
      Event physical_close_context(RegionTreeContext ctx,
                                   const RegionRequirement &req,
                                   VersionInfo &version_info,
                                   Operation *op,
                                   const InstanceSet &targets
#ifdef DEBUG_HIGH_LEVEL
                                   , unsigned index
                                   , const char *log_name
                                   , UniqueID uid
#endif
                                   );
      CompositeRef virtual_close_context(RegionTreeContext ctx,
                                         const RegionRequirement &req,
                                         VersionInfo &version_info
#ifdef DEBUG_HIGH_LEVEL
                                         , unsigned index
                                         , const char *log_name
                                         , UniqueID uid
#endif
                                         );
      void register_virtual_region(RegionTreeContext ctx,
                                   CompositeView *composite_view,
                                   RegionRequirement &req,
                                   VersionInfo &version_info);
      Event copy_across(RegionTreeContext src_ctx,
                        RegionTreeContext dst_ctx,
                        const RegionRequirement &src_req,
                        const RegionRequirement &dst_req,
                        const InstanceSet &src_targets, 
                        const InstanceSet &dst_targets,
                        VersionInfo &src_version_info, int src_composite,
                        Operation *op, Event precondition);
      Event reduce_across(RegionTreeContext src_ctx,
                          RegionTreeContext dst_ctx,
                          const RegionRequirement &src_req,
                          const RegionRequirement &dst_req,
                          const InstanceSet &src_targets,
                          const InstanceSet &dst_targets,
                          VersionInfo &src_version_info, int src_composite,
                          Operation *op, Event precondition);
      int physical_convert_mapping(const RegionRequirement &req,
                               const std::vector<MappingInstance> &chosen,
                               const InstanceSet &valid, InstanceSet &result,
                               std::vector<FieldID> &missing_fields);
      bool physical_convert_postmapping(const RegionRequirement &req,
                               const std::vector<MappingInstance> &chosen,
                               const InstanceSet &valid, InstanceSet &result);
      bool is_valid_mapping(const InstanceRef &ref, 
                            const RegionRequirement &req);
#else
      InstanceRef initialize_current_context(RegionTreeContext ctx,
                    const RegionRequirement &req, PhysicalManager *manager,
                    Event term_event, unsigned depth,
                    std::map<PhysicalManager*,InstanceView*> &top_views);
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
                                       Processor target_proc,
                                       const InstanceRef &result
#ifdef DEBUG_HIGH_LEVEL
                                       , const char *log_name
                                       , UniqueID uid
#endif
                                       );
      // Map a virtual region to a composite instance 
      CompositeRef map_virtual_region(RegionTreeContext ctx,
                                      RegionRequirement &req,
                                      unsigned index,
                                      VersionInfo &version_info
#ifdef DEBUG_HIGH_LEVEL
                                      , const char *log_name
                                      , UniqueID uid
#endif
                                      );
      InstanceRef register_physical_region(RegionTreeContext ctx,
                                           const MappingRef &ref,
                                           RegionRequirement &req,
                                           unsigned index,
                                           VersionInfo &version_info,
                                           Operation *op,
                                           Processor local_proc,
                                           Event term_event
#ifdef DEBUG_HIGH_LEVEL
                                           , const char *log_name
                                           , UniqueID uid
#endif
                                           );
      void register_virtual_region(RegionTreeContext ctx,
                                   CompositeView *composite_view,
                                   RegionRequirement &req,
                                   VersionInfo &version_info);
      bool perform_close_operation(RegionTreeContext ctx,
                                   RegionRequirement &req,
                                   SingleTask *parent_ctx,
                                   Processor local_proc,
                     const LegionMap<ColorPoint,FieldMask>::aligned &targets,
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
#endif
      // This takes ownership of the value buffer
      void fill_fields(RegionTreeContext ctx,
                       const RegionRequirement &req,
                       const void *value, size_t value_size,
                       VersionInfo &version_info);
      InstanceRef attach_file(RegionTreeContext ctx,
                              const RegionRequirement &req,
                              AttachOp *attach_op,
                              VersionInfo &version_info);
      Event detach_file(RegionTreeContext ctx, 
                        const RegionRequirement &req, DetachOp *detach_op, 
                        VersionInfo &version_info, const InstanceRef &ref);
    public:
      void send_back_logical_state(RegionTreeContext local_context,
                                   RegionTreeContext remote_context,
                                   const RegionRequirement &req,
                                   AddressSpaceID target);
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
      FieldSpaceNode* create_node(FieldSpace space, Event dist_alloc);
      RegionNode*     create_node(LogicalRegion r, PartitionNode *par);
      PartitionNode*  create_node(LogicalPartition p, RegionNode *par);
    public:
      IndexSpaceNode* get_node(IndexSpace space);
      IndexPartNode*  get_node(IndexPartition part);
      FieldSpaceNode* get_node(FieldSpace space);
      RegionNode*     get_node(LogicalRegion handle, bool need_check = true);
      PartitionNode*  get_node(LogicalPartition handle, bool need_check = true);
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
    public:
      bool are_disjoint(IndexSpace parent, IndexSpace child);
      bool are_disjoint(IndexSpace parent, IndexPartition child);
      bool are_disjoint(IndexPartition one, IndexPartition two); 
    public:
      bool are_compatible(IndexSpace left, IndexSpace right);
      bool is_dominated(IndexSpace src, IndexSpace dst);
    public:
      bool compute_index_path(IndexSpace parent, IndexSpace child,
                              std::vector<ColorPoint> &path);
      bool compute_partition_path(IndexSpace parent, IndexPartition child,
                                  std::vector<ColorPoint> &path); 
    public:
      void initialize_path(IndexSpace child, IndexSpace parent,
                           RegionTreePath &path);
      void initialize_path(IndexPartition child, IndexSpace parent,
                           RegionTreePath &path);
      void initialize_path(IndexSpace child, IndexPartition parent,
                           RegionTreePath &path);
      void initialize_path(IndexPartition child, IndexPartition parent,
                           RegionTreePath &path);
    public:
      FatTreePath* compute_fat_path(IndexSpace child, IndexSpace parent,
                               std::map<IndexTreeNode*,FatTreePath*> &storage,
                               bool test_overlap, bool &overlap);
      FatTreePath* compute_fat_path(IndexSpace child, 
                                    IndexPartition parent,
                               std::map<IndexTreeNode*,FatTreePath*> &storage,
                               bool test_overlap, bool &overlap);
      FatTreePath* compute_full_fat_path(IndexSpace handle);
      FatTreePath* compute_full_fat_path(IndexPartition handle);
    protected:
      FatTreePath* compute_fat_path(IndexTreeNode *child, IndexTreeNode *parent,
                                 std::map<IndexTreeNode*,FatTreePath*> &storage,
                                 bool test_overlap, bool &overlap);
      FatTreePath* compute_full_fat_path(IndexSpaceNode *node);
      FatTreePath* compute_full_fat_path(IndexPartNode *node);
    public:
      // Interfaces to the low-level runtime
      Event issue_copy(const Domain &dom, Operation *op,
                       const std::vector<Domain::CopySrcDstField> &src_fields,
                       const std::vector<Domain::CopySrcDstField> &dst_fields,
                       Event precondition = Event::NO_EVENT);
      Event issue_fill(const Domain &dom, UniqueID uid,
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
                                       size_t field_size, UniqueID op_id);
      PhysicalInstance create_instance(const Domain &dom, Memory target,
                                       const std::vector<size_t> &field_sizes,
                                       size_t blocking_factor, UniqueID op_id);
      PhysicalInstance create_instance(const Domain &dom, Memory target,
                                       size_t field_size, ReductionOpID redop,
                                       UniqueID op_id);
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
                                       AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable);
      void attach_semantic_information(IndexPartition handle, SemanticTag tag,
                                       AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable);
      void attach_semantic_information(FieldSpace handle, SemanticTag tag,
                                       AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable);
      void attach_semantic_information(FieldSpace handle, FieldID fid,
                                       SemanticTag tag, AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable);
      void attach_semantic_information(LogicalRegion handle, SemanticTag tag,
                                       AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable);
      void attach_semantic_information(LogicalPartition handle, SemanticTag tag,
                                       AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable);
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
      REGISTER_LOGICAL_DEPS_CALL,
      CLOSE_PHYSICAL_NODE_CALL,
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
      INITIALIZE_CURRENT_STATE_CALL,
      INVALIDATE_CURRENT_STATE_CALL,
      PERFORM_DEPENDENCE_CHECKS_CALL,
      PERFORM_CLOSING_CHECKS_CALL,
      REMAP_REGION_CALL,
      REGISTER_REGION_CALL,
      CLOSE_PHYSICAL_STATE_CALL,
      GARBAGE_COLLECT_CALL,
      NOTIFY_INVALID_CALL,
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
      virtual void get_colors(std::set<ColorPoint> &colors) = 0;
      virtual void send_node(AddressSpaceID target, bool up, bool down) = 0;
    public:
      virtual bool is_index_space_node(void) const = 0;
      virtual IndexSpaceNode* as_index_space_node(void) = 0;
      virtual IndexPartNode* as_index_part_node(void) = 0;
      virtual AddressSpaceID get_owner_space(void) const = 0;
    public:
      void attach_semantic_information(SemanticTag tag, AddressSpaceID source,
                             const void *buffer, size_t size, bool is_mutable);
      void retrieve_semantic_information(SemanticTag tag,
                                         const void *&result, size_t &size);
      virtual void send_semantic_request(AddressSpaceID target, 
                                         SemanticTag tag) = 0;
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                        const void *buffer, size_t size, bool is_mutable) = 0;
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
      struct SemanticRequestArgs {
        HLRTaskID hlr_id;
        IndexSpaceNode *proxy_this;
        SemanticTag tag;
        AddressSpaceID source;
      };
      struct ChildRequestFunctor {
      public:
        ChildRequestFunctor(Runtime *rt, Serializer &r, AddressSpaceID t)
          : runtime(rt), rez(r), target(t) { }
      public:
        void apply(AddressSpaceID next);
      private:
        Runtime *const runtime;
        Serializer &rez;
        AddressSpaceID target;
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
      virtual AddressSpaceID get_owner_space(void) const;
      static AddressSpaceID get_owner_space(IndexSpace handle, Runtime *rt);
    public:
      virtual IndexTreeNode* get_parent(void) const;
      virtual size_t get_num_elmts(void);
      virtual void get_colors(std::set<ColorPoint> &colors);
    public:
      virtual void send_semantic_request(AddressSpaceID target, 
                                         SemanticTag tag);
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                           const void *buffer, size_t size, bool is_mutable);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source);
      static void handle_semantic_request(RegionTreeForest *forest,
                                 Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(RegionTreeForest *forest,
                                 Deserializer &derez, AddressSpaceID source);
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
          std::map<DomainPoint, Realm::IndexSpace> &subspaces,
          bool mutable_results, Event precondition);
      Event create_subspaces_by_image(
          const std::vector<FieldDataDescriptor> &field_data,
          std::map<Realm::IndexSpace, Realm::IndexSpace> &subpsaces,
          bool mutable_results, Event precondition);
      Event create_subspaces_by_preimage(
          const std::vector<FieldDataDescriptor> &field_data,
          std::map<Realm::IndexSpace, Realm::IndexSpace> &subspaces,
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
      void send_child_node(AddressSpaceID target, 
                    const ColorPoint &child_color, UserEvent to_trigger);
    public:
      static void handle_node_request(RegionTreeForest *context,
                                      Deserializer &derez,
                                      AddressSpaceID source);
      static void handle_node_return(Deserializer &derez);
      static void handle_node_child_request(RegionTreeForest *context,
                                            Deserializer &derez);
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
      struct SemanticRequestArgs {
        HLRTaskID hlr_id;
        IndexPartNode *proxy_this;
        SemanticTag tag;
        AddressSpaceID source;
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
      virtual AddressSpaceID get_owner_space(void) const;
      static AddressSpaceID get_owner_space(IndexPartition handle, Runtime *rt);
    public:
      virtual IndexTreeNode* get_parent(void) const;
      virtual size_t get_num_elmts(void);
      virtual void get_colors(std::set<ColorPoint> &colors);
    public:
      virtual void send_semantic_request(AddressSpaceID target, 
                                         SemanticTag tag);
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                             const void *buffer, size_t size, bool is_mutable);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source);
      static void handle_semantic_request(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
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
                                Realm::IndexSpace::IndexSpaceOperation op);
      Event create_by_operation(IndexSpaceNode *left, IndexPartNode *right,
                                Realm::IndexSpace::IndexSpaceOperation op);
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
        FieldInfo(void) : field_size(0), idx(0), serdez_id(0),
                          local(false), destroyed(false) { }
        FieldInfo(size_t size, unsigned id, bool loc, CustomSerdezID sid)
          : field_size(size), idx(id), serdez_id(sid),
            local(loc), destroyed(false) { }
      public:
        size_t field_size;
        unsigned idx;
        CustomSerdezID serdez_id;
        bool local;
        bool destroyed;
      };
      struct SendFieldAllocationFunctor {
      public:
        SendFieldAllocationFunctor(FieldSpace h, FieldID f, size_t s,
                                   unsigned i, CustomSerdezID sid,
                                   Runtime *rt)
          : handle(h), field(f), size(s), index(i), 
            serdez_id(sid), runtime(rt) { }
      public:
        void apply(AddressSpaceID target);
      private:
        FieldSpace handle;
        FieldID field;
        size_t size;
        unsigned index;
        CustomSerdezID serdez_id;
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
      struct UpgradeFunctor {
      public:
        UpgradeFunctor(std::map<AddressSpaceID,UserEvent> &ts,
                       std::set<Event> &pre)
          : to_send(ts), preconditions(pre) { }
      public:
        void apply(AddressSpaceID target);
      private:
        std::map<AddressSpaceID,UserEvent> &to_send;
        std::set<Event> &preconditions;
      };
      struct SemanticRequestArgs {
        HLRTaskID hlr_id;
        FieldSpaceNode *proxy_this;
        SemanticTag tag;
        AddressSpaceID source;
      };
      struct SemanticFieldRequestArgs {
        HLRTaskID hlr_id;
        FieldSpaceNode *proxy_this;
        FieldID fid;
        SemanticTag tag;
        AddressSpaceID source;
      };
    public:
      FieldSpaceNode(FieldSpace sp, Event dist_alloc,
                     RegionTreeForest *ctx);
      FieldSpaceNode(const FieldSpaceNode &rhs);
      ~FieldSpaceNode(void);
    public:
      FieldSpaceNode& operator=(const FieldSpaceNode &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
      AddressSpaceID get_owner_space(void) const; 
      static AddressSpaceID get_owner_space(FieldSpace handle, Runtime *rt);
    public:
      void attach_semantic_information(SemanticTag tag, AddressSpaceID source,
                            const void *buffer, size_t size, bool is_mutable);
      void attach_semantic_information(FieldID fid, SemanticTag tag,
                                       AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable);
      void retrieve_semantic_information(SemanticTag tag,
                                         const void *&result, size_t &size);
      void retrieve_semantic_information(FieldID fid, SemanticTag tag,
                                         const void *&result, size_t &size);
      void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                             const void *result, size_t size, bool is_mutable);
      void send_semantic_field_info(AddressSpaceID target, FieldID fid,
            SemanticTag tag, const void *result, size_t size, bool is_mutable);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source);
      void process_semantic_field_request(FieldID fid, SemanticTag tag, 
                                          AddressSpaceID source);
      static void handle_semantic_request(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_field_semantic_request(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_field_semantic_info(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
    public:
      void allocate_field(FieldID fid, size_t size, bool local, 
                          CustomSerdezID serdez_id);
      void allocate_field_index(FieldID fid, size_t size, 
                                AddressSpaceID runtime, unsigned index,
                                CustomSerdezID serdez_id);
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
                                  std::vector<unsigned> &indexes,
                                  std::vector<CustomSerdezID> &serdez);
    public:
      InstanceManager* create_instance(Memory location, Domain dom,
                                       const std::set<FieldID> &fields,
                                       size_t blocking_factor, unsigned depth,
                                       RegionNode *node, DistributedID did,
                                       UniqueID op_id, bool &remote);
      ReductionManager* create_reduction(Memory location, Domain dom,
                                        FieldID fid, bool reduction_list,
                                        RegionNode *node, ReductionOpID redop,
                                        DistributedID did, UniqueID op_id,
                                        bool &remote_creation);
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
                                   const std::vector<unsigned> &indexes,
                                   const std::vector<CustomSerdezID> &serdez);
      LayoutDescription* register_layout_description(LayoutDescription *desc);
    public:
      void upgrade_distributed_alloc(UserEvent to_trigger);
      void process_upgrade(UserEvent to_trigger, Event ready_event);
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
      static void handle_distributed_alloc_request(RegionTreeForest *forest,
                                                   Deserializer &derez);
      static void handle_distributed_alloc_upgrade(RegionTreeForest *forest,
                                                   Deserializer &derez);
    public:
      static void handle_remote_instance_creation(RegionTreeForest *forest,
                                Deserializer &derez, AddressSpaceID source);
      static void handle_remote_reduction_creation(RegionTreeForest *forest,
                                Deserializer &derez, AddressSpaceID source);
    public:
      // Help with debug printing
      char* to_string(const FieldMask &mask) const;
      void get_field_ids(const FieldMask &mask,
                         std::vector<FieldID> &fields) const;
    protected:
      // Assume we are already holding the node lock
      // when calling these methods
      unsigned allocate_index(bool local, int goal=-1);
      void free_index(unsigned index);
    public:
      const FieldSpace handle;
      const bool is_owner;
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
    private:
      /*
       * Every field space contains a permutation transformer that
       * can translate a field mask from any other node onto
       * this node, this is only necessary when we are doing
       * distributed field allocations on multiple nodes.
       */
      LegionMap<AddressSpaceID,FieldPermutation>::aligned transformers;
      // Track if we are in a distributed allocation mode
      // and if not, are we the owner space
      Event distributed_allocation;
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
      static AddressSpaceID get_owner_space(RegionTreeID tid, Runtime *rt);
    public:
      void set_restricted_fields(ContextID ctx, FieldMask &child_restricted);
      inline PhysicalState* get_physical_state(ContextID ctx, VersionInfo &info,
                                               bool capture = true)
      {
        // First check to see if the version info already has a state
        PhysicalState *result = info.find_physical_state(this, capture);  
#ifdef DEBUG_HIGH_LEVEL
        assert(result != NULL);
#endif
        return result;
      }
      inline CurrentState& get_current_state(ContextID ctx)
      {
        return *(current_states.lookup_entry(ctx, this, ctx));
      }
      inline CurrentState* get_current_state_ptr(ContextID ctx)
      {
        return current_states.lookup_entry(ctx, this, ctx);
      }
    public:
      void attach_semantic_information(SemanticTag tag, AddressSpaceID source,
                            const void *buffer, size_t size, bool is_mutable);
      void retrieve_semantic_information(SemanticTag tag,
                                         const void *&result, size_t &size);
      virtual void send_semantic_request(AddressSpaceID target, 
                                         SemanticTag tag) = 0;
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                          const void *buffer, size_t size, bool is_mutable) = 0;
    public:
      // Logical traversal operations
      void register_logical_node(ContextID ctx,
                                 const LogicalUser &user,
                                 RegionTreePath &path,
                                VersionInfo &version_info,
                                 RestrictInfo &restrict_info,
                                 const TraceInfo &trace_info,
                                 const bool projecting,
                                 const bool report_uninitialized = false);
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
                                     const TraceInfo &trace_info,
                                     const bool report_uninitialized = false);
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
                              bool permit_leave_open,
                              bool read_only_close);
      void siphon_logical_children(LogicalCloser &closer,
                                   CurrentState &state,
                                   const FieldMask &closing_mask,
                                   bool record_close_operations,
                                   const ColorPoint &next_child,
                                   FieldMask &open_below);
      // Note that 'allow_next_child' and 
      // 'record_closed_fields' are mutually exclusive
      void perform_close_operations(LogicalCloser &closer,
                                    const FieldMask &closing_mask,
                                    FieldState &closing_state,
                                    const ColorPoint &next_child, 
                                    bool allow_next_child,
                                    bool upgrade_next_child, 
                                    bool permit_leave_open,
                                    bool read_only_close,
                                    bool record_close_operations,
                                    bool record_closed_fields,
                                   LegionDeque<FieldState>::aligned &new_states,
                                    FieldMask &output_mask);
      void merge_new_field_state(CurrentState &state, 
                                 const FieldState &new_state);
      void merge_new_field_states(CurrentState &state, 
                            const LegionDeque<FieldState>::aligned &new_states);
      void filter_prev_epoch_users(CurrentState &state, const FieldMask &mask);
      void filter_curr_epoch_users(CurrentState &state, const FieldMask &mask);
      void report_uninitialized_usage(const LogicalUser &user,
                                    const FieldMask &uninitialized);
      void record_logical_reduction(CurrentState &state, ReductionOpID redop,
                                    const FieldMask &user_mask);
      void clear_logical_reduction_fields(CurrentState &state,
                                          const FieldMask &cleared_mask);
      void sanity_check_logical_state(CurrentState &state);
      void register_logical_dependences(ContextID ctx, Operation *op,
                                        const FieldMask &field_mask,
                                        bool dominate);
      void add_restriction(ContextID ctx, const FieldMask &restricted_mask);
      void release_restriction(ContextID ctx, const FieldMask &restricted_mask);
      void record_logical_restrictions(ContextID ctx, RestrictInfo &info,
                                       const FieldMask &mask);
    public:
      void send_back_logical_state(ContextID local_ctx, ContextID remote_ctx,
                                   const FieldMask &send_mask, 
                                   AddressSpaceID target);
      void process_logical_state_return(Deserializer &derez,
                                        AddressSpaceID source);
      static void handle_logical_state_return(RegionTreeForest *forest,
                                              Deserializer &derez,
                                              AddressSpaceID source);
    public:
      void initialize_current_state(ContextID ctx);
      void invalidate_current_state(ContextID ctx, bool logical_users_only);
    public:
      // Physical traversal operations
      // Entry
      void close_physical_node(PhysicalCloser &closer,
                               const FieldMask &closing_mask);
      void siphon_physical_children(PhysicalCloser &closer,
                                    PhysicalState *state,
                                    const FieldMask &closing_mask,
                                    const std::set<ColorPoint> &next_children);
      void close_physical_child(PhysicalCloser &closer,
                                PhysicalState *state,
                                const FieldMask &closing_mask,
                                const ColorPoint &target_child,
                                FieldMask &child_mask,
                                const std::set<ColorPoint> &next_children,
                                bool &changed);
      // Analogous methods to those above except for closing to a composite view
      CompositeRef create_composite_instance(ContextID ctx_id,
                       const LegionMap<ColorPoint,FieldMask>::aligned &targets,
                                     const std::set<ColorPoint> &next_children,
                                     const FieldMask &closing_mask,
                                     VersionInfo &version_info,
                                     bool register_instance); 
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
                                     const FieldMask &mask, bool needs_space,
                                     VersionInfo &version_info);
      void find_copy_across_instances(const TraversalInfo &info,
                                      MaterializedView *target,
                 LegionMap<MaterializedView*,FieldMask>::aligned &src_instances,
               LegionMap<DeferredView*,FieldMask>::aligned &deferred_instances);
      // Since figuring out how to issue copies is expensive, try not
      // to hold the physical state lock when doing them. NOTE IT IS UNSOUND
      // TO CALL THIS METHOD WITH A SET OF VALID INSTANCES ACQUIRED BY PASSING
      // 'TRUE' TO THE find_valid_instance_views METHOD!!!!!!!!
      void issue_update_copies(const TraversalInfo &info,
                               MaterializedView *target, 
                               FieldMask copy_mask,
            const LegionMap<LogicalView*,FieldMask>::aligned &valid_instances,
                               CopyTracker *tracker = NULL);
      void sort_copy_instances(const TraversalInfo &info,
                               MaterializedView *target,
                               FieldMask &copy_mask,
               const LegionMap<LogicalView*,FieldMask>::aligned &copy_instances,
                 LegionMap<MaterializedView*,FieldMask>::aligned &src_instances,
               LegionMap<DeferredView*,FieldMask>::aligned &deferred_instances);
      // Issue copies for fields with the same event preconditions
      static void issue_grouped_copies(RegionTreeForest *context,
                                       const TraversalInfo &info,
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
          const LegionMap<ReductionView*,FieldMask>::aligned &valid_reductions,
                                   Operation *op,
                                   CopyTracker *tracker = NULL);
      void invalidate_instance_views(PhysicalState *state,
                                     const FieldMask &invalid_mask); 
      void invalidate_reduction_views(PhysicalState *state,
                                      const FieldMask &invalid_mask);
      // Look for a view to remove from the set of valid views
      void filter_valid_views(PhysicalState *state, LogicalView *to_filter);
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
                            const TraversalInfo &info,
                            CopyTracker *tracker = NULL);
      LogicalView* convert_reference(const InstanceRef &ref, ContextID ctx);
    public: // Help for physical closes
      void find_complete_fields(const FieldMask &scope_fields,
          const LegionMap<ColorPoint,FieldMask>::aligned &children,
          FieldMask &complete_fields);
      void convert_target_views(const InstanceSet &targets, ContextID ctx,
                                std::vector<MaterializedView*> &target_views);
    public:
      void add_persistent_view(ContextID ctx, MaterializedView *persist_view);
      void remove_persistent_view(ContextID ctx,MaterializedView *persist_view);
    public:
      bool register_logical_view(LogicalView *view);
      void unregister_logical_view(LogicalView *view);
      LogicalView* find_view(DistributedID did);
      VersionState* find_remote_version_state(ContextID ctx, VersionID vid,
                                  DistributedID did, AddressSpaceID owner);
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
      virtual AddressSpaceID get_owner_space(void) const = 0;
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
                        const LegionMap<ColorPoint,FieldMask>::aligned &targets,
                                            const VersionInfo &close_info,
                                            const VersionInfo &version_info,
                                            const RestrictInfo &res_info,
                                            const TraceInfo &trace_info) = 0;
      virtual ReadCloseOp* create_read_only_close_op(Operation *creator,
                                            const FieldMask &closing_mask,
                        const LegionMap<ColorPoint,FieldMask>::aligned &targets,
                                            const TraceInfo &trace_info) = 0;
      virtual Event perform_close_operation(const TraversalInfo &info,
                                            const FieldMask &closing_mask,
                const LegionMap<ColorPoint,FieldMask>::aligned &target_children,
                                            const InstanceSet &targets, 
                                            VersionInfo &version_info,
                                 const std::set<ColorPoint> &next_children) = 0;
      virtual MaterializedView * create_instance(Memory target_mem,
                                                const std::set<FieldID> &fields,
                                                size_t blocking_factor,
                                                unsigned depth, Operation *op,
                                                bool &remote_creation) = 0;
      virtual ReductionView* create_reduction(Memory target_mem,
                                              FieldID fid,
                                              bool reduction_list,
                                              ReductionOpID redop,
                                              Operation *op,
                                              bool &remote_creation) = 0;
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
      DynamicTable<CurrentStateAllocator> current_states;
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
      struct SemanticRequestArgs {
        HLRTaskID hlr_id;
        RegionNode *proxy_this;
        SemanticTag tag;
        AddressSpaceID source;
      };
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
      virtual AddressSpaceID get_owner_space(void) const;
      static AddressSpaceID get_owner_space(LogicalRegion handle, Runtime *rt);
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
                        const LegionMap<ColorPoint,FieldMask>::aligned &targets,
                                            const VersionInfo &close_info,
                                            const VersionInfo &version_info,
                                            const RestrictInfo &res_info,
                                            const TraceInfo &trace_info);
      virtual ReadCloseOp* create_read_only_close_op(Operation *creator,
                                            const FieldMask &closing_mask,
                        const LegionMap<ColorPoint,FieldMask>::aligned &targets,
                                            const TraceInfo &trace_info);
      virtual Event perform_close_operation(const TraversalInfo &info,
                                            const FieldMask &closing_mask,
                const LegionMap<ColorPoint,FieldMask>::aligned &target_children,
                                            const InstanceSet &targets,
                                            VersionInfo &version_info,
                                     const std::set<ColorPoint> &next_children);
      virtual MaterializedView* create_instance(Memory target_mem,
                                                const std::set<FieldID> &fields,
                                                size_t blocking_factor,
                                                unsigned depth, Operation *op,
                                                bool &remote);
      virtual ReductionView* create_reduction(Memory target_mem,
                                              FieldID fid,
                                              bool reduction_list,
                                              ReductionOpID redop,
                                              Operation *op,
                                              bool &remote_creation);
      virtual void send_node(AddressSpaceID target);
      static void handle_node_creation(RegionTreeForest *context,
                            Deserializer &derez, AddressSpaceID source);
    public:
      virtual void send_semantic_request(AddressSpaceID target, 
                                         SemanticTag tag);
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                             const void *buffer, size_t size, bool is_mutable);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source);
      static void handle_semantic_request(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
    public:
      static void handle_top_level_request(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_top_level_return(Deserializer &derez);
    public:
      // Logging calls
      virtual void print_logical_context(ContextID ctx, 
                                         TreeStateLogger *logger,
                                         const FieldMask &mask);
      virtual void print_physical_context(ContextID ctx, 
                                          TreeStateLogger *logger,
                                          const FieldMask &mask);
      void print_logical_state(CurrentState &state,
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
      CompositeRef map_virtual_region(ContextID ctx, 
                                      const FieldMask &virtual_mask,
                                      VersionInfo &version_info);
      void register_region(const TraversalInfo &info, Event term_event,
                           const RegionUsage &usage, 
                           const FieldMask &user_mask,
                           InstanceSet &targets);
      void register_virtual(ContextID ctx, CompositeView *view,
                            VersionInfo &version_info,
                            const FieldMask &composite_mask);
      void seed_state(ContextID ctx, Event term_event,
                             const RegionUsage &usage,
                             const FieldMask &user_mask,
                             LogicalView *new_view);
      Event close_state(const TraversalInfo &info, Event term_event,
                        RegionUsage &usage, const InstanceSet &targets);
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
      Event detach_file(ContextID ctx, DetachOp *detach_op, 
                        VersionInfo &version_info, const InstanceRef &ref);
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
      struct SemanticRequestArgs {
        HLRTaskID hlr_id;
        PartitionNode *proxy_this;
        SemanticTag tag;
        AddressSpaceID source;
      };
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
      virtual AddressSpaceID get_owner_space(void) const;
      static AddressSpaceID get_owner_space(LogicalPartition handle, 
                                            Runtime *runtime);
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
                        const LegionMap<ColorPoint,FieldMask>::aligned &targets,
                                            const VersionInfo &close_info,
                                            const VersionInfo &version_info,
                                            const RestrictInfo &res_info,
                                            const TraceInfo &trace_info);
      virtual ReadCloseOp* create_read_only_close_op(Operation *creator,
                                            const FieldMask &closing_mask,
                        const LegionMap<ColorPoint,FieldMask>::aligned &targets,
                                            const TraceInfo &trace_info);
      virtual Event perform_close_operation(const TraversalInfo &info,
                                            const FieldMask &closing_mask,
                const LegionMap<ColorPoint,FieldMask>::aligned &target_children,
                                            const InstanceSet &targets,
                                            VersionInfo &version_info,
                                     const std::set<ColorPoint> &next_children);
      virtual MaterializedView* create_instance(Memory target_mem,
                                                const std::set<FieldID> &fields,
                                                size_t blocking_factor,
                                                unsigned depth, Operation *op,
                                                bool &remote);
      virtual ReductionView* create_reduction(Memory target_mem,
                                              FieldID fid,
                                              bool reduction_list,
                                              ReductionOpID redop,
                                              Operation *op,
                                              bool &remote_creation);
      virtual void send_node(AddressSpaceID target);
    public:
      virtual void send_semantic_request(AddressSpaceID target, 
                                         SemanticTag tag);
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                             const void *buffer, size_t size, bool is_mutable);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source);
      static void handle_semantic_request(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
    public:
      // Logging calls
      virtual void print_logical_context(ContextID ctx, 
                                         TreeStateLogger *logger,
                                         const FieldMask &mask);
      virtual void print_physical_context(ContextID ctx, 
                                          TreeStateLogger *logger,
                                          const FieldMask &mask);
      void print_logical_state(CurrentState &state,
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

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_REGION_TREE_H__

// EOF

