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


#ifndef __LEGION_REGION_TREE_H__
#define __LEGION_REGION_TREE_H__

#include "legion/legion_types.h"
#include "legion/legion_utilities.h"
#include "legion/legion_allocation.h"
#include "legion/legion_analysis.h"
#include "legion/garbage_collection.h"
#include "legion/field_tree.h"

namespace Legion {
  namespace Internal {

    /**
     * \struct FieldDataDescriptor
     * A small helper class for performing dependent
     * partitioning operations
     */
    struct FieldDataDescriptor {
    public:
      IndexSpace index_space;
      PhysicalInstance inst;
      size_t field_offset;
    };
    
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
      void prepare_for_shutdown(void);
    public:
      void create_index_space(IndexSpace handle, const void *realm_is,
                              DistributedID did, ShardMapping *mapping = NULL);
      IndexSpace find_or_create_union_space(InnerContext *ctx,
                              const std::vector<IndexSpace> &sources,
                              ShardMapping *mapping = NULL);
      IndexSpace find_or_create_intersection_space(InnerContext *ctx,
                              const std::vector<IndexSpace> &sources,
                              ShardMapping *mapping = NULL);
      IndexSpace find_or_create_difference_space(InnerContext *ctx,
                              IndexSpace left, IndexSpace right,
                              ShardMapping *mapping = NULL);
      void find_or_create_sharded_index_space(InnerContext *ctx,
                              IndexSpace handle, IndexSpace local,
                              DistributedID did, ShardMapping *mapping);
      RtEvent create_pending_partition(IndexPartition pid,
                                       IndexSpace parent,
                                       IndexSpace color_space,
                                       LegionColor partition_color,
                                       PartitionKind part_kind,
                                       DistributedID did,
                                       ApEvent partition_ready,
                  ApBarrier partial_pending = ApBarrier::NO_AP_BARRIER);
      void create_pending_cross_product(IndexPartition handle1,
                                        IndexPartition handle2,
                  std::map<IndexSpace,IndexPartition> &user_handles,
                                           PartitionKind kind,
                                           LegionColor &part_color,
                                           ApEvent domain_ready,
                                           ShardID shard = 0,
                                           size_t total_shards = 1);
      // For control replication contexts
      RtEvent create_pending_partition_shard(ShardID owner_shard,
                                             ReplicateContext *ctx,
                                             IndexPartition pid,
                                             IndexSpace parent,
                                             IndexSpace color_space,
                                             LegionColor &partition_color,
                                             PartitionKind part_kind,
                                             DistributedID did,
                                             ValueBroadcast<bool> *part_result,
                                             ApEvent partition_ready,
                                             ShardMapping *mapping,
                                             RtEvent creation_ready,
                   ApBarrier partial_pending = ApBarrier::NO_AP_BARRIER);
      void destroy_index_space(IndexSpace handle, AddressSpaceID source);
      void destroy_index_partition(IndexPartition handle, 
                                   AddressSpaceID source);
    public:
      ApEvent create_equal_partition(Operation *op, 
                                     IndexPartition pid, 
                                     size_t granularity,
                                     ShardID shard = 0,
                                     size_t total_shards = 1);
      ApEvent create_partition_by_union(Operation *op,
                                        IndexPartition pid,
                                        IndexPartition handle1,
                                        IndexPartition handle2,
                                        ShardID shard = 0, 
                                        size_t total_shards = 1);
      ApEvent create_partition_by_intersection(Operation *op,
                                               IndexPartition pid,
                                               IndexPartition handle1,
                                               IndexPartition handle2,
                                               ShardID shard = 0,
                                               size_t total_shards = 1);
      ApEvent create_partition_by_difference(Operation *op,
                                           IndexPartition pid,
                                           IndexPartition handle1,
                                           IndexPartition handle2,
                                           ShardID shard = 0,
                                           size_t total_shards = 1);
      ApEvent create_partition_by_restriction(IndexPartition pid,
                                              const void *transform,
                                              const void *extent,
                                              ShardID shard = 0,
                                              size_t total_shards = 1);
      ApEvent create_cross_product_partitions(Operation *op,
                                              IndexPartition base,
                                              IndexPartition source,
                                              LegionColor part_color,
                                              ShardID shard = 0,
                                              size_t total_shards = 1);
    public:  
      ApEvent create_partition_by_field(Operation *op,
                                        IndexPartition pending,
                    const std::vector<FieldDataDescriptor> &instances,
                                        ApEvent instances_ready);
      ApEvent create_partition_by_image(Operation *op,
                                        IndexPartition pending,
                                        IndexPartition projection,
                    const std::vector<FieldDataDescriptor> &instances,
                                        ApEvent instances_ready,
                                        ShardID shard = 0,
                                        size_t total_shards = 1);
      ApEvent create_partition_by_image_range(Operation *op,
                                              IndexPartition pending,
                                              IndexPartition projection,
                    const std::vector<FieldDataDescriptor> &instances,
                                              ApEvent instances_ready,
                                              ShardID shard = 0,
                                              size_t total_shards = 1);
      ApEvent create_partition_by_preimage(Operation *op,
                                           IndexPartition pending,
                                           IndexPartition projection,
                    const std::vector<FieldDataDescriptor> &instances,
                                           ApEvent instances_ready);
      ApEvent create_partition_by_preimage_range(Operation *op,
                                                 IndexPartition pending,
                                                 IndexPartition projection,
                    const std::vector<FieldDataDescriptor> &instances,
                                                 ApEvent instances_ready);
      ApEvent create_association(Operation *op, 
                                 IndexSpace domain, IndexSpace range,
                    const std::vector<FieldDataDescriptor> &instances,
                                 ApEvent instances_ready);
    public:
      bool check_partition_by_field_size(IndexPartition pid,
                                         FieldSpace fspace, FieldID fid,
                                         bool is_range,
                                         bool use_color_space = false);
      bool check_association_field_size(IndexSpace is,
                                        FieldSpace fspace, FieldID fid);
    public:
      ApEvent compute_pending_space(Operation *op, IndexSpace result,
                                    const std::vector<IndexSpace> &handles,
                                    bool is_union, ShardID shard = 0,
                                    size_t total_shards = 1);
      ApEvent compute_pending_space(Operation *op, IndexSpace result,
                                    IndexPartition handle,
                                    bool is_union, ShardID shard = 0,
                                    size_t total_shards = 1);
      ApEvent compute_pending_space(Operation *op, IndexSpace result,
                                    IndexSpace initial,
                                    const std::vector<IndexSpace> &handles,
                                    ShardID shard = 0, size_t total_shards = 1);
    public:
      IndexPartition get_index_partition(IndexSpace parent, Color color); 
      bool has_index_subspace(IndexPartition parent,
                              const void *realm_color, TypeTag type_tag);
      IndexSpace get_index_subspace(IndexPartition parent, 
                                    const void *realm_color,
                                    TypeTag type_tag);
      void get_index_space_domain(IndexSpace handle, 
                                  void *realm_is, TypeTag type_tag);
      IndexSpace get_index_partition_color_space(IndexPartition p);
      void get_index_space_partition_colors(IndexSpace sp,
                                            std::set<Color> &colors);
      void get_index_space_color(IndexSpace handle, 
                                 void *realm_color, TypeTag type_tag); 
      Color get_index_partition_color(IndexPartition handle);
      IndexSpace get_parent_index_space(IndexPartition handle);
      bool has_parent_index_partition(IndexSpace handle);
      IndexPartition get_parent_index_partition(IndexSpace handle);
      unsigned get_index_space_depth(IndexSpace handle);
      unsigned get_index_partition_depth(IndexPartition handle);
      size_t get_domain_volume(IndexSpace handle);
      bool is_index_partition_disjoint(IndexPartition p);
      bool is_index_partition_complete(IndexPartition p);
      bool has_index_partition(IndexSpace parent, Color color);
    public:
      void create_field_space(FieldSpace handle, DistributedID did,
                              ShardMapping *mapping = NULL);
      void destroy_field_space(FieldSpace handle, AddressSpaceID source);
      // Return true if local is set to true and we actually performed the 
      // allocation.  It is an error if the field already existed and the
      // allocation was not local.
      bool allocate_field(FieldSpace handle, size_t field_size, 
                          FieldID fid, CustomSerdezID serdez_id);
      void free_field(FieldSpace handle, FieldID fid);
      void allocate_fields(FieldSpace handle, const std::vector<size_t> &sizes,
                           const std::vector<FieldID> &resulting_fields,
                           CustomSerdezID serdez_id);
      void free_fields(FieldSpace handle, const std::vector<FieldID> &to_free);
    public:
      bool allocate_local_fields(FieldSpace handle, 
                                 const std::vector<FieldID> &resulting_fields,
                                 const std::vector<size_t> &sizes,
                                 CustomSerdezID serdez_id,
                                 const std::set<unsigned> &allocated_indexes,
                                 std::vector<unsigned> &new_indexes);
      void free_local_fields(FieldSpace handle,
                             const std::vector<FieldID> &to_free,
                             const std::vector<unsigned> &indexes);
      void update_local_fields(FieldSpace handle,
                               const std::vector<FieldID> &fields,
                               const std::vector<size_t> &sizes,
                               const std::vector<CustomSerdezID> &serdez_ids,
                               const std::vector<unsigned> &indexes);
      void remove_local_fields(FieldSpace handle,
                               const std::vector<FieldID> &to_remove);
    public:
      void get_all_fields(FieldSpace handle, std::set<FieldID> &fields);
      void get_all_regions(FieldSpace handle, std::set<LogicalRegion> &regions);
      size_t get_field_size(FieldSpace handle, FieldID fid);
      void get_field_space_fields(FieldSpace handle, 
                                  std::vector<FieldID> &fields);
    public:
      void create_logical_region(LogicalRegion handle,
                                 ShardMapping *mapping = NULL);
      void destroy_logical_region(LogicalRegion handle, 
                                  AddressSpaceID source);
      void destroy_logical_partition(LogicalPartition handle,
                                     AddressSpaceID source);
    public:
      LogicalPartition get_logical_partition(LogicalRegion parent, 
                                             IndexPartition handle);
      LogicalPartition get_logical_partition_by_color(LogicalRegion parent, 
                                                      Color color);
      bool has_logical_partition_by_color(LogicalRegion parent, Color color);
      LogicalPartition get_logical_partition_by_tree(
          IndexPartition handle, FieldSpace space, RegionTreeID tid);
      LogicalRegion get_logical_subregion(LogicalPartition parent,
                                          IndexSpace handle);
      LogicalRegion get_logical_subregion_by_color(LogicalPartition parent,
                                  const void *realm_color, TypeTag type_tag);
      bool has_logical_subregion_by_color(LogicalPartition parent,
                                  const void *realm_color, TypeTag type_tag);
      LogicalRegion get_logical_subregion_by_tree(
            IndexSpace handle, FieldSpace space, RegionTreeID tid);
      void get_logical_region_color(LogicalRegion handle, 
                                    void *realm_color, TypeTag type_tag);
      Color get_logical_partition_color(LogicalPartition handle);
      LogicalRegion get_parent_logical_region(LogicalPartition handle);
      bool has_parent_logical_partition(LogicalRegion handle);
      LogicalPartition get_parent_logical_partition(LogicalRegion handle);
      size_t get_domain_volume(LogicalRegion handle);
    public:
      // Index space operation methods
      void find_launch_space_domain(IndexSpace handle, Domain &launch_domain);
      void validate_slicing(IndexSpace input_space,
                            const std::vector<IndexSpace> &slice_spaces,
                            MultiTask *task, MapperManager *mapper);
      void log_launch_space(IndexSpace handle, UniqueID op_id);
    public:
      // Logical analysis methods
      void perform_dependence_analysis(Operation *op, unsigned idx,
                                       RegionRequirement &req,
                                       RestrictInfo &restrict_info,
                                       VersionInfo &version_info,
                                       ProjectionInfo &projection_info,
                                       RegionTreePath &path);
      void perform_deletion_analysis(DeletionOp *op, unsigned idx,
                                     RegionRequirement &req,
                                     RestrictInfo &restrict_info,
                                     RegionTreePath &path);
      // Used by dependent partition operations
      void find_open_complete_partitions(Operation *op, unsigned idx,
                                         const RegionRequirement &req,
                                     std::vector<LogicalPartition> &partitions);
      // For privileges flowing back across node boundaries
      void send_back_logical_state(RegionTreeContext context,
                                   UniqueID context_uid,
                                   const RegionRequirement &req,
                                   AddressSpaceID target);
    public:
      void perform_versioning_analysis(Operation *op, unsigned idx,
                                       const RegionRequirement &req,
                                       const RegionTreePath &path,
                                       VersionInfo &version_info,
                                       std::set<RtEvent> &ready_events,
                                       bool partial_traversal = false,
                                       FieldMask *filter_mask = NULL,
                                       RegionTreeNode *parent_node = NULL,
              // For computing split masks for projection epochs only
                                       UniqueID logical_context_uid = 0,
              const LegionMap<ProjectionEpochID,
                              FieldMask>::aligned *advance_epochs = NULL,
                                       bool skip_parent_check = false);
      void advance_version_numbers(Operation *op, unsigned idx,
                                   bool update_parent_state,
                                   bool parent_is_upper_bound,
                                   UniqueID logical_ctx_uid,
                                   bool dedup_opens, bool dedup_advances, 
                                   ProjectionEpochID open_epoch,
                                   ProjectionEpochID advance_epoch,
                                   RegionTreeNode *parent, 
                                   const RegionTreePath &path,
                                   const FieldMask &advance_mask,
             const LegionMap<unsigned,FieldMask>::aligned &dirty_previous,
                                   std::set<RtEvent> &ready_events);
      void advance_remote_versions(Operation *op, unsigned idx,
                                   const RegionRequirement &req,
                                   bool parent_is_upper_bound,
                                   UniqueID logical_ctx_uid,
                                   const VersioningSet<> &versions,
                                   std::set<RtEvent> &ready_events);
      void invalidate_versions(RegionTreeContext ctx, LogicalRegion handle);
      void invalidate_all_versions(RegionTreeContext ctx);
    public:
      void initialize_current_context(RegionTreeContext ctx,
                    const RegionRequirement &req, const InstanceSet &source,
                    ApEvent term_event, InnerContext *context, unsigned index,
                    std::map<PhysicalManager*,InstanceView*> &top_views,
                    std::set<RtEvent> &applied_events);
      void initialize_virtual_context(RegionTreeContext ctx,
                                      const RegionRequirement &req);
      void invalidate_current_context(RegionTreeContext ctx, bool users_only,
                                      LogicalRegion handle);
      bool match_instance_fields(const RegionRequirement &req1,
                                 const RegionRequirement &req2,
                                 const InstanceSet &inst1,
                                 const InstanceSet &inst2);
    public:
      Restriction* create_coherence_restriction(const RegionRequirement &req,
                                                const InstanceSet &instances);
      bool add_acquisition(const std::list<Restriction*> &restrictions,
                           AcquireOp *op, const RegionRequirement &req);
      bool remove_acquisition(const std::list<Restriction*> &restrictions,
                              ReleaseOp *op, const RegionRequirement &req);
      void add_restriction(std::list<Restriction*> &restrictions, AttachOp *op,
                           InstanceManager *inst, const RegionRequirement &req);
      bool remove_restriction(std::list<Restriction*> &restrictions,
                              DetachOp *op, const RegionRequirement &req);
      void perform_restricted_analysis(
                              const std::list<Restriction*> &restrictions,
                              const RegionRequirement &req, 
                              RestrictInfo &restrict_info);
    public: // Physical analysis methods
      void physical_premap_only(Operation *op, unsigned index,
                                const RegionRequirement &req,
                                VersionInfo &version_info,
                                InstanceSet &valid_instances);
      void physical_register_only(const RegionRequirement &req,
                                  VersionInfo &version_info,
                                  RestrictInfo &restrict_info,
                                  Operation *op, unsigned index,
                                  ApEvent term_event,
                                  bool defer_add_users,
                                  bool need_read_only_reservations,
                                  std::set<RtEvent> &map_applied,
                                  InstanceSet &targets,
                                  const ProjectionInfo *proj_info,
                                  const PhysicalTraceInfo &trace_info
#ifdef DEBUG_LEGION
                                 , const char *log_name
                                 , UniqueID uid
#endif
                                 );
      // For when we deferred registration of users
      void physical_register_users(Operation *op, ApEvent term_event,
                                   const RegionRequirement &requirement,
                                   VersionInfo &version_info,
                                   RestrictInfo &restrict_info,
                                   InstanceSet &targets,
                                   std::set<RtEvent> &map_applied_events,
                                   const PhysicalTraceInfo &trace_info);
      void physical_register_users(Operation *op, ApEvent term_event,
                   const std::vector<RegionRequirement> &regions,
                   const std::vector<bool> &to_skip,
                   std::vector<VersionInfo> &version_infos,
                   std::vector<RestrictInfo> &restrict_infos,
                   std::deque<InstanceSet> &targets,
                   std::set<RtEvent> &map_applied_events,
                   const PhysicalTraceInfo &trace_info);
      CompositeView* physical_perform_close(const RegionRequirement &req,
                                  VersionInfo &version_info,
                                  InterCloseOp *op, unsigned index,
                                  RegionTreeNode *close_node,
                                  const FieldMask &closing_mask,
                                  CompositeViewSummary &summary,
                                  std::set<RtEvent> &map_applied,
                                  const RestrictInfo &restrict_info,
                                  const InstanceSet &targets,
                                  // projection_info can be NULL
                                  const ProjectionInfo *projection_info,
                                  const PhysicalTraceInfo &trace_info
#ifdef DEBUG_LEGION
                                  , const char *log_name
                                  , UniqueID uid
#endif
                                  );
      ApEvent physical_close_context(RegionTreeContext ctx,
                                     const RegionRequirement &req,
                                     VersionInfo &version_info,
                                     Operation *op, unsigned index,
                                     std::set<RtEvent> &map_applied,
                                     InstanceSet &targets,
                                     const PhysicalTraceInfo &trace_info
#ifdef DEBUG_LEGION
                                     , const char *log_name
                                     , UniqueID uid
#endif
                                     );
      ApEvent copy_across(const RegionRequirement &src_req,
                          const RegionRequirement &dst_req,
                                InstanceSet &src_targets, 
                          const InstanceSet &dst_targets,
                          VersionInfo &src_version_info,
                          VersionInfo &dst_version_info, 
                          ApEvent term_event, Operation *op,
                          unsigned src_index, unsigned dst_index,
                          ApEvent precondition, PredEvent pred_guard,
                          std::set<RtEvent> &map_applied,
                          const PhysicalTraceInfo &trace_info);
      ApEvent reduce_across(const RegionRequirement &src_req,
                            const RegionRequirement &dst_req,
                            const InstanceSet &src_targets,
                            const InstanceSet &dst_targets,
                            Operation *op, ApEvent precondition,
                            PredEvent predication_guard,
                            const PhysicalTraceInfo &trace_info);
    public:
      int physical_convert_mapping(Operation *op,
                               const RegionRequirement &req,
                               const std::vector<MappingInstance> &chosen,
                               InstanceSet &result, RegionTreeID &bad_tree,
                               std::vector<FieldID> &missing_fields,
                               std::map<PhysicalManager*,
                                    std::pair<unsigned,bool> > *acquired,
                               std::vector<PhysicalManager*> &unacquired,
                               const bool do_acquire_checks);
      bool physical_convert_postmapping(Operation *op,
                               const RegionRequirement &req,
                               const std::vector<MappingInstance> &chosen,
                               InstanceSet &result, RegionTreeID &bad_tree,
                               std::map<PhysicalManager*,
                                    std::pair<unsigned,bool> > *acquired,
                               std::vector<PhysicalManager*> &unacquired,
                               const bool do_acquire_checks);
      void log_mapping_decision(UniqueID uid, unsigned index,
                                const RegionRequirement &req,
                                const InstanceSet &targets,
                                bool postmapping = false);
    public: // helper method for the above two methods
      void perform_missing_acquires(Operation *op,
                 std::map<PhysicalManager*,std::pair<unsigned,bool> > &acquired,
                               const std::vector<PhysicalManager*> &unacquired);
    public:
      bool are_colocated(const std::vector<InstanceSet*> &instances,
                         FieldSpace handle, const std::set<FieldID> &fields,
                         unsigned &idx1, unsigned &idx2);
    public:
      // This takes ownership of the value buffer
      ApEvent fill_fields(Operation *op,
                          const RegionRequirement &req,
                          const unsigned index,
                          const void *value, size_t value_size,
                          VersionInfo &version_info,
                          RestrictInfo &restrict_info,
                          InstanceSet &instances, ApEvent precondition,
                          std::set<RtEvent> &map_applied_events,
                          PredEvent true_guard, PredEvent false_guard,
                          const PhysicalTraceInfo &trace_info);
      InstanceManager* create_external_instance(AttachOp *attach_op,
                                const RegionRequirement &req,
                                const std::vector<FieldID> &field_set);
      InstanceRef attach_external(AttachOp *attach_op, unsigned index,
                                  const RegionRequirement &req,
                                  InstanceManager *ext_instance,
                                  VersionInfo &version_info,
                                  std::set<RtEvent> &map_applied_events);
      ApEvent detach_external(const RegionRequirement &req, DetachOp *detach_op,
                              unsigned index, VersionInfo &version_info, 
                              const InstanceRef &ref, 
                              std::set<RtEvent> &map_applied_events);
    public:
      // Debugging method for checking context state
      void check_context_state(RegionTreeContext ctx);
    public:
      // We know the domain of the index space
      IndexSpaceNode* create_node(IndexSpace is, const void *realm_is, 
                                  IndexPartNode *par, LegionColor color,
                                  DistributedID did,
                                  ApEvent is_ready = ApEvent::NO_AP_EVENT,
                                  IndexSpaceExprID expr_id = 0,
                                  ShardMapping *shard_mapping = NULL);
      IndexSpaceNode* create_node(IndexSpace is, const void *realm_is, 
                                  IndexPartNode *par, LegionColor color,
                                  DistributedID did,
                                  ApUserEvent is_ready,
                                  ShardMapping *shard_mapping = NULL);
      // We know the disjointness of the index partition
      IndexPartNode*  create_node(IndexPartition p, IndexSpaceNode *par,
                                  IndexSpaceNode *color_space, 
                                  LegionColor color, bool disjoint,int complete,
                                  DistributedID did, ApEvent partition_ready, 
                                  ApBarrier partial_pending,
                                  ShardMapping *shard_mapping = NULL);
      // Give the event for when the disjointness information is ready
      IndexPartNode*  create_node(IndexPartition p, IndexSpaceNode *par,
                                  IndexSpaceNode *color_space,LegionColor color,
                                  RtEvent disjointness_ready_event,int complete,
                                  DistributedID did, ApEvent partition_ready, 
                                  ApBarrier partial_pending,
                                  ShardMapping *shard_mapping = NULL);
      FieldSpaceNode* create_node(FieldSpace space, DistributedID did,
                                  ShardMapping *shard_mapping = NULL);
      FieldSpaceNode* create_node(FieldSpace space, DistributedID did,
                                  Deserializer &derez);
      RegionNode*     create_node(LogicalRegion r, PartitionNode *par,
                                  ShardMapping *shard_mapping = NULL);
      PartitionNode*  create_node(LogicalPartition p, RegionNode *par);
    public:
      IndexSpaceNode* get_node(IndexSpace space);
      IndexPartNode*  get_node(IndexPartition part, RtEvent *defer = NULL);
      FieldSpaceNode* get_node(FieldSpace space);
      RegionNode*     get_node(LogicalRegion handle, bool need_check = true);
      PartitionNode*  get_node(LogicalPartition handle, bool need_check = true);
      RegionNode*     get_tree(RegionTreeID tid);
      // Request but don't block
      RtEvent request_node(IndexSpace space);
      // Find a local node if it exists and return it with reference
      // otherwise return NULL
      RegionNode*     find_local_node(LogicalRegion handle);
      PartitionNode*  find_local_node(LogicalPartition handle);
    public:
      bool has_node(IndexSpace space);
      bool has_node(IndexPartition part);
      bool has_node(FieldSpace space);
      bool has_node(LogicalRegion handle);
      bool has_node(LogicalPartition handle);
      bool has_tree(RegionTreeID tid);
      bool has_field(FieldSpace space, FieldID fid);
    public:
      void remove_node(IndexSpace space);
      void remove_node(IndexPartition part);
      void remove_node(FieldSpace space);
      void remove_node(LogicalRegion handle, bool top);
      void remove_node(LogicalPartition handle);
    public:
      bool is_top_level_index_space(IndexSpace handle);
      bool is_top_level_region(LogicalRegion handle);
    public:
      bool is_subregion(LogicalRegion child, LogicalRegion parent);
      bool is_subregion(LogicalRegion child, LogicalPartition parent);
      bool is_disjoint(IndexPartition handle);
      bool is_disjoint(LogicalPartition handle);
    public:
      bool are_disjoint(IndexSpace one, IndexSpace two);
      bool are_disjoint(IndexSpace one, IndexPartition two);
      bool are_disjoint(IndexPartition one, IndexPartition two); 
      // Can only use the region tree for proving disjointness here
      bool are_disjoint_tree_only(IndexTreeNode *one, IndexTreeNode *two,
                                  IndexTreeNode *&common_ancestor);
    public:
      bool are_compatible(IndexSpace left, IndexSpace right);
      bool is_dominated(IndexSpace src, IndexSpace dst);
    public:
      bool compute_index_path(IndexSpace parent, IndexSpace child,
                              std::vector<LegionColor> &path);
      bool compute_partition_path(IndexSpace parent, IndexPartition child,
                                  std::vector<LegionColor> &path); 
    public:
      void initialize_path(IndexSpace child, IndexSpace parent,
                           RegionTreePath &path);
      void initialize_path(IndexPartition child, IndexSpace parent,
                           RegionTreePath &path);
      void initialize_path(IndexSpace child, IndexPartition parent,
                           RegionTreePath &path);
      void initialize_path(IndexPartition child, IndexPartition parent,
                           RegionTreePath &path);
      void initialize_path(IndexTreeNode* child, IndexTreeNode *parent,
                           RegionTreePath &path);
#ifdef DEBUG_LEGION
    public:
      unsigned get_projection_depth(LogicalRegion result, LogicalRegion upper);
      unsigned get_projection_depth(LogicalRegion result, 
                                    LogicalPartition upper);
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
      bool retrieve_semantic_information(IndexSpace handle, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
      bool retrieve_semantic_information(IndexPartition handle, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
      bool retrieve_semantic_information(FieldSpace handle, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
      bool retrieve_semantic_information(FieldSpace handle, FieldID fid,
                                         SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
      bool retrieve_semantic_information(LogicalRegion handle, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
      bool retrieve_semantic_information(LogicalPartition part, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
    public:
      // These three methods a something pretty awesome and crazy
      // We want to do common sub-expression elimination on index space
      // unions, intersections, and difference operations to avoid repeating
      // expensive Realm dependent partition calls where possible, by 
      // running everything through this interface we first check to see
      // if these operations have been requested before and if so will 
      // return the common sub-expression, if not we will actually do 
      // the computation and memoize it for the future
      IndexSpaceExpression* union_index_spaces(IndexSpaceExpression *lhs,
                                               IndexSpaceExpression *rhs);
      IndexSpaceExpression* union_index_spaces(
                                 const std::set<IndexSpaceExpression*> &exprs);
    protected:
      // Internal version
      IndexSpaceExpression* union_index_spaces(
                               const std::vector<IndexSpaceExpression*> &exprs);
    public:
      IndexSpaceExpression* intersect_index_spaces(
                                               IndexSpaceExpression *lhs,
                                               IndexSpaceExpression *rhs);
      IndexSpaceExpression* intersect_index_spaces(
                                 const std::set<IndexSpaceExpression*> &exprs);
    protected:
      IndexSpaceExpression* intersect_index_spaces(
                               const std::vector<IndexSpaceExpression*> &exprs);
    public:
      IndexSpaceExpression* subtract_index_spaces(IndexSpaceExpression *lhs, 
                                                  IndexSpaceExpression *rhs);
    public:
      // Methods for removing index space expression when they are done
      void invalidate_index_space_expression(
                            const std::vector<IndexSpaceOperation*> &parents);
      void remove_union_operation(IndexSpaceOperation *expr, 
                            const std::vector<IndexSpaceExpression*> &exprs);
      void remove_intersection_operation(IndexSpaceOperation *expr, 
                            const std::vector<IndexSpaceExpression*> &exprs);
      void remove_subtraction_operation(IndexSpaceOperation *expr,
                       IndexSpaceExpression *lhs, IndexSpaceExpression *rhs);
    public:
      // Remote expression methods
      IndexSpaceExpression* find_or_create_remote_expression(
              IndexSpaceExprID remote_expr_id, Deserializer &derez);
      IndexSpaceExpression* find_remote_expression(
              IndexSpaceExprID remote_expr_id);
      void unregister_remote_expression(IndexSpaceExprID remote_expr_id);
    public:
      Runtime *const runtime;
    protected:
      mutable LocalLock lookup_lock;
      mutable LocalLock lookup_is_op_lock;
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
      // pending events for requested nodes
      std::map<IndexSpace,RtEvent>       index_space_requests;
      std::map<IndexPartition,RtEvent>    index_part_requests;
      std::map<FieldSpace,RtEvent>       field_space_requests;
      std::map<RegionTreeID,RtEvent>     region_tree_requests;
    private:
      // Index space operations
      std::map<IndexSpaceExprID/*first*/,ExpressionTrieNode*> union_ops;
      std::map<IndexSpaceExprID/*first*/,ExpressionTrieNode*> intersection_ops;
      std::map<IndexSpaceExprID/*lhs*/,ExpressionTrieNode*> difference_ops;
      // Remote expressions
      std::map<IndexSpaceExprID,IndexSpaceExpression*> remote_expressions;
    };

    /**
     * \class IndexSpaceExpression
     * An IndexSpaceExpression represents a set computation
     * one on or more index spaces. IndexSpaceExpressions
     * currently are either IndexSpaceNodes at the leaves
     * or have intermeidate set operations that are either
     * set union, intersection, or difference.
     */
    class IndexSpaceExpression {
    public:
      struct TightenIndexSpaceArgs : public LgTaskArgs<TightenIndexSpaceArgs> {
      public:
        static const LgTaskID TASK_ID = 
          LG_TIGHTEN_INDEX_SPACE_TASK_ID;
      public:
        TightenIndexSpaceArgs(IndexSpaceExpression *proxy)
          : LgTaskArgs<TightenIndexSpaceArgs>(implicit_provenance),
            proxy_this(proxy) { proxy->add_expression_reference(); }
      public:
        IndexSpaceExpression *const proxy_this;
      };
    public:
      IndexSpaceExpression(LocalLock &lock);
      IndexSpaceExpression(TypeTag tag, Runtime *runtime, LocalLock &lock); 
      IndexSpaceExpression(TypeTag tag, IndexSpaceExprID id, LocalLock &lock);
      virtual ~IndexSpaceExpression(void);
    public:
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag, 
                                           bool need_tight_result) = 0;
      virtual void tighten_index_space(void) = 0;
      virtual bool check_empty(void) = 0;
      virtual void pack_expression(Serializer &rez, AddressSpaceID target) = 0;
      // Should only be called on inherited types
      virtual void record_remote_instance(AddressSpaceID target) 
        { assert(false); }
      virtual void add_expression_reference(void) = 0;
      virtual bool remove_expression_reference(void) = 0;
      virtual bool test_intersection_nonblocking(IndexSpaceExpression *expr,
         RegionTreeForest *context, ApEvent &precondition, bool second = false);
      virtual IndexSpaceNode* find_or_create_node(InnerContext *ctx,
                                                  ShardMapping *mapping) = 0;
    public:
      static void handle_tighten_index_space(const void *args);
    public:
      void add_parent_operation(IndexSpaceOperation *op);
      void remove_parent_operation(IndexSpaceOperation *op);
    public:
      inline bool is_empty(void)
      {
        if (!has_empty)
        {
          empty = check_empty();
          __sync_synchronize();
          has_empty = true;
        }
        return empty;
      }
    public:
      static IndexSpaceExpression* unpack_expression(Deserializer &derez,
                         RegionTreeForest *forest, AddressSpaceID source);
    public:
      const TypeTag type_tag;
      const IndexSpaceExprID expr_id;
    private:
      LocalLock &expr_lock;
    protected:
      std::set<IndexSpaceOperation*> parent_operations;
      bool empty, has_empty;
    };

    // Shared functionality between index space operation and
    // remote expressions
    class IntermediateExpression : 
      public IndexSpaceExpression, public Collectable {
    public:
      struct DestructionFunctor {
      public:
        DestructionFunctor(Runtime *rt, Serializer &r)
          : runtime(rt), rez(r) { }
      public:
        void apply(AddressSpaceID target);
      public:
        Runtime *const runtime;
        Serializer &rez;
      };
    public:
      IntermediateExpression(TypeTag tag, RegionTreeForest *ctx);
      IntermediateExpression(TypeTag tag, RegionTreeForest *ctx, 
                             IndexSpaceExprID expr_id);
      virtual ~IntermediateExpression(void);
    public:
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag, 
                                           bool need_tight_result) = 0;
      virtual void tighten_index_space(void) = 0;
      virtual bool check_empty(void) = 0;
      virtual void pack_expression(Serializer &rez, AddressSpaceID target) = 0; 
      virtual void record_remote_instance(AddressSpaceID target);
      virtual void add_expression_reference(void);
      virtual bool remove_expression_reference(void);
      virtual IndexSpaceNode* find_or_create_node(InnerContext *ctx,
                                                  ShardMapping *mapping) = 0;
    public:
      static void handle_expression_invalidation(Deserializer &derez,
                                                 RegionTreeForest *forest);
    public:
      RegionTreeForest *const context;
    protected:
      mutable LocalLock inter_lock;
    protected:
      // An equivalent index space node with the same IndexSpaceExprID as this
      // We only make this if we actually need to since IndexSpaceNodes
      // are more expensive to maintain
      IndexSpaceNode *node;
    private:
      NodeSet remote_instances;
    };

    class IndexSpaceOperation : public IntermediateExpression {
    public:
      enum OperationKind {
        UNION_OP_KIND,
        INTERSECT_OP_KIND,
        DIFFERENCE_OP_KIND,
      };
    public:
      IndexSpaceOperation(TypeTag tag, OperationKind kind,
                          RegionTreeForest *ctx);
      virtual ~IndexSpaceOperation(void);
    public:
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag, 
                                           bool need_tight_result) = 0;
      virtual void tighten_index_space(void) = 0;
      virtual bool check_empty(void) = 0;
      virtual void pack_expression(Serializer &rez, AddressSpaceID target) = 0; 
      virtual bool remove_operation(RegionTreeForest *forest) = 0;
      virtual bool remove_expression_reference(void);
      virtual IndexSpaceNode* find_or_create_node(InnerContext *ctx,
                                                  ShardMapping *mapping) = 0;
    public:
      void invalidate_operation(std::deque<IndexSpaceOperation*> &to_remove);
    public:
      const OperationKind op_kind;
    private:
      int invalidated;
    };

    template<int DIM, typename T>
    class IndexSpaceOperationT : public IndexSpaceOperation {
    public:
      IndexSpaceOperationT(OperationKind kind, RegionTreeForest *ctx);
      virtual ~IndexSpaceOperationT(void);
    public:
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag,
                                           bool need_tight_result);
      virtual void tighten_index_space(void);
      virtual bool check_empty(void);
      virtual void pack_expression(Serializer &rez, AddressSpaceID target);
      virtual bool remove_operation(RegionTreeForest *forest) = 0;
      virtual IndexSpaceNode* find_or_create_node(InnerContext *ctx,
                                                  ShardMapping *mapping);
    public:
      ApEvent get_realm_index_space(Realm::IndexSpace<DIM,T> &space,
                                    bool need_tight_result);
    protected:
      Realm::IndexSpace<DIM,T> realm_index_space, tight_index_space;
      ApEvent realm_index_space_ready; 
      RtEvent tight_index_space_ready;
      bool is_index_space_tight;
    };

    template<int DIM, typename T>
    class IndexSpaceUnion : public IndexSpaceOperationT<DIM,T> {
    public:
      IndexSpaceUnion(const std::vector<IndexSpaceExpression*> &to_union,
                      RegionTreeForest *context);
      IndexSpaceUnion(const IndexSpaceUnion<DIM,T> &rhs);
      virtual ~IndexSpaceUnion(void);
    public:
      IndexSpaceUnion& operator=(const IndexSpaceUnion &rhs);
    public:
      virtual bool remove_operation(RegionTreeForest *forest);
    protected:
      const std::vector<IndexSpaceExpression*> sub_expressions;
    };

    class OperationCreator {
    public:
      virtual ~OperationCreator(void) { }
    public:
      virtual IndexSpaceOperation* create(void) = 0;
    };

    class UnionOpCreator : public OperationCreator {
    public:
      UnionOpCreator(RegionTreeForest *f, TypeTag t,
                     const std::vector<IndexSpaceExpression*> &e)
        : forest(f), type_tag(t), exprs(e) { }
    public:
      virtual IndexSpaceOperation* create(void)
      {
        NT_TemplateHelper::demux<UnionOpCreator>(type_tag, this);
        return result;
      }
    public:
      template<typename N, typename T>
      static inline void demux(UnionOpCreator *creator)
      {
        creator->result = new IndexSpaceUnion<N::N,T>(creator->exprs,
            creator->forest);
      }
    public:
      RegionTreeForest *const forest;
      const TypeTag type_tag;
      const std::vector<IndexSpaceExpression*> &exprs;
      IndexSpaceOperation *result;
    };

    template<int DIM, typename T>
    class IndexSpaceIntersection : public IndexSpaceOperationT<DIM,T> {
    public:
      IndexSpaceIntersection(const std::vector<IndexSpaceExpression*> &to_inter,
                             RegionTreeForest *context);
      IndexSpaceIntersection(const IndexSpaceIntersection &rhs);
      virtual ~IndexSpaceIntersection(void);
    public:
      IndexSpaceIntersection& operator=(const IndexSpaceIntersection &rhs);
    public:
      virtual bool remove_operation(RegionTreeForest *forest);
    protected:
      const std::vector<IndexSpaceExpression*> sub_expressions;
    };

    class IntersectionOpCreator : public OperationCreator {
    public:
      IntersectionOpCreator(RegionTreeForest *f, TypeTag t,
                            const std::vector<IndexSpaceExpression*> &e)
        : forest(f), type_tag(t), exprs(e) { }
    public:
      virtual IndexSpaceOperation* create(void)
      {
        NT_TemplateHelper::demux<IntersectionOpCreator>(type_tag, this);
        return result;
      }
    public:
      template<typename N, typename T>
      static inline void demux(IntersectionOpCreator *creator)
      {
        creator->result = new IndexSpaceIntersection<N::N,T>(creator->exprs,
            creator->forest);
      }
    public:
      RegionTreeForest *const forest;
      const TypeTag type_tag;
      const std::vector<IndexSpaceExpression*> &exprs;
      IndexSpaceOperation *result;
    };

    template<int DIM, typename T>
    class IndexSpaceDifference : public IndexSpaceOperationT<DIM,T> {
    public:
      IndexSpaceDifference(IndexSpaceExpression *lhs,IndexSpaceExpression *rhs,
                           RegionTreeForest *context);
      IndexSpaceDifference(const IndexSpaceDifference &rhs);
      virtual ~IndexSpaceDifference(void);
    public:
      IndexSpaceDifference& operator=(const IndexSpaceDifference &rhs);
    public:
      virtual bool remove_operation(RegionTreeForest *forest);
    protected:
      IndexSpaceExpression *const lhs;
      IndexSpaceExpression *const rhs;
    };

    class DifferenceOpCreator : public OperationCreator {
    public:
      DifferenceOpCreator(RegionTreeForest *f, TypeTag t,
                          IndexSpaceExpression *l, IndexSpaceExpression *r)
        : forest(f), type_tag(t), lhs(l), rhs(r) { }
    public:
      virtual IndexSpaceOperation* create(void)
      {
        NT_TemplateHelper::demux<DifferenceOpCreator>(type_tag, this);
        return result;
      }
    public:
      template<typename N, typename T>
      static inline void demux(DifferenceOpCreator *creator)
      {
        creator->result = new IndexSpaceDifference<N::N,T>(creator->lhs,
            creator->rhs, creator->forest);
      }
    public:
      RegionTreeForest *const forest;
      const TypeTag type_tag;
      IndexSpaceExpression *const lhs;
      IndexSpaceExpression *const rhs;
      IndexSpaceOperation *result;
    };

    template<int DIM, typename T>
    class RemoteExpression : public IntermediateExpression {
    public:
      RemoteExpression(Deserializer &derez, RegionTreeForest *ctx,
                       IndexSpaceExprID id);
      RemoteExpression(const RemoteExpression &rhs);
      virtual ~RemoteExpression(void);
    public:
      RemoteExpression& operator=(const RemoteExpression &rhs);
    public:
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag, 
                                           bool need_tight_result);
      virtual void tighten_index_space(void);
      virtual bool check_empty(void);
      virtual void pack_expression(Serializer &rez, AddressSpaceID target);
      virtual IndexSpaceNode* find_or_create_node(InnerContext *ctx,
                                                  ShardMapping *mapping);
    public:
      Realm::IndexSpace<DIM,T> realm_index_space;
      ApEvent realm_index_space_ready;
    };

    struct RemoteExpressionCreator {
    public:
      RemoteExpressionCreator(Deserializer &d, RegionTreeForest *ctx,
                              IndexSpaceExprID id)
        : derez(d), context(ctx), remote_expr_id(id) { }
    public:
      template<typename N, typename T>
      static inline void demux(RemoteExpressionCreator *creator)
      {
        creator->result = new RemoteExpression<N::N,T>(creator->derez,
            creator->context, creator->remote_expr_id);
      }
    public:
      Deserializer &derez;
      RegionTreeForest *const context;
      const IndexSpaceExprID remote_expr_id;
      IndexSpaceExpression *result;
    };

    /**
     * \class PendingIndexSpaceExpression
     * This class is a thunk that is used to represent index space
     * expressions that haven't been computed yet. It allows for 
     * partial specification of an upper bound expression that
     * allows us to do partial intersection testing.
     */
    class PendingIndexSpaceExpression : public IntermediateExpression {
    public:
      struct PendingIntersectionTest {
      public:
        PendingIntersectionTest(void) : other(NULL), 
          original_precondition(ApEvent::NO_AP_EVENT), second(false) { }
        PendingIntersectionTest(IndexSpaceExpression *o, ApEvent e, bool s)
          : other(o), original_precondition(e), second(s) { }
      public:
        inline bool operator<(const PendingIntersectionTest &rhs) const
        {
          if (other->expr_id < rhs.other->expr_id)
            return true;
          if (other->expr_id > rhs.other->expr_id)
            return false;
          if (original_precondition.id < rhs.original_precondition.id)
            return true;
          if (original_precondition.id > rhs.original_precondition.id)
            return false;
          return second < rhs.second;
        }
        inline bool operator==(const PendingIntersectionTest &rhs) const
        {
          if (other->expr_id != rhs.other->expr_id)
            return false;
          if (original_precondition.id != rhs.original_precondition.id)
            return false;
          return second == rhs.second;
        }
      public:
        IndexSpaceExpression *other;
        ApEvent original_precondition;
        bool second;
      };
    public:
      PendingIndexSpaceExpression(IndexSpaceExpression *upper_bound, 
                                  RegionTreeForest *ctx); 
      PendingIndexSpaceExpression(const PendingIndexSpaceExpression &rhs);
      virtual ~PendingIndexSpaceExpression(void);
    public:
      PendingIndexSpaceExpression& operator=(
                                  const PendingIndexSpaceExpression &rhs);
    public:
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag, 
                                           bool need_tight_result);
      virtual void tighten_index_space(void);
      virtual bool check_empty(void);
      virtual void pack_expression(Serializer &rez, AddressSpaceID target);
      virtual bool test_intersection_nonblocking(IndexSpaceExpression *expr,
         RegionTreeForest *context, ApEvent &precondition, bool second = false);
      virtual IndexSpaceNode* find_or_create_node(InnerContext *ctx,
                                                  ShardMapping *mapping);
    public:
      void set_result(IndexSpaceExpression *result);
      // This can be racy but in a good way
      inline IndexSpaceExpression* get_ready_expression(void)
      {
        if (result != NULL)
          return result;
        else
          return this;
      }
    public:
      IndexSpaceExpression *const upper_bound;
      RegionTreeForest *const context;
    protected:
      RtUserEvent ready_event;
      IndexSpaceExpression *result;
      // Pending intersection tests
      std::map<PendingIntersectionTest,ApUserEvent> intersection_tests;
    };

    /**
     * \class ExpressionTrieNode
     * This is a class for constructing a trie for index space
     * expressions so we can quickly detect commmon subexpression
     * in O(log N)^M time where N is the number of expressions
     * in total and M is the number of expression in the operation
     */
    class ExpressionTrieNode {
    public:
      ExpressionTrieNode(unsigned depth, IndexSpaceExprID expr_id, 
                         IndexSpaceOperation *op = NULL);
      ExpressionTrieNode(const ExpressionTrieNode &rhs);
      ~ExpressionTrieNode(void);
    public:
      ExpressionTrieNode& operator=(const ExpressionTrieNode &rhs);
    public:
      bool find_operation(
          const std::vector<IndexSpaceExpression*> &expressions,
          IndexSpaceOperation *&result, ExpressionTrieNode *&last);
      IndexSpaceOperation* find_or_create_operation( 
          const std::vector<IndexSpaceExpression*> &expressions,
          OperationCreator &creator);
      bool remove_operation(const std::vector<IndexSpaceExpression*> &exprs);
    public:
      const unsigned depth;
      const IndexSpaceExprID expr;
    protected:
      IndexSpaceOperation *local_operation;
      std::map<IndexSpaceExprID,IndexSpaceOperation*> operations;
      std::map<IndexSpaceExprID,ExpressionTrieNode*> nodes;
    protected:
      mutable LocalLock trie_lock;
    };

    /**
     * \class IndexTreeNode
     * The abstract base class for nodes in the index space trees.
     */
    class IndexTreeNode : public DistributedCollectable {
    public:
      IndexTreeNode(RegionTreeForest *ctx, unsigned depth,
                    LegionColor color, DistributedID did,
                    AddressSpaceID owner); 
      virtual ~IndexTreeNode(void);
    public:
      virtual void notify_active(ReferenceMutator *mutator) { }
      virtual void notify_inactive(ReferenceMutator *mutator) { }
      virtual void notify_valid(ReferenceMutator *mutator) = 0;
      virtual void notify_invalid(ReferenceMutator *mutator) = 0;
    public:
      virtual IndexTreeNode* get_parent(void) const = 0;
      virtual void get_colors(std::vector<LegionColor> &colors) = 0;
      virtual void send_node(AddressSpaceID target, bool up) = 0;
    public:
      virtual bool is_index_space_node(void) const = 0;
#ifdef DEBUG_LEGION
      virtual IndexSpaceNode* as_index_space_node(void) = 0;
      virtual IndexPartNode* as_index_part_node(void) = 0;
#else
      inline IndexSpaceNode* as_index_space_node(void);
      inline IndexPartNode* as_index_part_node(void);
#endif
      virtual AddressSpaceID get_owner_space(void) const = 0;
    public:
      void attach_semantic_information(SemanticTag tag, AddressSpaceID source,
                             const void *buffer, size_t size, bool is_mutable);
      bool retrieve_semantic_information(SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
      virtual void send_semantic_request(AddressSpaceID target, 
        SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready) = 0;
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                        const void *buffer, size_t size, bool is_mutable,
                        RtUserEvent ready = RtUserEvent::NO_RT_USER_EVENT) = 0;
    public:
      void update_creation_set(const ShardMapping &mapping);
    public:
      RegionTreeForest *const context;
      const unsigned depth;
      const LegionColor color;
    public:
      NodeSet child_creation;
      bool destroyed;
    protected:
      LocalLock &node_lock;
    protected:
      std::map<IndexTreeNode*,bool> dominators;
    protected:
      LegionMap<SemanticTag,SemanticInfo>::aligned semantic_info;
    protected:
      std::map<std::pair<LegionColor,LegionColor>,RtEvent> pending_tests;
    };

    /**
     * \class IndexSpaceNode
     * A class for representing a generic index space node.
     */
    class IndexSpaceNode : 
      public IndexTreeNode, public IndexSpaceExpression {
    public:
      struct DynamicIndependenceArgs : 
        public LgTaskArgs<DynamicIndependenceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_PART_INDEPENDENCE_TASK_ID;
      public:
        DynamicIndependenceArgs(IndexSpaceNode *par, 
                                IndexPartNode *l, IndexPartNode *r)
          : LgTaskArgs<DynamicIndependenceArgs>(implicit_provenance),
            parent(par), left(l), right(r) { }
      public:
        IndexSpaceNode *const parent;
        IndexPartNode *const left, *const right;
      };
      struct SemanticRequestArgs : public LgTaskArgs<SemanticRequestArgs> {
      public:
        static const LgTaskID TASK_ID = 
          LG_INDEX_SPACE_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        SemanticRequestArgs(IndexSpaceNode *proxy, 
                            SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticRequestArgs>(implicit_provenance),
            proxy_this(proxy), tag(t), source(src) { }
      public:
        IndexSpaceNode *const proxy_this;
        const SemanticTag tag;
        const AddressSpaceID source;
      };
      struct DeferChildArgs : public LgTaskArgs<DeferChildArgs> {
      public:
        static const LgTaskID TASK_ID = LG_INDEX_SPACE_DEFER_CHILD_TASK_ID;
      public:
        DeferChildArgs(IndexSpaceNode *proxy, LegionColor child, 
                       IndexPartNode *tar, RtUserEvent trig, AddressSpaceID src)
          : LgTaskArgs<DeferChildArgs>(implicit_provenance),
            proxy_this(proxy), child_color(child), target(tar), 
            to_trigger(trig), source(src) { }
      public:
        IndexSpaceNode *const proxy_this;
        const LegionColor child_color;
        IndexPartNode *const target;
        const RtUserEvent to_trigger;
        const AddressSpaceID source;
      };
      class IndexSpaceSetFunctor {
      public:
        IndexSpaceSetFunctor(Runtime *rt, AddressSpaceID src, Serializer &r)
          : runtime(rt), source(src), rez(r) { }
      public:
        void apply(AddressSpaceID target);
      public:
        Runtime *const runtime;
        const AddressSpaceID source;
        Serializer &rez;
      };
      class DestructionFunctor {
      public:
        DestructionFunctor(IndexSpaceNode *n, ReferenceMutator *m)
          : node(n), mutator(m) { }
      public:
        void apply(AddressSpaceID target);
      public:
        IndexSpaceNode *const node;
        ReferenceMutator *const mutator;
      };
    public:
      IndexSpaceNode(RegionTreeForest *ctx, IndexSpace handle,
                     IndexPartNode *parent, LegionColor color,
                     DistributedID did, ApEvent index_space_ready,
                     IndexSpaceExprID expr_id);
      IndexSpaceNode(const IndexSpaceNode &rhs);
      virtual ~IndexSpaceNode(void);
    public:
      IndexSpaceNode& operator=(const IndexSpaceNode &rhs);
    public:
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      virtual bool is_index_space_node(void) const;
#ifdef DEBUG_LEGION
      virtual IndexSpaceNode* as_index_space_node(void);
      virtual IndexPartNode* as_index_part_node(void);
#endif
      virtual AddressSpaceID get_owner_space(void) const;
      static AddressSpaceID get_owner_space(IndexSpace handle, Runtime *rt);
    public:
      virtual IndexTreeNode* get_parent(void) const;
      virtual void get_colors(std::vector<LegionColor> &colors);
    public:
      virtual void send_semantic_request(AddressSpaceID target, 
           SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready);
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                           const void *buffer, size_t size, bool is_mutable,
                           RtUserEvent ready = RtUserEvent::NO_RT_USER_EVENT);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source,
                            bool can_fail, bool wait_until, RtUserEvent ready);
      static void handle_semantic_request(RegionTreeForest *forest,
                                 Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(RegionTreeForest *forest,
                                 Deserializer &derez, AddressSpaceID source);
    public:
      bool has_color(const LegionColor c);
      IndexPartNode* get_child(const LegionColor c, 
                               RtEvent *defer = NULL, bool can_fail = false);
      void add_child(IndexPartNode *child);
      void remove_child(const LegionColor c);
      size_t get_num_children(void) const;
    public:
      bool are_disjoint(const LegionColor c1, const LegionColor c2); 
      void record_disjointness(bool disjoint, 
                               const LegionColor c1, const LegionColor c2);
      LegionColor generate_color(void);
      void record_remote_child(IndexPartition pid, LegionColor part_color);
    public:
      void add_instance(RegionNode *inst);
      bool has_instance(RegionTreeID tid);
      void remove_instance(RegionNode *inst);
    public:
      static void handle_disjointness_test(IndexSpaceNode *parent,
                                           IndexPartNode *left,
                                           IndexPartNode *right); 
    public:
      virtual void send_node(AddressSpaceID target, bool up);
      static void handle_node_creation(RegionTreeForest *context,
                                       Deserializer &derez, 
                                       AddressSpaceID source);
    public:
      static void handle_node_request(RegionTreeForest *context,
                                      Deserializer &derez,
                                      AddressSpaceID source);
      static void handle_node_return(Deserializer &derez);
      static void handle_node_child_request(RegionTreeForest *context,
                            Deserializer &derez, AddressSpaceID source);
      static void defer_node_child_request(const void *args);
      static void handle_node_child_response(Deserializer &derez);
      static void handle_colors_request(RegionTreeForest *context,
                            Deserializer &derez, AddressSpaceID source);
      static void handle_colors_response(Deserializer &derez);
      static void handle_index_space_set(RegionTreeForest *forest,
                           Deserializer &derez, AddressSpaceID source);
    public:
      // From IndexSpaceExpression
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag,
                                           bool need_tight_result) = 0;
      virtual void tighten_index_space(void) = 0;
      virtual bool check_empty(void) = 0;
      virtual void pack_expression(Serializer &rez, AddressSpaceID target) = 0;
      virtual void add_expression_reference(void);
      virtual bool remove_expression_reference(void);
      virtual IndexSpaceNode* find_or_create_node(InnerContext *ctx,
                                                  ShardMapping *mapping)
        { return this; }
      virtual void create_sharded_alias(IndexSpace alias, DistributedID did,
                                        ShardMapping *mapping) = 0;
    public:
      virtual void log_index_space_points(void) = 0;
      virtual ApEvent compute_pending_space(Operation *op,
            const std::vector<IndexSpace> &handles, bool is_union) = 0;
      virtual ApEvent compute_pending_space(Operation *op,
                              IndexPartition handle, bool is_union) = 0;
      virtual ApEvent compute_pending_difference(Operation *op, 
          IndexSpace initial, const std::vector<IndexSpace> &handles) = 0;
      virtual void get_index_space_domain(void *realm_is, TypeTag type_tag) = 0;
      virtual size_t get_volume(void) = 0;
      virtual size_t get_num_dims(void) const = 0;
      virtual bool contains_point(const void *realm_point,TypeTag type_tag) = 0;
      virtual bool destroy_node(AddressSpaceID source) = 0;
    public:
      virtual LegionColor get_max_linearized_color(void) = 0;
      virtual LegionColor linearize_color(const void *realm_color,
                                          TypeTag type_tag) = 0;
      virtual void delinearize_color(LegionColor color, 
                                     void *realm_color, TypeTag type_tag) = 0;
      virtual bool contains_color(LegionColor color, 
                                  bool report_error = false) = 0;
      virtual void instantiate_colors(std::vector<LegionColor> &colors) = 0;
      virtual Domain get_color_space_domain(void) = 0;
      virtual DomainPoint get_domain_point_color(void) const = 0;
      virtual DomainPoint delinearize_color_to_point(LegionColor c) = 0;
    public:
      bool intersects_with(IndexSpaceNode *rhs,bool compute = true);
      bool intersects_with(IndexPartNode *rhs, bool compute = true);
      bool dominates(IndexSpaceNode *rhs);
      bool dominates(IndexPartNode *rhs);
    public:
      virtual void pack_index_space(Serializer &rez, 
                                    bool include_size) const = 0;
      virtual void unpack_index_space(Deserializer &derez,
                                      AddressSpaceID source) = 0;
    public:
      virtual ApEvent create_equal_children(Operation *op,
                                            IndexPartNode *partition, 
                                            size_t granularity) = 0;
      virtual ApEvent create_equal_children(Operation *op,
                                            IndexPartNode *partition, 
                                            size_t granularity,
                                            ShardID shard,
                                            size_t total_shards) = 0;
      virtual ApEvent create_by_union(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *left,
                                      IndexPartNode *right) = 0;
      virtual ApEvent create_by_union(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *left,
                                      IndexPartNode *right,
                                      ShardID shard,
                                      size_t total_shards) = 0;
      virtual ApEvent create_by_intersection(Operation *op,
                                             IndexPartNode *partition,
                                             IndexPartNode *left,
                                             IndexPartNode *right) = 0;
      virtual ApEvent create_by_intersection(Operation *op,
                                             IndexPartNode *partition,
                                             IndexPartNode *left,
                                             IndexPartNode *right,
                                             ShardID shard,
                                             size_t total_shards) = 0;
      virtual ApEvent create_by_intersection(Operation *op,
                                             IndexPartNode *partition,
                                             // Left is implicit "this"
                                             IndexPartNode *right) = 0;
      virtual ApEvent create_by_difference(Operation *op,
                                           IndexPartNode *partition,
                                           IndexPartNode *left,
                                           IndexPartNode *right) = 0;
      virtual ApEvent create_by_difference(Operation *op,
                                           IndexPartNode *partition,
                                           IndexPartNode *left,
                                           IndexPartNode *right,
                                           ShardID shard,
                                           size_t total_shards) = 0;
      // Called on color space and not parent
      virtual ApEvent create_by_restriction(IndexPartNode *partition,
                                            const void *transform,
                                            const void *extent,
                                            int partition_dim,
                                            ShardID shard,
                                            size_t total_shards) = 0;
      virtual ApEvent create_by_field(Operation *op,
                                      IndexPartNode *partition,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready) = 0;
      virtual ApEvent create_by_image(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready,
                                      ShardID shard,
                                      size_t total_shards) = 0;
      virtual ApEvent create_by_image_range(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready,
                                      ShardID shard,
                                      size_t total_shards) = 0;
      virtual ApEvent create_by_preimage(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready) = 0;
      virtual ApEvent create_by_preimage_range(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready) = 0;
      virtual ApEvent create_association(Operation *op,
                                      IndexSpaceNode *range,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready) = 0;
      virtual bool check_field_size(size_t field_size, bool range) = 0;
    public:
      virtual ApEvent issue_copy(const PhysicalTraceInfo *info,RegionNode *node,
              const std::vector<CopySrcDstField> &src_fields,
              const std::vector<CopySrcDstField> &dst_fields,
              ApEvent precondition, PredEvent predicate_guard,
              IndexTreeNode *intersect, IndexSpaceExpression *mask,
              ReductionOpID redop = 0, bool reduction_fold = true,
              WriteSet *perf = NULL,const FieldMask *performed_mask = NULL) = 0;
      virtual ApEvent issue_fill(const PhysicalTraceInfo *info,RegionNode *node,
              const std::vector<CopySrcDstField> &dst_fields,
              const void *fill_value, size_t fill_size,
              ApEvent precondition, PredEvent predicate_guard,
#ifdef LEGION_SPY
              UniqueID fill_uid,
#endif
              IndexTreeNode *intersect = NULL,
              IndexSpaceExpression *mask = NULL,
              WriteSet *perf = NULL,const FieldMask *performed_mask = NULL) = 0;
    public:
      virtual Realm::InstanceLayoutGeneric* create_layout(
                           const Realm::InstanceLayoutConstraints &ilc,
                           const OrderingConstraint &constraint) = 0;
      virtual PhysicalInstance create_file_instance(const char *file_name,
				   const std::vector<Realm::FieldID> &field_ids,
                                   const std::vector<size_t> &field_sizes,
                                   legion_file_mode_t file_mode,
                                   ApEvent &ready_event) = 0;
      virtual PhysicalInstance create_hdf5_instance(const char *file_name,
                                   const std::vector<Realm::FieldID> &field_ids,
                                   const std::vector<size_t> &field_sizes,
                                   const std::vector<const char*> &field_files,
                                   bool read_only, ApEvent &ready_event) = 0;
      virtual PhysicalInstance create_external_instance(Memory memory,
                             uintptr_t base, Realm::InstanceLayoutGeneric *ilg,
                             ApEvent &ready_event) = 0;
    public:
      virtual void get_launch_space_domain(Domain &launch_domain) = 0;
      virtual void validate_slicing(const std::vector<IndexSpace> &slice_spaces,
                                    MultiTask *task, MapperManager *mapper) = 0;
      virtual void log_launch_space(UniqueID op_id) = 0;
      virtual IndexSpace create_shard_space(ShardingFunction *func, 
                                            ShardID shard,
                                            IndexSpace shard_space) = 0;
      virtual void destroy_shard_domain(const Domain &domain) = 0;
    public:
      const IndexSpace handle;
      IndexPartNode *const parent;
      const ApEvent index_space_ready;
    protected:
      // On the owner node track when the index space is set
      RtUserEvent               realm_index_space_set;
      // Keep track of whether we've tightened these bounds
      RtUserEvent               tight_index_space_set;
      bool                      tight_index_space;
      // Must hold the node lock when accessing the
      // remaining data structures
      std::map<LegionColor,IndexPartNode*> color_map;
      std::map<LegionColor,IndexPartition> remote_colors;
      std::set<RegionNode*> logical_nodes;
      std::set<std::pair<LegionColor,LegionColor> > disjoint_subsets;
      std::set<std::pair<LegionColor,LegionColor> > aliased_subsets;
    };

    /**
     * \class IndexSpaceNodeT
     * A templated class for handling any templated realm calls
     * associated with realm index spaces
     */
    template<int DIM, typename T>
    class IndexSpaceNodeT : public IndexSpaceNode,
                            public LegionHeapify<IndexSpaceNodeT<DIM,T> > {
    public:
      IndexSpaceNodeT(RegionTreeForest *ctx, IndexSpace handle,
                      IndexPartNode *parent, LegionColor color, 
                      const Realm::IndexSpace<DIM,T> *realm_is,
                      DistributedID did, ApEvent ready_event,
                      IndexSpaceExprID expr_id);
      IndexSpaceNodeT(const IndexSpaceNodeT &rhs);
      virtual ~IndexSpaceNodeT(void);
    public:
      IndexSpaceNodeT& operator=(const IndexSpaceNodeT &rhs);
    public:
      inline ApEvent get_realm_index_space(Realm::IndexSpace<DIM,T> &result,
                                           bool need_tight_result);
      inline void set_realm_index_space(AddressSpaceID source,
                                        const Realm::IndexSpace<DIM,T> &value);
    public:
      // From IndexSpaceExpression
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag,
                                           bool need_tight_result);
      virtual void tighten_index_space(void);
      virtual bool check_empty(void);
      virtual void pack_expression(Serializer &rez, AddressSpaceID target);
      virtual void create_sharded_alias(IndexSpace alias, DistributedID did,
                                        ShardMapping *mapping);
    public:
      virtual void log_index_space_points(void);
      void log_index_space_points(const Realm::IndexSpace<DIM,T> &space) const;
    public:
      virtual ApEvent compute_pending_space(Operation *op,
            const std::vector<IndexSpace> &handles, bool is_union);
      virtual ApEvent compute_pending_space(Operation *op,
                             IndexPartition handle, bool is_union);
      virtual ApEvent compute_pending_difference(Operation *op,
          IndexSpace initial, const std::vector<IndexSpace> &handles);
      virtual void get_index_space_domain(void *realm_is, TypeTag type_tag);
      virtual size_t get_volume(void);
      virtual size_t get_num_dims(void) const;
      virtual bool contains_point(const void *realm_point, TypeTag type_tag);
      virtual bool destroy_node(AddressSpaceID source);
    public:
      virtual LegionColor get_max_linearized_color(void);
      virtual LegionColor linearize_color(const void *realm_color,
                                          TypeTag type_tag);
      virtual void delinearize_color(LegionColor color, 
                                     void *realm_color, TypeTag type_tag);
      virtual bool contains_color(LegionColor color,
                                  bool report_error = false);
      virtual void instantiate_colors(std::vector<LegionColor> &colors);
      virtual Domain get_color_space_domain(void);
      virtual DomainPoint get_domain_point_color(void) const;
      virtual DomainPoint delinearize_color_to_point(LegionColor c);
    public:
      virtual void pack_index_space(Serializer &rez, bool include_size) const;
      virtual void unpack_index_space(Deserializer &derez,
                                      AddressSpaceID source);
    public:
      virtual ApEvent create_equal_children(Operation *op,
                                            IndexPartNode *partition, 
                                            size_t granularity);
      virtual ApEvent create_equal_children(Operation *op,
                                            IndexPartNode *partition, 
                                            size_t granularity,
                                            ShardID shard,
                                            size_t total_shards);
      virtual ApEvent create_by_union(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *left,
                                      IndexPartNode *right);
      virtual ApEvent create_by_union(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *left,
                                      IndexPartNode *right,
                                      ShardID shard, 
                                      size_t total_shards);
      virtual ApEvent create_by_intersection(Operation *op,
                                             IndexPartNode *partition,
                                             IndexPartNode *left,
                                             IndexPartNode *right);
      virtual ApEvent create_by_intersection(Operation *op,
                                             IndexPartNode *partition,
                                             IndexPartNode *left,
                                             IndexPartNode *right,
                                             ShardID shard,
                                             size_t total_shards);
      virtual ApEvent create_by_intersection(Operation *op,
                                             IndexPartNode *partition,
                                             // Left is implicit "this"
                                             IndexPartNode *right);
      virtual ApEvent create_by_difference(Operation *op,
                                           IndexPartNode *partition,
                                           IndexPartNode *left,
                                           IndexPartNode *right);
      virtual ApEvent create_by_difference(Operation *op,
                                           IndexPartNode *partition,
                                           IndexPartNode *left,
                                           IndexPartNode *right,
                                           ShardID shard,
                                           size_t total_shards);
      // Called on color space and not parent
      virtual ApEvent create_by_restriction(IndexPartNode *partition,
                                            const void *transform,
                                            const void *extent,
                                            int partition_dim,
                                            ShardID shard,
                                            size_t total_shards);
      template<int N>
      ApEvent create_by_restriction_helper(IndexPartNode *partition,
                                   const Realm::Matrix<N,DIM,T> &transform,
                                   const Realm::Rect<N,T> &extent,
                                   ShardID shard, size_t total_shards);
      virtual ApEvent create_by_field(Operation *op,
                                      IndexPartNode *partition,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      template<int COLOR_DIM, typename COLOR_T>
      ApEvent create_by_field_helper(Operation *op,
                                     IndexPartNode *partition,
                const std::vector<FieldDataDescriptor> &instances,
                                     ApEvent instances_ready);
      virtual ApEvent create_by_image(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready,
                                      ShardID shard,
                                      size_t total_shards);
      template<int DIM2, typename T2>
      ApEvent create_by_image_helper(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready,
                                      ShardID shard,
                                      size_t total_shards);
      virtual ApEvent create_by_image_range(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready,
                                      ShardID shard,
                                      size_t total_shards);
      template<int DIM2, typename T2>
      ApEvent create_by_image_range_helper(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready,
                                      ShardID shard,
                                      size_t total_shards);
      virtual ApEvent create_by_preimage(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      template<int DIM2, typename T2>
      ApEvent create_by_preimage_helper(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      virtual ApEvent create_by_preimage_range(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      template<int DIM2, typename T2>
      ApEvent create_by_preimage_range_helper(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      virtual ApEvent create_association(Operation *op,
                                      IndexSpaceNode *range,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      template<int DIM2, typename T2>
      ApEvent create_association_helper(Operation *op,
                                      IndexSpaceNode *range,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      virtual bool check_field_size(size_t field_size, bool range);
    public:
      virtual ApEvent issue_copy(const PhysicalTraceInfo *info,RegionNode *node,
              const std::vector<CopySrcDstField> &src_fields,
              const std::vector<CopySrcDstField> &dst_fields,
              ApEvent precondition, PredEvent predicate_guard,
              IndexTreeNode *intersect, IndexSpaceExpression *mask,
              ReductionOpID redop = 0, bool reduction_fold = true,
              WriteSet *perf = NULL, const FieldMask *performed_mask = NULL);
      virtual ApEvent issue_fill(const PhysicalTraceInfo *info,RegionNode *node,
              const std::vector<CopySrcDstField> &dst_fields,
              const void *fill_value, size_t fill_size,
              ApEvent precondition, PredEvent predicate_guard,
#ifdef LEGION_SPY
              UniqueID fill_uid,
#endif
              IndexTreeNode *intersect = NULL,
              IndexSpaceExpression *mask = NULL,
              WriteSet *perf = NULL, const FieldMask *performed_mask = NULL);
    public:
      virtual Realm::InstanceLayoutGeneric* create_layout(
                           const Realm::InstanceLayoutConstraints &ilc,
                           const OrderingConstraint &constraint);
      virtual PhysicalInstance create_file_instance(const char *file_name,
                                   const std::vector<Realm::FieldID> &field_ids,
                                   const std::vector<size_t> &field_sizes,
                                   legion_file_mode_t file_mode, 
                                   ApEvent &ready_event);
      virtual PhysicalInstance create_hdf5_instance(const char *file_name,
                                   const std::vector<Realm::FieldID> &field_ids,
                                   const std::vector<size_t> &field_sizes,
                                   const std::vector<const char*> &field_files,
                                   bool read_only, ApEvent &ready_event);
      virtual PhysicalInstance create_external_instance(Memory memory,
                             uintptr_t base, Realm::InstanceLayoutGeneric *ilg,
                             ApEvent &ready_event);
    public:
      virtual void get_launch_space_domain(Domain &launch_domain);
      virtual void validate_slicing(const std::vector<IndexSpace> &slice_spaces,
                                    MultiTask *task, MapperManager *mapper);
      virtual void log_launch_space(UniqueID op_id);
      virtual IndexSpace create_shard_space(ShardingFunction *func, 
                                            ShardID shard, 
                                            IndexSpace shard_space);
      virtual void destroy_shard_domain(const Domain &domain);
    public:
      bool contains_point(const Realm::Point<DIM,T> &point);
    protected:
      void compute_linearization_metadata(void);
    protected:
      Realm::IndexSpace<DIM,T> realm_index_space;
    protected: // linearization meta-data, computed on demand
      Realm::Point<DIM,long long> strides;
      Realm::Point<DIM,long long> offset;
      bool linearization_ready;
    public:
      struct CreateByFieldHelper {
      public:
        CreateByFieldHelper(IndexSpaceNodeT<DIM,T> *n,
                            Operation *o, IndexPartNode *p,
                            const std::vector<FieldDataDescriptor> &i,
                            ApEvent r)
          : node(n), op(o), partition(p), instances(i), ready(r) { }
      public:
        template<typename COLOR_DIM, typename COLOR_T>
        static inline void demux(CreateByFieldHelper *creator)
        {
          creator->result = 
           creator->node->template create_by_field_helper<COLOR_DIM::N,COLOR_T>(
                         creator->op, creator->partition, creator->instances,
                         creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM,T> *node;
        Operation *op;
        IndexPartNode *partition;
        const std::vector<FieldDataDescriptor> &instances;
        ApEvent ready, result;
      };
      struct CreateByImageHelper {
      public:
        CreateByImageHelper(IndexSpaceNodeT<DIM,T> *n,
                            Operation *o, IndexPartNode *p, IndexPartNode *j,
                            const std::vector<FieldDataDescriptor> &i,
                            ApEvent r, ShardID s, size_t t)
          : node(n), op(o), partition(p), projection(j), 
            instances(i), ready(r), shard(s), total_shards(t) { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByImageHelper *creator)
        {
          creator->result = 
           creator->node->template create_by_image_helper<DIM2::N,T2>(
               creator->op, creator->partition, creator->projection,
               creator->instances, creator->ready, creator->shard,
               creator->total_shards);
        }
      public:
        IndexSpaceNodeT<DIM,T> *node;
        Operation *op;
        IndexPartNode *partition;
        IndexPartNode *projection;
        const std::vector<FieldDataDescriptor> &instances;
        ApEvent ready, result;
        ShardID shard;
        size_t total_shards;
      };
      struct CreateByImageRangeHelper {
      public:
        CreateByImageRangeHelper(IndexSpaceNodeT<DIM,T> *n,
                            Operation *o, IndexPartNode *p, IndexPartNode *j,
                            const std::vector<FieldDataDescriptor> &i,
                            ApEvent r, ShardID s, size_t t)
          : node(n), op(o), partition(p), projection(j), 
            instances(i), ready(r), shard(s), total_shards(t) { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByImageRangeHelper *creator)
        {
          creator->result = creator->node->template 
            create_by_image_range_helper<DIM2::N,T2>(
               creator->op, creator->partition, creator->projection,
               creator->instances, creator->ready, creator->shard,
               creator->total_shards);
        }
      public:
        IndexSpaceNodeT<DIM,T> *node;
        Operation *op;
        IndexPartNode *partition;
        IndexPartNode *projection;
        const std::vector<FieldDataDescriptor> &instances;
        ApEvent ready, result;
        ShardID shard;
        size_t total_shards;
      };
      struct CreateByPreimageHelper {
      public:
        CreateByPreimageHelper(IndexSpaceNodeT<DIM,T> *n,
                            Operation *o, IndexPartNode *p, IndexPartNode *j,
                            const std::vector<FieldDataDescriptor> &i,
                            ApEvent r)
          : node(n), op(o), partition(p), projection(j), 
            instances(i), ready(r) { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByPreimageHelper *creator)
        {
          creator->result = 
           creator->node->template create_by_preimage_helper<DIM2::N,T2>(
               creator->op, creator->partition, creator->projection,
               creator->instances, creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM,T> *node;
        Operation *op;
        IndexPartNode *partition;
        IndexPartNode *projection;
        const std::vector<FieldDataDescriptor> &instances;
        ApEvent ready, result;
      };
      struct CreateByPreimageRangeHelper {
      public:
        CreateByPreimageRangeHelper(IndexSpaceNodeT<DIM,T> *n,
                            Operation *o, IndexPartNode *p, IndexPartNode *j,
                            const std::vector<FieldDataDescriptor> &i,
                            ApEvent r)
          : node(n), op(o), partition(p), projection(j), 
            instances(i), ready(r) { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByPreimageRangeHelper *creator)
        {
          creator->result = creator->node->template 
            create_by_preimage_range_helper<DIM2::N,T2>(
               creator->op, creator->partition, creator->projection,
               creator->instances, creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM,T> *node;
        Operation *op;
        IndexPartNode *partition;
        IndexPartNode *projection;
        const std::vector<FieldDataDescriptor> &instances;
        ApEvent ready, result;
      };
      struct CreateAssociationHelper {
      public:
        CreateAssociationHelper(IndexSpaceNodeT<DIM,T> *n,
                            Operation *o, IndexSpaceNode *g,
                            const std::vector<FieldDataDescriptor> &i,
                            ApEvent r)
          : node(n), op(o), range(g), instances(i), ready(r) { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateAssociationHelper *creator)
        {
          creator->result = creator->node->template 
            create_association_helper<DIM2::N,T2>(
               creator->op, creator->range, creator->instances, creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM,T> *node;
        Operation *op;
        IndexSpaceNode *range;
        const std::vector<FieldDataDescriptor> &instances;
        ApEvent ready, result;
      };
    };

    /**
     * \class IndexSpaceCreator
     * A small helper class for creating templated index spaces
     */
    class IndexSpaceCreator {
    public:
      IndexSpaceCreator(RegionTreeForest *f, IndexSpace s, const void *i,
                        IndexPartNode *p, LegionColor c, DistributedID d, 
                        ApEvent r, IndexSpaceExprID e)
        : forest(f), space(s), realm_is(i), parent(p), 
          color(c), did(d), ready(r), expr_id(e), result(NULL) { }
    public:
      template<typename N, typename T>
      static inline void demux(IndexSpaceCreator *creator)
      {
        const Realm::IndexSpace<N::N,T> *is = 
          (const Realm::IndexSpace<N::N,T>*)creator->realm_is;
        creator->result = new IndexSpaceNodeT<N::N,T>(creator->forest,
            creator->space, creator->parent, creator->color, is,
            creator->did, creator->ready, creator->expr_id);
      }
    public:
      RegionTreeForest *const forest;
      const IndexSpace space; 
      const void *const realm_is;
      IndexPartNode *const parent;
      const LegionColor color;
      const DistributedID did;
      const ApEvent ready;
      const IndexSpaceExprID expr_id;
      IndexSpaceNode *result;
    };

    /**
     * \class IndexPartNode
     * A node for representing a generic index partition.
     */
    class IndexPartNode : public IndexTreeNode {
    public:
      struct DisjointnessArgs : public LgTaskArgs<DisjointnessArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DISJOINTNESS_TASK_ID;
      public:
        DisjointnessArgs(IndexPartition p, ValueBroadcast<bool> *c, bool own)
          : LgTaskArgs<DisjointnessArgs>(implicit_provenance),
            pid(p), disjointness_collective(c), owner(own) { }
      public:
        const IndexPartition pid;
        ValueBroadcast<bool> *const disjointness_collective;
        const bool owner;
      };
    public:
      struct DynamicIndependenceArgs : 
        public LgTaskArgs<DynamicIndependenceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_SPACE_INDEPENDENCE_TASK_ID;
      public:
        DynamicIndependenceArgs(IndexPartNode *par, 
                                IndexSpaceNode *l, IndexSpaceNode *r)
          : LgTaskArgs<DynamicIndependenceArgs>(implicit_provenance),
            parent(par), left(l), right(r) { }
      public:
        IndexPartNode *const parent;
        IndexSpaceNode *const left, *const right;
      };
      struct SemanticRequestArgs : public LgTaskArgs<SemanticRequestArgs> {
      public:
        static const LgTaskID TASK_ID = LG_INDEX_PART_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        SemanticRequestArgs(IndexPartNode *proxy, 
                            SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticRequestArgs>(implicit_provenance),
            proxy_this(proxy), tag(t), source(src) { }
      public:
        IndexPartNode *const proxy_this;
        const SemanticTag tag;
        const AddressSpaceID source;
      };
      struct DeferChildArgs : public LgTaskArgs<DeferChildArgs> {
      public:
        static const LgTaskID TASK_ID = LG_INDEX_PART_DEFER_CHILD_TASK_ID;
      public:
        DeferChildArgs(IndexPartNode *proxy, LegionColor child,
                       IndexSpace *tar, RtUserEvent trig, AddressSpaceID src)
          : LgTaskArgs<DeferChildArgs>(implicit_provenance),
            proxy_this(proxy), child_color(child), target(tar),
            to_trigger(trig), source(src) { }
      public:
        IndexPartNode *const proxy_this;
        const LegionColor child_color;
        IndexSpace *const target;
        const RtUserEvent to_trigger;
        const AddressSpaceID source;
      };
      class DestructionFunctor {
      public:
        DestructionFunctor(IndexPartNode *n, ReferenceMutator *m)
          : node(n), mutator(m) { }
      public:
        void apply(AddressSpaceID target);
      public:
        IndexPartNode *const node;
        ReferenceMutator *const mutator;
      }; 
    public:
      IndexPartNode(RegionTreeForest *ctx, IndexPartition p,
                    IndexSpaceNode *par, IndexSpaceNode *color_space,
                    LegionColor c, bool disjoint, int complete,
                    DistributedID did, ApEvent partition_ready, 
                    ApBarrier partial_pending, ShardMapping *mapping);
      IndexPartNode(RegionTreeForest *ctx, IndexPartition p,
                    IndexSpaceNode *par, IndexSpaceNode *color_space,
                    LegionColor c, RtEvent disjointness_ready,
                    int complete, DistributedID did,
                    ApEvent partition_ready, ApBarrier partial_pending,
                    ShardMapping *mapping);
      IndexPartNode(const IndexPartNode &rhs);
      virtual ~IndexPartNode(void);
    public:
      IndexPartNode& operator=(const IndexPartNode &rhs);
    public:
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      virtual bool is_index_space_node(void) const;
#ifdef DEBUG_LEGION
      virtual IndexSpaceNode* as_index_space_node(void);
      virtual IndexPartNode* as_index_part_node(void);
#endif
      virtual AddressSpaceID get_owner_space(void) const;
      static AddressSpaceID get_owner_space(IndexPartition handle, Runtime *rt);
    public:
      virtual IndexTreeNode* get_parent(void) const;
      virtual void get_colors(std::vector<LegionColor> &colors);
    public:
      virtual void send_semantic_request(AddressSpaceID target, 
           SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready);
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                             const void *buffer, size_t size, bool is_mutable,
                             RtUserEvent ready = RtUserEvent::NO_RT_USER_EVENT);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source,
                            bool can_fail, bool wait_until, RtUserEvent ready);
      static void handle_semantic_request(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
    public:
      bool has_color(const LegionColor c);
      IndexSpaceNode* get_child(const LegionColor c, RtEvent *defer = NULL);
      void add_child(IndexSpaceNode *child);
      void remove_child(const LegionColor c);
      size_t get_num_children(void) const;
      void compute_disjointness(ValueBroadcast<bool> *collective, bool owner);
      void get_subspace_preconditions(std::set<ApEvent> &preconditions);
    public:
      bool is_disjoint(bool from_app = false);
      bool are_disjoint(const LegionColor c1, const LegionColor c2,
                        bool force_compute = false);
      void record_disjointness(bool disjoint,
                               const LegionColor c1, const LegionColor c2);
      bool is_complete(bool from_app = false);
      IndexSpaceExpression* get_union_expression(bool check_complete=true);
    public:
      void add_instance(PartitionNode *inst);
      bool has_instance(RegionTreeID tid);
      void remove_instance(PartitionNode *inst);
    public:
      ApEvent create_equal_children(Operation *op, size_t granularity,
                                    ShardID shard, size_t total_shards);
      ApEvent create_by_union(Operation *Op,
                              IndexPartNode *left, IndexPartNode *right,
                              ShardID shard, size_t total_shards);
      ApEvent create_by_intersection(Operation *op,
                              IndexPartNode *left, IndexPartNode *right,
                              ShardID shard, size_t total_shards);
      ApEvent create_by_difference(Operation *op,
                              IndexPartNode *left, IndexPartNode *right,
                              ShardID shard, size_t total_shards);
      ApEvent create_by_restriction(const void *transform, const void *extent,
                                    ShardID shard, size_t total_shards);
    public:
      bool compute_complete(void);
      bool intersects_with(IndexSpaceNode *other, bool compute = true);
      bool intersects_with(IndexPartNode *other, bool compute = true); 
      bool dominates(IndexSpaceNode *other);
      bool dominates(IndexPartNode *other);
      virtual bool destroy_node(AddressSpaceID source) = 0;
    public:
      static void handle_disjointness_computation(const void *args, 
                                                  RegionTreeForest *forest);
      static void handle_disjointness_test(IndexPartNode *parent,
                                           IndexSpaceNode *left,
                                           IndexSpaceNode *right);
    public:
      virtual void send_node(AddressSpaceID target, bool up);
      static void handle_node_creation(RegionTreeForest *context,
                                       Deserializer &derez, 
                                       AddressSpaceID source);
    public:
      static void handle_node_request(RegionTreeForest *context,
                                      Deserializer &derez,
                                      AddressSpaceID source);
      static void handle_node_return(Deserializer &derez);
      static void handle_node_child_request(
          RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source);
      static void defer_node_child_request(const void *args);
      static void handle_node_child_response(Deserializer &derez);
      static void handle_notification(RegionTreeForest *context, 
                                      Deserializer &derez);
    public:
      const IndexPartition handle;
      IndexSpaceNode *const parent;
      IndexSpaceNode *const color_space;
      const LegionColor total_children;
      const LegionColor max_linearized_color;
      const ApEvent partition_ready;
      const ApBarrier partial_pending;
      ShardMapping *const shard_mapping;
    protected:
      RtEvent disjoint_ready;
      bool disjoint;
    protected:
      bool has_complete, complete;
      volatile IndexSpaceExpression *union_expr;
    protected:
      // Must hold the node lock when accessing
      // the remaining data structures
      std::map<LegionColor,IndexSpaceNode*> color_map;
      std::map<LegionColor,RtUserEvent> pending_child_map;
      std::set<PartitionNode*> logical_nodes;
      std::set<std::pair<LegionColor,LegionColor> > disjoint_subspaces;
      std::set<std::pair<LegionColor,LegionColor> > aliased_subspaces;
    }; 

    /**
     * \class IndexPartNodeT
     * A template class for handling any templated realm calls
     * associated with realm index spaces
     */
    template<int DIM, typename T>
    class IndexPartNodeT : public IndexPartNode,
                           public LegionHeapify<IndexPartNodeT<DIM,T> > {
    public:
      IndexPartNodeT(RegionTreeForest *ctx, IndexPartition p,
                     IndexSpaceNode *par, IndexSpaceNode *color_space,
                     LegionColor c, bool disjoint, int complete,
                     DistributedID did, ApEvent partition_ready, 
                     ApBarrier pending, ShardMapping *shard_mapping);
      IndexPartNodeT(RegionTreeForest *ctx, IndexPartition p,
                     IndexSpaceNode *par, IndexSpaceNode *color_space,
                     LegionColor c, RtEvent disjointness_ready,
                     int complete, DistributedID did,
                     ApEvent partition_ready, ApBarrier pending,
                     ShardMapping *shard_mapping);
      IndexPartNodeT(const IndexPartNodeT &rhs);
      virtual ~IndexPartNodeT(void);
    public:
      IndexPartNodeT& operator=(const IndexPartNodeT &rhs);
    public:
      virtual bool destroy_node(AddressSpaceID source); 
    };

    /**
     * \class IndexPartCreator
     * A msall helper class for creating templated index partitions
     */
    class IndexPartCreator {
    public:
      IndexPartCreator(RegionTreeForest *f, IndexPartition p,
                       IndexSpaceNode *par, IndexSpaceNode *cs,
                       LegionColor c, bool d, int k, DistributedID id,
                       ApEvent r, ApBarrier pend, ShardMapping *m)
        : forest(f), partition(p), parent(par), color_space(cs),
          color(c), disjoint(d), complete(k), did(id), ready(r), 
          pending(pend), mapping(m) { }
      IndexPartCreator(RegionTreeForest *f, IndexPartition p,
                       IndexSpaceNode *par, IndexSpaceNode *cs,
                       LegionColor c, RtEvent d, int k, DistributedID id,
                       ApEvent r, ApBarrier pend, ShardMapping *m)
        : forest(f), partition(p), parent(par), color_space(cs),
          color(c), disjoint(false), complete(k), disjoint_ready(d),
          did(id), ready(r), pending(pend), mapping(m) { }
    public:
      template<typename N, typename T>
      static inline void demux(IndexPartCreator *creator)
      {
        if (creator->disjoint_ready.exists()) 
          creator->result = new IndexPartNodeT<N::N,T>(creator->forest,
              creator->partition, creator->parent, creator->color_space,
              creator->color, creator->disjoint_ready, creator->complete, 
              creator->did, creator->ready, creator->pending, creator->mapping);
        else
          creator->result = new IndexPartNodeT<N::N,T>(creator->forest,
              creator->partition, creator->parent, creator->color_space,
              creator->color, creator->disjoint, creator->complete, 
              creator->did, creator->ready, creator->pending, creator->mapping);
      }
    public:
      RegionTreeForest *const forest;
      const IndexPartition partition;
      IndexSpaceNode *const parent;
      IndexSpaceNode *const color_space;
      const LegionColor color;
      const bool disjoint;
      const int complete;
      const RtEvent disjoint_ready;
      const DistributedID did;
      const ApEvent ready;
      const ApBarrier pending;
      ShardMapping *const mapping;
      IndexPartNode *result;
    };

    /**
     * \class FieldSpaceNode
     * Represent a generic field space that can be
     * pointed at by nodes in the region trees.
     */
    class FieldSpaceNode : 
      public LegionHeapify<FieldSpaceNode>, public DistributedCollectable {
    public:
      struct FieldInfo {
      public:
        FieldInfo(void) : field_size(0), idx(0), serdez_id(0),
                          destroyed(false) { }
        FieldInfo(size_t size, unsigned id, CustomSerdezID sid)
          : field_size(size), idx(id), serdez_id(sid), destroyed(false) { }
      public:
        size_t field_size;
        unsigned idx;
        CustomSerdezID serdez_id;
        bool destroyed;
      };
      struct LocalFieldInfo {
      public:
        LocalFieldInfo(void)
          : size(0), count(0) { }
        LocalFieldInfo(size_t s)
          : size(s), count(0) { }
      public:
        size_t size;
        unsigned count;
      };
      struct FindTargetsFunctor {
      public:
        FindTargetsFunctor(std::deque<AddressSpaceID> &t)
          : targets(t) { }
      public:
        void apply(AddressSpaceID target);
      private:
        std::deque<AddressSpaceID> &targets;
      };
      struct SemanticRequestArgs : public LgTaskArgs<SemanticRequestArgs> {
      public:
        static const LgTaskID TASK_ID = 
          LG_FIELD_SPACE_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        SemanticRequestArgs(FieldSpaceNode *proxy, 
                            SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticRequestArgs>(implicit_provenance),
            proxy_this(proxy), tag(t), source(src) { }
      public:
        FieldSpaceNode *const proxy_this;
        const SemanticTag tag;
        const AddressSpaceID source;
      };
      struct SemanticFieldRequestArgs : 
        public LgTaskArgs<SemanticFieldRequestArgs> {
      public:
        static const LgTaskID TASK_ID = 
          LG_FIELD_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        SemanticFieldRequestArgs(FieldSpaceNode *proxy, FieldID f,
                                 SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticFieldRequestArgs>(implicit_provenance),
            proxy_this(proxy), fid(f), tag(t), source(src) { }
      public:
        FieldSpaceNode *const proxy_this;
        const FieldID fid;
        const SemanticTag tag;
        const AddressSpaceID source;
      };
      class DestructionFunctor {
      public:
        DestructionFunctor(FieldSpaceNode *n, ReferenceMutator *m)
          : node(n), mutator(m) { }
      public:
        void apply(AddressSpaceID target);
      public:
        FieldSpaceNode *const node;
        ReferenceMutator *const mutator;
      };
    public:
      FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx, DistributedID did);
      FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx,
                     DistributedID did, Deserializer &derez);
      FieldSpaceNode(const FieldSpaceNode &rhs);
      virtual ~FieldSpaceNode(void);
    public:
      FieldSpaceNode& operator=(const FieldSpaceNode &rhs);
      AddressSpaceID get_owner_space(void) const; 
      static AddressSpaceID get_owner_space(FieldSpace handle, Runtime *rt);
    public:
      virtual void notify_active(ReferenceMutator *mutator) { }
      virtual void notify_inactive(ReferenceMutator *mutator) { }
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      void attach_semantic_information(SemanticTag tag, AddressSpaceID source,
                            const void *buffer, size_t size, bool is_mutable);
      void attach_semantic_information(FieldID fid, SemanticTag tag,
                                       AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable);
      bool retrieve_semantic_information(SemanticTag tag,
             const void *&result, size_t &size, bool can_fail, bool wait_until);
      bool retrieve_semantic_information(FieldID fid, SemanticTag tag,
             const void *&result, size_t &size, bool can_fail, bool wait_until);
      void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                             const void *result, size_t size, bool is_mutable,
                             RtUserEvent ready = RtUserEvent::NO_RT_USER_EVENT);
      void send_semantic_field_info(AddressSpaceID target, FieldID fid,
            SemanticTag tag, const void *result, size_t size, bool is_mutable,
            RtUserEvent ready = RtUserEvent::NO_RT_USER_EVENT);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source,
                             bool can_fail, bool wait_until, RtUserEvent ready);
      void process_semantic_field_request(FieldID fid, SemanticTag tag, 
      AddressSpaceID source, bool can_fail, bool wait_until, RtUserEvent ready);
      static void handle_semantic_request(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_field_semantic_request(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_field_semantic_info(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
    public:
      RtEvent allocate_field(FieldID fid, size_t size,
                             CustomSerdezID serdez_id);
      RtEvent allocate_fields(const std::vector<size_t> &sizes,
                              const std::vector<FieldID> &fids,
                              CustomSerdezID serdez_id);
      void free_field(FieldID fid, AddressSpaceID source);
      void free_fields(const std::vector<FieldID> &to_free,
                       AddressSpaceID source);
    public:
      bool allocate_local_fields(const std::vector<FieldID> &fields,
                                 const std::vector<size_t> &sizes,
                                 CustomSerdezID serdez_id,
                                 const std::set<unsigned> &indexes,
                                 std::vector<unsigned> &new_indexes);
      void free_local_fields(const std::vector<FieldID> &to_free,
                             const std::vector<unsigned> &indexes);
      void update_local_fields(const std::vector<FieldID> &fields,
                               const std::vector<size_t> &sizes,
                               const std::vector<CustomSerdezID> &serdez_ids,
                               const std::vector<unsigned> &indexes);
      void remove_local_fields(const std::vector<FieldID> &to_removes);
    public:
      void update_creation_set(const ShardMapping &mapping);
      void process_alloc_notification(Deserializer &derez);
    public:
      bool has_field(FieldID fid);
      size_t get_field_size(FieldID fid);
      void get_all_fields(std::vector<FieldID> &to_set);
      void get_all_regions(std::set<LogicalRegion> &regions);
      void get_field_set(const FieldMask &mask, std::set<FieldID> &to_set);
      void get_field_set(const FieldMask &mask, std::vector<FieldID> &to_set);
      void get_field_set(const FieldMask &mask, const std::set<FieldID> &basis,
                         std::set<FieldID> &to_set);
    public:
      void add_instance(RegionNode *inst);
      RtEvent add_instance(LogicalRegion inst, AddressSpaceID source);
      bool has_instance(RegionTreeID tid);
      void remove_instance(RegionNode *inst);
      bool destroy_node(AddressSpaceID source);
    public:
      FieldMask get_field_mask(const std::set<FieldID> &fields) const;
      unsigned get_field_index(FieldID fid) const;
      void get_field_indexes(const std::vector<FieldID> &fields,
                             std::vector<unsigned> &indexes) const;
    public:
      void compute_field_layout(const std::vector<FieldID> &create_fields,
                                std::vector<size_t> &field_sizes,
                                std::vector<unsigned> &mask_index_map,
                                std::vector<CustomSerdezID> &serdez,
                                FieldMask &instance_mask);
    public:
      InstanceManager* create_external_instance(
            const std::vector<FieldID> &fields, RegionNode *node, AttachOp *op);
    public:
      LayoutDescription* find_layout_description(const FieldMask &field_mask,
                     unsigned num_dims, const LayoutConstraintSet &constraints);
      LayoutDescription* find_layout_description(const FieldMask &field_mask,
                                                LayoutConstraints *constraints);
      LayoutDescription* create_layout_description(const FieldMask &layout_mask,
                                                   const unsigned total_dims,
                                                 LayoutConstraints *constraints,
                                           const std::vector<unsigned> &indexes,
                                           const std::vector<FieldID> &fids,
                                           const std::vector<size_t> &sizes,
                                     const std::vector<CustomSerdezID> &serdez);
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
      static void handle_remote_instance_creation(RegionTreeForest *forest,
                                Deserializer &derez, AddressSpaceID source);
      static void handle_remote_reduction_creation(RegionTreeForest *forest,
                                Deserializer &derez, AddressSpaceID source);
    public:
      static void handle_alloc_request(RegionTreeForest *forest,
                                       Deserializer &derez);
      static void handle_alloc_notification(RegionTreeForest *forest,
                                            Deserializer &derez);
      static void handle_top_alloc(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_field_free(RegionTreeForest *forest,
                                    Deserializer &derez, AddressSpaceID source);
      static void handle_local_alloc_request(RegionTreeForest *forest,
                                             Deserializer &derez,
                                             AddressSpaceID source);
      static void handle_local_alloc_response(Deserializer &derez);
      static void handle_local_free(RegionTreeForest *forest,
                                    Deserializer &derez);
    public:
      // Help with debug printing
      char* to_string(const FieldMask &mask) const;
      void get_field_ids(const FieldMask &mask,
                         std::vector<FieldID> &fields) const;
    protected:
      // Assume we are already holding the node lock
      // when calling these methods
      int allocate_index(void);
      void free_index(unsigned index);
    protected:
      bool allocate_local_indexes(
            const std::vector<size_t> &sizes,
            const std::set<unsigned> &current_indexes,
                  std::vector<unsigned> &new_indexes);
      void free_local_indexes(const std::vector<unsigned> &indexes);
    public:
      const FieldSpace handle;
      RegionTreeForest *const context;
    private:
      LocalLock &node_lock;
      // Top nodes in the trees for which this field space is used
      std::set<LogicalRegion> logical_trees;
      std::set<RegionNode*> local_trees;
      std::map<FieldID,FieldInfo> fields;
      FieldMask available_indexes;
    private:
      // Keep track of the layouts associated with this field space
      // Index them by their hash of their field mask to help
      // differentiate them.
      std::map<LEGION_FIELD_MASK_FIELD_TYPE,LegionList<LayoutDescription*,
                          LAYOUT_DESCRIPTION_ALLOC>::tracked> layouts;
    private:
      LegionMap<SemanticTag,SemanticInfo>::aligned semantic_info;
      LegionMap<std::pair<FieldID,SemanticTag>,SemanticInfo>::aligned 
                                                    semantic_field_info;
    private:
      // Local field information
      std::vector<LocalFieldInfo> local_field_infos;
    public:
      bool destroyed;
    };
 
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
      RegionTreeNode(RegionTreeForest *ctx, FieldSpaceNode *column);
      virtual ~RegionTreeNode(void);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator) = 0;
      virtual void notify_valid(ReferenceMutator *mutator) { assert(false); }
      virtual void notify_invalid(ReferenceMutator *mutator) { assert(false); }
    public:
      static AddressSpaceID get_owner_space(RegionTreeID tid, Runtime *rt);
    public:
      inline PhysicalState* get_physical_state(VersionInfo &info)
      {
        // First check to see if the version info already has a state
        PhysicalState *result = info.find_physical_state(this);  
#ifdef DEBUG_LEGION
        assert(result != NULL);
#endif
        return result;
      }
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
      // Trampolines for copies and fills
      inline ApEvent issue_copy(const PhysicalTraceInfo *info,
                  const std::vector<CopySrcDstField> &src_fields,
                  const std::vector<CopySrcDstField> &dst_fields,
                  ApEvent precondition, PredEvent predicate_guard,
                  RegionTreeNode *intersect, IndexSpaceExpression *mask,
                  ReductionOpID redop = 0, bool reduction_fold = true,
                  WriteSet *performed = NULL,
                  const FieldMask *performed_mask = NULL);
      inline ApEvent issue_fill(const PhysicalTraceInfo *info,
                  const std::vector<CopySrcDstField> &dst_fields,
                  const void *fill_value, size_t fill_size,
                  ApEvent precondition, PredEvent predicate_guard,
#ifdef LEGION_SPY
                  UniqueID fill_uid,
#endif
                  RegionTreeNode *intersect = NULL,
                  IndexSpaceExpression *mask = NULL,
                  WriteSet *performed = NULL,
                  const FieldMask *performed_mask = NULL);
    public:
      void attach_semantic_information(SemanticTag tag, AddressSpaceID source,
                            const void *buffer, size_t size, bool is_mutable);
      bool retrieve_semantic_information(SemanticTag tag,
           const void *&result, size_t &size, bool can_fail, bool wait_until);
      virtual void send_semantic_request(AddressSpaceID target, 
        SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready) = 0;
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                         const void *buffer, size_t size, bool is_mutable,
                         RtUserEvent ready = RtUserEvent::NO_RT_USER_EVENT) = 0;
    public:
      // Logical traversal operations
      void initialize_logical_state(ContextID ctx,
                                    const FieldMask &init_dirty_mask);
      void register_logical_user(ContextID ctx,
                                 const LogicalUser &user,
                                 RegionTreePath &path,
                                 const LogicalTraceInfo &trace_info,
                                 VersionInfo &version_info,
                                 ProjectionInfo &projection_info,
                                 FieldMask &unopened_field_mask,
                     LegionMap<AdvanceOp*,LogicalUser>::aligned &advances,
                                 const bool replicate_context);
      void create_logical_open(ContextID ctx,
                               const FieldMask &open_mask,
                               const LogicalUser &creator,
                               const RegionTreePath &path,
                               const LogicalTraceInfo &trace_info);
      void create_logical_advance(ContextID ctx, LogicalState &state,
                                  const FieldMask &advance_mask,
                                  const LogicalUser &creator,
                                  const LogicalTraceInfo &trace_info,
                    LegionMap<AdvanceOp*,LogicalUser>::aligned &advances,
                                  bool parent_is_upper_bound,
                                  const LegionColor next_child);
      void register_local_user(LogicalState &state,
                               const LogicalUser &user,
                               const LogicalTraceInfo &trace_info);
      void add_open_field_state(LogicalState &state, bool arrived,
                                ProjectionInfo &projection_info,
                                const LogicalUser &user,
                                const FieldMask &open_mask,
                                const LegionColor next_child);
      void traverse_advance_analysis(ContextID ctx, AdvanceOp *advance,
                                     const LogicalUser &advance_user,
                                     const LogicalUser &create_user);
      void perform_advance_analysis(ContextID ctx, LogicalState &state, 
                                    AdvanceOp *advance,
                                    const LogicalUser &advance_user,
                                    const LogicalUser &create_user,
                                    const LegionColor next_child,
                                    const bool already_traced,
                                    const bool advance_root = true);
      void close_logical_node(LogicalCloser &closer,
                              const FieldMask &closing_mask,
                              bool read_only_close);
      void siphon_logical_children(LogicalCloser &closer,
                                   LogicalState &state,
                                   const FieldMask &closing_mask,
                                   const FieldMask *aliased_children,
                                   bool record_close_operations,
                                   const LegionColor next_child,
                                   FieldMask &open_below);
      void siphon_logical_projection(LogicalCloser &closer,
                                     LogicalState &state,
                                     const FieldMask &closing_mask,
                                     ProjectionInfo &proj_info,
                                     bool record_close_operations,
                                     FieldMask &open_below);
      void flush_logical_reductions(LogicalCloser &closer,
                                    LogicalState &state,
                                    FieldMask &reduction_flush_fields,
                                    bool record_close_operations,
                                    const LegionColor next_child,
                              LegionDeque<FieldState>::aligned &new_states);
      // Note that 'allow_next_child' and 
      // 'record_closed_fields' are mutually exclusive
      void perform_close_operations(LogicalCloser &closer,
                                    const FieldMask &closing_mask,
                                    FieldState &closing_state,
                                    const LegionColor next_child, 
                                    bool allow_next_child,
                                    const FieldMask *aliased_children,
                                    bool upgrade_next_child, 
                                    bool read_only_close,
                                    bool overwriting_close,
                                    bool record_close_operations,
                                    bool record_closed_fields,
                                    FieldMask &output_mask); 
      void merge_new_field_state(LogicalState &state, 
                                 const FieldState &new_state);
      void merge_new_field_states(LogicalState &state, 
                            const LegionDeque<FieldState>::aligned &new_states);
      void filter_prev_epoch_users(LogicalState &state, const FieldMask &mask);
      void filter_curr_epoch_users(LogicalState &state, const FieldMask &mask);
      void report_uninitialized_usage(Operation *op, unsigned index,
                                      const FieldMask &uninitialized);
      void record_logical_reduction(LogicalState &state, ReductionOpID redop,
                                    const FieldMask &user_mask);
      void clear_logical_reduction_fields(LogicalState &state,
                                          const FieldMask &cleared_mask);
      void sanity_check_logical_state(LogicalState &state);
      void register_logical_dependences(ContextID ctx, Operation *op,
                                        const FieldMask &field_mask,
                                        bool dominate);
      void register_logical_deletion(ContextID ctx,
                                     const LogicalUser &user,
                                     const FieldMask &check_mask,
                                     RegionTreePath &path,
                                     RestrictInfo &restrict_info,
                                     VersionInfo &version_info,
                                     const LogicalTraceInfo &trace_info,
                                     const bool replicate_context);
      void siphon_logical_deletion(LogicalCloser &closer,
                                   LogicalState &state,
                                   const FieldMask &current_mask,
                                   const LegionColor next_child,
                                   FieldMask &open_below,
                                   bool force_close_next);
    public:
      void send_back_logical_state(ContextID ctx, UniqueID context_uid,
                                   AddressSpaceID target);
      void process_logical_state_return(ContextID ctx, Deserializer &derez,
                                        AddressSpaceID source);
      static void handle_logical_state_return(Runtime *runtime,
                              Deserializer &derez, AddressSpaceID source);
    public:
      void compute_version_numbers(ContextID ctx, 
                                   const RegionTreePath &path,
                                   const RegionUsage &usage,
                                   const FieldMask &version_mask,
                                   FieldMask &unversioned_mask,
                                   Operation *op, unsigned idx,
                                   InnerContext *parent_ctx,
                                   VersionInfo &version_info,
                                   std::set<RtEvent> &ready_events,
                                   bool partial_traversal,
                                   UniqueID logical_context_uid,
        const LegionMap<ProjectionEpochID,FieldMask>::aligned *advance_epochs);
      void advance_version_numbers(ContextID ctx,
                                   const RegionTreePath &path,
                                   const FieldMask &advance_mask,
                                   InnerContext *parent_ctx,
                                   bool update_parent_state,
                                   bool skip_update_parent,
                                   UniqueID logical_context_uid,
                                   bool dedup_opens, bool dedup_advances, 
                                   ProjectionEpochID open_epoch,
                                   ProjectionEpochID advance_epoch,
            const LegionMap<unsigned,FieldMask>::aligned &dirty_previous,
                                   std::set<RtEvent> &ready_events);
    public:
      void initialize_current_state(ContextID ctx);
      void invalidate_current_state(ContextID ctx, bool users_only);
      void invalidate_deleted_state(ContextID ctx, 
                                    const FieldMask &deleted_mask);
      bool invalidate_version_state(ContextID ctx);
      void invalidate_version_managers(void);
    public:
      // Physical traversal operations
      CompositeView* create_composite_instance(ContextID ctx_id,
                                     FieldMask &closing_mask,
                                     VersionInfo &version_info,
                                     UniqueID logical_context_uid,
                                     InnerContext *owner_context,
                                     CompositeViewSummary &summary,
                                     InterCloseOp *op,
                                     std::set<RtEvent> &ready_events);
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
                               const RestrictInfo &restrict_info,
                               bool restrict_out = false);
      void sort_copy_instances(const TraversalInfo &info,
                               MaterializedView *target,
                               FieldMask &copy_mask,
               const LegionMap<LogicalView*,FieldMask>::aligned &copy_instances,
                 LegionMap<MaterializedView*,FieldMask>::aligned &src_instances,
               LegionMap<DeferredView*,FieldMask>::aligned &deferred_instances);
      bool sort_copy_instances_single(const TraversalInfo &info,
                                      MaterializedView *target,
                                      const FieldMask &copy_mask,
                              const std::vector<LogicalView*> &copy_instances,
                                      MaterializedView *&src_instance,
                                      DeferredView *&deferred_instance);
      // Issue copies for fields with the same event preconditions and masks
      void issue_grouped_copies(const TraversalInfo &info,
                                MaterializedView *dst, bool restrict_out,
                                PredEvent predicate_guard,
                      LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                const FieldMask &update_mask,
           const LegionMap<MaterializedView*,FieldMask>::aligned &src_instances,
                                VersionTracker *version_tracker,
                      LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                                CopyAcrossHelper *across_helper = NULL,
                                RegionTreeNode *intersect = NULL,
                                const WriteMasks *masks = NULL,
                                WriteSet *perf = NULL);
      // Helper method for the above to issue copies with a mask
      void issue_masked_copy(const TraversalInfo &info, 
                             const FieldMask &copy_mask,
                             MaterializedView *dst, bool restrict_out,
                             PredEvent predicate_guard, ApEvent copy_pre,
            const LegionMap<MaterializedView*,FieldMask>::aligned &src_instances,
                             VersionTracker *version_tracker,
                  LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                             CopyAcrossHelper *across_helper,
                             RegionTreeNode *intersect,
                             IndexSpaceExpression* mask,
                             WriteSet *performed);
      ApEvent issue_single_copy(const TraversalInfo &info,
                                MaterializedView *dst, bool restrict_out,
                                PredEvent predicate_guard, ApEvent copy_pre,
                                const FieldMask &copy_mask,
                                MaterializedView *src,
                                VersionTracker *version_tracker,
                                CopyAcrossHelper *across_helper,
                                RegionTreeNode *intersect,
                                IndexSpaceExpression *mask);
      void issue_update_reductions(LogicalView *target,
                                   const FieldMask &update_mask,
                                   VersionInfo &version_info,
          const LegionMap<ReductionView*,FieldMask>::aligned &valid_reductions,
                                   Operation *op, unsigned index,
                                   std::set<RtEvent> &map_applied_events,
                                   const PhysicalTraceInfo &trace_info,
                                   bool restrict_out = false);
      void invalidate_instance_views(PhysicalState *state,
                                     const FieldMask &invalid_mask); 
      void invalidate_reduction_views(PhysicalState *state,
                                      const FieldMask &invalid_mask);
      // Helper methods for doing copy/reduce-out for restricted coherence
      void issue_restricted_copies(const TraversalInfo &info,
         const RestrictInfo &restrict_info, 
         const InstanceSet &restricted_instances,
         const std::vector<MaterializedView*> &restricted_views,
         const LegionMap<LogicalView*,FieldMask>::aligned &copy_out_views);
      void issue_restricted_reductions(const TraversalInfo &info,
         const RestrictInfo &restrict_info,
         const InstanceSet &restricted_instances,
         const std::vector<InstanceView*> &restricted_views,
         const LegionMap<ReductionView*,FieldMask>::aligned &reduce_out_views);
      // Look for a view to remove from the set of valid views
      void filter_valid_views(PhysicalState *state, LogicalView *to_filter);
      void update_valid_views(PhysicalState *state, const FieldMask &valid_mask,
                              bool dirty, LogicalView *new_view);
      void update_valid_views(PhysicalState *state, const FieldMask &dirty_mask,
                              const std::vector<LogicalView*> &new_views,
                              const InstanceSet &corresponding_references);
      // I hate the container problem, same as previous except InstanceView 
      void update_valid_views(PhysicalState *state, const FieldMask &dirty_mask,
                              const std::vector<InstanceView*> &new_views,
                              const InstanceSet &corresponding_references);
      // More containter problems, we could use templates but whatever
      void update_valid_views(PhysicalState *state, const FieldMask &dirty_mask,
                              const std::vector<MaterializedView*> &new_views,
                              const InstanceSet &corresponding_references);
      void update_reduction_views(PhysicalState *state, 
                                  const FieldMask &valid_mask,
                                  ReductionView *new_view);
    public: // Help for physical analysis
      void find_complete_fields(const FieldMask &scope_fields,
          const LegionMap<LegionColor,FieldMask>::aligned &children,
          FieldMask &complete_fields);
      InstanceView* convert_manager(PhysicalManager *manager,InnerContext *ctx);
      InstanceView* convert_reference(const InstanceRef &ref,InnerContext *ctx);
      void convert_target_views(const InstanceSet &targets, 
          InnerContext *context, std::vector<InstanceView*> &target_views);
      // I hate the container problem, same as previous except MaterializedView
      void convert_target_views(const InstanceSet &targets, 
          InnerContext *context, std::vector<MaterializedView*> &target_views);
    public:
      bool register_instance_view(PhysicalManager *manager, 
                                  UniqueID context_uid, InstanceView *view);
      void unregister_instance_view(PhysicalManager *manager, 
                                    UniqueID context_uid);
      InstanceView* find_instance_view(PhysicalManager *manager,
                                       InnerContext *context);
    public:
      bool register_physical_manager(PhysicalManager *manager);
      void unregister_physical_manager(PhysicalManager *manager);
      PhysicalManager* find_manager(DistributedID did);
    public:
      void register_tracking_context(InnerContext *context);
      void unregister_tracking_context(InnerContext *context);
    public:
      virtual unsigned get_depth(void) const = 0;
      virtual LegionColor get_color(void) const = 0;
      virtual IndexTreeNode *get_row_source(void) const = 0;
      virtual IndexSpaceExpression* get_index_space_expression(void) const = 0;
      virtual RegionTreeID get_tree_id(void) const = 0;
      virtual RegionTreeNode* get_parent(void) const = 0;
      virtual RegionTreeNode* get_tree_child(const LegionColor c) = 0; 
      virtual bool is_region(void) const = 0;
#ifdef DEBUG_LEGION
      virtual RegionNode* as_region_node(void) const = 0;
      virtual PartitionNode* as_partition_node(void) const = 0;
#else
      inline RegionNode* as_region_node(void) const;
      inline PartitionNode* as_partition_node(void) const;
#endif
      virtual bool visit_node(PathTraverser *traverser) = 0;
      virtual bool visit_node(NodeTraverser *traverser) = 0;
      virtual AddressSpaceID get_owner_space(void) const = 0; 
    public:
      virtual bool are_children_disjoint(const LegionColor c1, 
                                         const LegionColor c2) = 0;
      virtual bool are_all_children_disjoint(void) = 0;
      virtual bool is_complete(void) = 0;
      virtual bool intersects_with(RegionTreeNode *other, 
                                   bool compute = true) = 0;
      virtual bool dominates(RegionTreeNode *other) = 0;
    public:
      virtual size_t get_num_children(void) const = 0;
      virtual void send_node(AddressSpaceID target) = 0;
      virtual InstanceView* find_context_view(PhysicalManager *manager, 
                                              InnerContext *context) = 0;
      virtual void print_logical_context(ContextID ctx, 
                                         TreeStateLogger *logger,
                                         const FieldMask &mask) = 0;
      virtual void print_physical_context(ContextID ctx, 
                                          TreeStateLogger *logger,
                                          const FieldMask &mask,
                                  std::deque<RegionTreeNode*> &to_traverse) = 0;
      virtual void print_context_header(TreeStateLogger *logger) = 0;
#ifdef DEBUG_LEGION
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
      static void perform_closing_checks(LogicalCloser &closer, bool read_only,
          typename LegionList<LogicalUser, ALLOC>::track_aligned &users, 
          const FieldMask &check_mask);
    public:
      inline FieldSpaceNode* get_column_source(void) const 
        { return column_source; }
    public:
      void update_creation_set(const ShardMapping &mapping);
    public:
      RegionTreeForest *const context;
      FieldSpaceNode *const column_source;
    public:
      NodeSet remote_instances;
      bool registered;
      bool destroyed;
#ifdef DEBUG_LEGION
    protected:
      bool currently_active; // should be monotonic
#endif
    protected:
      DynamicTable<LogicalStateAllocator> logical_states;
      DynamicTable<VersionManagerAllocator> current_versions;
    protected:
      LocalLock &node_lock;
      // While logical states and version managers have dense keys
      // within a node, distributed IDs don't so we use a map that
      // should rarely need to be accessed for tracking views
      // The distributed IDs here correspond to the Instance Manager
      // distributed ID.
      LegionMap<std::pair<PhysicalManager*,UniqueID>,InstanceView*,
                LOGICAL_VIEW_ALLOC>::tracked instance_views;
      LegionMap<DistributedID,PhysicalManager*,
                PHYSICAL_MANAGER_ALLOC>::tracked physical_managers;
      // Also need to track any contexts that are tracking this
      // region tree node in case we get deleted during execution
      std::set<InnerContext*> tracking_contexts;
    protected:
      LegionMap<SemanticTag,SemanticInfo>::aligned semantic_info;
    };

    /**
     * \class RegionNode
     * Represent a region in a region tree
     */
    class RegionNode : public RegionTreeNode, public LegionHeapify<RegionNode> {
    public:
      struct SemanticRequestArgs : public LgTaskArgs<SemanticRequestArgs> {
      public:
        static const LgTaskID TASK_ID = LG_REGION_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        SemanticRequestArgs(RegionNode *proxy, 
                            SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticRequestArgs>(implicit_provenance),
            proxy_this(proxy), tag(t), source(src) { }
      public:
        RegionNode *const proxy_this;
        const SemanticTag tag;
        const AddressSpaceID source;
      };
      class DestructionFunctor {
      public:
        DestructionFunctor(LogicalRegion h, Runtime *rt, AddressSpaceID src)
          : handle(h), runtime(rt), source(src) { }
      public:
        void apply(AddressSpaceID target);
      public:
        const LogicalRegion handle;
        Runtime *const runtime;
        const AddressSpaceID source;
      };
    public:
      RegionNode(LogicalRegion r, PartitionNode *par, IndexSpaceNode *row_src,
                 FieldSpaceNode *col_src, RegionTreeForest *ctx);
      RegionNode(const RegionNode &rhs);
      virtual ~RegionNode(void);
    public:
      RegionNode& operator=(const RegionNode &rhs);
    public:
      virtual void notify_inactive(ReferenceMutator *mutator);
    public:
      void record_registered(void);
    public:
      bool has_color(const LegionColor p);
      PartitionNode* get_child(const LegionColor p);
      void add_child(PartitionNode *child);
      void remove_child(const LegionColor p);
    public:
      void find_remote_instances(NodeSet &target_instances);
      bool destroy_node(AddressSpaceID source, bool root);
    public:
      virtual unsigned get_depth(void) const;
      virtual LegionColor get_color(void) const;
      virtual IndexTreeNode *get_row_source(void) const;
      virtual IndexSpaceExpression* get_index_space_expression(void) const;
      virtual RegionTreeID get_tree_id(void) const;
      virtual RegionTreeNode* get_parent(void) const;
      virtual RegionTreeNode* get_tree_child(const LegionColor c);
    public:
      virtual bool are_children_disjoint(const LegionColor c1, 
                                         const LegionColor c2);
      virtual bool are_all_children_disjoint(void);
      virtual bool is_region(void) const;
#ifdef DEBUG_LEGION
      virtual RegionNode* as_region_node(void) const;
      virtual PartitionNode* as_partition_node(void) const;
#endif
      virtual AddressSpaceID get_owner_space(void) const;
      static AddressSpaceID get_owner_space(LogicalRegion handle, Runtime *rt);
      virtual bool visit_node(PathTraverser *traverser);
      virtual bool visit_node(NodeTraverser *traverser);
      virtual bool is_complete(void);
      virtual bool intersects_with(RegionTreeNode *other, bool compute = true);
      virtual bool dominates(RegionTreeNode *other);
      virtual size_t get_num_children(void) const;
      virtual void send_node(AddressSpaceID target);
      static void handle_node_creation(RegionTreeForest *context,
                            Deserializer &derez, AddressSpaceID source);
    public:
      virtual void send_semantic_request(AddressSpaceID target, 
           SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready);
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                             const void *buffer, size_t size, bool is_mutable,
                             RtUserEvent ready = RtUserEvent::NO_RT_USER_EVENT);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source,
                            bool can_fail, bool wait_until, RtUserEvent ready);
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
                                          const FieldMask &mask,
                                      std::deque<RegionTreeNode*> &to_traverse);
      virtual void print_context_header(TreeStateLogger *logger);
      void print_logical_state(LogicalState &state,
                               const FieldMask &capture_mask,
                         LegionMap<LegionColor,FieldMask>::aligned &to_traverse,
                               TreeStateLogger *logger);
#ifdef DEBUG_LEGION
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
      void find_open_complete_partitions(ContextID ctx,
                                         const FieldMask &mask,
                    std::vector<LogicalPartition> &partitions);
    public:
      void premap_region(ContextID ctx, 
                         const RegionRequirement &req,
                         const FieldMask &valid_mask,
                         VersionInfo &version_info,
                         InstanceSet &targets);
      void register_region(const TraversalInfo &info, UniqueID logical_ctx_uid,
                           InnerContext *context, RestrictInfo &restrict_info, 
                           ApEvent term_event, const RegionUsage &usage, 
                           bool defer_add_users, InstanceSet &targets,
                           const ProjectionInfo *proj_info);
      void seed_state(ContextID ctx, ApEvent term_event,
                             const RegionUsage &usage,
                             const FieldMask &user_mask,
                             const InstanceSet &targets,
                             InnerContext *context, unsigned init_index,
                             const std::vector<LogicalView*> &corresponding,
                             std::set<RtEvent> &applied_events);
      void close_state(const TraversalInfo &info, RegionUsage &usage, 
                       UniqueID logical_context_uid, InnerContext *context, 
                       InstanceSet &targets);
      void fill_fields(ContextID ctx, const FieldMask &fill_mask,
                       const void *value, size_t value_size, 
                       UniqueID logical_ctx_uid, InnerContext *context, 
                       VersionInfo &version_info,
                       std::set<RtEvent> &map_applied_events,
                       PredEvent true_guard, PredEvent false_guard,
                       const PhysicalTraceInfo &trace_info
#ifdef LEGION_SPY
                       , UniqueID fill_op_uid
#endif
                       );
      ApEvent eager_fill_fields(ContextID ctx, Operation *op,
                              const unsigned index, 
                              UniqueID logical_ctx_uid, InnerContext *context,
                              const FieldMask &fill_mask,
                              const void *value, size_t value_size,
                              VersionInfo &version_info, InstanceSet &instances,
                              ApEvent precondition, PredEvent true_guard,
                              std::set<RtEvent> &map_applied_events,
                              const PhysicalTraceInfo &trace_info);
      InstanceRef attach_external(ContextID ctx, InnerContext *parent_ctx,
                                  const UniqueID logical_ctx_uid,
                                  const FieldMask &attach_mask,
                                  const RegionRequirement &req, 
                                  InstanceManager *manager, 
                                  VersionInfo &version_info,
                                  std::set<RtEvent> &map_applied_events);
      ApEvent detach_external(ContextID ctx, InnerContext *context, 
                              const UniqueID logical_ctx_uid,
                              VersionInfo &version_info, 
                              const InstanceRef &ref,
                              std::set<RtEvent> &map_applied_events);
    public:
      virtual InstanceView* find_context_view(PhysicalManager *manager,
                                              InnerContext *context);
      InstanceView* convert_reference_region(PhysicalManager *manager, 
                                             InnerContext *context);
      void convert_references_region(
                              const std::vector<PhysicalManager*> &managers,
                              std::vector<bool> &up_mask, InnerContext *context,
                              std::vector<InstanceView*> &results);
    public:
      const LogicalRegion handle;
      PartitionNode *const parent;
      IndexSpaceNode *const row_source;
    protected:
      std::map<LegionColor,PartitionNode*> color_map;
    };

    /**
     * \class PartitionNode
     * Represent an instance of a partition in a region tree.
     */
    class PartitionNode : public RegionTreeNode, 
                          public LegionHeapify<PartitionNode> {
    public:
      struct SemanticRequestArgs : public LgTaskArgs<SemanticRequestArgs> {
      public:
        static const LgTaskID TASK_ID = LG_PARTITION_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        SemanticRequestArgs(PartitionNode *proxy,
                            SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticRequestArgs>(implicit_provenance),
            proxy_this(proxy), tag(t), source(src) { }
      public:
        PartitionNode *const proxy_this;
        const SemanticTag tag;
        const AddressSpaceID source;
      };
      class DestructionFunctor {
      public:
        DestructionFunctor(LogicalPartition h, Runtime *rt, AddressSpaceID src)
          : handle(h), runtime(rt), source(src) { }
      public:
        void apply(AddressSpaceID target);
      public:
        const LogicalPartition handle;
        Runtime *const runtime;
        const AddressSpaceID source;
      };
    public:
      PartitionNode(LogicalPartition p, RegionNode *par, 
                    IndexPartNode *row_src, FieldSpaceNode *col_src,
                    RegionTreeForest *ctx);
      PartitionNode(const PartitionNode &rhs);
      virtual ~PartitionNode(void);
    public:
      PartitionNode& operator=(const PartitionNode &rhs);
    public:
      virtual void notify_inactive(ReferenceMutator *mutator);
    public:
      void record_registered(void);
    public:
      bool has_color(const LegionColor c);
      RegionNode* get_child(const LegionColor c);
      void add_child(RegionNode *child);
      void remove_child(const LegionColor c);
      bool destroy_node(AddressSpaceID source, bool root);
    public:
      virtual unsigned get_depth(void) const;
      virtual LegionColor get_color(void) const;
      virtual IndexTreeNode *get_row_source(void) const;
      virtual IndexSpaceExpression* get_index_space_expression(void) const;
      virtual RegionTreeID get_tree_id(void) const;
      virtual RegionTreeNode* get_parent(void) const;
      virtual RegionTreeNode* get_tree_child(const LegionColor c);
    public:
      virtual bool are_children_disjoint(const LegionColor c1, 
                                         const LegionColor c2);
      virtual bool are_all_children_disjoint(void);
      virtual bool is_region(void) const;
#ifdef DEBUG_LEGION
      virtual RegionNode* as_region_node(void) const;
      virtual PartitionNode* as_partition_node(void) const;
#endif
      virtual AddressSpaceID get_owner_space(void) const;
      static AddressSpaceID get_owner_space(LogicalPartition handle, 
                                            Runtime *runtime);
      virtual bool visit_node(PathTraverser *traverser);
      virtual bool visit_node(NodeTraverser *traverser);
      virtual bool is_complete(void);
      virtual bool intersects_with(RegionTreeNode *other, bool compute = true);
      virtual bool dominates(RegionTreeNode *other);
      virtual size_t get_num_children(void) const;
      virtual void send_node(AddressSpaceID target);
    public:
      virtual InstanceView* find_context_view(PhysicalManager *manager,
                                              InnerContext *context);
      InstanceView* convert_reference_partition(PhysicalManager *manager,
                                                InnerContext *context);
      void convert_references_partition(
                                  const std::vector<PhysicalManager*> &managers,
                                  std::vector<bool> &up_mask, 
                                  InnerContext *context,
                                  std::vector<InstanceView*> &results);
    public:
      virtual void send_semantic_request(AddressSpaceID target, 
           SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready);
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
                             const void *buffer, size_t size, bool is_mutable,
                             RtUserEvent ready = RtUserEvent::NO_RT_USER_EVENT);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source,
                            bool can_fail, bool wait_until, RtUserEvent ready);
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
                                          const FieldMask &mask,
                                      std::deque<RegionTreeNode*> &to_traverse);
      virtual void print_context_header(TreeStateLogger *logger);
      void print_logical_state(LogicalState &state,
                               const FieldMask &capture_mask,
                         LegionMap<LegionColor,FieldMask>::aligned &to_traverse,
                               TreeStateLogger *logger);
#ifdef DEBUG_LEGION
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
      std::map<LegionColor,RegionNode*> color_map;
    }; 

    // some inline implementations
#ifndef DEBUG_LEGION
    //--------------------------------------------------------------------------
    inline IndexSpaceNode* IndexTreeNode::as_index_space_node(void)
    //--------------------------------------------------------------------------
    {
      return static_cast<IndexSpaceNode*>(this);
    }

    //--------------------------------------------------------------------------
    inline IndexPartNode* IndexTreeNode::as_index_part_node(void)
    //--------------------------------------------------------------------------
    {
      return static_cast<IndexPartNode*>(this);
    }

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

    //--------------------------------------------------------------------------
    inline ApEvent RegionTreeNode::issue_copy(const PhysicalTraceInfo *info,
                  const std::vector<CopySrcDstField> &src_fields,
                  const std::vector<CopySrcDstField> &dst_fields,
                  ApEvent precondition, PredEvent predicate_guard,
                  RegionTreeNode *intersect, IndexSpaceExpression *mask,
                  ReductionOpID redop, bool reduction_fold,
                  WriteSet *performed, const FieldMask *performed_mask)
    //--------------------------------------------------------------------------
    {
      RegionNode *node = is_region() ? 
        as_region_node() : as_partition_node()->parent;
      return node->row_source->issue_copy(info, node, src_fields, dst_fields,
          precondition, predicate_guard, 
          (intersect == NULL) ? NULL : intersect->get_row_source(),
          mask, redop, reduction_fold, performed, performed_mask);
    }

    //--------------------------------------------------------------------------
    inline ApEvent RegionTreeNode::issue_fill(const PhysicalTraceInfo *info,
                  const std::vector<CopySrcDstField> &dst_fields,
                  const void *fill_value, size_t fill_size,
                  ApEvent precondition, PredEvent predicate_guard,
#ifdef LEGION_SPY
                  UniqueID fill_uid,
#endif
                  RegionTreeNode *intersect, IndexSpaceExpression *mask,
                  WriteSet *performed, const FieldMask *performed_mask)
    //--------------------------------------------------------------------------
    {
      RegionNode *node = is_region() ? 
        as_region_node() : as_partition_node()->parent;
#ifdef LEGION_SPY
      return node->row_source->issue_fill(info, node, dst_fields, 
          fill_value, fill_size, precondition, predicate_guard, fill_uid,
          (intersect == NULL) ? NULL : intersect->get_row_source(),
          mask, performed, performed_mask);
#else
      return node->row_source->issue_fill(info, node, dst_fields, 
          fill_value, fill_size, precondition, predicate_guard,
          (intersect == NULL) ? NULL : intersect->get_row_source(),
          mask, performed, performed_mask);
#endif
    }

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_REGION_TREE_H__

// EOF

