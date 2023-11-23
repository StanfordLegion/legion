/* Copyright 2023 Stanford University, NVIDIA Corporation
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
#include "legion/legion_profiling.h"
#include "legion/garbage_collection.h"
#include "legion/field_tree.h"

#include <algorithm>

namespace Legion {
  namespace Internal {

    /**
     * \struct FieldDataDescriptor
     * A small helper class for performing dependent
     * partitioning operations
     */
    struct FieldDataDescriptor {
    public:
      inline bool operator<(const FieldDataDescriptor &rhs) const
        { return (color < rhs.color); }
    public:
      Domain domain;
      DomainPoint color;
      PhysicalInstance inst;
    };

    struct DeppartResult {
    public:
      inline bool operator<(const DeppartResult &rhs) const
        { return (color < rhs.color); }
    public:
      Domain domain;
      LegionColor color;
    };

    /**
     * \struct IndirectRecord
     * A small helper class for performing exchanges of
     * instances for indirection copies
     */
    struct IndirectRecord {
    public:
      IndirectRecord(void) { }
      IndirectRecord(RegionTreeForest *forest, 
                     const RegionRequirement &req,
                     const InstanceSet &insts);
    public:
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      // In the same order as the fields for the actual copy
      std::vector<PhysicalInstance> instances;
      // Only valid if profiling is enabled
      std::vector<LgEvent> instance_events;
#ifdef LEGION_SPY
      IndexSpace index_space;
#endif
      Domain domain;
      ApEvent domain_ready;
    };

    /**
     * \struct PendingRemoteExpression
     * A small helper class for passing arguments associated
     * with deferred calls to unpack remote expressions
     */
    struct PendingRemoteExpression {
    public:
      PendingRemoteExpression(void)
        : handle(IndexSpace::NO_SPACE), remote_expr_id(0),
          source(0), is_index_space(false), done_ref_counting(false) { }
    public:
      IndexSpace handle;
      IndexSpaceExprID remote_expr_id;
      AddressSpaceID source;
      bool is_index_space;
      bool done_ref_counting;
    };

    /**
     * \class OperationCreator
     * A base class for handling the creation of index space operations
     */
    class OperationCreator {
    public:
      OperationCreator(RegionTreeForest *f);
      virtual ~OperationCreator(void); 
    public: 
      void produce(IndexSpaceOperation *op);
      IndexSpaceExpression* consume(void);
    public:
      virtual void create_operation(void) = 0;
    public:
      RegionTreeForest *const forest;
    protected:
      IndexSpaceOperation *result;
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
      struct DisjointnessArgs : public LgTaskArgs<DisjointnessArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DISJOINTNESS_TASK_ID;
      public:
        DisjointnessArgs(IndexPartition h, RtUserEvent r)
          : LgTaskArgs<DisjointnessArgs>(implicit_provenance),
            handle(h), ready(r) { }
      public:
        const IndexPartition handle;
        const RtUserEvent ready;
      };   
    public:
      RegionTreeForest(Runtime *rt);
      RegionTreeForest(const RegionTreeForest &rhs);
      ~RegionTreeForest(void);
    public:
      RegionTreeForest& operator=(const RegionTreeForest &rhs);
    public:
      IndexSpaceNode* create_index_space(IndexSpace handle, 
                              const Domain *domain,
                              DistributedID did, 
                              Provenance *provenance,
                              CollectiveMapping *mapping = NULL,
                              IndexSpaceExprID expr_id = 0,
                              ApEvent ready = ApEvent::NO_AP_EVENT,
                              RtEvent initialized = RtEvent::NO_RT_EVENT);
      IndexSpaceNode* create_union_space(IndexSpace handle, DistributedID did,
                              Provenance *provenance,
                              const std::vector<IndexSpace> &sources,
                              RtEvent initialized = RtEvent::NO_RT_EVENT,
                              CollectiveMapping *mapping = NULL,
                              IndexSpaceExprID expr_id = 0);
      IndexSpaceNode* create_intersection_space(IndexSpace handle, 
                              DistributedID did, Provenance *provenance,
                              const std::vector<IndexSpace> &sources,
                              RtEvent initialized = RtEvent::NO_RT_EVENT,
                              CollectiveMapping *mapping = NULL,
                              IndexSpaceExprID expr_id = 0);
      IndexSpaceNode* create_difference_space(IndexSpace handle,
                              DistributedID did, Provenance *provenance,
                              IndexSpace left, IndexSpace right,
                              RtEvent initialized = RtEvent::NO_RT_EVENT,
                              CollectiveMapping *mapping = NULL,
                              IndexSpaceExprID expr_id = 0);
      RtEvent create_pending_partition(InnerContext *ctx,
                                       IndexPartition pid,
                                       IndexSpace parent,
                                       IndexSpace color_space,
                                       LegionColor &partition_color,
                                       PartitionKind part_kind,
                                       DistributedID did,
                                       Provenance *provenance,
                                       CollectiveMapping *mapping = NULL,
                                       RtEvent initialized =
                                                   RtEvent::NO_RT_EVENT);
      void create_pending_cross_product(InnerContext *ctx,
                                        IndexPartition handle1,
                                        IndexPartition handle2,
                  std::map<IndexSpace,IndexPartition> &user_handles,
                                        PartitionKind kind,
                                        Provenance *provenance,
                                        LegionColor &part_color,
                                        std::set<RtEvent> &safe_events,
                                        ShardID shard = 0,
                                        const ShardMapping *mapping = NULL,
                          ValueBroadcast<LegionColor> *color_broadcast = NULL);
      void destroy_index_space(IndexSpace handle, AddressSpaceID source,
                               std::set<RtEvent> &applied_events,
                               const CollectiveMapping *mapping = NULL);
      void destroy_index_partition(IndexPartition handle,
                               std::set<RtEvent> &applied,
                               const CollectiveMapping *mapping = NULL);
    public:
      ApEvent create_equal_partition(Operation *op, 
                                     IndexPartition pid, 
                                     size_t granularity);
      ApEvent create_partition_by_weights(Operation *op,
                                          IndexPartition pid,
              const std::map<DomainPoint,FutureImpl*> &futures,
                                          size_t granularity);
      ApEvent create_partition_by_union(Operation *op,
                                        IndexPartition pid,
                                        IndexPartition handle1,
                                        IndexPartition handle2);
      ApEvent create_partition_by_intersection(Operation *op,
                                               IndexPartition pid,
                                               IndexPartition handle1,
                                               IndexPartition handle2);
      ApEvent create_partition_by_intersection(Operation *op,
                                               IndexPartition pid,
                                               IndexPartition part,
                                               const bool dominates);
      ApEvent create_partition_by_difference(Operation *op,
                                           IndexPartition pid,
                                           IndexPartition handle1,
                                           IndexPartition handle2);
      ApEvent create_partition_by_restriction(IndexPartition pid,
                                              const void *transform,
                                              const void *extent);
      ApEvent create_partition_by_domain(Operation *op, IndexPartition pid,
                          const std::map<DomainPoint,FutureImpl*> &futures,
                                         const Domain &future_map_domain,
                                         bool perform_intersections);
      ApEvent create_cross_product_partitions(Operation *op,
                                              IndexPartition base,
                                              IndexPartition source,
                                              LegionColor part_color,
                                              ShardID shard = 0,
                                              const ShardMapping *mapping=NULL);
    public:  
      ApEvent create_partition_by_field(Operation *op, FieldID fid,
                                        IndexPartition pending,
                    const std::vector<FieldDataDescriptor> &instances,
                          std::vector<DeppartResult> *results,
                                        ApEvent instances_ready);
      ApEvent create_partition_by_image(Operation *op, FieldID fid,
                                        IndexPartition pending,
                                        IndexPartition projection,
                      std::vector<FieldDataDescriptor> &instances,
                                        ApEvent instances_ready);
      ApEvent create_partition_by_image_range(Operation *op, FieldID fid,
                                              IndexPartition pending,
                                              IndexPartition projection,
                            std::vector<FieldDataDescriptor> &instances,
                                              ApEvent instances_ready);
      ApEvent create_partition_by_preimage(Operation *op, FieldID fid,
                                           IndexPartition pending,
                                           IndexPartition projection,
                    const std::vector<FieldDataDescriptor> &instances,
                    const std::map<DomainPoint,Domain> *remote_targets,
                          std::vector<DeppartResult> *results,
                                           ApEvent instances_ready);
      ApEvent create_partition_by_preimage_range(Operation *op, FieldID fid,
                                                 IndexPartition pending,
                                                 IndexPartition projection,
                    const std::vector<FieldDataDescriptor> &instances,
                    const std::map<DomainPoint,Domain> *remote_targets,
                          std::vector<DeppartResult> *results,
                                                 ApEvent instances_ready);
      ApEvent create_association(Operation *op, FieldID fid,
                                 IndexSpace domain, IndexSpace range,
                    const std::vector<FieldDataDescriptor> &instances,
                                 ApEvent instances_ready);
    public:
      ApEvent compute_pending_space(Operation *op, IndexSpace result,
                                    const std::vector<IndexSpace> &handles,
                                    bool is_union);
      ApEvent compute_pending_space(Operation *op, IndexSpace result,
                                    IndexPartition handle,
                                    bool is_union);
      ApEvent compute_pending_space(Operation *op, IndexSpace result,
                                    IndexSpace initial,
                                    const std::vector<IndexSpace> &handles);
    public:
      void set_pending_space_domain(IndexSpace target,
                                    Domain domain);
    public:
      IndexPartition get_index_partition(IndexSpace parent, Color color); 
      bool has_index_subspace(IndexPartition parent,
                              const void *realm_color, TypeTag type_tag);
      IndexSpace get_index_subspace(IndexPartition parent, 
                                    const void *realm_color,
                                    TypeTag type_tag);
      IndexSpace instantiate_subspace(IndexPartition parent, 
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
      FieldSpaceNode* create_field_space(FieldSpace handle, DistributedID did,
                                   Provenance *provenance,
                                   CollectiveMapping *mapping = NULL,
                                   RtEvent initialized = RtEvent::NO_RT_EVENT);
      void destroy_field_space(FieldSpace handle,
                               std::set<RtEvent> &applied,
                               const CollectiveMapping *mapping = NULL);
      // Return true if local is set to true and we actually performed the 
      // allocation.  It is an error if the field already existed and the
      // allocation was not local.
      RtEvent allocate_field(FieldSpace handle, size_t field_size, 
                             FieldID fid, CustomSerdezID serdez_id,
                             Provenance *provenance,
                             bool sharded_non_owner = false);
      FieldSpaceNode* allocate_field(FieldSpace handle, ApEvent ready,
                                     FieldID fid, CustomSerdezID serdez_id,
                                     Provenance *provenance,
                                     RtEvent &precondition,
                                     bool sharded_non_owner = false);
      void free_field(FieldSpace handle, FieldID fid, 
                      std::set<RtEvent> &applied,
                      bool sharded_non_owner = false);
      RtEvent allocate_fields(FieldSpace handle, 
                           const std::vector<size_t> &sizes,
                           const std::vector<FieldID> &resulting_fields,
                           CustomSerdezID serdez_id,
                           Provenance *provenance,
                           bool sharded_non_owner = false);
      FieldSpaceNode* allocate_fields(FieldSpace handle, ApEvent ready, 
                           const std::vector<FieldID> &resulting_fields,
                           CustomSerdezID serdez_id, 
                           Provenance *provenance, RtEvent &precondition,
                           bool sharded_non_owner = false);
      void free_fields(FieldSpace handle, 
                       const std::vector<FieldID> &to_free,
                       std::set<RtEvent> &applied,
                       bool sharded_non_owner = false);
      void free_field_indexes(FieldSpace handle,
                       const std::vector<FieldID> &to_free, 
                       RtEvent freed, bool sharded_non_owner = false);
    public:
      bool allocate_local_fields(FieldSpace handle, 
                                 const std::vector<FieldID> &resulting_fields,
                                 const std::vector<size_t> &sizes,
                                 CustomSerdezID serdez_id,
                                 const std::set<unsigned> &allocated_indexes,
                                 std::vector<unsigned> &new_indexes,
                                 Provenance *provenance);
      void free_local_fields(FieldSpace handle,
                             const std::vector<FieldID> &to_free,
                             const std::vector<unsigned> &indexes,
                             const CollectiveMapping *mapping = NULL);
      void update_local_fields(FieldSpace handle,
                               const std::vector<FieldID> &fields,
                               const std::vector<size_t> &sizes,
                               const std::vector<CustomSerdezID> &serdez_ids,
                               const std::vector<unsigned> &indexes,
                               Provenance *provenance);
      void remove_local_fields(FieldSpace handle,
                               const std::vector<FieldID> &to_remove);
    public:
      void get_all_fields(FieldSpace handle, std::set<FieldID> &fields);
      void get_all_regions(FieldSpace handle, std::set<LogicalRegion> &regions);
      size_t get_coordinate_size(IndexSpace handle, bool range);
      size_t get_field_size(FieldSpace handle, FieldID fid);
      CustomSerdezID get_field_serdez(FieldSpace handle, FieldID fid);
      void get_field_space_fields(FieldSpace handle, 
                                  std::vector<FieldID> &fields);
    public:
      RegionNode* create_logical_region(LogicalRegion handle, DistributedID did,
                                    Provenance *provenance,
                                    CollectiveMapping *mapping = NULL,
                                    RtEvent initialized = RtEvent::NO_RT_EVENT);
      void destroy_logical_region(LogicalRegion handle,
                                  std::set<RtEvent> &applied,
                                  const CollectiveMapping *mapping = NULL);
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
      void find_domain(IndexSpace handle, Domain &launch_domain);
      void validate_slicing(IndexSpace input_space,
                            const std::vector<IndexSpace> &slice_spaces,
                            MultiTask *task, MapperManager *mapper);
      void log_launch_space(IndexSpace handle, UniqueID op_id);
    public:
      // Logical analysis methods
      void perform_dependence_analysis(Operation *op, unsigned idx,
                                       const RegionRequirement &req,
                                       const ProjectionInfo &projection_info,
                                       const RegionTreePath &path,
                                       LogicalAnalysis &logical_analysis);
      bool perform_deletion_analysis(DeletionOp *op, unsigned idx,
                                     RegionRequirement &req,
                                     const RegionTreePath &path,
                                     bool invalidate_tree);
      // Used by dependent partition operations
      void find_open_complete_partitions(Operation *op, unsigned idx,
                                         const RegionRequirement &req,
                                     std::vector<LogicalPartition> &partitions);
    public:
      void perform_versioning_analysis(Operation *op, unsigned idx,
                                       const RegionRequirement &req,
                                       VersionInfo &version_info,
                                       std::set<RtEvent> &ready_events);
      void invalidate_current_context(RegionTreeContext ctx, bool users_only,
                                      RegionNode *top_node);
      bool match_instance_fields(const RegionRequirement &req1,
                                 const RegionRequirement &req2,
                                 const InstanceSet &inst1,
                                 const InstanceSet &inst2);
    public: // Physical analysis methods
      void physical_premap_region(Operation *op, unsigned index,
                                  RegionRequirement &req,
                                  const VersionInfo &version_info,
                                  InstanceSet &valid_instances,
                                  FieldMaskSet<ReplicatedView> &collectives,
                                  std::set<RtEvent> &map_applied_events);
      // Return a runtime event for when it's safe to perform
      // the registration for this equivalence set
      RtEvent physical_perform_updates(const RegionRequirement &req,
                                const VersionInfo &version_info,
                                Operation *op, unsigned index,
                                ApEvent precondition, ApEvent term_event,
                                const InstanceSet &targets,
                                const std::vector<PhysicalManager*> &sources,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &map_applied_events,
                                UpdateAnalysis *&analysis,
#ifdef DEBUG_LEGION
                                const char *log_name,
                                UniqueID uid,
#endif
                                const bool collective_rendezvous,
                                const bool record_valid = true,
                                const bool check_initialized = true,
                                const bool defer_copies = true);
      // Return an event for when the copy-out effects of the 
      // registration are done (e.g. for restricted coherence)
      ApEvent physical_perform_registration(RtEvent precondition,
                               UpdateAnalysis *analysis,
                               std::set<RtEvent> &map_applied_events,
                               bool symbolic = false);
      // Same as the two above merged together
      ApEvent physical_perform_updates_and_registration(
                                   const RegionRequirement &req,
                                   const VersionInfo &version_info,
                                   Operation *op, unsigned index,
                                   ApEvent precondition, ApEvent term_event,
                                   const InstanceSet &targets,
                                   const std::vector<PhysicalManager*> &sources,
                                   const PhysicalTraceInfo &trace_info,
                                   std::set<RtEvent> &map_applied_events,
#ifdef DEBUG_LEGION
                                   const char *log_name,
                                   UniqueID uid,
#endif
                                   const bool collective_rendezvous,
                                   const bool record_valid = true,
                                   const bool check_initialized = true);
      ApEvent acquire_restrictions(const RegionRequirement &req,
                                   const VersionInfo &version_info,
                                   AcquireOp *op, unsigned index,
                                   ApEvent precondition, ApEvent term_event,
                                   InstanceSet &restricted_instances,
                                   const PhysicalTraceInfo &trace_info,
                                   std::set<RtEvent> &map_applied_events
#ifdef DEBUG_LEGION
                                   , const char *log_name
                                   , UniqueID uid
#endif
                                   );
      ApEvent release_restrictions(const RegionRequirement &req,
                                   const VersionInfo &version_info,
                                   ReleaseOp *op, unsigned index,
                                   ApEvent precondition, ApEvent term_event,
                                   InstanceSet &restricted_instances,
                                   const std::vector<PhysicalManager*> &sources,
                                   const PhysicalTraceInfo &trace_info,
                                   std::set<RtEvent> &map_applied_events
#ifdef DEBUG_LEGION
                                   , const char *log_name
                                   , UniqueID uid
#endif
                                   );
      ApEvent copy_across(const RegionRequirement &src_req,
                          const RegionRequirement &dst_req,
                          const VersionInfo &src_version_info,
                          const VersionInfo &dst_version_info,
                          const InstanceSet &src_targets,
                          const InstanceSet &dst_targets, 
                          const std::vector<PhysicalManager*> &sources,
                          CopyOp *op, unsigned src_index, unsigned dst_index,
                          ApEvent precondition, ApEvent src_ready,
                          ApEvent dst_ready, PredEvent pred_guard,
                          const std::map<Reservation,bool> &reservations,
                          const PhysicalTraceInfo &trace_info,
                          std::set<RtEvent> &map_applied_events);
      ApEvent gather_across(const RegionRequirement &src_req,
                            const RegionRequirement &idx_req,
                            const RegionRequirement &dst_req,
                            std::vector<IndirectRecord> &records,
                            const InstanceSet &src_targets,
                            const InstanceSet &idx_targets,
                            const InstanceSet &dst_targets,
                            CopyOp *op, unsigned src_index,
                            unsigned idx_index, unsigned dst_index,
                            const bool gather_is_range,
                            const ApEvent init_precondition, 
                            const ApEvent src_ready,
                            const ApEvent dst_ready,
                            const ApEvent idx_ready,
                            const PredEvent pred_guard,
                            const ApEvent collective_precondition,
                            const ApEvent collective_postcondition,
                            const ApUserEvent local_precondition,
                            const std::map<Reservation,bool> &reservations,
                            const PhysicalTraceInfo &trace_info,
                            std::set<RtEvent> &map_applied_events,
                            const bool possible_src_out_of_range,
                            const bool compute_preimages);
      ApEvent scatter_across(const RegionRequirement &src_req,
                             const RegionRequirement &idx_req,
                             const RegionRequirement &dst_req,
                             const InstanceSet &src_targets,
                             const InstanceSet &idx_targets,
                             const InstanceSet &dst_targets,
                             std::vector<IndirectRecord> &records,
                             CopyOp *op, unsigned src_index,
                             unsigned idx_index, unsigned dst_index,
                             const bool scatter_is_range,
                             const ApEvent init_precondition, 
                             const ApEvent src_ready,
                             const ApEvent dst_ready,
                             const ApEvent idx_ready,
                             const PredEvent pred_guard,
                             const ApEvent collective_precondition,
                             const ApEvent collective_postcondition,
                             const ApUserEvent local_precondition,
                             const std::map<Reservation,bool> &reservations,
                             const PhysicalTraceInfo &trace_info,
                             std::set<RtEvent> &map_applied_events,
                             const bool possible_dst_out_of_range,
                             const bool possible_dst_aliasing,
                             const bool compute_preimages);
      ApEvent indirect_across(const RegionRequirement &src_req,
                              const RegionRequirement &src_idx_req,
                              const RegionRequirement &dst_req,
                              const RegionRequirement &dst_idx_req,
                              const InstanceSet &src_targets,
                              const InstanceSet &dst_targets,
                              std::vector<IndirectRecord> &src_records,
                              const InstanceSet &src_idx_target,
                              std::vector<IndirectRecord> &dst_records,
                              const InstanceSet &dst_idx_target, CopyOp *op,
                              unsigned src_index, unsigned dst_index,
                              unsigned src_idx_index, unsigned dst_idx_index,
                              const bool both_are_range,
                              const ApEvent init_precondition, 
                              const ApEvent src_ready,
                              const ApEvent dst_ready,
                              const ApEvent src_idx_ready,
                              const ApEvent dst_idx_ready,
                              const PredEvent pred_guard,
                              const ApEvent collective_precondition,
                              const ApEvent collective_postcondition,
                              const ApUserEvent local_precondition,
                              const std::map<Reservation,bool> &reservations,
                              const PhysicalTraceInfo &trace_info,
                              std::set<RtEvent> &map_applied_events,
                              const bool possible_src_out_of_range,
                              const bool possible_dst_out_of_range,
                              const bool possible_dst_aliasing,
                              const bool compute_preimages);
      void fill_fields(FillOp *op,
                       const RegionRequirement &req,
                       const unsigned index, FillView *fill_view,
                       const VersionInfo &version_info,
                       ApEvent precondition, PredEvent true_guard,
                       PredEvent false_guard,
                       const PhysicalTraceInfo &trace_info,
                       std::set<RtEvent> &map_applied_events);
      void discard_fields(DiscardOp *op, const unsigned index,
                       const RegionRequirement &req,
                       const VersionInfo &version_info,
                       const PhysicalTraceInfo &trace_info,
                       std::set<RtEvent> &map_applied_events);
      InstanceRef create_external_instance(AttachOp *attach_op,
                                const RegionRequirement &req,
                                const std::vector<FieldID> &field_set);
      ApEvent attach_external(AttachOp *attach_op, unsigned index,
                              const RegionRequirement &req,
                              const InstanceSet &external_instances,
                              const VersionInfo &version_info,
                              const ApEvent termination_event,
                              const PhysicalTraceInfo &trace_info,
                              std::set<RtEvent> &map_applied_events,
                              const bool restricted);
      ApEvent detach_external(const RegionRequirement &req, DetachOp *detach_op,
                              unsigned index, const VersionInfo &version_info,
                              const InstanceSet &target_instances,
                              const ApEvent termination_event,
                              const PhysicalTraceInfo &trace_info,
                              std::set<RtEvent> &map_applied_events,
                              RtEvent filter_precondition,
                              const bool second_analysis);
      void invalidate_fields(Operation *op, unsigned index,
                             const RegionRequirement &req,
                             const VersionInfo &version_info,
                             const PhysicalTraceInfo &trace_info,
                             std::set<RtEvent> &map_applied_events,
                             CollectiveMapping *collective_mapping,
                             const bool collective_first_local);
    public:
      void physical_convert_sources(Operation *op,
                               const RegionRequirement &req,
                               const std::vector<MappingInstance> &sources,
                               std::vector<PhysicalManager*> &result,
                               std::map<PhysicalManager*,unsigned> *acquired);
      int physical_convert_mapping(Operation *op,
                               const RegionRequirement &req,
                               std::vector<MappingInstance> &chosen,
                               InstanceSet &result, RegionTreeID &bad_tree,
                               std::vector<FieldID> &missing_fields,
                               std::map<PhysicalManager*,unsigned> *acquired,
                               std::vector<PhysicalManager*> &unacquired,
                               const bool do_acquire_checks,
                               const bool allow_partial_virtual = false);
      bool physical_convert_postmapping(Operation *op,
                               const RegionRequirement &req,
                               std::vector<MappingInstance> &chosen,
                               InstanceSet &result, RegionTreeID &bad_tree,
                               std::map<PhysicalManager*,unsigned> *acquired,
                               std::vector<PhysicalManager*> &unacquired,
                               const bool do_acquire_checks);
    public: // helper method for the above two methods
      void perform_missing_acquires(
                               std::map<PhysicalManager*,unsigned> &acquired,
                               const std::vector<PhysicalManager*> &unacquired);
#ifdef DEBUG_LEGION
    public:
      // Debugging method for checking context state
      void check_context_state(RegionTreeContext ctx);
#endif
    public:
      // We know the domain of the index space
      IndexSpaceNode* create_node(IndexSpace is, const void *bounds, 
                                  bool is_domain, IndexPartNode *par, 
                                  LegionColor color, DistributedID did,
                                  RtEvent initialized, Provenance *provenance,
                                  ApEvent is_ready = ApEvent::NO_AP_EVENT,
                                  IndexSpaceExprID expr_id = 0,
                                  CollectiveMapping *mapping = NULL,
                                  const bool add_root_reference = false,
                                  unsigned depth = UINT_MAX,
                                  const bool tree_valid = true);
      IndexSpaceNode* create_node(IndexSpace is,
                                  IndexPartNode &par, LegionColor color,
                                  DistributedID did, RtEvent initialized,
                                  Provenance *provenance,
                                  IndexSpaceExprID expr_id = 0,
                                  CollectiveMapping *mapping = NULL,
                                  unsigned depth = UINT_MAX);
      // We know the disjointness of the index partition
      IndexPartNode*  create_node(IndexPartition p, IndexSpaceNode *par,
                                  IndexSpaceNode *color_space, 
                                  LegionColor color, bool disjoint,int complete,
                                  DistributedID did, Provenance *provenance,
                                  RtEvent init,
                                  CollectiveMapping *mapping = NULL);
      // Give the event for when the disjointness information is ready
      IndexPartNode*  create_node(IndexPartition p, IndexSpaceNode *par,
                                  IndexSpaceNode *color_space,
                                  LegionColor color, int complete,
                                  DistributedID did, Provenance *provenance,
                                  RtEvent init,
                                  CollectiveMapping *mapping = NULL);
      FieldSpaceNode* create_node(FieldSpace space, DistributedID did,
                                  RtEvent init, Provenance *provenance,
                                  CollectiveMapping *mapping = NULL);
      FieldSpaceNode* create_node(FieldSpace space, DistributedID did,
                                  RtEvent initialized, Provenance *provenance,
                                  CollectiveMapping *mapping,
                                  Deserializer &derez);
      RegionNode*     create_node(LogicalRegion r, PartitionNode *par,
                                  RtEvent initialized, DistributedID did,
                                  Provenance *provenance = NULL,
                                  CollectiveMapping *mapping = NULL);
      PartitionNode*  create_node(LogicalPartition p, RegionNode *par);
    public:
      IndexSpaceNode* get_node(IndexSpace space, RtEvent *defer = NULL, 
                        const bool can_fail = false, const bool first = true);
      IndexPartNode*  get_node(IndexPartition part, RtEvent *defer = NULL, 
                        const bool can_fail = false, const bool first = true,
                        const bool local_only = false);
      FieldSpaceNode* get_node(FieldSpace space, 
                               RtEvent *defer = NULL, bool first = true);
      RegionNode*     get_node(LogicalRegion handle, 
                               bool need_check = true, bool first = true);
      PartitionNode*  get_node(LogicalPartition handle, bool need_check = true);
      RegionNode*     get_tree(RegionTreeID tid, bool first = true);
      // Request but don't block
      RtEvent find_or_request_node(IndexSpace space, AddressSpaceID target);
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
      void record_pending_index_space(IndexSpaceID space);
      void record_pending_partition(IndexPartitionID pid);
      void record_pending_field_space(FieldSpaceID space);
      void record_pending_region_tree(RegionTreeID tree);
    public:
      void revoke_pending_index_space(IndexSpaceID space);
      void revoke_pending_partition(IndexPartitionID pid);
      void revoke_pending_field_space(FieldSpaceID space);
      void revoke_pending_region_tree(RegionTreeID tree);
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
      bool check_types(TypeTag t1, TypeTag t2, bool &diff_dims);
      bool is_dominated(IndexSpace src, IndexSpace dst);
      bool is_dominated_tree_only(IndexSpace test, IndexPartition dominator);
      bool is_dominated_tree_only(IndexPartition test, IndexSpace dominator);
      bool is_dominated_tree_only(IndexPartition test,IndexPartition dominator);
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
                                       bool is_mutable, bool local_only);
      void attach_semantic_information(IndexPartition handle, SemanticTag tag,
                                       AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable, bool local_only);
      void attach_semantic_information(FieldSpace handle, SemanticTag tag,
                                       AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable, bool local_only);
      void attach_semantic_information(FieldSpace handle, FieldID fid,
                                       SemanticTag tag, AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable, bool local_only);
      void attach_semantic_information(LogicalRegion handle, SemanticTag tag,
                                       AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable, bool local_only);
      void attach_semantic_information(LogicalPartition handle, SemanticTag tag,
                                       AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable, bool local_only);
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
      //
      // Note that you do not need to worry about reference counting
      // expressions returned from these methods inside of tasks because 
      // we implicitly add references to them and store them in the 
      // implicit_live_expression data structure and then remove the 
      // references after the meta-task or runtime call is done executing.

      IndexSpaceExpression* union_index_spaces(IndexSpaceExpression *lhs,
                                              IndexSpaceExpression *rhs);
      IndexSpaceExpression* union_index_spaces(
                                 const std::set<IndexSpaceExpression*> &exprs);
    protected:
      // Internal version
      IndexSpaceExpression* union_index_spaces(
                               const std::vector<IndexSpaceExpression*> &exprs,
                               OperationCreator *creator = NULL);
    public:
      IndexSpaceExpression* intersect_index_spaces(
                                              IndexSpaceExpression *lhs,
                                              IndexSpaceExpression *rhs);
      IndexSpaceExpression* intersect_index_spaces(
                                 const std::set<IndexSpaceExpression*> &exprs);
    protected:
      IndexSpaceExpression* intersect_index_spaces(
                               const std::vector<IndexSpaceExpression*> &exprs,
                               OperationCreator *creator = NULL);
    public:
      IndexSpaceExpression* subtract_index_spaces(IndexSpaceExpression *lhs,
                  IndexSpaceExpression *rhs, OperationCreator *creator = NULL);
    public:
      IndexSpaceExpression* find_canonical_expression(IndexSpaceExpression *ex);
      void remove_canonical_expression(IndexSpaceExpression *expr, size_t vol);
    private:
      static inline bool compare_expressions(IndexSpaceExpression *one,
                                             IndexSpaceExpression *two);
      struct CompareExpressions {
      public:
        inline bool operator()(IndexSpaceExpression *one,
                               IndexSpaceExpression *two) const
        { return compare_expressions(one, two); }
      };
    public:
      // Methods for removing index space expression when they are done
      void remove_union_operation(IndexSpaceOperation *expr, 
                            const std::vector<IndexSpaceExpression*> &exprs);
      void remove_intersection_operation(IndexSpaceOperation *expr, 
                            const std::vector<IndexSpaceExpression*> &exprs);
      void remove_subtraction_operation(IndexSpaceOperation *expr,
                       IndexSpaceExpression *lhs, IndexSpaceExpression *rhs);
    public:
      // Remote expression methods
      IndexSpaceExpression* find_or_request_remote_expression(
              IndexSpaceExprID remote_expr_id, 
              IndexSpaceExpression *origin, RtEvent *wait_for = NULL);
      IndexSpaceExpression* find_remote_expression(
              const PendingRemoteExpression &pending_expression);
      void unregister_remote_expression(IndexSpaceExprID remote_expr_id);
      void handle_remote_expression_request(Deserializer &derez,
                                            AddressSpaceID source);
      void handle_remote_expression_response(Deserializer &derez,
                                             AddressSpaceID source);
    protected:
      IndexSpaceExpression* unpack_expression_value(Deserializer &derez,
                                                    AddressSpaceID source);
    public:
      Runtime *const runtime;
    protected:
      mutable LocalLock lookup_lock;
      mutable LocalLock lookup_is_op_lock;
      mutable LocalLock congruence_lock;
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
      std::map<IndexSpaceID,RtUserEvent> pending_index_spaces;
      std::map<IndexPartitionID,RtUserEvent> pending_partitions;
      std::map<FieldSpaceID,RtUserEvent> pending_field_spaces;
      std::map<RegionTreeID,RtUserEvent> pending_region_trees;
    private:
      // Index space operations
      std::map<IndexSpaceExprID/*first*/,ExpressionTrieNode*> union_ops;
      std::map<IndexSpaceExprID/*first*/,ExpressionTrieNode*> intersection_ops;
      std::map<IndexSpaceExprID/*lhs*/,ExpressionTrieNode*> difference_ops;
      // Remote expressions
      std::map<IndexSpaceExprID,IndexSpaceExpression*> remote_expressions;
      std::map<IndexSpaceExprID,RtEvent> pending_remote_expressions;
    private:
      // In order for the symbolic analysis to work, we need to know that
      // we don't have multiple symbols for congruent expressions. This data
      // structure is used to find congruent expressions where they exist
      std::map<std::pair<size_t,TypeTag>,
               std::set<IndexSpaceExpression*> > canonical_expressions;
    public:
      static const unsigned MAX_EXPRESSION_FANOUT = 32;
    };

    /**
     * \class PieceIteratorImpl
     * This is an interface for iterating over pieces 
     * which in this case are just a list of rectangles
     */
    class PieceIteratorImpl : public Collectable {
    public:
      virtual ~PieceIteratorImpl(void) { }
      virtual int get_next(int index, Domain &next_piece) = 0;
    };

    /**
     * \class PieceIteratorImplT
     * This is the templated version of this class that is
     * instantiated for each cominbation of type and dimensoinality
     */
    template<int DIM, typename T>
    class PieceIteratorImplT : public PieceIteratorImpl {
    public:
      PieceIteratorImplT(const void *piece_list, size_t piece_list_size,
                         IndexSpaceNodeT<DIM,T> *privilege_node); 
      virtual ~PieceIteratorImplT(void) { }
      virtual int get_next(int index, Domain &next_piece);
    protected:
      std::vector<Rect<DIM,T> > pieces;
    };

    /**
     * \class CopyAcrossExecutor
     * This is a virtual interface for performing copies between
     * two different fields including with lots of different kinds
     * of indirections and transforms.
     */
    class CopyAcrossExecutor : public InstanceNameClosure {
    public:
      struct DeferCopyAcrossArgs : public LgTaskArgs<DeferCopyAcrossArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_COPY_ACROSS_TASK_ID;
      public:
        DeferCopyAcrossArgs(CopyAcrossExecutor *e, Operation *o, 
            PredEvent guard, ApEvent copy_pre, ApEvent src_pre,
            ApEvent dst_pre, const PhysicalTraceInfo &info,
            bool replay, bool recurrent, unsigned stage);
      public:
        CopyAcrossExecutor *const executor;
        Operation *const op;
        PhysicalTraceInfo *const trace_info;
        const PredEvent guard;
        const ApEvent copy_precondition;
        const ApEvent src_indirect_precondition;
        const ApEvent dst_indirect_precondition;
        const ApUserEvent done_event;
        const unsigned stage;
        const bool replay;
        const bool recurrent_replay;
      };
    public:
      CopyAcrossExecutor(Runtime *rt, const bool preimages,
                         const std::map<Reservation,bool> &rsrvs)
        : runtime(rt), reservations(rsrvs), priority(0),
          compute_preimages(preimages) { }
      virtual ~CopyAcrossExecutor(void) { }
    public:
      // From InstanceNameClosure
      virtual LgEvent find_instance_name(PhysicalInstance inst) const = 0;
    public:
      virtual ApEvent execute(Operation *op, PredEvent pred_guard,
                              ApEvent copy_precondition,
                              ApEvent src_indirect_precondition, 
                              ApEvent dst_indirect_precondition,
                              const PhysicalTraceInfo &trace_info,
                              const bool replay = false,
                              const bool recurrent_replay = false,
                              const unsigned stage = 0) = 0;
      virtual void record_trace_immutable_indirection(bool source) = 0;
    public:
      static void handle_deferred_copy_across(const void *args);
    public:
      Runtime *const runtime; 
      // Reservations that must be acquired for performing this copy
      // across and whether they need to be acquired with exclusive
      // permissions or not
      const std::map<Reservation,bool> reservations;
      // Priority for this copy across
      int priority;
      // Say whether we should be computing preimages or not
      const bool compute_preimages;
    };
    
    /**
     * \class CopyAcrossUnstructured
     * Untyped base class for all unstructured copies between fields
     */
    class CopyAcrossUnstructured : public CopyAcrossExecutor {
    public:
      CopyAcrossUnstructured(Runtime *rt, const bool preimages,
                             const std::map<Reservation,bool> &rsrvs)
        : CopyAcrossExecutor(rt, preimages, rsrvs),
          src_indirect_field(0), dst_indirect_field(0),
          src_indirect_instance(PhysicalInstance::NO_INST),
          dst_indirect_instance(PhysicalInstance::NO_INST) { }
      virtual ~CopyAcrossUnstructured(void) { }
    public:
      // From InstanceNameClosure
      virtual LgEvent find_instance_name(PhysicalInstance inst) const;
    public:
      virtual ApEvent execute(Operation *op, PredEvent pred_guard,
                              ApEvent copy_precondition,
                              ApEvent src_indirect_precondition,
                              ApEvent dst_indirect_precondition,
                              const PhysicalTraceInfo &trace_info,
                              const bool replay = false,
                              const bool recurrent_replay = false,
                              const unsigned stage = 0) = 0;
      virtual void record_trace_immutable_indirection(bool source) = 0;
    public:
      void initialize_source_fields(RegionTreeForest *forest,
                                    const RegionRequirement &req,
                                    const InstanceSet &instances,
                                    const PhysicalTraceInfo &trace_info);
      void initialize_destination_fields(RegionTreeForest *forest,
                                    const RegionRequirement &req,
                                    const InstanceSet &instances,
                                    const PhysicalTraceInfo &trace_info,
                                    const bool exclusive_redop);
      void initialize_source_indirections(RegionTreeForest *forest,
                                    std::vector<IndirectRecord> &records,
                                    const RegionRequirement &src_req,
                                    const RegionRequirement &idx_req,
                                    const InstanceRef &indirect_instance,
                                    const bool both_are_range,
                                    const bool possible_out_of_range);
      void initialize_destination_indirections(RegionTreeForest *forest,
                                    std::vector<IndirectRecord> &records,
                                    const RegionRequirement &dst_req,
                                    const RegionRequirement &idx_req,
                                    const InstanceRef &indirect_instance,
                                    const bool both_are_range,
                                    const bool possible_out_of_range,
                                    const bool possible_aliasing,
                                    const bool exclusive_redop);
    public:
      // All the entries in these data structures are ordered by the
      // order of the fields in the original region requirements
      std::vector<CopySrcDstField> src_fields, dst_fields;
      std::vector<LgEvent> src_unique_events, dst_unique_events;
#ifdef LEGION_SPY
      RegionTreeID src_tree_id, dst_tree_id;
      unsigned unique_indirections_identifier;
#endif
    public:
      // All the 'instances' in the entries in these data strctures are
      // ordered by the order of the fields in the origin region requirements
      std::vector<IndirectRecord> src_indirections, dst_indirections;
      FieldID src_indirect_field, dst_indirect_field;
      PhysicalInstance src_indirect_instance, dst_indirect_instance;
      LgEvent src_indirect_instance_event, dst_indirect_instance_event;
      TypeTag src_indirect_type, dst_indirect_type;
      std::vector<unsigned> nonempty_indexes;
    public:
      RtEvent prev_done;
      ApEvent last_copy;
    public:
      bool both_are_range;
      bool possible_src_out_of_range;
      bool possible_dst_out_of_range;
      bool possible_dst_aliasing;
    };

    /**
     * \class CopyAcrossExecutorT
     * This is the templated version of the copy-across executor. It is
     * templated on the dimensions and coordinate type of the copy space
     * for the copy operation.
     */
    template<int DIM, typename T>
    class CopyAcrossUnstructuredT : public CopyAcrossUnstructured {
    public:
      typedef typename Realm::CopyIndirection<DIM,T>::Base CopyIndirection;
    public:
      struct ComputePreimagesHelper {
      public:
        ComputePreimagesHelper(CopyAcrossUnstructuredT<DIM,T> *u,
                               Operation *o, ApEvent p, bool s)
          : unstructured(u), op(o), precondition(p), source(s) { }
      public:
        template<typename N2, typename T2>
        static inline void demux(ComputePreimagesHelper *helper)
          { helper->result = helper->unstructured->template 
            perform_compute_preimages<N2::N,T2>(helper->new_preimages,
              helper->op, helper->precondition, helper->source); }
      public:
        std::vector<DomainT<DIM,T> > new_preimages;
        CopyAcrossUnstructuredT<DIM,T> *const unstructured;
        Operation *const op;
        const ApEvent precondition;
        ApEvent result;
        const bool source; 
      };
      struct RebuildIndirectionsHelper {
      public:
        RebuildIndirectionsHelper(CopyAcrossUnstructuredT<DIM,T> *u, bool s)
          : unstructured(u), source(s), empty(true) { }
      public:
        template<typename N2, typename T2>
        static inline void demux(RebuildIndirectionsHelper *helper)
          { helper->empty = helper->unstructured->template 
            rebuild_indirections<N2::N,T2>(helper->source); }
      public:
        CopyAcrossUnstructuredT<DIM,T> *const unstructured;
        const bool source;
        bool empty;
      };
    public:
      CopyAcrossUnstructuredT(Runtime *runtime, 
                              IndexSpaceExpression *expr,
                              const DomainT<DIM,T> &domain,
                              ApEvent domain_ready,
                              const std::map<Reservation,bool> &rsrvs,
                              const bool compute_preimages);
      virtual ~CopyAcrossUnstructuredT(void);
    public:
      virtual ApEvent execute(Operation *op, PredEvent pred_guard,
                              ApEvent copy_precondition,
                              ApEvent src_indirect_precondition,
                              ApEvent dst_indirect_precondition,
                              const PhysicalTraceInfo &trace_info,
                              const bool replay = false,
                              const bool recurrent_replay = false,
                              const unsigned stage = 0); 
      virtual void record_trace_immutable_indirection(bool source);
    public:
      ApEvent issue_individual_copies(const ApEvent precondition,
                      const Realm::ProfilingRequestSet &requests);
      template<int D2, typename T2>
      ApEvent perform_compute_preimages(std::vector<DomainT<DIM,T> > &preimages,
                Operation *op, ApEvent precondition, const bool source); 
      template<int D2, typename T2>
      bool rebuild_indirections(const bool source);
    public:
      IndexSpaceExpression *const expr;
      const DomainT<DIM,T> copy_domain;
      const ApEvent copy_domain_ready;
    protected:
      mutable LocalLock preimage_lock;
      std::deque<std::vector<DomainT<DIM,T> > > src_preimages, dst_preimages;
      std::vector<DomainT<DIM,T> > current_src_preimages, current_dst_preimages;
      std::vector<const CopyIndirection*> indirections;
      // Realm performs better if you can issue a separate copy for each of the
      // preimages so it doesn't have to do address splitting. Therefore when
      // we compute preimages and we only have a gather or a scatter copy then
      // we will attempt to issue individual copies for such cases. Note that
      // we don't bother doing this for full-indirection copies though as then
      // we would need to do the full quadratic intersection between each of
      // the source and destination preimages.
      std::vector<std::vector<unsigned> > individual_field_indexes;
      ApEvent src_indirect_spaces_precondition,dst_indirect_spaces_precondition;
#ifdef LEGION_SPY
      std::deque<ApEvent> src_preimage_preconditions;
      std::deque<ApEvent> dst_preimage_preconditions;
      ApEvent current_src_preimage_precondition;
      ApEvent current_dst_preimage_precondition;
#endif
      bool need_src_indirect_precondition, need_dst_indirect_precondition;
      bool src_indirect_immutable_for_tracing;
      bool dst_indirect_immutable_for_tracing;
      bool has_empty_preimages;
    };

    /**
     * \interface KDTree
     * A virtual interface to a KD tree
     */
    class KDTree {
    public:
      virtual ~KDTree(void) { }
    public:
      template<int DIM, typename T>
      inline KDNode<DIM,T>* as_kdnode(void);
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
        TightenIndexSpaceArgs(IndexSpaceExpression *proxy, 
                              DistributedCollectable *dc)
          : LgTaskArgs<TightenIndexSpaceArgs>(implicit_provenance),
            proxy_this(proxy), proxy_dc(dc)
          { proxy_dc->add_base_resource_ref(META_TASK_REF); }
      public:
        IndexSpaceExpression *const proxy_this;
        DistributedCollectable *const proxy_dc;
      };
    public:
      IndexSpaceExpression(LocalLock &lock);
      IndexSpaceExpression(TypeTag tag, Runtime *runtime, LocalLock &lock); 
      IndexSpaceExpression(TypeTag tag, IndexSpaceExprID id, LocalLock &lock);
      virtual ~IndexSpaceExpression(void);
    public:
      inline bool deterministic_pointer_less(const IndexSpaceExpression *rhs) 
        const { return (expr_id < rhs->expr_id); }
    public:
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag, 
                                           bool need_tight_result) = 0;
      virtual bool is_sparse() = 0;
      // If you ask for a tight index space you don't need to pay 
      // attention to the event returned as a precondition as it 
      // is guaranteed to be a no-event
      virtual ApEvent get_domain(Domain &domain, bool need_tight = true) = 0;
      virtual void tighten_index_space(void) = 0;
      virtual bool check_empty(void) = 0;
      virtual size_t get_volume(void) = 0;
      virtual void pack_expression(Serializer &rez, AddressSpaceID target) = 0;
      virtual void pack_expression_value(Serializer &rez,
                                         AddressSpaceID target) = 0;
    public:
#ifdef DEBUG_LEGION
      virtual bool is_valid(void) = 0;
#endif
      virtual DistributedID get_distributed_id(void) const = 0;
      virtual void add_canonical_reference(DistributedID source) = 0;
      virtual bool remove_canonical_reference(DistributedID source) = 0;
      virtual bool try_add_live_reference(void) = 0;
      virtual void add_base_expression_reference(ReferenceSource source,
                                                 unsigned count = 1) = 0;
      virtual void add_nested_expression_reference(DistributedID source,
                                                   unsigned count = 1) = 0;
      virtual bool remove_base_expression_reference(ReferenceSource source,
                                                    unsigned count = 1) = 0;
      virtual bool remove_nested_expression_reference(DistributedID source,
                                                      unsigned count = 1) = 0;
      virtual void add_tree_expression_reference(DistributedID source,
                                                 unsigned count = 1) = 0;
      virtual bool remove_tree_expression_reference(DistributedID source,
                                                    unsigned count = 1) = 0;
      virtual bool test_intersection_nonblocking(IndexSpaceExpression *expr,
         RegionTreeForest *context, ApEvent &precondition, bool second = false);
    public:
      virtual IndexSpaceNode* create_node(IndexSpace handle, DistributedID did,
          RtEvent initialized, Provenance *provenance,
          CollectiveMapping *mapping, IndexSpaceExprID expr_id = 0) = 0;
      virtual PieceIteratorImpl* create_piece_iterator(const void *piece_list,
                    size_t piece_list_size, IndexSpaceNode *privilege_node) = 0;
      virtual bool is_below_in_tree(IndexPartNode *p, LegionColor &child) const
        { return false; }
    public:
      virtual ApEvent issue_fill(Operation *op,
                           const PhysicalTraceInfo &trace_info,
                           const std::vector<CopySrcDstField> &dst_fields,
                           const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                           UniqueID fill_uid,
                           FieldSpace handle,
                           RegionTreeID tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard,
                           LgEvent unique_event,
                           CollectiveKind collective = COLLECTIVE_NONE,
                           int priority = 0, bool replay = false) = 0;
      virtual ApEvent issue_copy(Operation *op,
                           const PhysicalTraceInfo &trace_info,
                           const std::vector<CopySrcDstField> &dst_fields,
                           const std::vector<CopySrcDstField> &src_fields,
                           const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                           RegionTreeID src_tree_id,
                           RegionTreeID dst_tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard,
                           LgEvent src_unique, LgEvent dst_unique,
                           CollectiveKind collective = COLLECTIVE_NONE,
                           int priority = 0, bool replay = false) = 0;
      virtual CopyAcrossUnstructured* create_across_unstructured(
                           const std::map<Reservation,bool> &reservations,
                           const bool compute_preimages) = 0;
      virtual Realm::InstanceLayoutGeneric* create_layout(
                           const LayoutConstraintSet &constraints,
                           const std::vector<FieldID> &field_ids,
                           const std::vector<size_t> &field_sizes,
                           bool compact, void **piece_list = NULL,
                           size_t *piece_list_size = NULL,
                           size_t *num_pieces = NULL) = 0;
      // Return the expression with a resource ref on the expression
      virtual IndexSpaceExpression* create_layout_expression(
                           const void *piece_list, size_t piece_list_size) = 0;
      virtual bool meets_layout_expression(IndexSpaceExpression *expr,
         bool tight_bounds, const void *piece_list, size_t piece_list_size,
         const Domain *padding_delta) = 0;
    public:
      virtual IndexSpaceExpression* find_congruent_expression(
                  std::set<IndexSpaceExpression*> &expressions) = 0;
      virtual KDTree* get_sparsity_map_kd_tree(void) = 0;
    public:
      static void handle_tighten_index_space(const void *args);
      static AddressSpaceID get_owner_space(IndexSpaceExprID id, Runtime *rt);
    public:
      void add_derived_operation(IndexSpaceOperation *op);
      void remove_derived_operation(IndexSpaceOperation *op);
      void invalidate_derived_operations(DistributedID did,
                                         RegionTreeForest *context);
    public:
      inline bool is_empty(void)
      {
        if (!has_empty.load())
        {
          empty = check_empty();
          has_empty.store(true);
        }
        return empty;
      }
      inline size_t get_num_dims(void) const
        { return NT_TemplateHelper::get_dim(type_tag); }
    public:
      // Convert this index space expression to the canonical one that
      // represents all expressions that are all congruent
      IndexSpaceExpression* get_canonical_expression(RegionTreeForest *forest);
    protected:
      template<int DIM, typename T>
      inline ApEvent issue_fill_internal(RegionTreeForest *forest,Operation *op,
                               const Realm::IndexSpace<DIM,T> &space,
                               const PhysicalTraceInfo &trace_info,
                               const std::vector<CopySrcDstField> &dst_fields,
                               const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                               UniqueID fill_uid,
                               FieldSpace handle,
                               RegionTreeID tree_id,
#endif
                               ApEvent precondition, PredEvent pred_guard,
                               LgEvent unique_event, CollectiveKind collective,
                               int priority, bool replay);
      template<int DIM, typename T>
      inline ApEvent issue_copy_internal(RegionTreeForest *forest,Operation*op,
                               const Realm::IndexSpace<DIM,T> &space,
                               const PhysicalTraceInfo &trace_info,
                               const std::vector<CopySrcDstField> &dst_fields,
                               const std::vector<CopySrcDstField> &src_fields,
                               const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                               RegionTreeID src_tree_id,
                               RegionTreeID dst_tree_id,
#endif
                               ApEvent precondition, PredEvent pred_guard,
                               LgEvent src_unique, LgEvent dst_unique,
                               CollectiveKind collective,
                               int priority, bool replay);
      template<int DIM, typename T>
      inline Realm::InstanceLayoutGeneric* create_layout_internal(
                               const Realm::IndexSpace<DIM,T> &space,
                               const LayoutConstraintSet &constraints,
                               const std::vector<FieldID> &field_ids,
                               const std::vector<size_t> &field_sizes,
                               bool compact, void **piece_list = NULL,
                               size_t *piece_list_size = NULL,
                               size_t *num_pieces = NULL) const;
      template<int DIM, typename T>
      inline IndexSpaceExpression* create_layout_expression_internal(
                               RegionTreeForest *context,
                               const Realm::IndexSpace<DIM,T> &space,
                               const Rect<DIM,T> *rects, size_t num_rects);
      template<int DIM, typename T>
      inline bool meets_layout_expression_internal(
                         IndexSpaceExpression *space_expr, bool tight_bounds,
                         const Rect<DIM,T> *piece_list, size_t piece_list_size,
                         const Domain *padding_delta);
    public:
      template<int DIM, typename T>
      inline IndexSpaceExpression* find_congruent_expression_internal(
                        std::set<IndexSpaceExpression*> &expressions);
      template<int DIM, typename T>
      inline KDTree* get_sparsity_map_kd_tree_internal(void);
    public:
      static IndexSpaceExpression* unpack_expression(Deserializer &derez,
                         RegionTreeForest *forest, AddressSpaceID source); 
      static IndexSpaceExpression* unpack_expression(Deserializer &derez,
                         RegionTreeForest *forest, AddressSpaceID source,
                         PendingRemoteExpression &pending, RtEvent &wait_for);
    public:
      const TypeTag type_tag;
      const IndexSpaceExprID expr_id;
    private:
      LocalLock &expr_lock;
    protected:
      std::set<IndexSpaceOperation*> derived_operations;
      std::atomic<IndexSpaceExpression*> canonical;
      KDTree *sparsity_map_kd_tree;
      size_t volume;
      std::atomic<bool> has_volume;
      bool empty;
      std::atomic<bool> has_empty;
    };

    /**
     * This is a move-only object that tracks temporary references to
     * index space expressions that are returned from region tree ops
     */
    class IndexSpaceExprRef {
    public:
      IndexSpaceExprRef(void) : expr(NULL) { }
      IndexSpaceExprRef(IndexSpaceExpression *e)
        : expr(e)
      { 
        if (expr != NULL)
          expr->add_base_expression_reference(LIVE_EXPR_REF);
      }
      IndexSpaceExprRef(const IndexSpaceExprRef &rhs) = delete;
      IndexSpaceExprRef(IndexSpaceExprRef &&rhs)
        : expr(rhs.expr)
      {
        rhs.expr = NULL;
      }
      ~IndexSpaceExprRef(void)
      {
        if ((expr != NULL) && 
            expr->remove_base_expression_reference(LIVE_EXPR_REF))
          delete expr;
      }
      IndexSpaceExprRef& operator=(const IndexSpaceExprRef &rhs) = delete;
      inline IndexSpaceExprRef& operator=(IndexSpaceExprRef &&rhs)
      {
        if ((expr != NULL) && 
            expr->remove_base_expression_reference(LIVE_EXPR_REF))
          delete expr;
        expr = rhs.expr;
        rhs.expr = NULL;
        return *this;
      }
    public:
      inline bool operator==(const IndexSpaceExprRef &rhs) const
      {
        if (expr == NULL)
          return (rhs.expr == NULL);
        if (rhs.expr == NULL)
          return false;
        return (expr->expr_id == rhs.expr->expr_id);
      }
      inline bool operator<(const IndexSpaceExprRef &rhs) const
      {
        if (expr == NULL)
          return (rhs.expr != NULL);
        if (rhs.expr == NULL)
          return false;
        return (expr->expr_id < rhs.expr->expr_id);
      }
      inline IndexSpaceExpression* operator->(void) { return expr; }
      inline IndexSpaceExpression* operator&(void) { return expr; }
    protected:
      IndexSpaceExpression *expr;
    };

    class IndexSpaceOperation : public IndexSpaceExpression,
                                public DistributedCollectable {
    public:
      enum OperationKind {
        UNION_OP_KIND,
        INTERSECT_OP_KIND,
        DIFFERENCE_OP_KIND,
        REMOTE_EXPRESSION_KIND,
        INSTANCE_EXPRESSION_KIND,
      };
    public:
      IndexSpaceOperation(TypeTag tag, OperationKind kind,
                          RegionTreeForest *ctx);
      IndexSpaceOperation(TypeTag tag, RegionTreeForest *ctx,
          IndexSpaceExprID eid, DistributedID did, IndexSpaceOperation *origin);
      virtual ~IndexSpaceOperation(void);
    public:
      virtual void notify_local(void);
    public:
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag, 
                                           bool need_tight_result) = 0;
      // If you ask for a tight index space you don't need to pay 
      // attention to the event returned as a precondition as it 
      // is guaranteed to be a no-event
      virtual ApEvent get_domain(Domain &domain, bool need_tight = true) = 0;
      virtual void tighten_index_space(void) = 0;
      virtual bool check_empty(void) = 0;
      virtual size_t get_volume(void) = 0;
      virtual void pack_expression(Serializer &rez, AddressSpaceID target) = 0;
      virtual void pack_expression_value(Serializer &rez,
                                         AddressSpaceID target) = 0;
    public:
#ifdef DEBUG_LEGION
      virtual bool is_valid(void) 
        { return DistributedCollectable::is_global(); }
#endif
      virtual DistributedID get_distributed_id(void) const { return did; }
      virtual void add_canonical_reference(DistributedID source);
      virtual bool remove_canonical_reference(DistributedID source);
      virtual bool try_add_live_reference(void);
      virtual void add_base_expression_reference(ReferenceSource source,
                                                 unsigned count = 1);
      virtual void add_nested_expression_reference(DistributedID source,
                                                   unsigned count = 1);
      virtual bool remove_base_expression_reference(ReferenceSource source,
                                                    unsigned count = 1);
      virtual bool remove_nested_expression_reference(DistributedID source,
                                                      unsigned count = 1);
      virtual void add_tree_expression_reference(DistributedID source,
                                                 unsigned count = 1);
      virtual bool remove_tree_expression_reference(DistributedID source,
                                                    unsigned count = 1);
    public:
      virtual bool invalidate_operation(void) = 0;
      virtual void remove_operation(void) = 0;
      virtual IndexSpaceNode* create_node(IndexSpace handle, DistributedID did,
          RtEvent initialized, Provenance *provenance,
          CollectiveMapping *mapping, IndexSpaceExprID expr_id = 0) = 0;
    public:
      RegionTreeForest *const context;
      IndexSpaceOperation *const origin_expr;
      const OperationKind op_kind;
    protected:
      mutable LocalLock inter_lock;
      std::atomic<int> invalidated;
    };

    template<int DIM, typename T>
    class IndexSpaceOperationT : public IndexSpaceOperation {
    public:
      IndexSpaceOperationT(OperationKind kind, RegionTreeForest *ctx);
      IndexSpaceOperationT(RegionTreeForest *ctx, IndexSpaceExprID eid,
          DistributedID did, IndexSpaceOperation *op,
          TypeTag tag, Deserializer &derez);
      virtual ~IndexSpaceOperationT(void);
    public:
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag,
                                           bool need_tight_result);
      virtual bool is_sparse();
      // If you ask for a tight index space you don't need to pay 
      // attention to the event returned as a precondition as it 
      // is guaranteed to be a no-event
      virtual ApEvent get_domain(Domain &domain, bool need_tight = true);
      virtual void tighten_index_space(void);
      virtual bool check_empty(void);
      virtual size_t get_volume(void);
      virtual void pack_expression(Serializer &rez, AddressSpaceID target);
      virtual void pack_expression_value(Serializer &rez,
                                         AddressSpaceID target) = 0;
      virtual bool invalidate_operation(void) = 0;
      virtual void remove_operation(void) = 0;
      virtual IndexSpaceNode* create_node(IndexSpace handle, DistributedID did,
          RtEvent initialized, Provenance *provenance,
          CollectiveMapping *mapping, IndexSpaceExprID expr_id = 0);
      virtual PieceIteratorImpl* create_piece_iterator(const void *piece_list,
                      size_t piece_list_size, IndexSpaceNode *privilege_node);
    public:
      virtual ApEvent issue_fill(Operation *op,
                           const PhysicalTraceInfo &trace_info,
                           const std::vector<CopySrcDstField> &dst_fields,
                           const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                           UniqueID fill_uid,
                           FieldSpace handle,
                           RegionTreeID tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard,
                           LgEvent unique_event,
                           CollectiveKind collective = COLLECTIVE_NONE,
                           int priority = 0, bool replay = false);
      virtual ApEvent issue_copy(Operation *op,
                           const PhysicalTraceInfo &trace_info,
                           const std::vector<CopySrcDstField> &dst_fields,
                           const std::vector<CopySrcDstField> &src_fields,
                           const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                           RegionTreeID src_tree_id,
                           RegionTreeID dst_tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard,
                           LgEvent src_unique, LgEvent dst_unique,
                           CollectiveKind collective = COLLECTIVE_NONE,
                           int priority = 0, bool replay = false);
      virtual CopyAcrossUnstructured* create_across_unstructured(
                           const std::map<Reservation,bool> &reservations,
                           const bool compute_preimages);
      virtual Realm::InstanceLayoutGeneric* create_layout(
                           const LayoutConstraintSet &constraints,
                           const std::vector<FieldID> &field_ids,
                           const std::vector<size_t> &field_sizes,
                           bool compact, void **piece_list = NULL, 
                           size_t *piece_list_size = NULL,
                           size_t *num_pieces = NULL);
      virtual IndexSpaceExpression* create_layout_expression(
                           const void *piece_list, size_t piece_list_size);
      virtual bool meets_layout_expression(IndexSpaceExpression *expr,
         bool tight_bounds, const void *piece_list, size_t piece_list_size,
         const Domain *padding_delta);
    public:
      virtual IndexSpaceExpression* find_congruent_expression(
                  std::set<IndexSpaceExpression*> &expressions);
      virtual KDTree* get_sparsity_map_kd_tree(void);
    public:
      ApEvent get_realm_index_space(Realm::IndexSpace<DIM,T> &space,
                                    bool need_tight_result);
    protected:
      Realm::IndexSpace<DIM,T> realm_index_space, tight_index_space;
      ApEvent realm_index_space_ready; 
      RtEvent tight_index_space_ready;
      std::atomic<bool> is_index_space_tight;
    };

    template<int DIM, typename T>
    class IndexSpaceUnion : public IndexSpaceOperationT<DIM,T>,
        public LegionHeapify<IndexSpaceUnion<DIM,T> > {
    public:
      static const AllocationType alloc_type = UNION_EXPR_ALLOC;
    public:
      IndexSpaceUnion(const std::vector<IndexSpaceExpression*> &to_union,
                      RegionTreeForest *context);
      IndexSpaceUnion(const IndexSpaceUnion<DIM,T> &rhs);
      virtual ~IndexSpaceUnion(void);
    public:
      IndexSpaceUnion& operator=(const IndexSpaceUnion &rhs);
    public:
      virtual void pack_expression_value(Serializer &rez,AddressSpaceID target);
      virtual bool invalidate_operation(void);
      virtual void remove_operation(void);
    protected:
      const std::vector<IndexSpaceExpression*> sub_expressions;
    }; 

    class UnionOpCreator : public OperationCreator {
    public:
      UnionOpCreator(RegionTreeForest *f, TypeTag t,
                     const std::vector<IndexSpaceExpression*> &e)
        : OperationCreator(f), type_tag(t), exprs(e) { }
    public:
      template<typename N, typename T>
      static inline void demux(UnionOpCreator *creator)
      {
        creator->produce(new IndexSpaceUnion<N::N,T>(creator->exprs,
                                                     creator->forest));
      }
    public:
      virtual void create_operation(void)
        { NT_TemplateHelper::demux<UnionOpCreator>(type_tag, this); }
    public:
      const TypeTag type_tag;
      const std::vector<IndexSpaceExpression*> &exprs;
    };

    template<int DIM, typename T>
    class IndexSpaceIntersection : public IndexSpaceOperationT<DIM,T>,
        public LegionHeapify<IndexSpaceIntersection<DIM,T> > {
    public:
      static const AllocationType alloc_type = INTERSECTION_EXPR_ALLOC;
    public:
      IndexSpaceIntersection(const std::vector<IndexSpaceExpression*> &to_inter,
                             RegionTreeForest *context);
      IndexSpaceIntersection(const IndexSpaceIntersection &rhs);
      virtual ~IndexSpaceIntersection(void);
    public:
      IndexSpaceIntersection& operator=(const IndexSpaceIntersection &rhs);
    public:
      virtual void pack_expression_value(Serializer &rez,AddressSpaceID target);
      virtual bool invalidate_operation(void);
      virtual void remove_operation(void);
    protected:
      const std::vector<IndexSpaceExpression*> sub_expressions;
    };

    class IntersectionOpCreator : public OperationCreator {
    public:
      IntersectionOpCreator(RegionTreeForest *f, TypeTag t,
                            const std::vector<IndexSpaceExpression*> &e)
        : OperationCreator(f), type_tag(t), exprs(e) { }
    public:
      template<typename N, typename T>
      static inline void demux(IntersectionOpCreator *creator)
      {
        creator->produce(new IndexSpaceIntersection<N::N,T>(creator->exprs,
                                                            creator->forest));
      }
    public:
      virtual void create_operation(void)
        { NT_TemplateHelper::demux<IntersectionOpCreator>(type_tag, this); }
    public:
      const TypeTag type_tag;
      const std::vector<IndexSpaceExpression*> &exprs;
    };

    template<int DIM, typename T>
    class IndexSpaceDifference : public IndexSpaceOperationT<DIM,T>,
        public LegionHeapify<IndexSpaceDifference<DIM,T> > {
    public:
      static const AllocationType alloc_type = DIFFERENCE_EXPR_ALLOC;
    public:
      IndexSpaceDifference(IndexSpaceExpression *lhs,IndexSpaceExpression *rhs,
                           RegionTreeForest *context);
      IndexSpaceDifference(const IndexSpaceDifference &rhs);
      virtual ~IndexSpaceDifference(void);
    public:
      IndexSpaceDifference& operator=(const IndexSpaceDifference &rhs);
    public:
      virtual void pack_expression_value(Serializer &rez,AddressSpaceID target);
      virtual bool invalidate_operation(void);
      virtual void remove_operation(void);
    protected:
      IndexSpaceExpression *const lhs;
      IndexSpaceExpression *const rhs;
    };

    class DifferenceOpCreator : public OperationCreator {
    public:
      DifferenceOpCreator(RegionTreeForest *f, TypeTag t,
                          IndexSpaceExpression *l, IndexSpaceExpression *r)
        : OperationCreator(f), type_tag(t), lhs(l), rhs(r) { }
    public:
      template<typename N, typename T>
      static inline void demux(DifferenceOpCreator *creator)
      {
        creator->produce(new IndexSpaceDifference<N::N,T>(creator->lhs,
                                          creator->rhs, creator->forest));
      }
    public:
      virtual void create_operation(void)
        { NT_TemplateHelper::demux<DifferenceOpCreator>(type_tag, this); }
    public:
      const TypeTag type_tag;
      IndexSpaceExpression *const lhs;
      IndexSpaceExpression *const rhs;
    };

    /**
     * \class InstanceExpression 
     * This class stores an expression corresponding to the
     * rectangles that represent a physical instance
     */
    template<int DIM, typename T>
    class InstanceExpression : public IndexSpaceOperationT<DIM,T>,
        public LegionHeapify<InstanceExpression<DIM,T> > {
    public:
      static const AllocationType alloc_type = INSTANCE_EXPR_ALLOC;
    public:
      InstanceExpression(const Rect<DIM,T> *rects, size_t num_rects,
                         RegionTreeForest *context);
      InstanceExpression(const InstanceExpression<DIM,T> &rhs);
      virtual ~InstanceExpression(void);
    public:
      InstanceExpression& operator=(const InstanceExpression &rhs);
    public:
      virtual void pack_expression_value(Serializer &rez,AddressSpaceID target);
      virtual bool invalidate_operation(void);
      virtual void remove_operation(void);
    };

    class InstanceExpressionCreator
    {
    public:
      InstanceExpressionCreator(TypeTag t, const Domain &dom)
        :type_tag(t), dom(dom) { }

      virtual void create_operation()
      {
        NT_TemplateHelper::demux<InstanceExpressionCreator>(type_tag, this);
      }

      static RegionTreeForest *forest();

      template<typename N, typename T>
      static inline void demux(InstanceExpressionCreator *creator)
      {
        Rect<N::N, T> rect = creator->dom;
        creator->result = new InstanceExpression<N::N, T>(&rect, 1,forest());
      }

      static IndexSpaceOperation *create_with_domain(TypeTag tag,
                                                     const Domain &dom);
    public:
      const TypeTag type_tag;
      const Domain dom;
      IndexSpaceOperation *result;
    };

    /**
     * \class RemoteExpression
     * A copy of an expression that lives on a remote node.
     */
    template<int DIM, typename T>
    class RemoteExpression : public IndexSpaceOperationT<DIM,T>,
        public LegionHeapify<RemoteExpression<DIM,T> > {
    public:
      static const AllocationType alloc_type = REMOTE_EXPR_ALLOC;
    public:
      RemoteExpression(RegionTreeForest *context, IndexSpaceExprID eid,
          DistributedID did, IndexSpaceOperation *op,
          TypeTag type_tag, Deserializer &derez);
      RemoteExpression(const RemoteExpression<DIM,T> &rhs);
      virtual ~RemoteExpression(void);
    public:
      RemoteExpression& operator=(const RemoteExpression &op);
    public:
      virtual void pack_expression_value(Serializer &rez,AddressSpaceID target);
      virtual bool invalidate_operation(void);
      virtual void remove_operation(void);
    };

    class RemoteExpressionCreator {
    public:
      RemoteExpressionCreator(RegionTreeForest *f, TypeTag t, Deserializer &d)
        : forest(f), type_tag(t), derez(d), operation(NULL) { }
    public:
      template<typename N, typename T>
      static inline void demux(RemoteExpressionCreator *creator)
      {
        IndexSpaceExprID expr_id;
        creator->derez.deserialize(expr_id);
        DistributedID did;
        creator->derez.deserialize(did);
        IndexSpaceOperation *origin;
        creator->derez.deserialize(origin);
#ifdef DEBUG_LEGION
        assert(creator->operation == NULL);
#endif
        creator->operation =
            new RemoteExpression<N::N,T>(creator->forest, expr_id, did,
                            origin, creator->type_tag, creator->derez);
      }
    public:
      RegionTreeForest *const forest;
      const TypeTag type_tag;
      Deserializer &derez;
      IndexSpaceOperation *operation;
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
                         IndexSpaceExpression *op = NULL);
      ExpressionTrieNode(const ExpressionTrieNode &rhs);
      ~ExpressionTrieNode(void);
    public:
      ExpressionTrieNode& operator=(const ExpressionTrieNode &rhs);
    public:
      bool find_operation(
          const std::vector<IndexSpaceExpression*> &expressions,
          IndexSpaceExpression *&result, ExpressionTrieNode *&last);
      IndexSpaceExpression* find_or_create_operation( 
          const std::vector<IndexSpaceExpression*> &expressions,
          OperationCreator &creator);
      bool remove_operation(const std::vector<IndexSpaceExpression*> &exprs);
    public:
      const unsigned depth;
      const IndexSpaceExprID expr;
    protected:
      IndexSpaceExpression *local_operation;
      std::map<IndexSpaceExprID,IndexSpaceExpression*> operations;
      std::map<IndexSpaceExprID,ExpressionTrieNode*> nodes;
    protected:
      mutable LocalLock trie_lock;
    };

    /**
     * \class IndexTreeNode
     * The abstract base class for nodes in the index space trees.
     */
    class IndexTreeNode : public ValidDistributedCollectable {
    public:
      IndexTreeNode(RegionTreeForest *ctx, unsigned depth,
                    LegionColor color, DistributedID did,
                    RtEvent init_event, CollectiveMapping *mapping,
                    Provenance *provenance, bool tree_valid);
      virtual ~IndexTreeNode(void);
    public:
      virtual IndexTreeNode* get_parent(void) const = 0;
      virtual LegionColor get_colors(std::vector<LegionColor> &colors) = 0;
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
           const void *buffer, size_t size, bool is_mutable, bool local_only);
      bool retrieve_semantic_information(SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail, bool wait_until);
      virtual void send_semantic_request(AddressSpaceID target, 
        SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready) = 0;
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
       const void *buffer, size_t size, bool is_mutable, RtUserEvent ready) = 0;
    public:
      RegionTreeForest *const context;
      const unsigned depth;
      const LegionColor color;
      Provenance *const provenance;
    public:
      RtEvent initialized;
      NodeSet child_creation;
    protected:
      mutable LocalLock node_lock;
    protected:
      std::map<IndexTreeNode*,bool> dominators;
    protected:
      LegionMap<SemanticTag,SemanticInfo> semantic_info;
    };

    /**
     * \class IndexSpaceNode
     * A class for representing a generic index space node.
     */
    class IndexSpaceNode : 
      public IndexTreeNode, public IndexSpaceExpression {
    public:
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
                       std::atomic<IndexPartitionID> *tar,
                       RtUserEvent trig, AddressSpaceID src)
          : LgTaskArgs<DeferChildArgs>(implicit_provenance),
            proxy_this(proxy), child_color(child), target(tar), 
            to_trigger(trig), source(src) { }
      public:
        IndexSpaceNode *const proxy_this;
        const LegionColor child_color;
        std::atomic<IndexPartitionID> *const target;
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
    public:
      IndexSpaceNode(RegionTreeForest *ctx, IndexSpace handle,
                     IndexPartNode *parent, LegionColor color,
                     DistributedID did,
                     IndexSpaceExprID expr_id, RtEvent initialized,
                     unsigned depth, Provenance *provenance,
                     CollectiveMapping *mapping, bool tree_valid);
      IndexSpaceNode(const IndexSpaceNode &rhs) = delete;
      virtual ~IndexSpaceNode(void);
    public:
      IndexSpaceNode& operator=(const IndexSpaceNode &rhs) = delete;
    public:
      inline bool is_set(void) const { return index_space_set.load(); }
    public:
      virtual void notify_invalid(void);
      virtual void notify_local(void);
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
      virtual LegionColor get_colors(std::vector<LegionColor> &colors);
    public:
      virtual void send_semantic_request(AddressSpaceID target, 
           SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready);
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
          const void *buffer, size_t size, bool is_mutable, RtUserEvent ready);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source,
                            bool can_fail, bool wait_until, RtUserEvent ready);
      static void handle_semantic_request(RegionTreeForest *forest,
                                 Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(RegionTreeForest *forest,
                                 Deserializer &derez, AddressSpaceID source);
    public:
      bool has_color(const LegionColor color);
      LegionColor generate_color(LegionColor suggestion = INVALID_COLOR);
      void release_color(LegionColor color);
      IndexPartNode* get_child(const LegionColor c, 
                               RtEvent *defer = NULL, bool can_fail = false);
      void add_child(IndexPartNode *child);
      void remove_child(const LegionColor c);
      size_t get_num_children(void) const;
      RtEvent get_ready_event(void);
    public:
      bool are_disjoint(LegionColor c1, LegionColor c2); 
      void record_remote_child(IndexPartition pid, LegionColor part_color);
    public:
      void send_node(AddressSpaceID target, bool recurse, bool valid = true);
      void pack_node(Serializer &rez, AddressSpaceID target, 
                     bool recurse, bool valid);
      bool invalidate_root(AddressSpaceID source,
                           std::set<RtEvent> &applied,
                           const CollectiveMapping *mapping);
      static void handle_node_creation(RegionTreeForest *context,
                                       Deserializer &derez, 
                                       AddressSpaceID source);
    public:
      static void handle_node_request(RegionTreeForest *context,
                                      Deserializer &derez);
      static void handle_node_return(RegionTreeForest *context,
                                     Deserializer &derez);
      static void handle_node_child_request(RegionTreeForest *context,
                            Deserializer &derez, AddressSpaceID source);
      static void defer_node_child_request(const void *args);
      static void handle_node_child_response(RegionTreeForest *forest,
                                             Deserializer &derez);
      static void handle_colors_request(RegionTreeForest *context,
                            Deserializer &derez, AddressSpaceID source);
      static void handle_colors_response(Deserializer &derez);
      static void handle_index_space_set(RegionTreeForest *forest,
                           Deserializer &derez, AddressSpaceID source);
      static void handle_generate_color_request(RegionTreeForest *forest,
                           Deserializer &derez, AddressSpaceID source);
      static void handle_generate_color_response(Deserializer &derez);
      static void handle_release_color(RegionTreeForest *forest, 
                                       Deserializer &derez);
    public:
      // From IndexSpaceExpression
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag,
                                           bool need_tight_result) = 0;
      // If you ask for a tight index space you don't need to pay 
      // attention to the event returned as a precondition as it 
      // is guaranteed to be a no-event
      virtual ApEvent get_domain(Domain &domain, bool need_tight = true) = 0;
      
      virtual bool set_domain(const Domain &domain, bool broadcast = false) = 0;
      virtual bool set_bounds(const void *bounds, bool is_domain, 
                              bool inititializing, ApEvent is_ready) = 0;
      virtual bool set_output_union(
            const std::map<DomainPoint,DomainPoint> &sizes) = 0;
      virtual void tighten_index_space(void) = 0;
      virtual bool check_empty(void) = 0;
      virtual void pack_expression(Serializer &rez, AddressSpaceID target);
      virtual void pack_expression_value(Serializer &rez,AddressSpaceID target);
    public:
#ifdef DEBUG_LEGION
      virtual bool is_valid(void) 
        { return ValidDistributedCollectable::is_global(); }
#endif
      virtual DistributedID get_distributed_id(void) const { return did; }
      virtual void add_canonical_reference(DistributedID source);
      virtual bool remove_canonical_reference(DistributedID source);
      virtual bool try_add_live_reference(void);
      virtual void add_base_expression_reference(ReferenceSource source,
                                                 unsigned count = 1);
      virtual void add_nested_expression_reference(DistributedID source,
                                                   unsigned count = 1);
      virtual bool remove_base_expression_reference(ReferenceSource source,
                                                    unsigned count = 1);
      virtual bool remove_nested_expression_reference(DistributedID source,
                                                      unsigned count = 1);
      virtual void add_tree_expression_reference(DistributedID source,
                                                 unsigned count = 1);
      virtual bool remove_tree_expression_reference(DistributedID source,
                                                    unsigned count = 1);
    public:
      virtual IndexSpaceNode* create_node(IndexSpace handle, DistributedID did,
          RtEvent initialized,Provenance *provenance,
          CollectiveMapping *mapping, IndexSpaceExprID expr_id = 0) = 0;
      virtual PieceIteratorImpl* create_piece_iterator(const void *piece_list,
                    size_t piece_list_size, IndexSpaceNode *privilege_node) = 0;
      virtual bool is_below_in_tree(IndexPartNode *p, LegionColor &child) const;
    public:
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
      virtual bool contains_point(const DomainPoint &point) = 0;
    public:
      virtual LegionColor get_max_linearized_color(void) = 0;
      virtual LegionColor linearize_color(const DomainPoint &point) = 0;
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
      virtual size_t compute_color_offset(LegionColor color) = 0;
    public:
      bool intersects_with(IndexSpaceNode *rhs,bool compute = true);
      bool intersects_with(IndexPartNode *rhs, bool compute = true);
      bool dominates(IndexSpaceNode *rhs);
    public:
      virtual void pack_index_space(Serializer &rez, 
                                    bool include_size) const = 0;
      virtual bool unpack_index_space(Deserializer &derez,
                                      AddressSpaceID source) = 0;
    public:
      virtual ApEvent create_equal_children(Operation *op,
                                            IndexPartNode *partition, 
                                            size_t granularity) = 0;
      virtual ApEvent create_by_union(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *left,
                                      IndexPartNode *right) = 0;
      virtual ApEvent create_by_intersection(Operation *op,
                                             IndexPartNode *partition,
                                             IndexPartNode *left,
                                             IndexPartNode *right) = 0;
      virtual ApEvent create_by_intersection(Operation *op,
                                             IndexPartNode *partition,
                                             // Left is implicit "this"
                                             IndexPartNode *right,
                                             const bool dominates = false) = 0;
      virtual ApEvent create_by_difference(Operation *op,
                                           IndexPartNode *partition,
                                           IndexPartNode *left,
                                           IndexPartNode *right) = 0;
      // Called on color space and not parent
      virtual ApEvent create_by_restriction(IndexPartNode *partition,
                                            const void *transform,
                                            const void *extent,
                                            int partition_dim) = 0;
      virtual ApEvent create_by_domain(Operation *op,
                                       IndexPartNode *partition,
                 const std::map<DomainPoint,FutureImpl*> &futures,
                                       const Domain &future_map_domain,
                                       bool perform_intersections) = 0;
      virtual ApEvent create_by_weights(Operation *op,
                                        IndexPartNode *partition,
                const std::map<DomainPoint,FutureImpl*> &weights,
                                        size_t granularity) = 0;
      virtual ApEvent create_by_field(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                const std::vector<FieldDataDescriptor> &instances,
                      std::vector<DeppartResult> *results,
                                      ApEvent instances_ready) = 0;
      virtual ApEvent create_by_image(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                    std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready) = 0;
      virtual ApEvent create_by_image_range(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                    std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready) = 0;
      virtual ApEvent create_by_preimage(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                const std::map<DomainPoint,Domain> *remote_targets,
                      std::vector<DeppartResult> *results,
                                      ApEvent instances_ready) = 0;
      virtual ApEvent create_by_preimage_range(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                const std::map<DomainPoint,Domain> *remote_targets,
                      std::vector<DeppartResult> *results,
                                      ApEvent instances_ready) = 0;
      virtual ApEvent create_association(Operation *op, FieldID fid,
                                      IndexSpaceNode *range,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready) = 0;
      virtual size_t get_coordinate_size(bool range) const = 0;
    public:
      virtual PhysicalInstance create_file_instance(const char *file_name,
                                   const Realm::ProfilingRequestSet &requests,
				   const std::vector<Realm::FieldID> &field_ids,
                                   const std::vector<size_t> &field_sizes,
                                   legion_file_mode_t file_mode,
                                   ApEvent &ready_event) = 0;
      virtual PhysicalInstance create_hdf5_instance(const char *file_name,
                                   const Realm::ProfilingRequestSet &requests,
                                   const std::vector<Realm::FieldID> &field_ids,
                                   const std::vector<size_t> &field_sizes,
                                   const std::vector<const char*> &field_files,
                                   const OrderingConstraint &dimension_order,
                                   bool read_only, ApEvent &ready_event) = 0;
    public:
      virtual void validate_slicing(const std::vector<IndexSpace> &slice_spaces,
                                    MultiTask *task, MapperManager *mapper) = 0;
      virtual void log_launch_space(UniqueID op_id) = 0;
      virtual IndexSpace create_shard_space(ShardingFunction *func, 
                                            ShardID shard,
                                            IndexSpace shard_space,
                                            const Domain &shard_domain,
                              const std::vector<DomainPoint> &shard_points,
                                            Provenance *provenance) = 0;
      virtual void compute_range_shards(ShardingFunction *func,
                                        IndexSpace shard_space,
                              const std::vector<DomainPoint> &shard_points,
                                        const Domain &shard_domain,
                                        std::set<ShardID> &range_shards) = 0;
    public:
      const IndexSpace handle;
      IndexPartNode *const parent;
    protected:
      // Must hold the node lock when accessing these data structures
      std::map<LegionColor,IndexPartNode*> color_map;
      std::map<LegionColor,IndexPartition> remote_colors;
      std::set<RegionNode*> logical_nodes;
      std::set<std::pair<LegionColor,LegionColor> > disjoint_subsets;
      std::set<std::pair<LegionColor,LegionColor> > aliased_subsets;
    protected:
      static constexpr uintptr_t REMOVED_CHILD = 0xdead;
      Color                     next_uncollected_color;
      // Event for when the sparsity map is ready
      ApEvent                   index_space_valid;
      // Event to signal for anything waiting for the index space to be set
      RtUserEvent               index_space_ready;
      std::atomic<bool>         index_space_set;
      std::atomic<bool>         index_space_tight;
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
                      DistributedID did,
                      IndexSpaceExprID expr_id, RtEvent init,
                      unsigned depth, Provenance *provenance,
                      CollectiveMapping *mapping, bool tree_valid);
      IndexSpaceNodeT(const IndexSpaceNodeT &rhs) = delete;
      virtual ~IndexSpaceNodeT(void);
    public:
      IndexSpaceNodeT& operator=(const IndexSpaceNodeT &rhs) = delete;
    public:
      ApEvent get_realm_index_space(Realm::IndexSpace<DIM,T> &result,
				    bool need_tight_result);
      bool set_realm_index_space(const Realm::IndexSpace<DIM,T> &value,
                                 ApEvent valid, bool initialization = false,
                                 bool broadcast = false, 
                                 AddressSpaceID source = UINT_MAX);
      RtEvent get_realm_index_space_ready(bool need_tight_result);
    public:
      // From IndexSpaceExpression
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag,
                                           bool need_tight_result);
      virtual bool is_sparse();
      // If you ask for a tight index space you don't need to pay 
      // attention to the event returned as a precondition as it 
      // is guaranteed to be a no-event
      virtual ApEvent get_domain(Domain &domain, bool need_tight);
      virtual bool set_domain(const Domain &domain, bool broadcast = false);
      virtual bool set_bounds(const void *bounds, bool is_domain, 
                              bool inititializing, ApEvent is_ready);
      virtual bool set_output_union(
                const std::map<DomainPoint,DomainPoint> &sizes);
      virtual void tighten_index_space(void);
      virtual bool check_empty(void);
      virtual IndexSpaceNode* create_node(IndexSpace handle, DistributedID did,
          RtEvent initialized,Provenance *provenance,
          CollectiveMapping *mapping, IndexSpaceExprID expr_id = 0);
      virtual PieceIteratorImpl* create_piece_iterator(const void *piece_list,
                      size_t piece_list_size, IndexSpaceNode *privilege_node);
    public:
      void log_index_space_points(const Realm::IndexSpace<DIM,T> &space) const;
      void log_profiler_index_space_points(
                            const Realm::IndexSpace<DIM,T> &tight_space) const;
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
      virtual bool contains_point(const DomainPoint &point);
    public:
      virtual LegionColor get_max_linearized_color(void);
      virtual LegionColor linearize_color(const DomainPoint &point);
      virtual LegionColor linearize_color(const void *realm_color,
                                          TypeTag type_tag);
      LegionColor linearize_color(const Point<DIM,T> &color); 
      virtual void delinearize_color(LegionColor color, 
                                     void *realm_color, TypeTag type_tag);
      void delinearize_color(LegionColor color, Point<DIM,T> &point);
      virtual bool contains_color(LegionColor color,
                                  bool report_error = false);
      virtual void instantiate_colors(std::vector<LegionColor> &colors);
      virtual Domain get_color_space_domain(void);
      virtual DomainPoint get_domain_point_color(void) const;
      virtual DomainPoint delinearize_color_to_point(LegionColor c);
      virtual size_t compute_color_offset(LegionColor color);
    public:
      virtual void pack_index_space(Serializer &rez, bool include_size) const;
      virtual bool unpack_index_space(Deserializer &derez,
                                      AddressSpaceID source);
    public:
      virtual ApEvent create_equal_children(Operation *op,
                                            IndexPartNode *partition, 
                                            size_t granularity);
      virtual ApEvent create_by_union(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *left,
                                      IndexPartNode *right);
      virtual ApEvent create_by_intersection(Operation *op,
                                             IndexPartNode *partition,
                                             IndexPartNode *left,
                                             IndexPartNode *right);
      virtual ApEvent create_by_intersection(Operation *op,
                                             IndexPartNode *partition,
                                             // Left is implicit "this"
                                             IndexPartNode *right,
                                             const bool dominates = false);
      virtual ApEvent create_by_difference(Operation *op,
                                           IndexPartNode *partition,
                                           IndexPartNode *left,
                                           IndexPartNode *right);
      // Called on color space and not parent
      virtual ApEvent create_by_restriction(IndexPartNode *partition,
                                            const void *transform,
                                            const void *extent,
                                            int partition_dim);
      template<int N>
      ApEvent create_by_restriction_helper(IndexPartNode *partition,
                                   const Realm::Matrix<N,DIM,T> &transform,
                                   const Realm::Rect<N,T> &extent);
      virtual ApEvent create_by_domain(Operation *op,
                                       IndexPartNode *partition,
               const std::map<DomainPoint,FutureImpl*> &futures,
                                       const Domain &future_map_domain,
                                       bool perform_intersections);
      template<int COLOR_DIM, typename COLOR_T>
      ApEvent create_by_domain_helper(Operation *op,
                                      IndexPartNode *partition,
                const std::map<DomainPoint,FutureImpl*> &futures,
                                      const Domain &future_map_domain,
                                      bool perform_intersections);
      virtual ApEvent create_by_weights(Operation *op,
                                        IndexPartNode *partition,
                const std::map<DomainPoint,FutureImpl*> &weights,
                                        size_t granularity);
      template<int COLOR_DIM, typename COLOR_T>
      ApEvent create_by_weight_helper(Operation *op,
                                      IndexPartNode *partition,
              const std::map<DomainPoint,FutureImpl*> &weights,
                                      size_t granularity);
      virtual ApEvent create_by_field(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                const std::vector<FieldDataDescriptor> &instances,
                      std::vector<DeppartResult> *results,
                                      ApEvent instances_ready);
      template<int COLOR_DIM, typename COLOR_T>
      ApEvent create_by_field_helper(Operation *op, FieldID fid,
                                     IndexPartNode *partition,
                const std::vector<FieldDataDescriptor> &instances,
                      std::vector<DeppartResult> *results,
                                     ApEvent instances_ready);
      virtual ApEvent create_by_image(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                    std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      template<int DIM2, typename T2>
      ApEvent create_by_image_helper(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                    std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      virtual ApEvent create_by_image_range(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                    std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      template<int DIM2, typename T2>
      ApEvent create_by_image_range_helper(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                    std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      virtual ApEvent create_by_preimage(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                const std::map<DomainPoint,Domain> *remote_targets,
                      std::vector<DeppartResult> *results,
                                      ApEvent instances_ready);
      template<int DIM2, typename T2>
      ApEvent create_by_preimage_helper(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                const std::map<DomainPoint,Domain> *remote_targets,
                      std::vector<DeppartResult> *results,
                                      ApEvent instances_ready);
      virtual ApEvent create_by_preimage_range(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                const std::map<DomainPoint,Domain> *remote_targets,
                      std::vector<DeppartResult> *results,
                                      ApEvent instances_ready);
      template<int DIM2, typename T2>
      ApEvent create_by_preimage_range_helper(Operation *op, FieldID fid,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                const std::map<DomainPoint,Domain> *remote_targets,
                      std::vector<DeppartResult> *results,
                                      ApEvent instances_ready);
      virtual ApEvent create_association(Operation *op, FieldID fid,
                                      IndexSpaceNode *range,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      template<int DIM2, typename T2>
      ApEvent create_association_helper(Operation *op, FieldID fid,
                                      IndexSpaceNode *range,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      virtual size_t get_coordinate_size(bool range) const;
    public:
      virtual PhysicalInstance create_file_instance(const char *file_name,
                                   const Realm::ProfilingRequestSet &requests,
                                   const std::vector<Realm::FieldID> &field_ids,
                                   const std::vector<size_t> &field_sizes,
                                   legion_file_mode_t file_mode, 
                                   ApEvent &ready_event);
      virtual PhysicalInstance create_hdf5_instance(const char *file_name,
                                   const Realm::ProfilingRequestSet &requests,
                                   const std::vector<Realm::FieldID> &field_ids,
                                   const std::vector<size_t> &field_sizes,
                                   const std::vector<const char*> &field_files,
                                   const OrderingConstraint &dimension_order,
                                   bool read_only, ApEvent &ready_event);
    public:
      virtual ApEvent issue_fill(Operation *op,
                           const PhysicalTraceInfo &trace_info,
                           const std::vector<CopySrcDstField> &dst_fields,
                           const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                           UniqueID fill_uid,
                           FieldSpace handle,
                           RegionTreeID tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard,
                           LgEvent unique_event,
                           CollectiveKind collective = COLLECTIVE_NONE,
                           int priority = 0, bool replay = false);
      virtual ApEvent issue_copy(Operation *op,
                           const PhysicalTraceInfo &trace_info,
                           const std::vector<CopySrcDstField> &dst_fields,
                           const std::vector<CopySrcDstField> &src_fields,
                           const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                           RegionTreeID src_tree_id,
                           RegionTreeID dst_tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard,
                           LgEvent src_unique, LgEvent dst_unique,
                           CollectiveKind collective = COLLECTIVE_NONE,
                           int priority = 0, bool replay = false);
      virtual CopyAcrossUnstructured* create_across_unstructured(
                           const std::map<Reservation,bool> &reservations,
                           const bool compute_preimages);
      virtual Realm::InstanceLayoutGeneric* create_layout(
                           const LayoutConstraintSet &constraints,
                           const std::vector<FieldID> &field_ids,
                           const std::vector<size_t> &field_sizes,
                           bool compact, void **piece_list = NULL, 
                           size_t *piece_list_size = NULL,
                           size_t *num_pieces = NULL);
      virtual IndexSpaceExpression* create_layout_expression(
                           const void *piece_list, size_t piece_list_size);
      virtual bool meets_layout_expression(IndexSpaceExpression *expr,
         bool tight_bounds, const void *piece_list, size_t piece_list_size,
         const Domain *padding_delta);
    public:
      virtual IndexSpaceExpression* find_congruent_expression(
                  std::set<IndexSpaceExpression*> &expressions);
      virtual KDTree* get_sparsity_map_kd_tree(void);
    public:
      virtual void validate_slicing(const std::vector<IndexSpace> &slice_spaces,
                                    MultiTask *task, MapperManager *mapper);
      virtual void log_launch_space(UniqueID op_id);
      virtual IndexSpace create_shard_space(ShardingFunction *func, 
                                            ShardID shard, 
                                            IndexSpace shard_space,
                                            const Domain &shard_domain,
                                  const std::vector<DomainPoint> &shard_points,
                                            Provenance *provenance);
      virtual void compute_range_shards(ShardingFunction *func,
                                        IndexSpace shard_space,
                                  const std::vector<DomainPoint> &shard_points,
                                        const Domain &shard_domain,
                                        std::set<ShardID> &range_shards);
    public:
      bool contains_point(const Point<DIM,T> &point);
    protected:
      ColorSpaceLinearizationT<DIM,T>* compute_linearization_metadata(void);
    protected:
      DomainT<DIM,T> realm_index_space;
    protected:
      std::atomic<ColorSpaceLinearizationT<DIM,T>*> linearization;
    public:
      struct CreateByDomainHelper {
      public:
        CreateByDomainHelper(IndexSpaceNodeT<DIM,T> *n,
                             IndexPartNode *p, Operation *o,
                             const std::map<DomainPoint,FutureImpl*> &fts,
                             const Domain &domain, bool inter)
          : node(n), partition(p), op(o), futures(fts), 
            future_map_domain(domain), intersect(inter) { }
      public:
        template<typename COLOR_DIM, typename COLOR_T>
        static inline void demux(CreateByDomainHelper *creator)
        {
          creator->result = creator->node->template 
            create_by_domain_helper<COLOR_DIM::N,COLOR_T>(creator->op,
                creator->partition, creator->futures, 
                creator->future_map_domain, creator->intersect);
        }
      public:
        IndexSpaceNodeT<DIM,T> *const node;
        IndexPartNode *const partition;
        Operation *const op;
        const std::map<DomainPoint,FutureImpl*> &futures;
        const Domain &future_map_domain;
        const bool intersect;
        ApEvent result;
      };
      struct CreateByWeightHelper {
      public:
        CreateByWeightHelper(IndexSpaceNodeT<DIM,T> *n,
                             IndexPartNode *p, Operation *o,
                             const std::map<DomainPoint,FutureImpl*> &wts,
                             size_t g)
          : node(n), partition(p), op(o), weights(wts), granularity(g) { }
      public:
        template<typename COLOR_DIM, typename COLOR_T>
        static inline void demux(CreateByWeightHelper *creator)
        {
          creator->result = creator->node->template 
            create_by_weight_helper<COLOR_DIM::N,COLOR_T>(creator->op,
                creator->partition, creator->weights, creator->granularity);
        }
      public:
        IndexSpaceNodeT<DIM,T> *const node;
        IndexPartNode *const partition;
        Operation *const op;
        const std::map<DomainPoint,FutureImpl*> &weights;
        const size_t granularity;
        ApEvent result;
      };
      struct CreateByFieldHelper {
      public:
        CreateByFieldHelper(IndexSpaceNodeT<DIM,T> *n, 
                            Operation *o, FieldID f, IndexPartNode *p,
                            const std::vector<FieldDataDescriptor> &i,
                            std::vector<DeppartResult> *res, ApEvent r)
          : node(n), op(o), partition(p), instances(i), 
            results(res), ready(r), fid(f) { }
      public:
        template<typename COLOR_DIM, typename COLOR_T>
        static inline void demux(CreateByFieldHelper *creator)
        {
          creator->result = 
           creator->node->template create_by_field_helper<COLOR_DIM::N,COLOR_T>(
                         creator->op, creator->fid, creator->partition,
                         creator->instances, creator->results, creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM,T> *node;
        Operation *op;
        IndexPartNode *partition;
        const std::vector<FieldDataDescriptor> &instances;
        std::vector<DeppartResult> *const results;
        ApEvent ready, result;
        FieldID fid;
      };
      struct CreateByImageHelper {
      public:
        CreateByImageHelper(IndexSpaceNodeT<DIM,T> *n, Operation *o,
                            FieldID f, IndexPartNode *p, IndexPartNode *j,
                            std::vector<FieldDataDescriptor> &i,
                            ApEvent r)
          : node(n), op(o), partition(p), projection(j), 
            instances(i), ready(r), fid(f) { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByImageHelper *creator)
        {
          creator->result = 
           creator->node->template create_by_image_helper<DIM2::N,T2>(
               creator->op, creator->fid, creator->partition,
               creator->projection, creator->instances, creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM,T> *node;
        Operation *op;
        IndexPartNode *partition;
        IndexPartNode *projection;
        std::vector<FieldDataDescriptor> &instances;
        ApEvent ready, result;
        FieldID fid;
      };
      struct CreateByImageRangeHelper {
      public:
        CreateByImageRangeHelper(IndexSpaceNodeT<DIM,T> *n, Operation *o,
                            FieldID f, IndexPartNode *p, IndexPartNode *j,
                            std::vector<FieldDataDescriptor> &i,
                            ApEvent r)
          : node(n), op(o), partition(p), projection(j), 
            instances(i), ready(r), fid(f) { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByImageRangeHelper *creator)
        {
          creator->result = creator->node->template 
            create_by_image_range_helper<DIM2::N,T2>(
               creator->op, creator->fid, creator->partition,
               creator->projection, creator->instances, creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM,T> *node;
        Operation *op;
        IndexPartNode *partition;
        IndexPartNode *projection;
        std::vector<FieldDataDescriptor> &instances;
        ApEvent ready, result;
        FieldID fid;
      };
      struct CreateByPreimageHelper {
      public:
        CreateByPreimageHelper(IndexSpaceNodeT<DIM,T> *n, Operation *o, 
                            FieldID f, IndexPartNode *p, IndexPartNode *j,
                            const std::vector<FieldDataDescriptor> &i,
                            const std::map<DomainPoint,Domain> *t,
                            std::vector<DeppartResult> *res, ApEvent r)
          : node(n), op(o), partition(p), projection(j), instances(i),
            remote_targets(t), results(res), ready(r), fid(f) { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByPreimageHelper *creator)
        {
          creator->result = 
           creator->node->template create_by_preimage_helper<DIM2::N,T2>(
               creator->op, creator->fid, creator->partition,
               creator->projection, creator->instances,
               creator->remote_targets, creator->results, creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM,T> *node;
        Operation *op;
        IndexPartNode *partition;
        IndexPartNode *projection;
        const std::vector<FieldDataDescriptor> &instances;
        const std::map<DomainPoint,Domain> *const remote_targets;
        std::vector<DeppartResult> *const results;
        ApEvent ready, result;
        FieldID fid;
      };
      struct CreateByPreimageRangeHelper {
      public:
        CreateByPreimageRangeHelper(IndexSpaceNodeT<DIM,T> *n, Operation *o,
                            FieldID f, IndexPartNode *p, IndexPartNode *j,
                            const std::vector<FieldDataDescriptor> &i,
                            const std::map<DomainPoint,Domain> *t,
                            std::vector<DeppartResult> *res, ApEvent r)
          : node(n), op(o), partition(p), projection(j), instances(i),
            remote_targets(t), results(res), ready(r), fid(f) { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByPreimageRangeHelper *creator)
        {
          creator->result = creator->node->template 
            create_by_preimage_range_helper<DIM2::N,T2>(
               creator->op, creator->fid, creator->partition,
               creator->projection, creator->instances,
               creator->remote_targets, creator->results, creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM,T> *node;
        Operation *op;
        IndexPartNode *partition;
        IndexPartNode *projection;
        const std::vector<FieldDataDescriptor> &instances;
        const std::map<DomainPoint,Domain> *const remote_targets;
        std::vector<DeppartResult> *const results;
        ApEvent ready, result;
        FieldID fid;
      };
      struct CreateAssociationHelper {
      public:
        CreateAssociationHelper(IndexSpaceNodeT<DIM,T> *n,
                            Operation *o, FieldID f, IndexSpaceNode *g,
                            const std::vector<FieldDataDescriptor> &i,
                            ApEvent r)
          : node(n), op(o), range(g), instances(i), ready(r), fid(f) { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateAssociationHelper *creator)
        {
          creator->result = creator->node->template 
            create_association_helper<DIM2::N,T2>(
               creator->op, creator->fid, creator->range, 
               creator->instances, creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM,T> *node;
        Operation *op;
        IndexSpaceNode *range;
        const std::vector<FieldDataDescriptor> &instances;
        ApEvent ready, result;
        FieldID fid;
      };
    };

    /**
     * \class ColorSpaceLinearization
     * A color space linearation maps N-D color spaces to an
     * (almost) contiguous 1-D space that can be traversed by
     * the runtime with good locality between the points in 
     * N dimensions. It does this using generalized N-D Morton
     * curves. There are some catches though that prevent us 
     * from just using one big Morton curve in most cases. 
     * The first problem is that Morton curves must be done
     * on hypercubes with powers of 2 dimensions, which means we
     * either need to under approximate most rectangles in 
     * the color space. The second problem is in higher that
     * in higher dimenstions we might end up exceeding the
     * maximum number of bits we can use to represent the 
     * Morton curve since we only have 64-bits in our the
     * LegionColor type. We therefore often will end up tiling
     * the color space to meet these constraints.
     */
    template<int DIM, typename T>
    class ColorSpaceLinearizationT {
    public:
      class MortonTile {
      public:
        MortonTile(const Rect<DIM,T> &b, unsigned count, 
                    const int dims[DIM], unsigned order)
          : bounds(b), interesting_count(count), morton_order(order), index(0)
        {
          for (unsigned idx = 0; idx < DIM; idx++)
            interesting_dims[idx] = dims[idx];
        }
      public:
        LegionColor get_max_linearized_color(void) const;
        LegionColor linearize(const Point<DIM,T> &point) const;
        void delinearize(LegionColor color, Point<DIM,T> &point) const;
        bool contains_color(LegionColor color) const;
        size_t compute_color_offset(LegionColor color) const;
      public:
        Rect<DIM,T> bounds;
        int interesting_dims[DIM];
        unsigned interesting_count;
        unsigned morton_order;
        unsigned index;
      };
    public:
      ColorSpaceLinearizationT(const DomainT<DIM,T> &domain); 
      ColorSpaceLinearizationT(const ColorSpaceLinearizationT &rhs) = delete;
      ~ColorSpaceLinearizationT(void);
    public:
      ColorSpaceLinearizationT& operator=(
                              const ColorSpaceLinearizationT &rhs) = delete;
    public:
      LegionColor get_max_linearized_color(void) const;
      LegionColor linearize(const Point<DIM,T> &point) const;
      void delinearize(LegionColor color, Point<DIM,T> &point) const;
      bool contains_color(LegionColor color) const;
      size_t compute_color_offset(LegionColor color) const;
    protected:
      // Bounds of a rectangle contained in the color space
      std::vector<MortonTile*> morton_tiles;
      // The starting color for each tile (sorted)
      std::vector<LegionColor> color_offsets;
      // KD-Tree for looking up the owner tile for points
      KDNode<DIM,T,MortonTile*> *kdtree;
    };

    // Specialization for the case of DIM==1 since that is easy
    // No need for any fancy Morton curves here, we can pack
    // all the points nice and densely
    template<typename T>
    class ColorSpaceLinearizationT<1,T> {
    public:
      ColorSpaceLinearizationT(const DomainT<1,T> &domain);
    public:
      LegionColor get_max_linearized_color(void) const;
      LegionColor linearize(const Point<1,T> &point) const;
      void delinearize(LegionColor color, Point<1,T> &point) const;
      bool contains_color(LegionColor color) const;
      size_t compute_color_offset(LegionColor color) const;
    protected:
      // The lo point for each tile (sorted)
      std::vector<T> tiles;
      // Extents of each tile
      std::vector<size_t> extents;
      // The starting color for each tile
      std::vector<LegionColor> color_offsets;
    };

    /**
     * \class ColorSpaceIterator
     * A color space iterator helps iterating over a (subset)
     * of the colors in a color space for a particular partition.
     * It abstracts the details of whether the colors space is
     * sparse or dense and can deal with chunking for sharding.
     */
    class ColorSpaceIterator {
    public:
      ColorSpaceIterator(IndexPartNode *partition, bool local_only = false);
      ColorSpaceIterator(IndexPartNode *partition, 
                         ShardID local_shard, size_t total_shards);
    public:
      operator bool(void) const;
      LegionColor operator*(void) const;
      ColorSpaceIterator& operator++(int/*postfix*/);
      void step(void);
      static LegionColor compute_chunk(LegionColor max_color, 
                                       size_t total_shards);
    private:
      IndexSpaceNode *color_space;
      LegionColor current, end;
      bool simple_step;
    };

    /**
     * \class IndexSpaceCreator
     * A small helper class for creating templated index spaces
     */
    class IndexSpaceCreator {
    public:
      IndexSpaceCreator(RegionTreeForest *f, IndexSpace s, IndexPartNode *p,
                        LegionColor c, DistributedID d, IndexSpaceExprID e,
                        RtEvent init, unsigned dp, Provenance *prov,
                        CollectiveMapping *m, bool valid)
        : forest(f), space(s), parent(p), color(c), did(d), expr_id(e),
          initialized(init), depth(dp), provenance(prov), mapping(m),
          tree_valid(valid), result(NULL) { }
    public:
      template<typename N, typename T>
      static inline void demux(IndexSpaceCreator *creator)
      {
        creator->result = new IndexSpaceNodeT<N::N,T>(creator->forest,
            creator->space, creator->parent, creator->color, creator->did,
            creator->expr_id, creator->initialized, creator->depth,
            creator->provenance, creator->mapping, creator->tree_valid);
      }
    public:
      RegionTreeForest *const forest;
      const IndexSpace space; 
      IndexPartNode *const parent;
      const LegionColor color;
      const DistributedID did;
      const IndexSpaceExprID expr_id;
      const RtEvent initialized;
      const unsigned depth;
      Provenance *const provenance;
      CollectiveMapping *const mapping;
      const bool tree_valid;
      IndexSpaceNode *result;
    };

    /**
     * \class PartitionTracker
     * This is a small helper class that is used for figuring out
     * when to remove references to LogicalPartition objects. We
     * want to remove the references as soon as either the index
     * partition is destroyed or the logical region is destroyed.
     * We use this class to detect which one occurs first.
     */
    class PartitionTracker : public Collectable {
    public:
      PartitionTracker(PartitionNode *part);
      PartitionTracker(const PartitionTracker &rhs);
      ~PartitionTracker(void) { }
    public:
      PartitionTracker& operator=(const PartitionTracker &rhs);
    public:
      bool can_prune(void);
      bool remove_partition_reference(void);
    private:
      PartitionNode *const partition;
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
        DisjointnessArgs(IndexPartNode *proxy) 
          : LgTaskArgs<DisjointnessArgs>(implicit_provenance),
            proxy_this(proxy) { }
      public:
        IndexPartNode *const proxy_this;
      };
    public:
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
                       AddressSpaceID src)
          : LgTaskArgs<DeferChildArgs>(implicit_provenance),
            proxy_this(proxy), child_color(child), source(src) { }
      public:
        IndexPartNode *const proxy_this;
        const LegionColor child_color;
        const AddressSpaceID source;
      };
      class DeferFindShardRects : public LgTaskArgs<DeferFindShardRects> {
      public:
        static const LgTaskID TASK_ID = LG_INDEX_PART_DEFER_SHARD_RECTS_TASK_ID;
      public:
        DeferFindShardRects(IndexPartNode *proxy)
          : LgTaskArgs<DeferFindShardRects>(implicit_provenance),
            proxy_this(proxy) { }
      public:
        IndexPartNode *const proxy_this;
      };
      class RemoteDisjointnessFunctor {
      public:
        RemoteDisjointnessFunctor(Serializer &r, Runtime *rt);
      public:
        void apply(AddressSpaceID target);
      public:
        Serializer &rez;
        Runtime *const runtime;
      };
    protected:
      class InterferenceEntry {
      public:
        InterferenceEntry(void)
          : expr_id(0), older(NULL), newer(NULL) { }
      public:
        std::vector<LegionColor> colors;
        IndexSpaceExprID expr_id;
        InterferenceEntry *older;
        InterferenceEntry *newer;
      };
      class RemoteKDTracker {
      public:
        RemoteKDTracker(Runtime *runtime);
      public:
        RtEvent find_remote_interfering(const std::set<AddressSpaceID> &targets,
                          IndexPartition handle, IndexSpaceExpression *expr);
        void get_remote_interfering(std::set<LegionColor> &colors);
        RtUserEvent process_remote_interfering_response(Deserializer &derez);
      protected:
        mutable LocalLock tracker_lock;
        std::set<LegionColor> remote_colors;
        Runtime *const runtime;
        RtUserEvent done_event;
        std::atomic<unsigned> remaining;
      };
    public:
      IndexPartNode(RegionTreeForest *ctx, IndexPartition p,
                    IndexSpaceNode *par, IndexSpaceNode *color_space,
                    LegionColor c, bool disjoint, int complete,
                    DistributedID did, RtEvent initialized,
                    CollectiveMapping *mapping, Provenance *provenance);
      IndexPartNode(RegionTreeForest *ctx, IndexPartition p,
                    IndexSpaceNode *par, IndexSpaceNode *color_space,
                    LegionColor c, int complete, DistributedID did,
                    RtEvent initialized, CollectiveMapping *mapping,
                    Provenance *provenance);
      IndexPartNode(const IndexPartNode &rhs) = delete;
      virtual ~IndexPartNode(void);
    public:
      IndexPartNode& operator=(const IndexPartNode &rhs) = delete;
    public:
      virtual void notify_invalid(void);
      virtual void notify_local(void);
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
      virtual LegionColor get_colors(std::vector<LegionColor> &colors);
    public:
      virtual void send_semantic_request(AddressSpaceID target, 
           SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready);
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
          const void *buffer, size_t size, bool is_mutable, RtUserEvent ready);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source,
                            bool can_fail, bool wait_until, RtUserEvent ready);
      static void handle_semantic_request(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
    public:
      bool has_color(const LegionColor c);
      AddressSpaceID find_color_creator_space(LegionColor color, 
                                  CollectiveMapping *&child_mapping) const;
      IndexSpaceNode* get_child(const LegionColor c, RtEvent *defer = NULL);
      void add_child(IndexSpaceNode *child);
      void set_child(IndexSpaceNode *child);
      void add_tracker(PartitionTracker *tracker); 
      size_t get_num_children(void) const;
      bool compute_disjointness_and_completeness(void);
      bool update_disjoint_complete_result(uint64_t children_volume,
                                           uint64_t intersection_volume = 0);
      bool update_disjoint_complete_result(
          std::map<LegionColor,uint64_t> &children_volumes,
          std::map<std::pair<LegionColor,LegionColor>,
                   uint64_t> *intersection_volumes = NULL);
      bool finalize_disjoint_complete(void);
      void get_subspace_preconditions(std::set<ApEvent> &preconditions);
    public:
      void initialize_disjoint_complete_notifications(void);
      bool is_disjoint(bool from_app = false, bool false_if_not_ready = false);
      bool are_disjoint(LegionColor c1, LegionColor c2,
                        bool force_compute = false);
      bool is_complete(bool from_app = false, bool false_if_not_ready = false);
      bool handle_disjointness_update(Deserializer &derez);
    public:
      ApEvent create_equal_children(Operation *op, size_t granularity);
      ApEvent create_by_weights(Operation *op, 
          const std::map<DomainPoint,FutureImpl*> &weights, size_t granularity);
      ApEvent create_by_union(Operation *Op,
                              IndexPartNode *left, IndexPartNode *right);
      ApEvent create_by_intersection(Operation *op,
                              IndexPartNode *left, IndexPartNode *right);
      ApEvent create_by_intersection(Operation *op, IndexPartNode *original,
                                     const bool dominates);
      ApEvent create_by_difference(Operation *op,
                              IndexPartNode *left, IndexPartNode *right);
      ApEvent create_by_restriction(const void *transform, const void *extent);
      ApEvent create_by_domain(const std::map<DomainPoint,FutureImpl*> &futures,
                               const Domain &future_map_domain);
    public:
      bool intersects_with(IndexSpaceNode *other, bool compute = true);
      bool intersects_with(IndexPartNode *other, bool compute = true); 
      void find_interfering_children(IndexSpaceExpression *expr,
                                     std::vector<LegionColor> &colors);
      virtual bool find_interfering_children_kd(IndexSpaceExpression *expr,
                 std::vector<LegionColor> &colors, bool local_only = false) = 0;
    public:
      static void handle_disjointness_computation(const void *args, 
                                                  RegionTreeForest *forest);
    public:
      void send_node(AddressSpaceID target, bool recurse);
      void pack_node(Serializer &rez, AddressSpaceID target);
      static void handle_node_creation(RegionTreeForest *context,
                                       Deserializer &derez, 
                                       AddressSpaceID source);
    public:
      static void handle_node_request(RegionTreeForest *context,
                                      Deserializer &derez);
      static void handle_node_return(RegionTreeForest *context,
                                     Deserializer &derez);
      static void handle_node_child_request(
          RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source);
      static void defer_node_child_request(const void *args);
      static void defer_find_local_shard_rects(const void *args);
      static void handle_node_child_response(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_child_replication(RegionTreeForest *forest,
                                           Deserializer &derez);
      static void handle_node_disjoint_update(RegionTreeForest *forest,
                                              Deserializer &derez);
      static void handle_notification(RegionTreeForest *context, 
                                      Deserializer &derez);
    protected:
      RtEvent request_shard_rects(void);
      virtual void initialize_shard_rects(void) = 0;
      virtual bool find_local_shard_rects(void) = 0;
      virtual void pack_shard_rects(Serializer &rez, bool clear) = 0;
      virtual void unpack_shard_rects(Deserializer &derez) = 0;
      bool process_shard_rects_response(Deserializer &derez, AddressSpace src);
      bool perform_shard_rects_notification(void);
    public:
      static void handle_shard_rects_request(RegionTreeForest *forest,
                                             Deserializer &derez);
      static void handle_shard_rects_response(RegionTreeForest *forest,
                                  Deserializer &derez, AddressSpaceID source);
      static void handle_remote_interference_request(RegionTreeForest *forest,
                                  Deserializer &derez, AddressSpaceID source);
      static void handle_remote_interference_response(Deserializer &derez);
    public:
      const IndexPartition handle;
      IndexSpaceNode *const parent;
      IndexSpaceNode *const color_space;
      const LegionColor total_children;
      const LegionColor max_linearized_color;
    protected:
      // Must hold the node lock when accessing these data structures
      // the remaining data structures
      std::map<LegionColor,IndexSpaceNode*> color_map;
      std::map<LegionColor,RtUserEvent> pending_child_map;
      std::set<std::pair<LegionColor,LegionColor> > disjoint_subspaces;
      std::set<std::pair<LegionColor,LegionColor> > aliased_subspaces;
      std::list<PartitionTracker*> partition_trackers;
    protected:
      // Support for computing disjointness locally
      uint64_t total_children_volume, total_intersection_volume;
      std::map<LegionColor,uint64_t> total_children_volumes;
      std::map<std::pair<LegionColor,LegionColor>,
               uint64_t> total_intersection_volumes;
      unsigned remaining_local_disjoint_complete_notifications;
      unsigned remaining_global_disjoint_complete_notifications;
    protected:
      std::atomic<bool> has_disjoint, disjoint;
      std::atomic<bool> has_complete, complete;
      RtUserEvent disjoint_complete_ready;
    protected:
      // Members for the interference cache
      static const size_t MAX_INTERFERENCE_CACHE_SIZE = 64;
      std::map<IndexSpaceExprID,InterferenceEntry> interference_cache;
      InterferenceEntry *first_entry;
    protected:
      // Help for building distributed kd-trees with shard mappings
      RtUserEvent shard_rects_ready;
      unsigned remaining_rect_notifications;
    }; 

    /**
     * \class KDNode
     * A KDNode is used for performing fast interference tests for
     * expressions against rectangles from child subregions in a partition.
     */
    template<int DIM, typename T, typename RT>
    class KDNode {
    public:
      KDNode(const Rect<DIM,T> &bounds,
             std::vector<std::pair<Rect<DIM,T>,RT> > &subrects);
      KDNode(const KDNode &rhs) = delete;
      ~KDNode(void);
    public:
      KDNode& operator=(const KDNode &rhs) = delete;
    public:
      void find_interfering(const Rect<DIM,T> &test,
                            std::set<RT> &interfering) const;
      void record_inorder_traversal(std::vector<RT> &order) const;
      RT find(const Point<DIM,T> &point) const;
    public:
      const Rect<DIM,T> bounds;
    protected:
      KDNode<DIM,T,RT> *left;
      KDNode<DIM,T,RT> *right;
      std::vector<std::pair<Rect<DIM,T>,RT> > rects;
    };
    
    // Specialization for void case
    template<int DIM, typename T>
    class KDNode<DIM,T,void> : public KDTree {
    public:
      KDNode(const Rect<DIM,T> &bounds,
             std::vector<Rect<DIM,T> > &subrects);
      KDNode(const KDNode &rhs) = delete;
      virtual ~KDNode(void);
    public:
      KDNode& operator=(const KDNode &rhs) = delete;
    public:
      size_t count_rectangles(void) const;
      size_t count_intersecting_points(const Rect<DIM,T> &rect) const;
    public:
      const Rect<DIM,T> bounds;
    protected:
      KDNode<DIM,T,void> *left;
      KDNode<DIM,T,void> *right;
      std::vector<Rect<DIM,T> > rects;
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
                     DistributedID did, RtEvent initialized,
                     CollectiveMapping *mapping, Provenance *provenance);
      IndexPartNodeT(RegionTreeForest *ctx, IndexPartition p,
                     IndexSpaceNode *par, IndexSpaceNode *color_space,
                     LegionColor c, int complete, DistributedID did,
                     RtEvent initialized, CollectiveMapping *mapping,
                     Provenance *provenance);
      IndexPartNodeT(const IndexPartNodeT &rhs) = delete;
      virtual ~IndexPartNodeT(void);
    public:
      IndexPartNodeT& operator=(const IndexPartNodeT &rhs) = delete;
    public:
      virtual bool find_interfering_children_kd(IndexSpaceExpression *expr,
                 std::vector<LegionColor> &colors, bool local_only = false);
    protected:
      virtual void initialize_shard_rects(void);
      virtual bool find_local_shard_rects(void);
      virtual void pack_shard_rects(Serializer &rez, bool clear);
      virtual void unpack_shard_rects(Deserializer &derez);
    protected:
      KDNode<DIM,T,LegionColor> *kd_root;
      KDNode<DIM,T,AddressSpaceID> *kd_remote;
      RtUserEvent kd_remote_ready;
    protected:
      // Each color appears exactly once in this data structure
      std::vector<std::pair<Rect<DIM,T>,LegionColor> > *dense_shard_rects;
      // There might be multiple rectangles for each color here
      // These rectangles are just an approximation of the actual
      // points in the children with sparsity maps
      std::vector<std::pair<Rect<DIM,T>,LegionColor> > *sparse_shard_rects;
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
                       RtEvent initialized, 
                       CollectiveMapping *m, Provenance *prov)
        : forest(f), partition(p), parent(par), color_space(cs), color(c),
          has_disjoint(true), disjoint(d), complete(k), did(id),
          init(initialized), mapping(m), provenance(prov) { }
      IndexPartCreator(RegionTreeForest *f, IndexPartition p,
                       IndexSpaceNode *par, IndexSpaceNode *cs,
                       LegionColor c,  int k, DistributedID id,
                       RtEvent initialized,
                       CollectiveMapping *m, Provenance *prov)
        : forest(f), partition(p), parent(par), color_space(cs),
          color(c), has_disjoint(false), disjoint(false), complete(k),
          did(id), init(initialized), mapping(m), provenance(prov) { }
    public:
      template<typename N, typename T>
      static inline void demux(IndexPartCreator *creator)
      {
        if (!creator->has_disjoint)
          creator->result = new IndexPartNodeT<N::N,T>(creator->forest,
              creator->partition, creator->parent, creator->color_space,
              creator->color,  creator->complete, 
              creator->did, creator->init,
              creator->mapping, creator->provenance);
        else
          creator->result = new IndexPartNodeT<N::N,T>(creator->forest,
              creator->partition, creator->parent, creator->color_space,
              creator->color, creator->disjoint, creator->complete,
              creator->did, creator->init,
              creator->mapping, creator->provenance);
      }
    public:
      RegionTreeForest *const forest;
      const IndexPartition partition;
      IndexSpaceNode *const parent;
      IndexSpaceNode *const color_space;
      const LegionColor color;
      const bool has_disjoint;
      const bool disjoint;
      const int complete;
      const DistributedID did;
      const RtEvent init;
      CollectiveMapping *const mapping;
      Provenance *const provenance;
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
      enum FieldAllocationState {
        FIELD_ALLOC_INVALID, // field_infos is invalid
        FIELD_ALLOC_READ_ONLY, // field_infos is valid and read-only
        FIELD_ALLOC_PENDING, // about to have allocation privileges (owner-only)
        FIELD_ALLOC_EXCLUSIVE, // field_infos is valid and can allocate
        FIELD_ALLOC_COLLECTIVE,// same as above but exactly one total CR context
      };
    public:
      struct FieldInfo {
      public:
        FieldInfo(void);
        FieldInfo(size_t size, unsigned id, CustomSerdezID sid,
                  Provenance *prov, bool loc = false, bool collect = false);
        FieldInfo(ApEvent ready, unsigned id, CustomSerdezID sid,
                  Provenance *prov, bool loc = false, bool collect = false);
        FieldInfo(const FieldInfo &rhs);
        FieldInfo(FieldInfo &&rhs);
        ~FieldInfo(void);
      public:
        FieldInfo& operator=(const FieldInfo &rhs);
        FieldInfo& operator=(FieldInfo &&rhs);
      public:
        void serialize(Serializer &rez) const;
        void deserialize(Deserializer &derez);
      public:
        size_t field_size;
        ApEvent size_ready;
        unsigned idx;
        CustomSerdezID serdez_id;
        Provenance *provenance;
        bool collective;
        bool local;
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
      struct DeferRequestFieldInfoArgs : 
        public LgTaskArgs<DeferRequestFieldInfoArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_FIELD_INFOS_TASK_ID;
      public:
        DeferRequestFieldInfoArgs(const FieldSpaceNode *n, 
            std::map<FieldID,FieldInfo> *c, AddressSpaceID src, RtUserEvent t)
          : LgTaskArgs<DeferRequestFieldInfoArgs>(implicit_provenance),
            proxy_this(n), copy(c), source(src), to_trigger(t) { }
      public:
        const FieldSpaceNode *const proxy_this;
        std::map<FieldID,FieldInfo> *const copy;
        const AddressSpaceID source;
        const RtUserEvent to_trigger;
      };
    public:
      FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx, DistributedID did,
                     RtEvent initialized, CollectiveMapping *mapping,
                     Provenance *provenance);
      FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx, DistributedID did,
                     RtEvent initialized, CollectiveMapping *mapping,
                     Provenance *provenance, Deserializer &derez);
      FieldSpaceNode(const FieldSpaceNode &rhs) = delete;
      virtual ~FieldSpaceNode(void);
    public:
      FieldSpaceNode& operator=(const FieldSpaceNode &rhs) = delete;
      AddressSpaceID get_owner_space(void) const; 
      static AddressSpaceID get_owner_space(FieldSpace handle, Runtime *rt);
    public:
      virtual void notify_local(void) { }
    public:
      void attach_semantic_information(SemanticTag tag, AddressSpaceID source,
            const void *buffer, size_t size, bool is_mutable, bool local_only);
      void attach_semantic_information(FieldID fid, SemanticTag tag,
                                       AddressSpaceID source,
                                       const void *buffer, size_t size,
                                       bool is_mutable, bool local_only);
      bool retrieve_semantic_information(SemanticTag tag,
             const void *&result, size_t &size, bool can_fail, bool wait_until);
      bool retrieve_semantic_information(FieldID fid, SemanticTag tag,
             const void *&result, size_t &size, bool can_fail, bool wait_until);
      void send_semantic_info(AddressSpaceID target, SemanticTag tag,
           const void *result, size_t size, bool is_mutable, RtUserEvent ready);
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
      RtEvent create_allocator(AddressSpaceID source,
          RtUserEvent ready = RtUserEvent::NO_RT_USER_EVENT,
          bool sharded_owner_context = false, bool owner_shard = false);
      RtEvent destroy_allocator(AddressSpaceID source,
          bool sharded_owner_context = false, bool owner_shard = false);
    public:
      void initialize_fields(const std::vector<size_t> &sizes,
                             const std::vector<FieldID> &resulting_fields,
                             CustomSerdezID serdez_id, Provenance *provenance,
                             bool collective = false);
      void initialize_fields(ApEvent sizes_ready,
                             const std::vector<FieldID> &resulting_fields,
                             CustomSerdezID serdez_id, Provenance *provenance,
                             bool collective = false);
      RtEvent allocate_field(FieldID fid, size_t size,
                             CustomSerdezID serdez_id,
                             Provenance *provenance,
                             bool sharded_non_owner = false);
      RtEvent allocate_field(FieldID fid, ApEvent size_ready,
                             CustomSerdezID serdez_id,
                             Provenance *provenance,
                             bool sharded_non_owner = false);
      RtEvent allocate_fields(const std::vector<size_t> &sizes,
                              const std::vector<FieldID> &fids,
                              CustomSerdezID serdez_id,
                              Provenance *provenance,
                              bool sharded_non_owner = false);
      RtEvent allocate_fields(ApEvent sizes_ready,
                              const std::vector<FieldID> &fids,
                              CustomSerdezID serdez_id,
                              Provenance *provenance,
                              bool sharded_non_owner = false);
      void update_field_size(FieldID fid, size_t field_size, 
          std::set<RtEvent> &update_events, AddressSpaceID source);
      void free_field(FieldID fid, AddressSpaceID source,
                       std::set<RtEvent> &applied,
                       bool sharded_non_owner = false);
      void free_fields(const std::vector<FieldID> &to_free,
                       AddressSpaceID source, std::set<RtEvent> &applied,
                       bool sharded_non_owner = false);
      void free_field_indexes(const std::vector<FieldID> &to_free,
                              RtEvent freed_event,
                              bool sharded_non_owner = false); 
    public:
      bool allocate_local_fields(const std::vector<FieldID> &fields,
                                 const std::vector<size_t> &sizes,
                                 CustomSerdezID serdez_id,
                                 const std::set<unsigned> &indexes,
                                 std::vector<unsigned> &new_indexes,
                                 Provenance *provenance);
      void free_local_fields(const std::vector<FieldID> &to_free,
                             const std::vector<unsigned> &indexes,
                             const CollectiveMapping *mapping);
      void update_local_fields(const std::vector<FieldID> &fields,
                               const std::vector<size_t> &sizes,
                               const std::vector<CustomSerdezID> &serdez_ids,
                               const std::vector<unsigned> &indexes,
                               Provenance *provenance);
      void remove_local_fields(const std::vector<FieldID> &to_removes);
    public:
      bool has_field(FieldID fid);
      size_t get_field_size(FieldID fid);
      CustomSerdezID get_field_serdez(FieldID fid);
      void get_all_fields(std::vector<FieldID> &to_set);
      void get_all_regions(std::set<LogicalRegion> &regions);
      void get_field_set(const FieldMask &mask, TaskContext *context,
                         std::set<FieldID> &to_set) const;
      void get_field_set(const FieldMask &mask, TaskContext *context,
                         std::vector<FieldID> &to_set) const;
      void get_field_set(const FieldMask &mask,
          const std::set<FieldID> &basis, std::set<FieldID> &to_set) const;
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
      InstanceRef create_external_instance(const std::set<FieldID> &priv_fields,
            const std::vector<FieldID> &fields, RegionNode *node, AttachOp *op);
      PhysicalManager* create_external_manager(PhysicalInstance inst,
            ApEvent ready_event, size_t instance_footprint, 
            LayoutConstraintSet &constraints, 
            const std::vector<FieldID> &field_set,
            const std::vector<size_t> &field_sizes, const FieldMask &file_mask,
            const std::vector<unsigned> &mask_index_map, LgEvent unique_event,
            RegionNode *node, const std::vector<CustomSerdezID> &serdez,
            DistributedID did, CollectiveMapping *collective_mapping = NULL);
      static void handle_external_create_request(Deserializer &derez,
                                Runtime *runtime, AddressSpaceID source);
      static void handle_external_create_response(Deserializer &derez);
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
                                      Deserializer &derez);
      static void handle_node_return(Deserializer &derez);
      static void handle_allocator_request(RegionTreeForest *forest,
                                           Deserializer &derez,
                                           AddressSpaceID source);
      static void handle_allocator_response(RegionTreeForest *forest,
                                            Deserializer &derez);
      static void handle_allocator_invalidation(RegionTreeForest *forest,
                                                Deserializer &derez);
      static void handle_allocator_flush(RegionTreeForest *forest, 
                                         Deserializer &derez);
      static void handle_allocator_free(RegionTreeForest *forest,
                                        Deserializer &derez,
                                        AddressSpaceID source);
      static void handle_infos_request(RegionTreeForest *forest,
                                       Deserializer &derez);
      static void handle_infos_response(RegionTreeForest *forest,
                                        Deserializer &derez);
    public:
      static void handle_remote_instance_creation(RegionTreeForest *forest,
                                Deserializer &derez, AddressSpaceID source);
      static void handle_remote_reduction_creation(RegionTreeForest *forest,
                                Deserializer &derez, AddressSpaceID source);
    public:
      static void handle_alloc_request(RegionTreeForest *forest,
                                       Deserializer &derez);
      static void handle_field_free(RegionTreeForest *forest,
                                    Deserializer &derez, AddressSpaceID source);
      static void handle_field_free_indexes(RegionTreeForest *forest,
                                            Deserializer &derez);
      static void handle_layout_invalidation(RegionTreeForest *forest,
                                             Deserializer &derez,
                                             AddressSpaceID source);
      static void handle_local_alloc_request(RegionTreeForest *forest,
                                             Deserializer &derez,
                                             AddressSpaceID source);
      static void handle_local_alloc_response(Deserializer &derez);
      static void handle_local_free(RegionTreeForest *forest,
                                    Deserializer &derez);
      static void handle_field_size_update(RegionTreeForest *forest,
                                           Deserializer &derez, 
                                           AddressSpaceID source);
      static void handle_defer_infos_request(const void *args);
    public:
      // Help with debug printing
      char* to_string(const FieldMask &mask, TaskContext *ctx) const;
    protected:
      // Assume we are already holding the node lock
      // when calling these methods
      int allocate_index(RtEvent &ready_event, bool initializing = false);
      void free_index(unsigned index, RtEvent free_event);
      void invalidate_layouts(unsigned index, std::set<RtEvent> &applied,
                              AddressSpaceID source, bool need_lock = true);
    protected:
      RtEvent request_field_infos_copy(std::map<FieldID,FieldInfo> *copy,
          AddressSpaceID source, 
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT) const;
      void record_read_only_infos(const std::map<FieldID,FieldInfo> &infos);
      void process_allocator_response(Deserializer &derez);
      void process_allocator_invalidation(RtUserEvent done, 
                                          bool flush, bool merge);
      bool process_allocator_flush(Deserializer &derez);
      void process_allocator_free(Deserializer &derez, AddressSpaceID source);
    protected:
      bool allocate_local_indexes(CustomSerdezID serdez,
            const std::vector<size_t> &sizes,
            const std::set<unsigned> &current_indexes,
                  std::vector<unsigned> &new_indexes);
    public:
      const FieldSpace handle;
      RegionTreeForest *const context;
      Provenance *const provenance;
      RtEvent initialized;
    private:
      mutable LocalLock node_lock;
      std::map<FieldID,FieldInfo> field_infos; // depends on allocation_state
      // Local field sizes
      std::vector<std::pair<size_t,CustomSerdezID> > local_index_infos;
    private:
      // Keep track of the layouts associated with this field space
      // Index them by their hash of their field mask to help
      // differentiate them.
      std::map<LEGION_FIELD_MASK_FIELD_TYPE,LegionList<LayoutDescription*,
                          LAYOUT_DESCRIPTION_ALLOC> > layouts;
    private:
      LegionMap<SemanticTag,SemanticInfo> semantic_info;
      LegionMap<std::pair<FieldID,SemanticTag>,SemanticInfo>
                                                    semantic_field_info;
    private:
      // Track which node is the owner for allocation privileges
      FieldAllocationState allocation_state;
      // For all normal (aka non-local) fields we track which indexes in the 
      // field mask have not been allocated. Only valid on the allocation owner
      FieldMask unallocated_indexes;
      // Use a list here so that we cycle through all the indexes
      // that have been freed before we reuse to avoid false aliasing
      // We may pull things out from the middle though
      std::list<std::pair<unsigned,RtEvent> > available_indexes;
      // Keep track of the nodes with remote copies of field_infos
      mutable std::set<AddressSpaceID> remote_field_infos;
      // An event for recording when we are available for allocation
      // on the owner node in the case we had to send invalidations
      RtEvent pending_field_allocation;
      // Total number of outstanding allocators
      unsigned outstanding_allocators;
      // Total number of outstanding invalidations (owner node only)
      unsigned outstanding_invalidations;
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
      RegionTreeNode(RegionTreeForest *ctx, FieldSpaceNode *column,
                     RtEvent initialized, RtEvent tree_init, 
                     Provenance *provenance = NULL, DistributedID did = 0,
                     CollectiveMapping *mapping = NULL);
      virtual ~RegionTreeNode(void);
    public:
      static AddressSpaceID get_owner_space(RegionTreeID tid, Runtime *rt);
    public:
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
    public:
      void attach_semantic_information(SemanticTag tag, AddressSpaceID source,
            const void *buffer, size_t size, bool is_mutable, bool local_only);
      bool retrieve_semantic_information(SemanticTag tag,
           const void *&result, size_t &size, bool can_fail, bool wait_until);
      virtual void send_semantic_request(AddressSpaceID target, 
        SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready) = 0;
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
       const void *buffer, size_t size, bool is_mutable, RtUserEvent ready) = 0;
    public:
      // Logical traversal operations
      void register_logical_user(ContextID ctx,
                                 const LogicalUser &user,
                                 const RegionTreePath &path,
                                 const LogicalTraceInfo &trace_info,
                                 const ProjectionInfo &projection_info,
                                 FieldMask &unopened_field_mask,
                                 FieldMask &already_closed_mask,
                                 FieldMask &disjoint_complete_below,
                                 FieldMask &first_touch_refinement,
                                 FieldMaskSet<RefinementOp> &refinements,
                                 LogicalAnalysis &logical_analysis,
                                 const bool track_disjoint_complete_below,
                                 const bool check_unversioned);
      void register_local_user(LogicalState &state,
                               const LogicalUser &user,
                               const LogicalTraceInfo &trace_info);
      void add_open_field_state(LogicalState &state, bool arrived,
                                const ProjectionInfo &projection_info,
                                const LogicalUser &user,
                                const FieldMask &open_mask,
                                RegionTreeNode *next_child);
      void close_logical_node(LogicalCloser &closer,
                              const FieldMask &closing_mask,
                              const bool read_only_close);
      void siphon_logical_children(LogicalCloser &closer,
                                   LogicalState &state,
                                   const FieldMask &closing_mask,
                                   const FieldMask *aliased_children,
                                   bool record_close_operations,
                                   RegionTreeNode *next_child,
                                   FieldMask &open_below);
      void siphon_logical_projection(LogicalCloser &closer,
                                     LogicalState &state,
                                     const FieldMask &closing_mask,
                                     const ProjectionInfo &proj_info,
                                     bool record_close_operations,
                                     FieldMask &open_below);
      void flush_logical_reductions(LogicalCloser &closer,
                                    LogicalState &state,
                                    FieldMask &reduction_flush_fields,
                                    bool record_close_operations,
                                    RegionTreeNode *next_child,
                                    LegionDeque<FieldState> &states);
      // Note that 'allow_next_child' and 
      // 'record_closed_fields' are mutually exclusive
      void perform_close_operations(LogicalCloser &closer,
                                    const FieldMask &closing_mask,
                                    FieldState &closing_state,
                                    RegionTreeNode *next_child,
                                    bool allow_next_child,
                                    const FieldMask *aliased_children,
                                    bool upgrade_next_child, 
                                    bool read_only_close,
                                    bool overwriting_close,
                                    bool record_close_operations,
                                    bool record_closed_fields,
                                    FieldMask &output_mask); 
      void merge_new_field_state(LogicalState &state, FieldState &new_state);
      void merge_new_field_states(LogicalState &state, 
                                  LegionDeque<FieldState> &new_states);
      void filter_prev_epoch_users(LogicalState &state, const FieldMask &mask);
      void filter_curr_epoch_users(LogicalState &state, const FieldMask &mask,
                                   const bool tracing);
      void filter_disjoint_complete_accesses(LogicalState &state,
                                             const FieldMask &mask);
      void report_uninitialized_usage(Operation *op, unsigned index,
                                      const RegionUsage usage,
                                      const FieldMask &uninitialized,
                                      RtUserEvent reported);
      void record_logical_reduction(LogicalState &state, ReductionOpID redop,
                                    const FieldMask &user_mask);
      void clear_logical_reduction_fields(LogicalState &state,
                                          const FieldMask &cleared_mask);
      void sanity_check_logical_state(LogicalState &state);
      void perform_tree_dominance_analysis(ContextID ctx,
                                           const LogicalUser &user,
                                           const FieldMask &field_mask,
                                           Operation *skip_op = NULL,
                                           GenerationID skip_gen = 0);
      void invalidate_disjoint_complete_tree(ContextID ctx, 
                                        const FieldMask &invalidate_mask,
                                        const bool invalidate_self);
      void register_logical_deletion(ContextID ctx,
                                     const LogicalUser &user,
                                     const FieldMask &check_mask,
                                     const RegionTreePath &path,
                                     const LogicalTraceInfo &trace_info,
                                     FieldMask &already_closed_mask,
                                     bool invalidate_tree); 
      void siphon_logical_deletion(LogicalCloser &closer,
                                   LogicalState &state,
                                   const FieldMask &current_mask,
                                   RegionTreeNode *next_child,
                                   FieldMask &open_below,
                                   bool force_close_next);
      void record_close_no_dependences(ContextID ctx,
                                       const LogicalUser &user);
    public:
      void migrate_logical_state(ContextID src, ContextID dst, bool merge);
      void migrate_version_state(ContextID src, ContextID dst, 
                                 std::set<RtEvent> &applied, bool merge);
      void pack_logical_state(ContextID ctx, Serializer &rez, 
                              const bool invalidate); 
      void unpack_logical_state(ContextID ctx, Deserializer &derez,
                                AddressSpaceID source);
      void pack_version_state(ContextID ctx, Serializer &rez, 
                              const bool invalidate,
                              std::set<RtEvent> &applied_events); 
      void unpack_version_state(ContextID ctx, Deserializer &derez, 
                                AddressSpaceID source);
    public:
      void initialize_current_state(ContextID ctx);
      void invalidate_current_state(ContextID ctx, bool users_only);
      void invalidate_deleted_state(ContextID ctx, 
                                    const FieldMask &deleted_mask);
      void invalidate_logical_states(void);
    public:
      virtual unsigned get_depth(void) const = 0;
      virtual LegionColor get_color(void) const = 0;
      virtual IndexTreeNode *get_row_source(void) const = 0;
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
    public:
      virtual size_t get_num_children(void) const = 0;
      virtual void send_node(Serializer &rez, AddressSpaceID target) = 0;
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
          LegionList<LogicalUser, ALLOC> &users, 
          const FieldMask &check_mask, const FieldMask &open_below,
          bool validates_regions, Operation *to_skip = NULL, 
          GenerationID skip_gen = 0);
      template<AllocationType ALLOC>
      static void perform_closing_checks(LogicalCloser &closer,
          LegionList<LogicalUser, ALLOC> &users, 
          const FieldMask &check_mask);
      template<AllocationType ALLOC>
      static void perform_nodep_checks(const LogicalUser &user,
          const LegionList<LogicalUser, ALLOC> &users);
    public:
      inline FieldSpaceNode* get_column_source(void) const 
        { return column_source; }
    public:
      RegionTreeForest *const context;
      FieldSpaceNode *const column_source;
      Provenance *const provenance;
      RtEvent initialized;
      const RtEvent tree_initialized; // top level tree initialization
    public:
      bool registered;
    protected:
      DynamicTable<LogicalStateAllocator> logical_states;
      DynamicTable<VersionManagerAllocator> current_versions;
    protected:
      mutable LocalLock node_lock;
    protected:
      LegionMap<SemanticTag,SemanticInfo> semantic_info;
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
      struct DeferComputeEquivalenceSetArgs : 
        public LgTaskArgs<DeferComputeEquivalenceSetArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_COMPUTE_EQ_SETS_TASK_ID;
      public:
        DeferComputeEquivalenceSetArgs(RegionNode *proxy, ContextID x,
            InnerContext *c, EqSetTracker *t, const AddressSpaceID ts,
            IndexSpaceExpression *e, const FieldMask &m, 
            const UniqueID id, const AddressSpaceID s, const bool covers);
      public:
        RegionNode *const proxy_this;
        const ContextID ctx;
        InnerContext *const context;
        EqSetTracker *const target;
        const AddressSpaceID target_space;
        IndexSpaceExpression *const expr;
        FieldMask *const mask;
        const UniqueID opid;
        const AddressSpaceID source;
        const RtUserEvent ready;
        const bool expr_covers;
      };
    public:
      RegionNode(LogicalRegion r, PartitionNode *par, IndexSpaceNode *row_src,
             FieldSpaceNode *col_src, RegionTreeForest *ctx, 
             DistributedID did, RtEvent initialized, RtEvent tree_initialized,
             CollectiveMapping *mapping, Provenance *provenance);
      RegionNode(const RegionNode &rhs) = delete;
      virtual ~RegionNode(void);
    public:
      RegionNode& operator=(const RegionNode &rhs) = delete;
    public:
      virtual void notify_local(void);
    public:
      void record_registered(void);
    public:
      bool has_color(const LegionColor p);
      PartitionNode* get_child(const LegionColor p);
      void add_child(PartitionNode *child);
      void remove_child(const LegionColor p);
      void add_tracker(PartitionTracker *tracker);
      void initialize_disjoint_complete_tree(ContextID ctx, const FieldMask &m);
      void refine_disjoint_complete_tree(ContextID ctx, PartitionNode *child,
                                         RefinementOp *refinement, 
                                         const FieldMask &refinement_mask,
                                         std::set<RtEvent> &applied_events);
      bool filter_unversioned_fields(ContextID ctx, TaskContext *context,
                                     const FieldMask &filter_mask,
                                     RegionRequirement &req);
    public:
      virtual unsigned get_depth(void) const;
      virtual LegionColor get_color(void) const;
      virtual IndexTreeNode *get_row_source(void) const;
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
      virtual size_t get_num_children(void) const;
      virtual void send_node(Serializer &rez, AddressSpaceID target);
      static void handle_node_creation(RegionTreeForest *context,
                            Deserializer &derez, AddressSpaceID source);
    public:
      virtual void send_semantic_request(AddressSpaceID target, 
           SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready);
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
          const void *buffer, size_t size, bool is_mutable, RtUserEvent ready);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source,
                            bool can_fail, bool wait_until, RtUserEvent ready);
      static void handle_semantic_request(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
    public:
      static void handle_top_level_request(RegionTreeForest *forest,
                                   Deserializer &derez);
      static void handle_top_level_return(RegionTreeForest *forest,
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
                               FieldMaskSet<PartitionNode> &to_traverse,
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
      // Support for refinements and versioning
      void update_disjoint_complete_tree(ContextID ctx, RefinementOp *op,
                                         const FieldMask &refinement_mask,
                                         FieldMask &refined_partition,
                                         std::set<RtEvent> &applied_events);
      void initialize_versioning_analysis(ContextID ctx, EquivalenceSet *set,
                                          const FieldMask &mask);
      void initialize_nonexclusive_virtual_analysis(ContextID ctx,
                                  const FieldMask &mask,
                                  const FieldMaskSet<EquivalenceSet> &eq_sets);
      void perform_versioning_analysis(ContextID ctx, 
                                       InnerContext *parent_ctx,
                                       VersionInfo *version_info,
                                       const FieldMask &version_mask,
                                       const UniqueID opid, 
                                       const AddressSpaceID original_source,
                                       std::set<RtEvent> &ready_events);
      void compute_equivalence_sets(ContextID ctx,
                                    InnerContext *parent_ctx,
                                    EqSetTracker *target,
                                    const AddressSpaceID target_space,
                                    IndexSpaceExpression *expr,
                                    const FieldMask &mask,
                                    const UniqueID opid,
                                    const AddressSpaceID original_source,
                                    std::set<RtEvent> &ready_events,
                                    const bool downward_only,
                                    const bool expr_covers);
      static void handle_deferred_compute_equivalence_sets(const void *args);
      void invalidate_refinement(ContextID ctx, const FieldMask &mask,
                                 bool self, InnerContext &source_context,
                                 std::set<RtEvent> &applied_events, 
                                 std::vector<EquivalenceSet*> &to_release,
                                 bool nonexclusive_virtual_root = false);
      void record_refinement(ContextID ctx, EquivalenceSet *set, 
                             const FieldMask &mask);
      void propagate_refinement(ContextID ctx, PartitionNode *child,
                                const FieldMask &mask);
    public:
      void find_open_complete_partitions(ContextID ctx,
                                         const FieldMask &mask,
                    std::vector<LogicalPartition> &partitions);
    public:
      const LogicalRegion handle;
      PartitionNode *const parent;
      IndexSpaceNode *const row_source;
    protected:
      std::map<LegionColor,PartitionNode*> color_map;
      std::list<PartitionTracker*> partition_trackers;
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
    public:
      PartitionNode(LogicalPartition p, RegionNode *par, 
                    IndexPartNode *row_src, FieldSpaceNode *col_src,
                    RegionTreeForest *ctx, RtEvent init, RtEvent tree);
      PartitionNode(const PartitionNode &rhs);
      virtual ~PartitionNode(void);
    public:
      PartitionNode& operator=(const PartitionNode &rhs);
    public:
      virtual void notify_local(void);
    public:
      void record_registered(void);
    public:
      bool has_color(const LegionColor c);
      RegionNode* get_child(const LegionColor c);
      void add_child(RegionNode *child);
    public:
      virtual unsigned get_depth(void) const;
      virtual LegionColor get_color(void) const;
      virtual IndexTreeNode *get_row_source(void) const;
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
      virtual size_t get_num_children(void) const;
      virtual void send_node(Serializer &rez, AddressSpaceID target);
    public:
      virtual void send_semantic_request(AddressSpaceID target, 
           SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready);
      virtual void send_semantic_info(AddressSpaceID target, SemanticTag tag,
          const void *buffer, size_t size, bool is_mutable, RtUserEvent ready);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source,
                            bool can_fail, bool wait_until, RtUserEvent ready);
      static void handle_semantic_request(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
    public:
      void update_disjoint_complete_tree(ContextID ctx, RefinementOp *op,
                                         const FieldMask &refinement_mask,
                                         std::set<RtEvent> &applied_events);
      void compute_equivalence_sets(ContextID ctx,
                                    InnerContext *context,
                                    EqSetTracker *target,
                                    const AddressSpaceID target_space,
                                    IndexSpaceExpression *expr,
                                    const FieldMask &mask,
                                    const UniqueID opid,
                                    const AddressSpaceID source,
                                    std::set<RtEvent> &ready_events,
                                    const bool downward_only,
                                    const bool expr_covers);
      void invalidate_refinement(ContextID ctx, const FieldMask &mask,
                                 std::set<RtEvent> &applied_events,
                                 std::vector<EquivalenceSet*> &to_release,
                                 InnerContext &source_context);
      void propagate_refinement(ContextID ctx, RegionNode *child,
                                const FieldMask &mask);
      void propagate_refinement(ContextID ctx, 
                                const std::vector<RegionNode*> &children,
                                const FieldMask &mask);
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
                               FieldMaskSet<RegionNode> &to_traverse,
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

    //--------------------------------------------------------------------------
    /*static*/ inline bool RegionTreeForest::compare_expressions(
                           IndexSpaceExpression *one, IndexSpaceExpression *two)
    //--------------------------------------------------------------------------
    {
      return (one->expr_id < two->expr_id);
    }
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

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_REGION_TREE_H__

// EOF

