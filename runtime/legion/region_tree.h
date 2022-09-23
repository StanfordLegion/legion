/* Copyright 2022 Stanford University, NVIDIA Corporation
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
      IndexSpace index_space;
      PhysicalInstance inst;
      size_t field_offset;
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
                     const InstanceSet &insts,
                     const DomainPoint &key);
    public:
      // In the same order as the fields for the actual copy
      std::vector<PhysicalInstance> instances;
#ifdef LEGION_SPY
      std::vector<ApEvent> instance_events;
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
      struct DeferPhysicalRegistrationArgs : 
        public LgTaskArgs<DeferPhysicalRegistrationArgs>, 
        public PhysicalTraceInfo {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_PHYSICAL_REGISTRATION_TASK_ID;
      public:
        DeferPhysicalRegistrationArgs(UniqueID uid, UpdateAnalysis *ana,
                  InstanceSet &t, RtUserEvent map_applied, ApEvent &res,
                  const PhysicalTraceInfo &info)
          : LgTaskArgs<DeferPhysicalRegistrationArgs>(uid), 
            PhysicalTraceInfo(info), analysis(ana), 
            map_applied_done(map_applied), targets(t), result(res) 
          // This is kind of scary, Realm is about to make a copy of this
          // without our knowledge, but we need to preserve the correctness
          // of reference counting on PhysicalTraceRecorders, so just add
          // an extra reference here that we will remove when we're handled.
          { 
            analysis->add_reference(); 
            if (rec != NULL) rec->add_recorder_reference();
          }
      public:
        inline void remove_recorder_reference(void) const
          { if ((rec != NULL) && rec->remove_recorder_reference()) delete rec; }
      public:
        UpdateAnalysis *const analysis;
        RtUserEvent map_applied_done;
        InstanceSet &targets;
        ApEvent &result;
      };
    public:
      RegionTreeForest(Runtime *rt);
      RegionTreeForest(const RegionTreeForest &rhs);
      ~RegionTreeForest(void);
    public:
      RegionTreeForest& operator=(const RegionTreeForest &rhs);
    public:
      IndexSpaceNode* create_index_space(IndexSpace handle, 
                              const Domain *domain, DistributedID did, 
                              Provenance *provenance,
                              ApEvent ready = ApEvent::NO_AP_EVENT,
                              std::set<RtEvent> *applied = NULL);
      IndexSpaceNode* create_union_space(IndexSpace handle, DistributedID did,
                              const char *provenance,
                              const std::vector<IndexSpace> &sources,
                              RtEvent initialized = RtEvent::NO_RT_EVENT,
                              std::set<RtEvent> *applied = NULL);
      IndexSpaceNode* create_intersection_space(IndexSpace handle, 
                              DistributedID did, const char *provenance,
                              const std::vector<IndexSpace> &sources,
                              RtEvent initialized = RtEvent::NO_RT_EVENT,
                              std::set<RtEvent> *applied = NULL);
      IndexSpaceNode* create_difference_space(IndexSpace handle,
                              DistributedID did, const char *provenance,
                              IndexSpace left, IndexSpace right,
                              RtEvent initialized = RtEvent::NO_RT_EVENT,
                              std::set<RtEvent> *applied = NULL);
      RtEvent create_pending_partition(TaskContext *ctx,
                                       IndexPartition pid,
                                       IndexSpace parent,
                                       IndexSpace color_space,
                                       LegionColor partition_color,
                                       PartitionKind part_kind,
                                       DistributedID did,
                                       Provenance *provenance,
                                       ApEvent partition_ready,
            ApUserEvent partial_pending = ApUserEvent::NO_AP_USER_EVENT,
                                       std::set<RtEvent> *applied = NULL);
      void create_pending_cross_product(TaskContext *ctx,
                                        IndexPartition handle1,
                                        IndexPartition handle2,
                  std::map<IndexSpace,IndexPartition> &user_handles,
                                        PartitionKind kind,
                                        Provenance *provenance,
                                        LegionColor &part_color,
                                        ApEvent domain_ready,
                                        std::set<RtEvent> &safe_events);
      void compute_partition_disjointness(IndexPartition handle,
                                          RtUserEvent ready_event);
      void destroy_index_space(IndexSpace handle, AddressSpaceID source,
                               std::set<RtEvent> &preconditions);
      void destroy_index_partition(IndexPartition handle, 
                                   std::set<RtEvent> &preconditions);
    public:
      ApEvent create_equal_partition(Operation *op, 
                                     IndexPartition pid, 
                                     size_t granularity);
      ApEvent create_partition_by_weights(Operation *op,
                                          IndexPartition pid,
                                          const FutureMap &map,
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
                                         const FutureMap &future_map,
                                         bool perform_intersections);
      ApEvent create_cross_product_partitions(Operation *op,
                                              IndexPartition base,
                                              IndexPartition source,
                                              LegionColor part_color);
    public:  
      ApEvent create_partition_by_field(Operation *op,
                                        IndexPartition pending,
                    const std::vector<FieldDataDescriptor> &instances,
                                        ApEvent instances_ready);
      ApEvent create_partition_by_image(Operation *op,
                                        IndexPartition pending,
                                        IndexPartition projection,
                    const std::vector<FieldDataDescriptor> &instances,
                                        ApEvent instances_ready);
      ApEvent create_partition_by_image_range(Operation *op,
                                              IndexPartition pending,
                                              IndexPartition projection,
                    const std::vector<FieldDataDescriptor> &instances,
                                              ApEvent instances_ready);
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
      IndexSpace find_pending_space(IndexPartition parent,
                                    const void *realm_color,
                                    TypeTag type_tag,
                                    ApUserEvent &domain_ready);
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
      FieldSpaceNode* create_field_space(FieldSpace handle, DistributedID did,
                   const char *provenance, std::set<RtEvent> *applied = NULL);
      void destroy_field_space(FieldSpace handle,
                               std::set<RtEvent> &preconditions);
      // Return true if local is set to true and we actually performed the 
      // allocation.  It is an error if the field already existed and the
      // allocation was not local.
      bool allocate_field(FieldSpace handle, size_t field_size, 
                          FieldID fid, CustomSerdezID serdez_id,
                          const char *provenance);
      FieldSpaceNode* allocate_field(FieldSpace handle, ApEvent ready,
                                     FieldID fid, CustomSerdezID serdez_id,
                                     const char *provenance);
      void free_field(FieldSpace handle, FieldID fid,
                      std::set<RtEvent> &preconditions);
      void allocate_fields(FieldSpace handle, const std::vector<size_t> &sizes,
                           const std::vector<FieldID> &resulting_fields,
                           CustomSerdezID serdez_id, const char *provenance);
      FieldSpaceNode* allocate_fields(FieldSpace handle, ApEvent ready, 
                           const std::vector<FieldID> &resulting_fields,
                           CustomSerdezID serdez_id, const char *provenance);
      void free_fields(FieldSpace handle, 
                       const std::vector<FieldID> &to_free,
                       std::set<RtEvent> &preconditions);
      void free_field_indexes(FieldSpace handle,
                       const std::vector<FieldID> &to_free, RtEvent freed);
    public:
      bool allocate_local_fields(FieldSpace handle, 
                                 const std::vector<FieldID> &resulting_fields,
                                 const std::vector<size_t> &sizes,
                                 CustomSerdezID serdez_id,
                                 const std::set<unsigned> &allocated_indexes,
                                 std::vector<unsigned> &new_indexes,
                                 const char *provenance);
      void free_local_fields(FieldSpace handle,
                             const std::vector<FieldID> &to_free,
                             const std::vector<unsigned> &indexes);
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
      void create_logical_region(LogicalRegion handle, DistributedID did,
                                 const char *provenance,
                                 std::set<RtEvent> *applied = NULL);
      void destroy_logical_region(LogicalRegion handle, 
                                  std::set<RtEvent> &preconditions);
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
                                       const ProjectionInfo &projection_info,
                                       RegionTreePath &path,
                                       std::set<RtEvent> &applied_events);
      void perform_deletion_analysis(DeletionOp *op, unsigned idx,
                                     RegionRequirement &req,
                                     RegionTreePath &path,
                                     std::set<RtEvent> &applied_events,
                                     bool invalidate_tree);
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
                                       VersionInfo &version_info,
                                       std::set<RtEvent> &ready_events);
      void invalidate_versions(RegionTreeContext ctx, LogicalRegion handle);
      void invalidate_all_versions(RegionTreeContext ctx);
    public:
      void initialize_current_context(RegionTreeContext ctx,
                    const RegionRequirement &req, const bool restricted,
                    const InstanceSet &sources, ApEvent term_event, 
                    InnerContext *context, unsigned index,
                    std::map<PhysicalManager*,InstanceView*> &top_views,
                    std::set<RtEvent> &applied_events);
      void invalidate_current_context(RegionTreeContext ctx, bool users_only,
                                      LogicalRegion handle);
      bool match_instance_fields(const RegionRequirement &req1,
                                 const RegionRequirement &req2,
                                 const InstanceSet &inst1,
                                 const InstanceSet &inst2);
    public: // Physical analysis methods
      void physical_premap_region(Operation *op, unsigned index,
                                  RegionRequirement &req,
                                  VersionInfo &version_info,
                                  InstanceSet &valid_instances,
                                  std::set<RtEvent> &map_applied_events);
      // Return a runtime event for when it's safe to perform
      // the registration for this equivalence set
      RtEvent physical_perform_updates(const RegionRequirement &req,
                                VersionInfo &version_info,
                                Operation *op, unsigned index,
                                ApEvent precondition, ApEvent term_event,
                                const InstanceSet &targets,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &map_applied_events,
                                UpdateAnalysis *&analysis,
#ifdef DEBUG_LEGION
                                const char *log_name,
                                UniqueID uid,
#endif
                                const bool record_valid = true,
                                const bool check_initialized = true,
                                const bool defer_copies = true);
      // Return an event for when the copy-out effects of the 
      // registration are done (e.g. for restricted coherence)
      ApEvent physical_perform_registration(UpdateAnalysis *analysis,
                                 InstanceSet &targets,
                                 const PhysicalTraceInfo &trace_info,
                                 std::set<RtEvent> &map_applied_events);
      // Same as the two above merged together
      ApEvent physical_perform_updates_and_registration(
                                   const RegionRequirement &req,
                                   VersionInfo &version_info,
                                   Operation *op, unsigned index,
                                   ApEvent precondition, ApEvent term_event,
                                   InstanceSet &targets,
                                   const PhysicalTraceInfo &trace_info,
                                   std::set<RtEvent> &map_applied_events,
#ifdef DEBUG_LEGION
                                   const char *log_name,
                                   UniqueID uid,
#endif
                                   const bool record_valid = true,
                                   const bool check_initialized = true);
      // A helper method for deferring the computation of registration
      RtEvent defer_physical_perform_registration(RtEvent register_pre,
                           UpdateAnalysis *analysis, InstanceSet &targets,
                           std::set<RtEvent> &map_applied_events,
                           ApEvent &result, const PhysicalTraceInfo &info);
      void handle_defer_registration(const void *args);
      ApEvent acquire_restrictions(const RegionRequirement &req,
                                   VersionInfo &version_info,
                                   AcquireOp *op, unsigned index,
                                   ApEvent term_event,
                                   InstanceSet &restricted_instances,
                                   const PhysicalTraceInfo &trace_info,
                                   std::set<RtEvent> &map_applied_events
#ifdef DEBUG_LEGION
                                   , const char *log_name
                                   , UniqueID uid
#endif
                                   );
      ApEvent release_restrictions(const RegionRequirement &req,
                                   VersionInfo &version_info,
                                   ReleaseOp *op, unsigned index,
                                   ApEvent precondition, ApEvent term_event,
                                   InstanceSet &restricted_instances,
                                   const PhysicalTraceInfo &trace_info,
                                   std::set<RtEvent> &map_applied_events
#ifdef DEBUG_LEGION
                                   , const char *log_name
                                   , UniqueID uid
#endif
                                   );
      ApEvent copy_across(const RegionRequirement &src_req,
                          const RegionRequirement &dst_req,
                          VersionInfo &src_version_info,
                          VersionInfo &dst_version_info,
                          const InstanceSet &src_targets,
                          const InstanceSet &dst_targets, CopyOp *op,
                          unsigned src_index, unsigned dst_index,
                          ApEvent precondition, PredEvent pred_guard,
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
      // This takes ownership of the value buffer
      ApEvent fill_fields(FillOp *op,
                          const RegionRequirement &req,
                          const unsigned index, FillView *fill_view,
                          VersionInfo &version_info, ApEvent precondition,
                          PredEvent true_guard,
                          const PhysicalTraceInfo &trace_info,
                          std::set<RtEvent> &map_applied_events);
      InstanceRef create_external_instance(AttachOp *attach_op,
                                const RegionRequirement &req,
                                const std::vector<FieldID> &field_set);
      ApEvent attach_external(AttachOp *attach_op, unsigned index,
                              const RegionRequirement &req,
                              // Two views are usually the same but different
                              // in cases of control replication
                              std::vector<InstanceView*> &local_views,
                              std::set<LogicalView*> &registration_views,
                              const ApEvent termination_event,
                              VersionInfo &version_info,
                              const PhysicalTraceInfo &trace_info,
                              std::set<RtEvent> &map_applied_events,
                              const bool restricted);
      ApEvent detach_external(const RegionRequirement &req, DetachOp *detach_op,
                              unsigned index, VersionInfo &version_info, 
                              InstanceView *local_view,
                              const PhysicalTraceInfo &trace_info,
                              std::set<RtEvent> &map_applied_events,
                              LogicalView *registration_view = NULL);
      void invalidate_fields(Operation *op, unsigned index,
                             VersionInfo &version_info,
                             const PhysicalTraceInfo &trace_info,
                             std::set<RtEvent> &map_applied_events);
      // Support for tracing
      void find_invalid_instances(Operation *op, unsigned index,
                                  VersionInfo &version_info,
                                  const FieldMaskSet<InstanceView> &valid_views,
                                  FieldMaskSet<InstanceView> &invalid_instances,
                                  std::set<RtEvent> &map_applied_events);
      void update_valid_instances(Operation *op, unsigned index,
                                  VersionInfo &version_info,
                                  const FieldMaskSet<InstanceView> &valid_views,
                                  const PhysicalTraceInfo &trace_info,
                                  std::set<RtEvent> &map_applied_events);
    public:
      int physical_convert_mapping(Operation *op,
                               const RegionRequirement &req,
                               const std::vector<MappingInstance> &chosen,
                               InstanceSet &result, RegionTreeID &bad_tree,
                               std::vector<FieldID> &missing_fields,
                               std::map<PhysicalManager*,unsigned> *acquired,
                               std::vector<PhysicalManager*> &unacquired,
                               const bool do_acquire_checks,
                               const bool allow_partial_virtual = false);
      bool physical_convert_postmapping(Operation *op,
                               const RegionRequirement &req,
                               const std::vector<MappingInstance> &chosen,
                               InstanceSet &result, RegionTreeID &bad_tree,
                               std::map<PhysicalManager*,unsigned> *acquired,
                               std::vector<PhysicalManager*> &unacquired,
                               const bool do_acquire_checks);
      void log_mapping_decision(const UniqueID unique_id, TaskContext *context,
                                const unsigned index, 
                                const RegionRequirement &req,
                                const InstanceSet &targets,
                                bool postmapping = false);
    public: // helper method for the above two methods
      void perform_missing_acquires(Operation *op,
                               std::map<PhysicalManager*,unsigned> &acquired,
                               const std::vector<PhysicalManager*> &unacquired);
    public:
      // Debugging method for checking context state
      void check_context_state(RegionTreeContext ctx);
    public:
      // We know the domain of the index space
      IndexSpaceNode* create_node(IndexSpace is, const void *bounds, 
                                  bool is_domain, IndexPartNode *par, 
                                  LegionColor color, DistributedID did,
                                  RtEvent initialized, Provenance *provenance,
                                  ApEvent is_ready = ApEvent::NO_AP_EVENT,
                                  IndexSpaceExprID expr_id = 0,
                                  std::set<RtEvent> *applied = NULL,
                                  bool add_root_reference = false,
                                  unsigned depth = UINT_MAX);
      IndexSpaceNode* create_node(IndexSpace is, const void *realm_is, 
                                  IndexPartNode *par, LegionColor color,
                                  DistributedID did, RtEvent initialized,
                                  Provenance *provenance, ApUserEvent is_ready,
                                  std::set<RtEvent> *applied = NULL,
                                  unsigned depth = UINT_MAX);
      // We know the disjointness of the index partition
      IndexPartNode*  create_node(IndexPartition p, IndexSpaceNode *par,
                                  IndexSpaceNode *color_space, 
                                  LegionColor color, bool disjoint,int complete,
                                  DistributedID did, Provenance *provenance,
                                  ApEvent partition_ready, 
                                  ApUserEvent partial_pending, RtEvent init,
                                  std::set<RtEvent> *applied = NULL);
      // Give the event for when the disjointness information is ready
      IndexPartNode*  create_node(IndexPartition p, IndexSpaceNode *par,
                                  IndexSpaceNode *color_space,LegionColor color,
                                  RtEvent disjointness_ready_event,int complete,
                                  DistributedID did, Provenance *provenance,
                                  ApEvent partition_ready, 
                                  ApUserEvent partial_pending, RtEvent init,
                                  std::set<RtEvent> *applied = NULL);
      FieldSpaceNode* create_node(FieldSpace space, DistributedID did, 
                                  RtEvent initialized, Provenance *provenance, 
                                  std::set<RtEvent> *applied = NULL);
      FieldSpaceNode* create_node(FieldSpace space, DistributedID did, 
                                  RtEvent initialized, Provenance *provenance,
                                  Deserializer &derez);
      RegionNode*     create_node(LogicalRegion r, PartitionNode *par, 
                                  RtEvent initialized, DistributedID did,
                                  Provenance *provenance = NULL,
                                  std::set<RtEvent> *applied = NULL);
      PartitionNode*  create_node(LogicalPartition p, RegionNode *par,
                                  std::set<RtEvent> *applied = NULL);
    public:
      IndexSpaceNode* get_node(IndexSpace space, RtEvent *defer = NULL,
                               const bool can_fail = false);
      IndexPartNode*  get_node(IndexPartition part, RtEvent *defer = NULL,
                               const bool can_fail = false,
                               const bool local_only = false);
      FieldSpaceNode* get_node(FieldSpace space, RtEvent *defer = NULL);
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
                                              IndexSpaceExpression *rhs,
                                              ReferenceMutator *mutator = NULL);
      IndexSpaceExpression* union_index_spaces(
                                 const std::set<IndexSpaceExpression*> &exprs,
                                 ReferenceMutator *mutator = NULL);
    protected:
      // Internal version
      IndexSpaceExpression* union_index_spaces(
                               const std::vector<IndexSpaceExpression*> &exprs,
                               OperationCreator *creator = NULL);
    public:
      IndexSpaceExpression* intersect_index_spaces(
                                              IndexSpaceExpression *lhs,
                                              IndexSpaceExpression *rhs,
                                              ReferenceMutator *mutator = NULL);
      IndexSpaceExpression* intersect_index_spaces(
                                 const std::set<IndexSpaceExpression*> &exprs,
                                 ReferenceMutator *mutator = NULL);
    protected:
      IndexSpaceExpression* intersect_index_spaces(
                               const std::vector<IndexSpaceExpression*> &exprs,
                               OperationCreator *creator = NULL);
    public:
      IndexSpaceExpression* subtract_index_spaces(IndexSpaceExpression *lhs,
                  IndexSpaceExpression *rhs, OperationCreator *creator = NULL,
                  ReferenceMutator *mutator = NULL);
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
    class CopyAcrossExecutor : public Collectable {
    public:
      struct DeferCopyAcrossArgs : public LgTaskArgs<DeferCopyAcrossArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_COPY_ACROSS_TASK_ID;
      public:
        DeferCopyAcrossArgs(CopyAcrossExecutor *e, Operation *o, 
            PredEvent guard, ApEvent copy_pre, ApEvent src_pre,
            ApEvent dst_pre, const PhysicalTraceInfo &info,
            bool recurrent, unsigned stage);
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
        const bool recurrent_replay;
      };
    public:
      CopyAcrossExecutor(Runtime *rt, const bool preimages,
                         const std::map<Reservation,bool> &rsrvs)
        : runtime(rt), reservations(rsrvs), compute_preimages(preimages) { }
      virtual ~CopyAcrossExecutor(void) { }
    public:
      virtual ApEvent execute(Operation *op, PredEvent pred_guard,
                              ApEvent copy_precondition,
                              ApEvent src_indirect_precondition, 
                              ApEvent dst_indirect_precondition,
                              const PhysicalTraceInfo &trace_info,
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
        : CopyAcrossExecutor(rt, preimages, rsrvs) { }
      virtual ~CopyAcrossUnstructured(void) { }
    public:
      virtual ApEvent execute(Operation *op, PredEvent pred_guard,
                              ApEvent copy_precondition,
                              ApEvent src_indirect_precondition,
                              ApEvent dst_indirect_precondition,
                              const PhysicalTraceInfo &trace_info,
                              const bool recurrent_replay = false,
                              const unsigned stage = 0) = 0;
      virtual void record_trace_immutable_indirection(bool source) = 0;
    public:
      void initialize_source_fields(RegionTreeForest *forest,
                                    const RegionRequirement &req,
                                    const InstanceSet &instances,
                                    const std::vector<InstanceView*> &views,
                                    const PhysicalTraceInfo &trace_info);
      void initialize_destination_fields(RegionTreeForest *forest,
                                    const RegionRequirement &req,
                                    const InstanceSet &instances,
                                    const std::vector<InstanceView*> &views,
                                    const PhysicalTraceInfo &trace_info,
                                    const bool exclusive_redop);
      void initialize_source_indirections(RegionTreeForest *forest,
                                    std::vector<IndirectRecord> &records,
                                    const RegionRequirement &src_req,
                                    const RegionRequirement &idx_req,
                                    const InstanceRef &indirect_instance,
                                    const DomainPoint &index_point,
                                    const bool both_are_range,
                                    const bool possible_out_of_range);
      void initialize_destination_indirections(RegionTreeForest *forest,
                                    std::vector<IndirectRecord> &records,
                                    const RegionRequirement &dst_req,
                                    const RegionRequirement &idx_req,
                                    const InstanceRef &indirect_instance,
                                    const DomainPoint &index_point,
                                    const bool both_are_range,
                                    const bool possible_out_of_range,
                                    const bool possible_aliasing,
                                    const bool exclusive_redop);
    public:
      // All the entries in these data structures are ordered by the
      // order of the fields in the original region requirements
      std::vector<CopySrcDstField> src_fields, dst_fields;
#ifdef LEGION_SPY
      std::vector<Realm::CopySrcDstField> realm_src_fields, realm_dst_fields;
      RegionTreeID src_tree_id, dst_tree_id;
      unsigned unique_indirections_identifier;
#endif
    public:
      // All the 'instances' in the entries in these data strctures are
      // ordered by the order of the fields in the origin region requirements
      std::vector<IndirectRecord> src_indirections, dst_indirections;
      FieldID src_indirect_field, dst_indirect_field;
      PhysicalInstance src_indirect_instance, dst_indirect_instance;
#ifdef LEGION_SPY
      ApEvent src_indirect_instance_event, dst_indirect_instance_event;
#endif
      TypeTag src_indirect_type, dst_indirect_type;
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
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag, 
                                           bool need_tight_result) = 0;
      virtual Domain get_domain(ApEvent &ready, bool need_tight) = 0; 
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
      virtual bool try_add_canonical_reference(DistributedID source) = 0;
      virtual bool remove_canonical_reference(DistributedID source) = 0;
      virtual bool try_add_live_reference(ReferenceSource source) = 0;
      virtual bool remove_live_reference(ReferenceSource source) = 0;
      virtual void add_base_expression_reference(ReferenceSource source,
          ReferenceMutator *mutator = NULL, unsigned count = 1) = 0;
      virtual void add_nested_expression_reference(DistributedID source,
          std::set<RtEvent> &applied_events, unsigned count = 1) = 0;
      virtual void add_nested_expression_reference(DistributedID source,
          ReferenceMutator *mutator = NULL, unsigned count = 1) = 0;
      virtual bool remove_base_expression_reference(ReferenceSource source,
                                                    unsigned count = 1) = 0;
      virtual bool remove_nested_expression_reference(DistributedID source,
                                                      unsigned count = 1) = 0;
      virtual void add_tree_expression_reference(DistributedID source,
                                                 unsigned count = 1) = 0;
      virtual bool remove_tree_expression_reference(DistributedID source,
                                                    unsigned count = 1) = 0;
    public:
      virtual IndexSpaceNode* create_node(IndexSpace handle,
                      DistributedID did, RtEvent initialized,
                      Provenance *provenance, std::set<RtEvent> *applied) = 0;
      virtual PieceIteratorImpl* create_piece_iterator(const void *piece_list,
                    size_t piece_list_size, IndexSpaceNode *privilege_node) = 0;
    public:
      virtual ApEvent issue_fill(const PhysicalTraceInfo &trace_info,
                           const std::vector<CopySrcDstField> &dst_fields,
                           const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                           UniqueID fill_uid,
                           FieldSpace handle,
                           RegionTreeID tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard) = 0;
      virtual ApEvent issue_copy(const PhysicalTraceInfo &trace_info,
                           const std::vector<CopySrcDstField> &dst_fields,
                           const std::vector<CopySrcDstField> &src_fields,
                           const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                           RegionTreeID src_tree_id,
                           RegionTreeID dst_tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard) = 0;
      virtual CopyAcrossUnstructured* create_across_unstructured(
                           const std::map<Reservation,bool> &reservations,
                           const bool compute_preimages) = 0;
      virtual Realm::InstanceLayoutGeneric* create_layout(
                           const LayoutConstraintSet &constraints,
                           const std::vector<FieldID> &field_ids,
                           const std::vector<size_t> &field_sizes,
                           bool compact,LayoutConstraintKind *unsat_kind = NULL,
                           unsigned *unsat_index = NULL,void **piece_list =NULL,
                           size_t *piece_list_size = NULL) = 0;
      // Return the expression with a resource ref on the expression
      virtual IndexSpaceExpression* create_layout_expression(
                           const void *piece_list, size_t piece_list_size) = 0;
      virtual bool meets_layout_expression(IndexSpaceExpression *expr,
         bool tight_bounds, const void *piece_list, size_t piece_list_size) = 0;
    public:
      virtual IndexSpaceExpression* find_congruent_expression(
                  std::set<IndexSpaceExpression*> &expressions) = 0;
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
        if (!has_empty)
        {
          empty = check_empty();
          __sync_synchronize();
          has_empty = true;
        }
        return empty;
      }
      inline size_t get_num_dims(void) const
        { return NT_TemplateHelper::get_dim(type_tag); }
      inline void record_remote_owner_valid_reference(void)
        { remote_owner_valid_references.fetch_add(1); }
    public:
      // Convert this index space expression to the canonical one that
      // represents all expressions that are all congruent
      IndexSpaceExpression* get_canonical_expression(RegionTreeForest *forest);
    protected:
      template<int DIM, typename T>
      inline ApEvent issue_fill_internal(RegionTreeForest *forest,
                               const Realm::IndexSpace<DIM,T> &space,
                               const PhysicalTraceInfo &trace_info,
                               const std::vector<CopySrcDstField> &dst_fields,
                               const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                               UniqueID fill_uid,
                               FieldSpace handle,
                               RegionTreeID tree_id,
#endif
                               ApEvent precondition, PredEvent pred_guard);
      template<int DIM, typename T>
      inline ApEvent issue_copy_internal(RegionTreeForest *forest,
                               const Realm::IndexSpace<DIM,T> &space,
                               const PhysicalTraceInfo &trace_info,
                               const std::vector<CopySrcDstField> &dst_fields,
                               const std::vector<CopySrcDstField> &src_fields,
                               const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                               RegionTreeID src_tree_id,
                               RegionTreeID dst_tree_id,
#endif
                               ApEvent precondition, PredEvent pred_guard);
      template<int DIM, typename T>
      inline Realm::InstanceLayoutGeneric* create_layout_internal(
                               const Realm::IndexSpace<DIM,T> &space,
                               const LayoutConstraintSet &constraints,
                               const std::vector<FieldID> &field_ids,
                               const std::vector<size_t> &field_sizes,
                               bool compact, LayoutConstraintKind *unsat_kind,
                               unsigned *unsat_index, void **piece_list = NULL,
                               size_t *piece_list_size = NULL) const;
      template<int DIM, typename T>
      inline IndexSpaceExpression* create_layout_expression_internal(
                               RegionTreeForest *context,
                               const Realm::IndexSpace<DIM,T> &space,
                               const Rect<DIM,T> *rects, size_t num_rects);
      template<int DIM, typename T>
      inline bool meets_layout_expression_internal(
                         IndexSpaceExpression *space_expr, bool tight_bounds,
                         const Rect<DIM,T> *piece_list, size_t piece_list_size);
    public:
      template<int DIM, typename T>
      inline IndexSpaceExpression* find_congruent_expression_internal(
                        std::set<IndexSpaceExpression*> &expressions);
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
      std::atomic<unsigned> remote_owner_valid_references;
      size_t volume;
      bool has_volume;
      bool empty, has_empty;
    };

    /**
     * This is a move-only object that tracks temporary references to
     * index space expressions that are returned from region tree ops
     */
    class IndexSpaceExprRef {
    public:
      IndexSpaceExprRef(void) : expr(NULL) { }
      IndexSpaceExprRef(IndexSpaceExpression *e, ReferenceMutator *m = NULL)
        : expr(e)
      { 
        if (expr != NULL)
        {
          if (m == NULL)
          {
            LocalReferenceMutator local_mutator;
            expr->add_base_expression_reference(LIVE_EXPR_REF, &local_mutator);
          }
          else
            expr->add_base_expression_reference(LIVE_EXPR_REF, m);
        }
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
      class InactiveFunctor {
      public:
        InactiveFunctor(IndexSpaceOperation *o, ReferenceMutator *m)
          : op(o), mutator(m) { }
      public:
        void apply(AddressSpaceID target);
      public:
        IndexSpaceOperation *const op;
        ReferenceMutator *const mutator;
      };
    public:
      IndexSpaceOperation(TypeTag tag, OperationKind kind,
                          RegionTreeForest *ctx);
      IndexSpaceOperation(TypeTag tag, RegionTreeForest *ctx,
          IndexSpaceExprID eid, DistributedID did, AddressSpaceID owner,
          IndexSpaceOperation *origin);
      virtual ~IndexSpaceOperation(void);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag, 
                                           bool need_tight_result) = 0;
      virtual Domain get_domain(ApEvent &ready, bool need_tight) = 0;
      virtual void tighten_index_space(void) = 0;
      virtual bool check_empty(void) = 0;
      virtual size_t get_volume(void) = 0;
      virtual void pack_expression(Serializer &rez, AddressSpaceID target) = 0;
      virtual void pack_expression_value(Serializer &rez,
                                         AddressSpaceID target) = 0;
    public:
#ifdef DEBUG_LEGION
      virtual bool is_valid(void) { return check_valid(); }
#endif
      virtual DistributedID get_distributed_id(void) const { return did; }
      virtual bool try_add_canonical_reference(DistributedID source);
      virtual bool remove_canonical_reference(DistributedID source);
      virtual bool try_add_live_reference(ReferenceSource source);
      virtual bool remove_live_reference(ReferenceSource source);
      virtual void add_base_expression_reference(ReferenceSource source,
          ReferenceMutator *mutator = NULL, unsigned count = 1);
      virtual void add_nested_expression_reference(DistributedID source,
          std::set<RtEvent> &applied_events, unsigned count = 1);
      virtual void add_nested_expression_reference(DistributedID source,
          ReferenceMutator *mutator = NULL, unsigned count = 1);
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
      virtual IndexSpaceNode* create_node(IndexSpace handle,
                      DistributedID did, RtEvent initialized,
                      Provenance *provenance, std::set<RtEvent> *applied) = 0;
    public:
      RegionTreeForest *const context;
      IndexSpaceOperation *const origin_expr;
      const OperationKind op_kind;
    protected:
      mutable LocalLock inter_lock;
      std::atomic<int> invalidated;
#ifdef DEBUG_LEGION
    private:
      bool tree_active;
#endif
    };

    template<int DIM, typename T>
    class IndexSpaceOperationT : public IndexSpaceOperation {
    public:
      IndexSpaceOperationT(OperationKind kind, RegionTreeForest *ctx);
      IndexSpaceOperationT(RegionTreeForest *ctx, IndexSpaceExprID eid,
          DistributedID did, AddressSpaceID owner, IndexSpaceOperation *op,
          TypeTag tag, Deserializer &derez);
      virtual ~IndexSpaceOperationT(void);
    public:
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag,
                                           bool need_tight_result);
      virtual Domain get_domain(ApEvent &ready, bool need_tight);
      virtual void tighten_index_space(void);
      virtual bool check_empty(void);
      virtual size_t get_volume(void);
      virtual void pack_expression(Serializer &rez, AddressSpaceID target);
      virtual void pack_expression_value(Serializer &rez,
                                         AddressSpaceID target) = 0;
      virtual bool invalidate_operation(void) = 0;
      virtual void remove_operation(void) = 0;
      virtual IndexSpaceNode* create_node(IndexSpace handle,
                          DistributedID did, RtEvent initialized,
                          Provenance *provenance, std::set<RtEvent> *applied);
      virtual PieceIteratorImpl* create_piece_iterator(const void *piece_list,
                      size_t piece_list_size, IndexSpaceNode *privilege_node);
    public:
      virtual ApEvent issue_fill(const PhysicalTraceInfo &trace_info,
                           const std::vector<CopySrcDstField> &dst_fields,
                           const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                           UniqueID fill_uid,
                           FieldSpace handle,
                           RegionTreeID tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard);
      virtual ApEvent issue_copy(const PhysicalTraceInfo &trace_info,
                           const std::vector<CopySrcDstField> &dst_fields,
                           const std::vector<CopySrcDstField> &src_fields,
                           const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                           RegionTreeID src_tree_id,
                           RegionTreeID dst_tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard);
      virtual CopyAcrossUnstructured* create_across_unstructured(
                           const std::map<Reservation,bool> &reservations,
                           const bool compute_preimages);
      virtual Realm::InstanceLayoutGeneric* create_layout(
                           const LayoutConstraintSet &constraints,
                           const std::vector<FieldID> &field_ids,
                           const std::vector<size_t> &field_sizes,
                           bool compact,LayoutConstraintKind *unsat_kind = NULL,
                           unsigned *unsat_index = NULL,void **piece_list =NULL, 
                           size_t *piece_list_size = NULL);
      virtual IndexSpaceExpression* create_layout_expression(
                           const void *piece_list, size_t piece_list_size);
      virtual bool meets_layout_expression(IndexSpaceExpression *expr,
         bool tight_bounds, const void *piece_list, size_t piece_list_size);
    public:
      virtual IndexSpaceExpression* find_congruent_expression(
                  std::set<IndexSpaceExpression*> &expressions);
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
          DistributedID did, AddressSpaceID own, IndexSpaceOperation *op,
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
        AddressSpaceID owner_space;
        creator->derez.deserialize(owner_space);
        IndexSpaceOperation *origin;
        creator->derez.deserialize(origin);
#ifdef DEBUG_LEGION
        assert(creator->operation == NULL);
#endif
        creator->operation =
            new RemoteExpression<N::N,T>(creator->forest, expr_id, did,
              owner_space, origin, creator->type_tag, creator->derez);
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
    class IndexTreeNode : public DistributedCollectable {
    public:
      struct SendNodeRecord {
      public:
        SendNodeRecord(IndexTreeNode *n, bool valid = false,
            bool add = false, bool pack = false, bool has_ref = false)
          : node(n), still_valid(valid), add_root_reference(add),
            pack_space(pack), has_reference(has_ref) { }
      public:
        IndexTreeNode *node;
        bool still_valid;
        bool add_root_reference;
        bool pack_space;
        bool has_reference;
      };
    public:
      IndexTreeNode(RegionTreeForest *ctx, unsigned depth,
                    LegionColor color, DistributedID did,
                    AddressSpaceID owner, RtEvent init_event,
                    Provenance *provenance);
      virtual ~IndexTreeNode(void);
    public:
      virtual void notify_active(ReferenceMutator *mutator) { }
      virtual void notify_inactive(ReferenceMutator *mutator) { }
      virtual void notify_valid(ReferenceMutator *mutator) = 0;
      virtual void notify_invalid(ReferenceMutator *mutator) = 0;
    public:
      virtual IndexTreeNode* get_parent(void) const = 0;
      virtual void get_colors(std::vector<LegionColor> &colors) = 0;
      virtual bool send_node(AddressSpaceID target, RtEvent done,
                             RtEvent &send_precondition,
                             std::set<IndexTreeNode*> &visited,
                             std::vector<SendNodeRecord> &nodes_to_send,
                             const bool above = false) = 0;
      virtual void pack_node(Serializer &rez, AddressSpaceID target,
                             const SendNodeRecord &record) = 0;
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
    protected:
      std::map<std::pair<LegionColor,LegionColor>,RtEvent> pending_tests;
    protected:
      // Map tracking send events for creating this tree node on remote nodes
      std::map<AddressSpaceID,RtEvent> send_effects;
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
                                IndexPartNode *l, IndexPartNode *r);
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
      class InactiveFunctor {
      public:
        InactiveFunctor(IndexSpaceNode *n, ReferenceMutator *m,
                        std::map<AddressSpaceID,RtEvent> &effects)
          : node(n), mutator(m), send_effects(effects) { }
      public:
        void apply(AddressSpaceID target);
      public:
        IndexSpaceNode *const node;
        ReferenceMutator *const mutator;
        std::map<AddressSpaceID,RtEvent> &send_effects;
      };
      class InvalidateRootFunctor {
      public:
        InvalidateRootFunctor(AddressSpaceID src, IndexSpaceNode *n, 
                              ReferenceMutator &m, Runtime *rt,
                              const std::map<AddressSpaceID,RtEvent> &e)
          : source(src), node(n), runtime(rt), mutator(m), effects(e) { }
      public:
        void apply(AddressSpaceID target);
      public:
        const AddressSpaceID source;
        IndexSpaceNode *const node;
        Runtime *const runtime;
        ReferenceMutator &mutator;
        const std::map<AddressSpaceID,RtEvent> &effects;
      };
    public:
      IndexSpaceNode(RegionTreeForest *ctx, IndexSpace handle,
                     IndexPartNode *parent, LegionColor color,
                     DistributedID did, ApEvent index_space_ready,
                     IndexSpaceExprID expr_id, RtEvent initialized,
                     unsigned depth, Provenance *provenance);
      IndexSpaceNode(const IndexSpaceNode &rhs);
      virtual ~IndexSpaceNode(void);
    public:
      IndexSpaceNode& operator=(const IndexSpaceNode &rhs);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
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
    public:
      bool are_disjoint(LegionColor c1, LegionColor c2); 
      void record_disjointness(bool disjoint, 
                               LegionColor c1, LegionColor c2);
      void record_remote_child(IndexPartition pid, LegionColor part_color);
    public:
      static void handle_disjointness_test(const void *args);
    public:
      virtual bool send_node(AddressSpaceID target, RtEvent done,
                             RtEvent &send_precondition,
                             std::set<IndexTreeNode*> &visited,
                             std::vector<SendNodeRecord> &nodes_to_send,
                             const bool above = false);
      virtual void pack_node(Serializer &rez, AddressSpaceID target,
                             const SendNodeRecord &record);
      void invalidate_tree(void);
      void invalidate_root(AddressSpaceID source,
                           std::set<RtEvent> &applied);
      static void handle_node_creation(RegionTreeForest *context,
                                       Deserializer &derez, 
                                       AddressSpaceID source);
    public:
      static void handle_node_request(RegionTreeForest *context,
                                      Deserializer &derez,
                                      AddressSpaceID source);
      static void handle_node_return(RegionTreeForest *context,
                                     Deserializer &derez,
                                     AddressSpaceID source);
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
      virtual Domain get_domain(ApEvent &ready, bool need_tight) = 0;
      virtual bool set_domain(const Domain &domain, AddressSpaceID space) = 0;
      virtual void tighten_index_space(void) = 0;
      virtual bool check_empty(void) = 0;
      virtual void pack_expression(Serializer &rez, AddressSpaceID target);
      virtual void pack_expression_value(Serializer &rez,AddressSpaceID target);
    public:
#ifdef DEBUG_LEGION
      virtual bool is_valid(void) { return check_valid(); }
#endif
      virtual DistributedID get_distributed_id(void) const { return did; }
      virtual bool try_add_canonical_reference(DistributedID source);
      virtual bool remove_canonical_reference(DistributedID source);
      virtual bool try_add_live_reference(ReferenceSource source);
      virtual bool remove_live_reference(ReferenceSource source);
      virtual void add_base_expression_reference(ReferenceSource source,
          ReferenceMutator *mutator = NULL, unsigned count = 1);
      virtual void add_nested_expression_reference(DistributedID source,
          std::set<RtEvent> &applied_events, unsigned count = 1);
      virtual void add_nested_expression_reference(DistributedID source,
          ReferenceMutator *mutator = NULL, unsigned count = 1);
      virtual bool remove_base_expression_reference(ReferenceSource source,
                                                    unsigned count = 1);
      virtual bool remove_nested_expression_reference(DistributedID source,
                                                      unsigned count = 1);
      virtual void add_tree_expression_reference(DistributedID source,
                                                 unsigned count = 1);
      virtual bool remove_tree_expression_reference(DistributedID source,
                                                    unsigned count = 1);
    public:
      virtual IndexSpaceNode* create_node(IndexSpace handle,
                    DistributedID did, RtEvent initialized,
                    Provenance *provenance, std::set<RtEvent> *applied) = 0; 
      virtual PieceIteratorImpl* create_piece_iterator(const void *piece_list,
                    size_t piece_list_size, IndexSpaceNode *privilege_node) = 0;
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
      // Caller takes ownership for the iterator
      virtual ColorSpaceIterator* create_color_space_iterator(void) = 0;
    public:
      bool intersects_with(IndexSpaceNode *rhs,bool compute = true);
      bool intersects_with(IndexPartNode *rhs, bool compute = true);
      bool dominates(IndexSpaceNode *rhs);
      bool dominates(IndexPartNode *rhs);
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
                                       FutureMapImpl *future_map,
                                       bool perform_intersections) = 0;
      virtual ApEvent create_by_weights(Operation *op,
                                        IndexPartNode *partition,
                                        FutureMapImpl *future_map,
                                        size_t granularity) = 0;
      virtual ApEvent create_by_field(Operation *op,
                                      IndexPartNode *partition,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready) = 0;
      virtual ApEvent create_by_image(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready) = 0;
      virtual ApEvent create_by_image_range(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready) = 0;
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
      virtual size_t get_coordinate_size(bool range) const = 0;
    public:
      virtual PhysicalInstance create_file_instance(const char *file_name,
				   const std::vector<Realm::FieldID> &field_ids,
                                   const std::vector<size_t> &field_sizes,
                                   legion_file_mode_t file_mode,
                                   ApEvent &ready_event) = 0;
      virtual PhysicalInstance create_hdf5_instance(const char *file_name,
                                   const std::vector<Realm::FieldID> &field_ids,
                                   const std::vector<size_t> &field_sizes,
                                   const std::vector<const char*> &field_files,
                                   const OrderingConstraint &dimension_order,
                                   bool read_only, ApEvent &ready_event) = 0;
    public:
      virtual void get_launch_space_domain(Domain &launch_domain) = 0;
      virtual void validate_slicing(const std::vector<IndexSpace> &slice_spaces,
                                    MultiTask *task, MapperManager *mapper) = 0;
      virtual void log_launch_space(UniqueID op_id) = 0;
    public:
      const IndexSpace handle;
      IndexPartNode *const parent;
      const ApEvent index_space_ready;
    protected:
      // Must hold the node lock when accessing these data structures
      std::map<LegionColor,IndexPartNode*> color_map;
      std::map<LegionColor,IndexPartition> remote_colors;
      std::set<RegionNode*> logical_nodes;
      std::set<std::pair<LegionColor,LegionColor> > disjoint_subsets;
      std::set<std::pair<LegionColor,LegionColor> > aliased_subsets;
    protected:
      unsigned                  send_references; 
      // On the owner node track when the index space is set
      RtUserEvent               realm_index_space_set;
      // Keep track of whether we've tightened these bounds
      RtUserEvent               tight_index_space_set;
      bool                      tight_index_space;
      // Keep track of whether we're still valid on the owner
      bool                      tree_valid;
#ifdef DEBUG_LEGION
      // Keep track of whether we are active, should only happen once
      bool                      tree_active;
#endif
      // Keep track of whether we've had our application 
      // reference removed if this is a root node 
      bool                      root_valid;
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
                      const void *bounds, bool is_domain,
                      DistributedID did, ApEvent ready_event,
                      IndexSpaceExprID expr_id, RtEvent init,
                      unsigned depth, Provenance *provenance);
      IndexSpaceNodeT(const IndexSpaceNodeT &rhs);
      virtual ~IndexSpaceNodeT(void);
    public:
      IndexSpaceNodeT& operator=(const IndexSpaceNodeT &rhs);
    public:
      ApEvent get_realm_index_space(Realm::IndexSpace<DIM,T> &result,
				    bool need_tight_result);
      bool set_realm_index_space(AddressSpaceID source,
				 const Realm::IndexSpace<DIM,T> &value);
    public:
      // From IndexSpaceExpression
      virtual ApEvent get_expr_index_space(void *result, TypeTag tag,
                                           bool need_tight_result);
      virtual Domain get_domain(ApEvent &ready, bool need_tight);
      virtual bool set_domain(const Domain &domain, AddressSpaceID space);
      virtual void tighten_index_space(void);
      virtual bool check_empty(void);
      virtual IndexSpaceNode* create_node(IndexSpace handle,
                            DistributedID did, RtEvent initialized,
                            Provenance *provenance, std::set<RtEvent> *applied);
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
      virtual LegionColor linearize_color(const void *realm_color,
                                          TypeTag type_tag);
      LegionColor linearize_color(Point<DIM,T> color); 
      virtual void delinearize_color(LegionColor color, 
                                     void *realm_color, TypeTag type_tag);
      virtual bool contains_color(LegionColor color,
                                  bool report_error = false);
      virtual void instantiate_colors(std::vector<LegionColor> &colors);
      virtual Domain get_color_space_domain(void);
      virtual DomainPoint get_domain_point_color(void) const;
      virtual DomainPoint delinearize_color_to_point(LegionColor c);
      // Caller takes ownership for the iterator
      virtual ColorSpaceIterator* create_color_space_iterator(void);
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
                                       FutureMapImpl *future_map,
                                       bool perform_intersections);
      template<int COLOR_DIM, typename COLOR_T>
      ApEvent create_by_domain_helper(Operation *op,
                                      IndexPartNode *partition,
                                      FutureMapImpl *future_map,
                                      bool perform_intersections);
      virtual ApEvent create_by_weights(Operation *op,
                                        IndexPartNode *partition,
                                        FutureMapImpl *future_map,
                                        size_t granularity);
      template<int COLOR_DIM, typename COLOR_T>
      ApEvent create_by_weight_helper(Operation *op,
                                      IndexPartNode *partition,
                                      FutureMapImpl *future_map,
                                      size_t granularity);
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
                                      ApEvent instances_ready);
      template<int DIM2, typename T2>
      ApEvent create_by_image_helper(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      virtual ApEvent create_by_image_range(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
      template<int DIM2, typename T2>
      ApEvent create_by_image_range_helper(Operation *op,
                                      IndexPartNode *partition,
                                      IndexPartNode *projection,
                const std::vector<FieldDataDescriptor> &instances,
                                      ApEvent instances_ready);
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
      virtual size_t get_coordinate_size(bool range) const;
    public:
      virtual PhysicalInstance create_file_instance(const char *file_name,
                                   const std::vector<Realm::FieldID> &field_ids,
                                   const std::vector<size_t> &field_sizes,
                                   legion_file_mode_t file_mode, 
                                   ApEvent &ready_event);
      virtual PhysicalInstance create_hdf5_instance(const char *file_name,
                                   const std::vector<Realm::FieldID> &field_ids,
                                   const std::vector<size_t> &field_sizes,
                                   const std::vector<const char*> &field_files,
                                   const OrderingConstraint &dimension_order,
                                   bool read_only, ApEvent &ready_event);
    public:
      virtual ApEvent issue_fill(const PhysicalTraceInfo &trace_info,
                           const std::vector<CopySrcDstField> &dst_fields,
                           const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                           UniqueID fill_uid,
                           FieldSpace handle,
                           RegionTreeID tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard);
      virtual ApEvent issue_copy(const PhysicalTraceInfo &trace_info,
                           const std::vector<CopySrcDstField> &dst_fields,
                           const std::vector<CopySrcDstField> &src_fields,
                           const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                           RegionTreeID src_tree_id,
                           RegionTreeID dst_tree_id,
#endif
                           ApEvent precondition, PredEvent pred_guard);
      virtual CopyAcrossUnstructured* create_across_unstructured(
                           const std::map<Reservation,bool> &reservations,
                           const bool compute_preimages);
      virtual Realm::InstanceLayoutGeneric* create_layout(
                           const LayoutConstraintSet &constraints,
                           const std::vector<FieldID> &field_ids,
                           const std::vector<size_t> &field_sizes,
                           bool compact,LayoutConstraintKind *unsat_kind = NULL,
                           unsigned *unsat_index = NULL,void **piece_list =NULL, 
                           size_t *piece_list_size = NULL);
      virtual IndexSpaceExpression* create_layout_expression(
                           const void *piece_list, size_t piece_list_size);
      virtual bool meets_layout_expression(IndexSpaceExpression *expr,
         bool tight_bounds, const void *piece_list, size_t piece_list_size);
    public:
      virtual IndexSpaceExpression* find_congruent_expression(
                  std::set<IndexSpaceExpression*> &expressions);
    public:
      virtual void get_launch_space_domain(Domain &launch_domain);
      virtual void validate_slicing(const std::vector<IndexSpace> &slice_spaces,
                                    MultiTask *task, MapperManager *mapper);
      virtual void log_launch_space(UniqueID op_id);
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
      struct CreateByDomainHelper {
      public:
        CreateByDomainHelper(IndexSpaceNodeT<DIM,T> *n,
                              IndexPartNode *p, Operation *o,
                              FutureMapImpl *fm, bool inter)
          : node(n), partition(p), op(o), future_map(fm), intersect(inter) { }
      public:
        template<typename COLOR_DIM, typename COLOR_T>
        static inline void demux(CreateByDomainHelper *creator)
        {
          creator->result = creator->node->template 
            create_by_domain_helper<COLOR_DIM::N,COLOR_T>(creator->op,
                creator->partition, creator->future_map, creator->intersect);
        }
      public:
        IndexSpaceNodeT<DIM,T> *const node;
        IndexPartNode *const partition;
        Operation *const op;
        FutureMapImpl *const future_map;
        const bool intersect;
        ApEvent result;
      };
      struct CreateByWeightHelper {
      public:
        CreateByWeightHelper(IndexSpaceNodeT<DIM,T> *n,
                             IndexPartNode *p, Operation *o,
                             FutureMapImpl *fm, size_t g)
          : node(n), partition(p), op(o), future_map(fm), granularity(g) { }
      public:
        template<typename COLOR_DIM, typename COLOR_T>
        static inline void demux(CreateByWeightHelper *creator)
        {
          creator->result = creator->node->template 
            create_by_weight_helper<COLOR_DIM::N,COLOR_T>(creator->op,
                creator->partition, creator->future_map, creator->granularity);
        }
      public:
        IndexSpaceNodeT<DIM,T> *const node;
        IndexPartNode *const partition;
        Operation *const op;
        FutureMapImpl *const future_map;
        const size_t granularity;
        ApEvent result;
      };
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
           creator->op, creator->partition, creator->instances, creator->ready);
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
                            ApEvent r)
          : node(n), op(o), partition(p), projection(j), 
            instances(i), ready(r) { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByImageHelper *creator)
        {
          creator->result = 
           creator->node->template create_by_image_helper<DIM2::N,T2>(
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
      struct CreateByImageRangeHelper {
      public:
        CreateByImageRangeHelper(IndexSpaceNodeT<DIM,T> *n,
                            Operation *o, IndexPartNode *p, IndexPartNode *j,
                            const std::vector<FieldDataDescriptor> &i,
                            ApEvent r)
          : node(n), op(o), partition(p), projection(j), 
            instances(i), ready(r) { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByImageRangeHelper *creator)
        {
          creator->result = creator->node->template 
            create_by_image_range_helper<DIM2::N,T2>(
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
     * \class ColorSpaceIterator
     * A helper class for iterating over sparse color spaces
     * It can be used for non-sparse spaces as well, but we
     * usually have more efficient ways of iterating over those
     */
    class ColorSpaceIterator {
    public:
      virtual ~ColorSpaceIterator(void) { }
    public:
      virtual bool is_valid(void) const = 0;
      virtual LegionColor yield_color(void) = 0;
    };

    template<int DIM, typename T>
    class ColorSpaceIteratorT : public ColorSpaceIterator, 
                                public PointInDomainIterator<DIM,T> {
    public:
      ColorSpaceIteratorT(const DomainT<DIM,T> &d,
                          IndexSpaceNodeT<DIM,T> *color_space);
      virtual ~ColorSpaceIteratorT(void) { }
    public:
      virtual bool is_valid(void) const;
      virtual LegionColor yield_color(void);
    public:
      IndexSpaceNodeT<DIM,T> *const color_space;
    };

    /**
     * \class IndexSpaceCreator
     * A small helper class for creating templated index spaces
     */
    class IndexSpaceCreator {
    public:
      IndexSpaceCreator(RegionTreeForest *f, IndexSpace s, const void *b,
                        bool is_dom, IndexPartNode *p, LegionColor c, 
                        DistributedID d, ApEvent r, IndexSpaceExprID e,
                        RtEvent init, unsigned dp, Provenance *prov)
        : forest(f), space(s), bounds(b), is_domain(is_dom), parent(p), 
          color(c), did(d), ready(r), expr_id(e), initialized(init), depth(dp),
          provenance(prov), result(NULL) { }
    public:
      template<typename N, typename T>
      static inline void demux(IndexSpaceCreator *creator)
      {
        creator->result = new IndexSpaceNodeT<N::N,T>(creator->forest,
            creator->space, creator->parent, creator->color, creator->bounds,
            creator->is_domain, creator->did, creator->ready, creator->expr_id,
            creator->initialized, creator->depth, creator->provenance);
      }
    public:
      RegionTreeForest *const forest;
      const IndexSpace space; 
      const void *const bounds;
      const bool is_domain;
      IndexPartNode *const parent;
      const LegionColor color;
      const DistributedID did;
      const ApEvent ready;
      const IndexSpaceExprID expr_id;
      const RtEvent initialized;
      const unsigned depth;
      Provenance *const provenance;
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
      bool remove_partition_reference(ReferenceMutator *mutator);
    private:
      PartitionNode *const partition;
    };

    /**
     * \class IndexPartNode
     * A node for representing a generic index partition.
     */
    class IndexPartNode : public IndexTreeNode {
    public:
      struct DynamicIndependenceArgs : 
        public LgTaskArgs<DynamicIndependenceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_SPACE_INDEPENDENCE_TASK_ID;
      public:
        DynamicIndependenceArgs(IndexPartNode *par, 
                                IndexSpaceNode *l, IndexSpaceNode *r);
      public:
        IndexPartNode *const parent;
        IndexSpaceNode *const left, *const right;
      };
      struct PendingChildArgs : public LgTaskArgs<PendingChildArgs> {
      public:
        static const LgTaskID TASK_ID = LG_PENDING_CHILD_TASK_ID;
      public:
        PendingChildArgs(IndexPartNode *par, LegionColor child)
          : LgTaskArgs<PendingChildArgs>(implicit_provenance),
            parent(par), pending_child(child) { }
      public:
        IndexPartNode *const parent;
        const LegionColor pending_child;
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
                       std::atomic<IndexSpaceID> *tar,
                       RtUserEvent trig, AddressSpaceID src)
          : LgTaskArgs<DeferChildArgs>(implicit_provenance),
            proxy_this(proxy), child_color(child), target(tar),
            to_trigger(trig), source(src) { }
      public:
        IndexPartNode *const proxy_this;
        const LegionColor child_color;
        std::atomic<IndexSpaceID> *const target;
        const RtUserEvent to_trigger;
        const AddressSpaceID source;
      };
      class RemoteDisjointnessFunctor {
      public:
        RemoteDisjointnessFunctor(Serializer &r, Runtime *rt)
          : rez(r), runtime(rt) { }
      public:
        void apply(AddressSpaceID target);
      public:
        Serializer &rez;
        Runtime *const runtime;
      };
      class InvalidFunctor {
      public:
        InvalidFunctor(IndexPartNode *n, ReferenceMutator *m,
                       std::map<AddressSpaceID,RtEvent> &effects)
          : node(n), mutator(m), send_effects(effects) { }
      public:
        void apply(AddressSpaceID target);
      public:
        IndexPartNode *const node;
        ReferenceMutator *const mutator;
        std::map<AddressSpaceID,RtEvent> &send_effects;
      }; 
    public:
      IndexPartNode(RegionTreeForest *ctx, IndexPartition p,
                    IndexSpaceNode *par, IndexSpaceNode *color_space,
                    LegionColor c, bool disjoint, int complete, 
                    DistributedID did, ApEvent partition_ready, 
                    ApUserEvent partial_pending, RtEvent init,
                    Provenance *provenance);
      IndexPartNode(RegionTreeForest *ctx, IndexPartition p,
                    IndexSpaceNode *par, IndexSpaceNode *color_space,
                    LegionColor c, RtEvent disjointness_ready,
                    int complete, DistributedID did, ApEvent partition_ready,
                    ApUserEvent partial_pending, RtEvent init,
                    Provenance *provenance);
      IndexPartNode(const IndexPartNode &rhs);
      virtual ~IndexPartNode(void);
    public:
      IndexPartNode& operator=(const IndexPartNode &rhs);
    public:
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
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
          const void *buffer, size_t size, bool is_mutable, RtUserEvent ready);
      void process_semantic_request(SemanticTag tag, AddressSpaceID source,
                            bool can_fail, bool wait_until, RtUserEvent ready);
      static void handle_semantic_request(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
      static void handle_semantic_info(RegionTreeForest *forest,
                                   Deserializer &derez, AddressSpaceID source);
    public:
      bool has_color(const LegionColor c);
      IndexSpaceNode* get_child(const LegionColor c, RtEvent *defer = NULL);
      bool add_child(IndexSpaceNode *child);
      void add_tracker(PartitionTracker *tracker); 
      size_t get_num_children(void) const;
      void get_subspace_preconditions(std::set<ApEvent> &preconditions);
    public:
      void compute_disjointness(RtUserEvent ready_event);
      bool is_disjoint(bool from_app = false);
      bool are_disjoint(LegionColor c1, LegionColor c2,
                        bool force_compute = false);
      void record_disjointness(bool disjoint,
                               LegionColor c1, LegionColor c2);
      bool is_complete(bool from_app = false, bool false_if_not_ready = false);
      IndexSpaceExpression* get_union_expression(bool check_complete=true);
      void record_remote_disjoint_ready(RtUserEvent ready);
      void record_remote_disjoint_result(const bool disjoint_result);
    public:
      void add_pending_child(const LegionColor child_color,
                            ApUserEvent domain_ready);
      bool get_pending_child(const LegionColor child_color,
                             ApUserEvent &domain_ready);
      void remove_pending_child(const LegionColor child_color);
      static void handle_pending_child_task(const void *args);
    public:
      ApEvent create_equal_children(Operation *op, size_t granularity);
      ApEvent create_by_weights(Operation *op, const FutureMap &weights,
                                size_t granularity);
      ApEvent create_by_union(Operation *Op,
                              IndexPartNode *left, IndexPartNode *right);
      ApEvent create_by_intersection(Operation *op,
                              IndexPartNode *left, IndexPartNode *right);
      ApEvent create_by_intersection(Operation *op, IndexPartNode *original,
                                     const bool dominates);
      ApEvent create_by_difference(Operation *op,
                              IndexPartNode *left, IndexPartNode *right);
      ApEvent create_by_restriction(const void *transform, const void *extent);
      ApEvent create_by_domain(FutureMapImpl *future_map);
    public:
      bool compute_complete(void);
      bool intersects_with(IndexSpaceNode *other, bool compute = true);
      bool intersects_with(IndexPartNode *other, bool compute = true); 
      bool dominates(IndexSpaceNode *other);
      bool dominates(IndexPartNode *other);
    public:
      static void handle_disjointness_test(const void *args);
    public:
      virtual bool send_node(AddressSpaceID target, RtEvent done,
                             RtEvent &send_precondition,
                             std::set<IndexTreeNode*> &visited,
                             std::vector<SendNodeRecord> &nodes_to_send,
                             const bool above = false);
      virtual void pack_node(Serializer &rez, AddressSpaceID target,
                             const SendNodeRecord &record);
      static void handle_node_creation(RegionTreeForest *context,
                                       Deserializer &derez, 
                                       AddressSpaceID source);
    public:
      static void handle_node_request(RegionTreeForest *context,
                                      Deserializer &derez,
                                      AddressSpaceID source);
      static void handle_node_return(RegionTreeForest *context,
                                     Deserializer &derez,
                                     AddressSpaceID source);
      static void handle_node_child_request(
          RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source);
      static void defer_node_child_request(const void *args);
      static void handle_node_child_response(RegionTreeForest *forest,
                                             Deserializer &derez);
      static void handle_node_disjoint_update(RegionTreeForest *forest,
                                              Deserializer &derez);
      static void handle_notification(RegionTreeForest *context, 
                                      Deserializer &derez);
    public:
      const IndexPartition handle;
      IndexSpaceNode *const parent;
      IndexSpaceNode *const color_space;
      const LegionColor total_children;
      const LegionColor max_linearized_color;
      const ApEvent partition_ready;
      const ApUserEvent partial_pending;
    protected:
      // Must hold the node lock when accessing these data structures
      // the remaining data structures
      std::map<LegionColor,IndexSpaceNode*> color_map;
      std::map<LegionColor,RtUserEvent> pending_child_map;
      std::set<std::pair<LegionColor,LegionColor> > disjoint_subspaces;
      std::set<std::pair<LegionColor,LegionColor> > aliased_subspaces;
      std::list<PartitionTracker*> partition_trackers;
    protected:
      // Support for pending child spaces that still need to be computed
      std::map<LegionColor,ApUserEvent> pending_children;
      // Support for remote disjoint events being stored
      RtUserEvent remote_disjoint_ready;
    protected:
      RtEvent disjoint_ready;
      bool disjoint;
    protected:
      bool has_complete, complete;
      bool tree_valid;
      unsigned send_count;
      RtUserEvent send_done;
      std::atomic<IndexSpaceExpression*> union_expr;
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
                     ApUserEvent pending, RtEvent initialized,
                     Provenance *provenance);
      IndexPartNodeT(RegionTreeForest *ctx, IndexPartition p,
                     IndexSpaceNode *par, IndexSpaceNode *color_space,
                     LegionColor c, RtEvent disjointness_ready, 
                     int complete, DistributedID did, ApEvent partition_ready,
                     ApUserEvent pending, RtEvent initialized,
                     Provenance *provenance);
      IndexPartNodeT(const IndexPartNodeT &rhs);
      virtual ~IndexPartNodeT(void);
    public:
      IndexPartNodeT& operator=(const IndexPartNodeT &rhs);
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
                       ApEvent r, ApUserEvent pend, RtEvent initialized,
                       Provenance *prov)
        : forest(f), partition(p), parent(par), color_space(cs),
          color(c), disjoint(d), complete(k), did(id), ready(r), 
          pending(pend), init(initialized), provenance(prov) { }
      IndexPartCreator(RegionTreeForest *f, IndexPartition p,
                       IndexSpaceNode *par, IndexSpaceNode *cs,
                       LegionColor c, RtEvent d, int k, DistributedID id, 
                       ApEvent r, ApUserEvent pend, RtEvent initialized,
                       Provenance *prov)
        : forest(f), partition(p), parent(par), color_space(cs),
          color(c), disjoint(false), complete(k), disjoint_ready(d), did(id),
          ready(r), pending(pend), init(initialized), provenance(prov) { }
    public:
      template<typename N, typename T>
      static inline void demux(IndexPartCreator *creator)
      {
        if (creator->disjoint_ready.exists()) 
          creator->result = new IndexPartNodeT<N::N,T>(creator->forest,
              creator->partition, creator->parent, creator->color_space,
              creator->color, creator->disjoint_ready, creator->complete,
              creator->did, creator->ready, creator->pending, 
              creator->init, creator->provenance);
        else
          creator->result = new IndexPartNodeT<N::N,T>(creator->forest,
              creator->partition, creator->parent, creator->color_space,
              creator->color, creator->disjoint, creator->complete, 
              creator->did, creator->ready, creator->pending,
              creator->init, creator->provenance);
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
      const ApUserEvent pending;
      const RtEvent init;
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
                  Provenance *prov, bool loc=false);
        FieldInfo(ApEvent ready, unsigned id, CustomSerdezID sid,
                  Provenance *prov, bool loc=false);
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
      FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx, 
                     DistributedID did, RtEvent initialized,
                     Provenance *provenance);
      FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx, DistributedID did,
                     RtEvent initialized, Provenance *provenance, 
                     Deserializer &derez);
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
          RtUserEvent ready = RtUserEvent::NO_RT_USER_EVENT);
      RtEvent destroy_allocator(AddressSpaceID source);
    public:
      void initialize_fields(const std::vector<size_t> &sizes,
                             const std::vector<FieldID> &resulting_fields,
                             CustomSerdezID serdez_id,
                             const char *provenance);
      void initialize_fields(ApEvent sizes_ready,
                             const std::vector<FieldID> &resulting_fields,
                             CustomSerdezID serdez_id,
                             Provenance *provenance);
      RtEvent allocate_field(FieldID fid, size_t size,
                             CustomSerdezID serdez_id,
                             const char *provenance);
      RtEvent allocate_field(FieldID fid, ApEvent size_ready,
                             CustomSerdezID serdez_id,
                             const char *provenance);
      RtEvent allocate_fields(const std::vector<size_t> &sizes,
                              const std::vector<FieldID> &fids,
                              CustomSerdezID serdez_id,
                              const char *provenance);
      RtEvent allocate_fields(ApEvent sizes_ready,
                              const std::vector<FieldID> &fids,
                              CustomSerdezID serdez_id,
                              const char *provenance);
      void update_field_size(FieldID fid, size_t field_size, 
          std::set<RtEvent> &update_events, AddressSpaceID source);
      void free_field(FieldID fid, AddressSpaceID source,
                      std::set<RtEvent> &applied);
      void free_fields(const std::vector<FieldID> &to_free,
                       AddressSpaceID source, std::set<RtEvent> &applied);
      void free_field_indexes(const std::vector<FieldID> &to_free,
                              RtEvent freed_event); 
    public:
      bool allocate_local_fields(const std::vector<FieldID> &fields,
                                 const std::vector<size_t> &sizes,
                                 CustomSerdezID serdez_id,
                                 const std::set<unsigned> &indexes,
                                 std::vector<unsigned> &new_indexes,
                                 const char *provenance);
      void free_local_fields(const std::vector<FieldID> &to_free,
                             const std::vector<unsigned> &indexes);
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
      InstanceRef create_external_instance(
            const std::vector<FieldID> &fields, RegionNode *node, AttachOp *op);
      PhysicalManager* create_external_manager(PhysicalInstance inst,
            ApEvent ready_event, size_t instance_footprint, 
            LayoutConstraintSet &constraints, 
            const std::vector<FieldID> &field_set,
            const std::vector<size_t> &field_sizes, const FieldMask &file_mask,
            const std::vector<unsigned> &mask_index_map,
            RegionNode *node, const std::vector<CustomSerdezID> &serdez);
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
                                      Deserializer &derez,
                                      AddressSpaceID source);
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
      void process_allocator_invalidation(RtUserEvent done);
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
                     Provenance *provenance = NULL, DistributedID did = 0);
      virtual ~RegionTreeNode(void);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator) = 0;
      virtual void notify_valid(ReferenceMutator *mutator) = 0;
      virtual void notify_invalid(ReferenceMutator *mutator) = 0;
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
                                 RegionTreePath &path,
                                 const LogicalTraceInfo &trace_info,
                                 const ProjectionInfo &projection_info,
                                 FieldMask &unopened_field_mask,
                                 FieldMask &already_closed_mask,
                                 std::set<RtEvent> &applied_events);
      void register_local_user(LogicalState &state,
                               const LogicalUser &user,
                               const LogicalTraceInfo &trace_info);
      void add_open_field_state(LogicalState &state, bool arrived,
                                const ProjectionInfo &projection_info,
                                const LogicalUser &user,
                                const FieldMask &open_mask,
                                RegionTreeNode *next_child,
                                std::set<RtEvent> &applied_events);
      void close_logical_node(LogicalCloser &closer,
                              const FieldMask &closing_mask,
                              const bool read_only_close);
      void siphon_logical_children(LogicalCloser &closer,
                                   LogicalState &state,
                                   const FieldMask &closing_mask,
                                   const FieldMask *aliased_children,
                                   bool record_close_operations,
                                   RegionTreeNode *next_child,
                                   FieldMask &open_below,
                                   std::set<RtEvent> &applied_events);
      void siphon_logical_projection(LogicalCloser &closer,
                                     LogicalState &state,
                                     const FieldMask &closing_mask,
                                     const ProjectionInfo &proj_info,
                                     bool record_close_operations,
                                     FieldMask &open_below,
                                     std::set<RtEvent> &applied_events);
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
      void report_uninitialized_usage(Operation *op, unsigned index,
                                      const RegionUsage usage,
                                      const FieldMask &uninitialized,
                                      RtUserEvent reported);
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
                                     const LogicalTraceInfo &trace_info,
                                     FieldMask &already_closed_mask,
                                     std::set<RtEvent> &applied_events,
                                     bool invalidate_tree);
      void siphon_logical_deletion(LogicalCloser &closer,
                                   LogicalState &state,
                                   const FieldMask &current_mask,
                                   RegionTreeNode *next_child,
                                   FieldMask &open_below,
                                   bool force_close_next,
                                   std::set<RtEvent> &applied_events);
      void record_close_no_dependences(ContextID ctx,
                                       const LogicalUser &user);
    public:
      void send_back_logical_state(ContextID ctx, UniqueID context_uid,
                                   AddressSpaceID target);
      void process_logical_state_return(ContextID ctx, Deserializer &derez,
                                        AddressSpaceID source);
      static void handle_logical_state_return(Runtime *runtime,
                              Deserializer &derez, AddressSpaceID source); 
    public:
      void initialize_current_state(ContextID ctx);
      void invalidate_current_state(ContextID ctx, bool users_only);
      void invalidate_deleted_state(ContextID ctx, 
                                    const FieldMask &deleted_mask);
      bool invalidate_version_state(ContextID ctx);
      void invalidate_logical_states(void);
      void invalidate_version_managers(void);
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
#ifdef DEBUG_LEGION
    protected:
      bool currently_active; // should be monotonic
#endif
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
      class InvalidFunctor {
      public:
        InvalidFunctor(RegionNode *n, ReferenceMutator *m)
          : node(n), mutator(m) { }
      public:
        void apply(AddressSpaceID target);
      public:
        RegionNode *const node;
        ReferenceMutator *const mutator;
      };
    public:
      RegionNode(LogicalRegion r, PartitionNode *par, IndexSpaceNode *row_src,
             FieldSpaceNode *col_src, RegionTreeForest *ctx, 
             DistributedID did, RtEvent initialized, 
             RtEvent tree_initialized, Provenance *provenance);
      RegionNode(const RegionNode &rhs);
      virtual ~RegionNode(void);
    public:
      RegionNode& operator=(const RegionNode &rhs);
    public:
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
    public:
      void record_registered(void);
    public:
      bool has_color(const LegionColor p);
      PartitionNode* get_child(const LegionColor p);
      void add_child(PartitionNode *child);
      void remove_child(const LegionColor p);
      void add_tracker(PartitionTracker *tracker);
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
                                   Deserializer &derez, AddressSpaceID source);
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
      RtEvent perform_versioning_analysis(ContextID ctx,
                                          InnerContext *parent_ctx,
                                          VersionInfo *version_info,
                                          LogicalRegion upper_bound,
                                          const FieldMask &version_mask,
                                          Operation *op);
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
#ifdef DEBUG_LEGION
      bool currently_valid;
#endif
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
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
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

