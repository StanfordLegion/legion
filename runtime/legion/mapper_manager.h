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

#ifndef __MAPPER_MANAGER_H__
#define __MAPPER_MANAGER_H__

#include "legion/legion_types.h"
#include "legion/legion_mapping.h"

namespace Legion {
  namespace Internal {

    class MappingCallInfo {
    public:
      MappingCallInfo(MapperManager *man, MappingCallKind k,
                      Operation *op = NULL); 
    public:
      MapperManager*const               manager;
      RtUserEvent                       resume;
      MappingCallKind                   kind;
      Operation*                        operation;
      std::map<PhysicalManager*,
        std::pair<unsigned/*count*/,bool/*created*/> >* acquired_instances;
      unsigned long long                start_time;
      unsigned long long                stop_time;
    };

    /**
     * \class MapperManager
     * This is the base class for a bunch different kinds of mapper
     * managers. Some calls into this manager from the mapper will
     * be handled right away, while other we may need to defer and
     * possibly preempt.  This later class of calls are the ones that
     * are made virtual so that the 
     */
    class MapperManager {
    public:
      struct AcquireStatus {
      public:
        std::set<PhysicalManager*> instances;
        std::vector<bool> results;
      };
      struct DeferMessageArgs : public LgTaskArgs<DeferMessageArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_MAPPER_MESSAGE_TASK_ID;
      public:
        MapperManager *manager;
        Processor sender;
        unsigned kind;
        void *message;
        size_t size;
        bool broadcast;
      };
    public:
      MapperManager(Runtime *runtime, Mapping::Mapper *mapper, 
                    MapperID map_id, Processor p);
      virtual ~MapperManager(void);
    public:
      const char* get_mapper_name(void);
    public: // Task mapper calls
      void invoke_select_task_options(TaskOp *task, Mapper::TaskOptions *output,
                                      MappingCallInfo *info = NULL);
      void invoke_premap_task(TaskOp *task, Mapper::PremapTaskInput *input,
                              Mapper::PremapTaskOutput *output, 
                              MappingCallInfo *info = NULL);
      void invoke_slice_task(TaskOp *task, Mapper::SliceTaskInput *input,
                               Mapper::SliceTaskOutput *output, 
                               MappingCallInfo *info = NULL);
      void invoke_map_task(TaskOp *task, Mapper::MapTaskInput *input,
                           Mapper::MapTaskOutput *output, 
                           MappingCallInfo *info = NULL);
      void invoke_select_task_variant(TaskOp *task, 
                                      Mapper::SelectVariantInput *input,
                                      Mapper::SelectVariantOutput *output,
                                      MappingCallInfo *info = NULL);
      void invoke_post_map_task(TaskOp *task, Mapper::PostMapInput *input,
                                Mapper::PostMapOutput *output,
                                MappingCallInfo *info = NULL);
      void invoke_select_task_sources(TaskOp *task, 
                                      Mapper::SelectTaskSrcInput *input,
                                      Mapper::SelectTaskSrcOutput *output,
                                      MappingCallInfo *info = NULL);
      void invoke_task_create_temporary(TaskOp *task,
                                      Mapper::CreateTaskTemporaryInput *input,
                                      Mapper::CreateTaskTemporaryOutput *output,
                                      MappingCallInfo *info = NULL);
      void invoke_task_speculate(TaskOp *task, 
                                 Mapper::SpeculativeOutput *output,
                                 MappingCallInfo *info = NULL);
      void invoke_task_report_profiling(TaskOp *task, 
                                        Mapper::TaskProfilingInfo *input,
                                        MappingCallInfo *info = NULL);
    public: // Inline mapper calls
      void invoke_map_inline(MapOp *op, Mapper::MapInlineInput *input,
                             Mapper::MapInlineOutput *output, 
                             MappingCallInfo *info = NULL);
      void invoke_select_inline_sources(MapOp *op, 
                                        Mapper::SelectInlineSrcInput *input,
                                        Mapper::SelectInlineSrcOutput *output,
                                        MappingCallInfo *info = NULL);
      void invoke_inline_create_temporary(MapOp *op,
                                    Mapper::CreateInlineTemporaryInput *input,
                                    Mapper::CreateInlineTemporaryOutput *output,
                                    MappingCallInfo *info = NULL);
      void invoke_inline_report_profiling(MapOp *op, 
                                          Mapper::InlineProfilingInfo *input,
                                          MappingCallInfo *info = NULL);
    public: // Copy mapper calls
      void invoke_map_copy(CopyOp *op,
                           Mapper::MapCopyInput *input,
                           Mapper::MapCopyOutput *output,
                           MappingCallInfo *info = NULL);
      void invoke_select_copy_sources(CopyOp *op,
                                      Mapper::SelectCopySrcInput *input,
                                      Mapper::SelectCopySrcOutput *output,
                                      MappingCallInfo *info = NULL);
      void invoke_copy_create_temporary(CopyOp *op,
                                  Mapper::CreateCopyTemporaryInput *input,
                                  Mapper::CreateCopyTemporaryOutput *output,
                                  MappingCallInfo *info = NULL);
      void invoke_copy_speculate(CopyOp *op, Mapper::SpeculativeOutput *output,
                                 MappingCallInfo *info = NULL);
      void invoke_copy_report_profiling(CopyOp *op,
                                        Mapper::CopyProfilingInfo *input,
                                        MappingCallInfo *info = NULL);
    public: // Close mapper calls
      void invoke_map_close(CloseOp *op,
                            Mapper::MapCloseInput *input,
                            Mapper::MapCloseOutput *output,
                            MappingCallInfo *info = NULL);
      void invoke_select_close_sources(CloseOp *op,
                                       Mapper::SelectCloseSrcInput *input,
                                       Mapper::SelectCloseSrcOutput *output,
                                       MappingCallInfo *info = NULL);
      void invoke_close_create_temporary(CloseOp *op,
                                     Mapper::CreateCloseTemporaryInput *input,
                                     Mapper::CreateCloseTemporaryOutput *output,
                                     MappingCallInfo *info = NULL);
      void invoke_close_report_profiling(CloseOp *op,
                                         Mapper::CloseProfilingInfo *input,
                                         MappingCallInfo *info = NULL);
    public: // Acquire mapper calls
      void invoke_map_acquire(AcquireOp *op,
                              Mapper::MapAcquireInput *input,
                              Mapper::MapAcquireOutput *output,
                              MappingCallInfo *info = NULL);
      void invoke_acquire_speculate(AcquireOp *op,
                                    Mapper::SpeculativeOutput *output,
                                    MappingCallInfo *info = NULL);
      void invoke_acquire_report_profiling(AcquireOp *op,
                                           Mapper::AcquireProfilingInfo *input,
                                           MappingCallInfo *info = NULL);
    public: // Release mapper calls
      void invoke_map_release(ReleaseOp *op,
                              Mapper::MapReleaseInput *input,
                              Mapper::MapReleaseOutput *output,
                              MappingCallInfo *info = NULL);
      void invoke_select_release_sources(ReleaseOp *op,
                                         Mapper::SelectReleaseSrcInput *input,
                                         Mapper::SelectReleaseSrcOutput *output,
                                         MappingCallInfo *info = NULL);
      void invoke_release_create_temporary(ReleaseOp *op,
                                  Mapper::CreateReleaseTemporaryInput *input,
                                  Mapper::CreateReleaseTemporaryOutput *output,
                                  MappingCallInfo *info = NULL);
      void invoke_release_speculate(ReleaseOp *op,
                                    Mapper::SpeculativeOutput *output,
                                    MappingCallInfo *info = NULL);
      void invoke_release_report_profiling(ReleaseOp *op,
                                           Mapper::ReleaseProfilingInfo *input,
                                           MappingCallInfo *info = NULL);
    public: // Partition mapper calls
      void invoke_select_partition_projection(DependentPartitionOp *op,
                          Mapper::SelectPartitionProjectionInput *input,
                          Mapper::SelectPartitionProjectionOutput *output,
                          MappingCallInfo *info = NULL);
      void invoke_map_partition(DependentPartitionOp *op,
                          Mapper::MapPartitionInput *input,
                          Mapper::MapPartitionOutput *output,
                          MappingCallInfo *info = NULL);
      void invoke_select_partition_sources(DependentPartitionOp *op,
                          Mapper::SelectPartitionSrcInput *input,
                          Mapper::SelectPartitionSrcOutput *output,
                          MappingCallInfo *info = NULL);
      void invoke_partition_create_temporary(DependentPartitionOp *op,
                          Mapper::CreatePartitionTemporaryInput *input,
                          Mapper::CreatePartitionTemporaryOutput *output,
                          MappingCallInfo *info = NULL);
      void invoke_partition_report_profiling(DependentPartitionOp *op,
                          Mapper::PartitionProfilingInfo *input,
                          MappingCallInfo *info = NULL);
    public: // Task execution mapper calls
      void invoke_configure_context(TaskOp *task,
                                    Mapper::ContextConfigOutput *output,
                                    MappingCallInfo *info = NULL);
      void invoke_select_tunable_value(TaskOp *task,
                                       Mapper::SelectTunableInput *input,
                                       Mapper::SelectTunableOutput *output,
                                       MappingCallInfo *info = NULL);
    public: // must epoch and graph mapper calls
      void invoke_map_must_epoch(MustEpochOp *op,
                                 Mapper::MapMustEpochInput *input,
                                 Mapper::MapMustEpochOutput *output,
                                 MappingCallInfo *info = NULL);
      void invoke_map_dataflow_graph(Mapper::MapDataflowGraphInput *input,
                                     Mapper::MapDataflowGraphOutput *output,
                                     MappingCallInfo *info = NULL);
    public: // memoization mapper calls
      void invoke_memoize_operation(Mappable *mappable,
                                    Mapper::MemoizeInput *input,
                                    Mapper::MemoizeOutput *output,
                                    MappingCallInfo *info = NULL);
    public: // scheduling and stealing mapper calls
      void invoke_select_tasks_to_map(Mapper::SelectMappingInput *input,
                                      Mapper::SelectMappingOutput *output,
                                      MappingCallInfo *info = NULL);
      void invoke_select_steal_targets(Mapper::SelectStealingInput *input,
                                       Mapper::SelectStealingOutput *output,
                                       MappingCallInfo *info = NULL);
      void invoke_permit_steal_request(Mapper::StealRequestInput *input,
                                       Mapper::StealRequestOutput *output,
                                       MappingCallInfo *info = NULL);
    public: // handling mapper calls
      void invoke_handle_message(Mapper::MapperMessage *message,
                                 void *check_defer = NULL,
                                 MappingCallInfo *info = NULL);
      void invoke_handle_task_result(Mapper::MapperTaskResult *result,
                                     MappingCallInfo *info = NULL);
    public:
      virtual bool is_locked(MappingCallInfo *info) = 0;
      virtual void lock_mapper(MappingCallInfo *info, bool read_only) = 0;
      virtual void unlock_mapper(MappingCallInfo *info) = 0;
    public:
      virtual bool is_reentrant(MappingCallInfo *info) = 0;
      virtual void enable_reentrant(MappingCallInfo *info) = 0;
      virtual void disable_reentrant(MappingCallInfo *info) = 0;
    protected:
      virtual MappingCallInfo* begin_mapper_call(MappingCallKind kind,
                                 Operation *op, RtEvent &precondition) = 0;
      virtual void pause_mapper_call(MappingCallInfo *info) = 0;
      virtual void resume_mapper_call(MappingCallInfo *info) = 0;
      virtual void finish_mapper_call(MappingCallInfo *info) = 0;
    public:
      void update_mappable_tag(MappingCallInfo *info,
                               const Mappable &mappable, MappingTagID tag);
      void update_mappable_data(MappingCallInfo *info, const Mappable &mappable,
                                const void *mapper_data, size_t data_size);
    public:
      void send_message(MappingCallInfo *info, Processor target, 
                        const void *message, size_t message_size, 
                        unsigned message_kind);
      void broadcast(MappingCallInfo *info, const void *message, 
                     size_t message_size, unsigned message_kind, int radix);
    public:
      void pack_physical_instance(MappingCallInfo *info, Serializer &rez,
                                  MappingInstance instance);
      void unpack_physical_instance(MappingCallInfo *info, Deserializer &derez,
                                    MappingInstance &instance);
    public:
      MapperEvent create_mapper_event(MappingCallInfo *ctx);
      bool has_mapper_event_triggered(MappingCallInfo *ctx, MapperEvent event);
      void trigger_mapper_event(MappingCallInfo *ctx, MapperEvent event);
      void wait_on_mapper_event(MappingCallInfo *ctx, MapperEvent event);
    public:
      const ExecutionConstraintSet& 
        find_execution_constraints(MappingCallInfo *ctx, 
            TaskID task_id, VariantID vid);
      const TaskLayoutConstraintSet&
        find_task_layout_constraints(MappingCallInfo *ctx, 
            TaskID task_id, VariantID vid);
      const LayoutConstraintSet&
        find_layout_constraints(MappingCallInfo *ctx, LayoutConstraintID id);
      LayoutConstraintID register_layout(MappingCallInfo *ctx, 
                                const LayoutConstraintSet &constraints,
                                FieldSpace handle);
      void release_layout(MappingCallInfo *ctx, LayoutConstraintID layout_id);
      bool do_constraints_conflict(MappingCallInfo *ctx, 
                    LayoutConstraintID set1, LayoutConstraintID set2);
      bool do_constraints_entail(MappingCallInfo *ctx,
                    LayoutConstraintID source, LayoutConstraintID target);
    public:
      void find_valid_variants(MappingCallInfo *ctx, TaskID task_id,
                               std::vector<VariantID> &valid_variants,
                               Processor::Kind kind);
      bool is_leaf_variant(MappingCallInfo *ctx, TaskID task_id, 
                           VariantID variant_id);
      bool is_inner_variant(MappingCallInfo *ctx, TaskID task_id, 
                            VariantID variant_id);
      bool is_idempotent_variant(MappingCallInfo *ctx,
                                 TaskID task_id, VariantID variant_id);
    public:
      void filter_variants(MappingCallInfo *ctx, const Task &task,
            const std::vector<std::vector<MappingInstance> > &chosen_instances,
                           std::vector<VariantID> &variants);
      void filter_instances(MappingCallInfo *ctx, const Task &task,
                            VariantID chosen_variant,
                  std::vector<std::vector<MappingInstance> > &chosen_instances,
                  std::vector<std::set<FieldID> > &missing_fields);
      void filter_instances(MappingCallInfo *ctx, const Task &task,
                            unsigned index, VariantID chosen_variant,
                            std::vector<MappingInstance> &instances,
                            std::set<FieldID> &missing_fields);
    public:
      bool create_physical_instance(MappingCallInfo *ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints, 
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, 
                                    bool acquire, GCPriority priority);
      bool create_physical_instance(MappingCallInfo *ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result,
                                    bool acquire, GCPriority priority);
      bool find_or_create_physical_instance(
                                    MappingCallInfo *ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints, 
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    bool acquire, GCPriority priority,
                                    bool tight_region_bounds);
      bool find_or_create_physical_instance(
                                    MappingCallInfo *ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    bool acquire, GCPriority priority,
                                    bool tight_region_bounds);
      bool find_physical_instance(  MappingCallInfo *ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_region_bounds);
      bool find_physical_instance(  MappingCallInfo *ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_region_bounds);
      void set_garbage_collection_priority(MappingCallInfo *ctx, 
                                    const MappingInstance &instance, 
                                    GCPriority priority);
      bool acquire_instance(        MappingCallInfo *ctx, 
                                    const MappingInstance &instance);
      bool acquire_instances(       MappingCallInfo *ctx,
                                    const std::vector<MappingInstance> &insts);
      bool acquire_and_filter_instances(MappingCallInfo *ctx,
                                    std::vector<MappingInstance> &instances);
      bool acquire_instances(       MappingCallInfo *ctx, const std::vector<
                                    std::vector<MappingInstance> > &instances);
      bool acquire_and_filter_instances(MappingCallInfo *ctx, std::vector<
                                    std::vector<MappingInstance> > &instances);
      void release_instance(        MappingCallInfo *ctx, 
                                    const MappingInstance &instance);
      void release_instances(       MappingCallInfo *ctx,
                                    const std::vector<MappingInstance> &insts);
      void release_instances(       MappingCallInfo *ctx, const std::vector<
                                    std::vector<MappingInstance> > &instances);
    public:
      void record_acquired_instance(MappingCallInfo *info, 
                                    PhysicalManager *manager, bool created);
      void release_acquired_instance(MappingCallInfo *info,
                                     PhysicalManager *manager);
      void check_region_consistency(MappingCallInfo *info, const char *call,
                                    const std::vector<LogicalRegion> &regions);
      bool perform_local_acquires(MappingCallInfo *info,
                                  const std::vector<MappingInstance> &instances,
                       std::map<MemoryManager*,AcquireStatus> &acquire_requests,
                                  std::vector<unsigned> *to_erase); 
      bool perform_remote_acquires(MappingCallInfo *info,
                      std::map<MemoryManager*,AcquireStatus> &acquire_requests);
    public:
      IndexSpace create_index_space(MappingCallInfo *info, const Domain &domain,
                                    const void *realm_is, TypeTag type_tag);
      bool has_index_partition(MappingCallInfo *info,
                               IndexSpace parent, Color color);
      IndexPartition get_index_partition(MappingCallInfo *info,
                                         IndexSpace parent, Color color);
      IndexSpace get_index_subspace(MappingCallInfo *info,
                                    IndexPartition p, Color c);
      IndexSpace get_index_subspace(MappingCallInfo *info,
                                    IndexPartition p, 
                                    const DomainPoint &color);
      bool has_multiple_domains(MappingCallInfo *info, IndexSpace handle);
      Domain get_index_space_domain(MappingCallInfo *info, IndexSpace handle);
      void get_index_space_domains(MappingCallInfo *info, IndexSpace handle,
                                   std::vector<Domain> &domains);
      Domain get_index_partition_color_space(MappingCallInfo *info,
                                             IndexPartition p);
      void get_index_space_partition_colors(MappingCallInfo *info, 
                                            IndexSpace sp, 
                                            std::set<Color> &colors);
      bool is_index_partition_disjoint(MappingCallInfo *info,
                                       IndexPartition p);
      bool is_index_partition_complete(MappingCallInfo *info,
                                       IndexPartition p);
      Color get_index_space_color(MappingCallInfo *info, IndexSpace handle);
      DomainPoint get_index_space_color_point(MappingCallInfo *info,
                                              IndexSpace handle);
      Color get_index_partition_color(MappingCallInfo *info, 
                                      IndexPartition handle);
      IndexSpace get_parent_index_space(MappingCallInfo *info,
                                        IndexPartition handle);
      bool has_parent_index_partition(MappingCallInfo *info,
                                      IndexSpace handle);
      IndexPartition get_parent_index_partition(MappingCallInfo *info,
                                                IndexSpace handle);
      unsigned get_index_space_depth(MappingCallInfo *info, IndexSpace handle);
      unsigned get_index_partition_depth(MappingCallInfo *info, 
                                         IndexPartition handle);
    public:
      size_t get_field_size(MappingCallInfo *info, 
                            FieldSpace handle, FieldID fid);
      void get_field_space_fields(MappingCallInfo *info, FieldSpace handle, 
                                  std::vector<FieldID> &fields);
    public:
      LogicalPartition get_logical_partition(MappingCallInfo *info,
                                             LogicalRegion parent, 
                                             IndexPartition handle);
      LogicalPartition get_logical_partition_by_color(MappingCallInfo *info,
                                                      LogicalRegion parent, 
                                                      Color color);
      LogicalPartition get_logical_partition_by_color(MappingCallInfo *info,
                                                      LogicalRegion parent,
                                                      const DomainPoint &color);
      LogicalPartition get_logical_partition_by_tree(MappingCallInfo *info,
                                                     IndexPartition handle, 
                                           FieldSpace fspace, RegionTreeID tid);
      LogicalRegion get_logical_subregion(MappingCallInfo *info,
                                          LogicalPartition parent, 
                                          IndexSpace handle);
      LogicalRegion get_logical_subregion_by_color(MappingCallInfo *info,
                                                   LogicalPartition parent, 
                                                   Color color);
      LogicalRegion get_logical_subregion_by_color(MappingCallInfo *info,
                                                   LogicalPartition parent,
                                                   const DomainPoint &color);
      LogicalRegion get_logical_subregion_by_tree(MappingCallInfo *info,
                                                  IndexSpace handle, 
                                          FieldSpace fspace, RegionTreeID tid);
      Color get_logical_region_color(MappingCallInfo *info, 
                                     LogicalRegion handle);
      DomainPoint get_logical_region_color_point(MappingCallInfo *info, 
                                                 LogicalRegion handle);
      Color get_logical_partition_color(MappingCallInfo *info,
                                        LogicalPartition handle);
      LogicalRegion get_parent_logical_region(MappingCallInfo *info,
                                              LogicalPartition handle);
      bool has_parent_logical_partition(MappingCallInfo *info, 
                                        LogicalRegion handle);
      LogicalPartition get_parent_logical_partition(MappingCallInfo *info,
                                                    LogicalRegion handle);
    public:
      bool retrieve_semantic_information(MappingCallInfo *ctx, TaskID task_id,
          SemanticTag tag, const void *&result, size_t &size, 
          bool can_fail, bool wait_until_ready);
      bool retrieve_semantic_information(MappingCallInfo *ctx,IndexSpace handle,
          SemanticTag tag, const void *&result, size_t &size,
          bool can_fail, bool wait_until_ready);
      bool retrieve_semantic_information(MappingCallInfo *ctx, 
          IndexPartition handle, SemanticTag tag, const void *&result,
          size_t &size, bool can_fail, bool wait_until_ready);
      bool retrieve_semantic_information(MappingCallInfo *ctx,FieldSpace handle,
          SemanticTag tag, const void *&result, size_t &size, 
          bool can_fail, bool wait_until_ready);
      bool retrieve_semantic_information(MappingCallInfo *ctx,FieldSpace handle,
          FieldID fid, SemanticTag tag, const void *&result, size_t &size,
          bool can_fail, bool wait_until_ready);
      bool retrieve_semantic_information(MappingCallInfo *ctx, 
          LogicalRegion handle, SemanticTag tag, const void *&result, 
          size_t &size, bool can_fail, bool wait_until_ready);
      bool retrieve_semantic_information(MappingCallInfo *ctx,
          LogicalPartition handle, SemanticTag tag, const void *&result,
          size_t &size, bool can_fail, bool wait_until_ready);
    public:
      void retrieve_name(MappingCallInfo *ctx, TaskID task_id, 
                         const char *&result);
      void retrieve_name(MappingCallInfo *ctx, IndexSpace handle,
                         const char *&result);
      void retrieve_name(MappingCallInfo *ctx, IndexPartition handle,
                         const char *&result);
      void retrieve_name(MappingCallInfo *ctx, FieldSpace handle,
                         const char *&result);
      void retrieve_name(MappingCallInfo *ctx, FieldSpace handle, 
                         FieldID fid, const char *&result);
      void retrieve_name(MappingCallInfo *ctx, LogicalRegion handle,
                         const char *&result);
      void retrieve_name(MappingCallInfo *ctx, LogicalPartition handle,
                         const char *&result);
    protected:
      // Both these must be called while holding the lock
      MappingCallInfo* allocate_call_info(MappingCallKind kind, 
                                          Operation *op, bool need_lock);
      void free_call_info(MappingCallInfo *info, bool need_lock);
    public:
      static const char* get_mapper_call_name(MappingCallKind kind);
    public:
      void defer_message(Mapper::MapperMessage *message);
      static void handle_deferred_message(const void *args);
    public:
      // For stealing
      void process_advertisement(Processor advertiser); 
      void perform_stealing(std::multimap<Processor,MapperID> &targets);
    public:
      // For advertising
      void process_failed_steal(Processor thief);
      void perform_advertisements(std::set<Processor> &failed_waiters);
    public:
      Runtime *const runtime;
      Mapping::Mapper *const mapper;
      const MapperID mapper_id;
      const Processor processor;
      const bool profile_mapper;
    protected:
      mutable LocalLock mapper_lock;
    protected:
      std::vector<MappingCallInfo*> available_infos;
    protected: // Steal request information
      // Mappers on other processors that we've tried to steal from and failed
      std::set<Processor> steal_blacklist;
      // Mappers that have tried to steal from us and which we
      // should advertise work when we have it
      std::set<Processor> failed_thiefs;
    };

    /**
     * \class SerializingManager
     * In this class at most one mapper call can be running at 
     * a time. Mapper calls that invoke expensive runtime operations
     * can be pre-empted and it is up to the mapper to control
     * whether additional mapper calls when the call is blocked.
     */
    class SerializingManager : public MapperManager {
    public:
      SerializingManager(Runtime *runtime, Mapping::Mapper *mapper,
                         MapperID map_id, Processor p, bool reentrant);
      SerializingManager(const SerializingManager &rhs);
      virtual ~SerializingManager(void);
    public:
      SerializingManager& operator=(const SerializingManager &rhs);
    public:
      virtual bool is_locked(MappingCallInfo *info);
      virtual void lock_mapper(MappingCallInfo *info, bool read_only);
      virtual void unlock_mapper(MappingCallInfo *info);
    public:
      virtual bool is_reentrant(MappingCallInfo *info);
      virtual void enable_reentrant(MappingCallInfo *info);
      virtual void disable_reentrant(MappingCallInfo *info);
    protected:
      virtual MappingCallInfo* begin_mapper_call(MappingCallKind kind,
                                 Operation *op, RtEvent &precondition);
      virtual void pause_mapper_call(MappingCallInfo *info);
      virtual void resume_mapper_call(MappingCallInfo *info);
      virtual void finish_mapper_call(MappingCallInfo *info);
    protected:
      // Must be called while holding the mapper reservation
      RtUserEvent complete_pending_pause_mapper_call(void);
      RtUserEvent complete_pending_finish_mapper_call(void);
    protected:
      // The one executing call if any otherwise NULL
      MappingCallInfo *executing_call; 
      // Calls yet to start running
      std::deque<MappingCallInfo*> pending_calls; 
      // Number of calls paused due to runtime work
      unsigned paused_calls;
      // Calls that are ready to resume after runtime work
      std::deque<MappingCallInfo*> ready_calls;
      // Calls that are waiting for diabling of reentrancy
      std::deque<MappingCallInfo*> non_reentrant_calls;
      // Whether or not we are currently supporting reentrant calls
      bool permit_reentrant;
      // A flag checking whether we have a pending paused mapper call
      bool pending_pause_call;
      // A flag checking whether we have a pending finished call
      bool pending_finish_call;
    };

    /**
     * \class ConcurrentManager
     * In this class many mapper calls can be running concurrently.
     * It is upper to the mapper to lock itself when necessary to 
     * protect internal state. Mappers can be locked in exclusive
     * or non-exclusive modes.
     */
    class ConcurrentManager : public MapperManager {
    public:
      enum LockState {
        UNLOCKED_STATE,
        READ_ONLY_STATE,
        EXCLUSIVE_STATE,
      };
    public:
      ConcurrentManager(Runtime *runtime, Mapping::Mapper *mapper,
                        MapperID map_id, Processor p);
      ConcurrentManager(const ConcurrentManager &rhs);
      virtual ~ConcurrentManager(void);
    public:
      ConcurrentManager& operator=(const ConcurrentManager &rhs);
    public:
      virtual bool is_locked(MappingCallInfo *info);
      virtual void lock_mapper(MappingCallInfo *info, bool read_only);
      virtual void unlock_mapper(MappingCallInfo *info);
    public:
      virtual bool is_reentrant(MappingCallInfo *info);
      virtual void enable_reentrant(MappingCallInfo *info);
      virtual void disable_reentrant(MappingCallInfo *info);
    protected:
      virtual MappingCallInfo* begin_mapper_call(MappingCallKind kind,
                                 Operation *op, RtEvent &precondition);
      virtual void pause_mapper_call(MappingCallInfo *info);
      virtual void resume_mapper_call(MappingCallInfo *info);
      virtual void finish_mapper_call(MappingCallInfo *info);
    protected:
      // Must be called while holding the lock
      void release_lock(std::vector<RtUserEvent> &to_trigger); 
    protected:
      LockState lock_state;
      std::set<MappingCallInfo*> current_holders;
      std::deque<MappingCallInfo*> read_only_waiters;
      std::deque<MappingCallInfo*> exclusive_waiters;
    };

    /**
     * \class MapperContinuation
     * A class for deferring mapper calls
     */
    class MapperContinuation {
    public:
      struct ContinuationArgs : public LgTaskArgs<ContinuationArgs> {
      public:
        static const LgTaskID TASK_ID = LG_MAPPER_CONTINUATION_TASK_ID;
      public:
        ContinuationArgs(UniqueID op_id, MapperContinuation *c)
          : LgTaskArgs<ContinuationArgs>(op_id), continuation(c) { }
      public:
        MapperContinuation *const continuation;
      };
    public:
      MapperContinuation(MapperManager *manager,
                         MappingCallInfo *info);
    public:
      void defer(Runtime *runtime, RtEvent precondition, Operation *op = NULL);
    public:
      virtual void execute(void) = 0;
    public:
      static void handle_continuation(const void *args);
    public:
      MapperManager *const manager;
      MappingCallInfo *const info;
    };

    template<typename T1,
             void (MapperManager::*CALL)(T1*, MappingCallInfo*)>
    class MapperContinuation1 : public MapperContinuation {
    public:
      MapperContinuation1(MapperManager *man, T1 *a1, 
                          MappingCallInfo *info)
        : MapperContinuation(man, info), arg1(a1) { }
    public:
      virtual void execute(void)
      { (manager->*CALL)(arg1, info); }
    public:
      T1 *const arg1;
    };

    template<typename T1, typename T2, 
             void (MapperManager::*CALL)(T1*, T2*, MappingCallInfo*)>
    class MapperContinuation2 : public MapperContinuation {
    public:
      MapperContinuation2(MapperManager *man, T1 *a1, T2 *a2,
                          MappingCallInfo *info)
        : MapperContinuation(man, info), arg1(a1), arg2(a2) { }
    public:
      virtual void execute(void)
      { (manager->*CALL)(arg1, arg2, info); }
    public:
      T1 *const arg1;
      T2 *const arg2;
    };

    template<typename T1, typename T2, typename T3,
             void (MapperManager::*CALL)(T1*, T2*, T3*, MappingCallInfo*)>
    class MapperContinuation3 : public MapperContinuation {
    public:
      MapperContinuation3(MapperManager *man, T1 *a1, T2 *a2, T3 *a3,
                          MappingCallInfo *info)
        : MapperContinuation(man, info), arg1(a1), arg2(a2), arg3(a3) { }
    public:
      virtual void execute(void)
      { (manager->*CALL)(arg1, arg2, arg3, info); }
    public:
      T1 *const arg1;
      T2 *const arg2;
      T3 *const arg3;
    };

  };
};

#endif // __MAPPER_MANAGER_H__
