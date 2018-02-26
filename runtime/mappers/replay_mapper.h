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

#ifndef __REPLAY_MAPPER_H__
#define __REPLAY_MAPPER_H__

#include "legion.h"

#include <stdlib.h>
#include <assert.h>
#include <algorithm>

namespace Legion {
  namespace Mapping {

    /**
     * \class ReplayMapper
     * The replay mapper can take as input a replay file generated
     * LegionSpy and use it to replay all the mapping decisions made
     * in a previous run. This is useful for debugging purposes where
     * we want to replay the same execution of a program over and
     * over to discover the source of a runtime bug.
     */
    class ReplayMapper : public Mapper { 
    public:
      struct InstanceInfo {
      public:
        InstanceInfo(void) : original_id(0), 
            creating_instance(false), instance_valid(false) { }
      public:
        PhysicalInstance get_instance(MapperRuntime *runtime,
                                      MapperContext ctx, LogicalRegion handle);
        void create_instance(MapperRuntime *runtime, MapperContext ctx,
                             LogicalRegion handle);
        void record_created_instance(MapperRuntime *runtime, MapperContext ctx,
                                     PhysicalInstance result);
        void decrement_use_count(MapperRuntime *runtime, 
                                 MapperContext ctx, bool first);
      public:
        unsigned long original_id;
        unsigned num_uses;
        Processor creator;
        bool is_owner;
      public:
        Memory target_memory;
        LayoutConstraintSet layout_constraints;
        std::vector<std::vector<DomainPoint> > region_paths;
      public:
        PhysicalInstance instance;
        bool creating_instance;
        bool instance_valid;
        MapperEvent request_event;
      };
      struct RequirementMapping {
      public:
        void map_requirement(MapperRuntime *runtime, MapperContext ctx,
           LogicalRegion handle, std::vector<PhysicalInstance> &targets);
      public:
        std::vector<InstanceInfo*> instances;
      };
      struct TemporaryMapping {
      public:
        void map_temporary(MapperRuntime *runtime, MapperContext ctx,
                           LogicalRegion handle, unsigned long original_dst,
                           PhysicalInstance &result);
      public:
        std::map<unsigned long/*original dst*/,InstanceInfo*> instances;
      };
      struct TunableMapping {
      public:
        void set_tunable(void *&value, size_t &size);
      public:
        unsigned tunable_size;
        void *tunable_value;
      };
      struct TaskMappingInfo {
      public:
        TaskMappingInfo(void) : next_tunable(0) { }
      public:
        UniqueID original_unique_id;
        Processor target_proc;
        VariantID variant;
        TaskPriority priority;
        unsigned next_tunable;
      public:
        std::map<unsigned,RequirementMapping*> premappings;
        std::vector<RequirementMapping*> mappings;
        std::map<unsigned,RequirementMapping*> postmappings;
        std::map<unsigned,TemporaryMapping*> temporaries;
        std::vector<TunableMapping*> tunables;
        std::vector<UniqueID/*original*/> operation_ids;
        std::vector<UniqueID/*original*/> close_ids; 
      };
      struct InlineMappingInfo {
      public:
        RequirementMapping *mapping;
        TemporaryMapping *temporary;
      };
      struct CopyMappingInfo {
      public:
        std::vector<RequirementMapping*> src_mappings;
        std::vector<RequirementMapping*> dst_mappings;
        std::map<unsigned,TemporaryMapping*> src_temporaries;
        std::map<unsigned,TemporaryMapping*> dst_temporaries;
      };
      struct CloseMappingInfo {
      public:
        RequirementMapping *mapping;
        TemporaryMapping* temporary;
      };
      struct ReleaseMappingInfo {
      public:
        TemporaryMapping *temporary;
      };
    public:
      enum ReplayMessageKind {
        ID_MAPPING_MESSAGE,
        INSTANCE_MAPPING_MESSAGE,
        CREATE_INSTANCE_MESSAGE,
        INSTANCE_CREATION_MESSAGE,
        DECREMENT_USE_MESSAGE,
      };
    public:
      ReplayMapper(MapperRuntime *rt, Machine machine, Processor local, 
                   const char *replay_file, const char *mapper_name = NULL);
      ReplayMapper(const ReplayMapper &rhs);
      virtual ~ReplayMapper(void);
    public:
      ReplayMapper& operator=(const ReplayMapper &rhs);
    public:
      static const char* create_replay_name(Processor p);
    public:
      virtual const char* get_mapper_name(void) const;
      virtual MapperSyncModel get_mapper_sync_model(void) const;
    public: // Task mapping calls
      virtual void select_task_options(const MapperContext    ctx,
                                       const Task&            task,
                                             TaskOptions&     output);
      virtual void premap_task(const MapperContext      ctx,
                               const Task&              task, 
                               const PremapTaskInput&   input,
                               PremapTaskOutput&        output);
      virtual void slice_task(const MapperContext      ctx,
                              const Task&              task, 
                              const SliceTaskInput&    input,
                                    SliceTaskOutput&   output);
      virtual void map_task(const MapperContext      ctx,
                            const Task&              task,
                            const MapTaskInput&      input,
                                  MapTaskOutput&     output);
      virtual void select_task_variant(const MapperContext          ctx,
                                       const Task&                  task,
                                       const SelectVariantInput&    input,
                                             SelectVariantOutput&   output);
      virtual void postmap_task(const MapperContext      ctx,
                                const Task&              task,
                                const PostMapInput&      input,
                                      PostMapOutput&     output);
      virtual void select_task_sources(const MapperContext        ctx,
                                       const Task&                task,
                                       const SelectTaskSrcInput&  input,
                                             SelectTaskSrcOutput& output);
      virtual void create_task_temporary_instance(
                                    const MapperContext              ctx,
                                    const Task&                      task,
                                    const CreateTaskTemporaryInput&  input,
                                          CreateTaskTemporaryOutput& output);
      virtual void speculate(const MapperContext      ctx,
                             const Task&              task,
                                   SpeculativeOutput& output);
      virtual void report_profiling(const MapperContext      ctx,
                                    const Task&              task,
                                    const TaskProfilingInfo& input);
    public: // Inline mapping calls
      virtual void map_inline(const MapperContext        ctx,
                              const InlineMapping&       inline_op,
                              const MapInlineInput&      input,
                                    MapInlineOutput&     output);
      virtual void select_inline_sources(const MapperContext        ctx,
                                       const InlineMapping&         inline_op,
                                       const SelectInlineSrcInput&  input,
                                             SelectInlineSrcOutput& output);
      virtual void create_inline_temporary_instance(
                                  const MapperContext                ctx,
                                  const InlineMapping&               inline_op,
                                  const CreateInlineTemporaryInput&  input,
                                        CreateInlineTemporaryOutput& output);
      virtual void report_profiling(const MapperContext         ctx,
                                    const InlineMapping&        inline_op,
                                    const InlineProfilingInfo&  input);
    public: // Copy mapping calls
      virtual void map_copy(const MapperContext      ctx,
                            const Copy&              copy,
                            const MapCopyInput&      input,
                                  MapCopyOutput&     output);
      virtual void select_copy_sources(const MapperContext          ctx,
                                       const Copy&                  copy,
                                       const SelectCopySrcInput&    input,
                                             SelectCopySrcOutput&   output);
      virtual void create_copy_temporary_instance(
                                  const MapperContext              ctx,
                                  const Copy&                      copy,
                                  const CreateCopyTemporaryInput&  input,
                                        CreateCopyTemporaryOutput& output);
      virtual void speculate(const MapperContext      ctx,
                             const Copy& copy,
                                   SpeculativeOutput& output);
      virtual void report_profiling(const MapperContext      ctx,
                                    const Copy&              copy,
                                    const CopyProfilingInfo& input);
    public: // Close mapping calls
      virtual void map_close(const MapperContext       ctx,
                             const Close&              close,
                             const MapCloseInput&      input,
                                   MapCloseOutput&     output);
      virtual void select_close_sources(const MapperContext        ctx,
                                        const Close&               close,
                                        const SelectCloseSrcInput&  input,
                                              SelectCloseSrcOutput& output);
      virtual void create_close_temporary_instance(
                                  const MapperContext               ctx,
                                  const Close&                      close,
                                  const CreateCloseTemporaryInput&  input,
                                        CreateCloseTemporaryOutput& output);
      virtual void report_profiling(const MapperContext       ctx,
                                    const Close&              close,
                                    const CloseProfilingInfo& input);
    public: // Acquire mapping calls
      virtual void map_acquire(const MapperContext         ctx,
                               const Acquire&              acquire,
                               const MapAcquireInput&      input,
                                     MapAcquireOutput&     output);
      virtual void speculate(const MapperContext         ctx,
                             const Acquire&              acquire,
                                   SpeculativeOutput&    output);
      virtual void report_profiling(const MapperContext         ctx,
                                    const Acquire&              acquire,
                                    const AcquireProfilingInfo& input);
    public: // Release mapping calls
      virtual void map_release(const MapperContext         ctx,
                               const Release&              release,
                               const MapReleaseInput&      input,
                                     MapReleaseOutput&     output);
      virtual void select_release_sources(const MapperContext       ctx,
                                     const Release&                 release,
                                     const SelectReleaseSrcInput&   input,
                                           SelectReleaseSrcOutput&  output);
      virtual void create_release_temporary_instance(
                                   const MapperContext                 ctx,
                                   const Release&                      release,
                                   const CreateReleaseTemporaryInput&  input,
                                         CreateReleaseTemporaryOutput& output);
      virtual void speculate(const MapperContext         ctx,
                             const Release&              release,
                                   SpeculativeOutput&    output);
      virtual void report_profiling(const MapperContext         ctx,
                                    const Release&              release,
                                    const ReleaseProfilingInfo& input);
    public: // Partition mapping calls
      virtual void select_partition_projection(const MapperContext  ctx,
                          const Partition&                          partition,
                          const SelectPartitionProjectionInput&     input,
                                SelectPartitionProjectionOutput&    output);
      virtual void map_partition(const MapperContext        ctx,
                                 const Partition&           partition,
                                 const MapPartitionInput&   input,
                                       MapPartitionOutput&  output);
      virtual void select_partition_sources(
                                   const MapperContext             ctx,
                                   const Partition&                partition,
                                   const SelectPartitionSrcInput&  input,
                                         SelectPartitionSrcOutput& output);
      virtual void create_partition_temporary_instance(
                              const MapperContext                   ctx,
                              const Partition&                      partition,
                              const CreatePartitionTemporaryInput&  input,
                                    CreatePartitionTemporaryOutput& output);
      virtual void report_profiling(const MapperContext              ctx,
                                    const Partition&                 partition,
                                    const PartitionProfilingInfo&    input);
    public: // Task execution mapping calls
      virtual void configure_context(const MapperContext         ctx,
                                     const Task&                 task,
                                           ContextConfigOutput&  output);
      virtual void select_tunable_value(const MapperContext         ctx,
                                        const Task&                 task,
                                        const SelectTunableInput&   input,
                                              SelectTunableOutput&  output);
    public: // Must epoch mapping
      virtual void map_must_epoch(const MapperContext           ctx,
                                  const MapMustEpochInput&      input,
                                        MapMustEpochOutput&     output);
    public: // Dataflow graph mapping
      virtual void map_dataflow_graph(const MapperContext           ctx,
                                      const MapDataflowGraphInput&  input,
                                            MapDataflowGraphOutput& output);
    public: // Memoization control
      virtual void memoize_operation(const MapperContext  ctx,
                                     const Mappable&      mappable,
                                     const MemoizeInput&  input,
                                           MemoizeOutput& output);
    public: // Mapping control and stealing
      virtual void select_tasks_to_map(const MapperContext          ctx,
                                       const SelectMappingInput&    input,
                                             SelectMappingOutput&   output);
      virtual void select_steal_targets(const MapperContext         ctx,
                                        const SelectStealingInput&  input,
                                              SelectStealingOutput& output);
      virtual void permit_steal_request(const MapperContext         ctx,
                                        const StealRequestInput&    intput,
                                              StealRequestOutput&   output);
    public: // handling
      virtual void handle_message(const MapperContext           ctx,
                                  const MapperMessage&          message);
      virtual void handle_task_result(const MapperContext           ctx,
                                      const MapperTaskResult&       result);
    protected:
      unsigned long find_original_instance_id(MapperContext ctx, 
                                              unsigned long current_id);
      void update_original_instance_id(MapperContext ctx, 
                      unsigned long current_id, unsigned long original_id);
    protected:
      InstanceInfo* unpack_instance(FILE *f) const;
      TaskMappingInfo* unpack_task_mapping(FILE *f) const;
      InlineMappingInfo* unpack_inline_mapping(FILE *f) const;
      CopyMappingInfo* unpack_copy_mapping(FILE *f) const;
      CloseMappingInfo* unpack_close_mapping(FILE *f) const;
      ReleaseMappingInfo* unpack_release_mapping(FILE *f) const;
      RequirementMapping* unpack_requirement(FILE *f) const;
      TemporaryMapping* unpack_temporary(FILE *f) const;
      TunableMapping* unpack_tunable(FILE *f) const;
    protected:
      TaskMappingInfo* find_task_mapping(MapperContext ctx, const Task &task,
                                  const DomainPoint &p, bool parent = false);
      InlineMappingInfo* find_inline_mapping(MapperContext ctx,
                                             const InlineMapping &inline_op);
      CopyMappingInfo* find_copy_mapping(MapperContext ctx, const Copy &copy);
      CloseMappingInfo* find_close_mapping(MapperContext ctx, 
                                           const Close &close);
      ReleaseMappingInfo* find_release_mapping(MapperContext ctx,
                                               const Release &release);
    protected:
      template<typename T>
      static inline void ignore_result(T arg) { }
    protected:
      const Machine machine;
      const Processor local_proc;
      const char *const mapper_name;
    protected:
      std::map<unsigned long,InstanceInfo*>              instance_infos;
      std::map<std::pair<UniqueID/*original*/,DomainPoint>,TaskMappingInfo*>
                                                         task_mappings;
      std::map<UniqueID/*original*/,InlineMappingInfo*>  inline_mappings;
      std::map<UniqueID/*original*/,CopyMappingInfo*>    copy_mappings;
      std::map<UniqueID/*original*/,CloseMappingInfo*>   close_mappings;
      std::map<UniqueID/*original*/,ReleaseMappingInfo*> release_mappings;
    protected:
      UniqueID                                           top_level_id;
      std::map<UniqueID/*current*/,UniqueID/*original*/> original_mappings;
      std::map<UniqueID/*current*/,MapperEvent>          pending_task_ids;
    protected:
      std::map<unsigned long/*current*/,unsigned long/*original*/> 
                                                         original_instances;
      std::map<unsigned long/*current*/,MapperEvent>     pending_instance_ids;
    };

  }; // namespace Mapping
}; // namespace Legion

#endif // _REPLAY_MAPPER_H___

