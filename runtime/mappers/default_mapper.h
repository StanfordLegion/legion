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


#ifndef __DEFAULT_MAPPER_H__
#define __DEFAULT_MAPPER_H__

#include "legion.h"
#include "legion_mapping.h"
#include "mapping_utilities.h"

#include <cstdlib>
#include <cassert>
#include <algorithm>

namespace Legion {
  namespace Mapping {

    /**
     * \class DefaultMapper
     * The default mapper class is our base implementation of the
     * mapper interface that relies on some simple heuristics 
     * to perform most of them calls for general purpose Legion
     * applications.  You should feel free to extend this class
     * with your own heuristics by overriding some or all of the
     * methods.  You can also ignore this implementation entirely
     * and perform your own implementation of the mapper interface.
     */
    class DefaultMapper : public Mapper {
    public:
      enum DefaultTunables { // tunable IDs recognized by the default mapper
        DEFAULT_TUNABLE_NODE_COUNT,
        DEFAULT_TUNABLE_LOCAL_CPUS,
        DEFAULT_TUNABLE_LOCAL_GPUS,
        DEFAULT_TUNABLE_LOCAL_IOS,
        DEFAULT_TUNABLE_GLOBAL_CPUS,
        DEFAULT_TUNABLE_GLOBAL_GPUS,
        DEFAULT_TUNABLE_GLOBAL_IOS,
      };
      enum MappingKind {
        TASK_MAPPING,
        INLINE_MAPPING,
        COPY_MAPPING,
        CLOSE_MAPPING,
        ACQUIRE_MAPPING,
        RELEASE_MAPPING,
      };
      enum MapperMessageType
      {
        INVALID_MESSAGE = 0,
        PROFILING_SAMPLE = 1,
        ADVERTISEMENT = 2,
      };
    protected: // Internal types
      struct VariantInfo {
      public:
        VariantInfo(void)
          : variant(0), tight_bound(false), is_inner(false) { }
      public:
        VariantID            variant;
        Processor::Kind      proc_kind;
        bool                 tight_bound;
        bool                 is_inner;
      };
      struct CachedTaskMapping {
      public:
        unsigned long long                          task_hash;
        VariantID                                   variant;
        std::vector<std::vector<PhysicalInstance> > mapping;
        bool                                        has_reductions;
      };
      struct MapperMsgHdr {
      public:
        MapperMsgHdr(void) : magic(0xABCD), type(INVALID_MESSAGE) { }
        bool is_valid_mapper_msg() const
        {
          return magic == 0xABCD && type != INVALID_MESSAGE;
        }
        uint32_t magic;
        MapperMessageType type;
      };
      struct ProfilingSampleMsg : public MapperMsgHdr {
      public:
        ProfilingSampleMsg(void) : MapperMsgHdr(), task_id(0) { }
        Processor::TaskFuncID task_id;
        Utilities::MappingProfiler::Profile sample;
      };
    public:
      DefaultMapper(MapperRuntime *rt, Machine machine, Processor local, 
                    const char *maper_name = NULL);
      DefaultMapper(const DefaultMapper &rhs);
      virtual ~DefaultMapper(void);
    public:
      DefaultMapper& operator=(const DefaultMapper &rhs);
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
      virtual void speculate(const MapperContext         ctx,
                             const Release&              release,
                                   SpeculativeOutput&    output);
      virtual void report_profiling(const MapperContext         ctx,
                                    const Release&              release,
                                    const ReleaseProfilingInfo& input);
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
    public: // Mapping control and stealing
      virtual void select_tasks_to_map(const MapperContext          ctx,
                                       const SelectMappingInput&    input,
                                             SelectMappingOutput&   output);
      virtual void select_steal_targets(const MapperContext         ctx,
                                        const SelectStealingInput&  input,
                                              SelectStealingOutput& output);
      virtual void permit_steal_request(const MapperContext         ctx,
                                        const StealRequestInput&    input,
                                              StealRequestOutput&   output);
    public: // handling
      virtual void handle_message(const MapperContext           ctx,
                                  const MapperMessage&          message);
      virtual void handle_task_result(const MapperContext           ctx,
                                      const MapperTaskResult&       result);
    public: // These virtual methods are not part of the mapper interface
            // but make it possible for inheriting mappers to easily
            // override policies that the default mapper is employing
            // All method calls start with 'default_policy_'
      virtual Processor default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task);
      virtual void default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs);
      virtual void default_policy_rank_processor_kinds(
                                    MapperContext ctx, const Task &task, 
                                    std::vector<Processor::Kind> &ranking);
      virtual VariantID default_policy_select_best_variant(MapperContext ctx,
                                    const Task &task, Processor::Kind kind, 
                                    VariantID vid1, VariantID vid2,
                                    const ExecutionConstraintSet &execution1,
                                    const ExecutionConstraintSet &execution2,
                                    const TaskLayoutConstraintSet &layout1,
                                    const TaskLayoutConstraintSet &layout2);
      virtual Memory default_policy_select_target_memory(MapperContext ctx, 
                                    Processor target_proc);
      virtual LayoutConstraintID default_policy_select_layout_constraints(
                                    MapperContext ctx, Memory target_memory,
                                    const RegionRequirement &req,
                                    MappingKind mapping_kind,
                                    bool needs_field_constraint_check,
                                    bool &force_new_instances);
      virtual void default_policy_select_constraints(MapperContext ctx,
                                    LayoutConstraintSet &constraints,
                                    Memory target_memory,
                                    const RegionRequirement &req);
      virtual void default_policy_select_constraint_fields(
                                    MapperContext ctx,
                                    const RegionRequirement &req,
                                    std::vector<FieldID> &fields);
      virtual LogicalRegion default_policy_select_instance_region(
                                    MapperContext ctx, Memory target_memory,
                                    const RegionRequirement &req,
                                    const LayoutConstraintSet &constraints,
                                    bool force_new_instances, 
                                    bool meets_constraints);
      virtual int default_policy_select_garbage_collection_priority(
                                    MapperContext ctx, 
                                    MappingKind kind, Memory memory, 
                                    const PhysicalInstance &instance,
                                    bool meets_fill_constraints,bool reduction);
      virtual void default_policy_select_sources(MapperContext,
                                    const PhysicalInstance &target,
                                    const std::vector<PhysicalInstance> &source,
                                    std::deque<PhysicalInstance> &ranking);
      virtual bool default_policy_select_close_virtual(const MapperContext,
                                                       const Close &);
    protected: // help for generating random numbers
      long default_generate_random_integer(void) const;
      double default_generate_random_real(void) const;
    protected: // member helper methods
      Processor select_random_processor(
                              const std::vector<Processor> &procs) const;
      VariantInfo find_preferred_variant(const Task &task, MapperContext ctx,
                                 bool needs_tight_bound, bool cache = true,
                                 Processor::Kind kind = Processor::NO_KIND);
      void default_slice_task(const Task &task,
                              const std::vector<Processor> &local_procs,
                              const std::vector<Processor> &remote_procs,
                              const SliceTaskInput &input,
                                    SliceTaskOutput &output,
            std::map<Domain,std::vector<TaskSlice> > &cached_slices) const;
      bool default_create_custom_instances(MapperContext ctx, 
                              Processor target, Memory target_memory,
                              const RegionRequirement &req, unsigned index,
                              std::set<FieldID> &needed_fields, // will destroy
                              const TaskLayoutConstraintSet &layout_constraints,
                              bool needs_field_constraint_check,
                              std::vector<PhysicalInstance> &instances);
      bool default_make_instance(MapperContext ctx, Memory target_memory,
                              const LayoutConstraintSet &constraints, 
                              PhysicalInstance &result, MappingKind kind,
                              bool force_new, bool meets,
                              const RegionRequirement &req);
      void default_report_failed_instance_creation(const Task &task, 
                              unsigned index, Processor target_proc, 
                              Memory target_memory) const;
      void default_remove_cached_task(MapperContext ctx, VariantID variant,
                              unsigned long long task_hash,
                              const std::pair<TaskID,Processor> &cache_key,
                              const std::vector<
                                std::vector<PhysicalInstance> > &post_filter);
      template<bool IS_SRC>
      void default_create_copy_instance(MapperContext ctx, const Copy &copy,
                              const RegionRequirement &req, unsigned index,
                              std::vector<PhysicalInstance> &instances);
    protected: // static helper methods
      static const char* create_default_name(Processor p);
      template<int DIM>
      static void default_decompose_points(
                              const LegionRuntime::Arrays::Rect<DIM> &point_rect,
                              const std::vector<Processor> &targets,
                              const LegionRuntime::Arrays::Point<DIM> &blocking, 
                              bool recurse, bool stealable,
                              std::vector<TaskSlice> &slices);
      template<int DIM>
      static LegionRuntime::Arrays::Point<DIM> default_select_blocking_factor(
                            long long int factor, const LegionRuntime::Arrays::
                            Rect<DIM> &rect_to_factor);
      static unsigned long long compute_task_hash(const Task &task);
      static inline bool physical_sort_func(
                         const std::pair<PhysicalInstance,unsigned> &left,
                         const std::pair<PhysicalInstance,unsigned> &right)
    { return (left.second < right.second); }
    protected:
      const Processor       local_proc;
      const Processor::Kind local_kind;
      const AddressSpace    node_id;
      const Machine         machine;
      const char *const     mapper_name;
    protected:
      mutable unsigned short random_number_generator[3];
    protected: 
      // Make these data structures mutable anticipating when the machine
      // can change shape dynamically
      unsigned               total_nodes;
      // There are a couple of parameters from the machine description that 
      // the default mapper uses to determine how to perform mapping.
      std::vector<Processor> local_ios;
      std::vector<Processor> local_cpus;
      std::vector<Processor> local_gpus;
      std::vector<Processor> remote_ios;
      std::vector<Processor> remote_cpus;
      std::vector<Processor> remote_gpus;
    protected: 
      // Cached mapping information about the application
      std::map<Domain,std::vector<TaskSlice> > cpu_slices_cache,
                                               gpu_slices_cache,io_slices_cache;
      std::map<TaskID,VariantInfo>             preferred_variants; 
      std::map<std::pair<TaskID,Processor>,
               std::list<CachedTaskMapping> >  cached_task_mappings;
      std::map<std::pair<Memory::Kind,FieldSpace>,
               LayoutConstraintID>             layout_constraint_cache;
      std::map<std::pair<Memory::Kind,ReductionOpID>,
               LayoutConstraintID>             reduction_constraint_cache;
    protected:
      // The maximum number of tasks a mapper will allow to be stolen at a time
      // Controlled by -dm:thefts
      unsigned max_steals_per_theft;
      // The maximum number of times that a single task is allowed to be stolen
      // Controlled by -dm:count
      unsigned max_steal_count;
      // Do a breadth-first traversal of the task tree, by default we do
      // a depth-first traversal to improve locality
      bool breadth_first_traversal;
      // Track whether stealing is enabled
      bool stealing_enabled;
      // The maximum number of tasks scheduled per step
      unsigned max_schedule_count;
    };

  }; // namespace Mapping
}; // namespace Legion

// For backwards compatibility
namespace LegionRuntime {
  namespace HighLevel {
    typedef Legion::Mapping::DefaultMapper DefaultMapper;
  };
};

#endif // __DEFAULT_MAPPER_H__

// EOF

