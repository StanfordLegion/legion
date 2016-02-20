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
    protected: // Internal types
      struct VariantInfo {
      public:
        VariantInfo(void)
          : variant(0), tight_bound(false) { }
      public:
        VariantID            variant;
        Processor::Kind      proc_kind;
        bool                 tight_bound;
      };
      struct CachedTaskMapping {
      public:
        unsigned long long                          task_hash;
        VariantID                                   variant;
        std::vector<std::vector<PhysicalInstance> > mapping;
      };
    public:
      DefaultMapper(Machine machine, Processor local, 
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
                                        const StealRequestInput&    intput,
                                              StealRequestOutput&   output);
    public: // handling
      virtual void handle_message(const MapperContext           ctx,
                                  const MapperMessage&          message);
      virtual void handle_task_result(const MapperContext           ctx,
                                      const MapperTaskResult&       result);
    public: // These virtual methods are not part of the mapper interface
            // but make it possible for inheriting mappers to easily
            // override heuristics that the default mapper is employing
      virtual void rank_processor_kinds(const Task &task,
                                        std::vector<Processor::Kind> &ranking);
      virtual VariantID select_best_variant(
                                    const Task &task, Processor::Kind kind, 
                                    VariantID vid1, VariantID vid2,
                                    const ExecutionConstraintSet &execution1,
                                    const ExecutionConstraintSet &execution2,
                                    const TaskLayoutConstraintSet &layout1,
                                    const TaskLayoutConstraintSet &layout2);
    protected: // help for generating random numbers
      long generate_random_integer(void) const;
      double generate_random_real(void) const;
    protected: // member helper methods
      Processor default_select_initial_processor(const Task &task,
                                                 MapperContext ctx); 
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
      void default_create_custom_instance(MapperContext ctx, Processor target,
                              const RegionRequirement &req,
                              std::vector<PhysicalInstance> &destination);
      void default_create_reduction_instance(MapperContext ctx,
                              Processor target, const RegionRequirement &req,
                              std::vector<PhysicalInstance> &destination);
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
            int factor, const LegionRuntime::Arrays::Rect<DIM> &rect_to_factor);
      static unsigned long long compute_task_hash(const Task &task);
    protected:
      const Processor       local_proc;
      const Processor::Kind local_kind;
      const AddressSpace    node_id;
      const Machine         machine;
      const char *const     mapper_name;
    private:
      mutable unsigned short random_number_generator[3];
    protected: 
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
      // Whether or not copies can be made to avoid Write-After-Read dependences
      // Controlled by -dm:war
      bool war_enabled;
      // Track whether stealing is enabled
      bool stealing_enabled;
      // The maximum number of tasks scheduled per step
      unsigned max_schedule_count;
    protected:
      // Utilities for use within the default mapper 
      Utilities::MachineQueryInterface machine_interface;
      Utilities::MappingMemoizer memoizer;
      Utilities::MappingProfiler profiler;
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

