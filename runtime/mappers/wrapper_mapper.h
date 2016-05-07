#ifndef __WRAPPER_MAPPER_h__
#define __WRAPPER_MAPPER_h__

#include "legion.h"
#include "legion_mapping.h"



#include <cstdlib>
#include <cassert>
#include <algorithm>

namespace Legion {
  namespace Mapping {

	struct select_task_options_message{
	int tag;
	std::string task_name;
	Mapper::TaskOptions output;
	};

	struct get_input_message{
	int tag;
	Processor processor;
	std::map<Processor, int> procs_map;
	std::map<std::string, int> tasks_map;
	std::map<Memory, int> mems_map;
	};
	

	class WrapperMapper: public Mapper{

		public:
			Mapper* dmapper;
			static std::vector<int> tasks_list;
			static std::map<std::string, int> tasks_map;
			static std::vector<int> functions_list;
			static std::map<int, int> methods_map;
			static std::set<Memory> all_mems;
			static std::set<Processor> all_procs;
			static std::vector<Memory> mems_list;
			static std::vector<Processor> procs_list;
			static std::map<Processor, int> procs_map;
			static std::map<Memory, int> mems_map;
			static std::map<int, std::string> task_names_map;
			static bool inputtaken;
			static bool databroadcasted;
			static Processor ownerprocessor;
			static MapperEvent mapevent;
			static int broadcastcount;
			WrapperMapper(Mapper* dmapper, MapperRuntime *rt, Machine machine, Processor local);
			~WrapperMapper(void);
			void get_input();
			//void get_input(const MapperContext ctx);
			void get_select_task_options_input(const MapperContext ctx, std::string task_name, TaskOptions& output);
			void get_map_task_input(Task *task);
			bool InputNumberCheck(std::string strUserInput);
			//std::string wait_task_name;
			Mapper::TaskOptions wait_task_options;

public:
      const char* get_mapper_name(void) const;
      MapperSyncModel get_mapper_sync_model(void) const;


		public: // Task mapping calls
       void select_task_options(const MapperContext    ctx,
                                       const Task&            task,
                                             TaskOptions&     output);
       void premap_task(const MapperContext      ctx,
                               const Task&              task, 
                               const PremapTaskInput&   input,
                               PremapTaskOutput&        output);
       void slice_task(const MapperContext      ctx,
                              const Task&              task, 
                              const SliceTaskInput&    input,
                                    SliceTaskOutput&   output);
       void map_task(const MapperContext      ctx,
                            const Task&              task,
                            const MapTaskInput&      input,
                                  MapTaskOutput&     output);
       void select_task_variant(const MapperContext          ctx,
                                       const Task&                  task,
                                       const SelectVariantInput&    input,
                                             SelectVariantOutput&   output);
       void postmap_task(const MapperContext      ctx,
                                const Task&              task,
                                const PostMapInput&      input,
                                      PostMapOutput&     output);
       void select_task_sources(const MapperContext        ctx,
                                       const Task&                task,
                                       const SelectTaskSrcInput&  input,
                                             SelectTaskSrcOutput& output);
       void speculate(const MapperContext      ctx,
                             const Task&              task,
                                   SpeculativeOutput& output);
       void report_profiling(const MapperContext      ctx,
                                    const Task&              task,
                                    const TaskProfilingInfo& input);
    public: // Inline mapping calls
       void map_inline(const MapperContext        ctx,
                              const InlineMapping&       inline_op,
                              const MapInlineInput&      input,
                                    MapInlineOutput&     output);
       void select_inline_sources(const MapperContext        ctx,
                                       const InlineMapping&         inline_op,
                                       const SelectInlineSrcInput&  input,
                                             SelectInlineSrcOutput& output);
       void report_profiling(const MapperContext         ctx,
                                    const InlineMapping&        inline_op,
                                    const InlineProfilingInfo&  input);
    public: // Copy mapping calls
       void map_copy(const MapperContext      ctx,
                            const Copy&              copy,
                            const MapCopyInput&      input,
                                  MapCopyOutput&     output);
       void select_copy_sources(const MapperContext          ctx,
                                       const Copy&                  copy,
                                       const SelectCopySrcInput&    input,
                                             SelectCopySrcOutput&   output);
       void speculate(const MapperContext      ctx,
                             const Copy& copy,
                                   SpeculativeOutput& output);
       void report_profiling(const MapperContext      ctx,
                                    const Copy&              copy,
                                    const CopyProfilingInfo& input);
    public: // Close mapping calls
       void map_close(const MapperContext       ctx,
                             const Close&              close,
                             const MapCloseInput&      input,
                                   MapCloseOutput&     output);
       void select_close_sources(const MapperContext        ctx,
                                        const Close&               close,
                                        const SelectCloseSrcInput&  input,
                                              SelectCloseSrcOutput& output);
       void report_profiling(const MapperContext       ctx,
                                    const Close&              close,
                                    const CloseProfilingInfo& input);
    public: // Acquire mapping calls
       void map_acquire(const MapperContext         ctx,
                               const Acquire&              acquire,
                               const MapAcquireInput&      input,
                                     MapAcquireOutput&     output);
       void speculate(const MapperContext         ctx,
                             const Acquire&              acquire,
                                   SpeculativeOutput&    output);
       void report_profiling(const MapperContext         ctx,
                                    const Acquire&              acquire,
                                    const AcquireProfilingInfo& input);
    public: // Release mapping calls
       void map_release(const MapperContext         ctx,
                               const Release&              release,
                               const MapReleaseInput&      input,
                                     MapReleaseOutput&     output);
       void select_release_sources(const MapperContext       ctx,
                                     const Release&                 release,
                                     const SelectReleaseSrcInput&   input,
                                           SelectReleaseSrcOutput&  output);
       void speculate(const MapperContext         ctx,
                             const Release&              release,
                                   SpeculativeOutput&    output);
       void report_profiling(const MapperContext         ctx,
                                    const Release&              release,
                                    const ReleaseProfilingInfo& input);
    public: // Task execution mapping calls
       void configure_context(const MapperContext         ctx,
                                     const Task&                 task,
                                           ContextConfigOutput&  output);
       void select_tunable_value(const MapperContext         ctx,
                                        const Task&                 task,
                                        const SelectTunableInput&   input,
                                              SelectTunableOutput&  output);
    public: // Must epoch mapping
       void map_must_epoch(const MapperContext           ctx,
                                  const MapMustEpochInput&      input,
                                        MapMustEpochOutput&     output);
    public: // Dataflow graph mapping
       void map_dataflow_graph(const MapperContext           ctx,
                                      const MapDataflowGraphInput&  input,
                                            MapDataflowGraphOutput& output);
    public: // Mapping control and stealing
       void select_tasks_to_map(const MapperContext          ctx,
                                       const SelectMappingInput&    input,
                                             SelectMappingOutput&   output);
       void select_steal_targets(const MapperContext         ctx,
                                        const SelectStealingInput&  input,
                                              SelectStealingOutput& output);
       void permit_steal_request(const MapperContext         ctx,
                                        const StealRequestInput&    input,
                                              StealRequestOutput&   output);
    public: // handling
       void handle_message(const MapperContext           ctx,
                                  const MapperMessage&          message);
       void handle_task_result(const MapperContext           ctx,
                                      const MapperTaskResult&       result);

    protected:
      const Processor       local_proc;
      const Processor::Kind local_kind;
      const AddressSpace    node_id;
      const Machine         machine;
      //const char *const     mapper_name;

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
    typedef Legion::Mapping::WrapperMapper WrapperMapper;
  };
};

#endif 

