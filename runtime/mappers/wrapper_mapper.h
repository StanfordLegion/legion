/* Copyright 2024 Stanford University, NVIDIA Corporation
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

/*
   This wrapper mapper takes a Legion mapper as a parameter and wraps it,
   so that the user can print information or stop program execution when 
   important properties of a task are set (properties set by 
   select_task_options())
   One can wrap individual mappers using the constructor
   Legion::Mapping::WrapperMapper
   (Mapper* mapper, MapperRuntime *rt, Machine machine, Processor local);

   At the start of the program execution, the user can enter the task names or 
   the processor indexes (index for a processor can be seen using the 
   "processors" command) to be monitored in a command-line interface.
   The commands to do so are as follows:

   stop task +<task_name> --> To stop when properties for task 
   "task_name" are set
   print task +<task_name> --> To print the properties of task "task_name"
   task -<task_name> --> To remove task "task_name" from the lists of tasks 
   which are being monitored
   processors --> To see the list of processors with their corresponding indexes
   stop processor +<processor_index> --> To stop when any task is mapped to a 
   particular processor
   print processor +<processor_index> --> To print properties when any task is 
   mapped to a particular processor
   processor -<processor_index> --> To remove a processor from the lists of 
   processors which are being monitored
   exit --> To exit from the command-line interface
   help --> To show the list of commands

   On stopping at a task/processor, one can change properties. 
   One can also enter the above command line interface to add or remove tasks 
   or processors by typing "change".
 */

#ifndef __WRAPPER_MAPPER_h__
#define __WRAPPER_MAPPER_h__

#include "legion.h"

#include <stdlib.h>
#include <assert.h>
#include <algorithm>

namespace Legion {
  namespace Mapping {

    //Struct to send properties set by select_task_options
    struct select_task_options_message{
      int tag;
      int  task_name;
      Mapper::TaskOptions output;
      int action;
    };
    /*
       struct get_input_message{
       int tag;
       Processor processor;
       std::map<Processor, int> procs_map;
       std::map<std::string, int> tasks_map;
       std::map<Memory, int> mems_map;
       };
     */	

    class WrapperMapper: public Mapper{

      public:
	Mapper* dmapper;
	//static std::map<std::string, int> tasks_map;
	static std::vector<std::string> print_tasks;
	static std::vector<std::string> stop_tasks;
	static std::map<int, int> methods_map;
	static std::set<Memory> all_mems;
	static std::set<Processor> all_procs;
	static std::map<Processor, int> procs_map;
	static std::map<int, int> procs_map_int;
	static bool inputtaken;
	static bool databroadcasted;
	static Processor ownerprocessor;
	static Processor localowner;
	static MapperEvent mapevent;
	static int broadcastcount;
	WrapperMapper(Mapper* dmapper, MapperRuntime *rt, 
	    Machine machine, Processor local);
	~WrapperMapper(void);
	void get_input();
	void get_input(const MapperContext ctx);
	void get_select_task_options_input
	  (const MapperContext ctx, std::string task_name, TaskOptions& output);
	bool InputNumberCheck(std::string strUserInput);
	Mapper::TaskOptions wait_task_options;

      public:
	void Deserialize(std::string rec_string);
	std::string Serialize(const std::map<std::string, int> &tasks_map, 
	    const std::map<int, int> &procs_map );
	std::string Serialize(const std::vector<std::string> &print_tasks, 
	    const std::vector<std::string> &stop_tasks,
	    const std::map<int, int> &procs_map);
	const char* get_mapper_name(void) const override;
	MapperSyncModel get_mapper_sync_model(void) const override;

        using Mapper::report_profiling;
      public: // Task mapping calls
	void select_task_options(const MapperContext    ctx,
	    const Task&            task,
	    TaskOptions&     output) override;
	void premap_task(const MapperContext      ctx,
	    const Task&              task, 
	    const PremapTaskInput&   input,
	    PremapTaskOutput&        output) override;
	void slice_task(const MapperContext      ctx,
	    const Task&              task, 
	    const SliceTaskInput&    input,
	    SliceTaskOutput&   output) override;
	void map_task(const MapperContext      ctx,
	    const Task&              task,
	    const MapTaskInput&      input,
	    MapTaskOutput&     output) override;
	void select_task_variant(const MapperContext          ctx,
	    const Task&                  task,
	    const SelectVariantInput&    input,
	    SelectVariantOutput&   output) override;
	void postmap_task(const MapperContext      ctx,
	    const Task&              task,
	    const PostMapInput&      input,
	    PostMapOutput&     output) override;
	void select_task_sources(const MapperContext        ctx,
	    const Task&                task,
	    const SelectTaskSrcInput&  input,
	    SelectTaskSrcOutput& output) override;
	void report_profiling(const MapperContext      ctx,
	    const Task&              task,
	    const TaskProfilingInfo& input) override;
      public: // Inline mapping calls
	void map_inline(const MapperContext        ctx,
	    const InlineMapping&       inline_op,
	    const MapInlineInput&      input,
	    MapInlineOutput&     output) override;
	void select_inline_sources(const MapperContext        ctx,
	    const InlineMapping&         inline_op,
	    const SelectInlineSrcInput&  input,
	    SelectInlineSrcOutput& output) override;
	void report_profiling(const MapperContext         ctx,
	    const InlineMapping&        inline_op,
	    const InlineProfilingInfo&  input) override;
      public: // Copy mapping calls
	void map_copy(const MapperContext      ctx,
	    const Copy&              copy,
	    const MapCopyInput&      input,
	    MapCopyOutput&     output) override;
	void select_copy_sources(const MapperContext          ctx,
	    const Copy&                  copy,
	    const SelectCopySrcInput&    input,
	    SelectCopySrcOutput&   output) override;
	void report_profiling(const MapperContext      ctx,
	    const Copy&              copy,
	    const CopyProfilingInfo& input) override;
      public: // Close mapping calls
	void select_close_sources(const MapperContext        ctx,
	    const Close&               close,
	    const SelectCloseSrcInput&  input,
	    SelectCloseSrcOutput& output) override;
	void report_profiling(const MapperContext       ctx,
	    const Close&              close,
	    const CloseProfilingInfo& input) override;
      public: // Acquire mapping calls
	void map_acquire(const MapperContext         ctx,
	    const Acquire&              acquire,
	    const MapAcquireInput&      input,
	    MapAcquireOutput&     output) override;
	void report_profiling(const MapperContext         ctx,
	    const Acquire&              acquire,
	    const AcquireProfilingInfo& input) override;
      public: // Release mapping calls
	void map_release(const MapperContext         ctx,
	    const Release&              release,
	    const MapReleaseInput&      input,
	    MapReleaseOutput&     output) override;
	void select_release_sources(const MapperContext       ctx,
	    const Release&                 release,
	    const SelectReleaseSrcInput&   input,
	    SelectReleaseSrcOutput&  output) override;
	void report_profiling(const MapperContext         ctx,
	    const Release&              release,
	    const ReleaseProfilingInfo& input) override;
      public: // Task execution mapping calls
	void configure_context(const MapperContext         ctx,
	    const Task&                 task,
	    ContextConfigOutput&  output) override;
	void select_tunable_value(const MapperContext         ctx,
	    const Task&                 task,
	    const SelectTunableInput&   input,
	    SelectTunableOutput&  output) override;
      public: // Must epoch mapping
	void map_must_epoch(const MapperContext           ctx,
	    const MapMustEpochInput&      input,
	    MapMustEpochOutput&     output) override;
      public: // Dataflow graph mapping
	void map_dataflow_graph(const MapperContext           ctx,
	    const MapDataflowGraphInput&  input,
	    MapDataflowGraphOutput& output) override;
      public: // Mapping control and stealing
	void select_tasks_to_map(const MapperContext          ctx,
	    const SelectMappingInput&    input,
	    SelectMappingOutput&   output) override;
	void select_steal_targets(const MapperContext         ctx,
	    const SelectStealingInput&  input,
	    SelectStealingOutput& output) override;
	void permit_steal_request(const MapperContext         ctx,
	    const StealRequestInput&    input,
	    StealRequestOutput&   output) override;
      public: // handling
	void handle_message(const MapperContext           ctx,
	    const MapperMessage&          message) override;
	void handle_task_result(const MapperContext           ctx,
	    const MapperTaskResult&       result) override;

      protected:
	MapperRuntime *const mrt;
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

#endif 

