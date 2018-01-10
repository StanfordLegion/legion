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

#include "legion.h"
#include "mappers/shim_mapper.h"

#include <stdlib.h>
#include <assert.h>
#include <algorithm>

#define STATIC_MAX_PERMITTED_STEALS   4
#define STATIC_MAX_STEAL_COUNT        2
#define STATIC_SPLIT_FACTOR           2
#define STATIC_BREADTH_FIRST          false
#define STATIC_WAR_ENABLED            false 
#define STATIC_STEALING_ENABLED       false
#define STATIC_MAX_SCHEDULE_COUNT     8
#define STATIC_NUM_PROFILE_SAMPLES    1
#define STATIC_MAX_FAILED_MAPPINGS    8

// This is the implementation of the shim mapper which provides
// backwards compatibility with the old version of the mapper interface

namespace Legion {
  namespace Mapping {

    using namespace Utilities;

    Logger log_shim("shim_mapper");

    //--------------------------------------------------------------------------
    ShimMapper::RegionRequirement::RegionRequirement(void)
    //--------------------------------------------------------------------------
    {
      restricted = false;
      max_blocking_factor = INT_MAX;
      virtual_map = false;
      early_map = false;
      enable_WAR_optimization = false;
      reduction_list = false;
      make_persistent = false;
      blocking_factor = INT_MAX;
      mapping_failed = false;
      selected_memory = Memory::NO_MEMORY;
    }

    //--------------------------------------------------------------------------
    ShimMapper::RegionRequirement& ShimMapper::RegionRequirement::operator=(
                                           const Legion::RegionRequirement &rhs)
    //--------------------------------------------------------------------------
    {
      region = rhs.region;
      partition = rhs.partition;
      privilege_fields = rhs.privilege_fields;
      instance_fields = rhs.instance_fields;
      privilege = rhs.privilege;
      prop = rhs.prop;
      parent = rhs.parent;
      redop = rhs.redop;
      tag = rhs.tag;
      flags = rhs.flags;
      handle_type = rhs.handle_type;
      projection = rhs.projection;
      return *this;
    }

    //--------------------------------------------------------------------------
    ShimMapper::Task::Task(const Legion::Task &t, TaskVariantCollection *var)
      : Legion::Task(t), variants(var), unique_id(t.get_unique_id()), 
        context_index(t.get_context_index()), depth(t.get_depth()), 
        task_name(t.get_task_name())
    //--------------------------------------------------------------------------
    {
      inline_task = false;
      map_locally = true;
      spawn_task = false;
      profile_task = false;
      selected_variant = 0;
      task_priority = 0;
      post_map_task = false;
      // Also fill in our region requirements by copying, sucks but whatever
      regions.resize(t.regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        regions[idx] = t.regions[idx];
    }

    //--------------------------------------------------------------------------
    ShimMapper::Mappable::MappableKind 
                                 ShimMapper::Task::get_mappable_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ShimMapper::Mappable::TASK_MAPPABLE;
    }

    //--------------------------------------------------------------------------
    ShimMapper::Task* ShimMapper::Task::as_mappable_task(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<Task*>(this);
    }

    //--------------------------------------------------------------------------
    ShimMapper::Copy* ShimMapper::Task::as_mappable_copy(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    ShimMapper::Inline* ShimMapper::Task::as_mappable_inline(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    UniqueID ShimMapper::Task::get_unique_mappable_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_id;
    }

    //--------------------------------------------------------------------------
    UniqueID ShimMapper::Task::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_id;
    }

    //--------------------------------------------------------------------------
    unsigned ShimMapper::Task::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    int ShimMapper::Task::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return depth;
    }

    //--------------------------------------------------------------------------
    const char* ShimMapper::Task::get_task_name(void) const
    //--------------------------------------------------------------------------
    {
      return task_name;
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::Task::has_trace(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    ShimMapper::Inline::Inline(const Legion::InlineMapping &m)
      : Legion::InlineMapping(m), unique_id(m.get_unique_id()), 
        context_index(m.get_context_index()), depth(m.get_depth())
    //--------------------------------------------------------------------------
    {
      requirement = m.requirement;
    }

    //--------------------------------------------------------------------------
    ShimMapper::Mappable::MappableKind 
                               ShimMapper::Inline::get_mappable_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ShimMapper::Mappable::INLINE_MAPPABLE;
    }

    //--------------------------------------------------------------------------
    ShimMapper::Task* ShimMapper::Inline::as_mappable_task(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    ShimMapper::Copy* ShimMapper::Inline::as_mappable_copy(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    ShimMapper::Inline* ShimMapper::Inline::as_mappable_inline(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<Inline*>(this);
    }

    //--------------------------------------------------------------------------
    UniqueID ShimMapper::Inline::get_unique_mappable_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_id;
    }

    //--------------------------------------------------------------------------
    UniqueID ShimMapper::Inline::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_id;
    }

    //--------------------------------------------------------------------------
    unsigned ShimMapper::Inline::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    int ShimMapper::Inline::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return depth;
    }
    
    //--------------------------------------------------------------------------
    ShimMapper::Copy::Copy(const Legion::Copy &c)
      : Legion::Copy(c), unique_id(c.get_unique_id()), 
        context_index(c.get_context_index()), depth(c.get_depth())
    //--------------------------------------------------------------------------
    {
      // fill in our region requirements by copying, sucks but whatever
      src_requirements.resize(c.src_requirements.size());
      dst_requirements.resize(c.dst_requirements.size());
      for (unsigned idx = 0; idx < c.src_requirements.size(); idx++)
      {
        src_requirements[idx] = c.src_requirements[idx];
        dst_requirements[idx] = c.dst_requirements[idx];
      }
    }

    //--------------------------------------------------------------------------
    ShimMapper::Mappable::MappableKind 
                                 ShimMapper::Copy::get_mappable_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ShimMapper::Mappable::COPY_MAPPABLE;
    }

    //--------------------------------------------------------------------------
    ShimMapper::Task* ShimMapper::Copy::as_mappable_task(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    ShimMapper::Copy* ShimMapper::Copy::as_mappable_copy(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<Copy*>(this);
    }

    //--------------------------------------------------------------------------
    ShimMapper::Inline* ShimMapper::Copy::as_mappable_inline(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    UniqueID ShimMapper::Copy::get_unique_mappable_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_id;
    }

    //--------------------------------------------------------------------------
    UniqueID ShimMapper::Copy::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_id;
    }

    //--------------------------------------------------------------------------
    unsigned ShimMapper::Copy::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    int ShimMapper::Copy::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return depth;
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::TaskVariantCollection::has_variant(Processor::Kind kind, 
                                                  bool single, bool index_space)
    //--------------------------------------------------------------------------
    {
      for (std::map<VariantID,Variant>::const_iterator it = variants.begin();
            it != variants.end(); it++)
      {
        if ((it->second.proc_kind == kind) && 
            ((it->second.single_task <= single) || 
            (it->second.index_space <= index_space)))
        {
          return true;
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    VariantID ShimMapper::TaskVariantCollection::get_variant(
                            Processor::Kind kind, bool single, bool index_space)
    //--------------------------------------------------------------------------
    {
      for (std::map<VariantID,Variant>::const_iterator it = variants.begin();
            it != variants.end(); it++)
      {
        if ((it->second.proc_kind == kind) && 
            ((it->second.single_task <= single) || 
            (it->second.index_space <= index_space)))
        {
          return it->first;
        }
      }
      return 0;
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::TaskVariantCollection::has_variant(VariantID vid)
    //--------------------------------------------------------------------------
    {
      return (variants.find(vid) != variants.end());
    }

    //--------------------------------------------------------------------------
    const ShimMapper::TaskVariantCollection::Variant& 
                   ShimMapper::TaskVariantCollection::get_variant(VariantID vid)
    //--------------------------------------------------------------------------
    {
      assert(variants.find(vid) != variants.end());
      return variants[vid];
    }

    //--------------------------------------------------------------------------
    void ShimMapper::TaskVariantCollection::add_variant(
            Processor::TaskFuncID low_id, Processor::Kind kind, 
            bool single, bool index, bool inner, bool leaf, VariantID vid)
    //--------------------------------------------------------------------------
    {
      if (vid == AUTO_GENERATE_ID)
      {
        for (unsigned idx = 0; idx < AUTO_GENERATE_ID; idx++)
        {
          if (variants.find(idx) == variants.end())
          {
            vid = idx;
            break;
          }
        }
      }
      variants[vid] = Variant(low_id, kind, single, index, inner, leaf, vid);
    }

    //--------------------------------------------------------------------------
    const ShimMapper::TaskVariantCollection::Variant& 
     ShimMapper::TaskVariantCollection::select_variant(bool single, bool index, 
                                                       Processor::Kind kind)
    //--------------------------------------------------------------------------
    {
      for (std::map<VariantID,Variant>::const_iterator it = variants.begin();
            it != variants.end(); it++)
      {
        if ((it->second.proc_kind == kind) && 
            (it->second.single_task <= single) &&
            (it->second.index_space <= index))
        {
          return it->second;
        }
      }
      return variants[0];
    }

    //--------------------------------------------------------------------------
    ShimMapper::ShimMapper(Machine m, Runtime *rt, MapperRuntime *mrt, 
                           Processor local, const char *name/*=NULL*/)
      : DefaultMapper(mrt, m, local,(name == NULL) ? 
          create_shim_name(local) : name),
        mapper_runtime(mrt),
        local_kind(local.kind()), machine(m), runtime(rt),
        max_steals_per_theft(STATIC_MAX_PERMITTED_STEALS),
        max_steal_count(STATIC_MAX_STEAL_COUNT),
        splitting_factor(STATIC_SPLIT_FACTOR),
        breadth_first_traversal(STATIC_BREADTH_FIRST),
        war_enabled(STATIC_WAR_ENABLED),
        stealing_enabled(STATIC_STEALING_ENABLED),
        max_schedule_count(STATIC_MAX_SCHEDULE_COUNT),
        max_failed_mappings(STATIC_MAX_FAILED_MAPPINGS),
        machine_interface(MachineQueryInterface(m))
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Initializing the shim mapper for processor " IDFMT "",
                     local_proc.id);
      // Check to see if there any input arguments to parse
      {
        int argc = Runtime::get_input_args().argc;
        char **argv = Runtime::get_input_args().argv;
        unsigned num_profiling_samples = STATIC_NUM_PROFILE_SAMPLES;
        // Parse the input arguments looking for ones for the shim mapper
        for (int i=1; i < argc; i++)
        {
#define INT_ARG(argname, varname) do {      \
          if (!strcmp(argv[i], argname)) {  \
            varname = atoi(argv[++i]);      \
            continue;                       \
          } } while(0);
#define BOOL_ARG(argname, varname) do {       \
          if (!strcmp(argv[i], argname)) {    \
            varname = (atoi(argv[++i]) != 0); \
            continue;                         \
          } } while(0);
          INT_ARG("-dm:thefts", max_steals_per_theft);
          INT_ARG("-dm:count", max_steal_count);
          INT_ARG("-dm:split", splitting_factor);
          BOOL_ARG("-dm:war", war_enabled);
          BOOL_ARG("-dm:steal", stealing_enabled);
          BOOL_ARG("-dm:bft", breadth_first_traversal);
          INT_ARG("-dm:sched", max_schedule_count);
          INT_ARG("-dm:prof",num_profiling_samples);
          INT_ARG("-dm:fail",max_failed_mappings);
#undef BOOL_ARG
#undef INT_ARG
        }
        profiler.set_needed_profiling_samples(num_profiling_samples);
      }
    }

    //--------------------------------------------------------------------------
    ShimMapper::ShimMapper(const ShimMapper &rhs)
      : DefaultMapper(rhs.mapper_runtime, Machine::get_machine(), 
          Processor::NO_PROC),
        mapper_runtime(rhs.mapper_runtime),
        local_kind(Processor::LOC_PROC), machine(rhs.machine), runtime(NULL),
        machine_interface(Utilities::MachineQueryInterface(rhs.machine))
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ShimMapper::~ShimMapper(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ShimMapper& ShimMapper::operator=(const ShimMapper &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    /*static*/ const char* ShimMapper::create_shim_name(Processor p)
    //--------------------------------------------------------------------------
    {
      const size_t buffer_size = 64;
      char *result = (char*)malloc(buffer_size*sizeof(char));
      snprintf(result, buffer_size-1,
                "Shim Mapper on Processor " IDFMT "", p.id);
      return result;
    }

    //--------------------------------------------------------------------------
    Mapper::MapperSyncModel ShimMapper::get_mapper_sync_model(void) const
    //--------------------------------------------------------------------------
    {
      // The way we track current_ctx requires non-reentrancy
      return SERIALIZED_NON_REENTRANT_MAPPER_MODEL;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::select_task_options(const MapperContext  ctx,
                                         const Legion::Task   &task,
                                               TaskOptions    &output)
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Shim mapper select_task_options in %s", get_mapper_name());
      // Wrap the task with one of our tasks
      Task local_task(task, find_task_variant_collection(ctx, task.task_id,
                                                         task.get_task_name()));
      // Copy over the defaults
      local_task.target_proc = output.initial_proc;
      local_task.inline_task = output.inline_task;
      local_task.map_locally = output.map_locally;
      local_task.spawn_task = output.stealable;
      local_task.profile_task = false;
      // Save the current context before doing any old calls
      current_ctx = ctx;
      // Invoke the old mapper call
      this->select_task_options(&local_task);
      // Copy the results back
      output.initial_proc = local_task.target_proc;
      output.inline_task = local_task.inline_task;
      output.map_locally = local_task.map_locally;
      output.stealable = local_task.spawn_task;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::select_task_options(Task *task)
    //--------------------------------------------------------------------------
    {
      task->inline_task = false;
      task->spawn_task = false;
      task->map_locally = true;
      task->target_proc = local_proc;
      task->profile_task = false;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::select_task_variant(const MapperContext          ctx,
                                         const Legion::Task&          task,
                                         const SelectVariantInput&    input,
                                               SelectVariantOutput&   output)
    //--------------------------------------------------------------------------
    {
      DefaultMapper::select_task_variant(ctx, task, input, output);
    }

    //--------------------------------------------------------------------------
    void ShimMapper::map_task(const MapperContext       ctx,
                              const Legion::Task&       task,
                              const MapTaskInput&       input,
                                    MapTaskOutput&      output)
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Shim mapper map_task in %s", get_mapper_name());
      // Make our local task wrapper
      Task local_task(task, find_task_variant_collection(ctx, task.task_id,
                                                         task.get_task_name()));
      // fill in the current instances
      for (unsigned idx = 0; idx < local_task.regions.size(); idx++)
        initialize_requirement_mapping_fields(local_task.regions[idx],
                                              input.valid_instances[idx]);
      // mark which regions are restricted (based on premapping)
      for (std::vector<unsigned>::const_iterator it = 
            input.premapped_regions.begin(); it !=
            input.premapped_regions.end(); it++)
        local_task.regions[*it].restricted = true;
      // Save the current context before doing any old calls
      current_ctx = ctx;
      // Call select variant first to get it over with
      select_task_variant(&local_task);
      const std::set<unsigned> premapped_set(input.premapped_regions.begin(),
                                             input.premapped_regions.end());
      // Do this in a while-true loop until we succeed in mapping
      while (true)
      {
        // Now keep calling map task until we succeed
        bool report = map_task(&local_task);
        bool success = true;
        // Now find or make the physical instances asked for each region
        for (unsigned idx = 0; idx < local_task.regions.size(); idx++)
        {
          // Check to see if this is a premapped region in which case 
          // we can skip the conversion cause it doesn't matter
          if (premapped_set.find(idx) != premapped_set.end())
            continue;
          if (!convert_requirement_mapping(ctx, local_task.regions[idx],
                                           output.chosen_instances[idx]))
          {
            success = false;
            break;
          }
        }
        if (success) 
        {
          if (report)
            notify_mapping_result(&local_task);
          // Translate the results back
          if (!local_task.additional_procs.empty())
            output.target_procs.insert(output.target_procs.end(),
                local_task.additional_procs.begin(), 
                local_task.additional_procs.end());
          if (local_task.additional_procs.find(task.target_proc) == 
              local_task.additional_procs.end())
            output.target_procs.push_back(task.target_proc);
          output.task_priority = local_task.task_priority;
          output.postmap_task = local_task.post_map_task;
          output.chosen_variant = local_task.selected_variant;
          break; 
        }
        // Report the failed mapping
        notify_mapping_failed(&local_task);
        // otherwise clear everything out for another try
        local_task.additional_procs.clear();
        local_task.task_priority = 0;
        local_task.post_map_task = false;
        for (unsigned idx = 0; idx < local_task.regions.size(); idx++)
        {
          local_task.regions[idx].mapping_failed = false;
          local_task.regions[idx].selected_memory = Memory::NO_MEMORY;
        }
        mapper_runtime->release_instances(ctx, output.chosen_instances);
        // Back around the loop we go
      }
    }

    //--------------------------------------------------------------------------
    void ShimMapper::select_task_variant(Task *task)
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Select task variant in shim mapper for task %s "
                    "(ID %lld) for processor " IDFMT "", task->variants->name,
                    task->get_unique_task_id(), local_proc.id);
      Processor::Kind target_kind = task->target_proc.kind();
      if (!task->variants->has_variant(target_kind, 
            !(task->is_index_space), task->is_index_space))
      {
        log_shim.error("Mapper unable to find variant for task %s (ID %lld)",
                       task->variants->name, 
                       task->get_unique_task_id());
        assert(false);
      }
      task->selected_variant = task->variants->get_variant(target_kind,
                                                      !(task->is_index_space),
                                                      task->is_index_space);
      if (target_kind == Processor::LOC_PROC)
      {
        for (unsigned idx = 0; idx < task->regions.size(); idx++)
          // Elliott needs SOA for the compiler.
          task->regions[idx].blocking_factor = // 1;
            task->regions[idx].max_blocking_factor;
      }
      else
      {
        for (unsigned idx = 0; idx < task->regions.size(); idx++)
          task->regions[idx].blocking_factor = 
            task->regions[idx].max_blocking_factor;
      }
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::map_task(Task *task)
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Map task in shim mapper for task %s "
                    "(ID %lld) for processor " IDFMT "",
                    task->variants->name,
                    task->get_unique_task_id(), local_proc.id);
      Processor::Kind target_kind = task->target_proc.kind();
      for (unsigned idx = 0; idx < task->regions.size(); idx++)
      {
        // See if this instance is restricted
        if (!task->regions[idx].restricted)
        {
          // Check to see if our memoizer already has mapping for us to use
          if (memoizer.has_mapping(task->target_proc, task, idx))
          {
            memoizer.recall_mapping(task->target_proc, task, idx,
                                    task->regions[idx].target_ranking);
          }
          else
          {
            machine_interface.find_memory_stack(task->target_proc,
                    task->regions[idx].target_ranking,
                    (task->target_proc.kind() == Processor::LOC_PROC));
            memoizer.record_mapping(task->target_proc, task, idx,
                                    task->regions[idx].target_ranking);
          }
        }
        else
        {
          assert(task->regions[idx].current_instances.size() == 1);
          Memory target = (task->regions[idx].current_instances.begin())->first;
          task->regions[idx].target_ranking.push_back(target);
        }
        task->regions[idx].virtual_map = false;
        task->regions[idx].enable_WAR_optimization = war_enabled;
        task->regions[idx].reduction_list = false;
        task->regions[idx].make_persistent = false;
        if (target_kind == Processor::LOC_PROC)
          // Elliott needs SOA for the compiler.
          task->regions[idx].blocking_factor = // 1;
            task->regions[idx].max_blocking_factor;
        else
          task->regions[idx].blocking_factor = 
            task->regions[idx].max_blocking_factor;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::notify_mapping_result(const Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      UniqueID uid = mappable->get_unique_mappable_id();
      // We should only get this for tasks in the default mapper
      if (mappable->get_mappable_kind() == Mappable::TASK_MAPPABLE)
      {
        const Task *task = mappable->as_mappable_task();
        assert(task != NULL);
        log_shim.spew("Notify mapping for task %s (ID %lld) in "
                      "shim mapper for processor " IDFMT "",
                      task->variants->name, uid, local_proc.id);
        for (unsigned idx = 0; idx < task->regions.size(); idx++)
        {
          memoizer.notify_mapping(task->target_proc, task, idx, 
                                  task->regions[idx].selected_memory);
        }
      }
      std::map<UniqueID,unsigned>::iterator finder = failed_mappings.find(uid);
      if (finder != failed_mappings.end())
        failed_mappings.erase(finder);
    }

    //--------------------------------------------------------------------------
    void ShimMapper::notify_mapping_failed(const Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      UniqueID uid = mappable->get_unique_mappable_id(); 
      log_shim.warning("Notify failed mapping for operation ID %lld "
                       "in shim mapper for processor " IDFMT "! Retrying...",
                       uid, local_proc.id);
      std::map<UniqueID,unsigned>::iterator finder = failed_mappings.find(uid);
      if (finder == failed_mappings.end())
        failed_mappings[uid] = 1;
      else
      {
        finder->second++;
        if (finder->second == max_failed_mappings)
        {
          log_shim.error("Reached maximum number of failed mappings "
                         "for operation ID %lld in shim mapper for "
                         "processor " IDFMT "!  Try implementing a "
                         "custom mapper or changing the size of the "
                         "memories in the low-level runtime. "
                         "Failing out ...", uid, local_proc.id);
          assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ShimMapper::map_copy(const MapperContext               ctx,
                              const Legion::Copy&               copy,
                              const MapCopyInput&               input,
                                    MapCopyOutput&              output)
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Shim mapper map_copy in %s", get_mapper_name());
      Copy local_copy(copy);
      // Fill in the current instances
      for (unsigned idx = 0; idx < local_copy.src_requirements.size(); idx++)
      {
        initialize_requirement_mapping_fields(local_copy.src_requirements[idx],
                                              input.src_instances[idx]);
        initialize_requirement_mapping_fields(local_copy.dst_requirements[idx],
                                              input.dst_instances[idx]);
      }
      // Save the current context before doing any old calls
      current_ctx = ctx;
      // Do this in a while-true loop until we succeed in mapping
      while (true)
      {
        // Now keep calling map copy until we succeed
        bool report = map_copy(&local_copy);
        bool success = true;
        for (unsigned idx = 0; idx < local_copy.src_requirements.size(); idx++)
        {
          if (!convert_requirement_mapping(ctx,local_copy.src_requirements[idx],
                                           output.src_instances[idx]))
          {
            success = false;
            break;
          }
          if (!convert_requirement_mapping(ctx,local_copy.dst_requirements[idx],
                                           output.dst_instances[idx]))
          {
            success = false;
            break;
          }
        }
        if (success)
        {
          if (report)
            notify_mapping_result(&local_copy);
          break;
        }
        // Report the failed mapping
        notify_mapping_failed(&local_copy);
        // clear everything out for another try
        for (unsigned idx = 0; idx < local_copy.src_requirements.size(); idx++)
        {
          local_copy.src_requirements[idx].mapping_failed = false;
          local_copy.src_requirements[idx].selected_memory = Memory::NO_MEMORY;
          local_copy.dst_requirements[idx].mapping_failed = false;
          local_copy.dst_requirements[idx].selected_memory = Memory::NO_MEMORY;
        }
        mapper_runtime->release_instances(ctx, output.src_instances);
        mapper_runtime->release_instances(ctx, output.dst_instances);
        // Back around the loop we go
      }
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::map_copy(Copy *copy)
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Map copy for copy ID %lld in shim mapper "
                    "for processor " IDFMT "", 
                    copy->get_unique_copy_id(), local_proc.id);
      std::vector<Memory> local_stack; 
      machine_interface.find_memory_stack(local_proc, local_stack,
                                          (local_kind == Processor::LOC_PROC)); 
      assert(copy->src_requirements.size() == copy->dst_requirements.size());
      for (unsigned idx = 0; idx < copy->src_requirements.size(); idx++)
      {
        copy->src_requirements[idx].virtual_map = false;
        copy->src_requirements[idx].early_map = false;
        copy->src_requirements[idx].enable_WAR_optimization = war_enabled;
        copy->src_requirements[idx].reduction_list = false;
        copy->src_requirements[idx].make_persistent = false;
        if (!copy->src_requirements[idx].restricted)
        {
          copy->src_requirements[idx].target_ranking = local_stack;
        }
        copy->dst_requirements[idx].virtual_map = false;
        copy->dst_requirements[idx].early_map = false;
        copy->dst_requirements[idx].enable_WAR_optimization = war_enabled;
        copy->dst_requirements[idx].reduction_list = false;
        copy->dst_requirements[idx].make_persistent = false;
        if (!copy->dst_requirements[idx].restricted)
        {
          copy->dst_requirements[idx].target_ranking = local_stack;
        }
        if (local_kind == Processor::LOC_PROC)
        {
          // Elliott needs SOA for the compiler.
          copy->src_requirements[idx].blocking_factor = // 1;
            copy->src_requirements[idx].max_blocking_factor;
          copy->dst_requirements[idx].blocking_factor = // 1;
            copy->dst_requirements[idx].max_blocking_factor;
        }
        else
        {
          copy->src_requirements[idx].blocking_factor = 
            copy->src_requirements[idx].max_blocking_factor;
          copy->dst_requirements[idx].blocking_factor = 
            copy->dst_requirements[idx].max_blocking_factor;
        } 
      }
      // No profiling on copies yet
      return false;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::map_inline(const MapperContext        ctx,
                                const InlineMapping&       inline_op,
                                const MapInlineInput&      input,
                                      MapInlineOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Shim mapper map_inline in %s", get_mapper_name());
      Inline local_inline(inline_op);
      initialize_requirement_mapping_fields(local_inline.requirement,
                                            input.valid_instances);
      // Save the current context before doing any old calls
      current_ctx = ctx;
      // Do this in a while-true loop until we succeed in mapping
      while (true)
      {
        // Now keep calling map inline until we succeed
        bool report = map_inline(&local_inline);
        if (convert_requirement_mapping(ctx, local_inline.requirement,
                                        output.chosen_instances))
        {
          // We succeeded
          if (report)
            notify_mapping_result(&local_inline);
          break;
        }
        // Report the mapping failed
        notify_mapping_failed(&local_inline);
        local_inline.requirement.mapping_failed = false;
        local_inline.requirement.selected_memory = Memory::NO_MEMORY;
        mapper_runtime->release_instances(ctx, output.chosen_instances);
        // Back around the loop we go
      }
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::map_inline(Inline *inline_op)
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Map inline for operation ID %lld in shim "
                    "mapper for processor " IDFMT "",
                    inline_op->get_unique_inline_id(), local_proc.id);
      inline_op->requirement.virtual_map = false;
      inline_op->requirement.early_map = false;
      inline_op->requirement.enable_WAR_optimization = war_enabled;
      inline_op->requirement.reduction_list = false;
      inline_op->requirement.make_persistent = false;
      if (!inline_op->requirement.restricted)
      {
        machine_interface.find_memory_stack(local_proc, 
                                          inline_op->requirement.target_ranking,
                                          (local_kind == Processor::LOC_PROC));
      }
      else
      {
        assert(inline_op->requirement.current_instances.size() == 1);
        Memory target = 
          (inline_op->requirement.current_instances.begin())->first;
        inline_op->requirement.target_ranking.push_back(target);
      }
      if (local_kind == Processor::LOC_PROC)
        // Elliott needs SOA for the compiler.
        inline_op->requirement.blocking_factor = // 1;
          inline_op->requirement.max_blocking_factor;
      else
        inline_op->requirement.blocking_factor = 
          inline_op->requirement.max_blocking_factor;
      // No profiling on inline mappings
      return false;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::slice_task(const MapperContext             ctx,
                                const Legion::Task&             task, 
                                const SliceTaskInput&           input,
                                      SliceTaskOutput&          output)
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Shim mapper slice_task in %s", get_mapper_name());
      Task local_task(task, find_task_variant_collection(ctx, task.task_id,
                                                         task.get_task_name()));
      slice_domain(&local_task, input.domain, output.slices);
    }

    //--------------------------------------------------------------------------
    void ShimMapper::slice_domain(const Task *task, const Domain &domain,
                                  std::vector<DomainSplit> &slices)
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Slice index space in shim mapper for task %s "
                    "(ID %lld) for processor " IDFMT "", task->variants->name,
                    task->get_unique_task_id(), local_proc.id);

      Processor::Kind best_kind;
      if (profiler.profiling_complete(task))
        best_kind = profiler.best_processor_kind(task);
      else
        best_kind = Processor::LOC_PROC;
      std::set<Processor> all_procs;
      machine.get_all_processors(all_procs);
      machine_interface.filter_processors(machine, best_kind, all_procs);
      std::vector<Processor> procs(all_procs.begin(),all_procs.end());

      ShimMapper::decompose_index_space(domain, procs, 
                                        splitting_factor, slices);
    }

    //--------------------------------------------------------------------------
    template <unsigned DIM>
    static void round_robin_point_assign(const Domain &domain, 
                                         const std::vector<Processor> &targets,
					 unsigned splitting_factor, 
                                   std::vector<ShimMapper::DomainSplit> &slices)
    //--------------------------------------------------------------------------
    {
      Rect<DIM,coord_t> r = domain;

      std::vector<Processor>::const_iterator target_it = targets.begin();
      for(PointInRectIterator<DIM> pir(r); pir(); pir++) 
      {
        // rect containing a single point
        Rect<DIM> subrect(*pir, *pir);
	ShimMapper::DomainSplit ds(subrect, 
            *target_it++, false /* recurse */, false /* stealable */);
	slices.push_back(ds);
	if(target_it == targets.end())
	  target_it = targets.begin();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShimMapper::decompose_index_space(const Domain &domain, 
                                          const std::vector<Processor> &targets,
                                          unsigned splitting_factor, 
                                          std::vector<DomainSplit> &slices)
    //--------------------------------------------------------------------------
    {
      switch(domain.get_dim()) {
      case 2:
	round_robin_point_assign<2>(domain, targets, splitting_factor, slices);
	return;

      case 3:
	round_robin_point_assign<3>(domain, targets, splitting_factor, slices);
	return;

	// cases 0 and 1 fall through to old code for now
      }

      // Only handle these two cases right now
      assert((domain.get_dim() == 0) || (domain.get_dim() == 1));
      if (domain.get_dim() == 0)
        assert(false);
      else
      {
        // Only works for one dimensional rectangles right now
        assert(domain.get_dim() == 1);
        Rect<1,coord_t> rect = domain;
        unsigned num_elmts = rect.volume();
        unsigned num_chunks = targets.size()*splitting_factor;
        if (num_chunks > num_elmts)
          num_chunks = num_elmts;
        // Number of elements per chunk rounded up
        // which works because we know that rectangles are contiguous
        unsigned lower_bound = num_elmts/num_chunks;
        unsigned upper_bound = lower_bound+1;
        unsigned number_small = num_chunks - (num_elmts % num_chunks);
        unsigned index = 0;
        for (unsigned idx = 0; idx < num_chunks; idx++)
        {
          unsigned elmts = (idx < number_small) ? lower_bound : upper_bound;
          Point<1,coord_t> lo(index);  
          Point<1,coord_t> hi(index+elmts-1);
          index += elmts;
          Rect<1,coord_t> chunk(rect.lo+lo,rect.lo+hi);
          unsigned proc_idx = idx % targets.size();
          slices.push_back(DomainSplit(chunk, targets[proc_idx], false, false));
        }
      }
    }

    //--------------------------------------------------------------------------
    void ShimMapper::select_tunable_value(const MapperContext         ctx,
					  const Legion::Task&         task,
					  const SelectTunableInput&   input,
                                                SelectTunableOutput&  output)
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Shim mapper select_tunable_value in %s", get_mapper_name());
      // Wrap the task with one of our tasks
      Task local_task(task, find_task_variant_collection(ctx, task.task_id,
                                                         task.get_task_name()));
      // Save the current context before doing any old calls
      current_ctx = ctx;
      // call old version - returns int directly, but we'll store as size_t
      //  for consistency with the new DefaultMapper tunables
      size_t *result = (size_t*)malloc(sizeof(size_t));
      output.value = result;
      output.size = sizeof(size_t);
      *result = get_tunable_value(&local_task,
				  input.tunable_id,
				  input.mapping_tag);
    }

    //--------------------------------------------------------------------------
    int ShimMapper::get_tunable_value(const Task *task, 
				      TunableID tid, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      log_shim.error("Shim mapper doesn't support any tunables directly!");
      assert(0);
      //should never happen, but function needs return statement
      return(STATIC_MAX_FAILED_MAPPINGS);
    }

    //--------------------------------------------------------------------------
    void ShimMapper::handle_message(const MapperContext ctx,
                                    const MapperMessage& message)
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Shim mapper handle_message in %s", get_mapper_name());
      handle_message(message.sender, message.message, message.size);
    }

    //--------------------------------------------------------------------------
    void ShimMapper::handle_message(Processor source,
                                    const void *message, size_t length)
    //--------------------------------------------------------------------------
    {
      log_shim.spew("Old handle message call for %s", get_mapper_name());
      // We should never get this call since the base shim mapper
      // never sends any kind of messages
      assert(false);
    }

    //--------------------------------------------------------------------------
    Color ShimMapper::get_logical_region_color(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return mapper_runtime->get_logical_region_color(current_ctx, handle);
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::has_parent_logical_partition(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return mapper_runtime->has_parent_logical_partition(current_ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition ShimMapper::get_parent_logical_partition(
                                                           LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return mapper_runtime->get_parent_logical_partition(current_ctx, handle);
    }
    
    //--------------------------------------------------------------------------
    LogicalRegion ShimMapper::get_parent_logical_region(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return mapper_runtime->get_parent_logical_region(current_ctx, handle);
    }

    //--------------------------------------------------------------------------
    void ShimMapper::get_field_space_fields(FieldSpace handle, std::set<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      return mapper_runtime->get_field_space_fields(current_ctx, handle, fields);
    }

    //--------------------------------------------------------------------------
    void ShimMapper::broadcast_message(const void *message, size_t message_size)
    //--------------------------------------------------------------------------
    {
      mapper_runtime->broadcast(current_ctx, message, message_size);
    }

    //--------------------------------------------------------------------------
    ShimMapper::TaskVariantCollection* ShimMapper::find_task_variant_collection(
                       MapperContext ctx, TaskID task_id, const char *task_name)
    //--------------------------------------------------------------------------
    {
      std::map<TaskID,TaskVariantCollection*>::const_iterator finder = 
        task_variant_collections.find(task_id);
      if (finder != task_variant_collections.end())
        return finder->second;
      TaskVariantCollection *collection = new TaskVariantCollection(task_id,
                                                task_name, false, 0/*ret*/);
      // Get all the variants for each of the processor kinds
      std::vector<VariantID> cpu_variants, gpu_variants, io_variants;  
      mapper_runtime->find_valid_variants(ctx, task_id, 
                                    cpu_variants, Processor::LOC_PROC);
      mapper_runtime->find_valid_variants(ctx, task_id,
                                    gpu_variants, Processor::TOC_PROC);
      mapper_runtime->find_valid_variants(ctx, task_id,
                                    io_variants, Processor::IO_PROC);
      for (std::vector<VariantID>::const_iterator it = cpu_variants.begin();
            it != cpu_variants.end(); it++)
      {
        bool is_leaf = mapper_runtime->is_leaf_variant(ctx, task_id, *it);
        bool is_inner = mapper_runtime->is_inner_variant(ctx, task_id, *it);
        collection->add_variant(*it, Processor::LOC_PROC, true, true,
                                is_inner, is_leaf, *it);
      }
      for (std::vector<VariantID>::const_iterator it = gpu_variants.begin();
            it != gpu_variants.end(); it++)
      {
        bool is_leaf = mapper_runtime->is_leaf_variant(ctx, task_id, *it);
        bool is_inner = mapper_runtime->is_inner_variant(ctx, task_id, *it);
        collection->add_variant(*it, Processor::TOC_PROC, true, true,
                                is_inner, is_leaf, *it);
      }
      for (std::vector<VariantID>::const_iterator it = io_variants.begin();
            it != io_variants.end(); it++)
      {
        bool is_leaf = mapper_runtime->is_leaf_variant(ctx, task_id, *it);
        bool is_inner = mapper_runtime->is_inner_variant(ctx, task_id, *it);
        collection->add_variant(*it, Processor::IO_PROC, true, true,
                                is_inner, is_leaf, *it);
      }
      task_variant_collections[task_id] = collection;
      return collection;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::initialize_requirement_mapping_fields(
             RegionRequirement &req, const std::vector<PhysicalInstance> &valid)
    //--------------------------------------------------------------------------
    {
      std::map<Memory,bool> &current = req.current_instances; 
      for (std::vector<PhysicalInstance>::const_iterator it = 
            valid.begin(); it != valid.end(); it++)
      {
        std::set<FieldID> space_fields = req.privilege_fields; 
        it->remove_space_fields(space_fields);
        bool has_space = false;
        if (space_fields.empty())
          has_space = true;
        std::map<Memory,bool>::iterator finder = 
          current.find(it->get_location());
        if (finder != current.end())
        {
          if (!finder->second && has_space)
            finder->second = true;
        }
        else
          current[it->get_location()] = has_space;
      }
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::convert_requirement_mapping(MapperContext ctx,
                  RegionRequirement &req, std::vector<PhysicalInstance> &result)
    //--------------------------------------------------------------------------
    {
      if (req.virtual_map)
      {
        result.push_back(PhysicalInstance::get_virtual_instance());
        return true;
      }
      if (req.reduction_list)
      {
        log_shim.error("Shim Mapper Error: List reduction instances "
                       "are not currently supported.");
        assert(false);
      }
      if (req.target_ranking.empty())
      {
        req.mapping_failed = true;
        return false;
      }
      std::vector<LogicalRegion> space_regions(1, req.region);
      LayoutConstraintSet constraints;
      std::set<FieldID> fields = req.privilege_fields;
      GCPriority gc_priority = 0;
      if (req.make_persistent)
        gc_priority = GC_NEVER_PRIORITY; 
      if (!req.additional_fields.empty())
        fields.insert(req.additional_fields.begin(),
                      req.additional_fields.end());
      if (req.blocking_factor == 1)
        initialize_aos_constraints(constraints, fields, req.redop);
      else if (req.blocking_factor == INT_MAX)
        initialize_soa_constraints(constraints, fields, req.redop);
      else
      {
        log_shim.error("Shim Mapper Error: Illegal layout constraints. "
                       "Only SOA and AOS are supported.");
        assert(false);
      }
      const bool create_only = 
        (req.privilege == REDUCE) || req.enable_WAR_optimization;
      // Try all the memories
      for (std::vector<Memory>::const_iterator it = req.target_ranking.begin();
            it != req.target_ranking.end(); it++)
      {
        PhysicalInstance instance;
        if (create_only)
        {
          if (mapper_runtime->create_physical_instance(ctx, *it, constraints,
                space_regions, instance, true/*acquire*/, gc_priority))
          {
            result.push_back(instance);
            req.selected_memory = instance.get_location();
            return true;
          }
        }
        else
        {
          bool created;
          if (mapper_runtime->find_or_create_physical_instance(ctx, *it,
                constraints, space_regions, instance, created, 
                true/*acquire*/, gc_priority))
          {
            result.push_back(instance);
            req.selected_memory = instance.get_location();
            return true;
          }
        }
      }
      req.mapping_failed = true;
      return false;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::initialize_aos_constraints(
                                   LayoutConstraintSet &constraints, 
                                   const std::set<FieldID> &fields, 
                                   ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID> all_fields(fields.begin(), fields.end());
      if (redop > 0)
      {
        assert(all_fields.size() == 1);
        constraints.add_constraint(SpecializedConstraint(
                                      REDUCTION_FOLD_SPECIALIZE, redop))
          .add_constraint(FieldConstraint(all_fields, true/*contiguous*/,
                                          true/*inorder*/));
      }
      else
      {
        std::vector<DimensionKind> dimension_ordering(4);
        dimension_ordering[0] = DIM_F;
        dimension_ordering[1] = DIM_X;
        dimension_ordering[2] = DIM_Y;
        dimension_ordering[3] = DIM_Z;
        constraints.add_constraint(SpecializedConstraint())
          .add_constraint(FieldConstraint(all_fields, true/*contiguous*/,
                                          true/*inorder*/))
          .add_constraint(OrderingConstraint(dimension_ordering, 
                                             false/*contigous*/));
      }
    }

    //--------------------------------------------------------------------------
    void ShimMapper::initialize_soa_constraints(
                                   LayoutConstraintSet &constraints,
                                   const std::set<FieldID> &fields,
                                   ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID> all_fields(fields.begin(), fields.end());
      if (redop > 0)
      {
        assert(all_fields.size() == 1);
        constraints.add_constraint(SpecializedConstraint(
                                    REDUCTION_FOLD_SPECIALIZE, redop)).
          add_constraint(FieldConstraint(all_fields, true/*contiguous*/,
                                         true/*inorder*/));
      }
      else
      {
        std::vector<FieldID> all_fields(fields.begin(), fields.end());
        std::vector<DimensionKind> dimension_ordering(4);
        dimension_ordering[0] = DIM_X;
        dimension_ordering[1] = DIM_Y;
        dimension_ordering[2] = DIM_Z;
        dimension_ordering[3] = DIM_F;
        constraints.add_constraint(SpecializedConstraint())
          .add_constraint(FieldConstraint(all_fields, false/*contiguous*/,
                                          false/*inorder*/))
          .add_constraint(OrderingConstraint(dimension_ordering, 
                                             false/*contigous*/));
      }
    }

  }; // namespace Mapping
}; // namespace Legion

