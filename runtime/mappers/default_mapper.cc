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

#include "legion.h"
#include "default_mapper.h"

#include <cstdlib>
#include <cassert>
#include <limits.h>
#include <algorithm>

#define STATIC_MAX_PERMITTED_STEALS   4
#define STATIC_MAX_STEAL_COUNT        2
#define STATIC_BREADTH_FIRST          false
#define STATIC_STEALING_ENABLED       false
#define STATIC_MAX_SCHEDULE_COUNT     8

// This is the default implementation of the mapper interface for 
// the general low level runtime

namespace Legion {
  namespace Mapping {

    using namespace Utilities;
    using namespace LegionRuntime::Arrays;

    LegionRuntime::Logger::Category log_mapper("default_mapper");

    //--------------------------------------------------------------------------
    /*static*/ const char* DefaultMapper::create_default_name(Processor p)
    //--------------------------------------------------------------------------
    {
      const size_t buffer_size = 64;
      char *result = (char*)malloc(buffer_size*sizeof(char));
      snprintf(result, buffer_size-1,
                "Default Mapper on Processor " IDFMT "", p.id);
      return result;
    }

    //--------------------------------------------------------------------------
    DefaultMapper::DefaultMapper(Machine m, Processor local, const char *name) 
      : Mapper(), local_proc(local), local_kind(local.kind()), 
        node_id(local.address_space()), machine(m),
        mapper_name((name == NULL) ? create_default_name(local) : strdup(name)),
        max_steals_per_theft(STATIC_MAX_PERMITTED_STEALS),
        max_steal_count(STATIC_MAX_STEAL_COUNT),
        breadth_first_traversal(STATIC_BREADTH_FIRST),
        stealing_enabled(STATIC_STEALING_ENABLED),
        max_schedule_count(STATIC_MAX_SCHEDULE_COUNT)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Initializing the default mapper for "
                            "processor " IDFMT "",
                 local_proc.id);
      // Check to see if there any input arguments to parse
      {
        int argc = HighLevelRuntime::get_input_args().argc;
        char **argv = HighLevelRuntime::get_input_args().argv;
        // Parse the input arguments looking for ones for the default mapper
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
          BOOL_ARG("-dm:steal", stealing_enabled);
          BOOL_ARG("-dm:bft", breadth_first_traversal);
          INT_ARG("-dm:sched", max_schedule_count);
#undef BOOL_ARG
#undef INT_ARG
        }
      }
      if (stealing_enabled)
      {
        log_mapper.warning("Default mapper does not have a stealing algorithm "
                           "implemented yet so we are ignoring the request "
                           "for enabling stealing at the moment.");
        stealing_enabled = false;
      }
      // Get all the processors and gpus on the local node
      std::set<Processor> all_procs;
      machine.get_all_processors(all_procs);
      for (std::set<Processor>::const_iterator it = all_procs.begin();
            it != all_procs.end(); it++)
      {
        AddressSpace node = it->address_space();
        if (node == node_id)
        {
          switch (it->kind())
          {
            case Processor::TOC_PROC:
              {
                local_gpus.push_back(*it);
                break;
              }
            case Processor::LOC_PROC:
              {
                local_cpus.push_back(*it);
                break;
              }
            case Processor::IO_PROC:
              {
                local_ios.push_back(*it);
                break;
              }
            default: // ignore anything else
              break;
          }
        }
        else
        {
          switch (it->kind())
          {
            case Processor::TOC_PROC:
              {
                // See if we already have a target GPU processor for this node
                if (node >= remote_gpus.size())
                  remote_gpus.resize(node+1, Processor::NO_PROC);
                if (!remote_gpus[node].exists())
                  remote_gpus[node] = *it;
                break;
              }
            case Processor::LOC_PROC:
              {
                // See if we already have a target CPU processor for this node
                if (node >= remote_cpus.size())
                  remote_cpus.resize(node+1, Processor::NO_PROC);
                if (!remote_cpus[node].exists())
                  remote_cpus[node] = *it;
                break;
              }
            case Processor::IO_PROC:
              {
                // See if we already have a target I/O processor for this node
                if (node >= remote_ios.size())
                  remote_ios.resize(node+1, Processor::NO_PROC);
                if (!remote_ios[node].exists())
                  remote_ios[node] = *it;
                break;
              }
            default: // ignore anything else
              break;
          }
        }
      }
      assert(!local_cpus.empty()); // better have some cpus
      // check to make sure we complete sets of ios, cpus, and gpus
      for (unsigned idx = 0; idx < remote_cpus.size(); idx++) {
        if (!remote_cpus[idx].exists()) { 
          log_mapper.error("Default mapper error: no CPUs detected on "
                           "node %d! There must be CPUs on all nodes "
                           "for the default mapper to function.", idx);
          assert(false);
        }
      }
      total_nodes = remote_cpus.size() + 1;
      if (!local_gpus.empty()) {
        for (unsigned idx = 0; idx < remote_gpus.size(); idx++) {
          if (!remote_gpus[idx].exists())
          {
            log_mapper.error("Default mapper has GPUs on node %d, but "
                             "could not detect GPUs on node %d. The "
                             "current default mapper implementation "
                             "assumes symmetric heterogeneity.", node_id, idx);
            assert(false);
          }
        }
      }
      if (!local_ios.empty()) {
        for (unsigned idx = 0; idx < remote_ios.size(); idx++) {
          if (!remote_ios[idx].exists()) {
            log_mapper.error("Default mapper has I/O procs on node %d, but "
                             "could not detect I/O procs on node %d. The "
                             "current default mapper implementation assumes "
                             "symmetric heterogeneity.", node_id, idx);
            assert(false);
          }
        }
      } 
      // Initialize our random number generator
      const size_t short_bits = 8*sizeof(unsigned short);
      long long short_mask = 0;
      for (unsigned i = 0; i < short_bits; i++)
        short_mask |= (1LL << i);
      for (int i = 0; i < 3; i++)
        random_number_generator[i] = (unsigned short)((local_proc.id & 
                            (short_mask << (i*short_bits))) >> (i*short_bits));
    }

    //--------------------------------------------------------------------------
    long DefaultMapper::default_generate_random_integer(void) const
    //--------------------------------------------------------------------------
    {
      return nrand48(random_number_generator);
    }
    
    //--------------------------------------------------------------------------
    double DefaultMapper::default_generate_random_real(void) const
    //--------------------------------------------------------------------------
    {
      return erand48(random_number_generator);
    }

    //--------------------------------------------------------------------------
    DefaultMapper::DefaultMapper(const DefaultMapper &rhs)
      : Mapper(), local_proc(Processor::NO_PROC),
        local_kind(Processor::LOC_PROC), node_id(0), 
        machine(rhs.machine), mapper_name(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DefaultMapper::~DefaultMapper(void)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Deleting default mapper for processor " IDFMT "",
                  local_proc.id);
      free(const_cast<char*>(mapper_name));
    }

    //--------------------------------------------------------------------------
    DefaultMapper& DefaultMapper::operator=(const DefaultMapper &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    const char* DefaultMapper::get_mapper_name(void) const
    //--------------------------------------------------------------------------
    {
      return mapper_name;
    }

    //--------------------------------------------------------------------------
    Mapper::MapperSyncModel DefaultMapper::get_mapper_sync_model(void) const
    //--------------------------------------------------------------------------
    {
      // Default mapper operates with the serialized re-entrant sync model
      return SERIALIZED_REENTRANT_MAPPER_MODEL;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_task_options(const MapperContext    ctx,
                                            const Task&            task,
                                                  TaskOptions&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_task_options in %s", get_mapper_name());
      output.initial_proc = default_policy_select_initial_processor(ctx, task);
      output.inline_task = false;
      output.stealable = stealing_enabled; 
      output.map_locally = true;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_policy_select_initial_processor(
                                            MapperContext ctx, const Task &task)
    //--------------------------------------------------------------------------
    {
      VariantInfo info = find_preferred_variant(task, ctx,false/*needs tight*/);
      // If we are the right kind then we return ourselves
      if (info.proc_kind == local_kind)
        return local_proc;
      // Otherwise pick a local one of the right type
      switch (info.proc_kind)
      {
        case Processor::LOC_PROC:
          {
            assert(!local_cpus.empty());
            return select_random_processor(local_cpus); 
          }
        case Processor::TOC_PROC:
          {
            assert(!local_gpus.empty());
            return select_random_processor(local_gpus);
          }
        case Processor::IO_PROC:
          {
            assert(!local_ios.empty());
            return select_random_processor(local_ios);
          }
        default:
          assert(false);
      }
      return Processor::NO_PROC;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::select_random_processor(
                                    const std::vector<Processor> &options) const
    //--------------------------------------------------------------------------
    {
      const size_t total_procs = options.size();
      const int index = default_generate_random_integer() % total_procs;
      return options[index];
    }

    //--------------------------------------------------------------------------
    DefaultMapper::VariantInfo DefaultMapper::find_preferred_variant(
                                     const Task &task, MapperContext ctx, 
                                     bool needs_tight_bound, bool cache_result,
                                     Processor::Kind specific)
    //--------------------------------------------------------------------------
    {
      // Do a quick test to see if we have cached the result
      std::map<TaskID,VariantInfo>::const_iterator finder = 
                                        preferred_variants.find(task.task_id);
      if (finder != preferred_variants.end() && 
          (!needs_tight_bound || finder->second.tight_bound))
        return finder->second;
      // Otherwise we actually need to pick one
      // Ask the runtime for the variant IDs for the given task type
      std::vector<VariantID> variants;
      mapper_rt_find_valid_variants(ctx, task.task_id, variants);
      if (!variants.empty())
      {
        Processor::Kind best_kind = Processor::NO_KIND;
        if (finder == preferred_variants.end() || 
            (specific != Processor::NO_KIND))
        {
          // Do the weak part first and figure out which processor kind
          // we want to focus on first
          std::vector<Processor::Kind> ranking;
          if (specific == Processor::NO_KIND)
            default_policy_rank_processor_kinds(ctx, task, ranking);
          else
            ranking.push_back(specific);
          assert(!ranking.empty());
          // Go through the kinds in the rankings
          for (unsigned idx = 0; idx < ranking.size(); idx++)
          {
            // See if we have any local processor of this kind
            switch (ranking[idx])
            {
              case Processor::TOC_PROC:
                {
                  if (local_gpus.empty())
                    continue;
                  break;
                }
              case Processor::LOC_PROC:
                {
                  if (local_cpus.empty())
                    continue;
                  break;
                }
              case Processor::IO_PROC:
                {
                  if (local_ios.empty())
                    continue;
                  break;
                }
              default:
                assert(false); // unknown processor type
            }
            // See if we have any variants of this kind
            mapper_rt_find_valid_variants(ctx, task.task_id, 
                                          variants, ranking[idx]);
            // If we have valid variants and we have processors we are
            // good to use this set of variants
            if (!ranking.empty())
            {
              best_kind = ranking[idx];
              break;
            }
          }
          // This is really bad if we didn't find any variants
          if (best_kind == Processor::NO_KIND)
          {
            log_mapper.error("Failed to find any valid variants for task %s "
                             "on the current machine. All variants for this "
                             "task are for processor kinds which are not "
                             "present on this machine.", task.get_task_name());
            assert(false);
          }
        }
        else
        {
          // We already know which kind to focus, so just get our 
          // variants for this processor kind
          best_kind = finder->second.proc_kind;
          mapper_rt_find_valid_variants(ctx, task.task_id, variants, best_kind);
        }
        assert(!variants.empty());
        VariantInfo result;
        result.proc_kind = best_kind;
        // We only need to do this second part if we need a tight bound
        if (needs_tight_bound)
        {
          if (variants.size() > 1)
          {
            // Iterate through the variants and pick the best one
            // for this task
            VariantID best_variant = variants[0];
            const ExecutionConstraintSet *best_execution_constraints = 
              &(mapper_rt_find_execution_constraints(ctx, 
                                            task.task_id, best_variant));
            const TaskLayoutConstraintSet *best_layout_constraints = 
              &(mapper_rt_find_task_layout_constraints(ctx, 
                                            task.task_id, best_variant));
            for (unsigned idx = 1; idx < variants.size(); idx++)
            {
              const ExecutionConstraintSet &next_execution_constraints = 
                mapper_rt_find_execution_constraints(ctx, 
                                            task.task_id, variants[idx]);
              const TaskLayoutConstraintSet &next_layout_constraints = 
                mapper_rt_find_task_layout_constraints(ctx, 
                                            task.task_id, variants[idx]);
              VariantID chosen = default_policy_select_best_variant(ctx,
                  task, best_kind, best_variant, variants[idx],
                  *best_execution_constraints, next_execution_constraints,
                  *best_layout_constraints, next_layout_constraints);
              assert((chosen == best_variant) || (chosen == variants[idx]));
              if (chosen != best_variant)
              {
                best_variant = variants[idx];
                best_execution_constraints = &next_execution_constraints;
                best_layout_constraints = &next_layout_constraints;
              }
            }
            result.variant = best_variant;
          }
          else
            result.variant = variants[0]; // only one choice
          result.tight_bound = true;
        }
        else
        {
          // Not tight, so just pick the first one
          result.variant = variants[0];
          // It is a tight bound if there is only one of them
          result.tight_bound = (variants.size() == 1);
        }
        // Save the result in the cache if we weren't asked for
        // a variant for a specific kind
        if (cache_result)
        {
          result.is_inner = mapper_rt_is_inner_variant(ctx, task.task_id,
                                                       result.variant);
          preferred_variants[task.task_id] = result;
        }
        return result;
      }
      // TODO: handle the presence of generators here
      log_mapper.error("Default mapper was unable to find any variants for "
                       "task %s. The application must register at least one "
                       "variant for all task kinds.", task.get_task_name());
      assert(false);
      return VariantInfo();
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::default_policy_rank_processor_kinds(MapperContext ctx,
                        const Task &task, std::vector<Processor::Kind> &ranking)
    //--------------------------------------------------------------------------
    {
      // Default mapper is ignorant about task IDs so just do whatever
      ranking.resize(3);
      // Prefer GPUs over everything else, teehee! :)
      ranking[0] = Processor::TOC_PROC;
      // I/O processors are specialized so prefer them next
      ranking[1] = Processor::IO_PROC;
      // CPUs go last (suck it Intel)
      ranking[2] = Processor::LOC_PROC;
    }

    //--------------------------------------------------------------------------
    VariantID DefaultMapper::default_policy_select_best_variant(
                                      MapperContext ctx, const Task &task, 
                                      Processor::Kind kind,
                                      VariantID vid1, VariantID vid2,
                                      const ExecutionConstraintSet &execution1,
                                      const ExecutionConstraintSet &execution2,
                                      const TaskLayoutConstraintSet &layout1,
                                      const TaskLayoutConstraintSet &layout2)
    //--------------------------------------------------------------------------
    {
      // TODO: better algorithm for picking the best variants on this machine
      // For now we do something really stupid, chose the larger variant
      // ID because if it was registered later is likely more specialized :)
      if (vid1 < vid2)
        return vid2;
      return vid1;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::premap_task(const MapperContext      ctx,
                                    const Task&              task, 
                                    const PremapTaskInput&   input,
                                          PremapTaskOutput&  output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default premap_task in %s", get_mapper_name());
      // Iterate over the premap regions
      bool has_variant_info = false;
      VariantInfo info;
      for (std::map<unsigned,std::vector<PhysicalInstance> >::const_iterator
            it = input.valid_instances.begin(); 
            it != input.valid_instances.end(); it++)
      {
        // If this region requirements is restricted, then we can just
        // copy over the instances because we know we have to use them
        if (task.regions[it->first].is_restricted())
        {
          output.premapped_instances.insert(*it);
          continue;
        }
        // These are non-restricted regions which means they have to be
        // shared by everyone in this task
        // TODO: some caching here
        if (total_nodes > 1) {
          // multi-node, see how big the index space is, if it is big
          // enough to span more than our node, put it in gasnet memory
          // otherwise we can fall through to the single node case
          Memory target_memory = Memory::NO_MEMORY;
          std::set<Memory> visible_memories;
          machine.get_visible_memories(task.target_proc, visible_memories);
          Memory global_memory = Memory::NO_MEMORY;
          for (std::set<Memory>::const_iterator vit = visible_memories.begin();
                vit != visible_memories.end(); vit++)
          {
            if (vit->kind() == Memory::GLOBAL_MEM)
            {
              global_memory = *vit;
              break;
            }
          }
          switch (task.target_proc.kind())
          {
            case Processor::IO_PROC:
              {
                if (task.index_domain.get_volume() > local_ios.size())
                {
                  if (!global_memory.exists())
                  {
                    log_mapper.error("Default mapper failure. No memory found "
                        "for I/O task %s (ID %lld) which is visible "
                        "for all points in the index space.",
                        task.get_task_name(), task.get_unique_id());
                    assert(false);
                  }
                  else
                    target_memory = global_memory;
                }
                break;
              }
            case Processor::LOC_PROC:
              {
                if (task.index_domain.get_volume() > local_cpus.size())
                {
                  if (!global_memory.exists())
                  {
                    log_mapper.error("Default mapper failure. No memory found "
                        "for CPU task %s (ID %lld) which is visible "
                        "for all point in the index space.",
                        task.get_task_name(), task.get_unique_id());
                    assert(false);
                  }
                  else
                    target_memory = global_memory;
                }
                break;
              }
            case Processor::TOC_PROC:
              {
                if (task.index_domain.get_volume() > local_gpus.size())
                {
                  log_mapper.error("Default mapper failure. No memory found "
                      "for GPU task %s (ID %lld) which is visible "
                      "for all points in the index space.",
                      task.get_task_name(), task.get_unique_id());
                  assert(false);
                }
                break;
              }
            default:
              assert(false); // unrecognized processor kind
          }
          if (target_memory.exists())
          {
            if (!has_variant_info)
            {
              info = find_preferred_variant(task, ctx, 
                  true/*needs tight bound*/, true/*cache*/,
                  task.target_proc.kind());
              has_variant_info = true;
            }
            // Map into the target memory and we are done
            std::set<FieldID> needed_fields = 
              task.regions[it->first].privilege_fields;
            const TaskLayoutConstraintSet &layout_constraints =
              mapper_rt_find_task_layout_constraints(ctx,
                                        task.task_id, info.variant);
            if (!default_create_custom_instances(ctx, task.target_proc,
                  target_memory, task.regions[it->first], it->first,
                  needed_fields, layout_constraints, true/*needs check*/,
                  output.premapped_instances[it->first]))
            {
              default_report_failed_instance_creation(task, it->first, 
                                              task.target_proc, target_memory);
            }
            continue;
          }
        }
        // should be local to a node
        // see where we are mapping  
        Memory target_memory = Memory::NO_MEMORY;
        std::set<Memory> visible_memories;
        machine.get_visible_memories(task.target_proc, visible_memories);
        switch (task.target_proc.kind())
        {
          case Processor::IO_PROC:
          case Processor::LOC_PROC:
            {
              // Put these regions in system memory      
              for (std::set<Memory>::const_iterator it = 
                   visible_memories.begin(); it != visible_memories.end(); it++)
              {
                if (it->kind() == Memory::SYSTEM_MEM)
                {
                  target_memory = *it;
                  break;
                }
              }
              if (!target_memory.exists())
              {
                log_mapper.error("Default mapper error. No memory found for "
                    "CPU task %s (ID %lld) which is visible for all points "
                    "in the index space.", task.get_task_name(), 
                    task.get_unique_id());
                assert(false);
              }
              break;
            }
          case Processor::TOC_PROC:
            {
              // Otherwise for GPUs put the instance in zero-copy memory
              for (std::set<Memory>::const_iterator it = 
                   visible_memories.begin(); it != visible_memories.end(); it++)
              {
                if (it->kind() == Memory::Z_COPY_MEM)
                {
                  target_memory = *it;
                  break;
                }
              }
              if (!target_memory.exists())
              {
                log_mapper.error("Default mapper error. No memory found for "
                    "GPU task %s (ID %lld) which is visible for all points "
                    "in the index space.", task.get_task_name(),
                    task.get_unique_id());
                assert(false);
              }
              break;
            }
          default:
            assert(false); // unknown processor kind
        }
        assert(target_memory.exists());
        if (!has_variant_info)
        {
          info = find_preferred_variant(task, ctx, 
              true/*needs tight bound*/, true/*cache*/,
              task.target_proc.kind());
          has_variant_info = true;
        }
        // Map into the target memory and we are done
        std::set<FieldID> needed_fields = 
          task.regions[it->first].privilege_fields;
        const TaskLayoutConstraintSet &layout_constraints =
          mapper_rt_find_task_layout_constraints(ctx,
                                    task.task_id, info.variant);
        if (!default_create_custom_instances(ctx, task.target_proc,
              target_memory, task.regions[it->first], it->first,
              needed_fields, layout_constraints, true/*needs check*/,
              output.premapped_instances[it->first]))
        {
          default_report_failed_instance_creation(task, it->first, 
                                          task.target_proc, target_memory);
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::slice_task(const MapperContext      ctx,
                                   const Task&              task, 
                                   const SliceTaskInput&    input,
                                         SliceTaskOutput&   output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default slice_task in %s", get_mapper_name());
      // Whatever kind of processor we are is the one this task should
      // be scheduled on as determined by select initial task
      switch (local_kind)
      {
        case Processor::LOC_PROC:
          {
            default_slice_task(task, local_cpus, remote_cpus, 
                               input, output, cpu_slices_cache);
            break;
          }
        case Processor::TOC_PROC:
          {
            default_slice_task(task, local_gpus, remote_gpus, 
                               input, output, gpu_slices_cache);
            break;
          }
        case Processor::IO_PROC:
          {
            default_slice_task(task, local_ios, remote_ios, 
                               input, output, io_slices_cache);
            break;
          }
        default:
          assert(false); // unimplemented processor kind
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::default_slice_task(const Task &task,
                                           const std::vector<Processor> &local,
                                           const std::vector<Processor> &remote,
                                           const SliceTaskInput& input,
                                                 SliceTaskOutput &output,
                  std::map<Domain,std::vector<TaskSlice> > &cached_slices) const
    //--------------------------------------------------------------------------
    {
      // Before we do anything else, see if it is in the cache
      std::map<Domain,std::vector<TaskSlice> >::const_iterator finder = 
        cached_slices.find(input.domain);
      if (finder != cached_slices.end()) {
        output.slices = finder->second;
        return;
      }
      // Figure out how many points are in this index space task
      const size_t total_points = input.domain.get_volume();
      // Do two-level slicing, first slice into slices that fit on a
      // node and then slice across the processors of the right kind
      // on the local node. If we only have one node though, just break
      // into chunks that evenly divide among processors.
      switch (input.domain.get_dim())
      {
        case 1:
          {
            Rect<1> point_rect = input.domain.get_rect<1>();
            if (remote.size() > 1) {
              if (total_points <= local.size()) {
                Point<1> blocking_factor(total_points/*splitting factor*/);
                default_decompose_points<1>(point_rect, local, 
                    blocking_factor, false/*recurse*/, 
                    stealing_enabled, output.slices);
              } else {
                Point<1> blocking_factor(local.size());
                default_decompose_points<1>(point_rect, remote,
                    blocking_factor, true/*recurse*/, 
                    stealing_enabled, output.slices);
              }
            } else {
              Point<1> blocking_factor(total_points/local.size());
              default_decompose_points<1>(point_rect, local,
                  blocking_factor, false/*recurse*/,
                  stealing_enabled, output.slices);
            }
            break;
          }
        case 2:
          {
            Rect<2> point_rect = input.domain.get_rect<2>();
            if (remote.size() > 1) {
              if (total_points <= local.size()) {
                Point<2> blocking_factor = 
                  default_select_blocking_factor<2>(total_points, point_rect);
                default_decompose_points<2>(point_rect, local, 
                    blocking_factor, false/*recurse*/, 
                    stealing_enabled, output.slices);
              } else {
                Point<2> blocking_factor = 
                  default_select_blocking_factor<2>(local.size(), point_rect);
                default_decompose_points<2>(point_rect, remote,
                    blocking_factor, true/*recurse*/, 
                    stealing_enabled, output.slices);
              }
            } else {
              Point<2> blocking_factor = default_select_blocking_factor<2>(
                  total_points/local.size(), point_rect);
              default_decompose_points<2>(point_rect, local,
                  blocking_factor, false/*recurse*/,
                  stealing_enabled, output.slices);
            }
            break;
          }
        case 3:
          {
            Rect<3> point_rect = input.domain.get_rect<3>();
            if (remote.size() > 1) {
              if (total_points <= local.size()) {
                Point<3> blocking_factor = 
                  default_select_blocking_factor<3>(total_points, point_rect);
                default_decompose_points<3>(point_rect, local, 
                    blocking_factor, false/*recurse*/, 
                    stealing_enabled, output.slices);
              } else {
                Point<3> blocking_factor = 
                  default_select_blocking_factor<3>(local.size(), point_rect);
                default_decompose_points<3>(point_rect, remote,
                    blocking_factor, true/*recurse*/, 
                    stealing_enabled, output.slices);
              }
            } else {
              Point<3> blocking_factor = default_select_blocking_factor<3>(
                  total_points/local.size(), point_rect);
              default_decompose_points<3>(point_rect, local,
                  blocking_factor, false/*recurse*/,
                  stealing_enabled, output.slices);
            }
            break;
          }
        default: // don't support other dimensions right now
          assert(false);
      }
      // Save the result in the cache
      cached_slices[input.domain] = output.slices;
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    /*static*/ void DefaultMapper::default_decompose_points(
                                         const Rect<DIM> &point_rect,
                                         const std::vector<Processor> &targets,
                                         const Point<DIM> &blocking_factor,
                                         bool recurse, bool stealable,
                                         std::vector<TaskSlice> &slices)
    //--------------------------------------------------------------------------
    {
      Blockify<DIM> blocking(blocking_factor); 
      unsigned next_index = 0;
      bool is_perfect = true;
      for (int idx = 0; idx < DIM; idx++) {
        if ((point_rect.dim_size(idx) % blocking_factor[idx]) != 0) {
          is_perfect = false;
          break;
        }
      }
      // We need to check to see if this point rectangle is base at the origin
      // because the blockify operation depends on it
      Point<DIM> origin;
      for (int i = 0; i < DIM; i++)
        origin.x[i] = 0;
      if (origin == point_rect.lo)
      {
        // Simple case, rectangle is based at the origin
        Rect<DIM> blocks = blocking.image_convex(point_rect);
        if (is_perfect)
        {
          slices.resize(blocks.volume());
          for (typename Blockify<DIM>::PointInOutputRectIterator 
                pir(blocks); pir; pir++, next_index++)
          {
            Rect<DIM> slice_points = blocking.preimage(pir.p);
            TaskSlice &slice = slices[next_index];
            slice.domain = Domain::from_rect<DIM>(slice_points);
            slice.proc = targets[next_index % targets.size()];
            slice.recurse = recurse;
            slice.stealable = stealable;
          }
        }
        else
        {
          slices.reserve(blocks.volume());
          for (typename Blockify<DIM>::PointInOutputRectIterator 
                pir(blocks); pir; pir++)
          {
            Rect<DIM> upper_bound = blocking.preimage(pir.p);
            // Check for edge cases with intersections
            Rect<DIM> slice_points = upper_bound.intersection(point_rect);
            if (slice_points.volume() == 0)
              continue;
            slices.resize(next_index+1);
            TaskSlice &slice = slices[next_index];
            slice.domain = Domain::from_rect<DIM>(slice_points);
            slice.proc = targets[next_index % targets.size()];
            slice.recurse = recurse;
            slice.stealable = stealable;
            next_index++;
          }
        }
      }
      else
      {
        // Rectangle is not based at the origin so we have to 
        // translate the point rectangle there, do the blocking, 
        // and then translate back
        const Point<DIM> &translation = point_rect.lo;
        Rect<DIM> translated_rect = point_rect - translation;
        Rect<DIM> blocks = blocking.image_convex(translated_rect);
        if (is_perfect)
        {
          slices.resize(blocks.volume());
          for (typename Blockify<DIM>::PointInOutputRectIterator 
                pir(blocks); pir; pir++, next_index++)
          {
            Rect<DIM> slice_points = blocking.preimage(pir.p) + translation;
            TaskSlice &slice = slices[next_index];
            slice.domain = Domain::from_rect<DIM>(slice_points);
            slice.proc = targets[next_index % targets.size()];
            slice.recurse = recurse;
            slice.stealable = stealable;
          }
        }
        else
        {
          slices.reserve(blocks.volume());
          for (typename Blockify<DIM>::PointInOutputRectIterator 
                pir(blocks); pir; pir++)
          {
            Rect<DIM> upper_bound = blocking.preimage(pir.p) + translation;
            // Check for edge cases with intersections
            Rect<DIM> slice_points = upper_bound.intersection(point_rect);
            if (slice_points.volume() == 0)
              continue;
            slices.resize(next_index+1);
            TaskSlice &slice = slices[next_index];
            slice.domain = Domain::from_rect<DIM>(slice_points);
            slice.proc = targets[next_index % targets.size()];
            slice.recurse = recurse;
            slice.stealable = stealable;
            next_index++;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    /*static*/ Point<DIM> DefaultMapper::default_select_blocking_factor( 
                                         int factor, const Rect<DIM> &to_factor)
    //--------------------------------------------------------------------------
    {
      if (factor == 1)
      {
        int result[DIM];
        for (int i = 0; i < DIM; i++)
          result[i] = 1;
        return Point<DIM>(result);
      }
      // Fundamental theorem of arithmetic time!
      const unsigned num_primes = 32;
      const int primes[num_primes] = { 2, 3, 5, 7, 11, 13, 17, 19, 
                                       23, 29, 31, 37, 41, 43, 47, 53,
                                       59, 61, 67, 71, 73, 79, 83, 89,
                                       97, 101, 103, 107, 109, 113, 127, 131 };
      // Increase the size of the prime number table if you ever hit this
      assert(factor <= (primes[num_primes-1] * primes[num_primes-1]));
      // Factor into primes
      std::vector<int> prime_factors;
      for (unsigned idx = 0; idx < num_primes; idx++)
      {
        const int prime = primes[idx];
        if ((prime * prime) > factor)
          break;
        while ((factor % prime) == 0)
        {
          prime_factors.push_back(prime);
          factor /= prime;
        }
        if (factor == 1)
          break;
      }
      if (factor > 1)
        prime_factors.push_back(factor);
      // Assign prime factors onto the dimensions for the target rect
      // but don't ever exceed the size of a given dimension, do this from the
      // largest primes down to the smallest to give ourselves as much 
      // flexibility as possible to get as fine a partitioning as possible
      // for maximum parallelism
      int result[DIM];
      for (int i = 0; i < DIM; i++)
        result[i] = 1;
      int exhausted_dims = 0;
      int dim_chunks[DIM];
      for (int i = 0; i < DIM; i++)
      {
        dim_chunks[i] = to_factor.dim_size(i);
        if (dim_chunks[i] <= 1)
          exhausted_dims++;
      }
      for (int idx = prime_factors.size()-1; idx >= 0; idx--)
      {
        // Find the dimension with the biggest dim_chunk 
        int next_dim = -1;
        int max_chunk = -1;
        for (int i = 0; i < DIM; i++)
        {
          if (dim_chunks[i] > max_chunk)
          {
            max_chunk = dim_chunks[i];
            next_dim = i;
          }
        }
        const int next_prime = prime_factors[idx];
        // If this dimension still has chunks at least this big
        // then we can divide it by this factor
        if (max_chunk >= next_prime)
        {
          result[next_dim] *= next_prime;
          dim_chunks[next_dim] /= next_prime;
          if (dim_chunks[next_dim] <= 1)
          {
            exhausted_dims++;
            // If we've exhausted all our dims, we are done
            if (exhausted_dims == DIM)
              break;
          }
        }
      }
      return Point<DIM>(result);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_task(const MapperContext      ctx,
                                 const Task&              task,
                                 const MapTaskInput&      input,
                                       MapTaskOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_task in %s", get_mapper_name());
      Processor::Kind target_kind = task.target_proc.kind();
      // Get the variant that we are going to use to map this task
      VariantInfo chosen = find_preferred_variant(task, ctx,
                        true/*needs tight bound*/, true/*cache*/, target_kind);
      output.chosen_variant = chosen.variant;
      // TODO: some criticality analysis to assign priorities
      output.task_priority = 0;
      output.postmap_task = false;
      // Figure out our target processors
      if (task.target_proc.address_space() == node_id)
      {
        switch (task.target_proc.kind())
        {
          case Processor::TOC_PROC:
            {
              // GPUs have their own memories so they only get one
              output.target_procs.push_back(task.target_proc);
              break;
            }
          case Processor::LOC_PROC:
            {
              // Put any of our local cpus on here
              // TODO: NUMA-ness needs to go here
              // If we're part of a must epoch launch, our 
              // target proc will be sufficient
              if (!task.must_epoch_task)
                output.target_procs.insert(output.target_procs.end(),
                    local_cpus.begin(), local_cpus.end());
              else
                output.target_procs.push_back(task.target_proc);
              break;
            }
          case Processor::IO_PROC:
            {
              // Put any of our I/O procs here
              // If we're part of a must epoch launch, our
              // target proc will be sufficient
              if (!task.must_epoch_task)
                output.target_procs.insert(output.target_procs.end(),
                    local_ios.begin(), local_ios.end());
              else
                output.target_procs.push_back(task.target_proc);
              break;
            }
          default:
            assert(false); // unrecognized processor kind
        }
      }
      else
        output.target_procs.push_back(task.target_proc);
      // See if we have an inner variant, if we do virtually map all the regions
      // We don't even both caching these since they are so simple
      if (chosen.is_inner)
      {
        for (unsigned idx = 0; idx < task.regions.size(); idx++)
          output.chosen_instances[idx].push_back(
              PhysicalInstance::get_virtual_instance());
        return;
      }
      // First, let's see if we've cached a result of this task mapping
      const unsigned long long task_hash = compute_task_hash(task);
      std::pair<TaskID,Processor> cache_key(task.task_id, task.target_proc);
      std::map<std::pair<TaskID,Processor>,
               std::list<CachedTaskMapping> >::const_iterator 
        finder = cached_task_mappings.find(cache_key);
      bool needs_field_constraint_check = false;
      Memory target_memory = default_policy_select_target_memory(ctx, 
                                                         task.target_proc);
      if (finder != cached_task_mappings.end())
      {
        bool found = false;
        bool has_reductions = false;
        // Iterate through and see if we can find one with our variant and hash
        for (std::list<CachedTaskMapping>::const_iterator it = 
              finder->second.begin(); it != finder->second.end(); it++)
        {
          if ((it->variant == output.chosen_variant) &&
              (it->task_hash == task_hash))
          {
            // Have to copy it before we do the external call which 
            // might invalidate our iterator
            output.chosen_instances = it->mapping;
            has_reductions = it->has_reductions;
            found = true;
            break;
          }
        }
        if (found)
        {
          // If we have reductions, make those instances now since we
          // never cache the reduction instances
          if (has_reductions)
          {
            const TaskLayoutConstraintSet &layout_constraints =
              mapper_rt_find_task_layout_constraints(ctx,
                                  task.task_id, output.chosen_variant);
            for (unsigned idx = 0; idx < task.regions.size(); idx++)
            {
              if (task.regions[idx].privilege == REDUCE)
              {
                std::set<FieldID> copy = task.regions[idx].privilege_fields;
                if (!default_create_custom_instances(ctx, task.target_proc,
                    target_memory, task.regions[idx], idx, copy, 
                    layout_constraints, needs_field_constraint_check, 
                    output.chosen_instances[idx]))
                {
                  default_report_failed_instance_creation(task, idx, 
                                              task.target_proc, target_memory);
                }
              }
            }
          }
          // See if we can acquire these instances still
          if (mapper_rt_acquire_and_filter_instances(ctx, 
                                                     output.chosen_instances))
            return;
          // We need to check the constraints here because we had a
          // prior mapping and it failed, which may be the result
          // of a change in the allocated fields of a field space
          needs_field_constraint_check = true;
          // If some of them were deleted, go back and remove this entry
          // Have to renew our iterators since they might have been
          // invalidated during the 'acquire_and_filter_instances' call
          default_remove_cached_task(ctx, output.chosen_variant,
                        task_hash, cache_key, output.chosen_instances);
        }
      }
      // We didn't find a cached version of the mapping so we need to 
      // do a full mapping, we already know what variant we want to use
      // so let's use one of the acceleration functions to figure out
      // which instances still need to be mapped.
      std::vector<std::set<FieldID> > missing_fields(task.regions.size());
      mapper_rt_filter_instances(ctx, task, output.chosen_variant,
                                 output.chosen_instances, missing_fields);
      // Track which regions have already been mapped 
      std::vector<bool> done_regions(task.regions.size(), false);
      if (!input.premapped_regions.empty())
        for (std::vector<unsigned>::const_iterator it = 
              input.premapped_regions.begin(); it != 
              input.premapped_regions.end(); it++)
          done_regions[*it] = true;
      const TaskLayoutConstraintSet &layout_constraints = 
        mapper_rt_find_task_layout_constraints(ctx, 
                              task.task_id, output.chosen_variant);
      // Now we need to go through and make instances for any of our
      // regions which do not have space for certain fields
      bool has_reductions = false;
      for (unsigned idx = 0; idx < task.regions.size(); idx++)
      {
        if (done_regions[idx])
          continue;
        // Skip any empty regions
        if ((task.regions[idx].privilege == NO_ACCESS) ||
            (task.regions[idx].privilege_fields.empty()) ||
            missing_fields[idx].empty())
          continue;
        // See if this is a reduction      
        if (task.regions[idx].privilege == REDUCE)
        {
          has_reductions = true;
          if (!default_create_custom_instances(ctx, task.target_proc,
                  target_memory, task.regions[idx], idx, missing_fields[idx],
                  layout_constraints, needs_field_constraint_check,
                  output.chosen_instances[idx]))
          {
            default_report_failed_instance_creation(task, idx, 
                                        task.target_proc, target_memory);
          }
          continue;
        }
        // Otherwise make normal instances for the given region
        if (!default_create_custom_instances(ctx, task.target_proc,
                target_memory, task.regions[idx], idx, missing_fields[idx],
                layout_constraints, needs_field_constraint_check,
                output.chosen_instances[idx]))
        {
          default_report_failed_instance_creation(task, idx,
                                      task.target_proc, target_memory);
        }
      }
      // Now that we are done, let's cache the result so we can use it later
      std::list<CachedTaskMapping> &map_list = cached_task_mappings[cache_key];
      map_list.push_back(CachedTaskMapping());
      CachedTaskMapping &cached_result = map_list.back();
      cached_result.task_hash = task_hash; 
      cached_result.variant = output.chosen_variant;
      cached_result.mapping = output.chosen_instances;
      cached_result.has_reductions = has_reductions;
      // We don't ever save reduction instances in our cache 
      if (has_reductions) {
        for (unsigned idx = 0; idx < task.regions.size(); idx++) {
          if (task.regions[idx].privilege != REDUCE)
            continue;
          cached_result.mapping[idx].clear();
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::default_remove_cached_task(MapperContext ctx,
        VariantID chosen_variant, unsigned long long task_hash,
        const std::pair<TaskID,Processor> &cache_key,
        const std::vector<std::vector<PhysicalInstance> > &post_filter)
    //--------------------------------------------------------------------------
    {
      std::map<std::pair<TaskID,Processor>,
               std::list<CachedTaskMapping> >::iterator
                 finder = cached_task_mappings.find(cache_key);
      if (finder != cached_task_mappings.end())
      {
        // Keep a list of instances for which we need to downgrade
        // their garbage collection priorities since we are no
        // longer caching the results
        std::deque<PhysicalInstance> to_downgrade;
        for (std::list<CachedTaskMapping>::iterator it = 
              finder->second.begin(); it != finder->second.end(); it++)
        {
          if ((it->variant == chosen_variant) &&
              (it->task_hash == task_hash))
          {
            // Record all the instances for which we will need to
            // down grade their garbage collection priority 
            for (unsigned idx1 = 0; (idx1 < it->mapping.size()) &&
                  (idx1 < post_filter.size()); idx1++)
            {
              if (!it->mapping[idx1].empty())
              {
                if (!post_filter[idx1].empty()) {
                  // Still all the same
                  if (post_filter[idx1].size() == it->mapping[idx1].size())
                    continue;
                  // See which ones are no longer in our set
                  for (unsigned idx2 = 0; 
                        idx2 < it->mapping[idx1].size(); idx2++)
                  {
                    PhysicalInstance current = it->mapping[idx1][idx2];
                    bool still_valid = false;
                    for (unsigned idx3 = 0; 
                          idx3 < post_filter[idx1].size(); idx3++)
                    {
                      if (current == post_filter[idx1][idx3]) 
                      {
                        still_valid = true;
                        break;
                      }
                    }
                    if (!still_valid)
                      to_downgrade.push_back(current);
                  }
                } else {
                  // if the chosen instances are empty, record them all
                  to_downgrade.insert(to_downgrade.end(),
                      it->mapping[idx1].begin(), it->mapping[idx1].end());
                }
              }
            }
            finder->second.erase(it);
            break;
          }
        }
        if (finder->second.empty())
          cached_task_mappings.erase(finder);
        if (!to_downgrade.empty())
        {
          for (std::deque<PhysicalInstance>::const_iterator it =
                to_downgrade.begin(); it != to_downgrade.end(); it++)
            mapper_rt_set_garbage_collection_priority(ctx, *it, 0/*priority*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ unsigned long long DefaultMapper::compute_task_hash(
                                                               const Task &task)
    //--------------------------------------------------------------------------
    {
      // Use Sean's "cheesy" hash function    
      const unsigned long long c1 = 0x5491C27F12DB3FA4; // big number, mix 1+0s
      const unsigned long long c2 = 353435096; // chosen by fair dice roll
      // We have to hash all region requirements including region names,
      // privileges, coherence modes, reduction operators, and fields
      unsigned long long result = c2 + task.task_id;
      for (unsigned idx = 0; idx < task.regions.size(); idx++)
      {
        const RegionRequirement &req = task.regions[idx];
        result = result * c1 + c2 + req.handle_type;
        if (req.handle_type != PART_PROJECTION) {
          result = result * c1 + c2 + req.region.get_tree_id();
          result = result * c1 + c2 + req.region.get_index_space().get_id();
          result = result * c1 + c2 + req.region.get_field_space().get_id();
        } else {
          result = result * c1 + c2 + req.partition.get_tree_id();
          result = result * c1 + c2 + 
                                  req.partition.get_index_partition().get_id();
          result = result * c1 + c2 + req.partition.get_field_space().get_id();
        }
        for (std::set<FieldID>::const_iterator it = 
              req.privilege_fields.begin(); it != 
              req.privilege_fields.end(); it++)
          result = result * c1 + c2 + *it;
        result = result * c1 + c2 + req.privilege;
        result = result * c1 + c2 + req.prop;
        result = result * c1 + c2 + req.redop;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::default_create_custom_instances(MapperContext ctx,
                          Processor target_proc, Memory target_memory,
                          const RegionRequirement &req, unsigned index, 
                          std::set<FieldID> &needed_fields,
                          const TaskLayoutConstraintSet &layout_constraints,
                          bool needs_field_constraint_check,
                          std::vector<PhysicalInstance> &instances)
    //--------------------------------------------------------------------------
    {
      // Before we do anything else figure out our 
      // constraints for any instances of this task, then we'll
      // see if these constraints conflict with or are satisfied by
      // any of the other constraints 
      bool force_new_instances = false;
      LayoutConstraintID our_layout_id = 
       default_policy_select_layout_constraints(ctx, target_memory, req,
                           needs_field_constraint_check, force_new_instances);
      const LayoutConstraintSet &our_constraints = 
                        mapper_rt_find_layout_constraints(ctx, our_layout_id);
      for (std::multimap<unsigned,LayoutConstraintID>::const_iterator lay_it =
            layout_constraints.layouts.lower_bound(index); lay_it !=
            layout_constraints.layouts.upper_bound(index); lay_it++)
      {
        // Get the constraints
        const LayoutConstraintSet &index_constraints = 
                        mapper_rt_find_layout_constraints(ctx, lay_it->second);
        std::vector<FieldID> overlaping_fields;
        const std::vector<FieldID> &constraint_fields = 
          index_constraints.field_constraint.get_field_set();
        for (unsigned idx = 0; idx < constraint_fields.size(); idx++)
        {
          FieldID fid = constraint_fields[idx];
          std::set<FieldID>::iterator finder = needed_fields.find(fid);
          if (finder != needed_fields.end())
          {
            overlaping_fields.push_back(fid);
            // Remove from the needed fields since we're going to handle it
            needed_fields.erase(finder);
          }
        }
        // If we don't have any overlapping fields, then keep going
        if (overlaping_fields.empty())
          continue;
        // Now figure out how to make an instance
        instances.resize(instances.size()+1);
        // Check to see if these constraints conflict with our constraints
        if (mapper_rt_do_constraints_conflict(ctx, 
                                              our_layout_id, lay_it->second))
        {
          // They conflict, so we're just going to make an instance
          // using these constraints
          if (!default_make_instance(ctx, target_memory, index_constraints,
                     instances.back(), TASK_MAPPING, force_new_instances, 
                     false/*meets*/, false/*reduction*/, req, target_proc))
            return false;
        }
        else if (mapper_rt_do_constraints_entail(ctx, 
                                                 lay_it->second, our_layout_id))
        {
          // These constraints do everything we want to do and maybe more
          // so we can just use them directly
          if (!default_make_instance(ctx, target_memory, index_constraints,
                      instances.back(), TASK_MAPPING, force_new_instances, 
                      true/*meets*/, false/*reduction*/, req, target_proc))
            return false;
        }
        else
        {
          // These constraints don't do as much as we want but don't
          // conflict so make an instance with them and our constraints 
          LayoutConstraintSet creation_constraints = index_constraints;
          default_policy_fill_constraints(ctx, creation_constraints, 
                                          target_memory, req);
          creation_constraints.add_constraint(
              FieldConstraint(overlaping_fields,
                false/*contig*/, false/*inorder*/));
          if (!default_make_instance(ctx, target_memory, creation_constraints,
                         instances.back(), TASK_MAPPING, force_new_instances, 
                         true/*meets*/, false/*reduction*/, req, target_proc))
            return false;
        }
      }
      // If we don't have anymore needed fields, we are done
      if (needed_fields.empty())
        return true;
      // There are no constraints for these fields so we get to do what we want
      instances.resize(instances.size()+1);
      LayoutConstraintSet creation_constraints = our_constraints;
      creation_constraints.add_constraint(
          FieldConstraint(needed_fields, false/*contig*/, false/*inorder*/));
      if (!default_make_instance(ctx, target_memory, creation_constraints, 
                instances.back(), TASK_MAPPING, force_new_instances, 
                true/*meets*/,  false/*reduction*/, req, target_proc))
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    Memory DefaultMapper::default_policy_select_target_memory(MapperContext ctx,
                                                          Processor target_proc)
    //--------------------------------------------------------------------------
    {
      // Find the visible memories from the processor for the given kind
      std::set<Memory> visible_memories;
      machine.get_visible_memories(target_proc, visible_memories);
      if (visible_memories.empty())
      {
        log_mapper.error("No visible memories from processor " IDFMT "! "
                         "This machine is really messed up!", target_proc.id);
        assert(false);
      }
      // Figure out the memory with the highest-bandwidth
      Memory chosen = Memory::NO_MEMORY;
      unsigned best_bandwidth = 0;
      std::vector<Machine::ProcessorMemoryAffinity> affinity(1);
      for (std::set<Memory>::const_iterator it = visible_memories.begin();
            it != visible_memories.end(); it++)
      {
        affinity.clear();
        machine.get_proc_mem_affinity(affinity, target_proc, *it);
        assert(affinity.size() == 1);
        if (!chosen.exists() || (affinity[0].bandwidth > best_bandwidth)) {
          chosen = *it;
          best_bandwidth = affinity[0].bandwidth;
        }
      }
      assert(chosen.exists());
      return chosen;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintID DefaultMapper::default_policy_select_layout_constraints(
                                    MapperContext ctx, Memory target_memory, 
                                    const RegionRequirement &req,
                                    bool needs_field_constraint_check,
                                    bool &force_new_instances)
    //--------------------------------------------------------------------------
    {
      // Do something special for reductions
      if (req.privilege == REDUCE)
      {
        // Always make new reduction instances
        force_new_instances = true;
        std::pair<Memory::Kind,ReductionOpID> constraint_key(
            target_memory.kind(), req.redop);
        std::map<std::pair<Memory::Kind,ReductionOpID>,LayoutConstraintID>::
          const_iterator finder = reduction_constraint_cache.find(
                                                            constraint_key);
        // No need to worry about field constraint checks here
        // since we don't actually have any field constraints
        if (finder != reduction_constraint_cache.end())
          return finder->second;
        LayoutConstraintSet constraints;
        default_policy_fill_constraints(ctx, constraints, target_memory, req);
        LayoutConstraintID result = mapper_rt_register_layout(ctx, constraints);
        // Save the result
        reduction_constraint_cache[constraint_key] = result;
        return result;
      }
      // We always set force_new_instances to false since we are
      // deciding to optimize for minimizing memory usage instead
      // of avoiding Write-After-Read (WAR) dependences
      force_new_instances = false;
      // See if we've already made a constraint set for this layout
      std::pair<Memory::Kind,FieldSpace> constraint_key(target_memory.kind(),
                                               req.region.get_field_space());
      std::map<std::pair<Memory::Kind,FieldSpace>,LayoutConstraintID>::
        const_iterator finder = layout_constraint_cache.find(constraint_key);
      if (finder != layout_constraint_cache.end())
      {
        // If we don't need a constraint check we are already good
        if (!needs_field_constraint_check)
          return finder->second;
        // Check that the fields still are the same, if not, fall through
        // so that we make a new set of constraints
        const LayoutConstraintSet &old_constraints =
                      mapper_rt_find_layout_constraints(ctx, finder->second);
        // Should be only one unless things have changed
        const std::vector<FieldID> &old_set = 
                          old_constraints.field_constraint.get_field_set();
        // Check to make sure the field sets are still the same
        std::vector<FieldID> new_fields;
        mapper_rt_get_field_space_fields(ctx, constraint_key.second,new_fields);
        if (new_fields.size() == old_set.size())
        {
          std::set<FieldID> old_fields(old_set.begin(), old_set.end());
          bool still_equal = true;
          for (unsigned idx = 0; idx < new_fields.size(); idx++)
          {
            if (old_fields.find(new_fields[idx]) == old_fields.end())
            {
              still_equal = false; 
              break;
            }
          }
          if (still_equal)
            return finder->second;
        }
        // Otherwise we fall through and make a new constraint which
        // will also update the cache
      }
      // Fill in the constraints 
      LayoutConstraintSet constraints;
      default_policy_fill_constraints(ctx, constraints, target_memory, req);
      // Do the registration
      LayoutConstraintID result = mapper_rt_register_layout(ctx, constraints);
      // Record our results, there is a benign race here as another mapper
      // call could have registered the exact same registration constraints
      // here if we were preempted during the registration call. The 
      // constraint sets are identical though so it's all good.
      layout_constraint_cache[constraint_key] = result;
      return result; 
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::default_policy_fill_constraints(MapperContext ctx,
                   LayoutConstraintSet &constraints, Memory target_memory,
                   const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      // See if we are doing a reduction instance
      if (req.privilege == REDUCE)
      {
        // Make reduction fold instances
        constraints.add_constraint(SpecializedConstraint(
              SpecializedConstraint::REDUCTION_FOLD_SPECIALIZE)) 
          .add_constraint(MemoryConstraint(target_memory.kind()));
      }
      else
      {
        // Normal instance creation
        FieldSpace handle = req.region.get_field_space();
        std::vector<FieldID> all_fields;
        mapper_rt_get_field_space_fields(ctx, handle, all_fields);
        std::vector<DimensionKind> dimension_ordering(4);
        dimension_ordering[0] = DIM_X;
        dimension_ordering[1] = DIM_Y;
        dimension_ordering[2] = DIM_Z;
        dimension_ordering[3] = DIM_F;
        // Our base default mapper will try to make instances of containing
        // all fields (in any order) laid out in SOA format to encourage 
        // maximum re-use by any tasks which use subsets of the fields
        constraints.add_constraint(SpecializedConstraint())
          .add_constraint(MemoryConstraint(target_memory.kind()))
          .add_constraint(FieldConstraint(all_fields, false/*contiguous*/,
                                          false/*inorder*/))
          .add_constraint(OrderingConstraint(dimension_ordering, 
                                             false/*contigous*/));
      }
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::default_make_instance(MapperContext ctx, 
        Memory target_memory, const LayoutConstraintSet &constraints,
        PhysicalInstance &result, MappingKind kind, bool force_new, bool meets, 
        bool reduction, const RegionRequirement &req, Processor target_proc)
    //--------------------------------------------------------------------------
    {
      bool created = true;
      LogicalRegion target_region = 
        default_policy_select_instance_region(ctx, target_memory, req,
                                    constraints, force_new, meets, reduction);
      // TODO: deal with task layout constraints that require multiple
      // region requirements to be mapped to the same instance
      std::vector<LogicalRegion> target_regions(1, target_region);
      if (force_new) {
        if (!mapper_rt_create_physical_instance(ctx, target_memory, constraints,
                                                target_regions, result))
          return false;
      } else {
        if (!mapper_rt_find_or_create_physical_instance(ctx, target_memory, 
                                constraints, target_regions, result, created))
          return false;
      }
      if (created)
      {
        int priority = default_policy_select_garbage_collection_priority(ctx, 
                         kind, target_memory, result, meets, reduction);
        if (priority != 0)
          mapper_rt_set_garbage_collection_priority(ctx, result, priority);
      }
      return true;
    }

    //--------------------------------------------------------------------------
    LogicalRegion DefaultMapper::default_policy_select_instance_region(
                                MapperContext ctx, Memory target_memory,
                                const RegionRequirement &req,
                                const LayoutConstraintSet &layout_constraints,
                                bool force_new_instances, 
                                bool meets_constraints, bool reduction)
    //--------------------------------------------------------------------------
    {
      // If it is not something we are making a big region for just
      // return the region that is actually needed
      LogicalRegion result = req.region; 
      if (!meets_constraints || reduction)
        return result;
      // Simple heuristic here, if we are on a single node, we go all the
      // way to the root since the first-level partition is likely just
      // across processors in the node, however, if we are on multiple nodes
      // we go to the region under the first-level partition since the 
      // first partition is normally the one across all nodes
      if (total_nodes == 1)
      {
        while (mapper_rt_has_parent_logical_partition(ctx, result))
        {
          LogicalPartition parent = 
            mapper_rt_get_parent_logical_partition(ctx, result);
          result = mapper_rt_get_parent_logical_region(ctx, parent);
        }
        return result;
      }
      else
      {
        // Find the region one-level down 
        // (unless the application actually asked for the root region)
        if (!mapper_rt_has_parent_logical_partition(ctx, result))
          return result;
        LogicalPartition parent = 
          mapper_rt_get_parent_logical_partition(ctx, result);
        LogicalRegion next = mapper_rt_get_parent_logical_region(ctx, parent);
        while (mapper_rt_has_parent_logical_partition(ctx, next))
        {
          result = next;
          parent = mapper_rt_get_parent_logical_partition(ctx, next);
          next = mapper_rt_get_parent_logical_region(ctx, parent); 
        }
        return result;
      }
    }

    //--------------------------------------------------------------------------
    int DefaultMapper::default_policy_select_garbage_collection_priority(
                                MapperContext ctx, MappingKind kind,
                                Memory memory, const PhysicalInstance &inst,
                                bool meets_fill_constraints, bool reduction)
    //--------------------------------------------------------------------------
    {
      // Pretty simple: keep our big instances around
      // as long as possible, delete reduction instances
      // as soon as possible, otherwise we are ambivalent
      // Only have higher priority for things we cache
      if (meets_fill_constraints && (kind == TASK_MAPPING))
        return INT_MAX;
      if (reduction)
        return INT_MIN;
      return 0;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::default_report_failed_instance_creation(
                                 const Task &task, unsigned index, 
                                 Processor target_proc, Memory target_mem) const
    //--------------------------------------------------------------------------
    {
      log_mapper.error("Default mapper failed allocation for region "
                       "requirement %d of task %s (UID %lld) in memory " IDFMT
                       "for processor " IDFMT ". This means the working set "
                       "of your application is too big for the allotted "
                       "capacity of the given memory under the default "
                       "mapper's mapping scheme. You have three choices: "
                       "ask Realm to allocate more memory, write a custom "
                       "mapper to better manage working sets, or find a bigger "
                       "machine. Good luck!", index, task.get_task_name(),
                       task.get_unique_id(), target_proc.id, target_mem.id);
      assert(false);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_task_variant(const MapperContext          ctx,
                                            const Task&                  task,
                                            const SelectVariantInput&    input,
                                                  SelectVariantOutput&   output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_task_variant in %s", get_mapper_name());
      VariantInfo result = find_preferred_variant(task, ctx,
                                  true/*needs tight bound*/, false/*cache*/,
                                  local_kind/*need our kind specifically*/);
      output.chosen_variant = result.variant;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::postmap_task(const MapperContext      ctx,
                                     const Task&              task,
                                     const PostMapInput&      input,
                                           PostMapOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default postmap_task in %s", get_mapper_name());
      // TODO: teach the default mapper about resilience
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_task_sources(const MapperContext        ctx,
                                            const Task&                task,
                                            const SelectTaskSrcInput&  input,
                                                  SelectTaskSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_task_sources in %s", get_mapper_name());
      default_policy_select_sources(ctx, input.target, input.source_instances,
                                    output.chosen_ranking);
    } 

    //--------------------------------------------------------------------------
    void DefaultMapper::default_policy_select_sources(MapperContext ctx,
                                   const PhysicalInstance &target, 
                                   const std::vector<PhysicalInstance> &sources,
                                   std::deque<PhysicalInstance> &ranking)
    //--------------------------------------------------------------------------
    {
      // For right now we'll rank instances by the bandwidth of the memory
      // they are in to the destination 
      // TODO: consider layouts when ranking source  to help out the DMA system
      std::map<Memory,unsigned/*bandwidth*/> source_memories;
      Memory destination_memory = target.get_location();
      std::vector<MemoryMemoryAffinity> affinity(1);
      // fill in a vector of the sources with their bandwidths and sort them
      std::vector<std::pair<PhysicalInstance,
                          unsigned/*bandwidth*/> > band_ranking(sources.size());
      for (unsigned idx = 0; idx < sources.size(); idx++)
      {
        const PhysicalInstance &instance = sources[idx];
        Memory location = instance.get_location();
        std::map<Memory,unsigned>::const_iterator finder = 
          source_memories.find(location);
        if (finder == source_memories.end())
        {
          affinity.clear();
          machine.get_mem_mem_affinity(affinity, destination_memory, location);
          unsigned memory_bandwidth = 0;
          if (affinity.empty()) {
            // TODO: More graceful way of dealing with multi-hop copies
            log_mapper.warning("WARNING: Default mapper is potentially "
                               "requesting a multi-hop copy between memories "
                               IDFMT " and " IDFMT "!", location.id,
                               destination_memory.id);
          } else {
            assert(affinity.size() == 1);
            memory_bandwidth = affinity[0].bandwidth;
          }
          source_memories[location] = memory_bandwidth;
          band_ranking[idx] = 
            std::pair<PhysicalInstance,unsigned>(instance, memory_bandwidth);
        }
        else
          band_ranking[idx] = 
            std::pair<PhysicalInstance,unsigned>(instance, finder->second);
      }
      // Sort them by bandwidth
      std::sort(band_ranking.begin(), band_ranking.end(), physical_sort_func);
      // Iterate from largest bandwidth to smallest
      for (std::vector<std::pair<PhysicalInstance,unsigned> >::
            const_reverse_iterator it = band_ranking.rbegin(); 
            it != band_ranking.rend(); it++)
        ranking.push_back(it->first);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::speculate(const MapperContext      ctx,
                                  const Task&              task,
                                        SpeculativeOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default speculate for Task in %s", get_mapper_name());
      // Default mapper doesn't speculate
      output.speculate = false;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::report_profiling(const MapperContext      ctx,
                                         const Task&              task,
                                         const TaskProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default report_profiling for Task in %s", 
                      get_mapper_name());
      // We don't ask for any task profiling right now so assert if we see this
      assert(false);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_inline(const MapperContext        ctx,
                                   const InlineMapping&       inline_op,
                                   const MapInlineInput&      input,
                                         MapInlineOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_inline in %s", get_mapper_name());
      // Copy over all the valid instances, then try to do an acquire on them
      // and see which instances are no longer valid
      output.chosen_instances = input.valid_instances;
      mapper_rt_acquire_and_filter_instances(ctx, output.chosen_instances);
      // Now see if we have any fields which we still make space for
      std::set<FieldID> missing_fields = inline_op.requirement.privilege_fields;
      for (std::vector<PhysicalInstance>::const_iterator it = 
            output.chosen_instances.begin(); it != 
            output.chosen_instances.end(); it++)
      {
        it->remove_space_fields(missing_fields);
        if (missing_fields.empty())
          break;
      }
      // If we've satisfied all our fields, then we are done
      if (missing_fields.empty())
        return;
      // Otherwise, let's make an instance for our missing fields
      Memory target_memory = default_policy_select_target_memory(ctx,
                                      inline_op.parent_task->current_proc);
      bool force_new_instances = false;
      LayoutConstraintID our_layout_id = 
       default_policy_select_layout_constraints(ctx, target_memory, 
                                             inline_op.requirement, 
                                             true/*needs check*/, 
                                             force_new_instances);
      LayoutConstraintSet creation_constraints = 
                        mapper_rt_find_layout_constraints(ctx, our_layout_id);
      creation_constraints.add_constraint(
          FieldConstraint(missing_fields, false/*contig*/, false/*inorder*/));
      output.chosen_instances.resize(output.chosen_instances.size()+1);
      if (!default_make_instance(ctx, target_memory, creation_constraints,
            output.chosen_instances.back(), INLINE_MAPPING, 
            force_new_instances, true/*meets*/, false/*reduction*/, 
            inline_op.requirement, inline_op.parent_task->current_proc))
      {
        // If we failed to make it that is bad
        log_mapper.error("Default mapper failed allocation for region "
                         "requirement of inline mapping in task %s (UID %lld) "
                         "in memory " IDFMT "for processor " IDFMT ". This "
                         "means the working set of your application is too big "
                         "for the allotted capacity of the given memory under "
                         "the default mapper's mapping scheme. You have three "
                         "choices: ask Realm to allocate more memory, write a "
                         "custom mapper to better manage working sets, or find "
                         "a bigger machine. Good luck!", 
                         inline_op.parent_task->get_task_name(),
                         inline_op.parent_task->get_unique_id(),
                         inline_op.parent_task->current_proc.id, 
                         target_memory.id);
        assert(false);
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_inline_sources(const MapperContext     ctx,
                                         const InlineMapping&         inline_op,
                                         const SelectInlineSrcInput&  input,
                                               SelectInlineSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_inline_sources in %s", get_mapper_name());
      default_policy_select_sources(ctx, input.target, input.source_instances,
                                    output.chosen_ranking);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::report_profiling(const MapperContext         ctx,
                                         const InlineMapping&        inline_op,
                                         const InlineProfilingInfo&  input)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default report_profiling for Inline in %s", 
                      get_mapper_name());
      // We don't ask for any task profiling right now so assert if we see this
      assert(false);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_copy(const MapperContext      ctx,
                                 const Copy&              copy,
                                 const MapCopyInput&      input,
                                       MapCopyOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_copy in %s", get_mapper_name());
      // For the sources always use an existing instances and virtual
      // instances for the rest, for the destinations, hope they are
      // restricted, otherwise we really don't know what to do
      for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
      {
        output.src_instances[idx] = input.src_instances[idx];
        // Stick this on for good measure, at worst it will be ignored
        output.src_instances[idx].push_back(
            PhysicalInstance::get_virtual_instance());
      }
      for (unsigned idx = 0; idx < copy.dst_requirements.size(); idx++)
      {
        output.dst_instances[idx] = input.dst_instances[idx];
        if (!copy.dst_requirements[idx].is_restricted())
        {
          // If this is not restricted, see if we have all the fields covered
          std::set<FieldID> missing_fields = 
            copy.dst_requirements[idx].privilege_fields;
          for (std::vector<PhysicalInstance>::const_iterator it = 
                output.dst_instances[idx].begin(); it !=
                output.dst_instances[idx].end(); it++)
          {
            it->remove_space_fields(missing_fields);
            if (missing_fields.empty())
              break;
          }
          // If we still have fields, we need to make an instance
          // We clearly need to take a guess, let's see if we can find
          // one of our instances to use.
          if (!missing_fields.empty())
          {
            std::set<Memory> visible_mems;
            machine.get_visible_memories(local_proc, visible_mems);
            for (std::set<Memory>::const_iterator it = visible_mems.begin(); 
                  it != visible_mems.end(); it++)
            {
              Memory target_memory = (*it); 
              LayoutConstraintSet constraint_set;
              constraint_set.add_constraint(SpecializedConstraint())
                .add_constraint(MemoryConstraint(target_memory.kind()))
                .add_constraint(FieldConstraint(missing_fields, 
                        false/*contiguous*/, false/*inorder*/));
              std::vector<LogicalRegion> target_regions(1, 
                                    copy.dst_requirements[idx].region);
              PhysicalInstance result;
              if (mapper_rt_find_physical_instance(ctx, target_memory,
                                                   constraint_set,
                                                   target_regions,
                                                   result))
              {
                output.dst_instances[idx].push_back(result);
                result.remove_space_fields(missing_fields);
                if (missing_fields.empty())
                  break;
              }
            }
            if (!missing_fields.empty())
            {
              log_mapper.error("Default mapper error. No idea where to place "
                              "destination regions for explicit copy operation "
                              "in task %s (ID %lld) because the region is not "
                              "constricted and we couldn't find any instances "
                              "to use locally.", 
                              copy.parent_task->get_task_name(),
                              copy.parent_task->get_unique_id());
              assert(false);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_copy_sources(const MapperContext          ctx,
                                            const Copy&                  copy,
                                            const SelectCopySrcInput&    input,
                                                  SelectCopySrcOutput&   output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_copy_sources in %s", get_mapper_name());
      default_policy_select_sources(ctx, input.target, input.source_instances,
                                    output.chosen_ranking);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::speculate(const MapperContext      ctx,
                                  const Copy& copy,
                                        SpeculativeOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default speculate for Copy in %s", get_mapper_name());
      // Default mapper doesn't speculate
      output.speculate = false;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::report_profiling(const MapperContext      ctx,
                                         const Copy&              copy,
                                         const CopyProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default report_profiling for Copy in %s", 
                      get_mapper_name());
      // We don't ask for any task profiling right now so assert if we see this
      assert(false);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_close(const MapperContext       ctx,
                                  const Close&              close,
                                  const MapCloseInput&      input,
                                        MapCloseOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_close in %s", get_mapper_name());
      // Simple heuristic for closes, if we have an instance use it,
      // otherwise just make a virtual instance
      output.chosen_instances = input.valid_instances;
      output.chosen_instances.push_back(
                                PhysicalInstance::get_virtual_instance());
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_close_sources(const MapperContext        ctx,
                                             const Close&               close,
                                             const SelectCloseSrcInput&  input,
                                                   SelectCloseSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_close_sources in %s", get_mapper_name());
      default_policy_select_sources(ctx, input.target, input.source_instances,
                                    output.chosen_ranking);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::report_profiling(const MapperContext       ctx,
                                         const Close&              close,
                                         const CloseProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default report_profiling for Close in %s", 
                      get_mapper_name());
      // We don't ask for any task profiling right now so assert if we see this
      assert(false);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_acquire(const MapperContext         ctx,
                                    const Acquire&              acquire,
                                    const MapAcquireInput&      input,
                                          MapAcquireOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_acquire in %s", get_mapper_name());
      // Nothing to do here for now until we start using profiling
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::speculate(const MapperContext         ctx,
                                  const Acquire&              acquire,
                                        SpeculativeOutput&    output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default speculate for Acquire in %s", get_mapper_name());
      // Default mapper doesn't speculate
      output.speculate = false;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::report_profiling(const MapperContext         ctx,
                                         const Acquire&              acquire,
                                         const AcquireProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default report_profiling for Acquire in %s", 
                      get_mapper_name());
      // We don't ask for any task profiling right now so assert if we see this
      assert(false);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_release(const MapperContext         ctx,
                                    const Release&              release,
                                    const MapReleaseInput&      input,
                                          MapReleaseOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_release in %s", get_mapper_name());
      // Nothing to do here for now until we start using profiling
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_release_sources(const MapperContext      ctx,
                                         const Release&                 release,
                                         const SelectReleaseSrcInput&   input,
                                               SelectReleaseSrcOutput&  output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_release_sources in %s",get_mapper_name());
      default_policy_select_sources(ctx, input.target, input.source_instances,
                                    output.chosen_ranking);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::speculate(const MapperContext         ctx,
                                  const Release&              release,
                                        SpeculativeOutput&    output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default speculate for Release in %s", get_mapper_name());
      // Default mapper doesn't speculate
      output.speculate = false;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::report_profiling(const MapperContext         ctx,
                                         const Release&              release,
                                         const ReleaseProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default report_profiling for Release in %s", 
                      get_mapper_name());
      // We don't ask for any task profiling right now so assert if we see this
      assert(false);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::configure_context(const MapperContext         ctx,
                                          const Task&                 task,
                                                ContextConfigOutput&  output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default configure_context in %s", get_mapper_name());
      // Use the defaults here
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_tunable_value(const MapperContext         ctx,
                                             const Task&                 task,
                                             const SelectTunableInput&   input,
                                                   SelectTunableOutput&  output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_tunable_value in %s", get_mapper_name());
      size_t *result = (size_t*)malloc(sizeof(size_t));
      output.value = result;
      output.size = sizeof(size_t);
      switch (input.tunable_id)
      {
        case DEFAULT_TUNABLE_NODE_COUNT:
          {
            *result = total_nodes;
            break;
          }
        case DEFAULT_TUNABLE_LOCAL_CPUS:
          {
            *result = local_cpus.size();
            break;
          }
        case DEFAULT_TUNABLE_LOCAL_GPUS:
          {
            *result = local_gpus.size();
            break;
          }
        case DEFAULT_TUNABLE_LOCAL_IOS:
          {
            *result = local_ios.size();
            break;
          }
        case DEFAULT_TUNABLE_GLOBAL_CPUS:
          {
            // TODO: deal with machine asymmetry here
            *result = (local_cpus.size() * total_nodes);
            break;
          }
        case DEFAULT_TUNABLE_GLOBAL_GPUS:
          {
            // TODO: deal with machine asymmetry here
            *result = (local_gpus.size() * total_nodes);
            break;
          }
        case DEFAULT_TUNABLE_GLOBAL_IOS:
          {
            // TODO: deal with machine asymmetry here
            *result = (local_ios.size() * total_nodes);
            break;
          }
        default:
          {
            log_mapper.error("Default mapper error. Unrecognized tunable ID %d "
                             "requested in task %s (ID %lld).", 
                             input.tunable_id, task.get_task_name(),
                             task.get_unique_id());
            assert(false);
          }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_must_epoch(const MapperContext           ctx,
                                       const MapMustEpochInput&      input,
                                             MapMustEpochOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_must_epoch in %s", get_mapper_name());
      // Figure out how to assign tasks to CPUs first. We know we can't
      // do must epochs for anthing but CPUs at the moment.
      if (total_nodes > 1)
      {
        std::set<Processor> all_procs;
        machine.get_all_processors(all_procs);
        std::set<Processor>::const_iterator proc_finder = all_procs.begin();
        for (unsigned idx = 0; idx < input.tasks.size(); idx++)
        {
          // Find the next CPU
          while ((proc_finder != all_procs.end()) && 
              (proc_finder->kind() != Processor::LOC_PROC))
            proc_finder++;
          if (proc_finder == all_procs.end())
          {
            log_mapper.error("Default mapper error. Not enough CPUs for must "
                             "epoch launch of task %s with %ld tasks", 
                             input.tasks[0]->get_task_name(),
                             input.tasks.size());
            assert(false);
          }
          output.task_processors[idx] = *proc_finder;
        }
      }
      else
      {
        if (input.tasks.size() > local_cpus.size())
        {
          log_mapper.error("Default mapper error. Not enough CPUs for must "
                           "epoch launch of task %s with %ld tasks", 
                           input.tasks[0]->get_task_name(),
                           input.tasks.size());
          assert(false);
        }
        for (unsigned idx = 0; idx < input.tasks.size(); idx++)
          output.task_processors[idx] = local_cpus[idx];
      }
      // Now let's map all the constraints first, and then we'll call map
      // task for all the tasks and tell it that we already premapped the
      // constrainted instances
      for (unsigned cid = 0; cid < input.constraints.size(); cid++)
      {
        const MappingConstraint &constraint = input.constraints[cid];
        std::vector<PhysicalInstance> &constraint_mapping = 
                                              output.constraint_mappings[cid];
        int index1 = -1, index2 = -1;
        for (unsigned idx = 0; (idx < input.tasks.size()) &&
              ((index1 == -1) || (index2 == -1)); idx++)
        {
          if (constraint.t1 == input.tasks[idx])
            index1 = idx;
          if (constraint.t2 == input.tasks[idx])
            index2 = idx;
        }
        assert((index1 >= 0) && (index2 >= 0));
        // Figure out which memory to use
        // TODO: figure out how to use registered memory in the multi-node case
        Memory target1 = default_policy_select_target_memory(ctx,
                                              output.task_processors[index1]);
        Memory target2 = default_policy_select_target_memory(ctx,
                                              output.task_processors[index2]);
        // Pick our target memory
        Memory target_memory = Memory::NO_MEMORY;
        if (target1 != target2)
        {
          // See if one of them is not no access so we can pick the other
          if (constraint.t1->regions[constraint.idx1].is_no_access())
            target_memory = target2;
          else if (constraint.t2->regions[constraint.idx2].is_no_access())
            target_memory = target1;
          else
          {
            log_mapper.error("Default mapper error. Unable to pick a common "
                             "memory for tasks %s (ID %lld) and %s (ID %lld) "
                             "in a must epoch launch. This will require a "
                             "custom mapper.", constraint.t1->get_task_name(),
                             constraint.t1->get_unique_id(), 
                             constraint.t2->get_task_name(), 
                             constraint.t2->get_unique_id());
            assert(false);
          }
        }
        else // both the same so this is easy
          target_memory = target1;
        assert(target_memory.exists());
        // Figure out the variants that are going to be used by the two tasks    
        VariantInfo info1 = find_preferred_variant(*constraint.t1, ctx,
                              true/*needs tight bound*/, Processor::LOC_PROC);
        VariantInfo info2 = find_preferred_variant(*constraint.t2, ctx,
                              true/*needs tight_bound*/, Processor::LOC_PROC);
        // Map it the one way and filter the other so that we can make sure
        // that they are both going to use the same instance
        std::set<FieldID> needed_fields = 
          constraint.t1->regions[constraint.idx1].privilege_fields;
        needed_fields.insert(
            constraint.t2->regions[constraint.idx2].privilege_fields.begin(),
            constraint.t2->regions[constraint.idx2].privilege_fields.end());
        const TaskLayoutConstraintSet &layout_constraints1 = 
          mapper_rt_find_task_layout_constraints(ctx, 
                                      constraint.t1->task_id, info1.variant);
        if (!default_create_custom_instances(ctx, 
              output.task_processors[index1], target_memory,
              constraint.t1->regions[constraint.idx1], constraint.idx1,
              needed_fields, layout_constraints1, true/*needs check*/,
              constraint_mapping))
        {
          log_mapper.error("Default mapper error. Unable to make instance(s) "
                           "in memory " IDFMT " for index %d of constrained "
                           "task %s (ID %lld) in must epoch launch.",
                           target_memory.id, constraint.idx1, 
                           constraint.t1->get_task_name(),
                           constraint.t1->get_unique_id());
          assert(false);
        }
        // Copy the results over and make sure they are still good 
        const size_t num_instances = constraint_mapping.size();
        assert(num_instances > 0);
        std::set<FieldID> missing_fields;
        mapper_rt_filter_instances(ctx, *constraint.t2, constraint.idx2,
                     info2.variant, constraint_mapping, missing_fields);
        if (num_instances != constraint_mapping.size())
        {
          log_mapper.error("Default mapper error. Unable to make instance(s) "
                           "for index %d of constrained task %s (ID %lld) in "
                           "must epoch launch. Most likely this is because "
                           "conflicting constraints are requested for regions "
                           "which must be mapped to the same instance. You "
                           "will need to write a custom mapper to fix this.",
                           constraint.idx2, constraint.t2->get_task_name(),
                           constraint.t2->get_unique_id());
          assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_dataflow_graph(const MapperContext           ctx,
                                           const MapDataflowGraphInput&  input,
                                                 MapDataflowGraphOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_dataflow_graph in %s", get_mapper_name());
      // TODO: Implement this
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_tasks_to_map(const MapperContext          ctx,
                                            const SelectMappingInput&    input,
                                                  SelectMappingOutput&   output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_tasks_to_map in %s", get_mapper_name());
      if (breadth_first_traversal)
      {
        unsigned count = 0;
        for (std::list<const Task*>::const_iterator it = 
              input.ready_tasks.begin(); (count < max_schedule_count) && 
              (it != input.ready_tasks.end()); it++)
        {
          output.map_tasks.insert(*it);
          count++;
        }
      }
      else
      {
        // Find the depth of the deepest task
        int max_depth = 0;
        for (std::list<const Task*>::const_iterator it = 
              input.ready_tasks.begin(); it != input.ready_tasks.end(); it++)
        {
          int depth = (*it)->get_depth();
          if (depth > max_depth)
            max_depth = depth;
        }
        unsigned count = 0;
        // Only schedule tasks from the max depth in any pass
        for (std::list<const Task*>::const_iterator it = 
              input.ready_tasks.begin(); (count < max_schedule_count) && 
              (it != input.ready_tasks.end()); it++)
        {
          if ((*it)->get_depth() == max_depth)
          {
            output.map_tasks.insert(*it);
            count++;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_steal_targets(const MapperContext         ctx,
                                             const SelectStealingInput&  input,
                                                   SelectStealingOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_steal_targets in %s", get_mapper_name());
      // TODO: implement a work-stealing algorithm here 
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::permit_steal_request(const MapperContext         ctx,
                                             const StealRequestInput&    intput,
                                                   StealRequestOutput&   output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default permit_steal_request in %s", get_mapper_name());
      // TODO: implement a work stealing algorithm here
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::handle_message(const MapperContext           ctx,
                                       const MapperMessage&          message)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default handle_message in %s", get_mapper_name());
      // We don't send messages in the default mapper so assert if see this
      assert(false);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::handle_task_result(const MapperContext           ctx,
                                           const MapperTaskResult&       result)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default handle task result in %s", get_mapper_name());
      // We don't launch tasks in the default mapper so assert if we see this
      assert(false);
    }

  }; // namespace Mapping
}; // namespace Legion

