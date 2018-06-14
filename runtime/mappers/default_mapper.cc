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
#include "mappers/default_mapper.h"

#include <stdlib.h>
#include <assert.h>
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

    Logger log_mapper("default_mapper");

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
    DefaultMapper::DefaultMapper(MapperRuntime *rt, Machine m, 
                                 Processor local, const char *name) 
      : Mapper(rt), local_proc(local), local_kind(local.kind()), 
        node_id(local.address_space()), machine(m),
        mapper_name((name == NULL) ? create_default_name(local) : strdup(name)),
        next_local_gpu(0), next_local_cpu(0), next_local_io(0),
        next_local_procset(0), next_local_omp(0), next_local_py(0),
        next_global_gpu(Processor::NO_PROC),
        next_global_cpu(Processor::NO_PROC), next_global_io(Processor::NO_PROC),
        next_global_procset(Processor::NO_PROC),
        next_global_omp(Processor::NO_PROC), next_global_py(Processor::NO_PROC),
        global_gpu_query(NULL), global_cpu_query(NULL), global_io_query(NULL),
        global_procset_query(NULL), global_omp_query(NULL),
        global_py_query(NULL),
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
      Machine::ProcessorQuery all_procs(machine);
      for (Machine::ProcessorQuery::iterator it = all_procs.begin();
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
            case Processor::PY_PROC:
              {
                local_pys.push_back(*it);
                break;
              }
            case Processor::PROC_SET:
              {
                local_procsets.push_back(*it);
                break;
              }
            case Processor::OMP_PROC:
              {
                local_omps.push_back(*it);
                break;
              }
            default: // ignore anything else
              break;
          }
        }
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
          case Processor::PY_PROC:
            {
              // See if we already have a target I/O processor for this node
              if (node >= remote_pys.size())
                remote_pys.resize(node+1, Processor::NO_PROC);
              if (!remote_pys[node].exists())
                remote_pys[node] = *it;
              break;
            }
          case Processor::PROC_SET:
            {
              // See if we already have a target processor set for this node
              if (node >= remote_procsets.size())
                remote_procsets.resize(node+1, Processor::NO_PROC);
              if (!remote_procsets[node].exists())
                remote_procsets[node] = *it;
              break;
            }
          case Processor::OMP_PROC:
            {
              // See if we already have a target OMP processor for this node
              if (node >= remote_omps.size())
                remote_omps.resize(node+1, Processor::NO_PROC);
              if (!remote_omps[node].exists())
                remote_omps[node] = *it;
              break;
            }
          default: // ignore anything else
            break;
        }
      }
      // check to make sure we complete sets of ios, cpus, and gpus
      for (unsigned idx = 0; idx < remote_cpus.size(); idx++) {
	if (idx == node_id) continue;  // ignore our own node
        if (!remote_cpus[idx].exists()) { 
          log_mapper.error("Default mapper error: no CPUs detected on "
                           "node %d! There must be CPUs on all nodes "
                           "for the default mapper to function.", idx);
          assert(false);
        }
      }
      total_nodes = remote_cpus.size();
      if (!local_gpus.empty()) {
        for (unsigned idx = 0; idx < remote_gpus.size(); idx++) {
	  if (idx == node_id) continue;  // ignore our own node
          if (!remote_gpus[idx].exists())
          {
            log_mapper.error("Default mapper has GPUs on node %d, but "
                             "could not detect GPUs on node %d. The "
                             "current default mapper implementation "
                             "assumes symmetric heterogeneity.", node_id, idx);
            assert(false);
          }
        }
        if (total_nodes == 0) total_nodes = remote_gpus.size();
      }
      if (!local_ios.empty()) {
        for (unsigned idx = 0; idx < remote_ios.size(); idx++) {
	  if (idx == node_id) continue;  // ignore our own node
          if (!remote_ios[idx].exists()) {
            log_mapper.error("Default mapper has I/O procs on node %d, but "
                             "could not detect I/O procs on node %d. The "
                             "current default mapper implementation assumes "
                             "symmetric heterogeneity.", node_id, idx);
            assert(false);
          }
        }
        if (total_nodes == 0) total_nodes = remote_ios.size();
      }
      if (!local_omps.empty()) {
        for (unsigned idx = 0; idx < remote_omps.size(); idx++) {
	  if (idx == node_id) continue;  // ignore our own node
          if (!remote_omps[idx].exists()) {
            log_mapper.error("Default mapper has OMP procs on node %d, but "
                             "could not detect OMP procs on node %d. The "
                             "current default mapper implementation assumes "
                             "symmetric heterogeneity.", node_id, idx);
            assert(false);
          }
        }
        if (total_nodes == 0) total_nodes = remote_omps.size();
      } 
      if (!local_pys.empty()) {
        for (unsigned idx = 0; idx < remote_pys.size(); idx++) {
	  if (idx == node_id) continue;  // ignore our own node
          if (!remote_pys[idx].exists()) {
            log_mapper.error("Default mapper has Python procs on node %d, but "
                             "could not detect Python procs on node %d. The "
                             "current default mapper implementation assumes "
                             "symmetric heterogeneity.", node_id, idx);
            assert(false);
          }
        }
        if (total_nodes == 0) total_nodes = remote_pys.size();
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
      : Mapper(rhs.runtime), local_proc(Processor::NO_PROC),
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
      // This is the best choice for the default mapper assuming
      // there is locality in the remote mapped tasks
      output.map_locally = false;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_policy_select_initial_processor(
                                            MapperContext ctx, const Task &task)
    //--------------------------------------------------------------------------
    {
      if (have_proc_kind_variant(ctx, task.task_id, Processor::PROC_SET)) {
        return default_get_next_global_procset();
      }
 
      VariantInfo info = 
        default_find_preferred_variant(task, ctx, false/*needs tight*/);
      // If we are the right kind and this is an index space task launch
      // then we return ourselves
      if (task.is_index_space)
      {
        if (info.proc_kind == local_kind)
          return local_proc;
        // Otherwise round robin onto our local queue of the right kind
        switch (info.proc_kind)
        {
          case Processor::LOC_PROC:
            return default_get_next_local_cpu();
          case Processor::TOC_PROC:
            return default_get_next_local_gpu();
          case Processor::IO_PROC:
            return default_get_next_local_io();
          case Processor::OMP_PROC:
            return default_get_next_local_omp();
          case Processor::PY_PROC:
            return default_get_next_local_py();
          default: // make warnings go away
            break;
        }
        // should never get here
        assert(false);
        return Processor::NO_PROC;
      }
      // Do different things depending on our depth in the task tree
      const int depth = task.get_depth();
      switch (depth)
      {
        case 0:
          {
            // Top-level task: try to stay in place, otherwise choose
            // a suitable local processor.
            if (info.proc_kind == local_kind)
              return local_proc;
            switch (info.proc_kind)
            {
              case Processor::LOC_PROC:
                return default_get_next_local_cpu();
              case Processor::TOC_PROC:
                return default_get_next_local_gpu();
              case Processor::IO_PROC:
                return default_get_next_local_io();
              case Processor::OMP_PROC:
                return default_get_next_local_omp();
              case Processor::PY_PROC:
                return default_get_next_local_py();
              default: // make warnings go away
                break;
            }
          }
        case 1:
          {
            // First-level tasks: assume we should distribute these
            // evenly around the machine unless we've been explicitly 
	    // told not to
            // TODO: Fix this when we implement a good stealing algorithm
            // to instead encourage locality
	    if ((task.tag & SAME_ADDRESS_SPACE) == 0)
            {
	      switch (info.proc_kind)
              {
                case Processor::LOC_PROC:
                  return default_get_next_global_cpu();
                case Processor::TOC_PROC:
                  return default_get_next_global_gpu();
                case Processor::IO_PROC: // Don't distribute I/O
                  return default_get_next_local_io();
                case Processor::OMP_PROC:
                  return default_get_next_global_omp();
                case Processor::PY_PROC:
                  return default_get_next_global_py();
                default: // make warnings go away
                  break;
              }
            }
            // fall through to local assignment code below
          }
        default:
          {
            // N-level tasks: assume we keep these local to our
            // current node as the distribution was done at level 1
            switch (info.proc_kind)
            {
              case Processor::LOC_PROC:
                return default_get_next_local_cpu();
              case Processor::TOC_PROC:
                return default_get_next_local_gpu();
              case Processor::IO_PROC:
                return default_get_next_local_io();
              case Processor::OMP_PROC:
                return default_get_next_local_omp();
              case Processor::PY_PROC:
                return default_get_next_local_py();
              default: // make warnings go away
                break;
            }
          }
      }
      // should never get here
      assert(false);
      return Processor::NO_PROC;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_select_random_processor(
                                    const std::vector<Processor> &options) const
    //--------------------------------------------------------------------------
    {
      const size_t total_procs = options.size();
      const int index = default_generate_random_integer() % total_procs;
      return options[index];
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_get_next_local_cpu(void)
    //--------------------------------------------------------------------------
    {
      Processor result = local_cpus[next_local_cpu++];
      if (next_local_cpu == local_cpus.size())
        next_local_cpu = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_get_next_global_cpu(void)
    //--------------------------------------------------------------------------
    {
      if (total_nodes == 1)
        return default_get_next_local_cpu();
      if (!next_global_cpu.exists())
      {
        global_cpu_query = new Machine::ProcessorQuery(machine);
        global_cpu_query->only_kind(Processor::LOC_PROC);
        next_global_cpu = global_cpu_query->first();
      }
      Processor result = next_global_cpu;
      next_global_cpu = global_cpu_query->next(result);
      if (!next_global_cpu.exists())
      {
        delete global_cpu_query;
        global_cpu_query = NULL;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_get_next_local_gpu(void)
    //--------------------------------------------------------------------------
    {
      Processor result = local_gpus[next_local_gpu++];
      if (next_local_gpu == local_gpus.size())
        next_local_gpu = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_get_next_global_gpu(void)
    //--------------------------------------------------------------------------
    {
      if (total_nodes == 1)
        return default_get_next_local_gpu();
      if (!next_global_gpu.exists())
      {
        global_gpu_query = new Machine::ProcessorQuery(machine);
        global_gpu_query->only_kind(Processor::TOC_PROC);
        next_global_gpu = global_gpu_query->first();
      }
      Processor result = next_global_gpu;
      next_global_gpu = global_gpu_query->next(result);
      if (!next_global_gpu.exists())
      {
        delete global_gpu_query;
        global_gpu_query = NULL;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_get_next_local_io(void)
    //--------------------------------------------------------------------------
    {
      Processor result = local_ios[next_local_io++];
      if (next_local_io == local_ios.size())
        next_local_io = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_get_next_global_io(void)
    //--------------------------------------------------------------------------
    {
      if (total_nodes == 1)
        return default_get_next_local_io();
      if (!next_global_io.exists())
      {
        global_io_query = new Machine::ProcessorQuery(machine);
        global_io_query->only_kind(Processor::IO_PROC);
        next_global_io = global_io_query->first();
      }
      Processor result = next_global_io;
      next_global_io = global_io_query->next(result);
      if (!next_global_io.exists())
      {
        delete global_io_query;
        global_io_query = NULL;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_get_next_local_py(void)
    //--------------------------------------------------------------------------
    {
      Processor result = local_pys[next_local_py++];
      if (next_local_py == local_pys.size())
        next_local_py = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_get_next_global_py(void)
    //--------------------------------------------------------------------------
    {
      if (total_nodes == 1)
        return default_get_next_local_py();
      if (!next_global_py.exists())
      {
        global_py_query = new Machine::ProcessorQuery(machine);
        global_py_query->only_kind(Processor::PY_PROC);
        next_global_py = global_py_query->first();
      }
      Processor result = next_global_py;
      next_global_py = global_py_query->next(result);
      if (!next_global_py.exists())
      {
        delete global_py_query;
        global_py_query = NULL;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_get_next_local_procset(void)
    //--------------------------------------------------------------------------
    {
      Processor result = local_procsets[next_local_procset++];
      if (next_local_procset == local_procsets.size())
        next_local_procset = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_get_next_global_procset(void)
    //--------------------------------------------------------------------------
    {
      if (total_nodes == 1)
        return default_get_next_local_procset();
      if (!next_global_procset.exists())
      {
        global_procset_query = new Machine::ProcessorQuery(machine);
        global_procset_query->only_kind(Processor::PROC_SET);
        next_global_procset = global_procset_query->first();
      }
      Processor result = next_global_procset;
      next_global_procset = global_procset_query->next(result);
      if (!next_global_procset.exists())
      {
        delete global_procset_query;
        global_procset_query = NULL;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_get_next_local_omp(void)
    //--------------------------------------------------------------------------
    {
      Processor result = local_omps[next_local_omp++];
      if (next_local_omp == local_omps.size())
        next_local_omp = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_get_next_global_omp(void)
    //--------------------------------------------------------------------------
    {
      if (total_nodes == 1)
        return default_get_next_local_omp();
      if (!next_global_omp.exists())
      {
        global_omp_query = new Machine::ProcessorQuery(machine);
        global_omp_query->only_kind(Processor::OMP_PROC);
        next_global_omp = global_omp_query->first();
      }
      Processor result = next_global_omp;
      next_global_omp = global_omp_query->next(result);
      if (!next_global_omp.exists())
      {
        delete global_omp_query;
        global_omp_query = NULL;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    DefaultMapper::VariantInfo DefaultMapper::default_find_preferred_variant(
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

      Machine::ProcessorQuery all_procsets(machine);
      all_procsets.only_kind(Processor::PROC_SET);
      Machine::ProcessorQuery::iterator procset_finder = all_procsets.begin();


      /* if we have a procset variant use it */
      if (have_proc_kind_variant(ctx, task.task_id, Processor::PROC_SET)) {
        std::vector<VariantID> variants;
        runtime->find_valid_variants(ctx, task.task_id, 
                                     variants, Processor::PROC_SET);
        if (variants.size() > 0) {
          VariantInfo result;
          result.proc_kind = Processor::PROC_SET;
          result.variant = variants[0];
          result.tight_bound = (variants.size() == 1);
          result.is_inner = false;
          return result;
        }
      }

      // Otherwise we actually need to pick one
      // Ask the runtime for the variant IDs for the given task type
      std::vector<VariantID> variants;
      runtime->find_valid_variants(ctx, task.task_id, variants);
      if (!variants.empty())
      {
        variants.clear();
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
              case Processor::PY_PROC:
                {
                  if (local_pys.empty())
                    continue;
                  break;
                }
              case Processor::PROC_SET:
                {
                  if (local_procsets.empty())
                    continue;
                  break;
                }
              case Processor::OMP_PROC:
                {
                  if (local_omps.empty())
                    continue;
                  break;
                }
              default:
                assert(false); // unknown processor type
            }

            // See if we have any variants of this kind
            runtime->find_valid_variants(ctx, task.task_id, 
                                          variants, ranking[idx]);
            // If we have valid variants and we have processors we are
            // good to use this set of variants
            if (!variants.empty())
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
          runtime->find_valid_variants(ctx, task.task_id, 
                                              variants, best_kind);
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
              &(runtime->find_execution_constraints(ctx, 
                                            task.task_id, best_variant));
            const TaskLayoutConstraintSet *best_layout_constraints = 
              &(runtime->find_task_layout_constraints(ctx, 
                                            task.task_id, best_variant));
            for (unsigned idx = 1; idx < variants.size(); idx++)
            {
              const ExecutionConstraintSet &next_execution_constraints = 
                runtime->find_execution_constraints(ctx, 
                                            task.task_id, variants[idx]);
              const TaskLayoutConstraintSet &next_layout_constraints = 
                runtime->find_task_layout_constraints(ctx, 
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
          result.is_inner = runtime->is_inner_variant(ctx, task.task_id,
                                                       result.variant);
          if (result.is_inner)
          {
            // Default mapper assumes virtual mappings for all inner
            // tasks, so see if there are any layout constraints that
            // are inconsistent with this approach
            const TaskLayoutConstraintSet &next_layout_constraints = 
                runtime->find_task_layout_constraints(ctx, 
                                            task.task_id, result.variant);
            if (!next_layout_constraints.layouts.empty())
            {
              for (std::multimap<unsigned,LayoutConstraintID>::const_iterator 
                    it = next_layout_constraints.layouts.begin();
                    it != next_layout_constraints.layouts.end(); it++)
              {
                const LayoutConstraintSet &req_cons = 
                    runtime->find_layout_constraints(ctx, it->second);
                if ((req_cons.specialized_constraint.kind != NO_SPECIALIZE) &&
                   (req_cons.specialized_constraint.kind != VIRTUAL_SPECIALIZE))
                {
                  result.is_inner = false;
                  break;
                }
              }
            }
          }
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
      // GPU > OMP > procset > cpu > IO > Python
      // It is up to the caller to filter out processor kinds that aren't
      // suitable for a given task
      if (local_gpus.size() > 0) ranking.push_back(Processor::TOC_PROC);
      if (local_omps.size() > 0) ranking.push_back(Processor::OMP_PROC);
      if (local_procsets.size() > 0) ranking.push_back(Processor::PROC_SET);
      ranking.push_back(Processor::LOC_PROC);
      if (local_ios.size() > 0) ranking.push_back(Processor::IO_PROC);
      if (local_pys.size() > 0) ranking.push_back(Processor::PY_PROC);
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
      bool has_restricted_regions = false;
      for (std::map<unsigned,std::vector<PhysicalInstance> >::const_iterator
            it = input.valid_instances.begin(); 
            it != input.valid_instances.end(); it++)
      {
        // If this region requirements is restricted, then we can just
        // copy over the instances because we know we have to use them
        if (task.regions[it->first].is_restricted())
        {
          output.premapped_instances.insert(*it);
          runtime->acquire_instances(ctx, it->second);
          has_restricted_regions = true;
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
          Machine::MemoryQuery visible_memories(machine);
          visible_memories.has_affinity_to(task.target_proc)
            .only_kind(Memory::GLOBAL_MEM);
          Memory global_memory = Memory::NO_MEMORY;
          if (visible_memories.count() > 0)
            global_memory = visible_memories.first();
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
            case Processor::PY_PROC:
              {
                if (task.index_domain.get_volume() > local_pys.size())
                {
                  if (!global_memory.exists())
                  {
                    log_mapper.error("Default mapper failure. No memory found "
                        "for Python task %s (ID %lld) which is visible "
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
            case Processor::PROC_SET:
              {
                if (task.index_domain.get_volume() > local_procsets.size())
                {
                  if (!global_memory.exists())
                  {
                    log_mapper.error("Default mapper failure. No memory found "
                        "for ProcessorSet task %s (ID %lld) which is visible "
                        "for all point in the index space.",
                        task.get_task_name(), task.get_unique_id());
                    assert(false);
                  }
                  else
                    target_memory = global_memory;
                }
                break;
              }
            case Processor::OMP_PROC:
              {
                if (task.index_domain.get_volume() > local_omps.size())
                {
                  if (!global_memory.exists())
                  {
                    log_mapper.error("Default mapper failure. No memory found "
                        "for OMP task %s (ID %lld) which is visible "
                        "for all point in the index space.",
                        task.get_task_name(), task.get_unique_id());
                    assert(false);
                  }
                  else
                    target_memory = global_memory;
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
              info = default_find_preferred_variant(task, ctx, 
                  true/*needs tight bound*/, true/*cache*/,
                  task.target_proc.kind());
              has_variant_info = true;
            }
            // Map into the target memory and we are done
            std::set<FieldID> needed_fields = 
              task.regions[it->first].privilege_fields;
            const TaskLayoutConstraintSet &layout_constraints =
              runtime->find_task_layout_constraints(ctx,
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
        Machine::MemoryQuery visible_memories(machine);
        visible_memories.has_affinity_to(task.target_proc);
        switch (task.target_proc.kind())
        {
          case Processor::LOC_PROC:
          case Processor::IO_PROC:
          case Processor::PROC_SET:
          case Processor::OMP_PROC:
          case Processor::PY_PROC:
            {
              visible_memories.only_kind(Memory::SYSTEM_MEM);
              if (visible_memories.count() == 0)
              {
                log_mapper.error("Default mapper error. No memory found for "
                    "CPU task %s (ID %lld) which is visible for all points "
                    "in the index space.", task.get_task_name(), 
                    task.get_unique_id());
                assert(false);
              }
              target_memory = visible_memories.first();
              break;
            }
          case Processor::TOC_PROC:
            {
              // Otherwise for GPUs put the instance in zero-copy memory
              visible_memories.only_kind(Memory::Z_COPY_MEM);
              if (visible_memories.count() == 0)
              {
                log_mapper.error("Default mapper error. No memory found for "
                    "GPU task %s (ID %lld) which is visible for all points "
                    "in the index space.", task.get_task_name(),
                    task.get_unique_id());
                assert(false);
              }
              target_memory = visible_memories.first();
              break;
            }
          default:
            assert(false); // unknown processor kind
        }
        assert(target_memory.exists());
        if (!has_variant_info)
        {
          info = default_find_preferred_variant(task, ctx, 
              true/*needs tight bound*/, true/*cache*/,
              task.target_proc.kind());
          has_variant_info = true;
        }
        // Map into the target memory and we are done
        std::set<FieldID> needed_fields = 
          task.regions[it->first].privilege_fields;
        const TaskLayoutConstraintSet &layout_constraints =
          runtime->find_task_layout_constraints(ctx,
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
      // If we have any restricted regions, put the task 
      // back on the origin processor
      if (has_restricted_regions)
        output.new_target_proc = task.orig_proc;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::slice_task(const MapperContext      ctx,
                                   const Task&              task, 
                                   const SliceTaskInput&    input,
                                         SliceTaskOutput&   output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default slice_task in %s", get_mapper_name());

      std::vector<VariantID> variants;
      runtime->find_valid_variants(ctx, task.task_id, variants);
      /* find if we have a procset variant for task */
      for(unsigned i = 0; i < variants.size(); i++)
      {
        const ExecutionConstraintSet exset =
           runtime->find_execution_constraints(ctx, task.task_id, variants[i]);
        if(exset.processor_constraint.kind == Processor::PROC_SET) {

           // Before we do anything else, see if it is in the cache
           std::map<Domain,std::vector<TaskSlice> >::const_iterator finder =
             procset_slices_cache.find(input.domain);
           if (finder != procset_slices_cache.end()) {
                   output.slices = finder->second;
                   return;
           }

          output.slices.resize(input.domain.get_volume());
          unsigned idx = 0;
          Rect<1> rect = input.domain;
          for (PointInRectIterator<1> pir(rect); pir(); pir++, idx++)
          {
            Rect<1> slice(*pir, *pir);
            output.slices[idx] = TaskSlice(slice,
              remote_procsets[idx % remote_cpus.size()],
              false/*recurse*/, false/*stealable*/);
          }

          // Save the result in the cache
          procset_slices_cache[input.domain] = output.slices;
          return;
        }
      }
 

      // Whatever kind of processor we are is the one this task should
      // be scheduled on as determined by select initial task
      Processor::Kind target_kind =
        task.must_epoch_task ? local_proc.kind() : task.target_proc.kind();
      switch (target_kind)
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
        case Processor::PY_PROC:
          {
            default_slice_task(task, local_pys, remote_pys, 
                               input, output, py_slices_cache);
            break;
          }
        case Processor::PROC_SET:
          {
            default_slice_task(task, local_procsets, remote_procsets, 
                               input, output, procset_slices_cache);
            break;
          }
        case Processor::OMP_PROC:
          {
            default_slice_task(task, local_omps, remote_omps,
                               input, output, omp_slices_cache);
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

#if 1
      // The two-level decomposition doesn't work so for now do a
      // simple one-level decomposition across all the processors.
      Machine::ProcessorQuery all_procs(machine);
      all_procs.only_kind(local[0].kind());
      if ((task.tag & SAME_ADDRESS_SPACE) != 0)
	all_procs.local_address_space();
      std::vector<Processor> procs(all_procs.begin(), all_procs.end());

      switch (input.domain.get_dim())
      {
        case 1:
          {
            DomainT<1,coord_t> point_space = input.domain; 
            Point<1,coord_t> num_blocks(procs.size());
            default_decompose_points<1>(point_space, procs,
                  num_blocks, false/*recurse*/,
                  stealing_enabled, output.slices);
            break;
          }
        case 2:
          {
            DomainT<2,coord_t> point_space = input.domain;
            Point<2,coord_t> num_blocks =
              default_select_num_blocks<2>(procs.size(), point_space.bounds);
            default_decompose_points<2>(point_space, procs,
                num_blocks, false/*recurse*/,
                stealing_enabled, output.slices);
            break;
          }
        case 3:
          {
            DomainT<3,coord_t> point_space = input.domain;
            Point<3,coord_t> num_blocks =
              default_select_num_blocks<3>(procs.size(), point_space.bounds);
            default_decompose_points<3>(point_space, procs,
                num_blocks, false/*recurse*/,
                stealing_enabled, output.slices);
            break;
          }
        default: // don't support other dimensions right now
          assert(false);
      }
#else
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
            DomainT<1,coord_t> point_space = input.domain;
            if (remote.size() > 1) {
              if (total_points <= local.size()) {
                Point<1,coord_t> num_blocks(local.size());
                default_decompose_points<1>(point_space, local,
                    num_blocks, false/*recurse*/,
                    stealing_enabled, output.slices);
              } else {
                Point<1,coord_t> num_blocks(remote.size());
                default_decompose_points<1>(point_space, remote,
                    num_blocks, true/*recurse*/,
                    stealing_enabled, output.slices);
              }
            } else {
              Point<1,coord_t> num_blocks(local.size());
              default_decompose_points<1>(point_space, local,
                  num_blocks, false/*recurse*/,
                  stealing_enabled, output.slices);
            }
            break;
          }
        case 2:
          {
            DomainT<2,coord_t> point_space = input.domain;
            if (remote.size() > 1) {
              if (total_points <= local.size()) {
                Point<2,coord_t> num_blocks =
                  default_select_num_blocks<2>(local.size(),point_space.bounds);
                default_decompose_points<2>(point_space, local,
                    num_blocks, false/*recurse*/,
                    stealing_enabled, output.slices);
              } else {
                Point<2,coord_t> num_blocks =
                 default_select_num_blocks<2>(remote.size(),point_space.bounds);
                default_decompose_points<2>(point_space, remote,
                    num_blocks, true/*recurse*/,
                    stealing_enabled, output.slices);
              }
            } else {
              Point<2,coord_t> num_blocks =
                default_select_num_blocks<2>(local.size(), point_space.bounds);
              default_decompose_points<2>(point_space, local,
                  num_blocks, false/*recurse*/,
                  stealing_enabled, output.slices);
            }
            break;
          }
        case 3:
          {
            DomainT<3,coord_t> point_space = input.domain;
            if (remote.size() > 1) {
              if (total_points <= local.size()) {
                Point<3,coord_t> num_blocks =
                  default_select_num_blocks<3>(local.size(),point_space.bounds);
                default_decompose_points<3>(point_space, local,
                    num_blocks, false/*recurse*/,
                    stealing_enabled, output.slices);
              } else {
                Point<3,coord_t> num_blocks =
                 default_select_num_blocks<3>(remote.size(),point_space.bounds);
                default_decompose_points<3>(point_space, remote,
                    num_blocks, true/*recurse*/,
                    stealing_enabled, output.slices);
              }
            } else {
              Point<3,coord_t> num_blocks =
                default_select_num_blocks<3>(local.size(), point_space.bounds);
              default_decompose_points<3>(point_space, local,
                  num_blocks, false/*recurse*/,
                  stealing_enabled, output.slices);
            }
            break;
          }
        default: // don't support other dimensions right now
          assert(false);
      }
#endif

      // Save the result in the cache
      cached_slices[input.domain] = output.slices;
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
      VariantInfo chosen = default_find_preferred_variant(task, ctx,
                        true/*needs tight bound*/, true/*cache*/, target_kind);
      output.chosen_variant = chosen.variant;
      output.task_priority = default_policy_select_task_priority(ctx, task);
      output.postmap_task = false;
      // Figure out our target processors
      default_policy_select_target_processors(ctx, task, output.target_procs);

      // See if we have an inner variant, if we do virtually map all the regions
      // We don't even both caching these since they are so simple
      if (chosen.is_inner)
      {
        // Check to see if we have any relaxed coherence modes in which
        // case we can no longer do virtual mappings so we'll fall through
        bool has_relaxed_coherence = false;
        for (unsigned idx = 0; idx < task.regions.size(); idx++)
        {
          if (task.regions[idx].prop != EXCLUSIVE)
          {
            has_relaxed_coherence = true;
            break;
          }
        }
        if (!has_relaxed_coherence)
        {
          std::vector<unsigned> reduction_indexes;
          for (unsigned idx = 0; idx < task.regions.size(); idx++)
          {
            // As long as this isn't a reduction-only region requirement
            // we will do a virtual mapping, for reduction-only instances
            // we will actually make a physical instance because the runtime
            // doesn't allow virtual mappings for reduction-only privileges
            if (task.regions[idx].privilege == REDUCE)
              reduction_indexes.push_back(idx);
            else
              output.chosen_instances[idx].push_back(
                  PhysicalInstance::get_virtual_instance());
          }
          if (!reduction_indexes.empty())
          {
            const TaskLayoutConstraintSet &layout_constraints =
                runtime->find_task_layout_constraints(ctx,
                                      task.task_id, output.chosen_variant);
            for (std::vector<unsigned>::const_iterator it = 
                  reduction_indexes.begin(); it != 
                  reduction_indexes.end(); it++)
            {
              Memory target_memory = default_policy_select_target_memory(ctx,
                                                         task.target_proc,
                                                         task.regions[*it]);
              std::set<FieldID> copy = task.regions[*it].privilege_fields;
              if (!default_create_custom_instances(ctx, task.target_proc,
                  target_memory, task.regions[*it], *it, copy, 
                  layout_constraints, false/*needs constraint check*/, 
                  output.chosen_instances[*it]))
              {
                default_report_failed_instance_creation(task, *it, 
                                            task.target_proc, target_memory);
              }
            }
          }
          return;
        }
      }
      // Should we cache this task?
      CachedMappingPolicy cache_policy =
        default_policy_select_task_cache_policy(ctx, task);

      // First, let's see if we've cached a result of this task mapping
      const unsigned long long task_hash = compute_task_hash(task);
      std::pair<TaskID,Processor> cache_key(task.task_id, task.target_proc);
      std::map<std::pair<TaskID,Processor>,
               std::list<CachedTaskMapping> >::const_iterator 
        finder = cached_task_mappings.find(cache_key);
      // This flag says whether we need to recheck the field constraints,
      // possibly because a new field was allocated in a region, so our old
      // cached physical instance(s) is(are) no longer valid
      bool needs_field_constraint_check = false;
      if (cache_policy == DEFAULT_CACHE_POLICY_ENABLE && finder != cached_task_mappings.end())
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
              runtime->find_task_layout_constraints(ctx,
                                  task.task_id, output.chosen_variant);
            for (unsigned idx = 0; idx < task.regions.size(); idx++)
            {
              if (task.regions[idx].privilege == REDUCE)
              {
                Memory target_memory = default_policy_select_target_memory(ctx,
                                                         task.target_proc,
                                                         task.regions[idx]);
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
          if (runtime->acquire_and_filter_instances(ctx, 
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
      runtime->filter_instances(ctx, task, output.chosen_variant,
                                 output.chosen_instances, missing_fields);
      // Track which regions have already been mapped 
      std::vector<bool> done_regions(task.regions.size(), false);
      if (!input.premapped_regions.empty())
        for (std::vector<unsigned>::const_iterator it = 
              input.premapped_regions.begin(); it != 
              input.premapped_regions.end(); it++)
          done_regions[*it] = true;
      const TaskLayoutConstraintSet &layout_constraints = 
        runtime->find_task_layout_constraints(ctx, 
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
        Memory target_memory = default_policy_select_target_memory(ctx,
                                                         task.target_proc,
                                                         task.regions[idx]);
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
	// Did the application request a virtual mapping for this requirement?
	if ((task.regions[idx].tag & DefaultMapper::VIRTUAL_MAP) != 0)
	{
	  PhysicalInstance virt_inst = PhysicalInstance::get_virtual_instance();
	  output.chosen_instances[idx].push_back(virt_inst);
	  continue;
	}
        // Check to see if any of the valid instances satisfy this requirement
        {
          std::vector<PhysicalInstance> valid_instances;

          for (std::vector<PhysicalInstance>::const_iterator
                 it = input.valid_instances[idx].begin(),
                 ie = input.valid_instances[idx].end(); it != ie; ++it)
          {
            if (it->get_location() == target_memory)
              valid_instances.push_back(*it);
          }

          std::set<FieldID> valid_missing_fields;
          runtime->filter_instances(ctx, task, idx, output.chosen_variant,
                                    valid_instances, valid_missing_fields);

#ifndef NDEBUG
          bool check =
#endif
            runtime->acquire_and_filter_instances(ctx, valid_instances);
          assert(check);

          output.chosen_instances[idx] = valid_instances;
          missing_fields[idx] = valid_missing_fields;

          if (missing_fields[idx].empty())
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
      if (cache_policy == DEFAULT_CACHE_POLICY_ENABLE) {
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
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs)
    //--------------------------------------------------------------------------
    {
      if (task.target_proc.address_space() == node_id)
      {
        switch (task.target_proc.kind())
        {
          case Processor::TOC_PROC:
            {
              // GPUs have their own memories so they only get one
              target_procs.push_back(task.target_proc);
              break;
            }
          case Processor::LOC_PROC:
            {
              // Put any of our local cpus on here
              // TODO: NUMA-ness needs to go here
              // If we're part of a must epoch launch, our 
              // target proc will be sufficient
              if (!task.must_epoch_task)
                target_procs.insert(target_procs.end(),
                    local_cpus.begin(), local_cpus.end());
              else
                target_procs.push_back(task.target_proc);
              break;
            }
          case Processor::IO_PROC:
            {
              // Put any of our I/O procs here
              // If we're part of a must epoch launch, our
              // target proc will be sufficient
              if (!task.must_epoch_task)
                target_procs.insert(target_procs.end(),
                    local_ios.begin(), local_ios.end());
              else
                target_procs.push_back(task.target_proc);
              break;
            }
          case Processor::PY_PROC:
            {
              // Put any of our Python procs here
              // If we're part of a must epoch launch, our
              // target proc will be sufficient
              if (!task.must_epoch_task)
                target_procs.insert(target_procs.end(),
                    local_pys.begin(), local_pys.end());
              else
                target_procs.push_back(task.target_proc);
              break;
            }
          case Processor::PROC_SET:
            {
              target_procs.push_back(task.target_proc);
              break;
            }
          case Processor::OMP_PROC:
            {
              // Put any of our local omps on here
              // TODO: NUMA-ness needs to go here
              // If we're part of a must epoch launch, our 
              // target proc will be sufficient
              if (!task.must_epoch_task)
                target_procs.insert(target_procs.end(),
                    local_omps.begin(), local_omps.end());
              else
                target_procs.push_back(task.target_proc);
              break;
            }
          default:
            assert(false); // unrecognized processor kind
        }
      }
      else
        target_procs.push_back(task.target_proc);
    }

    //--------------------------------------------------------------------------
    TaskPriority DefaultMapper::default_policy_select_task_priority(
                                    MapperContext ctx, const Task &task)
    //--------------------------------------------------------------------------
    {
      // TODO: some criticality analysis to assign priorities
      return 0;
    }

    //--------------------------------------------------------------------------
    DefaultMapper::CachedMappingPolicy
    DefaultMapper::default_policy_select_task_cache_policy(
                                    MapperContext ctx, const Task &task)
    //--------------------------------------------------------------------------
    {
      // Always cache task result.
      return DEFAULT_CACHE_POLICY_ENABLE;
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
          {
            if (it->is_external_instance())
              continue;
            runtime->set_garbage_collection_priority(ctx, *it, 0/*priority*/);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ unsigned long long DefaultMapper::compute_task_hash(
                                                               const Task &task)
    //--------------------------------------------------------------------------
    {
      // Use Sean's "cheesy" hash function    
      const unsigned long long c1 = 0x5491C27F12DB3FA5; // big number, mix 1+0s
      const unsigned long long c2 = 353435097; // chosen by fair dice roll
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
      // Special case for reduction instances, no point in checking
      // for existing ones and we also know that currently we can only
      // make a single instance for each field of a reduction
      if (req.privilege == REDUCE)
      {
        // Iterate over the fields one by one for now, once Realm figures
        // out how to deal with reduction instances that contain
        bool force_new_instances = true; // always have to force new instances
        LayoutConstraintID our_layout_id = 
         default_policy_select_layout_constraints(ctx, target_memory, req,
               TASK_MAPPING, needs_field_constraint_check, force_new_instances);
        LayoutConstraintSet our_constraints = 
                      runtime->find_layout_constraints(ctx, our_layout_id);
        instances.resize(instances.size() + req.privilege_fields.size());
        unsigned idx = 0;
        for (std::set<FieldID>::const_iterator it = 
              req.privilege_fields.begin(); it !=
              req.privilege_fields.end(); it++, idx++)
        {
          our_constraints.field_constraint.field_set.clear();
          our_constraints.field_constraint.field_set.push_back(*it);
          if (!default_make_instance(ctx, target_memory, our_constraints,
                       instances[idx], TASK_MAPPING, force_new_instances,
                       true/*meets*/, req))
            return false;
        }
        return true; 
      }
      // Before we do anything else figure out our 
      // constraints for any instances of this task, then we'll
      // see if these constraints conflict with or are satisfied by
      // any of the other constraints 
      bool force_new_instances = false;
      LayoutConstraintID our_layout_id = 
       default_policy_select_layout_constraints(ctx, target_memory, req,
               TASK_MAPPING, needs_field_constraint_check, force_new_instances);
      const LayoutConstraintSet &our_constraints = 
                    runtime->find_layout_constraints(ctx, our_layout_id);
      for (std::multimap<unsigned,LayoutConstraintID>::const_iterator lay_it =
            layout_constraints.layouts.lower_bound(index); lay_it !=
            layout_constraints.layouts.upper_bound(index); lay_it++)
      {
        // Get the constraints
        const LayoutConstraintSet &index_constraints = 
                  runtime->find_layout_constraints(ctx, lay_it->second);
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
        if (runtime->do_constraints_conflict(ctx, 
                                              our_layout_id, lay_it->second))
        {
          // They conflict, so we're just going to make an instance
          // using these constraints
          if (!default_make_instance(ctx, target_memory, index_constraints,
                     instances.back(), TASK_MAPPING, force_new_instances, 
                     false/*meets*/, req))
            return false;
        }
        else if (runtime->do_constraints_entail(ctx, 
                                                 lay_it->second, our_layout_id))
        {
          // These constraints do everything we want to do and maybe more
          // so we can just use them directly
          if (!default_make_instance(ctx, target_memory, index_constraints,
                      instances.back(), TASK_MAPPING, force_new_instances, 
                      true/*meets*/, req))
            return false;
        }
        else
        {
          // These constraints don't do as much as we want but don't
          // conflict so make an instance with them and our constraints 
          LayoutConstraintSet creation_constraints = index_constraints;
          default_policy_select_constraints(ctx, creation_constraints, 
                                            target_memory, req);
          creation_constraints.add_constraint(
              FieldConstraint(overlaping_fields,
                false/*contig*/, false/*inorder*/));
          if (!default_make_instance(ctx, target_memory, creation_constraints,
                         instances.back(), TASK_MAPPING, force_new_instances, 
                         true/*meets*/, req))
            return false;
        }
      }
      // If we don't have anymore needed fields, we are done
      if (needed_fields.empty())
        return true;
      // There are no constraints for these fields so we get to do what we want
      instances.resize(instances.size()+1);
      LayoutConstraintSet creation_constraints = our_constraints;
      std::vector<FieldID> creation_fields;
      default_policy_select_instance_fields(ctx, req, needed_fields,
          creation_fields);
      creation_constraints.add_constraint(
          FieldConstraint(creation_fields, false/*contig*/, false/*inorder*/));
      if (!default_make_instance(ctx, target_memory, creation_constraints, 
                instances.back(), TASK_MAPPING, force_new_instances, 
                true/*meets*/,  req))
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    Memory DefaultMapper::default_policy_select_target_memory(MapperContext ctx,
                                                   Processor target_proc,
                                                   const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      bool prefer_rdma = ((req.tag & DefaultMapper::PREFER_RDMA_MEMORY) != 0);

      // TODO: deal with the updates in machine model which will
      //       invalidate this cache
      std::map<Processor,Memory>::iterator it;
      if (prefer_rdma)
      {
	it = cached_rdma_target_memory.find(target_proc);
	if (it != cached_rdma_target_memory.end()) return it->second;
      } else {
        it = cached_target_memory.find(target_proc);
	if (it != cached_target_memory.end()) return it->second;
      }

      // Find the visible memories from the processor for the given kind
      Machine::MemoryQuery visible_memories(machine);
      visible_memories.has_affinity_to(target_proc);
      if (visible_memories.count() == 0)
      {
        log_mapper.error("No visible memories from processor " IDFMT "! "
                         "This machine is really messed up!", target_proc.id);
        assert(false);
      }
      // Figure out the memory with the highest-bandwidth
      Memory best_memory = Memory::NO_MEMORY;
      unsigned best_bandwidth = 0;
      Memory best_rdma_memory = Memory::NO_MEMORY;
      unsigned best_rdma_bandwidth = 0;
      std::vector<Machine::ProcessorMemoryAffinity> affinity(1);
      for (Machine::MemoryQuery::iterator it = visible_memories.begin();
            it != visible_memories.end(); it++)
      {
        affinity.clear();
        machine.get_proc_mem_affinity(affinity, target_proc, *it,
				      false /*not just local affinities*/);
        assert(affinity.size() == 1);
        if (!best_memory.exists() || (affinity[0].bandwidth > best_bandwidth)) {
          best_memory = *it;
          best_bandwidth = affinity[0].bandwidth;
        }
        if ((it->kind() == Memory::REGDMA_MEM) &&
	    (!best_rdma_memory.exists() ||
	     (affinity[0].bandwidth > best_rdma_bandwidth))) {
          best_rdma_memory = *it;
          best_rdma_bandwidth = affinity[0].bandwidth;
        }
      }
      assert(best_memory.exists());
      if (prefer_rdma)
      {
	if (!best_rdma_memory.exists()) best_rdma_memory = best_memory;
	cached_rdma_target_memory[target_proc] = best_rdma_memory;
	return best_rdma_memory;
      } else {
	cached_target_memory[target_proc] = best_memory;
	return best_memory;
      }
    }

    //--------------------------------------------------------------------------
    LayoutConstraintID DefaultMapper::default_policy_select_layout_constraints(
                                    MapperContext ctx, Memory target_memory, 
                                    const RegionRequirement &req,
                                    MappingKind mapping_kind,
                                    bool needs_field_constraint_check,
                                    bool &force_new_instances)
    //--------------------------------------------------------------------------
    {
      // Do something special for reductions and 
      // it is not an explicit region-to-region copy
      if ((req.privilege == REDUCE) && (mapping_kind != COPY_MAPPING))
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
        default_policy_select_constraints(ctx, constraints, target_memory, req);
        LayoutConstraintID result = 
          runtime->register_layout(ctx, constraints);
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
                runtime->find_layout_constraints(ctx, finder->second);
        // Should be only one unless things have changed
        const std::vector<FieldID> &old_set = 
                          old_constraints.field_constraint.get_field_set();
        // Check to make sure the field sets are still the same
        std::vector<FieldID> new_fields;
        runtime->get_field_space_fields(ctx, 
                                        constraint_key.second,new_fields);
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
      default_policy_select_constraints(ctx, constraints, target_memory, req);
      // Do the registration
      LayoutConstraintID result = 
        runtime->register_layout(ctx, constraints);
      // Record our results, there is a benign race here as another mapper
      // call could have registered the exact same registration constraints
      // here if we were preempted during the registration call. The 
      // constraint sets are identical though so it's all good.
      layout_constraint_cache[constraint_key] = result;
      return result; 
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::default_policy_select_constraints(MapperContext ctx,
                     LayoutConstraintSet &constraints, Memory target_memory,
                     const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      // See if we are doing a reduction instance
      if (req.privilege == REDUCE)
      {
        // Make reduction fold instances
        constraints.add_constraint(SpecializedConstraint(
                            REDUCTION_FOLD_SPECIALIZE, req.redop))
          .add_constraint(MemoryConstraint(target_memory.kind()));
      }
      else
      {
        // Normal instance creation
        std::vector<FieldID> fields;
        default_policy_select_constraint_fields(ctx, req, fields);
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
          .add_constraint(FieldConstraint(fields, false/*contiguous*/,
                                          false/*inorder*/))
          .add_constraint(OrderingConstraint(dimension_ordering, 
                                             false/*contigous*/));
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::default_policy_select_constraint_fields(
                                    MapperContext ctx,
                                    const RegionRequirement &req,
                                    std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      FieldSpace handle = req.region.get_field_space();
      runtime->get_field_space_fields(ctx, handle, fields);
    }


    //--------------------------------------------------------------------------
    bool DefaultMapper::default_make_instance(MapperContext ctx, 
        Memory target_memory, const LayoutConstraintSet &constraints,
        PhysicalInstance &result, MappingKind kind, bool force_new, bool meets, 
        const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      bool created = true;
      LogicalRegion target_region = 
        default_policy_select_instance_region(ctx, target_memory, req,
                                              constraints, force_new, meets);
      // TODO: deal with task layout constraints that require multiple
      // region requirements to be mapped to the same instance
      std::vector<LogicalRegion> target_regions(1, target_region);
      if (force_new || (req.privilege == REDUCE && (kind != COPY_MAPPING))) {
        if (!runtime->create_physical_instance(ctx, target_memory, 
              constraints, target_regions, result))
          return false;
      } else {
        if (!runtime->find_or_create_physical_instance(ctx, 
              target_memory, constraints, target_regions, result, created))
          return false;
      }
      if (created)
      {
        int priority = default_policy_select_garbage_collection_priority(ctx, 
                kind, target_memory, result, meets, (req.privilege == REDUCE));
        if ((priority != 0) && !result.is_external_instance())
          runtime->set_garbage_collection_priority(ctx, result,priority);
      }
      return true;
    }

    //--------------------------------------------------------------------------
    LogicalRegion DefaultMapper::default_policy_select_instance_region(
                                MapperContext ctx, Memory target_memory,
                                const RegionRequirement &req,
                                const LayoutConstraintSet &layout_constraints,
                                bool force_new_instances, 
                                bool meets_constraints)
    //--------------------------------------------------------------------------
    {
      // If it is not something we are making a big region for just
      // return the region that is actually needed
      LogicalRegion result = req.region; 
      if (!meets_constraints || (req.privilege == REDUCE))
        return result;

      // If the application requested that we use the exact region requested,
      // honor that
      if ((req.tag & DefaultMapper::EXACT_REGION) != 0)
        return result;

      // Heuristically use the exact region if the target memory is either a GPU
      // framebuffer or a zero copy memory.
      if (target_memory.kind() == Memory::GPU_FB_MEM ||
          target_memory.kind() == Memory::Z_COPY_MEM)
        return result;

      // Simple heuristic here, if we are on a single node, we go all the
      // way to the root since the first-level partition is likely just
      // across processors in the node, however, if we are on multiple nodes
      // we try to find the first level that effectively partitions the root
      // into one subregion per node.
      if (total_nodes == 1)
      {
        while (runtime->has_parent_logical_partition(ctx, result))
        {
          LogicalPartition parent =
            runtime->get_parent_logical_partition(ctx, result);
          result = runtime->get_parent_logical_region(ctx, parent);
        }
        return result;
      }
      else
      {
        // Fall through if the application actually asked for the root
        if (!runtime->has_parent_logical_partition(ctx, result))
          return result;

        std::vector<LogicalRegion> path;
        std::vector<size_t> volumes;

        path.push_back(result);
        volumes.push_back(runtime->get_index_space_domain(ctx,
                                        result.get_index_space()).get_volume());

        // Collect the size of subregion at each level
        LogicalRegion next = result;
        while (runtime->has_parent_logical_partition(ctx, next))
        {
          LogicalPartition parent =
            runtime->get_parent_logical_partition(ctx, next);
          next = runtime->get_parent_logical_region(ctx, parent);
          path.push_back(next);
          volumes.push_back(
            runtime->get_index_space_domain(ctx, next.get_index_space()).get_volume());
        }

        // Acculumate the "effective" fanout at each level and
        // stop the search once we have one subregion per node.
        double effective_fanout = 1.0;
        for (off_t idx = (off_t)path.size() - 2; idx >= 0; --idx)
        {
          effective_fanout *= (double)volumes[idx + 1] / volumes[idx];
          if ((unsigned)effective_fanout >= total_nodes)
            return path[idx];
        }

        // If we reached this point, the partitions were not meant to assign
        // one subregion per node. So, stop pretending to be smart and
        // just return the exact target.
        return result;
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::default_policy_select_instance_fields(
                                    MapperContext ctx,
                                    const RegionRequirement &req,
                                    const std::set<FieldID> &needed_fields,
                                    std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      if (total_nodes == 1)
      {
        FieldSpace handle = req.region.get_field_space();
        runtime->get_field_space_fields(ctx, handle, fields);
      }
      else
      {
        fields.insert(fields.end(), needed_fields.begin(), needed_fields.end());
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
      if (reduction)
        return GC_FIRST_PRIORITY;
      if (meets_fill_constraints && (kind == TASK_MAPPING))
        return GC_NEVER_PRIORITY;
      return GC_DEFAULT_PRIORITY;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::default_report_failed_instance_creation(
                                 const Task &task, unsigned index, 
                                 Processor target_proc, Memory target_mem) const
    //--------------------------------------------------------------------------
    {
      log_mapper.error("Default mapper failed allocation for region "
                       "requirement %d of task %s (UID %lld) in memory " IDFMT
                       " for processor " IDFMT ". This means the working set "
                       "of your application is too big for the allotted "
                       "capacity of the given memory under the default "
                       "mapper's mapping scheme. You have three choices: "
                       "ask Realm to allocate more memory, write a custom "
                       "mapper to better manage working sets, or find a bigger "
                       "machine. Good luck!", index, task.get_task_name(),
                       task.get_unique_id(), target_mem.id, target_proc.id);
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
      VariantInfo result = default_find_preferred_variant(task, ctx,
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
          machine.get_mem_mem_affinity(affinity, location, destination_memory,
				       false /*not just local affinities*/);
          unsigned memory_bandwidth = 0;
          if (!affinity.empty()) {
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
    void DefaultMapper::create_task_temporary_instance(
                                        const MapperContext       ctx,
                                        const Task&               task,
                                        const CreateTaskTemporaryInput& input,
                                              CreateTaskTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default create_task_temporary_instance in %s", 
                      get_mapper_name());
      output.temporary_instance = default_policy_create_temporary(ctx,
                          task.regions[input.region_requirement_index].region,
                          input.destination_instance);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance DefaultMapper::default_policy_create_temporary(
         const MapperContext ctx, LogicalRegion region, PhysicalInstance target)
    //--------------------------------------------------------------------------
    {
      PhysicalInstance result;
      std::vector<LogicalRegion> create_regions(1, region);
      // Always make the temporary with the same layout as the target
      LayoutConstraintID layout_id = target.get_layout_id();
      // Try making it in the same memory
      Memory target_mem = target.get_location();
      // Give these temporary instances minimum priority
      if (runtime->create_physical_instance(ctx, target_mem, layout_id,
              create_regions, result, true/*acquire*/, GC_MAX_PRIORITY))
        return result;
      // If that didn't work, try making it in any memory with affinity
      // to the target memory
      Machine::MemoryQuery other_mems(machine);
      other_mems.best_affinity_to(target_mem);
      for (Machine::MemoryQuery::iterator it = other_mems.begin();
            it != other_mems.end(); it++)
      {
        if (runtime->create_physical_instance(ctx, *it, layout_id,
              create_regions, result, true/*acquire*/, GC_MAX_PRIORITY))
          return result;
      }
      // If that didn't work we'll punt for now. We can try puting it in 
      // some other weird memory that may result in a multi-hop copy later
      log_mapper.error("Default mapper error. Mapper %s failed to create "
                       "a temporary instance in any reasonable memory.",
                       get_mapper_name());
      assert(false);
      return result;
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
      // check to see if we have any constraints we have to abide by
      LayoutConstraintSet creation_constraints;
      bool force_new_instances = false;
      Memory target_memory = Memory::NO_MEMORY;
      if (inline_op.layout_constraint_id > 0)
      {
        // Find our constraints
        creation_constraints = runtime->find_layout_constraints(ctx, 
                                            inline_op.layout_constraint_id);
        Memory target_memory;
        if (creation_constraints.memory_constraint.is_valid())
        {
          Machine::MemoryQuery valid_mems(machine);
          valid_mems.has_affinity_to(inline_op.parent_task->current_proc);
          valid_mems.only_kind(
              creation_constraints.memory_constraint.get_kind());
          if (valid_mems.count() == 0)
          {
            log_mapper.error("Default mapper error. Mapper %s could find no "
                             "valid memories for the constraints requested by "
                             "inline mapping %lld in parent task %s (ID %lld).",
                             get_mapper_name(), inline_op.get_unique_id(), 
                             inline_op.parent_task->get_task_name(),
                             inline_op.parent_task->get_unique_id());
            assert(false);
          }
          target_memory = valid_mems.first(); // just take the first one
        }
        else
          target_memory = default_policy_select_target_memory(ctx,
                                  inline_op.parent_task->current_proc,
                                  inline_op.requirement);
        if (creation_constraints.field_constraint.field_set.empty())
          creation_constraints.add_constraint(FieldConstraint(
                inline_op.requirement.privilege_fields, false/*contig*/));
      }
      else
      {
        // No constraints so do what we want
        // Copy over all the valid instances, then try to do an acquire on them
        // and see which instances are no longer valid
        output.chosen_instances = input.valid_instances;
        if (!output.chosen_instances.empty())
          runtime->acquire_and_filter_instances(ctx, 
                                            output.chosen_instances);
        // Now see if we have any fields which we still make space for
        std::set<FieldID> missing_fields = 
          inline_op.requirement.privilege_fields;
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
        target_memory = default_policy_select_target_memory(ctx,
                                        inline_op.parent_task->current_proc,
                                        inline_op.requirement);
        LayoutConstraintID our_layout_id = 
         default_policy_select_layout_constraints(ctx, target_memory, 
                                               inline_op.requirement, 
                                               INLINE_MAPPING,
                                               true/*needs check*/, 
                                               force_new_instances);
        creation_constraints = 
                runtime->find_layout_constraints(ctx, our_layout_id);
        creation_constraints.add_constraint(
            FieldConstraint(missing_fields, false/*contig*/, false/*inorder*/));
      }
      output.chosen_instances.resize(output.chosen_instances.size()+1);
      if (!default_make_instance(ctx, target_memory, creation_constraints,
            output.chosen_instances.back(), INLINE_MAPPING, 
            force_new_instances, true/*meets*/, inline_op.requirement))
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
                         target_memory.id,
                         inline_op.parent_task->current_proc.id);
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
    void DefaultMapper::create_inline_temporary_instance(
                                      const MapperContext         ctx,
                                      const InlineMapping&        inline_op,
                                      const CreateInlineTemporaryInput& input,
                                            CreateInlineTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default create_inline_temporary_instance in %s",
                      get_mapper_name());
      output.temporary_instance = default_policy_create_temporary(ctx,
            inline_op.requirement.region, input.destination_instance);
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
      bool has_unrestricted = false;
      for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
      {
        output.src_instances[idx] = input.src_instances[idx];
        if (!output.src_instances[idx].empty())
          runtime->acquire_and_filter_instances(ctx,
                                              output.src_instances[idx]);
        // Check to see if we are doing a reduce-across in which case we
        // need to actually create a real physical instance
        if (copy.dst_requirements[idx].privilege == REDUCE)
        {
          // If the source is restricted, we know we are good
          if (!copy.src_requirements[idx].is_restricted())
            default_create_copy_instance<true/*is src*/>(ctx, copy, 
                copy.src_requirements[idx], idx, output.src_instances[idx]);
        }
        else // Stick this on for good measure, at worst it will be ignored
          output.src_instances[idx].push_back(
              PhysicalInstance::get_virtual_instance());
        output.dst_instances[idx] = input.dst_instances[idx];
        if (!output.dst_instances[idx].empty())
          runtime->acquire_and_filter_instances(ctx,
                                  output.dst_instances[idx]);
        if (!copy.dst_requirements[idx].is_restricted())
          has_unrestricted = true;
      }
      // If the destinations were all restricted we know we got everything
      if (has_unrestricted)
      {
        for (unsigned idx = 0; idx < copy.dst_requirements.size(); idx++)
        {
          output.dst_instances[idx] = input.dst_instances[idx];
          if (!copy.dst_requirements[idx].is_restricted())
            default_create_copy_instance<false/*is src*/>(ctx, copy, 
                copy.dst_requirements[idx], idx, output.dst_instances[idx]);
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
    void DefaultMapper::create_copy_temporary_instance(
                                    const MapperContext                 ctx,
                                    const Copy&                         copy,
                                    const CreateCopyTemporaryInput&     input,
                                          CreateCopyTemporaryOutput&    output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default create_copy_temporary_instance in %s",
                      get_mapper_name());
      if (input.src_requirement)
        output.temporary_instance = default_policy_create_temporary(ctx,
            copy.src_requirements[input.region_requirement_index].region, 
            input.destination_instance);
      else
        output.temporary_instance = default_policy_create_temporary(ctx,
            copy.dst_requirements[input.region_requirement_index].region,
            input.destination_instance);
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
      // otherwise see if we should make a composite or a real instance.
      output.chosen_instances = input.valid_instances;
      if (!output.chosen_instances.empty())
        runtime->acquire_and_filter_instances(ctx, 
                                                  output.chosen_instances);

      if (default_policy_select_close_virtual(ctx, close)) {
        output.chosen_instances.push_back(
                                  PhysicalInstance::get_virtual_instance());
      } else {
        // Make one big instance at the location where the parent task
        // is running.
        Memory target_memory =
          default_policy_select_target_memory(ctx, 
                                              close.parent_task->current_proc,
                                              close.requirement);
        LayoutConstraintSet constraints;
        default_policy_select_constraints(ctx, constraints, target_memory,
                                          close.requirement);

        output.chosen_instances.resize(output.chosen_instances.size()+1);
        if (!default_make_instance(ctx, target_memory, constraints,
              output.chosen_instances.back(), CLOSE_MAPPING,
              false/*force*/, true/*meets*/, close.requirement))
        {
          // If we failed to make it that is bad
          log_mapper.error("Default mapper failed allocation for region "
                         "requirement of close in task %s (UID %lld) "
                         "in memory " IDFMT "for processor " IDFMT ". This "
                         "means the working set of your application is too big "
                         "for the allotted capacity of the given memory under "
                         "the default mapper's mapping scheme. You have three "
                         "choices: ask Realm to allocate more memory, write a "
                         "custom mapper to better manage working sets, or find "
                         "a bigger machine. Good luck!",
                         close.parent_task->get_task_name(),
                         close.parent_task->get_unique_id(),
                         close.parent_task->current_proc.id,
                         target_memory.id);
          assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::default_policy_select_close_virtual(
                          const MapperContext ctx,
                          const Close&        close)
    //--------------------------------------------------------------------------
    {
      return true;
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
    void DefaultMapper::create_close_temporary_instance(
                                      const MapperContext             ctx,
                                      const Close&                    close,
                                      const CreateCloseTemporaryInput& input,
                                            CreateCloseTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default create_close_temporary_instance in %s",
                      get_mapper_name());
      output.temporary_instance = default_policy_create_temporary(ctx,
                close.requirement.region, input.destination_instance);
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
    void DefaultMapper::create_release_temporary_instance(
                                    const MapperContext                 ctx,
                                    const Release&                      release,
                                    const CreateReleaseTemporaryInput&  input,
                                          CreateReleaseTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default create_release_temporary_instance in %s",
                      get_mapper_name());
      output.temporary_instance = default_policy_create_temporary(ctx,
                  release.logical_region, input.destination_instance);
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
    void DefaultMapper::select_partition_projection(const MapperContext  ctx,
                        const Partition&                           partition,
                        const SelectPartitionProjectionInput&      input,
                              SelectPartitionProjectionOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_partition_projection in %s", 
                      get_mapper_name());
      // If we have a complete partition then use it
      if (!input.open_complete_partitions.empty())
        output.chosen_partition = input.open_complete_partitions[0];
      else
        output.chosen_partition = LogicalPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_partition(const MapperContext        ctx,
                                      const Partition&           partition,
                                      const MapPartitionInput&   input,
                                            MapPartitionOutput&  output)
    //--------------------------------------------------------------------------
    {
      // No constraints on mapping partitions
      // Copy over all the valid instances, then try to do an acquire on them
      // and see which instances are no longer valid
      output.chosen_instances = input.valid_instances;
      if (!output.chosen_instances.empty())
        runtime->acquire_and_filter_instances(ctx, 
                                          output.chosen_instances);
      // Now see if we have any fields which we still make space for
      std::set<FieldID> missing_fields = 
        partition.requirement.privilege_fields;
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
                                      partition.parent_task->current_proc,
                                      partition.requirement);
      bool force_new_instances = false;
      LayoutConstraintID our_layout_id = 
       default_policy_select_layout_constraints(ctx, target_memory, 
                                             partition.requirement, 
                                             PARTITION_MAPPING,
                                             true/*needs check*/, 
                                             force_new_instances);
      LayoutConstraintSet creation_constraints = 
              runtime->find_layout_constraints(ctx, our_layout_id);
      creation_constraints.add_constraint(
          FieldConstraint(missing_fields, false/*contig*/, false/*inorder*/));
      output.chosen_instances.resize(output.chosen_instances.size()+1);
      if (!default_make_instance(ctx, target_memory, creation_constraints,
            output.chosen_instances.back(), PARTITION_MAPPING, 
            force_new_instances, true/*meets*/, partition.requirement))
      {
        // If we failed to make it that is bad
        log_mapper.error("Default mapper failed allocation for region "
                         "requirement of partition in task %s (UID %lld) "
                         "in memory " IDFMT "for processor " IDFMT ". This "
                         "means the working set of your application is too big "
                         "for the allotted capacity of the given memory under "
                         "the default mapper's mapping scheme. You have three "
                         "choices: ask Realm to allocate more memory, write a "
                         "custom mapper to better manage working sets, or find "
                         "a bigger machine. Good luck!", 
                         partition.parent_task->get_task_name(),
                         partition.parent_task->get_unique_id(),
                         target_memory.id,
                         partition.parent_task->current_proc.id);
        assert(false);
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_partition_sources(
                                     const MapperContext             ctx,
                                     const Partition&                partition,
                                     const SelectPartitionSrcInput&  input,
                                           SelectPartitionSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_partition_sources in %s", 
                      get_mapper_name());
      default_policy_select_sources(ctx, input.target, input.source_instances,
                                    output.chosen_ranking);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::create_partition_temporary_instance(
                            const MapperContext                   ctx,
                            const Partition&                      partition,
                            const CreatePartitionTemporaryInput&  input,
                                  CreatePartitionTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default create_partition_temporary_instance in %s",
                      get_mapper_name());
      output.temporary_instance = default_policy_create_temporary(ctx,
            partition.requirement.region, input.destination_instance);
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::report_profiling(const MapperContext              ctx,
                                         const Partition&             partition,
                                         const PartitionProfilingInfo&    input)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default report_profiling for Partition in %s", 
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
        case DEFAULT_TUNABLE_LOCAL_GPUS:
          {
            *result = local_gpus.size();
            break;
          }
        case DEFAULT_TUNABLE_LOCAL_CPUS:
          {
            *result = local_cpus.size();
            break;
          }
        case DEFAULT_TUNABLE_LOCAL_IOS:
          {
            *result = local_ios.size();
            break;
          }
        case DEFAULT_TUNABLE_LOCAL_OMPS:
          {
            *result = local_omps.size();
            break;
          }
        case DEFAULT_TUNABLE_LOCAL_PYS:
          {
            *result = local_pys.size();
            break;
          }
        case DEFAULT_TUNABLE_GLOBAL_GPUS:
          {
            // TODO: deal with machine asymmetry here
            *result = (local_gpus.size() * total_nodes);
            break;
          }
        case DEFAULT_TUNABLE_GLOBAL_CPUS:
          {
            // TODO: deal with machine asymmetry here
            *result = (local_cpus.size() * total_nodes);
            break;
          }
        case DEFAULT_TUNABLE_GLOBAL_IOS:
          {
            // TODO: deal with machine asymmetry here
            *result = (local_ios.size() * total_nodes);
            break;
          }
        case DEFAULT_TUNABLE_GLOBAL_OMPS:
          {
            // TODO: deal with machine asymmetry here
            *result = (local_omps.size() * total_nodes);
            break;
          }
        case DEFAULT_TUNABLE_GLOBAL_PYS:
          {
            // TODO: deal with machine asymmetry here
            *result = (local_pys.size() * total_nodes);
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
    bool DefaultMapper::have_proc_kind_variant(const MapperContext ctx,
					       TaskID id, Processor::Kind kind)
    //--------------------------------------------------------------------------
    {
      std::vector<VariantID> variants;
      runtime->find_valid_variants(ctx, id, variants);

      for(unsigned i = 0; i < variants.size(); i++)
      {
	const ExecutionConstraintSet exset =
	  runtime->find_execution_constraints(ctx, id, variants[i]);
	if(exset.processor_constraint.kind == kind)
	  return true;
      }
      return false; 
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::default_policy_select_must_epoch_processors(
                              MapperContext ctx,
                              const std::vector<std::set<const Task *> > &tasks,
                              Processor::Kind proc_kind,
                              std::map<const Task *, Processor> &target_procs)
    //--------------------------------------------------------------------------
    {
      // our default policy will be to try to spread must epoch tasks across all
      // address spaces as evenly as possible - tasks in the same subvector need
      // to go in the same address space because they have instances they both
      // want to access
      std::map<AddressSpaceID, std::deque<Processor> > as2proc;
      size_t n_procs = 0;
      {
	Machine::ProcessorQuery pq = Machine::ProcessorQuery(machine)
	  .only_kind(proc_kind);
	for(Machine::ProcessorQuery::iterator it = pq.begin();
	    it != pq.end();
	    ++it) {
	  as2proc[it->address_space()].push_back(*it);
	  n_procs++;
	}
      }

      // if we don't have enough processors, we can't satisfy this request
      size_t total_tasks = 0;
      for(std::vector<std::set<const Task *> >::const_iterator it = 
            tasks.begin(); it != tasks.end(); ++it)
	total_tasks += it->size();
      if(n_procs < total_tasks) {
	log_mapper.error() << "Default mapper error. Not enough procs of kind "
			   << proc_kind << " for must epoch launch with "
			   << total_tasks << " tasks.";
	return false;
      }

      // round-robin across the address spaces until we satisfy everyone - favor
      //  groups with larger sizes to maximize chance of satisfying things (we
      //  won't handle all cases - i.e. the bin packing problem)
      size_t n_left = total_tasks;
      size_t last_group_size = total_tasks + 1; // larger than any group
      while(n_left) {
	// first find the largest remaining group
	size_t group_size = 0;
	for(std::vector<std::set<const Task *> >::const_iterator it = 
              tasks.begin(); it != tasks.end(); ++it)
	  if((it->size() > group_size) && (it->size() < last_group_size))
	    group_size = it->size();
	last_group_size = group_size; // remember for next time around

	// now iterate over groups of the current-max-size and assign them 
        // round-robin to address spaces
	assert(group_size > 0);
	std::map<AddressSpaceID, std::deque<Processor> >::iterator 
          prev_as = as2proc.end();
	for(std::vector<std::set<const Task *> >::const_iterator it = 
              tasks.begin(); it != tasks.end(); ++it) {
	  if(it->size() != group_size) continue;

	  // first choice is the space after the one we used last time
	  std::map<AddressSpaceID, std::deque<Processor> >::iterator 
            curr_as = prev_as;
	  if(curr_as != as2proc.end())
	    ++curr_as;
	  if(curr_as == as2proc.end())
	    curr_as = as2proc.begin();
	  
	  // skip address spaces that are too small
	  if(curr_as->second.size() < group_size) {
	    std::map<AddressSpaceID, std::deque<Processor> >::iterator 
	      next_as = curr_as;
	    do {
	      if(next_as != as2proc.end())
		++next_as;
	      if(next_as == as2proc.end())
		next_as = as2proc.begin();
	      // if we wrap around, nothing is large enough and we're toast
	      if(next_as == curr_as) {
	      }
        
              log_mapper.error("must_epoch: no address space has enough "
                               "processors to fit a group of %d tasks!",
                                int(group_size));
	      assert(false);
	    } while(next_as->second.size() < group_size);
	    curr_as = next_as;
	  }
	  prev_as = curr_as;
	  
	  log_mapper.info() << "must_epoch: assigning " << group_size 
                            << " tasks to AS " << curr_as->first;

	  // assign tasks in this group to processors in this space
	  for(std::set<const Task *>::const_iterator it2 = it->begin();
	      it2 != it->end();
	      ++it2) {
	    target_procs[*it2] = curr_as->second.front();
	    curr_as->second.pop_front();
	    n_left--;
	  }
	}
      }
      return true;
    }

    //--------------------------------------------------------------------------
    Memory DefaultMapper::
      default_policy_select_constrained_instance_constraints(
                                  MapperContext ctx,
                                  const std::vector<const Task *> &tasks,
                                  const std::vector<unsigned> &req_indexes,
                                  const std::vector<Processor> &target_procs,
                                  const std::set<LogicalRegion> &needed_regions,
                                  const std::set<FieldID> &needed_fields,
                                  LayoutConstraintSet &constraints)
    //--------------------------------------------------------------------------
    {
      // go through the requirements of the various tasks and hope for exactly
      // one requirement that doesn't have NO_ACCESS
      std::vector<unsigned> accessing_task_idxs;
      for(unsigned i = 0; i < tasks.size(); i++) {
	if(tasks[i]->regions[req_indexes[i]].is_no_access())
	  continue;

	accessing_task_idxs.push_back(i);
      }

      // check for case of no tasks that want access
      if(accessing_task_idxs.empty()) {
        log_mapper.error("Must epoch has no tasks that require direct "
         "access to an instance - DefaultMapper doesn't know how to pick one.");
        assert(false);
      }

      // pick the first (or only) task as the "home task" - it'll be the one we
      //  ask for layout choices
      unsigned home_task_idx = accessing_task_idxs[0];
      Processor target_proc = target_procs[home_task_idx];
      Memory target_memory = default_policy_select_target_memory(ctx,
                      target_proc,
                      tasks[home_task_idx]->regions[req_indexes[home_task_idx]]);
      // if we have more than one task, double-check that this memory is kosher 
      // with the other ones too
      if(accessing_task_idxs.size() > 1) {
	for(size_t i = 1; i < accessing_task_idxs.size(); i++) {
	  Processor p2 = target_procs[accessing_task_idxs[i]];
	  if(!machine.has_affinity(p2, target_memory)) {
            log_mapper.error("Default Mapper Error.  Memory chosen for "
                             "constrained instance was %llu, but is not "
                             "visible to task on processor %llu",
                              target_memory.id, p2.id);
	    assert(false);
	  }
	}
      }

      // all layout constraints must be satisified for all accessing tasks
      for(std::vector<unsigned>::iterator it = accessing_task_idxs.begin(); 
	  it != accessing_task_idxs.end();
	  ++it) {
	VariantInfo info = default_find_preferred_variant(*(tasks[*it]), ctx, 
           true/*needs tight bound*/, true/*cache*/, target_procs[*it].kind());
        const TaskLayoutConstraintSet &tlc = 
          runtime->find_task_layout_constraints(ctx, tasks[*it]->task_id, 
                                                info.variant);
	std::multimap<unsigned,LayoutConstraintID>::const_iterator it2 = 
          tlc.layouts.lower_bound(req_indexes[*it]);
	while((it2 != tlc.layouts.end()) && (it2->first == req_indexes[*it])) {
	  const LayoutConstraintSet &req_cons = 
            runtime->find_layout_constraints(ctx, it2->second);
	  if(constraints.conflicts(req_cons)) {
            log_mapper.error("Default mapper error.  Layout constraint "
                             "violation in must_epoch instance creation.");
	    assert(false);
	  }
	  ++it2;
	}
      }
      return target_memory;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_must_epoch(const MapperContext           ctx,
                                       const MapMustEpochInput&      input,
                                             MapMustEpochOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_must_epoch in %s", get_mapper_name());

      // First get a set of distinct processors to target - first we need to go
      // through the tasks in the must epoch and split them by which kind of
      // processor they want
      std::map<Processor::Kind, std::vector<const Task *> > tasks_by_kind;
      for(unsigned i = 0; i < input.tasks.size(); i++) {
	// see which processor kinds are preferred, but filter by which ones 
        // have available variants and available processors
	std::vector<Processor::Kind> ranking;
	default_policy_rank_processor_kinds(ctx, *(input.tasks[i]), ranking);
	std::vector<Processor::Kind>::iterator it = ranking.begin();
	while(true) {
	  assert(it != ranking.end());
          // Check to see if we actually have processors of this kind
          switch (*it)
          {
            case Processor::TOC_PROC:
              {
                if (local_gpus.empty())
                {
                  ++it;
                  continue;
                }
                break;
              }
            case Processor::OMP_PROC:
              {
                if (local_omps.empty())
                {
                  ++it;
                  continue;
                }
                break;
              }
            case Processor::PROC_SET:
              {
                if (local_procsets.empty())
                {
                  ++it;
                  continue;
                }
                break;
              }
            case Processor::LOC_PROC:
              {
                assert(!local_cpus.empty());
                break;
              }
            case Processor::IO_PROC:
              {
                if (local_ios.empty())
                {
                  ++it;
                  continue;
                }
                break;
              }
            case Processor::PY_PROC:
              {
                if (local_pys.empty())
                {
                  ++it;
                  continue;
                }
                break;
              }
            default:
              assert(false); // unknown processor kind
          }
	  if(have_proc_kind_variant(ctx, input.tasks[i]->task_id, *it)) {
	    tasks_by_kind[*it].push_back(input.tasks[i]);
	    break;
	  }
	  ++it;
	}
      }
      // now try to satisfy each kind
      std::map<const Task*,Processor> proc_map;
      for(std::map<Processor::Kind,
               std::vector<const Task *> >::iterator it = tasks_by_kind.begin();
	  it != tasks_by_kind.end();
	  ++it) {
	// yuck - have to build "equivalence classes" of tasks based on whether 
        // they have any common region requirements that they both want access 
        // to the "best" algorithm would be a union-find approach on the task
	//  "common access" graph, but we'll go with a slightly less optimal
	//  insertion-sort-style algorithm
	std::vector<std::set<const Task *> > task_groups;
	for(std::vector<const Task *>::const_iterator it2 = it->second.begin();
	    it2 != it->second.end();
	    ++it2) {
	  const Task *t = *it2;
	  std::set<unsigned> merges;

	  for(std::vector<MappingConstraint>::const_iterator it3 = 
              input.constraints.begin(); it3 != input.constraints.end();++it3) {
	    const MappingConstraint &c = *it3;
	    std::set<unsigned> poss_merges;
	    bool need_merge = false;
	    for(unsigned idx = 0; idx < c.constrained_tasks.size(); idx++) {
	      const Task *t2 = c.constrained_tasks[idx];
	      // ignore any region requirement that is no_access
	      if(t2->regions[c.requirement_indexes[idx]].is_no_access())
		continue;
	      if(t == t2) {
		// we're one of the tasks, so we'll need to merge with anybody
		//  else that wants access
		need_merge = true;
	      } else {
		// see if this task is in one of the existing groups
		for(unsigned j = 0; j < task_groups.size(); j++)
		  if(task_groups[j].count(t2) > 0) {
		    poss_merges.insert(j);
		    break;
		  }
	      }
	    }
	    // add in the possible merges if we need them
	    if(need_merge && !poss_merges.empty())
	      merges.insert(poss_merges.begin(), poss_merges.end());
	  }

	  // three cases to deal with
	  if(merges.empty()) {
	    // case 1: no merges needed - start a new group
	    std::set<const Task *> ng;
	    ng.insert(t);
	    task_groups.push_back(ng);
	  } else if(merges.size() == 1) {
	    // case 2: merge needed with exactly one group 
            // - just add to that group
	    task_groups[*(merges.begin())].insert(t);
	  } else {
	    // case 3: need to merge multiple groups, so add ourselves to the 
            // first one and then combine all the rest into the first as well, 
            // working backwards so we can do vector::erase() calls
	    unsigned first = *(merges.begin());
	    task_groups[first].insert(t);
	    std::set<unsigned>::reverse_iterator it3 = merges.rbegin();
	    while(*it3 != first) {
	      task_groups[first].insert(task_groups[*it3].begin(),
					task_groups[*it3].end());
	      task_groups.erase(task_groups.begin() + *it3);
	    }
	  }
	}

#ifndef NDEBUG
	bool ok =
#endif
                  default_policy_select_must_epoch_processors(ctx,
							      task_groups,
							      it->first,
							      proc_map);
	assert(ok);
      }
      // everything's assigned, so copy the answers into the output
      for(unsigned i = 0; i < input.tasks.size(); i++) {
	assert(proc_map.count(input.tasks[i]) != 0);
	output.task_processors[i] = proc_map[input.tasks[i]];
      }

      // Now let's map the constraints, find one requirement to use for
      // mapping each of the constraints, but get the set of fields we
      // care about and the set of logical regions for all the requirements
      for (unsigned cid = 0; cid < input.constraints.size(); cid++)
      {
        const MappingConstraint &constraint = input.constraints[cid];
        std::vector<PhysicalInstance> &constraint_mapping = 
                                              output.constraint_mappings[cid];
        std::set<LogicalRegion> needed_regions;
        std::set<FieldID> needed_fields;
        for (unsigned idx = 0; idx < constraint.constrained_tasks.size(); idx++)
        {
          const Task *task = constraint.constrained_tasks[idx];
          unsigned req_idx = constraint.requirement_indexes[idx];
          needed_regions.insert(task->regions[req_idx].region);
          needed_fields.insert(task->regions[req_idx].privilege_fields.begin(),
                               task->regions[req_idx].privilege_fields.end());
        }

	// Now delegate to a policy routine to decide on a memory and layout
	// constraints for this constrained instance
	std::vector<Processor> target_procs;
	for(std::vector<const Task *>::const_iterator it = 
            constraint.constrained_tasks.begin();
	    it != constraint.constrained_tasks.end();
	    ++it)
	  target_procs.push_back(proc_map[*it]);
	LayoutConstraintSet layout_constraints;
	layout_constraints.add_constraint(FieldConstraint(needed_fields,
                                                false /*!contiguous*/));
	Memory mem = default_policy_select_constrained_instance_constraints(
				     ctx,
				     constraint.constrained_tasks,
				     constraint.requirement_indexes,
				     target_procs,
				     needed_regions,
				     needed_fields,
				     layout_constraints);

	LogicalRegion to_create = ((needed_regions.size() == 1) ?
  				     *(needed_regions.begin()) :
                       default_find_common_ancestor(ctx, needed_regions));
	PhysicalInstance inst;
	bool created;
	bool ok = runtime->find_or_create_physical_instance(ctx, mem,
            layout_constraints, std::vector<LogicalRegion>(1,to_create),
							    inst, created,
							    true /*acquire*/);
	assert(ok);
	if(!ok)
        {
          log_mapper.error("Default mapper error. Unable to make instance(s) "
			   "in memory " IDFMT " for index %d of constrained "
			   "task %s (ID %lld) in must epoch launch.",
			   mem.id, constraint.requirement_indexes[0],
			   constraint.constrained_tasks[0]->get_task_name(), 
			   constraint.constrained_tasks[0]->get_unique_id());
            assert(false);
	}
	constraint_mapping.push_back(inst);
      }
    }

    //--------------------------------------------------------------------------
    LogicalRegion DefaultMapper::default_find_common_ancestor(
                      MapperContext ctx, const std::set<LogicalRegion> &regions)
    //--------------------------------------------------------------------------
    {
      assert(!regions.empty());
      if (regions.size() == 1)
        return *(regions.begin());
      LogicalRegion result = LogicalRegion::NO_REGION;
      unsigned result_depth = 0;
      for (std::set<LogicalRegion>::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (!result.exists())
        {
          result = *it;
          result_depth = 
            runtime->get_index_space_depth(ctx, result.get_index_space());
          continue;
        }
        // Quick check to see if we are done
        if ((*it) == result)
          continue;
        // Get them to the same depth
        LogicalRegion next = *it;
        unsigned next_depth = 
          runtime->get_index_space_depth(ctx, next.get_index_space());
        while (next_depth > result_depth)
        {
          LogicalPartition part = 
            runtime->get_parent_logical_partition(ctx, next);
          next = runtime->get_parent_logical_region(ctx, part);
          next_depth -= 2;
        }
        while (result_depth > next_depth)
        {
          LogicalPartition part = 
            runtime->get_parent_logical_partition(ctx, result);
          result = runtime->get_parent_logical_region(ctx, part);
          result_depth -= 2;
        }
        // Make them both go up until you find the common ancestor
        while (result != next)
        {
          LogicalPartition next_part = 
            runtime->get_parent_logical_partition(ctx, next);
          next = runtime->get_parent_logical_region(ctx, next_part);
          LogicalPartition result_part = 
            runtime->get_parent_logical_partition(ctx, result);
          result = runtime->get_parent_logical_region(ctx, result_part);
          // still need to track result depth
          result_depth -= 2;
        }
      }
      return result;
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
                                             const StealRequestInput&    input,
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

