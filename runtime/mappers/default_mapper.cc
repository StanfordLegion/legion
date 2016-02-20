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
#include <algorithm>

#define STATIC_MAX_PERMITTED_STEALS   4
#define STATIC_MAX_STEAL_COUNT        2
#define STATIC_BREADTH_FIRST          false
#define STATIC_WAR_ENABLED            false 
#define STATIC_STEALING_ENABLED       false
#define STATIC_MAX_SCHEDULE_COUNT     8
#define STATIC_NUM_PROFILE_SAMPLES    1

// This is the default implementation of the mapper interface for 
// the general low level runtime

namespace Legion {
  namespace Mapping {

    using namespace Utilities;
    using namespace LegionRuntime::Arrays;

    LegionRuntime::Logger::Category log_mapper("default_mapper");

    enum MapperMessageType
    {
      INVALID_MESSAGE = 0,
      PROFILING_SAMPLE = 1,
    };

    struct MapperMsgHdr
    {
      MapperMsgHdr(void) : magic(0xABCD), type(INVALID_MESSAGE) { }
      bool is_valid_mapper_msg() const
      {
        return magic == 0xABCD && type != INVALID_MESSAGE;
      }
      uint32_t magic;
      MapperMessageType type;
    };

    struct ProfilingSampleMsg : public MapperMsgHdr
    {
      ProfilingSampleMsg(void) : MapperMsgHdr(), task_id(0) { }
      Processor::TaskFuncID task_id;
      MappingProfiler::Profile sample;
    };

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
        war_enabled(STATIC_WAR_ENABLED),
        stealing_enabled(STATIC_STEALING_ENABLED),
        max_schedule_count(STATIC_MAX_SCHEDULE_COUNT),
        machine_interface(Utilities::MachineQueryInterface(m))
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Initializing the default mapper for "
                            "processor " IDFMT "",
                 local_proc.id);
      // Check to see if there any input arguments to parse
      {
        int argc = HighLevelRuntime::get_input_args().argc;
        char **argv = HighLevelRuntime::get_input_args().argv;
        unsigned num_profiling_samples = STATIC_NUM_PROFILE_SAMPLES;
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
          BOOL_ARG("-dm:war", war_enabled);
          BOOL_ARG("-dm:steal", stealing_enabled);
          BOOL_ARG("-dm:bft", breadth_first_traversal);
          INT_ARG("-dm:sched", max_schedule_count);
          INT_ARG("-dm:prof",num_profiling_samples);
#undef BOOL_ARG
#undef INT_ARG
        }
        profiler.set_needed_profiling_samples(num_profiling_samples);
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
      for (int i = 0; i < short_bits; i++)
        short_mask |= (1LL << i);
      for (int i = 0; i < 3; i++)
        random_number_generator[i] = (unsigned short)((local_proc.id & 
                            (short_mask << (i*short_bits))) >> (i*short_bits));
    }

    //--------------------------------------------------------------------------
    long DefaultMapper::generate_random_integer(void) const
    //--------------------------------------------------------------------------
    {
      return nrand48(random_number_generator);
    }
    
    //--------------------------------------------------------------------------
    double DefaultMapper::generate_random_real(void) const
    //--------------------------------------------------------------------------
    {
      return erand48(random_number_generator);
    }

    //--------------------------------------------------------------------------
    DefaultMapper::DefaultMapper(const DefaultMapper &rhs)
      : Mapper(), local_proc(Processor::NO_PROC),
        local_kind(Processor::LOC_PROC), node_id(0), machine(rhs.machine),
        mapper_name(NULL), machine_interface(Utilities::MachineQueryInterface(
              Machine::get_machine()))
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
    Mapper::MapperSyncModel Mapper::get_mapper_sync_model(void) const
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
      output.initial_proc = default_select_initial_processor(task, ctx);
      output.inline_task = false;
      output.stealable = stealing_enabled; 
      output.map_locally = true;
    }

    //--------------------------------------------------------------------------
    Processor DefaultMapper::default_select_initial_processor(const Task &task,
                                                              MapperContext ctx)
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
      const int index = generate_random_integer() % total_procs;
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
      find_valid_variants(ctx, task.task_id, variants);
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
            rank_processor_kinds(task, ranking);
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
            find_valid_variants(ctx, task.task_id, variants, ranking[idx]);
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
          find_valid_variants(ctx, task.task_id, variants, best_kind);
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
              &(find_execution_constraints(ctx, best_variant));
            const TaskLayoutConstraintSet *best_layout_constraints = 
              &(find_layout_constraints(ctx, best_variant));
            for (unsigned idx = 1; idx < variants.size(); idx++)
            {
              const ExecutionConstraintSet &next_execution_constraints = 
                find_execution_constraints(ctx, variants[idx]);
              const TaskLayoutConstraintSet &next_layout_constraints = 
                find_layout_constraints(ctx, variants[idx]);
              VariantID chosen = select_best_variant(task, 
                  best_kind, best_variant, variants[idx],
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
          preferred_variants[task.task_id] = result;
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
    void DefaultMapper::rank_processor_kinds(const Task &task,
                                          std::vector<Processor::Kind> &ranking)
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
    VariantID DefaultMapper::select_best_variant(const Task &task, 
                                      Processor::Kind proc_kind,
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
                Point<1> blocking_factor(1/*splitting factor*/);
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
                const int factors[2] = { 1, 1 };
                Point<2> blocking_factor(factors);
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
                const int factors[3] = { 1, 1, 1 };
                Point<3> blocking_factor(factors);
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
      Rect<DIM> blocks = blocking.image_convex(point_rect);
      unsigned next_index = 0;
      bool is_perfect = true;
      for (int idx = 0; idx < DIM; idx++) {
        if ((point_rect.dim_size(idx) % blocking_factor[idx]) != 0) {
          is_perfect = false;
          break;
        }
      }
      if (is_perfect)
      {
        slices.resize(blocks.volume());
        for (typename Blockify<DIM>::PointInOutputRectIterator pir(slices);
              pir; pir++, next_index++)
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
        for (typename Blockify<DIM>::PointInOutputRectIterator pir(slices); 
              pir; pir++)
        {
          Rect<DIM> upper_bound = blocking.preimage(pir.p);
          // Check for edge cases with intersections
          Rect<DIM> slice_points = point_rect.intersection(upper_bound);
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

    //--------------------------------------------------------------------------
    template<int DIM>
    /*static*/ Point<DIM> DefaultMapper::default_select_blocking_factor( 
                                         int factor, const Rect<DIM> &to_factor)
    //--------------------------------------------------------------------------
    {
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
              output.target_procs.insert(output.target_procs.end(),
                  local_cpus.begin(), local_cpus.end());
              break;
            }
          case Processor::IO_PROC:
            {
              // Put any of our I/O procs here
              output.target_procs.insert(output.target_procs.end(),
                  local_ios.begin(), local_ios.end());
              break;
            }
          default:
            assert(false); // unrecognized processor kind
        }
      }
      else
        output.target_procs.push_back(task.target_proc);
      // First, let's see if we've cached a result of this task mapping
      const unsigned long long task_hash = compute_task_hash(task);
      std::pair<TaskID,Processor> cache_key(task.task_id, task.target_proc);
      std::map<std::pair<TaskID,Processor>,
               std::list<CachedTaskMapping> >::iterator 
        finder = cached_task_mappings.find(cache_key);
      if (finder != cached_task_mappings.end())
      {
        bool found = false;
        // Iterate through and see if we can find one with our variant and hash
        for (std::list<CachedTaskMapping>::const_iterator it = 
              finder->second.begin(); finder->second.end(); it++)
        {
          if ((it->variant == output.chosen_variant) &&
              (it->task_hash == task_hash))
          {
            // Have to copy it before we do the external call which 
            // might invalidate our iterator
            output.chosen_instances = it->mapping;
            found = true;
            break;
          }
        }
        if (found)
        {
          // See if we can acquire these instances still
          if (acquire_instances(ctx, output.chosen_instances))
            return;
          // If some of them were deleted, go back and remove this entry
          // Have to renew our iterators since they might have been
          // invalidated during the 'acquire_instances' call
          finder = cached_task_mappings.find(cache_key);
          if (finder != cached_task_mappings.end())
          {
            for (std::list<CachedTaskMapping>::const_iterator it = 
                  finder->second.begin(); finder->second.end(); it++)
            {
              if ((it->variant == output.chosen_variant) &&
                  (it->task_hash == task_hash))
              {
                finder->second.erase(it);
                break;
              }
            }
            if (finder->second.empty())
              cached_task_mappings.erase(finder);
          }
        }
      }
      // We didn't find a cached version of the mapping so we need to 
      // do a full mapping, we already know what variant we want to use
      // so let's use one of the acceleration functions to figure out
      // which instances still need to be mapped.



      // Track which regions have already been mapped 
      std::vector<bool> done_regions(task.regions.size(), false);
      if (!input.premapped_regions.empty())
        for (std::vector<unsigned>::const_iterator it = 
              input.premapped_regions.begin(); it != 
              input.premapped_regions.end(); it++)
          done_regions[*it] = true;
      // Our algorithm here is simple, first try to just map this task
      // using any existing instances. If there are none, then we'll 
      // immediately break out. We'll then try to validate the mapping
      // with the runtime. If it works, we're done, otherwise, we'll
      // do the slower mapping process of making instances that match
      // the requirements of the given instance.
      bool try_validation = true;
      for (unsigned idx = 0; idx < task.regions.size(); idx++)
      {
        // Skip any premapped regions
        if (done_regions[idx])
          continue;
        // Skip any empty regions
        if ((task.regions[idx].privilege == NO_ACCESS) ||
            (task.regions[idx].privilege_fields.empty()))
        {
          done_regions[idx] = true; // we know we're done here
          continue;
        }
        const std::vector<PhysicalInstance> &valid = input.valid_instances[idx];
        if (valid.empty()) {
          // Check for reduction, in which case we always just make an instance
          if (task.regions[idx].privilege == REDUCE)
          {
            default_create_reduction_instance(ctx, task.target_proc,
                task.regions[idx], output.chosen_instances[idx]);
            done_regions[idx] = true; // this is now good
            continue;
          }
          // No valid instances, so we need to do the more complex mapping 
          try_validation = false;
          // Clear out anything we don't know doesn't work
          for (unsigned idx2 = 0; idx2 < idx; idx2++)
          {
            if (done_regions[idx2])
              continue;
            output.chosen_instances[idx2].clear();
          }
        } else {
          // if we know we're not going to try to validate yet keep going
          if (!try_validation)
            continue;
          // Otherwise, just copy the valid instances over
          output.chosen_instances[idx] = valid;
        }
      }
      if (try_validation)
      {
        std::vector<ValidationError> region_errors;
        ValidationError error = 
          validate_task_mapping(ctx, output, region_errors); 
        // If we're all good, then we're done
        if (error == NO_VALIDATION_ERROR)
          return;
        // See what kind of validation failures we have, some of them
        // just mean we have to do some more sophisticated mapping, 
        // others are indicative of bugs in the mapper itself
        if (error & (ILLEGAL_VARIANT_ERROR | MIXED_PROCESSOR_ERROR))
        {
          log_mapper.error("Default mapper bug. Failed task mapping of "
                           "task %s with error %x. Time to bust out the "
                           "debugger.", task.get_task_name(), error);
          assert(false);
        }
        // See which instances failed, mark the ones that succeeded
        // as premapped so that we don't try to do them again
        for (unsigned idx = 0; idx < region_errors.size(); idx++)
        {
          if (region_errors[idx] == NO_VALIDATION_ERROR)
          {
            done_regions[idx] = true;
            continue;
          }
          output.chosen_instances[idx].clear();
        }
      }
      // Now let's map all the remaining regions the hard way 
      const TaskLayoutConstraintSet &layout_constraints = 
        find_layout_constraints(ctx, output.chosen_variant);
      for (unsigned idx = 0; idx < task.regions.size(); idx++)
      {
        if (done_regions[idx])
          continue;
        default_create_custom_instance(ctx, task.target_proc, idx,
          task.regions[idx], layout_constraints, output.chosen_instances[idx]);
      }
      // The runtime is going to do this check anyway, might as well
      // do it ourselves to find anything wrong now
      std::vector<ValidationError> region_errors;
      ValidationError error = validate_task_mapping(ctx, output, region_errors);
      if (error != NO_VALIDATION_ERROR)
      {
        log_mapper.error("Default mapper bug. Failed to do custom task mapping "
                         "of task %s with error %x. Time to bust out the "
                         "debugger.", task.get_task_name(), error);
        assert(false);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ unsigned long long DefaultMapper::compute_task_hash(
                                                               const Task &task)
    //--------------------------------------------------------------------------
    {
      // Use Sean's "cheesy" hash function    
      const unsigned long long c1 = 0x5491C27F12DB3FA4;
      const unsigned long long c2 = 353435096;
      // We have to hash all region requirements including region names,
      // privileges, coherence modes, reduction operators, and fields
      unsigned long long result = c2 + task->task_id;
      for (unsigned idx = 0; idx < task->regions.size(); idx++)
      {
        const RegionRequirement &req = task->regions[idx];
        result = result * c1 + c2 + req.handle_type;
        if (req.handle_type != PART_PROJECTION) {
          result = result * c1 + c2 + req.region.get_tree_id();
          result = result * c1 + c2 + req.region.get_index_space().get_id();
          result = result * c1 + c2 + req.region.get_field_space().get_id();
        } else {
          result = result * c1 + c2 + req.region.get_tree_id();
          result = result * c1 + c2 + req.region.get_index_partition().get_id();
          result = result * c1 + c2 + req.region.get_field_space().get_id();
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
    void DefaultMapper::default_create_custom_instance(MapperContext ctx,
                          Processor target, unsigned index,
                          const RegionRequirement &req,
                          const TaskLayoutConstraintSet &layout_constraints,
                          std::vector<PysicalInstance> &destination)
    //--------------------------------------------------------------------------
    {
      std::set<FieldID> unhandled_fields = req.privilege_fields;
      // Iterate over the constraints for this region requirement
      for (std::multimap<unsigned,LayoutConstraintID>::const_iterator it = 
            layout_constraints.layouts.lower_bound(); it != 
            layout_constraints.layouts.upper_bound(); it++)
      {
        // Check to see if this set of constraints conflicts with
        // the goal layout of the default mapper, if it does, then
        // we'll just use these constraints, otherwise, we'll 
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::default_create_reduction_instance(MapperContext ctx,
                          Processor target, const RegionRequirement &req,
                          std::vector<PhysicalInstance> &destination)
    //--------------------------------------------------------------------------
    {

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

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_inline(const MapperContext        ctx,
                                   const InlineMapping&       inline_op,
                                   const MapInlineInput&      input,
                                         MapInlineOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_inline in %s", get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_inline_sources(const MapperContext     ctx,
                                         const InlineMapping&         inline_op,
                                         const SelectInlineSrcInput&  input,
                                               SelectInlineSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_inline_sources in %s", get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::report_profiling(const MapperContext         ctx,
                                         const InlineMapping&        inline_op,
                                         const InlineProfilingInfo&  input)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default report_profiling for Inline in %s", 
                      get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_copy(const MapperContext      ctx,
                                 const Copy&              copy,
                                 const MapCopyInput&      input,
                                       MapCopyOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_copy in %s", get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_copy_sources(const MapperContext          ctx,
                                            const Copy&                  copy,
                                            const SelectCopySrcInput&    input,
                                                  SelectCopySrcOutput&   output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_copy_sources in %s", get_mapper_name());

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

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_close(const MapperContext       ctx,
                                  const Close&              close,
                                  const MapCloseInput&      input,
                                        MapCloseOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_close in %s", get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_close_sources(const MapperContext        ctx,
                                             const Close&               close,
                                             const SelectCloseSrcInput&  input,
                                                   SelectCloseSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_close_sources in %s", get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::report_profiling(const MapperContext       ctx,
                                         const Close&              close,
                                         const CloseProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default report_profiling for Close in %s", 
                      get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_acquire(const MapperContext         ctx,
                                    const Acquire&              acquire,
                                    const MapAcquireInput&      input,
                                          MapAcquireOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_acquire in %s", get_mapper_name());

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

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_release(const MapperContext         ctx,
                                    const Release&              release,
                                    const MapReleaseInput&      input,
                                          MapReleaseOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_release in %s", get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_release_sources(const MapperContext      ctx,
                                         const Release&                 release,
                                         const SelectReleaseSrcInput&   input,
                                               SelectReleaseSrcOutput&  output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_release_sources in %s",get_mapper_name());

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

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::configure_context(const MapperContext         ctx,
                                          const Task&                 task,
                                                ContextConfigOutput&  output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default configure_context in %s", get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_tunable_value(const MapperContext         ctx,
                                             const Task&                 task,
                                             const SelectTunableInput&   input,
                                                   SelectTunableOutput&  output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default select_tunable_value in %s", get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_must_epoch(const MapperContext           ctx,
                                       const MapMustEpochInput&      input,
                                             MapMustEpochOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_must_epoch in %s", get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::map_dataflow_graph(const MapperContext           ctx,
                                           const MapDataflowGraphInput&  input,
                                                 MapDataflowGraphOutput& output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default map_dataflow_graph in %s", get_mapper_name());

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
        unsigned max_depth = 0;
        for (std::list<const Task*>::const_iterator it = 
              input.ready_tasks.begin(); it != input.ready_tasks.end(); it++)
        {
          unsigned depth = (*it)->get_depth();
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

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::permit_steal_request(const MapperContext         ctx,
                                             const StealRequestInput&    intput,
                                                   StealRequestOutput&   output)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default permit_steal_request in %s", get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::handle_message(const MapperContext           ctx,
                                       const MapperMessage&          message)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default handle_message in %s", get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::handle_task_result(const MapperContext           ctx,
                                           const MapperTaskResult&       result)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Default handle task result in %s", get_mapper_name());

    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_task_options(Task *task)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Select task options in default mapper " 
                            "for proecessor " IDFMT "", local_proc.id);
      task->inline_task = false;
      task->spawn_task = stealing_enabled;
      task->map_locally = false; 
      task->profile_task = !profiler.profiling_complete(task);
      task->task_priority = 0; // No prioritization
      // For selecting a target processor see if we have finished profiling
      // the given task otherwise send it to a processor of the right kind
      if (profiler.profiling_complete(task))
      {
        Processor::Kind best_kind = profiler.best_processor_kind(task);
        // If our local processor is the right kind then do that
        if (best_kind == local_kind)
          task->target_proc = local_proc;
        else
        {
          // Otherwise select a random processor of the right kind
          std::set<Processor> all_procs;
	  machine.get_all_processors(all_procs);
          task->target_proc = select_random_processor(all_procs, best_kind, 
                                                      machine);
        }
      }
      else
      {
        // Get the next kind to find
        Processor::Kind next_kind = profiler.next_processor_kind(task);
        if (next_kind == local_kind)
          task->target_proc = local_proc;
        else
        {
          std::set<Processor> all_procs;
	  machine.get_all_processors(all_procs);
          task->target_proc = select_random_processor(all_procs, 
                                                      next_kind, machine);
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::target_task_steal(const std::set<Processor> &blacklist,
                                          std::set<Processor> &targets)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Target task steal in default mapper for "
                            "processor " IDFMT "",local_proc.id);
      if (stealing_enabled)
      {
        // Choose a random processor from our group that is not on the blacklist
        std::set<Processor> diff_procs; 
        std::set<Processor> all_procs;
	machine.get_all_processors(all_procs);
        // Remove ourselves
        all_procs.erase(local_proc);
        std::set_difference(all_procs.begin(),all_procs.end(),
                            blacklist.begin(),blacklist.end(),
                            std::inserter(diff_procs,diff_procs.end()));
        if (diff_procs.empty())
          return;
        unsigned index = (lrand48()) % (diff_procs.size());
        for (std::set<Processor>::const_iterator it = diff_procs.begin();
              it != diff_procs.end(); it++)
        {
          if (!index--)
          {
            log_mapper.spew("Attempting a steal from processor " IDFMT
                                  " on processor " IDFMT "",
                                  local_proc.id,it->id);
            targets.insert(*it);
            break;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::permit_task_steal(Processor thief,
                                          const std::vector<const Task*> &tasks,
                                          std::set<const Task*> &to_steal)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Permit task steal in default mapper for "
                            "processor " IDFMT "",local_proc.id);

      if (stealing_enabled)
      {
        // First see if we're even allowed to steal anything
        if (max_steals_per_theft == 0)
          return;
        // We're allowed to steal something, go through and find a task to steal
        unsigned total_stolen = 0;
        for (std::vector<const Task*>::const_iterator it = tasks.begin();
              it != tasks.end(); it++)
        {
          if ((*it)->steal_count < max_steal_count)
          {
            log_mapper.debug("Task %s (ID %lld) stolen from "
                                   "processor " IDFMT " by processor " IDFMT "",
                                   (*it)->variants->name, 
                                   (*it)->get_unique_task_id(), 
                                   local_proc.id, thief.id);
            to_steal.insert(*it);
            total_stolen++;
            // Check to see if we're done
            if (total_stolen == max_steals_per_theft)
              return;
            // If not, do locality aware task stealing, try to steal other 
            // tasks that use the same logical regions.  Don't need to 
            // worry about all the tasks we've already seen since we 
            // either stole them or decided not for some reason
            for (std::vector<const Task*>::const_iterator inner_it = it;
                  inner_it != tasks.end(); inner_it++)
            {
              // Check to make sure this task hasn't 
              // been stolen too much already
              if ((*inner_it)->steal_count >= max_steal_count)
                continue;
              // Check to make sure it's not one of 
              // the tasks we've already stolen
              if (to_steal.find(*inner_it) != to_steal.end())
                continue;
              // If its not the same check to see if they have 
              // any of the same logical regions
              for (std::vector<RegionRequirement>::const_iterator reg_it1 = 
                    (*it)->regions.begin(); reg_it1 != 
                    (*it)->regions.end(); reg_it1++)
              {
                bool shared = false;
                for (std::vector<RegionRequirement>::const_iterator reg_it2 = 
                      (*inner_it)->regions.begin(); reg_it2 != 
                      (*inner_it)->regions.end(); reg_it2++)
                {
                  // Check to make sure they have the same type of region 
                  // requirement, and that the region (or partition) 
                  // is the same.
                  if (reg_it1->handle_type == reg_it2->handle_type)
                  {
                    if (((reg_it1->handle_type == SINGULAR) || 
                          (reg_it1->handle_type == REG_PROJECTION)) &&
                        (reg_it1->region == reg_it2->region))
                    {
                      shared = true;
                      break;
                    }
                    if ((reg_it1->handle_type == PART_PROJECTION) &&
                        (reg_it1->partition == reg_it2->partition))
                    {
                      shared = true;
                      break;
                    }
                  }
                }
                if (shared)
                {
                  log_mapper.debug("Task %s (ID %lld) stolen from "
                                         "processor " IDFMT " by processor " 
                                         IDFMT "",
                                         (*inner_it)->variants->name, 
                                         (*inner_it)->get_unique_task_id(), 
                                         local_proc.id, thief.id);
                  // Add it to the list of steals and either return or break
                  to_steal.insert(*inner_it);
                  total_stolen++;
                  if (total_stolen == max_steals_per_theft)
                    return;
                  // Otherwise break, onto the next task
                  break;
                }
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::pre_map_task(Task *task)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Pre-map task in default mapper for task %s "
                            "(ID %lld) for processor " IDFMT "",
                            task->variants->name,
                            task->get_unique_task_id(), local_proc.id);
      for (unsigned idx = 0; idx < task->regions.size(); idx++)
      {
        if (task->regions[idx].must_early_map)
        {
          task->regions[idx].virtual_map = false;
          task->regions[idx].early_map = true;
          task->regions[idx].enable_WAR_optimization = war_enabled;
          task->regions[idx].reduction_list = false;
          task->regions[idx].make_persistent = false;
          // Elliott needs SOA for the compiler.
          task->regions[idx].blocking_factor = // 1;
            task->regions[idx].max_blocking_factor;

	  // respect restricted regions' current placement
	  if (task->regions[idx].restricted)
	  {
	    assert(task->regions[idx].current_instances.size() == 1);
	    task->regions[idx].target_ranking.push_back(
	      (task->regions[idx].current_instances.begin())->first);
	  }
	  else
	  {
	    Memory global = machine_interface.find_global_memory();
	    assert(global.exists());
	    task->regions[idx].target_ranking.push_back(global);
	  }
        }
        else
          task->regions[idx].early_map = false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_task_variant(Task *task)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Select task variant in default mapper for task %s "
                            "(ID %lld) for processor " IDFMT "",
                            task->variants->name,
                            task->get_unique_task_id(), local_proc.id);
      Processor::Kind target_kind = task->target_proc.kind();
      if (!task->variants->has_variant(target_kind, 
            !(task->is_index_space), task->is_index_space))
      {
        log_mapper.error("Mapper unable to find variant for "
                               "task %s (ID %lld)",
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
    bool DefaultMapper::map_task(Task *task)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Map task in default mapper for task %s "
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
						(task->target_proc.kind()
                                                        == Processor::LOC_PROC));
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
    void DefaultMapper::post_map_task(Task *task)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Post map task in default mapper for task %s "
                      "(ID %lld) for processor " IDFMT "",
                      task->variants->name,
                      task->get_unique_task_id(), local_proc.id);
      // Do nothing for now
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::map_copy(Copy *copy)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Map copy for copy ID %lld in default mapper "
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
        else
        {
          // There is only one choice anyway, so let the runtime find
          // it. Currently, the runtime will fail to notify the mapper
          // of existing instances for reduction copies, so this
          // assertion may fail spuriously.

          // assert(copy->src_requirements[idx].current_instances.size() == 1);
          // Memory target =
          //   (copy->src_requirements[idx].current_instances.begin())->first;
          // copy->src_requirements[idx].target_ranking.push_back(target);
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
        else
        {
          // There is only one choice anyway, so let the runtime find
          // it. Currently, the runtime will fail to notify the mapper
          // of existing instances for reduction copies, so this
          // assertion may fail spuriously.

          // assert(copy->dst_requirements[idx].current_instances.size() == 1);
          // Memory target =
          //   (copy->dst_requirements[idx].current_instances.begin())->first;
          // copy->dst_requirements[idx].target_ranking.push_back(target);
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
    bool DefaultMapper::map_inline(Inline *inline_op)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Map inline for operation ID %lld in default "
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
    bool DefaultMapper::map_must_epoch(const std::vector<Task*> &tasks,
                             const std::vector<MappingConstraint> &constraints,
                             MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Map must epoch in default mapper for processor"
                            " " IDFMT " ",
                            local_proc.id);
      // First fixup any target processors to ensure that they are all
      // pointed at different processors.  We know for now that all must epoch
      // tasks need to be running on CPUs so get the set of CPU processors.
      const std::set<Processor> &all_cpus = 
        machine_interface.filter_processors(Processor::LOC_PROC);
      assert(all_cpus.size() >= tasks.size());
      // Round robing the tasks onto the processors
      std::set<Processor>::const_iterator proc_it = all_cpus.begin();
      for (std::vector<Task*>::const_iterator it = tasks.begin();
            it != tasks.end(); it++, proc_it++)
      {
        (*it)->target_proc = *proc_it;
      }
      // Map all the tasks like normal, then go through and fix up the
      // mapping requests based on constraints.
      for (std::vector<Task*>::const_iterator it = tasks.begin();
            it != tasks.end(); it++)
      {
        map_task(*it);
      }
      // For right now, we'll put everything in the global memory
      Memory global_mem = machine_interface.find_global_memory();
      assert(global_mem.exists());
      for (std::vector<MappingConstraint>::const_iterator it =
            constraints.begin(); it != constraints.end(); it++)
      {
        it->t1->regions[it->idx1].target_ranking.clear();
        it->t1->regions[it->idx1].target_ranking.push_back(global_mem);
        it->t2->regions[it->idx2].target_ranking.clear();
        it->t2->regions[it->idx2].target_ranking.push_back(global_mem);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::notify_mapping_result(const Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      UniqueID uid = mappable->get_unique_mappable_id();
      // We should only get this for tasks in the default mapper
      if (mappable->get_mappable_kind() == Mappable::TASK_MAPPABLE)
      {
        const Task *task = mappable->as_mappable_task();
        assert(task != NULL);
        log_mapper.spew("Notify mapping for task %s (ID %lld) in "
                              "default mapper for processor " IDFMT "",
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
    void DefaultMapper::notify_mapping_failed(const Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      UniqueID uid = mappable->get_unique_mappable_id(); 
      log_mapper.warning("Notify failed mapping for operation ID %lld "
                      "in default mapper for processor " IDFMT "! Retrying...",
                       uid, local_proc.id);
      std::map<UniqueID,unsigned>::iterator finder = failed_mappings.find(uid);
      if (finder == failed_mappings.end())
        failed_mappings[uid] = 1;
      else
      {
        finder->second++;
        if (finder->second == max_failed_mappings)
        {
          log_mapper.error("Reached maximum number of failed mappings "
                                 "for operation ID %lld in default mapper for "
                                 "processor " IDFMT "!  Try implementing a "
                                 "custom mapper or changing the size of the "
                                 "memories in the low-level runtime. "
                                 "Failing out ...", uid, local_proc.id);
          assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::rank_copy_targets(const Mappable *mappable,
                                     LogicalRegion rebuild_region,
                                     const std::set<Memory> &current_instances,
                                     bool complete,
                                     size_t max_blocking_factor,
                                     std::set<Memory> &to_reuse,
                                     std::vector<Memory> &to_create,
                                     bool &create_one, size_t &blocking_factor)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Rank copy targets for mappable (ID %lld) in "
                            "default mapper for processor " IDFMT "",
                            mappable->get_unique_mappable_id(), local_proc.id);
      if (current_instances.empty())
      {
        // Pick the global memory
        Memory global = machine_interface.find_global_memory();
        if (!global.exists())
        {
          bool found = false;
          std::vector<Memory> stack;
          machine_interface.find_memory_stack(local_proc, stack,
              local_proc.kind() == Processor::LOC_PROC);
          // If there is no global memory, try finding RDMA memory
          for (unsigned idx = 0; idx < stack.size() && !found; ++idx)
            if (stack[idx].kind() == Memory::REGDMA_MEM)
            {
              global = stack[idx];
              found = true;
            }
          // If failed, try using system memory
          for (unsigned idx = 0; idx < stack.size() && !found; ++idx)
            if (stack[idx].kind() == Memory::SYSTEM_MEM)
            {
              global = stack[idx];
              found = true;
            }
          assert(true);
        }
        to_create.push_back(global);
        // Only make one new instance
        create_one = true;
        blocking_factor = max_blocking_factor; // 1
      }
      else
      {
        to_reuse.insert(current_instances.begin(),current_instances.end());
        create_one = false;
        blocking_factor = max_blocking_factor; // 1
      }
      // Don't make any composite instances since they're 
      // not fully supported yet
      return false;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::rank_copy_sources(const Mappable *mappable,
                                      const std::set<Memory> &current_instances,
                                      Memory dst_mem, 
                                      std::vector<Memory> &chosen_order)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Select copy source in default mapper for "
                            "processor " IDFMT "", local_proc.id);
      // Handle the simple case of having the destination 
      // memory in the set of instances 
      if (current_instances.find(dst_mem) != current_instances.end())
      {
        chosen_order.push_back(dst_mem);
        return;
      }

      machine_interface.find_memory_stack(dst_mem, 
                                          chosen_order, true/*latency*/);
      if (chosen_order.empty())
      {
        // This is the multi-hop copy because none 
        // of the memories had an affinity
        // SJT: just send the first one
        if(current_instances.size() > 0) {
          chosen_order.push_back(*(current_instances.begin()));
        } else {
          assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::notify_profiling_info(const Task *task)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Notify profiling info for task %s (ID %lld) in "
                            "default mapper for processor " IDFMT "",
                            task->variants->name, 
                            task->get_unique_task_id(), task->target_proc.id);
      memoizer.commit_mapping(task->target_proc, task);
      MappingProfiler::Profile sample;
      sample.execution_time = task->stop_time - task->start_time;
      sample.target_processor = task->target_proc;
      sample.index_point = task->index_point;

      profiler.add_profiling_sample(task->task_id, sample);
      if (profiler.get_profiling_option(task->task_id).gather_in_orig_proc &&
          task->target_proc != task->orig_proc)
      {
        ProfilingSampleMsg msg;
        msg.type = PROFILING_SAMPLE;
        msg.task_id = task->task_id;
        msg.sample = sample;
        send_message(task->orig_proc, &msg, sizeof(msg));
      }
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::speculate_on_predicate(const Mappable *mappable, 
                                               bool &spec_value)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Speculate on predicate for mappable (ID %lld) in "
                            "default mapper for processor " IDFMT "",
                            mappable->get_unique_mappable_id(),
                            local_proc.id);
      // While the runtime supports speculation, it currently doesn't
      // know how to roll back from mis-speculation, so for the moment
      // we don't speculate.
      return false;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::configure_context(Task *task)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Configure context for task %s (ID %lld) in "
                            "default mapper for processor " IDFMT "",
                            task->variants->name, 
                            task->get_unique_task_id(), task->target_proc.id);
      // Do nothing so we just use the preset defaults
    }

    //--------------------------------------------------------------------------
    int DefaultMapper::get_tunable_value(const Task *task, TunableID tid,
                                         MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Get tunable value for task %s (ID %lld) in "
                            "default mapper for processor " IDFMT "",
                            task->variants->name,
                            task->get_unique_task_id(), task->target_proc.id);
      // For right now the default mapper doesn't know how to guess
      // for tunable variables, so instead simply assert.  In the future
      // we might consider employing a performance profiling directed
      // approach to guessing for tunable variables.
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::handle_message(Processor source,
                                       const void *message, size_t length)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Handle message in default mapper for processor " 
                            IDFMT "", local_proc.id);
      const MapperMsgHdr* header = reinterpret_cast<const MapperMsgHdr*>(message);
      if (header->is_valid_mapper_msg())
      {
        switch (header->type)
        {
          case PROFILING_SAMPLE:
            {
              const ProfilingSampleMsg* msg =
                reinterpret_cast<const ProfilingSampleMsg*>(message);
              profiler.add_profiling_sample(msg->task_id, msg->sample);
              break;
            }
          default:
            {
              // this should not happen
              assert(false);
              break;
            }
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::handle_mapper_task_result(MapperEvent event,
                                                  const void *result,
                                                  size_t result_size)
    //--------------------------------------------------------------------------
    {
      log_mapper.spew("Handle mapper task result in default mapper "
                            "for processor " IDFMT "", local_proc.id);
      // We don't launch any sub tasks so we should never receive a result
      assert(false);
    }

  }; // namespace Mapping
}; // namespace Legion

