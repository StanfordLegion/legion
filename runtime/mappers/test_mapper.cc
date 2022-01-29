/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#include "mappers/test_mapper.h"
#include <sys/time.h>

#define INLINE_RATIO    100 // 1 in 100
#define REAL_CLOSE_RATIO 100 // 1 in 100 will make a real instance
#define COPY_VIRTUAL_RATIO 10 // 1 in 10 copies will make a virtual source
#define INNER_VIRTUAL_RATIO 2 // 1 in 2 virtual inner task region requirements
#define REDISTRIBUTE_RATIO 10 // 1 in 10 we will send a task to a new processor
#define STEAL_PERMIT_RATIO 2 // 1 in 2 chance a steal will succeed

namespace Legion {
  namespace Mapping {

    Logger log_test_mapper("test_mapper");

    //--------------------------------------------------------------------------
    /*static*/ const char* TestMapper::create_test_name(Processor p)
    //--------------------------------------------------------------------------
    {
      const size_t buffer_size = 64;
      char *result = (char*)malloc(buffer_size*sizeof(char));
      snprintf(result, buffer_size-1,
                "Test Mapper on Processor " IDFMT "", p.id);
      return result;
    }

    //--------------------------------------------------------------------------
    TestMapper::TestMapper(MapperRuntime *rt, Machine m, 
                           Processor local, const char *name)
      : DefaultMapper(rt,m,local,(name == NULL) ? create_test_name(local) :name)
    //--------------------------------------------------------------------------
    {
      // Check to see if there any input arguments to parse
      long long seed = -1;
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
          INT_ARG("-tm:seed", seed);
#undef INT_ARG
        }
      }
      if (seed == -1)
      {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        seed = ((long long)tv.tv_sec) * 1000 + ((long long)tv.tv_usec) / 1000;
      }
      // Initialize our random number generator
      const size_t short_bits = 8*sizeof(unsigned short);
      long long short_mask = 0;
      for (unsigned i = 0; i < short_bits; i++)
        short_mask |= (1LL << i);
      for (int i = 0; i < 3; i++)
        random_number_generator[i] = (unsigned short)((seed & 
                            (short_mask << (i*short_bits))) >> (i*short_bits));
    }

    //--------------------------------------------------------------------------
    TestMapper::TestMapper(const TestMapper &rhs)
      : DefaultMapper(rhs.runtime, rhs.machine, 
                      rhs.local_proc, rhs.mapper_name)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TestMapper::~TestMapper(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TestMapper& TestMapper::operator=(const TestMapper &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TestMapper::select_task_options(const MapperContext ctx,
                                         const Task&         task,
                                               TaskOptions&  output)
    //--------------------------------------------------------------------------
    {
      Processor::Kind kind = select_random_processor_kind(ctx, task.task_id);
      // 1 in 100 times we'll inline the task if we are the right kind
      if ((kind == local_proc.kind()) && 
          (generate_random_integer() % INLINE_RATIO) == 0)
      {
        output.initial_proc = local_proc;
        output.inline_task = true;
        output.stealable = false;
        output.map_locally = true;
      }
      else
      {
        output.initial_proc = select_random_processor(kind);
        output.inline_task = false;
        output.stealable = true;
        output.map_locally = true;
        //output.map_locally = ((generate_random_integer() % 2) == 0);
      }
    }

    //--------------------------------------------------------------------------
    Processor TestMapper::select_random_processor(Processor::Kind kind)
    //--------------------------------------------------------------------------
    {
      Machine::ProcessorQuery random(machine);
      random.only_kind(kind);
      int chosen = generate_random_integer() % random.count();
      Machine::ProcessorQuery::iterator it = random.begin();
      for (int idx = 0; idx < chosen; idx++) it++;
      return (*it);
    }

    //--------------------------------------------------------------------------
    Processor::Kind TestMapper::select_random_processor_kind(MapperContext ctx,
                                                             TaskID task_id)
    //--------------------------------------------------------------------------
    {
      const std::map<VariantID,Processor::Kind> &variant_kinds = 
        find_task_variants(ctx, task_id);
      if (variant_kinds.size() == 1)
        return variant_kinds.begin()->second;
      int chosen = generate_random_integer() % variant_kinds.size();
      std::map<VariantID,Processor::Kind>::const_iterator it = 
        variant_kinds.begin();
      for (int idx = 0; idx < chosen; idx++) it++;
      return it->second;
    }

    //--------------------------------------------------------------------------
    void TestMapper::find_task_processor_kinds(MapperContext ctx, 
                               TaskID task_id, std::set<Processor::Kind> &kinds)
    //--------------------------------------------------------------------------
    {
      const std::map<VariantID,Processor::Kind> &variants = 
                                          find_task_variants(ctx, task_id);
      for (std::map<VariantID,Processor::Kind>::const_iterator it = 
            variants.begin(); it != variants.end(); it++)
      {
        kinds.insert(it->second);
      }
    }

    //--------------------------------------------------------------------------
    const std::map<VariantID,Processor::Kind>& TestMapper::find_task_variants(
                                              MapperContext ctx, TaskID task_id)
    //--------------------------------------------------------------------------
    {
      std::map<TaskID,std::map<VariantID,Processor::Kind> >::const_iterator
        finder = variant_processor_kinds.find(task_id);
      if (finder != variant_processor_kinds.end())
        return finder->second;
      std::vector<VariantID> valid_variants;
      runtime->find_valid_variants(ctx, task_id, valid_variants);
      std::map<VariantID,Processor::Kind> kinds;
      for (std::vector<VariantID>::const_iterator it = valid_variants.begin();
            it != valid_variants.end(); it++)
      {
        const ExecutionConstraintSet &constraints = 
          runtime->find_execution_constraints(ctx, task_id, *it);
        if (constraints.processor_constraint.is_valid())
          kinds[*it] = constraints.processor_constraint.valid_kinds[0];
        else
          kinds[*it] = Processor::LOC_PROC; // assume CPU
      }
      std::map<VariantID,Processor::Kind> &result = 
        variant_processor_kinds[task_id];
      result = kinds;
      return result;
    }

    //--------------------------------------------------------------------------
    void TestMapper::slice_task(const MapperContext      ctx,
                                const Task&              task,
                                const SliceTaskInput&    input,
                                      SliceTaskOutput&   output)
    //--------------------------------------------------------------------------
    {
      // Iterate over all the points and send them all over the world
      output.slices.resize(input.domain.get_volume());
      unsigned idx = 0;
      switch (input.domain.get_dim())
      {
        case 1:
          {
            Rect<1,coord_t> rect = input.domain;
            for (PointInRectIterator<1> pir(rect); pir(); pir++, idx++)
            {
              Rect<1,coord_t> slice(*pir, *pir);
              output.slices[idx] = TaskSlice(slice,
                  select_random_processor(task.target_proc.kind()),
                  false/*recurse*/, true/*stealable*/);
            }
            break;
          }
        case 2:
          {
            Rect<2,coord_t> rect = input.domain;
            for (PointInRectIterator<2> pir(rect); pir(); pir++, idx++)
            {
              Rect<2,coord_t> slice(*pir, *pir);
              output.slices[idx] = TaskSlice(slice,
                  select_random_processor(task.target_proc.kind()),
                  false/*recurse*/, true/*stealable*/);
            }
            break;
          }
        case 3:
          {
            Rect<3,coord_t> rect = input.domain;
            for (PointInRectIterator<3> pir(rect); pir(); pir++, idx++)
            {
              Rect<3,coord_t> slice(*pir, *pir);
              output.slices[idx] = TaskSlice(slice,
                  select_random_processor(task.target_proc.kind()),
                  false/*recurse*/, true/*stealable*/);
            }
            break;
          }
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    void TestMapper::map_task(const MapperContext         ctx,
                              const Task&                 task,
                              const MapTaskInput&         input,
                                    MapTaskOutput&        output)
    //--------------------------------------------------------------------------
    {
      // Pick a random variant, then pick separate instances for all the 
      // fields in a region requirement
      const std::map<VariantID,Processor::Kind> &variant_kinds = 
        find_task_variants(ctx, task.task_id);
      std::vector<VariantID> variants;
      for (std::map<VariantID,Processor::Kind>::const_iterator it = 
            variant_kinds.begin(); it != variant_kinds.end(); it++)
      {
        if (task.target_proc.kind() == it->second)
          variants.push_back(it->first);
      }
      assert(!variants.empty());
      if (variants.size() > 1)
      {
        int chosen = generate_random_integer() % variants.size();
        output.chosen_variant = variants[chosen];
      }
      else
        output.chosen_variant = variants[0];
      output.target_procs.push_back(task.target_proc);
      std::vector<bool> premapped(task.regions.size(), false);
      for (unsigned idx = 0; idx < input.premapped_regions.size(); idx++)
      {
        unsigned index = input.premapped_regions[idx];
        output.chosen_instances[index] = input.valid_instances[index];
        premapped[index] = true;
      }
      // Get the execution layout constraints for this variant
      const TaskLayoutConstraintSet &layout_constraints = 
        runtime->find_task_layout_constraints(ctx, task.task_id, 
                                               output.chosen_variant);
      bool is_inner_variant = runtime->is_inner_variant(ctx, 
                                  task.task_id, output.chosen_variant);
      for (unsigned idx = 0; idx < task.regions.size(); idx++)
      {
        if (premapped[idx])
          continue;
        if (task.regions[idx].is_restricted())
        {
          output.chosen_instances[idx] = input.valid_instances[idx];
          continue;
        }
        // If this is an inner task, see if we want to make a virtual instance
        if (is_inner_variant && 
            ((generate_random_integer() % INNER_VIRTUAL_RATIO) == 0)) 
        {
          output.chosen_instances[idx].push_back(
              PhysicalInstance::get_virtual_instance());
          continue;
        }
        // See if we have any layout constraints for this index
        // If we do we have to follow them, otherwise we can 
        // let all hell break loose and do what we want
        if (layout_constraints.layouts.find(idx) != 
              layout_constraints.layouts.end())
        {
          std::vector<LayoutConstraintID> constraints;
          for (std::multimap<unsigned,LayoutConstraintID>::const_iterator it = 
                layout_constraints.layouts.lower_bound(idx); it !=
                layout_constraints.layouts.upper_bound(idx); it++)
            constraints.push_back(it->second);
          map_constrained_requirement(ctx, task.regions[idx], TASK_MAPPING,
              constraints, output.chosen_instances[idx], task.target_proc);
        }
        else
          map_random_requirement(ctx, task.regions[idx], 
                                 output.chosen_instances[idx],
                                 task.target_proc);
      }
      // Give it a random priority
      output.task_priority = generate_random_integer();
    }

    //--------------------------------------------------------------------------
    void TestMapper::map_constrained_requirement(MapperContext ctx,
        const RegionRequirement &req, MappingKind mapping_kind, 
        const std::vector<LayoutConstraintID> &constraints,
        std::vector<PhysicalInstance> &chosen_instances, Processor restricted) 
    //--------------------------------------------------------------------------
    {
      chosen_instances.resize(constraints.size());
      unsigned output_idx = 0;
      for (std::vector<LayoutConstraintID>::const_iterator lay_it = 
            constraints.begin(); lay_it != 
            constraints.end(); lay_it++, output_idx++)
      {
        const LayoutConstraintSet &layout_constraints = 
          runtime->find_layout_constraints(ctx, *lay_it);
        // TODO: explore the constraints in more detail and exploit randomness
        // We'll use the default mapper to fill in any constraint holes for now
        Machine::MemoryQuery all_memories(machine);
        if (restricted.exists())
          all_memories.has_affinity_to(restricted);
        // This could be a big data structure in a big machine
        std::map<unsigned,Memory> random_memories;
        for (Machine::MemoryQuery::iterator it = all_memories.begin();
              it != all_memories.end(); it++)
        {
          random_memories[generate_random_integer()] = *it;
        }
        bool made_instance = false;
        while (!random_memories.empty())
        {
          std::map<unsigned,Memory>::iterator it = random_memories.begin();
          Memory target = it->second;
          random_memories.erase(it);
          if (target.capacity() == 0)
            continue;
          if (default_make_instance(ctx, target, layout_constraints,
                chosen_instances[output_idx], mapping_kind, 
                true/*force new*/, false/*meets*/, req))
          {
            made_instance = true;
            break;
          }
        }
        if (!made_instance)
        {
          log_test_mapper.error("Test mapper %s ran out of memory",
                                get_mapper_name());
          assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TestMapper::map_random_requirement(MapperContext ctx,
        const RegionRequirement &req, 
        std::vector<PhysicalInstance> &chosen_instances, Processor restricted)
    //--------------------------------------------------------------------------
    {
      
      std::vector<LogicalRegion> regions(1, req.region);
      chosen_instances.resize(req.privilege_fields.size());
      unsigned output_idx = 0;
      // Iterate over all the fields and make a separate instance and
      // put it in random places
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
            it != req.privilege_fields.end(); it++, output_idx++)
      {
        std::vector<FieldID> field(1, *it);
        // Try a bunch of memories in a random order until we find one 
        // that succeeds
        Machine::MemoryQuery all_memories(machine);
        if (restricted.exists())
          all_memories.has_affinity_to(restricted);
        // This could be a big data structure in a big machine
        std::map<unsigned,Memory> random_memories;
        for (Machine::MemoryQuery::iterator it = all_memories.begin();
              it != all_memories.end(); it++)
        {
          random_memories[generate_random_integer()] = *it;
        }
        bool made_instance = false;
        while (!random_memories.empty())
        {
          std::map<unsigned,Memory>::iterator it = random_memories.begin();
          Memory target = it->second;
          random_memories.erase(it);
          if (target.capacity() == 0)
            continue;
          // TODO: put in arbitrary constraints to mess with the DMA system
          LayoutConstraintSet constraints;
          default_policy_select_constraints(ctx, constraints, target, req);
          // Overwrite the field constraints 
          constraints.field_constraint = FieldConstraint(field, false);
          // Try to make the instance, we always make new instances to
          // generate as much data movement and dependence analysis as
          // we possibly can, it will also stress the garbage collector
          if (runtime->create_physical_instance(ctx, target, constraints,
                                   regions, chosen_instances[output_idx]))
          {
            made_instance = true;
            break;
          }
        }
        if (!made_instance)
        {
          log_test_mapper.error("Test mapper %s ran out of memory",
                                get_mapper_name());
          assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TestMapper::select_task_variant(const MapperContext        ctx,
                                         const Task&                task,
                                         const SelectVariantInput&  input,
                                               SelectVariantOutput& output)
    //--------------------------------------------------------------------------
    {
      const std::map<VariantID,Processor::Kind> &variant_kinds = 
        find_task_variants(ctx, task.task_id);
      std::vector<VariantID> variants;
      for (std::map<VariantID,Processor::Kind>::const_iterator it = 
            variant_kinds.begin(); it != variant_kinds.end(); it++)
      {
        if (task.target_proc.kind() == it->second)
          variants.push_back(it->first);
      }
      assert(!variants.empty());
      runtime->filter_variants(ctx, task, input.chosen_instances, variants);
      assert(!variants.empty());
      if (variants.size() == 1)
      {
        output.chosen_variant = variants[0];
        return;
      }
      int chosen = generate_random_integer() % variants.size();
      output.chosen_variant = variants[chosen];
    }

    //--------------------------------------------------------------------------
    void TestMapper::select_task_sources(const MapperContext        ctx,
                                         const Task&                task,
                                         const SelectTaskSrcInput&  input,
                                               SelectTaskSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      // Pick a random order
      select_random_source_order(input.source_instances, 
                                 output.chosen_ranking); 
    }

    //--------------------------------------------------------------------------
    void TestMapper::select_random_source_order(
                                const std::vector<PhysicalInstance> &sources,
                                      std::deque<PhysicalInstance> &ranking)
    //--------------------------------------------------------------------------
    {
      std::vector<bool> handled(sources.size(), false);
      for (int count = sources.size(); count > 1; count--) 
      {
        int chosen = generate_random_integer() % count;
        int index = 0;
        while (chosen > 0)
        {
          if (!handled[index])
            chosen--;
          index++;
        }
        while (handled[index])
          index++;
        assert(index < int(sources.size()));
        ranking.push_back(sources[index]);
      }
      // Do the last one
      for (unsigned idx = 0; idx < sources.size(); idx++)
      {
        if (!handled[idx])
        {
          ranking.push_back(sources[idx]);
          break;
        }
      }
    }

    //--------------------------------------------------------------------------
    void TestMapper::speculate(const MapperContext            ctx,
                               const Task&                    task,
                                     SpeculativeOutput&       output)
    //--------------------------------------------------------------------------
    {
      // TODO: turn on random speculation
      output.speculate = false;
    }

    //--------------------------------------------------------------------------
    void TestMapper::map_inline(const MapperContext        ctx,
                                const InlineMapping&       inline_op,
                                const MapInlineInput&      input,
                                      MapInlineOutput&     output)
    //--------------------------------------------------------------------------
    {
      if (inline_op.layout_constraint_id > 0)
      {
        std::vector<LayoutConstraintID> constraints(1, 
                                          inline_op.layout_constraint_id);
        map_constrained_requirement(ctx, inline_op.requirement, INLINE_MAPPING,
                                    constraints, output.chosen_instances);
      }
      else
        map_random_requirement(ctx, inline_op.requirement,
                               output.chosen_instances);
    }

    //--------------------------------------------------------------------------
    void TestMapper::select_inline_sources(const MapperContext    ctx,
                                     const InlineMapping&         inline_op,
                                     const SelectInlineSrcInput&  input,
                                           SelectInlineSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      select_random_source_order(input.source_instances,
                                 output.chosen_ranking);
    }

    //--------------------------------------------------------------------------
    void TestMapper::map_copy(const MapperContext      ctx,
                              const Copy&              copy,
                              const MapCopyInput&      input,
                                    MapCopyOutput&     output)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
      {
        if (copy.src_requirements[idx].is_restricted())
        {
          output.src_instances[idx] = input.src_instances[idx];
          continue;
        }
        if ((copy.dst_requirements[idx].privilege == LEGION_READ_WRITE) &&
            ((generate_random_integer() % COPY_VIRTUAL_RATIO) == 0))
        {
          output.src_instances[idx].push_back(
              PhysicalInstance::get_virtual_instance());
          continue;
        }
        map_random_requirement(ctx, copy.src_requirements[idx], 
                               output.src_instances[idx]);
      }
      for (unsigned idx = 0; idx < copy.dst_requirements.size(); idx++)
      {
        if (copy.dst_requirements[idx].is_restricted())
        {
          output.dst_instances[idx] = input.dst_instances[idx];
          continue;
        }
        map_random_requirement(ctx, copy.dst_requirements[idx],
                               output.dst_instances[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void TestMapper::select_copy_sources(const MapperContext          ctx,
                                         const Copy&                  copy,
                                         const SelectCopySrcInput&    input,
                                               SelectCopySrcOutput&   output)
    //--------------------------------------------------------------------------
    {
      select_random_source_order(input.source_instances,
                                 output.chosen_ranking);
    }

    //--------------------------------------------------------------------------
    void TestMapper::speculate(const MapperContext      ctx,
                               const Copy&              copy,
                                     SpeculativeOutput& output)
    //--------------------------------------------------------------------------
    {
      // TODO: speculate sometimes
      output.speculate = false;
    }

    //--------------------------------------------------------------------------
    void TestMapper::map_close(const MapperContext       ctx,
                               const Close&              close,
                               const MapCloseInput&      input,
                                     MapCloseOutput&     output)
    //--------------------------------------------------------------------------
    {
      // Figure out if we want to generate a real instance or a composite
      if ((generate_random_integer() % REAL_CLOSE_RATIO) == 0)
      {
        // make real instances for all the fields
        map_random_requirement(ctx, close.requirement, output.chosen_instances);
      }
      else // just make a composite instance
        output.chosen_instances.push_back(
                                  PhysicalInstance::get_virtual_instance());
    }

    //--------------------------------------------------------------------------
    void TestMapper::select_close_sources(const MapperContext        ctx,
                                          const Close&               close,
                                          const SelectCloseSrcInput&  input,
                                                SelectCloseSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      select_random_source_order(input.source_instances,
                                 output.chosen_ranking);
    }

    //--------------------------------------------------------------------------
    void TestMapper::speculate(const MapperContext         ctx,
                               const Acquire&              acquire,
                                     SpeculativeOutput&    output)
    //--------------------------------------------------------------------------
    {
      // TODO: enable speculation
      output.speculate = false;
    }

    //--------------------------------------------------------------------------
    void TestMapper::select_release_sources(const MapperContext       ctx,
                                       const Release&                 release,
                                       const SelectReleaseSrcInput&   input,
                                             SelectReleaseSrcOutput&  output)
    //--------------------------------------------------------------------------
    {
      select_random_source_order(input.source_instances,
                                 output.chosen_ranking);
    }

    //--------------------------------------------------------------------------
    void TestMapper::speculate(const MapperContext         ctx,
                               const Release&              release,
                                     SpeculativeOutput&    output)
    //--------------------------------------------------------------------------
    {
      // TODO: enable speculation
      output.speculate = false;
    }

    //--------------------------------------------------------------------------
    void TestMapper::select_tasks_to_map(const MapperContext          ctx,
                                         const SelectMappingInput&    input,
                                               SelectMappingOutput&   output)
    //--------------------------------------------------------------------------
    {
      // Pick a random task to map
      unsigned map_index = generate_random_integer() % input.ready_tasks.size();
      std::list<const Task*>::const_iterator it = input.ready_tasks.begin();
      for (unsigned idx = 0; idx < map_index; idx++) it++;
      output.map_tasks.insert(*it);
      // Maybe send another task to a different processor
      if ((input.ready_tasks.size() > 1) && 
          ((generate_random_integer() % REDISTRIBUTE_RATIO) == 0))
      {
        unsigned distribute_index = 
          generate_random_integer() % (input.ready_tasks.size()-1);
        std::list<const Task*>::const_iterator it2 = input.ready_tasks.begin();
        for (unsigned idx = 0; idx < distribute_index; idx++)
        {
          // Skip the one we are mapping
          if (it2 == it)
            it2++;
          it2++;
        }
        // Pick a random processor of the same kind
        Processor target = select_random_processor(local_proc.kind()); 
        if (target != local_proc)
          output.relocate_tasks[*it2] = target; 
      }
    }

    //--------------------------------------------------------------------------
    void TestMapper::select_steal_targets(const MapperContext         ctx,
                                          const SelectStealingInput&  input,
                                                SelectStealingOutput& output)
    //--------------------------------------------------------------------------
    {
      // Always send a steal request
      Processor target = select_random_processor(local_proc.kind());
      if ((target != local_proc) && 
          (input.blacklist.find(target) == input.blacklist.end()))
        output.targets.insert(target);
    }

    //--------------------------------------------------------------------------
    void TestMapper::permit_steal_request(const MapperContext         ctx,
                                          const StealRequestInput&    input,
                                                StealRequestOutput&   output)
    //--------------------------------------------------------------------------
    {
      if ((generate_random_integer() % STEAL_PERMIT_RATIO) == 0)
      {
        unsigned index = 
          generate_random_integer() % input.stealable_tasks.size();
        std::vector<const Task*>::const_iterator it = 
          input.stealable_tasks.begin();
        for (unsigned idx = 0; idx < index; idx++) it++;
        output.stolen_tasks.insert(*it);
      }
    }

  }; // namespace Mapping 
}; // namespace Legion

// EOF

