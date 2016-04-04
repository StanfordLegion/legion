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

#include "test_mapper.h"

#define INLINE_RATIO    100 // 1 in 100

namespace Legion {
  namespace Mapping {

    LegionRuntime::Logger::Category log_test("test_mapper");

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
    TestMapper::TestMapper(Machine m, Processor local, const char *name)
      : machine(m), local_proc(local),
        mapper_name((name == NULL) ? create_test_name(local) : name)
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
        seed = local_proc.id;
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
      : machine(rhs.machine), local_proc(rhs.local_proc), 
        mapper_name(rhs.mapper_name)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TestMapper::~TestMapper(void)
    //--------------------------------------------------------------------------
    {
      free(const_cast<char*>(mapper_name));
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
    const char* TestMapper::get_mapper_name(void) const
    //--------------------------------------------------------------------------
    {
      return mapper_name;
    }

    //--------------------------------------------------------------------------
    Mapper::MapperSyncModel TestMapper::get_mapper_sync_model(void) const
    //--------------------------------------------------------------------------
    {
      // Need atomicity to protect our random number generator
      return SERIALIZED_REENTRANT_MAPPER_MODEL;
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
        output.map_locally = ((generate_random_integer() % 2) == 0);
      }
    }

    //--------------------------------------------------------------------------
    long TestMapper::generate_random_integer(void) const
    //--------------------------------------------------------------------------
    {
      return nrand48(random_number_generator);
    }
    
    //--------------------------------------------------------------------------
    double TestMapper::generate_random_real(void) const
    //--------------------------------------------------------------------------
    {
      return erand48(random_number_generator);
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
      mapper_rt_find_valid_variants(ctx, task_id, valid_variants);
      std::map<VariantID,Processor::Kind> kinds;
      for (std::vector<VariantID>::const_iterator it = valid_variants.begin();
            it != valid_variants.end(); it++)
      {
        const ExecutionConstraintSet &constraints = 
          mapper_rt_find_execution_constraints(ctx, task_id, *it);
        if (constraints.processor_constraint.is_valid())
          kinds[*it] = constraints.processor_constraint.get_kind();
        else
          kinds[*it] = Processor::LOC_PROC; // assume CPU
      }
      std::map<VariantID,Processor::Kind> &result = 
        variant_processor_kinds[task_id];
      result = kinds;
      return result;
    }

    //--------------------------------------------------------------------------
    void TestMapper::premap_task(const MapperContext     ctx,
                                 const Task&             task,
                                 const PremapTaskInput&  input,
                                       PremapTaskOutput& output)
    //--------------------------------------------------------------------------
    {
      for (std::map<unsigned,std::vector<PhysicalInstance> >::const_iterator
            it = input.valid_instances.begin();
            it != input.valid_instances.end(); it++)
      {
        // If it is restricted, then we can't do anything random
        if (task.regions[it->first].is_restricted())
        {
          output.premapped_instances.insert(*it);
          continue;
        }
        // TODO figure out how to pick a location for all points
        assert(false);
      }
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
            LegionRuntime::Arrays::Rect<1> rect = input.domain.get_rect<1>();
            for (LegionRuntime::Arrays::GenericPointInRectIterator<1> pir(rect);
                  pir; pir++, idx++)
            {
              Rect<1> slice(pir.p, pir.p);
              output.slices[idx] = TaskSlice(Domain::from_rect<1>(slice),
                  select_random_processor(task.target_proc.kind()),
                  false/*recurse*/, true/*stealable*/);
            }
            break;
          }
        case 2:
          {
            LegionRuntime::Arrays::Rect<2> rect = input.domain.get_rect<2>();
            for (LegionRuntime::Arrays::GenericPointInRectIterator<2> pir(rect);
                  pir; pir++, idx++)
            {
              Rect<2> slice(pir.p, pir.p);
              output.slices[idx] = TaskSlice(Domain::from_rect<2>(slice),
                  select_random_processor(task.target_proc.kind()),
                  false/*recurse*/, true/*stealable*/);
            }
            break;
          }
        case 3:
          {
            LegionRuntime::Arrays::Rect<3> rect = input.domain.get_rect<3>();
            for (LegionRuntime::Arrays::GenericPointInRectIterator<3> pir(rect);
                  pir; pir++, idx++)
            {
              Rect<3> slice(pir.p, pir.p);
              output.slices[idx] = TaskSlice(Domain::from_rect<3>(slice),
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
      if (variants.size() == 1)
      {
        output.chosen_variant = variants[0];
        return;
      }
      mapper_rt_filter_variants(ctx, task, input.chosen_instances, variants);
      assert(!variants.empty());
      if (variants.size() == 1)
      {
        output.chosen_variant = variants[0];
        return;
      }
      int chosen = generate_random_integer() % variants.size();
      output.chosen_variant = variants[chosen];
    }

    //------------------------------------------------------------------------
    void TestMapper::postmap_task(const MapperContext   ctx,
                                  const Task&           task,
                                  const PostMapInput&   input,
                                        PostMapOutput&  output)
    //------------------------------------------------------------------------
    {
      // Do nothing 
    }

    //------------------------------------------------------------------------
    void TestMapper::select_task_sources(const MapperContext        ctx,
                                         const Task&                task,
                                         const SelectTaskSrcInput&  input,
                                               SelectTaskSrcOutput& output)
    //------------------------------------------------------------------------
    {
      // Pick a random order
      select_random_source_order(input.source_instances, 
                                 output.chosen_ranking); 
    }

    //------------------------------------------------------------------------
    void TestMapper::select_random_source_order(
                                const std::vector<PhysicalInstance> &sources,
                                      std::deque<PhysicalInstance> &ranking)
    //------------------------------------------------------------------------
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
        while (!handled[index])
          index++;
        assert(index < sources.size());
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
    
  }; // namespace Mapping 
}; // namespace Legion

// EOF

