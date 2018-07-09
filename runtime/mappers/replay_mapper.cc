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

#include "mappers/replay_mapper.h"
#include "legion/legion_utilities.h"

namespace Legion {
  namespace Mapping {

    Logger log_replay("replay_mapper");

    //--------------------------------------------------------------------------
    /*static*/ const char* ReplayMapper::create_replay_name(Processor p)
    //--------------------------------------------------------------------------
    {
      const size_t buffer_size = 64;
      char *result = (char*)malloc(buffer_size*sizeof(char));
      snprintf(result, buffer_size-1,
                "Replay Mapper on Processor " IDFMT "", p.id);
      return result;
    }

    //--------------------------------------------------------------------------
    ReplayMapper::ReplayMapper(MapperRuntime *rt, Machine m, Processor local,
                               const char *replay_file, const char *name)
      : Mapper(rt), machine(m), local_proc(local), 
        mapper_name((name == NULL) ? create_replay_name(local) : name)
    //--------------------------------------------------------------------------
    {
      FILE *f = fopen(replay_file, "rb");
      // First validate that the machine is the same
      {
        unsigned num_processors;
        ignore_result(fread(&num_processors, sizeof(num_processors), 1, f));
        std::map<Processor,Processor::Kind> orig_processors;
        for (unsigned idx = 0; idx < num_processors; idx++)
        {
          Processor p;
          ignore_result(fread(&p.id, sizeof(p.id), 1, f));
          Processor::Kind k;
          ignore_result(fread(&k, sizeof(k), 1, f));
          orig_processors[p] = k;
        }
        Machine::ProcessorQuery all_procs(machine);
        if (all_procs.count() != num_processors)
        {
          log_replay.error("Replay mapper failure. Processor count mismatch.");
          assert(false);
        }
        for (Machine::ProcessorQuery::iterator it = all_procs.begin();
              it != all_procs.end(); it++)
        {
          std::map<Processor,Processor::Kind>::const_iterator finder = 
            orig_processors.find(*it);
          if (finder == orig_processors.end())
          {
            log_replay.error("Replay mapper failure. Missing processor in "
                             "machine configuration.");
            assert(false);
          }
          if (it->kind() != finder->second)
          {
            log_replay.error("Replay mapper failure. Processor kind mismatch "
                             "in machine configuration.");
            assert(false);
          }
        }
      }
      {
        unsigned num_memories;
        ignore_result(fread(&num_memories, sizeof(num_memories), 1, f));
        std::map<Memory,Memory::Kind> orig_memories;
        for (unsigned idx = 0; idx < num_memories; idx++)
        {
          Memory m;
          ignore_result(fread(&m.id, sizeof(m.id), 1, f));
          Memory::Kind k;
          ignore_result(fread(&k, sizeof(k), 1, f));
          orig_memories[m] = k;
        }
        Machine::MemoryQuery all_mems(machine);
        if (all_mems.count() != num_memories)
        {
          log_replay.error("Replay mapper failure. Memory count mismatch.");
          assert(false);
        }
        for (Machine::MemoryQuery::iterator it = all_mems.begin();
              it != all_mems.end(); it++)
        {
          std::map<Memory,Memory::Kind>::const_iterator finder = 
            orig_memories.find(*it);
          if (finder == orig_memories.end())
          {
            log_replay.error("Replay mapper failure. Missing memory in "
                             "machine configuration.");
            assert(false);
          }
          if (it->kind() != finder->second)
          {
            log_replay.error("Replay mapper failure. Memory kind mismatch "
                             "in machine configuration.");
            assert(false);
          }
        }
      }
      // Now build the mapping data structures
      unsigned num_instances;
      ignore_result(fread(&num_instances, sizeof(num_instances), 1, f));
      for (unsigned idx = 0; idx < num_instances; idx++)
      {
        InstanceInfo *instance = unpack_instance(f);
        instance_infos[instance->original_id] = instance;
      }
      // Unpack the ID of the top level task
      ignore_result(fread(&top_level_id, sizeof(top_level_id), 1, f));
      unsigned num_tasks;
      ignore_result(fread(&num_tasks, sizeof(num_tasks), 1, f));
      for (unsigned idx = 0; idx < num_tasks; idx++)
      {
        std::pair<UniqueID,DomainPoint> key;
        ignore_result(fread(&key.first, sizeof(key.first), 1, f));
        ignore_result(fread(&key.second.dim, sizeof(key.second.dim), 1, f));
        for (int i = 0; i < key.second.dim; i++)
          ignore_result(fread(key.second.point_data+i, 
                      sizeof(key.second.point_data[i]), 1, f));
        task_mappings[key] = unpack_task_mapping(f);     
      }
      unsigned num_inlines;
      ignore_result(fread(&num_inlines, sizeof(num_inlines), 1, f));
      for (unsigned idx = 0; idx < num_inlines; idx++)
      {
        UniqueID uid;
        ignore_result(fread(&uid, sizeof(uid), 1, f));
        inline_mappings[uid] = unpack_inline_mapping(f);
      }
      unsigned num_copies;
      ignore_result(fread(&num_copies, sizeof(num_copies), 1, f));
      for (unsigned idx = 0; idx < num_copies; idx++)
      {
        UniqueID uid;
        ignore_result(fread(&uid, sizeof(uid), 1, f));
        copy_mappings[uid] = unpack_copy_mapping(f);
      }
      unsigned num_closes;
      ignore_result(fread(&num_closes, sizeof(num_closes), 1, f));
      for (unsigned idx = 0; idx < num_closes; idx++)
      {
        UniqueID uid;
        ignore_result(fread(&uid, sizeof(uid), 1, f));
        close_mappings[uid] = unpack_close_mapping(f);
      }
      unsigned num_releases;
      ignore_result(fread(&num_releases, sizeof(num_releases), 1, f));
      for (unsigned idx = 0; idx < num_releases; idx++)
      {
        UniqueID uid;
        ignore_result(fread(&uid, sizeof(uid), 1, f));
        release_mappings[uid] = unpack_release_mapping(f);
      }
      fclose(f);
    }

    //--------------------------------------------------------------------------
    ReplayMapper::ReplayMapper(const ReplayMapper &rhs)
      : Mapper(rhs.runtime), machine(rhs.machine), 
        local_proc(rhs.local_proc), mapper_name(rhs.mapper_name)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplayMapper::~ReplayMapper(void)
    //--------------------------------------------------------------------------
    {
      free(const_cast<char*>(mapper_name));
    }

    //--------------------------------------------------------------------------
    ReplayMapper& ReplayMapper::operator=(const ReplayMapper &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    const char* ReplayMapper::get_mapper_name(void) const    
    //--------------------------------------------------------------------------
    {
      return mapper_name;
    }

    //--------------------------------------------------------------------------
    Mapper::MapperSyncModel ReplayMapper::get_mapper_sync_model(void) const
    //--------------------------------------------------------------------------
    {
      return SERIALIZED_REENTRANT_MAPPER_MODEL;
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_task_options(const MapperContext    ctx,
                                           const Task&            task,
                                                 TaskOptions&     output)
    //--------------------------------------------------------------------------
    {
      TaskMappingInfo *mapping = find_task_mapping(ctx, task, task.index_point);
      output.initial_proc = mapping->target_proc;
      output.inline_task = false;
      output.stealable = false;
      output.map_locally = true;
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::premap_task(const MapperContext      ctx,
                                   const Task&              task, 
                                   const PremapTaskInput&   input,
                                         PremapTaskOutput&  output)
    //--------------------------------------------------------------------------
    {
      TaskMappingInfo *mapping = find_task_mapping(ctx, task, task.index_point);
      output.new_target_proc = mapping->target_proc;
      for (std::map<unsigned,RequirementMapping*>::const_iterator it = 
            mapping->premappings.begin(); it != 
            mapping->premappings.end(); it++)
        it->second->map_requirement(runtime, ctx,task.regions[it->first].parent,
                                    output.premapped_instances[it->first]);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::slice_task(const MapperContext      ctx,
                                  const Task&              task, 
                                  const SliceTaskInput&    input,
                                        SliceTaskOutput&   output)
    //--------------------------------------------------------------------------
    {
      // Slice it into each of the individual points and distribute them
      // to the appropriate processors, do not recurse
      output.slices.resize(input.domain.get_volume());
      unsigned idx = 0;
      for (Domain::DomainPointIterator itr(task.index_domain); itr; itr++,idx++)
      {
        TaskMappingInfo *mapping = find_task_mapping(ctx, task, itr.p);
        TaskSlice &slice = output.slices[idx]; 
        switch (task.index_domain.get_dim())
        {
          case 1:
            {
              slice.domain = Domain::from_point<1>(itr.p.get_point<1>());
              break;
            }
          case 2:
            {
              slice.domain = Domain::from_point<2>(itr.p.get_point<2>());
              break;
            }
          case 3:
            {
              slice.domain = Domain::from_point<3>(itr.p.get_point<3>());
              break;
            }
          default:
            assert(false);
        }
        slice.proc = mapping->target_proc;
        slice.recurse = false;
        slice.stealable = false;
      }
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_task(const MapperContext      ctx,
                                const Task&              task,
                                const MapTaskInput&      input,
                                      MapTaskOutput&     output)
    //--------------------------------------------------------------------------
    {
      TaskMappingInfo *mapping = find_task_mapping(ctx, task, task.index_point);
      // Legion Spy might overapproximate for newly created regions
      assert(task.regions.size() <= mapping->mappings.size());
      for (unsigned idx = 0; idx < task.regions.size(); idx++)
        mapping->mappings[idx]->map_requirement(runtime, ctx, 
            task.regions[idx].parent, output.chosen_instances[idx]);
      output.target_procs.push_back(mapping->target_proc);
      output.chosen_variant = mapping->variant;
      output.task_priority = mapping->priority;
      output.postmap_task = !mapping->postmappings.empty();
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_task_variant(const MapperContext          ctx,
                                           const Task&                  task,
                                           const SelectVariantInput&    input,
                                                 SelectVariantOutput&   output)
    //--------------------------------------------------------------------------
    {
      TaskMappingInfo *mapping = find_task_mapping(ctx, task, task.index_point);
      output.chosen_variant = mapping->variant;
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::postmap_task(const MapperContext      ctx,
                                    const Task&              task,
                                    const PostMapInput&      input,
                                          PostMapOutput&     output)
    //--------------------------------------------------------------------------
    {
      TaskMappingInfo *mapping = find_task_mapping(ctx, task, task.index_point);
      assert(output.chosen_instances.size() == task.regions.size());
      for (std::map<unsigned,RequirementMapping*>::const_iterator it = 
            mapping->postmappings.begin(); it != 
            mapping->postmappings.end(); it++)
        it->second->map_requirement(runtime, ctx,task.regions[it->first].parent,
                                    output.chosen_instances[it->first]);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_task_sources(const MapperContext        ctx,
                                           const Task&                task,
                                           const SelectTaskSrcInput&  input,
                                                 SelectTaskSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      // TODO; update this once we record the output of select task sources
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::create_task_temporary_instance(
                                    const MapperContext              ctx,
                                    const Task&                      task,
                                    const CreateTaskTemporaryInput&  input,
                                          CreateTaskTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      TaskMappingInfo *mapping = find_task_mapping(ctx, task, task.index_point);
      assert(mapping->temporaries.find(input.region_requirement_index) !=
             mapping->temporaries.end());
      unsigned long original_dst = find_original_instance_id(ctx,
                                 input.destination_instance.get_instance_id());
      mapping->temporaries[input.region_requirement_index]->map_temporary(
            runtime, ctx, task.regions[input.region_requirement_index].parent, 
            original_dst, output.temporary_instance);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::speculate(const MapperContext      ctx,
                                 const Task&              task,
                                       SpeculativeOutput& output)
    //--------------------------------------------------------------------------
    {
      output.speculate = false;
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::report_profiling(const MapperContext       ctx,
                                        const Task&               task,
                                        const TaskProfilingInfo&  input)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_inline(const MapperContext        ctx,
                                  const InlineMapping&       inline_op,
                                  const MapInlineInput&      input,
                                        MapInlineOutput&     output)
    //--------------------------------------------------------------------------
    {
      InlineMappingInfo *mapping = find_inline_mapping(ctx, inline_op);
      mapping->mapping->map_requirement(runtime, ctx, 
          inline_op.requirement.parent, output.chosen_instances);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_inline_sources(const MapperContext     ctx,
                                        const InlineMapping&         inline_op,
                                        const SelectInlineSrcInput&  input,
                                              SelectInlineSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      // TODO: update this once we record the output of select inline sources
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::create_inline_temporary_instance(
                                  const MapperContext                ctx,
                                  const InlineMapping&               inline_op,
                                  const CreateInlineTemporaryInput&  input,
                                        CreateInlineTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      InlineMappingInfo *mapping = find_inline_mapping(ctx, inline_op);
      unsigned long original_dst = find_original_instance_id(ctx,
                                 input.destination_instance.get_instance_id());
      mapping->temporary->map_temporary(runtime, ctx, 
         inline_op.requirement.parent, original_dst, output.temporary_instance);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::report_profiling(const MapperContext         ctx,
                                        const InlineMapping&        inline_op,
                                        const InlineProfilingInfo&  input)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_copy(const MapperContext      ctx,
                                const Copy&              copy,
                                const MapCopyInput&      input,
                                      MapCopyOutput&     output)
    //--------------------------------------------------------------------------
    {
      CopyMappingInfo *mapping = find_copy_mapping(ctx, copy);
      for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
      {
        mapping->src_mappings[idx]->map_requirement(runtime, ctx, 
            copy.src_requirements[idx].parent, output.src_instances[idx]);
      }
      for (unsigned idx = 0; idx < copy.dst_requirements.size(); idx++)
      {
        mapping->dst_mappings[idx]->map_requirement(runtime, ctx,
            copy.dst_requirements[idx].parent, output.dst_instances[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_copy_sources(const MapperContext          ctx,
                                           const Copy&                  copy,
                                           const SelectCopySrcInput&    input,
                                                 SelectCopySrcOutput&   output)
    //--------------------------------------------------------------------------
    {
      // TODO: Update this once we record the output of select copy sources
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::create_copy_temporary_instance(
                                  const MapperContext              ctx,
                                  const Copy&                      copy,
                                  const CreateCopyTemporaryInput&  input,
                                        CreateCopyTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      CopyMappingInfo *mapping = find_copy_mapping(ctx, copy);
      if (input.src_requirement)
      {
        assert(mapping->src_temporaries.find(input.region_requirement_index) !=
               mapping->src_temporaries.end());
        unsigned long original_dst = find_original_instance_id(ctx,
                                 input.destination_instance.get_instance_id());
        mapping->src_temporaries[input.region_requirement_index]->map_temporary(
            runtime, ctx, 
            copy.src_requirements[input.region_requirement_index].parent, 
            original_dst, output.temporary_instance);
      }
      else
      {
        assert(mapping->dst_temporaries.find(input.region_requirement_index) !=
               mapping->dst_temporaries.end());
        unsigned long original_dst = find_original_instance_id(ctx,
                                 input.destination_instance.get_instance_id());
        mapping->dst_temporaries[input.region_requirement_index]->map_temporary(
            runtime, ctx, 
            copy.dst_requirements[input.region_requirement_index].parent,
            original_dst, output.temporary_instance);
      }
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::speculate(const MapperContext      ctx,
                                 const Copy&              copy,
                                       SpeculativeOutput& output)
    //--------------------------------------------------------------------------
    {
      output.speculate = false;
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::report_profiling(const MapperContext      ctx,
                                        const Copy&              copy,
                                        const CopyProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }
    
    //--------------------------------------------------------------------------
    void ReplayMapper::map_close(const MapperContext       ctx,
                                 const Close&              close,
                                 const MapCloseInput&      input,
                                       MapCloseOutput&     output)
    //--------------------------------------------------------------------------
    {
      CloseMappingInfo *mapping = find_close_mapping(ctx, close); 
      mapping->mapping->map_requirement(runtime, ctx, close.requirement.parent,
                                        output.chosen_instances);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_close_sources(const MapperContext        ctx,
                                            const Close&               close,
                                            const SelectCloseSrcInput&  input,
                                                  SelectCloseSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      // TODO: update this once we record the output of select close sources
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::create_close_temporary_instance(
                                  const MapperContext               ctx,
                                  const Close&                      close,
                                  const CreateCloseTemporaryInput&  input,
                                        CreateCloseTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      CloseMappingInfo *mapping = find_close_mapping(ctx, close);
      unsigned long original_dst = find_original_instance_id(ctx,
                                 input.destination_instance.get_instance_id());
      mapping->temporary->map_temporary(runtime, ctx, 
          close.requirement.parent, original_dst, output.temporary_instance);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::report_profiling(const MapperContext       ctx,
                                        const Close&              close,
                                        const CloseProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_acquire(const MapperContext         ctx,
                                   const Acquire&              acquire,
                                   const MapAcquireInput&      input,
                                         MapAcquireOutput&     output)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::speculate(const MapperContext         ctx,
                                 const Acquire&              acquire,
                                       SpeculativeOutput&    output)
    //--------------------------------------------------------------------------
    {
      output.speculate = false;
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::report_profiling(const MapperContext         ctx,
                                        const Acquire&              acquire,
                                        const AcquireProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_release(const MapperContext         ctx,
                                   const Release&              release,
                                   const MapReleaseInput&      input,
                                         MapReleaseOutput&     output)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_release_sources(const MapperContext      ctx,
                                        const Release&                 release,
                                        const SelectReleaseSrcInput&   input,
                                              SelectReleaseSrcOutput&  output)
    //--------------------------------------------------------------------------
    {
      // TODO: update this once we record the output of select release sources
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::speculate(const MapperContext         ctx,
                                 const Release&              release,
                                       SpeculativeOutput&    output)
    //--------------------------------------------------------------------------
    {
      output.speculate = false;
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::create_release_temporary_instance(
                                   const MapperContext                 ctx,
                                   const Release&                      release,
                                   const CreateReleaseTemporaryInput&  input,
                                         CreateReleaseTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      ReleaseMappingInfo *mapping = find_release_mapping(ctx, release);
      unsigned long original_dst = find_original_instance_id(ctx,
                                 input.destination_instance.get_instance_id());
      mapping->temporary->map_temporary(runtime, ctx, 
          release.parent_region, original_dst, output.temporary_instance);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::report_profiling(const MapperContext         ctx,
                                        const Release&              release,
                                        const ReleaseProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_partition_projection(const MapperContext  ctx,
                        const Partition&                          partition,
                        const SelectPartitionProjectionInput&     input,
                              SelectPartitionProjectionOutput&    output)
    //--------------------------------------------------------------------------
    {
      assert(false); // TODO
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_partition(const MapperContext        ctx,
                               const Partition&           partition,
                               const MapPartitionInput&   input,
                                     MapPartitionOutput&  output)
    //--------------------------------------------------------------------------
    {
      assert(false); // TODO
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_partition_sources(
                                     const MapperContext             ctx,
                                     const Partition&                partition,
                                     const SelectPartitionSrcInput&  input,
                                           SelectPartitionSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      assert(false); // TODO
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::create_partition_temporary_instance(
                            const MapperContext                   ctx,
                            const Partition&                      partition,
                            const CreatePartitionTemporaryInput&  input,
                                  CreatePartitionTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      assert(false); // TODO
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::report_profiling(const MapperContext              ctx,
                                  const Partition&                 partition,
                                  const PartitionProfilingInfo&    input)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::configure_context(const MapperContext         ctx,
                                         const Task&                 task,
                                               ContextConfigOutput&  output)
    //--------------------------------------------------------------------------
    {
      // Use the default values, but broadcast the unique ID mapping
      TaskMappingInfo *mapping = find_task_mapping(ctx, task, task.index_point);
      Legion::Serializer rez;
      rez.serialize(task.get_unique_id());
      rez.serialize(mapping->original_unique_id);
      runtime->broadcast(ctx, rez.get_buffer(), 
                         rez.get_used_bytes(), ID_MAPPING_MESSAGE);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_tunable_value(const MapperContext         ctx,
                                            const Task&                 task,
                                            const SelectTunableInput&   input,
                                                  SelectTunableOutput&  output)
    //--------------------------------------------------------------------------
    {
      TaskMappingInfo *mapping = find_task_mapping(ctx, task, task.index_point);
      assert(mapping->next_tunable < mapping->tunables.size());
      mapping->tunables[mapping->next_tunable++]->set_tunable(output.value,
                                                              output.size);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_must_epoch(const MapperContext           ctx,
                                      const MapMustEpochInput&      input,
                                            MapMustEpochOutput&     output)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < input.constraints.size(); idx++)
      {
        const MappingConstraint &constraint = input.constraints[idx];
        const Task *task = constraint.constrained_tasks[0];
        TaskMappingInfo *mapping = find_task_mapping(ctx, *task, 
                                                     task->index_point);
        assert(constraint.requirement_indexes[0] < mapping->mappings.size());
        mapping->mappings[constraint.requirement_indexes[0]]->map_requirement(
          runtime, ctx, task->regions[constraint.requirement_indexes[0]].parent,
          output.constraint_mappings[idx]);
      }
      for (unsigned idx = 0; idx < input.tasks.size(); idx++)
      {
        const Task *task = input.tasks[idx];
        TaskMappingInfo *mapping = find_task_mapping(ctx, *task, 
                                                     task->index_point);
        output.task_processors[idx] = mapping->target_proc;
      }
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_dataflow_graph(const MapperContext           ctx,
                                          const MapDataflowGraphInput&  input,
                                                MapDataflowGraphOutput& output)
    //--------------------------------------------------------------------------
    {
      // TODO: update this when we have dataflow operations
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::memoize_operation(const MapperContext  ctx,
                                         const Mappable&      mappable,
                                         const MemoizeInput&  input,
                                               MemoizeOutput& output)
    //--------------------------------------------------------------------------
    {
      // TODO: update this when we record memoization decision
      output.memoize = false;
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_tasks_to_map(const MapperContext          ctx,
                                           const SelectMappingInput&    input,
                                                 SelectMappingOutput&   output)
    //--------------------------------------------------------------------------
    {
      // Just map all the ready tasks
      for (std::list<const Task*>::const_iterator it = 
            input.ready_tasks.begin(); it != input.ready_tasks.end(); it++)
        output.map_tasks.insert(*it);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_steal_targets(const MapperContext         ctx,
                                            const SelectStealingInput&  input,
                                                  SelectStealingOutput& output)
    //--------------------------------------------------------------------------
    {
      // Nothing to do, no stealing in the replay mapper
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::permit_steal_request(const MapperContext         ctx,
                                            const StealRequestInput&    input,
                                                  StealRequestOutput&   output)
    //--------------------------------------------------------------------------
    {
      // Nothing to do, no stealing in the replay mapper
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::handle_message(const MapperContext           ctx,
                                      const MapperMessage&          message)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(message.message, message.size);
      switch (message.kind)
      {
        case ID_MAPPING_MESSAGE:
          {
            UniqueID current_id, original_id;
            derez.deserialize(current_id);
            derez.deserialize(original_id);
            // Update the ID map
            original_mappings[current_id] = original_id;
            // Trigger any pending events
            std::map<UniqueID,MapperEvent>::iterator finder = 
              pending_task_ids.find(current_id);
            if (finder != pending_task_ids.end())
            {
              MapperEvent to_trigger = finder->second;
              pending_task_ids.erase(finder);
              runtime->trigger_mapper_event(ctx, to_trigger);
            }
            break;
          }
        case INSTANCE_MAPPING_MESSAGE:
          {
            unsigned long current_id, original_id;
            derez.deserialize(current_id);
            derez.deserialize(original_id);
            original_instances[current_id] = original_id;
            // Trigger any pending events
            std::map<unsigned long,MapperEvent>::iterator finder = 
              pending_instance_ids.find(current_id);
            if (finder != pending_instance_ids.end())
            {
              MapperEvent to_trigger = finder->second;
              pending_instance_ids.erase(finder);
              runtime->trigger_mapper_event(ctx, to_trigger);
            }
            break;
          }
        case CREATE_INSTANCE_MESSAGE:
          {
            unsigned long original_id;
            derez.deserialize(original_id);
            LogicalRegion handle;
            derez.deserialize(handle);
            std::map<unsigned long,InstanceInfo*>::const_iterator finder = 
              instance_infos.find(original_id);
            assert(finder != instance_infos.end());
            finder->second->create_instance(runtime, ctx, handle);
            break;
          }
        case INSTANCE_CREATION_MESSAGE:
          {
            unsigned long original_id;
            derez.deserialize(original_id);
            PhysicalInstance instance;
            runtime->unpack_physical_instance(ctx, derez, instance);
            std::map<unsigned long,InstanceInfo*>::const_iterator finder = 
              instance_infos.find(original_id);
            assert(finder != instance_infos.end());
            finder->second->record_created_instance(runtime, ctx, instance); 
            update_original_instance_id(ctx, 
                instance.get_instance_id(), original_id);
            break;
          }
        case DECREMENT_USE_MESSAGE:
          {
            unsigned long original_id;
            derez.deserialize(original_id);
            std::map<unsigned long,InstanceInfo*>::const_iterator finder = 
              instance_infos.find(original_id);
            assert(finder != instance_infos.end());
            finder->second->decrement_use_count(runtime, ctx, false);
            break;
          }
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::handle_task_result(const MapperContext           ctx,
                                          const MapperTaskResult&       result)
    //--------------------------------------------------------------------------
    {
      // Nothing to do since we should never get one of these
      assert(false);
    }

    //--------------------------------------------------------------------------
    unsigned long ReplayMapper::find_original_instance_id(MapperContext ctx,
                                                       unsigned long current_id)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned long,unsigned long>::const_iterator finder = 
        original_instances.find(current_id);
      if (finder != original_instances.end())
        return finder->second;
      // Otherwise wait, see if someone else is already waiting
      std::map<unsigned long,MapperEvent>::const_iterator wait_finder = 
        pending_instance_ids.find(current_id);
      if (wait_finder == pending_instance_ids.end())
      {
        MapperEvent wait_on = runtime->create_mapper_event(ctx);
        pending_instance_ids[current_id] = wait_on;
        runtime->wait_on_mapper_event(ctx, wait_on);
      }
      else
        runtime->wait_on_mapper_event(ctx, wait_finder->second);
      finder = original_instances.find(current_id);
      // When we wake up, it should be there
      assert(finder != original_instances.end());
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::update_original_instance_id(MapperContext ctx,
                            unsigned long current_id, unsigned long original_id)
    //--------------------------------------------------------------------------
    {
      original_instances[current_id] = original_id;
      std::map<unsigned long,MapperEvent>::iterator finder = 
        pending_instance_ids.find(current_id);
      if (finder != pending_instance_ids.end())
      {
        MapperEvent to_trigger = finder->second;
        pending_instance_ids.erase(finder);
        runtime->trigger_mapper_event(ctx, to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    ReplayMapper::InstanceInfo* ReplayMapper::unpack_instance(FILE *f) const
    //--------------------------------------------------------------------------
    {
      InstanceInfo *info = new InstanceInfo();
      ignore_result(fread(&info->original_id, sizeof(info->original_id), 1, f));
      ignore_result(fread(&info->num_uses, sizeof(info->num_uses), 1, f));
      ignore_result(fread(&info->creator.id, sizeof(info->creator.id), 1, f));
      info->is_owner = (info->creator == local_proc);
      ignore_result(fread(&info->target_memory.id, 
                    sizeof(info->target_memory.id), 1, f)); 
      // Unpack the layout constraints 
      LayoutConstraintSet &layout = info->layout_constraints;  
      {
        SpecializedConstraint &spec = layout.specialized_constraint;
        ignore_result(fread(&spec.kind, sizeof(spec.kind), 1, f));
        ignore_result(fread(&spec.redop, sizeof(spec.redop), 1, f));
      }
      {
        MemoryConstraint &mem = layout.memory_constraint;
        ignore_result(fread(&mem.kind, sizeof(mem.kind), 1, f));
        mem.has_kind = true;
      }
      {
        FieldConstraint &fields = layout.field_constraint;
        unsigned num_fields;
        ignore_result(fread(&num_fields, sizeof(num_fields), 1, f));
        fields.field_set.resize(num_fields);
        for (unsigned idx = 0; idx < num_fields; idx++)
          ignore_result(fread(&fields.field_set[idx], 
                        sizeof(fields.field_set[idx]), 1, f));
        unsigned contiguous;
        ignore_result(fread(&contiguous, sizeof(contiguous), 1, f));
        fields.contiguous = (contiguous != 0);
        unsigned inorder;
        ignore_result(fread(&inorder, sizeof(inorder), 1, f));
        fields.inorder = (inorder != 0);
      }
      {
        OrderingConstraint &order = layout.ordering_constraint;
        unsigned num_dims;
        ignore_result(fread(&num_dims, sizeof(num_dims), 1, f));
        order.ordering.resize(num_dims);
        for (unsigned idx = 0; idx < num_dims; idx++)
          ignore_result(fread(&order.ordering[idx], 
                        sizeof(order.ordering[idx]), 1, f));
        unsigned contiguous;
        ignore_result(fread(&contiguous, sizeof(contiguous), 1, f));
        order.contiguous = (contiguous != 0);
      }
      {
        unsigned num_constraints;
        ignore_result(fread(&num_constraints, 
                            sizeof(num_constraints), 1, f));
        layout.splitting_constraints.resize(num_constraints);
        for (unsigned idx = 0; idx < num_constraints; idx++)
        {
          SplittingConstraint &split = layout.splitting_constraints[idx];
          ignore_result(fread(&split.kind, sizeof(split.kind), 1, f));
          ignore_result(fread(&split.value, sizeof(split.value), 1, f));
          unsigned chunks;
          ignore_result(fread(&chunks, sizeof(chunks), 1, f));
          split.chunks = (chunks != 0);
        }
      }
      {
        unsigned num_constraints;
        ignore_result(fread(&num_constraints, 
                            sizeof(num_constraints), 1, f));
        layout.dimension_constraints.resize(num_constraints);
        for (unsigned idx = 0; idx < num_constraints; idx++)
        {
          DimensionConstraint &dim = layout.dimension_constraints[idx];
          ignore_result(fread(&dim.kind, sizeof(dim.kind), 1, f));
          ignore_result(fread(&dim.eqk, sizeof(dim.eqk), 1, f));
          ignore_result(fread(&dim.value, sizeof(dim.value), 1, f));
        }
      }
      {
        unsigned num_constraints;
        ignore_result(fread(&num_constraints, 
                            sizeof(num_constraints), 1, f));
        layout.alignment_constraints.resize(num_constraints);
        for (unsigned idx = 0; idx < num_constraints; idx++)
        {
          AlignmentConstraint &align = layout.alignment_constraints[idx];
          ignore_result(fread(&align.fid, sizeof(align.fid), 1, f));
          ignore_result(fread(&align.eqk, sizeof(align.eqk), 1, f));
          ignore_result(fread(&align.alignment, 
                              sizeof(align.alignment), 1, f));
        }
      }
      {
        unsigned num_constraints;
        ignore_result(fread(&num_constraints, sizeof(num_constraints), 1, f));
        layout.offset_constraints.resize(num_constraints);
        for (unsigned idx = 0; idx < num_constraints; idx++)
        {
          OffsetConstraint &offset = layout.offset_constraints[idx]; 
          ignore_result(fread(&offset.fid, sizeof(offset.fid), 1, f));
          ignore_result(fread(&offset.offset, sizeof(offset.offset), 1, f));
        }
      }
      // Unpack the paths for describing the logical subregions
      unsigned num_paths;
      ignore_result(fread(&num_paths, sizeof(num_paths), 1, f));
      info->region_paths.resize(num_paths);
      for (unsigned nidx = 0; nidx < num_paths; nidx++)
      {
        std::vector<DomainPoint> &path = info->region_paths[nidx];
        unsigned num_points;
        ignore_result(fread(&num_points, sizeof(num_points), 1, f));
        path.resize(num_points);
        for (unsigned pidx = 0; pidx < num_points; pidx++)
        {
          DomainPoint &point = path[pidx]; 
          ignore_result(fread(&point.dim, sizeof(point.dim), 1, f));
          for (int i = 0; i < point.dim; i++)
            ignore_result(fread(point.point_data+i, 
                          sizeof(point.point_data[i]), 1, f));
        }
      }
      return info;
    }

    //--------------------------------------------------------------------------
    ReplayMapper::TaskMappingInfo* 
                                ReplayMapper::unpack_task_mapping(FILE *f) const
    //--------------------------------------------------------------------------
    {
      TaskMappingInfo *info = new TaskMappingInfo();
      ignore_result(fread(&info->original_unique_id, 
                    sizeof(info->original_unique_id), 1, f));
      ignore_result(fread(&info->target_proc, sizeof(info->target_proc), 1, f));
      ignore_result(fread(&info->priority, sizeof(info->priority), 1, f));
      ignore_result(fread(&info->variant, sizeof(info->variant), 1, f));
      unsigned num_premappings;
      ignore_result(fread(&num_premappings, sizeof(num_premappings), 1, f));
      for (unsigned idx = 0; idx < num_premappings; idx++)
      {
        unsigned index;
        ignore_result(fread(&index, sizeof(index), 1, f));
        info->premappings[index] = unpack_requirement(f);
      }
      unsigned num_mappings;
      ignore_result(fread(&num_mappings, sizeof(num_mappings), 1, f));
      info->mappings.resize(num_mappings);
      for (unsigned idx = 0; idx < num_mappings; idx++)
        info->mappings[idx] = unpack_requirement(f);
      unsigned num_postmappings;
      ignore_result(fread(&num_postmappings, sizeof(num_postmappings), 1, f));
      for (unsigned idx = 0; idx < num_postmappings; idx++)
      {
        unsigned index;
        ignore_result(fread(&index, sizeof(index), 1, f));
        info->postmappings[index] = unpack_requirement(f);
      }
      unsigned num_temporaries;
      ignore_result(fread(&num_temporaries, sizeof(num_temporaries), 1, f));
      for (unsigned idx = 0; idx < num_temporaries; idx++)
      {
        unsigned index;
        ignore_result(fread(&index, sizeof(index), 1, f));
        info->temporaries[index] = unpack_temporary(f);
      }
      unsigned num_tunables;
      ignore_result(fread(&num_tunables, sizeof(num_tunables), 1, f));
      info->tunables.resize(num_tunables);
      for (unsigned idx = 0; idx < num_tunables; idx++)
      {
        unsigned index;
        ignore_result(fread(&index, sizeof(index), 1, f));
        info->tunables[index] = unpack_tunable(f);
      }
      unsigned num_operations;
      ignore_result(fread(&num_operations, sizeof(num_operations), 1, f));
      info->operation_ids.resize(num_operations);
      for (unsigned idx = 0; idx < num_operations; idx++)
        ignore_result(fread(&info->operation_ids[idx], 
                            sizeof(info->operation_ids[idx]), 1, f));
      unsigned num_closes;
      ignore_result(fread(&num_closes, sizeof(num_closes), 1, f));
      info->close_ids.resize(num_closes);
      for (unsigned idx = 0; idx < num_closes; idx++)
        ignore_result(fread(&info->close_ids[idx], 
                            sizeof(info->close_ids[idx]), 1, f));
      return info;
    }

    //--------------------------------------------------------------------------
    ReplayMapper::InlineMappingInfo* 
                              ReplayMapper::unpack_inline_mapping(FILE *f) const
    //--------------------------------------------------------------------------
    {
      InlineMappingInfo *info = new InlineMappingInfo();
      unsigned num_mappings;
      ignore_result(fread(&num_mappings, sizeof(num_mappings), 1, f));
      assert((num_mappings == 0) || (num_mappings == 1));
      if (num_mappings == 1)
        info->mapping = unpack_requirement(f);
      else
        info->mapping = NULL;
      unsigned num_temporaries;
      ignore_result(fread(&num_temporaries, sizeof(num_temporaries), 1, f));
      assert((num_temporaries == 0) || (num_temporaries == 1));
      if (num_temporaries == 1)
        info->temporary = unpack_temporary(f);
      else
        info->temporary = NULL;
      return info;
    }

    //--------------------------------------------------------------------------
    ReplayMapper::CopyMappingInfo* 
                                ReplayMapper::unpack_copy_mapping(FILE *f) const
    //--------------------------------------------------------------------------
    {
      CopyMappingInfo *info = new CopyMappingInfo();
      unsigned num_src_mappings;
      ignore_result(fread(&num_src_mappings, sizeof(num_src_mappings), 1, f));
      info->src_mappings.resize(num_src_mappings);
      for (unsigned idx = 0; idx < num_src_mappings; idx++)
        info->src_mappings[idx] = unpack_requirement(f);
      unsigned num_dst_mappings;
      ignore_result(fread(&num_dst_mappings, sizeof(num_dst_mappings), 1, f));
      for (unsigned idx = 0; idx < num_dst_mappings; idx++)
        info->dst_mappings[idx] = unpack_requirement(f);
      unsigned num_src_temporaries;
      ignore_result(fread(&num_src_temporaries, 
                          sizeof(num_src_temporaries), 1, f));
      for (unsigned idx = 0; idx < num_src_temporaries; idx++)
      {
        unsigned index;
        ignore_result(fread(&index, sizeof(index), 1, f));
        info->src_temporaries[index] = unpack_temporary(f);
      }
      unsigned num_dst_temporaries;
      ignore_result(fread(&num_dst_temporaries, 
                          sizeof(num_dst_temporaries), 1, f));
      for (unsigned idx = 0; idx < num_dst_temporaries; idx++)
      {
        unsigned index;
        ignore_result(fread(&index, sizeof(index), 1, f));
        info->dst_temporaries[index] = unpack_temporary(f);
      }
      return info;
    }

    //--------------------------------------------------------------------------
    ReplayMapper::CloseMappingInfo* 
                               ReplayMapper::unpack_close_mapping(FILE *f) const
    //--------------------------------------------------------------------------
    {
      CloseMappingInfo *info = new CloseMappingInfo();
      unsigned num_mappings;
      ignore_result(fread(&num_mappings, sizeof(num_mappings), 1, f));
      assert((num_mappings == 0) || (num_mappings == 1));
      if (num_mappings == 1)
        info->mapping = unpack_requirement(f);
      else
        info->mapping = NULL;
      unsigned num_temporaries;
      ignore_result(fread(&num_temporaries, sizeof(num_temporaries), 1, f));
      assert((num_temporaries == 0) || (num_temporaries == 1));
      if (num_temporaries == 1)
        info->temporary = unpack_temporary(f);
      else
        info->temporary = NULL;
      return info;
    }

    //--------------------------------------------------------------------------
    ReplayMapper::ReleaseMappingInfo* 
                             ReplayMapper::unpack_release_mapping(FILE *f) const
    //--------------------------------------------------------------------------
    {
      ReleaseMappingInfo *info = new ReleaseMappingInfo();
      unsigned num_temporaries;
      ignore_result(fread(&num_temporaries, sizeof(num_temporaries), 1, f));
      assert((num_temporaries == 0) || (num_temporaries == 1));
      if (num_temporaries == 1)
        info->temporary = unpack_temporary(f);
      else
        info->temporary = NULL;
      return info;
    }

    //--------------------------------------------------------------------------
    ReplayMapper::RequirementMapping* 
                                 ReplayMapper::unpack_requirement(FILE *f) const
    //--------------------------------------------------------------------------
    {
      RequirementMapping *req = new RequirementMapping(); 
      unsigned num_instances;
      ignore_result(fread(&num_instances, sizeof(num_instances), 1, f));
      req->instances.resize(num_instances);
      for (unsigned idx = 0; idx < num_instances; idx++)
      {
        unsigned long original_id;
        ignore_result(fread(&original_id, sizeof(original_id), 1, f));
        std::map<unsigned long,InstanceInfo*>::const_iterator finder = 
          instance_infos.find(original_id);
        assert(finder != instance_infos.end());
        req->instances[idx] = finder->second;
      }
      return req;
    }

    //--------------------------------------------------------------------------
    ReplayMapper::TemporaryMapping*
                                   ReplayMapper::unpack_temporary(FILE *f) const
    //--------------------------------------------------------------------------
    {
      TemporaryMapping *temp = new TemporaryMapping();
      unsigned num_instances;
      ignore_result(fread(&num_instances, sizeof(num_instances), 1, f));
      for (unsigned idx = 0; idx < num_instances; idx++)
      {
        unsigned long original_dst;
        ignore_result(fread(&original_dst, sizeof(original_dst), 1, f));
        unsigned long original_id;
        ignore_result(fread(&original_id, sizeof(original_id), 1, f));
        std::map<unsigned long,InstanceInfo*>::const_iterator finder = 
          instance_infos.find(original_id);
        assert(finder != instance_infos.end());
        temp->instances[original_dst] = finder->second;
      }
      return temp;
    }

    //--------------------------------------------------------------------------
    ReplayMapper::TunableMapping* ReplayMapper::unpack_tunable(FILE *f) const
    //--------------------------------------------------------------------------
    {
      TunableMapping *tunable = new TunableMapping();
      ignore_result(fread(&tunable->tunable_size, 
                          sizeof(tunable->tunable_size), 1, f));
      tunable->tunable_value = malloc(tunable->tunable_size);
      unsigned string_length;
      ignore_result(fread(&string_length, sizeof(string_length), 1, f));
      char *string = (char*)malloc(string_length);
      ignore_result(fread(string, string_length, 1, f));
      // Now convert the hex string back into the value
      unsigned byte_index = 0;
      unsigned *target = (unsigned*)tunable->tunable_value;
      for (unsigned word_idx = 0; 
            word_idx < (tunable->tunable_size/4); word_idx++)
      {
        unsigned next_word = 0;
        for (unsigned i = 0; i < (8*sizeof(next_word)/4); i++, byte_index++)
        {
          unsigned next = 0;
          if (string[byte_index] >= 'A')
            next = (string[byte_index] - 'A') + 10;
          else
            next = string[byte_index] - '0';
          next_word |= (next << (i*4));
        }
        target[word_idx] = next_word;
      }

      free(string);
      return tunable;
    }

    //--------------------------------------------------------------------------
    ReplayMapper::TaskMappingInfo* ReplayMapper::find_task_mapping(
         MapperContext ctx, const Task &task, const DomainPoint &p, bool parent)
    //--------------------------------------------------------------------------
    {
      UniqueID unique_id = task.get_unique_id();
      // First check to see if we've already got it
      std::map<UniqueID,UniqueID>::const_iterator finder = 
        original_mappings.find(unique_id);
      if (finder != original_mappings.end())
      {
        // We've already got it
        std::pair<UniqueID,DomainPoint> key(finder->second, p);
        assert(task_mappings.find(key) != task_mappings.end());
        return task_mappings[key];
      }
      if (parent)
      {
        // We're doing the lookup for a parent task mapping
        // These come from broadcasts, so wait for it
        // See if someone else is already waiting
        std::map<UniqueID,MapperEvent>::const_iterator wait_finder = 
          pending_task_ids.find(unique_id);
        if (wait_finder == pending_task_ids.end())
        {
          MapperEvent wait_on = runtime->create_mapper_event(ctx);
          pending_task_ids[unique_id] = wait_on;
          runtime->wait_on_mapper_event(ctx, wait_on);
        }
        else
          runtime->wait_on_mapper_event(ctx, wait_finder->second);
        // When we wake up it should be there
        assert(original_mappings.find(unique_id) != original_mappings.end());
        std::pair<UniqueID,DomainPoint> key(original_mappings[unique_id], p);
        assert(task_mappings.find(key) != task_mappings.end());
        return task_mappings[key];
      }
      else if (task.get_depth() == 0)
      {
        // Handle the root case
        std::pair<UniqueID,DomainPoint> key(top_level_id, p);
        assert(task_mappings.find(key) != task_mappings.end());
        // Save the ID 
        original_mappings[unique_id] = top_level_id;
        return task_mappings[key];
      }
      else
      {

        // We're doing the lookup for a child task mapping
        // Find the parent task ID mapping
        TaskMappingInfo *parent_info = 
          find_task_mapping(ctx, *task.parent_task, 
                            task.parent_task->index_point, true/*parent*/);
        // Now that we've got the parent, look up our original operation id
        unsigned operation_index = task.get_context_index();
        assert(operation_index < parent_info->operation_ids.size());
        const UniqueID original_id = parent_info->operation_ids[operation_index];
        original_mappings[unique_id] = original_id;
        if (task.is_index_space)
        {
          // If this is an index task, just return the first one
          // with the proper ID
          for (std::map<std::pair<UniqueID,DomainPoint>,
                        TaskMappingInfo*>::const_iterator it = 
                task_mappings.begin(); it != task_mappings.end(); it++)
          {
            if (it->first.first == original_id)
              return it->second;
          }
          assert(false);
        }
        // Single task case
        std::pair<UniqueID,DomainPoint> key(original_id, p);
        assert(task_mappings.find(key) != task_mappings.end());
        // Save the ID
        return task_mappings[key];
      }
    }

    //--------------------------------------------------------------------------
    ReplayMapper::InlineMappingInfo* ReplayMapper::find_inline_mapping(
                              MapperContext ctx, const InlineMapping &inline_op)
    //--------------------------------------------------------------------------
    {
      UniqueID unique_id = inline_op.get_unique_id();
      // Check to see if we've already got it
      std::map<UniqueID,UniqueID>::const_iterator finder = 
        original_mappings.find(unique_id);
      if (finder != original_mappings.end())
      {
        assert(inline_mappings.find(finder->second) != inline_mappings.end());
        return inline_mappings[finder->second];
      }
      TaskMappingInfo *parent_info = 
        find_task_mapping(ctx, *inline_op.parent_task, 
                          inline_op.parent_task->index_point, true/*parent*/);
      // Now that we've got the parent, look up our original operation id
      unsigned operation_index = inline_op.get_context_index();
      assert(operation_index < parent_info->operation_ids.size());
      UniqueID original_id = parent_info->operation_ids[operation_index];
      assert(inline_mappings.find(original_id) != inline_mappings.end());
      // Save the ID
      original_mappings[unique_id] = original_id;
      return inline_mappings[original_id];
    }

    //--------------------------------------------------------------------------
    ReplayMapper::CopyMappingInfo* ReplayMapper::find_copy_mapping(
                                            MapperContext ctx, const Copy &copy)
    //--------------------------------------------------------------------------
    {
      UniqueID unique_id = copy.get_unique_id();
      // Check to see if we've already got it
      std::map<UniqueID,UniqueID>::const_iterator finder = 
        original_mappings.find(unique_id);
      if (finder != original_mappings.end())
      {
        assert(copy_mappings.find(finder->second) != copy_mappings.end());
        return copy_mappings[finder->second];
      }
      TaskMappingInfo *parent_info = 
        find_task_mapping(ctx, *copy.parent_task, 
                          copy.parent_task->index_point, true/*parent*/);
      // Now that we've got the parent, look up our original operation id
      unsigned operation_index = copy.get_context_index();
      assert(operation_index < parent_info->operation_ids.size());
      UniqueID original_id = parent_info->operation_ids[operation_index];
      assert(copy_mappings.find(original_id) != copy_mappings.end());
      // Save the ID
      original_mappings[unique_id] = original_id;
      return copy_mappings[original_id];
    }

    //--------------------------------------------------------------------------
    ReplayMapper::CloseMappingInfo* ReplayMapper::find_close_mapping(
                                          MapperContext ctx, const Close &close)
    //--------------------------------------------------------------------------
    {
      UniqueID unique_id = close.get_unique_id();
      // Check to see if we've already got it
      std::map<UniqueID,UniqueID>::const_iterator finder = 
        original_mappings.find(unique_id);
      if (finder != original_mappings.end())
      {
        assert(close_mappings.find(finder->second) != close_mappings.end());
        return close_mappings[finder->second];
      }
      TaskMappingInfo *parent_info = 
        find_task_mapping(ctx, *close.parent_task, 
                          close.parent_task->index_point, true/*parent*/);
      // Now that we've got the parent, look up our original operation id
      unsigned operation_index = close.get_context_index();
      assert(operation_index < parent_info->close_ids.size());
      UniqueID original_id = parent_info->close_ids[operation_index];
      assert(close_mappings.find(original_id) != close_mappings.end());
      // Save the ID
      original_mappings[unique_id] = original_id;
      return close_mappings[original_id];
    }

    //--------------------------------------------------------------------------
    ReplayMapper::ReleaseMappingInfo* ReplayMapper::find_release_mapping(
                                      MapperContext ctx, const Release &release)
    //--------------------------------------------------------------------------
    {
      UniqueID unique_id = release.get_unique_id();
      // Check to see if we've already got it
      std::map<UniqueID,UniqueID>::const_iterator finder = 
        original_mappings.find(unique_id);
      if (finder != original_mappings.end())
      {
        assert(release_mappings.find(finder->second) != release_mappings.end());
        return release_mappings[finder->second];
      }
      TaskMappingInfo *parent_info = 
        find_task_mapping(ctx, *release.parent_task,
                          release.parent_task->index_point, true/*parent*/);
      // Now that we've got the parent, look up our original operation id
      unsigned operation_index = release.get_context_index();
      assert(operation_index < parent_info->operation_ids.size());
      UniqueID original_id = parent_info->operation_ids[operation_index];
      assert(release_mappings.find(original_id) != release_mappings.end());
      // Save the ID
      original_mappings[unique_id] = original_id;
      return release_mappings[original_id];
    }

    //--------------------------------------------------------------------------
    PhysicalInstance ReplayMapper::InstanceInfo::get_instance(
                MapperRuntime *runtime, MapperContext ctx, LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // If it is not valid, we either need to make it if we are the
      // owner or send a message to the owner to request to make it
      if (!instance_valid)
        create_instance(runtime, ctx, handle);
      // Acquire it
      if (!runtime->acquire_instance(ctx, instance))
      {
        log_replay.error("Failed to acquire instance");
        assert(false);
      }
      // Update the use count
      decrement_use_count(runtime, ctx, true);
      return instance;
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::InstanceInfo::create_instance(MapperRuntime *runtime,
                                        MapperContext ctx, LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // Handle duplicate create requests
      if (instance_valid || creating_instance)
        return;
      if (is_owner)
      {
        // Mark that we are about to try to make the instance
        // so we can avoid duplicate creation attempts
        creating_instance = true;
        // Get the vector of LogicalRegions
        // First go up the tree until we know we are at the root
        LogicalRegion root = handle;
        while (runtime->has_parent_logical_partition(ctx, root))
        {
          LogicalPartition part = 
            runtime->get_parent_logical_partition(ctx, root);
          root = runtime->get_parent_logical_region(ctx, part);
        }
        // Now we've got the root, so we can find the proper subregions
        assert(!region_paths.empty());
        std::vector<LogicalRegion> regions(region_paths.size());
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          LogicalRegion next = root;
          const std::vector<DomainPoint> &path = region_paths[idx]; 
          assert((path.size() % 2) == 0);
          for (unsigned i = 0; i < path.size(); i+=2)
          {
            LogicalPartition part = 
              runtime->get_logical_partition_by_color(ctx, next, path[i]);
            next = runtime->get_logical_subregion_by_color(ctx, part,path[i+1]);
          }
          regions[idx] = next;
        }
        // Make the instance
        if (!runtime->create_physical_instance(ctx, target_memory, 
              layout_constraints, regions, instance, 
              false/*acquire*/, GC_NEVER_PRIORITY))
        {
          log_replay.error("Failed to create instance");
          assert(false);
        }
        instance_valid = true;
        creating_instance = false;
        // Broadcast the instance to all the other mappers
        Legion::Serializer rez;
        rez.serialize(original_id);
        runtime->pack_physical_instance(ctx, rez, instance);
        runtime->broadcast(ctx, rez.get_buffer(), 
                           rez.get_used_bytes(), INSTANCE_CREATION_MESSAGE); 
      }
      else
      {
        // Send a request to make it if one hasn't been sent yet
        if (!request_event.exists())
        {
          // Make the event before sending the message
          request_event = runtime->create_mapper_event(ctx);
          Legion::Serializer rez;
          rez.serialize(original_id);
          rez.serialize(handle);
          runtime->send_message(ctx, creator, rez.get_buffer(), 
                                rez.get_used_bytes(), CREATE_INSTANCE_MESSAGE);
        }
        runtime->wait_on_mapper_event(ctx, request_event);
      }
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::InstanceInfo::record_created_instance(
             MapperRuntime *runtime, MapperContext ctx, PhysicalInstance result)
    //--------------------------------------------------------------------------
    {
      // no need to do this if we are the owner
      if (is_owner)
        return;
      instance = result;
      instance_valid = true;
      // See if we had a request event that needs to be triggered
      if (request_event.exists())
        runtime->trigger_mapper_event(ctx, request_event);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::InstanceInfo::decrement_use_count(MapperRuntime *runtime,
                                                  MapperContext ctx, bool first)
    //--------------------------------------------------------------------------
    {
      if (is_owner)
      {
        // Legion spy isn't computing use counts properly
        // TODO: fix this
#if 0
        assert(num_uses > 0);
        num_uses--;
#endif
        // If we're done using it, we can set a high GC priority
        if (num_uses == 0)
          runtime->set_garbage_collection_priority(ctx, instance, 
                                                   GC_MAX_PRIORITY);
      }
      else if (first)
      {
        // Send a user decrement message
        runtime->broadcast(ctx, &original_id, 
                           sizeof(original_id), DECREMENT_USE_MESSAGE);
      }
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::RequirementMapping::map_requirement(
                   MapperRuntime *runtime, MapperContext ctx, 
                   LogicalRegion handle, std::vector<PhysicalInstance> &targets)
    //--------------------------------------------------------------------------
    {
      // We'll always put the virtual instance at the end
      targets.resize(instances.size()+1);
      for (unsigned idx = 0; idx < instances.size(); idx++)
        targets[idx] = instances[idx]->get_instance(runtime, ctx, handle);
      targets[instances.size()] = PhysicalInstance::get_virtual_instance();
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::TemporaryMapping::map_temporary(MapperRuntime *runtime,
                           MapperContext ctx, LogicalRegion handle, 
                           unsigned long original_dst, PhysicalInstance &result)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned long,InstanceInfo*>::const_iterator finder = 
        instances.find(original_dst);
      assert(finder != instances.end());
      result = finder->second->get_instance(runtime, ctx, handle);
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::TunableMapping::set_tunable(void *&value, size_t &size)
    //--------------------------------------------------------------------------
    {
      size = tunable_size;
      value = malloc(tunable_size);
      memcpy(value, tunable_value, tunable_size);
    }

  }; // namespace Mapping 
}; // namespace Legion

// EOF

