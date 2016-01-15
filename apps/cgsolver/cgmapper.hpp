/* Copyright 2016 Stanford University
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

#ifndef cgmapper_hpp
#define cgmapper_hpp

#include <iostream>
#include "legion.h"
#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;

enum {
  SUBREGION_TUNABLE,
  PREDICATED_TUNABLE,
};

enum {
  PARTITIONING_MAPPER_ID,
};

class PartitioningMapper : public DefaultMapper {
public:
  PartitioningMapper(Machine machine, HighLevelRuntime *rt, Processor local);
public:
  virtual bool map_task(Task *task);  	
  virtual bool map_copy(Copy *copy);
  virtual bool map_inline(Inline *inline_operation);
  virtual int get_tunable_value(const Task *task, TunableID tid, MappingTagID tag);
  virtual bool rank_copy_targets(const Mappable *mappable,
                                 LogicalRegion rebuild_region,
                                 const std::set<Memory> &current_instances,
                                 bool complete,
                                 size_t max_blocking_factor,
                                 std::set<Memory> &to_reuse,
                                 std::vector<Memory> &to_create,
                                 bool &create_one,
                                 size_t &blocking_factor);
};

void mapper_registration(Machine machine, HighLevelRuntime *rt,
                          const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin(); it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(
        new PartitioningMapper(machine, rt, *it), *it);
  }
  return;
}


PartitioningMapper::PartitioningMapper(Machine m, HighLevelRuntime *rt, Processor p) 
  : DefaultMapper(m, rt, p)
{
    std::set<Processor> all_procs;
    machine.get_all_processors(all_procs);

    if ((*(all_procs.begin())) == local_proc) {
    
    printf("There are %ld processors:\n", all_procs.size());

    // get the list of all the memories available
    // on the target architecture and print out their info.
    std::set<Memory> all_mems; 
    machine.get_all_memories(all_mems);
    
    printf("There are %ld memories:\n", all_mems.size());
   
    std::set<Memory> vis_mems; 
    machine.get_visible_memories(local_proc, vis_mems);
    printf("There are %ld memories visible from processor %x\n",
        vis_mems.size(), local_proc.id);

  }
  return;
}

int PartitioningMapper::get_tunable_value(const Task *task,
                                          TunableID tid,
                                          MappingTagID tag)
{
   if (tid == SUBREGION_TUNABLE)
   {
     const std::set<Processor> &cpu_procs = 
       	   machine_interface.filter_processors(Processor::LOC_PROC);
     return cpu_procs.size();
   }
   else if (tid == PREDICATED_TUNABLE)
   {
      // This seems like a good value to hide latency for now
      return 4;
   }

   // Should never get here
   assert(false);
   return 0;
}

bool PartitioningMapper::map_task(Task *task) {

	// Put everything in the system memory
    	Memory sys_mem = 
      	machine_interface.find_memory_kind(task->target_proc,
                                           Memory::SYSTEM_MEM);
    	assert(sys_mem.exists());
    	for (unsigned idx = 0; idx < task->regions.size(); idx++)
    	{
      		task->regions[idx].target_ranking.push_back(sys_mem);
      		task->regions[idx].virtual_map = false;
      		task->regions[idx].enable_WAR_optimization = war_enabled;
      		task->regions[idx].reduction_list = false;
      		
		// make everything SOA
      		task->regions[idx].blocking_factor = 
        	task->regions[idx].max_blocking_factor;
       } 
    	return true;
}

bool PartitioningMapper::map_copy(Copy *copy) {
	
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
          		assert(copy->src_requirements[idx].current_instances.size() == 1);
          		Memory target = 
            		(copy->src_requirements[idx].current_instances.begin())->first;
          		copy->src_requirements[idx].target_ranking.push_back(target);
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
          		assert(copy->dst_requirements[idx].current_instances.size() == 1);
          		Memory target = 
            		(copy->dst_requirements[idx].current_instances.begin())->first;
          		copy->dst_requirements[idx].target_ranking.push_back(target);
        	}	
         
		// make it SOA 
		copy->src_requirements[idx].blocking_factor = 
            	copy->src_requirements[idx].max_blocking_factor;
          
		// make it SOA
		copy->dst_requirements[idx].blocking_factor = 
            	copy->dst_requirements[idx].max_blocking_factor; 
      }

      // No profiling on copies yet
      return true;
}

bool PartitioningMapper::map_inline(Inline *inline_operation) {

	// let the default mapper do its thing, 
  	bool ret = DefaultMapper::map_inline(inline_operation);
	
	// then override the blocking factor to force SOA
  	RegionRequirement& req = inline_operation->requirement;
  	req.blocking_factor = req.max_blocking_factor;
  
  	return ret;
}

bool PartitioningMapper::rank_copy_targets(const Mappable *mappable,
                                     LogicalRegion rebuild_region,
                                     const std::set<Memory> &current_instances,
                                     bool complete,
                                     size_t max_blocking_factor,
                                     std::set<Memory> &to_reuse,
                                     std::vector<Memory> &to_create,
                                     bool &create_one,
                                     size_t &blocking_factor)
{
	
      typedef std::set<Memory>::iterator it;
      const std::set<Memory> &sys_mem =
      machine_interface.filter_memories(Memory::SYSTEM_MEM);
       
      create_one = false;
      blocking_factor = max_blocking_factor;

      if(current_instances.empty()) {
      	
	for(it i=sys_mem.begin(); i != sys_mem.end(); i++) {

		to_create.push_back(*i);
	}
	
      }
      else
      {
	for(it i=current_instances.begin(); i != current_instances.end(); i++) {
	
		Memory::Kind kind = i->kind();
                if(kind == Memory::SYSTEM_MEM) {
			to_reuse.insert(*i);
		}
	}
	
	for(it i=sys_mem.begin(); i != sys_mem.end(); i++) {
		
		it pos = to_reuse.find(*i);
		if(pos == to_reuse.end() ){

			to_create.push_back(*i);
		}
	}

      }

      // Don't make any composite instances since they're 
      // not fully supported yet
      return false;
    
}
#endif
