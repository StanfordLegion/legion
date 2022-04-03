/* Copyright 2022 Stanford University
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


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"

#include "test_mapper.h"
#include "default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

/*
 * In this example, we perform the same
 * daxpy computation as example 07.  While
 * the source code for the actual computation
 * between the two examples is identical, we 
 * show to create a custom mapper that changes 
 * the mapping of the computation onto the hardware.
 */

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  DAXPY_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_Z,
};

enum {
  SUBREGION_TUNABLE,
};

enum {
  PARTITIONING_MAPPER_ID = 1,
};

/*
 * One of the primary goals of Legion is
 * to make it easy to remap applications
 * onto different hardware.  Up to this point
 * all of our applications have been mapped
 * by the DefaultMapper that we provide.  The
 * DefaultMapper class provides heuristics for
 * performing mappings that are good but often
 * not optimal for a specific application or
 * architecture.   By creating a custom mapper
 * programmers can make application- or
 * architecture-specific mapping decisions.
 * Furthermore, many custom mappers can be
 * used to map the same application onto many
 * different architectures without having to
 * modify the primary application code.
 *
 * A common concern when mapping applications
 * onto a target machine is whether or not
 * mapping impacts correctness.  In Legion
 * all mapping decisions are orthogonal to
 * correctness.  In cases when correctness may
 * be impacted by a mapping decision (e.g. a
 * mapper maps a physical region for a task
 * into a memory not visible from the target
 * processor), then the Legion runtime will
 * notify the mapper that it tried to perform
 * an illegal mapping and allow it to retry.
 *
 * To introduce how to write a custom mapper
 * we'll implement an adversarial mapper 
 * that makes random mapping decisions
 * designed to stress the Legion runtime. 
 * We'll report the chosen mapping decisions
 * to show that Legion computes the correct
 * answer regardless of the mapping.
 */

// Mappers are classes that implement the
// mapping interface declared in legion.h.
// Legion provides a default implementation
// of this interface defined by the
// DefaultMapper class.  Programmers can
// implement custom mappers either by 
// extending the DefaultMapper class
// or by declaring their own class which
// implements the mapping interface from
// scratch.  Here we will extend the
// DefaultMapper which means that we only
// need to override the methods that we
// want to in order to change our mapping.
// In this example, we'll override four
// of the mapping calls to illustrate
// how they work.
class AdversarialMapper : public TestMapper {
public:
  AdversarialMapper(Machine machine, 
      Runtime *rt, Processor local);
public:
  using TestMapper::report_profiling;
  virtual void select_task_options(const MapperContext    ctx,
				   const Task&            task,
				         TaskOptions&     output);
  virtual void slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                                SliceTaskOutput& output);
  virtual void map_task(const MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                              MapTaskOutput& output);
  virtual void report_profiling(const MapperContext      ctx,
				const Task&              task,
				const TaskProfilingInfo& input);
};

class PartitioningMapper : public DefaultMapper {
public:
  PartitioningMapper(Machine machine,
      Runtime *rt, Processor local);
public:
  virtual void select_tunable_value(const MapperContext ctx,
                                    const Task& task,
                                    const SelectTunableInput& input,
                                          SelectTunableOutput& output);
};

// Mappers are created after the Legion runtime
// starts but before the application begins 
// executing.  To create mappers the application
// registers a callback function for the runtime
// to perform prior to beginning execution.  In
// this example we call it the 'mapper_registration'
// function.  (See below for where we register
// this callback with the runtime.)  The callback
// function must have this type, which allows the
// runtime to pass the necessary parameters in
// for creating new mappers.
//
// In Legion, mappers are identified by a MapperID.
// Zero is reserved for the DefaultMapper, but 
// other mappers can replace it by using the 
// 'replace_default_mapper' call.  Other mappers
// can be registered with their own IDs using the
// 'add_mapper' method call on the runtime.
//
// The model for Legion is that there should always
// be one mapper for every processor in the system.
// This guarantees that there is never contention
// for the mappers because multiple processors need
// to access the same mapper object.  When the
// runtime invokes the 'mapper_registration' callback,
// it gives a list of local processors which 
// require a mapper if a new mapper class is to be
// created.  In a multi-node setting, the runtime
// passes in a subset of the processors for which
// mappers need to be created on the local node.
//
// Here we override the DefaultMapper ID so that
// all tasks that normally would have used the
// DefaultMapper will now use our AdversarialMapper.
// We create one new mapper for each processor
// and register it with the runtime.
void mapper_registration(Machine machine, Runtime *rt,
                          const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(
        new AdversarialMapper(machine, rt, *it), *it);
    rt->add_mapper(PARTITIONING_MAPPER_ID,
        new PartitioningMapper(machine, rt, *it), *it);
  }
}

// Here is the constructor for our adversarial mapper.
// We'll use the constructor to illustrate how mappers can
// get access to information regarding the current machine.
AdversarialMapper::AdversarialMapper(Machine m, 
                                     Runtime *rt, Processor p)
  : TestMapper(rt->get_mapper_runtime(), m, p) // pass arguments through to TestMapper 
{
  // The machine object is a singleton object that can be
  // used to get information about the underlying hardware.
  // The machine pointer will always be provided to all
  // mappers, but can be accessed anywhere by the static
  // member method Machine::get_machine().  Here we get
  // a reference to the set of all processors in the machine
  // from the machine object.  Note that the Machine object
  // actually comes from the Legion low-level runtime, most
  // of which is not directly needed by application code.
  // Typedefs in legion_types.h ensure that all necessary
  // types for querying the machine object are in scope
  // in the Legion namespace.
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  // Recall that we create one mapper for every processor.  We
  // only want to print out this information one time, so only
  // do it if we are the mapper for the first processor in the
  // list of all processors in the machine.
  if (all_procs.begin()->id + 1 == local_proc.id)
  {
    // Print out how many processors there are and each
    // of their kinds.
    printf("There are %zd processors:\n", all_procs.size());
    for (std::set<Processor>::const_iterator it = all_procs.begin();
          it != all_procs.end(); it++)
    {
      // For every processor there is an associated kind
      Processor::Kind kind = it->kind();
      switch (kind)
      {
        // Latency-optimized cores (LOCs) are CPUs
        case Processor::LOC_PROC:
          {
            printf("  Processor ID " IDFMT " is CPU\n", it->id); 
            break;
          }
        // Throughput-optimized cores (TOCs) are GPUs
        case Processor::TOC_PROC:
          {
            printf("  Processor ID " IDFMT " is GPU\n", it->id);
            break;
          }
        // Processor for doing I/O
        case Processor::IO_PROC:
          {
            printf("  Processor ID " IDFMT " is I/O Proc\n", it->id);
            break;
          }
        // Utility processors are helper processors for
        // running Legion runtime meta-level tasks and 
        // should not be used for running application tasks
        case Processor::UTIL_PROC:
          {
            printf("  Processor ID " IDFMT " is utility\n", it->id);
            break;
          }
        default:
	  {
	    printf("  Processor " IDFMT " is unknown (kind=%d)\n", it->id, it->kind());
	    break;
	  }
      }
    }
    // We can also get the list of all the memories available
    // on the target architecture and print out their info.
    std::set<Memory> all_mems;
    machine.get_all_memories(all_mems);
    printf("There are %zd memories:\n", all_mems.size());
    for (std::set<Memory>::const_iterator it = all_mems.begin();
          it != all_mems.end(); it++)
    {
      Memory::Kind kind = it->kind();
      size_t memory_size_in_kb = it->capacity() >> 10;
      switch (kind)
      {
        // RDMA addressable memory when running with GASNet
        case Memory::GLOBAL_MEM:
          {
            printf("  GASNet Global Memory ID " IDFMT " has %zd KB\n", 
                    it->id, memory_size_in_kb);
            break;
          }
        // DRAM on a single node
        case Memory::SYSTEM_MEM:
          {
            printf("  System Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Pinned memory on a single node
        case Memory::REGDMA_MEM:
          {
            printf("  Pinned Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // A memory associated with a single socket
        case Memory::SOCKET_MEM:
          {
            printf("  Socket Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Zero-copy memory betweeen CPU DRAM and
        // all GPUs on a single node
        case Memory::Z_COPY_MEM:
          {
            printf("  Zero-Copy Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // GPU framebuffer memory for a single GPU
        case Memory::GPU_FB_MEM:
          {
            printf("  GPU Frame Buffer Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Disk memory on a single node
        case Memory::DISK_MEM:
          {
            printf("  Disk Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // HDF framebuffer memory for a single GPU
        case Memory::HDF_MEM:
          {
            printf("  HDF Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // File memory on a single node
        case Memory::FILE_MEM:
          {
            printf("  File Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Block of memory sized for L3 cache
        case Memory::LEVEL3_CACHE:
          {
            printf("  Level 3 Cache ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Block of memory sized for L2 cache
        case Memory::LEVEL2_CACHE:
          {
            printf("  Level 2 Cache ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Block of memory sized for L1 cache
        case Memory::LEVEL1_CACHE:
          {
            printf("  Level 1 Cache ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        default:
	  {
	    printf("  Memory " IDFMT " is unknown (kind=%d)\n", it->id, it->kind());
	    break;
	  }
      }
    }

    // The Legion machine model represented by the machine object
    // can be thought of as a graph with processors and memories
    // as the two kinds of nodes.  There are two kinds of edges
    // in this graph: processor-memory edges and memory-memory
    // edges.  An edge between a processor and a memory indicates
    // that the processor can directly perform load and store
    // operations to that memory.  Memory-memory edges indicate
    // that data movement can be directly performed between the
    // two memories.  To illustrate how this works we examine
    // all the memories visible to our local processor in 
    // this mapper.  We can get our set of visible memories
    // using the 'get_visible_memories' method on the machine.
    std::set<Memory> vis_mems;
    machine.get_visible_memories(local_proc, vis_mems);
    printf("There are %zd memories visible from processor " IDFMT "\n",
            vis_mems.size(), local_proc.id);
    for (std::set<Memory>::const_iterator it = vis_mems.begin();
          it != vis_mems.end(); it++)
    {
      // Edges between nodes are called affinities in the
      // machine model.  Affinities also come with approximate
      // indications of the latency and bandwidth between the 
      // two nodes.  Right now these are unit-less measurements,
      // but our plan is to teach the Legion runtime to profile
      // these values on start-up to give them real values
      // and further increase the portability of Legion applications.
      std::vector<ProcessorMemoryAffinity> affinities;
      int results = 
        machine.get_proc_mem_affinity(affinities, local_proc, *it);
      // We should only have found 1 results since we
      // explicitly specified both values.
      assert(results == 1);
      printf("  Memory " IDFMT " has bandwidth %d and latency %d\n",
              it->id, affinities[0].bandwidth, affinities[0].latency);
    }
  }
}

// The first mapper call that we override is the 
// select_task_options call.  This mapper call is
// performed on every task launch immediately after
// it is made.  The call asks the mapper to select 
// set properties of the task:
//
//  inline_task - whether the task should be directly
//    inlined into its parent task, using the parent
//    task's physical regions.  
//  spawn_task - whether the task is eligible for 
//    stealing (based on Cilk-style semantics)
//  map_locally - whether the task should be mapped
//    by the processor on which it was launched or
//    whether it should be mapped by the processor
//    where it will run.
//  profile_task - should the runtime collect profiling
//    information about the task while it is executing
//  target_proc - which processor should the task be
//    sent to once all of its mapping dependences have
//    been satisfied.
//
//  Note that these properties are all set on the Task
//  object declared in legion.h.  The Task object is
//  the representation of a task that the Legion runtime
//  provides to the mapper for specifying mapping
//  decisions.  Note that there are similar objects
//  for inline mappings as well as other operations.
//
//  For our adversarial mapper, we perform the default
//  choices for all options except the last one.  Here
//  we choose a random processor in our system to 
//  send the task to.
void AdversarialMapper::select_task_options(const MapperContext ctx,
                                            const Task& task,
                                                  TaskOptions& output)
{
  output.inline_task = false;
  output.stealable = false;
  output.map_locally = true;
  Processor::Kind kind = select_random_processor_kind(ctx, task.task_id);
  output.initial_proc = select_random_processor(kind);
}

// The second call that we override is the slice_domain
// method. The slice_domain call is used by the runtime
// to query the mapper about the best way to distribute
// the points in an index space task launch throughout
// the machine. The mapper is given the task and the domain
// to slice and then asked to generate sub-domains to be
// sent to different processors in the form of DomainSplit
// objects. DomainSplit objects describe the sub-domain,
// the target processor for the sub-domain, whether the
// generated slice can be stolen, and finally whether 
// slice_domain' should be recursively called on the
// slice when it arrives at its destination.
//
// In this example we use a utility method from the DefaultMapper
// called decompose_index_space to decompose our domain. We 
// recursively split the domain in half during each call of
// slice_domain and send each slice to a random processor.
// We continue doing this until the leaf domains have only
// a single point in them. This creates a tree of slices of
// depth log(N) in the number of points in the domain with
// each slice being sent to a random processor.
void AdversarialMapper::slice_task(const MapperContext      ctx,
                                   const Task&              task,
                                   const SliceTaskInput&    input,
                                         SliceTaskOutput&   output)
{
  // Iterate over all the points and send them all over the world
  output.slices.resize(input.domain.get_volume());
  unsigned idx = 0;
  switch (input.domain.get_dim())
  {
    case 1:
      {
        Rect<1> rect = input.domain;
        for (PointInRectIterator<1> pir(rect); pir(); pir++, idx++)
        {
          Rect<1> slice(*pir, *pir);
          output.slices[idx] = TaskSlice(slice,
              select_random_processor(task.target_proc.kind()),
              false/*recurse*/, true/*stealable*/);
        }
        break;
      }
    case 2:
      {
        Rect<2> rect = input.domain;
        for (PointInRectIterator<2> pir(rect); pir(); pir++, idx++)
        {
          Rect<2> slice(*pir, *pir);
          output.slices[idx] = TaskSlice(slice,
              select_random_processor(task.target_proc.kind()),
              false/*recurse*/, true/*stealable*/);
        }
        break;
      }
    case 3:
      {
        Rect<3> rect = input.domain;
        for (PointInRectIterator<3> pir(rect); pir(); pir++, idx++)
        {
          Rect<3> slice(*pir, *pir);
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

// The next mapping call that we override is the map_task
// mapper method. Once a task has been assigned to map on
// a specific processor (the target_proc) then this method
// is invoked by the runtime to select the memories in 
// which to create physical instances for each logical region.
// The mapper communicates this information to the runtime
// via the mapping fields on RegionRequirements. The memories
// containing currently valid physical instances for each
// logical region is provided by the runtime in the 
// 'current_instances' field. The mapper must specify an
// ordered list of memories for the runtime to try when
// creating a physical instance in the 'target_ranking'
// vector field of each RegionRequirement. The runtime
// attempts to find or make a physical instance in each 
// memory until it succeeds. If the runtime fails to find
// or make a physical instance in any of the memories, then
// the mapping fails and the mapper will be notified that
// the task failed to map using the 'notify_mapping_failed'
// mapper call. If the mapper does nothing, then the task
// is placed back on the list of tasks eligible to be mapped.
// There are other fields that the mapper can set in the
// process of the map_task call that we do not cover here.
//
// In this example, the mapper finds the set of all visible
// memories from the target processor and then puts them
// in a random order as the target set of memories, thereby
// challenging the Legion runtime to maintain correctness
// of data moved through random sets of memories.
void AdversarialMapper::map_task(const MapperContext         ctx,
                                 const Task&                 task,
                                 const MapTaskInput&         input,
                                       MapTaskOutput&        output)
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
    int chosen = default_generate_random_integer() % variants.size();
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
  for (unsigned idx = 0; idx < task.regions.size(); idx++)
  {
    if (premapped[idx])
      continue;
    if (task.regions[idx].is_restricted())
    {
      output.chosen_instances[idx] = input.valid_instances[idx];
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
  output.task_priority = default_generate_random_integer();

  // Finally, let's ask for some profiling data to see the impact of our choices
  {
    using namespace ProfilingMeasurements;
    output.task_prof_requests.add_measurement<OperationStatus>();
    output.task_prof_requests.add_measurement<OperationTimeline>();
    output.task_prof_requests.add_measurement<RuntimeOverhead>();
  }
}

void AdversarialMapper::report_profiling(const MapperContext      ctx,
					 const Task&              task,
					 const TaskProfilingInfo& input)
{
  // Local import of measurement names saves typing here without polluting
  // namespace for everybody else
  using namespace ProfilingMeasurements;

  // You are not guaranteed to get measurements you asked for, so make sure to
  // check the result of calls to get_measurement (or just call has_measurement
  // first).  Also, the call returns a copy of the result that you must delete
  // yourself.
  OperationStatus *status = 
    input.profiling_responses.get_measurement<OperationStatus>();
  if (status)
  {
    switch (status->result)
    {
      case OperationStatus::COMPLETED_SUCCESSFULLY:
        {
          printf("Task %s COMPLETED SUCCESSFULLY\n", task.get_task_name());
          break;
        }
      case OperationStatus::COMPLETED_WITH_ERRORS:
        {
          printf("Task %s COMPLETED WITH ERRORS\n", task.get_task_name());
          break;
        }
      case OperationStatus::INTERRUPT_REQUESTED:
        {
          printf("Task %s was INTERRUPTED\n", task.get_task_name());
          break;
        }
      case OperationStatus::TERMINATED_EARLY:
        {
          printf("Task %s TERMINATED EARLY\n", task.get_task_name());
          break;
        }
      case OperationStatus::CANCELLED:
        {
          printf("Task %s was CANCELLED\n", task.get_task_name());
          break;
        }
      default:
        assert(false); // shouldn't get any of the rest currently
    }
    delete status;
  }
  else
    printf("No operation status for task %s\n", task.get_task_name());

  OperationTimeline *timeline =
    input.profiling_responses.get_measurement<OperationTimeline>();
  if (timeline)
  {
    printf("Operation timeline for task %s: ready=%lld start=%lld stop=%lld\n",
	   task.get_task_name(),
	   timeline->ready_time,
	   timeline->start_time,
	   timeline->end_time);
    delete timeline;
  }
  else
    printf("No operation timeline for task %s\n", task.get_task_name());

  RuntimeOverhead *overhead =
    input.profiling_responses.get_measurement<RuntimeOverhead>();
  if (overhead)
  {
    long long total = (overhead->application_time +
		       overhead->runtime_time +
		       overhead->wait_time);
    if (total <= 0) total = 1;
    printf("Runtime overhead for task %s: runtime=%.1f%% wait=%.1f%%\n",
	   task.get_task_name(),
	   (100.0 * overhead->runtime_time / total),
	   (100.0 * overhead->wait_time / total));
    delete overhead;
  }
  else
    printf("No runtime overhead data for task %s\n", task.get_task_name()); 
}

PartitioningMapper::PartitioningMapper(Machine m,
                                       Runtime *rt,
                                       Processor p)
  : DefaultMapper(rt->get_mapper_runtime(), m, p)
{
}

void PartitioningMapper::select_tunable_value(const MapperContext ctx,
                                              const Task& task,
                                              const SelectTunableInput& input,
                                                    SelectTunableOutput& output)
{
  if (input.tunable_id == SUBREGION_TUNABLE)
  {
    Machine::ProcessorQuery all_procs(machine);
    all_procs.only_kind(Processor::LOC_PROC);
    runtime->pack_tunable<size_t>(all_procs.count(), output);
    return;
  }
  // Should never get here
  assert(false);
}

/*
 * Everything below here is the standard daxpy example
 * except for the registration of the callback function
 * for creating custom mappers which is explicitly commented
 * and the call to select_tunable_value to determine the number
 * of sub-regions.
 */
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 1024; 
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
    }
  }
  int num_subregions =
    runtime->select_tunable_value(ctx, SUBREGION_TUNABLE,
                                  PARTITIONING_MAPPER_ID).get_result<size_t>();

  printf("Running daxpy for %d elements...\n", num_elements);
  printf("Partitioning data into %d sub-regions...\n", num_subregions);

  Rect<1> elem_rect(0,num_elements-1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace input_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(double),FID_X);
    allocator.allocate_field(sizeof(double),FID_Y);
  }
  FieldSpace output_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, output_fs);
    allocator.allocate_field(sizeof(double),FID_Z);
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is, output_fs);

  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);

  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, ip);

  ArgumentMap arg_map;

  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, 
                              TaskArgument(NULL, 0), arg_map);
  init_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/, 
                        WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.add_field(0, FID_X);
  runtime->execute_index_space(ctx, init_launcher);

  init_launcher.region_requirements[0].privilege_fields.clear();
  init_launcher.region_requirements[0].instance_fields.clear();
  init_launcher.add_field(0, FID_Y);
  runtime->execute_index_space(ctx, init_launcher);

  const double alpha = drand48();
  IndexLauncher daxpy_launcher(DAXPY_TASK_ID, color_is,
                TaskArgument(&alpha, sizeof(alpha)), arg_map);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, input_lr));
  daxpy_launcher.add_field(0, FID_X);
  daxpy_launcher.add_field(0, FID_Y);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, output_lr));
  daxpy_launcher.add_field(1, FID_Z);
  runtime->execute_index_space(ctx, daxpy_launcher);
                    
  TaskLauncher check_launcher(CHECK_TASK_ID, TaskArgument(&alpha, sizeof(alpha)));
  check_launcher.add_region_requirement(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  check_launcher.region_requirements[0].add_field(FID_X);
  check_launcher.region_requirements[0].add_field(FID_Y);
  check_launcher.add_region_requirement(
      RegionRequirement(output_lr, READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.region_requirements[1].add_field(FID_Z);
  runtime->execute_task(ctx, check_launcher);

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_field_space(ctx, output_fs);
  runtime->destroy_index_space(ctx, is);
  runtime->destroy_index_space(ctx, color_is);
}

void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];
  printf("Initializing field %d for block %d...\n", fid, point);

  const FieldAccessor<WRITE_DISCARD,double,1> acc(regions[0], fid);
  // Note here that we get the domain for the subregion for
  // this task from the runtime which makes it safe for running
  // both as a single task and as part of an index space of tasks.
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc[*pir] = drand48();
}

void daxpy_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(double));
  const double alpha = *((const double*)task->args);
  const int point = task->index_point.point_data[0];

  const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);
  const FieldAccessor<READ_ONLY,double,1> acc_y(regions[0], FID_Y);
  const FieldAccessor<WRITE_DISCARD,double,1> acc_z(regions[1], FID_Z);
  printf("Running daxpy computation with alpha %.8g for point %d...\n", 
          alpha, point);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc_z[*pir] = alpha * acc_x[*pir] + acc_y[*pir];
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(double));
  const double alpha = *((const double*)task->args);

  const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);
  const FieldAccessor<READ_ONLY,double,1> acc_y(regions[0], FID_Y);
  const FieldAccessor<READ_ONLY,double,1> acc_z(regions[1], FID_Z);

  printf("Checking results...");
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  bool all_passed = true;
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
  {
    double expected = alpha * acc_x[*pir] + acc_y[*pir];
    double received = acc_z[*pir];
    // Probably shouldn't check for floating point equivalence but
    // the order of operations are the same should they should
    // be bitwise equal.
    if (expected != received)
      all_passed = false;
  }
  if (all_passed)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_FIELD_TASK_ID, "init_field");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_field_task>(registrar, "init_field");
  }

  {
    TaskVariantRegistrar registrar(DAXPY_TASK_ID, "daxpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<daxpy_task>(registrar, "daxpy");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  // Here is where we register the callback function for 
  // creating custom mappers.
  Runtime::add_registration_callback(mapper_registration);

  return Runtime::start(argc, argv);
}

