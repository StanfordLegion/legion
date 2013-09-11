/* Copyright 2013 Stanford University
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
#include <algorithm>
#include <cmath>

#include "legion.h"

using namespace LegionRuntime::HighLevel;

using namespace LegionRuntime::Accessor;


#define TEST_STEALING

#define MAX_STEAL_COUNT 4

enum {
  TOP_LEVEL_TASK_ID,
  TASKID_MAIN,
  TASKID_INIT_VECTORS,
  TASKID_ADD_VECTORS,
};

#define DEFAULT_NUM_BLOCKS 64
#define BLOCK_SIZE 256

struct Entry {
  float v;
};

struct Block {
  float alpha;
  LogicalRegion r_x, r_y, r_z;
  ptr_t entry_x[BLOCK_SIZE], entry_y[BLOCK_SIZE], entry_z[BLOCK_SIZE];
  unsigned id;
};

// computes z = alpha * x + y
struct MainArgs {
  unsigned num_blocks;
  unsigned num_elems;
  float alpha;
  IndexSpace ispace;
  FieldSpace fspace;
  LogicalRegion r_x, r_y, r_z;
};

float get_rand_float() {
  return (((float)2*rand()-RAND_MAX)/((float)RAND_MAX));
}

void top_level_task(const void *args, size_t arglen,
		    const std::vector<RegionRequirement> &reqs,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {
  InputArgs *inputs = (InputArgs*)args;
  MainArgs main_args;
  main_args.num_blocks = DEFAULT_NUM_BLOCKS;
  for (int i = 1; i < inputs->argc; i++) {
    if (!strcmp(inputs->argv[i], "-blocks")) {
      main_args.num_blocks = atoi(inputs->argv[++i]);
      continue;
    }
  }

  printf("saxpy: num elems = %d\n", main_args.num_blocks * BLOCK_SIZE);
  main_args.num_elems = main_args.num_blocks * BLOCK_SIZE;
  main_args.ispace = runtime->create_index_space(ctx, main_args.num_elems);
  main_args.fspace = runtime->create_field_space(ctx);
  main_args.r_x = runtime->create_logical_region(ctx, main_args.ispace, main_args.fspace);
  main_args.r_y = runtime->create_logical_region(ctx, main_args.ispace, main_args.fspace);
  main_args.r_z = runtime->create_logical_region(ctx, main_args.ispace, main_args.fspace);

  std::vector<IndexSpaceRequirement> indexes;
  indexes.push_back(IndexSpaceRequirement(main_args.ispace, ALLOCABLE, main_args.ispace));

  std::vector<FieldSpaceRequirement> fields;
  fields.push_back(FieldSpaceRequirement(main_args.fspace, ALLOCABLE));

  std::set<FieldID> priveledge_fields;
  std::vector<FieldID> instance_fields;
  // Defer actual field allocation until main_task.

  std::vector<RegionRequirement> main_regions;
  main_regions.push_back(RegionRequirement(main_args.r_x, priveledge_fields, instance_fields,
                                           READ_WRITE, EXCLUSIVE, main_args.r_x));
  main_regions.push_back(RegionRequirement(main_args.r_y, priveledge_fields, instance_fields,
                                           READ_WRITE, EXCLUSIVE, main_args.r_y));
  main_regions.push_back(RegionRequirement(main_args.r_z, priveledge_fields, instance_fields,
                                           READ_WRITE, EXCLUSIVE, main_args.r_z));

  Future f = runtime->execute_task(ctx, TASKID_MAIN, indexes, fields, main_regions,
				   TaskArgument(&main_args, sizeof(MainArgs)));
  f.get_void_result();

  // Destroy our logical regions clean up the region trees
  runtime->destroy_logical_region(ctx, main_args.r_x);
  runtime->destroy_logical_region(ctx, main_args.r_y);
  runtime->destroy_logical_region(ctx, main_args.r_z);
  runtime->destroy_index_space(ctx, main_args.ispace);
  runtime->destroy_field_space(ctx, main_args.fspace);
}

void main_task(const void *args, size_t arglen,
               const std::vector<RegionRequirement> &reqs,
               const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime) {
  MainArgs *main_args = (MainArgs *)args;
  PhysicalRegion r_x = regions[0];
  PhysicalRegion r_y = regions[1];
  PhysicalRegion r_z = regions[2];

  main_args->alpha = get_rand_float();
  printf("alpha: %f\n", main_args->alpha);

  // Set up index and field spaces
  IndexAllocator alloc = runtime->create_index_allocator(ctx, main_args->ispace);
  FieldAllocator field_alloc = runtime->create_field_allocator(ctx, main_args->fspace);
  FieldID field_id = field_alloc.allocate_field(sizeof(float));

  // Allocate space in the regions
  std::vector<Block> blocks(main_args->num_blocks);
  ptr_t initial_index = alloc.alloc(main_args->num_elems);
  ptr_t next_index = initial_index;
  for (unsigned i = 0; i < main_args->num_blocks; i++) {
    blocks[i].alpha = main_args->alpha;
    blocks[i].id = i;
    for (unsigned j = 0; j < BLOCK_SIZE; j++) {
      blocks[i].entry_x[j] = next_index;
      blocks[i].entry_y[j] = next_index;
      blocks[i].entry_z[j] = next_index;
      next_index++;
    }
  }
  printf("Done\n");

  // Partition the regions
  printf("Paritioning...");
  IndexSpace colors = runtime->create_index_space(ctx, main_args->num_blocks);
  runtime->create_index_allocator(ctx, colors).alloc(main_args->num_blocks);
  Coloring coloring;
  for (unsigned i = 0; i < main_args->num_blocks; i++) {
    coloring[i].ranges.insert(std::pair<ptr_t, ptr_t>(BLOCK_SIZE*i, BLOCK_SIZE*(i + 1)-1));
  }
  IndexPartition partition = runtime->create_index_partition(ctx, main_args->ispace, coloring, true/*disjoint*/);
  LogicalPartition p_x = runtime->get_logical_partition(ctx, main_args->r_x, partition);
  LogicalPartition p_y = runtime->get_logical_partition(ctx, main_args->r_y, partition);
  LogicalPartition p_z = runtime->get_logical_partition(ctx, main_args->r_z, partition);
  for (unsigned i = 0; i < main_args->num_blocks; i++) {
    blocks[i].r_x = runtime->get_logical_subregion_by_color(ctx, p_x, i);
    blocks[i].r_y = runtime->get_logical_subregion_by_color(ctx, p_y, i);
    blocks[i].r_z = runtime->get_logical_subregion_by_color(ctx, p_z, i);
  }
  printf("Done\n");

  // Unmap all regions
  runtime->unmap_region(ctx, r_x);
  runtime->unmap_region(ctx, r_y);
  runtime->unmap_region(ctx, r_z);

  // Argument map
  ArgumentMap arg_map = runtime->create_argument_map(ctx);
  for (unsigned i = 0; i < main_args->num_blocks; i++) {
    unsigned point[1] = {i};
    arg_map.set_point_arg<unsigned, 1>(point, TaskArgument(&(blocks[i]), sizeof(Block)));
  }

  // No further allocation of indexes or fields will be performed
  std::vector<IndexSpaceRequirement> index_reqs;
  index_reqs.push_back(IndexSpaceRequirement(main_args->ispace, NO_MEMORY, main_args->ispace));
  std::vector<FieldSpaceRequirement> field_reqs;
  field_reqs.push_back(FieldSpaceRequirement(main_args->fspace, NO_MEMORY));

  // Need access to fields created above
  std::set<FieldID> priveledge_fields;
  priveledge_fields.insert(field_id);
  std::vector<FieldID> instance_fields;
  instance_fields.push_back(field_id);

  // Empty global argument
  TaskArgument global(NULL, 0);

  // Regions for init task
  std::vector<RegionRequirement> init_regions;
  init_regions.push_back(RegionRequirement(p_x, 0, priveledge_fields, instance_fields, WRITE_ONLY, EXCLUSIVE, main_args->r_x));
  init_regions.push_back(RegionRequirement(p_y, 0, priveledge_fields, instance_fields, WRITE_ONLY, EXCLUSIVE, main_args->r_y));

  // Launch init task
  FutureMap init_f =
    runtime->execute_index_space(ctx, TASKID_INIT_VECTORS, colors,
                                 index_reqs, field_reqs, init_regions, global, arg_map, Predicate::TRUE_PRED, false);
  init_f.wait_all_results();

  printf("STARTING MAIN SIMULATION LOOP\n");
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);

  // Regions for add task
  std::vector<RegionRequirement> add_regions;
  add_regions.push_back(RegionRequirement(p_x, 0, priveledge_fields, instance_fields, READ_ONLY, EXCLUSIVE, main_args->r_x));
  add_regions.push_back(RegionRequirement(p_y, 0, priveledge_fields, instance_fields, READ_ONLY, EXCLUSIVE, main_args->r_y));
  add_regions.push_back(RegionRequirement(p_z, 0, priveledge_fields, instance_fields, WRITE_ONLY, EXCLUSIVE, main_args->r_z));

  // Launch add task
  FutureMap add_f =
    runtime->execute_index_space(ctx, TASKID_ADD_VECTORS, colors,
                                 index_reqs, field_reqs, add_regions, global, arg_map, Predicate::TRUE_PRED, false);
  add_f.wait_all_results();

  // Print results
  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
                     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  LegionRuntime::DetailedTimer::report_timers();

  // Validate the results
  {
    PhysicalRegion r_x =
      runtime->map_region(ctx, RegionRequirement(main_args->r_x, priveledge_fields, instance_fields, READ_ONLY, EXCLUSIVE, main_args->r_x));
    PhysicalRegion r_y =
      runtime->map_region(ctx, RegionRequirement(main_args->r_y, priveledge_fields, instance_fields, READ_ONLY, EXCLUSIVE, main_args->r_y));
    PhysicalRegion r_z = 
      runtime->map_region(ctx, RegionRequirement(main_args->r_z, priveledge_fields, instance_fields, READ_ONLY, EXCLUSIVE, main_args->r_z));
    r_x.wait_until_valid();
    r_y.wait_until_valid();
    r_z.wait_until_valid();

    RegionAccessor<AccessorType::Generic, Entry> a_x = r_x.get_accessor().typeify<Entry>();
    RegionAccessor<AccessorType::Generic, Entry> a_y = r_y.get_accessor().typeify<Entry>();
    RegionAccessor<AccessorType::Generic, Entry> a_z = r_z.get_accessor().typeify<Entry>();

#if 0
    printf("z values: ");
    for (unsigned i = 0; i < main_args->num_blocks; i++)
    {
      for (unsigned j = 0; j < BLOCK_SIZE; j++)
      {
        unsigned entry_z = blocks[i].entry_z[j];
        Entry z_val = a_z.read(ptr_t<Entry>(entry_z));
        printf("%f ",z_val.v);
      }
    }
    printf("\n");
#endif

    // Print the first four numbers
    int count = 0;
    bool success = true;
    for (unsigned i = 0; i < main_args->num_blocks; i++) {
      for (unsigned j = 0; j < BLOCK_SIZE; j++) {
        ptr_t entry_x = blocks[i].entry_x[j];
        ptr_t entry_y = blocks[i].entry_y[j];
        ptr_t entry_z = blocks[i].entry_z[j];

        Entry x_val = a_x.read(entry_x);
        Entry y_val = a_y.read(entry_y);
        Entry z_val = a_z.read(entry_z);
        float compute = main_args->alpha * x_val.v + y_val.v;
        //printf("%f * %f + %f should equal %f\n",main_args->alpha, x_val.v, y_val.v, z_val.v);
        if (z_val.v != compute)
        {
          printf("Failure at %d of block %d.  Expected %f but received %f\n",
              j, i, compute, z_val.v);
          success = false;
          break;
        }
        else if (count < 4) // Print the first four elements to make sure they aren't all zero
        {
          printf("%f ",z_val.v);
          count++;
          if (count == 4)
            printf("\n");
        }
      }
    }
    if (success)
      printf("SUCCESS!\n");
    else
      printf("FAILURE!\n");

    // Unmap the regions now that we're done with them
    runtime->unmap_region(ctx, r_x);
    runtime->unmap_region(ctx, r_y);
    runtime->unmap_region(ctx, r_z);
    runtime->destroy_index_space(ctx, colors); 
  }
}

void init_vectors_task(const void *global_args, size_t global_arglen,
                       const void *local_args, size_t local_arglen,
                       const DomainPoint &point,
                       const std::vector<RegionRequirement> &reqs,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, HighLevelRuntime *runtime) {
  PhysicalRegion r_x = regions[0];
  PhysicalRegion r_y = regions[1];

  RegionAccessor<AccessorType::Generic, Entry> a_x = r_x.get_accessor().typeify<Entry>();
  RegionAccessor<AccessorType::Generic, Entry> a_y = r_y.get_accessor().typeify<Entry>();

  assert(local_args != NULL);
  Block *block = (Block *)local_args;
  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    Entry entry_x;
    entry_x.v = get_rand_float();
    a_x.write((block->entry_x[i]), entry_x);

    Entry entry_y;
    entry_y.v = get_rand_float();
    a_y.write((block->entry_y[i]), entry_y);
  }
}

void add_vectors_task(const void *global_args, size_t global_arglen,
                      const void *local_args, size_t local_arglen,
                      const DomainPoint &point,
                      const std::vector<RegionRequirement> &reqs,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, HighLevelRuntime *runtime) {
  Block *block = (Block *)local_args;
  PhysicalRegion r_x = regions[0];
  PhysicalRegion r_y = regions[1];
  PhysicalRegion r_z = regions[2];

  RegionAccessor<AccessorType::Generic, Entry> a_x = r_x.get_accessor().typeify<Entry>();
  RegionAccessor<AccessorType::Generic, Entry> a_y = r_y.get_accessor().typeify<Entry>();
  RegionAccessor<AccessorType::Generic, Entry> a_z = r_z.get_accessor().typeify<Entry>();

  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    float x = a_x.read(block->entry_x[i]).v;
    float y = a_y.read(block->entry_y[i]).v;
    
    Entry entry_z;
    entry_z.v = block->alpha * x + y;
    a_z.write((block->entry_z[i]), entry_z);
  }
}

void add_vectors_task_aos(const void *global_args, size_t global_arglen,
			  const void *local_args, size_t local_arglen,
			  const unsigned point[1],
			  const std::vector<RegionRequirement> &reqs,
			  const std::vector<PhysicalRegion> &regions,
			  Context ctx, HighLevelRuntime *runtime) {
  Block *block = (Block *)local_args;
  PhysicalRegion r_x = regions[0];
  PhysicalRegion r_y = regions[1];
  PhysicalRegion r_z = regions[2];

  // RegionAccessor<AccessorType::AOS<sizeof(Entry)>, float> 
  //   a_x = r_x.get_accessor().typeify<Entry>().get_field_accessor(&Entry::v).get_aos_accessor();
  RegionAccessor<AccessorType::AOS<sizeof(Entry)>, float> 
    a_x = r_x.get_accessor().typeify<Entry>().get_field_accessor(&Entry::v).convert<AccessorType::AOS<sizeof(Entry)> >();
  RegionAccessor<AccessorType::AOS<sizeof(Entry)>, float>
    a_y = r_y.get_accessor().typeify<Entry>().get_field_accessor(&Entry::v).convert<AccessorType::AOS<sizeof(Entry)> >();
  RegionAccessor<AccessorType::AOS<sizeof(Entry)>, float>
    a_z = r_z.get_accessor().typeify<Entry>().get_field_accessor(&Entry::v).convert<AccessorType::AOS<sizeof(Entry)> >();

  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    float x = a_x.read(block->entry_x[i]);
    float y = a_y.read(block->entry_y[i]);
    
    float z = block->alpha * x + y;
    a_z.write((block->entry_z[i]), z);
  }
}

struct Add {
  typedef float LHS;
  typedef float RHS;

  template <bool EXCL> static void apply(float& lhs, float rhs) { lhs += rhs; }
  template <bool EXCL> static void fold(float& rhs1, float rhs2) { rhs1 += rhs2; }
  static const float identity = 0.0f;
};

template <typename AT>
void add_vectors_task_gen(const void *global_args, size_t global_arglen,
			  const void *local_args, size_t local_arglen,
			  const unsigned point[1],
			  const std::vector<RegionRequirement> &reqs,
			  const std::vector<PhysicalRegion> &regions,
			  Context ctx, HighLevelRuntime *runtime) {
  Block *block = (Block *)local_args;
  PhysicalRegion r_x = regions[0];
  PhysicalRegion r_y = regions[1];
  PhysicalRegion r_z = regions[2];

  RegionAccessor<AT, float> 
    a_x = r_x.get_accessor().typeify<Entry>().get_field_accessor(&Entry::v).convert<AT>();
  RegionAccessor<AT, float>
    a_y = r_y.get_accessor().typeify<Entry>().get_field_accessor(&Entry::v).convert<AT>();
  RegionAccessor<AT, float>
    a_z = r_z.get_accessor().typeify<Entry>().get_field_accessor(&Entry::v).convert<AT>();

  RegionAccessor<AccessorType::ReductionFold<Add>, float> rf = r_x.get_accessor().typeify<Entry>().get_field_accessor(&Entry::v).convert<AccessorType::ReductionFold<Add> >();

  //RegionAccessor<AccessorType::ReductionList<Add>, float> rl = r_x.get_accessor().typeify<Entry>().get_field_accessor(&Entry::v).convert<AccessorType::ReductionList<Add> >();

  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    float x = a_x.read(block->entry_x[i]);
    float y = a_y.read(block->entry_y[i]);
    
    float z = block->alpha * x + y;
    a_z.write((block->entry_z[i]), z);
    //a_z.template reduce<Add>(block->entry_z[i], z);
    rf.reduce(block->entry_z[i], z);
    //rl.reduce(block->entry_z[i], z);
  }
}


#if 0
static bool sort_by_proc_id(const std::pair<Processor, Memory> &a,
                            const std::pair<Processor, Memory> &b) {
  return a.first.id < b.first.id;
}

template<typename T>
T safe_prioritized_pick(const std::vector<T> &vec, T choice1, T choice2) {
  for (unsigned i = 0; i < vec.size(); i++)
    if (vec[i] == choice1)
      return choice1;
  for (unsigned i = 0; i < vec.size(); i++)
    if (vec[i] == choice2)
      return choice2;
  assert(false);
  T garbage = { 0 };
  return garbage;
}

class TestMapper : public Mapper {
public:
  TestMapper(Machine *machine, HighLevelRuntime *runtime, Processor local)
    : Mapper(machine, runtime, local) 
  { 
    const std::set<Memory> &visible = machine->get_visible_memories(local);  
    if (local.id == 1)
    {
      for (std::set<Memory>::const_iterator it = visible.begin();
            it != visible.end(); it++)
      {
        printf("Mapper has memory %x\n",it->id);
      }
    }
    std::set<Memory>::const_iterator it = visible.begin();
    for (unsigned idx = 0; idx < 4; idx++)
    {
      ordered_mems.push_back(*it);
      it++;
    }
    last_memory = *it;
  }
public:
  virtual bool map_task_region(const Task *task, Processor target, MappingTagID tag, bool inline_mapping, bool pre_mapping,
                                const RegionRequirement &req, unsigned index,
                                const std::map<Memory,bool> &current_instances, std::vector<Memory> &target_ranking,
                                std::set<FieldID> &additional_fields, bool &enable_WAR_optimization)
  {
    enable_WAR_optimization = false;
#if 0
    printf("Valid instances: ");
    for (std::map<Memory,bool>::const_iterator it = current_instances.begin();
          it != current_instances.end(); it++)
    {
      printf("%d ", it->first.id);
    }
    printf("\n");
#endif
    switch (task->task_id)
    {
      case TOP_LEVEL_TASK_ID:
        assert(false);
        break;
      case TASKID_MAIN:
        assert(inline_mapping);
        target_ranking.push_back(last_memory);
        break;
      case TASKID_INIT_VECTORS:
        {
        assert(task->is_index_space);
        assert(task->index_point != NULL);
        //unsigned point = *((unsigned*)task->index_point);
        Memory target = {((local_proc.id) % 4) + 1};
        //printf("Mapping logical region (%d,%x) of point %d to memory %x for init vectors index %d\n", req.region.get_tree_id(), req.region.get_index_space().id, point, target.id, index);
        target_ranking.push_back(target);
        break;
        }
      case TASKID_ADD_VECTORS:
        {
        assert(task->is_index_space);
        assert(task->index_point != NULL);
        //unsigned point2 = *((unsigned*)task->index_point);
        Memory target = {local_proc.id};
        //printf("Mapping logical region (%d,%x) of point %d to memory %x for add vectors index %d\n",req.region.get_tree_id(), req.region.get_index_space().id, point2, target.id, index);
        target_ranking.push_back(target);
        break;
        }
      default:
        assert(false);
    }
    return true;
  }

  virtual void notify_failed_mapping(const Task *task, const RegionRequirement &req, unsigned index, bool inline_mapping)
  {
    assert(false);
  }

#if 0
  virtual void rank_copy_targets(const Task *task, MappingTagID tag, bool inline_mapping,
                                  const RegionRequirement &req, unsigned index,
                                  const std::set<Memory> &current_instances,
                                  std::set<Memory> &to_reuse,
                                  std::vector<Memory> &to_create, bool &create_one)
  {

  }
#endif
private:
  std::vector<Memory> ordered_mems;
  Memory last_memory;
};
#endif

void create_mappers(Machine *machine, HighLevelRuntime *runtime,
                    const std::set<Processor> &local_procs) {
  //runtime->replace_default_mapper(new TestMapper(machine, runtime, local));
}

int main(int argc, char **argv) {
  srand(time(NULL));

  HighLevelRuntime::set_registration_callback(create_mappers);
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_single_task<top_level_task>(TOP_LEVEL_TASK_ID, Processor::LOC_PROC, false, "top_level_task");
  HighLevelRuntime::register_single_task<main_task>(TASKID_MAIN, Processor::LOC_PROC, false, "main_task");
  HighLevelRuntime::register_index_task<init_vectors_task>(TASKID_INIT_VECTORS, Processor::LOC_PROC, true, "init_vectors");
  HighLevelRuntime::register_index_task<add_vectors_task>(TASKID_ADD_VECTORS, Processor::LOC_PROC, true, "add_vectors");
  //HighLevelRuntime::register_index_task<unsigned,1,add_vectors_task_gen<AccessorType::AOS<sizeof(float)> > >(TASKID_ADD_VECTORS, Processor::LOC_PROC, true, "add_vectors");
  //HighLevelRuntime::register_index_task<unsigned,1,add_vectors_task_aos>(TASKID_ADD_VECTORS, Processor::LOC_PROC, true, "add_vectors");

  return HighLevelRuntime::start(argc, argv);
}
