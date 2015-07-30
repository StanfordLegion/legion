/* Copyright 2015 Stanford University
 * Copyright 2015 Los Alamos National Laboratory 
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
#include "legion_io.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

/*
 * In this example we illustrate how to do parallel copy of multiple
 * fields within the same logical region. 
 */


void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int num_elements = 1024;
  
  printf("Running legion IO tester for %d elements...\n", num_elements);

  /* give me 4 color points to address my decomposition */ 
  Point<2> color_lo; color_lo.x[0] = 0; color_lo.x[1] = 0;
  Point<2> color_hi; color_hi.x[0] = 1; color_hi.x[1] = 1;
  Rect<2> color_bounds(color_lo, color_hi); 
  Domain color_domain = Domain::from_rect<2>(color_bounds);

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
        runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double),FID_TEMP);
    allocator.allocate_field(sizeof(double),FID_SAL);
    allocator.allocate_field(sizeof(double),FID_KE);
    allocator.allocate_field(sizeof(double),FID_VOR);
  }

  FieldSpace persistent_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
        runtime->create_field_allocator(ctx, persistent_fs);
    allocator.allocate_field(sizeof(double),FID_TEMP);
  }
 
 
  Point<2> elem_rect_lo; elem_rect_lo.x[0] = 0; elem_rect_lo.x[1]=0;
  Point<2> elem_rect_hi; elem_rect_hi.x[0] = 31; elem_rect_hi.x[1]=31;
  
  Rect<2> elem_rect( elem_rect_lo, elem_rect_hi );// elem_rect(Point<2>a,Point<2>b);
  
  IndexSpace is = runtime->create_index_space(ctx, 
                          Domain::from_rect<2>(elem_rect));
  

  LogicalRegion ocean_lr = runtime->create_logical_region(ctx, is, fs);

  LogicalRegion persistent_lr = runtime->create_logical_region(ctx, is, persistent_fs);

  // make me an 16*16 patch for decomposition 
  Point<2> patch_color; patch_color.x[0] = 16; patch_color.x[1] = 16;
  Blockify<2> coloring(patch_color); // coloring(num_elements/num_subregions);

  IndexPartition ip  = runtime->create_index_partition(ctx, is, coloring);
  runtime->attach_name(ip, "ip");
  LogicalPartition ocean_lp = runtime->get_logical_partition(ctx, ocean_lr, ip);

  LogicalPartition persistent_lp = runtime->get_logical_partition(ctx, persistent_lr, ip);
  
  
  runtime->attach_name(ocean_lp, "ocean_lp");
  ArgumentMap arg_map;

  
  // First initialize fields with some data
  // IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_domain,
  //                             TaskArgument(NULL, 0), arg_map);

  // // Use data parallel and task parallel 
  // init_launcher.add_region_requirement(
  //     RegionRequirement(ocean_lp, 0/*projection ID*/,
  //                       WRITE_DISCARD, EXCLUSIVE, ocean_lr));
  // init_launcher.add_field(0, FID_TEMP);
  // runtime->execute_index_space(ctx, init_launcher);

  // init_launcher.region_requirements[0].privilege_fields.clear();
  // init_launcher.region_requirements[0].instance_fields.clear();
  // init_launcher.region_requirements[0].add_field(FID_SAL);
  // runtime->execute_index_space(ctx, init_launcher);

  // init_launcher.region_requirements[0].privilege_fields.clear();
  // init_launcher.region_requirements[0].instance_fields.clear();
  // init_launcher.region_requirements[0].add_field(FID_KE);
  // runtime->execute_index_space(ctx, init_launcher);

  // init_launcher.region_requirements[0].privilege_fields.clear();
  // init_launcher.region_requirements[0].instance_fields.clear();
  // init_launcher.region_requirements[0].add_field(FID_VOR);
  // runtime->execute_index_space(ctx, init_launcher);

  // 
  std::map<FieldID, const char*> field_map;
  field_map.insert(std::make_pair(FID_TEMP, "bam/baz"));
  PersistentRegion ocean_pr = PersistentRegion(runtime);
  ocean_pr.create_persistent_subregions(ctx, "ocean_pr.hdf5", persistent_lr, persistent_lp, color_domain, field_map);
  

  ocean_pr.write_persistent_subregions(ctx, ocean_lr, ocean_lp);
  
  runtime->destroy_logical_region(ctx, ocean_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
}

// The standard initialize field task from earlier examples
void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);
  
  Domain dom = runtime->get_index_space_domain(ctx, 
                                               task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();
  FieldID fid = *(task->regions[0].privilege_fields.begin());
  RegionAccessor<AccessorType::Generic, double> acc_temp = 
      regions[0].get_field_accessor(fid).typeify<double>();
  for (GenericPointInRectIterator<2> pir(rect); pir; pir++)
      {
          acc_temp.write(DomainPoint::from_point<2>(pir.p), drand48());
      }
}


void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 8);
  assert(task->arglen == sizeof(int));
  bool all_passed = true;
  int values_checked = 0;
  Domain dom = runtime->get_index_space_domain(ctx, 
                   task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();
  
  for(int i = 0; i < 4; i++) {
      RegionAccessor<AccessorType::Generic, double> acc_src = 
          regions[0].get_field_accessor(i).typeify<double>();
      RegionAccessor<AccessorType::Generic, double> acc_dst = 
          regions[0].get_field_accessor(i+4).typeify<double>();
      for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
          if(acc_src.read(DomainPoint::from_point<1>(pir.p)) !=
             acc_dst.read(DomainPoint::from_point<1>(pir.p)))
              all_passed = false;
          values_checked++;
      }
      
  }
  if (all_passed)
      printf("SUCCESS! checked %d values\n", values_checked);
  else
      printf("FAILURE!\n");
}
  
int main(int argc, char **argv)
{
  //volatile int x=0;
  // while(false || x == 0) {
    
  // }
  
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);
  HighLevelRuntime::register_legion_task<init_field_task>(INIT_FIELD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/);
  HighLevelRuntime::register_legion_task<check_task>(CHECK_TASK_ID,
         Processor::LOC_PROC, true/*single*/, true/*index*/);
  PersistentRegion_init();
  
  return HighLevelRuntime::start(argc, argv);
}
