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
#include <unistd.h>
#include <math.h>
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

/*
 * In this example we illustrate how to do parallel I/O of subregions 
 *  in a larger logical region to individual HDF5 files 
 */


void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime)
{
  int num_elements = 1024;
  int sub_regions = 64;
  int option_char;
  volatile int debug_flag = 0;
  
  const InputArgs &command_args = HighLevelRuntime::get_input_args();
  while ((option_char = getopt(command_args.argc, command_args.argv, ":dn:s:")) != -1) {
    switch (option_char)
    {
    case 'd': debug_flag = 1; break;
    case 'n': num_elements = atoi (optarg); break;
    case 's': sub_regions = atoi (optarg); break;
    case '?': fprintf (stderr,
                       "usage: %s [dn<size>s<size>]\n", command_args.argv[0]);
    }
  }
  
  
  while(debug_flag == 1) {
    
  }

  int elem_rect_hi_val = sqrt(num_elements) - 1;
  int color_hi_val = sqrt(sub_regions)-1;
  int patch_val = sqrt(num_elements / sub_regions); 
  
  
  std::cout << "Running legion IO tester with "
            << num_elements << " elements and "
            << sub_regions << " subregions" << std::endl;

  /* give me color points to address my decomposition */ 
  Point<2> color_lo; color_lo.x[0] = 0; color_lo.x[1] = 0;
  Point<2> color_hi; color_hi.x[0] = color_hi.x[1] = color_hi_val;
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
  Point<2> elem_rect_hi; elem_rect_hi.x[0] = elem_rect_hi.x[1] = elem_rect_hi_val;
  Rect<2> elem_rect( elem_rect_lo, elem_rect_hi );
  
  IndexSpace is = runtime->create_index_space(ctx, 
    Domain::from_rect<2>(elem_rect));
  

  LogicalRegion ocean_lr = runtime->create_logical_region(ctx, is, fs);

  LogicalRegion persistent_lr = runtime->create_logical_region(ctx, is, persistent_fs);

  Point<2> patch_color; patch_color.x[0] = patch_color.x[1] = patch_val;
  Blockify<2> coloring(patch_color); 

  IndexPartition ip  = runtime->create_index_partition(ctx, is, coloring);
  runtime->attach_name(ip, "ip");
  LogicalPartition ocean_lp = runtime->get_logical_partition(ctx, ocean_lr, ip);

  LogicalPartition persistent_lp = runtime->get_logical_partition(ctx, persistent_lr, ip);
  
  
  runtime->attach_name(ocean_lp, "ocean_lp");
  
  //First initialize fields with some data
  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_domain,
                              TaskArgument(&patch_val, sizeof(patch_val)), ArgumentMap());

  // Use data parallel and task parallel 
  init_launcher.add_region_requirement(
    RegionRequirement(ocean_lp, 0/*projection ID*/,
                      WRITE_DISCARD, EXCLUSIVE, ocean_lr));
  init_launcher.add_field(0, FID_TEMP);
  runtime->execute_index_space(ctx, init_launcher);
  
  // init_launcher.region_requirements[0].privilege_fields.clear();
  // init_launcher.region_requirements[0].instance_fields.clear();
  // init_launcher.region_requirements[0].add_field(FID_SAL);
  // runtime->execute_index_space(ctx, init_launcher);

  std::map<FieldID, const char*> field_map;
  field_map.insert(std::make_pair(FID_TEMP, "bam/baz"));
  PersistentRegion ocean_pr = PersistentRegion(runtime);
  ocean_pr.create_persistent_subregions(ctx, "ocean_pr.hdf5", persistent_lr, persistent_lp, color_domain, field_map);
  

  ocean_pr.write_persistent_subregions(ctx, ocean_lr, ocean_lp);
  

  LogicalRegion ocean_check_lr = runtime->create_logical_region(ctx, is, fs);
  LogicalPartition ocean_check_lp = runtime->get_logical_partition(ctx, ocean_check_lr, ip);
  
  ocean_pr.read_persistent_subregions(ctx, ocean_check_lr, ocean_check_lp);

  IndexLauncher check_launcher(CHECK_TASK_ID, color_domain,
                               TaskArgument(NULL, 0), ArgumentMap());
  
  check_launcher.add_region_requirement(
    RegionRequirement(ocean_check_lp, 0/*projection ID*/,
                      READ_ONLY, EXCLUSIVE, ocean_check_lr));

  check_launcher.add_region_requirement(
    RegionRequirement(ocean_lp, 0/*projection ID*/,
                      READ_ONLY, EXCLUSIVE, ocean_lr));

  check_launcher.region_requirements[0].add_field(FID_TEMP);
  check_launcher.region_requirements[1].add_field(FID_TEMP);
  
  runtime->execute_index_space(ctx, check_launcher);


  
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
  int extent = *(const int*) task->args;
 
  std::cout << "init_field_task extent is: " << extent << " domain point is:[" << task->index_point[0] << "," <<
    task->index_point[1] << "]" << " linearization is: " << task->index_point[0]*extent+task->index_point[1]*extent <<  std::endl;
  
  Domain dom = runtime->get_index_space_domain(ctx, 
    task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();
  FieldID fid = *(task->regions[0].privilege_fields.begin());
  RegionAccessor<AccessorType::Generic, double> acc_temp = 
    regions[0].get_field_accessor(fid).typeify<double>();
  for (GenericPointInRectIterator<2> pir(rect); pir; pir++)
  {
    acc_temp.write(DomainPoint::from_point<2>(pir.p), task->index_point[0]*extent+task->index_point[1]*extent + drand48());
  }
}


void check_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, HighLevelRuntime *runtime)
{
  assert(task->regions.size() == 2);
  assert(task->regions[0].instance_fields.size() ==
         task->regions[1].instance_fields.size());
  bool all_passed = true;
  int values_checked = 0;
  Domain dom = runtime->get_index_space_domain(ctx, 
    task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();
  
  for(unsigned i = 0; i < task->regions[0].instance_fields.size(); i++) {
    RegionAccessor<AccessorType::Generic, double> acc_src = 
      regions[0].get_field_accessor(i).typeify<double>();
    RegionAccessor<AccessorType::Generic, double> acc_dst = 
      regions[1].get_field_accessor(i).typeify<double>();
    for (GenericPointInRectIterator<2> pir(rect); pir; pir++) {
      if(acc_src.read(DomainPoint::from_point<2>(pir.p)) !=
         acc_dst.read(DomainPoint::from_point<2>(pir.p))) 
        all_passed = false;
//      std::cout << "src is: " << acc_src.read(DomainPoint::from_point<2>(pir.p))
//                << " dst is: " << acc_dst.read(DomainPoint::from_point<2>(pir.p)) << std::endl;
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


