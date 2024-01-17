/* Copyright 2023 Stanford University
 * Copyright 2023 Los Alamos National Laboratory
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

// this test makes use of lots of deprecated Legion API calls - ignore for now
#define LEGION_DEPRECATED(x)

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"
#include "legion_io.h"
#include <unistd.h>
#include <math.h>
#include <cmath>

using namespace Legion;

/*
 * In this example we illustrate how to do parallel I/O of subregions 
 *  in a larger logical region to individual HDF5 files 
 */


void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, Runtime *runtime)
{
  uint64_t num_elements = 1024;
  int sub_regions = 64;
  int ndim = 2;
  volatile int debug_flag = 0;
  
  const InputArgs &command_args = Runtime::get_input_args();
  for (int i = 1; i < command_args.argc; i++)
  {
    if (!strcmp(command_args.argv[i],"-n"))
      num_elements = atoll(command_args.argv[++i]);
    if (!strcmp(command_args.argv[i],"-s"))
      sub_regions = atoi(command_args.argv[++i]);
    if (!strcmp(command_args.argv[i],"-d"))
      debug_flag = 1;
    if (!strcmp(command_args.argv[i], "-r"))
      ndim = atoi(command_args.argv[++i]);
  }

  
  while (debug_flag == 1) {}

  assert(ndim == 2 || ndim == 3);

  int elem_rect_hi_val;
  int color_hi_val;
  int patch_val;

  switch (ndim) {
    case 2:
      elem_rect_hi_val = sqrt(num_elements) - 1;
      color_hi_val = sqrt(sub_regions)-1;
      patch_val = sqrt(num_elements / sub_regions); 
      break;

    case 3:
      elem_rect_hi_val = std::ceil(std::pow(num_elements, 1/3.0)) - 1;
      color_hi_val = std::ceil(std::pow(sub_regions, 1/3.0)) - 1;
      patch_val = std::ceil(std::pow(num_elements / sub_regions, 1/3.0));
      std::cout << "num_elements: " << num_elements << " subregions: " << sub_regions 
		<< " patch_val: " << patch_val << " color_hi_val: " <<  color_hi_val 
		<< std::endl; 
      assert(num_elements == (std::pow(patch_val, 3) * std::pow(color_hi_val+1, 3)));
      assert(num_elements == std::pow(elem_rect_hi_val+1, 3));
      break;

    default:
      assert(0);
  }

  std::cout << "Running legion IO tester with "
            << num_elements << " elements and "
            << sub_regions << " subregions" << std::endl;

  /* give me color points to address my decomposition */ 
  Domain color_domain;
  switch (ndim) {
    case 2:
      {
        Point<2> color_lo; color_lo.x[0] = 0; color_lo.x[1] = 0;
        Point<2> color_hi; color_hi.x[0] = color_hi.x[1] = color_hi_val;
        Rect<2> color_bounds(color_lo, color_hi); 
        color_domain = Domain::from_rect<2>(color_bounds);
      }
      break;

    case 3:
      color_domain = Domain::from_rect<3>(Rect<3>(
            make_point(0, 0, 0),
            make_point(
              color_hi_val,
              color_hi_val,
              color_hi_val)));
      break;

    default:
      assert(0);
  }

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double),FID_TEMP);
    //    allocator.allocate_field(sizeof(double),FID_SAL);
    //    allocator.allocate_field(sizeof(double),FID_KE);
    //    allocator.allocate_field(sizeof(double),FID_VOR);
  }

  FieldSpace persistent_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, persistent_fs);
    allocator.allocate_field(sizeof(double),FID_TEMP);
  }
 
  Domain elem_domain;
  switch (ndim) {
    case 2:
      {
        Point<2> elem_rect_lo; elem_rect_lo.x[0] = 0; elem_rect_lo.x[1]=0;
        Point<2> elem_rect_hi; elem_rect_hi.x[0] = elem_rect_hi.x[1] = elem_rect_hi_val;
        Rect<2> elem_rect( elem_rect_lo, elem_rect_hi );
        elem_domain = Domain::from_rect<2>(elem_rect);
      }
      break;

    case 3:
      elem_domain = Domain::from_rect<3>(Rect<3>(
            make_point(0, 0, 0),
            make_point(
              elem_rect_hi_val,
              elem_rect_hi_val,
              elem_rect_hi_val)));
      break;

    default:
      assert(0);
  }
 
  
  IndexSpace is = runtime->create_index_space(ctx, elem_domain);

  LogicalRegion ocean_lr = runtime->create_logical_region(ctx, is, fs);
  LogicalRegion persistent_lr = runtime->create_logical_region(ctx, is, persistent_fs);

  IndexPartition ip;
  switch (ndim) {
    case 2:
      {
        Point<2> patch_color; patch_color.x[0] = patch_color.x[1] = patch_val;
        Blockify<2> coloring(patch_color); 
        ip  = runtime->create_index_partition(ctx, is, coloring);
      }
      break;

    case 3:
      {
        Blockify<3> coloring(make_point(patch_val, patch_val, patch_val));
        ip  = runtime->create_index_partition(ctx, is, coloring);
      }
      break;

    default:
      assert(0);
  }
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
  
  std::map<FieldID, std::string> field_string_map;
  field_string_map.insert(std::make_pair(FID_TEMP, "bam/baz"));
  
  PersistentRegion ocean_pr = PersistentRegion(runtime);
  ocean_pr.create_persistent_subregions(ctx, "ocean_pr.hdf5", persistent_lr,
      persistent_lp, color_domain, field_string_map);
  
  ocean_pr.write_persistent_subregions(ctx, ocean_lr, ocean_lp);

  LogicalRegion 
ocean_check_lr = runtime->create_logical_region(ctx, is, fs);
  LogicalPartition ocean_check_lp = runtime->get_logical_partition(ctx, ocean_check_lr, ip);

  ocean_pr.read_persistent_subregions(ctx, ocean_check_lr, ocean_check_lp);

#ifdef TESTERIO_CHECK
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

#endif
  
  runtime->destroy_logical_region(ctx, ocean_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
}

// The standard initialize field task from earlier examples
void init_field_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);
  int extent = *(const int*) task->args;
  
#ifdef IOTESTER_VERBOSE
  char hostname[128];
  gethostname(hostname, sizeof hostname);

  std::cout << hostname << " init_field_task extent is: " << extent << " domain point is:[" << task->index_point[0] << "," <<
    task->index_point[1] << "]" << " linearization is: " << task->index_point[0]*extent+task->index_point[1]*extent <<  std::endl;
#endif

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  Domain dom = runtime->get_index_space_domain(ctx, 
    task->regions[0].region.get_index_space());

  switch (dom.get_dim()) {
    case 2:
      {
        Rect<2> rect = dom;
        FieldAccessor<LEGION_WRITE_DISCARD,double,2> acc_temp(regions[0], fid);
        for (PointInRectIterator<2> pir(rect); pir(); pir++) {
          acc_temp.write(*pir,
              task->index_point[0]*extent +
              task->index_point[1]*extent + drand48());
        }
      }
      break;

    case 3:
      {
        Rect<3> rect = dom;
        FieldAccessor<LEGION_WRITE_DISCARD,double,3> acc_temp(regions[0], fid);
        for (PointInRectIterator<3> pir(rect); pir(); pir++) {
          acc_temp.write(*pir,
              task->index_point[0]*extent +
              task->index_point[1]*extent +
              task->index_point[2]*extent + drand48());
        }
      }
      break;

    default:
      assert(0);
  }
}


void check_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
#ifdef TESTERIO_TIMERS
  struct timespec ts;
  current_utc_time(&ts);   
  std::cout << "domain point: " << task->index_point
            << "; read ends & Check begins at:  seconds: " << ts.tv_sec
            << " nanos: " << ts.tv_nsec << std::endl; 
#endif
  assert(task->regions.size() == 2);
  assert(task->regions[0].instance_fields.size() ==
         task->regions[1].instance_fields.size());

  bool all_passed = true;
  int values_checked = 0;

  Domain dom = runtime->get_index_space_domain(ctx, 
    task->regions[0].region.get_index_space());

  switch (dom.get_dim()) {
    case 2:
      {
        Rect<2> rect = dom;
        for (unsigned i = 0; i < task->regions[0].instance_fields.size(); i++) {
          FieldAccessor<LEGION_READ_ONLY,double,2> acc_src(regions[0], i);
          FieldAccessor<LEGION_READ_ONLY,double,2> acc_dst(regions[1], i);
          for (PointInRectIterator<2> pir(rect); pir(); pir++) {
            if (acc_src.read(*pir) != acc_dst.read(*pir))
              all_passed = false;
            values_checked++;
          }
        }
      }
      break;

    case 3:
      {
        Rect<3> rect = dom;
        for (unsigned i = 0; i < task->regions[0].instance_fields.size(); i++) {
          FieldAccessor<LEGION_READ_ONLY,double,3> acc_src(regions[0], i);
          FieldAccessor<LEGION_READ_ONLY,double,3> acc_dst(regions[1], i);
          for (PointInRectIterator<3> pir(rect); pir(); pir++) {
            if (acc_src[*pir] != acc_dst[*pir])
              all_passed = false;
            values_checked++;
          }
        }
      }
      break;

    default:
      assert(0);
  }
  
  if (all_passed)
    printf("SUCCESS! checked %d values\n", values_checked);
  else
    printf("FAILURE!\n");
}
  
int main(int argc, char **argv)
{
#ifdef TESTERIO_TIMERS 
  char hostname[128];
  gethostname(hostname, sizeof hostname);
  std::cout << hostname << " in main prior to task registrion " << std::endl; 
#endif

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  }

  {
    TaskVariantRegistrar registrar(INIT_FIELD_TASK_ID, "init_field");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<init_field_task>(registrar, "init_field");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  PersistentRegion_init();
  return Runtime::start(argc, argv);
}


