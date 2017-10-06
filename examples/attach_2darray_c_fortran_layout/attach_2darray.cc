/* Copyright 2017 Los Alamos National Laboratory
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
using namespace Legion;

template<typename FT, int N, typename T = coord_t>
using AccessorRO = FieldAccessor<READ_ONLY,FT,N,T,
                                 Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t>
using AccessorWD = FieldAccessor<WRITE_DISCARD,FT,N,T,
                                 Realm::AffineAccessor<FT,N,T> >;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  READ_FIELD_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_A,
  FID_B,
};

typedef struct{
    double x;
    double y;
}xy_t;

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 10; 
  // See if we have any command line arguments to parse
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
    }
  }
  printf("Running attach array for %d 2D array dimension...\n", num_elements);

  // Create our logical regions using the same schema that
  // we used in the previous example.
  Point<2> lo(0, 0);
  Point<2> hi(num_elements-1, num_elements-1);
  const Rect<2> elem_rect(lo, hi);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect); 
  FieldSpace input_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(double),FID_X);
    allocator.allocate_field(sizeof(double),FID_Y);
    allocator.allocate_field(sizeof(double),FID_A);
    allocator.allocate_field(sizeof(double),FID_B);
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  
  /* init array */
  int i;
  double val = 0.0;
  xy_t *xy_ptr = (xy_t*)malloc(sizeof(xy_t)*(num_elements*num_elements));
  double *a_ptr = (double*)malloc(sizeof(double)*(num_elements*num_elements));
  double *b_ptr = (double*)malloc(sizeof(double)*(num_elements*num_elements));
  
  for (i = 0; i < num_elements*num_elements; i++) {
      xy_ptr[i].x = val;
      xy_ptr[i].y = val + 0.1;
      a_ptr[i] = val + 0.2;
      b_ptr[i] = val + 0.3;
      val += 1.0;
  }
  
  /* attach array */
  std::map<FieldID, size_t> offset_x;
  offset_x[FID_X] = 0;
  printf("Attach AOS array in fortran layout, fid %d, ptr %p\n", 
         FID_X, xy_ptr);  
  PhysicalRegion pr_x = runtime->attach_array_aos(ctx, input_lr, input_lr, 
                                                  xy_ptr, sizeof(xy_t), 
                                                  offset_x, 0);
  
  std::map<FieldID, size_t> offset_y;
  offset_y[FID_Y] = sizeof(double);
  printf("Attach AOS array in c layout, fid %d, ptr %p\n", 
         FID_Y, ((unsigned char*)(xy_ptr))+sizeof(double));  
  PhysicalRegion pr_y = runtime->attach_array_aos(ctx, input_lr, input_lr, 
                                                  xy_ptr, sizeof(xy_t), 
                                                  offset_y, 1);
  
  std::map<FieldID,void*> field_pointer_map_a;
  field_pointer_map_a[FID_A] = a_ptr;
  printf("Attach SOA array in fortran layout, fid %d, ptr %p\n", 
         FID_A, a_ptr);  
  PhysicalRegion pr_a = runtime->attach_array_soa(ctx, input_lr, input_lr, 
                                                  field_pointer_map_a, 0); 
  
  std::map<FieldID,void*> field_pointer_map_b;
  field_pointer_map_b[FID_B] = b_ptr;
  printf("Attach SOA array in c layout, fid %d, ptr %p\n", FID_B, b_ptr);  
  PhysicalRegion pr_b = runtime->attach_array_soa(ctx, input_lr, input_lr, 
                                                  field_pointer_map_b, 1); 

  TaskLauncher read_launcher(READ_FIELD_TASK_ID, 
                             TaskArgument(&num_elements, 
                             sizeof(num_elements)));
  read_launcher.add_region_requirement(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  read_launcher.add_field(0/*idx*/, FID_X);
  Future fx = runtime->execute_task(ctx, read_launcher);
  
  read_launcher.region_requirements[0].privilege_fields.clear();
  read_launcher.region_requirements[0].instance_fields.clear();
  read_launcher.add_field(0/*idx*/, FID_Y);
  Future fy = runtime->execute_task(ctx, read_launcher);
  
  read_launcher.region_requirements[0].privilege_fields.clear();
  read_launcher.region_requirements[0].instance_fields.clear();
  read_launcher.add_field(0/*idx*/, FID_A);
  Future fa = runtime->execute_task(ctx, read_launcher);

  read_launcher.region_requirements[0].privilege_fields.clear();
  read_launcher.region_requirements[0].instance_fields.clear();
  read_launcher.add_field(0/*idx*/, FID_B);
  Future fb = runtime->execute_task(ctx, read_launcher);
  
  fx.wait();
  fy.wait();
  fa.wait();
  fb.wait();

  runtime->detach_array(ctx, pr_x);
  runtime->detach_array(ctx, pr_y);
  runtime->detach_array(ctx, pr_a);
  runtime->detach_array(ctx, pr_b);
  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_index_space(ctx, is);
  free(xy_ptr);
  free(a_ptr);
  free(b_ptr);
}

// Note that tasks get a physical region for every region requirement
// that they requested when they were launched in the vector of 'regions'.
// In some cases the mapper may have chosen not to map the logical region
// which means that the task has the necessary privileges to access the
// region but not a physical instance to access.
void read_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  // Check that the inputs look right since we have no
  // static checking to help us out.
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int num_elements = *((const int*)task->args);
  int i = 0;
  
  const AccessorRO<double,2> acc(regions[0], fid);

  Rect<2> rect = runtime->get_index_space_domain(ctx, 
                  task->regions[0].region.get_index_space());
  printf("READ field %d, addr %p\n", fid, acc.ptr(rect.lo));
  for (PointInRectIterator<2> pir(rect); pir(); pir++) {
    printf("%.1f\t", acc[*pir]);
    i ++;
    if (i == num_elements) {
      printf("\n");
      i = 0;
    }
  }
  printf("\n");
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
    TaskVariantRegistrar registrar(READ_FIELD_TASK_ID, "read_field");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<read_field_task>(registrar, "read_field");
  }

  return Runtime::start(argc, argv);
}
