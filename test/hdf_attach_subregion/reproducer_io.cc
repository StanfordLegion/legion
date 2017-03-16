/* Copyright 2015 Stanford UniversiAty
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
#include "hdf5.h"

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  WRITE_VALUES_TASK_ID,
};

enum FieldIDs {
  FID_TEMP,
};


using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

void split_path_file(char** p, char** f, const char *pf);


struct Piece {
    LogicalRegion parent_lr;
    LogicalRegion child_lr;
    char shard_name[40];
}; 

void write_values_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, HighLevelRuntime *runtime) {
    Piece piece = *((Piece*) task->args);
    assert(regions[0].get_logical_region() == piece.child_lr);
    std::map<FieldID, const char*> field_map;
    field_map.insert(std::make_pair(FID_TEMP, "bam/baz"));
    runtime->unmap_region(ctx, regions[0]); 
    PhysicalRegion pr = runtime->attach_hdf5(ctx, piece.shard_name,
                                             piece.child_lr, piece.child_lr,
                                             field_map, LEGION_FILE_READ_WRITE);
    runtime->remap_region(ctx, pr);
    pr.wait_until_valid();
    RegionAccessor<AccessorType::Generic, double> acc_temp =
      pr.get_field_accessor(FID_TEMP).typeify<double>();
    Domain dom = runtime->get_index_space_domain(ctx,
                             piece.child_lr.get_index_space());
    
    Rect<2> rect = dom.get_rect<2>();
    for (GenericPointInRectIterator<2> pt(rect); pt; pt++)
    {
      std::cout << "write!" << std::endl;
      acc_temp.write(DomainPoint::from_point<2>(pt.p), drand48());
    }
    
    runtime->detach_hdf5(ctx, pr); 
    std::cout << "write_values_task complete" << std::endl; 
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  Point<2> color_lo; color_lo.x[0] = 0; color_lo.x[1] = 0;
  Point<2> color_hi; color_hi.x[0] = 3; color_hi.x[1] = 3;
  Rect<2> color_bounds(color_lo, color_hi); 
  Domain color_domain = Domain::from_rect<2>(color_bounds);

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
    
  LogicalRegion persistent_lr = runtime->create_logical_region(ctx, is, persistent_fs);
  
  Point<2> patch_color; patch_color.x[0] = 8; patch_color.x[1] = 8;
  Blockify<2> coloring(patch_color);
  
  IndexPartition ip  = runtime->create_index_partition(ctx, is, coloring);
  
  LogicalPartition persistent_lp = runtime->get_logical_partition(ctx, persistent_lr, ip);
  
  hid_t group_id, dataset_id, dataspace_id, dtype_id, shard_file_id;
  herr_t status;
  
  char name[40], shard_name[40];

  sprintf(name, "test_pr.hdf5");
  //file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  dtype_id  = H5Tcopy (H5T_NATIVE_DOUBLE);

      
   int i = 0;
   for(Legion::Domain::DomainPointIterator itr(color_domain); itr && i<1; itr++) {
       Piece piece; 
       DomainPoint dp = itr.p;
       piece.parent_lr = persistent_lr; 
       piece.child_lr = runtime->get_logical_subregion_by_color(ctx, persistent_lp, dp);
       IndexSpace child_is = runtime->get_index_subspace(ctx, persistent_lp.get_index_partition(), dp); 
       FieldSpace child_fs = piece.child_lr.get_field_space();
       Domain d = runtime->get_index_space_domain(ctx, child_is);
       int dim = d.get_dim();
       std::cout << "Found a logical region:  Dimension " << dim <<  std::endl;
       int x_min = 0, y_min = 0,
           x_max = 0, y_max = 0;
       
       sprintf(shard_name, "%d-%s", i, name);
       sprintf(piece.shard_name, "%s", shard_name);
       
       shard_file_id = H5Fcreate(shard_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
       int *shard_dims;
       x_min = d.get_rect<2>().lo.x[0];
       y_min = d.get_rect<2>().lo.x[1];
       x_max = d.get_rect<2>().hi.x[0];
       y_max = d.get_rect<2>().hi.x[1];
       std::cout << "domain rect is: [[" << x_min << "," << y_min
                 << "],[" << x_max  << "," << y_max << "]]" << std::endl; 
       
       hsize_t dims[2];
       dims[0] = x_max-x_min+1;
       dims[1] = y_max-y_min+1;
       dataspace_id = H5Screate_simple(2, dims, NULL);
       shard_dims = (int*) malloc(4*sizeof(int)); 
       shard_dims[0] = x_min;
       shard_dims[1] = y_min;
       shard_dims[2] = x_max;
       shard_dims[3] = y_max;
       
       TaskLauncher write_launcher(WRITE_VALUES_TASK_ID,
                                   TaskArgument(&piece, sizeof(Piece))); 
       write_launcher.add_region_requirement(
                      RegionRequirement(piece.child_lr,
                                        READ_WRITE, EXCLUSIVE, persistent_lr));

       

       typedef std::map<FieldID, const char*>::iterator it_type;
       std::map<FieldID, const char*> field_map;
       field_map.insert(std::make_pair(FID_TEMP, "bam/baz"));
       
       for(it_type iterator = field_map.begin(); iterator != field_map.end(); iterator++) {
            FieldID fid = iterator->first;
            write_launcher.region_requirements[0].add_field(fid, false);
            char* ds;
            char* gp;
            split_path_file(&gp, &ds, iterator->second);
            size_t field_size = runtime->get_field_size(ctx, child_fs, fid);
            status = H5Tset_size(dtype_id, field_size);
            if(H5Lexists(shard_file_id, gp, H5P_DEFAULT)) { 
                group_id = H5Gopen2(shard_file_id, gp, H5P_DEFAULT);
            } else { 
                group_id = H5Gcreate2(shard_file_id, gp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            } 
            if(H5Lexists(group_id, ds, H5P_DEFAULT)) { 
                dataset_id = H5Dopen2(group_id, ds, H5P_DEFAULT); 
            } else { 
                dataset_id = H5Dcreate2(group_id, ds, H5T_NATIVE_DOUBLE, dataspace_id,
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            }
            
       }
       
       H5Dclose(dataset_id);
       H5Gclose(group_id);
       H5Fclose(shard_file_id);
       
       runtime->execute_task(ctx, write_launcher);

#if 0 
       //PhysicalRegion pr = runtime->attach_hdf5(ctx, shard_name,
       //                                         piece.child_lr, persistent_lr,
       //                                         field_map, LEGION_FILE_READ_WRITE); 
#endif
       i++;
   }
   std::cout << "Leaving top level task now" << std::endl;
   runtime->destroy_logical_region(ctx, persistent_lr);
   runtime->destroy_field_space(ctx, persistent_fs);
   runtime->destroy_index_space(ctx, is);
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
    Processor::LOC_PROC, true/*single*/, false/*index*/);
  HighLevelRuntime::register_legion_task<write_values_task>(WRITE_VALUES_TASK_ID,
    Processor::LOC_PROC, true /*single*/, true /*index*/);
  //Runtime rt;

  //rt.init(&argc, &argv);
  //rt.register_task(TOP_LEVEL_TASK_ID, top_level_task);

  //rt.run(TOP_LEVEL_TASK_ID, Runtime::ONE_TASK_ONLY);

  //rt.shutdown();

  //return -1;
  return HighLevelRuntime::start(argc, argv);
    
  
}

void split_path_file(char** p, char** f, const char *pf) {
    char *slash = (char*)pf, *next;
    while ((next = strpbrk(slash + 1, "\\/"))) slash = next;
    if (pf != slash) slash++;
    *p = strndup(pf, slash - pf);
    *f = strdup(slash);
}
