/* Copyright 2017 Stanford University, NVIDIA Corporation
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


#undef SMALL_INDEX_LAUNCH//toggle this to demonstrate two different problems abh 6/10/17


#include "long_running.h"

enum {
  TOP_LEVEL_TASK_ID,
  GENERATE_IMAGE_DATA_TASK_ID,
  VERIFY_COMPOSITED_IMAGE_DATA_TASK_ID
};

enum FieldIDs {
  FID_FIELD_R = 0,
  FID_FIELD_G,
  FID_FIELD_B,
  FID_FIELD_A,
  FID_FIELD_Z,
  FID_FIELD_USERDATA,
};

static int taskCount = 0;
static int numPoints = 0;

using namespace LegionRuntime::Arrays;

namespace Legion {
  
  
  class ImageReduction {
  public:
    typedef float PixelField;
    static const int image_region_dimensions = 3;
    ImageReduction(Legion::Runtime* runtime, Legion::Context context){
      mRuntime = runtime;
      mContext = context;
      
      // logical region
      
      Point<3> origin = Point<3>::ZEROES();
      Point<3> regionSize;
#ifdef SMALL_INDEX_LAUNCH
      regionSize.x[0] = 32;//320;
      regionSize.x[1] = 24;//240;
#else
      regionSize.x[0] = 320;
      regionSize.x[1] = 240;
#endif
      regionSize.x[2] = 4;
      Rect<image_region_dimensions> imageBounds(origin, regionSize - Point<image_region_dimensions>::ONES());
      mDomain = Domain::from_rect<image_region_dimensions>(imageBounds);
      pixels = mRuntime->create_index_space(mContext, mDomain);
      mRuntime->attach_name(pixels, "image index space");
      fields = imageFields();
      region = mRuntime->create_logical_region(mContext, pixels, fields);
      mRuntime->attach_name(region, "image");
      
      // logical partition
      
      Point<3> fragmentSize;
      fragmentSize.x[0] = 16;
      fragmentSize.x[1] = 1;
      fragmentSize.x[2] = 1;
      Blockify<image_region_dimensions> coloring(fragmentSize);
      imageFragmentIndexPartition = mRuntime->create_index_partition(mContext, region.get_index_space(), coloring);
      mRuntime->attach_name(imageFragmentIndexPartition, "image fragment index");
      partition = mRuntime->get_logical_partition(mContext, region, imageFragmentIndexPartition);
      mRuntime->attach_name(partition, "image fragment partition");
      Point<3> numFragments;
      numFragments.x[0] = regionSize.x[0] / fragmentSize.x[0];
      numFragments.x[1] = regionSize.x[1] / fragmentSize.x[1];
      numFragments.x[2] = regionSize.x[2] / fragmentSize.x[2];
      Rect<image_region_dimensions> fragmentBounds(origin, numFragments - Point<image_region_dimensions>::ONES());
      fragmentDomain = Domain::from_rect<image_region_dimensions>(fragmentBounds);
      numPoints = fragmentDomain.get_volume();
      std::cout << "fragment domain " << fragmentDomain << " num points " << numPoints << std::endl;
      
      // index launch
      
      UsecTimer indexLaunch("time for index launch");
      indexLaunch.start();
      
      ArgumentMap argMap;
      IndexTaskLauncher treeCompositeLauncher(GENERATE_IMAGE_DATA_TASK_ID, fragmentDomain, TaskArgument(NULL, 0), argMap);
      RegionRequirement req(partition, 0, READ_WRITE, EXCLUSIVE, region);
      req.add_field(FID_FIELD_R);
      req.add_field(FID_FIELD_G);
      req.add_field(FID_FIELD_B);
      req.add_field(FID_FIELD_A);
      req.add_field(FID_FIELD_Z);
      req.add_field(FID_FIELD_USERDATA);
      treeCompositeLauncher.add_region_requirement(req);
      FutureMap futures = mRuntime->execute_index_space(mContext, treeCompositeLauncher);
      futures.wait_all_results();
      indexLaunch.stop();
      std::cout << indexLaunch.to_string() << std::endl;
    }
    
    virtual ~ImageReduction(){
      mRuntime->destroy_index_space(mContext, pixels);
      mRuntime->destroy_logical_region(mContext, region);
      mRuntime->destroy_index_partition(mContext, imageFragmentIndexPartition);
      mRuntime->destroy_logical_partition(mContext, partition);
      std::cout << "executed destructor" << std::endl;
    }
    
    FieldSpace imageFields() {
      FieldSpace fields = mRuntime->create_field_space(mContext);
      mRuntime->attach_name(fields, "pixel fields");
      {
        FieldAllocator allocator = mRuntime->create_field_allocator(mContext, fields);
        FieldID fidr = allocator.allocate_field(sizeof(PixelField), FID_FIELD_R);
        assert(fidr == FID_FIELD_R);
        FieldID fidg = allocator.allocate_field(sizeof(PixelField), FID_FIELD_G);
        assert(fidg == FID_FIELD_G);
        FieldID fidb = allocator.allocate_field(sizeof(PixelField), FID_FIELD_B);
        assert(fidb == FID_FIELD_B);
        FieldID fida = allocator.allocate_field(sizeof(PixelField), FID_FIELD_A);
        assert(fida == FID_FIELD_A);
        FieldID fidz = allocator.allocate_field(sizeof(PixelField), FID_FIELD_Z);
        assert(fidz == FID_FIELD_Z);
        FieldID fidUserdata = allocator.allocate_field(sizeof(PixelField), FID_FIELD_USERDATA);
        assert(fidUserdata == FID_FIELD_USERDATA);
      }
      return fields;
    }
    
    
  private:
    Domain mDomain;
    IndexSpace pixels;
    FieldSpace fields;
    LogicalRegion region;
    Legion::Runtime* mRuntime;
    Legion::Context mContext;
    IndexPartition imageFragmentIndexPartition;
    LogicalPartition partition;
    Domain fragmentDomain;
  };
  
  void generate_image_data_task(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, HighLevelRuntime *runtime) {
    PhysicalRegion region = regions[0];
    taskCount++;
    if(taskCount % 100 == 0) {
      std::cout << taskCount << " tasks out of " << numPoints << std::endl;
    }
  }
  
  
  void top_level_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, HighLevelRuntime *runtime) {
    
    for(int i = 0; i < 100000; ++i) {
      ImageReduction image_reduction(runtime, ctx);
    }
    
  }
}

int main(int argc, char *argv[]) {
  
  std::cout << " this program demonstrates memory leaks and an index launch that never completes" << std::endl;
  
  Legion::HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  
  Legion::HighLevelRuntime::register_legion_task<Legion::top_level_task>(TOP_LEVEL_TASK_ID,
                                                                         Legion::Processor::LOC_PROC, true/*single*/, false/*index*/,
                                                                         AUTO_GENERATE_ID, Legion::TaskConfigOptions(false/*leaf*/), "top_level_task");
  
  Legion::HighLevelRuntime::register_legion_task<Legion::generate_image_data_task>(GENERATE_IMAGE_DATA_TASK_ID,
                                                                                   Legion::Processor::LOC_PROC, false/*single*/, true/*index*/,
                                                                                   AUTO_GENERATE_ID, Legion::TaskConfigOptions(true/*leaf*/), "generate_image_data_task");
  
  return Legion::HighLevelRuntime::start(argc, argv);
}


