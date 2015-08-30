/* Copyright 2015 Stanford University
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

#ifndef legionvector_hpp
#define legionvector_hpp

#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

#ifndef OFFSETS_ARE_DENSE
#define OFFSETS_ARE_DENSE
template<unsigned DIM, typename T>
static inline bool offsets_are_dense(const Rect<DIM> &bounds, const ByteOffset *offset)
{
  off_t exp_offset = sizeof(T);
  for (unsigned i = 0; i < DIM; i++) {
    bool found = false;
    for (unsigned j = 0; j < DIM; j++) {
      if (offset[j].offset == exp_offset) {
        found = true;
        exp_offset *= (bounds.hi[j] - bounds.lo[j] + 1);
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}
#endif

enum VTaskIDs{
	V_INIT_TASK_ID = 1,
};

enum VFieldIDs{
        FID_X = 0,
};

template<typename T>
void Initialize_Task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx,
                      HighLevelRuntime *runtime);

template<typename T>
class Array{

	public:
	int64_t size;
	int64_t nparts;
	FieldID fid;
	Domain color_domain;
        Rect<1> rect;
        IndexSpace is;
        IndexPartition ip;
        FieldSpace fs;
        LogicalRegion lr;
        LogicalPartition lp;

	Array(void){this-> fid = FID_X;};
	Array(int64_t size, int64_t nparts, Context ctx, HighLevelRuntime *runtime);
	void DestroyArray(Context ctx, HighLevelRuntime *runtime);
	void Initialize(Context ctx, HighLevelRuntime *runtime);
	void Initialize(T *input, Context ctx, HighLevelRuntime *runtime);
	void RandomInit(Context ctx, HighLevelRuntime *runtime);
	void PrintVals(Context ctx, HighLevelRuntime *runtime);
	void GiveNorm(T *exact, Context ctx, HighLevelRuntime *runtime);
	
};

template<typename T>
Array<T>::Array(int64_t size, int64_t nparts, Context ctx, HighLevelRuntime *runtime){

	this-> fid = FID_X;
	this-> size = size;
	this-> nparts = nparts;

	rect = Rect<1>(Point<1>(0), Point<1>(size-1));
	is = runtime->create_index_space(ctx,
					 Domain::from_rect<1>(rect));
	fs = runtime->create_field_space(ctx);
	{
		FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
		allocator.allocate_field(sizeof(int64_t), FID_X);
	}
	lr = runtime->create_logical_region(ctx, is, fs);

	// partition the logical region
	Rect<1> color_bounds(Point<1>(0), Point<1>(nparts-1));
	color_domain = Domain::from_rect<1>(color_bounds);
	
	DomainColoring coloring;
	int index = 0;
	const int local_size = (size + nparts -1) / nparts;
	for(int color = 0; color < nparts-1; color++){
		assert((index + local_size) <= size);
		Rect<1> subrect(Point<1>(index), Point<1>(index + local_size - 1));
		coloring[color] = Domain::from_rect<1>(subrect);
		index += local_size;
	}
	Rect<1> subrect(Point<1>(index), Point<1>(size-1));
	coloring[nparts-1] = Domain::from_rect<1>(subrect);

	ip = runtime->create_index_partition(ctx, is, color_domain,
					     coloring, true/*disjoint*/);
	lp = runtime->get_logical_partition(ctx, lr, ip);

}


template<typename T>
void Array<T>::DestroyArray(Context ctx, HighLevelRuntime *runtime){

	runtime->destroy_logical_region(ctx, lr);
        runtime->destroy_field_space(ctx, fs);
        runtime->destroy_index_space(ctx, is);

}

template<typename T>
void Array<T>::Initialize(Context ctx, HighLevelRuntime *runtime) {

	ArgumentMap arg_map;

        IndexLauncher init_launcher(V_INIT_TASK_ID, color_domain,
                                    TaskArgument(NULL, 0), arg_map);
        init_launcher.add_region_requirement(
                        RegionRequirement(lp, 0, WRITE_DISCARD, EXCLUSIVE, lr));
        init_launcher.region_requirements[0].add_field(FID_X);

        runtime->execute_index_space(ctx, init_launcher);
	
	return;
}

template<typename T>
void Array<T>::Initialize(T *input, Context ctx, HighLevelRuntime *runtime) {

		RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
        req.add_field(FID_X);

        InlineLauncher init_launcher(req);
        PhysicalRegion init_region = runtime->map_region(ctx, init_launcher);
        init_region.wait_until_valid();

        RegionAccessor<AccessorType::Generic, T> acc_x =
        init_region.get_field_accessor(FID_X).typeify<T>();

        // insert input values into the logical region
        Rect<1> subrect;
        ByteOffset offsets[1];
        T *x_ptr = acc_x.template raw_rect_ptr<1>(rect, subrect, offsets);
        if (!x_ptr || (subrect != rect) ||
            !offsets_are_dense<1,T>(rect, offsets))
        {
                GenericPointInRectIterator<1> itr(rect);

                for(int i=0; i<size; i++) {
                        acc_x.write(DomainPoint::from_point<1>(itr.p), input[i]);
                        itr++;
                }
        }
        else
        {
          for (int i = 0; i < size;i++) {
            x_ptr[i] = input[i];
          }
        }

        runtime->unmap_region(ctx, init_region);
        return;
}

template<typename T>
void Array<T>::RandomInit(Context ctx, HighLevelRuntime *runtime) {

		RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
        req.add_field(FID_X);

        InlineLauncher init_launcher(req);
        PhysicalRegion init_region = runtime->map_region(ctx, init_launcher);
        init_region.wait_until_valid();

        RegionAccessor<AccessorType::Generic, T> acc_x =
        init_region.get_field_accessor(FID_X).typeify<T>();

        // insert input values into the logical region
        Rect<1> subrect;
        ByteOffset offsets[1];
        T *x_ptr = acc_x.template raw_rect_ptr<1>(rect, subrect, offsets);
        if (!x_ptr || (subrect != rect) ||
            !offsets_are_dense<1,T>(rect, offsets))
        {
                GenericPointInRectIterator<1> itr(rect);

                for(int i=0; i<size; i++) {
                        acc_x.write(DomainPoint::from_point<1>(itr.p), drand48());
                        itr++;
                }
        }
        else
        {
          for (int i = 0; i < size;i++) {
            x_ptr[i] = drand48();
          }
        }

        runtime->unmap_region(ctx, init_region);
	return;
}

// Register the initialization task
template<typename T>
void RegisterVectorTask(void){
        HighLevelRuntime::register_legion_task<Initialize_Task<T> >(V_INIT_TASK_ID,
                                                 Processor::LOC_PROC,
                                                 true /*single*/,
                                                 true /*index*/);

        return;
}



template<typename T>
void Initialize_Task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx,
                      HighLevelRuntime *runtime){

	assert(regions.size() == 1);
	assert(task->regions.size() == 1);

	RegionAccessor<AccessorType::Generic, T> acc_x =
        regions[0].get_field_accessor(FID_X).typeify<T>();

	Domain dom = runtime->get_index_space_domain(ctx,
			      task->regions[0].region.get_index_space());
	Rect<1> rect = dom.get_rect<1>();

	// initialize the vector with zero
        Rect<1> subrect;
        ByteOffset offsets[1];
        T *x_ptr = acc_x.template raw_rect_ptr<1>(rect, subrect, offsets);
        if (!x_ptr || (rect != subrect) || 
            !offsets_are_dense<1,T>(rect, offsets))
	{
		GenericPointInRectIterator<1> itr(rect);

                for(int i=0; i < rect.volume(); i++) {
                        acc_x.write(DomainPoint::from_point<1>(itr.p), 0.0);
                        itr++;
                }
        }
        else
        {
          const size_t volume = rect.volume();
          for (int i = 0; i < volume; i++) {
            x_ptr[i] = 0.0;
          }
        }
	
	return;
}

template<typename T>
void Array<T>::PrintVals(Context ctx, HighLevelRuntime *runtime) {

	RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_X);

        InlineLauncher init_launcher(req);
        PhysicalRegion init_region = runtime->map_region(ctx, init_launcher);
        init_region.wait_until_valid();

        RegionAccessor<AccessorType::Generic, T> acc_x =
        init_region.get_field_accessor(FID_X).typeify<T>();

	T norm = 0.0;
        // insert input values into the logical region
        {
                GenericPointInRectIterator<1> itr(rect);

                for(int i=0; i<size; i++) {
                        T val = acc_x.read(DomainPoint::from_point<1>(itr.p));
			std::cout<<i<<"  "<<val<<std::endl;

			//norm += (exact[i]-val) * (exact[i]-val);
                        itr++;
                }
		//norm = sqrt(norm/size);
		//std::cout<<"L2Norm = "<<norm<<std::endl;
        }

        runtime->unmap_region(ctx, init_region);

	return;
}

template<typename T>
void Array<T>::GiveNorm(T *exact, Context ctx, HighLevelRuntime *runtime) {

        RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_X);

        InlineLauncher init_launcher(req);
        PhysicalRegion init_region = runtime->map_region(ctx, init_launcher);
        init_region.wait_until_valid();

        RegionAccessor<AccessorType::Generic, T> acc_x =
        init_region.get_field_accessor(FID_X).typeify<T>();

        T norm = 0.0;
        // insert input values into the logical region
        {
                GenericPointInRectIterator<1> itr(rect);

                for(int i=0; i<size; i++) {
                        T val = acc_x.read(DomainPoint::from_point<1>(itr.p));
                        //std::cout<<i<<"   "<<std::abs(exact[i]-val)<<std::endl;

			(std::abs(exact[i]-val) > norm) ? norm = std::abs(exact[i]-val) : norm = norm;
                        itr++;
                }
                std::cout<<"L_inf_Norm = "<<norm<<std::endl;
        }

        runtime->unmap_region(ctx, init_region);

        return;
}

#endif
