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

#ifndef cgoperators_hpp
#define cgoperators_hpp

#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include "legion.h"

#include "legionvector.hpp"
#include "ell_sparsematrix.hpp"


#if defined(__SSE2__)
//#include <x86intrin.h>
#include <emmintrin.h>
#endif
#if defined(__AVX__)
#include <immintrin.h>
#endif

#if defined(__SSE2__) || defined(__AVX__)
template<unsigned BOUNDARY>
static inline bool aligned(const void *ptr)
{
  return ((((unsigned long)ptr) & (BOUNDARY-1)) == 0);
}
#endif

static inline bool offset_mismatch(int i, const ByteOffset *off1, const ByteOffset *off2)
{
  while (i-- > 0)
    if ((off1++)->offset != (off2++)->offset)
      return true;
  return false;
}

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

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

enum TaksIDs{
	SPMV_TASK_ID = 3,
	SUBTRACT_TASK_ID = 4,
        SUBTRACT_INPLACE_TASK_ID = 5,
	COPY_TASK_ID = 6,
	DOT_TASK_ID = 7,
	ADD_TASK_ID = 8,
        ADD_INPLACE_TASK_ID = 9,
        AXPY_INPLACE_TASK_ID = 10,
	L2NORM_TASK_ID = 11,
	DIVIDE_TASK_ID = 12,
        CONVERGENCE_TASK_ID = 13,
};

enum OpIDs{
	REDUCE_ID = 1,
};

template<typename T>
class TaskArgs1{

	public:
	int scalar;
	FieldID A_row_fid;
	FieldID A_col_fid;
	FieldID A_val_fid;
	FieldID x_fid;
	FieldID Ax_fid;

	TaskArgs1(void){};
	TaskArgs1(const SpMatrix &A, const Array<T> &x, Array<T> &Ax, int64_t scalar){
		
		this-> scalar = scalar;
		this-> A_row_fid = A.row_fid;
		this-> A_col_fid = A.col_fid;
		this-> A_val_fid = A.val_fid;
		this-> x_fid = x.fid;
		this-> Ax_fid = Ax.fid;
	}

};

template<typename T>
class TaskArgs2{
	
	public:
	T scalar;
	FieldID x_fid;
	FieldID y_fid;
	FieldID z_fid;

	TaskArgs2(void);
	TaskArgs2(const Array<T> &x, const Array<T> &y, Array<T> &z, T scalar = 0.0){

		this-> scalar = scalar;
		this-> x_fid = x.fid;
		this-> y_fid = y.fid;
		this-> z_fid = z.fid;
	}

	TaskArgs2(const Array<T> &x, Array<T> &y){
		
		this-> x_fid = x.fid;
                this-> y_fid = y.fid;
	}

        TaskArgs2(Array<T> &x, const Array<T> &y){
                this-> x_fid = x.fid;
                this-> y_fid = y.fid;
        }

};

// Reduction Op
class FutureSum {
	
  public:

  typedef double LHS;
  typedef double RHS;
  static const double identity;

  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);

  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

const double FutureSum::identity = 0.0;

template<>
void FutureSum::apply<true>(LHS &lhs, RHS rhs)
{
  lhs += rhs;
}

template<>
void FutureSum::apply<false>(LHS &lhs, RHS rhs)
{
  int64_t *target = (int64_t *)&lhs;
  union { int64_t as_int; double as_T; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_T = oldval.as_T + rhs;
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

template<>
void FutureSum::fold<true>(RHS &rhs1, RHS rhs2)
{
  rhs1 += rhs2;
}

template<>
void FutureSum::fold<false>(RHS &rhs1, RHS rhs2)
{
  int64_t *target = (int64_t *)&rhs1;
  union { int64_t as_int; double as_T; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_T = oldval.as_T + rhs2;
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

// A_x = A * x
template<typename T>
void spmv(const SpMatrix &A, const Array<T> &x, Array<T> &A_x, 
          const Predicate &pred, Context ctx,  HighLevelRuntime *runtime){
	
	
	ArgumentMap arg_map;

	TaskArgs1<T> spmv_args(A, x, A_x, A.max_nzeros);

	IndexLauncher spmv_launcher(SPMV_TASK_ID, x.color_domain,
				    TaskArgument(&spmv_args, sizeof(spmv_args)), 
                                    arg_map, pred);

	spmv_launcher.add_region_requirement(
			RegionRequirement(A.row_lp, 0, READ_ONLY, EXCLUSIVE, A.row_lr));
	spmv_launcher.region_requirements[0].add_field(A.row_fid);

	spmv_launcher.add_region_requirement(
                        RegionRequirement(A.elem_lp, 0, READ_ONLY, EXCLUSIVE, A.elem_lr));
	spmv_launcher.region_requirements[1].add_field(A.val_fid);
	spmv_launcher.region_requirements[1].add_field(A.col_fid); 

	// Note: all elements of vector x is given to each process
	spmv_launcher.add_region_requirement(
                        RegionRequirement(x.lr, 0, READ_ONLY, EXCLUSIVE, x.lr));
        spmv_launcher.region_requirements[2].add_field(x.fid);

	spmv_launcher.add_region_requirement(
                        RegionRequirement(A_x.lp, 0, WRITE_DISCARD, EXCLUSIVE, A_x.lr));
        spmv_launcher.region_requirements[3].add_field(A_x.fid);

	runtime->execute_index_space(ctx, spmv_launcher);
	
	return;
}

template<typename T>
void spmv_task(const Task *task,
	       const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime){

	assert(regions.size() == 4);
        assert(task->regions.size() == 4);

        const  TaskArgs1<T> task_args = *((const TaskArgs1<T>*)task->args);

	int max_nzeros = task_args.scalar;

        RegionAccessor<AccessorType::Generic, int64_t> acc_num_nzeros =
        regions[0].get_field_accessor(task_args.A_row_fid).template typeify<int64_t>();

        RegionAccessor<AccessorType::Generic, int64_t> acc_col =
        regions[1].get_field_accessor(task_args.A_col_fid).template typeify<int64_t>();

        RegionAccessor<AccessorType::Generic, T> acc_vals =
        regions[1].get_field_accessor(task_args.A_val_fid).template typeify<T>();

	RegionAccessor<AccessorType::Generic, T> acc_x =
        regions[2].get_field_accessor(task_args.x_fid).template typeify<T>();

	RegionAccessor<AccessorType::Generic, T> acc_Ax =
        regions[3].get_field_accessor(task_args.Ax_fid).template typeify<T>();

        Domain row_dom = runtime->get_index_space_domain(ctx,
                         task->regions[0].region.get_index_space());
        Rect<1> row_rect = row_dom.get_rect<1>();

        Domain elem_dom = runtime->get_index_space_domain(ctx,
                          task->regions[1].region.get_index_space());
        Rect<1> elem_rect = elem_dom.get_rect<1>();

	// apply matrix vector multiplication
	{
		T sum;	
		DomainPoint pir;
		pir.dim = 1;

		GenericPointInRectIterator<1> itr1(row_rect);
		GenericPointInRectIterator<1> itr2(elem_rect);

		for(int i=0; i < row_rect.volume(); i++){

			sum = 0.0;
			int limit = acc_num_nzeros.read(DomainPoint::from_point<1>(itr1.p));

			for(int counter=0; counter < limit; counter++){

				int ind = acc_col.read(DomainPoint::from_point<1>(itr2.p));
				pir.point_data[0] = ind;
			
				sum += acc_vals.read(DomainPoint::from_point<1>(itr2.p)) * acc_x.read(pir);
				itr2++;
			}
			
			acc_Ax.write(DomainPoint::from_point<1>(itr1.p), sum);
			itr1++;
			itr2.p.x[0] = itr2.p.x[0] + (max_nzeros - limit);
		}
	}
	
	return;
}

static bool dense_spmv(int max_nzeros, const Rect<1> &subgrid_bounds, 
		       		   const Rect<1> &elem_bounds,
		               const Rect<1> &vec_bounds,
		       RegionAccessor<AccessorType::Generic,int64_t> &fa_nzero,
                       RegionAccessor<AccessorType::Generic,int64_t> &fa_col,
                       RegionAccessor<AccessorType::Generic,double> &fa_val,
                       RegionAccessor<AccessorType::Generic,double> &fa_x,
                       RegionAccessor<AccessorType::Generic,double> &fa_ax)
{
  Rect<1> subrect;
  ByteOffset in_offsets[1], offsets[1];

  const int64_t *in_nzero_ptr = fa_nzero.raw_rect_ptr<1>(subgrid_bounds, subrect, in_offsets);
  if (!in_nzero_ptr || (subrect != subgrid_bounds) ||
      !offsets_are_dense<1,int64_t>(subgrid_bounds, in_offsets)) return false;

  const int64_t *in_col_ptr = fa_col.raw_rect_ptr<1>(elem_bounds, subrect, in_offsets);
  if (!in_col_ptr || (subrect != elem_bounds) ||
      !offsets_are_dense<1,int64_t>(elem_bounds, in_offsets)) return false;

  const double *in_val_ptr = fa_val.raw_rect_ptr<1>(elem_bounds, subrect, offsets);
  if (!in_val_ptr || (subrect != elem_bounds) || 
      offset_mismatch(1, in_offsets, offsets)) return false;

  const double *in_x_ptr = fa_x.raw_rect_ptr<1>(vec_bounds, subrect, offsets);
  if (!in_x_ptr || (subrect != vec_bounds) ||
      !offsets_are_dense<1,double>(vec_bounds, offsets)) return false;

  // No offset check here
  double *out_ax_ptr = fa_ax.raw_rect_ptr<1>(subgrid_bounds, subrect, offsets);
  if (!out_ax_ptr || (subrect != subgrid_bounds) ||
      !offsets_are_dense<1,double>(subgrid_bounds, offsets)) return false;

  int n_rows = subgrid_bounds.volume();
#define STRIP_SIZE 256
  while (n_rows > 0) {
    if (n_rows>= STRIP_SIZE) {
      for (int i = 0; i < STRIP_SIZE; i++) {
        double sum = 0.0;
        int limit = in_nzero_ptr[i];
        for (int j = 0; j < limit; j++) {
          int index = in_col_ptr[i*max_nzeros+j];
          double val = in_val_ptr[i*max_nzeros+j];
          double xval = in_x_ptr[index];
          sum += (val * xval);
        }
        out_ax_ptr[i] = sum;
      }
      n_rows -= STRIP_SIZE;
      in_nzero_ptr += STRIP_SIZE;
      in_col_ptr += (max_nzeros*STRIP_SIZE);
      in_val_ptr += (max_nzeros*STRIP_SIZE);
      out_ax_ptr += STRIP_SIZE;
    } else {
      for (int i = 0; i < n_rows; i++) {
        double sum = 0.0;
	int limit = in_nzero_ptr[i];
        for (int j = 0; j < limit; j++) {
          int index = in_col_ptr[i*max_nzeros+j]; 
          double val = in_val_ptr[i*max_nzeros+j];
          double xval = in_x_ptr[index];
          sum += (val * xval);         
        }
        out_ax_ptr[i] = sum;
      }
      n_rows = 0;
    }
  }
#undef STRIP_SIZE
  return true;
}

template<>
void spmv_task<double>(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, HighLevelRuntime *runtime) 
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);

  const  TaskArgs1<double> task_args = *((const TaskArgs1<double>*)task->args);

  const int max_nzeros = task_args.scalar;

  RegionAccessor<AccessorType::Generic, int64_t> acc_num_nzeros =
  regions[0].get_field_accessor(task_args.A_row_fid).typeify<int64_t>();

  RegionAccessor<AccessorType::Generic, int64_t> acc_col =
  regions[1].get_field_accessor(task_args.A_col_fid).typeify<int64_t>();

  RegionAccessor<AccessorType::Generic, double> acc_vals =
  regions[1].get_field_accessor(task_args.A_val_fid).typeify<double>();

  RegionAccessor<AccessorType::Generic, double> acc_x =
  regions[2].get_field_accessor(task_args.x_fid).typeify<double>();

  RegionAccessor<AccessorType::Generic, double> acc_Ax =
  regions[3].get_field_accessor(task_args.Ax_fid).typeify<double>();

  Domain elem_dom = runtime->get_index_space_domain(ctx,
                    task->regions[1].region.get_index_space());
  Rect<1> elem_rect = elem_dom.get_rect<1>();

  Domain vec_dom = runtime->get_index_space_domain(ctx,
		   task->regions[2].region.get_index_space());
  Rect<1> vec_rect = vec_dom.get_rect<1>();

  Domain row_dom = runtime->get_index_space_domain(ctx,
                   task->regions[3].region.get_index_space());
  Rect<1> row_rect = row_dom.get_rect<1>();

  
  if(dense_spmv(max_nzeros, row_rect, elem_rect, vec_rect, acc_num_nzeros, acc_col, acc_vals, acc_x, acc_Ax)){
    return;
  }  
  else{

  	// Otherwise we fall back
  	GenericPointInRectIterator<1> itr1(row_rect);
  	GenericPointInRectIterator<1> itr2(elem_rect);
  	const int volume = row_rect.volume();
  	DomainPoint pir;
  	pir.dim = 1;
  	for (int i = 0; i < volume; i++) {
    		double sum = 0.0;
		int limit = acc_num_nzeros.read(DomainPoint::from_point<1>(itr1.p));

    		for (int j = 0; j < limit; j++) {

			int ind = acc_col.read(DomainPoint::from_point<1>(itr2.p));
                                pir.point_data[0] = ind;

                                sum += acc_vals.read(DomainPoint::from_point<1>(itr2.p)) * acc_x.read(pir);
                                itr2++;
                        }

                        acc_Ax.write(DomainPoint::from_point<1>(itr1.p), sum);
                        itr1++;
                        itr2.p.x[0] = itr2.p.x[0] + (max_nzeros - limit);
  	}
  }
 
 return;
}

// b -= Ax
template<typename T>
void subtract_inplace(Array<T> &b, const Array<T> &A_x, Future coef, 
                      const Predicate &pred, Context ctx, HighLevelRuntime *runtime){

        ArgumentMap arg_map;

	TaskArgs2<T> subtract_args(b, A_x);
	
        IndexLauncher subtract_launcher(SUBTRACT_INPLACE_TASK_ID, b.color_domain,
                              TaskArgument(&subtract_args, sizeof(subtract_args)),
                              arg_map, pred);

	subtract_launcher.add_future(coef);

        subtract_launcher.add_region_requirement(
                        RegionRequirement(b.lp, 0, READ_WRITE, EXCLUSIVE, b.lr));
        subtract_launcher.region_requirements[0].add_field(b.fid);

        subtract_launcher.add_region_requirement(
                        RegionRequirement(A_x.lp, 0, READ_ONLY, EXCLUSIVE, A_x.lr));
        subtract_launcher.region_requirements[1].add_field(A_x.fid);

        runtime->execute_index_space(ctx, subtract_launcher);

	return;
}

// b - Ax
template<typename T>
void subtract(const Array<T> &b, const Array<T> &A_x, Array<T> &r, T coef, Context ctx, HighLevelRuntime *runtime){

        ArgumentMap arg_map;

	TaskArgs2<T> subtract_args(b, A_x, r, coef);
	
        IndexLauncher subtract_launcher(SUBTRACT_TASK_ID, b.color_domain,
                                    TaskArgument(&subtract_args, sizeof(subtract_args)), arg_map);

        subtract_launcher.add_region_requirement(
                        RegionRequirement(b.lp, 0, READ_ONLY, EXCLUSIVE, b.lr));
        subtract_launcher.region_requirements[0].add_field(b.fid);

        subtract_launcher.add_region_requirement(
                        RegionRequirement(A_x.lp, 0, READ_ONLY, EXCLUSIVE, A_x.lr));
        subtract_launcher.region_requirements[1].add_field(A_x.fid);

        subtract_launcher.add_region_requirement(
                        RegionRequirement(r.lp, 0, WRITE_DISCARD, EXCLUSIVE, r.lr));
        subtract_launcher.region_requirements[2].add_field(r.fid);

        runtime->execute_index_space(ctx, subtract_launcher);

	return;
}

template<typename T>
void subtract_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime){

        assert(regions.size() == 3);
        assert(task->regions.size() == 3);

        const TaskArgs2<T> task_args = *((const TaskArgs2<T>*)task->args);

	T alpha = task_args.scalar;

        RegionAccessor<AccessorType::Generic, T> acc_x =
        regions[0].get_field_accessor(task_args.x_fid).template typeify<T>();

        RegionAccessor<AccessorType::Generic, T> acc_y =
        regions[1].get_field_accessor(task_args.y_fid).template typeify<T>();

        RegionAccessor<AccessorType::Generic, T> acc_z =
        regions[2].get_field_accessor(task_args.z_fid).template typeify<T>();

        Domain dom = runtime->get_index_space_domain(ctx,
                         task->regions[0].region.get_index_space());
        Rect<1> rect = dom.get_rect<1>();

        // apply vector subtraction
        {
                GenericPointInRectIterator<1> itr1(rect);

                for(int i=0; i< rect.volume(); i++){
			T result = acc_x.read(DomainPoint::from_point<1>(itr1.p)) - 
					alpha * acc_y.read(DomainPoint::from_point<1>(itr1.p));
			
			 acc_z.write(DomainPoint::from_point<1>(itr1.p), result);
			 itr1++;
                 }
        }

        return;
}

template<typename T>
void subtract_inplace_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, HighLevelRuntime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->futures.size() == 1);

  const TaskArgs2<T> task_args = *((const TaskArgs2<T>*)task->args);

  Future dummy = task->futures[0];
  T alpha = dummy.template get_result<T>();

  RegionAccessor<AccessorType::Generic, T> acc_x =
  regions[0].get_field_accessor(task_args.x_fid).template typeify<T>();

  RegionAccessor<AccessorType::Generic, T> acc_y =
  regions[1].get_field_accessor(task_args.y_fid).template typeify<T>();

  Domain dom = runtime->get_index_space_domain(ctx,
                   task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  GenericPointInRectIterator<1> itr(rect);
  const int volume = rect.volume();
  for (int i = 0; i < volume; i++) {
    T temp = acc_x.read(DomainPoint::from_point<1>(itr.p)) - 
              alpha * acc_y.read(DomainPoint::from_point<1>(itr.p));
    acc_x.write(DomainPoint::from_point<1>(itr.p), temp);
    itr++;
  }
}

static bool dense_subtract(const Rect<1> &subgrid_bounds, double alpha,
                           RegionAccessor<AccessorType::Generic,double> &fa_x,
                           RegionAccessor<AccessorType::Generic,double> &fa_y,
                           RegionAccessor<AccessorType::Generic,double> &fa_z)
{
  Rect<1> subrect;
  ByteOffset in_offsets[1], offsets[1];

  const double *in_x_ptr = fa_x.raw_rect_ptr<1>(subgrid_bounds, subrect, in_offsets);
  if (!in_x_ptr || (subrect != subgrid_bounds) ||
      !offsets_are_dense<1,double>(subgrid_bounds, in_offsets)) return false;

  const double *in_y_ptr = fa_y.raw_rect_ptr<1>(subgrid_bounds, subrect, offsets);
  if (!in_y_ptr || (subrect != subgrid_bounds) || 
      offset_mismatch(1, in_offsets, offsets)) return false;

  double *out_z_ptr = fa_z.raw_rect_ptr<1>(subgrid_bounds, subrect, offsets);
  if (!out_z_ptr || (subrect != subgrid_bounds) ||
      offset_mismatch(1, in_offsets, offsets)) return false;

  int n_pts = subgrid_bounds.volume();
#define STRIP_SIZE 256
  while (n_pts > 0) {
    if (n_pts >= STRIP_SIZE) {
#if defined(__AVX__)
      __m256d alphad = _mm256_set1_pd(alpha);
      if (aligned<32>(in_x_ptr) && aligned<32>(in_y_ptr) && aligned<32>(out_z_ptr)) {
        for (int i = 0; i < (STRIP_SIZE>>2); i++) {
          __m256d x = _mm256_load_pd(in_x_ptr+(i<<2));
          __m256d y = _mm256_load_pd(in_y_ptr+(i<<2));
          _mm256_store_pd(out_z_ptr+(i<<2), _mm256_sub_pd(x,_mm256_mul_pd(alphad,y)));
        }
      } else {
        for (int i = 0; i < (STRIP_SIZE>>2); i++) {
          __m256d x = _mm256_loadu_pd(in_x_ptr+(i<<2));
          __m256d y = _mm256_loadu_pd(in_y_ptr+(i<<2));
          _mm256_storeu_pd(out_z_ptr+(i<<2), _mm256_sub_pd(x,_mm256_mul_pd(alphad,y)));
        }
      }
#elif defined(__SSE2__)
      __m128d alphad = _mm_set1_pd(alpha);
      if (aligned<16>(in_x_ptr) && aligned<16>(in_y_ptr) && aligned<16>(out_z_ptr)) {
        for (int i = 0; i < (STRIP_SIZE>>1); i++) {
          __m128d x = _mm_load_pd(in_x_ptr+(i<<1));
          __m128d y = _mm_load_pd(in_y_ptr+(i<<1));
          _mm_stream_pd(out_z_ptr+(i<<1), _mm_sub_pd(x,_mm_mul_pd(alphad,y)));
        }
      } else {
        for (int i = 0; i < (STRIP_SIZE>>1); i++) {
          __m128d x = _mm_loadu_pd(in_x_ptr+(i<<1));
          __m128d y = _mm_loadu_pd(in_y_ptr+(i<<1));
          _mm_storeu_pd(out_z_ptr+(i<<1), _mm_sub_pd(x,_mm_mul_pd(alphad,y)));
        }
      }
#else
      for (int i = 0; i < STRIP_SIZE; i++)
        out_z_ptr[i] = in_x_ptr[i] - alpha * in_y_ptr[i];
#endif
      n_pts -= STRIP_SIZE;
      in_x_ptr += STRIP_SIZE;
      in_y_ptr += STRIP_SIZE;
      out_z_ptr += STRIP_SIZE;
    } else {
      for (int i = 0; i < n_pts; i++)
        out_z_ptr[i] = in_x_ptr[i] - alpha * in_y_ptr[i];
      n_pts = 0;
    }
  }
#ifdef __AVX__
  _mm256_zeroall();
#endif
#undef STRIP_SIZE
  return true;
}

template<>
void subtract_task<double>(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);

  const TaskArgs2<double> task_args = *((const TaskArgs2<double>*)task->args);

  double alpha = task_args.scalar;

  RegionAccessor<AccessorType::Generic, double> acc_x =
  regions[0].get_field_accessor(task_args.x_fid).typeify<double>();

  RegionAccessor<AccessorType::Generic, double> acc_y =
  regions[1].get_field_accessor(task_args.y_fid).typeify<double>();

  RegionAccessor<AccessorType::Generic, double> acc_z =
  regions[2].get_field_accessor(task_args.z_fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
                         task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  if (dense_subtract(rect, alpha, acc_x, acc_y, acc_z))
    return;

  // Otherwise fall back
  GenericPointInRectIterator<1> itr(rect);
  const int volume = rect.volume();
  for (int i = 0; i < volume; i++) {
    double result = acc_x.read(DomainPoint::from_point<1>(itr.p)) -
                    alpha * acc_y.read(DomainPoint::from_point<1>(itr.p));
    acc_z.write(DomainPoint::from_point<1>(itr.p), result);
    itr++;
  }
}

static bool dense_subtract_inplace(const Rect<1> &subgrid_bounds, double alpha,
                                   RegionAccessor<AccessorType::Generic,double> &fa_x,
                                   RegionAccessor<AccessorType::Generic,double> &fa_y)
{
  Rect<1> subrect;
  ByteOffset in_offsets[1], offsets[1];

  double *inout_x_ptr = fa_x.raw_rect_ptr<1>(subgrid_bounds, subrect, in_offsets);
  if (!inout_x_ptr || (subrect != subgrid_bounds) ||
      !offsets_are_dense<1,double>(subgrid_bounds, in_offsets)) return false;

  const double *in_y_ptr = fa_y.raw_rect_ptr<1>(subgrid_bounds, subrect, offsets);
  if (!in_y_ptr || (subrect != subgrid_bounds) || 
      offset_mismatch(1, in_offsets, offsets)) return false;

  int n_pts = subgrid_bounds.volume();
#define STRIP_SIZE 256
  while (n_pts > 0) {
    if (n_pts >= STRIP_SIZE) {
#if defined(__AVX__)
      __m256d alphad = _mm256_set1_pd(alpha);
      if (aligned<32>(inout_x_ptr) && aligned<32>(in_y_ptr)) {
        for (int i = 0; i < (STRIP_SIZE>>2); i++) {
          __m256d x = _mm256_load_pd(inout_x_ptr+(i<<2));
          __m256d y = _mm256_load_pd(in_y_ptr+(i<<2));
          _mm256_store_pd(inout_x_ptr+(i<<2), _mm256_sub_pd(x,_mm256_mul_pd(alphad,y)));
        }
      } else {
        for (int i = 0; i < (STRIP_SIZE>>2); i++) {
          __m256d x = _mm256_loadu_pd(inout_x_ptr+(i<<2));
          __m256d y = _mm256_loadu_pd(in_y_ptr+(i<<2));
          _mm256_storeu_pd(inout_x_ptr+(i<<2), _mm256_sub_pd(x,_mm256_mul_pd(alphad,y)));
        }
      }
#elif defined(__SSE2__)
      __m128d alphad = _mm_set1_pd(alpha);
      if (aligned<16>(inout_x_ptr) && aligned<16>(in_y_ptr)) {
        for (int i = 0; i < (STRIP_SIZE>>1); i++) {
          __m128d x = _mm_load_pd(inout_x_ptr+(i<<1));
          __m128d y = _mm_load_pd(in_y_ptr+(i<<1));
          _mm_store_pd(inout_x_ptr+(i<<1), _mm_sub_pd(x,_mm_mul_pd(alphad,y)));
        }
      } else {
        for (int i = 0; i < (STRIP_SIZE>>1); i++) {
          __m128d x = _mm_loadu_pd(inout_x_ptr+(i<<1));
          __m128d y = _mm_loadu_pd(in_y_ptr+(i<<1));
          _mm_storeu_pd(inout_x_ptr+(i<<1), _mm_sub_pd(x,_mm_mul_pd(alphad,y)));
        }
      }
#else
      for (int i = 0; i < STRIP_SIZE; i++)
        inout_x_ptr[i] -= (alpha * in_y_ptr[i]);
#endif
      n_pts -= STRIP_SIZE;
      inout_x_ptr += STRIP_SIZE;
      in_y_ptr += STRIP_SIZE;
    } else {
      for (int i = 0; i < n_pts; i++)
        inout_x_ptr[i] -= (alpha * in_y_ptr[i]);
      n_pts = 0;
    }
  }
#ifdef __AVX__
  _mm256_zeroall();
#endif
#undef STRIP_SIZE
  return true;
}

template<>
void subtract_inplace_task<double>(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->futures.size() == 1);

  const TaskArgs2<double> task_args = *((const TaskArgs2<double>*)task->args);

  Future dummy = task->futures[0]; 
  double alpha = dummy.get_result<double>();

  RegionAccessor<AccessorType::Generic, double> acc_x =
  regions[0].get_field_accessor(task_args.x_fid).typeify<double>();

  RegionAccessor<AccessorType::Generic, double> acc_y =
  regions[1].get_field_accessor(task_args.y_fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
                   task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  if (dense_subtract_inplace(rect, alpha, acc_x, acc_y))
    return;

  GenericPointInRectIterator<1> itr(rect);
  const int volume = rect.volume();
  for (int i = 0; i < volume; i++) {
    double temp = acc_x.read(DomainPoint::from_point<1>(itr.p)) - 
              alpha * acc_y.read(DomainPoint::from_point<1>(itr.p));
    acc_x.write(DomainPoint::from_point<1>(itr.p), temp);
    itr++;
  }
}

// p = r_old
template<typename T>
void copy(const Array<T> &r, Array<T> &p, Context ctx, HighLevelRuntime *runtime){


        CopyLauncher equal_launcher;

        equal_launcher.add_copy_requirements(
                        RegionRequirement(r.lr, READ_ONLY, EXCLUSIVE, r.lr),
			RegionRequirement(p.lr, WRITE_DISCARD, EXCLUSIVE, p.lr));

        equal_launcher.src_requirements[0].add_field(r.fid);

        equal_launcher.dst_requirements[0].add_field(p.fid);

        runtime->issue_copy_operation(ctx, equal_launcher);
	return;
}

// x' * y
template<typename T>
Future dot(const Array<T> &x, Array<T> &y, const Predicate &pred,
           const Future &false_result, Context ctx, HighLevelRuntime *runtime){
	
        ArgumentMap arg_map;

	TaskArgs2<T> dot_args(x, y);

        IndexLauncher dot_launcher(DOT_TASK_ID, x.color_domain,
                                    TaskArgument(&dot_args, sizeof(dot_args)), 
                                    arg_map, pred);

        dot_launcher.add_region_requirement(
                        RegionRequirement(x.lp, 0, READ_ONLY, EXCLUSIVE, x.lr));
        dot_launcher.region_requirements[0].add_field(x.fid);

        dot_launcher.add_region_requirement(
                        RegionRequirement(y.lp, 0, READ_ONLY, EXCLUSIVE, y.lr));
        dot_launcher.region_requirements[1].add_field(y.fid);

        dot_launcher.set_predicate_false_future(false_result);

        Future result = runtime->execute_index_space(ctx, dot_launcher, REDUCE_ID);
	
	return(result);
}

template<typename T>
T  dot_task(const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx, HighLevelRuntime *runtime){  

        assert(regions.size() == 2);
        assert(task->regions.size() == 2);

        const TaskArgs2<T> task_args = *((const TaskArgs2<T>*)task->args);
                
        RegionAccessor<AccessorType::Generic, T> acc_x =
        regions[0].get_field_accessor(task_args.x_fid).template typeify<T>();

        RegionAccessor<AccessorType::Generic, T> acc_y =
        regions[1].get_field_accessor(task_args.y_fid).template typeify<T>();

        Domain dom = runtime->get_index_space_domain(ctx,
                         task->regions[0].region.get_index_space());
        Rect<1> rect = dom.get_rect<1>();
	
	T sum = 0.0;
        // now apply the dot product using reduction operator!!!
        {
		GenericPointInRectIterator<1> itr1(rect);

		for(int i=0; i < rect.volume(); i++){
			sum += acc_x.read(DomainPoint::from_point<1>(itr1.p)) * 
				acc_y.read(DomainPoint::from_point<1>(itr1.p));

			itr1++;				
		}
        }
        return(sum);
}

static bool dense_dot(const Rect<1> &subgrid_bounds, double &result, 
                      RegionAccessor<AccessorType::Generic,double> &fa_x,
                      RegionAccessor<AccessorType::Generic,double> &fa_y)
{
  Rect<1> subrect;
  ByteOffset in_offsets[1], offsets[1];

  const double *in_x_ptr = fa_x.raw_rect_ptr<1>(subgrid_bounds, subrect, in_offsets);
  if (!in_x_ptr || (subrect != subgrid_bounds) ||
      !offsets_are_dense<1,double>(subgrid_bounds, in_offsets)) return false;

  const double *in_y_ptr = fa_y.raw_rect_ptr<1>(subgrid_bounds, subrect, offsets);
  if (!in_y_ptr || (subrect != subgrid_bounds) || 
      offset_mismatch(1, in_offsets, offsets)) return false;

  int n_pts = subgrid_bounds.volume();
  result = 0.0;
#if defined(__AVX__)
  __m256d temp = _mm256_set1_pd(0.0);
#elif defined(__SSE2__)
  __m128d temp = _mm_set1_pd(0.0);
#endif
#define STRIP_SIZE 256
  while (n_pts > 0) {
    if (n_pts >= STRIP_SIZE) {
#if defined(__AVX__)
      if (aligned<32>(in_x_ptr) && aligned<32>(in_y_ptr)) {
        for (int i = 0; i < (STRIP_SIZE>>2); i++) {
          __m256d x = _mm256_load_pd(in_x_ptr+(i<<2));
          __m256d y = _mm256_load_pd(in_y_ptr+(i<<2));
          temp = _mm256_add_pd(temp,_mm256_mul_pd(x,y));
        }
      } else {
        for (int i = 0; i < (STRIP_SIZE>>2); i++) {
          __m256d x = _mm256_loadu_pd(in_x_ptr+(i<<2));
          __m256d y = _mm256_loadu_pd(in_y_ptr+(i<<2));
          temp = _mm256_add_pd(temp, _mm256_mul_pd(x,y));
        }
      }
#elif defined(__SSE2__)
      if (aligned<16>(in_x_ptr) && aligned<16>(in_y_ptr)) {
        for (int i = 0; i < (STRIP_SIZE>>1); i++) {
          __m128d x = _mm_load_pd(in_x_ptr+(i<<1));
          __m128d y = _mm_load_pd(in_y_ptr+(i<<1));
          temp = _mm_add_pd(temp,_mm_mul_pd(x,y));
        }
      } else {
        for (int i = 0; i < (STRIP_SIZE>>1); i++) {
          __m128d x = _mm_loadu_pd(in_x_ptr+(i<<1));
          __m128d y = _mm_loadu_pd(in_y_ptr+(i<<1));
          temp = _mm_add_pd(temp,_mm_mul_pd(x,y));
        }
      }
#else
      for (int i = 0; i < STRIP_SIZE; i++) {
        result += (in_x_ptr[i] * in_y_ptr[i]);
      }
#endif
      n_pts -= STRIP_SIZE;
      in_x_ptr += STRIP_SIZE;
      in_y_ptr += STRIP_SIZE;
    } else {
      for (int i = 0; i < n_pts; i++) {
        result += (in_x_ptr[i] * in_y_ptr[i]); 
      }
      n_pts = 0;
    }
  }
#if defined(__AVX__)
  __m128d lower = _mm256_extractf128_pd(temp, 0);
  __m128d upper = _mm256_extractf128_pd(temp, 1);
  result += _mm_cvtsd_f64(lower);
  result += _mm_cvtsd_f64(_mm_shuffle_pd(lower,lower,1));
  result += _mm_cvtsd_f64(upper);
  result += _mm_cvtsd_f64(_mm_shuffle_pd(upper,upper,1));
  _mm256_zeroall();
#elif defined(__SSE2__)
  result += _mm_cvtsd_f64(temp);
  result += _mm_cvtsd_f64(_mm_shuffle_pd(temp,temp,1));
#endif
#undef STRIP_SIZE
  return true;
}

template<>
double dot_task<double>(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);

  const TaskArgs2<double> task_args = *((const TaskArgs2<double>*)task->args);
          
  RegionAccessor<AccessorType::Generic, double> acc_x =
  regions[0].get_field_accessor(task_args.x_fid).typeify<double>();

  RegionAccessor<AccessorType::Generic, double> acc_y =
  regions[1].get_field_accessor(task_args.y_fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
                   task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();
  
  double sum = 0.0;
  if (dense_dot(rect, sum, acc_x, acc_y))
    return sum;

  GenericPointInRectIterator<1> itr(rect);
  const int volume = rect.volume();
  for (int i = 0; i < volume; i++) {
    sum += (acc_x.read(DomainPoint::from_point<1>(itr.p)) * 
            acc_y.read(DomainPoint::from_point<1>(itr.p)));
    itr++;
  }
  return sum;
}

// z = x + y
template<typename T>
void add(const Array<T> &x, const Array<T> &y, Array<T> &z, Future coef, Context ctx, HighLevelRuntime *runtime){

        ArgumentMap arg_map;

	TaskArgs2<T> add_args(x, y, z);

        IndexLauncher add_launcher(ADD_TASK_ID, x.color_domain,
                                    TaskArgument(&add_args, sizeof(add_args)), arg_map);

	add_launcher.add_future(coef);

        add_launcher.add_region_requirement(
                        RegionRequirement(x.lp, 0, READ_ONLY, EXCLUSIVE, x.lr));
        add_launcher.region_requirements[0].add_field(x.fid);

        add_launcher.add_region_requirement(
                        RegionRequirement(y.lp, 0, READ_ONLY, EXCLUSIVE, y.lr));
        add_launcher.region_requirements[1].add_field(y.fid);

        add_launcher.add_region_requirement(
                        RegionRequirement(z.lp, 0, WRITE_DISCARD, EXCLUSIVE, z.lr));
        add_launcher.region_requirements[2].add_field(z.fid);

        runtime->execute_index_space(ctx, add_launcher);

	return;
}

template<typename T>
void add_inplace(Array<T> &x, const Array<T> &y, Future coef, 
                 const Predicate &pred, Context ctx, HighLevelRuntime *runtime){
        ArgumentMap arg_map;

	TaskArgs2<T> add_args(x, y);

        IndexLauncher add_launcher(ADD_INPLACE_TASK_ID, x.color_domain,
                                    TaskArgument(&add_args, sizeof(add_args)), 
                                    arg_map, pred);

	add_launcher.add_future(coef);

        add_launcher.add_region_requirement(
                        RegionRequirement(x.lp, 0, READ_WRITE, EXCLUSIVE, x.lr));
        add_launcher.region_requirements[0].add_field(x.fid);

        add_launcher.add_region_requirement(
                        RegionRequirement(y.lp, 0, READ_ONLY, EXCLUSIVE, y.lr));
        add_launcher.region_requirements[1].add_field(y.fid);

        runtime->execute_index_space(ctx, add_launcher);

	return;

}

template<typename T>
void add_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime){

        assert(regions.size() == 3);
        assert(task->regions.size() == 3);
        assert(task->futures.size() == 1);

        const TaskArgs2<T> task_args = *((const TaskArgs2<T>*)task->args);

  	T alpha;
  	Future dummy = task->futures[0];
  	alpha = dummy.get_result<T>();

        RegionAccessor<AccessorType::Generic, T> acc_x =
        regions[0].get_field_accessor(task_args.x_fid).template typeify<T>();

        RegionAccessor<AccessorType::Generic, T> acc_y =
        regions[1].get_field_accessor(task_args.y_fid).template typeify<T>();

        RegionAccessor<AccessorType::Generic, T> acc_z =
        regions[2].get_field_accessor(task_args.z_fid).template typeify<T>();

        Domain dom = runtime->get_index_space_domain(ctx,
                         task->regions[0].region.get_index_space());
        Rect<1> rect = dom.get_rect<1>();

        // apply vector addition
        {
                GenericPointInRectIterator<1> itr1(rect);

                for(int i=0; i< rect.volume(); i++){
                        T result = acc_x.read(DomainPoint::from_point<1>(itr1.p)) +
                                        alpha * acc_y.read(DomainPoint::from_point<1>(itr1.p));

                         acc_z.write(DomainPoint::from_point<1>(itr1.p), result);
                         itr1++;
                 }
        }

        return;
}

template<typename T>
void add_inplace_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, HighLevelRuntime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->futures.size() == 1);

  const TaskArgs2<T> task_args = *((const TaskArgs2<T>*)task->args);

  T alpha;
  Future dummy = task->futures[0];
  alpha = dummy.template get_result<T>();

  RegionAccessor<AccessorType::Generic, T> acc_x =
  regions[0].get_field_accessor(task_args.x_fid).template typeify<T>();

  RegionAccessor<AccessorType::Generic, T> acc_y =
  regions[1].get_field_accessor(task_args.y_fid).template typeify<T>();

  Domain dom = runtime->get_index_space_domain(ctx,
                   task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  GenericPointInRectIterator<1> itr(rect);
  const int volume = rect.volume();
  for (int i = 0; i < volume; i++) {
    T temp = acc_x.read(DomainPoint::from_point<1>(itr.p)) +
              alpha * acc_y.read(DomainPoint::from_point<1>(itr.p));
    acc_x.write(DomainPoint::from_point<1>(itr.p), temp);
    itr++;
  }
}

static bool dense_add(const Rect<1> &subgrid_bounds, double alpha,
                      RegionAccessor<AccessorType::Generic,double> &fa_x,
                      RegionAccessor<AccessorType::Generic,double> &fa_y,
                      RegionAccessor<AccessorType::Generic,double> &fa_z)
{
  Rect<1> subrect;
  ByteOffset in_offsets[1], offsets[1];

  const double *in_x_ptr = fa_x.raw_rect_ptr<1>(subgrid_bounds, subrect, in_offsets);
  if (!in_x_ptr || (subrect != subgrid_bounds) ||
      !offsets_are_dense<1,double>(subgrid_bounds, in_offsets)) return false;

  const double *in_y_ptr = fa_y.raw_rect_ptr<1>(subgrid_bounds, subrect, offsets);
  if (!in_y_ptr || (subrect != subgrid_bounds) || 
      offset_mismatch(1, in_offsets, offsets)) return false;

  double *out_z_ptr = fa_z.raw_rect_ptr<1>(subgrid_bounds, subrect, offsets);
  if (!out_z_ptr || (subrect != subgrid_bounds) ||
      offset_mismatch(1, in_offsets, offsets)) return false;

  int n_pts = subgrid_bounds.volume();
#define STRIP_SIZE 256
  while (n_pts > 0) {
    if (n_pts >= STRIP_SIZE) {
#if defined(__AVX__)
      __m256d alphad = _mm256_set1_pd(alpha);
      if (aligned<32>(in_x_ptr) && aligned<32>(in_y_ptr) && aligned<32>(out_z_ptr)) {
        for (int i = 0; i < (STRIP_SIZE>>2); i++) {
          __m256d x = _mm256_load_pd(in_x_ptr+(i<<2));
          __m256d y = _mm256_load_pd(in_y_ptr+(i<<2));
          _mm256_store_pd(out_z_ptr+(i<<2), _mm256_add_pd(x,_mm256_mul_pd(alphad,y)));
        }
      } else {
        for (int i = 0; i < (STRIP_SIZE>>2); i++) {
          __m256d x = _mm256_loadu_pd(in_x_ptr+(i<<2));
          __m256d y = _mm256_loadu_pd(in_y_ptr+(i<<2));
          _mm256_storeu_pd(out_z_ptr+(i<<2), _mm256_add_pd(x,_mm256_mul_pd(alphad,y)));
        }
      }
#elif defined(__SSE2__)
      __m128d alphad = _mm_set1_pd(alpha);
      if (aligned<16>(in_x_ptr) && aligned<16>(in_y_ptr) && aligned<16>(out_z_ptr)) {
        for (int i = 0; i < (STRIP_SIZE>>1); i++) {
          __m128d x = _mm_load_pd(in_x_ptr+(i<<1));
          __m128d y = _mm_load_pd(in_y_ptr+(i<<1));
          _mm_stream_pd(out_z_ptr+(i<<1), _mm_add_pd(x,_mm_mul_pd(alphad,y)));
        }
      } else {
        for (int i = 0; i < (STRIP_SIZE>>1); i++) {
          __m128d x = _mm_loadu_pd(in_x_ptr+(i<<1));
          __m128d y = _mm_loadu_pd(in_y_ptr+(i<<1));
          _mm_storeu_pd(out_z_ptr+(i<<1), _mm_add_pd(x,_mm_mul_pd(alphad,y)));
        }
      }
#else
      for (int i = 0; i < STRIP_SIZE; i++)
        out_z_ptr[i] = in_x_ptr[i] + alpha * in_y_ptr[i];
#endif
      n_pts -= STRIP_SIZE;
      in_x_ptr += STRIP_SIZE;
      in_y_ptr += STRIP_SIZE;
      out_z_ptr += STRIP_SIZE;
    } else {
      for (int i = 0; i < n_pts; i++)
        out_z_ptr[i] = in_x_ptr[i] + alpha * in_y_ptr[i];
      n_pts = 0;
    }
  }
#ifdef __AVX__
  _mm256_zeroall();
#endif
#undef STRIP_SIZE
  return true;
}

template<>
void add_task<double>(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);

  const TaskArgs2<double> task_args = *((const TaskArgs2<double>*)task->args);

  double alpha;
  Future dummy = task->futures[0];
  alpha = dummy.get_result<double>();

  RegionAccessor<AccessorType::Generic, double> acc_x =
  regions[0].get_field_accessor(task_args.x_fid).typeify<double>();

  RegionAccessor<AccessorType::Generic, double> acc_y =
  regions[1].get_field_accessor(task_args.y_fid).typeify<double>();

  RegionAccessor<AccessorType::Generic, double> acc_z =
  regions[2].get_field_accessor(task_args.z_fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
                   task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  if (dense_add(rect, alpha, acc_x, acc_y, acc_z))
    return;

  GenericPointInRectIterator<1> itr(rect);
  const int volume = rect.volume();
  for (int i = 0; i < volume; i++) {
    double result = acc_x.read(DomainPoint::from_point<1>(itr.p)) + 
                    alpha * acc_y.read(DomainPoint::from_point<1>(itr.p));
    acc_z.write(DomainPoint::from_point<1>(itr.p), result);
    itr++;
  }
}

static bool dense_add_inplace(const Rect<1> &subgrid_bounds, double alpha,
                              RegionAccessor<AccessorType::Generic,double> &fa_x,
                              RegionAccessor<AccessorType::Generic,double> &fa_y)
{
  Rect<1> subrect;
  ByteOffset in_offsets[1], offsets[1];

  double *inout_x_ptr = fa_x.raw_rect_ptr<1>(subgrid_bounds, subrect, in_offsets);
  if (!inout_x_ptr || (subrect != subgrid_bounds) ||
      !offsets_are_dense<1,double>(subgrid_bounds, in_offsets)) return false;

  const double *in_y_ptr = fa_y.raw_rect_ptr<1>(subgrid_bounds, subrect, offsets);
  if (!in_y_ptr || (subrect != subgrid_bounds) || 
      offset_mismatch(1, in_offsets, offsets)) return false;

  int n_pts = subgrid_bounds.volume();
#define STRIP_SIZE 256
  while (n_pts > 0) {
    if (n_pts >= STRIP_SIZE) {
#if defined(__AVX__)
      __m256d alphad = _mm256_set1_pd(alpha);
      if (aligned<32>(inout_x_ptr) && aligned<32>(in_y_ptr)) {
        for (int i = 0; i < (STRIP_SIZE>>2); i++) {
          __m256d x = _mm256_load_pd(inout_x_ptr+(i<<2));
          __m256d y = _mm256_load_pd(in_y_ptr+(i<<2));
          _mm256_store_pd(inout_x_ptr+(i<<2), _mm256_add_pd(x,_mm256_mul_pd(alphad,y)));
        }
      } else {
        for (int i = 0; i < (STRIP_SIZE>>2); i++) {
          __m256d x = _mm256_loadu_pd(inout_x_ptr+(i<<2));
          __m256d y = _mm256_loadu_pd(in_y_ptr+(i<<2));
          _mm256_storeu_pd(inout_x_ptr+(i<<2), _mm256_add_pd(x,_mm256_mul_pd(alphad,y)));
        }
      }
#elif defined(__SSE2__)
      __m128d alphad = _mm_set1_pd(alpha);
      if (aligned<16>(inout_x_ptr) && aligned<16>(in_y_ptr)) {
        for (int i = 0; i < (STRIP_SIZE>>1); i++) {
          __m128d x = _mm_load_pd(inout_x_ptr+(i<<1));
          __m128d y = _mm_load_pd(in_y_ptr+(i<<1));
          _mm_store_pd(inout_x_ptr+(i<<1), _mm_add_pd(x,_mm_mul_pd(alphad,y)));
        }
      } else {
        for (int i = 0; i < (STRIP_SIZE>>1); i++) {
          __m128d x = _mm_loadu_pd(inout_x_ptr+(i<<1));
          __m128d y = _mm_loadu_pd(in_y_ptr+(i<<1));
          _mm_storeu_pd(inout_x_ptr+(i<<1), _mm_add_pd(x,_mm_mul_pd(alphad,y)));
        }
      }
#else
      for (int i = 0; i < STRIP_SIZE; i++)
        inout_x_ptr[i] += (alpha * in_y_ptr[i]);
#endif
      n_pts -= STRIP_SIZE;
      inout_x_ptr += STRIP_SIZE;
      in_y_ptr += STRIP_SIZE;
    } else {
      for (int i = 0; i < n_pts; i++)
        inout_x_ptr[i] += (alpha * in_y_ptr[i]);
      n_pts = 0;
    }
  }
#ifdef __AVX__
  _mm256_zeroall();
#endif
#undef STRIP_SIZE
  return true;
}

template<>
void add_inplace_task<double>(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, HighLevelRuntime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->futures.size() == 1);

  const TaskArgs2<double> task_args = *((const TaskArgs2<double>*)task->args);

  double alpha;
  Future dummy = task->futures[0];
  alpha = dummy.get_result<double>();

  RegionAccessor<AccessorType::Generic, double> acc_x =
  regions[0].get_field_accessor(task_args.x_fid).typeify<double>();

  RegionAccessor<AccessorType::Generic, double> acc_y =
  regions[1].get_field_accessor(task_args.y_fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
                   task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  if (dense_add_inplace(rect, alpha, acc_x, acc_y))
    return;

  GenericPointInRectIterator<1> itr(rect);
  const int volume = rect.volume();
  for (int i = 0; i < volume; i++) {
    double temp = acc_x.read(DomainPoint::from_point<1>(itr.p)) +
              alpha * acc_y.read(DomainPoint::from_point<1>(itr.p));
    acc_x.write(DomainPoint::from_point<1>(itr.p), temp);
    itr++;
  }
}

template<typename T>
void axpy_inplace(const Array<T> &x, Array<T> &y, Future coef, 
                  const Predicate &pred, Context ctx, HighLevelRuntime *runtime){
        ArgumentMap arg_map;

	TaskArgs2<T> add_args(x, y);

        IndexLauncher add_launcher(AXPY_INPLACE_TASK_ID, x.color_domain,
                                    TaskArgument(&add_args, sizeof(add_args)), 
                                    arg_map, pred);

	add_launcher.add_future(coef);

        add_launcher.add_region_requirement(
                        RegionRequirement(x.lp, 0, READ_ONLY, EXCLUSIVE, x.lr));
        add_launcher.region_requirements[0].add_field(x.fid);

        add_launcher.add_region_requirement(
                        RegionRequirement(y.lp, 0, READ_WRITE, EXCLUSIVE, y.lr));
        add_launcher.region_requirements[1].add_field(y.fid);

        runtime->execute_index_space(ctx, add_launcher);

	return;

}

template<typename T>
void axpy_inplace_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, HighLevelRuntime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->futures.size() == 1);

  const TaskArgs2<T> task_args = *((const TaskArgs2<T>*)task->args);

  T alpha;
  Future dummy = task->futures[0];
  alpha = dummy.template get_result<T>();

  RegionAccessor<AccessorType::Generic, T> acc_x =
  regions[0].get_field_accessor(task_args.x_fid).template typeify<T>();

  RegionAccessor<AccessorType::Generic, T> acc_y =
  regions[1].get_field_accessor(task_args.y_fid).template typeify<T>();

  Domain dom = runtime->get_index_space_domain(ctx,
                   task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  GenericPointInRectIterator<1> itr(rect);
  const int volume = rect.volume();
  for (int i = 0; i < volume; i++) {
    T temp = acc_x.read(DomainPoint::from_point<1>(itr.p)) +
              alpha * acc_y.read(DomainPoint::from_point<1>(itr.p));
    acc_y.write(DomainPoint::from_point<1>(itr.p), temp);
    itr++;
  }
}

static bool dense_axpy_inplace(const Rect<1> &subgrid_bounds, double alpha,
                               RegionAccessor<AccessorType::Generic,double> &fa_x,
                               RegionAccessor<AccessorType::Generic,double> &fa_y)
{
  Rect<1> subrect;
  ByteOffset in_offsets[1], offsets[1];

  const double *in_x_ptr = fa_x.raw_rect_ptr<1>(subgrid_bounds, subrect, in_offsets);
  if (!in_x_ptr || (subrect != subgrid_bounds) ||
      !offsets_are_dense<1,double>(subgrid_bounds, in_offsets)) return false;

  double *inout_y_ptr = fa_y.raw_rect_ptr<1>(subgrid_bounds, subrect, offsets);
  if (!inout_y_ptr || (subrect != subgrid_bounds) || 
      offset_mismatch(1, in_offsets, offsets)) return false;

  int n_pts = subgrid_bounds.volume();
#define STRIP_SIZE 256
  while (n_pts > 0) {
    if (n_pts >= STRIP_SIZE) {
#if defined(__AVX__)
      __m256d alphad = _mm256_set1_pd(alpha);
      if (aligned<32>(in_x_ptr) && aligned<32>(inout_y_ptr)) {
        for (int i = 0; i < (STRIP_SIZE>>2); i++) {
          __m256d x = _mm256_load_pd(in_x_ptr+(i<<2));
          __m256d y = _mm256_load_pd(inout_y_ptr+(i<<2));
          _mm256_store_pd(inout_y_ptr+(i<<2), _mm256_add_pd(x,_mm256_mul_pd(alphad,y)));
        }
      } else {
        for (int i = 0; i < (STRIP_SIZE>>2); i++) {
          __m256d x = _mm256_loadu_pd(in_x_ptr+(i<<2));
          __m256d y = _mm256_loadu_pd(inout_y_ptr+(i<<2));
          _mm256_storeu_pd(inout_y_ptr+(i<<2), _mm256_add_pd(x,_mm256_mul_pd(alphad,y)));
        }
      }
#elif defined(__SSE2__)
      __m128d alphad = _mm_set1_pd(alpha);
      if (aligned<16>(in_x_ptr) && aligned<16>(inout_y_ptr)) {
        for (int i = 0; i < (STRIP_SIZE>>1); i++) {
          __m128d x = _mm_load_pd(in_x_ptr+(i<<1));
          __m128d y = _mm_load_pd(inout_y_ptr+(i<<1));
          _mm_stream_pd(inout_y_ptr+(i<<1), _mm_add_pd(x,_mm_mul_pd(alphad,y)));
        }
      } else {
        for (int i = 0; i < (STRIP_SIZE>>1); i++) {
          __m128d x = _mm_loadu_pd(in_x_ptr+(i<<1));
          __m128d y = _mm_loadu_pd(inout_y_ptr+(i<<1));
          _mm_storeu_pd(inout_y_ptr+(i<<1), _mm_add_pd(x,_mm_mul_pd(alphad,y)));
        }
      }
#else
      for (int i = 0; i < STRIP_SIZE; i++)
        inout_y_ptr[i] = in_x_ptr[i] + alpha * inout_y_ptr[i];
#endif
      n_pts -= STRIP_SIZE;
      in_x_ptr += STRIP_SIZE;
      inout_y_ptr += STRIP_SIZE;
    } else {
      for (int i = 0; i < n_pts; i++)
        inout_y_ptr[i] = in_x_ptr[i] + alpha * inout_y_ptr[i];
      n_pts = 0;
    }
  }
#ifdef __AVX__
  _mm256_zeroall();
#endif
#undef STRIP_SIZE
  return true;
}

template<>
void axpy_inplace_task<double>(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, HighLevelRuntime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->futures.size() == 1);

  const TaskArgs2<double> task_args = *((const TaskArgs2<double>*)task->args);

  double alpha;
  Future dummy = task->futures[0];
  alpha = dummy.get_result<double>();

  RegionAccessor<AccessorType::Generic, double> acc_x =
  regions[0].get_field_accessor(task_args.x_fid).typeify<double>();

  RegionAccessor<AccessorType::Generic, double> acc_y =
  regions[1].get_field_accessor(task_args.y_fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
                   task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  if (dense_axpy_inplace(rect, alpha, acc_x, acc_y))
    return;

  GenericPointInRectIterator<1> itr(rect);
  const int volume = rect.volume();
  for (int i = 0; i < volume; i++) {
    double temp = acc_x.read(DomainPoint::from_point<1>(itr.p)) +
              alpha * acc_y.read(DomainPoint::from_point<1>(itr.p));
    acc_y.write(DomainPoint::from_point<1>(itr.p), temp);
    itr++;
  }
}

template<typename T>
T L2norm(Array<T> &x, Context ctx, HighLevelRuntime *runtime){

	Future norm = dot(x, x, Predicate::TRUE_PRED, Future(), ctx, runtime);
	return(sqrt(norm.get_result<T>()));
}

template<typename T>
Future compute_scalar(Future x1, Future x2, const Predicate &pred,
                      const Future &false_result, Context ctx, HighLevelRuntime *runtime) {

	TaskLauncher divide(DIVIDE_TASK_ID, TaskArgument(NULL, 0), pred);
	divide.add_future(x1);
	divide.add_future(x2);
        divide.set_predicate_false_future(false_result);
	Future result = runtime->execute_task(ctx, divide);

	return(result);
}

template<typename T>
T compute_scalar_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
               	      Context ctx, HighLevelRuntime *runtime){

	assert(task->futures.size() == 2);

	Future x1 = task->futures[0];
	T r1 = x1.get_result<T>();

	Future x2 = task->futures[1];
	T r2 = x2.get_result<T>();

	// we can add some check later, that the denominator is not close to zero.
	
	return (r1 / r2);
}

template<typename T>
Predicate test_convergence(Future norm, T norm0, T threshold, 
                           const Predicate &pred, Context ctx, HighLevelRuntime *runtime)
{
  T args[2];
  args[0] = norm0;
  args[1] = threshold;
  TaskLauncher convergence(CONVERGENCE_TASK_ID,
                           TaskArgument(args, sizeof(args)), pred);
  // If we end up predicated false, then we have converged
  bool false_result = true;
  convergence.set_predicate_false_result(TaskArgument(&false_result, sizeof(false_result)));
  convergence.add_future(norm);
  Future converged = runtime->execute_task(ctx, convergence);
  Predicate conv_pred = runtime->create_predicate(ctx, converged);
  // Return the negated predicate since we only 
  // want to continue executing if we haven't converged
  // And it together with previous predicate to handle the
  // case where we've already converged
  return runtime->predicate_and(ctx, pred, runtime->predicate_not(ctx, conv_pred));
}

template<typename T>
bool test_convergence_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, HighLevelRuntime *runtime)
{
  assert(task->arglen == 2*sizeof(T));
  assert(task->futures.size() == 1);
  Future dummy = task->futures[0];
  T norm = dummy.template get_result<T>();

  const T *args = (const T*)task->args;
  T norm0 = args[0];
  T threshold = args[1];

  // Do the comparison, return true if we've converged
  return ((sqrt(norm)/norm0) < threshold);
}

template<typename T>
static void RegisterOperatorTasks(void) {

	HighLevelRuntime::register_legion_task<spmv_task<T> >(SPMV_TASK_ID, 
	Processor::LOC_PROC, true/*single*/, true/*index*/,
	AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "spmv");

	HighLevelRuntime::register_legion_task<subtract_task<T> >(SUBTRACT_TASK_ID,
        Processor::LOC_PROC, true/*single*/, true/*index*/,
        AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "subtract");

        HighLevelRuntime::register_legion_task<subtract_inplace_task<T> >(SUBTRACT_INPLACE_TASK_ID,
        Processor::LOC_PROC, true/*single*/, true/*index*/,
        AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "subtract_inplace");

	HighLevelRuntime::register_legion_task<T, dot_task<T> >(DOT_TASK_ID,
        Processor::LOC_PROC, true/*single*/, true/*index*/, 
        AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "dotproduct");

	HighLevelRuntime::register_reduction_op<FutureSum>(REDUCE_ID);

	HighLevelRuntime::register_legion_task<add_task<T> >(ADD_TASK_ID,
        Processor::LOC_PROC, true/*single*/, true/*index*/,
        AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "add");

        HighLevelRuntime::register_legion_task<add_inplace_task<T> >(ADD_INPLACE_TASK_ID,
        Processor::LOC_PROC, true/*single*/, true/*index*/,
        AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "add_inplace");

        HighLevelRuntime::register_legion_task<axpy_inplace_task<T> >(AXPY_INPLACE_TASK_ID,
        Processor::LOC_PROC, true/*single*/, true/*index*/,
        AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "axpy_inplace");

	HighLevelRuntime::register_legion_task<T, compute_scalar_task<T> >(DIVIDE_TASK_ID,
        Processor::LOC_PROC, true/*single*/, true/*index*/, 
        AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "compute_scalar");

        HighLevelRuntime::register_legion_task<bool, test_convergence_task<T> >(CONVERGENCE_TASK_ID,
        Processor::LOC_PROC, true/*single*/, true/*index*/,
        AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "test_convergence");

	return;
}
#endif
