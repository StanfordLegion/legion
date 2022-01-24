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
#include <unistd.h>

#include <legion.h>
#include <realm/cmdline.h>
#include <mappers/default_mapper.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>
#ifdef USE_KOKKOS_KERNELS
#include <KokkosBlas.hpp>
#endif

#include <typeinfo>

using namespace Legion;

Logger log_app("app");

enum {
  TOP_LEVEL_TASK_ID,
  INIT_TASK_ID,
  SAXPY_TASK_ID,
  SDOT_TASK_ID
};

enum {
  FID_X = 100,
  FID_Y,
};

struct SaxpyTaskArgs {
  float alpha;
};

typedef FieldAccessor<READ_ONLY, float, 1, coord_t,
		      Realm::AffineAccessor<float, 1, coord_t> > AccessorRO;
typedef FieldAccessor<READ_WRITE, float, 1, coord_t,
		      Realm::AffineAccessor<float, 1, coord_t> > AccessorRW;

// we'll do saxpy with a functor and sdot below with a lambda
template <class execution_space>
class SaxpyFunctor {
public:
  float alpha;
  // Legion uses a much richer set of layout types than just Kokkos'
  //  LayoutLeft and LayoutRight, and those choices are made by the mapper,
  //  not the application code, so Kokkos Views (or OffsetViews) that are
  //  referring to instances of logical regions will generally want to use
  //  LayoutStride for maximum flexibility
  Kokkos::Experimental::OffsetView<const float *,
				   Kokkos::LayoutStride,
				   execution_space> x;
  Kokkos::Experimental::OffsetView<float *,
				   Kokkos::LayoutStride,
				   execution_space> y;

  SaxpyFunctor(float _alpha,
	       Kokkos::Experimental::OffsetView<const float *,
	                                        Kokkos::LayoutStride,
	                                        execution_space> _x,
	       Kokkos::Experimental::OffsetView<float *,
	                                        Kokkos::LayoutStride,
	                                        execution_space> _y)
    : alpha(_alpha), x(_x), y(_y) {}

  KOKKOS_FUNCTION void operator()(int i) const
  {
    y(i) += alpha * x(i);
  }
};

// we have to wrap our tasks in classes so that we can use them as template
//  template parameters below (function templates cannot be template template
//  parameters)
template <typename execution_space>
class SaxpyTask {
public:
  static void task_body(const Task *task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, Runtime *runtime)
  {
    printf("kokkos(%s) saxpy task on processor " IDFMT ", kind %d\n",
           typeid(execution_space).name(),
           runtime->get_executing_processor(ctx).id,
           runtime->get_executing_processor(ctx).kind());
    const SaxpyTaskArgs& args = *reinterpret_cast<const SaxpyTaskArgs *>(task->args);
    AccessorRO acc_x(regions[0], FID_X);
    AccessorRW acc_y(regions[1], FID_Y);
  
    Rect<1> subspace = runtime->get_index_space_domain(ctx,
						       task->regions[0].region.get_index_space());

#ifdef USE_KOKKOS_KERNELS
    // only do half the child tasks with kokkos-kernels because we want to
    //  test application-supplied kernels too
    if((task->index_point[0] % 2) == 0) {
      // KokkosBlas uses views and the relative indexing they imply, so
      //  we can ask the accessors to convert directly to views
      // NOTE: the indexing is relative to the base of the index space on
      //  which privileges were requested - if that's not the same as the
      //  subspace you want KokkosBlas to iterate over, you'll need to pass
      //  in appropriate subviews
      Kokkos::View<const float *,
		   Kokkos::LayoutStride,
		   typename execution_space::memory_space> x = acc_x.accessor;
      Kokkos::View<float *,
		   Kokkos::LayoutStride,
		   typename execution_space::memory_space> y = acc_y.accessor;

      KokkosBlas::axpy(args.alpha, x, y);
    } else
#endif
    {
      Kokkos::Experimental::OffsetView<const float *,
				       Kokkos::LayoutStride,
				       typename execution_space::memory_space> x_ofs = acc_x.accessor;
      Kokkos::Experimental::OffsetView<float *,
				       Kokkos::LayoutStride,
				       typename execution_space::memory_space> y_ofs = acc_y.accessor;

      Kokkos::RangePolicy<execution_space> range(runtime->get_executing_processor(ctx).kokkos_work_space(),
						 subspace.lo.x,
						 subspace.hi.x + 1);
      Kokkos::parallel_for(range,
			   SaxpyFunctor<execution_space>(args.alpha, x_ofs, y_ofs));
    }
  }
};

template <typename execution_space>
class SdotTask {
public:
  static float task_body(const Task *task,
			 const std::vector<PhysicalRegion> &regions,
			 Context ctx, Runtime *runtime)
  {
    printf("kokkos(%s) sdot task on processor " IDFMT ", kind %d\n",
           typeid(execution_space).name(),
           runtime->get_executing_processor(ctx).id,
           runtime->get_executing_processor(ctx).kind());
    Rect<1> subspace = runtime->get_index_space_domain(ctx,
						       task->regions[0].region.get_index_space());

    AccessorRO acc_x(regions[0], task->regions[0].instance_fields[0]);
    AccessorRO acc_y(regions[1], task->regions[1].instance_fields[0]);

    Kokkos::Experimental::OffsetView<const float *,
				     Kokkos::LayoutStride,
				     typename execution_space::memory_space> x = acc_x.accessor;
    Kokkos::Experimental::OffsetView<const float *,
				     Kokkos::LayoutStride,
				     typename execution_space::memory_space> y = acc_y.accessor;

#ifdef USE_KOKKOS_KERNELS
    // only do half the child tasks with kokkos-kernels because we want to
    //  test application-supplied kernels too
    if((task->index_point[0] % 2) == 0) {
      // we've constructed absolutely-indexed OffsetViews above, but
      //  KokkosBlas wants relatively-indexed Views, and there are
      //  two ways of converting OffsetViews to Views:
      Kokkos::View<const float *,
		   Kokkos::LayoutStride,
		   typename execution_space::memory_space> x_rel, y_rel;

      // option 1: if you're sure the OffsetView starts in the right place
      //   (i.e. the subspace on which you have privileges matches what
      //   KokkosBlas is going to compute over), you can just use
      //   OffsetView::view() to convert
      assert(x.begin(0) == subspace.lo.x);
      x_rel = x.view();

      // option 2: if you're not sure what the OffsetView's bounds are
      //   (or if you just like more self-documenting code) you can create
      //   the subview with the exact bounds you want and then convert that
      y_rel = Kokkos::Experimental::subview(y,
					    std::make_pair(subspace.lo.x,
							   subspace.hi.x + 1))
	.view();

      // the KokkosBlas::dot implementation that returns a float directly
      //  performs a fence on all execution spaces, which is not permitted by
      //  default in Legion - see:
      //     https://github.com/kokkos/kokkos-kernels/issues/757
      //
      // instead, use the variant that fills a (managed) view and explicitly
      //  copy back to a host mirror
      float result_host;
      {
	Kokkos::View<float,
		     typename execution_space::memory_space> result("result");
	KokkosBlas::dot(result, x_rel, y_rel);

	// can't use `kokkos_work_space` here because KokkosBlas::dot didn't
	Kokkos::deep_copy(execution_space(), result_host, result);
      }
      execution_space().fence();
      return result_host;
    }
#endif
    Kokkos::RangePolicy<execution_space> range(runtime->get_executing_processor(ctx).kokkos_work_space(),
					       subspace.lo.x,
					       subspace.hi.x + 1);
    float sum = 0.0f;
    // Kokkos does not support CUDA lambdas by default - check that they
    //  are present
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_CUDA_LAMBDA)
    #error Kokkos built without --with-cuda_options=enable_lambda !
#endif
    Kokkos::parallel_reduce(range,
			    KOKKOS_LAMBDA ( int j, float &update ) {
			      update += x(j) * y(j);
			    }, sum);
    return sum;
  }
};

template <typename execution_space>
class InitTask {
public:
  static void task_body(const Task *task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, Runtime *runtime)
  {
    printf("kokkos(%s) init task on processor " IDFMT ", kind %d\n",
           typeid(execution_space).name(),
           runtime->get_executing_processor(ctx).id,
           runtime->get_executing_processor(ctx).kind());

    const float offset = *reinterpret_cast<const float *>(task->args);

    Rect<1> subspace = runtime->get_index_space_domain(ctx,
						       task->regions[0].region.get_index_space());

    AccessorRW acc(regions[0], task->regions[0].instance_fields[0]);

    // you can use relative indexing for your own kernels too - just make
    //  sure you do the right thing when operating on a partitioned
    //  subregion!
    Kokkos::View<float *,
		 Kokkos::LayoutStride,
		 typename execution_space::memory_space> view = acc.accessor;

    size_t n_elements = subspace.hi.x - subspace.lo.x + 1;
    Kokkos::RangePolicy<execution_space> range(runtime->get_executing_processor(ctx).kokkos_work_space(),
					       0, n_elements);
    Kokkos::parallel_for(range,
			 KOKKOS_LAMBDA (int i) {
			   // using a relative address, but value to store
			   //  is based on global index
			   // have to use a relative address!
			   view(i) = (i + subspace.lo.x) + offset;
			 });
  }
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  size_t num_elements = 32768;
  size_t num_pieces = 4;
  int mixed_processors = 1;

  const InputArgs &command_args = Runtime::get_input_args();

  bool ok = Realm::CommandLineParser()
    .add_option_int("-n", num_elements)
    .add_option_int("-p", num_pieces)
    .add_option_int("-mixed", mixed_processors)
    .parse_command_line(command_args.argc, (const char **)command_args.argv);

  if(!ok) {
    log_app.fatal() << "error parsing command line arguments";
    exit(1);
  }

  Rect<1> r_top = Rect<1>(0, num_elements - 1);
  IndexSpace is_top = runtime->create_index_space(ctx, r_top);

  IndexSpace launch_space = runtime->create_index_space(ctx,
							Rect<1>(1, num_pieces));
  IndexPartition ip_dist = runtime->create_equal_partition(ctx, is_top,
							   launch_space);

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(float), FID_X);
    fa.allocate_field(sizeof(float), FID_Y);
  }

  LogicalRegion lr = runtime->create_logical_region(ctx, is_top, fs);
  LogicalPartition lp_dist = runtime->get_logical_partition(ctx, lr, ip_dist);

  // initial values for x and y - don't use simple constant fills because we
  //  want to detect indexing errors with absolute vs. relative indexing
  {
    float offset = 1.0;
    IndexTaskLauncher itl(INIT_TASK_ID, launch_space,
			  TaskArgument(&offset, sizeof(offset)),
			  ArgumentMap());
    itl.add_region_requirement(RegionRequirement(lp_dist,
						 0 /*IDENTITY PROJECTION*/,
						 WRITE_DISCARD, EXCLUSIVE, lr)
			       .add_field(FID_X));
    runtime->execute_index_space(ctx, itl);
  }
  {
    float offset = 2.0;
    IndexTaskLauncher itl(INIT_TASK_ID, launch_space,
			  TaskArgument(&offset, sizeof(offset)),
			  ArgumentMap());
    itl.add_region_requirement(RegionRequirement(lp_dist,
						 0 /*IDENTITY PROJECTION*/,
						 WRITE_DISCARD, EXCLUSIVE, lr)
			       .add_field(FID_Y));
    runtime->execute_index_space(ctx, itl);
  }

  // index launch to compute saxpy on chunks of vector in parallel
  {
    SaxpyTaskArgs args;
    args.alpha = 0.5f;
    
    IndexTaskLauncher itl(SAXPY_TASK_ID, launch_space,
			  TaskArgument(&args, sizeof(args)),
			  ArgumentMap());
    itl.add_region_requirement(RegionRequirement(lp_dist,
						 0 /*IDENTITY PROJECTION*/,
						 READ_ONLY, EXCLUSIVE, lr)
			       .add_field(FID_X));
    itl.add_region_requirement(RegionRequirement(lp_dist,
						 0 /*IDENTITY PROJECTION*/,
						 READ_WRITE, EXCLUSIVE, lr)
			       .add_field(FID_Y));
    runtime->execute_index_space(ctx, itl);
  }

  // second index launch to compute a dot product on the result
  Future f_sum;
  {
    IndexTaskLauncher itl(SDOT_TASK_ID, launch_space,
			  TaskArgument(),
			  ArgumentMap());

    // prefer CPU variants here so that we try mixing and matching CPU
    //  and GPU tasks when both are available
    if(mixed_processors)
      itl.tag |= Legion::Mapping::DefaultMapper::PREFER_CPU_VARIANT;

    itl.add_region_requirement(RegionRequirement(lp_dist,
						 0 /*IDENTITY PROJECTION*/,
						 READ_ONLY, EXCLUSIVE, lr)
			       .add_field(FID_X));
    itl.add_region_requirement(RegionRequirement(lp_dist,
						 0 /*IDENTITY PROJECTION*/,
						 READ_ONLY, EXCLUSIVE, lr)
			       .add_field(FID_Y));
    f_sum = runtime->execute_index_space(ctx, itl,
					 LEGION_REDOP_SUM_FLOAT32);
  }
  float act_sum = f_sum.get_result<float>(true /*silence_warnings*/);
  // there's probably a closed-form version of this...
  float exp_sum = 0;
  for(size_t i = 0; i < num_elements; i++)
    exp_sum += (i + 1.0) * ((i + 2.0) + 0.5 * (i + 1.0));

  printf("got %g, exp %g\n", act_sum, exp_sum);

  // floating point math isn't exact, but if we're off by more than 10%,
  //  something's probably wrong
  if(fabsf(act_sum - exp_sum) < (0.1 * exp_sum)) {
    // close enough
  } else {
    printf("results definitely do not match!\n");
    runtime->set_return_code(1);
  }

  // tear stuff down
  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_index_partition(ctx, ip_dist);
  runtime->destroy_index_space(ctx, is_top);
  runtime->destroy_field_space(ctx, fs);
}

template <template<typename> class PORTABLE_KOKKOS_TASK>
void preregister_kokkos_task(TaskID task_id, const char *name)
{
#ifdef KOKKOS_ENABLE_SERIAL
  // register a serial version on the CPU
  {
    TaskVariantRegistrar registrar(task_id, name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<
      PORTABLE_KOKKOS_TASK<Kokkos::Serial>::task_body >(registrar, name);
  }
#endif

  // register an openmp version, if available
#ifdef KOKKOS_ENABLE_OPENMP
  {
    TaskVariantRegistrar registrar(task_id, name);
#ifdef REALM_USE_OPENMP
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
#else
    // if realm is not openmp-aware, put this on normal cpu cores and
    //  hope for the best
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
#endif
    Runtime::preregister_task_variant<
      PORTABLE_KOKKOS_TASK<Kokkos::OpenMP>::task_body >(registrar, name);
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA) and defined(REALM_USE_CUDA)
  // register a serial version on the CPU
  {
    TaskVariantRegistrar registrar(task_id, name);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<
      PORTABLE_KOKKOS_TASK<Kokkos::Cuda>::task_body >(registrar, name);
  }
#endif
}

// for tasks with a non-void return value
template <typename RV, template<typename> class PORTABLE_KOKKOS_TASK>
void preregister_kokkos_task(TaskID task_id, const char *name)
{
#ifdef KOKKOS_ENABLE_SERIAL
  // register a serial version on the CPU
  {
    TaskVariantRegistrar registrar(task_id, name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<
      RV, PORTABLE_KOKKOS_TASK<Kokkos::Serial>::task_body >(registrar, name);
  }
#endif

  // register an openmp version, if available
#ifdef KOKKOS_ENABLE_OPENMP
  {
    TaskVariantRegistrar registrar(task_id, name);
#ifdef REALM_USE_OPENMP
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
#else
    // if realm is not openmp-aware, put this on normal cpu cores and
    //  hope for the best
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
#endif
    Runtime::preregister_task_variant<
      RV, PORTABLE_KOKKOS_TASK<Kokkos::OpenMP>::task_body >(registrar, name);
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA) and defined(REALM_USE_CUDA)
  // register a serial version on the CPU
  {
    TaskVariantRegistrar registrar(task_id, name);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<
      RV, PORTABLE_KOKKOS_TASK<Kokkos::Cuda>::task_body >(registrar, name);
  }
#endif
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  preregister_kokkos_task<InitTask>(INIT_TASK_ID, "init");
  preregister_kokkos_task<SaxpyTask>(SAXPY_TASK_ID, "saxpy");
  preregister_kokkos_task<float, SdotTask>(SDOT_TASK_ID, "sdot");

  return Runtime::start(argc, argv);
}

