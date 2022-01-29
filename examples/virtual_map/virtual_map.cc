/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// This example demonstrates way in which virtual mapping can be useful -
// generating variable-sized output from a task

#include <legion.h>
#include <realm/cmdline.h>
#include <mappers/default_mapper.h>

using namespace Legion;
using namespace Legion::Mapping;

enum TaskID
{
  TOP_LEVEL_TASK_ID,
  MAKE_DATA_TASK_ID,
  USE_DATA_TASK_ID,
};

enum {
  FID_DATA = 55,
};

enum {
  PID_ALLOCED_DATA = 4,
};

enum {
  PFID_USE_DATA_TASK = 77,
};

Logger log_app("app");

struct MakeDataTaskArgs {
  int random_seed;
  size_t average_output_size;
};

void top_level_task(const Task *task, 
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  size_t num_subregions = 4;
  size_t average_output_size = 100;
  size_t total_region_size = 1 << 30;
  int random_seed = 12345;
  bool use_projection_functor = false;

  const InputArgs &command_args = Runtime::get_input_args();

  bool ok = Realm::CommandLineParser()
    .add_option_int("-n", num_subregions)
    .add_option_int("-s", average_output_size)
    .add_option_int("-l", total_region_size)
    .add_option_int("-seed", random_seed)
    .add_option_bool("-proj", use_projection_functor)
    .parse_command_line(command_args.argc, (const char **)command_args.argv);

  if(!ok) {
    log_app.fatal() << "error parsing command line arguments";
    exit(1);
  }

  // create a top-level region that is (easily) large enough to hold all of our
  // data - we need not worry about a tight bound, because we will never create
  // any instances of this size
  Rect<1> bounds(0, total_region_size - 1);
  IndexSpaceT<1> is = runtime->create_index_space(ctx, bounds);
  runtime->attach_name(is, "is");

  // our field space will just have a single field
  FieldSpace fs = runtime->create_field_space(ctx);
  runtime->attach_name(fs, "fs");
  {
    FieldAllocator fsa = runtime->create_field_allocator(ctx, fs);
    fsa.allocate_field(sizeof(int), FID_DATA);
    runtime->attach_name(fs, FID_DATA, "data");
  }

  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
  runtime->attach_name(lr, "lr");

  // partition it into equal pieces - again, these are expected to be much
  // larger than the actual output of the generator tasks
  Rect<1> subregion_idxs(0, num_subregions - 1);
  size_t elements_per_subregion = total_region_size / num_subregions;
  assert(elements_per_subregion >= (2 * average_output_size));
  IndexPartition ip = runtime->create_partition_by_blockify(ctx, is, 
                                    Point<1>(elements_per_subregion));
  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);

  // perform an index launch to have each subregion filled in with a variable
  // amount of data
  {
    MakeDataTaskArgs args;
    args.random_seed = random_seed;
    args.average_output_size = average_output_size;
    IndexLauncher launcher(MAKE_DATA_TASK_ID, subregion_idxs,
			   TaskArgument(&args, sizeof(args)),
			   ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(lp,
						      0, // identity projection
						      WRITE_DISCARD,
						      EXCLUSIVE,
						      lr,
						      DefaultMapper::VIRTUAL_MAP)
				    .add_field(FID_DATA));
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();
  }

  // now perform a second index launch to use the variable-sized chunks of
  // data
  {
    IndexLauncher launcher(USE_DATA_TASK_ID, subregion_idxs,
			   TaskArgument(0, 0),
			   ArgumentMap());
    if(use_projection_functor) {
      // use a projection functor to tell the runtime precisely which sub-subregion
      // each point task will use so that it can be mapped before the task executes -
      // make sure the default mapper doesn't try to map the parent region though
      launcher.add_region_requirement(RegionRequirement(lp,
							PFID_USE_DATA_TASK,
							READ_ONLY,
							EXCLUSIVE,
							lr,
							DefaultMapper::EXACT_REGION)
				      .add_field(FID_DATA));
    } else {
      // OR, use another virtual mapping and let the use_data_task walk the region
      // tree itself and do an inline mapping within the task
      launcher.add_region_requirement(RegionRequirement(lp,
							0, // identity projection
							READ_ONLY,
							EXCLUSIVE,
							lr,
							DefaultMapper::VIRTUAL_MAP)
				      .add_field(FID_DATA));
    }
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();
  }
}

void make_data_task(const Task *task, 
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  const MakeDataTaskArgs& args = *static_cast<const MakeDataTaskArgs *>(task->args);
  int subregion_index = task->index_point.get_point<1>();

  // we should _NOT_ have a mapping of our region
  assert(!regions[0].is_mapped());

  // first, decide how much data we're going to produce
  unsigned short xsubi[3];
  xsubi[0] = args.random_seed >> 16;
  xsubi[1] = args.random_seed;
  xsubi[2] = subregion_index;
  coord_t output_size = nrand48(xsubi) % (2 * args.average_output_size + 1);

  log_app.print() << "make data: subregion " << subregion_index << ", output size = " << output_size;

  // now we can define a (sub-)subregion of our subregion that is the right size
  // to hold the output
  LogicalRegion my_lr = regions[0].get_logical_region();
  IndexSpaceT<1> my_is(my_lr.get_index_space());
  Rect<1> my_bounds = Domain(runtime->get_index_space_domain(my_is));
  coord_t bounds_lo = my_bounds.lo[0];

  // remember rectangle bounds are inclusive on both sides
  Rect<1> alloc_bounds(my_bounds.lo, bounds_lo + output_size - 1);

  // create a new partition with a known part_color so that other tasks can find
  // the color space will consist of the single color '0'
  IndexSpaceT<1> color_space = runtime->create_index_space(ctx, Rect<1>(0, 0));
  Transform<1,1> transform;
  transform[0][0] = 0;
  IndexPartition alloc_ip = runtime->create_partition_by_restriction(ctx, my_is,
                                                                     color_space,
                                                                     transform,
                                                                     alloc_bounds,
                                                                     DISJOINT_KIND, // trivial
                                                                     PID_ALLOCED_DATA);

  // now we get the name of the logical subregion we just created and inline map
  // it to generate our output
  LogicalPartition alloc_lp = runtime->get_logical_partition(my_lr, alloc_ip);
  IndexSpace alloc_is = runtime->get_index_subspace(alloc_ip, DomainPoint::from_point<1>(0));
  LogicalRegion alloc_lr = runtime->get_logical_subregion(alloc_lp, alloc_is);

  log_app.debug() << "created subregion " << alloc_lr;

  // tell the default mapper that we want exactly this region to be mapped
  // otherwise, its heuristics may cause it to try to map the (huge) parent
  InlineLauncher launcher(RegionRequirement(alloc_lr,
					    WRITE_DISCARD,
					    EXCLUSIVE,
					    my_lr,
					    DefaultMapper::EXACT_REGION)
			  .add_field(FID_DATA));
  PhysicalRegion pr = runtime->map_region(ctx, launcher);

  // this would be done as part of asking for an accessor, but do it explicitly for
  // didactic purposes - while inline mappings can often be expensive due to stalls or 
  // data movement, this should be fast because we are just allocating new space
  pr.wait_until_valid();

  {
    const FieldAccessor<WRITE_DISCARD,int,1,coord_t,
            Realm::AffineAccessor<int,1,coord_t> > ra(pr, FID_DATA);

    for(coord_t i = 0; i < output_size; i++)
      ra[bounds_lo + i] = (subregion_index * 10000) + i;
  }

  // release our inline mapping
  runtime->unmap_region(ctx, pr);

  // no need to explicitly return the name of the subregion we created - it can be foud
  // by walking the region tree using known colors
}

void use_data_task(const Task *task, 
		   const std::vector<PhysicalRegion> &regions,
		   Context ctx, Runtime *runtime)
{
  int subregion_index = task->index_point.get_point<1>();

  PhysicalRegion pr = regions[0];
  Rect<1> my_bounds;
  bool was_virtual_mapped = !regions[0].is_mapped();
  if(was_virtual_mapped) {
    // find the actual subregion containing our input data by walking the
    // region tree
    LogicalRegion my_lr = regions[0].get_logical_region();
    IndexSpace my_is = my_lr.get_index_space();

    IndexPartition alloc_ip = runtime->get_index_partition(my_is, PID_ALLOCED_DATA);

    // getting from the index partition to the logical subregion is identical to 
    // what was done in make_data_task()
    LogicalPartition alloc_lp = runtime->get_logical_partition(my_lr, alloc_ip);
    IndexSpace alloc_is = runtime->get_index_subspace(alloc_ip, DomainPoint::from_point<1>(0));
    LogicalRegion alloc_lr = runtime->get_logical_subregion(alloc_lp, alloc_is);

    log_app.debug() << "looked up subregion " << alloc_lr;

    // learn the input size from the bounds of the allocated subspace
    my_bounds = runtime->get_index_space_domain(alloc_is);

    // again, tell the default mapper that we want exactly this region to be mapped
    // this is important if the mapper sends us to a different place than where the
    // data was produced and a copy is needed
    InlineLauncher launcher(RegionRequirement(alloc_lr,
					      READ_ONLY,
					      EXCLUSIVE,
					      my_lr,
					      DefaultMapper::EXACT_REGION)
			    .add_field(FID_DATA));
    pr = runtime->map_region(ctx, launcher);
  } else {
    // we've got the exact region mapped for us, so ask for its size
    LogicalRegion my_lr = regions[0].get_logical_region();
    IndexSpace my_is = my_lr.get_index_space();
    my_bounds = runtime->get_index_space_domain(my_is);
  }

  coord_t bounds_lo = my_bounds.lo[0];
  coord_t input_size = my_bounds.volume();

  log_app.print() << "use data: subregion " << subregion_index << ", input size = " << input_size
		  << (was_virtual_mapped ? " (virtual mapped)" : " (NOT virtual mapped)");


  // check that the data is what we expect
  int errors = 0;
  {
    const FieldAccessor<READ_ONLY,int,1,coord_t,
            Realm::AffineAccessor<int,1,coord_t> > ra(pr, FID_DATA);

    for(coord_t i = 0; i < input_size; i++) {
      int exp = (subregion_index * 10000) + i;
      int act = ra[bounds_lo + i];
      if(exp != act) {
	// don't print more than 10 errors
	if(errors < 10)
	  log_app.error() << "mismatch in subregion " << subregion_index << ": ["
			  << (bounds_lo + i) << "] = " << act << " (expected " << exp << ")";
	errors++;
      }
    }	       
  }

  // unmap any inline mapping we did
  if(was_virtual_mapped)
    runtime->unmap_region(ctx, pr);

  //return errors;
}

class UseDataProjectionFunctor : public ProjectionFunctor {
public:
  UseDataProjectionFunctor(void);

  using ProjectionFunctor::project;
  virtual LogicalRegion project(Context ctx, Task *task,
				unsigned index,
				LogicalRegion upper_bound,
				const DomainPoint &point);

  virtual LogicalRegion project(Context ctx, Task *task, 
				unsigned index,
				LogicalPartition upper_bound,
				const DomainPoint &point);

  virtual unsigned get_depth(void) const;
};

UseDataProjectionFunctor::UseDataProjectionFunctor(void) {}

// region -> region: UNUSED
LogicalRegion UseDataProjectionFunctor::project(Context ctx, Task *task,
						unsigned index,
						LogicalRegion upper_bound,
						const DomainPoint &point)
{
  assert(0);
  return LogicalRegion::NO_REGION;
}

// partition -> region path: [index].PID_ALLOCED_DATA.0
LogicalRegion UseDataProjectionFunctor::project(Context ctx, Task *task, 
						unsigned index,
						LogicalPartition upper_bound,
						const DomainPoint &point)
{
  // if our application did more than one index launch using this projection functor,
  // we could consider memoizing the result of this lookup to reduce overhead on subsequent
  // launches
  LogicalRegion lr1 = runtime->get_logical_subregion_by_color(ctx, upper_bound, point);
  LogicalPartition lp1 = runtime->get_logical_partition_by_color(ctx, lr1, PID_ALLOCED_DATA);
  LogicalRegion lr2 = runtime->get_logical_subregion_by_color(ctx, lp1, 0);
  return lr2;
}

unsigned UseDataProjectionFunctor::get_depth(void) const
{
  return 1;
}

int main(int argc, char **argv)
{
  // Register our task variants
  {
    TaskVariantRegistrar top_level_registrar(TOP_LEVEL_TASK_ID);
    top_level_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    top_level_registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(top_level_registrar, 
                                                      "Top Level Task");
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  }
  {
    TaskVariantRegistrar make_data_registrar(MAKE_DATA_TASK_ID);
    make_data_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<make_data_task>(make_data_registrar, 
                                                      "Make Data Task");
  }
  {
    TaskVariantRegistrar use_data_registrar(USE_DATA_TASK_ID);
    use_data_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<use_data_task>(use_data_registrar, 
						     "Use Data Task");
  }

  Runtime::preregister_projection_functor(PFID_USE_DATA_TASK,
					  new UseDataProjectionFunctor);

  // Fire up the runtime
  return Runtime::start(argc, argv);
}
