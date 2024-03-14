#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <math.h>
#include "legion.h"
#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

enum TaskIDs {
   TopLevelTask_ID,
   InitTask_ID,
   StencilTask_ID,
   CheckTask_ID,
};

enum FieldIDs {
   FID_x,
   FID_y,
};

template<typename FT, int N, typename T = coord_t> using AccessorRO = Legion::FieldAccessor< READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T> >;
template<typename FT, int N, typename T = coord_t> using AccessorRW = Legion::FieldAccessor<READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T> >;
template<typename FT, int N, typename T = coord_t> using AccessorWO = Legion::FieldAccessor<WRITE_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T> >;
// No privileges necessary on padded accessors since you're allowed to read and write on any padded space
template<typename FT, int N, typename T = coord_t> using AccessorPAD = Legion::PaddingAccessor<FT, N, T, Realm::AffineAccessor<FT, N, T> >;
template<typename FT, int N, typename T = coord_t> using MultiAccessorRO = Legion::FieldAccessor< READ_ONLY, FT, N, T, Realm::MultiAffineAccessor<FT, N, T> >;

// Number of points
static constexpr int Np = 200;
// Number of tiles
static constexpr int Nt = 1;

// utility functions
inline int warpIndex(int i) {
   i = i % Np;
   if (i < 0) i += Np;
   return i;
};

inline double warpX(const int i, double x) {
   x += floor(i/Np) * Np;
   if (i < 0) x -= Np;
   return x;
};


// Initialization task
void init_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
   assert(regions.size() == 1);

   const AccessorWO<   int, 1> acc_x(regions[0], FID_x);
   const AccessorWO<double, 1> acc_y(regions[0], FID_y);

   Rect<1> r = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());

#ifdef REALM_USE_OPENMP
   #pragma omp parallel for
#endif
   for (int k = r.lo; k <= r.hi; k++) acc_x[k] = k;

#ifdef REALM_USE_OPENMP
   #pragma omp parallel for
#endif
   for (int k = r.lo; k <= r.hi; k++) acc_y[k] = 0.0;

};

// Stencil task
void stencil_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);

   const MultiAccessorRO<int, 1>  acc_x_gh   (regions[0], FID_x);
   const AccessorRO<     int, 1>  acc_x      (regions[1], FID_x);
   const AccessorPAD<    int, 1>  acc_pad    (regions[1], FID_x);
   const AccessorWO<  double, 1>  acc_y      (regions[2], FID_y);

   Rect<1> r = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

   ///////////////////////////////////////////////////////////////////////////////////////////
   // assemble data
   acc_pad[r.lo - 1] = warpX(r.lo-1, acc_x_gh[warpIndex(r.lo-1)]);
   acc_pad[r.hi + 1] = warpX(r.hi+1, acc_x_gh[warpIndex(r.hi+1)]);

   // execute stencil calculation
#ifdef REALM_USE_OPENMP
    #pragma omp parallel for
#endif
   for (int k = r.lo; k <= r.hi; k++) acc_y[k] = 0.5*(acc_x[k+1] - acc_x[k-1]);
   ///////////////////////////////////////////////////////////////////////////////////////////
};

// Check task
void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
   assert(regions.size() == 1);

   const AccessorRO<double, 1> acc_y(regions[0], FID_y);

   Rect<1> r = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());

   for (int k = r.lo; k <= r.hi; k++) assert(acc_y[k] == 1.0);
   std::cout << "SUCCESS!!!" << std::endl;
};

// Top level task
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
   // Declare main index space
   IndexSpace fluid_is = runtime->create_index_space(ctx, Rect<1>(0, Np-1));
   runtime->attach_name(fluid_is, "Fluid_is");

   // Declare the field space
   FieldSpace fluid_fs = runtime->create_field_space(ctx);
   {
      FieldAllocator allocator = runtime->create_field_allocator(ctx, fluid_fs);
      FieldID fid1 = allocator.allocate_field(sizeof(   int), FID_x);
      assert(fid1 == FID_x);
      FieldID fid2 = allocator.allocate_field(sizeof(double), FID_y);
      assert(fid2 == FID_y);
   }

   // Declare the region
   LogicalRegion fluid = runtime->create_logical_region(ctx, fluid_is, fluid_fs);
   runtime->attach_name(fluid, "Fluid");

   // Declare partition index space
   IndexSpace tiles = runtime->create_index_space(ctx, Rect<1>(0, Nt-1));
   runtime->attach_name(fluid_is, "Fluid_is");

   // Make the partitions
   // - Main tiles
   LogicalPartition p_fluid = runtime->get_logical_partition(ctx, fluid,
                                 runtime->create_equal_partition(ctx, fluid_is, tiles));

   std::map<DomainPoint,Domain> domains;
   // - Upper stencil points
   for (int k = 0; k < Nt; k++) {
      IndexSpace tile_is = runtime->get_index_subspace(ctx, p_fluid.get_index_partition(), k);
      Rect<1> tile = runtime->get_index_space_domain(ctx, tile_is);
      // Upper bound
      int up = warpIndex(tile.hi+1);
      domains[k] = Rect<1>(up, up);
      //std::cout << k << " " << tile << " " << domains[k] << std::endl;
   }
   IndexPartition p_up = runtime->create_partition_by_domain(ctx, fluid_is, domains, tiles);

   // - Lower stencil points
   for (int k = 0; k < Nt; k++) {
      IndexSpace tile_is = runtime->get_index_subspace(ctx, p_fluid.get_index_partition(), k);
      Rect<1> tile = runtime->get_index_space_domain(ctx, tile_is);
      // Upper bound
      int lo = warpIndex(tile.lo-1);
      domains[k] = Rect<1>(lo, lo);
      //std::cout << k << " " << tile << " " << domains[k] << std::endl;
   }
   IndexPartition p_lo = runtime->create_partition_by_domain(ctx, fluid_is, domains, tiles);
   // - All stencil points
   LogicalPartition p_stencil = runtime->get_logical_partition(ctx, fluid,
                                     runtime->create_partition_by_union(ctx, fluid_is, p_lo, p_up, tiles));

   // Init the fields
   IndexTaskLauncher init_launcher(InitTask_ID, tiles, TaskArgument(NULL, 0), ArgumentMap());
   init_launcher.add_region_requirement(RegionRequirement(p_fluid, 0/*projection ID*/,
                                                          WRITE_DISCARD, EXCLUSIVE, fluid));
   init_launcher.region_requirements[0].add_field(FID_x);
   init_launcher.region_requirements[0].add_field(FID_y);
   runtime->execute_index_space(ctx, init_launcher);

   // Compute the stencil operation
   IndexTaskLauncher stencil_launcher(StencilTask_ID, tiles, TaskArgument(NULL, 0), ArgumentMap());
   stencil_launcher.add_region_requirement(RegionRequirement(p_stencil, 0/*projection ID*/,
                                                             READ_ONLY, EXCLUSIVE, fluid));
   stencil_launcher.add_region_requirement(RegionRequirement(p_fluid, 0/*projection ID*/,
                                                             READ_ONLY, EXCLUSIVE, fluid));
   stencil_launcher.add_region_requirement(RegionRequirement(p_fluid, 0/*projection ID*/,
                                                             WRITE_DISCARD, EXCLUSIVE, fluid));
   stencil_launcher.region_requirements[0].add_field(FID_x);
   stencil_launcher.region_requirements[1].add_field(FID_x);
   stencil_launcher.region_requirements[2].add_field(FID_y);
   runtime->execute_index_space(ctx, stencil_launcher);

   // Check result
   IndexTaskLauncher check_launcher(CheckTask_ID, tiles, TaskArgument(NULL, 0), ArgumentMap());
   check_launcher.add_region_requirement(RegionRequirement(p_fluid, 0/*projection ID*/,
                                                           READ_ONLY, EXCLUSIVE, fluid));
   check_launcher.region_requirements[0].add_field(FID_y);
   runtime->execute_index_space(ctx, check_launcher);

   // Clean up
   runtime->destroy_logical_region(ctx, fluid);
   runtime->destroy_field_space(ctx, fluid_fs);
   runtime->destroy_index_space(ctx, tiles);
   runtime->destroy_index_space(ctx, fluid_is);
};

class PaddingMapper : public DefaultMapper {
public:
  PaddingMapper(Machine m, Runtime *rt, Processor local)
    : DefaultMapper(rt->get_mapper_runtime(), m, local, "Padding Mapper") { }
public:
  void default_policy_select_constraints(MapperContext ctx,
      LayoutConstraintSet &constraints, Memory target, const RegionRequirement &req) override
  {
    DefaultMapper::default_policy_select_constraints(ctx, constraints, target, req);
    if ((req.privilege_fields.find(FID_x) != req.privilege_fields.end()) &&
        (constraints.specialized_constraint.kind != LEGION_COMPACT_SPECIALIZE))
      constraints.padding_constraint = PaddingConstraint(DomainPoint(1), DomainPoint(1));
  }
};

static void update_mappers(Machine machine, Runtime *runtime,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin(); it != local_procs.end(); it++)
    runtime->replace_default_mapper(new PaddingMapper(machine, runtime, *it));
}

int main(int argc, char **argv)
{

   // Register the specialized constraint for compact sparse instances (max 5 blocks with 10% maximum memory overhead)
   LayoutConstraintID CSI_const;
   {
      LayoutConstraintRegistrar registrar;
      registrar.add_constraint(SpecializedConstraint(/* kind =         */ LEGION_COMPACT_SPECIALIZE,
                                                     /* redop =        */ 0,
                                                     /* no_access =    */ false,
                                                     /* exact =        */ false,
                                                     /* max_pieces =   */ 2,
                                                     /* max_overhead = */ 1));
      CSI_const = Runtime::preregister_layout(registrar);
   }


   // Specialized constraint for affine instances
   LayoutConstraintID Aff_const;
   {
      LayoutConstraintRegistrar registrar;
      registrar.add_constraint(PaddingConstraint(DomainPoint(1), DomainPoint(1)));
      Aff_const = Runtime::preregister_layout(registrar);
   }

   {
      Runtime::set_top_level_task_id(TopLevelTask_ID);
      TaskVariantRegistrar registrar(TopLevelTask_ID, "top_level_variant");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_inner(true);
      registrar.set_replicable(true);
      Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
   }

   {
      TaskVariantRegistrar registrar(InitTask_ID, "initTask_variant");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf(true);
      Runtime::preregister_task_variant<init_task>(registrar, "init_task");
   }

   {
      TaskVariantRegistrar registrar(StencilTask_ID, "stencilTask_variant");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.add_layout_constraint_set(0, CSI_const);
      registrar.add_layout_constraint_set(1, Aff_const);
      registrar.set_leaf(true);
      Runtime::preregister_task_variant<stencil_task>(registrar, "stencil_task");
   }

   {
      TaskVariantRegistrar registrar(CheckTask_ID, "checkTask_variant");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf(true);
      Runtime::preregister_task_variant<check_task>(registrar, "check_task");
   }

   Runtime::add_registration_callback(update_mappers);

   return Runtime::start(argc, argv);
};
