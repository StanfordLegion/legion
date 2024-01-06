#include "legion.h"
#include "mappers/default_mapper.h"
#include "realm/cuda/cuda_access.h" // ExternalCudaMemoryResource
#include <cuda_runtime.h>

using namespace Legion;

#define CUDA_CHECK(call)                                        \
  do {                                                          \
    cudaError_t status = call;                                  \
    if (status != cudaSuccess) {                                \
      printf("CUDA error at %s:%d : '%s'\n",                    \
             __FILE__, __LINE__, cudaGetErrorString(status));   \
      exit(EXIT_FAILURE);                                       \
    }                                                           \
  } while (0)

enum TaskIDs {
  TID_TOP_LEVEL,
  TID_MAKE_FUTURE,
  TID_CHECK_FUTURE,
};

__global__
void write_result(int *ptr, int offset)
{
  *ptr = 42 + offset;
}

__global__
void check_result(const int *ptr, int points, bool *result)
{
  *result = (*ptr == ((points-1)*points/2 + 42*points)); 
}

__host__
void freedevice(const Realm::ExternalInstanceResource &allocation)
{
  const Realm::ExternalCudaMemoryResource *resource =
    static_cast<const Realm::ExternalCudaMemoryResource*>(&allocation);
  CUDA_CHECK( cudaFree((void*)resource->base) );
}

__host__
void make_future(const void *args, size_t arglen,
                 const void *userdata, size_t userlen, Processor proc) {
  const Task *task = NULL;
  const std::vector<PhysicalRegion> *regions = NULL;
  Context ctx;
  Runtime *runtime = NULL;
  Runtime::legion_task_preamble(args, arglen, proc, task, regions, ctx, runtime);

  int *result_d = NULL;
  CUDA_CHECK( cudaMalloc((void**)&result_d, sizeof(int)) );
  write_result<<<1,1>>>(result_d, task->index_point[0]);

  int device;
  CUDA_CHECK( cudaGetDevice(&device) );
  const Realm::ExternalCudaMemoryResource resource(device, result_d, sizeof(int));
  Runtime::legion_task_postamble(ctx, result_d, sizeof(int), true/*owned*/,
      resource, freedevice);
}

__host__
void check_future(const void *args, size_t arglen,
                 const void *userdata, size_t userlen, Processor proc)
{
  const Task *task = NULL;
  const std::vector<PhysicalRegion> *regions = NULL;
  Context ctx;
  Runtime *runtime = NULL;
  Runtime::legion_task_preamble(args, arglen, proc, task, regions, ctx, runtime);

  const int points = task->index_domain.get_volume(); 
  assert(task->futures.size() == 1);
  const int *result_d = (const int*)task->futures.front().get_buffer(Memory::GPU_FB_MEM);

  bool *pass_d = NULL;
  CUDA_CHECK( cudaMalloc((void**)&pass_d, sizeof(bool)) );
  check_result<<<1,1>>>(result_d, points, pass_d);

  int device;
  CUDA_CHECK( cudaGetDevice(&device) );
  const Realm::ExternalCudaMemoryResource resource(device, pass_d, sizeof(bool));
  Runtime::legion_task_postamble(ctx, pass_d, sizeof(bool), true/*owned*/, 
      resource, freedevice);
}

__host__
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime *runtime) {
  Future fgpus = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_GLOBAL_GPUS);
  const int num_gpus = fgpus.get_result<long>(false/*silence warnings*/);
  if (num_gpus <= 0) {
    std::cout << "No GPUs found so exiting early" << std::endl;
    return;
  }
  std::cout << "Found " << num_gpus << " GPUs" << std::endl;
  IndexSpace is_gpu = runtime->create_index_space(ctx, Rect<1>(0, num_gpus - 1));

  IndexTaskLauncher make_future_launcher(
    TID_MAKE_FUTURE, is_gpu, TaskArgument(NULL, 0), ArgumentMap());
  Future f1 = runtime->execute_index_space(ctx, make_future_launcher, LEGION_REDOP_VALUE(SUM, INT32));
  IndexTaskLauncher check_future_launcher(
    TID_CHECK_FUTURE, is_gpu, TaskArgument(NULL, 0), ArgumentMap());
  check_future_launcher.add_future(f1);
  Future f2 = runtime->execute_index_space(ctx, check_future_launcher, LEGION_REDOP_VALUE(PROD, BOOL));
  bool result = f2.get_result<bool>(true/*silence warnings*/);
  if (result)
    printf("SUCCESS\n");
  else
    printf("FAILURE\n");
}

class DeviceMapper : public Mapping::DefaultMapper {
public:
  DeviceMapper(Mapping::MapperRuntime *rt, Machine machine, 
                           Processor local, const char *mapper_name)
    : DefaultMapper(rt, machine, local, mapper_name) { }

  virtual void map_task(Mapping::MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                              MapTaskOutput& output)
  {
    DefaultMapper::map_task(ctx, task, input, output);
    if (!task.futures.empty()) {
      output.future_locations.resize(task.futures.size());
      for (unsigned idx = 0; idx < task.futures.size(); idx++)
      {
        Machine::MemoryQuery query(machine);
        query.only_kind(Memory::GPU_FB_MEM);
        query.local_address_space();
        query.best_affinity_to(task.current_proc);
        assert(query.count() == 1);
        output.future_locations[idx] = query.first();
      }
    }
  }
};

__host__
void update_mappers(Machine machine, Runtime *runtime,
                    const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it =
        local_procs.begin(); it != local_procs.end(); it++)
  {
    DeviceMapper *mapper =
      new DeviceMapper(runtime->get_mapper_runtime(),
                                   machine, *it, "device_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

__host__
int main(int argc, char **argv) {
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.set_replicable();
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(TID_MAKE_FUTURE, "make_future");
    registrar.set_leaf();
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    CodeDescriptor desc(make_future);
    Runtime::preregister_task_variant(registrar, desc, NULL, 0, "make_future",
        LEGION_AUTO_GENERATE_ID, sizeof(int), true/*has return type*/);
  }
  {
    TaskVariantRegistrar registrar(TID_CHECK_FUTURE, "check_future");
    registrar.set_leaf();
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    CodeDescriptor desc(check_future);
    Runtime::preregister_task_variant(registrar, desc, NULL, 0, "check_future",
        LEGION_AUTO_GENERATE_ID, sizeof(bool), true/*has return type*/);
  }
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  Runtime::add_registration_callback(update_mappers);
  return Runtime::start(argc, argv);
}
