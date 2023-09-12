#include "legion.h"
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
};

__global__
void write_result(int *ptr, int offset)
{
  *ptr = 42 + offset;
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
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime *runtime) {
  int num_gpus = -1;
  const InputArgs& command_args = Runtime::get_input_args();
  for (int i = 1; i < command_args.argc; i++) {
    auto s = std::string(command_args.argv[i]);
    if (s.compare("-ll:gpu") == 0 && i + 1 < command_args.argc) {
      num_gpus = std::stoi(command_args.argv[i + 1]);
      break;
    }
  }
  if (num_gpus <= 0) {
    std::cout << "Unable to parse #gpus" << std::endl;
    return;
  }
  num_gpus *= Machine::get_machine().get_address_space_count();
  std::cout << "Got #GPUs: " << num_gpus << std::endl;
  IndexSpace is_gpu = runtime->create_index_space(ctx, Rect<1>(0, num_gpus - 1));

  IndexTaskLauncher make_future_launcher(
    TID_MAKE_FUTURE, is_gpu, TaskArgument(NULL, 0), ArgumentMap());
  FutureMap fm = runtime->execute_index_space(ctx, make_future_launcher);
  for (int g = 0; g < num_gpus; g++) {
    Future f = fm.get_future(DomainPoint(Point<1>(g))); 
    int result = f.get_result<int>(true/*silence wait warnings*/);
    if (result != (42 + g)) {
      printf("FAILED on GPU %d: received %d but expected %d\n", g, result, 42 + g);
      assert(false);
    }
  }
  printf("SUCCESS\n");
}

__host__
int main(int argc, char **argv) {
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(TID_MAKE_FUTURE, "make_future");
    registrar.set_leaf();
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    CodeDescriptor desc(make_future);
    Runtime::preregister_task_variant(registrar, desc, NULL, 0, "make_future",
                            LEGION_AUTO_GENERATE_ID, true/*has return type*/);
  }
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  return Runtime::start(argc, argv);
}
