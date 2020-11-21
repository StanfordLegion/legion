#include <nccl.h>
#include <legion.h>

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

#define NCCL_CHECK(call)                                        \
  do {                                                          \
    ncclResult_t status = call;                                 \
    if (status != ncclSuccess) {                                \
      printf("NCCL error at %s:%d : '%s'\n",                    \
             __FILE__, __LINE__, ncclGetErrorString(status));   \
      exit(EXIT_FAILURE);                                       \
    }                                                           \
  } while (0)

ncclComm_t* get_nccl_comm(size_t gpu_idx, size_t num_gpus) {
  static std::vector<ncclComm_t> comms(num_gpus, 0);
  assert(comms.size() == num_gpus);
  return &(comms[gpu_idx]);
}

enum TaskIDs {
  TID_NCCL_GET_ID,
  TID_NCCL_INIT,
  TID_NCCL_REDUCE,
  TID_NCCL_FINALIZE,
  TID_TOP_LEVEL
};

// Get NCCL unique ID on one rank, to broadcast to all others.
ncclUniqueId nccl_get_id_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx,
                              Runtime *runtime) {
  ncclUniqueId nccl_id;
  NCCL_CHECK(ncclGetUniqueId(&nccl_id));
  return nccl_id;
}

void nccl_init_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime *runtime) {
  size_t num_gpus = task->index_domain.get_volume();
  size_t gpu_idx = task->index_point[0];
  std::cout << "On GPU " << gpu_idx << ": processor " << Processor::get_executing_processor() << std::endl;
  ncclComm_t* nccl_comm = get_nccl_comm(gpu_idx, num_gpus);
  ncclUniqueId nccl_id = task->futures[0].get_result<ncclUniqueId>();
  NCCL_CHECK(ncclCommInitRank(nccl_comm, num_gpus, nccl_id, gpu_idx));
}

void nccl_reduce_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx,
                      Runtime *runtime) {
  size_t num_gpus = task->index_domain.get_volume();
  size_t gpu_idx = task->index_point[0];
  ncclComm_t nccl_comm = *(get_nccl_comm(gpu_idx, num_gpus));
  cudaStream_t task_stream;
  CUDA_CHECK(cudaStreamCreate(&task_stream));
  int* host_buf = static_cast<int*>(malloc(42 * sizeof(int)));
  for (int i = 0; i < 42; ++i) {
    host_buf[i] = i;
  }
  int *dev_buf;
  CUDA_CHECK(cudaMalloc(&dev_buf, 42 * sizeof(int)));
  CUDA_CHECK(cudaMemcpyAsync(dev_buf, host_buf, 42 * sizeof(int), cudaMemcpyHostToDevice, task_stream));
  CUDA_CHECK(cudaStreamSynchronize(task_stream));
  NCCL_CHECK(ncclAllReduce(dev_buf, dev_buf, 42, ncclInt, ncclSum, nccl_comm, task_stream));
  CUDA_CHECK(cudaStreamSynchronize(task_stream));
  CUDA_CHECK(cudaMemcpyAsync(host_buf, dev_buf, 42 * sizeof(int), cudaMemcpyDeviceToHost, task_stream));
  CUDA_CHECK(cudaStreamSynchronize(task_stream));
  if (gpu_idx == 0) {
    for (int i = 0; i < 42; ++i) {
      assert(host_buf[i] == i * num_gpus);
    }
  }
  CUDA_CHECK(cudaFree(dev_buf));
  free(host_buf);
}

void nccl_finalize_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx,
                        Runtime *runtime) {
  size_t num_gpus = task->index_domain.get_volume();
  size_t gpu_idx = task->index_point[0];
  ncclComm_t nccl_comm = *(get_nccl_comm(gpu_idx, num_gpus));
  NCCL_CHECK(ncclCommDestroy(nccl_comm));
}

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

  TaskLauncher nccl_get_id_launcher(
    TID_NCCL_GET_ID, TaskArgument(NULL, 0));
  Future nccl_get_id_future = runtime->execute_task(ctx, nccl_get_id_launcher);

  IndexTaskLauncher nccl_init_launcher(
    TID_NCCL_INIT, is_gpu, TaskArgument(NULL, 0), ArgumentMap());
  nccl_init_launcher.add_future(nccl_get_id_future);
  FutureMap nccl_init_futures = runtime->execute_index_space(ctx, nccl_init_launcher);
  nccl_init_futures.wait_all_results();

  IndexTaskLauncher nccl_reduce_launcher(
    TID_NCCL_REDUCE, is_gpu, TaskArgument(NULL, 0), ArgumentMap());
  FutureMap nccl_reduce_futures = runtime->execute_index_space(ctx, nccl_reduce_launcher);
  nccl_reduce_futures.wait_all_results();

  IndexTaskLauncher nccl_finalize_launcher(
    TID_NCCL_FINALIZE, is_gpu, TaskArgument(NULL, 0), ArgumentMap());
  FutureMap nccl_finalize_futures = runtime->execute_index_space(ctx, nccl_finalize_launcher);
  nccl_finalize_futures.wait_all_results();
}

int main(int argc, char **argv) {
  // This needs to be set, otherwise NCCL will try to use group kernel launches,
  // which are not compatible with the Realm CUDA hijack.
  setenv("NCCL_LAUNCH_MODE", "PARALLEL", true);
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(TID_NCCL_GET_ID, "nccl_get_id");
    registrar.set_leaf();
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<ncclUniqueId,nccl_get_id_task>(registrar, "nccl_get_id");
  }
  {
    TaskVariantRegistrar registrar(TID_NCCL_INIT, "nccl_init");
    registrar.set_leaf();
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<nccl_init_task>(registrar, "nccl_init");
  }
  {
    TaskVariantRegistrar registrar(TID_NCCL_REDUCE, "nccl_reduce");
    registrar.set_leaf();
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<nccl_reduce_task>(registrar, "nccl_reduce");
  }
  {
    TaskVariantRegistrar registrar(TID_NCCL_FINALIZE, "nccl_finalize");
    registrar.set_leaf();
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<nccl_finalize_task>(registrar, "nccl_finalize");
  }
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  return Runtime::start(argc, argv);
}
