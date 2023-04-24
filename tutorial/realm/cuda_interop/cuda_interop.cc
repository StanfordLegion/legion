//------------------------------------------------------------------------------
// Copyright 2023 NVIDIA Corp.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

#include <cuda_runtime_api.h>
#include <realm.h>
#include <realm/cuda/cuda_access.h>

#if !defined(REALM_USE_CUDA) && !defined(REALM_USE_HIP)
#error Realm not compiled with either CUDA or HIP enabled
#endif // !defined(REALM_USE_CUDA) || !defined(REALM_USE_HIP)

using namespace Realm;

Logger log_app("app");

static void checkCudaError(cudaError_t e, const char *file, int line)
{
  if(REALM_UNLIKELY(e != cudaSuccess)) {
    log_app.error() << "CUDA Error detected at " << file << '(' << line << "): " << e;
    abort();
  }
}

#define CHECK_CUDART(x) checkCudaError(x, __FILE__, __LINE__)

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE,
  GPU_LAUNCH_TASK,
  CHECK_TASK
};

// See gpu_kernel.cu for the definition of this device entry point
void copyKernel(Rect<2> bounds, AffineAccessor<float, 2> linear_accessor,
                cudaSurfaceObject_t surface);

struct GPUTaskArgs {
  Rect<2> bounds;
  cudaSurfaceObject_t surface_obj;
  RegionInstance linear_instance;
};

// This is the task entry point for the GPU to launch a kernel.
static void gpu_launch_task(const void *data, size_t datalen, const void *userdata,
                            size_t userlen, Processor p)
{
  // Retrieve the arguments passed to us
  assert(datalen == sizeof(GPUTaskArgs));
  GPUTaskArgs task_args = *static_cast<const GPUTaskArgs *>(data);

  // Retrieve the AffineAccessor, which will give us the GPU address we can pass below.
  AffineAccessor<float, 2> linear_accessor(task_args.linear_instance, 0);

  // Get the suggested occupancy of the kernel
  int grid_size = 1, block_size = 128;
  {
    int num_sms = 1, dev = -1;
    cudaFuncAttributes func_attributes;
    CHECK_CUDART(
        cudaFuncGetAttributes(&func_attributes, reinterpret_cast<void *>(copyKernel)));
    block_size = func_attributes.maxThreadsPerBlock;
    CHECK_CUDART(cudaGetDevice(&dev));
    CHECK_CUDART(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
    CHECK_CUDART(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &grid_size, reinterpret_cast<void *>(copyKernel), block_size, 0));
    grid_size *= num_sms;
    // Limit the grid size by the number of elements if the number of elements is small
    if(static_cast<size_t>(grid_size * block_size) > task_args.bounds.volume()) {
      grid_size = (task_args.bounds.volume() + block_size - 1) / block_size;
    }
  }

  // Queue the work!
  cudaStream_t stream;
  CHECK_CUDART(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // In order to be able to
  // compile this file without nvcc, we forego the usual <<<>>> syntax and use
  // cudaLaunchKernel instead.
  void *args[] = {&task_args.bounds, &linear_accessor, &task_args.surface_obj};
  CHECK_CUDART(cudaLaunchKernel(reinterpret_cast<void *>(copyKernel), grid_size,
                                block_size, args, 0, stream));

  // The queued cuda work is automatically synchronized before the event for this task is
  // triggered, so we don't need to do any synchronization of the previous work

  // Clean up the stream in the end (does not synchronize with launched work, just cleans
  // up the stream to prevent a resource leak)
  CHECK_CUDART(cudaStreamDestroy(stream));
}

struct CheckTaskArgs {
  RegionInstance host_linear_instance;
  float expected_value;
};

// This is the task entry point for checking the results match the expected value.
static void check_task(const void *data, size_t datalen, const void *userdata,
                       size_t userlen, Processor p)
{
  assert(datalen == sizeof(CheckTaskArgs));
  const CheckTaskArgs &task_args = *static_cast<const CheckTaskArgs *>(data);
  const Rect<2> bounds = task_args.host_linear_instance.get_indexspace<2>().bounds;
  const float expected_value = task_args.expected_value;
  AffineAccessor<float, 2> linear_accessor(task_args.host_linear_instance, 0);
  bool failed = false;
  for(PointInRectIterator<2> pir(bounds); pir.valid; pir.step()) {
    float value = linear_accessor[pir.p];
    if(value != expected_value) {
      log_app.error() << "array[" << pir.p << "] = " << value << " != " << expected_value;
      failed = true;
    }
  }
  if(failed) {
    abort();
  } else {
    log_app.print() << "PASSED!";
  }
}

// This is the main task entry point that will initialize and coordinate all the various
// tasks we have.
static void main_task(const void *data, size_t datalen, const void *userdata,
                      size_t userlen, Processor gpu)
{
  const size_t width = 1024, height = 1024;
  std::vector<size_t> field_sizes(1, sizeof(float));
  // Create a rectangle boundary of [(0,0), (width,height)), where the end is open-ended.
  // Realm's indexspaces are close ended on both ends, so subtract one from the width and
  // height to account for this.
  Rect<2> bounds(Point<2>(0, 0), Point<2>(width - 1, height - 1));

  // ==== Allocating GPU memory with Realm ====
  // FInd the fastest suitable memory that will fit our problem space
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .has_capacity(bounds.volume() * sizeof(float))
                       .best_affinity_to(gpu)
                       .first();

  // Alternatively, we can pick a specific memory type for our problem:
  // gpu_mem = Machine::MemoryQuery(Machine::get_machine())
  //  .only_kind(Memory::GPU_FB_MEM).first();       // Realm managed slice (see -ll:fbmem)
  //  .only_kind(Memory::GPU_DYNAMIC_MEM).first();  // i.e. cudaMalloc
  //  .only_kind(Memory::GPU_MANAGED_MEM).first();  // i.e. cudaMallocManaged
  //  .only_kind(Memory::Z_COPY_MEM).first();       // i.e. cudaMallocHost

  assert((gpu_mem != Memory::NO_MEMORY) && "Failed to find suitable GPU memory to use!");

  std::cout << "Choosing GPU memory type " << gpu_mem.kind() << " for GPU processor "
            << gpu << std::endl;

  // Now create a 2D instance like we normally would
  RegionInstance linear_instance = RegionInstance::NO_INST;
  Event linear_instance_ready_event =
      RegionInstance::create_instance(linear_instance, gpu_mem, bounds, field_sizes,
                                      /*SOA*/ 1, ProfilingRequestSet());

  // ==== Registering CUDA Arrays in Realm ====
  // Allocate our cuda array as we would normally, creating the surface object for the gpu
  // task later
  cudaArray_t array;
  cudaSurfaceObject_t surface_obj;
  {
    cudaExtent extent;
    cudaChannelFormatDesc fmt =
        cudaCreateChannelDesc(8 * sizeof(float), 0, 0, 0, cudaChannelFormatKindFloat);
    cudaResourceDesc resource_descriptor = {};

    extent.width = width;
    extent.height = height;
    extent.depth = 1;
    CHECK_CUDART(cudaMalloc3DArray(&array, &fmt, extent, cudaArraySurfaceLoadStore));

    resource_descriptor.resType = cudaResourceTypeArray;
    resource_descriptor.res.array.array = array;
    CHECK_CUDART(cudaCreateSurfaceObject(&surface_obj, &resource_descriptor));
  }

  // Create an array resource descriptor to describe the memory in which to build an
  // instance from
  RegionInstance array_instance = RegionInstance::NO_INST;
  Event array_ready_event = Event::NO_EVENT;
  {
    InstanceLayout<2, int> layout;
    InstanceLayoutGeneric::FieldLayout &field_layout = layout.fields[0];

    int gpu_idx = -1;
    CHECK_CUDART(cudaGetDevice(&gpu_idx));

    layout.space = IndexSpace<2>(bounds);
    field_layout.list_idx = 0;
    field_layout.rel_offset = 0;
    field_layout.size_in_bytes = field_sizes[0];
    layout.piece_lists.resize(1);
    // Specify a non-affine layout by specifying a CudaArrayLayoutPiece that spans the
    // entire cuda array
    CudaArrayLayoutPiece<2, int> *layout_piece = new CudaArrayLayoutPiece<2, int>;
    layout_piece->bounds = bounds;
    layout.piece_lists[0].pieces.push_back(layout_piece);

    ExternalCudaArrayResource cuda_array_external_resource(gpu_idx, array);

    array_ready_event = RegionInstance::create_external_instance(
        array_instance, cuda_array_external_resource.suggested_memory(), layout.clone(),
        cuda_array_external_resource, ProfilingRequestSet());
  }

  Event instances_ready_event =
      Event::merge_events(linear_instance_ready_event, array_ready_event);

  // ==== Data Movement ====
  Event fill_done_event = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    // Realm does not currently support non-affine fill operations, so fill the linear
    // array with the value we want and copy the linear array to the cuda array instance,
    // making sure to chain the events as we go along
    // While we could use cuda APIs to achieve the same result (using cudaMemcpy3D or
    // cudaMemcpyToArray), we would need to synchronize the instance creation at this
    // point for the linear array, or allocate our own gpu-accessible staging buffer, or
    // have cuda do a pageable memcpy to the array.  Instead, we just let realm handle it
    // for us asynchrnously.

    // Fill the linear array with zeros.
    srcs[0].set_fill<float>(0.0f);
    dsts[0].set_field(linear_instance, 0, field_sizes[0]);
    fill_done_event =
        bounds.copy(srcs, dsts, ProfilingRequestSet(), instances_ready_event);

    // Copy the linear array to the cuda array, filling it with zeros.
    srcs[0].set_field(linear_instance, 0, field_sizes[0]);
    dsts[0].set_field(array_instance, 0, field_sizes[0]);
    fill_done_event = bounds.copy(srcs, dsts, ProfilingRequestSet(), fill_done_event);

    // Fill the linear array with ones.
    srcs[0].set_fill<float>(1.0f);
    dsts[0].set_field(linear_instance, 0, field_sizes[0]);
    fill_done_event = bounds.copy(srcs, dsts, ProfilingRequestSet(), fill_done_event);
  }

  // ==== Task Spawning ====
  Event gpu_task_done_event = Event::NO_EVENT;
  {
    GPUTaskArgs args;
    args.bounds = bounds;
    args.linear_instance = linear_instance;
    args.surface_obj = surface_obj;
    gpu_task_done_event =
        gpu.spawn(GPU_LAUNCH_TASK, &args, sizeof(args), fill_done_event);
  }

  // Pick a CPU processor and use it to check the results
  // We could use a TOC_PROC here, or even the same processor we're currently running on,
  // but this processor is dedicated to submitting GPU work and isn't really fit for
  // CPU-only tasks
  Processor check_processor = Machine::ProcessorQuery(Machine::get_machine())
                                  .only_kind(Processor::LOC_PROC)
                                  .local_address_space()
                                  .first();
  assert((check_processor != Processor::NO_PROC) &&
         "Failed to find suitable CPU processor to check results!");

  // While the processing task is running, create some CPU-visible storage to store the
  // result.  Add affinity to the GPU for better copy performance at the cost of
  // consuming pinned memory.
  Memory cpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .has_capacity(width * height * sizeof(float))
                       .has_affinity_to(check_processor)
                       .has_affinity_to(gpu)
                       .first();

  // We could use any memory that has some affinity to the check_processor,
  // even memory that isn't visible to the GPU:
  // cpu_mem = Machine::MemoryQuery(Machine::get_machine())
  // .best_affinity_to(check_processor).first();                  // i.e. malloc()
  // .has_affinity_to(gpu).only_kind(Memory::Z_COPY_MEM).first(); // i.e. cudaMallocHost()

  assert((cpu_mem != Memory::NO_MEMORY) && "Failed to find suitable CPU memory to use!");

  std::cout << "Choosing CPU memory type " << cpu_mem.kind() << " for CPU processor "
            << check_processor << std::endl;

  RegionInstance check_instance = RegionInstance::NO_INST;
  Event check_instance_ready_event = RegionInstance::create_instance(
      check_instance, cpu_mem, bounds, field_sizes, 1, ProfilingRequestSet());

  // Copy the result back, waiting on the processing to complete
  Event copy_done_event = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    // Initialize the host memory with some data
    srcs[0].set_fill<float>(2.0f);
    dsts[0].set_field(check_instance, 0, field_sizes[0]);
    copy_done_event =
        bounds.copy(srcs, dsts, ProfilingRequestSet(), check_instance_ready_event);
    // Overwrite the previous fill with the data from the array
    srcs[0].set_field(array_instance, 0, field_sizes[0]);
    dsts[0].set_field(check_instance, 0, field_sizes[0]);
    copy_done_event =
        bounds.copy(srcs, dsts, ProfilingRequestSet(),
                    Event::merge_events(copy_done_event, gpu_task_done_event));
  }

  Event check_task_done_event = Event::NO_EVENT;
  {
    CheckTaskArgs args;
    args.expected_value = 1.0f;
    args.host_linear_instance = check_instance;
    check_task_done_event =
        check_processor.spawn(CHECK_TASK, &args, sizeof(args), copy_done_event);
  }

  // Clean up everything
  check_instance.destroy(check_task_done_event);
  array_instance.destroy(copy_done_event);
  linear_instance.destroy(copy_done_event);

  // Because we use cuda here, we need to either spawn a separate task to free up the
  // external resources after the copy_done_event has fired, or wait here to do so,
  // otherwise we can leak the cuda array
  copy_done_event.wait();
  CHECK_CUDART(cudaDestroySurfaceObject(surface_obj));
  CHECK_CUDART(cudaFreeArray(array));

  // Now that everything is cleaned up and all our tasks are queued up, request for a
  // shutdown once the check task is done
  Runtime::get_runtime().shutdown(check_task_done_event);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task), ProfilingRequestSet(), 0, 0)
      .wait();

  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                   GPU_LAUNCH_TASK, CodeDescriptor(gpu_launch_task),
                                   ProfilingRequestSet(), 0, 0)
      .wait();

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, CHECK_TASK,
                                   CodeDescriptor(check_task), ProfilingRequestSet(), 0,
                                   0)
      .wait();

  // Pick a GPU that's in our local address space (as opposed to on a remote node)
  // We launch the top level task on a GPU Processor in order to make cuda API calls for
  // the target device
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::TOC_PROC)
                    .first();

  assert((p != Processor::NO_PROC) && "Unable to find suitable GPU processor");

  // collective launch of this task across all ranks, so we could utililze processors on
  // other ranks.  This task will indicate the event that will signal shutdown
  rt.collective_spawn(p, MAIN_TASK, 0, 0);

  // now sleep this thread until that shutdown actually happens
  return rt.wait_for_shutdown();
}