# Realm-CUDA Interop

## Introduction
In this tutorial, we look at how to interoperate with CUDA in order to utilize
the Realm task-based programming model to take advantage of the raw computing power
of the GPU.

This tutorial will review several features and best practices for working with
Realm and CUDA, listed below:

* [Enabling CUDA in Realm](#enabling-cuda-in-realm)
* [Registering GPU Tasks](#registering-gpu-tasks)
* [Allocating Memory Through Realm](#allocating-memory-through-realm)
  * [CUDA Arrays](#cuda-arrays)
* [Data Movement](#data-movement)
* [Best Practices](#best-practices)

## Enabling CUDA in Realm

To enable CUDA in Realm, we need to ensure we have an appropriate CUDA Toolkit available to compile with.  Follow your system's documentation or the official [CUDA Documentation][1] on how to do this.

If using CMAKE to build Realm, all that is required to enable CUDA is to add `-DLegion_USE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=<path_to_cuda>` to our normal cmake configure line.  For example:

```bash
$ cmake -DLegion_USE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..
```

Realm provides various low-level command line arguments to configure how GPUs are set up prior to application use.  One of these is `-ll:gpus N`, which will tell Realm the number of GPUs that Realm can use. ->
An example of a command line argument is -ll:gpus N, which specifies the number of GPUs that Realm can utilize.  When combined with CUDA_VISIBLE_DEVICES or CUDA_DEVICE_ORDER, or some of Realm's other low level command line arguments, we can define exactly what GPU is associated with the specific instance of the application.

Other notable low-level command line arguments are the following:

| Argument               | Description                                      |
|------------------------|--------------------------------------------------|
| `-ll:gpus N`           | Associates the first N gpus to this application  |
| `-cuda:skipgpus N`     | Number of gpus to skip between each chosen GPU   |
| `-ll:gpu_ids x,y,z`    | Specifies the CUDA device ids of the GPUs to associate with the application |
| `-ll:pin`              | Register local processor's memory to pin for DMA use |
| `-cuda:hostreg N`      | Specify the amount of memory on local processors to pin for DMA use (default is 1GiB)  |
| `-cuda:callbacks`      | Allow Realm managed fences to use cuStreamAddCallback instead of polling on cudaEvents |
| `-cuda:nohijack`       | Suppress the runtime warning about the CUDA Hijack not active |
| `-cuda:skipbusy`       | Skip any gpus that do not initialize                    |
| `-cuda:minavailmem N`  | Skip any gpus that do not have at minimum N MiB of memory |
| `-cuda:legacysync`     | Track task CUDA progress with an event on the default stream (does not capture NON_BLOCKING streams) |
| `-cuda:contextsync`    | Force Realm to call cuCtxSynchronize after a `TOC_PROC` task completes   |
| `-cuda:maxctxsync N`   | Maximum number of outstanding background context synchronization threads |
| `-cuda:lmemresize`     | Set the CU_CTX_LMEM_RESIZE_TO_MAX flag on the context   |
| `-cuda:mtdma  `        | Enable multi-threaded DMA                               |
| `-cuda:ipc`            | Use legacy [CUDA IPC][2] for interprocess communication |

## Registering GPU Tasks

Looking at `main()` in the tutorial, we see most of the standard Realm boilerplate code.  The main difference is that when we want to do something with CUDA, we want to target the `TOC_PROC` (also known as the *throughput optimized compute processor*) rather than the `LOC_PROC` kind (also known as the *latency optimized compute processor*).  It is because when a task is run on a `TOC_LOC` processor, Realm will properly set up the thread state to target the associated device.  In addition, any CUDA work launched or enqueued in a stream will automatically be synchronized in the background after the thread has returned from the task.  We will get back to why this is important later.

To target a `TOC_PROC` for CUDA, we simply change what kind of processors a task id can target when we register it:
```c++
Processor::register_task_by_kind(Processor::TOC_PROC,
                                 false /*!global*/,
                                 MAIN_TASK,
                                 CodeDescriptor(main_task),
                                 ProfilingRequestSet(),
                                 0, 0).wait();
```

Afterwards, we just need to find a `TOC_PROC` to spawn the main task onto:

```c++
Processor p = Machine::ProcessorQuery(Machine::get_machine())
                  .only_kind(Processor::TOC_PROC)
                  .first();
```

And just like any other processor, launch the registered GPU task on it!

## Allocating Memory Through Realm

For this tutorial, we are looking at doing something relatively simple: allocate a linear 2D array and a [cudaSurfaceObject][3], fill each with some data and copy the linear array to the cudaSurfaceObject.  In order to fully leverage Realm's data movement and other asynchronous features, we will create RegionInstances for each of these memories.  This gives us the same interface to work with that we have seen in previous tutorials.

Allocating linear GPU memory is fairly close to how it is done for standard CPU.  Below are highlights from `main_task()`:

1) Define the memory layout (remember Realm's bounds are inclusive, whereas CUDA's are not, so we subtract one from the width and height here).
```c++
std::vector<size_t> field_sizes(1, sizeof(float));
Rect<2> bounds(Point<2>(0, 0), Point<2>(width - 1, height - 1));
```
2) Find the specific memory you want to allocate that suits your needs.
```c++
Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                     .has_capacity(bounds.volume() * sizeof(float))
                     .best_affinity_to(gpu)
                     .first();
```
3) Create a Realm::RegionInstance, just like in previous tutorials.
```c++
RegionInstance::create_instance(linear_instance, gpu_mem, bounds, field_sizes,
                                /*SOA*/ 1, ProfilingRequestSet());
```

Looking at the MemoryQuery, just like ProcessorQuery, there are different memory kinds specific to GPUs that can be leveraged depending on various use cases.  By default, this tutorial just picks the best one that has enough room for our work, but an application may want specific features of other memory kinds.  These memory kinds are described as follows:

| Memory Kind       | Description                                               |
|-------------------|-----------------------------------------------------------|
| `GPU_FB_MEM`      | Pre-allocated cudaMalloc() memory managed by Realm        |
| `GPU_DYNAMIC_MEM` | Maps to cudaMalloc() for each instance creation           |
| `GPU_MANAGED_MEM` | Pre-allocated cudaMallocManaged() memory managed by Realm |
| `Z_COPY_MEM`      | Pre-allocated cudaMallocHost() memory managed by Realm    |

For more information on the features of each of these kinds, consult the [CUDA Documentation][1].  Each of these has command line arguments for controlling the size of these memories:

| Argument               | Description                                                         |
|------------------------|---------------------------------------------------------------------|
| `-ll:fsize N`          | Specify the pre-allocated size of GPU_FB_MEM                        |
| `-ll:zsize N`          | Specify the pre-allocated size of Z_COPY_MEM                        |
| `-ll:msize N`          | Specify the pre-allocated size of GPU_MANAGED_MEM                   |
| `-cuda:dynfb`          | Enable the `GPU_DYNAMIC_MEM` memory kind                            |
| `-cuda:dynfb_max N`    | Cap the size of `GPU_DYNAMIC_MEM` to the specified amount (default is the GPU's framebuffer size) |

For now, we will just have Realm pick the one that matches our requirements using `best_affinity_to()`.

### CUDA Arrays

Now that the linear memory is allocated, time to allocate the `cudaSurfaceObject_t`.  `cudaSurfaceObject_t` is great to use if processing elements requires neighborhood lookups (e.g., image convolutions) because their layout is not linear. However a [Z-order curve][4], also known as "block linear", improves cache utilization for such access patterns.

While a little more complicated, allocating CUDA surface objects can be essentially broken down into the same basic steps as linear memory with one exception; instead of querying for what memory to use, we will allocate the memory directly with CUDA APIs and use the `ExternalCudaArrayResource` class to register the memory with Realm to use.  This means we have to manage the memory lifetime ourselves, but we can still leverage all that Realm provides after we complete the registration:

1) Allocate the cuda array and bind a cudaSurfaceObject_t to be used later.  If this code is unfamiliar to you, check out the [simpleSurfaceWrite][5] CUDA sample for more information.
```c++
cudaArray_t array;
cudaSurfaceObject_t surface_obj;
cudaExtent extent;
cudaChannelFormatDesc fmt =
    cudaCreateChannelDesc(8 * sizeof(float), 0, 0, 0, cudaChannelFormatKindFloat);
cudaResourceDesc resource_descriptor = {};

extent.width = width;
extent.height = height;
extent.depth = 1;
cudaMalloc3DArray(&array, &fmt, extent, cudaArraySurfaceLoadStore);

resource_descriptor.resType = cudaResourceTypeArray;
resource_descriptor.res.array.array = array;
cudaCreateSurfaceObject(&surface_obj, &resource_descriptor);
```

2) Describe the layout of the memory to Realm.  We have to describe this as a non-affine layout and utilize the `CudaArrayLayoutPiece` class in order to tell Realm this is not ordinarily linear memory and needs to be treated as special.
```c++
    InstanceLayout<2, int> layout;
    InstanceLayoutGeneric::FieldLayout &field_layout = layout.fields[0];

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
```

3) Create a Realm::RegionInstance with the given ExternalCudaArrayResource, this time using create_external_instance instead.
```c++
RegionInstance::create_external_instance(
    array_instance, cuda_array_external_resource.suggested_memory(),
    layout.clone(), cuda_array_external_resource, ProfilingRequestSet());
```

## Data Movement

Now that we have allocated memory for the data, let us fill it in.  The logic on how to do this with Realm is exactly the same as in previous tutorials:

```c++
// Fill the linear array with ones.
srcs[0].set_fill<float>(1.0f);
dsts[0].set_field(linear_instance, 0, field_sizes[0]);
fill_done_event = bounds.copy(srcs, dsts, ProfilingRequestSet(), fill_done_event);
```

As of this writing, Realm does not currently support non-affine fill operations, so we will reuse `linear_instance` in order to first fill it, then copy it to the array properly:

```c++
// Copy the linear array to the cuda array, filling it with zeros.
srcs[0].set_field(linear_instance, 0, field_sizes[0]);
dsts[0].set_field(array_instance, 0, field_sizes[0]);
fill_done_event = bounds.copy(srcs, dsts, ProfilingRequestSet(), fill_done_event);
```

While we could use CUDA APIs to do these operations, we would first need to synchronize on the creation of the instances directly.  With Realm, we can describe the events that are needed to coordinate and execute all of these operations asynchronously.

So far, we have created Realm managed GPU-visible linear memory, registered a CUDA Array with Realm, and initialized these with appropriate values.  We now have everything we need to launch work on the GPU with CUDA.  For this tutorial, we will be using Realm's AffineAccessor from the device, as it can contain both the device address and the strided layout of the memory.

As an aside, sometimes you need to pass the GPU address to a library or device code directly, which you can get from the linear_accessor.  Additionally, for N-dimensional instances like this, we will need to make sure to handle larger strides of memory chosen by Realm for performance reasons in order to index it properly:

```c++
float *linear_ptr = &linear_accessor[Point<2>(0, 0)];
size_t pitch_in_elements = linear_accessor.strides[1] / linear_accessor.strides[0];
```

The device code for processing is fairly simple, but the key highlights are the fact that we can use some Realm structures like `Rect<N,T>`, `AffineAccessor<T,N>`, and `Point<N,T>`.

```c++
__global__ void copyKernel(Rect<2> bounds, AffineAccessor<float, 2> linear_accessor,
                           cudaSurfaceObject_t surface)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;

  // Have each thread copy the value from the linear array to the surface
  for(; tid < bounds.volume(); tid += stride) {
    size_t y = tid / (bounds.hi[0] + 1);
    size_t x = tid - y * (bounds.hi[0] + 1);
    float value = linear_accessor[Point<2>(x, y)];
    surf2Dwrite<float>(value, surface, x * sizeof(float), y, cudaBoundaryModeTrap);
  }
}
```

Once the data is processed, we can read it back by creating some host visible memory to copy to. When creating the instance for reading, we can choose any memory that our `check_processor` has an affinity to (and can therefore access).  In the tutorial, we let Realm pick the memory to use; however, we can specify that we want memory that is also accessible to the GPU in order to optimize the copy operation.  Without GPU access to the memory, the copy would have to be chunked up and done in stages, which can add extra overhead for large copies.  However, it may be beneficial for the case when such communication is infrequent or when pinned memory resources are limited.

```c++
cpu_mem = Machine::MemoryQuery(Machine::get_machine())
              .has_capacity(width * height * sizeof(float))
              .has_affinity_to(check_processor)
              .has_affinity_to(gpu)
              .first();
```

Then we can perform a copy transfer just like we did earlier to this new CPU visible instance and launch our checking task to ensure the results are correct.  We use a `LOC_PROC` processor for this checking task as this is a CPU bound task, and there is no GPU management in this task.  It allows us to free up the `TOC_PROC` processor for more GPU work if we want to.

And that's all there is to it!

## Best Practices

To recap some best practices mentioned above:

* **Try not to call `cuda*Synchronize` APIs like `cudaDeviceSynchronize()`**.  These calls block the thread and prevent Realm from cooperatively scheduling other for that processor.  Instead, try to queue work in streams, coordinate parallel streams with CUDA events and allow the task to complete without synchronizing these streams.  Realm will ensure all the streams will synchronize prior to the task being considered complete by the application.

* **Ensure both tasks and the device work launched can make forward progress**.  Just like how tasks in Realm need to be scheduled via Realm event's wait() and the like to ensure the forward progress of the application, CUDA device code also needs this guarantee.  This guarantee can be harder to enforce with device code that needs external communication, like a flag on a local CPU, RDMA with a NIC, or even between blocks launched on the device.  Please see the [CUDA Programming Guide][1] for more information, specifically parts dealing with device-side synchronization like [CUDA Cooperative Groups][7].

* **Use `TOC_PROC` type processors when manipulating GPU state**.  These are tied to a specific GPU state, and the current device context is already set up when the task runs.

* **Consider using `LOC_PROC` processors for heavy CPU work**: `TOC_PROC` processor kinds are meant to manipulate GPU state, e.g., launching work, querying device information, etc.  These processors can be limited in CPU processing threads, or can NUMA bound to a specific NUMA node, or do heavier weight operations to ensure a consistent GPU state.  This makes `TOC_PROC` processors ill-suited for heavily CPU-bound tasks, which is why we launch a `LOC_PROC` task for our checking task in the tutorial code.

* **Pick memories and processors that have affinities to each other**.  It can achieve the best data movement performance.

* If an application does not care about some of the specific features a particular memory kind provides, try to **allow Realm to choose the best memory** to use by utilizing MemoryQuery and has_affinity_to, best_affinity_to, and has_capacity.  This allows an application to be flexible to the current state of the system as well as what the system may provide without application modifications.

* **Take into account the application's access patterns to memory**.  For example, large allocations that infrequently need to be updated between the CPU and GPU do not need to have an affinity to both the CPU and GPU.  As such memory is normally allocated in pinned system memory, this can bottleneck the GPU's access to the memory due to much lower bandwidth; however, it can also consume system memory resources such that other parts of the application (or even the entire system) can under perform.  Instead, if an allocation is primarily accessed from the CPU and is infrequently updated, it may be best just to use CPU local memory.  On the other hand, if there is a lot of GPU/CPU communication happening, using memory that has an affinity to both may justify the cost.  See section 13.1 in the [CUDA Best Practices Guide][6] for more.

* **Register pre-allocated CUDA memories with Realm with RegionInstances**.  This allows the application to leverage Realm's asynchronous programming model, data movement, and task partitioning mechanisms to its fullest

More best practices in regard to CUDA can be found in the [CUDA Best Practices Guide][6] and the [CUDA Programming Guide][1].

[1]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/ "CUDA Programming Guide"
[2]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#interprocess-communication "CUDA Legacy Interprocess Communication"
[3]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-object-api "CUDA Surface Object API"
[4]: https://en.wikipedia.org/wiki/Z-order_curve "Z-Order Curves"
[5]: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleSurfaceWrite "simpleSurfaceWrite CUDA Sample"
[6]: https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf "CUDA Best Practices Guide"
[7]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cooperative-groups "CUDA Cooperative Groups"
