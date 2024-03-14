#include "realm.h"
#include "realm/id.h"
#include "realm/cmdline.h"
#include "realm/cuda/cuda_access.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <csignal>
#include <cmath>

#include <time.h>

#include "osdep.h"

#include "cuda_arrays.h"

#include <cuda_runtime.h>

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  GPU_TASK_BASE,
};

enum {
  FID_INT = 44,
  FID_DOUBLE = 88,
  FID_FLOAT = 99,
};

namespace TestConfig {
  int size = 10;  // extent in each dimension
  unsigned dim_mask = 7; // i.e. 1-D, 2-D, 3-D
  unsigned bloat_mask = 0;
};

// cuda arrays need to be created with a cuda context resident, so run them as
//  tasks on the gpu
template <int N, typename T>
struct GPUTaskArgs {
  RegionInstance src_inst;
  Rect<N,T> extent;
  float alpha;
};

template <typename T>
static cudaExtent rect_to_extent(Rect<1, T> r)
{
  return make_cudaExtent(r.hi.x - r.lo.x + 1, 0, 0);
}

template <typename T>
static cudaExtent rect_to_extent(Rect<2, T> r)
{
  return make_cudaExtent(r.hi.x - r.lo.x + 1,
                         r.hi.y - r.lo.y + 1, 0);
}

template <typename T>
static cudaExtent rect_to_extent(Rect<3, T> r)
{
  return make_cudaExtent(r.hi.x - r.lo.x + 1,
                         r.hi.y - r.lo.y + 1,
                         r.hi.z - r.lo.z + 1);
}

template <int N, typename T>
void gpu_task(const void *data, size_t datalen,
              const void *userdata, size_t userlen, Processor p)
{
  assert(datalen == sizeof(GPUTaskArgs<N,T>));
  const GPUTaskArgs<N,T>& args = *static_cast<const GPUTaskArgs<N,T> *>(data);

  // create a couple of cuArrays and an (external) linear layout version
  cudaChannelFormatDesc fmt = cudaCreateChannelDesc(32, 0, 0, 0,
                                                    cudaChannelFormatKindFloat);
  cudaArray_t arr1, arr2;
  CHECK_CUDART( cudaMalloc3DArray(&arr1, &fmt, rect_to_extent(args.extent),
                                  cudaArraySurfaceLoadStore) );
  CHECK_CUDART( cudaMalloc3DArray(&arr2, &fmt, rect_to_extent(args.extent),
                                  cudaArraySurfaceLoadStore) );

  size_t elems = args.extent.volume();
  void *linear;
  CHECK_CUDART( cudaMalloc(&linear, elems * sizeof(float)) );

  // create instances to describe these two arrays
  int devid;
  CHECK_CUDART( cudaGetDevice(&devid) );

#define USE_EXT_INSTS
#ifdef USE_EXT_INSTS
  RegionInstance arr_inst1, arr_inst2, linear_inst;
  {
    InstanceLayout<N,T> layout;
    layout.space = args.extent;
    InstanceLayoutGeneric::FieldLayout& fl = layout.fields[FID_FLOAT];
    fl.list_idx = 0;
    fl.rel_offset = 0;
    fl.size_in_bytes = sizeof(float);
    layout.piece_lists.resize(1);
    CudaArrayLayoutPiece<N,T> *clp = new CudaArrayLayoutPiece<N,T>;
    clp->bounds = args.extent;
    layout.piece_lists[0].pieces.push_back(clp);
    {
      ExternalCudaArrayResource external(devid, arr1);
      RegionInstance::create_external_instance(arr_inst1,
                                               external.suggested_memory(),
                                               layout.clone(),
                                               external,
                                               ProfilingRequestSet()).wait();
    }
    {
      ExternalCudaArrayResource external(devid, arr2);
      RegionInstance::create_external_instance(arr_inst2,
                                               external.suggested_memory(),
                                               layout.clone(),
                                               external,
                                               ProfilingRequestSet()).wait();
    }
  }
  {
    std::map<FieldID, size_t> field_sizes;
    field_sizes[FID_FLOAT] = sizeof(float);
    InstanceLayoutConstraints ilc(field_sizes, 1);
    int dim_order[N];
    for(int i = 0; i < N; i++) dim_order[i] = i;
    InstanceLayoutGeneric *ilg = InstanceLayoutGeneric::choose_instance_layout<N,T>(args.extent, ilc, dim_order);
    ExternalCudaMemoryResource external(devid, linear, elems * sizeof(float));
    RegionInstance::create_external_instance(linear_inst,
                                             external.suggested_memory(),
                                             ilg,
                                             external,
                                             ProfilingRequestSet()).wait();
  }
#endif

  cudaSurfaceObject_t surf1, surf2;
  cudaResourceDesc desc;
  memset(&desc, 0, sizeof(desc));
  desc.resType = cudaResourceTypeArray;
  desc.res.array.array = arr1;
  CHECK_CUDART( cudaCreateSurfaceObject(&surf1, &desc) );
  desc.res.array.array = arr2;
  CHECK_CUDART( cudaCreateSurfaceObject(&surf2, &desc) );

#ifdef USE_EXT_INSTS
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_field(args.src_inst, FID_FLOAT, sizeof(float));
    dsts[0].set_field(linear_inst, FID_FLOAT, sizeof(float));
    args.extent.copy(srcs, dsts, ProfilingRequestSet()).wait();

    srcs[0].set_field(linear_inst, FID_FLOAT, sizeof(float));
    dsts[0].set_field(arr_inst1, FID_FLOAT, sizeof(float));
    args.extent.copy(srcs, dsts, ProfilingRequestSet()).wait();
  }
#else
  AffineAccessor<float,N,T> acc(args.src_inst, FID_FLOAT);

  uintptr_t l0, l1;
  {
    Point<N,T> p = args.extent.lo;
    l0 = reinterpret_cast<uintptr_t>(&acc[p]);
    p[1]++;
    l1 = reinterpret_cast<uintptr_t>(&acc[p]);
  }
  CHECK_CUDART( cudaMemcpy2DToArray(arr1, 0, 0,
                                    reinterpret_cast<const void *>(l0),
                                    l1 - l0,
                                    4*(args.extent.hi[0] - args.extent.lo[0] + 1),
                                    args.extent.hi[1] - args.extent.lo[1] + 1,
                                    cudaMemcpyHostToDevice) );
#endif

  smooth_kernel(args.extent, args.alpha, surf1, surf2);

#ifdef USE_EXT_INSTS
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_fill<float>(-1.0f);
    dsts[0].set_field(args.src_inst, FID_FLOAT, sizeof(float));
    args.extent.copy(srcs, dsts, ProfilingRequestSet()).wait();

    srcs[0].set_field(arr_inst2, FID_FLOAT, sizeof(float));
    dsts[0].set_field(args.src_inst, FID_FLOAT, sizeof(float));
    args.extent.copy(srcs, dsts, ProfilingRequestSet()).wait();

    arr_inst1.destroy();
    arr_inst2.destroy();
    linear_inst.destroy();
  }
#else
  for(PointInRectIterator<N,T> pir(args.extent); pir.valid; pir.step()) {
    acc[pir.p] = -1;
  }
  CHECK_CUDART( cudaMemcpy2DFromArray(reinterpret_cast<void *>(l0),
                                      l1 - l0,
                                      arr2, 0, 0,
                                      4*(args.extent.hi[0] - args.extent.lo[0] + 1),
                                      args.extent.hi[1] - args.extent.lo[1] + 1,
                                      cudaMemcpyDeviceToHost) );
  for(PointInRectIterator<N,T> pir(args.extent); pir.valid; pir.step()) {
    float f = acc[pir.p];
    log_app.print() << "result[" << pir.p << "] = " << f;
  }
#endif  

  CHECK_CUDART( cudaDestroySurfaceObject(surf1) );
  CHECK_CUDART( cudaDestroySurfaceObject(surf2) );

  CHECK_CUDART( cudaFreeArray(arr1) );
  CHECK_CUDART( cudaFreeArray(arr2) );
  CHECK_CUDART( cudaFree(linear) );
}

template <int N, typename T>
bool test_cuda_arrays(Rect<N,T> extent, Memory src_mem,
                      Processor tgt_proc, Memory tgt_mem)
{
  RegionInstance src_data;
  {
    int dim_order[N];
    for(int i = 0; i < N; i++) dim_order[i] = i;
    std::map<FieldID, size_t> field_sizes;
    field_sizes[FID_FLOAT] = sizeof(float);
    Rect<N,T> bloated = extent;
    if((TestConfig::bloat_mask & 1) != 0)
      bloated.hi[0] += 4;
    if((N>1) && ((TestConfig::bloat_mask & 2) != 0))
      bloated.hi[1] += 5;
    if((N>2) && ((TestConfig::bloat_mask & 4) != 0))
      bloated.hi[2] += 6;
    InstanceLayoutGeneric *layout =
      InstanceLayoutGeneric::choose_instance_layout(IndexSpace<N,T>(bloated),
                                                    InstanceLayoutConstraints(field_sizes, 0),
                                                    dim_order);
    RegionInstance::create_instance(src_data, src_mem, layout,
                                    ProfilingRequestSet()).wait();
  }

  {
    AffineAccessor<float,N,T> acc(src_data, FID_FLOAT);
    float f = 1.0;
    for(PointInRectIterator<N,T> pir(extent); pir.valid; pir.step()) {
      acc[pir.p] = f;
      f += 1.0;
    }
  }

  float alpha = 0.5f;
  {
    GPUTaskArgs<N,T> args;
    args.src_inst = src_data;
    args.extent = extent;
    args.alpha = alpha;

    Event e = tgt_proc.spawn(GPU_TASK_BASE + N, &args, sizeof(args));
    e.wait();
  }

  int errors = 0;
  {
    AffineAccessor<float,N,T> acc(src_data, FID_FLOAT);
    float f = 1.0;
    for(PointInRectIterator<N,T> pir(extent); pir.valid; pir.step()) {
      // expected value is the same as the original in the interior but
      //  changes on boundaries
      float exp = f;
      float adj = alpha / 6.0f;
      for(int i = 0; i < N; i++) {
        if(pir.p[i] == extent.lo[i])
          exp += adj;
        if(pir.p[i] == extent.hi[i])
          exp -= adj;
        adj *= (extent.hi[i] - extent.lo[i] + 1);
      }

      float act = acc[pir.p];
      if(fabsf(exp - act) < 1e-3 * exp) {
        // all good
      } else {
        // mismatch
        if(errors++ < 10)
          log_app.error() << "mismatch at " << pir.p << ": exp=" << exp << " act=" << act << " err=" << fabsf(exp - act);
      }

      f += 1.0f;
    }
  }

  src_data.destroy();

  return (errors == 0);
}

void top_level_task(const void *data, size_t datalen,
                    const void *userdata, size_t userlen, Processor p)
{
  // source memory should be one we can access
  Memory src_mem = Machine::MemoryQuery(Machine::get_machine()).has_affinity_to(p).first();
  assert(src_mem.exists());

  // target should be the last gpu we can find (it'll be remote if we're
  //  multi-node)
  Machine::ProcessorQuery pq(Machine::get_machine());
  pq.only_kind(Processor::TOC_PROC);
  std::vector<Processor> gpus(pq.begin(), pq.end());
  assert(!gpus.empty());

  Processor tgt_proc = gpus[gpus.size() - 1];
  Memory tgt_mem = Machine::MemoryQuery(Machine::get_machine()).best_affinity_to(tgt_proc).first();
  assert(tgt_mem.exists());

  bool ok = true;

  if(TestConfig::dim_mask & 1) {
    // 1-D
    Rect<1> r(0, TestConfig::size-1);

    if(ok) ok = test_cuda_arrays(r, src_mem, tgt_proc, tgt_mem);
  }

  if(TestConfig::dim_mask & 2) {
    // 2-D
    Rect<2> r(Point<2>(0, 0), Point<2>(TestConfig::size-1, TestConfig::size-1));

    if(ok) ok = test_cuda_arrays(r, src_mem, tgt_proc, tgt_mem);
  }

  if(TestConfig::dim_mask & 4) {
    // 3-D
    Rect<3> r(Point<3>(0, 0, 0),
              Point<3>(TestConfig::size-1, TestConfig::size-1, TestConfig::size-1));

    if(ok) ok = test_cuda_arrays(r, src_mem, tgt_proc, tgt_mem);
  }

  Runtime::get_runtime().shutdown(Event::NO_EVENT,
				  ok ? 0 : 1);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int("-size", TestConfig::size)
    .add_option_int("-dims", TestConfig::dim_mask)
    .add_option_int("-bloat", TestConfig::bloat_mask);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                   GPU_TASK_BASE + 1,
				   CodeDescriptor(gpu_task<1,int>),
				   ProfilingRequestSet(),
				   0, 0).wait();
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                   GPU_TASK_BASE + 2,
				   CodeDescriptor(gpu_task<2,int>),
				   ProfilingRequestSet(),
				   0, 0).wait();
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                   GPU_TASK_BASE + 3,
				   CodeDescriptor(gpu_task<3,int>),
				   ProfilingRequestSet(),
				   0, 0).wait();

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  assert(p.exists());

  // collective launch of a single task - top level will issue shutdown
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // now sleep this thread until that shutdown actually happens
  int result = rt.wait_for_shutdown();
  
  return result;
}
