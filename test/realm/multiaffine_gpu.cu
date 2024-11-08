#include "realm.h"
#include "multiaffine.h"

#ifdef REALM_USE_HIP
#include "hip_cuda_compat/hip_cuda.h"
#include "realm/hip/hiphijack_api.h"
#endif


using namespace Realm;

template <typename T>
__device__ Point<1,T> choose_thread_point(Rect<1,T> bounds)
{
  return Point<1,T>(bounds.lo[0] + (blockIdx.x * blockDim.x) + threadIdx.x);
}

template <typename T>
__device__ Point<2,T> choose_thread_point(Rect<2,T> bounds)
{
  return Point<2,T>(bounds.lo[0] + (blockIdx.x * blockDim.x) + threadIdx.x,
		    bounds.lo[1] + (blockIdx.y * blockDim.y) + threadIdx.y);
}

template <int N, typename T>
__global__ void ptr_write_task_kernel(IndexSpace<N,T> space,
				      MultiAffineAccessor<void *,N,T> acc)
{
  Point<N,T> p = choose_thread_point(space.bounds);
  if(space.bounds.contains(p)) {
     void **ptr = acc.ptr(p);
     *ptr = ptr+1;
  }
}

template <int N, typename T>
void ptr_write_task_gpu(const void *args, size_t arglen,
			const void *userdata, size_t userlen, Processor p)
{
  const PtrWriteTaskArgs<N,T>& targs = *static_cast<const PtrWriteTaskArgs<N,T> *>(args);

  MultiAffineAccessor<void *,N,T> acc(targs.inst, FID_ADDR);

  int bx, by, bz, tx, ty, tz;
  int sx = targs.space.bounds.hi[0] - targs.space.bounds.lo[0] + 1;
  tx = std::min(1024, sx);
  bx = ((sx - 1) / tx) + 1;
  if(N > 1) {
    int sy = targs.space.bounds.hi[1] - targs.space.bounds.lo[1] + 1;
    ty = std::min(1024 / tx, sy);
    by = ((sy - 1) / ty) + 1;
  } else
    ty = by = 1;
  if(N > 2) {
    int sz = targs.space.bounds.hi[2] - targs.space.bounds.lo[2] + 1;
    tz = std::min(1024 / (tx * ty), sz);
    bz = ((sz - 1) / tz) + 1;
  } else
    tz = bz = 1;

  dim3 grid_dim(bx, by, bz);
  dim3 blk_dim(tx, ty, tz);
  ptr_write_task_kernel<<<grid_dim, blk_dim
#ifdef REALM_USE_HIP
                          , 0, hipGetTaskStream()
#endif
                       >>>(targs.space, acc);
}

void register_multiaffine_gpu_tasks()
{
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
				   PTR_WRITE_TASK_BASE + 1,
				   CodeDescriptor(ptr_write_task_gpu<1,int>),
				   ProfilingRequestSet(),
				   0, 0).wait();
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
				   PTR_WRITE_TASK_BASE + 2,
				   CodeDescriptor(ptr_write_task_gpu<2,int>),
				   ProfilingRequestSet(),
				   0, 0).wait();
}
