/* Copyright 2020 Stanford University, NVIDIA Corporation
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

#include <map>
#include "realm.h"
#include "legion/legion_types.h"
#include "legion/legion_redop.h"

#define THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_SM 4

namespace Legion {

  namespace Internal {

    template<int N>
    struct DimOrder {
      int index[N];
    };

    template<typename REDOP, int N, typename T, bool EXCLUSIVE>
    __global__ void
    __launch_bounds__(THREADS_PER_BLOCK,MIN_BLOCKS_PER_SM)
    fold_kernel(const Realm::AffineAccessor<typename REDOP::RHS,N,T> dst,
                const Realm::AffineAccessor<typename REDOP::RHS,N,T> src,
                const DimOrder<N> order,
                const Realm::Point<N,T> lo,
                const Realm::Point<N,T> extents,
                const size_t max_offset)
    {
      size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
      if (offset >= max_offset)
        return;
      Point<N,T> point = lo;
      for (int i = 0; i < N; i++)
      {
        const int index = order.index[i];
        point[index] += (offset / extents[index]);
        offset = offset % extents[index];
      }
      REDOP::template fold<EXCLUSIVE>(dst[point], src[point]);
    }

    template<typename REDOP, int N, typename T, bool EXCLUSIVE>
    __global__ void
    __launch_bounds__(THREADS_PER_BLOCK,MIN_BLOCKS_PER_SM)
    apply_kernel(const Realm::AffineAccessor<typename REDOP::LHS,N,T> dst,
                 const Realm::AffineAccessor<typename REDOP::RHS,N,T> src,
                 const DimOrder<N> order,
                 const Realm::Point<N,T> lo,
                 const Realm::Point<N,T> extents,
                 const size_t max_offset)
    {
      size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
      if (offset >= max_offset)
        return;
      Point<N,T> point = lo;
      for (int i = 0; i < N; i++)
      {
        const int index = order.index[i];
        point[index] += (offset / extents[index]);
        offset = offset % extents[index];
      }
      REDOP::template apply<EXCLUSIVE>(dst[point], src[point]);
    } 

    template<typename REDOP>
    struct ReductionRunner {
    public:
      __host__
      ReductionRunner(const char *b) : buffer(b) { }
    public: 
      template<int N, typename T>
      __host__
      inline void run(void)
      {
        const Realm::IndexSpace<N,T> space = 
          *((const Realm::IndexSpace<N,T>*)buffer);
        buffer += sizeof(space);
        const FieldID field_id = *((const FieldID*)buffer);
        buffer += sizeof(field_id);
        const Realm::RegionInstance dst = 
          *((const Realm::RegionInstance*)buffer);
        buffer += sizeof(dst);
        const Realm::RegionInstance src = 
          *((const Realm::RegionInstance*)buffer);
        buffer += sizeof(src);
        const bool exclusive = *((const bool*)buffer);
        buffer += sizeof(exclusive);
        const bool fold = *((const bool*)buffer);

        const Realm::AffineAccessor<typename REDOP::RHS,N,T> 
            src_accessor(src, field_id, space.bounds);
        // Compute the order of dimensions to walk based on sorting the 
        // strides for the source accessor, we'll optimistically assume 
        // the two instances are laid out the same way, if we're wrong
        // it will still be correct, just slow
        std::map<ptrdiff_t,int> strides;
        for (int i = 0; i < N; i++)
        {
          std::pair<std::map<ptrdiff_t,int>::iterator,bool> result = 
            strides.insert(std::pair<ptrdiff_t,int>(src_accessor.strides[i],i));
          // Strides should be unique across dimensions unless the extent is 1
          assert(result.second || (space.bounds.hi[i] == space.bounds.lo[i]));
        }
        // Put the dimensions in order from largest to smallest
        DimOrder<N> order;
        std::map<ptrdiff_t,int>::const_reverse_iterator rit = strides.rbegin();
        for (int i = 0; i < N; i++, rit++)
          order.index[i] = rit->second;
        
        if (fold)
        {
          const Realm::AffineAccessor<typename REDOP::RHS,N,T> 
            dst_accessor(dst, field_id, space.bounds);
          if (!space.dense()) 
          {
            // The index space is not dense so launch a kernel for each rect
            Realm::IndexSpaceIterator<N,T> iterator(space);
            while (true)
            {
              const size_t volume = iterator.rect.volume();
              const size_t blocks = 
                (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
              Point<N,T> extents;
              size_t pitch = 1;
              for (std::map<ptrdiff_t,int>::const_iterator it = 
                    strides.begin(); it != strides.end(); it++)
              {
                extents[it->second] = pitch;
                pitch *= ((iterator.rect.hi[it->second] - 
                            iterator.rect.lo[it->second]) + 1);
              }
              if (exclusive)
                fold_kernel<REDOP,N,T,true><<<blocks,THREADS_PER_BLOCK>>>(
                    dst_accessor, src_accessor, order, 
                    iterator.rect.lo, extents, volume); 
              else
                fold_kernel<REDOP,N,T,false><<<blocks,THREADS_PER_BLOCK>>>(
                    dst_accessor, src_accessor, order, 
                    iterator.rect.lo, extents, volume);
              if (!iterator.step())
                break;
            }
          }
          else
          {
            // Space is dense so we just need a single kernel launch here
            const size_t volume = space.bounds.volume();
            const size_t blocks = 
              (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            Point<N,T> extents;
            size_t pitch = 1;
            for (std::map<ptrdiff_t,int>::const_iterator it = 
                  strides.begin(); it != strides.end(); it++)
            {
              extents[it->second] = pitch;
              pitch *= ((space.bounds.hi[it->second] - 
                          space.bounds.lo[it->second]) + 1);
            }
            if (exclusive)
              fold_kernel<REDOP,N,T,true><<<blocks,THREADS_PER_BLOCK>>>(
                  dst_accessor, src_accessor, order, 
                  space.bounds.lo, extents, volume);
            else
              fold_kernel<REDOP,N,T,false><<<blocks,THREADS_PER_BLOCK>>>(
                  dst_accessor, src_accessor, order, 
                  space.bounds.lo, extents, volume);
          }
        }
        else
        {
          const Realm::AffineAccessor<typename REDOP::LHS,N,T> 
            dst_accessor(dst, field_id, space.bounds);
          if (!space.dense()) 
          {
            // The index space is not dense so launch a kernel for each rect
            Realm::IndexSpaceIterator<N,T> iterator(space);
            while (true)
            {
              const size_t volume = iterator.rect.volume();
              const size_t blocks = 
                (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
              Point<N,T> extents;
              size_t pitch = 1;
              for (std::map<ptrdiff_t,int>::const_iterator it = 
                    strides.begin(); it != strides.end(); it++)
              {
                extents[it->second] = pitch;
                pitch *= ((iterator.rect.hi[it->second] - 
                            iterator.rect.lo[it->second]) + 1);
              }
              if (exclusive)
                apply_kernel<REDOP,N,T,true><<<blocks,THREADS_PER_BLOCK>>>(
                    dst_accessor, src_accessor, order, 
                    iterator.rect.lo, extents, volume); 
              else
                apply_kernel<REDOP,N,T,false><<<blocks,THREADS_PER_BLOCK>>>(
                    dst_accessor, src_accessor, order, 
                    iterator.rect.lo, extents, volume);
              if (!iterator.step())
                break;
            }
          }
          else
          {
            // Space is dense so we just need a single kernel launch here
            const size_t volume = space.bounds.volume();
            const size_t blocks = 
              (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            Point<N,T> extents;
            size_t pitch = 1;
            for (std::map<ptrdiff_t,int>::const_iterator it = 
                  strides.begin(); it != strides.end(); it++)
            {
              extents[it->second] = pitch;
              pitch *= ((space.bounds.hi[it->second] - 
                          space.bounds.lo[it->second]) + 1);
            }
            if (exclusive)
              apply_kernel<REDOP,N,T,true><<<blocks,THREADS_PER_BLOCK>>>(
                  dst_accessor, src_accessor, order, 
                  space.bounds.lo, extents, volume);
            else
              apply_kernel<REDOP,N,T,false><<<blocks,THREADS_PER_BLOCK>>>(
                  dst_accessor, src_accessor, order, 
                  space.bounds.lo, extents, volume);
          }
        }
      }

      template<typename N, typename T>
      __host__
      static inline void demux(ReductionRunner<REDOP> *runner)
      {
        runner->run<N::N,T>();
      }
    public:
      const char *buffer;
    };

    // This is a Realm task function signature that we use for launching
    // off kernels that perform reductions between a reduction instance
    // and a normal instance on a GPU since Realm does not support this yet.
    template<typename REDOP>
    __host__ 
    void reduction_helper(const void *args, size_t arglen,
        const void *user_data,size_t user_data_size, Processor proc)
    {
      const char *buffer = (const char*)args;
      const TypeTag type_tag = *((const TypeTag*)buffer);
      buffer += sizeof(type_tag);
      ReductionRunner<REDOP> runner(buffer);
      NT_TemplateHelper::demux<ReductionRunner<REDOP> >(type_tag, &runner);
    }

#define REGISTER_GPU_REDUCTION_TASK(id, type)                 \
    {                                                         \
      CodeDescriptor desc(reduction_helper<type>);            \
      for (std::set<Processor>::const_iterator it =           \
            gpus.begin(); it != gpus.end(); it++)             \
        registered_events.insert(RtEvent(                     \
              it->register_task(LG_REDOP_TASK_ID + id, desc,  \
              no_requests, NULL, 0)));                        \
    }

    __host__
    void register_builtin_gpu_reduction_tasks(
        const std::set<Processor> &gpus, std::set<RtEvent> &registered_events)
    {
      Realm::ProfilingRequestSet no_requests;
      // Sum Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_BOOL, SumReduction<bool>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_INT16, SumReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_INT32, SumReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_INT64, SumReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_UINT16, SumReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_UINT32, SumReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_UINT64, SumReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_FLOAT16, SumReduction<__half>);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_FLOAT32, SumReduction<float>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_FLOAT64, SumReduction<double>);
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_COMPLEX32, SumReduction<complex<__half> >);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_COMPLEX64, SumReduction<complex<float> >);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_COMPLEX128, SumReduction<complex<double> >);
#endif
      // Difference Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_INT16, DiffReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_INT32, DiffReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_INT64, DiffReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_UINT16, DiffReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_UINT32, DiffReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_UINT64, DiffReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_FLOAT16, DiffReduction<__half>);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_FLOAT32, DiffReduction<float>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_FLOAT64, DiffReduction<double>);
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_COMPLEX32, DiffReduction<complex<__half> >);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_COMPLEX64, DiffReduction<complex<float> >);
#endif
      // Product Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_BOOL, ProdReduction<bool>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_INT16, ProdReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_INT32, ProdReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_INT64, ProdReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_UINT16, ProdReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_UINT32, ProdReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_UINT64, ProdReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_FLOAT16, ProdReduction<__half>);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_FLOAT32, ProdReduction<float>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_FLOAT64, ProdReduction<double>);
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_COMPLEX32, ProdReduction<complex<__half> >);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_COMPLEX64, ProdReduction<complex<float> >);
#endif
      // Divide Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_INT16, DivReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_INT32, DivReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_INT64, DivReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_UINT16, DivReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_UINT32, DivReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_UINT64, DivReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_FLOAT16, DivReduction<__half>);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_FLOAT32, DivReduction<float>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_FLOAT64, DivReduction<double>);
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_COMPLEX32, DivReduction<complex<__half> >);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_COMPLEX64, DivReduction<complex<float> >);
#endif
      // Max Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_BOOL, MaxReduction<bool>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_INT16, MaxReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_INT32, MaxReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_INT64, MaxReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_UINT16, MaxReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_UINT32, MaxReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_UINT64, MaxReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_FLOAT16, MaxReduction<__half>);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_FLOAT32, MaxReduction<float>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_FLOAT64, MaxReduction<double>);
      // Min Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_BOOL, MinReduction<bool>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_INT16, MinReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_INT32, MinReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_INT64, MinReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_UINT16, MinReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_UINT32, MinReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_UINT64, MinReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_FLOAT16, MinReduction<__half>);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_FLOAT32, MinReduction<float>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_FLOAT64, MinReduction<double>);
    }

  }; 
};
