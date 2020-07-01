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
#include "legion.h"
#include "cuda_runtime.h"

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
                const DeferredBuffer<Rect<N,T>,1> piece_rects,
                const DeferredBuffer<size_t,1> scan_volumes,
                const DimOrder<N> order,
                const size_t max_offset, const size_t max_rects)
    {
      size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
      if (offset >= max_offset)
        return;
      // Perform a binary search for the rectangle that we are in
      coord_t first = 0;
      coord_t last = max_rects - 1;
      coord_t mid = 0;
      while (first <= last) {
        mid = (first + last) / 2;
        if (scan_volumes[mid+1] <= offset)
          first = mid + 1;
        else if (offset < scan_volumes[mid])
          last = mid - 1;
        else
          break;
      }
      const Rect<N,T> rect = piece_rects[mid];
      Point<N,T> point = rect.lo;
      size_t pitch = 1;
      for (int i = 0; i < N; i++)
      {
        const int index = order.index[i];
        point[index] += (offset / pitch);
        offset = offset % pitch;
        pitch *= ((rect.hi[index] - rect.lo[index]) + 1);
      }
      REDOP::template fold<EXCLUSIVE>(dst[point], src[point]);
    }

    template<typename REDOP, int N, typename T, bool EXCLUSIVE>
    __global__ void
    __launch_bounds__(THREADS_PER_BLOCK,MIN_BLOCKS_PER_SM)
    apply_kernel(const Realm::AffineAccessor<typename REDOP::LHS,N,T> dst,
                 const Realm::AffineAccessor<typename REDOP::RHS,N,T> src,
                 const DeferredBuffer<Rect<N,T>,1> piece_rects,
                 const DeferredBuffer<size_t,1> scan_volumes,
                 const DimOrder<N> order,
                 const size_t max_offset, const size_t max_rects)
    {
      size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
      if (offset >= max_offset)
        return;
      // Perform a binary search for the rectangle that we are in
      int first = 0;
      int last = max_rects - 1;
      int mid = 0;
      while (first <= last) {
        mid = (first + last) / 2;
        if (scan_volumes[mid+1] <= offset)
          first = mid + 1;
        else if (offset < scan_volumes[mid])
          last = mid - 1;
        else
          break;
      }
      const Rect<N,T> rect = piece_rects[mid];
      Point<N,T> point = rect.lo;
      size_t pitch = 1;
      for (int i = 0; i < N; i++)
      {
        const int index = order.index[i];
        point[index] += (offset / pitch);
        offset = offset % pitch;
        pitch *= ((rect.hi[index] - rect.lo[index]) + 1);
      }
      REDOP::template apply<EXCLUSIVE>(dst[point], src[point]);
    } 

    template<typename REDOP>
    struct ReductionRunner {
    public:
      __host__
      ReductionRunner(const void *b, size_t s) 
        : buffer(((const char*)b)), index(0), size(s) { }
      __host__
      ~ReductionRunner(void)
      {
        assert(index == size);
      }
    public: 
      template<typename T> __host__
      inline void deserialize(T &element)
      {
        assert((index + sizeof(T)) <= size);
        element = *((const T*)(buffer+index));
        index += sizeof(T);
      }
      template<int N, typename T> __host__
      inline void run(void)
      {
        Realm::IndexSpace<N,T> space;
        deserialize(space);
        Realm::Event ready = space.make_valid();
        bool fold, exclusive;;
        deserialize<bool>(fold);
        deserialize<bool>(exclusive);
        size_t num_fields;
        deserialize(num_fields);
        std::vector<FieldID> src_fields(num_fields);
        std::vector<FieldID> dst_fields(num_fields);;
        std::vector<Realm::RegionInstance> src_insts(num_fields);
        std::vector<Realm::RegionInstance> dst_insts(num_fields);
        for (unsigned idx = 0; idx < num_fields; idx++)
        {
          deserialize(dst_insts[idx]);
          deserialize(src_insts[idx]);
          deserialize(dst_fields[idx]);
          deserialize(src_fields[idx]);
        }
        size_t num_pieces;
        deserialize(num_pieces);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        // Iterate over all the pieces
        for (unsigned pidx = 0; pidx < num_pieces; pidx++)
        {
          Rect<N,T> piece_rect;
          deserialize(piece_rect);
          std::vector<Rect<N,T> > piece_rects;
          std::vector<size_t> scan_volumes;
          size_t sum_volume = 0;
          for (Realm::IndexSpaceIterator<N,T> itr(space); itr.valid; itr.step())
          {
            const Rect<N,T> intersection = piece_rect.intersection(itr.rect);
            if (intersection.empty())
              continue;
            piece_rects.push_back(intersection);
            scan_volumes.push_back(sum_volume);
            sum_volume += intersection.volume();
          }
          if (piece_rects.empty())
            continue;
          scan_volumes.push_back(sum_volume);
          assert(scan_volumes.size() == (piece_rects.size() + 1));
          const Rect<1,coord_t> bounds(0, piece_rects.size()-1);
          const Rect<1,coord_t> scan_bounds(0, piece_rects.size());
          DeferredBuffer<Rect<N,T>,1> 
            device_piece_rects(Memory::GPU_FB_MEM, bounds);
          DeferredBuffer<size_t,1>
            device_scan_volumes(Memory::GPU_FB_MEM, scan_bounds);
          cudaMemcpyAsync(device_piece_rects.ptr(bounds), &piece_rects.front(),
              piece_rects.size() * sizeof(Rect<N,T>), cudaMemcpyHostToDevice);
          cudaMemcpyAsync(device_scan_volumes.ptr(scan_bounds), &scan_volumes.front(),
              scan_volumes.size() * sizeof(size_t), cudaMemcpyHostToDevice);
          const size_t blocks = 
                  (sum_volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
          // Iterate over all the fields we should handle
          for (unsigned fidx = 0; fidx < num_fields; fidx++)
          {
            // Make accessors for the source and destination for this piece
            const Realm::AffineAccessor<typename REDOP::RHS,N,T>
              src_accessor(src_insts[fidx], src_fields[fidx], piece_rect);
            // Compute the order of dimensions to walk based on sorting the 
            // strides for the source accessor, we'll optimistically assume 
            // the two instances are laid out the same way, if we're wrong
            // it will still be correct, just slow
            std::map<size_t,int> strides;
            for (int i = 0; i < N; i++)
            {
              std::pair<std::map<size_t,int>::iterator,bool> result = 
                strides.insert(std::pair<size_t,int>(
                      src_accessor.strides[i],i));
              // Strides should be unique across dimensions unless extent is 1
              assert(result.second || (piece_rect.hi[i] == piece_rect.lo[i]));
            }
            // Put the dimensions in order from largest to smallest
            DimOrder<N> order;
            std::map<size_t,int>::const_reverse_iterator rit = 
              strides.rbegin();
            for (int i = 0; i < N; i++, rit++)
              order.index[i] = rit->second;
            // See if we are folding or applying
            if (fold)
            {
              const Realm::AffineAccessor<typename REDOP::RHS,N,T> 
                dst_accessor(dst_insts[fidx], dst_fields[fidx], piece_rect);
              // Now launch the kernel
              if (exclusive)
                fold_kernel<REDOP,N,T,true><<<blocks,THREADS_PER_BLOCK>>>(
                    dst_accessor, src_accessor, device_piece_rects,
                    device_scan_volumes, order, sum_volume, piece_rects.size());
              else
                fold_kernel<REDOP,N,T,false><<<blocks,THREADS_PER_BLOCK>>>(
                    dst_accessor, src_accessor, device_piece_rects,
                    device_scan_volumes, order, sum_volume, piece_rects.size());
            }
            else
            {
              const Realm::AffineAccessor<typename REDOP::LHS,N,T> 
                dst_accessor(dst_insts[fidx], dst_fields[fidx], piece_rect);
              // Now launch the kernel
              if (exclusive)
                apply_kernel<REDOP,N,T,true><<<blocks,THREADS_PER_BLOCK>>>(
                    dst_accessor, src_accessor, device_piece_rects,
                    device_scan_volumes, order, sum_volume, piece_rects.size()); 
              else
                apply_kernel<REDOP,N,T,false><<<blocks,THREADS_PER_BLOCK>>>(
                    dst_accessor, src_accessor, device_piece_rects,
                    device_scan_volumes, order, sum_volume, piece_rects.size());
            }
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
      size_t index;
      size_t size;
    };

    // This is a Realm task function signature that we use for launching
    // off kernels that perform reductions between a reduction instance
    // and a normal instance on a GPU since Realm does not support this yet.
    template<typename REDOP>
    __host__ 
    void reduction_helper(const void *args, size_t arglen,
        const void *user_data,size_t user_data_size, Processor proc)
    {
      ReductionRunner<REDOP> runner(args, arglen);
      TypeTag type_tag;
      runner.deserialize(type_tag);
      NT_TemplateHelper::demux<ReductionRunner<REDOP> >(type_tag, &runner);
    }

#define REGISTER_GPU_REDUCTION_TASK(id, type)                               \
    {                                                                       \
      CodeDescriptor desc(reduction_helper<type>);                          \
      for (std::set<Processor>::const_iterator it =                         \
            gpus.begin(); it != gpus.end(); it++)                           \
        registered_events.insert(RtEvent(                                   \
              it->register_task(LG_REDOP_TASK_ID + id - LEGION_REDOP_BASE,  \
                desc, no_requests, NULL, 0)));                              \
    }

    __host__
    void register_builtin_gpu_reduction_tasks(
        const std::set<Processor> &gpus, std::set<RtEvent> &registered_events)
    {
      Realm::ProfilingRequestSet no_requests;
      // Sum Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_BOOL, SumReduction<bool>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_INT8, SumReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_INT16, SumReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_INT32, SumReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_INT64, SumReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_UINT8, SumReduction<uint8_t>);
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
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_INT8, DiffReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_INT16, DiffReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_INT32, DiffReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_INT64, DiffReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_UINT8, DiffReduction<uint8_t>);
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
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_INT8, ProdReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_INT16, ProdReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_INT32, ProdReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_INT64, ProdReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_UINT8, ProdReduction<uint8_t>);
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
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_INT8, DivReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_INT16, DivReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_INT32, DivReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_INT64, DivReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_UINT8, DivReduction<uint8_t>);
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
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_INT8, MaxReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_INT16, MaxReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_INT32, MaxReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_INT64, MaxReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_UINT8, MaxReduction<uint8_t>);
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
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_INT8, MinReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_INT16, MinReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_INT32, MinReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_INT64, MinReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_UINT8, MinReduction<uint8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_UINT16, MinReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_UINT32, MinReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_UINT64, MinReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_FLOAT16, MinReduction<__half>);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_FLOAT32, MinReduction<float>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_FLOAT64, MinReduction<double>);
      // Bitwise-OR Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_INT8, OrReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_INT16, OrReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_INT32, OrReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_INT64, OrReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_UINT8, OrReduction<uint8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_UINT16, OrReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_UINT32, OrReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_UINT64, OrReduction<uint64_t>);
      // Bitwise-AND Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_INT8, AndReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_INT16, AndReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_INT32, AndReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_INT64, AndReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_UINT8, AndReduction<uint8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_UINT16, AndReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_UINT32, AndReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_UINT64, AndReduction<uint64_t>);
      // Bitwise-XOR Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_BOOL, XorReduction<bool>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_INT8, XorReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_INT16, XorReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_INT32, XorReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_INT64, XorReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_UINT8, XorReduction<uint8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_UINT16, XorReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_UINT32, XorReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_UINT64, XorReduction<uint64_t>);
    }

  }; 
};
