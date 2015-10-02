/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#include "cuda_module.h"

#include "realm/tasks.h"
#include "realm/logging.h"
#include "realm/cmdline.h"

#include "lowlevel_dma.h"

#include "realm/cuda/cudart_hijack.h"

#include <stdio.h>

namespace Realm {
  namespace Cuda {

    // dma code is still in old namespace
    typedef LegionRuntime::LowLevel::DmaRequest DmaRequest;
    typedef LegionRuntime::LowLevel::OASVec OASVec;
    typedef LegionRuntime::LowLevel::InstPairCopier InstPairCopier;
    typedef LegionRuntime::LowLevel::MemPairCopier MemPairCopier;
    typedef LegionRuntime::LowLevel::MemPairCopierFactory MemPairCopierFactory;

    Logger log_gpu("gpu");
    Logger log_gpudma("gpudma");

#ifdef EVENT_GRAPH_TRACE
    extern Logger log_event_graph;
#endif
    Logger log_stream("gpustream");

    class stringbuilder {
    public:
      operator std::string(void) const { return ss.str(); }
      template <typename T>
      stringbuilder& operator<<(T data) { ss << data; return *this; }
    protected:
      std::stringstream ss;
    };


  ////////////////////////////////////////////////////////////////////////
  //
  // class GPUStream

    GPUStream::GPUStream(GPU *_gpu, GPUWorker *_worker)
      : gpu(_gpu), worker(_worker)
    {
      assert(worker != 0);
      CHECK_CU( cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) );
      log_stream.info() << "CUDA stream " << stream << " created for GPU " << gpu;
    }

    GPUStream::~GPUStream(void)
    {
      // log_stream.info() << "CUDA stream " << stream << " destroyed - max copies = " 
      // 			<< pending_copies.capacity() << ", max events = " << pending_events.capacity();

      CHECK_CU( cuStreamDestroy(stream) );
    }

    GPU *GPUStream::get_gpu(void) const
    {
      return gpu;
    }
    
    CUstream GPUStream::get_stream(void) const
    {
      return stream;
    }

    // may be called by anybody to enqueue a copy or an event
    void GPUStream::add_copy(GPUMemcpy *copy)
    {
      bool add_to_worker = false;
      {
	AutoHSLLock al(mutex);

	// remember to add ourselves to the worker if we didn't already have work
	add_to_worker = pending_copies.empty();

	pending_copies.push_back(copy);
      }

      if(add_to_worker)
	worker->add_stream(this);
    }

    void GPUStream::add_fence(GPUWorkFence *fence)
    {
      CUevent e = gpu->event_pool.get_event();

      CHECK_CU( cuEventRecord(e, stream) );

      log_stream.debug() << "CUDA event " << e << " recorded on stream " << stream << " (GPU " << gpu << ")";

      add_event(e, fence, 0);
    }

    void GPUStream::add_notification(GPUCompletionNotification *notification)
    {
      CUevent e = gpu->event_pool.get_event();

      CHECK_CU( cuEventRecord(e, stream) );

      add_event(e, 0, notification);
    }

    void GPUStream::add_event(CUevent event, GPUWorkFence *fence, 
			      GPUCompletionNotification *notification)
    {
      bool add_to_worker = false;
      {
	AutoHSLLock al(mutex);

	// remember to add ourselves to the worker if we didn't already have work
	add_to_worker = pending_events.empty();

	PendingEvent e;
	e.event = event;
	e.fence = fence;
	e.notification = notification;

	pending_events.push_back(e);
      }

      if(add_to_worker)
	worker->add_stream(this);
    }

    // to be called by a worker (that should already have the GPU context
    //   current) - returns true if any work remains
    bool GPUStream::issue_copies(void)
    {
      while(true) {
	GPUMemcpy *copy = 0;
	{
	  AutoHSLLock al(mutex);

	  if(pending_copies.empty())
	    return false;  // no work left

	  copy = pending_copies.front();
	  pending_copies.pop_front();
	}

	copy->execute(this);

	// no backpressure on copies yet - keep going until list is empty
      }
    }

    bool GPUStream::reap_events(void)
    {
      // peek at the first event
      CUevent event;
      bool event_valid = false;
      {
	AutoHSLLock al(mutex);

	if(pending_events.empty())
	  return false;  // no work left

	event = pending_events.front().event;
	event_valid = true;
      }

      // we'll keep looking at events until we find one that hasn't triggered
      while(event_valid) {
	CUresult res = cuEventQuery(event);

	if(res == CUDA_ERROR_NOT_READY)
	  return true; // oldest event hasn't triggered - check again later

	// no other kind of error is expected
	assert(res == CUDA_SUCCESS);

	log_stream.debug() << "CUDA event " << event << " triggered on stream " << stream << " (GPU " << gpu << ")";

	// give event back to GPU for reuse
	gpu->event_pool.return_event(event);

	// this event has triggered, so figure out the fence/notification to trigger
	//  and also peek at the next event
	GPUWorkFence *fence = 0;
	GPUCompletionNotification *notification = 0;

	{
	  AutoHSLLock al(mutex);

	  const PendingEvent &e = pending_events.front();
	  assert(e.event == event);
	  fence = e.fence;
	  notification = e.notification;
	  pending_events.pop_front();

	  if(pending_events.empty())
	    event_valid = false;
	  else
	    event = pending_events.front().event;
	}

	if(fence)
	  fence->mark_finished();

	if(notification)
	  notification->request_completed();
      }

      // if we get all the way to here, we're (temporarily, at least) out of work
      return false;
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GPUMemcpy

    GPUMemcpy::GPUMemcpy(GPU *_gpu, GPUMemcpyKind _kind)
      : gpu(_gpu), kind(_kind)
    {} 


  ////////////////////////////////////////////////////////////////////////
  //
  // class GPUMemcpy1D

    GPUMemcpy1D::GPUMemcpy1D(GPU *_gpu,
			     void *_dst, const void *_src, size_t _bytes, GPUMemcpyKind _kind,
			     GPUCompletionNotification *_notification)
      : GPUMemcpy(_gpu, _kind), dst(_dst), src(_src), 
	mask(0), elmt_size(_bytes), notification(_notification)
    {}

    GPUMemcpy1D::GPUMemcpy1D(GPU *_gpu,
			     void *_dst, const void *_src, 
			     const ElementMask *_mask, size_t _elmt_size,
			     GPUMemcpyKind _kind,
			     GPUCompletionNotification *_notification)
      : GPUMemcpy(_gpu, _kind), dst(_dst), src(_src),
	mask(_mask), elmt_size(_elmt_size), notification(_notification)
    {}

    GPUMemcpy1D::~GPUMemcpy1D(void)
    {}

    void GPUMemcpy1D::do_span(off_t pos, size_t len)
    {
      off_t span_start = pos * elmt_size;
      size_t span_bytes = len * elmt_size;

      switch (kind)
      {
        case GPU_MEMCPY_HOST_TO_DEVICE:
          {
            CHECK_CU( cuMemcpyHtoDAsync((CUdeviceptr)(((char*)dst)+span_start),
                                        (((char*)src)+span_start),
                                        span_bytes,
                                        local_stream->get_stream()) );
            break;
          }
        case GPU_MEMCPY_DEVICE_TO_HOST:
          {
            CHECK_CU( cuMemcpyDtoHAsync((((char*)dst)+span_start),
                                        (CUdeviceptr)(((char*)src)+span_start),
                                        span_bytes,
                                        local_stream->get_stream()) );
            break;
          }
        case GPU_MEMCPY_DEVICE_TO_DEVICE:
          {
            CHECK_CU( cuMemcpyDtoDAsync((CUdeviceptr)(((char*)dst)+span_start),
                                        (CUdeviceptr)(((char*)src)+span_start),
                                        span_bytes,
                                        local_stream->get_stream()) );
            break;
          }
        case GPU_MEMCPY_PEER_TO_PEER:
          {
            CUcontext src_ctx, dst_ctx;
            CHECK_CU( cuPointerGetAttribute(&src_ctx, 
                  CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr)src) );
            CHECK_CU( cuPointerGetAttribute(&dst_ctx, 
                  CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr)dst) );
            CHECK_CU( cuMemcpyPeerAsync((CUdeviceptr)(((char*)dst)+span_start), dst_ctx,
                                        (CUdeviceptr)(((char*)src)+span_start), src_ctx,
                                        span_bytes,
                                        local_stream->get_stream()) );
            break;
          }
        default:
          assert(false);
      }
    }

    void GPUMemcpy1D::execute(GPUStream *stream)
    {
      DetailedTimer::ScopedPush sp(TIME_COPY);
      log_gpudma.info("gpu memcpy: dst=%p src=%p bytes=%zd kind=%d",
                   dst, src, elmt_size, kind);
      // save stream into local variable for do_spam (which may be called indirectly
      //  by ElementMask::forall_ranges)
      local_stream = stream;
      if(mask) {
        ElementMask::forall_ranges(*this, *mask);
      } else {
        do_span(0, 1);
      }
      
      if(notification)
	stream->add_notification(notification);

      log_gpudma.info("gpu memcpy complete: dst=%p src=%p bytes=%zd kind=%d",
                   dst, src, elmt_size, kind);
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GPUMemcpy2D

    GPUMemcpy2D::GPUMemcpy2D(GPU *_gpu,
			     void *_dst, const void *_src,
			     off_t _dst_stride, off_t _src_stride,
			     size_t _bytes, size_t _lines,
			     GPUMemcpyKind _kind,
			     GPUCompletionNotification *_notification)
      : GPUMemcpy(_gpu, _kind), dst(_dst), src(_src),
	dst_stride((_dst_stride < (off_t)_bytes) ? _bytes : _dst_stride), 
	src_stride((_src_stride < (off_t)_bytes) ? _bytes : _src_stride),
	bytes(_bytes), lines(_lines), notification(_notification)
    {}

    GPUMemcpy2D::~GPUMemcpy2D(void)
    {}

    void GPUMemcpy2D::execute(GPUStream *stream)
    {
      log_gpudma.info("gpu memcpy 2d: dst=%p src=%p "
                   "dst_off=%ld src_off=%ld bytes=%ld lines=%ld kind=%d",
                   dst, src, (long)dst_stride, (long)src_stride, bytes, lines, kind); 
      CUDA_MEMCPY2D copy_info;
      if (kind == GPU_MEMCPY_PEER_TO_PEER) {
        // If we're doing peer to peer, just let unified memory it deal with it
        copy_info.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
        copy_info.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
      } else {
        // otherwise we know the answers here 
        copy_info.srcMemoryType = (kind == GPU_MEMCPY_HOST_TO_DEVICE) ?
          CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE;
        copy_info.dstMemoryType = (kind == GPU_MEMCPY_DEVICE_TO_HOST) ? 
          CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE;
      }
      copy_info.srcDevice = (CUdeviceptr)src;
      copy_info.srcHost = src;
      copy_info.srcPitch = src_stride;
      copy_info.srcY = 0;
      copy_info.srcXInBytes = 0;
      copy_info.dstDevice = (CUdeviceptr)dst;
      copy_info.dstHost = dst;
      copy_info.dstPitch = dst_stride;
      copy_info.dstY = 0;
      copy_info.dstXInBytes = 0;
      copy_info.WidthInBytes = bytes;
      copy_info.Height = lines;
      CHECK_CU( cuMemcpy2DAsync(&copy_info, stream->get_stream()) );

      if(notification)
	stream->add_notification(notification);

      log_gpudma.info("gpu memcpy 2d complete: dst=%p src=%p "
                   "dst_off=%ld src_off=%ld bytes=%ld lines=%ld kind=%d",
                   dst, src, (long)dst_stride, (long)src_stride, bytes, lines, kind);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // mem pair copiers for DMA channels

    class GPUtoFBMemPairCopier : public MemPairCopier {
    public:
      GPUtoFBMemPairCopier(Memory _src_mem, GPU *_gpu)
	: gpu(_gpu)
      {
	MemoryImpl *src_impl = get_runtime()->get_memory_impl(_src_mem);
	src_base = (const char *)(src_impl->get_direct_ptr(0, src_impl->size));
	assert(src_base);
      }

      virtual ~GPUtoFBMemPairCopier(void) { }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
					OASVec &oas_vec)
      {
	return new LegionRuntime::LowLevel::SpanBasedInstPairCopier<GPUtoFBMemPairCopier>(this, src_inst, 
											  dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	//printf("gpu write of %zd bytes\n", bytes);
	gpu->copy_to_fb(dst_offset, src_base + src_offset, bytes);
        record_bytes(bytes);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     off_t src_stride, off_t dst_stride, size_t lines)
      {
        gpu->copy_to_fb_2d(dst_offset, src_base + src_offset,
                           dst_stride, src_stride, bytes, lines);
        record_bytes(bytes * lines);
      }

      virtual void flush(DmaRequest *req)
      {
        if(total_reqs > 0)
          gpu->fence_to_fb(req);
        MemPairCopier::flush(req);
      }

    protected:
      const char *src_base;
      GPU *gpu;
    };

    class GPUfromFBMemPairCopier : public MemPairCopier {
    public:
      GPUfromFBMemPairCopier(GPU *_gpu, Memory _dst_mem)
	: gpu(_gpu)
      {
	MemoryImpl *dst_impl = get_runtime()->get_memory_impl(_dst_mem);
	dst_base = (char *)(dst_impl->get_direct_ptr(0, dst_impl->size));
	assert(dst_base);
      }

      virtual ~GPUfromFBMemPairCopier(void) { }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
					OASVec &oas_vec)
      {
	return new LegionRuntime::LowLevel::SpanBasedInstPairCopier<GPUfromFBMemPairCopier>(this, src_inst, 
											    dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	//printf("gpu read of %zd bytes\n", bytes);
	gpu->copy_from_fb(dst_base + dst_offset, src_offset, bytes);
        record_bytes(bytes);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     off_t src_stride, off_t dst_stride, size_t lines)
      {
        gpu->copy_from_fb_2d(dst_base + dst_offset, src_offset,
                             dst_stride, src_stride, bytes, lines);
        record_bytes(bytes * lines);
      }

      virtual void flush(DmaRequest *req)
      {
        if(total_reqs > 0)
          gpu->fence_from_fb(req);
        MemPairCopier::flush(req);
      }

    protected:
      char *dst_base;
      GPU *gpu;
    };
     
    class GPUinFBMemPairCopier : public MemPairCopier {
    public:
      GPUinFBMemPairCopier(GPU *_gpu)
	: gpu(_gpu)
      {
      }

      virtual ~GPUinFBMemPairCopier(void) { }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
					OASVec &oas_vec)
      {
	return new LegionRuntime::LowLevel::SpanBasedInstPairCopier<GPUinFBMemPairCopier>(this, src_inst, 
											  dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	//printf("gpu write of %zd bytes\n", bytes);
	gpu->copy_within_fb(dst_offset, src_offset, bytes);
        record_bytes(bytes);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     off_t src_stride, off_t dst_stride, size_t lines)
      {
        gpu->copy_within_fb_2d(dst_offset, src_offset,
                               dst_stride, src_stride, bytes, lines);
        record_bytes(bytes * lines);
      }

      virtual void flush(DmaRequest *req)
      {
        if(total_reqs > 0)
          gpu->fence_within_fb(req);
        MemPairCopier::flush(req);
      }

    protected:
      GPU *gpu;
    };

    class GPUPeerMemPairCopier : public MemPairCopier {
    public:
      GPUPeerMemPairCopier(GPU *_src, GPU *_dst)
        : src(_src), dst(_dst)
      {
      }

      virtual ~GPUPeerMemPairCopier(void) { }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
					OASVec &oas_vec)
      {
        return new LegionRuntime::LowLevel::SpanBasedInstPairCopier<GPUPeerMemPairCopier>(this, src_inst,
											  dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
        src->copy_to_peer(dst, dst_offset, src_offset, bytes);
        record_bytes(bytes);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
                     off_t src_stride, off_t dst_stride, size_t lines)
      {
        src->copy_to_peer_2d(dst, dst_offset, src_offset,
                             dst_stride, src_stride, bytes, lines);
        record_bytes(bytes * lines);
      }

      virtual void flush(DmaRequest *req)
      {
        if(total_reqs > 0)
          src->fence_to_peer(req, dst);
        MemPairCopier::flush(req);
      }

    protected:
      GPU *src, *dst;
    };

    class GPUDMAChannel_H2D : public MemPairCopierFactory {
    public:
      GPUDMAChannel_H2D(GPU *_gpu);

      virtual bool can_perform_copy(Memory src_mem, Memory dst_mem,
				    ReductionOpID redop_id, bool fold);

      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold);

    protected:
      GPU *gpu;
    };

    class GPUDMAChannel_D2H : public MemPairCopierFactory {
    public:
      GPUDMAChannel_D2H(GPU *_gpu);

      virtual bool can_perform_copy(Memory src_mem, Memory dst_mem,
				    ReductionOpID redop_id, bool fold);

      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold);

    protected:
      GPU *gpu;
    };

    class GPUDMAChannel_D2D : public MemPairCopierFactory {
    public:
      GPUDMAChannel_D2D(GPU *_gpu);

      virtual bool can_perform_copy(Memory src_mem, Memory dst_mem,
				    ReductionOpID redop_id, bool fold);

      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold);

    protected:
      GPU *gpu;
    };

    class GPUDMAChannel_P2P : public MemPairCopierFactory {
    public:
      GPUDMAChannel_P2P(GPU *_gpu);

      virtual bool can_perform_copy(Memory src_mem, Memory dst_mem,
				    ReductionOpID redop_id, bool fold);

      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold);

    protected:
      GPU *gpu;
    };


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUDMAChannel_H2D

    GPUDMAChannel_H2D::GPUDMAChannel_H2D(GPU *_gpu)
      : MemPairCopierFactory(stringbuilder() << "gpu_h2d (" << _gpu->proc->me << ")")
      , gpu(_gpu)
    {}

    bool GPUDMAChannel_H2D::can_perform_copy(Memory src_mem, Memory dst_mem,
					     ReductionOpID redop_id, bool fold)
    {
      // copies from pinned system memory to _our_ fb, no reduction support

      if(redop_id != 0)
	return false;

      if(gpu->pinned_sysmems.count(src_mem) == 0)
	return false;

      if(dst_mem != gpu->fbmem->me)
	return false;

      return true;
    }

    MemPairCopier *GPUDMAChannel_H2D::create_copier(Memory src_mem, Memory dst_mem,
						    ReductionOpID redop_id, bool fold)
    {
      return new GPUtoFBMemPairCopier(src_mem, gpu);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUDMAChannel_D2H

    GPUDMAChannel_D2H::GPUDMAChannel_D2H(GPU *_gpu)
      : MemPairCopierFactory(stringbuilder() << "gpu_d2h (" << _gpu->proc->me << ")")
      , gpu(_gpu)
    {}

    bool GPUDMAChannel_D2H::can_perform_copy(Memory src_mem, Memory dst_mem,
					     ReductionOpID redop_id, bool fold)
    {
      // copies from _our_ fb to pinned system memory, no reduction support

      if(redop_id != 0)
	return false;

      if(src_mem != gpu->fbmem->me)
	return false;

      if(gpu->pinned_sysmems.count(dst_mem) == 0)
	return false;

      return true;
    }

    LegionRuntime::LowLevel::MemPairCopier *GPUDMAChannel_D2H::create_copier(Memory src_mem, Memory dst_mem,
									     ReductionOpID redop_id, bool fold)
    {
      return new GPUfromFBMemPairCopier(gpu, dst_mem);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUDMAChannel_D2D

    GPUDMAChannel_D2D::GPUDMAChannel_D2D(GPU *_gpu)
      : MemPairCopierFactory(stringbuilder() << "gpu_d2d (" << _gpu->proc->me << ")")
      , gpu(_gpu)
    {}

    bool GPUDMAChannel_D2D::can_perform_copy(Memory src_mem, Memory dst_mem,
					     ReductionOpID redop_id, bool fold)
    {
      // copies entirely within our fb, no reduction support

      if(redop_id != 0)
	return false;

      Memory our_fb = gpu->fbmem->me;

      if((src_mem != our_fb) || (dst_mem != our_fb))
	return false;  // they can't both be our FB

      return true;
    }

    MemPairCopier *GPUDMAChannel_D2D::create_copier(Memory src_mem, Memory dst_mem,
						    ReductionOpID redop_id, bool fold)
    {
      return new GPUinFBMemPairCopier(gpu);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUDMAChannel_P2P

    GPUDMAChannel_P2P::GPUDMAChannel_P2P(GPU *_gpu)
      : MemPairCopierFactory(stringbuilder() << "gpu_p2p (" << _gpu->proc->me << ")")
      , gpu(_gpu)
    {}

    bool GPUDMAChannel_P2P::can_perform_copy(Memory src_mem, Memory dst_mem,
					     ReductionOpID redop_id, bool fold)
    {
      // copies from _our_ fb to a peer's fb, no reduction support

      if(redop_id != 0)
	return false;

      if(src_mem != gpu->fbmem->me)
	return false;

      if(gpu->peer_fbs.count(dst_mem) == 0)
	return false;

      return true;
    }

    MemPairCopier *GPUDMAChannel_P2P::create_copier(Memory src_mem, Memory dst_mem,
						    ReductionOpID redop_id, bool fold)
    {
      // TODO: remove this - the p2p copier doesn't actually need it
      MemoryImpl *dst_impl = get_runtime()->get_memory_impl(dst_mem);
      GPU *dst_gpu = ((GPUFBMemory *)dst_impl)->gpu;

      return new GPUPeerMemPairCopier(gpu, dst_gpu);
    }


    void GPU::create_dma_channels(Realm::RuntimeImpl *r)
    {
      if(!pinned_sysmems.empty()) {
	r->add_dma_channel(new GPUDMAChannel_H2D(this));
	r->add_dma_channel(new GPUDMAChannel_D2H(this));
      } else {
	log_gpu.warning() << "GPU " << proc->me << " has no pinned system memories!?";
      }

      r->add_dma_channel(new GPUDMAChannel_D2D(this));

      // only create a p2p channel if we have peers
      for(std::vector<GPU *>::iterator it = module->gpus.begin();
	  it != module->gpus.end();
	  it++) {
	// ignore ourselves
	if(*it == this) continue;

	// ignore gpus that we don't expect to be able to peer with
	if(info->peers.count((*it)->info->device) == 0)
	  continue;

	// enable peer access
	{
	  AutoGPUContext agc(this);
	  CHECK_CU( cuCtxEnablePeerAccess((*it)->context, 0) );
	}
	peer_fbs.insert((*it)->fbmem->me);
	log_gpu.print() << "peer access enabled from FB " << fbmem->me << " to FB " << (*it)->fbmem->me;
      }
      if(!peer_fbs.empty())
	r->add_dma_channel(new GPUDMAChannel_P2P(this));
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUWorkFence

    GPUWorkFence::GPUWorkFence(Realm::Operation *op)
      : Realm::Operation::AsyncWorkItem(op)
    {}

    void GPUWorkFence::request_cancellation(void)
    {
      // ignored - no way to shoot down CUDA work
    }

    void GPUWorkFence::enqueue_on_stream(GPUStream *stream)
    {
      if(stream->get_gpu()->module->cfg_fences_use_callbacks) {
	CHECK_CU( cuStreamAddCallback(stream->get_stream(), &cuda_callback, (void *)this, 0) );
      } else {
	stream->add_fence(this);
      }
    }

    /*static*/ void GPUWorkFence::cuda_callback(CUstream stream, CUresult res, void *data)
    {
      GPUWorkFence *me = (GPUWorkFence *)data;

      assert(res == CUDA_SUCCESS);
      me->mark_finished();
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUMemcpyFence

    GPUMemcpyFence::GPUMemcpyFence(GPU *_gpu, GPUMemcpyKind _kind,
				   GPUWorkFence *_fence)
      : GPUMemcpy(_gpu, _kind), fence(_fence)
    {
      //log_stream.info() << "gpu memcpy fence " << this << " (fence = " << fence << ") created";
    }

    void GPUMemcpyFence::execute(GPUStream *stream)
    {
      //log_stream.info() << "gpu memcpy fence " << this << " (fence = " << fence << ") executed";
      fence->enqueue_on_stream(stream);
#ifdef FORCE_GPU_STREAM_SYNCHRONIZE
      CHECK_CU( cuStreamSynchronize(stream->get_stream()) );
#endif
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUEventPool

    GPUEventPool::GPUEventPool(int _batch_size)
      : batch_size(_batch_size), current_size(0), total_size(0)
    {
      // don't immediately fill the pool because we're not managing the context ourselves
    }

    // allocating the initial batch of events and cleaning up are done with
    //  these methods instead of constructor/destructor because we don't
    //  manage the GPU context in this helper class
    void GPUEventPool::init_pool(int init_size /*= 0 -- default == batch size */)
    {
      assert(available_events.empty());

      if(init_size == 0)
	init_size = batch_size;

      available_events.resize(init_size);

      current_size = init_size;
      total_size = init_size;

      // TODO: measure how much benefit is derived from CU_EVENT_DISABLE_TIMING and
      //  consider using them for completion callbacks
      for(int i = 0; i < init_size; i++)
	CHECK_CU( cuEventCreate(&available_events[i], CU_EVENT_DEFAULT) );
    }

    void GPUEventPool::empty_pool(void)
    {
      // shouldn't be any events running around still
      assert(current_size == total_size);

      for(int i = 0; i < current_size; i++)
	CHECK_CU( cuEventDestroy(available_events[i]) );

      current_size = 0;
      total_size = 0;

      // free internal vector storage
      std::vector<CUevent>().swap(available_events);
    }

    CUevent GPUEventPool::get_event(void)
    {
      AutoHSLLock al(mutex);

      if(current_size == 0) {
	// if we need to make an event, make a bunch
	current_size = batch_size;
	total_size += batch_size;

	log_stream.info() << "event pool " << this << " depleted - adding " << batch_size << " events";
      
	// resize the vector (considering all events that might come back)
	available_events.resize(total_size);

	for(int i = 0; i < batch_size; i++)
	  CHECK_CU( cuEventCreate(&available_events[i], CU_EVENT_DEFAULT) );
      }

      return available_events[--current_size];
    }

    void GPUEventPool::return_event(CUevent e)
    {
      AutoHSLLock al(mutex);

      assert(current_size < total_size);

      available_events[current_size++] = e;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUTaskScheduler<T>

    // we want to subclass the scheduler to replace the execute_task method, but we also want to
    //  allow the use of user or kernel threads, so we apply a bit of template magic (which only works
    //  because the constructors for the KernelThreadTaskScheduler and UserThreadTaskScheduler classes
    //  have the same prototypes)

    template <typename T>
    class GPUTaskScheduler : public T {
    public:
      GPUTaskScheduler(Processor _proc, Realm::CoreReservation& _core_rsrv,
		       GPUProcessor *_gpu_proc);

      virtual ~GPUTaskScheduler(void);

    protected:
      virtual bool execute_task(Task *task);

      // might also need to override the thread-switching methods to keep TLS up to date

      GPUProcessor *gpu_proc;
    };

    template <typename T>
    GPUTaskScheduler<T>::GPUTaskScheduler(Processor _proc,
					  Realm::CoreReservation& _core_rsrv,
					  GPUProcessor *_gpu_proc)
      : T(_proc, _core_rsrv), gpu_proc(_gpu_proc)
    {
      // nothing else
    }

    template <typename T>
    GPUTaskScheduler<T>::~GPUTaskScheduler(void)
    {
    }

    namespace ThreadLocal {
      static __thread GPUProcessor *current_gpu_proc = 0;
    };

    template <typename T>
    bool GPUTaskScheduler<T>::execute_task(Task *task)
    {
      // use TLS to make sure that the task can find the current GPU processor when it makes
      //  CUDA RT calls
      // TODO: either eliminate these asserts or do TLS swapping when using user threads
      assert(ThreadLocal::current_gpu_proc == 0);
      ThreadLocal::current_gpu_proc = gpu_proc;

      // push the CUDA context for this GPU onto this thread
      gpu_proc->gpu->push_context();

      // bump the current stream
      // TODO: sanity-check whether this even works right when GPU tasks suspend
      GPUStream *s = gpu_proc->gpu->switch_to_next_task_stream();

      // we'll use a "work fence" to track when the kernels launched by this task actually
      //  finish - this must be added to the task _BEFORE_ we execute
      GPUWorkFence *fence = new GPUWorkFence(task);
      task->add_async_work_item(fence);

      bool ok = T::execute_task(task);

      // now enqueue the fence on the local stream
      fence->enqueue_on_stream(s);

      // A useful debugging macro
#ifdef FORCE_GPU_STREAM_SYNCHRONIZE
      CHECK_CU( cuStreamSynchronize(s->get_stream()) );
#endif

      // pop the CUDA context for this GPU back off
      gpu_proc->gpu->pop_context();

      assert(ThreadLocal::current_gpu_proc == gpu_proc);
      ThreadLocal::current_gpu_proc = 0;

      return ok;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUProcessor

    GPUProcessor::GPUProcessor(GPU *_gpu, Processor _me, Realm::CoreReservationSet& crs,
                               size_t _stack_size)
      : LocalTaskProcessor(_me, Processor::TOC_PROC)
      , gpu(_gpu)
    {
#if 0
      assert(streams > 0);
      task_streams.resize(streams);
      // Make our context and then immediately pop it off
      CHECK_CU( cuDeviceGet(&proc_dev, gpu_index) );

      CHECK_CU( cuCtxCreate(&proc_ctx, CU_CTX_MAP_HOST |
                            CU_CTX_SCHED_BLOCKING_SYNC, proc_dev) );

      // allocate zero-copy memory
      CHECK_CU( cuMemHostAlloc(&zcmem_cpu_base,
                               zcmem_size + zcmem_reserve,
                               (CU_MEMHOSTALLOC_PORTABLE |
                                CU_MEMHOSTALLOC_DEVICEMAP)) );
      CHECK_CU( cuMemHostGetDevicePointer((CUdeviceptr*)&zcmem_gpu_base,
                                          zcmem_cpu_base, 0) );

      // allocate frame buffer memory
      CHECK_CU( cuMemAlloc((CUdeviceptr*)&fbmem_gpu_base, fbmem_size + fbmem_reserve) );

      // allocate pinned buffer for kernel arguments
      kernel_buffer_size = 8192; // default four pages
      kernel_arg_size = 0;
      CHECK_CU( cuMemAllocHost((void**)&kernel_arg_buffer, kernel_buffer_size) );
      
      // get/create our worker
      if(use_shared_worker) {
	// we're the first, so go create it
	if(shared_worker_users == 0) {
	  shared_worker = new GPUWorker;

	  // shared worker ALWAYS uses a background thread
	  shared_worker->start_background_thread(crs, _stack_size);
	}

	shared_worker_users++;
	gpu_worker = shared_worker;
      } else {
	// build our own worker
	gpu_worker = new GPUWorker;

	if(use_background_workers)
	  gpu_worker->start_background_thread(crs, _stack_size);
      }

      initialize_cuda_stuff();

      // prime the event pool
      event_pool.init_pool();

      CHECK_CU( cuCtxPopCurrent(&proc_ctx) );
#endif
      Realm::CoreReservationParameters params;
      params.set_num_cores(1);
      params.set_alu_usage(params.CORE_USAGE_SHARED);
      params.set_fpu_usage(params.CORE_USAGE_SHARED);
      params.set_ldst_usage(params.CORE_USAGE_SHARED);
      params.set_max_stack_size(_stack_size);

      std::string name = stringbuilder() << "GPU proc " << _me;

      core_rsrv = new Realm::CoreReservation(name, crs, params);

#ifdef REALM_USE_USER_THREADS_FOR_GPU
      Realm::UserThreadTaskScheduler *sched = new GPUTaskScheduler<Realm::UserThreadTaskScheduler>(me, *core_rsrv, this);
      // no config settings we want to tweak yet
#else
      Realm::KernelThreadTaskScheduler *sched = new GPUTaskScheduler<Realm::KernelThreadTaskScheduler>(me, *core_rsrv, this);
      // no config settings we want to tweak yet
#endif
      set_scheduler(sched);
    }

    GPUProcessor::~GPUProcessor(void)
    {
      delete core_rsrv;
    }

#if 0
    void* GPUProcessor::get_zcmem_cpu_base(void) const
    {
      return ((char *)zcmem_cpu_base) + zcmem_reserve;
    }

    void* GPUProcessor::get_fbmem_gpu_base(void) const
    {
      return ((char *)fbmem_gpu_base) + fbmem_reserve;
    }

    size_t GPUProcessor::get_zcmem_size(void) const
    {
      return zcmem_size;
    }

    size_t GPUProcessor::get_fbmem_size(void) const
    {
      return fbmem_size;
    }
#endif

    void GPU::copy_to_fb(off_t dst_offset, const void *src, size_t bytes,
			 GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy1D(this,
					(void *)(fbmem->base + dst_offset),
					src, bytes, GPU_MEMCPY_HOST_TO_DEVICE, notification);
      host_to_device_stream->add_copy(copy);
    }

    void GPU::copy_to_fb(off_t dst_offset, const void *src,
			 const ElementMask *mask, size_t elmt_size,
			 GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy1D(this,
					(void *)(fbmem->base + dst_offset),
					src, mask, elmt_size,
					GPU_MEMCPY_HOST_TO_DEVICE, notification);
      host_to_device_stream->add_copy(copy);
    }

    void GPU::copy_from_fb(void *dst, off_t src_offset, size_t bytes,
			   GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy1D(this,
					dst, (const void *)(fbmem->base + src_offset),
					bytes, GPU_MEMCPY_DEVICE_TO_HOST, notification);
      device_to_host_stream->add_copy(copy);
    } 

    void GPU::copy_from_fb(void *dst, off_t src_offset,
			   const ElementMask *mask, size_t elmt_size,
			   GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy1D(this,
					dst, (const void *)(fbmem->base + src_offset),
					mask, elmt_size,
					GPU_MEMCPY_DEVICE_TO_HOST, notification);
      device_to_host_stream->add_copy(copy);
    }

    void GPU::copy_within_fb(off_t dst_offset, off_t src_offset,
			     size_t bytes,
			     GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy1D(this,
					(void *)(fbmem->base + dst_offset),
					(const void *)(fbmem->base + src_offset),
					bytes, GPU_MEMCPY_DEVICE_TO_DEVICE, notification);
      device_to_device_stream->add_copy(copy);
    }

    void GPU::copy_within_fb(off_t dst_offset, off_t src_offset,
			     const ElementMask *mask, size_t elmt_size,
			     GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy1D(this,
					(void *)(fbmem->base + dst_offset),
					(const void *)(fbmem->base + src_offset),
					mask, elmt_size, GPU_MEMCPY_DEVICE_TO_DEVICE,
					notification);
      device_to_device_stream->add_copy(copy);
    }

    void GPU::copy_to_fb_2d(off_t dst_offset, const void *src, 
                                     off_t dst_stride, off_t src_stride,
                                     size_t bytes, size_t lines,
				     GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy2D(this,
					(void *)(fbmem->base + dst_offset),
					src, dst_stride, src_stride, bytes, lines,
					GPU_MEMCPY_HOST_TO_DEVICE, notification);
      host_to_device_stream->add_copy(copy);
    }

    void GPU::copy_from_fb_2d(void *dst, off_t src_offset,
			      off_t dst_stride, off_t src_stride,
			      size_t bytes, size_t lines,
			      GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy2D(this, dst,
					(const void *)(fbmem->base + src_offset),
					dst_stride, src_stride, bytes, lines,
					GPU_MEMCPY_DEVICE_TO_HOST, notification);
      device_to_host_stream->add_copy(copy);
    }

    void GPU::copy_within_fb_2d(off_t dst_offset, off_t src_offset,
                                         off_t dst_stride, off_t src_stride,
                                         size_t bytes, size_t lines,
					 GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy2D(this,
					(void *)(fbmem->base + dst_offset),
					(const void *)(fbmem->base + src_offset),
					dst_stride, src_stride, bytes, lines,
					GPU_MEMCPY_DEVICE_TO_DEVICE, notification);
      device_to_device_stream->add_copy(copy);
    }

    void GPU::copy_to_peer(GPU *dst, off_t dst_offset,
			   off_t src_offset, size_t bytes,
			   GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy1D(this,
					(void *)(dst->fbmem->base + dst_offset),
					(const void *)(fbmem->base + src_offset),
					bytes, GPU_MEMCPY_PEER_TO_PEER, notification);
      peer_to_peer_stream->add_copy(copy);
    }

    void GPU::copy_to_peer_2d(GPU *dst,
			      off_t dst_offset, off_t src_offset,
			      off_t dst_stride, off_t src_stride,
			      size_t bytes, size_t lines,
			      GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy2D(this,
					(void *)(dst->fbmem->base + dst_offset),
					(const void *)(fbmem->base + src_offset),
					dst_stride, src_stride, bytes, lines,
					GPU_MEMCPY_PEER_TO_PEER, notification);
      peer_to_peer_stream->add_copy(copy);
    }

    void GPU::fence_to_fb(Realm::Operation *op)
    {
      GPUWorkFence *f = new GPUWorkFence(op);

      // this must be done before we enqueue the callback with CUDA
      op->add_async_work_item(f);

      host_to_device_stream->add_copy(new GPUMemcpyFence(this,
							 GPU_MEMCPY_HOST_TO_DEVICE,
							 f));
    }

    void GPU::fence_from_fb(Realm::Operation *op)
    {
      GPUWorkFence *f = new GPUWorkFence(op);

      // this must be done before we enqueue the callback with CUDA
      op->add_async_work_item(f);

      device_to_host_stream->add_copy(new GPUMemcpyFence(this,
							 GPU_MEMCPY_DEVICE_TO_HOST,
							 f));
    }

    void GPU::fence_within_fb(Realm::Operation *op)
    {
      GPUWorkFence *f = new GPUWorkFence(op);

      // this must be done before we enqueue the callback with CUDA
      op->add_async_work_item(f);

      device_to_device_stream->add_copy(new GPUMemcpyFence(this,
							   GPU_MEMCPY_DEVICE_TO_DEVICE,
							   f));
    }

    void GPU::fence_to_peer(Realm::Operation *op, GPU *dst)
    {
      GPUWorkFence *f = new GPUWorkFence(op);

      // this must be done before we enqueue the callback with CUDA
      op->add_async_work_item(f);

      peer_to_peer_stream->add_copy(new GPUMemcpyFence(this,
						       GPU_MEMCPY_PEER_TO_PEER,
						       f));
    }

#if 0
    void GPUProcessor::register_host_memory(Realm::MemoryImpl *m)
    {
      size_t size = m->size;
      void *base = m->get_direct_ptr(0, size);
      assert(base != 0);

      if (true /*SJT: why this? !shutdown*/)
      {
        CHECK_CU( cuCtxPushCurrent(proc_ctx) );
        CHECK_CU( cuMemHostRegister(base, size, CU_MEMHOSTREGISTER_PORTABLE) ); 
        CHECK_CU( cuCtxPopCurrent(&proc_ctx) );
      }
    }

    void GPUProcessor::enable_peer_access(GPUProcessor *peer)
    {
      peer->handle_peer_access(proc_ctx);
      peer_gpus.insert(peer);
    }

    void GPUProcessor::handle_peer_access(CUcontext peer_ctx)
    {
      CHECK_CU( cuCtxPushCurrent(proc_ctx) );
      CHECK_CU( cuCtxEnablePeerAccess(peer_ctx, 0) );
      CHECK_CU( cuCtxPopCurrent(&proc_ctx) );
    }
#endif

#if 0
    bool GPU::can_access_peer(GPU *peer)
    {
      return(info->peers.count(peer->info->device) > 0);
    }
#endif

    GPUStream *GPU::get_current_task_stream(void)
    {
      return task_streams[current_stream];
    }

    GPUStream *GPU::switch_to_next_task_stream(void)
    {
      current_stream++;
      if(current_stream >= task_streams.size())
	current_stream = 0;
      return task_streams[current_stream];
    }

#if 0
    void GPUProcessor::initialize_cuda_stuff(void)
    {
      // load any modules, functions, and variables that we deferred
      const std::map<void*,void**> &deferred_modules = get_deferred_modules();
      for (std::map<void*,void**>::const_iterator it = deferred_modules.begin();
            it != deferred_modules.end(); it++)
      {
        ModuleInfo &info = modules[it->second];
        FatBin *fatbin_args = (FatBin*)it->first;
        assert((fatbin_args->data != NULL) ||
               (fatbin_args->filename_or_fatbins != NULL));
        if (fatbin_args->data != NULL)
        {
          load_module(&(info.module), fatbin_args->data);
        } else {
          CHECK_CU( cuModuleLoad(&(info.module), 
                (const char*)fatbin_args->filename_or_fatbins) );
        }
      }
      const std::map<void*,void**> &deferred_cubins = get_deferred_cubins();
      for (std::map<void*,void**>::const_iterator it = deferred_cubins.begin();
            it != deferred_cubins.end(); it++)
      {
        ModuleInfo &info = modules[it->second];
        CHECK_CU( cuModuleLoadData(&(info.module), it->first) );
      }
      const std::deque<DeferredFunction> &deferred_functions = get_deferred_functions();
      for (std::deque<DeferredFunction>::const_iterator it = deferred_functions.begin();
            it != deferred_functions.end(); it++)
      {
        internal_register_function(it->handle, it->host_fun, it->device_fun);
      }
      const std::deque<DeferredVariable> &deferred_variables = get_deferred_variables();
      for (std::deque<DeferredVariable>::const_iterator it = deferred_variables.begin();
            it != deferred_variables.end(); it++)
      {
        internal_register_var(it->handle, it->host_var, it->device_name, 
                              it->external, it->size, it->constant, it->global);
      } 

      // initialize the streams for copy operations
      host_to_device_stream = new GPUStream(this, gpu_worker);
      device_to_host_stream = new GPUStream(this, gpu_worker);
      device_to_device_stream = new GPUStream(this, gpu_worker);
      peer_to_peer_stream = new GPUStream(this, gpu_worker);

      for(unsigned idx = 0; idx < task_streams.size(); idx++)
	task_streams[idx] = new GPUStream(this, gpu_worker);

      log_gpu.info("gpu initialized: zcmem=%p/%p fbmem=%p",
		   zcmem_cpu_base, zcmem_gpu_base, fbmem_gpu_base);
    }
#endif

    void GPUProcessor::shutdown(void)
    {
      log_gpu.info("shutting down");

      // shut down threads/scheduler
      LocalTaskProcessor::shutdown();

      // synchronize the device so we can flush any printf buffers - do
      //  this after shutting down the threads so that we know all work is done
      {
	AutoGPUContext agc(gpu);

	CHECK_CU( cuCtxSynchronize() );
      }

#if 0
      // now clean up the GPU worker
      if(use_shared_worker) {
	shared_worker_users -= 1;
	if(shared_worker_users == 0) {
	  shared_worker->shutdown_background_thread();
	  delete shared_worker;
	  shared_worker = 0;
	}
      } else {
	if(use_background_workers)
	  gpu_worker->shutdown_background_thread();
	delete gpu_worker;
      }
#endif
    }

    GPUWorker::GPUWorker(void)
      : condvar(lock)
      , core_rsrv(0), worker_thread(0), worker_shutdown_requested(false)
    {}

    GPUWorker::~GPUWorker(void)
    {
      // shutdown should have already been called
      assert(worker_thread == 0);
    }

    void GPUWorker::start_background_thread(Realm::CoreReservationSet &crs,
					    size_t stack_size)
    {
      core_rsrv = new Realm::CoreReservation("GPU worker thread", crs,
					     Realm::CoreReservationParameters());

      Realm::ThreadLaunchParameters tlp;

      worker_thread = Realm::Thread::create_kernel_thread<GPUWorker,
							  &GPUWorker::thread_main>(this,
										   tlp,
										   *core_rsrv,
										   0);
    }

    void GPUWorker::shutdown_background_thread(void)
    {
      {
	AutoHSLLock al(lock);
	worker_shutdown_requested = true;
	condvar.broadcast();
      }

      worker_thread->join();
      delete worker_thread;
      worker_thread = 0;

      delete core_rsrv;
      core_rsrv = 0;
    }

    void GPUWorker::add_stream(GPUStream *stream)
    {
      AutoHSLLock al(lock);

      // if the stream is already in the set, nothing to do
      if(active_streams.count(stream) > 0)
	return;

      active_streams.insert(stream);

      condvar.broadcast();
    }

    bool GPUWorker::process_streams(bool sleep_on_empty)
    {
      // we start by grabbing the list of active streams, replacing it with an
      //  empty list - this way we don't have to hold the lock the whole time
      // for any stream that we leave work on, we'll add it back in
      std::set<GPUStream *> streams;
      {
	AutoHSLLock al(lock);

	while(active_streams.empty()) {
	  if(!sleep_on_empty || worker_shutdown_requested) return false;
	  condvar.wait();
	}

	streams.swap(active_streams);
      }

      bool any_work_left = false;
      for(std::set<GPUStream *>::const_iterator it = streams.begin();
	  it != streams.end();
	  it++) {
	GPUStream *s = *it;
	bool stream_work_left = false;

	if(s->issue_copies())
	  stream_work_left = true;

	if(s->reap_events())
	  stream_work_left = true;

	if(stream_work_left) {
	  add_stream(s);
	  any_work_left = true;
	}
      }

      return any_work_left;
    }

    void GPUWorker::thread_main(void)
    {
      // TODO: consider busy-waiting in some cases to reduce latency?
      while(!worker_shutdown_requested) {
	bool work_left = process_streams(true);

	// if there was work left, yield our thread for now to avoid a tight spin loop
	// TODO: enqueue a callback so we can go to sleep and wake up sooner than a kernel
	//  timeslice?
	if(work_left)
	  Realm::Thread::yield();
      }
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPU

    GPUFBMemory::GPUFBMemory(Memory _me, GPU *_gpu, CUdeviceptr _base, size_t _size)
      : MemoryImpl(_me, _size, MKIND_GPUFB, 512, Memory::GPU_FB_MEM)
      , gpu(_gpu), base(_base)
    {
      free_blocks[0] = size;
    }

    GPUFBMemory::~GPUFBMemory(void) {}

    RegionInstance GPUFBMemory::create_instance(IndexSpace is,
						const int *linearization_bits,
						size_t bytes_needed,
						size_t block_size,
						size_t element_size,
						const std::vector<size_t>& field_sizes,
						ReductionOpID redopid,
						off_t list_size,
						const Realm::ProfilingRequestSet &reqs,
						RegionInstance parent_inst)
    {
      return create_instance_local(is, linearization_bits, bytes_needed,
				   block_size, element_size, field_sizes, redopid,
				   list_size, reqs, parent_inst);
    }

    void GPUFBMemory::destroy_instance(RegionInstance i, 
				       bool local_destroy)
    {
      destroy_instance_local(i, local_destroy);
    }

    off_t GPUFBMemory::alloc_bytes(size_t size)
    {
      return alloc_bytes_local(size);
    }

    void GPUFBMemory::free_bytes(off_t offset, size_t size)
    {
      free_bytes_local(offset, size);
    }

    // these work, but they are SLOW
    void GPUFBMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      // create an async copy and then wait for it to finish...
      gpu->copy_from_fb(dst, offset, size);
      CHECK_CU( cuStreamSynchronize(gpu->device_to_host_stream->get_stream()) );
    }

    void GPUFBMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      // create an async copy and then wait for it to finish...
      gpu->copy_to_fb(offset, src, size);
      CHECK_CU( cuStreamSynchronize(gpu->host_to_device_stream->get_stream()) );
    }

    void *GPUFBMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return (void *)(base + offset);
    }

    int GPUFBMemory::get_home_node(off_t offset, size_t size)
    {
      return -1;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUZCMemory

    GPUZCMemory::GPUZCMemory(Memory _me,
			     CUdeviceptr _gpu_base, void *_cpu_base, size_t _size)
      : MemoryImpl(_me, _size, MKIND_ZEROCOPY, 256, Memory::Z_COPY_MEM)
      , gpu_base(_gpu_base), cpu_base((char *)_cpu_base)
    {
      free_blocks[0] = size;
    }

    GPUZCMemory::~GPUZCMemory(void) {}

    RegionInstance GPUZCMemory::create_instance(IndexSpace is,
						const int *linearization_bits,
						size_t bytes_needed,
						size_t block_size,
						size_t element_size,
						const std::vector<size_t>& field_sizes,
						ReductionOpID redopid,
						off_t list_size,
						const Realm::ProfilingRequestSet &reqs,
						RegionInstance parent_inst)
    {
      return create_instance_local(is, linearization_bits, bytes_needed,
				   block_size, element_size, field_sizes, redopid,
				   list_size, reqs, parent_inst);
    }

    void GPUZCMemory::destroy_instance(RegionInstance i, 
				       bool local_destroy)
    {
      destroy_instance_local(i, local_destroy);
    }

    off_t GPUZCMemory::alloc_bytes(size_t size)
    {
      return alloc_bytes_local(size);
    }

    void GPUZCMemory::free_bytes(off_t offset, size_t size)
    {
      free_bytes_local(offset, size);
    }

    void GPUZCMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      memcpy(dst, cpu_base+offset, size);
    }

    void GPUZCMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      memcpy(cpu_base+offset, src, size);
    }

    void *GPUZCMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return (cpu_base + offset);
    }

    int GPUZCMemory::get_home_node(off_t offset, size_t size)
    {
      return ID(me).node();
    }

#ifdef POINTER_CHECKS
    static unsigned *get_gpu_valid_mask(RegionMetaDataUntyped region)
    {
	const ElementMask &mask = region.get_valid_mask();
	void *valid_mask_base;
	for(size_t p = 0; p < mask.raw_size(); p += 4)
	  log_gpudma.info("  raw mask data[%zd] = %08x\n", p,
		       ((unsigned *)(mask.get_raw()))[p>>2]);
        CHECK_CU( cuMemAlloc((cuDevicePtr*)(&valid_mask_base), mask.raw_size()) );
	log_gpudma.info("copy of valid mask (%zd bytes) created at %p",
		     mask.raw_size(), valid_mask_base);
        CHECK_CU( cuMemcpyHtoD(vald_mask_base, 
                               mask.get_raw(),
                               mask.raw_size()) );
	return (unsigned *)&(((ElementMaskImpl *)valid_mask_base)->bits);
    }
#endif

    // Helper methods for emulating the cuda runtime
    /*static*/ GPUProcessor* GPUProcessor::get_current_gpu_proc(void)
    {
      return ThreadLocal::current_gpu_proc;
    }

#if 0
    /*static*/ std::map<void*,void**>& GPUProcessor::get_deferred_modules(void)
    {
      static std::map<void*,void**> deferred_modules;
      return deferred_modules;
    }

    /*static*/ std::map<void*,void**>& GPUProcessor::get_deferred_cubins(void)
    {
      static std::map<void*,void**> deferred_cubins;
      return deferred_cubins;
    }

    /*static*/ std::deque<GPUProcessor::DeferredFunction>&
                                      GPUProcessor::get_deferred_functions(void)
    {
      static std::deque<DeferredFunction> deferred_functions;
      return deferred_functions;
    }

    /*static*/ std::deque<GPUProcessor::DeferredVariable>&
                                      GPUProcessor::get_deferred_variables(void)
    {
      static std::deque<DeferredVariable> deferred_variables;
      return deferred_variables;
    }
#endif

#if 0
    void** GPUProcessor::internal_register_fat_binary(void *fat_bin)
    {
      void **handle = (void**)malloc(sizeof(void**));
      *handle = fat_bin;
      ModuleInfo &info = modules[handle];
      FatBin *fatbin_args = (FatBin*)fat_bin;
      assert((fatbin_args->data != NULL) ||
             (fatbin_args->filename_or_fatbins != NULL));
      if (fatbin_args->data != NULL)
      {
        load_module(&(info.module), fatbin_args->data);
      } else {
        CHECK_CU( cuModuleLoad(&(info.module), (const char*)fatbin_args->filename_or_fatbins) );
      }
      // add this to the list of local modules to be created during this task
      task_modules.insert(handle);
      return handle;
    }

    void** GPUProcessor::internal_register_cuda_binary(void *cubin)
    {
      void **handle = (void**)malloc(sizeof(void**));
      *handle = cubin;
      ModuleInfo &info = modules[handle];
      CHECK_CU( cuModuleLoadData(&(info.module), cubin) );
      // add this to the list of local modules to be created during this task
      task_modules.insert(handle);
      return handle;
    }
#endif

    void GPUProcessor::stream_synchronize(cudaStream_t stream)
    {
      // same as device_synchronize for now
      GPUStream *current = gpu->get_current_task_stream();
      CHECK_CU( cuStreamSynchronize(current->get_stream()) );
    }

    void GPUProcessor::device_synchronize(void)
    {
      GPUStream *current = gpu->get_current_task_stream();
      CHECK_CU( cuStreamSynchronize(current->get_stream()) );
    }

#if 0
    /*static*/ void** GPUProcessor::defer_module_load(void *fat_bin)
    {
      void **handle = (void**)malloc(sizeof(void**));
      *handle = fat_bin;
      // Assume we don't need a lock here because all this is sequential
      get_deferred_modules()[fat_bin] = handle;
      return handle;
    }

    /*static*/ void** GPUProcessor::defer_cubin_load(void *cubin)
    {
      void **handle = (void**)malloc(sizeof(void**));
      *handle = cubin;
      // Assume we don't need a lock here because all this is sequential
      get_deferred_cubins()[cubin] = handle;
      return handle;
    }

    /*static*/ void** GPUProcessor::register_fat_binary(void *fat_bin)
    {
      GPUProcessor *local = find_local_gpu(); 
      // Ignore anything that goes on during start-up
      if (local == NULL)
        return GPUProcessor::defer_module_load(fat_bin);
      return local->internal_register_fat_binary(fat_bin);
    }

    /*static*/ void** GPUProcessor::register_cuda_binary(void *cubin,
                                                         size_t cubinSize)
    {
      void* cubinCopy = malloc(cubinSize);
      memcpy(cubinCopy, cubin, cubinSize);
      GPUProcessor *local = find_local_gpu();
      // Ignore anything that goes on during start-up
      if (local == NULL)
        return GPUProcessor::defer_cubin_load(cubinCopy);
      return local->internal_register_cuda_binary(cubinCopy);
    }

    void GPUProcessor::internal_unregister_fat_binary(void **fat_bin)
    {
      // It's never safe to unload our modules here
      // We need to wait until we synchronize the current
      // task stream with all the things that we've done
      CHECK_CU( cuCtxSynchronize() );
      std::map<void**,ModuleInfo>::iterator finder = modules.find(fat_bin);
      assert(finder != modules.end());
      CHECK_CU( cuModuleUnload(finder->second.module) );
      for (std::set<const void*>::const_iterator it = finder->second.host_aliases.begin();
            it != finder->second.host_aliases.end(); it++)
      {
        device_functions.erase(*it);
      }
      for (std::set<const void*>::const_iterator it = finder->second.var_aliases.begin();
            it != finder->second.var_aliases.end(); it++)
      {
        device_variables.erase(*it);
      }
      modules.erase(finder);
      free(fat_bin);
    }
    
    /*static*/ void GPUProcessor::unregister_fat_binary(void **fat_bin)
    {
      // Do nothing, our task contexts will clean themselves up after they are done
    }

    void GPUProcessor::internal_register_var(void **fat_bin, char *host_var,
                                             const char *device_name,
                                             bool ext, int size, bool constant, bool global)
    {
      std::map<void**,ModuleInfo>::iterator mod_finder = modules.find(fat_bin);
      assert(mod_finder != modules.end());
      ModuleInfo &info = mod_finder->second;
      // Check to see if we already have this symbol
      std::map<const void*,VarInfo>::const_iterator var_finder = 
        device_variables.find(host_var);
      if (var_finder == device_variables.end())
      {
        VarInfo target;
        CHECK_CU( cuModuleGetGlobal(&(target.ptr), &(target.size), info.module, device_name) );
        target.name = device_name;
        device_variables[host_var] = target;
        info.var_aliases.insert(host_var);
      }
    }

    /*static*/ void GPUProcessor::defer_variable_load(void **fat_bin,
                                                                char *host_var,
                                                                const char *device_name,
                                                                bool ext, int size,
                                                                bool constant, bool global)
    {
      DeferredVariable var;
      var.handle = fat_bin;
      var.host_var = host_var;
      var.device_name = device_name;
      var.external = ext;
      var.size = size;
      var.constant = constant;
      var.global = global;
      get_deferred_variables().push_back(var);
    }

    /*static*/ void GPUProcessor::register_var(void **fat_bin, char *host_var,
                                               char *device_addr, const char *device_name,
                                               int ext, int size, int constant, int global)
    {
      GPUProcessor *local = find_local_gpu();
      if (local == NULL)
      {
        GPUProcessor::defer_variable_load(fat_bin, host_var, device_name,
                                                    (ext == 1), size, 
                                                    (constant == 1), (global == 1));
        return;
      }
      local->internal_register_var(fat_bin, host_var, device_name, 
                                   (ext == 1), size, (constant == 1), (global == 1));
    }

    void GPUProcessor::internal_register_function(void **fat_bin, const char *host_fun,
                                                  const char *device_fun) 
    {
      // Check to see if we already loaded this function
      std::map<const void*,CUfunction>::const_iterator func_finder = 
        device_functions.find(host_fun);
      if (func_finder != device_functions.end())
        return;
      // Otherwise we need to load it
      std::map<void**,ModuleInfo>::iterator mod_finder = modules.find(fat_bin);
      assert(mod_finder != modules.end());
      ModuleInfo &info = mod_finder->second;      
      info.host_aliases.insert(host_fun);
      CUfunction *func = &(device_functions[host_fun]); 
      CHECK_CU( cuModuleGetFunction(func, info.module, device_fun) );
    }

    /*static*/ void GPUProcessor::defer_function_load(void **fat_bin,
                                                                const char *host_fun,
                                                                const char *device_fun)
    {
      DeferredFunction df;
      df.handle = fat_bin;
      df.host_fun = host_fun;
      df.device_fun = device_fun;
      // Assume we don't need a lock here
      get_deferred_functions().push_back(df); 
    }

    /*static*/ void GPUProcessor::register_function(void **fat_bin, const char *host_fun,
                                                    char *device_fun, const char *device_name,
                                                    int thread_limit, uint3 *tid, uint3 *bid,
                                                    dim3 *bDim, dim3 *gDim, int *wSize)
    {
      GPUProcessor *local = find_local_gpu();
      if (local == NULL)
      {
        GPUProcessor::defer_function_load(fat_bin, host_fun, device_fun);
        return;
      }
      local->internal_register_function(fat_bin, host_fun, device_fun);
    }

    char GPUProcessor::internal_init_module(void **fat_bin)
    {
      // We don't really care about managed runtimes
      return 1;
    }

    /*static*/ char GPUProcessor::init_module(void **fat_bin)
    {
      GPUProcessor *local = find_local_gpu();
      if (local == NULL)
        return 1;
      return local->internal_init_module(fat_bin);
    }
#endif
    
    GPUProcessor::LaunchConfig::LaunchConfig(dim3 _grid, dim3 _block, size_t _shared)
      : grid(_grid), block(_block), shared(_shared)
    {}

    void GPUProcessor::configure_call(dim3 grid_dim,
				      dim3 block_dim,
				      size_t shared_mem,
				      cudaStream_t stream)
    {
      launch_configs.push_back(LaunchConfig(grid_dim, block_dim, shared_mem));
    }

    void GPUProcessor::setup_argument(const void *arg,
				      size_t size, size_t offset)
    {
      size_t required = offset + size;

      if(required > kernel_args.size())
	kernel_args.resize(required);

      memcpy(&kernel_args[offset], arg, size);
    }

    void GPUProcessor::launch(const void *func)
    {
      // make sure we have a launch config
      assert(!launch_configs.empty());
      LaunchConfig &config = launch_configs.back();

      // Find our function
      CUfunction f = gpu->lookup_function(func);

      size_t arg_size = kernel_args.size();
      void *extra[] = { 
        CU_LAUNCH_PARAM_BUFFER_POINTER, &kernel_args[0],
        CU_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        CU_LAUNCH_PARAM_END
      };

      // Launch the kernel on our stream dammit!
      CHECK_CU( cuLaunchKernel(f, 
			       config.grid.x, config.grid.y, config.grid.z,
                               config.block.x, config.block.y, config.block.z,
                               config.shared,
			       gpu->get_current_task_stream()->get_stream(),
			       NULL, extra) );

      // pop the config we just used
      launch_configs.pop_back();

      // clear out the kernel args
      kernel_args.clear();
    }

    void GPUProcessor::gpu_memcpy(void *dst, const void *src, size_t size,
				  cudaMemcpyKind kind)
    {
      CUstream current = gpu->get_current_task_stream()->get_stream();
      // the synchronous copy still uses cuMemcpyAsync so that we can limit the
      //  synchronization to just the right stream
      CHECK_CU( cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, size, current) );
      CHECK_CU( cuStreamSynchronize(current) );
    }

    void GPUProcessor::gpu_memcpy_async(void *dst, const void *src, size_t size,
					cudaMemcpyKind kind, cudaStream_t stream)
    {
      CUstream current = gpu->get_current_task_stream()->get_stream();
      CHECK_CU( cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, size, current) );
      // no synchronization here
    }

    void GPUProcessor::gpu_memcpy_to_symbol(const void *dst, const void *src,
					    size_t size, size_t offset,
					    cudaMemcpyKind kind)
    {
      CUstream current = gpu->get_current_task_stream()->get_stream();
      CUdeviceptr var_base = gpu->lookup_variable(dst);
      CHECK_CU( cuMemcpyAsync(var_base + offset,
			      (CUdeviceptr)src, size, current) );
      CHECK_CU( cuStreamSynchronize(current) );
    }

    void GPUProcessor::gpu_memcpy_to_symbol_async(const void *dst, const void *src,
						  size_t size, size_t offset,
						  cudaMemcpyKind kind, cudaStream_t stream)
    {
      CUstream current = gpu->get_current_task_stream()->get_stream();
      CUdeviceptr var_base = gpu->lookup_variable(dst);
      CHECK_CU( cuMemcpyAsync(var_base + offset,
			      (CUdeviceptr)src, size, current) );
      // no synchronization here
    }

    void GPUProcessor::gpu_memcpy_from_symbol(void *dst, const void *src,
					      size_t size, size_t offset,
					      cudaMemcpyKind kind)
    {
      CUstream current = gpu->get_current_task_stream()->get_stream();
      CUdeviceptr var_base = gpu->lookup_variable(src);
      CHECK_CU( cuMemcpyAsync((CUdeviceptr)dst,
			      var_base + offset,
			      size, current) );
      CHECK_CU( cuStreamSynchronize(current) );
    }

    void GPUProcessor::gpu_memcpy_from_symbol_async(void *dst, const void *src,
						    size_t size, size_t offset,
						    cudaMemcpyKind kind, cudaStream_t stream)
    {
      CUstream current = gpu->get_current_task_stream()->get_stream();
      CUdeviceptr var_base = gpu->lookup_variable(src);
      CHECK_CU( cuMemcpyAsync((CUdeviceptr)dst,
			      var_base + offset,
			      size, current) );
      // no synchronization here
    }

#if 0
    /*static*/ cudaError_t GPUProcessor::device_synchronize(void)
    {
      // Users are dumb, never let them mess us up
      GPUProcessor *local = find_local_gpu();
      assert(local != NULL);
      return local->internal_stream_synchronize();
    }

    /*static*/ cudaError_t GPUProcessor::set_shared_memory_config(cudaSharedMemConfig config)
    {
      CHECK_CU( cuCtxSetSharedMemConfig(
        (config == cudaSharedMemBankSizeDefault) ? CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE :
        (config == cudaSharedMemBankSizeFourByte) ? CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE :
                                                    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE) );
      return cudaSuccess;
    }

    /*static*/ const char* GPUProcessor::get_error_string(cudaError_t error)
    {
      const char *result;
      CHECK_CU( cuGetErrorString((CUresult)error, &result) );
      return result;
    }

    /*static*/ cudaError_t GPUProcessor::get_device(int *device)
    {
      GPUProcessor *local = find_local_gpu();
      CHECK_CU( cuDeviceGet(device, local->gpu_index) );
      return cudaSuccess;
    }

    /*static*/ cudaError_t GPUProcessor::get_device_properties(cudaDeviceProp *prop, int device)
    {
      CHECK_CU( cuDeviceGetName(prop->name, 255, device) );
      CHECK_CU( cuDeviceTotalMem(&(prop->totalGlobalMem), device) );
#define GET_DEVICE_PROP(member, name)   \
      {                                 \
        int tmp;                        \
        CHECK_CU( cuDeviceGetAttribute(&tmp, CU_DEVICE_ATTRIBUTE_##name, device) ); \
        prop->member = tmp;             \
      }
      // SCREW TEXTURES AND SURFACES FOR NOW!
      GET_DEVICE_PROP(sharedMemPerBlock, MAX_SHARED_MEMORY_PER_BLOCK);
      GET_DEVICE_PROP(regsPerBlock, MAX_REGISTERS_PER_BLOCK);
      GET_DEVICE_PROP(warpSize, WARP_SIZE);
      GET_DEVICE_PROP(memPitch, MAX_PITCH);
      GET_DEVICE_PROP(maxThreadsPerBlock, MAX_THREADS_PER_BLOCK);
      GET_DEVICE_PROP(maxThreadsDim[0], MAX_BLOCK_DIM_X);
      GET_DEVICE_PROP(maxThreadsDim[1], MAX_BLOCK_DIM_Y);
      GET_DEVICE_PROP(maxThreadsDim[2], MAX_BLOCK_DIM_Z);
      GET_DEVICE_PROP(maxGridSize[0], MAX_GRID_DIM_X);
      GET_DEVICE_PROP(maxGridSize[1], MAX_GRID_DIM_Y);
      GET_DEVICE_PROP(maxGridSize[2], MAX_GRID_DIM_Z);
      GET_DEVICE_PROP(clockRate, CLOCK_RATE);
      GET_DEVICE_PROP(totalConstMem, TOTAL_CONSTANT_MEMORY);
      GET_DEVICE_PROP(major, COMPUTE_CAPABILITY_MAJOR);
      GET_DEVICE_PROP(minor, COMPUTE_CAPABILITY_MINOR);
      GET_DEVICE_PROP(deviceOverlap, GPU_OVERLAP);
      GET_DEVICE_PROP(multiProcessorCount, MULTIPROCESSOR_COUNT);
      GET_DEVICE_PROP(kernelExecTimeoutEnabled, KERNEL_EXEC_TIMEOUT);
      GET_DEVICE_PROP(integrated, INTEGRATED);
      GET_DEVICE_PROP(canMapHostMemory, CAN_MAP_HOST_MEMORY);
      GET_DEVICE_PROP(computeMode, COMPUTE_MODE);
      GET_DEVICE_PROP(concurrentKernels, CONCURRENT_KERNELS);
      GET_DEVICE_PROP(ECCEnabled, ECC_ENABLED);
      GET_DEVICE_PROP(pciBusID, PCI_BUS_ID);
      GET_DEVICE_PROP(pciDeviceID, PCI_DEVICE_ID);
      GET_DEVICE_PROP(pciDomainID, PCI_DOMAIN_ID);
      GET_DEVICE_PROP(tccDriver, TCC_DRIVER);
      GET_DEVICE_PROP(asyncEngineCount, ASYNC_ENGINE_COUNT);
      GET_DEVICE_PROP(unifiedAddressing, UNIFIED_ADDRESSING);
      GET_DEVICE_PROP(memoryClockRate, MEMORY_CLOCK_RATE);
      GET_DEVICE_PROP(memoryBusWidth, GLOBAL_MEMORY_BUS_WIDTH);
      GET_DEVICE_PROP(l2CacheSize, L2_CACHE_SIZE);
      GET_DEVICE_PROP(maxThreadsPerMultiProcessor, MAX_THREADS_PER_MULTIPROCESSOR);
      GET_DEVICE_PROP(streamPrioritiesSupported, STREAM_PRIORITIES_SUPPORTED);
      GET_DEVICE_PROP(globalL1CacheSupported, GLOBAL_L1_CACHE_SUPPORTED);
      GET_DEVICE_PROP(localL1CacheSupported, LOCAL_L1_CACHE_SUPPORTED);
      GET_DEVICE_PROP(sharedMemPerMultiprocessor, MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
      GET_DEVICE_PROP(regsPerMultiprocessor, MAX_REGISTERS_PER_MULTIPROCESSOR);
      GET_DEVICE_PROP(managedMemory, MANAGED_MEMORY);
      GET_DEVICE_PROP(isMultiGpuBoard, MULTI_GPU_BOARD);
      GET_DEVICE_PROP(multiGpuBoardGroupID, MULTI_GPU_BOARD_GROUP_ID);
#undef GET_DEVICE_PROP
      return cudaSuccess;
    }

    /*static*/ cudaError_t GPUProcessor::get_func_attributes(cudaFuncAttributes *attr,
                                                             const void *func)
    {
      CUfunction handle;
      GPUProcessor *local = find_local_gpu();
      local->find_function_handle(func, &handle);
      CHECK_CU( cuFuncGetAttribute(&(attr->binaryVersion), CU_FUNC_ATTRIBUTE_BINARY_VERSION, handle) );
      CHECK_CU( cuFuncGetAttribute(&(attr->cacheModeCA), CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, handle) );
      int tmp;
      CHECK_CU( cuFuncGetAttribute(&tmp, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, handle) );
      attr->constSizeBytes = tmp;
      CHECK_CU( cuFuncGetAttribute(&tmp, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, handle) );
      attr->localSizeBytes = tmp;
      CHECK_CU( cuFuncGetAttribute(&(attr->maxThreadsPerBlock), CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, handle) );
      CHECK_CU( cuFuncGetAttribute(&(attr->numRegs), CU_FUNC_ATTRIBUTE_NUM_REGS, handle) );
      CHECK_CU( cuFuncGetAttribute(&(attr->ptxVersion), CU_FUNC_ATTRIBUTE_PTX_VERSION, handle) );
      CHECK_CU( cuFuncGetAttribute(&tmp, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, handle) );
      attr->sharedSizeBytes = tmp;
      return cudaSuccess;
    }
#endif

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPU

    GPU::GPU(CudaModule *_module, GPUInfo *_info, GPUWorker *_worker,
	     int num_streams)
      : module(_module), info(_info), worker(_worker)
      , proc(0), fbmem(0), current_stream(0)
    {
      // create a CUDA context for our device - automatically becomes current
      CHECK_CU( cuCtxCreate(&context, 
			    CU_CTX_MAP_HOST | CU_CTX_SCHED_BLOCKING_SYNC,
			    info->device) );

      event_pool.init_pool();

      host_to_device_stream = new GPUStream(this, worker);
      device_to_host_stream = new GPUStream(this, worker);
      device_to_device_stream = new GPUStream(this, worker);
      peer_to_peer_stream = new GPUStream(this, worker);

      task_streams.resize(num_streams);
      for(int idx = 0; idx < num_streams; idx++)
	task_streams[idx] = new GPUStream(this, worker);

      pop_context();

      // now hook into the cuda runtime fatbin/etc. registration path
      GlobalRegistrations::add_gpu_context(this);
    }

    GPU::~GPU(void)
    {
      push_context();

      event_pool.empty_pool();

      // destroy streams
      delete host_to_device_stream;
      delete device_to_host_stream;
      delete device_to_device_stream;
      delete peer_to_peer_stream;
      while(!task_streams.empty()) {
	delete task_streams.back();
	task_streams.pop_back();
      }

      // free memory
      CHECK_CU( cuMemFree(fbmem_base) );

      CHECK_CU( cuCtxDestroy(context) );
    }

    void GPU::push_context(void)
    {
      CHECK_CU( cuCtxPushCurrent(context) );
    }

    void GPU::pop_context(void)
    {
      // the context we pop had better be ours...
      CUcontext popped;
      CHECK_CU( cuCtxPopCurrent(&popped) );
      assert(popped == context);
    }

    void GPU::create_processor(RuntimeImpl *runtime, size_t stack_size)
    {
      Processor p = runtime->next_local_processor_id();
      proc = new GPUProcessor(this, p,
			      runtime->core_reservation_set(),
			      stack_size);
      runtime->add_processor(proc);
    }

    void GPU::create_fb_memory(RuntimeImpl *runtime, size_t size)
    {
      // need the context so we can get an allocation in the right place
      {
	AutoGPUContext agc(this);

	CHECK_CU( cuMemAlloc(&fbmem_base, size) );
      }

      Memory m = runtime->next_local_memory_id();
      fbmem = new GPUFBMemory(m, this, fbmem_base, size);
      runtime->add_memory(fbmem);
    }

    void GPU::register_fat_binary(const FatBin *fatbin)
    {
      AutoGPUContext agc(this);

      log_gpu.info() << "registering fat binary " << fatbin << " with GPU " << this;

      // have we see this one already?
      if(device_modules.count(fatbin) > 0) {
	log_gpu.warning() << "duplicate registration of fat binary data " << fatbin;
	return;
      }

      if(fatbin->data != 0) {
	// binary data to be loaded with cuModuleLoad(Ex)
	CUmodule module = load_cuda_module(fatbin->data);
	device_modules[fatbin] = module;
	return;
      }

      assert(0);
    }
    
    void GPU::register_variable(const RegisteredVariable *var)
    {
      AutoGPUContext agc(this);

      log_gpu.info() << "registering variable " << var->device_name << " (" << var->host_var << ") with GPU " << this;

      // have we seen it already?
      if(device_variables.count(var->host_var) > 0) {
	log_gpu.warning() << "duplicate registration of variable " << var->device_name;
	return;
      }

      // get the module it lives in
      std::map<const FatBin *, CUmodule>::const_iterator it = device_modules.find(var->fat_bin);
      assert(it != device_modules.end());
      CUmodule module = it->second;

      CUdeviceptr ptr;
      size_t size;
      CHECK_CU( cuModuleGetGlobal(&ptr, &size, module, var->device_name) );
      device_variables[var->host_var] = ptr;
    }
    
    void GPU::register_function(const RegisteredFunction *func)
    {
      AutoGPUContext agc(this);

      log_gpu.info() << "registering function " << func->device_fun << " (" << func->host_fun << ") with GPU " << this;

      // have we seen it already?
      if(device_functions.count(func->host_fun) > 0) {
	log_gpu.warning() << "duplicate registration of function " << func->device_fun;
	return;
      }

      // get the module it lives in
      std::map<const FatBin *, CUmodule>::const_iterator it = device_modules.find(func->fat_bin);
      assert(it != device_modules.end());
      CUmodule module = it->second;

      CUfunction f;
      CHECK_CU( cuModuleGetFunction(&f, module, func->device_fun) );
      device_functions[func->host_fun] = f;
    }

    CUfunction GPU::lookup_function(const void *func)
    {
      std::map<const void *, CUfunction>::iterator finder = device_functions.find(func);
      assert(finder != device_functions.end());
      return finder->second;
    }

    CUdeviceptr GPU::lookup_variable(const void *var)
    {
      std::map<const void *, CUdeviceptr>::iterator finder = device_variables.find(var);
      assert(finder != device_variables.end());
      return finder->second;
    }

    CUmodule GPU::load_cuda_module(const void *data)
    {
      const unsigned num_options = 4;
      CUjit_option jit_options[num_options];
      void*        option_vals[num_options];
      const size_t buffer_size = 16384;
      char* log_info_buffer = (char*)malloc(buffer_size);
      char* log_error_buffer = (char*)malloc(buffer_size);
      jit_options[0] = CU_JIT_INFO_LOG_BUFFER;
      jit_options[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
      jit_options[2] = CU_JIT_ERROR_LOG_BUFFER;
      jit_options[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
      option_vals[0] = log_info_buffer;
      option_vals[1] = (void*)buffer_size;
      option_vals[2] = log_error_buffer;
      option_vals[3] = (void*)buffer_size;
      CUmodule module;
      CUresult result = cuModuleLoadDataEx(&module, data, num_options, 
                                           jit_options, option_vals); 
      if (result != CUDA_SUCCESS)
      {
#ifdef __MACH__
        if (result == CUDA_ERROR_OPERATING_SYSTEM) {
          log_gpu.error("ERROR: Device side asserts are not supported by the "
                              "CUDA driver for MAC OSX, see NVBugs 1628896.");
        }
#endif
        if (result == CUDA_ERROR_NO_BINARY_FOR_GPU) {
          log_gpu.error("ERROR: The binary was compiled for the wrong GPU "
                              "architecture. Update the 'GPU_ARCH' flag at the top "
                              "of runtime/runtime.mk to match your current GPU "
                              "architecture.");
        }
        log_gpu.error("Failed to load CUDA module! Error log: %s", 
                log_error_buffer);
#if CUDA_VERSION >= 6050
        const char *name, *str;
        CHECK_CU( cuGetErrorName(result, &name) );
        CHECK_CU( cuGetErrorString(result, &str) );
        fprintf(stderr,"CU: cuModuleLoadDataEx = %d (%s): %s\n",
                result, name, str);
#else
        fprintf(stderr,"CU: cuModuleLoadDataEx = %d\n", result);
#endif
        assert(0);
      }
      else
        log_gpu.info("Loaded CUDA Module. JIT Output: %s", log_info_buffer);
      free(log_info_buffer);
      free(log_error_buffer);
      return module;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class AutoGPUContext

    AutoGPUContext::AutoGPUContext(GPU& _gpu)
      : gpu(&_gpu)
    {
      gpu->push_context();
    }

    AutoGPUContext::AutoGPUContext(GPU *_gpu)
      : gpu(_gpu)
    {
      gpu->push_context();
    }

    AutoGPUContext::~AutoGPUContext(void)
    {
      gpu->pop_context();
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class CudaModule

    // our interface to the rest of the runtime

    REGISTER_REALM_MODULE(CudaModule);

    CudaModule::CudaModule(void)
      : Module("cuda")
      , cfg_zc_mem_size_in_mb(64)
      , cfg_fb_mem_size_in_mb(256)
      , cfg_num_gpus(0)
      , cfg_gpu_streams(12)
      , cfg_use_background_workers(true)
      , cfg_use_shared_worker(true)
      , cfg_fences_use_callbacks(false)
      , shared_worker(0), zcmem_cpu_base(0), zcmem(0)
    {}
      
    CudaModule::~CudaModule(void)
    {}

    /*static*/ Module *CudaModule::create_module(RuntimeImpl *runtime,
						 std::vector<std::string>& cmdline)
    {
      // before we do anything, make sure there's a CUDA driver and GPUs to talk to
      std::vector<GPUInfo *> infos;
      {
	CUresult ret = cuInit(0);
	if(ret != CUDA_SUCCESS) {
	  log_gpu.warning() << "cuInit(0) returned " << ret << " - module not loaded";
	  return 0;
	}

	int num_devices;
	CHECK_CU( cuDeviceGetCount(&num_devices) );
	for(int i = 0; i < num_devices; i++) {
	  GPUInfo *info = new GPUInfo;

	  // TODO: consider environment variables or other ways to tell if certain
	  //  GPUs should be ignored

	  info->index = i;
	  CHECK_CU( cuDeviceGet(&info->device, i) );
	  CHECK_CU( cuDeviceGetName(info->name, GPUInfo::MAX_NAME_LEN, info->device) );
	  CHECK_CU( cuDeviceComputeCapability(&info->compute_major,
					      &info->compute_minor,
					      info->device) );
	  CHECK_CU( cuDeviceTotalMem(&info->total_mem, info->device) );
	  CHECK_CU( cuDeviceGetProperties(&info->props, info->device) );

	  log_gpu.info() << "GPU #" << i << ": " << info->name << " ("
			 << info->compute_major << '.' << info->compute_minor
			 << ") " << (info->total_mem >> 20) << " MB";

	  infos.push_back(info);
	}

	if(infos.empty()) {
	  log_gpu.warning() << "no CUDA-capable GPUs found - module not loaded";
	  return 0;
	}

	// query peer-to-peer access (all pairs)
	for(std::vector<GPUInfo *>::iterator it1 = infos.begin();
	    it1 != infos.end();
	    it1++)
	  for(std::vector<GPUInfo *>::iterator it2 = infos.begin();
	      it2 != infos.end();
	      it2++)
	    if(it1 != it2) {
	      int can_access;
	      CHECK_CU( cuDeviceCanAccessPeer(&can_access,
					      (*it1)->device,
					      (*it2)->device) );
	      if(can_access)
		(*it1)->peers.insert((*it2)->device);
	    }
      }

      CudaModule *m = new CudaModule;

      // give the gpu info we assembled to the module
      m->gpu_info.swap(infos);

      // first order of business - read command line parameters
      {
	CommandLineParser cp;

	cp.add_option_int("-ll:fsize", m->cfg_fb_mem_size_in_mb)
	  .add_option_int("-ll:zsize", m->cfg_zc_mem_size_in_mb)
	  .add_option_int("-ll:gpu", m->cfg_num_gpus)
	  .add_option_int("-ll:streams", m->cfg_gpu_streams)
	  .add_option_int("-ll:gpuworker", m->cfg_use_shared_worker)
	  .add_option_int("-ll:pin", m->cfg_pin_sysmem);
	
	bool ok = cp.parse_command_line(cmdline);
	if(!ok) {
	  log_gpu.error() << "error reading CUDA command line parameters";
	  exit(1);
	}
      }

      return m;
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void CudaModule::initialize(RuntimeImpl *runtime)
    {
      Module::initialize(runtime);

      // sanity-check: do we even have enough gpus?
      if(cfg_num_gpus > gpu_info.size()) {
	log_gpu.fatal() << cfg_num_gpus << " GPUs requested, but only " << gpu_info.size() << " available!";
	assert(false);
      }

      // if we are using a shared worker, create that next
      if(cfg_use_shared_worker) {
	shared_worker = new GPUWorker;

	if(cfg_use_background_workers)
	  shared_worker->start_background_thread(runtime->core_reservation_set(),
						 1 << 20); // hardcoded worker stack size
      }

      // just use the GPUs in order right now
      gpus.resize(cfg_num_gpus);
      for(unsigned i = 0; i < cfg_num_gpus; i++) {
	// either create a worker for this GPU or use the shared one
	GPUWorker *worker;
	if(cfg_use_shared_worker) {
	  worker = shared_worker;
	} else {
	  worker = new GPUWorker;

	  if(cfg_use_background_workers)
	    worker->start_background_thread(runtime->core_reservation_set(),
					    1 << 20); // hardcoded worker stack size
	}

	GPU *g = new GPU(this, gpu_info[i], worker, cfg_gpu_streams);

	if(!cfg_use_shared_worker)
	  dedicated_workers[g] = worker;

	gpus[i] = g;
      }
#if 0
        if (num_local_gpus > (peer_gpus.size() + dumb_gpus.size()))
        {
          printf("Requested %d GPUs, but only %ld GPUs exist on node %d\n",
            num_local_gpus, peer_gpus.size()+dumb_gpus.size(), gasnet_mynode());
          assert(false);
        }

	// TODO: let the CUDA module actually parse config variables
	assert(gpu_worker_thread == true);

#endif
    }

    // create any memories provided by this module (default == do nothing)
    //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
    void CudaModule::create_memories(RuntimeImpl *runtime)
    {
      Module::create_memories(runtime);

      // each GPU needs its FB memory
      for(std::vector<GPU *>::iterator it = gpus.begin();
	  it != gpus.end();
	  it++)
	(*it)->create_fb_memory(runtime, cfg_fb_mem_size_in_mb << 20);

      // a single ZC memory for everybody
      if((cfg_zc_mem_size_in_mb > 0) && !gpus.empty()) {
	CUdeviceptr zcmem_gpu_base;
	// borrow GPU 0's context for the allocation call
	{
	  AutoGPUContext agc(gpus[0]);

	  CHECK_CU( cuMemHostAlloc(&zcmem_cpu_base, 
				   cfg_zc_mem_size_in_mb << 20,
				   CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP) );
	  CHECK_CU( cuMemHostGetDevicePointer(&zcmem_gpu_base,
					      zcmem_cpu_base,
					      0) );
	  // right now there are asssumptions in several places that unified addressing keeps
	  //  the CPU and GPU addresses the same
	  assert(zcmem_cpu_base == (void *)zcmem_gpu_base);
	}

	Memory m = runtime->next_local_memory_id();
	zcmem = new GPUZCMemory(m, zcmem_gpu_base, zcmem_cpu_base, 
				cfg_zc_mem_size_in_mb << 20);
	runtime->add_memory(zcmem);

	// add the ZC memory as a pinned memory to all GPUs
	for(unsigned i = 0; i < gpus.size(); i++) {
	  CUdeviceptr gpuptr;
	  CUresult ret;
	  {
	    AutoGPUContext agc(gpus[i]);
	    ret = cuMemHostGetDevicePointer(&gpuptr, zcmem_cpu_base, 0);
	  }
	  if((ret == CUDA_SUCCESS) && (gpuptr == zcmem_gpu_base)) {
	    gpus[i]->pinned_sysmems.insert(zcmem->me);
	  } else {
	    log_gpu.warning() << "GPU #" << i << " has an unexpected mapping for ZC memory!";
	  }
	}
      }

#if 0
	  Memory m = ID(ID::ID_MEMORY,
			gasnet_mynode(),
			n->memories.size(), 0).convert<Memory>();
	  Cuda::GPUFBMemory *fbm = new Cuda::GPUFBMemory(m, gp);
	  n->memories.push_back(fbm);

	  gpu_fbmems[gp] = fbm;

	  Memory m2 = ID(ID::ID_MEMORY,
			 gasnet_mynode(),
			 n->memories.size(), 0).convert<Memory>();
	  Cuda::GPUZCMemory *zcm = new Cuda::GPUZCMemory(m2, gp);
	  n->memories.push_back(zcm);

	  gpu_zcmems[gp] = zcm;
#endif
    }

    // create any processors provided by the module (default == do nothing)
    //  (each new ProcessorImpl should use a Processor from
    //   RuntimeImpl::next_local_processor_id)
    void CudaModule::create_processors(RuntimeImpl *runtime)
    {
      Module::create_processors(runtime);

      // each GPU needs a processor
      for(std::vector<GPU *>::iterator it = gpus.begin();
	  it != gpus.end();
	  it++)
	(*it)->create_processor(runtime,
				2 << 20); // TODO: don't use hardcoded stack size...
#if 0
	  Processor p = ID(ID::ID_PROCESSOR, 
			   gasnet_mynode(), 
			   n->processors.size()).convert<Processor>();
	  //printf("GPU's ID is " IDFMT "\n", p.id);
 	  Cuda::GPUProcessor *gp = new Cuda::GPUProcessor(p, core_reservations,
                                              (i < peer_gpus.size() ?
                                                peer_gpus[i] : 
                                                dumb_gpus[i-peer_gpus.size()]), 
                                              zc_mem_size_in_mb << 20,
                                              fb_mem_size_in_mb << 20,
                                              stack_size_in_mb << 20,
                                              num_gpu_streams);
	  n->processors.push_back(gp);
	  local_gpus.push_back(gp);
#endif
    }

    // create any DMA channels provided by the module (default == do nothing)
    void CudaModule::create_dma_channels(RuntimeImpl *runtime)
    {
      // before we create dma channels, see how many of the system memory ranges
      //  we can register with CUDA
      if(cfg_pin_sysmem && !gpus.empty()) {
	std::vector<MemoryImpl *>& local_mems = runtime->nodes[gasnet_mynode()].memories;
	for(std::vector<MemoryImpl *>::iterator it = local_mems.begin();
	    it != local_mems.end();
	    it++) {
	  // ignore FB/ZC memories or anything that doesn't have a "direct" pointer
	  if(((*it)->kind == MemoryImpl::MKIND_GPUFB) ||
	     ((*it)->kind == MemoryImpl::MKIND_ZEROCOPY))
	    continue;

	  void *base = (*it)->get_direct_ptr(0, (*it)->size);
	  if(base == 0)
	    continue;

	  // using GPU 0's context, attempt a portable registration
	  CUresult ret;
	  {
	    AutoGPUContext agc(gpus[0]);
	    ret = cuMemHostRegister(base, (*it)->size, 
				    CU_MEMHOSTREGISTER_PORTABLE |
				    CU_MEMHOSTREGISTER_DEVICEMAP);
	  }
	  if(ret != CUDA_SUCCESS) {
	    log_gpu.info() << "failed to register mem " << (*it)->me << " (" << base << " + " << (*it)->size << ") : "
			   << ret;
	    continue;
	  }

	  // now go through each GPU and verify that it got a GPU pointer (it may not match the CPU
	  //  pointer, but that's ok because we'll never refer to it directly)
	  for(unsigned i = 0; i < gpus.size(); i++) {
	    CUdeviceptr gpuptr;
	    CUresult ret;
	    {
	      AutoGPUContext agc(gpus[i]);
	      ret = cuMemHostGetDevicePointer(&gpuptr, zcmem_cpu_base, 0);
	    }
	    if(ret == CUDA_SUCCESS) {
	      // no test for && ((void *)gpuptr == base)) {
	      log_gpu.info() << "memory " << (*it)->me << " successfully registered with GPU " << gpus[i]->proc->me;
	      gpus[i]->pinned_sysmems.insert((*it)->me);
	    } else {
	      log_gpu.warning() << "GPU #" << i << " has no mapping for registered memory (" << base << ") !?";
	    }
	  }
	}
      }

      // now actually let each GPU make its channels
      for(std::vector<GPU *>::iterator it = gpus.begin();
	  it != gpus.end();
	  it++)
	(*it)->create_dma_channels(runtime);

      Module::create_dma_channels(runtime);
#if 0
      gp->create_dma_channels(this);
        // Now pin any CPU memories
        if(pin_sysmem_for_gpu) {
	  for(std::vector<MemoryImpl *>::iterator it = n->memories.begin();
	      it != n->memories.end();
	      it++)
	    if((*it)->kind == MemoryImpl::MKIND_SYSMEM) {
	      LocalCPUMemory *m = (LocalCPUMemory *)(*it);
	      // TODO: should this really only be the first GPU!?
	      local_gpus[0]->register_host_memory(m);
	    }
	}

        // Register peer access for any GPUs which support it
        if ((num_local_gpus > 1) && (peer_gpus.size() > 1))
        {
          unsigned peer_count = (num_local_gpus < peer_gpus.size()) ? 
                                  num_local_gpus : peer_gpus.size();
          // Needs to go both ways so register in all directions
          for (unsigned i = 0; i < peer_count; i++)
          {
            CUdevice device;
            CHECK_CU( cuDeviceGet(&device, peer_gpus[i]) );
            for (unsigned j = 0; j < peer_count; j++)
            {
              if (i == j) continue;
              CUdevice peer;
              CHECK_CU( cuDeviceGet(&peer, peer_gpus[j]) );
              int can_access;
              CHECK_CU( cuDeviceCanAccessPeer(&can_access, device, peer) );
              if (can_access)
                local_gpus[i]->enable_peer_access(local_gpus[j]);
            }
          }
        }
      }
#endif
    }

    // create any code translators provided by the module (default == do nothing)
    void CudaModule::create_code_translators(RuntimeImpl *runtime)
    {
      Module::create_code_translators(runtime);
    }

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void CudaModule::cleanup(void)
    {
      // clean up worker(s)
      if(shared_worker) {
	if(cfg_use_background_workers)
	  shared_worker->shutdown_background_thread();

	delete shared_worker;
	shared_worker = 0;
      }
      for(std::map<GPU *, GPUWorker *>::iterator it = dedicated_workers.begin();
	  it != dedicated_workers.end();
	  it++) {
	GPUWorker *worker = it->second;

	if(cfg_use_background_workers)
	  worker->shutdown_background_thread();

	delete worker;
      }
      dedicated_workers.clear();

      // use GPU 0's context to free ZC memory (if any)
      if(zcmem_cpu_base) {
	assert(!gpus.empty());
	AutoGPUContext agc(gpus[0]);
	CHECK_CU( cuMemFreeHost(zcmem_cpu_base) );
      }

      for(std::vector<GPU *>::iterator it = gpus.begin();
	  it != gpus.end();
	  it++)
	delete *it;
      gpus.clear();
      
      Module::cleanup();
    }


  }; // namespace Cuda
}; // namespace Realm

