/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#include "realm/cuda/cuda_module.h"

#include "realm/tasks.h"
#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/event_impl.h"

#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/channel.h"

#include "realm/cuda/cudart_hijack.h"

#include "realm/activemsg.h"
#include "realm/utils.h"

#ifdef REALM_USE_VALGRIND_ANNOTATIONS
#include <valgrind/memcheck.h>
#endif

#include <stdio.h>
#include <string.h>

namespace Realm {
  namespace Cuda {

    Logger log_gpu("gpu");
    Logger log_gpudma("gpudma");
    Logger log_cudart("cudart");

#ifdef EVENT_GRAPH_TRACE
    extern Logger log_event_graph;
#endif
    Logger log_stream("gpustream");

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

	{
	  AutoGPUContext agc(gpu);
	  copy->execute(this);
	}

	// TODO: recycle these
	delete copy;

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
	if(res != CUDA_SUCCESS) {
	  const char *ename = 0;
	  const char *estr = 0;
	  cuGetErrorName(res, &ename);
	  cuGetErrorString(res, &estr);
	  log_gpu.fatal() << "CUDA error reported on GPU " << gpu->info->index << ": " << estr << " (" << ename << ")";
	  assert(0);
	}

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
	  fence->mark_finished(true /*successful*/);

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
	elmt_size(_bytes), notification(_notification)
    {}

    GPUMemcpy1D::~GPUMemcpy1D(void)
    {}

    void GPUMemcpy1D::do_span(off_t pos, size_t len)
    {
      off_t span_start = pos * elmt_size;
      size_t span_bytes = len * elmt_size;

      CUstream raw_stream = local_stream->get_stream();
      log_stream.debug() << "memcpy added to stream " << raw_stream;

      switch (kind)
      {
        case GPU_MEMCPY_HOST_TO_DEVICE:
          {
            CHECK_CU( cuMemcpyHtoDAsync((CUdeviceptr)(((char*)dst)+span_start),
                                        (((char*)src)+span_start),
                                        span_bytes,
                                        raw_stream) );
            break;
          }
        case GPU_MEMCPY_DEVICE_TO_HOST:
          {
            CHECK_CU( cuMemcpyDtoHAsync((((char*)dst)+span_start),
                                        (CUdeviceptr)(((char*)src)+span_start),
                                        span_bytes,
                                        raw_stream) );
#ifdef REALM_USE_VALGRIND_ANNOTATIONS
	    VALGRIND_MAKE_MEM_DEFINED((((char*)dst)+span_start), span_bytes);
#endif
            break;
          }
        case GPU_MEMCPY_DEVICE_TO_DEVICE:
          {
            CHECK_CU( cuMemcpyDtoDAsync((CUdeviceptr)(((char*)dst)+span_start),
                                        (CUdeviceptr)(((char*)src)+span_start),
                                        span_bytes,
                                        raw_stream) );
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
                                        raw_stream) );
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
      do_span(0, 1);
      
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
		      dst, src, (long)dst_stride, (long)src_stride, (long)bytes, (long)lines, kind); 
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
		      dst, src, (long)dst_stride, (long)src_stride, (long)bytes, (long)lines, kind);
    }

  ////////////////////////////////////////////////////////////////////////
  //
  // class GPUMemcpy3D
    GPUMemcpy3D::GPUMemcpy3D(GPU *_gpu,
                             void *_dst, const void *_src,
                             off_t _dst_stride, off_t _src_stride,
                             off_t _dst_height, off_t _src_height,
                             size_t _bytes, size_t _height, size_t _depth,
                             GPUMemcpyKind _kind,
                             GPUCompletionNotification *_notification)
       : GPUMemcpy(_gpu, _kind), dst(_dst), src(_src),
	dst_stride((_dst_stride < (off_t)_bytes) ? _bytes : _dst_stride), 
	src_stride((_src_stride < (off_t)_bytes) ? _bytes : _src_stride),
        dst_height((_dst_height < (off_t)_height) ? _height : _dst_height),
        src_height((_src_height < (off_t)_height) ? _height : _src_height),
	bytes(_bytes), height(_height), depth(_depth),
        notification(_notification)
    {}

    GPUMemcpy3D::~GPUMemcpy3D(void)
    {}
    
    void GPUMemcpy3D::execute(GPUStream *stream)
    {
      log_gpudma.info("gpu memcpy 3d: dst=%p src=%p"
                      "dst_off=%ld src_off=%ld dst_hei = %ld src_hei = %ld"
                      "bytes=%ld height=%ld depth=%ld kind=%d",
                      dst, src, (long)dst_stride, (long)src_stride,
                      (long)dst_height, (long)src_height, (long)bytes, (long)height,
                      (long)depth, kind);
      CUDA_MEMCPY3D copy_info;
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
      copy_info.srcHeight = src_height;
      copy_info.srcY = 0;
      copy_info.srcZ = 0;
      copy_info.srcXInBytes = 0;
      copy_info.srcLOD = 0;
      copy_info.dstDevice = (CUdeviceptr)dst;
      copy_info.dstHost = dst;
      copy_info.dstPitch = dst_stride;
      copy_info.dstHeight = dst_height;
      copy_info.dstY = 0;
      copy_info.dstZ = 0;
      copy_info.dstXInBytes = 0;
      copy_info.dstLOD = 0;
      copy_info.WidthInBytes = bytes;
      copy_info.Height = height;
      copy_info.Depth = depth;
      CHECK_CU( cuMemcpy3DAsync(&copy_info, stream->get_stream()) );

      if(notification)
        stream->add_notification(notification);

       log_gpudma.info("gpu memcpy 3d complete: dst=%p src=%p"
		       "dst_off=%ld src_off=%ld dst_hei = %ld src_hei = %ld"
		       "bytes=%ld height=%ld depth=%ld kind=%d",
		       dst, src, (long)dst_stride, (long)src_stride,
		       (long)dst_height, (long)src_height, (long)bytes, (long)height,
		       (long)depth, kind);
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GPUMemset1D

    GPUMemset1D::GPUMemset1D(GPU *_gpu,
		  void *_dst, size_t _bytes,
		  const void *_fill_data, size_t _fill_data_size,
		  GPUCompletionNotification *_notification)
      : GPUMemcpy(_gpu, GPU_MEMCPY_DEVICE_TO_DEVICE)
      , dst(_dst), bytes(_bytes)
      , fill_data_size(_fill_data_size)
      , notification(_notification)
    {
      if(fill_data_size <= MAX_DIRECT_SIZE) {
	memcpy(fill_data.direct, _fill_data, fill_data_size);
      } else {
	fill_data.indirect = new char[fill_data_size];
	assert(fill_data.indirect != 0);
	memcpy(fill_data.indirect, _fill_data, fill_data_size);
      }
    }

    GPUMemset1D::~GPUMemset1D(void)
    {
      if(fill_data_size > MAX_DIRECT_SIZE)
	delete[] fill_data.indirect;
    }

    void GPUMemset1D::execute(GPUStream *stream)
    {
      DetailedTimer::ScopedPush sp(TIME_COPY);
      log_gpudma.info("gpu memset: dst=%p bytes=%zd fill_data_size=%zd",
		      dst, bytes, fill_data_size);

      CUstream raw_stream = stream->get_stream();

      switch(fill_data_size) {
      case 1:
	{
	  CHECK_CU( cuMemsetD8Async(CUdeviceptr(dst), 
				    *reinterpret_cast<const unsigned char *>(fill_data.direct),
				    bytes,
				    raw_stream) );
	  break;
	}
      case 2:
	{
	  CHECK_CU( cuMemsetD16Async(CUdeviceptr(dst), 
				     *reinterpret_cast<const unsigned short *>(fill_data.direct),
				     bytes >> 1,
				     raw_stream) );
	  break;
	}
      case 4:
	{
	  CHECK_CU( cuMemsetD32Async(CUdeviceptr(dst), 
				     *reinterpret_cast<const unsigned int *>(fill_data.direct),
				     bytes >> 2,
				     raw_stream) );
	  break;
	}
      default:
	{
	  // use strided 2D memsets to deal with larger patterns
	  size_t elements = bytes / fill_data_size;
	  const char *srcdata = ((fill_data_size <= MAX_DIRECT_SIZE) ?
				   fill_data.direct :
				   fill_data.indirect);
	  // 16- and 32-bit fills must be aligned on every piece
	  if((fill_data_size & 3) == 0) {
	    for(size_t offset = 0; offset < fill_data_size; offset += 4) {
	      unsigned int val = *reinterpret_cast<const unsigned int *>(srcdata + offset);
	      CHECK_CU( cuMemsetD2D32Async(CUdeviceptr(dst) + offset,
					   fill_data_size /*pitch*/,
					   val,
					   1 /*width*/, elements /*height*/,
					   raw_stream) );
	    }
	  } else if((fill_data_size & 1) == 0) {
	    for(size_t offset = 0; offset < fill_data_size; offset += 2) {
	      unsigned short val = *reinterpret_cast<const unsigned short *>(srcdata + offset);
	      CHECK_CU( cuMemsetD2D16Async(CUdeviceptr(dst) + offset,
					   fill_data_size /*pitch*/,
					   val,
					   1 /*width*/, elements /*height*/,
					   raw_stream) );
	    }
	  } else {
	    for(size_t offset = 0; offset < fill_data_size; offset += 1) {
	      unsigned int val = *(srcdata + offset);
	      CHECK_CU( cuMemsetD2D8Async(CUdeviceptr(dst) + offset,
					  fill_data_size /*pitch*/,
					  val,
					  1 /*width*/, elements /*height*/,
					  raw_stream) );
	    }
	  }
	}
      }
      
      if(notification)
	stream->add_notification(notification);

      log_gpudma.info("gpu memset complete: dst=%p bytes=%zd fill_data_size=%zd",
		      dst, bytes, fill_data_size);
    }

  ////////////////////////////////////////////////////////////////////////
  //
  // class GPUMemset2D

    GPUMemset2D::GPUMemset2D(GPU *_gpu,
			     void *_dst, size_t _stride,
			     size_t _bytes, size_t _lines,
			     const void *_fill_data, size_t _fill_data_size,
			     GPUCompletionNotification *_notification)
      : GPUMemcpy(_gpu, GPU_MEMCPY_DEVICE_TO_DEVICE)
      , dst(_dst), dst_stride(_stride)
      , bytes(_bytes), lines(_lines)
      , fill_data_size(_fill_data_size)
      , notification(_notification)
    {
      if(fill_data_size <= MAX_DIRECT_SIZE) {
	memcpy(fill_data.direct, _fill_data, fill_data_size);
      } else {
	fill_data.indirect = new char[fill_data_size];
	assert(fill_data.indirect != 0);
	memcpy(fill_data.indirect, _fill_data, fill_data_size);
      }
    }

    GPUMemset2D::~GPUMemset2D(void)
    {
      if(fill_data_size > MAX_DIRECT_SIZE)
	delete[] fill_data.indirect;
    }

    void GPUMemset2D::execute(GPUStream *stream)
    {
      DetailedTimer::ScopedPush sp(TIME_COPY);
      log_gpudma.info("gpu memset 2d: dst=%p dst_off=%ld bytes=%zd lines=%zd fill_data_size=%zd",
		      dst, dst_stride, bytes, lines, fill_data_size);

      CUstream raw_stream = stream->get_stream();

      switch(fill_data_size) {
      case 1:
	{
	  CHECK_CU( cuMemsetD2D8Async(CUdeviceptr(dst), dst_stride,
				      *reinterpret_cast<const unsigned char *>(fill_data.direct),
				      bytes, lines,
				      raw_stream) );
	  break;
	}
      case 2:
	{
	  CHECK_CU( cuMemsetD2D16Async(CUdeviceptr(dst), dst_stride,
				       *reinterpret_cast<const unsigned short *>(fill_data.direct),
				       bytes >> 1, lines,
				       raw_stream) );
	  break;
	}
      case 4:
	{
	  CHECK_CU( cuMemsetD2D32Async(CUdeviceptr(dst), dst_stride,
				       *reinterpret_cast<const unsigned int *>(fill_data.direct),
				       bytes >> 2, lines,
				       raw_stream) );
	  break;
	}
      default:
	{
	  // use strided 2D memsets to deal with larger patterns
	  size_t elements = bytes / fill_data_size;
	  const char *srcdata = ((fill_data_size <= MAX_DIRECT_SIZE) ?
				   fill_data.direct :
				   fill_data.indirect);
	  // 16- and 32-bit fills must be aligned on every piece
	  if((fill_data_size & 3) == 0) {
	    for(size_t offset = 0; offset < fill_data_size; offset += 4) {
	      unsigned int val = *reinterpret_cast<const unsigned int *>(srcdata + offset);
	      for(size_t l = 0; l < lines; l++)
		CHECK_CU( cuMemsetD2D32Async(CUdeviceptr(dst) + offset + (l * dst_stride),
					     fill_data_size /*pitch*/,
					     val,
					     1 /*width*/, elements /*height*/,
					     raw_stream) );
	    }
	  } else if((fill_data_size & 1) == 0) {
	    for(size_t offset = 0; offset < fill_data_size; offset += 2) {
	      unsigned short val = *reinterpret_cast<const unsigned short *>(srcdata + offset);
	      for(size_t l = 0; l < lines; l++)
		CHECK_CU( cuMemsetD2D16Async(CUdeviceptr(dst) + offset + (l * dst_stride),
					     fill_data_size /*pitch*/,
					     val,
					     1 /*width*/, elements /*height*/,
					     raw_stream) );
	    }
	  } else {
	    for(size_t offset = 0; offset < fill_data_size; offset += 1) {
	      unsigned int val = *(srcdata + offset);
	      for(size_t l = 0; l < lines; l++)
		CHECK_CU( cuMemsetD2D8Async(CUdeviceptr(dst) + offset + (l * dst_stride),
					    fill_data_size /*pitch*/,
					    val,
					    1 /*width*/, elements /*height*/,
					    raw_stream) );
	    }
	  }
	}
      }
      
      if(notification)
	stream->add_notification(notification);

      log_gpudma.info("gpu memset 2d complete: dst=%p dst_off=%ld bytes=%zd lines=%zd fill_data_size=%zd",
		      dst, dst_stride, bytes, lines, fill_data_size);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // mem pair copiers for DMA channels

#ifdef OLD_COPIERS
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
	return new SpanBasedInstPairCopier<GPUtoFBMemPairCopier>(this, src_inst, 
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
	return new SpanBasedInstPairCopier<GPUfromFBMemPairCopier>(this, src_inst, 
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
	return new SpanBasedInstPairCopier<GPUinFBMemPairCopier>(this, src_inst, 
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
        return new SpanBasedInstPairCopier<GPUPeerMemPairCopier>(this, src_inst,
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
#endif

    class GPUDMAChannel_H2D : public MemPairCopierFactory {
    public:
      GPUDMAChannel_H2D(GPU *_gpu);

      virtual bool can_perform_copy(Memory src_mem, Memory dst_mem,
				    ReductionOpID redop_id, bool fold);

#ifdef OLD_COPIERS
      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold);
#endif

    protected:
      GPU *gpu;
    };

    class GPUDMAChannel_D2H : public MemPairCopierFactory {
    public:
      GPUDMAChannel_D2H(GPU *_gpu);

      virtual bool can_perform_copy(Memory src_mem, Memory dst_mem,
				    ReductionOpID redop_id, bool fold);

#ifdef OLD_COPIERS
      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold);
#endif

    protected:
      GPU *gpu;
    };

    class GPUDMAChannel_D2D : public MemPairCopierFactory {
    public:
      GPUDMAChannel_D2D(GPU *_gpu);

      virtual bool can_perform_copy(Memory src_mem, Memory dst_mem,
				    ReductionOpID redop_id, bool fold);

#ifdef OLD_COPIERS
      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold);
#endif

    protected:
      GPU *gpu;
    };

    class GPUDMAChannel_P2P : public MemPairCopierFactory {
    public:
      GPUDMAChannel_P2P(GPU *_gpu);

      virtual bool can_perform_copy(Memory src_mem, Memory dst_mem,
				    ReductionOpID redop_id, bool fold);

#ifdef OLD_COPIERS
      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold);
#endif

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

      if(!(gpu->fbmem) || (dst_mem != gpu->fbmem->me))
	return false;

      return true;
    }

#ifdef OLD_COPIERS
    MemPairCopier *GPUDMAChannel_H2D::create_copier(Memory src_mem, Memory dst_mem,
						    ReductionOpID redop_id, bool fold)
    {
      return new GPUtoFBMemPairCopier(src_mem, gpu);
    }
#endif


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

      if(!(gpu->fbmem) || (src_mem != gpu->fbmem->me))
	return false;

      if(gpu->pinned_sysmems.count(dst_mem) == 0)
	return false;

      return true;
    }

#ifdef OLD_COPIERS
    MemPairCopier *GPUDMAChannel_D2H::create_copier(Memory src_mem, Memory dst_mem,
						    ReductionOpID redop_id, bool fold)
    {
      return new GPUfromFBMemPairCopier(gpu, dst_mem);
    }
#endif


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

      if(!gpu->fbmem)
	return false;

      Memory our_fb = gpu->fbmem->me;

      if((src_mem != our_fb) || (dst_mem != our_fb))
	return false;  // they can't both be our FB

      return true;
    }

#ifdef OLD_COPIERS
    MemPairCopier *GPUDMAChannel_D2D::create_copier(Memory src_mem, Memory dst_mem,
						    ReductionOpID redop_id, bool fold)
    {
      return new GPUinFBMemPairCopier(gpu);
    }
#endif


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

      if(!(gpu->fbmem) || (src_mem != gpu->fbmem->me))
	return false;

      if(gpu->peer_fbs.count(dst_mem) == 0)
	return false;

      return true;
    }

#ifdef OLD_COPIERS
    MemPairCopier *GPUDMAChannel_P2P::create_copier(Memory src_mem, Memory dst_mem,
						    ReductionOpID redop_id, bool fold)
    {
      // TODO: remove this - the p2p copier doesn't actually need it
      MemoryImpl *dst_impl = get_runtime()->get_memory_impl(dst_mem);
      GPU *dst_gpu = ((GPUFBMemory *)dst_impl)->gpu;

      return new GPUPeerMemPairCopier(gpu, dst_gpu);
    }
#endif


    void GPU::create_dma_channels(Realm::RuntimeImpl *r)
    {
      // <NEW_DMA>
      // Not a good design choice
      // For now, channel_manager will creates all channels
      // for GPUs in dma_all_gpus
      register_gpu_in_dma_systems(this);
      // </NEW_DMA>

      // if we don't have any framebuffer memory, we can't do any DMAs
      if(!fbmem)
	return;

      if(!pinned_sysmems.empty()) {
	//r->add_dma_channel(new GPUDMAChannel_H2D(this));
	//r->add_dma_channel(new GPUDMAChannel_D2H(this));

	// TODO: move into the dma channels themselves
	for(std::set<Memory>::const_iterator it = pinned_sysmems.begin();
	    it != pinned_sysmems.end();
	    ++it) {
	  // don't create affinities for IB memories right now
	  if(!ID(*it).is_memory()) continue;

	  Machine::MemoryMemoryAffinity mma;
	  mma.m1 = fbmem->me;
	  mma.m2 = *it;
	  mma.bandwidth = 20; // "medium"
	  mma.latency = 200;  // "bad"
	  r->add_mem_mem_affinity(mma);
	}
      } else {
	log_gpu.warning() << "GPU " << proc->me << " has no pinned system memories!?";
      }

      //r->add_dma_channel(new GPUDMAChannel_D2D(this));

      // only create a p2p channel if we have peers (and an fb)
      if(!peer_fbs.empty()) {
	//r->add_dma_channel(new GPUDMAChannel_P2P(this));

	// TODO: move into the dma channels themselves
	for(std::set<Memory>::const_iterator it = peer_fbs.begin();
	    it != peer_fbs.end();
	    ++it) {
	  Machine::MemoryMemoryAffinity mma;
	  mma.m1 = fbmem->me;
	  mma.m2 = *it;
	  mma.bandwidth = 10; // assuming pcie, this should be ~half the bw and
	  mma.latency = 400;  // ~twice the latency as zcmem
	  r->add_mem_mem_affinity(mma);
	}
      }
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

    void GPUWorkFence::print(std::ostream& os) const
    {
      os << "GPUWorkFence";
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
      me->mark_finished(true /*succesful*/);
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

    // this flag will be set on the first call into any of the hijack code in
    //  cudart_hijack.cc
    //  an application is linked with -lcudart, we will NOT be hijacking the
    //  application's calls, and the cuda module needs to know that)
    /*extern*/ bool cudart_hijack_active = false;

    // used in GPUTaskScheduler<T>::execute_task below
    static bool already_issued_hijack_warning = false;

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

      // if our hijack code is not active, the application may have put some work for this
      //  task on streams we don't know about, so it takes an expensive device synchronization
      //  to guarantee that any work enqueued on a stream in the future is ordered with respect
      //  to this task's results
      if(!cudart_hijack_active) {
	// print a warning if this is the first time and it hasn't been suppressed
	if(!(gpu_proc->gpu->module->cfg_suppress_hijack_warning ||
	     already_issued_hijack_warning)) {
	  already_issued_hijack_warning = true;
	  log_gpu.warning() << "CUDART hijack code not active"
			    << " - device synchronizations required after every GPU task!";
	}
	CHECK_CU( cuCtxSynchronize() );
      }

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

    void GPU::copy_to_fb(off_t dst_offset, const void *src, size_t bytes,
			 GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy1D(this,
					(void *)(fbmem->base + dst_offset),
					src, bytes, GPU_MEMCPY_HOST_TO_DEVICE, notification);
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

    void GPU::copy_to_fb_3d(off_t dst_offset, const void *src,
                            off_t dst_stride, off_t src_stride,
                            off_t dst_height, off_t src_height,
                            size_t bytes, size_t height, size_t depth,
                            GPUCompletionNotification *notification /* = 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy3D(this,
					(void *)(fbmem->base + dst_offset),
					src, dst_stride, src_stride,
                                        dst_height, src_height,
                                        bytes, height, depth,
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

    void GPU::copy_from_fb_3d(void *dst, off_t src_offset,
                              off_t dst_stride, off_t src_stride,
                              off_t dst_height, off_t src_height,
                              size_t bytes, size_t height, size_t depth,
                              GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy3D(this, dst,
					(const void *)(fbmem->base + src_offset),
					dst_stride, src_stride,
                                        dst_height, src_height,
                                        bytes, height, depth,
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

    void GPU::copy_within_fb_3d(off_t dst_offset, off_t src_offset,
                                off_t dst_stride, off_t src_stride,
                                off_t dst_height, off_t src_height,
                                size_t bytes, size_t height, size_t depth,
                                GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy3D(this,
					(void *)(fbmem->base + dst_offset),
					(const void *)(fbmem->base + src_offset),
					dst_stride, src_stride,
                                        dst_height, src_height,
                                        bytes, height, depth,
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

    void GPU::copy_to_peer_3d(GPU *dst, off_t dst_offset, off_t src_offset,
                              off_t dst_stride, off_t src_stride,
                              off_t dst_height, off_t src_height,
                              size_t bytes, size_t height, size_t depth,
                              GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemcpy3D(this,
					(void *)(dst->fbmem->base + dst_offset),
					(const void *)(fbmem->base + src_offset),
					dst_stride, src_stride,
                                        dst_height, src_height,
                                        bytes, height, depth,
					GPU_MEMCPY_PEER_TO_PEER, notification);
      peer_to_peer_stream->add_copy(copy);
    }

    static size_t reduce_fill_size(const void *fill_data, size_t fill_data_size)
    {
      const char *as_char = reinterpret_cast<const char *>(fill_data);
      // try powers of 2 up to 128 bytes
      for(size_t step = 1; step <= 128; step <<= 1) {
	bool ok = (fill_data_size % step) == 0;  // must divide evenly
	for(size_t pos = step; ok && (pos < fill_data_size); pos += step)
	  for(size_t i = 0; i < step; i++)
	    if(as_char[pos + i] != as_char[pos + i - step]) {
	      ok = false;
	      break;
	    }
	if(ok)
	  return step;
      }
      // no attempt to optimize non-power-of-2 repeat patterns right now
      return fill_data_size;
    }

    void GPU::fill_within_fb(off_t dst_offset,
			     size_t bytes,
			     const void *fill_data, size_t fill_data_size,
			     GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemset1D(this,
					(void *)(fbmem->base + dst_offset),
					bytes,
					fill_data,
					reduce_fill_size(fill_data, fill_data_size),
					notification);
      device_to_device_stream->add_copy(copy);
    }

    void GPU::fill_within_fb_2d(off_t dst_offset, off_t dst_stride,
				size_t bytes, size_t lines,
				const void *fill_data, size_t fill_data_size,
				GPUCompletionNotification *notification /*= 0*/)
    {
      GPUMemcpy *copy = new GPUMemset2D(this,
					(void *)(fbmem->base + dst_offset),
					dst_stride,
					bytes, lines,
					fill_data,
					reduce_fill_size(fill_data, fill_data_size),
					notification);
      device_to_device_stream->add_copy(copy);
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
    // class BlockingCompletionNotification

    class BlockingCompletionNotification : public GPUCompletionNotification {
    public:
      BlockingCompletionNotification(void);
      virtual ~BlockingCompletionNotification(void);

      virtual void request_completed(void);

      virtual void wait(void);

    public:
      GASNetHSL mutex;
      GASNetCondVar cv;
      bool completed;
    };

    BlockingCompletionNotification::BlockingCompletionNotification(void)
      : cv(mutex)
      , completed(false)
    {}

    BlockingCompletionNotification::~BlockingCompletionNotification(void)
    {}

    void BlockingCompletionNotification::request_completed(void)
    {
      AutoHSLLock a(mutex);

      assert(!completed);
      completed = true;
      cv.broadcast();
    }

    void BlockingCompletionNotification::wait(void)
    {
      AutoHSLLock a(mutex);

      while(!completed)
	cv.wait();
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
      BlockingCompletionNotification bcn;
      gpu->copy_from_fb(dst, offset, size, &bcn);
      bcn.wait();
    }

    void GPUFBMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      // create an async copy and then wait for it to finish...
      BlockingCompletionNotification bcn;
      gpu->copy_to_fb(offset, src, size, &bcn);
      bcn.wait();
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
      return ID(me).memory.owner_node;
    }

    // Helper methods for emulating the cuda runtime
    /*static*/ GPUProcessor* GPUProcessor::get_current_gpu_proc(void)
    {
      return ThreadLocal::current_gpu_proc;
    }

    void GPUProcessor::push_call_configuration(dim3 grid_dim, dim3 block_dim,
                                               size_t shared_size, void *stream)
    {
      call_configs.push_back(CallConfig(grid_dim, block_dim,
                                        shared_size, (cudaStream_t)stream));
    }

    void GPUProcessor::pop_call_configuration(dim3 *grid_dim, dim3 *block_dim,
                                              size_t *shared_size, void *stream)
    {
      assert(!call_configs.empty());
      const CallConfig &config = call_configs.back();
      *grid_dim = config.grid;
      *block_dim = config.block;
      *shared_size = config.shared;
      *((cudaStream_t*)stream) = config.stream;
      call_configs.pop_back();
    }

    void GPUProcessor::stream_synchronize(cudaStream_t stream)
    {
      // same as device_synchronize for now
      device_synchronize();
    }

    GPUPreemptionWaiter::GPUPreemptionWaiter(GPU *g) : gpu(g)
    {
      GenEventImpl *impl = GenEventImpl::create_genevent();
      wait_event = impl->current_event();
    }

    void GPUPreemptionWaiter::request_completed(void)
    {
      GenEventImpl::trigger(wait_event, false/*poisoned*/);
    }

    void GPUPreemptionWaiter::preempt(void)
    {
      // Realm threads don't obey a stack discipline for
      // preemption so we can't leave our context on the stack
      gpu->pop_context();
      wait_event.wait();
      // When we wake back up, we have to push our context again
      gpu->push_context();
    }

    void GPUProcessor::device_synchronize(void)
    {
      GPUStream *current = gpu->get_current_task_stream();
      // We don't actually want to block the GPU processor
      // when synchronizing, so we instead register a cuda
      // event on the stream and then use it triggering to
      // indicate that the stream is caught up
      // Make a completion notification to be notified when
      // the event has actually triggered
      GPUPreemptionWaiter waiter(gpu);
      // Register the waiter with the stream 
      current->add_notification(&waiter); 
      // Perform the wait, this will preempt the thread
      waiter.preempt();
    }
    
    void GPUProcessor::event_create(cudaEvent_t *event, int flags)
    {
      // int cu_flags = CU_EVENT_DEFAULT;
      // if((flags & cudaEventBlockingSync) != 0)
      // 	cu_flags |= CU_EVENT_BLOCKING_SYNC;
      // if((flags & cudaEventDisableTiming) != 0)
      // 	cu_flags |= CU_EVENT_DISABLE_TIMING;

      // get an event from our event pool (ignoring the flags for now)
      CUevent e = gpu->event_pool.get_event();
      *event = e;
    }

    void GPUProcessor::event_destroy(cudaEvent_t event)
    {
      // assume the event is one of ours and put it back in the pool
      CUevent e = event;
      if(e)
	gpu->event_pool.return_event(e);
    }

    void GPUProcessor::event_record(cudaEvent_t event, cudaStream_t stream)
    {
      // ignore the provided stream and record the event on this task's assigned stream
      CUevent e = event;
      GPUStream *current = gpu->get_current_task_stream();
      CHECK_CU( cuEventRecord(e, current->get_stream()) );
    }

    void GPUProcessor::event_synchronize(cudaEvent_t event)
    {
      // TODO: consider suspending task rather than busy-waiting here...
      CUevent e = event;
      CHECK_CU( cuEventSynchronize(e) );
    }
      
    void GPUProcessor::event_elapsed_time(float *ms, cudaEvent_t start, cudaEvent_t end)
    {
      // TODO: consider suspending task rather than busy-waiting here...
      CUevent e1 = start;
      CUevent e2 = end;
      CHECK_CU( cuEventElapsedTime(ms, e1, e2) );
    }
      
    GPUProcessor::LaunchConfig::LaunchConfig(dim3 _grid, dim3 _block, size_t _shared)
      : grid(_grid), block(_block), shared(_shared)
    {}

    GPUProcessor::CallConfig::CallConfig(dim3 _grid, dim3 _block, 
                                         size_t _shared, cudaStream_t _stream)
      : LaunchConfig(_grid, _block, _shared), stream(_stream)
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

      CUstream raw_stream = gpu->get_current_task_stream()->get_stream();
      log_stream.debug() << "kernel " << func << " added to stream " << raw_stream;

      // Launch the kernel on our stream dammit!
      CHECK_CU( cuLaunchKernel(f, 
			       config.grid.x, config.grid.y, config.grid.z,
                               config.block.x, config.block.y, config.block.z,
                               config.shared,
			       raw_stream,
			       NULL, extra) );

      // pop the config we just used
      launch_configs.pop_back();

      // clear out the kernel args
      kernel_args.clear();
    }

    void GPUProcessor::launch_kernel(const void *func,
                                     dim3 grid_dim,
                                     dim3 block_dim,
                                     void **args,
                                     size_t shared_memory,
                                     cudaStream_t stream)
    {
      // Find our function
      CUfunction f = gpu->lookup_function(func);

      CUstream raw_stream = gpu->get_current_task_stream()->get_stream();
      log_stream.debug() << "kernel " << func << " added to stream " << raw_stream;

      // Launch the kernel on our stream dammit!
      CHECK_CU( cuLaunchKernel(f,
                               grid_dim.x, grid_dim.y, grid_dim.z,
                               block_dim.x, block_dim.y, block_dim.z,
                               shared_memory,
                               raw_stream,
                               args, NULL) );
    }

    void GPUProcessor::gpu_memcpy(void *dst, const void *src, size_t size,
				  cudaMemcpyKind kind)
    {
      CUstream current = gpu->get_current_task_stream()->get_stream();
      // the synchronous copy still uses cuMemcpyAsync so that we can limit the
      //  synchronization to just the right stream
      CHECK_CU( cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, size, current) );
      stream_synchronize(current);
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
      stream_synchronize(current);
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
      stream_synchronize(current);
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

    void GPUProcessor::gpu_memset(void *dst, int value, size_t count)
    {
      CUstream current = gpu->get_current_task_stream()->get_stream();
      CHECK_CU( cuMemsetD8Async((CUdeviceptr)dst, (unsigned char)value, 
                                  count, current) );
    }

    void GPUProcessor::gpu_memset_async(void *dst, int value, 
                                        size_t count, cudaStream_t stream)
    {
      CUstream current = gpu->get_current_task_stream()->get_stream();
      CHECK_CU( cuMemsetD8Async((CUdeviceptr)dst, (unsigned char)value,
                                  count, current) );
    }


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

      // this processor is able to access its own FB and the ZC mem (if any)
      if(fbmem) {
	Machine::ProcessorMemoryAffinity pma;
	pma.p = p;
	pma.m = fbmem->me;
	pma.bandwidth = 200;  // "big"
	pma.latency = 5;      // "ok"
	runtime->add_proc_mem_affinity(pma);
      }

      if(module->zcmem) {
	Machine::ProcessorMemoryAffinity pma;
	pma.p = p;
	pma.m = module->zcmem->me;
	pma.bandwidth = 20; // "medium"
	pma.latency = 200;  // "bad"
	runtime->add_proc_mem_affinity(pma);
      }

      // peer access
      for(std::vector<GPU *>::iterator it = module->gpus.begin();
	  it != module->gpus.end();
	  it++) {
	// ignore ourselves
	if(*it == this) continue;

	// ignore gpus that we don't expect to be able to peer with
	if(info->peers.count((*it)->info->device) == 0)
	  continue;

	// ignore gpus with no fb
	if(!((*it)->fbmem))
	  continue;

	// enable peer access
	{
	  AutoGPUContext agc(this);
	  CHECK_CU( cuCtxEnablePeerAccess((*it)->context, 0) );
	}
	log_gpu.info() << "peer access enabled from GPU " << p << " to FB " << (*it)->fbmem->me;
	peer_fbs.insert((*it)->fbmem->me);

	{
	  Machine::ProcessorMemoryAffinity pma;
	  pma.p = p;
	  pma.m = (*it)->fbmem->me;
	  pma.bandwidth = 10; // assuming pcie, this should be ~half the bw and
	  pma.latency = 400;  // ~twice the latency as zcmem
	  runtime->add_proc_mem_affinity(pma);
	}
      }
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

      log_gpu.debug() << "registering variable " << var->device_name << " (" << var->host_var << ") with GPU " << this;

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

      log_gpu.debug() << "registering function " << func->device_fun << " (" << func->host_fun << ") with GPU " << this;

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

    CudaModule::CudaModule(void)
      : Module("cuda")
      , cfg_zc_mem_size_in_mb(64)
      , cfg_zc_ib_size_in_mb(256)
      , cfg_fb_mem_size_in_mb(256)
      , cfg_num_gpus(0)
      , cfg_gpu_streams(12)
      , cfg_use_background_workers(true)
      , cfg_use_shared_worker(true)
      , cfg_pin_sysmem(true)
      , cfg_fences_use_callbacks(false)
      , cfg_suppress_hijack_warning(false)
      , shared_worker(0), zcmem_cpu_base(0)
      , zcib_cpu_base(0), zcmem(0)
    {}
      
    CudaModule::~CudaModule(void)
    {
      delete_container_contents(gpu_info);
    }

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
          CHECK_CU( cuDeviceGetAttribute(&info->compute_major,
                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, info->device) );
          CHECK_CU( cuDeviceGetAttribute(&info->compute_minor,
                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, info->device) );
	  CHECK_CU( cuDeviceTotalMem(&info->total_mem, info->device) );

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
	      if(can_access) {
		log_gpu.info() << "p2p access from device " << (*it1)->index
			       << " to device " << (*it2)->index;
		(*it1)->peers.insert((*it2)->device);
	      }
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
	  .add_option_int("-ll:ib_zsize", m->cfg_zc_ib_size_in_mb)
	  .add_option_int("-ll:gpu", m->cfg_num_gpus)
	  .add_option_int("-ll:streams", m->cfg_gpu_streams)
	  .add_option_int("-ll:gpuworker", m->cfg_use_shared_worker)
	  .add_option_int("-ll:pin", m->cfg_pin_sysmem)
	  .add_option_bool("-cuda:callbacks", m->cfg_fences_use_callbacks)
	  .add_option_bool("-cuda:nohijack", m->cfg_suppress_hijack_warning);
	
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
    }

    // create any memories provided by this module (default == do nothing)
    //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
    void CudaModule::create_memories(RuntimeImpl *runtime)
    {
      Module::create_memories(runtime);

      // each GPU needs its FB memory
      if(cfg_fb_mem_size_in_mb > 0)
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

      // allocate intermediate buffers in ZC memory for DMA engine
      if ((cfg_zc_ib_size_in_mb > 0) && !gpus.empty()) {
        CUdeviceptr zcib_gpu_base;
        {
          AutoGPUContext agc(gpus[0]);
          CHECK_CU( cuMemHostAlloc(&zcib_cpu_base,
                                   cfg_zc_ib_size_in_mb << 20,
                                   CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP) );
          CHECK_CU( cuMemHostGetDevicePointer(&zcib_gpu_base,
                                              zcib_cpu_base, 0) );
          // right now there are asssumptions in several places that unified addressing keeps
          //  the CPU and GPU addresses the same
          assert(zcib_cpu_base == (void *)zcib_gpu_base); 
        }
        Memory m = runtime->next_local_ib_memory_id();
        GPUZCMemory* ib_mem;
        ib_mem = new GPUZCMemory(m, zcib_gpu_base, zcib_cpu_base,
                                 cfg_zc_ib_size_in_mb << 20);
        runtime->add_ib_memory(ib_mem);
        // add the ZC memory as a pinned memory to all GPUs
        for (unsigned i = 0; i < gpus.size(); i++) {
          CUdeviceptr gpuptr;
          CUresult ret;
          {
            AutoGPUContext agc(gpus[i]);
            ret = cuMemHostGetDevicePointer(&gpuptr, zcib_cpu_base, 0);
          }
          if ((ret == CUDA_SUCCESS) && (gpuptr == zcib_gpu_base)) {
            gpus[i]->pinned_sysmems.insert(ib_mem->me);
          } else {
            log_gpu.warning() << "GPU #" << i << "has an unexpected mapping for"
            << " intermediate buffers in ZC memory!";
          }
        }
      }
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
    }

    // create any DMA channels provided by the module (default == do nothing)
    void CudaModule::create_dma_channels(RuntimeImpl *runtime)
    {
      // before we create dma channels, see how many of the system memory ranges
      //  we can register with CUDA
      if(cfg_pin_sysmem && !gpus.empty()) {
	std::vector<MemoryImpl *>& local_mems = runtime->nodes[my_node_id].memories;
	// <NEW_DMA> also add intermediate buffers into local_mems
	std::vector<MemoryImpl *>& local_ib_mems = runtime->nodes[my_node_id].ib_memories;
	std::vector<MemoryImpl *> all_local_mems;
	all_local_mems.insert(all_local_mems.end(), local_mems.begin(), local_mems.end());
	all_local_mems.insert(all_local_mems.end(), local_ib_mems.begin(), local_ib_mems.end());
	// </NEW_DMA>
	for(std::vector<MemoryImpl *>::iterator it = all_local_mems.begin();
	    it != all_local_mems.end();
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
	  registered_host_ptrs.push_back(base);

	  // now go through each GPU and verify that it got a GPU pointer (it may not match the CPU
	  //  pointer, but that's ok because we'll never refer to it directly)
	  for(unsigned i = 0; i < gpus.size(); i++) {
	    CUdeviceptr gpuptr;
	    CUresult ret;
	    {
	      AutoGPUContext agc(gpus[i]);
	      ret = cuMemHostGetDevicePointer(&gpuptr, base, 0);
	    }
	    if(ret == CUDA_SUCCESS) {
	      // no test for && ((void *)gpuptr == base)) {
	      log_gpu.info() << "memory " << (*it)->me << " successfully registered with GPU " << gpus[i]->proc->me;
	      gpus[i]->pinned_sysmems.insert((*it)->me);
	    } else {
	      log_gpu.warning() << "GPU #" << i << " has no mapping for registered memory (" << (*it)->me << " at " << base << ") !?";
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

      if(zcib_cpu_base) {
	assert(!gpus.empty());
	AutoGPUContext agc(gpus[0]);
	CHECK_CU( cuMemFreeHost(zcib_cpu_base) );
      }

      // also unregister any host memory at this time
      if(!registered_host_ptrs.empty()) {
	AutoGPUContext agc(gpus[0]);
	for(std::vector<void *>::const_iterator it = registered_host_ptrs.begin();
	    it != registered_host_ptrs.end();
	    ++it)
	  CHECK_CU( cuMemHostUnregister(*it) );
	registered_host_ptrs.clear();
      }

      for(std::vector<GPU *>::iterator it = gpus.begin();
	  it != gpus.end();
	  it++)
	delete *it;
      gpus.clear();
      
      Module::cleanup();
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // struct RegisteredFunction

    RegisteredFunction::RegisteredFunction(const FatBin *_fat_bin, const void *_host_fun,
					   const char *_device_fun)
      : fat_bin(_fat_bin), host_fun(_host_fun), device_fun(_device_fun)
    {}
     
    ////////////////////////////////////////////////////////////////////////
    //
    // struct RegisteredVariable

    RegisteredVariable::RegisteredVariable(const FatBin *_fat_bin, const void *_host_var,
					   const char *_device_name, bool _external,
					   int _size, bool _constant, bool _global)
      : fat_bin(_fat_bin), host_var(_host_var), device_name(_device_name),
	external(_external), size(_size), constant(_constant), global(_global)
    {}


    ////////////////////////////////////////////////////////////////////////
    //
    // class GlobalRegistrations

    GlobalRegistrations::GlobalRegistrations(void)
    {}

    GlobalRegistrations::~GlobalRegistrations(void)
    {
      delete_container_contents(variables);
      delete_container_contents(functions);
      // we don't own fat binary pointers, but we can forget them
      fat_binaries.clear();
    }

    /*static*/ GlobalRegistrations& GlobalRegistrations::get_global_registrations(void)
    {
      static GlobalRegistrations reg;
      return reg;
    }

    // called by a GPU when it has created its context - will result in calls back
    //  into the GPU for any modules/variables/whatever already registered
    /*static*/ void GlobalRegistrations::add_gpu_context(GPU *gpu)
    {
      GlobalRegistrations& g = get_global_registrations();

      AutoHSLLock al(g.mutex);

      // add this gpu to the list
      assert(g.active_gpus.count(gpu) == 0);
      g.active_gpus.insert(gpu);

      // and now tell it about all the previous-registered stuff
      for(std::vector<FatBin *>::iterator it = g.fat_binaries.begin();
	  it != g.fat_binaries.end();
	  it++)
	gpu->register_fat_binary(*it);

      for(std::vector<RegisteredVariable *>::iterator it = g.variables.begin();
	  it != g.variables.end();
	  it++)
	gpu->register_variable(*it);

      for(std::vector<RegisteredFunction *>::iterator it = g.functions.begin();
	  it != g.functions.end();
	  it++)
	gpu->register_function(*it);
    }

    /*static*/ void GlobalRegistrations::remove_gpu_context(GPU *gpu)
    {
      GlobalRegistrations& g = get_global_registrations();

      AutoHSLLock al(g.mutex);

      assert(g.active_gpus.count(gpu) > 0);
      g.active_gpus.erase(gpu);
    }

    // called by __cuda(un)RegisterFatBinary
    /*static*/ void GlobalRegistrations::register_fat_binary(FatBin *fatbin)
    {
      GlobalRegistrations& g = get_global_registrations();

      AutoHSLLock al(g.mutex);

      // add the fat binary to the list and tell any gpus we know of about it
      g.fat_binaries.push_back(fatbin);

      for(std::set<GPU *>::iterator it = g.active_gpus.begin();
	  it != g.active_gpus.end();
	  it++)
	(*it)->register_fat_binary(fatbin);
    }

    /*static*/ void GlobalRegistrations::unregister_fat_binary(FatBin *fatbin)
    {
      GlobalRegistrations& g = get_global_registrations();

      AutoHSLLock al(g.mutex);

      // remove the fatbin from the list - don't bother telling gpus
      std::vector<FatBin *>::iterator it = g.fat_binaries.begin();
      while(it != g.fat_binaries.end())
	if(*it == fatbin)
	  it = g.fat_binaries.erase(it);
	else
	  it++;
    }

    // called by __cudaRegisterVar
    /*static*/ void GlobalRegistrations::register_variable(RegisteredVariable *var)
    {
      GlobalRegistrations& g = get_global_registrations();

      AutoHSLLock al(g.mutex);

      // add the variable to the list and tell any gpus we know
      g.variables.push_back(var);

      for(std::set<GPU *>::iterator it = g.active_gpus.begin();
	  it != g.active_gpus.end();
	  it++)
	(*it)->register_variable(var);
    }

    // called by __cudaRegisterFunction
    /*static*/ void GlobalRegistrations::register_function(RegisteredFunction *func)
    {
      GlobalRegistrations& g = get_global_registrations();

      AutoHSLLock al(g.mutex);

      // add the function to the list and tell any gpus we know
      g.functions.push_back(func);

      for(std::set<GPU *>::iterator it = g.active_gpus.begin();
	  it != g.active_gpus.end();
	  it++)
	(*it)->register_function(func);
    }


  }; // namespace Cuda
}; // namespace Realm

