/* Copyright 2022 Stanford University, NVIDIA Corporation
 *                Los Alamos National Laboratory
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

#include "realm/hip/hip_module.h"
#include "realm/hip/hip_internal.h"
#include "realm/hip/hip_access.h"

#include "realm/tasks.h"
#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/event_impl.h"
#include "realm/idx_impl.h"

#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/channel.h"
#include "realm/transfer/ib_memory.h"

#ifdef REALM_USE_HIP_HIJACK
#include "realm/hip/hip_hijack.h"
#endif

#include "realm/mutex.h"
#include "realm/utils.h"

#ifdef REALM_USE_VALGRIND_ANNOTATIONS
#include <valgrind/memcheck.h>
#endif

#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

#define IS_DEFAULT_STREAM(stream)   \
  ((stream) == 0)

namespace Realm {
  namespace Hip {

    Logger log_gpu("hip");
    Logger log_gpudma("hipdma");
    Logger log_cudart("cudart");
    Logger log_hipipc("hipipc");

    Logger log_stream("hipstream");  

  ////////////////////////////////////////////////////////////////////////
  //
  // class GPUStream

    GPUStream::GPUStream(GPU *_gpu, GPUWorker *_worker,
		         int rel_priority /*= 0*/)
      : gpu(_gpu), worker(_worker), issuing_copies(false)
    {
      assert(worker != 0);
      
      // the math here is designed to balance the context's priority range
      //  around a relative priority of 0, favoring an extra negative (higher
      //  priority) option
      int abs_priority = (gpu->greatest_stream_priority +
                          rel_priority +
                          ((gpu->least_stream_priority -
                            gpu->greatest_stream_priority + 1) / 2));
      // CUDA promises to clamp to the actual range, so we don't have to
      CHECK_HIP( hipStreamCreateWithPriority(&stream, hipStreamNonBlocking,
                                           abs_priority) );
      log_stream.info() << "stream created: gpu=" << gpu
                        << " stream=" << stream << " priority=" << abs_priority;
    }

    GPUStream::~GPUStream(void)
    {
      // log_stream.info() << "HIP stream " << stream << " destroyed - max copies = " 
      // 			<< pending_copies.capacity() << ", max events = " << pending_events.capacity();

      CHECK_HIP( hipStreamDestroy(stream) );
    }

    GPU *GPUStream::get_gpu(void) const
    {
      return gpu;
    }
    
    hipStream_t GPUStream::get_stream(void) const
    {
      return stream;
    }

    // may be called by anybody to enqueue a copy or an event
    void GPUStream::add_copy(GPUMemcpy *copy)
    {
      assert(0 && "hit old copy path"); // shouldn't be used any more
      bool add_to_worker = false;
      {
      	AutoLock<> al(mutex);

      	// if we didn't already have work AND if there's not an active
      	//  worker issuing copies, request attention
      	add_to_worker = (pending_copies.empty() &&
                         pending_events.empty() &&
                         !issuing_copies);

      	pending_copies.push_back(copy);
      }

      if(add_to_worker)
        worker->add_stream(this);
    }

    void GPUStream::add_fence(GPUWorkFence *fence)
    {
      hipEvent_t e = gpu->event_pool.get_event();

      CHECK_HIP( hipEventRecord(e, stream) );

      log_stream.debug() << "HIP fence event " << e << " recorded on stream " << stream << " (GPU " << gpu << ")";

      add_event(e, fence, 0);
    }

    void GPUStream::add_start_event(GPUWorkStart *start)
    {
      hipEvent_t e = gpu->event_pool.get_event();

      CHECK_HIP( hipEventRecord(e, stream) );

      log_stream.debug() << "HIP start event " << e << " recorded on stream " << stream << " (GPU " << gpu << ")";

      // record this as a start event
      add_event(e, 0, 0, start);
    }

    void GPUStream::add_notification(GPUCompletionNotification *notification)
    {
      hipEvent_t e = gpu->event_pool.get_event();

      CHECK_HIP( hipEventRecord(e, stream) );

      add_event(e, 0, notification);
    }

    void GPUStream::add_event(hipEvent_t event, GPUWorkFence *fence, 
			      GPUCompletionNotification *notification, GPUWorkStart *start)
    {
      bool add_to_worker = false;
      {
	      AutoLock<> al(mutex);

      	// if we didn't already have work AND if there's not an active
      	//  worker issuing copies, request attention
      	add_to_worker = (pending_copies.empty() &&
      			             pending_events.empty() &&
      			             !issuing_copies);


	      PendingEvent e;
	      e.event = event;
	      e.fence = fence;
	      e.start = start;
	      e.notification = notification;

	      pending_events.push_back(e);
      }

      if(add_to_worker)
	      worker->add_stream(this);
    }
    
    void GPUStream::wait_on_streams(const std::set<GPUStream*> &other_streams)
    {
      assert(!other_streams.empty());
      for (std::set<GPUStream*>::const_iterator it = 
            other_streams.begin(); it != other_streams.end(); it++)
      {
        if (*it == this)
          continue;
        hipEvent_t e = gpu->event_pool.get_event();

        CHECK_HIP( hipEventRecord(e, (*it)->get_stream()) );

        log_stream.debug() << "HIP stream " << stream << " waiting on stream " 
                           << (*it)->get_stream() << " (GPU " << gpu << ")";

        CHECK_HIP( hipStreamWaitEvent(stream, e, 0) );

        // record this event on our stream
        add_event(e, 0);
      }
    }
    
    bool GPUStream::has_work(void) const
    {
      return(!pending_events.empty() || !pending_copies.empty());
    }
    
    // atomically checks rate limit counters and returns true if 'bytes'
    //  worth of copies can be submitted or false if not (in which case
    //  the progress counter on the xd will be updated when it should try
    //  again)
    bool GPUStream::ok_to_submit_copy(size_t bytes, XferDes *xd)
    {
      return true;
    }

    // to be called by a worker (that should already have the GPU context
    //   current) - returns true if any work remains
    bool GPUStream::issue_copies(TimeLimit work_until)
    {
      // we have to make sure copies for a given stream are issued
      //  in order, so grab the thing at the front of the queue, but
      //  also set a flag taking ownership of the head of the queue
      GPUMemcpy *copy = 0;
      {
        AutoLock<> al(mutex);
        
      	// if the flag is set, we can't do any copies
      	if(issuing_copies || pending_copies.empty()) {
      	  // no copies left, but stream might have other work left
      	  return has_work();
	      }
        
      	copy = pending_copies.front();
      	pending_copies.pop_front();
      	issuing_copies = true;
      }

      while(true) {
	      {
	        AutoGPUContext agc(gpu);
	        copy->execute(this);
	      }

	      // TODO: recycle these
	      delete copy;

        // don't take another copy (but do clear the ownership flag)
      	//  if we're out of time
      	bool expired = work_until.is_expired();

      	{
      	  AutoLock<> al(mutex);

      	  if(pending_copies.empty()) {
      	    issuing_copies = false;
      	    // no copies left, but stream might have other work left
      	    return has_work();
      	  } else {
      	    if(expired) {
      	      issuing_copies = false;
      	      // definitely still work to do
      	      return true;
      	    } else {
      	      // take the next copy
      	      copy = pending_copies.front();
      	      pending_copies.pop_front();
      	    }
      	  }
      	}
      }
    }

    bool GPUStream::reap_events(TimeLimit work_until)
    {
      // peek at the first event
      hipEvent_t event;
      bool event_valid = false;
      {
        AutoLock<> al(mutex);

        if(pending_events.empty())
        // no events left, but stream might have other work left
	      return has_work();

        event = pending_events.front().event;
        event_valid = true;
      }

      // we'll keep looking at events until we find one that hasn't triggered
      bool work_left = true;
      while(event_valid) {
        hipError_t res = hipEventQuery(event);

        if(res == hipErrorNotReady)
          return true; // oldest event hasn't triggered - check again later

        // no other kind of error is expected
        if(res != hipSuccess) {
          const char *ename = 0;
          const char *estr = 0;
          ename = hipGetErrorName(res);
          estr = hipGetErrorString(res);
          log_gpu.fatal() << "HIP error reported on GPU " << gpu->info->index << ": " << estr << " (" << ename << ")";
          assert(0);
        }

        log_stream.debug() << "HIP event " << event << " triggered on stream " << stream << " (GPU " << gpu << ")";

        // give event back to GPU for reuse
        gpu->event_pool.return_event(event);

        // this event has triggered, so figure out the fence/notification to trigger
        //  and also peek at the next event
        GPUWorkFence *fence = 0;
              GPUWorkStart *start = 0;
        GPUCompletionNotification *notification = 0;

        {
          AutoLock<> al(mutex);

          const PendingEvent &e = pending_events.front();
          assert(e.event == event);
          fence = e.fence;
                start = e.start;
          notification = e.notification;
          pending_events.pop_front();

          if(pending_events.empty()) {
            event_valid = false;
            work_left = has_work();
          } else
            event = pending_events.front().event;
        }

        if (start) {
          start->mark_gpu_work_start();
        }
        if(fence)
          fence->mark_finished(true /*successful*/);

        if(notification)
          notification->request_completed();
        
        // don't repeat if we're out of time
      	if(event_valid && work_until.is_expired())
      	  return true;
      }

      // if we get here, we ran out of events, but there might have been
      //  other kinds of work that we need to let the caller know about
      return work_left;
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

      hipStream_t raw_stream = local_stream->get_stream();
      log_stream.debug() << "memcpy added to stream " << raw_stream;
#if 0
      switch (kind)
      {
        case GPU_MEMCPY_HOST_TO_DEVICE:
          {
            CHECK_HIP( hipMemcpyHtoDAsync((hipDeviceptr_t)(((char*)dst)+span_start),
                                        (((char*)src)+span_start),
                                        span_bytes,
                                        raw_stream) );
            break;
          }
        case GPU_MEMCPY_DEVICE_TO_HOST:
          {
            CHECK_HIP( hipMemcpyDtoHAsync((((char*)dst)+span_start),
                                        (hipDeviceptr_t)(((char*)src)+span_start),
                                        span_bytes,
                                        raw_stream) );
#ifdef REALM_USE_VALGRIND_ANNOTATIONS
            VALGRIND_MAKE_MEM_DEFINED((((char*)dst)+span_start), span_bytes);
#endif
            break;
          }
        case GPU_MEMCPY_DEVICE_TO_DEVICE:
        case GPU_MEMCPY_PEER_TO_PEER:
          {
            CHECK_HIP( hipMemcpyDtoDAsync((hipDeviceptr_t)(((char*)dst)+span_start),
                                        (hipDeviceptr_t)(((char*)src)+span_start),
                                        span_bytes,
                                        raw_stream) );
            break;
          }
        default:
          assert(false);
      }
#else
      hipMemcpyKind copy_type;
      switch (kind)
      {
        case GPU_MEMCPY_HOST_TO_DEVICE:
          {
            copy_type = hipMemcpyHostToDevice;
            break;
          }
        case GPU_MEMCPY_DEVICE_TO_HOST:
          {
            copy_type = hipMemcpyDeviceToHost;
#ifdef REALM_USE_VALGRIND_ANNOTATIONS
            VALGRIND_MAKE_MEM_DEFINED((((char*)dst)+span_start), span_bytes);
#endif
            break;
          }
        case GPU_MEMCPY_DEVICE_TO_DEVICE:
          {
            copy_type = hipMemcpyDeviceToDevice;
            break;
          }
        case GPU_MEMCPY_PEER_TO_PEER:
          {
            copy_type = hipMemcpyDefault;
            break;
          }
        default:
          assert(false);
      }
      CHECK_HIP( hipMemcpyAsync((void *)(((char*)dst)+span_start),
                               (const void*)(((char*)src)+span_start),
                               span_bytes, copy_type,
                               raw_stream) );
#endif
    }

    void GPUMemcpy1D::execute(GPUStream *stream)
    {
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
	      dst_stride(_dst_stride), 
	      src_stride(_src_stride),
	      bytes(_bytes), lines(_lines), notification(_notification)
    {}

    GPUMemcpy2D::~GPUMemcpy2D(void)
    {}

    void GPUMemcpy2D::execute(GPUStream *stream)
    {
      log_gpudma.info("gpu memcpy 2d: dst=%p src=%p "
                   "dst_off=%ld src_off=%ld bytes=%ld lines=%ld kind=%d",
                      dst, src, (long)dst_stride, (long)src_stride, (long)bytes, (long)lines, kind); 
#if 0      
      hip_Memcpy2D copy_info;

      // peer memory counts as DEVICE here
#ifdef __HIP_PLATFORM_NVIDIA__
      copy_info.srcMemoryType = (kind == GPU_MEMCPY_HOST_TO_DEVICE) ?
        CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE;
      copy_info.dstMemoryType = (kind == GPU_MEMCPY_DEVICE_TO_HOST) ?
      	CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE;
#else
      copy_info.srcMemoryType = (kind == GPU_MEMCPY_HOST_TO_DEVICE) ?
        hipMemoryTypeHost : hipMemoryTypeDevice;
      copy_info.dstMemoryType = (kind == GPU_MEMCPY_DEVICE_TO_HOST) ?
      	hipMemoryTypeHost : hipMemoryTypeDevice;
#endif
      copy_info.srcDevice = (hipDeviceptr_t)src;
      copy_info.srcHost = src;
      copy_info.srcPitch = src_stride;
      copy_info.srcY = 0;
      copy_info.srcXInBytes = 0;
      copy_info.dstDevice = (hipDeviceptr_t)dst;
      copy_info.dstHost = dst;
      copy_info.dstPitch = dst_stride;
      copy_info.dstY = 0;
      copy_info.dstXInBytes = 0;
      copy_info.WidthInBytes = bytes;
      copy_info.Height = lines;
      CHECK_HIP( hipMemcpyParam2DAsync(&copy_info, stream->get_stream()) );
#else
      hipMemcpyKind copy_type;
      if (kind == GPU_MEMCPY_PEER_TO_PEER) {
        // If we're doing peer to peer, just let unified memory it deal with it
        copy_type = hipMemcpyDefault;
      } else if (kind == GPU_MEMCPY_HOST_TO_DEVICE) {
        copy_type = hipMemcpyHostToDevice;
      } else if (kind == GPU_MEMCPY_DEVICE_TO_HOST) {
        copy_type = hipMemcpyDeviceToHost;
      } else if (kind == GPU_MEMCPY_DEVICE_TO_DEVICE) {
        copy_type = hipMemcpyDeviceToDevice;
      } else {
       assert(0);
      }

      CHECK_HIP( hipMemcpy2DAsync(dst, dst_stride, src, src_stride, bytes, lines, copy_type, stream->get_stream()) );
      
#endif

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
                             off_t _dst_pstride, off_t _src_pstride,
                             size_t _bytes, size_t _height, size_t _depth,
                             GPUMemcpyKind _kind,
                             GPUCompletionNotification *_notification)
      : GPUMemcpy(_gpu, _kind), dst(_dst), src(_src),
	      dst_stride(_dst_stride), 
	      src_stride(_src_stride),
        dst_pstride(_dst_pstride),
        src_pstride(_src_pstride),
	      bytes(_bytes), height(_height), depth(_depth),
        notification(_notification)
    {}

    GPUMemcpy3D::~GPUMemcpy3D(void)
    {}
    
    void GPUMemcpy3D::execute(GPUStream *stream)
    {
      log_gpudma.info("gpu memcpy 3d: dst=%p src=%p "
                      "dst_str=%ld src_str=%ld dst_pstr=%ld src_pstr=%ld "
                      "bytes=%ld height=%ld depth=%ld kind=%d",
                      dst, src, (long)dst_stride, (long)src_stride,
                      (long)dst_pstride, (long)src_pstride,
		                  (long)bytes, (long)height, (long)depth, kind);

      // cuMemcpy3D requires that the src/dst plane strides must be multiples
      //  of the src/dst line strides - if that doesn't hold (e.g. transpose
      //  copies), we fall back to a bunch of 2d copies for now, but should
      //  consider specialized kernels in the future

      if(((src_pstride % src_stride) == 0) && ((dst_pstride % dst_stride) == 0)) {
#ifdef __HIP_PLATFORM_NVIDIA__
        cudaMemcpyKind copy_type;
        if (kind == GPU_MEMCPY_PEER_TO_PEER) {
          // If we're doing peer to peer, just let unified memory it deal with it
          copy_type = cudaMemcpyDefault;
        } else if (kind == GPU_MEMCPY_HOST_TO_DEVICE) {
          copy_type = cudaMemcpyHostToDevice;
        } else if (kind == GPU_MEMCPY_DEVICE_TO_HOST) {
          copy_type = cudaMemcpyDeviceToHost;
        } else if (kind == GPU_MEMCPY_DEVICE_TO_DEVICE) {
          copy_type = cudaMemcpyDeviceToDevice;
        } else {
          assert(0);
        }
#else
        hipMemcpyKind copy_type;
        if (kind == GPU_MEMCPY_PEER_TO_PEER) {
          // If we're doing peer to peer, just let unified memory it deal with it
          copy_type = hipMemcpyDefault;
        } else if (kind == GPU_MEMCPY_HOST_TO_DEVICE) {
          copy_type = hipMemcpyHostToDevice;
        } else if (kind == GPU_MEMCPY_DEVICE_TO_HOST) {
          copy_type = hipMemcpyDeviceToHost;
        } else if (kind == GPU_MEMCPY_DEVICE_TO_DEVICE) {
          copy_type = hipMemcpyDeviceToDevice;
        } else {
          assert(0);
        }
#endif
        hipMemcpy3DParms copy_info = {0};
        copy_info.srcPtr = make_hipPitchedPtr((void*)src, src_stride, bytes, src_pstride/src_stride);
        copy_info.dstPtr = make_hipPitchedPtr(dst, dst_stride, bytes, dst_pstride/dst_stride);
        copy_info.srcPos = make_hipPos(0,0,0);
        copy_info.dstPos = make_hipPos(0,0,0);
        copy_info.extent = make_hipExtent(bytes, height, depth);
        copy_info.kind = copy_type;
        CHECK_HIP( hipMemcpy3DAsync(&copy_info, stream->get_stream()) );
      } else {
      	// we can unroll either lines (height) or planes (depth) - choose the
      	//  smaller of the two to minimize API calls
      	size_t count, lines_2d;
      	off_t src_pitch, dst_pitch, src_delta, dst_delta;
      	if(height <= depth) {
      	  // 2d copies use depth
      	  lines_2d = depth;
      	  src_pitch = src_pstride;
      	  dst_pitch = dst_pstride;
      	  // and we'll step in height between those copies
      	  count = height;
      	  src_delta = src_stride;
      	  dst_delta = dst_stride;
      	} else {
      	  // 2d copies use height
      	  lines_2d = height;
      	  src_pitch = src_stride;
      	  dst_pitch = dst_stride;
      	  // and we'll step in depth between those copies
      	  count = depth;
      	  src_delta = src_pstride;
      	  dst_delta = dst_pstride;
      	}

        unsigned char *src_ptr = (unsigned char*)src;
        unsigned char *dst_ptr = (unsigned char*)dst;
        hipMemcpyKind copy_type;
        if (kind == GPU_MEMCPY_PEER_TO_PEER) {
          // If we're doing peer to peer, just let unified memory it deal with it
          copy_type = hipMemcpyDefault;
        } else if (kind == GPU_MEMCPY_HOST_TO_DEVICE) {
          copy_type = hipMemcpyHostToDevice;
        } else if (kind == GPU_MEMCPY_DEVICE_TO_HOST) {
          copy_type = hipMemcpyDeviceToHost;
        } else {
          assert(0);
        }

      	for(size_t i = 0; i < count; i++) {
      	  CHECK_HIP( hipMemcpy2DAsync((void*)dst_ptr, dst_pitch, (void*)src_ptr, src_pitch, bytes, lines_2d, copy_type, stream->get_stream()) );
      	  src_ptr += src_delta;
          dst_ptr += dst_delta;
      	}
      }

      if(notification)
        stream->add_notification(notification);

      log_gpudma.info("gpu memcpy 3d complete: dst=%p src=%p "
                      "dst_str=%ld src_str=%ld dst_pstr=%ld src_pstr=%ld "
                      "bytes=%ld height=%ld depth=%ld kind=%d",
                      dst, src, (long)dst_stride, (long)src_stride,
                      (long)dst_pstride, (long)src_pstride,
		                  (long)bytes, (long)height, (long)depth, kind);
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
      log_gpudma.info("gpu memset: dst=%p bytes=%zd fill_data_size=%zd",
                      dst, bytes, fill_data_size);

      hipStream_t raw_stream = stream->get_stream();

      switch(fill_data_size) {
      case 1:
        {
          unsigned char fill_u8;
          memcpy(&fill_u8, fill_data.direct, 1);
          CHECK_HIP( hipMemsetD8Async(hipDeviceptr_t(dst), 
                                     fill_u8, bytes,
                                     raw_stream) );
          break;
        }
      case 2:
        {
          unsigned short fill_u16;
          memcpy(&fill_u16, fill_data.direct, 2);
          CHECK_HIP( hipMemsetD16Async(hipDeviceptr_t(dst), 
                                      fill_u16, bytes >> 1,
                                      raw_stream) );
          break;
        }
      case 4:
        {
          unsigned int fill_u32;
          memcpy(&fill_u32, fill_data.direct, 4);
          CHECK_HIP( hipMemsetD32Async(hipDeviceptr_t(dst), 
                                      fill_u32, bytes >> 2,
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
#if 0
          if((fill_data_size & 3) == 0) {
            for(size_t offset = 0; offset < fill_data_size; offset += 4) {
              unsigned int val = *reinterpret_cast<const unsigned int *>(srcdata + offset);
              CHECK_HIP( cuMemsetD2D32Async(CUdeviceptr(dst) + offset,
                  fill_data_size /*pitch*/,
                  val,
                  1 /*width*/, elements /*height*/,
                  raw_stream) );
            }
          } else if((fill_data_size & 1) == 0) {
            for(size_t offset = 0; offset < fill_data_size; offset += 2) {
              unsigned short val = *reinterpret_cast<const unsigned short *>(srcdata + offset);
              CHECK_HIP( cuMemsetD2D16Async(CUdeviceptr(dst) + offset,
                  fill_data_size /*pitch*/,
                  val,
                  1 /*width*/, elements /*height*/,
                  raw_stream) );
            }
          } else 
#endif    
          {
            for(size_t offset = 0; offset < fill_data_size; offset += 1) {
              unsigned char fill_u8;
              memcpy(&fill_u8, srcdata + offset, 1);
              CHECK_HIP( hipMemset2DAsync((void *)(static_cast<char*>(dst) + offset),
                                         fill_data_size /*pitch*/,
                                         fill_u8,
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
      log_gpudma.info("gpu memset 2d: dst=%p dst_str=%ld bytes=%zd lines=%zd fill_data_size=%zd",
                      dst, dst_stride, bytes, lines, fill_data_size);

      hipStream_t raw_stream = stream->get_stream();

      switch(fill_data_size) {
      case 1:
      case 2:
      case 4:
        {
          unsigned char fill_u8;
          memcpy(&fill_u8, fill_data.direct, 1);
          CHECK_HIP( hipMemset2DAsync((void *)(dst), dst_stride,
                                     fill_u8, bytes, lines,
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
#if 0
          if((fill_data_size & 3) == 0) {
            for(size_t offset = 0; offset < fill_data_size; offset += 4) {
              unsigned int val = *reinterpret_cast<const unsigned int *>(srcdata + offset);
              for(size_t l = 0; l < lines; l++)
                CHECK_HIP( cuMemsetD2D32Async(CUdeviceptr(dst) + offset + (l * dst_stride),
                                              fill_data_size /*pitch*/,
                                              val,
                                              1 /*width*/, elements /*height*/,
                                              raw_stream) );
            }
          } else if((fill_data_size & 1) == 0) {
            for(size_t offset = 0; offset < fill_data_size; offset += 2) {
              unsigned short val = *reinterpret_cast<const unsigned short *>(srcdata + offset);
              for(size_t l = 0; l < lines; l++)
                CHECK_HIP( cuMemsetD2D16Async(CUdeviceptr(dst) + offset + (l * dst_stride),
                                              fill_data_size /*pitch*/,
                                              val,
                                              1 /*width*/, elements /*height*/,
                                              raw_stream) );
            }
          } else 
#endif    
          {
            for(size_t offset = 0; offset < fill_data_size; offset += 1) {
              unsigned char fill_u8;
              memcpy(&fill_u8, srcdata + offset, 1);
              for(size_t l = 0; l < lines; l++)
                CHECK_HIP( hipMemset2DAsync((void *)(static_cast<char*>(dst) + offset + (l * dst_stride)),
                                            fill_data_size /*pitch*/,
                                            fill_u8,
                                            1 /*width*/, elements /*height*/,
                                            raw_stream) );
            }
          }
        }
      }
      
      if(notification)
	stream->add_notification(notification);

      log_gpudma.info("gpu memset 2d complete: dst=%p dst_str=%ld bytes=%zd lines=%zd fill_data_size=%zd",
                      dst, dst_stride, bytes, lines, fill_data_size);
    }
    
  ////////////////////////////////////////////////////////////////////////
  //
  // class GPUMemset3D

    GPUMemset3D::GPUMemset3D(GPU *_gpu,
			     void *_dst, size_t _dst_stride, size_t _dst_pstride,
			     size_t _bytes, size_t _height, size_t _depth,
			     const void *_fill_data, size_t _fill_data_size,
			     GPUCompletionNotification *_notification)
      : GPUMemcpy(_gpu, GPU_MEMCPY_DEVICE_TO_DEVICE)
      , dst(_dst), dst_stride(_dst_stride), dst_pstride(_dst_pstride)
      , bytes(_bytes), height(_height), depth(_depth)
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

    GPUMemset3D::~GPUMemset3D(void)
    {
      if(fill_data_size > MAX_DIRECT_SIZE)
	      delete[] fill_data.indirect;
    }

    void GPUMemset3D::execute(GPUStream *stream)
    {
      log_gpudma.info("gpu memset 3d: dst=%p dst_str=%ld dst_pstr=%ld bytes=%zd height=%zd depth=%zd fill_data_size=%zd",
                      dst, dst_stride, dst_pstride,
                      bytes, height, depth, fill_data_size);

      hipStream_t raw_stream = stream->get_stream();

      // there don't appear to be cuMemsetD3D... calls, so we'll do
      //  cuMemsetD2D...'s on the first plane and then memcpy3d to the other
      switch(fill_data_size) {
      case 1:
      case 2:
      case 4:
      	{
          unsigned char fill_u8;
          memcpy(&fill_u8, fill_data.direct, 1);
      	  CHECK_HIP( hipMemset2DAsync((void*)(dst), dst_stride,
                                      fill_u8, bytes, height,
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
#if 0
      	  // 16- and 32-bit fills must be aligned on every piece
      	  if((fill_data_size & 3) == 0) {
      	    for(size_t offset = 0; offset < fill_data_size; offset += 4) {
      	      unsigned int val = *reinterpret_cast<const unsigned int *>(srcdata + offset);
      	      for(size_t l = 0; l < height; l++)
            		CHECK_HIP( cuMemsetD2D32Async(CUdeviceptr(dst) + offset + (l * dst_stride),
            					     fill_data_size /*pitch*/,
            					     val,
            					     1 /*width*/, elements /*height*/,
            					     raw_stream) );
      	    }
      	  } else if((fill_data_size & 1) == 0) {
      	    for(size_t offset = 0; offset < fill_data_size; offset += 2) {
      	      unsigned short val = *reinterpret_cast<const unsigned short *>(srcdata + offset);
      	      for(size_t l = 0; l < height; l++)
            		CHECK_HIP( cuMemsetD2D16Async(CUdeviceptr(dst) + offset + (l * dst_stride),
            					     fill_data_size /*pitch*/,
            					     val,
            					     1 /*width*/, elements /*height*/,
            					     raw_stream) );
      	    }
      	  } else
#endif 
          {
      	    for(size_t offset = 0; offset < fill_data_size; offset += 1) {
      	      unsigned char fill_u8;
              memcpy(&fill_u8, srcdata + offset, 1);
      	      for(size_t l = 0; l < height; l++)
            		CHECK_HIP( hipMemset2DAsync((void*)(static_cast<char*>(dst) + offset + (l * dst_stride)),
                                           fill_data_size /*pitch*/,
                                           fill_u8,
                                           1 /*width*/, elements /*height*/,
                                           raw_stream) );
      	    }
      	  }
      	}
      }

      if(depth > 1) {
        assert((dst_pstride % dst_stride) == 0);
        hipMemcpy3DParms copy_info = {0};
        copy_info.srcPtr = make_hipPitchedPtr((void*)dst, dst_stride, bytes, dst_pstride / dst_stride);
        copy_info.srcPos = make_hipPos(0,0,0);
        copy_info.dstPos = make_hipPos(0,0,0);
#ifdef __HIP_PLATFORM_NVIDIA__
        copy_info.kind = cudaMemcpyDeviceToDevice;
#else
        copy_info.kind = hipMemcpyDeviceToDevice;
#endif
        // can't use a srcHeight of 0 to reuse planes, so fill N-1 remaining
        //  planes in log(N) copies
        for(size_t done = 1; done < depth; done <<= 1) {
          size_t todo = std::min(done, depth - done);
          unsigned char *dstDevice = (unsigned char*)dst + (done * dst_pstride);
          copy_info.dstPtr = make_hipPitchedPtr((void*)dstDevice, dst_stride, bytes, dst_pstride/dst_stride);
          copy_info.extent = make_hipExtent(bytes, height, todo);
          CHECK_HIP( hipMemcpy3DAsync(&copy_info, raw_stream) );
        }
      }

      if(notification)
      	stream->add_notification(notification);

      log_gpudma.info("gpu memset 3d complete: dst=%p dst_str=%ld dst_pstr=%ld bytes=%zd height=%zd depth=%zd fill_data_size=%zd",
                      dst, dst_stride, dst_pstride,
                      bytes, height, depth, fill_data_size);
    }

    void GPU::create_dma_channels(Realm::RuntimeImpl *r)
    {
      // if we don't have any framebuffer memory, we can't do any DMAs
      if(!fbmem)
	      return;
      
      r->add_dma_channel(new GPUChannel(this, XFER_GPU_IN_FB, &r->bgwork));
      r->add_dma_channel(new GPUfillChannel(this, &r->bgwork));
      r->add_dma_channel(new GPUreduceChannel(this, &r->bgwork));

      if(!pinned_sysmems.empty()) {
        r->add_dma_channel(new GPUChannel(this, XFER_GPU_TO_FB, &r->bgwork));
        r->add_dma_channel(new GPUChannel(this, XFER_GPU_FROM_FB, &r->bgwork));

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
        log_gpu.warning() << "GPU " << proc->me << " has no accessible system memories!?";
      }

      // only create a p2p channel if we have peers (and an fb)
      if(!peer_fbs.empty() || !hipipc_mappings.empty()) {
        r->add_dma_channel(new GPUChannel(this, XFER_GPU_PEER_FB, &r->bgwork));

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

        for(std::vector<HipIpcMapping>::const_iterator it = hipipc_mappings.begin();
            it != hipipc_mappings.end();
            ++it) {
          Machine::MemoryMemoryAffinity mma;
          mma.m1 = fbmem->me;
          mma.m2 = it->mem;
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
      // ignored - no way to shoot down HIP work
    }

    void GPUWorkFence::print(std::ostream& os) const
    {
      os << "GPUWorkFence";
    }

    void GPUWorkFence::enqueue_on_stream(GPUStream *stream)
    {
      if(stream->get_gpu()->module->cfg_fences_use_callbacks) {
        CHECK_HIP( hipStreamAddCallback(stream->get_stream(), &cuda_callback, (void *)this, 0) );
      } else {
        stream->add_fence(this);
      }
    }

    /*static*/ void GPUWorkFence::cuda_callback(hipStream_t stream, hipError_t res, void *data)
    {
      GPUWorkFence *me = (GPUWorkFence *)data;

      assert(res == hipSuccess);
      me->mark_finished(true /*succesful*/);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUWorkStart
    GPUWorkStart::GPUWorkStart(Realm::Operation *op)
      : Realm::Operation::AsyncWorkItem(op)
    {
    }

    void GPUWorkStart::print(std::ostream& os) const
    {
      os << "GPUWorkStart";
    }

    void GPUWorkStart::enqueue_on_stream(GPUStream *stream)
    {
      if(stream->get_gpu()->module->cfg_fences_use_callbacks) {
        CHECK_HIP( hipStreamAddCallback(stream->get_stream(), &cuda_start_callback, (void *)this, 0) );
      } else {
        stream->add_start_event(this);
      }
    }
    
    void GPUWorkStart::mark_gpu_work_start()
    {
      op->mark_gpu_work_start();
      mark_finished(true);
    }

    /*static*/ void GPUWorkStart::cuda_start_callback(hipStream_t stream, hipError_t res, void *data)
    {
      GPUWorkStart *me = (GPUWorkStart *)data;
      assert(res == hipSuccess);
      // record the real start time for the operation
      me->mark_gpu_work_start();
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
      CHECK_HIP( hipStreamSynchronize(stream->get_stream()) );
#endif
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUEventPool

    GPUEventPool::GPUEventPool(int _batch_size)
      : batch_size(_batch_size), current_size(0), total_size(0), external_count(0)
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
        CHECK_HIP( hipEventCreateWithFlags(&available_events[i], hipEventDefault) );
    }

    void GPUEventPool::empty_pool(void)
    {
      // shouldn't be any events running around still
      assert((current_size + external_count) == total_size);
      if(external_count)
        log_stream.warning() << "Application leaking " << external_count << " cuda events";

      for(int i = 0; i < current_size; i++)
        CHECK_HIP( hipEventDestroy(available_events[i]) );

      current_size = 0;
      total_size = 0;

      // free internal vector storage
      std::vector<hipEvent_t>().swap(available_events);
    }

    hipEvent_t GPUEventPool::get_event(bool external)
    {
      AutoLock<> al(mutex);

      if(current_size == 0) {
        // if we need to make an event, make a bunch
        current_size = batch_size;
        total_size += batch_size;

        log_stream.info() << "event pool " << this << " depleted - adding " << batch_size << " events";
            
        // resize the vector (considering all events that might come back)
        available_events.resize(total_size);

        for(int i = 0; i < batch_size; i++)
          CHECK_HIP( hipEventCreateWithFlags(&available_events[i], hipEventDefault) );
      }

      if(external)
        external_count++;

      return available_events[--current_size];
    }

    void GPUEventPool::return_event(hipEvent_t e, bool external)
    {
      AutoLock<> al(mutex);

      assert(current_size < total_size);

      if(external) {
        assert(external_count);
        external_count--;
      }
   
      available_events[current_size++] = e;
    }
    
    
    ////////////////////////////////////////////////////////////////////////
    //
    // class ContextSynchronizer

    ContextSynchronizer::ContextSynchronizer(GPU *_gpu,
                                             int _device_id,
                              					     CoreReservationSet& crs,
                              					     int _max_threads)
      : gpu(_gpu)
      , device_id(_device_id)
      , max_threads(_max_threads)
      , condvar(mutex)
      , shutdown_flag(false)
      , total_threads(0)
      , sleeping_threads(0)
      , syncing_threads(0)
    {
      Realm::CoreReservationParameters params;
      params.set_num_cores(1);
      params.set_alu_usage(params.CORE_USAGE_SHARED);
      params.set_fpu_usage(params.CORE_USAGE_MINIMAL);
      params.set_ldst_usage(params.CORE_USAGE_MINIMAL);
      params.set_max_stack_size(1 << 20);

      std::string name = stringbuilder() << "GPU ctxsync " << device_id;

      core_rsrv = new Realm::CoreReservation(name, crs, params);
    }

    ContextSynchronizer::~ContextSynchronizer()
    {
      assert(total_threads == 0);
      delete core_rsrv;
    }

    void ContextSynchronizer::shutdown_threads()
    {
      // set the shutdown flag and wake up everybody
      {
      	AutoLock<> al(mutex);
      	shutdown_flag = true;
      	if(sleeping_threads > 0)
      	  condvar.broadcast();
      }

      for(int i = 0; i < total_threads; i++) {
      	worker_threads[i]->join();
      	delete worker_threads[i];
      }

      worker_threads.clear();
      total_threads = false;
      sleeping_threads = false;
      syncing_threads = false;
      shutdown_flag = false;
    }

    void ContextSynchronizer::add_fence(GPUWorkFence *fence)
    {
      bool start_new_thread = false;
      {
      	AutoLock<> al(mutex);

      	fences.push_back(fence);

      	// if all the current threads are asleep or busy syncing, we
      	//  need to do something
      	if((sleeping_threads + syncing_threads) == total_threads) {
      	  // is there a sleeping thread we can wake up to handle this?
      	  if(sleeping_threads > 0) {
      	    // just poke one of them
      	    condvar.signal();
      	  } else {
      	    // can we start a new thread?  (if not, we'll just have to
      	    //  be patient)
      	    if(total_threads < max_threads) {
      	      total_threads++;
      	      syncing_threads++; // threads starts as if it's syncing
      	      start_new_thread = true;
      	    }
      	  }
      	}
      }

      if(start_new_thread) {
      	Realm::ThreadLaunchParameters tlp;

      	Thread *t = Realm::Thread::create_kernel_thread<ContextSynchronizer,
                                          							&ContextSynchronizer::thread_main>(this,
                                                                  											   tlp,
                                                                  											   *core_rsrv,
                                                                  											   0);
      	// need the mutex to put this thread in the list
      	{
      	  AutoLock<> al(mutex);
      	  worker_threads.push_back(t);
      	}
      }
    }

    void ContextSynchronizer::thread_main()
    {
      while(true) {
      	GPUWorkFence::FenceList my_fences;

      	// attempt to get a non-empty list of fences to synchronize,
      	//  sleeping when needed and paying attention to the shutdown
      	//  flag
      	{
      	  AutoLock<> al(mutex);

      	  syncing_threads--;

      	  while(true) {
      	    if(shutdown_flag)
      	      return;

      	    if(fences.empty()) {
      	      // sleep until somebody tells us there's stuff to do
      	      sleeping_threads++;
      	      condvar.wait();
      	      sleeping_threads--;
      	    } else {
      	      // grab everything (a single sync covers however much stuff
      	      //  was pushed ahead of it)
      	      syncing_threads++;
      	      my_fences.swap(fences);
      	      break;
      	    }
      	  }
      	}

      	// shouldn't get here with an empty list
      	assert(!my_fences.empty());

      	log_stream.debug() << "starting ctx sync: ctx=" << device_id;

      	{
      	  AutoGPUContext agc(gpu);

      	  hipError_t res = hipDeviceSynchronize();

      	  // complain loudly about any errors
      	  if(res != hipSuccess) {
      	    const char *ename = 0;
      	    const char *estr = 0;
      	    ename = hipGetErrorName(res);
      	    estr = hipGetErrorString(res);
      	    log_gpu.fatal() << "HIP error reported on GPU " << gpu->info->index << ": " << estr << " (" << ename << ")";
      	    abort();
      	  }
      	}

      	log_stream.debug() << "finished ctx sync: ctx=" << device_id;

      	// mark all the fences complete
      	while(!my_fences.empty()) {
      	  GPUWorkFence *fence = my_fences.pop_front();
      	  fence->mark_finished(true /*successful*/);
      	}

      	// and go back around for more...
      }
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
      virtual void execute_internal_task(InternalTask *task);

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
      static REALM_THREAD_LOCAL GPUProcessor *current_gpu_proc = 0;
      static REALM_THREAD_LOCAL GPUStream *current_gpu_stream = 0;
      static REALM_THREAD_LOCAL std::set<GPUStream*> *created_gpu_streams = 0;
    };

#ifdef REALM_USE_HIP_HIJACK
    // this flag will be set on the first call into any of the hijack code in
    //  cudart_hijack.cc
    //  an application is linked with -lcudart, we will NOT be hijacking the
    //  application's calls, and the cuda module needs to know that)
    /*extern*/ bool cudart_hijack_active = false;

    // for most HIP API entry points, calling them from a non-GPU task is
    //  a fatal error - for others (e.g. cudaDeviceSynchronize), it's either
    //  silently permitted (0), warned (1), or a fatal error (2) based on this
    //  setting
    /*extern*/ int cudart_hijack_nongpu_sync = 2;

    // used in GPUTaskScheduler<T>::execute_task below
    static bool already_issued_hijack_warning = false;
    static bool already_issued_hijack_enabled_warning = false;
#endif

    template <typename T>
    bool GPUTaskScheduler<T>::execute_task(Task *task)
    {
      // use TLS to make sure that the task can find the current GPU processor when it makes
      //  HIP RT calls
      // TODO: either eliminate these asserts or do TLS swapping when using user threads
      assert(ThreadLocal::current_gpu_proc == 0);
      ThreadLocal::current_gpu_proc = gpu_proc;

      // push the HIP context for this GPU onto this thread
      gpu_proc->gpu->push_context();

      // bump the current stream
      // TODO: sanity-check whether this even works right when GPU tasks suspend
      assert(ThreadLocal::current_gpu_stream == 0);
      GPUStream *s = gpu_proc->gpu->get_next_task_stream();
      ThreadLocal::current_gpu_stream = s;
      assert(!ThreadLocal::created_gpu_streams);

      // we'll use a "work fence" to track when the kernels launched by this task actually
      //  finish - this must be added to the task _BEFORE_ we execute
      GPUWorkFence *fence = new GPUWorkFence(task);
      task->add_async_work_item(fence);

      // event to record the GPU start time for the task, if requested
      if(task->wants_gpu_work_start()) {
	      GPUWorkStart *start = new GPUWorkStart(task);
	      task->add_async_work_item(start);
	      start->enqueue_on_stream(s);
      }

      bool ok = T::execute_task(task);
      
      // if the user could have put work on any other streams then make our
      // stream wait on those streams as well
      // TODO: update this so that it works when GPU tasks suspend
      if(ThreadLocal::created_gpu_streams)
      {
        s->wait_on_streams(*ThreadLocal::created_gpu_streams);
        delete ThreadLocal::created_gpu_streams;
        ThreadLocal::created_gpu_streams = 0;
      }

      // if this is our first task, we might need to decide whether
      //  full context synchronization is required for a task to be
      //  "complete"
      if(gpu_proc->gpu->module->cfg_task_context_sync < 0) {
#ifdef REALM_USE_HIP_HIJACK
        // normally hijack code will catch all the work and put it on the
        //  right stream, but if we haven't seen it used, there may be a
        //  static copy of the cuda runtime that's in use and foiling the
        //  hijack
        if(cudart_hijack_active) {
          gpu_proc->gpu->module->cfg_task_context_sync = 0;
          if (!already_issued_hijack_enabled_warning) {
            already_issued_hijack_enabled_warning = true;
            log_gpu.warning() << "HIP hijack is active"
                  << " - device synchronizations not required after every GPU task!";
          }
        } else {
          if(!(gpu_proc->gpu->module->cfg_suppress_hijack_warning ||
               already_issued_hijack_warning)) {
            already_issued_hijack_warning = true;
            log_gpu.warning() << "HIP hijack code not active"
                              << " - device synchronizations required after every GPU task!";
          }
          //gpu_proc->gpu->module->cfg_task_context_sync = 1;
        }
#else
        // without hijack or legacy sync requested, ctxsync is needed
        gpu_proc->gpu->module->cfg_task_context_sync = 1;
#endif
      }

      if(gpu_proc->gpu->module->cfg_task_context_sync)
        gpu_proc->ctxsync.add_fence(fence);
      else
	      fence->enqueue_on_stream(s);
      
      // A useful debugging macro
#ifdef FORCE_GPU_STREAM_SYNCHRONIZE
      CHECK_HIP( hipStreamSynchronize(s->get_stream()) );
#endif
      
      // pop the HIP context for this GPU back off
      gpu_proc->gpu->pop_context();

      assert(ThreadLocal::current_gpu_proc == gpu_proc);
      ThreadLocal::current_gpu_proc = 0;
      assert(ThreadLocal::current_gpu_stream == s);
      ThreadLocal::current_gpu_stream = 0;

      return ok;
    }

    template <typename T>
    void GPUTaskScheduler<T>::execute_internal_task(InternalTask *task)
    {
      // use TLS to make sure that the task can find the current GPU processor when it makes
      //  HIP RT calls
      // TODO: either eliminate these asserts or do TLS swapping when using user threads
      assert(ThreadLocal::current_gpu_proc == 0);
      ThreadLocal::current_gpu_proc = gpu_proc;

      // push the HIP context for this GPU onto this thread
      gpu_proc->gpu->push_context();

      assert(ThreadLocal::current_gpu_stream == 0);
      GPUStream *s = gpu_proc->gpu->get_next_task_stream();
      ThreadLocal::current_gpu_stream = s;
      assert(!ThreadLocal::created_gpu_streams);

      // internal tasks aren't allowed to wait on events, so any cuda synch
      //  calls inside the call must be blocking
      gpu_proc->block_on_synchronize = true;

      // execute the internal task, whatever it is
      T::execute_internal_task(task);

      // if the user could have put work on any other streams then make our
      // stream wait on those streams as well
      // TODO: update this so that it works when GPU tasks suspend	
      if(ThreadLocal::created_gpu_streams)
      {	
        s->wait_on_streams(*ThreadLocal::created_gpu_streams);	
        delete ThreadLocal::created_gpu_streams;
        ThreadLocal::created_gpu_streams = 0;
      }

      // we didn't use streams here, so synchronize the whole context
      CHECK_HIP( hipDeviceSynchronize() );
      gpu_proc->block_on_synchronize = false;

      // pop the HIP context for this GPU back off
      gpu_proc->gpu->pop_context();

      assert(ThreadLocal::current_gpu_proc == gpu_proc);
      ThreadLocal::current_gpu_proc = 0;
      assert(ThreadLocal::current_gpu_stream == s);
      ThreadLocal::current_gpu_stream = 0;
    }


    ///////////////////////////////////////////////////////////////////////
    //
    // class GPUProcessor

    GPUProcessor::GPUProcessor(GPU *_gpu, Processor _me, Realm::CoreReservationSet& crs,
                               size_t _stack_size)
      : LocalTaskProcessor(_me, Processor::TOC_PROC)
      , gpu(_gpu)
      , block_on_synchronize(false)
      , ctxsync(_gpu, _gpu->device_id, crs, _gpu->module->cfg_max_ctxsync_threads)
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
      void *dptr;
      GPUStream *stream;
      if(dst) {
        dptr = (void *)(dst->fbmem->base + dst_offset);
        stream = peer_to_peer_streams[dst->info->index];
      } else {
        dptr = reinterpret_cast<void *>(dst_offset);
        // HACK!
        stream = hipipc_streams.begin()->second;
      }
      GPUMemcpy *copy = new GPUMemcpy1D(this,
                                        dptr,
                                        (const void *)(fbmem->base + src_offset),
                                        bytes, GPU_MEMCPY_PEER_TO_PEER, notification);
      stream->add_copy(copy);
    }

    void GPU::copy_to_peer_2d(GPU *dst,
                              off_t dst_offset, off_t src_offset,
                              off_t dst_stride, off_t src_stride,
                              size_t bytes, size_t lines,
                              GPUCompletionNotification *notification /*= 0*/)
    {
      void *dptr;
      GPUStream *stream;
      if(dst) {
        dptr = (void *)(dst->fbmem->base + dst_offset);
        stream = peer_to_peer_streams[dst->info->index];
      } else {
        dptr = reinterpret_cast<void *>(dst_offset);
        // HACK!
        stream = hipipc_streams.begin()->second;
      }
      GPUMemcpy *copy = new GPUMemcpy2D(this,
                                        dptr,
                                        (const void *)(fbmem->base + src_offset),
                                        dst_stride, src_stride, bytes, lines,
                                        GPU_MEMCPY_PEER_TO_PEER, notification);
      stream->add_copy(copy);
    }

    void GPU::copy_to_peer_3d(GPU *dst, off_t dst_offset, off_t src_offset,
                              off_t dst_stride, off_t src_stride,
                              off_t dst_height, off_t src_height,
                              size_t bytes, size_t height, size_t depth,
                              GPUCompletionNotification *notification /*= 0*/)
    {
      void *dptr;
      GPUStream *stream;
      if(dst) {
        dptr = (void *)(dst->fbmem->base + dst_offset);
        stream = peer_to_peer_streams[dst->info->index];
      } else {
        dptr = reinterpret_cast<void *>(dst_offset);
        // HACK!
        stream = hipipc_streams.begin()->second;
      }
      GPUMemcpy *copy = new GPUMemcpy3D(this,
                                        dptr,
                                        (const void *)(fbmem->base + src_offset),
                                        dst_stride, src_stride,
                                        dst_height, src_height,
                                        bytes, height, depth,
                                        GPU_MEMCPY_PEER_TO_PEER, notification);
      stream->add_copy(copy);
    }

    static size_t reduce_fill_size(const void *fill_data, size_t fill_data_size)
    {
      const char *as_char = static_cast<const char *>(fill_data);
      // try powers of 2 up to 128 bytes
      for(size_t step = 1; step <= 128; step <<= 1) {
        // must divide evenly
        if((fill_data_size % step) != 0)
          continue;

        // compare to ourselves shifted by the step size - it if matches then
        //  the first few bytes repeat through the rest
        if(!memcmp(as_char, as_char + step, fill_data_size - step))
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
    
    void GPU::fill_within_fb_3d(off_t dst_offset, off_t dst_stride,
    				                    off_t dst_height,
    				                    size_t bytes, size_t height, size_t depth,
    				                    const void *fill_data, size_t fill_data_size,
    				                    GPUCompletionNotification *notification /*= 0*/)
    {
      // GPUMemcpy *copy = new GPUMemset3D(this,
      //           (void *)(fbmem->base + dst_offset),
      //           dst_stride,
      //           dst_height,
      //           bytes, height, depth,
      //           fill_data,
      //           reduce_fill_size(fill_data, fill_data_size),
      //           notification);
      // device_to_device_stream->add_copy(copy);
    }
    
    void GPU::fence_to_fb(Realm::Operation *op)
    {
      GPUWorkFence *f = new GPUWorkFence(op);

      // this must be done before we enqueue the callback with HIP
      op->add_async_work_item(f);

      host_to_device_stream->add_copy(new GPUMemcpyFence(this,
                                                         GPU_MEMCPY_HOST_TO_DEVICE,
                                                         f));
    }

    void GPU::fence_from_fb(Realm::Operation *op)
    {
      GPUWorkFence *f = new GPUWorkFence(op);

      // this must be done before we enqueue the callback with HIP
      op->add_async_work_item(f);

      device_to_host_stream->add_copy(new GPUMemcpyFence(this,
                                                         GPU_MEMCPY_DEVICE_TO_HOST,
                                                         f));
    }

    void GPU::fence_within_fb(Realm::Operation *op)
    {
      GPUWorkFence *f = new GPUWorkFence(op);

      // this must be done before we enqueue the callback with HIP
      op->add_async_work_item(f);

      device_to_device_stream->add_copy(new GPUMemcpyFence(this,
                                                           GPU_MEMCPY_DEVICE_TO_DEVICE,
                                                           f));
    }

    void GPU::fence_to_peer(Realm::Operation *op, GPU *dst)
    {
      GPUWorkFence *f = new GPUWorkFence(op);

      // this must be done before we enqueue the callback with HIP
      op->add_async_work_item(f);

      GPUMemcpyFence *fence = new GPUMemcpyFence(this,
                                                 GPU_MEMCPY_PEER_TO_PEER,
                                                 f);
      peer_to_peer_streams[dst->info->index]->add_copy(fence);
    }

    GPUStream* GPU::find_stream(hipStream_t stream) const
    {
      for (std::vector<GPUStream*>::const_iterator it = 
            task_streams.begin(); it != task_streams.end(); it++)
        if ((*it)->get_stream() == stream)
          return *it;
      return NULL;
    }
    
    GPUStream* GPU::get_null_task_stream(void) const	
    {	
      GPUStream *stream = ThreadLocal::current_gpu_stream;	
      assert(stream != NULL);
      return stream;
    }

    GPUStream* GPU::get_next_task_stream(bool create)
    {
      if(create && !ThreadLocal::created_gpu_streams)
      {
        // First time we get asked to create, user our current stream
        ThreadLocal::created_gpu_streams = new std::set<GPUStream*>();
        assert(ThreadLocal::current_gpu_stream);
        ThreadLocal::created_gpu_streams->insert(ThreadLocal::current_gpu_stream);
        return ThreadLocal::current_gpu_stream;
      }
      unsigned index = next_task_stream.fetch_add(1) % task_streams.size();
      GPUStream *result = task_streams[index];
      if (create)
        ThreadLocal::created_gpu_streams->insert(result);
      return result;
    }

    GPUStream *GPU::get_next_d2d_stream()
    {
      unsigned d2d_stream_index = (next_d2d_stream.fetch_add(1) %
                                   module->cfg_d2d_streams);
      return device_to_device_streams[d2d_stream_index];
    }

    const GPU::HipIpcMapping *GPU::find_ipc_mapping(Memory mem) const
    {
      for(std::vector<HipIpcMapping>::const_iterator it = hipipc_mappings.begin();
          it != hipipc_mappings.end();
          ++it)
        if(it->mem == mem)
          return &*it;

      return 0;
    }

    void GPUProcessor::shutdown(void)
    {
      log_gpu.info("shutting down");

      // shut down threads/scheduler
      LocalTaskProcessor::shutdown();
      
      ctxsync.shutdown_threads();

      // synchronize the device so we can flush any printf buffers - do
      //  this after shutting down the threads so that we know all work is done
      {
        AutoGPUContext agc(gpu);

        CHECK_HIP( hipDeviceSynchronize() );
      }
    }

    GPUWorker::GPUWorker(void)	
      : BackgroundWorkItem("gpu worker")
      , condvar(lock)
      , core_rsrv(0), worker_thread(0)
      , thread_sleeping(false)
      , worker_shutdown_requested(false)
    {}

    GPUWorker::~GPUWorker(void)
    {
      // shutdown should have already been called
      assert(worker_thread == 0);
    }

    void GPUWorker::start_background_thread(Realm::CoreReservationSet &crs,
					    size_t stack_size)
    {
      // shouldn't be doing this if we've registered as a background work item
      assert(manager == 0);
      
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
        AutoLock<> al(lock);
      	worker_shutdown_requested.store(true);
      	if(thread_sleeping) {
      	  thread_sleeping = false;
      	  condvar.broadcast();
      	}
      }

      worker_thread->join();
      delete worker_thread;
      worker_thread = 0;

      delete core_rsrv;
      core_rsrv = 0;
    }

    void GPUWorker::add_stream(GPUStream *stream)
    {
      bool was_empty = false;
      {
      	AutoLock<> al(lock);

#ifdef DEBUG_REALM
      	// insist that the caller de-duplicate these
      	for(ActiveStreamQueue::iterator it = active_streams.begin();
      	    it != active_streams.end();
      	    ++it)
      	  assert(*it != stream);
#endif
      	was_empty = active_streams.empty();
      	active_streams.push_back(stream);

      	if(thread_sleeping) {
      	  thread_sleeping = false;
      	  condvar.broadcast();
      	}
      }

      // if we're a background work item, request attention if needed
      if(was_empty && (manager != 0))
      	make_active();
    }

    bool GPUWorker::do_work(TimeLimit work_until)
    {
      // pop the first stream off the list and immediately become re-active
      //  if more streams remain
      GPUStream *stream = 0;
      bool still_not_empty = false;
      {
        AutoLock<> al(lock);

        assert(!active_streams.empty());
        stream = active_streams.front();
        active_streams.pop_front();
        still_not_empty = !active_streams.empty();
      }
      if(still_not_empty)
	      make_active();

      // do work for the stream we popped, paying attention to the cutoff
      //  time
      bool requeue_stream = false;

      if(stream->reap_events(work_until)) {
        // still work (e.g. copies) to do
        if(work_until.is_expired()) {
          // out of time - save it for later
          requeue_stream = true;
        } else {
          if(stream->issue_copies(work_until))
            requeue_stream = true;
        }
      }

      bool was_empty = false;
      if(requeue_stream) {
        AutoLock<> al(lock);

        was_empty = active_streams.empty();
        active_streams.push_back(stream);
      }
      // note that we can need requeueing even if we called make_active above!
      return was_empty;
    }
    
    bool GPUWorker::process_streams(bool sleep_on_empty)
    {
      GPUStream *cur_stream = 0;
      GPUStream *first_stream = 0;
      bool requeue_stream = false;
    
      while(true) {
      	// grab the front stream in the list
      	{
      	  AutoLock<> al(lock);

      	  // if we didn't finish work on the stream from the previous
      	  //  iteration, add it back to the end
      	  if(requeue_stream)
      	    active_streams.push_back(cur_stream);

      	  while(active_streams.empty()) {
      	    // sleep only if this was the first attempt to get a stream
      	    if(sleep_on_empty && (first_stream == 0) &&
      	       !worker_shutdown_requested.load()) {
      	      thread_sleeping = true;
      	      condvar.wait();
      	    } else
      	      return false;
      	  }

      	  cur_stream = active_streams.front();
      	  // did we wrap around?  if so, stop for now
      	  if(cur_stream == first_stream)
      	    return true;

      	  active_streams.pop_front();
      	  if(!first_stream)
      	    first_stream = cur_stream;
      	}

      	// and do some work for it
      	requeue_stream = false;

      	// both reap_events and issue_copies report whether any kind of work
      	//  remains, so we have to be careful to avoid double-requeueing -
      	//  if the first call returns false, we can't try the second one
      	//  because we may be doing (or failing to do and then requeuing)
      	//  somebody else's work
      	if(!cur_stream->reap_events(TimeLimit())) continue;
      	if(!cur_stream->issue_copies(TimeLimit())) continue;

      	// if we fall all the way through, the queues never went empty at
      	//  any time, so it's up to us to requeue
      	requeue_stream = true;
      }
    }

    void GPUWorker::thread_main(void)
    {
      // TODO: consider busy-waiting in some cases to reduce latency?
      while(!worker_shutdown_requested.load()) {
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
      atomic<bool> completed;
    };

    BlockingCompletionNotification::BlockingCompletionNotification(void)
      : completed(false)
    {}

    BlockingCompletionNotification::~BlockingCompletionNotification(void)
    {}

    void BlockingCompletionNotification::request_completed(void)
    {
      // no condition variable needed - the waiter is spinning
      completed.store(true);
    }

    void BlockingCompletionNotification::wait(void)
    {
      // blocking completion is horrible and should die as soon as possible
      // in the mean time, we need to assist with background work to avoid
      //  the risk of deadlock
      // note that this means you can get NESTED blocking completion
      //  notifications, which is just one of the ways this is horrible
      BackgroundWorkManager::Worker worker;

      worker.set_manager(&(get_runtime()->bgwork));

      while(!completed.load())
        worker.do_work(-1 /* as long as it takes */,
        	             &completed /* until this is set */);
      
    }
	

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUFBMemory

    GPUFBMemory::GPUFBMemory(Memory _me, GPU *_gpu, char *_base, size_t _size)
      : LocalManagedMemory(_me, _size, MKIND_GPUFB, 512, Memory::GPU_FB_MEM, 0)
      , gpu(_gpu), base(_base)
    {
     // advertise for potential gpudirect support
      local_segment.assign(NetworkSegmentInfo::HipDeviceMem,
			                     reinterpret_cast<void *>(base), size,
			                     reinterpret_cast<uintptr_t>(gpu));
      segment = &local_segment;
    }

    GPUFBMemory::~GPUFBMemory(void) {}

    // these work, but they are SLOW
    void GPUFBMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      // use a blocking copy - host memory probably isn't pinned anyway
      {
        AutoGPUContext agc(gpu);
        CHECK_HIP( hipMemcpy
                   (dst, reinterpret_cast<void*>(base + offset), size, hipMemcpyDeviceToHost) );
      }
    }

    void GPUFBMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      // use a blocking copy - host memory probably isn't pinned anyway
      {
        AutoGPUContext agc(gpu);
        CHECK_HIP( hipMemcpy
                  (reinterpret_cast<void*>(base + offset), src, size, hipMemcpyHostToDevice) );
      }
    }

    void *GPUFBMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return (void *)(base + offset);
    }

    // GPUFBMemory supports ExternalHipMemoryResource and
    //  ExternalHipArrayResource (not implemented)
    bool GPUFBMemory::attempt_register_external_resource(RegionInstanceImpl *inst,
                                                         size_t& inst_offset)
    {
      {
        ExternalHipMemoryResource *res = dynamic_cast<ExternalHipMemoryResource *>(inst->metadata.ext_resource);
        if(res) {
          // automatic success
          inst_offset = res->base - reinterpret_cast<uintptr_t>(base); // offset relative to our base
          return true;
        }
      }

      // TODO: add hip array

      // not a kind we recognize
      return false;
    }

    void GPUFBMemory::unregister_external_resource(RegionInstanceImpl *inst)
    {
      // TODO: add hip array
    }

    // for re-registration purposes, generate an ExternalInstanceResource *
    //  (if possible) for a given instance, or a subset of one
    ExternalInstanceResource *GPUFBMemory::generate_resource_info(RegionInstanceImpl *inst,
                                                                  const IndexSpaceGeneric *subspace,
                                                                  span<const FieldID> fields,
                                                                  bool read_only)
    {
      // compute the bounds of the instance relative to our base
      assert(inst->metadata.is_valid() &&
             "instance metadata must be valid before accesses are performed");
      assert(inst->metadata.layout);
      InstanceLayoutGeneric *ilg = inst->metadata.layout;
      uintptr_t rel_base, extent;
      if(subspace == 0) {
        // want full instance
        rel_base = 0;
        extent = ilg->bytes_used;
      } else {
        assert(!fields.empty());
        uintptr_t limit;
        for(size_t i = 0; i < fields.size(); i++) {
          uintptr_t f_base, f_limit;
          if(!subspace->impl->compute_affine_bounds(ilg, fields[i], f_base, f_limit))
            return 0;
          if(i == 0) {
            rel_base = f_base;
            limit = f_limit;
          } else {
            rel_base = std::min(rel_base, f_base);
            limit = std::max(limit, f_limit);
          }
        }
        extent = limit - rel_base;
      }

      uintptr_t abs_base = (reinterpret_cast<uintptr_t>(this->base) + inst->metadata.inst_offset + rel_base);

      return new ExternalHipMemoryResource(gpu->info->index,
                                           abs_base, extent, read_only);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUDynamicMemory

    GPUDynamicFBMemory::GPUDynamicFBMemory(Memory _me, GPU *_gpu,
                                           size_t _max_size)
      : MemoryImpl(_me, _max_size, MKIND_GPUFB, Memory::GPU_DYNAMIC_MEM, 0)
      , gpu(_gpu)
      , cur_size(0)
    {
      // mark what context we belong to
      add_module_specific(new HipDeviceMemoryInfo(gpu->device_id));
    }

    GPUDynamicFBMemory::~GPUDynamicFBMemory(void)
    {
      // free any remaining allocations
      AutoGPUContext agc(gpu);
      AutoLock<> al(mutex);
      for(std::map<RegionInstance, void*>::const_iterator it = alloc_bases.begin();
          it != alloc_bases.end();
          ++it)
        CHECK_HIP( hipFree(it->second) );
      alloc_bases.clear();
    }

    MemoryImpl::AllocationResult GPUDynamicFBMemory::allocate_storage_immediate(RegionInstanceImpl *inst,
                                                                                bool need_alloc_result,
                                                                                bool poisoned,
                                                                                TimeLimit work_until)
    {
      // poisoned allocations are cancellled
      if(poisoned) {
        inst->notify_allocation(ALLOC_CANCELLED,
                                RegionInstanceImpl::INSTOFFSET_FAILED,
                                work_until);
        return ALLOC_CANCELLED;
      }

      // attempt hipMalloc, except for size=0 allocations
      size_t size = inst->metadata.layout->bytes_used;
      void* base = NULL;
      if(size > 0) {
        hipError_t ret;
        {
          AutoGPUContext agc(gpu);
          // TODO: handle large alignments?
          ret = hipMalloc(&base, size);
          if((ret != hipSuccess) && (ret != hipErrorOutOfMemory))
            REPORT_HIP_ERROR("hipMalloc", ret);
        }
        if(ret == hipErrorOutOfMemory) {
          log_gpu.warning() << "out of memory in hipMalloc: size=" << size;
          inst->notify_allocation(ALLOC_INSTANT_FAILURE,
                                  RegionInstanceImpl::INSTOFFSET_FAILED,
                                  work_until);
          return ALLOC_INSTANT_FAILURE;
        }
      }

      // insert entry into our alloc_bases map
      {
        AutoLock<> al(mutex);
        alloc_bases[inst->me] = base;
      }

      inst->notify_allocation(ALLOC_INSTANT_SUCCESS, reinterpret_cast<size_t>(base), work_until);
      return ALLOC_INSTANT_SUCCESS;
    }

    void GPUDynamicFBMemory::release_storage_immediate(RegionInstanceImpl *inst,
                                                       bool poisoned,
                                                       TimeLimit work_until)
    {
      // ignore poisoned releases
      if(poisoned)
        return;

      // for external instances, all we have to do is ack the destruction
      if(inst->metadata.ext_resource != 0) {
        unregister_external_resource(inst);
        inst->notify_deallocation();
        return;
      }

      void* base;
      {
        AutoLock<> al(mutex);
        std::map<RegionInstance, void*>::iterator it = alloc_bases.find(inst->me);
        if(it == alloc_bases.end()) {
          log_gpu.fatal() << "attempt to release unknown instance: inst=" << inst->me;
          abort();
        }
        base = it->second;
        alloc_bases.erase(it);
      }

      if(base != 0) {
        AutoGPUContext agc(gpu);
        CHECK_HIP( hipFree(base) );
      }

      inst->notify_deallocation();
    }

    // these work, but they are SLOW
    void GPUDynamicFBMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      // use a blocking copy - host memory probably isn't pinned anyway
      {
        AutoGPUContext agc(gpu);
        CHECK_HIP( hipMemcpy
                  (dst, reinterpret_cast<void *>(offset), size, hipMemcpyDeviceToHost) );
      }
    }

    void GPUDynamicFBMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      // use a blocking copy - host memory probably isn't pinned anyway
      {
        AutoGPUContext agc(gpu);
        CHECK_HIP( hipMemcpy
                  (reinterpret_cast<void *>(offset), src, size, hipMemcpyHostToDevice) );
      }
    }

    void *GPUDynamicFBMemory::get_direct_ptr(off_t offset, size_t size)
    {
      // offset 'is' the pointer for instances in this memory
      return reinterpret_cast<void *>(offset);
    }

    // GPUFBMemory supports ExternalHipMemoryResource and
    //  ExternalHipArrayResource (not implemented)
    bool GPUDynamicFBMemory::attempt_register_external_resource(RegionInstanceImpl *inst,
                                                                size_t& inst_offset)
    {
      {
        ExternalHipMemoryResource *res = dynamic_cast<ExternalHipMemoryResource *>(inst->metadata.ext_resource);
        if(res) {
          // automatic success
          inst_offset = res->base; // "offsets" are absolute in dynamic fbmem
          return true;
        }
      }

      // TODO: add hip array

      // not a kind we recognize
      return false;
    }

    void GPUDynamicFBMemory::unregister_external_resource(RegionInstanceImpl *inst)
    {
      // TODO: add hip array
    }

    // for re-registration purposes, generate an ExternalInstanceResource *
    //  (if possible) for a given instance, or a subset of one
    ExternalInstanceResource *GPUDynamicFBMemory::generate_resource_info(RegionInstanceImpl *inst,
                                                                         const IndexSpaceGeneric *subspace,
                                                                         span<const FieldID> fields,
                                                                         bool read_only)
    {
      // compute the bounds of the instance relative to our base
      assert(inst->metadata.is_valid() &&
             "instance metadata must be valid before accesses are performed");
      assert(inst->metadata.layout);
      InstanceLayoutGeneric *ilg = inst->metadata.layout;
      uintptr_t rel_base, extent;
      if(subspace == 0) {
        // want full instance
        rel_base = 0;
        extent = ilg->bytes_used;
      } else {
        assert(!fields.empty());
        uintptr_t limit;
        for(size_t i = 0; i < fields.size(); i++) {
          uintptr_t f_base, f_limit;
          if(!subspace->impl->compute_affine_bounds(ilg, fields[i], f_base, f_limit))
            return 0;
          if(i == 0) {
            rel_base = f_base;
            limit = f_limit;
          } else {
            rel_base = std::min(rel_base, f_base);
            limit = std::max(limit, f_limit);
          }
        }
        extent = limit - rel_base;
      }

      uintptr_t abs_base = (inst->metadata.inst_offset + rel_base);

      return new ExternalHipMemoryResource(gpu->info->index,
                                           abs_base, extent, read_only);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUZCMemory

    GPUZCMemory::GPUZCMemory(Memory _me,
                             char *_gpu_base, void *_cpu_base, size_t _size,
                             MemoryKind _kind, Memory::Kind _lowlevel_kind)
      : LocalManagedMemory(_me, _size, _kind, 256, _lowlevel_kind, 0)
      , gpu_base(_gpu_base), cpu_base((char *)_cpu_base)
    {
      // advertise ourselves as a host or managed memory, as appropriate
      NetworkSegmentInfo::MemoryType mtype;
      if(_kind == MemoryImpl::MKIND_MANAGED)
        mtype = NetworkSegmentInfo::HipManagedMem;
      else
        mtype = NetworkSegmentInfo::HostMem;
      local_segment.assign(mtype, cpu_base, size);
      segment = &local_segment;
    }

    GPUZCMemory::~GPUZCMemory(void) {}

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

    // GPUZCMemory supports ExternalHipPinnedHostResource
    bool GPUZCMemory::attempt_register_external_resource(RegionInstanceImpl *inst,
                                                         size_t& inst_offset)
    {
      {
        ExternalHipPinnedHostResource *res = dynamic_cast<ExternalHipPinnedHostResource *>(inst->metadata.ext_resource);
        if(res) {
          // automatic success - offset relative to our base
          inst_offset = res->base - reinterpret_cast<uintptr_t>(cpu_base);
          return true;
        }
      }

      // not a kind we recognize
      return false;
    }

    void GPUZCMemory::unregister_external_resource(RegionInstanceImpl *inst)
    {
      // nothing actually to clean up
    }

    // for re-registration purposes, generate an ExternalInstanceResource *
    //  (if possible) for a given instance, or a subset of one
    ExternalInstanceResource *GPUZCMemory::generate_resource_info(RegionInstanceImpl *inst,
                                                                  const IndexSpaceGeneric *subspace,
                                                                  span<const FieldID> fields,
                                                                  bool read_only)
    {
      // compute the bounds of the instance relative to our base
      assert(inst->metadata.is_valid() &&
             "instance metadata must be valid before accesses are performed");
      assert(inst->metadata.layout);
      InstanceLayoutGeneric *ilg = inst->metadata.layout;
      uintptr_t rel_base, extent;
      if(subspace == 0) {
        // want full instance
        rel_base = 0;
        extent = ilg->bytes_used;
      } else {
        assert(!fields.empty());
        uintptr_t limit;
        for(size_t i = 0; i < fields.size(); i++) {
          uintptr_t f_base, f_limit;
          if(!subspace->impl->compute_affine_bounds(ilg, fields[i], f_base, f_limit))
            return 0;
          if(i == 0) {
            rel_base = f_base;
            limit = f_limit;
          } else {
            rel_base = std::min(rel_base, f_base);
            limit = std::max(limit, f_limit);
          }
        }
        extent = limit - rel_base;
      }

      void *mem_base = (this->cpu_base +
                        inst->metadata.inst_offset +
                        rel_base);

      return new ExternalHipPinnedHostResource(reinterpret_cast<uintptr_t>(mem_base),
                                               extent, read_only);
    }

    
    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUFBIBMemory

    GPUFBIBMemory::GPUFBIBMemory(Memory _me, GPU *_gpu,
                                 char *_base, size_t _size)
      : IBMemory(_me, _size, MKIND_GPUFB, Memory::GPU_FB_MEM,
                 reinterpret_cast<void *>(_base), 0)
      , gpu(_gpu)
      , base(_base)
    {
      // advertise for potential gpudirect support
      local_segment.assign(NetworkSegmentInfo::HipDeviceMem,
			   reinterpret_cast<void *>(_base), _size,
			   reinterpret_cast<uintptr_t>(_gpu));
      segment = &local_segment;
    }
    

    // Helper methods for emulating the cuda runtime
    /*static*/ GPUProcessor* GPUProcessor::get_current_gpu_proc(void)
    {
      return ThreadLocal::current_gpu_proc;
    }

#ifdef REALM_USE_HIP_HIJACK
    void GPUProcessor::push_call_configuration(dim3 grid_dim, dim3 block_dim,
                                               size_t shared_size, void *stream)
    {
      call_configs.push_back(CallConfig(grid_dim, block_dim,
                                        shared_size, (hipStream_t)stream));
    }

    void GPUProcessor::pop_call_configuration(dim3 *grid_dim, dim3 *block_dim,
                                              size_t *shared_size, void *stream)
    {
      assert(!call_configs.empty());
      const CallConfig &config = call_configs.back();
      *grid_dim = config.grid;
      *block_dim = config.block;
      *shared_size = config.shared;
      *((hipStream_t*)stream) = config.stream;
      call_configs.pop_back();
    }
#endif
    
    void GPUProcessor::stream_wait_on_event(hipStream_t stream, hipEvent_t event)
    {
      if (IS_DEFAULT_STREAM(stream))
        CHECK_HIP( hipStreamWaitEvent(
              ThreadLocal::current_gpu_stream->get_stream(), event, 0) );
      else
        CHECK_HIP( hipStreamWaitEvent(stream, event, 0) );
    }

    void GPUProcessor::stream_synchronize(hipStream_t stream)
    {
      // same as device_synchronize if stream is zero
      if (!IS_DEFAULT_STREAM(stream))
      {
        if(!block_on_synchronize) {
          GPUStream *s = gpu->find_stream(stream);
          if(s) {
            // We don't actually want to block the GPU processor
            // when synchronizing, so we instead register a cuda
            // event on the stream and then use it triggering to
            // indicate that the stream is caught up
            // Make a completion notification to be notified when
            // the event has actually triggered
            GPUPreemptionWaiter waiter(gpu);
            // Register the waiter with the stream 
            s->add_notification(&waiter); 
            // Perform the wait, this will preempt the thread
            waiter.preempt();
          } else {
            log_gpu.warning() << "WARNING: Detected unknown HIP stream "
              << stream << " that Realm did not create which suggests "
              << "that there is another copy of the HIP runtime "
              << "somewhere making its own streams... be VERY careful.";
            CHECK_HIP( hipStreamSynchronize(stream) );
          }
        } else {
          // oh well...
          CHECK_HIP( hipStreamSynchronize(stream) );
        }
      }
      else
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
      GPUStream *current = ThreadLocal::current_gpu_stream;

      if(ThreadLocal::created_gpu_streams)
      {
        current->wait_on_streams(*ThreadLocal::created_gpu_streams); 
        delete ThreadLocal::created_gpu_streams;
        ThreadLocal::created_gpu_streams = 0;
      }
      
      if(!block_on_synchronize) {
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
      } else {
        // oh well...
        CHECK_HIP( hipStreamSynchronize(current->get_stream()) ); 	
      }
    }

#ifdef REALM_USE_HIP_HIJACK    
    void GPUProcessor::event_create(hipEvent_t *event, int flags)
    {
      // int cu_flags = CU_EVENT_DEFAULT;
      // if((flags & cudaEventBlockingSync) != 0)
      // 	cu_flags |= CU_EVENT_BLOCKING_SYNC;
      // if((flags & cudaEventDisableTiming) != 0)
      // 	cu_flags |= CU_EVENT_DISABLE_TIMING;

      // get an event from our event pool (ignoring the flags for now)
      hipEvent_t e = gpu->event_pool.get_event(true/*external*/);
      *event = e;
    }

    void GPUProcessor::event_destroy(hipEvent_t event)
    {
      // assume the event is one of ours and put it back in the pool
      hipEvent_t e = event;
      if(e)
        gpu->event_pool.return_event(e, true/*external*/);
    }

    void GPUProcessor::event_record(hipEvent_t event, hipStream_t stream)
    {
      // ignore the provided stream and record the event on this task's assigned stream
      hipEvent_t e = event;
      if(IS_DEFAULT_STREAM(stream))
        stream = ThreadLocal::current_gpu_stream->get_stream();
      CHECK_HIP( hipEventRecord(e, stream) );
    }

    void GPUProcessor::event_synchronize(hipEvent_t event)
    {
      // TODO: consider suspending task rather than busy-waiting here...
      hipEvent_t e = event;
      CHECK_HIP( hipEventSynchronize(e) );
    }
      
    void GPUProcessor::event_elapsed_time(float *ms, hipEvent_t start, hipEvent_t end)
    {
      // TODO: consider suspending task rather than busy-waiting here...
      hipEvent_t e1 = start;
      hipEvent_t e2 = end;
      CHECK_HIP( hipEventElapsedTime(ms, e1, e2) );
    }
      
    GPUProcessor::LaunchConfig::LaunchConfig(dim3 _grid, dim3 _block, size_t _shared)
      : grid(_grid), block(_block), shared(_shared)
    {}

    GPUProcessor::CallConfig::CallConfig(dim3 _grid, dim3 _block, 
                                         size_t _shared, hipStream_t _stream)
      : LaunchConfig(_grid, _block, _shared), stream(_stream)
    {}

    void GPUProcessor::configure_call(dim3 grid_dim,
                                      dim3 block_dim,
                                      size_t shared_mem,
                                      hipStream_t stream)
                                      {
      launch_configs.push_back(CallConfig(grid_dim, block_dim, shared_mem, stream));
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
      CallConfig &config = launch_configs.back();

      // Find our function
      hipFunction_t f = gpu->lookup_function(func);

      size_t arg_size = kernel_args.size();
      void *extra[] = { 
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernel_args[0],
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END
      };

      if (IS_DEFAULT_STREAM(config.stream))
        config.stream = ThreadLocal::current_gpu_stream->get_stream();
      log_stream.debug() << "kernel " << func << " added to stream " << config.stream;

      // Launch the kernel on our stream dammit!
      CHECK_HIP( hipModuleLaunchKernel(f, 
			                                config.grid.x, config.grid.y, config.grid.z,
                                      config.block.x, config.block.y, config.block.z,
                                      config.shared,
			                                config.stream,
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
                                     hipStream_t stream)
    {
      if (IS_DEFAULT_STREAM(stream))
        stream = ThreadLocal::current_gpu_stream->get_stream();
      log_stream.debug() << "kernel " << func << " added to stream " << stream;
      /*
      // Launch the kernel on our stream dammit!
      CHECK_HIP( hipLaunchKernelGGL(func,
                               grid_dim, block_dim,
                               shared_memory,
                               stream,
                               args) );
    */
    }
#endif

    void GPUProcessor::gpu_memcpy(void *dst, const void *src, size_t size,
                                  hipMemcpyKind kind)
    {
      hipStream_t current = ThreadLocal::current_gpu_stream->get_stream();
      // the synchronous copy still uses cuMemcpyAsync so that we can limit the
      //  synchronization to just the right stream
      CHECK_HIP( hipMemcpyAsync(dst, src, size, kind, current) );
      stream_synchronize(current);    
    }

    void GPUProcessor::gpu_memcpy_async(void *dst, const void *src, size_t size,
                                        hipMemcpyKind kind, hipStream_t stream)
    {
      if (IS_DEFAULT_STREAM(stream))
        stream = ThreadLocal::current_gpu_stream->get_stream();
      CHECK_HIP( hipMemcpyAsync(dst, src, size, kind, stream) );
      // no synchronization here
    }

#ifdef REALM_USE_HIP_HIJACK
    void GPUProcessor::gpu_memcpy_to_symbol(const void *dst, const void *src,
                                            size_t size, size_t offset,
                                            hipMemcpyKind kind)
    {
      hipStream_t current = ThreadLocal::current_gpu_stream->get_stream();
      char *var_base = gpu->lookup_variable(dst);
      CHECK_HIP( hipMemcpyAsync((void *)(var_base + offset),
                                src, size, kind, current) );
      stream_synchronize(current);
    }

    void GPUProcessor::gpu_memcpy_to_symbol_async(const void *dst, const void *src,
                                                  size_t size, size_t offset,
                                                  hipMemcpyKind kind, hipStream_t stream)
    {
      if (IS_DEFAULT_STREAM(stream))
        stream = ThreadLocal::current_gpu_stream->get_stream();
      char *var_base = gpu->lookup_variable(dst);
      CHECK_HIP( hipMemcpyAsync((void *)(var_base + offset),
                                src, size, kind, stream) );
      // no synchronization here   
    }

    void GPUProcessor::gpu_memcpy_from_symbol(void *dst, const void *src,
                                              size_t size, size_t offset,
                                              hipMemcpyKind kind)
    {
      hipStream_t current = ThreadLocal::current_gpu_stream->get_stream();
      char *var_base = gpu->lookup_variable(src);
      CHECK_HIP( hipMemcpyAsync(dst,
			      (void *)(var_base + offset),
			      size, kind, current) );
      stream_synchronize(current);    
    }

    void GPUProcessor::gpu_memcpy_from_symbol_async(void *dst, const void *src,
                                                    size_t size, size_t offset,
                                                    hipMemcpyKind kind, hipStream_t stream)
    {
      if (IS_DEFAULT_STREAM(stream))
        stream = ThreadLocal::current_gpu_stream->get_stream();
      char *var_base = gpu->lookup_variable(src);
      CHECK_HIP( hipMemcpyAsync(dst,
			                        (void *)(var_base + offset),
			                        size, kind, stream) );
      // no synchronization here    
    }
#endif

    void GPUProcessor::gpu_memset(void *dst, int value, size_t count)
    {
      hipStream_t current = ThreadLocal::current_gpu_stream->get_stream();
      CHECK_HIP( hipMemsetAsync(dst, (unsigned char)value, 
                                count, current) );    
    }

    void GPUProcessor::gpu_memset_async(void *dst, int value, 
                                        size_t count, hipStream_t stream)
    {
      if (IS_DEFAULT_STREAM(stream))
        stream = ThreadLocal::current_gpu_stream->get_stream();
      CHECK_HIP( hipMemsetAsync(dst, (unsigned char)value,
                                count, stream) );    
    }
    
    ////////////////////////////////////////////////////////////////////////
    //
    // class GPU

    GPU::GPU(HipModule *_module, GPUInfo *_info, GPUWorker *_worker,
             int _device_id)
      : module(_module), info(_info), worker(_worker)
      , proc(0), fbmem(0), fb_ibmem(0)
      , device_id(_device_id), fbmem_base(0), fb_ibmem_base(0)
      , next_task_stream(0), next_d2d_stream(0)
    {
      push_context();

      CHECK_HIP( hipDeviceGetStreamPriorityRange(&least_stream_priority,
                                                &greatest_stream_priority) );

      event_pool.init_pool();

      host_to_device_stream = new GPUStream(this, worker);
      device_to_host_stream = new GPUStream(this, worker);
      
      device_to_device_streams.resize(module->cfg_d2d_streams, 0);
      for(unsigned i = 0; i < module->cfg_d2d_streams; i++)
        device_to_device_streams[i] = new GPUStream(this, worker,
                                                    module->cfg_d2d_stream_priority);

      // only create p2p streams for devices we can talk to
      peer_to_peer_streams.resize(module->gpu_info.size(), 0);
      for(std::vector<GPUInfo *>::const_iterator it = module->gpu_info.begin();
          it != module->gpu_info.end();
          ++it)
        if(info->peers.count((*it)->device) != 0)
          peer_to_peer_streams[(*it)->index] = new GPUStream(this, worker);

      task_streams.resize(module->cfg_task_streams);
      for(unsigned i = 0; i < module->cfg_task_streams; i++)
        task_streams[i] = new GPUStream(this, worker);

      pop_context();

#ifdef REALM_USE_HIP_HIJACK
      // now hook into the cuda runtime fatbin/etc. registration path
      GlobalRegistrations::add_gpu_context(this);
#endif
    }

    GPU::~GPU(void)
    {
      push_context();

      event_pool.empty_pool();

      // destroy streams
      delete host_to_device_stream;
      delete device_to_host_stream;
    
      delete_container_contents(device_to_device_streams);

      for(std::vector<GPUStream *>::iterator it = peer_to_peer_streams.begin();
          it != peer_to_peer_streams.end();
          ++it)
        if(*it)
          delete *it;

      for(std::map<NodeID, GPUStream *>::iterator it = hipipc_streams.begin();
          it != hipipc_streams.end();
          ++it)
        delete it->second;

      delete_container_contents(task_streams);

      // free memory
      if(fbmem_base)
        CHECK_HIP( hipFree((void *)fbmem_base) );
      
      if(fb_ibmem_base)
        CHECK_HIP( hipFree((void *)fb_ibmem_base) );

      //CHECK_HIP( hipDevicePrimaryCtxRelease(info->device) );
    }

    void GPU::push_context(void)
    {
      //CHECK_HIP( hipCtxPushCurrent(context) );
      CHECK_HIP( hipSetDevice(device_id) );
    }

    void GPU::pop_context(void)
    {
      // the context we pop had better be ours...
      //hipCtx_t popped;
      //CHECK_HIP( hipCtxPopCurrent(&popped) );
      //assert(popped == context);
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

      	// enable peer access (this part is different from CUDA since runtime API has no CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
        //  (don't try if it's the same physical device underneath)
      	if(info != (*it)->info) {
      	  AutoGPUContext agc(this);

          CHECK_HIP( hipDeviceEnablePeerAccess((*it)->device_id, 0) );
      	}
      	log_gpu.info() << "peer access enabled from GPU " << p << " to FB " << (*it)->fbmem->me;
      	peer_fbs.insert((*it)->fbmem->me);
        
        if((*it)->fb_ibmem)
          peer_fbs.insert((*it)->fb_ibmem->me);

      	{
      	  Machine::ProcessorMemoryAffinity pma;
      	  pma.p = p;
      	  pma.m = (*it)->fbmem->me;
      	  pma.bandwidth = 10; // assuming pcie, this should be ~half the bw and
      	  pma.latency = 400;  // ~twice the latency as zcmem
      	  runtime->add_proc_mem_affinity(pma);
      	}
      }

      // look for any other local memories that belong to our context or
      //  peer-able contexts
      const Node& n = get_runtime()->nodes[Network::my_node_id];
      for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
          it != n.memories.end();
          ++it) {
        HipDeviceMemoryInfo *cdm = (*it)->find_module_specific<HipDeviceMemoryInfo>();
        if(!cdm) continue;
        if(cdm->device_id == device_id) {
          Machine::ProcessorMemoryAffinity pma;
          pma.p = p;
          pma.m = (*it)->me;
          pma.bandwidth = 200;  // "big"
          pma.latency = 5;      // "ok"
          runtime->add_proc_mem_affinity(pma);
        } else {
          // if the other context is associated with a gpu and we've got peer
          //  access, use it
          // TODO: add option to enable peer access at this point?  might be
          //  expensive...
          if(cdm->gpu && (info->peers.count(cdm->gpu->info->device) > 0)) {
            Machine::ProcessorMemoryAffinity pma;
            pma.p = p;
            pma.m = (*it)->me;
            pma.bandwidth = 10; // assuming pcie, this should be ~half the bw and
            pma.latency = 400;  // ~twice the latency as zcmem
            runtime->add_proc_mem_affinity(pma);
          }
        }
      }
    }

    void GPU::create_fb_memory(RuntimeImpl *runtime, size_t size, size_t ib_size)
    {
      // need the context so we can get an allocation in the right place
      {
      	AutoGPUContext agc(this);

      	hipError_t ret = hipMalloc((void **)&fbmem_base, size);
        assert(ret == hipSuccess);
	      if(ret != hipSuccess) {
	        if(ret == hipErrorMemoryAllocation) {
      	    size_t free_bytes, total_bytes;
      	    CHECK_HIP( hipMemGetInfo(&free_bytes, &total_bytes) );
      	    log_gpu.fatal() << "insufficient memory on gpu " << info->index
                            << ": " << size << " bytes needed (from -ll:fsize), "
                            << free_bytes << " (out of " << total_bytes << ") available";
	        } else {
	          const char *errstring = "error message not available";
	          errstring = hipGetErrorName(ret);
      	    log_gpu.fatal() << "unexpected error from hipMalloc on gpu " << info->index
                            << ": result=" << ret
                            << " (" << errstring << ")";
	        }
	        abort();
	      }
      }

      Memory m = runtime->next_local_memory_id();
      fbmem = new GPUFBMemory(m, this, static_cast<char*>(fbmem_base), size);
      runtime->add_memory(fbmem);
      
      // FB ibmem is a separate allocation for now (consider merging to make
      //  total number of allocations, network registrations, etc. smaller?)
      if(ib_size > 0) {
        {
          AutoGPUContext agc(this);

          hipError_t ret = hipMalloc((void **)&fb_ibmem_base, ib_size);
          if(ret != hipSuccess) {
            if(ret == hipErrorMemoryAllocation) {
              size_t free_bytes, total_bytes;
              CHECK_HIP( hipMemGetInfo(&free_bytes, &total_bytes) );
              log_gpu.fatal() << "insufficient memory on gpu " << info->index
                              << ": " << ib_size << " bytes needed (from -ll:ib_fsize), "
                              << free_bytes << " (out of " << total_bytes << ") available";
            } else {
                const char *errstring = "error message not available";
                errstring = hipGetErrorName(ret);
                log_gpu.fatal() << "unexpected error from hipMalloc on gpu " << info->index
                                << ": result=" << ret
                                << " (" << errstring << ")";
            }
            abort();
          }
        }

        Memory m = runtime->next_local_ib_memory_id();
        fb_ibmem = new GPUFBIBMemory(m, this, fb_ibmem_base, ib_size);
        runtime->add_ib_memory(fb_ibmem);
      }
    }

    void GPU::create_dynamic_fb_memory(RuntimeImpl *runtime, size_t max_size)
    {
      // if the max_size is non-zero, also limit by what appears to be
      //  currently available
      if(max_size > 0) {
        AutoGPUContext agc(this);

        size_t free_bytes, total_bytes;
        CHECK_HIP( hipMemGetInfo(&free_bytes, &total_bytes) );
        if(total_bytes < max_size)
          max_size = total_bytes;
      }

      Memory m = runtime->next_local_memory_id();
      GPUDynamicFBMemory *dfb = new GPUDynamicFBMemory(m, this, max_size);
      runtime->add_memory(dfb);
    }

#ifdef REALM_USE_HIP_HIJACK
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
        hipModule_t module = load_hip_module(fatbin->data);
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
      std::map<const FatBin *, hipModule_t>::const_iterator it = device_modules.find(var->fat_bin);
      assert(it != device_modules.end());
      hipModule_t module = it->second;

      hipDeviceptr_t ptr;
      size_t size;
      CHECK_HIP( hipModuleGetGlobal(&ptr, &size, module, var->device_name) );
      device_variables[var->host_var] = reinterpret_cast<char*>(ptr);
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
      std::map<const FatBin *, hipModule_t>::const_iterator it = device_modules.find(func->fat_bin);
      assert(it != device_modules.end());
      hipModule_t module = it->second;

      hipFunction_t f;
      CHECK_HIP( hipModuleGetFunction(&f, module, func->device_fun) );
      device_functions[func->host_fun] = f;
    }

    hipFunction_t GPU::lookup_function(const void *func)
    {
      std::map<const void *, hipFunction_t>::iterator finder = device_functions.find(func);
      assert(finder != device_functions.end());
      return finder->second;
    }

    char* GPU::lookup_variable(const void *var)
    {
      std::map<const void *, char *>::iterator finder = device_variables.find(var);
      assert(finder != device_variables.end());
      return finder->second;
    }
#endif

    hipModule_t GPU::load_hip_module(const void *data)
    {
      const unsigned num_options = 4;
      hipJitOption jit_options[num_options];
      void*        option_vals[num_options];
      const size_t buffer_size = 16384;
      char* log_info_buffer = (char*)malloc(buffer_size);
      char* log_error_buffer = (char*)malloc(buffer_size);
      jit_options[0] = hipJitOptionInfoLogBuffer;
      jit_options[1] = hipJitOptionInfoLogBufferSizeBytes;
      jit_options[2] = hipJitOptionErrorLogBuffer;
      jit_options[3] = hipJitOptionErrorLogBufferSizeBytes;
      option_vals[0] = log_info_buffer;
      option_vals[1] = (void*)buffer_size;
      option_vals[2] = log_error_buffer;
      option_vals[3] = (void*)buffer_size;
      hipModule_t module;
      hipError_t result = hipModuleLoadDataEx(&module, data, num_options, 
                                           jit_options, option_vals); 
      if (result != hipSuccess)
      {
#ifdef REALM_ON_MACOS
        if (result == hipErrorOperatingSystem) {
          log_gpu.error("ERROR: Device side asserts are not supported by the "
                              "HIP driver for MAC OSX, see NVBugs 1628896.");
        } else
#endif
        if (result == hipErrorNoBinaryForGpu) {
          log_gpu.error("ERROR: The binary was compiled for the wrong GPU "
                              "architecture. Update the 'GPU_ARCH' flag at the top "
                              "of runtime/runtime.mk to match/include your current GPU "
			      "architecture (%d).",
			(info->major * 10 + info->minor));
        } else {
	  log_gpu.error("Failed to load HIP module! Error log: %s", 
			log_error_buffer);
#if HIP_VERBOSE_ERROR_MSG == 1
	  const char *name, *str;
	  name = hipGetErrorName(result);
	  str = hipGetErrorString(result);
	  fprintf(stderr,"CU: hipModuleLoadDataEx = %d (%s): %s\n",
		  result, name, str);
#else
	  fprintf(stderr,"CU: hipModuleLoadDataEx = %d\n", result);
#endif
	}
	abort();
      }
      else
        log_gpu.info("Loaded HIP Module. JIT Output: %s", log_info_buffer);
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
      if(gpu)
        gpu->push_context();
    }

    AutoGPUContext::~AutoGPUContext(void)
    {
      if(gpu)
        gpu->pop_context();
    }
 

    ////////////////////////////////////////////////////////////////////////
    //
    // class CudaModule

    // our interface to the rest of the runtime

    HipModule *hip_module_singleton = 0;

    HipModule::HipModule(void)
      : Module("hip")
      , cfg_zc_mem_size(64 << 20)
      , cfg_zc_ib_size(256 << 20)
      , cfg_fb_mem_size(256 << 20)
      , cfg_fb_ib_size(128 << 20)
      , cfg_use_dynamic_fb(true)
      , cfg_dynfb_max_size(~size_t(0))
      , cfg_num_gpus(0)
      , cfg_task_streams(12)
      , cfg_d2d_streams(4)
      , cfg_use_worker_threads(false)
      , cfg_use_shared_worker(true)
      , cfg_pin_sysmem(true)
      , cfg_fences_use_callbacks(false)
      , cfg_suppress_hijack_warning(false)
      , cfg_skip_gpu_count(0)
      , cfg_skip_busy_gpus(false)
      , cfg_min_avail_mem(0)
      , cfg_task_context_sync(-1)
      , cfg_max_ctxsync_threads(4)
      , cfg_multithread_dma(false)
      , cfg_hostreg_limit(1 << 30)
      , cfg_d2d_stream_priority(-1)
      , cfg_use_hip_ipc(false)
      , shared_worker(0), zcmem_cpu_base(0)
      , zcib_cpu_base(0), zcmem(0)
      , hipipc_condvar(hipipc_mutex)
      , hipipc_responses_needed(0)
      , hipipc_releases_needed(0)
      , hipipc_exports_remaining(0)
    {
      assert(!hip_module_singleton);
      hip_module_singleton = this;
    }
      
    HipModule::~HipModule(void)
    {
      delete_container_contents(gpu_info);
      assert(hip_module_singleton == this);
      hip_module_singleton = 0;
    }

    /*static*/ Module *HipModule::create_module(RuntimeImpl *runtime,
                                                std::vector<std::string>& cmdline)
    {
      HipModule *m = new HipModule;
      
      // first order of business - read command line parameters
      {
        CommandLineParser cp;

      	cp.add_option_int_units("-ll:fsize", m->cfg_fb_mem_size, 'm')
      	  .add_option_int_units("-ll:zsize", m->cfg_zc_mem_size, 'm')
          .add_option_int_units("-ll:ib_fsize", m->cfg_fb_ib_size, 'm')
      	  .add_option_int_units("-ll:ib_zsize", m->cfg_zc_ib_size, 'm')
          .add_option_int("-hip:dynfb", m->cfg_use_dynamic_fb)
          .add_option_int_units("-hip:dynfb_max", m->cfg_dynfb_max_size, 'm')
      	  .add_option_int("-ll:gpu", m->cfg_num_gpus)
          .add_option_string("-ll:gpu_ids", m->cfg_gpu_idxs)
          .add_option_int("-ll:streams", m->cfg_task_streams)
          .add_option_int("-ll:d2d_streams", m->cfg_d2d_streams)
          .add_option_int("-ll:d2d_priority", m->cfg_d2d_stream_priority)
          .add_option_int("-ll:gpuworkthread", m->cfg_use_worker_threads)
      	  .add_option_int("-ll:gpuworker", m->cfg_use_shared_worker)
      	  .add_option_int("-ll:pin", m->cfg_pin_sysmem)
      	  .add_option_bool("-hip:callbacks", m->cfg_fences_use_callbacks)
      	  .add_option_bool("-hip:nohijack", m->cfg_suppress_hijack_warning)	
      	  .add_option_int("-hip:skipgpus", m->cfg_skip_gpu_count)
      	  .add_option_bool("-hip:skipbusy", m->cfg_skip_busy_gpus)
      	  .add_option_int_units("-hip:minavailmem", m->cfg_min_avail_mem, 'm')
          .add_option_int("-hip:contextsync", m->cfg_task_context_sync)
          .add_option_int("-hip:maxctxsync", m->cfg_max_ctxsync_threads)
          .add_option_int("-hip:mtdma", m->cfg_multithread_dma)
          .add_option_int_units("-hip:hostreg", m->cfg_hostreg_limit, 'm')
          .add_option_int("-hip:ipc", m->cfg_use_hip_ipc);
#ifdef REALM_USE_HIP_HIJACK
        cp.add_option_int("-hip:nongpusync", cudart_hijack_nongpu_sync);
#endif	
        
        bool ok = cp.parse_command_line(cmdline);
      	if(!ok) {
      	  log_gpu.error() << "error reading HIP command line parameters";
      	  exit(1);
      	}
      }

      // before we do anything, make sure there's a HIP driver and GPUs to talk to
      std::vector<GPUInfo *> infos;
      {
        int num_devices;
      	CHECK_HIP( hipGetDeviceCount(&num_devices) );
      	if(num_devices == 0) {
          // continue on so that we recognize things like -ll:gpu, but there
          //  are no devices to be found
          log_gpu.info() << "hipInit reports no devices found";
      	} else {
      	  for(int i = 0; i < num_devices; i++) {
      	    GPUInfo *info = new GPUInfo;

      	    info->index = i;
      	    info->device = i;
            hipDeviceProp_t dev_prop;
            CHECK_HIP( hipGetDeviceProperties(&dev_prop, info->device) );
            memcpy(info->name, dev_prop.name, GPUInfo::MAX_NAME_LEN);
            info->major = dev_prop.major;
            info->minor = dev_prop.minor;
            info->totalGlobalMem = dev_prop.totalGlobalMem;
#ifdef REALM_USE_HIP_HIJACK
            // We only need the rest of these properties for the hijack
#define GET_DEVICE_PROP(member, name)					\
            do {								\
              int tmp;								\
              CHECK_HIP( hipDeviceGetAttribute(&tmp, hipDeviceAttribute##name, info->device) ); \
              info->member = tmp;						\
            } while(0)
            // SCREW TEXTURES AND SURFACES FOR NOW!
            GET_DEVICE_PROP(sharedMemPerBlock, MaxSharedMemoryPerBlock);
            GET_DEVICE_PROP(regsPerBlock, MaxRegistersPerBlock);
            GET_DEVICE_PROP(warpSize, WarpSize);
            // GET_DEVICE_PROP(memPitch, MAX_PITCH);
            GET_DEVICE_PROP(maxThreadsPerBlock, MaxThreadsPerBlock);
            GET_DEVICE_PROP(maxThreadsDim[0], MaxBlockDimX);
            GET_DEVICE_PROP(maxThreadsDim[1], MaxBlockDimY);
            GET_DEVICE_PROP(maxThreadsDim[2], MaxBlockDimZ);
            GET_DEVICE_PROP(maxGridSize[0], MaxGridDimX);
            GET_DEVICE_PROP(maxGridSize[1], MaxGridDimY);
            GET_DEVICE_PROP(maxGridSize[2], MaxGridDimZ);
            GET_DEVICE_PROP(clockRate, ClockRate);
            GET_DEVICE_PROP(totalConstMem, TotalConstantMemory);
            // GET_DEVICE_PROP(deviceOverlap, GPU_OVERLAP);
            GET_DEVICE_PROP(multiProcessorCount, MultiprocessorCount );
            // GET_DEVICE_PROP(kernelExecTimeoutEnabled, KERNEL_EXEC_TIMEOUT);
            // GET_DEVICE_PROP(integrated, INTEGRATED);
            // GET_DEVICE_PROP(canMapHostMemory, CAN_MAP_HOST_MEMORY);
            GET_DEVICE_PROP(computeMode, ComputeMode);
            GET_DEVICE_PROP(concurrentKernels, ConcurrentKernels);
            // GET_DEVICE_PROP(ECCEnabled, ECC_ENABLED);
            GET_DEVICE_PROP(pciBusID, PciBusId);
            GET_DEVICE_PROP(pciDeviceID, PciDeviceId);
            // GET_DEVICE_PROP(pciDomainID, PCI_DOMAIN_ID);
            // GET_DEVICE_PROP(tccDriver, TCC_DRIVER);
            // GET_DEVICE_PROP(asyncEngineCount, ASYNC_ENGINE_COUNT);
            // GET_DEVICE_PROP(unifiedAddressing, UNIFIED_ADDRESSING);
            GET_DEVICE_PROP(memoryClockRate, MemoryClockRate);
            GET_DEVICE_PROP(memoryBusWidth, MemoryBusWidth);
            GET_DEVICE_PROP(l2CacheSize, L2CacheSize);
            GET_DEVICE_PROP(maxThreadsPerMultiProcessor, MaxThreadsPerMultiProcessor);
            // GET_DEVICE_PROP(streamPrioritiesSupported, STREAM_PRIORITIES_SUPPORTED);
            // GET_DEVICE_PROP(globalL1CacheSupported, GLOBAL_L1_CACHE_SUPPORTED);
            // GET_DEVICE_PROP(localL1CacheSupported, LOCAL_L1_CACHE_SUPPORTED);
            GET_DEVICE_PROP(maxSharedMemoryPerMultiProcessor, MaxSharedMemoryPerMultiprocessor);
            // GET_DEVICE_PROP(regsPerMultiprocessor, MAX_REGISTERS_PER_MULTIPROCESSOR);
            // GET_DEVICE_PROP(managedMemory, MANAGED_MEMORY);
            GET_DEVICE_PROP(isMultiGpuBoard, IsMultiGpuBoard);
            // GET_DEVICE_PROP(multiGpuBoardGroupID, MULTI_GPU_BOARD_GROUP_ID);
// #if CUDA_VERSION >= 8000
//             GET_DEVICE_PROP(singleToDoublePrecisionPerfRatio, SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO);
//             GET_DEVICE_PROP(pageableMemoryAccess, PAGEABLE_MEMORY_ACCESS);
//             GET_DEVICE_PROP(concurrentManagedAccess, CONCURRENT_MANAGED_ACCESS);
// #endif
// #if CUDA_VERSION >= 9000
//             GET_DEVICE_PROP(computePreemptionSupported, COMPUTE_PREEMPTION_SUPPORTED);
//             GET_DEVICE_PROP(canUseHostPointerForRegisteredMem, CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM);
//             GET_DEVICE_PROP(cooperativeLaunch, COOPERATIVE_LAUNCH);
//             GET_DEVICE_PROP(cooperativeMultiDeviceLaunch, COOPERATIVE_MULTI_DEVICE_LAUNCH);
//             GET_DEVICE_PROP(sharedMemPerBlockOptin, MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
// #endif
// #if CUDA_VERSION >= 9200
//             GET_DEVICE_PROP(pageableMemoryAccessUsesHostPageTables, PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES);
//             GET_DEVICE_PROP(directManagedMemAccessFromHost, DIRECT_MANAGED_MEM_ACCESS_FROM_HOST);
// #endif
// #if CUDA_VERSION >= 11000
//             GET_DEVICE_PROP(maxBlocksPerMultiProcessor, MAX_BLOCKS_PER_MULTIPROCESSOR);
//             GET_DEVICE_PROP(accessPolicyMaxWindowSize, MAX_ACCESS_POLICY_WINDOW_SIZE);
// #endif
#undef GET_DEVICE_PROP
#endif // REALM_USE_HIP_HIJACK
      	    log_gpu.info() << "GPU #" << i << ": " << info->name << " ("
      	     		   << info->major << '.' << info->minor
      			   << ") " << (info->totalGlobalMem >> 20) << " MB";

            infos.push_back(info);
          }
        }

      	if(infos.empty()) {
      	  log_gpu.warning() << "no HIP-capable GPUs found - module not loaded";
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
      	      CHECK_HIP( hipDeviceCanAccessPeer(&can_access,
      					      (*it1)->device,
      					      (*it2)->device) );
      	      if(can_access) {
            		log_gpu.info() << "p2p access from device " << (*it1)->index
            			       << " to device " << (*it2)->index;
            		(*it1)->peers.insert((*it2)->device);
      	      }
      	    } else {
              // two contexts on the same device can always "peer to peer"
              (*it1)->peers.insert((*it2)->device);
            }
      }

      // give the gpu info we assembled to the module
      m->gpu_info.swap(infos);

      return m;
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void HipModule::initialize(RuntimeImpl *runtime)
    {
      Module::initialize(runtime);

      // if we are using a shared worker, create that next
      if(cfg_use_shared_worker) {
	      shared_worker = new GPUWorker;

	      if(cfg_use_worker_threads)
	        shared_worker->start_background_thread(runtime->core_reservation_set(),
						                                     1 << 20); // hardcoded worker stack size
        else
          shared_worker->add_to_manager(&(runtime->bgwork));
      }

      // decode specific device id list if given
      std::vector<unsigned> fixed_indices;
      if(!cfg_gpu_idxs.empty()) {
        const char *p = cfg_gpu_idxs.c_str();
        while(true) {
          if(!isdigit(*p)) {
            log_gpu.fatal() << "invalid number in hip device list: '" << p << "'";
            abort();
          }
          unsigned v = 0;
          do {
            v = (v * 10) + (*p++ - '0');
          } while(isdigit(*p));
          if(v >= gpu_info.size()) {
            log_gpu.fatal() << "requested hip device id out of range: " << v << " >= " << gpu_info.size();
            abort();
          }
          fixed_indices.push_back(v);
          if(!*p) break;
          if(*p == ',') {
            p++;  // skip comma and parse another integer
          } else {
            log_gpu.fatal() << "invalid separator in hip device list: '" << p << "'";
            abort();
          }
        }
        // if num_gpus was specified, they should match
        if(cfg_num_gpus > 0) {
          if(cfg_num_gpus != fixed_indices.size()) {
            log_gpu.fatal() << "mismatch between '-ll:gpu' and '-ll:gpu_ids'";
            abort();
          }
        } else
          cfg_num_gpus = fixed_indices.size();
        // also disable skip count and skip busy options
        cfg_skip_gpu_count = 0;
        cfg_skip_busy_gpus = false;
      }

      // just use the GPUs in order right now
      gpus.resize(cfg_num_gpus);
      unsigned gpu_count = 0;
      // try to get cfg_num_gpus, working through the list in order
      for(size_t i = cfg_skip_gpu_count;
          (i < gpu_info.size()) && (gpu_count < cfg_num_gpus);
          i++) {
        int idx = (fixed_indices.empty() ? i : fixed_indices[i]);

        // try to create a context and possibly check available memory - in order
      	//  to be compatible with an application's use of the cuda runtime, we
      	//  need this to be the device's "primary context"

      	// set context flags before we create it, but it's ok to be told that
      	//  it's too late
        // {
        //   hipError_t res = hipDevicePrimaryCtxSetFlags(gpu_info[i]->device,
        //               hipDeviceScheduleBlockingSync);
        //   assert((res == hipSuccess) || (res == hipErrorSetOnActiveProcess) || (res == hipErrorContextAlreadyInUse));
        // }
        //
        // hipCtx_t context;
        //         hipError_t res = hipDevicePrimaryCtxRetain(&context,
        //                                                    gpu_info[i]->device);
        hipError_t res = hipSetDevice(gpu_info[idx]->device);
        CHECK_HIP( hipSetDeviceFlags(hipDeviceMapHost | hipDeviceScheduleBlockingSync) );    	
        // a busy GPU might return INVALID_DEVICE or OUT_OF_MEMORY here
      	if((res == hipErrorInvalidDevice) ||
      	   (res == hipErrorOutOfMemory)) {
      	  if(cfg_skip_busy_gpus) {
      	    log_gpu.info() << "GPU " << gpu_info[idx]->device << " appears to be busy (res=" << res << ") - skipping";
      	    continue;
      	  } else {
      	    log_gpu.fatal() << "GPU " << gpu_info[idx]->device << " appears to be in use - use CUDA_VISIBLE_DEVICES, -cuda:skipgpus, or -cuda:skipbusy to select other GPUs";
      	    abort();
      	  }
      	}
      	// any other error is a (unknown) problem
      	CHECK_HIP(res);

      	if(cfg_min_avail_mem > 0) {
      	  size_t total_mem, avail_mem;
      	  CHECK_HIP( hipMemGetInfo(&avail_mem, &total_mem) );
      	  if(avail_mem < cfg_min_avail_mem) {
      	    log_gpu.info() << "GPU " << gpu_info[idx]->device << " does not have enough available memory (" << avail_mem << " < " << cfg_min_avail_mem << ") - skipping";
      	    //CHECK_HIP( hipDevicePrimaryCtxRelease(gpu_info[i]->device) );
      	    continue;
      	  }
      	}
        
        // either create a worker for this GPU or use the shared one
        GPUWorker *worker;
        if(cfg_use_shared_worker) {
          worker = shared_worker;
        } else {
          worker = new GPUWorker;

          if(cfg_use_worker_threads)
            worker->start_background_thread(runtime->core_reservation_set(),
                    1 << 20); // hardcoded worker stack size
                else
                  worker->add_to_manager(&(runtime->bgwork));
        }

        GPU *g = new GPU(this, gpu_info[idx], worker, idx);

        if(!cfg_use_shared_worker)
          dedicated_workers[g] = worker;

        gpus[gpu_count++] = g;
      }
      
      // did we actually get the requested number of GPUs?
      if(gpu_count < cfg_num_gpus) {
	      log_gpu.fatal() << cfg_num_gpus << " GPUs requested, but only " << gpu_count << " available!";
	      assert(false);
      }
    }

    // create any memories provided by this module (default == do nothing)
    //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
    void HipModule::create_memories(RuntimeImpl *runtime)
    {
      Module::create_memories(runtime);

      // each GPU needs its FB memory
      if(cfg_fb_mem_size > 0)
        for(std::vector<GPU *>::iterator it = gpus.begin();
            it != gpus.end();
            it++)
          (*it)->create_fb_memory(runtime, cfg_fb_mem_size, cfg_fb_ib_size);

      if(cfg_use_dynamic_fb)
        for(std::vector<GPU *>::iterator it = gpus.begin();
            it != gpus.end();
            it++)
          (*it)->create_dynamic_fb_memory(runtime, cfg_dynfb_max_size);

      // a single ZC memory for everybody
      if((cfg_zc_mem_size > 0) && !gpus.empty()) {
        char *zcmem_gpu_base;
        // borrow GPU 0's context for the allocation call
        {
          AutoGPUContext agc(gpus[0]);

          hipError_t ret = hipHostMalloc(&zcmem_cpu_base, 
                cfg_zc_mem_size,
                hipHostMallocPortable | hipHostMallocMapped);
          if(ret != hipSuccess) {
            if(ret == hipErrorMemoryAllocation) {
              log_gpu.fatal() << "insufficient device-mappable host memory: "
                  << cfg_zc_mem_size << " bytes needed (from -ll:zsize)";
            } else {
              const char *errstring = "error message not available";
      #if HIP_VERBOSE_ERROR_MSG == 1
              errstring = hipGetErrorName(ret);
      #endif
              log_gpu.fatal() << "unexpected error from cuMemHostAlloc: result=" << ret
                  << " (" << errstring << ")";
            }
            abort();
          }
          CHECK_HIP( hipHostGetDevicePointer((void **)&zcmem_gpu_base,
                      zcmem_cpu_base,
                      0) );
          // right now there are asssumptions in several places that unified addressing keeps
          //  the CPU and GPU addresses the same
          assert(zcmem_cpu_base == (void *)zcmem_gpu_base);
        }

        Memory m = runtime->next_local_memory_id();
        zcmem = new GPUZCMemory(m, zcmem_gpu_base, zcmem_cpu_base, 
                                cfg_zc_mem_size,
                                MemoryImpl::MKIND_ZEROCOPY, Memory::Kind::Z_COPY_MEM);
        runtime->add_memory(zcmem);

        // add the ZC memory as a pinned memory to all GPUs
        for(unsigned i = 0; i < gpus.size(); i++) {
          char *gpuptr;
          hipError_t ret;
          {
            AutoGPUContext agc(gpus[i]);
            ret = hipHostGetDevicePointer((void **)&gpuptr, zcmem_cpu_base, 0);
          }
          if((ret == hipSuccess) && (gpuptr == zcmem_gpu_base)) {
            gpus[i]->pinned_sysmems.insert(zcmem->me);
          } else {
            log_gpu.warning() << "GPU #" << i << " has an unexpected mapping for ZC memory!";
          }
        }
      }

      // allocate intermediate buffers in ZC memory for DMA engine
      if ((cfg_zc_ib_size > 0) && !gpus.empty()) {
        char *zcib_gpu_base;
        {
          AutoGPUContext agc(gpus[0]);
          CHECK_HIP( hipHostMalloc(&zcib_cpu_base,
                                   cfg_zc_ib_size,
                                   hipHostMallocPortable | hipHostMallocMapped) );
          CHECK_HIP( hipHostGetDevicePointer((void **)&zcib_gpu_base,
                                              zcib_cpu_base, 0) );
          // right now there are asssumptions in several places that unified addressing keeps
          //  the CPU and GPU addresses the same
          assert(zcib_cpu_base == (void *)zcib_gpu_base); 
        }
        Memory m = runtime->next_local_ib_memory_id();
        IBMemory* ib_mem;
        ib_mem = new IBMemory(m, cfg_zc_ib_size,
			      MemoryImpl::MKIND_ZEROCOPY, Memory::Z_COPY_MEM,
			      zcib_cpu_base, 0);
        runtime->add_ib_memory(ib_mem);
        // add the ZC memory as a pinned memory to all GPUs
        for (unsigned i = 0; i < gpus.size(); i++) {
          char *gpuptr;
          hipError_t ret;
          {
            AutoGPUContext agc(gpus[i]);
            ret = hipHostGetDevicePointer((void **)&gpuptr, zcib_cpu_base, 0);
          }
          if ((ret == hipSuccess) && (gpuptr == zcib_gpu_base)) {
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
    void HipModule::create_processors(RuntimeImpl *runtime)
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
    void HipModule::create_dma_channels(RuntimeImpl *runtime)
    {
      // before we create dma channels, see how many of the system memory ranges
      //  we can register with HIP
      if(cfg_pin_sysmem && !gpus.empty()) {
        const std::vector<MemoryImpl *>& local_mems = runtime->nodes[Network::my_node_id].memories;
        // <NEW_DMA> also add intermediate buffers into local_mems
        const std::vector<IBMemory *>& local_ib_mems = runtime->nodes[Network::my_node_id].ib_memories;
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
          
          // skip any memory that's over the max size limit for host
          //  registration
          if((cfg_hostreg_limit > 0) &&
              ((*it)->size > cfg_hostreg_limit)) {
            log_gpu.info() << "memory " << (*it)->me
                            << " is larger than hostreg limit ("
                            << (*it)->size << " > " << cfg_hostreg_limit
                            << ") - skipping registration";
            continue;
          }

          void *base = (*it)->get_direct_ptr(0, (*it)->size);
          if(base == 0)
            continue;

          // using GPU 0's context, attempt a portable registration
          hipError_t ret;
          {
            AutoGPUContext agc(gpus[0]);
            ret = hipHostRegister(base, (*it)->size, 
                  hipHostRegisterPortable |
                  hipHostRegisterMapped);
          }
          if(ret != hipSuccess) {
            log_gpu.info() << "failed to register mem " << (*it)->me << " (" << base << " + " << (*it)->size << ") : "
                << ret;
            continue;
          }
          registered_host_ptrs.push_back(base);

          // now go through each GPU and verify that it got a GPU pointer (it may not match the CPU
          //  pointer, but that's ok because we'll never refer to it directly)
          for(unsigned i = 0; i < gpus.size(); i++) {
            char *gpuptr;
            hipError_t ret;
            {
              AutoGPUContext agc(gpus[i]);
              ret = hipHostGetDevicePointer((void **)&gpuptr, base, 0);
            }
            if(ret == hipSuccess) {
              // no test for && ((void *)gpuptr == base)) {
              log_gpu.info() << "memory " << (*it)->me << " successfully registered with GPU " << gpus[i]->proc->me;
              gpus[i]->pinned_sysmems.insert((*it)->me);
            } else {
              log_gpu.warning() << "GPU #" << i << " has no mapping for registered memory (" << (*it)->me << " at " << base << ") !?";
            }
          }
        }
      }

      // ask any ipc-able nodes to share handles with us
      if(cfg_use_hip_ipc) {
        NodeSet ipc_peers = Network::all_peers;

#ifdef REALM_ON_LINUX
        if(!ipc_peers.empty()) {
          log_hipipc.info() << "requesting hip ipc handles from "
                             << ipc_peers.size() << " peers";

          // we'll need a reponse (and ultimately, a release) from each peer
          hipipc_responses_needed.fetch_add(ipc_peers.size());
          hipipc_releases_needed.fetch_add(ipc_peers.size());

          ActiveMessage<HipIpcRequest> amsg(ipc_peers);
          amsg->hostid = gethostid();
          amsg.commit();

          // wait for responses
          {
            AutoLock<> al(hipipc_mutex);
            while(hipipc_responses_needed.load_acquire() > 0)
              hipipc_condvar.wait();
          }
          log_hipipc.info() << "responses complete";
        }
#endif
      }

      // now actually let each GPU make its channels
      for(std::vector<GPU *>::iterator it = gpus.begin();
          it != gpus.end();
          it++)
        (*it)->create_dma_channels(runtime);

      Module::create_dma_channels(runtime);
    }

    // create any code translators provided by the module (default == do nothing)
    void HipModule::create_code_translators(RuntimeImpl *runtime)
    {
      Module::create_code_translators(runtime);
    }

    // if a module has to do cleanup that involves sending messages to other
    //  nodes, this must be done in the pre-detach cleanup
    void HipModule::pre_detach_cleanup(void)
    {
      if(cfg_use_hip_ipc) {
        // release all of our ipc mappings, notify our peers
        NodeSet ipc_peers;

        for(std::vector<GPU *>::iterator it = gpus.begin();
            it != gpus.end();
            ++it) {
          if(!(*it)->hipipc_mappings.empty()) {
            AutoGPUContext agc(*it);

            for(std::vector<GPU::HipIpcMapping>::iterator it2 = (*it)->hipipc_mappings.begin();
                it2 != (*it)->hipipc_mappings.end();
                ++it2) {
              ipc_peers.add(it2->owner);
              CHECK_HIP( hipIpcCloseMemHandle((void*)(it2->local_base)) );
            }
          }
        }

        if(!ipc_peers.empty()) {
          ActiveMessage<HipIpcRelease> amsg(ipc_peers);
          amsg.commit();
        }

        // now wait for similar notifications from any peers we gave mappings
        //  to before we start freeing the underlying allocations
        {
          AutoLock<> al(hipipc_mutex);
          while(hipipc_releases_needed.load_acquire() > 0)
            hipipc_condvar.wait();
        }
        log_hipipc.info() << "releases complete";
      }
    }

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void HipModule::cleanup(void)
    {
      // clean up worker(s)
      if(shared_worker) {
#ifdef DEBUG_REALM
	      shared_worker->shutdown_work_item();
#endif
      	if(cfg_use_worker_threads)
      	  shared_worker->shutdown_background_thread();

        delete shared_worker;
        shared_worker = 0;
      }
      for(std::map<GPU *, GPUWorker *>::iterator it = dedicated_workers.begin();
          it != dedicated_workers.end();
          it++) {
        GPUWorker *worker = it->second;

#ifdef DEBUG_REALM
        worker->shutdown_work_item();
#endif
        if(cfg_use_worker_threads)
          worker->shutdown_background_thread();

        delete worker;
      }
      dedicated_workers.clear();

      // use GPU 0's context to free ZC memory (if any)
      if(zcmem_cpu_base) {
        assert(!gpus.empty());
        AutoGPUContext agc(gpus[0]);
        CHECK_HIP( hipHostFree(zcmem_cpu_base) );
      }

      if(zcib_cpu_base) {
        assert(!gpus.empty());
        AutoGPUContext agc(gpus[0]);
        CHECK_HIP( hipHostFree(zcib_cpu_base) );
      }

      // also unregister any host memory at this time
      if(!registered_host_ptrs.empty()) {
        AutoGPUContext agc(gpus[0]);
        for(std::vector<void *>::const_iterator it = registered_host_ptrs.begin();
            it != registered_host_ptrs.end();
            ++it)
          CHECK_HIP( hipHostUnregister(*it) );
        registered_host_ptrs.clear();
      }

      for(std::vector<GPU *>::iterator it = gpus.begin();
          it != gpus.end();
          it++) {
#ifdef REALM_USE_HIP_HIJACK
        GlobalRegistrations::remove_gpu_context(*it);
#endif
        delete *it;
      }
      gpus.clear();
      
      Module::cleanup();
    }


#ifdef REALM_USE_HIP_HIJACK
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

      AutoLock<> al(g.mutex);

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

      AutoLock<> al(g.mutex);

      assert(g.active_gpus.count(gpu) > 0);
      g.active_gpus.erase(gpu);
    }

    // called by __cuda(un)RegisterFatBinary
    /*static*/ void GlobalRegistrations::register_fat_binary(FatBin *fatbin)
    {
      GlobalRegistrations& g = get_global_registrations();

      AutoLock<> al(g.mutex);

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

      AutoLock<> al(g.mutex);

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

      AutoLock<> al(g.mutex);

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

      AutoLock<> al(g.mutex);

      // add the function to the list and tell any gpus we know
      g.functions.push_back(func);

      for(std::set<GPU *>::iterator it = g.active_gpus.begin();
          it != g.active_gpus.end();
          it++)
        (*it)->register_function(func);
    }
#endif

    // active messages for establishing cuda ipc mappings

    struct HipIpcResponseEntry {
      Memory mem;
      uintptr_t base_ptr;
      hipIpcMemHandle_t handle;
    };


    ////////////////////////////////////////////////////////////////////////
    //
    // struct HipIpcRequest

    /*static*/ void HipIpcRequest::handle_message(NodeID sender,
                                                  const HipIpcRequest& args,
                                                  const void *data,
                                                  size_t datalen)
    {
      log_hipipc.info() << "request from node " << sender;
      assert(hip_module_singleton);

      std::vector<HipIpcResponseEntry> exported;

      // only export if we've got ipc enabled locally
      bool do_export = false;
      if(hip_module_singleton->cfg_use_hip_ipc) {
#ifdef REALM_ON_LINUX
        // host id has to match as well
        long hostid = gethostid();
        if(hostid == args.hostid)
          do_export = true;
        else
          log_hipipc.info() << "hostid mismatch - us=" << hostid << " them=" << args.hostid;
#endif
      }

      if(do_export) {
        for(std::vector<GPU *>::iterator it = hip_module_singleton->gpus.begin();
            it != hip_module_singleton->gpus.end();
            ++it) {
          HipIpcResponseEntry entry;
          {
            AutoGPUContext agc(*it);

            hipError_t ret = hipIpcGetMemHandle(&entry.handle,
                                                (*it)->fbmem_base);
            log_hipipc.info() << "getmem handle " << std::hex << (*it)->fbmem_base << std::dec << " -> " << ret;
            if(ret == hipSuccess) {
              entry.mem = (*it)->fbmem->me;
              entry.base_ptr = reinterpret_cast<uintptr_t>((*it)->fbmem_base);
              exported.push_back(entry);
            }
          }
        }
      }

      // if we're not exporting anything to this requestor, don't wait for
      //  a release either (having the count hit 0 here is a weird corner
      //  case)
      if(exported.empty()) {
        AutoLock<> al(hip_module_singleton->hipipc_mutex);
        int prev = hip_module_singleton->hipipc_releases_needed.fetch_sub(1);
        if(prev == 1)
          hip_module_singleton->hipipc_condvar.broadcast();
      }

      size_t bytes = exported.size() * sizeof(HipIpcResponseEntry);
      ActiveMessage<HipIpcResponse> amsg(sender, bytes);
      amsg->count = exported.size();
      amsg.add_payload(exported.data(), bytes);
      amsg.commit();
    }

    ActiveMessageHandlerReg<HipIpcRequest> hip_ipc_request_handler;


    ////////////////////////////////////////////////////////////////////////
    //
    // struct HipIpcResponse

    /*static*/ void HipIpcResponse::handle_message(NodeID sender,
                                                   const HipIpcResponse& args,
                                                   const void *data,
                                                   size_t datalen)
    {
      assert(hip_module_singleton);

      assert(datalen == (args.count * sizeof(HipIpcResponseEntry)));
      const HipIpcResponseEntry *entries = static_cast<const HipIpcResponseEntry *>(data);

      if(args.count) {
        for(std::vector<GPU *>::iterator it = hip_module_singleton->gpus.begin();
            it != hip_module_singleton->gpus.end();
            ++it) {
          {
            AutoGPUContext agc(*it);

            // attempt to import each entry
            for(unsigned i = 0; i < args.count; i++) {
              void* dptr;
              hipError_t ret = hipIpcOpenMemHandle(&dptr,
                                                   entries[i].handle,
                                                   hipIpcMemLazyEnablePeerAccess);
              log_hipipc.info() << "open result " << entries[i].mem
                                 << " orig=" << std::hex << entries[i].base_ptr
                                 << " local=" << dptr << std::dec
                                 << " ret=" << ret;

              if(ret != hipSuccess)
                continue; // complain louder?

              // take the cudaipc mutex to actually add the mapping
              GPU::HipIpcMapping mapping;
              mapping.owner = sender;
              mapping.mem = entries[i].mem;
              mapping.local_base = reinterpret_cast<uintptr_t>(dptr);
              mapping.address_offset = reinterpret_cast<uintptr_t>(entries[i].base_ptr) - reinterpret_cast<uintptr_t>(dptr);
              {
                AutoLock<> al(hip_module_singleton->hipipc_mutex);
                (*it)->hipipc_mappings.push_back(mapping);

                // do we have a stream for this target?
                if((*it)->hipipc_streams.count(sender) == 0)
                  (*it)->hipipc_streams[sender] = new GPUStream(*it,
                                                                 (*it)->worker);
              }
            }
          }
        }
      }

      // decrement the number of responses needed and wake the requestor if
      //  we're done
      {
        AutoLock<> al(hip_module_singleton->hipipc_mutex);
        int prev = hip_module_singleton->hipipc_responses_needed.fetch_sub(1);
        if(prev == 1)
          hip_module_singleton->hipipc_condvar.broadcast();
      }
    }

    ActiveMessageHandlerReg<HipIpcResponse> hip_ipc_response_handler;


    ////////////////////////////////////////////////////////////////////////
    //
    // struct HipIpcRelease

    /*static*/ void HipIpcRelease::handle_message(NodeID sender,
                                                  const HipIpcRelease& args,
                                                  const void *data,
                                                  size_t datalen)
    {
      assert(hip_module_singleton);

      // no actual work to do - we're just waiting until all of our peers
      //  have released ipc mappings before we continue
      {
        AutoLock<> al(hip_module_singleton->hipipc_mutex);
        int prev = hip_module_singleton->hipipc_releases_needed.fetch_sub(1);
        if(prev == 1)
          hip_module_singleton->hipipc_condvar.broadcast();
      }
    }

    ActiveMessageHandlerReg<HipIpcRelease> hip_ipc_release_handler;
    
#ifndef REALM_USE_HIP_HIJACK
    extern "C" {
      REALM_PUBLIC_API
      hipStream_t hipGetTaskStream()
      {
        return 0;
      }
    }; // extern "C"

#endif    

  }; // namespace Hip
}; // namespace Realm

