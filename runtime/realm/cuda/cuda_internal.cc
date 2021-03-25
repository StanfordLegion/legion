/* Copyright 2021 Stanford University, NVIDIA Corporation
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

#include "realm/cuda/cuda_internal.h"

#ifndef REALM_USE_CUDART_HIJACK
// we do nearly everything with the driver API, but if we're not pretending
//  to be the cuda runtime, we need to be able to talk to the real runtime
//  for a few things
#include <cuda_runtime.h>
#endif

namespace Realm {

  extern Logger log_xd;

  namespace Cuda {

    extern Logger log_stream;


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUXferDes

      GPUXferDes::GPUXferDes(uintptr_t _dma_op, Channel *_channel,
			           NodeID _launch_node, XferDesID _guid,
				   const std::vector<XferDesPortInfo>& inputs_info,
				   const std::vector<XferDesPortInfo>& outputs_info,
				   int _priority)
	: XferDes(_dma_op, _channel, _launch_node, _guid,
		  inputs_info, outputs_info,
		  _priority, 0, 0)
      {
	if((inputs_info.size() >= 1) &&
	   (input_ports[0].mem->kind == MemoryImpl::MKIND_GPUFB)) {
	  // all input ports should agree on which gpu they target
	  src_gpu = ((GPUFBMemory*)(input_ports[0].mem))->gpu;
	  for(size_t i = 1; i < input_ports.size(); i++) {
	    // exception: control and indirect ports should be readable from cpu
	    if((int(i) == input_control.control_port_idx) ||
	       (int(i) == output_control.control_port_idx) ||
	       input_ports[i].is_indirect_port) {
	      assert((input_ports[i].mem->kind == MemoryImpl::MKIND_SYSMEM) ||
		     (input_ports[i].mem->kind == MemoryImpl::MKIND_ZEROCOPY) ||
		     (input_ports[i].mem->kind == MemoryImpl::MKIND_MANAGED));
	      continue;
	    }
	    assert(input_ports[i].mem == input_ports[0].mem);
	  }
	} else
	  src_gpu = 0;

	if((outputs_info.size() >= 1) &&
	   (output_ports[0].mem->kind == MemoryImpl::MKIND_GPUFB)) {
	  // all output ports should agree on which gpu they target
	  dst_gpu = ((GPUFBMemory*)(output_ports[0].mem))->gpu;
	  for(size_t i = 1; i < output_ports.size(); i++)
	    assert(output_ports[i].mem == output_ports[0].mem);
	} else
	  dst_gpu = 0;

	// if we're doing a multi-hop copy, we'll dial down the request
	//  sizes to improve pipelining
	bool multihop_copy = false;
	for(size_t i = 1; i < input_ports.size(); i++)
	  if(input_ports[i].peer_guid != XFERDES_NO_GUID)
	    multihop_copy = true;
	for(size_t i = 1; i < output_ports.size(); i++)
	  if(output_ports[i].peer_guid != XFERDES_NO_GUID)
	    multihop_copy = true;

	if(src_gpu != 0) {
	  if(dst_gpu != 0) {
	    if(src_gpu == dst_gpu) {
	      kind = XFER_GPU_IN_FB;
	      // ignore max_req_size value passed in - it's probably too small
	      max_req_size = 1 << 30;
	    } else {
	      kind = XFER_GPU_PEER_FB;
	      // ignore max_req_size value passed in - it's probably too small
	      max_req_size = 256 << 20;
	    }
	  } else {
	    kind = XFER_GPU_FROM_FB;
	    if(multihop_copy)
	      max_req_size = 4 << 20;
	  }
	} else {
	  if(dst_gpu != 0) {
	    kind = XFER_GPU_TO_FB;
	    if(multihop_copy)
	      max_req_size = 4 << 20;
	  } else {
	    assert(0);
	  }
	}

	const int max_nr = 10; // FIXME
        for (int i = 0; i < max_nr; i++) {
          GPURequest* gpu_req = new GPURequest;
          gpu_req->xd = this;
	  gpu_req->event.req = gpu_req;
          available_reqs.push(gpu_req);
        }
      }
	
      long GPUXferDes::get_requests(Request** requests, long nr)
      {
        GPURequest** reqs = (GPURequest**) requests;
	// TODO: add support for 3D CUDA copies (just 1D and 2D for now)
	unsigned flags = (TransferIterator::LINES_OK |
			  TransferIterator::PLANES_OK);
        long new_nr = default_get_requests(requests, nr, flags);
        for (long i = 0; i < new_nr; i++) {
          switch (kind) {
            case XFER_GPU_TO_FB:
            {
              //reqs[i]->src_base = src_buf_base + reqs[i]->src_off;
	      reqs[i]->src_base = input_ports[reqs[i]->src_port_idx].mem->get_direct_ptr(reqs[i]->src_off,
											 reqs[i]->nbytes);
	      assert(reqs[i]->src_base != 0);
              //reqs[i]->dst_gpu_off = /*dst_buf.alloc_offset +*/ reqs[i]->dst_off;
              break;
            }
            case XFER_GPU_FROM_FB:
            {
              //reqs[i]->src_gpu_off = /*src_buf.alloc_offset +*/ reqs[i]->src_off;
              //reqs[i]->dst_base = dst_buf_base + reqs[i]->dst_off;
	      reqs[i]->dst_base = output_ports[reqs[i]->dst_port_idx].mem->get_direct_ptr(reqs[i]->dst_off,
											  reqs[i]->nbytes);
	      assert(reqs[i]->dst_base != 0);
              break;
            }
            case XFER_GPU_IN_FB:
            {
              //reqs[i]->src_gpu_off = /*src_buf.alloc_offset*/ + reqs[i]->src_off;
              //reqs[i]->dst_gpu_off = /*dst_buf.alloc_offset*/ + reqs[i]->dst_off;
              break;
            }
            case XFER_GPU_PEER_FB:
            {
              //reqs[i]->src_gpu_off = /*src_buf.alloc_offset +*/ reqs[i]->src_off;
              //reqs[i]->dst_gpu_off = /*dst_buf.alloc_offset +*/ reqs[i]->dst_off;
              // also need to set dst_gpu for peer xfer
              reqs[i]->dst_gpu = dst_gpu;
              break;
            }
            default:
              assert(0);
          }
        }
        return new_nr;
      }

      bool GPUXferDes::progress_xd(GPUChannel *channel,
				   TimeLimit work_until)
      {
	Request *rq;
	bool did_work = false;
	do {
	  long count = get_requests(&rq, 1);
	  if(count > 0) {
	    channel->submit(&rq, count);
	    did_work = true;
	  } else
	    break;
	} while(!work_until.is_expired());

	return did_work;
      }

      void GPUXferDes::notify_request_read_done(Request* req)
      {
        default_notify_request_read_done(req);
      }

      void GPUXferDes::notify_request_write_done(Request* req)
      {
        default_notify_request_write_done(req);
      }

      void GPUXferDes::flush()
      {
      }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUChannel

      GPUChannel::GPUChannel(GPU* _src_gpu, XferDesKind _kind,
			     BackgroundWorkManager *bgwork)
	: SingleXDQChannel<GPUChannel,GPUXferDes>(bgwork,
						  _kind,
						  stringbuilder() << "cuda channel (gpu=" << _src_gpu->info->index << " kind=" << (int)_kind << ")")
      {
        src_gpu = _src_gpu;

	Memory fbm = src_gpu->fbmem->me;

	switch(_kind) {
	case XFER_GPU_TO_FB:
	  {
	    unsigned bw = 0; // TODO
	    unsigned latency = 0;
	    for(std::set<Memory>::const_iterator it = src_gpu->pinned_sysmems.begin();
		it != src_gpu->pinned_sysmems.end();
		++it)
	      add_path(*it, fbm, bw, latency, false, false,
		       XFER_GPU_TO_FB);

	    for(std::set<Memory>::const_iterator it = src_gpu->managed_mems.begin();
		it != src_gpu->managed_mems.end();
		++it)
	      add_path(*it, fbm, bw, latency, false, false,
		       XFER_GPU_TO_FB);

	    break;
	  }

	case XFER_GPU_FROM_FB:
	  {
	    unsigned bw = 0; // TODO
	    unsigned latency = 0;
	    for(std::set<Memory>::const_iterator it = src_gpu->pinned_sysmems.begin();
		it != src_gpu->pinned_sysmems.end();
		++it)
	      add_path(fbm, *it, bw, latency, false, false,
		       XFER_GPU_FROM_FB);

	    for(std::set<Memory>::const_iterator it = src_gpu->managed_mems.begin();
		it != src_gpu->managed_mems.end();
		++it)
	      add_path(fbm, *it, bw, latency, false, false,
		       XFER_GPU_FROM_FB);

	    break;
	  }

	case XFER_GPU_IN_FB:
	  {
	    // self-path
	    unsigned bw = 0; // TODO
	    unsigned latency = 0;
	    add_path(fbm, fbm, bw, latency, false, false,
		     XFER_GPU_IN_FB);

	    break;
	  }

	case XFER_GPU_PEER_FB:
	  {
	    // just do paths to peers - they'll do the other side
	    unsigned bw = 0; // TODO
	    unsigned latency = 0;
	    for(std::set<Memory>::const_iterator it = src_gpu->peer_fbs.begin();
		it != src_gpu->peer_fbs.end();
		++it)
	      add_path(fbm, *it, bw, latency, false, false,
		       XFER_GPU_PEER_FB);

	    break;
	  }

	default:
	  assert(0);
	}
      }

      GPUChannel::~GPUChannel()
      {
      }

      XferDes *GPUChannel::create_xfer_des(uintptr_t dma_op,
					   NodeID launch_node,
					   XferDesID guid,
					   const std::vector<XferDesPortInfo>& inputs_info,
					   const std::vector<XferDesPortInfo>& outputs_info,
					   int priority,
					   XferDesRedopInfo redop_info,
					   const void *fill_data, size_t fill_size)
      {
	assert(redop_info.id == 0);
	assert(fill_size == 0);
	return new GPUXferDes(dma_op, this, launch_node, guid,
			      inputs_info, outputs_info,
			      priority);
      }

      long GPUChannel::submit(Request** requests, long nr)
      {
        for (long i = 0; i < nr; i++) {
          GPURequest* req = (GPURequest*) requests[i];
	  // no serdez support
	  assert(req->xd->input_ports[req->src_port_idx].serdez_op == 0);
	  assert(req->xd->output_ports[req->dst_port_idx].serdez_op == 0);

	  // empty transfers don't need to bounce off the GPU
	  if(req->nbytes == 0) {
	    req->xd->notify_request_read_done(req);
	    req->xd->notify_request_write_done(req);
	    continue;
	  }

	  switch(req->dim) {
	    case Request::DIM_1D: {
	      switch (kind) {
                case XFER_GPU_TO_FB:
		  src_gpu->copy_to_fb(req->dst_off, req->src_base,
				      req->nbytes, &req->event);
		  break;
                case XFER_GPU_FROM_FB:
		  src_gpu->copy_from_fb(req->dst_base, req->src_off,
					req->nbytes, &req->event);
		  break;
                case XFER_GPU_IN_FB:
		  src_gpu->copy_within_fb(req->dst_off, req->src_off,
					  req->nbytes, &req->event);
		  break;
                case XFER_GPU_PEER_FB:
		  src_gpu->copy_to_peer(req->dst_gpu, req->dst_off,
					req->src_off, req->nbytes,
					&req->event);
		  break;
                default:
		  assert(0);
	      }
	      break;
	    }

	    case Request::DIM_2D: {
              switch (kind) {
	        case XFER_GPU_TO_FB:
		  src_gpu->copy_to_fb_2d(req->dst_off, req->src_base,
					 req->dst_str, req->src_str,
					 req->nbytes, req->nlines, &req->event);
		  break;
	        case XFER_GPU_FROM_FB:
		  src_gpu->copy_from_fb_2d(req->dst_base, req->src_off,
					   req->dst_str, req->src_str,
					   req->nbytes, req->nlines,
					   &req->event);
		  break;
                case XFER_GPU_IN_FB:
		  src_gpu->copy_within_fb_2d(req->dst_off, req->src_off,
					     req->dst_str, req->src_str,
					     req->nbytes, req->nlines,
					     &req->event);
		  break;
                case XFER_GPU_PEER_FB:
		  src_gpu->copy_to_peer_2d(req->dst_gpu, req->dst_off,
					   req->src_off, req->dst_str,
					   req->src_str, req->nbytes,
					   req->nlines, &req->event);
		  break;
                default:
		  assert(0);
	      }
	      break;
	    }

	    case Request::DIM_3D: {
              switch (kind) {
	        case XFER_GPU_TO_FB:
		  src_gpu->copy_to_fb_3d(req->dst_off, req->src_base,
					 req->dst_str, req->src_str,
					 req->dst_pstr, req->src_pstr,
					 req->nbytes, req->nlines, req->nplanes,
					 &req->event);
		  break;
	        case XFER_GPU_FROM_FB:
		  src_gpu->copy_from_fb_3d(req->dst_base, req->src_off,
					   req->dst_str, req->src_str,
					   req->dst_pstr, req->src_pstr,
					   req->nbytes, req->nlines, req->nplanes,
					   &req->event);
		  break;
                case XFER_GPU_IN_FB:
		  src_gpu->copy_within_fb_3d(req->dst_off, req->src_off,
					     req->dst_str, req->src_str,
					     req->dst_pstr, req->src_pstr,
					     req->nbytes, req->nlines, req->nplanes,
					     &req->event);
		  break;
                case XFER_GPU_PEER_FB:
		  src_gpu->copy_to_peer_3d(req->dst_gpu,
					   req->dst_off, req->src_off,
					   req->dst_str, req->src_str,
					   req->dst_pstr, req->src_pstr,
					   req->nbytes, req->nlines, req->nplanes,
					   &req->event);
		  break;
                default:
		  assert(0);
	      }
	      break;
	    }

	    default:
	      assert(0);
	  }

          //pending_copies.push_back(req);
        }
        return nr;
      }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUCompletionEvent

      void GPUCompletionEvent::request_completed(void)
      {
	req->xd->notify_request_read_done(req);
	req->xd->notify_request_write_done(req);
      }

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUTransfercompletion

    GPUTransferCompletion::GPUTransferCompletion(XferDes *_xd,
                                                 int _read_port_idx,
                                                 size_t _read_offset,
                                                 size_t _read_size,
                                                 int _write_port_idx,
                                                 size_t _write_offset,
                                                 size_t _write_size)
      : xd(_xd)
      , read_port_idx(_read_port_idx)
      , read_offset(_read_offset)
      , read_size(_read_size)
      , write_port_idx(_write_port_idx)
      , write_offset(_write_offset)
      , write_size(_write_size)
    {}

    void GPUTransferCompletion::request_completed(void)
    {
      if(read_port_idx >= 0)
        xd->update_bytes_read(read_port_idx, read_offset, read_size);
      if(write_port_idx >= 0)
        xd->update_bytes_write(write_port_idx, write_offset, write_size);
      xd->remove_reference();
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUfillXferDes

    GPUfillXferDes::GPUfillXferDes(uintptr_t _dma_op, Channel *_channel,
                                   NodeID _launch_node, XferDesID _guid,
                                   const std::vector<XferDesPortInfo>& inputs_info,
                                   const std::vector<XferDesPortInfo>& outputs_info,
                                   int _priority,
                                   const void *_fill_data, size_t _fill_size)
      : XferDes(_dma_op, _channel, _launch_node, _guid,
                inputs_info, outputs_info,
                _priority, _fill_data, _fill_size)
    {
      kind = XFER_GPU_IN_FB;

      // no direct input data for us
      assert(input_control.control_port_idx == -1);
      input_control.current_io_port = -1;

      // cuda memsets are ideally 8/16/32 bits, so try to _reduce_ the fill
      //  size if there's duplication
      if((fill_size > 1) && (memcmp(fill_data,
                                    static_cast<char *>(fill_data) + 1,
                                    fill_size - 1) == 0))
        reduced_fill_size = 1;  // can use memset8
      else if((fill_size > 2) && ((fill_size >> 1) == 0) &&
              (memcmp(fill_data,
                      static_cast<char *>(fill_data) + 2,
                      fill_size - 2) == 0))
        reduced_fill_size = 2;  // can use memset16
      else if((fill_size > 4) && ((fill_size >> 2) == 0) &&
              (memcmp(fill_data,
                      static_cast<char *>(fill_data) + 4,
                      fill_size - 4) == 0))
        reduced_fill_size = 4;  // can use memset32
      else
        reduced_fill_size = fill_size; // will have to do it in pieces
    }

    long GPUfillXferDes::get_requests(Request** requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }

    bool GPUfillXferDes::progress_xd(GPUfillChannel *channel,
                                     TimeLimit work_until)
    {
      bool did_work = false;
      ReadSequenceCache rseqcache(this, 2 << 20);
      ReadSequenceCache wseqcache(this, 2 << 20);

      while(true) {
        size_t min_xfer_size = 4096;  // TODO: make controllable
        size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
        if(max_bytes == 0)
          break;

        XferPort *out_port = 0;
        size_t out_span_start = 0;
        if(output_control.current_io_port >= 0) {
          out_port = &output_ports[output_control.current_io_port];
          out_span_start = out_port->local_bytes_total;
        }

        bool done = false;

        size_t total_bytes = 0;
        if(out_port != 0) {
          // input and output both exist - transfer what we can
          log_xd.info() << "gpufill chunk: min=" << min_xfer_size
                        << " max=" << max_bytes;

          uintptr_t out_base = reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

          AutoGPUContext agc(channel->gpu);
          GPUStream *stream = channel->gpu->device_to_device_stream;

          while(total_bytes < max_bytes) {
            AddressListCursor& out_alc = out_port->addrcursor;

            uintptr_t out_offset = out_alc.get_offset();

            // the reported dim is reduced for partially consumed address
            //  ranges - whatever we get can be assumed to be regular
            int out_dim = out_alc.get_dim();

            // fast paths for 8/16/32 bit memsets exist for 1-D and 2-D
            switch(reduced_fill_size) {
            case 1: {
              // memset8
              if(out_dim == 1) {
                size_t bytes = out_alc.remaining(0);
                CHECK_CU( cuMemsetD8Async(CUdeviceptr(out_base + out_offset),
                                          *reinterpret_cast<const uint8_t *>(fill_data),
                                          bytes,
                                          stream->get_stream()) );
                out_alc.advance(0, bytes);
                total_bytes += bytes;
              } else {
                size_t bytes = out_alc.remaining(0);
                size_t lines = out_alc.remaining(1);
                CHECK_CU( cuMemsetD2D8Async(CUdeviceptr(out_base + out_offset),
                                            out_alc.get_stride(1),
                                            *reinterpret_cast<const uint8_t *>(fill_data),
                                            bytes, lines,
                                            stream->get_stream()) );
                out_alc.advance(1, lines);
                total_bytes += bytes * lines;
              }
              break;
            }

            case 2: {
              // memset16
              if(out_dim == 1) {
                size_t bytes = out_alc.remaining(0);
#ifdef DEBUG_REALM
                assert((bytes & 1) == 0);
#endif
                CHECK_CU( cuMemsetD16Async(CUdeviceptr(out_base + out_offset),
                                           *reinterpret_cast<const uint16_t *>(fill_data),
                                           bytes >> 1,
                                           stream->get_stream()) );
                out_alc.advance(0, bytes);
                total_bytes += bytes;
              } else {
                size_t bytes = out_alc.remaining(0);
                size_t lines = out_alc.remaining(1);
#ifdef DEBUG_REALM
                assert((bytes & 1) == 0);
                assert((out_alc.get_stride(1) & 1) == 0);
#endif
                CHECK_CU( cuMemsetD2D16Async(CUdeviceptr(out_base + out_offset),
                                             out_alc.get_stride(1),
                                             *reinterpret_cast<const uint16_t *>(fill_data),
                                             bytes >> 1, lines,
                                             stream->get_stream()) );
                out_alc.advance(1, lines);
                total_bytes += bytes * lines;
              }
              break;
            }

            case 4: {
              // memset32
              if(out_dim == 1) {
                size_t bytes = out_alc.remaining(0);
#ifdef DEBUG_REALM
                assert((bytes & 3) == 0);
#endif
                CHECK_CU( cuMemsetD32Async(CUdeviceptr(out_base + out_offset),
                                           *reinterpret_cast<const uint32_t *>(fill_data),
                                           bytes >> 2,
                                           stream->get_stream()) );
                out_alc.advance(0, bytes);
                total_bytes += bytes;
              } else {
                size_t bytes = out_alc.remaining(0);
                size_t lines = out_alc.remaining(1);
#ifdef DEBUG_REALM
                assert((bytes & 3) == 0);
                assert((out_alc.get_stride(1) & 3) == 0);
#endif
                CHECK_CU( cuMemsetD2D32Async(CUdeviceptr(out_base + out_offset),
                                             out_alc.get_stride(1),
                                             *reinterpret_cast<const uint32_t *>(fill_data),
                                             bytes >> 2, lines,
                                             stream->get_stream()) );
                out_alc.advance(1, lines);
                total_bytes += bytes * lines;
              }
              break;
            }

            default: {
              // more general approach - use strided 2d copies to fill the first
              //  line, and then we can use logarithmic doublings to deal with
              //  multiple lines and/or planes
              size_t bytes = out_alc.remaining(0);
              size_t elems = bytes / reduced_fill_size;
#ifdef DEBUG_REALM
              assert((bytes % reduced_fill_size) == 0);
#endif
              size_t partial_bytes = 0;
              if((reduced_fill_size & 3) == 0) {
                // 32-bit partial fills allowed
                while(partial_bytes <= (reduced_fill_size - 4)) {
                  CHECK_CU( cuMemsetD2D32Async(CUdeviceptr(out_base + out_offset + partial_bytes),
                                               reduced_fill_size,
                                               reinterpret_cast<const uint32_t *>(fill_data)[partial_bytes >> 2],
                                               1 /*"width"*/, elems /*"height"*/,
                                               stream->get_stream()) );
                  partial_bytes += 4;
                }
              }
              if((reduced_fill_size & 1) == 0) {
                // 16-bit partial fills allowed
                while(partial_bytes <= (reduced_fill_size - 2)) {
                  CHECK_CU( cuMemsetD2D16Async(CUdeviceptr(out_base + out_offset + partial_bytes),
                                               reduced_fill_size,
                                               reinterpret_cast<const uint16_t *>(fill_data)[partial_bytes >> 1],
                                               1 /*"width"*/, elems /*"height"*/,
                                               stream->get_stream()) );
                  partial_bytes += 2;
                }
              }
              // leftover or unaligned bytes are done 8 bits at a time
              while(partial_bytes < reduced_fill_size) {
                CHECK_CU( cuMemsetD2D16Async(CUdeviceptr(out_base + out_offset + partial_bytes),
                                             reduced_fill_size,
                                             reinterpret_cast<const uint16_t *>(fill_data)[partial_bytes],
                                             1 /*"width"*/, elems /*"height"*/,
                                             stream->get_stream()) );
                partial_bytes += 1;
              }

              if(out_dim == 1) {
                // all done
                out_alc.advance(0, bytes);
                total_bytes += bytes;
              } else {
                size_t lines = out_alc.remaining(1);
                size_t lstride = out_alc.get_stride(1);

                CUDA_MEMCPY2D copy2d;
                copy2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                copy2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                copy2d.srcDevice = CUdeviceptr(out_base + out_offset);
                copy2d.srcHost = 0;
                copy2d.srcPitch = lstride;
                copy2d.srcY = 0;
                copy2d.srcXInBytes = 0;
                copy2d.dstHost = 0;
                copy2d.dstPitch = lstride;
                copy2d.dstY = 0;
                copy2d.dstXInBytes = 0;
                copy2d.WidthInBytes = bytes;

                size_t lines_done = 1;  // first line already valid
                while(lines_done < lines) {
                  size_t todo = std::min(lines_done, lines - lines_done);
                  copy2d.dstDevice = CUdeviceptr(out_base + out_offset +
                                                 (lines_done * lstride));
                  copy2d.Height = todo;
                  CHECK_CU( cuMemcpy2DAsync(&copy2d, stream->get_stream()) );
                  lines_done += todo;
                }

                if(out_dim == 2) {
                  out_alc.advance(1, lines);
                  total_bytes += bytes * lines;
                } else {
                  size_t planes = out_alc.remaining(2);
                  size_t pstride = out_alc.get_stride(2);

                  // logarithmic version requires that pstride be a multiple of
                  //  lstride
                  if((pstride % lstride) == 0) {
                    CUDA_MEMCPY3D copy3d;

                    copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                    copy3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                    copy3d.srcDevice = CUdeviceptr(out_base + out_offset);
                    copy3d.srcHost = 0;
                    copy3d.srcPitch = lstride;
                    copy3d.srcHeight = pstride / lstride;
                    copy3d.srcY = 0;
                    copy3d.srcZ = 0;
                    copy3d.srcXInBytes = 0;
                    copy3d.srcLOD = 0;
                    copy3d.dstHost = 0;
                    copy3d.dstPitch = lstride;
                    copy3d.dstHeight = pstride / lstride;
                    copy3d.dstY = 0;
                    copy3d.dstZ = 0;
                    copy3d.dstXInBytes = 0;
                    copy3d.dstLOD = 0;
                    copy3d.WidthInBytes = bytes;
                    copy3d.Height = lines;

                    size_t planes_done = 1;  // first plane already valid
                    while(planes_done < planes) {
                      size_t todo = std::min(planes_done, planes - planes_done);
                      copy3d.dstDevice = CUdeviceptr(out_base + out_offset +
                                                     (planes_done * pstride));
                      copy3d.Depth = todo;
                      CHECK_CU( cuMemcpy3DAsync(&copy3d, stream->get_stream()) );
                      planes_done += todo;
                    }

                    out_alc.advance(2, planes);
                    total_bytes += bytes * lines * planes;
                  } else {
                    // plane-at-a-time fallback - can reuse most of copy2d
                    //  setup above
                    copy2d.Height = lines;

                    for(size_t p = 1; p < planes; p++) {
                      copy2d.dstDevice = CUdeviceptr(out_base + out_offset +
                                                     (p * pstride));
                      CHECK_CU( cuMemcpy2DAsync(&copy2d, stream->get_stream()) );
                    }
                  }
                }
              }
              break;
            }
            }

            // stop if it's been too long, but make sure we do at least the
            //  minimum number of bytes
            if((total_bytes >= min_xfer_size) && work_until.is_expired()) break;
          }

          // however many fills/copies we submitted, put in a single fence that
          //  will tell us that they're all done
          add_reference(); // released by transfer completion
          stream->add_notification(new GPUTransferCompletion(this,
                                                             -1, 0, 0,
                                                             output_control.current_io_port,
                                                             out_span_start,
                                                             total_bytes));
	  out_span_start += total_bytes;

	  done = record_address_consumption(total_bytes, total_bytes);
        }

        did_work = true;

        output_control.remaining_count -= total_bytes;
        if(output_control.control_port_idx >= 0)
          done = ((output_control.remaining_count == 0) &&
                  output_control.eos_received);

        if(done)
          iteration_completed.store_release(true);

        if(done || work_until.is_expired())
          break;
      }

      rseqcache.flush();

      return did_work;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUfillChannel

    GPUfillChannel::GPUfillChannel(GPU *_gpu, BackgroundWorkManager *bgwork)
      : SingleXDQChannel<GPUfillChannel,GPUfillXferDes>(bgwork,
                                                        XFER_GPU_IN_FB,
                                                        stringbuilder() << "cuda fill channel (gpu=" << _gpu->info->index << ")")
      , gpu(_gpu)
    {
      Memory fbm = gpu->fbmem->me;

      unsigned bw = 0; // TODO
      unsigned latency = 0;

      add_path(Memory::NO_MEMORY, fbm,
               bw, latency, false, false, XFER_GPU_IN_FB);

      xdq.add_to_manager(bgwork);
    }

    XferDes *GPUfillChannel::create_xfer_des(uintptr_t dma_op,
                                             NodeID launch_node,
                                             XferDesID guid,
                                             const std::vector<XferDesPortInfo>& inputs_info,
                                             const std::vector<XferDesPortInfo>& outputs_info,
                                             int priority,
                                             XferDesRedopInfo redop_info,
                                             const void *fill_data, size_t fill_size)
    {
      assert(redop_info.id == 0);
      return new GPUfillXferDes(dma_op, this, launch_node, guid,
                                inputs_info, outputs_info,
                                priority,
                                fill_data, fill_size);
    }

    long GPUfillChannel::submit(Request** requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUreduceXferDes

    GPUreduceXferDes::GPUreduceXferDes(uintptr_t _dma_op, Channel *_channel,
                                       NodeID _launch_node, XferDesID _guid,
                                       const std::vector<XferDesPortInfo>& inputs_info,
                                       const std::vector<XferDesPortInfo>& outputs_info,
                                       int _priority,
                                       XferDesRedopInfo _redop_info)
      : XferDes(_dma_op, _channel, _launch_node, _guid,
                inputs_info, outputs_info,
                _priority, 0, 0)
      , redop_info(_redop_info)
    {
      kind = XFER_GPU_IN_FB;
      redop = get_runtime()->reduce_op_table.get(redop_info.id, 0);
      assert(redop);

      GPU *gpu = checked_cast<GPUreduceChannel *>(channel)->gpu;

      // select reduction kernel now - translate to CUfunction if possible
      void *host_proxy = (redop_info.is_fold ?
                            redop->cuda_fold_nonexcl_fn :
                            redop->cuda_apply_nonexcl_fn);
#ifdef REALM_USE_CUDART_HIJACK
      // we have the host->device mapping table for functions
      kernel = gpu->lookup_function(host_proxy);
#else
  #if CUDA_VERSION >= 11000
      // we can ask the runtime to perform the mapping for us
      int orig_device;
      CHECK_CUDART( cudaGetDevice(&orig_device) );
      CHECK_CUDART( cudaSetDevice(gpu->info->index) );
      CHECK_CUDART( cudaGetFuncBySymbol(&kernel, host_proxy) );
      CHECK_CUDART( cudaSetDevice(orig_device) );
  #else
      // no way to ask the runtime to perform the mapping, so we'll have
      //  to actually launch the kernels with the runtime API
      kernel_host_proxy = host_proxy;
  #endif
#endif

      stream = gpu->device_to_device_stream;
    }

    long GPUreduceXferDes::get_requests(Request** requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }

    bool GPUreduceXferDes::progress_xd(GPUreduceChannel *channel,
                                     TimeLimit work_until)
    {
      bool did_work = false;
      ReadSequenceCache rseqcache(this, 2 << 20);
      ReadSequenceCache wseqcache(this, 2 << 20);

      const size_t in_elem_size = redop->sizeof_rhs;
      const size_t out_elem_size = (redop_info.is_fold ? redop->sizeof_rhs : redop->sizeof_lhs);
      assert(redop_info.in_place);  // TODO: support for out-of-place reduces

      struct KernelArgs {
        uintptr_t dst_base, dst_stride;
        uintptr_t src_base, src_stride;
        uintptr_t count;
      };
      KernelArgs *args = 0; // allocate on demand
      size_t args_size = sizeof(KernelArgs) + redop->sizeof_userdata;

      while(true) {
        size_t min_xfer_size = 4096;  // TODO: make controllable
        size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
        if(max_bytes == 0)
          break;

        XferPort *in_port = 0, *out_port = 0;
        size_t in_span_start = 0, out_span_start = 0;
        if(input_control.current_io_port >= 0) {
          in_port = &input_ports[input_control.current_io_port];
          in_span_start = in_port->local_bytes_total;
        }
        if(output_control.current_io_port >= 0) {
          out_port = &output_ports[output_control.current_io_port];
          out_span_start = out_port->local_bytes_total;
        }

        // have to count in terms of elements, which requires redoing some math
        //  if in/out sizes do not match
        size_t max_elems;
        if(in_elem_size == out_elem_size) {
          max_elems = max_bytes / in_elem_size;
        } else {
          max_elems = std::min(input_control.remaining_count / in_elem_size,
                               output_control.remaining_count / out_elem_size);
          if(in_port != 0)
            max_elems = std::min(max_elems,
                                 in_port->addrlist.bytes_pending() / in_elem_size);
          if(out_port != 0)
            max_elems = std::min(max_elems,
                                 out_port->addrlist.bytes_pending() / out_elem_size);
        }

        size_t total_elems = 0;
        if(in_port != 0) {
          if(out_port != 0) {
            // input and output both exist - transfer what we can
            log_xd.info() << "gpureduce chunk: min=" << min_xfer_size
                          << " max_elems=" << max_elems;

            uintptr_t in_base = reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(0, 0));
            uintptr_t out_base = reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

            while(total_elems < max_elems) {
              AddressListCursor& in_alc = in_port->addrcursor;
              AddressListCursor& out_alc = out_port->addrcursor;

              uintptr_t in_offset = in_alc.get_offset();
              uintptr_t out_offset = out_alc.get_offset();

              // the reported dim is reduced for partially consumed address
              //  ranges - whatever we get can be assumed to be regular
              int in_dim = in_alc.get_dim();
              int out_dim = out_alc.get_dim();

              // the current reduction op interface can reduce multiple elements
              //  with a fixed address stride, which looks to us like either
              //  1D (stride = elem_size), or 2D with 1 elem/line

              size_t icount = in_alc.remaining(0) / in_elem_size;
              size_t ocount = out_alc.remaining(0) / out_elem_size;
              size_t istride, ostride;
              if((in_dim > 1) && (icount == 1)) {
                in_dim = 2;
                icount = in_alc.remaining(1);
                istride = in_alc.get_stride(1);
              } else {
                in_dim = 1;
                istride = in_elem_size;
              }
              if((out_dim > 1) && (ocount == 1)) {
                out_dim = 2;
                ocount = out_alc.remaining(1);
                ostride = out_alc.get_stride(1);
              } else {
                out_dim = 1;
                ostride = out_elem_size;
              }

              size_t elems_left = max_elems - total_elems;
              size_t elems = std::min(std::min(icount, ocount), elems_left);
              assert(elems > 0);

              // allocate kernel arg structure if this is our first call
              if(!args) {
                args = static_cast<KernelArgs *>(alloca(args_size));
                if(redop->sizeof_userdata)
                  memcpy(args+1, redop->userdata, redop->sizeof_userdata);
              }

              args->dst_base = out_base + out_offset;
              args->dst_stride = ostride;
              args->src_base = in_base + in_offset;
              args->src_stride = istride;
              args->count = elems;

              size_t threads_per_block = 256;
              size_t blocks_per_grid = 1 + ((elems - 1) / threads_per_block);

              {
                AutoGPUContext agc(channel->gpu);

#if defined(REALM_USE_CUDART_HIJACK) || (CUDA_VERSION >= 11000)
                void *extra[] = {
                  CU_LAUNCH_PARAM_BUFFER_POINTER, args,
                  CU_LAUNCH_PARAM_BUFFER_SIZE,    &args_size,
                  CU_LAUNCH_PARAM_END
                };

                CHECK_CU( cuLaunchKernel(kernel,
                                         blocks_per_grid, 1, 1,
                                         threads_per_block, 1, 1,
                                         0 /*sharedmem*/,
                                         stream->get_stream(),
                                         0 /*params*/,
                                         extra) );
#else
                int orig_device;
                void *params[] = {
                  &args->dst_base,
                  &args->dst_stride,
                  &args->src_base,
                  &args->src_stride,
                  &args->count,
                  args+1
                };
                CHECK_CUDART( cudaGetDevice(&orig_device) );
                CHECK_CUDART( cudaSetDevice(channel->gpu->info->index) );
                CHECK_CUDART( cudaLaunchKernel(kernel_host_proxy,
                                               dim3(blocks_per_grid, 1, 1),
                                               dim3(threads_per_block, 1, 1),
                                               params,
                                               0 /*sharedMem*/,
                                               (cudaStream_t)(stream->get_stream())) );
                CHECK_CUDART( cudaSetDevice(orig_device) );
#endif

                // insert fence to track completion of reduction kernel
                add_reference(); // released by transfer completion
                stream->add_notification(new GPUTransferCompletion(this,
                                                                   input_control.current_io_port,
                                                                   in_span_start,
                                                                   elems * in_elem_size,
                                                                   output_control.current_io_port,
                                                                   out_span_start,
                                                                   elems * out_elem_size));
              }

              in_span_start += elems * in_elem_size;
              out_span_start += elems * out_elem_size;

              in_alc.advance(in_dim-1,
                             elems * ((in_dim == 1) ? in_elem_size : 1));
              out_alc.advance(out_dim-1,
                              elems * ((out_dim == 1) ? out_elem_size : 1));

#ifdef DEBUG_REALM
              assert(elems <= elems_left);
#endif
              total_elems += elems;

              // stop if it's been too long, but make sure we do at least the
              //  minimum number of bytes
              if(((total_elems * in_elem_size) >= min_xfer_size) &&
                 work_until.is_expired()) break;
            }
          } else {
            // input but no output, so skip input bytes
            total_elems = max_elems;
            in_port->addrcursor.skip_bytes(total_elems * in_elem_size);

            rseqcache.add_span(input_control.current_io_port,
                               in_span_start, total_elems * in_elem_size);
            in_span_start += total_elems * in_elem_size;
          }
        } else {
          if(out_port != 0) {
            // output but no input, so skip output bytes
            total_elems = max_elems;
            out_port->addrcursor.skip_bytes(total_elems * out_elem_size);

            wseqcache.add_span(output_control.current_io_port,
                               out_span_start, total_elems * out_elem_size);
            out_span_start += total_elems * out_elem_size;
          } else {
            // skipping both input and output is possible for simultaneous
            //  gather+scatter
            total_elems = max_elems;
          }
        }

        bool done = record_address_consumption(total_elems * in_elem_size,
                                               total_elems * out_elem_size);

        did_work = true;

        if(done || work_until.is_expired())
          break;
      }

      rseqcache.flush();
      wseqcache.flush();

      return did_work;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUreduceChannel

    GPUreduceChannel::GPUreduceChannel(GPU *_gpu, BackgroundWorkManager *bgwork)
      : SingleXDQChannel<GPUreduceChannel,GPUreduceXferDes>(bgwork,
                                                            XFER_GPU_IN_FB,
                                                            stringbuilder() << "cuda reduce channel (gpu=" << _gpu->info->index << ")")
      , gpu(_gpu)
    {
      Memory fbm = gpu->fbmem->me;

      unsigned bw = 0; // TODO
      unsigned latency = 0;

      // intra-FB reduction
      add_path(fbm, fbm,
               bw, latency, true /*redops*/, false, XFER_GPU_IN_FB);

      // zero-copy to FB (no need for intermediate buffer in FB)
      for(std::set<Memory>::const_iterator it = gpu->pinned_sysmems.begin();
          it != gpu->pinned_sysmems.end();
          ++it)
        add_path(*it, fbm,
                 bw, latency, true /*redops*/, false, XFER_GPU_IN_FB);

      // unlike normal cuda p2p copies where we want to push from the source,
      //  reductions are always sent to the destination memory's gpu to keep the
      //  RMW loop as tight as possible
      for(std::set<Memory>::const_iterator it = gpu->peer_fbs.begin();
          it != gpu->peer_fbs.end();
          ++it)
        add_path(*it, fbm,
                 bw, latency, true /*redops*/, false, XFER_GPU_IN_FB);

      xdq.add_to_manager(bgwork);
    }

    /*static*/ bool GPUreduceChannel::is_gpu_redop(ReductionOpID redop_id)
    {
      if(redop_id == 0)
        return false;

      ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(redop_id, 0);
      assert(redop);

      // there's four different kernels, but they should be all or nothing, so
      //  just check one
      if(!redop->cuda_apply_excl_fn)
        return false;

      return true;
    }

    bool GPUreduceChannel::supports_path(Memory src_mem, Memory dst_mem,
                                         CustomSerdezID src_serdez_id,
                                         CustomSerdezID dst_serdez_id,
                                         ReductionOpID redop_id,
                                         XferDesKind *kind_ret /*= 0*/,
                                         unsigned *bw_ret /*= 0*/,
                                         unsigned *lat_ret /*= 0*/)
    {
      // give all the normal supports_path logic a chance to reject it first
      if(!Channel::supports_path(src_mem, dst_mem, src_serdez_id, dst_serdez_id,
                                 redop_id, kind_ret, bw_ret, lat_ret))
        return false;

      // if everything else was ok, check that we have a reduction op (if not,
      //   we want the cudamemcpy path to pick this up instead) and that it has
      //   cuda kernels available
      return is_gpu_redop(redop_id);
    }

    RemoteChannelInfo *GPUreduceChannel::construct_remote_info() const
    {
      return new GPUreduceRemoteChannelInfo(node, kind,
                                            reinterpret_cast<uintptr_t>(this),
                                            paths);
    }

    XferDes *GPUreduceChannel::create_xfer_des(uintptr_t dma_op,
                                               NodeID launch_node,
                                               XferDesID guid,
                                               const std::vector<XferDesPortInfo>& inputs_info,
                                               const std::vector<XferDesPortInfo>& outputs_info,
                                               int priority,
                                               XferDesRedopInfo redop_info,
                                               const void *fill_data, size_t fill_size)
    {
      assert(fill_size == 0);
      return new GPUreduceXferDes(dma_op, this, launch_node, guid,
                                  inputs_info, outputs_info,
                                  priority,
                                  redop_info);
    }

    long GPUreduceChannel::submit(Request** requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUreduceRemoteChannelInfo
    //

    GPUreduceRemoteChannelInfo::GPUreduceRemoteChannelInfo(NodeID _owner,
                                                           XferDesKind _kind,
                                                           uintptr_t _remote_ptr,
                                                           const std::vector<Channel::SupportedPath>& _paths)
      : SimpleRemoteChannelInfo(_owner, _kind, _remote_ptr, _paths)
    {}

    RemoteChannel *GPUreduceRemoteChannelInfo::create_remote_channel()
    {
      GPUreduceRemoteChannel *rc = new GPUreduceRemoteChannel(remote_ptr);
      rc->node = owner;
      rc->kind = kind;
      rc->paths.swap(paths);
      return rc;
    }

    // these templates can go here because they're only used by the helper below
    template <typename S>
    bool GPUreduceRemoteChannelInfo::serialize(S& serializer) const
    {
      return ((serializer << owner) &&
              (serializer << kind) &&
              (serializer << remote_ptr) &&
              (serializer << paths));
    }

    template <typename S>
    /*static*/ RemoteChannelInfo *GPUreduceRemoteChannelInfo::deserialize_new(S& deserializer)
    {
      NodeID owner;
      XferDesKind kind;
      uintptr_t remote_ptr;
      std::vector<Channel::SupportedPath> paths;

      if((deserializer >> owner) &&
         (deserializer >> kind) &&
         (deserializer >> remote_ptr) &&
         (deserializer >> paths)) {
        return new GPUreduceRemoteChannelInfo(owner, kind, remote_ptr, paths);
      } else {
        return 0;
      }
    }

    /*static*/ Serialization::PolymorphicSerdezSubclass<RemoteChannelInfo,
                                                        GPUreduceRemoteChannelInfo> GPUreduceRemoteChannelInfo::serdez_subclass;


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUreduceRemoteChannel
    //

    GPUreduceRemoteChannel::GPUreduceRemoteChannel(uintptr_t _remote_ptr)
      : RemoteChannel(_remote_ptr)
    {}

    bool GPUreduceRemoteChannel::supports_path(Memory src_mem, Memory dst_mem,
                                               CustomSerdezID src_serdez_id,
                                               CustomSerdezID dst_serdez_id,
                                               ReductionOpID redop_id,
                                               XferDesKind *kind_ret /*= 0*/,
                                               unsigned *bw_ret /*= 0*/,
                                               unsigned *lat_ret /*= 0*/)
    {
      // give all the normal supports_path logic a chance to reject it first
      if(!Channel::supports_path(src_mem, dst_mem, src_serdez_id, dst_serdez_id,
                                 redop_id, kind_ret, bw_ret, lat_ret))
        return false;

      // if everything else was ok, check that we have a reduction op (if not,
      //   we want the cudamemcpy path to pick this up instead) and that it has
      //   cuda kernels available
      return GPUreduceChannel::is_gpu_redop(redop_id);
    }


  }; // namespace Cuda

}; // namespace Realm
