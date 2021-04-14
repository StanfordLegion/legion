/* Copyright 2021 Stanford University, NVIDIA Corporation
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

#include "realm/hip/hip_internal.h"

namespace Realm {
  
  extern Logger log_xd;

  namespace Hip {
    
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
		     (input_ports[i].mem->kind == MemoryImpl::MKIND_ZEROCOPY));
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
	// TODO: add support for 3D HIP copies (just 1D and 2D for now)
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
                uint8_t fill_u8;
                memcpy(&fill_u8, fill_data, 1);
                if(out_dim == 1) {
                  size_t bytes = out_alc.remaining(0);
                  CHECK_CU( hipMemsetD8Async((hipDeviceptr_t)(out_base + out_offset),
                                            fill_u8,
                                            bytes,
                                            stream->get_stream()) );
                  out_alc.advance(0, bytes);
                  total_bytes += bytes;
                } else {
                  size_t bytes = out_alc.remaining(0);
                  size_t lines = out_alc.remaining(1);
                  CHECK_CU( hipMemset2DAsync((void*)(out_base + out_offset),
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
                uint16_t fill_u16;
                memcpy(&fill_u16, fill_data, 2);
                if(out_dim == 1) {
                  size_t bytes = out_alc.remaining(0);
  #ifdef DEBUG_REALM
                  assert((bytes & 1) == 0);
  #endif
                  CHECK_CU( hipMemsetD16Async((hipDeviceptr_t)(out_base + out_offset),
                                             fill_u16,
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
                  CHECK_CU( hipMemset2DAsync((void*)(out_base + out_offset),
                                               out_alc.get_stride(1),
                                               *reinterpret_cast<const uint8_t *>(fill_data),
                                               bytes, lines,
                                               stream->get_stream()) );
                  out_alc.advance(1, lines);
                  total_bytes += bytes * lines;
                }
                break;
              }

              case 4: {
                // memset32
                uint32_t fill_u32;
                memcpy(&fill_u32, fill_data, 4);
                if(out_dim == 1) {
                  size_t bytes = out_alc.remaining(0);
  #ifdef DEBUG_REALM
                  assert((bytes & 3) == 0);
  #endif
                  CHECK_CU( hipMemsetD32Async((hipDeviceptr_t)(out_base + out_offset),
                                             fill_u32,
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
                  CHECK_CU( hipMemset2DAsync((void*)(out_base + out_offset),
                                               out_alc.get_stride(1),
                                               *reinterpret_cast<const uint8_t *>(fill_data),
                                               bytes, lines,
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
                // if((reduced_fill_size & 3) == 0) {
                //   // 32-bit partial fills allowed
                //   while(partial_bytes <= (reduced_fill_size - 4)) {
                //     uint32_t fill_u32;
                //     memcpy(&fill_u32,
                //            reinterpret_cast<const uint8_t *>(fill_data) + partial_bytes,
                //            4);
                //     CHECK_CU( hipMemset2DAsync((void*)(out_base + out_offset + partial_bytes),
                //                                  reduced_fill_size,
                //                                  fill_u32,
                //                                  1 /*"width"*/, elems /*"height"*/,
                //                                  stream->get_stream()) );
                //     partial_bytes += 4;
                //   }
                // }
                // if((reduced_fill_size & 1) == 0) {
                //   // 16-bit partial fills allowed
                //   while(partial_bytes <= (reduced_fill_size - 2)) {
                //     uint16_t fill_u16;
                //     memcpy(&fill_u16,
                //            reinterpret_cast<const uint8_t *>(fill_data) + partial_bytes,
                //            2);                                              
                //     CHECK_CU( hipMemset2DAsync((void*)(out_base + out_offset + partial_bytes),
                //                                  reduced_fill_size,
                //                                  fill_u16,
                //                                  1 /*"width"*/, elems /*"height"*/,
                //                                  stream->get_stream()) );
                //     partial_bytes += 2;
                //   }
                // }
                // leftover or unaligned bytes are done 8 bits at a time
                while(partial_bytes < reduced_fill_size) {
                  uint8_t fill_u8;
                  memcpy(&fill_u8,
                         reinterpret_cast<const uint8_t *>(fill_data) + partial_bytes,
                         1);
                  CHECK_CU( hipMemset2DAsync((void*)(out_base + out_offset + partial_bytes),
                                             reduced_fill_size,
                                             fill_u8,
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
                  printf("memset memcpy2d\n");

                  void *srcDevice = (void*)(out_base + out_offset);

                  size_t lines_done = 1;  // first line already valid
                  while(lines_done < lines) {
                    size_t todo = std::min(lines_done, lines - lines_done);
                    void *dstDevice = (void*)(out_base + out_offset +
                                                   (lines_done * lstride));
                    CHECK_CU( hipMemcpy2DAsync(dstDevice, lstride, srcDevice, lstride, bytes, todo, hipMemcpyDeviceToDevice, stream->get_stream()) );
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
                      printf("memset memcpy3d\n");
                      hipMemcpy3DParms copy3d = {0};
                      void *srcDevice = (void*)(out_base + out_offset);
                      copy3d.srcPtr = make_hipPitchedPtr((void*)srcDevice, lstride, bytes, pstride/lstride);
                      copy3d.srcPos = make_hipPos(0,0,0);
                      copy3d.dstPos = make_hipPos(0,0,0);
#ifdef __HIP_PLATFORM_NVCC__
                      copy3d.kind = cudaMemcpyDeviceToDevice;
#else
                      copy3d.kind = hipMemcpyDeviceToDevice;
#endif

                      size_t planes_done = 1;  // first plane already valid
                      while(planes_done < planes) {
                        size_t todo = std::min(planes_done, planes - planes_done);
                        void *dstDevice = (void*)(out_base + out_offset +
                                                  (planes_done * pstride));
                        copy3d.dstPtr = make_hipPitchedPtr(dstDevice, lstride, bytes, pstride/lstride);
                        copy3d.extent = make_hipExtent(bytes, lines, todo);
                        CHECK_CU( hipMemcpy3DAsync(&copy3d, stream->get_stream()) );
                        planes_done += todo;
                      }

                      out_alc.advance(2, planes);
                      total_bytes += bytes * lines * planes;
                    } else {
                      // plane-at-a-time fallback - can reuse most of copy2d
                      //  setup above

                      for(size_t p = 1; p < planes; p++) {
                        void *dstDevice = (void*)(out_base + out_offset +
                                                       (p * pstride));
                        CHECK_CU( hipMemcpy2DAsync(dstDevice, lstride, srcDevice, lstride, bytes, lines, hipMemcpyDeviceToDevice, stream->get_stream()) );
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


  }; // namespace Hip

}; // namespace Realm
