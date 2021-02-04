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

namespace Realm {

  namespace Cuda {

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUXferDes

      GPUXferDes::GPUXferDes(DmaRequest *_dma_request, Channel *_channel,
			           NodeID _launch_node, XferDesID _guid,
				   const std::vector<XferDesPortInfo>& inputs_info,
				   const std::vector<XferDesPortInfo>& outputs_info,
				   bool _mark_start,
				   uint64_t _max_req_size, long max_nr, int _priority,
				   XferDesFence* _complete_fence)
	: XferDes(_dma_request, _channel, _launch_node, _guid,
		  inputs_info, outputs_info,
		  _mark_start,
		  _max_req_size, _priority,
		  _complete_fence)
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

      XferDes *GPUChannel::create_xfer_des(DmaRequest *dma_request,
					   NodeID launch_node,
					   XferDesID guid,
					   const std::vector<XferDesPortInfo>& inputs_info,
					   const std::vector<XferDesPortInfo>& outputs_info,
					   bool mark_started,
					   uint64_t max_req_size, long max_nr, int priority,
					   XferDesFence *complete_fence)
      {
	return new GPUXferDes(dma_request, this, launch_node, guid,
			      inputs_info, outputs_info,
			      mark_started, max_req_size, max_nr, priority,
			      complete_fence);
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
    // class GPUXferDes

  }; // namespace Cuda

}; // namespace Realm
