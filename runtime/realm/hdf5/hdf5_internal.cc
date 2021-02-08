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

#include "realm/hdf5/hdf5_internal.h"

#include "realm/hdf5/hdf5_access.h"
#include "realm/logging.h"

namespace Realm {

  extern Logger log_hdf5;
  extern Logger log_request;

  namespace HDF5 {

    ////////////////////////////////////////////////////////////////////////
    //
    // class HDF5Memory

    HDF5Memory::HDF5Memory(Memory _me)
      : MemoryImpl(_me, 0 /*HDF doesn't have memory space*/, MKIND_HDF,
		   Memory::HDF_MEM, 0)
    {
    }

    HDF5Memory::~HDF5Memory(void)
    {
      // close all HDF metadata
    }

    void HDF5Memory::get_bytes(off_t offset, void *dst, size_t size)
    {
      assert(0);
    }

#if 0
    void HDF5Memory::get_bytes(ID::IDType inst_id, const DomainPoint& dp, int fid, void *dst, size_t size)
    {
      assert(0);
      HDFMetadata *metadata = hdf_metadata[inst_id];
      // use index to compute position in space
      assert(size == H5Tget_size(metadata->datatype_ids[fid]));
      hsize_t offset[3], count[3];
      for (int i = 0; i < metadata->ndims; i++) {
        offset[i] = dp.point_data[i] - metadata->lo[i];
      }
      count[0] = count[1] = count[2] = 1;
      hid_t dataspace_id = H5Dget_space(metadata->dataset_ids[fid]);
      H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
      hid_t memspace_id = H5Screate_simple(metadata->ndims, count, NULL);
      H5Dread(metadata->dataset_ids[fid], metadata->datatype_ids[fid], memspace_id, dataspace_id, H5P_DEFAULT, dst);
      H5Sclose(dataspace_id);
      H5Sclose(memspace_id);
    }
#endif

    void HDF5Memory::put_bytes(off_t offset, const void *src, size_t size)
    {
      assert(0);
    }

#if 0
    void HDF5Memory::put_bytes(ID::IDType inst_id, const DomainPoint& dp, int fid, const void *src, size_t size)
    {
      assert(0);
      HDFMetadata *metadata = hdf_metadata[inst_id];
      // use index to compute position in space
      assert(size == H5Tget_size(hdf_metadata[inst_id]->datatype_ids[fid]));
      hsize_t offset[3], count[3];
      for (int i = 0; i < metadata->ndims; i++) {
        offset[i] = dp.point_data[i] - metadata->lo[i];
      }
      count[0] = count[1] = count[2] = 1;
      hid_t dataspace_id = H5Dget_space(metadata->dataset_ids[fid]);
      H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
      hid_t memspace_id = H5Screate_simple(metadata->ndims, count, NULL);
      H5Dwrite(metadata->dataset_ids[fid], metadata->datatype_ids[fid], memspace_id, dataspace_id, H5P_DEFAULT, src);
      H5Sclose(dataspace_id);
      H5Sclose(memspace_id);
    }
#endif

    void *HDF5Memory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0; // cannot provide a pointer for it.
    }

    MemoryImpl::AllocationResult HDF5Memory::allocate_storage_immediate(RegionInstanceImpl *inst,
							    bool need_alloc_result,
							    bool poisoned,
							    TimeLimit work_until)
    {
      // if the allocation request doesn't include an external HDF5 resource,
      //  we fail it immediately
      ExternalHDF5Resource *res = dynamic_cast<ExternalHDF5Resource *>(inst->metadata.ext_resource);
      if(res == 0) {
	if(inst->metadata.ext_resource)
	  log_inst.warning() << "attempt to register non-hdf5 resource: mem=" << me << " resource=" << *(inst->metadata.ext_resource);
	else
	  log_inst.warning() << "attempt to allocate memory in hdf5 memory: layout=" << *(inst->metadata.layout);
	inst->notify_allocation(ALLOC_INSTANT_FAILURE, 0, work_until);
	return ALLOC_INSTANT_FAILURE;
      }

      // poisoned preconditions cancel the allocation request
      if(poisoned) {
	inst->notify_allocation(ALLOC_CANCELLED, 0, work_until);
	return ALLOC_CANCELLED;
      }

      // TODO: try to open the file once here to make sure it exists?

      AllocationResult result = ALLOC_INSTANT_SUCCESS;
      size_t inst_offset = 0;
      inst->notify_allocation(result, inst_offset, work_until);

      return result;
    }

    void HDF5Memory::release_storage_immediate(RegionInstanceImpl *inst,
					       bool poisoned,
					       TimeLimit work_until)
    {
      // nothing to do for a poisoned release
      if(poisoned)
	return;

      // we didn't save anything on the instance, so just ack and return
      inst->notify_deallocation();
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class HDF5XferDes

      HDF5XferDes::HDF5XferDes(uintptr_t _dma_op, Channel *_channel,
			       NodeID _launch_node, XferDesID _guid,
			       const std::vector<XferDesPortInfo>& inputs_info,
			       const std::vector<XferDesPortInfo>& outputs_info,
			       int _priority,
                               const void *_fill_data, size_t _fill_size)
	: XferDes(_dma_op, _channel, _launch_node, _guid,
		  inputs_info, outputs_info,
		  _priority, _fill_data, _fill_size)
	, req_in_use(false)
      {
	if((inputs_info.size() >= 1) &&
	   (input_ports[0].mem->kind == MemoryImpl::MKIND_HDF)) {
	  kind = XFER_HDF5_READ;
	} else if((outputs_info.size() >= 1) &&
		  (output_ports[0].mem->kind == MemoryImpl::MKIND_HDF)) {
	  kind = XFER_HDF5_WRITE;
	} else {
	  assert(0 && "neither source nor dest of HDFXferDes is hdf5!?");
	}

	hdf5_req.xd = this;
      }

      bool HDF5XferDes::request_available()
      {
	return !req_in_use;
      }

      Request* HDF5XferDes::dequeue_request()
      {
	assert(!req_in_use);
	req_in_use = true;
	hdf5_req.is_read_done = false;
	hdf5_req.is_write_done = false;
	// HDF5Request is handled by another thread, so must hold a reference
	add_reference();
        return &hdf5_req;
      }

      void HDF5XferDes::enqueue_request(Request* req)
      {
	assert(req_in_use);
	assert(req == &hdf5_req);
	req_in_use = false;
	remove_reference();
      }

     extern Logger log_hdf5;

      long HDF5XferDes::get_requests(Request** requests, long nr)
      {
        long idx = 0;
	
	while((idx < nr) && request_available()) {	  
	  // TODO: use control stream to determine which input/output ports
	  //  to use
	  int in_port_idx = 0;
	  int out_port_idx = 0;

	  XferPort *in_port = &input_ports[in_port_idx];
	  XferPort *out_port = &output_ports[out_port_idx];

	  // is our iterator done?
	  if(in_port->iter->done() || out_port->iter->done()) {
	    // non-ib iterators should end at the same time
	    assert((in_port->peer_guid != XFERDES_NO_GUID) || in_port->iter->done());
	    assert((out_port->peer_guid != XFERDES_NO_GUID) || out_port->iter->done());
	    iteration_completed.store_release(true);
	    break;
	  }

	  // no support for serdez ops
	  assert(in_port->serdez_op == 0);
	  assert(out_port->serdez_op == 0);

	  size_t max_bytes = max_req_size;

	  // if we're not the first in the chain, and we know the total bytes
	  //  written by the predecessor, don't exceed that
	  if(in_port->peer_guid != XFERDES_NO_GUID) {
	    size_t pre_max = in_port->remote_bytes_total.load() - in_port->local_bytes_total;
	    if(pre_max == 0) {
	      // due to unsynchronized updates to pre_bytes_total, this path
	      //  can happen for an empty transfer reading from an intermediate
	      //  buffer - handle it by looping around and letting the check
	      //  at the top of the loop notice it the second time around
	      if(in_port->local_bytes_total == 0)
		continue;
	      // otherwise, this shouldn't happen - we should detect this case
	      //  on the the transfer of those last bytes
	      assert(0);
	      iteration_completed.store_release(true);
	      break;
	    }
	    if(pre_max < max_bytes) {
	      log_request.info() << "pred limits xfer: " << max_bytes << " -> " << pre_max;
	      max_bytes = pre_max;
	    }

	    // further limit based on data that has actually shown up
	    max_bytes = in_port->seq_remote.span_exists(in_port->local_bytes_total, max_bytes);
	    if(max_bytes == 0)
	      break;
	  }

	  // similarly, limit our max transfer size based on the amount the
	  //  destination IB buffer can take (assuming there is an IB)
	  if(out_port->peer_guid != XFERDES_NO_GUID) {
	    max_bytes = out_port->seq_remote.span_exists(out_port->local_bytes_total, max_bytes);
	    if(max_bytes == 0)
	      break;
	  }

	  // HDF5 uses its own address info, instead of src/dst, we
	  //  distinguish between hdf5 and mem
	  TransferIterator *hdf5_iter = ((kind == XFER_HDF5_READ) ?
					   in_port->iter :
					   out_port->iter);
	  TransferIterator *mem_iter = ((kind == XFER_HDF5_READ) ?
					  out_port->iter :
					  in_port->iter);

	  TransferIterator::AddressInfo mem_info;
	  TransferIterator::AddressInfoHDF5 hdf5_info;

	  // always ask the HDF5 size for a step first
	  size_t hdf5_bytes = hdf5_iter->step_hdf5(max_bytes, hdf5_info,
						   true /*tentative*/);
          if(hdf5_bytes == 0) {
            // not enough space for even a single element - try again later
            break;
          }
	  // TODO: support 2D/3D for memory side of an HDF transfer?
	  size_t mem_bytes = mem_iter->step(hdf5_bytes, mem_info, 0,
					    true /*tentative*/);
	  if(mem_bytes == hdf5_bytes) {
	    // looks good - confirm the steps
	    hdf5_iter->confirm_step();
	    mem_iter->confirm_step();
	  } else {
	    // cancel the hdf5 step and try to just step by mem_bytes
	    assert(mem_bytes < hdf5_bytes);  // should never be larger
	    hdf5_iter->cancel_step();
	    hdf5_bytes = hdf5_iter->step_hdf5(mem_bytes, hdf5_info);
	    // multi-dimensional hdf5 iterators may round down the size,
	    //  so re-check the mem bytes
	    if(hdf5_bytes == mem_bytes) {
	      mem_iter->confirm_step();
	    } else {
	      mem_iter->cancel_step();
	      mem_bytes = mem_iter->step(hdf5_bytes, mem_info, 0);
	      // now must match
	      assert(hdf5_bytes == mem_bytes);
	    }
	  }

	  HDF5Request* new_req = (HDF5Request *)(dequeue_request());
	  new_req->src_port_idx = in_port_idx;
	  new_req->dst_port_idx = out_port_idx;
	  new_req->dim = Request::DIM_1D;
	  new_req->mem_base = ((kind == XFER_HDF5_READ) ?
			         out_port->mem :
			         in_port->mem)->get_direct_ptr(mem_info.base_offset,
							       mem_info.bytes_per_chunk);
	  // we'll open datasets on the first touch in this transfer
	  // (TODO: pre-open at instance attach time, but in thread-safe way)
	  HDF5Dataset *dset;
	  {
	    std::map<FieldID, HDF5Dataset *>::const_iterator it = datasets.find(hdf5_info.field_id);
	    if(it != datasets.end()) {
	      dset = it->second;
	    } else {
	      dset = HDF5Dataset::open(hdf5_info.filename->c_str(),
				       hdf5_info.dsetname->c_str(),
				       (kind == XFER_HDF5_READ));
	      assert(dset != 0);
	      assert(hdf5_info.extent.size() == size_t(dset->ndims));
	      datasets[hdf5_info.field_id] = dset;
	    }
	  }

	  new_req->dataset_id = dset->dset_id;
	  new_req->datatype_id = dset->dtype_id;

	  std::vector<hsize_t> mem_dims = hdf5_info.extent;
	  CHECK_HDF5( new_req->mem_space_id = H5Screate_simple(mem_dims.size(), mem_dims.data(), NULL) );
	  //std::vector<hsize_t> mem_start(DIM, 0);
	  //CHECK_HDF5( H5Sselect_hyperslab(new_req->mem_space_id, H5S_SELECT_SET, ms_start, NULL, count, NULL) );

	  CHECK_HDF5( new_req->file_space_id = H5Screate_simple(hdf5_info.dset_bounds.size(), hdf5_info.dset_bounds.data(), 0) );
	  CHECK_HDF5( H5Sselect_hyperslab(new_req->file_space_id, H5S_SELECT_SET, hdf5_info.offset.data(), 0, hdf5_info.extent.data(), 0) );

	  new_req->nbytes = hdf5_bytes;

	  new_req->read_seq_pos = in_port->local_bytes_total;
	  new_req->read_seq_count = hdf5_bytes;

	  // update bytes read unless we're using indirection
	  if(in_port->indirect_port_idx < 0) 
	    in_port->local_bytes_total += hdf5_bytes;

	  new_req->write_seq_pos = out_port->local_bytes_total;
	  new_req->write_seq_count = hdf5_bytes;
	  out_port->local_bytes_total += hdf5_bytes;

	  requests[idx++] = new_req;

	  // make sure iteration_completed is set appropriately before we
	  //  process the request (so that multi-hop successors are notified
	  //  properly)
	  if(hdf5_iter->done())
	    iteration_completed.store_release(true);
	}

	return idx;
      }

      bool HDF5XferDes::progress_xd(HDF5Channel *channel, TimeLimit work_until)
      {
        if(fill_size == 0) {
          // old path for copies - TODO: use newer style like fill below
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

        // fill only path for now
        bool did_work = false;
	ReadSequenceCache rseqcache(this, 2 << 20);  // flush after 2MB

        while(true) {
          size_t control_count = update_control_info(&rseqcache);
          if(control_count == 0)
            break;

          bool done = false;

          if(output_control.current_io_port >= 0) {
            XferPort *out_port = &output_ports[output_control.current_io_port];

            // can't do a single write larger than our (replicated) fill data
            size_t max_bytes = std::min(control_count, size_t(MAX_FILL_SIZE_IN_BYTES));
            TransferIterator::AddressInfoHDF5 hdf5_info;

            // always ask the HDF5 size for a step first
            size_t hdf5_bytes = out_port->iter->step_hdf5(max_bytes, hdf5_info,
                                                          false /*!tentative*/);
            assert(hdf5_bytes > 0);

            // if this is bigger than any transfer we've done so far, grow our
            //  buffer
            if(hdf5_bytes > fill_size)
              replicate_fill_data(hdf5_bytes);

            // we'll open datasets on the first touch in this transfer
            // (TODO: pre-open at instance attach time, but in thread-safe way)
            HDF5Dataset *dset;
            {
              std::map<FieldID, HDF5Dataset *>::const_iterator it = datasets.find(hdf5_info.field_id);
              if(it != datasets.end()) {
                dset = it->second;
              } else {
                dset = HDF5Dataset::open(hdf5_info.filename->c_str(),
                                         hdf5_info.dsetname->c_str(),
                                         (kind == XFER_HDF5_READ));
                assert(dset != 0);
                assert(hdf5_info.extent.size() == size_t(dset->ndims));
                datasets[hdf5_info.field_id] = dset;
              }
            }

            std::vector<hsize_t> mem_dims = hdf5_info.extent;
            hid_t mem_space_id, file_space_id;
            CHECK_HDF5( mem_space_id = H5Screate_simple(mem_dims.size(),
                                                        mem_dims.data(), 0) );

            CHECK_HDF5( file_space_id = H5Screate_simple(hdf5_info.dset_bounds.size(),
                                                         hdf5_info.dset_bounds.data(), 0) );
            CHECK_HDF5( H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET,
                                            hdf5_info.offset.data(), 0,
                                            hdf5_info.extent.data(), 0) );

            CHECK_HDF5( H5Dwrite(dset->dset_id, dset->dtype_id,
                                 mem_space_id, file_space_id,
                                 H5P_DEFAULT, fill_data) );

            CHECK_HDF5( H5Sclose(mem_space_id) );
            CHECK_HDF5( H5Sclose(file_space_id) );

            update_bytes_write(output_control.current_io_port,
                               out_port->local_bytes_total, hdf5_bytes);

            out_port->local_bytes_total += hdf5_bytes;
            out_port->local_bytes_cons.fetch_add(hdf5_bytes);
            done = out_port->iter->done();

            output_control.remaining_count -= hdf5_bytes;
          }

          did_work = true;

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

      void HDF5XferDes::notify_request_read_done(Request* req)
      {
	default_notify_request_read_done(req);
      }

      void HDF5XferDes::notify_request_write_done(Request* req)
      {
        HDF5Request* hdf_req = (HDF5Request*) req;
        //pthread_rwlock_wrlock(&hdf_metadata->hdf_memory->rwlock);
        CHECK_HDF5( H5Sclose(hdf_req->mem_space_id) );
        CHECK_HDF5( H5Sclose(hdf_req->file_space_id) );
        //pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);

	default_notify_request_write_done(req);
      }

      void HDF5XferDes::flush()
      {
        if (kind == XFER_HDF5_READ) {
        } else {
          assert(kind == XFER_HDF5_WRITE);
	  //CHECK_HDF5( H5Fflush(hdf_metadata->file_id, H5F_SCOPE_LOCAL) );
          // for (fit = oas_vec.begin(); fit != oas_vec.end(); fit++) {
          //   off_t hdf_idx = fit->dst_offset;
          //   hid_t dataset_id = hdf_metadata->dataset_ids[hdf_idx];
          //   //TODO: I am not sure if we need a lock here to protect HDFflush
          //   H5Fflush(dataset_id, H5F_SCOPE_LOCAL);
          // }
        }

	for(std::map<FieldID, HDF5Dataset *>::const_iterator it = datasets.begin();
	    it != datasets.end();
	    ++it)
	  it->second->close();
	datasets.clear();
      }


    ////////////////////////////////////////////////////////////////////////
    //
    // class HDF5Channel

      static const Memory::Kind cpu_mem_kinds[] = { Memory::SYSTEM_MEM,
						    Memory::REGDMA_MEM,
						    Memory::Z_COPY_MEM,
                                                    Memory::SOCKET_MEM };
      static const size_t num_cpu_mem_kinds = sizeof(cpu_mem_kinds) / sizeof(cpu_mem_kinds[0]);

      HDF5Channel::HDF5Channel(BackgroundWorkManager *bgwork)
	: SingleXDQChannel<HDF5Channel, HDF5XferDes>(bgwork,
						     XFER_NONE /*FIXME*/,
						     "hdf5 channel")
      {
	unsigned bw = 0; // TODO
	unsigned latency = 0;
	// any combination of SYSTEM/REGDMA/Z_COPY_MEM
	for(size_t i = 0; i < num_cpu_mem_kinds; i++) {
	  add_path(Memory::HDF_MEM, false,
		   cpu_mem_kinds[i], false,
		   bw, latency, false, false, XFER_HDF5_READ);

	  add_path(cpu_mem_kinds[i], false,
		   Memory::HDF_MEM, false,
		   bw, latency, false, false, XFER_HDF5_WRITE);
	}

        // also indicate willingness to handle fills to HDF5 (src == NO_MEMORY)
        add_path(Memory::NO_MEMORY,
                 Memory::HDF_MEM, false,
                 bw, latency, false, false, XFER_HDF5_WRITE);
      }

      HDF5Channel::~HDF5Channel() {}

      XferDes *HDF5Channel::create_xfer_des(uintptr_t dma_op,
					    NodeID launch_node,
					    XferDesID guid,
					    const std::vector<XferDesPortInfo>& inputs_info,
					    const std::vector<XferDesPortInfo>& outputs_info,
					    int priority,
					    XferDesRedopInfo redop_info,
					    const void *fill_data, size_t fill_size)
      {
	assert(redop_info.id == 0);
	return new HDF5XferDes(dma_op, this, launch_node, guid,
			       inputs_info, outputs_info,
			       priority,
                               fill_data, fill_size);
      }

      long HDF5Channel::submit(Request** requests, long nr)
      {
        HDF5Request** hdf_reqs = (HDF5Request**) requests;
        for (long i = 0; i < nr; i++) {
          HDF5Request* req = hdf_reqs[i];
	  // no serdez support
	  assert(req->xd->input_ports[req->src_port_idx].serdez_op == 0);
	  assert(req->xd->output_ports[req->dst_port_idx].serdez_op == 0);
          //pthread_rwlock_rdlock(req->rwlock);
          if (req->xd->kind == XFER_HDF5_READ)
            CHECK_HDF5( H5Dread(req->dataset_id, req->datatype_id,
				req->mem_space_id, req->file_space_id,
				H5P_DEFAULT, req->mem_base) );
	  else
            CHECK_HDF5( H5Dwrite(req->dataset_id, req->datatype_id,
				 req->mem_space_id, req->file_space_id,
				 H5P_DEFAULT, req->mem_base) );
          //pthread_rwlock_unlock(req->rwlock);
          req->xd->notify_request_read_done(req);
          req->xd->notify_request_write_done(req);
        }
        return nr;
      }


    ////////////////////////////////////////////////////////////////////////
    //
    // class HDF5Memory

  }; // namespace HDF5

}; // namespace Realm
