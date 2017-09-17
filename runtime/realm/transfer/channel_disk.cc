/* Copyright 2016 Stanford University
 * Copyright 2016 Los Alamos National Laboratory
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

#include "channel_disk.h"

namespace LegionRuntime {
  namespace LowLevel {
    typedef Realm::AsyncFileIOContext AsyncFileIOContext;

    FileXferDes::FileXferDes(DmaRequest* _dma_request,
			     gasnet_node_t _launch_node,
			     XferDesID _guid,
			     XferDesID _pre_xd_guid,
			     XferDesID _next_xd_guid,
			     uint64_t next_max_rw_gap,
			     size_t src_ib_offset, 
			     size_t src_ib_size,
			     bool mark_started,
			     RegionInstance inst,
			     Memory _src_mem, Memory _dst_mem,
			     TransferIterator *_src_iter, TransferIterator *_dst_iter,
			     uint64_t _max_req_size,
			     long max_nr,
			     int _priority,
			     XferOrder::Type _order,
			     XferKind _kind,
			     XferDesFence* _complete_fence)
      : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid,
                _next_xd_guid, next_max_rw_gap, src_ib_offset, src_ib_size,
		mark_started,
		//_src_buf, _dst_buf, _domain, _oas_vec,
		_src_mem, _dst_mem, _src_iter, _dst_iter,
		_max_req_size, _priority,
                _order, _kind, _complete_fence)
    {
      //MemoryImpl* src_mem_impl = get_runtime()->get_memory_impl(_src_buf.memory);
      //MemoryImpl* dst_mem_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
      switch (kind) {
        case XferDes::XFER_FILE_READ:
        {
          ID src_id(inst);
          unsigned src_index = src_id.instance.inst_idx;
          fd = ((FileMemory*)src_mem)->get_file_des(src_index);
          channel = get_channel_manager()->get_file_read_channel();
          //buf_base = (const char*) dst_mem_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
          assert(src_mem->kind == MemoryImpl::MKIND_FILE);
          break;
        }
        case XferDes::XFER_FILE_WRITE:
        {
          ID dst_id(inst);
          unsigned dst_index = dst_id.instance.inst_idx;
          fd = ((FileMemory*)dst_mem)->get_file_des(dst_index);
          channel = get_channel_manager()->get_file_write_channel();
          //buf_base = (const char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
          assert(dst_mem->kind == MemoryImpl::MKIND_FILE);
          break;
        }
        default:
          assert(0);
      }
      file_reqs = (FileRequest*) calloc(max_nr, sizeof(DiskRequest));
      for (int i = 0; i < max_nr; i++) {
        file_reqs[i].xd = this;
        file_reqs[i].fd = fd;
        enqueue_request(&file_reqs[i]);
      }
    }

    long FileXferDes::get_requests(Request** requests, long nr)
    {
      FileRequest** reqs = (FileRequest**) requests;
      long new_nr = default_get_requests(requests, nr);
      switch (kind) {
        case XferDes::XFER_FILE_READ:
        {
          for (long i = 0; i < new_nr; i++) {
            reqs[i]->file_off = reqs[i]->src_off;
            //reqs[i]->mem_base = (char*)(buf_base + reqs[i]->dst_off);
	    reqs[i]->mem_base = dst_mem->get_direct_ptr(reqs[i]->dst_off,
							reqs[i]->nbytes);
	    assert(reqs[i]->mem_base != 0);
          }
          break;
        }
        case XferDes::XFER_FILE_WRITE:
        {
          for (long i = 0; i < new_nr; i++) {
            //reqs[i]->mem_base = (char*)(buf_base + reqs[i]->src_off);
	    reqs[i]->mem_base = src_mem->get_direct_ptr(reqs[i]->src_off,
							reqs[i]->nbytes);
	    assert(reqs[i]->mem_base != 0);
            reqs[i]->file_off = reqs[i]->dst_off;
          }
          break;
        }
        default:
          assert(0);
      }
      return new_nr;
    }

    void FileXferDes::notify_request_read_done(Request* req)
    {
      default_notify_request_read_done(req);
    }

    void FileXferDes::notify_request_write_done(Request* req)
    {
      default_notify_request_write_done(req);
    }

    void FileXferDes::flush()
    {
      fsync(fd);
    }

    DiskXferDes::DiskXferDes(DmaRequest* _dma_request,
			     gasnet_node_t _launch_node,
			     XferDesID _guid,
			     XferDesID _pre_xd_guid,
			     XferDesID _next_xd_guid,
			     uint64_t next_max_rw_gap,
			     size_t src_ib_offset, 
			     size_t src_ib_size,
			     bool mark_started,
			     Memory _src_mem, Memory _dst_mem,
			     TransferIterator *_src_iter, TransferIterator *_dst_iter,
			     uint64_t _max_req_size,
			     long max_nr,
			     int _priority,
			     XferOrder::Type _order,
			     XferKind _kind,
			     XferDesFence* _complete_fence)
      : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid,
                _next_xd_guid, next_max_rw_gap, src_ib_offset, src_ib_size,
		mark_started,
		//_src_buf, _dst_buf, _domain, _oas_vec,
		_src_mem, _dst_mem, _src_iter, _dst_iter,
		_max_req_size, _priority,
                _order, _kind, _complete_fence)
    {
      //MemoryImpl* src_mem_impl = get_runtime()->get_memory_impl(_src_buf.memory);
      //MemoryImpl* dst_mem_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
      switch (kind) {
        case XferDes::XFER_DISK_READ:
        {
          channel = get_channel_manager()->get_disk_read_channel();
          fd = ((Realm::DiskMemory*)src_mem)->fd;
          //buf_base = (const char*) dst_mem_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
          assert(src_mem->kind == MemoryImpl::MKIND_DISK);
          break;
        }
        case XferDes::XFER_DISK_WRITE:
        {
          channel = get_channel_manager()->get_disk_write_channel();
          fd = ((Realm::DiskMemory*)dst_mem)->fd;
          //buf_base = (const char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
          assert(dst_mem->kind == MemoryImpl::MKIND_DISK);
          break;
        }
        default:
          assert(0);
      }
      disk_reqs = (DiskRequest*) calloc(max_nr, sizeof(DiskRequest));
      for (int i = 0; i < max_nr; i++) {
        disk_reqs[i].xd = this;
        disk_reqs[i].fd = fd;
        enqueue_request(&disk_reqs[i]);
      }
    }

    long DiskXferDes::get_requests(Request** requests, long nr)
    {
      DiskRequest** reqs = (DiskRequest**) requests;
      long new_nr = default_get_requests(requests, nr);
      switch (kind) {
        case XferDes::XFER_DISK_READ:
        {
          for (long i = 0; i < new_nr; i++) {
            reqs[i]->disk_off = /*src_buf.alloc_offset +*/ reqs[i]->src_off;
            //reqs[i]->mem_base = (char*)(buf_base + reqs[i]->dst_off);
	    reqs[i]->mem_base = dst_mem->get_direct_ptr(reqs[i]->dst_off,
							reqs[i]->nbytes);
	    assert(reqs[i]->mem_base != 0);
          }
          break;
        }
        case XferDes::XFER_DISK_WRITE:
        {
          for (long i = 0; i < new_nr; i++) {
            //reqs[i]->mem_base = (char*)(buf_base + reqs[i]->src_off);
	    reqs[i]->mem_base = src_mem->get_direct_ptr(reqs[i]->src_off,
							reqs[i]->nbytes);
	    assert(reqs[i]->mem_base != 0);
            reqs[i]->disk_off = /*dst_buf.alloc_offset +*/ reqs[i]->dst_off;
          }
          break;
        }
        default:
          assert(0);
      }
      return new_nr;
    }

    void DiskXferDes::notify_request_read_done(Request* req)
    {
      default_notify_request_read_done(req);
    }

    void DiskXferDes::notify_request_write_done(Request* req)
    {
      default_notify_request_write_done(req);
    }

    void DiskXferDes::flush()
    {
      fsync(fd);
    }

    FileChannel::FileChannel(long max_nr, XferDes::XferKind _kind)
    {
      kind = _kind;
    }

    FileChannel::~FileChannel()
    {
    }

    long FileChannel::submit(Request** requests, long nr)
    {
      AsyncFileIOContext* aio_ctx = AsyncFileIOContext::get_singleton();
      for (long i = 0; i < nr; i++) {
        FileRequest* req = (FileRequest*) requests[i];
        switch (kind) {
          case XferDes::XFER_FILE_READ:
            aio_ctx->enqueue_read(req->fd, req->file_off,
                                  req->nbytes, req->mem_base, req);
            break;
          case XferDes::XFER_FILE_WRITE:
            aio_ctx->enqueue_write(req->fd, req->file_off,
                                   req->nbytes, req->mem_base, req);
            break;
          default:
            assert(0);
        }
      }
      return nr;
    }

    void FileChannel::pull()
    {
      AsyncFileIOContext::get_singleton()->make_progress();
    }

    long FileChannel::available()
    {
      return AsyncFileIOContext::get_singleton()->available();
    }

    DiskChannel::DiskChannel(long max_nr, XferDes::XferKind _kind)
    {
      kind = _kind;
    }

    DiskChannel::~DiskChannel()
    {
    }

    long DiskChannel::submit(Request** requests, long nr)
    {
      AsyncFileIOContext* aio_ctx = AsyncFileIOContext::get_singleton();
      for (long i = 0; i < nr; i++) {
        DiskRequest* req = (DiskRequest*) requests[i];
        switch (kind) {
          case XferDes::XFER_DISK_READ:
            aio_ctx->enqueue_read(req->fd, req->disk_off,
                                  req->nbytes, req->mem_base, req);
            break;
          case XferDes::XFER_DISK_WRITE:
            aio_ctx->enqueue_write(req->fd, req->disk_off,
                                   req->nbytes, req->mem_base, req);
            break;
          default:
            assert(0);
        }
      }
      return nr;
    }

    void DiskChannel::pull()
    {
      AsyncFileIOContext::get_singleton()->make_progress();
    }

    long DiskChannel::available()
    {
      return AsyncFileIOContext::get_singleton()->available();
    }

#if 0
    template class FileXferDes<0>;
    template class FileXferDes<1>;
    template class FileXferDes<2>;
    template class FileXferDes<3>;
    template class DiskXferDes<0>;
    template class DiskXferDes<1>;
    template class DiskXferDes<2>;
    template class DiskXferDes<3>;
#endif
  } // namespace LowLevel
} // namespace LegionRuntime
