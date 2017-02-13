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
    template<unsigned DIM>
    FileXferDes<DIM>::FileXferDes(DmaRequest* _dma_request,
                                  gasnet_node_t _launch_node,
                                  XferDesID _guid,
                                  XferDesID _pre_xd_guid,
                                  XferDesID _next_xd_guid,
                                  bool mark_started,
				  RegionInstance inst,
                                  const Buffer& _src_buf,
                                  const Buffer& _dst_buf,
                                  const Domain& _domain,
                                  const std::vector<OffsetsAndSize>& _oas_vec,
                                  uint64_t _max_req_size,
                                  long max_nr,
                                  int _priority,
                                  XferOrder::Type _order,
                                  XferKind _kind,
                                  XferDesFence* _complete_fence)
      : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid,
                _next_xd_guid, mark_started, _src_buf, _dst_buf,
                _domain, _oas_vec, _max_req_size, _priority,
                _order, _kind, _complete_fence)
    {
      MemoryImpl* src_mem_impl = get_runtime()->get_memory_impl(_src_buf.memory);
      MemoryImpl* dst_mem_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
      switch (kind) {
        case XferDes::XFER_FILE_READ:
        {
          ID src_id(inst);
          unsigned src_index = src_id.instance.inst_idx;
          fd = ((FileMemory*)src_mem_impl)->get_file_des(src_index);
          channel = get_channel_manager()->get_file_read_channel();
          buf_base = (const char*) dst_mem_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
          assert(src_mem_impl->kind == MemoryImpl::MKIND_FILE);
          break;
        }
        case XferDes::XFER_FILE_WRITE:
        {
          ID dst_id(inst);
          unsigned dst_index = dst_id.instance.inst_idx;
          fd = ((FileMemory*)dst_mem_impl)->get_file_des(dst_index);
          channel = get_channel_manager()->get_file_write_channel();
          buf_base = (const char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
          assert(dst_mem_impl->kind == MemoryImpl::MKIND_FILE);
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

    template<unsigned DIM>
    long FileXferDes<DIM>::get_requests(Request** requests, long nr)
    {
      FileRequest** reqs = (FileRequest**) requests;
      long new_nr = default_get_requests<DIM>(requests, nr);
      switch (kind) {
        case XferDes::XFER_FILE_READ:
        {
          for (long i = 0; i < new_nr; i++) {
            reqs[i]->file_off = reqs[i]->src_off;
            reqs[i]->mem_base = (char*)(buf_base + reqs[i]->dst_off);
          }
          break;
        }
        case XferDes::XFER_FILE_WRITE:
        {
          for (long i = 0; i < new_nr; i++) {
            reqs[i]->mem_base = (char*)(buf_base + reqs[i]->src_off);
            reqs[i]->file_off = reqs[i]->dst_off;
          }
          break;
        }
        default:
          assert(0);
      }
      return new_nr;
    }

    template<unsigned DIM>
    void FileXferDes<DIM>::notify_request_read_done(Request* req)
    {
      default_notify_request_read_done(req);
    }

    template<unsigned DIM>
    void FileXferDes<DIM>::notify_request_write_done(Request* req)
    {
      default_notify_request_write_done(req);
    }

    template<unsigned DIM>
    void FileXferDes<DIM>::flush()
    {
      fsync(fd);
    }

    FileChannel::FileChannel(long max_nr, XferDes::XferKind _kind)
    {
      kind = _kind;
      ctx = 0;
      capacity = max_nr;
      int ret = io_setup(max_nr, &ctx);
      assert(ret >= 0);
      cb = (struct iocb*) calloc(max_nr, sizeof(struct iocb));
      cbs = (struct iocb**) calloc(max_nr, sizeof(struct iocb*));
      events = (struct io_event*) calloc(max_nr, sizeof(struct io_event));
      switch (kind) {
        case XferDes::XFER_FILE_READ:
          for (long i = 0; i < max_nr; i++) {
            memset(&cb[i], 0, sizeof(cb[i]));
            cb[i].aio_lio_opcode = IOCB_CMD_PREAD;
            available_cb.push_back(&cb[i]);
          }
          break;
        case XferDes::XFER_FILE_WRITE:
          for (long i = 0; i < max_nr; i++) {
            memset(&cb[i], 0, sizeof(cb[i]));
            cb[i].aio_lio_opcode = IOCB_CMD_PWRITE;
            available_cb.push_back(&cb[i]);
          }
          break;
        default:
          assert(0);
      }
    }

    FileChannel::~FileChannel()
    {
      io_destroy(ctx);
      free(cb);
      free(cbs);
      free(events);
    }

    long FileChannel::submit(Request** requests, long nr)
    {
      int ns = 0;
      switch (kind) {
        case XferDes::XFER_FILE_READ:
        case XferDes::XFER_FILE_WRITE:
          while (ns < nr && !available_cb.empty()) {
            FileRequest* req = (FileRequest*) requests[ns];
            cbs[ns] = available_cb.back();
            available_cb.pop_back();
            cbs[ns]->aio_fildes = req->fd;
            cbs[ns]->aio_data = (uint64_t) (req);
            cbs[ns]->aio_buf = (uint64_t) (req->mem_base);
            cbs[ns]->aio_offset = req->file_off;
            cbs[ns]->aio_nbytes = req->nbytes;
            ns++;
          }
          break;
        default:
          assert(0);
      }
      assert(ns == nr);
      int ret = io_submit(ctx, ns, cbs);
      if (ret < 0) {
        perror("io_submit error");
      }
      return ret;
    }

    void FileChannel::pull()
    {
      int nr = io_getevents(ctx, 0, capacity, events, NULL);
      if (nr < 0)
        perror("io_getevents error");
      for (int i = 0; i < nr; i++) {
        Request* req = (Request*) events[i].data;
        struct iocb* ret_cb = (struct iocb*) events[i].obj;
        available_cb.push_back(ret_cb);
        assert(events[i].res == (int64_t)ret_cb->aio_nbytes);
        req->xd->notify_request_read_done(req);
        req->xd->notify_request_write_done(req);
      }
    }

    long FileChannel::available()
    {
      return available_cb.size();
    }

#ifdef USE_DISK
      template class FileXferDes<0>;
      template class FileXferDes<1>;
      template class FileXferDes<2>;
      template class FileXferDes<3>;
#endif
  } // namespace LowLevel
} // namespace LegionRuntime
