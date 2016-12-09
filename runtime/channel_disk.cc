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
      size_t total_field_size = 0;
      for (unsigned i = 0; i < oas_vec.size(); i++) {
        total_field_size += oas_vec[i].size;
      }
      bytes_total = total_field_size * domain.get_volume();
      pre_bytes_write = (pre_xd_guid == XFERDES_NO_GUID) ? bytes_total : 0;
      if (DIM == 0) {
        li = NULL;
        // index space instances use 1D linearizations for translation
        me = new MaskEnumerator(domain.get_index_space(), src_buf.linearization.get_mapping<1>(),
                                dst_buf.linearization.get_mapping<1>(), order,
                                src_buf.is_ib, dst_buf.is_ib);
      } else {
        li = new Layouts::GenericLayoutIterator<DIM>(domain.get_rect<DIM>(), src_buf.linearization.get_mapping<DIM>(),
                                                     dst_buf.linearization.get_mapping<DIM>(), order);
        me = NULL;
      }
      offset_idx = 0;

      switch (kind) {
        case XferDes::XFER_FILE_READ:
        {
          ID src_id(inst);
          unsigned src_index = id.instance.inst_idx();
          fd = (FileMemory*)src_mem_impl->get_file_des(src_index);
          channel = channel_manager->get_file_read_channel();
          buf_base = (const char*) dst_mem_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
          assert(src_mem_impl->kind == MemoryImpl::MKIND_FILE);
          FileReadRequest* file_read_reqs
	    = (FileReadRequest*) calloc(max_nr, sizeof(FileReadRequest));
          for (int i = 0; i < max_nr; i++) {
            file_read_reqs[i].xd = this;
            available_reqs.push(&file_read_reqs[i]);
          }
          requests = file_read_reqs;
          break;
        }
        case XferDes::XFER_FILE_WRITE:
        {
          ID dst_id(inst);
          unsigned dst_index = id.instance.inst_idx();
          fd = (FileMemory*)dst_mem_impl->get_file_des(dst_index);
          channel = channel_manager->get_file_write_channel();
          buf_base = (const char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
          assert(dst_mem_impl->kind == MemoryImpl::MKIND_FILE);
          FileWriteRequest* file_write_reqs
	    = (FileWriteRequest*) calloc(max_nr, sizeof(FileWriteRequest));
          for (int i = 0; i < max_nr; i++) {
            file_write_reqs[i].xd = this;
            available_reqs.push(&file_write_reqs[i]);
          }
          requests = file_write_reqs;
          break;
        }
        default:
          assert(0);
      }
    }

    template<unsigned DIM>
    long FileXferDes<DIM>::get_requests(Request** requests, long nr)
    {
      long idx = 0;
      while (idx < nr && !available_reqs.empty() && offset_idx < oas_vec.size()) {
        off_t src_start, dst_start;
        size_t nbytes;
        if (DIM == 0) {
          simple_get_mask_request(src_start, dst_start, nbytes, me, offset_idx, min(available_reqs.size(), nr - idx));
        } else {
          simple_get_request<DIM>(src_start, dst_start, nbytes, li, offset_idx, min(available_reqs.size(), nr - idx));
        }
        if (nbytes == 0)
          break;
        while (nbytes > 0) {
          size_t req_size = nbytes;
          if (src_buf.is_ib) {
            src_start = src_start % src_buf.buf_size;
            req_size = umin(req_size, src_buf.buf_size - src_start);
          }
          if (dst_buf.is_ib) {
            dst_start = dst_start % dst_buf.buf_size;
            req_size = umin(req_size, dst_buf.buf_size - dst_start);
          }
          requests[idx] = available_reqs.front();
          available_reqs.pop();
          requests[idx]->is_read_done = false;
          requests[idx]->is_write_done = false;
          switch (kind) {
            case XferDes::XFER_FILE_READ:
            {
              FileReadRequest* file_read_req = (FileReadRequest*) requests[idx];
              file_read_req->fd = fd;
              file_read_req->src_offset = src_buf.alloc_offset + src_start;
              file_read_req->dst_buf = (uint64_t)(buf_base + dst_start);
              file_read_req->nbytes = req_size;
              break;
            }
            case XferDes::XFER_FILE_WRITE:
            {
              FileWriteRequest* disk_write_req = (FileWriteRequest*) requests[idx];
              file_write_req->fd = fd;
              file_write_req->src_buf = (uint64_t)(buf_base + src_start);
              file_write_req->dst_offset = dst_buf.alloc_offset + dst_start;
              file_write_req->nbytes = req_size;
              break;
            }
            default:
              assert(0);
          }
          src_start += req_size;
          dst_start += req_size;
          nbytes -= req_size;
          idx ++;
        }
      }
      return idx;
    }

    template<unsigned DIM>
    void FileXferDes<DIM>::notify_request_read_done(Request* req)
    {
      req->is_read_done = true;
      int64_t offset;
      uint64_t size;
      switch(kind) {
        case XferDes::XFER_FILE_READ:
          offset = ((FileReadRequest*)req)->src_offset - src_buf.alloc_offset;
          size = ((FileReadRequest*)req)->nbytes;
          break;
        case XferDes::XFER_FILE_WRITE:
          offset = ((FileWriteRequest*)req)->src_buf - (uint64_t) buf_base;
          size = ((FileWriteRequest*)req)->nbytes;
          break;
        default:
          assert(0);
      }
      simple_update_bytes_read(offset, size);
    }

    template<unsigned DIM>
    void FileXferDes<DIM>::notify_request_write_done(Request* req)
    {
      req->is_write_done = true;
      int64_t offset;
      uint64_t size;
      switch(kind) {
        case XferDes::XFER_FILE_READ:
          offset = ((FileReadRequest*)req)->dst_buf - (uint64_t) buf_base;
          size = ((FileReadRequest*)req)->nbytes;
          break;
        case XferDes::XFER_FILE_WRITE:
          offset = ((FileWriteRequest*)req)->dst_offset - dst_buf.alloc_offset;
          size = ((FileWriteRequest*)req)->nbytes;
          break;
        default:
          assert(0);
      }
      simple_update_bytes_write(offset, size);
      available_reqs.push(req);
      //printf("bytes_write = %lu, bytes_total = %lu\n", bytes_write, bytes_total);
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
          while (ns < nr && !available_cb.empty()) {
            FileReadRequest* file_read_req = (FileReadRequest*) requests[ns];
            cbs[ns] = available_cb.back();
            available_cb.pop_back();
            cbs[ns]->aio_fildes = file_read_req->fd;
            cbs[ns]->aio_data = (uint64_t) (file_read_req);
            cbs[ns]->aio_buf = file_read_req->dst_buf;
            cbs[ns]->aio_offset = file_read_req->src_offset;
            cbs[ns]->aio_nbytes = file_read_req->nbytes;
            ns++;
          }
          break;
        case XferDes::XFER_FILE_WRITE:
          while (ns < nr && !available_cb.empty()) {
            FileWriteRequest* file_write_req = (FileWriteRequest*) requests[ns];
            cbs[ns] = available_cb.back();
            available_cb.pop_back();
            cbs[ns]->aio_fildes = file_write_req->fd;
            cbs[ns]->aio_data = (uint64_t) (file_write_req);
            cbs[ns]->aio_buf = file_write_req->src_buf;
            cbs[ns]->aio_offset = file_write_req->dst_offset;
            cbs[ns]->aio_nbytes = file_write_req->nbytes;
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
  } // namespace LowLevel
} // namespace LegionRuntime
