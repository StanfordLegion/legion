/* Copyright 2015 Stanford University
 * Copyright 2015 Los Alamos National Laboratory
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

#include "channel.h"

namespace LegionRuntime {
  namespace LowLevel {
    Logger::Category log_new_dma("new_dma");
#ifdef USE_DISK 
    inline int io_setup(unsigned nr, aio_context_t *ctxp)
      {
        return syscall(__NR_io_setup, nr, ctxp);
      }

      inline int io_destroy(aio_context_t ctx)
      {
        return syscall(__NR_io_destroy, ctx);
      }

      inline int io_submit(aio_context_t ctx, long nr, struct iocb **iocbpp)
      {
        return syscall(__NR_io_submit, ctx, nr, iocbpp);
      }

      inline int io_getevents(aio_context_t ctx, long min_nr, long max_nr,
                              struct io_event *events, struct timespec *timeout)
      {
        return syscall(__NR_io_getevents, ctx, min_nr, max_nr, events, timeout);
      }
#endif /*USE_DISK*/

      // TODO: currently we use dma_all_gpus to track the set of GPU* created
#ifdef USE_CUDA
      std::vector<GPU*> dma_all_gpus;
#endif
      // we use a single queue for all xferDes
      static XferDesQueue *xferDes_queue = 0;

      // we use a single manager to organize all channels
      static ChannelManager *channel_manager = 0;

      static inline int max(int a, int b) { return (a < b) ? b : a; }
      static inline size_t umin(size_t a, size_t b) { return (a < b) ? a : b; }

      static inline bool scatter_ib(off_t start, size_t nbytes, size_t buf_size)
      {
        return (nbytes > 0) && (start / buf_size < (start + nbytes - 1) / buf_size);
      }

      void XferDes::mark_completed() {
        // notify owning DmaRequest upon completion of this XferDes
        //printf("complete XD = %lu\n", guid);
        if (launch_node == gasnet_mynode()) {
          complete_fence->mark_finished();
        } else {
          NotifyXferDesCompleteMessage::send_request(launch_node, complete_fence);
        }
      }


      template<unsigned DIM>
      bool XferDes::simple_get_request(off_t &src_start, off_t &dst_start, size_t &nbytes,
                              Layouts::GenericLayoutIterator<DIM>* li,
                              int &offset_idx, int available_slots)
      {
        assert(offset_idx < oas_vec.size());
        assert(li->any_left());
        nbytes = 0;
        int src_idx, dst_idx;
        // cannot exceed the max_req_size
        int todo = min(max_req_size / oas_vec[offset_idx].size, li->continuous_steps(src_idx, dst_idx));
        int src_in_block = src_buf.block_size - src_idx % src_buf.block_size;
        int dst_in_block = dst_buf.block_size - dst_idx % dst_buf.block_size;
        todo = min(todo, min(src_in_block, dst_in_block));
        src_start = calc_mem_loc(0, oas_vec[offset_idx].src_offset, oas_vec[offset_idx].size,
                                 src_buf.elmt_size, src_buf.block_size, src_idx);
        dst_start = calc_mem_loc(0, oas_vec[offset_idx].dst_offset, oas_vec[offset_idx].size,
                                 dst_buf.elmt_size, dst_buf.block_size, dst_idx);
        bool scatter_src_ib = false, scatter_dst_ib = false;
        // make sure we have source data ready
        if (src_buf.is_ib) {
          todo = min(todo, max(0, pre_bytes_write - src_start) / oas_vec[offset_idx].size);
          scatter_src_ib = scatter_ib(src_start, todo * oas_vec[offset_idx].size, src_buf.buf_size);
        }
        // make sure there are enough space in destination
        if (dst_buf.is_ib) {
          todo = min(todo, max(0, next_bytes_read + dst_buf.buf_size - dst_start) / oas_vec[offset_idx].size);
          scatter_dst_ib = scatter_ib(dst_start, todo * oas_vec[offset_idx].size, dst_buf.buf_size);
        }
        if((scatter_src_ib && scatter_dst_ib && available_slots < 3)
        ||((scatter_src_ib || scatter_dst_ib) && available_slots < 2))
          return false; // case we don't have enough slots

        nbytes = todo * oas_vec[offset_idx].size;
        li->move(todo);
        if (!li->any_left()) {
          li->reset();
          offset_idx ++;
        }
        return true;
      }

      template<unsigned DIM>
      bool XferDes::simple_get_request(
                    off_t &src_start, off_t &dst_start, size_t &nbytes,
                    Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >* &dsi,
                    Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >* &dso,
                    Rect<1> &irect, Rect<1> &orect,
                    int &done, int &offset_idx, int &block_start, int &total, int available_slots,
                    bool disable_batch)
      {
        src_start = calc_mem_loc(0, oas_vec[offset_idx].src_offset, oas_vec[offset_idx].size,
                                 src_buf.elmt_size, src_buf.block_size, done + irect.lo);
        dst_start = calc_mem_loc(0, oas_vec[offset_idx].dst_offset, oas_vec[offset_idx].size,
                                 dst_buf.elmt_size, dst_buf.block_size, done + orect.lo);
        nbytes = 0;
        bool scatter_src_ib = false, scatter_dst_ib = false;
        while (true) {
          // check to see if we can generate next request
          int src_in_block = src_buf.block_size - (done + irect.lo) % src_buf.block_size;
          int dst_in_block = dst_buf.block_size - (done + orect.lo) % dst_buf.block_size;
          int todo = min((max_req_size - nbytes) / oas_vec[offset_idx].size, min(total - done, min(src_in_block, dst_in_block)));
          // make sure we have source data ready
          if (src_buf.is_ib) {
            todo = min(todo, max(0, src_buf.alloc_offset + pre_bytes_write - (src_start + nbytes)) / oas_vec[offset_idx].size);
            scatter_src_ib = scatter_src_ib || scatter_ib(src_start, nbytes + todo * oas_vec[offset_idx].size, src_buf.buf_size);
          }
          // make sure there are enough space in destination
          if (dst_buf.is_ib) {
            todo = min(todo, max(0, dst_buf.alloc_offset + next_bytes_read + dst_buf.buf_size - (dst_start + nbytes)) / oas_vec[offset_idx].size);
            scatter_dst_ib = scatter_dst_ib || scatter_ib(dst_start, nbytes + todo * oas_vec[offset_idx].size, dst_buf.buf_size);
          }
          if((scatter_src_ib && scatter_dst_ib && available_slots < 3)
          ||((scatter_src_ib || scatter_dst_ib) && available_slots < 2))
            break;
          //printf("min(%d, %d, %d) \n =", (int)(max_req_size - nbytes) / oas_vec[offset_idx].size, total - done, min(src_in_block, dst_in_block));
          //printf("todo = %d, size = %d\n", todo, oas_vec[offset_idx].size);
          nbytes += todo * oas_vec[offset_idx].size;
          // see if we can batch more
          if (!disable_batch && todo == src_in_block && todo == dst_in_block && offset_idx + 1 < oas_vec.size()
          && src_buf.block_size == dst_buf.block_size && todo + done >= src_buf.block_size
          && oas_vec[offset_idx + 1].src_offset == oas_vec[offset_idx].src_offset + oas_vec[offset_idx].size
          && oas_vec[offset_idx + 1].dst_offset == oas_vec[offset_idx].dst_offset + oas_vec[offset_idx].size) {
            done = block_start;
            offset_idx += 1;
          }
          else {
            done += todo;
            break;
          }
        }

        if (nbytes > 0 &&
        (((done + irect.lo) % src_buf.block_size == 0 && done + irect.lo > block_start && order == XferOrder::SRC_FIFO)
        ||((done + orect.lo) % dst_buf.block_size == 0 && done + orect.lo > block_start && order == XferOrder::DST_FIFO)
        || (done == total))) {
          offset_idx ++;
          if (offset_idx < oas_vec.size()) {
            switch (order) {
              case XferOrder::SRC_FIFO:
                done = block_start - irect.lo;
                break;
              case XferOrder::DST_FIFO:
                done = block_start - orect.lo;
                break;
              case XferOrder::ANY_ORDER:
                assert(0);
                break;
              default:
                assert(0);
            }
          }
          else {
            int new_block_start;
            switch (order) {
              case XferOrder::SRC_FIFO:
                new_block_start = block_start + src_buf.block_size;
                new_block_start = new_block_start - new_block_start % src_buf.block_size;
                block_start = new_block_start;
                done = block_start - irect.lo;
                offset_idx = 0;
                if (block_start > irect.hi) {
                  dso->step();
                  if (dso->any_left) {
                    dsi->step();
                    if (dsi->any_left) {
                      delete dso;
                      dso = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dsi->subrect, *(dst_buf.linearization.get_mapping<DIM>()));
                    }
                  }
                  if (dso->any_left && dsi->any_left) {
                    Rect<DIM> subrect_check;
                    irect = src_buf.linearization.get_mapping<DIM>()->image_dense_subrect(dso->subrect, subrect_check);
                    orect = dso->image;
                    done = 0; offset_idx = 0; block_start = irect.lo; total = irect.hi[0] - irect.lo[0] + 1;
                  }
                }
                break;
              case XferOrder::DST_FIFO:
                new_block_start = block_start + dst_buf.block_size;
                new_block_start = new_block_start - new_block_start % dst_buf.block_size;
                block_start = new_block_start;
                done = block_start - orect.lo;
                offset_idx = 0;
                if (block_start > orect.hi) {
                  dsi->step();
                  if (!dsi->any_left) {
                    dso->step();
                    if (dso->any_left) {
                      delete dsi;
                      dsi = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dso->subrect, *(src_buf.linearization.get_mapping<DIM>()));
                    }
                  }
                  if (dso->any_left && dsi->any_left) {
                    Rect<DIM> subrect_check;
                    orect = dst_buf.linearization.get_mapping<DIM>()->image_dense_subrect(dsi->subrect, subrect_check);
                    irect = dsi->image;
                    done = 0; offset_idx = 0; block_start = orect.lo; total = orect.hi[0] - orect.lo[0] + 1;
                  }
                }
                break;
              case XferOrder::ANY_ORDER:
                assert(0);
                break;
              default:
                assert(0);
            }
          }
        }
        return (nbytes > 0);
      }

      inline void XferDes::simple_update_bytes_read(int64_t offset, uint64_t size)
      {
        //printf("update_read: offset = %ld, size = %lu, pre = %ld, next = %ld\n", offset, size, pre_xd_guid, next_xd_guid);
        if (pre_xd_guid != XFERDES_NO_GUID) {
          bool update = false;
          if ((int64_t)(bytes_read % src_buf.buf_size) == offset) {
            bytes_read += size;
            update = true;
          }
          else {
            //printf("[%lx] insert: key = %ld, value = %lu\n", guid, offset, size);
            segments_read[offset] = size;
          }
          std::map<int64_t, uint64_t>::iterator it;
          while (true) {
            it = segments_read.find(bytes_read % src_buf.buf_size);
            if (it == segments_read.end())
              break;
            bytes_read += it->second;
            update = true;
            //printf("[%lx] erase: key = %ld, value = %lu\n", guid, it->first, it->second);
            segments_read.erase(it);
          }
          if (update) {
            xferDes_queue->update_next_bytes_read(pre_xd_guid, bytes_read);
          }
        }
        else {
          bytes_read += size;
        }
      }

      inline void XferDes::simple_update_bytes_write(int64_t offset, uint64_t size)
      {
        //printf("update_write: offset = %ld, size = %lu, pre = %ld, next = %ld\n", offset, size, pre_xd_guid, next_xd_guid);
        if (next_xd_guid != XFERDES_NO_GUID) {
          bool update = false;
          if ((int64_t)(bytes_write % dst_buf.buf_size) == offset) {
            bytes_write += size;
            update = true;
          }
          else {
            segments_write[offset] = size;
          }
          std::map<int64_t, uint64_t>::iterator it;
          while (true) {
            it = segments_write.find(bytes_write % dst_buf.buf_size);
            if (it == segments_write.end())
              break;
            bytes_write += it->second;
            update = true;
            segments_write.erase(it);
          }
          if (update) {
            xferDes_queue->update_pre_bytes_write(next_xd_guid, bytes_write);
          }
        }
        else {
          bytes_write += size;
        }
        //printf("[%d] offset = %ld, bytes_writes[%lx]: %ld\n", gasnet_mynode(), offset, guid, bytes_write);
      }

      template<unsigned DIM>
      MemcpyXferDes<DIM>::MemcpyXferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                                        XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                                        const Buffer& _src_buf, const Buffer& _dst_buf,
                                        const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                                        uint64_t _max_req_size, long max_nr, int _priority,
                                        XferOrder::Type _order, XferDesFence* _complete_fence)
        : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid, _next_xd_guid, _src_buf, _dst_buf,
                  _domain, _oas_vec, _max_req_size, _priority, _order, XferDes::XFER_MEM_CPY, _complete_fence)
      {
        MemoryImpl* src_mem_impl = get_runtime()->get_memory_impl(_src_buf.memory);
        MemoryImpl* dst_mem_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
        assert(src_mem_impl->kind == MemoryImpl::MKIND_SYSMEM || src_mem_impl->kind == MemoryImpl::MKIND_ZEROCOPY);
        assert(dst_mem_impl->kind == MemoryImpl::MKIND_SYSMEM || dst_mem_impl->kind == MemoryImpl::MKIND_ZEROCOPY);
        channel = channel_manager->get_memcpy_channel();
        src_buf_base = (char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
        dst_buf_base = (char*) dst_mem_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
        size_t total_field_size = 0;
        for (int i = 0; i < oas_vec.size(); i++) {
          total_field_size += oas_vec[i].size;
        }
        bytes_total = total_field_size * domain.get_volume();
        pre_bytes_write = (pre_xd_guid == XFERDES_NO_GUID) ? bytes_total : 0;
        li = new Layouts::GenericLayoutIterator<DIM>(domain.get_rect<DIM>(), src_buf.linearization.get_mapping<DIM>(),
                                                     dst_buf.linearization.get_mapping<DIM>(), order);
        offset_idx = 0;
        requests = (MemcpyRequest*) calloc(max_nr, sizeof(MemcpyRequest));
        for (int i = 0; i < max_nr; i++) {
          requests[i].xd = this;
          available_reqs.push(&requests[i]);
        }
      }

      template<unsigned DIM>
      long MemcpyXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        MemcpyRequest** mem_cpy_reqs = (MemcpyRequest**) requests;
        long idx = 0;
        while (idx < nr && !available_reqs.empty() && offset_idx < oas_vec.size()) {
          off_t src_start, dst_start;
          size_t nbytes;
          simple_get_request<DIM>(src_start, dst_start, nbytes, li, offset_idx, min(available_reqs.size(), nr - idx));
          //printf("[MemcpyXferDes] offset_idx = %d, oas_vec.size() = %lu, nbytes = %lu\n", offset_idx, oas_vec.size(), nbytes);
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
            mem_cpy_reqs[idx] = (MemcpyRequest*) available_reqs.front();
            available_reqs.pop();
            //printf("[MemcpyXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start, dst_start, nbytes);
            mem_cpy_reqs[idx]->is_read_done = false;
            mem_cpy_reqs[idx]->is_write_done = false;
            mem_cpy_reqs[idx]->src_buf = (char*)(src_buf_base + src_start);
            mem_cpy_reqs[idx]->dst_buf = (char*)(dst_buf_base + dst_start);
            mem_cpy_reqs[idx]->nbytes = req_size;
            src_start += req_size; // here we don't have to mod src_buf.buf_size since it will be performed in next loop
            dst_start += req_size; //
            nbytes -= req_size;
            idx++;
          }
        }
        return idx;
      }

      template<unsigned DIM>
      void MemcpyXferDes<DIM>::notify_request_read_done(Request* req)
      {
        req->is_read_done = true;
        MemcpyRequest* mc_req = (MemcpyRequest*) req;
        simple_update_bytes_read(mc_req->src_buf - src_buf_base, mc_req->nbytes);
      }

      template<unsigned DIM>
      void MemcpyXferDes<DIM>::notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        MemcpyRequest* mc_req = (MemcpyRequest*) req;
        simple_update_bytes_write(mc_req->dst_buf - dst_buf_base, mc_req->nbytes);
        available_reqs.push(req);
      }

      template<unsigned DIM>
      void MemcpyXferDes<DIM>::flush()
      {
      }

      template<unsigned DIM>
      GASNetXferDes<DIM>::GASNetXferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                                        XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                                        const Buffer& _src_buf, const Buffer& _dst_buf,
                                        const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                                        uint64_t _max_req_size, long max_nr, int _priority,
                                        XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence)
        : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid, _next_xd_guid, _src_buf, _dst_buf,
                  _domain, _oas_vec, _max_req_size, _priority, _order, _kind, _complete_fence)
      {
        MemoryImpl* src_mem_impl = get_runtime()->get_memory_impl(_src_buf.memory);
        MemoryImpl* dst_mem_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
        size_t total_field_size = 0;
        for (int i = 0; i < oas_vec.size(); i++) {
          total_field_size += oas_vec[i].size;
        }
        bytes_total = total_field_size * domain.get_volume();
        pre_bytes_write = (pre_xd_guid == XFERDES_NO_GUID) ? bytes_total : 0;
        li = new Layouts::GenericLayoutIterator<DIM>(domain.get_rect<DIM>(), src_buf.linearization.get_mapping<DIM>(),
                                                     dst_buf.linearization.get_mapping<DIM>(), order);
        offset_idx = 0;

        switch (kind) {
          case XferDes::XFER_GASNET_READ:
          {
            channel = channel_manager->get_gasnet_read_channel();
            buf_base = (const char*) dst_mem_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
            assert(src_mem_impl->kind == MemoryImpl::MKIND_GLOBAL);
            assert(dst_mem_impl->kind == MemoryImpl::MKIND_SYSMEM || dst_mem_impl->kind == MemoryImpl::MKIND_ZEROCOPY);
            GASNetReadRequest* gasnet_read_reqs = (GASNetReadRequest*) calloc(max_nr, sizeof(GASNetReadRequest));
            for (int i = 0; i < max_nr; i++) {
              gasnet_read_reqs[i].xd = this;
              available_reqs.push(&gasnet_read_reqs[i]);
            }
            requests = gasnet_read_reqs;
            break;
          }
          case XferDes::XFER_GASNET_WRITE:
          {
            channel = channel_manager->get_gasnet_write_channel();
            buf_base = (const char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
            assert(src_mem_impl->kind == MemoryImpl::MKIND_SYSMEM || src_mem_impl->kind == MemoryImpl::MKIND_ZEROCOPY);
            assert(dst_mem_impl->kind == MemoryImpl::MKIND_GLOBAL);
            GASNetWriteRequest* gasnet_write_reqs = (GASNetWriteRequest*) calloc(max_nr, sizeof(GASNetWriteRequest));
            for (int i = 0; i < max_nr; i++) {
              gasnet_write_reqs[i].xd = this;
              available_reqs.push(&gasnet_write_reqs[i]);
            }
            requests = gasnet_write_reqs;
            break;
          }
          default:
            assert(false);
        }
      }

      template<unsigned DIM>
      long GASNetXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        long idx = 0;
        while (idx < nr && !available_reqs.empty() && offset_idx < oas_vec.size()) {
          off_t src_start, dst_start;
          size_t nbytes;
          simple_get_request<DIM>(src_start, dst_start, nbytes, li, offset_idx, min(available_reqs.size(), nr - idx));
          //printf("done = %d, offset_idx = %d\n", done, offset_idx);
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
              case XferDes::XFER_GASNET_READ:
              {
                //printf("[GASNetReadXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start, dst_start, req_size);
                GASNetReadRequest* gasnet_read_req = (GASNetReadRequest*) requests[idx];
                gasnet_read_req->src_offset = src_buf.alloc_offset + src_start;
                gasnet_read_req->dst_buf = (char*)(buf_base + dst_start);
                gasnet_read_req->nbytes = req_size;
                break;
              }
              case XferDes::XFER_GASNET_WRITE:
              {
                //printf("[GASNetWriteXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start, dst_start, req_size);
                GASNetWriteRequest* gasnet_write_req = (GASNetWriteRequest*) requests[idx];
                gasnet_write_req->src_buf = (char*)(buf_base + src_start);
                gasnet_write_req->dst_offset = dst_buf.alloc_offset + dst_start;
                gasnet_write_req->nbytes = req_size;
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
      void GASNetXferDes<DIM>::notify_request_read_done(Request* req)
      {
        req->is_read_done = true;
        int64_t offset;
        uint64_t size;
        switch(kind) {
          case XferDes::XFER_GASNET_READ:
            offset = ((GASNetReadRequest*)req)->src_offset - src_buf.alloc_offset;
            size = ((GASNetReadRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GASNET_WRITE:
            offset = ((GASNetWriteRequest*)req)->src_buf - buf_base;
            size = ((GASNetWriteRequest*)req)->nbytes;
            break;
          default:
            assert(0);
        }
        simple_update_bytes_read(offset, size);
      }

      template<unsigned DIM>
      void GASNetXferDes<DIM>::notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        int64_t offset;
        uint64_t size;
        switch(kind) {
          case XferDes::XFER_GASNET_READ:
            offset = ((GASNetReadRequest*)req)->dst_buf - buf_base;
            size = ((GASNetReadRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GASNET_WRITE:
            offset = ((GASNetWriteRequest*)req)->dst_offset - dst_buf.alloc_offset;
            size = ((GASNetWriteRequest*)req)->nbytes;
            break;
          default:
            assert(0);
        }
        simple_update_bytes_write(offset, size);
        available_reqs.push(req);
      }

      template<unsigned DIM>
      void GASNetXferDes<DIM>::flush()
      {
      }

      template<unsigned DIM>
      RemoteWriteXferDes<DIM>::RemoteWriteXferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                                                  XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                                                  const Buffer& _src_buf, const Buffer& _dst_buf,
                                                  const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                                                  uint64_t _max_req_size, long max_nr, int _priority,
                                                  XferOrder::Type _order, XferDesFence* _complete_fence)
        : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid, _next_xd_guid, _src_buf, _dst_buf,
                  _domain, _oas_vec, _max_req_size, _priority, _order, XferDes::XFER_REMOTE_WRITE, _complete_fence)
      {
        MemoryImpl* src_mem_impl = get_runtime()->get_memory_impl(_src_buf.memory);
        dst_mem_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
        assert(src_mem_impl->kind == MemoryImpl::MKIND_SYSMEM || src_mem_impl->kind == MemoryImpl::MKIND_ZEROCOPY);
        // make sure dst buffer is registered memory
        assert(dst_mem_impl->kind == MemoryImpl::MKIND_RDMA);
        channel = channel_manager->get_remote_write_channel();
        src_buf_base = (const char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
        // Note that we could use get_direct_ptr to get dst_buf_base, since it always returns 0
        dst_buf_base = ((const char*)((Realm::RemoteMemory*)dst_mem_impl)->regbase) + dst_buf.alloc_offset;
        size_t total_field_size = 0;
        for (int i = 0; i < oas_vec.size(); i++) {
          total_field_size += oas_vec[i].size;
        }
        bytes_total = total_field_size * domain.get_volume();
        pre_bytes_write = (pre_xd_guid==XFERDES_NO_GUID) ? bytes_total : 0;
        li = new Layouts::GenericLayoutIterator<DIM>(domain.get_rect<DIM>(),
                                                     src_buf.linearization.get_mapping<DIM>(),
                                                     dst_buf.linearization.get_mapping<DIM>(),
                                                     order);
        offset_idx = 0;
        requests = (RemoteWriteRequest*) calloc(max_nr, sizeof(RemoteWriteRequest));
        for (int i = 0; i < max_nr; i++) {
          requests[i].xd = this;
          requests[i].dst_node = ID(_dst_buf.memory).node();
          available_reqs.push(&requests[i]);
        }
      }

      template<unsigned DIM>
      long RemoteWriteXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        pthread_mutex_lock(&xd_lock);
        long idx = 0;
        while (idx < nr && !available_reqs.empty() && offset_idx < oas_vec.size()) {
          off_t src_start, dst_start;
          size_t nbytes;
          simple_get_request<DIM>(src_start, dst_start, nbytes, li, offset_idx, min(available_reqs.size(), nr - idx));
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
            assert(!available_reqs.empty());
            requests[idx] = available_reqs.front();
            available_reqs.pop();
            requests[idx]->is_read_done = false;
            requests[idx]->is_write_done = false;
            RemoteWriteRequest* req = (RemoteWriteRequest*) requests[idx];
            req->src_buf = (char*)(src_buf_base + src_start);
            // dst_offset count from the beginning of registered memory
            req->dst_buf = (char*)(dst_buf_base + dst_start);
            req->nbytes = req_size;
            src_start += req_size; // here we don't have to mod src_buf.buf_size since it will be performed in next loop
            dst_start += req_size; //
            nbytes -= req_size;
            idx ++;
          }
        }
        pthread_mutex_unlock(&xd_lock);
        return idx;
      }

      template<unsigned DIM>
      void RemoteWriteXferDes<DIM>::notify_request_read_done(Request* req)
      {
        pthread_mutex_lock(&xd_lock);
        req->is_read_done = true;
        int64_t offset = ((RemoteWriteRequest*)req)->src_buf - src_buf_base;
        uint64_t size = ((RemoteWriteRequest*)req)->nbytes;
        simple_update_bytes_read(offset, size);
        pthread_mutex_unlock(&xd_lock);
      }

      template<unsigned DIM>
      void RemoteWriteXferDes<DIM>::notify_request_write_done(Request* req)
      {
        pthread_mutex_lock(&xd_lock);
        req->is_write_done = true;
        int64_t offset = ((RemoteWriteRequest*)req)->dst_buf - dst_buf_base;
        uint64_t size = ((RemoteWriteRequest*)req)->nbytes;
        simple_update_bytes_write(offset, size);
        available_reqs.push(req);
        pthread_mutex_unlock(&xd_lock);
      }

      template<unsigned DIM>
      void RemoteWriteXferDes<DIM>::flush()
      {
        pthread_mutex_lock(&xd_lock);
        pthread_mutex_unlock(&xd_lock);
      }

#ifdef USE_DISK
      template<unsigned DIM>
      DiskXferDes<DIM>::DiskXferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                                    XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                                    const Buffer& _src_buf, const Buffer& _dst_buf,
                                    const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                                    uint64_t _max_req_size, long max_nr, int _priority,
                                    XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence)
        : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid, _next_xd_guid, _src_buf, _dst_buf,
                  _domain, _oas_vec, _max_req_size, _priority, _order, _kind, _complete_fence)
      {
        MemoryImpl* src_mem_impl = get_runtime()->get_memory_impl(_src_buf.memory);
        MemoryImpl* dst_mem_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
        size_t total_field_size = 0;
        for (int i = 0; i < oas_vec.size(); i++) {
          total_field_size += oas_vec[i].size;
        }
        bytes_total = total_field_size * domain.get_volume();
        pre_bytes_write = (pre_xd_guid == XFERDES_NO_GUID) ? bytes_total : 0;
        li = new Layouts::GenericLayoutIterator<DIM>(domain.get_rect<DIM>(), src_buf.linearization.get_mapping<DIM>(),
                                                     dst_buf.linearization.get_mapping<DIM>(), order);
        offset_idx = 0;

        switch (kind) {
          case XferDes::XFER_DISK_READ:
          {
            channel = channel_manager->get_disk_read_channel();
            fd = ((Realm::DiskMemory*)src_mem_impl)->fd;
            buf_base = (const char*) dst_mem_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
            assert(src_mem_impl->kind == MemoryImpl::MKIND_DISK);
            assert(dst_mem_impl->kind == MemoryImpl::MKIND_SYSMEM || dst_mem_impl->kind == MemoryImpl::MKIND_ZEROCOPY);
            DiskReadRequest* disk_read_reqs = (DiskReadRequest*) calloc(max_nr, sizeof(DiskReadRequest));
            for (int i = 0; i < max_nr; i++) {
              disk_read_reqs[i].xd = this;
              available_reqs.push(&disk_read_reqs[i]);
            }
            requests = disk_read_reqs;
            break;
          }
          case XferDes::XFER_DISK_WRITE:
          {
            channel = channel_manager->get_disk_write_channel();
            fd = ((Realm::DiskMemory*)dst_mem_impl)->fd;
            buf_base = (const char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
            assert(src_mem_impl->kind == MemoryImpl::MKIND_SYSMEM || src_mem_impl->kind == MemoryImpl::MKIND_ZEROCOPY);
            assert(dst_mem_impl->kind == MemoryImpl::MKIND_DISK);
            DiskWriteRequest* disk_write_reqs = (DiskWriteRequest*) calloc(max_nr, sizeof(DiskWriteRequest));
            for (int i = 0; i < max_nr; i++) {
              disk_write_reqs[i].xd = this;
              available_reqs.push(&disk_write_reqs[i]);
            }
            requests = disk_write_reqs;
            break;
          }
          default:
            assert(0);
        }
      }

      template<unsigned DIM>
      long DiskXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        long idx = 0;
        while (idx < nr && !available_reqs.empty() && offset_idx < oas_vec.size()) {
          off_t src_start, dst_start;
          size_t nbytes;
          simple_get_request<DIM>(src_start, dst_start, nbytes, li, offset_idx, min(available_reqs.size(), nr - idx));
          //printf("done = %d, offset_idx = %d\n", done, offset_idx);
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
              case XferDes::XFER_DISK_READ:
              {
                //printf("[DiskReadXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start, dst_start, req_size);
                DiskReadRequest* disk_read_req = (DiskReadRequest*) requests[idx];
                disk_read_req->fd = fd;
                disk_read_req->src_offset = src_buf.alloc_offset + src_start;
                disk_read_req->dst_buf = (uint64_t)(buf_base + dst_start);
                disk_read_req->nbytes = req_size;
                break;
              }
              case XferDes::XFER_DISK_WRITE:
              {
                //printf("[DiskWriteXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start, dst_start, req_size);
                DiskWriteRequest* disk_write_req = (DiskWriteRequest*) requests[idx];
                disk_write_req->fd = fd;
                disk_write_req->src_buf = (uint64_t)(buf_base + src_start);
                disk_write_req->dst_offset = dst_buf.alloc_offset + dst_start;
                disk_write_req->nbytes = req_size;
                break;
              }
              default:
                assert(0);
            }
            src_start += req_size; // here we don't have to mod src_buf.buf_size since it will be performed in next loop
            dst_start += req_size; //
            nbytes -= req_size;
            idx ++;
          }
        }
        return idx;
      }

      template<unsigned DIM>
      void DiskXferDes<DIM>::notify_request_read_done(Request* req)
      {
        req->is_read_done = true;
        int64_t offset;
        uint64_t size;
        switch(kind) {
          case XferDes::XFER_DISK_READ:
            offset = ((DiskReadRequest*)req)->src_offset - src_buf.alloc_offset;
            size = ((DiskReadRequest*)req)->nbytes;
            break;
          case XferDes::XFER_DISK_WRITE:
            offset = ((DiskWriteRequest*)req)->src_buf - (uint64_t) buf_base;
            size = ((DiskWriteRequest*)req)->nbytes;
            break;
          default:
            assert(0);
        }
        simple_update_bytes_read(offset, size);
      }

      template<unsigned DIM>
      void DiskXferDes<DIM>::notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        int64_t offset;
        uint64_t size;
        switch(kind) {
          case XferDes::XFER_DISK_READ:
            offset = ((DiskReadRequest*)req)->dst_buf - (uint64_t) buf_base;
            size = ((DiskReadRequest*)req)->nbytes;
            break;
          case XferDes::XFER_DISK_WRITE:
            offset = ((DiskWriteRequest*)req)->dst_offset - dst_buf.alloc_offset;
            size = ((DiskWriteRequest*)req)->nbytes;
            break;
          default:
            assert(0);
        }
        simple_update_bytes_write(offset, size);
        available_reqs.push(req);
        //printf("bytes_write = %lu, bytes_total = %lu\n", bytes_write, bytes_total);
      }

      template<unsigned DIM>
      void DiskXferDes<DIM>::flush()
      {
        fsync(fd);
      }
#endif /*USE_DISK*/
#ifdef USE_CUDA
      template<unsigned DIM>
      GPUXferDes<DIM>::GPUXferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                                  XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                                  const Buffer& _src_buf, const Buffer& _dst_buf,
                                  const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                                  uint64_t _max_req_size, long max_nr, int _priority,
                                  XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence)
      : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid, _next_xd_guid, _src_buf, _dst_buf,
                _domain, _oas_vec, _max_req_size, _priority, _order, _kind, _complete_fence)
      {
        MemoryImpl* src_mem_impl = get_runtime()->get_memory_impl(_src_buf.memory);
        MemoryImpl* dst_mem_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
        size_t total_field_size = 0;
        for (int i = 0; i < oas_vec.size(); i++) {
          total_field_size += oas_vec[i].size;
        }
        bytes_total = total_field_size * domain.get_volume();
        pre_bytes_write = (pre_xd_guid == XFERDES_NO_GUID) ? bytes_total : 0;
        li = new Layouts::GenericLayoutIterator<DIM>(domain.get_rect<DIM>(), src_buf.linearization.get_mapping<DIM>(),
                                                     dst_buf.linearization.get_mapping<DIM>(), order);
        offset_idx = 0;

        switch (kind) {
          case XferDes::XFER_GPU_TO_FB:
          {
            src_gpu = NULL;
            dst_gpu = ((GPUFBMemory*)dst_mem_impl)->gpu;;
            channel = channel_manager->get_gpu_to_fb_channel(dst_gpu);
            src_buf_base = (char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
            dst_buf_base = NULL;
            assert(src_mem_impl->kind == MemoryImpl::MKIND_SYSMEM || src_mem_impl->kind == MemoryImpl::MKIND_ZEROCOPY);
            assert(dst_mem_impl->kind == MemoryImpl::MKIND_GPUFB);
            GPUtoFBRequest* gpu_to_fb_reqs = (GPUtoFBRequest*) calloc(max_nr, sizeof(GPUtoFBRequest));
            for (int i = 0; i < max_nr; i++) {
              gpu_to_fb_reqs[i].xd = this;
              available_reqs.push(&gpu_to_fb_reqs[i]);
            }
            requests = gpu_to_fb_reqs;
            break;
          }
          case XferDes::XFER_GPU_FROM_FB:
          {
            src_gpu = ((GPUFBMemory*)src_mem_impl)->gpu;
            dst_gpu = NULL;
            channel = channel_manager->get_gpu_from_fb_channel(src_gpu);
            src_buf_base = NULL;
            dst_buf_base = (char*) dst_mem_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
            assert(src_mem_impl->kind == MemoryImpl::MKIND_GPUFB);
            assert(dst_mem_impl->kind == MemoryImpl::MKIND_SYSMEM || dst_mem_impl->MemoryImpl::MKIND_ZEROCOPY);
            GPUfromFBRequest* gpu_from_fb_reqs = (GPUfromFBRequest*) calloc(max_nr, sizeof(GPUfromFBRequest));
            for (int i = 0; i < max_nr; i++) {
              gpu_from_fb_reqs[i].xd = this;
              available_reqs.push(&gpu_from_fb_reqs[i]);
            }
            requests = gpu_from_fb_reqs;
            break;
          }
          case XferDes::XFER_GPU_IN_FB:
          {
            src_gpu = ((GPUFBMemory*)src_mem_impl)->gpu;
            dst_gpu = ((GPUFBMemory*)dst_mem_impl)->gpu;
            channel = channel_manager->get_gpu_in_fb_channel(src_gpu);
            src_buf_base = dst_buf_base = NULL;
            assert(src_mem_impl->kind == MemoryImpl::MKIND_GPUFB);
            assert(src_mem_impl->kind == MemoryImpl::MKIND_GPUFB);
            assert(src_gpu == dst_gpu);
            GPUinFBRequest* gpu_in_fb_reqs = (GPUinFBRequest*) calloc(max_nr, sizeof(GPUinFBRequest));
            for (int i = 0; i < max_nr; i++) {
              gpu_in_fb_reqs[i].xd = this;
              available_reqs.push(&gpu_in_fb_reqs[i]);
            }
            requests = gpu_in_fb_reqs;
            break;
          }
          case XferDes::XFER_GPU_PEER_FB:
          {
            src_gpu = ((GPUFBMemory*)src_mem_impl)->gpu;
            dst_gpu = ((GPUFBMemory*)dst_mem_impl)->gpu;
            channel = channel_manager->get_gpu_peer_fb_channel(src_gpu);
            src_buf_base = dst_buf_base = NULL;
            assert(src_mem_impl->kind == MemoryImpl::MKIND_GPUFB);
            assert(src_mem_impl->kind == MemoryImpl::MKIND_GPUFB);
            assert(((GPUFBMemory*)src_mem_impl)->gpu == dst_gpu);
            GPUpeerFBRequest* gpu_peer_fb_reqs = (GPUpeerFBRequest*) calloc(max_nr, sizeof(GPUpeerFBRequest));
            for (int i = 0; i < max_nr; i++) {
              gpu_peer_fb_reqs[i].xd = this;
              available_reqs.push(&gpu_peer_fb_reqs[i]);
            }
            requests = gpu_peer_fb_reqs;
            break;
          }
          default:
            assert(0);
        }
      }

      template<unsigned DIM>
      long GPUXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        long idx = 0;
        while (idx < nr && !available_reqs.empty() && offset_idx < oas_vec.size()) {
          off_t src_start, dst_start;
          size_t nbytes;
          simple_get_request<DIM>(src_start, dst_start, nbytes, li, offset_idx, min(available_reqs.size(), nr - idx));
          if (nbytes == 0)
            break;
          while (nbytes > 0) {
            size_t req_size = umin(nbytes, max_req_size);
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
              case XferDes::XFER_GPU_TO_FB:
              {
                printf("[GPUtoFBXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start, dst_start, req_size);
                GPUtoFBRequest* gpu_to_fb_req = (GPUtoFBRequest*) requests[idx];
                gpu_to_fb_req->src = src_buf_base + src_start;
                gpu_to_fb_req->dst_offset = dst_buf.alloc_offset + dst_start;
                gpu_to_fb_req->nbytes = req_size;
                gpu_to_fb_req->event.reset();
                break;
              }
              case XferDes::XFER_GPU_FROM_FB:
              {
                printf("[GPUfromFBXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start, dst_start, req_size);
                GPUfromFBRequest* gpu_from_fb_req = (GPUfromFBRequest*) requests[idx];
                gpu_from_fb_req->src_offset = src_buf.alloc_offset + src_start;
                gpu_from_fb_req->dst = dst_buf_base + dst_start;
                gpu_from_fb_req->nbytes = req_size;
                gpu_from_fb_req->event.reset();
                break;
              }
              case XferDes::XFER_GPU_IN_FB:
              {
                printf("[GPUinFBXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start, dst_start, req_size);
                GPUinFBRequest* gpu_in_fb_req = (GPUinFBRequest*) requests[idx];
                gpu_in_fb_req->src_offset = src_buf.alloc_offset + src_start;
                gpu_in_fb_req->dst_offset = dst_buf.alloc_offset + dst_start;
                gpu_in_fb_req->nbytes = req_size;
                gpu_in_fb_req->event.reset();
                break;
              }
              case XferDes::XFER_GPU_PEER_FB:
              {
                GPUpeerFBRequest* gpu_peer_fb_req = (GPUpeerFBRequest*) requests[idx];
                gpu_peer_fb_req->src_offset = src_buf.alloc_offset + src_start;
                gpu_peer_fb_req->dst_offset = dst_buf.alloc_offset + dst_start;
                gpu_peer_fb_req->nbytes = req_size;
                gpu_peer_fb_req->dst_gpu = dst_gpu;
                gpu_peer_fb_req->event.reset();
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
      void GPUXferDes<DIM>::notify_request_read_done(Request* req)
      {
        req->is_read_done = true;
        int64_t offset;
        uint64_t size;
        switch (kind) {
          case XferDes::XFER_GPU_TO_FB:
            offset = ((GPUtoFBRequest*)req)->src - src_buf_base;
            size = ((GPUtoFBRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GPU_FROM_FB:
            offset = ((GPUfromFBRequest*)req)->src_offset - src_buf.alloc_offset;
            size = ((GPUfromFBRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GPU_IN_FB:
            offset = ((GPUinFBRequest*)req)->src_offset - src_buf.alloc_offset;
            size = ((GPUinFBRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GPU_PEER_FB:
            offset = ((GPUpeerFBRequest*)req)->src_offset - src_buf.alloc_offset;
            size = ((GPUpeerFBRequest*)req)->nbytes;
            break;
          default:
            assert(0);
        }
        simple_update_bytes_read(offset, size);
      }

      template<unsigned DIM>
      void GPUXferDes<DIM>::notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        int64_t offset;
        uint64_t size;
        switch (kind) {
          case XferDes::XFER_GPU_TO_FB:
            offset = ((GPUtoFBRequest*)req)->dst_offset - dst_buf.alloc_offset;
            size = ((GPUtoFBRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GPU_FROM_FB:
            offset = ((GPUfromFBRequest*)req)->dst - dst_buf_base;
            size = ((GPUfromFBRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GPU_IN_FB:
            offset = ((GPUinFBRequest*)req)->dst_offset - dst_buf.alloc_offset;
            size = ((GPUfromFBRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GPU_PEER_FB:
            offset = ((GPUpeerFBRequest*)req)->dst_offset - dst_buf.alloc_offset;
            size = ((GPUpeerFBRequest*)req)->nbytes;
            break;
          default:
            assert(0);
        }
        printf("[GPU_Request_Done] dst_offset = %ld, nbytes = %lu\n", offset, size);
        simple_update_bytes_write(offset, size);
        available_reqs.push(req);
      }

      template<unsigned DIM>
      void GPUXferDes<DIM>::flush()
      {
      }
#endif

#ifdef USE_HDF
      template<unsigned DIM>
      HDFXferDes<DIM>::HDFXferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                                  XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                                  RegionInstance inst, const Buffer& _src_buf, const Buffer& _dst_buf,
                                  const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                                  uint64_t _max_req_size, long max_nr, int _priority,
                                  XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence)
        : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid, _next_xd_guid, _src_buf, _dst_buf,
                  _domain, _oas_vec, _max_req_size, _priority, _order, _kind, _complete_fence)
      {
        MemoryImpl* src_impl = get_runtime()->get_memory_impl(_src_buf.memory);
        MemoryImpl* dst_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
        // for now, we didn't consider HDF transfer for intermediate buffer
        // since ib may involve a different address space model
        assert(!src_buf.is_ib);
        assert(!dst_buf.is_ib);
        size_t total_field_size = 0;
        for (int i = 0; i < oas_vec.size(); i++) {
          total_field_size += oas_vec[i].size;
        }
        bytes_total = total_field_size * domain.get_volume();
        pre_bytes_write = (pre_xd_guid == XFERDES_NO_GUID) ? bytes_total : 0;
        Rect<DIM> subrect_check;
        switch (kind) {
          case XferDes::XFER_HDF_READ:
          {
            ID src_id(inst);
            unsigned src_index = src_id.index_l();
            pthread_rwlock_rdlock(&((HDFMemory*)get_runtime()->get_memory_impl(src_buf.memory))->rwlock);
            hdf_metadata = ((HDFMemory*) get_runtime()->get_memory_impl(src_buf.memory))->hdf_metadata_vec[src_index];
            pthread_rwlock_unlock(&((HDFMemory*)get_runtime()->get_memory_impl(src_buf.memory))->rwlock);
            channel = channel_manager->get_hdf_read_channel();
            buf_base = (char*) dst_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
            assert(src_impl->kind == MemoryImpl::MKIND_HDF);
            assert(dst_impl->kind == MemoryImpl::MKIND_SYSMEM || dst_impl->kind == MemoryImpl::MKIND_ZEROCOPY);
            HDFReadRequest* hdf_read_reqs = (HDFReadRequest*) calloc(max_nr, sizeof(HDFReadRequest));
            for (int i = 0; i < max_nr; i++) {
              hdf_read_reqs[i].xd = this;
              hdf_read_reqs[i].hdf_memory = hdf_metadata->hdf_memory;
              available_reqs.push(&hdf_read_reqs[i]);
            }
            requests = hdf_read_reqs;
            hli = new Layouts::HDFLayoutIterator<DIM>(domain.get_rect<DIM>(), dst_buf.linearization.get_mapping<DIM>(), dst_buf.block_size);
            //lsi = new GenericLinearSubrectIterator<Mapping<DIM, 1> >(domain.get_rect<DIM>(), *(dst_buf.linearization.get_mapping<DIM>()));
            // Make sure instance involves FortranArrayLinearization
            // assert(lsi->strides[0][0] == 1);
            // This is kind of tricky, but to avoid recomputing hdf dataset idx for every oas entry,
            // we change the src/dst offset to hdf dataset idx
            for (fit = oas_vec.begin(); fit != oas_vec.end(); fit++) {
              off_t offset = 0;
              int idx = 0;
              while (offset < (*fit).src_offset) {
                pthread_rwlock_rdlock(&hdf_metadata->hdf_memory->rwlock);
                offset += H5Tget_size(hdf_metadata->datatype_ids[idx]);
                pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);
                idx++;
              }
              assert(offset == (*fit).src_offset);
              (*fit).src_offset = idx;
            }
            fit = oas_vec.begin();
            break;
          }
          case XferDes::XFER_HDF_WRITE:
          {
            ID dst_id(inst);
            unsigned index = dst_id.index_l();
            pthread_rwlock_rdlock(&((HDFMemory*)get_runtime()->get_memory_impl(dst_buf.memory))->rwlock);
            hdf_metadata = ((HDFMemory*)get_runtime()->get_memory_impl(dst_buf.memory))->hdf_metadata_vec[index];
            pthread_rwlock_unlock(&((HDFMemory*)get_runtime()->get_memory_impl(dst_buf.memory))->rwlock);
            channel = channel_manager->get_hdf_write_channel();
            buf_base = (char*) src_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
            assert(src_impl->kind == MemoryImpl::MKIND_SYSMEM || src_impl->kind == MemoryImpl::MKIND_ZEROCOPY);
            assert(dst_impl->kind == MemoryImpl::MKIND_HDF);
            HDFWriteRequest* hdf_write_reqs = (HDFWriteRequest*) calloc(max_nr, sizeof(HDFWriteRequest));
            for (int i = 0; i < max_nr; i++) {
              hdf_write_reqs[i].xd = this;
              hdf_write_reqs[i].hdf_memory = hdf_metadata->hdf_memory;
              available_reqs.push(&hdf_write_reqs[i]);
            }
            requests = hdf_write_reqs;
            hli = new Layouts::HDFLayoutIterator<DIM>(domain.get_rect<DIM>(), src_buf.linearization.get_mapping<DIM>(), src_buf.block_size);
            // lsi = new GenericLinearSubrectIterator<Mapping<DIM, 1> >(domain.get_rect<DIM>(), *(src_buf.linearization.get_mapping<DIM>()));
            // Make sure instance involves FortranArrayLinearization
            // assert(lsi->strides[0][0] == 1);
            // This is kind of tricky, but to avoid recomputing hdf dataset idx for every oas entry,
            // we change the src/dst offset to hdf dataset idx
            for (fit = oas_vec.begin(); fit != oas_vec.end(); fit++) {
              off_t offset = 0;
              int idx = 0;
              while (offset < (*fit).dst_offset) {
                pthread_rwlock_rdlock(&hdf_metadata->hdf_memory->rwlock);
                offset += H5Tget_size(hdf_metadata->datatype_ids[idx]);
                pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);
                idx++;
              }
              assert(offset == (*fit).dst_offset);
              (*fit).dst_offset = idx;
            }
            fit = oas_vec.begin();
            break;
          }
          default:
            assert(0);
        }
      }

      template<unsigned DIM>
      long HDFXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        long ns = 0;
        while (ns < nr && !available_reqs.empty() && fit != oas_vec.end()) {
          requests[ns] = available_reqs.front();
          available_reqs.pop();
          requests[ns]->is_read_done = false;
          requests[ns]->is_write_done = false;
          switch (kind) {
            case XferDes::XFER_HDF_READ:
            {
              // Recall that src_offset means the index of the involving dataset in hdf file
              off_t hdf_idx = fit->src_offset;
              pthread_rwlock_rdlock(&hdf_metadata->hdf_memory->rwlock);
              size_t elemnt_size = H5Tget_size(hdf_metadata->datatype_ids[hdf_idx]);
              //int todo = min(pir->r.hi[0] - pir->p[0] + 1, dst_buf.block_size - lsi->mapping.image(pir->p) % dst_buf.block_size);
              HDFReadRequest* hdf_read_req = (HDFReadRequest*) requests[ns];
              hdf_read_req->dataset_id = hdf_metadata->dataset_ids[hdf_idx];
              hdf_read_req->rwlock = &hdf_metadata->dataset_rwlocks[hdf_idx];
              hdf_read_req->mem_type_id = hdf_metadata->datatype_ids[hdf_idx];
              hsize_t count[DIM], offset[DIM];
              size_t todo = 1;
              for (int i = 0; i < DIM; i++) {
                count[i] = hli->sub_rect.hi[i] - hli->sub_rect.lo[i] + 1;
                todo *= count[i];
                //offset[i] = pir->pa[i] - hdf_metadata->lo[i];
                offset[i] = hli->sub_rect.lo[i] - hdf_metadata->lo[i];
              }
              hdf_read_req->file_space_id = H5Dget_space(hdf_metadata->dataset_ids[hdf_idx]);
              // HDF dimension always start with zero, but Legion::Domain may start with any integer
              // We need to deal with the offset between them here
              herr_t ret = H5Sselect_hyperslab(hdf_read_req->file_space_id, H5S_SELECT_SET, offset, NULL, count, NULL);
              assert(ret >= 0);
              pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);
              pthread_rwlock_wrlock(&hdf_metadata->hdf_memory->rwlock);
              hdf_read_req->mem_space_id = H5Screate_simple(DIM, count, NULL);
              pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);
              off_t dst_offset = calc_mem_loc(0, fit->dst_offset, fit->size,
                                              dst_buf.elmt_size, dst_buf.block_size, hli->cur_idx);
              hdf_read_req->dst = buf_base + dst_offset;
              hdf_read_req->nbytes = todo * elemnt_size;
              break;
            }
            case XferDes::XFER_HDF_WRITE:
            {
              // Recall that src_offset means the index of the involving dataset in hdf file
              off_t hdf_idx = fit->dst_offset;
              pthread_rwlock_rdlock(&hdf_metadata->hdf_memory->rwlock);
              size_t elemnt_size = H5Tget_size(hdf_metadata->datatype_ids[hdf_idx]);
              //int todo = min(pir->r.hi[0] - pir->p[0] + 1, src_buf.block_size - lsi->mapping.image(pir->p) % src_buf.block_size);
              HDFWriteRequest* hdf_write_req = (HDFWriteRequest*) requests[ns];
              hdf_write_req->dataset_id = hdf_metadata->dataset_ids[hdf_idx];
              hdf_write_req->rwlock = &hdf_metadata->dataset_rwlocks[hdf_idx];
              hdf_write_req->mem_type_id = hdf_metadata->datatype_ids[hdf_idx];
              hsize_t count[DIM], offset[DIM];
              size_t todo = 1;
              for (int i = 0; i < DIM; i++) {
                count[i] = hli->sub_rect.hi[i] - hli->sub_rect.lo[i] + 1;
                todo *= count[i];
                offset[i] = hli->sub_rect.lo[i] - hdf_metadata->lo[i];
              }
              hdf_write_req->file_space_id = H5Dget_space(hdf_metadata->dataset_ids[hdf_idx]);
              // HDF dimension always start with zero, but Legion::Domain may start with any integer
              // We need to deal with the offset between them here
              herr_t ret = H5Sselect_hyperslab(hdf_write_req->file_space_id, H5S_SELECT_SET, offset, NULL, count, NULL);
              assert(ret >= 0);
              pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);
              pthread_rwlock_wrlock(&hdf_metadata->hdf_memory->rwlock);
              hdf_write_req->mem_space_id = H5Screate_simple(DIM, count, NULL);
              pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);
              off_t src_offset = calc_mem_loc(0, fit->src_offset, fit->size,
                                              src_buf.elmt_size, src_buf.block_size, hli->cur_idx);
              hdf_write_req->src = buf_base + src_offset;
              hdf_write_req->nbytes = todo * elemnt_size;
              break;
            }
            default:
              assert(0);
          }
          hli->step();
          if(!hli->any_left()) {
            fit++;
            Layouts::HDFLayoutIterator<DIM>* new_hli = new Layouts::HDFLayoutIterator<DIM>(hli->orig_rect, hli->mapping, hli->block_size);
            delete hli;
            hli = new_hli;
          }
          ns ++;
        }
        return ns;
      }

      template<unsigned DIM>
      void HDFXferDes<DIM>::notify_request_read_done(Request* req)
      {
        req->is_read_done = true;
        // close and release HDF resources
        // currently we don't support ib case
        assert(pre_xd_guid == XFERDES_NO_GUID);
        switch (kind) {
          case XferDes::XFER_HDF_READ:
          {
            HDFReadRequest* hdf_read_req = (HDFReadRequest*) req;
            bytes_read += hdf_read_req->nbytes;
            break;
          }
          case XferDes::XFER_HDF_WRITE:
          {
            HDFWriteRequest* hdf_write_req = (HDFWriteRequest*) req;
            bytes_read += hdf_write_req->nbytes;
            break;
          }
          default:
            assert(0);
        }
      }

      template<unsigned DIM>
      void HDFXferDes<DIM>::notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        // currently we don't support ib case
        assert(next_xd_guid == XFERDES_NO_GUID);
        switch (kind) {
          case XferDes::XFER_HDF_READ:
          {
            HDFReadRequest* hdf_read_req = (HDFReadRequest*) req;
            bytes_write += hdf_read_req->nbytes;
            pthread_rwlock_wrlock(&hdf_metadata->hdf_memory->rwlock);
            H5Sclose(hdf_read_req->mem_space_id);
            H5Sclose(hdf_read_req->file_space_id);
            pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);
            break;
          }
          case XferDes::XFER_HDF_WRITE:
          {
            HDFWriteRequest* hdf_write_req = (HDFWriteRequest*) req;
            bytes_write += hdf_write_req->nbytes;
            pthread_rwlock_wrlock(&hdf_metadata->hdf_memory->rwlock);
            H5Sclose(hdf_write_req->mem_space_id);
            H5Sclose(hdf_write_req->file_space_id);
            pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);
            break;
          }
          default:
            assert(0);
        }
        available_reqs.push(req);
      }

      template<unsigned DIM>
      void HDFXferDes<DIM>::flush()
      {
        if (kind == XferDes::XFER_HDF_READ) {
        } else {
          assert(kind == XferDes::XFER_HDF_WRITE);
          for (fit = oas_vec.begin(); fit != oas_vec.end(); fit++) {
            off_t hdf_idx = fit->dst_offset;
            hid_t dataset_id = hdf_metadata->dataset_ids[hdf_idx];
            //TODO: I am not sure if we need a lock here to protect HDFflush
            H5Fflush(dataset_id, H5F_SCOPE_LOCAL);
          }
        }
      }
#endif

      MemcpyChannel::MemcpyChannel(long max_nr)
      {
        kind = XferDes::XFER_MEM_CPY;
        capacity = max_nr;
        //cbs = (MemcpyRequest**) calloc(max_nr, sizeof(MemcpyRequest*));
      }

      MemcpyChannel::~MemcpyChannel()
      {
        //free(cbs);
      }

      long MemcpyChannel::submit(Request** requests, long nr)
      {
        MemcpyRequest** mem_cpy_reqs = (MemcpyRequest**) requests;
        for (int i = 0; i < nr; i++) {
          memcpy(mem_cpy_reqs[i]->dst_buf, mem_cpy_reqs[i]->src_buf, mem_cpy_reqs[i]->nbytes);
          mem_cpy_reqs[i]->xd->notify_request_read_done(mem_cpy_reqs[i]);
          mem_cpy_reqs[i]->xd->notify_request_write_done(mem_cpy_reqs[i]);
        }
        return nr;
      }

      void MemcpyChannel::pull()
      {
        /*
        while (true) {
          long np = worker->pull(cbs, capacity);
          for (int i = 0; i < np; i++) {
            cbs[i]->xd->notify_request_read_done(cbs[i]);
            cbs[i]->xd->notify_request_write_done(cbs[i]);
          }
          if (np != capacity)
            break;
        }
        */
      }

      long MemcpyChannel::available()
      {
        return capacity;
      }

      GASNetChannel::GASNetChannel(long max_nr, XferDes::XferKind _kind)
      {
        kind = _kind;
        capacity = max_nr;
      }

      GASNetChannel::~GASNetChannel()
      {
      }

      long GASNetChannel::submit(Request** requests, long nr)
      {
        switch (kind) {
          case XferDes::XFER_GASNET_READ:
            for (int i = 0; i < nr; i++) {
              GASNetReadRequest* read_req = (GASNetReadRequest*) requests[i];
              get_runtime()->global_memory->get_bytes(read_req->src_offset, read_req->dst_buf, read_req->nbytes);
              read_req->xd->notify_request_read_done(read_req);
              read_req->xd->notify_request_write_done(read_req);
            }
            break;
          case XferDes::XFER_GASNET_WRITE:
            for (int i = 0; i < nr; i++) {
              GASNetWriteRequest* write_req = (GASNetWriteRequest*) requests[i];
              get_runtime()->global_memory->put_bytes(write_req->dst_offset, write_req->src_buf, write_req->nbytes);
              write_req->xd->notify_request_read_done(write_req);
              write_req->xd->notify_request_write_done(write_req);
            }
            break;
          default:
            assert(0);
        }
        return nr;
      }

      void GASNetChannel::pull()
      {
      }

      long GASNetChannel::available()
      {
        return capacity;
      }

      RemoteWriteChannel::RemoteWriteChannel(long max_nr)
      {
        capacity = max_nr;
      }

      RemoteWriteChannel::~RemoteWriteChannel() {}

      long RemoteWriteChannel::submit(Request** requests, long nr)
      {
        assert(nr <= capacity);
        for (int i = 0; i < nr; i ++) {
          RemoteWriteRequest* req = (RemoteWriteRequest*) requests[i];
          XferDesRemoteWriteMessage::send_request(req->dst_node, req->dst_buf, req->src_buf, req->nbytes, req);
          capacity--;
        /*RemoteWriteRequest* req = (RemoteWriteRequest*) requests[i];
          req->complete_event = GenEventImpl::create_genevent()->current_event();
          Realm::RemoteWriteMessage::RequestArgs args;
          args.mem = req->dst_mem;
          args.offset = req->dst_offset;
          args.event = req->complete_event;
          args.sender = gasnet_mynode();
          args.sequence_id = 0;

          Realm::RemoteWriteMessage::Message::request(ID(args.mem).node(), args,
                                                      req->src_buf, req->nbytes,
                                                      PAYLOAD_KEEPREG,
                                                      req->dst_buf);*/
        }
        return nr;
      }

      void RemoteWriteChannel::pull()
      {
      }

      long RemoteWriteChannel::available()
      {
        return capacity;
      }

#ifdef USE_DISK
      DiskChannel::DiskChannel(long max_nr, XferDes::XferKind _kind)
      {
        kind = _kind;
        ctx = 0;
        capacity = max_nr;
        int ret = io_setup(max_nr, &ctx);
        assert(ret >= 0);
        assert(available_cb.empty());
        cb = (struct iocb*) calloc(max_nr, sizeof(struct iocb));
        cbs = (struct iocb**) calloc(max_nr, sizeof(struct iocb*));
        events = (struct io_event*) calloc(max_nr, sizeof(struct io_event));
        switch (kind) {
          case XferDes::XFER_DISK_READ:
            for (int i = 0; i < max_nr; i++) {
              memset(&cb[i], 0, sizeof(cb[i]));
              cb[i].aio_lio_opcode = IOCB_CMD_PREAD;
              available_cb.push_back(&cb[i]);
            }
            break;
          case XferDes::XFER_DISK_WRITE:
            for (int i = 0; i < max_nr; i++) {
              memset(&cb[i], 0, sizeof(cb[i]));
              cb[i].aio_lio_opcode = IOCB_CMD_PWRITE;
              available_cb.push_back(&cb[i]);
            }
            break;
          default:
            assert(0);
        }
      }

      DiskChannel::~DiskChannel()
      {
        io_destroy(ctx);
        free(cb);
        free(cbs);
        free(events);
      }

      long DiskChannel::submit(Request** requests, long nr)
      {
        int ns = 0;
        switch (kind) {
          case XferDes::XFER_DISK_READ:
            while (ns < nr && !available_cb.empty()) {
              DiskReadRequest* disk_read_req = (DiskReadRequest*) requests[ns];
              cbs[ns] = available_cb.back();
              available_cb.pop_back();
              cbs[ns]->aio_fildes = disk_read_req->fd;
              cbs[ns]->aio_data = (uint64_t) (disk_read_req);
              cbs[ns]->aio_buf = disk_read_req->dst_buf;
              cbs[ns]->aio_offset = disk_read_req->src_offset;
              cbs[ns]->aio_nbytes = disk_read_req->nbytes;
              ns++;
            }
            break;
          case XferDes::XFER_DISK_WRITE:
            while (ns < nr && !available_cb.empty()) {
              DiskWriteRequest* disk_write_req = (DiskWriteRequest*) requests[ns];
              cbs[ns] = available_cb.back();
              available_cb.pop_back();
              cbs[ns]->aio_fildes = disk_write_req->fd;
              cbs[ns]->aio_data = (uint64_t) (disk_write_req);
              cbs[ns]->aio_buf = disk_write_req->src_buf;
              cbs[ns]->aio_offset = disk_write_req->dst_offset;
              cbs[ns]->aio_nbytes = disk_write_req->nbytes;
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

      void DiskChannel::pull()
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

      long DiskChannel::available()
      {
        return available_cb.size();
      }
#endif /*USE_DISK*/
    
#ifdef USE_CUDA
      GPUChannel::GPUChannel(GPU* _src_gpu, long max_nr, XferDes::XferKind _kind)
      {
        src_gpu = _src_gpu;
        kind = _kind;
        capacity = max_nr;
      }

      GPUChannel::~GPUChannel()
      {
      }

      long GPUChannel::submit(Request** requests, long nr)
      {
        switch (kind) {
          case XferDes::XFER_GPU_TO_FB:
          {
            GPUtoFBRequest** gpu_to_fb_reqs = (GPUtoFBRequest**) requests;
            for (int i = 0; i < nr; i++) {
              //gpu_to_fb_reqs[i]->complete_event = GenEventImpl::create_genevent()->current_event();
              // Should be deleted!!!
              gpu_to_fb_reqs[i]->event.request_completed();
              src_gpu->copy_to_fb(gpu_to_fb_reqs[i]->dst_offset,
                                  gpu_to_fb_reqs[i]->src,
                                  gpu_to_fb_reqs[i]->nbytes,
                                  &gpu_to_fb_reqs[i]->event
                                  /*Event::NO_EVENT,
                                  gpu_to_fb_reqs[i]->complete_event*/);
              pending_copies.push_back(gpu_to_fb_reqs[i]);
            }
            break;
          }
          case XferDes::XFER_GPU_FROM_FB:
          {
            GPUfromFBRequest** gpu_from_fb_reqs = (GPUfromFBRequest**) requests;
            for (int i = 0; i < nr; i++) {
              //gpu_from_fb_reqs[i]->complete_event = GenEventImpl::create_genevent()->current_event();
              src_gpu->copy_from_fb(gpu_from_fb_reqs[i]->dst,
                                    gpu_from_fb_reqs[i]->src_offset,
                                    gpu_from_fb_reqs[i]->nbytes,
                                    &gpu_from_fb_reqs[i]->event
                                    /*Event::NO_EVENT,
                                    gpu_from_fb_reqs[i]->complete_event*/);
              pending_copies.push_back(gpu_from_fb_reqs[i]);
            }
            break;
          }
          case XferDes::XFER_GPU_IN_FB:
          {
            GPUinFBRequest** gpu_in_fb_reqs = (GPUinFBRequest**) requests;
            for (int i = 0; i < nr; i++) {
              //gpu_in_fb_reqs[i]->complete_event = GenEventImpl::create_genevent()->current_event();
              src_gpu->copy_within_fb(gpu_in_fb_reqs[i]->dst_offset,
                                      gpu_in_fb_reqs[i]->src_offset,
                                      gpu_in_fb_reqs[i]->nbytes,
                                      &gpu_in_fb_reqs[i]->event
                                      /*Event::NO_EVENT,
                                      gpu_in_fb_reqs[i]->complete_event*/);
              pending_copies.push_back(gpu_in_fb_reqs[i]);
            }
            break;
          }
          case XferDes::XFER_GPU_PEER_FB:
          {
            GPUpeerFBRequest** gpu_peer_fb_reqs = (GPUpeerFBRequest**) requests;
            for (int i = 0; i < nr; i++) {
              //gpu_peer_fb_reqs[i]->complete_event = GenEventImpl::create_genevent()->current_event();
              src_gpu->copy_to_peer(gpu_peer_fb_reqs[i]->dst_gpu,
                                    gpu_peer_fb_reqs[i]->dst_offset,
                                    gpu_peer_fb_reqs[i]->src_offset,
                                    gpu_peer_fb_reqs[i]->nbytes,
                                    &gpu_peer_fb_reqs[i]->event
                                    /*Event::NO_EVENT,
                                    gpu_peer_fb_reqs[i]->complete_event*/);
              pending_copies.push_back(gpu_peer_fb_reqs[i]);
            }
            break;
          }
          default:
            assert(0);
        }
        return nr;
      }

      void GPUChannel::pull()
      {
        switch (kind) {
          case XferDes::XFER_GPU_TO_FB:
            while (!pending_copies.empty()) {
              GPUtoFBRequest* gpu_to_fb_req = (GPUtoFBRequest*)pending_copies.front();
              if (gpu_to_fb_req->event.has_triggered()) {
                gpu_to_fb_req->xd->notify_request_read_done(gpu_to_fb_req);
                gpu_to_fb_req->xd->notify_request_write_done(gpu_to_fb_req);
                pending_copies.pop_front();
              }
              else
                break;
            }
            break;
          case XferDes::XFER_GPU_FROM_FB:
            while (!pending_copies.empty()) {
              GPUfromFBRequest* gpu_from_fb_req = (GPUfromFBRequest*)pending_copies.front();
              if (gpu_from_fb_req->event.has_triggered()) {
                gpu_from_fb_req->xd->notify_request_read_done(gpu_from_fb_req);
                gpu_from_fb_req->xd->notify_request_write_done(gpu_from_fb_req);
                pending_copies.pop_front();
              }
              else
                break;
            }
            break;
          case XferDes::XFER_GPU_IN_FB:
            while (!pending_copies.empty()) {
              GPUinFBRequest* gpu_in_fb_req = (GPUinFBRequest*)pending_copies.front();
              if (gpu_in_fb_req->event.has_triggered()) {
                gpu_in_fb_req->xd->notify_request_read_done(gpu_in_fb_req);
                gpu_in_fb_req->xd->notify_request_write_done(gpu_in_fb_req);
                pending_copies.pop_front();
              }
              else
                break;
            }
            break;
          case XferDes::XFER_GPU_PEER_FB:
            while (!pending_copies.empty()) {
              GPUpeerFBRequest* gpu_peer_fb_req = (GPUpeerFBRequest*)pending_copies.front();
              if (gpu_peer_fb_req->event.has_triggered()) {
                gpu_peer_fb_req->xd->notify_request_read_done(gpu_peer_fb_req);
                gpu_peer_fb_req->xd->notify_request_write_done(gpu_peer_fb_req);
                pending_copies.pop_front();
              }
              else
                break;
            }
            break;
          default:
            assert(0);
        }
      }

      long GPUChannel::available()
      {
        return capacity - pending_copies.size();
      }
#endif

#ifdef USE_HDF
      HDFChannel::HDFChannel(long max_nr, XferDes::XferKind _kind)
      {
        kind = _kind;
        capacity = max_nr;
      }

      HDFChannel::~HDFChannel() {}

      long HDFChannel::submit(Request** requests, long nr)
      {
        switch (kind) {
          case XferDes::XFER_HDF_READ:
          {
            HDFReadRequest** hdf_read_reqs = (HDFReadRequest**) requests;
            for (int i = 0; i < nr; i++) {
              HDFReadRequest* request = hdf_read_reqs[i];
              pthread_rwlock_rdlock(request->rwlock);
//              std::cout << "in HDFChannel::submit reading dataset_id: " << request->dataset_id << std::endl;
              H5Dread(request->dataset_id, request->mem_type_id, request->mem_space_id, request->file_space_id, H5P_DEFAULT, request->dst);
              pthread_rwlock_unlock(request->rwlock);
              request->xd->notify_request_read_done(request);
              request->xd->notify_request_write_done(request);
            }
            break;
          }
          case XferDes::XFER_HDF_WRITE:
          {
            HDFWriteRequest** hdf_read_reqs = (HDFWriteRequest**) requests;
            for (int i = 0; i < nr; i++) {
              HDFWriteRequest* request = hdf_read_reqs[i];
              pthread_rwlock_wrlock(request->rwlock);
//              std::cout << "in HDFChannel::submit writing dataset_id: " << request->dataset_id << std::endl;
              H5Dwrite(request->dataset_id, request->mem_type_id, request->mem_space_id, request->file_space_id, H5P_DEFAULT, request->src);
              pthread_rwlock_unlock(request->rwlock);
              request->xd->notify_request_read_done(request);
              request->xd->notify_request_write_done(request);
            }
            break;
          }
          default:
            assert(false);
        }
        return nr;
      }

      void HDFChannel::pull() {}

      long HDFChannel::available()
      {
        return capacity;
      }
#endif

#ifdef MEMCPY_THREAD_CODE
      void MemcpyThread::work()
      {
        while (true) {
          pthread_mutex_lock(&submit_lock);
          //printf("[MemcpyThread] CP#1\n");
          if (num_pending_reqs == 0)
            pthread_cond_wait(&condvar, &submit_lock);
          //printf("[MemcpyThread] Pull from pending queue\n");
          //printf("[MemcpyThread] num_pending_reqs = %ld\n", num_pending_reqs);
          assert(pending_queue.size() > 0);
          MemcpyRequest* cur_req = pending_queue.front();
          pending_queue.pop();
          num_pending_reqs --;
          pthread_mutex_unlock(&submit_lock);
          //printf("[MemcpyThread] Begin processing copy\n");
          //printf("[MemcpyThread] dst = %ld, src = %ld, nbytes = %lu\n", (off_t) cur_req->dst_buf, (off_t) cur_req->src_buf, cur_req->nbytes);
          memcpy(cur_req->dst_buf, cur_req->src_buf, cur_req->nbytes);
          //printf("[MemcpyThread] Finish processing copy\n");
          pthread_mutex_lock(&pull_lock);
          //printf("[MemcpyThread] Push into finished queue\n");
          finished_queue.push(cur_req);
          pthread_mutex_unlock(&pull_lock);
        }
      }

      void* MemcpyThread::start(void* arg)
      {
        printf("[MemcpyThread] start...\n");
        MemcpyThread* worker = (MemcpyThread*) arg;
        worker->work();
        return NULL;
      }
#endif
      /*static*/ void XferDesRemoteWriteMessage::handle_request(RequestArgs args, const void *data, size_t datalen)
      {
        // assert data copy is in right position
        assert(data == args.dst_buf);
        XferDesRemoteWriteAckMessage::send_request(args.sender, args.req);
      }

      /*static*/ void XferDesRemoteWriteAckMessage::handle_request(RequestArgs args)
      {
        RemoteWriteRequest* req = args.req;
        req->xd->notify_request_read_done(req);
        req->xd->notify_request_write_done(req);
        channel_manager->get_remote_write_channel()->notify_completion();
      }

      /*static*/ void XferDesCreateMessage::handle_request(RequestArgs args, const void *msgdata, size_t msglen)
      {
        const Payload *payload = (const Payload *)msgdata;
        std::vector<OffsetsAndSize> oas_vec(payload->oas_vec_size);
        for(size_t i = 0; i < payload->oas_vec_size; i++)
          oas_vec[i] = payload->oas_vec(i);
        Buffer src_buf, dst_buf;
        src_buf.deserialize(payload->src_buf_bits);
        src_buf.memory = args.src_mem;
        dst_buf.deserialize(payload->dst_buf_bits);
        dst_buf.memory = args.dst_mem;
        switch(payload->domain.dim) {
        case 0:
          fprintf(stderr, "Currently we doesn't support unstructured data...To be implemented\n");
          assert(0);
          break;
        case 1:
          create_xfer_des<1>(payload->dma_request, payload->launch_node,
                             payload->guid, payload->pre_xd_guid, payload->next_xd_guid,
                             src_buf, dst_buf, payload->domain, oas_vec,
                             payload->max_req_size, payload->max_nr, payload->priority,
                             payload->order, payload->kind, args.fence, args.inst);
          break;
        case 2:
          create_xfer_des<2>(payload->dma_request, payload->launch_node,
                             payload->guid, payload->pre_xd_guid, payload->next_xd_guid,
                             src_buf, dst_buf, payload->domain, oas_vec,
                             payload->max_req_size, payload->max_nr, payload->priority,
                             payload->order, payload->kind, args.fence, args.inst);
          break;
        case 3:
          create_xfer_des<3>(payload->dma_request, payload->launch_node,
                             payload->guid, payload->pre_xd_guid, payload->next_xd_guid,
                             src_buf, dst_buf, payload->domain, oas_vec,
                             payload->max_req_size, payload->max_nr, payload->priority,
                             payload->order, payload->kind, args.fence, args.inst);
          break;
        default:
          assert(0);
        }
      }

      /*static*/ void XferDesDestroyMessage::handle_request(RequestArgs args)
      {
        xferDes_queue->destroy_xferDes(args.guid);
      }

      /*static*/ void UpdateBytesWriteMessage::handle_request(RequestArgs args)
      {
        xferDes_queue->update_pre_bytes_write(args.guid, args.bytes_write);
      }

      /*static*/ void UpdateBytesReadMessage::handle_request(RequestArgs args)
      {
        xferDes_queue->update_next_bytes_read(args.guid, args.bytes_read);
      }

      void DMAThread::dma_thread_loop()
      {
        log_new_dma.info("start dma thread loop");
        while (!is_stopped) {
          bool is_empty = true;
          std::map<Channel*, PriorityXferDesQueue*>::iterator it;
          for (it = channel_to_xd_pool.begin(); it != channel_to_xd_pool.end(); it++) {
            if(!it->second->empty()) {
              is_empty = false;
              break;
            }
          }
          xd_queue->dequeue_xferDes(this, is_empty);

          for (it = channel_to_xd_pool.begin(); it != channel_to_xd_pool.end(); it++) {
            it->first->pull();
            long nr = it->first->available();
            if (nr == 0)
              continue;
            std::vector<XferDes*> finish_xferdes;
            PriorityXferDesQueue::iterator it2;
            for (it2 = it->second->begin(); it2 != it->second->end(); it2++) {
              assert((*it2)->channel == it->first);
              long nr_got = (*it2)->get_requests(requests, min(nr, max_nr));
              long nr_submitted = it->first->submit(requests, nr_got);
              nr -= nr_submitted;
              assert(nr_got == nr_submitted);
              if ((*it2)->is_completed()) {
                finish_xferdes.push_back(*it2);
                //printf("finish_xferdes.size() = %lu\n", finish_xferdes.size());
              }
              if (nr == 0)
                break;
            }
            while(!finish_xferdes.empty()) {
              XferDes *xd = finish_xferdes.back();
              finish_xferdes.pop_back();
              it->second->erase(xd);
              // We flush all changes into destination before mark this XferDes as completed
              xd->flush();
              xd->mark_completed();
              log_new_dma.info("Finish XferDes : id(%lx)", xd->guid);
              /*bool need_to_delete_dma_request = xd->mark_completed();
              if (need_to_delete_dma_request) {
                DmaRequest* dma_request = xd->dma_request;
                delete dma_request;
              }*/
            }
          }
        }
        log_new_dma.info("finish dma thread loop");
      }

      XferDesQueue* get_xdq_singleton()
      {
        return xferDes_queue;
      }
#ifdef USE_CUDA
      void register_gpu_in_dma_systems(GPU* gpu)
      {
        dma_all_gpus.push_back(gpu);
      }
#endif
      void start_channel_manager(int count, int max_nr, Realm::CoreReservationSet& crs)
      {
        xferDes_queue = new XferDesQueue(crs);
        channel_manager = new ChannelManager;
        xferDes_queue->start_worker(count, max_nr, channel_manager);
      }

      void XferDesQueue::start_worker(int count, int max_nr, ChannelManager* channel_manager) 
      {
        log_new_dma.info("XferDesQueue: start_workers");
        // TODO: count is currently ignored
        num_threads = 3;
#ifdef USE_HDF
        // Need a dedicated thread for handling HDF requests
        num_threads ++;
#endif
        dma_threads = (DMAThread**) calloc(num_threads, sizeof(DMAThread*));
        // dma thread #1: memcpy
        MemcpyChannel* memcpy_channel = channel_manager->create_memcpy_channel(max_nr);
        dma_threads[0] = new DMAThread(max_nr, xferDes_queue, memcpy_channel);
        // dma thread #2: async xfer
        std::vector<Channel*> async_channels, gasnet_channels;
        async_channels.push_back(channel_manager->create_remote_write_channel(max_nr));
#ifdef USE_DISK
        async_channels.push_back(channel_manager->create_disk_read_channel(max_nr));
        async_channels.push_back(channel_manager->create_disk_write_channel(max_nr));
#endif /*USE_DISK*/
#ifdef USE_CUDA
        std::vector<GPU*>::iterator it;
        for (it = dma_all_gpus.begin(); it != dma_all_gpus.end(); it ++) {
          async_channels.push_back(channel_manager->create_gpu_to_fb_channel(max_nr, *it));
          async_channels.push_back(channel_manager->create_gpu_from_fb_channel(max_nr, *it));
          async_channels.push_back(channel_manager->create_gpu_in_fb_channel(max_nr, *it));
          async_channels.push_back(channel_manager->create_gpu_peer_fb_channel(max_nr, *it));
        }
#endif
        dma_threads[1] = new DMAThread(max_nr, xferDes_queue, async_channels);
        gasnet_channels.push_back(channel_manager->create_gasnet_read_channel(max_nr));
        gasnet_channels.push_back(channel_manager->create_gasnet_write_channel(max_nr));
        dma_threads[2] = new DMAThread(max_nr, xferDes_queue, gasnet_channels);
#ifdef USE_HDF
        std::vector<Channel*> hdf_channels;
        hdf_channels.push_back(channel_manager->create_hdf_read_channel(max_nr));
        hdf_channels.push_back(channel_manager->create_hdf_write_channel(max_nr));
        dma_threads[3] = new DMAThread(max_nr, xferDes_queue, hdf_channels);
#endif
        for (int i = 0; i < num_threads; i++) {
          // register dma thread to XferDesQueue
           register_dma_thread(dma_threads[i]);
        }

        Realm::ThreadLaunchParameters tlp;

        for(int i = 0; i < num_threads; i++) {
          log_new_dma.info("Create a DMA worker thread");
          Realm::Thread *t = Realm::Thread::create_kernel_thread<DMAThread,
  	                                        &DMAThread::dma_thread_loop>(dma_threads[i],
  										                                 tlp,
  										                                 core_rsrv,
  										                                 0 /* default scheduler*/);
          worker_threads.push_back(t);
        }
      }

      void stop_channel_manager()
      {
        xferDes_queue->stop_worker();
        delete xferDes_queue;
        delete channel_manager;
      }

      void XferDesQueue::stop_worker() {
        for(int i = 0; i < num_threads; i++)
          dma_threads[i]->stop();
        // reap all the threads
        for(std::vector<Realm::Thread *>::iterator it = worker_threads.begin();
            it != worker_threads.end();
            it++) {
          (*it)->join();
          delete (*it);
        }
        worker_threads.clear();
        delete[] dma_threads;
      }


      template<unsigned DIM>
      void create_xfer_des(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                           XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                           const Buffer& _src_buf, const Buffer& _dst_buf,
                           const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                           uint64_t _max_req_size, long max_nr, int _priority,
                           XferOrder::Type _order, XferDes::XferKind _kind,
                           XferDesFence* _complete_fence, RegionInstance inst)
      {
        if (ID(_src_buf.memory).node() == gasnet_mynode()) {
          log_new_dma.info("Create local XferDes: id(%lx), pre(%lx), next(%lx), type(%d)",
                           _guid, _pre_xd_guid, _next_xd_guid, _kind);
          XferDes* xd;
          switch (_kind) {
          case XferDes::XFER_MEM_CPY:
            xd = new MemcpyXferDes<DIM>(_dma_request, _launch_node,
                                        _guid, _pre_xd_guid, _next_xd_guid,
                                        _src_buf, _dst_buf, _domain, _oas_vec,
                                        _max_req_size, max_nr, _priority,
                                        _order, _complete_fence);
            break;
          case XferDes::XFER_GASNET_READ:
          case XferDes::XFER_GASNET_WRITE:
            xd = new GASNetXferDes<DIM>(_dma_request, _launch_node,
                                        _guid, _pre_xd_guid, _next_xd_guid,
                                        _src_buf, _dst_buf, _domain, _oas_vec,
                                        _max_req_size, max_nr, _priority,
                                        _order, _kind, _complete_fence);
            break;
          case XferDes::XFER_REMOTE_WRITE:
            xd = new RemoteWriteXferDes<DIM>(_dma_request, _launch_node,
                                             _guid, _pre_xd_guid, _next_xd_guid,
                                             _src_buf, _dst_buf, _domain, _oas_vec,
                                             _max_req_size, max_nr, _priority,
                                             _order, _complete_fence);
            break;
#ifdef USE_DISK
          case XferDes::XFER_DISK_READ:
          case XferDes::XFER_DISK_WRITE:
            xd = new DiskXferDes<DIM>(_dma_request, _launch_node,
                                      _guid, _pre_xd_guid, _next_xd_guid,
                                      _src_buf, _dst_buf, _domain, _oas_vec,
                                      _max_req_size, max_nr, _priority,
                                      _order, _kind, _complete_fence);
            break;
#endif
#ifdef USE_CUDA
          case XferDes::XFER_GPU_FROM_FB:
          case XferDes::XFER_GPU_TO_FB:
          case XferDes::XFER_GPU_IN_FB:
          case XferDes::XFER_GPU_PEER_FB:
            xd = new GPUXferDes<DIM>(_dma_request, _launch_node,
                                     _guid, _pre_xd_guid, _next_xd_guid,
                                     _src_buf, _dst_buf, _domain, _oas_vec,
                                     _max_req_size, max_nr, _priority,
                                     _order, _kind, _complete_fence);
            break;
#endif
#ifdef USE_HDF
          case XferDes::XFER_HDF_READ:
          case XferDes::XFER_HDF_WRITE:
            xd = new HDFXferDes<DIM>(_dma_request, _launch_node,
                                     _guid, _pre_xd_guid, _next_xd_guid,
                                     inst, _src_buf, _dst_buf, _domain, _oas_vec,
                                     _max_req_size, max_nr, _priority,
                                     _order, _kind, _complete_fence);
            break;
#endif
        default:
          assert(false);
        }
        xferDes_queue->enqueue_xferDes_local(xd);
      } else {
        log_new_dma.info("Create remote XferDes: id(%lx), pre(%lx), next(%lx), type(%d)",
                         _guid, _pre_xd_guid, _next_xd_guid, _kind);
        XferDesCreateMessage::send_request(ID(_src_buf.memory).node(), _dma_request, _launch_node,
                                           _guid, _pre_xd_guid, _next_xd_guid,
                                           _src_buf, _dst_buf, _domain, _oas_vec,
                                           _max_req_size, max_nr, _priority,
                                           _order, _kind, _complete_fence, inst);
      }
    }

    void destroy_xfer_des(XferDesID _guid)
    {
      log_new_dma.info("Destroy XferDes: id(%lx)", _guid);
      gasnet_node_t execution_node = _guid >> (XferDesQueue::NODE_BITS + XferDesQueue::INDEX_BITS);
      if (execution_node == gasnet_mynode()) {
        xferDes_queue->destroy_xferDes(_guid);
      }
      else {
        XferDesDestroyMessage::send_request(execution_node, _guid);
      }
    }

      template class MemcpyXferDes<1>;
      template class MemcpyXferDes<2>;
      template class MemcpyXferDes<3>;
      template class GASNetXferDes<1>;
      template class GASNetXferDes<2>;
      template class GASNetXferDes<3>;
      template class RemoteWriteXferDes<1>;
      template class RemoteWriteXferDes<2>;
      template class RemoteWriteXferDes<3>;
#ifdef USE_DISK
      template class DiskXferDes<1>;
      template class DiskXferDes<2>;
      template class DiskXferDes<3>;
#endif /*USE_DISK*/
#ifdef USE_CUDA
      template class GPUXferDes<1>;
      template class GPUXferDes<2>;
      template class GPUXferDes<3>;
#endif
#ifdef USE_HDF
      template class HDFXferDes<1>;
      template class HDFXferDes<2>;
      template class HDFXferDes<3>;
#endif
  } // namespace LowLevel
} // namespace LegionRuntime

#ifdef DEADCODE
void DiskReadChannel::poll()
{
  int nr = io_getevents(ctx, 0, capacity, events, NULL);
  if (nr < 0)
    perror("io_getevents error");
  for (int i = 0; i < nr; i++) {
    DiskReadRequest* req = (DiskReadRequest*) events[i].data;
    struct iocb* ret_cb = (struct iocb*) events[i].obj;
    available_cb.push_back(ret_cb);
    assert(events[i].res == ret_cb->aio_nbytes);
    req->num_flying_aios --;
    if (req->num_flying_aios == 0 && cur_req != req) {
      // this indicates we have finished all aios within this req
      // time to recycle the request space
      req->xd->notify_request_read_done(req);
      req->xd->notify_request_write_done(req);
    }
  }
  // see if we can launch more aios
  int ns = 0;
  while (!available_cb.empty()) {
    if (cur_req != NULL) {
      // Case 1: we are dealing with a DiskReadRequest
      while (iter_1d != cur_req->copies_1D.end() && !available_cb.empty()) {
        cbs[ns] = available_cb.back();
        available_cb.pop_back();
        cbs[ns]->aio_fildes = cur_req->fd;
        cbs[ns]->aio_data = (int64_t) (cur_req);
        cbs[ns]->aio_buf = (*iter_1d)->dst_offset;
        cbs[ns]->aio_offset = (*iter_1d)->src_offset;
        cbs[ns]->aio_nbytes = (*iter_1d)->nbytes;
        cur_req->num_flying_aios ++;
        ns ++;
        iter_1d ++;
      }
      // handle 2D cases
      while (iter_2d != cur_req->copies_2D.end() && !available_cb.empty()) {
        cbs[ns] = available_cb.back();
        available_cb.pop_back();
        cbs[ns]->aio_fildes = cur_req->fd;
        cbs[ns]->aio_data = (int64_t) (cur_req);
        cbs[ns]->aio_buf = (*iter_2d)->dst_offset + (*iter_2d)->dst_stride * cur_line;
        cbs[ns]->aio_offset = (*iter_2d)->src_offset + (*iter_2d)->src_stride * cur_line;
        cbs[ns]->aio_nbytes = (*iter_2d)->nbytes;
        cur_req->num_flying_aios ++;
        ns ++;
        cur_line ++;
        if (cur_line == (*iter_2d)->nlines) {
          iter_2d ++; cur_line = 0;
        }
      }
      // submit aios
      if (ns > 0) {
        int ret = io_submit(ctx, ns, cbs);
        assert(ret == ns);
      }
      if (iter_1d == cur_req->copies_1D.end() && iter_2d == cur_req->copies_2D.end())
        cur_req = NULL;
    }
    else if (pending_reqs.empty()) {
      // Case 2: this indicates there is no more DiskReadRequest pending
      break;
    }
    else {
      // Case 3: move to the next DiskReadRequest
      cur_req = pending_reqs.front();
      iter_1d = cur_req->copies_1D.begin();
      iter_2d = cur_req->copies_2D.begin();
      cur_line = 0;
      pending_reqs.pop();
    }
  }
}

#endif
