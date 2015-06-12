/* Copyright 2015 Stanford University
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

      static inline int min(int a, int b) { return (a < b) ? a : b; }
      static inline int max(int a, int b) { return (a < b) ? b : a; }
      static inline size_t umin(size_t a, size_t b) { return (a < b) ? a : b; }

      static inline off_t calc_mem_loc(off_t alloc_offset, off_t field_start, int field_size, int elmt_size,
  				     int block_size, int index)
      {
        return (alloc_offset +                                      // start address
  	      ((index / block_size) * block_size * elmt_size) +   // full blocks
  	      (field_start * block_size) +                        // skip other fields
  	      ((index % block_size) * field_size));               // some some of our fields within our block
      }

      static inline bool scatter_ib(off_t start, size_t nbytes, size_t buf_size)
      {
        return (nbytes > 0) && (start / buf_size < (start + nbytes - 1) / buf_size);
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
        src_start = calc_mem_loc(src_buf->alloc_offset, oas_vec[offset_idx].src_offset, oas_vec[offset_idx].size,
                                 src_buf->elmt_size, src_buf->block_size, done + irect.lo);
        dst_start = calc_mem_loc(dst_buf->alloc_offset, oas_vec[offset_idx].dst_offset, oas_vec[offset_idx].size,
                                 dst_buf->elmt_size, dst_buf->block_size, done + orect.lo);
        nbytes = 0;
        bool scatter_src_ib = false, scatter_dst_ib = false;
        while (true) {
          // check to see if we can generate next request
          int src_in_block = src_buf->block_size - (done + irect.lo) % src_buf->block_size;
          int dst_in_block = dst_buf->block_size - (done + orect.lo) % dst_buf->block_size;
          int todo = min((max_req_size - nbytes) / oas_vec[offset_idx].size, min(total - done, min(src_in_block, dst_in_block)));
          // make sure we have source data ready
          if (src_buf->is_ib) {
            todo = min(todo, max(0, src_buf->alloc_offset + pre_bytes_write - (src_start + nbytes)) / oas_vec[offset_idx].size);
            scatter_src_ib = scatter_src_ib || scatter_ib(src_start, nbytes + todo * oas_vec[offset_idx].size, src_buf->buf_size);
          }
          // make sure there are enough space in destination
          if (dst_buf->is_ib) {
            todo = min(todo, max(0, dst_buf->alloc_offset + next_bytes_read + dst_buf->buf_size - (dst_start + nbytes)) / oas_vec[offset_idx].size);
            scatter_dst_ib = scatter_dst_ib || scatter_ib(dst_start, nbytes + todo * oas_vec[offset_idx].size, dst_buf->buf_size);
          }
          if((scatter_src_ib && scatter_dst_ib && available_slots < 3)
          ||((scatter_src_ib || scatter_dst_ib) && available_slots < 2))
            break;
          //printf("min(%d, %d, %d) \n =", (int)(max_req_size - nbytes) / oas_vec[offset_idx].size, total - done, min(src_in_block, dst_in_block));
          //printf("todo = %d, size = %d\n", todo, oas_vec[offset_idx].size);
          nbytes += todo * oas_vec[offset_idx].size;
          // see if we can batch more
          if (!disable_batch && todo == src_in_block && todo == dst_in_block && offset_idx + 1 < oas_vec.size()
          && src_buf->block_size == dst_buf->block_size && todo + done >= src_buf->block_size
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
        (((done + irect.lo) % src_buf->block_size == 0 && done + irect.lo > block_start && order == SRC_FIFO)
        ||((done + orect.lo) % dst_buf->block_size == 0 && done + orect.lo > block_start && order == DST_FIFO)
        || (done == total))) {
          offset_idx ++;
          if (offset_idx < oas_vec.size()) {
            switch (order) {
              case SRC_FIFO:
                done = block_start - irect.lo;
                break;
              case DST_FIFO:
                done = block_start - orect.lo;
                break;
              case ANY_ORDER:
                assert(0);
                break;
              default:
                assert(0);
            }
          }
          else {
            int new_block_start;
            switch (order) {
              case SRC_FIFO:
                new_block_start = block_start + src_buf->block_size;
                new_block_start = new_block_start - new_block_start % src_buf->block_size;
                block_start = new_block_start;
                done = block_start - irect.lo;
                offset_idx = 0;
                if (block_start > irect.hi) {
                  dso->step();
                  if (dso->any_left) {
                    dsi->step();
                    if (dsi->any_left) {
                      delete dso;
                      dso = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dsi->subrect, *(dst_buf->linearization.get_mapping<DIM>()));
                    }
                  }
                  if (dso->any_left && dsi->any_left) {
                    Rect<DIM> subrect_check;
                    irect = src_buf->linearization.get_mapping<DIM>()->image_dense_subrect(dso->subrect, subrect_check);
                    orect = dso->image;
                    done = 0; offset_idx = 0; block_start = irect.lo; total = irect.hi - irect.lo + 1;
                  }
                }
                break;
              case DST_FIFO:
                new_block_start = block_start + dst_buf->block_size;
                new_block_start = new_block_start - new_block_start % dst_buf->block_size;
                block_start = new_block_start;
                done = block_start - orect.lo;
                offset_idx = 0;
                if (block_start > orect.hi) {
                  dsi->step();
                  if (!dsi->any_left) {
                    dso->step();
                    if (dso->any_left) {
                      delete dsi;
                      dsi = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dso->subrect, *(src_buf->linearization.get_mapping<DIM>()));
                    }
                  }
                  if (dso->any_left && dsi->any_left) {
                    Rect<DIM> subrect_check;
                    orect = dst_buf->linearization.get_mapping<DIM>()->image_dense_subrect(dsi->subrect, subrect_check);
                    irect = dsi->image;
                    done = 0; offset_idx = 0; block_start = orect.lo; total = orect.hi - orect.lo + 1;
                  }
                }
                break;
              case ANY_ORDER:
                assert(0);
                break;
              default:
                assert(0);
            }
          }
        }
        return (nbytes > 0);
      }

      void XferDes::simple_update_bytes_read(int64_t offset, uint64_t size, std::map<int64_t, uint64_t>& segments_read)
      {
        if (pre_XferDes) {
          segments_read.insert(std::pair<int64_t, uint64_t>(offset, size));
          std::map<int64_t, uint64_t>::iterator it;
          bool update = false;
          while (true) {
            it = segments_read.find(bytes_read % src_buf->buf_size);
            if (it == segments_read.end())
              break;
            bytes_read += it->second;
            update = true;
            segments_read.erase(it);
          }
          if (update)
        	  pre_XferDes->update_next_bytes_read(bytes_read);
        }
        else {
          bytes_read += size;
        }
      }

      void XferDes::simple_update_bytes_write(int64_t offset, uint64_t size, std::map<int64_t, uint64_t>& segments_write)
      {
        if (next_XferDes) {
          assert(dst_buf->is_ib);
          segments_write.insert(std::pair<int64_t, uint64_t>(offset, size));
          std::map<int64_t, uint64_t>::iterator it;
          bool update = false;
          while (true) {
            it = segments_write.find(bytes_write % dst_buf->buf_size);
            if (it == segments_write.end())
              break;
            bytes_write += it->second;
            update = true;
            segments_write.erase(it);
          }
          if (next_XferDes != NULL && update)
            next_XferDes->update_pre_bytes_write(bytes_write);
        }
        else {
          bytes_write += size;
        }
      }


      template<unsigned DIM>
      MemcpyXferDes<DIM>::MemcpyXferDes(Channel* _channel, bool has_pre_XferDes,
                                        Buffer* _src_buf, Buffer* _dst_buf,
                                        const char* _src_mem_base, const char* _dst_mem_base,
                                        Domain _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                                        uint64_t _max_req_size, long max_nr, XferOrder _order)
      {
        kind = XferDes::XFER_MEM_CPY;
        channel = _channel;
        order = _order;
        bytes_read = bytes_write = 0;
        pre_XferDes = NULL;
        next_XferDes = NULL;
        next_bytes_read = 0;
        max_req_size = _max_req_size;
        src_buf = _src_buf;
        dst_buf = _dst_buf;
        src_mem_base = _src_mem_base;
        dst_mem_base = _dst_mem_base;
        domain = _domain;
        size_t total_field_size = 0;
        for (int i = 0; i < _oas_vec.size(); i++) {
          OffsetsAndSize oas;
          oas.src_offset = _oas_vec[i].src_offset;
          oas.dst_offset = _oas_vec[i].dst_offset;
          oas.size = _oas_vec[i].size;
          total_field_size += oas.size;
          oas_vec.push_back(oas);
        }
        bytes_total = total_field_size * domain.get_volume();
        pre_bytes_write = (!has_pre_XferDes) ? bytes_total : 0;
        order = _order;
        Rect<DIM> subrect_check;
        switch (order) {
          case SRC_FIFO:
            dsi = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(domain.get_rect<DIM>(), *(src_buf->linearization.get_mapping<DIM>()));
            dso = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dsi->subrect, *(dst_buf->linearization.get_mapping<DIM>()));
            orect = dso->image;
            irect = src_buf->linearization.get_mapping<DIM>()->image_dense_subrect(dso->subrect, subrect_check);
            done = 0; offset_idx = 0; block_start = irect.lo; total = irect.hi - irect.lo + 1;
            break;
          case DST_FIFO:
            dso = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(domain.get_rect<DIM>(), *(dst_buf->linearization.get_mapping<DIM>()));
            dsi = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dso->subrect, *(src_buf->linearization.get_mapping<DIM>()));
            irect = dsi->image;
            orect = dst_buf->linearization.get_mapping<DIM>()->image_dense_subrect(dsi->subrect, subrect_check);
            done = 0; offset_idx = 0; block_start = orect.lo; total = orect.hi - orect.lo + 1;
            break;
          case ANY_ORDER:
            assert(0);
            break;
          default:
            assert(0);
        }
        requests = (MemcpyRequest*) calloc(max_nr, sizeof(MemcpyRequest));
        for (int i = 0; i < max_nr; i++) {
          requests[i].xd = this;
          available_reqs.push(&requests[i]);
        }
        complete_event = GenEventImpl::create_genevent()->current_event();
      }

      template<unsigned DIM>
      long MemcpyXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        MemcpyRequest** mem_cpy_reqs = (MemcpyRequest**) requests;
        long idx = 0;
        while (idx < nr && !available_reqs.empty() && dsi->any_left && dso->any_left) {
          off_t src_start, dst_start;
          size_t nbytes;
          simple_get_request<DIM>(src_start, dst_start, nbytes, dsi, dso, irect, orect, done, offset_idx, block_start, total, min(available_reqs.size(), nr - idx));
          //printf("done = %d, offset_idx = %d\n", done, offset_idx);
          if (nbytes == 0)
            break;
          while (nbytes > 0) {
            size_t req_size = nbytes;
            if (src_buf->is_ib) {
              src_start = src_start % src_buf->buf_size;
              req_size = umin(req_size, src_buf->buf_size - src_start);
            }
            if (dst_buf->is_ib) {
              dst_start = dst_start % dst_buf->buf_size;
              req_size = umin(req_size, dst_buf->buf_size - dst_start);
            }
            mem_cpy_reqs[idx] = (MemcpyRequest*) available_reqs.front();
            available_reqs.pop();
            //printf("[MemcpyXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start - src_buf->alloc_offset, dst_start - dst_buf->alloc_offset, nbytes);
            mem_cpy_reqs[idx]->is_read_done = false;
            mem_cpy_reqs[idx]->is_write_done = false;
            mem_cpy_reqs[idx]->src_buf = (char*)(src_mem_base + src_start);
            mem_cpy_reqs[idx]->dst_buf = (char*)(dst_mem_base + dst_start);
            mem_cpy_reqs[idx]->nbytes = req_size;
            src_start += req_size; // here we don't have to mod src_buf->buf_size since it will be performed in next loop
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
        simple_update_bytes_read(mc_req->src_buf - src_mem_base, mc_req->nbytes, segments_read);
      }

      template<unsigned DIM>
      void MemcpyXferDes<DIM>::notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        MemcpyRequest* mc_req = (MemcpyRequest*) req;
        simple_update_bytes_write(mc_req->dst_buf - dst_mem_base, mc_req->nbytes, segments_write);
        available_reqs.push(req);
      }

      template<unsigned DIM>
      DiskXferDes<DIM>::DiskXferDes(Channel* _channel, bool has_pre_XferDes,
                                    Buffer* _src_buf, Buffer* _dst_buf,
                                    const char *_mem_base, int _fd,
                                    Domain _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                                    uint64_t _max_req_size, long max_nr,
                                    XferOrder _order, XferKind _kind)
      {
        kind = _kind;
        channel = _channel;
        order = _order;
        bytes_read = bytes_write = 0;
        pre_XferDes = NULL;
        next_XferDes = NULL;
        next_bytes_read = 0;
        max_req_size = _max_req_size;
        src_buf = _src_buf;
        dst_buf = _dst_buf;
        fd = _fd;
        mem_base = _mem_base;
        domain = _domain;
        size_t total_field_size = 0;
        for (int i = 0; i < _oas_vec.size(); i++) {
          OffsetsAndSize oas;
          oas.src_offset = _oas_vec[i].src_offset;
          oas.dst_offset = _oas_vec[i].dst_offset;
          oas.size = _oas_vec[i].size;
          total_field_size += oas.size;
          oas_vec.push_back(oas);
        }
        bytes_total = total_field_size * domain.get_volume();
        pre_bytes_write = (!has_pre_XferDes) ? bytes_total : 0;
        complete_event = GenEventImpl::create_genevent()->current_event();

        Rect<DIM> subrect_check;
        switch (order) {
          case SRC_FIFO:
            dsi = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(domain.get_rect<DIM>(), *(src_buf->linearization.get_mapping<DIM>()));
            dso = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dsi->subrect, *(dst_buf->linearization.get_mapping<DIM>()));
            orect = dso->image;
            irect = src_buf->linearization.get_mapping<DIM>()->image_dense_subrect(dso->subrect, subrect_check);
            done = 0; offset_idx = 0; block_start = irect.lo; total = irect.hi - irect.lo + 1;
            break;
          case DST_FIFO:
            dso = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(domain.get_rect<DIM>(), *(dst_buf->linearization.get_mapping<DIM>()));
            dsi = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dso->subrect, *(src_buf->linearization.get_mapping<DIM>()));
            irect = dsi->image;
            orect = dst_buf->linearization.get_mapping<DIM>()->image_dense_subrect(dsi->subrect, subrect_check);
            done = 0; offset_idx = 0; block_start = orect.lo; total = orect.hi - orect.lo + 1;
            break;
          case ANY_ORDER:
            assert(0);
            break;
          default:
            assert(0);
        }

        switch (kind) {
          case XferDes::XFER_DISK_READ:
          {
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
        while (idx < nr && !available_reqs.empty() && dsi->any_left && dso->any_left) {
          off_t src_start, dst_start;
          size_t nbytes;
          simple_get_request<DIM>(src_start, dst_start, nbytes, dsi, dso, irect, orect, done, offset_idx, block_start, total, min(available_reqs.size(), nr - idx));
          //printf("done = %d, offset_idx = %d\n", done, offset_idx);
          if (nbytes == 0)
            break;
          while (nbytes > 0) {
            size_t req_size = nbytes;
            if (src_buf->is_ib) {
              src_start = src_start % src_buf->buf_size;
              req_size = umin(req_size, src_buf->buf_size - src_start);
            }
            if (dst_buf->is_ib) {
              dst_start = dst_start % dst_buf->buf_size;
              req_size = umin(req_size, dst_buf->buf_size - dst_start);
            }
            requests[idx] = available_reqs.front();
            available_reqs.pop();
            requests[idx]->is_read_done = false;
            requests[idx]->is_write_done = false;
            switch (kind) {
              case XferDes::XFER_DISK_READ:
              {
                // printf("[DiskReadXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start - src_buf->alloc_offset, dst_start - dst_buf->alloc_offset, req_size);
                DiskReadRequest* disk_read_req = (DiskReadRequest*) requests[idx];
                disk_read_req->fd = fd;
                disk_read_req->src_offset = src_start;
                disk_read_req->dst_buf = (uint64_t)(mem_base + dst_start);
                disk_read_req->nbytes = req_size;
                break;
              }
              case XferDes::XFER_DISK_WRITE:
              {
                // printf("[DiskWriteXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start - src_buf->alloc_offset, dst_start - dst_buf->alloc_offset, req_size);
                DiskWriteRequest* disk_write_req = (DiskWriteRequest*) requests[idx];
                disk_write_req->fd = fd;
                disk_write_req->src_buf = (uint64_t)(mem_base + src_start);
                disk_write_req->dst_offset = dst_start;
                disk_write_req->nbytes = req_size;
                break;
              }
              default:
                assert(0);
            }
            src_start += req_size; // here we don't have to mod src_buf->buf_size since it will be performed in next loop
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
            offset = ((DiskReadRequest*)req)->src_offset - src_buf->alloc_offset;
            size = ((DiskReadRequest*)req)->nbytes;
            break;
          case XferDes::XFER_DISK_WRITE:
            offset = ((DiskWriteRequest*)req)->src_buf - (int64_t) mem_base;
            size = ((DiskWriteRequest*)req)->nbytes;
            break;
          default:
            assert(0);
        }
        simple_update_bytes_read(offset, size, segments_read);
      }

      template<unsigned DIM>
      void DiskXferDes<DIM>::notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        int64_t offset;
        uint64_t size;
        switch(kind) {
          case XferDes::XFER_DISK_READ:
            offset = ((DiskReadRequest*)req)->dst_buf - (int64_t) mem_base;
            size = ((DiskReadRequest*)req)->nbytes;
            break;
          case XferDes::XFER_DISK_WRITE:
            offset = ((DiskWriteRequest*)req)->dst_offset - dst_buf->alloc_offset;
            size = ((DiskWriteRequest*)req)->nbytes;
            break;
          default:
            assert(0);
        }
        simple_update_bytes_write(offset, size, segments_write);
        available_reqs.push(req);
        //printf("bytes_write = %lu, bytes_total = %lu\n", bytes_write, bytes_total);
      }

#ifdef USE_CUDA
      template<unsigned DIM>
      GPUXferDes<DIM>::GPUXferDes(Channel* _channel, bool has_pre_XferDes,
                                  Buffer* _src_buf, Buffer* _dst_buf,
                                  char* _src_mem_base, char* _dst_mem_base,
                                  Domain _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                                  uint64_t _max_req_size, long max_nr,
                                  XferOrder _order, XferKind _kind,
                                  GPUProcessor* _dst_gpu)
      {
        kind = _kind;
        channel = _channel;
        order = _order;
        bytes_read = bytes_write = 0;
        pre_XferDes = NULL;
        next_XferDes = NULL;
        next_bytes_read = 0;
        max_req_size = _max_req_size;
        src_buf = _src_buf;
        dst_buf = _dst_buf;
        src_mem_base = _src_mem_base;
        dst_mem_base = _dst_mem_base;
        dst_gpu = _dst_gpu;
        domain = _domain;
        size_t total_field_size = 0;
        for (int i = 0; i < _oas_vec.size(); i++) {
          OffsetsAndSize oas;
          oas.src_offset = _oas_vec[i].src_offset;
          oas.dst_offset = _oas_vec[i].dst_offset;
          oas.size = _oas_vec[i].size;
          total_field_size += oas.size;
          oas_vec.push_back(oas);
        }
        bytes_total = total_field_size * domain.get_volume();
        pre_bytes_write = (!has_pre_XferDes) ? bytes_total : 0;
        complete_event = GenEventImpl::create_genevent()->current_event();

        Rect<DIM> subrect_check;
        switch (order) {
          case SRC_FIFO:
            dsi = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(domain.get_rect<DIM>(), *(src_buf->linearization.get_mapping<DIM>()));
            dso = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dsi->subrect, *(dst_buf->linearization.get_mapping<DIM>()));
            orect = dso->image;
            irect = src_buf->linearization.get_mapping<DIM>()->image_dense_subrect(dso->subrect, subrect_check);
            done = 0; offset_idx = 0; block_start = irect.lo; total = irect.hi - irect.lo + 1;
            break;
          case DST_FIFO:
            dso = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(domain.get_rect<DIM>(), *(dst_buf->linearization.get_mapping<DIM>()));
            dsi = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dso->subrect, *(src_buf->linearization.get_mapping<DIM>()));
            irect = dsi->image;
            orect = dst_buf->linearization.get_mapping<DIM>()->image_dense_subrect(dsi->subrect, subrect_check);
            done = 0; offset_idx = 0; block_start = orect.lo; total = orect.hi - orect.lo + 1;
            break;
          case ANY_ORDER:
            assert(0);
            break;
          default:
            assert(0);
        }

        switch (kind) {
          case XferDes::XFER_GPU_TO_FB:
          {
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
        while (idx < nr && !available_reqs.empty() && dsi->any_left && dso->any_left) {
          off_t src_start, dst_start;
          size_t nbytes;
          simple_get_request<DIM>(src_start, dst_start, nbytes, dsi, dso, irect, orect, done, offset_idx, block_start, total, min(available_reqs.size(), nr - idx));
          if (nbytes == 0)
            break;
          while (nbytes > 0) {
            size_t req_size = nbytes;
            if (src_buf->is_ib) {
              src_start = src_start % src_buf->buf_size;
              req_size = umin(req_size, src_buf->buf_size - src_start);
            }
            if (dst_buf->is_ib) {
              dst_start = dst_start % dst_buf->buf_size;
              req_size = umin(req_size, dst_buf->buf_size - dst_start);
            }
            requests[idx] = available_reqs.front();
            available_reqs.pop();
            requests[idx]->is_read_done = false;
            requests[idx]->is_write_done = false;
            switch (kind) {
              case XferDes::XFER_GPU_TO_FB:
              {
                GPUtoFBRequest* gpu_to_fb_req = (GPUtoFBRequest*) requests[idx];
                gpu_to_fb_req->src = src_mem_base + src_start;
                gpu_to_fb_req->dst_offset = dst_start;
                gpu_to_fb_req->nbytes = req_size;
                break;
              }
              case XferDes::XFER_GPU_FROM_FB:
              {
                GPUfromFBRequest* gpu_from_fb_req = (GPUfromFBRequest*) requests[idx];
                gpu_from_fb_req->src_offset = src_start;
                gpu_from_fb_req->dst = dst_mem_base + dst_start;
                gpu_from_fb_req->nbytes = req_size;
                break;
              }
              case XferDes::XFER_GPU_IN_FB:
              {
                GPUinFBRequest* gpu_in_fb_req = (GPUinFBRequest*) requests[idx];
                gpu_in_fb_req->src_offset = src_start;
                gpu_in_fb_req->dst_offset = dst_start;
                gpu_in_fb_req->nbytes = req_size;
                break;
              }
              case XferDes::XFER_GPU_PEER_FB:
              {
                GPUpeerFBRequest* gpu_peer_fb_req = (GPUpeerFBRequest*) requests[idx];
                gpu_peer_fb_req->src_offset = src_start;
                gpu_peer_fb_req->dst_offset = dst_start;
                gpu_peer_fb_req->nbytes = req_size;
                gpu_peer_fb_req->dst_gpu = dst_gpu;
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
            offset = ((GPUtoFBRequest*)req)->src - src_mem_base;
            size = ((GPUtoFBRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GPU_FROM_FB:
            offset = ((GPUfromFBRequest*)req)->src_offset - src_buf->alloc_offset;
            size = ((GPUfromFBRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GPU_IN_FB:
            offset = ((GPUinFBRequest*)req)->src_offset - src_buf->alloc_offset;
            size = ((GPUinFBRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GPU_PEER_FB:
            offset = ((GPUpeerFBRequest*)req)->src_offset - src_buf->alloc_offset;
            size = ((GPUpeerFBRequest*)req)->nbytes;
            break;
          default:
            assert(0);
        }
        simple_update_bytes_read(offset, size, segments_read);
      }

      template<unsigned DIM>
      void GPUXferDes<DIM>::notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        int64_t offset;
        uint64_t size;
        switch (kind) {
          case XferDes::XFER_GPU_TO_FB:
            offset = ((GPUtoFBRequest*)req)->dst_offset - dst_buf->alloc_offset;
            size = ((GPUtoFBRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GPU_FROM_FB:
            offset = ((GPUfromFBRequest*)req)->dst - dst_mem_base;
            size = ((GPUfromFBRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GPU_IN_FB:
            offset = ((GPUinFBRequest*)req)->dst_offset - dst_buf->alloc_offset;
            size = ((GPUfromFBRequest*)req)->nbytes;
            break;
          case XferDes::XFER_GPU_PEER_FB:
            offset = ((GPUpeerFBRequest*)req)->dst_offset - dst_buf->alloc_offset;
            size = ((GPUpeerFBRequest*)req)->nbytes;
            break;
          default:
            assert(0);
        }
        simple_update_bytes_write(offset, size, segments_write);
        available_reqs.push(req);
      }
#endif

#ifdef USE_HDF
      template<unsigned DIM>
      HDFXferDes<DIM>::HDFXferDes(Channel* _channel, bool has_pre_XferDes,
                                  Buffer* _src_buf, Buffer* _dst_buf,
                                  char* _mem_base, HDFMemory::HDFMetadata* _hdf_metadata,
                                  Domain _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                                  long max_nr, XferOrder _order, XferKind _kind)
      {
        kind = _kind;
        order = _order;
        bytes_read = bytes_write = 0;
        pre_XferDes = NULL;
        next_XferDes = NULL;
        next_bytes_read = 0;
        src_buf = _src_buf;
        dst_buf = _dst_buf;
        // for now, we didn't consider HDF transfer for intermediate buffer
        // since ib may involve a different address space model
        assert(!src_buf->is_ib);
        assert(!dst_buf->is_ib);
        assert(!has_pre_XferDes);
        mem_base = _mem_base;
        hdf_metadata = _hdf_metadata;
        domain = _domain;
        size_t total_field_size = 0;
        for (int i = 0; i < _oas_vec.size(); i++) {
          OffsetsAndSize oas;
          oas.src_offset = _oas_vec[i].src_offset;
          oas.dst_offset = _oas_vec[i].dst_offset;
          oas.size = _oas_vec[i].size;
          total_field_size += oas.size;
          oas_vec.push_back(oas);
        }
        bytes_total = total_field_size * domain.get_volume();
        pre_bytes_write = (!has_pre_XferDes) ? bytes_total : 0;
        complete_event = GenEventImpl::create_genevent()->current_event();

        Rect<DIM> subrect_check;
        switch (kind) {
          case XferDes::XFER_HDF_READ:
          {
            HDFReadRequest* hdf_read_reqs = (HDFReadRequest*) calloc(max_nr, sizeof(HDFReadRequest));
            for (int i = 0; i < max_nr; i++) {
              hdf_read_reqs[i].xd = this;
              available_reqs.push(&hdf_read_reqs[i]);
            }
            requests = hdf_read_reqs;
            lsi = new GenericLinearSubrectIterator<Mapping<DIM, 1> >(domain.get_rect<DIM>(), dst_buf->linearization);
            // Make sure instance involves FortranArrayLinearization
            assert(lsi->strides[0] == 1);
            // This is kind of tricky, but to avoid recomputing hdf dataset idx for every oas entry,
            // we change the src/dst offset to hdf dataset idx
            for (fit = oas_vec.begin(); fit != oas_vec.end(); fit++) {
              off_t offset = 0;
              int idx = 0;
              while (offset < (*fit).src_offset) {
                offset += H5Tget_size(hdf_metadata->datatype_ids[idx]);
                idx++;
              }
              assert(offset == (*fit).src_offset);
              (*fit).src_offset = idx;
            }
            fit = oas_vec.begin();
            pir = new GenericPointInRectIterator<DIM>(domain.get_rect<DIM>());
            break;
          }
          case XferDes::XFER_HDF_WRITE:
          {
            HDFWriteRequest* hdf_write_reqs = (HDFWriteRequest*) calloc(max_nr, sizeof(HDFWriteRequest));
            for (int i = 0; i < max_nr; i++) {
              hdf_write_reqs[i].xd = this;
              available_reqs.push(&hdf_write_reqs[i]);
            }
            requests = hdf_write_reqs;
            lsi = new GenericLinearSubrectIterator<Mapping<DIM, 1> >(domain.get_rect<DIM>(), src_buf->linearization);
            // Make sure instance involves FortranArrayLinearization
            assert(lsi->strides[0] == 1);
            // This is kind of tricky, but to avoid recomputing hdf dataset idx for every oas entry,
            // we change the src/dst offset to hdf dataset idx
            for (fit = oas_vec.begin(); fit != oas_vec.end(); fit++) {
              off_t offset = 0;
              int idx = 0;
              while (offset < (*fit).dst_offset) {
                offset += H5Tget_size(hdf_metadata->datatype_ids[idx]);
                idx++;
              }
              assert(offset == (*fit).dst_offset);
              (*fit).dst_offset = idx;
            }
            fit = oas_vec.begin();
            pir = new GenericPointInRectIterator<DIM>(domain.get_rect<DIM>());
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
          off_t src_start, dst_start;
          requests[ns] = available_reqs.front();
          available_reqs.pop();
          requests[ns]->is_read_done = false;
          requests[ns]->is_write_done = false;
          int todo;
          switch (kind) {
            case XferDes::XFER_HDF_READ:
            {
              // Recall that src_offset means the index of the involving dataset in hdf file
              off_t hdf_idx = fit->src_offset;
              size_t elemnt_size = H5Tget_size(hdf_metadata->datatype_ids[hdf_idx]);
              todo = min(pir->r.hi[0] - pir->p[0], dst_buf->block_size - lsi->mapping.image(pir->p) % dst_buf->block_size);
              HDFReadRequest* hdf_read_req = (HDFReadRequest*) requests[ns];
              hdf_read_req->dataset_id = hdf_metadata->dataset_ids[hdf_idx];
              hdf_read_req->mem_type_id = hdf_metadata->datatype_ids[hdf_idx];
              int count[DIM];
              for (int i = 0; i < DIM; i++) count[i] = 1;
              count[0] = todo;
              hdf_read_req->file_space_id = H5Dget_space(hdf_metadata->dataset_ids[hdf_idx]);
              // HDF dimension always start with zero, but Legion::Domain may start with any integer
              // We need to deal with the offset between them here
              Point<DIM> offset(hdf_metadata->lo);
              herr_t ret = H5Sselect_hyperslab(hdf_read_req->file_space_id, H5S_SELECT_SET, pir->p - offset, NULL, count, NULL);
              hdf_read_req->mem_space_id = H5Screate_simple(DIM, count, NULL);
              off_t dst_offset = calc_mem_loc(dst_buf->alloc_offset, fit->dst_offset, fit->size,
                                              dst_buf->elmt_size, dst_buf->block_size, lsi->mapping.image(pir->p));
              hdf_read_req->dst = mem_base + dst_offset;
              hdf_read_req->nbytes = todo * elemnt_size;
              break;
            }
            case XferDes::XFER_HDF_WRITE:
            {
              // Recall that src_offset means the index of the involving dataset in hdf file
              off_t hdf_idx = fit->dst_offset;
              size_t elemnt_size = H5Tget_size(hdf_metadata->datatype_ids[hdf_idx]);
              int todo = min(pir->r.hi[0] - pir->p[0], src_buf->block_size - lsi->mapping.image(pir->p) % src_buf->block_size);
              HDFWriteRequest* hdf_write_req = (HDFWriteRequest*) requests[ns];
              hdf_write_req->dataset_id = hdf_metadata->dataset_ids[hdf_idx];
              hdf_write_req->mem_type_id = hdf_metadata->datatype_ids[hdf_idx];
              int count[DIM];
              for (int i = 0; i < DIM; i++) count[i] = 1;
              count[0] = todo;
              hdf_write_req->file_space_id = H5Dget_space(hdf_metadata->dataset_ids[hdf_idx]);
              // HDF dimension always start with zero, but Legion::Domain may start with any integer
              // We need to deal with the offset between them here
              Point<DIM> offset(hdf_metadata->lo);
              herr_t ret = H5Sselect_hyperslab(hdf_write_req->file_space_id, H5S_SELECT_SET, pir->p - offset, NULL, count, NULL);
              hdf_write_req->mem_space_id = H5Screate_simple(DIM, count, NULL);
              off_t src_offset = calc_mem_loc(src_buf->alloc_offset, fit->src_offset, fit->size,
                                              src_buf->elmt_size, src_buf->block_size, lsi->mapping.image(pir->p));
              hdf_write_req->src = mem_base + src_offset;
              hdf_write_req->nbytes = todo * elemnt_size;
              break;
            }
            default:
              assert(0);
          }
          while(todo > 0) pir++;
          if(!pir) {
            fit++;
            delete pir;
            pir = GenericPointInRectIterator<DIM>(domain.get_rect<DIM>());
          }
          ns ++;
        }
        return ns;
      }

      template<unsigned DIM>
      void HDFXferDes<DIM>::notify_request_read_done(Request* req)
      {
        req->is_read_done = true;
        // currently we don't support ib case
        assert(!pre_XferDes);
        switch (kind) {
          case XferDes::XFER_HDF_READ:
            bytes_read += ((HDFReadRequest*)req)->nbytes;
            break;
          case XferDes::XFER_HDF_WRITE:
            bytes_read += ((HDFWriteRequest*)req)->nbytes;
            break;
          default:
            assert(0);
        }
      }

      template<unsigned DIM>
      void HDFXferDes<DIM>::notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        // currently we don't support ib case
        assert(!next_XferDes);
        switch (kind) {
          case XferDes::XFER_HDF_READ:
            bytes_write += ((HDFReadRequest*)req)->nbytes;
            break;
          case XferDes::XFER_HDF_WRITE:
            bytes_write += ((HDFWriteRequest*)req)->nbytes;
            break;
          default:
            assert(0);
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

#ifdef USE_CUDA
      GPUChannel::GPUChannel(GPUProcessor* _src_gpu, long max_nr, XferDes::XferKind _kind)
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
              gpu_to_fb_reqs[i]->complete_event = GenEventImpl::create_genevent()->current_event();
              src_gpu->copy_to_fb(gpu_to_fb_reqs[i]->dst_offset,
                                  gpu_to_fb_reqs[i]->src,
                                  gpu_to_fb_reqs[i]->nbytes,
                                  Event::NO_EVENT,
                                  gpu_to_fb_reqs[i]->complete_event);
              pending_copies.push_back(gpu_to_fb_reqs[i]);
            }
            break;
          }
          case XferDes::XFER_GPU_FROM_FB:
          {
        	GPUfromFBRequest** gpu_from_fb_reqs = (GPUfromFBRequest**) requests;
        	for (int i = 0; i < nr; i++) {
        	  gpu_from_fb_reqs[i]->complete_event = GenEventImpl::create_genevent()->current_event();
        	  src_gpu->copy_from_fb(gpu_from_fb_reqs[i]->dst,
                                    gpu_from_fb_reqs[i]->src_offset,
                                    gpu_from_fb_reqs[i]->nbytes,
                                    Event::NO_EVENT,
                                    gpu_from_fb_reqs[i]->complete_event);
        	  pending_copies.push_back(gpu_from_fb_reqs[i]);
        	}
        	break;
          }
          case XferDes::XFER_GPU_IN_FB:
          {
            GPUinFBRequest** gpu_in_fb_reqs = (GPUinFBRequest**) requests;
            for (int i = 0; i < nr; i++) {
              gpu_in_fb_reqs[i]->complete_event = GenEventImpl::create_genevent()->current_event();
              src_gpu->copy_within_fb(gpu_in_fb_reqs[i]->dst_offset,
                                      gpu_in_fb_reqs[i]->src_offset,
                                      gpu_in_fb_reqs[i]->nbytes,
                                      Event::NO_EVENT,
                                      gpu_in_fb_reqs[i]->complete_event);
              pending_copies.push_back(gpu_in_fb_reqs[i]);
            }
            break;
          }
          case XferDes::XFER_GPU_PEER_FB:
          {
            GPUpeerFBRequest** gpu_peer_fb_reqs = (GPUpeerFBRequest**) requests;
            for (int i = 0; i < nr; i++) {
              gpu_peer_fb_reqs[i]->complete_event = GenEventImpl::create_genevent()->current_event();
              src_gpu->copy_to_peer(gpu_peer_fb_reqs[i]->dst_gpu,
                                    gpu_peer_fb_reqs[i]->dst_offset,
                                    gpu_peer_fb_reqs[i]->src_offset,
                                    gpu_peer_fb_reqs[i]->nbytes,
                                    Event::NO_EVENT,
                                    gpu_peer_fb_reqs[i]->complete_event);
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
              if (gpu_to_fb_req->complete_event.has_triggered()) {
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
              if (gpu_from_fb_req->complete_event.has_triggered()) {
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
              if (gpu_in_fb_req->complete_event.has_triggered()) {
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
              GPUinFBRequest* gpu_peer_fb_req = (GPUinFBRequest*)pending_copies.front();
              if (gpu_peer_fb_req->complete_event.has_triggered()) {
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
              H5Dread(request->dataset_id, request->mem_type_id, request->mem_space_id, request->file_space_id, H5P_DEFAULT, request->dst);
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
              H5Dwrite(request->dataset_id, request->mem_type_id, request->mem_space_id, request->file_space_id, H5P_DEFAULT, request->src);
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
      void DMAThread::dma_therad_loop() {
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
            if (nr > max_nr)
              nr = max_nr;
            if (nr == 0)
              continue;
            std::vector<XferDes*> finish_xferdes;
            PriorityXferDesQueue::iterator it2;
            for (it2 = it->second->begin(); it2 != it->second->end(); it2++) {
              assert((*it2)->channel == it->first);
              long nr_got = (*it2)->get_requests(requests, nr);
              long nr_submitted = it->first->submit(requests, nr_got);
              nr -= nr_submitted;
              assert(nr_got == nr_submitted);
              if ((*it2)->is_done()) {
                finish_xferdes.push_back(*it2);
              }
              if (nr ==0)
                break;
            }
            while(!finish_xferdes.empty()) {
              delete finish_xferdes.back();
              it->second->erase(finish_xferdes.back());
              finish_xferdes.pop_back();
            }
          }
        }
      }

#ifdef OLD_DMA_THREAD_CODE
      void* DMAThread::start(void* arg)
      {
        printf("[DMAThread] start...\n");
        DMAThread* dma = (DMAThread*) arg;
        while (true) {
          if (dma->is_stopped)
            break;
          //printf("[DMAThread] CP#1\n");
          pthread_mutex_lock(&dma->channel_lock);
          std::vector<Channel*>::iterator it;
          for (it = dma->channel_queue.begin(); it != dma->channel_queue.end(); it++) {
            //printf("[DMAThread] CP#2\n");
            (*it)->pull();
            //printf("[DMAThread] CP#3\n");
            long nr = (*it)->available();
            //printf("[DMAThread] available = %ld\n", nr);
            if (nr > dma->max_nr)
              nr = dma->max_nr;
            if (nr == 0)
              continue;
            pthread_mutex_lock(&dma->xferDes_lock);
            std::vector<XferDes*>::iterator it2;
            for (it2 = dma->xferDes_queue.begin(); it2 != dma->xferDes_queue.end(); it2++) {
              if ((*it2)->channel == (*it)) {
                assert((*it2)->kind == (*it)->kind);
                long nr_got = (*it2)->get_requests(dma->requests, nr);
                //printf("[DMAThread] nr_got = %ld\n", nr_got);
                long nr_submitted = (*it)->submit(dma->requests, nr_got);
                nr -= nr_submitted;
                //printf("[DMAThread] nr_submitted = %ld\n", nr_submitted);
                assert(nr_got == nr_submitted);
              }
            }
            pthread_mutex_unlock(&dma->xferDes_lock);
          }
          pthread_mutex_unlock(&dma->channel_lock);
        }
        return NULL;
      }
#endif

      template class MemcpyXferDes<1>;
      template class MemcpyXferDes<2>;
      template class MemcpyXferDes<3>;
      template class DiskXferDes<1>;
      template class DiskXferDes<2>;
      template class DiskXferDes<3>;
#ifdef USE_CUDA
      template class GPUXferDes<1>;
      template class GPUXferDes<2>;
      template class GPUXferDes<3>;
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
