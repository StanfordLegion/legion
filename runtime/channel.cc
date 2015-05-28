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
      MemcpyXferDes<DIM>::MemcpyXferDes(Channel* _channel,
                     bool has_pre_XferDes, bool has_next_XferDes,
                     Buffer* _src_buf, Buffer* _dst_buf,
                     char* _src_mem_base, char* _dst_mem_base,
                     Domain _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                     uint64_t _bytes_total, uint64_t _max_req_size, long max_nr,
                     XferOrder _order)
      {
        kind = XferDes::XFER_MEM_CPY;
        channel = _channel;
        order = _order;
        bytes_read = bytes_write = 0;
        pre_bytes_write = (!has_pre_XferDes) ? _bytes_total : 0;
        next_bytes_read = 0;
        bytes_total = _bytes_total;
        max_req_size = _max_req_size;
        src_buf = _src_buf;
        dst_buf = _dst_buf;
        src_mem_base = _src_mem_base;
        dst_mem_base = _dst_mem_base;
        domain = _domain;
        for (int i = 0; i < _oas_vec.size(); i++) {
          OffsetsAndSize oas;
          oas.src_offset = _oas_vec[i].src_offset;
          oas.dst_offset = _oas_vec[i].dst_offset;
          oas.size = _oas_vec[i].size;
          oas_vec.push_back(oas);
        }
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
      }

      template<unsigned DIM>
      long MemcpyXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        MemcpyRequest** mem_cpy_reqs = (MemcpyRequest**) requests;
        long idx = 0;
        while (idx < nr && !available_reqs.empty() && dsi && dso) {
          off_t src_start = calc_mem_loc(src_buf->alloc_offset, oas_vec[offset_idx].src_offset, oas_vec[offset_idx].size,
                                         src_buf->elmt_size, src_buf->block_size, done + irect.lo);
          off_t dst_start = calc_mem_loc(dst_buf->alloc_offset, oas_vec[offset_idx].dst_offset, oas_vec[offset_idx].size,
                                         dst_buf->elmt_size, dst_buf->block_size, done + orect.lo);
          size_t nbytes = 0;
          bool scatter_src_ib = false, scatter_dst_ib = false;
          while (true) {
            // check to see if we can generate next request
            int src_in_block = src_buf->block_size - (done + irect.lo) % src_buf->block_size;
            int dst_in_block = dst_buf->block_size - (done + orect.lo) % dst_buf->block_size;
            int todo = min((max_req_size - nbytes) / oas_vec[offset_idx].size, min(total - done, min(src_in_block, dst_in_block)));
            // make sure we have source data ready
            if (src_buf->is_ib) {
              todo = min(todo, (src_buf->alloc_offset + pre_bytes_write - src_start) / oas_vec[offset_idx].size);
              scatter_src_ib = scatter_src_ib || scatter_ib(src_start, nbytes + todo * oas_vec[offset_idx].size, src_buf->buf_size);
            }
            // make sure there are enough space in destination
            if (dst_buf->is_ib) {
              todo = min(todo, (dst_buf->alloc_offset + next_bytes_read + dst_buf->buf_size - dst_start) / oas_vec[offset_idx].size);
              scatter_dst_ib = scatter_dst_ib || scatter_ib(dst_start, nbytes + todo * oas_vec[offset_idx].size, dst_buf->buf_size);
            }
            if((scatter_src_ib && scatter_dst_ib && (idx + 3 > nr || available_reqs.size() < 3))
            ||((scatter_src_ib || scatter_dst_ib) && (idx + 2 > nr || available_reqs.size() < 2)))
              break;
            printf("min(%d, %d, %d)\n = ", (int)(max_req_size - nbytes) / oas_vec[offset_idx].size, total - done, min(src_in_block, dst_in_block));
            printf("todo = %d, size = %d\n", todo, oas_vec[offset_idx].size);
            nbytes += todo * oas_vec[offset_idx].size;
            // see if we can batchmore
            if (todo == src_in_block && todo == dst_in_block && offset_idx + 1 < oas_vec.size()
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

          if (((done + irect.lo) % src_buf->block_size == 0 && order == SRC_FIFO)
            ||((done + orect.lo) % dst_buf->block_size == 0 && order == DST_FIFO)
            || done == total) {
            offset_idx ++;
          }
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
                      free(dso);
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
                      free(dsi);
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
          if (nbytes == 0)
            break;
          while (nbytes > 0) {
            size_t req_size = nbytes;
            if (scatter_src_ib)
              req_size = umin(req_size, src_buf->buf_size - src_start % src_buf->buf_size);
            if (scatter_dst_ib)
              req_size = umin(req_size, dst_buf->buf_size - dst_start % dst_buf->buf_size);
            mem_cpy_reqs[idx] = (MemcpyRequest*) available_reqs.front();
            available_reqs.pop();
            printf("[MemcpyXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start - src_buf->alloc_offset, dst_start - dst_buf->alloc_offset, nbytes);
            mem_cpy_reqs[idx]->is_read_done = false;
            mem_cpy_reqs[idx]->is_write_done = false;
            mem_cpy_reqs[idx]->src_buf = (char*)(src_mem_base + src_start);
            mem_cpy_reqs[idx]->dst_buf = (char*)(dst_mem_base + dst_start);
            mem_cpy_reqs[idx]->nbytes = nbytes;
            idx++;
          }
        }
        return idx;
      }

      template<unsigned DIM>
      void MemcpyXferDes<DIM>::notify_request_read_done(Request* req)
      {
        req->is_read_done = true;
        DiskWriteRequest* dw_req = (DiskWriteRequest*) req;
        if (pre_XferDes) {
          segments_read.insert(std::pair<int64_t, uint64_t>(dw_req->src_buf - src_buf->alloc_offset, dw_req->nbytes));
          std::map<int64_t, uint64_t>::iterator it;
          bool update = false;
          while (true) {
            it = segments_read.find(bytes_read);
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
          bytes_read += dw_req->nbytes;
        }
      }

      template<unsigned DIM>
      void MemcpyXferDes<DIM>::notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        DiskWriteRequest* dw_req = (DiskWriteRequest*) req;
        if (next_XferDes) {
          segments_write.insert(std::pair<int64_t, uint64_t>(dw_req->dst_offset - dst_buf->alloc_offset, dw_req->nbytes));
          std::map<int64_t, uint64_t>::iterator it;
          bool update = false;
          while (true) {
            it = segments_write.find(bytes_write);
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
          bytes_write += dw_req->nbytes;
        }
        available_reqs.push(req);
      }

      template<unsigned DIM>
      DiskWriteXferDes<DIM>::DiskWriteXferDes(
                       Channel* _channel, int _fd,
                       bool has_pre_XferDes, bool has_next_XferDes,
                       Buffer* _src_buf, Buffer* _dst_buf,
                       uint64_t _bytes_total, long max_nr)
      {
        kind = XferDes::XFER_DISK_WRITE;
        fd = _fd;
        channel = _channel;
        bytes_read = bytes_write = 0;
        pre_XferDes = NULL;
        pre_bytes_write = (!has_pre_XferDes) ? _bytes_total : 0;
        next_XferDes = NULL;
        next_bytes_read = 0;
        bytes_total = _bytes_total;
        src_buf = _src_buf;
        dst_buf = _dst_buf;
        // Make sure this is the last XferDes in the chain
        assert(!has_next_XferDes);
        // set iterator
        dsi = Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(domain.get_rect<DIM>(), *(src_buf->linearization.get_mapping<DIM>()));
        dso = Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dsi.subrect, *(dst_buf->linearization.get_mapping<DIM>()));
        Rect<DIM> subrect_check;
        irect = src_buf->linearization.get_mapping<DIM>()->image_dense_subrect(dso.subrect, subrect_check);
        orect = dso.image;
        done = 0; offset_idx = 0; block_start = irect.lo;
        //allocate memory for DiskWRiteRequests, and push them into
        // available_reqs
        requests = (DiskWriteRequest*) calloc(max_nr, sizeof(DiskWriteRequest));
        for (int i = 0; i < max_nr; i++) {
          requests[i].xd = this;
          available_reqs.push(&requests[i]);
        }
      }

      template<unsigned DIM>
      long DiskWriteXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        DiskWriteRequest** disk_write_reqs = (DiskWriteRequest**) requests;
        long idx = 0;
        while (idx < nr && !available_reqs.empty() && dsi && dso) {
          off_t src_start = calc_mem_loc(src_buf->alloc_offset, oas_vec[offset_idx].src_offset, oas_vec[offset_idx].size,
                                         src_buf->elmt_size, src_buf->block_size, done + irect.lo);
          off_t dst_start = calc_mem_loc(dst_buf->alloc_offset, oas_vec[offset_idx].dst_offset, oas_vec[offset_idx].size,
                                         dst_buf->elmt_size, dst_buf->block_size, done + orect.lo);
          size_t nbytes = 0;
          while (true) {
            // check to see if we can generate next request
            int src_in_block = src_buf->block_size - (done + irect.lo) % src_buf->block_size;
            int dst_in_block = dst_buf->block_size - (done + orect.lo) % dst_buf->block_size;
            int todo = min((max_req_size - nbytes) / oas_vec[offset_idx].size, min(irect.hi - irect.lo + 1 - done, min(src_in_block, dst_in_block)));
            // make sure if we have source data ready
            if (src_buf->is_ib)
              todo = min(todo, (src_buf->alloc_offset + pre_bytes_write - src_start) / oas_vec[offset_idx].size);
            nbytes += todo * oas_vec[offset_idx].size;
            // see if we can batch more
            if (todo == src_in_block && todo == dst_in_block && offset_idx + 1 < oas_vec.size()
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
          if ((done + irect.lo) % src_buf->block_size == 0 || done == irect.hi - irect.lo + 1) {
            offset_idx ++;
          }
          if (offset_idx < oas_vec.size()) {
            done = block_start - irect.lo;
          }
          else {
            int new_block_start = block_start + src_buf->block_size;
            new_block_start = new_block_start - new_block_start % src_buf->block_size;
            block_start = new_block_start;
            done = block_start - irect.lo;
            offset_idx = 0;
            if (block_start > irect.hi) {
              dso++;
              if (!dso) {
                dsi++;
                if (dsi) {
                  dso = Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dsi.subrect, *(dst_buf->linearization.get_mapping<DIM>()));
                }
              }
              if (dso && dsi) {
                Rect<DIM> subrect_check;
                irect = src_buf->linearization.get_mapping<DIM>()->image_dense_subrect(dso.subrect, subrect_check);
                orect = dso.image;
                done = 0; offset_idx = 0; block_start = irect.lo;
              }
            }
          }
          if (nbytes == 0)
            break;
          disk_write_reqs[idx] = (DiskWriteRequest*) available_reqs.front();
          available_reqs.pop();
          disk_write_reqs[idx]->fd = fd;
          disk_write_reqs[idx]->is_read_done = false;
          disk_write_reqs[idx]->is_write_done = false;
          disk_write_reqs[idx]->src_buf = src_start;
          disk_write_reqs[idx]->dst_offset = dst_start;
          disk_write_reqs[idx]->nbytes = nbytes;
          idx ++;
        }
        return idx;
      }

      template<unsigned DIM>
      void DiskWriteXferDes<DIM>::notify_request_read_done(Request* req)
      {
        req->is_read_done = true;
        DiskWriteRequest* dw_req = (DiskWriteRequest*) req;
        if (pre_XferDes) {
          segments_read.insert(std::pair<int64_t, uint64_t>(dw_req->src_buf - src_buf->alloc_offset, dw_req->nbytes));
          std::map<int64_t, uint64_t>::iterator it;
          bool update = false;
          while (true) {
            it = segments_read.find(bytes_read);
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
          bytes_read += dw_req->nbytes;
        }
      }

      template<unsigned DIM>
      void DiskWriteXferDes<DIM>::notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        DiskWriteRequest* dw_req = (DiskWriteRequest*) req;
        if (next_XferDes) {
          segments_write.insert(std::pair<int64_t, uint64_t>(dw_req->dst_offset - dst_buf->alloc_offset, dw_req->nbytes));
          std::map<int64_t, uint64_t>::iterator it;
          bool update = false;
          while (true) {
            it = segments_write.find(bytes_write);
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
          bytes_write += dw_req->nbytes;
        }
        available_reqs.push(req);
      }

      template<unsigned DIM>
      DiskReadXferDes<DIM>::DiskReadXferDes(
                       Channel* _channel, int _fd,
                       bool has_pre_XferDes, bool has_next_XferDes,
                       Buffer* _src_buf, Buffer* _dst_buf,
                       uint64_t _bytes_total, long max_nr)
      {
        kind = XferDes::XFER_DISK_READ;
        fd = _fd;
        channel = _channel;
        bytes_read = bytes_write = 0;
        pre_XferDes = NULL;
        pre_bytes_write = _bytes_total;
        next_XferDes = NULL;
        next_bytes_read = 0;
        bytes_total = _bytes_total;
        src_buf = _src_buf;
        dst_buf = _dst_buf;
        // Make sure this is the last XferDes in the chain
        assert(!has_pre_XferDes);
        // set iterator
        dsi = Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(domain.get_rect<DIM>(), src_buf->linearization.get_mapping<DIM>());
        dso = Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dsi.subrect, dst_buf->linearization.get_mapping<DIM>());
        Rect<DIM> subrect_check;
        irect = src_buf->linearization.get_mapping<DIM>()->image_dense_subrect(dso.subrect, subrect_check);
        orect = dso.image;
        done = 0; offset_idx = 0; block_start = irect.lo;
        //allocate memory for DiskWRiteRequests, and push them into
        // available_reqs
        requests = (DiskWriteRequest*) calloc(max_nr, sizeof(DiskWriteRequest));
        for (int i = 0; i < max_nr; i++) {
          requests[i].xd = this;
          available_reqs.push(&requests[i]);
        }
      }

      template<unsigned DIM>
      long DiskReadXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        DiskReadRequest** disk_read_reqs = (DiskReadRequest**) requests;
        long idx = 0;
        /*
        while(idx < nr && !available_reqs.empty()) {
          disk_read_reqs[idx] = (DiskReadRequest*) available_reqs.front();
          available_reqs.pop();
          disk_read_reqs[idx]->fd = fd;
          disk_read_reqs[idx]->is_read_done = false;
          disk_read_reqs[idx]->is_write_done = false;
          // TODO: generate copy requests
          uint64_t size;
          size = gen_next_request(disk_read_reqs[idx]);
          bytes_submit += size;
          idx ++;
        }
        */
        return idx;
      }

      template<unsigned DIM>
      void DiskReadXferDes<DIM>::notify_request_read_done(Request* req)
      {
        req->is_read_done = true;
        DiskReadRequest* dw_req = (DiskReadRequest*) req;
        if (pre_XferDes) {
          segments_read.insert(std::pair<int64_t, uint64_t>(dw_req->src_offset - src_buf->alloc_offset, dw_req->nbytes));
          std::map<int64_t, uint64_t>::iterator it;
          bool update = false;
          while (true) {
            it = segments_read.find(bytes_read);
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
          bytes_read += dw_req->nbytes;
        }
      }

      template<unsigned DIM>
      void DiskReadXferDes<DIM>::notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        DiskReadRequest* dw_req = (DiskReadRequest*) req;
        if (next_XferDes) {
          segments_write.insert(std::pair<int64_t, uint64_t>(dw_req->dst_buf - dst_buf->alloc_offset, dw_req->nbytes));
          std::map<int64_t, uint64_t>::iterator it;
          bool update = false;
          while (true) {
            it = segments_write.find(bytes_write);
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
          bytes_write += dw_req->nbytes;
        }
        available_reqs.push(req);
      }

      MemcpyChannel::MemcpyChannel(long max_nr, MemcpyThread* _worker)
      {
        kind = XferDes::XFER_MEM_CPY;
        worker = _worker;
        capacity = max_nr;
        cbs = (MemcpyRequest**) calloc(max_nr, sizeof(MemcpyRequest*));
      }

      MemcpyChannel::~MemcpyChannel()
      {
        free(cbs);
      }

      long MemcpyChannel::submit(Request** requests, long nr)
      {
        MemcpyRequest** mem_cpy_reqs = (MemcpyRequest**) requests;
        long ns = worker->submit(mem_cpy_reqs, nr);
        return ns;
      }

      void MemcpyChannel::pull()
      {
        while (true) {
          long np = worker->pull(cbs, capacity);
          for (int i = 0; i < np; i++) {
            cbs[i]->xd->notify_request_read_done(cbs[i]);
            cbs[i]->xd->notify_request_write_done(cbs[i]);
          }
          if (np != capacity)
            break;
        }
      }

      long MemcpyChannel::available()
      {
        return capacity;
      }

      DiskReadChannel::DiskReadChannel(long max_nr)
      {
        kind = XferDes::XFER_DISK_READ;
        ctx = 0;
        capacity = max_nr;
        int ret = io_setup(max_nr, &ctx);
        assert(ret >= 0);
        assert(available_cb.empty());
        cb = (struct iocb*) calloc(max_nr, sizeof(struct iocb));
        cbs = (struct iocb**) calloc(max_nr, sizeof(struct iocb*));
        events = (struct io_event*) calloc(max_nr, sizeof(struct io_event));
        for (int i = 0; i < max_nr; i++) {
          memset(&cb[i], 0, sizeof(cb[i]));
          cb[i].aio_lio_opcode = IOCB_CMD_PREAD;
          available_cb.push_back(&cb[i]);
        }
      }

      DiskReadChannel::~DiskReadChannel()
      {
        io_destroy(ctx);
        free(cb);
        free(cbs);
        free(events);
      }

      long DiskReadChannel::submit(Request** requests, long nr)
      {
        DiskReadRequest** disk_read_reqs = (DiskReadRequest**) requests;
        int ns = 0;
        while (ns < nr && !available_cb.empty()) {
          cbs[ns] = available_cb.back();
          available_cb.pop_back();
          cbs[ns]->aio_fildes = disk_read_reqs[ns]->fd;
          cbs[ns]->aio_data = (uint64_t) (disk_read_reqs[ns]);
          cbs[ns]->aio_buf = disk_read_reqs[ns]->dst_buf;
          cbs[ns]->aio_offset = disk_read_reqs[ns]->src_offset;
          cbs[ns]->aio_nbytes = disk_read_reqs[ns]->nbytes;
          ns++;
        }
        assert(ns == nr);
        int ret = io_submit(ctx, ns, cbs);
        if (ret < 0) {
          perror("io_submit error");
        }
        return ret;
      }

      void DiskReadChannel::pull()
      {
        int nr = io_getevents(ctx, 0, capacity, events, NULL);
        if (nr < 0)
          perror("io_getevents error");
        for (int i = 0; i < nr; i++) {
          DiskReadRequest* req = (DiskReadRequest*) events[i].data;
          struct iocb* ret_cb = (struct iocb*) events[i].obj;
          available_cb.push_back(ret_cb);
          assert(events[i].res == (int64_t)ret_cb->aio_nbytes);
          req->xd->notify_request_read_done(req);
          req->xd->notify_request_write_done(req);
        }
      }

      long DiskReadChannel::available()
      {
        return available_cb.size();
      }

      DiskWriteChannel::DiskWriteChannel(long max_nr)
      {
        kind = XferDes::XFER_DISK_WRITE;
        ctx = 0;
        capacity = max_nr;
        int ret = io_setup(max_nr, &ctx);
        assert(ret >= 0);
        assert(available_cb.empty());
        cb = (struct iocb*) calloc(max_nr, sizeof(struct iocb));
        cbs = (struct iocb**) calloc(max_nr, sizeof(struct iocb*));
        events = (struct io_event*) calloc(max_nr, sizeof(struct io_event));
        for (int i = 0; i < max_nr; i++) {
          memset(&cb[i], 0, sizeof(cb[i]));
          cb[i].aio_lio_opcode = IOCB_CMD_PWRITE;
          available_cb.push_back(&cb[i]);
        }
      }

      DiskWriteChannel::~DiskWriteChannel()
      {
        io_destroy(ctx);
        free(cb);
        free(cbs);
        free(events);
      }

      long DiskWriteChannel::submit(Request** requests, long nr)
      {
        DiskWriteRequest** disk_write_reqs = (DiskWriteRequest**) requests;
        int ns = 0;
        while (ns < nr && !available_cb.empty()) {
          cbs[ns] = available_cb.back();
          available_cb.pop_back();
          cbs[ns]->aio_fildes = disk_write_reqs[ns]->fd;
          cbs[ns]->aio_data = (uint64_t) (disk_write_reqs[ns]);
          cbs[ns]->aio_buf = disk_write_reqs[ns]->src_buf;
          cbs[ns]->aio_offset = disk_write_reqs[ns]->dst_offset;
          cbs[ns]->aio_nbytes = disk_write_reqs[ns]->nbytes;
          ns++;
        }
        assert(ns == nr);
        int ret = io_submit(ctx, ns, cbs);
        if (ret < 0) {
          perror("io_submit error");
        }
        return ret;
      }

      void DiskWriteChannel::pull()
      {
        int nr = io_getevents(ctx, 0, capacity, events, NULL);
        if (nr < 0)
          perror("io_getevents error");
        for (int i = 0; i < nr; i++) {
          DiskWriteRequest* req = (DiskWriteRequest*) events[i].data;
          struct iocb* ret_cb = (struct iocb*) events[i].obj;
          available_cb.push_back(ret_cb);
          assert(events[i].res == (int64_t)ret_cb->aio_nbytes);
          req->xd->notify_request_read_done(req);
          req->xd->notify_request_write_done(req);
        }
      }

      long DiskWriteChannel::available()
      {
        return available_cb.size();
      }

      void MemcpyThread::work()
      {
        while (true) {
          pthread_mutex_lock(&submit_lock);
          printf("[MemcpyThread] CP#1\n");
          if (num_pending_reqs == 0)
            pthread_cond_wait(&condvar, &submit_lock);
          printf("[MemcpyThread] Pull from pending queue\n");
          printf("[MemcpyThread] num_pending_reqs = %ld\n", num_pending_reqs);
          assert(pending_queue.size() > 0);
          MemcpyRequest* cur_req = pending_queue.front();
          pending_queue.pop();
          num_pending_reqs --;
          pthread_mutex_unlock(&submit_lock);
          printf("[MemcpyThread] Begin processing copy\n");
          //printf("[MemcpyThread] dst = %ld, src = %ld, nbytes = %lu\n", (off_t) cur_req->dst_buf, (off_t) cur_req->src_buf, cur_req->nbytes);
          memcpy(cur_req->dst_buf, cur_req->src_buf, cur_req->nbytes);
          printf("[MemcpyThread] Finish processing copy\n");
          pthread_mutex_lock(&pull_lock);
          printf("[MemcpyThread] Push into finished queue\n");
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

      void* DMAThread::start(void* arg)
      {
        printf("[DMAThread] start...\n");
        DMAThread* dma = (DMAThread*) arg;
        while (true) {
          if (dma->is_stopped)
            break;
          printf("[DMAThread] CP#1\n");
          pthread_mutex_lock(&dma->channel_lock);
          std::vector<Channel*>::iterator it;
          for (it = dma->channel_queue.begin(); it != dma->channel_queue.end(); it++) {
            printf("[DMAThread] CP#2\n");
            (*it)->pull();
            printf("[DMAThread] CP#3\n");
            long nr = (*it)->available();
            printf("[DMAThread] available = %ld\n", nr);
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
                printf("[DMAThread] nr_got = %ld\n", nr_got);
                long nr_submitted = (*it)->submit(dma->requests, nr_got);
                nr -= nr_submitted;
                printf("[DMAThread] nr_submitted = %ld\n", nr_submitted);
                assert(nr_got == nr_submitted);
              }
            }
            pthread_mutex_unlock(&dma->xferDes_lock);
          }
          pthread_mutex_unlock(&dma->channel_lock);
        }
        return NULL;
      }

      template class MemcpyXferDes<1>;
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
