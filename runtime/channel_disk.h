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
#ifndef LOWLEVEL_CHANNEL_DISK
#define LOWLEVEL_CHANNEL_DISK

#include "channel.h"

namespace LegionRuntime {
  namespace LowLevel {
    class FileReadRequest : public Request {
    public:
      int fd;
      uint64_t dst_buf;
      int64_t src_offset;
      uint64_t nbytes;
    };

    class FileWriteRequest : public Request {
    public:
      int fd;
    };

    template<unsigned DIM>
    class FileXferDes : public XferDes {
    public:
      FileXferDes(
          DmaRequest* _dma_request, gasnet_node_t _launch_node,
          XferDesID _guid, XferDesID _pre_guid, XferDesID _next_guid,
          bool mark_started, const Buffer& _src_buf, const Buffer& _dst_buf,
          const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
          uint64_t max_req_size, long max_nr, int _priority,
          XferOrder::Type _order, XferDesFence* _complete_fence);

      ~FileXferDes()
     {
        // clear available_reqs
        while (!available_reqs.empty()) {
          available_reqs.pop();
        }
        if (DIM == 0) {
          delete me;
        } else {
          delete li;
        }
        free(requests);
        if (src_buf.is_ib) {
          get_runtime()->get_memory_impl(src_buf.memory)->free_bytes(src_buf.alloc_offset, src_buf.buf_size);
        }
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();
    private:
      Request* requests;
      Layouts::GenericLayoutIterator<DIM>* li;
      MaskEnumerator* me;
      unsigned offset_idx;
      int fd; // The file that stores the physical instance
    };

    class FileChannel : public Channel {
    public:
      FileChannel(long max_nr, XferDes::XferKind _kind);
      ~FileChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    private:
      aio_context_t ctx;
      long capacity;
      std::vector<struct iocb*> available_cb;
      struct iocb* cb;
      struct iocb** cbs;
      struct io_event* events;
    };

  } // namespace LowLevel
} // namespace LegionRuntime
#endif /*LOWLEVEL_CHANNEL_DISK*/
