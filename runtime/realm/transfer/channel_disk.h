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
    class FileRequest : public Request {
    public:
      int fd;
      char *mem_base;
      off_t file_off;
    };
    class DiskRequest : public Request {
    public:
      int fd;
      char *mem_base;
      off_t disk_off;
    };

    template<unsigned DIM>
    class FileXferDes : public XferDes {
    public:
      FileXferDes(
          DmaRequest* _dma_request, gasnet_node_t _launch_node,
          XferDesID _guid, XferDesID _pre_guid, XferDesID _next_guid,
          bool mark_started, RegionInstance inst,
          const Buffer& _src_buf, const Buffer& _dst_buf,
          const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
          uint64_t max_req_size, long max_nr, int _priority,
          XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence);

      ~FileXferDes()
      {
        free(file_reqs);
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();
    private:
      FileRequest* file_reqs;
      int fd; // The file that stores the physical instance
      const char *buf_base;
    };

    template<unsigned DIM>
    class DiskXferDes : public XferDes {
    public:
      DiskXferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                  XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                  bool mark_started, const Buffer& _src_buf, const Buffer& _dst_buf,
                  const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                  uint64_t _max_req_size, long max_nr, int _priority,
                  XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence);

      ~DiskXferDes() {
        free(disk_reqs);
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

    private:
      int fd;
      DiskRequest* disk_reqs;
      const char *buf_base;
    };

    class FileChannel : public Channel {
    public:
      FileChannel(long max_nr, XferDes::XferKind _kind);
      ~FileChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    };

    class DiskChannel : public Channel {
    public:
      DiskChannel(long max_nr, XferDes::XferKind _kind);
      ~DiskChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    };

  } // namespace LowLevel
} // namespace LegionRuntime
#endif /*LOWLEVEL_CHANNEL_DISK*/
