/* Copyright 2018 Stanford University
 * Copyright 2018 Los Alamos National Laboratory
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

#include "realm/transfer/channel.h"

namespace Realm {

    class FileRequest : public Request {
    public:
      int fd;
      void *mem_base; // could be source or dest
      off_t file_off;
    };
    class DiskRequest : public Request {
    public:
      int fd;
      void *mem_base; // could be source or dest
      off_t disk_off;
    };

    class FileXferDes : public XferDes {
    public:
      FileXferDes(
          DmaRequest* _dma_request, NodeID _launch_node,
          XferDesID _guid, XferDesID _pre_guid, XferDesID _next_guid,
	  uint64_t next_max_rw_gap,
	  size_t src_ib_offset, size_t src_ib_size,
          bool mark_started, RegionInstance inst,
	  Memory _src_mem, Memory _dst_mem,
	  TransferIterator *_src_iter, TransferIterator *_dst_iter,
	  CustomSerdezID _src_serdez_id, CustomSerdezID _dst_serdez_id,
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
      std::string filename;
      int fd; // The file that stores the physical instance
      //const char *buf_base;
    };

    class DiskXferDes : public XferDes {
    public:
      DiskXferDes(DmaRequest* _dma_request, NodeID _launch_node,
                  XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
		  uint64_t next_max_rw_gap,
		  size_t src_ib_offset, size_t src_ib_size,
                  bool mark_started,
		  Memory _src_mem, Memory _dst_mem,
		  TransferIterator *_src_iter, TransferIterator *_dst_iter,
		  CustomSerdezID _src_serdez_id, CustomSerdezID _dst_serdez_id,
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
      //const char *buf_base;
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

}; // namespace Realm

#endif /*LOWLEVEL_CHANNEL_DISK*/
