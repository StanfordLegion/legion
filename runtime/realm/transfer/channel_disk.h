/* Copyright 2022 Stanford University
 * Copyright 2022 Los Alamos National Laboratory
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

    class FileChannel;

    class FileXferDes : public XferDes {
    public:
      FileXferDes(uintptr_t _dma_op, Channel *_channel,
		  NodeID _launch_node, XferDesID _guid,
		  const std::vector<XferDesPortInfo>& inputs_info,
		  const std::vector<XferDesPortInfo>& outputs_info,
		  int _priority);

      ~FileXferDes()
      {
        free(file_reqs);
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

      bool progress_xd(FileChannel *channel, TimeLimit work_until);

    private:
      FileRequest* file_reqs;
      FileMemory::OpenFileInfo *file_info;
    };

    class DiskChannel;

    class DiskXferDes : public XferDes {
    public:
      DiskXferDes(uintptr_t _dma_op, Channel *_channel,
		  NodeID _launch_node, XferDesID _guid,
		  const std::vector<XferDesPortInfo>& inputs_info,
		  const std::vector<XferDesPortInfo>& outputs_info,
		  int _priority);

      ~DiskXferDes() {
        free(disk_reqs);
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

      bool progress_xd(DiskChannel *channel, TimeLimit work_until);

    private:
      int fd;
      DiskRequest* disk_reqs;
      //const char *buf_base;
    };

    class FileChannel : public SingleXDQChannel<FileChannel, FileXferDes> {
    public:
      FileChannel(BackgroundWorkManager *bgwork);
      ~FileChannel();

      // TODO: farm I/O work off to dedicated threads if needed
      static const bool is_ordered = true;

      virtual XferDes *create_xfer_des(uintptr_t dma_op,
				       NodeID launch_node,
				       XferDesID guid,
				       const std::vector<XferDesPortInfo>& inputs_info,
				       const std::vector<XferDesPortInfo>& outputs_info,
				       int priority,
				       XferDesRedopInfo redop_info,
				       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request** requests, long nr);
    };

    class DiskChannel : public SingleXDQChannel<DiskChannel, DiskXferDes> {
    public:
      DiskChannel(BackgroundWorkManager *bgwork);
      ~DiskChannel();

      // TODO: farm I/O work off to dedicated threads if needed
      static const bool is_ordered = true;

      virtual XferDes *create_xfer_des(uintptr_t dma_op,
				       NodeID launch_node,
				       XferDesID guid,
				       const std::vector<XferDesPortInfo>& inputs_info,
				       const std::vector<XferDesPortInfo>& outputs_info,
				       int priority,
				       XferDesRedopInfo redop_info,
				       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request** requests, long nr);
    };

}; // namespace Realm

#endif /*LOWLEVEL_CHANNEL_DISK*/
