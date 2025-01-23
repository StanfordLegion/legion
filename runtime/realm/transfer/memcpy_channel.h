/* Copyright 2024 Stanford University
 * Copyright 2024 Los Alamos National Laboratory
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
#ifndef MEMCPY_CHANNEL
#define MEMCPY_CHANNEL

#include <vector>

#include "realm/mem_impl.h"
#include "realm/bgwork.h"
#include "realm/transfer/channel.h"

namespace Realm {

  class MemcpyChannel;

  class MemcpyXferDes : public XferDes {
  public:
    MemcpyXferDes(uintptr_t _dma_op, Channel *_channel, NodeID _launch_node,
                  XferDesID _guid, const std::vector<XferDesPortInfo> &inputs_info,
                  const std::vector<XferDesPortInfo> &outputs_info, int _priority);

    long get_requests(Request **requests, long nr);
    void notify_request_read_done(Request *req);
    void notify_request_write_done(Request *req);
    void flush();

    virtual bool request_available();
    virtual Request *dequeue_request();
    virtual void enqueue_request(Request *req);

    bool progress_xd(MemcpyChannel *channel, TimeLimit work_until);

  private:
    bool memcpy_req_in_use = false;
    MemcpyRequest memcpy_req;
    bool has_serdez = false;
  };

  class MemcpyChannel : public SingleXDQChannel<MemcpyChannel, MemcpyXferDes> {
  public:
    MemcpyChannel(BackgroundWorkManager *_bgwork, const Node *_node,
                  const std::unordered_map<realm_id_t, SharedMemoryInfo>
                      &remote_shared_memory_mappings);

    // multiple concurrent memcpys ok
    static const bool is_ordered = false;

    ~MemcpyChannel();

    virtual uint64_t
    supports_path(ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
                  CustomSerdezID dst_serdez_id, ReductionOpID redop_id,
                  size_t total_bytes, const std::vector<size_t> *src_frags,
                  const std::vector<size_t> *dst_frags, XferDesKind *kind_ret = 0,
                  unsigned *bw_ret = 0, unsigned *lat_ret = 0);

    virtual XferDes *create_xfer_des(uintptr_t dma_op, NodeID launch_node, XferDesID guid,
                                     const std::vector<XferDesPortInfo> &inputs_info,
                                     const std::vector<XferDesPortInfo> &outputs_info,
                                     int priority, XferDesRedopInfo redop_info,
                                     const void *fill_data, size_t fill_size,
                                     size_t fill_total);

    virtual long submit(Request **requests, long nr);

    const Node *node = nullptr;
    bool is_stopped = false;
  };
}; // namespace Realm

#endif
