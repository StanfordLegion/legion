/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#include "realm/transfer/addrsplit_channel.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressSplitChannel
  //

  AddressSplitChannel::AddressSplitChannel(BackgroundWorkManager *bgwork)
    : SingleXDQChannel<AddressSplitChannel, AddressSplitXferDesBase>(
          bgwork, XFER_ADDR_SPLIT, "address split")
  {
    assert(!local_addrsplit_channel);
    local_addrsplit_channel = this;
  }

  AddressSplitChannel::~AddressSplitChannel()
  {
    assert(local_addrsplit_channel == this);
    local_addrsplit_channel = 0;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressSplitXferDesBase
  //

  AddressSplitXferDesBase::AddressSplitXferDesBase(
      uintptr_t _dma_op, Channel *_channel, NodeID _launch_node, XferDesID _guid,
      const std::vector<XferDesPortInfo> &inputs_info,
      const std::vector<XferDesPortInfo> &outputs_info, int _priority)
    : XferDes(_dma_op, _channel, _launch_node, _guid, inputs_info, outputs_info,
              _priority, 0, 0)
  {}

  long AddressSplitXferDesBase::get_requests(Request **requests, long nr)
  {
    // unused
    assert(0);
    return 0;
  }

  void AddressSplitXferDesBase::notify_request_read_done(Request *req)
  {
    // unused
    assert(0);
  }

  void AddressSplitXferDesBase::notify_request_write_done(Request *req)
  {
    // unused
    assert(0);
  }

  void AddressSplitXferDesBase::flush()
  {
    // do nothing
  }
}; // namespace Realm
