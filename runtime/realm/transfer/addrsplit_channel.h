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

#ifndef ADDRESS_SPLIT_CHANNEL_H
#define ADDRESS_SPLIT_CHANNEL_H

#include "realm/indexspace.h"
#include "realm/transfer/channel.h"

namespace Realm {

  class XferDes;
  class AddressList;

  class AddressSplitChannel;

  class AddressSplitXferDesBase : public XferDes {
  protected:
    AddressSplitXferDesBase(uintptr_t dma_op, Channel *_channel, NodeID _launch_node,
                            XferDesID _guid,
                            const std::vector<XferDesPortInfo> &inputs_info,
                            const std::vector<XferDesPortInfo> &outputs_info,
                            int _priority);

  public:
    virtual bool progress_xd(AddressSplitChannel *channel, TimeLimit work_until) = 0;

    long get_requests(Request **requests, long nr);
    void notify_request_read_done(Request *req);
    void notify_request_write_done(Request *req);
    void flush();
  };

  class AddressSplitChannel
    : public SingleXDQChannel<AddressSplitChannel, AddressSplitXferDesBase> {
  public:
    AddressSplitChannel(BackgroundWorkManager *bgwork);
    virtual ~AddressSplitChannel() = default;

    // do as many of these concurrently as we like
    static constexpr bool is_ordered = false;

    virtual void enqueue_ready_xd(XferDes *xd)
    {
      SingleXDQChannel<AddressSplitChannel, AddressSplitXferDesBase>::enqueue_ready_xd(
          xd);
    }

    // override this to make sure it's never called
    virtual XferDes *create_xfer_des(uintptr_t dma_op, NodeID launch_node, XferDesID guid,
                                     const std::vector<XferDesPortInfo> &inputs_info,
                                     const std::vector<XferDesPortInfo> &outputs_info,
                                     int priority, XferDesRedopInfo redop_info,
                                     const void *fill_data, size_t fill_size,
                                     size_t fill_total);

    virtual long submit(Request **requests, long nr);
  };

  template <int N, typename T>
  struct AddressSplitXferDesCreateMessage : public XferDesCreateMessageBase {
  public:
    static void handle_message(NodeID sender,
                               const AddressSplitXferDesCreateMessage<N, T> &args,
                               const void *msgdata, size_t msglen);
  };

  template <int N, typename T>
  class AddressSplitCommunicator {
  public:
    virtual ~AddressSplitCommunicator() = default;
    virtual void create(NodeID target_node, NodeID launch_node, XferDesID guid,
                        uintptr_t dma_op, const void *msgdata, size_t msglen);
  };

  template <int N, typename T>
  class AddressSplitXferDesFactory : public XferDesFactory {
  public:
    AddressSplitXferDesFactory(size_t _bytes_per_element,
                               const std::vector<IndexSpace<N, T>> &_spaces,
                               AddressSplitChannel *_addrsplit_channel);

    AddressSplitXferDesFactory(size_t _bytes_per_element,
                               const std::vector<IndexSpace<N, T>> &_spaces,
                               AddressSplitChannel *_addrsplit_channel,
                               AddressSplitCommunicator<N, T> *_comm);

    virtual ~AddressSplitXferDesFactory() = default;

  public:
    virtual bool needs_release();

    virtual void create_xfer_des(uintptr_t dma_op, NodeID launch_node, NodeID target_node,
                                 XferDesID guid,
                                 const std::vector<XferDesPortInfo> &inputs_info,
                                 const std::vector<XferDesPortInfo> &outputs_info,
                                 int priority, XferDesRedopInfo redop_info,
                                 const void *fill_data, size_t fill_size,
                                 size_t fill_total);

    static inline ActiveMessageHandlerReg<AddressSplitXferDesCreateMessage<N, T>> areg;

  protected:
    size_t bytes_per_element{0};
    std::vector<IndexSpace<N, T>> spaces;
    AddressSplitChannel *addrsplit_channel{0};
    std::unique_ptr<AddressSplitCommunicator<N, T>> comm{nullptr};
  };

  template <int N, typename T>
  class AddressSplitXferDes : public AddressSplitXferDesBase {
  public:
    AddressSplitXferDes(uintptr_t _dma_op, Channel *_channel, NodeID _launch_node,
                        XferDesID _guid, const std::vector<XferDesPortInfo> &inputs_info,
                        const std::vector<XferDesPortInfo> &outputs_info, int _priority,
                        size_t _element_size,
                        const std::vector<IndexSpace<N, T>> &_spaces);

    ~AddressSplitXferDes() = default;

    virtual Event request_metadata();

    virtual bool progress_xd(AddressSplitChannel *channel, TimeLimit work_until);

  protected:
    int find_point_in_spaces(Point<N, T> p, int guess_idx) const;

    std::vector<IndexSpace<N, T>> spaces;
    size_t element_size;
    static constexpr size_t MAX_POINTS = 64;
    size_t point_index, point_count;
    Point<N, T> points[MAX_POINTS];
    int output_space_id;
    unsigned output_count;
    ControlPort::Encoder ctrl_encoder;
  };

}; // namespace Realm

#include "realm/transfer/addrsplit_channel.inl"

#endif // ifndef ADDRESS_SPLIT_CHANNEL_H
