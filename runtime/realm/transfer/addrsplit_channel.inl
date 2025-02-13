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

#include "realm/serialize.h"
#include "realm/transfer/addrsplit_channel.h"

namespace Realm {

  extern Logger log_dma;
  extern Logger log_ib_alloc;
  extern Logger log_xplan;
  extern Logger log_xpath;
  extern Logger log_xpath_cache;

  AddressSplitChannel *get_local_addrsplit_channel();

  ////////////////////////////////////////////////////////////////////////
  //
  // class XferDesCreateMessage<AddressSplitXferDes<N,T>>
  //

  template <int N, typename T>
  class AddressSplitXferDes;

  template <int N, typename T>
  void AddressSplitCommunicator<N, T>::create(NodeID target_node, NodeID launch_node,
                                              XferDesID guid, uintptr_t dma_op,
                                              const void *msgdata, size_t msglen)
  {
    ActiveMessage<AddressSplitXferDesCreateMessage<N, T>> amsg(target_node, msglen);
    amsg->launch_node = launch_node;
    amsg->guid = guid;
    amsg->dma_op = dma_op;
    amsg.add_payload(msgdata, msglen);
    amsg.commit();
  }

  template <int N, typename T>
  void AddressSplitXferDesCreateMessage<N, T>::handle_message(
      NodeID sender, const AddressSplitXferDesCreateMessage<N, T> &args,
      const void *msgdata, size_t msglen)
  {
    std::vector<XferDesPortInfo> inputs_info, outputs_info;
    int priority = 0;
    size_t element_size = 0;
    std::vector<IndexSpace<N, T>> spaces;

    Realm::Serialization::FixedBufferDeserializer fbd(msgdata, msglen);

    bool ok = ((fbd >> inputs_info) && (fbd >> outputs_info) && (fbd >> priority) &&
               (fbd >> element_size) && (fbd >> spaces));
    assert(ok);
    assert(fbd.bytes_left() == 0);

    auto local_addrsplit_channel = get_local_addrsplit_channel();
    assert(local_addrsplit_channel);

    XferDes *xd = new AddressSplitXferDes<N, T>(
        args.dma_op, local_addrsplit_channel, args.launch_node, args.guid, inputs_info,
        outputs_info, priority, element_size, spaces);

    local_addrsplit_channel->enqueue_ready_xd(xd);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressSplitXferDesFactory<N,T>
  //

  template <int N, typename T>
  AddressSplitXferDesFactory<N, T>::AddressSplitXferDesFactory(
      size_t _bytes_per_element, const std::vector<IndexSpace<N, T>> &_spaces,
      AddressSplitChannel *_addrsplit_channel)
    : bytes_per_element(_bytes_per_element)
    , spaces(_spaces)
    , addrsplit_channel(_addrsplit_channel)
    , comm(std::make_unique<AddressSplitCommunicator<N, T>>())
  {}

  template <int N, typename T>
  AddressSplitXferDesFactory<N, T>::AddressSplitXferDesFactory(
      size_t _bytes_per_element, const std::vector<IndexSpace<N, T>> &_spaces,
      AddressSplitChannel *_addrsplit_channel, AddressSplitCommunicator<N, T> *_comm)
    : bytes_per_element(_bytes_per_element)
    , spaces(_spaces)
    , addrsplit_channel(_addrsplit_channel)
    , comm(_comm)
  {}

  template <int N, typename T>
  bool AddressSplitXferDesFactory<N, T>::needs_release()
  {
    return true;
  }

  template <int N, typename T>
  void AddressSplitXferDesFactory<N, T>::create_xfer_des(
      uintptr_t dma_op, NodeID launch_node, NodeID target_node, XferDesID guid,
      const std::vector<XferDesPortInfo> &inputs_info,
      const std::vector<XferDesPortInfo> &outputs_info, int priority,
      XferDesRedopInfo redop_info, const void *fill_data, size_t fill_size,
      size_t fill_total)
  {
    assert(redop_info.id == 0);
    assert(fill_size == 0);
    if(target_node == Network::my_node_id) {
      // local creation
      // assert(!inst.exists());
      assert(addrsplit_channel != 0);

      XferDes *xd = new AddressSplitXferDes<N, T>(dma_op, addrsplit_channel, launch_node,
                                                  guid, inputs_info, outputs_info,
                                                  priority, bytes_per_element, spaces);

      addrsplit_channel->enqueue_ready_xd(xd);
    } else {
      // remote creation
      Serialization::ByteCountSerializer bcs;
      {
        bool ok = ((bcs << inputs_info) && (bcs << outputs_info) && (bcs << priority) &&
                   (bcs << bytes_per_element) && (bcs << spaces));
        assert(ok);
      }
      size_t req_size = bcs.bytes_used();
      Serialization::DynamicBufferSerializer buffer(req_size);
      {
        bool ok =
            ((buffer << inputs_info) && (buffer << outputs_info) &&
             (buffer << priority) && (buffer << bytes_per_element) && (buffer << spaces));
        assert(ok);
      }
      comm->create(target_node, launch_node, guid, dma_op, buffer.get_buffer(), req_size);
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressSplitXferDes<N,T>
  //

  template <int N, typename T>
  AddressSplitXferDes<N, T>::AddressSplitXferDes(
      uintptr_t _dma_op, Channel *_channel, NodeID _launch_node, XferDesID _guid,
      const std::vector<XferDesPortInfo> &inputs_info,
      const std::vector<XferDesPortInfo> &outputs_info, int _priority,
      size_t _element_size, const std::vector<IndexSpace<N, T>> &_spaces)
    : AddressSplitXferDesBase(_dma_op, _channel, _launch_node, _guid, inputs_info,
                              outputs_info, _priority)
    , spaces(_spaces)
    , element_size(_element_size)
    , point_index(0)
    , point_count(0)
    , output_space_id(-1)
    , output_count(0)
  {
    ctrl_encoder.set_port_count(spaces.size());
  }

  template <int N, typename T>
  int AddressSplitXferDes<N, T>::find_point_in_spaces(Point<N, T> p, int guess_idx) const
  {
    // try the guessed (e.g. same as previous hit) space first
    if(guess_idx >= 0) {
      if(spaces[guess_idx].contains(p)) {
        return guess_idx;
      }
    }

    // try all the rest
    for(size_t i = 0; i < spaces.size(); i++) {
      if(i != size_t(guess_idx)) {
        if(spaces[i].contains(p)) {
          return i;
        }
      }
    }

    return -1;
  }

  template <int N, typename T>
  Event AddressSplitXferDes<N, T>::request_metadata()
  {
    std::vector<Event> events;

    for(size_t i = 0; i < spaces.size(); i++) {
      Event e = spaces[i].make_valid();
      if(e.exists()) {
        events.push_back(e);
      }
    }

    return Event::merge_events(events);
  }

  template <int N, typename T>
  bool AddressSplitXferDes<N, T>::progress_xd(AddressSplitChannel *channel,
                                              TimeLimit work_until)
  {
    assert(!iteration_completed.load());

    ReadSequenceCache rseqcache(this);
    WriteSequenceCache wseqcache(this);

    bool did_work = false;
    while(true) {
      size_t output_bytes = 0;
      bool input_done = false;
      while(true) {
        // step 1: get some points if we are out
        if(point_index >= point_count) {
          if(input_ports[0].iter->done()) {
            input_done = true;
            break;
          }

          TransferIterator::AddressInfo p_info;
          size_t max_bytes = MAX_POINTS * sizeof(Point<N, T>);
          if(input_ports[0].peer_guid != XFERDES_NO_GUID) {
            max_bytes = input_ports[0].seq_remote.span_exists(
                input_ports[0].local_bytes_total, max_bytes);
            // round down to multiple of sizeof(Point<N,T>)
            size_t rem = max_bytes % sizeof(Point<N, T>);
            if(rem > 0) {
              max_bytes -= rem;
            }
            if(max_bytes < sizeof(Point<N, T>)) {
              // check to see if this is the end of the input
              if(input_ports[0].local_bytes_total ==
                 input_ports[0].remote_bytes_total.load_acquire()) {
                input_done = true;
              }
              break;
            }
          }
          size_t bytes =
              input_ports[0].iter->step(max_bytes, p_info, 0, false /*!tentative*/);
          if(bytes == 0) {
            break;
          }
          const void *srcptr =
              input_ports[0].mem->get_direct_ptr(p_info.base_offset, bytes);
          assert(srcptr != 0);
          memcpy(points, srcptr, bytes);
          // handle reads of partial points
          while((bytes % sizeof(Point<N, T>)) != 0) {
            // get some more - should never be empty
            size_t todo = input_ports[0].iter->step(max_bytes - bytes, p_info, 0,
                                                    false /*!tentative*/);
            assert(todo > 0);
            const void *srcptr =
                input_ports[0].mem->get_direct_ptr(p_info.base_offset, todo);
            assert(srcptr != 0);
            memcpy(reinterpret_cast<char *>(points) + bytes, srcptr, todo);
            bytes += todo;
          }

          point_count = bytes / sizeof(Point<N, T>);
          assert(bytes == (point_count * sizeof(Point<N, T>)));
          point_index = 0;
          rseqcache.add_span(0, input_ports[0].local_bytes_total, bytes);
          input_ports[0].local_bytes_total += bytes;
          did_work = true;
        }

        // step 2: process the first point we've got on hand
        int new_space_id = find_point_in_spaces(points[point_index], output_space_id);

        // can only extend an existing run with another point from the same
        //  space
        if(output_count == 0) {
          output_space_id = new_space_id;
        } else if(new_space_id != output_space_id) {
          break;
        }

        // if it matched a space, we have to emit the point to that space's
        //  output address stream before we can accept the point
        if(output_space_id != -1) {
          XferPort &op = output_ports[output_space_id];
          if(op.seq_remote.span_exists(op.local_bytes_total + output_bytes,
                                       sizeof(Point<N, T>)) < sizeof(Point<N, T>)) {
            break;
          }
          TransferIterator::AddressInfo o_info;
          size_t partial = 0;
          while(partial < sizeof(Point<N, T>)) {
            size_t bytes = op.iter->step(sizeof(Point<N, T>) - partial, o_info, 0,
                                         false /*!tentative*/);
            void *dstptr = op.mem->get_direct_ptr(o_info.base_offset, bytes);
            assert(dstptr != 0);
            memcpy(dstptr, reinterpret_cast<const char *>(&points[point_index]) + partial,
                   bytes);
            partial += bytes;
          }
          output_bytes += sizeof(Point<N, T>);
        }
        output_count++;
        point_index++;
      }

      // if we wrote any points out, update their validity now
      if(output_bytes > 0) {
        assert(output_space_id >= 0);
        wseqcache.add_span(output_space_id,
                           output_ports[output_space_id].local_bytes_total, output_bytes);
        output_ports[output_space_id].local_bytes_total += output_bytes;
        output_ports[output_space_id].local_bytes_cons.fetch_add(output_bytes);
        did_work = true;
      }

      // now try to write the control information
      if((output_count > 0) || input_done) {
        XferPort &cp = output_ports[spaces.size()];

        // may take us a few tries to send the control word
        bool ctrl_sent = false;
        size_t old_lbt = cp.local_bytes_total;
        do {
          if(cp.seq_remote.span_exists(cp.local_bytes_total, sizeof(unsigned)) <
             sizeof(unsigned)) {
            break; // no room to write control work
          }

          TransferIterator::AddressInfo c_info;
          size_t bytes = cp.iter->step(sizeof(unsigned), c_info, 0, false /*!tentative*/);
          assert(bytes == sizeof(unsigned));
          void *dstptr = cp.mem->get_direct_ptr(c_info.base_offset, sizeof(unsigned));
          assert(dstptr != 0);

          unsigned cword;
          ctrl_sent = ctrl_encoder.encode(cword, output_count * element_size,
                                          output_space_id, input_done);
          memcpy(dstptr, &cword, sizeof(unsigned));

          cp.local_bytes_total += sizeof(unsigned);
          cp.local_bytes_cons.fetch_add(sizeof(unsigned));
        } while(!ctrl_sent);

        if(input_done && ctrl_sent) {
          begin_completion();

          // mark all address streams as done (dummy write update)
          for(size_t i = 0; i < spaces.size(); i++) {
            wseqcache.add_span(i, output_ports[i].local_bytes_total, 0);
          }
        }

        // push out the partial write even if we're not done
        if(cp.local_bytes_total > old_lbt) {
          wseqcache.add_span(spaces.size(), old_lbt, cp.local_bytes_total - old_lbt);
          did_work = true;
        }

        // but only actually clear the output_count if we sent the whole
        //  control packet
        if(!ctrl_sent) {
          break;
        }

        output_space_id = -1;
        output_count = 0;
      } else {
        break;
      }

      if(iteration_completed.load() || work_until.is_expired()) {
        break;
      }
    }

    rseqcache.flush();
    wseqcache.flush();

    return did_work;
  }
}; // namespace Realm
