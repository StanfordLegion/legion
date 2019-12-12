/* Copyright 2019 Stanford University
 * Copyright 2019 Los Alamos National Laboratory
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

// INCLDUED FROM channel.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/transfer/channel.h"

#include "realm/transfer/transfer.h"

TYPE_IS_SERIALIZABLE(Realm::Memory);

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // struct XferDesPortInfo
  //

  template <typename S>
  inline bool serialize(S& s, const XferDesPortInfo& i)
  {
    return ((s << i.port_type) &&
	    (s << i.peer_guid) &&
	    (s << i.peer_port_idx) &&
	    (s << i.indirect_port_idx) &&
	    (s << i.mem) &&
	    (s << i.inst) &&
	    (s << i.ib_offset) &&
	    (s << i.ib_size) &&
	    (s << *i.iter) &&
	    (s << i.serdez_id));
  }

  template <typename S>
  inline bool deserialize(S& s, XferDesPortInfo& i)
  {
    if(!((s >> i.port_type) &&
	 (s >> i.peer_guid) &&
	 (s >> i.peer_port_idx) &&
	 (s >> i.indirect_port_idx) &&
	 (s >> i.mem) &&
	 (s >> i.inst) &&
	 (s >> i.ib_offset) &&
	 (s >> i.ib_size)))
      return false;
    i.iter = TransferIterator::deserialize_new(s);
    if(!i.iter) return false;
    if(!((s >> i.serdez_id)))
      return false;
    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SimpleXferDesFactory<T>
  //

  template <typename T>
  inline SimpleXferDesFactory<T>::SimpleXferDesFactory()
  {}

  template <typename T>
  inline SimpleXferDesFactory<T>::~SimpleXferDesFactory()
  {}

  template <typename T>
  /*static*/ inline SimpleXferDesFactory<T> *SimpleXferDesFactory<T>::get_singleton()
  {
    static SimpleXferDesFactory<T> singleton;
    return &singleton;
  }
  
  template <typename T>
  inline void SimpleXferDesFactory<T>::release()
  {
    // do nothing since we are a singleton
  }

  template <typename T>
  void SimpleXferDesFactory<T>::create_xfer_des(DmaRequest *dma_request,
						NodeID launch_node,
						NodeID target_node,
						XferDesID guid,
						const std::vector<XferDesPortInfo>& inputs_info,
						const std::vector<XferDesPortInfo>& outputs_info,
						bool mark_started,
						uint64_t max_req_size, long max_nr, int priority,
						XferDesFence *complete_fence,
						RegionInstance inst /*= RegionInstance::NO_INST*/)
  {
    if(target_node == Network::my_node_id) {
      // local creation
      assert(!inst.exists());
      XferDes *xd = new T(dma_request, launch_node, guid,
			  inputs_info, outputs_info,
			  mark_started,
			  max_req_size, max_nr, priority,
			  complete_fence);

      get_xdq_singleton()->enqueue_xferDes_local(xd);
    } else {
      // marking the transfer started has to happen locally
      if(mark_started)
	dma_request->mark_started();
      
      // remote creation
      ActiveMessage<XferDesCreateMessage<T> > amsg(target_node, 65536);
      amsg->inst = inst;
      amsg->complete_fence  = complete_fence;
      amsg->launch_node = launch_node;
      amsg->guid = guid;
      amsg->dma_request = dma_request;
      
      bool ok = ((amsg << inputs_info) &&
		 (amsg << outputs_info) &&
		 (amsg << false /*mark_started*/) &&
		 (amsg << max_req_size) &&
		 (amsg << max_nr) &&
		 (amsg << priority));
      assert(ok);
      amsg.commit();
    }
  }
  

  ////////////////////////////////////////////////////////////////////////
  //
  // class XferDesCreateMessage<T>
  //

  template <typename T>
  /*static*/ void XferDesCreateMessage<T>::handle_message(NodeID sender,
							  const XferDesCreateMessage<T> &args,
							  const void *msgdata,
							  size_t msglen)
  {
    std::vector<XferDesPortInfo> inputs_info, outputs_info;
    bool mark_started = false;
    uint64_t max_req_size = 0;
    long max_nr = 0;
    int priority = 0;

    Realm::Serialization::FixedBufferDeserializer fbd(msgdata, msglen);

    bool ok = ((fbd >> inputs_info) &&
	       (fbd >> outputs_info) &&
	       (fbd >> mark_started) &&
	       (fbd >> max_req_size) &&
	       (fbd >> max_nr) &&
	       (fbd >> priority));
    assert(ok);
    assert(fbd.bytes_left() == 0);
  
    assert(!args.inst.exists());
    XferDes *xd = new T(args.dma_request, args.launch_node,
			args.guid,
			inputs_info,
			outputs_info,
			mark_started,
			max_req_size, max_nr, priority,
			args.complete_fence);

    get_xdq_singleton()->enqueue_xferDes_local(xd);
  }

};
