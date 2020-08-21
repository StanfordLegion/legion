/* Copyright 2020 Stanford University
 * Copyright 2020 Los Alamos National Laboratory
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

      xd->channel->enqueue_ready_xd(xd);
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

      // normally ownership of input and output iterators would be taken
      //  by the local XferDes we create, but here we sent a copy, so delete
      //  the originals
      for(std::vector<XferDesPortInfo>::const_iterator it = inputs_info.begin();
	  it != inputs_info.end();
	  ++it)
	delete it->iter;

      for(std::vector<XferDesPortInfo>::const_iterator it = outputs_info.begin();
	  it != outputs_info.end();
	  ++it)
	delete it->iter;
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

    xd->channel->enqueue_ready_xd(xd);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class XferDes
  //

  // transfer descriptors are reference counted rather than explcitly
  //  deleted
  inline void XferDes::add_reference(void)
  {
    reference_count.fetch_add(1);
  }

  inline void XferDes::remove_reference(void)
  {
    unsigned prev = reference_count.fetch_sub_acqrel(1);
    if(prev == 1)
      delete this;
  }

  inline unsigned XferDes::current_progress(void)
  {
    unsigned val = progress_counter.load();
    return val;
  }

  // checks to see if progress has been made since the last read of the
  //  progress counter - atomically marks the xd for wakeup if not
  inline bool XferDes::check_for_progress(unsigned last_counter)
  {
    // attempt a compare and swap to set the LSB of the counter - if
    //  it fails, we know the counter has been bumped
    return !progress_counter.compare_exchange(last_counter,
					      last_counter + 1);
  }

  // updates the progress counter, waking up the xd if needed
  inline void XferDes::update_progress(void)
  {
    // no point in doing an update on an xd that's known to be done
    if(transfer_completed.load()) return;

    // add 2 to the counter (i.e. preserving the LSB) - if LSB was/is set,
    //  attempt to add 1 to clear it and if successful, wake up the xd
    unsigned prev = progress_counter.fetch_add(2);
    if((prev & 1) != 0) {
      prev += 2;  // since we added 2 above
      if(progress_counter.compare_exchange(prev, prev + 1)) {
	channel->wakeup_xd(this);
	return;
      }
      // on failure, do not retry - somebody else will have the right
      //  prev value
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class XDQueue<CHANNEL, XD>
  //

  template <typename CHANNEL, typename XD>
  XDQueue<CHANNEL,XD>::XDQueue(CHANNEL *_channel,
			       const std::string& _name,
			       bool _ordered)
    : BackgroundWorkItem(_name)
    , channel(_channel)
    , ordered_mode(_ordered)
    , in_ordered_worker(false)
  {}

  template <typename CHANNEL, typename XD>
  void XDQueue<CHANNEL,XD>::enqueue_xd(XD *xd,
				       bool at_front /*= false*/)
  {
    bool was_empty;
    {
      AutoLock<> a(mutex);

      // if we were empty, we'll need to request attention from a worker
      // exception: if a worker is currently running in ordered mode,
      //  we're not completed empty
      was_empty = !in_ordered_worker && ready_xds.empty();

      if(at_front)
	ready_xds.push_front(xd);
      else
	ready_xds.push_back(xd);
    }
    if(was_empty)
      make_active();
  }

  template <typename CHANNEL, typename XD>
  void XDQueue<CHANNEL,XD>::do_work(TimeLimit work_until)
  {
    bool first_iteration = true;
    while(true) {
      // take a request off the front of the queue and unless we're
      //  ordered, re-mark ourselves active if more remain (so other
      //  workers can help)
      XD *xd = 0;
      bool still_more = false;
      {
	AutoLock<> a(mutex);
	if(!ready_xds.empty()) {
	  xd = checked_cast<XD *>(ready_xds.pop_front());
	  still_more = !ready_xds.empty();
	  in_ordered_worker = ordered_mode;
	} else
	  in_ordered_worker = false;
      }
      // should aways get an xd on the first iteration
      if(first_iteration) {
        assert(xd != 0);
	first_iteration = false;
      }
      if(!xd) return;  // nothing to do
      if(still_more && !ordered_mode)
	make_active();

      // now process this transfer request, paying attention to our deadline

      // if this is the first time we've touched this xd, mark it started
      if(xd->mark_start) {
	xd->dma_request->mark_started();
	xd->mark_start = false;
      }

      while(true) {
	if(!xd->is_completed()) {
	  // take a snapshot of the xd's progress counter before we work
	  //  on it
	  unsigned progress = xd->current_progress();

	  bool did_work = xd->progress_xd(channel, work_until);

	  // if we didn't do any work, and we're not done (i.e. by
	  //  concluding there wasn't any work to actually do), re-check
	  //  the progress counter and sleep the xd if needed
	  if(!did_work && !xd->is_completed()) {
	    if(!xd->check_for_progress(progress)) {
	      // just drop the xd here (i.e. do not re-enqueue) - somebody
	      //  else will (or already has) when new data comes in
	      log_new_dma.info() << "xd sleeping: xd=" << xd;
	      break;
	    }
	  }
	}

	if(xd->is_completed()) {
	  xd->flush();
	  log_new_dma.info("Finish XferDes : id(" IDFMT ")", xd->guid);
	  xd->mark_completed();
	  break;
	}

	// stop mid-transfer if we've run out of time
	if(work_until.is_expired()) {
	  //log_new_dma.print() << "time limit elapsed";
	  // requeue, and request attention if we were empty or in
	  //  ordered mode
	  bool was_empty;
	  {
	    AutoLock<> al(mutex);
	    was_empty = in_ordered_worker || ready_xds.empty();
	    ready_xds.push_front(xd);
	    in_ordered_worker = false;
	  }
	  if(was_empty)
	    make_active();
	  return;
	}
      }

      // if we're not in ordered mode, we're not allowed to take another
      //   ready xd
      if(!ordered_mode)
	break;
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SingleXDQChannel<CHANNEL, XD>
  //

  extern XferDesQueue *xferDes_queue;

  template <typename CHANNEL, typename XD>
  SingleXDQChannel<CHANNEL,XD>::SingleXDQChannel(BackgroundWorkManager *bgwork,
						 XferDesKind _kind,
						 const std::string &_name)
    : Channel(_kind)
    , xdq(static_cast<CHANNEL *>(this), _name, CHANNEL::is_ordered)
  {
    xdq.add_to_manager(bgwork);
  }

  template <typename CHANNEL, typename XD>
  void SingleXDQChannel<CHANNEL,XD>::shutdown()
  {
#ifdef DEBUG_REALM
    xdq.shutdown_work_item();
#endif
  }

  template <typename CHANNEL, typename XD>
  void SingleXDQChannel<CHANNEL,XD>::enqueue_ready_xd(XferDes *xd)
  {
    // xferDes_queue needs to know about this for guid->xd translation
    if(xferDes_queue->enqueue_xferDes_local(xd, false /*!add_to_queue*/))
      xdq.enqueue_xd(checked_cast<XD *>(xd));
  }

  template <typename CHANNEL, typename XD>
  void SingleXDQChannel<CHANNEL,XD>::wakeup_xd(XferDes *xd)
  {
    // add this back to the xdq at the front of the list
    log_new_dma.info() << "xd woken: xd=" << xd;
    xdq.enqueue_xd(checked_cast<XD *>(xd), true);
  }


};
