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

// INCLDUED FROM channel.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/transfer/channel.h"

#include "realm/transfer/transfer.h"

TYPE_IS_SERIALIZABLE(Realm::Memory);
TYPE_IS_SERIALIZABLE(Realm::XferDesKind);
TYPE_IS_SERIALIZABLE(Realm::XferDesRedopInfo);
TYPE_IS_SERIALIZABLE(Realm::Channel::SupportedPath);

namespace Realm {

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
    unsigned val = progress_counter.load_acquire();
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
    // add 2 to the counter (i.e. preserving the LSB) - if LSB was/is set,
    //  attempt to add 1 to clear it and if successful, wake up the xd
    unsigned prev = progress_counter.fetch_add_acqrel(2);
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
  XDQueue<CHANNEL,XD>::XDQueue(LocalChannel *_channel,
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
  bool XDQueue<CHANNEL,XD>::do_work(TimeLimit work_until)
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
      if(!xd) return false;  // nothing to do
      if(still_more && !ordered_mode)
	make_active();

      // now process this transfer request, paying attention to our deadline

      while(true) {
	if(!xd->transfer_completed.load_acquire()) {
	  // take a snapshot of the xd's progress counter before we work
	  //  on it
	  unsigned progress = xd->current_progress();

	  bool did_work = xd->progress_xd(static_cast<CHANNEL *>(channel),
                                          work_until);

	  // if we didn't do any work, and we're not done (i.e. by
	  //  concluding there wasn't any work to actually do), re-check
	  //  the progress counter and sleep the xd if needed
	  if(!did_work && !xd->transfer_completed.load_acquire()) {
	    if(!xd->check_for_progress(progress)) {
	      // just drop the xd here (i.e. do not re-enqueue) - somebody
	      //  else will (or already has) when new data comes in
	      log_new_dma.info() << "xd sleeping: xd=" << xd
				 << " id=" << std::hex << xd->guid << std::dec;
	      break;
	    }
	  }
	}

	if(xd->transfer_completed.load_acquire()) {
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

          // request bgwork requeuing if we made ourselves nonempty
          return was_empty;
	}
      }

      // if we're not in ordered mode, we're not allowed to take another
      //   ready xd
      if(!ordered_mode)
	break;
    }

    // no work left (and/or somebody else is tracking it)
    return false;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Channel
  //

  inline std::ostream& operator<<(std::ostream& os, const Channel& c)
  {
    c.print(os);
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteChannelInfo
  //

  template <typename S>
  inline bool serialize(S& serializer, const RemoteChannelInfo& rci)
  {
    return Serialization::PolymorphicSerdezHelper<RemoteChannelInfo>::serialize(serializer, rci);
  }

  template <typename S>
  /*static*/ RemoteChannelInfo *RemoteChannelInfo::deserialize_new(S& deserializer)
  {
    return Serialization::PolymorphicSerdezHelper<RemoteChannelInfo>::deserialize_new(deserializer);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteChannelInfo
  //

  template <typename S>
  bool SimpleRemoteChannelInfo::serialize(S& serializer) const
  {
    return ((serializer << owner) &&
            (serializer << kind) &&
            (serializer << remote_ptr) &&
            (serializer << paths));
  }

  template <typename S>
  /*static*/ RemoteChannelInfo *SimpleRemoteChannelInfo::deserialize_new(S& deserializer)
  {
    NodeID owner;
    XferDesKind kind;
    uintptr_t remote_ptr;
    std::vector<Channel::SupportedPath> paths;

    if((deserializer >> owner) &&
       (deserializer >> kind) &&
       (deserializer >> remote_ptr) &&
       (deserializer >> paths)) {
      return new SimpleRemoteChannelInfo(owner, kind, remote_ptr, paths);
    } else {
      return 0;
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SingleXDQChannel<CHANNEL, XD>
  //

  template <typename CHANNEL, typename XD>
  SingleXDQChannel<CHANNEL,XD>::SingleXDQChannel(BackgroundWorkManager *bgwork,
						 XferDesKind _kind,
						 const std::string &_name,
                                                 int _numa_domain /*= -1*/)
    : LocalChannel(_kind)
    , xdq(this, _name, CHANNEL::is_ordered)
  {
    xdq.add_to_manager(bgwork, _numa_domain);
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
    if(xd->xferDes_queue->enqueue_xferDes_local(xd,
						false /*!add_to_queue*/))
      xdq.enqueue_xd(checked_cast<XD *>(xd));
  }

  template <typename CHANNEL, typename XD>
  void SingleXDQChannel<CHANNEL,XD>::wakeup_xd(XferDes *xd)
  {
    // add this back to the xdq at the front of the list
    log_new_dma.info() << "xd woken: xd=" << xd
		       << " id=" << std::hex << xd->guid << std::dec;;
    xdq.enqueue_xd(checked_cast<XD *>(xd), true);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class XferDes::SequenceCache<UPDATE>
  //

  template <void (XferDes::*UPDATE)(int port_idx, size_t offset, size_t size)>
  XferDes::SequenceCache<UPDATE>::SequenceCache(XferDes *_xd,
						size_t _flush_bytes /*= 0*/)
    : xd(_xd)
    , total_bytes(0)
    , flush_bytes(_flush_bytes)
  {
    for(size_t i = 0; i < MAX_ENTRIES; i++)
      ports[i] = -1;
  }

  template <void (XferDes::*UPDATE)(int port_idx, size_t offset, size_t size)>
  void XferDes::SequenceCache<UPDATE>::add_span(int port_idx, size_t offset, size_t size)
  {
    // invalid ports are ignored (don't filter empty spans - used for flushes)
    if(port_idx < 0) return;

    // see if we've seen this port index before
    for(size_t i = 0; i < MAX_ENTRIES; i++)
      if(ports[i] == port_idx) {
	// ok, now see if we can append to the existing span
	if(offset == (offsets[i] + sizes[i])) {
	  // we can add to the existing span
	  sizes[i] += size;
	} else {
	  // not contiguous - push old span out and start a new one
	  (xd->*UPDATE)(port_idx, offsets[i], sizes[i]);
	  total_bytes -= sizes[i];
	  offsets[i] = offset;
	  sizes[i] = size;
	}
	total_bytes += size;
	if((flush_bytes > 0) && (total_bytes >= flush_bytes))
	  flush();
	return;
      }

    // no match - replace the largest existing span or use an available slot
    int biggest_idx = 0;
    size_t biggest_size = sizes[0];
    for(size_t i = 0; i < MAX_ENTRIES; i++) {
      if(ports[i] < 0) {
	ports[i] = port_idx;
	offsets[i] = offset;
	sizes[i] = size;
	total_bytes += size;
	if((flush_bytes > 0) && (total_bytes >= flush_bytes))
	  flush();
	return;
      }

      if(sizes[i] > biggest_size) {
	biggest_idx = i;
	biggest_size = sizes[i];
      }
    }

    if(size > biggest_size) {
      // new span is the biggest, so just send it through
      (xd->*UPDATE)(port_idx, offset, size);
    } else {
      // kick out an old entry and use its slot
      (xd->*UPDATE)(ports[biggest_idx], offsets[biggest_idx], biggest_size);
      total_bytes -= biggest_size;

      ports[biggest_idx] = port_idx;
      offsets[biggest_idx] = offset;
      sizes[biggest_idx] = size;
      total_bytes += size;
      if((flush_bytes > 0) && (total_bytes >= flush_bytes))
	flush();
    }	
  }

  template <void (XferDes::*UPDATE)(int port_idx, size_t offset, size_t size)>
  void XferDes::SequenceCache<UPDATE>::flush()
  {
    for(size_t i = 0; i < MAX_ENTRIES; i++)
      if(ports[i] >= 0) {
	(xd->*UPDATE)(ports[i], offsets[i], sizes[i]);
	ports[i] = -1;
      }
    total_bytes = 0;
  }


};
