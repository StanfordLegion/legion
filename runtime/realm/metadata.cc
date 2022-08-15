/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// Realm object metadata base implementation

#include "realm/metadata.h"

#include "realm/event_impl.h"
#include "realm/inst_impl.h"
#include "realm/runtime_impl.h"

namespace Realm {

  Logger log_metadata("metadata");

  ////////////////////////////////////////////////////////////////////////
  //
  // class MetadataBase
  //

    MetadataBase::MetadataBase(void)
      : state(STATE_INVALID), valid_event(Event::NO_EVENT)
      , frag_buffer(0), frag_bytes_received(0)
    {}

    MetadataBase::~MetadataBase(void)
    {}

    void MetadataBase::mark_valid(NodeSet& early_reqs)
    {
      // take lock so we can make sure we get a precise list of early requestors
      // if there was an event (i.e. an impatient local reqiest), trigger it too
      Event to_trigger = Event::NO_EVENT;
      {
	AutoLock<> a(mutex);
	early_reqs = remote_copies;
	state = STATE_VALID;
	to_trigger = valid_event;
	valid_event = Event::NO_EVENT;
      }

      if(to_trigger.exists())
	GenEventImpl::trigger(to_trigger, false /*!poisoned*/);
    }

    bool MetadataBase::handle_request(int requestor)
    {
      // just add the requestor to the list of remote nodes with copies, can send
      //   response if the data is already valid
      AutoLock<> a(mutex);

      assert(!remote_copies.contains(requestor));
      remote_copies.add(requestor);

      return is_valid();
    }

    void MetadataBase::handle_response(void)
    {
      // update the state, and
      // if there was an event, we'll trigger it
      Event to_trigger = Event::NO_EVENT;
      {
	AutoLock<> a(mutex);

	switch(state) {
	case STATE_REQUESTED:
	  {
	    to_trigger = valid_event;
	    valid_event = Event::NO_EVENT;
	    state = STATE_VALID;
	    break;
	  }

	default:
	  assert(0);
	}
      }

      if(to_trigger.exists())
	GenEventImpl::trigger(to_trigger, false /*!poisoned*/);
    }

    Event MetadataBase::request_data(int owner, ID::IDType id)
    {
      // early out - valid data need not be re-requested
      if(state == STATE_VALID) 
	return Event::NO_EVENT;

      Event e = Event::NO_EVENT;
      bool issue_request = false;
      {
	AutoLock<> a(mutex);

	switch(state) {
	case STATE_VALID:
	  {
	    // possible if the data came in between our early out check
	    // above and our taking of the lock - nothing more to do
	    break;
	  }

	case STATE_INVALID: 
	  {
	    // if the current state is invalid, we'll need to issue a request
	    state = STATE_REQUESTED;
	    valid_event = GenEventImpl::create_genevent()->current_event();
            e = valid_event;
	    // actually, no need to issue a request if we're the owner
	    issue_request = (owner != Network::my_node_id);
	    break;
	  }

	case STATE_REQUESTED:
	  {
	    // request has already been issued, but return the event again
	    assert(valid_event.exists());
            e = valid_event;
	    break;
	  }

	case STATE_INVALIDATE:
	  assert(0 && "requesting metadata we've been told is invalid!");

	case STATE_CLEANUP:
	  assert(0 && "requesting metadata in CLEANUP state!");
	}
      }

      if(issue_request) {
	ActiveMessage<MetadataRequestMessage> amsg(owner);
	amsg->id = id;
	amsg.commit();
      }

      return e;
    }

  bool MetadataBase::initiate_cleanup(ID::IDType id, bool local_only /*= false*/)
    {
      NodeSet invals_to_send;
      {
	AutoLock<> a(mutex);

	assert(state == STATE_VALID);

	// eagerly invalidate local contents
	do_invalidate();

	if(remote_copies.empty()) {
	  state = STATE_INVALID;
	} else {
          if(local_only) {
            // ignore remote copies
            remote_copies.clear();
            state = STATE_INVALID;
          } else {
            state = STATE_CLEANUP;
            invals_to_send = remote_copies;
          }
	}
      }

      // send invalidations outside the locked section
      if(invals_to_send.empty())
	return true;

      ActiveMessage<MetadataInvalidateMessage> amsg(invals_to_send);
      amsg->id = id;
      amsg.commit();
      // can't free object until we receive all the acks
      return false;
    }

    void MetadataBase::handle_invalidate(void)
    {
      AutoLock<> a(mutex);

      switch(state) {
      case STATE_VALID: 
	{
	  // was valid, now invalid (up to app to make sure no races exist)
	  state = STATE_INVALID;
	  do_invalidate();
	  break;
	}

      case STATE_REQUESTED:
	{
	  // hopefully rare case where invalidation passes response to initial request
	  state = STATE_INVALIDATE;
	  break;
	}

      default:
	assert(0);
      }
    }

    bool MetadataBase::handle_inval_ack(int sender)
    {
      bool last_copy;
      {
	AutoLock<> a(mutex);

	assert(remote_copies.contains(sender));
	remote_copies.remove(sender);
	last_copy = remote_copies.empty();
	if(last_copy)
	  state = STATE_INVALID;
      }

      return last_copy;
    }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class MetadataRequestMessage
  //

  /*static*/ void MetadataRequestMessage::handle_message(NodeID sender, const MetadataRequestMessage &args,
							 const void *data, size_t datalen)
  {
    // switch on different types of objects that can have metadata
    ID id(args.id);

    log_metadata.info() << "metadata for " << id << " requested by " << sender;

    bool send_response = false;
    Serialization::DynamicBufferSerializer dbs(4096);

    if(id.is_instance()) {
      RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.id);
      bool valid = impl->metadata.handle_request(sender);
      if(valid) {
        send_response = true;
        impl->metadata.serialize_msg(dbs);
      }
    }
    else {
      assert(0);
    }

    if(send_response) {
      size_t offset = 0;
      size_t total_bytes = dbs.bytes_used();

      while(offset < total_bytes) {
        size_t to_send = std::min(total_bytes - offset,
                                  ActiveMessage<MetadataResponseMessage>::recommended_max_payload(sender,
                                                                                                  false /*without congestion*/));

	ActiveMessage<MetadataResponseMessage> amsg(sender, to_send);
	amsg->id = args.id;
        amsg->offset = offset;
        amsg->total_bytes = total_bytes;
        amsg.add_payload(static_cast<const char *>(dbs.get_buffer()) + offset,
                         to_send);
	amsg.commit();

        offset += to_send;
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MetadataResponseMessage
  //

  /*static*/ void MetadataResponseMessage::handle_message(NodeID sender,
							  const MetadataResponseMessage &args,
							  const void *data, size_t datalen)
  {
    log_metadata.info("metadata for " IDFMT " received - %zd bytes",
		      args.id, datalen);

    // switch on different types of objects that can have metadata
    ID id(args.id);
    if(id.is_instance()) {
      RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.id);
      if(args.total_bytes == datalen) {
        // complete message - can deserialize directly
        impl->metadata.deserialize(data, datalen);
        impl->metadata.handle_response();
      } else {
        // fragment reassembly required
        char *buffer = impl->metadata.frag_buffer.load_acquire();
        if(!buffer) {
          char *new_buffer = new char[args.total_bytes];
          if(impl->metadata.frag_buffer.compare_exchange(buffer, new_buffer)) {
            // we're using our buffer
            buffer = new_buffer;
          } else {
            // somebody else allocated, free ours
            delete[] new_buffer;
          }
        }
        assert((args.offset + datalen) <= args.total_bytes);
        memcpy(buffer + args.offset, data, datalen);
        size_t prev_bytes_done = impl->metadata.frag_bytes_received.fetch_add(datalen);
        assert((prev_bytes_done + datalen) <= args.total_bytes);
        if((prev_bytes_done + datalen) == args.total_bytes) {
          // safe to deserialize now, but detach reassembly buffer first (to avoid
          //  races with reuse of the instance)
          impl->metadata.frag_buffer.store(0);
          impl->metadata.frag_bytes_received.store_release(0);

          impl->metadata.deserialize(buffer, args.total_bytes);
          impl->metadata.handle_response();

          delete[] buffer;
        }
      }
    } else {
      assert(0);
    }
  }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class MetadataInvalidateMessage
  //

  /*static*/ void MetadataInvalidateMessage::handle_message(NodeID sender,const MetadataInvalidateMessage &args,
							    const void *data, size_t datalen)
  {
    log_metadata.info("received invalidate request for " IDFMT, args.id);

    //
    // switch on different types of objects that can have metadata
    ID id(args.id);
    if(id.is_instance()) {
      RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.id);
      impl->metadata.handle_invalidate();
    } else {
      assert(0);
    }

    // ack the request
    ActiveMessage<MetadataInvalidateAckMessage> amsg(sender);
    amsg->id = args.id;
    amsg.commit();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MetadataInvalidateAckMessage
  //

  /*static*/ void MetadataInvalidateAckMessage::handle_message(NodeID sender,
							       const MetadataInvalidateAckMessage &args,
							       const void *data, size_t datalen)
  {
    log_metadata.info("received invalidate ack for " IDFMT, args.id);

    // switch on different types of objects that can have metadata
    bool last_ack = false;
    ID id(args.id);
    if(id.is_instance()) {
      RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.id);
      last_ack = impl->metadata.handle_inval_ack(sender);
      if(last_ack)
	impl->recycle_instance();
    } else {
      assert(0);
    }

    if(last_ack) {
      log_metadata.info("last inval ack received for " IDFMT, args.id);
    }
  }

  ActiveMessageHandlerReg<MetadataRequestMessage> metadata_request_message_handler;
  ActiveMessageHandlerReg<MetadataResponseMessage> metadata_response_message_handler;
  ActiveMessageHandlerReg<MetadataInvalidateMessage> metadata_invalidate_message_handler;
  ActiveMessageHandlerReg<MetadataInvalidateAckMessage> metadata_invalidate_ack_message_handler;

}; // namespace Realm
