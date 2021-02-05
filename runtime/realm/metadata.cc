/* Copyright 2021 Stanford University, NVIDIA Corporation
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

    bool MetadataBase::initiate_cleanup(ID::IDType id)
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
	  state = STATE_CLEANUP;
	  invals_to_send = remote_copies;
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
    if(id.is_instance()) {
      RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.id);
      bool valid = impl->metadata.handle_request(sender);
      if(valid) {
	Serialization::ByteCountSerializer bcs;
	impl->metadata.serialize_msg(bcs);
	size_t req_size = bcs.bytes_used();
	ActiveMessage<MetadataResponseMessage> amsg(sender, req_size);
	impl->metadata.serialize_msg(amsg);
	amsg->id = args.id;
	log_metadata.info("metadata for " IDFMT " requested by %d",
			  args.id, sender);
	amsg.commit();
      }
    }
    else {
      assert(0);
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
      impl->metadata.deserialize(data, datalen);
      impl->metadata.handle_response();
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
