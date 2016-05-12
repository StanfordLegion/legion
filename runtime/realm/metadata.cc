/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "metadata.h"

#include "event_impl.h"
#include "inst_impl.h"
#include "runtime_impl.h"

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

    void MetadataBase::mark_valid(void)
    {
      // don't actually need lock for this
      assert(remote_copies.empty()); // should not have any valid remote copies if we weren't valid
      state = STATE_VALID;
    }

    void MetadataBase::handle_request(int requestor)
    {
      // just add the requestor to the list of remote nodes with copies
      AutoHSLLock a(mutex);

      assert(is_valid());
      assert(!remote_copies.contains(requestor));
      remote_copies.add(requestor);
    }

    void MetadataBase::handle_response(void)
    {
      // update the state, and
      // if there was an event, we'll trigger it
      Event to_trigger = Event::NO_EVENT;
      {
	AutoHSLLock a(mutex);

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

      // sanity-check - should never be requesting data from ourselves
      assert(((unsigned)owner) != gasnet_mynode());

      Event e = Event::NO_EVENT;
      bool issue_request = false;
      {
	AutoHSLLock a(mutex);

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
	    issue_request = true;
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

      if(issue_request)
	MetadataRequestMessage::send_request(owner, id);

      return e;
    }

    void MetadataBase::await_data(bool block /*= true*/)
    {
      // early out - valid data means no waiting
      if(state == STATE_VALID) return;

      // take lock to get event - must have already been requested (we don't have enough
      //  information to do that now)
      Event e = Event::NO_EVENT;
      {
	AutoHSLLock a(mutex);

	assert(state != STATE_INVALID);
	e = valid_event;
      }

      if(!e.has_triggered())
        e.wait(); // FIXME
    }

    bool MetadataBase::initiate_cleanup(ID::IDType id)
    {
      NodeSet invals_to_send;
      {
	AutoHSLLock a(mutex);

	assert(state == STATE_VALID);

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

      MetadataInvalidateMessage::broadcast_request(invals_to_send, id);

      // can't free object until we receive all the acks
      return false;
    }

    void MetadataBase::handle_invalidate(void)
    {
      AutoHSLLock a(mutex);

      switch(state) {
      case STATE_VALID: 
	{
	  // was valid, now invalid (up to app to make sure no races exist)
	  state = STATE_INVALID;
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
	AutoHSLLock a(mutex);

	assert(remote_copies.contains(sender));
	remote_copies.remove(sender);
	last_copy = remote_copies.empty();
      }

      return last_copy;
    }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class MetadataRequestMessage
  //

  /*static*/ void MetadataRequestMessage::handle_request(RequestArgs args)
  {
    void *data = 0;
    size_t datalen = 0;
    
    // switch on different types of objects that can have metadata
    switch(ID(args.id).type()) {
    case ID::ID_INSTANCE:
      {
	RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.id);
	impl->metadata.handle_request(args.node);
	data = impl->metadata.serialize(datalen);
	break;
      }

    default:
      assert(0);
    }

    log_metadata.info("metadata for " IDFMT " requested by %d - %zd bytes",
		      args.id, args.node, datalen);
    MetadataResponseMessage::send_request(args.node, args.id, data, datalen, PAYLOAD_FREE);
  }

  /*static*/ void MetadataRequestMessage::send_request(gasnet_node_t target, ID::IDType id)
  {
    RequestArgs args;

    args.node = gasnet_mynode();
    args.id = id;
    Message::request(target, args);
  }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class MetadataResponseMessage
  //

  /*static*/ void MetadataResponseMessage::handle_request(RequestArgs args,
							  const void *data,
							  size_t datalen)
  {
    log_metadata.info("metadata for " IDFMT " received - %zd bytes",
		      args.id, datalen);

    // switch on different types of objects that can have metadata
    switch(ID(args.id).type()) {
    case ID::ID_INSTANCE:
      {
	RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.id);
	impl->metadata.deserialize(data, datalen);
	impl->metadata.handle_response();
	break;
      }

    default:
      assert(0);
    }
  }

  /*static*/ void MetadataResponseMessage::send_request(gasnet_node_t target,
							ID::IDType id, 
							const void *data,
							size_t datalen,
							int payload_mode)
  {
    RequestArgs args;

    args.id = id;
    Message::request(target, args, data, datalen, payload_mode);
  }
  
  
  ////////////////////////////////////////////////////////////////////////
  //
  // class MetadataInvalidateMessage
  //

  /*static*/ void MetadataInvalidateMessage::handle_request(RequestArgs args)
  {
    log_metadata.info("received invalidate request for " IDFMT, args.id);

    //
    // switch on different types of objects that can have metadata
    switch(ID(args.id).type()) {
    case ID::ID_INSTANCE:
      {
	RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.id);
	impl->metadata.handle_invalidate();
	break;
      }

    default:
      assert(0);
    }

    // ack the request
    MetadataInvalidateAckMessage::send_request(args.owner, args.id);
  }

  /*static*/ void MetadataInvalidateMessage::send_request(gasnet_node_t target,
							  ID::IDType id)
  {
    RequestArgs args;

    args.owner = gasnet_mynode();
    args.id = id;
    Message::request(target, args);
  }

  template <typename T>
  struct BroadcastHelper : public T::RequestArgs {
    inline void apply(gasnet_node_t target)
    {
      T::Message::request(target, *this);
    }

    void broadcast(const NodeSet& targets)
    {
      targets.map(*this);
    }
  };
  
  /*static*/ void MetadataInvalidateMessage::broadcast_request(const NodeSet& targets,
							       ID::IDType id)
  {
    BroadcastHelper<MetadataInvalidateMessage> args;

    args.owner = gasnet_mynode();
    args.id = id;
    args.broadcast(targets);
  }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class MetadataInvalidateAckMessage
  //

  /*static*/ void MetadataInvalidateAckMessage::handle_request(RequestArgs args)
  {
    log_metadata.info("received invalidate ack for " IDFMT, args.id);

    // switch on different types of objects that can have metadata
    bool last_ack = false;
    switch(ID(args.id).type()) {
    case ID::ID_INSTANCE:
      {
	RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.id);
	last_ack = impl->metadata.handle_inval_ack(args.node);
	break;
      }

    default:
      assert(0);
    }

    if(last_ack) {
      log_metadata.info("last inval ack received for " IDFMT, args.id);
    }
  }
   
  /*static*/ void MetadataInvalidateAckMessage::send_request(gasnet_node_t target, ID::IDType id)
  {
    RequestArgs args;

    args.node = gasnet_mynode();
    args.id = id;
    Message::request(target, args);
  }
  

}; // namespace Realm
