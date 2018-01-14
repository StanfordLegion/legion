/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#ifndef REALM_METADATA_H
#define REALM_METADATA_H

#include "realm/event.h"
#include "realm/id.h"
#include "realm/nodeset.h"

#include "realm/activemsg.h"

namespace Realm {

  class GenEventImpl;

    class MetadataBase {
    public:
      MetadataBase(void);
      ~MetadataBase(void);

      enum State { STATE_INVALID,
		   STATE_VALID,
		   STATE_REQUESTED,
		   STATE_INVALIDATE,  // if invalidate passes normal request response
		   STATE_CLEANUP };

      bool is_valid(void) const { return state == STATE_VALID; }
      
      // used by owner, may need to send responses to early requests
      void mark_valid(NodeSet& early_reqs);

      // returns true if a response should be sent immediately (i.e. data is valid)
      bool handle_request(int requestor);

      // returns an Event for when data will be valid
      Event request_data(int owner, ID::IDType id);
      void await_data(bool block = true);  // request must have already been made
      void handle_response(void);
      void handle_invalidate(void);

      // these return true once all remote copies have been invalidated
      bool initiate_cleanup(ID::IDType id);
      bool handle_inval_ack(int sender);

    protected:
      GASNetHSL mutex;
      State state;  // current state
      Event valid_event;
      NodeSet remote_copies;
    };

    // active messages
    
    struct MetadataRequestMessage {
      struct RequestArgs {
	int node;
	ID::IDType id;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<METADATA_REQUEST_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(NodeID target, ID::IDType id);
    };

    struct MetadataResponseMessage {
      struct RequestArgs : public BaseMedium {
	ID::IDType id;
      };
	
      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<METADATA_RESPONSE_MSGID,
					 RequestArgs,
					 handle_request> Message;

      static void send_request(NodeID target, ID::IDType id, 
			       const void *data, size_t datalen, int payload_mode);
      static void broadcast_request(const NodeSet& targets, ID::IDType id,
				    const void *data, size_t datalen);
    };

    struct MetadataInvalidateMessage {
      struct RequestArgs {
	int owner;
	ID::IDType id;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<METADATA_INVALIDATE_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(NodeID target, ID::IDType id);
      static void broadcast_request(const NodeSet& targets, ID::IDType id);
    };

    struct MetadataInvalidateAckMessage {
      struct RequestArgs {
	NodeID node;
	ID::IDType id;
      };

      static void handle_request(RequestArgs args);
    
      typedef ActiveMessageShortNoReply<METADATA_INVALIDATE_ACK_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(NodeID target, ID::IDType id);
    };
    
}; // namespace Realm

#endif // ifndef REALM_METADATA_H
