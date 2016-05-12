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

#ifndef REALM_METADATA_H
#define REALM_METADATA_H

#include "event.h"
#include "id.h"
#include "nodeset.h"

#include "activemsg.h"

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

      void mark_valid(void); // used by owner
      void handle_request(int requestor);

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

      static void send_request(gasnet_node_t target, ID::IDType id);
    };

    struct MetadataResponseMessage {
      struct RequestArgs : public BaseMedium {
	ID::IDType id;
      };
	
      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<METADATA_RESPONSE_MSGID,
					 RequestArgs,
					 handle_request> Message;

      static void send_request(gasnet_node_t target, ID::IDType id, 
			       const void *data, size_t datalen, int payload_mode);
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

      static void send_request(gasnet_node_t target, ID::IDType id);
      static void broadcast_request(const NodeSet& targets, ID::IDType id);
    };

    struct MetadataInvalidateAckMessage {
      struct RequestArgs {
	gasnet_node_t node;
	ID::IDType id;
      };

      static void handle_request(RequestArgs args);
    
      typedef ActiveMessageShortNoReply<METADATA_INVALIDATE_ACK_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(gasnet_node_t target, ID::IDType id);
    };
    
}; // namespace Realm

#endif // ifndef REALM_METADATA_H
