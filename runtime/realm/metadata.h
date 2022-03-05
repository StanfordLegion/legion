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

#ifndef REALM_METADATA_H
#define REALM_METADATA_H

#include "realm/event.h"
#include "realm/id.h"
#include "realm/nodeset.h"
#include "realm/network.h"
#include "realm/mutex.h"
#include "realm/atomics.h"

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

      void handle_response(void);
      void handle_invalidate(void);

      // these return true once all remote copies have been invalidated
      bool initiate_cleanup(ID::IDType id, bool local_only = false);
      bool handle_inval_ack(int sender);

    protected:
      virtual void do_invalidate(void) = 0;

      Mutex mutex;
      State state;  // current state
      Event valid_event;
      NodeSet remote_copies;

      // for handling fragmentation of metadata responses
      friend struct MetadataResponseMessage;
      atomic<char *> frag_buffer;
      atomic<size_t> frag_bytes_received;
    };

    // active messages
    
    struct MetadataRequestMessage {
      ID::IDType id;

      static void handle_message(NodeID sender,const MetadataRequestMessage &msg,
				 const void *data, size_t datalen);
    };

    struct MetadataResponseMessage {
      ID::IDType id;
      size_t offset, total_bytes;
      static void handle_message(NodeID sender,const MetadataResponseMessage &msg,
				 const void *data, size_t datalen);
    };

    struct MetadataInvalidateMessage {
      ID::IDType id;

      static void handle_message(NodeID sender,const MetadataInvalidateMessage &msg,
				 const void *data, size_t datalen);
    };

    struct MetadataInvalidateAckMessage {
      ID::IDType id;

      static void handle_message(NodeID sender,const MetadataInvalidateAckMessage &msg,
				 const void *data, size_t datalen);
    };
    
}; // namespace Realm

#endif // ifndef REALM_METADATA_H
