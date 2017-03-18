/* Copyright 2017 Stanford University, NVIDIA Corporation
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

// IndexSpace implementation for Realm

#ifndef REALM_IDX_IMPL_H
#define REALM_IDX_IMPL_H

#include "indexspace.h"
#include "id.h"

#include "activemsg.h"

#include "rsrv_impl.h"

namespace Realm {

    struct ElementMaskImpl {
      //int count, offset;
      //typedef unsigned long long uint64;
      uint64_t bits[0];

      static size_t bytes_needed(coord_t offset, coord_t count)
      {
	size_t need = (((count + 63) >> 6) << 3);
	return need;
      }
	
    };

    class IndexSpaceImpl {
    public:
      IndexSpaceImpl(void);
      ~IndexSpaceImpl(void);

      void init(IndexSpace _me, unsigned _init_owner);

      void init(IndexSpace _me, IndexSpace _parent,
		size_t _num_elmts,
		const ElementMask *_initial_valid_mask = 0, bool _frozen = false);

      static const ID::ID_Types ID_TYPE = ID::ID_INDEXSPACE;

      bool is_parent_of(IndexSpace other);

      size_t instance_size(const ReductionOpUntyped *redop = 0,
			   coord_t list_size = -1);

      coord_t instance_adjust(const ReductionOpUntyped *redop = 0);

      Event request_valid_mask(void);

      IndexSpace me;
      ReservationImpl lock;
      IndexSpaceImpl *next_free;

      struct StaticData {
	IndexSpace parent;
	bool frozen;
	size_t num_elmts;
        size_t first_elmt, last_elmt;
        // This had better damn well be the last field
        // in the struct in order to avoid race conditions!
	bool valid;
      };
      struct CoherentData : public StaticData {
	unsigned valid_mask_owners;
	int avail_mask_owner;
      };

      CoherentData locked_data;
      GASNetHSL valid_mask_mutex;
      ElementMask *valid_mask;
      int valid_mask_count;
      bool valid_mask_complete;
      Event valid_mask_event;
      int valid_mask_first, valid_mask_last;
      bool valid_mask_contig;
      ElementMask *avail_mask;
    };

    class IndexSpaceAllocatorImpl {
    public:
      IndexSpaceAllocatorImpl(IndexSpaceImpl *_is_impl);

      ~IndexSpaceAllocatorImpl(void);

      coord_t alloc_elements(size_t count = 1);

      void reserve_elements(coord_t ptr, size_t count = 1);

      void free_elements(coord_t ptr, size_t count = 1);

      IndexSpaceImpl *is_impl;
    };

    // active messages

    struct ValidMaskRequestMessage {
      struct RequestArgs {
	IndexSpace is;
	int sender;
      };

      static void handle_request(RequestArgs args);
      
      typedef ActiveMessageShortNoReply<VALID_MASK_REQ_MSGID,
 				        RequestArgs,
				        handle_request> Message;

      static void send_request(gasnet_node_t target, IndexSpace is);
    };

    struct ValidMaskDataMessage {
      struct RequestArgs : public BaseMedium {
	IndexSpace is;
	unsigned block_id;
	coord_t first_element;
	size_t num_elements;
	coord_t first_enabled_elmt;
	coord_t last_enabled_elmt;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<VALID_MASK_DATA_MSGID,
				         RequestArgs,
				         handle_request> Message;
      
      static void send_request(gasnet_node_t target, IndexSpace is, unsigned block_id,
			       coord_t first_element, size_t num_elements,
			       coord_t first_enabled_elmt, coord_t last_enabled_elmt,
			       const void *data, size_t datalen, int payload_mode);
    };

    struct ValidMaskFetchMessage {
      struct RequestArgs {
        IndexSpace is;
        Event complete;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<VALID_MASK_FTH_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(gasnet_node_t target, IndexSpace is, Event complete);
    };

}; // namespace Realm

#endif // ifndef REALM_IDX_IMPL_H
