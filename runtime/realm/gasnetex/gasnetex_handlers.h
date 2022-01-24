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

// GASNet-EX network module message handlers

#ifndef GASNETEX_HANDLERS_H
#define GASNETEX_HANDLERS_H

#include "realm/gasnetex/gasnetex_internal.h"


namespace Realm {

  namespace GASNetEXHandlers {

    extern gex_AM_Entry_t handler_table[];
    extern size_t handler_table_size;

    size_t max_request_medium(gex_EP_t src_ep,
			      gex_Rank_t tgt_rank,
			      gex_EP_Index_t tgt_ep_index,
			      size_t hdr_bytes,
			      gex_Event_t *lc_opt, gex_Flags_t flags);

    size_t max_request_long(gex_EP_t src_ep,
			    gex_Rank_t tgt_rank,
			    gex_EP_Index_t tgt_ep_index,
			    size_t hdr_bytes,
			    gex_Event_t *lc_opt, gex_Flags_t flags);

    int send_request_short(gex_EP_t src_ep,
			   gex_Rank_t tgt_rank,
			   gex_EP_Index_t tgt_ep_index,
			   gex_AM_Arg_t arg0,
			   const void *hdr, size_t hdr_bytes,
			   gex_Flags_t flags);

    int send_request_medium(gex_EP_t src_ep,
			    gex_Rank_t tgt_rank,
			    gex_EP_Index_t tgt_ep_index,
			    gex_AM_Arg_t arg0,
			    const void *hdr, size_t hdr_bytes,
			    const void *data, size_t data_bytes,
			    gex_Event_t *lc_opt, gex_Flags_t flags);

    gex_AM_SrcDesc_t prepare_request_medium(gex_EP_t src_ep,
					    gex_Rank_t tgt_rank,
					    gex_EP_Index_t tgt_ep_index,
					    size_t hdr_bytes,
					    const void *data,
					    size_t min_data_bytes,
					    size_t max_data_bytes,
					    gex_Event_t *lc_opt,
					    gex_Flags_t flags);

    void commit_request_medium(gex_AM_SrcDesc_t srcdesc,
			       gex_AM_Arg_t arg0,
			       const void *hdr, size_t hdr_bytes,
			       size_t data_bytes);

    int send_request_long(gex_EP_t src_ep,
			  gex_Rank_t tgt_rank,
			  gex_EP_Index_t tgt_ep_index,
			  gex_AM_Arg_t arg0,
			  const void *hdr, size_t hdr_bytes,
			  const void *data, size_t data_bytes,
			  gex_Event_t *lc_opt, gex_Flags_t flags,
			  uintptr_t dest_addr);

    // sends a long as a "reverse get": the _header_ is sent as the
    //  payload of a medium that contains the src ep_index/ptr so that
    //  the target can do an RMA get of the actual payload
    int send_request_rget(gex_TM_t prim_tm,
			  gex_Rank_t tgt_rank,
			  gex_EP_Index_t tgt_ep_index,
			  gex_AM_Arg_t arg0,
			  const void *hdr, size_t hdr_bytes,
			  gex_EP_Index_t payload_ep_index,
			  const void *payload, size_t payload_bytes,
			  gex_Event_t *lc_opt, gex_Flags_t flags,
			  uintptr_t dest_addr);

    // send the header of a long after the payload has been delivered via
    //  an RMA put: header is sent as a medium
    int send_request_put_header(gex_EP_t src_ep,
                                gex_Rank_t tgt_rank,
                                gex_EP_Index_t tgt_ep_index,
                                gex_AM_Arg_t arg0,
                                const void *hdr, size_t hdr_bytes,
                                uintptr_t dest_addr, size_t payload_bytes,
                                gex_Event_t *lc_opt, gex_Flags_t flags);

    gex_AM_SrcDesc_t prepare_request_batch(gex_EP_t src_ep,
					   gex_Rank_t tgt_rank,
					   gex_EP_Index_t tgt_ep_index,
					   const void *data,
					   size_t min_data_bytes,
					   size_t max_data_bytes,
					   gex_Event_t *lc_opt,
					   gex_Flags_t flags);

    void commit_request_batch(gex_AM_SrcDesc_t srcdesc,
			      gex_AM_Arg_t arg0, gex_AM_Arg_t cksum,
			      size_t data_bytes);

    int send_completion_reply(gex_EP_t src_ep,
			      gex_Rank_t tgt_rank,
			      gex_EP_Index_t tgt_ep_index,
			      const gex_AM_Arg_t *args,
			      size_t nargs,
			      gex_Flags_t flags);

  };

}; // namespace Realm

#endif

