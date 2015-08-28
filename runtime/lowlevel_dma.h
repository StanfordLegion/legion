/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#ifndef LOWLEVEL_DMA_H
#define LOWLEVEL_DMA_H

#include "lowlevel_impl.h"
#include "activemsg.h"

namespace Realm {
  class CoreReservationSet;
};

namespace LegionRuntime {
  namespace LowLevel {
    struct RemoteCopyArgs : public BaseMedium {
      ReductionOpID redop_id;
      bool red_fold;
      Event before_copy, after_copy;
      int priority;
    };

    struct RemoteFillArgs : public BaseMedium {
      RegionInstance inst;
      unsigned offset, size;
      Event before_fill, after_fill;
      int priority;
    };

    extern void handle_remote_copy(RemoteCopyArgs args, const void *data, size_t msglen);

    extern void handle_remote_fill(RemoteFillArgs args, const void *data, size_t msglen);

    enum DMAActiveMessageIDs {
      REMOTE_COPY_MSGID = 200,
      REMOTE_FILL_MSGID = 201,
    };

    typedef ActiveMessageMediumNoReply<REMOTE_COPY_MSGID,
				       RemoteCopyArgs,
				       handle_remote_copy> RemoteCopyMessage;

    typedef ActiveMessageMediumNoReply<REMOTE_FILL_MSGID,
                                       RemoteFillArgs,
                                       handle_remote_fill> RemoteFillMessage;

    extern void init_dma_handler(void);

    extern void start_dma_worker_threads(int count, Realm::CoreReservationSet& crs);
    extern void stop_dma_worker_threads(void);

    /*
    extern Event enqueue_dma(IndexSpace idx,
			     RegionInstance src, 
			     RegionInstance target,
			     size_t elmt_size,
			     size_t bytes_to_copy,
			     Event before_copy,
			     Event after_copy = Event::NO_EVENT);
    */

    // An important helper method used in other places
    static inline off_t calc_mem_loc(off_t alloc_offset, off_t field_start, int field_size, int elmt_size,
				     int block_size, int index)
    {
      return (alloc_offset +                                      // start address
	      ((index / block_size) * block_size * elmt_size) +   // full blocks
	      (field_start * block_size) +                        // skip other fields
	      ((index % block_size) * field_size));               // some some of our fields within our block
    }

  };
};

#endif
