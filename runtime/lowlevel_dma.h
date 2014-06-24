/* Copyright 2014 Stanford University
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

namespace LegionRuntime {
  namespace LowLevel {
    struct RemoteCopyArgs : public BaseMedium {
      ReductionOpID redop_id;
      bool red_fold;
      Event before_copy, after_copy;
      int priority;
    };

    extern void handle_remote_copy(RemoteCopyArgs args, const void *data, size_t msglen);

    enum DMAActiveMessageIDs {
      REMOTE_COPY_MSGID = 200,
    };

    typedef ActiveMessageMediumNoReply<REMOTE_COPY_MSGID,
				       RemoteCopyArgs,
				       handle_remote_copy> RemoteCopyMessage;

    extern void init_dma_handler(void);

    extern void start_dma_worker_threads(int count);
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
  };
};

#endif
