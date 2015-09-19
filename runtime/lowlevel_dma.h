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

  class DmaRequestQueue;
  typedef uint64_t XferDesID;
  class DmaRequest : public Realm::Operation {
  public:
    DmaRequest(int _priority, Event _after_copy)
	: Operation(_after_copy, Realm::ProfilingRequestSet()),
	  state(STATE_INIT), priority(_priority) {
      pthread_mutex_init(&request_lock, NULL);
    }

    DmaRequest(int _priority, Event _after_copy,
               const Realm::ProfilingRequestSet &reqs)
      : Realm::Operation(_after_copy, reqs), state(STATE_INIT),
        priority(_priority) {
      pthread_mutex_init(&request_lock, NULL);
    }

    virtual ~DmaRequest(void) {
      pthread_mutex_destroy(&request_lock);
    }

    virtual bool check_readiness(bool just_check, DmaRequestQueue *rq) = 0;

    virtual bool handler_safe(void) = 0;

    virtual void perform_dma(void) = 0;

    enum State {
	STATE_INIT,
	STATE_METADATA_FETCH,
	STATE_BEFORE_EVENT,
	STATE_INST_LOCK,
	STATE_READY,
	STATE_QUEUED,
	STATE_DONE
    };

    State state;
    int priority;

    // <NEWDMA>
	pthread_mutex_t request_lock;
    std::vector<XferDesID> path;
    std::set<XferDesID> complete_xd;

    // Returns true if all xfer des of this DmaRequest
    // have been marked completed
    // This return val is a signal for delete this DmaRequest
    bool notify_xfer_des_completion(XferDesID guid)
    {
      pthread_mutex_lock(&request_lock);
      complete_xd.insert(guid);
      bool all_completed = (complete_xd.size() == path.size());
      pthread_mutex_unlock(&request_lock);
      return all_completed;
    }
    // </NEWDMA>

    class Waiter : public EventWaiter {
    public:
      Waiter(void) { }
      virtual ~Waiter(void) { }
    public:
	Reservation current_lock;
	DmaRequestQueue *queue;
	DmaRequest *req;

	void sleep_on_event(Event e, Reservation l = Reservation::NO_RESERVATION);

	virtual bool event_triggered(void);
	virtual void print_info(FILE *f);
    };
  };

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
      //int priority;
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

    extern void start_dma_system(int count, int max_nr
#ifdef USE_CUDA
                          ,std::vector<GPUProcessor*> &local_gpus
#endif
                         );

    extern void stop_dma_system(void);

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
