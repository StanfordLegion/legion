/* Copyright 2013 Stanford University
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

#include "activemsg.h"
#include "utilities.h"

#ifndef __GNUC__
#include "atomics.h" // for __sync_add_and_fetch
#endif

#include <queue>
#include <cassert>

#include "lowlevel_impl.h"

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

#define CHECK_GASNET(cmd) do { \
  int ret = (cmd); \
  if(ret != GASNET_OK) { \
    fprintf(stderr, "GASNET: %s = %d (%s, %s)\n", #cmd, ret, gasnet_ErrorName(ret), gasnet_ErrorDesc(ret)); \
    exit(1); \
  } \
} while(0)

enum { MSGID_LONG_EXTENSION = 253,
       MSGID_FLIP_REQ = 254,
       MSGID_FLIP_ACK = 255 };

#ifdef DEBUG_MEM_REUSE
static int payload_count = 0;
#endif

static const int DEFERRED_FREE_COUNT = 100;
gasnet_hsl_t deferred_free_mutex;
int deferred_free_pos;
void *deferred_frees[DEFERRED_FREE_COUNT];

void init_deferred_frees(void)
{
  gasnet_hsl_init(&deferred_free_mutex);
  deferred_free_pos = 0;
  for(int i = 0; i < DEFERRED_FREE_COUNT; i++)
    deferred_frees[i] = 0;
}

void deferred_free(void *ptr)
{
#ifdef DEBUG_MEM_REUSE
  printf("%d: deferring free of %p\n", gasnet_mynode(), ptr);
#endif
  gasnet_hsl_lock(&deferred_free_mutex);
  void *oldptr = deferred_frees[deferred_free_pos];
  deferred_frees[deferred_free_pos] = ptr;
  deferred_free_pos = (deferred_free_pos + 1) % DEFERRED_FREE_COUNT;
  gasnet_hsl_unlock(&deferred_free_mutex);
  if(oldptr) {
#ifdef DEBUG_MEM_REUSE
    printf("%d: actual free of %p\n", gasnet_mynode(), oldptr);
#endif
    //free(oldptr);
  }
}

struct OutgoingMessage {
  OutgoingMessage(unsigned _msgid, unsigned _num_args, const void *_args)
    : msgid(_msgid), num_args(_num_args),
      payload(0), payload_size(0), payload_mode(PAYLOAD_NONE)
  {
    for(unsigned i = 0; i < _num_args; i++)
      args[i] = ((const int *)_args)[i];
  }

  ~OutgoingMessage(void)
  {
    if((payload_mode == PAYLOAD_COPY) || (payload_mode == PAYLOAD_FREE)) {
      if(payload_size > 0) {
#ifdef DEBUG_MEM_REUSE
	for(size_t i = 0; i < payload_size >> 2; i++)
	  ((unsigned *)payload)[i] = ((0xdc + gasnet_mynode()) << 24) + payload_num;
	//memset(payload, 0xdc+gasnet_mynode(), payload_size);
	printf("%d: freeing payload %x = [%p, %p)\n",
	       gasnet_mynode(), payload_num, payload, ((char *)payload) + payload_size);
#endif
	deferred_free(payload);
      }
    }
  }

  void set_payload(void *_payload, size_t _payload_size, int _payload_mode);

  unsigned msgid;
  unsigned num_args;
  void *payload;
  size_t payload_size;
  int payload_mode;
  int args[16];
#ifdef DEBUG_MEM_REUSE
  int payload_num;
#endif
};
    
class ActiveMessageEndpoint {
public:
  static const int NUM_LMBS = 2;
  static const size_t LMB_SIZE = (4 << 20);

  ActiveMessageEndpoint(gasnet_node_t _peer, const gasnet_seginfo_t *seginfos)
    : peer(_peer)
  {
    gasnet_hsl_init(&mutex);
    gasnett_cond_init(&cond);

    cur_write_lmb = 0;
    cur_write_offset = 0;
    cur_write_count = 0;

    cur_long_ptr = 0;
    cur_long_chunk_idx = 0;
    cur_long_size = 0;

    for(int i = 0; i < NUM_LMBS; i++) {
      lmb_w_bases[i] = ((char *)(seginfos[peer].addr)) + (seginfos[peer].size - LMB_SIZE * (gasnet_mynode() * NUM_LMBS + i + 1));
      lmb_r_bases[i] = ((char *)(seginfos[gasnet_mynode()].addr)) + (seginfos[peer].size - LMB_SIZE * (peer * NUM_LMBS + i + 1));
      lmb_r_counts[i] = 0;
      lmb_w_avail[i] = true;
    }
  }

  int push_messages(int max_to_send = 0, bool wait = false)
  {
    int count = 0;

    while((max_to_send == 0) || (count < max_to_send)) {
      // attempt to get the mutex that covers the outbound queues - do not
      //  block
      int ret = gasnet_hsl_trylock(&mutex);
      if(ret == GASNET_ERR_NOT_READY) break;

      // try to send a long message, but only if we have an LMB available
      //  on the receiving end
      if((out_long_hdrs.size() > 0) && lmb_w_avail[cur_write_lmb]) {
	OutgoingMessage *hdr;
	hdr = out_long_hdrs.front();

	// no payload?  this happens when a short/medium message needs to be ordered with long messages
	if(hdr->payload_size == 0) {
	  out_long_hdrs.pop();
	  gasnet_hsl_unlock(&mutex);
	  send_short(hdr);
	  delete hdr;
	  count++;
	  continue;
	}

	// do we have enough room in the current LMB?
	assert(hdr->payload_size <= LMB_SIZE);
	if((cur_write_offset + hdr->payload_size) <= LMB_SIZE) {
	  // we can send the message - update lmb pointers and remove the
	  //  packet from the queue, and then drop them mutex before
	  //  sending the message
	  char *dest_ptr = lmb_w_bases[cur_write_lmb] + cur_write_offset;
	  cur_write_offset += hdr->payload_size;
	  cur_write_count++;
	  out_long_hdrs.pop();

	  gasnet_hsl_unlock(&mutex);
#ifdef DEBUG_LMB
	  printf("LMB: sending %zd bytes %d->%d, [%p,%p)\n",
		 hdr->payload_size, gasnet_mynode(), peer,
		 dest_ptr, dest_ptr + hdr->payload_size);
#endif
	  send_long(hdr, dest_ptr);
	  delete hdr;
	  count++;
	  continue;
	} else {
	  // can't send the message, so flip the buffer that's now full
	  int flip_buffer = cur_write_lmb;
	  int flip_count = cur_write_count;
	  lmb_w_avail[cur_write_lmb] = false;
	  cur_write_lmb = (cur_write_lmb + 1) % NUM_LMBS;
	  cur_write_offset = 0;
	  cur_write_count = 0;

	  // now let go of the lock and send the flip request
	  gasnet_hsl_unlock(&mutex);

#ifdef DEBUG_LMB
	  printf("LMB: flipping buffer %d for %d->%d, [%p,%p), count=%d\n",
		 flip_buffer, gasnet_mynode(), peer, lmb_w_bases[flip_buffer],
		 lmb_w_bases[flip_buffer]+LMB_SIZE, flip_count);
#endif

	  gasnet_AMRequestShort2(peer, MSGID_FLIP_REQ,
				 flip_buffer, flip_count);

	  continue;
	}
      }

      // couldn't send a long message, try a short message
      if(out_short_hdrs.size() > 0) {
	OutgoingMessage *hdr = out_short_hdrs.front();
	out_short_hdrs.pop();

	// now let go of lock and send message
	gasnet_hsl_unlock(&mutex);

	send_short(hdr);
	delete hdr;
	count++;
	continue;
      }

      // Couldn't do anything so if we were told to wait, goto sleep
      if (wait)
      {
        gasnett_cond_wait(&cond, &mutex.lock);
      }
      // if we get here, we didn't find anything to do, so break out of loop
      //  after releasing the lock
      gasnet_hsl_unlock(&mutex);
      break;
    }

    return count;
  }

  void enqueue_message(OutgoingMessage *hdr, bool in_order)
  {
    // need to hold the mutex in order to push onto one of the queues
    gasnet_hsl_lock(&mutex);

    // messages that don't need space in the LMB can progress when the LMB is full
    //  (unless they need to maintain ordering with long packets)
    if(!in_order && (hdr->payload_size <= gasnet_AMMaxMedium()))
      out_short_hdrs.push(hdr);
    else
      out_long_hdrs.push(hdr);
    // Signal in case there is a sleeping sender
    gasnett_cond_signal(&cond);

    gasnet_hsl_unlock(&mutex);
  }

  void handle_long_msgptr(void *ptr)
  {
    // can figure out which buffer it is without holding lock
    int r_buffer = -1;
    for(int i = 0; i < NUM_LMBS; i++)
      if((ptr >= lmb_r_bases[i]) && (ptr < (lmb_r_bases[i] + LMB_SIZE))) {
      r_buffer = i;
      break;
    }
    if(r_buffer < 0) {
      // probably a medium message?
      return;
    }
    //assert(r_buffer >= 0);

#ifdef DEBUG_LMB
    printf("LMB: received %p for %d->%d in buffer %d, [%p, %p)\n",
	   ptr, peer, gasnet_mynode(), r_buffer, lmb_r_bases[r_buffer],
	   lmb_r_bases[r_buffer] + LMB_SIZE);
#endif

    // now take the lock to increment the r_count and decide if we need
    //  to ack (can't actually send it here, so queue it up)
    gasnet_hsl_lock(&mutex);
    lmb_r_counts[r_buffer]++;
    if(lmb_r_counts[r_buffer] == 0) {
#ifdef DEBUG_LMB
      printf("LMB: acking flip of buffer %d for %d->%d, [%p,%p)\n",
	     r_buffer, peer, gasnet_mynode(), lmb_r_bases[r_buffer],
	     lmb_r_bases[r_buffer]+LMB_SIZE);
#endif

      OutgoingMessage *hdr = new OutgoingMessage(MSGID_FLIP_ACK, 1, &r_buffer);
      out_short_hdrs.push(hdr);
      // wake up a sender
      gasnett_cond_signal(&cond);
    }
    gasnet_hsl_unlock(&mutex);
  }

  size_t adjust_long_msgsize(void *ptr, size_t orig_size)
  {
    // can figure out which buffer it is without holding lock
    int r_buffer = -1;
    for(int i = 0; i < NUM_LMBS; i++)
      if((ptr >= lmb_r_bases[i]) && (ptr < (lmb_r_bases[i] + LMB_SIZE))) {
      r_buffer = i;
      break;
    }
    if(r_buffer < 0) {
      // probably a medium message?
      return orig_size;
    }

    // need to hold the lock
    gasnet_hsl_lock(&mutex);
    size_t adjusted_size = orig_size;

    if(cur_long_ptr != 0) {
      // pointer had better match and we must have received all the chunks
      assert(cur_long_ptr == ptr);
      assert(cur_long_chunk_idx == 1);

      adjusted_size += cur_long_size;

      // printf("%d: adjusting long message size (%p, %d, %zd): %zd -> %zd\n",
      // 	     gasnet_mynode(), cur_long_ptr, cur_long_chunk_idx, cur_long_size, orig_size, adjusted_size);

      cur_long_ptr = 0;
      cur_long_chunk_idx = 0;
      cur_long_size = 0;
    }
    gasnet_hsl_unlock(&mutex);

    return adjusted_size;
  }

  void handle_long_extension(void *ptr, int chunk_idx, int size)
  {
    // can figure out which buffer it is without holding lock
    int r_buffer = -1;
    for(int i = 0; i < NUM_LMBS; i++)
      if((ptr >= lmb_r_bases[i]) && (ptr < (lmb_r_bases[i] + LMB_SIZE))) {
      r_buffer = i;
      break;
    }
    // not ok for this to be outside an LMB
    assert(r_buffer >= 0);

    // need to hold the lock
    gasnet_hsl_lock(&mutex);

    if(cur_long_ptr != 0) {
      // continuing an extension - this had better be the next index down
      assert(ptr == cur_long_ptr);
      assert(chunk_idx == (cur_long_chunk_idx - 1));
      
      cur_long_chunk_idx = chunk_idx;
      cur_long_size += size;
    } else {
      // starting a new extension
      cur_long_ptr = ptr;
      cur_long_chunk_idx = chunk_idx;
      cur_long_size = size;
    }

    gasnet_hsl_unlock(&mutex);
  }

  // called when the remote side tells us that there will be no more
  //  messages sent for a given buffer - as soon as we've received them all,
  //  we can ack
  void handle_flip_request(int buffer, int count)
  {
#ifdef DEBUG_LMB
    printf("LMB: received flip of buffer %d for %d->%d, [%p,%p), count=%d\n",
	   buffer, peer, gasnet_mynode(), lmb_r_bases[buffer],
	   lmb_r_bases[buffer]+LMB_SIZE, count);
#endif

    gasnet_hsl_lock(&mutex);
    lmb_r_counts[buffer] -= count;
    if(lmb_r_counts[buffer] == 0) {
#ifdef DEBUG_LMB
      printf("LMB: acking flip of buffer %d for %d->%d, [%p,%p)\n",
	     buffer, peer, gasnet_mynode(), lmb_r_bases[buffer],
	     lmb_r_bases[buffer]+LMB_SIZE);
#endif

      OutgoingMessage *hdr = new OutgoingMessage(MSGID_FLIP_ACK, 1, &buffer);
      out_short_hdrs.push(hdr);
      // Wake up a sender
      gasnett_cond_signal(&cond);
    }
    gasnet_hsl_unlock(&mutex);
  }

  // called when the remote side says it has received all the messages in a
  //  given buffer - we can that mark that write buffer as available again
  //  (don't even need to take the mutex!)
  void handle_flip_ack(int buffer)
  {
#ifdef DEBUG_LMB
    printf("LMB: received flip ack of buffer %d for %d->%d, [%p,%p)\n",
	   buffer, gasnet_mynode(), peer, lmb_w_bases[buffer],
	   lmb_w_bases[buffer]+LMB_SIZE);
#endif

    lmb_w_avail[buffer] = true;
    // wake up a sender in case we had messages waiting for free space
    gasnet_hsl_lock(&mutex);
    gasnett_cond_signal(&cond);
    gasnet_hsl_unlock(&mutex);
  }

protected:
  void send_short(OutgoingMessage *hdr)
  {
    LegionRuntime::LowLevel::DetailedTimer::ScopedPush sp(TIME_AM);
    switch(hdr->num_args) {
    case 1:
      if(hdr->payload_mode != PAYLOAD_NONE)
	gasnet_AMRequestMedium1(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0]);
      else
	gasnet_AMRequestShort1(peer, hdr->msgid, hdr->args[0]);
      break;

    case 2:
      if(hdr->payload_mode != PAYLOAD_NONE)
	gasnet_AMRequestMedium2(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1]);
      else
	gasnet_AMRequestShort2(peer, hdr->msgid, hdr->args[0], hdr->args[1]);
      break;

    case 3:
      if(hdr->payload_mode != PAYLOAD_NONE)
	gasnet_AMRequestMedium3(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2]);
      else
	gasnet_AMRequestShort3(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2]);
      break;

    case 4:
      if(hdr->payload_mode != PAYLOAD_NONE)
	gasnet_AMRequestMedium4(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3]);
      else
	gasnet_AMRequestShort4(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2],
			       hdr->args[3]);
      break;

    case 5:
      if(hdr->payload_mode != PAYLOAD_NONE)
	gasnet_AMRequestMedium5(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4]);
      else
	gasnet_AMRequestShort5(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2],
			       hdr->args[3], hdr->args[4]);
      break;

    case 6:
      if(hdr->payload_mode != PAYLOAD_NONE)
	gasnet_AMRequestMedium6(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5]);
      else
	gasnet_AMRequestShort6(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2],
			       hdr->args[3], hdr->args[4], hdr->args[5]);
      break;

    case 8:
      if(hdr->payload_mode != PAYLOAD_NONE)
	gasnet_AMRequestMedium8(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7]);
      else
	gasnet_AMRequestShort8(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2],
			       hdr->args[3], hdr->args[4], hdr->args[5],
			       hdr->args[6], hdr->args[7]);
      break;

    case 10:
      if(hdr->payload_mode != PAYLOAD_NONE)
	gasnet_AMRequestMedium10(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				 hdr->args[0], hdr->args[1], hdr->args[2],
				 hdr->args[3], hdr->args[4], hdr->args[5],
				 hdr->args[6], hdr->args[7], hdr->args[8],
				 hdr->args[9]);
      else
	gasnet_AMRequestShort10(peer, hdr->msgid,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7], hdr->args[8],
				hdr->args[9]);
      break;

    case 12:
      if(hdr->payload_mode != PAYLOAD_NONE)
	gasnet_AMRequestMedium12(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				 hdr->args[0], hdr->args[1], hdr->args[2],
				 hdr->args[3], hdr->args[4], hdr->args[5],
				 hdr->args[6], hdr->args[7], hdr->args[8],
				 hdr->args[9], hdr->args[10], hdr->args[11]);
      else
	gasnet_AMRequestShort12(peer, hdr->msgid,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7], hdr->args[8],
				hdr->args[9], hdr->args[10], hdr->args[11]);
      break;

    case 16:
      if(hdr->payload_mode != PAYLOAD_NONE)
	gasnet_AMRequestMedium16(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				 hdr->args[0], hdr->args[1], hdr->args[2],
				 hdr->args[3], hdr->args[4], hdr->args[5],
				 hdr->args[6], hdr->args[7], hdr->args[8],
				 hdr->args[9], hdr->args[10], hdr->args[11],
				 hdr->args[12], hdr->args[13], hdr->args[14],
				 hdr->args[15]);
      else
	gasnet_AMRequestShort16(peer, hdr->msgid,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7], hdr->args[8],
				hdr->args[9], hdr->args[10], hdr->args[11],
				hdr->args[12], hdr->args[13], hdr->args[14],
				hdr->args[15]);
      break;

    default:
      fprintf(stderr, "need to support short/medium of size=%d\n", hdr->num_args);
      assert(1==2);
    }
  }
  
  void send_long(OutgoingMessage *hdr, void *dest_ptr)
  {
    LegionRuntime::LowLevel::DetailedTimer::ScopedPush sp(TIME_AM);

    size_t max_long_req = 65000; // gasnet_AMMaxLongRequest();

    size_t msg_size = hdr->payload_size;
    if(msg_size > max_long_req) {
      size_t chunks = (hdr->payload_size + max_long_req - 1) / max_long_req;

      gasnet_handlerarg_t dest_ptr_lo = ((uint64_t)dest_ptr) & 0x0FFFFFFFFULL;
      gasnet_handlerarg_t dest_ptr_hi = ((uint64_t)dest_ptr) >> 32;

      for(unsigned i = chunks-1; i > 0; i--) {
	size_t size = ((i == (chunks-1)) ? (hdr->payload_size % max_long_req) : max_long_req);
	gasnet_AMRequestLong3(peer, MSGID_LONG_EXTENSION, 
			      ((char *)(hdr->payload))+(i * max_long_req), size, 
			      ((char *)dest_ptr)+(i * max_long_req),
			      dest_ptr_lo, dest_ptr_hi, i);
      }

      msg_size = max_long_req;
    }

    switch(hdr->num_args) {
    case 1:
      gasnet_AMRequestLong1(peer, hdr->msgid, 
			    hdr->payload, msg_size, dest_ptr,
			    hdr->args[0]);
      break;

    case 2:
      gasnet_AMRequestLong2(peer, hdr->msgid, 
			    hdr->payload, msg_size, dest_ptr,
			    hdr->args[0], hdr->args[1]);
      break;

    case 3:
      gasnet_AMRequestLong3(peer, hdr->msgid, 
			    hdr->payload, msg_size, dest_ptr,
			    hdr->args[0], hdr->args[1], hdr->args[2]);
      break;

    case 4:
      gasnet_AMRequestLong4(peer, hdr->msgid, 
			    hdr->payload, msg_size, dest_ptr,
			    hdr->args[0], hdr->args[1], hdr->args[2],
			    hdr->args[3]);
      break;

    case 6:
      gasnet_AMRequestLong6(peer, hdr->msgid, 
			    hdr->payload, msg_size, dest_ptr,
			    hdr->args[0], hdr->args[1], hdr->args[2],
			    hdr->args[3], hdr->args[4], hdr->args[5]);
      break;
    case 7:
      gasnet_AMRequestLong7(peer, hdr->msgid,
                            hdr->payload, msg_size, dest_ptr,
                            hdr->args[0], hdr->args[1], hdr->args[2],
                            hdr->args[3], hdr->args[4], hdr->args[5],
                            hdr->args[6]);
      break;
    case 8:
      gasnet_AMRequestLong8(peer, hdr->msgid,
                            hdr->payload, msg_size, dest_ptr,
                            hdr->args[0], hdr->args[1], hdr->args[2],
                            hdr->args[3], hdr->args[4], hdr->args[5],
                            hdr->args[6], hdr->args[7]);
      break;
    case 9:
      gasnet_AMRequestLong9(peer, hdr->msgid,
                            hdr->payload, msg_size, dest_ptr,
                            hdr->args[0], hdr->args[1], hdr->args[2],
                            hdr->args[3], hdr->args[4], hdr->args[5],
                            hdr->args[6], hdr->args[7], hdr->args[8]);
      break;
    case 10:
      gasnet_AMRequestLong10(peer, hdr->msgid,
                            hdr->payload, msg_size, dest_ptr,
                            hdr->args[0], hdr->args[1], hdr->args[2],
                            hdr->args[3], hdr->args[4], hdr->args[5],
                            hdr->args[6], hdr->args[7], hdr->args[8],
                            hdr->args[9]);
      break;
    case 11:
      gasnet_AMRequestLong11(peer, hdr->msgid,
                            hdr->payload, msg_size, dest_ptr,
                            hdr->args[0], hdr->args[1], hdr->args[2],
                            hdr->args[3], hdr->args[4], hdr->args[5],
                            hdr->args[6], hdr->args[7], hdr->args[8],
                            hdr->args[9], hdr->args[10]);
      break;
    case 12:
      gasnet_AMRequestLong12(peer, hdr->msgid,
                            hdr->payload, msg_size, dest_ptr,
                            hdr->args[0], hdr->args[1], hdr->args[2],
                            hdr->args[3], hdr->args[4], hdr->args[5],
                            hdr->args[6], hdr->args[7], hdr->args[8],
                            hdr->args[9], hdr->args[10], hdr->args[11]);
      break;
    case 13:
      gasnet_AMRequestLong13(peer, hdr->msgid,
			    hdr->payload, msg_size, dest_ptr,
                            hdr->args[0], hdr->args[1], hdr->args[2],
                            hdr->args[3], hdr->args[4], hdr->args[5],
                            hdr->args[6], hdr->args[7], hdr->args[8],
                            hdr->args[9], hdr->args[10], hdr->args[11],
			    hdr->args[12]);
      break;
    case 14:
      gasnet_AMRequestLong14(peer, hdr->msgid,
			    hdr->payload, msg_size, dest_ptr,
                            hdr->args[0], hdr->args[1], hdr->args[2],
                            hdr->args[3], hdr->args[4], hdr->args[5],
                            hdr->args[6], hdr->args[7], hdr->args[8],
                            hdr->args[9], hdr->args[10], hdr->args[11],
			    hdr->args[12], hdr->args[13]);
      break;
    case 15:
      gasnet_AMRequestLong15(peer, hdr->msgid,
                            hdr->payload, msg_size, dest_ptr,
                            hdr->args[0], hdr->args[1], hdr->args[2],
                            hdr->args[3], hdr->args[4], hdr->args[5],
                            hdr->args[6], hdr->args[7], hdr->args[8],
                            hdr->args[9], hdr->args[10], hdr->args[11],
                            hdr->args[12], hdr->args[13], hdr->args[14]);
      break;
    case 16:
      gasnet_AMRequestLong16(peer, hdr->msgid,
                            hdr->payload, msg_size, dest_ptr,
                            hdr->args[0], hdr->args[1], hdr->args[2],
                            hdr->args[3], hdr->args[4], hdr->args[5],
                            hdr->args[6], hdr->args[7], hdr->args[8],
                            hdr->args[9], hdr->args[10], hdr->args[11],
                            hdr->args[12], hdr->args[13], hdr->args[14],
                            hdr->args[15]);
      break;

    default:
      fprintf(stderr, "need to support long of size=%d\n", hdr->num_args);
      assert(3==4);
    }
  }

  gasnet_node_t peer;
  
  gasnet_hsl_t mutex;
  gasnett_cond_t cond;
  std::queue<OutgoingMessage *> out_short_hdrs;
  std::queue<OutgoingMessage *> out_long_hdrs;

  int cur_write_lmb, cur_write_count;
  size_t cur_write_offset;
  char *lmb_w_bases[NUM_LMBS];
  char *lmb_r_bases[NUM_LMBS];
  int lmb_r_counts[NUM_LMBS];
  bool lmb_w_avail[NUM_LMBS];
  void *cur_long_ptr;
  int cur_long_chunk_idx;
  size_t cur_long_size;
};

void OutgoingMessage::set_payload(void *_payload, size_t _payload_size, int _payload_mode)
{
  if(_payload_mode != PAYLOAD_NONE) {
    assert(_payload_size <= ActiveMessageEndpoint::LMB_SIZE);
    //assert(_payload_size <= gasnet_AMMaxLongRequest());
#ifdef ENFORCE_MPI_CONDUIT_MAX_LONG_REQUEST
    assert(_payload_size <= 65000); // MPI conduit's max
#endif
    payload_mode = _payload_mode;
    payload_size = _payload_size;
    if(_payload && (payload_mode == PAYLOAD_COPY)) {
      payload = malloc(payload_size);
      assert(payload != 0);
#ifdef DEBUG_MEM_REUSE
#ifdef __GNUC__
      payload_num = __sync_add_and_fetch(&payload_count, 1);
#else
      payload_num = LegionRuntime::LowLevel::__sync_add_and_fetch(&payload_count, 1);
#endif
      printf("%d: copying payload %x = [%p, %p) (%zd)\n",
	     gasnet_mynode(), payload_num, payload, ((char *)payload) + payload_size,
	     payload_size);
#endif
      memcpy(payload, _payload, payload_size);
    } else 
      payload = _payload;
  }
}

static ActiveMessageEndpoint **endpoints;

static void handle_long_extension(gasnet_token_t token, void *buf, size_t nbytes,
				  gasnet_handlerarg_t ptr_lo, gasnet_handlerarg_t ptr_hi, gasnet_handlerarg_t chunk_idx)
{
  gasnet_node_t src;
  gasnet_AMGetMsgSource(token, &src);

  void *dest_ptr = (void *)((((uint64_t)ptr_hi) << 32) | ((uint64_t)(uint32_t)ptr_lo));
  endpoints[src]->handle_long_extension(dest_ptr, chunk_idx, nbytes);
}

static void handle_flip_req(gasnet_token_t token,
		     int flip_buffer, int flip_count)
{
  gasnet_node_t src;
  gasnet_AMGetMsgSource(token, &src);
  endpoints[src]->handle_flip_request(flip_buffer, flip_count);
}

static void handle_flip_ack(gasnet_token_t token,
			    int ack_buffer)
{
  gasnet_node_t src;
  gasnet_AMGetMsgSource(token, &src);
  endpoints[src]->handle_flip_ack(ack_buffer);
}

void init_endpoints(gasnet_handlerentry_t *handlers, int hcount,
		    int gasnet_mem_size_in_mb)
{
  // add in our internal handlers and space we need for LMBs
  int attach_size = ((gasnet_mem_size_in_mb << 20) +
		     (gasnet_nodes() * 
		      ActiveMessageEndpoint::NUM_LMBS *
		      ActiveMessageEndpoint::LMB_SIZE));

  handlers[hcount].index = MSGID_LONG_EXTENSION;
  handlers[hcount].fnptr = (void (*)())handle_long_extension;
  hcount++;
  handlers[hcount].index = MSGID_FLIP_REQ;
  handlers[hcount].fnptr = (void (*)())handle_flip_req;
  hcount++;
  handlers[hcount].index = MSGID_FLIP_ACK;
  handlers[hcount].fnptr = (void (*)())handle_flip_ack;
  hcount++;

  CHECK_GASNET( gasnet_attach(handlers, hcount,
			      attach_size, 0) );

  endpoints = new ActiveMessageEndpoint *[gasnet_nodes()];

  gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[gasnet_nodes()];
  CHECK_GASNET( gasnet_getSegmentInfo(seginfos, gasnet_nodes()) );

  for(int i = 0; i < gasnet_nodes(); i++)
    if(i == gasnet_mynode())
      endpoints[i] = 0;
    else
      endpoints[i] = new ActiveMessageEndpoint(i, seginfos);

  delete[] seginfos;

  init_deferred_frees();
}

static int num_polling_threads = 0;
static pthread_t *polling_threads = 0;
static int num_sending_threads = 0;
static pthread_t *sending_threads = 0;

// do a little bit of polling to try to move messages along, but return
//  to the caller rather than spinning
void do_some_polling(void)
{
  for(int i = 0; i < gasnet_nodes(); i++) {
    if(!endpoints[i]) continue; // skip our own node

    endpoints[i]->push_messages(0);
  }

  gasnet_AMPoll();
}

static void *gasnet_poll_thread_loop(void *data)
{
  // each polling thread basically does an endless loop of trying to send
  //  outgoing messages and then polling
  while(1) {
    do_some_polling();
    //usleep(10000);
  }
  return 0;
}

void start_polling_threads(int count)
{
  num_polling_threads = count;
  polling_threads = new pthread_t[count];

  for(int i = 0; i < count; i++) {
    pthread_attr_t attr;
    CHECK_PTHREAD( pthread_attr_init(&attr) );
    if(LegionRuntime::LowLevel::proc_assignment)
      LegionRuntime::LowLevel::proc_assignment->bind_thread(-1, &attr, "AM polling thread");    
    CHECK_PTHREAD( pthread_create(&polling_threads[i], 0, 
				  gasnet_poll_thread_loop, 0) );
    CHECK_PTHREAD( pthread_attr_destroy(&attr) );
  }
}

static void* sender_thread_loop(void *index)
{
  long idx = (long)index;
  while (1) {
    endpoints[idx]->push_messages(10000,true);
  }
  return 0;
}

void start_sending_threads(void)
{
  num_sending_threads = gasnet_nodes();
  sending_threads = new pthread_t[num_sending_threads];

  for (int i = 0; i < gasnet_nodes(); i++)
  {
    if (i == gasnet_mynode()) continue;
    pthread_attr_t attr;
    CHECK_PTHREAD( pthread_attr_init(&attr) );
    if(LegionRuntime::LowLevel::proc_assignment)
      LegionRuntime::LowLevel::proc_assignment->bind_thread(-1, &attr, "AM sender thread");    
    CHECK_PTHREAD( pthread_create(&sending_threads[i], 0,
                                  sender_thread_loop, (void*)i));
    CHECK_PTHREAD( pthread_attr_destroy(&attr) );
  }
}
	
void enqueue_message(gasnet_node_t target, int msgid,
		     const void *args, size_t arg_size,
		     const void *payload, size_t payload_size,
		     int payload_mode)
{
  assert(target != gasnet_mynode());

  OutgoingMessage *hdr = new OutgoingMessage(msgid, 
					     (arg_size + sizeof(int) - 1) / sizeof(int),
					     args);

  hdr->set_payload((void *)payload, payload_size, payload_mode);

  endpoints[target]->enqueue_message(hdr, true); // TODO: decide when OOO is ok?
}

void handle_long_msgptr(gasnet_node_t source, void *ptr)
{
  assert(source != gasnet_mynode());

  endpoints[source]->handle_long_msgptr(ptr);
}

extern size_t adjust_long_msgsize(gasnet_node_t source, void *ptr, size_t orig_size)
{
  assert(source != gasnet_mynode());

  return(endpoints[source]->adjust_long_msgsize(ptr, orig_size));
}
