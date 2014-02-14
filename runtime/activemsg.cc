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

gasnet_seginfo_t *segment_info = 0;

static bool is_registered(void *ptr)
{
  off_t offset = ((char *)ptr) - ((char *)(segment_info[gasnet_mynode()].addr));
  if((offset >= 0) && (offset < segment_info[gasnet_mynode()].size))
    return true;
  return false;
}

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
    free(oldptr);
  }
}

class SrcDataPool {
public:
  SrcDataPool(void *base, size_t size);
  ~SrcDataPool(void);

  // debug
  void record_srcptr(void *ptr);
  void *alloc_srcptr(size_t size_needed);
  void release_srcptr(void *srcptr);

  static void release_srcptr_handler(gasnet_token_t token, gasnet_handlerarg_t arg0, gasnet_handlerarg_t arg1);

protected:
  gasnet_hsl_t mutex;
  std::map<char *, size_t> free_list;
  std::map<char *, size_t> in_use;
  // debug
  std::map<void *, off_t> alloc_counts;
};

static SrcDataPool *srcdatapool = 0;

/*static*/ void SrcDataPool::release_srcptr_handler(gasnet_token_t token,
						    gasnet_handlerarg_t arg0,
						    gasnet_handlerarg_t arg1)
{
  uintptr_t srcptr = (((uintptr_t)(unsigned)arg1) << 32) | ((uintptr_t)(unsigned)arg0);
  assert(srcdatapool != 0);
  srcdatapool->release_srcptr((void *)srcptr);
}

// wrapper so we don't have to expose SrcDataPool implementation
void release_srcptr(void *srcptr)
{
  assert(srcdatapool != 0);
  srcdatapool->release_srcptr(srcptr);
}

SrcDataPool::SrcDataPool(void *base, size_t size)
{
  gasnet_hsl_init(&mutex);
  free_list[(char *)base] = size;
}

SrcDataPool::~SrcDataPool(void)
{
  size_t total = 0;
  size_t nonzero = 0;
  for(std::map<void *, off_t>::const_iterator it = alloc_counts.begin(); it != alloc_counts.end(); it++) {
    total++;
    if(it->second != 0) {
      printf("HELP!  srcptr %p on node %d has final count of %zd\n", it->first, gasnet_mynode(), it->second);
      nonzero++;
    }
  }
  printf("SrcDataPool:  node %d: %zd total srcptrs, %zd nonzero\n", gasnet_mynode(), total, nonzero);
}

void SrcDataPool::record_srcptr(void *srcptr)
{
#ifdef DEBUG_SRCPTRPOOL
  printf("recording srcptr = %p (node %d)\n", srcptr, gasnet_mynode());
#endif
  gasnet_hsl_lock(&mutex);
  alloc_counts[srcptr]++;
  gasnet_hsl_unlock(&mutex);
}

void *SrcDataPool::alloc_srcptr(size_t size_needed)
{
  // round up size to something reasonable
  const size_t BLOCK_SIZE = 64;
  if(size_needed % BLOCK_SIZE) {
    size_needed = BLOCK_SIZE * ((size_needed / BLOCK_SIZE) + 1);
  }

  gasnet_hsl_lock(&mutex);
  // walk the free list until we find something big enough
  std::map<char *, size_t>::iterator it = free_list.begin();
  while(it != free_list.end()) {
    if(it->second == size_needed) {
      // exact match
#ifdef DEBUG_SRCPTRPOOL
      printf("found %p + %zd - exact\n", it->first, it->second);
#endif
      char *srcptr = it->first;
      free_list.erase(it);
      in_use[srcptr] = size_needed;
      gasnet_hsl_unlock(&mutex);
      return srcptr;
    }

    if(it->second > size_needed) {
      // match with some left over
#ifdef DEBUG_SRCPTRPOOL
      printf("found %p + %zd > %zd\n", it->first, it->second, size_needed);
#endif
      char *srcptr = it->first + (it->second - size_needed);
      it->second -= size_needed;
      in_use[srcptr] = size_needed;
      gasnet_hsl_unlock(&mutex);
      return srcptr;
    }

    // not big enough - keep looking
    it++;
  }
  gasnet_hsl_unlock(&mutex);
  return 0;
}

void SrcDataPool::release_srcptr(void *srcptr)
{
  char *srcptr_c = (char *)srcptr;
#ifdef DEBUG_SRCPTRPOOL
  printf("releasing srcptr = %p (node %d)\n", srcptr, gasnet_mynode());
#endif
  gasnet_hsl_lock(&mutex);
  // look up the pointer to find its size
  std::map<char *, size_t>::iterator it = in_use.find(srcptr_c);
  assert(it != in_use.end());
  size_t size = it->second;
  in_use.erase(it);  // remove from in use list

  // we'd better not be in the free list ourselves
  assert(free_list.find(srcptr_c) == free_list.end());

  // now add to the free list
  if(free_list.size() == 0) {
    // no free entries, so just add ourselves
    free_list[srcptr_c] = size;
  } else {
    // find the ranges above and below us to see if we can merge
    std::map<char *, size_t>::iterator below = free_list.lower_bound(srcptr_c);
    if(below == free_list.end()) {
      // nothing below us
      std::map<char *, size_t>::iterator above = free_list.begin();
#ifdef DEBUG_SRCPTRPOOL
      printf("merge?  NONE %p+%zd %p+%zd\n", srcptr, size, above->first, above->second);
#endif
      if(above->first == (srcptr_c + size)) {
	// yes, remove the above entry and add its size to ours
	size += above->second;
	free_list.erase(above);
      }
      // either way, insert ourselves now
      free_list[srcptr_c] = size;
    } else {
      std::map<char *, size_t>::iterator above = below;  above++;
      if(above == free_list.end()) {
	// nothing above us
#ifdef DEBUG_SRCPTRPOOL
	printf("merge?  %p+%zd %p+%zd NONE\n", below->first, below->second, srcptr, size);
#endif
	if(srcptr_c == (below->first + below->second)) {
	  // yes, change the below entry instead of adding ourselves
	  below->second += size;
	} else {
	  free_list[srcptr_c] = size;
	}
      } else {
#ifdef DEBUG_SRCPTRPOOL
	printf("merge?  %p+%zd %p+%zd %p+%zd\n", below->first, below->second, srcptr, size, above->first, above->second);
#endif
	// first check the above
	if(above->first == (srcptr_c + size)) {
	  // yes, remove the above entry and add its size to ours
	  size += above->second;
	  free_list.erase(above);
	}
	// now check the below
	if(srcptr_c == (below->first + below->second)) {
	  // yes, change the below entry instead of adding ourselves
	  below->second += size;
	} else {
	  free_list[srcptr_c] = size;
	}
      }
    }
  }
  gasnet_hsl_unlock(&mutex);
}

struct OutgoingMessage {
  OutgoingMessage(unsigned _msgid, unsigned _num_args, const void *_args)
    : msgid(_msgid), num_args(_num_args),
      payload(0), payload_size(0), payload_mode(PAYLOAD_NONE), dstptr(0)
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

  void set_payload(void *_payload, size_t _payload_size, int _payload_mode, void *_dstptr = 0);
  void set_payload(const SpanList& spans, size_t _payload_size, int _payload_mode, void *_dstptr = 0);

  unsigned msgid;
  unsigned num_args;
  void *payload;
  size_t payload_size;
  int payload_mode;
  void *dstptr;
  int args[16];
#ifdef DEBUG_MEM_REUSE
  int payload_num;
#endif
};
    
class ActiveMessageEndpoint {
public:
  struct ChunkInfo {
  public:
    ChunkInfo(void) : base_ptr(NULL), chunks(0), total_size(0) { }
    ChunkInfo(void *base, int c, size_t size)
      : base_ptr(base), chunks(c), total_size(size) { }
  public:
    void *base_ptr;
    int chunks;
    size_t total_size;
  };
public:
  static const int NUM_LMBS = 2;
  static const size_t LMB_SIZE = (4 << 20);

  ActiveMessageEndpoint(gasnet_node_t _peer)
    : peer(_peer)
  {
    gasnet_hsl_init(&mutex);
    gasnett_cond_init(&cond);

    cur_write_lmb = 0;
    cur_write_offset = 0;
    cur_write_count = 0;

    //cur_long_ptr = 0;
    //cur_long_chunk_idx = 0;
    //cur_long_size = 0;
    next_outgoing_message_id = 0;

    for(int i = 0; i < NUM_LMBS; i++) {
      lmb_w_bases[i] = ((char *)(segment_info[peer].addr)) + (segment_info[peer].size - LMB_SIZE * (gasnet_mynode() * NUM_LMBS + i + 1));
      lmb_r_bases[i] = ((char *)(segment_info[gasnet_mynode()].addr)) + (segment_info[peer].size - LMB_SIZE * (peer * NUM_LMBS + i + 1));
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

	// do we have a known destination pointer on the target?  if so, no need to use LMB
	if(hdr->dstptr != 0) {
	  //printf("sending long message directly to %p (%zd bytes)\n", hdr->dstptr, hdr->payload_size);
	  out_long_hdrs.pop();
	  gasnet_hsl_unlock(&mutex);
	  send_long(hdr, hdr->dstptr);
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
          // keep write offset aligned to 128B
          if(cur_write_offset & 0x7f)
            cur_write_offset = ((cur_write_offset >> 7) + 1) << 7;
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

#if 0
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
#endif

  bool adjust_long_msgsize(void *&ptr, size_t &buffer_size, 
                           int message_id, int chunks)
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
      return true;
    }

    // Quick out, if there was only one chunk, then we are good to go
    if (chunks == 1)
      return true;

    bool ready = false;;
    // now we need to hold the lock
    gasnet_hsl_lock(&mutex);
    // See if we've seen this message id before
    std::map<int,ChunkInfo>::iterator finder = 
      observed_messages.find(message_id);
    if (finder == observed_messages.end())
    {
      // haven't seen it before, mark that we've seen the first chunk
      observed_messages[message_id] = ChunkInfo(ptr, 1, buffer_size);
    }
    else
    {
      // Update the pointer with the smallest one which is the base
      if (((unsigned long)(ptr)) < ((unsigned long)(finder->second.base_ptr)))
        finder->second.base_ptr = ptr;
      finder->second.total_size += buffer_size;
      finder->second.chunks++;
      // See if we've seen the last chunk
      if (finder->second.chunks == chunks)
      {
        // We've seen all the chunks, now update the pointer
        // and the buffer size and mark that we can handle the message
        ptr = finder->second.base_ptr;
        buffer_size = finder->second.total_size;
        ready = true;
        // Remove the entry from the map
        observed_messages.erase(finder);
      }
      // Otherwise we're not done yet
    }
    gasnet_hsl_unlock(&mutex);
    return ready;
  }

#if 0
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
#endif

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

    const size_t max_long_req = gasnet_AMMaxLongRequest();

    // Get a new message ID for this message
    // We know that all medium and long active messages use the
    // BaseMedium class as their base type for sending so the first
    // two fields hdr->args[0] and hdr->args[1] can be used for
    // storing the message ID and the number of chunks
    hdr->args[0] = next_outgoing_message_id++;
    int chunks = (hdr->payload_size + max_long_req - 1) / max_long_req;
    hdr->args[1] = chunks;
    if(hdr->payload_mode == PAYLOAD_SRCPTR) {
      //srcdatapool->record_srcptr(hdr->payload);
      gasnet_handlerarg_t srcptr_lo = ((uint64_t)(hdr->payload)) & 0x0FFFFFFFFULL;
      gasnet_handlerarg_t srcptr_hi = ((uint64_t)(hdr->payload)) >> 32;
      hdr->args[2] = srcptr_lo;
      hdr->args[3] = srcptr_hi;
    } else {
      hdr->args[2] = 0;
      hdr->args[3] = 0;
    }
      

#if 0
    size_t msg_size = hdr->payload_size;
    if(msg_size > max_long_req) {
      size_t chunks = (hdr->payload_size + max_long_req - 1) / max_long_req;

      gasnet_handlerarg_t dest_ptr_lo = ((uint64_t)dest_ptr) & 0x0FFFFFFFFULL;
      gasnet_handlerarg_t dest_ptr_hi = ((uint64_t)dest_ptr) >> 32;

      for(unsigned i = chunks-1; i > 0; i--) {
	size_t size = ((i == (chunks-1)) ? (hdr->payload_size % max_long_req) : max_long_req);
	gasnet_AMRequestLongAsync3(peer, MSGID_LONG_EXTENSION, 
			      ((char *)(hdr->payload))+(i * max_long_req), size, 
			      ((char *)dest_ptr)+(i * max_long_req),
			      dest_ptr_lo, dest_ptr_hi, i);
      }

      msg_size = max_long_req;
    }
#endif

    for (int i = (chunks-1); i >= 0; i--)
    {
      // every chunk but the last is the max size - the last one is whatever
      //   is left (which may also be the max size if it divided evenly)
      size_t size = ((i < (chunks - 1)) ?
                       max_long_req :
                       (hdr->payload_size - (chunks - 1) * max_long_req));
      switch(hdr->num_args) {
      case 1:
        // should never get this case since we
        // should always be sending at least two args
        assert(false);
        //gasnet_AMRequestLongAsync1(peer, hdr->msgid, 
        //                      hdr->payload, msg_size, dest_ptr,
        //                      hdr->args[0]);
        break;

      case 2:
        gasnet_AMRequestLongAsync2(peer, hdr->msgid, 
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1]);
        break;

      case 3:
        gasnet_AMRequestLongAsync3(peer, hdr->msgid, 
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2]);
        break;

      case 4:
        gasnet_AMRequestLongAsync4(peer, hdr->msgid, 
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3]);
        break;
      case 5:
        gasnet_AMRequestLongAsync5(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size,
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4]);
        break;
      case 6:
        gasnet_AMRequestLongAsync6(peer, hdr->msgid, 
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5]);
        break;
      case 7:
        gasnet_AMRequestLongAsync7(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6]);
        break;
      case 8:
        gasnet_AMRequestLongAsync8(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7]);
        break;
      case 9:
        gasnet_AMRequestLongAsync9(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8]);
        break;
      case 10:
        gasnet_AMRequestLongAsync10(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8],
                              hdr->args[9]);
        break;
      case 11:
        gasnet_AMRequestLongAsync11(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8],
                              hdr->args[9], hdr->args[10]);
        break;
      case 12:
        gasnet_AMRequestLongAsync12(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8],
                              hdr->args[9], hdr->args[10], hdr->args[11]);
        break;
      case 13:
        gasnet_AMRequestLongAsync13(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8],
                              hdr->args[9], hdr->args[10], hdr->args[11],
                              hdr->args[12]);
        break;
      case 14:
        gasnet_AMRequestLongAsync14(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8],
                              hdr->args[9], hdr->args[10], hdr->args[11],
                              hdr->args[12], hdr->args[13]);
        break;
      case 15:
        gasnet_AMRequestLongAsync15(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8],
                              hdr->args[9], hdr->args[10], hdr->args[11],
                              hdr->args[12], hdr->args[13], hdr->args[14]);
        break;
      case 16:
        gasnet_AMRequestLongAsync16(peer, hdr->msgid,
                              ((char*)hdr->payload+(i*max_long_req)), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
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
  //void *cur_long_ptr;
  //int cur_long_chunk_idx;
  //size_t cur_long_size;
  std::map<int/*message id*/,ChunkInfo> observed_messages;
  int next_outgoing_message_id;
};

void OutgoingMessage::set_payload(void *_payload, size_t _payload_size, int _payload_mode,
				  void *_dstptr)
{
  // die if a payload has already been attached
  assert(payload_mode == PAYLOAD_NONE);

  // no payload?  easy case
  if(_payload_mode == PAYLOAD_NONE)
    return;

  // payload must be non-empty, and fit in the LMB unless we have a dstptr for it
  assert(_payload_size > 0);
  assert((_dstptr != 0) || (_payload_size <= ActiveMessageEndpoint::LMB_SIZE));

  // copy the destination pointer through
  dstptr = _dstptr;

  // do we need to place this data in the srcdata pool?
  // for now, yes, unless it's KEEP and in already-registered memory
  bool need_srcdata;
  if(_payload_mode == PAYLOAD_KEEP) {
    bool is_reg = is_registered(_payload);
    //printf("KEEP payload registration: %p %d\n", _payload, is_reg ? 1 : 0);
    need_srcdata = !is_reg;
  } else {
    need_srcdata = true;
  }
  // if we don't have a srcdata pool, obviously don't try to use it
  if(!srcdatapool)
    need_srcdata = false;
  
  if(need_srcdata) {
    // try to get the needed space in the srcdata pool
    assert(srcdatapool);
    void *srcptr = srcdatapool->alloc_srcptr(_payload_size);
    if(srcptr != 0) {
      // printf("%d: copying payload to srcdatapool: %p -> %p (%zd)\n", 
      // 	     gasnet_mynode(), _payload, srcptr, _payload_size);
      memcpy(srcptr, _payload, _payload_size);

      if(_payload_mode == PAYLOAD_FREE) {
	// done with the requestor's data - free it now
	free(_payload);
      }

      payload_mode = PAYLOAD_SRCPTR;
      payload = srcptr;
      payload_size = _payload_size;
      return;
    }

    // fall through or die?
    assert(0);
  }

  // use data where it is or copy to non-registered memory
  payload_mode = _payload_mode;
  payload_size = _payload_size;
  // make a copy if we were asked to
  if(_payload_mode == PAYLOAD_COPY) {
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

static void gather_spans(const SpanList& spans, void *dst, size_t expected_size)
{
  char *dst_c = (char *)dst;
  off_t bytes_left = expected_size;
  for(SpanList::const_iterator it = spans.begin(); it != spans.end(); it++) {
    assert(it->second <= bytes_left);
    memcpy(dst_c, it->first, it->second);
    dst_c += it->second;
    bytes_left -= it->second;
  }
  assert(bytes_left == 0);
}

void OutgoingMessage::set_payload(const SpanList& spans, size_t _payload_size, int _payload_mode,
				  void *_dstptr)
{
  // die if a payload has already been attached
  assert(payload_mode == PAYLOAD_NONE);

  // no payload?  easy case
  if(_payload_mode == PAYLOAD_NONE)
    return;

  // payload must be non-empty, and fit in the LMB unless we have a dstptr for it
  assert(_payload_size > 0);
  assert((_dstptr != 0) || (_payload_size <= ActiveMessageEndpoint::LMB_SIZE));

  // copy the destination pointer through
  dstptr = _dstptr;

  // we always need to allocate some memory to gather the data for sending - prefer to get this
  //  from srcdata pool
  payload_size = _payload_size;
  if(srcdatapool) {
    payload = srcdatapool->alloc_srcptr(_payload_size);
    if(payload) {
      payload_mode = PAYLOAD_SRCPTR;
    } else {
      assert(0);
    }
  }
  if(!payload) {
    payload = malloc(_payload_size);
    assert(payload != 0);
    payload_mode = PAYLOAD_COPY;
  }

  gather_spans(spans, payload, payload_size);
  // doesn't really make sense to call this one with PAYLOAD_FREE
  assert(_payload_mode != PAYLOAD_FREE);
}

static ActiveMessageEndpoint **endpoints;

#if 0
static void handle_long_extension(gasnet_token_t token, void *buf, size_t nbytes,
				  gasnet_handlerarg_t ptr_lo, gasnet_handlerarg_t ptr_hi, gasnet_handlerarg_t chunk_idx)
{
  gasnet_node_t src;
  gasnet_AMGetMsgSource(token, &src);

  void *dest_ptr = (void *)((((uint64_t)ptr_hi) << 32) | ((uint64_t)(uint32_t)ptr_lo));
  endpoints[src]->handle_long_extension(dest_ptr, chunk_idx, nbytes);
}
#endif

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
		    int gasnet_mem_size_in_mb,
		    int registered_mem_size_in_mb)
{
  size_t srcdatapool_size = 64 << 20;
  size_t lmb_size = (gasnet_nodes() * 
		     ActiveMessageEndpoint::NUM_LMBS *
		     ActiveMessageEndpoint::LMB_SIZE);

  // add in our internal handlers and space we need for LMBs
  int attach_size = ((gasnet_mem_size_in_mb << 20) +
		     (registered_mem_size_in_mb << 20) +
		     srcdatapool_size +
		     lmb_size);

#if 0
  handlers[hcount].index = MSGID_LONG_EXTENSION;
  handlers[hcount].fnptr = (void (*)())handle_long_extension;
  hcount++;
#endif
  handlers[hcount].index = MSGID_FLIP_REQ;
  handlers[hcount].fnptr = (void (*)())handle_flip_req;
  hcount++;
  handlers[hcount].index = MSGID_FLIP_ACK;
  handlers[hcount].fnptr = (void (*)())handle_flip_ack;
  hcount++;
  handlers[hcount].index = MSGID_RELEASE_SRCPTR;
  handlers[hcount].fnptr = (void (*)())SrcDataPool::release_srcptr_handler;
  hcount++;

  CHECK_GASNET( gasnet_attach(handlers, hcount,
			      attach_size, 0) );

  segment_info = new gasnet_seginfo_t[gasnet_nodes()];
  CHECK_GASNET( gasnet_getSegmentInfo(segment_info, gasnet_nodes()) );

  char *my_segment = (char *)(segment_info[gasnet_mynode()].addr);
  char *gasnet_mem_base = my_segment;  my_segment += (gasnet_mem_size_in_mb << 20);
  char *reg_mem_base = my_segment;  my_segment += (registered_mem_size_in_mb << 20);
  char *srcdatapool_base = my_segment;  my_segment += srcdatapool_size;
  char *lmb_base = my_segment;  my_segment += lmb_size;
  assert(my_segment <= ((char *)(segment_info[gasnet_mynode()].addr) + segment_info[gasnet_mynode()].size));

#ifndef NO_SRCDATAPOOL
  srcdatapool = new SrcDataPool(srcdatapool_base, srcdatapool_size);
#endif

  endpoints = new ActiveMessageEndpoint *[gasnet_nodes()];

  for(int i = 0; i < gasnet_nodes(); i++)
    if(i == gasnet_mynode())
      endpoints[i] = 0;
    else
      endpoints[i] = new ActiveMessageEndpoint(i);

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
                                  sender_thread_loop, (void*)long(i)));
    CHECK_PTHREAD( pthread_attr_destroy(&attr) );
  }
}
	
void enqueue_message(gasnet_node_t target, int msgid,
		     const void *args, size_t arg_size,
		     const void *payload, size_t payload_size,
		     int payload_mode, void *dstptr)
{
  assert(target != gasnet_mynode());

  OutgoingMessage *hdr = new OutgoingMessage(msgid, 
					     (arg_size + sizeof(int) - 1) / sizeof(int),
					     args);

  hdr->set_payload((void *)payload, payload_size, payload_mode, dstptr);

  endpoints[target]->enqueue_message(hdr, true); // TODO: decide when OOO is ok?
}

void enqueue_message(gasnet_node_t target, int msgid,
		     const void *args, size_t arg_size,
		     const SpanList& spans, size_t payload_size,
		     int payload_mode, void *dstptr)
{
  assert(target != gasnet_mynode());

  OutgoingMessage *hdr = new OutgoingMessage(msgid, 
  					     (arg_size + sizeof(int) - 1) / sizeof(int),
  					     args);

  hdr->set_payload(spans, payload_size, payload_mode, dstptr);

  endpoints[target]->enqueue_message(hdr, true); // TODO: decide when OOO is ok?
}

void handle_long_msgptr(gasnet_node_t source, void *ptr)
{
  assert(source != gasnet_mynode());

  endpoints[source]->handle_long_msgptr(ptr);
}

#if 0
extern size_t adjust_long_msgsize(gasnet_node_t source, void *ptr, size_t orig_size)
{
  assert(source != gasnet_mynode());

  return(endpoints[source]->adjust_long_msgsize(ptr, orig_size));
}
#endif

extern bool adjust_long_msgsize(gasnet_node_t source, void *&ptr, size_t &buffer_size,
                                const void *args, size_t arglen)
{
  assert(source != gasnet_mynode());
  assert(arglen >= 2*sizeof(int));
  const int *arg_ptr = (const int*)args;

  return (endpoints[source]->adjust_long_msgsize(ptr, buffer_size,
                                                 arg_ptr[0], arg_ptr[1]));
}
