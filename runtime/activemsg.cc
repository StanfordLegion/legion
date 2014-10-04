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

#include "activemsg.h"
#include "utilities.h"

#ifndef __GNUC__
#include "atomics.h" // for __sync_add_and_fetch
#endif

#include <queue>
#include <cassert>

#include "lowlevel_impl.h"

#define NO_DEBUG_AMREQUESTS

enum { MSGID_LONG_EXTENSION = 253,
       MSGID_FLIP_REQ = 254,
       MSGID_FLIP_ACK = 255 };

#ifdef DEBUG_MEM_REUSE
static int payload_count = 0;
#endif

static const int DEFERRED_FREE_COUNT = 128;
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

class PayloadSource {
public:
  PayloadSource(void) { }
  virtual ~PayloadSource(void) { }
public:
  virtual void copy_data(void *dest) = 0;
  virtual void *get_contig_pointer(void) { return 0; }
};

class ContiguousPayload : public PayloadSource {
public:
  ContiguousPayload(void *_srcptr, size_t _size, int _mode);
  virtual ~ContiguousPayload(void) { }
  virtual void copy_data(void *dest);
  virtual void *get_contig_pointer(void) { return srcptr; }
protected:
  void *srcptr;
  size_t size;
  int mode;
};

class TwoDPayload : public PayloadSource {
public:
  TwoDPayload(const void *_srcptr, size_t _line_size, size_t _line_count,
	      ptrdiff_t _line_stride, int _mode);
  virtual ~TwoDPayload(void) { }
  virtual void copy_data(void *dest);
protected:
  const void *srcptr;
  size_t line_size, line_count;
  ptrdiff_t line_stride;
  int mode;
};

class SpanPayload : public PayloadSource {
public:
  SpanPayload(const SpanList& _spans, size_t _size, int _mode);
  virtual ~SpanPayload(void) { }
  virtual void copy_data(void *dest);
protected:
  SpanList spans;
  size_t size;
  int mode;
};

struct OutgoingMessage {
  OutgoingMessage(unsigned _msgid, unsigned _num_args, const void *_args);
  ~OutgoingMessage(void);

  void set_payload(PayloadSource *_payload, size_t _payload_size,
		   int _payload_mode, void *_dstptr = 0);
  void reserve_srcdata(void);
#if 0
  void set_payload(void *_payload, size_t _payload_size,
		   int _payload_mode, void *_dstptr = 0);
  void set_payload(void *_payload, size_t _line_size,
		   off_t _line_stride, size_t _line_count,
		   int _payload_mode, void *_dstptr = 0);
  void set_payload(const SpanList& spans, size_t _payload_size,
		   int _payload_mode, void *_dstptr = 0);
#endif

  void assign_srcdata_pointer(void *ptr);

  unsigned msgid;
  unsigned num_args;
  void *payload;
  size_t payload_size;
  int payload_mode;
  void *dstptr;
  PayloadSource *payload_src;
  int args[16];
#ifdef DEBUG_MEM_REUSE
  int payload_num;
#endif
};

LegionRuntime::Logger::Category log_sdp("srcdatapool");

class SrcDataPool {
public:
  SrcDataPool(void *base, size_t size);
  ~SrcDataPool(void);

  class Lock {
  public:
    Lock(SrcDataPool& _sdp) : sdp(_sdp) { gasnet_hsl_lock(&sdp.mutex); }
    ~Lock(void) { gasnet_hsl_unlock(&sdp.mutex); }
  protected:
    SrcDataPool& sdp;
  };

  // allocators must already hold the lock - prove it by passing a reference
  void *alloc_srcptr(size_t size_needed, Lock& held_lock);

  // enqueuing a pending message must also hold the lock
  void add_pending(OutgoingMessage *msg, Lock& held_lock);

  // releasing memory will take the lock itself
  void release_srcptr(void *srcptr);

  static void release_srcptr_handler(gasnet_token_t token, gasnet_handlerarg_t arg0, gasnet_handlerarg_t arg1);

protected:
  size_t round_up_size(size_t size);

  friend class SrcDataPool::Lock;
  gasnet_hsl_t mutex;
  size_t total_size;
  std::map<char *, size_t> free_list;
  std::queue<OutgoingMessage *> pending_allocations;
  // debug
  std::map<char *, size_t> in_use;
  std::map<void *, off_t> alloc_counts;
};

static SrcDataPool *srcdatapool = 0;

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
  total_size = size;
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

size_t SrcDataPool::round_up_size(size_t size)
{
  const size_t BLOCK_SIZE = 64;
  size_t remainder = size % BLOCK_SIZE;
  if(remainder)
    return size + (BLOCK_SIZE - remainder);
  else
    return size;
}

void *SrcDataPool::alloc_srcptr(size_t size_needed, Lock& held_lock)
{
  // sanity check - if the requested size is larger than will ever fit, fail
  if(size_needed > total_size)
    return 0;

  // early out - if our pending allocation queue is non-empty, they're
  //  first in line, so fail this allocation
  if(!pending_allocations.empty())
    return 0;

  // round up size to something reasonable
  size_needed = round_up_size(size_needed);

  // walk the free list until we find something big enough
  // only use the a bigger chunk if we absolutely have to 
  // in order to avoid segmentation problems.
  std::map<char *, size_t>::iterator it = free_list.begin();
  char *smallest_upper_bound = 0;
  size_t smallest_upper_size = 0;
  while(it != free_list.end()) {
    if(it->second == size_needed) {
      // exact match
      log_sdp.debug("found %p + %zd - exact", it->first, it->second);

      char *srcptr = it->first;
      free_list.erase(it);
      in_use[srcptr] = size_needed;

      return srcptr;
    }

    if(it->second > size_needed) {
      // match with some left over
      // Check to see if it is smaller
      // than the largest upper bound
      if (smallest_upper_bound == 0) {
        smallest_upper_bound = it->first;
        smallest_upper_size = it->second;
      } else if (it->second < smallest_upper_size) {
        smallest_upper_bound = it->first;
        smallest_upper_size = it->second;
      }
    }

    // not big enough - keep looking
    it++;
  }
  if (smallest_upper_bound != 0) {
    it = free_list.find(smallest_upper_bound);
    char *srcptr = it->first + (it->second - size_needed);
    it->second -= size_needed;
    in_use[srcptr] = size_needed;

    log_sdp.debug("found %p + %zd > %zd", it->first, it->second, size_needed);

    return srcptr;
  }

  // allocation failed - let caller decide what to do (probably add it as a
  //   pending allocation after maybe moving data)
  return 0;
}

void SrcDataPool::add_pending(OutgoingMessage *msg, Lock& held_lock)
{
  // simple - just add to our queue
  log_sdp.debug("pending allocation: %zd for %p", msg->payload_size, msg);

  // sanity check - if the requested size is larger than will ever fit, 
  //  we're just dead
  if(msg->payload_size > total_size) {
    log_sdp.error("allocation of %zd can never be satisfied! (max = %zd)",
		  msg->payload_size, total_size);
    assert(0);
  }

  pending_allocations.push(msg);
}

void SrcDataPool::release_srcptr(void *srcptr)
{
  char *srcptr_c = (char *)srcptr;

  log_sdp.debug("releasing srcptr = %p", srcptr);

  // releasing a srcptr span may result in some pending allocations being
  //   satisfied - keep a list so their actual copies can happen without
  //   holding the SDP lock
  std::vector<std::pair<OutgoingMessage *, void *> > satisfied;
  {
    Lock held_lock(*this);

    // look up the pointer to find its size
    std::map<char *, size_t>::iterator it = in_use.find(srcptr_c);
    assert(it != in_use.end());
    size_t size = it->second;
    in_use.erase(it);  // remove from in use list

    // we'd better not be in the free list ourselves
    assert(free_list.find(srcptr_c) == free_list.end());

    // see if we can absorb any adjacent ranges
    if(!free_list.empty()) {
      std::map<char *, size_t>::iterator above = free_list.lower_bound(srcptr_c);
      
      // look below first
      while(above != free_list.begin()) {
	std::map<char *, size_t>::iterator below = above;  below--;
	
	log_sdp.spew("merge?  %p+%zd %p+%zd NONE", below->first, below->second, srcptr_c, size);

	if((below->first + below->second) != srcptr_c)
	  break;

	srcptr_c = below->first;
	size += below->second;
	free_list.erase(below);
      }

      // now look above
      while(above != free_list.end()) {
	log_sdp.spew("merge?  NONE %p+%zd %p+%zd", srcptr_c, size, above->first, above->second);

	if((srcptr_c + size) != above->first)
	  break;

	size += above->second;
	std::map<char *, size_t>::iterator to_nuke(above++);
	free_list.erase(to_nuke);
      }
    }

    // is this possibly-merged span large enough to satisfy the first pending
    //  allocation (if any)?
    if(!pending_allocations.empty() && 
       (size >= pending_allocations.front()->payload_size)) {
      OutgoingMessage *msg = pending_allocations.front();
      pending_allocations.pop();
      size_t act_size = round_up_size(msg->payload_size);
      in_use[srcptr_c] = act_size;
      satisfied.push_back(std::make_pair(msg, srcptr_c));

      // was anything left?  if so, add it to the list of free spans
      if(size > act_size)
	free_list[srcptr_c + act_size] = size - act_size;

      // now see if we can satisfy any other pending allocations - use the
      //  normal allocator routine here because there might be better choices
      //  to use than the span we just freed (assuming any of it is left)
      while(!pending_allocations.empty()) {
	OutgoingMessage *msg = pending_allocations.front();
	void *ptr = alloc_srcptr(msg->payload_size, held_lock);
	if(!ptr) break;

	satisfied.push_back(std::make_pair(msg, ptr));
	pending_allocations.pop();
      }
    } else {
      // no?  then no other span will either, so just add this to the free list
      //  and return
      free_list[srcptr_c] = size;
    }
  }

  // with the lock released, tell any messages that got srcptr's so they can
  //   do their copies
  if(!satisfied.empty())
    for(std::vector<std::pair<OutgoingMessage *, void *> >::iterator it = satisfied.begin();
	it != satisfied.end();
	it++) {
      log_sdp.debug("satisfying pending allocation: %p for %p",
		    it->second, it->first);
      it->first->assign_srcdata_pointer(it->second);
    }
}

OutgoingMessage::OutgoingMessage(unsigned _msgid, unsigned _num_args,
				 const void *_args)
  : msgid(_msgid), num_args(_num_args),
    payload(0), payload_size(0), payload_mode(PAYLOAD_NONE), dstptr(0),
    payload_src(0)
{
  for(unsigned i = 0; i < _num_args; i++)
    args[i] = ((const int *)_args)[i];
}
    
OutgoingMessage::~OutgoingMessage(void)
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
  if (payload_src != 0) {
    assert(payload_mode == PAYLOAD_KEEPREG);
    delete payload_src;
    payload_src = 0;
  }
}

// these values can be overridden by command-line parameters
static int num_lmbs = 2;
static size_t lmb_size = 1 << 20; // 1 MB
static bool force_long_messages = true;

// returns the largest payload that can be sent to a node (to a non-pinned
//   address)
size_t get_lmb_size(int target_node)
{
  // not node specific right yet
  return lmb_size;
}

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

    lmb_w_bases = new char *[num_lmbs];
    lmb_r_bases = new char *[num_lmbs];
    lmb_r_counts = new int[num_lmbs];
    lmb_w_avail = new bool[num_lmbs];

    for(int i = 0; i < num_lmbs; i++) {
      lmb_w_bases[i] = ((char *)(segment_info[peer].addr)) + (segment_info[peer].size - lmb_size * (gasnet_mynode() * num_lmbs + i + 1));
      lmb_r_bases[i] = ((char *)(segment_info[gasnet_mynode()].addr)) + (segment_info[peer].size - lmb_size * (peer * num_lmbs + i + 1));
      lmb_r_counts[i] = 0;
      lmb_w_avail[i] = true;
    }
#ifdef TRACE_MESSAGES
    sent_messages = 0;
    received_messages = 0;
#endif
  }

  ~ActiveMessageEndpoint(void)
  {
    delete[] lmb_w_bases;
    delete[] lmb_r_bases;
    delete[] lmb_r_counts;
    delete[] lmb_w_avail;
  }

  void record_message(bool sent_reply) 
  {
#ifdef TRACE_MESSAGES
    __sync_fetch_and_add(&received_messages, 1);
    if (sent_reply)
      __sync_fetch_and_add(&sent_messages, 1);
#endif
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

	// is the message still waiting on space in the srcdatapool?
	if(hdr->payload_mode == PAYLOAD_PENDING) {
	  gasnet_hsl_unlock(&mutex);
	  break;
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
	assert(hdr->payload_size <= lmb_size);
	if((cur_write_offset + hdr->payload_size) <= lmb_size) {
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
	  cur_write_lmb = (cur_write_lmb + 1) % num_lmbs;
	  cur_write_offset = 0;
	  cur_write_count = 0;

	  // now let go of the lock and send the flip request
	  gasnet_hsl_unlock(&mutex);

#ifdef DEBUG_LMB
	  printf("LMB: flipping buffer %d for %d->%d, [%p,%p), count=%d\n",
		 flip_buffer, gasnet_mynode(), peer, lmb_w_bases[flip_buffer],
		 lmb_w_bases[flip_buffer]+lmb_size, flip_count);
#endif

	  CHECK_GASNET( gasnet_AMRequestShort2(peer, MSGID_FLIP_REQ,
                                               flip_buffer, flip_count) );
#ifdef TRACE_MESSAGES
          __sync_fetch_and_add(&sent_messages, 1);
#endif

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

    // once we have the lock, we can safely move the message's payload to
    //  srcdatapool
    hdr->reserve_srcdata();

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

  bool handle_long_msgptr(void *ptr)
  {
    // can figure out which buffer it is without holding lock
    int r_buffer = -1;
    for(int i = 0; i < num_lmbs; i++)
      if((ptr >= lmb_r_bases[i]) && (ptr < (lmb_r_bases[i] + lmb_size))) {
      r_buffer = i;
      break;
    }
    if(r_buffer < 0) {
      // probably a medium message?
      return false;
    }
    //assert(r_buffer >= 0);

#ifdef DEBUG_LMB
    printf("LMB: received %p for %d->%d in buffer %d, [%p, %p)\n",
	   ptr, peer, gasnet_mynode(), r_buffer, lmb_r_bases[r_buffer],
	   lmb_r_bases[r_buffer] + lmb_size);
#endif

    // now take the lock to increment the r_count and decide if we need
    //  to ack (can't actually send it here, so queue it up)
    bool message_added = false;
    gasnet_hsl_lock(&mutex);
    lmb_r_counts[r_buffer]++;
    if(lmb_r_counts[r_buffer] == 0) {
#ifdef DEBUG_LMB
      printf("LMB: acking flip of buffer %d for %d->%d, [%p,%p)\n",
	     r_buffer, peer, gasnet_mynode(), lmb_r_bases[r_buffer],
	     lmb_r_bases[r_buffer]+lmb_size);
#endif

      OutgoingMessage *hdr = new OutgoingMessage(MSGID_FLIP_ACK, 1, &r_buffer);
      out_short_hdrs.push(hdr);
      message_added = true;
      // wake up a sender
      gasnett_cond_signal(&cond);
    }
    gasnet_hsl_unlock(&mutex);
    return message_added;
  }

  bool adjust_long_msgsize(void *&ptr, size_t &buffer_size, 
                           int message_id, int chunks)
  {
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

  // called when the remote side tells us that there will be no more
  //  messages sent for a given buffer - as soon as we've received them all,
  //  we can ack
  bool handle_flip_request(int buffer, int count)
  {
#ifdef DEBUG_LMB
    printf("LMB: received flip of buffer %d for %d->%d, [%p,%p), count=%d\n",
	   buffer, peer, gasnet_mynode(), lmb_r_bases[buffer],
	   lmb_r_bases[buffer]+lmb_size, count);
#endif
#ifdef TRACE_MESSAGES
    __sync_fetch_and_add(&received_messages, 1);
#endif
    bool message_added = false;
    gasnet_hsl_lock(&mutex);
    lmb_r_counts[buffer] -= count;
    if(lmb_r_counts[buffer] == 0) {
#ifdef DEBUG_LMB
      printf("LMB: acking flip of buffer %d for %d->%d, [%p,%p)\n",
	     buffer, peer, gasnet_mynode(), lmb_r_bases[buffer],
	     lmb_r_bases[buffer]+lmb_size);
#endif

      OutgoingMessage *hdr = new OutgoingMessage(MSGID_FLIP_ACK, 1, &buffer);
      out_short_hdrs.push(hdr);
      message_added = true;
      // Wake up a sender
      gasnett_cond_signal(&cond);
    }
    gasnet_hsl_unlock(&mutex);
    return message_added;
  }

  // called when the remote side says it has received all the messages in a
  //  given buffer - we can that mark that write buffer as available again
  //  (don't even need to take the mutex!)
  void handle_flip_ack(int buffer)
  {
#ifdef DEBUG_LMB
    printf("LMB: received flip ack of buffer %d for %d->%d, [%p,%p)\n",
	   buffer, gasnet_mynode(), peer, lmb_w_bases[buffer],
	   lmb_w_bases[buffer]+lmb_size);
#endif
#ifdef TRACE_MESSAGES
    __sync_fetch_and_add(&received_messages, 1);
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
#ifdef DEBUG_AMREQUESTS
    printf("%d->%d: %s %d %d %p %zd / %x %x %x %x / %x %x %x %x / %x %x %x %x / %x %x %x %x\n",
	   gasnet_mynode(), peer, 
	   ((hdr->payload_mode == PAYLOAD_NONE) ? "SHORT" : "MEDIUM"),
	   hdr->num_args, hdr->msgid,
	   hdr->payload, hdr->payload_size,
	   hdr->args[0], hdr->args[1], hdr->args[2],
	   hdr->args[3], hdr->args[4], hdr->args[5],
	   hdr->args[6], hdr->args[7], hdr->args[8],
	   hdr->args[9], hdr->args[10], hdr->args[11],
	   hdr->args[12], hdr->args[13], hdr->args[14], hdr->args[15]);
    fflush(stdout);
#endif
#ifdef TRACE_MESSAGES
    __sync_fetch_and_add(&sent_messages, 1);
#endif
    switch(hdr->num_args) {
    case 1:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium1(peer, hdr->msgid, hdr->payload, 
                                              hdr->payload_size, hdr->args[0]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort1(peer, hdr->msgid, hdr->args[0]) );
      }
      break;

    case 2:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium2(peer, hdr->msgid, hdr->payload, hdr->payload_size,
                                              hdr->args[0], hdr->args[1]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort2(peer, hdr->msgid, hdr->args[0], hdr->args[1]) );
      }
      break;

    case 3:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium3(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort3(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2]) );
      }
      break;

    case 4:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium4(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort4(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2],
			       hdr->args[3]) );
      }
      break;

    case 5:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium5(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort5(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2],
			       hdr->args[3], hdr->args[4]) );
      }
      break;

    case 6:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium6(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort6(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2],
			       hdr->args[3], hdr->args[4], hdr->args[5]) );
      }
      break;

    case 8:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium8(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort8(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2],
			       hdr->args[3], hdr->args[4], hdr->args[5],
			       hdr->args[6], hdr->args[7]) );
      }
      break;

    case 10:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium10(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				 hdr->args[0], hdr->args[1], hdr->args[2],
				 hdr->args[3], hdr->args[4], hdr->args[5],
				 hdr->args[6], hdr->args[7], hdr->args[8],
				 hdr->args[9]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort10(peer, hdr->msgid,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7], hdr->args[8],
				hdr->args[9]) );
      }
      break;

    case 12:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium12(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				 hdr->args[0], hdr->args[1], hdr->args[2],
				 hdr->args[3], hdr->args[4], hdr->args[5],
				 hdr->args[6], hdr->args[7], hdr->args[8],
				 hdr->args[9], hdr->args[10], hdr->args[11]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort12(peer, hdr->msgid,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7], hdr->args[8],
				hdr->args[9], hdr->args[10], hdr->args[11]) );
      }
      break;

    case 16:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium16(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				 hdr->args[0], hdr->args[1], hdr->args[2],
				 hdr->args[3], hdr->args[4], hdr->args[5],
				 hdr->args[6], hdr->args[7], hdr->args[8],
				 hdr->args[9], hdr->args[10], hdr->args[11],
				 hdr->args[12], hdr->args[13], hdr->args[14],
				 hdr->args[15]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort16(peer, hdr->msgid,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7], hdr->args[8],
				hdr->args[9], hdr->args[10], hdr->args[11],
				hdr->args[12], hdr->args[13], hdr->args[14],
				hdr->args[15]) );
      }
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
    int message_id_start;
    if(hdr->args[0] == BaseMedium::MESSAGE_ID_MAGIC) {
      assert(hdr->args[1] == BaseMedium::MESSAGE_CHUNKS_MAGIC);
      message_id_start = 0;
      //printf("CASE 1\n");
    } else {
      assert(hdr->args[2] == BaseMedium::MESSAGE_ID_MAGIC);
      assert(hdr->args[3] == BaseMedium::MESSAGE_CHUNKS_MAGIC);
      message_id_start = 2;
      //printf("CASE 2\n");
    }
    hdr->args[message_id_start] = next_outgoing_message_id++;
    int chunks = (hdr->payload_size + max_long_req - 1) / max_long_req;
    hdr->args[message_id_start + 1] = chunks;
    if(hdr->payload_mode == PAYLOAD_SRCPTR) {
      //srcdatapool->record_srcptr(hdr->payload);
      gasnet_handlerarg_t srcptr_lo = ((uint64_t)(hdr->payload)) & 0x0FFFFFFFFULL;
      gasnet_handlerarg_t srcptr_hi = ((uint64_t)(hdr->payload)) >> 32;
      hdr->args[message_id_start + 2] = srcptr_lo;
      hdr->args[message_id_start + 3] = srcptr_hi;
    } else {
      hdr->args[message_id_start + 2] = 0;
      hdr->args[message_id_start + 3] = 0;
    }
      
    for (int i = (chunks-1); i >= 0; i--)
    {
      // every chunk but the last is the max size - the last one is whatever
      //   is left (which may also be the max size if it divided evenly)
      size_t size = ((i < (chunks - 1)) ?
                       max_long_req :
                       (hdr->payload_size - (chunks - 1) * max_long_req));
#ifdef DEBUG_AMREQUESTS
      printf("%d->%d: LONG %d %d %p %zd %p / %x %x %x %x / %x %x %x %x / %x %x %x %x / %x %x %x %x\n",
	     gasnet_mynode(), peer, hdr->num_args, hdr->msgid,
	     ((char*)hdr->payload)+(i*max_long_req), size, 
	     ((char*)dest_ptr)+(i*max_long_req),
	     hdr->args[0], hdr->args[1], hdr->args[2],
	     hdr->args[3], hdr->args[4], hdr->args[5],
	     hdr->args[6], hdr->args[7], hdr->args[8],
	     hdr->args[9], hdr->args[10], hdr->args[11],
	     hdr->args[12], hdr->args[13], hdr->args[14], hdr->args[15]);
      fflush(stdout);
#endif
#ifdef TRACE_MESSAGES
      __sync_fetch_and_add(&sent_messages, 1);
#endif
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
        CHECK_GASNET( gasnet_AMRequestLongAsync2(peer, hdr->msgid, 
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1]) );
        break;

      case 3:
        CHECK_GASNET( gasnet_AMRequestLongAsync3(peer, hdr->msgid, 
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2]) );
        break;

      case 4:
        CHECK_GASNET( gasnet_AMRequestLongAsync4(peer, hdr->msgid, 
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3]) );
        break;
      case 5:
        CHECK_GASNET (gasnet_AMRequestLongAsync5(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size,
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4]) );
        break;
      case 6:
        CHECK_GASNET( gasnet_AMRequestLongAsync6(peer, hdr->msgid, 
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5]) );
        break;
      case 7:
        CHECK_GASNET( gasnet_AMRequestLongAsync7(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6]) );
        break;
      case 8:
        CHECK_GASNET( gasnet_AMRequestLongAsync8(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7]) );
        break;
      case 9:
        CHECK_GASNET( gasnet_AMRequestLongAsync9(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8]) );
        break;
      case 10:
        CHECK_GASNET( gasnet_AMRequestLongAsync10(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8],
                              hdr->args[9]) );
        break;
      case 11:
        CHECK_GASNET( gasnet_AMRequestLongAsync11(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8],
                              hdr->args[9], hdr->args[10]) );
        break;
      case 12:
        CHECK_GASNET( gasnet_AMRequestLongAsync12(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8],
                              hdr->args[9], hdr->args[10], hdr->args[11]) );
        break;
      case 13:
        CHECK_GASNET( gasnet_AMRequestLongAsync13(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8],
                              hdr->args[9], hdr->args[10], hdr->args[11],
                              hdr->args[12]) );
        break;
      case 14:
        CHECK_GASNET( gasnet_AMRequestLongAsync14(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8],
                              hdr->args[9], hdr->args[10], hdr->args[11],
                              hdr->args[12], hdr->args[13]) );
        break;
      case 15:
        CHECK_GASNET( gasnet_AMRequestLongAsync15(peer, hdr->msgid,
                              ((char*)hdr->payload)+(i*max_long_req), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8],
                              hdr->args[9], hdr->args[10], hdr->args[11],
                              hdr->args[12], hdr->args[13], hdr->args[14]) );
        break;
      case 16:
        CHECK_GASNET( gasnet_AMRequestLongAsync16(peer, hdr->msgid,
                              ((char*)hdr->payload+(i*max_long_req)), size, 
                              ((char*)dest_ptr)+(i*max_long_req),
                              hdr->args[0], hdr->args[1], hdr->args[2],
                              hdr->args[3], hdr->args[4], hdr->args[5],
                              hdr->args[6], hdr->args[7], hdr->args[8],
                              hdr->args[9], hdr->args[10], hdr->args[11],
                              hdr->args[12], hdr->args[13], hdr->args[14],
                              hdr->args[15]) );
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
public:
  std::queue<OutgoingMessage *> out_short_hdrs;
  std::queue<OutgoingMessage *> out_long_hdrs;

  int cur_write_lmb, cur_write_count;
  size_t cur_write_offset;
  char **lmb_w_bases; // [num_lmbs]
  char **lmb_r_bases; // [num_lmbs]
  int *lmb_r_counts; // [num_lmbs]
  bool *lmb_w_avail; // [num_lmbs]
  //void *cur_long_ptr;
  //int cur_long_chunk_idx;
  //size_t cur_long_size;
  std::map<int/*message id*/,ChunkInfo> observed_messages;
  int next_outgoing_message_id;
#ifdef TRACE_MESSAGES
  int sent_messages;
  int received_messages;
#endif
};

void OutgoingMessage::set_payload(PayloadSource *_payload_src,
				  size_t _payload_size, int _payload_mode,
				  void *_dstptr)
{
  // die if a payload has already been attached
  assert(payload_mode == PAYLOAD_NONE);

  // no payload?  easy case
  if(_payload_mode == PAYLOAD_NONE)
    return;

  // payload must be non-empty, and fit in the LMB unless we have a dstptr for it
  log_sdp.info("setting payload (%zd, %d)", _payload_size, _payload_mode);
  assert(_payload_size > 0);
  assert((_dstptr != 0) || (_payload_size <= lmb_size));

  // just copy down everything - we won't attempt to grab a srcptr until
  //  we have reserved our spot in an outgoing queue
  dstptr = _dstptr;
  payload = 0;
  payload_size = _payload_size;
  payload_mode = _payload_mode;
  payload_src = _payload_src;
}

// called once we have reserved a space in a given outgoing queue so that
//  we can't get put behind somebody who failed a srcptr allocation (and might
//  need us to go out the door to remove the blockage)
void OutgoingMessage::reserve_srcdata(void)
{
  // no payload case is easy
  if(payload_mode == PAYLOAD_NONE) return;

  // if the payload is stable and in registered memory AND contiguous, we can
  //  just use it
  if((payload_mode == PAYLOAD_KEEPREG) && payload_src->get_contig_pointer()) {
    payload = payload_src->get_contig_pointer();
    return;
  }

  // do we need to place this data in the srcdata pool?
  // for now, yes, unless we don't have a srcdata pool at all
  bool need_srcdata = (srcdatapool != 0);
  
  if(need_srcdata) {
    // try to get the needed space in the srcdata pool
    assert(srcdatapool);

    void *srcptr = 0;
    {
      // take the SDP lock
      SrcDataPool::Lock held_lock(*srcdatapool);

      srcptr = srcdatapool->alloc_srcptr(payload_size, held_lock);
      log_sdp.info("got %p (%d)", srcptr, payload_mode);

      if(srcptr != 0) {
	// allocation succeeded - update state, but do copy below, after
	//  we've released the lock
	payload_mode = PAYLOAD_SRCPTR;
	payload = srcptr;
      } else {
	// if the allocation fails, we have to queue ourselves up

	// if we've been instructed to copy the data, that has to happen now
	if(payload_mode == PAYLOAD_COPY) {
	  void *copy_ptr = malloc(payload_size);
	  assert(copy_ptr != 0);
	  payload_src->copy_data(copy_ptr);
	  delete payload_src;
	  payload_src = new ContiguousPayload(copy_ptr, payload_size,
					      PAYLOAD_FREE);
	}

	payload_mode = PAYLOAD_PENDING;
	srcdatapool->add_pending(this, held_lock);
      }
    }

    // do the copy now if the allocation succeeded
    if(srcptr != 0) {
      payload_src->copy_data(srcptr);
      delete payload_src;
      payload_src = 0;
    }
  } else {
    // no srcdatapool needed, but might still have to copy
    if(payload_src->get_contig_pointer() &&
       (payload_mode != PAYLOAD_COPY)) {
      payload = payload_src->get_contig_pointer();
      delete payload_src;
      payload_src = 0;
    } else {
      // make a copy
      payload = malloc(payload_size);
      assert(payload != 0);
      payload_src->copy_data(payload);
      delete payload_src;
      payload_src = 0;
      payload_mode = PAYLOAD_FREE;
    }
  }
}

void OutgoingMessage::assign_srcdata_pointer(void *ptr)
{
  assert(payload_mode == PAYLOAD_PENDING);
  assert(payload_src != 0);
  payload_src->copy_data(ptr);
  delete payload_src;
  payload_src = 0;

  payload = ptr;
  payload_mode = PAYLOAD_SRCPTR;
}

class EndpointManager {
public:
  EndpointManager(int num_endpoints)
    : total_endpoints(num_endpoints)
  {
    endpoints = new ActiveMessageEndpoint*[num_endpoints];
    outstanding_messages = (int*)malloc(num_endpoints*sizeof(int));
    for (int i = 0; i < num_endpoints; i++)
    {
      if (i == gasnet_mynode())
        endpoints[i] = 0;
      else
        endpoints[i] = new ActiveMessageEndpoint(i);
      outstanding_messages[i] = 0;
    }
  }
public:
  void handle_flip_request(gasnet_node_t src, int flip_buffer, int flip_count)
  {
#define TRACK_MESSAGES
#ifdef TRACK_MESSAGES
    bool added_message = 
#endif
      endpoints[src]->handle_flip_request(flip_buffer, flip_count);
#ifdef TRACK_MESSAGES
    if (added_message)
      __sync_fetch_and_add(outstanding_messages+src,1);
#endif
  }
  void handle_flip_ack(gasnet_node_t src, int ack_buffer)
  {
    endpoints[src]->handle_flip_ack(ack_buffer);
  }
  void push_messages(int max_to_send = 0, bool wait = false)
  {
#ifdef TRACK_MESSAGES
    if (wait)
    {
      // If we have to wait, do the normal thing and iterate
      // over all the end points, and update message counts
      for (int i = 0; i < total_endpoints; i++)
      {
        int pushed = endpoints[i]->push_messages(max_to_send, true); 
        __sync_fetch_and_add(outstanding_messages+i, -pushed);
      }
    }
    else
    {
      for (int i = 0; i < total_endpoints; i++)
      {
        int messages = *((volatile int*)outstanding_messages+i);
        if (messages == 0) continue;
        int pushed = endpoints[i]->push_messages(max_to_send, false);
        __sync_fetch_and_add(outstanding_messages+i, -pushed);
      }
    }
#else
    for (int i = 0; i < total_endpoints; i++)
    {
      if (endpoints[i] == 0) continue;
      endpoints[i]->push_messages(max_to_send, wait);
    }
#endif
  }
  void enqueue_message(gasnet_node_t target, OutgoingMessage *hdr, bool in_order)
  {
#ifdef TRACK_MESSAGES
    __sync_fetch_and_add(outstanding_messages+target,1);
#endif
    endpoints[target]->enqueue_message(hdr, in_order);
  }
  void handle_long_msgptr(gasnet_node_t source, void *ptr)
  {
#ifdef TRACK_MESSAGES
    bool message_added =
#endif
      endpoints[source]->handle_long_msgptr(ptr);
#ifdef TRACK_MESSAGES
    if (message_added)
      __sync_fetch_and_add(outstanding_messages+source,1);
#endif
  }
  bool adjust_long_msgsize(gasnet_node_t source, void *&ptr, size_t &buffer_size,
                           int message_id, int chunks)
  {
    return endpoints[source]->adjust_long_msgsize(ptr, buffer_size, message_id, chunks);
  }
  void report_activemsg_status(FILE *f)
  {
#ifdef TRACE_MESSAGES
    int mynode = gasnet_mynode();
    for (int i = 0; i < total_endpoints; i++) {
      if (endpoints[i] == 0) continue;

      ActiveMessageEndpoint *e = endpoints[i];
      fprintf(f, "AMS: %d<->%d: S=%d R=%d\n", 
              mynode, i, e->sent_messages, e->received_messages);
    }
    fflush(f);
#else
    // for each node, report depth of outbound queues and LMB state
    int mynode = gasnet_mynode();
    for(int i = 0; i < total_endpoints; i++) {
      if (endpoints[i] == 0) continue;

      ActiveMessageEndpoint *e = endpoints[i];

      fprintf(f, "AMS: %d->%d: S=%zd L=%zd(%zd) W=%d,%d,%zd,%c,%c R=%d,%d\n",
              mynode, i,
              e->out_short_hdrs.size(),
              e->out_long_hdrs.size(), (e->out_long_hdrs.size() ? 
                                        (e->out_long_hdrs.front())->payload_size : 0),
              e->cur_write_lmb, e->cur_write_count, e->cur_write_offset,
              (e->lmb_w_avail[0] ? 'y' : 'n'), (e->lmb_w_avail[1] ? 'y' : 'n'),
              e->lmb_r_counts[0], e->lmb_r_counts[1]);
    }
    fflush(f);
#endif
  }
  void record_message(gasnet_node_t source, bool sent_reply)
  {
    endpoints[source]->record_message(sent_reply);
  }
private:
  const int total_endpoints;
  ActiveMessageEndpoint **endpoints;
  // This vector of outstanding message counts is accessed
  // by atomic intrinsics and is not protected by the lock
  int *outstanding_messages;
};

static EndpointManager *endpoint_manager;

static void handle_flip_req(gasnet_token_t token,
		     int flip_buffer, int flip_count)
{
  gasnet_node_t src;
  CHECK_GASNET( gasnet_AMGetMsgSource(token, &src) );
  endpoint_manager->handle_flip_request(src, flip_buffer, flip_count);
}

static void handle_flip_ack(gasnet_token_t token,
			    int ack_buffer)
{
  gasnet_node_t src;
  CHECK_GASNET( gasnet_AMGetMsgSource(token, &src) );
  endpoint_manager->handle_flip_ack(src, ack_buffer);
}

void init_endpoints(gasnet_handlerentry_t *handlers, int hcount,
		    int gasnet_mem_size_in_mb,
		    int registered_mem_size_in_mb,
		    int argc, const char *argv[])
{
  size_t srcdatapool_size = 64 << 20;

  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-ll:numlmbs")) {
      num_lmbs = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-ll:lmbsize")) {
      lmb_size = ((size_t)atoi(argv[++i])) << 10; // convert KB to bytes
      continue;
    }

    if(!strcmp(argv[i], "-ll:forcelong")) {
      force_long_messages = atoi(argv[++i]) != 0;
      continue;
    }

    if(!strcmp(argv[i], "-ll:sdpsize")) {
      srcdatapool_size = ((size_t)atoi(argv[++i])) << 20; // convert MB to bytes
      continue;
    }
  }

  size_t total_lmb_size = (gasnet_nodes() * 
			   num_lmbs *
			   lmb_size);

  // add in our internal handlers and space we need for LMBs
  size_t attach_size = ((((size_t)gasnet_mem_size_in_mb) << 20) +
			(((size_t)registered_mem_size_in_mb) << 20) +
			srcdatapool_size +
			total_lmb_size);

  if(gasnet_mynode() == 0)
    printf("Pinned Memory Usage: GASNET=%d, RMEM=%d, LMB=%zd, SDP=%zd, total=%zd\n",
	   gasnet_mem_size_in_mb, registered_mem_size_in_mb,
	   total_lmb_size >> 20, srcdatapool_size >> 20,
	   attach_size >> 20);

  // Don't bother checking this here.  Some GASNet conduits lie if 
  // the GASNET_PHYSMEM_MAX variable is not set.
#if 0
  if (attach_size > gasnet_getMaxLocalSegmentSize())
  {
    fprintf(stderr,"ERROR: Legion exceeded maximum GASNet segment size. "
                   "Requested %ld bytes but maximum set by GASNET "
                   "configuration is %ld bytes.  Legion will now exit...",
                   attach_size, gasnet_getMaxLocalSegmentSize());
    assert(false);
  }
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
  char *lmb_base = my_segment;  my_segment += total_lmb_size;
  assert(my_segment <= ((char *)(segment_info[gasnet_mynode()].addr) + segment_info[gasnet_mynode()].size)); 

#ifndef NO_SRCDATAPOOL
  srcdatapool = new SrcDataPool(srcdatapool_base, srcdatapool_size);
#endif

  endpoint_manager = new EndpointManager(gasnet_nodes());

  init_deferred_frees();
}

static int num_polling_threads = 0;
static pthread_t *polling_threads = 0;
static int num_sending_threads = 0;
static pthread_t *sending_threads = 0;
static volatile bool thread_shutdown_flag = false;

// do a little bit of polling to try to move messages along, but return
//  to the caller rather than spinning
void do_some_polling(void)
{
  endpoint_manager->push_messages(0);

  CHECK_GASNET( gasnet_AMPoll() );
}

static void *gasnet_poll_thread_loop(void *data)
{
  // each polling thread basically does an endless loop of trying to send
  //  outgoing messages and then polling
  while(!thread_shutdown_flag) {
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
#ifdef DEADLOCK_TRACE
    LegionRuntime::LowLevel::Runtime::get_runtime()->add_thread(&polling_threads[i]);
#endif
  }
}

static void* sender_thread_loop(void *index)
{
  long idx = (long)index;
  while (!thread_shutdown_flag) {
    endpoint_manager->push_messages(10000,true);
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
#ifdef DEADLOCK_TRACE
    LegionRuntime::LowLevel::Runtime::get_runtime()->add_thread(&sending_threads[i]);
#endif
  }
}

void stop_activemsg_threads(void)
{
  thread_shutdown_flag = true;

  if(polling_threads) {
    for(int i = 0; i < num_polling_threads; i++) {
      void *dummy;
      CHECK_PTHREAD( pthread_join(polling_threads[i], &dummy) );
    }
    num_polling_threads = 0;
    delete[] polling_threads;
  }
	
  if(sending_threads) {
    for(int i = 0; i < num_sending_threads; i++) {
      void *dummy;
      CHECK_PTHREAD( pthread_join(sending_threads[i], &dummy) );
    }
    num_sending_threads = 0;
    delete[] sending_threads;
  }

  thread_shutdown_flag = false;
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

  // if we have a contiguous payload that is in the KEEP mode, and in
  //  registered memory, we may be able to avoid a copy
  if((payload_mode == PAYLOAD_KEEP) && is_registered((void *)payload))
    payload_mode = PAYLOAD_KEEPREG;

  if (payload_mode != PAYLOAD_NONE)
    hdr->set_payload(new ContiguousPayload((void *)payload, payload_size, payload_mode),
                     payload_size, payload_mode,
                     dstptr);

  endpoint_manager->enqueue_message(target, hdr, true); // TODO: decide when OOO is ok?
}

void enqueue_message(gasnet_node_t target, int msgid,
		     const void *args, size_t arg_size,
		     const void *payload, size_t line_size,
		     off_t line_stride, size_t line_count,
		     int payload_mode, void *dstptr)
{
  assert(target != gasnet_mynode());

  OutgoingMessage *hdr = new OutgoingMessage(msgid, 
					     (arg_size + sizeof(int) - 1) / sizeof(int),
					     args);

  if (payload_mode != PAYLOAD_NONE)
    hdr->set_payload(new TwoDPayload(payload, line_size, 
                                     line_count, line_stride, payload_mode),
                                     line_size * line_count, payload_mode, dstptr);

  endpoint_manager->enqueue_message(target, hdr, true); // TODO: decide when OOO is ok?
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

  if (payload_mode != PAYLOAD_NONE)
    hdr->set_payload(new SpanPayload(spans, payload_size, payload_mode),
                     payload_size, payload_mode,
                     dstptr);

  endpoint_manager->enqueue_message(target, hdr, true); // TODO: decide when OOO is ok?
}

void handle_long_msgptr(gasnet_node_t source, void *ptr)
{
  assert(source != gasnet_mynode());

  endpoint_manager->handle_long_msgptr(source, ptr);
}

/*static*/ void SrcDataPool::release_srcptr_handler(gasnet_token_t token,
						    gasnet_handlerarg_t arg0,
						    gasnet_handlerarg_t arg1)
{
  uintptr_t srcptr = (((uintptr_t)(unsigned)arg1) << 32) | ((uintptr_t)(unsigned)arg0);
  // We may get pointers which are zero because we had to send a reply
  // Just ignore them
  if (srcptr != 0)
    srcdatapool->release_srcptr((void *)srcptr);
#ifdef TRACE_MESSAGES
  gasnet_node_t src;
  CHECK_GASNET( gasnet_AMGetMsgSource(token, &src) );
  endpoint_manager->record_message(src, false/*sent reply*/);
#endif
}

ContiguousPayload::ContiguousPayload(void *_srcptr, size_t _size, int _mode)
  : srcptr(_srcptr), size(_size), mode(_mode)
{}

void ContiguousPayload::copy_data(void *dest)
{
  log_sdp.info("contig copy %p <- %p (%zd bytes)", dest, srcptr, size);
  memcpy(dest, srcptr, size);
  if(mode == PAYLOAD_FREE)
    free(srcptr);
}

TwoDPayload::TwoDPayload(const void *_srcptr, size_t _line_size,
			 size_t _line_count,
			 ptrdiff_t _line_stride, int _mode)
  : srcptr(_srcptr), line_size(_line_size), line_count(_line_count),
    line_stride(_line_stride), mode(_mode)
{}

void TwoDPayload::copy_data(void *dest)
{
  char *dst_c = (char *)dest;
  const char *src_c = (const char *)srcptr;

  for(size_t i = 0; i < line_count; i++) {
    memcpy(dst_c, src_c, line_size);
    dst_c += line_size;
    src_c += line_stride;
  }
}

SpanPayload::SpanPayload(const SpanList&_spans, size_t _size, int _mode)
  : spans(_spans), size(_size), mode(_mode)
{}

void SpanPayload::copy_data(void *dest)
{
  char *dst_c = (char *)dest;
  off_t bytes_left = size;
  for(SpanList::const_iterator it = spans.begin(); it != spans.end(); it++) {
    assert(it->second <= bytes_left);
    memcpy(dst_c, it->first, it->second);
    dst_c += it->second;
    bytes_left -= it->second;
  }
  assert(bytes_left == 0);
}

extern bool adjust_long_msgsize(gasnet_node_t source, void *&ptr, size_t &buffer_size,
                                const void *args, size_t arglen)
{
  assert(source != gasnet_mynode());
  assert(arglen >= 2*sizeof(int));
  const int *arg_ptr = (const int*)args;

  return endpoint_manager->adjust_long_msgsize(source, ptr, buffer_size,
                                               arg_ptr[0], arg_ptr[1]);
}

extern void report_activemsg_status(FILE *f)
{
  endpoint_manager->report_activemsg_status(f); 
}

extern void record_message(gasnet_node_t source, bool sent_reply)
{
#ifdef TRACE_MESSAGES
  endpoint_manager->record_message(source, sent_reply);
#endif
}

