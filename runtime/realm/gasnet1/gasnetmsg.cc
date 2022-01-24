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

#include "realm/realm_config.h"
#include "realm/atomics.h"


#include "realm/gasnet1/gasnetmsg.h"
#include "realm/mutex.h"
#include "realm/cmdline.h"

#include <queue>
#include <assert.h>

#ifdef DETAILED_MESSAGE_TIMING
#include <sys/stat.h>
#include <fcntl.h>
#endif

#include "realm/threads.h"
#include "realm/timers.h"
#include "realm/logging.h"

// so OpenMPI borrowed gasnet's platform-detection code and didn't change
//  the define names - work around it by undef'ing anything set via mpi.h
//  before we include gasnet.h
#undef __PLATFORM_COMPILER_GNU_VERSION_STR

#ifndef GASNET_PAR
#define GASNET_PAR
#endif
#include <gasnet.h>

#ifndef GASNETT_THREAD_SAFE
#define GASNETT_THREAD_SAFE
#endif
#include <gasnet_tools.h>

// eliminate GASNet warnings for unused static functions
REALM_ATTR_UNUSED(static const void *ignore_gasnet_warning1) = (void *)_gasneti_threadkey_init;
REALM_ATTR_UNUSED(static const void *ignore_gasnet_warning2) = (void *)_gasnett_trace_printf_noop;

#define NO_DEBUG_AMREQUESTS

using namespace Realm;

enum { MSGID_CHANNEL_FLUSH = 250,
       MSGID_COMPLETION_REPLY = 252,
       MSGID_LONG_EXTENSION = 253,
       MSGID_FLIP_REQ = 254,
       MSGID_FLIP_ACK = 255 };

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

NodeID get_message_source(token_t token)
{
  gasnet_node_t src;
  CHECK_GASNET( gasnet_AMGetMsgSource(reinterpret_cast<gasnet_token_t>(token), &src) );
#ifdef DEBUG_AMREQUESTS
  printf("%d: source = %d\n", gasnet_mynode(), src);
#endif
  return src;
}  

#if 0
void send_srcptr_release(token_t token, uint64_t srcptr)
{
  CHECK_GASNET( gasnet_AMReplyShort2(reinterpret_cast<gasnet_token_t>(token), MSGID_RELEASE_SRCPTR, (handlerarg_t)srcptr, (handlerarg_t)(srcptr >> 32)) );
}
#endif

#ifdef DEBUG_MEM_REUSE
static int payload_count = 0;
#endif

Realm::Logger log_amsg("activemsg");
Realm::Logger log_spill("spill");

#ifdef ACTIVE_MESSAGE_TRACE
Realm::Logger log_amsg_trace("amtrace");

void record_am_handler(int handler_id, const char *description, bool reply)
{
  log_amsg_trace.info("AM Handler: %d %s %s", handler_id, description,
		      (reply ? "Reply" : "Request"));
}
#endif

static const int DEFERRED_FREE_COUNT = 128;
Realm::Mutex deferred_free_mutex;
int deferred_free_pos;
void *deferred_frees[DEFERRED_FREE_COUNT];

gasnet_seginfo_t *segment_info = 0;
atomic<size_t> total_messages_sent(0);
atomic<size_t> total_messages_rcvd(0);

static bool is_registered(void *ptr)
{
  ssize_t offset = ((char *)ptr) - ((char *)(segment_info[gasnet_mynode()].addr));
  if((offset >= 0) && ((size_t)offset < segment_info[gasnet_mynode()].size))
    return true;
  return false;
}

void init_deferred_frees(void)
{
  deferred_free_pos = 0;
  for(int i = 0; i < DEFERRED_FREE_COUNT; i++)
    deferred_frees[i] = 0;
}

void deferred_free(void *ptr)
{
#ifdef DEBUG_MEM_REUSE
  printf("%d: deferring free of %p\n", gasnet_mynode(), ptr);
#endif
  deferred_free_mutex.lock();
  void *oldptr = deferred_frees[deferred_free_pos];
  deferred_frees[deferred_free_pos] = ptr;
  deferred_free_pos = (deferred_free_pos + 1) % DEFERRED_FREE_COUNT;
  deferred_free_mutex.unlock();
  if(oldptr) {
#ifdef DEBUG_MEM_REUSE
    printf("%d: actual free of %p\n", gasnet_mynode(), oldptr);
#endif
    free(oldptr);
  }
}


struct OutgoingMessage {
  OutgoingMessage(unsigned _msgid, unsigned _num_args, const void *_args,
		  PendingCompletion *_comp);
  ~OutgoingMessage(void);

  void set_payload(PayloadSource *_payload, size_t _payload_size,
		   int _payload_mode, void *_dstptr = 0);
  inline void set_payload_empty(void) { payload_mode = PAYLOAD_EMPTY; }
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
  PendingCompletion *comp;
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

Realm::Logger log_sdp("srcdatapool");

class SrcDataPool {
public:
  SrcDataPool(void *base, size_t size);
  ~SrcDataPool(void);

  class Lock {
  public:
    Lock(SrcDataPool& _sdp) : sdp(_sdp) { sdp.mutex.lock(); }
    ~Lock(void) { sdp.mutex.unlock(); }
  protected:
    SrcDataPool& sdp;
  };

  // allocators must already hold the lock - prove it by passing a reference
  void *alloc_srcptr(size_t size_needed, Lock& held_lock);

  // enqueuing a pending message must also hold the lock
  void add_pending(OutgoingMessage *msg, Lock& held_lock);

  // releasing memory will take the lock itself
  void release_srcptr(void *srcptr);

  // attempt to allocate spill memory (may block, and then return false, if limit is hit)
  bool alloc_spill_memory(size_t size_needed, int msgtype, Lock& held_lock,
			  bool first_try);

  // release spilled memory (usually by moving it into actual srcdatapool)
  void release_spill_memory(size_t size_released, int msgtype, Lock& held_lock);

  void print_spill_data(Realm::Logger::LoggingLevel level = Realm::Logger::LEVEL_WARNING);

  static void release_srcptr_handler(gasnet_token_t token, gasnet_handlerarg_t arg0, gasnet_handlerarg_t arg1);

protected:
  size_t round_up_size(size_t size);

  friend class SrcDataPool::Lock;
  Realm::Mutex mutex;
  Realm::Mutex::CondVar condvar;
  size_t total_size;
  std::map<char *, size_t> free_list;
  std::queue<OutgoingMessage *> pending_allocations;
  // debug
  std::map<char *, size_t> in_use;
  std::map<void *, ssize_t> alloc_counts;

  atomic<size_t> current_spill_bytes, peak_spill_bytes;
  atomic<size_t> current_spill_threshold;
#define TRACK_PER_MESSAGE_SPILLING
#ifdef TRACK_PER_MESSAGE_SPILLING
  atomic<size_t> current_permsg_spill_bytes[256], peak_permsg_spill_bytes[256];
  atomic<size_t> total_permsg_spill_bytes[256];
#endif
  int current_suspended_spillers, total_suspended_spillers;
  double total_suspended_time;
public:
  static size_t max_spill_bytes;
  static size_t print_spill_threshold, print_spill_step;
};

static SrcDataPool *srcdatapool = 0;

size_t SrcDataPool::max_spill_bytes = 0;  // default = no limit
size_t SrcDataPool::print_spill_threshold = 1 << 30;  // default = 1 GB
size_t SrcDataPool::print_spill_step = 1 << 30;       // default = 1 GB

class SrcptrReleaser {
public:
  SrcptrReleaser(void *_ptr) : ptr(_ptr) {}
  void operator()() const { srcdatapool->release_srcptr(ptr); }
protected:
  void *ptr;
};

namespace Realm {
  namespace ThreadLocal {
    // certain threads are exempt from the max spillage due to deadlock concerns
    REALM_THREAD_LOCAL bool always_allow_spilling = false;

    // we need to tunnel information through calls into GASNet that
    //  call GASNet active message handlers, so use TLS
    REALM_THREAD_LOCAL TimeLimit *gasnet_work_until = 0;
  };
};

// wrapper so we don't have to expose SrcDataPool implementation
void release_srcptr(void *srcptr)
{
  assert(srcdatapool != 0);
  srcdatapool->release_srcptr(srcptr);
}

SrcDataPool::SrcDataPool(void *base, size_t size)
  : condvar(mutex)
{
  free_list[(char *)base] = size;
  total_size = size;

  current_spill_bytes.store(0);
  peak_spill_bytes.store(0);
#ifdef TRACK_PER_MESSAGE_SPILLING
  for(int i = 0; i < 256; i++) {
    current_permsg_spill_bytes[i].store(0);
    peak_permsg_spill_bytes[i].store(0);
    total_permsg_spill_bytes[i].store(0);
  }
#endif
  current_spill_threshold.store(print_spill_threshold);

  current_suspended_spillers = total_suspended_spillers = 0;
  total_suspended_time = 0;
}

SrcDataPool::~SrcDataPool(void)
{
  size_t total = 0;
  size_t nonzero = 0;
  for(std::map<void *, ssize_t>::const_iterator it = alloc_counts.begin(); it != alloc_counts.end(); it++) {
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
    assert(0);

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

bool SrcDataPool::alloc_spill_memory(size_t size_needed, int msgtype, Lock& held_lock,
				     bool first_try)
{

  // case 1: it fits, so add to total and see if we need to print stuff
  while(true) {
    size_t old_spill_bytes = current_spill_bytes.load();
    size_t new_spill_bytes = old_spill_bytes + size_needed;

    if((max_spill_bytes != 0) &&
       !ThreadLocal::always_allow_spilling &&
       !ThreadLocal::in_message_handler &&
       (new_spill_bytes > max_spill_bytes))
      break;

    if(current_spill_bytes.compare_exchange(old_spill_bytes,
					    new_spill_bytes)) {
      if(new_spill_bytes > max_spill_bytes)
	peak_spill_bytes.fetch_max(new_spill_bytes);

      size_t old_threshold = current_spill_threshold.load();
      if(new_spill_bytes > old_threshold) {
	if(current_spill_threshold.compare_exchange(old_threshold,
						    new_spill_bytes + print_spill_step))
	  print_spill_data();
      }

#ifdef TRACK_PER_MESSAGE_SPILLING
      size_t new_permsg_spill_bytes = current_permsg_spill_bytes[msgtype].fetch_add(size_needed) + size_needed;
      peak_permsg_spill_bytes[msgtype].fetch_max(new_permsg_spill_bytes);
#endif
      return true;
    }
  }
  
  // case 2: we've hit the max allowable spill amount, so stall until room is available

  // sanity-check: would this ever fit?
  assert(size_needed <= max_spill_bytes);

  log_spill.debug() << "max spill amount reached - suspension required ("
		    << current_spill_bytes.load() << " + " << size_needed << " > " << max_spill_bytes;

  // if this is the first try for this message, increase the total waiter count and complain
  current_suspended_spillers++;
  if(first_try) {
    total_suspended_spillers++;
    if(total_suspended_spillers == 1)
      print_spill_data();
  }

  double t1 = Realm::Clock::current_time();

  // sleep until the message would fit, although we won't try to allocate on this pass
  //  (this allows the caller to try the srcdatapool again first)
  while((current_spill_bytes.load() + size_needed) > max_spill_bytes) {
    condvar.wait();
    log_spill.debug() << "awake - rechecking: "
		      << current_spill_bytes.load() << " + " << size_needed << " > " << max_spill_bytes << "?";
  }

  current_suspended_spillers--;

  double t2 = Realm::Clock::current_time();
  double delta = t2 - t1;

  log_spill.debug() << "spill suspension complete: " << delta << " seconds";
  total_suspended_time += delta;

  return false; // allocation failed, should now be retried
}

void SrcDataPool::release_spill_memory(size_t size_released, int msgtype, Lock& held_lock)
{
  current_spill_bytes.fetch_sub(size_released);

#ifdef TRACK_PER_MESSAGE_SPILLING
  current_permsg_spill_bytes[msgtype].fetch_sub(size_released);
#endif

  // if there are any threads blocked on spilling data, wake them
  if(current_suspended_spillers > 0) {
    log_spill.debug() << "waking " << current_suspended_spillers << " suspended spillers";
    condvar.broadcast();
  }
}

void SrcDataPool::print_spill_data(Realm::Logger::LoggingLevel level)
{
  Realm::LoggerMessage msg = log_spill.newmsg(level);

  msg << "current spill usage = " << current_spill_bytes.load()
      << " bytes, peak = " << peak_spill_bytes.load();
#ifdef TRACK_PER_MESSAGE_SPILLING
  for(int i = 0; i < 256; i++)
    if(total_permsg_spill_bytes[i].load() > 0)
      msg << "\n"
	  << "  MSG " << i << ": "
	  << "cur=" << current_permsg_spill_bytes[i].load()
	  << " peak=" << peak_permsg_spill_bytes[i].load()
	  << " total=" << total_permsg_spill_bytes[i].load();
#endif
  if(total_suspended_spillers > 0)
    msg << "\n"
	<< "   suspensions=" << total_suspended_spillers
	<< " avg time=" << (total_suspended_time / total_suspended_spillers);
}

#if 0
#ifdef TRACK_ACTIVEMSG_SPILL_ALLOCS
size_t current_total_spill_bytes, peak_total_spill_bytes;
size_t current_spill_threshold;
size_t current_spill_step;
atomic<size_t> current_spill_bytes[256], peak_spill_bytes[256], total_spill_bytes[256];

void init_spill_tracking(void)
{
  if(getenv("ACTIVEMSG_SPILL_THRESHOLD")) {
    // value is in megabytes
    current_spill_threshold = atoi(getenv("ACTIVEMSG_SPILL_THRESHOLD")) << 20;
  } else
    current_spill_threshold = 1 << 30; // 1GB

  if(getenv("ACTIVEMSG_SPILL_STEP")) {
    // value is in megabytes
    current_spill_step = atoi(getenv("ACTIVEMSG_SPILL_STEP")) << 20;
  } else
    current_spill_step = 1 << 30; // 1GB

  current_total_spill_bytes = 0;
  for(int i = 0; i < 256; i++) {
    current_spill_bytes[i].store(0);
    peak_spill_bytes[i].store(0);
    total_spill_bytes[i].store(0);
  }
}
#endif

void print_spill_data(void)
{
  printf("spill node %d: current spill usage = %zd bytes, peak = %zd\n",
	 gasnet_mynode(), current_total_spill_bytes, peak_total_spill_bytes);
  for(int i = 0; i < 256; i++)
    if(total_spill_bytes[i] > 0) {
      printf("spill node %d:  MSG %d: cur=%zd peak=%zd total=%zd\n",
	     gasnet_mynode(), i,
	     current_spill_bytes[i].load(),
	     peak_spill_bytes[i].load(),
	     total_spill_bytes[i].load());
    }
}

void record_spill_alloc(int msgid, size_t bytes)
{
  size_t newcur = current_spill_bytes[msgid].fetch_add(bytes) + bytes;
  peak_spill_bytes[msgid].fetch_max(newcur);
  total_spill_bytes[msgid].fetch_add(bytes);

  size_t newtotal = current_total_spill_bytes.fetch_add(bytes) + bytes;
  peak_total_spill_bytes.fetch_max(newtotal);

  size_t old_thresh = current_spill_threshold.load();
  if(newtotal > old_thresh) {
    if(current_spill_threshold.compare_exchange(old_thresh,
						old_thresh + current_spill_step))
      print_spill_data();
  }
}

void record_spill_free(int msgid, size_t bytes)
{
  current_total_spill_bytes.fetch_sub(bytes);
  current_spill_bytes[msgid].fetch_sub(bytes);
}
#endif

OutgoingMessage::OutgoingMessage(unsigned _msgid, unsigned _num_args,
				 const void *_args, PendingCompletion *_comp)
  : msgid(_msgid), num_args(_num_args),
    comp(_comp),
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
      {
	// TODO: find way to avoid taking lock here?
	SrcDataPool::Lock held_lock(*srcdatapool);
	srcdatapool->release_spill_memory(payload_size, msgid, held_lock);
      }
      //record_spill_free(msgid, payload_size);
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
static int max_msgs_to_send = 8;
static bool strict_shutdown = true;
static size_t srcdatapool_size = 64 << 20;

// returns the largest payload that can be sent to a node (to a non-pinned
//   address)
size_t get_lmb_size(NodeID target_node)
{
  // not node specific right yet
  return lmb_size;
}

#ifdef DETAILED_MESSAGE_TIMING
static const size_t DEFAULT_MESSAGE_MAX_COUNT = 16 << 20;  // 16 million messages should be plenty

// some helper state to make sure we don't repeatedly log stall conditions
enum {
  MSGLOGSTATE_NORMAL,
  MSGLOGSTATE_SRCDATAWAIT,
  MSGLOGSTATE_LMBWAIT,
};

// little helper that automatically gets the current time
struct CurrentTime {
public:
  CurrentTime(void)
  {
#define USE_REALM_TIMER
#ifdef USE_REALM_TIMER
    now = Realm::Clock::current_time_in_nanoseconds();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    sec = ts.tv_sec;
    nsec = ts.tv_nsec;
#endif
  }

#ifdef USE_REALM_TIMER
  long long now;
#else
  unsigned sec, nsec;
#endif
};

struct MessageTimingData {
public:
  unsigned short msg_id;
  //char write_lmb;
  unsigned short target;
  unsigned msg_size;
  long long start;
  unsigned dur_nsec;
  unsigned queue_depth;
};

class DetailedMessageTiming {
public:
  DetailedMessageTiming(void)
    : message_count(0)
  {
    path = getenv("LEGION_MESSAGE_TIMING_PATH");
    if(path) {
      char *e = getenv("LEGION_MESSAGE_TIMING_MAX");
      if(e)
	message_max_count = atoi(e);
      else
	message_max_count = DEFAULT_MESSAGE_MAX_COUNT;

      message_timing = new MessageTimingData[message_max_count];
    } else {
      message_max_count = 0;
      message_timing = 0;
    }
  }

  ~DetailedMessageTiming(void)
  {
    delete[] message_timing;
  }

  int get_next_index(void)
  {
    if(message_max_count)
      return message_count.fetch_add(1);
    else
      return -1;
  }

  void record(int index, int peer, unsigned short msg_id, char write_lmb, unsigned msg_size, unsigned queue_depth,
	      const CurrentTime& t_start, const CurrentTime& t_end)
  {
    if((index >= 0) && (index < message_max_count)) {
      MessageTimingData &mtd(message_timing[index]);
      mtd.msg_id = msg_id | (write_lmb << 12);
      //mtd.write_lmb = write_lmb;
      mtd.target = peer;
      mtd.msg_size = msg_size;
#ifdef USE_REALM_TIMER
      mtd.start = t_start.now;
      unsigned long long delta = t_end.now - t_start.now;
#else
      long long now = (1000000000LL * t_start.sec) + t_start.nsec;
      mtd.start = now;
      unsigned long long delta = (t_end.sec - t_start.sec) * 1000000000ULL + t_end.nsec - t_start.nsec;
#endif
      mtd.dur_nsec = (delta > (unsigned)-1) ? ((unsigned)-1) : delta;
      mtd.queue_depth = queue_depth;
    }
  }

  // dump timing data from all the endpoints to a file
  void dump_detailed_timing_data(void)
  {
    if(!path) return;

    char filename[256];
    strcpy(filename, path);
    int l = strlen(filename);
    if(l && (filename[l-1] != '/'))
      filename[l++] = '/';
    sprintf(filename+l, "msgtiming_%d.dat", gasnet_mynode());
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    assert(fd >= 0);

    int count = message_count;
    if(count > message_max_count)
      count = message_max_count;
    ssize_t to_write = count * sizeof(MessageTimingData);
    if(to_write) {
      ssize_t amt = write(fd, message_timing, to_write);
      assert(amt == to_write);
    }

    close(fd);
  }

protected:
  const char *path;
  atomic<int> message_count;
  int message_max_count;
  MessageTimingData *message_timing;
};

static DetailedMessageTiming detailed_message_timing;
#endif


static IncomingMessageManager *incoming_message_manager = 0;

#if 0
extern void enqueue_incoming(NodeID sender, IncomingMessage *msg)
{
#ifdef ACTIVE_MESSAGE_TRACE
  log_amsg_trace.info("Active Message Received: %d %d %ld",
                      msg->get_msgid(), msg->get_peer(), msg->get_msgsize());
#endif
#ifdef DEBUG_AMREQUESTS
  printf("%d: incoming(%d, %p)\n", gasnet_mynode(), sender, msg);
#endif
  assert(incoming_message_manager != 0);
  incoming_message_manager->add_incoming_message(sender, msg);
}
#endif

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
    : peer(_peer), cond(mutex)
    , out_channel_closed(false), messages_sent(0)
    , in_channel_closed(false), messages_handled(0), messages_expected(size_t(-1))
  {
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
#ifdef DETAILED_MESSAGE_TIMING
    message_log_state = MSGLOGSTATE_NORMAL;
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
    recevied_messages.fetch_add(1);
    if (sent_reply)
      sent_messages.fetch_add(1);
#endif
  }

  // returns true if the limit was reached before all messages could
  //  be sent (i.e. there's still more to send)
  bool push_messages(int max_to_send, bool wait, TimeLimit work_until)
  {
    int count = 0;

    bool still_more = true;
    bool first_iteration = true;
    while(still_more && ((max_to_send == 0) || (count < max_to_send))) {
      // don't check for timeout on the first try
      if(!first_iteration && work_until.is_expired())
	return still_more;
      first_iteration = false;

      // attempt to get the mutex that covers the outbound queues - do not
      //  block
      if(!mutex.trylock()) break;

      // short messages are used primarily for flow control, so always try to send those first
      // (if a short message needs to be ordered with long messages, it goes in the long message
      // queue)
      if(out_short_hdrs.size() > 0) {
	OutgoingMessage *hdr = out_short_hdrs.front();
	out_short_hdrs.pop();
	still_more = !(out_short_hdrs.empty() && out_long_hdrs.empty());

#ifdef DETAILED_MESSAGE_TIMING
	int timing_idx = detailed_message_timing.get_next_index(); // grab this while we still hold the lock
	unsigned qdepth = out_long_hdrs.size(); // sic - we assume the short queue is always near-empty
	message_log_state = MSGLOGSTATE_NORMAL;
#endif
	// now let go of lock and send message
	mutex.unlock();

#ifdef DETAILED_MESSAGE_TIMING
	CurrentTime start_time;
#endif
	send_short(hdr);
#ifdef DETAILED_MESSAGE_TIMING
	detailed_message_timing.record(timing_idx, peer, hdr->msgid, -1, hdr->num_args*4, qdepth, start_time, CurrentTime());
#endif
	delete hdr;
	count++;
	continue;
      }

      // try to send a long message, but only if we have an LMB available
      //  on the receiving end
      if(out_long_hdrs.size() > 0) {
	OutgoingMessage *hdr;
	hdr = out_long_hdrs.front();

	// no payload?  this happens when a short/medium message needs to be ordered with long messages
	if(hdr->payload_size == 0) {
	  out_long_hdrs.pop();
	  still_more = !(out_short_hdrs.empty() && out_long_hdrs.empty());
#ifdef DETAILED_MESSAGE_TIMING
	  int timing_idx = detailed_message_timing.get_next_index(); // grab this while we still hold the lock
	  unsigned qdepth = out_long_hdrs.size();
	  message_log_state = MSGLOGSTATE_NORMAL;
#endif
	  mutex.unlock();
#ifdef DETAILED_MESSAGE_TIMING
	  CurrentTime start_time;
#endif
	  send_short(hdr);
#ifdef DETAILED_MESSAGE_TIMING
	  detailed_message_timing.record(timing_idx, peer, hdr->msgid, -1, hdr->num_args*4, qdepth, start_time, CurrentTime());
#endif
	  delete hdr;
	  count++;
	  continue;
	}

	// is the message still waiting on space in the srcdatapool?
	if(hdr->payload_mode == PAYLOAD_PENDING) {
#ifdef DETAILED_MESSAGE_TIMING
	  // log this if we haven't already
	  int timing_idx = -1;
	  unsigned qdepth = out_long_hdrs.size();
	  if(message_log_state != MSGLOGSTATE_SRCDATAWAIT) {
	    timing_idx = detailed_message_timing.get_next_index();
	    message_log_state = MSGLOGSTATE_SRCDATAWAIT;
	  }
#endif
	  mutex.unlock();
#ifdef DETAILED_MESSAGE_TIMING
	  CurrentTime now;
	  detailed_message_timing.record(timing_idx, peer, hdr->msgid, -2, hdr->num_args*4 + hdr->payload_size, qdepth, now, now);
#endif
	  break;
	}

	// do we have a known destination pointer on the target?  if so, no need to use LMB
	if(hdr->dstptr != 0) {
	  //printf("sending long message directly to %p (%zd bytes)\n", hdr->dstptr, hdr->payload_size);
	  out_long_hdrs.pop();
	  still_more = !(out_short_hdrs.empty() && out_long_hdrs.empty());
#ifdef DETAILED_MESSAGE_TIMING
	  int timing_idx = detailed_message_timing.get_next_index(); // grab this while we still hold the lock
	  unsigned qdepth = out_long_hdrs.size();
	  message_log_state = MSGLOGSTATE_NORMAL;
#endif
	  mutex.unlock();
#ifdef DETAILED_MESSAGE_TIMING
	  CurrentTime start_time;
#endif
	  send_long(hdr, hdr->dstptr);
#ifdef DETAILED_MESSAGE_TIMING
	  detailed_message_timing.record(timing_idx, peer, hdr->msgid, -1, hdr->num_args*4 + hdr->payload_size, qdepth, start_time, CurrentTime());
#endif
	  delete hdr;
	  count++;
	  continue;
	}

	// are we waiting for the next LMB to become available?
	if(!lmb_w_avail[cur_write_lmb]) {
#ifdef DETAILED_MESSAGE_TIMING
	  // log this if we haven't already
	  int timing_idx = -1;
	  unsigned qdepth = out_long_hdrs.size();
	  if(message_log_state != MSGLOGSTATE_LMBWAIT) {
	    timing_idx = detailed_message_timing.get_next_index();
	    message_log_state = MSGLOGSTATE_LMBWAIT;
	  }
#endif
	  mutex.unlock();
#ifdef DETAILED_MESSAGE_TIMING
	  CurrentTime now;
	  detailed_message_timing.record(timing_idx, peer, hdr->msgid, -3, hdr->num_args*4 + hdr->payload_size, qdepth, now, now);
#endif
	  break;
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
	  still_more = !(out_short_hdrs.empty() && out_long_hdrs.empty());

#ifdef DETAILED_MESSAGE_TIMING
	  int timing_idx = detailed_message_timing.get_next_index(); // grab this while we still hold the lock
	  unsigned qdepth = out_long_hdrs.size();
	  message_log_state = MSGLOGSTATE_NORMAL;
#endif
	  mutex.unlock();
#ifdef DEBUG_LMB
	  printf("LMB: sending %zd bytes %d->%d, [%p,%p)\n",
		 hdr->payload_size, gasnet_mynode(), peer,
		 dest_ptr, dest_ptr + hdr->payload_size);
#endif
#ifdef DETAILED_MESSAGE_TIMING
	  CurrentTime start_time;
#endif
	  send_long(hdr, dest_ptr);
#ifdef DETAILED_MESSAGE_TIMING
	  detailed_message_timing.record(timing_idx, peer, hdr->msgid, cur_write_lmb, hdr->num_args*4 + hdr->payload_size, qdepth, start_time, CurrentTime());
#endif
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

#ifdef DETAILED_MESSAGE_TIMING
	  int timing_idx = detailed_message_timing.get_next_index(); // grab this while we still hold the lock
	  unsigned qdepth = out_long_hdrs.size();
	  message_log_state = MSGLOGSTATE_NORMAL;
#endif
	  // now let go of the lock and send the flip request
	  mutex.unlock();

#ifdef DEBUG_LMB
	  printf("LMB: flipping buffer %d for %d->%d, [%p,%p), count=%d\n",
		 flip_buffer, gasnet_mynode(), peer, lmb_w_bases[flip_buffer],
		 lmb_w_bases[flip_buffer]+lmb_size, flip_count);
#endif
#ifdef ACTIVE_MESSAGE_TRACE
          log_amsg_trace.info("Active Message Request: %d %d 2 0",
			      MSGID_FLIP_REQ, peer);
#endif
#ifdef DETAILED_MESSAGE_TIMING
	  CurrentTime start_time;
#endif
	  CHECK_GASNET( gasnet_AMRequestShort2(peer, MSGID_FLIP_REQ,
                                               flip_buffer, flip_count) );
#ifdef DETAILED_MESSAGE_TIMING
	  detailed_message_timing.record(timing_idx, peer, MSGID_FLIP_REQ, flip_buffer, 8, qdepth, start_time, CurrentTime());
#endif
#ifdef TRACE_MESSAGES
	  sent_messages.fetch_add(1);
#endif

	  continue;
	}
      }

      // Couldn't do anything so if we were told to wait, goto sleep
      if (wait)
      {
	cond.wait();
      }
      // if we get here, we didn't find anything to do, so break out of loop
      //  after releasing the lock
      mutex.unlock();
      break;
    }

    return still_more;
  }

  bool enqueue_message(OutgoingMessage *hdr, bool in_order)
  {
    // BEFORE we take the message manager's mutex, we reserve the ability to
    //  allocate spill data
    if((hdr->payload_size > 0) &&
       ((hdr->payload_mode == PAYLOAD_COPY) ||
	(hdr->payload_mode == PAYLOAD_FREE))) {
      SrcDataPool::Lock held_lock(*srcdatapool);
      bool first_try = true;
      while(!srcdatapool->alloc_spill_memory(hdr->payload_size,
					     hdr->msgid,
					     held_lock,
					     first_try)) {
	log_spill.debug() << "spill reservation failed - retrying...";
	first_try = false;
      }
    }

    // need to hold the mutex in order to push onto one of the queues
    mutex.lock();

    // keep track of total messages sent, and don't allow any after we've
    //  "closed" the channel
    if(out_channel_closed) {
      if(hdr->msgid == MSGID_NEW_ACTIVEMSG) {
	// fish real message ID out of payload
	unsigned short id;
	memcpy(&id, reinterpret_cast<const char *>(hdr->args)+sizeof(BaseMedium),
	       sizeof(unsigned short));
	if(strict_shutdown) {
	  log_amsg.fatal() << "post-shutdown message: dst=" << peer << " type=" << activemsg_handler_table.lookup_message_name(id);
	  abort();
	} else
	  log_amsg.warning() << "dropping post-shutdown message - hang possible: dst=" << peer << " type=" << activemsg_handler_table.lookup_message_name(id);
      } else {
	// TODO: these are probably fine - ignore silently?
	log_amsg.fatal() << "post-shutdown message: dst=" << peer << " msgid=" << hdr->msgid;
	abort();
      }
      mutex.unlock();
      return true; // we don't want to add a todo entry
    }
    messages_sent++;

    // once we have the lock, we can safely move the message's payload to
    //  srcdatapool
    hdr->reserve_srcdata();

    // tell caller if we were empty before for managing todo lists
    bool was_empty = out_short_hdrs.empty() && out_long_hdrs.empty();

    // messages that don't need space in the LMB can progress when the LMB is full
    //  (unless they need to maintain ordering with long packets)
    if(!in_order && (hdr->payload_size <= gasnet_AMMaxMedium()))
      out_short_hdrs.push(hdr);
    else
      out_long_hdrs.push(hdr);
    // Signal in case there is a sleeping sender
    cond.signal();

    mutex.unlock();

    return was_empty;
  }

  int long_msgptr_to_index(const void *ptr)
  {
    // use 0 to mean "no match", so add 1 to any index we match
    for(int i = 0; i < num_lmbs; i++)
      if((ptr >= lmb_r_bases[i]) && (ptr < (lmb_r_bases[i] + lmb_size)))
	return i + 1;

    return 0;
  }

  // returns true if a message is enqueue AND we were empty before
  bool handle_long_msgptr(int msgptr_index)
  {
    if(msgptr_index <= 0)
      return false;

    int r_buffer = msgptr_index - 1;

#ifdef DEBUG_LMB
    printf("LMB: received %p for %d->%d in buffer %d, [%p, %p)\n",
	   ptr, peer, gasnet_mynode(), r_buffer, lmb_r_bases[r_buffer],
	   lmb_r_bases[r_buffer] + lmb_size);
#endif

    // now take the lock to increment the r_count and decide if we need
    //  to ack (can't actually send it here, so queue it up)
    bool message_added_to_empty_queue = false;
    mutex.lock();
    lmb_r_counts[r_buffer]++;
    if(lmb_r_counts[r_buffer] == 0) {
#ifdef DEBUG_LMB
      printf("LMB: acking flip of buffer %d for %d->%d, [%p,%p)\n",
	     r_buffer, peer, gasnet_mynode(), lmb_r_bases[r_buffer],
	     lmb_r_bases[r_buffer]+lmb_size);
#endif

      message_added_to_empty_queue = out_short_hdrs.empty() && out_long_hdrs.empty();
      OutgoingMessage *hdr = new OutgoingMessage(MSGID_FLIP_ACK, 1, &r_buffer, 0);
      out_short_hdrs.push(hdr);
      // wake up a sender
      cond.signal();
    }
    mutex.unlock();
    return message_added_to_empty_queue;
  }

  bool adjust_long_msgsize(void *&ptr, size_t &buffer_size, int frag_info)
  {
    int message_id = frag_info >> 12;
    int chunks = frag_info & 4095;
#ifdef DEBUG_AMREQUESTS
    printf("%d: adjust(%p, %zd, %d, %d)\n", gasnet_mynode(), ptr, buffer_size, message_id, chunks);
#endif
    // Quick out, if there was only one chunk, then we are good to go
    if (chunks == 1)
      return true;

    bool ready = false;;
    // now we need to hold the lock
    mutex.lock();
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
    mutex.unlock();
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
    received_messages.fetch_add(1);
#endif
    bool message_added_to_empty_queue = false;
    mutex.lock();
    lmb_r_counts[buffer] -= count;
    if(lmb_r_counts[buffer] == 0) {
#ifdef DEBUG_LMB
      printf("LMB: acking flip of buffer %d for %d->%d, [%p,%p)\n",
	     buffer, peer, gasnet_mynode(), lmb_r_bases[buffer],
	     lmb_r_bases[buffer]+lmb_size);
#endif

      message_added_to_empty_queue = out_short_hdrs.empty() && out_long_hdrs.empty();
      OutgoingMessage *hdr = new OutgoingMessage(MSGID_FLIP_ACK, 1, &buffer, 0);
      out_short_hdrs.push(hdr);
      // Wake up a sender
      cond.signal();
    }
    mutex.unlock();
    return message_added_to_empty_queue;
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
    received_messages.fetch_add(1);
#endif

    lmb_w_avail[buffer] = true;
    // wake up a sender in case we had messages waiting for free space
    mutex.lock();
    cond.signal();
    mutex.unlock();
  }

  void flush_outbound_messages(bool is_quiescent)
  {
    // take a snapshot of messages we've sent since the last flush
    size_t sent_snapshot;
    {
      AutoLock<> al(mutex);
      sent_snapshot = messages_sent;
      messages_sent = 0;
    }
    log_amsg.info() << "sending flush: " << gasnet_mynode() << "->" << peer << ": is_quiescent=" << is_quiescent << " count=" << sent_snapshot;
    handlerarg_t count_hi = sent_snapshot >> 32;
    handlerarg_t count_lo = sent_snapshot & 0xFFFFFFFFULL;
    CHECK_GASNET( gasnet_AMRequestShort3(peer, MSGID_CHANNEL_FLUSH,
					 count_lo, count_hi,
					 is_quiescent) );
  }

#if 0
  bool inbound_messages_flushed(bool verbose)
  {
    if(!in_channel_closed.load_acquire()) {
      if(verbose)
	log_amsg.print() << "flush check: " << peer << "->" << gasnet_mynode() << ": exp=?? cur=" << messages_handled.load();
      return false;
    }
    size_t exp = messages_expected.load();
    size_t cur = messages_handled.load();
    if(verbose)
      log_amsg.print() << "flush check: " << peer << "->" << gasnet_mynode() << ": exp=" << exp << " cur=" << cur;
    return(exp == cur);
  }
#endif

  void handle_channel_flush(handlerarg_t count_lo, handlerarg_t count_hi,
			    handlerarg_t is_quiescent)
  {
    size_t count = (size_t(count_hi) << 32) + size_t(count_lo);
    log_amsg.info() << "received flush: " << peer << "->" << gasnet_mynode() << ": count=" << count << " cur=" << messages_handled.load() << " is_quiescent=" << is_quiescent;

    {
      AutoLock<> al(quiescence_checker.mutex);
      // first, clear the overall quiesence flag if the peer isn't
      if(!is_quiescent)
	quiescence_checker.is_quiescent = false;
      // second, decrement the handled count by the amount we're expected to
      //  have handled
      size_t old_count = messages_handled.fetch_sub(count);
      // if counts match, all messages are already handled and we can
      //  contribute to the check completion
      // if we've handled MORE messages than the sender sent, then we've
      //  clearly failed the quiescence check
      if(old_count >= count) {
        if(old_count > count)
          quiescence_checker.is_quiescent = false;
	if(++quiescence_checker.messages_received == int(gasnet_nodes() - 1))
	  quiescence_checker.condvar.broadcast();
      }
    }
  }

protected:
  void send_short(OutgoingMessage *hdr)
  {
    if(hdr->msgid == MSGID_NEW_ACTIVEMSG) {
      // sanity check that we know where the frag/comp_info fields are
      int info_start;
      if(hdr->args[0] == BaseMedium::FRAG_INFO_MAGIC) {
	assert(hdr->args[1] == BaseMedium::COMP_INFO_MAGIC);
	info_start = 0;
      } else {
	assert(0);
      }

      hdr->args[info_start + 0] = 0; // no fragmentation for these messages

      // do we need local/remote completion?
      if((hdr->comp != 0) && completion_manager.mark_ready(hdr->comp)) {
	handlerarg_t comp_info = hdr->comp->index << 2;
	// bottom two bits of state are the remote/local completion bits
	comp_info += (hdr->comp->state.load() & 3);
	hdr->args[info_start + 1] = comp_info;
      } else
	hdr->args[info_start + 1] = 0; // no completion needed
    }

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
    sent_messages.fetch_add(1);
#endif
#ifdef ACTIVE_MESSAGE_TRACE
    log_amsg_trace.info("Active Message Request: %d %d %d %ld",
			hdr->msgid, peer, hdr->num_args, 
			(hdr->payload_mode == PAYLOAD_NONE) ? 
			  0 : hdr->payload_size);
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

    case 7:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium7(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort7(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2],
			       hdr->args[3], hdr->args[4], hdr->args[5],
			       hdr->args[6]) );
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

    case 9:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium9(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				 hdr->args[0], hdr->args[1], hdr->args[2],
				 hdr->args[3], hdr->args[4], hdr->args[5],
				 hdr->args[6], hdr->args[7], hdr->args[8]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort9(peer, hdr->msgid,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7], hdr->args[8]) );
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

    case 11:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium11(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				 hdr->args[0], hdr->args[1], hdr->args[2],
				 hdr->args[3], hdr->args[4], hdr->args[5],
				 hdr->args[6], hdr->args[7], hdr->args[8],
				 hdr->args[9], hdr->args[10]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort11(peer, hdr->msgid,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7], hdr->args[8],
				hdr->args[9], hdr->args[10]) );
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

    case 13:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium13(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				 hdr->args[0], hdr->args[1], hdr->args[2],
				 hdr->args[3], hdr->args[4], hdr->args[5],
				 hdr->args[6], hdr->args[7], hdr->args[8],
				 hdr->args[9], hdr->args[10], hdr->args[11],
				 hdr->args[12]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort13(peer, hdr->msgid,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7], hdr->args[8],
				hdr->args[9], hdr->args[10], hdr->args[11],
				hdr->args[12]) );
      }
      break;

    case 14:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium14(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				 hdr->args[0], hdr->args[1], hdr->args[2],
				 hdr->args[3], hdr->args[4], hdr->args[5],
				 hdr->args[6], hdr->args[7], hdr->args[8],
				 hdr->args[9], hdr->args[10], hdr->args[11],
				 hdr->args[12], hdr->args[13]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort14(peer, hdr->msgid,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7], hdr->args[8],
				hdr->args[9], hdr->args[10], hdr->args[11],
				hdr->args[12], hdr->args[13]) );
      }
      break;

    case 15:
      if(hdr->payload_mode != PAYLOAD_NONE) {
	CHECK_GASNET( gasnet_AMRequestMedium15(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				 hdr->args[0], hdr->args[1], hdr->args[2],
				 hdr->args[3], hdr->args[4], hdr->args[5],
				 hdr->args[6], hdr->args[7], hdr->args[8],
				 hdr->args[9], hdr->args[10], hdr->args[11],
				 hdr->args[12], hdr->args[13], hdr->args[14]) );
      } else {
	CHECK_GASNET( gasnet_AMRequestShort15(peer, hdr->msgid,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7], hdr->args[8],
				hdr->args[9], hdr->args[10], hdr->args[11],
				hdr->args[12], hdr->args[13], hdr->args[14]) );
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
    // sanity check that we know where the frag/comp_info fields are
    int info_start;
    if(hdr->args[0] == BaseMedium::FRAG_INFO_MAGIC) {
      assert(hdr->args[1] == BaseMedium::COMP_INFO_MAGIC);
      info_start = 0;
    } else {
      assert(0);
    }

    // is fragmentation required?
    int chunks;
    const size_t max_long_req = gasnet_AMMaxLongRequest();
    if(hdr->payload_size <= max_long_req) {
      // nope, no fragmentation
      chunks = 1;
      hdr->args[info_start + 0] = 0;
    } else {
      chunks = 1 + ((hdr->payload_size - 1) / max_long_req);
      assert(chunks <= 4095);  // 12 bits for chunk count
      hdr->args[info_start + 0] = ((next_outgoing_message_id << 12) +
				   chunks);
      next_outgoing_message_id++;
    }

    // do we need local/remote completion?
    if((hdr->comp != 0) && completion_manager.mark_ready(hdr->comp)) {
      handlerarg_t comp_info = hdr->comp->index << 2;
      // bottom two bits of state are the remote/local completion bits
      comp_info += (hdr->comp->state.load() & 3);
      hdr->args[info_start + 1] = comp_info;
    } else
      hdr->args[info_start + 1] = 0; // no completion needed

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
      sent_messages.fetch_add(1);
#endif
#ifdef ACTIVE_MESSAGE_TRACE
      log_amsg_trace.info("Active Message Request: %d %d %d %ld",
			  hdr->msgid, peer, hdr->num_args, size); 
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
  
  Realm::Mutex mutex;
  Realm::Mutex::CondVar cond;
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
  bool out_channel_closed;
  size_t messages_sent;
  // these are updated asynchronously
  Realm::atomic<bool> in_channel_closed;
  Realm::atomic<size_t> messages_handled, messages_expected;
#ifdef TRACE_MESSAGES
  atomic<int> sent_messages;
  atomic<int> received_messages;
#endif
#ifdef DETAILED_MESSAGE_TIMING
  int message_log_state;
#endif
};

void OutgoingMessage::set_payload(PayloadSource *_payload_src,
				  size_t _payload_size, int _payload_mode,
				  void *_dstptr)
{
  // die if a payload has already been attached
  assert(payload_mode == PAYLOAD_NONE);
  // We should never be called if either of these are true
  assert(_payload_mode != PAYLOAD_NONE);
  assert(_payload_size > 0);

  // payload must be non-empty, and fit in the LMB unless we have a dstptr for it
  log_sdp.info("setting payload (%zd, %d)", _payload_size, _payload_mode);
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
  // no or empty payload cases are easy
  if((payload_mode == PAYLOAD_NONE) ||
     (payload_mode == PAYLOAD_EMPTY)) return;

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
	// if we had reserved spill space, we can give it back now
	if((payload_mode == PAYLOAD_COPY) || (payload_mode == PAYLOAD_FREE)) {
	  log_spill.debug() << "returning " << payload_size << " unneeded bytes of spill";
	  srcdatapool->release_spill_memory(payload_size, msgid, held_lock);
	}

	// allocation succeeded - update state, but do copy below, after
	//  we've released the lock
	payload_mode = PAYLOAD_SRCPTR;
	payload = srcptr;
	if(comp) {
	  // TODO: pull out local completions and do them after the memcpy below
	} else {
	  comp = completion_manager.get_available();
	}
	void *lc = comp->add_local_completion(sizeof(CompletionCallback<SrcptrReleaser>));
	new(lc) CompletionCallback<SrcptrReleaser>(SrcptrReleaser(srcptr));
      } else {
	// if the allocation fails, we have to queue ourselves up

	// if we've been instructed to copy the data, that has to happen now
	// SJT: try to figure out/remember why this has to be done with lock held?
	if(payload_mode == PAYLOAD_COPY) {
	  void *copy_ptr = malloc(payload_size);
	  assert(copy_ptr != 0);
	  payload_src->copy_data(copy_ptr);
	  delete payload_src;
	  payload_src = new ContiguousPayload(copy_ptr, payload_size,
					      PAYLOAD_FREE);
	  // TODO: can do local completions after this point
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
      payload_mode = payload_src->get_payload_mode();
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
      // TODO: can do local completions after this point
    }
  }
}

void OutgoingMessage::assign_srcdata_pointer(void *ptr)
{
  assert(payload_mode == PAYLOAD_PENDING);
  assert(payload_src != 0);
  payload_src->copy_data(ptr);

  if(comp) {
    // TODO: do local completions here
  } else {
    comp = completion_manager.get_available();
  }
  void *lc = comp->add_local_completion(sizeof(CompletionCallback<SrcptrReleaser>));
  new(lc) CompletionCallback<SrcptrReleaser>(SrcptrReleaser(ptr));

  bool was_using_spill = (payload_src->get_payload_mode() == PAYLOAD_FREE);

  delete payload_src;
  payload_src = 0;

  payload = ptr;
  payload_mode = PAYLOAD_SRCPTR;

  if(was_using_spill) {
    // TODO: find way to avoid taking lock here?
    SrcDataPool::Lock held_lock(*srcdatapool);
    srcdatapool->release_spill_memory(payload_size, msgid, held_lock);
  }
}

class EndpointManager : public BackgroundWorkItem {
public:
  EndpointManager(int num_endpoints, int num_dedicated_workers,
		  Realm::CoreReservationSet& crs,
		  bool _use_bgwork)
    : BackgroundWorkItem("gasnet1 worker")
    , total_endpoints(num_endpoints)
    , dedicated_workers(num_dedicated_workers)
    , condvar(mutex)
    , bgworker_active(_use_bgwork)
  {
    endpoints = new ActiveMessageEndpoint*[num_endpoints];
    for (int i = 0; i < num_endpoints; i++)
    {
      if (((gasnet_node_t)i) == gasnet_mynode())
        endpoints[i] = 0;
      else
        endpoints[i] = new ActiveMessageEndpoint(i);
    }

    // keep a todo list of endpoints with non-empty queues
    todo_list = new int[total_endpoints + 1];  // one extra to distinguish full/empty
    todo_oldest = todo_newest = 0;

#ifdef TRACE_MESSAGES
    char filename[80];
    sprintf(filename, "ams_%d.log", gasnet_mynode());
    msgtrace_file = fopen(filename, "w");
    if(!msgtrace_file) {
      log_amsg.fatal() << "could not open message trace file '" << filename << "': " << strerror(errno);
      abort();
    }
    last_msgtrace_report.store(int(Realm::Clock::current_time())); // just keep the integer seconds
#endif

    // for worker threads
    shutdown_flag = false;
    if(num_dedicated_workers > 0)
      core_rsrv = new Realm::CoreReservation("EndpointManager workers", crs,
					     Realm::CoreReservationParameters());
    else
      core_rsrv = 0;
  }

  ~EndpointManager(void)
  {
#ifdef TRACE_MESSAGES
    report_activemsg_status(msgtrace_file);
    fclose(msgtrace_file);
#endif

    delete core_rsrv;
    delete[] todo_list;
  }

public:
  void add_todo_entry(int target)
  {
    //printf("%d: adding target %d to list\n", gasnet_mynode(), target);
    mutex.lock();
    todo_list[todo_newest] = target;
    todo_newest++;
    if(todo_newest > total_endpoints)
      todo_newest = 0;
    assert(todo_newest != todo_oldest); // should never wrap around
    // wake up any sleepers
    condvar.broadcast();
    mutex.unlock();
  }

  void handle_flip_request(gasnet_node_t src, int flip_buffer, int flip_count)
  {
    bool was_empty = endpoints[src]->handle_flip_request(flip_buffer, flip_count);
    if(was_empty)
      add_todo_entry(src);
  }
  void handle_flip_ack(gasnet_node_t src, int ack_buffer)
  {
    endpoints[src]->handle_flip_ack(ack_buffer);
  }

  void handle_channel_flush(gasnet_node_t src,
			    gasnet_handlerarg_t count_lo,
			    gasnet_handlerarg_t count_hi,
			    gasnet_handlerarg_t is_quiescent)
  {
    endpoints[src]->handle_channel_flush(count_lo, count_hi, is_quiescent);
  }

  // returns whether any more messages remain
  bool push_messages(int max_to_send, bool wait, TimeLimit work_until)
  {
    while(true) {
      // get the next entry from the todo list, waiting if requested
      if(wait) {
	mutex.lock();
	while(todo_oldest == todo_newest) {
	  //printf("outgoing todo list is empty - sleeping\n");
	  condvar.wait();
	}
      } else {
	// try to take our lock so we can pop an endpoint from the todo list
	if(!mutex.trylock()) {
	  // no lock, so we don't actually know if we have any messages - be conservative
	  return true;
	}

	// give up if list is empty too
	if(todo_oldest == todo_newest) {
	  // it would be nice to sanity-check here that all endpoints have 
	  //  empty queues, but there's a race condition here with endpoints
	  //  that have added messages but not been able to put themselves on
	  //  the todo list yet
	  mutex.unlock();
	  return false;
	}
      }

      // have the lock here, and list is non-empty - pop the front one
      int target = todo_list[todo_oldest];
      todo_oldest++;
      if(todo_oldest > total_endpoints)
	todo_oldest = 0;
      mutex.unlock();

      //printf("sending messages to %d\n", target);
      bool still_more = endpoints[target]->push_messages(max_to_send, wait,
							 work_until);
      //printf("done sending to %d - still more = %d\n", target, still_more);

      // if we didn't send them all, put this target back on the list
      if(still_more)
	add_todo_entry(target);

      // if time has elapsed, exit
      if(work_until.is_expired())
	return still_more;

      // we get stuck in this loop if a sender is waiting on an LMB flip, so make
      //  sure we do some polling inside the loop
      CHECK_GASNET( gasnet_AMPoll() );
    }
  }
  void enqueue_message(gasnet_node_t target, OutgoingMessage *hdr, bool in_order)
  {
    bool was_empty = endpoints[target]->enqueue_message(hdr, in_order);
    if(was_empty)
      add_todo_entry(target);
  }
  int long_msgptr_to_index(gasnet_node_t source, const void *ptr)
  {
    return endpoints[source]->long_msgptr_to_index(ptr);
  }
  void handle_long_msgptr(gasnet_node_t source, int msgptr_index)
  {
    bool was_empty = endpoints[source]->handle_long_msgptr(msgptr_index);
    if(was_empty)
      add_todo_entry(source);
  }
  bool adjust_long_msgsize(gasnet_node_t source, void *&ptr, size_t &buffer_size,
                           int frag_info)
  {
    return endpoints[source]->adjust_long_msgsize(ptr, buffer_size, frag_info);
  }
  void report_activemsg_status(FILE *f)
  {
#ifdef TRACE_MESSAGES
    int mynode = gasnet_mynode();
    for (int i = 0; i < total_endpoints; i++) {
      if (endpoints[i] == 0) continue;

      ActiveMessageEndpoint *e = endpoints[i];
      fprintf(f, "AMS: %d<->%d: S=%d R=%d\n", 
              mynode, i, e->sent_messages.load(), e->received_messages.load());
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

  void count_handled_message(gasnet_node_t source)
  {
    size_t new_count = endpoints[source]->messages_handled.fetch_add(1) + 1;
    if(new_count == 0) {
      // this was the last straggler message before a flush, so update the
      //  quiescence checker
      AutoLock<> al(quiescence_checker.mutex);
      if(++quiescence_checker.messages_received == int(gasnet_nodes() - 1))
	quiescence_checker.condvar.broadcast();
    }
  }

  void flush_message_channels(bool is_quiescent)
  {
    // step 1: send close messages
    for(int i = 0; i < total_endpoints; i++)
      if(endpoints[i])
	endpoints[i]->flush_outbound_messages(is_quiescent);
#if 0
    // step 2: wait until outbound queues are actually empty
    while(push_messages(0, false/*!wait*/, TimeLimit()))
      CHECK_GASNET( gasnet_AMPoll() );

    // step 3: wait until we've handled every incoming message we expect
    long long next_print = Realm::Clock::current_time_in_nanoseconds() + 1000000000;
    long long timeout = Realm::Clock::current_time_in_nanoseconds() + 15000000000LL;
    int cur = 0;
    while(cur < total_endpoints) {
      if(endpoints[cur]) {
	long long now = Realm::Clock::current_time_in_nanoseconds();
	bool verbose = false;
	if(now > next_print) {
	  verbose = true;
	  next_print = now + 1000000000;
	}
	if(now > timeout) {
	  log_amsg.fatal() << "timeout waiting for messages to flush";
	  abort();
	}
	if(!(endpoints[cur]->inbound_messages_flushed(verbose))) {
	  CHECK_GASNET( gasnet_AMPoll() );
	  continue;
	}
      }
      // on to the next node
      cur++;
    }
#endif
  }
  
  void start_polling_threads(void);

  void stop_threads(void);

  virtual bool do_work(TimeLimit work_until);

protected:
  // runs in a separate thread
  void polling_worker_loop(void);

private:
  const int total_endpoints, dedicated_workers;
  ActiveMessageEndpoint **endpoints;
  Realm::Mutex mutex;
  Realm::Mutex::CondVar condvar;
  bool bgworker_active;
  int *todo_list;
  int todo_oldest, todo_newest;
  bool shutdown_flag;
  Realm::CoreReservation *core_rsrv;
  std::vector<Realm::Thread *> polling_threads;
#ifdef TRACE_MESSAGES
  FILE *msgtrace_file;
  atomic<int> last_msgtrace_report;
#endif
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

static void handle_channel_flush(gasnet_token_t token,
				 handlerarg_t count_lo,
				 handlerarg_t count_hi,
				 handlerarg_t is_quiescent)
{
  gasnet_node_t src;
  CHECK_GASNET( gasnet_AMGetMsgSource(token, &src) );
  endpoint_manager->handle_channel_flush(src, count_lo, count_hi, is_quiescent);
}

#if 0
class IncomingMessageNew : public IncomingMessage {
 public:
  IncomingMessageNew(NodeID _src, void *_buf, size_t _nbytes,
		     ActiveMessageHandlerTable::MessageHandler _handler);

  virtual void run_handler(void);

  virtual int get_peer(void);
  virtual int get_msgid(void);
  virtual size_t get_msgsize(void);

  NodeID src;
  void *buf;
  size_t nbytes;
  ActiveMessageHandlerTable::MessageHandler handler;
  union {
    struct {
      BaseMedium base;
      unsigned short msgid;
      unsigned short sender;
      unsigned payload_len;
    } hdr;
    gasnet_handlerarg_t args[16];
  } header;
};

IncomingMessageNew::IncomingMessageNew(NodeID _src, void *_buf, size_t _nbytes,
				       ActiveMessageHandlerTable::MessageHandler _handler)
  : src(_src)
  , buf(_buf)
  , nbytes(_nbytes)
  , handler(_handler)
{}

void IncomingMessageNew::run_handler(void)
{
  long long t_start = 0;
  if(Config::profile_activemsg_handlers)
    t_start = Clock::current_time_in_nanoseconds();

  (*handler)(header.hdr.sender, header.args+6, buf, nbytes);

  long long t_end = 0;
  if(Config::profile_activemsg_handlers)
    t_end = Clock::current_time_in_nanoseconds();

  handle_long_msgptr(src, buf);

  if(Config::profile_activemsg_handlers)
    activemsg_handler_table.record_message_handler_call(header.hdr.msgid,
							t_start, t_end);
}

int IncomingMessageNew::get_peer(void)
{
  return src;
}

int IncomingMessageNew::get_msgid(void)
{
  return header.hdr.msgid;
}

size_t IncomingMessageNew::get_msgsize(void)
{
  return nbytes;
}
#endif

typedef union {
  struct {
    BaseMedium base;
    unsigned short msgid;
    unsigned short sender;
    unsigned payload_len;
  } hdr;
  gasnet_handlerarg_t args[16];
} MessageHeader;

static void incoming_message_handled(NodeID sender,
				     uintptr_t comp_info,
				     uintptr_t msgptr_index)
{
  endpoint_manager->count_handled_message(sender);
  handle_long_msgptr(sender, msgptr_index);
  if(comp_info != 0)
    CHECK_GASNET( gasnet_AMRequestShort1(sender, MSGID_COMPLETION_REPLY,
					 comp_info) );
}

static void handle_completion_reply(gasnet_token_t token,
				    gasnet_handlerarg_t arg0)
{
  int index = arg0 >> 2;
  bool do_local = ((arg0 & PendingCompletion::LOCAL_PENDING_BIT) != 0);
  bool do_remote = ((arg0 & PendingCompletion::REMOTE_PENDING_BIT) != 0);
  //printf("handle completion %d %d %d\n", index, do_local, do_remote);
  completion_manager.invoke_completions(index, do_local, do_remote);
}

static void handle_new_activemsg(gasnet_token_t token,
				 void *buf, size_t nbytes,
				 gasnet_handlerarg_t arg0,
				 gasnet_handlerarg_t arg1,
				 gasnet_handlerarg_t arg2,
				 gasnet_handlerarg_t arg3,
				 gasnet_handlerarg_t arg4,
				 gasnet_handlerarg_t arg5,
				 gasnet_handlerarg_t arg6,
				 gasnet_handlerarg_t arg7,
				 gasnet_handlerarg_t arg8,
				 gasnet_handlerarg_t arg9,
				 gasnet_handlerarg_t arg10,
				 gasnet_handlerarg_t arg11,
				 gasnet_handlerarg_t arg12,
				 gasnet_handlerarg_t arg13,
				 gasnet_handlerarg_t arg14,
				 gasnet_handlerarg_t arg15)
{
  NodeID src = get_message_source(token);

  gasnet_handlerarg_t frag_info = arg0;
  gasnet_handlerarg_t comp_info = arg1;
  
  bool handle_now = ((frag_info == 0) ||
		     adjust_long_msgsize(src, buf, nbytes, frag_info));
  if(handle_now) {
    total_messages_rcvd.fetch_add(1);

    unsigned short msgid = arg4 & 0xffff;

    MessageHeader header;
    header.args[0] = arg0;
    header.args[1] = arg1;
    header.args[2] = arg2;
    header.args[3] = arg3;
    header.args[4] = arg4;
    header.args[5] = arg5;
    header.args[6] = arg6;
    header.args[7] = arg7;
    header.args[8] = arg8;
    header.args[9] = arg9;
    header.args[10] = arg10;
    header.args[11] = arg11;
    header.args[12] = arg12;
    header.args[13] = arg13;
    header.args[14] = arg14;
    header.args[15] = arg15;

#ifdef ACTIVE_MESSAGE_TRACE
    log_amsg_trace.info("Active Message Received: %d %d %ld",
			msg->get_msgid(), msg->get_peer(), msg->get_msgsize());
#endif
#ifdef DEBUG_AMREQUESTS
    printf("%d: incoming(%d, %p)\n", gasnet_mynode(), src, msg);
#endif
    assert(incoming_message_manager != 0);
    // we'll always do local completion here, but remote completion has to be
    //  deferred if the message isn't handled inline
    int deferred_comp_info;
    if((comp_info & PendingCompletion::REMOTE_PENDING_BIT) != 0)
      deferred_comp_info = comp_info & ~PendingCompletion::LOCAL_PENDING_BIT;
    else
      deferred_comp_info = 0;

    int msgptr_index = endpoint_manager->long_msgptr_to_index(src, buf);

    bool handled = 
      incoming_message_manager->add_incoming_message(src, msgid,
						     &header.args[6], 10*4,
						     PAYLOAD_COPY,
						     buf, nbytes,
						     PAYLOAD_KEEP,
						     incoming_message_handled,
						     deferred_comp_info,
						     msgptr_index,
						     ((ThreadLocal::gasnet_work_until != 0) ?
						        *ThreadLocal::gasnet_work_until :
						        TimeLimit()));
    if(handled) {
      // we need to call the callback ourselves - we'll both local and remote
      //  completions as a gasnet reply (incoming_message_handled would use a
      //  request, which is not legal here)
      incoming_message_handled(src, 0, msgptr_index);
      if(comp_info != 0)
	CHECK_GASNET( gasnet_AMReplyShort1(token, MSGID_COMPLETION_REPLY,
					   comp_info) );
    } else {
      if((comp_info & PendingCompletion::LOCAL_PENDING_BIT) != 0) {
	gasnet_handlerarg_t local_comp_only = (comp_info &
					       ~PendingCompletion::REMOTE_PENDING_BIT);
	CHECK_GASNET( gasnet_AMReplyShort1(token, MSGID_COMPLETION_REPLY,
					   local_comp_only) );
      }
    }
  } else
    record_message(src, false);
}

void gasnet_parse_command_line(std::vector<std::string>& cmdline)
{
  Realm::CommandLineParser cp;
  cp.add_option_int("-ll:numlmbs", num_lmbs)
    .add_option_int_units("-ll:lmbsize", lmb_size, 'k')
    .add_option_int("-ll:forcelong", force_long_messages)
    .add_option_int_units("-ll:sdpsize", srcdatapool_size, 'm')
    .add_option_int("-ll:maxsend", max_msgs_to_send)
    .add_option_int("-ll:strict_shutdown", strict_shutdown)
    .add_option_int_units("-ll:spillwarn", SrcDataPool::print_spill_threshold, 'm')
    .add_option_int_units("-ll:spillstep", SrcDataPool::print_spill_step, 'm')
    .add_option_int_units("-ll:spillstall", SrcDataPool::max_spill_bytes, 'm');

  bool ok = cp.parse_command_line(cmdline);
  assert(ok);
}

void init_endpoints(size_t gasnet_mem_size,
		    size_t registered_mem_size,
		    size_t registered_ib_mem_size,
		    CoreReservationSet& crs,
		    int num_worker_threads,
		    BackgroundWorkManager& bgwork,
		    bool poll_use_bgwork,
		    IncomingMessageManager *message_manager)
{
  size_t total_lmb_size = (gasnet_nodes() * 
			   num_lmbs *
			   lmb_size);

  // add in our internal handlers and space we need for LMBs
  size_t attach_size = (gasnet_mem_size +
			registered_mem_size +
			registered_ib_mem_size +
			srcdatapool_size +
			total_lmb_size);

  if(gasnet_mynode() == 0) {
    log_amsg.info() << "Pinned Memory Usage: GASNET="
                    << (gasnet_mem_size << 20) << " MB, RMEM="
                    << (registered_mem_size >> 20) << " MB, IBRMEM="
                    << (registered_ib_mem_size >> 20) << " MB, LMB="
                    << (total_lmb_size >> 20) << " MB, SDP="
                    << (srcdatapool_size >> 20) << " MB, total="
                    << (attach_size >> 20) << " MB";
#ifdef DEBUG_REALM_STARTUP
    Realm::TimeStamp ts("entering gasnet_attach", false);
    fflush(stdout);
#endif
  }

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

  const int MAX_HANDLERS = 5;
  gasnet_handlerentry_t handlers[MAX_HANDLERS];
  int hcount = 0;

  handlers[hcount].index = MSGID_NEW_ACTIVEMSG;
  handlers[hcount].fnptr = (void (*)())handle_new_activemsg;
  hcount++;
  handlers[hcount].index = MSGID_FLIP_REQ;
  handlers[hcount].fnptr = (void (*)())handle_flip_req;
  hcount++;
  handlers[hcount].index = MSGID_FLIP_ACK;
  handlers[hcount].fnptr = (void (*)())handle_flip_ack;
  hcount++;
  handlers[hcount].index = MSGID_COMPLETION_REPLY;
  handlers[hcount].fnptr = (void (*)())handle_completion_reply;
  hcount++;
  handlers[hcount].index = MSGID_CHANNEL_FLUSH;
  handlers[hcount].fnptr = (void (*)())handle_channel_flush;
  hcount++;
  assert(hcount <= MAX_HANDLERS);
#ifdef ACTIVE_MESSAGE_TRACE
  record_am_handler(MSGID_FLIP_REQ, "Flip Request AM");
  record_am_handler(MSGID_FLIP_ACK, "Flip Acknowledgement AM");
  record_am_handler(MSGID_RELEASE_SRCPTR, "Release Source Pointer AM");
#endif

  CHECK_GASNET( gasnet_attach(handlers, hcount,
			      attach_size, 0) );

#ifdef DEBUG_REALM_STARTUP
  if(gasnet_mynode() == 0) {
    Realm::TimeStamp ts("exited gasnet_attach", false);
    fflush(stdout);
  }
#endif

  segment_info = new gasnet_seginfo_t[gasnet_nodes()];
  CHECK_GASNET( gasnet_getSegmentInfo(segment_info, gasnet_nodes()) );

  char *my_segment = (char *)(segment_info[gasnet_mynode()].addr);
  /*char *gasnet_mem_base = my_segment;*/  my_segment += gasnet_mem_size;
  /*char *reg_mem_base = my_segment;*/  my_segment += registered_mem_size;
  /*char *reg_ib_mem_base = my_segment;*/ my_segment += registered_ib_mem_size;
  char *srcdatapool_base = my_segment;  my_segment += srcdatapool_size;
  /*char *lmb_base = my_segment;*/  my_segment += total_lmb_size;
  assert(my_segment <= ((char *)(segment_info[gasnet_mynode()].addr) + segment_info[gasnet_mynode()].size)); 

#ifndef NO_SRCDATAPOOL
  if(srcdatapool_size > 0)
    srcdatapool = new SrcDataPool(srcdatapool_base, srcdatapool_size);
#endif

  endpoint_manager = new EndpointManager(gasnet_nodes(), num_worker_threads, crs,
					 poll_use_bgwork);
  if(poll_use_bgwork)
    endpoint_manager->add_to_manager(&bgwork);

  incoming_message_manager = message_manager;

  init_deferred_frees();
}

void EndpointManager::start_polling_threads(void)
{
  polling_threads.resize(dedicated_workers);
  for(int i = 0; i < dedicated_workers; i++)
    polling_threads[i] = Realm::Thread::create_kernel_thread<EndpointManager, 
							     &EndpointManager::polling_worker_loop>(this,
												    Realm::ThreadLaunchParameters(),
												    *core_rsrv);

  make_active();
}

void EndpointManager::stop_threads(void)
{
  // none of our threads actually sleep, so we can just set the flag and wait for them to notice
  shutdown_flag = true;
  
  for(std::vector<Realm::Thread *>::iterator it = polling_threads.begin();
      it != polling_threads.end();
      it++) {
    (*it)->join();
    delete (*it);
  }
  polling_threads.clear();

  // make sure no background worker is still trying to do our work
  {
    AutoLock<> al(mutex);
    while(bgworker_active)
      condvar.wait();
  }
#ifdef DEBUG_REALM
  shutdown_work_item();
#endif

#ifdef CHECK_OUTGOING_MESSAGES
  if(todo_oldest != todo_newest) {
    fprintf(stderr, "HELP!  shutdown occured with messages outstanding on node %d!\n", gasnet_mynode());
    while(todo_oldest != todo_newest) {
      int target = todo_list[todo_oldest];
      fprintf(stderr, "target = %d\n", target);
      while(!endpoints[target]->out_short_hdrs.empty()) {
	OutgoingMessage *m = endpoints[target]->out_short_hdrs.front();
	fprintf(stderr, "  SHORT: %d %d %zd\n", m->msgid, m->num_args, m->payload_size);
	endpoints[target]->out_short_hdrs.pop();
      }
      while(!endpoints[target]->out_long_hdrs.empty()) {
	OutgoingMessage *m = endpoints[target]->out_long_hdrs.front();
	fprintf(stderr, "  LONG: %d %d %zd\n", m->msgid, m->num_args, m->payload_size);
	endpoints[target]->out_long_hdrs.pop();
      }
      todo_oldest++;
      if(todo_oldest > total_endpoints)
	todo_oldest = 0;
    }
    //assert(false);
  }
#endif
}

bool EndpointManager::do_work(TimeLimit work_until)
{
  // make sure nested active mesage calls respect the time limit
  ThreadLocal::gasnet_work_until = &work_until;

  push_messages(max_msgs_to_send, false /*!wait*/, work_until);

  // poll if we're not out of time
  if(!work_until.is_expired())
    CHECK_GASNET( gasnet_AMPoll() );

  // relax the time limit again
  ThreadLocal::gasnet_work_until = 0;

  // always re-activate ourselves unless we're shutting down
  if(shutdown_flag) {
    AutoLock<> al(mutex);
    bgworker_active = false;
    condvar.broadcast();
    return false;
  } else
    return true;
}

void EndpointManager::polling_worker_loop(void)
{
  while(true) {
    bool still_more = endpoint_manager->push_messages(max_msgs_to_send,
						      false /*!wait*/,
						      TimeLimit());

    // check for shutdown, but only if we've pushed all of our messages
    if(shutdown_flag && !still_more)
      break;

    CHECK_GASNET( gasnet_AMPoll() );

#ifdef TRACE_MESSAGES
    // see if it's time to write out another update
    int now = (int)(Realm::Clock::current_time());
    int old = last_msgtrace_report.load();
    if(now > (old + 29)) {
      // looks like it's time - use an atomic test to see if we should do it
      if(last_msgtrace_report.compare_exchange(old, new))
	report_activemsg_status(msgtrace_file);
    }
#endif
  }
}

void start_polling_threads(void)
{
  endpoint_manager->start_polling_threads();
}

void start_handler_threads(size_t stack_size)
{
  incoming_message_manager->start_handler_threads(stack_size);
}

#if 0
void flush_activemsg_channels(void)
{
  // start with a barrier to make sure everybody has sent any
  //  requests that we might need to respond to first
  gasnet_barrier_notify(0, GASNET_BARRIERFLAG_ANONYMOUS);
  gasnet_barrier_wait(0, GASNET_BARRIERFLAG_ANONYMOUS);

  endpoint_manager->flush_message_channels();
  
  // finally another barrier to hopefully clear out any implicit replies
  gasnet_barrier_notify(0, GASNET_BARRIERFLAG_ANONYMOUS);
  gasnet_barrier_wait(0, GASNET_BARRIERFLAG_ANONYMOUS);
}
#endif

void stop_activemsg_threads(void)
{
  endpoint_manager->stop_threads();
	
  delete endpoint_manager;

#ifdef DETAILED_MESSAGE_TIMING
  // dump timing data from all the endpoints to a file
  detailed_message_timing.dump_detailed_timing_data();
#endif

  // print final spill stats at a low logging level
  srcdatapool->print_spill_data(Realm::Logger::LEVEL_INFO);
}
	
void enqueue_message(NodeID target, int msgid,
		     const void *args, size_t arg_size,
		     const void *payload, size_t payload_size,
		     int payload_mode, PendingCompletion *comp,
		     void *dstptr)
{
  assert((gasnet_node_t)target != gasnet_mynode());

  OutgoingMessage *hdr = new OutgoingMessage(msgid, 
					     (arg_size + sizeof(int) - 1) / sizeof(int),
					     args,
					     comp);

  // if we have a contiguous payload that is in the KEEP mode, and in
  //  registered memory, we may be able to avoid a copy
  if((payload_mode == PAYLOAD_KEEP) && is_registered((void *)payload))
    payload_mode = PAYLOAD_KEEPREG;

  if (payload_mode != PAYLOAD_NONE)
  {
    if (payload_size > 0) {
      PayloadSource *payload_src = 
        new ContiguousPayload((void *)payload, payload_size, payload_mode);
      hdr->set_payload(payload_src, payload_size, payload_mode, dstptr);
    } else {
      hdr->set_payload_empty();
    }
  }

  total_messages_sent.fetch_add(1);
  endpoint_manager->enqueue_message(target, hdr, true); // TODO: decide when OOO is ok?
}

void enqueue_message(NodeID target, int msgid,
		     const void *args, size_t arg_size,
		     const void *payload, size_t line_size,
		     off_t line_stride, size_t line_count,
		     int payload_mode, PendingCompletion *comp,
		     void *dstptr)
{
  assert((gasnet_node_t)target != gasnet_mynode());

  OutgoingMessage *hdr = new OutgoingMessage(msgid, 
					     (arg_size + sizeof(int) - 1) / sizeof(int),
					     args,
					     comp);

  if (payload_mode != PAYLOAD_NONE)
  {
    size_t payload_size = line_size * line_count;
    if (payload_size > 0) {
      PayloadSource *payload_src = new TwoDPayload(payload, line_size, 
                                       line_count, line_stride, payload_mode);
      hdr->set_payload(payload_src, payload_size, payload_mode, dstptr);
    } else {
      hdr->set_payload_empty();
    }
  }
  total_messages_sent.fetch_add(1);
  endpoint_manager->enqueue_message(target, hdr, true); // TODO: decide when OOO is ok?
}

void enqueue_message(NodeID target, int msgid,
		     const void *args, size_t arg_size,
		     const SpanList& spans, size_t payload_size,
		     int payload_mode, PendingCompletion *comp,
		     void *dstptr)
{
  assert((gasnet_node_t)target != gasnet_mynode());

  OutgoingMessage *hdr = new OutgoingMessage(msgid, 
  					     (arg_size + sizeof(int) - 1) / sizeof(int),
  					     args,
					     comp);

  if (payload_mode != PAYLOAD_NONE)
  {
    if (payload_size > 0) {
      PayloadSource *payload_src = new SpanPayload(spans, payload_size, payload_mode);
      hdr->set_payload(payload_src, payload_size, payload_mode, dstptr);
    } else {
      hdr->set_payload_empty();
    }
  }

  total_messages_sent.fetch_add(1);
  endpoint_manager->enqueue_message(target, hdr, true); // TODO: decide when OOO is ok?
}

void handle_long_msgptr(NodeID source, int msgptr_index)
{
  assert((gasnet_node_t)source != gasnet_mynode());

  endpoint_manager->handle_long_msgptr(source, msgptr_index);
}

/*static*/ void SrcDataPool::release_srcptr_handler(gasnet_token_t token,
						    gasnet_handlerarg_t arg0,
						    gasnet_handlerarg_t arg1)
{
  uintptr_t srcptr = (((uint64_t)(uint32_t)arg1) << 32) | ((uint32_t)arg0);
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


extern bool adjust_long_msgsize(NodeID source, void *&ptr, size_t &buffer_size,
				int frag_info)
{
  // special case: if the buffer size is zero, it's an empty message and no adjustment
  //   is needed
  if(buffer_size == 0)
    return true;

  assert((gasnet_node_t)source != gasnet_mynode());

  return endpoint_manager->adjust_long_msgsize(source, ptr, buffer_size,
					       frag_info);
}

extern void report_activemsg_status(FILE *f)
{
  endpoint_manager->report_activemsg_status(f); 
}

extern void record_message(NodeID source, bool sent_reply)
{
#ifdef TRACE_MESSAGES
  endpoint_manager->record_message(source, sent_reply);
#endif
}

QuiescenceChecker quiescence_checker;

QuiescenceChecker::QuiescenceChecker()
  : last_message_count(0)
  , condvar(mutex)
  , messages_received(0)
  , is_quiescent(true)
{}

size_t QuiescenceChecker::sample_messages_received_count(void)
{
  return total_messages_rcvd.load();
}

bool QuiescenceChecker::perform_check(size_t sampled_receive_count)
{
  // figure out if we are quiescent - did we send any messages since last time?
  size_t total_sent = total_messages_sent.load();
  bool any_messages_sent = (total_sent != last_message_count);
  bool new_messages_rcvd = (sampled_receive_count != total_messages_rcvd.load());

  log_amsg.info() << "local quiescence: prev=" << last_message_count
		  << " cur=" << total_sent
                  << " new_rcvd=" << new_messages_rcvd;

  last_message_count = total_sent;

  bool local_quiescence = (!any_messages_sent && !new_messages_rcvd);
  endpoint_manager->flush_message_channels(local_quiescence);

  // now take the lock and wait for the responses to come back
  bool result;
  {
    AutoLock<> al(mutex);

    // clear the flag if we sent any messages either
    if(!local_quiescence)
      is_quiescent = false;

    while(messages_received < int(gasnet_nodes() - 1))
      condvar.wait();

    result = is_quiescent;

    // reset so we can try again
    is_quiescent = true;
    messages_received = 0;
  }

  // if it didn't work, use a barrier to make sure everybody's ready to try
  //  again
  if(!result) {
    gasnet_barrier_notify(0, GASNET_BARRIERFLAG_ANONYMOUS);
    gasnet_barrier_wait(0, GASNET_BARRIERFLAG_ANONYMOUS);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////
//
// class ContiguousPayload
//

ContiguousPayload::ContiguousPayload(void *_srcptr, size_t _size, int _mode)
  : srcptr(_srcptr), size(_size), mode(_mode)
{}

void ContiguousPayload::copy_data(void *dest)
{
  //  log_sdp.info("contig copy %p <- %p (%zd bytes)", dest, srcptr, size);
  memcpy(dest, srcptr, size);
  if(mode == PAYLOAD_FREE)
    free(srcptr);
}

////////////////////////////////////////////////////////////////////////
//
// class TwoDPayload
//

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

////////////////////////////////////////////////////////////////////////
//
// class SpanPayload
//

SpanPayload::SpanPayload(const SpanList&_spans, size_t _size, int _mode)
  : spans(_spans), size(_size), mode(_mode)
{}

void SpanPayload::copy_data(void *dest)
{
  char *dst_c = (char *)dest;
  size_t bytes_left = size;
  for(SpanList::const_iterator it = spans.begin(); it != spans.end(); it++) {
    assert(it->second <= (size_t)bytes_left);
    memcpy(dst_c, it->first, it->second);
    dst_c += it->second;
    bytes_left -= it->second;
    assert(bytes_left >= 0);
  }
  assert(bytes_left == 0);
}

////////////////////////////////////////////////////////////////////////
//
// struct PendingCompletion
//

PendingCompletion::PendingCompletion()
  : index(-1)
  , next_free(0)
  , state(0)
  , local_bytes(0)
  , remote_bytes(0)
{}

void *PendingCompletion::add_local_completion(size_t bytes)
{
  assert((state.load() & READY_BIT) == 0);
#ifdef DEBUG_REALM
  assert((bytes % Realm::CompletionCallbackBase::ALIGNMENT) == 0);
#endif
  assert((local_bytes + remote_bytes + bytes) <= TOTAL_CAPACITY);

  // local completions are stored from the front
  void *ptr = storage + local_bytes;
  local_bytes += bytes;
  state.fetch_or(LOCAL_PENDING_BIT);

  return ptr;
}

void *PendingCompletion::add_remote_completion(size_t bytes)
{
  assert((state.load() & READY_BIT) == 0);
#ifdef DEBUG_REALM
  assert((bytes % Realm::CompletionCallbackBase::ALIGNMENT) == 0);
#endif
  assert((local_bytes + remote_bytes + bytes) <= TOTAL_CAPACITY);

  // remote completions are stored from the back
  remote_bytes += bytes;
  void *ptr = storage + (TOTAL_CAPACITY - remote_bytes);
  state.fetch_or(REMOTE_PENDING_BIT);

  return ptr;
}

bool PendingCompletion::mark_ready()
{
  if(state.load() != 0) {
    unsigned prev = state.fetch_or_acqrel(READY_BIT);
    assert((prev & READY_BIT) == 0);
    return true;
  } else {
    // we're empty, so don't set ready and tell the caller to recycle us
    return false;
  }
}

bool PendingCompletion::invoke_local_completions()
{
  // sanity-check that we have local completions pending, but do not clear
  //  the bit until we've invoked and destroyed those completions
  // however, if we observe now that the remote pending bit is not set, we
  //  don't need to check again later
  unsigned prev_state = state.load_acquire();
  assert((prev_state & LOCAL_PENDING_BIT) != 0);

  // local completions are at the start of the storage
  Realm::CompletionCallbackBase::invoke_all(storage, local_bytes);
  Realm::CompletionCallbackBase::destroy_all(storage, local_bytes);
  local_bytes = 0;

  // if the remote pending bit was set before, atomically clear the local
  //  bit while checking if remote is still set - if it still is, the remote
  //  completion callback will take care of freeing us
  if((prev_state & REMOTE_PENDING_BIT) != 0) {
    prev_state = state.fetch_and_acqrel(~LOCAL_PENDING_BIT);
    if((prev_state & REMOTE_PENDING_BIT) != 0)
      return false;
  }

  // clear ready bit (and local pending bit if we skipped the remote check)
  state.store_release(0);
  return true;
}
  
bool PendingCompletion::invoke_remote_completions()
{
  // sanity-check that we have remote completions pending, but do not clear
  //  the bit until we've invoked and destroyed those completions
  // however, if we observe now that the local pending bit is not set, we
  //  don't need to check again later
  unsigned prev_state = state.load_acquire();
  assert((prev_state & REMOTE_PENDING_BIT) != 0);

  // remote completions are at the end of the storage
  void *remote_start = storage + (TOTAL_CAPACITY - remote_bytes);
  Realm::CompletionCallbackBase::invoke_all(remote_start, remote_bytes);
  Realm::CompletionCallbackBase::destroy_all(remote_start, remote_bytes);
  remote_bytes = 0;

  // if the local pending bit was set before, atomically clear the remote
  //  bit while checking if local is still set - if it still is, the local
  //  completion callback will take care of freeing us
  if((prev_state & LOCAL_PENDING_BIT) != 0) {
    prev_state = state.fetch_and_acqrel(~REMOTE_PENDING_BIT);
    if((prev_state & LOCAL_PENDING_BIT) != 0)
      return false;
  }

  // clear ready bit (and remote pending bit if we skipped the remote check)
  state.store_release(0);
  return true;
}

////////////////////////////////////////////////////////////////////////
//
// struct PendingCompletionManager
//

/*extern*/ PendingCompletionManager completion_manager;

PendingCompletionManager::PendingCompletionManager()
  : first_free(0)
  , num_groups(0)
{
  for(size_t i = 0; i < (1 << LOG2_MAXGROUPS); i++)
    groups[i].store(0);
}

PendingCompletionManager::~PendingCompletionManager()
{
  size_t i = num_groups.load();
  while(i > 0)
    delete groups[--i].load();
}

PendingCompletion *PendingCompletionManager::get_available()
{
  // a pop attempt must take the mutex, but if we fail, we drop the
  //  mutex while we allocate and initialize another block
  {
    AutoLock<> al(mutex);
    // pop can still lose to a concurrent push, so iterate until we
    //  succeed or obseve an empty free list
    PendingCompletion *pc = first_free.load_acquire();
    while(pc != 0) {
      if(first_free.compare_exchange(pc, pc->next_free)) {
	// success - return completion we popped
	pc->next_free = 0;
	return pc;
      } else
	continue;  // try again - `index` was updated
    }
  }

  // allocate a new group and then add it to the list, playing nice with
  //  any other threads that are doing the same
  PendingCompletionGroup *newgrp = new PendingCompletionGroup;
  size_t grp_index = num_groups.load();
  while(true) {
    PendingCompletionGroup *expected = 0;
    if(groups[grp_index].compare_exchange(expected, newgrp)) {
      // success - this is our index
      break;
    } else {
      grp_index++;
      assert(grp_index < (1 << LOG2_MAXGROUPS));
    }
  }
  // increment the num_groups - it's ok if these increments happen out of
  //  order
  num_groups.fetch_add(1);

  // give all these new completions their indices
  for(size_t i = 0; i < (1 << PendingCompletionGroup::LOG2_GROUPSIZE); i++)
    newgrp->entries[i].index = ((grp_index << PendingCompletionGroup::LOG2_GROUPSIZE) + i);

  // we'll return the first entry to the caller, but we need to add the rest
  //  to the free list
  PendingCompletion *new_head = &newgrp->entries[1];
  for(size_t i = 2; i < (1 << PendingCompletionGroup::LOG2_GROUPSIZE); i++)
    newgrp->entries[i - 1].next_free = &newgrp->entries[i];
  PendingCompletion **new_tail = &newgrp->entries[(1 << PendingCompletionGroup::LOG2_GROUPSIZE) - 1].next_free;

  // compare-exchange loop to push to the free list
  PendingCompletion *prev_head = first_free.load();
  while(true) {
    *new_tail = prev_head;
    if(first_free.compare_exchange(prev_head, new_head))
      break;
  }

  return &newgrp->entries[0];
}

bool PendingCompletionManager::mark_ready(PendingCompletion *comp)
{
  if(comp->mark_ready()) {
    // completion is non-empty
    return true;
  } else {
    // completion is empty - add it back to the free list
#ifdef DEBUG_REALM
    assert(comp->next_free == 0);
#endif
    PendingCompletion *prev_head = first_free.load();
    while(true) {
      comp->next_free = prev_head;
      if(first_free.compare_exchange(prev_head, comp))
	break;
    }
    // tell caller to forget about 'comp'
    return false;
  }
}

void PendingCompletionManager::invoke_completions(int index, bool do_local,
						  bool do_remote)
{
  int grp_index = index >> PendingCompletionGroup::LOG2_GROUPSIZE;
  assert((grp_index >= 0) && (grp_index < (1 << LOG2_MAXGROUPS)));
  PendingCompletionGroup *grp = groups[grp_index].load();
  assert(grp != 0);
  int sub_index = index & ((1 << PendingCompletionGroup::LOG2_GROUPSIZE) - 1);
  PendingCompletion *comp = &grp->entries[sub_index];

  bool done;
  if(do_local && comp->invoke_local_completions()) {
    done = true; // no need to check remote
  } else {
    done = do_remote && comp->invoke_remote_completions();
  }

  // if we're done with this completion, put it back on the free list
  if(done) {
    PendingCompletion *prev_head = first_free.load();
    while(true) {
      comp->next_free = prev_head;
      if(first_free.compare_exchange(prev_head, comp))
	break;
    }
  }
}
