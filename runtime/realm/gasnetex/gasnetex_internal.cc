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

// GASNet-EX network module internals

#include "realm/gasnetex/gasnetex_internal.h"

#include "realm/gasnetex/gasnetex_handlers.h"

#include "realm/runtime_impl.h"
#include "realm/mem_impl.h"
#include "realm/logging.h"

#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_module.h"
#include "realm/cuda/cuda_internal.h"
#endif

#ifdef REALM_USE_HIP
#include "realm/hip/hip_module.h"
#include "realm/hip/hip_internal.h"
#endif

#include <gasnet_coll.h>
#include <gasnet_mk.h>

namespace Realm {

  // defined in gasnetex_module.cc
  extern Logger log_gex;

  Logger log_gex_obmgr("gexobmgr");
  Logger log_gex_xpair("gexxpair");
  Logger log_gex_quiesce("gexquiesce");
  Logger log_gex_msg("gexmsg");
  Logger log_gex_comp("gexcomp");
  Logger log_gex_bind("gexbind");

  static const unsigned MSGID_BITS = 12;

  // bit twiddling tricks
  static bool is_pow2(size_t val)
  {
    return ((val & (val - 1)) == 0);
  }

  template <typename T>
  static T roundup_pow2(T val, size_t alignment)
  {
    assert(is_pow2(alignment));
    return ((val + alignment - 1) & ~(alignment - 1));
  }

  // a packet checksum covers the header and payload, but also message
  //  type and lengths
  static uint32_t compute_packet_crc(gex_AM_Arg_t arg0,
				     const void *header_base,
				     size_t header_size,
				     const void *payload_base,
				     size_t payload_size)
  {
    uint32_t accum = 0xFFFFFFFF;
    accum = crc32c_accumulate(accum, &arg0, sizeof(arg0));
    accum = crc32c_accumulate(accum, &header_size, sizeof(header_size));
    accum = crc32c_accumulate(accum, &payload_size, sizeof(payload_size));
    accum = crc32c_accumulate(accum, header_base, header_size);
    if(payload_base)
      accum = crc32c_accumulate(accum, payload_base, payload_size);
    return ~accum;
  }

  // computes a packets crc and writes it into the last part of the header
  static void insert_packet_crc(gex_AM_Arg_t arg0,
				void *header_base, size_t header_size,
				const void *payload_base, size_t payload_size)
  {
    // make sure not to include the part of the header that will hold the
    //  CRC
    gex_AM_Arg_t crc = compute_packet_crc(arg0,
					  header_base,
					  header_size - sizeof(gex_AM_Arg_t),
					  payload_base,
					  payload_size);
    memcpy(static_cast<char *>(header_base) + header_size - sizeof(gex_AM_Arg_t),
	   &crc,
	   sizeof(gex_AM_Arg_t));
  }

  // checks a packet's crc and barfs on mismatch
  static void verify_packet_crc(gex_AM_Arg_t arg0,
				const void *header_base, size_t header_size,
				const void *payload_base, size_t payload_size)
  {
    // make sure not to include the part of the header that will hold the
    //  CRC
    gex_AM_Arg_t exp_crc, act_crc;
    memcpy(&exp_crc,
	   static_cast<const char *>(header_base) + header_size - sizeof(gex_AM_Arg_t),
	   sizeof(gex_AM_Arg_t));
    act_crc = compute_packet_crc(arg0,
				 header_base,
				 header_size - sizeof(gex_AM_Arg_t),
				 payload_base,
				 payload_size);
    if(exp_crc != act_crc) {
      log_gex.fatal() << "CRC MISMATCH: arg0=" << arg0
		      << " header_size=" << header_size
		      << " payload_size=" << payload_size
		      << " exp=" << std::hex << exp_crc
		      << " act=" << act_crc << std::dec;
      abort();
    }
  }

  namespace ThreadLocal {
    REALM_THREAD_LOCAL const TimeLimit *gex_work_until = nullptr;
    REALM_THREAD_LOCAL bool in_am_handler = false;
  };


  ////////////////////////////////////////////////////////////////////////
  //
  // class OutbufMetadata
  //

  OutbufMetadata::OutbufMetadata()
    : is_overflow(false)
    , next_overflow(nullptr)
    , realbuf(nullptr)
  {
    // clear all the packet types here so that we don't have to clear the
    //  entire array every time
    for(int i = 0; i < MAX_PACKETS; i++)
      pktbuf_pkt_types[i].store(PKTTYPE_INVALID);
  }

  void OutbufMetadata::set_state(State new_state)
  {
    switch(new_state) {
    case STATE_DATABUF:
      {
	assert(state == STATE_IDLE);
	databuf_rsrv_offset = 0;
	databuf_use_count = 0;
	remain_count.store(0);
	state = STATE_DATABUF;
	break;
      }
    case STATE_PKTBUF:
      {
	assert(state == STATE_IDLE);
	pktbuf_total_packets.store(0);
	pktbuf_rsrv_offset = 0;
#ifdef DEBUG_REALM
	for(int i = 0; i < MAX_PACKETS; i++) {
	  pktbuf_pkt_ends[i] = 0;
	  pktbuf_pkt_types[i].store(PKTTYPE_INVALID);
	}
#endif
	pktbuf_ready_packets.store(0);
	pktbuf_sent_packets = 0;
	pktbuf_sent_offset = 0;
	pktbuf_use_count = 0;
	remain_count.store(0);
	state = STATE_PKTBUF;
	break;
      }
    case STATE_IDLE:
      {
	assert(remain_count.load() == 0);
	state = STATE_IDLE;
	break;
      }
    default: assert(0);
    }
  }

  void OutbufMetadata::dec_usecount()
  {
    assert((state == STATE_DATABUF) || (state == STATE_PKTBUF));
    int prev = remain_count.fetch_sub(1);
    if(prev == 1)
      manager->free_outbuf(this);
  }

  uintptr_t OutbufMetadata::databuf_reserve(size_t bytes)
  {
    assert(state == STATE_DATABUF);
    size_t eff_bytes = roundup_pow2(bytes, 128);
    size_t old_rsrv = databuf_rsrv_offset;
    size_t new_rsrv = old_rsrv + eff_bytes;
    if(new_rsrv <= size) {
      databuf_use_count += 1;
      databuf_rsrv_offset = new_rsrv;
      log_gex.debug() << "dbuf reserve: this=" << this
		      << " base=" << std::hex << (baseptr + old_rsrv) << std::dec;
      return baseptr + old_rsrv;
    } else
      return 0;
  }

  void OutbufMetadata::databuf_close()
  {
    assert(state == STATE_DATABUF);
    // if the result of adding use_count to remain_count is 0, all uses have
    //  already been completed and we can free ourselves
    int prev = remain_count.fetch_add(databuf_use_count);
    if((prev + databuf_use_count) == 0)
      manager->free_outbuf(this);
  }

  bool OutbufMetadata::pktbuf_reserve(size_t bytes,
				      int& pktidx, uintptr_t& offset)
  {
    assert(state == STATE_PKTBUF);
    // reservation fails if we've hit the packet count limit or don't have
    //  enough space left
    if((pktbuf_total_packets.load() == MAX_PACKETS) ||
       ((pktbuf_rsrv_offset + bytes) > size))
      return false;

    pktidx = pktbuf_total_packets.load();
    offset = pktbuf_rsrv_offset;
    pktbuf_rsrv_offset += bytes;
    pktbuf_pkt_ends[pktidx] = pktbuf_rsrv_offset;
    pktbuf_total_packets.store(pktidx + 1);
    return true;
  }

  void OutbufMetadata::pktbuf_close()
  {
    assert(state == STATE_PKTBUF);
    if(is_overflow) {
      // update the use count on the realbuf, and then we can just delete
      //  ourselves
      assert(realbuf.load());
      int prev = realbuf.load()->remain_count.fetch_add(pktbuf_use_count);
      if((prev + pktbuf_use_count) == 0)
	manager->free_outbuf(realbuf.load());

      log_gex_obmgr.info() << "delete overflow: ovbuf=" << this << " baseptr=" << std::hex << baseptr << std::dec;
#ifdef REALM_ON_WINDOWS
      _aligned_free(reinterpret_cast<void *>(baseptr));
#else
      free(reinterpret_cast<void *>(baseptr));
#endif
      delete this;
    } else {
      // if the result of adding use_count to remain_count is 0, all uses have
      //  already been completed and we can free ourselves
      int prev = remain_count.fetch_add(pktbuf_use_count);
      if((prev + pktbuf_use_count) == 0)
	manager->free_outbuf(this);
    }
  }

  uintptr_t OutbufMetadata::pktbuf_get_offset(int pktidx)
  {
    assert(state == STATE_PKTBUF);

    // the start of packet i is the end of packet i-1
    if(pktidx == 0) {
      return 0;
    } else {
      assert(pktidx <= pktbuf_total_packets.load());
      return pktbuf_pkt_ends[pktidx - 1];
    }
  }

  bool OutbufMetadata::pktbuf_commit(int pktidx, PktType pkttype,
				     bool update_realbuf)
  {
    assert(state == STATE_PKTBUF);
    // we may get a stale value of pktbuf_total_packets, but it's good enough
    //  for our purposes
    assert((pktidx >= 0) && (pktidx < pktbuf_total_packets.load()));

    // if we were an overflow packet, the realbuf update (if requested)
    //  must come first
    if(update_realbuf) {
      assert(realbuf.load());
      PktType prev = realbuf.load()->pktbuf_pkt_types[pktidx].exchange(pkttype);
      assert(prev == PKTTYPE_INVALID);
    }

    // indicate our readiness by setting our packet type
    PktType prev = pktbuf_pkt_types[pktidx].exchange(pkttype);
    assert(prev == PKTTYPE_INVALID);

    // try to append the known-ready range (i.e. [pktidx,new_ready) ) to the
    //  ready packets counter
    int old_ready = pktidx;
    while(true) {
      // see if any newer-but-consecutive packets are ready to go
      //  (note that the first round of this has no ordering w.r.t. other
      //  committers, so is a best-effort check - if we succeed at a CAS below,
      //  we'll recheck with the ordering benefits that the CAS provides)
      int new_ready = pktidx + 1;
      while((new_ready < MAX_PACKETS) &&
            (pktbuf_pkt_types[new_ready].load() != PKTTYPE_INVALID))
        new_ready++;

      if(!pktbuf_ready_packets.compare_exchange(old_ready, new_ready)) {
        // CAS failed - three cases to consider
        if(old_ready < pktidx) {
          // there are still some packets before us that aren't ready -
          //  whoever bumps the counter for them will take care of us
          return false;
        } else if(old_ready >= new_ready) {
          // somebody else has covered the entire range we wanted to add, so
          //  everything has been taken care of
          return false;
        } else {
          // somebody updated for part of the range, but not the whole
          //  thing - we need to try again to do the rest (using the
          //  updated value of old_ready)
          continue;
        }
      }

      // each time we bump the count, it's possible the committer of the
      //  now-oldest-unready packet lost the race to see our update, so check
      //  the next packet - if it's invalid, our work is done
      if((new_ready >= MAX_PACKETS) ||
         (pktbuf_pkt_types[new_ready].load() == PKTTYPE_INVALID))
        break;

      // possible race - at least one new packet looks ready, so try again
      old_ready = new_ready;
    }

    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class OutbufManager
  //

  OutbufManager::OutbufManager()
    : BackgroundWorkItem("gex-obmgr")
    , metadatas(0)
    , outbuf_size(0)
    , first_available(0)
    , num_overflow(0)
    , num_reserved(0)
    , overflow_head(nullptr)
    , overflow_tail(&overflow_head)
    , reserved_head(nullptr)
  {}

  OutbufManager::~OutbufManager()
  {
    delete[] metadatas;
  }

  void OutbufManager::init(size_t _outbuf_count, size_t _outbuf_size,
			   uintptr_t _baseptr)
  {
    assert(_outbuf_count > 0);
    outbuf_size = _outbuf_size;
    metadatas = new OutbufMetadata[_outbuf_count];
    for(size_t i = 0; i < _outbuf_count; i++) {
      metadatas[i].state = OutbufMetadata::STATE_IDLE;
      metadatas[i].manager = this;
      metadatas[i].nextbuf = (i == 0) ? nullptr : &metadatas[i-1];
      metadatas[i].baseptr = _baseptr + (i * _outbuf_size);
      metadatas[i].size = _outbuf_size;
    }
    first_available = &metadatas[_outbuf_count - 1];
  }

  OutbufMetadata *OutbufManager::alloc_outbuf(OutbufMetadata::State state,
					      bool overflow_ok)
  {
    OutbufMetadata *md;
    {
      AutoLock<> al(mutex);
      md = first_available;
      if(md)
	first_available = md->nextbuf;
    }

    if(REALM_LIKELY(md != nullptr)) {
      md->nextbuf = nullptr;
      md->set_state(state);
      log_gex_obmgr.info() << "alloc outbuf: outbuf=" << md << " state=" << state << " baseptr=" << std::hex << md->baseptr << std::dec;
      return md;
    } else {
      if(overflow_ok) {
	// dynamically allocate a buffer and a metadata and enqueue it into
	//  into the overflow list
	md = new OutbufMetadata();
	md->state = OutbufMetadata::STATE_IDLE;
	md->manager = this;
	md->nextbuf = nullptr;
	md->size = outbuf_size;
	md->is_overflow = true;
	md->set_state(state);

#ifdef REALM_ON_WINDOWS
	void *buffer = _aligned_malloc(outbuf_size, 16);
	assert(buffer);
#else
	void *buffer;
	int ret = posix_memalign(&buffer, 16, outbuf_size);
	assert(ret == 0);
#endif
	md->baseptr = reinterpret_cast<uintptr_t>(buffer);

	OutbufMetadata *realbuf = nullptr;
	{
	  AutoLock<> al(mutex);
	  // see if a real buffer showed up while we were doing this
	  if(first_available) {
	    realbuf = first_available;
	    first_available = realbuf->nextbuf;
	  } else {
	    num_overflow += 1;
	    *overflow_tail = md;
	    overflow_tail = &md->next_overflow;
	  }
	}

	if(realbuf) {
	  // we look a bit silly destroying things we just created, but it's
	  //  better than forcing copies and stuff down the road
#ifdef REALM_ON_WINDOWS
	  _aligned_free(buffer);
#else
	  free(buffer);
#endif
	  delete md;

	  // set up the realbuf as above
	  realbuf->nextbuf = nullptr;
	  realbuf->set_state(state);
	  log_gex_obmgr.info() << "alloc outbuf(2): outbuf=" << realbuf << " state=" << state << " baseptr=" << std::hex << realbuf->baseptr << std::dec;
	  return realbuf;
	} else {
	  log_gex_obmgr.info() << "alloc overflow: ovbuf=" << md << " state=" << state << " baseptr=" << std::hex << md->baseptr << std::dec;

	  return md;
	}

      } else {
	// no buffer to give
	return nullptr;
      }
    }
  }

  void OutbufManager::free_outbuf(OutbufMetadata *md)
  {
    // overflow bufs should never get here
    assert(!md->is_overflow);
    log_gex_obmgr.info() << "free outbuf: outbuf=" << md << " state=" << md->state << " baseptr=" << std::hex << md->baseptr << std::dec;
    md->set_state(OutbufMetadata::STATE_IDLE);

    bool new_work = false;
    {
      AutoLock<> al(mutex);

      if(num_overflow > num_reserved) {
	// divert this to back a pending overflow buf
	new_work = (num_reserved == 0);
	num_reserved += 1;
	md->nextbuf = reserved_head;
	reserved_head = md;
      } else {
	md->nextbuf = first_available;
	first_available = md;
      }
    }

    if(new_work)
      make_active();
  }

  bool OutbufManager::do_work(TimeLimit work_until)
  {
    // take the mutex and grab the first overflow and reserved realbuf -
    //  don't decrement the counts yet, because we don't want more than one
    //  thread working on this at a time
    OutbufMetadata *ovbuf;
    OutbufMetadata *realbuf;
    {
      AutoLock<> al(mutex);
      ovbuf = overflow_head;
      overflow_head = ovbuf->next_overflow;
      if(!overflow_head)
	overflow_tail = &overflow_head;
      realbuf = reserved_head;
      reserved_head = realbuf->nextbuf;
    }

    while(true) {
      ovbuf->next_overflow = nullptr;
      realbuf->nextbuf = nullptr;

      log_gex_obmgr.info() << "resolve overflow: ovbuf=" << ovbuf << " realbuf=" << realbuf;

#ifdef DEBUG_REALM
      // try to make it obvious if we send a packet that doesn't get copied
      //  right
      memset(reinterpret_cast<void *>(realbuf->baseptr), outbuf_size, 0xae);
#endif

      // copy what we can here before we link the buffers
      switch(ovbuf->state) {
      case OutbufMetadata::STATE_PKTBUF:
	{
	  realbuf->set_state(OutbufMetadata::STATE_PKTBUF);

	  // grab a snapshot of the number of total packet count
	  int num_pkts = ovbuf->pktbuf_total_packets.load();
	  for(int i = 0; i < num_pkts; i++) {
	    OutbufMetadata::PktType pkttype = ovbuf->pktbuf_pkt_types[i].load_acquire();
	    if(pkttype != OutbufMetadata::PKTTYPE_INVALID) {
              // try to copy the packet contents now
              OutbufMetadata::PktType expected = OutbufMetadata::PKTTYPE_INVALID;
              if(realbuf->pktbuf_pkt_types[i].compare_exchange(expected,
                                                               OutbufMetadata::PKTTYPE_COPY_IN_PROGRESS)) {
                if(pkttype != OutbufMetadata::PKTTYPE_CANCELLED) {
                  uintptr_t pktstart = ((i > 0) ?
				          ovbuf->pktbuf_pkt_ends[i - 1] :
				          0);
                  uintptr_t pktend = ovbuf->pktbuf_pkt_ends[i];
                  log_gex_obmgr.debug() << "resolve copy: " << realbuf
                                        << " " << pktstart << " " << pktend;
                  memcpy(reinterpret_cast<void *>(realbuf->baseptr + pktstart),
                         reinterpret_cast<const void *>(ovbuf->baseptr + pktstart),
                         pktend - pktstart);
                }
                realbuf->pktbuf_pkt_types[i].store_release(pkttype);
                // we don't update realbuf->pktbuf_ready_packets - the ovbuf
                //  remains authoritative on which packets are actually ready
              }
            }
	  }
	  break;
	}

      default: assert(0);
      }

      // finish by connecting the realbuf to the ovbuf
      ovbuf->realbuf.store_release(realbuf);

      // all done - see if we're out of time
      bool is_expired = work_until.is_expired();

      // now retake the mutex, decrement counts, and take another pair if
      //  one is ready and we've still got time
      {
	AutoLock<> al(mutex);

	num_overflow -= 1;
	num_reserved -= 1;
	if(num_reserved == 0)
	  return false; // all done
	if(is_expired)
	  break;
	// dequeue
	ovbuf = overflow_head;
	overflow_head = ovbuf->next_overflow;
	if(!overflow_head)
	  overflow_tail = &overflow_head;
	realbuf = reserved_head;
	reserved_head = realbuf->nextbuf;
      }
    }

    // if we fall out of the loop (rather than returning), there's still work
    //  for somebody to do
    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PendingCompletion
  //

  PendingCompletion::PendingCompletion()
    : index(-1)
    , next_free(0)
    , state(0)
    , local_bytes(0)
    , remote_bytes(0)
  {}

  void *PendingCompletion::add_local_completion(size_t bytes, bool late_ok)
  {
    if(!late_ok)
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

  bool PendingCompletion::mark_ready(unsigned exp_local,
				     unsigned exp_remote)
  {
    if(state.load() != 0) {
      // store the expected number of completion responses and then
      //  set the READY bit
      local_left.store(exp_local);
      remote_left.store(exp_remote);
      unsigned prev = state.fetch_or_acqrel(READY_BIT);
      assert((prev & READY_BIT) == 0);

      // common case is that counts are non-zero and we're done here
      if(REALM_LIKELY((exp_local != 0) && (exp_remote != 0)))
	return true;

      // do one/both sets of completions here...
      unsigned to_clear = 0;
      if((exp_local == 0) && ((prev & LOCAL_PENDING_BIT) != 0)) {
	Realm::CompletionCallbackBase::invoke_all(storage, local_bytes);
	Realm::CompletionCallbackBase::destroy_all(storage, local_bytes);
	local_bytes = 0;
	to_clear |= LOCAL_PENDING_BIT;
      }
      if((exp_remote == 0) && ((prev & REMOTE_PENDING_BIT) != 0)) {
	void *remote_start = storage + (TOTAL_CAPACITY - remote_bytes);
	Realm::CompletionCallbackBase::invoke_all(remote_start, remote_bytes);
	Realm::CompletionCallbackBase::destroy_all(remote_start, remote_bytes);
	remote_bytes = 0;
	to_clear |= REMOTE_PENDING_BIT;
      }

      prev = state.fetch_and(~to_clear);
      assert((prev & to_clear) == to_clear);
      if((prev & ~to_clear) == READY_BIT) {
	// clear the ready bit and recycle - nothing left here
	state.store(0);
	manager->recycle_comp(this);
	return false;
      } else {
	// stick one flavor of completion left to happen later
	return true;
      }
    } else {
      // we're empty, so tell caller and recycle ourselves
      manager->recycle_comp(this);
      return false;
    }
  }

  bool PendingCompletion::has_local_completions()
  {
    return((state.load() & LOCAL_PENDING_BIT) != 0);
  }

  bool PendingCompletion::has_remote_completions()
  {
    return((state.load() & REMOTE_PENDING_BIT) != 0);
  }

  bool PendingCompletion::invoke_local_completions()
  {
    unsigned prev_count = local_left.fetch_sub(1);
    assert(prev_count > 0);
    if(prev_count > 1) return false;  // we're not the last

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
    unsigned prev_count = remote_left.fetch_sub(1);
    assert(prev_count > 0);
    if(prev_count > 1) return false;  // we're not the last

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
  // class PendingCompletionManager
  //

  PendingCompletionManager::PendingCompletionManager()
    : first_free(0)
    , num_groups(0)
    , num_pending(0)
  {
    // set the soft limit at 50% of max capacity
    pending_soft_limit = size_t(1) << (LOG2_MAXGROUPS +
                                       PendingCompletionGroup::LOG2_GROUPSIZE -
                                       1);

    for(size_t i = 0; i < (1 << LOG2_MAXGROUPS); i++)
      groups[i].store(0);
  }

  PendingCompletionManager::~PendingCompletionManager()
  {
    int i = num_groups.load();
    while(i > 0)
      delete groups[--i].load();
  }

  PendingCompletion *PendingCompletionManager::get_available()
  {
    // a pop attempt must take the mutex, but if we fail, we drop the
    //  mutex while we allocate and initialize another block
    {
      AutoLock<> al(mutex);

      // whether we succeed here or eventually by creating our own, the
      //  number of pending completions goes up
      num_pending.fetch_add(1);

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
    int grp_index = num_groups.load();
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

    // give all these new completions their indices and pointer to us
    for(size_t i = 0; i < (1 << PendingCompletionGroup::LOG2_GROUPSIZE); i++) {
      newgrp->entries[i].index = ((grp_index << PendingCompletionGroup::LOG2_GROUPSIZE) + i);
      newgrp->entries[i].manager = this;
    }

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

  void PendingCompletionManager::recycle_comp(PendingCompletion *comp)
  {
#ifdef DEBUG_REALM
    assert(comp->state.load() == 0);
    assert(comp->next_free == 0);
#endif
    PendingCompletion *prev_head = first_free.load();
    while(true) {
      comp->next_free = prev_head;
      if(first_free.compare_exchange(prev_head, comp))
	break;
    }
    num_pending.fetch_sub(1);
  }

  PendingCompletion *PendingCompletionManager::lookup_completion(int index)
  {
    int grp_index = index >> PendingCompletionGroup::LOG2_GROUPSIZE;
    if(REALM_UNLIKELY((grp_index < 0) || (grp_index >= num_groups.load()))) {
      log_gex_comp.fatal() << "completion index out of range: index=" << index
                           << " num_groups=" << num_groups.load();
      abort();
    }
    PendingCompletionGroup *grp = groups[grp_index].load();
    assert(grp != 0);
    int sub_index = index & ((1 << PendingCompletionGroup::LOG2_GROUPSIZE) - 1);
    return &grp->entries[sub_index];
  }

  void PendingCompletionManager::invoke_completions(PendingCompletion *comp,
						    bool do_local,
						    bool do_remote)
  {
    bool done;
    if(do_local && comp->invoke_local_completions()) {
      done = true; // no need to check remote
    } else {
      done = do_remote && comp->invoke_remote_completions();
    }

    // if we're done with this completion, put it back on the free list
    if(done)
      recycle_comp(comp);
  }

  size_t PendingCompletionManager::num_completions_pending()
  {
    return num_pending.load();
  }

  bool PendingCompletionManager::over_pending_completion_soft_limit() const
  {
    return (num_pending.load() >= pending_soft_limit);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ChunkedRecycler<T,CHUNK_SIZE>
  //

  template <typename T, unsigned CHUNK_SIZE>
  ChunkedRecycler<T,CHUNK_SIZE>::ChunkedRecycler()
    : free_head(0)
    , free_tail(&free_head)
    , chunks_head(nullptr)
    , cur_alloc(0)
    , cur_capacity(0)
    , max_alloc(0)
  {}

  template <typename T, unsigned CHUNK_SIZE>
  ChunkedRecycler<T,CHUNK_SIZE>::~ChunkedRecycler()
  {
    // shouldn't tear this down with any objects outstanding
    assert(cur_alloc.load() == 0);
    // before we destroy any chunks, scan the entire free object list to see
    //  if any are associated with blocks being reclaimed (they wouldn't be on
    //  our list)
    while(free_head) {
      Chunk *backval = reinterpret_cast<const WithPtr *>(free_head)->backptr;
      free_head = reinterpret_cast<const WithPtr *>(free_head)->nextptr;
      if(backval) {
	unsigned prev = backval->remaining_count.fetch_sub(1);
	if(prev == 1)
	  delete backval;
      }
    }
    while(chunks_head) {
      Chunk *next = chunks_head->next_chunk;
      delete chunks_head;
      chunks_head = next;
    }
  }

  template <typename T, unsigned CHUNK_SIZE>
  T *ChunkedRecycler<T,CHUNK_SIZE>::alloc_obj()
  {
    // fast case - get the first available chunk off the free list
    uintptr_t ptr = 0;
    {
      AutoLock<> al(mutex);
      if(free_head) {
	ptr = free_head;
	free_head = *reinterpret_cast<const uintptr_t *>(free_head);
	if(!free_head)
	  free_tail = &free_head;
	size_t prev = cur_alloc.fetch_add(1);
	if(prev >= max_alloc)
	  max_alloc = prev + 1;
      }
    }
    if(ptr)
      return new(reinterpret_cast<void *>(ptr)) T;

    // slow case - we have to make a new chunk
    Chunk *new_chunk = new Chunk;
    new_chunk->remaining_count.store(CHUNK_SIZE);
    // we'll use the first element and chain the rest together for addition
    //  to the free list
    ptr = reinterpret_cast<uintptr_t>(&new_chunk->elements[0]);
    uintptr_t my_head = reinterpret_cast<uintptr_t>(&new_chunk->elements[1]);
    uintptr_t *my_tail = &new_chunk->elements[CHUNK_SIZE - 1].nextptr;
    for(unsigned i = 1; i < CHUNK_SIZE - 1; i++)
      new_chunk->elements[i].nextptr = reinterpret_cast<uintptr_t>(&new_chunk->elements[i +1]);
    new_chunk->elements[CHUNK_SIZE - 1].nextptr = 0;
    for(unsigned i = 0; i < CHUNK_SIZE; i++)
      new_chunk->elements[i].backptr = nullptr;

    {
      AutoLock<> al(mutex);
      *free_tail = my_head;
      free_tail = my_tail;
      size_t prev = cur_alloc.fetch_add(1);
      if(prev >= max_alloc)
	max_alloc = prev + 1;
      cur_capacity += CHUNK_SIZE;
      new_chunk->next_chunk = chunks_head;
      chunks_head = new_chunk;
    }

    return new(reinterpret_cast<void *>(ptr)) T;
  }

  template <typename T, unsigned CHUNK_SIZE>
  void ChunkedRecycler<T,CHUNK_SIZE>::free_obj(T *obj)
  {
    // first, destroy the object
    obj->~T();

    // now, manufacture a pointer to the backptr to see if it's nonzero
    uintptr_t ptr = reinterpret_cast<uintptr_t>(obj);
    Chunk *backval = reinterpret_cast<const WithPtr *>(ptr)->backptr;

    if(REALM_LIKELY(backval == nullptr)) {
      // reinsert into free list
      Chunk *to_reclaim = nullptr;
      uintptr_t *my_tail = new(reinterpret_cast<void *>(ptr)) uintptr_t(0);
      {
	AutoLock<> al(mutex);
	*free_tail = ptr;
	free_tail = my_tail;
	size_t prev = cur_alloc.fetch_sub(1);
	// TODO: much too aggressive - add a programmable threshold here
	// should we take a chunk out of circulation?
	if((prev - 1) < (cur_capacity - CHUNK_SIZE)) {
	  to_reclaim = chunks_head;
	  chunks_head = chunks_head->next_chunk;
	  cur_capacity -= CHUNK_SIZE;
	}
      }
      if(to_reclaim) {
	// reclamation is delayed - we just mark each element so that the
	//  next time it's freed it'll take the other branch below
	for(unsigned i = 0; i < CHUNK_SIZE; i++)
	  to_reclaim->elements[i].backptr = to_reclaim;
      }
    } else {
      // we're part of a chunk that's been marked for reclamation - make sure
      //  we decrement the alloc count though
      cur_alloc.fetch_sub(1);

      unsigned prev = backval->remaining_count.fetch_sub(1);
      if(prev == 1)
	delete backval;
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class XmitSrcDestPair
  //

  XmitSrcDestPair::XmitSrcDestPair(GASNetEXInternal *_internal,
				   gex_EP_Index_t _src_ep_index,
				   gex_Rank_t _tgt_rank,
				   gex_EP_Index_t _tgt_ep_index)
    : internal(_internal)
    , src_ep_index(_src_ep_index)
    , tgt_rank(_tgt_rank)
    , tgt_ep_index(_tgt_ep_index)
    , packets_reserved(0)
    , packets_sent(0)
    , push_mutex_check("xpair push", this)
    , first_pbuf(nullptr)
    , cur_pbuf(nullptr)
    , imm_fail_count(0)
    , has_ready_packets(false)
    , first_fail_time(-1)
    , put_head(nullptr)
    , put_tailp(&put_head)
    , comp_reply_wrptr(0)
    , comp_reply_rdptr(0)
    , comp_reply_count(0)
    , comp_reply_capacity(64)
  {
    comp_reply_data = new gex_AM_Arg_t[comp_reply_capacity];
  }

  XmitSrcDestPair::~XmitSrcDestPair()
  {
    delete[] comp_reply_data;
  }

  bool XmitSrcDestPair::has_packets_queued() const
  {
    // both of these counters can be incrementing asynchronously, so read
    //  the sent counter first and if we lose a race with that, it just looks
    //  like the packet in question wasn't quite sent yet
    size_t num_sent = packets_sent.load_acquire();
    size_t num_reserved = packets_reserved.load();
    assert(num_reserved >= num_sent);
    return (num_reserved > num_sent);
  }

  void XmitSrcDestPair::record_immediate_packet()
  {
    packets_reserved.fetch_add(1);
    packets_sent.fetch_add(1);
  }

  bool XmitSrcDestPair::reserve_pbuf_helper(size_t total_bytes,
					    bool overflow_ok,
					    OutbufMetadata *&pktbuf, int& pktidx,
					    uintptr_t& baseptr)
  {
    // reservation uses a mutex to cover our first/cur_pbuf and, by extension,
    //  the pbuf(s) as well
    OutbufMetadata *popped_head = nullptr;
    {
      AutoLock<> al(mutex);

      // can we add to an existing current pbuf?
      uintptr_t offset = 0;
      if(cur_pbuf && cur_pbuf->pktbuf_reserve(total_bytes, pktidx, offset)) {
	pktbuf = cur_pbuf;
	if(cur_pbuf->is_overflow) {
	  // go straight to the realbuf if we have it already
	  OutbufMetadata *realbuf = cur_pbuf->realbuf.load();
	  if(realbuf)
	    baseptr = realbuf->baseptr + offset;
	  else
	    baseptr = cur_pbuf->baseptr + offset;
	} else
	  baseptr = cur_pbuf->baseptr + offset;
      } else {
	// get a new pbuf
	OutbufMetadata *new_pbuf;
	new_pbuf = internal->obmgr.alloc_outbuf(OutbufMetadata::STATE_PKTBUF,
						overflow_ok);
	if(!new_pbuf)
	  return false;  // out of space

	if(cur_pbuf) {
	  // bit of a special case - if the current is the first and it's idle,
	  //  but we can't fit in it, we need to pop it off because nobody else
	  //  will
	  if(!has_ready_packets && (cur_pbuf == first_pbuf.load()) &&
	     (cur_pbuf->pktbuf_sent_packets == cur_pbuf->pktbuf_total_packets.load())) {
	    popped_head = first_pbuf.load();
	    first_pbuf.store(new_pbuf);
	  } else
	    cur_pbuf->nextbuf = new_pbuf;
	} else
	  first_pbuf.store(new_pbuf);
	cur_pbuf = new_pbuf;

	// this must succeed
	bool ok = new_pbuf->pktbuf_reserve(total_bytes, pktidx, offset);
	assert(ok);
	pktbuf = new_pbuf;
	if(new_pbuf->is_overflow) {
	  // highly unlikely, go straight to the realbuf if we have it already
	  OutbufMetadata *realbuf = new_pbuf->realbuf.load();
	  if(realbuf)
	    baseptr = realbuf->baseptr + offset;
	  else
	    baseptr = new_pbuf->baseptr + offset;
	} else
	  baseptr = new_pbuf->baseptr + offset;
      }
    }
    if(popped_head)
      popped_head->pktbuf_close();

    return true;
  }

  bool XmitSrcDestPair::reserve_pbuf_inline(size_t hdr_bytes,
					    size_t payload_bytes,
					    bool overflow_ok,
					    OutbufMetadata *&pktbuf, int& pktidx,
					    void *&hdr_base, void *&payload_base)
  {
#ifdef DEBUG_REALM
    if(payload_bytes > 0) {
      gex_Event_t dummy;
      gex_Event_t *lc_opt = &dummy;
      gex_Flags_t flags = 0;
      size_t max_payload =
        GASNetEXHandlers::max_request_medium(internal->eps[src_ep_index],
                                             tgt_rank,
                                             tgt_ep_index,
                                             hdr_bytes,
                                             lc_opt,
                                             flags);
      if(payload_bytes > max_payload) {
        log_gex_xpair.fatal() << "medium payload too large!  src="
                              << Network::my_node_id << "/" << src_ep_index
                              << " tgt=" << tgt_rank
                              << "/" << tgt_ep_index
                              << " max=" << max_payload << " act=" << payload_bytes;
        abort();
      }
    }
#endif

    // a packet needs 8 bytes (arg0 + hdr/payload size), the header,
    // and then the payload aligned to 16 bytes (start and end)
    size_t pad_hdr_bytes = roundup_pow2(hdr_bytes + 2*sizeof(gex_AM_Arg_t), 16);
    size_t pad_payload_bytes = roundup_pow2(payload_bytes, 16);
    size_t total_bytes = pad_hdr_bytes + pad_payload_bytes;

    // reservation uses a mutex to cover our first/cur_pbuf and, by extension,
    //  the pbuf(s) as well
    uintptr_t baseptr = 0;
    if(!reserve_pbuf_helper(total_bytes, overflow_ok, pktbuf, pktidx, baseptr))
      return false;

    // use the two leading args to store hdr/payload sizes - we'll rewrite
    //  it to include msgid/etc. on commit
    gex_AM_Arg_t *info = reinterpret_cast<gex_AM_Arg_t *>(baseptr);
    info[0] = hdr_bytes;
    info[1] = payload_bytes;
    hdr_base = reinterpret_cast<void *>(baseptr + 2*sizeof(gex_AM_Arg_t));
    if(payload_bytes)
      payload_base = reinterpret_cast<void *>(baseptr + pad_hdr_bytes);
    else
      payload_base = nullptr;

    packets_reserved.fetch_add(1);
    return true;
  }

  bool XmitSrcDestPair::reserve_pbuf_long_rget(size_t hdr_bytes,
					       bool overflow_ok,
					       OutbufMetadata *&pktbuf,
					       int& pktidx,
					       void *&hdr_base)
  {
    // a packet needs 8 bytes (arg0 + hdr/payload size), the header,
    // and then the LongRgetData "payload" aligned to 16 bytes (start and end)
    size_t pad_hdr_bytes = roundup_pow2(hdr_bytes + 2*sizeof(gex_AM_Arg_t), 16);
    size_t pad_payload_bytes = roundup_pow2(sizeof(LongRgetData), 16);
    size_t total_bytes = pad_hdr_bytes + pad_payload_bytes;

    // reservation uses a mutex to cover our first/cur_pbuf and, by extension,
    //  the pbuf(s) as well
    uintptr_t baseptr = 0;
    if(!reserve_pbuf_helper(total_bytes, overflow_ok, pktbuf, pktidx, baseptr))
      return false;

    // store the header size in the leading args for now - we'll rewrite
    //  it to include msgid/etc. on commit
    gex_AM_Arg_t *info = reinterpret_cast<gex_AM_Arg_t *>(baseptr);
    info[0] = hdr_bytes;
    hdr_base = reinterpret_cast<void *>(baseptr + 2*sizeof(gex_AM_Arg_t));

    packets_reserved.fetch_add(1);
    return true;
  }

  bool XmitSrcDestPair::reserve_pbuf_put(bool overflow_ok,
                                         OutbufMetadata *&pktbuf,
                                         int& pktidx)
  {
    // a put just needs the PutMetadata in the queue
    size_t total_bytes = sizeof(PutMetadata);

    // reservation uses a mutex to cover our first/cur_pbuf and, by extension,
    //  the pbuf(s) as well
    uintptr_t baseptr = 0;
    if(!reserve_pbuf_helper(total_bytes, overflow_ok, pktbuf, pktidx, baseptr))
      return false;

    packets_reserved.fetch_add(1);
    return true;
  }

  // looks up the baseptr of the packet in the pktbuf, dealing with copies
  //  (or not) from overflow to a realbuf
  bool XmitSrcDestPair::commit_pbuf_helper(OutbufMetadata *pktbuf, int pktidx,
					   const void *hdr_base,
					   uintptr_t& baseptr)
  {
    uintptr_t offset = pktbuf->pktbuf_get_offset(pktidx);

    // see if we can move the packet to a real backing buffer now instead of
    //  in the injection thread
    baseptr = pktbuf->baseptr + offset;
    bool update_realbuf = false;
    if(pktbuf->is_overflow) {
      OutbufMetadata *realbuf = pktbuf->realbuf.load();
      if(realbuf) {
	// maybe we actually saw this at prepare time - see if the hdr_base
	//  is in the overflow range or the realbuf's range
	uintptr_t hdr_start = reinterpret_cast<uintptr_t>(hdr_base);
	if((hdr_start >= pktbuf->baseptr) &&
	   (hdr_start < (pktbuf->baseptr + pktbuf->size))) {
	  // it's in the overflow, so do the copy here
	  uintptr_t pktstart = offset;
	  uintptr_t pktend = pktbuf->pktbuf_pkt_ends[pktidx];
	  log_gex_obmgr.debug() << "commit copy: " << realbuf
				<< " " << pktstart << " " << pktend;
	  memcpy(reinterpret_cast<void *>(realbuf->baseptr + pktstart),
		 reinterpret_cast<const void *>(pktbuf->baseptr + pktstart),
		 pktend - pktstart);
	} else {
	  assert(!hdr_start ||
                 ((hdr_start >= realbuf->baseptr) &&
                  (hdr_start < (realbuf->baseptr + pktbuf->size))));
	  // already in right place!
	  uintptr_t pktstart = offset;
	  uintptr_t pktend = pktbuf->pktbuf_pkt_ends[pktidx];
	  log_gex_obmgr.debug() << "commit nocopy: " << realbuf
				<< " " << pktstart << " " << pktend;
	}
	baseptr = realbuf->baseptr + offset;
	update_realbuf = true;
      }
    }
    return update_realbuf;
  }

  void XmitSrcDestPair::commit_pbuf_inline(OutbufMetadata *pktbuf, int pktidx,
					   const void *hdr_base,
					   gex_AM_Arg_t arg0,
					   size_t act_payload_bytes)
  {
    uintptr_t baseptr;
    bool update_realbuf = commit_pbuf_helper(pktbuf, pktidx, hdr_base,
					     baseptr);

    gex_AM_Arg_t *info = reinterpret_cast<gex_AM_Arg_t *>(baseptr);
    size_t orig_hdr_bytes = info[0];
    size_t orig_payload_bytes = info[1];
    assert(act_payload_bytes <= orig_payload_bytes);
    bool is_short = (act_payload_bytes < orig_payload_bytes);

    // pack hdr/payload size into a single arg: { payload[25:0], hdr[7:2] }
    assert(((orig_hdr_bytes & 3) == 0) && (orig_hdr_bytes <= 127));
    assert(act_payload_bytes < (1U << 26));
    gex_AM_Arg_t sizes = (act_payload_bytes << 6) + (orig_hdr_bytes >> 2);
    info[0] = sizes;
    info[1] = arg0;
    bool new_work = pktbuf->pktbuf_commit(pktidx,
					  (is_short ?
					     OutbufMetadata::PKTTYPE_INLINE_SHORT :
					     OutbufMetadata::PKTTYPE_INLINE),
					  update_realbuf);
    // if this commit exposed work in that pbuf and it looks like this is our
    //  oldest pbuf, try to enqueue ourselves
    bool enqueue_pair = false;
    if(new_work && (pktbuf == first_pbuf.load())) {
      AutoLock<> al(mutex);

      // only actually enqueue if we aren't already and if this is still the
      //  current packet (avoids a race with the packet getting consumed by
      //  a pusher that's already running
      if((pktbuf == first_pbuf.load()) && !has_ready_packets) {
	has_ready_packets = true;
	enqueue_pair = (!put_head.load() &&
                        (comp_reply_count.load() == 0));
      }
    }

    if(enqueue_pair)
      request_push(false /*!force_critical*/);
  }

  void XmitSrcDestPair::commit_pbuf_long(OutbufMetadata *pktbuf, int pktidx,
					 const void *hdr_base,
					 gex_AM_Arg_t arg0,
					 const void *payload_base,
					 size_t payload_bytes,
					 uintptr_t dest_addr,
					 OutbufMetadata *databuf)
  {
    uintptr_t baseptr;
    bool update_realbuf = commit_pbuf_helper(pktbuf, pktidx, hdr_base,
					     baseptr);

    gex_AM_Arg_t *info = reinterpret_cast<gex_AM_Arg_t *>(baseptr);
    size_t hdr_bytes = info[0];

    // pack hdr/payload size into a single arg: { payload[25:0], hdr[7:2] }
    // encode a payload size of -1 to indicate long/rget
    assert(((hdr_bytes & 3) == 0) && (hdr_bytes <= 127));
    gex_AM_Arg_t sizes = (((1U << 22) - 1) << 6) + (hdr_bytes >> 2);
    info[0] = sizes;
    info[1] = arg0;

    size_t pad_hdr_bytes = roundup_pow2(hdr_bytes + 2*sizeof(gex_AM_Arg_t), 16);
    LongRgetData *extra = reinterpret_cast<LongRgetData *>(baseptr +
							   pad_hdr_bytes);
    extra->payload_base = payload_base;
    extra->payload_bytes = payload_bytes;
    extra->dest_addr = dest_addr;
    extra->l.databuf = databuf;

    bool new_work = pktbuf->pktbuf_commit(pktidx,
					  OutbufMetadata::PKTTYPE_LONG,
					  update_realbuf);

    // if this commit exposed work in that pbuf and it looks like this is our
    //  oldest pbuf, try to enqueue ourselves
    bool enqueue_pair = false;
    if(new_work && (pktbuf == first_pbuf.load())) {
      AutoLock<> al(mutex);

      // only actually enqueue if we aren't already and if this is still the
      //  current packet (avoids a race with the packet getting consumed by
      //  a pusher that's already running
      if((pktbuf == first_pbuf.load()) && !has_ready_packets) {
	has_ready_packets = true;
	enqueue_pair = (!put_head.load() &&
                        (comp_reply_count.load() == 0));
      }
    }

    if(enqueue_pair)
      request_push(false /*!force_critical*/);
  }

  void XmitSrcDestPair::commit_pbuf_rget(OutbufMetadata *pktbuf, int pktidx,
					 const void *hdr_base,
					 gex_AM_Arg_t arg0,
					 const void *payload_base,
					 size_t payload_bytes,
					 uintptr_t dest_addr,
					 gex_EP_Index_t src_ep_index,
					 gex_EP_Index_t tgt_ep_index)
  {
    uintptr_t baseptr;
    bool update_realbuf = commit_pbuf_helper(pktbuf, pktidx, hdr_base,
					     baseptr);

    gex_AM_Arg_t *info = reinterpret_cast<gex_AM_Arg_t *>(baseptr);
    size_t hdr_bytes = info[0];

    // pack hdr/payload size into a single arg: { payload[25:0], hdr[7:2] }
    // encode a payload size of -1 to indicate long/rget
    assert(((hdr_bytes & 3) == 0) && (hdr_bytes <= 127));
    gex_AM_Arg_t sizes = (((1U << 22) - 1) << 6) + (hdr_bytes >> 2);
    info[0] = sizes;
    info[1] = arg0;

    size_t pad_hdr_bytes = roundup_pow2(hdr_bytes + 2*sizeof(gex_AM_Arg_t), 16);
    LongRgetData *extra = reinterpret_cast<LongRgetData *>(baseptr +
							   pad_hdr_bytes);
    extra->payload_base = payload_base;
    extra->payload_bytes = payload_bytes;
    extra->dest_addr = dest_addr;
    extra->r.src_ep_index = src_ep_index;
    extra->r.tgt_ep_index = tgt_ep_index;

    bool new_work = pktbuf->pktbuf_commit(pktidx,
					  OutbufMetadata::PKTTYPE_RGET,
					  update_realbuf);
    // if this commit exposed work in that pbuf and it looks like this is our
    //  oldest pbuf, try to enqueue ourselves
    bool enqueue_pair = false;
    if(new_work && (pktbuf == first_pbuf.load())) {
      AutoLock<> al(mutex);

      // only actually enqueue if we aren't already and if this is still the
      //  current packet (avoids a race with the packet getting consumed by
      //  a pusher that's already running
      if((pktbuf == first_pbuf.load()) && !has_ready_packets) {
	has_ready_packets = true;
	enqueue_pair = (!put_head.load() &&
                        (comp_reply_count.load() == 0));
      }
    }

    if(enqueue_pair)
      request_push(false /*!force_critical*/);
  }

  void XmitSrcDestPair::commit_pbuf_put(OutbufMetadata *pktbuf, int pktidx,
                                        PendingPutHeader *put,
                                        const void *payload_base,
                                        size_t payload_size,
                                        uintptr_t dest_addr)
  {
    uintptr_t baseptr;
    bool update_realbuf = commit_pbuf_helper(pktbuf, pktidx,
                                             nullptr /*hdr_base*/,
					     baseptr);

    PutMetadata *meta = reinterpret_cast<PutMetadata *>(baseptr);
    meta->src_addr = payload_base;
    meta->dest_addr = dest_addr;
    meta->payload_bytes = payload_size;
    meta->put = put;

    bool new_work = pktbuf->pktbuf_commit(pktidx,
					  OutbufMetadata::PKTTYPE_PUT,
					  update_realbuf);
    // if this commit exposed work in that pbuf and it looks like this is our
    //  oldest pbuf, try to enqueue ourselves
    bool enqueue_pair = false;
    if(new_work && (pktbuf == first_pbuf.load())) {
      AutoLock<> al(mutex);

      // only actually enqueue if we aren't already and if this is still the
      //  current packet (avoids a race with the packet getting consumed by
      //  a pusher that's already running
      if((pktbuf == first_pbuf.load()) && !has_ready_packets) {
	has_ready_packets = true;
	enqueue_pair = (!put_head.load() &&
                        (comp_reply_count.load() == 0));
      }
    }

    if(enqueue_pair)
      request_push(false /*!force_critical*/);
  }

  void XmitSrcDestPair::cancel_pbuf(OutbufMetadata *pktbuf, int pktidx)
  {
    assert(0);
  }

  void XmitSrcDestPair::enqueue_completion_reply(gex_AM_Arg_t comp_info)
  {
#ifdef DEBUG_REALM
    // should never ask for a completion with neither local or remote
    //  bits set
    assert((comp_info & (PendingCompletion::LOCAL_PENDING_BIT |
                         PendingCompletion::REMOTE_PENDING_BIT)) != 0);
#endif

    // attempt an immediate send if they are enabled and we don't appear
    //  to have any completion replies queued up
    if(internal->module->cfg_use_immediate &&
       (comp_reply_count.load() == 0)) {
      // always in immediate mode
      gex_Flags_t flags = GEX_FLAG_IMMEDIATE;

      int ret = GASNetEXHandlers::send_completion_reply(internal->eps[src_ep_index],
							tgt_rank,
							tgt_ep_index,
							&comp_info,
							1,
							flags);
      if(ret == GASNET_OK) {
	log_gex_comp.info() << "comp immed " << tgt_rank << "/" << tgt_ep_index
			    << " " << comp_info;
	return;
      }
    }

    log_gex_comp.info() << "comp enqueue " << tgt_rank << "/" << tgt_ep_index
			<< " " << comp_info;

    bool enqueue_pair = false;
    {
      AutoLock<> al(mutex);

      unsigned cur_count = comp_reply_count.load();
      enqueue_pair = (!has_ready_packets && !put_head.load() &&
                      (cur_count == 0));

      // do we need to resize the queue?
      if(cur_count == comp_reply_capacity) {
	log_gex_comp.info() << "comp reply resize " << comp_reply_capacity;
	gex_AM_Arg_t *new_data = new gex_AM_Arg_t[comp_reply_capacity * 2];
	for(unsigned i = 0; i < cur_count; i++) {
	  new_data[i] = comp_reply_data[comp_reply_rdptr];
	  comp_reply_rdptr = (comp_reply_rdptr + 1) % comp_reply_capacity;
	}
	delete[] comp_reply_data;
	comp_reply_data = new_data;
	comp_reply_capacity *= 2;
	comp_reply_wrptr = cur_count;
	comp_reply_rdptr = 0;
      }
      comp_reply_data[comp_reply_wrptr] = comp_info;
      comp_reply_wrptr = (comp_reply_wrptr + 1) % comp_reply_capacity;
      comp_reply_count.store(cur_count + 1);
    }
    if(enqueue_pair)
      request_push(false /*!force_critical*/);
  }

  void XmitSrcDestPair::enqueue_put_header(PendingPutHeader *put)
  {
    log_gex.info() << "completed put: "
                   << put->src_ep_index << "/"
                   << std::hex << put->src_ptr << std::dec
                   << " -> " << put->target << "/" << put->tgt_ep_index << "/"
                   << std::hex << put->tgt_ptr << std::dec
                   << " size=" << put->payload_bytes << " arg0=" << put->arg0;

    // attempt an immediate injection if it is permitted (this is always older
    //  than any queued messages, so gets to jump ahead of them)
    if(internal->module->cfg_use_immediate) {
      gex_Event_t *lc_opt = GEX_EVENT_NOW;  // insist on local copy of header
      gex_Flags_t flags = GEX_FLAG_IMMEDIATE;

      int ret = GASNetEXHandlers::send_request_put_header(internal->eps[src_ep_index],
                                                          tgt_rank,
                                                          tgt_ep_index,
                                                          put->arg0,
                                                          put->hdr_data,
                                                          put->hdr_size,
                                                          put->tgt_ptr,
                                                          put->payload_bytes,
                                                          lc_opt,
                                                          flags);
      if(ret == GASNET_OK) {
        log_gex.debug() << "put header immediate";
        internal->put_alloc.free_obj(put);
        return;
      }
    }

    // otherwise have to enqueue
    bool enqueue_pair = false;
    {
      AutoLock<> al(mutex);

      enqueue_pair = (!has_ready_packets && !put_head.load() &&
                      (comp_reply_count.load() == 0));

      put->next_put.store(nullptr);
      (*put_tailp).store_release(put);
      put_tailp = &put->next_put;
    }
    if(enqueue_pair)
      request_push(false /*!force_critical*/);
  }

  void XmitSrcDestPair::request_push(bool force_critical)
  {
    // as soon as we're enqueued, some bgworker might start running
    //  push_packets so the mutual exclusion zone starts now
    push_mutex_check.lock();

    if(!force_critical && (internal->module->cfg_crit_timeout >= 0))
      internal->injector.add_ready_xpair(this);
    else
      internal->poller.add_critical_xpair(this);
  }

  void XmitSrcDestPair::push_packets(bool immediate_mode, TimeLimit work_until)
  {
    log_gex.debug() << "pushing: " << src_ep_index
		    << " " << tgt_rank << " " << tgt_ep_index;

    // we can test comp_reply_count without taking the mutex because if it's
    //  nonzero, only we can reduce it
    if(comp_reply_count.load() > 0) {
      const unsigned MAX_COMPS = 16;
      unsigned ncomps = 0;
      gex_AM_Arg_t comps[MAX_COMPS];
      bool requeue = false;
      bool do_push = false;

      // take the mutex to peek at the first pending comps (even if the
      //  rdptr can't move, resizes are possible)
      {
	AutoLock<> al(mutex);
	ncomps = std::min(MAX_COMPS, comp_reply_count.load());
	assert(ncomps > 0);
	for(unsigned i = 0; i < ncomps; i++)
	  comps[i] = comp_reply_data[(comp_reply_rdptr + i) % comp_reply_capacity];
      }

      do {
	gex_Flags_t flags = 0;
	if(immediate_mode) flags |= GEX_FLAG_IMMEDIATE;

	int ret = GASNetEXHandlers::send_completion_reply(internal->eps[src_ep_index],
							  tgt_rank,
							  tgt_ep_index,
							  comps,
							  ncomps,
							  flags);
	if(ret == GASNET_OK) {
	  log_gex_comp.debug() << "comp reply " << ncomps << " sent";
	  for(unsigned i = 0; i < ncomps; i++)
	    log_gex_comp.info() << "comp dequeue " << tgt_rank << "/" << tgt_ep_index
				<< " " << comps[i];
	  first_fail_time = -1;

	  bool is_expired = work_until.is_expired();

	  // retake mutex to actually bump the rdptr, take more comps if
	  //  they exist, or decide whether to requeue or push packets if not
	  {
	    AutoLock<> al(mutex);
	    comp_reply_rdptr = (comp_reply_rdptr + ncomps) % comp_reply_capacity;
	    unsigned new_count = comp_reply_count.load() - ncomps;
	    comp_reply_count.store(new_count);

	    if(new_count > 0) {
	      if(is_expired) {
		// can't take them now, so requeue
		ncomps = 0;
		requeue = true;
	      } else {
		ncomps = std::min(MAX_COMPS, new_count);
		for(unsigned i = 0; i < ncomps; i++)
		  comps[i] = comp_reply_data[(comp_reply_rdptr + i) % comp_reply_capacity];
	      }
	    } else {
	      // no more completion replies, but do pushing if there are
	      //  ready packets
	      ncomps = 0;
	      do_push = has_ready_packets || put_head.load();
              if(!do_push) push_mutex_check.unlock();
	    }
	  }
	} else {
	  assert(immediate_mode);
	  log_gex_comp.debug() << "comp reply " << ncomps << " failed";
	  // failed - always go to the poller after hitting backpressure
	  if(first_fail_time < 0)
	    first_fail_time = Clock::current_time_in_nanoseconds();
          push_mutex_check.unlock();
          request_push(true /*force_critical*/);
	  return;
	}
      } while(ncomps > 0);

      if(requeue) {
	assert(!do_push);
        push_mutex_check.unlock();
        request_push(false /*!force_critical*/);
      }

      if(!do_push)
	return;
    }

    // next up, the headers of any completed puts - only we can dequeue, so grab
    //  a copy of the head and walk the list without a lock
    PendingPutHeader *orig_put = put_head.load_acquire();
    if(orig_put) {
      PendingPutHeader *cur_put = orig_put;
      PendingPutHeader *prev_put = orig_put; // used if we fall off end
      while(cur_put) {
        gex_Event_t *lc_opt = GEX_EVENT_NOW;  // insist on local copy of header
        gex_Flags_t flags = 0;
        if(immediate_mode) flags |= GEX_FLAG_IMMEDIATE;

        int ret = GASNetEXHandlers::send_request_put_header(internal->eps[src_ep_index],
                                                            tgt_rank,
                                                            tgt_ep_index,
                                                            cur_put->arg0,
                                                            cur_put->hdr_data,
                                                            cur_put->hdr_size,
                                                            cur_put->tgt_ptr,
                                                            cur_put->payload_bytes,
                                                            lc_opt,
                                                            flags);
        if(ret == GASNET_OK) {
          // move on to the next packet (we'll free things once we've actually
          //  removed them from the list)
          prev_put = cur_put;
          cur_put = cur_put->next_put.load_acquire();

          // also stop if we're out of time
          if(work_until.is_expired()) break;
        } else {
          // failed, stop trying
          assert(immediate_mode);
          log_gex.debug() << "failed to send put header";
          if(first_fail_time < 0)
            first_fail_time = Clock::current_time_in_nanoseconds();
          break;
        }
      }

      // if we sent anything, we need to remove it from the list, which
      //  requires the lock
      if(cur_put != orig_put) {
        bool now_empty = false;
        bool just_replies = false;
        {
          AutoLock<> al(mutex);

          if(cur_put) {
            // didn't consume all, so we just need to update the head (tail is
            //  still valid)
            put_head.store_release(cur_put);
          } else {
            if(put_tailp == &prev_put->next_put) {
              // tail is the end of our list, so we're now empty
              put_head.store(nullptr);
              put_tailp = &put_head;

              // if we have ready packets, we'll continue on, but if not, we're
              //  either empty or we just have replies, in which case we need
              //  to requeue (but not continue on to trying to send packets)
              if(!has_ready_packets) {
                if(comp_reply_count.load() == 0) {
                  now_empty = true;
                  push_mutex_check.unlock();
                } else
                  just_replies = true;
              }
            } else {
              // list has grown, but head is whatever was hooked onto the end
              //  of the last put we did
              PendingPutHeader *new_head = prev_put->next_put.load();
              assert(new_head);
              put_head.store(new_head);
	      // fix up 'cur_put' to the part of the list we didn't do because
	      //   we didn't know existed until now
	      cur_put = new_head;
            }
          }
        }

        // now it's safe to free the put headers we sent
        PendingPutHeader *del_put = orig_put;
        while(del_put != cur_put) {
          PendingPutHeader *next_del = del_put->next_put.load();
          internal->put_alloc.free_obj(del_put);
          del_put = next_del;
        }

        if(just_replies) {
          push_mutex_check.unlock();
          request_push(false /*!force_critical*/);
          return;
        }

        // if removing the entries made us empty, somebody else is going to
        //  requeue us for work and we can't do any more here
        if(now_empty)
          return;
      }

      // finally, if we didn't send all the put headers we knew about, we need
      //  to requeue for later
      if(cur_put) {
        push_mutex_check.unlock();
        request_push(true /*force_critical*/);
        return;
      }
    }

    // get the head of our pbuf list - if it's empty, something's wrong because
    //  we shouldn't have gotten here
    OutbufMetadata *head = first_pbuf.load_acquire();
    assert(head);

    while(true) {
      assert(head->state == OutbufMetadata::STATE_PKTBUF);

      OutbufMetadata *realbuf;
      if(head->is_overflow) {
	realbuf = head->realbuf.load_acquire();
	// if this is an overflow buf and we don't have a backing realbuf yet,
	//  we can't send anything - always send to the poller so that we
	//  don't waste injector time
	// TODO: safely put this xpair to sleep?
	if(!realbuf) {
	  log_gex_xpair.debug() << "re-enqueue (overflow stall) " << this;
	  if(first_fail_time < 0)
	    first_fail_time = Clock::current_time_in_nanoseconds();
          push_mutex_check.unlock();
          request_push(true /*force_critical*/);
	  return;
	}
      } else {
	// we are the real buffer
	realbuf = head;
      }

      // grab snapshot of the ready packet count, synchronizing the pkt types
      //  for any packet that's ready at this instant
      int ready_packets = head->pktbuf_ready_packets.load_acquire();
      while(head->pktbuf_sent_packets < ready_packets) {
	bool pkt_sent = false;
        bool force_critical = false;
	OutbufMetadata::PktType pkttype = head->pktbuf_pkt_types[head->pktbuf_sent_packets].load();

	// see if we can batch multiple messages into a single packet
	// has to be enabled, and the first packet has to be INLINE or RGET
	int batch_size = 1;
	static const int MAX_BATCH_SIZE = 16;
	if(internal->module->cfg_batch_messages &&
	   ((pkttype == OutbufMetadata::PKTTYPE_INLINE) ||
	    (pkttype == OutbufMetadata::PKTTYPE_RGET))) {
	  while(((head->pktbuf_sent_packets + batch_size) < ready_packets) &&
		(batch_size < MAX_BATCH_SIZE)) {
	    OutbufMetadata::PktType pkttype2 =
	      head->pktbuf_pkt_types[head->pktbuf_sent_packets + batch_size].load();
	    if((pkttype2 == OutbufMetadata::PKTTYPE_INLINE) ||
	       (pkttype2 == OutbufMetadata::PKTTYPE_RGET)) {
	      // add it and keep going
	      batch_size++;
	      continue;
	    }
	    if(pkttype2 == OutbufMetadata::PKTTYPE_INLINE_SHORT) {
	      // add it but then we have to stop
	      batch_size++;
	      break;
	    }
	    if((pkttype2 == OutbufMetadata::PKTTYPE_LONG) ||
               (pkttype2 == OutbufMetadata::PKTTYPE_PUT)) {
	      // can't be part of a batch
	      break;
	    }
	    assert(0);
	  }
	}

	if(head->is_overflow) {
	  // we might have to do the copy of data from the ovbuf to realbuf
	  //  here
	  for(int i = 0; i < batch_size; i++) {
	    int pktidx = head->pktbuf_sent_packets + i;
	    OutbufMetadata::PktType realtype = realbuf->pktbuf_pkt_types[pktidx].load_acquire();
	    if(realtype == OutbufMetadata::PKTTYPE_INVALID) {
              // attempt to perform late copy - use CAS to avoid race with
              //  resolve copy
              if(realbuf->pktbuf_pkt_types[pktidx].compare_exchange(realtype,
                                                                    OutbufMetadata::PKTTYPE_COPY_IN_PROGRESS)) {
                uintptr_t pktstart = ((pktidx > 0) ?
				        head->pktbuf_pkt_ends[pktidx - 1] :
				        0);
                uintptr_t pktend = head->pktbuf_pkt_ends[pktidx];
                log_gex_obmgr.debug() << "late copy: " << realbuf
                                      << " " << pktstart << " " << pktend;
                memcpy(reinterpret_cast<void *>(realbuf->baseptr + pktstart),
                       reinterpret_cast<const void *>(head->baseptr + pktstart),
                       pktend - pktstart);
                OutbufMetadata::PktType pkttype2 = head->pktbuf_pkt_types[pktidx].load();
                realbuf->pktbuf_pkt_types[pktidx].store_release(pkttype2);
                realtype = pkttype2;
              }
            }
	    if(realtype == OutbufMetadata::PKTTYPE_COPY_IN_PROGRESS) {
              // stop batch here because we don't want to wait for the
              //  already-started copy
              log_gex_obmgr.debug() << "batch shortened due to copy in progress: pktidx=" << pktidx;
              batch_size = i;
              break;
            }
#ifdef DEBUG_REALM
            {
	      OutbufMetadata::PktType pkttype2 = head->pktbuf_pkt_types[pktidx].load();
	      assert(realtype == pkttype2);
	    }
#endif
	  }
	}

        bool batch_attempted = false;
	if(batch_size > 1) {
	  // the minimum size we want is to send TWO packets (if one, why batch?)
          //  - max is all of them (watch out for INLINE_SHORT at end)
	  uintptr_t batch_startofs = head->pktbuf_sent_offset;
	  int first_idx = head->pktbuf_sent_packets;
	  int last_idx = head->pktbuf_sent_packets + batch_size - 1;
	  size_t max_size;
	  if(head->pktbuf_pkt_types[last_idx].load() != OutbufMetadata::PKTTYPE_INLINE_SHORT) {
	    // simple - just get from pkt_ends
	    max_size = head->pktbuf_pkt_ends[last_idx] - batch_startofs;
	  } else {
	    // have to read the info to get the actual size
	    const gex_AM_Arg_t *info =
	      reinterpret_cast<const gex_AM_Arg_t *>(realbuf->baseptr +
						     head->pktbuf_pkt_ends[last_idx - 1]);
	    size_t hdr_bytes = (info[0] & 0x3f) << 2;
	    size_t payload_bytes = info[0] >> 6;

	    max_size = (head->pktbuf_pkt_ends[last_idx - 1] +
			roundup_pow2(hdr_bytes +
				     2*sizeof(gex_AM_Arg_t), 16) +
			roundup_pow2(payload_bytes, 16)) - batch_startofs;
	  }
          // clamp to max_size if it was shortened due to an INLINE_SHORT
	  size_t min_size = std::min((head->pktbuf_pkt_ends[first_idx + 1] -
                                      batch_startofs),
                                     max_size);

	  const void *payload_data =
	    reinterpret_cast<const void *>(realbuf->baseptr + batch_startofs);

	  gex_Event_t done = GEX_EVENT_INVALID;
	  gex_Event_t *lc_opt = &done;

	  gex_Flags_t flags = 0;
	  if(immediate_mode) flags |= GEX_FLAG_IMMEDIATE;

	  gex_AM_SrcDesc_t sd = GEX_AM_SRCDESC_NO_OP;
	  // double-check that our size is acceptable for a client-allocated
	  //  message - messages on the very limit of fitting into a medium
          //  payload may not work as a batch
	  size_t max_payload =
	    GASNetEXHandlers::max_request_medium(internal->eps[src_ep_index],
						 tgt_rank,
						 tgt_ep_index,
						 2 * sizeof(gex_AM_Arg_t), /* header_size */
						 lc_opt,
						 GEX_FLAG_AM_PREPARE_LEAST_CLIENT);
	  if(min_size <= max_payload) {
            batch_attempted = true;

	    sd = GASNetEXHandlers::prepare_request_batch(internal->eps[src_ep_index],
							 tgt_rank,
							 tgt_ep_index,
							 payload_data,
							 min_size,
							 max_size,
							 lc_opt,
							 flags);

            if(sd != GEX_AM_SRCDESC_NO_OP) {
              // success - let's see how much space we were given
              size_t act_size = gex_AM_SrcDescSize(sd);
              if(act_size < max_size) {
                // not as much as we asked for - have to reduce batch size
                int reduced_count = 1;
                while(head->pktbuf_pkt_ends[first_idx + reduced_count] <=
                      (batch_startofs + act_size)) {
                  reduced_count++;
                  assert(reduced_count < batch_size);
                }
                // should always be at least two messages and smaller than the
                //  the overall batch size (which means no INLINE_SHORTs to worry
                //  about)
                assert((reduced_count >= 2) && (reduced_count < batch_size));
                last_idx = first_idx + reduced_count - 1;
                max_size = (head->pktbuf_pkt_ends[last_idx] - batch_startofs);
                batch_size = reduced_count;
              }

              uint32_t cksum = 0;
              if(internal->module->cfg_do_checksums) {
                uint32_t accum = 0xFFFFFFFF;
                accum = crc32c_accumulate(accum, &batch_size, sizeof(batch_size));
                accum = crc32c_accumulate(accum, &max_size, sizeof(max_size));
                accum = crc32c_accumulate(accum, payload_data, max_size);
                cksum = ~accum;

#ifdef VERIFY_BATCH_CONTENTS_CRCS
                // sanity-check the checksums of individual packets in the batch
                const char *baseptr = static_cast<const char *>(payload_data);
                for(int i = 0; i < batch_size; i++) {
                  gex_AM_Arg_t info[2];
                  memcpy(info, baseptr, 2*sizeof(gex_AM_Arg_t));

                  size_t hdr_bytes = (info[0] & 0x3f) << 2;
                  size_t payload_bytes = info[0] >> 6;
                  gex_AM_Arg_t msg_arg0 = info[1];

                  size_t pad_hdr_bytes = roundup_pow2(hdr_bytes + 2*sizeof(gex_AM_Arg_t),
                                                      16);

                  uint32_t expcrc;
                  memcpy(&expcrc, baseptr + 2*sizeof(gex_AM_Arg_t) + hdr_bytes - sizeof(uint32_t), sizeof(uint32_t));

                  if(payload_bytes == 0) {
                    uint32_t actcrc = compute_packet_crc(msg_arg0,
                                                         baseptr + 2*sizeof(gex_AM_Arg_t),
                                                         hdr_bytes - sizeof(uint32_t),
                                                         0, 0);
                    if(expcrc != actcrc) {
                      log_gex.fatal() << "CRC SHORT " << i << " " << static_cast<const void *>(baseptr)
                                      << " " << head << " " << realbuf
                                      << " " << std::hex << expcrc << " " << actcrc << std::dec;
                      abort();
                    }
                    baseptr += pad_hdr_bytes;
                  } else if(payload_bytes < ((1U << 22) - 1)) {
                    // medium message
                    uint32_t actcrc = compute_packet_crc(msg_arg0,
                                                         baseptr + 2*sizeof(gex_AM_Arg_t),
                                                         hdr_bytes - sizeof(uint32_t),
                                                         baseptr + pad_hdr_bytes,
                                                         payload_bytes);
                    if(expcrc != actcrc) {
                      log_gex.fatal() << "CRC MEDIUM " << i << " " << static_cast<const void *>(baseptr)
                                      << " " << head << " " << realbuf
                                      << " " << std::hex << expcrc << " " << actcrc << std::dec;
                      abort();
                    }
                    baseptr += pad_hdr_bytes + roundup_pow2(payload_bytes, 16);
                  } else {
                    // reverse get
                    XmitSrcDestPair::LongRgetData extra;
                    memcpy(&extra, baseptr + pad_hdr_bytes, sizeof(XmitSrcDestPair::LongRgetData));
                    uint32_t actcrc = compute_packet_crc(msg_arg0,
                                                         baseptr + 2*sizeof(gex_AM_Arg_t),
                                                         hdr_bytes - sizeof(uint32_t),
                                                         0,
                                                         extra.payload_bytes);
                    if(expcrc != actcrc) {
                      log_gex.fatal() << "CRC RGET " << i << " " << static_cast<const void *>(baseptr)
                                      << " " << head << " " << realbuf
                                      << " " << std::hex << expcrc << " " << actcrc << std::dec;
                      abort();
                    }
                    baseptr += pad_hdr_bytes + roundup_pow2(sizeof(XmitSrcDestPair::LongRgetData), 16);
                  }
                }
#endif
              }
              GASNetEXHandlers::commit_request_batch(sd, batch_size, cksum,
                                                     max_size);

              pkt_sent = true;
              for(int i = 0; i < batch_size; i++)
                realbuf->pktbuf_pkt_types[first_idx + i].store(OutbufMetadata::PKTTYPE_INVALID);
              head->pktbuf_sent_offset = head->pktbuf_pkt_ends[last_idx];
              head->pktbuf_sent_packets += batch_size;
              head->pktbuf_use_count++;  // expect decrement on local comp
              packets_sent.fetch_add(batch_size);

              GASNetEXEvent *ev = internal->event_alloc.alloc_obj();
              ev->set_event(done);
              ev->set_pktbuf(realbuf);
              internal->poller.add_pending_event(ev);
            } else {
              assert(immediate_mode);  // should not happen without immediate
              log_gex_xpair.info() << "xpair retry: xpair=" << this;
            }
          }
        }

        // if we didn't have multiple packets to batch up, or they couldn't
        //  fit in a batch, try sending just the first packet
        if((batch_size > 0) && !batch_attempted) {
	  switch(pkttype) {
	  case OutbufMetadata::PKTTYPE_INLINE:
	  case OutbufMetadata::PKTTYPE_INLINE_SHORT:
	    {
	      const gex_AM_Arg_t *info =
		reinterpret_cast<const gex_AM_Arg_t *>(realbuf->baseptr +
						       head->pktbuf_sent_offset);
	      const void *hdr_data =
		reinterpret_cast<const void *>(realbuf->baseptr +
					       head->pktbuf_sent_offset +
					       2*sizeof(gex_AM_Arg_t));
	      size_t hdr_bytes = (info[0] & 0x3f) << 2;
	      size_t payload_bytes = info[0] >> 6;
	      gex_AM_Arg_t arg0 = info[1];
	      if(payload_bytes == 0) {
		// short message
		gex_Flags_t flags = 0;
		if(immediate_mode) flags |= GEX_FLAG_IMMEDIATE;

		int ret = GASNetEXHandlers::send_request_short(internal->eps[src_ep_index],
							       tgt_rank,
							       tgt_ep_index,
							       arg0,
							       hdr_data,
							       hdr_bytes,
							       flags);
		if(ret == GASNET_OK) {
		  pkt_sent = true;
		  realbuf->pktbuf_pkt_types[head->pktbuf_sent_packets].store(OutbufMetadata::PKTTYPE_INVALID);
		  head->pktbuf_sent_offset = head->pktbuf_pkt_ends[head->pktbuf_sent_packets];
		  head->pktbuf_sent_packets++;
		  // no need to increment use count for a short
		  packets_sent.fetch_add(1);
		} else {
		  assert(immediate_mode);  // should not happen without immediate
		  log_gex_xpair.info() << "xpair retry: xpair=" << this;
		}
	      } else {
		const void *payload_data =
		  reinterpret_cast<const void *>(realbuf->baseptr +
						 head->pktbuf_sent_offset +
						 roundup_pow2(hdr_bytes +
							      2*sizeof(gex_AM_Arg_t), 16));

		gex_Flags_t flags = 0;
		if(immediate_mode) flags |= GEX_FLAG_IMMEDIATE;

		gex_Event_t done = GEX_EVENT_INVALID;
		gex_Event_t *lc_opt = &done;

#ifdef DEBUG_REALM
                {
                  size_t max_payload =
                    GASNetEXHandlers::max_request_medium(internal->eps[src_ep_index],
                                                         tgt_rank,
                                                         tgt_ep_index,
                                                         hdr_bytes,
                                                         lc_opt,
                                                         flags);
                  if(payload_bytes > max_payload) {
                    log_gex_xpair.fatal() << "medium payload too large!  src="
                                          << Network::my_node_id << "/" << src_ep_index
                                          << " tgt=" << tgt_rank << "/" << tgt_ep_index
                                          << " max=" << max_payload << " act=" << payload_bytes;
                    abort();
                  }
                }
#endif

		int ret = GASNetEXHandlers::send_request_medium(internal->eps[src_ep_index],
								tgt_rank,
								tgt_ep_index,
								arg0,
								hdr_data,
								hdr_bytes,
								payload_data,
								payload_bytes,
								lc_opt,
								flags);
		if(ret == GASNET_OK) {
		  pkt_sent = true;
		  realbuf->pktbuf_pkt_types[head->pktbuf_sent_packets].store(OutbufMetadata::PKTTYPE_INVALID);
		  head->pktbuf_sent_offset = head->pktbuf_pkt_ends[head->pktbuf_sent_packets];
		  head->pktbuf_sent_packets++;
		  head->pktbuf_use_count++;  // expect decrement on local comp
		  packets_sent.fetch_add(1);

		  GASNetEXEvent *ev = internal->event_alloc.alloc_obj();
		  ev->set_event(done);
		  ev->set_pktbuf(realbuf);
		  internal->poller.add_pending_event(ev);
		} else {
		  assert(immediate_mode);  // should not happen without immediate
		  log_gex_xpair.info() << "xpair retry: xpair=" << this;
		}
	      }
	      break;
	    }

	  case OutbufMetadata::PKTTYPE_LONG:
	    {
	      const gex_AM_Arg_t *info =
		reinterpret_cast<const gex_AM_Arg_t *>(realbuf->baseptr +
						       head->pktbuf_sent_offset);
	      const void *hdr_data =
		reinterpret_cast<const void *>(realbuf->baseptr +
					       head->pktbuf_sent_offset +
					       2*sizeof(gex_AM_Arg_t));
	      size_t hdr_bytes = (info[0] & 0x3f) << 2;
	      gex_AM_Arg_t arg0 = info[1];

	      const LongRgetData *extra =
		reinterpret_cast<const LongRgetData *>(realbuf->baseptr +
						       head->pktbuf_sent_offset +
						       roundup_pow2(hdr_bytes +
								    2*sizeof(gex_AM_Arg_t), 16));

	      gex_Flags_t flags = 0;
	      if(immediate_mode) flags |= GEX_FLAG_IMMEDIATE;

	      // use a gex_Event_t for either local completion or databuf usecount
	      PendingCompletion *local_comp = internal->extract_arg0_local_comp(arg0);
	      gex_Event_t done = GEX_EVENT_INVALID;
	      gex_Event_t *lc_opt;
	      if(local_comp || extra->l.databuf)
		lc_opt = &done;
	      else
		lc_opt = GEX_EVENT_GROUP; // don't care

#ifdef DEBUG_REALM
              {
                size_t max_payload =
                  GASNetEXHandlers::max_request_long(internal->eps[src_ep_index],
                                                     tgt_rank,
                                                     tgt_ep_index,
                                                     hdr_bytes,
                                                     lc_opt,
                                                     flags);
                if(extra->payload_bytes > max_payload) {
                  log_gex_xpair.fatal() << "long payload too large!  src="
                                        << Network::my_node_id << "/" << src_ep_index
                                        << " tgt=" << tgt_rank << "/" << tgt_ep_index
                                        << " max=" << max_payload << " act=" << extra->payload_bytes;
                  abort();
                }
              }
#endif

	      int ret = GASNetEXHandlers::send_request_long(internal->eps[src_ep_index],
							    tgt_rank,
							    tgt_ep_index,
							    arg0,
							    hdr_data,
							    hdr_bytes,
							    extra->payload_base,
							    extra->payload_bytes,
							    lc_opt,
							    flags,
							    extra->dest_addr);
	      if(ret == GASNET_OK) {
		pkt_sent = true;
		realbuf->pktbuf_pkt_types[head->pktbuf_sent_packets].store(OutbufMetadata::PKTTYPE_INVALID);
		head->pktbuf_sent_offset = head->pktbuf_pkt_ends[head->pktbuf_sent_packets];
		head->pktbuf_sent_packets++;
		// no need to increment use count for a long (header is copied)
		packets_sent.fetch_add(1);

		if(local_comp || extra->l.databuf) {
		  GASNetEXEvent *ev = internal->event_alloc.alloc_obj();
		  ev->set_event(done);
		  ev->set_local_comp(local_comp);
		  ev->set_databuf(extra->l.databuf);
		  internal->poller.add_pending_event(ev);
		}
	      } else {
		assert(immediate_mode);  // should not happen without immediate
		log_gex_xpair.info() << "xpair retry: xpair=" << this;
	      }
	      break;
	    }

	  case OutbufMetadata::PKTTYPE_RGET:
	    {
	      const gex_AM_Arg_t *info =
		reinterpret_cast<const gex_AM_Arg_t *>(realbuf->baseptr +
						       head->pktbuf_sent_offset);
	      const void *hdr_data =
		reinterpret_cast<const void *>(realbuf->baseptr +
					       head->pktbuf_sent_offset +
					       2*sizeof(gex_AM_Arg_t));
	      size_t hdr_bytes = (info[0] & 0x3f) << 2;
	      gex_AM_Arg_t arg0 = info[1];

	      const LongRgetData *extra =
		reinterpret_cast<const LongRgetData *>(realbuf->baseptr +
						       head->pktbuf_sent_offset +
						       roundup_pow2(hdr_bytes +
								    2*sizeof(gex_AM_Arg_t), 16));

	      gex_Flags_t flags = 0;
	      if(immediate_mode) flags |= GEX_FLAG_IMMEDIATE;

	      // "local completion" of the rget source data has to be signalled
	      //  by the target, but we can use lc_opt to know the pktbuf
	      //  containing the header is clear for reuse
	      gex_Event_t done = GEX_EVENT_INVALID;
	      gex_Event_t *lc_opt = &done;

	      int ret = GASNetEXHandlers::send_request_rget(internal->prim_tm,
							    tgt_rank,
							    extra->r.tgt_ep_index,
							    arg0,
							    hdr_data,
							    hdr_bytes,
							    extra->r.src_ep_index,
							    extra->payload_base,
							    extra->payload_bytes,
							    lc_opt,
							    flags,
							    extra->dest_addr);
	      if(ret == GASNET_OK) {
		pkt_sent = true;
		realbuf->pktbuf_pkt_types[head->pktbuf_sent_packets].store(OutbufMetadata::PKTTYPE_INVALID);
		head->pktbuf_sent_offset = head->pktbuf_pkt_ends[head->pktbuf_sent_packets];
		head->pktbuf_sent_packets++;
		head->pktbuf_use_count++;  // expect decrement on local comp
		packets_sent.fetch_add(1);

		GASNetEXEvent *ev = internal->event_alloc.alloc_obj();
		ev->set_event(done);
		ev->set_pktbuf(realbuf);
		internal->poller.add_pending_event(ev);
	      } else {
		assert(immediate_mode);  // should not happen without immediate
		log_gex_xpair.info() << "xpair retry: xpair=" << this;
	      }
	      break;
	    }

          case OutbufMetadata::PKTTYPE_PUT:
            {
              const PutMetadata *meta =
                reinterpret_cast<const PutMetadata *>(realbuf->baseptr +
                                                      head->pktbuf_sent_offset);

              gex_Flags_t flags = 0;
              if(immediate_mode) flags |= GEX_FLAG_IMMEDIATE;

              // local completion just requires local completion of the payload,
              //  as we've already made a copy of the header, but only ask for
              //  it if the message needs it
              gex_Event_t lc_event = GEX_EVENT_INVALID;
              gex_Event_t *lc_opt = (meta->put->local_comp ?
                                       &lc_event :
                                       GEX_EVENT_DEFER);

              gex_TM_t pair = gex_TM_Pair(internal->eps[src_ep_index],
                                          tgt_ep_index);

              gex_Event_t rc_event = GEX_EVENT_NO_OP;
#ifndef REALM_GEX_RMA_HONORS_IMMEDIATE_FLAG
              // conduit isn't promising to return right away in the face of
              //  back-pressure, so don't even try in immediate mode
              if(immediate_mode) {
                // further retries in immediate mode won't help either...
                force_critical = true;
              } else
#endif
              {
                rc_event = gex_RMA_PutNB(pair,
                                         tgt_rank,
                                         reinterpret_cast<void *>(meta->dest_addr),
                                         const_cast<void *>(meta->src_addr),
                                         meta->payload_bytes,
                                         lc_opt,
                                         flags);
              }

              if(rc_event != GEX_EVENT_NO_OP) {
                // successful injection
                pkt_sent = true;

                GASNetEXEvent *leaf = 0;
                // local completion (if needed)
                if(meta->put->local_comp) {
                  GASNetEXEvent *ev = internal->event_alloc.alloc_obj();
                  ev->set_event(lc_event);
                  ev->set_local_comp(meta->put->local_comp);
                  internal->poller.add_pending_event(ev);
                  leaf = ev;  // must be connected to root event below
                }

                // remote completion (always needed)
                {
                  GASNetEXEvent *ev = internal->event_alloc.alloc_obj();
                  ev->set_event(rc_event);
                  ev->set_put(meta->put);
                  if(leaf)
                    ev->set_leaf(leaf);
                  internal->poller.add_pending_event(ev);
                }

		realbuf->pktbuf_pkt_types[head->pktbuf_sent_packets].store(OutbufMetadata::PKTTYPE_INVALID);
		head->pktbuf_sent_offset = head->pktbuf_pkt_ends[head->pktbuf_sent_packets];
		head->pktbuf_sent_packets++;
		// no need to increment use count for a put (done with metadata)
		packets_sent.fetch_add(1);
              } else {
		assert(immediate_mode);  // should not happen without immediate
		log_gex_xpair.info() << "xpair retry: xpair=" << this;
	      }
	      break;
	    }

	  default: assert(0);
	  }
	}

	if(pkt_sent) {
	  first_fail_time = -1;
	  // switch back to immediate mode after a successful packet send
	  //  unless it's disabled
	  if(internal->module->cfg_crit_timeout >= 0)
	    immediate_mode = true;
	} else {
	  // if we failed to send a packet, stop trying and reenqueue ourselves
          if(force_critical) {
            first_fail_time = 0; // so long ago we're guaranteed to be critical
          } else {
            if(first_fail_time < 0)
              first_fail_time = Clock::current_time_in_nanoseconds();
          }
	  // always go to the poller after hitting backpressure
	  log_gex_xpair.debug() << "re-enqueue (send failed) " << this;
          push_mutex_check.unlock();
          request_push(true /*force_critical*/);
	  return;
	}

	// if time has expired and we didn't get through what we knew about,
	//  definitely requeue ourselves
	if(work_until.is_expired() &&
	   (head->pktbuf_sent_packets < ready_packets)) {
	  // we made progress, so use the injector next if we can
	  log_gex_xpair.debug() << "re-enqueue (expired) " << this;
          push_mutex_check.unlock();
          request_push(false /*!force_critical*/);
	  return;
	}
      }

      // if we think we consumed this entire pktbuf, take the mutex and
      //  try to move to the next one if it exists
      OutbufMetadata *new_head = nullptr;
      bool requeue = false;
      {
	AutoLock<> al(mutex);

	// packets can't be added while we hold the mutex, so check our sent
	//  count against the total
	if(head->pktbuf_sent_packets == head->pktbuf_total_packets.load()) {
	  // nothing left here - check if this is the current pbuf
	  if(head == cur_pbuf) {
	    // still writing to this one, so we're done for now - no requeue
	    has_ready_packets = false;
	    requeue = put_head.load() || (comp_reply_count.load() != 0);
            push_mutex_check.unlock();
	  } else {
	    // we can remove the head and work on the next one
	    new_head = head->nextbuf;
	    first_pbuf.store(new_head);
	  }
	} else {
	  // more stuff in this pktbuf - requeue if we see any have become
	  //  ready while we were pushing packets
	  if(head->pktbuf_ready_packets.load() > head->pktbuf_sent_packets) {
	    requeue = true;
	  } else {
	    has_ready_packets = false;
	    requeue = put_head.load() || (comp_reply_count.load() != 0);
	  }
          push_mutex_check.unlock();
	}
      }

      if(new_head) {
	// close out the old head - it'll be free'd once all uses are done
	head->pktbuf_close();
	head = new_head;
      } else {
	// done for now, requeue if we know we're still nonempty
	if(requeue) {
	  // go to poller so that we don't waste injector time while packets
	  //  are being committed
	  log_gex_xpair.debug() << "re-enqueue (refill race) " << this;
          request_push(true /*force_critical*/);
	}
	return;
      }
    }
  }

  long long XmitSrcDestPair::time_since_failure() const
  {
    if(first_fail_time < 0)
      return -1;
    else
      return (Clock::current_time_in_nanoseconds() - first_fail_time);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class XmitSrc
  //

  XmitSrc::XmitSrc(GASNetEXInternal *_internal, gex_EP_Index_t _src_ep_index)
    : internal(_internal)
    , src_ep_index(_src_ep_index)
  {
    // allocate enough space to store pointers for the max number of endpoints
    //  for all target ranks, but start them all out as nullptrs
    size_t count = (internal->prim_size *
		    GASNET_MAXEPS);
    pairs = new atomic<XmitSrcDestPair *>[count];

    for(size_t i = 0; i < count; i++)
      pairs[i].store(nullptr);
  }

  XmitSrc::~XmitSrc()
  {
    size_t count = (internal->prim_size *
		    GASNET_MAXEPS);
    for(size_t i = 0; i < count; i++) {
      XmitSrcDestPair *p = pairs[i].load();
      if(p) delete p;
    }
    delete[] pairs;
  }

  XmitSrcDestPair *XmitSrc::lookup_pair(gex_Rank_t tgt_rank,
					gex_EP_Index_t tgt_ep_index)
  {
    assert(tgt_rank < internal->prim_size);
    assert(tgt_ep_index < GASNET_MAXEPS);
    // do ep 0 for all targets, then ep 1, ... (better locality for common case)
    size_t index = (tgt_ep_index * internal->prim_size) + tgt_rank;

    // fast case - pointer already exists
    XmitSrcDestPair *p = pairs[index].load_acquire();
    if(p) return p;

    // slow case - allocate a new one and try to put it in, detecting races
    XmitSrcDestPair *newp = new XmitSrcDestPair(internal, src_ep_index,
						tgt_rank, tgt_ep_index);
    assert(newp != 0);
    if(pairs[index].compare_exchange(p, newp)) {
      // success - the one we allocated is the right one
      return newp;
    } else {
      // failure - somebody else got in first, so free ours and use that one
      delete newp;
      return p;
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNetEXEvent
  //

  GASNetEXEvent::GASNetEXEvent()
    : event(GEX_EVENT_INVALID)
    , local_comp(nullptr)
    , pktbuf(nullptr)
    , databuf(nullptr)
    , rget(nullptr)
    , put(nullptr)
    , leaf(nullptr)
  {}

  gex_Event_t GASNetEXEvent::get_event() const
  {
    return event;
  }

  GASNetEXEvent& GASNetEXEvent::set_event(gex_Event_t _event)
  {
    event = _event;
    return *this;
  }

  GASNetEXEvent& GASNetEXEvent::set_local_comp(PendingCompletion *_local_comp)
  {
    local_comp = _local_comp;
    return *this;
  }

  GASNetEXEvent& GASNetEXEvent::set_pktbuf(OutbufMetadata *_pktbuf)
  {
    pktbuf = _pktbuf;
    return *this;
  }

  GASNetEXEvent& GASNetEXEvent::set_databuf(OutbufMetadata *_databuf)
  {
    databuf = _databuf;
    return *this;
  }

  GASNetEXEvent& GASNetEXEvent::set_rget(PendingReverseGet *_rget)
  {
    rget = _rget;
    return *this;
  }

  GASNetEXEvent& GASNetEXEvent::set_put(PendingPutHeader *_put)
  {
    put = _put;
    return *this;
  }

  GASNetEXEvent& GASNetEXEvent::set_leaf(GASNetEXEvent *_leaf)
  {
    leaf = _leaf;
    return *this;
  }

  void GASNetEXEvent::propagate_to_leaves()
  {
    if(leaf)
      leaf->event = GEX_EVENT_NO_OP;
  }

  void GASNetEXEvent::trigger(GASNetEXInternal *internal)
  {
    event = GEX_EVENT_INVALID;
    if(local_comp)
      internal->compmgr.invoke_completions(local_comp,
					   true /*local*/, false /*!remote*/);
    if(pktbuf)
      pktbuf->dec_usecount();
    if(databuf)
      databuf->dec_usecount();
    if(rget)
      rget->rgetter->reverse_get_complete(rget);
    if(put)
      put->xpair->enqueue_put_header(put);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNetEXInjector
  //

  GASNetEXInjector::GASNetEXInjector(GASNetEXInternal *_internal)
    : BackgroundWorkItem("gex-inj")
    , internal(_internal)
  {}

  void GASNetEXInjector::add_ready_xpair(XmitSrcDestPair *xpair)
  {
    log_gex_xpair.info() << "xpair ready: xpair=" << xpair;

    // take mutex to enqueue pair
    bool new_work;
    {
      AutoLock<> al(mutex);
      new_work = ready_xpairs.empty();
      ready_xpairs.push_back(xpair);
    }

    if(new_work)
      make_active();
  }

  bool GASNetEXInjector::has_work_remaining()
  {
    AutoLock<> al(mutex);
    return !ready_xpairs.empty();
  }

  bool GASNetEXInjector::do_work(TimeLimit work_until)
  {
    // we're not supposed to end up handling AMs, but set this just in case
    //  we do
    ThreadLocal::gex_work_until = &work_until;

    // take the first xpair on the list, and immediately requeue if there
    //  are other ready xpairs that another thread can work on
    bool more_work;
    XmitSrcDestPair *xpair;
    {
      AutoLock<> al(mutex);
      xpair = ready_xpairs.pop_front();
      more_work = !ready_xpairs.empty();
    }
    assert(xpair);
    if(more_work)
      make_active();

    log_gex_xpair.info() << "xpair active: xpair=" << xpair;

    // now tell the xpair to inject as many packets as it can until it
    //   runs out of time or hits backpressure
    xpair->push_packets(true /*immediate_mode*/, work_until);

    ThreadLocal::gex_work_until = nullptr;

    // push_packets will have requeued us already if needed
    return false;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNetEXPoller
  //

  GASNetEXPoller::GASNetEXPoller(GASNetEXInternal *_internal)
    : BackgroundWorkItem("gex-poll")
    , internal(_internal)
    , shutdown_flag(false)
    , shutdown_cond(mutex)
    , pollwait_flag(false)
    , pollwait_cond(mutex)
  {}

  void GASNetEXPoller::begin_polling()
  {
    make_active();
  }

  void GASNetEXPoller::end_polling()
  {
    AutoLock<> al(mutex);

    assert(!shutdown_flag.load());
    shutdown_flag.store(true);
    shutdown_cond.wait();
  }

  void GASNetEXPoller::add_critical_xpair(XmitSrcDestPair *xpair)
  {
    log_gex_xpair.info() << "xpair critical: xpair=" << xpair;

    // take mutex to enqueue pair
    {
      AutoLock<> al(mutex);
      critical_xpairs.push_back(xpair);
    }
    // no need to signal because the poller is always awake
  }

  void GASNetEXPoller::add_pending_event(GASNetEXEvent *event)
  {
    // take mutex to enqueue event
    {
      AutoLock<> al(mutex);
      pending_events.push_back(event);
    }
    // no need to signal because the poller is always awake
  }

  bool GASNetEXPoller::has_work_remaining()
  {
    AutoLock<> al(mutex);
    return (!critical_xpairs.empty() || !pending_events.empty());
  }

  bool GASNetEXPoller::do_work(TimeLimit work_until)
  {
    ThreadLocal::gex_work_until = &work_until;

    // we're going to try to be frugal about acquiring mutexes here, so peek
    //  ahead in the critical xpair list to avoid the extra mutex acquire that
    //  would observe an empty list
    bool have_crit_xpairs = false;

    // first go through all(?) the pending events to see if any have
    //  finished
    {
      GASNetEXEvent::EventList to_check, still_pending, to_complete;

      // atomically grab all the known ones so that we don't have to hold the
      //  mutex while we're testing the events
      // don't wait on contention though - we'll get to events and critical
      //  xpairs next time
      if(mutex.trylock()) {
	to_check.swap(pending_events);
        have_crit_xpairs = !critical_xpairs.empty();
        mutex.unlock();
      }

      // go through events in order, either trigger or move to 'still_pending'
      while(!to_check.empty()) {
	GASNetEXEvent *ev = to_check.pop_front();
        // if the GASNet event is GEX_EVENT_NO_OP, that means we were a leaf
        //  event and the root event has already been successfully tested,
        //  so we automatically succeed (it would be illegal to check again)
        gex_Event_t gev = ev->get_event();
        int ret = ((gev == GEX_EVENT_NO_OP) ?
                     GASNET_OK :
                     gex_Event_Test(gev));
	switch(ret) {
	case GASNET_OK:
	  {
            // even if we don't handle callbacks right away, we have to deal
            //  with root/leaf event relationships before we can safely test
            //  any more events
            ev->propagate_to_leaves();
            to_complete.push_back(ev);
	    break;
	  }
	case GASNET_ERR_NOT_READY:
	  {
	    still_pending.push_back(ev);
	    break;
	  }
	default:
	  {
	    log_gex.fatal() << "wait = " << ret;
	    abort();
	  }
	}
      }

      // if some events remain, put them back on the _front_ of the list
      if(!still_pending.empty()) {
	AutoLock<> al(mutex);
	still_pending.absorb_append(pending_events);
	still_pending.swap(pending_events);
      }

      // if we have any completed events, give them to the completer
      if(!to_complete.empty())
        internal->completer.add_ready_events(to_complete);
    }

    // try to push packets for any xmit pairs that are critical (i.e. cannot
    //  use immediate mode)
    while(have_crit_xpairs) {
      XmitSrcDestPair *xpair = nullptr;

      // don't wait on contention for the mutex - just skip and get it
      //  next time around
      if(mutex.trylock()) {
#ifdef DEBUG_REALM
        assert(!critical_xpairs.empty());
#endif
        xpair = critical_xpairs.pop_front();
        have_crit_xpairs = !critical_xpairs.empty();
        mutex.unlock();
      }
      if(!xpair) break;

      log_gex_xpair.info() << "xpair active: xpair=" << xpair;

      // even a critical xpair will stay in immediate mode until it times out
      bool immediate_mode;
      if(internal->module->cfg_crit_timeout < 0)
	immediate_mode = false;
      else
	immediate_mode = (xpair->time_since_failure() <
			  internal->module->cfg_crit_timeout);

      // if we're not in immediate mode, do some polling to hopefully free up
      //  resources
      if(!immediate_mode)
        gasnet_AMPoll();

      // ask the pair to push packets, it'll requeue itself if needed
      xpair->push_packets(immediate_mode, work_until);

      // if time's expired, stop transmitting and do at least one poll
      if(work_until.is_expired())
	break;
    }

    // sample pollwait flag before we perform gasnet_AMPoll
    bool pollwait_snapshot = pollwait_flag.load();

    // no gex version of this?
    gasnet_AMPoll();

    ThreadLocal::gex_work_until = nullptr;

    // if there was a pollwaiter before we started the poll, we can wake it
    //  now
    if(pollwait_snapshot) {
      AutoLock<> al(mutex);
      pollwait_flag.store(false);
      pollwait_cond.broadcast();
    }

    // if a shutdown has been requested, wake the waiter - if not, requeue
    if(shutdown_flag.load()) {
      AutoLock<> al(mutex);
      shutdown_flag.store(false);
      shutdown_cond.broadcast();
      return false;
    } else
      return true;
  }

  void GASNetEXPoller::wait_for_full_poll_cycle()
  {
    AutoLock<> al(mutex);
    pollwait_flag.store(true);
    pollwait_cond.wait();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNetEXCompleter
  //

  GASNetEXCompleter::GASNetEXCompleter(GASNetEXInternal *_internal)
    : BackgroundWorkItem("gex-complete")
    , internal(_internal)
    , has_work(false)
  {}

  void GASNetEXCompleter::add_ready_events(GASNetEXEvent::EventList& newly_ready)
  {
    bool enqueue = false;

    if(!newly_ready.empty()) {
      AutoLock<> al(mutex);
      // use has_work rather than list emptiness to decide whether to enqueue
      enqueue = !has_work.load();
      has_work.store(true);
      ready_events.absorb_append(newly_ready);
    }

    if(enqueue)
      make_active();
  }

  bool GASNetEXCompleter::has_work_remaining()
  {
    return has_work.load();
  }

  bool GASNetEXCompleter::do_work(TimeLimit work_until)
  {
    // grab all the events but don't clear 'has_work' since we don't want
    //  to be reactivated yet
    GASNetEXEvent::EventList todo;
    {
      AutoLock<> al(mutex);
      todo.swap(ready_events);
    }

    while(!todo.empty()) {
      GASNetEXEvent *ev = todo.pop_front();
      ev->trigger(internal);
      internal->event_alloc.free_obj(ev);

      if(work_until.is_expired())
        break;
    }

    // retake lock to either put back events we didn't get to or clear
    //  'has_work' flag
    bool requeue = false;
    {
      AutoLock<> al(mutex);
      if(todo.empty()) {
        if(ready_events.empty())
          has_work.store(false);
        else
          requeue = true;  // new events showed up
      } else {
        // the events we didn't get to should be at the front of the list
        todo.absorb_append(ready_events);
        ready_events.swap(todo);
        requeue = true;
      }
    }

    return requeue;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ReverseGetter
  //

  ReverseGetter::ReverseGetter(GASNetEXInternal *_internal)
    : BackgroundWorkItem("gex-rget")
    , internal(_internal)
    , head(0)
    , tailp(&head)
  {}

  void ReverseGetter::add_reverse_get(gex_Rank_t srcrank, gex_EP_Index_t src_ep_index,
				      gex_EP_Index_t tgt_ep_index,
				      gex_AM_Arg_t arg0,
				      const void *hdr, size_t hdr_bytes,
				      uintptr_t src_ptr, uintptr_t tgt_ptr,
				      size_t payload_bytes)
  {
    PendingReverseGet *rget = rget_alloc.alloc_obj();
    rget->rgetter = this;
    rget->next_rget = 0;
    rget->srcrank = srcrank;
    rget->src_ep_index = src_ep_index;
    rget->tgt_ep_index = tgt_ep_index;
    rget->arg0 = arg0;
    rget->hdr_size = hdr_bytes;
    assert(hdr_bytes <= PendingReverseGet::MAX_HDR_SIZE);
    memcpy(rget->hdr_data, hdr, hdr_bytes);
    rget->src_ptr = src_ptr;
    rget->tgt_ptr = tgt_ptr;
    rget->payload_bytes = payload_bytes;

    // take mutex to add this to the list - note whether the list was empty
    bool was_empty;
    {
      AutoLock<> al(mutex);
      was_empty = !head;
      *tailp = rget;
      tailp = &rget->next_rget;
    }
    if(was_empty)
      make_active();
  }

  bool ReverseGetter::has_work_remaining()
  {
    AutoLock<> al(mutex);
    return (head != nullptr);
  }

  bool ReverseGetter::do_work(TimeLimit work_until)
  {
    // we're not going to use immedate mode for rgets, so don't have more
    //  than one dequeuer at a time - do this by peeking at the head but
    //  not popping it until we've actually issued the rget
    PendingReverseGet *rget;
    {
      AutoLock<> al(mutex);
      assert(head);
      rget = head;
    }

    while(true) {
      log_gex.info() << "issuing get: "
		     << rget->srcrank << "/" << rget->src_ep_index << "/"
		     << std::hex << rget->src_ptr << std::dec
		     << " -> " << rget->tgt_ep_index << "/"
		     << std::hex << rget->tgt_ptr << std::dec
		     << " size=" << rget->payload_bytes;
      // be careful here - the "target" is the issuer of the get
      gex_TM_t pair = gex_TM_Pair(internal->eps[rget->tgt_ep_index],
				  rget->src_ep_index);
      gex_Event_t done = gex_RMA_GetNB(pair,
				       reinterpret_cast<void *>(rget->tgt_ptr),
				       rget->srcrank,
				       reinterpret_cast<void *>(rget->src_ptr),
				       rget->payload_bytes,
				       0 /*flags*/);

      // once we add the event to the list, rget might be recycled at any
      //  time, so pop it off the list and peek at the next entry (if any)
      //  before we do that
      PendingReverseGet *next_rget;
      {
	AutoLock<> al(mutex);
	assert(head == rget); // this shouldn't have changed
	next_rget = head->next_rget;
	head = next_rget;
	if(!next_rget)
	  tailp = &head;
      }

      GASNetEXEvent *ev = internal->event_alloc.alloc_obj();
      ev->set_event(done);
      ev->set_rget(rget);
      internal->poller.add_pending_event(ev);

      if(next_rget) {
	if(work_until.is_expired()) {
	  // requeue ourselves and stop for now
          return true;
	} else {
	  // keep going with the next one
	  rget = next_rget;
	}
      } else
	break;
    }

    // no work left
    return false;
  }

  void ReverseGetter::reverse_get_complete(PendingReverseGet *rget)
  {
    log_gex.info() << "rget done: " << rget << " " << rget->arg0;

    // now we can pretend we are a normal long message
    gex_AM_Arg_t comp = internal->handle_long(rget->srcrank,
					      rget->arg0,
					      rget->hdr_data,
					      rget->hdr_size,
					      reinterpret_cast<void *>(rget->tgt_ptr),
					      rget->payload_bytes);
    if(comp != 0) {
      XmitSrcDestPair *xpair = internal->xmitsrcs[0]->lookup_pair(rget->srcrank,
								  0 /*prim endpoint*/);
      xpair->enqueue_completion_reply(comp);
#if 0
      GASNetEXHandlers::send_completion_reply(internal->eps[0],
					      rget->srcrank,
					      0 /*prim endpoint*/,
					      &comp,
					      1,
					      0 /*flags*/);
#endif
    }

    rget_alloc.free_obj(rget);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNetEXInternal
  //

  GASNetEXInternal::GASNetEXInternal(GASNetEXModule *_module,
				     RuntimeImpl *_runtime)
    : module(_module)
    , runtime(_runtime)
    , poller(this)
    , injector(this)
    , completer(this)
    , rgetter(this)
    , total_packets_received(0)
    , databuf_md(nullptr)
  {}

  GASNetEXInternal::~GASNetEXInternal()
  {}

  void GASNetEXInternal::init(int *argc, const char ***argv)
  {
    gex_Flags_t flags = 0;
    // NOTE: we do NOT set GEX_FLAG_USES_GASNET1 here - we're going to try to
    //  avoid any use of the legacy entry points
    gex_EP_t prim_ep = GEX_EP_INVALID;
    CHECK_GEX( gex_Client_Init(&client, &prim_ep, &prim_tm,
			       "realm-gex", argc,
			       const_cast<char ***>(argv), flags) );
    eps.push_back(prim_ep);
    prim_rank = gex_TM_QueryRank(prim_tm);
    prim_size = gex_TM_QuerySize(prim_tm);
    Network::my_node_id = prim_rank;
    Network::max_node_id = prim_size - 1;
    Network::all_peers.add_range(0, prim_size - 1);
    Network::all_peers.remove(prim_rank);

    // stick a pointer to ourselves in the endpoint CData so that handlers
    //  can find us
    gex_EP_SetCData(prim_ep, this);

    CHECK_GEX( gex_EP_RegisterHandlers(prim_ep,
				       GASNetEXHandlers::handler_table,
				       GASNetEXHandlers::handler_table_size) );

    xmitsrcs.push_back(new XmitSrc(this, 0 /*ep_index*/));

#if REALM_GEX_API >= 1300
    // once we've done the basic init, shut off verbose errors from GASNet
    //  and we'll report failures ourselves
    gex_System_SetVerboseErrors(0);
#endif

    poller.add_to_manager(&runtime->bgwork);
    poller.begin_polling();

    injector.add_to_manager(&runtime->bgwork);

    completer.add_to_manager(&runtime->bgwork);

    rgetter.add_to_manager(&runtime->bgwork);

    obmgr.add_to_manager(&runtime->bgwork);
  }

  uintptr_t GASNetEXInternal::attach(size_t size)
  {
    log_gex.info() << "gasnet versions: release=" << REALM_GEX_RELEASE << " api=" << REALM_GEX_API;

    // the primordial segment consists of:
    // 1) storage for any NetworkSegments we're allowed to allocate
    // 2) outbufs
    size_t user_size = roundup_pow2(size, 128);
    prim_segsize = (user_size +
		    (module->cfg_outbuf_count * module->cfg_outbuf_size));

    log_gex.debug() << "attaching prim segment: size=" << prim_segsize;
    CHECK_GEX( gex_Segment_Attach(&prim_segment, prim_tm, prim_segsize) );
    void *base = gex_Segment_QueryAddr(prim_segment);
    log_gex.debug() << "prim segment allocated: base=" << base;

    uintptr_t base_as_uint = reinterpret_cast<uintptr_t>(base);
    segments_by_addr.push_back({ base_as_uint, base_as_uint+prim_segsize,
	                         0, prim_segment,
	                         NetworkSegmentInfo::HostMem, 0 });

    obmgr.init(module->cfg_outbuf_count, module->cfg_outbuf_size,
	       base_as_uint + user_size);

    return base_as_uint;
  }

  bool GASNetEXInternal::attempt_binding(void *base, size_t size,
					 NetworkSegmentInfo::MemoryType memtype,
					 NetworkSegmentInfo::MemoryTypeExtraData memextra,
					 gex_EP_Index_t *ep_indexp)
  {
    log_gex_bind.info() << "segment bind?: base=" << base << " size=" << size
			<< " mtype=" << memtype << " extra=" << memextra;

#if (REALM_GEX_RELEASE == 20201100) && !defined(GASNET_CONDUIT_IBV)
    // in 2020.11.0, conduits other than ibv would assert-fail in
    //  gex_EP_Create, so don't even attempt binding
    return false;
#else
    // rely on error results from gex_{EP,Segment}_Create to determine what
    //  is supported in a conduit-agnostic way

    gex_MK_t mk = GEX_MK_INVALID;

    // see if we can get/make a supported memory kind for this segment
    do {
      if(module->cfg_bind_hostmem &&
	 (memtype == NetworkSegmentInfo::HostMem)) {
	mk = GEX_MK_HOST;
	break;
      }

#if defined(GASNET_HAVE_MK_CLASS_CUDA_UVA) && defined(REALM_USE_CUDA)
      // create a gex_MK_t for the GPU that owns this memory
      if(module->cfg_bind_cudamem &&
	 (memtype == NetworkSegmentInfo::CudaDeviceMem)) {
	const Cuda::GPU *gpu = reinterpret_cast<Cuda::GPU *>(memextra);
	gex_MK_Create_args_t args;
	args.gex_flags = 0;
	args.gex_class = GEX_MK_CLASS_CUDA_UVA;
	args.gex_args.gex_class_cuda_uva.gex_CUdevice = gpu->info->device;
	int ret = gex_MK_Create(&mk,
				client,
				&args,
				0 /*flags*/);
	if(ret != GASNET_OK) {
	  log_gex_bind.info() << "mk_create failed?  ret=" << ret
                              << " mtype=" << memtype << " extra=" << memextra
                              << " gpu_index=" << gpu->info->index;
	  return false;
	}
	break;
      }
#endif

#if defined(GASNET_HAVE_MK_CLASS_HIP) && defined(REALM_USE_HIP) && defined(__HIP_PLATFORM_HCC__)
      // create a gex_MK_t for the GPU that owns this memory, it only supports building HIP for AMD GPU (__HIP_PLATFORM_HCC_) 
      if(module->cfg_bind_hipmem &&
	 (memtype == NetworkSegmentInfo::HipDeviceMem)) {
	const Hip::GPU *gpu = reinterpret_cast<Hip::GPU *>(memextra);
	gex_MK_Create_args_t args;
	args.gex_flags = 0;
	args.gex_class = GEX_MK_CLASS_HIP;
	args.gex_args.gex_class_hip.gex_hipDevice = gpu->info->device;
	int ret = gex_MK_Create(&mk,
				client,
				&args,
				0 /*flags*/);
	if(ret != GASNET_OK) {
	  log_gex_bind.info() << "mk_create failed?  ret=" << ret
                              << " mtype=" << memtype << " extra=" << memextra
                              << " gpu_index=" << gpu->info->index;
	  return false;
	}
	break;
      }
#endif

      // out of ideas - give up
      return false;
    } while(0);

    assert(mk != GEX_MK_INVALID);

    // attempt to create a GASNet segment
    // GEX: in 2020.11.0, this will generally assert-fail rather than
    //  returning an error code (e.g. for exceeding BAR1 size)
    gex_Segment_t segment = GEX_SEGMENT_INVALID;
    {
      int ret = gex_Segment_Create(&segment, client, base, size,
				   mk, 0 /*flags*/);
      if(ret != GASNET_OK) {
	log_gex_bind.info() << "segment_create failed?  ret=" << ret
                            << " mtype=" << memtype << " base=" << base
                            << " size=" << size << " extra=" << memextra;
	return false;
      }
    }

    // create an endpoint to which we'll bind this segment
    gex_EP_Capabilities_t caps = 0;
    // TODO: request more capabilities as gasnet adds them
    caps |= GEX_EP_CAPABILITY_RMA;
    gex_EP_t ep = GEX_EP_INVALID;
    {
      int ret = gex_EP_Create(&ep, client, caps, 0 /*flags*/);
      if(ret != GASNET_OK) {
	log_gex_bind.info() << "ep_create failed?  ret=" << ret
                            << " caps=" << caps;
	// TODO: destroy the segment we created?
	return false;
      }
    }

    gex_EP_Index_t ep_index = gex_EP_QueryIndex(ep);
    assert(ep_index == eps.size());
    eps.push_back(ep);
    gex_EP_SetCData(ep, this);

    assert(ep_index == xmitsrcs.size());
    xmitsrcs.push_back(new XmitSrc(this, ep_index));

    gex_EP_BindSegment(ep, segment, 0 /*flags*/);

    uintptr_t base_as_uint = reinterpret_cast<uintptr_t>(base);
    segments_by_addr.push_back({ base_as_uint, base_as_uint+size,
	                         ep_index, segment,
	                         memtype, memextra });

    // return ep_index and report success
    assert(ep_indexp);
    *ep_indexp = ep_index;

    log_gex_bind.info() << "segment bound: base=" << base << " size=" << size
			<< " mtype=" << memtype << " extra=" << memextra
			<< " ep_index=" << ep_index;
    return true;
#endif
  }

  struct GASNetEXInternal::SegmentInfoSorter {
    bool operator()(const GASNetEXInternal::SegmentInfo& lhs,
		    const GASNetEXInternal::SegmentInfo& rhs) const
    {
      return lhs.base < rhs.base;
    }
  };

  void GASNetEXInternal::publish_bindings()
  {
    // publish all of our endpoints in one go
    CHECK_GEX( gex_EP_PublishBoundSegment(prim_tm,
					  eps.data(),
					  eps.size(),
					  0 /*flags*/) );

    // also, now that we know all binding is done, sort the by_addr list
    std::sort(segments_by_addr.begin(), segments_by_addr.end(),
	      SegmentInfoSorter());

    // sanity-check that there's no overlap
    for(size_t i = 1; i < segments_by_addr.size(); i++)
      assert(segments_by_addr[i-1].limit <= segments_by_addr[i].base);
  }

  void GASNetEXInternal::detach()
  {
    poller.end_polling();

#ifdef DEBUG_REALM
    poller.shutdown_work_item();
    injector.shutdown_work_item();
    completer.shutdown_work_item();
    rgetter.shutdown_work_item();
    obmgr.shutdown_work_item();
#endif

    // since we used GEX_EVENT_GROUP for long AMs when we didn't care
    //  about local completion, do a dummy wait here in case anything
    //  in the gasnet code checks for leaked events
    gex_NBI_Wait(GEX_EC_AM, 0 /*flags*/);

    for(size_t i = 0; i < xmitsrcs.size(); i++)
      delete xmitsrcs[i];
    xmitsrcs.clear();
  }

  void GASNetEXInternal::barrier()
  {
    gex_Event_t done = gex_Coll_BarrierNB(prim_tm, 0);
    gex_Event_Wait(done);
  }

  void GASNetEXInternal::broadcast(gex_Rank_t root,
				   const void *val_in, void *val_out,
				   size_t bytes)
  {
    gex_Event_t done = gex_Coll_BroadcastNB(prim_tm, root,
					    val_out, val_in,
					    bytes, 0);
    gex_Event_Wait(done);
  }

  void GASNetEXInternal::gather(gex_Rank_t root,
				const void *val_in, void *vals_out,
				size_t bytes)
  {
    // GASNetEX doesn't have a gather collective?
    // this isn't performance critical right now, so cobble it together from
    //  a bunch of broadcasts
    void *dummy = (root == prim_rank) ? 0 : alloca(bytes);
    for(gex_Rank_t i = 0; i < prim_size; i++) {
      void *dst;
      if(root == prim_rank)
	dst = static_cast<char *>(vals_out) + (i * bytes);
      else
	dst = dummy;
      gex_Event_t done = gex_Coll_BroadcastNB(prim_tm, i,
					      dst, val_in,
					      bytes, 0);
      gex_Event_Wait(done);
    }
  }

  size_t GASNetEXInternal::sample_messages_received_count()
  {
    return total_packets_received.load();
  }

  bool GASNetEXInternal::check_for_quiescence(size_t sampled_receive_count)
  {
    // in order to be quiescent, the following things should be true:
    // 1) no unsent packets in any xpair
    // 2) no unsent completions in any xpair
    // 3) no pending complations remain
    // 4) no reverse gets in progress
    // 5) all packets sent have been received

    // we handle 1-4 by having each count how many of 1-4 are violated locally
    //  and then doing a reduction sum over all nodes
    // we handle 5 by having each rank determine the total packets it has sent
    //  and received (regardless of which xpair), summing those over all ranks,
    //  and demanding that the sum of packets sent matches the sum of received
    uint64_t local_counts[3] = { 0, 0, 0 };

    if(poller.has_work_remaining()) {
      log_gex_quiesce.debug() << "poller busy";
      local_counts[0]++;
    }
    if(injector.has_work_remaining()) {
      log_gex_quiesce.debug() << "injector busy";
      local_counts[0]++;
    }
    if(completer.has_work_remaining()) {
      log_gex_quiesce.debug() << "completer busy";
      local_counts[0]++;
    }
    if(rgetter.has_work_remaining()) {
      log_gex_quiesce.debug() << "rgetter busy";
      local_counts[0]++;
    }
    if(compmgr.num_completions_pending() > 0) {
      log_gex_quiesce.debug() << "compmgr busy";
      assert(0);
      local_counts[0]++;
    }
    for(gex_EP_Index_t src_ep_index = 0; src_ep_index < xmitsrcs.size(); src_ep_index++) {
      atomic<XmitSrcDestPair *> *pairptrs = xmitsrcs[src_ep_index]->pairs;
      for(gex_EP_Index_t tgt_ep_index = 0; tgt_ep_index < GASNET_MAXEPS; tgt_ep_index++)
	for(gex_Rank_t tgt_rank = 0; tgt_rank < prim_size; tgt_rank++) {
	  XmitSrcDestPair *xpair = (pairptrs++)->load_acquire();
	  if(!xpair) continue;
	  size_t num_rsrvd = xpair->packets_reserved.load();
	  if(num_rsrvd > 0) {
	    log_gex_quiesce.debug() << "xpair reserved: "
				    << src_ep_index
				    << "->" << tgt_rank << "/" << tgt_ep_index
				    << " " << num_rsrvd;
	    local_counts[1] += num_rsrvd;
	  }
	}
    }
    // use the receive count that was sampled before the incoming message
    //  manager was drained - if the actual 'total_packets_received' has
    //  increased since, we're obviously not quiescent, but we need every
    //  other rank to know that too
    local_counts[2] = sampled_receive_count;

    log_gex_quiesce.debug() << "local counts: " << local_counts[0]
			    << " " << local_counts[1] << " " << local_counts[2];

    uint64_t total_counts[3];
    gex_Flags_t flags = 0;
    gex_Event_t done = gex_Coll_ReduceToAllNB(prim_tm,
					      total_counts, local_counts,
					      GEX_DT_U64, sizeof(uint64_t),
					      3, GEX_OP_ADD,
					      nullptr, nullptr, flags);

    // wait on completion of the collective reduction, but keep track of time
    //  and complain if it takes too long
    long long t_start = Clock::current_time_in_nanoseconds();
    long long t_complain = t_start;
    do {
      long long t_now = Clock::current_time_in_nanoseconds();
      if(t_now >= (t_complain + 1000000000 /*1 sec*/)) {
	log_gex_quiesce.info() << "allreduce not done after " << (t_now - t_start) << " ns";
	t_complain = t_now;
      }
      // we can't perform the poll ourselves, but make sure at least one full
      //  poll succeeds before we continue
      poller.wait_for_full_poll_cycle();
    } while(gex_Event_Test(done) != GASNET_OK);

    if(prim_rank == 0)
      log_gex_quiesce.info() << "total counts: " << total_counts[0]
			    << " " << total_counts[1] << " " << total_counts[2];
    else
      log_gex_quiesce.debug() << "total counts: " << total_counts[0]
			      << " " << total_counts[1] << " " << total_counts[2];

    return ((total_counts[0] == 0) &&
	    (total_counts[1] == total_counts[2]));
  }

  PendingCompletion *GASNetEXInternal::get_available_comp()
  {
    return compmgr.get_available();
  }

  PendingCompletion *GASNetEXInternal::early_local_completion(PendingCompletion *comp)
  {
    if(comp && comp->has_local_completions()) {
      bool done = comp->invoke_local_completions();
      if(done) {
	compmgr.recycle_comp(comp);
	return nullptr;
      }
    }
    return comp;
  }

  size_t GASNetEXInternal::recommended_max_payload(gex_Rank_t target,
						   gex_EP_Index_t target_ep_index,
						   bool with_congestion,
						   size_t header_size,
						   uintptr_t dest_payload_addr)
  {
    // TODO: ideally make this a per-target counter?
    if(with_congestion && compmgr.over_pending_completion_soft_limit())
      return 0;

    if(dest_payload_addr == 0) {
      // medium message
      size_t limit = GASNetEXHandlers::max_request_medium(eps[0],
                                                          target,
                                                          target_ep_index,
                                                          header_size,
                                                          GEX_EVENT_NOW,
                                                          0 /*flags*/);

      // message goes inline into pktbuf, so limit to that size as well
      size_t pad_hdr_bytes = roundup_pow2(header_size + 2*sizeof(gex_AM_Arg_t), 16);
      limit = std::min(limit, module->cfg_outbuf_size - pad_hdr_bytes);

      // also use a hard limit from the command line, if present
      if(module->cfg_max_medium)
        limit = std::min(limit, module->cfg_max_medium);

      return limit;
    } else {
      // TODO: these should go through pktbufs - see issue 1138
      //  disabled for now
      return 0;
#if 0
      // long message
      size_t limit = GASNetEXHandlers::max_request_long(eps[0],
							target,
							target_ep_index,
							header_size,
							GEX_EVENT_NOW,
							0 /*flags*/);
      // without a known source address, we're going to have to copy
      //  the source data into an outbuf, so limit to the outbuf size
      limit = std::min(limit, size_t(16384 /*TODO*/));
      // generally we'll want to avoid enormous packets clogging up the
      //  tubes
      if(module->cfg_max_long > 0)
	limit = std::min(limit, module->cfg_max_long);

      return limit;
#endif
    }
  }

  size_t GASNetEXInternal::recommended_max_payload(gex_Rank_t target,
						   gex_EP_Index_t target_ep_index,
						   const void *data, size_t bytes_per_line,
						   size_t lines, size_t line_stride,
						   bool with_congestion,
						   size_t header_size,
						   uintptr_t dest_payload_addr)
  {
    // TODO: ideally make this a per-target counter?
    if(with_congestion && compmgr.over_pending_completion_soft_limit())
      return 0;

    if(dest_payload_addr == 0) {
      // medium message
      size_t limit = GASNetEXHandlers::max_request_medium(eps[0],
                                                          target,
                                                          target_ep_index,
                                                          header_size,
                                                          GEX_EVENT_NOW,
                                                          0 /*flags*/);

      // message goes inline into pktbuf, so limit to that size as well
      size_t pad_hdr_bytes = roundup_pow2(header_size + 2*sizeof(gex_AM_Arg_t), 16);
      limit = std::min(limit, module->cfg_outbuf_size - pad_hdr_bytes);

      // also use a hard limit from the command line, if present
      if(module->cfg_max_medium)
        limit = std::min(limit, module->cfg_max_medium);

      return limit;
    } else {
      // long message
      size_t limit = GASNetEXHandlers::max_request_long(eps[0],
							target,
							target_ep_index,
							header_size,
							GEX_EVENT_NOW,
							0 /*flags*/);
      // additional constraints may apply based on where the source data is
      const SegmentInfo *src_seg = find_segment(data);
      if(src_seg) {
	// can use as source of AM or as remote of get, but both require
	//  contiguous data, so limit to a single line worth of data
        //
        // don't offer a path that will require a copy - the dma system
        //  should use an intermediate buffer in that case
        limit = std::min(limit, bytes_per_line);
      } else {
        // TODO: these should go through pktbufs - see issue 1138
        //  disabled for now
        limit = 0;
        // actually, make this a hard error for now - all source addresses should
        //  be registered because we don't know that the cpu can copy data into a
        //  pktbuf
        log_gex.fatal() << "request for max payload with non-registered src = " << data;
        abort();
#if 0
	// data will have to be copied into an outbuf, so don't exceed that
	limit = std::min(limit, size_t(16384 /*TODO*/));
#endif
      }
      // generally we'll want to avoid enormous packets clogging up the
      //  tubes
      if(module->cfg_max_long > 0)
	limit = std::min(limit, module->cfg_max_long);

      return limit;
    }
  }

  size_t GASNetEXInternal::recommended_max_payload(bool with_congestion,
						   size_t header_size)
  {
    // TODO: ideally make this a per-target counter?
    if(with_congestion && compmgr.over_pending_completion_soft_limit())
      return 0;

    size_t limit = gex_AM_LUBRequestMedium();

    // message goes inline into pktbuf, so limit to that size as well
    size_t pad_hdr_bytes = roundup_pow2(header_size + 2*sizeof(gex_AM_Arg_t), 16);
    limit = std::min(limit, module->cfg_outbuf_size - pad_hdr_bytes);

    // also use a hard limit from the command line, if present
    if(module->cfg_max_medium)
      limit = std::min(limit, module->cfg_max_medium);

    return limit;
  }

  PreparedMessage *GASNetEXInternal::prepare_message(gex_Rank_t target,
						     gex_EP_Index_t target_ep_index,
						     unsigned short msgid,
						     void *&header_base,
						     size_t header_size,
						     void *&payload_base,
						     size_t payload_size,
						     uintptr_t dest_payload_addr)
  {
    PreparedMessage *msg = prep_alloc.alloc_obj();

    msg->strategy = PreparedMessage::STRAT_UNKNOWN;
    msg->target = target;
    msg->source_ep_index = 0; // we may adjust this below
    msg->target_ep_index = target_ep_index;
    msg->msgid = msgid;
    msg->dest_payload_addr = dest_payload_addr;
    msg->databuf = nullptr;
#ifdef DEBUG_REALM
    msg->temp_buffer = nullptr;
    msg->pktbuf = nullptr;
    msg->pktidx = -1;
#endif
    msg->put = nullptr;

    // even if immediates are allowed via configuration, we can't inject
    //  messages if we're inside an AM request/reply handler
    bool imm_ok = (module->cfg_use_immediate &&
		   !ThreadLocal::in_am_handler);

    // TODO: document this decision tree rigorously so that it can be
    //  optimized and missing leaves found
    if(payload_size == 0) {
      // no payload
      payload_base = nullptr;

      XmitSrcDestPair *xpair = xmitsrcs[0]->lookup_pair(target,
							target_ep_index);
      if(imm_ok && xpair->has_packets_queued()) {
	// suppress immediate mode
	imm_ok = false;
      }

      do {
	// choice 1: attempt immediate injection at commit time
	if(imm_ok) {
	  // TODO: check for contention
	  msg->strategy = PreparedMessage::STRAT_SHORT_IMMEDIATE;
	  break;
	}

	// choice 2: reserve space in a pktbuf to avoid a copy on enqueue
	{
	  bool overflow_ok = true; // TODO: try immediate later
	  bool rsrv_ok = xpair->reserve_pbuf_inline(header_size, 0 /*payload_size*/,
						    overflow_ok,
						    msg->pktbuf, msg->pktidx,
						    header_base, payload_base);
	  if(rsrv_ok) {
	    msg->strategy = PreparedMessage::STRAT_SHORT_PBUF;
	    break;
	  }
	}

	assert(0);
      } while(false);
    } else {
      // are we writing to a specified location on the destination?
      if(dest_payload_addr == 0) {
	// no specified destination, so this'll be a medium message

	XmitSrcDestPair *xpair = xmitsrcs[0]->lookup_pair(target,
							  target_ep_index);
	if(imm_ok && xpair->has_packets_queued()) {
	  // suppress immediate mode
	  imm_ok = false;
	}

	do {
	  // choice 1: negotiated payload
#ifdef GASNET_NATIVE_NP_ALLOC_REQ_MEDIUM
	  if(imm_ok && module->cfg_use_negotiated) {
	    // we want a GASNet-allocated buffer, so do NOT offer our own,
	    //  even if we have it (TODO: get guidance on whether this is
	    //  always the right choice)

	    gex_AM_SrcDesc_t sd = GEX_AM_SRCDESC_NO_OP;
	    // double-check that our size is acceptable for a GASNet-allocated
	    //  message
	    size_t max_payload =
	      GASNetEXHandlers::max_request_medium(eps[0],
						   target,
						   target_ep_index,
						   header_size,
						   nullptr,
						   GEX_FLAG_AM_PREPARE_LEAST_ALLOC);
	    if(payload_size <= max_payload) {
	      gex_Flags_t flags = GEX_FLAG_IMMEDIATE;
	      sd = GASNetEXHandlers::prepare_request_medium(eps[0],
							    target,
							    target_ep_index,
							    header_size,
							    nullptr /*data*/,
							    payload_size,
							    payload_size,
							    nullptr /*lc_opt*/,
							    flags);
	    }
	    if(sd != GEX_AM_SRCDESC_NO_OP) {
	      // success - use the GASNet-allocated payload
	      payload_base = gex_AM_SrcDescAddr(sd);
	      msg->srcdesc = sd;
	      msg->strategy = PreparedMessage::STRAT_MEDIUM_PREP;
	      break;
	    } else {
	      // failure - fall through to the next choice(s)
	    }
	  }
#endif

	  // choice 2: if the caller has a place to hold the data, try for
	  //  an immediate FPAM at commit time?
	  if(imm_ok && payload_base) {
	    msg->strategy = PreparedMessage::STRAT_MEDIUM_IMMEDIATE;
	    break;
	  }

	  // choice 3: reserve space in a pktbuf to avoid a copy on enqueue
	  {
	    //  allow a spill into an overflow pktbuf if needed (a dynamic
	    //   allocation has to happen somewhere)
	    bool overflow_ok = true;
	    bool rsrv_ok = xpair->reserve_pbuf_inline(header_size, payload_size,
						      overflow_ok,
						      msg->pktbuf, msg->pktidx,
						      header_base, payload_base);
	    if(rsrv_ok) {
	      msg->strategy = PreparedMessage::STRAT_MEDIUM_PBUF;
	      break;
	    }
	  }

	  assert(0);
	} while(false);
#if 0
	// TODO: lots of better choices than malloc'ing a temp buffer
	// consider whether source is in prim segment for client NPAM
	msg->temp_buffer = malloc(payload_size);
	payload_base = msg->temp_buffer;
	msg->strategy = PreparedMessage::STRAT_MEDIUM_MALLOCSRC;
#endif
      } else {
	// this'll be a long AM or RMA ops, but either way we need the
	//  source data in a registered segment
	const SegmentInfo *srcseg = find_segment(payload_base);

	if(!srcseg) {
	  // TODO: attempt a medium-sized NPAM long, once that's a thing

          // TODO: fallback should be inline data in the pktbuf (issue #1138)
          assert(0);
	}

	msg->source_ep_index = srcseg->ep_index;

	// we can use long if both endpoints are AM-capable (currently only
	//  prim endpoint is), otherwise rget
	bool use_long = (!module->cfg_force_rma &&
                         (!srcseg || (srcseg->ep_index == 0)) &&
			 (target_ep_index == 0));
        // TODO: will we never need to make a put vs. get decision on a
        //  per-endpoint basis?
        bool use_rmaput = (!use_long && module->cfg_use_rma_put);
#ifndef REALM_GEX_RMA_HONORS_IMMEDIATE_FLAG
        // if we're using RMA put and the conduit doesn't actually honor
        //  GEX_FLAG_IMMEDIATE, disable immediate mode
        if(use_rmaput)
          imm_ok = false;
#endif

	XmitSrcDestPair *xpair;
	if(use_long || use_rmaput) {
	  xpair = xmitsrcs[srcseg->ep_index]->lookup_pair(target,
							  target_ep_index);
	} else {
	  // an rget is actually sent between prim endpoints
	  xpair = xmitsrcs[0]->lookup_pair(target, 0);
	}

	if(imm_ok && xpair->has_packets_queued()) {
	  // suppress immediate mode
	  imm_ok = false;
	}

        // rma puts, whether the put itself is queued, always need the header
        //  information in an object that outlives the put injection
        if(use_rmaput) {
          msg->put = put_alloc.alloc_obj();

          // header will eventually be sent between primordial endpoints
          msg->put->xpair = xmitsrcs[0]->lookup_pair(target, 0);

          // have message header go directly into the PendingPutHeader
          assert(header_size <= PendingPutHeader::MAX_HDR_SIZE);
          msg->put->hdr_size = header_size;
          header_base = msg->put->hdr_data;

          msg->put->target = target;
          msg->put->tgt_ptr = dest_payload_addr;

          msg->put->local_comp = nullptr;

          // src/tgt ep index and src_ptr only really needed for logging/debug
          msg->put->src_ep_index = (srcseg ? srcseg->ep_index : 0);
          msg->put->tgt_ep_index = target_ep_index;
          msg->put->src_ptr = reinterpret_cast<uintptr_t>(payload_base);
        }

	do {
	  // choice 1: negotiated payload
	  if(imm_ok && use_long && module->cfg_use_negotiated) {
	    // TODO: NPAM (only if native!)
	  }

	  // choice 2: wait and try an immediate FPAM at commit time
	  if(imm_ok) {
	    msg->strategy = (use_long   ? PreparedMessage::STRAT_LONG_IMMEDIATE :
                             use_rmaput ? PreparedMessage::STRAT_PUT_IMMEDIATE :
                                          PreparedMessage::STRAT_RGET_IMMEDIATE);
	    break;
	  }

	  // choice 3: we're going to enqueue, so reserve space for the
	  //  header now
	  {
	    // allow spill of header into overflow for now (TODO: allow
	    //  backpressure to caller)
	    bool overflow_ok = true;
	    bool rsrv_ok =
              (use_rmaput ?
                 xpair->reserve_pbuf_put(overflow_ok,
                                         msg->pktbuf, msg->pktidx) :
                 xpair->reserve_pbuf_long_rget(header_size,
                                               overflow_ok,
                                               msg->pktbuf,
                                               msg->pktidx,
                                               header_base));
	    if(rsrv_ok) {
	      msg->strategy = (use_long   ? PreparedMessage::STRAT_LONG_PBUF :
                               use_rmaput ? PreparedMessage::STRAT_PUT_PBUF :
                                            PreparedMessage::STRAT_RGET_PBUF);
	      break;
	    }
	  }

	  assert(0);
	} while(false);
      }
    }

    return msg;
  }

  const GASNetEXInternal::SegmentInfo *GASNetEXInternal::find_segment(const void *srcptr) const
  {
    uintptr_t ptr = reinterpret_cast<uintptr_t>(srcptr);
    // binary search
    unsigned lo = 0;
    unsigned hi = segments_by_addr.size();
    while(lo < hi) {
      unsigned mid = (lo + hi) >> 1;
      if(ptr < segments_by_addr[mid].base) {
	hi = mid;
      } else if(ptr >= segments_by_addr[mid].limit) {
	lo = mid + 1;
      } else {
	return &segments_by_addr[mid];
      }
    }
    return nullptr;
  }



  void GASNetEXInternal::commit_message(PreparedMessage *msg,
					PendingCompletion *comp,
					void *header_base,
					size_t header_size,
					void *payload_base,
					size_t payload_size)
  {
    log_gex_msg.info() << "commit: tgt=" << msg->target
		       << " msgid=" << msg->msgid
		       << " strat=" << int(msg->strategy)
		       << " header=" << header_size
		       << " payload=" << payload_size
		       << " dest=" << std::hex << msg->dest_payload_addr << std::dec;

    bool do_local_comp = false;

    switch(msg->strategy) {
    case PreparedMessage::STRAT_SHORT_IMMEDIATE:
      {
	XmitSrcDestPair *xpair = xmitsrcs[0]->lookup_pair(msg->target,
							  msg->target_ep_index);

	gex_AM_Arg_t arg0 = msg->msgid;
	if(comp) {
	  // we'll do local completion (if any) ourselves
	  do_local_comp = comp->has_local_completions();

	  // remote completion needs to bounce off target
	  if(comp->has_remote_completions()) {
	    unsigned comp_info = ((comp->index << 2) +
				  PendingCompletion::REMOTE_PENDING_BIT);
	    arg0 |= (comp_info << MSGID_BITS);
	  }
	}

	if(module->cfg_do_checksums)
	  insert_packet_crc(arg0, header_base, header_size, nullptr, 0);

	// this is always done with immediate mode
	gex_Flags_t flags = GEX_FLAG_IMMEDIATE;

	int ret = GASNetEXHandlers::send_request_short(eps[0],
						       msg->target,
						       msg->target_ep_index,
						       arg0,
						       header_base,
						       header_size,
						       flags);
	if(ret == GASNET_OK) {
	  // success
	  xpair->record_immediate_packet();
	} else {
	  log_gex_msg.info() << "immediate failed - queueing message";
	  // could not immediately inject it, so enqueue now
	  void *act_hdr_base = header_base;
	  void *act_payload_base = nullptr;
	  OutbufMetadata *pktbuf;
	  int pktidx;
	  bool ok = xpair->reserve_pbuf_inline(header_size,
					       0 /*payload_size*/,
					       true /*overflow_ok*/,
					       pktbuf, pktidx,
					       act_hdr_base, act_payload_base);
	  assert(ok); // can't handle backpressure at this point
	  // probably need to copy header data
	  if(act_hdr_base != header_base)
	    memcpy(act_hdr_base, header_base, header_size);
	  // and now commit
	  xpair->commit_pbuf_inline(pktbuf, pktidx, act_hdr_base,
				    arg0, 0 /*payload_size*/);
	}

	break;
      }

    case PreparedMessage::STRAT_SHORT_PBUF:
      {
	gex_AM_Arg_t arg0 = msg->msgid;
	if(comp) {
	  // we'll do local completion (if any) ourselves
	  do_local_comp = comp->has_local_completions();

	  // remote completion needs to bounce off target
	  if(comp->has_remote_completions()) {
	    unsigned comp_info = ((comp->index << 2) +
				  PendingCompletion::REMOTE_PENDING_BIT);
	    arg0 |= (comp_info << MSGID_BITS);
	  }
	}

	if(module->cfg_do_checksums)
	  insert_packet_crc(arg0, header_base, header_size, nullptr, 0);

	XmitSrcDestPair *xpair = xmitsrcs[0]->lookup_pair(msg->target,
							  msg->target_ep_index);
	xpair->commit_pbuf_inline(msg->pktbuf, msg->pktidx,
				  header_base,
				  arg0, 0 /*payload_size*/);
	break;
      }

    case PreparedMessage::STRAT_MEDIUM_IMMEDIATE:
      {
	// medium with a caller-provided buffer - attempt an immediate FPAM
	//  and enqueue if that doesn't work

	XmitSrcDestPair *xpair = xmitsrcs[0]->lookup_pair(msg->target,
							  msg->target_ep_index);

	gex_AM_Arg_t arg0 = msg->msgid;
	if(comp) {
	  // we'll do local completion (if any) ourselves
	  do_local_comp = comp->has_local_completions();

	  // remote completion needs to bounce off target
	  if(comp->has_remote_completions()) {
	    unsigned comp_info = ((comp->index << 2) +
				  PendingCompletion::REMOTE_PENDING_BIT);
	    arg0 |= (comp_info << MSGID_BITS);
	  }
	}

	if(module->cfg_do_checksums)
	  insert_packet_crc(arg0, header_base, header_size,
			    payload_base, payload_size);

	// this is always done with immediate mode, and force GASNet to copy
	//  the payload before returning (a copy has to happen, and it's
	//  best to pay for it here)
	gex_Event_t *lc_opt = GEX_EVENT_NOW;
	gex_Flags_t flags = GEX_FLAG_IMMEDIATE;

#ifdef DEBUG_REALM
        {
          size_t max_payload =
            GASNetEXHandlers::max_request_medium(eps[0],
                                                 msg->target,
                                                 msg->target_ep_index,
                                                 header_size,
                                                 lc_opt,
                                                 flags);
          if(payload_size > max_payload) {
            log_gex_xpair.fatal() << "medium payload too large!  src="
                                  << Network::my_node_id << "/0"
                                  << " tgt=" << msg->target
                                  << "/" << msg->target_ep_index
                                  << " max=" << max_payload << " act=" << payload_size;
            abort();
          }
        }
#endif

	int ret = GASNetEXHandlers::send_request_medium(eps[0],
							msg->target,
							msg->target_ep_index,
							arg0,
							header_base,
							header_size,
							payload_base,
							payload_size,
							lc_opt,
							flags);
	if(ret == GASNET_OK) {
	  // success
	  xpair->record_immediate_packet();
	} else {
	  log_gex_msg.info() << "immediate failed - queueing message";
	  // could not immediately inject it, so enqueue now
	  void *act_hdr_base = header_base;
	  void *act_payload_base = payload_base;
	  OutbufMetadata *pktbuf;
	  int pktidx;
	  bool ok = xpair->reserve_pbuf_inline(header_size,
					       payload_size,
					       true /*overflow_ok*/,
					       pktbuf, pktidx,
					       act_hdr_base, act_payload_base);
	  assert(ok); // can't handle backpressure at this point
	  // probably need to copy data
	  if(act_hdr_base != header_base)
	    memcpy(act_hdr_base, header_base, header_size);
	  if(act_payload_base != payload_base)
	    memcpy(act_payload_base, payload_base, payload_size);
	  // and now commit
	  xpair->commit_pbuf_inline(pktbuf, pktidx, act_hdr_base,
				    arg0, payload_size);
	}

	break;
      }

    case PreparedMessage::STRAT_MEDIUM_PBUF:
      {
	// medium, written into a pbuf to be queued

	gex_AM_Arg_t arg0 = msg->msgid;
	if(comp) {
	  // we'll do local completion (if any) ourselves
	  do_local_comp = comp->has_local_completions();

	  // remote completion needs to bounce off target
	  if(comp->has_remote_completions()) {
	    unsigned comp_info = ((comp->index << 2) +
				  PendingCompletion::REMOTE_PENDING_BIT);
	    arg0 |= (comp_info << MSGID_BITS);
	  }
	}

	if(module->cfg_do_checksums)
	  insert_packet_crc(arg0, header_base, header_size,
			    payload_base, payload_size);

	XmitSrcDestPair *xpair = xmitsrcs[0]->lookup_pair(msg->target,
							  msg->target_ep_index);
	xpair->commit_pbuf_inline(msg->pktbuf, msg->pktidx,
				  header_base,
				  arg0, payload_size);
	break;
      }

    case PreparedMessage::STRAT_MEDIUM_MALLOCSRC:
      {
	// medium
	size_t max_med_size =
	  GASNetEXHandlers::max_request_medium(eps[0],
					       msg->target,
					       msg->target_ep_index,
					       header_size,
					       GEX_EVENT_NOW,
					       0 /*flags*/);
	log_gex.debug() << "max med = " << max_med_size;
	if(payload_size > max_med_size) {
	  log_gex.fatal() << "medium size exceeded: " << payload_size << " > " << max_med_size;
	  abort();
	}

	gex_AM_Arg_t arg0 = msg->msgid;
	if(comp) {
	  // we'll do local completion (if any) ourselves
	  do_local_comp = comp->has_local_completions();

	  // remote completion needs to bounce off target
	  if(comp->has_remote_completions()) {
	    unsigned comp_info = ((comp->index << 2) +
				  PendingCompletion::REMOTE_PENDING_BIT);
	    arg0 |= (comp_info << MSGID_BITS);
	  }
	}
	GASNetEXHandlers::send_request_medium(eps[0],
					      msg->target,
					      msg->target_ep_index,
					      arg0,
					      header_base, header_size,
					      payload_base, payload_size,
					      GEX_EVENT_NOW,
					      0 /*flags*/);
	assert(payload_base == msg->temp_buffer);
	free(msg->temp_buffer);
	break;
      }

    case PreparedMessage::STRAT_MEDIUM_PREP:
      {
	gex_AM_Arg_t arg0 = msg->msgid;
	if(comp) {
	  // we'll do local completion (if any) ourselves
	  do_local_comp = comp->has_local_completions();

	  // remote completion needs to bounce off target
	  if(comp->has_remote_completions()) {
	    unsigned comp_info = ((comp->index << 2) +
				  PendingCompletion::REMOTE_PENDING_BIT);
	    arg0 |= (comp_info << MSGID_BITS);
	  }
	}

	if(module->cfg_do_checksums)
	  insert_packet_crc(arg0, header_base, header_size,
			    payload_base, payload_size);

	GASNetEXHandlers::commit_request_medium(msg->srcdesc,
						arg0,
						header_base,
						header_size,
						payload_size);

	XmitSrcDestPair *xpair = xmitsrcs[0]->lookup_pair(msg->target,
							  msg->target_ep_index);
	xpair->record_immediate_packet();

	break;
      }

    case PreparedMessage::STRAT_LONG_IMMEDIATE:
      {
	// long, attempt immediate FPAM, and enqueue if that doesn't work

	XmitSrcDestPair *xpair = xmitsrcs[0]->lookup_pair(msg->target,
							  msg->target_ep_index);

	size_t max_long_size =
	  GASNetEXHandlers::max_request_long(eps[0],
					     msg->target,
					     msg->target_ep_index,
					     header_size,
					     GEX_EVENT_GROUP /*dontcare*/,
					     0 /*flags*/);
	log_gex.debug() << "max long = " << max_long_size;
	if(payload_size > max_long_size) {
	  log_gex.fatal() << "long size exceeded: " << payload_size << " > " << max_long_size;
	  abort();
	}

#ifdef DEBUG_REALM
	// it's technically ok for a long srcptr to be out of segment,
	//  but it causes dynamic registration, which we want to avoid
	const SegmentInfo *src_seg = find_segment(payload_base);
	if(!src_seg) {
	  log_gex.fatal() << "HELP! long srcptr not in segment: ptr="
			  << payload_base;
	  abort();
	}
#endif

	gex_AM_Arg_t arg0 = msg->msgid;

	// we'll use lc_opt to handle dbuf use count decrements and/or
	//  caller requested local completion notification
	gex_Event_t done = GEX_EVENT_INVALID;
	gex_Event_t *lc_opt;
	// cache this because we can't look at 'comp' after sending the
	//  packet if it only had remote completions
	bool has_local = comp && comp->has_local_completions();
	if(msg->databuf || has_local)
	  lc_opt = &done;
	else
	  lc_opt = GEX_EVENT_GROUP; // don't care
	gex_Flags_t flags = GEX_FLAG_IMMEDIATE;

	// remote completion needs to bounce off target
	if(comp && comp->has_remote_completions()) {
	  unsigned comp_info = ((comp->index << 2) +
				PendingCompletion::REMOTE_PENDING_BIT);
	  arg0 |= (comp_info << MSGID_BITS);
	}

	// don't include the actual payload in crc for longs
	if(module->cfg_do_checksums)
	  insert_packet_crc(arg0, header_base, header_size,
			    nullptr, payload_size);

#ifdef DEBUG_REALM
        {
          size_t max_payload =
            GASNetEXHandlers::max_request_long(eps[0],
                                               msg->target,
                                               msg->target_ep_index,
                                               header_size,
                                               lc_opt,
                                               flags);
          if(payload_size > max_payload) {
            log_gex_xpair.fatal() << "long payload too large!  src="
                                  << Network::my_node_id << "/0"
                                  << " tgt=" << msg->target
                                  << "/" << msg->target_ep_index
                                  << " max=" << max_payload << " act=" << payload_size;
            abort();
          }
        }
#endif

	int ret = GASNetEXHandlers::send_request_long(eps[0],
						      msg->target,
						      msg->target_ep_index,
						      arg0,
						      header_base,
						      header_size,
						      payload_base,
						      payload_size,
						      lc_opt, flags,
						      msg->dest_payload_addr);
	if(ret == GASNET_OK) {
	  xpair->record_immediate_packet();

	  if(msg->databuf || has_local) {
	    GASNetEXEvent *ev = event_alloc.alloc_obj();
	    ev->set_event(done);
	    ev->set_databuf(msg->databuf);
	    if(has_local)
	      ev->set_local_comp(comp);
	    poller.add_pending_event(ev);
	  }
	} else {
	  log_gex_msg.info() << "immediate failed - queueing message";
	  // could not immediately inject it, so enqueue now
	  void *act_hdr_base = header_base;
	  OutbufMetadata *pktbuf;
	  int pktidx;
	  bool ok = xpair->reserve_pbuf_long_rget(header_size,
						  true /*overflow_ok*/,
						  pktbuf, pktidx,
						  act_hdr_base);
	  assert(ok); // can't handle backpressure at this point
	  // probably need to copy header
	  if(act_hdr_base != header_base)
	    memcpy(act_hdr_base, header_base, header_size);

	  // regenerate arg0 to include any local completion
	  // NOTE: we don't need to regenerate a checksum because
	  //  push_packets will strip the local completion back out
	  gex_AM_Arg_t arg0_with_local = msg->msgid;
	  if(comp) {
	    assert(comp->has_local_completions() ||
		   comp->has_remote_completions());

	    unsigned comp_info = ((comp->index << 2) +
				  (comp->state.load() & 3));
	    arg0_with_local |= (comp_info << MSGID_BITS);
	  }

	  // and now commit
	  xpair->commit_pbuf_long(pktbuf, pktidx, act_hdr_base,
				  arg0_with_local,
				  payload_base, payload_size,
				  msg->dest_payload_addr,
				  msg->databuf);
	}

	break;
      }

    case PreparedMessage::STRAT_LONG_PBUF:
      {
	// long, header already in a pktbuf to be queued

	gex_AM_Arg_t arg0 = msg->msgid;
	if(comp) {
	  // both local and remote completion are delegated to the target
	  assert(comp->has_local_completions() ||
		 comp->has_remote_completions());

	  unsigned comp_info = ((comp->index << 2) +
				(comp->state.load() & 3));
	  arg0 |= (comp_info << MSGID_BITS);
	}

	// don't include the actual payload in crc for longs
	if(module->cfg_do_checksums) {
	  // yuck...  push_packets is going to remove the local completion
	  // so make sure we compute the crc appropriately
	  gex_AM_Arg_t arg0_without_local;
	  if(comp && comp->has_remote_completions())
	    arg0_without_local = arg0 & ~(PendingCompletion::LOCAL_PENDING_BIT << MSGID_BITS);
	  else
	    arg0_without_local = msg->msgid;
	  insert_packet_crc(arg0_without_local, header_base, header_size,
			    nullptr, payload_size);
	}

	XmitSrcDestPair *xpair = xmitsrcs[0]->lookup_pair(msg->target,
							  msg->target_ep_index);
	xpair->commit_pbuf_long(msg->pktbuf, msg->pktidx,
				header_base,
				arg0, payload_base, payload_size,
				msg->dest_payload_addr,
				msg->databuf);
	break;
      }

#if 0
    case PreparedMessage::STRAT_LONG_DBUF_IMMEDIATE:
      {
	// long
	size_t max_long_size =
	  GASNetEXHandlers::max_request_long(eps[0],
					     msg->target,
					     msg->target_ep_index,
					     header_size,
					     GEX_EVENT_GROUP /*dontcare*/,
					     0 /*flags*/);
	log_gex.debug() << "max long = " << max_long_size;
	if(payload_size > max_long_size) {
	  log_gex.fatal() << "long size exceeded: " << payload_size << " > " << max_long_size;
	  abort();
	}

#ifdef DEBUG_REALM
	// it's technically ok for a long srcptr to be out of segment,
	//  but it causes dynamic registration, which we want to avoid
	const SegmentInfo *src_seg = find_segment(payload_base);
	if(!src_seg) {
	  log_gex.fatal() << "HELP! long srcptr not in segment: ptr="
			  << payload_base;
	  abort();
	}
#endif

	gex_AM_Arg_t arg0 = msg->msgid;

	// we need to add a local completion to reduce the dbuf use count
	// TODO: use lc_opt directly
	if(!comp)
	  comp = compmgr.get_available();
	{
	  size_t csize = sizeof(CompletionCallback<OutbufUsecountDec>);
	  void *ptr = comp->add_local_completion(csize, true /*late ok*/);
	  new(ptr) CompletionCallback<OutbufUsecountDec>(OutbufUsecountDec(msg->databuf));
	}

	unsigned comp_info = ((comp->index << 2) +
			      (comp->state.load() & 3));
	arg0 |= (comp_info << MSGID_BITS);

	// TODO: IMMEDIATE flag, lc_opt
	GASNetEXHandlers::send_request_long(eps[0],
					    msg->target,
					    msg->target_ep_index,
					    arg0,
					    header_base, header_size,
					    payload_base, payload_size,
					    GEX_EVENT_GROUP /*dontcare*/,
					    0 /*flags*/,
					    msg->dest_payload_addr);
	break;
      }
#endif
    case PreparedMessage::STRAT_RGET_IMMEDIATE:
      {
	// rget goes to prim endpoint on target
	XmitSrcDestPair *xpair = xmitsrcs[0]->lookup_pair(msg->target, 0);

	gex_AM_Arg_t arg0 = msg->msgid;

	if(msg->databuf) {
	  // we need to add a "local" completion to reduce the dbuf use count
	  if(!comp)
	    comp = compmgr.get_available();
	  {
	    size_t csize = sizeof(CompletionCallback<OutbufUsecountDec>);
	    void *ptr = comp->add_local_completion(csize, true /*late ok*/);
	    new(ptr) CompletionCallback<OutbufUsecountDec>(OutbufUsecountDec(msg->databuf));
	  }
	}
	if(comp) {
	  // both local and remote completion are delegated to the target
	  assert(comp->has_local_completions() ||
		 comp->has_remote_completions());

	  unsigned comp_info = ((comp->index << 2) +
				(comp->state.load() & 3));
	  arg0 |= (comp_info << MSGID_BITS);
	}

	// do not include payload in rget checksum - we may not be able to
	//  read it
	if(module->cfg_do_checksums)
	  insert_packet_crc(arg0, header_base, header_size,
			    nullptr, payload_size);

	// look up the source segment so we can send the ep_index to target
	const SegmentInfo *src_seg = find_segment(payload_base);
	assert(src_seg != 0);

	// an rget sends the header as payload, and we don't have a place
	//  to hold that, so insist it is sent/copied before GASNet returns
	gex_Event_t *lc_opt = GEX_EVENT_NOW;
	gex_Flags_t flags = GEX_FLAG_IMMEDIATE;

	int ret = GASNetEXHandlers::send_request_rget(prim_tm,
						      msg->target,
						      msg->target_ep_index,
						      arg0,
						      header_base,
						      header_size,
						      src_seg->ep_index,
						      payload_base,
						      payload_size,
						      lc_opt, flags,
						      msg->dest_payload_addr);
	if(ret == GASNET_OK) {
	  xpair->record_immediate_packet();
	} else {
	  log_gex_msg.info() << "immediate failed - queueing message";
	  // could not immediately inject it, so enqueue now
	  void *act_hdr_base = header_base;
	  OutbufMetadata *pktbuf;
	  int pktidx;
	  bool ok = xpair->reserve_pbuf_long_rget(header_size,
						  true /*overflow_ok*/,
						  pktbuf, pktidx,
						  act_hdr_base);
	  assert(ok); // can't handle backpressure at this point
	  // probably need to copy header
	  if(act_hdr_base != header_base)
	    memcpy(act_hdr_base, header_base, header_size);

	  xpair->commit_pbuf_rget(pktbuf, pktidx,
				  act_hdr_base,
				  arg0, payload_base, payload_size,
				  msg->dest_payload_addr,
				  src_seg->ep_index,
				  msg->target_ep_index);
	}

	break;
      }

    case PreparedMessage::STRAT_RGET_PBUF:
      {
	gex_AM_Arg_t arg0 = msg->msgid;

	if(msg->databuf) {
	  // we need to add a "local" completion to reduce the dbuf use count
	  if(!comp)
	    comp = compmgr.get_available();
	  {
	    size_t csize = sizeof(CompletionCallback<OutbufUsecountDec>);
	    void *ptr = comp->add_local_completion(csize, true /*late ok*/);
	    new(ptr) CompletionCallback<OutbufUsecountDec>(OutbufUsecountDec(msg->databuf));
	  }
	}
	if(comp) {
	  // both local and remote completion are delegated to the target
	  assert(comp->has_local_completions() ||
		 comp->has_remote_completions());

	  unsigned comp_info = ((comp->index << 2) +
				(comp->state.load() & 3));
	  arg0 |= (comp_info << MSGID_BITS);
	}

	// do not include payload in rget checksum - we may not be able to
	//  read it
	if(module->cfg_do_checksums)
	  insert_packet_crc(arg0, header_base, header_size,
			    nullptr, payload_size);

	// look up the source segment so we can send the ep_index to target
	const SegmentInfo *src_seg = find_segment(payload_base);
	assert(src_seg != 0);

	// rget goes to prim endpoint on target
	XmitSrcDestPair *xpair = xmitsrcs[0]->lookup_pair(msg->target, 0);

	xpair->commit_pbuf_rget(msg->pktbuf, msg->pktidx,
				header_base,
				arg0, payload_base, payload_size,
				msg->dest_payload_addr,
				src_seg->ep_index,
				msg->target_ep_index);
	break;
      }

    case PreparedMessage::STRAT_PUT_IMMEDIATE:
      {
#ifdef REALM_GEX_RMA_HONORS_IMMEDIATE_FLAG
	// rma put, header already in a PendingPutHeader, attempt to inject
        //  without using a pbuf

	XmitSrcDestPair *xpair = xmitsrcs[0]->lookup_pair(msg->target,
							  msg->target_ep_index);

	gex_AM_Arg_t arg0 = msg->msgid;
	if(comp) {
          // local completion can be signalled once the put is completed
          if(comp->has_local_completions())
            msg->put->local_comp = comp;

          // remote goes with the header's AM
          if(comp->has_remote_completions()) {
            unsigned comp_info = ((comp->index << 2) +
                                  PendingCompletion::REMOTE_PENDING_BIT);
            arg0 |= (comp_info << MSGID_BITS);
          }
	}
        msg->put->arg0 = arg0;
        msg->put->payload_bytes = payload_size;

	// don't include the actual payload in crc for longs
	if(module->cfg_do_checksums) {
	  insert_packet_crc(arg0, header_base, header_size,
			    nullptr, payload_size);
	}

        gex_Flags_t flags = GEX_FLAG_IMMEDIATE;

        // local completion just requires local completion of the payload,
        //  as we've already made a copy of the header, but only ask for
        //  it if the message needs it
        gex_Event_t lc_event = GEX_EVENT_INVALID;
        gex_Event_t *lc_opt = (msg->put->local_comp ?
                                 &lc_event :
                                 GEX_EVENT_DEFER);

#ifdef DEBUG_REALM
	const SegmentInfo *srcseg = find_segment(payload_base);
        assert(srcseg);
	assert(srcseg->ep_index == msg->source_ep_index);
#endif
        gex_TM_t pair = gex_TM_Pair(eps[msg->source_ep_index],
                                    msg->target_ep_index);
        gex_Event_t rc_event = gex_RMA_PutNB(pair,
                                             msg->target,
                                             reinterpret_cast<void *>(msg->dest_payload_addr),
                                             const_cast<void *>(payload_base),
                                             payload_size,
                                             lc_opt,
                                             flags);

        if(rc_event != GEX_EVENT_NO_OP) {
	  xpair->record_immediate_packet();

          GASNetEXEvent *leaf = 0;
          // local completion (if needed)
          if(msg->put->local_comp) {
            GASNetEXEvent *ev = event_alloc.alloc_obj();
            ev->set_event(lc_event);
            ev->set_local_comp(msg->put->local_comp);
            poller.add_pending_event(ev);
            leaf = ev;  // must be connected to root event below
          }

          // remote completion (always needed)
          {
            GASNetEXEvent *ev = event_alloc.alloc_obj();
            ev->set_event(rc_event);
            ev->set_put(msg->put);
            if(leaf)
              ev->set_leaf(leaf);
            poller.add_pending_event(ev);
          }
        } else {
	  log_gex_msg.info() << "immediate failed - queueing message";
	  // could not immediately inject it, so enqueue now
	  OutbufMetadata *pktbuf;
	  int pktidx;
          bool ok = xpair->reserve_pbuf_put(true /*overflow_ok*/,
                                            pktbuf, pktidx);
	  assert(ok); // can't handle backpressure at this point

          xpair->commit_pbuf_put(pktbuf, pktidx,
                                 msg->put,
                                 payload_base, payload_size,
                                 msg->dest_payload_addr);
        }
#else
        // should not have chosen this in prepare_message...
        log_gex.fatal() << "STRAT_PUT_IMMEDIATE used without immediate support!";
        abort();
#endif
        break;
      }

    case PreparedMessage::STRAT_PUT_PBUF:
      {
	// rma put, header already in a PendingPutHeader, put in pbuf

	gex_AM_Arg_t arg0 = msg->msgid;
	if(comp) {
          // local completion can be signalled once the put is completed
          if(comp->has_local_completions())
            msg->put->local_comp = comp;

          // remote goes with the header's AM
          if(comp->has_remote_completions()) {
            unsigned comp_info = ((comp->index << 2) +
                                  PendingCompletion::REMOTE_PENDING_BIT);
            arg0 |= (comp_info << MSGID_BITS);
          }
	}
        msg->put->arg0 = arg0;
        msg->put->payload_bytes = payload_size;

	// don't include the actual payload in crc for longs
	if(module->cfg_do_checksums) {
	  insert_packet_crc(arg0, header_base, header_size,
			    nullptr, payload_size);
	}

	XmitSrcDestPair *xpair = xmitsrcs[msg->source_ep_index]->lookup_pair(msg->target,
									     msg->target_ep_index);
	xpair->commit_pbuf_put(msg->pktbuf, msg->pktidx,
                               msg->put,
                               payload_base, payload_size,
                               msg->dest_payload_addr);
	break;
      }

#if 0
    case PreparedMessage::STRAT_RGET_DBUF_IMMEDIATE:
      {
	gex_AM_Arg_t arg0 = msg->msgid;

	// we need to add a "local" completion to reduce the dbuf use count
	if(!comp)
	  comp = compmgr.get_available();
	{
	  size_t csize = sizeof(CompletionCallback<OutbufUsecountDec>);
	  void *ptr = comp->add_local_completion(csize, true /*late ok*/);
	  new(ptr) CompletionCallback<OutbufUsecountDec>(OutbufUsecountDec(msg->databuf));
	}

	unsigned comp_info = ((comp->index << 2) +
			      (comp->state.load() & 3));
	arg0 |= (comp_info << MSGID_BITS);

	// look up the source segment so we can send the ep_index to target
	const SegmentInfo *src_seg = find_segment(payload_base);
	assert(src_seg != 0);

	// TODO: IMMEDIATE flag
	GASNetEXHandlers::send_request_rget(prim_tm,
					    msg->target,
					    msg->target_ep_index,
					    arg0,
					    header_base, header_size,
					    src_seg->ep_index,
					    payload_base, payload_size,
					    0 /*flags*/,
					    msg->dest_payload_addr);
	break;
      }
#endif
    default:
      assert(0);
    }
#if 0
    if(payload_size == 0) {
      // short
      gex_AM_Arg_t arg0 = msg->msgid;
      if(comp) {
	// we'll do local completion (if any) ourselves
	do_local_comp = comp->has_local_completions();

	// remote completion needs to bounce off target
	if(comp->has_remote_completions()) {
	  unsigned comp_info = ((comp->index << 2) +
				PendingCompletion::REMOTE_PENDING_BIT);
	  arg0 |= (comp_info << MSGID_BITS);
	}
      }
      GASNetEXHandlers::send_request_short(eps[0],
					   msg->target,
					   msg->target_ep_index,
					   arg0,
					   header_base, header_size,
					   0 /*flags*/);
    } else {
      const SegmentInfo *src_seg = find_segment(payload_base);
      if(src_seg)
	log_gex.info() << "srcptr " << payload_base << " in ep=" << src_seg->ep_index << " mtype=" << src_seg->memtype;
      else
	log_gex.info() << "srcptr " << payload_base << " not in segment";
      if(msg->dest_payload_addr == 0) {
	// medium
	size_t max_med_size =
	  GASNetEXHandlers::max_request_medium(eps[0],
					       msg->target,
					       msg->target_ep_index,
					       header_size,
					       GEX_EVENT_NOW,
					       0 /*flags*/);
	log_gex.debug() << "max med = " << max_med_size;
	if(payload_size > max_med_size) {
	  log_gex.fatal() << "medium size exceeded: " << payload_size << " > " << max_med_size;
	  abort();
	}

	// mediums are sent from the primordial ep, so the source should
	//  either be in its segment or none at all
	if(src_seg && (src_seg->ep_index != 0)) {
	  log_gex.fatal() << "HELP! medium srcptr in non-prim seg: ptr="
			  << payload_base << " ep_index=" << src_seg->ep_index;
	  abort();
	}

	gex_AM_Arg_t arg0 = msg->msgid;
	if(comp) {
	  // we'll do local completion (if any) ourselves
	  do_local_comp = comp->has_local_completions();

	  // remote completion needs to bounce off target
	  if(comp->has_remote_completions()) {
	    unsigned comp_info = ((comp->index << 2) +
				  PendingCompletion::REMOTE_PENDING_BIT);
	    arg0 |= (comp_info << MSGID_BITS);
	  }
	}
	GASNetEXHandlers::send_request_medium(eps[0],
					      msg->target,
					      msg->target_ep_index,
					      arg0,
					      header_base, header_size,
					      payload_base, payload_size,
					      GEX_EVENT_NOW,
					      0 /*flags*/);
      } else {
	// long
	size_t max_long_size =
	  GASNetEXHandlers::max_request_long(eps[0],
					     msg->target,
					     msg->target_ep_index,
					     header_size,
					     GEX_EVENT_GROUP /*dontcare*/,
					     0 /*flags*/);
	log_gex.debug() << "max long = " << max_long_size;
	if(payload_size > max_long_size) {
	  log_gex.fatal() << "long size exceeded: " << payload_size << " > " << max_long_size;
	  abort();
	}

	// it's technically ok for a long srcptr to be out of segment,
	//  but it causes dynamic registration, which we want to avoid
	if(!src_seg) {
	  log_gex.fatal() << "HELP! long srcptr not in segment: ptr="
			  << payload_base;
	  abort();
	}

	gex_AM_Arg_t arg0 = msg->msgid;
	if(comp) {
	  // both local and remote completion are delegated to the target
	  assert(comp->has_local_completions() ||
		 comp->has_remote_completions());

	  unsigned comp_info = ((comp->index << 2) +
				(comp->state.load() & 3));
	  arg0 |= (comp_info << MSGID_BITS);
	}
	if((src_seg->ep_index == 0) && (msg->target_ep_index == 0)) {
	  // use long AM for RMA between primordial endpoints
	  GASNetEXHandlers::send_request_long(eps[0],
					      msg->target,
					      msg->target_ep_index,
					      arg0,
					      header_base, header_size,
					      payload_base, payload_size,
					      GEX_EVENT_GROUP /*dontcare*/,
					      0 /*flags*/,
					      msg->dest_payload_addr);
	} else {
	  // currently have to use medium + RMA get if either src or tgt
	  //  is not the primordial endpoint
	  GASNetEXHandlers::send_request_rget(prim_tm,
					      msg->target,
					      msg->target_ep_index,
					      arg0,
					      header_base, header_size,
					      src_seg->ep_index,
					      payload_base, payload_size,
					      0 /*flags*/,
					      msg->dest_payload_addr);
	}
      }
    }
#endif
    if(do_local_comp) {
      if(comp->invoke_local_completions())
	compmgr.recycle_comp(comp);
    }

    prep_alloc.free_obj(msg);
  }

  void GASNetEXInternal::cancel_message(PreparedMessage *msg)
  {
    if(msg->temp_buffer)
      free(msg->temp_buffer);

    prep_alloc.free_obj(msg);
  }

  PendingCompletion *GASNetEXInternal::extract_arg0_local_comp(gex_AM_Arg_t& arg0)
  {
    unsigned comp_info = unsigned(arg0) >> MSGID_BITS;
    if((comp_info & PendingCompletion::LOCAL_PENDING_BIT) != 0) {
      PendingCompletion *comp = compmgr.lookup_completion(comp_info >> 2);
      // remove local bit, or whole thing if remote bit isn't set
      if((comp_info & PendingCompletion::REMOTE_PENDING_BIT) != 0)
	arg0 &= ~(PendingCompletion::LOCAL_PENDING_BIT << MSGID_BITS);
      else
	arg0 &= (1U << MSGID_BITS) - 1;
      return comp;
    } else {
      // no completion and no change
      return nullptr;
    }
  }

  /*static*/ void GASNetEXInternal::short_message_complete(NodeID sender,
							   uintptr_t objptr,
							   uintptr_t comp_info)
  {
    GASNetEXInternal *me = reinterpret_cast<GASNetEXInternal *>(objptr);

    XmitSrcDestPair *xpair = me->xmitsrcs[0]->lookup_pair(sender,
							  0 /*tgt_ep_index*/);
    xpair->enqueue_completion_reply(comp_info);
#if 0
    // TODO: handle backpressure here?
    gex_AM_Arg_t comp = comp_info;
    GASNetEXHandlers::send_completion_reply(me->eps[0],
					    sender,
					    0 /*tgt_ep_index*/,
					    &comp,
					    1,
					    0 /*flags*/);
#endif
  }

  /*static*/ void GASNetEXInternal::medium_message_complete(NodeID sender,
							    uintptr_t objptr,
							    uintptr_t comp_info)
  {
    GASNetEXInternal *me = reinterpret_cast<GASNetEXInternal *>(objptr);

    XmitSrcDestPair *xpair = me->xmitsrcs[0]->lookup_pair(sender,
							  0 /*tgt_ep_index*/);
    xpair->enqueue_completion_reply(comp_info);
#if 0
    // TODO: handle backpressure here?
    gex_AM_Arg_t comp = comp_info;
    GASNetEXHandlers::send_completion_reply(me->eps[0],
					    sender,
					    0 /*tgt_ep_index*/,
					    &comp,
					    1,
					    0 /*flags*/);
#endif
  }

  /*static*/ void GASNetEXInternal::long_message_complete(NodeID sender,
							  uintptr_t objptr,
							  uintptr_t comp_info)
  {
    GASNetEXInternal *me = reinterpret_cast<GASNetEXInternal *>(objptr);

    XmitSrcDestPair *xpair = me->xmitsrcs[0]->lookup_pair(sender,
							  0 /*tgt_ep_index*/);
    xpair->enqueue_completion_reply(comp_info);
#if 0
    // TODO: handle backpressure here?
    gex_AM_Arg_t comp = comp_info;
    GASNetEXHandlers::send_completion_reply(me->eps[0],
					    sender,
					    0 /*tgt_ep_index*/,
					    &comp,
					    1,
					    0 /*flags*/);
#endif
  }

  gex_AM_Arg_t GASNetEXInternal::handle_short(gex_Rank_t srcrank, gex_AM_Arg_t arg0,
					      const void *hdr, size_t hdr_bytes)
  {
    log_gex_msg.info() << "got short: " << srcrank << " "
		       << arg0 << " " << hdr_bytes;
    total_packets_received.fetch_add(1);

    if(module->cfg_do_checksums) {
      verify_packet_crc(arg0, hdr, hdr_bytes, nullptr, 0);
      hdr_bytes -= sizeof(gex_AM_Arg_t);
    }

    unsigned short msgid = arg0 & ((1U << MSGID_BITS) - 1);
    unsigned comp_info = unsigned(arg0) >> MSGID_BITS;
    // should never get a short message that wants local completion
    assert((comp_info & PendingCompletion::LOCAL_PENDING_BIT) == 0);

    // for a medium message, GASNet owns the payload, so the message manager
    //  needs to make a copy if it can't handle the message immediately
    IncomingMessageManager::CallbackFnptr cb_fnptr = 0;
    IncomingMessageManager::CallbackData cb_data1 = 0;
    IncomingMessageManager::CallbackData cb_data2 = 0;
    // request a callback if remote completion is needed
    if((comp_info & PendingCompletion::REMOTE_PENDING_BIT) != 0) {
      cb_fnptr = short_message_complete;
      cb_data1 = reinterpret_cast<uintptr_t>(this);
      cb_data2 = comp_info;
    }
    ThreadLocal::in_am_handler = true;
    bool handled = runtime->message_manager->add_incoming_message(srcrank,
								  msgid,
								  hdr,
								  hdr_bytes,
								  PAYLOAD_COPY,
								  nullptr,
								  0,
								  PAYLOAD_NONE,
								  cb_fnptr,
								  cb_data1,
								  cb_data2,
								  ((ThreadLocal::gex_work_until != nullptr) ?
								     *ThreadLocal::gex_work_until :
								     TimeLimit::relative(0)));
    ThreadLocal::in_am_handler = false;

    if(handled) {
      // if the message was handled immediately, we can use a reply for
      //  completion information
      return comp_info;
    } else {
      // remote isn't done and there's no local completion for shorts
      return 0;
    }
  }

  gex_AM_Arg_t GASNetEXInternal::handle_medium(gex_Rank_t srcrank, gex_AM_Arg_t arg0,
					       const void *hdr, size_t hdr_bytes,
					       const void *data, size_t data_bytes)
  {
    log_gex_msg.info() << "got medium: " << srcrank << " "
		       << arg0 << " " << hdr_bytes << " " << data_bytes;
    total_packets_received.fetch_add(1);

    if(module->cfg_do_checksums) {
      verify_packet_crc(arg0, hdr, hdr_bytes, data, data_bytes);
      hdr_bytes -= sizeof(gex_AM_Arg_t);
    }

    unsigned short msgid = arg0 & ((1U << MSGID_BITS) - 1);
    unsigned comp_info = unsigned(arg0) >> MSGID_BITS;

    // for a medium message, GASNet owns the payload, so the message manager
    //  needs to make a copy if it can't handle the message immediately
    IncomingMessageManager::CallbackFnptr cb_fnptr = 0;
    IncomingMessageManager::CallbackData cb_data1 = 0;
    IncomingMessageManager::CallbackData cb_data2 = 0;
    // request a callback if remote completion is needed
    if((comp_info & PendingCompletion::REMOTE_PENDING_BIT) != 0) {
      cb_fnptr = medium_message_complete;
      cb_data1 = reinterpret_cast<uintptr_t>(this);
      cb_data2 = comp_info & ~PendingCompletion::LOCAL_PENDING_BIT;
    }
    ThreadLocal::in_am_handler = true;
    bool handled = runtime->message_manager->add_incoming_message(srcrank,
								  msgid,
								  hdr,
								  hdr_bytes,
								  PAYLOAD_COPY,
								  data,
								  data_bytes,
								  PAYLOAD_COPY,
								  cb_fnptr,
								  cb_data1,
								  cb_data2,
								  ((ThreadLocal::gex_work_until != nullptr) ?
								     *ThreadLocal::gex_work_until :
								     TimeLimit::relative(0)));
    ThreadLocal::in_am_handler = false;

    if(handled) {
      // if the message was handled immediately, we can use a reply for
      //  completion information
      return comp_info;
    } else {
      if((comp_info & PendingCompletion::LOCAL_PENDING_BIT) != 0) {
        // reply is allowed for local completion, but remote isn't done
        return (comp_info &
                ~PendingCompletion::REMOTE_PENDING_BIT);
      } else
        return 0;
    }
  }

  gex_AM_Arg_t GASNetEXInternal::handle_long(gex_Rank_t srcrank, gex_AM_Arg_t arg0,
					     const void *hdr, size_t hdr_bytes,
					     const void *data, size_t data_bytes)
  {
    log_gex_msg.info() << "got long: " << srcrank << " "
		       << arg0 << " " << hdr_bytes << " " << data_bytes
                       << " " << data;
    total_packets_received.fetch_add(1);

    if(module->cfg_do_checksums) {
      verify_packet_crc(arg0, hdr, hdr_bytes, nullptr, data_bytes);
      hdr_bytes -= sizeof(gex_AM_Arg_t);
    }

    unsigned short msgid = arg0 & ((1U << MSGID_BITS) - 1);
    unsigned comp_info = unsigned(arg0) >> MSGID_BITS;

    IncomingMessageManager::CallbackFnptr cb_fnptr = 0;
    IncomingMessageManager::CallbackData cb_data1 = 0;
    IncomingMessageManager::CallbackData cb_data2 = 0;
    // request a callback if remote completion is needed
    if((comp_info & PendingCompletion::REMOTE_PENDING_BIT) != 0) {
      cb_fnptr = long_message_complete;
      cb_data1 = reinterpret_cast<uintptr_t>(this);
      cb_data2 = comp_info & ~PendingCompletion::LOCAL_PENDING_BIT;
    }
    ThreadLocal::in_am_handler = true;
    bool handled = runtime->message_manager->add_incoming_message(srcrank,
								  msgid,
								  hdr,
								  hdr_bytes,
								  PAYLOAD_COPY,
								  data,
								  data_bytes,
								  PAYLOAD_KEEP,
								  cb_fnptr,
								  cb_data1,
								  cb_data2,
								  ((ThreadLocal::gex_work_until != nullptr) ?
								     *ThreadLocal::gex_work_until :
								     TimeLimit::relative(0)));
    ThreadLocal::in_am_handler = false;

    if(handled) {
      // if the message was handled immediately, we can use a reply for
      //  completion information
      return comp_info;
    } else {
      if((comp_info & PendingCompletion::LOCAL_PENDING_BIT) != 0) {
        // reply is allowed for local completion, but remote isn't done
        return (comp_info &
                ~PendingCompletion::REMOTE_PENDING_BIT);
      } else
        return 0;
    }
  }

  void GASNetEXInternal::handle_reverse_get(gex_Rank_t srcrank, gex_EP_Index_t src_ep_index,
					    gex_EP_Index_t tgt_ep_index,
					    gex_AM_Arg_t arg0,
					    const void *hdr, size_t hdr_bytes,
					    uintptr_t src_ptr, uintptr_t tgt_ptr,
					    size_t payload_bytes)
  {
    log_gex_msg.info() << "got rget: " << srcrank << " "
		       << arg0 << " " << hdr_bytes << " " << payload_bytes;
    // don't increment total_packets_received here - we'll do that when we
    //  eventually handle this as a long (after data has been moved)

    // verify the crc here so that we can trust the payload_bytes amount,
    //  but don't shorten the packet because handle_long will want to check
    //  again
    if(module->cfg_do_checksums)
      verify_packet_crc(arg0, hdr, hdr_bytes, nullptr, payload_bytes);

    rgetter.add_reverse_get(srcrank, src_ep_index, tgt_ep_index,
			    arg0, hdr, hdr_bytes,
			    src_ptr, tgt_ptr, payload_bytes);
  }

  size_t GASNetEXInternal::handle_batch(gex_Rank_t srcrank, gex_AM_Arg_t arg0,
                                        gex_AM_Arg_t cksum,
					const void *data, size_t data_bytes,
					gex_AM_Arg_t *comps)
  {
    log_gex_msg.info() << "got batch: " << srcrank << " "
		       << arg0 << " " << data_bytes;

    int npkts = arg0;
    size_t ncomps = 0;
    uintptr_t baseptr = reinterpret_cast<uintptr_t>(data);
    uintptr_t ofs = 0;

    if(module->cfg_do_checksums) {
      uint32_t accum = 0xFFFFFFFF;
      accum = crc32c_accumulate(accum, &npkts, sizeof(npkts));
      accum = crc32c_accumulate(accum, &data_bytes, sizeof(data_bytes));
      accum = crc32c_accumulate(accum, data, data_bytes);
      gex_AM_Arg_t act_cksum = ~accum;
      if(act_cksum != cksum) {
        log_gex.fatal() << "CRC MISMATCH: batch_size=" << npkts
                        << " payload_size=" << data_bytes
                        << " exp=" << std::hex << cksum
                        << " act=" << act_cksum << std::dec;
        abort();
      }
    }

    for(int i = 0; i < npkts; i++) {
      assert((ofs + 2*sizeof(gex_AM_Arg_t)) <= data_bytes);
      const gex_AM_Arg_t *info =
	reinterpret_cast<const gex_AM_Arg_t *>(baseptr + ofs);

      size_t hdr_bytes = (info[0] & 0x3f) << 2;
      size_t payload_bytes = info[0] >> 6;
      gex_AM_Arg_t msg_arg0 = info[1];

      const void *hdr_data =
	reinterpret_cast<const void *>(baseptr + ofs +
				       2*sizeof(gex_AM_Arg_t));

      size_t pad_hdr_bytes = roundup_pow2(hdr_bytes + 2*sizeof(gex_AM_Arg_t),
					  16);

      if(payload_bytes == 0) {
	// short message
	ofs += pad_hdr_bytes;
	assert(ofs <= data_bytes);  // avoid overrun

	gex_AM_Arg_t comp = handle_short(srcrank, msg_arg0,
					 hdr_data, hdr_bytes);
	if(comp != 0)
	  comps[ncomps++] = comp;
      } else if(payload_bytes < ((1U << 22) - 1)) {
	// medium message

	const void *payload_data =
	  reinterpret_cast<const void *>(baseptr + ofs + pad_hdr_bytes);

	ofs += pad_hdr_bytes + roundup_pow2(payload_bytes, 16);
	assert(ofs <= data_bytes);  // avoid overrun

	gex_AM_Arg_t comp = handle_medium(srcrank, msg_arg0,
					  hdr_data, hdr_bytes,
					  payload_data, payload_bytes);
	if(comp != 0)
	  comps[ncomps++] = comp;
      } else {
	// reverse get

	const XmitSrcDestPair::LongRgetData *extra =
	  reinterpret_cast<const XmitSrcDestPair::LongRgetData *>(baseptr +
								  ofs +
								  pad_hdr_bytes);

	ofs += pad_hdr_bytes + roundup_pow2(sizeof(XmitSrcDestPair::LongRgetData), 16);
	assert(ofs <= data_bytes);  // avoid overrun

	handle_reverse_get(srcrank,
			   extra->r.src_ep_index,
			   extra->r.tgt_ep_index,
			   msg_arg0,
			   hdr_data, hdr_bytes,
			   reinterpret_cast<uintptr_t>(extra->payload_base),
			   extra->dest_addr,
			   extra->payload_bytes);
      }
    }
    // when the dust settles, we should have used exactly all the data
    assert(ofs == data_bytes);

    return ncomps;
  }

  void GASNetEXInternal::handle_completion_reply(gex_Rank_t srcrank,
						 const gex_AM_Arg_t *args,
						 size_t nargs)
  {
    ThreadLocal::in_am_handler = true;

    for(size_t i = 0; i < nargs; i++) {
      log_gex_comp.info() << "got comp: " << srcrank << " " << args[i];

      int index = args[i] >> 2;
      bool do_local = ((args[i] & PendingCompletion::LOCAL_PENDING_BIT) != 0);
      bool do_remote = ((args[i] & PendingCompletion::REMOTE_PENDING_BIT) != 0);
#ifdef DEBUG_REALM
      assert(do_local || do_remote);
#endif

      PendingCompletion *comp = compmgr.lookup_completion(index);
      compmgr.invoke_completions(comp, do_local, do_remote);
    }

    ThreadLocal::in_am_handler = false;
  }

  uintptr_t GASNetEXInternal::databuf_reserve(size_t bytes_needed,
					      OutbufMetadata **mdptr)
  {
    assert(bytes_needed <= module->cfg_outbuf_size);

    AutoLock<> al(databuf_mutex);
    if(databuf_md) {
      uintptr_t base = databuf_md->databuf_reserve(bytes_needed);
      if(base != 0) {
	if(mdptr) *mdptr = databuf_md;
	return base;
      }
      // failed - close this one out and get a new one
      databuf_md->databuf_close();
    }
    databuf_md = obmgr.alloc_outbuf(OutbufMetadata::STATE_DATABUF,
				    false /*!overflow_ok*/);
    if(!databuf_md)
      return 0;
    // fresh outbuf - this must succeed
    uintptr_t base = databuf_md->databuf_reserve(bytes_needed);
    assert(base != 0);
    if(mdptr) *mdptr = databuf_md;
    return base;
  }


}; // namespace Realm
