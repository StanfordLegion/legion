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

// IB (Intermediate Buffer) Memory implementations for Realm

#include "realm/transfer/ib_memory.h"

#include "realm/transfer/transfer.h"

namespace Realm {

  Logger log_ib_alloc("ib_alloc");
  extern Logger log_malloc;


  void free_intermediate_buffer(Memory mem, off_t offset, size_t size)
  {
    if(NodeID(ID(mem).memory_owner_node()) == Network::my_node_id) {
      get_runtime()->get_ib_memory_impl(mem)->free_bytes_local(offset, size);
    } else {
      ActiveMessage<RemoteIBReleaseSingle> amsg(ID(mem).memory_owner_node());
      amsg->memory = mem;
      amsg->offset = offset;
      amsg->size = size;
      amsg.commit();
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IBMemory
  //

  IBMemory::IBMemory(Memory _me, size_t _size,
                     MemoryKind _kind, Memory::Kind _lowlevel_kind,
                     void *prealloc_base, NetworkSegment *_segment)
    : MemoryImpl(_me, _size, _kind, _lowlevel_kind, _segment)
    , base(static_cast<char *>(prealloc_base))
    , ibreq_head(0)
    , ibreq_tail(&ibreq_head)
  {
    free_blocks[0] = _size;
  }

  IBMemory::~IBMemory()
  {
  }

  // old-style allocation used by IB memories
  // make bad offsets really obvious (+1 PB)
  static const off_t ZERO_SIZE_INSTANCE_OFFSET = 1ULL << ((sizeof(off_t) == 8) ? 50 : 30);

  off_t IBMemory::alloc_bytes_local(size_t size)
  {
    AutoLock<> al(mutex);

    off_t offset = do_alloc(size);
    return offset;
  }

  off_t IBMemory::do_alloc(size_t size)
  {
    // for zero-length allocations, return a special "offset"
    if(size == 0) {
      return this->size + ZERO_SIZE_INSTANCE_OFFSET;
    }

    const size_t alignment = 256;

    if(alignment > 0) {
      off_t leftover = size % alignment;
      if(leftover > 0) {
        log_malloc.info("padding allocation from %zd to %zd",
                        size, (size_t)(size + (alignment - leftover)));
        size += (alignment - leftover);
      }
    }
    // HACK: pad the size by a bit to see if we have people falling off
    //  the end of their allocations
    size += 0;

    // try to minimize footprint by allocating at the highest address possible
    if(!free_blocks.empty()) {
      std::map<off_t, off_t>::iterator it = free_blocks.end();
      do {
        --it;  // predecrement since we started at the end

        if(it->second == (off_t)size) {
          // perfect match
          off_t retval = it->first;
          free_blocks.erase(it);
          log_malloc.info("alloc full block: mem=" IDFMT " size=%zd ofs=%zd", me.id, size, (ssize_t)retval);
#if 0
          usage += size;
          if(usage > peak_usage) peak_usage = usage;
          size_t footprint = this->size - retval;
          if(footprint > peak_footprint) peak_footprint = footprint;
#endif
          return retval;
        }

        if(it->second > (off_t)size) {
          // some left over
          off_t leftover = it->second - size;
          off_t retval = it->first + leftover;
          it->second = leftover;
          log_malloc.info("alloc partial block: mem=" IDFMT " size=%zd ofs=%zd", me.id, size, (ssize_t)retval);
#if 0
          usage += size;
          if(usage > peak_usage) peak_usage = usage;
          size_t footprint = this->size - retval;
          if(footprint > peak_footprint) peak_footprint = footprint;
#endif
          return retval;
        }
      } while(it != free_blocks.begin());
    }

    // no blocks large enough - boo hoo
    log_malloc.info("alloc FAILED: mem=" IDFMT " size=%zd", me.id, size);
    return -1;
  }

  void IBMemory::free_bytes_local(off_t offset, size_t size)
  {
    log_malloc.info() << "free block: mem=" << me << " size=" << size << " ofs=" << offset;

    PendingIBRequests *satisfied;
    {
      AutoLock<> al(mutex);

      do_free(offset, size);

      satisfied = satisfy_pending_reqs();
    }

    if(satisfied)
      forward_satisfied_reqs(satisfied);
  }

  PendingIBRequests *IBMemory::satisfy_pending_reqs()
  {
    PendingIBRequests *reqs = ibreq_head;
    PendingIBRequests *last_sat = 0;

    // see if we can satisfy any pending requests
    while(reqs) {
#ifdef DEBUG_REALM
      assert(reqs->memories[reqs->current_req] == me);
#endif
      off_t offset = do_alloc(reqs->sizes[reqs->current_req]);
      if(offset == -1)
        break;
      reqs->offsets[reqs->current_req] = offset;

      unsigned next_req = reqs->current_req + 1;
      // have to be able to allocate any other requests to the same memory
      bool all_ok = true;
      while((next_req < reqs->count) &&
            (reqs->memories[next_req] == me)) {
        offset = do_alloc(reqs->sizes[next_req]);
        if(offset == -1) {
          all_ok = false;
          size_t bytes_needed = reqs->sizes[next_req];
          for(unsigned i = reqs->current_req; i < next_req; i++) {
            do_free(reqs->offsets[i], reqs->sizes[i]);
            reqs->offsets[i] = -1;
            bytes_needed += reqs->sizes[i];
          }
          // check that this could ever succeed
          if(bytes_needed > size) {
            log_ib_alloc.fatal() << "impossible: op=" << reqs->sender
                                 << "/0x" << std::hex << reqs->req_op << std::dec
                                 << " mem=" << me
                                 << " needed=" << bytes_needed << " avail=" << size;
            abort();
          }
          break;
        }
        reqs->offsets[next_req] = offset;
        next_req++;
      }
      if(!all_ok)
        break;

      log_ib_alloc.debug() << "satisfied: op=" << reqs->sender
                           << "/0x" << std::hex << reqs->req_op << std::dec
                           << " index=" << (reqs->first_req + reqs->current_req)
                           << "+" << (next_req - reqs->current_req)
                           << " mem=" << me;

      reqs->current_req = next_req;
      last_sat = reqs;
      reqs = reqs->next_req;
    }

    if(last_sat) {
      // up to and including last sat can be removed from the pending list
      //  and returned to the caller
      PendingIBRequests *first_sat = ibreq_head;
      ibreq_head = last_sat->next_req;
      if(!ibreq_head)
        ibreq_tail = &ibreq_head;
      last_sat->next_req = 0;
      return first_sat;
    } else
      return 0;
  }

  void IBMemory::forward_satisfied_reqs(PendingIBRequests *reqs)
  {
    while(reqs) {
      PendingIBRequests *next = reqs->next_req;
      reqs->next_req = 0;

      if(reqs->current_req == reqs->count) {
        // all done - return completed request(s) to original requestor
        if(reqs->sender == Network::my_node_id) {
          // oh, hey, that's us!
          TransferOperation *op = reinterpret_cast<TransferOperation *>(reqs->req_op);
          op->notify_ib_allocations(reqs->count, reqs->first_req,
                                    reqs->offsets.data());
        } else {
          if(reqs->count == 1) {
            ActiveMessage<RemoteIBAllocResponseSingle> amsg(reqs->sender);
            amsg->req_op = reqs->req_op;
            amsg->req_index = reqs->first_req;
            amsg->offset = reqs->offsets[0];
            amsg.commit();
          } else {
            size_t bytes = reqs->count * sizeof(off_t);
            ActiveMessage<RemoteIBAllocResponseMultiple> amsg(reqs->sender,
                                                              bytes);
            amsg->req_op = reqs->req_op;
            amsg->count = reqs->count;
            amsg->first_index = reqs->first_req;
            amsg.add_payload(reqs->offsets.data(), bytes);
            amsg.commit();
          }
        }

        delete reqs;
      } else {
        // on to the next memory
        Memory next_mem = reqs->memories[reqs->current_req];
        NodeID next_owner = ID(next_mem).memory_owner_node();
        if(next_owner == Network::my_node_id) {
          // local memory
          IBMemory *ibmem = get_runtime()->get_ib_memory_impl(next_mem);
          ibmem->enqueue_requests(reqs);
        } else {
          // remote - forward it on (since we satisfied some locally, we know
          //  this is a multiple case)
          size_t bytes = ((reqs->count * (sizeof(Memory) + sizeof(size_t))) +
                          (reqs->current_req * sizeof(off_t)));
          ActiveMessage<RemoteIBAllocRequestMultiple> amsg(next_owner, bytes);
          amsg->requestor = reqs->sender;
          amsg->count = reqs->count;
          amsg->first_index = reqs->first_req;
          amsg->curr_index = reqs->current_req;
          amsg->req_op = reqs->req_op;
          amsg->immediate = false;
          amsg.add_payload(reqs->memories.data(), reqs->count * sizeof(Memory));
          amsg.add_payload(reqs->sizes.data(), reqs->count * sizeof(size_t));
          amsg.add_payload(reqs->offsets.data(), reqs->current_req * sizeof(off_t));
          amsg.commit();

          delete reqs;
        }
      }

      reqs = next;
    }
  }

  void IBMemory::do_free(off_t offset, size_t size)
  {
    // frees of zero bytes should have the special offset
    if(size == 0) {
      assert((size_t)offset == this->size + ZERO_SIZE_INSTANCE_OFFSET);
      return;
    }

    const size_t alignment = 256;

    if(alignment > 0) {
      off_t leftover = size % alignment;
      if(leftover > 0) {
        log_malloc.info("padding free from %zd to %zd",
                        size, (size_t)(size + (alignment - leftover)));
        size += (alignment - leftover);
      }
    }

#if 0
    usage -= size;
    // only made things smaller, so can't impact the peak usage
#endif

    if(free_blocks.size() > 0) {
      // find the first existing block that comes _after_ us
      std::map<off_t, off_t>::iterator after = free_blocks.lower_bound(offset);
      if(after != free_blocks.end()) {
        // found one - is it the first one?
        if(after == free_blocks.begin()) {
          // yes, so no "before"
          assert((offset + (off_t)size) <= after->first); // no overlap!
          if((offset + (off_t)size) == after->first) {
            // merge the ranges by eating the "after"
            size += after->second;
            free_blocks.erase(after);
          }
          free_blocks[offset] = size;
        } else {
          // no, get range that comes before us too
          std::map<off_t, off_t>::iterator before = after; before--;

          // if we're adjacent to the after, merge with it
          assert((offset + (off_t)size) <= after->first); // no overlap!
          if((offset + (off_t)size) == after->first) {
            // merge the ranges by eating the "after"
            size += after->second;
            free_blocks.erase(after);
          }

          // if we're adjacent with the before, grow it instead of adding
          //  a new range
          assert((before->first + before->second) <= offset);
          if((before->first + before->second) == offset) {
            before->second += size;
          } else {
            free_blocks[offset] = size;
          }
        }
      } else {
        // nothing's after us, so just see if we can merge with the range
        //  that's before us

        std::map<off_t, off_t>::iterator before = after; before--;

        // if we're adjacent with the before, grow it instead of adding
        //  a new range
        assert((before->first + before->second) <= offset);
        if((before->first + before->second) == offset) {
          before->second += size;
        } else {
          free_blocks[offset] = size;
        }
      }
    } else {
      // easy case - nothing was free, so now just our block is
      free_blocks[offset] = size;
    }
  }

  void *IBMemory::get_direct_ptr(off_t offset, size_t size)
  {
    assert(NodeID(ID(me).memory_owner_node()) == Network::my_node_id);
    assert((offset >= 0) && ((size_t)(offset + size) <= this->size));
    return (base + offset);
  }

  // not used by IB memories
  MemoryImpl::AllocationResult IBMemory::allocate_storage_immediate(RegionInstanceImpl *inst,
                                                                    bool need_alloc_result,
                                                                    bool poisoned,
                                                                    TimeLimit work_until)
  {
    abort();
  }

  void IBMemory::release_storage_immediate(RegionInstanceImpl *inst,
                                           bool poisoned,
                                           TimeLimit work_until)
  {
    abort();
  }

  void IBMemory::get_bytes(off_t offset, void *dst, size_t size)
  {
    abort();
  }

  void IBMemory::put_bytes(off_t offset, const void *src, size_t size)
  {
    abort();
  }

  bool IBMemory::attempt_immediate_allocation(NodeID requestor,
                                              uintptr_t req_op,
                                              size_t count,
                                              const size_t *sizes,
                                              off_t *offsets)
  {
    assert(NodeID(ID(me).memory_owner_node()) == Network::my_node_id);
    AutoLock<> al(mutex);

    // if there are pending requests, we can't cut in line
    if(ibreq_head)
      return false;

    for(size_t i = 0; i < count; i++) {
      off_t offset = do_alloc(sizes[i]);
      if(offset == -1) {
        // failed - give back any we did allocate
        size_t bytes_needed = sizes[i];
        for(size_t j = 0; j < i; j++) {
          do_free(offsets[j], sizes[j]);
          offsets[j] = -1;
          bytes_needed += sizes[j];
        }
        // check that this could ever succeed
        if(bytes_needed > size) {
          log_ib_alloc.fatal() << "impossible: op=" << requestor
                               << "/0x" << std::hex << req_op << std::dec
                               << " mem=" << me
                               << " needed=" << bytes_needed << " avail=" << size;
          abort();
        }
        return false;
      }
      offsets[i] = offset;
    }

    return true;
  }

  void IBMemory::enqueue_requests(PendingIBRequests *reqs)
  {
    assert(NodeID(ID(reqs->memories[reqs->current_req]).memory_owner_node()) == Network::my_node_id);
    log_ib_alloc.debug() << "pending: op=" << reqs->sender
                         << "/0x" << std::hex << reqs->req_op << std::dec
                         << " index=" << (reqs->first_req + reqs->current_req)
                         << "+" << (reqs->count - reqs->current_req)
                         << " mem=" << me;

    PendingIBRequests *satisfied = 0;
    {
      AutoLock<> al(mutex);

      bool was_empty = (ibreq_head == 0);
      *ibreq_tail = reqs;
      ibreq_tail = &reqs->next_req;

      if(was_empty)
        satisfied = satisfy_pending_reqs();
    }

    if(satisfied)
      forward_satisfied_reqs(satisfied);
  }

  void IBMemory::free_multiple(size_t count,
                               const off_t *offsets, const size_t *sizes)
  {
    PendingIBRequests *satisfied;
    {
      AutoLock<> al(mutex);

      for(size_t i = 0; i < count; i++)
        do_free(offsets[i], sizes[i]);

      satisfied = satisfy_pending_reqs();
    }

    if(satisfied)
      forward_satisfied_reqs(satisfied);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PendingIBRequests
  //

  PendingIBRequests::PendingIBRequests(NodeID _sender, uintptr_t _req_op,
                                       unsigned _count,
                                       unsigned _first_req, unsigned _current_req)
    : next_req(0)
    , sender(_sender)
    , req_op(_req_op)
    , count(_count)
    , first_req(_first_req)
    , current_req(_current_req)
  {
    // no memory/size data yet, but we know how much space we'll need
    memories.reserve(_count);
    sizes.reserve(_count);
    offsets.resize(_count, -1);
  }

  PendingIBRequests::PendingIBRequests(NodeID _sender, uintptr_t _req_op,
                                       unsigned _count,
                                       unsigned _first_req, unsigned _current_req,
                                       const Memory *_memories,
                                       const size_t *_sizes,
                                       const off_t *_offsets)
    : next_req(0)
    , sender(_sender)
    , req_op(_req_op)
    , count(_count)
    , first_req(_first_req)
    , current_req(_current_req)
  {
    memories.assign(_memories, _memories + _count);
    sizes.assign(_sizes, _sizes + _count);
    offsets.resize(_count, -1);
    if(_current_req > 0) {
      assert(_offsets);
      memcpy(offsets.data(), _offsets, _current_req * sizeof(off_t));
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // struct RemoteIBAllocRequestSingle
  //

  /*static*/ void RemoteIBAllocRequestSingle::handle_message(NodeID sender,
                                                             const RemoteIBAllocRequestSingle &args,
                                                             const void *data,
                                                             size_t msglen)
  {
    IBMemory *ib_mem = get_runtime()->get_ib_memory_impl(args.memory);

    // always attempt immediate first - return offset on success or even on
    //  failure if sender insisted on an immediate response
    off_t offset = -1;
    if(ib_mem->attempt_immediate_allocation(sender, args.req_op,
                                            1, &args.size, &offset) ||
       args.immediate) {
      log_ib_alloc.debug() << "satisfied: op=" << sender
                           << "/0x" << std::hex << args.req_op << std::dec
                           << " index=" << args.req_index << "+1"
                           << " mem=" << args.memory;

      ActiveMessage<RemoteIBAllocResponseSingle> amsg(sender);
      amsg->req_op = args.req_op;
      amsg->req_index = args.req_index;
      amsg->offset = offset;
      amsg.commit();
      return;
    }

    // create a PendingIBRequests block and enqueue it
    PendingIBRequests *reqs = new PendingIBRequests(sender, args.req_op,
                                                    1, args.req_index, 0,
                                                    &args.memory, &args.size,
                                                    0);
    ib_mem->enqueue_requests(reqs);
  }

  ActiveMessageHandlerReg<RemoteIBAllocRequestSingle> remote_ib_alloc_req_single_handler;


  ////////////////////////////////////////////////////////////////////////
  //
  // struct RemoteIBAllocRequestMultiple
  //

  /*static*/ void RemoteIBAllocRequestMultiple::handle_message(NodeID sender,
                                                               const RemoteIBAllocRequestMultiple &args,
                                                               const void *data,
                                                               size_t msglen)
  {
    assert(msglen == ((args.count * (sizeof(Memory) + sizeof(size_t))) +
                      (args.curr_index * sizeof(off_t))));
    const Memory *memories = static_cast<const Memory *>(data);
    const size_t *sizes = reinterpret_cast<const size_t *>(static_cast<const char *>(data) +
                                                           (args.count * sizeof(Memory)));
    const off_t *offsets = reinterpret_cast<const off_t *>(static_cast<const char *>(data) +
                                                           (args.count * (sizeof(Memory) +
                                                                          sizeof(size_t))));

    unsigned rem_count = args.count - args.curr_index;
    assert(rem_count > 0);

    // see what we can satisfy immediately
    unsigned immed_count = 0;
    off_t *immed_offsets = static_cast<off_t *>(alloca(rem_count *
                                                       sizeof(off_t)));
    while(true) {
      Memory tgt_mem = memories[args.curr_index + immed_count];
      NodeID owner = ID(tgt_mem).memory_owner_node();
      // if we've gotten to IB requests that are non-local, stop
      if(owner != Network::my_node_id) {
        assert(immed_count > 0); // shouldn't happen on first request
        break;
      }

      unsigned same_mem = 1;
      while(((immed_count + same_mem) < args.count) &&
            (tgt_mem == memories[args.curr_index + immed_count + same_mem]))
        same_mem += 1;

      IBMemory *ib_mem = get_runtime()->get_ib_memory_impl(tgt_mem);
      if(ib_mem->attempt_immediate_allocation(args.requestor, args.req_op,
                                              same_mem,
                                              sizes + args.curr_index + immed_count,
                                              immed_offsets + immed_count)) {
        log_ib_alloc.debug() << "satisfied: op=" << args.requestor
                             << "/0x" << std::hex << args.req_op << std::dec
                             << " index=" << (args.first_index + args.curr_index + immed_count)
                             << "+" << same_mem
                             << " mem=" << tgt_mem;

        immed_count += same_mem;
        assert(immed_count <= rem_count);
        if(immed_count == rem_count) {
          if(args.requestor == Network::my_node_id) {
            // local notification - do in two parts because the offsets are in
            //  two different arrays
            TransferOperation *op = reinterpret_cast<TransferOperation *>(args.req_op);
            assert(args.curr_index > 0);  // shouldn't be here if all ibs were local
            op->notify_ib_allocations(args.curr_index, args.first_index,
                                      offsets);
            op->notify_ib_allocations(immed_count,
                                      args.first_index + args.curr_index,
                                      immed_offsets);
          } else {
            ActiveMessage<RemoteIBAllocResponseMultiple> amsg(args.requestor,
                                                              args.count * sizeof(off_t));
            amsg->req_op = args.req_op;
            amsg->count = args.count;
            amsg->first_index = args.first_index;
            amsg.add_payload(offsets, args.curr_index * sizeof(off_t));
            amsg.add_payload(immed_offsets, immed_count * sizeof(off_t));
            amsg.commit();
          }

          return;
        }
      } else {
        // TODO: handle early return of a partially-failed immediate request
        assert(!args.immediate);
        break;
      }
    }

    // still some left to do - are they local?
    Memory tgt_mem = memories[args.curr_index + immed_count];
    NodeID owner = ID(tgt_mem).memory_owner_node();
    if(owner == Network::my_node_id) {
      // initialize the PendingIBRequests with the data from the incoming
      //  message, and then add any immediate successes we had
      PendingIBRequests *reqs = new PendingIBRequests(args.requestor,
                                                      args.req_op,
                                                      args.count,
                                                      args.first_index,
                                                      args.curr_index,
                                                      memories,
                                                      sizes,
                                                      offsets);
      if(immed_count) {
        for(unsigned i = 0; i < immed_count; i++)
          reqs->offsets[args.curr_index + i] = immed_offsets[i];
        reqs->current_req += immed_count;
      }

      IBMemory *ib_mem = get_runtime()->get_ib_memory_impl(tgt_mem);
      ib_mem->enqueue_requests(reqs);
    } else {
      // new payload is previous payload with any new offsets tacked on the end
      size_t bytes = msglen + (immed_count * sizeof(off_t));
      ActiveMessage<RemoteIBAllocRequestMultiple> amsg(owner, bytes);
      amsg->requestor = args.requestor;
      amsg->count = args.count;
      amsg->first_index = args.first_index;
      amsg->curr_index = args.curr_index + immed_count;
      amsg->req_op = args.req_op;
      amsg->immediate = args.immediate;
      amsg.add_payload(data, msglen);
      assert(immed_count > 0);
      amsg.add_payload(immed_offsets, immed_count * sizeof(off_t));
      amsg.commit();
    }
  }

  ActiveMessageHandlerReg<RemoteIBAllocRequestMultiple> remote_ib_alloc_req_multi_handler;


  ////////////////////////////////////////////////////////////////////////
  //
  // struct RemoteIBAllocResponseSingle
  //

  /*static*/ void RemoteIBAllocResponseSingle::handle_message(NodeID sender,
                                                              const RemoteIBAllocResponseSingle &args,
                                                              const void *data,
                                                              size_t msglen)
  {
    TransferOperation *op = reinterpret_cast<TransferOperation *>(args.req_op);
    op->notify_ib_allocation(args.req_index, args.offset);
  }

  ActiveMessageHandlerReg<RemoteIBAllocResponseSingle> remote_ib_alloc_resp_single_handler;


  ////////////////////////////////////////////////////////////////////////
  //
  // struct RemoteIBAllocResponseMultiple
  //

  /*static*/ void RemoteIBAllocResponseMultiple::handle_message(NodeID sender,
                                                                const RemoteIBAllocResponseMultiple &args,
                                                                const void *data,
                                                                size_t msglen)
  {
    TransferOperation *op = reinterpret_cast<TransferOperation *>(args.req_op);
    op->notify_ib_allocations(args.count, args.first_index,
                              ((msglen > 0) ? static_cast<const off_t *>(data) :
                                              0));
  }

  ActiveMessageHandlerReg<RemoteIBAllocResponseMultiple> remote_ib_alloc_resp_multi_handler;


  ////////////////////////////////////////////////////////////////////////
  //
  // struct RemoteIBReleaseSingle
  //

  /*static*/ void RemoteIBReleaseSingle::handle_message(NodeID sender,
                                                        const RemoteIBReleaseSingle &args,
                                                        const void *data,
                                                        size_t msglen)
  {
    IBMemory *ib_mem = get_runtime()->get_ib_memory_impl(args.memory);
    ib_mem->free_bytes_local(args.offset, args.size);
  }

  ActiveMessageHandlerReg<RemoteIBReleaseSingle> remote_ib_release_single_handler;


  ////////////////////////////////////////////////////////////////////////
  //
  // struct RemoteIBReleaseMultiple
  //

  /*static*/ void RemoteIBReleaseMultiple::handle_message(NodeID sender,
                                                          const RemoteIBReleaseMultiple &args,
                                                          const void *data,
                                                          size_t msglen)
  {
    assert(msglen == (args.count * (sizeof(Memory) + sizeof(size_t) + sizeof(off_t))));
    const Memory *memories = static_cast<const Memory *>(data);
    const size_t *sizes = reinterpret_cast<const size_t *>(static_cast<const char *>(data) +
                                                           (args.count * sizeof(Memory)));
    const off_t *offsets = reinterpret_cast<const off_t *>(static_cast<const char *>(data) +
                                                           (args.count * (sizeof(Memory) +
                                                                          sizeof(size_t))));

    unsigned i = 0;
    while(i < args.count) {
      // count how many (consecutive) entries are for the same ibmem
      unsigned same_mem = 1;
      while(((i + same_mem) < args.count) &&
            (memories[i] == memories[i + same_mem]))
        same_mem++;

      IBMemory *ib_mem = get_runtime()->get_ib_memory_impl(memories[i]);
      ib_mem->free_multiple(same_mem, offsets+i, sizes+i);

      i += same_mem;
    }
  }

  ActiveMessageHandlerReg<RemoteIBReleaseMultiple> remote_ib_release_multi_handler;


};
