/* Copyright 2022 Stanford University, NVIDIA Corporation
 * Copyright 2022 Los Alamos National Laboratory
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
#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/channel.h"
#include "realm/threads.h"
#include "realm/transfer/transfer.h"
#include "realm/transfer/channel_disk.h"
#include "realm/transfer/ib_memory.h"

#include <errno.h>
// included for file memory data transfer
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
#include <unistd.h>
#endif
#ifdef REALM_USE_KERNEL_AIO
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#endif
#ifdef REALM_USE_LIBAIO
#include <aio.h>
#endif

#include <queue>
#include <algorithm>
#include <iomanip>

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

#include "realm/timers.h"
#include "realm/serialize.h"

namespace Realm {

    Logger log_dma("dma");
    //extern Logger log_new_dma;
    Logger log_aio("aio");

    static atomic<unsigned> rdma_sequence_no(1);

    static AsyncFileIOContext *aio_context = 0;

#ifdef REALM_USE_KERNEL_AIO
    inline int io_setup(unsigned nr, aio_context_t *ctxp)
    {
      return syscall(__NR_io_setup, nr, ctxp);
    }

    inline int io_destroy(aio_context_t ctx)
    {
      return syscall(__NR_io_destroy, ctx);
    }

    inline int io_submit(aio_context_t ctx, long nr, struct iocb **iocbpp)
    {
      return syscall(__NR_io_submit, ctx, nr, iocbpp);
    }

    inline int io_getevents(aio_context_t ctx, long min_nr, long max_nr,
                            struct io_event *events, struct timespec *timeout)
    {
      return syscall(__NR_io_getevents, ctx, min_nr, max_nr, events, timeout);
    }

    class KernelAIOWrite : public AsyncFileIOContext::AIOOperation {
    public:
      KernelAIOWrite(aio_context_t aio_ctx,
                     int fd, size_t offset, size_t bytes,
		     const void *buffer, Request* request = NULL);
      virtual void launch(void);
      virtual bool check_completion(void);

    public:
      aio_context_t ctx;
      struct iocb cb;
    };

    KernelAIOWrite::KernelAIOWrite(aio_context_t aio_ctx,
				   int fd, size_t offset, size_t bytes,
				   const void *buffer, Request* request)
    {
      completed = false;
      ctx = aio_ctx;
      memset(&cb, 0, sizeof(cb));
      cb.aio_data = (uint64_t)this;
      cb.aio_fildes = fd;
      cb.aio_lio_opcode = IOCB_CMD_PWRITE;
      cb.aio_buf = (uint64_t)buffer;
      cb.aio_offset = offset;
      cb.aio_nbytes = bytes;
      req = request;
    }

    void KernelAIOWrite::launch(void)
    {
      struct iocb *cbs[1];
      cbs[0] = &cb;
      log_aio.debug("write issued: op=%p cb=%p", this, &cb);
#ifndef NDEBUG
      int ret =
#endif
	io_submit(ctx, 1, cbs);
      assert(ret == 1);
    }

    bool KernelAIOWrite::check_completion(void)
    {
      return completed;
    }

    class KernelAIORead : public AsyncFileIOContext::AIOOperation {
    public:
      KernelAIORead(aio_context_t aio_ctx,
                     int fd, size_t offset, size_t bytes,
		     void *buffer, Request* request = NULL);
      virtual void launch(void);
      virtual bool check_completion(void);

    public:
      aio_context_t ctx;
      struct iocb cb;
    };

    KernelAIORead::KernelAIORead(aio_context_t aio_ctx,
				 int fd, size_t offset, size_t bytes,
				 void *buffer, Request* request)
    {
      completed = false;
      ctx = aio_ctx;
      memset(&cb, 0, sizeof(cb));
      cb.aio_data = (uint64_t)this;
      cb.aio_fildes = fd;
      cb.aio_lio_opcode = IOCB_CMD_PREAD;
      cb.aio_buf = (uint64_t)buffer;
      cb.aio_offset = offset;
      cb.aio_nbytes = bytes;
      req = request;
    }

    void KernelAIORead::launch(void)
    {
      struct iocb *cbs[1];
      cbs[0] = &cb;
      log_aio.debug("read issued: op=%p cb=%p", this, &cb);
#ifndef NDEBUG
      int ret =
#endif
	io_submit(ctx, 1, cbs);
      assert(ret == 1);
    }

    bool KernelAIORead::check_completion(void)
    {
      return completed;
    }
#endif

#ifdef REALM_USE_LIBAIO
    class PosixAIOWrite : public AsyncFileIOContext::AIOOperation {
    public:
      PosixAIOWrite(int fd, size_t offset, size_t bytes,
		    const void *buffer, Request* request = NULL);
      virtual void launch(void);
      virtual bool check_completion(void);

    public:
      struct aiocb cb;
    };

    PosixAIOWrite::PosixAIOWrite(int fd, size_t offset, size_t bytes,
				 const void *buffer, Request* request)
    {
      completed = false;
      memset(&cb, 0, sizeof(cb));
      cb.aio_fildes = fd;
      cb.aio_buf = (void *)buffer;
      cb.aio_offset = offset;
      cb.aio_nbytes = bytes;
      req = request;
    }

    void PosixAIOWrite::launch(void)
    {
      log_aio.debug("write issued: op=%p cb=%p", this, &cb);
#ifndef NDEBUG
      int ret =
#endif
	aio_write(&cb);
      assert(ret == 0);
    }

    bool PosixAIOWrite::check_completion(void)
    {
      int ret = aio_error(&cb);
      if(ret == EINPROGRESS) return false;
      log_aio.debug("write returned: op=%p cb=%p ret=%d", this, &cb, ret);
      assert(ret == 0);
      return true;
    }

    class PosixAIORead : public AsyncFileIOContext::AIOOperation {
    public:
      PosixAIORead(int fd, size_t offset, size_t bytes,
		   void *buffer, Request* request = NULL);
      virtual void launch(void);
      virtual bool check_completion(void);

    public:
      struct aiocb cb;
    };

    PosixAIORead::PosixAIORead(int fd, size_t offset, size_t bytes,
			       void *buffer, Request* request)
    {
      completed = false;
      memset(&cb, 0, sizeof(cb));
      cb.aio_fildes = fd;
      cb.aio_buf = buffer;
      cb.aio_offset = offset;
      cb.aio_nbytes = bytes;
      req = request;
    }

    void PosixAIORead::launch(void)
    {
      log_aio.debug("read issued: op=%p cb=%p", this, &cb);
#ifndef NDEBUG
      int ret =
#endif
	aio_read(&cb);
      assert(ret == 0);
    }

    bool PosixAIORead::check_completion(void)
    {
      int ret = aio_error(&cb);
      if(ret == EINPROGRESS) return false;
      log_aio.debug("read returned: op=%p cb=%p ret=%d", this, &cb, ret);
      assert(ret == 0);
      return true;
    }
#endif

    class AIOFence : public Operation::AsyncWorkItem {
    public:
      AIOFence(Operation *_op) : Operation::AsyncWorkItem(_op) {}
      virtual void request_cancellation(void) {}
      virtual void print(std::ostream& os) const { os << "AIOFence"; }
    };

    class AIOFenceOp : public AsyncFileIOContext::AIOOperation {
    public:
      AIOFenceOp(Operation *_req);
      virtual void launch(void);
      virtual bool check_completion(void);

    public:
      Operation *req;
      AIOFence *f;
    };

    AIOFenceOp::AIOFenceOp(Operation *_req)
    {
      completed = false;
      req = _req;
      f = new AIOFence(req);
      req->add_async_work_item(f);
    }

    void AIOFenceOp::launch(void)
    {
      log_aio.debug("fence launched: op=%p req=%p", this, req);
      completed = true;
    }

    bool AIOFenceOp::check_completion(void)
    {
      assert(completed);
      log_aio.debug("fence completed: op=%p req=%p", this, req);
      f->mark_finished(true /*successful*/);
      return true;
    }

    AsyncFileIOContext::AsyncFileIOContext(int _max_depth)
      : BackgroundWorkItem("async file IO")
      , max_depth(_max_depth)
    {
#ifdef REALM_USE_KERNEL_AIO
      aio_ctx = 0;
#ifndef NDEBUG
      int ret =
#endif
	io_setup(max_depth, &aio_ctx);
      assert(ret == 0);
#endif
    }

    AsyncFileIOContext::~AsyncFileIOContext(void)
    {
      assert(pending_operations.empty());
      assert(launched_operations.empty());
#ifdef REALM_USE_KERNEL_AIO
#ifndef NDEBUG
      int ret =
#endif
	io_destroy(aio_ctx);
      assert(ret == 0);
#endif
    }

    void AsyncFileIOContext::enqueue_write(int fd, size_t offset, 
					   size_t bytes, const void *buffer,
                                           Request* req)
    {
#ifdef REALM_USE_KERNEL_AIO
      KernelAIOWrite *op = new KernelAIOWrite(aio_ctx,
					      fd, offset, bytes, buffer, req);
#elif defined(REALM_USE_LIBAIO)
      PosixAIOWrite *op = new PosixAIOWrite(fd, offset, bytes, buffer, req);
#else
      AsyncFileIOContext::AIOOperation* op = 0;
      assert(0);
#endif
      bool was_empty;
      {
	AutoLock<> al(mutex);
	was_empty = launched_operations.empty();
	if(launched_operations.size() < (size_t)max_depth) {
	  op->launch();
	  launched_operations.push_back(op);
	} else {
	  pending_operations.push_back(op);
	}
      }
      if(was_empty)
	make_active();
    }

    void AsyncFileIOContext::enqueue_read(int fd, size_t offset, 
					  size_t bytes, void *buffer,
                                          Request* req)
    {
#ifdef REALM_USE_KERNEL_AIO
      KernelAIORead *op = new KernelAIORead(aio_ctx,
					    fd, offset, bytes, buffer, req);
#elif defined(REALM_USE_LIBAIO)
      PosixAIORead *op = new PosixAIORead(fd, offset, bytes, buffer, req);
#else
      AsyncFileIOContext::AIOOperation* op = 0;
      assert(0);
#endif
      bool was_empty;
      {
	AutoLock<> al(mutex);
	was_empty = launched_operations.empty();
	if(launched_operations.size() < (size_t)max_depth) {
	  op->launch();
	  launched_operations.push_back(op);
	} else {
	  pending_operations.push_back(op);
	}
      }
      if(was_empty)
	make_active();
    }

    void AsyncFileIOContext::enqueue_fence(Operation *req)
    {
      AIOFenceOp *op = new AIOFenceOp(req);
      bool was_empty;
      {
	AutoLock<> al(mutex);
	was_empty = launched_operations.empty();
	if(launched_operations.size() < (size_t)max_depth) {
	  op->launch();
	  launched_operations.push_back(op);
	} else {
	  pending_operations.push_back(op);
	}
      }
      if(was_empty)
	make_active();
    }

    bool AsyncFileIOContext::empty(void)
    {
      AutoLock<> al(mutex);
      return launched_operations.empty();
    }

    long AsyncFileIOContext::available(void)
    {
      AutoLock<> al(mutex);
      return (max_depth - launched_operations.size());
    }

    void AsyncFileIOContext::make_progress(void)
    {
      AutoLock<> al(mutex);

      // first, reap as many events as we can - oldest first
#ifdef REALM_USE_KERNEL_AIO
      while(true) {
	struct io_event events[8];
	struct timespec ts;
	ts.tv_sec = 0;
	ts.tv_nsec = 0;  // no delay
	int ret = io_getevents(aio_ctx, 1, 8, events, &ts);
	if(ret == 0) break;
	log_aio.debug("io_getevents returned %d events", ret);
	for(int i = 0; i < ret; i++) {
	  AIOOperation *op = (AIOOperation *)(events[i].data);
	  log_aio.debug("io_getevents: event[%d] = %p", i, op);
	  op->completed = true;
	}
      }
#endif

      // now actually mark events completed in oldest-first order
      while(!launched_operations.empty()) {
	AIOOperation *op = launched_operations.front();
	if(!op->check_completion()) break;
	log_aio.debug("aio op completed: op=%p", op);
        // <NEW_DMA>
        if (op->req != NULL) {
          Request* request = (Request*)(op->req);
          request->xd->notify_request_read_done(request);
          request->xd->notify_request_write_done(request);
        }
        // </NEW_DMA>
	delete op;
	launched_operations.pop_front();
      }

      // finally, if there are any pending ops, and room for them, launch them
      while((launched_operations.size() < (size_t)max_depth) &&
	    !pending_operations.empty()) {
	AIOOperation *op = pending_operations.front();
	pending_operations.pop_front();
	op->launch();
	launched_operations.push_back(op);
      }
    }

    bool AsyncFileIOContext::do_work(TimeLimit work_until)
    {
      // first, reap as many events as we can - oldest first
#ifdef REALM_USE_KERNEL_AIO
      assert(!launched_operations.empty());
      while {
	struct io_event events[8];
	struct timespec ts;
	ts.tv_sec = 0;
	ts.tv_nsec = 0;  // no delay
	int ret = io_getevents(aio_ctx, 1, 8, events, &ts);
	if(ret == 0) break;
	log_aio.debug("io_getevents returned %d events", ret);
	for(int i = 0; i < ret; i++) {
	  AIOOperation *op = (AIOOperation *)(events[i].data);
	  log_aio.debug("io_getevents: event[%d] = %p", i, op);
	  op->completed = true;
	}
	// only try again if we got a full set the time before
	if(ret < 8) break;
      } while(!work_until.is_expired());
#endif

      // have to check for completed ops even if we didn't get any this
      //  time - there may have been some last time that we didn't have
      //  enough time to notify
      {
	AutoLock<> al(mutex);

	// now actually mark events completed in oldest-first order
	while(!work_until.is_expired()) {
	  AIOOperation *op = launched_operations.front();
	  if(!op->check_completion()) break;
	  log_aio.debug("aio op completed: op=%p", op);
	  // <NEW_DMA>
	  if (op->req != NULL) {
	    Request* request = (Request*)(op->req);
	    request->xd->notify_request_read_done(request);
	    request->xd->notify_request_write_done(request);
	  }
	  // </NEW_DMA>
	  delete op;
	  launched_operations.pop_front();
	  if(launched_operations.empty()) {
	    if(pending_operations.empty()) {
	      // finished work - we can return without requeuing ourselves
	      return false;
	    } else {
	      // launch some pending work below
	      break;
	    }
	  }
	}

	// finally, if there are any pending ops, and room for them, and
	//   time left, launch them
	while((launched_operations.size() < (size_t)max_depth) &&
	      !pending_operations.empty() &&
	      !work_until.is_expired()) {
	  AIOOperation *op = pending_operations.front();
	  pending_operations.pop_front();
	  op->launch();
	  launched_operations.push_back(op);
	}
      }

      // if we fall through to here, there's still polling for either old
      //  or newly launched work to do
      return true;
    }

    /*static*/
    AsyncFileIOContext* AsyncFileIOContext::get_singleton() {
      return aio_context;
    }

    Channel *get_xfer_channel(Memory src_mem, Memory dst_mem,
			      CustomSerdezID src_serdez_id,
			      CustomSerdezID dst_serdez_id,
			      ReductionOpID redop_id,
			      XferDesKind *pkind = 0)
    {
      Channel *channel = 0;
      XferDesKind kind = XFER_NONE;

      // look at the dma channels available on the source node
      NodeID src_node = ID(src_mem).memory_owner_node();
      NodeID dst_node = ID(dst_mem).memory_owner_node();
      const Node& n = get_runtime()->nodes[src_node];
      for(std::vector<Channel *>::const_iterator it = n.dma_channels.begin();
	  it != n.dma_channels.end();
	  ++it) {
	unsigned bw = 0;
	unsigned latency = 0;
	if((*it)->supports_path(src_mem, dst_mem,
				src_serdez_id, dst_serdez_id,
				redop_id,
                                0, 0, 0, // FIXME
				&kind, &bw, &latency)) {
	  channel = *it;
	  break;
	}
      }

      // if that didn't work, try the destination node (if different)
      if((kind == XFER_NONE) && (dst_node != src_node)) {
	const Node& n = get_runtime()->nodes[dst_node];
	for(std::vector<Channel *>::const_iterator it = n.dma_channels.begin();
	    it != n.dma_channels.end();
	    ++it) {
	  unsigned bw = 0;
	  unsigned latency = 0;
	  if((*it)->supports_path(src_mem, dst_mem,
				  src_serdez_id, dst_serdez_id,
				  redop_id,
                                  0, 0, 0, // FIXME
				  &kind, &bw, &latency)) {
	    channel = *it;
	    break;
	  }
	}
      }

      if(pkind) *pkind = kind;
      return channel;
    }

    bool find_shortest_path(Memory src_mem, Memory dst_mem,
			    CustomSerdezID serdez_id,
                            ReductionOpID redop_id,
			    MemPathInfo& info,
			    bool skip_final_memcpy /*= false*/)
    {
      // make sure we write a fresh MemPathInfo
      info.path.clear();
      info.xd_channels.clear();
      //info.xd_kinds.clear();
      //info.xd_target_nodes.clear();

      // fast case - can we go straight from src to dst?
      XferDesKind kind;
      Channel *channel = get_xfer_channel(src_mem, dst_mem,
					  serdez_id, serdez_id,
                                          redop_id, &kind);
      if(channel) {
	info.path.push_back(src_mem);
	if(!skip_final_memcpy || (kind != XFER_MEM_CPY)) {
	  info.path.push_back(dst_mem);
	  //info.xd_kinds.push_back(kind);
	  info.xd_channels.push_back(channel);
	}
      } else {
	std::map<Memory, std::vector<Memory> > dist;
	std::map<Memory, std::vector<Channel *> > channels;
	std::list<Memory> mems_left;
	std::queue<Memory> active_nodes;
	Node* node = &(get_runtime()->nodes[ID(src_mem).memory_owner_node()]);
	for (std::vector<IBMemory*>::const_iterator it = node->ib_memories.begin();
           it != node->ib_memories.end(); it++) {
	  mems_left.push_back((*it)->me);
	}
	if(ID(dst_mem).memory_owner_node() != ID(src_mem).memory_owner_node()) {
	  node = &(get_runtime()->nodes[ID(dst_mem).memory_owner_node()]);
	  for (std::vector<IBMemory*>::const_iterator it = node->ib_memories.begin();
	       it != node->ib_memories.end(); it++) {
	    mems_left.push_back((*it)->me);
	  }
	}
	for(std::list<Memory>::iterator it = mems_left.begin(); it != mems_left.end(); ) {
	  // we know we're doing at least one hop, so no dst_serdez or redop here
	  channel = get_xfer_channel(src_mem, *it, serdez_id, 0, 0);
	  if(channel) {
	    std::vector<Memory>& v = dist[*it];
	    v.push_back(src_mem);
	    v.push_back(*it);
	    channels[*it].push_back(channel);
	    active_nodes.push(*it);
	    it = mems_left.erase(it);
	  } else
	    ++it;
	}
	while(true) {
	  if(active_nodes.empty())
	    return false;

	  Memory cur = active_nodes.front();
	  active_nodes.pop();
	  std::vector<Memory> sub_path = dist[cur];
	  
	  // can we reach the destination from here (handling potential
	  //  deserialization or reduction)
	  channel = get_xfer_channel(cur, dst_mem, 0, serdez_id, redop_id, &kind);
	  if(channel) {
	    info.path = dist[cur];
	    //info.xd_kinds = kinds[cur];
	    info.xd_channels = channels[cur];
	    if(!skip_final_memcpy || (kind != XFER_MEM_CPY)) {
	      info.path.push_back(dst_mem);
	      //info.xd_kinds.push_back(kind);
	      info.xd_channels.push_back(channel);
	    }
	    break;
	  }

	  // no, look for another intermediate hop
	  for(std::list<Memory>::iterator it = mems_left.begin(); it != mems_left.end(); ) {
	    channel = get_xfer_channel(cur, *it, 0, 0, 0);
	    if(channel) {
	      std::vector<Memory>& v = dist[*it];
	      v = dist[cur];
	      v.push_back(*it);
	      //std::vector<XferDesKind>& k = kinds[*it];
	      //k = kinds[cur];
	      //k.push_back(kind);
	      std::vector<Channel *>& c = channels[*it];
	      c = channels[cur];
	      c.push_back(channel);
	      active_nodes.push(*it);
	      it = mems_left.erase(it);
	    } else
	      ++it;
	  }
	}
      }

      return true;
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class WrappingFIFOIterator
  //

  WrappingFIFOIterator::WrappingFIFOIterator(size_t _base, size_t _size)
    : base(_base)
    , size(_size)
    , offset(0)
    , tentative_valid(false)
  {}

  template <typename S>
  /*static*/ TransferIterator *WrappingFIFOIterator::deserialize_new(S& deserializer)
  {
    size_t base, size, offset, prev_offset;
    bool tentative_valid;
    if((deserializer >> base) &&
       (deserializer >> size) &&
       (deserializer >> offset) &&
       (deserializer >> prev_offset) &&
       (deserializer >> tentative_valid)) {
      WrappingFIFOIterator *wfi = new WrappingFIFOIterator(base, size);
      wfi->offset = offset;
      wfi->prev_offset = prev_offset;
      wfi->tentative_valid = tentative_valid;
      return wfi;
    } else
      return 0;
  }   

  void WrappingFIFOIterator::reset(void)
  {
    offset = 0;
  }

  bool WrappingFIFOIterator::done(void)
  {
    // we never know when we're done
    return false;
  }

  size_t WrappingFIFOIterator::step(size_t max_bytes, AddressInfo &info,
				    unsigned flags,
				    bool tentative /*= false*/)
  {
    assert(!tentative_valid);

    if(tentative) {
      prev_offset = offset;
      tentative_valid = true;
    }

    info.base_offset = base + offset;
    info.num_lines = 1;
    info.line_stride = 0;
    info.num_planes = 1;
    info.plane_stride = 0;
    size_t bytes;
    size_t bytes_left = size - offset;
    if(bytes_left <= max_bytes) {
      offset = 0;
      bytes = bytes_left;
    } else {
      offset += max_bytes;
      bytes = max_bytes;
    }
    info.bytes_per_chunk = bytes;
    return bytes;
  }

  size_t WrappingFIFOIterator::step_custom(size_t max_bytes,
                                           AddressInfoCustom& info,
                                           bool tentative /*= false*/)
  {
    // not supported
    assert(0);
    return 0;
  }

  void WrappingFIFOIterator::confirm_step(void)
  {
    assert(tentative_valid);
    tentative_valid = false;
  }

  void WrappingFIFOIterator::cancel_step(void)
  {
    assert(tentative_valid);
    offset = prev_offset;
    tentative_valid = false;
  }

  bool WrappingFIFOIterator::get_addresses(AddressList &addrlist,
                                           const InstanceLayoutPieceBase *&nonaffine)
  {
    nonaffine = 0;

    // add a very tall 2d "rectangle" that uses a stride of 0 for the
    //  line pitch (flow control will prevent any given copy from touching
    //  the same location twice)
    size_t lines = std::max<size_t>((1 << 30) / size, 1);
    int dim = (lines > 1) ? 2 : 1;
    size_t *data = addrlist.begin_nd_entry(dim);
    if(!data)
      return true;  // can't add more until some is consumed


    // 1-D span from [base,base+size)
    data[0] = (size << 4) + 2 /*dim*/;
    data[1] = base;
    if(dim == 2) {
      data[2] = lines;
      data[3] = 0; // stride
    }
    addrlist.commit_nd_entry(dim, size * lines);

    return false;  // we can add more if asked
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<TransferIterator, WrappingFIFOIterator> WrappingFIFOIterator::serdez_subclass;

  template <typename S>
  bool WrappingFIFOIterator::serialize(S& serializer) const
  {
    return ((serializer << base) &&
	    (serializer << size) &&
	    (serializer << offset) &&
	    (serializer << prev_offset) &&
	    (serializer << tentative_valid));
  }


    void start_dma_system(BackgroundWorkManager *bgwork)
    {
      aio_context = new AsyncFileIOContext(256);
      aio_context->add_to_manager(bgwork);
    }

    void stop_dma_system(void)
    {
#ifdef DEBUG_REALM
      aio_context->shutdown_work_item();
#endif
      delete aio_context;
      aio_context = 0;
    }

};
