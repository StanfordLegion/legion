/* Copyright 2018 Stanford University
 * Copyright 2018 Los Alamos National Laboratory
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
#ifndef LOWLEVEL_CHANNEL
#define LOWLEVEL_CHANNEL

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <map>
#include <vector>
#include <deque>
#include <queue>
#include <assert.h>
#include <pthread.h>
#include <string.h>
#include "realm/transfer/lowlevel_dma.h"

#include "realm/id.h"
#include "realm/runtime_impl.h"
#include "realm/mem_impl.h"
#include "realm/inst_impl.h"

#ifdef USE_CUDA
#include "realm/cuda/cuda_module.h"
#endif

#ifdef USE_HDF
#include "realm/hdf5/hdf5_internal.h"
#endif

namespace Realm {

    class XferDes;
    class Channel;

    extern Logger log_new_dma;

    class Buffer {
    public:
      enum MemoryKind {
        MKIND_CPUMEM,
        MKIND_GPUFB,
        MKIND_DISK
      };

      enum {
        MAX_SERIALIZATION_LEN = 5 * sizeof(int64_t) / sizeof(int)
      };

      Buffer(void)
            : alloc_offset(0), is_ib(false), block_size(0), elmt_size(0),
              buf_size(0), memory(Memory::NO_MEMORY) {}

      Buffer(RegionInstanceImpl::Metadata* metadata, Memory _memory)
            : alloc_offset(metadata->alloc_offset),
              is_ib(false), block_size(metadata->block_size), elmt_size(metadata->elmt_size),
              buf_size(metadata->size),
              memory(_memory){}

      Buffer(off_t _alloc_offset, bool _is_ib,
             int _block_size, int _elmt_size, size_t _buf_size,
             Memory _memory)
            : alloc_offset(_alloc_offset),
              is_ib(_is_ib), block_size(_block_size), elmt_size(_elmt_size),
              buf_size(_buf_size),
              memory(_memory){}

      Buffer& operator=(const Buffer& other)
      {
        alloc_offset = other.alloc_offset;
        is_ib = other.is_ib;
        block_size = other.block_size;
        elmt_size = other.elmt_size;
        buf_size = other.buf_size;
        memory = other.memory;
        return *this;
      }


      ~Buffer() {
      }

      // Note that we don't serialize memory in current implementation
      // User has to manually set memory after deserialize
      void serialize(int* data) const
      {
        int64_t* data64 = (int64_t*) data;
        *data64 = alloc_offset; data64++;
        *data64 = is_ib; data64++;
        *data64 = block_size; data64++;
        *data64 = elmt_size; data64++;
        *data64 = buf_size; data64++;
      }

      void deserialize(const int* data)
      {
        int64_t* cur = (int64_t*) data;
        alloc_offset = *cur; cur++;
        is_ib = *cur; cur++;
        block_size = *cur; cur++;
        elmt_size = *cur; cur++;
        buf_size = *cur; cur++;
      }

      enum DimensionKind {
        DIM_X, // first logical index space dimension
        DIM_Y, // second logical index space dimension
        DIM_Z, // ...
        DIM_F, // field dimension
        INNER_DIM_X, // inner dimension for tiling X
        OUTER_DIM_X, // outer dimension for tiling X
        INNER_DIM_Y, // ...
        OUTER_DIM_Y,
        INNER_DIM_Z,
        OUTER_DIM_Z,
        INNER_DIM_F,
        OUTER_DIM_F,
      };

      // std::vector<size_t> field_ordering;
      // std::vector<size_t> field_sizes;
      // std::vector<DimensionKind> dimension_ordering;
      // std::vector<size_t> dim_size;
      off_t alloc_offset;
      bool is_ib;
      size_t block_size, elmt_size;
      //int inner_stride[3], outer_stride[3], inner_dim_size[3];

      //MemoryKind memory_kind;

      // buffer size of this intermediate buffer.
      // 0 indicates this buffer is large enough to hold
      // entire data set.
      // A number smaller than bytes_total means we need
      // to reuse the buffer.
      size_t buf_size;

      // The memory instance on which this buffer relies
      Memory memory;
    };

    class Request {
    public:
      enum Dimension {
        DIM_1D,
        DIM_2D,
	DIM_3D
      };
      // a pointer to the owning xfer descriptor
      // this should set at Request creation
      XferDes* xd;
      // src/dst offset in the src/dst instance
      off_t src_off, dst_off;
      // src/dst (line) strides
      off_t src_str, dst_str;
      // src/dst plane strides
      off_t src_pstr, dst_pstr;
      // number of bytes being transferred
      size_t nbytes, nlines, nplanes;
      // a flag indicating whether this request read has been done
      bool is_read_done;
      // a flag indicating whether this request write has been done
      bool is_write_done;
      // whether I am a 1D or 2D transfer
      Dimension dim;
      // sequence info - used to update read/write counts, handle reordering
      size_t read_seq_pos, read_seq_count;
      size_t write_seq_pos, write_seq_count;
    };

    class SequenceAssembler {
    public:
      SequenceAssembler(void);
      SequenceAssembler(const SequenceAssembler& copy_from);
      ~SequenceAssembler(void);

      void swap(SequenceAssembler& other);

      // asks if a span exists - return value is number of bytes from the
      //  start that do
      size_t span_exists(size_t start, size_t count);

      // returns the amount by which the contiguous range has been increased
      //  (i.e. from [pos, pos+retval) )
      size_t add_span(size_t pos, size_t count);

    protected:
      GASNetHSL mutex;
      size_t contig_amount;  // everything from [0, contig_amount) is covered
      size_t first_noncontig; // nothing in [contig_amount, first_noncontig) 
      std::map<size_t, size_t> spans;  // noncontiguous spans
    };

    class MemcpyRequest : public Request {
    public:
      const void *src_base;
      void *dst_base;
      //size_t nbytes;
    };

    class GASNetRequest : public Request {
    public:
      void *mem_base; // could be source or dest
      off_t gas_off;
      //off_t src_offset;
      //size_t nbytes;
    };

    class RemoteWriteRequest : public Request {
    public:
      NodeID dst_node;
      const void *src_base;
      void *dst_base;
      //size_t nbytes;
    };

#ifdef USE_CUDA
    class GPUCompletionEvent : public Cuda::GPUCompletionNotification {
    public:
      GPUCompletionEvent(void) {triggered = false;}
      void request_completed(void) {triggered = true;}
      void reset(void) {triggered = false;}
      bool has_triggered(void) {return triggered;}
    private:
      bool triggered;
    };

    class GPURequest : public Request {
    public:
      const void *src_base;
      void *dst_base;
      off_t src_gpu_off, dst_gpu_off;
      Cuda::GPU* dst_gpu;
      GPUCompletionEvent event;
    };
#endif

#ifdef USE_HDF
    class HDFRequest : public Request {
    public:
      void *mem_base; // could be source or dest
      hid_t dataset_id, datatype_id;
      hid_t mem_space_id, file_space_id;
    };
#endif

    class XferOrder {
    public:
      enum Type {
        SRC_FIFO,
        DST_FIFO,
        ANY_ORDER
      };
    };

    class XferDesFence : public Realm::Operation::AsyncWorkItem {
    public:
      XferDesFence(Realm::Operation *op) : Realm::Operation::AsyncWorkItem(op) {}
      virtual void request_cancellation(void) {
    	// ignored for now
      }
      virtual void print(std::ostream& os) const { os << "XferDesFence"; }
    };

    class XferDes {
    public:
      enum XferKind {
        XFER_NONE,
        XFER_DISK_READ,
        XFER_DISK_WRITE,
        XFER_SSD_READ,
        XFER_SSD_WRITE,
        XFER_GPU_TO_FB,
        XFER_GPU_FROM_FB,
        XFER_GPU_IN_FB,
        XFER_GPU_PEER_FB,
        XFER_MEM_CPY,
        XFER_GASNET_READ,
        XFER_GASNET_WRITE,
        XFER_REMOTE_WRITE,
        XFER_HDF_READ,
        XFER_HDF_WRITE,
        XFER_FILE_READ,
        XFER_FILE_WRITE
      };
    public:
      // a pointer to the DmaRequest that contains this XferDes
      DmaRequest* dma_request;
      // a boolean indicating if we have marked started
      bool mark_start;
      // ID of the node that launches this XferDes
      NodeID launch_node;
      //uint64_t /*bytes_submit, */bytes_read, bytes_write/*, bytes_total*/;
      bool iteration_completed;
      uint64_t read_bytes_total, write_bytes_total; // not valid until iteration_completed == true
      uint64_t read_bytes_cons, write_bytes_cons; // used for serdez, updated atomically
      uint64_t pre_bytes_total;
      //uint64_t next_bytes_read;
      MemoryImpl *src_mem;
      MemoryImpl *dst_mem;
      TransferIterator *src_iter;
      TransferIterator *dst_iter;
      const CustomSerdezUntyped *src_serdez_op;
      const CustomSerdezUntyped *dst_serdez_op;
      size_t src_ib_offset, src_ib_size;
      // maximum size for a single request
      uint64_t max_req_size;
      // priority of the containing XferDes
      int priority;
      // current, previous and next XferDes in the chain, XFERDES_NO_GUID
      // means this XferDes is the first/last one.
      XferDesID guid, pre_xd_guid, next_xd_guid;
      // XferKind of the Xfer Descriptor
      XferKind kind;
      // XferOrder of the Xfer Descriptor
      XferOrder::Type order;
      SequenceAssembler seq_read, seq_write, seq_pre_write, seq_next_read;
      // queue that contains all available free requests
      std::queue<Request*> available_reqs;
      enum {
        XFERDES_NO_GUID = 0
      };
      // channel this XferDes describes
      Channel* channel;
      // event is triggered when the XferDes is completed
      XferDesFence* complete_fence;
      // xd_lock is designed to provide thread-safety for
      // SIMULTANEOUS invocation to get_requests,
      // notify_request_read_done, and notify_request_write_done
      pthread_mutex_t xd_lock, update_read_lock, update_write_lock;
      // default iterators provided to generate requests
      //Layouts::GenericLayoutIterator<DIM>* li;
      unsigned offset_idx;
    public:
      XferDes(DmaRequest* _dma_request, NodeID _launch_node,
              XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
	      uint64_t _next_max_rw_gap, size_t src_ib_offset, size_t src_ib_size,
              bool _mark_start,
	      Memory _src_mem, Memory _dst_mem,
	      TransferIterator *_src_iter, TransferIterator *_dst_iter,
	      CustomSerdezID _src_serdez_id, CustomSerdezID _dst_serdez_id,
              uint64_t _max_req_size, int _priority,
              XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence);

      virtual ~XferDes();

      virtual long get_requests(Request** requests, long nr) = 0;

      virtual void notify_request_read_done(Request* req) = 0;

      virtual void notify_request_write_done(Request* req) = 0;

      virtual void flush() = 0;
 
      long default_get_requests(Request** requests, long nr, unsigned flags = 0);
      void default_notify_request_read_done(Request* req);
      void default_notify_request_write_done(Request* req);

      virtual void update_bytes_read(size_t offset, size_t size);
      virtual void update_bytes_write(size_t offset, size_t size);
      void update_pre_bytes_write(size_t offset, size_t size, size_t pre_bytes_total);
      void update_next_bytes_read(size_t offset, size_t size);

      bool is_completed(void);

      void mark_completed();

#if 0
      void update_pre_bytes_write(size_t new_val) {
        pthread_mutex_lock(&update_write_lock);
        if (pre_bytes_write < new_val)
          pre_bytes_write = new_val;
        /*uint64_t old_val = pre_bytes_write;
        while (old_val < new_val) {
          pre_bytes_write.compare_exchange_strong(old_val, new_val);
        }*/
        pthread_mutex_unlock(&update_write_lock);
      }

      void update_next_bytes_read(size_t new_val) {
        pthread_mutex_lock(&update_read_lock);
        if (next_bytes_read < new_val)
          next_bytes_read = new_val;
        /*uint64_t old_val = next_bytes_read;
        while (old_val < new_val) {
          next_bytes_read.compare_exchange_strong(old_val, new_val);
        }*/
        pthread_mutex_unlock(&update_read_lock);
      }
#endif

      Request* dequeue_request() {
        Request* req = available_reqs.front();
	available_reqs.pop();
	req->is_read_done = false;
	req->is_write_done = false;
        return req;
      }

      void enqueue_request(Request* req) {
        available_reqs.push(req);
      }

    };

    class MemcpyXferDes : public XferDes {
    public:
      MemcpyXferDes(DmaRequest* _dma_request, NodeID _launch_node,
                    XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
		    uint64_t _next_max_rw_gap, size_t src_ib_offset, size_t src_ib_size,
                    bool mark_started,
		    Memory _src_mem, Memory _dst_mem,
		    TransferIterator *_src_iter, TransferIterator *_dst_iter,
		    CustomSerdezID _src_serdez_id, CustomSerdezID _dst_serdez_id,
                    uint64_t max_req_size, long max_nr, int _priority,
                    XferOrder::Type _order, XferDesFence* _complete_fence);

      ~MemcpyXferDes()
      {
        free(memcpy_reqs);
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

    private:
      MemcpyRequest* memcpy_reqs;
      //const char *src_buf_base, *dst_buf_base;
    };

    class GASNetXferDes : public XferDes {
    public:
      GASNetXferDes(DmaRequest* _dma_request, NodeID _launch_node,
                    XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
		    uint64_t _next_max_rw_gap, size_t src_ib_offset, size_t src_ib_size,
                    bool mark_started,
		    Memory _src_mem, Memory _dst_mem,
		    TransferIterator *_src_iter, TransferIterator *_dst_iter,
		    CustomSerdezID _src_serdez_id, CustomSerdezID _dst_serdez_id,
                    uint64_t _max_req_size, long max_nr, int _priority,
                    XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence);

      ~GASNetXferDes()
      {
        free(gasnet_reqs);
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

    private:
      GASNetRequest* gasnet_reqs;
    };

    class RemoteWriteXferDes : public XferDes {
    public:
      RemoteWriteXferDes(DmaRequest* _dma_request, NodeID _launch_node,
                         XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
			 uint64_t _next_max_rw_gap, size_t src_ib_offset, size_t src_ib_size,
                         bool mark_started,
			 Memory _src_mem, Memory _dst_mem,
			 TransferIterator *_src_iter, TransferIterator *_dst_iter,
			 CustomSerdezID _src_serdez_id, CustomSerdezID _dst_serdez_id,
                         uint64_t max_req_size, long max_nr, int _priority,
                         XferOrder::Type _order, XferDesFence* _complete_fence);

      ~RemoteWriteXferDes()
      {
        free(requests);
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

      // doesn't do pre_bytes_write updates, since the remote write message
      //  takes care of it with lower latency
      virtual void update_bytes_write(size_t offset, size_t size);

    private:
      RemoteWriteRequest* requests;
      char *dst_buf_base;
    };

#ifdef USE_CUDA
    class GPUXferDes : public XferDes {
    public:
      GPUXferDes(DmaRequest* _dma_request, NodeID _launch_node,
                 XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
		 uint64_t _next_max_rw_gap, size_t src_ib_offset, size_t src_ib_size,
                 bool mark_started,
		 Memory _src_mem, Memory _dst_mem,
		 TransferIterator *_src_iter, TransferIterator *_dst_iter,
		 CustomSerdezID _src_serdez_id, CustomSerdezID _dst_serdez_id,
                 uint64_t _max_req_size, long max_nr, int _priority,
                 XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence);
      ~GPUXferDes()
      {
        while (!available_reqs.empty()) {
          GPURequest* gpu_req = (GPURequest*) available_reqs.front();
          available_reqs.pop();
          delete gpu_req;
        }
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

    private:
      //GPURequest* gpu_reqs;
      //char *src_buf_base;
      //char *dst_buf_base;
      Cuda::GPU *dst_gpu, *src_gpu;
    };
#endif

#ifdef USE_HDF
    class HDFXferDes : public XferDes {
    public:
      HDFXferDes(DmaRequest* _dma_request, NodeID _launch_node,
                 XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
		 uint64_t _next_max_rw_gap, size_t src_ib_offset, size_t src_ib_size,
                 bool mark_started,
                 RegionInstance inst,
		 Memory _src_mem, Memory _dst_mem,
		 TransferIterator *_src_iter, TransferIterator *_dst_iter,
		 CustomSerdezID _src_serdez_id, CustomSerdezID _dst_serdez_id,
                 uint64_t _max_req_size, long max_nr, int _priority,
                 XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence);
      ~HDFXferDes()
      {
        free(hdf_reqs);
        //delete lsi;
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

      struct HDFFileInfo {
	hid_t file_id;
	std::map<std::string, hid_t> dset_ids;
      };

    private:
      HDFRequest* hdf_reqs;
      std::map<std::string, HDFFileInfo *> file_infos;
      //char *buf_base;
      //const HDF5Memory::HDFMetadata *hdf_metadata;
      //std::vector<OffsetsAndSize>::iterator fit;
      //GenericPointInRectIterator<DIM>* pir;
      //GenericLinearSubrectIterator<Mapping<DIM, 1> >* lsi;
      //Layouts::HDFLayoutIterator<DIM>* hli;
    };
#endif

    class Channel {
    public:
      Channel(XferDes::XferKind _kind)
	: node(my_node_id), kind(_kind) {}
      virtual ~Channel() {};
    public:
      // which node manages this channel
      NodeID node;
      // the kind of XferDes this channel can accept
      XferDes::XferKind kind;
      /*
       * Submit nr asynchronous requests into the channel instance.
       * This is supposed to be a non-blocking function call, and
       * should immediately return the number of requests that are
       * successfully submitted.
       */
      virtual long submit(Request** requests, long nr) = 0;

      /*
       *
       */
      virtual void pull() = 0;

      /*
       * Return the number of slots that are available for
       * submitting requests
       */
      virtual long available() = 0;

      struct SupportedPath {
	enum SrcDstType {
	  SPECIFIC_MEMORY,
	  LOCAL_KIND,
	  GLOBAL_KIND
	};
	SrcDstType src_type, dst_type;
	union {
	  Memory src_mem;
	  Memory::Kind src_kind;
	};
	union {
	  Memory dst_mem;
	  Memory::Kind dst_kind;
	};
	unsigned bandwidth; // units = MB/s = B/us
	unsigned latency;   // units = ns
	bool redops_allowed; // TODO: list of redops?
	bool serdez_allowed; // TODO: list of serdez ops?
      };

      const std::vector<SupportedPath>& get_paths(void) const;

      virtual bool supports_path(Memory src_mem, Memory dst_mem,
				 CustomSerdezID src_serdez_id,
				 CustomSerdezID dst_serdez_id,
				 ReductionOpID redop_id,
				 unsigned *bw_ret = 0,
				 unsigned *lat_ret = 0);

      template <typename S>
      bool serialize_remote_info(S& serializer) const;

      void print(std::ostream& os) const;

    protected:
      void add_path(Memory src_mem, Memory dst_mem,
		    unsigned bandwidth, unsigned latency,
		    bool redops_allowed, bool serdez_allowed);
      void add_path(Memory src_mem, Memory::Kind dst_kind, bool dst_global,
		    unsigned bandwidth, unsigned latency,
		    bool redops_allowed, bool serdez_allowed);
      void add_path(Memory::Kind src_kind, bool src_global,
		    Memory::Kind dst_kind, bool dst_global,
		    unsigned bandwidth, unsigned latency,
		    bool redops_allowed, bool serdez_allowed);

      std::vector<SupportedPath> paths;
      // std::deque<Copy_1D> copies_1D;
      // std::deque<Copy_2D> copies_2D;
    };

 
    std::ostream& operator<<(std::ostream& os, const Channel::SupportedPath& p);

    inline std::ostream& operator<<(std::ostream& os, const Channel& c)
    {
      c.print(os);
      return os;
    }

    template <typename S>
    inline bool Channel::serialize_remote_info(S& serializer) const
    {
      return ((serializer << node) &&
	      (serializer << kind) &&
	      (serializer << paths));
    }

    class RemoteChannel : public Channel {
    protected:
      RemoteChannel(void);

    public:
      template <typename S>
      static RemoteChannel *deserialize_new(S& serializer);

      /*
       * Submit nr asynchronous requests into the channel instance.
       * This is supposed to be a non-blocking function call, and
       * should immediately return the number of requests that are
       * successfully submitted.
       */
      virtual long submit(Request** requests, long nr);

      /*
       *
       */
      virtual void pull();

      /*
       * Return the number of slots that are available for
       * submitting requests
       */
      virtual long available();
    };

    template <typename S>
    /*static*/ RemoteChannel *RemoteChannel::deserialize_new(S& serializer)
    {
      RemoteChannel *rc = new RemoteChannel;
      bool ok = ((serializer >> rc->node) &&
		 (serializer >> rc->kind) &&
		 (serializer >> rc->paths));
      if(ok) {
	return rc;
      } else {
	delete rc;
	return 0;
      }
    }

    class MemcpyChannel;

    class MemcpyThread {
    public:
      MemcpyThread(MemcpyChannel* _channel) : channel(_channel) {}
      void thread_loop();
      static void* start(void* arg);
      void stop();
    private:
      MemcpyChannel* channel;
      std::deque<MemcpyRequest*> thread_queue;
    };

    class MemcpyChannel : public Channel {
    public:
      MemcpyChannel(long max_nr);
      ~MemcpyChannel();
      void stop();
      void get_request(std::deque<MemcpyRequest*>& thread_queue);
      void return_request(std::deque<MemcpyRequest*>& thread_queue);
      long submit(Request** requests, long nr);
      void pull();
      long available();

      virtual bool supports_path(Memory src_mem, Memory dst_mem,
				 CustomSerdezID src_serdez_id,
				 CustomSerdezID dst_serdez_id,
				 ReductionOpID redop_id,
				 unsigned *bw_ret = 0,
				 unsigned *lat_ret = 0);

      bool is_stopped;
    private:
      std::deque<MemcpyRequest*> pending_queue, finished_queue;
      pthread_mutex_t pending_lock, finished_lock;
      pthread_cond_t pending_cond;
      long capacity;
      bool sleep_threads;
      //std::vector<MemcpyRequest*> available_cb;
      //MemcpyRequest** cbs;
    };

    class GASNetChannel : public Channel {
    public:
      GASNetChannel(long max_nr, XferDes::XferKind _kind);
      ~GASNetChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    private:
      long capacity;
    };

    class RemoteWriteChannel : public Channel {
    public:
      RemoteWriteChannel(long max_nr);
      ~RemoteWriteChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
      void notify_completion() {
        __sync_fetch_and_add(&capacity, 1);
      }
    private:
      // RemoteWriteChannel is maintained by dma threads
      // and active message threads, so we need atomic ops
      // for preventing data race
      long capacity;
    };
   
#ifdef USE_CUDA
    class GPUChannel : public Channel {
    public:
      GPUChannel(Cuda::GPU* _src_gpu, long max_nr, XferDes::XferKind _kind);
      ~GPUChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    private:
      Cuda::GPU* src_gpu;
      long capacity;
      std::deque<Request*> pending_copies;
    };
#endif

#ifdef USE_HDF
    class HDFChannel : public Channel {
    public:
      HDFChannel(long max_nr, XferDes::XferKind _kind);
      ~HDFChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    private:
      long capacity;
    };
#endif

    class FileChannel;
    class DiskChannel;

    class ChannelManager {
    public:
      ChannelManager(void) {
        memcpy_channel = NULL;
        gasnet_read_channel = gasnet_write_channel = NULL;
        remote_write_channel = NULL;
        disk_read_channel = NULL;
        disk_write_channel = NULL;
        file_read_channel = NULL;
        file_write_channel = NULL;
#ifdef USE_HDF
        hdf_read_channel = NULL;
        hdf_write_channel = NULL;
#endif
      }
      ~ChannelManager(void);
      MemcpyChannel* create_memcpy_channel(long max_nr);
      GASNetChannel* create_gasnet_read_channel(long max_nr);
      GASNetChannel* create_gasnet_write_channel(long max_nr);
      RemoteWriteChannel* create_remote_write_channel(long max_nr);
      DiskChannel* create_disk_read_channel(long max_nr);
      DiskChannel* create_disk_write_channel(long max_nr);
      FileChannel* create_file_read_channel(long max_nr);
      FileChannel* create_file_write_channel(long max_nr);
#ifdef USE_CUDA
      GPUChannel* create_gpu_to_fb_channel(long max_nr, Cuda::GPU* src_gpu);
      GPUChannel* create_gpu_from_fb_channel(long max_nr, Cuda::GPU* src_gpu);
      GPUChannel* create_gpu_in_fb_channel(long max_nr, Cuda::GPU* src_gpu);
      GPUChannel* create_gpu_peer_fb_channel(long max_nr, Cuda::GPU* src_gpu);
#endif
#ifdef USE_HDF
      HDFChannel* create_hdf_read_channel(long max_nr);
      HDFChannel* create_hdf_write_channel(long max_nr);
#endif

      MemcpyChannel* get_memcpy_channel() {
        return memcpy_channel;
      }
      GASNetChannel* get_gasnet_read_channel() {
        return gasnet_read_channel;
      }
      GASNetChannel* get_gasnet_write_channel() {
        return gasnet_write_channel;
      }
      RemoteWriteChannel* get_remote_write_channel() {
        return remote_write_channel;
      }
      DiskChannel* get_disk_read_channel() {
        return disk_read_channel;
      }
      DiskChannel* get_disk_write_channel() {
        return disk_write_channel;
      }
      FileChannel* get_file_read_channel() {
        return file_read_channel;
      }
      FileChannel* get_file_write_channel() {
        return file_write_channel;
      }
#ifdef USE_CUDA
      GPUChannel* get_gpu_to_fb_channel(Cuda::GPU* gpu) {
        std::map<Cuda::GPU*, GPUChannel*>::iterator it;
        it = gpu_to_fb_channels.find(gpu);
        assert(it != gpu_to_fb_channels.end());
        return (it->second);
      }
      GPUChannel* get_gpu_from_fb_channel(Cuda::GPU* gpu) {
        std::map<Cuda::GPU*, GPUChannel*>::iterator it;
        it = gpu_from_fb_channels.find(gpu);
        assert(it != gpu_from_fb_channels.end());
        return (it->second);
      }
      GPUChannel* get_gpu_in_fb_channel(Cuda::GPU* gpu) {
        std::map<Cuda::GPU*, GPUChannel*>::iterator it;
        it = gpu_in_fb_channels.find(gpu);
        assert(it != gpu_in_fb_channels.end());
        return (it->second);
      }
      GPUChannel* get_gpu_peer_fb_channel(Cuda::GPU* gpu) {
        std::map<Cuda::GPU*, GPUChannel*>::iterator it;
        it = gpu_peer_fb_channels.find(gpu);
        assert(it != gpu_peer_fb_channels.end());
        return (it->second);
      }
#endif
#ifdef USE_HDF
      HDFChannel* get_hdf_read_channel() {
        return hdf_read_channel;
      }
      HDFChannel* get_hdf_write_channel() {
        return hdf_write_channel;
      }
#endif
    public:
      MemcpyChannel* memcpy_channel;
      GASNetChannel *gasnet_read_channel, *gasnet_write_channel;
      RemoteWriteChannel* remote_write_channel;
      DiskChannel *disk_read_channel, *disk_write_channel;
      FileChannel *file_read_channel, *file_write_channel;
#ifdef USE_CUDA
      std::map<Cuda::GPU*, GPUChannel*> gpu_to_fb_channels, gpu_in_fb_channels, gpu_from_fb_channels, gpu_peer_fb_channels;
#endif
#ifdef USE_HDF
      HDFChannel *hdf_read_channel, *hdf_write_channel;
#endif
    };

    class CompareXferDes {
    public:
      bool operator() (XferDes* a, XferDes* b) {
        if(a->priority == b->priority)
          return (a < b);
        else 
          return (a->priority < b->priority);
      }
    };
    //typedef std::priority_queue<XferDes*, std::vector<XferDes*>, CompareXferDes> PriorityXferDesQueue;
    typedef std::set<XferDes*, CompareXferDes> PriorityXferDesQueue;

    class XferDesQueue;
    class DMAThread {
    public:
      DMAThread(long _max_nr, XferDesQueue* _xd_queue, std::vector<Channel*>& _channels) {
        for (std::vector<Channel*>::iterator it = _channels.begin(); it != _channels.end(); it ++) {
          channel_to_xd_pool[*it] = new PriorityXferDesQueue;
        }
        xd_queue = _xd_queue;
        max_nr = _max_nr;
        is_stopped = false;
        requests = (Request**) calloc(max_nr, sizeof(Request*));
        sleep = false;
        pthread_mutex_init(&enqueue_lock, NULL);
        pthread_cond_init(&enqueue_cond, NULL);
      }
      DMAThread(long _max_nr, XferDesQueue* _xd_queue, Channel* _channel) {
        channel_to_xd_pool[_channel] = new PriorityXferDesQueue;
        xd_queue = _xd_queue;
        max_nr = _max_nr;
        is_stopped = false;
        requests = (Request**) calloc(max_nr, sizeof(Request*));
        sleep = false;
        pthread_mutex_init(&enqueue_lock, NULL);
        pthread_cond_init(&enqueue_cond, NULL);
      }
      ~DMAThread() {
        std::map<Channel*, PriorityXferDesQueue*>::iterator it;
        for (it = channel_to_xd_pool.begin(); it != channel_to_xd_pool.end(); it++) {
          delete it->second;
        }
        free(requests);
        pthread_mutex_destroy(&enqueue_lock);
        pthread_cond_destroy(&enqueue_cond);
      }
      void dma_thread_loop();
      // Thread start function that takes an input of DMAThread
      // instance, and start to execute the requests from XferDes
      // by using its channels.
      static void* start(void* arg) {
        DMAThread* dma_thread = (DMAThread*) arg;
        dma_thread->dma_thread_loop();
        return NULL;
      }

      void stop() {
        pthread_mutex_lock(&enqueue_lock);
        is_stopped = true;
        pthread_cond_signal(&enqueue_cond);
        pthread_mutex_unlock(&enqueue_lock);
      }
    public:
      pthread_mutex_t enqueue_lock;
      pthread_cond_t enqueue_cond;
      std::map<Channel*, PriorityXferDesQueue*> channel_to_xd_pool;
      bool sleep;
      bool is_stopped;
    private:
      // maximum allowed num of requests for a single
      long max_nr;
      Request** requests;
      XferDesQueue* xd_queue;
    };

    struct NotifyXferDesCompleteMessage {
      struct RequestArgs {
        XferDesFence* fence;
      };

      static void handle_request(RequestArgs args)
      {
        args.fence->mark_finished(true/*successful*/);
      }

      typedef ActiveMessageShortNoReply<XFERDES_NOTIFY_COMPLETION_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(NodeID target, XferDesFence* fence)
      {
        RequestArgs args;
        args.fence = fence;
        Message::request(target, args);
      }
    };

    struct XferDesRemoteWriteMessage {
      struct RequestArgs : public BaseMedium {
        //void *dst_buf;
        RemoteWriteRequest *req;
        NodeID sender;
	XferDesID next_xd_guid;
	size_t span_start, span_size, pre_bytes_total;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<XFERDES_REMOTEWRITE_MSGID,
                                         RequestArgs,
                                         handle_request> Message;

      static void send_request(NodeID target, void *dst_buf,
                               const void *src_buf, size_t nbytes,
                               RemoteWriteRequest* req,
			       XferDesID next_xd_guid,
			       size_t span_start,
			       size_t span_size,
			       size_t pre_bytes_total) 
      {
        RequestArgs args;
        //args.dst_buf = dst_buf;
        args.req = req;
	args.next_xd_guid = next_xd_guid;
	args.span_start = span_start;
	args.span_size = span_size;
	args.pre_bytes_total = pre_bytes_total;
        args.sender = my_node_id;
        //TODO: need to ask Sean what payload mode we should use
        Message::request(target, args, src_buf, nbytes, PAYLOAD_KEEP, dst_buf);
      }

      static void send_request(NodeID target,  void *dst_buf,
                               const void *src_buf, size_t nbytes, off_t src_str,
                               size_t nlines, RemoteWriteRequest* req,
			       XferDesID next_xd_guid,
			       size_t span_start,
			       size_t span_size,
			       size_t pre_bytes_total) 
      {
        RequestArgs args;
	//args.dst_buf = dst_buf;
	args.req = req;
	args.next_xd_guid = next_xd_guid;
	args.span_start = span_start;
	args.span_size = span_size;
	args.pre_bytes_total = pre_bytes_total;
	args.sender = my_node_id;
        //TODO: need to ask Sean what payload mode we should use
        Message::request(target, args, src_buf, nbytes, src_str, nlines,
                         PAYLOAD_KEEP, dst_buf);
      }
    };

    struct XferDesRemoteWriteAckMessage {
      struct RequestArgs {
        RemoteWriteRequest* req;
      };

      static void handle_request(RequestArgs args);
      typedef ActiveMessageShortNoReply<XFERDES_REMOTEWRITE_ACK_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(NodeID target, RemoteWriteRequest* req)
      {
        RequestArgs args;
        args.req = req;
        Message::request(target, args);
      }
    };

    struct XferDesCreateMessage {
      struct RequestArgs : public BaseMedium {
        RegionInstance inst;
        Memory src_mem, dst_mem;
        XferDesFence* fence;
      };

      // TODO: replace with new serialization stuff
      struct Payload {
        DmaRequest* dma_request;
        NodeID launch_node;
        XferDesID guid, pre_xd_guid, next_xd_guid;
        bool mark_started;
        uint64_t max_req_size;
        long max_nr;
        int priority;
        XferOrder::Type order;
        XferDes::XferKind kind;
        //Domain domain;
        int src_buf_bits[Buffer::MAX_SERIALIZATION_LEN], dst_buf_bits[Buffer::MAX_SERIALIZATION_LEN];
        size_t oas_vec_size; // as long as it needs to be
        OffsetsAndSize oas_vec_start;
        const OffsetsAndSize &oas_vec(int idx) const { return *((&oas_vec_start)+idx); }
        OffsetsAndSize &oas_vec(int idx) { return *((&oas_vec_start)+idx); }
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<XFERDES_CREATE_MSGID,
                                         RequestArgs,
                                         handle_request> Message;

      static void send_request(NodeID target, DmaRequest* dma_request, NodeID launch_node,
                               XferDesID guid, XferDesID pre_xd_guid, XferDesID next_xd_guid,
			       uint64_t next_max_rw_gap, size_t src_ib_offset, size_t src_ib_size,
                               bool mark_started,
			       Memory _src_mem, Memory _dst_mem,
			       TransferIterator *_src_iter, TransferIterator *_dst_iter,
			       CustomSerdezID _src_serdez_id, CustomSerdezID _dst_serdez_id,
                               uint64_t max_req_size, long max_nr, int priority,
                               XferOrder::Type order, XferDes::XferKind kind,
                               XferDesFence* fence, RegionInstance inst = RegionInstance::NO_INST);
    };

    struct XferDesDestroyMessage {
      struct RequestArgs {
        XferDesID guid;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<XFERDES_DESTROY_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(NodeID target, XferDesID guid)
      {
        RequestArgs args;
        args.guid = guid;
        Message::request(target, args);
      }
    };

    struct UpdateBytesWriteMessage {
      struct RequestArgs {
        XferDesID guid;
	size_t span_start, span_size, pre_bytes_total;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<XFERDES_UPDATE_BYTES_WRITE_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(NodeID target, XferDesID guid,
			       size_t span_start, size_t span_size,
			       size_t pre_bytes_total)
      {
        RequestArgs args;
        args.guid = guid;
	args.span_start = span_start;
	args.span_size = span_size;
	args.pre_bytes_total = pre_bytes_total;
        Message::request(target, args);
      }
    };

    struct UpdateBytesReadMessage {
      struct RequestArgs {
        XferDesID guid;
	size_t span_start, span_size;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<XFERDES_UPDATE_BYTES_READ_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(NodeID target, XferDesID guid,
			       size_t span_start, size_t span_size)
      {
        RequestArgs args;
        args.guid = guid;
	args.span_start = span_start;
	args.span_size = span_size;
        Message::request(target, args);
      }
    };

    class XferDesQueue {
    public:
      struct XferDesWithUpdates{
        XferDesWithUpdates(void): xd(NULL), pre_bytes_total((size_t)-1) {}
        XferDes* xd;
	size_t pre_bytes_total;
	SequenceAssembler seq_pre_write;
      };
      enum {
        NODE_BITS = 16,
        INDEX_BITS = 32
      };
      XferDesQueue(int num_dma_threads, bool pinned, CoreReservationSet& crs)
      //: core_rsrv("DMA request queue", crs, CoreReservationParameters())
      {
        if (pinned) {
          CoreReservationParameters params;
          params.set_num_cores(num_dma_threads);
          params.set_alu_usage(params.CORE_USAGE_EXCLUSIVE);
          params.set_fpu_usage(params.CORE_USAGE_EXCLUSIVE);
          params.set_ldst_usage(params.CORE_USAGE_SHARED);
          core_rsrv = new CoreReservation("DMA threads", crs, params);
        } else {
          core_rsrv = new CoreReservation("DMA threads", crs, CoreReservationParameters());
        }
        pthread_mutex_init(&queues_lock, NULL);
        pthread_rwlock_init(&guid_lock, NULL);
        // reserve the first several guid
        next_to_assign_idx = 10;
        num_threads = 0;
        num_memcpy_threads = 0;
        dma_threads = NULL;
      }

      ~XferDesQueue() {
        delete core_rsrv;
        // clean up the priority queues
        pthread_mutex_lock(&queues_lock);
        std::map<Channel*, PriorityXferDesQueue*>::iterator it2;
        for (it2 = queues.begin(); it2 != queues.end(); it2++) {
          delete it2->second;
        }
        pthread_mutex_unlock(&queues_lock);
        pthread_mutex_destroy(&queues_lock);
        pthread_rwlock_destroy(&guid_lock);
      }

      XferDesID get_guid(NodeID execution_node)
      {
        // GUID rules:
        // First NODE_BITS indicates which node will execute this xd
        // Next NODE_BITS indicates on which node this xd is generated
        // Last INDEX_BITS means a unique idx, which is used to resolve conflicts
        XferDesID idx = __sync_fetch_and_add(&next_to_assign_idx, 1);
        return (((XferDesID)execution_node << (NODE_BITS + INDEX_BITS)) | ((XferDesID)my_node_id << INDEX_BITS) | idx);
      }

      void update_pre_bytes_write(XferDesID xd_guid,
				  size_t span_start, size_t span_size,
				  size_t pre_bytes_total)
      {
        NodeID execution_node = xd_guid >> (NODE_BITS + INDEX_BITS);
        if (execution_node == my_node_id) {
          pthread_rwlock_wrlock(&guid_lock);
          std::map<XferDesID, XferDesWithUpdates>::iterator it = guid_to_xd.find(xd_guid);
          if (it != guid_to_xd.end()) {
            if (it->second.xd != NULL) {
              //it->second.xd->update_pre_bytes_write(bytes_write);
	      it->second.xd->update_pre_bytes_write(span_start, span_size, pre_bytes_total);
            } else {
              // if (bytes_write > it->second.pre_bytes_write)
              //   it->second.pre_bytes_write = bytes_write;
	      it->second.seq_pre_write.add_span(span_start, span_size);
	      if(pre_bytes_total != ((size_t)-1)) {
		assert((it->second.pre_bytes_total == pre_bytes_total) ||
		       (it->second.pre_bytes_total == (size_t)-1));
		it->second.pre_bytes_total = pre_bytes_total;
	      }
            }
          } else {
            XferDesWithUpdates& xdup = guid_to_xd[xd_guid];
	    xdup.seq_pre_write.add_span(span_start, span_size);
	    xdup.pre_bytes_total = pre_bytes_total;
          }
          pthread_rwlock_unlock(&guid_lock);
        }
        else {
          // send a active message to remote node
          UpdateBytesWriteMessage::send_request(execution_node, xd_guid,
						span_start, span_size,
						pre_bytes_total);
        }
      }

      void update_next_bytes_read(XferDesID xd_guid,
				  size_t span_start, size_t span_size)
      {
        NodeID execution_node = xd_guid >> (NODE_BITS + INDEX_BITS);
        if (execution_node == my_node_id) {
          pthread_rwlock_rdlock(&guid_lock);
          std::map<XferDesID, XferDesWithUpdates>::iterator it = guid_to_xd.find(xd_guid);
          if (it != guid_to_xd.end()) {
	    assert(it->second.xd != NULL);
	    it->second.xd->update_next_bytes_read(span_start, span_size);
	  } else {
            // This means this update goes slower than future updates, which marks
            // completion of xfer des (ID = xd_guid). In this case, it is safe to drop the update
	  }
          pthread_rwlock_unlock(&guid_lock);
        }
        else {
          // send a active message to remote node
          UpdateBytesReadMessage::send_request(execution_node, xd_guid,
					       span_start, span_size);
        }
      }

      void register_dma_thread(DMAThread* dma_thread)
      {
        std::map<Channel*, PriorityXferDesQueue*>::iterator it;
        pthread_mutex_lock(&queues_lock);
        for(it = dma_thread->channel_to_xd_pool.begin(); it != dma_thread->channel_to_xd_pool.end(); it++) {
          channel_to_dma_thread[it->first] = dma_thread;
          queues[it->first] = new PriorityXferDesQueue;
        }
        pthread_mutex_unlock(&queues_lock);
      }

      void destroy_xferDes(XferDesID guid) {
        pthread_rwlock_wrlock(&guid_lock);
        std::map<XferDesID, XferDesWithUpdates>::iterator it = guid_to_xd.find(guid);
        assert(it != guid_to_xd.end());
        assert(it->second.xd != NULL);
        XferDes* xd = it->second.xd;
        guid_to_xd.erase(it);
        pthread_rwlock_unlock(&guid_lock);
        delete xd;
      }

      void enqueue_xferDes_local(XferDes* xd) {
        pthread_rwlock_wrlock(&guid_lock);
        std::map<XferDesID, XferDesWithUpdates>::iterator git = guid_to_xd.find(xd->guid);
        if (git != guid_to_xd.end()) {
          // xerDes_queue has received updates of this xferdes
          // need to integrate these updates into xferdes
          assert(git->second.xd == NULL);
	  git->second.xd = xd;
	  xd->pre_bytes_total = git->second.pre_bytes_total;
	  xd->seq_pre_write.swap(git->second.seq_pre_write);
        } else {
	  XferDesWithUpdates& xdup = guid_to_xd[xd->guid];
	  xdup.xd = xd;
        }
        pthread_rwlock_unlock(&guid_lock);
        std::map<Channel*, DMAThread*>::iterator it;
        it = channel_to_dma_thread.find(xd->channel);
        assert(it != channel_to_dma_thread.end());
        DMAThread* dma_thread = it->second;
        pthread_mutex_lock(&dma_thread->enqueue_lock);
        pthread_mutex_lock(&queues_lock);
        std::map<Channel*, PriorityXferDesQueue*>::iterator it2;
        it2 = queues.find(xd->channel);
        assert(it2 != queues.end());
        // push ourself into the priority queue
        it2->second->insert(xd);
        pthread_mutex_unlock(&queues_lock);
        if (dma_thread->sleep) {
          dma_thread->sleep = false;
          pthread_cond_signal(&dma_thread->enqueue_cond);
        }
        pthread_mutex_unlock(&dma_thread->enqueue_lock);
      }

      bool dequeue_xferDes(DMAThread* dma_thread, bool wait_on_empty) {
        pthread_mutex_lock(&dma_thread->enqueue_lock);
        std::map<Channel*, PriorityXferDesQueue*>::iterator it;
        if (wait_on_empty) {
          bool empty = true;
          for(it = dma_thread->channel_to_xd_pool.begin(); it != dma_thread->channel_to_xd_pool.end(); it++) {
            pthread_mutex_lock(&queues_lock);
            std::map<Channel*, PriorityXferDesQueue*>::iterator it2;
            it2 = queues.find(it->first);
            assert(it2 != queues.end());
            if (it2->second->size() > 0)
              empty = false;
            pthread_mutex_unlock(&queues_lock);
          }

          if (empty && !dma_thread->is_stopped) {
            dma_thread->sleep = true;
            pthread_cond_wait(&dma_thread->enqueue_cond, &dma_thread->enqueue_lock);
          }
        }

        for(it = dma_thread->channel_to_xd_pool.begin(); it != dma_thread->channel_to_xd_pool.end(); it++) {
          pthread_mutex_lock(&queues_lock);
          std::map<Channel*, PriorityXferDesQueue*>::iterator it2;
          it2 = queues.find(it->first);
          assert(it2 != queues.end());
          it->second->insert(it2->second->begin(), it2->second->end());
          it2->second->clear();
          pthread_mutex_unlock(&queues_lock);
        }
        pthread_mutex_unlock(&dma_thread->enqueue_lock);
        return true;
      }

      void start_worker(int count, int max_nr, ChannelManager* channel_manager);

      void stop_worker();

    protected:
      std::map<Channel*, DMAThread*> channel_to_dma_thread;
      std::map<Channel*, PriorityXferDesQueue*> queues;
      std::map<XferDesID, XferDesWithUpdates> guid_to_xd;
      pthread_mutex_t queues_lock;
      pthread_rwlock_t guid_lock;
      XferDesID next_to_assign_idx;
      CoreReservation* core_rsrv;
      int num_threads, num_memcpy_threads;
      DMAThread** dma_threads;
      MemcpyThread** memcpy_threads;
      std::vector<Thread*> worker_threads;
    };

    XferDesQueue* get_xdq_singleton();
    ChannelManager* get_channel_manager();
#ifdef USE_CUDA
    void register_gpu_in_dma_systems(Cuda::GPU* gpu);
#endif
    void start_channel_manager(int count, bool pinned, int max_nr, CoreReservationSet& crs);
    void stop_channel_manager();

    void create_xfer_des(DmaRequest* _dma_request,
			 NodeID _launch_node,
			 NodeID _target_node,
                         XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
			 uint64_t _next_max_rw_gap, size_t src_ib_offset, size_t src_ib_size,
                         bool mark_started,
			 Memory _src_mem, Memory _dst_mem,
			 TransferIterator *_src_iter, TransferIterator *_dst_iter,
			 CustomSerdezID _src_serdez_id, CustomSerdezID _dst_serdez_id,
                         uint64_t _max_req_size, long max_nr, int _priority,
                         XferOrder::Type _order, XferDes::XferKind _kind,
                         XferDesFence* _complete_fence, RegionInstance inst = RegionInstance::NO_INST);

    void destroy_xfer_des(XferDesID _guid);
}; // namespace Realm

#endif

