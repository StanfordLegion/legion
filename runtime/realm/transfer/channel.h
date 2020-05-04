/* Copyright 2020 Stanford University
 * Copyright 2020 Los Alamos National Laboratory
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

#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_module.h"
#endif

#ifdef REALM_USE_HDF5
#include "realm/hdf5/hdf5_internal.h"
#include "realm/hdf5/hdf5_access.h"
#endif

namespace Realm {

    class XferDes;
    class XferDesQueue;
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
      int src_port_idx, dst_port_idx;
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
      atomic<size_t> contig_amount;  // everything from [0, contig_amount) is covered
      atomic<size_t> first_noncontig; // nothing in [contig_amount, first_noncontig) 
      Mutex *mutex;
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
      //NodeID dst_node;
      const void *src_base;
      void *dst_base;
      //size_t nbytes;
    };

#ifdef REALM_USE_CUDA
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
      //off_t src_gpu_off, dst_gpu_off;
      Cuda::GPU* dst_gpu;
      GPUCompletionEvent event;
    };
#endif

#ifdef REALM_USE_HDF5
    class HDFRequest : public Request {
    public:
      void *mem_base; // could be source or dest
      hid_t dataset_id, datatype_id;
      hid_t mem_space_id, file_space_id;
    };
#endif

    class XferDesFence : public Realm::Operation::AsyncWorkItem {
    public:
      XferDesFence(Realm::Operation *op) : Realm::Operation::AsyncWorkItem(op) {}
      virtual void request_cancellation(void) {
    	// ignored for now
      }
      virtual void print(std::ostream& os) const { os << "XferDesFence"; }
    };

    struct XferDesPortInfo {
      enum /*PortType*/ {
	DATA_PORT,
	GATHER_CONTROL_PORT,
	SCATTER_CONTROL_PORT,
      };
      int port_type;
      XferDesID peer_guid;
      int peer_port_idx;
      int indirect_port_idx;
      Memory mem;
      RegionInstance inst;
      size_t ib_offset, ib_size;
      TransferIterator *iter;
      CustomSerdezID serdez_id;
    };

    class XferDes {
    public:
      // a pointer to the DmaRequest that contains this XferDes
      DmaRequest* dma_request;
      // a boolean indicating if we have marked started
      bool mark_start;
      // ID of the node that launches this XferDes
      NodeID launch_node;
      //uint64_t /*bytes_submit, */bytes_read, bytes_write/*, bytes_total*/;
      bool iteration_completed;
      // current input and output port mask
      uint64_t current_in_port_mask, current_out_port_mask;
      uint64_t current_in_port_remain, current_out_port_remain;
      struct XferPort {
	MemoryImpl *mem;
	TransferIterator *iter;
	const CustomSerdezUntyped *serdez_op;
	XferDesID peer_guid;
	int peer_port_idx;
	int indirect_port_idx;
	bool is_indirect_port;
	size_t local_bytes_total;
	atomic<size_t> local_bytes_cons, remote_bytes_total;
	SequenceAssembler seq_local, seq_remote;
	// used to free up intermediate input buffers as soon as all data
	//  has been read (rather than waiting for overall transfer chain
	//  to complete)
	Memory ib_mem;
	size_t ib_offset, ib_size;
      };
      std::vector<XferPort> input_ports, output_ports;
      struct ControlPortState {
	int control_port_idx;
	int current_io_port;
	size_t remaining_count; // units of bytes for normal (elements for serialized data?)
	bool eos_received;
      };
      ControlPortState input_control, output_control;
      // maximum size for a single request
      uint64_t max_req_size;
      // priority of the containing XferDes
      int priority;
      // current, previous and next XferDes in the chain, XFERDES_NO_GUID
      // means this XferDes is the first/last one.
      XferDesID guid;//, pre_xd_guid, next_xd_guid;
      // XferDesKind of the Xfer Descriptor
      XferDesKind kind;
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
      Mutex xd_lock, update_read_lock, update_write_lock;
      // default iterators provided to generate requests
      //Layouts::GenericLayoutIterator<DIM>* li;
      // SJT:what is this for?
      //unsigned offset_idx;
    public:
      XferDes(DmaRequest *_dma_request, NodeID _launch_node, XferDesID _guid,
	      const std::vector<XferDesPortInfo>& inputs_info,
	      const std::vector<XferDesPortInfo>& outputs_info,
	      bool _mark_start,
              uint64_t _max_req_size, int _priority,
              XferDesFence* _complete_fence);

      virtual ~XferDes();

      virtual Event request_metadata();

      virtual long get_requests(Request** requests, long nr) = 0;

      virtual void notify_request_read_done(Request* req) = 0;

      virtual void notify_request_write_done(Request* req) = 0;

      virtual void flush() = 0;
 
      long default_get_requests(Request** requests, long nr, unsigned flags = 0);
      void default_notify_request_read_done(Request* req);
      void default_notify_request_write_done(Request* req);

      virtual void update_bytes_read(int port_idx, size_t offset, size_t size);
      virtual void update_bytes_write(int port_idx, size_t offset, size_t size);
      void update_pre_bytes_write(int port_idx, size_t offset, size_t size, size_t pre_bytes_total);
      void update_next_bytes_read(int port_idx, size_t offset, size_t size);

      bool is_completed(void);

      void mark_completed();

#if 0
      void update_pre_bytes_write(size_t new_val) {
	update_write_lock.lock();
        if (pre_bytes_write < new_val)
          pre_bytes_write = new_val;
        /*uint64_t old_val = pre_bytes_write;
        while (old_val < new_val) {
          pre_bytes_write.compare_exchange_strong(old_val, new_val);
        }*/
	update_write_lock.unlock();
      }

      void update_next_bytes_read(size_t new_val) {
	update_read_lock.lock();
        if (next_bytes_read < new_val)
          next_bytes_read = new_val;
        /*uint64_t old_val = next_bytes_read;
        while (old_val < new_val) {
          next_bytes_read.compare_exchange_strong(old_val, new_val);
        }*/
	update_read_lock.unlock();
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

      class DeferredXDEnqueue : public Realm::EventWaiter {
      public:
	void defer(XferDesQueue *_xferDes_queue,
		   XferDes *_xd, Event wait_on);

	virtual void event_triggered(bool poisoned);
	virtual void print(std::ostream& os) const;
	virtual Event get_finish_event(void) const;

      protected:
	XferDesQueue *xferDes_queue;
	XferDes *xd; // TODO: eliminate this based on a known offset
      };
      DeferredXDEnqueue deferred_enqueue;
    };

    class MemcpyXferDes : public XferDes {
    public:
      MemcpyXferDes(DmaRequest *_dma_request, NodeID _launch_node, XferDesID _guid,
		    const std::vector<XferDesPortInfo>& inputs_info,
		    const std::vector<XferDesPortInfo>& outputs_info,
		    bool _mark_start,
		    uint64_t _max_req_size, long max_nr, int _priority,
		    XferDesFence* _complete_fence);

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
      GASNetXferDes(DmaRequest *_dma_request, NodeID _launch_node, XferDesID _guid,
		    const std::vector<XferDesPortInfo>& inputs_info,
		    const std::vector<XferDesPortInfo>& outputs_info,
		    bool _mark_start,
		    uint64_t _max_req_size, long max_nr, int _priority,
		    XferDesFence* _complete_fence);

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
      RemoteWriteXferDes(DmaRequest *_dma_request, NodeID _launch_node, XferDesID _guid,
			 const std::vector<XferDesPortInfo>& inputs_info,
			 const std::vector<XferDesPortInfo>& outputs_info,
			 bool _mark_start,
			 uint64_t _max_req_size, long max_nr, int _priority,
			 XferDesFence* _complete_fence);

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
      virtual void update_bytes_write(int port_idx, size_t offset, size_t size);

    private:
      RemoteWriteRequest* requests;
      //char *dst_buf_base;
    };

#ifdef REALM_USE_CUDA
    class GPUXferDes : public XferDes {
    public:
      GPUXferDes(DmaRequest *_dma_request, NodeID _launch_node, XferDesID _guid,
		 const std::vector<XferDesPortInfo>& inputs_info,
		 const std::vector<XferDesPortInfo>& outputs_info,
		 bool _mark_start,
		 uint64_t _max_req_size, long max_nr, int _priority,
		 XferDesFence* _complete_fence);

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

#ifdef REALM_USE_HDF5
    class HDFXferDes : public XferDes {
    public:
      HDFXferDes(DmaRequest *_dma_request, NodeID _launch_node, XferDesID _guid,
		 const std::vector<XferDesPortInfo>& inputs_info,
		 const std::vector<XferDesPortInfo>& outputs_info,
		 bool _mark_start,
		 uint64_t _max_req_size, long max_nr, int _priority,
		 XferDesFence* _complete_fence);

      ~HDFXferDes()
      {
        free(hdf_reqs);
        //delete lsi;
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

    private:
      HDFRequest* hdf_reqs;
      std::map<FieldID, HDF5::HDF5Dataset *> datasets;
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
      Channel(XferDesKind _kind)
	: node(Network::my_node_id), kind(_kind) {}
      virtual ~Channel() {};
    public:
      // which node manages this channel
      NodeID node;
      // the kind of XferDes this channel can accept
      XferDesKind kind;

      // attempt to make progress on the specified xferdes
      virtual long progress_xd(XferDes *xd, long max_nr);
      
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

      virtual bool supports_path(Memory src_mem, Memory dst_mem,
				 CustomSerdezID src_serdez_id,
				 CustomSerdezID dst_serdez_id,
				 ReductionOpID redop_id,
				 unsigned *bw_ret = 0,
				 unsigned *lat_ret = 0);
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
      Mutex pending_lock, finished_lock;
      CondVar pending_cond;
      atomic<long> capacity;
      bool sleep_threads;
      //std::vector<MemcpyRequest*> available_cb;
      //MemcpyRequest** cbs;
    };

    class GASNetChannel : public Channel {
    public:
      GASNetChannel(long max_nr, XferDesKind _kind);
      ~GASNetChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    private:
      atomic<long> capacity;
    };

    class RemoteWriteChannel : public Channel {
    public:
      RemoteWriteChannel(long max_nr);
      ~RemoteWriteChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
      void notify_completion();

    private:
      // RemoteWriteChannel is maintained by dma threads
      // and active message threads, so we need atomic ops
      // for preventing data race
      atomic<long> capacity;
    };
   
#ifdef REALM_USE_CUDA
    class GPUChannel : public Channel {
    public:
      GPUChannel(Cuda::GPU* _src_gpu, long max_nr, XferDesKind _kind);
      ~GPUChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    private:
      Cuda::GPU* src_gpu;
      atomic<long> capacity;
      std::deque<Request*> pending_copies;
    };
#endif

#ifdef REALM_USE_HDF5
    class HDFChannel : public Channel {
    public:
      HDFChannel(long max_nr, XferDesKind _kind);
      ~HDFChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    private:
      atomic<long> capacity;
    };
#endif

    class FileChannel;
    class DiskChannel;

    class AddressSplitChannel : public Channel {
    public:
      AddressSplitChannel();
      virtual ~AddressSplitChannel();
      
      virtual long progress_xd(XferDes *xd, long max_nr);
      virtual long submit(Request** requests, long nr);
      virtual void pull();
      virtual long available();
    };
  
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
#ifdef REALM_USE_HDF5
        hdf_read_channel = NULL;
        hdf_write_channel = NULL;
#endif
	addr_split_channel = 0;
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
      AddressSplitChannel *create_addr_split_channel();
#ifdef REALM_USE_CUDA
      GPUChannel* create_gpu_to_fb_channel(long max_nr, Cuda::GPU* src_gpu);
      GPUChannel* create_gpu_from_fb_channel(long max_nr, Cuda::GPU* src_gpu);
      GPUChannel* create_gpu_in_fb_channel(long max_nr, Cuda::GPU* src_gpu);
      GPUChannel* create_gpu_peer_fb_channel(long max_nr, Cuda::GPU* src_gpu);
#endif
#ifdef REALM_USE_HDF5
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
      AddressSplitChannel *get_address_split_channel() {
	return addr_split_channel;
      }
#ifdef REALM_USE_CUDA
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
#ifdef REALM_USE_HDF5
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
#ifdef REALM_USE_CUDA
      std::map<Cuda::GPU*, GPUChannel*> gpu_to_fb_channels, gpu_in_fb_channels, gpu_from_fb_channels, gpu_peer_fb_channels;
#endif
#ifdef REALM_USE_HDF5
      HDFChannel *hdf_read_channel, *hdf_write_channel;
#endif
      AddressSplitChannel *addr_split_channel;
    };

    class CompareXferDes {
    public:
      bool operator() (XferDes* a, XferDes* b) const {
        if(a->priority == b->priority)
          return (a < b);
        else 
          return (a->priority < b->priority);
      }
    };
    //typedef std::priority_queue<XferDes*, std::vector<XferDes*>, CompareXferDes> PriorityXferDesQueue;
    typedef std::set<XferDes*, CompareXferDes> PriorityXferDesQueue;

    class DMAThread {
    public:
      DMAThread(long _max_nr, XferDesQueue* _xd_queue, std::vector<Channel*>& _channels)
	: enqueue_cond(enqueue_lock)
      {
        for (std::vector<Channel*>::iterator it = _channels.begin(); it != _channels.end(); it ++) {
          channel_to_xd_pool[*it] = new PriorityXferDesQueue;
        }
        xd_queue = _xd_queue;
        max_nr = _max_nr;
        is_stopped = false;
        requests = (Request**) calloc(max_nr, sizeof(Request*));
        sleep = false;
      }
      DMAThread(long _max_nr, XferDesQueue* _xd_queue, Channel* _channel) 
	: enqueue_cond(enqueue_lock)
      {
        channel_to_xd_pool[_channel] = new PriorityXferDesQueue;
        xd_queue = _xd_queue;
        max_nr = _max_nr;
        is_stopped = false;
        requests = (Request**) calloc(max_nr, sizeof(Request*));
        sleep = false;
      }
      ~DMAThread() {
        std::map<Channel*, PriorityXferDesQueue*>::iterator it;
        for (it = channel_to_xd_pool.begin(); it != channel_to_xd_pool.end(); it++) {
          delete it->second;
        }
        free(requests);
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
	enqueue_lock.lock();
        is_stopped = true;
	enqueue_cond.signal();
	enqueue_lock.unlock();
      }
    public:
      Mutex enqueue_lock;
      CondVar enqueue_cond;
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
      XferDesFence* fence;

      static void handle_message(NodeID sender,
				 const NotifyXferDesCompleteMessage &args,
				 const void *data,
				 size_t datalen)
      {
        args.fence->mark_finished(true/*successful*/);
      }
      static void send_request(NodeID target, XferDesFence* fence)
      {
	ActiveMessage<NotifyXferDesCompleteMessage> amsg(target);
	amsg->fence = fence;
	amsg.commit();
      }
    };

    struct XferDesRemoteWriteMessage {
      RemoteWriteRequest *req;
      XferDesID next_xd_guid;
      int next_port_idx;
      unsigned span_size; // 32 bits to fit in packet size
      size_t span_start, /*span_size,*/ pre_bytes_total;

      static void handle_message(NodeID sender,
				 const XferDesRemoteWriteMessage &args,
				 const void *data,
				 size_t datalen);

      static void send_request(NodeID target, void *dst_buf,
                               const void *src_buf, size_t nbytes,
                               RemoteWriteRequest* req,
			       XferDesID next_xd_guid,
			       int next_port_idx,
			       size_t span_start,
			       size_t span_size,
			       size_t pre_bytes_total) 
      {
	ActiveMessage<XferDesRemoteWriteMessage> amsg(target, nbytes, dst_buf);
	amsg->req = req;
	amsg->next_xd_guid = next_xd_guid;
	amsg->next_port_idx = next_port_idx;
	amsg->span_start = span_start;
	assert(span_size <= UINT_MAX);
	amsg->span_size = span_size;
	amsg->pre_bytes_total = pre_bytes_total;
        //TODO: need to ask Sean what payload mode we should use
	amsg.add_payload(src_buf, nbytes, PAYLOAD_KEEP);
	amsg.commit();
      }

      static void send_request(NodeID target,  void *dst_buf,
                               const void *src_buf, size_t nbytes, off_t src_str,
                               size_t nlines, RemoteWriteRequest* req,
			       XferDesID next_xd_guid,
			       int next_port_idx,
			       size_t span_start,
			       size_t span_size,
			       size_t pre_bytes_total) 
      {
	size_t payload_size = nbytes*nlines;
	ActiveMessage<XferDesRemoteWriteMessage> amsg(target, payload_size, dst_buf);
	amsg->req = req;
	amsg->next_xd_guid = next_xd_guid;
	amsg->next_port_idx = next_port_idx;
	amsg->span_start = span_start;
	amsg->span_size = span_size;
	amsg->pre_bytes_total = pre_bytes_total;
        //TODO: need to ask Sean what payload mode we should use
	PayloadSource *payload_src = new TwoDPayload(src_buf, nbytes, nlines, src_str, PAYLOAD_KEEP);
	payload_src->copy_data(amsg.payload_ptr(payload_size));
	amsg.commit();
      }
    };

    struct XferDesRemoteWriteAckMessage {
      RemoteWriteRequest* req;

      static void handle_message(NodeID sender,
				 const XferDesRemoteWriteAckMessage &args,
				 const void *data,
				 size_t datalen);
      static void send_request(NodeID target, RemoteWriteRequest* req)
      {
	ActiveMessage<XferDesRemoteWriteAckMessage> amsg(target);
        amsg->req = req;
	amsg.commit();
      }
    };

    struct XferDesDestroyMessage {
      XferDesID guid;
      static void handle_message(NodeID sender,
				 const XferDesDestroyMessage &args,
				 const void *data,
				 size_t datalen);
      static void send_request(NodeID target, XferDesID guid)
      {
	ActiveMessage<XferDesDestroyMessage> amsg(target);
        amsg->guid = guid;
	amsg.commit();
      }
    };

    struct UpdateBytesWriteMessage {
      XferDesID guid;
      int port_idx;
      size_t span_start, span_size, pre_bytes_total;

      static void handle_message(NodeID sender,
				 const UpdateBytesWriteMessage &args,
				 const void *data,
				 size_t datalen);

      static void send_request(NodeID target, XferDesID guid,
			       int port_idx,
			       size_t span_start, size_t span_size,
			       size_t pre_bytes_total)
      {
	ActiveMessage<UpdateBytesWriteMessage> amsg(target);
        amsg->guid = guid;
	amsg->port_idx = port_idx;
	amsg->span_start = span_start;
	amsg->span_size = span_size;
	amsg->pre_bytes_total = pre_bytes_total;
	amsg.commit();
      }
    };

    struct UpdateBytesReadMessage {
      XferDesID guid;
      int port_idx;
      size_t span_start, span_size;

      static void handle_message(NodeID sender,
				 const UpdateBytesReadMessage &args,
				 const void *data,
				 size_t datalen);

      static void send_request(NodeID target, XferDesID guid,
			       int port_idx,
			       size_t span_start, size_t span_size)
      {
	ActiveMessage<UpdateBytesReadMessage> amsg(target);
        amsg->guid = guid;
	amsg->port_idx = port_idx;
	amsg->span_start = span_start;
	amsg->span_size = span_size;
	amsg.commit();
      }
    };

    class XferDesFactory {
    protected:
      // do not destroy directly - use release()
      virtual ~XferDesFactory() {}

    public:
      virtual void release() = 0;

      virtual void create_xfer_des(DmaRequest *dma_request,
				   NodeID launch_node,
				   NodeID target_node,
				   XferDesID guid,
				   const std::vector<XferDesPortInfo>& inputs_info,
				   const std::vector<XferDesPortInfo>& outputs_info,
				   bool mark_started,
				   uint64_t max_req_size, long max_nr, int priority,
				   XferDesFence *complete_fence,
				   RegionInstance inst = RegionInstance::NO_INST) = 0;
    };

    struct XferDesCreateMessageBase {
      RegionInstance inst;
      XferDesFence *complete_fence;
      DmaRequest *dma_request;
      XferDesID guid;
      NodeID launch_node;
    };

    template <typename T>
    struct XferDesCreateMessage : public XferDesCreateMessageBase {
      static void handle_message(NodeID sender,
				 const XferDesCreateMessage<T> &args,
				 const void *data,
				 size_t datalen);
    };

    template <typename T>
    class SimpleXferDesFactory : public XferDesFactory {
    protected:
      // simple factories use singletons, so no direct construction/destruction
      SimpleXferDesFactory();
      ~SimpleXferDesFactory();

    public:
      static SimpleXferDesFactory<T> *get_singleton();
      virtual void release();

      virtual void create_xfer_des(DmaRequest *dma_request,
				   NodeID launch_node,
				   NodeID target_node,
				   XferDesID guid,
				   const std::vector<XferDesPortInfo>& inputs_info,
				   const std::vector<XferDesPortInfo>& outputs_info,
				   bool mark_started,
				   uint64_t max_req_size, long max_nr, int priority,
				   XferDesFence *complete_fence,
				   RegionInstance inst = RegionInstance::NO_INST);
    };
      
    class XferDesQueue {
    public:
      struct XferDesWithUpdates{
        XferDesWithUpdates(void): xd(NULL) {}
        XferDes* xd;
	std::map<int, size_t> pre_bytes_total;
	std::map<int, SequenceAssembler> seq_pre_write;
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
        // reserve the first several guid
        next_to_assign_idx.store(10);
        num_threads = 0;
        num_memcpy_threads = 0;
        dma_threads = NULL;
      }

      ~XferDesQueue() {
        delete core_rsrv;
        // clean up the priority queues
	queues_lock.lock();  // probably don't need lock here
        std::map<Channel*, PriorityXferDesQueue*>::iterator it2;
        for (it2 = queues.begin(); it2 != queues.end(); it2++) {
          delete it2->second;
        }
	queues_lock.unlock();
      }

      XferDesID get_guid(NodeID execution_node)
      {
        // GUID rules:
        // First NODE_BITS indicates which node will execute this xd
        // Next NODE_BITS indicates on which node this xd is generated
        // Last INDEX_BITS means a unique idx, which is used to resolve conflicts
        XferDesID idx = next_to_assign_idx.fetch_add(1);
        return (((XferDesID)execution_node << (NODE_BITS + INDEX_BITS)) | ((XferDesID)Network::my_node_id << INDEX_BITS) | idx);
      }

      void update_pre_bytes_write(XferDesID xd_guid, int port_idx,
				  size_t span_start, size_t span_size,
				  size_t pre_bytes_total)
      {
        NodeID execution_node = xd_guid >> (NODE_BITS + INDEX_BITS);
        if (execution_node == Network::my_node_id) {
	  RWLock::AutoWriterLock al(guid_lock);
          std::map<XferDesID, XferDesWithUpdates>::iterator it = guid_to_xd.find(xd_guid);
          if (it != guid_to_xd.end()) {
            if (it->second.xd != NULL) {
              //it->second.xd->update_pre_bytes_write(bytes_write);
	      it->second.xd->update_pre_bytes_write(port_idx, span_start, span_size, pre_bytes_total);
            } else {
	      it->second.seq_pre_write[port_idx].add_span(span_start, span_size);

	      std::map<int, size_t>::iterator it2 = it->second.pre_bytes_total.find(port_idx);
	      if(it2 != it->second.pre_bytes_total.end()) {
		if(pre_bytes_total != size_t(-1)) {
		  assert((it2->second == pre_bytes_total) ||
			 (it2->second == size_t(-1)));
		  it2->second = pre_bytes_total;
		}
	      } else
		it->second.pre_bytes_total[port_idx] = pre_bytes_total;

            }
          } else {
            XferDesWithUpdates& xdup = guid_to_xd[xd_guid];
	    xdup.seq_pre_write[port_idx].add_span(span_start, span_size);
	    xdup.pre_bytes_total[port_idx] = pre_bytes_total;
          }
        }
        else {
          // send a active message to remote node
          UpdateBytesWriteMessage::send_request(execution_node, xd_guid,
						port_idx,
						span_start, span_size,
						pre_bytes_total);
        }
      }

      void update_next_bytes_read(XferDesID xd_guid, int port_idx,
				  size_t span_start, size_t span_size)
      {
        NodeID execution_node = xd_guid >> (NODE_BITS + INDEX_BITS);
        if (execution_node == Network::my_node_id) {
	  RWLock::AutoReaderLock al(guid_lock);
          std::map<XferDesID, XferDesWithUpdates>::iterator it = guid_to_xd.find(xd_guid);
          if (it != guid_to_xd.end()) {
	    assert(it->second.xd != NULL);
	    it->second.xd->update_next_bytes_read(port_idx, span_start, span_size);
	  } else {
            // This means this update goes slower than future updates, which marks
            // completion of xfer des (ID = xd_guid). In this case, it is safe to drop the update
	  }
        }
        else {
          // send a active message to remote node
          UpdateBytesReadMessage::send_request(execution_node, xd_guid,
					       port_idx,
					       span_start, span_size);
        }
      }

      void register_dma_thread(DMAThread* dma_thread)
      {
        std::map<Channel*, PriorityXferDesQueue*>::iterator it;
        queues_lock.lock();
        for(it = dma_thread->channel_to_xd_pool.begin(); it != dma_thread->channel_to_xd_pool.end(); it++) {
          channel_to_dma_thread[it->first] = dma_thread;
          queues[it->first] = new PriorityXferDesQueue;
        }
	queues_lock.unlock();
      }

      void destroy_xferDes(XferDesID guid) {
	XferDes *xd;
	{
	  RWLock::AutoWriterLock al(guid_lock);
	  std::map<XferDesID, XferDesWithUpdates>::iterator it = guid_to_xd.find(guid);
	  assert(it != guid_to_xd.end());
	  assert(it->second.xd != NULL);
	  xd = it->second.xd;
	  guid_to_xd.erase(it);
	}
        delete xd;
      }

      void enqueue_xferDes_local(XferDes* xd);

      bool dequeue_xferDes(DMAThread* dma_thread, bool wait_on_empty);

      void start_worker(int count, int max_nr, ChannelManager* channel_manager);

      void stop_worker();

    protected:
      std::map<Channel*, DMAThread*> channel_to_dma_thread;
      std::map<Channel*, PriorityXferDesQueue*> queues;
      std::map<XferDesID, XferDesWithUpdates> guid_to_xd;
      Mutex queues_lock;
      RWLock guid_lock;
      atomic<XferDesID> next_to_assign_idx;
      CoreReservation* core_rsrv;
      int num_threads, num_memcpy_threads;
      DMAThread** dma_threads;
      MemcpyThread** memcpy_threads;
      std::vector<Thread*> worker_threads;
    };

    XferDesQueue* get_xdq_singleton();
    ChannelManager* get_channel_manager();
#ifdef REALM_USE_CUDA
    void register_gpu_in_dma_systems(Cuda::GPU* gpu);
#endif
    void start_channel_manager(int count, bool pinned, int max_nr, CoreReservationSet& crs);
    void stop_channel_manager();

    void destroy_xfer_des(XferDesID _guid);
}; // namespace Realm

#include "realm/transfer/channel.inl"

#endif

