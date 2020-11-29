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

#include "realm/realm_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#ifndef REALM_ON_WINDOWS
#include <unistd.h>
#include <pthread.h>
#endif
#include <fcntl.h>
#include <map>
#include <vector>
#include <deque>
#include <queue>
#include <assert.h>
#include <string.h>
#include "realm/transfer/lowlevel_dma.h"

#include "realm/id.h"
#include "realm/runtime_impl.h"
#include "realm/mem_impl.h"
#include "realm/inst_impl.h"
#include "realm/bgwork.h"

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

      // NOT thread-safe - caller must ensure neither *this nor other is being
      //  modified during this call
      void swap(SequenceAssembler& other);

      // asks if a span exists - return value is number of bytes from the
      //  start that do
      size_t span_exists(size_t start, size_t count);

      // returns the amount by which the contiguous range has been increased
      //  (i.e. from [pos, pos+retval) )
      size_t add_span(size_t pos, size_t count);

    protected:
      atomic<size_t> contig_amount_x2;  // everything from [0, contig_amount) is covered - LSB indicates potential presence of noncontig spans
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
      //void *dst_base;
      //size_t nbytes;
    };

#ifdef REALM_USE_CUDA
    class GPURequest;

    class GPUCompletionEvent : public Cuda::GPUCompletionNotification {
    public:
      void request_completed(void);

      GPURequest *req;
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
    class HDF5Request : public Request {
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

    class AddressList {
    public:
      AddressList();

      size_t *begin_nd_entry(int max_dim);
      void commit_nd_entry(int act_dim, size_t bytes);

      size_t bytes_pending() const;
      
    protected:
      friend class AddressListCursor;

      const size_t *read_entry();

      size_t total_bytes;
      unsigned write_pointer;
      unsigned read_pointer;
      static const size_t MAX_ENTRIES = 1000;
      size_t data[MAX_ENTRIES];
    };

    class AddressListCursor {
    public:
      AddressListCursor();

      void set_addrlist(AddressList *_addrlist);

      int get_dim();
      uintptr_t get_offset();
      uintptr_t get_stride(int dim);
      size_t remaining(int dim);
      void advance(int dim, size_t amount);

      void skip_bytes(size_t bytes);
      
    protected:
      AddressList *addrlist;
      bool partial;
      static const int MAX_DIM = 8;
      int partial_dim;
      size_t pos[MAX_DIM];
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
      atomic<bool> iteration_completed;
      atomic<bool> transfer_completed;
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
	atomic<bool> needs_pbt_update;
	size_t local_bytes_total;
	atomic<size_t> local_bytes_cons, remote_bytes_total;
	SequenceAssembler seq_local, seq_remote;
	// used to free up intermediate input buffers as soon as all data
	//  has been read (rather than waiting for overall transfer chain
	//  to complete)
	Memory ib_mem;
	size_t ib_offset, ib_size;
	AddressList addrlist;
	AddressListCursor addrcursor;
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
      // used to track by upstream/downstream xds so that we can safely
      //  sleep xds that are stalled
      atomic<unsigned> progress_counter;

      atomic<unsigned> reference_count;

      // intrusive list for queued XDs in a channel
      IntrusivePriorityListLink<XferDes> xd_link;
      REALM_PMTA_DEFN(XferDes,IntrusivePriorityListLink<XferDes>,xd_link);
      REALM_PMTA_DEFN(XferDes,int,priority);
      typedef IntrusivePriorityList<XferDes, int, REALM_PMTA_USE(XferDes,xd_link), REALM_PMTA_USE(XferDes,priority), DummyLock> XferDesList;
    protected:
      // this will be removed soon
      // queue that contains all available free requests
      Mutex available_req_mutex;
      std::queue<Request*> available_reqs;

    public:
      XferDes(DmaRequest *_dma_request, NodeID _launch_node, XferDesID _guid,
	      const std::vector<XferDesPortInfo>& inputs_info,
	      const std::vector<XferDesPortInfo>& outputs_info,
	      bool _mark_start,
              uint64_t _max_req_size, int _priority,
              XferDesFence* _complete_fence);

      // transfer descriptors are reference counted rather than explcitly
      //  deleted
      void add_reference(void);
      void remove_reference(void);

    protected:
      virtual ~XferDes();

    public:
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
      void update_pre_bytes_write(int port_idx, size_t offset, size_t size);
      void update_pre_bytes_total(int port_idx, size_t pre_bytes_total);
      void update_next_bytes_read(int port_idx, size_t offset, size_t size);

      bool is_completed(void);

      void mark_completed();

      unsigned current_progress(void);

      // checks to see if progress has been made since the last read of the
      //  progress counter - atomically marks the xd for wakeup if not
      bool check_for_progress(unsigned last_counter);

      // updates the progress counter, waking up the xd if needed
      void update_progress(void);

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

      virtual bool request_available()
      {
	AutoLock<> al(available_req_mutex);
	return !available_reqs.empty();
      }

      virtual Request* dequeue_request() {
        Request* req;
	{
	  AutoLock<> al(available_req_mutex);
	  req = available_reqs.front();
	  available_reqs.pop();
	}
	req->is_read_done = false;
	req->is_write_done = false;
	// by default, an "active" request holds a reference on the xd
	add_reference();
        return req;
      }

      virtual void enqueue_request(Request* req) {
	{
	  AutoLock<> al(available_req_mutex);
	  available_reqs.push(req);
	}
	remove_reference();
      }

      class DeferredXDEnqueue : public Realm::EventWaiter {
      public:
	void defer(XferDesQueue *_xferDes_queue,
		   XferDes *_xd, Event wait_on);

	virtual void event_triggered(bool poisoned, TimeLimit work_until);
	virtual void print(std::ostream& os) const;
	virtual Event get_finish_event(void) const;

      protected:
	XferDesQueue *xferDes_queue;
	XferDes *xd; // TODO: eliminate this based on a known offset
      };
      DeferredXDEnqueue deferred_enqueue;

      // helper widget to cache spans so that SequenceAssembler updates are as
      //  large as possible
      template <void (XferDes::*UPDATE)(int port_idx, size_t offset, size_t size)>
      class SequenceCache {
      public:
	SequenceCache(XferDes *_xd, size_t _flush_bytes = 0);

	void add_span(int port_idx, size_t offset, size_t size);
	void flush();

      protected:
	static const size_t MAX_ENTRIES = 4;

	XferDes *xd;
	int ports[MAX_ENTRIES];
	size_t offsets[MAX_ENTRIES];
	size_t sizes[MAX_ENTRIES];
	size_t total_bytes, flush_bytes;
      };
      typedef SequenceCache<&XferDes::update_bytes_read> ReadSequenceCache;
      typedef SequenceCache<&XferDes::update_bytes_write> WriteSequenceCache;

      size_t update_control_info(ReadSequenceCache *rseqcache);

      // a helper routine for individual XferDes implementations - tries to get
      //  addresses and check flow control for at least 'min_xfer_size' bytes
      //  worth of transfers from a single input to a single output, and returns
      //  the number of bytes that can be transferred before another call to
      //  this method
      // returns 0 if the transfer is complete OR if there are fewer than the
      //  minimum requested bytes available and there's reason to believe that
      //  trying again later will result in a larger chunk
      // as a side effect, the input/output control information is updated - the
      //  actual input/output ports involved in the next transfer are stored there
      size_t get_addresses(size_t min_xfer_size, ReadSequenceCache *rseqcache);

      // after a call to 'get_addresses', this call updates the various data
      //  structures to record that transfers for 'total_bytes' bytes were at
      //  least initiated - return value is whether iteration is complete
      bool record_address_consumption(size_t total_bytes);
    };

    class MemcpyChannel;

    class MemcpyXferDes : public XferDes {
    public:
      MemcpyXferDes(DmaRequest *_dma_request, NodeID _launch_node, XferDesID _guid,
		    const std::vector<XferDesPortInfo>& inputs_info,
		    const std::vector<XferDesPortInfo>& outputs_info,
		    bool _mark_start,
		    uint64_t _max_req_size, long max_nr, int _priority,
		    XferDesFence* _complete_fence);

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

      virtual bool request_available();
      virtual Request* dequeue_request();
      virtual void enqueue_request(Request* req);

      bool progress_xd(MemcpyChannel *channel, TimeLimit work_until);

    private:
      bool memcpy_req_in_use;
      MemcpyRequest memcpy_req;
      bool has_serdez;
      //const char *src_buf_base, *dst_buf_base;
    };

    class GASNetChannel;

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

      bool progress_xd(GASNetChannel *channel, TimeLimit work_until);

    private:
      GASNetRequest* gasnet_reqs;
    };

    class RemoteWriteChannel;

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

      bool progress_xd(RemoteWriteChannel *channel, TimeLimit work_until);

      // writes directly to a contiguous chunk of destination
      struct Write1DMessage {
	XferDesID next_xd_guid;
	int next_port_idx;
	size_t span_start, pre_bytes_total;

	static void handle_message(NodeID sender,
				   const Write1DMessage &args,
				   const void *data,
				   size_t datalen);
	static bool handle_inline(NodeID sender,
				  const Write1DMessage &args,
				  const void *data,
				  size_t datalen,
				  TimeLimit work_until);
      };

    private:
      RemoteWriteRequest* requests;
      //char *dst_buf_base;
    };

#ifdef REALM_USE_CUDA
    class GPUChannel;

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

      bool progress_xd(GPUChannel *channel, TimeLimit work_until);

    private:
      //GPURequest* gpu_reqs;
      //char *src_buf_base;
      //char *dst_buf_base;
      Cuda::GPU *dst_gpu, *src_gpu;
    };
#endif

#ifdef REALM_USE_HDF5
    class HDF5Channel;

    class HDF5XferDes : public XferDes {
    public:
      HDF5XferDes(DmaRequest *_dma_request, NodeID _launch_node, XferDesID _guid,
		  const std::vector<XferDesPortInfo>& inputs_info,
		  const std::vector<XferDesPortInfo>& outputs_info,
		  bool _mark_start,
		  uint64_t _max_req_size, long max_nr, int _priority,
		  XferDesFence* _complete_fence);

      ~HDF5XferDes()
      {
        //free(hdf_reqs);
        //delete lsi;
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

      virtual bool request_available();
      virtual Request* dequeue_request();
      virtual void enqueue_request(Request* req);

      bool progress_xd(HDF5Channel *channel, TimeLimit work_until);

    private:
      bool req_in_use;
      HDF5Request hdf5_req;
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

      // TODO: make pure virtual
      virtual void shutdown() {}

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
	  GLOBAL_KIND,
	  LOCAL_RDMA,
	  REMOTE_RDMA,
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
	XferDesKind xd_kind;
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
				 XferDesKind *kind_ret = 0,
				 unsigned *bw_ret = 0,
				 unsigned *lat_ret = 0);

      template <typename S>
      bool serialize_remote_info(S& serializer) const;

      void print(std::ostream& os) const;

      virtual void enqueue_ready_xd(XferDes *xd) = 0;
      virtual void wakeup_xd(XferDes *xd) = 0;

    protected:
      void add_path(Memory src_mem, Memory dst_mem,
		    unsigned bandwidth, unsigned latency,
		    bool redops_allowed, bool serdez_allowed,
		    XferDesKind xd_kind);
      void add_path(Memory src_mem, Memory::Kind dst_kind, bool dst_global,
		    unsigned bandwidth, unsigned latency,
		    bool redops_allowed, bool serdez_allowed,
		    XferDesKind xd_kind);
      void add_path(Memory::Kind src_kind, bool src_global,
		    Memory::Kind dst_kind, bool dst_global,
		    unsigned bandwidth, unsigned latency,
		    bool redops_allowed, bool serdez_allowed,
		    XferDesKind xd_kind);
      // TODO: allow rdma path to limit by kind?
      void add_path(bool local_loopback,
		    unsigned bandwidth, unsigned latency,
		    bool redops_allowed, bool serdez_allowed,
		    XferDesKind xd_kind);

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

      virtual void shutdown();

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
				 XferDesKind *kind_ret = 0,
				 unsigned *bw_ret = 0,
				 unsigned *lat_ret = 0);

      virtual void enqueue_ready_xd(XferDes *xd) { assert(0); }
      virtual void wakeup_xd(XferDes *xd) { assert(0); }
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

    template <typename CHANNEL, typename XD>
    class XDQueue : public BackgroundWorkItem {
    public:
      XDQueue(CHANNEL *_channel, const std::string& _name,
	      bool _ordered);

      void enqueue_xd(XD *xd, bool at_front = false);

      virtual void do_work(TimeLimit work_until);

    protected:
      CHANNEL *channel;
      bool ordered_mode, in_ordered_worker;
      Mutex mutex;
      XferDes::XferDesList ready_xds;
    };

    template <typename CHANNEL, typename XD>
    class SingleXDQChannel : public Channel {
    public:
      SingleXDQChannel(BackgroundWorkManager *bgwork,
		       XferDesKind _kind,
		       const std::string& _name);

      virtual void shutdown();

      virtual void enqueue_ready_xd(XferDes *xd);
      virtual void wakeup_xd(XferDes *xd);

      // TODO: remove!
      void pull() { assert(0); }
      long available() { assert(0); return 0; }
      virtual long progress_xd(XferDes *xd, long max_nr) { assert(0); return 0; }
    protected:
      XDQueue<CHANNEL, XD> xdq;
    };

    class MemcpyChannel : public SingleXDQChannel<MemcpyChannel, MemcpyXferDes> {
    public:
      MemcpyChannel(BackgroundWorkManager *bgwork);

      // multiple concurrent memcpys ok
      static const bool is_ordered = false;

      ~MemcpyChannel();

      virtual bool supports_path(Memory src_mem, Memory dst_mem,
				 CustomSerdezID src_serdez_id,
				 CustomSerdezID dst_serdez_id,
				 ReductionOpID redop_id,
				 XferDesKind *kind_ret = 0,
				 unsigned *bw_ret = 0,
				 unsigned *lat_ret = 0);

      virtual long submit(Request** requests, long nr);

      bool is_stopped;
    };

    class GASNetChannel : public SingleXDQChannel<GASNetChannel, GASNetXferDes> {
    public:
      GASNetChannel(BackgroundWorkManager *bgwork, XferDesKind _kind);
      ~GASNetChannel();

      // no more than one GASNet xfer of each type at a time
      static const bool is_ordered = true;
      
      long submit(Request** requests, long nr);
    };

    class RemoteWriteChannel : public SingleXDQChannel<RemoteWriteChannel, RemoteWriteXferDes> {
    public:
      RemoteWriteChannel(BackgroundWorkManager *bgwork);
      ~RemoteWriteChannel();

      // multiple concurrent RDMAs ok
      static const bool is_ordered = false;

      long submit(Request** requests, long nr);
    };
   
#ifdef REALM_USE_CUDA
    class GPUChannel : public SingleXDQChannel<GPUChannel, GPUXferDes> {
    public:
      GPUChannel(Cuda::GPU* _src_gpu, XferDesKind _kind,
		 BackgroundWorkManager *bgwork);
      ~GPUChannel();

      // multiple concurrent cuda copies ok
      static const bool is_ordered = false;

      long submit(Request** requests, long nr);

    private:
      Cuda::GPU* src_gpu;
      //std::deque<Request*> pending_copies;
    };
#endif

#ifdef REALM_USE_HDF5
    // single channel handles both HDF5 reads and writes
    class HDF5Channel : public SingleXDQChannel<HDF5Channel, HDF5XferDes> {
    public:
      HDF5Channel(BackgroundWorkManager *bgwork);
      ~HDF5Channel();

      // handle HDF5 requests in order - no concurrency
      static const bool is_ordered = true;

      long submit(Request** requests, long nr);
    };
#endif

    class FileChannel;
    class DiskChannel;

    class AddressSplitChannel;

    class AddressSplitXferDesBase : public XferDes {
    protected:
      AddressSplitXferDesBase(DmaRequest *_dma_request, NodeID _launch_node, XferDesID _guid,
			      const std::vector<XferDesPortInfo>& inputs_info,
			      const std::vector<XferDesPortInfo>& outputs_info,
			      bool _mark_start,
			      uint64_t _max_req_size, long max_nr, int _priority,
			      XferDesFence* _complete_fence);

    public:
      virtual bool progress_xd(AddressSplitChannel *channel, TimeLimit work_until) = 0;

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();
    };

    class AddressSplitChannel : public SingleXDQChannel<AddressSplitChannel, AddressSplitXferDesBase> {
    public:
      AddressSplitChannel(BackgroundWorkManager *bgwork);
      virtual ~AddressSplitChannel();

      // do as many of these concurrently as we like
      static const bool is_ordered = false;
      
      virtual long submit(Request** requests, long nr) { assert(0); return 0; }
    };
  
    class ChannelManager {
    public:
      ChannelManager(void) {
        memcpy_channel = NULL;
        gasnet_read_channel = gasnet_write_channel = NULL;
        remote_write_channel = NULL;
        disk_channel = NULL;
        file_channel = NULL;
#ifdef REALM_USE_HDF5
        hdf5_channel = NULL;
#endif
	addr_split_channel = 0;
      }
      ~ChannelManager(void);
      MemcpyChannel* create_memcpy_channel(BackgroundWorkManager *bgwork);
      GASNetChannel* create_gasnet_read_channel(BackgroundWorkManager *bgwork);
      GASNetChannel* create_gasnet_write_channel(BackgroundWorkManager *bgwork);
      RemoteWriteChannel* create_remote_write_channel(BackgroundWorkManager *bgwork);
      DiskChannel* create_disk_channel(BackgroundWorkManager *bgwork);
      FileChannel* create_file_channel(BackgroundWorkManager *bgwork);
      AddressSplitChannel *create_addr_split_channel(BackgroundWorkManager *bgwork);
#ifdef REALM_USE_CUDA
      GPUChannel* create_gpu_to_fb_channel(Cuda::GPU* src_gpu,
					   BackgroundWorkManager *bgwork);
      GPUChannel* create_gpu_from_fb_channel(Cuda::GPU* src_gpu,
					     BackgroundWorkManager *bgwork);
      GPUChannel* create_gpu_in_fb_channel(Cuda::GPU* src_gpu,
					   BackgroundWorkManager *bgwork);
      GPUChannel* create_gpu_peer_fb_channel(Cuda::GPU* src_gpu,
					     BackgroundWorkManager *bgwork);
#endif
#ifdef REALM_USE_HDF5
      HDF5Channel* create_hdf5_channel(BackgroundWorkManager *bgwork);
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
      DiskChannel* get_disk_channel() {
        return disk_channel;
      }
      FileChannel* get_file_channel() {
        return file_channel;
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
      HDF5Channel* get_hdf5_channel() {
        return hdf5_channel;
      }
#endif
    public:
      MemcpyChannel* memcpy_channel;
      GASNetChannel *gasnet_read_channel, *gasnet_write_channel;
      RemoteWriteChannel* remote_write_channel;
      DiskChannel *disk_channel;
      FileChannel *file_channel;
#ifdef REALM_USE_CUDA
      std::map<Cuda::GPU*, GPUChannel*> gpu_to_fb_channels, gpu_in_fb_channels, gpu_from_fb_channels, gpu_peer_fb_channels;
#endif
#ifdef REALM_USE_HDF5
      HDF5Channel *hdf5_channel;
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

      static void send_request(NodeID target,
			       const RemoteAddress& dst_buf,
                               const void *src_buf, size_t nbytes,
                               RemoteWriteRequest* req,
			       XferDesID next_xd_guid,
			       int next_port_idx,
			       size_t span_start,
			       size_t span_size,
			       size_t pre_bytes_total) 
      {
	ActiveMessage<XferDesRemoteWriteMessage> amsg(target,
						      src_buf, nbytes,
						      dst_buf);
	amsg->req = req;
	amsg->next_xd_guid = next_xd_guid;
	amsg->next_port_idx = next_port_idx;
	amsg->span_start = span_start;
	assert(span_size <= UINT_MAX);
	amsg->span_size = (unsigned)span_size;
	amsg->pre_bytes_total = pre_bytes_total;
	amsg.commit();
      }

      static void send_request(NodeID target,
			       const RemoteAddress& dst_buf,
                               const void *src_buf, size_t nbytes, off_t src_str,
                               size_t nlines, RemoteWriteRequest* req,
			       XferDesID next_xd_guid,
			       int next_port_idx,
			       size_t span_start,
			       size_t span_size,
			       size_t pre_bytes_total) 
      {
	ActiveMessage<XferDesRemoteWriteMessage> amsg(target,
						      src_buf, nbytes,
						      nlines, src_str,
						      dst_buf);
	amsg->req = req;
	amsg->next_xd_guid = next_xd_guid;
	amsg->next_port_idx = next_port_idx;
	amsg->span_start = span_start;
        assert(span_size <= UINT_MAX);
        amsg->span_size = (unsigned)span_size;
	amsg->pre_bytes_total = pre_bytes_total;
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

    struct UpdateBytesTotalMessage {
      XferDesID guid;
      int port_idx;
      size_t pre_bytes_total;

      static void handle_message(NodeID sender,
				 const UpdateBytesTotalMessage &args,
				 const void *data,
				 size_t datalen);
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
      XferDesQueue()
      //: core_rsrv("DMA request queue", crs, CoreReservationParameters())
      {
        // reserve the first several guid
        next_to_assign_idx.store(10);
      }

      ~XferDesQueue() {
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
				  size_t span_start, size_t span_size);
      void update_pre_bytes_total(XferDesID xd_guid, int port_idx,
				  size_t pre_bytes_total);
      void update_next_bytes_read(XferDesID xd_guid, int port_idx,
				  size_t span_start, size_t span_size);

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
	xd->remove_reference();
      }

      // returns true if xd is ready, false if enqueue has been deferred
      bool enqueue_xferDes_local(XferDes* xd, bool add_to_queue = true);

      void start_worker(ChannelManager* channel_manager,
			BackgroundWorkManager *bgwork);

      void stop_worker();

    protected:
      std::map<XferDesID, XferDesWithUpdates> guid_to_xd;
      Mutex queues_lock;
      RWLock guid_lock;
      atomic<XferDesID> next_to_assign_idx;
    };

    XferDesQueue* get_xdq_singleton();
    ChannelManager* get_channel_manager();
#ifdef REALM_USE_CUDA
    void register_gpu_in_dma_systems(Cuda::GPU* gpu);
#endif
    void start_channel_manager(BackgroundWorkManager *bgwork);
    void stop_channel_manager();

    void destroy_xfer_des(XferDesID _guid);
}; // namespace Realm

#include "realm/transfer/channel.inl"

#endif

