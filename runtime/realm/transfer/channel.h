/* Copyright 2022 Stanford University
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

#include "realm/id.h"
#include "realm/runtime_impl.h"
#include "realm/mem_impl.h"
#include "realm/inst_impl.h"
#include "realm/bgwork.h"
#include "realm/utils.h"

namespace Realm {

    class XferDes;
    class XferDesQueue;
    class Channel;
    class DmaRequest;
    class TransferIterator;

    extern Logger log_new_dma;

    typedef unsigned long long XferDesID;

    enum XferDesKind {
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
      XFER_HDF5_READ,
      XFER_HDF5_WRITE,
      XFER_FILE_READ,
      XFER_FILE_WRITE,
      XFER_ADDR_SPLIT,
      XFER_MEM_FILL,
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

      // NOT thread-safe - caller must ensure neither *this nor other is being
      //  modified during this call
      void swap(SequenceAssembler& other);

      // imports data from this assembler into another (this is thread-safe
      //  on the `other` but assumes no changes being made on `this`)
      void import(SequenceAssembler& other) const;

      bool empty() const;

      // asks if a span exists - return value is number of bytes from the
      //  start that do
      size_t span_exists(size_t start, size_t count);

      // returns the amount by which the contiguous range has been increased
      //  (i.e. from [pos, pos+retval) )
      size_t add_span(size_t pos, size_t count);

    protected:
      Mutex *ensure_mutex();

      atomic<size_t> contig_amount_x2;  // everything from [0, contig_amount) is covered - LSB indicates potential presence of noncontig spans
      atomic<size_t> first_noncontig; // nothing in [contig_amount, first_noncontig)
      atomic<Mutex *> mutex; // created on first use
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

    struct XferDesRedopInfo {
      ReductionOpID id;
      bool is_fold;
      bool in_place;
      bool is_exclusive;

      // default constructor == no reduction requested
      XferDesRedopInfo() : id(0), is_fold(false), in_place(false), is_exclusive(false) {}

      XferDesRedopInfo(ReductionOpID _id, bool _is_fold, bool _in_place, bool _is_exclusive)
      : id(_id), is_fold(_is_fold), in_place(_in_place), is_exclusive(_is_exclusive) {}
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

      int get_dim() const;
      uintptr_t get_offset() const;
      uintptr_t get_stride(int dim) const;
      size_t remaining(int dim) const;
      void advance(int dim, size_t amount);

      void skip_bytes(size_t bytes);
      
    protected:
      AddressList *addrlist;
      bool partial;
      // we need to be one larger than any index space realm supports, since
      //  we use the contiguous bytes within a field as a "dimension" in some
      //  cases
      static const int MAX_DIM = REALM_MAX_DIM + 1;
      int partial_dim;
      size_t pos[MAX_DIM];
    };

    std::ostream& operator<<(std::ostream& os, const AddressListCursor& alc);

    // a control port is used to steer inputs/outputs of transfer descriptors -
    //   the information is encoded into 32b packets which may be read/written
    //   at different times due to flow control, so the encoder and decoder
    //   both need to be stateful
    namespace ControlPort {
      // apart from the first control word (which carries the space_shift
      //   amount), the bottom two bits of each control word mean:
      static const unsigned CTRL_LO_MORE = 0; // 00: count.lo, space_index, not last
      static const unsigned CTRL_LO_LAST = 1; // 01: count.lo, space_index, last
      static const unsigned CTRL_MID     = 2; // 10: count.mid (32 bits)
      static const unsigned CTRL_HIGH    = 3; // 11: count.hi (2+space_shift bits)

      class Encoder {
      public:
        Encoder();
        ~Encoder();

        void set_port_count(size_t ports);

        // encodes some/all of the { count, port, last } packet into the next
        //  32b - returns true if encoding is complete or false if it should
        //  be called again with the same arguments for another 32b packet
        bool encode(unsigned& data, size_t count, int port, bool last);

      protected:
        unsigned short port_shift;

        enum State {
          STATE_INIT,
          STATE_HAVE_PORT_COUNT,
          STATE_IDLE,
          STATE_SENT_HIGH,
          STATE_SENT_MID,
          STATE_DONE
        };
        unsigned char state;
      };

      class Decoder {
      public:
        Decoder();
        ~Decoder();

        // decodes the next 32b of packed data, returning true if a complete
        //  { count, port, last } has been received
        bool decode(unsigned data,
                    size_t& count, int& port, bool& last);

      protected:
        size_t temp_count;
        unsigned short port_shift;
      };
    };

    class XferDes {
    public:
      // a pointer to the DmaRequest that contains this XferDes
      uintptr_t dma_op;
      XferDesQueue *xferDes_queue;
      // ID of the node that launches this XferDes
      NodeID launch_node;
      //uint64_t /*bytes_submit, */bytes_read, bytes_write/*, bytes_total*/;
      atomic<bool> iteration_completed;
      atomic<int64_t> bytes_write_pending;
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
        ControlPort::Decoder decoder;
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

      void *fill_data;
      size_t fill_size, orig_fill_size;
      // for most fills, we can use an inline buffer with good alignment
      static const size_t ALIGNED_FILL_STORAGE_SIZE = 32;
      struct UnalignedStorage { char data[ALIGNED_FILL_STORAGE_SIZE]; };
      REALM_ALIGNED_TYPE_CONST(AlignedStorage, UnalignedStorage, 16);
      AlignedStorage inline_fill_storage;

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
      XferDes(uintptr_t _dma_op, Channel *_channel,
	      NodeID _launch_node, XferDesID _guid,
	      const std::vector<XferDesPortInfo>& inputs_info,
	      const std::vector<XferDesPortInfo>& outputs_info,
	      int _priority,
              const void *_fill_data, size_t fill_size);

      // transfer descriptors are reference counted rather than explcitly
      //  deleted
      void add_reference(void);
      void remove_reference(void);

    protected:
      virtual ~XferDes();

    public:
      virtual Event request_metadata();

      virtual long get_requests(Request** requests, long nr) = 0;

      virtual void notify_request_read_done(Request* req);

      virtual void notify_request_write_done(Request* req);

      virtual void flush();
 
      long default_get_requests(Request** requests, long nr, unsigned flags = 0);
      void default_notify_request_read_done(Request* req);
      void default_notify_request_write_done(Request* req);

      virtual void update_bytes_read(int port_idx, size_t offset, size_t size);
      virtual void update_bytes_write(int port_idx, size_t offset, size_t size);
      void update_pre_bytes_write(int port_idx, size_t offset, size_t size);
      void update_pre_bytes_total(int port_idx, size_t pre_bytes_total);
      void update_next_bytes_read(int port_idx, size_t offset, size_t size);

      // called once iteration is complete, but we need to track in flight
      //  writes, flush byte counts, etc.
      void begin_completion();

      void mark_completed();

      unsigned current_progress(void);

      // checks to see if progress has been made since the last read of the
      //  progress counter - atomically marks the xd for wakeup if not
      bool check_for_progress(unsigned last_counter);

      // updates the progress counter, waking up the xd if needed
      void update_progress(void);

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
        // update progress counter if iteration isn't completed yet - it might
        //  have been waiting for another request object
        if(!iteration_completed.load())
          update_progress();
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
      size_t get_addresses(size_t min_xfer_size, ReadSequenceCache *rseqcache,
                           const InstanceLayoutPieceBase *&in_nonaffine,
                           const InstanceLayoutPieceBase *&out_nonaffine);

      // after a call to 'get_addresses', this call updates the various data
      //  structures to record that transfers for 'total_{read,write}_bytes' bytes
      //  were at least initiated - return value is whether iteration is complete
      bool record_address_consumption(size_t total_read_bytes,
                                      size_t total_write_bytes);

      // fills can be more efficient if the fill data is replicated into a larger
      //  block
      void replicate_fill_data(size_t new_size);
    };

    class MemcpyChannel;

    class MemcpyXferDes : public XferDes {
    public:
      MemcpyXferDes(uintptr_t _dma_op, Channel *_channel,
		    NodeID _launch_node, XferDesID _guid,
		    const std::vector<XferDesPortInfo>& inputs_info,
		    const std::vector<XferDesPortInfo>& outputs_info,
		    int _priority);

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

    class MemfillChannel;

    class MemfillXferDes : public XferDes {
    public:
      MemfillXferDes(uintptr_t _dma_op, Channel *_channel,
		     NodeID _launch_node, XferDesID _guid,
		     const std::vector<XferDesPortInfo>& inputs_info,
		     const std::vector<XferDesPortInfo>& outputs_info,
		     int _priority,
		     const void *_fill_data, size_t _fill_size,
                     size_t _fill_total);

      long get_requests(Request** requests, long nr);

      virtual bool request_available();
      virtual Request* dequeue_request();
      virtual void enqueue_request(Request* req);

      bool progress_xd(MemfillChannel *channel, TimeLimit work_until);
    };

    class MemreduceChannel;

    class MemreduceXferDes : public XferDes {
    public:
      MemreduceXferDes(uintptr_t _dma_op, Channel *_channel,
                       NodeID _launch_node, XferDesID _guid,
                       const std::vector<XferDesPortInfo>& inputs_info,
                       const std::vector<XferDesPortInfo>& outputs_info,
                       int _priority,
                       XferDesRedopInfo _redop_info);

      long get_requests(Request** requests, long nr);

      bool progress_xd(MemreduceChannel *channel, TimeLimit work_until);

    protected:
      XferDesRedopInfo redop_info;
      const ReductionOpUntyped *redop;
    };

    class GASNetChannel;

    class GASNetXferDes : public XferDes {
    public:
      GASNetXferDes(uintptr_t _dma_op, Channel *_channel,
		    NodeID _launch_node, XferDesID _guid,
		    const std::vector<XferDesPortInfo>& inputs_info,
		    const std::vector<XferDesPortInfo>& outputs_info,
		    int _priority);

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
      RemoteWriteXferDes(uintptr_t _dma_op, Channel *_channel,
			 NodeID _launch_node, XferDesID _guid,
			 const std::vector<XferDesPortInfo>& inputs_info,
			 const std::vector<XferDesPortInfo>& outputs_info,
			 int _priority);

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
	size_t span_start;

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

    class XferDesFactory {
    protected:
      // do not destroy directly - use release()
      virtual ~XferDesFactory() {}

    public:
      virtual void release() = 0;

      virtual void create_xfer_des(uintptr_t dma_op,
				   NodeID launch_node,
				   NodeID target_node,
				   XferDesID guid,
				   const std::vector<XferDesPortInfo>& inputs_info,
				   const std::vector<XferDesPortInfo>& outputs_info,
				   int priority,
				   XferDesRedopInfo redop_info,
				   const void *fill_data, size_t fill_size,
                                   size_t fill_total) = 0;
    };

    struct XferDesCreateMessageBase {
      //RegionInstance inst;
      uintptr_t dma_op;
      XferDesID guid;
      NodeID launch_node;
      uintptr_t channel;
    };

    struct SimpleXferDesCreateMessage : public XferDesCreateMessageBase {
      static void handle_message(NodeID sender,
				 const SimpleXferDesCreateMessage &args,
				 const void *data,
				 size_t datalen);
    };

    // a simple xfer des factory knows how to create an xfer des with no
    //  extra information on a single channel
    class SimpleXferDesFactory : public XferDesFactory {
    public:
      SimpleXferDesFactory(uintptr_t _channel);

      virtual void release();

      virtual void create_xfer_des(uintptr_t dma_op,
				   NodeID launch_node,
				   NodeID target_node,
				   XferDesID guid,
				   const std::vector<XferDesPortInfo>& inputs_info,
				   const std::vector<XferDesPortInfo>& outputs_info,
				   int priority,
				   XferDesRedopInfo redop_info,
				   const void *fill_data, size_t fill_size,
                                   size_t fill_total);

    protected:
      uintptr_t channel;
    };

    class RemoteChannelInfo;
    class RemoteChannel;

    class Channel {
    public:
      Channel(XferDesKind _kind)
	: node(Network::my_node_id), kind(_kind) {}
      virtual ~Channel() {};

      // TODO: make pure virtual
      virtual void shutdown() {}

      // most channels can hand out a factory that makes arbitrary xds on
      //  that channel
      virtual XferDesFactory *get_factory() = 0;

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
          MEMORY_BITMASK,
	};
	SrcDstType src_type, dst_type;
        struct MemBitmask {
          NodeID node;
          static const int BITMASK_SIZE = (1 << ID::MEMORY_INDEX_WIDTH) >> 6;
          uint64_t mems[BITMASK_SIZE], ib_mems[BITMASK_SIZE];
        };
	union {
	  Memory src_mem;
	  Memory::Kind src_kind;
          MemBitmask src_bitmask;
	};
	union {
	  Memory dst_mem;
	  Memory::Kind dst_kind;
          MemBitmask dst_bitmask;
	};
	XferDesKind xd_kind;
	unsigned bandwidth; // units = MB/s = B/us
	unsigned latency;   // units = ns
        unsigned frag_overhead; // units = ns
        unsigned char max_src_dim, max_dst_dim;
	bool redops_allowed; // TODO: list of redops?
	bool serdez_allowed; // TODO: list of serdez ops?

        // mutators to modify less-common fields
        SupportedPath& set_max_dim(int src_and_dst_dim);
        SupportedPath& set_max_dim(int src_dim, int dst_dim);
        SupportedPath& allow_redops();
        SupportedPath& allow_serdez();

        // only valid when a SupportedPath is modifiable by the above methods
        //  (i.e. only on the creator node and only until another path is added)
        SupportedPath *chain;

        void populate_memory_bitmask(span <const Memory> mems,
                                     NodeID node,
                                     MemBitmask& bitmask);
      };

      const std::vector<SupportedPath>& get_paths(void) const;

      // returns 0 if the path is not supported, or a strictly-positive
      //  estimate of the time required (in nanoseconds) to transfer data
      //  along a supported path
      virtual uint64_t supports_path(Memory src_mem, Memory dst_mem,
                                     CustomSerdezID src_serdez_id,
                                     CustomSerdezID dst_serdez_id,
                                     ReductionOpID redop_id,
                                     size_t total_bytes,
                                     const std::vector<size_t> *src_frags,
                                     const std::vector<size_t> *dst_frags,
                                     XferDesKind *kind_ret = 0,
                                     unsigned *bw_ret = 0,
                                     unsigned *lat_ret = 0);

      virtual RemoteChannelInfo *construct_remote_info() const;

      void print(std::ostream& os) const;

      virtual void enqueue_ready_xd(XferDes *xd) = 0;
      virtual void wakeup_xd(XferDes *xd) = 0;

    protected:
      // returns the added path for further modification, but reference is
      //  only valid until the next call to 'add_path'
      SupportedPath& add_path(span<const Memory> src_mems,
                              span<const Memory> dst_mems,
                              unsigned bandwidth, unsigned latency,
                              unsigned frag_overhead,
                              XferDesKind xd_kind);
      SupportedPath& add_path(span<const Memory> src_mems,
                              Memory::Kind dst_kind, bool dst_global,
                              unsigned bandwidth, unsigned latency,
                              unsigned frag_overhead,
                              XferDesKind xd_kind);
      SupportedPath& add_path(Memory::Kind src_kind, bool src_global,
                              span<const Memory> dst_mems,
                              unsigned bandwidth, unsigned latency,
                              unsigned frag_overhead,
                              XferDesKind xd_kind);
      SupportedPath& add_path(Memory::Kind src_kind, bool src_global,
                              Memory::Kind dst_kind, bool dst_global,
                              unsigned bandwidth, unsigned latency,
                              unsigned frag_overhead,
                              XferDesKind xd_kind);
      // TODO: allow rdma path to limit by kind?
      SupportedPath& add_path(bool local_loopback,
                              unsigned bandwidth, unsigned latency,
                              unsigned frag_overhead,
                              XferDesKind xd_kind);

      std::vector<SupportedPath> paths;
    };

 
    std::ostream& operator<<(std::ostream& os, const Channel::SupportedPath& p);

    class LocalChannel : public Channel {
    public:
      LocalChannel(XferDesKind _kind);

      virtual XferDes *create_xfer_des(uintptr_t dma_op,
				       NodeID launch_node,
				       XferDesID guid,
				       const std::vector<XferDesPortInfo>& inputs_info,
				       const std::vector<XferDesPortInfo>& outputs_info,
				       int priority,
				       XferDesRedopInfo redop_info,
				       const void *fill_data, size_t fill_size,
                                       size_t fill_total) = 0;

      virtual XferDesFactory *get_factory();

    protected:
      SimpleXferDesFactory factory_singleton;
    };      

    // polymorphic container for info necessary to create a remote channel
    class REALM_INTERNAL_API_EXTERNAL_LINKAGE RemoteChannelInfo {
    public:
      virtual ~RemoteChannelInfo() {};

      virtual RemoteChannel *create_remote_channel() = 0;

      template <typename S>
      static RemoteChannelInfo *deserialize_new(S& deserializer);
    };

    template <typename S>
    bool serialize(S& serializer, const RemoteChannelInfo& rci);

    class SimpleRemoteChannelInfo : public RemoteChannelInfo {
    public:
      SimpleRemoteChannelInfo(NodeID _owner, XferDesKind _kind,
                              uintptr_t _remote_ptr,
                              const std::vector<Channel::SupportedPath>& _paths);

      virtual RemoteChannel *create_remote_channel();

      template <typename S>
      bool serialize(S& serializer) const;

      template <typename S>
      static RemoteChannelInfo *deserialize_new(S& deserializer);

    protected:
      SimpleRemoteChannelInfo();

      static Serialization::PolymorphicSerdezSubclass<RemoteChannelInfo, SimpleRemoteChannelInfo> serdez_subclass;

      NodeID owner;
      XferDesKind kind;
      uintptr_t remote_ptr;
      std::vector<Channel::SupportedPath> paths;
    };

    class RemoteChannel : public Channel {
    protected:
      friend class SimpleRemoteChannelInfo;

      RemoteChannel(uintptr_t _remote_ptr);

      virtual void shutdown();

      virtual XferDesFactory *get_factory();

    public:
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

      virtual uint64_t supports_path(Memory src_mem, Memory dst_mem,
                                     CustomSerdezID src_serdez_id,
                                     CustomSerdezID dst_serdez_id,
                                     ReductionOpID redop_id,
                                     size_t total_bytes,
                                     const std::vector<size_t> *src_frags,
                                     const std::vector<size_t> *dst_frags,
                                     XferDesKind *kind_ret = 0,
                                     unsigned *bw_ret = 0,
                                     unsigned *lat_ret = 0);

      virtual void enqueue_ready_xd(XferDes *xd) { assert(0); }
      virtual void wakeup_xd(XferDes *xd) { assert(0); }

    protected:
      SimpleXferDesFactory factory_singleton;
    };

    template <typename CHANNEL, typename XD>
    class XDQueue : public BackgroundWorkItem {
    public:
      XDQueue(LocalChannel *_channel, const std::string& _name,
	      bool _ordered);

      void enqueue_xd(XD *xd, bool at_front = false);

      virtual bool do_work(TimeLimit work_until);

    protected:
      friend CHANNEL;

      LocalChannel *channel;
      bool ordered_mode, in_ordered_worker;
      Mutex mutex;
      XferDes::XferDesList ready_xds;
    };

    template <typename CHANNEL, typename XD>
    class SingleXDQChannel : public LocalChannel {
    public:
      SingleXDQChannel(BackgroundWorkManager *bgwork,
		       XferDesKind _kind,
		       const std::string& _name,
                       int _numa_domain = -1);

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

      // helper to list all memories that can be reached by load/store instructions
      //  on the cpu in the current process
      static void enumerate_local_cpu_memories(std::vector<Memory>& mems);

      virtual uint64_t supports_path(Memory src_mem, Memory dst_mem,
                                     CustomSerdezID src_serdez_id,
                                     CustomSerdezID dst_serdez_id,
                                     ReductionOpID redop_id,
                                     size_t total_bytes,
                                     const std::vector<size_t> *src_frags,
                                     const std::vector<size_t> *dst_frags,
                                     XferDesKind *kind_ret = 0,
                                     unsigned *bw_ret = 0,
                                     unsigned *lat_ret = 0);

      virtual XferDes *create_xfer_des(uintptr_t dma_op,
				       NodeID launch_node,
				       XferDesID guid,
				       const std::vector<XferDesPortInfo>& inputs_info,
				       const std::vector<XferDesPortInfo>& outputs_info,
				       int priority,
				       XferDesRedopInfo redop_info,
				       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      virtual long submit(Request** requests, long nr);

      bool is_stopped;
    };

    class MemfillChannel : public SingleXDQChannel<MemfillChannel, MemfillXferDes> {
    public:
      MemfillChannel(BackgroundWorkManager *bgwork);

      // multiple concurrent memfills ok
      static const bool is_ordered = false;

      ~MemfillChannel();

      virtual XferDes *create_xfer_des(uintptr_t dma_op,
				       NodeID launch_node,
				       XferDesID guid,
				       const std::vector<XferDesPortInfo>& inputs_info,
				       const std::vector<XferDesPortInfo>& outputs_info,
				       int priority,
				       XferDesRedopInfo redop_info,
				       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      virtual long submit(Request** requests, long nr);

      bool is_stopped;
    };

    class MemreduceChannel : public SingleXDQChannel<MemreduceChannel, MemreduceXferDes> {
    public:
      MemreduceChannel(BackgroundWorkManager *bgwork);

      // multiple concurrent memreduces ok
      static const bool is_ordered = false;

      // override because we don't want to claim non-reduction copies
      virtual uint64_t supports_path(Memory src_mem, Memory dst_mem,
                                     CustomSerdezID src_serdez_id,
                                     CustomSerdezID dst_serdez_id,
                                     ReductionOpID redop_id,
                                     size_t total_bytes,
                                     const std::vector<size_t> *src_frags,
                                     const std::vector<size_t> *dst_frags,
                                     XferDesKind *kind_ret = 0,
                                     unsigned *bw_ret = 0,
                                     unsigned *lat_ret = 0);

      virtual XferDes *create_xfer_des(uintptr_t dma_op,
				       NodeID launch_node,
				       XferDesID guid,
				       const std::vector<XferDesPortInfo>& inputs_info,
				       const std::vector<XferDesPortInfo>& outputs_info,
				       int priority,
				       XferDesRedopInfo redop_info,
				       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      virtual long submit(Request** requests, long nr);

      bool is_stopped;
    };

    class GASNetChannel : public SingleXDQChannel<GASNetChannel, GASNetXferDes> {
    public:
      GASNetChannel(BackgroundWorkManager *bgwork, XferDesKind _kind);
      ~GASNetChannel();

      // no more than one GASNet xfer of each type at a time
      static const bool is_ordered = true;
      
      virtual XferDes *create_xfer_des(uintptr_t dma_op,
				       NodeID launch_node,
				       XferDesID guid,
				       const std::vector<XferDesPortInfo>& inputs_info,
				       const std::vector<XferDesPortInfo>& outputs_info,
				       int priority,
				       XferDesRedopInfo redop_info,
				       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request** requests, long nr);
    };

    class RemoteWriteChannel : public SingleXDQChannel<RemoteWriteChannel, RemoteWriteXferDes> {
    public:
      RemoteWriteChannel(BackgroundWorkManager *bgwork);
      ~RemoteWriteChannel();

      // multiple concurrent RDMAs ok
      static const bool is_ordered = false;

      virtual XferDes *create_xfer_des(uintptr_t dma_op,
				       NodeID launch_node,
				       XferDesID guid,
				       const std::vector<XferDesPortInfo>& inputs_info,
				       const std::vector<XferDesPortInfo>& outputs_info,
				       int priority,
				       XferDesRedopInfo redop_info,
				       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request** requests, long nr);
    };
   
    class AddressSplitChannel;

    class AddressSplitXferDesBase : public XferDes {
    protected:
      AddressSplitXferDesBase(uintptr_t dma_op, Channel *_channel,
			      NodeID _launch_node, XferDesID _guid,
			      const std::vector<XferDesPortInfo>& inputs_info,
			      const std::vector<XferDesPortInfo>& outputs_info,
			      int _priority);

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

      // override this to make sure it's never called
      virtual XferDes *create_xfer_des(uintptr_t dma_op,
				       NodeID launch_node,
				       XferDesID guid,
				       const std::vector<XferDesPortInfo>& inputs_info,
				       const std::vector<XferDesPortInfo>& outputs_info,
				       int priority,
				       XferDesRedopInfo redop_info,
				       const void *fill_data, size_t fill_size,
                                       size_t fill_total) { assert(0); return 0; }

      virtual long submit(Request** requests, long nr) { assert(0); return 0; }

    protected:
      static AddressSplitChannel *local_channel;
    };
  
    class TransferOperation;
    struct NotifyXferDesCompleteMessage {
      TransferOperation *op;
      XferDesID xd_id;

      static void handle_message(NodeID sender,
				 const NotifyXferDesCompleteMessage &args,
				 const void *data,
				 size_t datalen);

      static void send_request(NodeID target, TransferOperation *op, XferDesID xd_id);
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
      size_t span_start, span_size;

      static void handle_message(NodeID sender,
				 const UpdateBytesWriteMessage &args,
				 const void *data,
				 size_t datalen);

      static void send_request(NodeID target, XferDesID guid,
			       int port_idx,
			       size_t span_start, size_t span_size)
      {
	ActiveMessage<UpdateBytesWriteMessage> amsg(target);
        amsg->guid = guid;
	amsg->port_idx = port_idx;
	amsg->span_start = span_start;
	amsg->span_size = span_size;
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

    // object used to hold input progress (pre_write and bytes_total) before
    //  we've actually created the correct xd
    class XferDesPlaceholder {
    public:
      XferDesPlaceholder();

    protected:
      ~XferDesPlaceholder();

    public:
      void add_reference();
      void remove_reference();

      void update_pre_bytes_write(int port_idx,
				  size_t span_start, size_t span_size);
      void update_pre_bytes_total(int port_idx, size_t pre_bytes_total);

      void set_real_xd(XferDes *_xd);

    protected:
      static const int INLINE_PORTS = 4;
      atomic<unsigned> refcount;
      XferDes *xd;
      size_t inline_bytes_total[INLINE_PORTS];
      SequenceAssembler inline_pre_write[INLINE_PORTS];
      Mutex extra_mutex;
      std::map<int, size_t> extra_bytes_total;
      std::map<int, SequenceAssembler> extra_pre_write;
    };

    class XferDesQueue {
    public:
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

      static XferDesQueue* get_singleton();

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

      void destroy_xferDes(XferDesID guid);

      // returns true if xd is ready, false if enqueue has been deferred
      bool enqueue_xferDes_local(XferDes* xd, bool add_to_queue = true);

    protected:
      // guid_to_xd maps a guid to either an XferDes * (as a uintptr_t) or
      //  a XferDesPlaceholder * (as a uintptr_t with the LSB set)
      Mutex guid_lock;
      std::map<XferDesID, uintptr_t> guid_to_xd;

      Mutex queues_lock;
      atomic<XferDesID> next_to_assign_idx;
    };

    void destroy_xfer_des(XferDesID _guid);
}; // namespace Realm

#include "realm/transfer/channel.inl"

#endif

