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

#ifndef GASNETEX_INTERNAL_H
#define GASNETEX_INTERNAL_H

#include "realm/gasnetex/gasnetex_module.h"

#include "realm/bgwork.h"
#include "realm/atomics.h"
#include "realm/activemsg.h"
#include "realm/lists.h"

#ifndef GASNET_PAR
  #if defined(GASNET_SEQ) || defined(GASNET_PARSYNC)
    #error Realm requires GASNet-EX be used in parallel threading mode!
  #else
    #define GASNET_PAR
  #endif
#endif
#include <gasnetex.h>

// there are two independent "version" that we may need to consider for
//  conditional compilation:
//
// 1) REALM_GEX_RELEASE refers to specific releases - GASNet-EX uses year.month
//      for major.minor, and we'll assume no more than 100 patch levels to
//      avoid conflicts, but there is no guarantee of chronological
//      monotonicity of behavior, so tests should be either equality against
//      a specific release or a bounded comparison when two or more consecutive
//      releases are of interest.  However, there should never be anything of
//      form: if REALM_GEX_RELEASE >= xyz
#define REALM_GEX_RELEASE ((10000*GASNET_RELEASE_VERSION_MAJOR)+(100*GASNET_RELEASE_VERSION_MINOR)+GASNET_RELEASE_VERSION_PATCH)

// 2) REALM_GEX_API refers to versioning of the GASNet-EX specification -
//      currently this is defined in terms of major.minor, but we'll include
//      space for a patch level if that ever becomes important.  In contrast to
//      the release numbering, we will assume that the specification is
//      roughly monotonic in that a change is expected to remain in future
//      specs except for hopefully-uncommon cases where it changes again
#define REALM_GEX_API  ((10000*GEX_SPEC_VERSION_MAJOR)+(100*GEX_SPEC_VERSION_MINOR))

#if REALM_GEX_API < 1200
#error Realm depends on GASNet-EX features that first appeared in the 0.12 spec, first available in the 2020.11.0 release.  For earlier versions of GASNet-EX, use the legacy API via the gasnet1 network layer.
  #include <stop_compilation_due_to_gasnetex_version_mismatch>
#endif

// post 2020.11.0, GASNet has defines that say which operations are native
//  rather than emulated by their reference implementation - those defines
//  aren't there for 2020.11.0, but the only one that version has that we
//  care about is NPAM medium
#if REALM_GEX_RELEASE == 20201100
  // NOTE: technically it's only native for the IBV/ARIES/SMP conduits,
  //  but enable it as well on the MPI conduit so that we get more test
  //  coverage of the code paths (and it's probably not making the MPI
  //  conduit performance any worse)
  #if defined(GASNET_CONDUIT_IBV) || defined(GASNET_CONDUIT_ARIES) || defined(GASNET_CONDUIT_SMP) || defined(GASNET_CONDUIT_MPI)
    #define GASNET_NATIVE_NP_ALLOC_REQ_MEDIUM
  #endif
#endif

// the GASNet-EX API defines the GEX_FLAG_IMMEDIATE flag to be a best-effort
//  thing, with calls that accept the flag still being allowed to block -
//  as of 2022.3.0, for any conduit other than aries "best effort" is actually
//  "no effort" for RMA operations and we want to avoid using them in
//  immediate-mode situations
// NOTE: as with the NPAM stuff above, we'll pretend that MPI honors it as
//  well so that we get code coverage in CI tests
#if defined(GASNET_CONDUIT_ARIES) || defined(GASNET_CONDUIT_MPI)
  #define REALM_GEX_RMA_HONORS_IMMEDIATE_FLAG
#endif

// eliminate GASNet warnings for unused static functions
#include <gasnet_tools.h>
REALM_ATTR_UNUSED(static const void *ignore_gasnet_warning1) = (void *)_gasneti_threadkey_init;
REALM_ATTR_UNUSED(static const void *ignore_gasnet_warning2) = (void *)_gasnett_trace_printf_noop;

#define CHECK_GEX(cmd) do { \
  int ret = (cmd); \
  if(ret != GASNET_OK) { \
    fprintf(stderr, "GEX: %s = %d (%s, %s)\n", #cmd, ret, gasnet_ErrorName(ret), gasnet_ErrorDesc(ret)); \
    exit(1); \
  } \
} while(0)

namespace Realm {

  // rdma pointers need to identify which endpoint they belong to
  struct GASNetEXRDMAInfo {
    uintptr_t base;
    gex_EP_Index_t ep_index;
  };

  class OutbufManager;

  class OutbufMetadata {
  public:
    OutbufMetadata();

    enum State {
      STATE_IDLE,
      STATE_DATABUF,
      STATE_PKTBUF,
    };

    void dec_usecount();

    uintptr_t databuf_reserve(size_t bytes);
    void databuf_close();

    enum PktType {
      PKTTYPE_INVALID,
      PKTTYPE_INLINE,
      PKTTYPE_INLINE_SHORT,
      PKTTYPE_LONG,
      PKTTYPE_RGET,
      PKTTYPE_PUT,
      PKTTYPE_CANCELLED,
      PKTTYPE_COPY_IN_PROGRESS,
    };

    static const int MAX_PACKETS = 256;

    bool pktbuf_reserve(size_t bytes, int& pktidx, uintptr_t& offset);
    void pktbuf_close();
    uintptr_t pktbuf_get_offset(int pktidx);
    bool pktbuf_commit(int pktidx, PktType pkttype, bool update_realbuf);

  protected:
    friend class OutbufManager;
    friend class XmitSrcDestPair;

    void set_state(State new_state);

    State state;
    OutbufManager *manager;
    OutbufMetadata *nextbuf;
    uintptr_t baseptr;
    size_t size;

    bool is_overflow;
    OutbufMetadata *next_overflow;
    atomic<OutbufMetadata *> realbuf;

    atomic<int> remain_count;

    // dbuf reservations are NOT thread-safe - external serialization is used
    size_t databuf_rsrv_offset;
    int databuf_use_count;

    // pbuf reservatsions are NOT thread-safe - external serialization is used
    atomic<int> pktbuf_total_packets;  // unsynchronized read ok
    size_t pktbuf_rsrv_offset;
    size_t pktbuf_pkt_ends[MAX_PACKETS];  // stores where a packet _ends_

    // pbuf commits are lock-free
    atomic<PktType> pktbuf_pkt_types[MAX_PACKETS];
    atomic<int> pktbuf_ready_packets;

    // pbuf consumption is NOT thread-safe - one pusher at a time
    int pktbuf_sent_packets;
    size_t pktbuf_sent_offset;
    int pktbuf_use_count;
  };

  class OutbufUsecountDec {
  public:
    OutbufUsecountDec(OutbufMetadata *_md) : md(_md) {}

    void operator()() const { md->dec_usecount(); }

  protected:
    OutbufMetadata* md;
  };

  class OutbufManager : public BackgroundWorkItem {
  public:
    OutbufManager();
    ~OutbufManager();

    void init(size_t _outbuf_count, size_t _outbuf_size,
	      uintptr_t _baseptr);

    OutbufMetadata *alloc_outbuf(OutbufMetadata::State state,
				 bool overflow_ok);
    void free_outbuf(OutbufMetadata *md);

    virtual bool do_work(TimeLimit work_until);

  protected:
    OutbufMetadata *metadatas;
    size_t outbuf_size;
    Mutex mutex;
    OutbufMetadata *first_available;
    // we manage the copying over overflow bufs back to real outbufs - track
    //  both pending overflow bufs and reserved outbufs and request bgwork
    //  time when we've got at least one of each
    size_t num_overflow, num_reserved;
    OutbufMetadata *overflow_head;
    OutbufMetadata **overflow_tail;
    OutbufMetadata *reserved_head;
  };

  class PendingCompletionManager;

  class PendingCompletion {
  public:
    PendingCompletion();

    void *add_local_completion(size_t bytes, bool late_ok = false);
    void *add_remote_completion(size_t bytes);

    // marks ready and returns true if non-empty, else resets and returns false
    bool mark_ready(unsigned exp_local, unsigned exp_remote);

    bool has_local_completions();
    bool has_remote_completions();

    // these two calls can be concurrent, which complicates the determination of
    //  who can free the entry
    bool invoke_local_completions();
    bool invoke_remote_completions();

    int index;
    PendingCompletion *next_free;
    PendingCompletionManager *manager;
    // these three bits need to be in a single atomic location
    static const unsigned LOCAL_PENDING_BIT = 1;
    static const unsigned REMOTE_PENDING_BIT = 2;
    static const unsigned READY_BIT = 4;
    Realm::atomic<unsigned> state;
    Realm::atomic<unsigned> local_left, remote_left;
    size_t local_bytes, remote_bytes;

    static const size_t TOTAL_CAPACITY = 256;
    typedef char Storage_unaligned[TOTAL_CAPACITY];
    REALM_ALIGNED_TYPE_CONST(Storage_aligned, Storage_unaligned,
			     Realm::CompletionCallbackBase::ALIGNMENT);
    Storage_aligned storage;
  };

  struct PendingCompletionGroup {
    static const size_t LOG2_GROUPSIZE = 8; // 256 per group

    PendingCompletion entries[1 << LOG2_GROUPSIZE];
  };

  class PendingCompletionManager {
  public:
    PendingCompletionManager();
    ~PendingCompletionManager();

    PendingCompletion *get_available();
    void recycle_comp(PendingCompletion *comp);

    PendingCompletion *lookup_completion(int index);
    void invoke_completions(PendingCompletion *comp,
			    bool do_local, bool do_remote);

    size_t num_completions_pending();

    bool over_pending_completion_soft_limit() const;

  protected:
    // NOTE: we stuff completion IDs, message IDs and 2 more bits into a
    //  32-bit word, so we're limited to 2^(30-12) = 256K completions
    static const size_t LOG2_MAXGROUPS = 10;

    // protects pops from the free list (to avoid A-B-A problem), but NOT pushes
    Realm::Mutex mutex;
    Realm::atomic<PendingCompletion *> first_free;
    Realm::atomic<int> num_groups; // number of groups currently allocated
    Realm::atomic<PendingCompletionGroup *> groups[1 << LOG2_MAXGROUPS];
    atomic<size_t> num_pending;
    size_t pending_soft_limit;  // try to stall traffic above this threshold
  };

  template <typename T, unsigned CHUNK_SIZE>
  class ChunkedRecycler {
  public:
    ChunkedRecycler();
    ~ChunkedRecycler();

    T *alloc_obj();
    void free_obj(T *obj);

  protected:
    // we'll store a pointer back to the chunk header just past each object
    // each object needs enough space for a pointer back to the chunk header
    struct Chunk;
    struct WithPtr {
      typedef char Storage_unaligned[sizeof(T)];
      REALM_ALIGNED_TYPE_SAMEAS(Storage_aligned, Storage_unaligned, T);
      union {
	Storage_aligned raw_storage;
	uintptr_t nextptr;
      };
      Chunk *backptr;
    };
    struct Chunk {
      atomic<unsigned> remaining_count;
      Chunk *next_chunk;
      WithPtr elements[CHUNK_SIZE];
    };

    Mutex mutex;
    uintptr_t free_head;
    uintptr_t *free_tail;
    Chunk *chunks_head;
    atomic<size_t> cur_alloc;
    size_t cur_capacity, max_alloc;
  };

  struct PendingPutHeader;

  struct PreparedMessage {
  protected:
    // should not be directly allocated
    template <typename T, unsigned CHUNK_SIZE>
    friend class ChunkedRecycler;
    PreparedMessage() {}
    ~PreparedMessage() {}

  public:
    enum Strategy {
      STRAT_UNKNOWN,
      STRAT_SHORT_IMMEDIATE, // AM short, attempt immediate
      STRAT_SHORT_PBUF,      // AM short, deferred in pktbuf
      STRAT_MEDIUM_IMMEDIATE, // AM medium, attempt immediate
      STRAT_MEDIUM_PBUF,     // AM medium, header/data in pktbuf
      STRAT_MEDIUM_MALLOCSRC, // AM medium, malloc'd temp source (HACK)
      STRAT_MEDIUM_PREP,     // AM medium, using NPAM
      STRAT_LONG_IMMEDIATE,
      STRAT_LONG_PBUF,
      STRAT_RGET_IMMEDIATE,
      STRAT_RGET_PBUF,
      STRAT_PUT_IMMEDIATE,
      STRAT_PUT_PBUF,
    };
    Strategy strategy;
    gex_Rank_t target;
    gex_EP_Index_t source_ep_index, target_ep_index;
    unsigned short msgid;
    uintptr_t dest_payload_addr;
    void *temp_buffer;
    OutbufMetadata *databuf;
    OutbufMetadata *pktbuf;
    int pktidx;
    gex_AM_SrcDesc_t srcdesc;
    PendingPutHeader *put;
  };

  // we'll keep separate transmit queues for each src/dst pair
  //  (i.e src_ep_index, dst_rank, dst_ep_index), hopefully avoiding any
  //  head of line blocking if networking resources are exhausted on a given
  //  path
  class XmitSrcDestPair {
  public:
    XmitSrcDestPair(GASNetEXInternal *_internal,
		    gex_EP_Index_t _src_ep_index,
		    gex_Rank_t _tgt_rank, gex_EP_Index_t _tgt_ep_index);
    ~XmitSrcDestPair();

    // indicates whether any packets are pending, including those that
    //  have not been committed yet - useful for quiescence detection but
    //  also for ensuring new packets don't jump the queue
    bool has_packets_queued() const;

    // a packet that is sent immediately is counted as if it was
    //  reserved and then sent in rapid succession
    void record_immediate_packet();

    // reserves space in a pbuf for an outbound packet (allocating and
    //  enqueuing a pbuf if needed)
    bool reserve_pbuf_inline(size_t hdr_bytes, size_t payload_bytes,
			     bool overflow_ok,
			     OutbufMetadata *&pktbuf, int& pktidx,
			     void *&hdr_base, void *&payload_base);
    bool reserve_pbuf_long_rget(size_t hdr_bytes,
				bool overflow_ok,
				OutbufMetadata *&pktbuf, int& pktidx,
				void *&hdr_base);
    bool reserve_pbuf_put(bool overflow_ok,
                          OutbufMetadata *&pktbuf, int& pktidx);
    void commit_pbuf_inline(OutbufMetadata *pktbuf, int pktidx,
			    const void *hdr_base,
			    gex_AM_Arg_t arg0, size_t act_payload_bytes);
    void commit_pbuf_long(OutbufMetadata *pktbuf, int pktidx,
			  const void *hdr_base,
			  gex_AM_Arg_t arg0,
			  const void *payload_base, size_t payload_bytes,
			  uintptr_t dest_addr,
			  OutbufMetadata *databuf);
    void commit_pbuf_rget(OutbufMetadata *pktbuf, int pktidx,
			  const void *hdr_base,
			  gex_AM_Arg_t arg0,
			  const void *payload_base, size_t payload_bytes,
			  uintptr_t dest_addr,
			  gex_EP_Index_t src_ep_index,
			  gex_EP_Index_t tgt_ep_index);
    void commit_pbuf_put(OutbufMetadata *pktbuf, int pktidx,
                         PendingPutHeader *put,
                         const void *payload_base, size_t payload_bytes,
                         uintptr_t dest_addr);
    void cancel_pbuf(OutbufMetadata *pktbuf, int pktidx);

    void enqueue_completion_reply(gex_AM_Arg_t comp_info);

    void enqueue_put_header(PendingPutHeader *put);

    // adds the xpair to the injector ready list or the poller critical
    //  pair list as appropriate, eventually resulting in a call to
    //  push_packets - MUST NOT be called until the push_packets
    //  that resulted from the previous enqueue is done-ish (i.e. not going
    //  to push anything else)
    void request_push(bool force_critical);

    void push_packets(bool immediate_mode, TimeLimit work_until);

    long long time_since_failure() const;

    IntrusiveListLink<XmitSrcDestPair> xpair_list_link;
    REALM_PMTA_DEFN(XmitSrcDestPair,IntrusiveListLink<XmitSrcDestPair>,xpair_list_link);
    typedef IntrusiveList<XmitSrcDestPair, REALM_PMTA_USE(XmitSrcDestPair,xpair_list_link), DummyLock> XmitPairList;

    struct LongRgetData {
      const void *payload_base;
      size_t payload_bytes;
      uintptr_t dest_addr;
      union {
	struct {
	  OutbufMetadata *databuf;
	} l;
	struct {
	  // rget needs to give both src and target ep index for data
	  gex_EP_Index_t src_ep_index, tgt_ep_index;
	} r;
      };
    };

    struct PutMetadata {
      const void *src_addr;
      uintptr_t dest_addr;
      size_t payload_bytes;
      PendingPutHeader *put;
    };

  protected:
    friend class GASNetEXInternal;

    bool reserve_pbuf_helper(size_t total_bytes, bool overflow_ok,
			     OutbufMetadata *&pktbuf, int& pktidx,
			     uintptr_t& baseptr);
    bool commit_pbuf_helper(OutbufMetadata *pktbuf, int pktidx,
			    const void *hdr_base, uintptr_t& baseptr);

    GASNetEXInternal *internal;
    gex_EP_Index_t src_ep_index;
    gex_Rank_t tgt_rank;
    gex_EP_Index_t tgt_ep_index;
    atomic<size_t> packets_reserved, packets_sent;
    Mutex mutex;
    // we don't hold the mutex while pushing packets, but we need definitely
    //  don't want multiple threads trying to push for the same src/dst pair
    MutexChecker push_mutex_check;
    atomic<OutbufMetadata *> first_pbuf;  // read without mutex
    OutbufMetadata *cur_pbuf;
    atomic<unsigned> imm_fail_count;
    bool has_ready_packets;
    long long first_fail_time;
    atomic<PendingPutHeader *> put_head;  // read without mutex
    atomic<PendingPutHeader *> *put_tailp;
    // circular queue of pending completion replys
    gex_AM_Arg_t *comp_reply_data;
    unsigned comp_reply_wrptr, comp_reply_rdptr;
    atomic<unsigned> comp_reply_count;  // read without mutex
    unsigned comp_reply_capacity;
    // TODO: track packets in flight to avoid clogging?
  };

  class XmitSrc {
  public:
    XmitSrc(GASNetEXInternal *_internal, gex_EP_Index_t _src_ep_index);
    ~XmitSrc();

    XmitSrcDestPair *lookup_pair(gex_Rank_t tgt_rank,
				 gex_EP_Index_t tgt_ep_index);

  protected:
    friend class GASNetEXInternal;

    GASNetEXInternal *internal;
    gex_EP_Index_t src_ep_index;

    // we'll allocate XmitSrcDestPair's on demand - atomics allow nonblocking
    //  lookup
    atomic<XmitSrcDestPair *> *pairs;

    // TODO: track in flight work at src level?
  };

  struct PendingReverseGet;

  class GASNetEXEvent {
  protected:
    // should not be directly allocated
    template <typename T, unsigned CHUNK_SIZE>
    friend class ChunkedRecycler;
    GASNetEXEvent();
    ~GASNetEXEvent() {}

  public:
    gex_Event_t get_event() const;

    GASNetEXEvent& set_event(gex_Event_t _event);
    GASNetEXEvent& set_local_comp(PendingCompletion *_local_comp);
    GASNetEXEvent& set_pktbuf(OutbufMetadata *_pktbuf);
    GASNetEXEvent& set_databuf(OutbufMetadata *_databuf);
    GASNetEXEvent& set_rget(PendingReverseGet *_rget);
    GASNetEXEvent& set_put(PendingPutHeader *_put);
    GASNetEXEvent& set_leaf(GASNetEXEvent *_leaf);

    void propagate_to_leaves();

    void trigger(GASNetEXInternal *internal);

    IntrusiveListLink<GASNetEXEvent> event_list_link;
    REALM_PMTA_DEFN(GASNetEXEvent,IntrusiveListLink<GASNetEXEvent>,event_list_link);
    typedef IntrusiveList<GASNetEXEvent, REALM_PMTA_USE(GASNetEXEvent,event_list_link), DummyLock> EventList;

  protected:
    gex_Event_t event;
    PendingCompletion *local_comp;
    OutbufMetadata *pktbuf;
    OutbufMetadata *databuf;
    PendingReverseGet *rget;
    PendingPutHeader *put;
    GASNetEXEvent *leaf;
  };

  // an injector tries to send packets, but is not allowed to stall -
  //  all requests must use GEX_FLAG_IMMEDIATE and any failure results in
  //  handing the xpair off to the poller (which is allowed to block eventually)
  class GASNetEXInjector : public BackgroundWorkItem {
  public:
    GASNetEXInjector(GASNetEXInternal *_internal);

    void add_ready_xpair(XmitSrcDestPair *xpair);

    bool has_work_remaining();

    virtual bool do_work(TimeLimit work_until);

  protected:
    GASNetEXInternal *internal;
    Mutex mutex;
    XmitSrcDestPair::XmitPairList ready_xpairs;
  };

  class GASNetEXPoller : public BackgroundWorkItem {
  public:
    GASNetEXPoller(GASNetEXInternal *_internal);

    void begin_polling();
    void end_polling();

    void add_critical_xpair(XmitSrcDestPair *xpair);

    void add_pending_event(GASNetEXEvent *event);

    bool has_work_remaining();

    virtual bool do_work(TimeLimit work_until);

    // causes calling thread to wait for a full call to gasnet_AMPoll() to
    //  be performed by the poller
    void wait_for_full_poll_cycle();

  protected:
    GASNetEXInternal *internal;
    Mutex mutex;
    atomic<bool> shutdown_flag;  // set/cleared inside mutex, but tested outside
    Mutex::CondVar shutdown_cond;
    atomic<bool> pollwait_flag;  // set/cleared inside mutex, but tested outside
    Mutex::CondVar pollwait_cond;
    XmitSrcDestPair::XmitPairList critical_xpairs;
    GASNetEXEvent::EventList pending_events;
  };

  class GASNetEXCompleter : public BackgroundWorkItem {
  public:
    GASNetEXCompleter(GASNetEXInternal *_internal);

    void add_ready_events(GASNetEXEvent::EventList& newly_ready);

    bool has_work_remaining();

    virtual bool do_work(TimeLimit work_until);

  protected:
    GASNetEXInternal *internal;
    Mutex mutex;
    atomic<bool> has_work;  // can be read without mutex
    GASNetEXEvent::EventList ready_events;
  };

  struct PendingPutHeader {
  protected:
    // should not be directly allocated
    template <typename T, unsigned CHUNK_SIZE>
    friend class ChunkedRecycler;
    PendingPutHeader() {}
    ~PendingPutHeader() {}

  public:
    gex_Rank_t target;
    gex_EP_Index_t src_ep_index, tgt_ep_index;
    gex_AM_Arg_t arg0;
    static const size_t MAX_HDR_SIZE = 128;
    size_t hdr_size;
    unsigned char hdr_data[MAX_HDR_SIZE];
    uintptr_t src_ptr, tgt_ptr;
    size_t payload_bytes;
    PendingCompletion *local_comp;
    XmitSrcDestPair *xpair;
    atomic<PendingPutHeader *> next_put;
  };

  // NOTE: ReverseGetter exists for now because we have to use RMAs instead
  //  of long AMs for transfers between segments bound to non-primordial
  //  endpoints (on the source and/or the target) - once GASNet supports
  //  AMs on all endpoints, this code should probably go away
  class ReverseGetter;

  struct PendingReverseGet {
  protected:
    // should not be directly allocated
    template <typename T, unsigned CHUNK_SIZE>
    friend class ChunkedRecycler;
    PendingReverseGet() {}
    ~PendingReverseGet() {}

  public:
    ReverseGetter *rgetter;
    PendingReverseGet *next_rget;
    gex_Rank_t srcrank;
    gex_EP_Index_t src_ep_index, tgt_ep_index;
    gex_AM_Arg_t arg0;
    static const size_t MAX_HDR_SIZE = 128;
    size_t hdr_size;
    unsigned char hdr_data[MAX_HDR_SIZE];
    uintptr_t src_ptr, tgt_ptr;
    size_t payload_bytes;
  };

  class ReverseGetter : public BackgroundWorkItem {
  public:
    ReverseGetter(GASNetEXInternal *_internal);

    void add_reverse_get(gex_Rank_t srcrank, gex_EP_Index_t src_ep_index,
			 gex_EP_Index_t tgt_ep_index,
			 gex_AM_Arg_t arg0,
			 const void *hdr, size_t hdr_bytes,
			 uintptr_t src_ptr, uintptr_t tgt_ptr,
			 size_t payload_bytes);

    bool has_work_remaining();

    virtual bool do_work(TimeLimit work_until);

  protected:
    friend class GASNetEXEvent;
    void reverse_get_complete(PendingReverseGet *rget);

    GASNetEXInternal *internal;
    Mutex mutex;
    PendingReverseGet *head;
    PendingReverseGet **tailp;
    ChunkedRecycler<PendingReverseGet, 8> rget_alloc;
  };

  class GASNetEXInternal {
  public:
    GASNetEXInternal(GASNetEXModule *_module, RuntimeImpl *_runtime);

    ~GASNetEXInternal();

    void init(int *argc, const char ***argv);
    uintptr_t attach(size_t size);

    bool attempt_binding(void *base, size_t size,
			 NetworkSegmentInfo::MemoryType memtype,
			 NetworkSegmentInfo::MemoryTypeExtraData memextra,
			 gex_EP_Index_t *ep_indexp);
    void publish_bindings();

    void detach();

    void barrier();
    void broadcast(gex_Rank_t root, const void *val_in, void *val_out, size_t bytes);
    void gather(gex_Rank_t root, const void *val_in, void *vals_out, size_t bytes);

    size_t sample_messages_received_count();
    bool check_for_quiescence(size_t sampled_receive_count);

    PendingCompletion *get_available_comp();
    PendingCompletion *early_local_completion(PendingCompletion *comp);

    size_t recommended_max_payload(gex_Rank_t target,
				   gex_EP_Index_t target_ep_index,
				   bool with_congestion,
				   size_t header_size,
				   uintptr_t dest_payload_addr);
    size_t recommended_max_payload(gex_Rank_t target,
				   gex_EP_Index_t target_ep_index,
				   const void *data, size_t bytes_per_line,
				   size_t lines, size_t line_stride,
				   bool with_congestion,
				   size_t header_size,
				   uintptr_t dest_payload_addr);
    size_t recommended_max_payload(bool with_congestion,
				   size_t header_size);

    PreparedMessage *prepare_message(gex_Rank_t target, gex_EP_Index_t target_ep_index,
				     unsigned short msgid,
				     void *&header_base, size_t header_size,
				     void *&payload_base, size_t payload_size,
				     uintptr_t dest_payload_addr);
    void commit_message(PreparedMessage *msg,
			PendingCompletion *comp,
			void *header_base, size_t header_size,
			void *payload_base, size_t payload_size);
    void cancel_message(PreparedMessage *msg);

    gex_AM_Arg_t handle_short(gex_Rank_t srcrank, gex_AM_Arg_t arg0,
			      const void *hdr, size_t hdr_bytes);
    gex_AM_Arg_t handle_medium(gex_Rank_t srcrank, gex_AM_Arg_t arg0,
			       const void *hdr, size_t hdr_bytes,
			       const void *data, size_t data_bytes);
    gex_AM_Arg_t handle_long(gex_Rank_t srcrank, gex_AM_Arg_t arg0,
			     const void *hdr, size_t hdr_bytes,
			     const void *data, size_t data_bytes);
    void handle_reverse_get(gex_Rank_t srcrank, gex_EP_Index_t src_ep_index,
			    gex_EP_Index_t tgt_ep_index,
			    gex_AM_Arg_t arg0,
			    const void *hdr, size_t hdr_bytes,
			    uintptr_t src_ptr, uintptr_t tgt_ptr,
			    size_t payload_bytes);
    size_t handle_batch(gex_Rank_t srcrank, gex_AM_Arg_t arg0,
                        gex_AM_Arg_t cksum,
			const void *data, size_t data_bytes,
			gex_AM_Arg_t *comps);
    void handle_completion_reply(gex_Rank_t srcrank,
				 const gex_AM_Arg_t *args, size_t nargs);

  protected:
    friend class ReverseGetter;
    friend class XmitSrc;
    friend class XmitSrcDestPair;
    friend class GASNetEXEvent;
    friend class GASNetEXPoller;
    friend class GASNetEXCompleter;

    // callbacks from IncomingMessageManager
    static void short_message_complete(NodeID sender, uintptr_t objptr,
				       uintptr_t comp_info);
    static void medium_message_complete(NodeID sender, uintptr_t objptr,
					uintptr_t comp_info);
    static void long_message_complete(NodeID sender, uintptr_t objptr,
				      uintptr_t comp_info);

    PendingCompletion *extract_arg0_local_comp(gex_AM_Arg_t& arg0);

    GASNetEXModule *module;
    RuntimeImpl *runtime;
    gex_Client_t client;
    // order in 'eps' should match GASNet's indexing
    std::vector<gex_EP_t> eps;
    gex_TM_t prim_tm;
    gex_Rank_t prim_rank, prim_size;
    gex_Segment_t prim_segment;
    size_t prim_segsize;

    struct SegmentInfo {
      uintptr_t base, limit;
      gex_EP_Index_t ep_index;
      gex_Segment_t segment;
      NetworkSegmentInfo::MemoryType memtype;
      NetworkSegmentInfo::MemoryTypeExtraData memextra;
    };
    struct SegmentInfoSorter;
    // this list is sorted by address to enable quick address lookup
    std::vector<SegmentInfo> segments_by_addr;

    const SegmentInfo *find_segment(const void *srcptr) const;

    std::vector<XmitSrc *> xmitsrcs;

    GASNetEXPoller poller;
    GASNetEXInjector injector;
    GASNetEXCompleter completer;
    ReverseGetter rgetter;
    PendingCompletionManager compmgr;
    OutbufManager obmgr;

    // TODO: split counter into per-thread values to avoid contention?
    atomic<uint64_t> total_packets_received;

    // manage a single open databuf for all endpoints
    Mutex databuf_mutex;
    OutbufMetadata *databuf_md;

    // allocator/managers for various objects we want to reuse
    ChunkedRecycler<GASNetEXEvent, 64> event_alloc;
    ChunkedRecycler<PreparedMessage, 32> prep_alloc;
    ChunkedRecycler<PendingPutHeader, 32> put_alloc;

    uintptr_t databuf_reserve(size_t bytes_needed, OutbufMetadata **mdptr);
  };

}; // namespace Realm

#endif

