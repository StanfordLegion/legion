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

// internal messaging subsystem using GASNet-1

#ifndef GASNETMSG_H
#define GASNETMSG_H

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>

#include <sys/types.h>

#include "realm/realm_config.h"
#include "realm/mutex.h"
#include "realm/activemsg.h"
#include "realm/bgwork.h"
#include "realm/atomics.h"

    enum ActiveMessageIDs {
      FIRST_AVAILABLE = 140,
      NODE_ANNOUNCE_MSGID,
      SPAWN_TASK_MSGID,
      LOCK_REQUEST_MSGID,
      LOCK_RELEASE_MSGID,
      LOCK_GRANT_MSGID,
      EVENT_SUBSCRIBE_MSGID,
      EVENT_TRIGGER_MSGID,
      EVENT_UPDATE_MSGID,
      REMOTE_MALLOC_MSGID,
      REMOTE_MALLOC_RPLID,
      CREATE_ALLOC_MSGID,
      CREATE_ALLOC_RPLID,
      CREATE_INST_MSGID,
      CREATE_INST_RPLID,
      VALID_MASK_REQ_MSGID,
      VALID_MASK_DATA_MSGID,
      VALID_MASK_FTH_MSGID,
      ROLL_UP_TIMER_MSGID,
      ROLL_UP_DATA_MSGID,
      CLEAR_TIMER_MSGID,
      DESTROY_INST_MSGID,
      REMOTE_WRITE_MSGID,
      REMOTE_REDUCE_MSGID,
      REMOTE_SERDEZ_MSGID,
      REMOTE_WRITE_FENCE_MSGID,
      REMOTE_WRITE_FENCE_ACK_MSGID,
      DESTROY_LOCK_MSGID,
      REMOTE_REDLIST_MSGID,
      MACHINE_SHUTDOWN_MSGID,
      BARRIER_ADJUST_MSGID,
      BARRIER_SUBSCRIBE_MSGID,
      BARRIER_TRIGGER_MSGID,
      BARRIER_MIGRATE_MSGID,
      METADATA_REQUEST_MSGID,
      METADATA_RESPONSE_MSGID, // should really be a reply
      METADATA_INVALIDATE_MSGID,
      METADATA_INVALIDATE_ACK_MSGID,
      XFERDES_REMOTEWRITE_MSGID,
      XFERDES_REMOTEWRITE_ACK_MSGID,
      XFERDES_CREATE_MSGID,
      XFERDES_DESTROY_MSGID,
      XFERDES_NOTIFY_COMPLETION_MSGID,
      XFERDES_UPDATE_BYTES_WRITE_MSGID,
      XFERDES_UPDATE_BYTES_READ_MSGID,
      REGISTER_TASK_MSGID,
      REGISTER_TASK_COMPLETE_MSGID,
      REMOTE_MICROOP_MSGID,
      REMOTE_MICROOP_COMPLETE_MSGID,
      REMOTE_SPARSITY_CONTRIB_MSGID,
      REMOTE_SPARSITY_REQUEST_MSGID,
      APPROX_IMAGE_RESPONSE_MSGID,
      SET_CONTRIB_COUNT_MSGID,
      REMOTE_ID_REQUEST_MSGID,
      REMOTE_ID_RESPONSE_MSGID,
      REMOTE_IB_ALLOC_REQUEST_MSGID,
      REMOTE_IB_ALLOC_RESPONSE_MSGID,
      REMOTE_IB_FREE_REQUEST_MSGID,
      REMOTE_COPY_MSGID,
      REMOTE_FILL_MSGID,
      MEM_STORAGE_ALLOC_REQ_MSGID,
      MEM_STORAGE_ALLOC_RESP_MSGID,
      MEM_STORAGE_RELEASE_REQ_MSGID,
      MEM_STORAGE_RELEASE_RESP_MSGID,
      CANCEL_OPERATION_MSGID,
    };



namespace Realm {
  class CoreReservationSet;
};

// for uint64_t, int32_t
#include <stdint.h>

#include <vector>

// these need to be consistent with gasnet.h - fragile for now
typedef int32_t handlerarg_t;
typedef void *token_t;

// gasnet_hsl_t in object form for templating goodness
extern void gasnet_parse_command_line(std::vector<std::string>& cmdline);
extern void init_endpoints(size_t gasnet_mem_size,
			   size_t registered_mem_size,
			   size_t registered_ib_mem_size,
			   Realm::CoreReservationSet& crs,
			   int num_polling_threads,
			   Realm::BackgroundWorkManager& bgwork,
			   bool poll_use_bgwork,
			   Realm::IncomingMessageManager *message_manager);
extern void start_polling_threads(void);
extern void start_handler_threads(size_t stacksize);
extern void flush_activemsg_channels(void);
extern void stop_activemsg_threads(void);
extern void report_activemsg_status(FILE *f);

// returns the largest payload that can be sent to a node (to a non-pinned
//   address)
extern size_t get_lmb_size(Realm::NodeID target_node);

/* Necessary base structure for all medium and long active messages */
struct BaseMedium {
  static const handlerarg_t FRAG_INFO_MAGIC = 0x0bad0bad;
  static const handlerarg_t COMP_INFO_MAGIC = 0x0a550a55;
  void set_magic(void) {
    frag_info = FRAG_INFO_MAGIC;
    comp_info = COMP_INFO_MAGIC;
  }
  handlerarg_t frag_info;
  handlerarg_t comp_info;
  handlerarg_t pad1, pad2;
};

struct BaseReply {
  void *srcptr;
};

extern void release_srcptr(void *ptr);

class PayloadSource {
public:
  PayloadSource(void) { }
  virtual ~PayloadSource(void) { }
public:
  virtual void copy_data(void *dest) = 0;
  virtual void *get_contig_pointer(void) { assert(false); return 0; }
  virtual int get_payload_mode(void) { return Realm::PAYLOAD_KEEP; }
};

class ContiguousPayload : public PayloadSource {
public:
  ContiguousPayload(void *_srcptr, size_t _size, int _mode);
  virtual ~ContiguousPayload(void) { }
  virtual void copy_data(void *dest);
  virtual void *get_contig_pointer(void) { return srcptr; }
  virtual int get_payload_mode(void) { return mode; }
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

typedef std::pair<const void *, size_t> SpanListEntry;
typedef std::vector<SpanListEntry> SpanList;

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

struct PendingCompletion;

extern void enqueue_message(Realm::NodeID target, int msgid,
			    const void *args, size_t arg_size,
			    const void *payload, size_t payload_size,
			    int payload_mode, PendingCompletion *comp,
			    void *dstptr = 0);

extern void enqueue_message(Realm::NodeID target, int msgid,
			    const void *args, size_t arg_size,
			    const void *payload, size_t line_size,
			    off_t line_stride, size_t line_count,
			    int payload_mode, PendingCompletion *comp,
			    void *dstptr = 0);

extern void enqueue_message(Realm::NodeID target, int msgid,
			    const void *args, size_t arg_size,
			    const SpanList& spans, size_t payload_size,
			    int payload_mode, PendingCompletion *comp,
			    void *dstptr = 0);

//extern void enqueue_incoming(Realm::NodeID sender, Realm::IncomingMessage *msg);

extern void handle_long_msgptr(Realm::NodeID source, int msgptr_index);
//extern size_t adjust_long_msgsize(gasnet_node_t source, void *ptr, size_t orig_size);
extern bool adjust_long_msgsize(Realm::NodeID source, void *&ptr, size_t &buffer_size,
				int frag_info);
extern void record_message(Realm::NodeID source, bool sent_reply);

#ifdef REALM_PROFILE_AM_HANDLERS
// have to define this two different ways because we can't put ifdefs in the macros below
extern void record_activemsg_profiling(int msgid,
				       const struct timespec& ts_start,
				       const struct timespec& ts_end);

template <int MSGID>
class ActiveMsgProfilingHelper {
 public:
  ActiveMsgProfilingHelper(void) 
  {
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
  }

  ~ActiveMsgProfilingHelper(void)
  {
    struct timespec ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    record_activemsg_profiling(MSGID, ts_start, ts_end);
  }

 protected:
  struct timespec ts_start;
};

#else
// not measuring AM handler times, so need a dummy class
template <int MSGID>
class ActiveMsgProfilingHelper {
 public:
  // have to define a constructor or the uses of this below will be 
  //  reported as unused variables...
  ActiveMsgProfilingHelper(void) {}
};
#endif

template <class MSGTYPE>
void dummy_short_handler(MSGTYPE dummy) {}

template <class MSGTYPE>
void dummy_medium_handler(MSGTYPE dummy, const void *data, size_t datalen) {}

template <class MSGTYPE, int MSGID,
          void (*SHORT_HNDL_PTR)(MSGTYPE),
          void (*MED_HNDL_PTR)(MSGTYPE, const void *, size_t),
          int MSG_N>
struct MessageRawArgs;

#if 0
template <class MSGTYPE, int MSGID, void (*SHORT_HNDL_PTR)(MSGTYPE), int MSG_N>
class IncomingShortMessage : public Realm::IncomingMessage {
 public:
  IncomingShortMessage(int _sender) 
    : sender(_sender) 
  {}

  virtual void run_handler(void)
  {
    ActiveMsgProfilingHelper<MSGID> amph;
    (*SHORT_HNDL_PTR)(u.typed);
  }

  virtual int get_peer(void) { return sender; }
  virtual int get_msgid(void) { return MSGID; }
  virtual size_t get_msgsize(void) { return sizeof(MSGTYPE); }

  int sender;
  union { 
    MessageRawArgs<MSGTYPE,MSGID,SHORT_HNDL_PTR,dummy_medium_handler<MSGTYPE>,MSG_N> raw;
    MSGTYPE typed;
  } u;
};

template <class MSGTYPE, int MSGID, 
          void (*MED_HNDL_PTR)(MSGTYPE, const void *, size_t), int MSG_N>
class IncomingMediumMessage : public Realm::IncomingMessage {
 public:
  IncomingMediumMessage(int _sender, const void *_msgdata, size_t _msglen)
    : sender(_sender), msgdata(_msgdata), msglen(_msglen)
  {}

  virtual void run_handler(void)
  {
    {
      ActiveMsgProfilingHelper<MSGID> amph;
      (*MED_HNDL_PTR)(u.typed, msgdata, msglen);
    }
    handle_long_msgptr(sender, msgdata);
  }

  virtual int get_peer(void) { return sender; }
  virtual int get_msgid(void) { return MSGID; }
  virtual size_t get_msgsize(void) { return sizeof(MSGTYPE) + msglen; }

  int sender;
  const void *msgdata;
  size_t msglen;
  union { 
    MessageRawArgs<MSGTYPE,MSGID,dummy_short_handler<MSGTYPE>,MED_HNDL_PTR,MSG_N> raw;
    MSGTYPE typed;
  } u;
};
#endif

#define HANDLERARG_DECL_1                     handlerarg_t arg0
#define HANDLERARG_DECL_2  HANDLERARG_DECL_1; handlerarg_t arg1
#define HANDLERARG_DECL_3  HANDLERARG_DECL_2; handlerarg_t arg2
#define HANDLERARG_DECL_4  HANDLERARG_DECL_3; handlerarg_t arg3
#define HANDLERARG_DECL_5  HANDLERARG_DECL_4; handlerarg_t arg4
#define HANDLERARG_DECL_6  HANDLERARG_DECL_5; handlerarg_t arg5
#define HANDLERARG_DECL_7  HANDLERARG_DECL_6; handlerarg_t arg6
#define HANDLERARG_DECL_8  HANDLERARG_DECL_7; handlerarg_t arg7
#define HANDLERARG_DECL_9  HANDLERARG_DECL_8; handlerarg_t arg8
#define HANDLERARG_DECL_10  HANDLERARG_DECL_9; handlerarg_t arg9
#define HANDLERARG_DECL_11  HANDLERARG_DECL_10; handlerarg_t arg10
#define HANDLERARG_DECL_12  HANDLERARG_DECL_11; handlerarg_t arg11
#define HANDLERARG_DECL_13  HANDLERARG_DECL_12; handlerarg_t arg12
#define HANDLERARG_DECL_14  HANDLERARG_DECL_13; handlerarg_t arg13
#define HANDLERARG_DECL_15  HANDLERARG_DECL_14; handlerarg_t arg14
#define HANDLERARG_DECL_16  HANDLERARG_DECL_15; handlerarg_t arg15

#define HANDLERARG_VALS_1                     arg0
#define HANDLERARG_VALS_2  HANDLERARG_VALS_1, arg1
#define HANDLERARG_VALS_3  HANDLERARG_VALS_2, arg2
#define HANDLERARG_VALS_4  HANDLERARG_VALS_3, arg3
#define HANDLERARG_VALS_5  HANDLERARG_VALS_4, arg4
#define HANDLERARG_VALS_6  HANDLERARG_VALS_5, arg5
#define HANDLERARG_VALS_7  HANDLERARG_VALS_6, arg6
#define HANDLERARG_VALS_8  HANDLERARG_VALS_7, arg7
#define HANDLERARG_VALS_9  HANDLERARG_VALS_8, arg8
#define HANDLERARG_VALS_10  HANDLERARG_VALS_9, arg9
#define HANDLERARG_VALS_11  HANDLERARG_VALS_10, arg10
#define HANDLERARG_VALS_12  HANDLERARG_VALS_11, arg11
#define HANDLERARG_VALS_13  HANDLERARG_VALS_12, arg12
#define HANDLERARG_VALS_14  HANDLERARG_VALS_13, arg13
#define HANDLERARG_VALS_15  HANDLERARG_VALS_14, arg14
#define HANDLERARG_VALS_16  HANDLERARG_VALS_15, arg15

#define HANDLERARG_PARAMS_1                      handlerarg_t arg0
#define HANDLERARG_PARAMS_2 HANDLERARG_PARAMS_1, handlerarg_t arg1
#define HANDLERARG_PARAMS_3 HANDLERARG_PARAMS_2, handlerarg_t arg2
#define HANDLERARG_PARAMS_4 HANDLERARG_PARAMS_3, handlerarg_t arg3
#define HANDLERARG_PARAMS_5 HANDLERARG_PARAMS_4, handlerarg_t arg4
#define HANDLERARG_PARAMS_6 HANDLERARG_PARAMS_5, handlerarg_t arg5
#define HANDLERARG_PARAMS_7 HANDLERARG_PARAMS_6, handlerarg_t arg6
#define HANDLERARG_PARAMS_8 HANDLERARG_PARAMS_7, handlerarg_t arg7
#define HANDLERARG_PARAMS_9 HANDLERARG_PARAMS_8, handlerarg_t arg8
#define HANDLERARG_PARAMS_10 HANDLERARG_PARAMS_9, handlerarg_t arg9
#define HANDLERARG_PARAMS_11 HANDLERARG_PARAMS_10, handlerarg_t arg10
#define HANDLERARG_PARAMS_12 HANDLERARG_PARAMS_11, handlerarg_t arg11
#define HANDLERARG_PARAMS_13 HANDLERARG_PARAMS_12, handlerarg_t arg12
#define HANDLERARG_PARAMS_14 HANDLERARG_PARAMS_13, handlerarg_t arg13
#define HANDLERARG_PARAMS_15 HANDLERARG_PARAMS_14, handlerarg_t arg14
#define HANDLERARG_PARAMS_16 HANDLERARG_PARAMS_15, handlerarg_t arg15

#define HANDLERARG_COPY_1(u)                    (u).raw.arg0 = arg0
#define HANDLERARG_COPY_2(u) HANDLERARG_COPY_1(u); (u).raw.arg1 = arg1
#define HANDLERARG_COPY_3(u) HANDLERARG_COPY_2(u); (u).raw.arg2 = arg2
#define HANDLERARG_COPY_4(u) HANDLERARG_COPY_3(u); (u).raw.arg3 = arg3
#define HANDLERARG_COPY_5(u) HANDLERARG_COPY_4(u); (u).raw.arg4 = arg4
#define HANDLERARG_COPY_6(u) HANDLERARG_COPY_5(u); (u).raw.arg5 = arg5
#define HANDLERARG_COPY_7(u) HANDLERARG_COPY_6(u); (u).raw.arg6 = arg6
#define HANDLERARG_COPY_8(u) HANDLERARG_COPY_7(u); (u).raw.arg7 = arg7
#define HANDLERARG_COPY_9(u) HANDLERARG_COPY_8(u); (u).raw.arg8 = arg8
#define HANDLERARG_COPY_10(u) HANDLERARG_COPY_9(u); (u).raw.arg9 = arg9
#define HANDLERARG_COPY_11(u) HANDLERARG_COPY_10(u); (u).raw.arg10 = arg10
#define HANDLERARG_COPY_12(u) HANDLERARG_COPY_11(u); (u).raw.arg11 = arg11
#define HANDLERARG_COPY_13(u) HANDLERARG_COPY_12(u); (u).raw.arg12 = arg12
#define HANDLERARG_COPY_14(u) HANDLERARG_COPY_13(u); (u).raw.arg13 = arg13
#define HANDLERARG_COPY_15(u) HANDLERARG_COPY_14(u); (u).raw.arg14 = arg14
#define HANDLERARG_COPY_16(u) HANDLERARG_COPY_15(u); (u).raw.arg15 = arg15

#define MACROPROXY(a,...) a(__VA_ARGS__)

extern Realm::NodeID get_message_source(token_t token);
extern void send_srcptr_release(token_t token, uint64_t srcptr);

#if 0
#define SPECIALIZED_RAW_ARGS(n) \
template <class MSGTYPE, int MSGID, \
          void (*SHORT_HNDL_PTR)(MSGTYPE), \
          void (*MED_HNDL_PTR)(MSGTYPE, const void *, size_t)> \
struct MessageRawArgs<MSGTYPE, MSGID, SHORT_HNDL_PTR, MED_HNDL_PTR, n> { \
  HANDLERARG_DECL_ ## n ; \
\
  typedef IncomingShortMessage<MSGTYPE,MSGID,SHORT_HNDL_PTR,n> ISHORT; \
  typedef IncomingMediumMessage<MSGTYPE,MSGID,MED_HNDL_PTR,n> IMED; \
  static void handler_short(token_t token, HANDLERARG_PARAMS_ ## n ) \
  { \
    Realm::NodeID src = get_message_source(token);	\
    /*printf("handling message from node %d (id=%d)\n", src, MSGID);*/	\
    ISHORT *imsg = new ISHORT(src); \
    HANDLERARG_COPY_ ## n(imsg->u) ;				\
    record_message(src, false); \
    enqueue_incoming(src, imsg); \
  } \
\
  static void handler_medium(token_t token, void *buf, size_t nbytes, \
                             HANDLERARG_PARAMS_ ## n ) \
  { \
    Realm::NodeID src = get_message_source(token);	\
    /*printf("handling medium message from node %d (id=%d)\n", src, MSGID);*/ \
    bool handle_now = adjust_long_msgsize(src, buf, nbytes, arg0/*, arg1*/); \
    if(handle_now) { \
      IMED *imsg = new IMED(src, buf, nbytes);			\
      HANDLERARG_COPY_ ## n(imsg->u);				\
      /* save a copy of the srcptr - imsg may be freed any time*/ \
      /*  after we enqueue it */                                \
      uint64_t srcptr = reinterpret_cast<uintptr_t>(imsg->u.typed.srcptr); \
      enqueue_incoming(src, imsg);				\
      /* we can (and should) release the srcptr immediately */	\
      if(srcptr) {				\
	assert(nbytes > 0); \
        record_message(src, true); \
	send_srcptr_release(token, srcptr); \
      } else \
        record_message(src, false);				\
    } else \
      record_message(src, false);				\
  } \
};

// all messages are at least 8 bytes - no RAW_ARGS(1)
SPECIALIZED_RAW_ARGS(2);
SPECIALIZED_RAW_ARGS(3);
SPECIALIZED_RAW_ARGS(4);
SPECIALIZED_RAW_ARGS(5);
SPECIALIZED_RAW_ARGS(6);
SPECIALIZED_RAW_ARGS(7);
SPECIALIZED_RAW_ARGS(8);
SPECIALIZED_RAW_ARGS(9);
SPECIALIZED_RAW_ARGS(10);
SPECIALIZED_RAW_ARGS(11);
SPECIALIZED_RAW_ARGS(12);
SPECIALIZED_RAW_ARGS(13);
SPECIALIZED_RAW_ARGS(14);
SPECIALIZED_RAW_ARGS(15);
SPECIALIZED_RAW_ARGS(16);
#endif

#ifdef ACTIVE_MESSAGE_TRACE
void record_am_handler(int msgid, const char *description, bool reply = false);
#endif

template <typename T>
union AtLeastEightBytes {
  T data;
  uint64_t padding;
};


enum { MSGID_NEW_ACTIVEMSG = 251 };

class QuiescenceChecker {
public:
  QuiescenceChecker();

  size_t sample_messages_received_count(void);
  bool perform_check(size_t sampled_receive_count);

public:
  size_t last_message_count;
  Realm::Mutex mutex;
  Realm::Mutex::CondVar condvar;
  int messages_received;
  bool is_quiescent;
};

extern QuiescenceChecker quiescence_checker;

struct PendingCompletion {
  PendingCompletion();

  void *add_local_completion(size_t bytes);
  void *add_remote_completion(size_t bytes);

  // marks ready and returns true if non-empty, else resets and returns false
  bool mark_ready();

  // these two calls can be concurrent, which complicates the determination of
  //  who can free the entry
  bool invoke_local_completions();
  bool invoke_remote_completions();

  int index;
  PendingCompletion *next_free;
  // these three bits need to be in a single atomic location
  static const unsigned LOCAL_PENDING_BIT = 1;
  static const unsigned REMOTE_PENDING_BIT = 2;
  static const unsigned READY_BIT = 4;
  Realm::atomic<unsigned> state;
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

  // marks a completion ready or recycles an empty completion
  bool mark_ready(PendingCompletion *comp);

  void invoke_completions(int index, bool do_local, bool do_remote);
  
protected:
  static const size_t LOG2_MAXGROUPS = 12; // 4K groups -> 1M completions

  // protects pops from the free list (to avoid A-B-A problem), but NOT pushes
  Realm::Mutex mutex;
  Realm::atomic<PendingCompletion *> first_free;
  Realm::atomic<size_t> num_groups; // number of groups currently allocated
  Realm::atomic<PendingCompletionGroup *> groups[1 << LOG2_MAXGROUPS];
};

extern PendingCompletionManager completion_manager;

#endif

