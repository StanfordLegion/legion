/* Copyright 2017 Stanford University, NVIDIA Corporation
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

// hopefully a more user-friendly C++ template wrapper for GASNet active
//  messages...

#ifndef ACTIVEMSG_H
#define ACTIVEMSG_H

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <vector>

#include <sys/types.h>

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
      REMOTE_MALLOC_RPLID = 150,
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
      DESTROY_INST_MSGID = 160,
      REMOTE_WRITE_MSGID,
      REMOTE_REDUCE_MSGID,
      REMOTE_SERDEZ_MSGID,
      REMOTE_WRITE_FENCE_MSGID,
      REMOTE_WRITE_FENCE_ACK_MSGID,
      DESTROY_LOCK_MSGID,
      REMOTE_REDLIST_MSGID,
      MACHINE_SHUTDOWN_MSGID,
      BARRIER_ADJUST_MSGID,
      BARRIER_SUBSCRIBE_MSGID = 170,
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
      REMOTE_IB_ALLOC_REQUEST_MSGID,
      REMOTE_IB_ALLOC_RESPONSE_MSGID,
      REMOTE_IB_FREE_REQUEST_MSGID,
    };


enum { PAYLOAD_NONE, // no payload in packet
       PAYLOAD_KEEP, // use payload pointer, guaranteed to be stable
       PAYLOAD_FREE, // take ownership of payload, free when done
       PAYLOAD_COPY, // make a copy of the payload
       PAYLOAD_SRCPTR, // payload has been copied to the src data pool
       PAYLOAD_PENDING, // payload needs to be copied, but hasn't yet
       PAYLOAD_KEEPREG, // use payload pointer, AND it's registered!
       PAYLOAD_EMPTY, // message can have payload, but this one is 0 bytes
};

typedef std::pair<const void *, size_t> SpanListEntry;
typedef std::vector<SpanListEntry> SpanList;

namespace Realm {
  class CoreReservationSet;
};

// if USE_GASNET isn't defined, then we replace all the good stuff with 
//  single-node-only stubs
#ifdef USE_GASNET

// for uint64_t
#include <stdint.h>

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

#ifdef CHECK_REENTRANT_MESSAGES
GASNETT_THREADKEY_DECLARE(in_handler);
#endif

// eliminate GASNet warnings for unused static functions
static const void *ignore_gasnet_warning1 __attribute__((unused)) = (void *)_gasneti_threadkey_init;
static const void *ignore_gasnet_warning2 __attribute__((unused)) = (void *)_gasnett_trace_printf_noop;

#include <vector>

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

// gasnet_hsl_t in object form for templating goodness
class GASNetHSL {
public:
  GASNetHSL(void) { gasnet_hsl_init(&mutex); }
  ~GASNetHSL(void) { gasnet_hsl_destroy(&mutex); }

private:
  // Should never be copied
  GASNetHSL(const GASNetHSL &rhs) { assert(false); }
  GASNetHSL& operator=(const GASNetHSL &rhs) { assert(false); return *this; }

public:
  void lock(void) { gasnet_hsl_lock(&mutex); }
  void unlock(void) { gasnet_hsl_unlock(&mutex); }

protected:
  friend class GASNetCondVar;
  gasnet_hsl_t mutex;
};

class GASNetCondVar {
public:
  GASNetCondVar(GASNetHSL &_mutex) 
    : mutex(_mutex)
  {
    gasnett_cond_init(&cond);
  }

  ~GASNetCondVar(void)
  {
    gasnett_cond_destroy(&cond);
  }

  // these require that you hold the lock when you call
  void signal(void)
  {
    gasnett_cond_signal(&cond);
  }

  void broadcast(void)
  {
    gasnett_cond_broadcast(&cond);
  }

  void wait(void)
  {
    gasnett_cond_wait(&cond, &mutex.mutex.lock);
  }

  GASNetHSL &mutex;

protected:
  gasnett_cond_t cond;
};

extern void init_endpoints(gasnet_handlerentry_t *handlers, int hcount,
			   int gasnet_mem_size_in_mb,
			   int registered_mem_size_in_mb,
			   int registered_ib_mem_size_in_mb,
			   Realm::CoreReservationSet& crs,
			   int argc, const char *argv[]);
extern void start_polling_threads(int count);
extern void start_handler_threads(int count, Realm::CoreReservationSet& crs, size_t stacksize);
extern void stop_activemsg_threads(void);
extern void report_activemsg_status(FILE *f);

// returns the largest payload that can be sent to a node (to a non-pinned
//   address)
extern size_t get_lmb_size(int target_node);

// do a little bit of polling to try to move messages along, but return
//  to the caller rather than spinning
extern void do_some_polling(void);

/* Necessary base structure for all medium and long active messages */
struct BaseMedium {
  static const gasnet_handlerarg_t MESSAGE_ID_MAGIC = 0x0bad0bad;
  static const gasnet_handlerarg_t MESSAGE_CHUNKS_MAGIC = 0x0a550a55;
  void set_magic(void) {
    message_id = MESSAGE_ID_MAGIC;
    message_chunks = MESSAGE_CHUNKS_MAGIC;
    srcptr = 0;
  }
  gasnet_handlerarg_t message_id;
  gasnet_handlerarg_t message_chunks;
  void *srcptr;
};

struct BaseReply {
  void *srcptr;
};

extern void release_srcptr(void *ptr);

enum { MSGID_RELEASE_SRCPTR = 252 };

extern void enqueue_message(gasnet_node_t target, int msgid,
			    const void *args, size_t arg_size,
			    const void *payload, size_t payload_size,
			    int payload_mode, void *dstptr = 0);

extern void enqueue_message(gasnet_node_t target, int msgid,
			    const void *args, size_t arg_size,
			    const void *payload, size_t line_size,
			    off_t line_stride, size_t line_count,
			    int payload_mode, void *dstptr = 0);

extern void enqueue_message(gasnet_node_t target, int msgid,
			    const void *args, size_t arg_size,
			    const SpanList& spans, size_t payload_size,
			    int payload_mode, void *dstptr = 0);

class IncomingMessage; // defined below
class IncomingMessageManager;

extern void enqueue_incoming(gasnet_node_t sender, IncomingMessage *msg);

extern void handle_long_msgptr(gasnet_node_t source, const void *ptr);
//extern size_t adjust_long_msgsize(gasnet_node_t source, void *ptr, size_t orig_size);
extern bool adjust_long_msgsize(gasnet_node_t source, void *&ptr, size_t &buffer_size,
				int message_id, int chunks);
extern void record_message(gasnet_node_t source, bool sent_reply);

#ifdef REALM_PROFILE_AM_HANDLERS
struct ActiveMsgHandlerStats {
  size_t count, sum, sum2, minval, maxval;

  ActiveMsgHandlerStats(void)
  : count(0), sum(0), sum2(0), minval(0), maxval(0) {}

  void record(struct timespec& ts_start, struct timespec& ts_end)
  {
    size_t val = 1000000000LL * (ts_end.tv_sec - ts_start.tv_sec) + ts_end.tv_nsec - ts_start.tv_nsec;
    if(!count || (val < minval)) minval = val;
    if(!count || (val > maxval)) maxval = val;
    count++;
    sum += val;
    sum2 += val * val;
  }
};

extern ActiveMsgHandlerStats handler_stats[256];

// have to define this two different ways because we can't put ifdefs in the macros below
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
    handler_stats[MSGID].record(ts_start, ts_end);
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

// abstract class for incoming messages - actual messages are
//  templated on their argument types and handler
class IncomingMessage {
 public:
  IncomingMessage(void)
    : next_msg(0)
  {}
  virtual ~IncomingMessage(void) {}

  virtual void run_handler(void) = 0;

  virtual int get_peer(void) = 0;
  virtual int get_msgid(void) = 0;
  virtual size_t get_msgsize(void) = 0;

  IncomingMessage *next_msg;
};

template <class MSGTYPE, int MSGID,
          void (*SHORT_HNDL_PTR)(MSGTYPE),
          void (*MED_HNDL_PTR)(MSGTYPE, const void *, size_t),
          int MSG_N>
struct MessageRawArgs;

template <class MSGTYPE>
void dummy_short_handler(MSGTYPE dummy) {}

template <class MSGTYPE>
void dummy_medium_handler(MSGTYPE dummy, const void *data, size_t datalen) {}

template <class MSGTYPE, int MSGID, void (*SHORT_HNDL_PTR)(MSGTYPE), int MSG_N>
class IncomingShortMessage : public IncomingMessage {
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
class IncomingMediumMessage : public IncomingMessage {
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

template <class T> struct HandlerReplyFuture {
  gasnet_hsl_t mutex;
  gasnett_cond_t condvar;
  bool valid;
  T value;

  HandlerReplyFuture(void) {
    gasnet_hsl_init(&mutex);
    gasnett_cond_init(&condvar);
    valid = false;
  }

  void set(T newval)
  {
    gasnet_hsl_lock(&mutex);
    valid = true;
    value = newval;
    gasnett_cond_broadcast(&condvar);
    gasnet_hsl_unlock(&mutex);
  }

  bool is_set(void) const { return valid; }

  void wait(void)
  {
    if(valid) return; // early out
    gasnet_hsl_lock(&mutex);
    while(!valid) gasnett_cond_wait(&condvar, &mutex.lock);
    gasnet_hsl_unlock(&mutex);
  }

  T get(void) const { return value; }
};

#define HANDLERARG_DECL_1                     gasnet_handlerarg_t arg0
#define HANDLERARG_DECL_2  HANDLERARG_DECL_1; gasnet_handlerarg_t arg1
#define HANDLERARG_DECL_3  HANDLERARG_DECL_2; gasnet_handlerarg_t arg2
#define HANDLERARG_DECL_4  HANDLERARG_DECL_3; gasnet_handlerarg_t arg3
#define HANDLERARG_DECL_5  HANDLERARG_DECL_4; gasnet_handlerarg_t arg4
#define HANDLERARG_DECL_6  HANDLERARG_DECL_5; gasnet_handlerarg_t arg5
#define HANDLERARG_DECL_7  HANDLERARG_DECL_6; gasnet_handlerarg_t arg6
#define HANDLERARG_DECL_8  HANDLERARG_DECL_7; gasnet_handlerarg_t arg7
#define HANDLERARG_DECL_9  HANDLERARG_DECL_8; gasnet_handlerarg_t arg8
#define HANDLERARG_DECL_10  HANDLERARG_DECL_9; gasnet_handlerarg_t arg9
#define HANDLERARG_DECL_11  HANDLERARG_DECL_10; gasnet_handlerarg_t arg10
#define HANDLERARG_DECL_12  HANDLERARG_DECL_11; gasnet_handlerarg_t arg11
#define HANDLERARG_DECL_13  HANDLERARG_DECL_12; gasnet_handlerarg_t arg12
#define HANDLERARG_DECL_14  HANDLERARG_DECL_13; gasnet_handlerarg_t arg13
#define HANDLERARG_DECL_15  HANDLERARG_DECL_14; gasnet_handlerarg_t arg14
#define HANDLERARG_DECL_16  HANDLERARG_DECL_15; gasnet_handlerarg_t arg15

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

#define HANDLERARG_PARAMS_1                      gasnet_handlerarg_t arg0
#define HANDLERARG_PARAMS_2 HANDLERARG_PARAMS_1, gasnet_handlerarg_t arg1
#define HANDLERARG_PARAMS_3 HANDLERARG_PARAMS_2, gasnet_handlerarg_t arg2
#define HANDLERARG_PARAMS_4 HANDLERARG_PARAMS_3, gasnet_handlerarg_t arg3
#define HANDLERARG_PARAMS_5 HANDLERARG_PARAMS_4, gasnet_handlerarg_t arg4
#define HANDLERARG_PARAMS_6 HANDLERARG_PARAMS_5, gasnet_handlerarg_t arg5
#define HANDLERARG_PARAMS_7 HANDLERARG_PARAMS_6, gasnet_handlerarg_t arg6
#define HANDLERARG_PARAMS_8 HANDLERARG_PARAMS_7, gasnet_handlerarg_t arg7
#define HANDLERARG_PARAMS_9 HANDLERARG_PARAMS_8, gasnet_handlerarg_t arg8
#define HANDLERARG_PARAMS_10 HANDLERARG_PARAMS_9, gasnet_handlerarg_t arg9
#define HANDLERARG_PARAMS_11 HANDLERARG_PARAMS_10, gasnet_handlerarg_t arg10
#define HANDLERARG_PARAMS_12 HANDLERARG_PARAMS_11, gasnet_handlerarg_t arg11
#define HANDLERARG_PARAMS_13 HANDLERARG_PARAMS_12, gasnet_handlerarg_t arg12
#define HANDLERARG_PARAMS_14 HANDLERARG_PARAMS_13, gasnet_handlerarg_t arg13
#define HANDLERARG_PARAMS_15 HANDLERARG_PARAMS_14, gasnet_handlerarg_t arg14
#define HANDLERARG_PARAMS_16 HANDLERARG_PARAMS_15, gasnet_handlerarg_t arg15

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

#define SPECIALIZED_RAW_ARGS(n) \
template <class MSGTYPE, int MSGID, \
          void (*SHORT_HNDL_PTR)(MSGTYPE), \
          void (*MED_HNDL_PTR)(MSGTYPE, const void *, size_t)> \
struct MessageRawArgs<MSGTYPE, MSGID, SHORT_HNDL_PTR, MED_HNDL_PTR, n> { \
  HANDLERARG_DECL_ ## n ; \
\
  typedef IncomingShortMessage<MSGTYPE,MSGID,SHORT_HNDL_PTR,n> ISHORT; \
  typedef IncomingMediumMessage<MSGTYPE,MSGID,MED_HNDL_PTR,n> IMED; \
  static void handler_short(gasnet_token_t token, HANDLERARG_PARAMS_ ## n ) \
  { \
    gasnet_node_t src; \
    CHECK_GASNET( gasnet_AMGetMsgSource(token, &src) ); \
    /*printf("handling message from node %d (id=%d)\n", src, MSGID);*/	\
    ISHORT *imsg = new ISHORT(src); \
    HANDLERARG_COPY_ ## n(imsg->u) ;				\
    record_message(src, false); \
    enqueue_incoming(src, imsg); \
  } \
\
  static void handler_medium(gasnet_token_t token, void *buf, size_t nbytes, \
                             HANDLERARG_PARAMS_ ## n ) \
  { \
    gasnet_node_t src; \
    CHECK_GASNET( gasnet_AMGetMsgSource(token, &src) ); \
    /*printf("handling medium message from node %d (id=%d)\n", src, MSGID);*/ \
    bool handle_now = adjust_long_msgsize(src, buf, nbytes, arg0, arg1); \
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
	CHECK_GASNET( gasnet_AMReplyShort2(token, MSGID_RELEASE_SRCPTR, (gasnet_handlerarg_t)srcptr, (gasnet_handlerarg_t)(srcptr >> 32)) ); \
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

#ifdef ACTIVE_MESSAGE_TRACE
void record_am_handler(int msgid, const char *description, bool reply = false);
#endif

template <typename T>
union AtLeastEightBytes {
  T data;
  uint64_t padding;
};

template <int MSGID, class MSGTYPE, void (*FNPTR)(MSGTYPE)>
class ActiveMessageShortNoReply {
 public:
  typedef MessageRawArgs<MSGTYPE,MSGID,FNPTR,dummy_medium_handler,(sizeof(AtLeastEightBytes<MSGTYPE>)+3)/4> MessageRawArgsType;

  static void request(gasnet_node_t dest, MSGTYPE args)
  {
    enqueue_message(dest, MSGID, &args, sizeof(MSGTYPE),
		    0, 0, PAYLOAD_NONE);
#ifdef OLD_AM_STUFF
#ifdef CHECK_REENTRANT_MESSAGES
    if(gasnett_threadkey_get(in_handler)) {
      printf("Help!  Message send inside handler!\n");
    } else {
#else
    {
#endif
      union {
        MessageRawArgsType raw;
        MSGTYPE typed;
      } u;
      u.typed = args;
      u.raw.request_short(dest);
    }
#endif
  }

  static int add_handler_entries(gasnet_handlerentry_t *entries, const char *description)
  {
    assert(sizeof(MessageRawArgsType) <= 64);  // max of 16 4-byte args
    entries[0].index = MSGID;
    entries[0].fnptr = (void (*)()) (MessageRawArgsType::handler_short);
#ifdef ACTIVE_MESSAGE_TRACE
    record_am_handler(MSGID, description);
#endif
    return 1;
  }
};

template <int MSGID, class MSGTYPE, void (*FNPTR)(MSGTYPE, const void *, size_t)>
class ActiveMessageMediumNoReply {
 public:
  typedef MessageRawArgs<MSGTYPE,MSGID,dummy_short_handler,FNPTR,(sizeof(AtLeastEightBytes<MSGTYPE>)+3)/4> MessageRawArgsType;

  static void request(gasnet_node_t dest, /*const*/ MSGTYPE &args, 
                      const void *data, size_t datalen,
		      int payload_mode, void *dstptr = 0)
  {
    args.set_magic();
    enqueue_message(dest, MSGID, &args, sizeof(MSGTYPE),
		    data, datalen, payload_mode, dstptr);
  }

  static void request(gasnet_node_t dest, /*const*/ MSGTYPE &args, 
                      const void *data, size_t line_len,
		      off_t line_stride, size_t line_count,
		      int payload_mode, void *dstptr = 0)
  {
    args.set_magic();
    enqueue_message(dest, MSGID, &args, sizeof(MSGTYPE),
		    data, line_len, line_stride, line_count, payload_mode, dstptr);
  }

  static void request(gasnet_node_t dest, /*const*/ MSGTYPE &args, 
                      const SpanList& spans, size_t datalen,
		      int payload_mode, void *dstptr = 0)
  {
    args.set_magic();
    enqueue_message(dest, MSGID, &args, sizeof(MSGTYPE),
		    spans, datalen, payload_mode, dstptr);
  }

  static int add_handler_entries(gasnet_handlerentry_t *entries, const char *description)
  {
    assert(sizeof(MessageRawArgsType) <= 64);  // max of 16 4-byte args
    entries[0].index = MSGID;
    entries[0].fnptr = (void (*)()) (MessageRawArgsType::handler_medium);
#ifdef ACTIVE_MESSAGE_TRACE
    record_am_handler(MSGID, description);
#endif
    return 1;
  }
};

/* template <int MSGID, class ARGTYPE, void (*FNPTR)(ARGTYPE, const void *, size_t)> */
/* class ActiveMessageMedLongNoReply { */
/*  public: */
/*   static void request(gasnet_node_t dest, gasnet_handler_t handler, */
/* 		      ARGTYPE args, const void *data, size_t datalen) */
/*   { */
/*     HandlerArgUnion<ARGTYPE, sizeof(ARGTYPE)/4> u; */
/*     u.typed = args; */
/*     u.raw.request_medium(dest, handler, data, datalen); */
/*   } */

/*   static int add_handler_entries(gasnet_handlerentry_t *entries) */
/*   { */
/*     entries[0].index = MSGID; */
/*     entries[0].fnptr = FNPTR; */
/*     return 1; */
/*   } */
/* }; */

#else // ifdef USE_GASNET

#define CHECK_GASNET(cmd) cmd

typedef unsigned gasnet_node_t;
#define gasnet_mynode() ((gasnet_node_t)0)
#define gasnet_nodes()  ((gasnet_node_t)1)

#include <pthread.h>

#if 0
// gasnet_hsl_t is a struct containing a pthread_mutex_t
typedef struct {
  //struct { pthread_mutex_t lock; } mutex;
  pthread_mutex_t lock;
} gasnet_hsl_t;
#define GASNET_HSL_INITIALIZER  { PTHREAD_MUTEX_INITIALIZER }

inline void gasnet_hsl_init(gasnet_hsl_t *mutex) 
{ pthread_mutex_init(&(mutex->lock), 0); }
inline void gasnet_hsl_destroy(gasnet_hsl_t *mutex)
{ pthread_mutex_destroy(&(mutex->lock)); }
inline void gasnet_hsl_lock(gasnet_hsl_t *mutex) 
{ pthread_mutex_lock(&(mutex->lock)); }
inline void gasnet_hsl_unlock(gasnet_hsl_t *mutex) 
{ pthread_mutex_unlock(&(mutex->lock)); }
#endif

#define GASNET_WAIT_BLOCK 0
inline void gasnet_set_waitmode(int) {}

// gasnet_hsl_t in object form for templating goodness
class GASNetHSL {
public:
  GASNetHSL(void) { pthread_mutex_init(&mutex, 0); }
  ~GASNetHSL(void) { pthread_mutex_destroy(&mutex); }

private:
  // Should never be copied
  GASNetHSL(const GASNetHSL &rhs) { assert(false); }
  GASNetHSL& operator=(const GASNetHSL &rhs) { assert(false); return *this; }

public:
  void lock(void) { pthread_mutex_lock(&mutex); }
  void unlock(void) { pthread_mutex_unlock(&mutex); }

protected:
  friend class GASNetCondVar;
  pthread_mutex_t mutex;
};

class GASNetCondVar {
public:
  GASNetCondVar(GASNetHSL &_mutex) 
    : mutex(_mutex)
  {
    pthread_cond_init(&cond, 0);
  }

  ~GASNetCondVar(void)
  {
    pthread_cond_destroy(&cond);
  }

  // these require that you hold the lock when you call
  void signal(void)
  {
    pthread_cond_signal(&cond);
  }

  void broadcast(void)
  {
    pthread_cond_broadcast(&cond);
  }

  void wait(void)
  {
    pthread_cond_wait(&cond, &mutex.mutex);
  }

  GASNetHSL &mutex;

protected:
  pthread_cond_t cond;
};

 // barriers
#define GASNET_BARRIERFLAG_ANONYMOUS 0
inline void gasnet_barrier_notify(int, int) {}
inline void gasnet_barrier_wait(int, int) {}

// threadkeys
class ThreadKey {
public:
  ThreadKey(void) { pthread_key_create(&key, 0); }
  ~ThreadKey(void) { pthread_key_delete(key); }
  void *get(void) { return pthread_getspecific(key); }
  void set(void *newval) { pthread_setspecific(key, newval); }
protected:
  pthread_key_t key;
};
#if 0
#define GASNETT_THREADKEY_DECLARE(keyname) extern ThreadKey keyname
#define GASNETT_THREADKEY_DEFINE(keyname) ThreadKey keyname

inline void *gasnett_threadkey_get(ThreadKey& key) { return key.get(); }
inline void gasnett_threadkey_set(ThreadKey& key, void *newval) { key.set(newval); }
#else
#define GASNETT_THREADKEY_DECLARE(keyname) extern ThreadKey& get_key_##keyname(void)
#define GASNETT_THREADKEY_DEFINE(keyname) \
  ThreadKey& get_key_##keyname(void) { \
    static ThreadKey key;         \
    return key;                   \
  }

#define gasnett_threadkey_set(keyname, value)  get_key_##keyname().set(value)
#define gasnett_threadkey_get(keyname) get_key_##keyname().get()

#endif

// active message placeholders

typedef int gasnet_handlerentry_t;
typedef int gasnet_handle_t;
typedef struct {
  void *addr;
  size_t size;
} gasnet_seginfo_t;

// define these somewhere so you only get one copy...
extern void *fake_gasnet_mem_base;
extern size_t fake_gasnet_mem_size;

inline void gasnet_init(int*, char ***) {}
inline void gasnet_getSegmentInfo(gasnet_seginfo_t *seginfos, gasnet_node_t count)
{
  seginfos[0].addr = fake_gasnet_mem_base;
  seginfos[0].size = fake_gasnet_mem_size;
}

inline void gasnet_get(void *, int, void *, size_t) { assert(0 && "No GASNet support"); }
inline void gasnet_get_nbi(void *, int, void *, size_t) { assert(0 && "No GASNet support"); }
inline void gasnet_put(int, void *, void *, size_t) { assert(0 && "No GASNet support"); }
inline void gasnet_put_nbi(int, void *, void *, size_t) { assert(0 && "No GASNet support"); }
inline void gasnet_wait_syncnbi_gets(void) {}
inline void gasnet_wait_syncnb(gasnet_handle_t) {}
inline void gasnet_begin_nbi_accessregion(void) {}
inline gasnet_handle_t gasnet_end_nbi_accessregion(void) { return 0; }
inline void gasnet_exit(int code) { exit(code); }

class BaseMedium { public: void *srcptr; };
class BaseReply {};

class ActiveMessagesNotImplemented {
public:
  static int add_handler_entries(gasnet_handlerentry_t *entries, const char *description)
  {
    // no error here - want to allow startup code to run ok
    return 0;
  }
};

template <int MSGID, class MSGTYPE, void (*FNPTR)(MSGTYPE)>
  class ActiveMessageShortNoReply : public ActiveMessagesNotImplemented {
public:
  static void request(gasnet_node_t dest, MSGTYPE args)
  {
    assert(0 && "compiled without USE_GASNET - active messages not available!");
  }
};

template <int MSGID, class MSGTYPE, void (*FNPTR)(MSGTYPE, const void *, size_t)>
class ActiveMessageMediumNoReply : public ActiveMessagesNotImplemented {
public:
  static void request(gasnet_node_t dest, /*const*/ MSGTYPE &args, 
                      const void *data, size_t datalen,
                      int payload_mode, void *dstptr = 0)
  {
    assert(0 && "compiled without USE_GASNET - active messages not available!");
  }

  static void request(gasnet_node_t dest, /*const*/ MSGTYPE &args, 
                      const void *data, size_t line_len,
		      off_t line_stride, size_t line_count,
		      int payload_mode, void *dstptr = 0)
  {
    assert(0 && "compiled without USE_GASNET - active messages not available!");
  }

  static void request(gasnet_node_t dest, /*const*/ MSGTYPE &args, 
                      const SpanList& spans, size_t datalen,
		      int payload_mode, void *dstptr = 0)
  {
    assert(0 && "compiled without USE_GASNET - active messages not available!");
  }
};

template <class T> struct HandlerReplyFuture {
  void wait(void) {}
  void set(T newval) { val = newval; }
  T get(void) const { return val; }
  T val;
};

inline void init_endpoints(gasnet_handlerentry_t *handlers, int hcount,
			   int gasnet_mem_size_in_mb,
			   int registered_mem_size_in_mb,
			   int registered_ib_mem_size_in_mb,
			   Realm::CoreReservationSet& crs,
                           int argc, const char *argv[])
{
  // just use malloc to obtain "gasnet" and/or "registered" memory
  fake_gasnet_mem_size = (gasnet_mem_size_in_mb + 
			  registered_mem_size_in_mb) << 20;
  fake_gasnet_mem_base = malloc(fake_gasnet_mem_size);
}

inline void start_polling_threads(int) {}
inline void start_handler_threads(int, Realm::CoreReservationSet&, size_t) {}
inline void stop_activemsg_threads(void)
{
  if(fake_gasnet_mem_base)
    free(fake_gasnet_mem_base);
  fake_gasnet_mem_base = 0;
}
    
inline void do_some_polling(void) {}
inline size_t get_lmb_size(int target_node) { return 0; }

#endif // ifdef USE_GASNET

    template <typename LT>
    class AutoLock {
    public:
      AutoLock(LT &mutex) : mutex(mutex), held(true)
      { 
	mutex.lock();
      }

      ~AutoLock(void) 
      {
	if(held)
	  mutex.unlock();
      }

      void release(void)
      {
	assert(held);
	mutex.unlock();
	held = false;
      }

      void reacquire(void)
      {
	assert(!held);
	mutex.lock();
	held = true;
      }
    protected:
      LT &mutex;
      bool held;
    };

    typedef AutoLock<GASNetHSL> AutoHSLLock;

#endif
