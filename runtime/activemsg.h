/* Copyright 2014 Stanford University
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

#define GASNET_PAR
#include <gasnet.h>

#define GASNETT_THREAD_SAFE
#include <gasnet_tools.h>

#ifdef CHECK_REENTRANT_MESSAGES
GASNETT_THREADKEY_DECLARE(in_handler);
#endif

#include "utilities.h"

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

typedef std::pair<const void *, size_t> SpanListEntry;
typedef std::vector<SpanListEntry> SpanList;

extern void init_endpoints(gasnet_handlerentry_t *handlers, int hcount,
			   int gasnet_mem_size_in_mb,
			   int registered_mem_size_in_mb,
			   int argc, const char *argv[]);
extern void start_polling_threads(int count);
extern void start_sending_threads(void);
extern void stop_activemsg_threads(void);
extern void report_activemsg_status(FILE *f);

// returns the largest payload that can be sent to a node (to a non-pinned
//   address)
extern size_t get_lmb_size(int target_node);

// do a little bit of polling to try to move messages along, but return
//  to the caller rather than spinning
extern void do_some_polling(void);

enum { PAYLOAD_NONE, // no payload in packet
       PAYLOAD_KEEP, // use payload pointer, guaranteed to be stable
       PAYLOAD_FREE, // take ownership of payload, free when done
       PAYLOAD_COPY, // make a copy of the payload
       PAYLOAD_SRCPTR, // payload has been copied to the src data pool
       PAYLOAD_PENDING, // payload needs to be copied, but hasn't yet
       PAYLOAD_KEEPREG, // use payload pointer, AND it's registered!
};

/* Necessary base structure for all medium and long active messages */
struct BaseMedium {
  static const gasnet_handlerarg_t MESSAGE_ID_MAGIC = 0x0bad0bad;
  static const gasnet_handlerarg_t MESSAGE_CHUNKS_MAGIC = 0x0a550a55;
  void set_magic(void) {
    message_id = MESSAGE_ID_MAGIC;
    message_chunks = MESSAGE_CHUNKS_MAGIC;
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

extern void handle_long_msgptr(gasnet_node_t source, void *ptr);
//extern size_t adjust_long_msgsize(gasnet_node_t source, void *ptr, size_t orig_size);
extern bool adjust_long_msgsize(gasnet_node_t source, void *&ptr, size_t &buffer_size,
                                const void *args, size_t arglen);
extern void record_message(gasnet_node_t source, bool sent_reply);

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
    gasnet_hsl_lock(&mutex);
    while(!valid) gasnett_cond_wait(&condvar, &mutex.lock);
    gasnet_hsl_unlock(&mutex);
  }

  T get(void) const { return value; }
};

template <class ARGTYPE, class RPLTYPE>
struct ArgsWithReplyInfo {
  HandlerReplyFuture<RPLTYPE> *fptr;
  ARGTYPE                      args;
};

template <class MSGTYPE, int MSGID,
          void (*SHORT_HNDL_PTR)(MSGTYPE),
          void (*MED_HNDL_PTR)(MSGTYPE, const void *, size_t),
          int MSG_N>
struct MessageRawArgs;

template <class REQTYPE, int REQID, class RPLTYPE, int RPLID,
          RPLTYPE (*SHORT_HNDL_PTR)(REQTYPE),
          RPLTYPE (*MEDIUM_HNDL_PTR)(REQTYPE, const void *, size_t), int RPL_N, int REQ_N>
  struct RequestRawArgs;

template <class RPLTYPE, int RPLID, int RPL_N> struct ShortReplyRawArgs;
template <class RPLTYPE, int RPLID, int RPL_N> struct MediumReplyRawArgs;

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

#define HANDLERARG_COPY_1                    u.raw.arg0 = arg0
#define HANDLERARG_COPY_2 HANDLERARG_COPY_1; u.raw.arg1 = arg1
#define HANDLERARG_COPY_3 HANDLERARG_COPY_2; u.raw.arg2 = arg2
#define HANDLERARG_COPY_4 HANDLERARG_COPY_3; u.raw.arg3 = arg3
#define HANDLERARG_COPY_5 HANDLERARG_COPY_4; u.raw.arg4 = arg4
#define HANDLERARG_COPY_6 HANDLERARG_COPY_5; u.raw.arg5 = arg5
#define HANDLERARG_COPY_7 HANDLERARG_COPY_6; u.raw.arg6 = arg6
#define HANDLERARG_COPY_8 HANDLERARG_COPY_7; u.raw.arg7 = arg7
#define HANDLERARG_COPY_9 HANDLERARG_COPY_8; u.raw.arg8 = arg8
#define HANDLERARG_COPY_10 HANDLERARG_COPY_9; u.raw.arg9 = arg9
#define HANDLERARG_COPY_11 HANDLERARG_COPY_10; u.raw.arg10 = arg10
#define HANDLERARG_COPY_12 HANDLERARG_COPY_11; u.raw.arg11 = arg11
#define HANDLERARG_COPY_13 HANDLERARG_COPY_12; u.raw.arg12 = arg12
#define HANDLERARG_COPY_14 HANDLERARG_COPY_13; u.raw.arg13 = arg13
#define HANDLERARG_COPY_15 HANDLERARG_COPY_14; u.raw.arg14 = arg14
#define HANDLERARG_COPY_16 HANDLERARG_COPY_15; u.raw.arg15 = arg15

#define MACROPROXY(a,...) a(__VA_ARGS__)

#define SPECIALIZED_RAW_ARGS(n) \
template <class MSGTYPE, int MSGID, \
          void (*SHORT_HNDL_PTR)(MSGTYPE), \
          void (*MED_HNDL_PTR)(MSGTYPE, const void *, size_t)> \
struct MessageRawArgs<MSGTYPE, MSGID, SHORT_HNDL_PTR, MED_HNDL_PTR, n> { \
  HANDLERARG_DECL_ ## n ; \
\
  static void handler_short(gasnet_token_t token, HANDLERARG_PARAMS_ ## n ) \
  { \
    gasnet_node_t src; \
    CHECK_GASNET( gasnet_AMGetMsgSource(token, &src) ); \
    /*printf("handling message from node %d (id=%d)\n", src, MSGID);*/	\
    union { \
      MessageRawArgs<MSGTYPE,MSGID,SHORT_HNDL_PTR,MED_HNDL_PTR,n> raw; \
      MSGTYPE typed; \
    } u; \
    HANDLERARG_COPY_ ## n ; \
    record_message(src, false); \
    (*SHORT_HNDL_PTR)(u.typed); \
  } \
\
  static void handler_medium(gasnet_token_t token, void *buf, size_t nbytes, \
                             HANDLERARG_PARAMS_ ## n ) \
  { \
    gasnet_node_t src; \
    CHECK_GASNET( gasnet_AMGetMsgSource(token, &src) ); \
    /*printf("handling medium message from node %d (id=%d)\n", src, MSGID);*/ \
    union { \
      MessageRawArgs<MSGTYPE,MSGID,SHORT_HNDL_PTR,MED_HNDL_PTR,n> raw; \
      MSGTYPE typed; \
    } u; \
    HANDLERARG_COPY_ ## n ; \
    record_message(src, true); \
    /*nbytes = adjust_long_msgsize(src, buf, nbytes);*/	\
    bool handle_now = adjust_long_msgsize(src, buf, nbytes, &u, sizeof(u)); \
    if (handle_now) { \
      (*MED_HNDL_PTR)(u.typed, buf, nbytes); \
      if(nbytes > 0/*gasnet_AMMaxMedium()*/) handle_long_msgptr(src, buf); \
      /* We need to send an reply no matter what since asynchronous active*/ \
      /* messages require a reply. */ \
      /*printf("sending release of srcptr %p (%d -> %d)\n", u.typed.srcptr, gasnet_mynode(), src);*/ \
      /* can't use offsetof() here because MSGTYPE is non-POD */ \
      assert((((char *)&(u.typed.srcptr)) - ((char *)&(u.raw))) == 8); \
      CHECK_GASNET( gasnet_AMReplyShort2(token, MSGID_RELEASE_SRCPTR, u.raw.arg2, u.raw.arg3) ); \
    } else { \
      CHECK_GASNET( gasnet_AMReplyShort2(token, MSGID_RELEASE_SRCPTR, 0, 0) ); \
    } \
  } \
}; \
\
template <class REQTYPE, int REQID, class RPLTYPE, int RPLID, \
          RPLTYPE (*SHORT_HNDL_PTR)(REQTYPE), \
          RPLTYPE (*MEDIUM_HNDL_PTR)(REQTYPE, const void *, size_t), int RPL_N>	\
struct RequestRawArgs<REQTYPE, REQID, RPLTYPE, RPLID, SHORT_HNDL_PTR, MEDIUM_HNDL_PTR, RPL_N, n> { \
  HANDLERARG_DECL_ ## n ; \
\
  static void handler_short(gasnet_token_t token, HANDLERARG_PARAMS_ ## n ) \
  { \
    gasnet_node_t src; \
    CHECK_GASNET( gasnet_AMGetMsgSource(token, &src) ); \
    /*printf("handling request from node %d\n", src);*/	\
    union { \
      RequestRawArgs<REQTYPE,REQID,RPLTYPE,RPLID,SHORT_HNDL_PTR,MEDIUM_HNDL_PTR,RPL_N,n> raw; \
      ArgsWithReplyInfo<REQTYPE,RPLTYPE> typed; \
    } u; \
    HANDLERARG_COPY_ ## n ; \
    record_message(src, true); \
\
    union { \
      ShortReplyRawArgs<RPLTYPE,RPLID,RPL_N> raw; \
      ArgsWithReplyInfo<RPLTYPE,RPLTYPE> typed; \
    } rpl_u; \
\
    rpl_u.typed.args = (*SHORT_HNDL_PTR)(u.typed.args); \
    rpl_u.typed.fptr = u.typed.fptr; \
    rpl_u.raw.reply_short(token); \
  } \
\
  static void handler_medium(gasnet_token_t token, void *buf, size_t nbytes, \
                             HANDLERARG_PARAMS_ ## n ) \
  { \
    gasnet_node_t src; \
    CHECK_GASNET( gasnet_AMGetMsgSource(token, &src) ); \
    /*printf("handling request from node %d\n", src);*/	\
    union { \
      RequestRawArgs<REQTYPE,REQID,RPLTYPE,RPLID,SHORT_HNDL_PTR,MEDIUM_HNDL_PTR,RPL_N,n> raw; \
      ArgsWithReplyInfo<REQTYPE,RPLTYPE> typed; \
    } u; \
    HANDLERARG_COPY_ ## n ; \
    /*printf("medium message ID=%d srcptr=%p, fptr=%p\n", REQID, u.typed.args.srcptr, u.typed.fptr);*/ \
    record_message(src, true); \
\
    union { \
      MediumReplyRawArgs<RPLTYPE,RPLID,RPL_N> raw; \
      ArgsWithReplyInfo<RPLTYPE,RPLTYPE> typed; \
    } rpl_u; \
\
    rpl_u.typed.args = (*MEDIUM_HNDL_PTR)(u.typed.args, buf, nbytes);	\
    if(nbytes > 0/*gasnet_AMMaxMedium()*/) handle_long_msgptr(src, buf); \
    rpl_u.typed.fptr = u.typed.fptr; \
    rpl_u.typed.args.srcptr = u.typed.args.srcptr; \
    /*printf("replying with srcptr=%p\n", u.typed.args.srcptr);*/	\
    rpl_u.raw.reply_short(token); \
  } \
}; \
\
template <class RPLTYPE, int RPLID> struct ShortReplyRawArgs<RPLTYPE, RPLID, n> { \
  HANDLERARG_DECL_ ## n ; \
\
  void reply_short(gasnet_token_t token) \
  { \
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_SYSTEM); \
    CHECK_GASNET( MACROPROXY(gasnet_AMReplyShort ## n, token, RPLID, HANDLERARG_VALS_ ## n ) ); \
  } \
 \
  static void handler_short(gasnet_token_t token, HANDLERARG_PARAMS_ ## n ) \
  { \
    gasnet_node_t src; \
    gasnet_AMGetMsgSource(token, &src); \
    /*printf("%d: handling reply from node %d\n", (int)gasnet_mynode(), src);*/ \
    union { \
      ShortReplyRawArgs<RPLTYPE,RPLID,n> raw; \
      ArgsWithReplyInfo<RPLTYPE,RPLTYPE> typed; \
    } u; \
    HANDLERARG_COPY_ ## n ; \
    u.typed.fptr->set(u.typed.args); \
  } \
};  \
\
template <class RPLTYPE, int RPLID> struct MediumReplyRawArgs<RPLTYPE, RPLID, n> { \
  HANDLERARG_DECL_ ## n ; \
\
  void reply_short(gasnet_token_t token) \
  { \
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_SYSTEM); \
    CHECK_GASNET( MACROPROXY(gasnet_AMReplyShort ## n, token, RPLID, HANDLERARG_VALS_ ## n ) ); \
  } \
 \
  static void handler_short(gasnet_token_t token, HANDLERARG_PARAMS_ ## n ) \
  { \
    gasnet_node_t src; \
    gasnet_AMGetMsgSource(token, &src); \
    /*printf("%d: handling reply from node %d\n", (int)gasnet_mynode(), src);*/ \
    union { \
      MediumReplyRawArgs<RPLTYPE,RPLID,n> raw; \
      ArgsWithReplyInfo<RPLTYPE,RPLTYPE> typed; \
    } u; \
    HANDLERARG_COPY_ ## n ; \
    if (u.typed.args.srcptr != 0) { \
      /*printf("recevied inline release of srcptr %p (%d -> %d)\n", u.typed.args.srcptr, src, gasnet_mynode());*/ \
      release_srcptr(u.typed.args.srcptr); \
    } \
    u.typed.fptr->set(u.typed.args); \
  } \
};

SPECIALIZED_RAW_ARGS(1);
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

template <class MSGTYPE>
void dummy_short_handler(MSGTYPE dummy) {}

template <class MSGTYPE>
void dummy_medium_handler(MSGTYPE dummy, const void *data, size_t datalen) {}

template <class MSGTYPE, class RPLTYPE>
RPLTYPE dummy_short_w_reply_handler(MSGTYPE dummy) { RPLTYPE dummyret; return dummyret; }

template <class MSGTYPE, class RPLTYPE>
RPLTYPE dummy_medium_w_reply_handler(MSGTYPE dummy, const void *data, size_t datalen) { RPLTYPE dummyret; return dummyret; }

template <int MSGID, class MSGTYPE, void (*FNPTR)(MSGTYPE)>
class ActiveMessageShortNoReply {
 public:
  typedef MessageRawArgs<MSGTYPE,MSGID,FNPTR,dummy_medium_handler,(sizeof(MSGTYPE)+3)/4> MessageRawArgsType;

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

  static int add_handler_entries(gasnet_handlerentry_t *entries)
  {
    assert(sizeof(MessageRawArgsType) <= 64);  // max of 16 4-byte args
    entries[0].index = MSGID;
    entries[0].fnptr = (void (*)()) (MessageRawArgsType::handler_short);
    return 1;
  }
};

template <int MSGID, class MSGTYPE, void (*FNPTR)(MSGTYPE, const void *, size_t)>
class ActiveMessageMediumNoReply {
 public:
  typedef MessageRawArgs<MSGTYPE,MSGID,dummy_short_handler,FNPTR,(sizeof(MSGTYPE)+3)/4> MessageRawArgsType;

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

  static int add_handler_entries(gasnet_handlerentry_t *entries)
  {
    assert(sizeof(MessageRawArgsType) <= 64);  // max of 16 4-byte args
    entries[0].index = MSGID;
    entries[0].fnptr = (void (*)()) (MessageRawArgsType::handler_medium);
    return 1;
  }
};

template <int REQID, int RPLID, class REQTYPE, class RPLTYPE,
          RPLTYPE (*FNPTR)(REQTYPE)>
class ActiveMessageShortReply {
 public:
  typedef RequestRawArgs<REQTYPE, REQID, RPLTYPE, RPLID, FNPTR, dummy_medium_w_reply_handler<REQTYPE, RPLTYPE>,
                         (sizeof(void*)+sizeof(RPLTYPE)+3)/4,
                         (sizeof(void*)+sizeof(REQTYPE)+3)/4> ReqRawArgsType;
  typedef ShortReplyRawArgs<RPLTYPE, RPLID, (sizeof(void*)+sizeof(RPLTYPE)+3)/4> RplRawArgsType;

  static RPLTYPE request(gasnet_node_t dest, REQTYPE args)
  {
    HandlerReplyFuture<RPLTYPE> future;
    ArgsWithReplyInfo<REQTYPE,RPLTYPE> args_with_reply;
    args_with_reply.fptr = &future;
    args_with_reply.args = args;
    enqueue_message(dest, REQID, &args_with_reply, sizeof(args_with_reply),
		    0, 0, PAYLOAD_NONE);
#ifdef OLD_AM_STUFF
    union {
      ReqRawArgsType raw;
      ArgsWithReplyInfo<REQTYPE,RPLTYPE> typed;
    } u;
      
    u.typed.fptr = &future;
    u.typed.args = args;
    u.typed.args.set_magic();

#ifdef CHECK_REENTRANT_MESSAGES
    if(gasnett_threadkey_get(in_handler)) {
      printf("Help!  Message send inside handler!\n");
    } else {
#else
    {
#endif
      u.raw.request_short(dest);
    }
#endif

    //printf("request sent - waiting for response\n");
    future.wait();
    return future.value;
  }

  static int add_handler_entries(gasnet_handlerentry_t *entries)
  {
    assert(sizeof(ReqRawArgsType) <= 64);  // max of 16 4-byte args
    assert(sizeof(RplRawArgsType) <= 64);  // max of 16 4-byte args
    entries[0].index = REQID;
    entries[0].fnptr = (void (*)()) (ReqRawArgsType::handler_short);
    entries[1].index = RPLID;
    entries[1].fnptr = (void (*)()) (RplRawArgsType::handler_short);
    return 2;
  }
};

template <int REQID, int RPLID, class REQTYPE, class RPLTYPE,
  RPLTYPE (*FNPTR)(REQTYPE, const void *, size_t)>
class ActiveMessageMediumReply {
 public:
  typedef RequestRawArgs<REQTYPE, REQID, RPLTYPE, RPLID, dummy_short_w_reply_handler<REQTYPE, RPLTYPE>, FNPTR,
                         (sizeof(void*)+sizeof(RPLTYPE)+3)/4,
                         (sizeof(void*)+sizeof(REQTYPE)+3)/4> ReqRawArgsType;
  typedef MediumReplyRawArgs<RPLTYPE, RPLID, (sizeof(void*)+sizeof(RPLTYPE)+3)/4> RplRawArgsType;

  static RPLTYPE request(gasnet_node_t dest, REQTYPE args,
			 const void *data, size_t datalen,
			 int payload_mode, void *dstptr = 0)
  {
    HandlerReplyFuture<RPLTYPE> future;
    ArgsWithReplyInfo<REQTYPE,RPLTYPE> args_with_reply;
    args_with_reply.fptr = &future;
    args_with_reply.args = args;
    args_with_reply.args.set_magic();
    enqueue_message(dest, REQID, &args_with_reply, sizeof(args_with_reply),
		    data, datalen, payload_mode, dstptr);
#ifdef OLD_AM_STUFF
    union {
      ReqRawArgsType raw;
      ArgsWithReplyInfo<REQTYPE,RPLTYPE> typed;
    } u;
      
    u.typed.fptr = &future;
    u.typed.args = args;
    u.typed.args.set_magic();

#ifdef CHECK_REENTRANT_MESSAGES
    if(gasnett_threadkey_get(in_handler)) {
      printf("Help!  Message send inside handler!\n");
    } else {
#else
    {
#endif
      u.raw.request_short(dest);
    }
#endif

    //printf("request sent - waiting for response\n");
    future.wait();
    return future.value;
  }

  static RPLTYPE request(gasnet_node_t dest, REQTYPE args,
			 const SpanList& spans, size_t datalen,
			 int payload_mode, void *dstptr = 0)
  {
    HandlerReplyFuture<RPLTYPE> future;
    ArgsWithReplyInfo<REQTYPE,RPLTYPE> args_with_reply;
    args_with_reply.fptr = &future;
    args_with_reply.args = args;
    args_with_reply.args.set_magic();
    enqueue_message(dest, REQID, &args_with_reply, sizeof(args_with_reply),
		    spans, datalen, payload_mode, dstptr);
#ifdef OLD_AM_STUFF
    union {
      ReqRawArgsType raw;
      ArgsWithReplyInfo<REQTYPE,RPLTYPE> typed;
    } u;
      
    u.typed.fptr = &future;
    u.typed.args = args;

#ifdef CHECK_REENTRANT_MESSAGES
    if(gasnett_threadkey_get(in_handler)) {
      printf("Help!  Message send inside handler!\n");
    } else {
#else
    {
#endif
      u.raw.request_short(dest);
    }
#endif

    //printf("request sent - waiting for response\n");
    future.wait();
    return future.value;
  }

  static int add_handler_entries(gasnet_handlerentry_t *entries)
  {
    assert(sizeof(ReqRawArgsType) <= 64);  // max of 16 4-byte args
    assert(sizeof(RplRawArgsType) <= 64);  // max of 16 4-byte args
    entries[0].index = REQID;
    entries[0].fnptr = (void (*)()) (ReqRawArgsType::handler_medium);
    entries[1].index = RPLID;
    entries[1].fnptr = (void (*)()) (RplRawArgsType::handler_short);
    return 2;
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

#endif
