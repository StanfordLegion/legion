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

// hopefully a more user-friendly C++ template wrapper for GASNet active
//  messages...

#ifndef ACTIVEMSG_H
#define ACTIVEMSG_H

#include "realm/realm_config.h"
#include "realm/mutex.h"
#include "realm/serialize.h"
#include "realm/nodeset.h"
#include "realm/network.h"
#include "realm/atomics.h"
#include "realm/threads.h"
#include "realm/bgwork.h"

namespace Realm {

  namespace Config {
    // if true, the number and min/max/avg/stddev duration of handler per
    //  message type is recorded and printed
    extern bool profile_activemsg_handlers;

    // the maximum time we're willing to spend on inline message
    //  handlers
    extern long long max_inline_message_time;
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

  class ActiveMessageImpl;
 
  template <typename T, size_t INLINE_STORAGE = 256>
    class ActiveMessage {
  public:
    // constructs an INACTIVE message object - call init(...) as needed
    ActiveMessage();

    // construct a new active message for either a single recipient or a mask
    //  of recipients
    // in addition to the header struct (T), a message can include a variable
    //  payload which can be delivered to a particular destination address
    ActiveMessage(NodeID _target, size_t _max_payload_size = 0);
    ActiveMessage(NodeID _target, size_t _max_payload_size,
		  const RemoteAddress& _dest_payload_addr);
    ActiveMessage(const Realm::NodeSet &_targets, size_t _max_payload_size = 0);

    // providing the payload (as a 1D or 2D reference, which must be PAYLOAD_KEEP)
    //  up front can avoid a copy if the source location is directly accessible
    //  by the networking hardware
    ActiveMessage(NodeID _target, const void *_data, size_t _datalen);
    ActiveMessage(NodeID _target, const void *_data, size_t _datalen,
		  const RemoteAddress& _dest_payload_addr);
    ActiveMessage(const Realm::NodeSet &_targets,
		  const void *_data, size_t _datalen);
    ActiveMessage(NodeID _target, const void *_data, size_t _bytes_per_line,
		  size_t _lines, size_t _line_stride);
    ActiveMessage(NodeID _target, const void *_data, size_t _bytes_per_line,
		  size_t _lines, size_t _line_stride,
		  const RemoteAddress& _dest_payload_addr);
    ActiveMessage(const Realm::NodeSet &_targets,
		  const void *_data, size_t _bytes_per_line,
		  size_t _lines, size_t _line_stride);

    ~ActiveMessage(void);

    // a version of `init` for each constructor above
    void init(NodeID _target,
	      size_t _max_payload_size = 0);
    void init(NodeID _target,
	      size_t _max_payload_size, const RemoteAddress& _dest_payload_addr);
    void init(const Realm::NodeSet &_targets, size_t _max_payload_size = 0);
    void init(NodeID _target, const void *_data, size_t _datalen);
    void init(NodeID _target, const void *_data, size_t _datalen,
	      const RemoteAddress& _dest_payload_addr);
    void init(const Realm::NodeSet &_targets, const void *_data, size_t _datalen);
    void init(NodeID _target, const void *_data, size_t _bytes_per_line,
	      size_t _lines, size_t _line_stride);
    void init(NodeID _target, const void *_data, size_t _bytes_per_line,
	      size_t _lines, size_t _line_stride,
	      const RemoteAddress& _dest_payload_addr);
    void init(const Realm::NodeSet &_targets,
	      const void *_data, size_t _bytes_per_line,
	      size_t _lines, size_t _line_stride);

    // large messages may need to be fragmented, so use cases that can
    //  handle the fragmentation at a higher level may want to know the
    //  largest size that is fragmentation-free - the answer can depend
    //  on whether the data is to be delivered to a known RemoteAddress
    //  and/or whether the source data location is known
    // a call that sets `with_congestion` may get a smaller value (maybe
    //  even 0) if the path to the named target(s) is getting full
    static size_t recommended_max_payload(NodeID target,
					  bool with_congestion);
    static size_t recommended_max_payload(const NodeSet& targets,
					  bool with_congestion);
    static size_t recommended_max_payload(NodeID target,
					  const RemoteAddress &dest_payload_addr,
					  bool with_congestion);
    static size_t recommended_max_payload(NodeID target,
					  const void *data, size_t bytes_per_line,
					  size_t lines, size_t line_stride,
					  bool with_congestion);
    static size_t recommended_max_payload(const NodeSet& targets,
					  const void *data, size_t bytes_per_line,
					  size_t lines, size_t line_stride,
					  bool with_congestion);
    static size_t recommended_max_payload(NodeID target,
					  const void *data, size_t bytes_per_line,
					  size_t lines, size_t line_stride,
					  const RemoteAddress &dest_payload_addr,
					  bool with_congestion);

    // operator-> gives access to the header structure
    T *operator->(void);

    // variable payload can be written to in three ways:
    //  (a) Realm-style serialization (currently eager)
    template <typename T2>
      bool operator<<(const T2& to_append);
    //  (b) old memcpy-like behavior (using the various payload modes)
    void add_payload(const void *data, size_t datalen,
		     int payload_mode = PAYLOAD_COPY);
    void add_payload(const void *data, size_t bytes_per_line,
		     size_t lines, size_t line_stride,
		     int payload_mode = PAYLOAD_COPY);
    //  (c) request for a pointer to write into (writes must be completed before
    //       call to commit or cancel)
    void *payload_ptr(size_t datalen);

    // register callbacks to be called upon:
    //  a) local completion - i.e. all source data has been read and can now
    //      be safely overwritten
    //  b) remote completion - message has been received AND HANDLED by target
    //
    // callbacks need to be "lightweight" - for heavier work, the message
    //  handler on the target can send an explicit response message
    template <typename CALLABLE>
    void add_local_completion(const CALLABLE& callable);
    template <typename CALLABLE>
    void add_remote_completion(const CALLABLE& callable);

    // every active message must eventually be commit()'ed or cancel()'ed
    void commit(void);
    void cancel(void);

  protected:
    ActiveMessageImpl *impl;
    T *header;
    Realm::Serialization::FixedBufferSerializer fbs;
    uint64_t inline_capacity[INLINE_STORAGE / sizeof(uint64_t)];
  };

  // type-erased wrappers for completion callbacks
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE CompletionCallbackBase {
  public:
    virtual ~CompletionCallbackBase();
    virtual void invoke() = 0;
    virtual size_t size() const = 0;
    virtual CompletionCallbackBase *clone_at(void *p) const = 0;

    static const size_t ALIGNMENT = 8;

    // helper functions for invoking/cloning/destroying a collection of callbacks
    static void invoke_all(void *start, size_t bytes);
    static void clone_all(void *dst, const void *src, size_t bytes);
    static void destroy_all(void *start, size_t bytes);
  };

  template <typename CALLABLE>
  class CompletionCallback : public CompletionCallbackBase {
  public:
    CompletionCallback(const CALLABLE &_callable);
    virtual void invoke();
    virtual size_t size() const;
    virtual CompletionCallbackBase *clone_at(void *p) const;

  protected:
    CALLABLE callable;
  };

  // per-network active message implementations are mostly opaque, but a few
  //  fields are exposed to avoid virtual function calls
  class ActiveMessageImpl {
  public:
    virtual ~ActiveMessageImpl() {}

    // reserves space for a local/remote completion - caller will
    //  placement-new the completion at the provided address
    virtual void *add_local_completion(size_t size) = 0;
    virtual void *add_remote_completion(size_t size) = 0;

    virtual void commit(size_t act_payload_size) = 0;
    virtual void cancel() = 0;

    void *header_base;
    void *payload_base;
    size_t payload_size;
  };

  class ActiveMessageHandlerRegBase;

  struct ActiveMessageHandlerStats {
    atomic<size_t> count, sum, sum2, minval, maxval;

    ActiveMessageHandlerStats(void);
    void record(long long t_start, long long t_end);
  };

  // singleton class that can convert message type->ID and ID->handler
  class ActiveMessageHandlerTable {
  public:
    ActiveMessageHandlerTable(void);
    ~ActiveMessageHandlerTable(void);

    typedef unsigned short MessageID;
    typedef void (*MessageHandler)(NodeID sender, const void *header,
				   const void *payload, size_t payload_size,
				   TimeLimit work_until);
    typedef void (*MessageHandlerNoTimeout)(NodeID sender, const void *header,
					    const void *payload, size_t payload_size);
    typedef bool (*MessageHandlerInline)(NodeID sender, const void *header,
					 const void *payload, size_t payload_size,
					 TimeLimit work_until);

    template <typename T>
      MessageID lookup_message_id(void) const;

    const char *lookup_message_name(MessageID id);
    void record_message_handler_call(MessageID id,
				     long long t_start, long long t_end);
    void report_message_handler_stats();

    static void append_handler_reg(ActiveMessageHandlerRegBase *new_reg);

    void construct_handler_table(void);

    typedef unsigned TypeHash;

    struct HandlerEntry {
      TypeHash hash;
      const char *name;
      MessageHandler handler;
      MessageHandlerNoTimeout handler_notimeout;
      MessageHandlerInline handler_inline;
      ActiveMessageHandlerStats stats;
    };

    HandlerEntry *lookup_message_handler(MessageID id);

  protected:
    static ActiveMessageHandlerRegBase *pending_handlers;

    std::vector<HandlerEntry> handlers;
  };

  extern ActiveMessageHandlerTable activemsg_handler_table;

  class ActiveMessageHandlerRegBase {
  public:
    virtual ~ActiveMessageHandlerRegBase(void) {}
    virtual ActiveMessageHandlerTable::MessageHandler get_handler(void) const = 0;
    virtual ActiveMessageHandlerTable::MessageHandlerNoTimeout get_handler_notimeout(void) const = 0;
    virtual ActiveMessageHandlerTable::MessageHandlerInline get_handler_inline(void) const = 0;

    ActiveMessageHandlerTable::TypeHash hash;
    const char *name;
    bool must_free;
    ActiveMessageHandlerRegBase *next_handler;
  };

  template <typename T, typename T2 = T>
  class ActiveMessageHandlerReg : public ActiveMessageHandlerRegBase {
  public:
    ActiveMessageHandlerReg(void);
    ~ActiveMessageHandlerReg(void);

    // when registering an active message handler, the following three methods
    //  are looked for in class T2
    // (a) void handle_message(NodeID, const T&, const void *, size_t, TimeLimit)
    // (b) void handle_message(NodeID, const T&, const void *, size_t)
    // (c) bool handle_inline(NodeID, const T&, const void *, size_t, TimeLimit)
    //
    // at least one of (a) or (b) must be present, with (a) being preferred
    //
    // if (c) is present, it will be used to attempt inline handling of
    //  active messages as they arrive, with the following constraints:
    //   (i) the handler must not block on any mutexes (trylocks are ok)
    //   (ii) the handler must not perform dynamic memory allocation/frees
    //   (iii) the handler must try very hard to stay within the specified
    //          time limit
    // if the inline handler is unable to satisfy these requirements, it should
    //  not attempt to handle the message, returning 'false' and letting it be
    //  queued as normal

    // returns either the requested kind of handler or a null pointer if
    //  it doesn't exist
    virtual ActiveMessageHandlerTable::MessageHandler get_handler(void) const;
    virtual ActiveMessageHandlerTable::MessageHandlerNoTimeout get_handler_notimeout(void) const;
    virtual ActiveMessageHandlerTable::MessageHandlerInline get_handler_inline(void) const;

    // this method does nothing, but can be called to force the instantiation
    //  of a handler registration object (needed when things are inside templates)
    void force_instantiation(void) {}
  };

  namespace ThreadLocal {
    // this flag will be true when we are running a message handler
    extern REALM_THREAD_LOCAL bool in_message_handler;
  };
  
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE IncomingMessageManager : public BackgroundWorkItem {
  public:
    IncomingMessageManager(int _nodes, int _dedicated_threads,
			   Realm::CoreReservationSet& crs);
    ~IncomingMessageManager(void);

    typedef uintptr_t CallbackData;
    typedef void (*CallbackFnptr)(NodeID, CallbackData, CallbackData);

    // adds an incoming message to the queue
    // returns true if the call was handled immediately (in which case the
    //  callback, if present, will NOT be called), or false if the message
    //  will be processed later
    bool add_incoming_message(NodeID sender,
			      ActiveMessageHandlerTable::MessageID msgid,
			      const void *hdr, size_t hdr_size,
			      int hdr_mode,
			      const void *payload, size_t payload_size,
			      int payload_mode,
			      CallbackFnptr callback_fnptr,
			      CallbackData callback_data1,
			      CallbackData callback_data2,
			      TimeLimit work_until);

    void start_handler_threads(size_t stack_size);

    // stalls caller until all incoming messages have been handled (and at
    //  least 'min_messages_handled' in total)
    void drain_incoming_messages(size_t min_messages_handled);

    void shutdown(void);

    virtual bool do_work(TimeLimit work_until);

    void handler_thread_loop(void);

  protected:
    struct MessageBlock;

    struct Message {
      MessageBlock *block;
      Message *next_msg;
      NodeID sender;
      ActiveMessageHandlerTable::HandlerEntry *handler;
      void *hdr;
      size_t hdr_size;
      bool hdr_needs_free;
      void *payload;
      size_t payload_size;
      bool payload_needs_free;
      CallbackFnptr callback_fnptr;
      CallbackData callback_data1, callback_data2;
    };

    struct MessageBlock {
      static MessageBlock *new_block(size_t _total_size);
      static void free_block(MessageBlock *block);

      void reset();

      // called with message manager lock held
      Message *append_message(size_t hdr_bytes_needed,
			      size_t payload_bytes_needed);

      // called _without_ message manager lock held
      void recycle_message(Message *msg, IncomingMessageManager *manager);

      size_t total_size, size_used;
      atomic<unsigned> use_count;
      MessageBlock *next_free;
    };

    int get_messages(Message *& head, Message **& tail, bool wait);
    bool return_messages(int sender, size_t num_handled,
                         Message *head, Message **tail);

    int nodes, dedicated_threads, sleeper_count;
    atomic<bool> bgwork_requested;
    int shutdown_flag;
    Message **heads;
    Message ***tails;
    bool *in_handler;
    int *todo_list; // list of nodes with non-empty message lists
    int todo_oldest, todo_newest;
    int handlers_active;
    bool drain_pending;
    size_t drain_min_count;
    size_t total_messages_handled;
    Mutex mutex;
    Mutex::CondVar condvar, drain_condvar;
    CoreReservation *core_rsrv;
    std::vector<Thread *> handler_threads;
    MessageBlock *current_block;
    MessageBlock *available_blocks;
    size_t num_available_blocks;
    size_t cfg_max_available_blocks, cfg_message_block_size;
  };

}; // namespace Realm

#include "realm/activemsg.inl"

#endif

