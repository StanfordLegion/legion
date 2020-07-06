/* Copyright 2020 Stanford University, NVIDIA Corporation
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
 
  template <typename T>
    class ActiveMessage {
  public:
    // construct a new active message for either a single recipient or a mask
    //  of recipients
    // in addition to the header struct (T), a message can include a variable
    //  payload which can be delivered to a particular destination address
    ActiveMessage(NodeID _target,
		  size_t _max_payload_size = 0, void *_dest_payload_addr = 0);
    ActiveMessage(const Realm::NodeSet &_targets, size_t _max_payload_size = 0);
    ~ActiveMessage(void);

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

    // every active message must eventually be commit()'ed or cancel()'ed
    void commit(void);
    void cancel(void);

  protected:
    ActiveMessageImpl *impl;
    T *header;
    Realm::Serialization::FixedBufferSerializer fbs;
    static const size_t INLINE_STORAGE = 256;
    uint64_t inline_capacity[INLINE_STORAGE / sizeof(uint64_t)];
  };

  // per-network active message implementations are mostly opaque, but a few
  //  fields are exposed to avoid virtual function calls
  class ActiveMessageImpl {
  public:
    virtual ~ActiveMessageImpl() {}

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
      bool must_free;
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
  
  class IncomingMessageManager : public BackgroundWorkItem {
  public:
    IncomingMessageManager(int _nodes, int _dedicated_threads,
			   Realm::CoreReservationSet& crs);
    ~IncomingMessageManager(void);

    typedef uintptr_t CallbackData;
    typedef void (*CallbackFnptr)(NodeID, CallbackData);

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
			      CallbackData callback_data,
			      TimeLimit work_until);

    void start_handler_threads(size_t stack_size);

    // stalls caller until all incoming messages have been handled
    void drain_incoming_messages(void);

    void shutdown(void);

    virtual void do_work(TimeLimit work_until);

    void handler_thread_loop(void);

  protected:
    struct Message {
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
      CallbackData callback_data;
    };

    int get_messages(Message *& head, Message **& tail, bool wait);
    void return_messages(int sender, Message *head, Message **tail);

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
    Realm::Mutex mutex;
    Realm::CondVar condvar, drain_condvar;
    Realm::CoreReservation *core_rsrv;
    std::vector<Realm::Thread *> handler_threads;
  };

}; // namespace Realm

#include "realm/activemsg.inl"

#endif

