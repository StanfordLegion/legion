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

namespace Realm {

  namespace Config {
    // if true, the number and min/max/avg/stddev duration of handler per
    //  message type is recorded and printed
    extern bool profile_activemsg_handlers;
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
  size_t count, sum, sum2, minval, maxval;

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
				 const void *payload, size_t payload_size);

  template <typename T>
  MessageID lookup_message_id(void) const;

  MessageHandler lookup_message_handler(MessageID id);
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
    ActiveMessageHandlerStats stats;
  };

protected:
  static ActiveMessageHandlerRegBase *pending_handlers;

  std::vector<HandlerEntry> handlers;
};

extern ActiveMessageHandlerTable activemsg_handler_table;

class ActiveMessageHandlerRegBase {
public:
  virtual ~ActiveMessageHandlerRegBase(void) {}
  virtual ActiveMessageHandlerTable::MessageHandler get_handler(void) const = 0;

  ActiveMessageHandlerTable::TypeHash hash;
  const char *name;
  bool must_free;
  ActiveMessageHandlerRegBase *next_handler;
};

template <typename T, typename T2 = T>
class ActiveMessageHandlerReg : public ActiveMessageHandlerRegBase {
public:
  ActiveMessageHandlerReg(void);
  virtual ActiveMessageHandlerTable::MessageHandler get_handler(void) const;

  // this method does nothing, but can be called to force the instantiation
  //  of a handler registration object (needed when things are inside templates)
  void force_instantiation(void) {}

protected:
  static void handler_wrapper(NodeID sender, const void *header,
			      const void *payload, size_t payload_size);
};

class PayloadSource {
public:
  PayloadSource(void) { }
  virtual ~PayloadSource(void) { }
public:
  virtual void copy_data(void *dest) = 0;
  virtual void *get_contig_pointer(void) { assert(false); return 0; }
  virtual int get_payload_mode(void) { return PAYLOAD_KEEP; }
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

}; // namespace Realm

#include "realm/activemsg.inl"

#endif

