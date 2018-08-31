/* Copyright 2018 Stanford University, NVIDIA Corporation
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

// INCLDUED FROM activemsg.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/activemsg.h"

////////////////////////////////////////////////////////////////////////
//
// class ActiveMessage<T>
//

template <typename T>
ActiveMessage<T>::ActiveMessage(NodeID _target,
				size_t _max_payload_size /*= 0*/,
				void *_dest_payload_addr /*= 0*/)
  : valid(true)
  , target(_target)
  , is_multicast(false)
  , max_payload_size(_max_payload_size)
  , dest_payload_addr(_dest_payload_addr)
{
  if(max_payload_size) {
    payload = reinterpret_cast<char *>(malloc(max_payload_size));
    fbs.reset(payload, max_payload_size);
  } else {
    payload = 0;
  }
}
    
template <typename T>
ActiveMessage<T>::ActiveMessage(const Realm::NodeSet &_targets,
				size_t _max_payload_size /*= 0*/)
  : valid(true)
  , targets(_targets)
  , is_multicast(true)
  , max_payload_size(_max_payload_size)
  , dest_payload_addr(0)
{
  if(max_payload_size) {
    payload = reinterpret_cast<char *>(malloc(max_payload_size));
    fbs.reset(payload, max_payload_size);
  } else {
    payload = 0;
  }
}

template <typename T>
ActiveMessage<T>::~ActiveMessage(void)
{
  assert(!valid);
}

// operator-> gives access to the header structure
template <typename T>
T *ActiveMessage<T>::operator->(void)
{
  return &(args.header);
}

// variable payload can be written to in three ways:
//  (a) Realm-style serialization (currently eager)
template <typename T>
template <typename T2>
bool ActiveMessage<T>::operator<<(const T2& to_append)
{
  bool ok = (fbs << to_append);
  return ok;
}

//  (b) old memcpy-like behavior (using the various payload modes)
template <typename T>
void ActiveMessage<T>::add_payload(const void *data, size_t datalen,
				   int payload_mode /*= PAYLOAD_COPY*/)
{
  bool ok = fbs.append_bytes(data, datalen);
  assert(ok);
  if(payload_mode == PAYLOAD_FREE)
    free(const_cast<void *>(data));
}

//  (c) request for a pointer to write into (writes must be completed before
//       call to commit or cancel)
template <typename T>
void *ActiveMessage<T>::payload_ptr(size_t datalen)
{
  char *eob = payload + max_payload_size;
  char *curpos = eob - fbs.bytes_left();
  char *nextpos = curpos + datalen;
  fbs.reset(nextpos, eob - nextpos);
  return curpos;
}

class ActiveMessageSender {
public:
  ActiveMessageSender(const void *_args, size_t _arglen,
		      const void *_payload, size_t _payload_len)
    : prev_target(-1)
    , args(_args)
    , arglen(_arglen)
    , payload(_payload)
    , payload_len(_payload_len)
  {}

  void apply(NodeID target)
  {
    if(prev_target != -1)
      enqueue_message(prev_target, MSGID_NEW_ACTIVEMSG,
		      args, arglen,
		      payload, payload_len, PAYLOAD_COPY);
    prev_target = target;
  }

  void finish(int payload_mode)
  {
    if(prev_target != -1)
      enqueue_message(prev_target, MSGID_NEW_ACTIVEMSG,
		      args, arglen,
		      payload, payload_len, payload_mode);
  }

  NodeID prev_target;
  const void *args;
  size_t arglen;
  const void *payload;
  size_t payload_len;
};

// every active message must eventually be commit()'ed or cancel()'ed
template <typename T>
void ActiveMessage<T>::commit(void)
{
  assert(valid);

  args.set_magic();
  args.msgid = activemsg_handler_table.lookup_message_id<T>();
  args.sender = my_node_id;
  
  if(max_payload_size)
    args.payload_len = max_payload_size - fbs.bytes_left();
  else
    args.payload_len = 0;

  if(is_multicast) {
    ActiveMessageSender ams(&args, sizeof(args), payload, args.payload_len);
    targets.map(ams);
    ams.finish(PAYLOAD_FREE);
  } else {
    enqueue_message(target, MSGID_NEW_ACTIVEMSG,
		    &args, sizeof(args),
		    payload, args.payload_len, PAYLOAD_FREE, dest_payload_addr);
  }
  valid = false;
}

template <typename T>
void ActiveMessage<T>::cancel(void)
{
  assert(valid);
  if(max_payload_size)
    free(payload);
  valid = false;
}


////////////////////////////////////////////////////////////////////////
//
// class ActiveMessageHandlerTable
//

template <typename T>
ActiveMessageHandlerTable::MessageID ActiveMessageHandlerTable::lookup_message_id(void) const
{
#ifdef USE_GASNET
  // first convert the type name into a hash
  TypeHash h = 0;
  const char *name = typeid(T).name();
  while(*name)
    h = h * 73 + *name++;

  // now binary search to find the index within our list
  MessageID lo = 0;
  MessageID hi = handlers.size();

  while(lo < hi) {
    MessageID mid = (lo + hi) >> 1;
    if(h < handlers[mid].hash)
      hi = mid;
    else if(h > handlers[mid].hash)
      lo = mid + 1;
    else
      return mid;
  }
  assert(0);
#endif
  return 0;
}


////////////////////////////////////////////////////////////////////////
//
// class ActiveMessageHandlerReg<T, T2>
//

template <typename T, typename T2>
ActiveMessageHandlerReg<T, T2>::ActiveMessageHandlerReg(void)
{
  hash = 0;
  const char *name = typeid(T).name();
  while(*name)
    hash = hash * 73 + *name++;

  ActiveMessageHandlerTable::append_handler_reg(this);
}

template <typename T, typename T2>
ActiveMessageHandlerTable::MessageHandler ActiveMessageHandlerReg<T, T2>::get_handler(void) const
{
  return &handler_wrapper;
}

template <typename T, typename T2>
/*static*/ void ActiveMessageHandlerReg<T, T2>::handler_wrapper(NodeID sender, const void *header,
								const void *payload, size_t payload_size)
{
  T2::handle_message(sender, *reinterpret_cast<const T *>(header),
		     payload, payload_size);
}
