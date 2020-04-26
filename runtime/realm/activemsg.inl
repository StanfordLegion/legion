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

// INCLDUED FROM activemsg.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/activemsg.h"

// for name demangling
#ifdef REALM_HAVE_CXXABI_H
#include <cxxabi.h>
#endif

namespace Realm {
  
  ////////////////////////////////////////////////////////////////////////
  //
  // class ActiveMessage<T>
  //

  template <typename T>
  ActiveMessage<T>::ActiveMessage(NodeID _target,
				  size_t _max_payload_size /*= 0*/,
				  void *_dest_payload_addr /*= 0*/)
  {
    unsigned short msgid = activemsg_handler_table.lookup_message_id<T>();
    impl = Network::create_active_message_impl(_target,
					       msgid,
					       sizeof(T),
					       _max_payload_size,
					       _dest_payload_addr,
					       &inline_capacity,
					       INLINE_STORAGE);
    header = new(impl->header_base) T;
    fbs.reset(impl->payload_base, impl->payload_size);
  }
    
  template <typename T>
  ActiveMessage<T>::ActiveMessage(const Realm::NodeSet &_targets,
				  size_t _max_payload_size /*= 0*/)
  {
    unsigned short msgid = activemsg_handler_table.lookup_message_id<T>();
    impl = Network::create_active_message_impl(_targets,
					       msgid,
					       sizeof(T),
					       _max_payload_size,
					       &inline_capacity,
					       INLINE_STORAGE);
    header = new(impl->header_base) T;
    fbs.reset(impl->payload_base, impl->payload_size);
  }

  template <typename T>
  ActiveMessage<T>::~ActiveMessage(void)
  {
    assert(impl == 0);
  }

  // operator-> gives access to the header structure
  template <typename T>
  T *ActiveMessage<T>::operator->(void)
  {
    return header;
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

  template <typename T>
  void ActiveMessage<T>::add_payload(const void *data, size_t bytes_per_line,
				     size_t lines, size_t line_stride,
				     int payload_mode /*= PAYLOAD_COPY*/)
  {
    // detect case where 2d collapses to 1d
    if(line_stride == bytes_per_line) {
      bool ok = fbs.append_bytes(data, bytes_per_line * lines);
      assert(ok);
    } else {
      for(size_t i = 0; i < lines; i++) {
	bool ok = fbs.append_bytes((static_cast<const char *>(data) +
				    (i * line_stride)),
				   bytes_per_line);
	assert(ok);
      }
    }
    if(payload_mode == PAYLOAD_FREE)
      free(const_cast<void *>(data));
  }

  //  (c) request for a pointer to write into (writes must be completed before
  //       call to commit or cancel)
  template <typename T>
  void *ActiveMessage<T>::payload_ptr(size_t datalen)
  {
    char *eob = reinterpret_cast<char *>(impl->payload_base) + impl->payload_size;
    char *curpos = eob - fbs.bytes_left();
    char *nextpos = curpos + datalen;
    fbs.reset(nextpos, eob - nextpos);
    return curpos;
  }

  // every active message must eventually be commit()'ed or cancel()'ed
  template <typename T>
  void ActiveMessage<T>::commit(void)
  {
    assert(impl != 0);

    size_t act_payload_len;
    if(impl->payload_size)
      act_payload_len = impl->payload_size - fbs.bytes_left();
    else
      act_payload_len = 0;

    impl->commit(act_payload_len);

    // now tear things down
    header->~T();
    impl->~ActiveMessageImpl();
    impl = 0;
  }

  template <typename T>
  void ActiveMessage<T>::cancel(void)
  {
    assert(impl != 0);
    impl->cancel();
  
    // now tear things down
    header->~T();
    impl->~ActiveMessageImpl();
    impl = 0;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ActiveMessageHandlerTable
  //

  template <typename T>
  ActiveMessageHandlerTable::MessageID ActiveMessageHandlerTable::lookup_message_id(void) const
  {
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
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ActiveMessageHandlerReg<T, T2>
  //

  template <typename T, typename T2>
  ActiveMessageHandlerReg<T, T2>::ActiveMessageHandlerReg(void)
  {
    hash = 0;
    // we always hash with the mangled name, but try to demangle for debugging
    //  purposes
    const char *mangled_name = typeid(T).name();
    const char *c = mangled_name;
    while(*c)
      hash = hash * 73 + *c++;

#ifdef REALM_HAVE_CXXABI_H
    int status = -4;
    // let __cxa_demagle do a malloc - we have no idea how much to request
    //  ahead of time
    char *demangled = abi::__cxa_demangle(mangled_name, 0, 0, &status);
    if(status == 0) {
      // success
      name = demangled;
      must_free = true;
    } else
#endif
    {
      name = mangled_name;
      must_free = false;
    }

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

};
