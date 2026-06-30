/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
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
  // class ActiveMessage<T, INLINE_STORAGE>
  //

  template <typename T, size_t INLINE_STORAGE>
  ActiveMessage<T, INLINE_STORAGE>::ActiveMessage()
    : impl(0)
    , header(0)
  {}

  template <typename T, size_t INLINE_STORAGE>
  ActiveMessage<T, INLINE_STORAGE>::ActiveMessage(NodeID _target,
                                                  size_t _max_payload_size /*= 0*/)
    : impl(0)
  {
    init(_target, _max_payload_size);
  }

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::init(NodeID _target,
                                              size_t _max_payload_size /*= 0*/)
  {
    assert(impl == 0);
    if constexpr(!is_wrapped_with_frag_info<T>::value) {
      if(_max_payload_size > 0) {
        size_t net_max = Network::max_payload_size(sizeof(T), nullptr);
        if(_max_payload_size > net_max) {
          init_chunked(_target, _max_payload_size);
          return;
        }
      }
    }
    network_max_payload_ = 0;
    unsigned short msgid = activemsg_handler_table.lookup_message_id<T>();
    impl = Network::create_active_message_impl(_target, msgid, sizeof(T),
                                               _max_payload_size, 0, 0, 0,
                                               &inline_capacity, sizeof(inline_capacity));
    header = new(impl->header_base) T;
    fbs.reset(impl->payload_base, impl->payload_size);
  }

  template <typename T, size_t INLINE_STORAGE>
  ActiveMessage<T, INLINE_STORAGE>::ActiveMessage(NodeID _target,
                                                  size_t _max_payload_size,
                                                  const RemoteAddress &_dest_payload_addr)
    : impl(0)
  {
    init(_target, _max_payload_size, _dest_payload_addr);
  }

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::init(NodeID _target, size_t _max_payload_size,
                                              const RemoteAddress &_dest_payload_addr)
  {
    assert(impl == 0);
    network_max_payload_ = 0;
    unsigned short msgid = activemsg_handler_table.lookup_message_id<T>();
    impl = Network::create_active_message_impl(_target, msgid, sizeof(T),
                                               _max_payload_size, _dest_payload_addr,
                                               &inline_capacity, sizeof(inline_capacity));
    header = new(impl->header_base) T;
    fbs.reset(impl->payload_base, impl->payload_size);
  }

  template <typename T, size_t INLINE_STORAGE>
  ActiveMessage<T, INLINE_STORAGE>::ActiveMessage(const Realm::NodeSet &_targets,
                                                  size_t _max_payload_size /*= 0*/)
    : impl(0)
  {
    init(_targets, _max_payload_size);
  }

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::init(const Realm::NodeSet &_targets,
                                              size_t _max_payload_size /*= 0*/)
  {
    assert(impl == 0);
    if constexpr(!is_wrapped_with_frag_info<T>::value) {
      if(_max_payload_size > 0) {
        size_t net_max = Network::max_payload_size(sizeof(T), nullptr);
        if(_max_payload_size > net_max) {
          init_chunked(_targets, _max_payload_size);
          return;
        }
      }
    }
    network_max_payload_ = 0;
    unsigned short msgid = activemsg_handler_table.lookup_message_id<T>();
    impl = Network::create_active_message_impl(_targets, msgid, sizeof(T),
                                               _max_payload_size, 0, 0, 0,
                                               &inline_capacity, sizeof(inline_capacity));
    header = new(impl->header_base) T;
    fbs.reset(impl->payload_base, impl->payload_size);
  }

  template <typename T, size_t INLINE_STORAGE>
  ActiveMessage<T, INLINE_STORAGE>::ActiveMessage(NodeID _target, const void *_data,
                                                  size_t _datalen)
    : impl(0)
  {
    init(_target, _data, _datalen);
  }

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::init(NodeID _target, const void *_data,
                                              size_t _datalen)
  {
    assert(impl == 0);
    if constexpr(!is_wrapped_with_frag_info<T>::value) {
      if(_datalen > 0) {
        size_t net_max = Network::max_payload_size(sizeof(T), _data);
        if(_datalen > net_max) {
          init_chunked_data(_target, _data, _datalen);
          return;
        }
      }
    }
    network_max_payload_ = 0;
    unsigned short msgid = activemsg_handler_table.lookup_message_id<T>();
    impl =
        Network::create_active_message_impl(_target, msgid, sizeof(T), _datalen, _data, 0,
                                            0, &inline_capacity, sizeof(inline_capacity));
    header = new(impl->header_base) T;
  }

  template <typename T, size_t INLINE_STORAGE>
  ActiveMessage<T, INLINE_STORAGE>::ActiveMessage(NodeID _target,
                                                  const LocalAddress &_src_payload_addr,
                                                  size_t _datalen,
                                                  const RemoteAddress &_dest_payload_addr)
    : impl(0)
  {
    init(_target, _src_payload_addr, _datalen, _dest_payload_addr);
  }

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::init(NodeID _target,
                                              const LocalAddress &_src_payload_addr,
                                              size_t _datalen,
                                              const RemoteAddress &_dest_payload_addr)
  {
    assert(impl == 0);
    network_max_payload_ = 0;
    unsigned short msgid = activemsg_handler_table.lookup_message_id<T>();
    impl = Network::create_active_message_impl(
        _target, msgid, sizeof(T), _datalen, _src_payload_addr, 0, 0, _dest_payload_addr,
        &inline_capacity, sizeof(inline_capacity));
    header = new(impl->header_base) T;
  }

  template <typename T, size_t INLINE_STORAGE>
  ActiveMessage<T, INLINE_STORAGE>::ActiveMessage(const Realm::NodeSet &_targets,
                                                  const void *_data, size_t _datalen)
    : impl(0)
  {
    init(_targets, _data, _datalen);
  }

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::init(const Realm::NodeSet &_targets,
                                              const void *_data, size_t _datalen)
  {
    assert(impl == 0);
    if constexpr(!is_wrapped_with_frag_info<T>::value) {
      if(_datalen > 0) {
        size_t net_max = Network::max_payload_size(sizeof(T), _data);
        if(_datalen > net_max) {
          init_chunked_data(_targets, _data, _datalen);
          return;
        }
      }
    }
    network_max_payload_ = 0;
    unsigned short msgid = activemsg_handler_table.lookup_message_id<T>();
    impl = Network::create_active_message_impl(_targets, msgid, sizeof(T), _datalen,
                                               _data, 0, 0, &inline_capacity,
                                               sizeof(inline_capacity));
    header = new(impl->header_base) T;
  }

  template <typename T, size_t INLINE_STORAGE>
  ActiveMessage<T, INLINE_STORAGE>::ActiveMessage(NodeID _target,
                                                  const LocalAddress &_src_payload_addr,
                                                  size_t _bytes_per_line, size_t _lines,
                                                  size_t _line_stride,
                                                  const RemoteAddress &_dest_payload_addr)
    : impl(0)
  {
    init(_target, _src_payload_addr, _bytes_per_line, _lines, _line_stride,
         _dest_payload_addr);
  }

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::init(NodeID _target,
                                              const LocalAddress &_src_payload_addr,
                                              size_t _bytes_per_line, size_t _lines,
                                              size_t _line_stride,
                                              const RemoteAddress &_dest_payload_addr)
  {
    assert(impl == 0);
    network_max_payload_ = 0;
    unsigned short msgid = activemsg_handler_table.lookup_message_id<T>();
    impl = Network::create_active_message_impl(
        _target, msgid, sizeof(T), _bytes_per_line * _lines, _src_payload_addr, _lines,
        _line_stride, _dest_payload_addr, &inline_capacity, sizeof(inline_capacity));
    header = new(impl->header_base) T;
  }

  template <typename T, size_t INLINE_STORAGE>
  ActiveMessage<T, INLINE_STORAGE>::~ActiveMessage(void)
  {
    assert((impl == 0) && (network_max_payload_ == 0));
  }

  template <typename T, size_t INLINE_STORAGE>
  /*static*/ size_t
  ActiveMessage<T, INLINE_STORAGE>::recommended_max_payload(NodeID target,
                                                            bool with_congestion)
  {
    return Network::recommended_max_payload(target, with_congestion, sizeof(T));
  }

  template <typename T, size_t INLINE_STORAGE>
  /*static*/ size_t
  ActiveMessage<T, INLINE_STORAGE>::recommended_max_payload(const NodeSet &targets,
                                                            bool with_congestion)
  {
    return Network::recommended_max_payload(targets, with_congestion, sizeof(T));
  }

  template <typename T, size_t INLINE_STORAGE>
  /*static*/ size_t ActiveMessage<T, INLINE_STORAGE>::recommended_max_payload(
      NodeID target, const RemoteAddress &dest_payload_addr, bool with_congestion)
  {
    return Network::recommended_max_payload(target, dest_payload_addr, with_congestion,
                                            sizeof(T));
  }

  template <typename T, size_t INLINE_STORAGE>
  /*static*/ size_t ActiveMessage<T, INLINE_STORAGE>::recommended_max_payload(
      NodeID target, const void *data, size_t bytes_per_line, size_t lines,
      size_t line_stride, bool with_congestion)
  {
    return Network::recommended_max_payload(target, data, bytes_per_line, lines,
                                            line_stride, with_congestion, sizeof(T));
  }

  template <typename T, size_t INLINE_STORAGE>
  /*static*/ size_t ActiveMessage<T, INLINE_STORAGE>::recommended_max_payload(
      const NodeSet &targets, const void *data, size_t bytes_per_line, size_t lines,
      size_t line_stride, bool with_congestion)
  {
    return Network::recommended_max_payload(targets, data, bytes_per_line, lines,
                                            line_stride, with_congestion, sizeof(T));
  }

  template <typename T, size_t INLINE_STORAGE>
  /*static*/ size_t ActiveMessage<T, INLINE_STORAGE>::recommended_max_payload(
      NodeID target, const LocalAddress &src_payload_addr, size_t bytes_per_line,
      size_t lines, size_t line_stride, const RemoteAddress &dest_payload_addr,
      bool with_congestion)
  {
    return Network::recommended_max_payload(target, src_payload_addr, bytes_per_line,
                                            lines, line_stride, dest_payload_addr,
                                            with_congestion, sizeof(T));
  }

  // operator-> gives access to the header structure
  template <typename T, size_t INLINE_STORAGE>
  T *ActiveMessage<T, INLINE_STORAGE>::operator->(void)
  {
    return header;
  }
  template <typename T, size_t INLINE_STORAGE>
  T &ActiveMessage<T, INLINE_STORAGE>::operator*(void)
  {
    return *header;
  }

  // variable payload can be written to in three ways:
  //  (a) Realm-style serialization (currently eager)
  template <typename T, size_t INLINE_STORAGE>
  template <typename T2>
  bool ActiveMessage<T, INLINE_STORAGE>::operator<<(const T2 &to_append)
  {
    bool ok = (fbs << to_append);
    return ok;
  }

  //  (b) old memcpy-like behavior (using the various payload modes)
  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::add_payload(const void *data, size_t datalen,
                                                     int payload_mode /*= PAYLOAD_COPY*/)
  {
    bool ok = fbs.append_bytes(data, datalen);
    assert(ok);
    if(payload_mode == PAYLOAD_FREE)
      free(const_cast<void *>(data));
  }

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::add_payload(const void *data,
                                                     size_t bytes_per_line, size_t lines,
                                                     size_t line_stride,
                                                     int payload_mode /*= PAYLOAD_COPY*/)
  {
    // detect case where 2d collapses to 1d
    if(line_stride == bytes_per_line) {
      bool ok = fbs.append_bytes(data, bytes_per_line * lines);
      assert(ok);
    } else {
      for(size_t i = 0; i < lines; i++) {
        bool ok = fbs.append_bytes((static_cast<const char *>(data) + (i * line_stride)),
                                   bytes_per_line);
        assert(ok);
      }
    }
    if(payload_mode == PAYLOAD_FREE)
      free(const_cast<void *>(data));
  }

  //  (c) request for a pointer to write into (writes must be completed before
  //       call to commit or cancel)
  template <typename T, size_t INLINE_STORAGE>
  void *ActiveMessage<T, INLINE_STORAGE>::payload_ptr(size_t datalen)
  {
    if(network_max_payload_ != 0) {
      assert(!chunk_alloc_.empty());
      char *buf_begin = reinterpret_cast<char *>(chunk_alloc_.data());
      char *eob = buf_begin + chunk_src_datalen_;
      char *curpos = eob - fbs.bytes_left();
      char *nextpos = curpos + datalen;
      fbs.reset(nextpos, eob - nextpos);
      return curpos;
    }
    char *eob = reinterpret_cast<char *>(impl->payload_base) + impl->payload_size;
    char *curpos = eob - fbs.bytes_left();
    char *nextpos = curpos + datalen;
    fbs.reset(nextpos, eob - nextpos);
    return curpos;
  }

  template <typename T, size_t INLINE_STORAGE>
  template <typename CALLABLE>
  void ActiveMessage<T, INLINE_STORAGE>::add_local_completion(const CALLABLE &callable)
  {
    assert(impl != 0);

    size_t bytes = sizeof(CompletionCallback<CALLABLE>);
    // round up
    bytes = (((bytes - 1) / CompletionCallbackBase::ALIGNMENT) + 1) *
            CompletionCallbackBase::ALIGNMENT;
    void *ptr = impl->add_local_completion(bytes);
    new(ptr) CompletionCallback<CALLABLE>(callable);
  }

  template <typename T, size_t INLINE_STORAGE>
  template <typename CALLABLE>
  void ActiveMessage<T, INLINE_STORAGE>::add_remote_completion(const CALLABLE &callable)
  {
    assert(impl != 0);

    size_t bytes = sizeof(CompletionCallback<CALLABLE>);
    // round up
    bytes = (((bytes - 1) / CompletionCallbackBase::ALIGNMENT) + 1) *
            CompletionCallbackBase::ALIGNMENT;
    void *ptr = impl->add_remote_completion(bytes);
    new(ptr) CompletionCallback<CALLABLE>(callable);
  }

  // every active message must eventually be commit()'ed or cancel()'ed
  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::commit(void)
  {
    if(network_max_payload_ != 0) {
      commit_chunked();
      return;
    }

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

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::cancel(void)
  {
    if(network_max_payload_ != 0) {
      // chunked mode - just clean up the local state
      header->~T();
      header = 0;
      chunk_alloc_.clear();
      chunk_alloc_.shrink_to_fit();
      chunk_targets_.clear();
      chunk_src_data_ = nullptr;
      chunk_src_datalen_ = 0;
      network_max_payload_ = 0;
      // impl is 0 in chunked mode, so match the normal-path postcondition
      return;
    }
    assert(impl != 0);
    impl->cancel();

    // now tear things down
    header->~T();
    impl->~ActiveMessageImpl();
    impl = 0;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class CompletionCallback<CALLABLE>
  //

  template <typename CALLABLE>
  CompletionCallback<CALLABLE>::CompletionCallback(const CALLABLE &_callable)
    : callable(_callable)
  {}

  template <typename CALLABLE>
  void CompletionCallback<CALLABLE>::invoke()
  {
    callable();
  }

  template <typename CALLABLE>
  size_t CompletionCallback<CALLABLE>::size() const
  {
    // double-check that our alignment is satisfactory
#ifdef DEBUG_REALM
    assert(REALM_ALIGNOF(CompletionCallback<CALLABLE>) <= ALIGNMENT);
#endif
    size_t bytes = sizeof(CompletionCallback<CALLABLE>);
    // round up to ALIGNMENT boundary
    bytes = (((bytes - 1) / ALIGNMENT) + 1) * ALIGNMENT;
    return bytes;
  }

  template <typename CALLABLE>
  CompletionCallbackBase *CompletionCallback<CALLABLE>::clone_at(void *p) const
  {
    return new(p) CompletionCallback<CALLABLE>(callable);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ActiveMessageHandlerTable
  //

  template <typename T>
  ActiveMessageHandlerTable::MessageID
  ActiveMessageHandlerTable::lookup_message_id(void) const
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
    return 0;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ActiveMessageHandlerReg<T, T2>
  //

  template <typename, typename = void>
  struct has_frag_info : std::false_type {};
  template <typename T>
  struct has_frag_info<T, std::void_t<decltype(std::declval<T>().frag_info)>>
    : std::true_type {};
  template <typename T>
  constexpr inline bool has_frag_info_v = has_frag_info<T>::value;

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

    if constexpr(has_frag_info_v<T>) {
      extract_frag_info = [](const void *hdr) -> const FragmentInfo & {
        return static_cast<const T *>(hdr)->frag_info;
      };
    }

    ActiveMessageHandlerTable::append_handler_reg(this);

    // automatically register a WrappedWithFragInfo<T> handler so that
    //  fragmented messages of any type can be reassembled
    if constexpr(!is_wrapped_with_frag_info<T>::value &&
                 std::is_same<T, T2>::value) {
      auto *wrapped_reg = new ActiveMessageHandlerReg<WrappedWithFragInfo<T>, T>();
      (void)wrapped_reg; // leaked intentionally - lives for program lifetime
    }
  }

  template <typename T, typename T2>
  ActiveMessageHandlerReg<T, T2>::~ActiveMessageHandlerReg(void)
  {
    if(must_free)
      free(const_cast<char *>(name));
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // ActiveMessage<T> chunked mode helpers
  //

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::init_chunked(NodeID _target,
                                                      size_t _max_payload_size)
  {
    if constexpr(is_wrapped_with_frag_info<T>::value) {
      assert(0 && "init_chunked called on WrappedWithFragInfo type");
    } else {
      size_t wrapped_hdr_size = sizeof(WrappedWithFragInfo<T>);
      network_max_payload_ = Network::max_payload_size(wrapped_hdr_size, nullptr);
      assert(network_max_payload_ > 0);

      chunk_targets_.add(_target);
      chunk_alloc_.resize(_max_payload_size);
      chunk_src_data_ = chunk_alloc_.data();
      chunk_src_datalen_ = _max_payload_size;
      header = new(&inline_capacity) T;
      fbs.reset(chunk_alloc_.data(), _max_payload_size);
    }
  }

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::init_chunked(const NodeSet &_targets,
                                                      size_t _max_payload_size)
  {
    if constexpr(is_wrapped_with_frag_info<T>::value) {
      assert(0 && "init_chunked called on WrappedWithFragInfo type");
    } else {
      size_t wrapped_hdr_size = sizeof(WrappedWithFragInfo<T>);
      network_max_payload_ = Network::max_payload_size(wrapped_hdr_size, nullptr);
      assert(network_max_payload_ > 0);

      chunk_targets_ = _targets;
      chunk_alloc_.resize(_max_payload_size);
      chunk_src_data_ = chunk_alloc_.data();
      chunk_src_datalen_ = _max_payload_size;
      header = new(&inline_capacity) T;
      fbs.reset(chunk_alloc_.data(), _max_payload_size);
    }
  }

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::init_chunked_data(NodeID _target,
                                                           const void *_data,
                                                           size_t _datalen)
  {
    if constexpr(is_wrapped_with_frag_info<T>::value) {
      assert(0 && "init_chunked_data called on WrappedWithFragInfo type");
    } else {
      // compute the per-chunk payload capacity using the wrapped header size
      // pass the caller's data pointer so the backend can check registration
      size_t wrapped_hdr_size = sizeof(WrappedWithFragInfo<T>);
      network_max_payload_ = Network::max_payload_size(wrapped_hdr_size, _data);
      assert(network_max_payload_ > 0);

      chunk_targets_.add(_target);
      chunk_src_data_ = _data;
      chunk_src_datalen_ = _datalen;
      // place the header in inline_capacity (it's not sent to the network yet)
      header = new(&inline_capacity) T;
      // no fbs/chunk_buffer_ needed - data is referenced directly
      // impl stays null in chunked mode
    }
  }

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::init_chunked_data(const NodeSet &_targets,
                                                           const void *_data,
                                                           size_t _datalen)
  {
    if constexpr(is_wrapped_with_frag_info<T>::value) {
      assert(0 && "init_chunked_data called on WrappedWithFragInfo type");
    } else {
      size_t wrapped_hdr_size = sizeof(WrappedWithFragInfo<T>);
      network_max_payload_ = Network::max_payload_size(wrapped_hdr_size, _data);
      assert(network_max_payload_ > 0);

      chunk_targets_ = _targets;
      chunk_src_data_ = _data;
      chunk_src_datalen_ = _datalen;
      header = new(&inline_capacity) T;
    }
  }

  template <typename T, size_t INLINE_STORAGE>
  void ActiveMessage<T, INLINE_STORAGE>::commit_chunked(void)
  {
    if constexpr(is_wrapped_with_frag_info<T>::value) {
      assert(0 && "commit_chunked called on WrappedWithFragInfo type");
    } else {
      assert(chunk_src_data_ != nullptr);
      if(!chunk_alloc_.empty()) {
        // buffer mode: actual payload is what was written via fbs
        chunk_src_datalen_ -= fbs.bytes_left();
      }
      const char *payload_data = static_cast<const char *>(chunk_src_data_);
      size_t total_payload = chunk_src_datalen_;

      // msg_id only needs to be unique per-sender, so using my_node_id is fine
      //  for both unicast and multicast chunked sends
      uint64_t msg_id = next_chunk_message_id(Network::my_node_id);
      size_t max_chunk = network_max_payload_;
      uint32_t total_chunks =
          static_cast<uint32_t>((total_payload + max_chunk - 1) / max_chunk);
      if(total_chunks == 0) {
        total_chunks = 1;
      }

      size_t offset = 0;
      for(uint32_t chunk_id = 0; chunk_id < total_chunks; ++chunk_id) {
        size_t chunk_size = std::min(max_chunk, total_payload - offset);

        ActiveMessage<WrappedWithFragInfo<T>> chunk_msg(chunk_targets_, chunk_size);
        chunk_msg->frag_info = {chunk_id, total_chunks, msg_id};
        chunk_msg->user = *header;
        if(chunk_size > 0) {
          chunk_msg.add_payload(payload_data + offset, chunk_size);
        }
        chunk_msg.commit();

        offset += chunk_size;
      }

      // clean up
      header->~T();
      header = 0;
      chunk_alloc_.clear();
      chunk_alloc_.shrink_to_fit();
      chunk_targets_.clear();
      chunk_src_data_ = nullptr;
      chunk_src_datalen_ = 0;
      network_max_payload_ = 0;
    }
  }

  template <typename T, size_t INLINE_STORAGE>
  /*static*/ uint64_t
  ActiveMessage<T, INLINE_STORAGE>::next_chunk_message_id(NodeID node_id)
  {
    static std::atomic<uint64_t> counter{0};
    uint64_t local = counter.fetch_add(1, std::memory_order_relaxed);
    return (static_cast<uint64_t>(node_id) << 48) | (local & ((1ULL << 48) - 1));
  }

  namespace HandlerWrappers {
    // this type only exists if you can supply a value of the right type
    template <typename T, T fnptr>
    struct HasRightType {};

    template <typename T,
              void (*HANDLER)(NodeID, const T &, const void *, size_t, TimeLimit)>
    static void wrap_handler(NodeID sender, const void *header, const void *payload,
                             size_t payload_size, TimeLimit work_until)
    {
      (*HANDLER)(sender, *reinterpret_cast<const T *>(header), payload, payload_size,
                 work_until);
    }

    template <typename T, void (*HANDLER)(NodeID, const T &, const void *, size_t)>
    static void wrap_handler_notimeout(NodeID sender, const void *header,
                                       const void *payload, size_t payload_size)
    {
      (*HANDLER)(sender, *reinterpret_cast<const T *>(header), payload, payload_size);
    }

    template <typename T,
              bool (*HANDLER)(NodeID, const T &, const void *, size_t, TimeLimit)>
    static bool wrap_handler_inline(NodeID sender, const void *header,
                                    const void *payload, size_t payload_size,
                                    TimeLimit work_until)
    {
      return (*HANDLER)(sender, *reinterpret_cast<const T *>(header), payload,
                        payload_size, work_until);
    }

    template <typename UserHdr,
              void (*HANDLER)(NodeID, const UserHdr &, const void *, size_t, TimeLimit)>
    static void wrap_handler_unwrap(NodeID sender, const void *header,
                                    const void *payload, size_t payload_size,
                                    TimeLimit work_until)
    {
      const auto &inner =
          reinterpret_cast<const WrappedWithFragInfo<UserHdr> *>(header)->user;
      (*HANDLER)(sender, inner, payload, payload_size, work_until);
    }

    template <typename UserHdr,
              void (*HANDLER)(NodeID, const UserHdr &, const void *, size_t)>
    static void wrap_handler_unwrap_notimeout(NodeID sender, const void *header,
                                              const void *payload, size_t payload_size)
    {
      const auto &inner =
          reinterpret_cast<const WrappedWithFragInfo<UserHdr> *>(header)->user;
      (*HANDLER)(sender, inner, payload, payload_size);
    }

    template <typename UserHdr,
              bool (*HANDLER)(NodeID, const UserHdr &, const void *, size_t, TimeLimit)>
    static bool wrap_handler_unwrap_inline(NodeID sender, const void *header,
                                           const void *payload, size_t payload_size,
                                           TimeLimit work_until)
    {
      const auto &inner =
          reinterpret_cast<const WrappedWithFragInfo<UserHdr> *>(header)->user;
      return (*HANDLER)(sender, inner, payload, payload_size, work_until);
    }

    // thsee overloads only exist if we have a handle_message method and it
    //  has the desired type
    template <typename T, typename T2>
    ActiveMessageHandlerTable::MessageHandler
    get_handler(HasRightType<void (*)(NodeID, const T &, const void *, size_t, TimeLimit),
                             &T2::handle_message> *)
    {
      return &wrap_handler<T, &T2::handle_message>;
    }
    template <typename T, typename T2>
    ActiveMessageHandlerTable::MessageHandlerNoTimeout
    get_handler_notimeout(HasRightType<void (*)(NodeID, const T &, const void *, size_t),
                                       &T2::handle_message> *)
    {
      return &wrap_handler_notimeout<T, &T2::handle_message>;
    }
    template <typename T, typename T2>
    ActiveMessageHandlerTable::MessageHandlerInline get_handler_inline(
        HasRightType<bool (*)(NodeID, const T &, const void *, size_t, TimeLimit),
                     &T2::handle_inline> *)
    {
      return &wrap_handler_inline<T, &T2::handle_inline>;
    }

    // these return null pointers if the ones above do not match
    template <typename T, typename T2>
    ActiveMessageHandlerTable::MessageHandler get_handler(...)
    {
      return 0;
    }
    template <typename T, typename T2>
    ActiveMessageHandlerTable::MessageHandlerNoTimeout get_handler_notimeout(...)
    {
      return 0;
    }
    template <typename T, typename T2>
    ActiveMessageHandlerTable::MessageHandlerInline get_handler_inline(...)
    {
      return 0;
    }

    template <typename T, typename T2,
              typename std::enable_if<
                  std::is_same<T, WrappedWithFragInfo<T2>>::value>::type * = nullptr>
    ActiveMessageHandlerTable::MessageHandler get_handler(
        HasRightType<void (*)(NodeID, const T2 &, const void *, size_t, TimeLimit),
                     &T2::handle_message> *)
    {
      return &wrap_handler_unwrap<T2, &T2::handle_message>;
    }

    template <typename T, typename T2,
              typename std::enable_if<
                  std::is_same<T, WrappedWithFragInfo<T2>>::value>::type * = nullptr>
    ActiveMessageHandlerTable::MessageHandlerNoTimeout
    get_handler_notimeout(HasRightType<void (*)(NodeID, const T2 &, const void *, size_t),
                                       &T2::handle_message> *)
    {
      return &wrap_handler_unwrap_notimeout<T2, &T2::handle_message>;
    }

    template <typename T, typename T2,
              typename std::enable_if<
                  std::is_same<T, WrappedWithFragInfo<T2>>::value>::type * = nullptr>
    ActiveMessageHandlerTable::MessageHandlerInline get_handler_inline(
        HasRightType<bool (*)(NodeID, const T2 &, const void *, size_t, TimeLimit),
                     &T2::handle_inline> *)
    {
      return &wrap_handler_unwrap_inline<T2, &T2::handle_inline>;
    }

  }; // namespace HandlerWrappers

  template <typename T, typename T2>
  ActiveMessageHandlerTable::MessageHandler
  ActiveMessageHandlerReg<T, T2>::get_handler(void) const
  {
    return HandlerWrappers::template get_handler<T, T2>(0);
  }

  template <typename T, typename T2>
  ActiveMessageHandlerTable::MessageHandlerNoTimeout
  ActiveMessageHandlerReg<T, T2>::get_handler_notimeout(void) const
  {
    return HandlerWrappers::template get_handler_notimeout<T, T2>(0);
  }

  template <typename T, typename T2>
  ActiveMessageHandlerTable::MessageHandlerInline
  ActiveMessageHandlerReg<T, T2>::get_handler_inline(void) const
  {
    return HandlerWrappers::template get_handler_inline<T, T2>(0);
  }

}; // namespace Realm
