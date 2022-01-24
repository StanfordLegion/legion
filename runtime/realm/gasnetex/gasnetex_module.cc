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

// GASNet-EX network module implementation for Realm

#include "realm/realm_config.h"

#include "realm/network.h"

#include "realm/gasnetex/gasnetex_module.h"
#include "realm/gasnetex/gasnetex_internal.h"

#include "realm/runtime_impl.h"
#include "realm/mem_impl.h"
#include "realm/logging.h"
#include "realm/transfer/ib_memory.h"

#ifdef REALM_GASNETEX_MODULE_DYNAMIC
REGISTER_REALM_NETWORK_MODULE_DYNAMIC(Realm::GASNetEXModule);
#endif

namespace Realm {

  Logger log_gex("gex");

  // bit twiddling tricks
  static bool is_pow2(size_t val)
  {
    return ((val & (val - 1)) == 0);
  }

  template <typename T>
  static T roundup_pow2(T val, size_t alignment)
  {
    assert(is_pow2(alignment));
    return ((val + alignment - 1) & ~(alignment - 1));
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNetEXRemoteMemory
  //

  class GASNetEXRemoteMemory : public RemoteMemory {
  public:
    GASNetEXRemoteMemory(Memory _me, size_t _size, Memory::Kind k,
			 uintptr_t _regbase, gex_EP_Index_t _ep_index);

    virtual void get_bytes(off_t offset, void *dst, size_t size);
    virtual void put_bytes(off_t offset, const void *src, size_t size);

    virtual bool get_remote_addr(off_t offset, RemoteAddress& remote_addr);

  protected:
    uintptr_t regbase;
    gex_EP_Index_t ep_index;
  };

  GASNetEXRemoteMemory::GASNetEXRemoteMemory(Memory _me, size_t _size,
					     Memory::Kind k,
					     uintptr_t _regbase,
					     gex_EP_Index_t _ep_index)
    : RemoteMemory(_me, _size, k, MKIND_RDMA)
    , regbase(_regbase)
    , ep_index(_ep_index)
  {}

  void GASNetEXRemoteMemory::get_bytes(off_t offset, void *dst, size_t size)
  {
    // TODO
    assert(0);
  }

  void GASNetEXRemoteMemory::put_bytes(off_t offset,
				      const void *src, size_t size)
  {
    // TODO
    assert(0);
  }

  bool GASNetEXRemoteMemory::get_remote_addr(off_t offset, RemoteAddress& remote_addr)
  {
    remote_addr.ptr = regbase + offset;
    remote_addr.extra = ep_index;
    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNetEXIBMemory
  //

  class GASNetEXIBMemory : public IBMemory {
  public:
    GASNetEXIBMemory(Memory _me, size_t _size, Memory::Kind k,
		     uintptr_t _regbase, gex_EP_Index_t _ep_index);

    virtual bool get_remote_addr(off_t offset, RemoteAddress& remote_addr);

  protected:
    uintptr_t regbase;
    gex_EP_Index_t ep_index;
  };

  GASNetEXIBMemory::GASNetEXIBMemory(Memory _me, size_t _size, Memory::Kind k,
				     uintptr_t _regbase,
				     gex_EP_Index_t _ep_index)
    : IBMemory(_me, _size, MKIND_REMOTE, k, 0, 0)
    , regbase(_regbase)
    , ep_index(_ep_index)
  {}

  bool GASNetEXIBMemory::get_remote_addr(off_t offset, RemoteAddress& remote_addr)
  {
    remote_addr.ptr = regbase + offset;
    remote_addr.extra = ep_index;
    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNetEXMessageImpl
  //

  class GASNetEXMessageImpl : public ActiveMessageImpl {
  public:
    GASNetEXMessageImpl(GASNetEXInternal *_internal,
			NodeID _target,
			unsigned short _msgid,
			size_t _header_size,
			size_t _max_payload_size,
			const void *_src_payload_addr,
			size_t _src_payload_lines,
			size_t _src_payload_line_stride,
			uintptr_t _dest_payload_addr,
			gex_EP_Index_t _dest_ep_index);
    GASNetEXMessageImpl(GASNetEXInternal *_internal,
			const Realm::NodeSet &_targets,
			unsigned short _msgid,
			size_t _header_size,
			size_t _max_payload_size,
			const void *_src_payload_addr,
			size_t _src_payload_lines,
			size_t _src_payload_line_stride);

    virtual ~GASNetEXMessageImpl();

    virtual void *add_local_completion(size_t size);
    virtual void *add_remote_completion(size_t size);

    virtual void commit(size_t act_payload_size);
    virtual void cancel();

  protected:
    GASNetEXInternal *internal;
    NodeID target;
    Realm::NodeSet targets;
    bool is_multicast;
    unsigned short msgid;
    const void *src_payload_addr;
    size_t src_payload_lines;
    size_t src_payload_line_stride;
    PreparedMessage *msg;
    size_t header_size;
    PendingCompletion *comp;
    static const size_t INLINE_SIZE = 128;
    gex_AM_Arg_t msg_data[INLINE_SIZE / sizeof(gex_AM_Arg_t)];
#if 0
    size_t header_size;
    struct FullHeader : public BaseMedium {
      unsigned short msgid;
      unsigned short sender;
      unsigned payload_len;
      unsigned long msg_header;  // type is unknown
    };
    FullHeader args; // must be last thing
#endif
  };

  GASNetEXMessageImpl::GASNetEXMessageImpl(GASNetEXInternal *_internal,
					   NodeID _target,
					   unsigned short _msgid,
					   size_t _header_size,
					   size_t _max_payload_size,
					   const void *_src_payload_addr,
					   size_t _src_payload_lines,
					   size_t _src_payload_line_stride,
					   uintptr_t _dest_payload_addr,
					   gex_EP_Index_t _dest_ep_index)
    : internal(_internal)
    , target(_target)
    , is_multicast(false)
    , msgid(_msgid)
    , src_payload_addr(_src_payload_addr)
    , src_payload_lines(_src_payload_lines)
    , src_payload_line_stride(_src_payload_line_stride)
    , comp(nullptr)
  {
    assert(_header_size <= INLINE_SIZE);
    header_size = _header_size;
    header_base = &msg_data[0]; // may be modified by prepare_message below
    size_t header_padded = roundup_pow2(header_size, 8);

    payload_size = _max_payload_size;
    if(payload_size > 0) {
      if((src_payload_addr != nullptr) && (src_payload_lines <= 1)) {
	// contiguous source data can be offered up directly
	payload_base = const_cast<void *>(src_payload_addr);
      } else if((header_padded + payload_size) <= INLINE_SIZE) {
	// offer up the rest of our internal storage for the payload
	payload_base = &msg_data[header_padded / sizeof(gex_AM_Arg_t)];
      } else {
	// internal logic will have to find us some memory to use
	payload_base = nullptr;
      }
    } else
      payload_base = nullptr;  // no payload

    msg = internal->prepare_message(target, _dest_ep_index, msgid,
				    header_base, header_size,
				    payload_base, payload_size,
				    _dest_payload_addr);
  }

  GASNetEXMessageImpl::GASNetEXMessageImpl(GASNetEXInternal *_internal,
					   const Realm::NodeSet &_targets,
					   unsigned short _msgid,
					   size_t _header_size,
					   size_t _max_payload_size,
					   const void *_src_payload_addr,
					   size_t _src_payload_lines,
					   size_t _src_payload_line_stride)
    : internal(_internal)
    , targets(_targets)
    , is_multicast(true)
    , msgid(_msgid)
    , src_payload_addr(_src_payload_addr)
    , src_payload_lines(_src_payload_lines)
    , src_payload_line_stride(_src_payload_line_stride)
    , comp(nullptr)
  {
    // for multicast messages, we store the payload in a temp buffer
    assert(_header_size <= INLINE_SIZE);
    header_size = _header_size;
    header_base = &msg_data[0];
    size_t header_padded = roundup_pow2(header_size, 8);

    payload_size = _max_payload_size;
    if(payload_size > 0) {
      if(src_payload_addr != nullptr) {
	// use the caller's storage for now
	payload_base = const_cast<void *>(src_payload_addr);
      } else if((header_padded + payload_size) <= INLINE_SIZE) {
	// we can use the rest of our internal storage
	payload_base = &msg_data[header_padded / sizeof(gex_AM_Arg_t)];
      } else {
	// have to dynamically allocate storage
	payload_base = malloc(payload_size);
	assert(payload_base != 0);
      }
    } else
      payload_base = nullptr;
  }

  GASNetEXMessageImpl::~GASNetEXMessageImpl()
  {
  }

  void *GASNetEXMessageImpl::add_local_completion(size_t size)
  {
    // if we don't already have a pending completion object, get one
    if(!comp)
      comp = internal->get_available_comp();
    return comp->add_local_completion(size);
  }

  void *GASNetEXMessageImpl::add_remote_completion(size_t size)
  {
    // if we don't already have a pending completion object, get one
    if(!comp)
      comp = internal->get_available_comp();
    return comp->add_remote_completion(size);
  }

  void GASNetEXMessageImpl::commit(size_t act_payload_size)
  {
    if(is_multicast) {
      // if/when we build a tree for multicast, expected local may be
      //  lower than expected remote
      unsigned exp_local = targets.size();
      unsigned exp_remote = exp_local;
      if(comp && !comp->mark_ready(exp_local, exp_remote))
	comp = nullptr;

      for(NodeID tgt : targets) {
	void *act_header = header_base;
	// can't use src payload directly if it's 2d
	void *act_payload = ((src_payload_lines <= 1) ? payload_base : nullptr);

	PreparedMessage *msg;
	msg = internal->prepare_message(tgt, 0 /*always prim EP*/, msgid,
					act_header, header_size,
					act_payload, act_payload_size,
					0 /* no RDMA destination*/);

	if(act_header != header_base)
	  memcpy(act_header, header_base, header_size);
	if(act_payload != payload_base) {
	  if(src_payload_lines > 1) {
	    // copy line by line
	    size_t bytes_per_line = act_payload_size / src_payload_lines;
	    for(size_t i = 0; i < src_payload_lines; i++)
	      memcpy(static_cast<char *>(act_payload) + (i * bytes_per_line),
		     static_cast<const char *>(payload_base) + (i * src_payload_line_stride),
		     bytes_per_line);
	  } else {
	    // simple memcpy
	    memcpy(act_payload, payload_base, act_payload_size);
	  }
	}

	internal->commit_message(msg, comp,
				 act_header, header_size,
				 act_payload, act_payload_size);
      }

      // if we dynamically allocated space for the payload, free that now
      if((payload_size > 0) && (src_payload_addr == nullptr) &&
	 ((header_size + payload_size) > INLINE_SIZE))
	free(payload_base);
    } else {
      // arm the pending completion (if present)
      if(comp && !comp->mark_ready(1 /*exp_local*/, 1 /*exp_remote*/))
	comp = nullptr;

      if((src_payload_addr != 0) && (payload_base != src_payload_addr)) {
	if(src_payload_lines > 1) {
	  // copy line by line
	  size_t bytes_per_line = act_payload_size / src_payload_lines;
	  for(size_t i = 0; i < src_payload_lines; i++)
	    memcpy(static_cast<char *>(payload_base) + (i * bytes_per_line),
		   static_cast<const char *>(src_payload_addr) + (i * src_payload_line_stride),
		   bytes_per_line);
	} else {
	  // simple memcpy
	  memcpy(payload_base, src_payload_addr, act_payload_size);
	}

	comp = internal->early_local_completion(comp);
      }

      internal->commit_message(msg, comp,
			       header_base, header_size,
			       payload_base, act_payload_size);
    }
  }

  void GASNetEXMessageImpl::cancel()
  {
    if(is_multicast) {
      // we never told the internal gex logic about this, so all we have to
      //  is free payload memory if we allocated it
      if((payload_size > 0) && (src_payload_addr == nullptr) &&
	 ((header_size + payload_size) > INLINE_SIZE))
	free(payload_base);
    } else {
      internal->cancel_message(msg);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNetEXModule
  //

  GASNetEXModule::GASNetEXModule()
    : NetworkModule("gasnetex")
    , cfg_use_immediate(true)
    , cfg_use_negotiated(true)
    , cfg_crit_timeout(50000 /* 50us */)
    , cfg_max_medium(0 /* use GASNet limit */)
    , cfg_max_long(4 << 20 /* 4 MB */)
    , cfg_bind_hostmem(true)
#ifdef REALM_USE_CUDA
    , cfg_bind_cudamem(true)
#endif
#ifdef REALM_USE_HIP
    , cfg_bind_hipmem(true)
#endif
    , cfg_do_checksums(true) // TODO
    , cfg_batch_messages(true)
    , cfg_outbuf_count(64)
    , cfg_outbuf_size(256 << 10 /* 256 KB*/)
    , cfg_force_rma(false)
      // in GASNet releases before 2021.8.3, bugs 4148 and 4150 made RMA puts
      // unsafe to use for ibv + CUDA_UVA memory, so disable them by default
#if REALM_GEX_RELEASE < 20210803
    , cfg_use_rma_put(false)
#else
    , cfg_use_rma_put(true)
#endif
    , internal(nullptr)
  {}

  GASNetEXModule::~GASNetEXModule()
  {
    delete internal;
  }

  /*static*/ NetworkModule *GASNetEXModule::create_network_module(RuntimeImpl *runtime,
								  int *argc,
								  const char ***argv)
  {
    GASNetEXModule *mod = new GASNetEXModule;

    mod->internal = new GASNetEXInternal(mod, runtime);

    // set some gasnet-related environment variables, taking care not to
    //  overwrite anything explicitly set by the user

    // do not probe amount of pinnable memory
    setenv("GASNET_PHYSMEM_PROBE", "0", 0 /*no overwrite*/);

    // do not comment about on-demand-paging, which we are uninterested in
    setenv("GASNET_ODP_VERBOSE", "0", 0 /*no overwrite*/);

    // if we are using the ibv conduit with multiple-hca support, we need
    //  to enable fenced puts to work around gasnet bug 3447
    //  (https://gasnet-bugs.lbl.gov/bugzilla/show_bug.cgi?id=3447), but
    //  we can't set the flag if gasnet does NOT have multiple-hca support
    //  because it'll print warnings
    // in 2021.3.0 and earlier releases, there is no official way to detect
    //  this, and we can't even see the internal GASNETC_HAVE_FENCED_PUTS
    //  define, so we use the same condition that's used to set that in
    //  gasnet_core_internal.h and hope it doesn't change
    // releases after 2021.3.0 will define/expose GASNET_IBV_MULTIRAIL for us
    //  to look at
#if GASNET_IBV_MULTIRAIL || GASNETC_IBV_MAX_HCAS_CONFIGURE
    setenv("GASNET_USE_FENCED_PUTS", "1", 0 /*no overwrite*/);
#endif

#ifdef REALM_GASNETEX_MODULE_DYNAMIC
    // if we're a dynamic module, we can't have GASNet trying to do stuff
    //  in atexit handlers - we will have been unloaded at that point
    setenv("GASNET_CATCH_EXIT", "0", 1 /*overwrite!*/);

    // if enabled, IBV's ODP (on-demand pinning) registers an atexit handler
    //  as well, so make sure it's not enabled (Realm doesn't really need ODP)
    setenv("GASNET_USE_ODP", "0", 1 /*overwrite!*/);
#endif

    // GASNetEX no longer modifies argc/argv, but we've got it here, so share
    //  with gasnet anyway
    mod->internal->init(argc, argv);

    return mod;
  }

  // actual parsing of the command line should wait until here if at all
  //  possible
  void GASNetEXModule::parse_command_line(RuntimeImpl *runtime,
					  std::vector<std::string>& cmdline)
  {
    CommandLineParser cp;
    cp.add_option_int("-gex:immediate", cfg_use_immediate)
      .add_option_int("-gex:negotiated", cfg_use_negotiated)
      .add_option_int_units("-gex:maxmed", cfg_max_medium)
      .add_option_int_units("-gex:maxlong", cfg_max_long, 'm')
      .add_option_int("-gex:crittime", cfg_crit_timeout)
      .add_option_int("-gex:bindhost", cfg_bind_hostmem)
#ifdef REALM_USE_CUDA
      .add_option_int("-gex:bindcuda", cfg_bind_cudamem)
#endif
#ifdef REALM_USE_HIP
      .add_option_int("-gex:bindhip", cfg_bind_hipmem)
#endif
      .add_option_int("-gex:cksum", cfg_do_checksums)
      .add_option_int("-gex:batch", cfg_batch_messages)
      .add_option_int("-gex:obcount", cfg_outbuf_count)
      .add_option_int_units("-gex:obsize", cfg_outbuf_size)
      .add_option_int("-gex:forcerma", cfg_force_rma)
      .add_option_int("-gex:rmaput", cfg_use_rma_put);
    size_t deprecated_gsize = 0;
    cp.add_option_int_units("-ll:gsize", deprecated_gsize, 'm');

    bool ok = cp.parse_command_line(cmdline);
    assert(ok);

    if(deprecated_gsize > 0) {
      log_gex.fatal() << "Realm GASNetEX implementation does not provide a 'global' memory - '-ll:gsize' not permitted";
      abort();
    }

    // enforce a minimum outbuf count for now - right now we need at least one
    //  pktbuf and at least one databuf per rank (TODO: shrink this by stealing
    //  outbufs from idle endpoints)
    const size_t min_obufs = 2 * size_t(Network::max_node_id /* i.e. prim_size - 1 */);
    if(cfg_outbuf_count < min_obufs) {
      // issue a warning unless the user asked for 0 to mean "minimum"
      if((cfg_outbuf_count > 0) && (Network::my_node_id == 0))
        log_gex.warning() << "outbuf count raised from requested "
                          << cfg_outbuf_count << " to required minimum "
                          << min_obufs << " - if memory capacity issues result, reduce outbuf size using -gex:obsize";
      cfg_outbuf_count = min_obufs;
    }
  }

  // "attaches" to the network, if that is meaningful - attempts to
  //  bind/register/(pick your network-specific verb) the requested memory
  //  segments with the network
  void GASNetEXModule::attach(RuntimeImpl *runtime,
			     std::vector<NetworkSegment *>& segments)
  {
    // total up the size of all segments we'll try to fit into the gasnet
    //  segment
    size_t inseg_bytes = 0;
    for(std::vector<NetworkSegment *>::iterator it = segments.begin();
	it != segments.end();
	++it) {
      // must be asking for non-zero storage
      if((*it)->bytes == 0) continue;
      // must not already be assigned an address
      if((*it)->base != 0) continue;
      // must be host memory (for now at least)
      if((*it)->memtype != NetworkSegmentInfo::HostMem) continue;
      // TODO: consider alignment
      inseg_bytes += (*it)->bytes;
    }

    uintptr_t segbase = internal->attach(inseg_bytes);

    // for any segment that was pre-allocated, attempt to bind it
    for(NetworkSegment *seg : segments) {
      if(seg->base == nullptr) continue;

      gex_EP_Index_t ep_index = 0;
      bool ok = internal->attempt_binding(seg->base, seg->bytes,
					  seg->memtype, seg->memextra,
					  &ep_index);
      if(ok) {
	GASNetEXRDMAInfo info;
	info.base = reinterpret_cast<uintptr_t>(seg->base);
	info.ep_index = ep_index;
	seg->add_rdma_info(this, &info, sizeof(info));
      }
    }

    // now assign the base to any segments we allocated
    uintptr_t offset = 0;
    for(std::vector<NetworkSegment *>::iterator it = segments.begin();
	it != segments.end();
	++it) {
      // must be asking for non-zero storage
      if((*it)->bytes == 0) continue;
      // must not already be assigned an address
      if((*it)->base != 0) continue;
      // must be host memory (for now at least)
      if((*it)->memtype != NetworkSegmentInfo::HostMem) continue;
      // TODO: consider alignment
      (*it)->base = reinterpret_cast<void *>(segbase + offset);
      GASNetEXRDMAInfo info;
      info.base = segbase + offset;
      info.ep_index = 0; // primordial endpoint is always #0
      (*it)->add_rdma_info(this, &info, sizeof(info));
      offset += (*it)->bytes;
    }

    // all done - publish our bindings (collective across all nodes)
    internal->publish_bindings();
  }

  void GASNetEXModule::create_memories(RuntimeImpl *runtime)
  {
  }

  // detaches from the network
  void GASNetEXModule::detach(RuntimeImpl *runtime,
			      std::vector<NetworkSegment *>& segments)
  {
    internal->detach();
  }

  // collective communication within this network
  void GASNetEXModule::barrier(void)
  {
    internal->barrier();
  }


  //  static const int GASNET_COLL_FLAGS = GASNET_COLL_IN_MYSYNC | GASNET_COLL_OUT_MYSYNC | GASNET_COLL_LOCAL;

  void GASNetEXModule::broadcast(NodeID root, const void *val_in, void *val_out, size_t bytes)
  {
    internal->broadcast(root, val_in, val_out, bytes);
  }

  void GASNetEXModule::gather(NodeID root, const void *val_in, void *vals_out, size_t bytes)
  {
    internal->gather(root, val_in, vals_out, bytes);
  }

  size_t GASNetEXModule::sample_messages_received_count(void)
  {
    return internal->sample_messages_received_count();
  }

  bool GASNetEXModule::check_for_quiescence(size_t sampled_receive_count)
  {
    return internal->check_for_quiescence(sampled_receive_count);
  }

  // used to create a remote proxy for a memory
  MemoryImpl *GASNetEXModule::create_remote_memory(Memory m, size_t size, Memory::Kind kind,
						  const ByteArray& rdma_info)
  {
    assert(kind != Memory::GLOBAL_MEM);

    assert(rdma_info.size() == sizeof(GASNetEXRDMAInfo));
    GASNetEXRDMAInfo info;
    memcpy(&info, rdma_info.base(), sizeof(GASNetEXRDMAInfo));

    return new GASNetEXRemoteMemory(m, size, kind, info.base, info.ep_index);
  }

  IBMemory *GASNetEXModule::create_remote_ib_memory(Memory m, size_t size, Memory::Kind kind,
						   const ByteArray& rdma_info)
  {
    assert(rdma_info.size() == sizeof(GASNetEXRDMAInfo));
    GASNetEXRDMAInfo info;
    memcpy(&info, rdma_info.base(), sizeof(GASNetEXRDMAInfo));

    return new GASNetEXIBMemory(m, size, kind, info.base, info.ep_index);
  }

  ActiveMessageImpl *GASNetEXModule::create_active_message_impl(NodeID target,
								unsigned short msgid,
								size_t header_size,
								size_t max_payload_size,
								const void *src_payload_addr,
								size_t src_payload_lines,
								size_t src_payload_line_stride,
								void *storage_base,
								size_t storage_size)
  {
    // if checksums are enabled, we'll tack it on to the end of the header
    //  to avoid any alignment issues
    if(cfg_do_checksums)
      header_size = roundup_pow2(header_size + sizeof(gex_AM_Arg_t),
                                 sizeof(gex_AM_Arg_t));

    assert(storage_size >= sizeof(GASNetEXMessageImpl));
    GASNetEXMessageImpl *impl = new(storage_base) GASNetEXMessageImpl(internal,
								      target,
								      msgid,
								      header_size,
								      max_payload_size,
								      src_payload_addr,
								      src_payload_lines,
								      src_payload_line_stride,
								      0, 0);
    return impl;
  }

  ActiveMessageImpl *GASNetEXModule::create_active_message_impl(NodeID target,
								unsigned short msgid,
								size_t header_size,
								size_t max_payload_size,
								const void *src_payload_addr,
								size_t src_payload_lines,
								size_t src_payload_line_stride,
								const RemoteAddress& dest_payload_addr,
								void *storage_base,
								size_t storage_size)
  {
    // if checksums are enabled, we'll tack it on to the end of the header
    //  to avoid any alignment issues
    if(cfg_do_checksums)
      header_size = roundup_pow2(header_size + sizeof(gex_AM_Arg_t),
                                 sizeof(gex_AM_Arg_t));

    assert(storage_size >= sizeof(GASNetEXMessageImpl));
    GASNetEXMessageImpl *impl = new(storage_base) GASNetEXMessageImpl(internal,
								      target,
								      msgid,
								      header_size,
								      max_payload_size,
								      src_payload_addr,
								      src_payload_lines,
								      src_payload_line_stride,
								      dest_payload_addr.ptr,
								      dest_payload_addr.extra);
    return impl;
  }

  ActiveMessageImpl *GASNetEXModule::create_active_message_impl(const NodeSet& targets,
								unsigned short msgid,
								size_t header_size,
								size_t max_payload_size,
								const void *src_payload_addr,
								size_t src_payload_lines,
								size_t src_payload_line_stride,
								void *storage_base,
								size_t storage_size)
  {
    // if checksums are enabled, we'll tack it on to the end of the header
    //  to avoid any alignment issues
    if(cfg_do_checksums)
      header_size = roundup_pow2(header_size + sizeof(gex_AM_Arg_t),
                                 sizeof(gex_AM_Arg_t));

    assert(storage_size >= sizeof(GASNetEXMessageImpl));
    if(targets.size() == 1) {
      // optimization - if there's exactly 1 target, redirect to the unicast mode
      NodeID target = *(targets.begin());
      GASNetEXMessageImpl *impl = new(storage_base) GASNetEXMessageImpl(internal,
									target,
									msgid,
									header_size,
									max_payload_size,
									src_payload_addr,
									src_payload_lines,
									src_payload_line_stride,
									0, 0);
      return impl;
    } else {
      // zero or 2+ targets - we'll make a temporary copy of the payload for now
      GASNetEXMessageImpl *impl = new(storage_base) GASNetEXMessageImpl(internal,
									targets,
									msgid,
									header_size,
									max_payload_size,
									src_payload_addr,
									src_payload_lines,
									src_payload_line_stride);
      return impl;
    }
  }

  size_t GASNetEXModule::recommended_max_payload(NodeID target,
						 bool with_congestion,
						 size_t header_size)
  {
    if(cfg_do_checksums)
      header_size = roundup_pow2(header_size + sizeof(gex_AM_Arg_t),
                                 sizeof(gex_AM_Arg_t));

    return internal->recommended_max_payload(target, 0 /*ep_index*/,
					     with_congestion,
					     header_size,
					     0 /*no dest_ptr*/);
  }

  size_t GASNetEXModule::recommended_max_payload(const NodeSet& targets,
						 bool with_congestion,
						 size_t header_size)
  {
    if(cfg_do_checksums)
      header_size = roundup_pow2(header_size + sizeof(gex_AM_Arg_t),
                                 sizeof(gex_AM_Arg_t));

    if(targets.size() == 1) {
      // optimization - if there's exactly 1 target, redirect to the unicast mode
      NodeID target = *(targets.begin());
      return internal->recommended_max_payload(target, 0 /*ep_index*/,
					       with_congestion,
					       header_size,
					       0 /*no dest_ptr*/);
    } else {
      // ask without specifying a target - gets conservative answer
      return internal->recommended_max_payload(with_congestion,
					       header_size);
    }
  }

  size_t GASNetEXModule::recommended_max_payload(NodeID target,
						 const RemoteAddress& dest_payload_addr,
						 bool with_congestion,
						 size_t header_size)
  {
    if(cfg_do_checksums)
      header_size = roundup_pow2(header_size + sizeof(gex_AM_Arg_t),
                                 sizeof(gex_AM_Arg_t));

    return internal->recommended_max_payload(target,
					     dest_payload_addr.extra,
					     with_congestion,
					     header_size,
					     dest_payload_addr.ptr);
  }

  size_t GASNetEXModule::recommended_max_payload(NodeID target,
						 const void *data, size_t bytes_per_line,
						 size_t lines, size_t line_stride,
						 bool with_congestion,
						 size_t header_size)
  {
    return internal->recommended_max_payload(target, 0 /*ep_index*/,
					     data, bytes_per_line,
					     lines, line_stride,
					     with_congestion,
					     header_size,
					     0 /*no dest_ptr*/);
  }

  size_t GASNetEXModule::recommended_max_payload(const NodeSet& targets,
						 const void *data, size_t bytes_per_line,
						 size_t lines, size_t line_stride,
						 bool with_congestion,
						 size_t header_size)
  {
    if(targets.size() == 1) {
      // optimization - if there's exactly 1 target, redirect to the unicast mode
      NodeID target = *(targets.begin());
      return internal->recommended_max_payload(target, 0 /*ep_index*/,
					       data, bytes_per_line,
					       lines, line_stride,
					       with_congestion,
					       header_size,
					       0 /*no dest_ptr*/);
    } else {
      // ask without specifying a target - gets conservative answer
      return internal->recommended_max_payload(with_congestion,
					       header_size);
    }
  }

  size_t GASNetEXModule::recommended_max_payload(NodeID target,
						 const void *data, size_t bytes_per_line,
						 size_t lines, size_t line_stride,
						 const RemoteAddress& dest_payload_addr,
						 bool with_congestion,
						 size_t header_size)
  {
    return internal->recommended_max_payload(target,
					     dest_payload_addr.extra,
					     data, bytes_per_line,
					     lines, line_stride,
					     with_congestion,
					     header_size,
					     dest_payload_addr.ptr);
  }


}; // namespace Realm
