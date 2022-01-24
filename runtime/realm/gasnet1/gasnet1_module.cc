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

// GASNet-1 network module implementation for Realm

// NOTE: this should work with GASNet-EX's backward-compatibility interfaces,
//  but it is not recommended - there are known performance regressions compared
//  to using an actual GASNet-1 implementation

#include "realm/network.h"

#include "realm/gasnet1/gasnet1_module.h"

#include "realm/gasnet1/gasnetmsg.h"

#include "realm/runtime_impl.h"
#include "realm/mem_impl.h"
#include "realm/transfer/ib_memory.h"

#ifndef GASNET_PAR
#define GASNET_PAR
#endif
#include <gasnet.h>
#include <gasnet_coll.h>
// eliminate GASNet warnings for unused static functions
REALM_ATTR_UNUSED(static const void *ignore_gasnet_warning1) = (void *)_gasneti_threadkey_init;
#ifdef _INCLUDED_GASNET_TOOLS_H
REALM_ATTR_UNUSED(static const void *ignore_gasnet_warning2) = (void *)_gasnett_trace_printf_noop;
#endif

#define CHECK_GASNET(cmd) do { \
  int ret = (cmd); \
  if(ret != GASNET_OK) { \
    fprintf(stderr, "GASNET: %s = %d (%s, %s)\n", #cmd, ret, gasnet_ErrorName(ret), gasnet_ErrorDesc(ret)); \
    exit(1); \
  } \
} while(0)

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNet1Memory
  //

  class GASNet1Memory : public LocalManagedMemory {
  public:
    static const size_t MEMORY_STRIDE = 1024;

    GASNet1Memory(Memory _me, size_t size_per_node, NetworkModule *_gasnet);

    virtual ~GASNet1Memory(void);

    virtual void get_bytes(off_t offset, void *dst, size_t size);

    virtual void put_bytes(off_t offset, const void *src, size_t size);
    
    virtual void *get_direct_ptr(off_t offset, size_t size);

    void get_batch(size_t batch_size,
		   const off_t *offsets, void * const *dsts, 
		   const size_t *sizes);
    
    void put_batch(size_t batch_size,
		   const off_t *offsets, const void * const *srcs, 
		   const size_t *sizes);

    // gets info related to rdma access from other nodes
    virtual const ByteArray *get_rdma_info(NetworkModule *network);

  protected:
    int num_nodes;
    off_t memory_stride;
    std::vector<char *> segbases;
    NetworkModule *gasnet;
  };

  GASNet1Memory::GASNet1Memory(Memory _me, size_t size_per_node,
			       NetworkModule *_gasnet)
    : LocalManagedMemory(_me, 0 /* we'll calculate it below */, MKIND_GLOBAL,
			 MEMORY_STRIDE, Memory::GLOBAL_MEM, 0)
    , gasnet(_gasnet)
  {
    num_nodes = Network::max_node_id + 1;
    segbases.resize(num_nodes);

    gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[num_nodes];
    CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );
      
    for(int i = 0; i < num_nodes; i++) {
      assert(seginfos[i].size >= size_per_node);
      segbases[i] = static_cast<char *>(seginfos[i].addr);
    }
    delete[] seginfos;

    size = size_per_node * num_nodes;
    memory_stride = MEMORY_STRIDE;
  }

  GASNet1Memory::~GASNet1Memory(void)
  {
  }

  void GASNet1Memory::get_bytes(off_t offset, void *dst, size_t size)
  {
    char *dst_c = (char *)dst;
    while(size > 0) {
      off_t blkid = (offset / memory_stride / num_nodes);
      off_t node = (offset / memory_stride) % num_nodes;
      off_t blkoffset = offset % memory_stride;
      size_t chunk_size = memory_stride - blkoffset;
      if(chunk_size > size) chunk_size = size;
      gasnet_get(dst_c, node, segbases[node]+(blkid * memory_stride)+blkoffset, chunk_size);
      offset += chunk_size;
      dst_c += chunk_size;
      size -= chunk_size;
    }
  }

  void GASNet1Memory::put_bytes(off_t offset, const void *src, size_t size)
  {
    char *src_c = (char *)src; // dropping const on purpose...
    while(size > 0) {
      off_t blkid = (offset / memory_stride / num_nodes);
      off_t node = (offset / memory_stride) % num_nodes;
      off_t blkoffset = offset % memory_stride;
      size_t chunk_size = memory_stride - blkoffset;
      if(chunk_size > size) chunk_size = size;
      gasnet_put(node, segbases[node]+(blkid * memory_stride)+blkoffset, src_c, chunk_size);
      offset += chunk_size;
      src_c += chunk_size;
      size -= chunk_size;
    }
  }

  void *GASNet1Memory::get_direct_ptr(off_t offset, size_t size)
  {
    return 0;  // can't give a pointer to the caller - have to use RDMA
  }

  void GASNet1Memory::get_batch(size_t batch_size,
				const off_t *offsets, void * const *dsts, 
				const size_t *sizes)
  {
#define NO_USE_NBI_ACCESSREGION
#ifdef USE_NBI_ACCESSREGION
    gasnet_begin_nbi_accessregion();
#endif
    for(size_t i = 0; i < batch_size; i++) {
      off_t offset = offsets[i];
      char *dst_c = (char *)(dsts[i]);
      size_t size = sizes[i];
      
      off_t blkid = (offset / memory_stride / num_nodes);
      off_t node = (offset / memory_stride) % num_nodes;
      off_t blkoffset = offset % memory_stride;

      while(size > 0) {
	size_t chunk_size = memory_stride - blkoffset;
	if(chunk_size > size) chunk_size = size;

	char *src_c = (segbases[node] +
		       (blkid * memory_stride) + blkoffset);
	if(node != Network::my_node_id) {
	  gasnet_get_nbi(dst_c, node, src_c, chunk_size);
	} else {
	  memcpy(dst_c, src_c, chunk_size);
	}

	dst_c += chunk_size;
	size -= chunk_size;
	blkoffset = 0;
	node = (node + 1) % num_nodes;
	if(node == 0) blkid++;
      }
    }

#ifdef USE_NBI_ACCESSREGION
    gasnet_handle_t handle = gasnet_end_nbi_accessregion();

    gasnet_wait_syncnb(handle);
#else
    gasnet_wait_syncnbi_gets();
#endif
  }

  void GASNet1Memory::put_batch(size_t batch_size,
				const off_t *offsets,
				const void * const *srcs, 
				const size_t *sizes)
  {
    gasnet_begin_nbi_accessregion();

    for(size_t i = 0; i < batch_size; i++) {
      off_t offset = offsets[i];
      const char *src_c = (char *)(srcs[i]);
      size_t size = sizes[i];

      off_t blkid = (offset / memory_stride / num_nodes);
      off_t node = (offset / memory_stride) % num_nodes;
      off_t blkoffset = offset % memory_stride;

      while(size > 0) {
	size_t chunk_size = memory_stride - blkoffset;
	if(chunk_size > size) chunk_size = size;

	char *dst_c = (segbases[node] +
		       (blkid * memory_stride) + blkoffset);

	if(node != Network::my_node_id) {
	  gasnet_put_nbi(node, dst_c, (void *)src_c, chunk_size);
	} else {
	  memcpy(dst_c, src_c, chunk_size);
	}

	src_c += chunk_size;
	size -= chunk_size;
	blkoffset = 0;
	node = (node + 1) % num_nodes;
	if(node == 0) blkid++;
      }
    }

    gasnet_handle_t handle = gasnet_end_nbi_accessregion();

    gasnet_wait_syncnb(handle);
  }

  // gets info related to rdma access from other nodes
  const ByteArray *GASNet1Memory::get_rdma_info(NetworkModule *network)
  {
    // provide a dummy rdma info for gasnet endpoints so that we get
    //  handled by the gasnet network module instead of turned into a
    //  normal RemoteMemory
    static ByteArray dummy_rdma_info;
    if(network == gasnet)
      return &dummy_rdma_info;
    else
      return 0;
  }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNet1RemoteMemory
  //

  class GASNet1RemoteMemory : public RemoteMemory {
  public:
    GASNet1RemoteMemory(Memory _me, size_t _size, Memory::Kind k,
			void *_regbase);

    virtual void get_bytes(off_t offset, void *dst, size_t size);
    virtual void put_bytes(off_t offset, const void *src, size_t size);

    virtual bool get_remote_addr(off_t offset, RemoteAddress& remote_addr);

  protected:
    char *regbase;
  };

  GASNet1RemoteMemory::GASNet1RemoteMemory(Memory _me, size_t _size,
					   Memory::Kind k,
					   void *_regbase)
    : RemoteMemory(_me, _size, k, MKIND_RDMA)
    , regbase(static_cast<char *>(_regbase))
  {}
  
  void GASNet1RemoteMemory::get_bytes(off_t offset, void *dst, size_t size)
  {
    gasnet_get(dst, ID(me).memory_owner_node(),
	       regbase + offset, size);
  }
  
  void GASNet1RemoteMemory::put_bytes(off_t offset,
				      const void *src, size_t size)
  {
    gasnet_put(ID(me).memory_owner_node(), regbase + offset,
	       const_cast<void *>(src), size);
  }

  bool GASNet1RemoteMemory::get_remote_addr(off_t offset, RemoteAddress& remote_addr)
  {
    remote_addr.ptr = reinterpret_cast<uintptr_t>(regbase + offset);
    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNet1IBMemory
  //

  class GASNet1IBMemory : public IBMemory {
  public:
    GASNet1IBMemory(Memory _me, size_t _size, Memory::Kind k, void *_regbase);

    virtual bool get_remote_addr(off_t offset, RemoteAddress& remote_addr);

  protected:
    char *regbase;
  };

  GASNet1IBMemory::GASNet1IBMemory(Memory _me, size_t _size, Memory::Kind k,
				   void *_regbase)
    : IBMemory(_me, _size, MKIND_REMOTE, k, 0, 0)
    , regbase(static_cast<char *>(_regbase))
  {}

  bool GASNet1IBMemory::get_remote_addr(off_t offset, RemoteAddress& remote_addr)
  {
    remote_addr.ptr = reinterpret_cast<uintptr_t>(regbase + offset);
    return true;
  }
  

  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNet1MessageImpl
  //

  class GASNet1MessageImpl : public ActiveMessageImpl {
  public:
    GASNet1MessageImpl(NodeID _target,
		       unsigned short _msgid,
		       size_t _header_size,
		       size_t _max_payload_size,
		       const void *_src_payload_addr,
		       size_t _src_payload_lines,
		       size_t _src_payload_line_stride,
		       void *_dest_payload_addr);
    GASNet1MessageImpl(const Realm::NodeSet &_targets,
		       unsigned short _msgid,
		       size_t _header_size,
		       size_t _max_payload_size,
		       const void *_src_payload_addr,
		       size_t _src_payload_lines,
		       size_t _src_payload_line_stride);
  
    virtual ~GASNet1MessageImpl();

    virtual void *add_local_completion(size_t size);
    virtual void *add_remote_completion(size_t size);

    virtual void commit(size_t act_payload_size);
    virtual void cancel();

  protected:
    NodeID target;
    Realm::NodeSet targets;
    bool is_multicast;
    const void *src_payload_addr;
    size_t src_payload_lines;
    size_t src_payload_line_stride;
    void *dest_payload_addr;
    PendingCompletion *comp;
    size_t header_size;
    struct FullHeader : public BaseMedium {
      unsigned short msgid;
      unsigned short sender;
      unsigned payload_len;
      unsigned long msg_header;  // type is unknown
    };
    FullHeader args; // must be last thing
  };

  GASNet1MessageImpl::GASNet1MessageImpl(NodeID _target,
					 unsigned short _msgid,
					 size_t _header_size,
					 size_t _max_payload_size,
					 const void *_src_payload_addr,
					 size_t _src_payload_lines,
					 size_t _src_payload_line_stride,
					 void *_dest_payload_addr)
    : target(_target)
    , is_multicast(false)
    , src_payload_addr(_src_payload_addr)
    , src_payload_lines(_src_payload_lines)
    , src_payload_line_stride(_src_payload_line_stride)
    , dest_payload_addr(_dest_payload_addr)
    , comp(0)
    , header_size(_header_size)
  {
    if(_max_payload_size && (src_payload_addr == 0)) {
      payload_base = reinterpret_cast<char *>(malloc(_max_payload_size));
    } else {
      payload_base = 0;
    }
    payload_size = _max_payload_size;
    args.msgid = _msgid;
    header_base = &args.msg_header;
    assert((sizeof(BaseMedium) + 8 + header_size) <= 16 * sizeof(handlerarg_t));
  }

  GASNet1MessageImpl::GASNet1MessageImpl(const Realm::NodeSet &_targets,
					 unsigned short _msgid,
					 size_t _header_size,
					 size_t _max_payload_size,
					 const void *_src_payload_addr,
					 size_t _src_payload_lines,
					 size_t _src_payload_line_stride)
    : targets(_targets)
    , is_multicast(true)
    , src_payload_addr(_src_payload_addr)
    , src_payload_lines(_src_payload_lines)
    , src_payload_line_stride(_src_payload_line_stride)
    , dest_payload_addr(0)
    , comp(0)
    , header_size(_header_size)
  {
    if(_max_payload_size && (src_payload_addr == 0)) {
      payload_base = reinterpret_cast<char *>(malloc(_max_payload_size));
    } else {
      payload_base = 0;
    }
    payload_size = _max_payload_size;
    args.msgid = _msgid;
    header_base = &args.msg_header;
    assert((sizeof(BaseMedium) + 8 + header_size) <= 16 * sizeof(handlerarg_t));
  }
  
  GASNet1MessageImpl::~GASNet1MessageImpl()
  {
  }

  void *GASNet1MessageImpl::add_local_completion(size_t size)
  {
    // if we don't already have a pending completion object, get one
    if(!comp)
      comp = completion_manager.get_available();
    return comp->add_local_completion(size);
  }

  void *GASNet1MessageImpl::add_remote_completion(size_t size)
  {
    // if we don't already have a pending completion object, get one
    if(!comp)
      comp = completion_manager.get_available();
    return comp->add_remote_completion(size);
  }

  void GASNet1MessageImpl::commit(size_t act_payload_size)
  {
    args.set_magic();
    args.sender = Network::my_node_id;
    args.payload_len = act_payload_size;

    if(is_multicast) {
      assert(dest_payload_addr == 0);
      assert(comp == 0);
      size_t count = targets.size();
      if(count > 0) {
	for(NodeSet::const_iterator it = targets.begin();
	    it != targets.end();
	    ++it) {
	  if(src_payload_addr != 0) {
	    if(src_payload_lines > 1)
	      enqueue_message(*it, MSGID_NEW_ACTIVEMSG,
			      &args, header_size+24,
			      src_payload_addr,
			      act_payload_size / src_payload_lines,
			      src_payload_line_stride, src_payload_lines,
			      PAYLOAD_KEEP, 0);
	    else
	      enqueue_message(*it, MSGID_NEW_ACTIVEMSG,
			      &args, header_size+24,
			      src_payload_addr, act_payload_size,
			      PAYLOAD_KEEP, 0);
	  } else
	    enqueue_message(*it, MSGID_NEW_ACTIVEMSG,
			    &args, header_size+24,
			    payload_base, act_payload_size,
			    ((count > 0) ? PAYLOAD_COPY : PAYLOAD_FREE), 0);
	  count--;
	}
      } else {
	// free the (unused) payload ourselves
	if((payload_size > 0) && (src_payload_addr == 0))
	  free(payload_base);
      }
    } else {
      if(src_payload_addr != 0) {
	if(src_payload_lines > 1)
	  enqueue_message(target, MSGID_NEW_ACTIVEMSG,
			  &args, header_size+24,
			  src_payload_addr, (act_payload_size / src_payload_lines),
			  src_payload_line_stride, src_payload_lines,
			  PAYLOAD_KEEP, comp, dest_payload_addr);
	else
	  enqueue_message(target, MSGID_NEW_ACTIVEMSG,
			  &args, header_size+24,
			  src_payload_addr, act_payload_size, PAYLOAD_KEEP,
			  comp, dest_payload_addr);
      } else
	enqueue_message(target, MSGID_NEW_ACTIVEMSG,
			&args, header_size+24,
			payload_base, act_payload_size, PAYLOAD_FREE,
			comp, dest_payload_addr);
    }
  }

  void GASNet1MessageImpl::cancel()
  {
    assert(comp == 0);
    if((payload_size > 0) && (src_payload_addr == 0))
      free(payload_base);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNet1Module
  //

  GASNet1Module::GASNet1Module(void)
    : NetworkModule("gasnet1")
    , active_msg_worker_bgwork(true)
    , active_msg_worker_threads(0)
    , gasnet_mem_size(0)
    , amsg_stack_size(2 << 20)
  {}
  
  /*static*/ NetworkModule *GASNet1Module::create_network_module(RuntimeImpl *runtime,
								 int *argc,
								 const char ***argv)
  {
    // gasnet_init() must be called before parsing command line arguments, as some
    //  spawners (e.g. the ssh spawner for gasnetrun_ibv) start with bogus args and
    //  fetch the real ones from somewhere during gasnet_init()

    // SJT: WAR for issue on Titan with duplicate cookies on Gemini
    //  communication domains
    char *orig_pmi_gni_cookie = getenv("PMI_GNI_COOKIE");
    if(orig_pmi_gni_cookie) {
      char new_pmi_gni_cookie[32];
      snprintf(new_pmi_gni_cookie, 32, "%d", 1+atoi(orig_pmi_gni_cookie));
      setenv("PMI_GNI_COOKIE", new_pmi_gni_cookie, 1 /*overwrite*/);
    }
    // SJT: another GASNET workaround - if we don't have GASNET_IB_SPAWNER set, assume it was MPI
    // (This is called GASNET_IB_SPAWNER for versions <= 1.24 and GASNET_SPAWNER for versions >= 1.26)
    if(!getenv("GASNET_IB_SPAWNER") && !getenv("GASNET_SPAWNER")) {
      setenv("GASNET_IB_SPAWNER", "mpi", 0 /*no overwrite*/);
      setenv("GASNET_SPAWNER", "mpi", 0 /*no overwrite*/);
    }

    // and one more... disable GASNet's probing of pinnable memory - it's
    //  painfully slow on most systems (the gemini conduit doesn't probe
    //  at all, so it's ok)
    // we can do this because in gasnet_attach() we will ask for exactly as
    //  much as we need, and we can detect failure there if that much memory
    //  doesn't actually exist
    // inconveniently, we have to set a PHYSMEM_MAX before we call
    //  gasnet_init and we don't have our argc/argv until after, so we can't
    //  set PHYSMEM_MAX correctly, but setting it to something really big to
    //  prevent all the early checks from failing gets us to that final actual
    //  alloc/pin in gasnet_attach ok
    {
      // the only way to control this is with environment variables, so set
      //  them unless the user has already set them (in which case, we assume
      //  they know what they're doing)
      // do handle the case where NOPROBE is set to 1, but PHYSMEM_MAX isn't
      const char *e = getenv("GASNET_PHYSMEM_NOPROBE");
      if(!e || (atoi(e) > 0)) {
	if(!e)
	  setenv("GASNET_PHYSMEM_NOPROBE", "1", 0 /*no overwrite*/);
	if(!getenv("GASNET_PHYSMEM_MAX")) {
	  // just because it's fun to read things like this 20 years later:
	  // "nobody will ever build a system with more than 1 TB of RAM..."
	  setenv("GASNET_PHYSMEM_MAX", "1T", 0 /*no overwrite*/);
	}
      }
    }

    // and yet another GASNet workaround: the Infiniband conduit seems to
    //  have a problem with AMRDMA mode, consuming receive buffers even for
    //  request targets that are in AMRDMA mode - disable the mode by default
#ifdef GASNET_CONDUIT_IBV
    if(!getenv("GASNET_AMRDMA_MAX_PEERS"))
      setenv("GASNET_AMRDMA_MAX_PEERS", "0", 0 /*no overwrite*/);
#endif

#ifdef DEBUG_REALM_STARTUP
    { // we don't have rank IDs yet, so everybody gets to spew
      char s[80];
      gethostname(s, 79);
      strcat(s, " enter gasnet_init");
      TimeStamp ts(s, false);
      fflush(stdout);
    }
#endif
    CHECK_GASNET( gasnet_init(argc, const_cast<char ***>(argv)) );
    Network::my_node_id = gasnet_mynode();
    Network::max_node_id = gasnet_nodes() - 1;
    Network::all_peers.add_range(0, gasnet_nodes() - 1);
    Network::all_peers.remove(gasnet_mynode());
#ifdef DEBUG_REALM_STARTUP
    { // once we're convinced there isn't skew here, reduce this to rank 0
      char s[80];
      gethostname(s, 79);
      strcat(s, " exit gasnet_init");
      TimeStamp ts(s, false);
      fflush(stdout);
    }
#endif

    return new GASNet1Module;
  }

  // actual parsing of the command line should wait until here if at all
  //  possible
  void GASNet1Module::parse_command_line(RuntimeImpl *runtime,
					 std::vector<std::string>& cmdline)
  {
    gasnet_parse_command_line(cmdline);

    CommandLineParser cp;
    cp.add_option_int_units("-ll:gsize", gasnet_mem_size, 'm')
      .add_option_int("-ll:amsg", active_msg_worker_threads)
      .add_option_int("-ll:amsg_bgwork", active_msg_worker_bgwork)
      .add_option_int_units("-ll:astack", amsg_stack_size, 'm');
    
    bool ok = cp.parse_command_line(cmdline);
    assert(ok);

    // make sure we have some way to get polling done
    assert(active_msg_worker_bgwork || (active_msg_worker_threads > 0));
  }

  // "attaches" to the network, if that is meaningful - attempts to
  //  bind/register/(pick your network-specific verb) the requested memory
  //  segments with the network
  void GASNet1Module::attach(RuntimeImpl *runtime,
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
      // must be host memory
      if((*it)->memtype != NetworkSegmentInfo::HostMem) continue;
      // TODO: consider alignment
      inseg_bytes += (*it)->bytes;
    }
    
    init_endpoints(gasnet_mem_size, inseg_bytes, 0,
		   *(runtime->core_reservations),
		   active_msg_worker_threads,
		   runtime->bgwork,
		   active_msg_worker_bgwork,
		   runtime->message_manager);

    // Put this here so that it complies with the GASNet specification and
    // doesn't make any calls between gasnet_init and gasnet_attach
    gasnet_set_waitmode(GASNET_WAIT_BLOCK);

#if ((GEX_SPEC_VERSION_MAJOR << 8) + GEX_SPEC_VERSION_MINOR) < 5
    // this needs to happen after init_endpoints
    gasnet_coll_init(0, 0, 0, 0, 0);
#endif

    gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[Network::max_node_id + 1];
    CHECK_GASNET( gasnet_getSegmentInfo(seginfos, Network::max_node_id + 1) );
    char *seg_base = reinterpret_cast<char *>(seginfos[Network::my_node_id].addr);
    seg_base += gasnet_mem_size;
    delete[] seginfos;

    // assign the base to any segment we allocated
    for(std::vector<NetworkSegment *>::iterator it = segments.begin();
	it != segments.end();
	++it) {
      // must be asking for non-zero storage
      if((*it)->bytes == 0) continue;
      // must not already be assigned an address
      if((*it)->base != 0) continue;
      // must be host memory
      if((*it)->memtype != NetworkSegmentInfo::HostMem) continue;
      // TODO: consider alignment
      (*it)->base = seg_base;
      // RDMA info for GASNet is just the local base pointer
      (*it)->add_rdma_info(this, &seg_base, sizeof(void *));
      seg_base += (*it)->bytes;
    }

    start_polling_threads();

    start_handler_threads(amsg_stack_size);
  }

  void GASNet1Module::create_memories(RuntimeImpl *runtime)
  {
    if(gasnet_mem_size > 0) {
      // only node 0 creates the gasnet memory
      if(Network::my_node_id == 0) {
	Memory m = runtime->next_local_memory_id();
	GASNet1Memory *mem = new GASNet1Memory(m, gasnet_mem_size, this);
	runtime->add_memory(mem);
      }
    }
  }
  
  // detaches from the network
  void GASNet1Module::detach(RuntimeImpl *runtime,
			     std::vector<NetworkSegment *>& segments)
  {
    stop_activemsg_threads();
  }

  // collective communication within this network
  void GASNet1Module::barrier(void)
  {
    gasnet_barrier_notify(0, GASNET_BARRIERFLAG_ANONYMOUS);
    gasnet_barrier_wait(0, GASNET_BARRIERFLAG_ANONYMOUS);
  }
    
  static const int GASNET_COLL_FLAGS = GASNET_COLL_IN_MYSYNC | GASNET_COLL_OUT_MYSYNC | GASNET_COLL_LOCAL;
  
  void GASNet1Module::broadcast(NodeID root, const void *val_in, void *val_out, size_t bytes)
  {
    gasnet_coll_broadcast(GASNET_TEAM_ALL, val_out, root,
			  const_cast<void *>(val_in), bytes,
			  GASNET_COLL_FLAGS);
  }
  
  void GASNet1Module::gather(NodeID root, const void *val_in, void *vals_out, size_t bytes)
  {
    gasnet_coll_gather(GASNET_TEAM_ALL, root,
		       vals_out, const_cast<void *>(val_in), bytes,
		       GASNET_COLL_FLAGS);
  }

  size_t GASNet1Module::sample_messages_received_count(void)
  {
    return quiescence_checker.sample_messages_received_count();
  }

  bool GASNet1Module::check_for_quiescence(size_t sampled_receive_count)
  {
    return quiescence_checker.perform_check(sampled_receive_count);
  }

  // used to create a remote proxy for a memory
  MemoryImpl *GASNet1Module::create_remote_memory(Memory m, size_t size, Memory::Kind kind,
						  const ByteArray& rdma_info)
  {
    if(kind == Memory::GLOBAL_MEM) {
      // this is actually our gasnet memory - make an aspect of it here too
      assert(size == (gasnet_mem_size * (Network::max_node_id + 1)));
      return new GASNet1Memory(m, gasnet_mem_size, this);
    } else {
      // otherwise it's some other kind of memory that we were able to register
      //  with gasnet

      // rdma info should be the pointer in the remote address space
      assert(rdma_info.size() == sizeof(void *));
      void *regbase;
      memcpy(&regbase, rdma_info.base(), sizeof(void *));

      return new GASNet1RemoteMemory(m, size, kind, regbase);
    }
  }
  
  IBMemory *GASNet1Module::create_remote_ib_memory(Memory m, size_t size, Memory::Kind kind,
						   const ByteArray& rdma_info)
  {
    // rdma info should be the pointer in the remote address space
    assert(rdma_info.size() == sizeof(void *));
    void *regbase;
    memcpy(&regbase, rdma_info.base(), sizeof(void *));

    return new GASNet1IBMemory(m, size, kind, regbase);
  }

  ActiveMessageImpl *GASNet1Module::create_active_message_impl(NodeID target,
							       unsigned short msgid,
							       size_t header_size,
							       size_t max_payload_size,
							       const void *src_payload_addr,
							       size_t src_payload_lines,
							       size_t src_payload_line_stride,
							       void *storage_base,
							       size_t storage_size)
  {
    assert(storage_size >= sizeof(GASNet1MessageImpl));
    GASNet1MessageImpl *impl = new(storage_base) GASNet1MessageImpl(target,
								    msgid,
								    header_size,
								    max_payload_size,
								    src_payload_addr,
								    src_payload_lines,
								    src_payload_line_stride,
								    0);
    return impl;
  }

  ActiveMessageImpl *GASNet1Module::create_active_message_impl(NodeID target,
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
    assert(storage_size >= sizeof(GASNet1MessageImpl));
    void *dest_ptr = reinterpret_cast<void *>(dest_payload_addr.ptr);
    assert(dest_ptr != 0);
    GASNet1MessageImpl *impl = new(storage_base) GASNet1MessageImpl(target,
								    msgid,
								    header_size,
								    max_payload_size,
								    src_payload_addr,
								    src_payload_lines,
								    src_payload_line_stride,
								    dest_ptr);
    return impl;
  }

  ActiveMessageImpl *GASNet1Module::create_active_message_impl(const NodeSet& targets,
							       unsigned short msgid,
							       size_t header_size,
							       size_t max_payload_size,
							       const void *src_payload_addr,
							       size_t src_payload_lines,
							       size_t src_payload_line_stride,
							       void *storage_base,
							       size_t storage_size)
  {
    assert(storage_size >= sizeof(GASNet1MessageImpl));
    GASNet1MessageImpl *impl = new(storage_base) GASNet1MessageImpl(targets,
								    msgid,
								    header_size,
								    max_payload_size,
								    src_payload_addr,
								    src_payload_lines,
								    src_payload_line_stride);
    return impl;
  }

  size_t GASNet1Module::recommended_max_payload(NodeID target,
						bool with_congestion,
						size_t header_size)
  {
    return gasnet_AMMaxMedium();
  }

  size_t GASNet1Module::recommended_max_payload(const NodeSet& targets,
						bool with_congestion,
						size_t header_size)
  {
    return gasnet_AMMaxMedium();
  }

  size_t GASNet1Module::recommended_max_payload(NodeID target,
						const RemoteAddress& dest_payload_addr,
						bool with_congestion,
						size_t header_size)
  {
    // RDMA uses long, but don't go above 4MB per packet for responsiveness
    size_t maxlong = gasnet_AMMaxLongRequest();
    return std::min(maxlong, size_t(4 << 20));
  }

  size_t GASNet1Module::recommended_max_payload(NodeID target,
						const void *data, size_t bytes_per_line,
						size_t lines, size_t line_stride,
						bool with_congestion,
						size_t header_size)
  {
    return gasnet_AMMaxMedium();
  }

  size_t GASNet1Module::recommended_max_payload(const NodeSet& targets,
							const void *data, size_t bytes_per_line,
							size_t lines, size_t line_stride,
						bool with_congestion,
						size_t header_size)
  {
    return gasnet_AMMaxMedium();
  }

  size_t GASNet1Module::recommended_max_payload(NodeID target,
						const void *data, size_t bytes_per_line,
						size_t lines, size_t line_stride,
						const RemoteAddress& dest_payload_addr,
						bool with_congestion,
						size_t header_size)
  {
    // RDMA uses long, but don't go above 4MB per packet for responsiveness
    // we also need the source to be contiguous, so clamp at a single
    //  line of the source data
    size_t maxlong = gasnet_AMMaxLongRequest();
    return std::min(maxlong, std::min(bytes_per_line, size_t(4 << 20)));
  }
  

}; // namespace Realm
