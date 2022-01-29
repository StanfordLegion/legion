
/* Copyright 2022 Stanford University, NVIDIA Corporation, Argonne National Laboratory
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

// MPI network module implementation for Realm

#include "realm/network.h"

#include "realm/mpi/mpi_module.h"
#include "realm/mpi/am_mpi.h"

#include "realm/runtime_impl.h"
#include "realm/mem_impl.h"
#include "realm/transfer/ib_memory.h"

void enqueue_message(int target, int msgid,
                     const void *args, size_t arg_size,
                     const void *payload, size_t payload_size,
		     size_t payload_lines, size_t payload_line_stride,
		     MPI_Aint dst_offset, void *remote_comp)
{
    Realm::MPI::AMSend(target, msgid, arg_size, payload_size,
		       (const char *) args, (const char *) payload,
		       payload_lines, payload_line_stride,
		       (dst_offset >= 0),
		       dst_offset, remote_comp);
}

namespace Realm {

    ////////////////////////////////////////////////////////////////////////
    //
    // class MPIMemory
    //

    /* A block of memory spread across multiple processes
     * To spread the memory access evenly, it uses round-robin policy with "memory_stride" chunk size.
     * NOTE: the MPI window is assumed to be opened with MPI_Win_unlock_all and to be freed by "user"
     */
    class MPIMemory : public LocalManagedMemory {
    public:
        static const size_t MEMORY_STRIDE = 1024;

        MPIMemory(Memory _me, size_t size_per_node, MPI_Win _win, NetworkModule *_network);

        virtual ~MPIMemory(void);

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
        void *baseptr; // not really needed
        MPI_Win win;
        NetworkModule *my_network;
    };

    MPIMemory::MPIMemory(Memory _me, size_t size_per_node,
                           MPI_Win _win, NetworkModule *_network)
        : LocalManagedMemory(_me, 0 /* we'll calculate it below */, MKIND_GLOBAL,
			     MEMORY_STRIDE, Memory::GLOBAL_MEM, 0)
        , win(_win)
        , my_network(_network)
    {
        num_nodes = Network::max_node_id + 1;
        memory_stride = MEMORY_STRIDE;

        size = size_per_node * num_nodes;
        
        current_allocator.add_range(0, size);
    }

    MPIMemory::~MPIMemory(void)
    {
        /* upper-layer's responsibility to free the window */
    }

    void MPIMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
        char *dst_c = (char *)dst;
        while(size > 0) {
            off_t blkid = (offset / memory_stride / num_nodes);
            off_t node = (offset / memory_stride) % num_nodes;
            off_t blkoffset = offset % memory_stride;
            size_t chunk_size = memory_stride - blkoffset;
            if(chunk_size > size) chunk_size = size;

            MPI_Aint disp = blkid * memory_stride + blkoffset;
            CHECK_MPI( MPI_Get(dst_c, chunk_size, MPI_BYTE, node, disp, chunk_size, MPI_BYTE, win) );

            offset += chunk_size;
            dst_c += chunk_size;
            size -= chunk_size;
        }
        CHECK_MPI( MPI_Win_flush_all(win) );
    }

    void MPIMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
        char *src_c = (char *)src; // dropping const on purpose...
        while(size > 0) {
            off_t blkid = (offset / memory_stride / num_nodes);
            off_t node = (offset / memory_stride) % num_nodes;
            off_t blkoffset = offset % memory_stride;
            size_t chunk_size = memory_stride - blkoffset;
            if(chunk_size > size) chunk_size = size;

            MPI_Aint disp = blkid * memory_stride + blkoffset;
            CHECK_MPI( MPI_Put(src_c, chunk_size, MPI_BYTE, node, disp, chunk_size, MPI_BYTE, win) );

            offset += chunk_size;
            src_c += chunk_size;
            size -= chunk_size;
        }
        CHECK_MPI( MPI_Win_flush_all(win) );
    }

    void *MPIMemory::get_direct_ptr(off_t offset, size_t size)
    {
        return 0;  // can't give a pointer to the caller - have to use RDMA
    }

    void MPIMemory::get_batch(size_t batch_size,
                               const off_t *offsets, void * const *dsts, 
                               const size_t *sizes)
    {
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

                MPI_Aint disp = blkid * memory_stride + blkoffset;
                CHECK_MPI( MPI_Get(dst_c, chunk_size, MPI_BYTE, node, disp, chunk_size, MPI_BYTE, win) );

                dst_c += chunk_size;
                size -= chunk_size;
                blkoffset = 0;
                node = (node + 1) % num_nodes;
                if(node == 0) blkid++;
            }
        }

        CHECK_MPI( MPI_Win_flush_all(win) );
    }

    void MPIMemory::put_batch(size_t batch_size,
                               const off_t *offsets,
                               const void * const *srcs, 
                               const size_t *sizes)
    {
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

                MPI_Aint disp = blkid * memory_stride + blkoffset;
                CHECK_MPI( MPI_Put(src_c, chunk_size, MPI_BYTE, node, disp, chunk_size, MPI_BYTE, win) );

                src_c += chunk_size;
                size -= chunk_size;
                blkoffset = 0;
                node = (node + 1) % num_nodes;
                if(node == 0) blkid++;
            }
        }

        CHECK_MPI( MPI_Win_flush_all(win) );
    }

    // gets info related to rdma access from other nodes
    const ByteArray *MPIMemory::get_rdma_info(NetworkModule *network)
    {
        // provide a dummy rdma info so that we get
        //  handled by the network module instead of turned into a
        //  normal RemoteMemory
        static ByteArray dummy_rdma_info;
        if(network == my_network)
            return &dummy_rdma_info;
        else
            return 0;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class MPIRemoteMemory
    //

    /* A block of memory on a remote process
    * Parent class RemoteMemory has node id in me.memory_owner_node()
    */
    class MPIRemoteMemory : public RemoteMemory {
      public:
        MPIRemoteMemory(Memory _me, size_t _size, Memory::Kind k,
    		    int _rank, MPI_Aint _base, MPI_Win _win);

        virtual void get_bytes(off_t offset, void *dst, size_t size);
        virtual void put_bytes(off_t offset, const void *src, size_t size);

        virtual bool get_remote_addr(off_t offset, RemoteAddress& remote_addr);

      protected:
        int rank;
        MPI_Aint base;
        MPI_Win win;
    };

    MPIRemoteMemory::MPIRemoteMemory(Memory _me, size_t _size,
                                     Memory::Kind k,
                                     int _rank, MPI_Aint _base, MPI_Win _win)
        : RemoteMemory(_me, _size, k, MKIND_RDMA)
        , rank(_rank)
        , base(_base)
        , win(_win)
    {
    }

    void MPIRemoteMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
        CHECK_MPI( MPI_Get(dst, size, MPI_BYTE, rank, base + offset, size, MPI_BYTE, win) );
        CHECK_MPI( MPI_Win_flush(rank, win) );
    }

    void MPIRemoteMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
        CHECK_MPI( MPI_Put(src, size, MPI_BYTE, rank, base + offset, size, MPI_BYTE, win) );
        CHECK_MPI( MPI_Win_flush(rank, win) );
    }

    bool MPIRemoteMemory::get_remote_addr(off_t offset, RemoteAddress& remote_addr)
    {
        // MPI puts and gets use offsets, but add this memory's base in the window
        remote_addr.ptr = (base + offset);
	return true;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class MPIIBMemory
    //

    class MPIIBMemory : public IBMemory {
    public:
      MPIIBMemory(Memory _me, size_t _size, Memory::Kind k, MPI_Aint _base);

      virtual bool get_remote_addr(off_t offset, RemoteAddress& remote_addr);

    protected:
      MPI_Aint base;
    };

    MPIIBMemory::MPIIBMemory(Memory _me, size_t _size, Memory::Kind k,
			     MPI_Aint _base)
      : IBMemory(_me, _size, MKIND_REMOTE, k, 0, 0)
      , base(_base)
    {}

    bool MPIIBMemory::get_remote_addr(off_t offset, RemoteAddress& remote_addr)
    {
        // MPI puts and gets use offsets, but add this memory's base in the window
        remote_addr.ptr = (base + offset);
	return true;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // struct CompletionList
    //

    struct CompletionList {
      size_t bytes;

      static const size_t TOTAL_CAPACITY = 256;
      typedef char Storage_unaligned[TOTAL_CAPACITY];
      REALM_ALIGNED_TYPE_CONST(Storage_aligned, Storage_unaligned,
			       Realm::CompletionCallbackBase::ALIGNMENT);
      Storage_aligned storage;
    };


    ////////////////////////////////////////////////////////////////////////
    //
    // class MPIMessageImpl
    //

    class MPIMessageImpl : public ActiveMessageImpl {
      public:
        MPIMessageImpl(NodeID _target,
                        unsigned short _msgid,
                        size_t _header_size,
		        size_t _max_payload_size,
		        const void *_src_payload_addr,
		        size_t _src_payload_lines,
		        size_t _src_payload_line_stride);
        MPIMessageImpl(NodeID _target,
                        unsigned short _msgid,
                        size_t _header_size,
                        size_t _max_payload_size,
		        const void *_src_payload_addr,
		        size_t _src_payload_lines,
		        size_t _src_payload_line_stride,
		        const RemoteAddress& _dest_payload_addr);
        MPIMessageImpl(const Realm::NodeSet &_targets,
                        unsigned short _msgid,
                        size_t _header_size,
		        size_t _max_payload_size,
		        const void *_src_payload_addr,
		        size_t _src_payload_lines,
		        size_t _src_payload_line_stride);

        virtual ~MPIMessageImpl();

        // reserves space for a local/remote completion - caller will
        //  placement-new the completion at the provided address
        virtual void *add_local_completion(size_t size);
        virtual void *add_remote_completion(size_t size);

        virtual void commit(size_t act_payload_size);
        virtual void cancel();

      protected:
        /* header_base, payload_base, playload_size */
        NodeID target;
        Realm::NodeSet targets;
        bool is_multicast;
        const void *src_payload_addr;
        size_t src_payload_lines;
        size_t src_payload_line_stride;
        MPI_Aint dest_payload_offset;
        size_t header_size;
        CompletionList *local_comp, *remote_comp;

        unsigned short msgid;
        unsigned long msg_header;
        // nothing should appear after 'msg_header'
    };

    MPIMessageImpl::MPIMessageImpl(NodeID _target,
                                     unsigned short _msgid,
                                     size_t _header_size,
                                     size_t _max_payload_size,
				     const void *_src_payload_addr,
				     size_t _src_payload_lines,
				     size_t _src_payload_line_stride)
        : target(_target)
        , is_multicast(false)
	, src_payload_addr(_src_payload_addr)
	, src_payload_lines(_src_payload_lines)
	, src_payload_line_stride(_src_payload_line_stride)
	, dest_payload_offset(-1)
        , header_size(_header_size)
	, local_comp(0)
	, remote_comp(0)
        , msgid(_msgid)
    {
        if(_max_payload_size && (src_payload_addr == 0)) {
            payload_base = reinterpret_cast<char *>(malloc(_max_payload_size));
        } else {
            payload_base = 0;
        }
        payload_size = _max_payload_size;
        header_base = &msg_header;
    }

    MPIMessageImpl::MPIMessageImpl(NodeID _target,
                                     unsigned short _msgid,
                                     size_t _header_size,
                                     size_t _max_payload_size,
				     const void *_src_payload_addr,
				     size_t _src_payload_lines,
				     size_t _src_payload_line_stride,
				     const RemoteAddress& _dest_payload_addr)
        : target(_target)
        , is_multicast(false)
	, src_payload_addr(_src_payload_addr)
	, src_payload_lines(_src_payload_lines)
	, src_payload_line_stride(_src_payload_line_stride)
	, dest_payload_offset(_dest_payload_addr.ptr)
        , header_size(_header_size)
	, local_comp(0)
	, remote_comp(0)
        , msgid(_msgid)
    {
        if(_max_payload_size && (src_payload_addr == 0)) {
            payload_base = reinterpret_cast<char *>(malloc(_max_payload_size));
        } else {
            payload_base = 0;
        }
        payload_size = _max_payload_size;
        header_base = &msg_header;
    }

    MPIMessageImpl::MPIMessageImpl(const Realm::NodeSet &_targets,
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
        , dest_payload_offset(-1)
        , header_size(_header_size)
	, local_comp(0)
	, remote_comp(0)
        , msgid(_msgid)
    {
        if(_max_payload_size && (src_payload_addr == 0)) {
            payload_base = reinterpret_cast<char *>(malloc(_max_payload_size));
        } else {
            payload_base = 0;
        }
        payload_size = _max_payload_size;
        header_base = &msg_header;
    }

    MPIMessageImpl::~MPIMessageImpl()
    {
    }

    void *MPIMessageImpl::add_local_completion(size_t size)
    {
      if(local_comp == 0) {
	local_comp = new CompletionList;
	local_comp->bytes = 0;
      }
      size_t ofs = local_comp->bytes;
      local_comp->bytes += size;
      assert(local_comp->bytes <= CompletionList::TOTAL_CAPACITY);
      return (local_comp->storage + ofs);
    }

    void *MPIMessageImpl::add_remote_completion(size_t size)
    {
      if(remote_comp == 0) {
	remote_comp = new CompletionList;
	remote_comp->bytes = 0;
      }
      size_t ofs = remote_comp->bytes;
      remote_comp->bytes += size;
      assert(remote_comp->bytes <= CompletionList::TOTAL_CAPACITY);
      return (remote_comp->storage + ofs);
    }

    void MPIMessageImpl::commit(size_t act_payload_size)
    {
        if(is_multicast) {
	    assert(dest_payload_offset < 0);
	    assert(remote_comp == 0);
	    for(NodeSet::const_iterator it = targets.begin();
		it != targets.end();
		++it)
	      if(src_payload_addr != 0)
		enqueue_message(*it, msgid, &msg_header, header_size,
				src_payload_addr, act_payload_size,
				src_payload_lines, src_payload_line_stride,
				-1, 0);
	      else
		enqueue_message(*it, msgid, &msg_header, header_size,
				payload_base, act_payload_size, 0, 0, -1, 0);
        } else {
	    if(src_payload_addr != 0)
	      enqueue_message(target, msgid, &msg_header, header_size,
			      src_payload_addr, act_payload_size,
			      src_payload_lines, src_payload_line_stride,
			      dest_payload_offset, remote_comp);
	    else
	      enqueue_message(target, msgid, &msg_header, header_size,
			      payload_base, act_payload_size, 0, 0,
			      dest_payload_offset, remote_comp);
        }
	if(payload_size && (src_payload_addr == 0))
	  free(payload_base);
	// we're only doing blocking transfers right now, so we can always do
	//  local completion here
	if(local_comp != 0) {
	  CompletionCallbackBase::invoke_all(local_comp->storage,
					     local_comp->bytes);
	  CompletionCallbackBase::destroy_all(local_comp->storage,
					      local_comp->bytes);
	  delete local_comp;
	}	  
    }

    // callback to invoke remote completions
    void MPI::AMComplete(void *comp)
    {
        CompletionList *remote_comp = static_cast<CompletionList *>(comp);
	CompletionCallbackBase::invoke_all(remote_comp->storage,
					   remote_comp->bytes);
	CompletionCallbackBase::destroy_all(remote_comp->storage,
					    remote_comp->bytes);
	delete remote_comp;
    }

    void MPIMessageImpl::cancel()
    {
	if(payload_size && (src_payload_addr == 0))
            free(payload_base);
    }

    /*---- polling threads ---------------------*/
    class AM_Manager {
    public:
        AM_Manager(){
            core_rsrv = NULL;
            p_thread = NULL;
            shutdown_flag.store(false);
            polls_done.store(0);
        }
        ~AM_Manager(void){}
        void init_corereservation(Realm::CoreReservationSet& crs){
            core_rsrv = new Realm::CoreReservation("AM workers", crs, Realm::CoreReservationParameters());
        }
        void release_corereservation() {
            delete core_rsrv;
            core_rsrv = NULL;
        }
        void start_thread(){
            Realm::ThreadLaunchParameters tlp;
            p_thread = Realm::Thread::create_kernel_thread<AM_Manager, &AM_Manager::thread_loop>(this, tlp, *core_rsrv);
        }
        void thread_loop(void){
            while (true) {
                if (shutdown_flag.load()) {
                    break;
                }
                Realm::MPI::AMPoll();
                polls_done.fetch_add(1);
            }
        }
        void stop_threads(){
            shutdown_flag.store(true);
            p_thread->join();
            delete p_thread;
            p_thread = NULL;
        }
        void ensure_polling_progress(){
            assert(!shutdown_flag.load());
            unsigned prev = polls_done.load();
            while(prev == polls_done.load())
              sched_yield();
        }
    protected:
        Realm::CoreReservation *core_rsrv;
        Realm::Thread *p_thread;
        Realm::atomic<bool> shutdown_flag;
        Realm::atomic<unsigned> polls_done;
    };

    AM_Manager g_am_manager;
  
  ////////////////////////////////////////////////////////////////////////
  //
  // class MPIModule
  //

  MPIModule::MPIModule(void)
    : NetworkModule("mpi")
    , g_am_win(MPI_WIN_NULL)
    , global_mem_size(0)
    /*
    , active_msg_worker_threads(1)
    , active_msg_handler_threads(1)
    , amsg_stack_size(2 << 20)
    */
  {}
  
  /*static*/ NetworkModule *MPIModule::create_network_module(RuntimeImpl *runtime,
								 int *argc,
								 const char ***argv)
  {
    // runtime, argc, argv are not used
#ifdef DEBUG_REALM_STARTUP
    { // we don't have rank IDs yet, so everybody gets to spew
      char s[80];
      gethostname(s, 79);
      strcat(s, " enter MPI init");
      TimeStamp ts(s, false);
      fflush(stdout);
    }
#endif
    int mpi_rank, mpi_size;
    Realm::MPI::AM_Init(&mpi_rank, &mpi_size);
    Network::my_node_id = mpi_rank;
    Network::max_node_id = mpi_size - 1;
    Network::all_peers.add_range(0, mpi_size - 1);
    Network::all_peers.remove(mpi_rank);
#ifdef DEBUG_REALM_STARTUP
    { // once we're convinced there isn't skew here, reduce this to rank 0
      char s[80];
      gethostname(s, 79);
      strcat(s, " exit MPI init");
      TimeStamp ts(s, false);
      fflush(stdout);
    }
#endif
    return new MPIModule;
  }

  // actual parsing of the command line should wait until here if at all
  //  possible
  void MPIModule::parse_command_line(RuntimeImpl *runtime,
					 std::vector<std::string>& cmdline)
  {
    CommandLineParser cp;
    cp.add_option_int_units("-ll:gsize", global_mem_size, 'm')
    //  .add_option_int("-ll:amsg", active_msg_worker_threads)
    //  .add_option_int("-ll:ahandlers", active_msg_handler_threads)
    //  .add_option_int_units("-ll:astack", amsg_stack_size, 'm')
    ;
    
    bool ok = cp.parse_command_line(cmdline);
    assert(ok);
  }

  // "attaches" to the network, if that is meaningful - attempts to
  //  bind/register/(pick your network-specific verb) the requested memory
  //  segments with the network
  void MPIModule::attach(RuntimeImpl *runtime,
			     std::vector<NetworkSegment *>& segments)
  {
    size_t attach_size = global_mem_size;
    for(std::vector<NetworkSegment *>::iterator it = segments.begin(); it != segments.end(); ++it) {
        if((*it)->bytes == 0) continue;
        if((*it)->base != 0) continue;
	if((*it)->memtype != NetworkSegmentInfo::HostMem) continue;
        attach_size += (*it)->bytes;
    }

    void *baseptr;
    CHECK_MPI( MPI_Win_allocate(attach_size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &baseptr, &g_am_win) );
    CHECK_MPI( MPI_Win_lock_all(0, g_am_win) );

    Realm::MPI::AM_init_long_messages(g_am_win, baseptr,
				      runtime->message_manager);

    int num_nodes = Network::max_node_id + 1;
    int size_p = sizeof(baseptr);
    g_am_bases = (void **) malloc(size_p * num_nodes);
    CHECK_MPI( MPI_Allgather(&baseptr, size_p, MPI_BYTE, g_am_bases, size_p, MPI_BYTE, MPI_COMM_WORLD) );

    char *seg_base = reinterpret_cast<char *>(baseptr);
    seg_base += global_mem_size;
    for(std::vector<NetworkSegment *>::iterator it = segments.begin(); it != segments.end(); ++it) {
        if((*it)->bytes == 0) continue;
        if((*it)->base != 0) continue;
	if((*it)->memtype != NetworkSegmentInfo::HostMem) continue;
        (*it)->base = seg_base;
        (*it)->add_rdma_info(this, &seg_base, sizeof(void *));
        seg_base += (*it)->bytes;
    }

    g_am_manager.init_corereservation(*(runtime->core_reservations));
    g_am_manager.start_thread();
  }

  void MPIModule::create_memories(RuntimeImpl *runtime)
  {
    if(global_mem_size > 0) {
      // only node 0 creates the global memory
      if(Network::my_node_id == 0) {
	Memory m = runtime->next_local_memory_id();
	MPIMemory *mem = new MPIMemory(m, global_mem_size, g_am_win, this);
	runtime->add_memory(mem);
      }
    }
  }
  
  // detaches from the network
  void MPIModule::detach(RuntimeImpl *runtime,
			     std::vector<NetworkSegment *>& segments)
  {
    g_am_manager.stop_threads();
    if (g_am_win != MPI_WIN_NULL) {
        CHECK_MPI( MPI_Win_unlock_all(g_am_win) );
        CHECK_MPI( MPI_Win_free(&g_am_win) );
    }
    g_am_manager.release_corereservation();
    Realm::MPI::AM_Finalize();
    free(g_am_bases);
  }

  // collective communication within this network
  void MPIModule::barrier(void)
  {
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
  }
    
  void MPIModule::broadcast(NodeID root, const void *val_in, void *val_out, size_t bytes)
  {
    if (Network::my_node_id == root) {
        memcpy(val_out, val_in, bytes);
    }
    CHECK_MPI( MPI_Bcast(val_out, bytes, MPI_BYTE, root, MPI_COMM_WORLD) );
  }
  
  void MPIModule::gather(NodeID root, const void *val_in, void *vals_out, size_t bytes)
  {
    CHECK_MPI( MPI_Gather(val_in, bytes, MPI_BYTE, vals_out, bytes, MPI_BYTE, root, MPI_COMM_WORLD) );
  }

  size_t MPIModule::sample_messages_received_count(void)
  {
    // we don't have the right count to match the incoming message manager
    //  (since we count completion replies too), so use a count of 0 that
    //  merely waits until the incoming manager is temporarily idle
    return 0;
  }

  static unsigned long prev_total_rcvd = 0;
  bool MPIModule::check_for_quiescence(size_t sampled_receive_count)
  {
    // ensure some progress happens on the poller before each quiescence check
    g_am_manager.ensure_polling_progress();

    // add up the total messages sent/rcvd by anybody since last time we tried
    unsigned long my_counts[2], totals[2];
    my_counts[0] = MPI::messages_sent.load();
    my_counts[1] = MPI::messages_rcvd.load();
    CHECK_MPI( MPI_Allreduce(my_counts, totals,
                            2, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD) );

    // we're quiescent if:
    //  a) the total messages rcvd is the same as total sent (i.e. none in
    //      flight), and
    //  b) the total messages rcvd is the same as last attempt (i.e. no new
    //      messages showed up during the check)
    bool quiesced = ((totals[0] == totals[1]) &&
                     (totals[1] == prev_total_rcvd));
    prev_total_rcvd = totals[1];
    return quiesced;
  }

  // used to create a remote proxy for a memory
  MemoryImpl *MPIModule::create_remote_memory(Memory m, size_t size, Memory::Kind kind,
					       const ByteArray& rdma_info)
  {
    if(kind == Memory::GLOBAL_MEM) {
      // just create a new window
      size_t size_per_node = size / (Network::max_node_id + 1);
      return new MPIMemory(m, size_per_node, g_am_win, this);
    } else {
      // it's some other kind of memory that we pre-registered
      // rdma info should be the pointer in the remote address space
      assert(rdma_info.size() == sizeof(void *));
      char *base;
      memcpy(&base, rdma_info.base(), sizeof(void *));
      // get displacement to the window
      int rank = ID(m).memory_owner_node();
      MPI_Aint disp = (MPI_Aint) base - (MPI_Aint) g_am_bases[rank];

      return new MPIRemoteMemory(m, size, kind, rank, disp, g_am_win);
    }
  }
  
  IBMemory *MPIModule::create_remote_ib_memory(Memory m, size_t size, Memory::Kind kind,
					       const ByteArray& rdma_info)
  {
    // rdma info should be the pointer in the remote address space
    assert(rdma_info.size() == sizeof(void *));
    char *base;
    memcpy(&base, rdma_info.base(), sizeof(void *));
    // get displacement to the window
    int rank = ID(m).memory_owner_node();
    MPI_Aint disp = (MPI_Aint) base - (MPI_Aint) g_am_bases[rank];

    return new MPIIBMemory(m, size, kind, disp);
  }
  
  ActiveMessageImpl *MPIModule::create_active_message_impl(NodeID target,
							    unsigned short msgid,
							    size_t header_size,
							    size_t max_payload_size,
							    const void *src_payload_addr,
							    size_t src_payload_lines,
							    size_t src_payload_line_stride,
							    void *storage_base,
							    size_t storage_size)
  {
    assert(storage_size >= sizeof(MPIMessageImpl));
    MPIMessageImpl *impl = new(storage_base) MPIMessageImpl(target,
						              msgid,
							      header_size,
							      max_payload_size,
							      src_payload_addr,
							      src_payload_lines,
							      src_payload_line_stride);
    return impl;
  }

  ActiveMessageImpl *MPIModule::create_active_message_impl(NodeID target,
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
    assert(storage_size >= sizeof(MPIMessageImpl));
    MPIMessageImpl *impl = new(storage_base) MPIMessageImpl(target,
						              msgid,
							      header_size,
							      max_payload_size,
							      src_payload_addr,
							      src_payload_lines,
							      src_payload_line_stride,
							      dest_payload_addr);
    return impl;
  }

  ActiveMessageImpl *MPIModule::create_active_message_impl(const NodeSet& targets,
							    unsigned short msgid,
							    size_t header_size,
							    size_t max_payload_size,
							    const void *src_payload_addr,
							    size_t src_payload_lines,
							    size_t src_payload_line_stride,
							    void *storage_base,
							    size_t storage_size)
  {
    assert(storage_size >= sizeof(MPIMessageImpl));
    MPIMessageImpl *impl = new(storage_base) MPIMessageImpl(targets,
							      msgid,
							      header_size,
							      max_payload_size,
							      src_payload_addr,
							      src_payload_lines,
							      src_payload_line_stride);
    return impl;
  }

  size_t MPIModule::recommended_max_payload(NodeID target,
					    bool with_congestion,
					    size_t header_size)
  {
    return (AM_BUF_SIZE - header_size);
  }

  size_t MPIModule::recommended_max_payload(const NodeSet& targets,
					    bool with_congestion,
					    size_t header_size)
  {
    return (AM_BUF_SIZE - header_size);
  }

  size_t MPIModule::recommended_max_payload(NodeID target,
					    const RemoteAddress& dest_payload_addr,
					    bool with_congestion,
					    size_t header_size)
  {
    // mostly arbitrary since we're using MPI_Put, but try to keep
    //   individual blocks from monopolizing the network interface for
    //   too long
    return 1 << 20; // 1 MB
  }
  
  size_t MPIModule::recommended_max_payload(NodeID target,
					    const void *data, size_t bytes_per_line,
					    size_t lines, size_t line_stride,
					    bool with_congestion,
					    size_t header_size)
  {
    // we don't care about source data location
    return recommended_max_payload(target, with_congestion, header_size);
  }

  size_t MPIModule::recommended_max_payload(const NodeSet& targets,
						const void *data, size_t bytes_per_line,
						size_t lines, size_t line_stride,
					    bool with_congestion,
					    size_t header_size)
  {
    // we don't care about source data location
    return recommended_max_payload(targets, with_congestion, header_size);
  }

  size_t MPIModule::recommended_max_payload(NodeID target,
					    const void *data, size_t bytes_per_line,
					    size_t lines, size_t line_stride,
					    const RemoteAddress& dest_payload_addr,
					    bool with_congestion,
					    size_t header_size)
  {
    // we don't care about source data location
    return recommended_max_payload(target, dest_payload_addr,
				   with_congestion, header_size);
  }


}; // namespace Realm
