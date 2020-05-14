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

// Memory implementations for Realm

#ifndef REALM_MEMORY_IMPL_H
#define REALM_MEMORY_IMPL_H

#include "realm/memory.h"
#include "realm/id.h"

#include "realm/activemsg.h"
#include "realm/operation.h"
#include "realm/profiling.h"
#include "realm/sampling.h"

#include "realm/event_impl.h"
#include "realm/rsrv_impl.h"

#ifdef REALM_USE_HDF5
#include <hdf5.h>
#endif

namespace Realm {

  namespace Config {
    // if true, Realm memories attempt to satisfy instance allocation requests
    //  on the basis of deferred instance destructions
    extern bool deferred_instance_allocation;
  };

  class RegionInstanceImpl;
  class NetworkModule;
  class NetworkSegment;
  class ByteArray;

  // manages a basic free list of ranges (using range type RT) and allocated
  //  ranges, which are tagged (tag type TT)
  // NOT thread-safe - must be protected from outside
  template <typename RT, typename TT>
  class BasicRangeAllocator {
  public:
    struct Range {
      //Range(RT _first, RT _last);

      RT first, last;  // half-open range: [first, last)
      unsigned prev, next;  // double-linked list of all ranges (by index)
      unsigned prev_free, next_free;  // double-linked list of just free ranges
    };

    std::map<TT, unsigned> allocated;  // direct lookup of allocated ranges by tag
#ifdef DEBUG_REALM
    std::map<RT, unsigned> by_first;   // direct lookup of all ranges by first
    // TODO: sized-based lookup of free ranges
#endif

    static const unsigned SENTINEL = 0;
    // TODO: small (medium?) vector opt
    std::vector<Range> ranges;

    BasicRangeAllocator(void);
    ~BasicRangeAllocator(void);

    void swap(BasicRangeAllocator<RT, TT>& swap_with);

    void add_range(RT first, RT last);
    bool can_allocate(TT tag, RT size, RT alignment);
    bool allocate(TT tag, RT size, RT alignment, RT& first);
    void deallocate(TT tag, bool missing_ok = false);
    bool lookup(TT tag, RT& first, RT& size);

  protected:
    unsigned first_free_range;
    unsigned alloc_range(RT first, RT last);
    void free_range(unsigned index);
  };
  
    class MemoryImpl {
    public:
      enum MemoryKind {
	MKIND_SYSMEM,  // directly accessible from CPU
	MKIND_GLOBAL,  // accessible via GASnet (spread over all nodes)
	MKIND_RDMA,    // remote, but accessible via RDMA
	MKIND_REMOTE,  // not accessible

	// defined even if REALM_USE_CUDA isn't
	// TODO: make kinds more extensible
	MKIND_GPUFB,   // GPU framebuffer memory (accessible via cudaMemcpy)

	MKIND_ZEROCOPY, // CPU memory, pinned for GPU access
	MKIND_DISK,    // disk memory accessible by owner node
	MKIND_FILE,    // file memory accessible by owner node
#ifdef REALM_USE_HDF5
	MKIND_HDF      // HDF memory accessible by owner node
#endif
      };

      MemoryImpl(Memory _me, size_t _size, MemoryKind _kind, size_t _alignment, Memory::Kind _lowlevel_kind);

      virtual ~MemoryImpl(void);

      // looks up an instance based on ID - creates a proxy object for
      //   unknown IDs (metadata must be requested explicitly)
      RegionInstanceImpl *get_instance(RegionInstance i);

      // adds a new instance to this memory, to be filled in by caller
      RegionInstanceImpl *new_instance(void);

      // releases a deleted instance so that it can be reused
      void release_instance(RegionInstance inst);
      
      // attempt to allocate storage for the specified instance
      enum AllocationResult {
	ALLOC_INSTANT_SUCCESS,
	ALLOC_INSTANT_FAILURE,
	ALLOC_DEFERRED,
	ALLOC_EVENTUAL_SUCCESS, // i.e. after a DEFERRED
	ALLOC_EVENTUAL_FAILURE,
	ALLOC_CANCELLED
      };
      virtual AllocationResult allocate_instance_storage(RegionInstance i,
							 size_t bytes,
							 size_t alignment,
							 bool need_alloc_result,
							 Event precondition, 
							 // this will be used for zero-size allocs
                    // TODO: ideally use something like (size_t)-2 here, but that will
                    //  currently confuse the file read/write path in dma land
							 size_t offset = 0);

      // release storage associated with an instance
      virtual void release_instance_storage(RegionInstanceImpl *inst,
					    Event precondition);

      virtual off_t alloc_bytes_local(size_t size);
      virtual void free_bytes_local(off_t offset, size_t size);

      virtual void get_bytes(off_t offset, void *dst, size_t size) = 0;
      virtual void put_bytes(off_t offset, const void *src, size_t size) = 0;

      virtual void apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
					size_t count, const void *entry_buffer)
      {
	assert(0);
      }

      virtual void *get_direct_ptr(off_t offset, size_t size) = 0;
      virtual int get_home_node(off_t offset, size_t size) = 0;

      // gets info related to rdma access from other nodes
      virtual const ByteArray *get_rdma_info(NetworkModule *network) { return 0; }

      Memory::Kind get_kind(void) const;

      struct InstanceList {
	std::vector<RegionInstanceImpl *> instances;
	std::vector<size_t> free_list;
	Mutex mutex;
      };
      
      // should only be called by RegionInstance::DeferredCreate
      void deferred_creation_triggered(RegionInstanceImpl *inst,
				       size_t bytes, size_t alignment,
				       bool need_alloc_result,
				       bool poisoned);

      // should only be called by RegionInstance::DeferredDestroy
      void deferred_destruction_triggered(RegionInstanceImpl *inst,
					  bool poisoned);

    protected:
      // for internal use by allocation routines - must be called with
      //  allocator_mutex held!
      AllocationResult attempt_deferrable_allocation(RegionInstanceImpl *inst,
						     size_t bytes,
						     size_t alignment,
						     size_t& inst_offset);

    public:
      Memory me;
      size_t size;
      MemoryKind kind;
      size_t alignment;
      Memory::Kind lowlevel_kind;

      // we keep a dedicated instance list for locally created
      //  instances, but we use a map indexed by creator node for others,
      //  and protect lookups in it with its own mutex
      std::map<NodeID, InstanceList *> instances_by_creator;
      Mutex instance_map_mutex;
      InstanceList local_instances;

      Mutex mutex; // protection for resizing vectors
      std::map<off_t, off_t> free_blocks;
      Mutex allocator_mutex;
      // we keep up to three heap states:
      // current: always valid - tracks all completed allocations and all
      //                releases that can be applied without risking deadlock
      // future: valid if pending_allocs exist - tracks heap state including
      //                all pending allocs and releases
      // release: valid if pending_allocs exist - models heap state with
      //                completed allocations and any ready releases
      BasicRangeAllocator<size_t, RegionInstance> current_allocator;
      BasicRangeAllocator<size_t, RegionInstance> future_allocator;
      BasicRangeAllocator<size_t, RegionInstance> release_allocator;
      unsigned cur_release_seqid;
      struct PendingAlloc {
	RegionInstanceImpl *inst;
	size_t bytes, alignment;
	unsigned last_release_seqid;
	PendingAlloc(RegionInstanceImpl *_inst, size_t _bytes, size_t _align,
		     unsigned _release_seqid);
      };
      struct PendingRelease {
	RegionInstanceImpl *inst;
	bool is_ready;
	unsigned seqid;

	PendingRelease(RegionInstanceImpl *_inst, bool _ready,
		       unsigned _seqid);
      };
      std::vector<PendingAlloc> pending_allocs;
      std::vector<PendingRelease> pending_releases;
      ProfilingGauges::AbsoluteGauge<size_t> usage, peak_usage, peak_footprint;
    };

    class LocalCPUMemory : public MemoryImpl {
    public:
      static const size_t ALIGNMENT = 256;

      LocalCPUMemory(Memory _me, size_t _size, int numa_node, Memory::Kind _lowlevel_kind,
		     void *prealloc_base = 0,
		     const NetworkSegment *_segment = 0);

      virtual ~LocalCPUMemory(void);

      virtual void get_bytes(off_t offset, void *dst, size_t size);
      virtual void put_bytes(off_t offset, const void *src, size_t size);
      virtual void *get_direct_ptr(off_t offset, size_t size);
      virtual int get_home_node(off_t offset, size_t size);

      virtual const ByteArray *get_rdma_info(NetworkModule *network);
      
    public:
      const int numa_node;
    public: //protected:
      char *base, *base_orig;
      bool prealloced;
      const NetworkSegment *segment;
    };

    class DiskMemory : public MemoryImpl {
    public:
      static const size_t ALIGNMENT = 256;

      DiskMemory(Memory _me, size_t _size, std::string _file);

      virtual ~DiskMemory(void);

      virtual void get_bytes(off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
                                       size_t count, const void *entry_buffer);

      virtual void *get_direct_ptr(off_t offset, size_t size);
      virtual int get_home_node(off_t offset, size_t size);

    public:
      int fd; // file descriptor
      std::string file;  // file name
    };

    class FileMemory : public MemoryImpl {
    public:
      static const size_t ALIGNMENT = 256;

      FileMemory(Memory _me);

      virtual ~FileMemory(void);

      virtual void get_bytes(off_t offset, void *dst, size_t size);
      void get_bytes(ID::IDType inst_id, off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);
      void put_bytes(ID::IDType inst_id, off_t offset, const void *src, size_t size);

      virtual void apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
                                       size_t count, const void *entry_buffer);

      virtual void *get_direct_ptr(off_t offset, size_t size);
      virtual int get_home_node(off_t offset, size_t size);

      int get_file_des(ID::IDType inst_id);
    public:
      std::vector<int> file_vec;
      Mutex vector_lock;
      off_t next_offset;
      std::map<off_t, int> offset_map;
    };

    class RemoteMemory : public MemoryImpl {
    public:
      RemoteMemory(Memory _me, size_t _size, Memory::Kind k,
		   MemoryKind mk = MKIND_REMOTE);
      virtual ~RemoteMemory(void);

      // these are disallowed on a remote memory
      virtual off_t alloc_bytes_local(size_t size);
      virtual void free_bytes_local(off_t offset, size_t size);
      
      virtual void get_bytes(off_t offset, void *dst, size_t size);
      virtual void put_bytes(off_t offset, const void *src, size_t size);
      virtual void *get_direct_ptr(off_t offset, size_t size);
      virtual int get_home_node(off_t offset, size_t size);

      virtual void *get_remote_addr(off_t offset);
    };


    // active messages

    struct MemStorageAllocRequest {
      Memory memory;
      RegionInstance inst;
      size_t bytes;
      unsigned alignment;
      bool need_alloc_result;
      Event precondition;

      static void handle_message(NodeID sender, const MemStorageAllocRequest &msg,
				 const void *data, size_t datalen);

    };

    struct MemStorageAllocResponse {
      RegionInstance inst;
      size_t offset;
      MemoryImpl::AllocationResult result;

      static void handle_message(NodeID sender, const MemStorageAllocResponse &msg,
				 const void *data, size_t datalen);

    };

    struct MemStorageReleaseRequest {
      Memory memory;
      RegionInstance inst;
      Event precondition;

      static void handle_message(NodeID sender, const MemStorageReleaseRequest &msg,
				 const void *data, size_t datalen);
    };

    struct MemStorageReleaseResponse {
      RegionInstance inst;

      static void handle_message(NodeID sender, const MemStorageReleaseResponse &msg,
				 const void *data, size_t datalen);
    };

    struct RemoteWriteMessage {
      Memory mem;
      off_t offset;
      unsigned sequence_id;

      static void handle_message(NodeID sender, const RemoteWriteMessage &msg,
				 const void *data, size_t datalen);
      // no simple send_request method here - see below
    };

    struct RemoteSerdezMessage {
      Memory mem;
      off_t offset;
      size_t count;
      CustomSerdezID serdez_id;
      unsigned sequence_id;

      static void handle_message(NodeID sender, const RemoteSerdezMessage &msg,
				 const void *data, size_t datalen);
      // no simple send_request method here - see below
    };

    struct RemoteReduceMessage {
      Memory mem;
      off_t offset;
      int stride;
      ReductionOpID redop_id;
      unsigned sequence_id;

      static void handle_message(NodeID sender, const RemoteReduceMessage &msg,
				 const void *data, size_t datalen);
      // no simple send_request method here - see below
    };
    class RemoteWriteFence;
    struct RemoteWriteFenceMessage {
      Memory mem;
      unsigned sequence_id;
      unsigned num_writes;
      RemoteWriteFence *fence;

      static void handle_message(NodeID sender, const RemoteWriteFenceMessage &msg,
				 const void *data, size_t datalen);
    };

    struct RemoteWriteFenceAckMessage {
      RemoteWriteFence *fence;
      // TODO: success/failure

      static void handle_message(NodeID sender, const RemoteWriteFenceAckMessage &msg,
				 const void *data, size_t datalen);
    };

    struct RemoteReduceListMessage {
      Memory mem;
      off_t offset;
      ReductionOpID redopid;

      static void handle_message(NodeID sender, const RemoteReduceListMessage &msg,
				 const void *data, size_t datalen);
    };
    
    class RemoteWriteFence : public Operation::AsyncWorkItem {
    public:
      RemoteWriteFence(Operation *op);

      virtual void request_cancellation(void);

      virtual void print(std::ostream& os) const;
    };

    // remote memory writes

    extern unsigned do_remote_write(Memory mem, off_t offset,
				    const void *data, size_t datalen,
				    unsigned sequence_id,
				    bool make_copy = false);

    extern unsigned do_remote_write(Memory mem, off_t offset,
				    const void *data, size_t datalen,
				    off_t stride, size_t lines,
				    unsigned sequence_id,
				    bool make_copy = false);
    
    extern unsigned do_remote_write(Memory mem, off_t offset,
				    const SpanList& spans, size_t datalen,
				    unsigned sequence_id,
				    bool make_copy = false);
    extern unsigned do_remote_serdez(Memory mem, off_t offset,
                                     CustomSerdezID serdez_id,
                                     const void *data, size_t datalen,
                                     unsigned sequence_id);
    extern unsigned do_remote_reduce(Memory mem, off_t offset,
				     ReductionOpID redop_id, bool red_fold,
				     const void *data, size_t count,
				     off_t src_stride, off_t dst_stride,
				     unsigned sequence_id,
				     bool make_copy = false);				     

    extern unsigned do_remote_apply_red_list(int node, Memory mem, off_t offset,
					     ReductionOpID redopid,
					     const void *data, size_t datalen,
					     unsigned sequence_id);

    extern void do_remote_fence(Memory mem, unsigned sequence_id,
                                unsigned count, RemoteWriteFence *fence);
    
}; // namespace Realm

#endif // ifndef REALM_MEM_IMPL_H

#include "realm/mem_impl.inl"
