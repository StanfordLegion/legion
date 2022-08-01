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

// Memory implementations for Realm

#ifndef REALM_MEMORY_IMPL_H
#define REALM_MEMORY_IMPL_H

#include "realm/memory.h"
#include "realm/id.h"
#include "realm/network.h"

#include "realm/activemsg.h"
#include "realm/operation.h"
#include "realm/profiling.h"
#include "realm/sampling.h"

#include "realm/event_impl.h"
#include "realm/rsrv_impl.h"

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

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE MemoryImpl {
  public:
    enum MemoryKind {
      MKIND_SYSMEM,  // directly accessible from CPU
      MKIND_GLOBAL,  // accessible via GASnet (spread over all nodes)
      MKIND_RDMA,    // remote, but accessible via RDMA
      MKIND_REMOTE,  // not accessible

      // defined even if REALM_USE_CUDA isn't
      // TODO: make kinds more extensible
      MKIND_GPUFB,   // GPU framebuffer memory (accessible via cudaMemcpy)
      MKIND_MANAGED, // memory that is coherent for both CPU and GPU

      MKIND_ZEROCOPY, // CPU memory, pinned for GPU access
      MKIND_DISK,    // disk memory accessible by owner node
      MKIND_FILE,    // file memory accessible by owner node
#ifdef REALM_USE_HDF5
      MKIND_HDF,     // HDF memory accessible by owner node
#endif
    };

    MemoryImpl(Memory _me, size_t _size,
	       MemoryKind _kind, Memory::Kind _lowlevel_kind,
	       NetworkSegment *_segment);

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

    // default implementation falls through (directly or indirectly) to
    //  allocate_storage_immediate -  method need only be overridden by
    //  memories that support deferred allocation
    virtual AllocationResult allocate_storage_deferrable(RegionInstanceImpl *inst,
							 bool need_alloc_result,
							 Event precondition);

    // release storage associated with an instance - this falls through to
    //  release_storage_immediate similarly to the above
    virtual void release_storage_deferrable(RegionInstanceImpl *inst,
					    Event precondition);

    // should only be called by RegionInstance::DeferredCreate or from
    //  allocate_storage_deferrable
    virtual AllocationResult allocate_storage_immediate(RegionInstanceImpl *inst,
							bool need_alloc_result,
							bool poisoned,
							TimeLimit work_until) = 0;

    // should only be called by RegionInstance::DeferredDestroy or from
    //  release_storage_deferrable
    virtual void release_storage_immediate(RegionInstanceImpl *inst,
					   bool poisoned,
					   TimeLimit work_until) = 0;

    // helpers used by the above when an instance being allocated or released
    //  is using an external resource
    virtual bool attempt_register_external_resource(RegionInstanceImpl *inst,
                                                    size_t& inst_offset);
    virtual void unregister_external_resource(RegionInstanceImpl *inst);

    // for re-registration purposes, generate an ExternalInstanceResource *
    //  (if possible) for a given instance, or a subset of one
    virtual ExternalInstanceResource *generate_resource_info(RegionInstanceImpl *inst,
							     const IndexSpaceGeneric *subspace,
							     span<const FieldID> fields,
							     bool read_only);

    // TODO: try to rip these out?
    virtual void get_bytes(off_t offset, void *dst, size_t size) = 0;
    virtual void put_bytes(off_t offset, const void *src, size_t size) = 0;

    virtual void *get_direct_ptr(off_t offset, size_t size) = 0;

    virtual void *get_inst_ptr(RegionInstanceImpl *inst,
			       off_t offset, size_t size);

    // gets info related to rdma access from other nodes
    const ByteArray *get_rdma_info(NetworkModule *network) const;
    
    virtual bool get_remote_addr(off_t offset, RemoteAddress& remote_addr);

    // gets the network segment info for potential registration
    NetworkSegment *get_network_segment();

    Memory::Kind get_kind(void) const;

    // TODO: lift into a helper superclass?
    template <typename T>
    T *find_module_specific();
    template <typename T>
    const T *find_module_specific() const;

    void add_module_specific(ModuleSpecificInfo *info);

    struct InstanceList {
      std::vector<RegionInstanceImpl *> instances;
      std::vector<size_t> free_list;
      Mutex mutex;
    };

  public:
    Memory me;
    size_t size;
    MemoryKind kind;
    Memory::Kind lowlevel_kind;
    NetworkSegment *segment;
    ModuleSpecificInfo *module_specific;

    // we keep a dedicated instance list for locally created
    //  instances, but we use a map indexed by creator node for others,
    //  and protect lookups in it with its own mutex
    std::map<NodeID, InstanceList *> instances_by_creator;
    Mutex instance_map_mutex;
    InstanceList local_instances;
  };

  class MemSpecificInfo {
  public:
    MemSpecificInfo();
    virtual ~MemSpecificInfo() {}

    MemSpecificInfo *next;
  };

  class PendingIBRequests {
  public:
    PendingIBRequests(NodeID _sender, uintptr_t _req_op,
                      unsigned _count, unsigned _first_req, unsigned _current_req);
    PendingIBRequests(NodeID _sender, uintptr_t _req_op,
                      unsigned _count, unsigned _first_req, unsigned _current_req,
                      const Memory *_memories, const size_t *_sizes,
                      const off_t *_offsets);

    PendingIBRequests *next_req;
    NodeID sender;
    uintptr_t req_op;
    unsigned count;
    unsigned first_req;
    unsigned current_req;
    std::vector<Memory> memories;
    std::vector<size_t> sizes;
    std::vector<off_t> offsets;
  };

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

    // a memory that manages its own allocations
    class LocalManagedMemory : public MemoryImpl {
    public:
      LocalManagedMemory(Memory _me, size_t _size, MemoryKind _kind,
			 size_t _alignment, Memory::Kind _lowlevel_kind,
			 NetworkSegment *_segment);

      virtual ~LocalManagedMemory(void);

      virtual AllocationResult allocate_storage_deferrable(RegionInstanceImpl *inst,
							   bool need_alloc_result,
							   Event precondition);

      virtual void release_storage_deferrable(RegionInstanceImpl *inst,
					      Event precondition);

      virtual AllocationResult allocate_storage_immediate(RegionInstanceImpl *inst,
							  bool need_alloc_result,
							  bool poisoned,
							  TimeLimit work_until);

      virtual void release_storage_immediate(RegionInstanceImpl *inst,
					     bool poisoned,
					     TimeLimit work_until);

    protected:
      // for internal use by allocation routines - must be called with
      //  allocator_mutex held!
      AllocationResult attempt_deferrable_allocation(RegionInstanceImpl *inst,
						     size_t bytes,
						     size_t alignment,
						     size_t& inst_offset);

      // attempts to satisfy pending allocations based on reordering releases to
      //  move the ready ones first - assumes 'release_allocator' has been
      //  properly maintained
      bool attempt_release_reordering(std::vector<std::pair<RegionInstanceImpl *, size_t> >& successful_allocs);

    public:
      size_t alignment;

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

    class LocalCPUMemory : public LocalManagedMemory {
    public:
      static const size_t ALIGNMENT = 256;

      LocalCPUMemory(Memory _me, size_t _size, int numa_node, Memory::Kind _lowlevel_kind,
		     void *prealloc_base = 0,
		     NetworkSegment *_segment = 0);

      virtual ~LocalCPUMemory(void);

      // LocalCPUMemory supports ExternalMemoryResource
      virtual bool attempt_register_external_resource(RegionInstanceImpl *inst,
                                                      size_t& inst_offset);
      virtual void unregister_external_resource(RegionInstanceImpl *inst);

      // for re-registration purposes, generate an ExternalInstanceResource *
      //  (if possible) for a given instance, or a subset of one
      virtual ExternalInstanceResource *generate_resource_info(RegionInstanceImpl *inst,
							       const IndexSpaceGeneric *subspace,
							       span<const FieldID> fields,
							       bool read_only);

      virtual void get_bytes(off_t offset, void *dst, size_t size);
      virtual void put_bytes(off_t offset, const void *src, size_t size);
      virtual void *get_direct_ptr(off_t offset, size_t size);

    public:
      const int numa_node;
    public: //protected:
      char *base, *base_orig;
      bool prealloced;
      NetworkSegment local_segment;
    };

    class DiskMemory : public LocalManagedMemory {
    public:
      static const size_t ALIGNMENT = 256;

      DiskMemory(Memory _me, size_t _size, std::string _file);

      virtual ~DiskMemory(void);

      virtual void get_bytes(off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);

    public:
      int fd; // file descriptor
      std::string file;  // file name
    };

    class FileMemory : public MemoryImpl {
    public:
      FileMemory(Memory _me);

      virtual ~FileMemory(void);

      virtual void get_bytes(off_t offset, void *dst, size_t size);
      void get_bytes(ID::IDType inst_id, off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);
      void put_bytes(ID::IDType inst_id, off_t offset, const void *src, size_t size);
      virtual void *get_direct_ptr(off_t offset, size_t size);

      virtual AllocationResult allocate_storage_immediate(RegionInstanceImpl *inst,
							  bool need_alloc_result,
							  bool poisoned,
							  TimeLimit work_until);

      virtual void release_storage_immediate(RegionInstanceImpl *inst,
					     bool poisoned,
					     TimeLimit work_until);

      // FileMemory supports ExternalFileResource
      virtual bool attempt_register_external_resource(RegionInstanceImpl *inst,
                                                      size_t& inst_offset);
      virtual void unregister_external_resource(RegionInstanceImpl *inst);

      // the 'mem_specific' data for a file instance contains OpenFileInfo
      class OpenFileInfo : public MemSpecificInfo {
      public:
	int fd;
	size_t offset;
      };
    };

    class REALM_INTERNAL_API_EXTERNAL_LINKAGE RemoteMemory : public MemoryImpl {
    public:
      RemoteMemory(Memory _me, size_t _size, Memory::Kind k,
		   MemoryKind mk = MKIND_REMOTE);
      virtual ~RemoteMemory(void);

      virtual AllocationResult allocate_storage_deferrable(RegionInstanceImpl *inst,
							   bool need_alloc_result,
							   Event precondition);

      virtual void release_storage_deferrable(RegionInstanceImpl *inst,
					      Event precondition);

      virtual AllocationResult allocate_storage_immediate(RegionInstanceImpl *inst,
							  bool need_alloc_result,
							  bool poisoned,
							  TimeLimit work_until);

      virtual void release_storage_immediate(RegionInstanceImpl *inst,
					     bool poisoned,
					     TimeLimit work_until);

      // these are disallowed on a remote memory
      virtual off_t alloc_bytes_local(size_t size);
      virtual void free_bytes_local(off_t offset, size_t size);
      
      virtual void get_bytes(off_t offset, void *dst, size_t size);
      virtual void put_bytes(off_t offset, const void *src, size_t size);
      virtual void *get_direct_ptr(off_t offset, size_t size);
    };


    // active messages

    struct MemStorageAllocRequest {
      Memory memory;
      RegionInstance inst;
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
				 const void *data, size_t datalen,
				 TimeLimit work_until);

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

}; // namespace Realm

#endif // ifndef REALM_MEM_IMPL_H

#include "realm/mem_impl.inl"
