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

#ifdef USE_HDF
#include <hdf5.h>
#endif

namespace Realm {

  class RegionInstanceImpl;

  // manages a basic free list of ranges (using range type RT) and allocated
  //  ranges, which are tagged (tag type TT)
  // NOT thread-safe - must be protected from outside
  template <typename RT, typename TT>
  class BasicRangeAllocator {
  public:
    struct Range {
      Range(RT _first, RT _last);

      RT first, last;  // half-open range: [first, last)
      Range *prev, *next;  // double-linked list of all ranges
      Range *prev_free, *next_free;  // double-linked list of just free ranges
    };

    std::map<TT, Range *> allocated;  // direct lookup of allocated ranges by tag
    std::map<RT, Range *> by_first;   // direct lookup of all ranges by first
    Range sentinel;
    // TODO: sized-based lookup of free ranges

    BasicRangeAllocator(void);
    ~BasicRangeAllocator(void);

    void add_range(RT first, RT last);
    bool allocate(TT tag, RT size, RT alignment, RT& first);
    void deallocate(TT tag);
  };
  
    class MemoryImpl {
    public:
      enum MemoryKind {
	MKIND_SYSMEM,  // directly accessible from CPU
	MKIND_GLOBAL,  // accessible via GASnet (spread over all nodes)
	MKIND_RDMA,    // remote, but accessible via RDMA
	MKIND_REMOTE,  // not accessible

	// defined even if USE_CUDA==0
	// TODO: make kinds more extensible
	MKIND_GPUFB,   // GPU framebuffer memory (accessible via cudaMemcpy)

	MKIND_ZEROCOPY, // CPU memory, pinned for GPU access
	MKIND_DISK,    // disk memory accessible by owner node
	MKIND_FILE,    // file memory accessible by owner node
#ifdef USE_HDF
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
      virtual bool allocate_instance_storage(RegionInstance i,
					     size_t bytes, size_t alignment,
					     Event precondition, 
                                             // this will be used for zero-size allocs
                    // TODO: ideally use something like (size_t)-2 here, but that will
                    //  currently confuse the file read/write path in dma land
                                             size_t offset = 0);

      // release storage associated with an instance
      virtual void release_instance_storage(RegionInstance i,
					    Event precondition);

      off_t alloc_bytes_local(size_t size);
      void free_bytes_local(off_t offset, size_t size);

      off_t alloc_bytes_remote(size_t size);
      void free_bytes_remote(off_t offset, size_t size);

      virtual off_t alloc_bytes(size_t size) = 0;
      virtual void free_bytes(off_t offset, size_t size) = 0;

      virtual void get_bytes(off_t offset, void *dst, size_t size) = 0;
      virtual void put_bytes(off_t offset, const void *src, size_t size) = 0;

      virtual void apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
					size_t count, const void *entry_buffer)
      {
	assert(0);
      }

      virtual void *get_direct_ptr(off_t offset, size_t size) = 0;
      virtual int get_home_node(off_t offset, size_t size) = 0;

      virtual void *local_reg_base(void) { return 0; };

      Memory::Kind get_kind(void) const;

      struct InstanceList {
	std::vector<RegionInstanceImpl *> instances;
	std::vector<size_t> free_list;
	GASNetHSL mutex;
      };
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
      GASNetHSL instance_map_mutex;
      InstanceList local_instances;

      GASNetHSL mutex; // protection for resizing vectors
      std::map<off_t, off_t> free_blocks;
      GASNetHSL allocator_mutex;
      BasicRangeAllocator<size_t, RegionInstance> allocator;
      ProfilingGauges::AbsoluteGauge<size_t> usage, peak_usage, peak_footprint;
    };

    class LocalCPUMemory : public MemoryImpl {
    public:
      static const size_t ALIGNMENT = 256;

      LocalCPUMemory(Memory _me, size_t _size,
		     void *prealloc_base = 0, bool _registered = false);

      virtual ~LocalCPUMemory(void);

      virtual off_t alloc_bytes(size_t size);
      virtual void free_bytes(off_t offset, size_t size);
      virtual void get_bytes(off_t offset, void *dst, size_t size);
      virtual void put_bytes(off_t offset, const void *src, size_t size);
      virtual void *get_direct_ptr(off_t offset, size_t size);
      virtual int get_home_node(off_t offset, size_t size);
      virtual void *local_reg_base(void);

    public: //protected:
      char *base, *base_orig;
      bool prealloced, registered;
    };

    class GASNetMemory : public MemoryImpl {
    public:
      static const size_t MEMORY_STRIDE = 1024;

      GASNetMemory(Memory _me, size_t size_per_node);

      virtual ~GASNetMemory(void);

      virtual off_t alloc_bytes(size_t size);

      virtual void free_bytes(off_t offset, size_t size);

      virtual void get_bytes(off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
					size_t count, const void *entry_buffer);

      virtual void *get_direct_ptr(off_t offset, size_t size);
      virtual int get_home_node(off_t offset, size_t size);

      void get_batch(size_t batch_size,
		     const off_t *offsets, void * const *dsts, 
		     const size_t *sizes);

      void put_batch(size_t batch_size,
		     const off_t *offsets, const void * const *srcs, 
		     const size_t *sizes);

    protected:
      int num_nodes;
      off_t memory_stride;
      std::vector<char *> segbases;
      //std::map<off_t, off_t> free_blocks;
    };

    class DiskMemory : public MemoryImpl {
    public:
      static const size_t ALIGNMENT = 256;

      DiskMemory(Memory _me, size_t _size, std::string _file);

      virtual ~DiskMemory(void);

      virtual off_t alloc_bytes(size_t size);

      virtual void free_bytes(off_t offset, size_t size);

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

      virtual off_t alloc_bytes(size_t size);

      virtual void free_bytes(off_t offset, size_t size);

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
      pthread_mutex_t vector_lock;
      off_t next_offset;
      std::map<off_t, int> offset_map;
    };

    class RemoteMemory : public MemoryImpl {
    public:
      RemoteMemory(Memory _me, size_t _size, Memory::Kind k, void *_regbase);
      virtual ~RemoteMemory(void);

      virtual off_t alloc_bytes(size_t size);
      virtual void free_bytes(off_t offset, size_t size);
      virtual void get_bytes(off_t offset, void *dst, size_t size);
      virtual void put_bytes(off_t offset, const void *src, size_t size);
      virtual void *get_direct_ptr(off_t offset, size_t size);
      virtual int get_home_node(off_t offset, size_t size);

    public:
      void *regbase;
    };


    // active messages

    struct MemStorageAllocRequest {
      struct RequestArgs {
	Memory memory;
	RegionInstance inst;
	size_t bytes;
	size_t alignment;
        size_t offset;
	Event precondition;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<MEM_STORAGE_ALLOC_REQ_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(NodeID target,
			       Memory memory, RegionInstance inst,
			       size_t bytes, size_t alignment,
			       Event precondition, size_t offset);
    };

    struct MemStorageAllocResponse {
      struct RequestArgs {
	RegionInstance inst;
	size_t offset;
	bool success;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<MEM_STORAGE_ALLOC_RESP_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(NodeID target,
			       RegionInstance inst,
			       size_t offset, bool success);
    };

    struct MemStorageReleaseRequest {
      struct RequestArgs {
	Memory memory;
	RegionInstance inst;
	Event precondition;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<MEM_STORAGE_RELEASE_REQ_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(NodeID target,
			       Memory memory,
			       RegionInstance inst,
			       Event precondition);
    };

    struct MemStorageReleaseResponse {
      struct RequestArgs {
	RegionInstance inst;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<MEM_STORAGE_RELEASE_RESP_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(NodeID target,
			       RegionInstance inst);
    };

    struct RemoteMemAllocRequest {
      struct RequestArgs {
	int sender;
	void *resp_ptr;
	Memory memory;
	size_t size;
      };

      struct ResponseArgs {
	void *resp_ptr;
	off_t offset;
      };

      static void handle_request(RequestArgs args);
      static void handle_response(ResponseArgs args);

      typedef ActiveMessageShortNoReply<REMOTE_MALLOC_MSGID,
 	                                RequestArgs,
	                                handle_request> Request;

      typedef ActiveMessageShortNoReply<REMOTE_MALLOC_RPLID,
 	                                ResponseArgs,
	                                handle_response> Response;

      static off_t send_request(NodeID target, Memory memory, size_t size);
    };

    struct RemoteWriteMessage {
      struct RequestArgs : public BaseMedium {
	Memory mem;
	off_t offset;
	unsigned sender;
	unsigned sequence_id;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<REMOTE_WRITE_MSGID,
				         RequestArgs,
				         handle_request> Message;

      // no simple send_request method here - see below
    };

    struct RemoteSerdezMessage {
      struct RequestArgs : public BaseMedium {
        Memory mem;
        off_t offset;
        size_t count;
        CustomSerdezID serdez_id;
        unsigned sender;
        unsigned sequence_id;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<REMOTE_SERDEZ_MSGID,
                                         RequestArgs,
                                         handle_request> Message;

      // no simple send_request method here - see below
    };

    struct RemoteReduceMessage {
      struct RequestArgs : public BaseMedium {
	Memory mem;
	off_t offset;
	int stride;
	ReductionOpID redop_id;
	//bool red_fold;
	unsigned sender;
	unsigned sequence_id;
      };
      
      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<REMOTE_REDUCE_MSGID,
				         RequestArgs,
				         handle_request> Message;

      // no simple send_request method here - see below
    };

    struct RemoteReduceListMessage {
      struct RequestArgs : public BaseMedium {
	Memory mem;
	off_t offset;
	ReductionOpID redopid;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);
      
      typedef ActiveMessageMediumNoReply<REMOTE_REDLIST_MSGID,
				         RequestArgs,
				         handle_request> Message;

      static void send_request(NodeID target, Memory mem, off_t offset,
			       ReductionOpID redopid,
			       const void *data, size_t datalen, int payload_mode);
    };
    
    class RemoteWriteFence : public Operation::AsyncWorkItem {
    public:
      RemoteWriteFence(Operation *op);

      virtual void request_cancellation(void);

      virtual void print(std::ostream& os) const;
    };

    struct RemoteWriteFenceMessage {
      struct RequestArgs {
	Memory mem;
	unsigned sender;
	unsigned sequence_id;
	unsigned num_writes;
	RemoteWriteFence *fence;
      };
       
      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<REMOTE_WRITE_FENCE_MSGID,
				        RequestArgs,
				        handle_request> Message;

      static void send_request(NodeID target, Memory memory,
			       unsigned sequence_id, unsigned num_writes,
			       RemoteWriteFence *fence);
    };
    
    struct RemoteWriteFenceAckMessage {
      struct RequestArgs {
	RemoteWriteFence *fence;
        // TODO: success/failure
      };
       
      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<REMOTE_WRITE_FENCE_ACK_MSGID,
				        RequestArgs,
				        handle_request> Message;

      static void send_request(NodeID target,
			       RemoteWriteFence *fence);
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
