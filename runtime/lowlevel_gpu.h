/* Copyright 2013 Stanford University
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

#ifndef LOWLEVEL_GPU_H
#define LOWLEVEL_GPU_H

#include "lowlevel_impl.h"
#include <cuda_runtime.h>

#define CHECK_CUDART(cmd) do { \
  cudaError_t ret = (cmd); \
  if(ret != cudaSuccess) { \
    fprintf(stderr, "CUDART: %s = %d (%s)\n", #cmd, ret, cudaGetErrorString(ret)); \
    assert(0); \
    exit(1); \
  } \
} while(0)

GASNETT_THREADKEY_DECLARE(gpu_thread);

namespace LegionRuntime {
  namespace LowLevel {
    class GPUProcessor : public Processor::Impl {
    public:
      GPUProcessor(Processor _me, int _gpu_index, Processor _util,
		   size_t _zcmem_size, size_t _fbmem_size, size_t _stack_size);

      ~GPUProcessor(void);

      void start_worker_thread(void);

      void *get_zcmem_cpu_base(void);
      void *get_fbmem_gpu_base(void);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstanceUntyped> instances_needed,
			      Event start_event, Event finish_event,
                              int priority);

      virtual void enable_idle_task(void);

      virtual void disable_idle_task(void);

      void copy_to_fb(off_t dst_offset, const void *src, size_t bytes,
		      Event start_event, Event finish_event);

      void copy_from_fb(void *dst, off_t src_offset, size_t bytes,
			Event start_event, Event finish_event);

      void copy_within_fb(off_t dst_offset, off_t src_offset,
			  size_t bytes,
			  Event start_event, Event finish_event);

      //void copy_to_fb_generic(off_t dst_offset, 
      //			      Memory::Impl *src_mem, off_t src_offset,
      //			      size_t bytes,
      //			      Event start_event, Event finish_event);

      //void copy_from_fb_generic(Memory::Impl *dst_mem, off_t dst_offset, 
      //				off_t src_offset, size_t bytes,
      //				Event start_event, Event finish_event);

      void copy_to_fb(off_t dst_offset, const void *src,
		      const ElementMask *mask, size_t elmt_size,
		      Event start_event, Event finish_event);

      void copy_from_fb(void *dst, off_t src_offset,
			const ElementMask *mask, size_t elmt_size,
			Event start_event, Event finish_event);

      void copy_within_fb(off_t dst_offset, off_t src_offset,
			  const ElementMask *mask, size_t elmt_size,
			  Event start_event, Event finish_event);

      //void copy_to_fb_generic(off_t dst_offset, 
      //			      Memory::Impl *src_mem, off_t src_offset,
      //			      const ElementMask *mask,
      //			      size_t elmt_size,
      //			      Event start_event, Event finish_event);

      //void copy_from_fb_generic(Memory::Impl *dst_mem, off_t dst_offset, 
      //				off_t src_offset,
      //				const ElementMask *mask, size_t elmt_size,
      //				Event start_event, Event finish_event);

    public:
      // Helper method for getting a thread's processor value
      static Processor get_processor(void);
    public:
      class Internal;

      GPUProcessor::Internal *internal;
    };

    class GPUFBMemory : public Memory::Impl {
    public:
      GPUFBMemory(Memory _me, GPUProcessor *_gpu);

      virtual ~GPUFBMemory(void);

      virtual RegionInstance create_instance(IndexSpace is,
					     const int *linearization_bits,
					     size_t bytes_needed,
					     size_t block_size,
					     size_t element_size,
					     const std::vector<size_t>& field_sizes,
					     ReductionOpID redopid,
					     off_t list_size,
					     RegionInstance parent_inst)
      {
	return create_instance_local(is, linearization_bits, bytes_needed,
				     block_size, element_size, field_sizes, redopid,
				     list_size, parent_inst);
      }

      virtual void destroy_instance(RegionInstance i, 
				    bool local_destroy)
      {
	destroy_instance_local(i, local_destroy);
      }

      virtual off_t alloc_bytes(size_t size)
      {
	return alloc_bytes_local(size);
      }

      virtual void free_bytes(off_t offset, size_t size)
      {
	free_bytes_local(offset, size);
      }

      virtual void get_bytes(off_t offset, void *dst, size_t size)
      {
	assert(0);
	//memcpy(dst, base+offset, size);
      }

      virtual void put_bytes(off_t offset, const void *src, size_t size)
      {
	assert(0);
	//memcpy(base+offset, src, size);
      }

      virtual void *get_direct_ptr(off_t offset, size_t size)
      {
	return (base + offset);
      }

      virtual int get_home_node(off_t offset, size_t size)
      {
	return -1;
      }

    public:
      GPUProcessor *gpu;
      char *base;
    };

    class GPUZCMemory : public Memory::Impl {
    public:
      GPUZCMemory(Memory _me, GPUProcessor *_gpu);

      virtual ~GPUZCMemory(void);

      virtual RegionInstance create_instance(IndexSpace is,
					     const int *linearization_bits,
					     size_t bytes_needed,
					     size_t block_size,
					     size_t element_size,
					     const std::vector<size_t>& field_sizes,
					     ReductionOpID redopid,
					     off_t list_size,
					     RegionInstance parent_inst)
      {
	return create_instance_local(is, linearization_bits, bytes_needed,
				     block_size, element_size, field_sizes, redopid,
				     list_size, parent_inst);
      }

      virtual void destroy_instance(RegionInstance i, 
				    bool local_destroy)
      {
	destroy_instance_local(i, local_destroy);
      }

      virtual off_t alloc_bytes(size_t size)
      {
	return alloc_bytes_local(size);
      }

      virtual void free_bytes(off_t offset, size_t size)
      {
	free_bytes_local(offset, size);
      }

      virtual void get_bytes(off_t offset, void *dst, size_t size)
      {
	memcpy(dst, cpu_base+offset, size);
      }

      virtual void put_bytes(off_t offset, const void *src, size_t size)
      {
	memcpy(cpu_base+offset, src, size);
      }

      virtual void *get_direct_ptr(off_t offset, size_t size)
      {
	return (cpu_base + offset);
      }

      virtual int get_home_node(off_t offset, size_t size)
      {
	return ID(me).node();
      }

    public:
      GPUProcessor *gpu;
      char *cpu_base;
    };

  }; // namespace LowLevel
}; // namespace LegionRuntime

#endif
