/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#ifndef REALM_CUDA_H
#define REALM_CUDA_H

#include "realm/realm_config.h"
#include "realm/module.h"
#include "realm/processor.h"
#include "realm/network.h"
#include "realm/atomics.h"
#include "realm/module_config.h"

// realm/cuda_module.h is designed to be include-able even when the system
//  doesn't actually have CUDA installed, so we need to declare types that
//  are compatible with the CUDA driver and runtime APIs - we can't "extern"
//  a typedef (e.g. cudaStream_t) but we can forward declare the underlying
//  struct that those types are pointers to
struct CUstream_st; // cudaStream_t == CUstream == CUstream_st *

namespace Realm {

  namespace NetworkSegmentInfo {
    // CUDA device memory - extra is a uintptr_t'd pointer to the GPU
    //  object
    static const MemoryType CudaDeviceMem = 2;

    // CUDA managed memory - extra is a uintptr_t'd pointer to _one of_
    //  the GPU objects
    static const MemoryType CudaManagedMem = 4;
  };

  
  namespace Cuda {

    // a running task on a CUDA processor is assigned a stream by Realm, and
    //  any work placed on this stream is automatically captured by the
    //  completion event for the task
    // when using the CUDA runtime hijack, Realm will force work launched via
    //  the runtime API to use the task's stream, but without hijack, or for
    //  code that uses the CUDA driver API, the task must explicitly request
    //  the stream that is associated with the task and place work on it to
    //  avoid more expensive forms of completion detection for the task
    // NOTE: this function will return a null pointer if called outside of a
    //  task running on a CUDA processor
    REALM_PUBLIC_API CUstream_st *get_task_cuda_stream();

    // when Realm is not using the CUDA runtime hijack to force work onto the
    //  task's stream, it conservatively uses a full context synchronization to
    //  make sure all device work launched by the task is captured by the task
    //  completion event - if a task uses `get_task_cuda_stream` and places all
    //  work on that stream, this API can be used to tell Realm on a per-task
    //  basis that full context synchronization is not required
    REALM_PUBLIC_API void set_task_ctxsync_required(bool is_required);

    // rather than using the APIs above, CUDA processors also support task
    //  implementations that are natively stream aware - if a task function uses
    //  the `Cuda::StreamAwareTaskFuncPtr` prototype below (instead of the normal
    //  `Processor::TaskFuncPtr`), the following differences apply:
    // a) it need not call `get_task_cuda_stream` because it gets the same value
    //   directly as an argument
    // b) by default, a context synchronization will NOT be performed as part of
    //   task completion detection (this can still be overridden with a call to
    //   `set_task_ctxsync_required(true)` if a task puts work outside the
    //   specified stream for some reason
    // c) if a stream-aware task has preconditions that involve device work, that
    //   work will be tied into the task's stream, but the task body may start
    //   executing BEFORE that work is complete (i.e. for correctness, all work
    //   launched by the task must be properly ordered (using the CUDA APIs)
    //   after anything already in the stream assigned to the task
    typedef void (*StreamAwareTaskFuncPtr)(const void *args, size_t arglen,
					   const void *user_data, size_t user_data_len,
					   Processor proc, CUstream_st *stream);

    class GPU;
    class GPUWorker;
    struct GPUInfo;
    class GPUZCMemory;
    class GPUReplHeapListener;

    class CudaModuleConfig : public ModuleConfig {
      friend class CudaModule;
    protected:
      CudaModuleConfig(void);

      bool discover_resource(void);
    public:  
      virtual bool get_resource(const std::string name, int &value) const;
      virtual bool get_resource(const std::string name, size_t &value) const;
      virtual void configure_from_cmdline(std::vector<std::string>& cmdline);

    public:
      // configurations
      size_t cfg_zc_mem_size = 64 << 20, cfg_zc_ib_size = 256 << 20;
      size_t cfg_fb_mem_size = 256 << 20, cfg_fb_ib_size = 128 << 20;
      size_t cfg_uvm_mem_size = 0;
      bool cfg_use_dynamic_fb = true;
      size_t cfg_dynfb_max_size = ~size_t(0);
      int cfg_num_gpus = 0;
      std::string cfg_gpu_idxs;
      unsigned cfg_task_streams = 12, cfg_d2d_streams = 4;
      bool cfg_use_worker_threads = false, cfg_use_shared_worker = true, cfg_pin_sysmem = true;
      bool cfg_fences_use_callbacks = false;
      bool cfg_suppress_hijack_warning = false;
      unsigned cfg_skip_gpu_count = 0;
      bool cfg_skip_busy_gpus = false;
      size_t cfg_min_avail_mem = 0;
      int cfg_task_legacy_sync = 0; // 0 = no, 1 = yes
      int cfg_task_context_sync = -1; // 0 = no, 1 = yes, -1 = default (based on hijack)
      int cfg_max_ctxsync_threads = 4;
      bool cfg_lmem_resize_to_max = false;
      bool cfg_multithread_dma = false;
      size_t cfg_hostreg_limit = 1 << 30;
      int cfg_d2d_stream_priority = -1;
      bool cfg_use_cuda_ipc = true;

      // resources
      bool resource_discovered = false;
      int res_num_gpus = 0;
      std::vector<size_t> res_fbmem_sizes;
    };


    // our interface to the rest of the runtime
    class REALM_PUBLIC_API CudaModule : public Module {
    protected:
      CudaModule(RuntimeImpl *_runtime);
      
    public:
      virtual ~CudaModule(void);

      static ModuleConfig *create_module_config(RuntimeImpl *runtime);

      static Module *create_module(RuntimeImpl *runtime);

      // do any general initialization - this is called after all configuration is
      //  complete
      virtual void initialize(RuntimeImpl *runtime);

      // create any memories provided by this module (default == do nothing)
      //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
      virtual void create_memories(RuntimeImpl *runtime);

      // create any processors provided by the module (default == do nothing)
      //  (each new ProcessorImpl should use a Processor from
      //   RuntimeImpl::next_local_processor_id)
      virtual void create_processors(RuntimeImpl *runtime);

      // create any DMA channels provided by the module (default == do nothing)
      virtual void create_dma_channels(RuntimeImpl *runtime);

      // create any code translators provided by the module (default == do nothing)
      virtual void create_code_translators(RuntimeImpl *runtime);

      // if a module has to do cleanup that involves sending messages to other
      //  nodes, this must be done in the pre-detach cleanup
      virtual void pre_detach_cleanup(void);

      // clean up any common resources created by the module - this will be called
      //  after all memories/processors/etc. have been shut down and destroyed
      virtual void cleanup(void);

      // free functions above are normally used, but these can be used directly
      //  if you already have a pointer to the CudaModule
      CUstream_st *get_task_cuda_stream();
      void set_task_ctxsync_required(bool is_required);

    public:
      CudaModuleConfig *config;
      RuntimeImpl *runtime;

      // "global" variables live here too
      GPUWorker *shared_worker;
      std::map<GPU *, GPUWorker *> dedicated_workers;
      std::vector<GPUInfo *> gpu_info;
      std::vector<GPU *> gpus;
      void *zcmem_cpu_base, *zcib_cpu_base;
      GPUZCMemory *zcmem;
      void *uvm_base; // guaranteed to be same for CPU and GPU
      GPUZCMemory *uvmmem;
      std::vector<void *> registered_host_ptrs;
      GPUReplHeapListener *rh_listener;

      Mutex cudaipc_mutex;
      Mutex::CondVar cudaipc_condvar;
      atomic<int> cudaipc_responses_needed;
      atomic<int> cudaipc_releases_needed;
      atomic<int> cudaipc_exports_remaining;
    };

  }; // namespace Cuda

}; // namespace Realm 

#include "realm/cuda/cuda_module.inl"

#endif
