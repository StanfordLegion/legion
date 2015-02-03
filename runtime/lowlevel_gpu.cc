/* Copyright 2015 Stanford University
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

#include "lowlevel_gpu.h"

#include <stdio.h>

namespace LegionRuntime {
  namespace LowLevel {
    GASNETT_THREADKEY_DEFINE(gpu_thread);

    extern Logger::Category log_gpu;
#ifdef EVENT_GRAPH_TRACE
    extern Logger::Category log_event_graph;
#endif

    enum GPUMemcpyKind {
      GPU_MEMCPY_HOST_TO_DEVICE,
      GPU_MEMCPY_DEVICE_TO_HOST,
      GPU_MEMCPY_DEVICE_TO_DEVICE,
      GPU_MEMCPY_PEER_TO_PEER,
    };

    class GPUJob : public EventWaiter {
    public:
      GPUJob(GPUProcessor *_gpu, Event _finish_event)
	: gpu(_gpu), finish_event(_finish_event) { }

      virtual ~GPUJob(void) {}

      virtual bool event_triggered(void) = 0;

      virtual void print_info(FILE *f) = 0;

      virtual void run_or_wait(Event start_event) = 0;

      void pre_execute(void)
      {
        CHECK_CU( cuEventCreate(&complete_event, 
                                CU_EVENT_DISABLE_TIMING) );
      }

      virtual void execute(void) = 0;

      bool is_finished(void)
      {
        CUresult result = cuEventQuery(complete_event);
        if (result == CUDA_SUCCESS)
          return true;
        else if (result == CUDA_ERROR_NOT_READY)
          return false;
        else
        {
          CHECK_CU( result );
        }
        return false;
      }

      virtual void finish_job(void)
      {
        // If we have a finish event then trigger it
        if (finish_event.exists())
	  Runtime::get_runtime()->get_genevent_impl(finish_event)->
            trigger(finish_event.gen, gasnet_mynode());
        // Destroy our event
        CHECK_CU( cuEventDestroy(complete_event) );
      }

    public:
      GPUProcessor *gpu;
      Event finish_event;
      CUevent complete_event;
    };

    class GPUTask : public GPUJob {
    public:
      GPUTask(GPUProcessor *_gpu, Event _finish_event,
	      Processor::TaskFuncID _func_id,
	      const void *_args, size_t _arglen,
              int priority);

      virtual ~GPUTask(void);

      virtual bool event_triggered(void);

      virtual void print_info(FILE *f);

      virtual void run_or_wait(Event start_event);

      virtual void execute(void);

      virtual void finish_job(void);

      void set_local_stream(CUstream s) { local_stream = s; }

      void record_modules(const std::set<void**> &m) { modules = m; }

      Processor::TaskFuncID func_id;
      void *args;
      size_t arglen;
      int priority;
      CUstream local_stream;
      std::set<void**> modules;
    };

    class GPUProcessor::Internal {
    public:
      GPUProcessor *const gpu;
      const int gpu_index;
      const size_t zcmem_size, fbmem_size;
      const size_t zcmem_reserve, fbmem_reserve;
      const bool gpu_dma_thread;
      void *zcmem_cpu_base;
      void *zcmem_gpu_base;
      void *fbmem_gpu_base;

      bool initialized;
      bool worker_enabled;
      bool shutdown_requested;
      bool idle_task_enabled;
      int current_stream;
      pthread_t gpu_thread;
      gasnet_hsl_t mutex;
      gasnett_cond_t parent_condvar, worker_condvar;
      
      std::list<GPUTask*> tasks;
      std::deque<GPUJob*> copies;

      // Our CUDA context that we will create
      CUdevice  proc_dev;
      CUcontext proc_ctx;

      // Streams for different copy types
      CUstream host_to_device_stream;
      CUstream device_to_host_stream;
      CUstream device_to_device_stream;
      CUstream peer_to_peer_stream;
      std::vector<CUstream> task_streams;

      // List of pending copies on each stream
      std::deque<GPUJob*> host_device_copies;
      std::deque<GPUJob*> device_host_copies;
      std::deque<GPUJob*> device_device_copies;
      std::deque<GPUJob*> peer_to_peer_copies;
      std::vector<std::deque<GPUJob*> > pending_tasks;

      Internal(GPUProcessor *_gpu, int _gpu_index, 
               int num_local_gpus,
               size_t _zcmem_size, size_t _fbmem_size,
               size_t _zcmem_res, size_t _fbmem_res,
               bool enabled, bool gpu_dma, int streams)
	: gpu(_gpu), gpu_index(_gpu_index),
          zcmem_size(_zcmem_size), fbmem_size(_fbmem_size),
          zcmem_reserve(_zcmem_res), fbmem_reserve(_fbmem_res),
          gpu_dma_thread(gpu_dma),
          initialized(false), worker_enabled(false), 
          shutdown_requested(false),
	  idle_task_enabled(enabled), current_stream(0)
      {
	gasnet_hsl_init(&mutex);
	gasnett_cond_init(&parent_condvar);
	gasnett_cond_init(&worker_condvar);
        assert(streams > 0);
        task_streams.resize(streams);
        pending_tasks.resize(streams);
        // Make our context and then immediately pop it off
        CHECK_CU( cuDeviceGet(&proc_dev, gpu_index) );

        CHECK_CU( cuCtxCreate(&proc_ctx, CU_CTX_MAP_HOST |
                              CU_CTX_SCHED_BLOCKING_SYNC, proc_dev) );
        
        CHECK_CU( cuCtxPopCurrent(&proc_ctx) );
      }

      Processor get_processor(void) const
      {
        return gpu->me;
      }

      CUstream get_current_task_stream(void)
      {
        return task_streams[current_stream];
      }

      void thread_main(void)
      {
	gasnett_threadkey_set(gpu_thread, gpu);

        // Push our context onto the stack
        CHECK_CU( cuCtxPushCurrent(proc_ctx) );

        // load any modules, functions, and variables that we deferred
        const std::map<void*,void**> &deferred_modules = get_deferred_modules();
        for (std::map<void*,void**>::const_iterator it = deferred_modules.begin();
              it != deferred_modules.end(); it++)
        {
          ModuleInfo &info = modules[it->second];
          FatBin *fatbin_args = (FatBin*)it->first;
          assert((fatbin_args->data != NULL) ||
                 (fatbin_args->filename_or_fatbins != NULL));
          if (fatbin_args->data != NULL)
          {
            CHECK_CU( cuModuleLoadData(&(info.module), fatbin_args->data) );
          } else {
            CHECK_CU( cuModuleLoad(&(info.module), 
                  (const char*)fatbin_args->filename_or_fatbins) );
          }
        }
        const std::deque<DeferredFunction> &deferred_functions = get_deferred_functions();
        for (std::deque<DeferredFunction>::const_iterator it = deferred_functions.begin();
              it != deferred_functions.end(); it++)
        {
          register_function(it->handle, it->host_fun, it->device_fun);
        }
        const std::deque<DeferredVariable> &deferred_variables = get_deferred_variables();
        for (std::deque<DeferredVariable>::const_iterator it = deferred_variables.begin();
              it != deferred_variables.end(); it++)
        {
          register_var(it->handle, it->host_var, it->device_name, 
                       it->external, it->size, it->constant, it->global);
        }
        
        // allocate zero-copy memory
        CHECK_CU( cuMemHostAlloc(&zcmem_cpu_base,
                                 zcmem_size + zcmem_reserve,
                                 (CU_MEMHOSTALLOC_PORTABLE |
                                  CU_MEMHOSTALLOC_DEVICEMAP)) );
        CHECK_CU( cuMemHostGetDevicePointer((CUdeviceptr*)&zcmem_gpu_base,
                                            zcmem_cpu_base, 0) );

        // allocate frame buffer memory
        CHECK_CU( cuMemAlloc((CUdeviceptr*)&fbmem_gpu_base, fbmem_size + fbmem_reserve) );

        // initialize the streams for copy operations
        CHECK_CU( cuStreamCreate(&host_to_device_stream,
                                 CU_STREAM_NON_BLOCKING) );
        CHECK_CU( cuStreamCreate(&device_to_host_stream,
                                 CU_STREAM_NON_BLOCKING) );
        CHECK_CU( cuStreamCreate(&device_to_device_stream,
                                 CU_STREAM_NON_BLOCKING) );
        CHECK_CU( cuStreamCreate(&peer_to_peer_stream,
                                 CU_STREAM_NON_BLOCKING) );
        for (unsigned idx = 0; idx < task_streams.size(); idx++)
        {
          CHECK_CU( cuStreamCreate(&task_streams[idx],
                                   CU_STREAM_NON_BLOCKING) );
        }

	log_gpu(LEVEL_INFO, "gpu initialized: zcmem=%p/%p fbmem=%p",
		zcmem_cpu_base, zcmem_gpu_base, fbmem_gpu_base);

	// set the initialized flag and maybe wake up parent
	{
	  AutoHSLLock a(mutex);
	  initialized = true;
	  gasnett_cond_signal(&parent_condvar);

	  // wait until we've been told to proceed
	  while(!worker_enabled) {
	    log_gpu.info("waiting for enable signal");
	    gasnett_cond_wait(&worker_condvar, &mutex.lock);
	  }
	}

	while(!shutdown_requested) {
	  // get all the copies and a job off the job queue - sleep if nothing there
	  GPUTask *task = NULL;
          std::vector<GPUJob*> ready_copies;
	  {
	    AutoHSLLock a(mutex);
	    while(tasks.empty() && copies.empty() && !shutdown_requested) {
	      // see if there's an idle task we should run
	      Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_IDLE);
              // As long as we have CUDA events on streams
              // we need to keep polling so don't ever go to
              // sleep while we have pending operations either
              // in the form of copies or kernel launches
              bool has_pending_operations = false;
              // If we have a DMA thread we don't need to check this
              if (!gpu_dma_thread && (!host_device_copies.empty() || 
                  !device_host_copies.empty() || !device_device_copies.empty()))
                has_pending_operations = true;
              // If we still have no pending operations, check our task
              // We always need to do this
              if (!has_pending_operations)
              {
                for (unsigned idx = 0; idx < pending_tasks.size(); idx++)
                {
                  if (!pending_tasks[idx].empty())
                  {
                    has_pending_operations = true;
                    break;
                  }
                }
              }
              //
	      if(idle_task_enabled && (it != task_id_table.end())) {
		gasnet_hsl_unlock(&mutex);
		log_gpu.spew("running scheduler thread");
		(it->second)(0, 0, gpu->me);
		log_gpu.spew("returned from scheduler thread");
		gasnet_hsl_lock(&mutex);
                // Can't go to sleep if we have tasks and/or copies to poll
	      } else if (!has_pending_operations) {
		log_gpu.debug("job queue empty - sleeping\n");
		gasnett_cond_wait(&worker_condvar, &mutex.lock);
		if(shutdown_requested) {
		  log_gpu.debug("awoke due to shutdown request...\n");
		  break;
		}
		log_gpu.debug("awake again...\n");
	      }
              else
              {
                // We have pending operations so break out
                break;
              }
	    }
	    if(shutdown_requested) break;
            if (!gpu_dma_thread)
            {
              ready_copies.insert(ready_copies.end(),copies.begin(),copies.end());
              copies.clear();
            }
            if (!tasks.empty())
            {
              task = tasks.front();
              tasks.pop_front();
            }
	  }

          // Check to see if any of our tasks are done
          check_for_complete_tasks();

          if (!gpu_dma_thread)
          {
            // Check to see if any of our copies have completed
            check_for_complete_copies();  

            // Launch all the copies first since we know that they
            // are going to be asynchronous on streams that won't block tasks
            // These calls well enqueue the copies on the right queue.
            for (std::vector<GPUJob*>::const_iterator it = ready_copies.begin();
                  it != ready_copies.end(); it++)
            {
              (*it)->pre_execute();
              (*it)->execute();
            }
          }

	  // charge all the time from the start of the execute until the end
	  //  of the device synchronize to the app - anything that wasn't
	  //  really the app will be claimed anyway
          if (task != NULL)
	  {
	    DetailedTimer::ScopedPush sp(TIME_KERNEL);
            task->set_local_stream(task_streams[current_stream]);
	    //printf("executing job %p\n", job);
            task->pre_execute();
            assert(task_modules.empty());
            assert(kernel_arguments.empty());
            argument_offset = 0;
	    task->execute();
            // When we are done, tell the task about all the modules
            // it needs to unload
            task->record_modules(task_modules);
            task_modules.clear();
            // Check to see if we are done or not
            if (task->is_finished())
            {
              task->finish_job();
              delete task;
            }
            else
            {
              // Add it to the set of tasks to query
              pending_tasks[current_stream].push_back(task);
            }
            // Update the current stream
            current_stream++;
            if ((size_t)current_stream >= task_streams.size())
              current_stream = 0;
	  }
	}

	log_gpu.info("shutting down");
        Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_SHUTDOWN);
        if(it != task_id_table.end()) {
          log_gpu(LEVEL_INFO, "calling processor shutdown task: proc=" IDFMT "", gpu->me.id);

          (it->second)(0, 0, gpu->me);

          log_gpu(LEVEL_INFO, "finished processor shutdown task: proc=" IDFMT "", gpu->me.id);
        }
	gpu->finished();
      }

      static void *thread_main_wrapper(void *data)
      {
	GPUProcessor::Internal *obj = (GPUProcessor::Internal *)data;
	obj->thread_main();
	return 0;
      }

      void create_gpu_thread(size_t stack_size)
      {
	pthread_attr_t attr;
	CHECK_PTHREAD( pthread_attr_init(&attr) );
        CHECK_PTHREAD( pthread_attr_setstacksize(&attr, stack_size) );
	if(proc_assignment)
	  proc_assignment->bind_thread(-1, &attr, "GPU worker");
	CHECK_PTHREAD( pthread_create(&gpu_thread, &attr, 
				      thread_main_wrapper,
				      (void *)this) );
	CHECK_PTHREAD( pthread_attr_destroy(&attr) );
#ifdef DEADLOCK_TRACE
        Runtime::get_runtime()->add_thread(&gpu_thread);
#endif

	// now wait until worker thread is ready
	{
	  AutoHSLLock a(mutex);
	  while(!initialized)
	    gasnett_cond_wait(&parent_condvar, &mutex.lock);
	}
      }

      void tasks_available(int priority)
      {
	AutoHSLLock al(mutex);
	gasnett_cond_signal(&worker_condvar);
      }

      void enqueue_task(GPUTask *job)
      {
        AutoHSLLock a(mutex);

	bool was_empty = tasks.empty() && copies.empty();
        // Add it based on its priority
        // Common case
        if (tasks.empty() || (tasks.back()->priority >= job->priority))
          tasks.push_back(job);
        else
        {
          // Uncommon case: go through the list until
          // we find someone who has a priority lower than ours.
          bool inserted = false;
          for (std::list<GPUTask*>::iterator it = tasks.begin();
                it != tasks.end(); it++)
          {
            if ((*it)->priority < job->priority)
            {
              tasks.insert(it, job);
              inserted = true;
              break;
            }
          }
          // Technically we shouldn't need this
          if (!inserted)
            tasks.push_back(job);
        }

	if(was_empty)
	  gasnett_cond_signal(&worker_condvar);
      }

      void enqueue_copy(GPUJob *job)
      {
        AutoHSLLock a(mutex);

        bool was_empty = tasks.empty() && copies.empty();
        copies.push_back(job);

        if (was_empty)
          gasnett_cond_signal(&worker_condvar);
      }

      void check_for_complete_tasks(void)
      {
        for (unsigned idx = 0; idx < pending_tasks.size(); idx++)
        {
          std::deque<GPUJob*> &pending_stream = pending_tasks[idx];
          while (!pending_stream.empty())
          {
            GPUJob *next = pending_stream.front();
            if (next->is_finished())
            {
              next->finish_job();
              delete next;
              pending_stream.pop_front();
            }
            else
              break; // If the first one wasn't done, the others won't be either
          }
        }
      }

      void add_host_device_copy(GPUJob *copy)
      {
        host_device_copies.push_back(copy);
      }
      void add_device_host_copy(GPUJob *copy)
      {
        device_host_copies.push_back(copy);
      }
      void add_device_device_copy(GPUJob *copy)
      {
        device_device_copies.push_back(copy);
      }
      void add_peer_to_peer_copy(GPUJob *copy)
      {
        peer_to_peer_copies.push_back(copy);
      }
      void check_for_complete_copies(void)
      {
        // Check to see if we have any pending copies to query
        while (!host_device_copies.empty())
        {
          GPUJob *next = host_device_copies.front();
          if (next->is_finished())
          {
            next->finish_job();
            delete next;
            host_device_copies.pop_front();
          }
          else
            break; // If the first one wasn't done, the others won't be either
        }
        while (!device_host_copies.empty())
        {
          GPUJob *next = device_host_copies.front();
          if (next->is_finished())
          {
            next->finish_job();
            delete next;
            device_host_copies.pop_front();
          }
          else
            break;
        }
        while (!device_device_copies.empty())
        {
          GPUJob *next = device_device_copies.front();
          if (next->is_finished())
          {
            next->finish_job();
            delete next;
            device_device_copies.pop_front();
          }
          else
            break;
        }
        while (!peer_to_peer_copies.empty())
        {
          GPUJob *next = peer_to_peer_copies.front();
          if (next->is_finished())
          {
            next->finish_job();
            delete next;
            peer_to_peer_copies.pop_front();
          }
          else
            break;
        }
      }
      void register_host_memory(void *base, size_t size)
      {
        if (!shutdown_requested)
        {
          CHECK_CU( cuCtxPushCurrent(proc_ctx) );
          CHECK_CU( cuMemHostRegister(base, size, CU_MEMHOSTREGISTER_PORTABLE) ); 
          CHECK_CU( cuCtxPopCurrent(&proc_ctx) );
        }
      }

      void enable_peer_access(GPUProcessor::Internal *neighbor)
      {
        neighbor->handle_peer_access(proc_ctx);
      }

      void handle_peer_access(CUcontext peer_ctx)
      {
        CHECK_CU( cuCtxPushCurrent(proc_ctx) );
        CHECK_CU( cuCtxEnablePeerAccess(peer_ctx, 0) );
        CHECK_CU( cuCtxPopCurrent(&proc_ctx) );
      }

      void handle_copies(void)
      {
        if (!shutdown_requested)
        {
          // Push our context onto the stack
          //CHECK_CU( cuCtxPushCurrent(proc_ctx) );
          // Don't check for errors here because CUDA is dumb
          cuCtxPushCurrent(proc_ctx);
          // First see if any of our copies are ready
          check_for_complete_copies();
          std::vector<GPUJob*> ready_copies;
          // Get any copies that are ready to be performed
          {
            AutoHSLLock a(mutex);
            ready_copies.insert(ready_copies.end(),copies.begin(),copies.end());
            copies.clear();
          }
          // Issue our copies
          for (std::vector<GPUJob*>::const_iterator it = ready_copies.begin();
                it != ready_copies.end(); it++)
          {
            (*it)->pre_execute();
            (*it)->execute();
          }
          // Now pop our context back off the stack
          //CHECK_CU( cuCtxPopCurrent(&proc_ctx) );
          // Don't check for errors here because CUDA is dumb
          cuCtxPopCurrent(&proc_ctx);
        }
      }
    public:
      // Our helper cuda calls
      void** register_fat_binary(void *fat_bin);
      void unregister_fat_binary(void **fat_bin);
      void register_var(void **fat_bin, char *host_var, const char *device_name,
                        bool ext, int size, bool constant, bool global);
      void register_function(void **fat_bin ,const char *host_fun,
                             const char *device_fun);
      char init_module(void **fat_bin);
    public:
      // Our cuda calls
      cudaError_t stream_synchronize(void);  
      cudaError_t configure_call(dim3 gird_dim, dim3 block_dim, size_t shared_mem);
      cudaError_t setup_argument(const void *arg, size_t size, size_t offset);
      cudaError_t launch(const void *func);
      cudaError_t gpu_memcpy(void *dst, const void *src, size_t size, bool sync);
      cudaError_t gpu_memcpy_to_symbol(void *dst, const void *src, size_t size,
                                       size_t offset, cudaMemcpyKind kind, bool sync);
      cudaError_t gpu_memcpy_from_symbol(void *dst, const void *src, size_t size,
                                         size_t offset, cudaMemcpyKind kind, bool sync);
    private:
      struct VarInfo {
      public:
        const char *name;
        CUdeviceptr ptr;
        size_t size;
      };
      struct ModuleInfo {
        CUmodule module;
        std::set<const void*> host_aliases;
        std::set<const void*> var_aliases;
      };
      struct LaunchConfig {
      public:
        dim3 grid;
        dim3 block;
        size_t shared;
      };
    private:
      // Support for our internal cuda runtime
      struct FatBin {
        int magic; // Hehe cuda magic (who knows what this does)
        int version;
        const unsigned long long *data;
        void *filename_or_fatbins;
      };
      std::map<void** /*fatbin*/,ModuleInfo> modules;
      std::map<const void*,CUfunction> device_functions;
      std::map<const void*,VarInfo> device_variables;
      std::deque<LaunchConfig> launch_configs;
      size_t argument_offset;
      std::vector<void*> kernel_arguments;
      // Modules allocated just during this task's lifetime
      std::set<void**> task_modules;
    public:
      // Support for deferring loading of modules and functions
      // until after we have initialized the runtime
      struct DeferredFunction {
        void **handle;
        const char *host_fun;
        const char *device_fun;
      };
      struct DeferredVariable {
        void **handle;
        char *host_var;
        const char *device_name;
        bool external;
        int size;
        bool constant;
        bool global;
      };
      static std::map<void*,void**>& get_deferred_modules(void);
      static std::deque<DeferredFunction>& get_deferred_functions(void);
      static std::deque<DeferredVariable>& get_deferred_variables(void);
      static void** defer_module_load(void *fat_bin);
      static void defer_function_load(void **fat_bin, const char *host_fun,
                                      const char *device_fun);
      static void defer_variable_load(void **fat_bin, char *host_var,
                                      const char *device_name,
                                      bool ext, int size,
                                      bool constant, bool global);
    };

    GPUTask::GPUTask(GPUProcessor *_gpu, Event _finish_event,
		     Processor::TaskFuncID _func_id,
		     const void *_args, size_t _arglen,
                     int _priority)
      : GPUJob(_gpu, _finish_event), func_id(_func_id), 
               arglen(_arglen), priority(_priority)
    {
      if(arglen) {
	args = malloc(arglen);
	memcpy(args, _args, arglen);
      } else {
	args = 0;
      }
    }

    GPUTask::~GPUTask(void)
    {
      if(args) free(args);
    }

    bool GPUTask::event_triggered(void)
    {
      log_gpu.info("gpu job %p now runnable", this);
      gpu->internal->enqueue_task(this);

      // don't delete
      return false;
    }

    void GPUTask::print_info(FILE *f)
    {
      fprintf(f,"GPU Task: %p after=" IDFMT "/%d\n",
          this, finish_event.id, finish_event.gen);
    }

    void GPUTask::run_or_wait(Event start_event)
    {
      if(start_event.has_triggered()) {
	log_gpu.info("job %p can start right away!?", this);
	gpu->internal->enqueue_task(this);
      } else {
	log_gpu.info("job %p waiting for " IDFMT "/%d", this, start_event.id, start_event.gen);
	start_event.impl()->add_waiter(start_event.gen, this);
      }
    }

    void GPUTask::execute(void)
    {
      Processor::TaskFuncPtr fptr = task_id_table[func_id];
      //char argstr[100];
      //argstr[0] = 0;
      //for(size_t i = 0; (i < arglen) && (i < 40); i++)
      //	sprintf(argstr+2*i, "%02x", ((unsigned char *)args)[i]);
      //if(arglen > 40) strcpy(argstr+80, "...");
      //log_gpu(LEVEL_DEBUG, "task start: %d (%p) (%s)", func_id, fptr, argstr);

      // make sure CUDA driver's state is ok before we start
      //assert(cudaGetLastError() == cudaSuccess);
#ifdef EVENT_GRAPH_TRACE
      assert(finish_event.exists());
      gpu->start_enclosing(finish_event); 
      unsigned long long start = TimeStamp::get_current_time_in_micros();
#endif
      (*fptr)(args, arglen, gpu->me);
#ifdef EVENT_GRAPH_TRACE
      unsigned long long stop = TimeStamp::get_current_time_in_micros();
      gpu->finish_enclosing();
      log_event_graph.debug("Task Time: (" IDFMT ",%d) %lld",
                            finish_event.id, finish_event.gen,
                            (stop - start));
#endif
      // Now log our CUDA event on the stream
      CHECK_CU( cuEventRecord(complete_event, local_stream) );
      // A useful debugging macro
#ifdef FORCE_GPU_STREAM_SYNCHRONIZE
      CHECK_CU( cuStreamSynchronize(local_stream) );
#endif
      // check for any uncaught driver errors after the task finishes
#if 0
      {
	cudaError_t result = cudaGetLastError();
	if (result != cudaSuccess) {
	  log_gpu.error("CUDA: uncaught driver error in task %d: %d (%s)",
			func_id, result, cudaGetErrorString(result));
	}
      }
#endif

      //log_gpu(LEVEL_DEBUG, "task end: %d (%p) (%s)", func_id, fptr, argstr);
    }

    void GPUTask::finish_job(void)
    {
      // Clear out all our modules that we created
      for (std::set<void**>::const_iterator it = modules.begin();
            it != modules.end(); it++)
      {
        gpu->internal->unregister_fat_binary(*it);
      }
      // Then do the normal stuff
      GPUJob::finish_job();
    }

    // An abstract base class for all GPU memcpy operations
    class GPUMemcpy : public GPUJob {
    public:
      GPUMemcpy(GPUProcessor *_gpu, Event _finish_event,
                GPUMemcpyKind _kind)
        : GPUJob(_gpu, _finish_event), kind(_kind)
      {
        if (kind == GPU_MEMCPY_HOST_TO_DEVICE)
          local_stream = gpu->internal->host_to_device_stream;
        else if (kind == GPU_MEMCPY_DEVICE_TO_HOST)
          local_stream = gpu->internal->device_to_host_stream;
        else if (kind == GPU_MEMCPY_DEVICE_TO_DEVICE)
          local_stream = gpu->internal->device_to_device_stream;
        else if (kind == GPU_MEMCPY_PEER_TO_PEER)
          local_stream = gpu->internal->peer_to_peer_stream;
        else
          assert(false); // who does host to host here?!?
      }

      virtual ~GPUMemcpy(void) { }
    public:
      virtual bool event_triggered(void)
      {
        log_gpu.info("gpu job %p now runnable", this);
        gpu->internal->enqueue_copy(this);

        // don't delete
        return false;
      }

      virtual void print_info(FILE *f)
      {
        fprintf(f,"GPU Memcpy: %p after=" IDFMT "/%d\n",
            this, finish_event.id, finish_event.gen);
      }

      virtual void run_or_wait(Event start_event)
      {
        if(start_event.has_triggered()) {
          log_gpu.info("job %p can start right away!?", this);
          gpu->internal->enqueue_copy(this);
        } else {
          log_gpu.info("job %p waiting for " IDFMT "/%d", this, start_event.id, start_event.gen);
          start_event.impl()->add_waiter(start_event.gen, this);
        }
      }

      virtual void execute(void) = 0;

      void post_execute(void)
      { 
        CHECK_CU( cuEventRecord(complete_event, local_stream) );
        if (kind == GPU_MEMCPY_HOST_TO_DEVICE)
          gpu->internal->add_host_device_copy(this);
        else if (kind == GPU_MEMCPY_DEVICE_TO_HOST)
          gpu->internal->add_device_host_copy(this);
        else if (kind == GPU_MEMCPY_DEVICE_TO_DEVICE)
          gpu->internal->add_device_device_copy(this);
        else if (kind == GPU_MEMCPY_PEER_TO_PEER)
          gpu->internal->add_peer_to_peer_copy(this);
        else
          assert(false);
      }

    protected:
      GPUMemcpyKind kind;
      CUstream local_stream;
    };

    class GPUMemcpy1D : public GPUMemcpy {
    public:
      GPUMemcpy1D(GPUProcessor *_gpu, Event _finish_event,
		void *_dst, const void *_src, size_t _bytes, GPUMemcpyKind _kind)
	: GPUMemcpy(_gpu, _finish_event, _kind), dst(_dst), src(_src), 
	  mask(0), elmt_size(_bytes)
      {}

      GPUMemcpy1D(GPUProcessor *_gpu, Event _finish_event,
		void *_dst, const void *_src, 
		const ElementMask *_mask, size_t _elmt_size,
		GPUMemcpyKind _kind)
	: GPUMemcpy(_gpu, _finish_event, _kind), dst(_dst), src(_src),
	  mask(_mask), elmt_size(_elmt_size)
      {}

      virtual ~GPUMemcpy1D(void) { }

      void do_span(off_t pos, size_t len)
      {
	off_t span_start = pos * elmt_size;
	size_t span_bytes = len * elmt_size;

        switch (kind)
        {
          case GPU_MEMCPY_HOST_TO_DEVICE:
            {
              CHECK_CU( cuMemcpyHtoDAsync((CUdeviceptr)(((char*)dst)+span_start),
                                          (((char*)src)+span_start),
                                          span_bytes,
                                          local_stream) );
              break;
            }
          case GPU_MEMCPY_DEVICE_TO_HOST:
            {
              CHECK_CU( cuMemcpyDtoHAsync((((char*)dst)+span_start),
                                          (CUdeviceptr)(((char*)src)+span_start),
                                          span_bytes,
                                          local_stream) );
              break;
            }
          case GPU_MEMCPY_DEVICE_TO_DEVICE:
            {
              CHECK_CU( cuMemcpyDtoDAsync((CUdeviceptr)(((char*)dst)+span_start),
                                          (CUdeviceptr)(((char*)src)+span_start),
                                          span_bytes,
                                          local_stream) );
              break;
            }
          case GPU_MEMCPY_PEER_TO_PEER:
            {
              CUcontext src_ctx, dst_ctx;
              CHECK_CU( cuPointerGetAttribute(&src_ctx, 
                    CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr)src) );
              CHECK_CU( cuPointerGetAttribute(&dst_ctx, 
                    CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr)dst) );
              CHECK_CU( cuMemcpyPeerAsync((CUdeviceptr)(((char*)dst)+span_start), dst_ctx,
                                          (CUdeviceptr)(((char*)src)+span_start), src_ctx,
                                          span_bytes,
                                          local_stream) );
              break;
            }
          default:
            assert(false);
        }
      }

      virtual void execute(void)
      {
	DetailedTimer::ScopedPush sp(TIME_COPY);
	log_gpu.info("gpu memcpy: dst=%p src=%p bytes=%zd kind=%d",
		     dst, src, elmt_size, kind);
	if(mask) {
	  ElementMask::forall_ranges(*this, *mask);
	} else {
	  do_span(0, 1);
	}
        post_execute();  
	log_gpu.info("gpu memcpy complete: dst=%p src=%p bytes=%zd kind=%d",
		     dst, src, elmt_size, kind);
      } 
    protected:
      void *dst;
      const void *src;
      const ElementMask *mask;
      size_t elmt_size;
    };

    class GPUMemcpy2D : public GPUMemcpy {
    public:
      GPUMemcpy2D(GPUProcessor *_gpu, Event _finish_event,
                  void *_dst, const void *_src,
                  off_t _dst_stride, off_t _src_stride,
                  size_t _bytes, size_t _lines,
                  GPUMemcpyKind _kind)
        : GPUMemcpy(_gpu, _finish_event, _kind), dst(_dst), src(_src),
          dst_stride((_dst_stride < (off_t)_bytes) ? _bytes : _dst_stride), 
          src_stride((_src_stride < (off_t)_bytes) ? _bytes : _src_stride),
          bytes(_bytes), lines(_lines)
      {}

      virtual ~GPUMemcpy2D(void) { }
    public:
      virtual void execute(void)
      {
        log_gpu.info("gpu memcpy 2d: dst=%p src=%p "
                     "dst_off=%ld src_off=%ld bytes=%ld lines=%ld kind=%d",
                     dst, src, dst_stride, src_stride, bytes, lines, kind); 
        CUDA_MEMCPY2D copy_info;
        if (kind == GPU_MEMCPY_PEER_TO_PEER) {
          // If we're doing peer to peer, just let unified memory it deal with it
          copy_info.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
          copy_info.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
        } else {
          // otherwise we know the answers here 
          copy_info.srcMemoryType = (kind == GPU_MEMCPY_HOST_TO_DEVICE) ?
            CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE;
          copy_info.dstMemoryType = (kind == GPU_MEMCPY_DEVICE_TO_HOST) ? 
            CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE;
        }
        copy_info.srcDevice = (CUdeviceptr)src;
        copy_info.srcHost = src;
        copy_info.srcPitch = src_stride;
        copy_info.srcY = 0;
        copy_info.srcXInBytes = 0;
        copy_info.dstDevice = (CUdeviceptr)dst;
        copy_info.dstHost = dst;
        copy_info.dstPitch = dst_stride;
        copy_info.dstY = 0;
        copy_info.dstXInBytes = 0;
        copy_info.WidthInBytes = bytes;
        copy_info.Height = lines;
        CHECK_CU( cuMemcpy2DAsync(&copy_info, local_stream) );
        post_execute();
        log_gpu.info("gpu memcpy 2d complete: dst=%p src=%p "
                     "dst_off=%ld src_off=%ld bytes=%ld lines=%ld kind=%d",
                     dst, src, dst_stride, src_stride, bytes, lines, kind);
      }
    protected:
      void *dst;
      const void *src;
      off_t dst_stride, src_stride;
      size_t bytes, lines;
    };

#if 0
    class GPUMemcpyGeneric : public GPUJob {
    public:
      struct PendingMemcpy {
      public:
        PendingMemcpy(off_t o, void *s, size_t b)
          : offset(o), src(s), bytes(b) { }
      public:
        off_t offset;
        void *src;
        size_t bytes;
      };
    public:
      GPUMemcpyGeneric(GPUProcessor *_gpu, Event _finish_event,
		       void *_gpu_ptr, Memory::Impl *_memory, off_t _mem_offset, size_t _bytes, cudaMemcpyKind _kind)
	: GPUJob(_gpu, _finish_event), gpu_ptr(_gpu_ptr), memory(_memory),
	  mem_offset(_mem_offset), mask(0), elmt_size(_bytes), kind(_kind)
      {}

      GPUMemcpyGeneric(GPUProcessor *_gpu, Event _finish_event,
		       void *_gpu_ptr, Memory::Impl *_memory, off_t _mem_offset,
		       const ElementMask *_mask, size_t _elmt_size,
		       cudaMemcpyKind _kind)
	: GPUJob(_gpu, _finish_event), gpu_ptr(_gpu_ptr), memory(_memory),
	  mem_offset(_mem_offset), mask(_mask), elmt_size(_elmt_size), kind(_kind)
      {}

      virtual bool event_triggered(void)
      {
        log_gpu.info("gpu job %p now runnable", this);
        gpu->internal->enqueue_copy(this);

        // don't delete
        return false;
      }

      virtual void run_or_wait(Event start_event)
      {
        if(start_event.has_triggered()) {
          log_gpu.info("job %p can start right away!?", this);
          gpu->internal->enqueue_copy(this);
        } else {
          log_gpu.info("job %p waiting for " IDFMT "/%d", this, start_event.id, start_event.gen);
          start_event.impl()->add_waiter(start_event, this);
        }
      }

      void do_span(off_t pos, size_t len)
      {
	off_t span_start = pos * elmt_size;
	size_t span_bytes = len * elmt_size;

        // First check to see if we need to make a new buffer
        if (span_bytes > remaining_bytes)
        {
          // Make a new buffer
          if (span_bytes > base_buffer_size)
          {
            CHECK_CUDART( cudaMallocHost((void**)&current_buffer, span_bytes) );
            remaining_bytes = span_bytes;
          }
          else
          {
            CHECK_CUDART( cudaMallocHost((void**)&current_buffer, base_buffer_size) );
            remaining_bytes = base_buffer_size;
          }
          allocated_buffers.push_back(current_buffer);
        }
        // Now issue the copy
        if (kind == cudaMemcpyDeviceToHost) {
          // In this case we have to issue the copy and then deffer
          // the memory copy until we know the full copy is complete
          CHECK_CUDART( cudaMemcpyAsync(current_buffer,
                                        ((char*)gpu_ptr)+span_start,
                                        span_bytes,
                                        cudaMemcpyDefault,
                                        local_stream) );
          pending_copies.push_back(PendingMemcpy(mem_offset+span_start,
                                                 current_buffer, span_bytes));
        } else {
          memory->get_bytes(mem_offset+span_start, current_buffer, span_bytes);
          CHECK_CUDART( cudaMemcpyAsync(((char*)gpu_ptr)+span_start,
                                        current_buffer,
                                        span_bytes,
                                        cudaMemcpyDefault,
                                        local_stream) );
        }
        // Update the pointer
        current_buffer += span_bytes;
        // Mark how many bytes we used
        remaining_bytes -= span_bytes;
      }

      virtual void execute(void)
      {
	DetailedTimer::ScopedPush sp(TIME_COPY);
        // Figure out which stream we are based on our kind
        if (kind == cudaMemcpyHostToDevice) {
          local_stream = gpu->internal->host_to_device_stream;
          gpu->internal->add_host_device_copy(this);
        }
        else if (kind == cudaMemcpyDeviceToHost) {
          local_stream = gpu->internal->device_to_host_stream;
          gpu->internal->add_device_host_copy(this);
        }
        else if (kind == cudaMemcpyDeviceToDevice) {
          local_stream = gpu->internal->device_to_device_stream;
          gpu->internal->add_device_device_copy(this);
        }
        else
          assert(false); // who does host to host here?!?
	log_gpu.info("gpu memcpy generic: gpuptr=%p mem=" IDFMT " offset=%zd bytes=%zd kind=%d",
		     gpu_ptr, memory->me.id, mem_offset, elmt_size, kind);
        // Initialize our first buffer
        CHECK_CUDART( cudaMallocHost((void**)&current_buffer, base_buffer_size) );
        remaining_bytes = base_buffer_size;
        allocated_buffers.push_back(current_buffer);
	if(mask) {
	  ElementMask::forall_ranges(*this, *mask);
	} else {
	  do_span(0, 1);
	}

	log_gpu.info("gpu memcpy generic done: gpuptr=%p mem=" IDFMT " offset=%zd bytes=%zd kind=%d",
		     gpu_ptr, memory->me.id, mem_offset, elmt_size, kind);
        // Make an event and put it on the stream
        CHECK_CUDART( cudaEventCreateWithFlags(&completion_event,
                                               cudaEventDisableTiming) );
        CHECK_CUDART( cudaEventRecord(completion_event, local_stream) );
      }

      virtual bool is_finished(void)
      {
        cudaError_t result = cudaEventQuery(completion_event);
        if (result == cudaSuccess)
          return true;
        else if (result == cudaErrorNotReady)
          return false;
        else
        {
          CHECK_CUDART( result );
        }
        return false;
      }

      virtual void finish_job(void)
      {
        // If we had any pending copies, do them before triggering the event
        for (unsigned idx = 0; idx < pending_copies.size(); idx++)
        {
          PendingMemcpy &copy = pending_copies[idx];
          memory->put_bytes(copy.offset, copy.src, copy.bytes);
        }
        // If we have a finish event then trigger it
        if (finish_event.exists())
          finish_event.impl()->trigger(finish_event.gen, gasnet_mynode());
        // Free up any buffers that we made when performing the copy
        for (unsigned idx = 0; idx < allocated_buffers.size(); idx++)
        {
          CHECK_CUDART( cudaFreeHost(allocated_buffers[idx]) );
        }
        allocated_buffers.clear();
        // Free up our event
        CHECK_CU( cuEventDestroy(completion_event) );
      }

    protected:
      void *gpu_ptr;
      Memory::Impl *memory;
      off_t mem_offset;
      const ElementMask *mask;
      size_t elmt_size;
      cudaMemcpyKind kind;
      cudaStream_t local_stream;
      cudaEvent_t completion_event;
      std::deque<void*> allocated_buffers;
      std::deque<PendingMemcpy> pending_copies;
    public:
      // Default buffer size
      static const size_t base_buffer_size = 65536;
    protected:
      size_t remaining_bytes;
      char *current_buffer;
    };
#endif

    GPUProcessor::GPUProcessor(Processor _me, int _gpu_index, 
             int num_local_gpus, Processor _util,
	     size_t _zcmem_size, size_t _fbmem_size, 
             size_t _stack_size, bool gpu_dma_thread, int _streams)
      : Processor::Impl(_me, Processor::TOC_PROC, _util)
    {
      internal = new GPUProcessor::Internal(this, _gpu_index, num_local_gpus,
                                            _zcmem_size, _fbmem_size,
                                            (16 << 20), (32 << 20),
                                            !_util.exists(), gpu_dma_thread, _streams);

      // enqueue a GPU init job before we do anything else
      Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_INIT);
#ifndef EVENT_GRAPH_TRACE
      if(it != task_id_table.end())
	internal->enqueue_task(new GPUTask(this, Event::NO_EVENT,
					  Processor::TASK_ID_PROCESSOR_INIT, 0, 0, 0));
#else
      enclosing_stack.push_back(Event::NO_EVENT);
      if(it != task_id_table.end())
        internal->enqueue_task(new GPUTask(this, Event::Impl::create_event(), 
                                           Processor::TASK_ID_PROCESSOR_INIT, 0, 0, 0));
#endif

      internal->create_gpu_thread(_stack_size);
    }

    GPUProcessor::~GPUProcessor(void)
    {
      delete internal;
    }

    /*static*/ Processor GPUProcessor::get_processor(void)
    {
      void *tls_val = gasnett_threadkey_get(gpu_thread);
      // If this happens there is a case we're not handling
      assert(tls_val != NULL);
      GPUProcessor *gpu = (GPUProcessor*)tls_val;
      return gpu->me;
    }

    void GPUProcessor::start_worker_thread(void)
    {
      AutoHSLLock a(internal->mutex);
      log_gpu.info("enabling worker thread");
      internal->worker_enabled = true;
      gasnett_cond_signal(&internal->worker_condvar);
    }

    void *GPUProcessor::get_zcmem_cpu_base(void)
    {
      return ((char *)internal->zcmem_cpu_base) + internal->zcmem_reserve;
    }

    void *GPUProcessor::get_fbmem_gpu_base(void)
    {
      return ((char *)internal->fbmem_gpu_base) + internal->fbmem_reserve;
    }

    void GPUProcessor::tasks_available(int priority)
    {
      internal->tasks_available(priority);
    }

    void GPUProcessor::enqueue_task(Task *task)
    {
      // should never be called
      assert(0);
    }

    void GPUProcessor::spawn_task(Processor::TaskFuncID func_id,
				  const void *args, size_t arglen,
				  //std::set<RegionInstanceUntyped> instances_needed,
				  Event start_event, Event finish_event,
                                  int priority)
    {
      log_gpu.info("new gpu task: func_id=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
		   func_id, start_event.id, start_event.gen, finish_event.id, finish_event.gen);
      if(func_id != 0) {
	(new GPUTask(this, finish_event,
		     func_id, args, arglen, priority))->run_or_wait(start_event);
      } else {
	AutoHSLLock a(internal->mutex);
	log_gpu.info("received shutdown request!");
	internal->shutdown_requested = true;
	gasnett_cond_signal(&internal->worker_condvar);
      }
    }

    void GPUProcessor::enable_idle_task(void)
    {
      log_gpu.info("idle task enabled for processor " IDFMT "", me.id);
#ifdef UTIL_PROCS_FOR_GPU
      if (util_proc)
        util_proc->enable_idle_task(this);
      else
#endif
      {
        AutoHSLLock a(internal->mutex);
        internal->idle_task_enabled = true;
        gasnett_cond_signal(&internal->worker_condvar);
      }
    }

    void GPUProcessor::disable_idle_task(void)
    {
      //log_gpu.info("idle task NOT disabled for processor " IDFMT "", me.id);
      log_gpu.info("idle task disabled for processor " IDFMT "", me.id);
#ifdef UTIL_PROCS_FOR_GPU
      if (util_proc)
        util_proc->disable_idle_task(this);
      else
#endif
        internal->idle_task_enabled = false;
    }

    void GPUProcessor::copy_to_fb(off_t dst_offset, const void *src, size_t bytes,
				  Event start_event, Event finish_event)
    {
      (new GPUMemcpy1D(this, finish_event,
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + dst_offset),
		     src,
		     bytes,
		     GPU_MEMCPY_HOST_TO_DEVICE))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_to_fb(off_t dst_offset, const void *src,
				  const ElementMask *mask, size_t elmt_size,
				  Event start_event, Event finish_event)
    {
      (new GPUMemcpy1D(this, finish_event,
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + dst_offset),
		     src,
		     mask, elmt_size,
		     GPU_MEMCPY_HOST_TO_DEVICE))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_from_fb(void *dst, off_t src_offset, size_t bytes,
				    Event start_event, Event finish_event)
    {
      (new GPUMemcpy1D(this, finish_event,
		     dst,
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + src_offset),
		     bytes,
		     GPU_MEMCPY_DEVICE_TO_HOST))->run_or_wait(start_event);
    } 

    void GPUProcessor::copy_from_fb(void *dst, off_t src_offset,
				    const ElementMask *mask, size_t elmt_size,
				    Event start_event, Event finish_event)
    {
      (new GPUMemcpy1D(this, finish_event,
		     dst,
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + src_offset),
		     mask, elmt_size,
		     GPU_MEMCPY_DEVICE_TO_HOST))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_within_fb(off_t dst_offset, off_t src_offset,
				      size_t bytes,
				      Event start_event, Event finish_event)
    {
      (new GPUMemcpy1D(this, finish_event,
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + dst_offset),
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + src_offset),
		     bytes,
		     GPU_MEMCPY_DEVICE_TO_DEVICE))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_within_fb(off_t dst_offset, off_t src_offset,
				      const ElementMask *mask, size_t elmt_size,
				      Event start_event, Event finish_event)
    {
      (new GPUMemcpy1D(this, finish_event,
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + dst_offset),
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + src_offset),
		     mask, elmt_size,
		     GPU_MEMCPY_DEVICE_TO_DEVICE))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_to_fb_2d(off_t dst_offset, const void *src, 
                                     off_t dst_stride, off_t src_stride,
                                     size_t bytes, size_t lines,
                                     Event start_event, Event finish_event)
    {
      (new GPUMemcpy2D(this, finish_event,
                       ((char*)internal->fbmem_gpu_base)+
                        (internal->fbmem_reserve + dst_offset),
                        src, dst_stride, src_stride, bytes, lines,
                        GPU_MEMCPY_HOST_TO_DEVICE))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_from_fb_2d(void *dst, off_t src_offset,
                                       off_t dst_stride, off_t src_stride,
                                       size_t bytes, size_t lines,
                                       Event start_event, Event finish_event)
    {
      (new GPUMemcpy2D(this, finish_event, dst,
                       ((char*)internal->fbmem_gpu_base)+
                        (internal->fbmem_reserve + src_offset),
                        dst_stride, src_stride, bytes, lines,
                        GPU_MEMCPY_DEVICE_TO_HOST))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_within_fb_2d(off_t dst_offset, off_t src_offset,
                                         off_t dst_stride, off_t src_stride,
                                         size_t bytes, size_t lines,
                                         Event start_event, Event finish_event)
    {
      (new GPUMemcpy2D(this, finish_event,
                       ((char*)internal->fbmem_gpu_base) + 
                        (internal->fbmem_reserve + dst_offset),
                       ((char*)internal->fbmem_gpu_base) + 
                        (internal->fbmem_reserve + src_offset),
                        dst_stride, src_stride, bytes, lines,
                        GPU_MEMCPY_DEVICE_TO_DEVICE))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_to_peer(GPUProcessor *dst, off_t dst_offset,
                                    off_t src_offset, size_t bytes,
                                    Event start_event, Event finish_event)
    {
      (new GPUMemcpy1D(this, finish_event,
              ((char*)dst->internal->fbmem_gpu_base) + 
                      (dst->internal->fbmem_reserve + dst_offset),
              ((char*)internal->fbmem_gpu_base) + 
                      (internal->fbmem_reserve + src_offset),
              bytes, GPU_MEMCPY_PEER_TO_PEER))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_to_peer_2d(GPUProcessor *dst,
                                       off_t dst_offset, off_t src_offset,
                                       off_t dst_stride, off_t src_stride,
                                       size_t bytes, size_t lines,
                                       Event start_event, Event finish_event)
    {
      (new GPUMemcpy2D(this, finish_event,
                       ((char*)dst->internal->fbmem_gpu_base) +
                                (dst->internal->fbmem_reserve + dst_offset),
                       ((char*)internal->fbmem_gpu_base) +
                                (internal->fbmem_reserve + src_offset),
                        dst_stride, src_stride, bytes, lines,
                        GPU_MEMCPY_PEER_TO_PEER))->run_or_wait(start_event);
    }

    void GPUProcessor::register_host_memory(void *base, size_t size)
    {
      internal->register_host_memory(base, size);
    }

    void GPUProcessor::enable_peer_access(GPUProcessor *peer)
    {
      internal->enable_peer_access(peer->internal);
      peer_gpus.insert(peer);
    }

    bool GPUProcessor::can_access_peer(GPUProcessor *peer) const
    {
      return (peer_gpus.find(peer) != peer_gpus.end());
    }

    void GPUProcessor::handle_copies(void)
    {
      internal->handle_copies();
    }

    /*static*/ GPUProcessor** GPUProcessor::node_gpus;
    /*static*/ size_t GPUProcessor::num_node_gpus;
    static volatile bool dma_shutdown_requested = false;
    static std::vector<pthread_t> dma_threads;

    struct gpu_dma_args {
      GPUProcessor **node_gpus;
      size_t num_node_gpus;
    };

    /*static*/
    void* GPUProcessor::gpu_dma_worker_loop(void *args)
    {
      size_t num_local = GPUProcessor::num_node_gpus;
      GPUProcessor **local_gpus = GPUProcessor::node_gpus;
      while (!dma_shutdown_requested)
      {
        // Iterate over all the GPU processors and perform the copies
        for (unsigned idx = 0; idx < num_local; idx++)
        {
          local_gpus[idx]->handle_copies();
        }
      }
      free(local_gpus);
      return NULL;
    }

    /*static*/
    void GPUProcessor::start_gpu_dma_thread(const std::vector<GPUProcessor*> &local)
    {
      GPUProcessor::num_node_gpus = local.size();
      GPUProcessor::node_gpus = (GPUProcessor**)malloc(local.size()*sizeof(GPUProcessor*));
      for (unsigned idx = 0; idx < local.size(); idx++)
        GPUProcessor::node_gpus[idx] = local[idx];
      pthread_attr_t attr;
      CHECK_PTHREAD( pthread_attr_init(&attr) );
      if (proc_assignment)
        proc_assignment->bind_thread(-1, &attr, "GPU DMA worker");
      pthread_t thread;
      CHECK_PTHREAD( pthread_create(&thread, 0, gpu_dma_worker_loop, 0) );
      CHECK_PTHREAD( pthread_attr_destroy(&attr) );
      dma_threads.push_back(thread);
#ifdef DEADLOCK_TRACE
      Runtime::get_runtime()->add_thread(&thread);
#endif
    }

    /*static*/
    void GPUProcessor::stop_gpu_dma_threads(void)
    {
      dma_shutdown_requested = true;

      // no need to signal right now - they're all spinning
      while(!dma_threads.empty()) {
	pthread_t t = dma_threads.back();
	dma_threads.pop_back();
	
	void *dummy;
	CHECK_PTHREAD( pthread_join(t, &dummy) );
      }

      dma_shutdown_requested = false;
    }

#if 0
    void GPUProcessor::copy_to_fb_generic(off_t dst_offset, 
					  Memory::Impl *src_mem, off_t src_offset,
					  size_t bytes,
					  Event start_event, Event finish_event)
    {
      (new GPUMemcpyGeneric(this, finish_event,
			    ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + dst_offset),
			    src_mem, src_offset,
			    bytes,
			    cudaMemcpyHostToDevice))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_to_fb_generic(off_t dst_offset, 
					  Memory::Impl *src_mem, off_t src_offset,
					  const ElementMask *mask, 
					  size_t elmt_size,
					  Event start_event, Event finish_event)
    {
      (new GPUMemcpyGeneric(this, finish_event,
			    ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + dst_offset),
			    src_mem, src_offset,
			    mask, elmt_size,
			    cudaMemcpyHostToDevice))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_from_fb_generic(Memory::Impl *dst_mem, off_t dst_offset, 
					    off_t src_offset, size_t bytes,
					    Event start_event, Event finish_event)
    {
      (new GPUMemcpyGeneric(this, finish_event,
			    ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + src_offset),
			    dst_mem, dst_offset,
			    bytes,
			    cudaMemcpyDeviceToHost))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_from_fb_generic(Memory::Impl *dst_mem, off_t dst_offset, 
					    off_t src_offset,
					    const ElementMask *mask,
					    size_t elmt_size,
					    Event start_event, Event finish_event)
    {
      (new GPUMemcpyGeneric(this, finish_event,
			    ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + src_offset),
			    dst_mem, dst_offset,
			    mask, elmt_size,
			    cudaMemcpyDeviceToHost))->run_or_wait(start_event);
    }
#endif

    // framebuffer memory

    GPUFBMemory::GPUFBMemory(Memory _me, GPUProcessor *_gpu)
      : Memory::Impl(_me, _gpu->internal->fbmem_size, MKIND_GPUFB, 512, Memory::GPU_FB_MEM),
	gpu(_gpu)
    {
      base = (char *)(gpu->get_fbmem_gpu_base());
      free_blocks[0] = size;
    }

    GPUFBMemory::~GPUFBMemory(void) {}

    // these work, but they are SLOW
    void GPUFBMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      // create an async copy and then wait for it to finish...
      Event e = GenEventImpl::create_genevent()->current_event();
      gpu->copy_from_fb(dst, offset, size, Event::NO_EVENT, e);
      e.wait(true /*blocking*/);
    }

    void GPUFBMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      // create an async copy and then wait for it to finish...
      Event e = GenEventImpl::create_genevent()->current_event();
      gpu->copy_to_fb(offset, src, size, Event::NO_EVENT, e);
      e.wait(true /*blocking*/);
    }

    // zerocopy memory

    GPUZCMemory::GPUZCMemory(Memory _me, GPUProcessor *_gpu)
      : Memory::Impl(_me, _gpu->internal->zcmem_size, MKIND_ZEROCOPY, 256, Memory::Z_COPY_MEM),
	gpu(_gpu)
    {
      cpu_base = (char *)(gpu->get_zcmem_cpu_base());
      free_blocks[0] = size;
    }

    GPUZCMemory::~GPUZCMemory(void) {}

#if 0
    template <>
    bool RegionAccessor<AccessorGeneric>::can_convert<AccessorGPU>(void) const
    {
      RegionInstance::Impl *i_impl = (RegionInstance::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      // make sure it's not a reduction fold-only instance
      StaticAccess<RegionInstance::Impl> i_data(i_impl);
      if(i_data->is_reduction) return false;

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == Memory::Impl::MKIND_GPUFB) return true;
      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) return true;
      return false;
    }
#endif

#ifdef POINTER_CHECKS
    static unsigned *get_gpu_valid_mask(RegionMetaDataUntyped region)
    {
	const ElementMask &mask = region.get_valid_mask();
	void *valid_mask_base;
	for(size_t p = 0; p < mask.raw_size(); p += 4)
	  log_gpu.info("  raw mask data[%zd] = %08x\n", p,
		       ((unsigned *)(mask.get_raw()))[p>>2]);
        CHECK_CU( cuMemAlloc((cuDevicePtr*)(&valid_mask_base), mask.raw_size()) );
	log_gpu.info("copy of valid mask (%zd bytes) created at %p",
		     mask.raw_size(), valid_mask_base);
        CHECK_CU( cuMemcpyHtoD(vald_mask_base, 
                               mask.get_raw(),
                               mask.raw_size()) );
	return (unsigned *)&(((ElementMaskImpl *)valid_mask_base)->bits);
    }
#endif

#ifdef OLD_ACCESSORS    
    template <>
    RegionAccessor<AccessorGPU> RegionAccessor<AccessorGeneric>::convert<AccessorGPU>(void) const
    {
      RegionInstance::Impl *i_impl = (RegionInstance::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      StaticAccess<RegionInstance::Impl> i_data(i_impl);

      assert(!i_data->is_reduction);

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == Memory::Impl::MKIND_GPUFB) {
	GPUFBMemory *fbm = (GPUFBMemory *)m_impl;
	void *base = (((char *)(fbm->gpu->internal->fbmem_gpu_base)) +
		      fbm->gpu->internal->fbmem_reserve);
	log_gpu.info("creating gpufb accessor (%p + %zd = %p) (%p)",
		     base, i_data->access_offset,
		     ((char *)base)+(i_data->access_offset),
		     ((char *)base)+(i_data->alloc_offset));
	RegionAccessor<AccessorGPU> ria(((char *)base)+(i_data->access_offset));
#ifdef POINTER_CHECKS 
        ria.first_elmt = i_data->first_elmt;
        ria.last_elmt  = i_data->last_elmt;
	ria.valid_mask_base = get_gpu_valid_mask(i_data->region);
#endif
	return ria;
      }

      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) {
	GPUZCMemory *zcm = (GPUZCMemory *)m_impl;
	void *base = (((char *)(zcm->gpu->internal->zcmem_gpu_base)) +
		      zcm->gpu->internal->zcmem_reserve);
	log_gpu.info("creating gpuzc accessor (%p + %zd = %p)",
		     base, i_data->access_offset,
		     ((char *)base)+(i_data->access_offset));
	RegionAccessor<AccessorGPU> ria(((char *)base)+(i_data->access_offset));
#ifdef POINTER_CHECKS 
        ria.first_elmt = i_data->first_elmt;
        ria.last_elmt  = i_data->last_elmt;
	ria.valid_mask_base = get_gpu_valid_mask(i_data->region);
#endif
	return ria;
      }

      assert(0);
    }

    template <>
    bool RegionAccessor<AccessorGeneric>::can_convert<AccessorGPUReductionFold>(void) const
    {
      RegionInstance::Impl *i_impl = (RegionInstance::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      // make sure it's a reduction fold-only instance
      StaticAccess<RegionInstance::Impl> i_data(i_impl);
      if(!i_data->is_reduction) return false;

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == Memory::Impl::MKIND_GPUFB) return true;
      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) return true;
      return false;
    }
    
    template <>
    RegionAccessor<AccessorGPUReductionFold> RegionAccessor<AccessorGeneric>::convert<AccessorGPUReductionFold>(void) const
    {
      RegionInstance::Impl *i_impl = (RegionInstance::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      StaticAccess<RegionInstance::Impl> i_data(i_impl);

      assert(i_data->is_reduction);

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == Memory::Impl::MKIND_GPUFB) {
	GPUFBMemory *fbm = (GPUFBMemory *)m_impl;
	void *base = (((char *)(fbm->gpu->internal->fbmem_gpu_base)) +
		      fbm->gpu->internal->fbmem_reserve);
	log_gpu.info("creating gpufb reduction accessor (%p + %zd = %p)",
		     base, i_data->access_offset,
		     ((char *)base)+(i_data->access_offset));
	RegionAccessor<AccessorGPUReductionFold> ria(((char *)base)+(i_data->access_offset));
#ifdef POINTER_CHECKS 
        ria.first_elmt = i_data->first_elmt;
        ria.last_elmt  = i_data->last_elmt;
	ria.valid_mask_base = get_gpu_valid_mask(i_data->region);
#endif
	return ria;
      }

      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) {
	GPUZCMemory *zcm = (GPUZCMemory *)m_impl;
	void *base = (((char *)(zcm->gpu->internal->zcmem_gpu_base)) +
		      zcm->gpu->internal->zcmem_reserve);
	log_gpu.info("creating gpuzc reduction accessor (%p + %zd = %p)",
		     base, i_data->access_offset,
		     ((char *)base)+(i_data->access_offset));
	RegionAccessor<AccessorGPUReductionFold> ria(((char *)base)+(i_data->access_offset));
#ifdef POINTER_CHECKS 
        ria.first_elmt = i_data->first_elmt;
        ria.last_elmt  = i_data->last_elmt;
	ria.valid_mask_base = get_gpu_valid_mask(i_data->region);
#endif
	return ria;
      }

      assert(0);
    }
#endif

    // Helper methods for emulating the cuda runtime
    /*static*/ GPUProcessor* GPUProcessor::find_local_gpu(void)
    {
      void *tls_val = gasnett_threadkey_get(gpu_thread);
      // This can return NULL during start-up
      if (tls_val == NULL)
        return NULL;
      assert(tls_val != NULL);
      GPUProcessor *local = (GPUProcessor*)tls_val;
      return local;
    }

    /*static*/ std::map<void*,void**>& GPUProcessor::Internal::get_deferred_modules(void)
    {
      static std::map<void*,void**> deferred_modules;
      return deferred_modules;
    }

    /*static*/ std::deque<GPUProcessor::Internal::DeferredFunction>&
                                      GPUProcessor::Internal::get_deferred_functions(void)
    {
      static std::deque<DeferredFunction> deferred_functions;
      return deferred_functions;
    }

    /*static*/ std::deque<GPUProcessor::Internal::DeferredVariable>&
                                      GPUProcessor::Internal::get_deferred_variables(void)
    {
      static std::deque<DeferredVariable> deferred_variables;
      return deferred_variables;
    }

    void** GPUProcessor::Internal::register_fat_binary(void *fat_bin)
    {
      void **handle = (void**)malloc(sizeof(void**));
      *handle = fat_bin;
      ModuleInfo &info = modules[handle];
      FatBin *fatbin_args = (FatBin*)fat_bin;
      assert((fatbin_args->data != NULL) ||
             (fatbin_args->filename_or_fatbins != NULL));
      if (fatbin_args->data != NULL)
      {
        CHECK_CU( cuModuleLoadData(&(info.module), fatbin_args->data) );
      } else {
        CHECK_CU( cuModuleLoad(&(info.module), (const char*)fatbin_args->filename_or_fatbins) );
      }
      // add this to the list of local modules to be created during this task
      task_modules.insert(handle);
      return handle;
    }

    /*static*/ void** GPUProcessor::Internal::defer_module_load(void *fat_bin)
    {
      void **handle = (void**)malloc(sizeof(void**));
      *handle = fat_bin;
      // Assume we don't need a lock here because all this is sequential
      get_deferred_modules()[fat_bin] = handle;
      return handle;
    }

    /*static*/ void** GPUProcessor::register_fat_binary(void *fat_bin)
    {
      GPUProcessor *local = find_local_gpu(); 
      // Ignore anything that goes on during start-up
      if (local == NULL)
        return GPUProcessor::Internal::defer_module_load(fat_bin);
      return local->internal->register_fat_binary(fat_bin);
    }

    void GPUProcessor::Internal::unregister_fat_binary(void **fat_bin)
    {
      // It's never safe to unload our modules here
      // We need to wait until we synchronize the current
      // task stream with all the things that we've done
      std::map<void**,ModuleInfo>::iterator finder = modules.find(fat_bin);
      assert(finder != modules.end());
      CHECK_CU( cuModuleUnload(finder->second.module) );
      for (std::set<const void*>::const_iterator it = finder->second.host_aliases.begin();
            it != finder->second.host_aliases.end(); it++)
      {
        device_functions.erase(*it);
      }
      for (std::set<const void*>::const_iterator it = finder->second.var_aliases.begin();
            it != finder->second.var_aliases.end(); it++)
      {
        device_variables.erase(*it);
      }
      modules.erase(finder);
      free(fat_bin);
    }
    
    /*static*/ void GPUProcessor::unregister_fat_binary(void **fat_bin)
    {
      // Do nothing, our task contexts will clean themselves up after they are done
    }

    void GPUProcessor::Internal::register_var(void **fat_bin, char *host_var,
                                              const char *device_name,
                                              bool ext, int size, bool constant, bool global)
    {
      std::map<void**,ModuleInfo>::iterator mod_finder = modules.find(fat_bin);
      assert(mod_finder != modules.end());
      ModuleInfo &info = mod_finder->second;
      // Check to see if we already have this symbol
      std::map<const void*,VarInfo>::const_iterator var_finder = 
        device_variables.find(host_var);
      if (var_finder == device_variables.end())
      {
        VarInfo target;
        CHECK_CU( cuModuleGetGlobal(&(target.ptr), &(target.size), info.module, device_name) );
        target.name = device_name;
        device_variables[host_var] = target;
        info.var_aliases.insert(host_var);
      }
    }

    /*static*/ void GPUProcessor::Internal::defer_variable_load(void **fat_bin,
                                                                char *host_var,
                                                                const char *device_name,
                                                                bool ext, int size,
                                                                bool constant, bool global)
    {
      DeferredVariable var;
      var.handle = fat_bin;
      var.host_var = host_var;
      var.device_name = device_name;
      var.external = ext;
      var.size = size;
      var.constant = constant;
      var.global = global;
      get_deferred_variables().push_back(var);
    }

    /*static*/ void GPUProcessor::register_var(void **fat_bin, char *host_var,
                                               char *device_addr, const char *device_name,
                                               int ext, int size, int constant, int global)
    {
      GPUProcessor *local = find_local_gpu();
      if (local == NULL)
      {
        GPUProcessor::Internal::defer_variable_load(fat_bin, host_var, device_name,
                                                    (ext == 1), size, 
                                                    (constant == 1), (global == 1));
        return;
      }
      local->internal->register_var(fat_bin, host_var, device_name, 
                                    (ext == 1), size, (constant == 1), (global == 1));
    }

    void GPUProcessor::Internal::register_function(void **fat_bin, const char *host_fun,
                                                   const char *device_fun) 
    {
      // Check to see if we already loaded this function
      std::map<const void*,CUfunction>::const_iterator func_finder = 
        device_functions.find(host_fun);
      if (func_finder != device_functions.end())
        return;
      // Otherwise we need to load it
      std::map<void**,ModuleInfo>::iterator mod_finder = modules.find(fat_bin);
      assert(mod_finder != modules.end());
      ModuleInfo &info = mod_finder->second;      
      info.host_aliases.insert(host_fun);
      CUfunction *func = &(device_functions[host_fun]); 
      CHECK_CU( cuModuleGetFunction(func, info.module, device_fun) );
    }

    /*static*/ void GPUProcessor::Internal::defer_function_load(void **fat_bin,
                                                                const char *host_fun,
                                                                const char *device_fun)
    {
      DeferredFunction df;
      df.handle = fat_bin;
      df.host_fun = host_fun;
      df.device_fun = device_fun;
      // Assume we don't need a lock here
      get_deferred_functions().push_back(df); 
    }

    /*static*/ void GPUProcessor::register_function(void **fat_bin, const char *host_fun,
                                                    char *device_fun, const char *device_name,
                                                    int thread_limit, uint3 *tid, uint3 *bid,
                                                    dim3 *bDim, dim3 *gDim, int *wSize)
    {
      GPUProcessor *local = find_local_gpu();
      if (local == NULL)
      {
        GPUProcessor::Internal::defer_function_load(fat_bin, host_fun, device_fun);
        return;
      }
      local->internal->register_function(fat_bin, host_fun, device_fun);
    }

    char GPUProcessor::Internal::init_module(void **fat_bin)
    {
      // We don't really care about managed runtimes
      return 1;
    }

    /*static*/ char GPUProcessor::init_module(void **fat_bin)
    {
      GPUProcessor *local = find_local_gpu();
      if (local == NULL)
        return 1;
      return local->internal->init_module(fat_bin);
    }

    // Helper methods for replacing the CUDA runtime API calls
    /*static*/ cudaError_t GPUProcessor::stream_create(cudaStream_t *stream)
    {
      log_gpu(LEVEL_ERROR,"Stream creation not permitted in Legion CUDA!");
      assert(false);
      return cudaSuccess;
    }

    /*static*/ cudaError_t GPUProcessor::stream_destroy(cudaStream_t stream)
    {
      log_gpu(LEVEL_ERROR,"Stream destruction not permitted in Legion CUDA!");
      assert(false);
      return cudaSuccess;
    }

    cudaError_t GPUProcessor::Internal::stream_synchronize(void)
    {
      CUstream current = get_current_task_stream();
      CHECK_CU( cuStreamSynchronize(current) );
      return cudaSuccess;
    }

    /*static*/ cudaError_t GPUProcessor::stream_synchronize(cudaStream_t stream)
    {
      // Just ignore whatever stream they passed in and synchronize on our stream
      GPUProcessor *local = find_local_gpu();
      assert(local != NULL);
      return local->internal->stream_synchronize();
    }

    cudaError_t GPUProcessor::Internal::configure_call(dim3 grid_dim,
                                                       dim3 block_dim,
                                                       size_t shared_mem)
    {
      LaunchConfig config;
      config.grid = grid_dim;
      config.block = block_dim;
      config.shared = shared_mem;
      launch_configs.push_back(config);
      return cudaSuccess;
    }

    /*static*/ cudaError_t GPUProcessor::configure_call(dim3 grid_dim, dim3 block_dim,
                                                        size_t shared_memory, 
                                                        cudaStream_t stream)
    {
      // Ignore their stream, it is meaningless
      GPUProcessor *local = find_local_gpu();
      assert(local != NULL);
      return local->internal->configure_call(grid_dim, block_dim, shared_memory);
    }

    cudaError_t GPUProcessor::Internal::setup_argument(const void *arg,
                                                       size_t size, size_t offset)
    {
      // Should always be pushing onto the back
      assert(offset >= argument_offset);
      argument_offset = offset + size;
      void *arg_copy = malloc(size);
      memcpy(arg_copy, arg, size);
      kernel_arguments.push_back(arg_copy);
      return cudaSuccess;
    }

    /*static*/ cudaError_t GPUProcessor::setup_argument(const void *arg, 
                                                        size_t size, size_t offset)
    {
      GPUProcessor *local = find_local_gpu();  
      assert(local != NULL);
      return local->internal->setup_argument(arg, size, offset);
    }

    cudaError_t GPUProcessor::Internal::launch(const void *func)
    {
      // Ready to do the launch 
      assert(!launch_configs.empty());
      LaunchConfig &config = launch_configs.back();
      // Find our function
      std::map<const void*,CUfunction>::const_iterator finder = 
        device_functions.find(func);
      assert(finder != device_functions.end());
      // Launch the kernel on our stream dammit!
      CHECK_CU( cuLaunchKernel(finder->second, config.grid.x, config.grid.y, config.grid.z,
                               config.block.x, config.block.y, config.block.z,
                               config.shared, get_current_task_stream(), 
                               (kernel_arguments.empty() ? 0 : &(kernel_arguments[0])), 0) );
      // Clean everything up from the launch
      launch_configs.pop_back();
      for (std::vector<void*>::iterator it = kernel_arguments.begin();
            it != kernel_arguments.end(); it++)
      {
        free(*it);
      }
      kernel_arguments.clear();
      argument_offset = 0;
      return cudaSuccess;
    }

    /*static*/ cudaError_t GPUProcessor::launch(const void *func)
    {
      GPUProcessor *local = find_local_gpu();
      assert(local != NULL);
      return local->internal->launch(func); 
    }

    /*static*/ cudaError_t GPUProcessor::gpu_malloc(void **ptr, size_t size)
    {
      CHECK_CU( cuMemAlloc((CUdeviceptr*)ptr, size) );
      return cudaSuccess;
    }

    /*static*/ cudaError_t GPUProcessor::gpu_free(void *ptr)
    {
      CHECK_CU( cuMemFree((CUdeviceptr)ptr) );
      return cudaSuccess;
    }

    cudaError_t GPUProcessor::Internal::gpu_memcpy(void *dst, const void *src,
                                                   size_t size, bool sync)
    {
      CUstream current = get_current_task_stream();
      CHECK_CU( cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, size, current) );
      if (sync)
      {
        CHECK_CU( cuStreamSynchronize(current) );
      }
      return cudaSuccess;
    }

    /*static*/ cudaError_t GPUProcessor::gpu_memcpy(void *dst, const void *src, 
                                                    size_t size, cudaMemcpyKind kind)
    {
      GPUProcessor *local = find_local_gpu();
      assert(local != NULL);
      return local->internal->gpu_memcpy(dst, src, size, true/*sync*/);
    }

    /*static*/ cudaError_t GPUProcessor::gpu_memcpy_async(void *dst, const void *src,
                                                          size_t size, cudaMemcpyKind kind,
                                                          cudaStream_t stream)
    {
      GPUProcessor *local = find_local_gpu();
      assert(local != NULL);
      return local->internal->gpu_memcpy(dst, src, size, false/*sync*/);
    }

    cudaError_t GPUProcessor::Internal::gpu_memcpy_to_symbol(void *dst, const void *src,
                                                             size_t size, size_t offset,
                                                             cudaMemcpyKind kind, bool sync)
    {
      std::map<const void*,VarInfo>::const_iterator finder = device_variables.find(dst);
      assert(finder != device_variables.end());
      CUstream current = get_current_task_stream();
      CHECK_CU( cuMemcpyAsync(finder->second.ptr+offset, (CUdeviceptr)src, size, current) );
      if (sync)
      {
        CHECK_CU( cuStreamSynchronize(current) );
      }
      return cudaSuccess;
    }

    /*static*/ cudaError_t GPUProcessor::gpu_memcpy_to_symbol(void *dst, const void *src,
                                                              size_t size, size_t offset,
                                                              cudaMemcpyKind kind, bool sync)
    {
      GPUProcessor *local = find_local_gpu();
      assert(local != NULL);
      return local->internal->gpu_memcpy_to_symbol(dst, src, size, offset, kind, sync);
    }

    cudaError_t GPUProcessor::Internal::gpu_memcpy_from_symbol(void *dst, const void *src,
                                                               size_t size, size_t offset,
                                                               cudaMemcpyKind kind, bool sync)
    {
      std::map<const void*,VarInfo>::const_iterator finder = device_variables.find(dst);
      assert(finder != device_variables.end());
      CUstream current = get_current_task_stream();
      CHECK_CU( cuMemcpyAsync((CUdeviceptr)dst, finder->second.ptr+offset, size, current) );
      if (sync)
      {
        CHECK_CU( cuStreamSynchronize(current) );
      }
      return cudaSuccess;
    }

    /*static*/ cudaError_t GPUProcessor::gpu_memcpy_from_symbol(void *dst, const void *src,
                                                                size_t size, size_t offset,
                                                                cudaMemcpyKind kind, bool sync)
    {
      GPUProcessor *local = find_local_gpu();
      assert(local != NULL);
      return local->internal->gpu_memcpy_from_symbol(dst, src, size, offset, kind, sync);
    }

    /*static*/ cudaError_t GPUProcessor::device_synchronize(void)
    {
      // Users are dumb, never let them mess us up
      GPUProcessor *local = find_local_gpu();
      assert(local != NULL);
      return local->internal->stream_synchronize();
    }

    /*static*/ cudaError_t GPUProcessor::set_shared_memory_config(cudaSharedMemConfig config)
    {
      CHECK_CU( cuCtxSetSharedMemConfig(
        (config == cudaSharedMemBankSizeDefault) ? CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE :
        (config == cudaSharedMemBankSizeFourByte) ? CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE :
                                                    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE) );
      return cudaSuccess;
    }

  }; // namespace LowLevel
}; // namespace LegionRuntime

// Our implementation of the CUDA runtime API for Legion
// so we can intercept all of these calls

// All these extern C methods are for internal implementations
// of functions of the cuda runtime API that nvcc assumes
// exists and can be used for code generation. They are all
// pretty simple to map to the driver API.

extern "C" void** __cudaRegisterFatBinary(void *fat_bin)
{
  return LegionRuntime::LowLevel::GPUProcessor::register_fat_binary(fat_bin);
}

extern "C" void __cudaUnregisterFatBinary(void **fat_bin)
{
  LegionRuntime::LowLevel::GPUProcessor::unregister_fat_binary(fat_bin);
}

extern "C" void __cudaRegisterVar(void **fat_bin,
                                  char *host_var,
                                  char *device_addr,
                                  const char *device_name,
                                  int ext, int size, int constant, int global)
{
  LegionRuntime::LowLevel::GPUProcessor::register_var(fat_bin, host_var, device_addr,
                                                      device_name, ext, size, 
                                                      constant, global);
}

extern "C" void __cudaRegisterFunction(void **fat_bin,
                                       const char *host_fun,
                                       char *device_fun,
                                       const char *device_name,
                                       int thread_limit,
                                       uint3 *tid, uint3 *bid,
                                       dim3 *bDim, dim3 *gDim,
                                       int *wSize)
{
  LegionRuntime::LowLevel::GPUProcessor::register_function(fat_bin, host_fun,
                                                           device_fun, device_name,
                                                           thread_limit, tid, bid,
                                                           bDim, gDim, wSize);
}

extern "C" char __cudaInitModule(void **fat_bin)
{
  return LegionRuntime::LowLevel::GPUProcessor::init_module(fat_bin);
}

// All the following methods are cuda runtime API calls that we 
// intercept and then either execute using the driver API or 
// modify in ways that are important to Legion.

extern cudaError_t cudaStreamCreate(cudaStream_t *stream)
{
  return LegionRuntime::LowLevel::GPUProcessor::stream_create(stream);
}

extern cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
  return LegionRuntime::LowLevel::GPUProcessor::stream_destroy(stream);
}

extern cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
  return LegionRuntime::LowLevel::GPUProcessor::stream_synchronize(stream);
}

extern cudaError_t cudaConfigureCall(dim3 grid_dim,
                                     dim3 block_dim,
                                     size_t shared_memory,
                                     cudaStream_t stream)
{
  return LegionRuntime::LowLevel::GPUProcessor::configure_call(grid_dim, block_dim,
                                                               shared_memory, stream);
}

extern cudaError_t cudaSetupArgument(const void *arg,
                                     size_t size,
                                     size_t offset)
{
  return LegionRuntime::LowLevel::GPUProcessor::setup_argument(arg, size, offset);
}

extern cudaError_t cudaLaunch(const void *func)
{
  return LegionRuntime::LowLevel::GPUProcessor::launch(func);
}

extern cudaError_t cudaMalloc(void **ptr, size_t size)
{
  return LegionRuntime::LowLevel::GPUProcessor::gpu_malloc(ptr, size);
}

extern cudaError_t cudaFree(void *ptr)
{
  return LegionRuntime::LowLevel::GPUProcessor::gpu_free(ptr);
}

extern cudaError_t cudaMemcpy(void *dst, const void *src, 
                              size_t size, cudaMemcpyKind kind)
{
  return LegionRuntime::LowLevel::GPUProcessor::gpu_memcpy(dst, src, size, kind);
}

extern cudaError_t cudaMemcpyAsync(void *dst, const void *src,
                                   size_t size, cudaMemcpyKind kind,
                                   cudaStream_t stream)
{
  return LegionRuntime::LowLevel::GPUProcessor::gpu_memcpy_async(dst, src, size, kind, stream);
}

extern cudaError_t cudaDeviceSynchronize(void)
{
  return LegionRuntime::LowLevel::GPUProcessor::device_synchronize();
}

extern cudaError_t cudaMemcpyToSymbol(void *dst, const void *src,
                                      size_t size, size_t offset,
                                      cudaMemcpyKind kind)
{
  return LegionRuntime::LowLevel::GPUProcessor::gpu_memcpy_to_symbol(dst, src, size, 
                                                      offset, kind, true/*sync*/);
}

extern cudaError_t cudaMemcpyToSymbol(void *dst, const void *src,
                                      size_t size, size_t offset,
                                      cudaMemcpyKind kind, cudaStream_t stream)
{
  return LegionRuntime::LowLevel::GPUProcessor::gpu_memcpy_to_symbol(dst, src, size,
                                                        offset, kind, false/*sync*/);
}

extern cudaError_t cudaMemcpyFromSymbol(void *dst, const void *src,
                                        size_t size, size_t offset,
                                        cudaMemcpyKind kind)
{
  return LegionRuntime::LowLevel::GPUProcessor::gpu_memcpy_from_symbol(dst, src, size,
                                                        offset, kind, true/*sync*/);
}

extern cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *src,
                                             size_t size, size_t offset,
                                             cudaMemcpyKind kind, cudaStream_t stream)
{
  return LegionRuntime::LowLevel::GPUProcessor::gpu_memcpy_from_symbol(dst, src, size,
                                                        offset, kind, false/*sync*/);
}

extern cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config)
{
  return LegionRuntime::LowLevel::GPUProcessor::set_shared_memory_config(config);
}

